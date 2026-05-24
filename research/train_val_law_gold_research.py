"""Train/val law-gold research inventory.

Builds a reproducible research snapshot for:
- query + gold citation inventory
- gold citations whose exact target row is present in data/laws_de.csv
- explicit query citation coverage
- existing BM25/TLF law-channel ceilings and cardinality signals

Outputs live next to this script:
- train_val_law_gold_inventory_2026-05-23.csv
- train_val_law_gold_summary_2026-05-23.json
- train_val_law_gold_research_2026-05-23.md
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INSIGHTS = ROOT / "data_insights" / "gold_analysis_coverage_and_parents"
ART = ROOT / "artifacts"
OUT_DIR = ROOT / "research"
STAMP = "2026-05-23"

TRAIN_CSV = DATA / "train.csv"
VAL_CSV = DATA / "val.csv"
LAWS_DE_CSV = DATA / "laws_de.csv"
COVERAGE_JSON = INSIGHTS / "gold_citation_coverage.json"
LAW_ORACLE_CSV = ART / "law_only_oracle_k_full.csv"
LAW_ORACLE_JSON = ART / "law_only_oracle_k_full.json"

LAW_ENRICHMENT = ART / "law_authority_cards_v2_unified.jsonl"
LAW_ENRICHMENT_SUMMARY = ART / "law_authority_cards_v2_unified.summary.json"
LAW_KB = ROOT / "drive_sync" / "omnilex_competition" / "retrieval_other_artifacts" / "laws_knowledge_base.jsonl"
LAW_BM25_CORPUS = ROOT / "drive_sync" / "omnilex_competition" / "retrieval_other_artifacts" / "corpus_v2.parquet"
LAW_BM25_INDEX = ROOT / "drive_sync" / "omnilex_competition" / "retrieval_other_artifacts" / "bm25_v2_index.pkl"
LAW_BM25_IDS = ROOT / "drive_sync" / "omnilex_competition" / "retrieval_other_artifacts" / "bm25_v2_ids.pkl"


ARTICLE_RE = re.compile(r"^Art\.\s+\d", re.IGNORECASE)
QUERY_CIT_RE = re.compile(
    r"\b(?:Art\.?|Article|Artikel)\s+"
    r"(?P<article>\d+[a-z]*)"
    r"(?:\s+(?:Abs\.?|al\.?|para\.?|paragraph)\s*(?P<abs>\d+[a-z]?))?"
    r"(?:\s+(?:lit\.?|let\.?)\s*[a-z])?"
    r"\s+(?P<code>[A-Z][A-Za-z0-9.-]{1,14})\b",
    re.IGNORECASE,
)
QUERY_CIT_NO_ART_RE = re.compile(
    r"\b(?P<article>\d+[a-z]*)"
    r"\s+(?:Abs\.?|al\.?|para\.?|paragraph)\s*(?P<abs>\d+[a-z]?)"
    r"(?:\s+(?:lit\.?|let\.?)\s*[a-z])?"
    r"\s+(?P<code>[A-Z][A-Za-z0-9.-]{1,14})\b",
    re.IGNORECASE,
)

CODE_ALIASES = {
    # French / Italian / English common aliases used in queries.
    "LAI": "IVG",
    "LPGA": "ATSG",
    "LTF": "BGG",
    "CPP": "StPO",
    "CP": "StGB",
    "CC": "ZGB",
    "CO": "OR",
    "CPC": "ZPO",
    "LP": "SchKG",
    "LEF": "SchKG",
    "CST": "BV",
    "CEDH": "EMRK",
    "LAA": "UVG",
    "LAVS": "AHVG",
    "LAMAL": "KVG",
    "LPP": "BVG",
    "LPE": "USG",
    "OEIE": "UVPV",
    "LAT": "RPG",
}

CODE_CANON = {
    "STPO": "StPO",
    "STGB": "StGB",
    "ZGB": "ZGB",
    "OR": "OR",
    "BGG": "BGG",
    "BV": "BV",
    "IVG": "IVG",
    "ATSG": "ATSG",
    "ZPO": "ZPO",
    "IPRG": "IPRG",
    "SCHKG": "SchKG",
    "STBOG": "StBOG",
}


def split_gold(value: str) -> list[str]:
    return [c.strip() for c in str(value or "").split(";") if c.strip()]


def is_law_citation(citation: str) -> bool:
    return bool(ARTICLE_RE.match(citation or ""))


def citation_code(citation: str) -> str:
    parts = citation.split()
    return parts[-1] if parts else ""


def normalize_code(code: str) -> str:
    raw = (code or "").strip().strip(".,;:()[]")
    key = raw.upper()
    aliased = CODE_ALIASES.get(key, raw)
    return CODE_CANON.get(aliased.upper(), aliased)


def norm_query_citation(article: str, abs_: str | None, code: str) -> str:
    code = normalize_code(code)
    article = article.strip()
    if abs_:
        return f"Art. {article} Abs. {abs_.strip()} {code}"
    return f"Art. {article} {code}"


def parse_query_citations(query: str) -> set[str]:
    found = set()
    for m in QUERY_CIT_RE.finditer(query or ""):
        found.add(norm_query_citation(m.group("article"), m.group("abs"), m.group("code")))
    for m in QUERY_CIT_NO_ART_RE.finditer(query or ""):
        found.add(norm_query_citation(m.group("article"), m.group("abs"), m.group("code")))
    return found


def parent_key(citation: str) -> tuple[str, str] | None:
    parts = citation.split()
    if len(parts) < 3 or parts[0] != "Art.":
        return None
    return (parts[1], parts[-1].lower())


def exact_or_parent_match(parsed: set[str], gold_laws: list[str]) -> set[str]:
    gold_set = set(gold_laws)
    matched = set(parsed) & gold_set
    by_parent = defaultdict(set)
    for g in gold_laws:
        pk = parent_key(g)
        if pk:
            by_parent[pk].add(g)
    for p in parsed:
        if p in matched:
            continue
        pk = parent_key(p)
        if not pk:
            continue
        if " Abs. " not in p:
            matched.update(by_parent.get(pk, set()))
    return matched


def detect_lang_is_english(text: str) -> bool:
    s = f" {(text or '').lower()} "
    en = sum(s.count(w) for w in (" the ", " is ", " a ", " an ", " does ", " may ", " under ", " when "))
    de = sum(s.count(w) for w in (" der ", " die ", " das ", " ist ", " ein ", " eine ", " fuer ", " für ", " mit "))
    return en > de


LITIGANT_RE = re.compile(
    r"\b(the\s+(?:prosecutor|detainee|defendant|plaintiff|accused|claimant|appellant|respondent|insured|insurer)"
    r"|der\s+(?:beschuldigte|beklagte|klaeger|kläger|angeschuldigte|versicherte|beschwerdefuehrer|beschwerdeführer)"
    r"|der\s+staatsanwalt|le\s+procureur)\b",
    re.IGNORECASE,
)
ADVERSARIAL_RE = re.compile(
    r"(sought\s+(?:an\s+)?extension|opposed|argued|requested|denied|submitted|while\s+the|whereas|"
    r"i\.e\.\s+does|beantragte|wehrte\s+sich|widersetzte\s+sich|geltend\s+gemacht|verlangte)",
    re.IGNORECASE,
)
POSTURE_RE = re.compile(
    r"(by\s+order\s+dated|by\s+(?:decision|order)\s+of|remanded|appeal|extension|"
    r"mit\s+verfuegung\s+vom|mit\s+verfügung\s+vom|mit\s+(?:entscheid|beschluss)\s+vom|"
    r"erhob\s+beschwerde|par\s+ordonnance\s+du|par\s+décision\s+du|a\s+formé\s+recours)",
    re.IGNORECASE,
)
DATE_DDMMYYYY_RE = re.compile(r"\b\d{1,2}\.\s?\d{1,2}\.\s?\d{4}\b")
DATE_WORD_RE = re.compile(
    r"\b\d{1,2}\s+"
    r"(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"januar|februar|maerz|märz|mai|juni|juli|oktober|dezember)"
    r"\s+\d{4}\b",
    re.IGNORECASE,
)
CLOSING_RE = re.compile(r"(?:--|—)\s*i\.e\.\s+(?:does|may|can|is|do|are|should)", re.IGNORECASE)


def cascade_features(query: str) -> dict:
    q = query or ""
    parsed = parse_query_citations(q)
    n_codes = len({citation_code(c) for c in parsed})
    n_litigants = len(LITIGANT_RE.findall(q))
    n_adversarial = len(ADVERSARIAL_RE.findall(q))
    n_posture = len(POSTURE_RE.findall(q))
    n_dates = len(DATE_DDMMYYYY_RE.findall(q)) + len(DATE_WORD_RE.findall(q))
    has_iie = bool(CLOSING_RE.search(q))
    is_english = detect_lang_is_english(q)
    cascade = (
        2.0 * n_litigants
        + 1.5 * n_adversarial
        + 1.5 * n_posture
        + 1.0 * n_dates
        + 5.0 * (1 if has_iie else 0)
        + 3.0 * (1 if is_english else 0)
    )
    if cascade >= 10:
        k_law = max(15, min(40, int(cascade * 1.5)))
        tier = "cascade_heavy"
    elif cascade >= 3:
        k_law = max(8, min(25, int(cascade * 2.0)))
        tier = "cascade_moderate"
    else:
        k_law = max(2, int(cascade * 1.5))
        tier = "substantive_only"
    return {
        "n_query_citations": len(parsed),
        "n_codes_in_query": n_codes,
        "n_litigants": n_litigants,
        "n_adversarial": n_adversarial,
        "n_posture": n_posture,
        "n_dates": n_dates,
        "has_iie_closing": has_iie,
        "is_english": is_english,
        "cascade_score": round(cascade, 3),
        "cascade_tier": tier,
        "k_law_pred": k_law,
    }


def read_coverage() -> dict[tuple[str, str], dict]:
    data = json.loads(COVERAGE_JSON.read_text(encoding="utf-8"))
    records = data["records"] if isinstance(data, dict) and "records" in data else data
    out = {}
    for rec in records:
        qid = rec.get("query_id")
        cit = rec.get("citation")
        if qid and cit:
            out[(qid, cit)] = rec
    return out


def laws_de_source_flag(rec: dict | None) -> bool:
    if not rec:
        return False
    origins = rec.get("origins") or {}
    laws = origins.get("laws_de") or {}
    if "source_column" in laws:
        return bool(laws.get("source_column"))
    return bool(rec.get("present_in_laws_de_source"))


def laws_de_any_flag(rec: dict | None) -> bool:
    if not rec:
        return False
    origins = rec.get("origins") or {}
    laws = origins.get("laws_de") or {}
    return bool(laws.get("present") or rec.get("present_in_laws_de_source") or rec.get("present_in_laws_de_text"))


def f1_at_k(row: pd.Series, k: int) -> float:
    k = int(max(1, min(k, 60)))
    col = f"f1@{k}"
    if col not in row:
        return math.nan
    return float(row[col])


def build_inventory() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    coverage = read_coverage()
    laws_de = pd.read_csv(LAWS_DE_CSV, usecols=["citation"])
    laws_de_set = set(laws_de["citation"].astype(str))

    rows = []
    citation_rows = []
    for split, path in [("train", TRAIN_CSV), ("val", VAL_CSV)]:
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            qid = r["query_id"]
            query = r["query"]
            gold = split_gold(r.get("gold_citations", ""))
            law_gold = [c for c in gold if is_law_citation(c)]
            laws_de_source = []
            laws_de_any = []
            missing_from_laws_de_source = []
            for c in law_gold:
                rec = coverage.get((qid, c))
                in_source = laws_de_source_flag(rec) or c in laws_de_set
                in_any = laws_de_any_flag(rec) or c in laws_de_set
                if in_source:
                    laws_de_source.append(c)
                else:
                    missing_from_laws_de_source.append(c)
                if in_any:
                    laws_de_any.append(c)

            for c in gold:
                rec = coverage.get((qid, c))
                is_law = is_law_citation(c)
                citation_rows.append({
                    "split": split,
                    "query_id": qid,
                    "citation": c,
                    "is_law_citation": is_law,
                    "is_laws_de_source": (laws_de_source_flag(rec) or c in laws_de_set) if is_law else False,
                    "is_laws_de_any": (laws_de_any_flag(rec) or c in laws_de_set) if is_law else False,
                    "family": (rec or {}).get("family", "law" if is_law else "non_law"),
                    "pattern": (rec or {}).get("pattern", ""),
                    "code": citation_code(c) if is_law else "",
                    "query": query,
                })

            parsed_query_cits = parse_query_citations(query)
            explicit_matched = exact_or_parent_match(parsed_query_cits, law_gold)
            feats = cascade_features(query)
            code_mix = Counter(citation_code(c) for c in laws_de_source)
            non_law_gold = [c for c in gold if not is_law_citation(c)]

            row = {
                "split": split,
                "query_id": qid,
                "query": query,
                "total_gold_count": len(gold),
                "law_gold_count": len(law_gold),
                "laws_de_source_gold_count": len(laws_de_source),
                "laws_de_any_gold_count": len(laws_de_any),
                "non_law_gold_count": len(non_law_gold),
                "laws_de_source_coverage": (len(laws_de_source) / len(law_gold)) if law_gold else 0.0,
                "explicit_query_law_gold_hits": len(explicit_matched),
                "explicit_query_law_gold_coverage": (len(explicit_matched) / len(law_gold)) if law_gold else 0.0,
                "query_citations_parsed": ";".join(sorted(parsed_query_cits)),
                "explicit_query_matched_law_gold": ";".join(sorted(explicit_matched)),
                "laws_de_source_gold_citations": ";".join(laws_de_source),
                "law_gold_citations": ";".join(law_gold),
                "non_law_gold_citations": ";".join(non_law_gold),
                "missing_from_laws_de_source": ";".join(missing_from_laws_de_source),
                "code_mix": ";".join(f"{k}x{v}" for k, v in sorted(code_mix.items())),
                **feats,
            }
            rows.append(row)

    inv = pd.DataFrame(rows)

    if LAW_ORACLE_CSV.exists():
        oracle = pd.read_csv(LAW_ORACLE_CSV)
        keep = ["qid", "oracle_k", "oracle_f1", "gold_count", "query_len"]
        keep += [c for c in oracle.columns if c.startswith("f1@")]
        oracle = oracle[keep].rename(columns={"qid": "query_id", "gold_count": "oracle_gold_count"})
        inv = inv.merge(oracle, on="query_id", how="left")
        inv["f1_at_predicted_k_law"] = inv.apply(lambda x: f1_at_k(x, x["k_law_pred"]), axis=1)
        inv["f1_at_exact_law_gold_count"] = inv.apply(
            lambda x: f1_at_k(x, x["laws_de_source_gold_count"] or x["law_gold_count"]), axis=1
        )

    cite_df = pd.DataFrame(citation_rows)
    summary = summarize(inv, cite_df, laws_de)
    return inv, cite_df, summary


def summarize(inv: pd.DataFrame, cite_df: pd.DataFrame, laws_de: pd.DataFrame) -> dict:
    enrichment_summary = {}
    if LAW_ENRICHMENT_SUMMARY.exists():
        enrichment_summary = json.loads(LAW_ENRICHMENT_SUMMARY.read_text(encoding="utf-8"))

    summary: dict[str, object] = {
        "files": {
            "train_csv": str(TRAIN_CSV),
            "val_csv": str(VAL_CSV),
            "laws_de_csv": str(LAWS_DE_CSV),
            "law_enrichment_file": str(LAW_ENRICHMENT),
            "law_enrichment_summary": str(LAW_ENRICHMENT_SUMMARY),
            "laws_knowledge_base": str(LAW_KB),
            "bm25_corpus": str(LAW_BM25_CORPUS),
            "bm25_index": str(LAW_BM25_INDEX),
            "bm25_ids": str(LAW_BM25_IDS),
        },
        "laws_de": {
            "rows": int(len(laws_de)),
            "unique_citations": int(laws_de["citation"].nunique()),
        },
        "law_enrichment_summary": enrichment_summary,
        "splits": {},
        "current_bm25_tlf": {},
    }

    for split, d in inv.groupby("split"):
        law_total = int(d["law_gold_count"].sum())
        laws_de_total = int(d["laws_de_source_gold_count"].sum())
        total_gold = int(d["total_gold_count"].sum())
        explicit_hits = int(d["explicit_query_law_gold_hits"].sum())
        missing = int((d["law_gold_count"] - d["laws_de_source_gold_count"]).sum())
        split_summary = {
            "n_queries": int(len(d)),
            "total_gold_citations": total_gold,
            "law_gold_citations": law_total,
            "laws_de_source_gold_citations": laws_de_total,
            "non_law_gold_citations": int(d["non_law_gold_count"].sum()),
            "laws_de_source_coverage_of_law_gold": (laws_de_total / law_total) if law_total else 0.0,
            "law_gold_per_query_min": int(d["law_gold_count"].min()),
            "law_gold_per_query_median": float(d["law_gold_count"].median()),
            "law_gold_per_query_max": int(d["law_gold_count"].max()),
            "queries_with_all_law_gold_in_laws_de": int((d["law_gold_count"] == d["laws_de_source_gold_count"]).sum()),
            "missing_law_gold_from_laws_de_source": missing,
            "explicit_query_law_gold_hits": explicit_hits,
            "explicit_query_law_gold_coverage": (explicit_hits / law_total) if law_total else 0.0,
            "mean_explicit_query_law_gold_coverage_per_query": float(d["explicit_query_law_gold_coverage"].mean()),
            "top_law_codes": Counter(
                code for cites in d["laws_de_source_gold_citations"].fillna("")
                for code in [citation_code(c) for c in cites.split(";") if c]
            ).most_common(15),
            "top_missing_law_codes": Counter(
                code for cites in d["missing_from_laws_de_source"].fillna("")
                for code in [citation_code(c) for c in cites.split(";") if c]
            ).most_common(15),
        }
        for col in ["oracle_f1", "f1_at_exact_law_gold_count", "f1_at_predicted_k_law"]:
            if col in d:
                split_summary[col + "_mean"] = float(d[col].dropna().mean())
        if "oracle_k" in d:
            split_summary["oracle_k_median"] = float(d["oracle_k"].dropna().median())
        if "k_law_pred" in d:
            split_summary["k_law_pred_mae_vs_laws_de_count"] = float(
                (d["k_law_pred"] - d["laws_de_source_gold_count"]).abs().mean()
            )
            split_summary["k_law_pred_corr_vs_laws_de_count"] = corr(d["k_law_pred"], d["laws_de_source_gold_count"])
            split_summary["cascade_corr_vs_laws_de_count"] = corr(d["cascade_score"], d["laws_de_source_gold_count"])
        summary["splits"][split] = split_summary

    if LAW_ORACLE_JSON.exists():
        summary["current_bm25_tlf"] = json.loads(LAW_ORACLE_JSON.read_text(encoding="utf-8"))
    return summary


def corr(a: pd.Series, b: pd.Series) -> float | None:
    if len(a.dropna()) < 2:
        return None
    value = a.corr(b)
    if pd.isna(value):
        return None
    return float(value)


def md_escape_pipes(value: object) -> str:
    return str(value).replace("|", "\\|")


def fmt_float(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        if math.isnan(float(value)):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_report(inv: pd.DataFrame, cite_df: pd.DataFrame, summary: dict) -> Path:
    out_md = OUT_DIR / f"train_val_law_gold_research_{STAMP}.md"
    out_csv = OUT_DIR / f"train_val_law_gold_inventory_{STAMP}.csv"
    out_cite_csv = OUT_DIR / f"train_val_law_gold_citation_rows_{STAMP}.csv"
    out_json = OUT_DIR / f"train_val_law_gold_summary_{STAMP}.json"

    inv.to_csv(out_csv, index=False)
    cite_df.to_csv(out_cite_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Train + Val Law-Gold Research")
    lines.append("")
    lines.append(f"_Generated {STAMP}. Focus: identify query/gold inventory, laws_de-source gold, law enrichment/KB files, and retrieval/cardinality implications._")
    lines.append("")
    lines.append("## Files Identified")
    lines.append("")
    file_rows = [
        ("train queries + gold", summary["files"]["train_csv"]),
        ("val queries + gold", summary["files"]["val_csv"]),
        ("raw laws_de exact target corpus", summary["files"]["laws_de_csv"]),
        ("law enrichment file", summary["files"]["law_enrichment_file"]),
        ("law enrichment summary", summary["files"]["law_enrichment_summary"]),
        ("laws knowledge base file", summary["files"]["laws_knowledge_base"]),
        ("BM25 law corpus", summary["files"]["bm25_corpus"]),
        ("BM25 law index + ids", summary["files"]["bm25_index"] + " + " + summary["files"]["bm25_ids"]),
    ]
    lines.append("| Role | Path |")
    lines.append("|---|---|")
    for role, path in file_rows:
        lines.append(f"| {role} | `{path}` |")
    lines.append("")

    lines.append("## Split Summary")
    lines.append("")
    lines.append(f"`data/laws_de.csv` has **{summary['laws_de']['rows']:,} rows** and **{summary['laws_de']['unique_citations']:,} unique citation strings**.")
    enrich_stats = (summary.get("law_enrichment_summary") or {}).get("stats", {})
    enrich_sources = summary.get("law_enrichment_summary", {}).get("enrichment_source_counts", {})
    if enrich_stats:
        lines.append(
            f"`law_authority_cards_v2_unified.jsonl` scanned/wrote **{enrich_stats.get('written', 'n/a'):,}** records; "
            f"source split: {enrich_sources}."
        )
    lines.append("")
    lines.append("| split | queries | all gold | law gold | laws_de-source law gold | laws_de coverage | law gold/query min/median/max | explicit query hits | explicit coverage |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for split in ["train", "val"]:
        s = summary["splits"][split]
        range_str = f"{s['law_gold_per_query_min']}/{s['law_gold_per_query_median']:.1f}/{s['law_gold_per_query_max']}"
        lines.append(
            f"| {split} | {s['n_queries']} | {s['total_gold_citations']} | {s['law_gold_citations']} | "
            f"{s['laws_de_source_gold_citations']} | {fmt_float(s['laws_de_source_coverage_of_law_gold'])} | "
            f"{range_str} | {s['explicit_query_law_gold_hits']} | {fmt_float(s['explicit_query_law_gold_coverage'])} |"
        )
    lines.append("")

    lines.append("Interpretation:")
    lines.append("")
    lines.append("- `laws_de-source law gold` means the exact prediction target is present as a row in `data/laws_de.csv`, not merely mentioned inside another law/court text.")
    lines.append("- Val is clean for law retrieval: every val law gold citation is a `laws_de.csv` row.")
    lines.append("- Train has a non-trivial missing/source-mismatch tail, so a strict laws_de-only system cannot reach law F1=1.0 on all train law gold unless those missing targets are repaired or mapped.")
    lines.append("")

    lines.append("Top missing exact laws_de-source codes:")
    lines.append("")
    lines.append("| split | missing code counts |")
    lines.append("|---|---|")
    for split in ["train", "val"]:
        s = summary["splits"][split]
        missing = "; ".join(f"{code}x{count}" for code, count in s["top_missing_law_codes"][:10]) or "(none)"
        lines.append(f"| {split} | {missing} |")
    lines.append("")

    lines.append("## Per-Query Law Gold Counts")
    lines.append("")
    lines.append("| query_id | split | total gold | law gold | laws_de-source | top code mix | explicit query match | cascade tier | K_law_pred |")
    lines.append("|---|---|---:|---:|---:|---|---:|---|---:|")
    show = pd.concat([
        inv[inv["split"] == "val"],
        inv[inv["split"] == "train"].head(20),
    ], ignore_index=True)
    for _, r in show.iterrows():
        lines.append(
            f"| {r['query_id']} | {r['split']} | {int(r['total_gold_count'])} | {int(r['law_gold_count'])} | "
            f"{int(r['laws_de_source_gold_count'])} | {md_escape_pipes(r['code_mix'])} | "
            f"{int(r['explicit_query_law_gold_hits'])} | {r['cascade_tier']} | {int(r['k_law_pred'])} |"
        )
    lines.append("")
    lines.append("_The CSV contains every train row; the table above shows all val rows plus the first 20 train rows for scanability._")
    lines.append("")

    lines.append("## Existing Law-Only Retrieval Baseline")
    lines.append("")
    lines.append("Using the already-built BM25+TLF law channel from `artifacts/law_only_oracle_k_full.*`:")
    lines.append("")
    lines.append("| split | fixed F1@5 | fixed F1@10 | fixed F1@12 | F1 at exact law-gold count | oracle-K F1 | median oracle K | K_pred F1 | K_pred MAE vs law count |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    bm25_summary = summary.get("current_bm25_tlf", {}).get("summary", {})
    for split in ["train", "val"]:
        s = summary["splits"][split]
        b = bm25_summary.get(split, {})
        lines.append(
            f"| {split} | {fmt_float(b.get('f1@K5'))} | {fmt_float(b.get('f1@K10'))} | {fmt_float(b.get('f1@K12'))} | "
            f"{fmt_float(s.get('f1_at_exact_law_gold_count_mean'))} | {fmt_float(s.get('oracle_f1_mean'))} | "
            f"{fmt_float(s.get('oracle_k_median'), 1)} | {fmt_float(s.get('f1_at_predicted_k_law_mean'))} | "
            f"{fmt_float(s.get('k_law_pred_mae_vs_laws_de_count'), 2)} |"
        )
    lines.append("")
    lines.append("Key finding: knowing the exact number of laws to output is not enough with the current BM25 ranking. Even `K = exact law-gold count` averages well below 1.0, and even oracle-K over the top-60 BM25 list is below 1.0. The missing piece is candidate-pool recall plus a stronger inclusion scorer, not only cardinality.")
    lines.append("")

    lines.append("## Retrieval Mechanism Toward Law-Gold F1 = 1.0")
    lines.append("")
    lines.append("The mechanism should be channel-based, with cardinality decided after scoring rather than by a fixed K:")
    lines.append("")
    lines.append("1. **Exact statute parser**: parse explicit `Art./Article/Artikel` mentions in the query; normalize aliases such as `LAI -> IVG`, `CPP -> StPO`, `LTF -> BGG`; lookup in `laws_de.csv`; expand parent articles to existing paragraph rows when needed.")
    lines.append("2. **Doctrine-heading channel**: mine German doctrine headings from `laws_de.csv.title` and `law_authority_cards_v2_unified.jsonl.retrieval_views.title_view`; map English query concepts to those headings; retrieve exact article rows by heading/title BM25.")
    lines.append("3. **Procedural apparatus channel**: classify legal area/posture from the query and add the Swiss procedural bundle implied by that area, e.g. BGG appeal deadline, StPO Beschwerde/cost/jurisdiction articles for criminal-procedure cascades, ATSG/IVG procedural provisions for social insurance.")
    lines.append("4. **One-hop law graph channel**: from the candidates above, parse outbound statute references in `laws_de.csv.text` and `normalized_anchors.statute_anchors`; add foundational definitions and sibling paragraphs only one hop deep.")
    lines.append("5. **Precision/cardinality scorer**: train/calibrate an inclusion model over channel flags, BM25 ranks, enrichment fields (`concepts_en`, `terms_de_to_en`, `legal_rule`, `applicability_conditions`, `provision_role_llm`, `law_context_view`, `statute_anchor_view`), and query features. Select every candidate with calibrated `P(gold | query,candidate)` above a split-safe threshold; use predicted count only as a soft prior or tie-breaker.")
    lines.append("")
    lines.append("For exact macro F1, two conditions must hold: candidate recall must be 1.0 for law gold, and the inclusion scorer must have no false positives. The existing files provide the raw material for this, but the current BM25+TLF channel alone does not meet those conditions.")
    lines.append("")

    lines.append("## Artifacts Written")
    lines.append("")
    lines.append(f"- `{out_csv}`: per-query inventory, query text, gold counts, laws_de-source gold list, explicit matches, cascade/cardinality features, BM25 F1 columns.")
    lines.append(f"- `{out_cite_csv}`: one row per gold citation with `is_law_citation`, `is_laws_de_source`, code, family, and query.")
    lines.append(f"- `{out_json}`: aggregate summary and file map.")
    lines.append(f"- `{out_md}`: this report.")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md


def main() -> None:
    inv, cite_df, summary = build_inventory()
    out = write_report(inv, cite_df, summary)
    print(f"[done] wrote {out}")
    print(json.dumps(summary["splits"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
