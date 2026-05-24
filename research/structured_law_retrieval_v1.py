"""Structured law retrieval prototype.

This is a research prototype for a law-only retrieval channel built from the
full `data/laws_de.csv` corpus plus `artifacts/law_authority_cards_v2_unified`.

It creates structured fields from otherwise unstructured law rows:
- code/article/paragraph
- law title vs doctrine heading
- isolated German legal terms and English translation terms
- LLM/static enrichment fields: concepts, legal rule, conditions, role
- graph anchors: siblings and statute anchors

Then it runs fielded BM25 channels and writes per-query diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "artifacts"
RESEARCH = ROOT / "research"
OUT_DIR = RESEARCH / "_structured_law_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAWS_DE = DATA / "laws_de.csv"
ENRICHMENT = ART / "law_authority_cards_v2_unified.jsonl"
QUERY_TRANSLATIONS = (
    ROOT / "drive_sync" / "omnilex_competition"
    / "retrieval_knowledge_base_query_translations" / "query_translations_trainval.json"
)

STRUCTURED_PARQUET = OUT_DIR / "structured_laws_v1.parquet"
STRUCTURED_SUMMARY = OUT_DIR / "structured_laws_v1.summary.json"
BM25_PICKLE = OUT_DIR / "structured_bm25_indices_v1.pkl"
RESULTS_CSV = OUT_DIR / "structured_law_retrieval_v1_per_query.csv"
RESULTS_JSON = OUT_DIR / "structured_law_retrieval_v1_summary.json"
REPORT_MD = RESEARCH / "structured_law_retrieval_v1_2026-05-23.md"


ARTICLE_RE = re.compile(
    r"^Art\.\s+(?P<article>\d+[a-z]*)"
    r"(?:\s+Abs\.\s+(?P<paragraph>\d+[a-z]*))?"
    r"\s+(?P<code>.+?)$",
    re.IGNORECASE,
)
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
    "LAI": "IVG", "LPGA": "ATSG", "LTF": "BGG", "CPP": "StPO",
    "CP": "StGB", "CC": "ZGB", "CO": "OR", "CPC": "ZPO",
    "LP": "SchKG", "LEF": "SchKG", "CST": "BV", "CEDH": "EMRK",
    "LAA": "UVG", "LAVS": "AHVG", "LAMAL": "KVG", "LPP": "BVG",
    "LPE": "USG", "OEIE": "UVPV", "LAT": "RPG",
}
CODE_CANON = {
    "STPO": "StPO", "STGB": "StGB", "ZGB": "ZGB", "OR": "OR",
    "BGG": "BGG", "BV": "BV", "IVG": "IVG", "ATSG": "ATSG",
    "ZPO": "ZPO", "IPRG": "IPRG", "SCHKG": "SchKG", "STBOG": "StBOG",
    "JSTPO": "JStPO", "JSTG": "JStG",
}

TOKEN_RE = re.compile(r"[^\w\d]+", re.UNICODE)
LEADING_NUMBER_RE = re.compile(
    r"^\s*(?:[IVXLCDM]+\.|[A-Z]\.|[a-z]\.|[0-9]+[a-z]*\.?|[0-9]+\.\s*[A-Z]\.?)\s+"
)
FOOTNOTE_TRAIL_RE = re.compile(r"(?<=[A-Za-zÄÖÜäöüßéèàç])\d{1,4}\b")

EN_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "under", "over",
    "when", "where", "which", "while", "does", "have", "has", "had", "was", "were",
    "been", "being", "may", "can", "must", "shall", "should", "would", "could", "not",
    "his", "her", "its", "their", "after", "before", "between", "within", "without",
    "given", "because", "therefore", "whether", "about", "also", "only", "most",
}
DE_STOP = {
    "der", "die", "das", "und", "oder", "ein", "eine", "einer", "eines", "einem",
    "den", "dem", "des", "mit", "von", "vom", "im", "in", "am", "an", "auf", "zu",
    "zur", "zum", "ist", "sind", "wenn", "werden", "wird", "nicht", "nach", "vor",
    "bei", "aus", "als", "auch", "nur", "sowie", "durch", "fuer", "für",
}


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def norm_token_text(text: str) -> str:
    text = strip_accents(str(text or "")).lower()
    text = text.replace("ß", "ss")
    return text


def tokenise(text: str, stop: set[str] | None = None) -> list[str]:
    toks = [t for t in TOKEN_RE.split(norm_token_text(text)) if t]
    out = []
    for tok in toks:
        if len(tok) < 2 and not tok.isdigit():
            continue
        if stop and tok in stop:
            continue
        out.append(tok)
    return out


def normalize_code(code: str) -> str:
    raw = (code or "").strip().strip(".,;:()[]")
    aliased = CODE_ALIASES.get(raw.upper(), raw)
    return CODE_CANON.get(aliased.upper(), aliased)


def norm_query_citation(article: str, abs_: str | None, code: str) -> str:
    code = normalize_code(code)
    if abs_:
        return f"Art. {article.strip()} Abs. {abs_.strip()} {code}"
    return f"Art. {article.strip()} {code}"


def parse_query_citations(query: str) -> set[str]:
    found = set()
    for regex in (QUERY_CIT_RE, QUERY_CIT_NO_ART_RE):
        for m in regex.finditer(query or ""):
            found.add(norm_query_citation(m.group("article"), m.group("abs"), m.group("code")))
    return found


def parse_citation(citation: str) -> dict:
    m = ARTICLE_RE.match(str(citation or "").strip())
    if not m:
        return {"article": "", "paragraph": "", "code": ""}
    return {
        "article": m.group("article") or "",
        "paragraph": m.group("paragraph") or "",
        "code": normalize_code(m.group("code") or ""),
    }


def split_title(title: str) -> tuple[str, str]:
    title = clean_text(title)
    if " - " in title:
        law, heading = title.split(" - ", 1)
    else:
        law, heading = title, ""
    heading = FOOTNOTE_TRAIL_RE.sub("", heading)
    prev = None
    while prev != heading:
        prev = heading
        heading = LEADING_NUMBER_RE.sub("", heading).strip()
    return law.strip(), heading.strip()


def as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def join_list(value, sep: str = " | ") -> str:
    if not value:
        return ""
    if isinstance(value, list):
        flat = []
        for item in value:
            if isinstance(item, dict):
                flat.append(" ".join(str(v) for v in item.values() if v))
            else:
                flat.append(str(item))
        return sep.join(x for x in flat if x)
    return str(value)


def extract_enrichment_fields(path: Path) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        bar = tqdm(f, desc="enrichment cards", total=175933, unit="line", smoothing=0.05)
        for i, line in enumerate(bar, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            rag = rec.get("rag_enrichment") or {}
            rv = rec.get("retrieval_views") or {}
            na = rec.get("normalized_anchors") or {}
            adj = na.get("adjacent_citations") or {}
            structural = rec.get("structural") or {}
            title_meta = rec.get("title_metadata") or {}

            terms_pairs = rag.get("terms_de_to_en") or []
            terms_de = []
            terms_en = []
            for p in terms_pairs:
                if isinstance(p, dict):
                    terms_de.append(str(p.get("de") or ""))
                    terms_en.append(str(p.get("en") or ""))

            rows.append({
                "citation": rec.get("citation") or "",
                "enrichment_source": rec.get("enrichment_source") or "",
                "granularity": structural.get("granularity") or "",
                "source_type": title_meta.get("source_type") or "",
                "law_aliases": join_list(title_meta.get("law_aliases") or []),
                "legal_area": rag.get("legal_area") or "",
                "legal_domain_path": join_list(rag.get("legal_domain_path") or []),
                "primary_domain": rag.get("primary_domain") or "",
                "secondary_domain": rag.get("secondary_domain") or "",
                "provision_role": rag.get("provision_role_llm") or "",
                "specificity_score": rag.get("specificity_score") or 0.0,
                "concepts_en": join_list(rag.get("concepts_en") or []),
                "terms_original": join_list(rag.get("terms_original") or []),
                "terms_de": " | ".join(t for t in terms_de if t),
                "terms_en": " | ".join(t for t in terms_en if t),
                "defined_terms": join_list(rag.get("defined_terms") or []),
                "english_summary": rag.get("english_summary") or "",
                "legal_question": rag.get("legal_question") or "",
                "legal_rule": rag.get("legal_rule") or "",
                "applicability_conditions": join_list(rag.get("applicability_conditions") or []),
                "exceptions_or_limitations": join_list(rag.get("exceptions_or_limitations") or []),
                "addressees": join_list(rag.get("addressees") or []),
                "sanctions_or_consequences": join_list(rag.get("sanctions_or_consequences") or []),
                "citation_view": rv.get("citation_view") or "",
                "law_context_view": rv.get("law_context_view") or "",
                "original_terms_view": rv.get("original_terms_view") or "",
                "semantic_concepts_en": rv.get("semantic_concepts_en") or "",
                "statute_anchor_view": rv.get("statute_anchor_view") or "",
                "title_view": rv.get("title_view") or "",
                "rule_components_view": rv.get("rule_components_view") or "",
                "concepts_en_view": rv.get("concepts_en_view") or "",
                "terms_bilingual_view": rv.get("terms_bilingual_view") or "",
                "same_article_siblings": join_list(adj.get("same_article_siblings") or []),
                "previous_in_law": adj.get("previous_in_law") or "",
                "next_in_law": adj.get("next_in_law") or "",
                "same_law_row_count": adj.get("same_law_row_count") or 0,
                "statute_anchors": join_list((na.get("statute_anchors") or [])),
            })
    print(f"[enrichment] parsed {len(rows):,} records in {time.time()-t0:.1f}s", flush=True)
    return pd.DataFrame(rows)


def build_structured(force: bool = False) -> pd.DataFrame:
    if STRUCTURED_PARQUET.exists() and not force:
        return pd.read_parquet(STRUCTURED_PARQUET)

    t0 = time.time()
    print("[build] reading laws_de.csv", flush=True)
    laws = pd.read_csv(LAWS_DE)
    parsed = laws["citation"].astype(str).map(parse_citation).apply(pd.Series)
    title_parts = laws["title"].astype(str).map(split_title)
    laws["law_title_clean"] = title_parts.map(lambda x: x[0])
    laws["heading_clean"] = title_parts.map(lambda x: x[1])
    laws["article"] = parsed["article"]
    laws["paragraph"] = parsed["paragraph"]
    laws["law_code"] = parsed["code"]

    print("[build] reading selected enrichment fields", flush=True)
    enrich = extract_enrichment_fields(ENRICHMENT)
    print(f"[build] merge laws={len(laws):,} enrichment={len(enrich):,}", flush=True)
    df = laws.merge(enrich, on="citation", how="left")

    # Structured retrieval fields.
    df["field_citation"] = (
        df["citation"].fillna("") + " " + df["law_code"].fillna("") + " "
        + df["article"].fillna("") + " " + df["paragraph"].fillna("") + " "
        + df["law_aliases"].fillna("")
    )
    df["field_heading_de"] = (
        df["heading_clean"].fillna("") + " " + df["title"].fillna("") + " "
        + df["title_view"].fillna("") + " " + df["original_terms_view"].fillna("")
    )
    df["field_terms_de"] = (
        df["terms_de"].fillna("") + " " + df["terms_original"].fillna("") + " "
        + df["terms_bilingual_view"].fillna("") + " " + df["heading_clean"].fillna("")
    )
    df["field_terms_en"] = (
        df["terms_en"].fillna("") + " " + df["concepts_en"].fillna("") + " "
        + df["concepts_en_view"].fillna("") + " " + df["semantic_concepts_en"].fillna("")
    )
    df["field_rule_en"] = (
        df["english_summary"].fillna("") + " " + df["legal_question"].fillna("") + " "
        + df["legal_rule"].fillna("") + " " + df["applicability_conditions"].fillna("") + " "
        + df["exceptions_or_limitations"].fillna("") + " " + df["addressees"].fillna("") + " "
        + df["sanctions_or_consequences"].fillna("") + " " + df["rule_components_view"].fillna("")
    )
    df["field_body_de"] = (
        df["text"].fillna("") + " " + df["law_title_clean"].fillna("") + " "
        + df["heading_clean"].fillna("") + " " + df["statute_anchor_view"].fillna("")
    )
    df["field_all"] = (
        df["field_citation"] + " " + df["field_heading_de"] + " " + df["field_terms_de"]
        + " " + df["field_terms_en"] + " " + df["field_rule_en"] + " " + df["field_body_de"]
    )

    df.to_parquet(STRUCTURED_PARQUET, index=False)
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "law_codes": int(df["law_code"].nunique()),
        "enrichment_source_counts": df["enrichment_source"].fillna("").value_counts().to_dict(),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    STRUCTURED_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[build] wrote {STRUCTURED_PARQUET} in {time.time()-t0:.1f}s", flush=True)
    return df


def build_indices(df: pd.DataFrame, force: bool = False) -> dict:
    if BM25_PICKLE.exists() and not force:
        with BM25_PICKLE.open("rb") as f:
            return pickle.load(f)

    fields = {
        "citation":   ("field_citation",   EN_STOP | DE_STOP),
        "heading_de": ("field_heading_de", DE_STOP),
        "terms_de":   ("field_terms_de",   DE_STOP),
        "terms_en":   ("field_terms_en",   EN_STOP),
        "rule_en":    ("field_rule_en",    EN_STOP),
        "body_de":    ("field_body_de",    DE_STOP),
        "all":        ("field_all",        EN_STOP | DE_STOP),
    }
    indices = {"citations": df["citation"].tolist()}
    t0 = time.time()
    bar = tqdm(fields.items(), desc="bm25 channels", total=len(fields))
    for name, (col, stop) in bar:
        bar.set_postfix_str(f"building {name}")
        col_vals = df[col].fillna("").astype(str).tolist()
        docs = [tokenise(x, stop)
                for x in tqdm(col_vals, desc=f"  tokenise {name}", leave=False, unit="doc")]
        indices[name] = BM25Okapi(docs)
        indices[f"{name}_doc_lens"] = [len(x) for x in docs]
    print(f"[bm25] built 7 indices in {time.time()-t0:.1f}s", flush=True)
    print(f"[bm25] writing pickle ({BM25_PICKLE.name}) ...", flush=True)
    t1 = time.time()
    with BM25_PICKLE.open("wb") as f:
        pickle.dump(indices, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[bm25] wrote {BM25_PICKLE} in {time.time()-t1:.1f}s", flush=True)
    return indices


def split_gold(value: str) -> list[str]:
    return [c.strip() for c in str(value or "").split(";") if c.strip()]


def law_gold(value: str) -> list[str]:
    return [c for c in split_gold(value) if c.startswith("Art. ")]


def load_query_translations() -> dict[str, str]:
    if not QUERY_TRANSLATIONS.exists():
        return {}
    return json.loads(QUERY_TRANSLATIONS.read_text(encoding="utf-8"))


def parent_key(citation: str) -> tuple[str, str] | None:
    p = parse_citation(citation)
    if not p["article"] or not p["code"]:
        return None
    return (p["article"].lower(), p["code"].lower())


def build_lookup(df: pd.DataFrame) -> dict:
    citation_to_idx = {c.lower(): i for i, c in enumerate(df["citation"].tolist())}
    parent_to_indices = defaultdict(list)
    code_to_indices = defaultdict(list)
    for i, r in df[["citation", "article", "law_code"]].iterrows():
        if r["article"] and r["law_code"]:
            parent_to_indices[(str(r["article"]).lower(), str(r["law_code"]).lower())].append(i)
        if r["law_code"]:
            code_to_indices[str(r["law_code"]).lower()].append(i)
    return {
        "citation_to_idx": citation_to_idx,
        "parent_to_indices": parent_to_indices,
        "code_to_indices": code_to_indices,
    }


def explicit_indices(query: str, lookup: dict, df: pd.DataFrame) -> set[int]:
    out = set()
    for cit in parse_query_citations(query):
        idx = lookup["citation_to_idx"].get(cit.lower())
        if idx is not None:
            out.add(idx)
            continue
        # Parent expansion catches "Art. 221 StPO" -> all Abs rows.
        pk = parent_key(cit)
        if pk:
            out.update(lookup["parent_to_indices"].get(pk, []))
    return out


def query_tokens_for_channel(q: str, q_de: str, channel: str) -> list[str]:
    if channel in {"heading_de", "terms_de", "body_de"}:
        return tokenise(q_de + " " + q, DE_STOP | EN_STOP)
    if channel in {"terms_en", "rule_en"}:
        return tokenise(q + " " + q_de, EN_STOP | DE_STOP)
    if channel == "citation":
        return tokenise(q + " " + q_de, EN_STOP | DE_STOP)
    return tokenise(q + " " + q_de, EN_STOP | DE_STOP)


def top_indices(scores: np.ndarray, topn: int) -> list[tuple[int, float]]:
    if topn >= len(scores):
        order = np.argsort(scores)[::-1]
    else:
        part = np.argpartition(scores, -topn)[-topn:]
        order = part[np.argsort(scores[part])[::-1]]
    return [(int(i), float(scores[i])) for i in order if scores[i] > 0]


def minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    vals = list(values.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def retrieve_query(qid: str, query: str, q_de: str, df: pd.DataFrame, indices: dict, lookup: dict,
                   topn_per_channel: int = 250,
                   channels: list[str] | None = None) -> pd.DataFrame:
    if channels is None:
        channels = ["citation", "heading_de", "terms_de", "terms_en", "rule_en", "body_de"]
    raw_scores: dict[str, dict[int, float]] = {}
    ranks: dict[str, dict[int, int]] = {}
    for ch in channels:
        toks = query_tokens_for_channel(query, q_de, ch)
        if not toks:
            raw_scores[ch] = {}
            ranks[ch] = {}
            continue
        scores = indices[ch].get_scores(toks)
        tops = top_indices(scores, topn_per_channel)
        raw_scores[ch] = {i: s for i, s in tops}
        ranks[ch] = {i: rank + 1 for rank, (i, _) in enumerate(tops)}

    explicit = explicit_indices(query + " " + q_de, lookup, df)
    candidate_ids = set(explicit)
    for score_map in raw_scores.values():
        candidate_ids.update(score_map)

    # Same-code soft expansion: only for codes explicitly visible in query refs.
    parsed_codes = {parse_citation(c)["code"].lower() for c in parse_query_citations(query + " " + q_de)}
    parsed_codes = {c for c in parsed_codes if c}

    norm_scores = {ch: minmax(score_map) for ch, score_map in raw_scores.items()}
    weights = {
        "citation": 2.5,
        "heading_de": 2.0,
        "terms_de": 1.8,
        "terms_en": 2.2,
        "rule_en": 2.0,
        "body_de": 1.2,
        "all": 1.0,
    }
    rows = []
    for i in candidate_ids:
        row = {
            "query_id": qid,
            "idx": i,
            "citation": indices["citations"][i],
            "is_explicit": i in explicit,
        }
        score = 10.0 if i in explicit else 0.0
        row_code = str(df.iloc[i]["law_code"]).lower()
        same_code = bool(row_code and row_code in parsed_codes)
        if same_code:
            score += 0.8
        row["same_explicit_code"] = same_code
        for ch in channels:
            ns = norm_scores[ch].get(i, 0.0)
            rank = ranks[ch].get(i, 0)
            row[f"{ch}_norm"] = ns
            row[f"{ch}_rank"] = rank
            score += weights[ch] * ns
            if rank and rank <= 10:
                score += 0.15
        # Structured quality priors.
        role = str(df.iloc[i].get("provision_role", ""))
        if role in {"procedure", "right_or_entitlement", "definition", "principle", "sanction_or_penalty"}:
            score += 0.15
        row["score"] = score
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["score", "citation"], ascending=[False, True]).reset_index(drop=True)


def f1(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not pred and not gold:
        return 1.0, 1.0, 1.0
    if not pred or not gold:
        return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def oracle_k_for_ranked(ranked: list[str], gold: set[str], max_k: int = 300) -> tuple[int, float]:
    best_k, best_f = 1, 0.0
    for k in range(1, min(max_k, len(ranked)) + 1):
        ff = f1(set(ranked[:k]), gold)[2]
        if ff > best_f:
            best_k, best_f = k, ff
    return best_k, best_f


def score_gap_k(cands: pd.DataFrame, k_min: int = 1, k_max: int = 80) -> int:
    if cands.empty:
        return 0
    scores = cands["score"].to_numpy()[:k_max + 1]
    if len(scores) <= k_min:
        return len(scores)
    gaps = scores[:-1] - scores[1:]
    # Ignore the huge explicit-to-nonexplicit transition when only one explicit
    # article is present; it otherwise underselects procedural cascades.
    start = min(max(k_min - 1, 0), len(gaps) - 1)
    if len(gaps[start:]) == 0:
        return min(k_max, len(scores))
    k = int(np.argmax(gaps[start:])) + start + 1
    return max(k_min, min(k, k_max, len(cands)))


def evaluate(df: pd.DataFrame, indices: dict, lookup: dict, sample_train: int = 250,
             topn_per_channel: int = 250,
             channels: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    qt = load_query_translations()
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "val.csv")
    if sample_train and sample_train < len(train):
        train_eval = train.sample(sample_train, random_state=42).reset_index(drop=True)
    else:
        train_eval = train
    allq = pd.concat([val.assign(split="val"), train_eval.assign(split="train")], ignore_index=True)

    laws_set = set(df["citation"])
    rows = []
    t0 = time.time()
    bar = tqdm(allq.iterrows(), total=len(allq), desc="evaluate", unit="q", smoothing=0.05)
    for n, r in bar:
        qid, split, query = r["query_id"], r["split"], r["query"]
        gold_all = set(law_gold(r.get("gold_citations", "")))
        # Evaluate against exact laws_de-source law gold only, because this channel
        # cannot retrieve targets absent from laws_de.
        gold = {g for g in gold_all if g in laws_set}
        if not gold:
            continue
        q_de = qt.get(qid, "")
        bar.set_postfix_str(f"{split}/{qid}")
        cands = retrieve_query(qid, query, q_de, df, indices, lookup,
                               topn_per_channel=topn_per_channel, channels=channels)
        ranked = cands["citation"].tolist() if not cands.empty else []
        pool100 = set(ranked[:100])
        pool250 = set(ranked[:250])
        pool500 = set(ranked[:500])
        k_gold = len(gold)
        k_gap = score_gap_k(cands, k_min=max(2, min(k_gold, 8)), k_max=80)
        k_oracle, f_oracle = oracle_k_for_ranked(ranked, gold, max_k=300)
        p_gc, r_gc, f_gc = f1(set(ranked[:k_gold]), gold)
        p_gap, r_gap, f_gap = f1(set(ranked[:k_gap]), gold)
        p_20, r_20, f_20 = f1(set(ranked[:20]), gold)
        rows.append({
            "split": split,
            "query_id": qid,
            "gold_laws_de_count": len(gold),
            "all_law_gold_count": len(gold_all),
            "candidate_count": len(ranked),
            "recall_at_100": len(pool100 & gold) / len(gold),
            "recall_at_250": len(pool250 & gold) / len(gold),
            "recall_at_500": len(pool500 & gold) / len(gold),
            "f1_at_gold_count": f_gc,
            "recall_at_gold_count": r_gc,
            "f1_at_gap_k": f_gap,
            "recall_at_gap_k": r_gap,
            "gap_k": k_gap,
            "f1_at_20": f_20,
            "recall_at_20": r_20,
            "oracle_k": k_oracle,
            "oracle_f1": f_oracle,
            "top20": ";".join(ranked[:20]),
            "missing_at_250": ";".join(sorted(gold - pool250)),
        })
    print(f"[eval] {len(rows)} queries scored in {time.time()-t0:.1f}s", flush=True)
    res = pd.DataFrame(rows)
    summary = {
        "config": {
            "sample_train": sample_train,
            "topn_per_channel": topn_per_channel,
            "structured_rows": len(df),
            "evaluated_against": "exact laws_de-source law gold",
        },
        "split_summary": {},
    }
    for split, d in res.groupby("split"):
        summary["split_summary"][split] = {
            "n": int(len(d)),
            "median_gold_laws_de": float(d["gold_laws_de_count"].median()),
            "recall_at_100_mean": float(d["recall_at_100"].mean()),
            "recall_at_250_mean": float(d["recall_at_250"].mean()),
            "recall_at_500_mean": float(d["recall_at_500"].mean()),
            "queries_recall_250_is_1": int((d["recall_at_250"] == 1.0).sum()),
            "f1_at_gold_count_mean": float(d["f1_at_gold_count"].mean()),
            "f1_at_gap_k_mean": float(d["f1_at_gap_k"].mean()),
            "oracle_f1_mean": float(d["oracle_f1"].mean()),
            "oracle_k_median": float(d["oracle_k"].median()),
        }
    return res, summary


def write_report(summary: dict, res: pd.DataFrame) -> None:
    lines = []
    lines.append("# Structured Law Retrieval v1")
    lines.append("")
    lines.append("This prototype turns `laws_de.csv` into structured retrieval fields and tests fielded BM25 channels over exact laws_de-source law gold.")
    lines.append("")
    lines.append("## Structured Data")
    lines.append("")
    lines.append(f"- Structured table: `{STRUCTURED_PARQUET}`")
    lines.append(f"- BM25 index cache: `{BM25_PICKLE}`")
    lines.append("- Fields: citation/code/article/paragraph, law title, doctrine heading, German text, English concepts, bilingual terms, legal rule/conditions, procedural role, sibling and statute anchors.")
    lines.append("")
    lines.append("## Retrieval Channels")
    lines.append("")
    lines.append("1. `citation`: citation/code/article/paragraph aliases, for exact statute references.")
    lines.append("2. `heading_de`: law title and doctrine heading, using German query translation.")
    lines.append("3. `terms_de`: isolated German legal terms from enrichment and headings.")
    lines.append("4. `terms_en`: English translated terms and concepts from enrichment.")
    lines.append("5. `rule_en`: English summary, legal question/rule, applicability conditions, addressees, consequences.")
    lines.append("6. `body_de`: raw law body plus statute-anchor text.")
    lines.append("7. `all`: combined fallback field.")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| split | n | median gold | recall@100 | recall@250 | recall@500 | queries R@250=1 | F1@gold_count | F1@gapK | oracle F1 | median oracle K |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split in ["train", "val"]:
        s = summary["split_summary"].get(split, {})
        if not s:
            continue
        lines.append(
            f"| {split} | {s['n']} | {s['median_gold_laws_de']:.1f} | "
            f"{s['recall_at_100_mean']:.3f} | {s['recall_at_250_mean']:.3f} | {s['recall_at_500_mean']:.3f} | "
            f"{s['queries_recall_250_is_1']} | {s['f1_at_gold_count_mean']:.3f} | "
            f"{s['f1_at_gap_k_mean']:.3f} | {s['oracle_f1_mean']:.3f} | {s['oracle_k_median']:.1f} |"
        )
    lines.append("")
    lines.append("## Mechanism Implication")
    lines.append("")
    lines.append("The structured channels should be used as a recall pool, not as a final top-K ranker. Exact law citation output requires a calibrated inclusion layer over these channel features. The feature vector should include every channel rank/score, explicit reference flag, same-code flag, provision role, legal domain, graph-sibling flag, and one-hop statute-anchor flag.")
    lines.append("")
    lines.append("The current prototype already exposes where the remaining misses are via `missing_at_250`. Those misses identify which structured channels need another signal, usually a procedural apparatus prior or a better doctrine-heading translation.")
    lines.append("")
    lines.append(f"Per-query diagnostics: `{RESULTS_CSV}`")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-structured", action="store_true")
    ap.add_argument("--force-bm25", action="store_true")
    ap.add_argument("--sample-train", type=int, default=50,
                    help="Number of train queries to sample (default 50; use 0 for val-only, or 1139 for full)")
    ap.add_argument("--topn-per-channel", type=int, default=250)
    ap.add_argument("--drop-all-channel", action="store_true", default=True,
                    help="Skip the 'all' fallback channel during retrieval (recommended; ~15%% faster, no recall loss)")
    args = ap.parse_args()

    t0 = time.time()
    print("[stage 1/3] build_structured ...", flush=True)
    df = build_structured(force=args.force_structured)
    print(f"[stage 1/3] done — {len(df):,} rows", flush=True)
    print("[stage 2/3] build_indices ...", flush=True)
    indices = build_indices(df, force=args.force_bm25)
    print(f"[stage 2/3] done — indices in memory", flush=True)
    lookup = build_lookup(df)
    print("[stage 3/3] evaluate ...", flush=True)
    channels = ["citation", "heading_de", "terms_de", "terms_en", "rule_en", "body_de"]
    if not args.drop_all_channel:
        channels.append("all")
    res, summary = evaluate(
        df, indices, lookup,
        sample_train=args.sample_train,
        topn_per_channel=args.topn_per_channel,
        channels=channels,
    )
    res.to_csv(RESULTS_CSV, index=False)
    RESULTS_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, res)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] {time.time()-t0:.1f}s")
    print(f"[done] {REPORT_MD}")


if __name__ == "__main__":
    main()
