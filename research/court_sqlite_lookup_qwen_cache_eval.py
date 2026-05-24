#!/usr/bin/env python
"""Evaluate cheap SQLite court lookup + cached Qwen3 rerank on val.

This is intentionally a lightweight diagnostic, not a production retriever:

1. Generate court candidates from unified_retrieval.sqlite using only cheap
   signals already in SQLite:
   - statute_links overlap with statutes parsed from the query
   - case_links overlap with cases/dockets parsed from the query
   - FTS over enriched concept-ish fields (legal_text, rag_text, keyword_text)
2. Rerank the candidate pool with existing Qwen3-Reranker cache files when
   available. This machine has no CUDA, so the script does not attempt fresh
   model inference.
3. Report court-gold recall for lookup candidates and cached-Qwen top 20.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VAL_CSV = ROOT / "data" / "val.csv"
SQLITE = ROOT / "artifacts" / "unified_retrieval.sqlite"
QWEN_CACHE_DIR = ROOT / "cache_endgame" / "rerank"
OUT_JSON = ROOT / "artifacts" / "court_sqlite_lookup_qwen_cache_val_results.json"
OUT_CSV = ROOT / "artifacts" / "court_sqlite_lookup_qwen_cache_val_per_query.csv"
OUT_TOP20 = ROOT / "artifacts" / "court_sqlite_lookup_qwen_cache_val_top20.jsonl"

RERANK_POOL = 1000
CONCEPT_FTS_LIMIT = 3000
STATUTE_LIMIT = 8000
CASE_LIMIT = 4000


BGE_FULL_RE = re.compile(
    r"\b(?:BGE|ATF)\s+\d{3}\s+[IVX]{1,4}\s+\d+[a-z]?(?:\s+E\.\s*[\d\.]+)?\b"
)
BGE_BASE_RE = re.compile(r"\b(?:BGE|ATF)\s+\d{3}\s+[IVX]{1,4}\s+\d+[a-z]?\b")
DOCKET_RE = re.compile(r"\b\d{1,2}[A-Z]{1,4}[_\.]\d{1,5}/\d{4}(?:\s+E\.\s*[\d\.]+)?\b")
DOCKET_BASE_RE = re.compile(r"\b\d{1,2}[A-Z]{1,4}[_\.]\d{1,5}/\d{4}\b")
ART_RE = re.compile(
    r"\bArt\.?\s*(?P<art>\d+[a-z]?)"
    r"(?:\s+(?:Abs|Bst|Lit|lit|Ziff|Ch|Cpv)\.?\s*[\dA-Za-z]+)*"
    r"\s+(?P<code>[A-Z][A-Za-z]{1,8}|\d{3}\.\d+)",
    re.IGNORECASE,
)

STOPWORDS = {
    "about", "above", "accused", "after", "against", "alleged", "already",
    "also", "among", "and", "another", "any", "are", "around", "because",
    "been", "before", "being", "between", "born", "but", "can", "citing",
    "claimant", "consistent", "containing", "could", "court", "dated",
    "december", "did", "does", "down", "during", "each", "essentially",
    "forum", "from", "further", "given", "ground", "grounds", "had", "has",
    "have", "here", "hers", "him", "his", "how", "including", "inter",
    "into", "january", "law", "lawful", "lawfully", "legal", "legally",
    "march", "maximum", "meaning", "month", "must", "nevertheless", "not",
    "october", "off", "order", "other", "over", "own", "primarily",
    "regard", "regarding", "several", "should", "since", "state", "such",
    "than", "that", "the", "their", "then", "there", "therefore", "these",
    "this", "those", "three", "through", "under", "until", "upon", "was",
    "were", "what", "when", "where", "whether", "which", "while", "who",
    "whose", "with", "within", "would", "years",
}

# Used only to prioritize query terms for the concept FTS channel. The final
# score still comes from literal overlap against SQLite-enriched fields.
LEGALISH = {
    "account", "administration", "allergic", "asthma", "bank", "board",
    "capacity", "child", "collusion", "contract", "contractual", "custodial",
    "custody", "damage", "debits", "delivery", "depreciation", "detention",
    "earning", "enforcement", "evidence", "exculpatory", "flight", "gift",
    "gratuitous", "gross", "handwritten", "hearing", "heard", "holographic",
    "incapacity", "innocence", "invalidity", "liability", "maintenance",
    "medical", "negligence", "overnight", "parent", "presumption",
    "proportionality", "protest", "psychiatric", "rehabilitation", "remand",
    "reoffending", "right", "risk", "signature", "sufficient", "suspicion",
    "testamentary", "title", "transfer", "trust", "visitation", "vocational",
    "welfare", "will", "witness", "witnesses", "work",
}


def sha1_query(query: str) -> str:
    return hashlib.sha1(query.encode("utf-8")).hexdigest()


def is_court_citation(citation: str) -> bool:
    return bool(citation and not citation.strip().startswith("Art."))


def parse_gold(gold_field: str) -> list[str]:
    return [c.strip() for c in str(gold_field or "").split(";") if c.strip()]


def extract_statutes(query: str) -> list[tuple[str, str]]:
    out = []
    for m in ART_RE.finditer(query):
        out.append((m.group("art"), m.group("code")))
    return sorted(set(out))


def extract_cases(query: str) -> tuple[list[str], list[str]]:
    full = set(BGE_FULL_RE.findall(query)) | set(DOCKET_RE.findall(query))
    base = set(BGE_BASE_RE.findall(query)) | set(DOCKET_BASE_RE.findall(query))
    return sorted(full), sorted(base)


def tokenize(text: str) -> list[str]:
    text = text.replace("pre-trial", "pretrial").replace("co-parent", "coparent")
    toks = []
    for raw in re.findall(r"[A-Za-z][A-Za-z-]{2,}", text.lower()):
        tok = raw.strip("-")
        if len(tok) < 4 or tok in STOPWORDS:
            continue
        toks.append(tok)
    return toks


def query_terms(query: str, max_terms: int = 12) -> list[str]:
    counts = Counter(tokenize(query))
    legal = [t for t, _ in counts.most_common() if t in LEGALISH]
    fallback = [t for t, _ in counts.most_common() if t not in legal]
    terms = legal + fallback
    # keep FTS cheap and reduce name/fact pollution
    return terms[:max_terms]


def fts_phrase(term: str) -> str:
    safe = re.sub(r'["\']', " ", term).strip()
    safe = re.sub(r"[^A-Za-z0-9_À-ÿ.-]+", " ", safe)
    return f'"{safe}"'


def concept_match(terms: list[str]) -> str:
    if not terms:
        return ""
    clause = " OR ".join(fts_phrase(t) for t in terms)
    return (
        f"keyword_text:({clause}) OR "
        f"rag_text:({clause}) OR "
        f"legal_text:({clause})"
    )


def metadata_text(row: dict[str, Any]) -> str:
    pieces = [
        row.get("legal_area") or "",
        row.get("primary_domain") or "",
        row.get("secondary_domain") or "",
        row.get("paragraph_role") or "",
    ]
    try:
        rag = json.loads(row.get("rag_json") or "{}")
    except json.JSONDecodeError:
        rag = {}
    for key in (
        "legal_area",
        "primary_domain",
        "secondary_domain",
        "topic",
        "subtopic",
        "micro_topic",
        "concepts_en",
        "english_legal_concepts",
        "search_keywords",
        "terms_original",
        "fact_pattern_tags",
        "legal_domain_path",
        "doctrinal_rule",
        "legal_test",
        "legal_rule",
        "court_holding",
    ):
        value = rag.get(key)
        if isinstance(value, list):
            pieces.extend(str(v) for v in value)
        elif value:
            pieces.append(str(value))
    return " ".join(pieces).lower()


def role_bonus(role: str | None) -> float:
    r = (role or "").lower()
    if r in {"holding", "reasoning", "legal_standard", "application"}:
        return 0.8
    if r in {"costs", "disposition", "notification", "procedural_history"}:
        return -1.5
    if r in {"facts", "admissibility"}:
        return -0.8
    return 0.0


def add_candidate(
    candidates: dict[str, dict[str, Any]],
    row: sqlite3.Row,
    channel: str,
    raw_score: float,
    extra: dict[str, Any] | None = None,
) -> None:
    doc_id = row["doc_id"]
    item = candidates.setdefault(
        doc_id,
        {
            "doc_id": doc_id,
            "citation": row["citation"],
            "authority_score": float(row["authority_score"] or 0.0),
            "specificity_score": float(row["specificity_score"] or 0.0),
            "paragraph_role": row["paragraph_role"],
            "legal_area": row["legal_area"],
            "primary_domain": row["primary_domain"],
            "secondary_domain": row["secondary_domain"],
            "rag_json": row["rag_json"],
            "channels": defaultdict(float),
            "channel_hits": defaultdict(int),
        },
    )
    item["channels"][channel] += float(raw_score)
    item["channel_hits"][channel] += 1
    if extra:
        for key, value in extra.items():
            item[key] = item.get(key, 0) + value if isinstance(value, (int, float)) else value


def lookup_candidates(con: sqlite3.Connection, query: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    statutes = extract_statutes(query)
    cases, case_bases = extract_cases(query)
    terms = query_terms(query)

    t0 = time.time()
    statute_rows = []
    if statutes:
        clauses = []
        params: list[Any] = []
        for art, code in statutes:
            clauses.append("sl.statute LIKE ?")
            params.append(f"%Art. {art}%{code}%")
        sql = (
            "SELECT d.doc_id, d.citation, d.authority_score, d.specificity_score, "
            "       d.paragraph_role, d.legal_area, d.primary_domain, d.secondary_domain, "
            "       d.rag_json, COUNT(*) AS hit_count "
            "FROM statute_links sl JOIN documents d USING (doc_id) "
            "WHERE sl.family='court' AND d.family='court' AND ("
            + " OR ".join(clauses)
            + ") GROUP BY d.doc_id "
            "ORDER BY hit_count DESC, d.authority_score DESC LIMIT ?"
        )
        params.append(STATUTE_LIMIT)
        statute_rows = con.execute(sql, params).fetchall()
        for r in statute_rows:
            add_candidate(candidates, r, "statute", 5.0 * float(r["hit_count"]), {"statute_hits": int(r["hit_count"])})

    case_rows = []
    if cases or case_bases:
        targets = cases + case_bases
        clauses = ["cl.target_citation LIKE ?" for _ in targets] + ["cl.target_base LIKE ?" for _ in targets]
        params = [f"%{t}%" for t in targets] + [f"%{t}%" for t in targets]
        sql = (
            "SELECT d.doc_id, d.citation, d.authority_score, d.specificity_score, "
            "       d.paragraph_role, d.legal_area, d.primary_domain, d.secondary_domain, "
            "       d.rag_json, COUNT(*) AS hit_count "
            "FROM case_links cl JOIN documents d USING (doc_id) "
            "WHERE cl.family='court' AND d.family='court' AND ("
            + " OR ".join(clauses)
            + ") GROUP BY d.doc_id "
            "ORDER BY hit_count DESC, d.authority_score DESC LIMIT ?"
        )
        params.append(CASE_LIMIT)
        case_rows = con.execute(sql, params).fetchall()
        for r in case_rows:
            add_candidate(candidates, r, "case", 4.0 * float(r["hit_count"]), {"case_hits": int(r["hit_count"])})

    concept_rows = []
    match = concept_match(terms)
    if match:
        sql = (
            "SELECT d.doc_id, d.citation, d.authority_score, d.specificity_score, "
            "       d.paragraph_role, d.legal_area, d.primary_domain, d.secondary_domain, "
            "       d.rag_json, bm25(documents_fts) AS fts_score "
            "FROM documents_fts JOIN documents d USING (doc_id) "
            "WHERE documents_fts MATCH ? AND d.family='court' "
            "ORDER BY fts_score LIMIT ?"
        )
        concept_rows = con.execute(sql, (match, CONCEPT_FTS_LIMIT)).fetchall()
        for rank, r in enumerate(concept_rows, start=1):
            add_candidate(candidates, r, "concept", 1.0 / math.sqrt(rank), {"concept_channel_rank": rank})

    for item in candidates.values():
        meta = metadata_text(item)
        overlap = sum(1 for t in terms if t in meta)
        item["concept_overlap"] = overlap
        item["lookup_score"] = (
            item["channels"].get("statute", 0.0)
            + item["channels"].get("case", 0.0)
            + 1.8 * overlap
            + 0.6 * float(item.get("authority_score") or 0.0)
            + 0.3 * float(item.get("specificity_score") or 0.0)
            + role_bonus(item.get("paragraph_role"))
            + item["channels"].get("concept", 0.0)
        )
        item["channels"] = dict(item["channels"])
        item["channel_hits"] = dict(item["channel_hits"])

    ranked = sorted(candidates.values(), key=lambda x: x["lookup_score"], reverse=True)
    diag = {
        "statutes": statutes,
        "cases": cases,
        "case_bases": case_bases,
        "terms": terms,
        "channel_counts": {
            "statute_rows": len(statute_rows),
            "case_rows": len(case_rows),
            "concept_rows": len(concept_rows),
            "unique_candidates": len(ranked),
        },
        "elapsed_s": round(time.time() - t0, 3),
    }
    return ranked, diag


def load_qwen_cache(query: str) -> dict[str, float]:
    path = QWEN_CACHE_DIR / f"{sha1_query(query)}.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return {str(k): float(v) for k, v in json.load(f).items()}


def eval_at(pred: list[str], gold: set[str], k: int) -> dict[str, float | int]:
    top = pred[:k]
    hits = len(set(top) & gold)
    return {
        "hits": hits,
        "recall": hits / len(gold) if gold else 0.0,
        "precision": hits / k if k else 0.0,
    }


def main() -> int:
    rows = list(csv.DictReader(VAL_CSV.open("r", encoding="utf-8")))
    con = sqlite3.connect(f"file:{SQLITE}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA cache_size=-200000")
    con.execute("PRAGMA temp_store=MEMORY")

    per_query = []
    top20_records = []

    for row in rows:
        qid = row["query_id"]
        query = row["query"]
        gold_all = parse_gold(row["gold_citations"])
        gold_court = {c for c in gold_all if is_court_citation(c)}

        candidates, diag = lookup_candidates(con, query)
        cand_citations = [c["citation"] for c in candidates]

        qwen_cache = load_qwen_cache(query)
        rerank_pool = candidates[:RERANK_POOL]
        scored = []
        missing_cache = 0
        for item in rerank_pool:
            doc_id = item["doc_id"]
            if doc_id in qwen_cache:
                enriched = dict(item)
                enriched["qwen_score"] = qwen_cache[doc_id]
                scored.append(enriched)
            else:
                missing_cache += 1
        scored.sort(key=lambda x: (x["qwen_score"], x["lookup_score"]), reverse=True)

        lookup_top20 = [c["citation"] for c in candidates[:20]]
        qwen_top20 = [c["citation"] for c in scored[:20]]

        rec100 = eval_at(cand_citations, gold_court, 100)
        rec500 = eval_at(cand_citations, gold_court, 500)
        rec1000 = eval_at(cand_citations, gold_court, 1000)
        rec_all = eval_at(cand_citations, gold_court, len(cand_citations))
        lookup20 = eval_at(lookup_top20, gold_court, 20)
        qwen20 = eval_at(qwen_top20, gold_court, 20)

        record = {
            "query_id": qid,
            "court_gold": len(gold_court),
            "candidate_count": len(candidates),
            "lookup_recall_at_100": rec100["recall"],
            "lookup_hits_at_100": rec100["hits"],
            "lookup_recall_at_500": rec500["recall"],
            "lookup_hits_at_500": rec500["hits"],
            "lookup_recall_at_1000": rec1000["recall"],
            "lookup_hits_at_1000": rec1000["hits"],
            "lookup_recall_all": rec_all["recall"],
            "lookup_hits_all": rec_all["hits"],
            "lookup_top20_hits": lookup20["hits"],
            "lookup_top20_recall": lookup20["recall"],
            "lookup_top20_precision": lookup20["precision"],
            "qwen_cache_scored_in_pool": len(scored),
            "qwen_cache_missing_in_pool": missing_cache,
            "qwen_cache_coverage": len(scored) / len(rerank_pool) if rerank_pool else 0.0,
            "qwen_top20_hits": qwen20["hits"],
            "qwen_top20_recall": qwen20["recall"],
            "qwen_top20_precision": qwen20["precision"],
            "lookup_elapsed_s": diag["elapsed_s"],
            "channel_counts": diag["channel_counts"],
            "query_statutes": diag["statutes"],
            "query_terms": diag["terms"],
        }
        per_query.append(record)

        top20_records.append(
            {
                "query_id": qid,
                "gold_court": sorted(gold_court),
                "diagnostics": diag,
                "lookup_top20": [
                    {
                        "rank": i + 1,
                        "citation": c["citation"],
                        "doc_id": c["doc_id"],
                        "lookup_score": c["lookup_score"],
                        "qwen_score": qwen_cache.get(c["doc_id"]),
                        "is_gold": c["citation"] in gold_court,
                        "channels": c.get("channels"),
                        "paragraph_role": c.get("paragraph_role"),
                    }
                    for i, c in enumerate(candidates[:20])
                ],
                "qwen_top20": [
                    {
                        "rank": i + 1,
                        "citation": c["citation"],
                        "doc_id": c["doc_id"],
                        "lookup_score": c["lookup_score"],
                        "qwen_score": c["qwen_score"],
                        "is_gold": c["citation"] in gold_court,
                        "channels": c.get("channels"),
                        "paragraph_role": c.get("paragraph_role"),
                    }
                    for i, c in enumerate(scored[:20])
                ],
            }
        )

        print(
            f"{qid}: gold={len(gold_court)} cand={len(candidates)} "
            f"lookupR@1000={rec1000['hits']}/{len(gold_court)} "
            f"qwenTop20={qwen20['hits']}/{len(gold_court)} "
            f"cache={len(scored)}/{len(rerank_pool)} "
            f"dt={diag['elapsed_s']}s"
        )

    total_gold = sum(r["court_gold"] for r in per_query)
    summary = {
        "experiment": "court_sqlite_lookup_plus_cached_qwen3_rerank",
        "note": (
            "Qwen rerank uses cache_endgame/rerank because this machine has no CUDA "
            "and Qwen/Qwen3-Reranker-8B weights are not present locally."
        ),
        "settings": {
            "rerank_pool": RERANK_POOL,
            "concept_fts_limit": CONCEPT_FTS_LIMIT,
            "statute_limit": STATUTE_LIMIT,
            "case_limit": CASE_LIMIT,
        },
        "totals": {
            "queries": len(per_query),
            "court_gold": total_gold,
            "lookup_hits_at_100": sum(r["lookup_hits_at_100"] for r in per_query),
            "lookup_hits_at_500": sum(r["lookup_hits_at_500"] for r in per_query),
            "lookup_hits_at_1000": sum(r["lookup_hits_at_1000"] for r in per_query),
            "lookup_hits_all": sum(r["lookup_hits_all"] for r in per_query),
            "lookup_top20_hits": sum(r["lookup_top20_hits"] for r in per_query),
            "qwen_top20_hits": sum(r["qwen_top20_hits"] for r in per_query),
        },
        "micro": {},
        "macro": {},
        "per_query": per_query,
    }
    for key in (
        "lookup_recall_at_100",
        "lookup_recall_at_500",
        "lookup_recall_at_1000",
        "lookup_recall_all",
        "lookup_top20_recall",
        "lookup_top20_precision",
        "qwen_top20_recall",
        "qwen_top20_precision",
        "qwen_cache_coverage",
    ):
        summary["macro"][key] = sum(float(r[key]) for r in per_query) / len(per_query)
    summary["micro"]["lookup_recall_at_100"] = summary["totals"]["lookup_hits_at_100"] / total_gold
    summary["micro"]["lookup_recall_at_500"] = summary["totals"]["lookup_hits_at_500"] / total_gold
    summary["micro"]["lookup_recall_at_1000"] = summary["totals"]["lookup_hits_at_1000"] / total_gold
    summary["micro"]["lookup_recall_all"] = summary["totals"]["lookup_hits_all"] / total_gold
    summary["micro"]["lookup_top20_recall"] = summary["totals"]["lookup_top20_hits"] / total_gold
    summary["micro"]["qwen_top20_recall"] = summary["totals"]["qwen_top20_hits"] / total_gold

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "query_id",
            "court_gold",
            "candidate_count",
            "lookup_hits_at_100",
            "lookup_recall_at_100",
            "lookup_hits_at_500",
            "lookup_recall_at_500",
            "lookup_hits_at_1000",
            "lookup_recall_at_1000",
            "lookup_hits_all",
            "lookup_recall_all",
            "lookup_top20_hits",
            "lookup_top20_recall",
            "lookup_top20_precision",
            "qwen_cache_scored_in_pool",
            "qwen_cache_missing_in_pool",
            "qwen_cache_coverage",
            "qwen_top20_hits",
            "qwen_top20_recall",
            "qwen_top20_precision",
            "lookup_elapsed_s",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_query:
            writer.writerow({k: r[k] for k in fieldnames})
    with OUT_TOP20.open("w", encoding="utf-8") as f:
        for rec in top20_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(json.dumps({"micro": summary["micro"], "macro": summary["macro"]}, indent=2))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_TOP20}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
