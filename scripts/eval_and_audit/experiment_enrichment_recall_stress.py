#!/usr/bin/env python
"""Stress-test whether enriched citation fields can recover golds from noise.

Experiment:
  1. Pick one validation query.
  2. Force its law and court gold citations into a candidate pool.
  3. Add random noise:
       - 100k court rows from court_authority_cards_v4_target_cards.jsonl
       - optional law rows from laws_de.csv
  4. Rank the mixed pool with BM25 over the same fields that the enriched RAG
     corpus will index.
  5. Report recall@1000 overall, law-only, and court-only.

This script does not use gold labels for scoring. Gold labels are used only
after ranking to measure recall.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class Config:
    query_id: str = "val_001"
    top_k: int = 1000
    court_noise_n: int = 100_000
    law_noise_n: int = 20_000
    seed: int = 271828

    val_csv: Path = ROOT / "data" / "val.csv"
    laws_csv: Path = ROOT / "data" / "laws_de.csv"
    court_v4_jsonl: Path = ROOT / "artifacts" / "court_authority_cards_v4.jsonl"
    court_noise_jsonl: Path = ROOT / "artifacts" / "court_authority_cards_v4_target_cards.jsonl"

    output_candidates_jsonl: Path = ROOT / "artifacts" / "enrichment_recall_stress_val_001_candidates.jsonl"
    output_top_csv: Path = ROOT / "artifacts" / "enrichment_recall_stress_val_001_top1000.csv"
    output_result_json: Path = ROOT / "artifacts" / "enrichment_recall_stress_val_001_results.json"

    max_source_chars: int = 1200
    write_candidate_jsonl: bool = True


CONFIG = Config()


TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÖØ-öø-ÿ0-9_]+")
ART_RE = re.compile(r"\bArt\.\s*\d+[A-Za-z0-9.]*", re.IGNORECASE)
BGE_RE = re.compile(r"\bBGE\s+\d{3}\s+[IVX]{1,4}\s+\d+[a-z]?\b", re.IGNORECASE)
DOCKET_RE = re.compile(r"\b\d{1,2}[A-Z]{1,4}[_\.]\d{1,5}/\d{4}\b", re.IGNORECASE)

ENGLISH_TO_SWISS_TERMS = {
    "pretrial": ["untersuchungshaft", "sicherheitshaft", "detention provisoire"],
    "detention": ["untersuchungshaft", "sicherheitshaft", "haft", "détention", "detention"],
    "collusion": ["kollusionsgefahr", "verdunkelungsgefahr", "collusion"],
    "risk": ["gefahr", "risque", "rischio"],
    "proportionality": ["verhältnismässigkeit", "verhaltnismassigkeit", "proportionnalité", "proporzionalità"],
    "flight": ["fluchtgefahr", "risque de fuite"],
    "witness": ["zeuge", "zeugen", "témoin", "testimone"],
    "evidence": ["beweismittel", "preuves", "mezzi di prova"],
    "appeal": ["beschwerde", "recours", "ricorso"],
    "invalidity": ["invalidität", "invalidenversicherung", "invalidité", "invalidità"],
    "capacity": ["arbeitsfähigkeit", "erwerbsfähigkeit", "capacité de travail"],
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\x00", " ").split())


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_text(v) for v in value if clean_text(v)]
    text = clean_text(value)
    return [text] if text else []


def unique_keep_order(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_text(value)
        key = text.lower()
        if text and key not in seen:
            out.append(text)
            seen.add(key)
    return out


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower().replace("_", ".") for m in TOKEN_RE.finditer(text)]


def expanded_query_text(query: str) -> str:
    tokens = set(tokenize(query))
    expansion: list[str] = [query]
    for english, terms in ENGLISH_TO_SWISS_TERMS.items():
        if english in tokens:
            expansion.extend(terms)
    return " ".join(expansion)


def citation_family(citation: str) -> str:
    if citation.startswith("Art."):
        return "law"
    if BGE_RE.search(citation) or DOCKET_RE.search(citation):
        return "court"
    return "other"


def read_val_query(path: Path, query_id: str) -> tuple[str, list[str]]:
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["query_id"] == query_id:
                gold = [x.strip() for x in row["gold_citations"].split(";") if x.strip()]
                return row["query"], gold
    raise KeyError(f"query_id not found: {query_id}")


def stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def flatten_term_map(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    out: list[str] = []
    for concept, terms in value.items():
        out.append(clean_text(concept))
        out.extend(as_list(terms))
    return unique_keep_order(out)


def rag_text(rag: dict[str, Any]) -> str:
    if not isinstance(rag, dict):
        return ""
    fields = [
        "english_summary",
        "legal_topic",
        "legal_question",
        "legal_rule",
        "court_holding",
        "factual_context",
    ]
    arrays = [
        "english_legal_concepts",
        "search_keywords",
        "natural_language_queries",
    ]
    parts = [clean_text(rag.get(field)) for field in fields]
    for field in arrays:
        parts.extend(as_list(rag.get(field)))
    return " | ".join(p for p in parts if p)


def make_law_doc(row: dict[str, str], *, is_gold: bool) -> dict[str, Any]:
    citation = clean_text(row.get("citation"))
    title = clean_text(row.get("title"))
    text = clean_text(row.get("text"))[: CONFIG.max_source_chars]
    law_code = citation.rsplit(" ", 1)[-1] if " " in citation else ""
    retrieval_text = " | ".join(
        p
        for p in [
            citation,
            title,
            law_code,
            text,
        ]
        if p
    )
    return {
        "doc_id": "law:" + citation,
        "source_family": "law",
        "citation": citation,
        "is_gold": is_gold,
        "retrieval_text": retrieval_text,
        "metadata": {
            "law_code": law_code,
            "title": title,
        },
    }


def make_court_doc(card: dict[str, Any], *, is_gold: bool) -> dict[str, Any]:
    citation = clean_text(card.get("citation"))
    rag = card.get("rag_enrichment") if isinstance(card.get("rag_enrichment"), dict) else {}
    structural = card.get("structural") if isinstance(card.get("structural"), dict) else {}
    retrieval_parts = [
        citation,
        clean_text(card.get("court_base")),
        clean_text(card.get("legal_area")),
        clean_text(card.get("retrieval_text_en")),
        clean_text(card.get("summary_en_proxy")),
        rag_text(rag),
        " ".join(as_list(card.get("authority_role"))),
        " ".join(as_list(card.get("issue_labels_en"))),
        " ".join(as_list(card.get("law_codes"))),
        " ".join(as_list(card.get("statutes_cited"))),
        " ".join(as_list(card.get("court_cases_cited"))),
        " ".join(flatten_term_map(card.get("matched_terms_multilingual"))),
        clean_text(card.get("text_excerpt_original"))[: CONFIG.max_source_chars],
    ]
    return {
        "doc_id": "court:" + citation,
        "source_family": "court",
        "citation": citation,
        "is_gold": is_gold,
        "retrieval_text": " | ".join(p for p in retrieval_parts if p),
        "metadata": {
            "court_base": clean_text(card.get("court_base")),
            "legal_area": clean_text(card.get("legal_area")),
            "law_codes": as_list(card.get("law_codes")),
            "statutes_cited": as_list(card.get("statutes_cited")),
            "court_cases_cited": as_list(card.get("court_cases_cited")),
            "authority_role": as_list(card.get("authority_role")),
            "source_count": int(structural.get("court_base_source_count") or 0),
            "text_ref_count": int(structural.get("court_base_text_ref_count") or 0),
            "has_rag": bool(rag),
        },
    }


def reservoir_sample_laws(path: Path, n: int, *, exclude: set[str], rng: random.Random) -> list[dict[str, Any]]:
    sample: list[dict[str, Any]] = []
    seen = 0
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            citation = clean_text(row.get("citation"))
            if not citation or citation in exclude:
                continue
            doc = make_law_doc(row, is_gold=False)
            seen += 1
            if len(sample) < n:
                sample.append(doc)
            else:
                j = rng.randrange(seen)
                if j < n:
                    sample[j] = doc
    return sample


def load_gold_laws_and_noise(path: Path, gold: set[str], law_noise_n: int, rng: random.Random) -> tuple[list[dict[str, Any]], set[str], list[dict[str, Any]]]:
    gold_docs: list[dict[str, Any]] = []
    found: set[str] = set()
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            citation = clean_text(row.get("citation"))
            if citation in gold:
                gold_docs.append(make_law_doc(row, is_gold=True))
                found.add(citation)
    noise = reservoir_sample_laws(path, law_noise_n, exclude=gold, rng=rng)
    return gold_docs, found, noise


def load_gold_courts(path: Path, gold: set[str]) -> tuple[list[dict[str, Any]], set[str]]:
    gold_docs: list[dict[str, Any]] = []
    found: set[str] = set()
    remaining = set(gold)
    for i, card in enumerate(stream_jsonl(path), start=1):
        citation = clean_text(card.get("citation"))
        if citation in remaining:
            gold_docs.append(make_court_doc(card, is_gold=True))
            found.add(citation)
            remaining.remove(citation)
            if not remaining:
                break
        if i % 500_000 == 0:
            print(f"[gold courts] scanned {i:,}; found {len(found):,}/{len(gold):,}", flush=True)
    return gold_docs, found


def reservoir_sample_courts(path: Path, n: int, *, exclude: set[str], rng: random.Random) -> list[dict[str, Any]]:
    sample: list[dict[str, Any]] = []
    seen = 0
    for i, card in enumerate(stream_jsonl(path), start=1):
        citation = clean_text(card.get("citation"))
        if not citation or citation in exclude:
            continue
        doc = make_court_doc(card, is_gold=False)
        seen += 1
        if len(sample) < n:
            sample.append(doc)
        else:
            j = rng.randrange(seen)
            if j < n:
                sample[j] = doc
        if i % 100_000 == 0:
            print(f"[court noise] scanned {i:,}; reservoir={len(sample):,}", flush=True)
    return sample


def bm25_rank(docs: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    query_tokens = tokenize(expanded_query_text(query))
    qtf = Counter(query_tokens)
    doc_tokens: list[list[str]] = []
    df: Counter[str] = Counter()
    lengths: list[int] = []

    for doc in docs:
        tokens = tokenize(doc["retrieval_text"])
        doc_tokens.append(tokens)
        lengths.append(len(tokens))
        df.update(set(tokens))

    n_docs = len(docs)
    avgdl = sum(lengths) / max(n_docs, 1)
    k1 = 1.2
    b = 0.75
    idf = {
        term: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in df.items()
    }

    explicit_citations = set()
    for pattern in [ART_RE, BGE_RE, DOCKET_RE]:
        explicit_citations.update(m.group(0).replace(".", "_") if pattern is DOCKET_RE else m.group(0) for m in pattern.finditer(query))

    query_has_stpo = "stpo" in set(query_tokens)
    query_has_detention = bool({"pretrial", "detention", "collusion", "proportionality"} & set(query_tokens))

    ranked: list[dict[str, Any]] = []
    for idx, (doc, tokens) in enumerate(zip(docs, doc_tokens)):
        tf = Counter(tokens)
        dl = lengths[idx] or 1
        score = 0.0
        for term, q_count in qtf.items():
            if term not in tf:
                continue
            f = tf[term]
            denom = f + k1 * (1 - b + b * dl / max(avgdl, 1e-9))
            score += idf.get(term, 0.0) * (f * (k1 + 1) / denom) * min(q_count, 3)

        metadata = doc.get("metadata") or {}
        citation = doc["citation"]

        # Query-derived, non-gold metadata boosts.
        if any(citation.startswith(c) or c in citation for c in explicit_citations):
            score += 8.0
        if query_has_stpo:
            law_codes = " ".join(as_list(metadata.get("law_codes")) + [clean_text(metadata.get("law_code"))]).lower()
            statutes = " ".join(as_list(metadata.get("statutes_cited"))).lower()
            if "stpo" in law_codes or "stpo" in statutes or citation.endswith("StPO"):
                score += 2.5
        if query_has_detention and "criminal procedure" in clean_text(metadata.get("legal_area")).lower():
            score += 1.2
        if doc["source_family"] == "court":
            roles = set(as_list(metadata.get("authority_role")))
            if "published_leading_decision" in roles:
                score += 0.4
            if "frequently_cited_authority" in roles:
                score += 0.3
            score += min(math.log1p(int(metadata.get("source_count") or 0)) / 10, 0.35)

        row = dict(doc)
        row["score"] = score
        ranked.append(row)

    ranked.sort(key=lambda d: (-d["score"], d["source_family"], d["citation"]))
    return ranked


def write_candidates(path: Path, docs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            write_doc = dict(doc)
            write_doc["retrieval_text"] = write_doc["retrieval_text"][:3000]
            f.write(json.dumps(write_doc, ensure_ascii=False) + "\n")


def write_top_csv(path: Path, ranked: list[dict[str, Any]], top_k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "score", "is_gold", "source_family", "citation", "has_rag", "retrieval_preview"],
        )
        writer.writeheader()
        for rank, doc in enumerate(ranked[:top_k], start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "score": f"{doc['score']:.6f}",
                    "is_gold": doc["is_gold"],
                    "source_family": doc["source_family"],
                    "citation": doc["citation"],
                    "has_rag": (doc.get("metadata") or {}).get("has_rag", ""),
                    "retrieval_preview": doc["retrieval_text"][:240],
                }
            )


def recall_report(ranked: list[dict[str, Any]], gold_available: set[str], top_k: int) -> dict[str, Any]:
    top = ranked[:top_k]
    top_gold = {doc["citation"] for doc in top if doc["citation"] in gold_available}
    available_by_family = defaultdict(set)
    top_by_family = defaultdict(set)
    for citation in gold_available:
        available_by_family[citation_family(citation)].add(citation)
    for citation in top_gold:
        top_by_family[citation_family(citation)].add(citation)

    def rec(family: str | None = None) -> float:
        denom = gold_available if family is None else available_by_family[family]
        numer = top_gold if family is None else top_by_family[family]
        return len(numer) / len(denom) if denom else 0.0

    return {
        "top_k": top_k,
        "available_gold_count": len(gold_available),
        "recalled_gold_count": len(top_gold),
        "recall": rec(),
        "law_available": len(available_by_family["law"]),
        "law_recalled": len(top_by_family["law"]),
        "law_recall": rec("law"),
        "court_available": len(available_by_family["court"]),
        "court_recalled": len(top_by_family["court"]),
        "court_recall": rec("court"),
        "missed_gold": sorted(gold_available - top_gold),
    }


def main() -> int:
    cfg = CONFIG
    rng = random.Random(cfg.seed)
    t0 = time.time()

    query, gold = read_val_query(cfg.val_csv, cfg.query_id)
    gold_law = {g for g in gold if citation_family(g) == "law"}
    gold_court = {g for g in gold if citation_family(g) == "court"}

    print(f"[query] {cfg.query_id}: {len(gold):,} gold ({len(gold_law):,} law, {len(gold_court):,} court)")
    print(f"[query text] {query[:240]}...")

    law_gold_docs, found_law, law_noise = load_gold_laws_and_noise(cfg.laws_csv, gold_law, cfg.law_noise_n, rng)
    print(f"[laws] found gold {len(found_law):,}/{len(gold_law):,}; noise={len(law_noise):,}")

    court_gold_docs, found_court = load_gold_courts(cfg.court_v4_jsonl, gold_court)
    print(f"[courts] found gold {len(found_court):,}/{len(gold_court):,}")

    court_noise = reservoir_sample_courts(cfg.court_noise_jsonl, cfg.court_noise_n, exclude=found_court, rng=rng)
    print(f"[courts] noise={len(court_noise):,}")

    docs = law_gold_docs + court_gold_docs + law_noise + court_noise
    random.Random(cfg.seed + 1).shuffle(docs)

    if cfg.write_candidate_jsonl:
        write_candidates(cfg.output_candidates_jsonl, docs)
        print(f"[write] candidates -> {cfg.output_candidates_jsonl}")

    print(f"[rank] ranking {len(docs):,} candidates")
    ranked = bm25_rank(docs, query)
    available_gold = found_law | found_court
    report = recall_report(ranked, available_gold, cfg.top_k)

    result = {
        "query_id": cfg.query_id,
        "query": query,
        "noise": {
            "court_noise_n": cfg.court_noise_n,
            "law_noise_n": cfg.law_noise_n,
            "seed": cfg.seed,
        },
        "gold": {
            "total_in_val": len(gold),
            "law_in_val": len(gold_law),
            "court_in_val": len(gold_court),
            "law_found_in_source": len(found_law),
            "court_found_in_source": len(found_court),
            "missing_law_gold": sorted(gold_law - found_law),
            "missing_court_gold": sorted(gold_court - found_court),
        },
        "candidate_pool": {
            "total": len(docs),
            "law_gold": len(law_gold_docs),
            "court_gold": len(court_gold_docs),
            "law_noise": len(law_noise),
            "court_noise": len(court_noise),
            "court_docs_with_rag": sum(1 for d in docs if d["source_family"] == "court" and (d.get("metadata") or {}).get("has_rag")),
        },
        "recall_at_top_k": report,
        "top_gold_ranks": [
            {
                "rank": rank,
                "citation": doc["citation"],
                "source_family": doc["source_family"],
                "score": round(doc["score"], 6),
            }
            for rank, doc in enumerate(ranked, start=1)
            if doc["citation"] in available_gold
        ][:200],
        "elapsed_seconds": round(time.time() - t0, 3),
    }

    write_top_csv(cfg.output_top_csv, ranked, cfg.top_k)
    cfg.output_result_json.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_result_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result["recall_at_top_k"], ensure_ascii=False, indent=2))
    print(f"[write] top {cfg.top_k} -> {cfg.output_top_csv}")
    print(f"[write] results -> {cfg.output_result_json}")
    print(f"[done] elapsed {result['elapsed_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
