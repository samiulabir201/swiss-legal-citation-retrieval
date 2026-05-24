#!/usr/bin/env python
"""Court-only recall stress test with 1M random court-noise rows.

This is a streaming variant of experiment_enrichment_recall_stress.py.
It tests the court side only:

  - force the selected validation query's gold court citations into the pool
  - add exactly 1,000,000 random non-gold court rows from court_authority_cards_v4.jsonl
  - rank with BM25-style scoring over authority-card retrieval fields
  - report court recall@1000

The script uses gold labels only for evaluation, never for scoring.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from experiment_enrichment_recall_stress import (
    ART_RE,
    BGE_RE,
    DOCKET_RE,
    Config,
    as_list,
    citation_family,
    clean_text,
    expanded_query_text,
    make_court_doc,
    read_val_query,
    stream_jsonl,
    tokenize,
)


ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


QUERY_ID = "val_001"
TOP_K = 1000
COURT_NOISE_N = 1_000_000
SEED = 314159

# Known line count from the v4 artifact. Keeping this fixed avoids a full
# counting pass before the two scoring passes.
TOTAL_COURT_V4_LINES = 2_476_315
SAMPLE_OVERSHOOT = 5000

VAL_CSV = ROOT / "data" / "val.csv"
COURT_V4_JSONL = ROOT / "artifacts" / "court_authority_cards_v4.jsonl"
OUTPUT_RESULT_JSON = ROOT / "artifacts" / "court_recall_stress_val_001_1m_results.json"
OUTPUT_TOP_CSV = ROOT / "artifacts" / "court_recall_stress_val_001_1m_top1000.csv"


def select_noise_line_indices() -> set[int]:
    rng = random.Random(SEED)
    sample_size = min(TOTAL_COURT_V4_LINES, COURT_NOISE_N + SAMPLE_OVERSHOOT)
    return set(rng.sample(range(1, TOTAL_COURT_V4_LINES + 1), sample_size))


def make_query_state(query: str) -> dict[str, Any]:
    query_tokens = tokenize(expanded_query_text(query))
    qtf = Counter(query_tokens)
    explicit_citations = set()
    for pattern in [ART_RE, BGE_RE, DOCKET_RE]:
        for match in pattern.finditer(query):
            explicit_citations.add(match.group(0).replace(".", "_") if pattern is DOCKET_RE else match.group(0))
    token_set = set(query_tokens)
    return {
        "query_tokens": query_tokens,
        "qtf": qtf,
        "explicit_citations": explicit_citations,
        "query_has_stpo": "stpo" in token_set,
        "query_has_detention": bool({"pretrial", "detention", "collusion", "proportionality"} & token_set),
    }


def is_candidate_line(
    line_no: int,
    citation: str,
    gold_court: set[str],
    selected_noise_lines: set[int],
    included_noise_lines: set[int] | None,
    noise_count: int,
) -> tuple[bool, bool]:
    if citation in gold_court:
        return True, True
    if included_noise_lines is not None:
        return line_no in included_noise_lines, False
    if noise_count >= COURT_NOISE_N:
        return False, False
    return line_no in selected_noise_lines, False


def pass_one_stats(
    query_state: dict[str, Any],
    gold_court: set[str],
    selected_noise_lines: set[int],
) -> tuple[dict[str, Any], set[int], set[str]]:
    q_terms = set(query_state["qtf"])
    df: Counter[str] = Counter()
    length_sum = 0
    doc_count = 0
    included_noise_lines: set[int] = set()
    found_gold: set[str] = set()

    for line_no, card in enumerate(stream_jsonl(COURT_V4_JSONL), start=1):
        citation = clean_text(card.get("citation"))
        include, is_gold = is_candidate_line(
            line_no,
            citation,
            gold_court,
            selected_noise_lines,
            included_noise_lines=None,
            noise_count=len(included_noise_lines),
        )
        if not include:
            if line_no % 250_000 == 0:
                print(
                    f"[pass1] scanned={line_no:,}; noise={len(included_noise_lines):,}; gold={len(found_gold):,}/{len(gold_court):,}",
                    flush=True,
                )
            continue

        if is_gold:
            found_gold.add(citation)
        else:
            included_noise_lines.add(line_no)

        doc = make_court_doc(card, is_gold=is_gold)
        tokens = tokenize(doc["retrieval_text"])
        length_sum += len(tokens)
        doc_count += 1
        df.update(set(tokens) & q_terms)

        if line_no % 250_000 == 0:
            print(
                f"[pass1] scanned={line_no:,}; docs={doc_count:,}; noise={len(included_noise_lines):,}; gold={len(found_gold):,}/{len(gold_court):,}",
                flush=True,
            )

        if len(included_noise_lines) >= COURT_NOISE_N and found_gold == gold_court:
            break

    stats = {
        "doc_count": doc_count,
        "avgdl": length_sum / max(doc_count, 1),
        "df": dict(df),
    }
    return stats, included_noise_lines, found_gold


def score_doc(doc: dict[str, Any], tokens: list[str], stats: dict[str, Any], query_state: dict[str, Any]) -> float:
    tf = Counter(t for t in tokens if t in query_state["qtf"])
    n_docs = stats["doc_count"]
    avgdl = max(stats["avgdl"], 1e-9)
    dl = len(tokens) or 1
    k1 = 1.2
    b = 0.75
    score = 0.0

    for term, q_count in query_state["qtf"].items():
        f = tf.get(term, 0)
        if not f:
            continue
        freq = stats["df"].get(term, 0)
        idf = math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
        denom = f + k1 * (1 - b + b * dl / avgdl)
        score += idf * (f * (k1 + 1) / denom) * min(q_count, 3)

    metadata = doc.get("metadata") or {}
    citation = doc["citation"]

    if any(citation.startswith(c) or c in citation for c in query_state["explicit_citations"]):
        score += 8.0

    if query_state["query_has_stpo"]:
        law_codes = " ".join(as_list(metadata.get("law_codes"))).lower()
        statutes = " ".join(as_list(metadata.get("statutes_cited"))).lower()
        if "stpo" in law_codes or "stpo" in statutes:
            score += 2.5

    if query_state["query_has_detention"] and "criminal procedure" in clean_text(metadata.get("legal_area")).lower():
        score += 1.2

    roles = set(as_list(metadata.get("authority_role")))
    if "published_leading_decision" in roles:
        score += 0.4
    if "frequently_cited_authority" in roles:
        score += 0.3
    score += min(math.log1p(int(metadata.get("source_count") or 0)) / 10, 0.35)
    return score


def pass_two_rank(
    query_state: dict[str, Any],
    gold_court: set[str],
    included_noise_lines: set[int],
    stats: dict[str, Any],
) -> tuple[list[dict[str, Any]], set[str]]:
    heap: list[tuple[float, int, dict[str, Any]]] = []
    found_gold: set[str] = set()
    seen_docs = 0
    max_needed_line = max(included_noise_lines) if included_noise_lines else 0

    for line_no, card in enumerate(stream_jsonl(COURT_V4_JSONL), start=1):
        citation = clean_text(card.get("citation"))
        include, is_gold = is_candidate_line(
            line_no,
            citation,
            gold_court,
            selected_noise_lines=set(),
            included_noise_lines=included_noise_lines,
            noise_count=COURT_NOISE_N,
        )
        if not include:
            if line_no > max_needed_line and found_gold == gold_court:
                break
            continue

        if is_gold:
            found_gold.add(citation)

        doc = make_court_doc(card, is_gold=is_gold)
        tokens = tokenize(doc["retrieval_text"])
        score = score_doc(doc, tokens, stats, query_state)
        seen_docs += 1
        row = {
            "score": score,
            "is_gold": is_gold,
            "citation": citation,
            "source_family": "court",
            "has_rag": (doc.get("metadata") or {}).get("has_rag", False),
            "retrieval_preview": doc["retrieval_text"][:260],
        }

        heap_item = (score, seen_docs, row)
        if len(heap) < TOP_K:
            heapq.heappush(heap, heap_item)
        elif heap_item[0] > heap[0][0]:
            heapq.heapreplace(heap, heap_item)

        if seen_docs % 100_000 == 0:
            print(f"[pass2] ranked={seen_docs:,}; heap={len(heap):,}; gold={len(found_gold):,}/{len(gold_court):,}", flush=True)

    top = [item[2] for item in heap]
    top.sort(key=lambda row: (-row["score"], row["citation"]))
    return top, found_gold


def write_top_csv(top: list[dict[str, Any]]) -> None:
    OUTPUT_TOP_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_TOP_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "score", "is_gold", "citation", "has_rag", "retrieval_preview"],
        )
        writer.writeheader()
        for rank, row in enumerate(top, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "score": f"{row['score']:.6f}",
                    "is_gold": row["is_gold"],
                    "citation": row["citation"],
                    "has_rag": row["has_rag"],
                    "retrieval_preview": row["retrieval_preview"],
                }
            )


def main() -> int:
    t0 = time.time()
    query, gold = read_val_query(VAL_CSV, QUERY_ID)
    gold_court = {g for g in gold if citation_family(g) == "court"}
    query_state = make_query_state(query)
    selected_noise_lines = select_noise_line_indices()

    print(f"[query] {QUERY_ID}: court gold={len(gold_court):,}")
    print(f"[noise] target={COURT_NOISE_N:,}; selected_lines={len(selected_noise_lines):,}; seed={SEED}")

    stats, included_noise_lines, found_gold_pass1 = pass_one_stats(query_state, gold_court, selected_noise_lines)
    print(
        f"[pass1 done] docs={stats['doc_count']:,}; noise={len(included_noise_lines):,}; "
        f"gold_found={len(found_gold_pass1):,}/{len(gold_court):,}; avgdl={stats['avgdl']:.2f}"
    )

    top, found_gold_pass2 = pass_two_rank(query_state, gold_court, included_noise_lines, stats)
    top_gold = {row["citation"] for row in top if row["citation"] in found_gold_pass2}
    recall = len(top_gold) / len(found_gold_pass2) if found_gold_pass2 else 0.0

    result = {
        "query_id": QUERY_ID,
        "query": query,
        "top_k": TOP_K,
        "court_noise_target": COURT_NOISE_N,
        "court_noise_actual": len(included_noise_lines),
        "seed": SEED,
        "gold_court_total": len(gold_court),
        "gold_court_found": len(found_gold_pass2),
        "missing_gold_in_source": sorted(gold_court - found_gold_pass2),
        "court_recalled": len(top_gold),
        "court_recall": recall,
        "missed_gold_at_top_k": sorted(found_gold_pass2 - top_gold),
        "top_gold_ranks": [
            {
                "rank": rank,
                "citation": row["citation"],
                "score": round(row["score"], 6),
            }
            for rank, row in enumerate(top, start=1)
            if row["citation"] in found_gold_pass2
        ],
        "elapsed_seconds": round(time.time() - t0, 3),
    }

    write_top_csv(top)
    OUTPUT_RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: result[k] for k in ["court_noise_actual", "gold_court_found", "court_recalled", "court_recall", "missed_gold_at_top_k", "elapsed_seconds"]}, ensure_ascii=False, indent=2))
    print(f"[write] top -> {OUTPUT_TOP_CSV}")
    print(f"[write] result -> {OUTPUT_RESULT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
