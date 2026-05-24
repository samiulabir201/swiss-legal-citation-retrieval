#!/usr/bin/env python
"""Court recall stress test at 1M noise with enriched gold card overrides.

Identical to experiment_court_recall_stress_1m.py except:
  - Loads val001_gold_court_enriched.jsonl as enriched card overrides
  - When a gold card's citation matches an override, uses the enriched version
  - All noise rows remain unenriched v4-only cards

This is an A/B comparison to test whether rag_enrichment fields help
gold cards surface above 1M noise rows:

  Baseline (v4-only):  13 / 23 = 0.565   [experiment_court_recall_stress_1m.py]
  Enriched (this):     ?  / 23 = ?

The two runs use the same SEED and the same 1M noise lines, so results
are directly comparable.
"""

from __future__ import annotations

import csv
import heapq
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from experiment_enrichment_recall_stress import (
    citation_family,
    clean_text,
    make_court_doc,
    read_val_query,
    stream_jsonl,
    tokenize,
)
from experiment_court_recall_stress_1m import (
    COURT_NOISE_N,
    SEED,
    TOP_K,
    is_candidate_line,
    make_query_state,
    score_doc,
    select_noise_line_indices,
)

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

QUERY_ID = "val_001"
VAL_CSV = ROOT / "data" / "val.csv"
COURT_V4_JSONL = ROOT / "artifacts" / "court_authority_cards_v4.jsonl"
ENRICHED_GOLD_JSONL = ROOT / "artifacts" / "val001_gold_court_enriched.jsonl"
OUTPUT_RESULT_JSON = ROOT / "artifacts" / "court_recall_enriched_val_001_1m_results.json"
OUTPUT_TOP_CSV = ROOT / "artifacts" / "court_recall_enriched_val_001_1m_top1000.csv"

BASELINE_RECALL = 0.5652173913043478
BASELINE_RECALLED = 13


def load_enriched_overrides(path: Path) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    if not path.exists():
        raise FileNotFoundError(
            f"Enriched gold JSONL not found: {path}\n"
            "Run enrich_val001_gold_cards.py first."
        )
    for card in stream_jsonl(path):
        citation = clean_text(card.get("citation"))
        if citation:
            overrides[citation] = card
    print(f"[overrides] Loaded {len(overrides)} enriched gold cards from {path.name}")
    for citation, card in sorted(overrides.items()):
        rag = card.get("rag_enrichment") or {}
        method = rag.get("method", "none")
        topic = rag.get("legal_topic", "")[:60]
        print(f"  {citation}  [{method}]  {topic}")
    return overrides


def maybe_override(
    card: dict[str, Any], overrides: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    citation = clean_text(card.get("citation"))
    return overrides.get(citation, card)


def pass_one_stats(
    query_state: dict[str, Any],
    gold_court: set[str],
    selected_noise_lines: set[int],
    overrides: dict[str, dict[str, Any]],
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
                    f"[pass1] scanned={line_no:,}; noise={len(included_noise_lines):,}; "
                    f"gold={len(found_gold):,}/{len(gold_court):,}",
                    flush=True,
                )
            continue

        if is_gold:
            found_gold.add(citation)
        else:
            included_noise_lines.add(line_no)

        card = maybe_override(card, overrides)
        doc = make_court_doc(card, is_gold=is_gold)
        tokens = tokenize(doc["retrieval_text"])
        length_sum += len(tokens)
        doc_count += 1
        df.update(set(tokens) & q_terms)

        if line_no % 250_000 == 0:
            print(
                f"[pass1] scanned={line_no:,}; docs={doc_count:,}; "
                f"noise={len(included_noise_lines):,}; gold={len(found_gold):,}/{len(gold_court):,}",
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


def pass_two_rank(
    query_state: dict[str, Any],
    gold_court: set[str],
    included_noise_lines: set[int],
    stats: dict[str, Any],
    overrides: dict[str, dict[str, Any]],
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

        card = maybe_override(card, overrides)
        doc = make_court_doc(card, is_gold=is_gold)
        tokens = tokenize(doc["retrieval_text"])
        score = score_doc(doc, tokens, stats, query_state)
        seen_docs += 1

        rag = card.get("rag_enrichment")
        row = {
            "score": score,
            "is_gold": is_gold,
            "citation": citation,
            "source_family": "court",
            "has_rag": bool(isinstance(rag, dict) and rag),
            "retrieval_preview": doc["retrieval_text"][:260],
        }

        heap_item = (score, seen_docs, row)
        if len(heap) < TOP_K:
            heapq.heappush(heap, heap_item)
        elif heap_item[0] > heap[0][0]:
            heapq.heapreplace(heap, heap_item)

        if seen_docs % 100_000 == 0:
            print(
                f"[pass2] ranked={seen_docs:,}; heap={len(heap):,}; "
                f"gold={len(found_gold):,}/{len(gold_court):,}",
                flush=True,
            )

    top = [item[2] for item in heap]
    top.sort(key=lambda r: (-r["score"], r["citation"]))
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
            writer.writerow({
                "rank": rank,
                "score": f"{row['score']:.6f}",
                "is_gold": row["is_gold"],
                "citation": row["citation"],
                "has_rag": row["has_rag"],
                "retrieval_preview": row["retrieval_preview"],
            })


def main() -> int:
    t0 = time.time()

    overrides = load_enriched_overrides(ENRICHED_GOLD_JSONL)
    query, gold = read_val_query(VAL_CSV, QUERY_ID)
    gold_court = {g for g in gold if citation_family(g) == "court"}
    query_state = make_query_state(query)
    selected_noise_lines = select_noise_line_indices()

    print(f"\n[query] {QUERY_ID}: court_gold={len(gold_court):,}; enriched_overrides={len(overrides)}")
    print(f"[noise] target={COURT_NOISE_N:,}; seed={SEED}")

    stats, included_noise_lines, found_gold_p1 = pass_one_stats(
        query_state, gold_court, selected_noise_lines, overrides
    )
    print(
        f"[pass1 done] docs={stats['doc_count']:,}; noise={len(included_noise_lines):,}; "
        f"gold_found={len(found_gold_p1):,}/{len(gold_court):,}; avgdl={stats['avgdl']:.2f}"
    )

    top, found_gold_p2 = pass_two_rank(
        query_state, gold_court, included_noise_lines, stats, overrides
    )
    top_gold = {row["citation"] for row in top if row["citation"] in found_gold_p2}
    recall = len(top_gold) / len(found_gold_p2) if found_gold_p2 else 0.0

    missed = sorted(found_gold_p2 - top_gold)
    top_gold_ranks = [
        {
            "rank": rank,
            "citation": row["citation"],
            "score": round(row["score"], 6),
            "has_rag": row["has_rag"],
        }
        for rank, row in enumerate(top, start=1)
        if row["citation"] in found_gold_p2
    ]

    result = {
        "query_id": QUERY_ID,
        "top_k": TOP_K,
        "court_noise_target": COURT_NOISE_N,
        "court_noise_actual": len(included_noise_lines),
        "seed": SEED,
        "enriched_overrides": len(overrides),
        "gold_court_total": len(gold_court),
        "gold_court_found": len(found_gold_p2),
        "missing_gold_in_source": sorted(gold_court - found_gold_p2),
        "court_recalled": len(top_gold),
        "court_recall": recall,
        "missed_gold_at_top_k": missed,
        "top_gold_ranks": top_gold_ranks,
        "baseline_recalled": BASELINE_RECALLED,
        "baseline_recall": BASELINE_RECALL,
        "delta_recalled": len(top_gold) - BASELINE_RECALLED,
        "delta_recall": round(recall - BASELINE_RECALL, 4),
        "elapsed_seconds": round(time.time() - t0, 3),
    }

    write_top_csv(top)
    OUTPUT_RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RESULT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("RESULTS (enriched gold vs v4-only baseline at 1M noise)")
    print("=" * 60)
    print(f"  Baseline:  {BASELINE_RECALLED} / {len(found_gold_p2)} = {BASELINE_RECALL:.4f}")
    print(f"  Enriched:  {len(top_gold)} / {len(found_gold_p2)} = {recall:.4f}")
    print(f"  Delta:     {result['delta_recalled']:+d} citations  ({result['delta_recall']:+.4f})")
    print(f"  Missed:    {missed}")
    print(f"  Elapsed:   {result['elapsed_seconds']:.1f}s")
    print(f"\n[write] top  -> {OUTPUT_TOP_CSV}")
    print(f"[write] result -> {OUTPUT_RESULT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
