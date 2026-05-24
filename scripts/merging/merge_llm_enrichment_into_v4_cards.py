#!/usr/bin/env python
"""Merge the 363k LLM enrichment with the 2.1M static-only v4 cards into a unified v5 corpus.

For each v4 card (line_idx 0..2,476,309):
  - If the line_idx is in the LLM index -> normalize with LLM enrichment + static metadata.
  - Else                                -> normalize with static metadata only
                                            (seed_metadata_fields handles the fallback:
                                            issue_labels_en -> concepts_en,
                                            matched_terms_multilingual -> terms_original,
                                            metadata.legal_area -> legal_area, etc.).

Both populations end up with the same retrieval_views shape, so the RAG retriever sees
one coherent corpus.

Usage:
  python scripts/merge_llm_enrichment_into_v4_cards.py \\
      --llm "C:/Users/samiul/Downloads/outputs_from_363k_run/court_llm_descriptors_0000000_all.jsonl"

Smoke test (1000 cards, fast):
  python scripts/merge_llm_enrichment_into_v4_cards.py --llm <path> --limit 1000 \\
      --output artifacts/court_authority_cards_v5_unified.smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from court_enrichment_normalizer import normalize_enriched_court_row  # noqa: E402

ART_DIR = ROOT / "artifacts"
DEFAULT_CARDS = ART_DIR / "court_authority_cards_v4.jsonl"
DEFAULT_OUTPUT = ART_DIR / "court_authority_cards_v5_unified.jsonl"
DEFAULT_SUMMARY = ART_DIR / "court_authority_cards_v5_unified.summary.json"
TOTAL_V4_CARDS = 2_476_310  # known, used for ETA only


def load_llm_index(path: Path) -> tuple[dict[int, dict], Counter]:
    """Load the LLM JSONL into {line_idx: llm_enrichment_dict}.

    Memory: ~700-900 MB for 363k records. Stays in RAM for O(1) lookup.
    Failed-status records are dropped here so they fall back to static-only enrichment.
    """
    print(f"[load] Reading LLM enrichment from {path}", flush=True)
    index: dict[int, dict] = {}
    counts = Counter()
    t0 = time.time()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                counts["bad_json"] += 1
                continue
            src = rec.get("_source_row")
            if src is None:
                counts["missing_source_row"] += 1
                continue
            status = (rec.get("llm_generation") or {}).get("status", "")
            if status.startswith("failed"):
                counts["failed_status"] += 1
                continue
            llm = rec.get("llm_enrichment") or {}
            if not isinstance(llm, dict) or not llm:
                counts["empty_enrichment"] += 1
                continue
            index[int(src)] = llm
            counts["loaded"] += 1
    elapsed = time.time() - t0
    print(f"[load]   loaded={counts['loaded']:,}  "
          f"failed={counts['failed_status']:,}  "
          f"bad_json={counts['bad_json']:,}  "
          f"empty={counts['empty_enrichment']:,}  "
          f"in {elapsed:.1f}s", flush=True)
    return index, counts


def merge(args: argparse.Namespace) -> dict:
    llm_index, llm_load_counts = load_llm_index(args.llm)

    stats = Counter()
    role_dist = Counter()
    source_dist = Counter()
    outcome_dist = Counter()
    field_fill = Counter()
    field_fill_by_source = {"static": Counter(), "llm+static": Counter()}
    legal_area_dist = Counter()
    language_dist = Counter()
    role_corrections = Counter()
    outcome_corrections = Counter()
    has_anchor = Counter()  # has_statute_anchor / has_case_anchor / has_terms / etc.

    t0 = time.time()
    print(f"[merge] Streaming {args.cards} -> {args.output}", flush=True)
    with args.cards.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line_idx, line in enumerate(fin):
            if args.limit and stats["scanned"] >= args.limit:
                break
            stats["scanned"] += 1
            line = line.strip()
            if not line:
                stats["empty_line"] += 1
                continue
            try:
                card = json.loads(line)
            except Exception:
                stats["card_parse_error"] += 1
                continue

            citation = card.get("citation", "") or ""
            text = card.get("text_excerpt_original", "") or ""
            llm_enrichment = llm_index.get(line_idx)
            enrichment_source = "llm+static" if llm_enrichment is not None else "static"

            try:
                normalized = normalize_enriched_court_row(
                    citation=citation,
                    text=text,
                    llm_enrichment=llm_enrichment or {},
                    deterministic_metadata=card,
                )
            except Exception as exc:
                stats["normalizer_error"] += 1
                normalized = {
                    "rag_enrichment": {},
                    "normalized_anchors": {},
                    "anchor_quality_flags": {"normalizer_error": [repr(exc)[:300]]},
                    "retrieval_views": {},
                    "enrichment_quality": {"normalizer": "error"},
                }

            rag = normalized["rag_enrichment"]
            anchors = normalized["normalized_anchors"]
            views = normalized["retrieval_views"]
            quality = normalized["enrichment_quality"]
            flags = normalized["anchor_quality_flags"]

            # Output record. Slim mode (default) drops the v4 passthrough fields
            # to keep file size manageable; full mode keeps them for full provenance.
            out_rec: dict = {
                "_source_row": line_idx,
                "citation": citation,
                "court_base": card.get("court_base", "") or "",
                "language": card.get("language", "") or "",
                "legal_area_static": card.get("legal_area", "") or "",
                "enrichment_source": enrichment_source,
                "text_excerpt_original": text,
                "rag_enrichment": rag,
                "normalized_anchors": anchors,
                "anchor_quality_flags": flags,
                "retrieval_views": views,
                "enrichment_quality": quality,
            }
            if args.full:
                out_rec.update({
                    "structural": card.get("structural", {}),
                    "issue_labels_en_v4": card.get("issue_labels_en", []),
                    "matched_terms_multilingual_v4": card.get("matched_terms_multilingual", {}),
                    "statutes_cited_v4": card.get("statutes_cited", []),
                    "court_cases_cited_v4": card.get("court_cases_cited", []),
                    "law_codes_v4": card.get("law_codes", []),
                    "authority_role_v4": card.get("authority_role", []),
                    "summary_en_proxy_v4": card.get("summary_en_proxy", ""),
                    "retrieval_text_en_v4": card.get("retrieval_text_en", ""),
                })

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            stats["written"] += 1
            source_dist[enrichment_source] += 1

            # ---- distribution counters --------------------------------------
            role_dist[rag.get("paragraph_role", "")] += 1
            outcome_dist[rag.get("outcome_signal", "none")] += 1
            la = rag.get("legal_area") or card.get("legal_area") or ""
            legal_area_dist[la[:80]] += 1
            language_dist[card.get("language", "") or "unknown"] += 1

            for k, v in rag.items():
                if isinstance(v, list):
                    if v:
                        field_fill[k] += 1
                        field_fill_by_source[enrichment_source][k] += 1
                elif isinstance(v, str):
                    if v.strip():
                        field_fill[k] += 1
                        field_fill_by_source[enrichment_source][k] += 1
                elif isinstance(v, (int, float)):
                    if v:
                        field_fill[k] += 1
                        field_fill_by_source[enrichment_source][k] += 1

            if quality.get("has_statute_anchor"):
                has_anchor["statute"] += 1
            if quality.get("has_case_anchor"):
                has_anchor["case"] += 1
            if quality.get("has_original_language_terms"):
                has_anchor["terms"] += 1
            if quality.get("has_specific_topic"):
                has_anchor["specific_topic"] += 1

            for corr in flags.get("role_corrections", []) or []:
                if isinstance(corr, dict):
                    role_corrections[(corr.get("from", ""), corr.get("to", ""))] += 1
            for corr in flags.get("outcome_corrections", []) or []:
                if isinstance(corr, dict):
                    outcome_corrections[(corr.get("from", ""), corr.get("to", ""))] += 1

            if stats["scanned"] % args.progress_every == 0:
                rate = stats["scanned"] / max(time.time() - t0, 1e-9)
                remaining = max(TOTAL_V4_CARDS - stats["scanned"], 0)
                eta = remaining / max(rate, 1e-9)
                print(f"  scanned={stats['scanned']:>10,}  written={stats['written']:>10,}  "
                      f"({rate:>6.0f} rec/s, ETA {eta/60:>5.1f} min)", flush=True)

    elapsed = time.time() - t0
    rate = stats["written"] / max(elapsed, 1e-9)

    summary = {
        "inputs": {"cards": str(args.cards), "llm": str(args.llm)},
        "output": str(args.output),
        "limit": args.limit,
        "full_passthrough": args.full,
        "elapsed_seconds": round(elapsed, 1),
        "rows_per_second": round(rate, 1),
        "stats": dict(stats),
        "llm_load_counts": dict(llm_load_counts),
        "enrichment_source_counts": dict(source_dist),
        "paragraph_role_distribution": role_dist.most_common(),
        "outcome_distribution": outcome_dist.most_common(),
        "language_distribution": language_dist.most_common(),
        "top_legal_areas": legal_area_dist.most_common(30),
        "has_anchor_counts": dict(has_anchor),
        "rag_field_fill_total": field_fill.most_common(),
        "rag_field_fill_static_only": field_fill_by_source["static"].most_common(),
        "rag_field_fill_llm_static": field_fill_by_source["llm+static"].most_common(),
        "rag_field_fill_pct_total": {
            k: round(c / max(stats["written"], 1) * 100, 2)
            for k, c in field_fill.items()
        },
        "rag_field_fill_pct_by_source": {
            src: {
                k: round(c / max(source_dist.get(src, 1), 1) * 100, 2)
                for k, c in counter.items()
            }
            for src, counter in field_fill_by_source.items()
        },
        "role_corrections_top20": [
            {"from": k[0], "to": k[1], "count": v}
            for k, v in role_corrections.most_common(20)
        ],
        "outcome_corrections_top20": [
            {"from": k[0], "to": k[1], "count": v}
            for k, v in outcome_corrections.most_common(20)
        ],
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[done] scanned={stats['scanned']:,}  written={stats['written']:,}  "
          f"in {elapsed/60:.1f} min ({rate:.0f} rec/s)")
    print(f"[done] static-only:  {source_dist.get('static', 0):,}")
    print(f"[done] llm+static:   {source_dist.get('llm+static', 0):,}")
    print(f"[done] normalizer_errors: {stats.get('normalizer_error', 0):,}")
    print(f"[done] output:       {args.output}  "
          f"({args.output.stat().st_size / 1024**3:.2f} GB)")
    print(f"[done] summary:      {args.summary}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", type=Path, required=True,
                    help="Path to court_llm_descriptors_0000000_all.jsonl from the 363k run.")
    ap.add_argument("--cards", type=Path, default=DEFAULT_CARDS,
                    help=f"v4 cards JSONL (default: {DEFAULT_CARDS})")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"Output unified JSONL (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY,
                    help=f"Output summary JSON (default: {DEFAULT_SUMMARY})")
    ap.add_argument("--progress-every", type=int, default=50_000,
                    help="Print progress every N scanned rows (default: 50000)")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = process all 2.4M cards. Set to e.g. 1000 for a smoke test.")
    ap.add_argument("--full", action="store_true",
                    help="Include all v4 passthrough fields in output (larger file). "
                         "Default: slim output (only normalized fields + text + key v4 metadata).")
    args = ap.parse_args()

    if not args.llm.exists():
        raise SystemExit(f"LLM JSONL not found: {args.llm}")
    if not args.cards.exists():
        raise SystemExit(f"v4 cards JSONL not found: {args.cards}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    merge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
