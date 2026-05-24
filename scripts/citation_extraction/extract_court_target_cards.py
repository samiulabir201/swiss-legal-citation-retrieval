#!/usr/bin/env python
"""Extract full v4 authority cards for selected enrichment target rows.

The target manifest from identify_court_enrichment_targets.py contains row
numbers and reasons, not the full card payload. This script materializes the
selected cards into a compact JSONL so Colab can enrich only target cards
without scanning the full 2.47M-row corpus every test run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"
DEFAULT_CARDS = ART_DIR / "court_authority_cards_v4.jsonl"
DEFAULT_TARGETS = ART_DIR / "court_authority_cards_v4_enrichment_targets.jsonl"
DEFAULT_OUTPUT = ART_DIR / "court_authority_cards_v4_target_cards.jsonl"
DEFAULT_SUMMARY = ART_DIR / "court_authority_cards_v4_target_cards.summary.json"


def load_targets(path: Path, limit: int = 0) -> tuple[set[int], dict[int, dict[str, Any]]]:
    indices: set[int] = set()
    meta: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row["line_idx"])
            indices.add(idx)
            meta[idx] = row
            if limit and len(indices) >= limit:
                break
    return indices, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    ap.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--limit", type=int, default=0, help="0 means all targets.")
    ap.add_argument("--progress-every", type=int, default=500000)
    args = ap.parse_args()

    target_indices, target_meta = load_targets(args.targets, args.limit)
    if not target_indices:
        raise SystemExit(f"No target rows loaded from {args.targets}")

    stats = Counter()
    reasons = Counter()
    areas = Counter()
    languages = Counter()
    found_indices: set[int] = set()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.cards.open(encoding="utf-8") as inp, args.output.open("w", encoding="utf-8") as out:
        for line_idx, line in enumerate(inp):
            stats["scanned_cards"] += 1
            if line_idx not in target_indices:
                if args.progress_every and stats["scanned_cards"] % args.progress_every == 0:
                    print(
                        f"[scan] scanned={stats['scanned_cards']:,} "
                        f"written={stats['written_targets']:,}",
                        flush=True,
                    )
                continue
            if not line.strip():
                continue
            card = json.loads(line)
            meta = target_meta[line_idx]
            card["_enrichment_target"] = meta
            out.write(json.dumps(card, ensure_ascii=False) + "\n")
            found_indices.add(line_idx)
            stats["written_targets"] += 1
            for reason in meta.get("reasons", []):
                reasons[reason] += 1
            areas[meta.get("legal_area", "")] += 1
            languages[meta.get("language", "")] += 1

            # Once every requested index has been materialized, stop early.
            if len(found_indices) >= len(target_indices):
                break

            if args.progress_every and stats["scanned_cards"] % args.progress_every == 0:
                print(
                    f"[scan] scanned={stats['scanned_cards']:,} "
                    f"written={stats['written_targets']:,}",
                    flush=True,
                )

    missing = sorted(target_indices - found_indices)
    summary = {
        "cards": str(args.cards),
        "targets": str(args.targets),
        "output": str(args.output),
        "limit": args.limit,
        "target_count": len(target_indices),
        "written_targets": stats["written_targets"],
        "missing_target_count": len(missing),
        "missing_target_indices_sample": missing[:20],
        "stats": dict(stats),
        "top_reasons": reasons.most_common(50),
        "top_legal_areas": areas.most_common(30),
        "top_languages": languages.most_common(10),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] targets={len(target_indices):,} written={stats['written_targets']:,}")
    print(f"[done] output={args.output}")
    print(f"[done] summary={args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
