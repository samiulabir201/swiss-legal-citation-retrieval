#!/usr/bin/env python
"""Build a slim LLM input JSONL from the static law authority cards.

Input:  artifacts/law_authority_cards_v1_static.jsonl
Output: artifacts/law_llm_input.jsonl

Each input record carries only what the LLM agent needs to produce the v2
enrichment without hallucinating: citation, law title + section path,
parsed structural fields, enactment year, static label hints (so the model
does not have to re-derive them), and the article text. Anchors and graph
fields stay on the static side and are merged back later.

Usage:
    python scripts/build_law_llm_input.py
    python scripts/build_law_llm_input.py --priority high,medium --skip-roles transitional_or_commencement,fees_or_costs
    python scripts/build_law_llm_input.py --limit 1000 --output artifacts/law_llm_input.smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"

DEFAULT_CARDS = ART_DIR / "law_authority_cards_v1_static.jsonl"
DEFAULT_OUTPUT = ART_DIR / "law_llm_input.jsonl"
DEFAULT_SUMMARY = ART_DIR / "law_llm_input.summary.json"

BOILERPLATE_ROLES = {
    "transitional_or_commencement",
    "fees_or_costs",
    "data_reporting",
}


def parse_csv_set(s: str | None) -> set[str]:
    if not s:
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


def keep_card(card: dict, priorities: set[str], skip_roles: set[str], min_chars: int) -> tuple[bool, str]:
    quality = card.get("enrichment_quality") or {}
    raw_text = (card.get("retrieval_views") or {}).get("raw_context") or ""
    if len(raw_text.strip()) < min_chars:
        return False, "below_min_chars"
    if priorities:
        if (quality.get("llm_priority") or "").lower() not in priorities:
            return False, "filtered_priority"
    if skip_roles:
        roles = set((card.get("rag_enrichment") or {}).get("provision_roles_static") or [])
        # Builder may store provision_roles directly under rag_enrichment, or
        # at top level — handle both.
        if not roles:
            roles = set(card.get("provision_roles") or [])
        if roles and roles.issubset(skip_roles):
            return False, "filtered_role"
    return True, "kept"


def build_record(card: dict) -> dict:
    """Slim per-row record for the LLM agent."""
    rag = card.get("rag_enrichment") or {}
    structural = card.get("structural") or card.get("normalized_anchors", {}).get("structural") or {}
    title_meta = card.get("title_metadata") or {}
    quality = card.get("enrichment_quality") or {}
    raw_text = (card.get("retrieval_views") or {}).get("raw_context") or ""

    static_hints = {
        "legal_area_static": rag.get("legal_area") or card.get("legal_area_static") or "",
        "domain_labels_en": rag.get("domain_labels_en") or card.get("domain_labels_en") or [],
        "issue_labels_en": rag.get("issue_labels_en") or card.get("issue_labels_en") or [],
        "provision_roles_static": rag.get("provision_roles_static")
            or card.get("provision_roles") or [],
        "matched_terms_multilingual": (
            rag.get("matched_terms_multilingual")
            or card.get("matched_terms_multilingual")
            or {}
        ),
    }

    return {
        "_source_row": int(card.get("_source_row", -1)),
        "citation": card.get("citation", "") or "",
        "law_title": card.get("law_title", "") or "",
        "title_section_path": card.get("title_section_path", "") or "",
        "structural": {
            "article": structural.get("article", ""),
            "units": structural.get("units", []),
            "law_code": structural.get("law_code", ""),
            "law_code_family": structural.get("law_code_family", ""),
            "granularity": structural.get("granularity", ""),
        },
        "title_metadata": {
            "source_type": title_meta.get("source_type", ""),
            "enactment_date": title_meta.get("enactment_date", ""),
            "enactment_year": title_meta.get("enactment_year", ""),
            "law_aliases": title_meta.get("law_aliases", []),
            "systematic_collection_sector": title_meta.get("systematic_collection_sector", ""),
        },
        "static_hints": static_hints,
        "llm_priority": quality.get("llm_priority", "medium"),
        "static_semantic_signal": quality.get("static_semantic_signal", ""),
        "text": raw_text,
        "_text_len": len(raw_text),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--priority", default="",
                    help="Comma-separated llm_priority values to keep (default: keep all). "
                         "E.g. 'high,medium' to skip 'low'.")
    ap.add_argument("--skip-roles", default="",
                    help="Comma-separated provision_roles_static values that, if a row is "
                         "ENTIRELY composed of, will be skipped. "
                         "E.g. 'transitional_or_commencement,fees_or_costs'.")
    ap.add_argument("--min-chars", type=int, default=20,
                    help="Skip rows with text shorter than this (default: 20).")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not args.cards.exists():
        raise SystemExit(f"Static cards JSONL not found: {args.cards}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    priorities = {p.lower() for p in parse_csv_set(args.priority)}
    skip_roles = parse_csv_set(args.skip_roles) or BOILERPLATE_ROLES & set()

    counts = Counter()
    skip_reasons = Counter()
    text_buckets = Counter()

    print(f"[build] reading {args.cards}", flush=True)
    print(f"[build] priorities={priorities or 'ALL'}  skip_roles={skip_roles or 'NONE'}  "
          f"min_chars={args.min_chars}", flush=True)

    t0 = time.time()
    with args.cards.open(encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            counts["scanned"] += 1
            try:
                card = json.loads(line)
            except Exception:
                counts["card_parse_error"] += 1
                continue
            keep, reason = keep_card(card, priorities, skip_roles, args.min_chars)
            if not keep:
                counts["skipped"] += 1
                skip_reasons[reason] += 1
                continue
            rec = build_record(card)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["written"] += 1
            n = rec["_text_len"]
            if n < 100: text_buckets["<100"] += 1
            elif n < 300: text_buckets["100-299"] += 1
            elif n < 600: text_buckets["300-599"] += 1
            elif n < 1500: text_buckets["600-1499"] += 1
            else: text_buckets[">=1500"] += 1
            if args.limit and counts["written"] >= args.limit:
                break

    elapsed = time.time() - t0
    summary = {
        "input": str(args.cards),
        "output": str(args.output),
        "filters": {
            "priorities": sorted(priorities),
            "skip_roles": sorted(skip_roles),
            "min_chars": args.min_chars,
            "limit": args.limit,
        },
        "elapsed_seconds": round(elapsed, 1),
        "counts": dict(counts),
        "skip_reasons": dict(skip_reasons),
        "text_length_buckets": dict(text_buckets),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] scanned={counts['scanned']:,}  written={counts['written']:,}  "
          f"skipped={counts['skipped']:,}  in {elapsed:.1f}s", flush=True)
    print(f"[done] output:  {args.output}  ({args.output.stat().st_size/1024**2:.1f} MB)", flush=True)
    print(f"[done] summary: {args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
