#!/usr/bin/env python
"""Audit court rows whose `statute_anchors` is empty: do their text fields
ACTUALLY contain statute citations that the build-time regex missed, or are
those rows genuinely without explicit statute citations (discussion / facts /
procedural narrative)?

This is the P2 question:
  - If many empty-anchor rows DO contain unmissed citations -> regex is buggy
    and we should re-extract on the corpus.
  - If most empty-anchor rows really have nothing to extract -> regex is fine
    and the gap is structural (we can't manufacture citations that aren't
    there).

How it works:
  1. Stream `court_authority_cards_v5_unified.jsonl`.
  2. For rows with empty statute_anchors, run a comprehensive regex over the
     same text field that build-time used (`text_excerpt_original`).
  3. Categorize matches into:
        - HARD MATCH:  "Art. N CODE" / "art. N CODE" / "articolo N CODE" /
                       "Artikel N CODE" — clearly a statute citation.
        - SECTION:     "§ N CODE" — German section sign.
        - SOFT TOKEN:  has 'Art.' or 'art.' but no recognized code follows
                       (likely incomplete citation or foreign-format).
        - NO CITATION: text contains zero of the above markers.
  4. Report counts + sample of hard misses (the regex should have caught these).

Usage:
    python scripts/audit_statute_anchor_regex.py [--limit N]

Output: prints to stdout. Save with shell redirect if needed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path


CARDS = Path(r"E:/swiss_citation_extraction/artifacts/court_authority_cards_v5_unified.jsonl")


# ---------------------------------------------------------------------------
# Comprehensive multilingual statute-citation patterns
# ---------------------------------------------------------------------------

# Known Swiss federal & cantonal codes — same set used in
# identify_court_enrichment_targets.py plus French/Italian aliases.
KNOWN_CODES = {
    "AHVG", "AHVV", "AIG", "AVIG", "ATSG", "AsylG", "AuG", "BankG", "BetmG",
    "BGFA", "BGG", "BPV", "BV", "BVG", "BBl", "AS",
    # French aliases (mapped equivalents):
    "CC", "CO", "CP", "CPC", "CPP", "Cst", "Cst.",
    "DBG", "DSG", "EMRK", "FINMAG", "FINIG", "FINIV", "FZG", "GwG", "GSchG",
    "HMG", "IPRG", "IRSG", "IVG", "JStG", "JStPO", "KG", "KKG", "KVG", "KVV",
    # French aliases:
    "LAA", "LAI", "LAMal", "LAVS", "LCart", "LCD", "LDA", "LDAl", "LDes",
    "LEI", "LEtr", "LIFD", "LFus", "LIA", "LJP", "LJPM", "LP", "LPGA",
    "LPP", "LSCPT", "LStup", "LTF",
    # Italian aliases:
    "LIPP",
    "MSchG", "MWSTG", "NHG", "OBG", "OG", "OJ", "OR", "OBV", "PatG", "PrHG",
    "RAG", "RPG", "RPV", "SchKG", "SebG", "StGB", "StPO", "StHG", "StBOG",
    "SVG", "TEVG", "URG", "USG", "UVG", "UWG", "VStG", "VStV", "VStrR",
    "VAG", "VAV", "VBVV", "VFP", "VVG", "VVEA", "VwVG", "ZGB", "ZPO",
    # New / additional
    "ATF", "DTF", "FIDLEG", "FIDLEV", "FinfraG", "FinfraV", "GBV", "GUMG",
    "HKsÜ", "HKsU", "LugÜ", "LugU", "NBG", "OBG", "PrHG", "RVOG", "SVKG",
    "TwwV", "UEG", "VFP", "WG", "WRV", "BPR", "BüG", "BueG", "BüV", "BueV",
    "ChemG", "EOG", "ERV", "EleG", "EpG", "FFG", "FZA", "GeneTG", "GUMG",
    "EBG", "TSchG", "VAV", "VVEA", "VWGV", "VST", "BankV",
}
# Lowercase aliases too
KNOWN_CODES_ALL = set(KNOWN_CODES) | {c.lower() for c in KNOWN_CODES}

CODE_TOKEN = r'[A-ZÆ][A-Za-zÆ0-9]{1,9}\.?(?:/(?:[A-Z]{2,3}|\d+))?'

# Hard pattern: Art./Artikel/art./articolo/Article + N + optional sub-units + CODE
HARD_PATTERNS = [
    # Art. 221 Abs. 1 lit. b StPO  /  Art. 221 StPO  /  Art. 221bis StPO
    re.compile(
        r'(?<!\w)(?:Art(?:icle)?\.?|Artikel|art\.?|art\.|articolo|articoli)\s+'
        r'\d+(?:[a-z]+)?'
        r'(?:\s+(?:Abs\.?|Absatz|al\.?|alinéa|alinea|cpv\.?)\s*\d+(?:[a-z]+)?)?'
        r'(?:\s+(?:lit\.?|let\.?|Bst\.?|Buchstabe)\s*[a-z])?'
        r'(?:\s+(?:Ziff\.?|Ziffer|n\.|no\.|num\.|cifra)\s*\d+(?:[a-z]+)?)?'
        rf'\s+(?P<code>{CODE_TOKEN})',
        re.IGNORECASE,
    ),
]

# Section sign patterns
SECTION_PATTERNS = [
    re.compile(r'(?<!\w)§§?\s*\d+(?:[a-z]+)?', re.IGNORECASE),
]

# Soft pattern: any "Art." token (likely-citation but no code captured)
SOFT_PATTERNS = [
    re.compile(r'(?<!\w)(?:Art(?:icle)?\.?|Artikel|articolo)\s+\d', re.IGNORECASE),
]


def classify_text(text: str) -> tuple[str, list[str]]:
    """Return (category, captured_citations). Categories:
    - HARD: "Art. N CODE" pattern matched and code is in KNOWN_CODES
    - HARD_UNKNOWN_CODE: pattern matched but code not in KNOWN_CODES
    - SECTION: § pattern matched
    - SOFT: Art./art./Artikel without recognized code
    - NONE: no statute markers
    """
    if not text:
        return "NONE", []

    hard_hits: list[str] = []
    hard_unknown: list[str] = []
    for pat in HARD_PATTERNS:
        for m in pat.finditer(text):
            code = m.group("code").rstrip('.')
            full = m.group(0).strip()
            if code in KNOWN_CODES_ALL or code.upper() in KNOWN_CODES_ALL:
                hard_hits.append(full)
            else:
                hard_unknown.append(f"{full} [code={code!r}]")
    if hard_hits:
        return "HARD", hard_hits
    if hard_unknown:
        return "HARD_UNKNOWN_CODE", hard_unknown

    sec_hits = []
    for pat in SECTION_PATTERNS:
        for m in pat.finditer(text):
            sec_hits.append(m.group(0))
    if sec_hits:
        return "SECTION", sec_hits

    for pat in SOFT_PATTERNS:
        for m in pat.finditer(text):
            return "SOFT", [m.group(0)]

    return "NONE", []


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most this many rows (default: full corpus)")
    p.add_argument("--show-samples", type=int, default=12,
                   help="Print this many sample HARD misses for inspection")
    args = p.parse_args(argv)

    if not CARDS.exists():
        print(f"missing {CARDS}", file=sys.stderr); return 2

    n_total = 0
    n_with_anchors = 0
    n_empty = 0
    cat_counts: Counter[str] = Counter()
    hard_miss_samples: list[dict] = []

    print(f"Auditing {CARDS} ({CARDS.stat().st_size/1024**3:.2f} GB)...", flush=True)
    t0 = time.time()
    with CARDS.open(encoding="utf-8") as fh:
        for line in fh:
            try: obj = json.loads(line)
            except Exception: continue
            n_total += 1
            if args.limit and n_total > args.limit: break

            rag = obj.get("rag_enrichment") or {}
            if rag.get("statute_anchors"):
                n_with_anchors += 1
                continue
            n_empty += 1

            text = obj.get("text_excerpt_original") or ""
            cat, hits = classify_text(text)
            cat_counts[cat] += 1
            if cat == "HARD" and len(hard_miss_samples) < args.show_samples:
                hard_miss_samples.append({
                    "citation": obj.get("citation",""),
                    "category": cat,
                    "regex_hits": hits[:6],
                    "text_preview": text[:280].replace("\n", " "),
                    "language": obj.get("language") or rag.get("language"),
                })

            if n_total % 250_000 == 0:
                print(f"  scanned {n_total:,} ({time.time()-t0:.1f}s)", flush=True)

    print(f"Done in {time.time()-t0:.1f}s\n")
    print(f"Total court rows scanned:       {n_total:,}")
    print(f"  with non-empty anchors:        {n_with_anchors:,} ({100*n_with_anchors/max(1,n_total):.1f}%)")
    print(f"  with empty anchors:            {n_empty:,} ({100*n_empty/max(1,n_total):.1f}%)")
    print()
    print("Empty-anchor row classification (by what regex DOES find in text):")
    for cat, n in cat_counts.most_common():
        pct = 100 * n / max(1, n_empty)
        print(f"  {cat:<20}  {n:>10,}  ({pct:5.1f}%)")
    print()
    print("Interpretation:")
    print("  HARD               -> regex SHOULD have caught these. Real bug; re-extract.")
    print("  HARD_UNKNOWN_CODE  -> citation present but code unrecognized. Add code to vocab.")
    print("  SECTION            -> § citation present. Build-time regex may not handle these.")
    print("  SOFT               -> 'Art.' present but no code; rare false positive.")
    print("  NONE               -> text genuinely has no statute citation. Empty is correct.")

    if hard_miss_samples:
        print()
        print(f"=== Sample HARD misses (regex found citations the build missed) ===")
        for s in hard_miss_samples:
            print(f"\n  citation: {s['citation']}")
            print(f"  language: {s['language']}")
            print(f"  regex_hits: {s['regex_hits']}")
            print(f"  text: {s['text_preview']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
