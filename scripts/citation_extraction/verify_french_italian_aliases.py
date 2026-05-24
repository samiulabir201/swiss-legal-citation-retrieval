"""Verify French/Italian alias coverage in extract_citation_graph.

This is a verification harness, not part of the production pipeline.

It samples French and Italian rows from data/court_considerations.csv
(detected by the presence of language-specific markers `c.` / `consid.`
since the CSV has no language column), runs `extract_references` on them,
and prints the citations that hit the new aliases (ATF, DTF, c., consid.,
cons., space-separated docket).

It also prints before/after extraction counts on a 1000-row sample by
running both the old regex set and the new regex set side-by-side.
Run from the project root:

    python scripts/verify_french_italian_aliases.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_citation_graph import (  # noqa: E402
    extract_references,
    extract_bge,
    extract_cases,
    BGE_RE,
    MODERN_CASE_RE,
    CONSIDERATION_RE,
    LOOSE_CONSIDERATION_RE,
)


# Old-version regexes for the before/after diff (copied verbatim from the
# pre-patch script so we can compare extraction counts on the same sample).
OLD_BGE_RE = re.compile(
    r"\bBGE\s+"
    r"(?P<volume>\d{3})\s+"
    r"(?P<division>[IVX]{1,4})\s+"
    r"(?P<page>\d+[a-z]?)"
    r"(?:\s+(?P<unit>E\.|S\.)\s*(?P<pinpoint>[A-Za-zIVX0-9]+(?:[./-][A-Za-z0-9]+)*(?:\s*f{1,2}\.)?))?",
)
OLD_MODERN_CASE_RE = re.compile(
    r"\b(?P<docket>\d{1,2}[A-Z]{1,2}[_\.]\d{1,5}/\d{4})\b"
    r"(?P<trailing>.{0,90})",
    re.DOTALL,
)
OLD_CONSIDERATION_RE = re.compile(
    r"\bE\.\s*(?P<consideration>[A-Za-zIVX0-9]+(?:[./-][A-Za-z0-9]+)*)"
)


def detect_lang(text: str) -> str | None:
    """Heuristic language tag: French if `c.` / `consid.` style French
    cues are dominant, Italian if Italian cues are dominant. Returns
    None for German or undetermined."""
    has_french = bool(re.search(r"\b(?:ATF|consid\.|c\. \d|du \d|art\. \d)\b", text))
    has_italian = bool(re.search(r"\b(?:DTF|consid\. \d|cons\. \d)\b", text))
    has_german_only = bool(re.search(r"\bBGE\b|\bvom \d|\bArt\. \d", text)) and not (
        has_french or has_italian
    )
    if has_italian and not has_french:
        return "it"
    if has_french and not has_italian:
        return "fr"
    if has_german_only:
        return "de"
    return None


def sample_rows(csv_path: Path, want_per_lang: int = 5, scan_limit: int = 200000):
    """Stream the CSV until we have `want_per_lang` rows for each of fr/it/de."""
    buckets: dict[str, list[dict]] = {"fr": [], "it": [], "de": []}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if i >= scan_limit:
                break
            lang = detect_lang(row.get("text") or "")
            if lang is None:
                continue
            if len(buckets[lang]) < want_per_lang:
                buckets[lang].append(row)
            if all(len(b) >= want_per_lang for b in buckets.values()):
                break
    return buckets


def show_lang_samples(buckets: dict) -> None:
    for lang in ("fr", "it", "de"):
        rows = buckets.get(lang, [])
        print(f"\n=== {lang.upper()} samples (n={len(rows)}) ===")
        for n, row in enumerate(rows, 1):
            text = row.get("text") or ""
            refs = extract_references(text)
            bge_refs = [r for r in refs if r.subfamily == "bge"]
            case_refs = [
                r for r in refs if r.pattern == "court_case"
            ]
            print(f"\n[{lang} #{n}] source citation: {row.get('citation', '')!r}")
            snippet = text[:300].replace("\n", " ")
            print(f"  text snippet: {snippet}")
            print(f"  -> {len(bge_refs)} BGE/ATF/DTF, {len(case_refs)} court cases")
            for r in bge_refs[:5]:
                raw = r.segments.get("reporter_raw", r.segments.get("reporter"))
                print(f"     bge: {r.citation}  (reporter_raw={raw})")
            for r in case_refs[:5]:
                drwa = r.segments.get("docket_raw", r.segments.get("docket"))
                print(f"     case: {r.citation}  (docket_raw={drwa})")


def count_old_vs_new(csv_path: Path, n_rows: int = 1000) -> None:
    old_bge = old_case = old_consid = 0
    new_bge = new_case = new_consid = 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if i >= n_rows:
                break
            text = row.get("text") or ""
            old_bge += sum(1 for _ in OLD_BGE_RE.finditer(text))
            old_case += sum(1 for _ in OLD_MODERN_CASE_RE.finditer(text))
            old_consid += sum(1 for _ in OLD_CONSIDERATION_RE.finditer(text))
            new_bge += sum(1 for _ in BGE_RE.finditer(text))
            new_case += sum(1 for _ in MODERN_CASE_RE.finditer(text))
            new_consid += sum(1 for _ in CONSIDERATION_RE.finditer(text))
    print(f"\n=== Before/After raw match counts on first {n_rows} rows ===")
    print(f"BGE-family hits   : old={old_bge:>6}  new={new_bge:>6}  delta=+{new_bge - old_bge}")
    print(f"Modern docket hits: old={old_case:>6}  new={new_case:>6}  delta=+{new_case - old_case}")
    print(f"Consideration hits: old={old_consid:>6}  new={new_consid:>6}  delta=+{new_consid - old_consid}")


def main() -> int:
    csv_path = ROOT / "data" / "court_considerations.csv"
    if not csv_path.exists():
        print(f"Missing CSV: {csv_path}", file=sys.stderr)
        return 1
    print(f"Reading {csv_path}")
    buckets = sample_rows(csv_path)
    show_lang_samples(buckets)
    count_old_vs_new(csv_path, n_rows=1000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
