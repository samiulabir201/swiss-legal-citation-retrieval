#!/usr/bin/env python3
"""Detect and repair UTF-8 mojibake in the law LLM enrichment JSONL.

Audit context
-------------
The retrieval audit reported `terms_de_to_en[].de` values like
`Aufsichtsbeh�rde`, `Pers�nlichkeitstests`, `Bundesf�hrerin` (U+FFFD
replacement chars) in
`law_json_llm_output/law_llm_descriptors_0000000_all.jsonl`. This
allegedly drove `terms_grounded_pct` from ~1.0 down to ~0.901.

What this script does
---------------------
1. Streams the JSONL line-by-line, decoding bytes as UTF-8.
   - We try strict UTF-8 first. If that fails on any line we re-decode
     with errors='replace' and flag the line as "bytes-corrupt".
   - We then walk the parsed JSON looking for U+FFFD, the classic
     "Ã¤/Ã¶/Ã¼/ÃŸ" mojibake pattern, and stray "?" between letters
     in fields where a `?` would never appear normally
     (`terms_de_to_en[].de`, `concepts_en`, `defined_terms[].term`,
     `english_summary`, `legal_rule`).
2. For each affected record we look up its source row in
   `data/laws_de.csv` via `_source_row` and try to recover the
   original German word by substring-matching the corrupted token
   (replacing each U+FFFD / `?` with one of `ä ö ü Ä Ö Ü ß`) against
   the source `text` and `title` columns.
3. Records that can be fully repaired are emitted with the corrected
   strings; records that can't are emitted unchanged with an
   `_utf8_repair` annotation listing the unrecoverable tokens.
4. We only write the `*.cleaned.jsonl` output if at least one record
   needed repair. Otherwise we print "INPUT LOOKS CLEAN" and exit 0.

Usage
-----
    python scripts/repair_law_enrichment_utf8.py \
        --jsonl law_json_llm_output/law_llm_descriptors_0000000_all.jsonl \
        --laws-csv data/laws_de.csv \
        --output law_json_llm_output/law_llm_descriptors_0000000_all.cleaned.jsonl \
        --report law_json_llm_output/law_llm_descriptors_0000000_all.repair_report.json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Detection heuristics
# ---------------------------------------------------------------------------

REPLACEMENT_CHAR = "�"

# Common UTF-8-decoded-as-cp1252 mojibake substrings:
#   ä -> Ã¤, ö -> Ã¶, ü -> Ã¼, ß -> ÃŸ, Ä -> Ã„, Ö -> Ã–, Ü -> Ãœ
MOJIBAKE_PATTERNS = ("Ã¤", "Ã¶", "Ã¼", "ÃŸ", "Ã„", "Ã–", "Ãœ", "ï¿½")

# Candidate replacements for a single corrupted byte position
UMLAUT_CANDIDATES = ("ä", "ö", "ü", "Ä", "Ö", "Ü", "ß", "é", "è", "à", "ç")

# Fields where a `?` is suspicious (terms / concepts / single-token slots)
TOKEN_FIELDS_PATH = (
    ("llm_enrichment", "terms_de_to_en", "*", "de"),
    ("llm_enrichment", "concepts_en", "*"),
    ("llm_enrichment", "defined_terms", "*", "term"),
)
# Fields where U+FFFD is suspicious but `?` is legitimate (free text)
TEXT_FIELDS_PATH = (
    ("llm_enrichment", "english_summary"),
    ("llm_enrichment", "legal_rule"),
    ("llm_enrichment", "legal_question"),
    ("llm_enrichment", "applicability_conditions", "*"),
    ("llm_enrichment", "exceptions_or_limitations", "*"),
    ("llm_enrichment", "addressees", "*"),
    ("llm_enrichment", "sanctions_or_consequences", "*"),
)


def is_token_corrupt(value: str) -> bool:
    """Token-style field: `?` between letters or U+FFFD or mojibake."""
    if not isinstance(value, str):
        return False
    if REPLACEMENT_CHAR in value:
        return True
    if any(p in value for p in MOJIBAKE_PATTERNS):
        return True
    # `?` flanked by letters (e.g. "Lug?" at end is also suspicious if it's a
    # standalone token, no trailing word context).
    if re.search(r"[A-Za-zäöüÄÖÜß]\?[A-Za-zäöüÄÖÜß]", value):
        return True
    return False


def is_text_corrupt(value: str) -> bool:
    """Free-text field: only U+FFFD and mojibake patterns count."""
    if not isinstance(value, str):
        return False
    if REPLACEMENT_CHAR in value:
        return True
    if any(p in value for p in MOJIBAKE_PATTERNS):
        return True
    return False


# ---------------------------------------------------------------------------
# Repair helpers
# ---------------------------------------------------------------------------

def fix_mojibake_pair_encoded(s: str) -> str:
    """If the string was UTF-8 mis-decoded as cp1252, round-trip it back."""
    try:
        return s.encode("cp1252", errors="strict").decode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def candidate_repairs(corrupt: str) -> List[str]:
    """Return possible repaired strings by substituting each U+FFFD / `?`
    with each umlaut candidate (Cartesian product, capped)."""
    positions = [i for i, c in enumerate(corrupt) if c == REPLACEMENT_CHAR or c == "?"]
    if not positions or len(positions) > 3:
        return []
    out: List[str] = [corrupt]
    for pos in positions:
        new_out: List[str] = []
        for cand in UMLAUT_CANDIDATES:
            for s in out:
                new_out.append(s[:pos] + cand + s[pos + 1 :])
        out = new_out
    return out


def repair_against_source(corrupt: str, source_text: str) -> Optional[str]:
    """Try to recover the original German term by matching against source.

    1. Try the pair-encoded round-trip first.
    2. Otherwise enumerate umlaut substitutions and pick the one that
       appears as a substring of `source_text`.
    """
    fixed = fix_mojibake_pair_encoded(corrupt)
    if fixed != corrupt and fixed in source_text:
        return fixed
    for cand in candidate_repairs(corrupt):
        if cand != corrupt and cand in source_text:
            return cand
    # Fallback: case-insensitive substring search
    lo_src = source_text.lower()
    for cand in candidate_repairs(corrupt):
        if cand != corrupt and cand.lower() in lo_src:
            return cand
    return None


# ---------------------------------------------------------------------------
# Field walker
# ---------------------------------------------------------------------------

def walk_token_fields(rec: Dict[str, Any]) -> Iterable[Tuple[List[Any], str]]:
    """Yield (path_list, value) for every token-style string field."""
    enrichment = rec.get("llm_enrichment") or {}
    for term in enrichment.get("terms_de_to_en") or []:
        if isinstance(term, dict) and isinstance(term.get("de"), str):
            yield (["llm_enrichment", "terms_de_to_en", term, "de"], term["de"])
    for i, c in enumerate(enrichment.get("concepts_en") or []):
        if isinstance(c, str):
            yield (["llm_enrichment", "concepts_en", i], c)
    for dt in enrichment.get("defined_terms") or []:
        if isinstance(dt, dict) and isinstance(dt.get("term"), str):
            yield (["llm_enrichment", "defined_terms", dt, "term"], dt["term"])


def walk_text_fields(rec: Dict[str, Any]) -> Iterable[Tuple[List[Any], str]]:
    enrichment = rec.get("llm_enrichment") or {}
    for fname in ("english_summary", "legal_rule", "legal_question"):
        v = enrichment.get(fname)
        if isinstance(v, str):
            yield (["llm_enrichment", fname], v)
    for fname in (
        "applicability_conditions",
        "exceptions_or_limitations",
        "addressees",
        "sanctions_or_consequences",
    ):
        for i, v in enumerate(enrichment.get(fname) or []):
            if isinstance(v, str):
                yield (["llm_enrichment", fname, i], v)


def set_field(rec: Dict[str, Any], path: List[Any], new_value: str) -> None:
    """Set a field given a path produced by walk_*; supports dict-by-ref
    list elements (path contains the dict object itself)."""
    enrichment = rec["llm_enrichment"]
    # path = ["llm_enrichment", "terms_de_to_en", <dict>, "de"]
    # path = ["llm_enrichment", "concepts_en", <int>]
    # path = ["llm_enrichment", "english_summary"]
    if path[1] in ("terms_de_to_en", "defined_terms"):
        # the third element is the dict-by-reference
        path[2][path[3]] = new_value
    elif path[1] in (
        "concepts_en",
        "applicability_conditions",
        "exceptions_or_limitations",
        "addressees",
        "sanctions_or_consequences",
    ):
        enrichment[path[1]][path[2]] = new_value
    else:
        enrichment[path[1]] = new_value


# ---------------------------------------------------------------------------
# laws_de.csv loader (lazy / on demand)
# ---------------------------------------------------------------------------

class LawSource:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._rows: Optional[List[Dict[str, str]]] = None

    def _ensure_loaded(self) -> None:
        if self._rows is not None:
            return
        sys.stderr.write(f"[info] loading {self.csv_path} ...\n")
        with open(self.csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            self._rows = list(reader)
        sys.stderr.write(f"[info] loaded {len(self._rows):,} source law rows\n")

    def get_text(self, source_row: int) -> str:
        self._ensure_loaded()
        if 0 <= source_row < len(self._rows):
            r = self._rows[source_row]
            return f"{r.get('title','')}\n{r.get('text','')}"
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--laws-csv", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--max-records", type=int, default=None,
                    help="optional cap for dry-runs")
    args = ap.parse_args()

    laws = LawSource(args.laws_csv)

    # Pre-scan: read raw bytes per line so we surface invalid-UTF-8 lines too.
    n_lines = 0
    n_invalid_bytes = 0
    n_recs_dirty = 0
    n_tokens_dirty = 0
    n_tokens_repaired = 0
    n_text_dirty = 0
    n_text_repaired = 0
    sample_unrecoverable: List[Dict[str, Any]] = []
    pattern_counter: Counter = Counter()

    out_lines: List[str] = []
    needs_writing = False

    with open(args.jsonl, "rb") as fh:
        for raw in fh:
            n_lines += 1
            if args.max_records is not None and n_lines > args.max_records:
                break
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                n_invalid_bytes += 1
                line = raw.decode("utf-8", errors="replace")

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Keep the line as-is; nothing we can do here.
                out_lines.append(line.rstrip("\n"))
                continue

            rec_dirty = False
            unrecoverable_here: List[str] = []
            source_row = rec.get("_source_row")
            source_text: Optional[str] = None

            # token-style fields ---------------------------------------------
            for path, val in list(walk_token_fields(rec)):
                if not is_token_corrupt(val):
                    continue
                n_tokens_dirty += 1
                rec_dirty = True
                if REPLACEMENT_CHAR in val:
                    pattern_counter["U+FFFD"] += 1
                if any(p in val for p in MOJIBAKE_PATTERNS):
                    pattern_counter["cp1252-mojibake"] += 1
                if source_text is None and isinstance(source_row, int):
                    source_text = laws.get_text(source_row)
                fixed = (
                    repair_against_source(val, source_text or "")
                    if source_text else None
                )
                # Even with no source, try the pair-encoding fix:
                if fixed is None:
                    pe = fix_mojibake_pair_encoded(val)
                    if pe != val and not is_token_corrupt(pe):
                        fixed = pe
                if fixed is not None:
                    set_field(rec, path, fixed)
                    n_tokens_repaired += 1
                else:
                    unrecoverable_here.append(val)

            # free-text fields -----------------------------------------------
            for path, val in list(walk_text_fields(rec)):
                if not is_text_corrupt(val):
                    continue
                n_text_dirty += 1
                rec_dirty = True
                if REPLACEMENT_CHAR in val:
                    pattern_counter["U+FFFD"] += 1
                if any(p in val for p in MOJIBAKE_PATTERNS):
                    pattern_counter["cp1252-mojibake"] += 1
                pe = fix_mojibake_pair_encoded(val)
                if pe != val and not is_text_corrupt(pe):
                    set_field(rec, path, pe)
                    n_text_repaired += 1
                else:
                    unrecoverable_here.append(val[:80])

            if rec_dirty:
                n_recs_dirty += 1
                if unrecoverable_here:
                    rec.setdefault("_utf8_repair", {})["unrecoverable"] = (
                        unrecoverable_here
                    )
                    if len(sample_unrecoverable) < 25:
                        sample_unrecoverable.append({
                            "_source_row": source_row,
                            "citation": rec.get("citation"),
                            "tokens": unrecoverable_here,
                        })
                else:
                    rec.setdefault("_utf8_repair", {})["status"] = "fully_repaired"
                needs_writing = True

            out_lines.append(json.dumps(rec, ensure_ascii=False))

    # ---------- report ---------------------------------------------------
    report = {
        "input": str(args.jsonl),
        "lines_total": n_lines,
        "lines_with_invalid_utf8_bytes": n_invalid_bytes,
        "records_with_mojibake": n_recs_dirty,
        "token_field_hits": n_tokens_dirty,
        "token_field_repaired": n_tokens_repaired,
        "text_field_hits": n_text_dirty,
        "text_field_repaired": n_text_repaired,
        "pattern_counts": dict(pattern_counter),
        "sample_unrecoverable": sample_unrecoverable,
        "output_written": bool(needs_writing),
        "output_path": str(args.output) if needs_writing else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    if needs_writing:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
            for ln in out_lines:
                fh.write(ln)
                fh.write("\n")
        print(
            f"[done] wrote {args.output} ({n_recs_dirty:,} records repaired, "
            f"{n_tokens_repaired}/{n_tokens_dirty} token-fields fixed, "
            f"{n_text_repaired}/{n_text_dirty} text-fields fixed)"
        )
    else:
        print(
            f"INPUT LOOKS CLEAN: scanned {n_lines:,} records, "
            f"0 mojibake hits, 0 invalid-UTF-8 lines. "
            "No cleaned file written."
        )
    print(f"[report] {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
