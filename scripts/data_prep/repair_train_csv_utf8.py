#!/usr/bin/env python3
"""Detect and repair UTF-8 mojibake in data/train.csv.

Audit context
-------------
The audit reported `train_0306` gold contains `Art. 5 Lug?` where the
correct citation is the Lugano Convention (`LugÜ`).

Strategy
--------
1. Read train.csv as raw bytes; surface invalid UTF-8 sequences.
2. After UTF-8 decode, scan the `gold_citations` column for:
   - U+FFFD replacement characters
   - cp1252-mojibake patterns (`Ã¼` etc.)
   - a `?` directly inside a citation token (e.g. `Lug?`, `OR?`,
     `URG?`) — citation strings should never contain a literal `?`.
3. For each suspicious token we try, in order:
   a. Round-trip pair-encoding fix (cp1252 -> utf-8).
   b. Substitution from a fixed dictionary of known law abbreviations
      with diacritics (`LugÜ`, `URG`, `OR`, `ZGB`, …). For Swiss law
      `Ü` is by far the most common umlaut in citation tokens.
   c. Brute-force candidate substitution (`ä ö ü Ä Ö Ü ß`) cross-
      checked against the union of all known citation tokens
      anywhere else in the corpus (train + test + val + laws_de).
4. Only writes `data/train_repaired.csv` if at least one repair was
   needed; otherwise prints "INPUT LOOKS CLEAN".

Usage
-----
    python scripts/repair_train_csv_utf8.py \
        --train data/train.csv \
        --output data/train_repaired.csv \
        --report data/train_repaired.report.json \
        --extra-corpora data/laws_de.csv data/test.csv data/val.csv
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
from typing import Iterable, List, Optional, Set

REPLACEMENT_CHAR = "�"
MOJIBAKE_PATTERNS = ("Ã¤", "Ã¶", "Ã¼", "ÃŸ", "Ã„", "Ã–", "Ãœ", "ï¿½")
UMLAUT_CANDIDATES = ("ä", "ö", "ü", "Ä", "Ö", "Ü", "ß")

# Hand-curated corrections for Swiss-law citation abbreviations whose
# canonical form contains a diacritic.  These are applied as last-mile
# token rewrites if substring match fails.
KNOWN_TOKEN_FIXES = {
    "Lug?": "LugÜ",
    "Lug�": "LugÜ",
    # Add more if the corpus surfaces them; everything else falls
    # through to the corpus-driven repair.
}

CITATION_SPLIT = re.compile(r"\s*;\s*")


def is_token_corrupt(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if REPLACEMENT_CHAR in value:
        return True
    if any(p in value for p in MOJIBAKE_PATTERNS):
        return True
    # A `?` inside a citation is suspicious (citations are never questions).
    if "?" in value:
        return True
    return False


def fix_mojibake_pair_encoded(s: str) -> str:
    try:
        return s.encode("cp1252", errors="strict").decode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def candidate_repairs(corrupt: str) -> List[str]:
    positions = [
        i for i, c in enumerate(corrupt)
        if c == REPLACEMENT_CHAR or c == "?"
    ]
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


def collect_known_tokens(paths: Iterable[Path]) -> Set[str]:
    """Build a vocabulary of clean citation-shaped tokens from the
    other CSVs so we can match against them."""
    vocab: Set[str] = set()
    abbr_re = re.compile(r"\b[A-ZÄÖÜa-zäöüß][A-ZÄÖÜa-zäöüß0-9.]{1,20}")
    for p in paths:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                for cell in row:
                    if not cell:
                        continue
                    for tok in abbr_re.findall(cell):
                        # citation-y tokens: Latin + maybe digits, contains
                        # at least one umlaut to be useful as a repair target
                        if any(ch in tok for ch in "ÄÖÜäöüß"):
                            vocab.add(tok)
    return vocab


def repair_token(token: str, vocab: Set[str]) -> Optional[str]:
    if token in KNOWN_TOKEN_FIXES:
        return KNOWN_TOKEN_FIXES[token]
    pe = fix_mojibake_pair_encoded(token)
    if pe != token and not is_token_corrupt(pe):
        return pe
    for cand in candidate_repairs(token):
        if cand != token and cand in vocab:
            return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--extra-corpora", nargs="*", type=Path, default=[])
    args = ap.parse_args()

    # Surface invalid-UTF-8 byte runs at the file level
    n_invalid_byte_lines = 0
    with open(args.train, "rb") as fh:
        for raw in fh:
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                n_invalid_byte_lines += 1

    vocab = collect_known_tokens([args.train, *args.extra_corpora])
    sys.stderr.write(f"[info] vocab of umlaut-containing tokens: {len(vocab):,}\n")

    n_rows = 0
    n_rows_dirty = 0
    n_tokens_dirty = 0
    n_tokens_repaired = 0
    pattern_counter: Counter = Counter()
    sample_unrecoverable: List[dict] = []

    out_rows: List[dict] = []
    needs_writing = False

    with open(args.train, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        for row in reader:
            n_rows += 1
            gold = row.get("gold_citations", "") or ""
            if not is_token_corrupt(gold):
                out_rows.append(row)
                continue

            tokens = CITATION_SPLIT.split(gold)
            new_tokens: List[str] = []
            row_repairs: List[dict] = []
            row_dirty = False
            for tok in tokens:
                if not is_token_corrupt(tok):
                    new_tokens.append(tok)
                    continue
                row_dirty = True
                n_tokens_dirty += 1
                if REPLACEMENT_CHAR in tok:
                    pattern_counter["U+FFFD"] += 1
                if any(p in tok for p in MOJIBAKE_PATTERNS):
                    pattern_counter["cp1252-mojibake"] += 1
                if "?" in tok and REPLACEMENT_CHAR not in tok:
                    pattern_counter["question-mark-in-citation"] += 1
                fixed = repair_token(tok, vocab)
                if fixed is not None:
                    new_tokens.append(fixed)
                    n_tokens_repaired += 1
                    row_repairs.append({"from": tok, "to": fixed})
                else:
                    new_tokens.append(tok)
                    row_repairs.append({"from": tok, "to": None})
                    if len(sample_unrecoverable) < 25:
                        sample_unrecoverable.append({
                            "query_id": row.get("query_id"),
                            "token": tok,
                        })
            if row_dirty:
                n_rows_dirty += 1
                row["gold_citations"] = ";".join(new_tokens)
                needs_writing = True
            out_rows.append(row)

    report = {
        "input": str(args.train),
        "rows_total": n_rows,
        "lines_with_invalid_utf8_bytes": n_invalid_byte_lines,
        "rows_with_mojibake_in_gold_citations": n_rows_dirty,
        "tokens_with_mojibake": n_tokens_dirty,
        "tokens_repaired": n_tokens_repaired,
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
        with open(args.output, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in out_rows:
                writer.writerow(r)
        print(
            f"[done] wrote {args.output} ({n_rows_dirty} rows repaired, "
            f"{n_tokens_repaired}/{n_tokens_dirty} tokens fixed)"
        )
    else:
        print(
            f"INPUT LOOKS CLEAN: scanned {n_rows} rows, "
            "0 mojibake hits in gold_citations. No repaired file written."
        )
    print(f"[report] {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
