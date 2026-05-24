"""Check coverage of laws_de.csv in the main descriptor file.

Compare:
1. Row count of laws_de.csv (data records, excluding header).
2. Row count of law_llm_input.jsonl (the slim per-row payload built from the
   CSV; the builder typically filters out rows with missing/empty text).
3. Row count of law_llm_descriptors_0000000_all.jsonl.

Then check what's missing: rows present in the CSV but not in the
descriptor file, broken down by likely cause (empty text, etc.).
"""
import csv
import json
import sys

CSV_PATH = r"E:\swiss_citation_extraction\data\laws_de.csv"
INPUT_JSONL = r"E:\swiss_citation_extraction\artifacts\law_llm_input.jsonl"
MAIN = r"E:\swiss_citation_extraction\law_json_llm_output\law_llm_descriptors_0000000_all.jsonl"


def csv_size_in_bytes(path: str) -> int:
    import os
    return os.path.getsize(path)


def count_csv_rows():
    """Count CSV rows; classify each by empty-text / short-text reasons."""
    csv.field_size_limit(sys.maxsize)
    n = 0
    n_empty_citation = 0
    n_empty_text = 0
    n_short_text = 0  # < 20 chars (matches notebook's min_text_chars default)
    n_with_text = 0
    examples_empty_text = []
    with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            n += 1
            citation = (row.get("citation") or "").strip()
            text = (row.get("text") or "").strip()
            if not citation:
                n_empty_citation += 1
                continue
            if not text:
                n_empty_text += 1
                if len(examples_empty_text) < 5:
                    examples_empty_text.append((row_idx, citation))
                continue
            if len(text) < 20:
                n_short_text += 1
                continue
            n_with_text += 1
    return {
        "rows": n,
        "empty_citation": n_empty_citation,
        "empty_text": n_empty_text,
        "short_text": n_short_text,
        "with_text": n_with_text,
        "examples_empty_text": examples_empty_text,
    }


def collect_source_rows(path: str):
    rows = set()
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sr = obj.get("_source_row")
            if sr is not None:
                rows.add(int(sr))
    return n, rows


def main():
    print(f"CSV size on disk : {csv_size_in_bytes(CSV_PATH) / 1024**2:.1f} MB")
    print(f"input.jsonl size : {csv_size_in_bytes(INPUT_JSONL) / 1024**2:.1f} MB")
    print(f"main.jsonl size  : {csv_size_in_bytes(MAIN) / 1024**2:.1f} MB")

    print("\nScanning laws_de.csv ...")
    csv_stats = count_csv_rows()
    print(f"  total CSV rows         : {csv_stats['rows']:,}")
    print(f"  with non-empty text    : {csv_stats['with_text']:,}")
    print(f"  empty citation         : {csv_stats['empty_citation']:,}")
    print(f"  empty text             : {csv_stats['empty_text']:,}")
    print(f"  short text (<20 chars) : {csv_stats['short_text']:,}")
    if csv_stats["examples_empty_text"]:
        print(f"  example empty-text rows: {csv_stats['examples_empty_text']}")

    print("\nScanning law_llm_input.jsonl ...")
    n_input, input_rows = collect_source_rows(INPUT_JSONL)
    print(f"  rows: {n_input:,}; unique _source_row: {len(input_rows):,}")

    print("\nScanning main descriptor jsonl ...")
    n_main, main_rows = collect_source_rows(MAIN)
    print(f"  rows: {n_main:,}; unique _source_row: {len(main_rows):,}")

    only_input = input_rows - main_rows
    only_main = main_rows - input_rows
    print(f"\n_source_row in input but missing from main : {len(only_input):,}")
    print(f"_source_row in main but not in input        : {len(only_main):,}")
    if only_input:
        print(f"  sample: {sorted(only_input)[:10]}")
    if only_main:
        print(f"  sample: {sorted(only_main)[:10]}")

    csv_data_rows = csv_stats["rows"]
    print("\nCoverage summary:")
    print(f"  CSV data rows                    : {csv_data_rows:,}")
    print(f"  Eligible for LLM (with text>=20) : {csv_stats['with_text']:,}")
    print(f"  In law_llm_input.jsonl           : {n_input:,}")
    print(f"  In main descriptor file          : {n_main:,}")
    if csv_data_rows:
        eligible_pct = csv_stats['with_text'] / csv_data_rows * 100
        covered_pct = n_main / csv_stats['with_text'] * 100 if csv_stats['with_text'] else 0
        print(f"  Eligible / total (CSV)           : {eligible_pct:.2f}%")
        print(f"  Covered (main / eligible)        : {covered_pct:.2f}%")


if __name__ == "__main__":
    main()
