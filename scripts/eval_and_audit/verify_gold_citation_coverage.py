#!/usr/bin/env python
"""Verify train/val gold citations against extracted citation patterns and graph."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from extract_citation_graph import classify_source_citation, extract_references, squash_ws


DATASETS = ("train", "val")
EXTRACTED_DATASETS = ("laws_de", "court_considerations")


def split_gold(value: str) -> list[str]:
    return [squash_ws(part) for part in (value or "").split(";") if squash_ws(part)]


def collect_gold(data_dir: Path, dataset: str) -> tuple[list[dict], Counter]:
    rows: list[dict] = []
    counts: Counter = Counter()
    path = data_dir / f"{dataset}.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            query_id = row.get("query_id", "")
            for citation in split_gold(row.get("gold_citations", "")):
                rows.append({"dataset": dataset, "query_id": query_id, "citation": citation})
                counts[citation] += 1
    return rows, counts


def classify_gold(citation: str) -> dict:
    parsed = classify_source_citation(citation)
    if parsed.pattern != "unknown":
        return {
            "covered": True,
            "family": parsed.family,
            "subfamily": parsed.subfamily,
            "pattern": parsed.pattern,
            "segments": parsed.segments,
            "matched_by": "classify_source_citation",
        }

    refs = extract_references(citation)
    exact = [ref for ref in refs if ref.citation == citation and ref.pattern != "unknown"]
    if exact:
        ref = exact[0]
        return {
            "covered": True,
            "family": ref.family,
            "subfamily": ref.subfamily,
            "pattern": ref.pattern,
            "segments": ref.segments,
            "matched_by": "extract_references_exact",
        }

    return {
        "covered": False,
        "family": parsed.family,
        "subfamily": parsed.subfamily,
        "pattern": parsed.pattern,
        "segments": parsed.segments,
        "matched_by": None,
    }


def lookup_origins(conn: sqlite3.Connection, citation: str) -> dict:
    origins = {
        dataset: {"present": False, "source_column": False, "text_reference": False, "origin_mask": 0}
        for dataset in EXTRACTED_DATASETS
    }
    for dataset in EXTRACTED_DATASETS:
        row = conn.execute(
            "SELECT origin_mask FROM citations WHERE dataset = ? AND citation = ?",
            (dataset, citation),
        ).fetchone()
        if row:
            origin_mask = row[0]
            origins[dataset] = {
                "present": True,
                "source_column": bool(origin_mask & 1),
                "text_reference": bool(origin_mask & 2),
                "origin_mask": origin_mask,
            }
    return origins


def summarize(records: list[dict], unique_citations: set[str]) -> dict:
    unique_records = {record["citation"]: record for record in records}
    pattern_counts = Counter(record["pattern"] for record in unique_records.values())
    family_counts = Counter(record["family"] for record in unique_records.values())
    not_pattern_covered = sorted(
        citation for citation, record in unique_records.items() if not record["pattern_covered"]
    )
    not_in_any_extracted = sorted(
        citation for citation, record in unique_records.items() if not record["present_in_any_extracted"]
    )
    text_only_any = sorted(
        citation
        for citation, record in unique_records.items()
        if record["present_as_text_reference_any"] and not record["present_in_source_column_any"]
    )
    text_any_not_source_any = sorted(
        citation
        for citation, record in unique_records.items()
        if record["present_as_text_reference_any"] and not record["present_in_source_column_any"]
    )
    return {
        "gold_mentions": len(records),
        "unique_gold_citations": len(unique_citations),
        "pattern_covered_unique": len(unique_citations) - len(not_pattern_covered),
        "pattern_uncovered_unique": len(not_pattern_covered),
        "present_in_any_extracted_unique": len(unique_citations) - len(not_in_any_extracted),
        "not_present_in_any_extracted_unique": len(not_in_any_extracted),
        "text_reference_only_unique": len(text_only_any),
        "text_reference_not_source_column_unique": len(text_any_not_source_any),
        "family_counts_unique": dict(family_counts),
        "pattern_counts_unique": dict(pattern_counts),
        "pattern_uncovered": not_pattern_covered,
        "not_present_in_any_extracted": not_in_any_extracted,
        "text_reference_only": text_only_any,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--insights-dir", type=Path, default=Path("data_insights"))
    parser.add_argument("--db", type=Path, default=Path("data_insights/citation_graph_extracted.sqlite"))
    args = parser.parse_args(argv)

    args.insights_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)

    all_records: list[dict] = []
    per_dataset_records: dict[str, list[dict]] = {}
    per_dataset_unique: dict[str, set[str]] = {}
    per_dataset_counts: dict[str, Counter] = {}

    for dataset in DATASETS:
        rows, counts = collect_gold(args.data_dir, dataset)
        per_dataset_counts[dataset] = counts
        per_dataset_unique[dataset] = set(counts)
        enriched_rows: list[dict] = []
        for row in rows:
            citation = row["citation"]
            classification = classify_gold(citation)
            origins = lookup_origins(conn, citation)
            present_in_source_column_any = any(origin["source_column"] for origin in origins.values())
            present_as_text_reference_any = any(origin["text_reference"] for origin in origins.values())
            present_in_any_extracted = any(origin["present"] for origin in origins.values())
            record = {
                **row,
                "mention_count_in_split": counts[citation],
                "pattern_covered": classification["covered"],
                "family": classification["family"],
                "subfamily": classification["subfamily"],
                "pattern": classification["pattern"],
                "matched_by": classification["matched_by"],
                "segments": classification["segments"],
                "origins": origins,
                "present_in_source_column_any": present_in_source_column_any,
                "present_as_text_reference_any": present_as_text_reference_any,
                "present_in_any_extracted": present_in_any_extracted,
                "text_reference_only_any": present_as_text_reference_any and not present_in_source_column_any,
            }
            enriched_rows.append(record)
            all_records.append(record)
        per_dataset_records[dataset] = enriched_rows

    unique_all = set().union(*per_dataset_unique.values())
    unique_records_by_citation = {}
    for record in all_records:
        unique_records_by_citation.setdefault(record["citation"], record)

    summary = {
        "db": str(args.db),
        "splits": {
            dataset: summarize(per_dataset_records[dataset], per_dataset_unique[dataset])
            for dataset in DATASETS
        },
        "combined": summarize(all_records, unique_all),
    }

    details = {
        "summary": summary,
        "unique_gold_citations": {
            dataset: sorted(per_dataset_unique[dataset])
            for dataset in DATASETS
        },
        "records": all_records,
    }

    for dataset in DATASETS:
        (args.insights_dir / f"{dataset}_gold_citations.json").write_text(
            json.dumps(sorted(per_dataset_unique[dataset]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (args.insights_dir / "gold_citation_coverage.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with (args.insights_dir / "gold_citation_coverage.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "split",
            "citation",
            "mention_count_in_split",
            "pattern_covered",
            "family",
            "subfamily",
            "pattern",
            "present_in_laws_de_source",
            "present_in_laws_de_text",
            "present_in_court_source",
            "present_in_court_text",
            "present_in_source_column_any",
            "present_as_text_reference_any",
            "present_in_any_extracted",
            "text_reference_only_any",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        seen_split_citation = set()
        for record in all_records:
            key = (record["dataset"], record["citation"])
            if key in seen_split_citation:
                continue
            seen_split_citation.add(key)
            writer.writerow(
                {
                    "split": record["dataset"],
                    "citation": record["citation"],
                    "mention_count_in_split": record["mention_count_in_split"],
                    "pattern_covered": record["pattern_covered"],
                    "family": record["family"],
                    "subfamily": record["subfamily"],
                    "pattern": record["pattern"],
                    "present_in_laws_de_source": record["origins"]["laws_de"]["source_column"],
                    "present_in_laws_de_text": record["origins"]["laws_de"]["text_reference"],
                    "present_in_court_source": record["origins"]["court_considerations"]["source_column"],
                    "present_in_court_text": record["origins"]["court_considerations"]["text_reference"],
                    "present_in_source_column_any": record["present_in_source_column_any"],
                    "present_as_text_reference_any": record["present_as_text_reference_any"],
                    "present_in_any_extracted": record["present_in_any_extracted"],
                    "text_reference_only_any": record["text_reference_only_any"],
                }
            )

    lines = [
        "# Gold citation coverage",
        "",
        "## Summary",
        "",
        "| Split | Gold mentions | Unique gold | Pattern covered | Pattern uncovered | Present in extracted graph | Missing from extracted graph | Text-reference only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        item = summary["splits"][dataset]
        lines.append(
            f"| {dataset} | {item['gold_mentions']:,} | {item['unique_gold_citations']:,} | "
            f"{item['pattern_covered_unique']:,} | {item['pattern_uncovered_unique']:,} | "
            f"{item['present_in_any_extracted_unique']:,} | {item['not_present_in_any_extracted_unique']:,} | "
            f"{item['text_reference_only_unique']:,} |"
        )
    item = summary["combined"]
    lines.append(
        f"| combined | {item['gold_mentions']:,} | {item['unique_gold_citations']:,} | "
        f"{item['pattern_covered_unique']:,} | {item['pattern_uncovered_unique']:,} | "
        f"{item['present_in_any_extracted_unique']:,} | {item['not_present_in_any_extracted_unique']:,} | "
        f"{item['text_reference_only_unique']:,} |"
    )

    lines.extend(["", "## Pattern Gaps", ""])
    gaps = summary["combined"]["pattern_uncovered"]
    lines.append("None." if not gaps else "\n".join(f"- `{citation}`" for citation in gaps))

    lines.extend(["", "## Gold Citations Missing From Extracted Graph", ""])
    missing = summary["combined"]["not_present_in_any_extracted"]
    lines.append("None." if not missing else "\n".join(f"- `{citation}`" for citation in missing))

    lines.extend(["", "## Gold Citations Found Only As Text References", ""])
    text_only = summary["combined"]["text_reference_only"]
    lines.append("None." if not text_only else "\n".join(f"- `{citation}`" for citation in text_only))

    (args.insights_dir / "gold_citation_coverage.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    compact_summary = {
        key: {
            "gold_mentions": value["gold_mentions"],
            "unique_gold_citations": value["unique_gold_citations"],
            "pattern_uncovered_unique": value["pattern_uncovered_unique"],
            "not_present_in_any_extracted_unique": value["not_present_in_any_extracted_unique"],
            "text_reference_only_unique": value["text_reference_only_unique"],
        }
        for key, value in {**summary["splits"], "combined": summary["combined"]}.items()
    }
    print(json.dumps(compact_summary, ensure_ascii=False, indent=2, sort_keys=True))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
