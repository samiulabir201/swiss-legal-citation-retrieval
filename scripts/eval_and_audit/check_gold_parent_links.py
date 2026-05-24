#!/usr/bin/env python
"""Check query-level parent links for gold citations without their own text row."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from extract_citation_graph import squash_ws


CSV_SPLITS = ("train", "val")
EXTRACTED_DATASETS = ("laws_de", "court_considerations")


def split_gold(value: str) -> list[str]:
    return [squash_ws(part) for part in (value or "").split(";") if squash_ws(part)]


def load_queries(data_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for split in CSV_SPLITS:
        with (data_dir / f"{split}.csv").open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                gold = split_gold(row.get("gold_citations", ""))
                rows.append(
                    {
                        "split": split,
                        "query_id": row.get("query_id", ""),
                        "gold": gold,
                        "gold_set": set(gold),
                    }
                )
    return rows


def lookup_origin_masks(conn: sqlite3.Connection, citations: set[str]) -> dict[str, dict]:
    origin_masks = {
        citation: {dataset: 0 for dataset in EXTRACTED_DATASETS}
        for citation in citations
    }
    for dataset in EXTRACTED_DATASETS:
        for citation in citations:
            row = conn.execute(
                "SELECT origin_mask FROM citations WHERE dataset = ? AND citation = ?",
                (dataset, citation),
            ).fetchone()
            if row:
                origin_masks[citation][dataset] = row[0]
    return origin_masks


def has_own_text(origin_masks: dict[str, int]) -> bool:
    return any(mask & 1 for mask in origin_masks.values())


def appears_as_text_reference(origin_masks: dict[str, int]) -> bool:
    return any(mask & 2 for mask in origin_masks.values())


def load_parent_edges(conn: sqlite3.Connection, targets: set[str]) -> dict[str, set[str]]:
    if not targets:
        return {}
    conn.execute("DROP TABLE IF EXISTS temp_gold_targets")
    conn.execute("CREATE TEMP TABLE temp_gold_targets (citation TEXT PRIMARY KEY)")
    conn.executemany(
        "INSERT INTO temp_gold_targets (citation) VALUES (?)",
        [(citation,) for citation in targets],
    )
    parents: dict[str, set[str]] = defaultdict(set)
    rows = conn.execute(
        """
        SELECT e.target, e.source
        FROM edges e
        JOIN temp_gold_targets t ON t.citation = e.target
        """
    )
    for target, source in rows:
        parents[target].add(source)
    conn.execute("DROP TABLE temp_gold_targets")
    return parents


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--insights-dir", type=Path, default=Path("data_insights"))
    parser.add_argument("--db", type=Path, default=Path("data_insights/citation_graph_extracted.sqlite"))
    args = parser.parse_args(argv)

    args.insights_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)

    query_rows = load_queries(args.data_dir)
    all_gold = {citation for row in query_rows for citation in row["gold_set"]}
    origin_masks = lookup_origin_masks(conn, all_gold)
    no_own_text = {
        citation
        for citation, masks in origin_masks.items()
        if not has_own_text(masks)
    }
    parent_edges = load_parent_edges(conn, no_own_text)

    records: list[dict] = []
    query_summary = Counter()
    unique_by_status: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[dict]] = defaultdict(list)

    for row in query_rows:
        gold_set = row["gold_set"]
        for citation in row["gold"]:
            if citation not in no_own_text:
                continue
            parents = sorted(parent_edges.get(citation, set()))
            parent_gold = sorted(parent for parent in parents if parent in gold_set)
            in_text_reference = appears_as_text_reference(origin_masks[citation])
            status = (
                "parent_in_same_query_gold"
                if parent_gold
                else "has_parents_but_none_in_same_query_gold"
                if parents
                else "no_parent_edges_found"
            )
            record = {
                "split": row["split"],
                "query_id": row["query_id"],
                "citation": citation,
                "has_own_text": False,
                "present_as_text_reference": in_text_reference,
                "parent_count_global": len(parents),
                "parent_in_same_query_gold": bool(parent_gold),
                "same_query_gold_parents": parent_gold,
                "sample_global_parents": parents[:20],
                "status": status,
                "origin_masks": origin_masks[citation],
            }
            records.append(record)
            query_summary[(row["split"], status)] += 1
            query_summary[(row["split"], "total_no_own_text_gold_mentions")] += 1
            unique_by_status[(row["split"], status)].add(citation)
            unique_by_status[(row["split"], "total_no_own_text_gold")].add(citation)
            unique_by_status[("combined", status)].add(citation)
            unique_by_status[("combined", "total_no_own_text_gold")].add(citation)
            if len(examples[(row["split"], status)]) < 10:
                examples[(row["split"], status)].append(record)
            if len(examples[("combined", status)]) < 10:
                examples[("combined", status)].append(record)

    exclusive_unique = {}
    for split in (*CSV_SPLITS, "combined"):
        total = unique_by_status[(split, "total_no_own_text_gold")]
        no_parents = unique_by_status[(split, "no_parent_edges_found")]
        ever_same_query_parent = unique_by_status[(split, "parent_in_same_query_gold")]
        with_parents = total - no_parents
        exclusive_unique[split] = {
            "total_no_own_text_gold": len(total),
            "with_parent_edges": len(with_parents),
            "no_parent_edges_found": len(no_parents),
            "with_parent_edges_ever_same_query_gold_parent": len(ever_same_query_parent),
            "with_parent_edges_never_same_query_gold_parent": len(with_parents - ever_same_query_parent),
        }

    summary = {
        "by_split_mentions": {
            split: {
                "total_no_own_text_gold_mentions": query_summary[(split, "total_no_own_text_gold_mentions")],
                "parent_in_same_query_gold": query_summary[(split, "parent_in_same_query_gold")],
                "has_parents_but_none_in_same_query_gold": query_summary[
                    (split, "has_parents_but_none_in_same_query_gold")
                ],
                "no_parent_edges_found": query_summary[(split, "no_parent_edges_found")],
            }
            for split in CSV_SPLITS
        },
        "by_split_unique": {
            split: {
                "total_no_own_text_gold": len(unique_by_status[(split, "total_no_own_text_gold")]),
                "parent_in_same_query_gold": len(unique_by_status[(split, "parent_in_same_query_gold")]),
                "has_parents_but_none_in_same_query_gold": len(
                    unique_by_status[(split, "has_parents_but_none_in_same_query_gold")]
                ),
                "no_parent_edges_found": len(unique_by_status[(split, "no_parent_edges_found")]),
            }
            for split in CSV_SPLITS
        },
        "exclusive_unique": exclusive_unique,
        "combined_unique": {
            "total_no_own_text_gold": len(unique_by_status[("combined", "total_no_own_text_gold")]),
            "parent_in_same_query_gold": len(unique_by_status[("combined", "parent_in_same_query_gold")]),
            "has_parents_but_none_in_same_query_gold": len(
                unique_by_status[("combined", "has_parents_but_none_in_same_query_gold")]
            ),
            "no_parent_edges_found": len(unique_by_status[("combined", "no_parent_edges_found")]),
        },
        "unique_no_own_text_citations": sorted(no_own_text),
        "unique_no_parent_edges_found": sorted(unique_by_status[("combined", "no_parent_edges_found")]),
        "unique_has_parents_but_none_in_same_query_gold": sorted(
            unique_by_status[("combined", "has_parents_but_none_in_same_query_gold")]
        ),
    }

    output = {
        "summary": summary,
        "records": records,
    }
    (args.insights_dir / "gold_parent_link_check.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with (args.insights_dir / "gold_parent_link_check.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "split",
            "query_id",
            "citation",
            "present_as_text_reference",
            "parent_count_global",
            "parent_in_same_query_gold",
            "same_query_gold_parents",
            "sample_global_parents",
            "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "split": record["split"],
                    "query_id": record["query_id"],
                    "citation": record["citation"],
                    "present_as_text_reference": record["present_as_text_reference"],
                    "parent_count_global": record["parent_count_global"],
                    "parent_in_same_query_gold": record["parent_in_same_query_gold"],
                    "same_query_gold_parents": ";".join(record["same_query_gold_parents"]),
                    "sample_global_parents": ";".join(record["sample_global_parents"]),
                    "status": record["status"],
                }
            )

    lines = [
        "# Gold parent-link check",
        "",
        "A gold citation is counted as having no own text when it does not appear as a `citation` row in either `laws_de.csv` or `court_considerations.csv`.",
        "",
        "## Summary",
        "",
        "| Split | No-own-text gold mentions | Same-query gold parent found | Has parents, none in same-query gold | No parent edge found |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in CSV_SPLITS:
        item = summary["by_split_mentions"][split]
        lines.append(
            f"| {split} | {item['total_no_own_text_gold_mentions']:,} | "
            f"{item['parent_in_same_query_gold']:,} | "
            f"{item['has_parents_but_none_in_same_query_gold']:,} | "
            f"{item['no_parent_edges_found']:,} |"
        )
    lines.extend(
        [
            "",
            "## Exclusive Unique Counts",
            "",
            "| Split | No-own-text unique gold | Has parent edges | Ever has same-query gold parent | Never has same-query gold parent | No parent edge found |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split in (*CSV_SPLITS, "combined"):
        item = summary["exclusive_unique"][split]
        lines.append(
            f"| {split} | {item['total_no_own_text_gold']:,} | "
            f"{item['with_parent_edges']:,} | "
            f"{item['with_parent_edges_ever_same_query_gold_parent']:,} | "
            f"{item['with_parent_edges_never_same_query_gold_parent']:,} | "
            f"{item['no_parent_edges_found']:,} |"
        )

    lines.extend(["", "## Examples Where Parent Is Not In Same Query Gold", ""])
    misses = examples.get(("combined", "has_parents_but_none_in_same_query_gold"), [])
    if misses:
        for record in misses:
            lines.append(
                f"- `{record['split']}:{record['query_id']}` `{record['citation']}` "
                f"parents include: {', '.join(f'`{p}`' for p in record['sample_global_parents'][:5])}"
            )
    else:
        lines.append("None.")

    lines.extend(["", "## Examples With No Parent Edge Found", ""])
    no_parents = examples.get(("combined", "no_parent_edges_found"), [])
    if no_parents:
        for record in no_parents:
            lines.append(f"- `{record['split']}:{record['query_id']}` `{record['citation']}`")
    else:
        lines.append("None.")

    (args.insights_dir / "gold_parent_link_check.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    compact_summary = {
        "by_split_mentions": summary["by_split_mentions"],
        "exclusive_unique": summary["exclusive_unique"],
    }
    print(json.dumps(compact_summary, ensure_ascii=False, indent=2, sort_keys=True))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
