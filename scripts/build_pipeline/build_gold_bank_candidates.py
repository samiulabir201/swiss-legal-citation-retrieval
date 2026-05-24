from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INSIGHTS_DIR = ROOT / "data_insights"
ART_DIR = ROOT / "artifacts"
DB_PATH = INSIGHTS_DIR / "citation_graph_extracted.sqlite"


def split_gold(value: str | float | None) -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def load_json_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")))


def load_source_sets() -> tuple[set[str], set[str]]:
    con = sqlite3.connect(DB_PATH)
    law_source = {
        row[0]
        for row in con.execute(
            "SELECT citation FROM citations WHERE dataset = 'laws_de' AND (origin_mask & 1) != 0"
        )
    }
    court_source = {
        row[0]
        for row in con.execute(
            "SELECT citation FROM citations WHERE dataset = 'court_considerations' AND (origin_mask & 1) != 0"
        )
    }
    con.close()
    return law_source, court_source


def load_known_gold_bank(bank_splits: list[str]) -> set[str]:
    bank = set()
    for split in bank_splits:
        bank |= load_json_list(INSIGHTS_DIR / f"{split}_gold_citations.json")
    return bank


def recall(candidates: set[str], gold: set[str]) -> float | None:
    if not gold:
        return None
    return len(candidates & gold) / len(gold)


def build(split: str, bank_splits: list[str]) -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    law_source, court_source = load_source_sets()
    known_bank = load_known_gold_bank(bank_splits)
    law_candidates = sorted(known_bank & law_source)
    court_candidates = sorted(known_bank & court_source)
    law_set = set(law_candidates)
    court_set = set(court_candidates)

    input_path = DATA_DIR / f"{split}.csv"
    bank_label = "_".join(bank_splits)
    output_path = ART_DIR / f"{split}_gold_bank_{bank_label}_candidate_sets.jsonl"
    summary_path = ART_DIR / f"{split}_gold_bank_{bank_label}_candidate_summary.csv"

    rows = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as in_handle, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as out_handle:
        reader = csv.DictReader(in_handle)
        for row in reader:
            query_id = str(row.get("query_id") or "")
            gold = set(split_gold(row.get("gold_citations")))
            gold_law = gold & law_source
            gold_court = gold & court_source
            record = {
                "query_id": query_id,
                "candidate_count": len(law_candidates) + len(court_candidates),
                "law_candidate_count": len(law_candidates),
                "court_candidate_count": len(court_candidates),
                "recall": recall(law_set | court_set, gold),
                "law_recall": recall(law_set, gold_law),
                "court_recall": recall(court_set, gold_court),
                "law_candidates": law_candidates,
                "court_candidates": court_candidates,
                "candidates": sorted(set(law_candidates) | set(court_candidates)),
            }
            out_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            rows.append(record)

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "candidate_count",
                "law_candidate_count",
                "court_candidate_count",
                "recall",
                "law_recall",
                "court_recall",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    law_recalls = [row["law_recall"] for row in rows if row["law_recall"] is not None]
    court_recalls = [row["court_recall"] for row in rows if row["court_recall"] is not None]

    print(f"Known bank splits: {','.join(bank_splits)}")
    print(f"Law candidates  : {len(law_candidates):,}")
    print(f"Court candidates: {len(court_candidates):,}")
    if law_recalls:
        print(f"Mean law recall  : {sum(law_recalls) / len(law_recalls):.6f}")
        print(f"Min law recall   : {min(law_recalls):.6f}")
    if court_recalls:
        print(f"Mean court recall: {sum(court_recalls) / len(court_recalls):.6f}")
        print(f"Min court recall : {min(court_recalls):.6f}")
    print(f"Saved candidates : {output_path}")
    print(f"Saved summary    : {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--bank-splits",
        default="train,val",
        help="Comma-separated known-gold files to use from data_insights, e.g. train,val or val.",
    )
    args = parser.parse_args()
    bank_splits = [x.strip() for x in args.bank_splits.split(",") if x.strip()]
    allowed = {"train", "val"}
    unknown = sorted(set(bank_splits) - allowed)
    if unknown:
        raise SystemExit(f"Unsupported bank split(s): {unknown}. Use train and/or val.")
    build(args.split, bank_splits)


if __name__ == "__main__":
    main()
