from __future__ import annotations

import csv
import json
import sqlite3
import sys
import unittest
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_citation_graph import classify_source_citation
from segment_lattice_v3 import audit_candidates, build_index, run_candidates


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


class ParserTests(unittest.TestCase):
    def test_law_code_or_and_long_article_suffixes_are_preserved(self) -> None:
        parsed = classify_source_citation("Art. 1 Abs. 1 OR")
        self.assertEqual(parsed.segments["law_code"], "OR")
        self.assertEqual(parsed.segments["law_code_family"], "uppercase_abbreviation")

        parsed = classify_source_citation("Art. 102abis Abs. 1 AsylG")
        self.assertEqual(parsed.segments["article"], "102abis")
        self.assertEqual(parsed.segments["law_code"], "AsylG")

        parsed = classify_source_citation("Art. 29dbis Abs. 1 USG")
        self.assertEqual(parsed.segments["article"], "29dbis")
        self.assertEqual(parsed.segments["law_code"], "USG")

    def test_multi_unit_law_and_court_corner_cases(self) -> None:
        parsed = classify_source_citation("Art. 221 Abs. 1 lit. b StPO")
        self.assertEqual(
            parsed.segments["units"],
            [
                {"marker": "Abs.", "value": "1", "category": "paragraph"},
                {"marker": "Bst.", "value": "b", "category": "letter"},
            ],
        )

        self.assertEqual(classify_source_citation("BGE 137 IV 122 E. 6.2").pattern, "court_bge")
        self.assertEqual(classify_source_citation("1B_210/2023 E. 4.1").segments["court_chamber"], "1")
        self.assertEqual(classify_source_citation("1C.1/1998 05.03.2002 E. 2").segments["separator_style"], "dot")
        self.assertEqual(classify_source_citation("B 1/00 29.01.2002 E. 1").segments["separator_style"], "space")

        empty_e = classify_source_citation("12T_1/2007 29.05.2007 E.")
        self.assertEqual(empty_e.pattern, "unknown")
        self.assertEqual(empty_e.segments["docket_prefix"], "12T")
        self.assertEqual(empty_e.segments["consideration_raw"], "")

        dash_e = classify_source_citation("1A.1/2000 08.05.2000 E. 1.-")
        self.assertEqual(dash_e.pattern, "unknown")
        self.assertEqual(dash_e.segments["consideration"], "1")

        compact = classify_source_citation("1C/8235")
        self.assertEqual(compact.pattern, "unknown")
        self.assertEqual(compact.segments["docket_prefix"], "1C")
        self.assertEqual(compact.segments["serial_number"], "8235")


class SegmentLatticeBuilderTests(unittest.TestCase):
    def test_registry_is_data_driven_and_candidate_generation_is_no_leak(self) -> None:
        tmp_root = ROOT / "artifacts" / f"test_segment_lattice_v3_{time.time_ns()}"
        tmp_root.mkdir(parents=True, exist_ok=False)
        base = tmp_root
        data_dir = base / "data"
        insights_dir = base / "data_insights"
        art_dir = base / "artifacts"

        write_csv(
                data_dir / "laws_de.csv",
                ["citation", "text", "title"],
                [
                    {
                        "citation": "Art. 1 Abs. 1 OR",
                        "text": "Zum Abschlusse eines Vertrages...",
                        "title": "Obligationenrecht - Vertragsschluss",
                    },
                    {
                        "citation": "Art. 221 Abs. 1 StPO",
                        "text": "Untersuchungshaft...",
                        "title": "Strafprozessordnung - Sicherheitshaft",
                    },
                ],
            )
        write_csv(
                data_dir / "court_considerations.csv",
                ["citation", "text"],
                [
                    {"citation": "BGE 137 IV 122 E. 6.2", "text": "Haft und Kollusionsgefahr"},
                    {"citation": "1C/8235", "text": "Compact slash example"},
                ],
            )
        write_csv(
            data_dir / "test.csv",
            ["query_id", "query"],
                [
                    {
                        "query_id": "q1",
                        "query": "Does Art. 1 Abs. 1 OR apply, and is 1C/8235 relevant?",
                    }
                ],
            )
        write_csv(
            data_dir / "val.csv",
            ["query_id", "query", "gold_citations"],
            [
                {
                    "query_id": "q1",
                    "query": "Does Art. 1 Abs. 1 OR apply, and is 1C/8235 relevant?",
                    "gold_citations": "Art. 1 Abs. 1 OR;1C/8235",
                }
            ],
        )

        # The law JSONL intentionally contains stale/null OR segments. The v3
        # builder must refresh source parsing from the canonical citation.
        write_jsonl(
                insights_dir / "laws_de_classified_citations.jsonl",
                [
                    {
                        "citation": "Art. 1 Abs. 1 OR",
                        "family": "law",
                        "subfamily": "statute_article",
                        "pattern": "statute_article",
                        "origin_mask": 1,
                        "segments": {
                            "article": "1",
                            "law_code": None,
                            "law_code_family": "unresolved_law_code",
                            "units": [{"marker": "Abs.", "value": "1", "category": "paragraph"}],
                        },
                    }
                ],
            )
        write_jsonl(
                insights_dir / "court_considerations_classified_citations.jsonl",
                [
                    {
                        "citation": "BGE 137 IV 122 E. 6.2",
                        "family": "court",
                        "subfamily": "bge",
                        "pattern": "court_bge",
                        "origin_mask": 1,
                        "segments": {
                            "reporter": "BGE",
                            "volume": "137",
                            "division": "IV",
                            "page": "122",
                            "pinpoint_unit": "E",
                            "pinpoint": "6.2",
                        },
                    },
                    {
                        "citation": "1C/8235",
                        "family": "unknown",
                        "subfamily": "unknown",
                        "pattern": "unknown",
                        "origin_mask": 1,
                        "segments": {"raw": "1C/8235"},
                    },
                ],
            )

        db_path = art_dir / "segment_lattice_v3.sqlite"
        cards_path = art_dir / "choice_cards_v3.json"
        build_index(data_dir, insights_dir, db_path, cards_path, force=True)

        conn = sqlite3.connect(db_path)
        self.assertIsNotNone(
            conn.execute(
                """
                SELECT 1 FROM option_registry
                WHERE segment_key = 'law_code' AND option_value = 'OR'
                  AND selector_kind = 'semantic'
                """
            ).fetchone()
        )
        self.assertIsNotNone(
            conn.execute(
                """
                SELECT 1 FROM option_registry
                WHERE segment_key = 'docket_prefix' AND option_value = '1C'
                """
            ).fetchone()
        )
        self.assertIsNotNone(
            conn.execute(
                """
                SELECT 1 FROM citation_segments
                WHERE citation = '1C/8235' AND segment_key = 'serial_number'
                  AND option_value = '8235'
                """
            ).fetchone()
        )
        conn.close()

        run_candidates("test", data_dir, db_path, art_dir, law_budget=10, court_budget=10)
        out_path = art_dir / "test_segment_lattice_v3_candidate_sets.jsonl"
        record = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertIn("Art. 1 Abs. 1 OR", record["candidates"])
        self.assertIn("1C/8235", record["candidates"])
        self.assertTrue(cards_path.exists())

        run_candidates("val", data_dir, db_path, art_dir, law_budget=10, court_budget=10)
        audit_candidates("val", data_dir, db_path, art_dir)
        summary_path = art_dir / "val_segment_lattice_v3_candidate_summary.csv"
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            summary = list(csv.DictReader(handle))
        self.assertEqual(summary[0]["recall"], "1.0")


if __name__ == "__main__":
    unittest.main()
