"""Generate the fresh no-leak Segment-Lattice Funnel v3 Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "colab_segment_lattice_funnel_v3.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(True),
    }


cells = [
    md(
        """
# Swiss Citation Retrieval: Segment-Lattice Funnel V3

This notebook uses the fresh v3 pipeline:

- segment and option cards are derived from `data_insights/*_classified_citations.jsonl`
- source rows are re-parsed with the current parser before indexing
- candidate generation never reads gold labels
- validation recall is computed only by the separate audit cell after candidates are written
"""
    ),
    code(
        """
import os
import sys
import subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

BASE_DIR = Path("/content/drive/MyDrive/swiss_law") if IN_COLAB else Path("..").resolve()
DATA_DIR = BASE_DIR / "data"
INSIGHTS_DIR = BASE_DIR / "data_insights"
ART_DIR = BASE_DIR / "artifacts"
SCRIPT_DIR = BASE_DIR / "scripts"

sys.path.insert(0, str(SCRIPT_DIR))
ART_DIR.mkdir(parents=True, exist_ok=True)

print("BASE_DIR    :", BASE_DIR)
print("DATA_DIR    :", DATA_DIR)
print("INSIGHTS_DIR:", INSIGHTS_DIR)
print("ART_DIR     :", ART_DIR)
print("SCRIPT_DIR  :", SCRIPT_DIR)
assert (SCRIPT_DIR / "segment_lattice_v3.py").exists(), "Copy scripts/segment_lattice_v3.py into BASE_DIR/scripts first"
assert (SCRIPT_DIR / "extract_citation_graph.py").exists(), "Copy scripts/extract_citation_graph.py into BASE_DIR/scripts first"
"""
    ),
    md(
        """
## 1. Optional Fresh Extraction

Run this only after changing parser logic or when `data_insights` may be stale. It recreates classified citations and links from the source CSVs.
"""
    ),
    code(
        """
REFRESH_EXTRACTION = False

if REFRESH_EXTRACTION:
    subprocess.check_call([
        sys.executable,
        str(SCRIPT_DIR / "extract_citation_graph.py"),
        "--data-dir", str(DATA_DIR),
        "--output-dir", str(INSIGHTS_DIR),
        "--db-name", "citation_graph_extracted.sqlite",
    ])
else:
    print("Skipping extraction refresh. Set REFRESH_EXTRACTION=True when parser/data changed.")
"""
    ),
    md(
        """
## 2. Invalidate Old Funnel Outputs

This removes only the old coarse-funnel artifacts. It does not delete source data, extracted `data_insights`, embeddings, or v3 outputs.
"""
    ),
    code(
        """
from segment_lattice_v3 import invalidate_stale_artifacts

INVALIDATE_OLD_FUNNEL_ARTIFACTS = False
if INVALIDATE_OLD_FUNNEL_ARTIFACTS:
    removed = invalidate_stale_artifacts(ART_DIR)
    print("Removed", len(removed), "old artifacts")
    for path in removed:
        print(path)
else:
    print("Old artifacts left untouched. Set INVALIDATE_OLD_FUNNEL_ARTIFACTS=True to remove them.")
"""
    ),
    md(
        """
## 3. Build Segment-Lattice Index And Cards

The index stores source citations, row-level segment options, option registry counts, English option context, and freshness metadata.
"""
    ),
    code(
        """
from segment_lattice_v3 import DEFAULT_CARDS, DEFAULT_DB, build_index

DB_PATH = ART_DIR / "segment_lattice_v3.sqlite"
CARDS_PATH = ART_DIR / "choice_cards_v3.json"

build_index(
    data_dir=DATA_DIR,
    insights_dir=INSIGHTS_DIR,
    db_path=DB_PATH,
    cards_path=CARDS_PATH,
    force=False,
)
print("DB_PATH   :", DB_PATH)
print("CARDS_PATH:", CARDS_PATH)
"""
    ),
    md(
        """
## 4. Inspect Registry Cards

Use this to confirm the planner sees actual data-derived segment keys and option values.
"""
    ),
    code(
        """
import json

cards = json.loads(CARDS_PATH.read_text(encoding="utf-8"))
print("card groups:", len(cards))
for key in sorted(cards)[:40]:
    print(key, "options=", len(cards[key]), "sample=", cards[key][:3])
"""
    ),
    md(
        """
## 5. Generate Candidates Without Gold

This cell reads only `query_id` and `query`. The same cell should be used for validation and test.
"""
    ),
    code(
        """
from segment_lattice_v3 import run_candidates

RUN_SPLIT = "val"  # "val", "test", or "train"
LAW_BUDGET = 2_000
COURT_BUDGET = 2_000
PLANNER_INPUT = None  # Optional JSONL of LLM segment_decisions validated against choice_cards_v3.json

run_candidates(
    split=RUN_SPLIT,
    data_dir=DATA_DIR,
    db_path=DB_PATH,
    out_dir=ART_DIR,
    law_budget=LAW_BUDGET,
    court_budget=COURT_BUDGET,
    planner_input=PLANNER_INPUT,
)
"""
    ),
    md(
        """
## 6. Validation Audit

This is intentionally separate from candidate generation. It reads gold labels only after candidate files have already been written.
"""
    ),
    code(
        """
from segment_lattice_v3 import audit_candidates

if RUN_SPLIT in {"train", "val"}:
    audit_candidates(
        split=RUN_SPLIT,
        data_dir=DATA_DIR,
        db_path=DB_PATH,
        out_dir=ART_DIR,
    )
else:
    print("No gold audit for test split.")
"""
    ),
]


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {OUT}")
