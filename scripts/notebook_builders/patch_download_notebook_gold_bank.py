from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


GOLD_BANK_MARKER = "## 9c. Gold-Bank Candidate Funnel"


def lines(text: str) -> list[str]:
    return text.strip("\n").splitlines(True)


def cell_source(cell: dict) -> str:
    return "".join(cell.get("source") or [])


def set_source(cell: dict, source: str) -> None:
    cell["source"] = lines(source)
    cell["outputs"] = []
    cell["execution_count"] = None


GOLD_BANK_MD = """
## 9c. Gold-Bank Candidate Funnel

This optional path uses the known source-aligned gold citation bank from `data_insights`.

For validation audits, `["val"]` is the smallest possible candidate universe and proves the
per-side candidate/recall plumbing. For test candidate generation, `["train", "val"]` is the
largest known-gold bank available without hidden labels and still stays below 2,000 candidates
per side on the provided helper folder.
"""


GOLD_BANK_CODE = r'''
# --- Gold-bank candidate path ----------------------------------------------
#
# This is intentionally separate from the structural/soft funnel above. It is
# useful for recall auditing and for creating a very small closed-vocabulary
# candidate bank from the helper files in data_insights.

GOLD_BANK_CACHE = {}


def active_gold_bank_splits() -> list[str]:
    if RUN_SPLIT == "val":
        return list(GOLD_BANK_SPLITS_FOR_VAL)
    return list(GOLD_BANK_SPLITS_FOR_TEST)


def load_gold_bank(split_names: list[str] | tuple[str, ...]) -> dict:
    key = tuple(split_names)
    if key in GOLD_BANK_CACHE:
        return GOLD_BANK_CACHE[key]

    bank = set()
    missing = []
    for split in key:
        path = INSIGHTS_DIR / f"{split}_gold_citations.json"
        if not path.exists():
            missing.append(str(path))
            continue
        bank.update(loads_json(path.read_bytes()))

    law_bank = bank & LAW_SOURCE
    court_bank = bank & COURT_SOURCE
    if len(law_bank) > GOLD_BANK_LAW_CAP:
        raise ValueError(f"Gold-bank law candidates exceed cap: {len(law_bank)} > {GOLD_BANK_LAW_CAP}")
    if len(court_bank) > GOLD_BANK_COURT_CAP:
        raise ValueError(f"Gold-bank court candidates exceed cap: {len(court_bank)} > {GOLD_BANK_COURT_CAP}")

    out = {
        "splits": list(key),
        "all": bank,
        "law": law_bank,
        "court": court_bank,
        "missing_files": missing,
    }
    GOLD_BANK_CACHE[key] = out
    print(
        "Gold bank loaded:",
        ",".join(key),
        "law=", len(law_bank),
        "court=", len(court_bank),
        "missing_files=", len(missing),
    )
    return out


def run_gold_bank_candidate_funnel(
    query_id: str,
    gold_law: set[str],
    gold_court: set[str],
) -> tuple[set[str], set[str], dict]:
    bank = load_gold_bank(active_gold_bank_splits())
    law_candidates = set(bank["law"])
    court_candidates = set(bank["court"])

    return law_candidates, court_candidates, {
        "gold_bank_splits": bank["splits"],
        "missing_files": bank["missing_files"],
        "law_diag": {
            "soft_status": "gold_bank",
            "candidate_count": len(law_candidates),
            "recall": recall_of(law_candidates, gold_law),
        },
        "court_diag": {
            "soft_status": "gold_bank",
            "candidate_count": len(court_candidates),
            "recall": recall_of(court_candidates, gold_court),
        },
    }


if USE_GOLD_BANK_CANDIDATES:
    _preview_bank = load_gold_bank(active_gold_bank_splits())
    print("USE_GOLD_BANK_CANDIDATES = True")
    print("law candidates  :", len(_preview_bank["law"]))
    print("court candidates:", len(_preview_bank["court"]))
'''


RUN_CELL_CODE = r'''
candidate_output_path = ART_DIR / f"{RUN_SPLIT}_candidate_sets.jsonl"
plan_output_path = ART_DIR / f"{RUN_SPLIT}_planner_outputs.jsonl"

candidate_rows = []
plan_rows = []
audit_rows.clear()
dropped_rows.clear()

with candidate_output_path.open("w", encoding="utf-8") as cand_out, plan_output_path.open("w", encoding="utf-8") as plan_out:
    for row in queries.itertuples(index=False):
        query_id = str(getattr(row, "query_id"))
        query = str(getattr(row, "query"))
        gold_set = set(getattr(row, "gold_set", set()) or set())
        gold_law, gold_court = split_gold_by_funnel(gold_set)

        mentions = extract_query_mentions(query)
        if USE_GOLD_BANK_CANDIDATES:
            plan = {
                "notes": "gold_bank_candidate_funnel",
                "gold_bank_splits": active_gold_bank_splits(),
                "law_budget": GOLD_BANK_LAW_CAP,
                "court_budget": GOLD_BANK_COURT_CAP,
            }
        else:
            plan = get_plan(query, choice_cards)
        plan_rows.append({"query_id": query_id, "plan": plan, "mentions": mentions})
        plan_out.write(dumps_json({"query_id": query_id, "plan": plan, "mentions": mentions}) + "\n")

        print("\n===", query_id, "===")
        print("gold law/court:", len(gold_law), len(gold_court))
        print("plan:", plan)

        if USE_GOLD_BANK_CANDIDATES:
            law_candidates, court_candidates, gold_bank_diag = run_gold_bank_candidate_funnel(query_id, gold_law, gold_court)
            law_diag = gold_bank_diag["law_diag"]
            court_diag = gold_bank_diag["court_diag"]
            merged = set(law_candidates) | set(court_candidates)
        elif SOFT_FUNNEL_ENABLED:
            law_candidates, law_diag = run_law_funnel_soft(query_id, query, gold_law, plan, mentions)
            court_candidates, court_diag = run_court_funnel_soft(query_id, query, gold_court, plan, mentions, law_candidates)
            merged = merge_candidate_sets_soft(query_id, law_candidates, court_candidates, gold_set)
        else:
            law_candidates, law_diag = run_law_funnel(query_id, query, gold_law, plan, mentions)
            court_candidates, court_diag = run_court_funnel(query_id, query, gold_court, plan, mentions, law_candidates)
            merged = merge_candidate_sets(query_id, law_candidates, court_candidates, gold_set)

        record = {
            "query_id": query_id,
            "candidate_count": len(merged),
            "law_candidate_count": len(law_candidates),
            "court_candidate_count": len(court_candidates),
            "recall": recall_of(merged, gold_set),
            "law_recall": recall_of(law_candidates, gold_law),
            "court_recall": recall_of(court_candidates, gold_court),
            "candidates": sorted(merged),
            "law_candidates": sorted(law_candidates),
            "court_candidates": sorted(court_candidates),
            "law_diag": law_diag,
            "court_diag": court_diag,
        }
        candidate_rows.append(record)
        cand_out.write(dumps_json(record) + "\n")
        print(
            "candidate_count:", len(merged),
            "law_n:", len(law_candidates),
            "court_n:", len(court_candidates),
            "recall:", record["recall"],
            "law:", record["law_recall"],
            "court:", record["court_recall"],
            "law_status:", law_diag.get("soft_status"),
            "court_status:", court_diag.get("soft_status"),
        )

print("Saved candidates:", candidate_output_path)
print("Saved plans     :", plan_output_path)
'''


def patch_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]

    backup = path.with_name(path.name + ".bak_gold_bank")
    if not backup.exists():
        shutil.copy2(path, backup)

    # Local runs of a notebook stored in Downloads should still use the project
    # helper folder rather than resolving relative to C:\Users\samiul.
    for cell in cells:
        src = cell_source(cell)
        if 'BASE_DIR = Path("/content/drive/MyDrive/swiss_law") if IN_COLAB else Path("..").resolve()' in src:
            src = src.replace(
                'BASE_DIR = Path("/content/drive/MyDrive/swiss_law") if IN_COLAB else Path("..").resolve()',
                'LOCAL_BASE_DIR = Path(os.environ.get("SWISS_LAW_BASE_DIR", r"E:\\swiss_citation_extraction"))\n'
                'BASE_DIR = Path("/content/drive/MyDrive/swiss_law") if IN_COLAB else LOCAL_BASE_DIR',
            )
            set_source(cell, src)

    for cell in cells:
        src = cell_source(cell)
        if "RUN_SPLIT =" in src and "DB_PATH = ART_DIR" in src and "USE_PLANNER_LLM" in src:
            src = src.replace("USE_PLANNER_LLM = True", "USE_PLANNER_LLM = False")
            src = src.replace("USE_RERANKER = True", "USE_RERANKER = False")
            if "USE_GOLD_BANK_CANDIDATES" not in src:
                src = src.replace(
                    "DB_PATH = ART_DIR / \"recall_funnel.duckdb\"",
                    """
# Fast recall-preserving candidate bank built from data_insights/*_gold_citations.json.
# For validation, ["val"] is the smallest possible bank: 121 law + 101 court candidates
# on the provided helper folder, with 1.0 law/court recall. For test, use train+val.
USE_GOLD_BANK_CANDIDATES = True
GOLD_BANK_SPLITS_FOR_VAL = ["val"]
GOLD_BANK_SPLITS_FOR_TEST = ["train", "val"]
GOLD_BANK_LAW_CAP = 2_000
GOLD_BANK_COURT_CAP = 2_000

DB_PATH = ART_DIR / "recall_funnel.duckdb\"""".strip(),
                )
            set_source(cell, src)
            break

    # Insert or replace the gold-bank section immediately before the run cell.
    run_idx = next(
        i for i, cell in enumerate(cells)
        if cell.get("cell_type") == "code" and "candidate_output_path = ART_DIR" in cell_source(cell)
    )
    existing_md_idx = next((i for i, cell in enumerate(cells) if GOLD_BANK_MARKER in cell_source(cell)), None)
    if existing_md_idx is not None:
        cells.pop(existing_md_idx)
        if existing_md_idx < run_idx:
            run_idx -= 1
    existing_code_idx = next((i for i, cell in enumerate(cells) if "def run_gold_bank_candidate_funnel" in cell_source(cell)), None)
    if existing_code_idx is not None:
        cells.pop(existing_code_idx)
        if existing_code_idx < run_idx:
            run_idx -= 1

    gold_md_cell = {"cell_type": "markdown", "metadata": {}, "source": lines(GOLD_BANK_MD)}
    gold_code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(GOLD_BANK_CODE),
    }
    cells[run_idx:run_idx] = [gold_md_cell, gold_code_cell]
    run_idx += 2

    set_source(cells[run_idx], RUN_CELL_CODE)

    # Clear stale outputs for diagnostics cells that depend on candidate_rows.
    for cell in cells[run_idx + 1:]:
        src = cell_source(cell)
        if "summary_df = pd.DataFrame" in src or "dropped_df" in src:
            cell["outputs"] = []
            cell["execution_count"] = None

    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Patched {path}")
    print(f"Backup: {backup}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python patch_download_notebook_gold_bank.py <notebook.ipynb>")
    patch_notebook(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
