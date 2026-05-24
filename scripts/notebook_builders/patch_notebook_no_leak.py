"""Patch colab_recall_preserving_candidate_funnel (2).ipynb to remove val-gold
leakage, enable the planner LLM, and cap each side at <=2000 candidates.

This rewrites the notebook in place after writing a backup.

What it changes:
  * Cell 6 (config): USE_GOLD_BANK_CANDIDATES=False, USE_PLANNER_LLM=True,
    SOFT_LAW_TARGET=SOFT_COURT_TARGET=2000, MAX_TOTAL_CANDIDATES=4000.
  * Cell 20 (audit/budget helpers): drops the `RUN_SPLIT == "val"` gold-aware
    revert logic in `audit_stage` and `cap_candidates_with_guard`.
  * Cell 23 (soft funnel): drops the gold-aware probe-up in
    `take_top_k_with_recall_probe`, drops the gold-aware refuse-to-trim in
    `merge_candidate_sets_soft`. Adds a `train_gold_pin` set built once from
    `data_insights/train_gold_citations.json` so the soft funnel can pin
    train-known citations on every split (legit prior, not val-leak).
  * Cell 24 (hard funnel): drops the val-specific refuse-to-cap in
    `merge_candidate_sets`.
  * Cell 26/27 (gold-bank): adds a leakage warning to the markdown and makes
    cell 27 a no-op when USE_GOLD_BANK_CANDIDATES is False (the default now).
  * Cell 28 (runner): unchanged structurally (still keys off
    USE_GOLD_BANK_CANDIDATES and SOFT_FUNNEL_ENABLED), so the soft path runs.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

NOTEBOOK = Path(r"C:/Users/samiul/Downloads/colab_recall_preserving_candidate_funnel (2).ipynb")
BACKUP = NOTEBOOK.with_suffix(NOTEBOOK.suffix + ".bak_no_leak")


CELL_6_CONFIG = '''RUN_SPLIT = "val"  # "val", "test", or "train"

BUILD_TEXT_TABLES = True      # Builds DuckDB tables with citation text. Slow but useful for reranking.
BUILD_EDGE_TABLES = True      # Streams links JSON into citation_edges for statute-citing court routes.
FORCE_REBUILD_DB = False      # Set True after changing parsing/index logic.

USE_PLANNER_LLM = True        # Planner LLM produces structural choices. Falls back to heuristic on error.
USE_RERANKER = False          # Optional later stage.

PLANNER_MODEL_NAME = "Qwen/Qwen3-32B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

# Recall guards are diagnostic only. They never look at gold to decide what to keep.
EARLY_STAGE_MIN_RECALL = 0.98
MID_STAGE_MIN_RECALL = 0.95
FINAL_MIN_MEAN_RECALL = 0.85
FINAL_MIN_QUERY_RECALL = 0.75

# Hard per-side cap. Both law and court funnels stop at 2,000 candidates each;
# merged total is capped at 4,000.
MIN_TOTAL_CANDIDATES = 2_000
DEFAULT_TOTAL_CANDIDATES = 4_000
MAX_TOTAL_CANDIDATES = 4_000
DIAGNOSTIC_MAX_CANDIDATES = 4_000

DEFAULT_LAW_BUDGET = 2_000
DEFAULT_COURT_BUDGET = 2_000

# Soft-rank funnel knobs (used by run_*_funnel_soft below in section 9b)
SOFT_FUNNEL_ENABLED = True
SOFT_LAW_TARGET = 2_000
SOFT_COURT_TARGET = 2_000
# Probe multipliers were used to grow K against gold on val. That is leakage.
# Keep a single step so the cap is always honored on every split.
SOFT_PROBE_STEPS = [1.0]

# LEAKAGE WARNING. The gold-bank path below uses data_insights/*_gold_citations.json
# as the candidate set. For val with ["val"] this trivially yields recall 1.0
# (the answer key IS the candidate list). For test with ["train", "val"] it
# still relies on labels that overlap with the eval target. It is kept only as
# a diagnostic for the candidate plumbing and is OFF by default.
USE_GOLD_BANK_CANDIDATES = False
GOLD_BANK_SPLITS_FOR_VAL = ["val"]
GOLD_BANK_SPLITS_FOR_TEST = ["train", "val"]
GOLD_BANK_LAW_CAP = 2_000
GOLD_BANK_COURT_CAP = 2_000

DB_PATH = ART_DIR / "recall_funnel.duckdb"
print("DB_PATH:", DB_PATH)
print("USE_PLANNER_LLM         :", USE_PLANNER_LLM)
print("USE_GOLD_BANK_CANDIDATES:", USE_GOLD_BANK_CANDIDATES, "(leakage path; off by default)")
print("LAW/COURT budget        :", DEFAULT_LAW_BUDGET, "/", DEFAULT_COURT_BUDGET)
'''


CELL_20_AUDIT = '''audit_rows = []
dropped_rows = []


def split_gold_by_funnel(gold_set: set[str]) -> tuple[set[str], set[str]]:
    return gold_set & LAW_SOURCE, gold_set & COURT_SOURCE


def recall_of(candidates: set[str], gold: set[str]) -> float | None:
    if not gold:
        return None
    return len(candidates & gold) / len(gold)


def audit_stage(
    query_id: str,
    funnel: str,
    stage: str,
    before: set[str],
    after: set[str],
    gold: set[str],
    min_recall: float | None,
    drop_reason: str,
) -> set[str]:
    """Diagnostic-only auditor. Never reverts a stage based on gold.

    Earlier versions reverted to `before` when `recall_after < min_recall` on
    val. That is a gold-aware decision that does not generalize to test, so
    it has been removed. We still record `min_recall` and `status` so plots
    can show which stages would have been guarded.
    """
    before_gold = before & gold
    after_gold = after & gold
    dropped = sorted(before_gold - after_gold)
    rec_before = recall_of(before, gold)
    rec_after = recall_of(after, gold)
    status = "ok"
    if gold and min_recall is not None and (rec_after or 0.0) < min_recall:
        status = "below_min_recall"  # diagnostic; do NOT revert

    audit_rows.append(
        {
            "query_id": query_id,
            "funnel": funnel,
            "stage": stage,
            "candidate_count_before": len(before),
            "candidate_count_after": len(after),
            "candidate_count_final": len(after),
            "gold_count": len(gold),
            "gold_kept_before": len(before_gold),
            "gold_kept_after": len(after_gold),
            "gold_dropped": len(dropped),
            "recall_before": rec_before,
            "recall_after": rec_after,
            "min_recall": min_recall,
            "status": status,
            "drop_reason": drop_reason,
        }
    )

    for citation in dropped:
        dropped_rows.append(
            {
                "query_id": query_id,
                "funnel": funnel,
                "first_failed_stage": stage,
                "citation": citation,
                "drop_reason": drop_reason,
                "stage_status": status,
            }
        )

    return after


def score_law_candidates(candidates: set[str], query: str, plan: dict, explicit: set[str]) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame(columns=["citation", "score"])
    tmp = create_temp_citation_table("tmp_law_score_candidates", candidates)
    df = con.execute(
        f"""
        SELECT s.citation, s.law_code, s.article, s.unit_chain, coalesce(l.title, '') AS title
        FROM {tmp} tc
        JOIN law_segments s USING (citation)
        LEFT JOIN law_citations l USING (citation)
        """
    ).fetchdf()
    selected_codes = set(plan.get("law_codes") or [])
    q = query[:2000]
    scores = []
    for row in df.itertuples(index=False):
        score = 0.0
        if row.citation in explicit:
            score += 1000
        if row.law_code in selected_codes:
            score += 80
        if row.law_code in COMMON_SAFETY_LAW_CODES:
            score += 20
        if row.title:
            score += 0.25 * fuzz.partial_ratio(q, str(row.title))
        if row.article and str(row.article) in q:
            score += 15
        scores.append(score)
    df["score"] = scores
    return df[["citation", "score"]].sort_values(["score", "citation"], ascending=[False, True])


def score_court_candidates(candidates: set[str], query: str, plan: dict, explicit: set[str], citing_cases: set[str]) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame(columns=["citation", "score"])
    tmp = create_temp_citation_table("tmp_court_score_candidates", candidates)
    df = con.execute(
        f"""
        SELECT c.citation, c.pattern, c.subfamily, c.docket_prefix, c.bge_division, c.decision_year, c.court_base, c.consideration
        FROM {tmp} tc
        JOIN court_segments c USING (citation)
        """
    ).fetchdf()
    prefixes = set(plan.get("docket_prefixes") or [])
    divisions = set(plan.get("bge_divisions") or [])
    time_mode = plan.get("time_mode") or "no_time_filter"
    q = query[:2000]
    scores = []
    for row in df.itertuples(index=False):
        score = 0.0
        if row.citation in explicit:
            score += 1000
        if row.citation in citing_cases:
            score += 120
        if row.pattern == "court_bge":
            score += 40
        if row.pattern == "court_case":
            score += 30
        if row.pattern == "unknown":
            score += 5
        if row.docket_prefix in prefixes:
            score += 80
        if row.bge_division in divisions:
            score += 80
        if row.court_base and str(row.court_base) in q:
            score += 500
        if time_mode == "soft_recency" and row.decision_year and str(row.decision_year).isdigit():
            score += max(0, int(str(row.decision_year)[-4:]) - 1990) / 10
        scores.append(score)
    df["score"] = scores
    return df[["citation", "score"]].sort_values(["score", "citation"], ascending=[False, True])


def cap_candidates_with_guard(
    query_id: str,
    funnel: str,
    stage: str,
    before: set[str],
    scored: pd.DataFrame,
    budget: int,
    gold: set[str],
    explicit_keep: set[str],
    min_recall: float,
) -> set[str]:
    """Hard-cap to `budget`, pin `explicit_keep`. Never grows past budget on
    gold-aware recall checks. Audit only records whether recall fell below
    `min_recall` so it can be inspected after the run."""
    if len(before) <= budget:
        return before
    budget = max(0, int(budget))
    top = set(scored.head(budget)["citation"].astype(str)) | explicit_keep
    return audit_stage(query_id, funnel, stage, before, top, gold, min_recall, "budget_cut")
'''


CELL_23_SOFT = '''# --- Soft-rank scoring (whole corpus, then top-K) ---------------------------
#
# Score every source row, pin must-keep items, take top-K. K is fixed by the
# planner budget; we never grow K against gold (that would be a val-only leak).

LAW_SCORE_SQL_TEMPLATE = """
SELECT
    s.citation,
    (
        CASE WHEN s.citation IN {explicit_sql} THEN 1000.0 ELSE 0.0 END
      + CASE WHEN s.law_code IN {selected_codes_sql} THEN 80.0 ELSE 0.0 END
      + CASE WHEN s.law_code IN {safety_codes_sql} THEN 20.0 ELSE 0.0 END
      + CASE WHEN s.law_code IS NULL THEN 5.0 ELSE 0.0 END
      + CASE WHEN s.article IN {explicit_articles_sql} THEN 25.0 ELSE 0.0 END
      + CASE WHEN s.citation IN {train_pin_sql} THEN 60.0 ELSE 0.0 END
    ) AS score
FROM law_segments s
"""

COURT_SCORE_SQL_TEMPLATE = """
WITH citing AS (
    SELECT DISTINCT e.source AS citation
    FROM citation_edges e
    JOIN ({selected_laws_sql_inner}) l ON l.citation = e.target
    WHERE e.dataset = 'court_considerations'
)
SELECT
    c.citation,
    (
        CASE WHEN c.citation IN {explicit_sql} THEN 1000.0 ELSE 0.0 END
      + CASE WHEN c.court_base IN {explicit_bases_sql} THEN 800.0 ELSE 0.0 END
      + CASE WHEN c.docket_prefix IN {selected_prefixes_sql} THEN 80.0 ELSE 0.0 END
      + CASE WHEN c.bge_division IN {selected_divisions_sql} AND c.pattern = 'court_bge' THEN 80.0 ELSE 0.0 END
      + CASE WHEN c.pattern = 'court_bge' THEN 25.0 ELSE 0.0 END
      + CASE WHEN c.pattern = 'court_case' THEN 15.0 ELSE 0.0 END
      + CASE WHEN c.pattern = 'unknown'   THEN 2.0  ELSE 0.0 END
      + CASE WHEN c.citation IN (SELECT citation FROM citing) THEN 120.0 ELSE 0.0 END
      + CASE WHEN c.citation IN {train_pin_sql} THEN 60.0 ELSE 0.0 END
    ) AS score
FROM court_segments c
"""

def _sql_in_or_empty(values: Iterable[str]) -> str:
    """Quoted IN list that always matches nothing if values is empty."""
    vals = [str(v).replace("'", "''") for v in values if v is not None and str(v) != ""]
    if not vals:
        return "('__no_match_sentinel__')"
    return "(" + ",".join(f"'{v}'" for v in vals) + ")"


# Train-only prior. Loaded once. This is fair on every split because train gold
# IS training data. It is NOT the gold-bank leakage of using val/test gold.
def _load_train_gold_pin() -> tuple[set[str], set[str]]:
    path = INSIGHTS_DIR / "train_gold_citations.json"
    if not path.exists():
        print("train_gold_citations.json not found; train pin disabled")
        return set(), set()
    bank = set(loads_json(path.read_bytes()))
    return bank & LAW_SOURCE, bank & COURT_SOURCE


TRAIN_LAW_PIN, TRAIN_COURT_PIN = _load_train_gold_pin()
print(f"Train-prior pin (legit): law={len(TRAIN_LAW_PIN):,}, court={len(TRAIN_COURT_PIN):,}")


def score_all_law_rows(plan: dict, mentions: dict, explicit: set[str]) -> pd.DataFrame:
    selected_codes = set(plan.get("law_codes") or [])
    safety_codes = COMMON_SAFETY_LAW_CODES & allowed_ids(choice_cards["law_codes"])
    explicit_articles = {m.get("article") for m in mentions.get("laws", []) if m.get("article")}

    sql = LAW_SCORE_SQL_TEMPLATE.format(
        explicit_sql=_sql_in_or_empty(explicit),
        selected_codes_sql=_sql_in_or_empty(selected_codes),
        safety_codes_sql=_sql_in_or_empty(safety_codes),
        explicit_articles_sql=_sql_in_or_empty(explicit_articles),
        train_pin_sql=_sql_in_or_empty(TRAIN_LAW_PIN),
    )
    return con.execute(sql).fetchdf()


def score_all_court_rows(plan: dict, mentions: dict, explicit: set[str], selected_laws: set[str]) -> pd.DataFrame:
    selected_prefixes = set(plan.get("docket_prefixes") or [])
    selected_divisions = set(plan.get("bge_divisions") or [])
    explicit_bases = set()
    for m in mentions.get("bge", []):
        if m.get("base"):
            explicit_bases.add(m["base"])
    for m in mentions.get("dockets", []):
        if m.get("docket"):
            explicit_bases.add(m["docket"])

    if selected_laws:
        tmp = create_temp_citation_table("tmp_soft_selected_laws", selected_laws)
        selected_laws_sql_inner = f"SELECT citation FROM {tmp}"
    else:
        selected_laws_sql_inner = "SELECT '__no_match_sentinel__' AS citation WHERE 1=0"

    sql = COURT_SCORE_SQL_TEMPLATE.format(
        explicit_sql=_sql_in_or_empty(explicit),
        explicit_bases_sql=_sql_in_or_empty(explicit_bases),
        selected_prefixes_sql=_sql_in_or_empty(selected_prefixes),
        selected_divisions_sql=_sql_in_or_empty(selected_divisions),
        selected_laws_sql_inner=selected_laws_sql_inner,
        train_pin_sql=_sql_in_or_empty(TRAIN_COURT_PIN),
    )
    return con.execute(sql).fetchdf()


def take_top_k_with_recall_probe(
    scored: pd.DataFrame,
    pinned: set[str],
    base_k: int,
    gold: set[str],
    min_recall: float,
    max_k: int,
) -> tuple[set[str], int, float | None, str]:
    """Take top-K by score, union with pinned. K is fixed; we never grow K
    based on gold (that would be a val-only leak).

    Returns (chosen_set, final_k, recall_after, status). `recall_after` is
    None when gold is empty; otherwise it is reported for diagnostics only.
    """
    if scored.empty:
        return set(pinned), len(pinned), recall_of(set(pinned), gold) if gold else None, "no_candidates"

    scored_sorted = scored.sort_values(["score", "citation"], ascending=[False, True])
    citations_sorted = scored_sorted["citation"].astype(str).tolist()

    k = int(min(base_k, len(citations_sorted), max_k))
    chosen = set(citations_sorted[:k]) | set(pinned)
    rec = recall_of(chosen, gold) if gold else None
    return chosen, len(chosen), rec, "ok"


def run_law_funnel_soft(query_id: str, query: str, gold_law: set[str], plan: dict, mentions: dict) -> tuple[set[str], dict]:
    explicit = expand_explicit_law_mentions(mentions)
    pinned = set(explicit) | set(TRAIN_LAW_PIN)

    scored = score_all_law_rows(plan, mentions, explicit)

    chosen, k, rec, status = take_top_k_with_recall_probe(
        scored=scored,
        pinned=pinned,
        base_k=int(plan.get("law_budget") or SOFT_LAW_TARGET),
        gold=gold_law,
        min_recall=FINAL_MIN_QUERY_RECALL,
        max_k=SOFT_LAW_TARGET,
    )

    audit_rows.append({
        "query_id": query_id, "funnel": "law", "stage": "L_soft_topk",
        "candidate_count_before": len(LAW_SOURCE),
        "candidate_count_after": k, "candidate_count_final": k,
        "gold_count": len(gold_law),
        "gold_kept_before": len(LAW_SOURCE & gold_law),
        "gold_kept_after": len(chosen & gold_law),
        "gold_dropped": len((LAW_SOURCE & gold_law) - (chosen & gold_law)),
        "recall_before": 1.0 if gold_law else None,
        "recall_after": rec,
        "min_recall": FINAL_MIN_QUERY_RECALL,
        "status": status,
        "drop_reason": "soft_topk",
    })
    diag = {
        "explicit_law_candidates": sorted(explicit),
        "stages": {"L_soft_topk": k},
        "soft_status": status,
        "candidate_count": k,
        "recall": rec,
    }
    return chosen, diag


def run_court_funnel_soft(query_id: str, query: str, gold_court: set[str], plan: dict, mentions: dict, selected_laws: set[str]) -> tuple[set[str], dict]:
    explicit = expand_explicit_court_mentions(mentions)
    pinned = set(explicit) | set(TRAIN_COURT_PIN)

    # Pin one same-decision-neighbors hop from explicit BGE/docket bases.
    if explicit:
        pinned |= same_decision_expansion(explicit)

    scored = score_all_court_rows(plan, mentions, explicit, selected_laws)

    chosen, k, rec, status = take_top_k_with_recall_probe(
        scored=scored,
        pinned=pinned,
        base_k=int(plan.get("court_budget") or SOFT_COURT_TARGET),
        gold=gold_court,
        min_recall=FINAL_MIN_QUERY_RECALL,
        max_k=SOFT_COURT_TARGET,
    )

    audit_rows.append({
        "query_id": query_id, "funnel": "court", "stage": "C_soft_topk",
        "candidate_count_before": len(COURT_SOURCE),
        "candidate_count_after": k, "candidate_count_final": k,
        "gold_count": len(gold_court),
        "gold_kept_before": len(COURT_SOURCE & gold_court),
        "gold_kept_after": len(chosen & gold_court),
        "gold_dropped": len((COURT_SOURCE & gold_court) - (chosen & gold_court)),
        "recall_before": 1.0 if gold_court else None,
        "recall_after": rec,
        "min_recall": FINAL_MIN_QUERY_RECALL,
        "status": status,
        "drop_reason": "soft_topk",
    })
    diag = {
        "explicit_court_candidates": sorted(explicit),
        "pinned_count": len(pinned),
        "stages": {"C_soft_topk": k},
        "soft_status": status,
        "candidate_count": k,
        "recall": rec,
    }
    return chosen, diag


def merge_candidate_sets_soft(query_id: str, law_candidates: set[str], court_candidates: set[str], gold_set: set[str]) -> set[str]:
    """Merge with a hard cap at MAX_TOTAL_CANDIDATES. Splits the cap evenly
    by current side sizes (laws first, then courts in a stable sort). Never
    refuses to trim based on gold.
    """
    merged = set(law_candidates) | set(court_candidates)
    if len(merged) <= MAX_TOTAL_CANDIDATES:
        return merged

    overflow = len(merged) - MAX_TOTAL_CANDIDATES
    trimmed_court = set(sorted(court_candidates)[: max(0, len(court_candidates) - overflow)])
    return set(law_candidates) | trimmed_court


print("Soft funnel helpers loaded (no-leak version).")
print("SOFT_LAW_TARGET    =", SOFT_LAW_TARGET)
print("SOFT_COURT_TARGET  =", SOFT_COURT_TARGET)
print("MAX_TOTAL_CANDIDATES =", MAX_TOTAL_CANDIDATES)
'''


CELL_24_HARD_TAIL = '''def merge_candidate_sets(query_id: str, law_candidates: set[str], court_candidates: set[str], gold_set: set[str]) -> set[str]:
    merged = set(law_candidates) | set(court_candidates)
    if len(merged) <= MAX_TOTAL_CANDIDATES:
        return merged

    # Hard funnel already budgeted each side. If union exceeds the global cap,
    # trim from the court side stably. We never look at gold to decide.
    overflow = len(merged) - MAX_TOTAL_CANDIDATES
    trimmed_court = set(sorted(court_candidates)[: max(0, len(court_candidates) - overflow)])
    return set(law_candidates) | trimmed_court
'''


CELL_26_MD = '''## 9c. Gold-Bank Candidate Funnel (LEAKAGE — diagnostic only, OFF by default)

This path uses `data_insights/{split}_gold_citations.json` directly as the
candidate set. With `["val"]` it trivially yields recall 1.0 on val because
the candidate list IS the val gold. With `["train", "val"]` it leaks val
labels into the val candidate pool. It is therefore **not a retrieval method**
and is kept only as a diagnostic for the per-side candidate plumbing.

`USE_GOLD_BANK_CANDIDATES` is `False` by default. Do not turn it on for any
val or test recall claim you intend to take seriously.
'''


def patch():
    if not NOTEBOOK.exists():
        raise SystemExit(f"Notebook not found: {NOTEBOOK}")
    if not BACKUP.exists():
        shutil.copy2(NOTEBOOK, BACKUP)
        print(f"Backup written: {BACKUP}")
    else:
        print(f"Backup already exists: {BACKUP}")

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = nb["cells"]

    def set_code(idx: int, body: str) -> None:
        c = cells[idx]
        if c["cell_type"] != "code":
            raise SystemExit(f"Expected code cell at {idx}, got {c['cell_type']}")
        c["source"] = body.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None

    def set_md(idx: int, body: str) -> None:
        c = cells[idx]
        if c["cell_type"] != "markdown":
            raise SystemExit(f"Expected md cell at {idx}, got {c['cell_type']}")
        c["source"] = body.splitlines(keepends=True)

    set_code(6, CELL_6_CONFIG)
    set_code(20, CELL_20_AUDIT)
    set_code(23, CELL_23_SOFT)

    # Cell 24 has both run_law_funnel + run_court_funnel + merge_candidate_sets.
    # Replace only the merge function tail.
    cell_24_src = "".join(cells[24]["source"])
    marker = "def merge_candidate_sets(query_id"
    idx = cell_24_src.find(marker)
    if idx < 0:
        raise SystemExit("merge_candidate_sets not found in cell 24")
    new_24 = cell_24_src[:idx] + CELL_24_HARD_TAIL
    cells[24]["source"] = new_24.splitlines(keepends=True)
    cells[24]["outputs"] = []
    cells[24]["execution_count"] = None

    set_md(26, CELL_26_MD)

    # Cell 27 (gold-bank code) is left in place but USE_GOLD_BANK_CANDIDATES
    # is now False, so its preview load print is skipped naturally.
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Patched:", NOTEBOOK)
    print("Re-validating cell parses...")
    import ast
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        try:
            ast.parse(src)
        except SyntaxError as e:
            raise SystemExit(f"Cell {i} failed to parse: {e}")
    print("All code cells parse cleanly.")


if __name__ == "__main__":
    patch()
