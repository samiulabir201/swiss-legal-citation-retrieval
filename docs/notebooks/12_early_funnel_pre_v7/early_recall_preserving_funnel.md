# early_recall_preserving_funnel

**Path:** `e:\swiss_citation_extraction\notebooks\12_early_funnel_pre_v7\early_recall_preserving_funnel.ipynb`

## Configuration

- `RUN_SPLIT = "val"` (supports `val`, `test`, `train`).
- DB build flags: `BUILD_TEXT_TABLES = True`, `BUILD_EDGE_TABLES = True`, `FORCE_REBUILD_DB = False`.
- `USE_PLANNER_LLM = True`, `USE_RERANKER = False`.
- Models: `PLANNER_MODEL_NAME = "Qwen/Qwen3-32B"`, `RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-8B"`, `EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"`.
- Recall guards (diagnostic only): `EARLY_STAGE_MIN_RECALL = 0.98`, `MID_STAGE_MIN_RECALL = 0.95`, `FINAL_MIN_MEAN_RECALL = 0.85`, `FINAL_MIN_QUERY_RECALL = 0.75`.
- Candidate caps: `MIN_TOTAL_CANDIDATES = 2_000`, `DEFAULT_TOTAL_CANDIDATES = 4_000`, `MAX_TOTAL_CANDIDATES = 4_000`, `DIAGNOSTIC_MAX_CANDIDATES = 4_000`.
- Per-side budgets: `DEFAULT_LAW_BUDGET = 2_000`, `DEFAULT_COURT_BUDGET = 2_000`.
- Soft-rank funnel: `SOFT_FUNNEL_ENABLED = True`, `SOFT_LAW_TARGET = 2_000`, `SOFT_COURT_TARGET = 2_000`, `SOFT_PROBE_STEPS = [1.0]` (no val-leakage K-growth).
- Gold-bank diagnostic (leakage path, OFF by default): `USE_GOLD_BANK_CANDIDATES = False`, `GOLD_BANK_SPLITS_FOR_VAL = ["val"]`, `GOLD_BANK_SPLITS_FOR_TEST = ["train", "val"]`, `GOLD_BANK_LAW_CAP = 2_000`, `GOLD_BANK_COURT_CAP = 2_000`.
- Base paths: Colab `/content/drive/MyDrive/swiss_law/{data,data_insights,artifacts}` (mirrored locally via `SWISS_LAW_BASE_DIR` env var, default `E:\swiss_citation_extraction`). DB at `ART_DIR / "recall_funnel.duckdb"`.

## Data

Input files (`FILES` dict), all reported `OK` from `/content/drive/MyDrive/swiss_law/...`:

- `data/laws_de.csv`, `data/court_considerations.csv`
- `data/train.csv`, `data/val.csv`, `data/test.csv`
- `data_insights/laws_de_classified_citations.jsonl`, `data_insights/court_considerations_classified_citations.jsonl`
- `data_insights/laws_de_links.json`, `data_insights/court_considerations_links.json`

DuckDB tables built into `recall_funnel.duckdb`: `citation_edges`, `court_considerations`, `court_decisions`, `court_segments`, `law_citations`, `law_segments`.

Source inventory (final candidate universe restricted to these citation IDs):

- `LAW_SOURCE`: 175,933
- `COURT_SOURCE`: 1,516,398
- `ALL_SOURCE`: 1,692,331
- law source rows ingested: 175,933; court source rows: 1,516,398.

Queries: 10 rows, split = `val`. Each row carries `query`, `gold_citations` (semicolon-joined), `gold_list`, `gold_set`.

Choice cards exported to `artifacts/choice_cards.json` with: 2,061 `law_codes`, 2 `law_unit_chains`, 3 `court_families`, 0 `bge_divisions`, 44 `docket_prefixes`, 9 `route_choices`, 4 `time_choices`. Top law codes by source count: `NO_CODE` (4419), `ZGB` (2383), `StPO` (1306), `StGB` (1238), `TSV` (1087), `VTS` (1022), `SchKG` (917), `ZPO` (857), `MStG` (810), `ZV` (717), `AVO` (716), `TSchV` (710), `BSV` (690), `BV` (658), `AIG` (629). Top docket prefixes: `2C` (271,690), `6B` (261,330), `5A` (197,522), `1C` (152,918), `4A` (151,334), `8C` (115,139), `1B` (76,363), `7B` (49,264), `1P` (30,816), `2A` (27,277).

## Pipeline

1. **Section 0 - Colab Setup** (cells 2-4): Mount Drive (or fall back to local `SWISS_LAW_BASE_DIR`), pip-install `duckdb, polars, pandas, pyarrow, orjson, ijson, regex, rapidfuzz, transformers, accelerate, bitsandbytes, sentence-transformers`, set up `BASE_DIR / DATA_DIR / INSIGHTS_DIR / ART_DIR`, verify input file presence.
2. **Section 1 - Configuration** (cell 6): Set split, model names, recall gates, per-side and total candidate budgets; print summary.
3. **Section 2 - Regex / citation normalization** (cell 8): `LAW_REF_RE` (Art./Artikel with article + bis/ter/quater suffixes, Abs./Bst./Ziff. unit chains), `WS_RE`. Used to detect query mentions and generate variant expansions; never adds final IDs outside the source inventory.
4. **Section 3 - Build compact DuckDB indexes** (cell 10): `segment_record(...)` parses `citation`, `pattern`, `family`, `subfamily`, `origin_mask`, `segments.{article,section,...}` into rows that feed `law_segments` and `court_segments`; optional `citation_edges` (from links JSONs) and citation-text tables; parquet snapshots written for quick reload.
5. **Section 4 - Source inventory + gold labels** (cell 12): Build `LAW_SOURCE`, `COURT_SOURCE`, `ALL_SOURCE` from DuckDB; `split_gold(...)` parses gold strings; load queries from the requested split.
6. **Section 5 - Choice card generation** (cell 14): SQL-derived cards (`law_codes`, `law_unit_chains`, `court_families`, `bge_divisions`, `docket_prefixes`, `route_choices`, `time_choices`) exported to `artifacts/choice_cards.json`. Planner may only select IDs from these cards.
7. **Section 6 - Planner** (cell 16): Switch-only planner (no candidate output). Heuristic safety set `COMMON_SAFETY_LAW_CODES` covers BGG, BV, StBOG, ZGB, OR/CO, StPO/CPP, StGB/CP, ZPO/CPC, ATSG, IVG, KVG, UVG, SchKG, IPRG, DBG, AIG. `PREFIX_HINTS` map English query keywords (detention, custody, collusion, criminal, offence/offense, insurance, ...) to docket prefixes. Uses the Qwen3-32B planner if `USE_PLANNER_LLM=True`, otherwise a broad deterministic fallback.
8. **Section 7 - Candidate expansion helpers** (cell 18): `sql_quote_list`, `fetch_set`, `create_temp_citation_table`, and the family of `selected_*_candidates(...)` SQL helpers that materialize per-route candidate sets.
9. **Section 8 - Recall audit + budgeting** (cell 20): `audit_rows`, `dropped_rows`, `split_gold_by_funnel`, `recall_of`, and `audit_stage(...)` record per-stage recall vs gate, drop reasons, and gold dropped per query.
10. **Section 9 - Hard-filter funnel runner** (cell 24): `run_law_funnel(...)` walks stages `L0_all_laws -> L1 (law-code filter) -> ...` with recall guards that revert a stage when its post-filter recall falls below the gate (`RECALL_GUARD_ON_VAL`).
11. **Section 9b - Soft-rank funnel (recall-aware top-K)** (cell 23): Whole-corpus score-and-cap variant. SQL templates `LAW_SCORE_SQL_TEMPLATE` (and the court counterpart) award `+1000` for explicit query-mentioned citations and additive boosts for planner-aligned `law_code`, prefix, family, route, and train-prior pins. Top-K per side at the configured `SOFT_LAW_TARGET / SOFT_COURT_TARGET`; `SOFT_PROBE_STEPS = [1.0]` keeps a single non-leakage probe. Reported: `Train-prior pin (legit): law=1,876, court=0`.
12. **Section 9c - Gold-bank candidate funnel** (cell 27): Closed-vocabulary diagnostic that reads `data_insights/{split}_gold_citations.json` directly into the candidate pool. Trivial recall 1.0 on val; leakage on test if `["train","val"]` is used. Kept OFF by `USE_GOLD_BANK_CANDIDATES = False`.
13. **Section 10 - Run candidate funnel** (cell 28): Iterates `queries`, runs planner + soft funnel + (optional) gold-bank, writes `{split}_candidate_sets.jsonl` and `{split}_planner_outputs.jsonl`, accumulates `audit_rows`, `dropped_rows`, and per-query `candidate_rows`.
14. **Section 11 - Recall diagnostics** (cell 30): Builds `audit_df`, `dropped_df`, `summary_df` and writes `artifacts/stage_recall_audit.csv`, `artifacts/dropped_gold_diagnostics.csv`, `artifacts/{split}_candidate_summary.csv`. Prints aggregate recall stats and flags queries below `FINAL_MIN_QUERY_RECALL`.
15. **Section 12 - Inspect first dropped gold citations** (cell 32): Displays `dropped_df.head(100)` and a grouped count by `funnel / first_failed_stage / drop_reason / stage_status`; used to decide whether to tighten or relax a filter.
16. **Section 13 - Optional reranker skeleton** (cell 34): Lazy loader for `Qwen/Qwen3-Reranker-8B` (`AutoModelForCausalLM`, left padding); gated behind `USE_RERANKER=False`. Reranker can only score existing candidate IDs; it cannot invent citations.
17. **Section 14 - Export submission candidate diagnostics** (cell 36): For `RUN_SPLIT == "test"` only, writes a per-query diagnostic with `candidate_count`, `law_candidate_count`, `court_candidate_count` to `ART_DIR`. Final prediction selection deferred to reranking/threshold calibration.

## Results

Per-query planner output and pool size on `val` (split=`val`, 10 queries):

```
=== val_001 ===
gold law/court: 19 12
plan: {'law_codes': ['StPO'], 'law_unit_chains': ['article_only'], 'court_families': [], 'bge_divisions': [], 'docket_prefixes': ['2C'], 'routes': ['bge_base_route', 'law_code_route', 'statute_citing_cases'], 'time_mode': 'no_time_filter', 'law_budget': 2000, 'court_budget': 2000, 'notes': "Art. 221 Abs. 1 lit. b StPO permits pre-trial detention extensions..."}
candidate_count: 4000 law_n: 3119 court_n: 2000 recall: 0.38095238095238093 law: 0.8421052631578947 court: 0.0 law_status: ok court_status: ok

=== val_002 ===
gold law/court: 20 3
plan: {'law_codes': ['AVIG', 'IVG'], 'law_unit_chains': ['paragraph'], ..., 'docket_prefixes': ['1C','5A','6B'], 'routes': ['law_code_route','same_decision_neighbors','statute_citing_cases'], 'law_budget': 2, 'court_budget': 3, ...}
candidate_count: 1879 law_n: 1876 court_n: 3 recall: 0.25 law: 0.45 court: 0.0 law_status: ok court_status: ok

=== val_003 ===
gold law/court: 24 7
plan: {'law_codes': ['BV','StPO'], 'docket_prefixes': ['2C'], 'routes': ['article_group_route','law_code_route','same_decision_neighbors','statute_citing_cases'], ...}
candidate_count: 4000 law_n: 3710 court_n: 2000 recall: 0.3829787234042553 law: 0.75 court: 0.0 law_status: ok court_status: ok

=== val_004 ===
gold law/court: 9 0
plan: {'law_codes': ['ZGB'], 'docket_prefixes': ['2C'], 'law_budget': 1, 'court_budget': 3, ...}
candidate_count: 1879 law_n: 1876 court_n: 3 recall: 0.3 law: 0.3333333333333333 court: None law_status: ok court_status: ok

=== val_005 ===
gold law/court: 6 0
plan: {'law_codes': ['StGB','StPO','ZGB'], 'docket_prefixes': ['6B'], 'law_budget': 5, 'court_budget': 5, ...}
candidate_count: 1881 law_n: 1876 court_n: 5 recall: 0.18181818181818182 law: 0.3333333333333333 court: None law_status: ok court_status: ok

=== val_006 ===
gold law/court: 11 0
plan: {'law_codes': ['StGB','StPO','ZGB','ZPO'], 'routes': ['law_code_route','regex_unknown_safety','statute_citing_cases'], 'law_budget': 4, 'court_budget': 10, ...}
candidate_count: 1886 law_n: 1876 court_n: 10 recall: 0.3333333333333333 law: 0.5454545454545454 court: None law_status: ok court_status: ok

=== val_007 ===
gold law/court: 15 0
plan: {'law_codes': ['ZGB'], 'docket_prefixes': ['5A'], 'law_budget': 3, 'court_budget': 2, ...}
candidate_count: 1884 law_n: 1882 court_n: 2 recall: 0.5789473684210527 law: 0.7333333333333333 court: None law_status: ok court_status: ok

=== val_008 ===
gold law/court: 20 4
plan: {'law_codes': ['StGB','VStrR','ZGB'], 'docket_prefixes': ['2C','6B'], 'law_budget': 3, 'court_budget': 1, ...}
candidate_count: 1877 law_n: 1876 court_n: 1 recall: 0.27586206896551724 law: 0.4 court: 0.0 law_status: ok court_status: ok

=== val_009 ===
gold law/court: 11 2
plan: {'law_codes': ['AHVG','SchKG','ZGB'], 'docket_prefixes': ['2C'], 'routes': ['docket_base_route','law_code_route','same_decision_neighbors','statute_citing_cases'], 'law_budget': 3, 'court_budget': 2, ...}
candidate_count: 1878 law_n: 1876 court_n: 2 recall: 0.2857142857142857 law: 0.36363636363636365 court: 0.0 law_status: ok court_status: ok

=== val_010 ===
gold law/court: 14 4
plan: {'law_codes': ['FinfraV','VZV','ZGB','ZPO'], 'docket_prefixes': ['2C'], 'routes': ['article_group_route','law_code_route','statute_citing_cases'], 'law_budget': 4, 'court_budget': 3, ...}
candidate_count: 1879 law_n: 1876 court_n: 3 recall: 0.08 law: 0.14285714285714285 court: 0.0 law_status: ok court_status: ok
```

Aggregate (Section 11 / `summary_df`):

```
  query_id  candidate_count  law_candidate_count  court_candidate_count    recall  law_recall  court_recall
0  val_001             4000                 3119                   2000  0.380952    0.842105           0.0
1  val_002             1879                 1876                      3  0.250000    0.450000           0.0
2  val_003             4000                 3710                   2000  0.382979    0.750000           0.0
3  val_004             1879                 1876                      3  0.300000    0.333333           NaN
4  val_005             1881                 1876                      5  0.181818    0.333333           NaN
5  val_006             1886                 1876                     10  0.333333    0.545455           NaN
6  val_007             1884                 1882                      2  0.578947    0.733333           NaN
7  val_008             1877                 1876                      1  0.275862    0.400000           0.0
8  val_009             1878                 1876                      2  0.285714    0.363636           0.0
9  val_010             1879                 1876                      3  0.080000    0.142857           0.0
```

Stage rollup:

```
  funnel        stage status  queries  mean_after_count  min_recall_after  mean_recall_after  total_dropped
0  court  C_soft_topk     ok       10             402.9          0.000000           0.000000             32
1    law  L_soft_topk     ok       10            2184.3          0.142857           0.489405             70
```

Headline numbers:

```
Mean recall:       0.3049606342609007
Min recall :       0.08
Mean law recall:   0.4894053315105946
Mean court recall: 0.0
Queries below final gate: <all 10 rows of summary_df re-printed>
```

Cell 32 (dropped-gold inspection): `No gold citations were dropped by audited stages, or no gold labels are available.`

Cell 34 (reranker skeleton): `Reranker disabled. Set USE_RERANKER=True after candidate recall passes.`

Outputs saved by the run:

- `artifacts/choice_cards.json`
- `artifacts/val_candidate_sets.jsonl`
- `artifacts/val_planner_outputs.jsonl`
- `artifacts/stage_recall_audit.csv`
- `artifacts/dropped_gold_diagnostics.csv`
- `artifacts/val_candidate_summary.csv`

Cell 28 also surfaces a Hugging Face Hub warning: `Error while fetching HF_TOKEN secret value from your vault: 'Requesting secret HF_TOKEN timed out...'` followed by `Warning: You are sending unauthenticated requests to the HF Hub` and a deprecation notice `[transformers] torch_dtype is deprecated! Use dtype instead!` during planner model download.

## Summary

This is the pre-v7 "early funnel" prototype that enforces a strict closed-vocabulary contract: every final candidate must come from the `citation` columns of `laws_de.csv` or `court_considerations.csv` (LAW_SOURCE=175,933, COURT_SOURCE=1,516,398), and the planner LLM, regex layer, and any reranker are only permitted to pick routes/filters, never invent IDs. The notebook builds compact DuckDB indexes, derives corpus-grounded choice cards (2,061 law codes, 44 docket prefixes, 9 routes), and offers three candidate paths: a hard-filter funnel with recall guards, a soft-rank score+top-K funnel (`SOFT_PROBE_STEPS=[1.0]` to avoid val K-growth leakage), and an OFF-by-default gold-bank diagnostic. The val run with the Qwen3-32B planner produces 1,877-4,000 candidates per query but recall remains weak: mean overall recall 0.3050, min 0.08, mean law recall 0.4894, mean court recall 0.0 (the court funnel returns only 1-10 IDs per query in 8 of 10 queries and never recovers a gold court citation). The stage rollup shows `L_soft_topk` and `C_soft_topk` both pass `status=ok` because the recall guards are non-strict, yet 70 gold law and 32 gold court citations are still dropped in aggregate. The reranker (Qwen3-Reranker-8B) is wired but disabled, and the notebook explicitly defers final prediction selection to a later reranking/threshold calibration step once candidate recall is acceptable.
