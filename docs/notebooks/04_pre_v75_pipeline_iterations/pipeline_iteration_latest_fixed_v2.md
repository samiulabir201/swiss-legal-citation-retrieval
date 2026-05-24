# pipeline_iteration_latest_fixed_v2.ipynb

**Path:** notebooks/04_pre_v75_pipeline_iterations/pipeline_iteration_latest_fixed_v2.ipynb

## Configuration

### Runtime
- Platform: Google Colab (Drive-mounted), 129 cells total (markdown + code).
- Project root: `/content/drive/MyDrive/swiss_law`.
- HF cache forced to `PROJECT_ROOT/model` via `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`.
- Working dir: `os.chdir(PROJECT_ROOT)`; project root prepended to `sys.path`.

### Editable runtime settings (cell 21)
- `SPLIT = "val"`
- `BACKEND = "local"`
- `STAGE3_GOLD_PRIOR_MODE = "off"` (alternatives: "train", "trainval")
- `STAGE2_MAX_ITERATIONS = 4`
- `STAGE4_SKIP_DIRECT_GEN = True` (direct-gen was 60% hallucinated on val)
- `STAGE4B_TOP_N = None` (Stage 4b cross-encoder disabled)
- `STAGE5_THRESHOLD = None` (tuned on val)
- `STAGE4_RANK_POOL_SIZE = 150` (raised from 80 because Stage 3 emits 8K-wide pools)
- `STAGE3_BUILD_STATUTE_TO_BGE = True`
- `STAGE3_STATUTE_TO_BGE_TOPK = 5`
- `STAGE3_USE_NUMBERED_INDEXES = True`
- `STAGE3_STATUTE_TO_NUMBERED_TOPK = 3` (down from 5)
- `STAGE3_NUMBERED_TO_BGE_TOPK = 3`
- `STAGE3_NUMBERED_COCITATION_TOPK = 1` (down from 3; was too noisy)
- `STAGE6_NUMBERED_BM25_QUANTILE = 0.90` (was 0.75)
- `STAGE6_NUMBERED_REQUIRE_GRAPH_SOURCE = True`
- `STAGE6_ABS_COLLAPSE = True`
- `STAGE6_ABS_MAX_VARIANTS = 2`
- `STAGE6_MIN_K = 10`
- `STAGE6_MAX_K = 45`
- `STAGE6_USE_ADAPTIVE_K = True`
- `STAGE6_PROCEDURAL_FLOOR = True`
- Stage 6 score source: `bm25_plus_llm` with `w_llm = 0.5`.
- Stage 6 K strategy: `score_gap` with `frac_of_max=0.55`, `elbow_drop=0.12`.

### Default pipeline config (cell 22; mirrors `configs/default.json`)
```json
{
  "stage1_backend": "local",
  "bm25_top_k": 100,
  "graph_top_k": 50,
  "mas_max_iterations": 4,
  "mas_bm25_top_k": 30,
  "reranker_backend": "local",
  "reranker_batch_size": 20,
  "confidence_weights": {
    "explicit_from_query": 0.40,
    "bm25_top10": 0.25,
    "citation_graph": 0.15,
    "llm_reranker": 0.15,
    "llm_direct_gen": 0.05
  },
  "threshold": 0.15
}
```

### Scoring (cell 48 / `scoring/confidence.py`)
- DEFAULT_WEIGHTS: `explicit_from_query 0.40`, `bm25_top10 0.15`, `bm25_initial 0.10`, `citation_graph 0.10`, `llm_reranker_tier3 0.25`, `llm_reranker 0.15`, `llm_direct_gen 0.05`, `llm_stage1 0.05`, `llm_stage1_procedural 0.05`, `co_citation 0.03`.
- `BM25_CONTINUOUS_WEIGHT = 0.10`.
- `CROSS_ENCODER_WEIGHT = 0.18` (replaces binary reranker signals when CE scores available).
- `MULTI_SIGNAL_BONUS = 0.10` (added when 3+ independent signals agree).

### Models
- `QWEN3_32B_HF_ID = "Qwen/Qwen3-32B"` (full bf16, ~64 GB VRAM, loaded once as singleton).
- `QWEN3_RERANKER_ID = "Qwen/Qwen3-Reranker-4B"` (Stage 4b, disabled in this run).
- Generation defaults: `temperature=0.6`, `top_k=20`, `top_p=0.95`, `max_new_tokens=2048`, `enable_thinking=False` (Stage 1 uses True).
- `device_map="auto"`, `torch_dtype=torch.bfloat16`.

### Libraries (cell 5/6)
- `pip install -qU deep-translator rank-bm25 scikit-learn`.
- Imports: `numpy`, `pandas`, `torch`, `deep_translator.GoogleTranslator`, `rank_bm25.BM25Okapi`, `scipy.sparse`, `tqdm`, `transformers (AutoTokenizer, AutoModelForCausalLM)`.

### Hardware
- GPU expected (H100 80 GB recommended for Qwen3-32B bf16 ~64 GB VRAM).
- Stages 0/3/5/6 CPU only; ~16 GB RAM for BM25 build.

## Data

Base directories (cell 6 / 19):
- `PROJECT_ROOT = /content/drive/MyDrive/swiss_law`
- `RAW_DATA_DIR = PROJECT_ROOT/data`
- `INDEX_DIR    = PROJECT_ROOT/index`
- `CHECKPOINTS_DIR = PROJECT_ROOT/checkpoints`
- `SUBMISSIONS_DIR = PROJECT_ROOT/submissions`
- `MODELS_DIR  = PROJECT_ROOT/model`

Raw competition data (`data/`):
- `data/train.csv` (1,139 rows; mean 4.1 cites/q; 99% Law / 1% BGE)
- `data/val.csv` (10 rows; min=10, max=47, mean=25.1 cites/q; 59% Law / 27% BGE / 13% Numbered)
- `data/test.csv` (40 rows)
- `data/court_considerations.csv` (2.26 GB present)
- `data/laws_de.csv` (175,933 rows; SR-style codes, FYI only)
- `data/sample_submission.csv`

Knowledge base / index artifacts (all under `INDEX_DIR`, all present):
- `corpus.parquet` (`LAW_CORPUS_PARQUET`)
- `laws_knowledge_base.jsonl` (`KB_JSONL`, 179,641 records)
- `bm25_v2_index.pkl` (+ `bm25_v2_ids.pkl`); chunked parts under `bm25_v2_index_parts/bm25_v2_index_parts/`
- `citation_graph.pkl`
- `citation_lookup.pkl`
- `citation_signal_lookup.pkl`
- `reference_graph_v2.pkl`
- `gold_cocitation_prior_train.pkl`, `gold_cocitation_prior_trainval.pkl`
- `statute_to_bge.pkl` (3,840 entries)
- `statute_to_numbered.pkl` (4,991 entries)
- `numbered_to_statutes.pkl` (80,569 entries)
- `numbered_to_bge.pkl` (67,513 entries)
- `numbered_cocitation.pkl` (81,709 entries)
- `token_law_freq.json`
- `citation_signal_schema_v2.json`
- `statute_signals.jsonl`, `case_signals.jsonl`
- `query_translations_trainval.json` (read-only source; copied to `CHECKPOINTS_DIR`)

Stage outputs (under `CHECKPOINTS_DIR`):
- `stage1_val.json`, `stage2_val.json`, `stage3_val.json`, `stage4_val.json`, `stage5_val.json`
- `bm25_scores_val.json`, `shadow_val.csv`, `configs/threshold.json`

Submissions:
- `submissions/submission_val.csv`

Inlined source modules originally at `/mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline/`:
- `agent/llm_backend.py`, `agent/verifier.py`, `agent/prompts/mas_prompts.py`
- `retrieval/sparse_retriever.py`, `retrieval/graph_retriever.py`, `retrieval/explicit_citations.py`, `retrieval/bm25_artifact`, `retrieval/citation_graph`
- `scoring/confidence.py`
- `indexing/build_lookup_tables.py`
- `stage1_query_analysis.py`, `stage2_mas_retrieval.py`, `stage3_graph_expansion.py`, `stage4_llm_reranker.py`, `stage5_verify_and_score.py`, `submit.py`

## Pipeline

### Stage A: Setup and Config (cells 0-27)
1. Mount Google Drive; pip install dependencies.
2. Set HF cache env vars before transformers import.
3. Standard + third-party imports; resolve all path constants.
4. Inline 11 utility cells (cells 8-18) for project modules (no CLI entrypoints).
5. Define editable runtime settings (SPLIT, BACKEND, stage knobs).
6. Define notebook helper wrappers `run_stageN(...)`, `get_checkpoint_path(...)`, `get_submission_path(...)`.

### Stage B: Pre-run Diagnostics (cells 28-35)
Seven read-only checks before any pipeline cell:
1. Data distribution (rows, types, cites/q).
2. KB integrity (record count, top codes, artifact existence).
3. Gold format audit (Abs / lit / bare Art fractions).
4. Procedural-boilerplate census across val gold.
5. BGE reachability scan over `court_considerations.csv`.
6. Test-vs-val domain overlap (law-code abbreviations).
7. Citation-graph edge inventory by type.

### Stage 1: Query Analysis (cells 49-61)
- Calls `run_batch_stage1(SPLIT="val", backend="local")`.
- Loads Qwen3-32B (singleton, bf16), BM25 sparse index.
- Output `checkpoints/stage1_val.json` with: `explicit_citations`, `candidate_articles` (10/q), `candidate_cases` (10/q), `search_queries_en` (~4/q), `search_queries_de` (5/q), `hyde_documents_de` (2/q), `law_areas`, `legal_issues`, `bm25_probe`.
- 6 diagnosis cells.

### Stage 2: MAS Sparse Retrieval (cells 62-76)
- Calls `run_batch_stage2(max_iterations=4)`.
- Loads BM25 index (216 chunks, 2,156,832 docs x 822,251 terms, ~212M nz).
- Iterative Planner -> Agent (REWRITE / SUPPLEMENT / DECOMPOSE / SUPPORTIVE / CROSSREF) -> BM25, batched 3 queries per batch.
- Writes `candidate_pool`, `sources` (`{cite: [source_labels]}`), `bm25_scores` to `stage2_val.json`.
- 6 diagnosis cells.
- Co-occurrence index loader (statute-to-BGE, statute-to-Numbered, numbered-to-statutes, numbered-to-BGE, numbered-cocitation) loaded from `INDEX_DIR`.

### Stage 3: Citation Graph Expansion (cells 77-87)
- Calls `run_batch_stage3(gold_prior_mode="off")`.
- Loads `citation_graph.pkl` (103,125 art->case links, 914,658 case->art links) + `article_to_abs` (43,066 entries) for sibling expansion.
- Five expansion channels per query: graph-cases, tm-BGE, tm-numbered, num->BGE, num-cocit.
- Writes `expanded_pool` to `stage3_val.json`.
- 5 diagnosis cells (incl. BGE safety-net, Abs-level coverage audit).

### Stage 4: LLM Reranker (cells 88-100)
- Calls `run_batch_stage4(skip_direct_gen=True, pool_size=150)`.
- Reuses Qwen3-32B singleton; pool capped at 150 (BM25-top + procedural + explicit + Stage-3 cases).
- Single-pass continuous 0-10 score per candidate (replaces tier 0/1/2/3 system).
- Writes `reranked`, `reranked_tiers`, `direct_gen`, `multi_signal`, `stage1_passthrough`, `all_citations`, etc. to `stage4_val.json`.
- 5 diagnosis cells (counts, hallucination rate, tier collapse, gold dropout, top-K recall).

### Stage 5: Verify, Score, Threshold Tune (cells 101-111)
- Calls `run_batch_stage5(threshold=None)` (val-mode threshold sweep).
- Loads citation `Verifier` + BM25 (continuous scoring) + cached BM25 scores per query.
- 5A normalize+verify (drops hallucinations, applies `_ABBREV_CASING` corpus->gold map).
- 5B composite confidence (signal weights + BM25 continuous + multi-signal bonus); cross-encoder absent so falls back to binary reranker tiers.
- 5C sweep 0.05-0.95 threshold; persists optimum to `configs/threshold.json`.
- Writes `stage5_val.json` + `submissions/submission_val.csv`.
- 6 diagnosis cells (global P/R/F1 per type, query-level F1, FP/FN error analysis, Abs flooding, procedural-prior potential).

### Stage C: Submission Validation (cells 112-116)
- `validate_submission(submission_val.csv)` checks columns + IDs against `test.csv`.

### Stage D: Rollup + Diagnostics (cells 117-122)
- `run_all_diagnostics(split=SPLIT)` runs D1 (ranking ceiling), D2 (adaptive-K oracle), D4 (CE calibration), D6 (build shadow_val.csv).
- D3 (BGE-M3 recall) and D5 (LLM ranker smoke) noted as GPU-side, run separately.

### Stage 6: Adaptive-K + Procedural Floor (cells 123-126)
- `run_stage6_adaptive_k(split=SPLIT)`.
- Score = `bm25_plus_llm` (BM25 normalized + 0.5 * llm_score/10), bypassing Stage 5 composite.
- Per-query K predictor (regressor on shadow-val); floor=10, ceiling=45.
- Numbered-cite suppression: BM25 quantile >= 0.90 AND must come from non-text-mined graph source.
- Abs-collapse: keep at most 2 Abs variants per Art root unless explicit.
- Procedural force-include gated on query trigger.
- Overwrites `submissions/submission_val.csv`.

### Stage E: Post-mortem notes (cells 127-128)
- End-to-end run instructions and a HyDE post-mortem (DE/HyDE variants empirically dead on this BM25-over-statute corpus).

## Results

### Pre-run diagnostics
1/7 DATA DISTRIBUTION
- val (10 rows): min=10 max=47 mean=25.1 median=22.0 cites/q; Law=149 (59%), BGE=69 (27%), Numbered=33 (13%).
- train (1,139): mean=4.1; Law=4602 (99%), BGE=57 (1%).
- test: 40 rows.
- Warning printed: val (25.1/q) > 2x train (4.1/q).

2/7 KB INTEGRITY
- 179,641 KB records; top codes `{EFZ:6500, OR:3667, EBA:2469, ZGB:2405, StPO:1306, StGB:1244, TSV:1094, VTS:1023, SchKG:919, ZV:874}`. KB ok.
- All graph artifacts present.

3/7 GOLD FORMAT AUDIT
- Abs: 121 (81%), bare_Art: 28 (19%) of val Art-citations.

4/7 PROCEDURAL CENSUS
- Only `Art. 100 Abs. 1 BGG` appears in >=2 val queries (9 queries; never in query text).

5/7 BGE REACHABILITY
- 54/54 val-gold BGE stems present in corpus.

6/7 TEST-vs-VAL DOMAIN OVERLAP
- Codes in test but NOT covered by val gold: `['CC', 'CO', 'MSchG', 'PrHG', 'SVG', 'UVG']`.

7/7 CITATION GRAPH INVENTORY
- citation_graph.pkl: 2,201,708 nodes; statute->BGE = 0; statute->num = 0; BGE->BGE = 0; BGE->stat = 0; other = 3,897,617.
- "** NO statute->BGE edges. BGE recall via graph expansion is impossible." (Mitigated by text-mined pickles loaded later.)

### Stage 1 (val)
- Qwen3-32B load + first batch generation: 763.5s (152.7s/query). Two batches of 5.
- explicit_citations: min=0 max=3 mean=0.9; candidate_articles: 10/q; candidate_cases: 10/q; search_queries_en: ~4/q; search_queries_de: 5/q; hyde_documents_de: 2/q (~99 tokens); law_areas: mean=1.8; legal_issues: mean=2.8.
- Explicit-extraction fidelity: 6/6 = 100%.
- Candidate-article validity: 100/100 = 100%.
- Routing precision = 0.361; routing recall = 0.955 (misses: val_003 missing BV; val_007 missing StGB).
- HyDE drift: 3/10 queries suspicious (val_004, val_007, val_008).
- Stage 1 free-recall ceiling: 126/251 = 50.2% (Law 55.0%, BGE 47.8%, Numbered 33.3%).
  - Per-q hits/gold: val_001 20/42, val_002 20/36, val_003 17/47, val_004 6/10, val_005 7/11, val_006 9/18, val_007 11/19, val_008 12/29, val_009 11/14, val_010 13/25.

### Stage 2 (val)
- BM25 load: 216 chunks in ~90 s; sparse 2,156,832 x 822,251 (212M nz). 171,654 law + 1,985,178 court.
- 4 iterations x batches of 3 queries; planner most often picks SUPPORTIVE.
- Final candidate pool: min=976 max=1430 mean=1183.5; by type Numbered=6947 (59%), Law=3258 (28%), BGE=833 (7%), Other=797 (7%).
- Per-type recall in pool: Law 117/149 = 78.5%, BGE 48/69 = 69.6%, Numbered 26/33 = 78.8%.
- R@k (bm25-score-sorted approx):
  - K=10  Law 44.3%  BGE 24.6%  Numbered 6.1%
  - K=50  Law 57.7%  BGE 60.9%  Numbered 66.7%
  - K=100 Law 57.7%  BGE 60.9%  Numbered 66.7%
  - K=200 Law 66.4%  BGE 66.7%  Numbered 75.8%
  - K=500 Law 67.8%  BGE 66.7%  Numbered 75.8%
  - K=1000 Law 71.1% BGE 68.1%  Numbered 78.8%
- Single- vs multi-query ablation (BM25 R@100): EN-raw alone Law 39.6% / BGE 65.2% / Numbered 75.8%; single DE 3.4% / 0% / 9.1%; all DE 13.4% / 1.4% / 9.1%; all DE+HyDE 14.1% / 2.9% / 15.2%.
- Dead query variants: DE_4, EN_3, HYDE_0, HYDE_1 (>=10 runs, <=1 gold hit).
- MAS action source totals: bm25_initial 7513, mas_supportive 3199, mas_crossref 934, mas_rewrite 351, mas_supplement 267, llm_stage1 200, llm_stage1_procedural 200, mas_decompose 178, bm25_top10 82, explicit_from_query 9. Avg overlap (cite reached by >1 action) = 0.07.

### Stage 3 (val)
- Co-occurrence indexes loaded: statute->BGE 3,840 / statute->Numbered 4,991 / numbered->statutes 80,569 / numbered->BGE 67,513 / numbered->co-cited 81,709.
- Per-query expansion (Stage 2 -> Stage 3 totals):
  - val_001: 976 -> 3721; val_002: 1122 -> 3983; val_003: 1102 -> 4431; val_004: 1066 -> 4568; val_005: 1229 -> 4703; val_006: 1105 -> 4969; val_007: 1430 -> 5792; val_008: 1338 -> 6083; val_009: 1040 -> 4154; val_010: 1427 -> 6200. Total 11,835 -> 48,604 (+36,769 from graph).
- Per-type recall post-expansion: Law 135/149 = 90.6%, BGE 58/69 = 84.1%, Numbered 26/33 = 78.8%.
- Graph reachability of val-gold BGEs: 1-hop 0/56 = 0%, 2-hop 0/56 = 0%; via text-mined statute_to_bge 26/56 = 46.4%.
- Procedural-cite coverage: 2/2 gold procedural events (Art. 82 BGG, Art. 113 BGG) covered by Stage 3.
- BGE safety-net: 0/5 missed BGEs recoverable via additional statute-text mining.
- Abs-level audit: exact 111/121 (92%), parent-only 8/121 (7%), missed 2/121 (2%). Abs-flooding (>=3 Abs of one Art in candidates) = 1284 events across 10 queries.

### Stage 4 (val)
- Pool=150 for every query; per-query ranker latencies 82.5-255.9 s.
- Ranker high/med counts: val_001 78 (47/31), val_002 99 (45/54), val_003 125 (125/0), val_004 66 (58/8), val_005 129 (84/31), val_006 148 (146/2), val_007 134 (134/0), val_008 110 (110/0), val_009 122 (49/73), val_010 149 (55/94).
- Counts: reranked/q min=66 max=149 mean=116; direct-gen=0 (skipped).
- Tier distribution (1160 scored cites): T10 6 (0.5%), T9 26 (2.2%), T8 44 (3.8%), T7 387 (33.4%), T6 390 (33.6%), T5 261 (22.5%), T4 24 (2.1%), T3 8 (0.7%), T2 14 (1.2%).
- Gold dropout Stage 3 -> Stage 4: 6/219 (val_002 Art. 56 Abs. 1 ATSG; val_003 BGE 143 IV 330 E. 2.1, BGE 145 IV 99 E. 3.1; val_007 Art. 292 StGB; val_008 BGE 149 IV 42 E. 3.5; val_010 BGE 134 III 151 E. 2.4).
- Stage 4 full-pool recall (Stage 5 ceiling): Law 133/149 = 89.3%, BGE 54/69 = 78.3%, Numbered 26/33 = 78.8%.

### Stage 5 (val)
- BM25 precompute cache built (10 queries, 44-80 s each); 22,816 verified, 2,646 dropped.
- No CE scores (`reranker_scores_val.json` not present); fell back to binary reranker.
- Threshold sweep optimum: threshold = 0.65, macro-F1 = 0.4820.
- Per-query F1 @ 0.65:
  - val_001 F1=0.6176 (pred 26, gold 42), val_002 0.4348 (10/36), val_003 0.4000 (18/47), val_004 0.3478 (13/10), val_005 0.3871 (20/11), val_006 0.4348 (5/18), val_007 0.4444 (17/19), val_008 0.5769 (23/29), val_009 0.5641 (25/14), val_010 0.6122 (24/25).
- Submission summary: queries=10, avg cites=18.1, max=26, zero-pred queries=0.
- Per-type @ 0.65: Law P=0.946 R=0.470 F1=0.628 (tp=70 fp=4 fn=79); Numbered P=0.179 R=0.364 F1=0.240 (tp=12 fp=55 fn=21); BGE P=0.650 R=0.377 F1=0.477 (tp=26 fp=14 fn=43).
- Errors: Misses Law 79 / BGE 43 / Numbered 21. FPs Numbered 55 / BGE 14 / Law 4.
- Abs-flooding in final predictions: 0 events.
- Procedural-prior potential: 2 gold-procedural TP catchable but introduces 148 FPs => selective domain-conditioned prior required.

### Submission validation
- `validate_submission(submission_val.csv)` -> "ERROR: Missing 40 test query IDs"; "Submission valid: False" (val-mode submission, fails test-ID schema by design).

### Diagnostics (D1, D2, D4, D6)
- D1 ranking ceiling: in_pool counts per query: val_001 42/42, val_002 32/36, val_003 35/47, val_004 8/10, val_005 10/11, val_006 14/18, val_007 16/19, val_008 26/29, val_009 14/14, val_010 22/25. Median rank 5.5-21.5; max rank 76-1593.
- D2 oracle-K: BM25 oracle-K macro-F1 = 0.6786; Stage-5 oracle-K macro-F1 = 0.5168.
- D4 CE calibration: skipped (no `reranker_scores_val.json`).
- D6 shadow_val: 253 train rows with gold>=6; wrote 150 rows to `checkpoints/shadow_val.csv` (gold-size mean=12.5, median=11, min=6, max=44).

### Stage 6 (val, final submission)
- Score source: `bm25_plus_llm` (w_llm=0.5).
- Post-processing: 18,876 -> 11,422 candidates after Abs-collapse + Numbered suppression (quantile=0.9, require_graph=True).
- K strategy: `score_gap` (frac_of_max=0.55, elbow_drop=0.12).
- Per-query (K / preds / gold / P / R / F1):
  - val_001 24/24/42 P=1.000 R=0.571 F1=0.727
  - val_002 25/25/36 P=1.000 R=0.694 F1=0.820
  - val_003 18/23/47 P=0.870 R=0.426 F1=0.571
  - val_004 10/13/10 P=0.462 R=0.600 F1=0.522
  - val_005 10/10/11 P=0.600 R=0.545 F1=0.571
  - val_006 10/10/18 P=0.900 R=0.500 F1=0.643
  - val_007 10/10/19 P=1.000 R=0.526 F1=0.690
  - val_008 15/15/29 P=1.000 R=0.517 F1=0.682
  - val_009 10/10/14 P=1.000 R=0.714 F1=0.833
  - val_010 14/14/25 P=1.000 R=0.560 F1=0.718
- **MACRO F1 = 0.6777, P = 0.8831, R = 0.5655.**
- Mean predictions per query: 15.4 (min=10, max=25). Saved adaptive-K submission to `submissions/submission_val.csv`.

### HyDE post-mortem (cell 128, summary table)
| Variant | Gold hits | Avg/run |
|---------|-----------|---------|
| EN_raw | 121 | 12.10 |
| EN_0 | 70 | 7.00 |
| DE_2 (best German) | 11 | 1.10 |
| DE_0 / DE_3 / DE_4 / HYDE_1 | <=5 | <=0.5 |

Decision recorded: keep HyDE in Stage 1 outputs for debugging only; do not route into BM25.

## Summary

The notebook orchestrates an 6-stage Swiss-citation pipeline (Query Analysis -> MAS sparse retrieval -> Citation-graph expansion -> LLM reranker -> Verify+score+threshold -> Adaptive-K) on a 10-query val split, using Qwen3-32B (bf16) for LLM stages and BM25 over a 2.16 M-doc combined law+court index for retrieval. Stage 5's global threshold sweep peaked at macro-F1=0.4820 (threshold=0.65), driven down by Numbered-citation false positives (P=0.179) and BGE recall (R=0.377). Stage 6's adaptive-K + Numbered suppression + Abs-collapse + bm25_plus_llm scoring lifted macro-F1 to 0.6777 (P=0.8831, R=0.5655) without re-running any upstream stage. Diagnostics showed the upstream pool was healthy (Stage 3 per-type recall 90.6/84.1/78.8%; D2 BM25 oracle-K F1=0.6786), the precomputed `citation_graph.pkl` had zero statute->BGE edges (mitigated by text-mined `statute_to_bge.pkl` covering 46.4% of val-gold BGE), HyDE/DE query variants were largely dead in BM25, and direct-gen was kept off due to prior hallucination. The biggest remaining gaps are Numbered-citation precision and BGE recall, which Stage 6 cuts but cannot eliminate; D2 oracle-K of 0.6786 sets the realistic ceiling for any ranker on this pool.
