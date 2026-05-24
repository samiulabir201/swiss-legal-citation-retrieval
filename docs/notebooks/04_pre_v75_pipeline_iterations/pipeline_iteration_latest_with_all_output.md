# pipeline_iteration_latest_with_all_output.ipynb

**Path:** notebooks/04_pre_v75_pipeline_iterations/pipeline_iteration_latest_with_all_output.ipynb

## Configuration

**Runtime environment:** Google Colab with Drive mount at `/content/drive`. `PROJECT_ROOT = /content/drive/MyDrive/swiss_law`. Working dir set via `os.chdir(PROJECT_ROOT)`.

**Models:**
- `QWEN3_32B_HF_ID = "Qwen/Qwen3-32B"` (full bf16, ~64 GB VRAM, used for Stage 1, Stage 2 planner/agents, Stage 4 reranker, direct-gen).
- `QWEN3_RERANKER_ID = "Qwen/Qwen3-Reranker-4B"` (Stage 4b cross-encoder; **disabled** in this run).
- `BACKEND = "local"`.

**Libraries (inline pip install):** `deep-translator`, `rank-bm25`, `scikit-learn`. Imports include `numpy`, `pandas`, `torch`, `scipy.sparse`, `tqdm`, `transformers (AutoTokenizer, AutoModelForCausalLM)`, `GoogleTranslator`, `BM25Okapi`.

**Runtime settings (cell 23):**
- `SPLIT = "val"`
- `BACKEND = "local"`
- `STAGE3_GOLD_PRIOR_MODE = "off"`
- `STAGE2_MAX_ITERATIONS = 4`
- `STAGE4_SKIP_DIRECT_GEN = False`
- `STAGE4B_TOP_N = None`
- `STAGE5_THRESHOLD = None`

**`default_config` (cell 24):**
```
{
  "stage1_backend": "local",
  "stage1_model": None,
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

**Stage 6 hyperparameters (cell 137):**
- `STAGE6_MIN_K = 10`, `STAGE6_MAX_K = 45`
- `STAGE6_USE_ADAPTIVE_K = True`, `STAGE6_PROCEDURAL_FLOOR = True`
- `STAGE6_SCORE_SOURCE = "bm25_plus_llm"`, `STAGE6_LLM_BOOST = 0.5`
- `STAGE6_K_STRATEGY = "score_gap"`, `STAGE6_FIXED_K = 15`
- score_gap params: `frac_of_max=0.55`, `elbow_drop=0.12`

**Hardware/notes:** GPU recommended H100 80 GB; OOM events occurred during Stage 2 batched LLM calls (fell back to sequential). BM25 sparse matrix is 2,156,832 docs x 822,251 terms (212,228,366 non-zero).

## Data

All paths anchored on `PROJECT_ROOT = /content/drive/MyDrive/swiss_law`.

**Raw competition data (under `data/`):**
- `data/train.csv`
- `data/val.csv`
- `data/test.csv`
- `data/court_considerations.csv` (2.26 GB)
- `data/laws_de.csv`
- `data/sample_submission.csv`

**Knowledge base / corpus (under `index/`, aka `LAW_DB`):**
- `index/corpus.parquet`
- `index/laws_knowledge_base.jsonl` (179,641 records)
- `index/bm25_v2_index.pkl` + `index/bm25_v2_ids.pkl`
- `index/bm25_v2_index_parts/bm25_v2_index_parts/` (216 chunks)
- `index/citation_graph.pkl` (2,201,708 nodes; 3,897,617 edges all 'other')
- `index/citation_lookup.pkl`
- `index/citation_signal_lookup.pkl`
- `index/reference_graph_v2.pkl`
- `index/gold_cocitation_prior_train.pkl`
- `index/gold_cocitation_prior_trainval.pkl`
- `index/query_translations_trainval.json`
- `index/token_law_freq.json`
- `index/citation_signal_schema_v2.json`
- `index/statute_signals.jsonl`
- `index/case_signals.jsonl`

**Missing/expected artifacts:** `index/statute_to_bge.parquet` (MISSING — Pre-diag 2/7 flagged).

**Runtime outputs:**
- `checkpoints/stage1_val.json`, `stage2_val.json`, `stage3_val.json`, `stage4_val.json`, `stage5_val.json`
- `checkpoints/bm25_scores_val.json`
- `checkpoints/shadow_val.csv` (150 rows; train rows with gold>=6, mean=12.5)
- `submissions/submission_val.csv`, `submissions/submission_val_pertype.csv`

**Models dir:** `model/` (HF cache via `HF_HOME` / `HUGGINGFACE_HUB_CACHE`).

## Pipeline

### Setup / Config
Cells 0-29: Drive mount, pip install, imports, base directory definition (`PROJECT_ROOT`, `INDEX_DIR`, `RAW_DATA_DIR`, `CHECKPOINTS_DIR`, `SUBMISSIONS_DIR`, `MODELS_DIR`), HF env vars, `default_config` dict, runner wrappers (`run_stage1..6`), and resolved per-split checkpoint paths.

### Inlined source modules (cells 9-19, 40-50, 54, 67, 79, 92, 102, 111, 123)
The notebook inlines the original project modules with their `__file__` stubs (originally under `/mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline/`):
- `retrieval/bm25_artifact.py`, `retrieval/sparse_retriever.py`, `retrieval/graph_retriever.py`, `retrieval/explicit_citations.py`
- `agent/llm_backend.py`, `agent/verifier.py`, `agent/prompts/mas_prompts.py`
- `indexing/build_lookup_tables.py`
- `scoring/confidence.py`
- `stage1_query_analysis.py`, `stage2_mas_retrieval.py`, `stage3_graph_expansion.py`, `stage4_llm_reranker.py`, `stage4b_cross_encoder.py`, `stage5_verify_and_score.py`, `submit.py`

### Pre-run diagnostics (cells 30-37, 7 cells)
Read-only checks before running Stage 1: data distribution, KB integrity, gold format audit (Abs/lit/bare), procedural-boilerplate census, BGE reachability in corpus, test vs val domain overlap, citation graph edge inventory.

### Stage 1 — Query Analysis (cell 56)
`run_stage1(split="val", backend="local")` loads domain knowledge (1,125 law abbreviations; 2,114 EN→DE term mappings; 69,837 DE keywords; 171,654 articles), loads combined BM25 index (216 chunks), loads Qwen3-32B, and processes 10 val queries in batches of 5. For each query: extracts explicit citations (regex), classifies law domains, generates DE search queries, generates HyDE documents. Writes `stage1_val.json`. Followed by 6 diagnostics (cells 58-63).

### Stage 2 — Multi-Agent Sparse Retrieval (cell 69)
`run_stage2(split="val", backend="local", max_iterations=4)` loads BM25 retriever and Qwen3-32B. Processes 10 queries in batches of 3. Each query iterates a Planner → Agent → BM25 loop with agent choices among REWRITE / SUPPLEMENT / SUPPORTIVE / DECOMPOSE / CROSSREF. Writes `stage2_val.json`. 6 diagnostics (cells 71-76).

### Stage 3 — Citation Graph Expansion (cell 81)
`run_stage3(split="val", gold_prior_mode="off")`. Loads citation graph (103,125 articles with case links; 914,658 cases with article links) and `article_to_abs` sibling expansion (43,066 entries). Expands each query's pool via 1-hop graph neighbours. Writes `stage3_val.json`. 5 diagnostics + BGE safety-net + Abs-level audit (cells 83-87).

### Stage 4 — LLM Reranker + Direct-Gen (cell 94)
`run_stage4(split="val", backend="local", skip_direct_gen=False)`. Loads Qwen3-32B, loads 220,490 citation texts from KB. Pool = top-80 candidates (BM25 + procedural + explicit + graph). LLM scores each candidate into tiers T2-T10. Direct-gen adds 15 citations per query from model memory. Writes `stage4_val.json`. 5 diagnostics (cells 96-100).

### Stage 4b — Cross-Encoder (cell 104)
`run_stage4b(split="val")` — **DISABLED**. The cell prints the skip reason; CE top-20 F1 was 0.17 on val vs BM25 top-20 F1 = 0.59. Stale CE file absent.

### Stage 5 — Verify + Score + Threshold (cell 113)
`run_stage5(split="val", threshold=None)`. Loads citation verifier and BM25 index, precomputes BM25 scores per candidate, scores with composite signals (no CE scores available), sweeps thresholds. Writes `stage5_val.json` and `submission_val.csv`. 6 diagnostics: global-threshold macro-F1, per-type LOO-CV thresholds, query-level F1, citation-level errors, Abs-flooding, procedural-prior potential (cells 115-120).

### Submission validation (cell 125)
`submit.py` validate path — fails because val submission has 10 queries while validator expects 40 test query IDs.

### Rollup dashboard (cell 127)
Single cell printing recall per stage per type.

### Diagnostics suite (cells 130, 132)
`run_all_diagnostics(split="val")` runs the fast D1 (ranking ceiling), D2 (adaptive-K oracle), D4 (CE calibration — skipped), D6 (shadow-val build). GPU diagnostics D3 (BGE-M3 recall) and D5 (LLM ranker smoke) are commented out.

### Stage 6 — Adaptive K + procedural floor (cells 137, 139)
Fits per-query K from score distribution (score_gap strategy), enforces `MIN_K=10` / `MAX_K=45`, force-includes universal procedural cites on appeal/Beschwerde/costs triggers, rewrites `submission_val.csv` using `bm25_plus_llm` score source with `w_llm=0.5`.

## Results

### Pre-diag 1/7 — data distribution
- val: 10 rows; cites/q min=10, max=47, mean=25.1, median=22.0; types Law=149 (59%), BGE=69 (27%), Numbered=33 (13%)
- train: 1139 rows; cites/q min=1, max=44, mean=4.1, median=2.0; types Law=4602 (99%), BGE=57 (1%)
- test: 40 rows
- WARNING: val (25.1/q) has >2× train (4.1/q) — NEVER tune thresholds on train

### Pre-diag 2/7 — KB integrity
- KB: 179,641 records; top codes EFZ=6500, OR=3667, EBA=2469, ZGB=2405, StPO=1306, StGB=1244, TSV=1094, VTS=1023, SchKG=919, ZV=874
- laws_de.csv: 175,933 rows; court_considerations.csv: 2.26 GB
- citation_lookup, bm25 index, statute→cases graph: present
- statute→BGE graph: MISSING (`index/statute_to_bge.parquet`)

### Pre-diag 3/7 — gold format audit
- Abs: 121 (81%); bare_Art: 28 (19%) — Abs-level matching critical

### Pre-diag 4/7 — procedural boilerplate
- 1 procedural cite in ≥2 val queries: `Art. 100 Abs. 1 BGG` in 9 queries (never in query text)

### Pre-diag 5/7 — BGE reachability
- Unique val-gold BGE stems: 54; present in corpus: 54/54

### Pre-diag 6/7 — test vs val domain overlap
- Codes in test but NOT covered by val: CC, CO, MSchG, PrHG, SVG, UVG

### Pre-diag 7/7 — citation graph edges
- nodes: 2,201,708; stat→BGE: 0; stat→num: 0; BGE→BGE: 0; BGE→stat: 0; other: 3,897,617
- "NO statute→BGE edges. BGE recall via graph expansion is impossible."

### Stage 1 runner
- Loaded 1,125 law abbreviations; 2,114 EN→DE mappings; 69,837 DE keywords; 171,654 articles
- BM25 chunk load: 216 chunks @ 2.49 it/s; sparse matrix built in 36.2 s
- 171,654 law + 1,985,178 court = 2,156,832 total docs
- Qwen3-32B loaded (~64 GB bf16). Batch 1 LLM: 286.5 s (57.3 s/query). Batch 2 LLM: 81.6 s (16.3 s/query).
- Stage 1 saved to `checkpoints/stage1_val.json`

### Stage 1 diag 1/6 — artifact summary
- explicit_citations: min=0 max=3 mean=0.8
- candidate_articles: 10 each; candidate_cases: 10 each
- search_queries_en: min=3 max=5 mean=4.5; search_queries_de: 5 each
- hyde_documents_de: 2 each; tokens min=84 max=108 mean=95.7
- law_areas: min=1 max=3 mean=1.9; legal_issues: min=1 max=5 mean=2.7

### Stage 1 diag 2/6 — explicit fidelity
- 6/6 = 100.0%

### Stage 1 diag 3/6 — candidate validity
- 100/100 valid

### Stage 1 diag 4/6 — law_area routing
- Routing precision mean = 0.330; recall mean = 0.955
- val_003 miss: BV; val_007 miss: StGB

### Stage 1 diag 5/6 — HyDE drift
- 2/10 queries suspicious (val_004, val_008)

### Stage 1 diag 6/6 — free-recall ceiling
- 126/251 = 50.2% (Law 82/149=55.0%, BGE 33/69=47.8%, Numbered 11/33=33.3%)
- Per-query: 001=48%, 002=56%, 003=36%, 004=60%, 005=64%, 006=50%, 007=58%, 008=41%, 009=79%, 010=52%

### Stage 2 runner
- Batches 1-4 timings: 711.6 s, 758.8 s, 762.1 s, 122.5 s
- OOM events at Iter 3-4 of Batch 1 and Batch 3 — fell back to sequential
- Candidate counts: val_001=1070, val_002=1050, val_003=1155, val_004=967, val_005=1132, val_006=1250, val_007=981, val_008=1212, val_009=1364, val_010=1384

### Stage 2 diag 1/6 — candidate counts
- cands/q min=967 max=1384 mean=1156.5
- Types: Numbered=6770 (59%), Law=3221 (28%), BGE=828 (7%), Other=746 (6%)

### Stage 2 diag 2/6 — recall per type
- Law: 119/149 = 79.9%; BGE: 51/69 = 73.9%; Numbered: 26/33 = 78.8%

### Stage 2 diag 3/6 — R@K curves
```
K     Law             BGE           Numbered
10    44.3%           24.6%         6.1%
50    57.7%           60.9%         66.7%
100   57.7%           60.9%         66.7%
200   65.1%           62.3%         72.7%
500   68.5%           69.6%         78.8%
1000  73.2%           72.5%         78.8%
```

### Stage 2 diag 4/6 — single-vs-multi-query ablation
- EN only (raw query): Law=39.6%, BGE=65.2%, Numbered=75.8%
- Single DE query: Law=4.0%, BGE=1.4%, Numbered=0.0%
- All DE queries (≤10): Law=19.5%, BGE=5.8%, Numbered=9.1%
- All DE + HyDE: Law=20.8%, BGE=7.2%, Numbered=15.2%

### Stage 2 diag 5/6 — dead query variants
- DE_0=0.20/run (DEAD), DE_3=0.30 (DEAD), DE_4=0.40 (DEAD), HYDE_1=0.30 (DEAD)
- EN_raw=12.10/run (best), EN_0=7.00, EN_1=5.10, EN_2=5.30

### Stage 2 diag 6/6 — MAS action source
- bm25_initial=7827; mas_supportive=2256; mas_supplement=818; mas_decompose=796; mas_crossref=414; mas_rewrite=393; llm_stage1=200; llm_stage1_procedural=200; bm25_top10=82; explicit_from_query=8
- Avg overlap fraction = 0.10

### Stage 3 runner
- 103,125 articles with case links; 914,658 cases with article links; article_to_abs 43,066 entries
- Expansion totals per query (cands → new): 1070→2324, 1050→2711, 1155→2825, 967→2533, 1132→2323, 1250→3667, 981→2533, 1212→3180, 1364→3500, 1384→3774
- Total 11565 → 29370 citations (+17805 from graph)

### Stage 3 diag 1/5 — counts + per-type recall
- cands/q min=2323 max=3774 mean=2937.0; BGE/q mean=587.8; Numbered/q mean=990.0
- Law: 140/149 = 94.0%; BGE: 57/69 = 82.6%; Numbered: 26/33 = 78.8%

### Stage 3 diag 2/5 — graph reachability
- val gold BGE reachable at 1-hop: 0/56 = 0.0%
- val gold BGE reachable at 2-hop: 0/56 = 0.0%

### Stage 3 diag 3/5 — procedural coverage
- Total procedural gold events: 2; covered: 2 = 100% (Art. 82 BGG, Art. 113 BGG)

### Stage 3 diag 4/5 — BGE safety-net
- Total recoverable: 0/8 = 0% (statute-text mining cannot recover missed BGEs)

### Stage 3 diag 5/5 — Abs-level audit
- abs_exact_hit: 114/121 = 94%; abs_parent_only: 5/121 = 4%; abs_missed: 2/121 = 2%
- Abs-flooding events (≥3 Abs variants for same Art): 1228 across 10 queries

### Stage 4 runner
- Pool size: 80 per query; Qwen3-32B; 220,490 citation texts loaded
- Per-query: scored counts + tier distribution + direct-gen added 15
- Timings: val_001 157.3 s, val_002 41.4 s, val_003 133.9 s, val_004 100.3 s, val_005 43.7 s, val_006 116.5 s, val_007 102.1 s, val_008 103.1 s, val_009 91.3 s, val_010 85.2 s

### Stage 4 diag 1/5 — counts
- reranked/q: min=35 max=80 mean=67.6; direct-gen/q: 15 each

### Stage 4 diag 2/5 — direct-gen hallucination
- Validity: 60/150 = 40.0% (sample halluc: Art. 422 StPO, Art. 428 StPO, BGE 147 IV 101, BGE 145 IV 321, etc.)

### Stage 4 diag 3/5 — tier distribution
- T10=4 (0.6%); T9=10 (1.5%); T8=56 (8.3%); T7=148 (21.9%); T6=304 (45.0%); T5=90 (13.3%); T4=19 (2.8%); T3=20 (3.0%); T2=25 (3.7%)

### Stage 4 diag 4/5 — gold dropout
- 56/223 gold dropped from Stage 3 → Stage 4

### Stage 4 diag 5/5 — pool recall + tiers
- Law: 103/149 = 69.1%; BGE: 42/69 = 60.9%; Numbered: 22/33 = 66.7%
- Gold tier assignment Law: T10=2, T5=1, T6=12, T7=13, T8=14, T9=7, absent=46, passthru=54
- Gold BGE: T6=3, T7=6, T8=6, absent=27, passthru=27
- Gold Numbered: T6=5, T8=2, absent=11, passthru=15

### Stage 4b — DISABLED
- Reason: CE top-20 F1 was 0.17 on val vs BM25 top-20 F1 = 0.59
- No stale CE file present

### Stage 5 runner
- BM25 precompute total ~48 s across 10 queries
- Total verified: 969, dropped: 261
- Best threshold: 0.65; Best macro-F1: 0.4746
- Per-query F1: val_001=0.3529, val_002=0.4000, val_003=0.3607, val_004=0.3571, val_005=0.5455, val_006=0.4348, val_007=0.4571, val_008=0.6038, val_009=0.6250, val_010=0.6087
- Submission: avg 14.5 citations, max 24, 0 empty queries

### Stage 5 diag 1/6 — global threshold + per-type
- Macro-F1: 0.4746 @ threshold 0.65
- Numbered: P=0.127 R=0.212 F1=0.159 (tp=7 fp=48 fn=26)
- BGE: P=0.920 R=0.333 F1=0.489 (tp=23 fp=2 fn=46)
- Law: P=0.969 R=0.423 F1=0.589 (tp=63 fp=2 fn=86)

### Stage 5 diag 2/6 — per-type LOO CV
- Held-out mean F1 (LOO): 0.4289
- Best per-type thresholds: Law=0.46, BGE=0.34, Numbered=0.46, Other=0.15

### Stage 5 diag 3/6 — query-level F1 @ 0.65
- val_001: pred=9 gold=42 TP=9 P=1.000 R=0.214 F1=0.353
- val_002: pred=9 gold=36 TP=9 P=1.000 R=0.250 F1=0.400
- val_003: pred=14 gold=47 TP=11 P=0.786 R=0.234 F1=0.361
- val_004: pred=18 gold=10 TP=5 P=0.278 R=0.500 F1=0.357
- val_005: pred=11 gold=11 TP=6 P=0.545 R=0.545 F1=0.545
- val_006: pred=5 gold=18 TP=5 P=1.000 R=0.278 F1=0.435
- val_007: pred=16 gold=19 TP=8 P=0.500 R=0.421 F1=0.457
- val_008: pred=24 gold=29 TP=16 P=0.667 R=0.552 F1=0.604
- val_009: pred=18 gold=14 TP=10 P=0.556 R=0.714 F1=0.625
- val_010: pred=21 gold=25 TP=14 P=0.667 R=0.560 F1=0.609

### Stage 5 diag 4/6 — citation-level errors
- Misses by type: Law=86; BGE=46; Numbered=26
- False positives by type: Numbered=48; BGE=2; Law=2

### Stage 5 diag 5/6 — Abs-flooding in predictions
- 0 events across 0 queries (eliminated)

### Stage 5 diag 6/6 — procedural-prior potential
- Gold procedural cites: 2; catchable: +2 TP but introduces 148 FP → selective domain-conditioned prior required

### Submission validation
- `ERROR: Missing 40 test query IDs` → `Submission valid: False` (val csv has 10 not 40)

### Omni-rollup recall table
```
stage           Law            BGE          Numbered
Stage1          82/149=55.0%   33/69=47.8%  11/33=33.3%
Stage2          119/149=79.9%  51/69=73.9%  26/33=78.8%
Stage3          140/149=94.0%  57/69=82.6%  26/33=78.8%
Stage4          103/149=69.1%  42/69=60.9%  22/33=66.7%
Stage5@0.65     63/149=42.3%   23/69=33.3%  7/33=21.2%
```
- Pool drops Stage3→Stage4: Law 24.8pp (94→69), BGE 21.7pp (83→61), Numbered 12.1pp (79→67)

### Fast diagnostics
- **D1 ranking ceiling on val:** per-query gold/in_pool/in_bm25/med_rank/max_rank — e.g. val_001: gold=42 in_pool=42 in_bm25=35 med_rank=18 max_rank=51
- **D2 adaptive-K oracle:** BM25 oracle-K macro-F1=0.6023; Stage-5 oracle-K macro-F1=0.5456
- **D4 CE calibration:** skipped (Stage 4b disabled)
- **D6 shadow-val build:** 253 train rows with gold≥6; wrote 150 to `shadow_val.csv`; gold-size mean=12.5 median=11 min=6 max=44

### Stage 6 — adaptive K + procedural floor
- Score source: bm25_plus_llm (w_llm=0.5); K strategy: score_gap (frac_of_max=0.55, elbow_drop=0.12)
- Per-query F1: val_001 K=34 F1=0.895; val_002 K=25 F1=0.820; val_003 K=17 F1=0.537; val_004 K=10 F1=0.522; val_005 K=10 F1=0.667; val_006 K=10 F1=0.643; val_007 K=10 F1=0.690; val_008 K=16 F1=0.711; val_009 K=11 F1=0.880; val_010 K=14 F1=0.718
- **MACRO F1 (adaptive-K): 0.7082; P=0.8962; R=0.6047**
- Mean predictions per query: 16.3 (min=10, max=34)
- Saved: `submissions/submission_val.csv`

## Summary

The notebook runs the BM25-first Swiss legal citation pipeline end-to-end on the 10-query val split with Qwen3-32B as the only LLM. Pre-diagnostics surfaced two structural problems before running: the citation graph contains zero statute→BGE edges (`statute_to_bge.parquet` missing), and val codes (OR, StPO, ZGB) do not cover six test codes (CC, CO, MSchG, PrHG, SVG, UVG). Retrieval through Stage 3 reached strong pool recall (Law 94.0%, BGE 82.6%, Numbered 78.8%), but Stage 4's LLM reranker dropped 56/223 gold citations and direct-gen hallucinated 60% of its outputs, collapsing Stage 5's composite-score submission to macro-F1 = 0.4746 at threshold 0.65. Stage 4b's cross-encoder was disabled after CE top-20 F1 (0.17) underperformed BM25 top-20 F1 (0.59). Stage 6's adaptive-K plus bm25_plus_llm rescoring lifted final macro-F1 to 0.7082 (P=0.8962, R=0.6047) — a +0.23 gain over Stage 5's composite path — confirming the lesson that BM25 with a light LLM boost beats the multi-signal confidence stack, and that per-query K selection via score-gap is essential given val's 10-47 citations/query range.
