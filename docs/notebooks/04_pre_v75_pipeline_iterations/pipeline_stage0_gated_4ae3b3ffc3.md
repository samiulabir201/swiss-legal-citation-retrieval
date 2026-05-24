# pipeline_stage0_gated_4ae3b3ffc3.ipynb

**Path:** notebooks/04_pre_v75_pipeline_iterations/pipeline_stage0_gated_4ae3b3ffc3.ipynb

## Configuration

### Runtime settings (cell 24)
- `SPLIT = "test"` (train | val | test)
- `BACKEND = "local"` (local | anthropic | openai)
- `START_STAGE = 1` (0 rebuilds Stage 0 then continues; 1..5 skips Stage 0 and starts there)
- `STAGE3_GOLD_PRIOR_MODE = "off"` (off | train | trainval)
- `STAGE2_MAX_ITERATIONS = 4`
- `STAGE4_SKIP_DIRECT_GEN = False`
- `STAGE4B_TOP_N = None` (None = stage default)
- `STAGE5_THRESHOLD = None` (None = stage default)

### default_config (cell 25)
```
{
 '_comment': 'Revised pipeline config — BM25-first, LegalMALR-adapted MAS architecture',
 'stage1_backend': 'local',
 'stage1_model': None,
 'bm25_top_k': 100,
 'graph_top_k': 50,
 'mas_max_iterations': 4,
 'mas_bm25_top_k': 30,
 'reranker_backend': 'local',
 'reranker_batch_size': 20,
 'confidence_weights': {
   'explicit_from_query': 0.4,
   'bm25_top10': 0.25,
   'citation_graph': 0.15,
   'llm_reranker': 0.15,
   'llm_direct_gen': 0.05
 },
 'threshold': 0.15
}
```

### Scoring `DEFAULT_WEIGHTS` (cell 8)
- `explicit_from_query: 0.40`
- `bm25_top10: 0.15`        (binary: in BM25 top-10 yes/no)
- `bm25_initial: 0.10`

### BM25 hyperparameters
- `k1 = 1.2`, `b = 0.75` (BM25 Okapi)
- `BM25_CHUNK_SIZE = 10000` (env `SWISS_BM25_CHUNK_SIZE`)
- `ARTIFACT_FORMAT = "chunked_bm25_v1"` (ChunkedBM25Okapi backed by scipy CSC sparse TF matrix)

### Stage-0 building constants
- `CHUNKSIZE = 200_000`, `MAX_CITES_PER_ROW = 30`, `MIN_CO_OCCURRENCE = 3` (citation graph)
- `CASE_CHUNKSIZE = 200_000`, `TEXT_SNIPPET_CHARS = 420`, `GRAPH_NEIGHBOR_LIMIT = 8`, `KEYWORD_LIMIT = 8`, `ENUM_ITEM_LIMIT = 8` (stage0d citation signals)
- `LAWS_CHUNKSIZE = 50_000` (stage0e reference graph)
- `SCHEMA_VERSION` values: `"citation-signals-v2"`, `"reference-graph-v2"`, `"gold-cocitation-prior-v1"`

### Models
- Primary LLM: `Qwen/Qwen3-32B` (Kaggle path `/kaggle/input/models/qwen-lm/qwen-3/transformers/32b/1`)
  - Loaded once as singleton via `agent/llm_backend.py`
  - Quantization configured as Q4/NF4 in comments but actual loader uses `torch_dtype=torch.bfloat16, device_map="auto"` (no `BitsAndBytesConfig` applied in `from_pretrained`)
  - Stated VRAM: "~2.7-2.8 GB with NF4 bfloat16 compute" in docstring; print after load: "VRAM: ~64GB (bf16)"
  - Sampling defaults: `temperature=0.6`, `top_k=20`, `top_p=0.95`, `max_new_tokens=2048`
  - Stage 1 uses `enable_thinking=True`; Stage 4 reranker uses `enable_thinking=False`
- Reranker: `Qwen/Qwen3-Reranker-4B` (via `SWISS_QWEN3_RERANKER_ID`) — Stage 4b cross-encoder

### Libraries (cell 5 imports + cell 23 requirements snapshot)
- `numpy`, `pandas`, `torch`, `scipy.sparse`, `tqdm`
- `rank-bm25==0.2.2`
- `transformers>=4.45.0`, `bitsandbytes>=0.44.0`, `accelerate>=1.0.0`, `huggingface_hub>=0.25.0`, `safetensors>=0.4.0`
- `deep-translator>=1.11.0` (`GoogleTranslator`)
- `anthropic>=0.40.0`, `openai>=1.50.0` (optional cloud backends)
- `pyarrow>=14.0.0`
- Cell 4 install: `pip install -qU deep-translator rank-bm25`

### Hardware
- Stated: Python 3.12 | CUDA 12.1 | RTX 4050 6GB (in requirements doc-comment)
- Kaggle environment with `/kaggle/input/` (read-only) and `/kaggle/working/` (writable)
- Stage 0 BM25 build: "~16GB RAM, CPU only"

## Data

Kaggle dataset layout (cell 5):

- `LAW_DB = /kaggle/input/datasets/samiulislam180041221/law-db`
- `COMP_DATA = /kaggle/input/datasets/moniruzzamanmahadi8/llm-agentic-legal-information-retrieval-dataset/llm-agentic-legal-information-retrieval`

Raw input files:
- `LAW_CORPUS_PARQUET = LAW_DB/corpus.parquet`
- `KB_JSONL = LAW_DB/laws_knowledge_base.jsonl`
- `COURT_CSV = COMP_DATA/court_considerations.csv`
- `LAWS_DE_CSV = COMP_DATA/laws_de.csv`
- `TRAIN_CSV = COMP_DATA/train.csv`
- `VAL_CSV = COMP_DATA/val.csv`
- `TEST_CSV = COMP_DATA/test.csv`
- `SAMPLE_SUBMISSION_CSV = COMP_DATA/sample_submission.csv`

Pre-built index artifacts (inside `LAW_DB`):
- `STATUTORY_BM25_PKL = bm25_v2_index.pkl`
- `STATUTORY_BM25_IDS = bm25_v2_ids.pkl`
- `CITATION_GRAPH_PKL = citation_graph.pkl`
- `LOOKUP_PKL = citation_lookup.pkl`
- `CITATION_SIGNAL_LOOKUP_PKL = citation_signal_lookup.pkl`
- `REFERENCE_GRAPH_V2_PKL = reference_graph_v2.pkl`
- `GOLD_COCITATION_PRIOR_TRAIN_PKL = gold_cocitation_prior_train.pkl`
- `GOLD_COCITATION_PRIOR_TRAINVAL_PKL = gold_cocitation_prior_trainval.pkl`
- `BM25_PARTS_DIR = LAW_DB/bm25_v2_index_parts/bm25_v2_index_parts`

Legacy / auxiliary artifacts:
- `QUERY_TRANSLATIONS_JSON_SRC = LAW_DB/query_translations_trainval.json` (read-only source; copied to checkpoints)
- `TOKEN_LAW_FREQ = LAW_DB/token_law_freq.json`
- `CITATION_SIGNAL_SCHEMA_JSON = LAW_DB/citation_signal_schema_v2.json`
- `STATUTE_SIGNALS_JSONL = LAW_DB/statute_signals.jsonl`
- `CASE_SIGNALS_JSONL = LAW_DB/case_signals.jsonl`

Model location:
- `QWEN3_32B_PATH = /kaggle/input/models/qwen-lm/qwen-3/transformers/32b/1`
- `QWEN3_32B_HF_ID = "Qwen/Qwen3-32B"`

Writable output dirs (Kaggle `/kaggle/working/`):
- `OUT_DIR = /kaggle/working/submissions`
- `CHECKPOINTS_DIR = /kaggle/working/checkpoints`

Notebook-visible path block (cell 21 — overrides above when run outside Kaggle):
- `PROJECT_ROOT = /mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline`
- `RAW_DATA_DIR = $SWISS_DATA_DIR | PROJECT_ROOT/data`
- `INDEX_DIR = PROJECT_ROOT/index`
- `CHECKPOINTS_DIR = PROJECT_ROOT/checkpoints`
- `SUBMISSIONS_DIR = PROJECT_ROOT/submissions`
- `MODELS_DIR = PROJECT_ROOT/models`
- Env vars exported: `SWISS_QWEN3_32B_PATH`, `SWISS_QWEN3_32B_HF_ID`, `SWISS_QWEN3_RERANKER_ID`, `SWISS_PIPELINE_ROOT`, `SWISS_DATA_DIR`

Data observed at load time (cell 50 output):
- Law corpus: 171,654 articles
- Court considerations raw rows: 2,476,315; unique citations: 1,985,178
- KB records: 171,654
- English law names loaded: 1,125
- Train queries: 1,139 | Val queries: 10
- Combined corpus after dedup: 2,156,832 documents
- Avg tokens per doc: 169
- Translation cache: 12 cached translations; 1,137 train queries auto-skipped (look German); 0 queries needing external translation

## Pipeline

### Stage 0 — Offline Index Building (gated by `START_STAGE`)

Cells 33–62 inline the Stage 0 module sources and call a runner. `START_STAGE <= 0` rebuilds; `>= 1` skips and uses pre-built artifacts in `LAW_DB`.

Sub-stages (`stage0_build_index.py`, cell 48):
- 0A. BM25 combined index over 171K laws + ~2M court considerations (`build_bm25_index.py`, cell 50)
- 0B. Citation graph (Art. ↔ Court bidirectional) (`build_citation_graph.py`, cell 52)
- 0C. Citation lookup tables: `citation_set`, `article_to_abs`, `abs_to_article`, `sr_to_abbrev`, `abbrev_to_sr` (`build_lookup_tables.py`, cell 54)
- 0D. Deterministic citation signal cards: `citation_signal_schema_v2.json`, `statute_signals.jsonl`, `case_signals.jsonl` (`stage0d_build_citation_signals.py`, cell 56)
- 0E. Typed reference graph v2: `article_to_articles`, `article_to_cases`, `case_to_articles`, `case_to_cases` (`stage0e_build_reference_graph_v2.py`, cell 58)
- 0F. Gold co-citation prior (train-only; optional train+val variant) (`stage0f_build_gold_cocitation_prior.py`, cell 60)

Stage 0 runner (cell 62):
```python
if stage0_requested():
    run_stage0_all(start_stage=START_STAGE, allow_stage0=True)
else:
    print("Stage 0 is SKIPPED ... Using pre-built indexes from", INDEX_DIR)
    print("  BM25 index:", STATUTORY_BM25_PKL.exists())
    print("  Citation graph:", CITATION_GRAPH_PKL.exists())
    print("  Lookup tables:", LOOKUP_PKL.exists())
```

### Shared utilities (cells 7–20, 33–45)

- `retrieval/bm25_artifact.py`: `ChunkedBM25Okapi` — sparse-CSC BM25 scorer (k1, b, idf, avgdl, epsilon).
- `scoring/confidence.py`: composite score = binary signal weights + continuous BM25 (0–0.20) + graduated multi-signal bonus; F1 threshold sweep.
- `retrieval/explicit_citations.py`: regex-extracts literal citations from query text; canonicalizes alternate abbreviations via `_ALT_ABBREV_TO_CANON`.
- `retrieval/sparse_retriever.py`: BM25 retrieval with PMI-based law abbreviation boosting, multi-query union with max-score fusion, `np.argpartition` top-K.
- `retrieval/graph_retriever.py`: `GraphRetriever` wrapping `CitationGraph` from `citation_graph.pkl`.
- `agent/llm_backend.py`: Qwen3-32B singleton; `generate()`, `generate_json()` (with OOM retry at halved tokens), `parse_json_response()`.
- `agent/verifier.py`: corpus-form → gold-form abbreviation casing map (`STGB → StGB`, `STPO → StPO`, `JSTG → JStG`, `SCHKG → SchKG`, …); normalization + existence check.

### Stage 1 — Query Analysis (`stage1_query_analysis.py`, cell 66)

6-step corpus-grounded query understanding. System prompt in cell 64 instructs the model to (1) classify legal domain(s) — Civil (ZGB/OR), Criminal (StGB/StPO), Public (BV/VwVG), Social insurance (ATSG/IVG/UVG/AVIG/BVG), IPR (IPRG), Debt collection (SchKG), Procedural (ZPO/StPO/BGG); (2) identify legal issues; (3) generate 3–5 German search queries.

Runner (cell 68): `stage1_outputs = run_stage1(split=SPLIT, backend=BACKEND)`.

### Stage 2 — Multi-Agent Sparse Retrieval (`stage2_mas_retrieval.py`, cell 72)

LegalMALR-adapted MAS over BM25. Agents (cell 10 prompts):
- Planner: decides next specialist or termination
- Rewrite: EN colloquial → precise DE legal terminology
- Supplement: makes implicit conditions explicit
- Decompose: splits multi-issue queries into sub-queries
- Supportive: procedural articles
- CrossRef: constitutional / general-clause provisions

Each agent produces DE queries → BM25 (`mas_bm25_top_k=30`). Up to `STAGE2_MAX_ITERATIONS=4` iterations. Target pool >100 candidates.

Runner (cell 74): `stage2_outputs = run_stage2(split=SPLIT, backend=BACKEND, max_iterations=STAGE2_MAX_ITERATIONS)`.

### Stage 3 — Citation Graph Expansion (`stage3_graph_expansion.py`, cell 77)

For each statute → add top-5 citing court decisions; for each case → add cited statutes. Optional gold co-citation prior (`STAGE3_GOLD_PRIOR_MODE`). Stated motivation: 59.4% of val citations are statutory, 40.6% case law; train data is 98.8% statutory.

Runner (cell 79): `stage3_outputs = run_stage3(split=SPLIT, gold_prior_mode=STAGE3_GOLD_PRIOR_MODE)`.

### Stage 4 — LLM Reranking + Direct Generation (`stage4_llm_reranker.py`, cell 84)

- 4A. Reranker: tier-3 (directly applicable) / tier-2 (related) selection from Stage 3 pool, no chain-of-thought (LegalMALR §3.4). Prompt in cell 81.
- 4B. Direct gen: memory-based additional citations from Qwen3-32B. Prompt in cell 82.

Runner (cell 86): `stage4_outputs = run_stage4(split=SPLIT, backend=BACKEND, skip_direct_gen=STAGE4_SKIP_DIRECT_GEN)`.

### Stage 4b — Cross-Encoder Scoring (`stage4b_cross_encoder.py`, cell 88)

Top-500 candidates per query scored by Qwen3-Reranker-4B as cross-encoder → `P(yes)`. Consumed by Stage 5 Strategy C: replaces binary `llm_reranker`/`tier3` signals with continuous score × 0.18.

Runner (cell 90): `stage4b_outputs = run_stage4b(split=SPLIT, top_n=STAGE4B_TOP_N)`.

### Stage 5 — Verify, Confidence, Threshold (`stage5_verify_and_score.py`, cell 93)

- 5A. Verification: normalize formatting, check existence, drop hallucinations, expand Art.→Abs.
- 5B. Confidence: binary signal weights + continuous BM25 (0–0.20) + graduated multi-signal bonus (2+, 3+, 4+ independent signals).
- 5C. Threshold sweep 0.05–0.95 for macro-F1 (val); apply tuned threshold on test.

Runner (cell 95):
```python
stage5_outputs = run_stage5(split=SPLIT, threshold=STAGE5_THRESHOLD)
submission_path = get_submission_path(SPLIT)
```

### Orchestrator (cell 98, runner cell 100)

`run_full_pipeline(split, start_stage, backend, stage3_gold_prior_mode)` — chained call equivalent to running stages sequentially. Commented out in cell 100 (manual stage-by-stage execution preferred).

### Submission (cell 103 source, cell 105 runner)

`submit.py` writes/validates competition CSV. Final cell:
```python
if submission_path.exists():
    submission_valid = validate_generated_submission(SPLIT)
```

## Results

Only Stage 0 / config cells were executed in this notebook (executions 164–178). Stages 1–5 cells (68, 74, 79, 86, 90, 95, 105) have no recorded outputs.

### Cell 21 — path config
```
PROJECT_ROOT    = /mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline
RAW_DATA_DIR    = /mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline/data
INDEX_DIR       = /mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline/index
CHECKPOINTS_DIR = /mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline/checkpoints
SUBMISSIONS_DIR = /mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline/submissions
Working dir     = /mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline
```

### Cell 23 — requirements echo (stderr SyntaxWarning)
```
<>:11: SyntaxWarning: invalid escape sequence '\S'
/tmp/ipykernel_106/3391981546.py:11: SyntaxWarning: invalid escape sequence '\S'
  #   .venv\Scripts\activate            (Windows)
```

### Cell 24 — runtime settings echo
```
{'SPLIT': 'test', 'BACKEND': 'local', 'START_STAGE': 1, 'STAGE3_GOLD_PRIOR_MODE': 'off'}
```

### Cell 25 — default_config (see Configuration section above)

### Cell 28
```
Notebook helper functions are ready.
Notebook helper functions are ready.
```

### Cell 30 — checkpoint paths
```
SUBMISSION_FILE  = /mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline/submissions/submission_test.csv
STAGE1_CHECKPOINT = /mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline/checkpoints/stage1_test.json
STAGE5_CHECKPOINT = /mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline/checkpoints/stage5_test.json
```

### Cell 50 — BM25 index build (Stage 0A, fails at save)
```
======================================================================
  LOADING DATA
======================================================================
  Law corpus: 171,654 articles
  Reading court_considerations.csv (2.47M rows) ...
    Raw rows: 2,476,315  |  Unique citations: 1,985,178
  Deduplicating: concatenating texts per citation ...
  Court corpus after dedup: 1,985,178 unique citations
  Loading KB JSONL ...
  KB records: 171,654
  English law names loaded: 1,125
  Train: 1,139 queries  |  Val: 10 queries

======================================================================
  LOADING QUERY TRANSLATIONS
======================================================================
  Loaded 12 cached translations.
  Auto mode skipped 1,137 queries that already look German.
  Translating 0 new queries across 0 unique texts ...
  No queries need external translation in this run.

======================================================================
  BUILDING DOCUMENT EXPANSION TOKENS
======================================================================
  Articles/courts with expansion tokens: 2,111
    Law: 1,958  |  Court: 153

======================================================================
  BUILDING DOCUMENT CORPUS (law + court)
======================================================================
  Building law document texts ...
  Law docs: 100%|####################| 171654/171654 [00:06<00:00, 27219.30it/s]
  Built 171,654 law docs  (1,958 with expansion)
  Building court document texts ...
  Court docs: 100%|################| 1985178/1985178 [00:42<00:00, 46577.41it/s]
  Built 1,985,178 court docs  (153 with expansion)

  Combined corpus: 2,156,832 documents
  Tokenising ...
  Tokenising: 100%|################| 2156832/2156832 [02:29<00:00, 14379.10it/s]
  Avg tokens per doc: 169

======================================================================
  BUILDING TOKEN->LAW MAP
======================================================================
  Mapping: 100%|##########################| 1139/1139 [00:00<00:00, 3809.50it/s]
  Token->law map: 20,034 unique tokens

======================================================================
  BUILDING BM25 INDEX  (k1=1.2, b=0.75)
======================================================================
  BM25 index: 100%|################| 2156832/2156832 [01:08<00:00, 31399.48it/s]
  Index built.

======================================================================
  SAVING INDEX
======================================================================
```

Error at save (cell 50):
```
OSError: [Errno 30] Read-only file system: '/kaggle/input/datasets/samiulislam180041221/law-db/bm25_v2_index_parts/doc_len.npy'
  in indexing/build_bm25_index.py: save_bm25_artifact(bm25, out_index)
  at np.save(artifact_dir / "doc_len.npy", np.asarray(bm25.doc_len, dtype=np.uint32), allow_pickle=False)
```

(Save target points at the read-only Kaggle input mirror rather than `/kaggle/working/`. The build itself completed successfully — only persistence failed.)

### Reference (from Project Overview markdown, cell 2 — expectations, not measurements)
| Component                       | Estimated Recall |
|---------------------------------|------------------|
| BM25 baseline (R@50)            | 0.73             |
| + MAS multi-agent reformulation | +0.08–0.12       |
| + Citation graph expansion      | +0.05–0.10       |
| + Explicit citation extraction  | +0.02–0.05       |
| + LLM direct generation         | +0.03–0.05       |
| Combined recall estimate        | 0.85–0.92        |
| After LLM reranking (precision) | 0.65–0.75        |
| Estimated F1                    | 0.70–0.80        |

Reference (from cell 88 docstring): Stage 4b cross-encoder Strategy C improves macro-F1 from 0.5400 to 0.6520 on val.

No execution metrics for Stage 1–5 are present in the saved notebook.

## Summary

Notebook is a notebook-clean re-packaging of the entire 5-stage BM25-first Swiss-law citation pipeline (Qwen3-32B singleton + MAS retrieval + citation-graph expansion + LLM reranker + Stage 4b cross-encoder + Stage 5 confidence/threshold). All project modules are inlined as utility/source cells; the runtime is configured with `SPLIT="test"`, `BACKEND="local"`, `START_STAGE=1` so Stage 0 is gated off and pre-built artifacts in `LAW_DB` are reused. The only stage actually executed in this saved session was a Stage 0A BM25 rebuild that completed corpus loading, document construction (2,156,832 docs, avg 169 tokens), and index build (k1=1.2, b=0.75) in well under 5 minutes but failed at persistence with `OSError: Read-only file system` because the save path resolves to `/kaggle/input/.../bm25_v2_index_parts/` instead of `/kaggle/working/`. Stages 1–5 were not run, so no retrieval/Macro-F1 numbers are produced here; the only metric references (BM25 R@50=0.73, full-pipeline F1 0.70–0.80 estimate, Stage 4b cross-encoder bumping val macro-F1 from 0.5400 to 0.6520) come from in-notebook prose, not from this run. Key takeaway: the gating mechanism and inline structure work, but `build_bm25_index.save_bm25_artifact` needs a writable output path before Stage 0 can be re-run on Kaggle.
