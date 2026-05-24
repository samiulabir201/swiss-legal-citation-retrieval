# pipeline_end_to_end_native.ipynb

**Path:** notebooks/04_pre_v75_pipeline_iterations/pipeline_end_to_end_native.ipynb

## Configuration

### Project root and runtime settings (cell 4, 6)
- `PROJECT_ROOT = Path(r"/mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline").resolve()`
- `SPLIT = "test"` (options: `"train"`, `"val"`, `"test"`)
- `BACKEND = "local"` (options: `"local"`, `"anthropic"`, `"openai"`)
- `START_STAGE = 1`
- `STAGE3_GOLD_PRIOR_MODE = "off"` (options: `"off"`, `"train"`, `"trainval"`)
- `STAGE2_MAX_ITERATIONS = 4`
- `STAGE4_SKIP_DIRECT_GEN = False`
- `STAGE4B_TOP_N = None` (None -> stage default)
- `STAGE5_THRESHOLD = None` (None -> stage default)

### `configs/default.json` (cell 7-8)
```json
{
    "_comment": "Revised pipeline config -- BM25-first, LegalMALR-adapted MAS architecture",
    "stage1_backend":  "local",
    "stage1_model":    null,
    "bm25_top_k":      100,
    "graph_top_k":     50,
    "mas_max_iterations": 4,
    "mas_bm25_top_k":    30,
    "reranker_backend": "local",
    "reranker_batch_size": 20,
    "confidence_weights": {
        "explicit_from_query": 0.40,
        "bm25_top10":          0.25,
        "citation_graph":      0.15,
        "llm_reranker":        0.15,
        "llm_direct_gen":      0.05
    },
    "threshold": 0.15
}
```

### Confidence weights (cell 27, `scoring/confidence.py`)
```
DEFAULT_WEIGHTS = {
    "explicit_from_query":    0.40,
    "bm25_top10":             0.15,
    "bm25_initial":           0.10,
    "citation_graph":         0.10,
    "llm_reranker_tier3":     0.25,
    "llm_reranker":           0.15,
    "llm_direct_gen":         0.05,
    "llm_stage1":             0.05,
    "llm_stage1_procedural":  0.05,
    "co_citation":            0.03,
}
BM25_CONTINUOUS_WEIGHT = 0.10
CROSS_ENCODER_WEIGHT   = 0.18
MULTI_SIGNAL_BONUS     = 0.10  # added when 3+ independent signals agree
```

### Models
- LLM (single GPU model for all stages): **Qwen3-4B**, 4-bit NF4 quantization, bfloat16 compute dtype.
  - Local path env: `SWISS_QWEN3_4B_PATH` (default `D:/dev/personal/Omnilex-Agentic-Retrieval-Competition-main/model/Qwen3-4B`).
  - HF id: `Qwen/Qwen3-4B`.
  - VRAM: ~2.7-2.8 GB.
  - Generation defaults: `max_new_tokens=2048`, `temperature=0.6`, `top_k=20`, `top_p=0.95`, `enable_thinking=False`.
  - OOM retry: clears CUDA cache and retries once with `max_new_tokens // 2`.
- Embedding model id: `Qwen/Qwen3-Embedding-4B` (env `SWISS_EMBEDDING_MODEL`).
- Cross-encoder (Stage 4b): **Qwen3-Reranker-4B** NF4 (~2.8 GB VRAM).
- Optional cloud backends: Anthropic `claude-sonnet-4-6`, OpenAI `gpt-4o`.

### Libraries / dependencies (cell 5 requirements_text)
- Python 3.12, CUDA 12.1 (RTX 4050 6GB).
- `rank-bm25==0.2.2`
- `transformers>=4.45.0`
- `bitsandbytes>=0.44.0`
- `accelerate>=1.0.0`
- `huggingface_hub>=0.25.0`
- `safetensors>=0.4.0`
- `torch>=2.4.0` (installed separately with CUDA 12.1 wheel)
- `anthropic>=0.40.0`, `openai>=1.50.0`
- `pandas>=2.0.0`, `numpy>=1.26.0`, `pyarrow>=14.0.0`
- `tqdm>=4.66.0`, `deep-translator>=1.11.0`

### BM25 retriever constants (cell 19, `retrieval/bm25_artifact.py`)
- `ARTIFACT_FORMAT = "chunked_bm25_v1"`
- `BM25_CHUNK_SIZE = max(1000, int(os.getenv("SWISS_BM25_CHUNK_SIZE", "10000")))`

### Signal-builder constants (cell 38, `stage0d_build_citation_signals.py`)
- `SCHEMA_VERSION = "citation-signals-v2"`
- `CASE_CHUNKSIZE = 200_000`
- `TEXT_SNIPPET_CHARS = 420`
- `GRAPH_NEIGHBOR_LIMIT = 8`
- `KEYWORD_LIMIT = 8`
- `ENUM_ITEM_LIMIT = 8`

### Reference graph constants (cell 40, `stage0e_build_reference_graph_v2.py`)
- `SCHEMA_VERSION = "reference-graph-v2"`
- `LAWS_CHUNKSIZE = 50_000`

### Hardware
- GPU: RTX 4050 6GB (Qwen3-4B Q4 ~2.8 GB VRAM, model loaded ONCE and reused across stages).
- CPU build: ~16 GB RAM required for BM25 index (2.47M court rows).

## Data

All paths defined in `data/data_paths.py` (cell 10). Defaults shown; env overrides via `SWISS_PIPELINE_ROOT`, `SWISS_DATA_DIR`, etc.

### Pipeline root (code / index / checkpoints / submissions)
- `PIPELINE_ROOT` env `SWISS_PIPELINE_ROOT` default `E:/swiss_law_pipeline_revised/swiss-law-pipeline`

### Raw data (env `SWISS_DATA_DIR` default `E:/data`)
- `E:/data/corpus.parquet` (LAW_CORPUS_PARQUET) — 171K law articles
- `E:/data/court_considerations.csv` (COURT_CSV) — 2.47M court rows (1.98M unique citations)
- `E:/data/laws_knowledge_base.jsonl` (KB_JSONL) — KB metadata per article
- `E:/data/laws_de.csv` (LAWS_DE_CSV)
- `E:/data/train.csv` (TRAIN_CSV)
- `E:/data/val.csv` (VAL_CSV)
- `E:/data/test.csv` (TEST_CSV)
- `E:/data/sample_submission.csv` (SAMPLE_SUBMISSION_CSV)
- `E:/data/query_translations_trainval.json` (QUERY_TRANSLATIONS_JSON, built by stage0)
- `E:/data/token_law_freq.json` (TOKEN_LAW_FREQ, built by stage0)
- `E:/data/citation_signal_schema_v2.json` (CITATION_SIGNAL_SCHEMA_JSON)
- `E:/data/statute_signals.jsonl` (STATUTE_SIGNALS_JSONL)
- `E:/data/case_signals.jsonl` (CASE_SIGNALS_JSONL)

### Index directory (`PIPELINE_ROOT/index`)
- `bm25_v2_index.pkl` (STATUTORY_BM25_PKL) — chunked BM25 artifact (combined law+court, ~2-4 GB)
- `bm25_v2_ids.pkl` (STATUTORY_BM25_IDS)
- `caselaw_bm25.pkl` / `caselaw_bm25_ids.pkl`
- `statutory_faiss.index` / `caselaw_faiss.index`
- `citation_graph.pkl` (CITATION_GRAPH_PKL)
- `cocitation_graph.pkl` (COCITATION_PKL)
- `citation_lookup.pkl` (LOOKUP_PKL)
- `citation_signal_lookup.pkl` (CITATION_SIGNAL_LOOKUP_PKL)
- `reference_graph_v2.pkl` (REFERENCE_GRAPH_V2_PKL)
- `gold_cocitation_prior_train.pkl` (GOLD_COCITATION_PRIOR_TRAIN_PKL)
- `gold_cocitation_prior_trainval.pkl` (GOLD_COCITATION_PRIOR_TRAINVAL_PKL)

### Models
- `MODELS_DIR = PIPELINE_ROOT / "models"`
- `QWEN3_4B_PATH` env `SWISS_QWEN3_4B_PATH` default `D:/dev/personal/Omnilex-Agentic-Retrieval-Competition-main/model/Qwen3-4B`

### Outputs
- `OUT_DIR = PIPELINE_ROOT / "submissions"`
- `CHECKPOINTS_DIR = PIPELINE_ROOT / "checkpoints"`
- Per-stage: `checkpoints/stage1_{split}.json` ... `checkpoints/stage5_{split}.json`
- Cross-encoder: `checkpoints/reranker_scores_{split}.json`, `checkpoints/bm25_scores_{split}.json`
- Submission: `submissions/submission_{split}.csv`

## Pipeline

### Notebook organization (cells 0-9)
Markdown front-matter cells describe a notebook that exposes the original project Python modules as source cells (using `__file__` shims) instead of importing files from disk. CLI `__main__` blocks were stripped; each stage is invoked via a notebook helper function.

### Stage 0 — Offline Index Building (CPU only, ~4-7h total, cells 28-44)

Helper functions exposed by `stage0_build_index` (cell 30) and dispatched by `run_stage0_all()` / `run_stage0_selected()` (cell 12):
- **0A. `build_bm25()`** (cell 32, `indexing/build_bm25_index.py`) — combined BM25 index over 171K law articles + ~1.98M unique court considerations (from 2.47M rows, deduplicated by grouping on citation and concatenating texts up to COURT_TEXT_LIMIT chars). Outputs `bm25_v2_index.pkl` + `bm25_v2_ids.pkl`, plus `query_translations_trainval.json` and `token_law_freq.json`. Uses PMI-based law abbreviation boosting.
- **0B. `build_citation_graph()`** (cell 34, `indexing/build_citation_graph.py`) — scans 2.47M court rows once; extracts citations from DE/FR/IT text (FR/IT normalized to DE form). Builds 5 structures: `article_to_cases`, `case_to_articles`, `case_to_cases`, `article_corpus_freq`, plus symmetric `cocitation` and `citation_doc_freq` (for PMI). Also ingests train.csv gold citations as co-citation signal.
- **0C. `build_lookup_tables()`** (cell 36, `indexing/build_lookup_tables.py`) — builds `citation_set`, `article_to_abs`, `abs_to_article`, `sr_to_abbrev` / `abbrev_to_sr` from `corpus.parquet`.
- **0D. `build_citation_signals()`** (cell 38, `stage0d_build_citation_signals.py`) — schema-v2 nested statute + case signal cards; emits `citation_signal_schema_v2.json`, `statute_signals.jsonl`, `case_signals.jsonl`, `citation_signal_lookup.pkl`.
- **0E. `build_reference_graph_v2()`** (cell 40, `stage0e_build_reference_graph_v2.py`) — typed edge sets: `article_to_articles`, `article_to_cases`, `case_to_articles`, `case_to_cases` + reverse views. Inputs `laws_de.csv`, `laws_knowledge_base.jsonl`, `citation_lookup.pkl`, `citation_graph.pkl`. Outputs `reference_graph_v2.pkl`.
- **0F. `build_gold_cocitation_prior()`** (cell 42, `stage0f_build_gold_cocitation_prior.py`) — co-citation prior from train.csv gold sets (default train-only); computes PMI/nPMI by pair-type (law-law, court-court, law-court). Optional `--trainval` variant (`build_gold_cocitation_prior_trainval()`) for competition-mode.

### Stage 1 — Query Analysis (GPU, ~7 min, cells 45-50)

6-step pipeline (cell 48, `stage1_query_analysis.py`):
1. **Explicit extraction** — regex from `retrieval/explicit_citations.py` (cell 21). Multi-citation pattern (`"Art. 38 and 39 CO"` → 2 cites). Alt-abbrev map normalizes CC→ZGB, CO→OR, LP→SchKG, CPC→ZPO, LDIP→IPRG, LFors→GestG, CPP→StPO, CP→StGB, LTF→BGG, PA→VwVG.
2. **BM25 probe** — raw EN query → top-10 statutory hits to ground later LLM reasoning.
3. **Domain scaffolding** — extract law abbreviations from probe hits; pull English names + DE keywords from KB → injected into LLM prompt as a cheat sheet.
4. **LLM analysis** — Qwen3-4B classifies domains, identifies issues with German Fachbegriffe, suggests candidate citations, generates focused DE search queries. Uses system prompt `agent/prompts/query_analyzer_prompt.txt` (cell 46): 6-step rubric (classify domain → identify issues → reason on applicable provisions → procedural cites → cross-references → echo explicit cites verbatim).
5. **HyDE** — LLM generates hypothetical DE legal-article text (style of a Swiss statute) for BM25 matching.
6. (Result merging into Stage 1 checkpoint.)

Output: `checkpoints/stage1_{split}.json`. Runner: `stage1_outputs = run_stage1(split=SPLIT, backend=BACKEND)` (cell 50).

### Stage 2 — Multi-Agent Sparse Retrieval (GPU+CPU, ~20 min, cells 51-56)

LegalMALR-adapted multi-agent loop (cell 54, `stage2_mas_retrieval.py`), all agents share one Qwen3-4B instance and only differ by system prompt (`agent/prompts/mas_prompts.py`, cell 52). Loop runs 2-4 iterations (default `STAGE2_MAX_ITERATIONS = 4`), stopping when no new candidates appear.

Agents:
- **Planner** — picks next action: REWRITE / SUPPLEMENT / DECOMPOSE / SUPPORTIVE / CROSSREF / TERMINATE. Outputs JSON `{reasoning, action}`.
- **Rewrite** — colloquial EN → precise DE legal terminology.
- **Supplement** — makes implicit conditions explicit (thresholds, standing, procedural).
- **Decompose** — splits multi-issue queries into sub-queries.
- **Supportive** — procedural / auxiliary provisions.
- **CrossRef** — constitutional + cross-referenced articles.

Each agent emits DE queries → BM25 → merge into candidate pool. Expected ~200+ candidates per query. Output: `checkpoints/stage2_{split}.json`. Runner: `stage2_outputs = run_stage2(...)` (cell 56).

### Stage 3 — Citation Graph Expansion (CPU, <1s per query, cells 57-61)

`stage3_graph_expansion.py` (cell 59). Uses `retrieval/graph_retriever.py` (cell 23):
- `articles_to_cases(article_citations, top_k_per_article=20)` — RRF score `1/(60+rank)` across articles.
- `cases_to_articles(case_citations)` — collect all cited articles.
- `bidirectional_expand(seed_articles, seed_cases, depth=1)`.

Optional gold co-citation prior controlled by `STAGE3_GOLD_PRIOR_MODE` (`off` / `train` / `trainval`). Output: `checkpoints/stage3_{split}.json`. Runner: `stage3_outputs = run_stage3(...)` (cell 61).

### Stage 4 — LLM Reranking + Direct Generation (GPU, ~10 min, cells 62-68)

`stage4_llm_reranker.py` (cell 66). Two sub-tasks:
- **4A. Reranking** — LLM rates each candidate on 0-3 scale (3 = directly applicable, 2 = closely related, 1 = tangential, 0 = not relevant). Each candidate shown with its article text from KB. Prompt: `agent/prompts/reranker_prompt.txt` (cell 63). Batch size 20. No chain-of-thought in output (LegalMALR §3.4). Outputs `{"selected": {"<citation>": <rating>, ...}}` keeping only ratings ≥ 2.
- **4B. Direct generation** — LLM generates 10-20 additional citations from memory not in the pool, grouped by Procedural / Constitutional / Substantive. Prompt: `agent/prompts/direct_gen_prompt.txt` (cell 64). Outputs `{"additional": [...]}`. Suppressible via `STAGE4_SKIP_DIRECT_GEN`.

Output: `checkpoints/stage4_{split}.json`. Runner: `stage4_outputs = run_stage4(...)` (cell 68).

### Stage 4b — Cross-Encoder Scoring (GPU, ~25 min, cells 69-72)

`stage4b_cross_encoder.py` (cell 70). Scores top-500 candidates per query (by base composite score) using **Qwen3-Reranker-4B** NF4 as a cross-encoder; emits P(yes) probability per (query, citation). Stage 5 (Strategy C) consumes these to replace binary `llm_reranker`/`tier3` signals with continuous CE score × 0.18 (`CROSS_ENCODER_WEIGHT = 0.18`).

Inputs: `stage4_{split}.json`, `bm25_scores_{split}.json`, `citation_lookup.pkl`, `laws_knowledge_base.jsonl`, `court_considerations.csv`. Output: `checkpoints/reranker_scores_{split}.json` (`{qid: {citation: float, ...}}`). Default batch_size 8; ~60 min for 10 val queries at 500 candidates. Runner: `stage4b_outputs = run_stage4b(...)` (cell 72).

### Stage 5 — Verify + Score + Threshold (CPU, ~30s, cells 73-77)

`stage5_verify_and_score.py` (cell 75). Three sub-stages:
- **5A. Verification** — `agent/verifier.py` (cell 17): normalize whitespace + abbreviation casing (via `_ABBREV_CASING` map: STGB→StGB, STPO→StPO, ... 40 entries), check existence in `citation_lookup.pkl`, drop hallucinations, expand article-level to Abs.-level when needed.
- **5B. Confidence scoring** — `scoring/confidence.py` (cell 27): per-citation composite = signal-weight sum + continuous BM25 score (per-query normalized × 0.10 up to 0.20) + optional cross-encoder × 0.18 (replacing binary reranker signals) + multi-signal bonus 0.10 when ≥3 independent signals agree. Independent signals: `explicit_from_query`, `bm25_top10`, `bm25_initial`, `citation_graph`, `llm_reranker`, `llm_reranker_tier3`, `llm_direct_gen`, `llm_stage1`.
- **5C. Threshold tuning (val only)** — sweep 0.05-0.95 to maximize macro-F1. Expected optimal range 0.15-0.30.

Output: `checkpoints/stage5_{split}.json`, `submissions/submission_{split}.csv`. Runner: `stage5_outputs = run_stage5(...)`; `submission_path = get_submission_path(SPLIT)` (cell 77).

### Pipeline orchestration (cells 78-82)
Optional one-shot: `run_full_pipeline(split, start_stage, backend, stage3_gold_prior_mode)` calls `pipeline.py` (cell 80) which sequences stages 1→5 (Stage 0 must be built separately). Commented out in cell 82.

### Submission validation (cells 83-87)
`submit.py` (cell 85) `validate_submission(submission_path)` checks columns `{query_id, predicted_citations}`, matches against `TEST_CSV` query_ids, and reports queries / avg citations / max / queries with 0. Notebook end (cell 87): prints submission path and runs validation if file exists.

## Results

The notebook contains **no executed cell outputs**. All 37 code cells have `execution_count = null` and zero items in their `outputs` list — this is a pristine source notebook, not a run artifact.

Performance numbers stated in the markdown narrative (cell 2, "Expected Performance" table) and in inline docstrings (not produced by execution in this notebook):

| Component | Estimated Recall |
| --- | --- |
| BM25 baseline (R@50) | 0.73 |
| + MAS multi-agent reformulation | +0.08-0.12 |
| + Citation graph expansion | +0.05-0.10 |
| + Explicit citation extraction | +0.02-0.05 |
| + LLM direct generation | +0.03-0.05 |
| **Combined recall estimate** | **0.85-0.92** |
| After LLM reranking (precision) | 0.65-0.75 |
| **Estimated F1** | **0.70-0.80** |

Other quantitative claims embedded in docstrings:
- BM25 R@50 = 0.73 vs Dense R@100 = 0.24 (justification for BM25-first design).
- 59.4% of val citations statutory, 40.6% case law; train data 98.8% statutory.
- Stage 4b validated on all 10 val queries: macro-F1 0.5400 → 0.6520 with `CROSS_ENCODER_WEIGHT = 0.18` (`scoring/confidence.py` docstring, cell 27).
- BM25 build: 2.47M court rows → 1.98M unique citations (deduplicated by citation grouping).
- Full pipeline timing: ~65-105 min for 40 queries (Stage 0 excluded).

## Summary

This notebook is a self-contained, never-executed "source-as-cells" view of the BM25-first, LegalMALR-adapted Swiss law citation pipeline (Stages 0-5 + Stage 4b cross-encoder). Each project module is pasted into a code cell with a `__file__` shim so imports behave like a normal module, CLI `__main__` blocks are stripped, and stage execution is exposed through thin notebook runners (`run_stage1` ... `run_stage5`, `run_stage4b`, `run_full_pipeline`). The architecture is: BM25 over a combined 171K-law + 1.98M-court index, multi-agent Qwen3-4B (NF4) for query reformulation, citation-graph expansion, LLM reranker + direct generation, Qwen3-Reranker-4B cross-encoder, then verification/confidence/threshold-tuning. No metrics were produced here — performance claims (BM25 R@50 0.73, estimated combined F1 0.70-0.80, Stage 4b macro-F1 0.5400→0.6520) come from prose in the project README/docstrings, not from cell outputs. The notebook functions as documentation and a one-shot orchestration shell rather than an experimental record.
