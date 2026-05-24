# pipeline_end_to_end_no_function_imports.ipynb

**Path:** notebooks/04_pre_v75_pipeline_iterations/pipeline_end_to_end_no_function_imports.ipynb

## Configuration

### Runtime settings (notebook cell 24)
- `SPLIT = "test"` ("train", "val", or "test")
- `BACKEND = "local"` ("local", "anthropic", or "openai")
- `START_STAGE = 1` (set to 0 only if rebuilding Stage 0 artifacts)
- `STAGE3_GOLD_PRIOR_MODE = "off"` ("off", "train", or "trainval")
- `STAGE2_MAX_ITERATIONS = 4`
- `STAGE4_SKIP_DIRECT_GEN = False`
- `STAGE4B_TOP_N = None` (stage default)
- `STAGE5_THRESHOLD = None` (stage default)

### `configs/default.json` (cell 26)
```json
{
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

### Models
- `QWEN3_32B_PATH = /kaggle/input/models/qwen-lm/qwen-3/transformers/32b/1`
- `QWEN3_32B_HF_ID = "Qwen/Qwen3-32B"` (loaded once via singleton; bfloat16, device_map="auto"; reported VRAM ~64GB bf16; doc-string also mentions ~2.7-2.8 GB with NF4 4-bit quantization)
- `QWEN3_RERANKER_ID = "Qwen/Qwen3-Reranker-4B"` (Stage 4b cross-encoder)
- Default generation: `max_new_tokens=2048`, `temperature=0.6`, `top_k=20`, `top_p=0.95`, `enable_thinking=False` (Stage 1 uses True, Stage 4 reranker uses False)

### Libraries (cell 5 imports / cell 23 requirements)
- Standard lib: argparse, csv, gc, importlib, json, math, os, pickle, random, re, sys, threading, time, traceback, collections, concurrent.futures, itertools, pathlib, typing
- Third-party: numpy, pandas, torch, deep_translator (GoogleTranslator), rank_bm25 (BM25Okapi), scipy (sparse), tqdm, transformers (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)
- Notebook-only install (cell 4): `%pip install -qU deep-translator rank-bm25`
- requirements_text pinned versions: `rank-bm25==0.2.2`, `transformers>=4.45.0`, `bitsandbytes>=0.44.0`, `accelerate>=1.0.0`, `huggingface_hub>=0.25.0`, `safetensors>=0.4.0`, `anthropic>=0.40.0`, `openai>=1.50.0`, `pandas>=2.0.0`, `numpy>=1.26.0`, `pyarrow>=14.0.0`, `tqdm>=4.66.0`, `deep-translator>=1.11.0`
- Target environment per requirements header: Python 3.12, CUDA 12.1, RTX 4050 6GB

### BM25 internals (cell 7)
- `ARTIFACT_FORMAT = "chunked_bm25_v1"`
- `BM25_CHUNK_SIZE = max(1000, int(os.getenv("SWISS_BM25_CHUNK_SIZE", "10000")))`
- `ChunkedBM25Okapi`: sparse-CSC backed BM25; init params k1, b, epsilon supplied externally

### Confidence scoring weights (cells 8, 45 — `scoring/confidence.py`)
```
DEFAULT_WEIGHTS = {
    "explicit_from_query":    0.40,
    "bm25_top10":             0.15,   # binary
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
CROSS_ENCODER_WEIGHT   = 0.18    # P(yes) * 0.18 replaces binary reranker signals
MULTI_SIGNAL_BONUS     = 0.10    # added when 3+ independent signals agree
_INDEPENDENT_SIGNALS   = {explicit_from_query, bm25_top10, bm25_initial,
                          citation_graph, llm_reranker, llm_reranker_tier3,
                          llm_direct_gen, llm_stage1}
```

### Citation graph build constants (cell 11)
- `CHUNKSIZE = 200_000`
- `MAX_CITES_PER_ROW = 30`
- `MIN_CO_OCCURRENCE = 3`

### Path / directory variables (cell 21)
- `PROJECT_ROOT = /mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline`
- `RAW_DATA_DIR = $SWISS_DATA_DIR` or `PROJECT_ROOT/data`
- `INDEX_DIR = PROJECT_ROOT/index`
- `CHECKPOINTS_DIR = PROJECT_ROOT/checkpoints`
- `SUBMISSIONS_DIR = PROJECT_ROOT/submissions`
- `MODELS_DIR = PROJECT_ROOT/models`

### Kaggle output directories (cell 5)
- `OUT_DIR = /kaggle/working/submissions`
- `CHECKPOINTS_DIR = /kaggle/working/checkpoints`

## Data

### Dataset 1 — law-db (Kaggle)
- `LAW_DB = /kaggle/input/datasets/samiulislam180041221/law-db`
- `BM25_PARTS_DIR = /kaggle/input/datasets/samiulislam180041221/law-db/bm25_v2_index_parts/bm25_v2_index_parts`
- `LAW_CORPUS_PARQUET = /kaggle/input/datasets/samiulislam180041221/law-db/corpus.parquet`
- `KB_JSONL = /kaggle/input/datasets/samiulislam180041221/law-db/laws_knowledge_base.jsonl`
- `STATUTORY_BM25_PKL = /kaggle/input/datasets/samiulislam180041221/law-db/bm25_v2_index.pkl`
- `STATUTORY_BM25_IDS = /kaggle/input/datasets/samiulislam180041221/law-db/bm25_v2_ids.pkl`
- `CITATION_GRAPH_PKL = /kaggle/input/datasets/samiulislam180041221/law-db/citation_graph.pkl`
- `LOOKUP_PKL = /kaggle/input/datasets/samiulislam180041221/law-db/citation_lookup.pkl`
- `CITATION_SIGNAL_LOOKUP_PKL = LAW_DB/citation_signal_lookup.pkl`
- `REFERENCE_GRAPH_V2_PKL = /kaggle/input/datasets/samiulislam180041221/law-db/reference_graph_v2.pkl`
- `GOLD_COCITATION_PRIOR_TRAIN_PKL = /kaggle/input/datasets/samiulislam180041221/law-db/gold_cocitation_prior_train.pkl`
- `GOLD_COCITATION_PRIOR_TRAINVAL_PKL = /kaggle/input/datasets/samiulislam180041221/law-db/gold_cocitation_prior_trainval.pkl`
- Legacy aliases under `LAW_DB`: `query_translations_trainval.json`, `token_law_freq.json`, `citation_signal_schema_v2.json`, `statute_signals.jsonl`, `case_signals.jsonl`

### Dataset 2 — competition CSVs
- `COMP_DATA = /kaggle/input/datasets/moniruzzamanmahadi8/llm-agentic-legal-information-retrieval-dataset/llm-agentic-legal-information-retrieval`
- `COURT_CSV = COMP_DATA/court_considerations.csv` (~2.47M court rows)
- `LAWS_DE_CSV = COMP_DATA/laws_de.csv`
- `TRAIN_CSV = COMP_DATA/train.csv`
- `VAL_CSV = COMP_DATA/val.csv`
- `TEST_CSV = COMP_DATA/test.csv`
- `SAMPLE_SUBMISSION_CSV = COMP_DATA/sample_submission.csv`

### Stage 0 inputs (per `stage0_build_index.py` docstring, cell 48)
- `data/corpus.parquet` (171K law articles)
- `data/court_considerations.csv` (2.47M court rows)
- `data/laws_knowledge_base.jsonl`
- `data/train.csv` + `data/val.csv`

### Stage 0 outputs (per docstring)
- `index/bm25_v2_index.pkl` (~2-4 GB)
- `index/bm25_v2_ids.pkl`
- `index/citation_graph.pkl`
- `index/citation_lookup.pkl`
- `index/citation_signal_lookup.pkl`
- `index/reference_graph_v2.pkl`
- `index/gold_cocitation_prior_train.pkl`
- `index/gold_cocitation_prior_trainval.pkl` (optional)
- `data/query_translations_trainval.json`
- `data/token_law_freq.json`
- `data/statute_signals.jsonl`
- `data/case_signals.jsonl`

### Per-stage checkpoint outputs
- `checkpoints/stage1_{split}.json`
- `checkpoints/stage2_{split}.json`
- `checkpoints/stage3_{split}.json`
- `checkpoints/stage4_{split}.json`
- `checkpoints/stage5_{split}.json`
- Final submission: `submissions/submission_{split}.csv`

## Pipeline

The notebook holds 106 cells. Code is structured as: (1) config block defining paths, models, and runtime knobs; (2) inlined source from project modules (utility/source cells) with no CLI `__main__` blocks; (3) per-stage runner cells; (4) optional whole-pipeline runner and submission validator.

### Stage 0 — Offline index building (CPU only, one-time)
Cells 47–62. Inlined source for `stage0_build_index.py`, `indexing/build_bm25_index.py`, `indexing/build_citation_graph.py`, `indexing/build_lookup_tables.py`, `stage0d_build_citation_signals.py`, `stage0e_build_reference_graph_v2.py`, `stage0f_build_gold_cocitation_prior.py`. Builds:
- 0A. BM25 combined index over 171K laws + ~2M court considerations
- 0B. Bidirectional Art. ↔ court-decision citation graph
- 0C. Citation lookup tables (normalization, existence, Abs. expansion, SR↔abbrev mapping)
- 0D. Deterministic statute/case signal cards
- 0E. Typed reference graph v2 (statute/case edges + reverse views)
- 0F. Train-only gold co-citation prior (and optional train+val variant)

Runner cell 62 only invokes `run_stage0_all()` when `START_STAGE == 0`; otherwise prints a skip message.

### Stage 1 — Query analysis (GPU, ~7 min target)
Cells 63–68. Inlines `agent/prompts/query_analyzer_prompt.txt` and `stage1_query_analysis.py`. Per query: extracts explicit citations by regex, classifies legal domains, identifies legal issues, generates 3-5 German search queries. Runner: `stage1_outputs = run_stage1(split=SPLIT, backend=BACKEND)`.

### Stage 2 — Multi-agent sparse retrieval (GPU+CPU, ~20 min target)
Cells 69–74. Inlines `agent/prompts/mas_prompts.py` (PLANNER, REWRITE, SUPPLEMENT, DECOMPOSE, SUPPORTIVE, CROSSREF agent prompts) and `stage2_mas_retrieval.py`. Per query: 2-4 iterations of Planner → Agent → BM25; agents generate DE queries; target ~200+ candidates. Runner: `stage2_outputs = run_stage2(split=SPLIT, backend=BACKEND, max_iterations=STAGE2_MAX_ITERATIONS)`.

### Stage 3 — Citation graph expansion (CPU, ~30 s target)
Cells 75–79. Inlines `stage3_graph_expansion.py`. Per statute, adds top 5 citing BGE decisions; per BGE decision, adds all cited statutes. Runner: `stage3_outputs = run_stage3(split=SPLIT, gold_prior_mode=STAGE3_GOLD_PRIOR_MODE)`.

### Stage 4 — LLM reranker + direct generation (GPU, ~10 min target)
Cells 80–86. Inlines `agent/prompts/reranker_prompt.txt`, `agent/prompts/direct_gen_prompt.txt`, `stage4_llm_reranker.py`. Reranker selects relevant citations (precision) and generates additional citations from memory (recall). No chain-of-thought in output. Runner: `stage4_outputs = run_stage4(split=SPLIT, backend=BACKEND, skip_direct_gen=STAGE4_SKIP_DIRECT_GEN)`.

### Stage 4b — Cross-encoder reranker
Cells 87–90. Inlines `stage4b_cross_encoder.py`. Uses Qwen3-Reranker-4B to produce continuous P(yes) scores for candidates; weight 0.18 in confidence scoring replaces binary reranker signals when available. Runner: `stage4b_outputs = run_stage4b(split=SPLIT, top_n=STAGE4B_TOP_N) if STAGE4B_TOP_N is not None else run_stage4b(split=SPLIT)`.

### Stage 5 — Verify, score, threshold (CPU, ~30 s target)
Cells 91–95. Inlines `stage5_verify_and_score.py`. 5A normalizes citations, checks existence, drops hallucinations. 5B composite confidence scoring (additive weights + continuous BM25 + multi-signal bonus + cross-encoder P(yes)). 5C sweeps thresholds 0.05–0.95 to maximize macro-F1 on val (default optimal ~0.15-0.30, low/inclusive). Runner: `stage5_outputs = run_stage5(split=SPLIT, threshold=STAGE5_THRESHOLD)` then prints `submission_path`.

### Pipeline orchestration
Cells 96–100. Inlines `pipeline.py`. Optional one-shot runner cell calls `run_full_pipeline(split, start_stage, backend, stage3_gold_prior_mode)` — commented out by default.

### Submission validation
Cells 101–105. Inlines `submit.py`. Final cell prints submission path; if it exists, calls `validate_generated_submission(SPLIT)` and prints `Submission valid: ...`.

## Results

No code cell in the notebook has any stored outputs. All `outputs` arrays are empty (verified across all 106 cells), so there are no executed numbers, metrics, errors, or runtime traces in this file. The notebook is shipped pre-execution.

Numerical targets quoted inside the notebook (markdown overview only, not measured here):
- BM25 baseline R@50: 0.73
- + MAS multi-agent reformulation: +0.08-0.12
- + Citation graph expansion: +0.05-0.10
- + Explicit citation extraction: +0.02-0.05
- + LLM direct generation: +0.03-0.05
- Combined recall estimate: 0.85-0.92
- After LLM reranking (precision): 0.65-0.75
- Estimated F1: 0.70-0.80
- Cross-encoder weight calibration (from `scoring/confidence.py` comment): "Validated at 0.18 on all 10 val queries: 0.5400 → 0.6520 macro-F1"

## Summary

This notebook is the notebook-clean wiring of the full BM25-first / LegalMALR-adapted multi-agent Swiss-law citation pipeline (Stages 0–5 + 4b cross-encoder) into a single Kaggle-ready file with no CLI `__main__` blocks. Configuration is concentrated in early cells: `SPLIT="test"`, `BACKEND="local"`, `START_STAGE=1` (reusing prebuilt Stage 0 artifacts on the law-db Kaggle dataset), with Qwen3-32B as the single shared GPU model and Qwen3-Reranker-4B as the cross-encoder. The notebook inlines the verbatim source of each project module (`agent/`, `retrieval/`, `scoring/`, `indexing/`, `stage*_*.py`, `pipeline.py`, `submit.py`) so the pipeline can run without `function_imports`; runner cells then call `run_stage1` through `run_stage5` plus `run_stage4b` and a final submission validator. It has not been executed: every output cell is empty, so no real recall, precision, F1, threshold, or runtime numbers are recorded; the only quoted figures are the design-target table in the project-overview markdown and a comment that cross-encoder weight 0.18 lifted macro-F1 from 0.5400 to 0.6520 on the 10 val queries. The take-away is that this file is a self-contained orchestration scaffold (test-split submission generator) ready to be run end-to-end against the pre-built indexes; producing actual metrics requires executing it.
