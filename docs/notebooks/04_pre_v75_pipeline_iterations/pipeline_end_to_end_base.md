# pipeline_end_to_end_base.ipynb

**Path:** notebooks/04_pre_v75_pipeline_iterations/pipeline_end_to_end_base.ipynb

This notebook is a documentation consolidation of the `swiss-law-pipeline` codebase. It contains 68 cells (24 code, 44 markdown). All code cells have `execution_count = null` and zero outputs — the notebook was never run; it stitches together repo source files verbatim with section markdown for readability.

## Configuration

### Hardware
- GPU: RTX 4050 6GB (declared target in `requirements.txt`)
- CUDA: 12.1
- Python: 3.12
- RAM expectation: ~16 GB free for Stage 0 BM25 build

### Models
- LLM (singleton, serves Stages 1, 2, 4): `Qwen/Qwen3-4B` loaded with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`, `device_map="auto"`, VRAM ~2.7–2.8 GB
- Local model path env: `SWISS_QWEN3_4B_PATH`, default `D:/dev/personal/Omnilex-Agentic-Retrieval-Competition-main/model/Qwen3-4B`
- HF fallback ID: `Qwen/Qwen3-4B`
- Embedding model (declared, not used in this BM25-first build): `Qwen/Qwen3-Embedding-4B`
- Cross-encoder (Stage 4b): `Qwen/Qwen3-Reranker-4B`, NF4, batch_size=8, top-N=500 candidates per query

### Generation defaults (`agent/llm_backend.py:generate`)
- `max_new_tokens=2048`, `temperature=0.6`, `top_k=20`, `top_p=0.95`, `enable_thinking=False` (Stage 1 uses `True`, Stage 4 reranker `False`)

### Libraries (`requirements.txt`)
```
rank-bm25==0.2.2
transformers>=4.45.0
bitsandbytes>=0.44.0
accelerate>=1.0.0
huggingface_hub>=0.25.0
safetensors>=0.4.0
anthropic>=0.40.0
openai>=1.50.0
pandas>=2.0.0
numpy>=1.26.0
pyarrow>=14.0.0
tqdm>=4.66.0
deep-translator>=1.11.0
```

### Pipeline config (`configs/default.json`)
- `stage1_backend`: `"local"`, `stage1_model`: `null`
- `bm25_top_k`: 100, `graph_top_k`: 50
- `mas_max_iterations`: 4, `mas_bm25_top_k`: 30
- `reranker_backend`: `"local"`, `reranker_batch_size`: 20
- `confidence_weights`: `explicit_from_query=0.40`, `bm25_top10=0.25`, `citation_graph=0.15`, `llm_reranker=0.15`, `llm_direct_gen=0.05`
- `threshold`: 0.15

### Scoring constants (`scoring/confidence.py`)
- `DEFAULT_WEIGHTS`: `explicit_from_query=0.40`, `bm25_top10=0.15`, `bm25_initial=0.10`, `citation_graph=0.10`, `llm_reranker_tier3=0.25`, `llm_reranker=0.15`, `llm_direct_gen=0.05`, `llm_stage1=0.05`, `llm_stage1_procedural=0.05`, `co_citation=0.03`
- `BM25_CONTINUOUS_WEIGHT = 0.10`
- `CROSS_ENCODER_WEIGHT = 0.18` (Strategy C, validated 0.5400 → 0.6520 macro-F1 on 10 val queries)

### BM25 constants
- Artifact format: `chunked_bm25_v1`
- `BM25_CHUNK_SIZE = max(1000, int(os.getenv("SWISS_BM25_CHUNK_SIZE", "10000")))`
- Tokenizer regex: `[^\w\d]+`, keeps tokens with `len ≥ 2` or numeric

## Data

Paths from `data/data_paths.py`:

### Pipeline root
- `PIPELINE_ROOT` env `SWISS_PIPELINE_ROOT`, default `E:/swiss_law_pipeline_revised/swiss-law-pipeline`

### Raw inputs (`DATA` env `SWISS_DATA_DIR`, default `E:/data`)
- `E:/data/corpus.parquet` — 171K law articles
- `E:/data/court_considerations.csv` — 2.47M court rows (1.98M unique citations after dedup)
- `E:/data/laws_knowledge_base.jsonl` — KB metadata per article
- `E:/data/laws_de.csv`
- `E:/data/train.csv`
- `E:/data/val.csv`
- `E:/data/test.csv`
- `E:/data/sample_submission.csv`
- `E:/data/query_translations_trainval.json` (built by stage 0)
- `E:/data/token_law_freq.json` (built by stage 0)
- `E:/data/citation_signal_schema_v2.json`
- `E:/data/statute_signals.jsonl`
- `E:/data/case_signals.jsonl`

### Index outputs (under `PIPELINE_ROOT/index/`)
- `bm25_v2_index.pkl`, `bm25_v2_ids.pkl`
- `caselaw_bm25.pkl`, `caselaw_bm25_ids.pkl`
- `statutory_faiss.index`, `caselaw_faiss.index`
- `citation_graph.pkl`, `cocitation_graph.pkl`
- `citation_lookup.pkl`, `citation_signal_lookup.pkl`
- `reference_graph_v2.pkl`
- `gold_cocitation_prior_train.pkl`, `gold_cocitation_prior_trainval.pkl`

### Other outputs
- `PIPELINE_ROOT/submissions/`, `PIPELINE_ROOT/checkpoints/`, `PIPELINE_ROOT/models/`

### Composition note (in `build_bm25_index.py`)
- Combined index over ~171K law articles and ~1.98M unique court citations (deduped from 2.47M rows via concat up to `COURT_TEXT_LIMIT` chars per citation group)

## Pipeline

### Stage 0 — Offline index building (CPU, ~3–7 h, one-time)
- 0A. Build combined BM25 index over law + court corpora (`indexing/build_bm25_index.py`)
- 0B. Build bidirectional citation graph Art. ↔ Court decision (`indexing/build_citation_graph.py`)
- 0C. Build citation lookup tables: normalization, existence, Abs.-expansion (`indexing/build_lookup_tables.py`)
- 0D. Build deterministic citation signal cards (`stage0d_build_citation_signals.py`)
- 0E. Build typed reference graph v2 (`stage0e_build_reference_graph_v2.py`)
- 0F. Build train-only gold co-citation prior; 0F* optional train+val variant (`stage0f_build_gold_cocitation_prior.py`)
- Flags: `--bm25-only`, `--graph-only`, `--lookup-only`, `--signals-only`, `--refgraph-only`, `--goldprior-only`, `--goldprior-trainval-only`

### Stage 1 — Query analysis (GPU, ~7 min for 40 queries)
Six steps in `stage1_query_analysis.py`:
1. Regex explicit citation extraction (`retrieval/explicit_citations.py`) — free recall
2. BM25 probe of raw EN query (CPU, <1 s) — top-10 results expose relevant law abbreviations
3. Domain scaffolding from KB (StPO, ATSG, BGG, ...) — DE keywords as LLM cheat sheet
4. Qwen3-4B analysis with scaffolding injected — classify domains, identify issues with German Fachbegriffe, suggest candidate citations, generate DE search queries (`enable_thinking=True`)
5. HyDE — LLM generates hypothetical DE statutory paragraph for BM25 matching
6. Cross-lingual EN→DE term expansion from KB
Output: `checkpoints/stage1_{split}.json`

### Stage 2 — Multi-Agent Sparse retrieval (GPU+CPU, ~20 min)
`stage2_mas_retrieval.py` runs LegalMALR-adapted planner loop, 2–4 iterations (`mas_max_iterations=4`, `mas_bm25_top_k=30`). Agents (all share Qwen3-4B; differ by system prompt in `agent/prompts/mas_prompts.py`):
- PLANNER — picks next agent or TERMINATE; outputs JSON `{reasoning, action}`
- REWRITE — colloquial EN → precise DE Fachbegriffe for BM25
- SUPPLEMENT — makes implicit conditions explicit
- DECOMPOSE — splits multi-issue queries
- SUPPORTIVE — procedural/auxiliary provisions
- CROSSREF — constitutional + cross-referenced articles
Each agent emits DE queries → BM25 → merged into candidate pool (~200+ per query expected).
Output: `checkpoints/stage2_{split}.json`

### Stage 3 — Citation graph expansion (CPU, <1 s/query, ~30 s total)
`stage3_graph_expansion.py` via `retrieval/graph_retriever.py`:
- Each statute → top-5 citing court decisions (RRF-style score `1/(60+rank)`, default `top_k_per_article=20`)
- Each case → cited statutes
- Recall multiplier for case law (val is 40.6% case law vs 98.8% statutory in train)
Output: `checkpoints/stage3_{split}.json`

### Stage 4 — LLM reranker + direct generation (GPU, ~10 min; ~35–60 min with text in batches of 50)
`stage4_llm_reranker.py`:
- 4A. Rerank: Qwen3-4B sees candidate text from KB and selects relevant items (no chain-of-thought in output, LegalMALR §3.4). Pre-filter to ~1200 candidates excluding co-citation-only noise.
- 4B. Direct generation: LLM emits memory-based extra citations; verified later in Stage 5.
Output: `checkpoints/stage4_{split}.json`

### Stage 4b — Cross-encoder scoring (GPU, ~25 min full / ~60 min for 10 val queries at batch 8)
`stage4b_cross_encoder.py` scores top-500 candidates per query with `Qwen/Qwen3-Reranker-4B` NF4, batch_size=8, emitting P(yes). Output: `checkpoints/reranker_scores_{split}.json` mapped `{qid: {citation: float}}`. Consumed by Stage 5 Strategy C (replaces binary reranker/tier3 with continuous score × 0.18).

### Stage 5 — Verify, score, threshold (CPU, ~30 s)
`stage5_verify_and_score.py` via `agent/verifier.py` and `scoring/confidence.py`:
- 5A. Normalize citation formatting; fix abbreviation casing via `_ABBREV_CASING` map (e.g. `STGB`→`StGB`, `STPO`→`StPO`, ~40 entries); check existence in corpus; drop hallucinations; expand article-level to Abs.-level
- 5B. Composite score = weighted sum of signals + continuous BM25 (per-query normalized) up to 0.20 + graduated multi-signal bonus (2+, 3+, 4+ independent signals)
- 5C. (val only) Sweep thresholds 0.05–0.95 to maximize macro-F1; expected optimum 0.15–0.30
Output: `checkpoints/stage5_{split}.json`, `submissions/submission_{split}.csv`

### Pipeline orchestrator (`pipeline.py`)
Runs Stage 1 → 5 sequentially per split (val/test). Supports `--stage N` resume. Verifies Stage 0 prerequisites (`bm25_v2_index.pkl`, `citation_graph.pkl`, `citation_lookup.pkl`). Total ~65–105 min for 40 queries.

### Submission (`submit.py`)
Generates and validates CSV. Validation checks columns `{query_id, predicted_citations}`, alignment with `test.csv` query IDs, and reports `Queries / Avg citations / Max citations / Queries with 0`.

### Explicit citation extraction (`retrieval/explicit_citations.py`)
Alt-abbreviation canonicalization map (always applied so FR/IT abbreviations resolve to DE corpus form):
- `CC→ZGB`, `CO→OR`, `LP→SchKG`, `CPC→ZPO`, `LDIP→IPRG`, `LFors→GestG`, `CPP→StPO`, `CP→StGB`, `LTF→BGG`, `PA→VwVG`
Regex matchers: `_SINGLE_STAT_PATTERN` and `_MULTI_STAT_PATTERN` (e.g. `Art. 221 Abs. 1 lit. b StPO`, multi-article joined by `,`/`and`/`und`/`or`/`oder`).

## Results

The notebook is unexecuted: every code cell has `execution_count = null` and `outputs = []`. There are no runtime metrics, no error tracebacks, and no produced artifacts within the notebook itself.

Performance numbers stated inside embedded source comments/markdown (not measured by this notebook):

| Item | Value (as stated in source) |
|---|---|
| BM25 baseline R@50 | 0.73 |
| Dense baseline R@100 | 0.24 |
| MAS multi-agent uplift | +0.08–0.12 recall |
| Citation graph uplift | +0.05–0.10 recall |
| Explicit citation extraction uplift | +0.02–0.05 recall |
| LLM direct generation uplift | +0.03–0.05 recall |
| Combined recall estimate | 0.85–0.92 |
| After LLM reranking (precision) | 0.65–0.75 |
| Estimated F1 | 0.70–0.80 |
| Strategy C cross-encoder (10 val queries) | 0.5400 → 0.6520 macro-F1 |
| Val gold composition | 59.4% statutory, 40.6% case law |
| Train gold composition | 98.8% statutory |
| Expected optimal threshold | 0.15–0.30 |
| Default threshold (config) | 0.15 |

Errors / failure modes documented in `README.md` troubleshooting section: OOM (Qwen3-4B NF4 ~2.8 GB), slow BM25 build (2.47M court rows, 16 GB RAM bottleneck), low F1 diagnostics per stage, resume support via incremental checkpoints.

## Summary

The notebook is a single ordered dump of the `swiss-law-pipeline` repository (README, `requirements.txt`, `configs/default.json`, `data/data_paths.py`, all stage scripts, agent prompts, retrievers, scorer, orchestrator, submission tool) wrapped with section markdown. It serves as a reading reference for the BM25-first, LegalMALR-adapted pipeline architecture: Qwen3-4B singleton driving Stages 1/2/4, BM25 over combined 171K-laws + 1.98M-court-citations index, citation graph for case-law recall, optional Qwen3-Reranker-4B cross-encoder at Stage 4b, and confidence-weighted F1 thresholding at Stage 5. Because no code was executed, the notebook produces no metrics — every recall/F1 number it carries is an estimate or prior measurement copied from source comments (notably the 0.5400 → 0.6520 macro-F1 cross-encoder result on 10 val queries with `CROSS_ENCODER_WEIGHT=0.18`). The value is documentary: it captures hyperparameters, signal weights, file layout, and run commands in one place for the pre-v7.5 pipeline iteration. Lesson implicit in the artifact: the system was designed around BM25 dominance (R@50=0.73 vs dense R@100=0.24), 4B-LLM-with-scaffolding rather than larger models, and explicit Stage 0 / lookup-table / co-citation-prior preprocessing to compensate for the 98.8%-statutory train / 40.6%-case-law val mismatch.
