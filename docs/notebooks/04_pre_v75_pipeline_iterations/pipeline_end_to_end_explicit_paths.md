# pipeline_end_to_end_explicit_paths.ipynb

**Path:** `notebooks/04_pre_v75_pipeline_iterations/pipeline_end_to_end_explicit_paths.ipynb`

A notebook-native packaging of the full 5-stage Swiss-law citation retrieval pipeline (BM25-first, LegalMALR-adapted MAS). Every project source file is inlined as a reference cell with a `__file__` shim, and stage runners are wrapped in `run_stageN(...)` helpers. Authored to make every path explicit at the top of the notebook and run inside a Linux/Colab-style environment (`/mnt/data/...`).

## Configuration

### Runtime constants (cell 4 / cell 8)

- `PROJECT_ROOT = /mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline` (resolved at runtime, `os.chdir`'d into).
- `RAW_DATA_DIR = ${SWISS_DATA_DIR}` env override, else `PROJECT_ROOT/data`.
- `INDEX_DIR = PROJECT_ROOT/index`, `CHECKPOINTS_DIR = PROJECT_ROOT/checkpoints`, `SUBMISSIONS_DIR = PROJECT_ROOT/submissions`, `MODELS_DIR = PROJECT_ROOT/models`.
- Env exports: `SWISS_PIPELINE_ROOT`, `SWISS_DATA_DIR`.
- `SPLIT = "test"`, `BACKEND = "local"`, `START_STAGE = 1`, `STAGE3_GOLD_PRIOR_MODE = "off"`.
- `STAGE2_MAX_ITERATIONS = 4`, `STAGE4_SKIP_DIRECT_GEN = False`, `STAGE4B_TOP_N = None`, `STAGE5_THRESHOLD = None`.

### `configs/default.json` (cell 9 / cell 10)

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

### Models (`data/data_paths.py`, cell 12)

- LLM: `Qwen/Qwen3-4B` (4-bit NF4, bnb), local path default `D:/dev/personal/Omnilex-Agentic-Retrieval-Competition-main/model/Qwen3-4B`, HF id `Qwen/Qwen3-4B`. Env overrides: `SWISS_QWEN3_4B_PATH`, `SWISS_QWEN3_4B_HF_ID`.
- Embedding/cross-encoder model id: `Qwen/Qwen3-Embedding-4B` (Stage 4b uses Qwen3-Reranker-4B per pipeline docstring).
- VRAM target: ~2.7-2.8 GB on RTX 4050 6 GB.

### Generation hyperparameters (`agent/llm_backend.py`, cell 19)

- `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`.
- `generate(max_new_tokens=2048, temperature=0.6, top_k=20, top_p=0.95, enable_thinking=False)`.
- JSON OOM retry: clears cache and re-runs with `max_new_tokens // 2`.
- Optional cloud backends: `claude-sonnet-4-6` (anthropic), `gpt-4o` (openai).

### BM25 / scoring constants

- `retrieval/bm25_artifact.py`: `BM25_CHUNK_SIZE = max(1000, int(os.getenv("SWISS_BM25_CHUNK_SIZE", "10000")))`; artifact format `chunked_bm25_v1`.
- `scoring/confidence.py` `DEFAULT_WEIGHTS`: `explicit_from_query=0.40, bm25_top10=0.15, bm25_initial=0.10, citation_graph=0.10, llm_reranker_tier3=0.25, llm_reranker=0.15, llm_direct_gen=0.05, llm_stage1=0.05, llm_stage1_procedural=0.05, co_citation=0.03`.
- Additional: `BM25_CONTINUOUS_WEIGHT = 0.10`, `CROSS_ENCODER_WEIGHT = 0.18`, `MULTI_SIGNAL_BONUS = 0.10` (graduated: +0.20 for >=4 signals, +0.10 for 3, +0.05 for 2).
- F1 threshold sweep: `np.arange(0.05, 0.95+0.01, 0.01)`.

### Libraries (`requirements.txt` shown in cell 7)

- Python 3.12, CUDA 12.1.
- `rank-bm25==0.2.2`, `transformers>=4.45.0`, `bitsandbytes>=0.44.0`, `accelerate>=1.0.0`, `huggingface_hub>=0.25.0`, `safetensors>=0.4.0`.
- `torch>=2.4.0` installed via cu121 wheel.
- `anthropic>=0.40.0`, `openai>=1.50.0` (optional ensemble backends).
- `pandas>=2.0.0`, `numpy>=1.26.0`, `pyarrow>=14.0.0`, `tqdm>=4.66.0`, `deep-translator>=1.11.0`.

### Citation normalization (`agent/verifier.py`, cell 21)

Hardcoded UPPERCASE-to-mixed-case abbreviation map `_ABBREV_CASING` covering 40+ Swiss law abbreviations (StGB, StPO, SchKG, ZGB, OR, BGG, ATSG, IPRG, ZPO, BV, ...). Multilingual canonicalization `_ALT_ABBREV_TO_CANON` for FR/IT: `CC->ZGB, CO->OR, LP->SchKG, CPC->ZPO, LDIP->IPRG, LFors->GestG, CPP->StPO, CP->StGB, LTF->BGG, PA->VwVG`.

## Data

All paths are declared in `data/data_paths.py` (cell 12) and overridable via env vars.

### Raw inputs (under `DATA = ${SWISS_DATA_DIR}` or `E:/data`)

- `corpus.parquet` (171K law articles)
- `court_considerations.csv` (2.47M court rows, ~1.98M unique citations)
- `laws_knowledge_base.jsonl` (KB metadata per article)
- `laws_de.csv`
- `train.csv`, `val.csv`, `test.csv`
- `sample_submission.csv`
- `query_translations_trainval.json` (built by Stage 0)
- `token_law_freq.json` (PMI token->law map, built by Stage 0)
- `citation_signal_schema_v2.json`
- `statute_signals.jsonl`, `case_signals.jsonl`

### Index artifacts (under `INDEX_DIR = PROJECT_ROOT/index`)

- `bm25_v2_index.pkl` + `bm25_v2_index_parts/doc_freqs_*.pkl, idf.pkl, doc_len.npy` (chunked BM25)
- `bm25_v2_ids.pkl`
- `caselaw_bm25.pkl`, `caselaw_bm25_ids.pkl`
- `statutory_faiss.index`, `caselaw_faiss.index` (defined; dense retrieval dropped)
- `citation_graph.pkl`
- `cocitation_graph.pkl`
- `citation_lookup.pkl`
- `citation_signal_lookup.pkl`
- `reference_graph_v2.pkl`
- `gold_cocitation_prior_train.pkl`
- `gold_cocitation_prior_trainval.pkl`

### Models / outputs

- `MODELS_DIR = PROJECT_ROOT/models`
- `CHECKPOINTS_DIR/stage{1..5}_{split}.json`
- `OUT_DIR = PROJECT_ROOT/submissions/submission_{split}.csv`

## Pipeline

### Stage 0 - Offline Index Build (CPU, ~4-7 h one-time)

Inlined script `stage0_build_index.py` (cell 34) with sub-builders `build_bm25, build_citation_graph, build_lookup_tables, build_citation_signals, build_reference_graph_v2, build_gold_cocitation_prior, build_gold_cocitation_prior_trainval`. Inlined helpers: `indexing/build_bm25_index.py` (36), `indexing/build_citation_graph.py` (38), `indexing/build_lookup_tables.py` (40), `stage0d_build_citation_signals.py` (42), `stage0e_build_reference_graph_v2.py` (44), `stage0f_build_gold_cocitation_prior.py` (46). Runner cell 48: `run_stage0_all()` (commented selector available).

Build steps 0A-0F per docstring:
- 0A: BM25 combined index over 171K laws + ~2M court considerations (~3-5 h, ~16 GB RAM).
- 0B: Bidirectional citation graph Art. <-> Case + co-citation (scans 2.47M rows).
- 0C: Lookup tables (`citation_set`, `article_to_abs`, `abs_to_article`, `sr_to_abbrev`, `abbrev_to_sr`).
- 0D: Deterministic statute/case signal cards -> `citation_signal_lookup.pkl`.
- 0E: Typed reference graph v2 (article_to_articles, article_to_cases, case_to_articles, case_to_cases) + reverse views.
- 0F / 0F*: Gold co-citation priors (train-only and optional train+val).

### Stage 1 - Query Analysis (GPU, ~7 min for 40 queries)

Runner cell 54: `stage1_outputs = run_stage1(split=SPLIT, backend=BACKEND)` invokes `stage1_query_analysis.run_batch(split, backend, model, resume)`.

Prompt `agent/prompts/query_analyzer_prompt.txt` (cell 50) instructs Qwen3-4B to:
1. Classify legal domains (ZGB/OR, StGB/StPO, BV/VwVG, ATSG/IVG, IPRG, SchKG, ZPO/StPO/BGG).
2. Identify legal issues + sub-issues + procedural questions.
3. Reason about applicable articles, Absatz, related articles, BGE Leitentscheide.
4. Add procedural citations (BGG admissibility, costs, time limits).
5. Add cross-references (constitutional / general-clause).
6. Extract explicit citations from query text verbatim.
7. Generate 3-5 German BM25 search queries per issue.

JSON output: `law_areas`, `explicit_citations`, `legal_issues[{issue, ...}]`, search queries.

### Stage 2 - Multi-Agent Sparse Retrieval (GPU+CPU, ~20 min)

Runner cell 60: `stage2_outputs = run_stage2(split=SPLIT, backend=BACKEND, max_iterations=STAGE2_MAX_ITERATIONS)`. Default `max_iterations=4`. Inlined `stage2_mas_retrieval.py` and `agent/prompts/mas_prompts.py` (cell 56).

Agents share one Qwen3-4B instance; planner picks the next agent per iteration:
- PLANNER: chooses next action ({REWRITE, SUPPLEMENT, DECOMPOSE, SUPPORTIVE, CROSSREF, TERMINATE}).
- REWRITE: EN -> precise DE Fachbegriffe ("pre-trial detention" -> "Untersuchungshaft", "collusion risk" -> "Kollusionsgefahr").
- SUPPLEMENT: implicit thresholds/standing/procedural conditions.
- DECOMPOSE: split multi-issue queries into sub-queries.
- SUPPORTIVE: procedural / cost / admissibility provisions.
- CROSSREF: constitutional + cross-referenced articles.

Each agent emits DE queries piped through `SparseRetriever` (cell 29): tokenise -> PMI token->law boosting (`_enhance_tokens`, `law_boost=10, top_n_laws=5`) -> `BM25.get_scores` -> np.argpartition top-K -> max-score fusion across queries. Search modes: `search_statutory`, `search_caselaw`, `search_combined`.

Target: ~200+ candidates per query.

### Stage 3 - Citation Graph Expansion (CPU, ~30 s)

Runner cell 65: `stage3_outputs = run_stage3(split=SPLIT, gold_prior_mode=STAGE3_GOLD_PRIOR_MODE)` (default `"off"`; `"train"` or `"trainval"` enable gold co-citation prior).

`GraphRetriever` (cell 27):
- `articles_to_cases(article_citations, top_k_per_article=20)`: returns citing cases ranked by RRF-style score `sum(1.0 / (60 + rank))`.
- `cases_to_articles(case_citations)`: returns the union of articles cited by those cases.
- `bidirectional_expand(seed_articles, seed_cases, depth=1)`: alternates the two directions.

### Stage 4 - LLM Reranking + Direct Generation (GPU, ~10 min)

Runner cell 72: `stage4_outputs = run_stage4(split=SPLIT, backend=BACKEND, skip_direct_gen=STAGE4_SKIP_DIRECT_GEN)`. Inlined `stage4_llm_reranker.py` (cell 70).

- 4A reranker prompt (cell 67): rate each candidate 0/1/2/3 (3=directly applicable, 2=closely related, 1=tangential, 0=not relevant). Output JSON `{"selected": {"<citation>": <rating>}}`. Includes provisions rated >= 2 only. Rule of thumb: from a batch of 30, expect 3-8 tier-3 and 5-15 tier-2.
- 4B direct-generation prompt (cell 68): generate 10-20 ADDITIONAL citations a Federal Tribunal would cite, biased to procedural (BGG 82/83/90/93/95/97/100/105/106/107/113, BGG 76/68/65-68 etc.), constitutional (BV 9/10/29/31/32), and substantive BGE leading cases. Output `{"additional": [...]}`.

### Stage 4b - Cross-Encoder Scoring (GPU, ~25 min)

Runner cell 76: `stage4b_outputs = run_stage4b(split=SPLIT)` or `run_stage4b(split=SPLIT, top_n=STAGE4B_TOP_N)` if not None. Loads `stage4b_cross_encoder.py` (cell 74); applies Qwen3-Reranker-4B P(yes) probability per the confidence-scoring docstring.

### Stage 5 - Verify + Score + Threshold (CPU, ~30 s)

Runner cell 81:
```
stage5_outputs = run_stage5(split=SPLIT, threshold=STAGE5_THRESHOLD)
submission_path = get_submission_path(SPLIT)
```

5A Verify (`agent/verifier.py`, cell 21): `Verifier.verify_and_normalize(citations, expand_to_abs=True)` normalizes whitespace/abbreviation casing, validates against `citation_lookup.pkl`, expands article-level to Abs-level when needed, strips `lit./Ziff.` to find parent Abs or base Art, drops hallucinated case citations unless they pass a strict regex (lenient fallback for valid Swiss court formats: `BGE \d{2,3} [IVX]+ \d+(\sE.\s[\d.]+)?(\sS.\s\d+)?` or `\d+[A-Z]_\d+/\d{4}` patterns).

Explicit-citation regex extraction (`retrieval/explicit_citations.py`, cell 25) provides FREE RECALL: parses single (`Art. X [Abs. Y [lit. z]] LAW`), multi (`Art. 38 and 39 CO` -> 2 citations), BGE (with/without Erwagung), and numbered case (`1B_210/2023 E. 4.1`) patterns.

5B Score (`scoring/confidence.py`, cell 31): additive weights + continuous BM25 `bm25_norm * 0.10` + cross-encoder `P(yes) * 0.18` (replaces binary reranker signals when available) + graduated multi-signal bonus.

5C Threshold sweep: `optimize_threshold` walks `np.arange(0.05, 0.96, 0.01)` and picks the threshold maximizing macro-F1 on val; test reuses the val threshold (expected optimal range 0.15-0.30).

### Pipeline orchestrator (cell 86, commented)

```
run_full_pipeline(split=SPLIT, start_stage=START_STAGE, backend=BACKEND,
                  stage3_gold_prior_mode=STAGE3_GOLD_PRIOR_MODE)
```

`pipeline.run_pipeline` checks Stage 0 outputs (`bm25_v2_index.pkl`, `citation_graph.pkl`, `citation_lookup.pkl`) and chains stages 1->5 (and 4b per docstring). Total wall time per docstring: ~65-105 min for 40 queries.

### Submission validation (cell 91)

```
submission_path = get_submission_path(SPLIT)
if submission_path.exists():
    submission_valid = validate_generated_submission(SPLIT)
```

`submit.validate_submission` requires columns `{query_id, predicted_citations}`, checks all `test.csv` ids are present, reports avg/max citation counts and rows with 0 citations.

## Results

The notebook is purely a packaging/runner notebook: **all 92 cells were stored unexecuted** (`outputs=0` for every code cell when inspected via PowerShell JSON parse). No metrics, no errors, no submission counts, no F1 scores are captured in the file itself.

Documented (non-empirical) expectations from the inlined docstrings/README cell 2:

| Component | Estimated Recall |
|---|---|
| BM25 baseline (R@50) | 0.73 |
| + MAS multi-agent reformulation | +0.08-0.12 |
| + Citation graph expansion | +0.05-0.10 |
| + Explicit citation extraction | +0.02-0.05 |
| + LLM direct generation | +0.03-0.05 |
| Combined recall estimate | 0.85-0.92 |
| After LLM reranking (precision) | 0.65-0.75 |
| Estimated F1 | 0.70-0.80 |

Other claims in the inlined text (not numbers run inside this notebook):
- BM25 R@50 = 0.73 vs Dense R@100 = 0.24 (justification for dropping dense).
- Cross-encoder weight 0.18 "validated at 0.18 on all 10 val queries: 0.5400 -> 0.6520 macro-F1" (per `scoring/confidence.py` docstring, cell 31).
- 40% of val citations are case law; train data is 98.8% statutory (`graph_retriever.py` docstring, cell 27).

## Summary

This notebook freezes the full BM25-first MAS pipeline (Stages 0 through 5 plus 4b) into a single notebook-runnable form, with every project module inlined as a reference cell and every path declared up front under `PROJECT_ROOT = /mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline`. The architecture pairs `Qwen3-4B` Q4/NF4 (~2.8 GB VRAM) for query analysis, MAS reformulation, and reranking with a chunked BM25 index over 171K laws + ~2M court rows, a bidirectional Art<->Case citation graph, regex explicit-citation free-recall, and a signal-counting + cross-encoder confidence score with a 0.05-0.95 macro-F1 threshold sweep. No execution traces are captured in this file - the cells are configured (`SPLIT="test"`, `BACKEND="local"`, `STAGE2_MAX_ITERATIONS=4`, `STAGE3_GOLD_PRIOR_MODE="off"`) but never ran here. The only numbers present are docstring claims (BM25 R@50=0.73, Dense R@100=0.24, expected F1 0.70-0.80, and a cross-encoder ablation 0.5400 -> 0.6520 macro-F1 at CE weight 0.18). The artifact's value is reproducibility plumbing - explicit paths, helper wrappers (`run_stage0..5`, `get_submission_path`, `validate_generated_submission`), and end-to-end source inlining - not new measurements.
