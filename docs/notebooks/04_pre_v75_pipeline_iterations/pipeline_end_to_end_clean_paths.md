# pipeline_end_to_end_clean_paths.ipynb

**Path:** `notebooks/04_pre_v75_pipeline_iterations/pipeline_end_to_end_clean_paths.ipynb`

Notebook-clean edition of the BM25-first, LegalMALR-adapted Swiss law citation retrieval pipeline. 89 cells total (52 markdown, 37 code). All code cells have `execution_count = null` and zero outputs — the notebook is a packaged source bundle, not an executed run. Cells 16-86 paste the project module source verbatim under `__file__ = ...` headers; cells 4-13, 45, 51, 57, 62, 69, 73, 78, 83, 88 are the notebook-native config and stage-runner cells.

## Configuration

### Runtime settings (cell 7)

| Variable | Value |
|---|---|
| `SPLIT` | `"test"` (options: `"train"`, `"val"`, `"test"`) |
| `BACKEND` | `"local"` (options: `"local"`, `"anthropic"`, `"openai"`) |
| `START_STAGE` | `1` |
| `STAGE3_GOLD_PRIOR_MODE` | `"off"` (options: `"off"`, `"train"`, `"trainval"`) |
| `STAGE2_MAX_ITERATIONS` | `4` |
| `STAGE4_SKIP_DIRECT_GEN` | `False` |
| `STAGE4B_TOP_N` | `None` (stage default) |
| `STAGE5_THRESHOLD` | `None` (stage default) |

### Paths (cell 4)

| Variable | Value |
|---|---|
| `PROJECT_ROOT` | `Path("/mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline").resolve()` |
| `RAW_DATA_DIR` | `os.environ.get("SWISS_DATA_DIR", PROJECT_ROOT / "data")` |
| `INDEX_DIR` | `PROJECT_ROOT / "index"` |
| `CHECKPOINTS_DIR` | `PROJECT_ROOT / "checkpoints"` |
| `SUBMISSIONS_DIR` | `PROJECT_ROOT / "submissions"` |
| `MODELS_DIR` | `PROJECT_ROOT / "models"` |

Env exports: `SWISS_PIPELINE_ROOT`, `SWISS_DATA_DIR`. The notebook does `os.chdir(PROJECT_ROOT)` and prepends `PROJECT_ROOT` to `sys.path`.

### `configs/default.json` (loaded by cell 8, dumped in cell 9)

```json
{
    "_comment": "Revised pipeline config — BM25-first, LegalMALR-adapted MAS architecture",
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

- **Qwen3-4B** with `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`. Loaded once as singleton (`agent/llm_backend.py`, cell 16). VRAM ~2.7-2.8 GB. Used by Stage 1, Stage 2 (all MAS agents), Stage 4 (reranker + direct-gen). Loaded from local `QWEN3_4B_PATH` if present, else HF id `QWEN3_4B_HF_ID`.
- **Qwen3-Reranker-4B** (NF4) cross-encoder in Stage 4b, ~2.8 GB VRAM.
- Cloud fallbacks: `anthropic` client (`claude-sonnet-4-6`), `openai` client (`gpt-4o`).

### Generation defaults (`generate()` in cell 16)

`max_new_tokens=2048`, `temperature=0.6`, `top_k=20`, `top_p=0.95`, `enable_thinking=False`. On CUDA OOM: clear cache and retry once with halved `max_new_tokens`.

### BM25 (`retrieval/bm25_artifact.py`, `retrieval/sparse_retriever.py`, cells 20, 26)

- `ChunkedBM25Okapi` with float32 doc-length array.
- Chunked artifact format `chunked_bm25_v1`, chunk size `SWISS_BM25_CHUNK_SIZE` env (default 10000, floor 1000).
- Tokenizer: lowercase, split on `[^\w\d]+`, drop tokens of length < 2 unless numeric.
- Combined index over law + court: `n_law` + `n_court` aligned to `citation_canon` list.
- PMI token→law boosting via `TOKEN_LAW_FREQ` map.

### Citation graph build (`indexing/build_citation_graph.py`, cell 35)

- `CHUNKSIZE = 200_000` rows per pandas read.
- `MAX_CITES_PER_ROW = 30` cap to avoid O(n²) on outliers.
- `MIN_CO_OCCURRENCE = 3` prune.
- Extracts German, French (`art. … al. … let.`), Italian (`art. … cpv. … lett.`) statutory refs; `BGE NNN [IVX]+ NNN E. x.x` and `1B_210/2023 E. 4.1` case refs.

### Confidence scoring (`scoring/confidence.py`, cell 28)

```python
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
BM25_CONTINUOUS_WEIGHT  = 0.10
CROSS_ENCODER_WEIGHT    = 0.18
MULTI_SIGNAL_BONUS      = 0.10   # added when 3+ independent signals agree
```

### Hardware (requirements_text in cell 6)

Python 3.12, CUDA 12.1, RTX 4050 6GB target. Install: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`; then `pip install -r requirements.txt`.

### Library pins (cell 6)

`rank-bm25==0.2.2`, `transformers>=4.45.0`, `bitsandbytes>=0.44.0`, `accelerate>=1.0.0`, `huggingface_hub>=0.25.0`, `safetensors>=0.4.0`, `anthropic>=0.40.0`, `openai>=1.50.0`, `pandas>=2.0.0`, `numpy>=1.26.0`, `pyarrow>=14.0.0`, `tqdm>=4.66.0`, `deep-translator>=1.11.0`.

### Verifier abbreviation casing map (cell 18)

42-entry `_ABBREV_CASING` table mapping UPPERCASE corpus abbreviations to mixed-case gold form (e.g. `STGB→StGB`, `STPO→StPO`, `SCHKG→SchKG`, `JSTG→JStG`, `BETMG→BetmG`, `GSCHG→GSchG`, `FINFRAG→FinfraG`).

### Multilingual abbreviation aliases (cell 22, `_ALT_ABBREV_TO_CANON`)

`CC→ZGB`, `CO→OR`, `LP→SchKG`, `CPC→ZPO`, `LDIP→IPRG`, `LFors→GestG`, `CPP→StPO`, `CP→StGB`, `LTF→BGG`, `PA→VwVG`.

## Data

Paths declared via `data.data_paths` (referenced, not redefined in notebook):

| Symbol | Description |
|---|---|
| `DATA` | base data dir (defaults to `PROJECT_ROOT/data`, override via `SWISS_DATA_DIR`) |
| `LAW_CORPUS_PARQUET` | `data/corpus.parquet` — ~171K law articles, columns include `citation_canon`, `title` |
| `COURT_CSV` | `data/court_considerations.csv` — 2.47M court consideration rows, ~1.98M unique citations |
| `KB_JSONL` | `data/laws_knowledge_base.jsonl` — KB metadata |
| `LAWS_DE_CSV` | `data/laws_de.csv` |
| `TRAIN_CSV` | `data/train.csv` — gold training labels |
| `VAL_CSV` | `data/val.csv` — gold validation labels (column `gold_citations`, ";"-separated) |
| `TEST_CSV` | `data/test.csv` — test queries |
| `QUERY_TRANSLATIONS_JSON` | `data/query_translations_trainval.json` — cached EN→DE translations |
| `TOKEN_LAW_FREQ` | `data/token_law_freq.json` — PMI token→law freq map |
| `STATUTE_SIGNALS_JSONL` | `data/statute_signals.jsonl` |
| `CASE_SIGNALS_JSONL` | `data/case_signals.jsonl` |
| `CITATION_SIGNAL_SCHEMA_JSON` | `data/citation_signal_schema_v2.json` |

Stage 0 outputs (under `INDEX_DIR`):

| Path | Description |
|---|---|
| `index/bm25_v2_index.pkl` | BM25Okapi combined index (law + court), ~2-4 GB |
| `index/bm25_v2_ids.pkl` | aligned `citation_canon`, `n_law`, `n_court` meta |
| `index/citation_graph.pkl` | `article_to_cases`, `case_to_articles`, `case_to_cases`, `article_corpus_freq`, `cocitation`, `citation_doc_freq` |
| `index/citation_lookup.pkl` | normalize/existence/Abs.-expansion + SR↔abbrev tables |
| `index/citation_signal_lookup.pkl` | statute + case signal lookup |
| `index/reference_graph_v2.pkl` | typed reference graph v2 |
| `index/gold_cocitation_prior_train.pkl` | train-only pair prior |
| `index/gold_cocitation_prior_trainval.pkl` | optional competition-mode pair prior |

Per-split outputs:

| Path | Stage |
|---|---|
| `checkpoints/stage1_{split}.json` | 1 |
| `checkpoints/stage2_{split}.json` | 2 |
| `checkpoints/stage3_{split}.json` | 3 |
| `checkpoints/stage4_{split}.json` | 4 |
| `checkpoints/reranker_scores_{split}.json` | 4b |
| `checkpoints/bm25_scores_{split}.json` | 5 first-run |
| `checkpoints/stage5_{split}.json` | 5 |
| `submissions/submission_{split}.csv` | 5 |

## Pipeline

### Stage 0 — Offline index build (cells 30-45)

CPU only, one-time, ~4-7 hours. Runner: `run_stage0_all()` calls `build_bm25`, `build_citation_graph`, `build_lookup_tables`, `build_citation_signals`, `build_reference_graph_v2`, `build_gold_cocitation_prior` in `stage0_build_index`.

- **0A BM25 build** (`indexing/build_bm25_index.py`, cell 33): tokenizes ~171K laws + ~1.98M deduped court rows (group by citation, concatenate texts up to `COURT_TEXT_LIMIT`), evaluates recall on train+val, writes `bm25_v2_index.pkl` + `bm25_v2_ids.pkl` + `query_translations_trainval.json` + `token_law_freq.json`. ~16 GB RAM peak.
- **0B citation graph** (`indexing/build_citation_graph.py`, cell 35): scans 2.47M rows once, extracts DE/FR/IT statutory refs + BGE/numbered-case refs, builds bidirectional + co-citation graphs, ingests `train.csv` gold as co-citation signal.
- **0C lookup tables** (`indexing/build_lookup_tables.py`, cell 37): from `corpus.parquet`, builds `citation_set`, `article_to_abs`, `abs_to_article`, `sr_to_abbrev`, `abbrev_to_sr`.
- **0D citation signals v2** (`stage0d_build_citation_signals.py`, cell 39): emits nested `statute_signals.jsonl` + `case_signals.jsonl` + `citation_signal_lookup.pkl`. Constants: `CASE_CHUNKSIZE=200_000`, `TEXT_SNIPPET_CHARS=420`, `GRAPH_NEIGHBOR_LIMIT=8`, `KEYWORD_LIMIT=8`, `ENUM_ITEM_LIMIT=8`.
- **0E reference graph v2** (`stage0e_build_reference_graph_v2.py`, cell 41): typed edge sets `article_to_articles`, `article_to_cases`, `case_to_articles`, `case_to_cases` + reverse views. `LAWS_CHUNKSIZE=50_000`.
- **0F gold co-citation prior** (`stage0f_build_gold_cocitation_prior.py`, cell 43): builds train-only (and optional train+val) pair priors with PMI/NPMI; schema `gold-cocitation-prior-v1`. Pair types: `law-law`, `court-court`, `law-court`.

### Stage 1 — Query analysis (cells 46-51)

GPU, ~7 min for 40 queries. Runner: `run_stage1(split=SPLIT, backend=BACKEND)` → `stage1_query_analysis.run_batch`. Six-step pipeline per query:

1. **Explicit extraction** — regex from query text via `retrieval/explicit_citations.py` patterns (`_SINGLE_STAT_PATTERN`, `_MULTI_STAT_PATTERN`).
2. **BM25 probe** — raw EN query against BM25, top-10 to identify candidate law areas.
3. **Domain scaffolding** — extract abbreviations from probe, look up EN names + DE keywords in KB.
4. **LLM analysis** — Qwen3-4B with `query_analyzer_prompt.txt` (cell 47), `enable_thinking=True`. Outputs JSON: `law_areas`, `explicit_citations`, `legal_issues`, `procedural_articles`, `candidate_articles`, `candidate_cases`, `search_queries_de`, `search_queries_en`.
5. **HyDE** — LLM drafts a hypothetical German statutory paragraph.
6. **Cross-lingual term expansion** — KB-grounded EN→DE term map.

Output `checkpoints/stage1_{split}.json` per-query: `explicit_citations`, `law_areas`, `legal_issues`, `candidate_articles`, `candidate_cases`, `search_queries_de`, `search_queries_en`, `hyde_documents_de`, `bm25_probe_laws`, `domain_context`, `procedural_articles`, `cross_lingual_terms`.

### Stage 2 — Multi-Agent Sparse Retrieval (cells 52-57)

GPU+CPU, ~20-40 s/query, ~20 min/val. Runner: `run_stage2(split, backend, max_iterations=STAGE2_MAX_ITERATIONS)`. LegalMALR-adapted MAS over BM25. Agents share one Qwen3-4B model (cell 53 `mas_prompts.py`):

- **PLANNER** — picks next agent or `TERMINATE`. JSON: `{"reasoning", "action"}`.
- **REWRITE** — EN colloquial → precise DE Fachbegriffe.
- **SUPPLEMENT** — makes implicit conditions explicit.
- **DECOMPOSE** — splits multi-issue queries.
- **SUPPORTIVE** — procedural/auxiliary provisions.
- **CROSSREF** — constitutional + cross-referenced articles.

Loop runs 2-4 iterations, terminating when no new candidates appear. Per agent: generate DE queries → BM25 → merge into pool. Target ~200+ candidates/query.

Output `checkpoints/stage2_{split}.json`: `{query_id: {candidate_pool, retrieval_log}}`.

### Stage 3 — Citation graph expansion (cells 58-62)

CPU, <1 s/query. Runner: `run_stage3(split, gold_prior_mode=STAGE3_GOLD_PRIOR_MODE)`. Uses `GraphRetriever` (cell 24): `articles_to_cases` adds top-5 citing cases per article with RRF score `1/(60+rank)`; `cases_to_articles` adds cited statutes; `bidirectional_expand` up to `depth=1`. Optionally folds in gold co-citation prior (`train` or `trainval`).

Output `checkpoints/stage3_{split}.json`: `{query_id: {expanded_pool, sources, graph_stats}}`.

### Stage 4 — LLM reranker + direct generation (cells 63-69)

GPU. Runner: `run_stage4(split, backend, skip_direct_gen=STAGE4_SKIP_DIRECT_GEN)`. Pre-filters to ~1200 candidates, batch_size=50 with text → ~24 batches/query → ~35-60 min for 10 val queries.

- **4A reranker** — `reranker_prompt.txt` (cell 64). Rates each candidate 0/1/2/3. Output `{"selected": {cite: rating, ...}}` (only ratings ≥ 2). No chain-of-thought in output (LegalMALR §3.4); `enable_thinking=False`.
- **4B direct generation** — `direct_gen_prompt.txt` (cell 65). LLM generates 10-20 additional citations grouped into PROCEDURAL / CONSTITUTIONAL / SUBSTANTIVE buckets. Output `{"additional": [...]}`.

Output `checkpoints/stage4_{split}.json`: `{query_id: {reranked, direct_gen, sources}}`.

### Stage 4b — Cross-encoder scoring (cells 70-73)

GPU, ~60 min for 10 val queries (500 candidates each, batch_size=8). Runner: `run_stage4b(split, top_n=STAGE4B_TOP_N)`. Scores top-500 candidates (by base composite) per query using Qwen3-Reranker-4B as cross-encoder. Output `checkpoints/reranker_scores_{split}.json`: `{qid: {citation: float P(yes), ...}}`.

### Stage 5 — Verify + score + threshold (cells 74-78)

CPU, ~30 s. Runner: `run_stage5(split, threshold=STAGE5_THRESHOLD)`.

- **5A verify** — `agent/verifier.py` (cell 18) `Verifier.verify_and_normalize`: normalize whitespace/abbrev/casing, `is_law_citation` / `is_case_citation` dispatch, expand Art. → Abs. via `lookup.expand_to_abs`, strip `lit./Ziff.` fallback to Abs.- then base-Art.-level. Cases kept if in `caselaw_ids` else format-regex fallback for `BGE NN [IVX]+ NN [E. x.x] [S. n]` and `\d+[A-Z]_\d+/\d{4} [DD.MM.YYYY] [E. x.x]`. Returns `(verified, dropped)` deduplicated.
- **5B confidence** — `scoring/confidence.score_citation`: additive weighted signals + continuous BM25 (`BM25_CONTINUOUS_WEIGHT=0.10`) + cross-encoder (`CROSS_ENCODER_WEIGHT=0.18`, replaces binary `llm_reranker/tier3` when CE score available, falls back otherwise) + multi-signal bonus 0.10 for 3+ independent signals from `_INDEPENDENT_SIGNALS = {explicit_from_query, bm25_top10, bm25_initial, citation_graph, llm_reranker, llm_reranker_tier3, llm_direct_gen, llm_stage1}`.
- **5C threshold** (val only) — sweep 0.05-0.95 to maximize macro-F1.

Output `checkpoints/stage5_{split}.json` + `submissions/submission_{split}.csv` (columns `query_id`, `predicted_citations`).

### Optional orchestrator (cells 79-83)

`run_full_pipeline(split, start_stage, backend, stage3_gold_prior_mode)` → `pipeline.run_pipeline`. Cell 83 has this commented out.

### Submission validation (cells 84-88)

`submit.validate_submission`: checks required cols `{query_id, predicted_citations}` and that test IDs match `TEST_CSV`. Cell 88 prints submission path; if file exists, calls `validate_generated_submission(SPLIT)`.

## Results

All 13 code runner cells (`execution_count = null`, zero outputs each):

- Cell 4 (path config print) — not executed.
- Cell 7 (runtime settings print) — not executed.
- Cell 8 (load `default.json`) — not executed.
- Cell 13 (checkpoint path prints) — not executed.
- Cell 45 (`run_stage0_all()`) — not executed.
- Cell 51 (`run_stage1`) — not executed.
- Cell 57 (`run_stage2`) — not executed.
- Cell 62 (`run_stage3`) — not executed.
- Cell 69 (`run_stage4`) — not executed.
- Cell 73 (`run_stage4b`) — not executed.
- Cell 78 (`run_stage5` + print submission path) — not executed.
- Cell 83 (commented-out `run_full_pipeline`) — not executed.
- Cell 88 (submission validation) — not executed.

No metrics, no execution logs, no errors recorded in the notebook. Quantitative claims are static documentation embedded in markdown cells:

- **Expected performance table (cell 2)**: BM25 baseline R@50=0.73; +MAS +0.08-0.12; +graph +0.05-0.10; +explicit +0.02-0.05; +LLM-gen +0.03-0.05; **combined recall estimate 0.85-0.92**; after LLM rerank precision 0.65-0.75; **estimated F1 0.70-0.80**.
- **Empirical anchor (cell 2)**: BM25 R@50=0.73 vs Dense R@100=0.24 → dense retrieval dropped.
- **Strategy C cross-encoder gain (cell 28 docstring)**: 0.5400 → 0.6520 macro-F1 on val (10 queries, weight 0.18).
- **Class distribution (cell 60 docstring)**: val citations 59.4% statutory / 40.6% case law; train 98.8% statutory.
- **Court corpus dedup (cell 33 docstring)**: 2.47M rows → 1.98M unique citations.

## Summary

The notebook packages the full 5-stage (Stage 0 + Stages 1-5 + 4b) Swiss law citation retrieval pipeline as a single end-to-end notebook with explicit `PROJECT_ROOT = /mnt/data/swiss_law_pipeline_unzipped/swiss_law_pipeline` paths and notebook-native runner wrappers around the underlying `stage*.py` modules. Architecture is BM25-first (rank-bm25 over ~2.15M docs) with Qwen3-4B (NF4, ~2.8 GB VRAM) as the single LLM serving Stage 1 query analysis, Stage 2 LegalMALR-style multi-agent MAS reformulation (Planner/Rewrite/Supplement/Decompose/Supportive/CrossRef), Stage 4A reranking, and Stage 4B direct generation, plus a Qwen3-Reranker-4B cross-encoder in Stage 4b. The notebook itself was never executed (all code cells have empty outputs and null execution counts), so there are no actual run metrics; the only quantitative figures come from markdown docstrings and recall/F1 estimates. Notable hardcoded recipe: confidence scoring with 10 binary signals + continuous BM25 (0.10) + cross-encoder (0.18, validated 0.54→0.652 macro-F1) + 3-signal bonus (0.10), and threshold sweep 0.05-0.95. Lesson encoded in the notebook itself: dense retrieval is dropped entirely (BM25 R@50=0.73 vs dense R@100=0.24), and the citation graph is the case-law recall multiplier because train is 98.8% statutory while val is 40.6% case law.
