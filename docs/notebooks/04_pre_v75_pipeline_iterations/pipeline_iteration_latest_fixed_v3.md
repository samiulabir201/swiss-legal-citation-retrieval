# pipeline_iteration_latest_fixed_v3.ipynb

**Path:** notebooks/04_pre_v75_pipeline_iterations/pipeline_iteration_latest_fixed_v3.ipynb

End-to-end Swiss-law citation retrieval pipeline (Stages 0–6), "Notebook-Clean Edition", configured to run the `test` split with adaptive-K Stage 6 final submission. 129 cells total (code + markdown + diagnostics + post-mortem).

## Configuration

### Runtime environment
- Platform: Google Colab with Drive mount at `/content/drive`. PROJECT_ROOT = `/content/drive/MyDrive/swiss_law`.
- Working dir auto-chdir'd to PROJECT_ROOT; PROJECT_ROOT inserted into `sys.path`.
- HF cache forced to `${PROJECT_ROOT}/model` BEFORE importing transformers (`HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`).
- pip-installs (top of notebook): `deep-translator`, `rank-bm25`, `scikit-learn`.
- Imports: `numpy, pandas, torch, deep_translator.GoogleTranslator, rank_bm25.BM25Okapi, scipy.sparse, tqdm, transformers.AutoTokenizer, transformers.AutoModelForCausalLM`.

### Top-level switches (cell 21)
```
SPLIT = "test"
BACKEND = "local"
STAGE3_GOLD_PRIOR_MODE = "off"
STAGE2_MAX_ITERATIONS  = 4
STAGE4_SKIP_DIRECT_GEN = True       # direct-gen was 60% hallucinated on val
STAGE4B_TOP_N          = None       # Stage 4b disabled
STAGE5_THRESHOLD       = None       # val: tuned; test: from configs/threshold.json
STAGE4_RANK_POOL_SIZE  = 150
STAGE3_BUILD_STATUTE_TO_BGE = True
STAGE3_STATUTE_TO_BGE_TOPK  = 5
STAGE3_USE_NUMBERED_INDEXES     = True
STAGE3_STATUTE_TO_NUMBERED_TOPK = 3
STAGE3_NUMBERED_TO_BGE_TOPK     = 3
STAGE3_NUMBERED_COCITATION_TOPK = 1
STAGE6_NUMBERED_BM25_QUANTILE         = 0.90
STAGE6_NUMBERED_REQUIRE_GRAPH_SOURCE  = False  # v3 reverted
STAGE6_ABS_COLLAPSE             = True
STAGE6_ABS_MAX_VARIANTS         = 2
```

### Default pipeline config (cell 22)
```
bm25_top_k: 100
graph_top_k: 50
mas_max_iterations: 4
mas_bm25_top_k: 30
reranker_backend: "local"
reranker_batch_size: 20
confidence_weights:
  explicit_from_query: 0.40
  bm25_top10:          0.25
  citation_graph:      0.15
  llm_reranker:        0.15
  llm_direct_gen:      0.05
threshold: 0.15
```

### Stage 6 knobs (cell 124)
```
STAGE6_MIN_K = 10
STAGE6_MAX_K = 45
STAGE6_USE_ADAPTIVE_K   = True
STAGE6_PROCEDURAL_FLOOR = True
STAGE6_SCORE_SOURCE = "bm25_plus_llm"   # alt: "stage5"
STAGE6_LLM_BOOST    = 0.5               # weight on (llm_score/10) over BM25 norm
STAGE6_K_STRATEGY   = "score_gap"       # alt: "regressor", "fixed"
STAGE6_FIXED_K      = 15
STAGE6_FRAC_OF_MAX  = 0.55              # overridable via configs/stage6_knobs.json
STAGE6_ELBOW_DROP   = 0.12
```
Procedural floor categories (regex triggers): `appeal_bgg`, `criminal`, `costs`, `constitutional` → force-include articles such as `Art. 42 BGG`, `Art. 66 Abs. 1 BGG`, `Art. 100 Abs. 1 BGG`, `Art. 81 BGG`, `Art. 422 StPO`, `Art. 29 Abs. 2 BV`, etc.

### Models
- `QWEN3_32B_HF_ID = "Qwen/Qwen3-32B"` — bf16, ~64GB VRAM (full). Used for Stage 1 query analysis and Stage 4 LLM reranking.
- `QWEN3_RERANKER_ID = "Qwen/Qwen3-Reranker-4B"` — referenced for Stage 4b (Stage 4b disabled in this run).
- Stage 2 MAS internally uses Qwen3-4B per design note (per-stage code).

### Hardware (observed in outputs)
- BM25 sparse matrix: 2,156,832 docs × 822,251 terms (212,228,366 non-zero), built in ~35s from 216 chunked pickle parts.
- Qwen3-32B load notice: `Qwen3-32B loaded. VRAM: ~64GB (bf16)`.
- Stage 2 reports `[OOM]` on iter-4 batched calls, falling back to sequential per query.

## Data

All paths are read from the Drive-mounted PROJECT_ROOT (`/content/drive/MyDrive/swiss_law`).

### Competition CSVs (`/content/drive/MyDrive/swiss_law/data/`)
- `train.csv` (1,139 rows; cites/q min=1, max=44, mean=4.1, median=2.0; Law=4602 (99%), BGE=57 (1%))
- `val.csv` (10 rows; cites/q min=10, max=47, mean=25.1, median=22.0; Law=149 (59%), BGE=69 (27%), Numbered=33 (13%))
- `test.csv` (40 rows)
- `court_considerations.csv` (2.26 GB — source for text-mined statute→BGE edges)
- `laws_de.csv` (175,933 rows; FYI only; SR-style codes)
- `sample_submission.csv`

### Corpus & knowledge base (`/content/drive/MyDrive/swiss_law/index/` ≡ LAW_DB)
- `corpus.parquet`
- `laws_knowledge_base.jsonl` — 179,641 records. Top codes: `EFZ`:6500, `OR`:3667, `EBA`:2469, `ZGB`:2405, `StPO`:1306, `StGB`:1244, `TSV`:1094, `VTS`:1023, `SchKG`:919, `ZV`:874.
- Domain knowledge load: Law abbreviations 1,125; EN→DE term mappings 2,114; DE keywords 69,837; Article texts 171,654; valid citations 171,654.

### Pre-built index artifacts (`index/`)
- `bm25_v2_index.pkl` + `bm25_v2_ids.pkl` + chunked `bm25_v2_index_parts/bm25_v2_index_parts/` (216 chunks)
- `citation_graph.pkl` — 2,201,708 nodes; edge inventory: stat→BGE 0, stat→num 0, BGE→BGE 0, BGE→stat 0, other 3,897,617. (Diagnostic flags this as "NO statute→BGE edges" — fixed via text-mined index below.)
- `citation_lookup.pkl`, `citation_signal_lookup.pkl`, `reference_graph_v2.pkl`
- `statute_to_bge.pkl` (text-mined; 3,840 entries)
- `statute_to_numbered.pkl` (4,991 entries)
- `numbered_to_statutes.pkl` (80,569 entries)
- `numbered_to_bge.pkl` (67,513 entries)
- `numbered_cocitation.pkl` (81,709 entries)
- `gold_cocitation_prior_train.pkl`, `gold_cocitation_prior_trainval.pkl`
- `query_translations_trainval.json` (read-only source; copied to checkpoints/ for write)
- `token_law_freq.json`, `citation_signal_schema_v2.json`, `statute_signals.jsonl`, `case_signals.jsonl`

### Configs / outputs
- `configs/default.json` (inlined into notebook), `configs/threshold.json`, `configs/stage6_knobs.json` (optional override).
- Checkpoints: `checkpoints/stage{1..5}_{split}.json`, `checkpoints/bm25_scores_{split}.json`, `checkpoints/reranker_scores_{split}.json` (absent in this run).
- Submissions: `submissions/submission_{split}.csv`.

## Pipeline

### Stage 0 — Offline index build
Out-of-notebook prereq. Builds BM25 over 171,654 law articles + 1,985,178 court rows (= 2,156,832 docs), citation graph, lookup tables. Text-mined statute→BGE edges are produced on first Stage 3 run from `court_considerations.csv` (per `STAGE3_BUILD_STATUTE_TO_BGE`).

### Pre-run diagnostics (7 cells, read-only)
1. Data distribution per split (counts, cites/q stats, type breakdown).
2. KB integrity (record count, top codes, file presence of all index pickles).
3. Gold-format audit on val (Abs vs bare_Art vs lit specificity).
4. Procedural-boilerplate census on val (proceduralism shared across queries).
5. BGE reachability of val-gold stems in corpus.
6. Test-vs-val domain overlap (law codes present per side).
7. Citation-graph edge inventory.

### Stage 1 — Query Analysis (Qwen3-32B, bf16)
- Per-query 5-step procedure from `agent/prompts/query_analyzer_prompt.txt`: classify legal domain → identify issues → reason about applicable provisions → procedural cites → German search queries.
- Outputs per query: `explicit_citations` (regex), `law_areas`, `legal_issues`, `candidate_articles` (10), `candidate_cases` (10), `search_queries_en` (2–5), `search_queries_de` (5), `hyde_documents_de` (~1–3 docs, 65–145 tokens each).
- Runner: `run_stage1(split=SPLIT, backend=BACKEND)` → `checkpoints/stage1_{split}.json`. Processed 40 test queries in batches of 5.
- Diagnostics (6 cells): artifact summary, explicit-extraction fidelity, candidate hallucination, law_area routing accuracy, HyDE drift, free-recall ceiling.

### Stage 2 — Multi-Agent Sparse Retrieval (MAS)
- LegalMALR-adapted Planner→Agent loop, `STAGE2_MAX_ITERATIONS=4`, batched (size 3).
- Agents: `REWRITE`, `SUPPLEMENT`, `DECOMPOSE`, `SUPPORTIVE`, `CROSSREF` (all share Qwen3-4B; prompts in `agent/prompts/mas_prompts.py`).
- Each agent emits 4–10 German search queries → BM25 over the sparse 2.16M-doc index (`mas_bm25_top_k=30`).
- New helper cell loads 5 co-occurrence indexes (`statute_to_bge`, `statute_to_numbered`, `numbered_to_statutes`, `numbered_to_bge`, `numbered_cocitation`).
- Runner: `run_stage2(split, backend, max_iterations=4)` → `checkpoints/stage2_{split}.json`.
- Diagnostics (6 cells): counts+type breakdown, per-type recall (val), R@k curves (val), single- vs multi-query ablation, dead query variants, MAS action source breakdown.

### Stage 3 — Citation Graph Expansion (CPU)
- Expansion sources: graph-cases (statute→citing decisions), text-mined statute→BGE, statute→numbered, numbered→BGE, numbered-co-citation. Per-query top-K set by `STAGE3_*_TOPK`.
- Gold-prior mode `off` (production); the `train` / `trainval` modes are wired but disabled.
- Runner: `run_stage3(split, gold_prior_mode=STAGE3_GOLD_PRIOR_MODE)` → `checkpoints/stage3_{split}.json`.
- Diagnostics (5 cells): candidate counts + per-type recall, 1-hop/2-hop reachability, procedural-cite coverage, BGE safety-net, Abs-level coverage audit.

### Stage 4 — LLM Reranking + (skipped) Direct Generation
- Reranker prompt: continuous 0–10 score per candidate in a single JSON pass; no `<think>` blocks. Replaces legacy tier-{0,1,2,3} multi-pass.
- Pool: `STAGE4_RANK_POOL_SIZE = 150` per query (BM25 ∪ procedural ∪ explicit ∪ graph). Candidates outside pool pass through with Stage 3 `multi_signal` (recall preserved).
- Direct-gen DISABLED (`STAGE4_SKIP_DIRECT_GEN=True`) due to ~60% hallucination on val.
- 220,490 citation texts loaded from KB for ranker context.
- Runner: `run_stage4(split, backend, skip_direct_gen=True)` → `checkpoints/stage4_{split}.json`.
- Diagnostics (5 cells): counts, direct-gen hallucination, tier distribution, gold dropout S3→S4, pool recall + tier analysis.

### Stage 5 — Verify + Confidence + Threshold
- 5A: normalize citations via `citation_lookup`, drop hallucinations.
- 5B: confidence = weighted sum (`confidence_weights` above) + BM25 continuous scoring (precomputed per query, cached to `bm25_scores_{split}.json`).
- 5C (val): sweep thresholds 0.05–0.95 → max macro-F1; (test): load tuned threshold from `configs/threshold.json`.
- Runner: `run_stage5(split, threshold=STAGE5_THRESHOLD)` → `checkpoints/stage5_{split}.json` + `submissions/submission_{split}.csv`.
- Diagnostics (6 cells): global-threshold macro-F1 + per-type, per-type calibrated thresholds, query-level F1, citation-level error analysis, Abs-flooding, procedural-prior potential.

### Diagnostics rollup (D1–D6)
- Single dashboard cell + `run_all_diagnostics(split)` runs the fast (no-GPU) ones: D1 ranking ceiling, D2 adaptive-K oracle, D4 query type, D6 shadow-val. D3 (BGE-M3 dense reachability) and D5 (LLM ranker smoke) are GPU-gated.

### Stage 6 — Adaptive-K + Procedural Floor + Final Submission
- Re-scores stage5 candidates using `bm25_plus_llm` (BM25 normalized + 0.5 * llm_score/10).
- Applies Abs-collapse (`STAGE6_ABS_MAX_VARIANTS=2`) and Numbered-FP suppression (`bm25_quantile=0.90`).
- Per-query K selected by `score_gap` strategy (`frac_of_max=0.55`, `elbow_drop=0.12`), clamped to [10, 45].
- Force-includes procedural citations matching trigger regexes (appeal/criminal/costs/constitutional).
- Writes the final `submissions/submission_{split}.csv`.

### HyDE post-mortem (markdown cell 128)
Tracked gold-hit counts per query variant on val:
| Variant | Gold hits | Avg/run |
|---------|-----------|---------|
| `EN_raw` | 121 | 12.10 |
| `EN_0`   | 70  | 7.00 |
| `DE_2`   | 11  | 1.10 |
| `DE_0/DE_3/DE_4/HYDE_1` | ≤5 | ≤0.5 |

Decision: HyDE prose remains in Stage 1 outputs for debugging but is NOT routed into BM25.

## Results

### Pre-diagnostics
- Data distribution: `val` rows=10 (10–47 cites/q, mean 25.1; Law 59% / BGE 27% / Numbered 13%). `train` rows=1139 (mean 4.1 cites/q; Law 99% / BGE 1%). `test` rows=40. Warning printed: `val (25.1/q) has >2× train (4.1/q) — NEVER tune thresholds on train`.
- KB integrity: 179,641 records; all 9 index pickles confirmed present.
- Gold-format audit (val Art-citations): Abs 121 (81%), bare_Art 28 (19%). `81% sub-article — Abs-level matching is critical`.
- Procedural-boilerplate census (val): only `Art. 100 Abs. 1 BGG` appears in ≥2 queries (9 queries; never in query text).
- BGE reachability: 54/54 unique val-gold BGE stems present in corpus.
- Test-vs-val domain overlap: test law codes `{StPO:5, SVG:4, PrHG:4, OR:4, UVG:4, IPRG:3, ZGB:3, CO:2, SchKG:2, ATSG:1, CC:1, ZPO:1, MSchG:1}`; val law codes `{OR:3, StPO:2, ZGB:2}`. Codes in test but NOT in val: `CC, CO, MSchG, PrHG, SVG, UVG` — flagged as threshold-generalization risk.
- Citation graph: 2,201,708 nodes; stat→BGE = 0, stat→num = 0, BGE→BGE = 0, BGE→stat = 0, "other" = 3,897,617. Diagnostic asserts `NO statute→BGE edges — BGE recall via graph expansion is impossible. Fix: build text-mined statute→BGE index from court_considerations.csv` (subsequently built).

### Stage 1 (40 queries, batched 5/run)
- First batch LLM call: 1020.1s (204.0s/query); subsequent batches ~77–80s total (~15–16s/query) after warm-up.
- Artifact summary: `explicit_citations` mean=0.8 (min 0 / max 4); `candidate_articles`=10 (constant); `candidate_cases`=10 (constant); `search_queries_en` mean=3.8; `search_queries_de`=5 (constant); `hyde_documents_de` count mean=2.0, tokens mean=93.2; `law_areas` mean=2.1; `legal_issues` mean=2.5.
- Candidate-article validity: 398/400 valid (99.5%); 2 hallucinations: `test_004: Art. 62 Abs. 1 SchKG`, `test_004: Art. 63 Abs. 2 SchKG`.
- HyDE drift: 22/40 queries flagged `LIKELY DRIFT` (xl_matched ≤1, seed_matched=0). Most test queries fail seed-match.

### Stage 2 (40 queries, batches of 3)
- First batch wall: 688.1s for 3 queries; subsequent ~660s. Iter-4 batched call repeatedly `[OOM]`, falls back to sequential.
- Result: `candidates/q: min=563 max=1541 mean=1133.2`.
- Type mix at S2 (totals across 40 queries): `Numbered=23947 (53%), Law=14613 (32%), BGE=3745 (8%), Other=3023 (7%)`.
- MAS action source breakdown (citations contributed across run): `bm25_initial=29200, mas_supportive=8846, mas_crossref=3680, mas_rewrite=2647, mas_supplement=2353, mas_decompose=1711, llm_stage1=800, llm_stage1_procedural=800, bm25_top10=400, explicit_from_query=30`. Avg overlap fraction 0.08 (actions are complementary).

### Stage 3 (graph expansion, ~<2s/query)
- Co-occurrence indexes loaded successfully (3,840 / 4,991 / 80,569 / 67,513 / 81,709 entries).
- Per-query expansions (sample): `test_001: 1395 → 7003 (+504 graph-cases, +244 tm-BGE, +292 tm-numbered, +1233 num→BGE, +628 num-cocit)`; `test_028: 1541 → 8466`.
- Totals: `45,328 → 235,113 citations (+189,785 from graph)`.
- Post-S3 distribution: `cands/q: min=3313 max=8466 mean=5877.8`; `BGE/q: min=1092 max=2733 mean=1920.8`; `Numbered/q: min=924 max=2368 mean=1795.5`.

### Stage 4 (LLM reranker, pool=150)
- Per-query scored counts (subset): test_001=68 (high=68/med=0) 77.4s; test_002=129 (high=8/med=74) 208.4s; test_012=198 (high=193/med=5) 167.5s; test_025=40 (high=40/med=0) 111.1s; test_036=86 (high=24/med=62) 154.1s. Pool size constant at 150 for all 40 queries. Wall time ~75–260s per query.
- Aggregate diagnostics: `reranked/q: min=36 max=198 mean=109.8`; `direct-gen/q: min=0 max=0 mean=0.0` (skipped).
- Direct-gen validity: `0/0 = 0.0%` (skipped).
- Tier distribution across all queries: T10=13 (0.3%), T9=50 (1.1%), T8=283 (6.4%), T7=901 (20.5%), T6=2267 (51.6%), T5=508 (11.6%), T4=106 (2.4%), T3=160 (3.6%), T2=52 (1.2%), T1=51 (1.2%).

### Stage 5 (verify + score)
- Precomputed per-query BM25 scoring counts: e.g. `test_001: 2154/2945 scored in 72.0s`; `test_028: 2693/3465 scored in 85.2s`. Range across 40 queries roughly 977–2693 scored / 1324–3465 candidates.
- No cross-encoder file at `checkpoints/reranker_scores_test.json` — falls back to "binary reranker only".
- Total verified: **91,380**; dropped: **12,064**.
- Loaded tuned threshold = **0.65** (val macro-F1 = **0.49752876955046743**).
- Submission summary (Stage 5 raw): Queries=40, Avg citations=24.5, Max=62, Queries with 0=10.
- Abs-flooding events (same Art ≥3 Abs variants): 0 across 0 queries.

### Submission validation
- `submission_test.csv` reports: Queries=40, Avg citations=24.5, Max=62, Queries with 0=10. VALID.

### Stage 6 (adaptive-K, final submission)
- Score source: `bm25_plus_llm (w_llm=0.5)`.
- Post-processing: `74,937 → 48,625 candidates` (Abs-collapse + Numbered suppression, quantile=0.9, require_graph=False).
- K strategy: `score_gap (frac_of_max=0.55, elbow_drop=0.12)`.
- Final submission: Mean predictions per query=44.8, min=39, max=45 (clamped at STAGE6_MAX_K=45).

### Other diagnostic prints
- Statute→BGE index successfully (re)loaded with 3,840 entries (vs original graph 0 stat→BGE edges).
- Stage 2 diag 4 (multi-query ablation) and Stage 2 diag 5 (dead query variants): "skipped (not val split or no Stage 1 checkpoint)".
- Stage 3 diag 4 (BGE safety-net): "skipped (val-only)".

## Summary

The notebook executes the full Stage 0→6 BM25-first / LegalMALR-adapted citation retrieval pipeline on the 40-query `test` split, ending with an adaptive-K Stage 6 submission. It successfully runs Stage 1 (Qwen3-32B query analysis: 99.5% candidate validity, 22/40 queries with HyDE drift), Stage 2 (MAS produces mean 1133 cands/q across 4 iterations, with OOM fallback on iter 4), Stage 3 (graph expansion grows the pool 5× to mean 5878 cands/q, decisively repairing the 0-edge statute→BGE gap with 3,840 text-mined entries), Stage 4 (Qwen3-32B reranker on a 150-candidate pool, ~75–260s/query, mean 109.8 scored), Stage 5 (BM25 continuous scoring + composite confidence, **val macro-F1 = 0.4975** at threshold 0.65), and Stage 6 (`bm25_plus_llm` re-score + Abs-collapse + Numbered FP suppression, score-gap K, mean 44.8 predictions/query). Worked: the text-mined statute→BGE index plugged the graph hole; multi-agent action sources are complementary (overlap 0.08); reranker tiering and Stage 6 Abs-collapse held flooding to 0 events; final submission validates. Failed/limitations: HyDE narrative variants are effectively dead (`DE_2` only 1.1 gold hits/run vs `EN_raw` 12.1) and were excluded from BM25; direct-gen was disabled because val showed ~60% hallucination; Stage 2 iter-4 hit OOM; 10/40 queries received 0 predictions out of Stage 5 (compensated by Stage 6 floor); val/test law-code overlap is poor (test has 6 codes absent from val), implying threshold transfer risk. Lessons captured inside the notebook: BM25 needs exact legal tokens (HyDE prose is the wrong signal); Stage 5's composite scoring under-performed pure `bm25_plus_llm` on val so Stage 6 bypasses it; val cites/q (25.1) is >2× train (4.1), so thresholds must be tuned on val only; and Numbered citations need a high BM25 quantile (0.90) to suppress FPs.
