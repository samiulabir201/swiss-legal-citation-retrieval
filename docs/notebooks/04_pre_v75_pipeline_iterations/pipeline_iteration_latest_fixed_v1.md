# pipeline_iteration_latest_fixed_v1.ipynb

**Path:** notebooks/04_pre_v75_pipeline_iterations/pipeline_iteration_latest_fixed_v1.ipynb

## Configuration

### Runtime / environment
- Platform: Google Colab (Drive mounted at `/content/drive`).
- `PROJECT_ROOT = /content/drive/MyDrive/swiss_law`.
- Working dir set to `PROJECT_ROOT`.
- HF cache (`HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`) = `PROJECT_ROOT/model`.
- `HF_HUB_DISABLE_XET=1`.
- Pip install: `deep-translator rank-bm25 scikit-learn`.

### Models
- LLM (Stage 1 / 2 / 4): `Qwen/Qwen3-32B` in bf16, full precision (no quantization). `device_map="auto"`. ~64 GB VRAM (H100 80 GB target). Default sampling: `temperature=0.6, top_k=20, top_p=0.95, max_new_tokens=2048`, `enable_thinking=False` (except Stage 1).
- Reranker (Stage 4b, disabled): `Qwen/Qwen3-Reranker-4B`.

### Libraries
- `numpy`, `pandas`, `torch`, `scipy.sparse`, `rank_bm25.BM25Okapi`, `tqdm`, `transformers (AutoTokenizer, AutoModelForCausalLM)`, `deep_translator.GoogleTranslator`.

### Runtime knobs (cell 21)
- `SPLIT = "val"`, `BACKEND = "local"`.
- `STAGE3_GOLD_PRIOR_MODE = "off"`.
- `STAGE2_MAX_ITERATIONS = 4`.
- `STAGE4_SKIP_DIRECT_GEN = True` (direct-gen was 60% hallucinated).
- `STAGE4_RANK_POOL_SIZE = 80` (Stage 4 ranker pool).
- `STAGE4B_TOP_N = None` (Stage 4b cross-encoder disabled).
- `STAGE5_THRESHOLD = None` (val: tuned).
- `STAGE3_BUILD_STATUTE_TO_BGE = True`, `STAGE3_STATUTE_TO_BGE_TOPK = 5`.
- `STAGE3_USE_NUMBERED_INDEXES = True`, `STAGE3_STATUTE_TO_NUMBERED_TOPK = 5`, `STAGE3_NUMBERED_TO_BGE_TOPK = 3`, `STAGE3_NUMBERED_COCITATION_TOPK = 3`.
- `STAGE6_NUMBERED_BM25_QUANTILE = 0.75`, `STAGE6_ABS_COLLAPSE = True`, `STAGE6_ABS_MAX_VARIANTS = 2`, `STAGE6_MIN_K = 10`, `STAGE6_MAX_K = 45`, `STAGE6_USE_ADAPTIVE_K = True`, `STAGE6_PROCEDURAL_FLOOR = True`, `STAGE6_SCORE_SOURCE = "bm25_plus_llm"`, `STAGE6_LLM_BOOST = 0.5`.

### Default pipeline config (cell 22)
```
bm25_top_k=100, graph_top_k=50,
mas_max_iterations=4, mas_bm25_top_k=30,
reranker_batch_size=20, threshold=0.15,
confidence_weights = {explicit_from_query:0.40, bm25_top10:0.25,
                     citation_graph:0.15, llm_reranker:0.15, llm_direct_gen:0.05}
```

### Scoring weights (Stage 5, cell 9 / 48)
```
DEFAULT_WEIGHTS = {explicit_from_query:0.40, bm25_top10:0.15, bm25_initial:0.10,
                   citation_graph:0.10, llm_reranker_tier3:0.25,
                   llm_reranker:0.15, llm_direct_gen:0.05, llm_stage1:0.05,
                   llm_stage1_procedural:0.05, co_citation:0.03}
BM25_CONTINUOUS_WEIGHT = 0.10
CROSS_ENCODER_WEIGHT   = 0.18
MULTI_SIGNAL_BONUS     = 0.10 (3+ signals); +0.20 (4+); +0.05 (2+)
```

### Constants
- `ARTIFACT_FORMAT = "chunked_bm25_v1"`, `BM25_CHUNK_SIZE = 10000`.
- BM25 corpus: 171,654 law + 1,985,178 court = 2,156,832 docs (822,251 terms, 212,228,366 non-zero entries).
- Alternate-language abbreviation canon map: CC→ZGB, CO→OR, LP→SchKG, CPC→ZPO, LDIP→IPRG, LFors→GestG, CPP→StPO, CP→StGB, LTF→BGG, PA→VwVG.
- BGE/numbered case regex patterns built into `extract_explicit_citations`.

## Data

All paths anchored on `PROJECT_ROOT = /content/drive/MyDrive/swiss_law`.

### Raw competition CSV (`PROJECT_ROOT/data/`)
- `train.csv` (1,139 rows, 4.1 cites/q mean, 99% Law / 1% BGE)
- `val.csv` (10 rows, 25.1 cites/q mean, 59% Law / 27% BGE / 13% Numbered)
- `test.csv` (40 rows)
- `sample_submission.csv`
- `laws_de.csv` (175,933 rows, SR-style codes, "FYI only")
- `court_considerations.csv` (2.26 GB present)

### Index artifacts (`PROJECT_ROOT/index/`)
- `corpus.parquet`
- `laws_knowledge_base.jsonl` (179,641 records — KB, canonical abbreviations)
- `bm25_v2_index.pkl` + `bm25_v2_ids.pkl`
- `bm25_v2_index_parts/bm25_v2_index_parts/` (216 chunked BM25 parts)
- `citation_graph.pkl` (103,125 articles with case links; 914,658 cases with article links; 2,201,708 nodes; statute→BGE/statute→num/BGE→BGE/BGE→stat edges all 0)
- `citation_lookup.pkl`
- `citation_signal_lookup.pkl`, `citation_signal_schema_v2.json`
- `reference_graph_v2.pkl`
- `gold_cocitation_prior_train.pkl`, `gold_cocitation_prior_trainval.pkl`
- `statute_to_bge.pkl` (3,840 entries)
- `statute_to_numbered.pkl` (4,991 entries)
- `numbered_to_statutes.pkl` (80,569 entries)
- `numbered_to_bge.pkl` (67,513 entries)
- `numbered_cocitation.pkl` (81,709 entries)
- `query_translations_trainval.json` (read-only src; copied to checkpoints)
- `token_law_freq.json`
- `statute_signals.jsonl`, `case_signals.jsonl`

### Runtime outputs
- `PROJECT_ROOT/checkpoints/stage{1..5}_{split}.json`
- `PROJECT_ROOT/checkpoints/bm25_scores_{split}.json`
- `PROJECT_ROOT/checkpoints/reranker_scores_{split}.json` (not present — Stage 4b disabled)
- `PROJECT_ROOT/checkpoints/shadow_val.csv` (150 train rows w/ gold≥6)
- `PROJECT_ROOT/submissions/submission_{split}.csv`
- `PROJECT_ROOT/configs/threshold.json`
- `PROJECT_ROOT/model/` (HF cache; Qwen3-32B safetensors)

## Pipeline

### Stage 0 — Pre-execution: pre-flight diagnostics (cells 29-35)
Seven read-only checks: (1) val/train/test distribution & type mix, (2) KB integrity vs. expected per-code floors, (3) gold Abs/lit/bare-Art format audit, (4) procedural-boilerplate census (BGG/ZPO/VwVG/ATSG in ≥2 queries), (5) BGE reachability in `court_considerations.csv`, (6) test-vs-val law-code domain overlap, (7) citation-graph edge inventory.

### Inlined utilities (cells 6-18)
- `ChunkedBM25Okapi`: BM25 over scipy CSC sparse TF matrix; `get_scores` and `get_batch_scores` fully vectorized.
- `load_bm25_artifact`: loads chunked artifact (n_chunks=216), idf, doc_len.
- `score_citation` / `score_all_citations`: additive composite + continuous BM25 + graduated multi-signal bonus.
- `optimize_threshold`: sweeps 0.05–0.95, step 0.01, on val gold to maximize macro-F1.
- `extract_explicit_citations`: regex for single/multi-Art statutory refs, BGE (with/without E.), numbered cases.
- `CitationGraph` (direct + co-citation + PMI variants).
- `CitationLookup` (canonical normalization, `expand_to_abs`, `collapse_to_article`).
- `tokenise`, `SparseRetriever.search_statutory/caselaw/combined`, PMI token→law boost (`law_boost=10, top_n_laws=5`).
- `GraphRetriever.articles_to_cases` (RRF-style `1/(60+rank)`), `cases_to_articles`, `bidirectional_expand`.
- `get_model_and_tokenizer` (singleton Qwen3-32B), `generate`, `generate_batch`, `generate_batch_multi_system`, `generate_json*` (OOM fallback to sequential, halved tokens).
- `Verifier.verify_and_normalize`: corpus-aware canonical-casing fix, Abs/lit fallback, lenient BGE/numbered regex.
- `_compute_bm25_scores`: per-query normalized BM25 over candidate citations only (single merged-token pass; no PMI).

### Stage 1 — Query Analysis (cell 54)
Six-step LLM analysis per query: (1) regex explicit-citation extraction, (2) BM25 probe, (3) domain scaffolding from KB, (4) Qwen3-32B legal-issue analysis, (5) HyDE German hypothetical-document generation, (6) cross-lingual term expansion. Batched (size 5) Qwen3-32B calls. Output: `stage1_{split}.json` with `explicit_citations`, `candidate_articles`, `candidate_cases`, `search_queries_en/de`, `hyde_documents_de`, `law_areas`, `legal_issues`, `bm25_probe`.

### Stage 2 — MAS Retrieval (cell 67)
Planner → {REWRITE, SUPPLEMENT, DECOMPOSE, SUPPORTIVE, CROSSREF} → BM25, max 4 iterations, batched across 3 queries with sequential OOM fallback. Initial BM25 + per-iteration agent-generated DE queries unioned into `candidate_pool`. Sources tracked per citation.

### Stage 3 — Graph Expansion (cell 81)
Loads 5 co-occurrence indexes (statute→BGE, statute→numbered, numbered→statutes, numbered→BGE, numbered_cocitation). For each Stage 2 candidate, expands via direct article→cases graph, text-mined statute→BGE/numbered, numbered→BGE, numbered_cocitation, plus Abs sibling expansion (`article_to_abs` 43,066 entries). Output: `expanded_pool` per query.

### Stage 4 — LLM Reranker (cell 94)
Single-pass Qwen3-32B continuous 0-10 scoring of a per-query pool of size 80 (BM25 top-N + procedural + explicit + Stage 3 case candidates). `max_new_tokens=4500`. Direct-gen disabled. Output: `reranked`, `reranked_tiers` (cite→0–10), `all_citations`, `sources`.

### Stage 5 — Verify + Score + Threshold (cell 105)
- 5A: `Verifier.verify_and_normalize` (canonical casing, Abs/lit fallback, lenient case patterns).
- 5B: continuous BM25 per-query normalized scores (cached to `bm25_scores_val.json`); composite confidence with binary signals + multi-signal bonus + BM25 continuous; cross-encoder skipped (no `reranker_scores_val.json`).
- 5C: threshold sweep on val gold → `configs/threshold.json`; submission written to `submissions/submission_val.csv`.

### Diagnostics dashboard (cells 119-121)
D1 ranking ceiling, D2 adaptive-K oracle, D4 CE calibration, D6 shadow-val builder. D3 (BGE-M3) and D5 (LLM-ranker smoke) gated GPU diagnostics.

### Stage 6 — Adaptive-K + Procedural floor (cell 126)
Re-scores Stage 5 candidates using `bm25_plus_llm` (BM25 normalized + 0.5·llm/10), applies Abs-collapse + numbered-suppression post-processing, picks per-query K from score-gap (`frac_of_max=0.55, elbow_drop=0.12`), bounded `[10, 45]`. Overwrites `submission_val.csv`.

## Results

### Pre-diag 1/7 — data distribution (cell 29)
```
[val]   rows=10    cites/q: min=10 max=47 mean=25.1 median=22.0
        types: Law=149 (59%), BGE=69 (27%), Numbered=33 (13%)
[train] rows=1139  cites/q: min=1 max=44 mean=4.1 median=2.0
        types: Law=4602 (99%), BGE=57 (1%)
[test]  rows=40
** WARNING: val (25.1/q) has >2× train (4.1/q) — NEVER tune thresholds on train
```

### Pre-diag 2/7 — KB integrity (cell 30)
```
KB (laws_knowledge_base.jsonl): 179,641 records
top codes: {'EFZ': 6500, 'OR': 3667, 'EBA': 2469, 'ZGB': 2405, 'StPO': 1306,
            'StGB': 1244, 'TSV': 1094, 'VTS': 1023, 'SchKG': 919, 'ZV': 874}
KB ok.
laws_de.csv: 175,933 rows
court_considerations.csv: 2.26 GB present
All required artifacts present.
```

### Pre-diag 3/7 — Gold format audit (cell 31)
```
Abs       121 (81%)
bare_Art   28 (19%)
NOTE: 81% of Art cites are sub-article — Abs-level matching is critical
```

### Pre-diag 4/7 — Procedural-boilerplate census (cell 32)
```
Procedural cites appearing in ≥2 val queries: 1
  Art. 100 Abs. 1 BGG  in 9 queries  (never in query text)
```

### Pre-diag 5/7 — BGE reachability (cell 33)
```
Unique val-gold BGE stems: 54
Present in corpus: 54/54
```

### Pre-diag 6/7 — Test vs val domain overlap (cell 34)
```
Law codes in test queries: {'StPO': 5, 'SVG': 4, 'PrHG': 4, 'OR': 4, 'UVG': 4,
  'IPRG': 3, 'ZGB': 3, 'CO': 2, 'SchKG': 2, 'ATSG': 1, 'CC': 1, 'ZPO': 1, 'MSchG': 1}
Law codes in val queries:  {'OR': 3, 'StPO': 2, 'ZGB': 2}
Law codes in val gold:     {'ZGB': 39, 'StPO': 36, 'OR': 18, 'BGG': 13,
                           'StGB': 12, 'IVG': 10, 'ATSG': 7, 'ZPO': 4, 'BV': 3,
                           'IPRG': 2, 'SchKG': 1}
** Codes in test but NOT covered by val: ['CC', 'CO', 'MSchG', 'PrHG', 'SVG', 'UVG']
```

### Pre-diag 7/7 — Citation graph edges (cell 35)
```
Loaded: index/citation_graph.pkl
nodes: 2,201,708
stat→BGE: 0   <-- KEY
stat→num: 0
BGE→BGE: 0
BGE→stat: 0
other: 3,897,617
** NO statute→BGE edges. BGE recall via graph expansion is impossible.
```

### Stage 1 — outputs
- Stage 1 batched LLM: 306.9 s for batch-1 (61.4 s/query), 93.0 s for batch-2 (18.6 s/query).
- Per-query: explicit_citations 0-3, candidate_articles=10, candidate_cases=10, search_queries_de=5.
- Diag 1/6 — artifact summary:
  ```
  explicit_citations         min=0 max=3 mean=0.9
  candidate_articles         min=10 max=10 mean=10.0
  candidate_cases            min=10 max=10 mean=10.0
  search_queries_en          min=2 max=5 mean=4.3
  search_queries_de          min=5 max=5 mean=5.0
  hyde_documents_de count    min=2 max=2 mean=2.0
  hyde_documents_de tokens   min=68 max=117 mean=95.2
  law_areas                  min=1 max=3 mean=1.9
  legal_issues               min=1 max=5 mean=2.7
  ```
- Diag 2/6 — Explicit-extraction fidelity on val: 6/6 = 100.0%.
- Diag 3/6 — Candidate-article validity: 100/100 = 100.0%.
- Diag 4/6 — Routing precision (mean): 0.313, Routing recall: 0.955.
- Diag 5/6 — HyDE drift: 3/10 queries suspicious (val_004, val_007, val_008).
- Diag 6/6 — Free-recall ceiling Stage 1: 126/251 = 50.2% (Law 55.0%, BGE 47.8%, Numbered 33.3%).

### Stage 2 — outputs
- Batch-1 runtime: 742.2 s, Batch-2: 683.3 s. OOM fallback to sequential triggered in iter 3-4 of batch 2.
- Per-iteration MAS deltas (val_001): SUPPORTIVE +116, DECOMPOSE +37, SUPPLEMENT +90, CROSSREF +225 → 1166 candidates.
- Diag 1/6: candidates/q min=1068 max=1503 mean=1243.2; by type Numbered=7365 (59%), Law=3315 (27%), BGE=934 (8%), Other=818 (7%).
- Diag 2/6 — val recall per type:
  ```
  Law: 122/149 = 81.9%
  BGE: 49/69 = 71.0%
  Numbered: 25/33 = 75.8%
  ```
- Diag 3/6 — R@K curves:
  ```
  K      Law          BGE          Numbered
   10  66/149=44.3%  17/69=24.6%   2/33= 6.1%
   50  86/149=57.7%  42/69=60.9%  22/33=66.7%
  100  86/149=57.7%  42/69=60.9%  22/33=66.7%
  200  90/149=60.4%  43/69=62.3%  24/33=72.7%
  500 102/149=68.5%  45/69=65.2%  24/33=72.7%
  1000 109/149=73.2% 48/69=69.6%  25/33=75.8%
  ```
- Diag 4/6 — single vs multi-query ablation:
  ```
  EN only (raw query)        Law: 59/149=39.6%, BGE: 45/69=65.2%, Numbered: 25/33=75.8%
  Single DE query            Law: 17/149=11.4%, BGE: 2/69=2.9%,  Numbered: 2/33=6.1%
  All DE queries (≤10)       Law: 28/149=18.8%, BGE: 4/69=5.8%,  Numbered: 2/33=6.1%
  All DE + HyDE              Law: 28/149=18.8%, BGE: 5/69=7.2%,  Numbered: 3/33=9.1%
  ```
- Diag 5/6 — variant gold-hit avg/run: EN_raw=12.10 (best), EN_0=7.10, EN_1=4.30; DE_1=0.00 DEAD, DE_4=0.30 DEAD, HYDE_1=0.00 DEAD.
- Diag 6/6 — Avg overlap fraction = 0.08 (actions complementary). Action totals: bm25_initial 7669, mas_supportive 1981, mas_crossref 1709, mas_rewrite 825, mas_supplement 775, llm_stage1 200, llm_stage1_procedural 200, mas_decompose 197, bm25_top10 82, explicit_from_query 9.

### Stage 3 — outputs
- Co-occurrence indexes loaded: statute→BGE 3,840; statute→Numbered 4,991; numbered→statutes 80,569; numbered→BGE 67,513; numbered→co-cited 81,709.
- Citation graph: 103,125 articles with case links, 914,658 cases with article links.
- Per-query expansion (val_001): 1166 → 7391 candidates (+235 graph-cases, +210 tm-BGE, +313 tm-numbered, +1215 num→BGE, +1824 num-cocit) in 0.48 s.
- Total: 12,432 → 85,653 (+73,221 from graph).
- Diag 1/5: cands/q min=7371 max=10604 mean=8565.3; BGE/q mean=2262.9; Numbered/q mean=3527.8. Recall: Law 141/149=94.6%, BGE 59/69=85.5%, Numbered 25/33=75.8%.
- Diag 2/5 — graph reachability 1-hop / 2-hop for val-gold BGE: **0/56 = 0.0%** at both hops (statute→BGE edges missing in `citation_graph.pkl`).
- Diag 3/5 — procedural coverage: Art. 82 BGG 1/1, Art. 113 BGG 1/1. Total proc gold events 2, covered 2 = 100%.
- Diag 4/5 — BGE safety-net: 0/4 missed BGEs recoverable via statute-text mining = 0%.
- Diag 5/5 — Abs-level coverage on val gold:
  ```
  abs_exact_hit   113/121 = 93%
  abs_parent_only   7/121 =  6%
  abs_missed        1/121 =  1%
  Abs-flooding events (Art with ≥3 Abs variants): 1328 across 10 queries
  ```

### Stage 4 — outputs
- Pool size: 80 per query (`bm25 + procedural + explicit + graph`).
- Per-query ranker: 57-79 scored, 79-201.8 s. val_001: 99.9 s, val_006: 201.8 s, val_007: 178.5 s.
- Diag 1/5: reranked/q min=57 max=79 mean=69.8; direct-gen/q = 0.
- Diag 2/5: direct-gen validity 0/0 (skip).
- Diag 3/5 — Tier distribution (0-10 scores):
  ```
  T10:    3 (0.4%)
  T9 :   19 (2.7%)
  T8 :   62 (8.9%)
  T7 :  174 (24.9%)
  T6 :  268 (38.4%)
  T5 :  137 (19.6%)
  T4 :   20 (2.9%)
  T3 :    5 (0.7%)
  T2 :    3 (0.4%)
  T1 :    7 (1.0%)
  ```
- Diag 4/5 — Gold dropped Stage 3 → Stage 4: 7/225 (val_002 Art. 56 Abs. 1 ATSG, BGE 140 V 193 E. 3.2; val_003 BGE 145 I 167, BGE 143 IV 330, Art. 16 ZGB, BGE 145 IV 99; val_007 Art. 292 StGB).
- Diag 5/5 — Stage 4 full-pool recall (ceiling for Stage 5): Law 138/149=92.6%, BGE 55/69=79.7%, Numbered 25/33=75.8%. Gold tier passthrough: Law 91 passthru, BGE 37 passthru / 14 absent.

### Stage 5 — outputs
```
Precomputing BM25 scores (cached): 1633-2597 cites scored per query in 43.7-98.0 s
Total verified: 23655, dropped: 2876
No cross-encoder scores — using binary reranker only.

Tuning threshold on val gold...
  Best threshold: 0.65
  Best macro-F1:  0.4836

Per-query F1 @ threshold=0.65:
  val_001: F1=0.3846 (pred=10, gold=42)
  val_002: F1=0.4348 (pred=10, gold=36)
  val_003: F1=0.5075 (pred=20, gold=47)
  val_004: F1=0.3077 (pred=16, gold=10)
  val_005: F1=0.5000 (pred=13, gold=11)
  val_006: F1=0.4348 (pred=5,  gold=18)
  val_007: F1=0.4848 (pred=14, gold=19)
  val_008: F1=0.6154 (pred=23, gold=29)
  val_009: F1=0.6250 (pred=18, gold=14)
  val_010: F1=0.5417 (pred=23, gold=25)

  Submission summary: Avg citations 15.2, Max 23, Queries with 0: 0.
```
- Diag 1/6 — Per-type @ thr=0.65: Law P=0.971 R=0.456 F1=0.621 (tp=68 fp=2 fn=81); BGE P=0.806 R=0.362 F1=0.500 (tp=25 fp=6 fn=44); Numbered P=0.118 R=0.182 F1=0.143 (tp=6 fp=45 fn=27).
- Diag 4/6 — Misses by type: Law=81, BGE=44, Numbered=27. False positives: Numbered=45, BGE=6, Law=2.
- Diag 5/6 — Abs-flooding events in predictions: 0 (suppressed).
- Diag 6/6 — Procedural-prior potential: catchable +2 TP at cost of +148 FP → selective prior required.

### Submission validation (cell 116)
```
Validating: /content/drive/MyDrive/swiss_law/submissions/submission_val.csv
  ERROR: Missing 40 test query IDs
Submission valid: False
```
(Expected — submission was written from `SPLIT="val"`, not test.)

### Diagnostics (cell 121)
```
D1 — Ranking ceiling on val:
  val_001: gold=42 in_pool=42 in_bm25=42 med_rank=21.5 max_rank=424
  val_002: gold=36 in_pool=34 in_bm25=32 med_rank=16.5 max_rank=1789
  val_003: gold=47 in_pool=39 in_bm25=35 med_rank=18   max_rank=486
  val_004: gold=10 in_pool=9  in_bm25=9  med_rank=29   max_rank=248
  val_005: gold=11 in_pool=10 in_bm25=10 med_rank=5.5  max_rank=75
  val_006: gold=18 in_pool=14 in_bm25=14 med_rank=7.5  max_rank=1029
  val_007: gold=19 in_pool=17 in_bm25=16 med_rank=8.5  max_rank=980
  val_008: gold=29 in_pool=27 in_bm25=27 med_rank=14   max_rank=1645
  val_009: gold=14 in_pool=14 in_bm25=14 med_rank=7.5  max_rank=213
  val_010: gold=25 in_pool=19 in_bm25=19 med_rank=10   max_rank=107

D2 — Adaptive-K oracle on val:
  BM25 oracle-K macro-F1:    0.6767
  Stage-5 oracle-K macro-F1: 0.5232

D4 — CE calibration: skipped (no reranker_scores_val.json).

D6 — Shadow-val: Train rows with gold>=6: 253. Wrote 150 rows to shadow_val.csv.
       Gold-size: mean=12.5 median=11 min=6 max=44.
```

### Stage 6 — adaptive-K (cell 126)
```
Score source: bm25_plus_llm (w_llm=0.5)
Post-proc: 19878 → 13086 candidates (Abs-collapse + Numbered suppression)
K strategy: score_gap (frac_of_max=0.55, elbow_drop=0.12)

  val_001: K=34 preds=34 gold=42 P=1.000 R=0.810 F1=0.895
  val_002: K=25 preds=25 gold=36 P=1.000 R=0.694 F1=0.820
  val_003: K=18 preds=23 gold=47 P=0.870 R=0.426 F1=0.571
  val_004: K=10 preds=13 gold=10 P=0.462 R=0.600 F1=0.522
  val_005: K=10 preds=10 gold=11 P=0.600 R=0.545 F1=0.571
  val_006: K=10 preds=10 gold=18 P=0.900 R=0.500 F1=0.643
  val_007: K=10 preds=10 gold=19 P=1.000 R=0.526 F1=0.690
  val_008: K=16 preds=16 gold=29 P=1.000 R=0.552 F1=0.711
  val_009: K=10 preds=10 gold=14 P=1.000 R=0.714 F1=0.833
  val_010: K=17 preds=17 gold=25 P=0.882 R=0.600 F1=0.714

  === MACRO F1 (adaptive-K): 0.6970  P=0.8713  R=0.5967 ===

  Saved adaptive-K submission: submissions/submission_val.csv
  Mean predictions per query: 16.8 (min=10, max=34)
```

### HyDE post-mortem (cell 128, prose)
EN_raw 121 gold hits vs DE_2 11, HYDE_1 0. Decision: keep HyDE in checkpoint for debugging; do not route into BM25.

## Summary

The notebook is a Colab-only, all-stages-in-one "fixed v1" rewrite of the BM25-first / LegalMALR-adapted Swiss citation pipeline. End-to-end runs on the 10-query val set produced a Stage-5 macro-F1 of **0.4836** at threshold 0.65, and a Stage-6 adaptive-K macro-F1 of **0.6970** (P=0.8713, R=0.5967) — Stage 6 (BM25+0.5·LLM, score-gap K with floor 10/ceiling 45 + Abs-collapse + Numbered suppression) was the decisive lift over Stage 5. What worked: Stage 1 explicit extraction (100%), candidate validity (100%), Stage 2 MAS recall (Law 81.9% / BGE 71.0%), Stage 3 graph expansion (Law 94.6% / BGE 85.5%), and the BM25 oracle-K ceiling at 0.6767 confirmed that recall was sufficient and ranking was the bottleneck. What failed: HyDE/DE search variants were dead (EN_raw dominated 12.10 vs 0-1.6 gold-hits/run), `citation_graph.pkl` had 0 statute→BGE / statute→numbered edges (worked around by the text-mined indexes), Stage 5 Numbered F1=0.143 with 45 FPs, and 7 gold dropped between Stage 3 and Stage 4. Key lessons: don't tune on train (val averages 25.1 cites/q vs train 4.1), 81% of Art gold is Abs-level so Abs-collapse must be selective, val/test law-code coverage gap (CC/CO/MSchG/PrHG/SVG/UVG only in test), and adaptive per-query K outperforms a single global threshold by ~22 F1 points on this distribution.
