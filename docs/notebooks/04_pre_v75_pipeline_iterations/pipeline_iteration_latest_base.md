# pipeline_iteration_latest_base.ipynb

**Path:** notebooks/04_pre_v75_pipeline_iterations/pipeline_iteration_latest_base.ipynb

## Configuration

### Runtime environment
- Notebook execution target: Google Colab (mounts `/content/drive`).
- `PROJECT_ROOT = /content/drive/MyDrive/swiss_law`
- `RAW_DATA_DIR = PROJECT_ROOT/data`
- `INDEX_DIR = PROJECT_ROOT/index`
- `CHECKPOINTS_DIR = PROJECT_ROOT/checkpoints`
- `SUBMISSIONS_DIR = PROJECT_ROOT/submissions`
- `MODELS_DIR = PROJECT_ROOT/model` (used as `HF_HOME` and `HUGGINGFACE_HUB_CACHE`)
- `os.chdir(PROJECT_ROOT)`; `PROJECT_ROOT` prepended to `sys.path`.
- 129 cells (mix of markdown source-inlines and code).

### Runtime knobs (cell 23)
- `SPLIT = "val"`
- `BACKEND = "local"`
- `STAGE3_GOLD_PRIOR_MODE = "off"`
- `STAGE2_MAX_ITERATIONS = 4`
- `STAGE4_SKIP_DIRECT_GEN = False`
- `STAGE4B_TOP_N = None` (stage default)
- `STAGE5_THRESHOLD = None` (stage default)

### default_config (cell 24)
```python
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

### Models
- `QWEN3_32B_HF_ID = "Qwen/Qwen3-32B"` (full bf16, ~64 GB VRAM, single GPU model for Stages 1/2/4 LLM calls).
- `QWEN3_RERANKER_ID = "Qwen/Qwen3-Reranker-4B"` (Stage 4b cross-encoder, bf16).
- Loader sets `os.environ["SWISS_QWEN3_32B_PATH/_HF_ID/_RERANKER_ID/_PIPELINE_ROOT/_DATA_DIR/_INDEX_DIR/_MODELS_DIR"]`.

### Libraries (cell 7 imports + cell 6 pip install)
- Standard: `argparse`, `csv`, `gc`, `importlib`, `json`, `math`, `os`, `pickle`, `random`, `re`, `sys`, `threading`, `time`, `traceback`, `collections.Counter/defaultdict`, `concurrent.futures.ThreadPoolExecutor/as_completed`, `itertools.combinations`, `pathlib.Path`, `typing.Any`.
- Third-party: `numpy`, `pandas`, `torch`, `deep_translator.GoogleTranslator`, `rank_bm25.BM25Okapi`, `scipy.sparse`, `tqdm`, `transformers.AutoTokenizer/AutoModelForCausalLM`.
- Installed inline: `pip install -qU deep-translator rank-bm25`.
- Project requirements declared (cell 22): `rank-bm25==0.2.2`, `transformers>=4.45.0`, `bitsandbytes>=0.44.0`, `accelerate>=1.0.0`, `huggingface_hub>=0.25.0`, `safetensors>=0.4.0`, `anthropic>=0.40.0`, `openai>=1.50.0`, `pandas>=2.0.0`, `numpy>=1.26.0`, `pyarrow>=14.0.0`, `tqdm>=4.66.0`, `deep-translator>=1.11.0`. Target Python 3.12, CUDA 12.1 (RTX 4050 6 GB stated locally; Colab notebook actually runs the 32B on a much larger GPU).

### Hardware
- Stages 1/2/4 require a GPU large enough for Qwen3-32B bf16 (~64 GB VRAM; H100 80 GB recommended).
- Stage 2 iter 4 hit `[OOM] Batch of 3 failed -- falling back to sequential` in three of four batches.
- Stage 4b runs the 4B reranker (CPU+GPU fits comfortably).

## Data

### Raw competition data (`COMP_DATA = RAW_DATA_DIR`)
- `/content/drive/MyDrive/swiss_law/data/train.csv` (1139 rows)
- `/content/drive/MyDrive/swiss_law/data/val.csv` (10 rows)
- `/content/drive/MyDrive/swiss_law/data/test.csv` (40 rows)
- `/content/drive/MyDrive/swiss_law/data/sample_submission.csv`
- `/content/drive/MyDrive/swiss_law/data/court_considerations.csv` (2.26 GB)
- `/content/drive/MyDrive/swiss_law/data/laws_de.csv` (175,933 rows, SR-style codes; FYI only)

### Knowledge base and corpus (`LAW_DB = INDEX_DIR`)
- `/content/drive/MyDrive/swiss_law/index/laws_knowledge_base.jsonl` (179,641 records; top codes EFZ 6500, OR 3667, EBA 2469, ZGB 2405, StPO 1306, StGB 1244, TSV 1094, VTS 1023, SchKG 919, ZV 874)
- `/content/drive/MyDrive/swiss_law/index/corpus.parquet`

### Pre-built index artifacts (`LAW_DB`)
- `bm25_v2_index.pkl` (and chunked parts under `bm25_v2_index_parts/bm25_v2_index_parts/`, 216 chunks)
- `bm25_v2_ids.pkl`
- `citation_graph.pkl` (2,201,708 nodes; 3,897,617 "other" edges; statute→BGE, statute→numbered, BGE→BGE, BGE→statute all 0)
- `citation_lookup.pkl`
- `citation_signal_lookup.pkl`
- `reference_graph_v2.pkl`
- `gold_cocitation_prior_train.pkl`
- `gold_cocitation_prior_trainval.pkl`
- `query_translations_trainval.json` (read-only source; copied to `CHECKPOINTS_DIR`)
- `token_law_freq.json`
- `citation_signal_schema_v2.json`
- `statute_signals.jsonl`, `case_signals.jsonl`
- Missing: `index/statute_to_bge.parquet` (flagged by Pre-diag 2/7 and Stage 3 diag 2/5).

### Outputs written
- `checkpoints/stage1_val.json`, `stage2_val.json`, `stage3_val.json`, `stage4_val.json`, `stage5_val.json`
- `checkpoints/reranker_scores_val.json`, `bm25_scores_val.json`, `ce_config.json`
- `submissions/submission_val.csv`, `submissions/submission_val_pertype.csv`

## Pipeline

### Stage 0 — Pre-run diagnostics (cells 31–37, 7 read-only checks)
1. **Data distribution.** val 10 rows, 10–47 cites/q (mean 25.1), types Law=149(59%) / BGE=69(27%) / Numbered=33(13%); train 1139 rows, 1–44 cites/q (mean 4.1), Law=4602(99%) / BGE=57(1%); test 40 rows. Warning issued: val is >2× train so thresholds must not be tuned on train.
2. **KB integrity.** Verifies `laws_knowledge_base.jsonl` and the presence of `citation_lookup.pkl`, `bm25_v2_index.pkl`, `citation_graph.pkl`. Reports `index/statute_to_bge.parquet` MISSING.
3. **Gold format audit.** Of val Art-citations: Abs-level 121 (81%), bare Art 28 (19%). Sub-article matching is critical.
4. **Procedural-boilerplate census.** Only one procedural cite (`Art. 100 Abs. 1 BGG`) recurs in ≥2 val gold sets; never appears in query text.
5. **BGE reachability.** Unique val-gold BGE stems 54; all 54 present in `court_considerations.csv`.
6. **Test vs val domain overlap.** Test query law codes include `CC, CO, MSchG, PrHG, SVG, UVG` not covered by val; val-tuned thresholds may not generalize.
7. **Citation graph edge inventory.** 0 statute→BGE, 0 statute→numbered, 0 BGE→BGE, 0 BGE→statute, 3,897,617 "other". Conclusion: graph cannot recall BGEs.

### Stage 1 — Query analysis (cells 56–63)
- `run_stage1(split=SPLIT, backend=BACKEND)` calls `run_batch_stage1`. Loads domain knowledge (1,125 law abbreviations, 2,114 EN→DE term mappings, 69,837 DE keywords, 171,654 article texts and citations), loads combined BM25 (171,654 law + 1,985,178 court = 2,156,832 docs, 822,251 terms, 212,228,366 nnz; build 35.4 s).
- Per query: explicit-citation regex extraction, law-domain routing, legal-issue identification, candidate articles/cases preview (10/10 each), 5 DE search queries, 2 DE HyDE documents (~68–124 tokens each).
- Qwen3-32B loaded once, batched: first 5 queries 277.2 s (55.4 s/query, model load), next 5 queries 77.9 s (15.6 s/query).
- Diagnostics: artifact summary; explicit-extraction fidelity (6/6 = 100%); candidate hallucination (100/100 valid); law_area routing precision 0.365 / recall 0.955; HyDE drift 3/10 suspicious (val_004, val_007, val_008); free-recall ceiling 126/251 = 50.2% (Law 55.0%, BGE 47.8%, Numbered 33.3%).

### Stage 2 — Multi-Agent Sparse retrieval (cells 69–76)
- `run_stage2(split, backend, max_iterations=4)` reloads BM25 index and Qwen3-32B; processes 10 queries in batches of 3 (final batch of 1).
- Each query: initial BM25 retrieval, then 4 iterations of `planner → agent (REWRITE | SUPPLEMENT | DECOMPOSE | SUPPORTIVE | CROSSREF) → BM25`.
- Iteration 4 of batches 1, 2, 3 hit `[OOM] Batch of 3 failed -- falling back to sequential` for both planner and agent calls.
- Batch wall times: 781.0 s, 668.3 s, 704.2 s, 104.9 s.
- Diagnostics: candidates/q min=973 max=1431 mean=1221.3; type breakdown Numbered=7016(57%) / Law=3590(29%) / BGE=866(7%) / Other=741(6%); per-type recall Law 79.2% / BGE 66.7% / Numbered 75.8%; R@k curves shown; ablation shows raw EN query alone gives Law 39.6% / BGE 65.2% / Numbered 75.8% while single/multi DE queries individually are much weaker (Law 6.7%/18.1%, BGE 2.9%/5.8%); dead query variants DE_2, DE_3, HYDE_0, HYDE_1 flagged; action sources dominated by `bm25_initial=7475` then `mas_supportive=2581`, `mas_crossref=1360`, `mas_rewrite=815`, `mas_decompose=577`, `llm_stage1=200`, `llm_stage1_procedural=200`, `mas_supplement=166`, `bm25_top10=82`, `explicit_from_query=8`; avg overlap fraction 0.08 (actions complementary).

### Stage 3 — Citation graph expansion (cells 81–87)
- `run_stage3(split, gold_prior_mode="off")`. Loads citation graph (103,125 articles with case links, 914,658 cases with article links) and `article_to_abs` (43,066 entries for sibling Abs expansion).
- Per-query expansion deltas (cands → cands_after, +cases, +arts): val_001 973→2117 (+172/+493), val_002 1406→3210 (+485/+377), val_003 976→2271 (+227/+447), val_004 1143→2883 (+409/+478), val_005 1167→2437 (+236/+419), val_006 1431→4189 (+859/+540), val_007 1171→3157 (+498/+560), val_008 1348→3941 (+621/+577), val_009 1394→3655 (+494/+615), val_010 1204→3113 (+394/+608). Total 12,213 → 30,973 (+18,760).
- Diagnostics: cands/q mean 3097.3, BGE/q mean 623.6, Numbered/q mean 1028.2; recall Law 95.3% / BGE 76.8% / Numbered 75.8%; graph reachability val-gold BGE 0/56 at 1-hop and 2-hop (statute→BGE edges missing); procedural-cite coverage 2/2 = 100%; BGE safety-net via court-text mining yields 0/9 missed BGEs; Abs-level coverage abs_exact_hit 116/121 = 96%, abs_parent_only 4/121 = 3%, abs_missed 1/121 = 1%; Abs-flooding 1353 events across 10 queries.

### Stage 4 — LLM reranker + direct generation (cells 94–100)
- `run_stage4(split, backend, skip_direct_gen=False)`. Loads 220,490 KB citation texts, pre-filters each query (e.g., 2117→2006, 3210→2838, 2271→2066), then runs Qwen3-32B reranker prompts in GPU batches of 3.
- Per-query: tier-3 (best) + tier-2 (ok) selections plus direct-gen citations.
- Examples: val_001 2006 reranked → 1364 selected (T3:407, T2:957) + 20 direct = 2035 total; val_002 2838 → 1673 (T3:429, T2:1244) + 26 = 3008; val_007 2802 → 1424 (T3:269, T2:1155) + 23 = 2888; val_010 2778 → 1388 (T3:381, T2:1007) + 28 = 2905.
- Batch wall times: 4153.4 s, plus subsequent batches (5667.1 s shown for batch 3, 1730.9 s for batch 4).
- Diagnostics: reranked/q min 1246 max 1997 mean 1502.6; direct-gen/q mean 25.0; direct-gen citation validity 99/250 = 39.6% (60% hallucinations listed); tier mix T3 27.6% / T2 72.4% overall; 0/220 gold dropped Stage 3→Stage 4; full-pool recall ceiling Law 95.3% / BGE 76.8% / Numbered 75.8%; gold tier assignment shows Law T2=36 T3=35 passthru=71 absent=7, BGE T2=27 T3=25 passthru=1 absent=16, Numbered T2=11 T3=10 passthru=4 absent=8.

### Stage 4b — Cross-encoder rescoring (cells 102–108)
- `run_stage4b(split, top_n=STAGE4B_TOP_N)`. Loads 220,490 article texts, streams `court_considerations.csv` for 3,564 case citations (3,419 court texts loaded), then loads Qwen3-Reranker-4B and scores the top-500 per query.
- Per-query scoring: 500 candidates each, 10–14 s, with 349–487 having text. Total 5000 citations in 2.7 minutes.
- Diagnostics: score distribution mean 0.124 stdev 0.261 (Law mean 0.159 n=871, BGE mean 0.096 n=1376, Numbered mean 0.127 n=2753); rank-change histogram (vs Stage 4): moved_up_>20 460(94.3%) / moved_up_5-20 11(2.3%) / unchanged_±5 7(1.4%) / moved_down_5-20 5(1.0%) / moved_down_>20 5(1.0%); val-gold rank improvement: 129 (93%) moved up, 10 (7%) moved down, mean delta +541.9.

### Stage 5 — Verification, confidence scoring, thresholding, submission (cells 113–120, 123–125)
- `run_stage5(split, threshold=None)` loads citation verifier, BM25 index for continuous scoring, cached BM25 scores, and reranker scores (5000 citations).
- Verification: total verified 25,911, dropped 3,241.
- Sweeps cross-encoder weight (Strategy C): base no-CE macro-F1 0.5075 @ thresh=0.56; best CE weight 0.14 → macro-F1 0.5837 @ thresh=0.44 (delta +0.0762); 5000 citations got CE boost, 13,067 had reranker subtracted. CE config saved to `checkpoints/ce_config.json`.
- Best threshold tuned on val gold: 0.44, macro-F1 = 0.5837.
- Submission generated with threshold=0.44, average 12.0 citations per query, max 16, zero empty predictions.

### Submission validation (cell 125)
- `submission_val.csv` validated for test schema → `Submission valid: False` (`ERROR: Missing 40 test query IDs`, as expected since this is the val submission).

## Results

### Stage 5 final scoring (verbatim, cell 113)
```
Base (no CE): macro-F1=0.5075 @ thresh=0.56
Best CE weight: 0.14, macro-F1=0.5837 @ thresh=0.44 (delta=+0.0762)

Tuning threshold on val gold...
  Best threshold: 0.44
  Best macro-F1:  0.5837

  Per-query F1 @ threshold=0.44:
    val_001: F1=0.5172 (pred=16, gold=42)
    val_002: F1=0.5600 (pred=14, gold=36)
    val_003: F1=0.4127 (pred=16, gold=47)
    val_004: F1=0.6000 (pred=10, gold=10)
    val_005: F1=0.6667 (pred=7, gold=11)
    val_006: F1=0.5517 (pred=11, gold=18)
    val_007: F1=0.6452 (pred=12, gold=19)
    val_008: F1=0.5128 (pred=10, gold=29)
    val_009: F1=0.8148 (pred=13, gold=14)
    val_010: F1=0.5556 (pred=11, gold=25)

  Macro F1: 0.5837
```

### Stage 5 diag 1/6 — global-threshold per-type at 0.44 (cell 115)
```
Global threshold: 0.44000000000000006
Macro-F1: 0.3720
Per-type:
  BGE        P=0.239 R=0.638 F1=0.348  (tp=44 fp=140 fn=25)
  Law        P=0.440 R=0.544 F1=0.486  (tp=81 fp=103 fn=68)
  Numbered   P=0.161 R=0.424 F1=0.233  (tp=14 fp=73 fn=19)
```

### Stage 5 diag 2/6 — LOO-CV per-type thresholds (cell 116)
```
Held-out mean F1 (LOO): 0.4488
Best per-type thresholds seen: {'Law': 0.46, 'BGE': 0.46, 'Numbered': 0.46, 'Other': 0.15}
wrote /content/drive/MyDrive/swiss_law/submissions/submission_val_pertype.csv
```

### Stage 5 diag 3/6 — query-level breakdown at threshold=0.44 (cell 117)
```
qid         pred  gold   TP      P      R     F1
val_001       54    42   26 0.481  0.619  0.542
val_002       55    36   21 0.382  0.583  0.462
val_003       56    47   22 0.393  0.468  0.427
val_004       32    10    4 0.125  0.400  0.190
val_005       29    11    6 0.207  0.545  0.300
val_006       39    18    9 0.231  0.500  0.316
val_007       48    19   11 0.229  0.579  0.328
val_008       50    29   16 0.320  0.552  0.405
val_009       59    14   11 0.186  0.786  0.301
val_010       33    25   13 0.394  0.520  0.448
```

### Stage 5 diag 4/6 — error analysis (cell 118)
```
MISSES by type: Law: 68, BGE: 25, Numbered: 19
FALSE POSITIVES by type: BGE: 140, Law: 103, Numbered: 73
```
Sample misses include `Art. 221 Abs. 2 StPO`, `Art. 100 Abs. 1 BGG`, BGE 148 V 21 E. 5.3, 1B_357/2022 E. 3.1, etc.

### Stage 5 diag 5/6 — Abs flooding in predictions (cell 119)
```
Abs-flooding events (same Art ≥3 Abs variants): 6 across 5 queries
  val_001: Art. 227 StPO → [Abs.1, Abs.7, Abs.4]
  val_002: Art. 8a IVG  → [Abs.1, Abs.2, Abs.4]
  val_003: Art. 227 StPO → [Abs.7, Abs.4, Abs.1]
  val_003: Art. 226 StPO → [Abs.1, Abs.5, Abs.2]
  val_006: Art. 257f OR → [Abs.3, Abs.1, Abs.4]
  val_009: Art. 285a ZGB → [Abs.2, Abs.1, Abs.3]
```

### Stage 5 diag 6/6 — procedural prior potential (cell 120)
```
Gold procedural cites total: 2
Catchable by 'always add' prior: +2 true positives
But introduces: 145 false positives
→ a SELECTIVE prior (domain-conditioned) is required
```

### Omni-rollup recall by stage (cell 127)
```
stage                  Law            BGE        Numbered
Stage1            82/149=55.0%    33/69=47.8%   11/33=33.3%
Stage2           118/149=79.2%    46/69=66.7%   25/33=75.8%
Stage3           142/149=95.3%    53/69=76.8%   25/33=75.8%
Stage4           142/149=95.3%    53/69=76.8%   25/33=75.8%
Stage4b          104/149=69.8%    51/69=73.9%   20/33=60.6%  (top-500 coverage only)
Stage5@0.44       81/149=54.4%    44/69=63.8%   14/33=42.4%
Pool drops (Stages 1-4, >5pp): none
```

### Stage 4 direct-gen hallucination examples (cell 97)
```
direct-gen validity: 99/250 = 39.6%
  HALLUC val_001: Art. 10 BV, Art. 221 Abs. 1 lit. a StPO, Art. 221 Abs. 3 StPO,
                  Art. 223 StPO, Art. 97 BGG, Art. 106 BGG, Art. 423 StPO,
                  BGE 148 IV 121 E. 3.2, BGE 146 IV 322 E. 4.1, BGE 139 IV 145 E. 3,
                  1B_141/2021 E. 3.3, 1B_145/2023 E. 2.2
  HALLUC val_002: Art. 83 Abs. 1 BGG, Art. 90 Abs. 2 BGG, Art. 97 BGG ...
```

### Submission validation (cell 125)
```
Validating: /content/drive/MyDrive/swiss_law/submissions/submission_val.csv
  ERROR: Missing 40 test query IDs
Submission valid: False
```

### Pre-run diagnostic warnings (cells 31–37, verbatim highlights)
- `** WARNING: val (25.1/q) has >2× train (4.1/q) — NEVER tune thresholds on train`
- `graph (statute→BGE): MISSING (index/statute_to_bge.parquet)`
- `81% of Art cites are sub-article — Abs-level matching is critical`
- `Procedural cites appearing in ≥2 val queries: 1`
- `** Codes in test but NOT covered by val: ['CC', 'CO', 'MSchG', 'PrHG', 'SVG', 'UVG']`
- `stat→BGE: 0 <-- KEY ... ** NO statute→BGE edges. BGE recall via graph expansion is impossible.`

### Runtime notes
- Stage 2 OOM events in iter 4 of batches 1–3 (batched planner+agent calls fell back to sequential).
- Stage 1 batch 1 (first 5 queries) included Qwen3-32B model download/load: 277.2 s; remaining batches ~78 s for 5 queries.

## Summary

This notebook is the notebook-clean, source-inlined version of the BM25-first / MAS Swiss-law citation retrieval pipeline (Stages 0 diagnostics → 1 query analysis → 2 multi-agent BM25 → 3 citation-graph expansion → 4 Qwen3-32B reranker + direct-gen → 4b Qwen3-Reranker-4B cross-encoder rescoring → 5 verification, CE-weighted confidence sweep, threshold tuning, and submission) executed end-to-end on the 10-query val split. The best configuration (CE weight 0.14, global threshold 0.44) yields macro-F1 = 0.5837 (per-query F1 ranges 0.4127–0.8148, mean prediction 12 cites/q). Recall climbs from Law/BGE/Numbered = 55/48/33% (Stage 1 ceiling) to 95/77/76% by Stage 3, and is preserved through Stage 4; the precision-oriented Stage 5 threshold trims final prediction recall to 54/64/42%. Key failure modes surfaced: zero statute→BGE edges in the citation graph (the major BGE recall gap), 60% Stage-4 direct-gen hallucination rate, Abs-level flooding in predictions, val/test domain mismatch on six law codes, and persistent BGE false positives (140) dominating the precision loss. The notebook itself does not exit cleanly as a test submission (validates only for val_*, missing 40 test IDs by design).
