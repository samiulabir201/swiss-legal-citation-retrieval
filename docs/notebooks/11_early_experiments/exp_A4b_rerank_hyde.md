# exp_A4b_rerank_hyde

**Path:** `e:\swiss_citation_extraction\notebooks\11_early_experiments_exp_A\exp_A4b_rerank_hyde.ipynb`

## Configuration

- Reranker: `BAAI/bge-reranker-v2-m3` (fp16, via `FlagEmbedding.FlagReranker`).
- Candidate source: Exp-3 `hyde+enum` top-1000 per query (Qwen3-32B), reused from `rerank_scores_A4.npz`.
- Query side for reranking: HyDE passages from `exp_A3_expansions.json` (field `hyde`); up to 3 passages per query.
- Pair scoring: each HyDE passage × 1000 candidate docs per query (3000 pairs/query), `batch_size=32`, `max_length=1024`, `normalize=True`.
- Aggregation per `(query, doc)`: `mean` and `max` over passages.
- Pip installs: `numpy==1.26.4`, `FlagEmbedding`, `pandas`, `numpy`, `transformers<4.45.0` (runtime restart expected after numpy install).
- Runtime: Google Colab with Drive mount; GPU = NVIDIA RTX PRO 6000 Blackwell Server Edition; CUDA required.
- Doc text construction: `citation | title | text`.
- Statute filter for evaluation: `is_stat(c) = not (c.startswith('BGE ') or matches '\d[A-Z]_' or '[A-Z]\d[A-Z]_')`.
- Win threshold: beat baseline RRF `stat_recall@50` (0.161). Useful threshold: `stat_recall@50 >= 0.30`.

## Data

- `ROOT = /content/drive/MyDrive/swiss_law/data`, `ART = ROOT/artifacts`.
- Required artifacts (asserted present): `laws_bgem3.npy`, `exp_A3_expansions.json`, `rerank_scores_A4.npz`, `exp_A4_report.json`.
- Inputs loaded: `val.csv` (10 queries, ids `val_001`..`val_010`), `laws_de.csv` (citation/title/text), `exp_A3_expansions.json` (per-qid keys: `query_en`, `query_de`, `hyde`, `enum`).
- `rerank_scores_A4.npz` fields used: `cands` (10×1000), `scores_de` (Exp-4 raw-DE rerank scores for comparison), `query_ids` (order matches `val.csv`).
- HyDE passages: each query yielded 3 passages (first-passage lengths 487-574 chars); the `hyde` string is split on `\n\s*\n` and clipped to first 3.
- Output written: `artifacts/rerank_scores_A4b.npz` (`cands`, `scores_hyde_mean`, `scores_hyde_max`, `query_ids`) and `artifacts/exp_A4b_report.json`.

## Pipeline

1. Cell 1 — install pinned `numpy==1.26.4` then `FlagEmbedding pandas numpy` (note: pip dependency-resolver conflicts logged for tobler, rasterio, shap, xarray-einstats, opencv-*, jaxlib, pytensor, jax, cupy-cuda12x, opencv-contrib, opencv; later for google-colab, numba, tensorflow, db-dtypes, gradio, bqplot, dask-cudf-cu12, cudf-cu12).
2. Cell 2 — mount Drive; assert all 4 required artifacts present; assert CUDA.
3. Cell 3 — load `val.csv`, `laws_de.csv`, `exp_A3_expansions.json`; build `docs` (`citation | title | text`); load `cands`, `scores_de` (renamed `scores_de_old`), `query_ids`; assert qid order matches `val`; print expansion keys sample for `val_001`.
4. Cell 4 — auto-detect HyDE field by scanning `['hyde_passages','hyde','hyde_text','hypothetical','passages']`; resolved to `hyde`; split each string on blank-line regex and take first 3 paragraphs; print per-query passage count and first-passage length.
5. Cell 5 — `pip install transformers<4.45.0`; load `FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)` (model files downloaded: tokenizer, sentencepiece.bpe.model 5.07MB, tokenizer.json 17.1MB, config, safetensors 2.27GB).
6. Cell 6 — for each query, for each of ≤3 passages, score 1000 (passage, doc) pairs; stack into `(n_pass, 1000)`; collect per query; aggregate `mean` and `max` over the passage axis; save `rerank_scores_A4b.npz`. Total rerank time: 42 s for 10 queries.
7. Cell 7 — define `eval_at_k` over `ks=(10,20,30,50,100,200,500)`; build four orderings: baseline RRF (identity 0..999), Exp-4 raw-DE rerank (`argsort(-scores_de_old)`), HyDE-mean rerank, HyDE-max rerank; compute `stat_recall@k` per variant.
8. Cell 8 — print per-query `hit@50` for all four variants plus HyDE-mean `hit@30/@20/@10`.
9. Cell 9 — assemble `report['meta']` (model, query_form, candidate_source, n_queries, baselines from Exp-3/4, win and useful thresholds); save `exp_A4b_report.json`; print combined recall table and PASS/FAIL gate decisions.

## Results

Cell 3 (sample expansion keys for `val_001`):

```
expansion keys for val_001 : ['query_en', 'query_de', 'hyde', 'enum']
  query_en: str | 'May a court lawfully order a three‑month extension of pre‑trial detention under Art. 221 Abs. 1 lit. b StPO (risk of col'
  query_de: str | 'Kann ein Gericht eine dreimonatige Verlängerung der Untersuchungshaft gemäß Art. 221 Abs. 1 lit. b StPO (Kollusionsgefah'
  hyde: str | 'Art. 221 Abs. 1 lit. b StPO ist dahin auszulegen, dass eine Verlängerung der Untersuchungshaft um drei Monate nur dann r'
  enum: str | 'Art. 107 Abs. 1 StPO — Pre-trial detention conditions  \nArt. 108 Abs. 1 StPO — Justification for detention  \nArt. 109 Ab'
```

Cell 4 (HyDE passage counts):

```
HyDE field: hyde
val_001: 3 passages, first len=492
val_002: 3 passages, first len=574
val_003: 3 passages, first len=498
val_004: 3 passages, first len=490
val_005: 3 passages, first len=522
val_006: 3 passages, first len=559
val_007: 3 passages, first len=487
val_008: 3 passages, first len=557
val_009: 3 passages, first len=539
val_010: 3 passages, first len=496
```

Cell 6 (rerank timing): 42 s total; per-query cumulative elapsed `[1]=5s, [2]=10s, [3]=12s, [4]=16s, [5]=20s, [6]=25s, [7]=29s, [8]=33s, [9]=38s, [10]=42s`.

Cell 7 — Recall by variant:

```
=== baseline RRF (no rerank) ===
  stat_recall@10 = 0.047
  stat_recall@20 = 0.101
  stat_recall@30 = 0.121
  stat_recall@50 = 0.161
  stat_recall@100 = 0.255
  stat_recall@200 = 0.315
  stat_recall@500 = 0.376
=== rerank with raw DE query (Exp-4 repro) ===
  stat_recall@10 = 0.047
  stat_recall@20 = 0.087
  stat_recall@30 = 0.101
  stat_recall@50 = 0.114
  stat_recall@100 = 0.148
  stat_recall@200 = 0.215
  stat_recall@500 = 0.342
=== rerank with HyDE passages (mean) ===
  stat_recall@10 = 0.087
  stat_recall@20 = 0.114
  stat_recall@30 = 0.134
  stat_recall@50 = 0.174
  stat_recall@100 = 0.242
  stat_recall@200 = 0.315
  stat_recall@500 = 0.369
=== rerank with HyDE passages (max) ===
  stat_recall@10 = 0.074
  stat_recall@20 = 0.114
  stat_recall@30 = 0.134
  stat_recall@50 = 0.174
  stat_recall@100 = 0.242
  stat_recall@200 = 0.302
  stat_recall@500 = 0.369
```

Cell 8 — Per-query `hit@50` plus HyDE-mean `hit@30/@20/@10`:

```
qid        gold | base  deQ hMean hMax | hMean@30 hMean@20 hMean@10
--------------------------------------------------------------------------------
val_001      19 |    6    5     5    5 |        5        5        4
val_002      20 |    3    1     3    2 |        2        1        1
val_003      24 |    3    2     4    4 |        3        3        2
val_004       9 |    2    3     2    2 |        2        2        2
val_005       6 |    2    1     2    2 |        2        1        0
val_006      11 |    4    0     1    1 |        1        1        1
val_007      15 |    1    1     4    4 |        3        2        2
val_008      20 |    0    1     0    0 |        0        0        0
val_009      11 |    1    3     4    5 |        1        1        0
val_010      14 |    2    0     1    1 |        1        1        1
```

Cell 9 — Final summary table and gate decisions:

```
================================================================================
variant                             @10    @20    @30    @50   @100   @200   @500
--------------------------------------------------------------------------------
baseline_rrf                      0.047  0.101  0.121  0.161  0.255  0.315  0.376
rerank_de_query                   0.047  0.087  0.101  0.114  0.148  0.215  0.342
rerank_hyde_mean                  0.087  0.114  0.134  0.174  0.242  0.315  0.369
rerank_hyde_max                   0.074  0.114  0.134  0.174  0.242  0.302  0.369

best HyDE-rerank @50 = 0.174  vs  baseline RRF @50 = 0.161
win threshold (beat baseline) -> PASS
useful threshold (>=0.30)     -> BELOW
```

## Summary

Exp-A4b reranked the Exp-3 `hyde+enum` top-1000 candidates with `BAAI/bge-reranker-v2-m3`, swapping the rerank query side from the raw German question (Exp-A4) to HyDE passages aggregated by mean and max over up to 3 passages per query. Register-matched HyDE rerank cleared the win gate, lifting `stat_recall@50` from baseline RRF 0.161 to 0.174 (mean and max tied) and beating Exp-A4's raw-DE rerank 0.114 across all K. However, baseline RRF still beats both HyDE-rerank variants at `@100` (0.255 vs 0.242) and `@500` (0.376 vs 0.369), and the absolute level remains well below the useful threshold of 0.30. Per-query, gains concentrate on `val_007` (1→4) and `val_009` (1→4/5); `val_001` and `val_006` lose ground vs baseline. Total rerank cost was 42 s on the supplied GPU; outputs were saved to `rerank_scores_A4b.npz` and `exp_A4b_report.json`.
