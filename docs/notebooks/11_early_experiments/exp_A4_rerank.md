# exp_A4_rerank

**Path:** `e:\swiss_citation_extraction\notebooks\11_early_experiments_exp_A\exp_A4_rerank.ipynb`

## Configuration

- Reranker: `BAAI/bge-reranker-v2-m3` (multilingual, fp16, batch_size=32, max_length=1024, normalize=True).
- Candidate source: hyde+enum top-1000 per query, reconstructed via RRF (k_rrf=60) from `query_vecs_A3.npz` ranks of `q_hyde` and `q_enum` against `laws_bgem3.npy` (175,933 docs x 1024 dim).
- Two query forms reranked: translated German (primary; `val_translated_de.pkl`) and raw English (`exp_A3_expansions.json[qid]['query_en']`).
- Doc text fed to reranker: `citation | title | text` (same concatenation as retriever index).
- Hardware: GPU required; observed device `NVIDIA RTX PRO 6000 Blackwell Server Edition`.
- Dependencies: `numpy==1.26.4`, `FlagEmbedding`, `pandas`, `transformers<4.45.0`.
- Evaluation: statute-only gold (filters out `BGE ...`, `\d[A-Z]_...`, `[A-Z]\d[A-Z]_...` patterns), recall at K in {10, 20, 30, 50, 100, 200, 500}.
- Useful floor stated: `stat_recall@50 >= 0.35`.

## Data

- `ROOT = /content/drive/MyDrive/swiss_law/data`
- `ART  = ROOT/artifacts`
- Inputs (asserted present):
  - `artifacts/laws_bgem3.npy` (doc embeddings, shape (175933, 1024))
  - `artifacts/query_vecs_A3.npz` (keys: `q_en`, `q_de`, `q_hyde`, `q_enum`, `query_ids`)
  - `artifacts/exp_A3_expansions.json` (Qwen3-32B expansions; provides `query_en`)
  - `val.csv` (10 rows; query_id order asserted == query_vecs order)
  - `laws_de.csv` (citation, title, text columns)
  - `val_translated_de.pkl` (dict qid -> translated DE query)
- Outputs written:
  - `artifacts/rerank_scores_A4.npz` (keys: `cands`, `scores_de`, `scores_en`, `query_ids`)
  - `artifacts/exp_A4_report.json` (aggregated + per-query metrics + meta)

## Pipeline

1. Cell 1: install `numpy==1.26.4`, `FlagEmbedding`, `pandas`, `numpy` (restart note).
2. Cell 2: mount Drive, assert required artifacts exist, check CUDA.
3. Cell 3: load `val.csv`, `laws_de.csv`, `val_translated_de.pkl`, `exp_A3_expansions.json`; build `docs` as `citation | title | text`; load `doc_emb` and `query_vecs_A3.npz`; assert query_id alignment.
4. Cell 4: `rank_all` computes full descending similarity ranks for `q_hyde` and `q_enum`; `rrf(...)` fuses them (k_rrf=60) to produce `cands` of shape (10, 1000).
5. Cell 5: install `transformers<4.45.0`; load `FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)`.
6. Cell 6: `rerank_for_queries(...)` scores each (query, doc) pair for all 1000 candidates; runs once for DE queries (~34s total) and once for EN queries (~30s total); saves `rerank_scores_A4.npz`.
7. Cell 7: `eval_at_k` computes statute-only `recall@K`; baseline order is identity over RRF candidates; rerank orders are `argsort(-scores_*)`. Prints aggregated recall for baseline, rerank_de, rerank_en.
8. Cell 8: per-query breakdown table at @50/@30/@20/@10.
9. Cell 9: attach `meta` (reranker name, candidate source, prior-exp baselines, floor); write `exp_A4_report.json`; print summary table and pass/fail vs floor 0.35.

## Results

GPU detected:

```
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
```

Corpus and candidate shapes:

```
corpus: (175933, 1024) | val: 10
candidates: (10, 1000)
```

Rerank timing: DE total 34s; EN total 30s (10 queries x 1000 pairs each, batch_size=32).

Aggregated statute recall:

```
=== baseline (hyde+enum RRF, no rerank) ===
  stat_recall@10 = 0.047
  stat_recall@20 = 0.101
  stat_recall@30 = 0.121
  stat_recall@50 = 0.161
  stat_recall@100 = 0.255
  stat_recall@200 = 0.315
  stat_recall@500 = 0.376
=== rerank with DE query ===
  stat_recall@10 = 0.047
  stat_recall@20 = 0.087
  stat_recall@30 = 0.101
  stat_recall@50 = 0.114
  stat_recall@100 = 0.148
  stat_recall@200 = 0.215
  stat_recall@500 = 0.342
=== rerank with EN query ===
  stat_recall@10 = 0.027
  stat_recall@20 = 0.040
  stat_recall@30 = 0.060
  stat_recall@50 = 0.067
  stat_recall@100 = 0.094
  stat_recall@200 = 0.148
  stat_recall@500 = 0.336
```

Per-query breakdown:

```
qid        gold |  base@50  de@50  en@50 |  de@30  de@20  de@10
----------------------------------------------------------------------
val_001      19 |        6      5      4 |      5      5      4
val_002      20 |        3      1      0 |      0      0      0
val_003      24 |        3      2      0 |      2      1      0
val_004       9 |        2      3      3 |      3      3      1
val_005       6 |        2      1      1 |      1      1      0
val_006      11 |        4      0      0 |      0      0      0
val_007      15 |        1      1      0 |      1      0      0
val_008      20 |        0      1      1 |      1      1      1
val_009      11 |        1      3      1 |      2      2      1
val_010      14 |        2      0      0 |      0      0      0
```

Final summary and verdict:

```
==========================================================================
variant                                     @10    @20    @30    @50   @100   @200
--------------------------------------------------------------------------
baseline_rrf                              0.047  0.101  0.121  0.161  0.255  0.315
rerank_de                                 0.047  0.087  0.101  0.114  0.148  0.215
rerank_en                                 0.027  0.040  0.060  0.067  0.094  0.148

best rerank stat_recall@50 = 0.114
(pre-rerank @50 = 0.161)
floor 0.35 -> STILL BELOW
```

Notable pip dependency-conflict warnings emitted during install (e.g., `numba 0.60.0 requires numpy<2.1`, `tensorflow 2.19.0 requires numpy<2.2.0`, `google-colab requires pandas==2.2.2`); environment proceeded without restart in the recorded run.

## Summary

The notebook reranks Exp-3's hyde+enum RRF top-1000 candidates with `BAAI/bge-reranker-v2-m3` using both translated-DE and raw-EN query forms on the 10-query val set. Reranking degrades statute recall across all K compared to the unranked RRF baseline: best `stat_recall@50` after rerank is 0.114 (DE), versus 0.161 for the baseline, and far below the stated 0.35 floor; `recall@500` also drops (0.342 DE / 0.336 EN vs 0.376 baseline). DE queries consistently beat EN queries, matching the document language, but neither surfaces gold into actionable top-K. Per-query results show val_006 and val_010 lose all hits at @50 after rerank, while val_008 and val_009 gain modestly. The verdict line confirms `STILL BELOW`, indicating the bge-reranker does not earn its keep on this candidate set and the gold density in the top-1000 is insufficient.
