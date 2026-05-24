# exp_A1_bgem3_statutes

**Path:** `e:\swiss_citation_extraction\notebooks\11_early_experiments_exp_A\exp_A1_bgem3_statutes.ipynb`

## Configuration

- Model: `BAAI/bge-m3` via `FlagEmbedding.BGEM3FlagModel`, `use_fp16=True`.
- Dependencies: `FlagEmbedding`, `pandas`, `numpy`, `transformers==4.44.2`.
- Runtime: Colab with GPU `NVIDIA RTX PRO 6000 Blackwell Server Edition`; `torch.cuda.is_available()` asserted.
- Drive root: `/content/drive/MyDrive/swiss_law/data`; artifacts dir `.../artifacts`.
- Corpus encoding: `batch_size=64`, `max_length=512`, `return_dense=True`, sparse/colbert disabled; embeddings cast to `float16` and cached at `artifacts/laws_bgem3.npy`.
- Query encoding: `batch_size=8`, `max_length=2048`, dense only.
- Retrieval: dense cosine via matmul on `float32` corpus matrix; top-k via `np.argpartition` + `argsort` refinement.
- Evaluation ks: `(50, 200, 500, 1000)`.
- Statute filter: citation is treated as statute when it does NOT start with `BGE ` and does NOT match `\d[A-Z]_` or `[A-Z]\d[A-Z]_`.
- Kill criterion: `recall@500 >= 0.40` on statutes for either query form.

## Data

- `val.csv` — 10 queries (assertion `len(val) == 10`).
- `laws_de.csv` — 175,933 statute documents; avg doc length 408 chars; doc string = `citation | title | text`.
- `val_translated_de.pkl` — dict `query_id -> German translation`; assertion that every val `query_id` has a translation.
- Drive listing reported: `['artifacts', 'court_considerations.csv', 'laws_de.csv', 'sample_submission.csv', 'test.csv', 'test_translated_de.pkl', 'train.csv', 'val.csv', 'val_translated_de.pkl']`.
- Two query variants are evaluated: raw English `val.query` and the German translation from the pkl.

## Pipeline

1. Install pinned deps (`transformers==4.44.2`, `FlagEmbedding`, `pandas`, `numpy`) and hard-kill runtime to refresh.
2. Mount Drive; set paths under `swiss_law/data` and ensure `artifacts/` exists.
3. Load `val.csv`, `laws_de.csv`, and `val_translated_de.pkl`; assert val size and translation coverage.
4. Build per-doc strings as `citation | title | text`; keep `citation` list aligned by index.
5. Load BGE-M3 (`use_fp16=True`) on GPU.
6. Encode corpus (cache-aware): if `artifacts/laws_bgem3.npy` exists, load it; else encode 175,933 docs (batch 64, max_len 512) and save as `float16` `(175933, 1024)`.
7. Encode EN and DE val queries → `q_en (10, 1024)`, `q_de (10, 1024)`.
8. Score: compute `sims = q @ doc_emb_f32.T`; top-k via `argpartition` then `argsort` for ordering.
9. For each query, parse `gold_citations` (`;` separator), filter to statute golds via the regex rule, and count hits at each k.
10. Aggregate recall = `sum(hits) / sum(n_gold_stat)` across queries; print and save report.
11. Save `artifacts/exp_A1_report.json` (per-query hits, aggregate recall, meta) and print PASS/FAIL verdict against the 0.40 threshold.

## Results

Cell 2 install output:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
gradio 5.50.0 requires pandas<3.0,>=1.0, but you have pandas 3.0.2 which is incompatible.
```

Drive after mount: `['artifacts', 'court_considerations.csv', 'laws_de.csv', 'sample_submission.csv', 'test.csv', 'test_translated_de.pkl', 'train.csv', 'val.csv', 'val_translated_de.pkl']`.

Corpus stats: `laws: 175933 docs | avg chars=408`.

GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`.

Corpus encoding: `encoded 175933 docs in 70s -> (175933, 1024)` (pre-tokenize 2749 batches @ ~401 it/s; inference 2749 batches @ ~45 it/s, ~60s).

Query encoding shapes: `q_en (10, 1024) q_de (10, 1024)`.

Recall table:

```
=== EN query ===
  stat_recall@50 = 0.074
  stat_recall@200 = 0.121
  stat_recall@500 = 0.215
  stat_recall@1000 = 0.302
=== DE query (translated) ===
  stat_recall@50 = 0.067
  stat_recall@200 = 0.114
  stat_recall@500 = 0.168
  stat_recall@1000 = 0.228
```

Report saved: `/content/drive/MyDrive/swiss_law/data/artifacts/exp_A1_report.json`.

Final verdict: `verdict: best recall@500 statutes = 0.215 (threshold 0.40 -> FAIL -> need HyDE)`.

## Summary

Exp-A1 measures BGE-M3 dense retrieval recall on the 10-query val set against the 175,933-doc Swiss statute corpus (`laws_de.csv`), comparing the raw English query against the German-translated query. Corpus embeddings are computed once (70s on the Blackwell GPU) and cached as float16 to `artifacts/laws_bgem3.npy`; queries are scored by a single dense matmul with top-k via `argpartition`. The English query outperforms the German translation at every k (e.g., `stat_recall@500 = 0.215` EN vs `0.168` DE; `@1000 = 0.302` vs `0.228`), but both fall well below the preregistered 0.40 kill threshold. The notebook explicitly records the FAIL verdict and concludes that BGE-M3 alone is insufficient for statute retrieval, mandating Exp-2 (HyDE) before any work on the court corpus. Persisted artifacts: `artifacts/laws_bgem3.npy` (175933x1024 float16) and `artifacts/exp_A1_report.json`.
