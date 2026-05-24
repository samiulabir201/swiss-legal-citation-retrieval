# early_dense_embedding_test

**Path:** `e:\swiss_citation_extraction\notebooks\12_early_funnel_pre_v7\early_dense_embedding_test.ipynb`

## Configuration

- **Model:** `Qwen/Qwen3-Embedding-8B` (Apache 2.0; #1 on MMTEB Multilingual leaderboard, June 2025, score 70.58).
- **Embedding dim:** 4096.
- **Precision:** bfloat16; attention implementation `sdpa` (flash-attn-2 install cancelled/failed, fell back to sdpa).
- **`torch.compile`** enabled (`mode='reduce-overhead'`, `fullgraph=False`).
- **Encoding params:** `BATCH=256`, `CHUNK=100_000`, `MAX_SEQ=768`, `padding_side='left'`, length-sorted batching, `normalize_embeddings=True`.
- **Query prompt prefix (Qwen-style instruct):** "Instruct: Given an English-language legal question or scenario about Swiss federal law, retrieve the Swiss statute articles or federal court decision considerations that are most directly relevant to answering it. The passage you retrieve will contain a citation header, structural metadata, the citations it references, and its German legal text.\nQuery: ".
- **Passage prefix:** none (Qwen3-Embedding does NOT prefix passages).
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition, 102.0 GB VRAM, CUDA 13.0, driver 580.82.07.
- **Seed:** 42.
- **Retrieval index:** FAISS `IndexFlatIP` on CPU, top-1000.
- **k_list:** `(10, 20, 30, 50, 100, 500, 1000)`.
- **Storage:** chunked `.npy` checkpoints under `/content/drive/MyDrive/swiss_law/artifacts/`, tag `qwen3_8b_enriched`.

## Data

- **Val queries:** `data/val.csv` — 10 queries (`val_001`–`val_010`).
- **Corpus CSVs:**
  - `data/laws_de.csv` (`source='laws'`, columns `citation`, `text`, `title`).
  - `data/court_considerations.csv` (`source='court'`, `title=''`).
- **`data_insights/` enrichment files:**
  - `gold_citation_coverage.csv` — 222/222 val golds present in source.
  - `laws_de_classified_citations.jsonl` — 197,945 entries.
  - `court_considerations_classified_citations.jsonl` — 2,416,056 entries (combined classified: 2,592,422).
  - `laws_de_links.json` — 31,217 source nodes.
  - `court_considerations_links.json` — 1,125,616 source nodes (combined: 1,156,833).
- **Corpus after dedup:** 2,161,111 rows (`court`: 1,985,178; `laws`: 175,933).
- **Enriched passage layout:** `[CITATION] cit — title` / `[STRUCTURE] family; subfamily; pattern; segments` / `[REFERENCES] up to 10 outgoing edges` / `[TEXT] first 1600 chars`. Mean enriched length: 957 chars. Build time: 18.3s.
- **Val gold counts per query:** val_001=42, val_002=36, val_003=47, val_004=10, val_005=11, val_006=18, val_007=19, val_008=29, val_009=14, val_010=25 (none dropped as unretrievable).
- **Doc embeddings:** loaded from 22 cached chunked `.npy` files, shape `(2161111, 4096)`, 35.4 GB.
- **Query embeddings:** shape `(10, 4096)`.

## Pipeline

1. **Mount Drive**, print GPU info, set seed.
2. **Install deps** — uninstall numpy, install `numpy==1.26.4 sentence-transformers transformers accelerate faiss-cpu pandas ijson`; attempt `flash-attn` (failed — falls back to sdpa).
3. **Verify Drive paths** for all 8 input files (all OK).
4. **Load base CSVs** — `val.csv`, `laws_de.csv`, `court_considerations.csv`; concatenate to single corpus, dedupe on `citation`.
5. **Load `data_insights/`** — classified-citations JSONL files into a single dict keyed by citation; both link JSONs into a single `source -> [references]` dict.
6. **Validate val gold** against `gold_citation_coverage.csv`; build `val_golds = {qid: [gold cits in corpus]}`.
7. **Build enriched passages** for every corpus row via `build_enriched_passage` (`format_segments` flattens article/units/law_code/volume/series/page/consideration/docket/year/legal_area); max 10 outgoing refs; max 1600 text chars.
8. **Free auxiliary dicts** (`classified`, `src2refs`, ...).
9. **Define evaluation** — `f1(pred,gold)`, `evaluate(...)` builds CPU FAISS `IndexFlatIP`, batched search of `max_k=1000`, reports per-query `F1_oracle_k`, `F1_k20`, `F1_k30`, and `R@k` for each `k`.
10. **Define `encode_chunked`** — sort-by-length batching, restore order, per-chunk `.npy` checkpointing.
11. **Load Qwen3-Embedding-8B** (bf16, sdpa, left-padding) via SentenceTransformers; set `max_seq_length=768`; enable `torch.compile`.
12. **Reuse cached embeddings** — 22 chunk files found, concatenated via `mmap_mode='r'`; no re-encoding.
13. **Encode val queries** with the instruct prefix.
14. **Evaluate** against full corpus; write `results_qwen3_8b_enriched.csv`.
15. **Print verdict** banner with target window 0.60–0.80.
16. **Per-query diagnostics** — sort by F1_oracle_k with 140-char query preview.

## Results

```
=== Qwen/Qwen3-Embedding-8B | enriched passages | full corpus (2,161,111 docs) ===
    qid  n_gold  F1_oracle_k  F1_k20  F1_k30  R@10  R@20  R@30  R@50  R@100  R@500  R@1000
val_001      42        0.000   0.000   0.000 0.000 0.000 0.000 0.000  0.000  0.048   0.167
val_002      36        0.000   0.000   0.000 0.000 0.000 0.000 0.000  0.028  0.028   0.083
val_003      47        0.000   0.000   0.000 0.000 0.000 0.000 0.000  0.000  0.064   0.106
val_004      10        0.000   0.067   0.050 0.000 0.100 0.100 0.300  0.400  0.600   0.600
val_005      11        0.091   0.065   0.049 0.091 0.091 0.091 0.091  0.091  0.364   0.364
val_006      18        0.111   0.105   0.083 0.111 0.111 0.111 0.222  0.222  0.500   0.611
val_007      19        0.105   0.103   0.082 0.053 0.105 0.105 0.105  0.105  0.211   0.263
val_008      29        0.034   0.041   0.034 0.034 0.034 0.034 0.069  0.103  0.103   0.103
val_009      14        0.071   0.059   0.045 0.071 0.071 0.071 0.071  0.143  0.357   0.357
val_010      25        0.000   0.000   0.000 0.000 0.000 0.000 0.040  0.080  0.200   0.240

  Macro F1 oracle-k : 0.041
  Macro F1 k=20     : 0.044
  Macro F1 k=30     : 0.034
  Mean Recall@10   : 0.036
  Mean Recall@50   : 0.090
  Mean Recall@100  : 0.117
  Mean Recall@500  : 0.247
  Mean Recall@1000 : 0.289
```

**Final verdict (notebook output):**

```
FINAL VERDICT — Dense embedding maximum-context ceiling on val (n=10), full corpus
  Model              : Qwen/Qwen3-Embedding-8B
  Passages enriched  : citation header + structure + outgoing refs + text
  Corpus size        : 2,161,111
  Best Macro F1      : 0.044
  Mean Recall@100    : 0.117
  Mean Recall@500    : 0.247
  Target window      : 0.600 – 0.800
VERDICT  : Dense embedding INSUFFICIENT — Observation 3 CONFIRMED with maximum context.
NEXT STEP: Move to LLM agentic retrieval / structured legal-knowledge pipeline.
           Even SOTA + every available signal cannot bridge the legal-conceptual gap.
```

**Per-query diagnostics (sorted by F1_oracle_k desc):** val_006 (0.111), val_007 (0.105), val_005 (0.091), val_009 (0.071), val_008 (0.034); val_001, val_002, val_003, val_004, val_010 all 0.000. R@500 ranges from 0.048 (val_001) to 0.611 (val_006).

**Side observations:**
- `flash-attn` build cancelled — pipeline runs on `sdpa` attention.
- pip dependency conflicts logged for numpy/pandas downgrades (non-fatal).
- Embeddings reused from prior run — no re-encoding cost on this execution.

## Summary

This notebook tests the upper bound of pure dense retrieval for Swiss legal citation matching using `Qwen/Qwen3-Embedding-8B` against the full 2.16M-document corpus, with every passage enriched by parsed citation segments, outgoing reference lists, and titles from `data_insights/`. Despite the SOTA encoder, bf16 + sdpa + torch.compile, max context, and a properly formatted instruct prompt, the 10-query val set yields a best Macro F1 of 0.044 and Mean Recall@500 of 0.247 — far below the 0.60–0.80 target window. Per-query, only val_004/val_005/val_006/val_009 see meaningful recall (R@500 ≥ 0.357), while val_001/val_002/val_003/val_010 remain near zero through k=100. The notebook explicitly concludes "Observation 3 CONFIRMED with maximum context" and recommends moving to LLM agentic retrieval / structured legal-knowledge pipelines. Results CSV is saved to `artifacts/results_qwen3_8b_enriched.csv`.
