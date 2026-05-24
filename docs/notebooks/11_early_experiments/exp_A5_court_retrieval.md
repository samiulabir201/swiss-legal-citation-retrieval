# exp_A5_court_retrieval

**Path:** `e:\swiss_citation_extraction\notebooks\11_early_experiments_exp_A\exp_A5_court_retrieval.ipynb`

## Configuration

- Encoder: `BAAI/bge-m3` (fp16, `max_length=512`, `batch_size=64`)
- Inference backend deps: `numpy==1.26.4`, `FlagEmbedding`, `pandas`, `vllm`, `transformers<4.45.0`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (~101.4 GB free VRAM)
- Storage roots:
  - `ROOT = /content/drive/MyDrive/swiss_law/data`
  - `ART  = ROOT/artifacts`
  - `CHUNKS_DIR = ART/court_chunks`
- Required inputs: `court_considerations.csv`, `val.csv`, `laws_de.csv`, `query_vecs_A3.npz`, `exp_A3_expansions.json`
- Final artifacts: `court_bgem3.npy` (~5.07 GB fp16), `court_citations.json`, `court_topk_A5.npz`, `exp_A5_report.json`
- Encoding chunking: `CHUNK_SIZE=100_000` rows/chunk; one `.npy` per chunk for resumability
- Retrieval: `TOPK=500`, GPU chunked matmul with `chunk=200_000` docs/block, fp16
- Fusion: Reciprocal Rank Fusion with `k_rrf=60` over `q_de`, `q_hyde`, `q_enum`
- Mining window: `MINE_K=200` top cases per query
- Gates:
  - `case_recall@200 >= 0.30`
  - `mined_stat_recall@200 >= 0.20`
  - `combined_stat_recall >= 0.45`

## Data

- Corpus: `court_considerations.csv` — 2,476,315 rows, unified DE/FR/IT (no per-language filter)
- Val set: 10 queries (`val.csv`)
- Statute lookup: `laws_de.csv` (175,933 rows; 9,594 unique `Art. N ABBR` prefixes)
- Query vectors from Exp-3: `q_en`, `q_de`, `q_hyde`, `q_enum`, each shape `(10, 1024)`
- Classification of gold citations:
  - `bge` if starts with `BGE `
  - `docket` if matches `^\d[A-Z]?_\d+/\d{4}`
  - else `statute`
- Trilingual mining regexes:
  - DE: `STATUTE_DE` — matches `Art. N [Abs./lit./Ziff.] {ZGB|OR|StGB|StPO|ZPO|BGG|BV|URG|MSchG|DBG|MWSTG|SchKG|AHVG|IVG|UVG|KVG|BGÖ|DSG|AIG|AuG|AsylG|VZV|SVG|VStrR|VStG|MStG|TSchG|RPG|FINIG|FINFRAG}`
  - FR: `STATUTE_FR` — matches `art. N [al./let.] {CO|CC|CP|CPP|CPC|LTF|Cst|LDIP|LP}`
  - IT: `STATUTE_IT` — matches `art. N [cpv./lett.] {CO|CC|CP|CPP|CPC|LTF|Cost|LDIP|LP}`
- Abbreviation normalisation maps (FR/IT → DE):
  - `FR2DE = {CO→OR, CC→ZGB, CP→StGB, CPP→StPO, CPC→ZPO, LTF→BGG, CST→BV, LDIP→IPRG, LP→SchKG}`
  - `IT2DE = FR2DE | {COST→BV}`

## Pipeline

1. **Install deps and mount Drive.** Pin numpy 1.26.4, install `FlagEmbedding`, `pandas`, `vllm`, and `transformers<4.45.0` (with multiple known dependency conflicts logged for tobler/rasterio/shap/xarray-einstats/opencv/jax/cupy/pytensor/gradio/db-dtypes/bqplot/dask-cudf/cudf/google-colab).
2. **Load val and Exp-3 query vectors.** Sanity-check `query_ids` order vs `val['query_id']`. Build `laws_cits` set and `laws_by_prefix` defaultdict from `laws_de.citation` using `PREFIX_PAT = ^(Art\.?\s*\d+[a-z]?\s+[A-ZÄÖÜ]+\b)`.
3. **Load BGE-M3 (`use_fp16=True`).** Skip if final embedding `court_bgem3.npy` and `court_citations.json` already cached.
4. **Encode `court_considerations.csv` in resumable chunks.** Stream CSV at `chunksize=100_000`, encode `df['text']` with `return_dense=True, return_sparse=False, return_colbert_vecs=False`, cast to fp16, save `court_chunks/court_NNNN.npy`. Skip chunk if file exists. Collect all citations in order into `court_citations.json`.
5. **Concatenate chunks → `court_bgem3.npy`** (`np.concatenate` on 25 chunks; ~5.07 GB fp16).
6. **Free model weights**, mmap-load `court_emb` and read `court_cits`.
7. **Top-K retrieval over court corpus.** `topk_court_gpu` performs chunked GPU matmul, maintaining a running top-K via concat+topk merge per 200K-doc block. Initialise scores with `-60000.0` (fp16-safe; comment notes the fix vs `-1e9`). Run for each of `q_de`, `q_hyde`, `q_enum`.
8. **RRF fusion** of the 3 query forms with `k_rrf=60`, producing `idx_rrf` shape `(10, 500)`.
9. **Case-recall evaluation.** `case_match` handles BGE prefix matching and docket-prefix matching (`DOCKET_RE = ^([1-9][A-Z]?_\d+/\d{4})`). `eval_case_recall` reports micro `case_recall@{50,100,200,500}` over BGE+docket gold.
10. **A5b — Statute mining from retrieved case text.** Re-stream the corpus once, collecting text only for the union of top-200 indices across all 10 queries (1,934 unique rows). For each query, walk top-K cases in rank order, apply `mine(text)` (DE+FR+IT regex with FR/IT abbreviation normalisation), and expand each mined `Art. N ABBR` prefix to all matching `laws_de` full citations via `laws_by_prefix` (also include the bare prefix if present in `laws_cits`).
11. **Mined-statute recall** vs the statute-only gold for top-50/100/200 case windows.
12. **Combined recall** via union of Exp-A4 statute candidates (`rerank_scores_A4.npz` → `cands`) at K_stat ∈ {50, 100, 200, 500} with mined statutes from top-200 cases.
13. **Per-query breakdown** + final artifact saves (`exp_A5_report.json`, `court_topk_A5.npz`) and gate verdicts.

## Results

Cell 2 GPU check:
```
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
Free VRAM (GB): 101.4
```

Cell 3 loads:
```
val: 10 | laws: 175933 | unique law prefixes: 9594
q_hyde: (10, 1024) | q_enum: (10, 1024) | q_de: (10, 1024)
```

Encoding (Cell 5, 25 chunks):
```
chunk 0000: 100000 rows | elapsed 2.2 min | pace 748 docs/sec
chunk 0001: 100000 rows | elapsed 3.7 min | pace 903 docs/sec
chunk 0002: 100000 rows | elapsed 5.1 min | pace 973 docs/sec
chunk 0003: 100000 rows | elapsed 6.6 min | pace 1013 docs/sec
chunk 0004: 100000 rows | elapsed 8.1 min | pace 1033 docs/sec
chunk 0005: 100000 rows | elapsed 9.6 min | pace 1045 docs/sec
chunk 0006: 100000 rows | elapsed 11.0 min | pace 1058 docs/sec
chunk 0007: 100000 rows | elapsed 12.5 min | pace 1066 docs/sec
chunk 0008: 100000 rows | elapsed 14.0 min | pace 1073 docs/sec
chunk 0009: 100000 rows | elapsed 15.5 min | pace 1078 docs/sec
chunk 0010: 100000 rows | elapsed 17.0 min | pace 1081 docs/sec
chunk 0011: 100000 rows | elapsed 18.4 min | pace 1086 docs/sec
chunk 0012: 100000 rows | elapsed 19.9 min | pace 1089 docs/sec
chunk 0013: 100000 rows | elapsed 21.4 min | pace 1088 docs/sec
chunk 0014: 100000 rows | elapsed 23.2 min | pace 1077 docs/sec
chunk 0015: 100000 rows | elapsed 24.8 min | pace 1073 docs/sec
chunk 0016: 100000 rows | elapsed 26.5 min | pace 1068 docs/sec
chunk 0017: 100000 rows | elapsed 28.3 min | pace 1061 docs/sec
chunk 0018: 100000 rows | elapsed 30.0 min | pace 1057 docs/sec
chunk 0019: 100000 rows | elapsed 31.6 min | pace 1055 docs/sec
chunk 0020: 100000 rows | elapsed 33.3 min | pace 1052 docs/sec
chunk 0021: 100000 rows | elapsed 34.9 min | pace 1050 docs/sec
chunk 0022: 100000 rows | elapsed 36.6 min | pace 1049 docs/sec
chunk 0023: 100000 rows | elapsed 38.3 min | pace 1045 docs/sec
chunk 0024: 76315 rows | elapsed 39.5 min | pace 1056 docs/sec
done all chunks in 39.5 min | 2476315 citations saved
```

Concatenation (Cell 6):
```
concatenating 25 chunks…
concatenated: (2476315, 1024) float16 (5.07 GB)
saved: /content/drive/MyDrive/swiss_law/data/artifacts/court_bgem3.npy
```

Retrieval (Cell 8): `q_de` 1s, `q_hyde` 0s, `q_enum` 0s. RRF top-K shape `(10, 500)`.

Case recall (Cell 10):
```
=== q_de ===
  case_recall@50 = 0.010
  case_recall@100 = 0.020
  case_recall@200 = 0.049
  case_recall@500 = 0.147
=== q_hyde ===
  case_recall@50 = 0.059
  case_recall@100 = 0.069
  case_recall@200 = 0.098
  case_recall@500 = 0.176
=== q_enum ===
  case_recall@50 = 0.029
  case_recall@100 = 0.049
  case_recall@200 = 0.059
  case_recall@500 = 0.108
=== RRF(de,hyde,enum) ===
  case_recall@50 = 0.049
  case_recall@100 = 0.059
  case_recall@200 = 0.108
  case_recall@500 = 0.157
```

Mining text collection (Cell 11):
```
reading text for 1934 unique case rows…
collected text for 1934 rows
mined sizes (q0): 0 4 7
```

Mined-statute recall (Cell 12):
```
mine over top-50 cases: mined_stat_recall = 0.047  (7/149)
mine over top-100 cases: mined_stat_recall = 0.047  (7/149)
mine over top-200 cases: mined_stat_recall = 0.060  (9/149)
combined stat_recall(stat_top@50 ∪ mined@200_cases) = 0.215
combined stat_recall(stat_top@100 ∪ mined@200_cases) = 0.302
combined stat_recall(stat_top@200 ∪ mined@200_cases) = 0.349
combined stat_recall(stat_top@500 ∪ mined@200_cases) = 0.409
```

Per-query breakdown (Cell 13):
```
qid        caseG  case@50 case@200 | statG mineN mineHit
----------------------------------------------------------------------
val_001       23        0        2 |    19     7       0
val_002       16        0        1 |    20     2       0
val_003       23        0        0 |    24     6       0
val_004        1        0        1 |     9    18       3
val_005        5        2        2 |     6    12       0
val_006        7        1        2 |    11    16       0
val_007        4        0        0 |    15    27       2
val_008        9        0        0 |    20    14       2
val_009        3        0        0 |    11    14       1
val_010       11        2        3 |    14     8       1
```

Final summary and gate verdict (Cell 14):
```
======================================================================
CASE RECALL  (best: RRF)
  case_recall@50 = 0.049
  case_recall@100 = 0.059
  case_recall@200 = 0.108
  case_recall@500 = 0.157
STATUTE MINING (top-200 cases)
  mined_stat_recall = 0.060
COMBINED STAT (Exp-3 candidates ∪ mined)
  combined@stat_top50 ∪ mined@200 = 0.215
  combined@stat_top100 ∪ mined@200 = 0.302
  combined@stat_top200 ∪ mined@200 = 0.349
  combined@stat_top500 ∪ mined@200 = 0.409

gate case_recall@200 ≥ 0.30  -> FAIL
gate mined_stat_recall ≥ 0.20 -> FAIL
gate combined_stat ≥ 0.45    -> FAIL
```

## Summary

Exp-A5 encoded the full 2.47M-row `court_considerations.csv` corpus with BGE-M3 (fp16, 1024-d, 5.07 GB) on an RTX PRO 6000 in ~39.5 minutes via 25 resumable 100K-row chunks, then ran top-500 retrieval on the val set with `q_de`, `q_hyde`, `q_enum`, and their RRF fusion. The best case-recall configuration was `q_hyde` alone (`case_recall@200 = 0.098`, `@500 = 0.176`); RRF gave `0.108 @200` and `0.157 @500`. A5b mined statute citations from the top-200 retrieved case texts via a trilingual DE/FR/IT regex with FR/IT-to-DE abbreviation normalisation, yielding `mined_stat_recall = 0.060` (9/149). Combining mined statutes with Exp-3/A4 statute candidates raised statute recall to 0.409 at `stat_top500 ∪ mined@200`. All three configured gates failed (case_recall@200 0.108 vs 0.30; mined_stat_recall 0.060 vs 0.20; combined 0.409 vs 0.45).
