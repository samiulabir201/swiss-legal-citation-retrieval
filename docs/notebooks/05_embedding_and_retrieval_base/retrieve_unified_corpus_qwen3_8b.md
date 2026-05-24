# retrieve_unified_corpus_qwen3_8b

**Path:** `e:\swiss_citation_extraction\notebooks\05_embedding_and_retrieval_base\retrieve_unified_corpus_qwen3_8b.ipynb`

## Configuration

- **Runtime:** Colab Blackwell instance. GPU detected at run time: NVIDIA RTX PRO 6000 Blackwell Server Edition, 102.0 GB VRAM, sm_120.
- **Drive paths:** `ROOT = /content/drive/MyDrive/swiss_law`, `DATA_DIR = ROOT/data`, `EMB_DIR = ROOT/artifacts/embeddings`, `META_ZIP = ROOT/artifacts/retrieval_metadata.zip`, `META_DIR = /content/metadata`, `OUT_DIR = ROOT/artifacts/eval`.
- **Eval split:** `SPLIT = 'val'`, `TOP_K = 10000`, `QUERY_LIMIT = 0` (no cap).
- **Recall reporting:** `RECALL_AT = (50, 100, 200, 500, 1000, 2000, 5000, 10000)`.
- **Channel budgets:**
  - `BUDGET_VECTOR = 5000`
  - `BUDGET_STATUTE = 800`
  - `BUDGET_CASE = 600`
  - `BUDGET_COURT_EXP = 1500`
  - `BUDGET_LAW_EXP = 400`
  - `BUDGET_LAW_DIRECT = 400`
  - `BUDGET_SAME_CODE = 300`
  - `BUDGET_STATUTE_COOC = 800` (NEW)
  - `BUDGET_CITE_GRAPH = 1500` (NEW)
- **Court-base expansion:** `CB_SEED_FROM_VECTOR = 1000`, `CB_PER_BASE = 30`.
- **Citation-graph expansion:** `CG_SEED_FROM_VECTOR = 500`, `CG_PER_DOC = 8`.
- **Fusion (RRF):** `RRF_K = 60`, `AUTHORITY_ALPHA = 0.20`. Channel weights: `vector=1.5`, `law_card_direct=2.5`, `statute=1.0`, `case=1.4`, `statute_co_occurrence=1.6`, `citation_graph=1.2`, `court_base_x=0.9`, `adjacent_law=0.7`, `same_law_code=0.5`.
- **Encoder:** `Qwen/Qwen3-Embedding-8B` via `sentence-transformers`, `torch_dtype=bfloat16`, `attn_implementation='sdpa'`, `padding_side='left'`, `max_seq_length=768`, `batch_size=8`, `normalize_embeddings=True`. Instruction prefix:
  `"Instruct: Given an English-language legal question or scenario about Swiss federal law, retrieve the Swiss statute articles or federal court decision considerations that are most directly relevant to answering it.\nQuery: "`.
- **Doc embeddings on GPU:** fp16, single dense tensor on CUDA. `DIM = 4096`.
- **Dependencies installed:** `sentence-transformers>=3.3`, `transformers>=4.51`, `accelerate>=0.34`, `pyarrow>=15`, `pandas`, `einops`.

## Data

- **Corpus rows (docs / manifest):** 2,652,248 (assertion verified `len(docs) == len(manifest)`).
- **Doc embedding chunks:** 27 files, `qwen3_8b_unified_chunk000..026.npy` (chunks 0-25 each 100,000 rows; chunk 026 has 52,248 rows). Concatenated into `doc_emb_gpu` of shape `(2652248, 4096)` torch.float16 on CUDA, occupying 21.73 GB VRAM (~21.7 GB total per the comment in cell 6).
- **Metadata zip (`retrieval_metadata.zip`, ~220 MB):** unzipped to `/content/metadata`, containing parquet tables and one JSON alias file. Loaded sizes:
  - `docs_meta.parquet`: 2,652,248 rows
  - `statute_links.parquet`: 5,295,519 rows
  - `case_links.parquet`: 1,813,789 rows
  - `adjacent_law_links.parquet`: 347,740 rows
  - `law_code_aliases.json`: 126 hardcoded cross-language alias groups
  - `qwen3_8b_unified_manifest.parquet`: 2,652,248 rows
- **Derived indexes built from the metadata:**
  - `court_base_groups`: 178,593 unique `court_base` keys
  - `statute_to_docs`: 171,498 unique normalized statute citations
  - `doc_to_statutes`: 1,414,972 docs with at least one statute citation
  - `law_cit_to_docs` (law-card citations): 70,935 unique normalized law-card citations
  - `law_code_to_docs`: 2,063 law-code groups
  - `case_to_docs`: 283,108 unique target citations; `case_base_to_docs`: 125,795 target bases
  - `doc_to_outgoing`: 744,046 docs with outgoing case citations
  - `neighbor_to_docs` (adjacent-law neighbors): 175,930
  - `citation_to_docid`: 2,161,111 unique citation strings
- **Alias coverage:** `ALL_KNOWN_FORMS` = 2,147 law-code forms (126 from hardcoded aliases plus 2,063 law codes from the corpus law family). Non-canonical forms get rewritten to the canonical SC number (e.g., `StPO` -> `312.0`, `CPP` -> `312.0`, `BGG` -> `173.110`).
- **Eval queries:** `data/val.csv` -> 10 rows, columns `['query_id', 'query', 'gold_citations']`. Query embedding matrix `Q` shape `(10, 4096)` float32, persisted to `EMB_DIR/qwen3_8b_query_val.npy`.

## Pipeline

1. **Install + mount** (cells 1-2). Pip-install dependencies, mount Google Drive at `/content/drive`, set paths, print GPU info, fix budgets and weights as constants.
2. **Metadata load** (cell 3). Unzip `retrieval_metadata.zip` once to `/content/metadata`, then load `docs_meta`, `statute_links`, `case_links`, `adjacent_law_links`, `law_code_aliases.json`, and the embedding manifest. Assert manifest row count matches docs.
3. **Alias map + citation normalization** (cell 4). Build `ALIAS_TO_FORMS`, `ALIAS_TO_CANONICAL`, `ALL_KNOWN_FORMS`. `normalize_citation()` collapses whitespace, rewrites non-canonical codes to the canonical SC number via a single regex, and optionally strips `Abs/Bst/Lit/Ziff/Ch/Cpv` suffix words. Applied symmetrically to gold and candidates.
4. **Inverted indexes** (cell 5). Construct `row_to_docid`, `docid_to_row`, `court_base_groups`, `statute_to_docs`, `doc_to_statutes`, `law_cit_to_docs`, `law_code_to_docs` (sorted by `authority_score` desc), `case_to_docs`, `case_base_to_docs`, `doc_to_outgoing`, `doc_to_outgoing_bases`, `neighbor_to_docs`, `docid_to_neighbors`, `citation_to_docid`, plus arrays `_citations_arr`, `_families_arr`, `_court_bases_arr`, `_authority_arr` and the helper `doc_meta(doc_id)`.
5. **Doc embeddings to GPU** (cell 6). Glob the 27 chunk files, sum rows, allocate `doc_emb_gpu` as a single torch fp16 tensor of shape `(total_rows, DIM)` on CUDA, then copy each `np.load(path)` chunk in slice-by-slice and free the host array.
6. **Anchor parser** (cell 7). `parse_anchors(query)` runs NFKC normalization plus hyphen unification, then matches:
   - `ART_PATTERN` for `Art. <num>[<letter>] [<suffix>...] <code>` where code is either an SC number (e.g., `442.132.3`) or an umlaut-aware / hyphenated abbreviation (optionally followed by a trailing number, e.g., `AsylV 2`, `V-HFKG`); rejects matches whose code is not in `ALL_KNOWN_FORMS`.
   - `BGE_FULL_RE` / `BGE_BASE_RE` for BGE citations including optional `E. <x.y>` considerations.
   - `DOCKET_RE` for modern dockets (`\d{1,2}[A-Z]+[._]\d{1,5}/\d{2,4}`).
   - `OLD_COURT_RE` for pre-2007 single-letter dockets (`U 421/00`, etc.).
   Returns `{cases, case_bases, dockets (modern + old), statutes}`.
7. **Channel implementations** (cell 8). Nine retrieval channels:
   - `vector_search`: normalize query, `doc_emb_gpu @ q.T` on GPU, `torch.topk(k)`.
   - `law_card_direct`: look up the canonicalized `Art. N <SC>` in `law_cit_to_docs` and pull law cards, ranked by authority.
   - `same_law_code`: for the parsed code (or its aliases), pull law cards from `law_code_to_docs` (sorted by authority).
   - `statute_anchor`: for each parsed `Art. N <code-form>`, pull docs from `statute_to_docs`.
   - `case_anchor`: pull from `case_to_docs` for full BGE citations, `case_base_to_docs` for bases + dockets, plus exact `_citations_arr` matches as fallback.
   - `statute_co_occurrence` (NEW): for parsed statutes, collect docs that cite them, count co-occurring statutes via `doc_to_statutes`, and pull law cards for the top-`top_cooc=300` co-citing statutes.
   - `citation_graph_expansion` (NEW v2 by `court_base`): for top seed court docs, take their own `court_base` and their outgoing `target_bases` (1-hop), then pull docs citing those bases (`case_base_to_docs`) and considerations within those bases (`court_base_groups`).
   - `expand_court_base`: for each seed's `court_base`, pull up to `CB_PER_BASE=30` siblings.
   - `expand_adjacent_law`: for each seed law card, pull docs sharing the next/previous article's neighbor citation.
8. **Reciprocal-rank fusion + retrieve()** (cell 9). `reciprocal_rank_fusion(channels, k_const=60, weights, alpha)` aggregates `w / (k_const + rank)` per doc across channels, applies `(1 + alpha * authority)`, sorts. `retrieve(query, q_emb, top_k)` parses anchors, runs all nine channels (using `seed_for_expansion = v[:1000] + s[:80] + c[:60] + lcd[:60]` for the expansion channels), measures per-channel timing, fuses, and returns the top-`TOP_K` plus channel counts and timings.
9. **Query encoding** (cell 10). Load `Qwen/Qwen3-Embedding-8B` (bf16, sdpa, left padding), set `max_seq_length=768`, prepend `QWEN_INSTRUCT` to each query, `model.encode(..., normalize_embeddings=True)` -> `Q` `(n, 4096)` float32. Save to `EMB_DIR/qwen3_8b_query_{SPLIT}.npy`, then `del model`, `gc.collect()`, `torch.cuda.empty_cache()`.
10. **Hybrid retrieval + recall@K** (cell 11). For each row in `val.csv`: parse gold (split on `;`), normalize gold via `normalize_citation`, run `retrieve(query, Q[i], TOP_K)`, normalize candidate citations, compute `recall@{50,100,200,500,1000,2000,5000,10000}` against the gold set, accumulate channel counts and timings, record dropped gold. Persist `val_summary.json`, `val_candidate_sets.jsonl` (top-1000 candidates per query), and `val_dropped_gold.csv` under `OUT_DIR`.
11. **Per-query diagnostics** (cell 12). Print a per-query table sorted worst-r@200 first, value-count of the most-frequently dropped gold citations, and aggregate recall@K.

## Results

Cell 1 (install):

```
[?25l   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0.0/1.8 MB ? eta -:--:--   ━━━━━━━━━━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━ 0.6/1.8 MB 19.2 MB/s eta 0:00:01   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 33.7 MB/s eta 0:00:00
[?25h
```

Cell 2 (mount + GPU):

```
Mounted at /content/drive
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition  VRAM: 102.0 GB  sm_120
split: val  top_k: 10000
```

Cell 3 (metadata load):

```
load time: 8.7s
docs        : 2,652,248
statute_links: 5,295,519
case_links  : 1,813,789
adj_links   : 347,740
manifest    : 2,652,248
```

Cell 4 (alias map + smoke checks):

```
known law-code forms: 2,147  (hardcoded=126, +from corpus=2,063)

  Art. 221 Abs. 1 lit. b StPO         -> Art. 221 312.0
  Art. 221 312.0                      -> Art. 221 312.0
  Art. 221 CPP                        -> Art. 221 312.0
  Art. 38a Abs. 1 TZV                 -> Art. 38a TZV
  Art. 20c Abs. 3 VÜPF                -> Art. 20c VÜPF
  Art. 67 Abs. 1 V-HFKG               -> Art. 67 V-HFKG
  Art. 51 Abs. 2 AsylV 2              -> Art. 51 AsylV 2
  Art. 13 Abs. 3bis 910.18            -> Art. 13 910.18
  Art. 8 Abs. 1 442.132.3             -> Art. 8 442.132.3
  BGE 137 IV 122 E. 6.2               -> BGE 137 IV 122 E. 6.2
  1B_210/2023 E. 4.1                  -> 1B_210/2023 E. 4.1
  1A.204/2004 14.12.2004 E. A         -> 1A.204/2004 14.12.2004 E. A
  U 421/00 07.05.2002 E. A            -> U 421/00 07.05.2002 E. A
  Art. 100 Abs. 1 BGG                 -> Art. 100 173.110
```

Cell 5 (inverted indexes):

```
  court_base groups: 178,593
  unique normalized statutes (court-paragraph citations): 171,498
  doc_to_statutes: 1,414,972 docs with at least one statute citation
  unique normalized law-card citations: 70,935
  law_code groups: 2,063
  unique target citations: 283,108  target_bases: 125,795
  doc_to_outgoing: 744,046 docs with outgoing case citations
  adjacent_law neighbors: 175,930
  citation_to_docid: 2,161,111
index build: 53.0s
```

Cell 6 (doc embeddings to GPU):

```
discovered 27 chunk file(s)
total rows across chunks: 2,652,248  (manifest has 2,652,248)
  qwen3_8b_unified_chunk000.npy             rows=100,000  cumulative=   100,000
  qwen3_8b_unified_chunk001.npy             rows=100,000  cumulative=   200,000
  qwen3_8b_unified_chunk002.npy             rows=100,000  cumulative=   300,000
  qwen3_8b_unified_chunk003.npy             rows=100,000  cumulative=   400,000
  qwen3_8b_unified_chunk004.npy             rows=100,000  cumulative=   500,000
  qwen3_8b_unified_chunk005.npy             rows=100,000  cumulative=   600,000
  qwen3_8b_unified_chunk006.npy             rows=100,000  cumulative=   700,000
  qwen3_8b_unified_chunk007.npy             rows=100,000  cumulative=   800,000
  qwen3_8b_unified_chunk008.npy             rows=100,000  cumulative=   900,000
  qwen3_8b_unified_chunk009.npy             rows=100,000  cumulative= 1,000,000
  qwen3_8b_unified_chunk010.npy             rows=100,000  cumulative= 1,100,000
  qwen3_8b_unified_chunk011.npy             rows=100,000  cumulative= 1,200,000
  qwen3_8b_unified_chunk012.npy             rows=100,000  cumulative= 1,300,000
  qwen3_8b_unified_chunk013.npy             rows=100,000  cumulative= 1,400,000
  qwen3_8b_unified_chunk014.npy             rows=100,000  cumulative= 1,500,000
  qwen3_8b_unified_chunk015.npy             rows=100,000  cumulative= 1,600,000
  qwen3_8b_unified_chunk016.npy             rows=100,000  cumulative= 1,700,000
  qwen3_8b_unified_chunk017.npy             rows=100,000  cumulative= 1,800,000
  qwen3_8b_unified_chunk018.npy             rows=100,000  cumulative= 1,900,000
  qwen3_8b_unified_chunk019.npy             rows=100,000  cumulative= 2,000,000
  qwen3_8b_unified_chunk020.npy             rows=100,000  cumulative= 2,100,000
  qwen3_8b_unified_chunk021.npy             rows=100,000  cumulative= 2,200,000
  qwen3_8b_unified_chunk022.npy             rows=100,000  cumulative= 2,300,000
  qwen3_8b_unified_chunk023.npy             rows=100,000  cumulative= 2,400,000
  qwen3_8b_unified_chunk024.npy             rows=100,000  cumulative= 2,500,000
  qwen3_8b_unified_chunk025.npy             rows=100,000  cumulative= 2,600,000
  qwen3_8b_unified_chunk026.npy             rows=52,248  cumulative= 2,652,248

load time: 330.4s
doc_emb_gpu: torch.Size([2652248, 4096]) torch.float16
VRAM allocated: 21.73 GB
```

Cell 7 (anchor parser smoke checks):

```
[val_001 query stub]
  statutes: ['Art. 221 312.0', 'Art. 100 173.110']
  cases: ['BGE 137 IV 122 E. 6.2']
  case_bases: ['BGE 137 IV 122']
  dockets: ['1B_210/2023']

[old-style stub]
  statutes: []
  cases: []
  case_bases: []
  dockets: ['H 418/99', 'I 89/02', 'U 421/00']

[rare codes stub]
  statutes: ['Art. 67 V-HFKG', 'Art. 20c VÜPF', 'Art. 51 AsylV 2']
  cases: []
  case_bases: []
  dockets: []

[period docket stub]
  statutes: ['Art. 8 442.132.3']
  cases: []
  case_bases: []
  dockets: ['1A.204/2004']
```

Cell 8 (channels banner):

```
channels: vector, law_card_direct, same_law_code, statute, case, statute_co_occurrence (NEW), citation_graph (NEW v2 — by court_base), court_base_x, adjacent_law
```

Cell 9 (retrieve banner):

```
retrieve() v4 ready: 9 channels, top_k= 10000
```

Cell 10 (query encoding):

```
WARNING:sentence_transformers.util.decorators:The `tokenizer_kwargs` argument was renamed and is now deprecated. Please use `processor_kwargs` instead.
val: 10 queries  cols: ['query_id', 'query', 'gold_citations']
free VRAM for model load: ~80.2 GB
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
model load: 54.9s  VRAM allocated: 36.86 GB
encoded 10 queries in 5.4s  shape=(10, 4096)
saved query embeddings to /content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_query_val.npy
VRAM after model free: 21.74 GB
```

Cell 11 (hybrid retrieval + recall@K):

```
  5/10  r@200=0.2327  r@1000=0.4096  r@10000=0.5693  gold_uniq=10 found=8  time/q=0.09s
  10/10  r@200=0.2535  r@1000=0.4013  r@10000=0.5307  gold_uniq=23 found=10  time/q=0.08s

=== summary ===
{
  "split": "val",
  "n_queries": 10,
  "top_k": 10000,
  "elapsed_s": 0.8,
  "recall_at_k": {
    "r@50": 0.16956798103856927,
    "r@100": 0.20380017112753684,
    "r@200": 0.2534912984145721,
    "r@500": 0.3673237901626648,
    "r@1000": 0.4013426329283106,
    "r@2000": 0.44681271839839615,
    "r@5000": 0.5116862620187429,
    "r@10000": 0.5306882730795774
  },
  "channel_counts_total": {
    "vector": 50000,
    "law_card_direct": 18,
    "same_law_code": 850,
    "statute": 2147,
    "case": 0,
    "statute_co_occurrence": 1466,
    "citation_graph": 15000,
    "court_base_x": 15000,
    "adjacent_law": 631
  },
  "channel_avg_time_ms": {
    "vector": 39,
    "law_card_direct": 0,
    "same_law_code": 0,
    "statute": 0,
    "case": 0,
    "statute_co_occurrence": 1,
    "court_base_x": 1,
    "citation_graph": 2,
    "adjacent_law": 0
  },
  "config": {
    "budgets": {
      "vector": 5000,
      "statute": 800,
      "case": 600,
      "court_base_x": 1500,
      "adjacent_law": 400,
      "law_card_direct": 400,
      "same_law_code": 300,
      "statute_co_occurrence": 800,
      "citation_graph": 1500
    },
    "rrf_k": 60,
    "authority_alpha": 0.2,
    "channel_weights": {
      "vector": 1.5,
      "law_card_direct": 2.5,
      "statute": 1.0,
      "case": 1.4,
      "statute_co_occurrence": 1.6,
      "citation_graph": 1.2,
      "court_base_x": 0.9,
      "adjacent_law": 0.7,
      "same_law_code": 0.5
    },
    "cb_seed_from_vector": 1000,
    "cg_seed_from_vector": 500
  }
}

wrote /content/drive/MyDrive/swiss_law/artifacts/eval/val_summary.json
wrote /content/drive/MyDrive/swiss_law/artifacts/eval/val_candidate_sets.jsonl
wrote /content/drive/MyDrive/swiss_law/artifacts/eval/val_dropped_gold.csv
```

Cell 12 (per-query inspection):

```
=== per-query (sorted worst recall@200 first) ===
query_id  gold_total  gold_unique  found    r@200   r@1000  r@10000  n_statutes_parsed  n_cases_parsed
 val_003          47           45      9 0.022222 0.066667 0.200000                  0               0
 val_001          42           39     32 0.102564 0.384615 0.820513                  1               0
 val_008          29           28      3 0.107143 0.107143 0.107143                  0               0
 val_010          25           23     10 0.130435 0.304348 0.434783                  0               0
 val_009          14           12      6 0.166667 0.416667 0.500000                  0               0
 val_002          36           34     16 0.294118 0.441176 0.470588                  1               0
 val_005          11           10      8 0.300000 0.600000 0.800000                  0               0
 val_007          19           17      9 0.411765 0.470588 0.529412                  2               0
 val_004          10            9      5 0.444444 0.555556 0.555556                  0               0
 val_006          18           18     16 0.555556 0.666667 0.888889                  3               0

=== top of dropped-gold (most-frequently missed citations across queries) ===
gold_citation
Art. 100 173.110          7
Art. 428 312.0            3
Art. 39 173.71            2
Art. 37 173.71            2
Art. 385 312.0            2
Art. 29 101               2
Art. 390 312.0            2
Art. 16 210               2
BGE 134 V 231 E. 5.1      1
BGE 133 I 270 E. 3.4.2    1
7B_496/2025 E. 3.2        1
Art. 82 173.110           1
BGE 139 V 176 E. 5.3      1
Art. 113 173.110          1
BGE 127 V 205 E. 4b       1
BGE 140 V 193 E. 3.2      1
BGE 139 V 399 E. 5.5      1
Art. 60 830.1             1
Art. 56 830.1             1
BGE 144 V 427 E. 3.2      1
BGE 132 V 93 E. 4         1
BGE 148 V 21 E. 5.3       1
9C_623/2020 E. 4.2        1
BGE 139 V 399 E. 5.3      1
BGE 135 V 39 E. 6.1       1

=== aggregate recall ===
  recall@   50  =  0.1696
  recall@  100  =  0.2038
  recall@  200  =  0.2535
  recall@  500  =  0.3673
  recall@ 1000  =  0.4013
  recall@ 2000  =  0.4468
  recall@ 5000  =  0.5117
  recall@10000  =  0.5307
```

## Summary

End-to-end notebook that runs hybrid retrieval over the 2.65M-doc unified Swiss legal corpus on a single 102 GB Blackwell GPU, fusing nine channels (Qwen3-Embedding-8B vector ANN at fp16, statute / case / law-card-direct / same-law-code anchors, statute co-occurrence, citation-graph by court_base, court-base sibling expansion, adjacent-law expansion) via reciprocal-rank fusion (k=60, authority alpha=0.20) and reporting recall@{50..10000} against `val.csv` after symmetric citation normalization. Setup loads all 27 `qwen3_8b_unified_chunk*.npy` chunks into one fp16 GPU tensor (21.73 GB VRAM), unzips `retrieval_metadata.zip` into seven inverted indexes (largest: `statute_links` at 5.29M rows, `case_links` at 1.81M rows), and registers 2,147 alias forms covering the corpus's 2,063 law-codes plus 126 hardcoded cross-language groups. On 10 val queries with `TOP_K=10000`, hybrid retrieval averages 0.08 s/query and yields aggregate recall@200=0.2535, recall@1000=0.4013, recall@10000=0.5307; the `case` channel returned zero hits across the run (no parsed BGE/docket anchors were resolved to gold-bearing docs for these 10 queries), and the citation-graph and court-base expansion channels each filled their full 1,500-item budget per query (15,000 total). Per-query r@200 ranges from 0.022 (val_003, no anchors parsed) to 0.556 (val_006, 3 statutes parsed); the single most-frequently dropped gold citation is `Art. 100 173.110` (BGG art. 100, missed in 7 of 10 queries). Outputs persisted to `OUT_DIR`: `val_summary.json`, `val_candidate_sets.jsonl` (top-1000 candidates per query), and `val_dropped_gold.csv`.
