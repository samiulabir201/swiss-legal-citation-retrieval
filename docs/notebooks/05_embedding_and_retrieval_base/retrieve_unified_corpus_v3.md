# retrieve_unified_corpus_v3

**Path:** `e:\swiss_citation_extraction\notebooks\05_embedding_and_retrieval_base\retrieve_unified_corpus_v3.ipynb`

## Configuration

- Runtime: Colab single GPU (`NVIDIA RTX PRO 6000 Blackwell Server Edition`, 102.0 GB VRAM, sm_120).
- Embedding model: `Qwen/Qwen3-Embedding-8B` loaded via `SentenceTransformer` with `torch_dtype=torch.bfloat16`, `attn_implementation='sdpa'`, `tokenizer_kwargs={'padding_side': 'left'}`, `max_seq_length=768`.
- Query instruction prefix:
  - `"Instruct: Given an English-language legal question or scenario about Swiss federal law, retrieve the Swiss statute articles or federal court decision considerations that are most directly relevant to answering it.\nQuery: "`
- Doc-embedding tensor: `doc_emb_gpu` shape `(2,652,248, 4096)` `torch.float16` on CUDA (VRAM ~21.73 GB).
- Eval target: `SPLIT='val'`, `TOP_K=10000`, `QUERY_LIMIT=0`.
- Channel budgets:
  - `BUDGET_VECTOR=5000`, `BUDGET_STATUTE=800`, `BUDGET_CASE=600`, `BUDGET_COURT_EXP=1500`, `BUDGET_LAW_EXP=400`, `BUDGET_LAW_DIRECT=400`, `BUDGET_SAME_CODE=300`, `BUDGET_STATUTE_COOC=800`, `BUDGET_CITE_GRAPH=1500`.
- Expansion seeding: `CB_SEED_FROM_VECTOR=1000`, `CB_PER_BASE=30`, `CG_SEED_FROM_VECTOR=500`, `CG_PER_DOC=8`.
- Fusion: `RRF_K=60`, `AUTHORITY_ALPHA=0.20`, and `CHANNEL_WEIGHTS = {vector: 1.5, law_card_direct: 2.5, statute: 1.0, case: 1.4, statute_co_occurrence: 1.6, citation_graph: 1.2, court_base_x: 0.9, adjacent_law: 0.7, same_law_code: 0.5}`.
- Paths: `ROOT=/content/drive/MyDrive/swiss_law`, `DATA_DIR={ROOT}/data`, `EMB_DIR={ROOT}/artifacts/embeddings`, `META_ZIP={ROOT}/artifacts/retrieval_metadata.zip`, `META_DIR=/content/metadata`, `OUT_DIR={ROOT}/artifacts/eval`.
- Recall-at-K targets: `(50, 100, 200, 500, 1000, 2000, 5000, 10000)`.
- Dependency installs: `sentence-transformers>=3.3`, `transformers>=4.51`, `accelerate>=0.34`, `pyarrow>=15`, `pandas`, `einops`.

## Data

- Query data: `{DATA_DIR}/val.csv` (10 queries; columns `query_id`, `query`, `gold_citations`).
- Doc embeddings: 27 chunk files `qwen3_8b_unified_chunk000..026.npy` (rows: 26 x 100,000 + 52,248 = 2,652,248), dim 4096, fp16; concatenated on GPU. Load time: 330.4s.
- Manifest: `qwen3_8b_unified_manifest.parquet` (2,652,248 rows; doc_id alignment with chunks).
- Retrieval metadata (from `retrieval_metadata.zip`, ~220 MB) extracted to `/content/metadata`:
  - `docs_meta.parquet`: 2,652,248 rows.
  - `statute_links.parquet`: 5,295,519 court-paragraph statute citations.
  - `case_links.parquet`: 1,813,789 outgoing case citations (`target_citation`, `target_base`).
  - `adjacent_law_links.parquet`: 347,740 adjacency links (`neighbor_citation`).
  - `law_code_aliases.json`: cross-language alias groups (DE/FR/IT/SC-number).
- Derived indexes after `normalize_citation` is applied:
  - `court_base_groups`: 178,593.
  - `statute_to_docs` (unique normalized statutes): 171,498; `doc_to_statutes` covers 1,414,972 docs.
  - `law_cit_to_docs` (normalized law-card citations): 70,935.
  - `law_code_to_docs` (sorted by `authority_score`): 2,063.
  - `case_to_docs` (target citations): 283,108; `case_base_to_docs` (target bases): 125,795.
  - `doc_to_outgoing` (outgoing case citations): 744,046; `doc_to_outgoing_bases` likewise.
  - `neighbor_to_docs` (adjacent-law): 175,930.
  - `citation_to_docid`: 2,161,111.
- Alias coverage: 2,147 known law-code forms (126 hardcoded + 2,063 from corpus `law_code` family).

## Pipeline

1. Install dependencies and mount Drive (cells 1-2).
2. Load metadata zip into `/content/metadata`; read `docs_meta`, `statute_links`, `case_links`, `adjacent_law_links`, `law_code_aliases.json`, manifest (cell 3, load 8.7s).
3. Build alias map (`ALIAS_TO_FORMS`, `ALIAS_TO_CANONICAL`, `ALL_KNOWN_FORMS`) and regex-based `normalize_citation` (whitespace collapse + alias->SC number rewrite + optional Abs/Bst/Lit/Ziff/Ch/Cpv suffix strip). Smoke tests on 14 form patterns (cell 4).
4. Build inverted indexes: `court_base_groups`, `statute_to_docs`, `doc_to_statutes`, `law_cit_to_docs`, `law_code_to_docs` (sorted by `authority_score`), `case_to_docs`, `case_base_to_docs`, `doc_to_outgoing`, `doc_to_outgoing_bases`, `neighbor_to_docs`, `docid_to_neighbors`, `citation_to_docid` (cell 5, build 53.0s).
5. Load all 27 doc-embedding chunk `.npy` files into `doc_emb_gpu` (2,652,248 x 4096 fp16) on CUDA; assert `total_rows == len(manifest)` (cell 6, load 330.4s, VRAM 21.73 GB).
6. Anchor parsing (`parse_anchors`): NFKC-normalize, replace non-breaking hyphens; regex extract `ART_PATTERN` (`Art. N[a-z]* (Abs|Bst|Lit|Ziff|Ch|Cpv)*` -> SC number or umlaut/hyphenated abbreviation, rejecting unknown codes), `BGE_FULL_RE`, `BGE_BASE_RE`, `DOCKET_RE` (`1B_210/2023`, `1A.204/2004`), `OLD_COURT_RE` (single-letter pre-2007). Old-style courts merged into `dockets`. Returns dict `{cases, case_bases, dockets, statutes}` where each statute carries `article`, `code`, `canonical`, `forms`, `suffix_raw`, `normalized` (cell 7, four smoke stubs printed).
7. Channel implementations (cell 8):
   - `vector_search`: normalize float32 query, half-precision matmul `doc_emb_gpu @ q.T`, top-k (5000).
   - `law_card_direct`: lookup law cards whose normalized citation equals `Art. {art} {canonical}`; sort by authority.
   - `same_law_code`: pull top-authority docs sharing the canonical law_code.
   - `statute_anchor`: union over `forms` of `statute_to_docs[Art. art code]`.
   - `case_anchor`: `case_to_docs[c]`, then `case_base_to_docs[b]` for bases/dockets, then literal-citation scan via `_citations_arr`.
   - `statute_co_occurrence` (NEW): from docs citing the target statute, count co-cited statutes (Counter), pull up to 3 law-card docs per top-300 co-occurring statutes; raw score = `authority + log1p(n)/10`.
   - `citation_graph_expansion` (NEW v2): from top vector seeds with `family=='court'`, collect their own `court_base` plus `doc_to_outgoing_bases` (1 hop); for each base pull docs from `case_base_to_docs[base]` and `court_base_groups[base]`.
   - `expand_court_base`: per seed's `court_base`, pull up to `CB_PER_BASE=30` sibling docs (cap `BUDGET_COURT_EXP=1500`).
   - `expand_adjacent_law`: for each seed of `family=='law'`, look up neighbor citations and the docs covering them.
8. Hybrid `retrieve`: run all 9 channels timing each, build a single `seed_for_expansion = v[:1000] + s[:80] + c[:60] + lcd[:60]`, fuse active channels with weighted reciprocal-rank `score += w/(k_const + rank)`, multiply by `(1 + alpha * authority)`, sort, take top-K (cell 9, `top_k=10000`).
9. Encode val queries (cell 10): read `val.csv` (10 queries), load Qwen3-Embedding-8B (model load 54.9s, VRAM 36.86 GB), encode with `batch_size=8`, `normalize_embeddings=True`, save `qwen3_8b_query_val.npy` (5.4s, shape `(10, 4096)`), free model (VRAM back to 21.74 GB).
10. Per-query eval loop (cell 11): for each row, parse gold via `';' .split`, normalize, `retrieve(query, Q[i], top_k=10000)`, normalize candidate citations, compute `recall@{50,100,200,500,1000,2000,5000,10000}`, accumulate channel counts/timings, track dropped gold. Write `val_summary.json`, `val_candidate_sets.jsonl` (top 1000 per query), `val_dropped_gold.csv`.
11. Per-query inspection table sorted by `r@200` ascending plus top-25 most-frequently-missed gold citations and aggregate recalls (cell 12).

## Results

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

Side outputs (selected):

- `load time: 8.7s` (metadata), `index build: 53.0s`, `load time: 330.4s` (doc embeddings), `model load: 54.9s`, `encoded 10 queries in 5.4s shape=(10, 4096)`, `VRAM after model free: 21.74 GB`.
- Alias smoke: `Art. 221 Abs. 1 lit. b StPO -> Art. 221 312.0`; `Art. 221 CPP -> Art. 221 312.0`; `Art. 100 Abs. 1 BGG -> Art. 100 173.110`; suffix-strip and SC canonicalization confirmed across 14 patterns.
- Anchor parser smoke: val_001 stub -> `statutes=['Art. 221 312.0', 'Art. 100 173.110']`, `cases=['BGE 137 IV 122 E. 6.2']`, `case_bases=['BGE 137 IV 122']`, `dockets=['1B_210/2023']`. Old-style stub -> `dockets=['H 418/99', 'I 89/02', 'U 421/00']`. Rare codes stub -> `statutes=['Art. 67 V-HFKG', 'Art. 20c VÜPF', 'Art. 51 AsylV 2']`. Period docket stub -> `statutes=['Art. 8 442.132.3']`, `dockets=['1A.204/2004']`.

## Summary

This v3 unified-corpus hybrid retriever fuses 9 channels - vector ANN against a 2.65M x 4096 fp16 Qwen3-Embedding-8B index plus statute-anchor, case-anchor, law-card-direct, same-law-code, statute-co-occurrence (new), citation-graph 1-hop via court_base (new v2), court-base expansion, and adjacent-law expansion - via authority-boosted reciprocal-rank fusion (RRF k=60, alpha=0.20) on val.csv (10 queries) with `top_k=10000`. Citation normalization (suffix-strip + 2,147-form alias/SC canonicalization, including 2,063 corpus law codes) is applied symmetrically to gold and candidates. Aggregate recall is r@200=0.2535, r@1000=0.4013, r@10000=0.5307; only val_001, val_002, val_006 and val_007 parse any statute or case anchors, and parsing collapses for the other six (which is why recall@200 is bimodal: val_006 hits 0.556 while val_003 sits at 0.022). The most-dropped gold is procedural-scaffold law cards in BGG (`Art. 100 173.110` missed 7x) and StPO (`Art. 428/385/390 312.0`), suggesting anchor parsing and the co-occurrence channel need richer English-query coverage for statute pulls before recall can climb above 0.53 at K=10000.
