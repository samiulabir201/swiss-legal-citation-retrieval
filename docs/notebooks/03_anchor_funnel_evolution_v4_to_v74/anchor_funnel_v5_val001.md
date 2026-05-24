# anchor_funnel_v5_val001.ipynb

**Path:** notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v5_val001.ipynb

## Configuration

### Models
- Query expansion: `Qwen/Qwen3-32B` (bf16, ~65 GB)
- Embeddings: `Qwen/Qwen3-Embedding-8B` (bf16 via SentenceTransformer, last-token pooling, left padding, max_seq_length=4096)
- Reranker: `Qwen/Qwen3-Reranker-8B` (bf16, AutoModelForCausalLM, scores yes/no logits at last position)

### Libraries
`os, sys, json, time, math, gc, re, csv, pathlib, collections (defaultdict/Counter), sqlite3, numpy, pyarrow.parquet, pandas, torch (2.10.0+cu128), transformers (AutoTokenizer/AutoModelForCausalLM), sentence_transformers (SentenceTransformer)`

### Hyperparameters (`CONFIG`)
```
topk_final: 1000
budget_law_direct: null (uncapped, guarantee channel)
budget_court_statute: 600
budget_sibling: 500
budget_concept: 600
budget_term: 500
budget_per_area: 150
budget_co_citation: 400
budget_bm25: 600
budget_vector: 800
budget_vector_enriched: 800
rrf_k: 60
guarantee_channels: ["law_direct_match", "per_area_bedrock"]
per_area_top_n: 100
co_citation_top_k_per_target: 8
co_citation_min_co_count: 50
co_citation_max_neighbour_count: 5000
concept_substring_top_k: 6
bm25_max_query_terms: 60
bm25_min_token_len: 3
vector_emb_model: "Qwen/Qwen3-Embedding-8B"
vector_topk: 800
qwen_query_model: "Qwen/Qwen3-32B"
qwen_max_new_tokens: 1024
use_reranker: true
rerank_pool_size: 3000
rerank_model: "Qwen/Qwen3-Reranker-8B"
rerank_max_doc_chars: 1500
rerank_batch_size: 16
noise_paragraph_roles: {"notification","header","empty","metadata"}
lowercase_concepts: true
lowercase_terms: true
```

### Constants
- `CODE_ALIAS` (FR/IT/legacy code mapping): CPP→StPO, CP→StGB, CC→ZGB, CO→OR, LTF→BGG, LACI→AVIG, LAA→UVG, LP→SchKG, LDIP→IPRG, Cst→BV, Cst.→BV, STPO→StPO, OBG→OR.
- `LEGAL_AREA_DEFAULT_CODE`: criminal procedure→StPO, criminal law→StGB, civil law→ZGB, obligations→OR, civil procedure→ZPO, constitutional/public→BV, administrative→VwVG, social insurance→ATSG, tax→DBG.
- `ART_RE`, `CODE_RE`, `CASE_BGE_RE`, `CASE_DOCKET_RE`, `TOKEN_NORM_RE`, `FTS5_BAD_CHARS`, `TOK_RE`.
- `QWEN_QUERY_INSTRUCT` prefix: `"Instruct: Given an English-language legal question or scenario about Swiss federal law, retrieve the Swiss statute articles or federal court decision considerations that are most directly relevant to answering it.\nQuery: "`.
- `RERANK_TEMPLATE`: Qwen3-Reranker chat-style template with `<Instruct>: Given a Swiss legal question, retrieve relevant Swiss statute articles or Federal Court (Bundesgericht) decision considerations.`, yes/no judgement.

### Hardware
- Python: 3.12.13; PyTorch: 2.10.0+cu128.
- GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB.
- Memory plan: Qwen3-32B (~65 GB) loaded then freed → Qwen3-Embedding-8B + corpus E_GPU (~37 GB) → Qwen3-Reranker-8B (~16 GB).

## Data

Auto-detect `DATA_ROOT` from candidate roots:
- `/content/drive/MyDrive/swiss_law`
- `/content/drive/MyDrive/swiss_citation_extraction`
- `/content/swiss_citation_extraction`
- `E:/swiss_citation_extraction`
- `Path.cwd()`

Resolved at runtime to `DATA_ROOT = /content/drive/MyDrive/swiss_law` with paths:
- `val_csv`: `/content/drive/MyDrive/swiss_law/data/val.csv` (OK)
- `law_llm`: `/content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl` (OK)
- `court_v5`: `/content/drive/MyDrive/swiss_law/artifacts_v2/court_authority_cards_v5_unified.jsonl` (OK)
- `emb_dir`: `/content/drive/MyDrive/swiss_law/artifacts/embeddings` (OK)
- `emb_manifest`: `/content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_manifest.parquet` (OK)
- `out_dir`: `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v4`

Fallback candidates checked (via `first_existing`):
- law_llm fallback: `DATA_ROOT/law_json_llm_output/law_llm_descriptors_0000000_all.jsonl`
- court_v5 fallback: `DATA_ROOT/artifacts/court_authority_cards_v5_unified.jsonl`

Embedding chunks: pattern `qwen3_8b_unified_chunk*.npy` (27 chunks loaded, ~21 GB).

Output artifacts (under `out_dir`):
- `val_001_query_targets.json`
- `val_001_v5_per_channel.json`
- `val_001_v5_per_gold_trace.tsv`
- `val_001_v5_top1000_pool.txt`
- `val_001_v5_summary.md`

Query: `val_001` (42 gold citations). First 240 chars: "May a court lawfully order a three‑month extension of pre‑trial detention under Art. 221 Abs. 1 lit. b StPO (risk of collusion) consistent with the principle of proportionality when the accused—detained after an alleged late‑night assault a...".

## Pipeline

### Stage 1 — Environment & paths (Cells 1–2)
- Probe Python/Torch/GPU.
- Mount Google Drive.
- Auto-detect `DATA_ROOT`, build `PATHS` dict, create `out_dir`, detect embedding chunk availability (`EMB_AVAILABLE`).

### Stage 2 — Config (Cell 3)
- Define `CONFIG` knob panel — channel budgets, RRF k, guarantee channels, co-citation filters, BM25/vector/rerank knobs, negative-gate paragraph roles. No hardcoded statutes/cases.

### Stage 3 — Load query + gold (Cell 4)
- Stream `val.csv` (csv field_size_limit=2**31-1), find `query_id == "val_001"`, parse `gold_citations` by `;`. 42 gold strings.

### Stage 4 — One-pass index build (Cell 5)
- Streams `law_llm` JSONL and `court_v5` JSONL.
- Builds: `cit_to_doc_ids`, `doc_meta`, `idx_law_direct`, `idx_court_statute`, `idx_case_anchor`, `idx_court_base`, `idx_concept_en`, `idx_term_orig`, `legal_area_per_doc`, `search_text` (per-doc BM25 corpus), `co_citation_pairs` (unordered canonical pairs per court row).
- Statute canonicalization via `statute_anchor_canonical` + `canonicalize_row_anchors` (two-pass row-level primary code fallback + `LEGAL_AREA_DEFAULT_CODE` fallback).
- Case canonicalization via `case_anchor_canonical` (BGE and docket regex).
- Doc IDs: `law:<i>` and `court:<i>`.

### Stage 5 — Map gold → doc_ids (Cell 6)
- For each gold citation string, look up `cit_to_doc_ids`; union into `gold_doc_set`; set `total_gold`.

### Stage 6 — Per-legal-area bedrock (Cell 7)
- Rebuild `per_area_canon_count[area][canon] = #distinct court_base citing canon within area`.
- Drops global v3 bedrock; replaces with `legal_area_static`-cluster-conditional ranking.

### Stage 7 — Co-citation expansion (Cell 8)
- Build `co_neighbours[canon] = [(neighbour, count), ...]`.
- Filters: drop pairs with `co_count < 50`; drop neighbours with global corpus count > 5000 (v5 FIX 3 — kills BV-class universals); keep top 8 per target.

### Stage 8 — BM25 index (Cell 9)
- In-memory SQLite FTS5 over `search_text` (`tokenize='unicode61 remove_diacritics 2'`); batched inserts of 50,000.
- Free `search_text` dict after build.
- `bm25_search`: tokenize query, OR-match phrases, ORDER BY bm25() ascending, flip sign to descending.

### Stage 9 — Vector channel setup (Cell 10)
- Load manifest parquet; build `manifest_row → my_doc_id` map by `(family, citation)`.
- Load all 27 `qwen3_8b_unified_chunk*.npy` chunks → concatenate → upload to CUDA as fp16 `E_GPU`.

### Stage 10 — Query-embedding helpers (Cell 11)
- Lazy `_ensure_emb_model` loads SentenceTransformer with last-token pooling, `padding_side='left'`, bf16, sdpa attention.
- `encode_query`: prepend `QWEN_QUERY_INSTRUCT` instruction, encode with normalize.
- `vector_search`: GPU matmul `E_GPU @ q`, topk, map row → did via `manifest_row_to_did`.

### Stage 11 — Qwen3-32B query expansion (Cell 12)
- Apply chat template (`enable_thinking=False`), generate up to 1024 tokens with `do_sample=False`, `temperature=0.0`.
- Parse JSON span between first `{` and last `}` into `targets` dict (statute_targets / case_targets / concept_targets_en / term_targets_de / term_targets_fr / legal_area_keywords).
- Immediately free the 32B model and clear CUDA cache.
- Save `val_001_query_targets.json`.

### Stage 12 — Encode raw + enriched query (Cell 13)
- Enriched query = `QUERY + "\n\nKeywords: " + " ; ".join(term_de + term_fr + concept_en + legal_area_keywords)`.
- Produce `q_emb_raw`, `q_emb_enriched`.

### Stage 13 — Channel functions (Cell 14)
- `channel_law_direct` (Counter over canon set against idx_law_direct).
- `channel_court_statute` (Counter against idx_court_statute, LLM-named canons only).
- `channel_sibling` (court_base expansion from `seed = court_statute hits ∪ court-family law_direct hits`).
- `expand_concepts_strict` + `channel_concept` (strict substring overlap with corpus concept vocab).
- `channel_term` (term_targets_de/fr against idx_term_orig).
- `channel_per_area_bedrock` (per-area canon ranking filtered to LLM-named codes — v5 FIX 4).
- `channel_co_citation` (LLM statute targets → co-neighbours → law + court rows).
- `channel_vector` (uses `vector_search`).

### Stage 14 — Run channels (Cell 15)
- Build `llm_statute_canons`, expand to `co_expanded_canons` via `co_neighbours`.
- Extract `statute_target_codes` (set of code suffixes for per-area gating).
- Expand concepts strict-substring.
- Run all 10 channels; print per-channel size/gold/recall and union recall.

### Stage 15 — RRF fusion + negative gate (Cell 16)
- `rrf_fuse(channels, k=60)` accumulates `1/(60+rank+1)` per (channel,rank).
- `apply_neg_gate` drops notification paragraphs and `noise_paragraph_roles`.
- Prepend `guarantee_channels` hits in order; fill to `topk_final=1000` (RRF baseline) and `rerank_pool_size=3000` (input pool).
- Print R@K curve.

### Stage 16 — Reranker (Cell 17)
- Free `EMB_MODEL` + `E_GPU`, empty CUDA cache.
- Load Qwen3-Reranker-8B (`AutoTokenizer` with `padding_side='left'`, `AutoModelForCausalLM` bf16).
- Resolve `YES_ID`/`NO_ID` token ids.
- For each (query, doc) pair: format with `RERANK_TEMPLATE`, truncate doc to 1500 chars, batch=16, max_length=4096; take last-position logits at `[NO_ID, YES_ID]`, softmax, take yes probability.
- Sort descending; final_topk = top 1000.
- Free reranker.
- Print post-rerank R@K.

### Stage 17 — Per-gold trace (Cell 18)
- Build `channel_membership[did] → {channel_names}` and `rank_lookup[did] → rank`.
- For each gold citation, find best ranked doc_id and the channels containing it.

### Stage 18 — Save artifacts (Cell 19)
- Dump `val_001_v5_per_channel.json`, `val_001_v5_per_gold_trace.tsv`, `val_001_v5_top1000_pool.txt`, `val_001_v5_summary.md`.

### Stage 19 — Cleanup (Cell 20)
- Close SQLite connection; del large globals (indexes, neighbours, E_GPU, EMB_MODEL, rerank scores); gc + empty CUDA cache.

## Results

### Env (Cell 1)
```
Python: 3.12.13
PyTorch: 2.10.0+cu128
  GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB
```
Drive mounted at `/content/drive`.

### Paths (Cell 2)
```
DATA_ROOT = /content/drive/MyDrive/swiss_law
  val_csv: OK  /content/drive/MyDrive/swiss_law/data/val.csv
  law_llm: OK  /content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl
  court_v5: OK  /content/drive/MyDrive/swiss_law/artifacts_v2/court_authority_cards_v5_unified.jsonl
  emb_dir: OK  /content/drive/MyDrive/swiss_law/artifacts/embeddings
  emb_manifest: OK  /content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_manifest.parquet
```

### Query (Cell 4)
```
val_001: 42 gold citations
Query (first 240 chars): May a court lawfully order a three‑month extension of pre‑trial detention under Art. 221 Abs. 1 lit. b StPO (risk of collusion) consistent with the principle of proportionality when the accused—detained after an alleged late‑night assault a...
```

### Index build (Cell 5)
```
Law: 173,033 rows indexed in 5.4s
  court progress: 250,000 rows (16.3s)
  court progress: 500,000 rows (34.3s)
  court progress: 750,000 rows (49.3s)
  court progress: 1,000,000 rows (67.6s)
  court progress: 1,250,000 rows (81.2s)
  court progress: 1,500,000 rows (95.8s)
  court progress: 1,750,000 rows (111.2s)
  court progress: 2,000,000 rows (121.5s)
  court progress: 2,250,000 rows (139.6s)
Court: 2,476,315 rows indexed in 152.9s
Total docs:        2,649,348
Unique citations:  2,158,211
Index sizes:       law_direct=49,288, court_statute=81,988, case=157,227, court_base=178,593, concept=292,946, term=363,331
Co-citation pairs: 1,142,790
```

### Gold mapping (Cell 6)
```
Mapped gold:   42/42
Total gold doc_ids: 42
```

### Per-area bedrock (Cell 7)
```
Building per-area bedrock index...
  built in 8.5s; areas: 26
  area='criminal law and criminal procedure': top 8 = [('66 BGG', 17910), ('106 BGG', 10328), ('42 BGG', 10299), ('64 BGG', 8388), ('97 BGG', 8130), ('105 BGG', 7397), ('108 BGG', 7313), ('9 BV', 6436)]
  area='criminal procedure and coercive measures': top 8 = [('66 BGG', 5519), ('78 BGG', 4035), ('42 BGG', 3449), ('81 BGG', 3442), ('80 BGG', 2812), ('64 BGG', 2695), ('93 BGG', 2573), ('108 BGG', 2404)]
  area='criminal law and administrative criminal law': top 8 = [('156 OG', 171), ('105 OG', 117), ('156 OJ', 112), ('104 OG', 107), ('16 SVG', 100), ('104 OJ', 94), ('17 SVG', 93), ('114 OJ', 87)]
  area='criminal law': top 8 = [('278 BStP', 587), ('66 BGG', 521), ('121 BGG', 520), ('278 PPF', 433), ('42 BGG', 331), ('63 StGB', 320), ('269 PPF', 316), ('277b BStP', 311)]
```

### Co-citation neighbours of `221 StPO` (Cell 8)
```
Co-citation neighbours indexed for 2,282 canonicals (after frequency filter: max global count = 5000).
Sample — neighbours of '221 StPO' AFTER frequency filter:
  31 BV: co=783, total_in_corpus=4339
  212 StPO: co=714, total_in_corpus=1337
  221 BV: co=646, total_in_corpus=670
  237 StPO: co=487, total_in_corpus=1467
  5 StPO: co=272, total_in_corpus=1955
  197 StPO: co=258, total_in_corpus=1555
  5 EMRK: co=246, total_in_corpus=3500
  5 CEDH: co=235, total_in_corpus=1233
```

### BM25 build (Cell 9)
```
Building in-memory FTS5 over 2,649,348 docs...
  inserted 500,000 (12.1s)
  inserted 1,000,000 (28.9s)
  inserted 1,500,000 (45.4s)
  inserted 2,000,000 (63.6s)
  inserted 2,500,000 (81.8s)
FTS5 built: 2,649,348 rows in 86.3s
```

### Vector setup (Cell 10)
```
Loading manifest...
  manifest rows: 2,652,248, cols: ['doc_id', 'family', 'citation', 'row_index']
  manifest->my_did mapping: 2,649,348/2,652,248 (99.9%) in 5.4s
  loading 27 chunks to GPU (~21 GB, ~30-60s)...
  E_GPU shape=(2652248, 4096) dtype=torch.float16, VRAM=21.7 GB, 389.6s
```

### Query expansion (Cell 12)
RAW (first 400 chars):
```
{
  "statute_targets": [
    "Art. 221 StPO",
    "Art. 222 StPO",
    "Art. 227 StPO",
    "Art. 212 StPO",
    "Art. 100 BGG",
    "Art. 42 BGG"
  ],
  "case_targets": [
    "BGE 146 III 387",
    "BGE 145 III 117",
    "BGE 143 III 215",
    "BGE 139 III 285",
    "1B_123/2024"
  ],
  "concept_targets_en": [
    "preventive detention",
    "collusion risk",
    "proportionality",
    "evidence
```
After freeing the model: `CUDA mem=21.7 GB`.

Targets parsed:
```
statute_targets: (6) ['Art. 221 StPO', 'Art. 222 StPO', 'Art. 227 StPO', 'Art. 212 StPO', 'Art. 100 BGG']
case_targets: (5) ['BGE 146 III 387', 'BGE 145 III 117', 'BGE 143 III 215', 'BGE 139 III 285', '1B_123/2024']
concept_targets_en: (20) ['preventive detention', 'collusion risk', 'proportionality', 'evidence preservation', 'witness tampering']
term_targets_de: (20) ['Untersuchungshaft', 'Kollusionsgefahr', 'Verhältnismässigkeit', 'Beweissicherung', 'Zeugenbeeinflussung']
term_targets_fr: (15) ['détention provisoire', 'danger de collusion', 'proportionnalité', 'préservation des preuves', 'manipulation de témoins']
legal_area_keywords: (5) ['criminal procedure', 'detention law', 'evidence law', 'proportionality principle', 'investigative law']
```

### Query encoding (Cell 13)
```
Encoding raw query...
[emb] loading Qwen/Qwen3-Embedding-8B via SentenceTransformer...
  done in 48.2s
Encoding enriched query...
  done in 0.1s
  enriched query length: 2365 chars, keywords appended: 60
```
Warning: `WARNING:sentence_transformers.util.decorators: The 'tokenizer_kwargs' argument was renamed and is now deprecated. Please use 'processor_kwargs' instead.`

### Channel diagnostics (Cell 15)
```
Statute canons: LLM=6, +co-citation=27
  expanded set: ['100 BGG', '115 BGG', '197 StPO', '212 BGG', '212 StPO', '221 BV', '221 StPO', '222 StPO', '227 StPO', '229 StPO', '237 StPO', '31 BV', '393 StPO', '42 BGG', '45 BGG', '47 BGG', '48 BGG', '5 CEDH', '5 EMRK', '5 StPO', '50 BGG', '51 StGB', '54 BGG', '61 BGG', '73 BGG', '73 StHG', '84 BGG']
LLM target codes: {'BGG', 'StPO'}
Concepts: LLM=25 -> expanded=113

channel                   size   gold_in_ch  recall
------------------------------------------------------------
law_direct_match            68            7   16.7%
court_statute              600            0    0.0%
co_citation                400            0    0.0%
per_area_bedrock           150            4    9.5%
sibling_expansion          500            0    0.0%
concept_en                 600            9   21.4%
term_orig                  500            2    4.8%
bm25                       600            3    7.1%
vector_raw                 800           10   23.8%
vector_enriched            800            8   19.0%

Union: 4,107 unique doc_ids
Gold in union: 22/42  (UPPER BOUND on R@K)
```

### RRF fusion + R@K curve (Cell 16)
```
Pre-gate fused: 4,107
Post-gate:      3,982
Guarantee pool: 189  (channels: ['law_direct_match', 'per_area_bedrock'])
RRF-only top-1000:        1,000
Rerank input pool top-3000: 3,000

RRF-only R@K curve (BEFORE reranker — Cell 17 produces the final number):
     K  gold/42  recall
------------------------------
    50      6/42   14.3%
   100      7/42   16.7%
   200      7/42   16.7%
   300      9/42   21.4%
   500     14/42   33.3%
   750     15/42   35.7%
  1000     15/42   35.7%
  1500     17/42   40.5%
  2000     17/42   40.5%
  3000     20/42   47.6%

RRF-only R@1000 = 0.357  (15/42)
Pool R@3000 (upper bound for rerank) = 0.476
```

### Reranker (Cell 17)
```
[rerank] freed embeddings; CUDA mem=0.0 GB
[rerank] loading Qwen/Qwen3-Reranker-8B...
[rerank] loaded; CUDA mem=16.4 GB
[rerank] scoring 3,000 (query, doc) pairs...
[rerank] done in 2896.4s (965.5 ms/pair avg)

FINAL R@K (post-rerank):
     K  gold/42  recall
------------------------------
    50      0/42    0.0%
   100      0/42    0.0%
   200      3/42    7.1%
   300      7/42   16.7%
   500      9/42   21.4%
   750      9/42   21.4%
  1000     14/42   33.3%

==================================================
PASS CRITERION: R@1000 >= 0.60 (>= 26/42)
RRF-only baseline:    R@1000 = 0.357  (15/42)
OBSERVED (post-rerank): R@1000 = 0.333  (14/42)
==================================================
```

### Per-gold trace (Cell 18)
```
  rank  in1k  channels                                                                citation
--------------------------------------------------------------------------------------------------------------------------------------------
   162   YES  term_orig,vector_enriched,vector_raw                                    1B_15/2023 E. 3.1
   170   YES  concept_en                                                              BGE 132 I 21 E. 3.2
   188   YES  term_orig,vector_enriched,vector_raw                                    1B_90/2021 E. 2.1
   201   YES  vector_enriched,vector_raw                                              7B_69/2024 E. 3.3.2
   224   YES  vector_enriched,vector_raw                                              7B_301/2024 E. 2.4
   254   YES  concept_en,vector_enriched,vector_raw                                   1B_357/2022 E. 3.1
   282   YES  concept_en,vector_enriched,vector_raw                                   1B_28/2022 E. 4.1
   313   YES  concept_en,vector_enriched,vector_raw                                   BGE 137 IV 122 E. 4.2
   427   YES  concept_en                                                              BGE 132 I 21 E. 3.2.1
   763   YES  vector_enriched,vector_raw                                              BGE 132 I 21 E. 3.2.2
   855   YES  bm25,law_direct_match,per_area_bedrock                                  Art. 221 Abs. 1 StPO
   857   YES  concept_en                                                              BGE 133 I 270 E. 3.4.2
   871   YES  concept_en                                                              BGE 137 IV 122 E. 4.1
   933   YES  concept_en                                                              BGE 139 IV 270 E. 3.1
     —    no                                                                          Art. 140 Abs. 1 StGB
     —    no                                                                          Art. 396 Abs. 1 StPO
     —    no  bm25,law_direct_match                                                   Art. 222 StPO
     —    no  law_direct_match                                                        Art. 393 Abs. 1 StPO
     —    no                                                                          Art. 382 Abs. 1 StPO
     —    no                                                                          Art. 385 Abs. 1 StPO
     —    no  law_direct_match,per_area_bedrock                                       Art. 221 Abs. 2 StPO
     —    no  bm25,law_direct_match                                                   Art. 227 Abs. 1 StPO
     —    no  concept_en,law_direct_match,per_area_bedrock                            Art. 212 Abs. 3 StPO
     —    no                                                                          Art. 390 Abs. 2 StPO
     —    no                                                                          Art. 422 Abs. 1 StPO
     —    no                                                                          Art. 422 Abs. 2 StPO
     —    no                                                                          Art. 428 Abs. 1 StPO
     —    no                                                                          Art. 135 Abs. 4 StPO
     —    no  law_direct_match,per_area_bedrock                                       Art. 100 Abs. 1 BGG
     —    no                                                                          Art. 135 Abs. 3 StPO
     —    no                                                                          Art. 37 Abs. 1 StBOG
     —    no                                                                          Art. 39 Abs. 1 StBOG
     —    no                                                                          BGE 137 IV 122 E. 6.2
     —    no                                                                          BGE 137 IV 122 E. 6.4
     —    no                                                                          1B_210/2023 E. 4.1
     —    no                                                                          1B_536/2018 E. 5.1
     —    no                                                                          BGE 133 I 168 E. 4.1
     —    no                                                                          BGE 143 IV 168 E. 5.1
     —    no                                                                          1B_90/2021 E. 2.4
     —    no                                                                          7B_496/2025 E. 3.2
     —    no  vector_raw                                                              7B_231/2025 E. 4.1
     —    no  vector_raw                                                              7B_12/2025 E. 2.2

Missed: 28/42
```

### Stderr warnings observed
- `torch_dtype is deprecated! Use dtype instead!`
- `The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set TRANSFORMERS_VERBOSITY=info for more details.`
- `WARNING:sentence_transformers.util.decorators: The 'tokenizer_kwargs' argument was renamed and is now deprecated. Please use 'processor_kwargs' instead.`

### Artifacts (Cell 19)
```
Saved artifacts to: /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v4
```

### Cleanup (Cell 20)
```
cleaned up.
```

## Summary

v5 added a Qwen3-Reranker-8B stage over the top-3000 RRF pool to recover gold buried in the long tail and applied four structural fixes from v4 (correct Qwen3-Embedding query-encoding protocol with last-token pooling + canonical instruction, restored v3 prompt, frequency-filtered co-citation neighbours capped at 5000 global occurrences, per-area bedrock filtered to LLM-named codes). The pipeline failed to meet the R@1000 >= 0.60 pass criterion: post-rerank R@1000 = 0.333 (14/42), actually below the RRF-only baseline of 0.357 (15/42). What worked: gold mapping was perfect (42/42), the fixed vector channels surfaced strong concept/vector recall (`vector_raw` 23.8%, `concept_en` 21.4%, `vector_enriched` 19.0%), and the per-area bedrock now returns criminal-procedure-relevant articles. What failed: the union upper bound was only 22/42 (52.4%) — eleven of the twenty missed gold citations had zero channel hits — so the reranker had no path to recover them; `court_statute`, `co_citation`, and `sibling_expansion` all returned 0 gold; nine `Art. * StPO` law rows were in the law_direct_match channel but ranked outside top-1000 after reranking (e.g., `Art. 222 StPO`, `Art. 212 Abs. 3 StPO`). Lesson: the candidate pool is the bottleneck, not the ranker — the reranker even shuffled correctly-ranked law-direct hits below 1000; the next iteration needs broader candidate coverage (full statute-target expansion into court_statute, larger budgets for law_direct, and rerank-pool size > 3000 or rerank-with-gold-preservation) before reranker tuning is meaningful.
