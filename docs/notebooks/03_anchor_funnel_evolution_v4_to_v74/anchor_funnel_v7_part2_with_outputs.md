# anchor_funnel_v7_part2_with_outputs.ipynb

**Path:** `notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v7_part2_with_outputs.ipynb`

Production diagnostic build of the v7 anchor-funnel retrieval pipeline, run over all 10 val queries. Adds a citation graph (4-layer, 23.65M edges in source DB) with forward / reverse / 2-hop channels, fixes the v6 sibling-budget bug, and includes a per-gold Phase 11 diagnosis loop on val_001. Target: `R@1000 >= 0.60` for val_001 (>= 26/42 gold); stretch >= 0.90.

## Configuration

### Runtime / hardware

- Python 3.12.13
- PyTorch 2.10.0+cu128
- GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB VRAM
- Colab (Google Drive mounted at `/content/drive`)

### Models

- Query expansion: `Qwen/Qwen3-32B` (bf16, ~65 GB VRAM, loaded once, freed after batch)
- Query embedding: `Qwen/Qwen3-Embedding-8B` via `sentence_transformers.SentenceTransformer` (`prompt_name="query"`, normalize_embeddings=True; ~16 GB VRAM)
- Corpus embeddings: pre-computed Qwen3-Embedding-8B vectors, 4096-dim fp16, loaded as a single `E_GPU` tensor (~22 GB sustained)
- Generation: `do_sample=False`, `max_new_tokens=1024`, `enable_thinking=False`, `pad_token_id=eos_token_id`

### Libraries

`json`, `re`, `time`, `gc`, `math`, `statistics`, `pathlib`, `collections.defaultdict/Counter`, `pandas`, `numpy`, `torch`, `transformers.AutoTokenizer/AutoModelForCausalLM`, `sentence_transformers.SentenceTransformer`, `sqlite3` (FTS5 in-memory + graph DB read).

### CONFIG dict (verbatim)

| Knob | Value | Note |
|---|---|---|
| `topk_final` | 1000 | final pool size |
| `budget_law_direct` | None | uncapped |
| `budget_court_statute` | 3000 | v7.1 lift (was 600) |
| `budget_concept` | 600 | |
| `budget_term` | 500 | |
| `budget_per_area` | 150 | |
| `budget_co_citation` | 1000 | v7.1 lift (was 400) |
| `budget_bm25` | 600 | |
| `budget_vector` | 800 | raw query |
| `budget_vector_enriched` | 800 | |
| `budget_backprop` | 400 | |
| `budget_sibling` | 2000 | v6 had 500 (non-deterministic slice) |
| `budget_graph_forward` | 1500 | |
| `budget_graph_reverse` | 1000 | |
| `enable_graph_2hop` | True | v7.1 enabled |
| `budget_graph_2hop` | 1500 | |
| `rrf_k` | 60 | RRF denominator |
| `guarantee_channels` | `["law_direct_match","per_area_bedrock","statute_backprop","graph_forward","sibling_expansion"]` | round-robin merged |
| `guarantee_per_channel` | 200 | 5×200=1000=topk_final |
| `per_area_top_n` | 100 | |
| `co_citation_top_k_per_target` | 8 | |
| `co_citation_min_co_count` | 50 | |
| `co_citation_max_neighbour_count` | 5000 | filter globally-common neighbours |
| `concept_substring_top_k` | 6 | |
| `bm25_max_query_terms` | 60 | cap |
| `bm25_min_token_len` | 3 | |
| `vector_emb_model` | `"Qwen/Qwen3-Embedding-8B"` | |
| `vector_topk` | 800 | |
| `qwen_query_model` | `"Qwen/Qwen3-32B"` | |
| `qwen_max_new_tokens` | 1024 | |
| `enhance_top_k_codes` | 5 | |
| `enhance_repeat_count` | 5 | |
| `enhance_min_idf` | 1.0 | |
| `noise_paragraph_roles` | `{"notification","header","empty","metadata"}` | neg-gate |
| `lowercase_concepts` | True | |
| `lowercase_terms` | True | |

### Constants

- `CODE_ALIAS`: `{CPP→StPO, CP→StGB, CC→ZGB, CO→OR, LTF→BGG, LACI→AVIG, LAA→UVG, LP→SchKG, LDIP→IPRG, Cst→BV, Cst.→BV, STPO→StPO, OBG→OR}`
- `LEGAL_AREA_DEFAULT_CODE`: mapping of 10 legal-area phrases to default codes (criminal procedure→StPO, civil law→ZGB, obligations→OR, etc.)
- Regex: `ART_RE = r"art\.?\s*(\d+[a-z]?)"`, `CODE_RE = r"\b([A-Z][A-Za-z]{1,8}\.?)\b"`, `CASE_BGE_RE = r"BGE\s+(\d+)\s+([IVX]+)\s+(\d+)"`, `CASE_DOCKET_RE = r"\b(\d[A-Z]_\d+/\d{4})\b"`

## Data

`DATA_ROOT` was auto-resolved to `/content/drive/MyDrive/swiss_law` (all paths verified `OK`):

- `val_csv`: `/content/drive/MyDrive/swiss_law/data/val.csv` — 10 English queries, gold citations semicolon-delimited (val_001: 42 gold, val_002: 36, val_003: 47, val_004: 10, val_005: 11, val_006: 18, val_007: 19, val_008: 29, val_009: 14, val_010: 25; total 251 gold)
- `law_llm`: `/content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl` — LLM enrichment of laws (173,033 rows after stream)
- `court_v5`: `/content/drive/MyDrive/swiss_law/artifacts_v2/court_authority_cards_v5_unified.jsonl` — court enrichment (2,476,315 rows after stream)
- `emb_dir`: `/content/drive/MyDrive/swiss_law/artifacts/embeddings` — 27 fp16 chunk `.npy` files
- `emb_manifest`: `/content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_manifest.parquet` — 2,652,248 rows; columns `[doc_id, family, citation, row_index]`
- `graph_db`: `/content/drive/MyDrive/swiss_law/data_insights/citation_graph_extracted.sqlite` — citation-edge table (`edges` with `dataset='court_considerations'`); 4 alias passes had been applied locally
- `out_dir`: `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7` — writes per-query artifacts and aggregate summary

Train.csv is never read (constraint). Outputs persisted: `<qid>/final_topk.json`, `<qid>/gold_in_top.json`, `<qid>/targets.json`, `<qid>/summary.json`, `aggregate_summary.json`, `val_001_gold_missed_diagnosis.json`, `config.json`.

## Pipeline

Twelve phases, single-model-on-GPU memory contract. Each model is loaded, used, and explicitly freed (`del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()`) before the next loads.

### Stage 1 — Setup (Phase 1)

1. Print Python / PyTorch / GPU.
2. Mount Drive (Colab) or skip.
3. Resolve `DATA_ROOT` from candidate list (Drive / Kaggle / local Windows / cwd); build `PATHS` dict; check existence and set `EMB_AVAILABLE`, `GRAPH_AVAILABLE`.
4. Print `CONFIG` dict.
5. Read `val.csv`, parse semicolon-delimited gold per query → `ALL_QUERIES` list of dicts `{query_id, query, gold}`.

### Stage 2 — Build all corpus indexes in one streaming pass (Phase 2)

Stream both JSONL files; build in-memory dicts:

- `cit_to_doc_ids[cit] -> list[doc_id]`
- `doc_meta[did] -> {citation, family, court_base, paragraph_role, is_notification_paragraph}`
- `idx_law_direct[canonical] -> set[law_did]` (from `statute_anchor_canonical(cit)`)
- `idx_court_statute[canonical] -> set[court_did]` (from `canonicalize_row_anchors(rag.statute_anchors, legal_area_static)`)
- `idx_case_anchor[case_canon] -> set[did]`
- `idx_court_base[court_base] -> set[did]` (siblings)
- `idx_concept_en[token] -> set[did]` (from `concepts_en` + `terms_de_to_en.en`)
- `idx_term_orig[token] -> set[did]` (DE+FR terms_original)
- `legal_area_per_doc[did] -> str`
- `search_text[did] -> str` (concat of citation + english_summary + legal_rule + legal_question + applicability_conditions + concepts + terms, max 2000 chars)
- `co_citation_pairs[(canon_a, canon_b)] -> count` (unordered within court row)
- `tlf[token][code] -> count` and `token_doc_count[token]` (only over law rows where canonical code known)
- `doc_statute_anchors[did] -> set[canonical]` (court rows only)

Then map val_001 gold to corpus doc_ids as a sanity check.

### Stage 3 — Citation graph load (Phase 3)

Build `_cit_to_did = {cit: dids[0]}`. Open `citation_graph_extracted.sqlite`, iterate `SELECT source, target FROM edges WHERE dataset='court_considerations'`, map both endpoints to doc_ids, skip if either missing or src==tgt, append to `idx_graph_out[s]` and `idx_graph_in[t]`.

### Stage 4 — Per-area bedrock + co-citation neighbours (Phase 4)

- `per_area_canon_count[area] = Counter(canonical_anchors_in_area)` from `legal_area_per_doc` + `doc_statute_anchors`.
- For each `(a,b) in co_citation_pairs` with `cnt >= co_citation_min_co_count`: append `(b, cnt)` to `co_neighbours[a]` and vice versa, then filter neighbours whose global `canon_count` exceeds `co_citation_max_neighbour_count` (5000) and keep top `2 × top_k_per_target` per source.

### Stage 5 — BM25 (Phase 5)

In-memory SQLite FTS5 (`tokenize='unicode61 remove_diacritics 2'`) over `search_text[did]` with `journal_mode=MEMORY`, `synchronous=OFF`; batch inserts of 50,000 rows. Define:

- `_enhance_codes(text)`: tokenize, compute per-token IDF over corpus, score each `tlf[token][code]` by `cnt * idf`, return top `enhance_top_k_codes` codes.
- `bm25_search(query_text, k)`: enrich query by repeating boosted codes × `enhance_repeat_count`, cap query tokens to `bm25_max_query_terms`, FTS5 `MATCH` with `OR`-joined terms, return flipped-sign BM25 scores.

### Stage 6 — Query expansion with Qwen3-32B (Phase 6)

Load Qwen3-32B once (bf16, `device_map="auto"`). For each of the 10 queries: build chat prompt (system + structured JSON schema asking for 10-25 items per category; few-shot example for detention/StPO), call `apply_chat_template(... return_dict=True, enable_thinking=False)`, run `generate(... do_sample=False)`. Parse JSON; store per-query `targets` in `ALL_TARGETS[qid]` with keys `statute_targets`, `case_targets`, `concept_targets_en`, `term_targets_de`, `term_targets_fr`, `legal_area_keywords`. Free GPU after the loop.

### Stage 7 — Encode queries with Qwen3-Embedding-8B (Phase 7)

Load Qwen3-Embedding-8B. For each query, encode `q_emb_raw` (verbatim) and `q_emb_enriched = query + " " + first 60 of union(term_de, term_fr, concept_en)`; move both to CPU. Cache `ALL_Q_EMB_RAW`, `ALL_Q_EMB_ENRICHED`, `ALL_ENRICHED_TEXT`. Free GPU.

### Stage 8 — Vector setup (Phase 8)

Read manifest parquet. For each row `(citation, family, row_index)`, look up `cit_to_doc_ids[cit]`, match by `family`, build `row_for_did[did]=ridx` and `my_did_for_row[ridx]=did`. Load 27 `qwen3_8b_unified_chunk*.npy` files to GPU and `torch.cat` to `E_GPU`. Define `vector_search(q_emb, k)` doing on-GPU normalize + matmul + topk.

### Stage 9 — Channels and per-query retrieval loop (Phase 9)

Channel functions:

- `channel_law_direct(canon_set, idx, budget)`: Counter over `idx_law_direct[canon]`.
- `channel_court_statute(statute_canons, idx, budget)`: Counter over `idx_court_statute[canon]`.
- `channel_sibling(seed, idx_court_base, doc_meta, budget)`: v7.1 — rank siblings by **seed-frequency** (count distinct seeds sharing each `court_base`), not alphabetical; emit `Counter.most_common(budget)`.
- `channel_graph_forward(seed, idx_graph_out, budget)`: Counter increments by edge multiplicity, drop seed itself.
- `channel_graph_reverse(seed, idx_graph_in, budget)`.
- `channel_graph_2hop(seed, idx_graph_out, budget)`: expand intermediates from seed, then forward again, exclude both seed and intermediates.
- `expand_concepts_strict(llm_concepts, corpus_concept_keys, top_k)`: lower / strip / substring-match against corpus concept keys.
- `channel_concept(expanded, idx, budget)`, `channel_term(targets, idx, budget)` (DE + FR).
- `channel_per_area_bedrock(legal_area_kws, statute_target_codes, per_area_canon_count, idx_law_direct, budget)`: select areas by substring match, take top `per_area_top_n` canons per area scored by max-area-count, filter by `statute_target_codes`, emit law rows.
- `channel_co_citation(targets, co_neighbours, idx_law_direct, idx_court_statute, budget)`.
- `channel_statute_backprop(seed_court_dids, doc_statute_anchors, idx_law_direct, budget)`: each caught court row votes for its anchored statutes' law rows.
- `channel_vector(q_emb, k)`: calls `vector_search`.

Fusion:

- `rrf_fuse(channels, k=60)`: `score[did] += 1/(k + rank + 1)`.
- `apply_neg_gate(doc_ids, doc_meta, noise_roles)`: drop `is_notification_paragraph` and rows whose `paragraph_role` ∈ `noise_paragraph_roles`.
- `round_robin_guarantee(channels_by_name, guarantee_names, per_channel_cap=200, total_cap=1000)`: round-robin pop from each guarantee channel iterator until `total_cap` reached, dedup via `seen` set.

Per-query loop (10 queries): canonicalize LLM statutes, expand with co-citation neighbours into `co_expanded_canons`, expand concepts, run all 14 channels, build a topical seed = union of court-family hits across channels (sibling/graph excluded). Run `channel_sibling`, extend seed with sibling hits, then run `channel_graph_forward`, `channel_graph_reverse`, `channel_graph_2hop`. Combine guarantee output with RRF tail (apply neg-gate first). Record per-channel recall, union ceiling, R@1000, RRF scores, ranked dids into `PER_QUERY[qid]`.

### Stage 10 — Aggregate (Phase 10)

Macro mean R@1000, micro R@1000, mean union ceiling. Per-channel mean recall with stdev, sorted. Identify worst / best query.

### Stage 11 — Per-gold diagnosis on val_001 (Phase 11)

Helpers: `channel_rank`, `channel_score`, `_tokens`, `gold_vector_cos` (cosine of query embedding vs E_GPU row), `concepts_pointing_to`, `terms_pointing_to`, `graph_paths_to`, `graph_paths_from`, `sibling_diagnosis`.

For each missed gold doc: print citation, family, court_base, paragraph_role, search_text head; for each channel print HIT (with rank) or MISS (with probe-specific reason: BM25 token overlap, vector cosine, expanded-concepts hitting row, seed-edges, sibling base coverage, canon membership, anchor overlap, etc.). Compute root cause (`RRF_RANK_TOO_LOW`, `NO_CHANNEL_HIT`, `GATED_OUT`) and a fix suggestion. Aggregate root-cause counter. Cross-query brief: per-qid caught / missed / union / best-channel; per-qid `in_union - in_final` gap.

### Stage 12 — Save + cleanup (Phase 12)

Write per-query `final_topk.json` / `gold_in_top.json` / `targets.json` / `summary.json` under `out_dir/<qid>/`, plus `aggregate_summary.json`, `val_001_gold_missed_diagnosis.json`, `config.json`. Delete `E_GPU`, gc, empty CUDA cache.

## Results

### Phase 1 — Setup outputs

```
Python: 3.12.13
PyTorch: 2.10.0+cu128
  GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB
Mounted at /content/drive
DATA_ROOT = /content/drive/MyDrive/swiss_law
  OK       val_csv        /content/drive/MyDrive/swiss_law/data/val.csv
  OK       law_llm        /content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl
  OK       court_v5       /content/drive/MyDrive/swiss_law/artifacts_v2/court_authority_cards_v5_unified.jsonl
  OK       emb_dir        /content/drive/MyDrive/swiss_law/artifacts/embeddings
  OK       emb_manifest   /content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_manifest.parquet
  OK       graph_db       /content/drive/MyDrive/swiss_law/data_insights/citation_graph_extracted.sqlite
  EMB_AVAILABLE   = True
  GRAPH_AVAILABLE = True
```

val.csv parsed: 10 queries; gold counts per query 42, 36, 47, 10, 11, 18, 19, 29, 14, 25.

### Phase 2 — Index build

```
Law: 173,033 rows indexed in 10.0s
Token->code association: 91,173 tokens, avg codes/token = 11.7
  court progress: 500,000 rows (27.5s)
  court progress: 1,000,000 rows (48.1s)
  court progress: 1,500,000 rows (70.0s)
  court progress: 2,000,000 rows (92.3s)
Court: 2,476,315 rows indexed in 115.9s
Total docs:        2,649,348
Unique citations:  2,158,211
Index sizes:       law_direct=49,288, court_statute=81,988, case=157,227, court_base=178,593, concept=292,946, term=363,331
Co-citation pairs: 1,142,790
```

val_001 gold mapping: `Mapped gold: 42/42`, `Total gold doc_ids: 42`.

### Phase 3 — Graph load

```
Built citation->doc_id map (2,158,211 entries)
Graph: 20,494,436 edges loaded, 3,155,263 skipped (cit not in corpus)
Graph: out-degree avg = 27.2, in-degree avg = 14.0
Graph: load time 100.2s
```

### Phase 4 — Per-area bedrock + co-citation

```
Building per-area bedrock index...
  built in 2.0s; areas: 26
  area='constitutional and public law': top 8 = [('66 BGG', 13212), ('29 BV', 12189), ('89 BGG', 11479), ('82 BGG', 11464), ('42 BGG', 10836), ('9 BV', 9650), ('106 BGG', 9625), ('68 BGG', 8649)]
  area='administrative, tax, migration, and regulatory law': top 8 = [('42 BGG', 20301), ('106 BGG', 19460), ('66 BGG', 17471), ('105 BGG', 16033), ('95 BGG', 15722), ('83 BGG', 15231), ('68 BGG', 14674), ('89 BGG', 12187)]
  area='civil law': top 8 = [('63 OJ', 3189), ('8 ZGB', 2740), ('55 OJ', 2518), ('64 OJ', 2235), ('55 OG', 2085), ('63 OG', 1906), ('159 OG', 1876), ('9 BV', 1875)]
  area='criminal law and criminal procedure': top 8 = [('66 BGG', 28149), ('42 BGG', 20746), ('106 BGG', 19537), ('64 BGG', 13843), ('108 BGG', 12101), ('97 BGG', 12075), ('105 BGG', 11759), ('81 BGG', 11102)]
```

Co-citation neighbours indexed for 2,282 canonicals (after frequency filter ≤ 5000). Sample neighbours of `221 StPO`: `31 BV (co=783, total=4339)`, `212 StPO (714/1337)`, `221 BV (646/670)`, `237 StPO (487/1467)`, `5 StPO (272/1955)`, `197 StPO (258/1555)`, `5 EMRK (246/3500)`, `5 CEDH (235/1233)`.

### Phase 5 — BM25 build

```
Building in-memory FTS5 over 2,649,348 docs...
  inserted 500,000 (11.5s) ... 2,500,000 (77.8s)
FTS5 built: 2,649,348 rows in 84.0s
```

### Phase 6 — Query expansion

Qwen3-32B load: 206.7s.

Per-query target counts (statute / concept / term_de / term_fr / area):

```
val_001:  0 / 0 / 0 / 0 / 0
val_002: 20 / 26 / 26 / 26 / 6
val_003:  0 / 0 / 0 / 0 / 0
val_004: 20 / 25 / 24 / 20 / 6
val_005:  0 / 0 / 0 / 0 / 0
val_006: 20 / 26 / 25 / 22 / 6
val_007:  0 / 0 / 0 / 0 / 0
val_008:  0 / 0 / 0 / 0 / 0
val_009:  0 / 0 / 0 / 0 / 0
val_010:  0 / 0 / 0 / 0 / 0
```

JSON parse succeeded for only 3 queries (val_002, val_004, val_006); the other 7 returned all-empty target lists (JSON parse failed silently → empty defaults). Free: `VRAM used=0.01 GB, free=94.96 GB`.

### Phase 7 — Query embedding

Qwen3-Embedding-8B load: 44.6s. Enriched lengths / keywords appended:

```
val_001: 1069 chars, 0 keywords     val_006: 2677, 60
val_002: 3071,        60            val_007: 1743,  0
val_003: 1678,         0            val_008: 1528,  0
val_004: 2351,        60            val_009: 1117,  0
val_005: 1582,         0            val_010: 1696,  0
```

Free: `VRAM used=0.01 GB, free=94.96 GB`.

### Phase 8 — Vector setup

```
manifest rows: 2,652,248, cols: ['doc_id', 'family', 'citation', 'row_index']
manifest->my_did mapping: 2,649,348/2,652,248 (99.9%) in 39.4s
loading 27 chunks to GPU (~21 GB)...
E_GPU shape=(2652248, 4096) dtype=torch.float16, VRAM used=20.2 GB, free=74.7 GB, load 251.4s
```

### Phase 9 — Per-query retrieval

Per-query R@1000 and per-channel recall (channel size shown alongside):

```
val_001 (42 gold)  union=0.762  R@1000=0.333
  law_direct=0/0.000  court_statute=0/0.000  co_citation=0/0.000  per_area=0/0.000
  statute_backprop=400/0.167  sibling=2000/0.190  graph_fwd=1500/0.262
  graph_rev=1000/0.214  graph_2hop=1500/0.024
  concept_en=0/0.000  term_orig=0/0.000  bm25=600/0.048
  vector_raw=800/0.119  vector_enriched=800/0.119

val_002 (36 gold)  union=0.667  R@1000=0.389
  law_direct=0/0.000  court_statute=3000/0.028  co_citation=1000/0.000  per_area=0/0.000
  statute_backprop=400/0.444  sibling=2000/0.056  graph_fwd=1500/0.389
  graph_rev=1000/0.000  graph_2hop=1500/0.028
  concept_en=600/0.000  term_orig=500/0.000  bm25=600/0.000
  vector_raw=800/0.000  vector_enriched=800/0.000

val_003 (47 gold)  union=0.404  R@1000=0.277
  law_direct=0/0.000  court_statute=0/0.000  co_citation=0/0.000  per_area=0/0.000
  statute_backprop=400/0.149  sibling=2000/0.000  graph_fwd=1500/0.213
  graph_rev=1000/0.000  graph_2hop=1500/0.021
  concept_en=0/0.000  term_orig=0/0.000  bm25=600/0.021
  vector_raw=800/0.085  vector_enriched=800/0.085

val_004 (10 gold)  union=0.800  R@1000=0.700
  law_direct=67/0.000  court_statute=3000/0.000  co_citation=1000/0.000  per_area=128/0.000
  statute_backprop=400/0.600  sibling=2000/0.000  graph_fwd=1500/0.400
  graph_rev=1000/0.000  graph_2hop=1500/0.000
  concept_en=600/0.200  term_orig=500/0.000  bm25=600/0.200
  vector_raw=800/0.400  vector_enriched=800/0.200

val_005 (11 gold)  union=0.818  R@1000=0.455
  law_direct=0/0.000  court_statute=0/0.000  co_citation=0/0.000  per_area=0/0.000
  statute_backprop=400/0.545  sibling=2000/0.000  graph_fwd=1500/0.273
  graph_rev=1000/0.000  graph_2hop=1500/0.182
  concept_en=0/0.000  term_orig=0/0.000  bm25=600/0.091
  vector_raw=800/0.364  vector_enriched=800/0.364

val_006 (18 gold)  union=0.778  R@1000=0.556
  law_direct=96/0.333  court_statute=3000/0.000  co_citation=1000/0.000  per_area=0/0.000
  statute_backprop=400/0.556  sibling=2000/0.000  graph_fwd=1500/0.333
  graph_rev=1000/0.000  graph_2hop=1500/0.000
  concept_en=600/0.111  term_orig=500/0.000  bm25=600/0.000
  vector_raw=800/0.167  vector_enriched=800/0.167

val_007 (19 gold)  union=0.789  R@1000=0.421
  law_direct=0/0.000  court_statute=0/0.000  co_citation=0/0.000  per_area=0/0.000
  statute_backprop=400/0.526  sibling=2000/0.105  graph_fwd=1500/0.368
  graph_rev=1000/0.000  graph_2hop=1500/0.000
  concept_en=0/0.000  term_orig=0/0.000  bm25=600/0.053
  vector_raw=800/0.053  vector_enriched=800/0.053

val_008 (29 gold)  union=0.655  R@1000=0.414
  law_direct=0/0.000  court_statute=0/0.000  co_citation=0/0.000  per_area=0/0.000
  statute_backprop=400/0.483  sibling=2000/0.000  graph_fwd=1500/0.207
  graph_rev=1000/0.000  graph_2hop=1500/0.138
  concept_en=0/0.000  term_orig=0/0.000  bm25=600/0.000
  vector_raw=800/0.103  vector_enriched=800/0.103

val_009 (14 gold)  union=0.643  R@1000=0.500
  law_direct=0/0.000  court_statute=0/0.000  co_citation=0/0.000  per_area=0/0.000
  statute_backprop=400/0.571  sibling=2000/0.000  graph_fwd=1500/0.286
  graph_rev=1000/0.000  graph_2hop=1500/0.071
  concept_en=0/0.000  term_orig=0/0.000  bm25=600/0.000
  vector_raw=800/0.000  vector_enriched=800/0.000

val_010 (25 gold)  union=0.720  R@1000=0.320
  law_direct=0/0.000  court_statute=0/0.000  co_citation=0/0.000  per_area=0/0.000
  statute_backprop=400/0.360  sibling=2000/0.040  graph_fwd=1500/0.280
  graph_rev=1000/0.000  graph_2hop=1500/0.000
  concept_en=0/0.000  term_orig=0/0.000  bm25=600/0.000
  vector_raw=800/0.240  vector_enriched=800/0.240
```

### Phase 10 — Aggregate

Per-query R@1000:

```
query     gold  union_ceil  R@1000
val_001     42       0.762   0.333  (14/42)
val_002     36       0.667   0.389  (14/36)
val_003     47       0.404   0.277  (13/47)
val_004     10       0.800   0.700  (7/10)
val_005     11       0.818   0.455  (5/11)
val_006     18       0.778   0.556  (10/18)
val_007     19       0.789   0.421  (8/19)
val_008     29       0.655   0.414  (12/29)
val_009     14       0.643   0.500  (7/14)
val_010     25       0.720   0.320  (8/25)

  Macro mean R@1000:     0.436   (10 queries)
  Micro R@1000:          0.390   (98/251 total gold)
  Macro mean union ceil: 0.704
```

Per-channel mean recall (ranked):

```
channel                 mean recall    std
statute_backprop           0.440      0.156
graph_forward              0.301      0.065
vector_raw                 0.153      0.134
vector_enriched            0.133      0.107
graph_2hop                 0.046      0.061
bm25                       0.041      0.061
sibling_expansion          0.039      0.061
law_direct_match           0.033      0.100
concept_en                 0.031      0.065
graph_reverse              0.021      0.064
court_statute              0.003      0.008
co_citation                0.000      0.000
per_area_bedrock           0.000      0.000
term_orig                  0.000      0.000

Worst query: val_003 (R@1000 = 0.277, union_ceiling = 0.404)
Best  query: val_004 (R@1000 = 0.700, union_ceiling = 0.800)
```

### Phase 11 — val_001 diagnosis

`Query token count (post-enhance): 104`. `R@1000 = 14/42 = 33.3%`; 28 gold missed.

Root-cause distribution for val_001:

```
RRF_RANK_TOO_LOW        14
NO_CHANNEL_HIT          10
GATED_OUT                4
```

Sample per-gold traces:

- `1B_210/2023 E. 4.1` (court, court_base=1B_210/2023, paragraph_role=legal_standard): only `graph_reverse` HIT rank 285; RRF score 0.00289 rank 2288 → RRF_RANK_TOO_LOW.
- `BGE 133 I 168 E. 4.1`: 2 channels hit, dropped by neg-gate (paragraph_role=legal_standard) → GATED_OUT.
- `BGE 132 I 21 E. 3.2.2`: 2 channels hit, dropped by neg-gate (paragraph_role=facts) → GATED_OUT.
- `Art. 428 Abs. 1 StPO` (law): canonical=`428 StPO` not in `co_expanded_canons`, seed-courts-citing=0, vector cos_raw=0.3191 / enr=0.3220 → NO_CHANNEL_HIT.

Channel-level hit rate on val_001's 28 missed gold (these had the gold in their candidate pool but RRF or gating dropped it below rank 1000):

```
law_direct_match      0       graph_reverse         5
court_statute         0       graph_2hop            1
co_citation           0       concept_en            0
per_area_bedrock      0       term_orig             0
statute_backprop      0       bm25                  0
sibling_expansion     7       vector_raw            4
graph_forward         3       vector_enriched       4
```

Cross-query brief (Phase 11.4):

```
qid       gold caught missed union     best-channel-for-query
val_001     42     14     28  0.762    graph_forward (recall 0.262)
val_002     36     14     22  0.667    statute_backprop (recall 0.444)
val_003     47     13     34  0.404    graph_forward (recall 0.213)
val_004     10      7      3  0.800    statute_backprop (recall 0.600)
val_005     11      5      6  0.818    statute_backprop (recall 0.545)
val_006     18     10      8  0.778    statute_backprop (recall 0.556)
val_007     19      8     11  0.789    statute_backprop (recall 0.526)
val_008     29     12     17  0.655    statute_backprop (recall 0.483)
val_009     14      7      7  0.643    statute_backprop (recall 0.571)
val_010     25      8     17  0.720    statute_backprop (recall 0.360)
```

Gold lost between candidate union and final top-1000 (RRF/gating gap):

```
qid       in_union  in_final  gap
val_001         32        14   18
val_002         24        14   10
val_003         19        13    6
val_004          8         7    1
val_005          9         5    4
val_006         14        10    4
val_007         15         8    7
val_008         19        12    7
val_009          9         7    2
val_010         18         8   10
```

### Phase 12 — Save + cleanup

```
Saved per-query results in /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/<qid>/  (10 query folders)
Aggregate summary at /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/aggregate_summary.json
  mean R@1000 (macro): 0.436
  R@1000        (micro): 0.390
VRAM after cleanup: 0.01 GB
cleaned up.
```

## Summary

The v7 anchor-funnel pipeline ran end-to-end across all 10 val queries on a 95 GB Blackwell GPU, yielding **macro mean R@1000 = 0.436** (micro 0.390, 98/251 gold) versus a mean union ceiling of 0.704 — i.e. roughly 38% of recoverable gold was lost to RRF / negative gating rather than to channel coverage. The pass criterion (R@1000 ≥ 0.60 on val_001) was missed: val_001 hit only 0.333 (14/42); only val_004 cleared 0.60. The two channels that consistently carry weight across queries are **statute_backprop (mean 0.440)** and **graph_forward (mean 0.301)**; vector channels follow at ~0.13–0.15, and `co_citation`, `per_area_bedrock`, and `term_orig` contribute essentially zero on average. A critical execution failure: Qwen3-32B query-expansion JSON parsed for only 3/10 queries (val_002, val_004, val_006) — the other 7 had silently-empty target dicts, which is why statute / concept / term / per-area channels return 0 size for those queries and explains the low ceiling on val_003 (0.404). Phase 11 diagnosis on val_001 attributes the 28 misses to 14 RRF_RANK_TOO_LOW, 10 NO_CHANNEL_HIT, and 4 GATED_OUT (gold dropped because `paragraph_role` was `legal_standard` or `facts`, both inside `noise_paragraph_roles`); lessons for v7.1+ are to harden JSON parsing / retry on empty targets, revisit the neg-gate role list (legal_standard is clearly signal not noise), and promote `graph_reverse` to a guarantee channel since it was the only hit for several val_001 detention-case gold but lived at ranks 285–7400+.
