# pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb

**Path:** `notebooks/02_v75_multiquery_canonical_pool/pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb`

The canonical Anchor-Funnel v7.5 multi-query pool: runs the full anchor-funnel pipeline on every query in `val.csv` with no per-id branching. Final cap `topk_final = 50,000` (post-gate channel union — measures upper bound of channel recall). Macro mean R@K = 0.901 on val (10 queries).

## Configuration

### Models

| Model | Use | Precision | VRAM |
|---|---|---|---|
| `Qwen/Qwen3-32B` | Query expansion (structured targets JSON) + HyDE (multi-aspect Bundesgericht-style answers) | bfloat16 | ~65 GB |
| `Qwen/Qwen3-Embedding-8B` | Query encoding (raw / enriched / HyDE-per-aspect) | fp16 | ~16 GB |
| `E_GPU` corpus embedding | Pre-encoded Qwen3-Embedding-8B corpus tensor (2,652,248 × 4096), brute-force GPU cosine | fp16 | 20.2 GB |

### Hardware

- Python 3.12.13, PyTorch 2.10.0+cu128
- 1 × NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB VRAM
- Mounted: `/content/drive` (Colab)

### CONFIG dict (verbatim from output)

```
topk_final:               50000
budget_law_direct:        None
budget_court_statute:     8000
budget_concept:           3000
budget_term:              2500
budget_per_area:          1500
budget_co_citation:       2500
budget_bm25:              2000
budget_vector:            2000
budget_vector_enriched:   2000
budget_vector_hyde:       2000
budget_backprop:          2000
budget_sibling:           5000
budget_graph_forward:     5000
budget_graph_reverse:     3000
enable_graph_2hop:        False
budget_graph_2hop:        1500
rrf_k:                    60
guarantee_per_channel:    130
code_family_top_k:        8
enable_prf:               True
prf_top_k_for_extraction: 4000
prf_new_statutes_k:       40
prf_new_concepts_k:       60
prf_new_terms_k:          60
per_area_top_n:           1000
co_citation_top_k_per_target:    50
co_citation_min_co_count:        50
co_citation_max_neighbour_count: 50000
concept_substring_top_k:  6
bm25_max_query_terms:     60
bm25_min_token_len:       3
vector_emb_model:         Qwen/Qwen3-Embedding-8B
vector_topk:              800
qwen_query_model:         Qwen/Qwen3-32B
qwen_max_new_tokens:      1024
enhance_top_k_codes:      5
enhance_repeat_count:     5
enhance_min_idf:          1.0
lowercase_concepts:       True
lowercase_terms:          True
noise_paragraph_roles:    {"notification","header","empty","metadata"}
```

### Guarantee channels (round-robin, per_channel_cap=130)

`law_direct_match`, `per_area_bedrock`, `statute_backprop`, `concept_en`, `graph_forward`, `sibling_expansion`, `term_orig` (7 channels × 130 = up to 910 guaranteed slots).

### Channel weights (weighted RRF)

```
statute_backprop:  2.5
graph_forward:     2.0
concept_en:        1.8
court_statute:     1.5
per_area_bedrock:  1.5
vector_hyde:       1.5
term_orig:         1.2
law_direct_match:  1.2
vector_raw:        1.0
vector_enriched:   1.0
sibling_expansion: 1.0
bm25:              0.8
co_citation:       0.7
graph_reverse:     0.5
graph_2hop:        0.0
```

### Substantive paragraph roles (override neg-gate)

`facts`, `reasoning`, `legal_standard`, `application`, `holding`, `citation`, `procedural_history`.

## Data

All paths resolved under `DATA_ROOT = /content/drive/MyDrive/swiss_law`.

| Key | Path | Notes |
|---|---|---|
| `val_csv` | `/content/drive/MyDrive/swiss_law/data/val.csv` | 10 English queries with gold citations |
| `law_llm` | `/content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl` | LLM enrichment of all 175k laws |
| `court_v5` | `/content/drive/MyDrive/swiss_law/artifacts_v2/court_authority_cards_v5_unified.jsonl` | Court enrichment (10.4 GB) |
| `emb_dir` | `/content/drive/MyDrive/swiss_law/artifacts/embeddings` | 27 fp16 chunks (21 GB) |
| `emb_manifest` | `/content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_manifest.parquet` | doc_id ↔ row_index |
| `graph_db` | `/content/drive/MyDrive/swiss_law/data_insights/citation_graph_extracted.sqlite` | 4-layer alias graph (~2.4 GB; 23.65M edges raw) |
| `out_dir` | `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7` | Output directory (note: legacy folder name, holds v7.5 snapshot) |

All flags reported `OK`. `EMB_AVAILABLE = True`, `GRAPH_AVAILABLE = True`.

### Query list

10 val queries. Gold counts: val_001=42, val_002=36, val_003=47, val_004=10, val_005=11, val_006=18, val_007=19, val_008=29, val_009=14, val_010=25 (total 251 gold across 10 queries).

### Corpus stats (after Phase 2 streaming pass)

- Law: 173,033 rows in 12.2s
- Court: 2,476,315 rows in 178.3s
- Total docs: 2,649,348
- Unique citations: 2,158,211
- Index sizes: `law_direct=49,288`, `court_statute=81,988`, `case=157,227`, `court_base=178,593`, `concept=292,946`, `term=363,331`, `term_lemma=521,146`, `term_keys=363,331`
- Co-citation pairs: 1,142,790
- Token→code association (`tlf`): 91,173 tokens, avg 11.7 codes/token

### Gold→doc_id mapping (Phase 2.5)

All 10 queries: gold == mapped (no unmapped citations).

| qid | gold | mapped | doc_ids |
|---|---:|---:|---:|
| val_001 | 42 | 42 | 42 |
| val_002 | 36 | 36 | 38 |
| val_003 | 47 | 47 | 47 |
| val_004 | 10 | 10 | 10 |
| val_005 | 11 | 11 | 11 |
| val_006 | 18 | 18 | 18 |
| val_007 | 19 | 19 | 19 |
| val_008 | 29 | 29 | 30 |
| val_009 | 14 | 14 | 14 |
| val_010 | 25 | 25 | 25 |

### Citation graph load (Phase 3)

- Built citation→doc_id map: 2,158,211 entries
- Graph edges loaded: **20,494,436** (3,155,263 skipped — citation not in corpus); SELECT WHERE `dataset='court_considerations'`
- Out-degree avg = 27.2, in-degree avg = 14.0
- Load time 103.2 s
- `idx_judgment_importance`: 178,593 judgments; median=7, p75=44, max=204,686

### Per-area bedrock + co-citation (Phase 4)

- 26 legal areas indexed (2.1 s)
- `criminal law and criminal procedure` top-8 canons: `66 BGG`, `42 BGG`, `106 BGG`, `64 BGG`, `108 BGG`, `97 BGG`, `105 BGG`, `81 BGG`
- Corpus-derived code-pair statistics: 54,790 directed pairs (0.7 s)
- StPO related codes (top 8): `BGG (50026)`, `BV (34792)`, `StGB (18981)`, `EMRK (6826)`, `Satz (2812)`, `CEDH (2549)`, `OR (1961)`, `ZGB (1123)`
- Co-citation neighbours indexed for 2,282 canonicals
- Sample for `221 StPO` (after frequency filter): `36 BV (915)`, `31 BV (783)`, `10 BV (731)`, `212 StPO (714)`, `221 BV (646)`, `237 StPO (487)`, `5 BV (393)`, `5 StPO (272)`

### BM25 (Phase 5)

- Legacy single-language FTS5: 2,649,348 rows built in 84.5 s
- Per-language FTS5 build: 70.7 s
  - de: 1,593,249 docs
  - fr: 793,023 docs
  - it: 125,583 docs
  - en: 137,493 docs
- Tokenizer: `unicode61 remove_diacritics 2`
- `enhance()`: corpus-derived top-K codes appended `enhance_repeat_count=5` times per code
- Cross-language score normalization: raw BM25 score divided by `log(N_lang + e)` for comparability

### Vector setup (Phase 6)

- Manifest rows: 2,652,248 (cols: `doc_id, family, citation, row_index`)
- Manifest→my_did mapping: 2,649,348/2,652,248 (99.9%) in 40.0 s
- E_GPU shape=(2,652,248, 4096), dtype=torch.float16, VRAM used=20.2 GB, free=74.7 GB, load 387.6 s
- Brute-force GPU cosine matmul; no FAISS

## Pipeline

### Phase 1 — Setup
1.1 Detect Python/PyTorch/GPU. 1.2 Mount Drive. 1.3 Resolve paths and `EMB_AVAILABLE`/`GRAPH_AVAILABLE` flags. 1.4 Print full CONFIG dict. 1.5 Stream val.csv into `ALL_QUERIES`.

### Phase 2 — Build all corpus indexes (one streaming pass)
- Stream `law_llm_descriptors_*.jsonl` (173,033 rows) and `court_authority_cards_v5_unified.jsonl` (2,476,315 rows) once.
- Statute canonicalizer (`statute_anchor_canonical`): regex extract `Art. N` + code, alias map (`CPP→StPO`, `CP→StGB`, `CC→ZGB`, `CO→OR`, `LTF→BGG`, …) returns `"N CODE"`.
- Case canonicalizer: BGE pattern `BGE V D P` and docket pattern `\d[A-Z]_\d+/\d{4}`.
- German lemmatizer for `term_orig`: recursive strip of suffixes `("en","es","em","er","e","n","s")` while stem ≥ 4 chars.
- Indexes populated: `cit_to_doc_ids`, `doc_meta`, `idx_law_direct`, `idx_court_statute`, `idx_case_anchor`, `idx_court_base`, `idx_concept_en`, `idx_term_orig`, `idx_term_lemma`, `term_orig_keys`, `legal_area_per_doc`, `search_text`, `co_citation_pairs`, `tlf`, `token_doc_count`, `doc_statute_anchors`, `doc_language`.
- 2.5 Map each query's gold citations → doc_id sets via `cit_to_doc_ids`.

### Phase 3 — Citation graph
- Connect SQLite at `citation_graph_extracted.sqlite`; select edges WHERE `dataset='court_considerations'`.
- For every `(src, tgt)`: lookup first-doc-id-per-citation, append to `idx_graph_out[s]` / `idx_graph_in[t]`.
- Build `idx_judgment_importance[court_base] = sum over its E-paragraph doc_ids of len(idx_graph_in[did])`.
- Layers encoded by upstream graph build: intra-judgment back-refs, date-stripped aliases, E.-range expansion, case-level fan-out.

### Phase 4 — Per-area bedrock + co-citation neighbours
- `per_area_canon_count[area]`: Counter of canonical statutes per legal area (built from `doc_statute_anchors` filtered by `legal_area_per_doc`).
- `code_pair_count`: symmetric counter of co-cited code pairs (StPO ↔ BGG, etc.).
- `co_neighbours[canon]`: top-K (=100) co-cited canons after frequency filter (`co_citation_max_neighbour_count=50000`), min co-count=50.

### Phase 5 — BM25 (FTS5)
- 5.A Build legacy single-language FTS5 over `search_text[did]`.
- 5.B Build per-language FTS5 (de/fr/it/en) routed by `doc_language[did]`.
- 5.C `_enhance_codes(text)`: per-token IDF-weighted top-K legal codes (capped by `enhance_min_idf=1.0`).
- 5.D Legacy `bm25_search()`.
- 5.E `bm25_search_multilang(query, targets, k_total)`: per-language query (raw English + DE/FR terms + EN concepts + enhance() code boost), each language gets a proportional budget (with floor ~6%), score-normalized by `log(N_lang + e)`, merged max-score per doc.

### Phase 6 — Vector setup
- Load manifest parquet, build `row_for_did` ↔ `my_did_for_row`.
- Concat 27 fp16 chunks `qwen3_8b_unified_chunk*.npy` to GPU.
- `vector_search(q_emb, k)`: normalize, `E_GPU @ q`, `torch.topk`.
- 6.4 Define `encode_query(text)` (SentenceTransformer + `prompt_name="query"` + normalize).

### Phase 7 — Query expansion + HyDE (Qwen3-32B)
- **Load once** (213.9 s; bf16 device_map=auto).
- **Pass 1 — structured targets.** System prompt names ~40 Swiss codes (StPO, StGB, BGG, ZGB, OR, ZPO, BV, EMRK, IPRG, IRSG, AHVG, IVG, AsylG, AIG, DBG, StHG, KVG, UVG, AVIG, PatG, MSchG, URG, FINIG, FINMAG, BankG, KKG, SchKG, VwVG, FZG, BVG, BPV, BetmG, RPG, NHG, USG, GSchG, SVG, GwG, …). Schema requests 10-20 `statute_targets`, case_targets, 15-25 `concept_targets_en`, 15-25 `term_targets_de`, 10-20 `term_targets_fr`, 3-6 `legal_area_keywords`. Two diverse few-shot examples (unemployment insurance; holographic will). Deterministic generate (do_sample=False, max_new_tokens=1024). Robust 3-strategy JSON parser: markdown fence → brace-balanced scan → greedy. Sampling retry at `temperature=0.4, top_p=0.95, max_new_tokens=1.5×` if first pass returns empty or <10 total items.
- **Pass 2 — Multi-aspect HyDE.** System: Swiss legal expert identifies 2-5 distinct legal aspects (different legal areas, not paraphrases) and writes one 3-5 sentence Bundesgericht/Tribunal-fédéral-style answer per aspect, citing Swiss articles inline (`Art. N CODE`) with German + French terms in parentheses. Format `ASPECT N: ... / ANSWER N: ...`. Parser uses regex `ANSWER\s+\d+\s*:\s*(.+?)(?=ASPECT\s+\d+\s*:|\Z)`. Each parsed answer ≥60 chars kept. Fallback: full response → query text.
- **Free** Qwen3-32B after both passes (`del qmod, qtok, ...`; CUDA mem returns to 20.2 GB occupied by E_GPU only).

### Phase 7.6 — Encode queries
- Load `SentenceTransformer("Qwen/Qwen3-Embedding-8B")`.
- For each query: encode raw, encode enriched (raw + first 60 of `term_de + term_fr + concept_en`), encode each HyDE aspect separately into `ALL_Q_EMB_HYDE_LIST[qid]` (list of tensors).

### Phase 8 — Channels (`run_channels`)
15 channels, four families. All channel implementations stored in cell 32. Highlights:
- `channel_law_direct(canon_set, idx, budget)`: counter over `idx_law_direct[canon]` matches.
- `channel_court_statute(canons, idx, idx_count, doc_meta, budget)`: per-doc accumulator with score = `Σ 1/log(2+global_count[canon]) × (1 + 0.3×(n_matches-1)) × role_weight`. `paragraph_role` weights: `reasoning/legal_standard/application/holding=1.5`, `facts/procedural_history/citation/neutral_default=1.0`, `costs/disposition/notification=0.6`, `neutral=0.4`, empty=0.4.
- `channel_sibling`: `score = seed_count[cb] × sqrt(1+log(1+importance)) × role_boost(1.5 if substantive)`.
- `channel_graph_forward`: target weighted by importance of its own judgment + role_boost.
- `channel_graph_reverse`: gated to seeds whose judgment has importance ≥ 5; edge weight = seed-judgment importance factor × target role_boost.
- `channel_concept` consumes weighted output of `expand_concepts_weighted` (exact=1.0, substring=0.85×shorter/longer with shared-meaningful-token gate, token overlap = shared/max).
- `channel_term`: per-query-term, MAX over paths (exact=1.0, lemma=0.7, substring), summed across query terms.
- `channel_per_area_bedrock`: select areas matching `legal_area_keywords` substring, take top-`per_area_top_n=1000` canons per selected area, filter by `statute_target_codes` (LLM-named + corpus-related codes).
- `channel_statute_backprop`: for each canon in `doc_statute_anchors[did]` for caught seed dids, score = `n_caught × 1/log(2+global_court_count[canon])`; emit corresponding `idx_law_direct[canon]`.
- `channel_co_citation`: per LLM `statute_targets[*]` canon, walk `co_neighbours[canon]` → for each (nb, n), `score = n × 1/log(2+canon_count[nb])`, emit law + court rows.
- `channel_vector(q_emb, k)`: `vector_search`.

### `run_channels` two-pass with PRF
1. **Pass 1 topical**: run all topical channels + bm25_multilang + vector_raw + vector_enriched + vector_hyde (per-aspect dense search RRF-fused with k=60 into one channel).
2. **PRF extraction**: mini-RRF (k=60) across pass-1 channels (skip `law_direct`), take top `prf_top_k_for_extraction=4000` **court** dids, aggregate their `statute_anchors` / `concepts_en` / `terms_original` annotations, keep top-N **new** signals: `prf_new_statutes_k=40` canons, `prf_new_concepts_k=60` concepts, `prf_new_terms_k=60` terms. Derive new codes from new canons.
3. **Pass 2**: re-run `law_direct_match`, `court_statute`, `concept_en` (new concepts at weight 0.5), `term_orig`, `per_area_bedrock`, `co_citation` with augmented inputs. Merge per channel via `_merge_channel` (max score per doc).
4. **Dual-pass graph**: build `seed_p1` (court hits from pass-1 topical + vectors + bm25) AND `seed_p2` (= seed_p1 ∪ pass-2 court hits). Run `channel_sibling`, `channel_graph_forward`, `channel_graph_reverse`, `channel_statute_backprop` **twice** (once per seed) and merge max-score. Preserves pass-1 gold against budget-cap displacement.

### Phase 9 — Fusion + master loop
- `rrf_fuse(channels, k=60, weights)`: weighted RRF, `score(did) = Σ_ch weights[ch] / (k + rank + 1)`.
- `apply_neg_gate(doc_ids, doc_meta, noise_roles)`: drop docs whose role ∈ `{notification, header, empty, metadata}` UNLESS role ∈ `SUBSTANTIVE_ROLES` (substantive role overrides the buggy `is_notification_paragraph` flag).
- `round_robin_guarantee(channels_by_name, guarantee_channel_names, per_channel_cap=130, total_cap=50000)`: emits one did per guarantee channel in round-robin until cap reached or iterators exhausted.
- `fuse_for_query(channels, gold_doc_set, total_gold)`: rank by RRF → gate → guarantee (gated) → fill from RRF tail → truncate at 50,000. Samples R@K at `K ∈ {50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 25000, 35000, 50000}`.
- **Master loop**: identical call per query in `ALL_QUERIES`; results stored in `PER_QUERY[qid]` with `channel_recalls`, `channel_hit_sets`, `union_did_set`, `missed_gold_dids`, `R_at_K`, `curve`, `final_topk`.

### Phase 10 — Aggregate + missed-gold diagnostic
10.1 Per-query table. 10.2 Macro mean R@K curve. 10.3 Per-channel mean recall. 10.4 List doc_ids absent from every channel's hit set + aggregate breakdown (family / language / paragraph_role / court_base prefix).

### Phase 11 — Save artifacts
11.1 Write `final_topk_<qid>.json` (rank/doc_id/citation), `summary_multiquery.json` (macro/micro recall + per-query channel recalls + R@K curves), `targets_multiquery.json`, `config.json`.
11.3 Cascade Snapshot to `out_dir/snapshot/`:
- `corpus_snapshot.json.gz` — per-doc `ct` (search_text[:3000]) + `cit` + `fam` + `cb` + `pr` + `ln` + top-15 statute_anchors / concepts / terms, scoped to union of `final_topk[:50000]` across queries
- `per_query_snapshot.json` — `final_topk` + `curve` + `channel_recalls` + gold counts
- `hyde_aspects.json` — `ALL_HYDE_ASPECTS`
- `all_targets.json` — `ALL_TARGETS`
- `gold_doc_sets.json` — `ALL_GOLD_DOC_SET`
- `config.json`, `paths.json`
11.4 VRAM cleanup (`del E_GPU, EMB_MODEL`).

## Results

### Query-expansion times (Pass 1 + HyDE)

```
[qexp] val_001   54.5s          stat=20 ce=20 td=20 tf=20 la= 6
[qexp] val_002   52.8s          stat=20 ce=24 td=24 tf=24 la= 6
[qexp] val_003   51.9s          stat=20 ce=25 td=25 tf=23 la= 6
[qexp] val_004   44.2s          stat=20 ce=20 td=20 tf=19 la= 6
[qexp] val_005  104.0s [retry]  stat=20 ce=20 td=20 tf=19 la= 5
[qexp] val_006   51.9s          stat=20 ce=24 td=24 tf=21 la= 6
[qexp] val_007   50.9s          stat=20 ce=25 td=25 tf=24 la= 6
[qexp] val_008   54.9s          stat=20 ce=25 td=26 tf=21 la= 6
[qexp] val_009   46.1s          stat=20 ce=25 td=25 tf=21 la= 5
[qexp] val_010   50.2s          stat=20 ce=20 td=20 tf=20 la= 6
```
val_005 alone required the sampling retry; all others produced ≥10 items on the deterministic first pass.

HyDE: 4 aspects (val_002, val_003, val_004, val_007), 5 aspects (val_001, val_005, val_006, val_008, val_009, val_010). Wall times 34.9–49.7 s/query.

### PRF augmentation per query (sample top canons / codes)

| qid | +canons | +codes | +concepts | +terms | new canons (top 8) |
|---|---:|---:|---:|---:|---|
| val_001 | 40 | 6 | 60 | 60 | `29 BV, 9 BV, 13 BV, 26 BV, 8 EMRK, 106 BGG, 27 BV, 42 BGG` |
| val_002 | 40 | 7 | 60 | 60 | `28 IVG, 16 ATSG, 8 ATSG, 4 IVG, 7 ATSG, 17 ATSG, 28a IVG, 7 LAI` |
| val_003 | 40 | 6 | 60 | 60 | `29 BV, 32 BV, 9 BV, 6 EMRK, 6 StPO, 29 StPO, 325 StPO, 9 StPO` |
| val_004 | 40 | 5 | 60 | 60 | `4 ZGB, 7 ZGB, 9 BV, 18 OR, 9 ZGB, 1 ZGB, 518 ZGB, 105 BGG` |
| val_005 | 40 | 9 | 60 | 60 | `308 ZGB, 273 ZGB, 292 StGB, 8 CEDH, 8 EMRK, 307 ZGB, 42 BGG, 159 ZGB` |
| val_006 | 40 | 7 | 60 | 60 | `398 OR, 105 BGG, 42 BGG, 64 StGB, 140 StGB, 106 BGG, 754 OR, 400 OR` |
| val_007 | 40 | 7 | 60 | 60 | `8 ZGB, 75 BGG, 106 BGG, 72 BGG, 42 BGG, 105 BGG, 16 ZGB, 90 BGG` |
| val_008 | 40 | 8 | 60 | 60 | `158 StGB, 138 StGB, 105 BGG, 9 BV, 29 BV, 140 StGB, 106 BGG, 64 StGB` |
| val_009 | 40 | 8 | 60 | 60 | `8 ZGB, 285 ZGB, 176 ZGB, 4 ZGB, 125 ZGB, 276 ZGB, 75 BGG, 163 ZGB` |
| val_010 | 40 | 7 | 60 | 60 | `105 BGG, 398 OR, 68 OR, 42 BGG, 8 ZGB, 2 ZGB, 257d OR, 106 BGG` |

PRF pool size = 4,000 court dids per query.

### Per-query channel recalls (size / gold_in_ch / recall)

#### val_001 (gold=42, total time 29.8s, R@K=0.929)
```
law_direct_match           218            7   16.7%
court_statute            15631           12   28.6%
co_citation               2500            0    0.0%
per_area_bedrock          1500           17   40.5%
statute_backprop          2390           17   40.5%
sibling_expansion         5319            4    9.5%
graph_forward             7281            9   21.4%
graph_reverse             3104            2    4.8%
graph_2hop                   0            0    0.0%
concept_en                5441            7   16.7%
term_orig                 4525            4    9.5%
bm25                      2000            5   11.9%
vector_raw                2000           13   31.0%
vector_enriched           2000           13   31.0%
vector_hyde               2000           13   31.0%
Union: 45,627 unique doc_ids   gold-in-union: 39/42
```

#### val_002 (gold=36, 33.5s, R@K=0.806)
```
law_direct_match            70           12   33.3%
court_statute            10441            2    5.6%
co_citation               2500            0    0.0%
per_area_bedrock          1604           19   52.8%
statute_backprop          2415           19   52.8%
sibling_expansion         7021            1    2.8%
graph_forward             6475           19   52.8%
graph_reverse             4944            1    2.8%
graph_2hop                   0            0    0.0%
concept_en                5677            0    0.0%
term_orig                 4273            1    2.8%
bm25                      2000            0    0.0%
vector_raw                2000            1    2.8%
vector_enriched           2000            2    5.6%
vector_hyde               2000            3    8.3%
Union: 42,834 unique doc_ids   gold-in-union: 29/36
```

#### val_003 (gold=47, 31.6s, R@K=0.766)
```
law_direct_match           217            6   12.8%
court_statute            15505            7   14.9%
co_citation               2500            1    2.1%
per_area_bedrock          1500           15   31.9%
statute_backprop          2297           19   40.4%
sibling_expansion         5514            2    4.3%
graph_forward             6782           13   27.7%
graph_reverse             3255            3    6.4%
graph_2hop                   0            0    0.0%
concept_en                4066            4    8.5%
term_orig                 4812            1    2.1%
bm25                      2000            2    4.3%
vector_raw                2000            4    8.5%
vector_enriched           2000            4    8.5%
vector_hyde               2000            2    4.3%
Union: 45,292 unique doc_ids   gold-in-union: 36/47
```

#### val_004 (gold=10, 28.3s, R@K=1.000)
```
law_direct_match           136            6   60.0%
court_statute             9252            1   10.0%
co_citation               2500            2   20.0%
per_area_bedrock          1500            2   20.0%
statute_backprop          2430            9   90.0%
sibling_expansion         5866            0    0.0%
graph_forward             7002            3   30.0%
graph_reverse             3607            0    0.0%
graph_2hop                   0            0    0.0%
concept_en                5842            5   50.0%
term_orig                 4083            0    0.0%
bm25                      2000            5   50.0%
vector_raw                2000            4   40.0%
vector_enriched           2000            5   50.0%
vector_hyde               2000            5   50.0%
Union: 43,683 unique doc_ids   gold-in-union: 10/10
```

#### val_005 (gold=11, 33.2s, R@K=1.000)
```
law_direct_match           127            6   54.5%
court_statute             8141            0    0.0%
co_citation               1956            1    9.1%
per_area_bedrock          1501            6   54.5%
statute_backprop          2658            6   54.5%
sibling_expansion         6139            0    0.0%
graph_forward             7880            5   45.5%
graph_reverse             3274            0    0.0%
graph_2hop                   0            0    0.0%
concept_en                5766            4   36.4%
term_orig                 4631            3   27.3%
bm25                      2000            2   18.2%
vector_raw                2000            7   63.6%
vector_enriched           2000            7   63.6%
vector_hyde               2000            6   54.5%
Union: 41,914 unique doc_ids   gold-in-union: 11/11
```

#### val_006 (gold=18, 29.4s, R@K=0.944)
```
law_direct_match           255            8   44.4%
court_statute            15602            2   11.1%
co_citation               2500            0    0.0%
per_area_bedrock             0            0    0.0%
statute_backprop          2277           11   61.1%
sibling_expansion         5578            0    0.0%
graph_forward             7317            7   38.9%
graph_reverse             3285            0    0.0%
graph_2hop                   0            0    0.0%
concept_en                5803            0    0.0%
term_orig                 4380            0    0.0%
bm25                      2000            0    0.0%
vector_raw                2000            4   22.2%
vector_enriched           1998            4   22.2%
vector_hyde               2000            8   44.4%
Union: 47,322 unique doc_ids   gold-in-union: 17/18
```

#### val_007 (gold=19, 31.9s, R@K=0.895)
```
law_direct_match           138            8   42.1%
court_statute             8564            0    0.0%
co_citation                669            0    0.0%
per_area_bedrock             0            0    0.0%
statute_backprop          2429           13   68.4%
sibling_expansion         6010            1    5.3%
graph_forward             7294            5   26.3%
graph_reverse             3408            0    0.0%
graph_2hop                   0            0    0.0%
concept_en                5149            0    0.0%
term_orig                 4816            1    5.3%
bm25                      2000            1    5.3%
vector_raw                2000            2   10.5%
vector_enriched           2000            1    5.3%
vector_hyde               2000           11   57.9%
Union: 41,351 unique doc_ids   gold-in-union: 17/19
```

#### val_008 (gold=29, 30.1s, R@K=0.862)
```
law_direct_match           211            9   31.0%
court_statute            13013            0    0.0%
co_citation               2500            0    0.0%
per_area_bedrock          1500           19   65.5%
statute_backprop          2420           18   62.1%
sibling_expansion         5273            0    0.0%
graph_forward             7906            9   31.0%
graph_reverse             3191            0    0.0%
graph_2hop                   0            0    0.0%
concept_en                6000            0    0.0%
term_orig                 4984            2    6.9%
bm25                      2000            0    0.0%
vector_raw                2000            3   10.3%
vector_enriched           2000            3   10.3%
vector_hyde               2000            5   17.2%
Union: 46,405 unique doc_ids   gold-in-union: 25/29
```

#### val_009 (gold=14, 35.7s, R@K=0.929)
```
law_direct_match           134            9   64.3%
court_statute             8940            0    0.0%
co_citation                276            0    0.0%
per_area_bedrock          1500            9   64.3%
statute_backprop          2490           11   78.6%
sibling_expansion         6037            1    7.1%
graph_forward             7174            6   42.9%
graph_reverse             3473            1    7.1%
graph_2hop                   0            0    0.0%
concept_en                5101            0    0.0%
term_orig                 4315            0    0.0%
bm25                      2000            1    7.1%
vector_raw                2000            0    0.0%
vector_enriched           2000            0    0.0%
vector_hyde               2000            4   28.6%
Union: 41,066 unique doc_ids   gold-in-union: 13/14
```

#### val_010 (gold=25, 28.3s, R@K=0.880)
```
law_direct_match           177            9   36.0%
court_statute            12061            3   12.0%
co_citation               2500            0    0.0%
per_area_bedrock             0            0    0.0%
statute_backprop          2363           11   44.0%
sibling_expansion         5372            2    8.0%
graph_forward             7344            7   28.0%
graph_reverse             3177            0    0.0%
graph_2hop                   0            0    0.0%
concept_en                4633            1    4.0%
term_orig                 4999            2    8.0%
bm25                      2000            0    0.0%
vector_raw                1998            6   24.0%
vector_enriched           1998            6   24.0%
vector_hyde               2000           11   44.0%
Union: 44,644 unique doc_ids   gold-in-union: 22/25
```

### Aggregate summary (Phase 10.1)

```
query       gold      R@K     caught      union   gate_union
------------------------------------------------------------
val_001       42    0.929   39/42      39/42      39/42
val_002       36    0.806   29/36      29/36      29/36
val_003       47    0.766   36/47      36/47      36/47
val_004       10    1.000   10/10      10/10      10/10
val_005       11    1.000   11/11      11/11      11/11
val_006       18    0.944   17/18      17/18      17/18
val_007       19    0.895   17/19      17/19      17/19
val_008       29    0.862   25/29      25/29      25/29
val_009       14    0.929   13/14      13/14      13/14
val_010       25    0.880   22/25      22/25      22/25
------------------------------------------------------------
MEAN                0.901              219/251    219/251
(micro union=0.873, gate union=0.873)
```

### Macro mean R@K curve (Phase 10.2)

```
     K   mean recall      min      max
----------------------------------------
    50        0.052    0.000   0.194
   100        0.121    0.000   0.273
   200        0.179    0.040   0.455
   300        0.353    0.128   0.643
   500        0.415    0.128   0.643
   750        0.436    0.170   0.643
  1000        0.530    0.170   0.800
  1500        0.571    0.234   0.900
  2000        0.616    0.255   0.900
  3000        0.663    0.383   0.900
  5000        0.728    0.404   0.909
  7500        0.768    0.489   0.909
 10000        0.822    0.617   1.000
 15000        0.839    0.638   1.000
 20000        0.852    0.660   1.000
 25000        0.867    0.681   1.000
 35000        0.891    0.745   1.000
 40748        0.891    0.745   1.000
 40953        0.891    0.745   1.000
 41310        0.894    0.745   1.000
 41408        0.894    0.745   1.000
 43179        0.894    0.745   1.000
 43992        0.894    0.745   1.000
 44138        0.896    0.766   1.000
 44440        0.901    0.766   1.000
 45871        0.901    0.766   1.000
 46812        0.901    0.766   1.000
```

### Per-channel mean recall (Phase 10.3)

```
channel                 mean recall   mean size
--------------------------------------------------
statute_backprop             0.592       2417
law_direct_match             0.395        168
graph_forward                0.344       7246
vector_hyde                  0.340       2000
per_area_bedrock             0.330       1060
vector_enriched              0.220       2000
vector_raw                   0.213       2000
concept_en                   0.116       5348
bm25                         0.097       2000
court_statute                0.082      11715
term_orig                    0.062       4582
sibling_expansion            0.037       5813
co_citation                  0.031       2040
graph_reverse                0.021       3472
graph_2hop                   0.000          0
```

### Missed-gold diagnostic (Phase 10.4)

35 doc_ids fell through every channel across all queries. Breakdown:

```
By family:
  court       22  (62.9%)
  law         13  (37.1%)

By language:
  ?           35  (100.0%)   [language metadata not surfaced]

By paragraph_role:
  (none)                     13  (37.1%)
  reasoning                   9  (25.7%)
  facts                       6  (17.1%)
  legal_standard              3   (8.6%)
  procedural_history          2   (5.7%)
  costs                       1   (2.9%)
  disposition                 1   (2.9%)

By court_base prefix:
  (no court_base, law row)  13  (37.1%)
  BGE                       12  (34.3%)
  6B                         3   (8.6%)
  8C                         2   (5.7%)
  7B / 9C / 1B / 2C / 5A     1 each
```

Specific misses include: `7B_496/2025 E. 3.2`, `Art. 37 Abs. 1 StBOG`, `Art. 39 Abs. 1 StBOG` (val_001); 9 misses for val_002 including `BGE 135 V 39 E. 6.1`, `BGE 127 V 205 E. 4b`, `BGE 132 V 93 E. 4`, `BGE 148 V 21 E. 5.3`, `BGE 140 V 193 E. 3.2`, `8C_510/2020 E. 2.4`, `8C_160/2016 E. 4.1`, `9C_623/2020 E. 4.2`, `Art. 18d IVG`; 11 misses for val_003 including `BGE 142 III 48 E. 4.1.1`, `BGE 124 III 5 E. 4`, `1B_192/2022 E. 4.1.2`, `BGE 131 III 601 E. 3.1`, `BGE 131 III 106 E. 1.1`, `2C_501/2020 E. 5.1`, and law rows `Art. 37/39 Abs. 1 StBOG`, `Art. 467/519/520 ZGB`; `BGE 128 III 419 E. 2.2` (val_006); `Art. 15 OR`, `Art. 98 Abs. 2 IPRG` (val_007); 5 misses for val_008 including `6B_904/2020 E. 1.1`, `BGE 131 III 91 E. 5.2`, `6B_1233/2016 E. 1` (twice — reasoning and disposition), `BGE 141 IV 132 E. 3.4.1`; `5A_954/2015 E. 3.3` (val_009); `Art. 176 Abs. 1 ZPO`, `Art. 181 Abs. 3 ZPO`, `Art. 300 ZPO` (val_010).

### Artifacts written (Phase 11)

`Wrote artifacts to /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7`:
- `summary_multiquery.json`
- `targets_multiquery.json`
- `config.json`
- `final_topk_<query_id>.json` × 10

Cascade snapshot (Phase 11.3) to `out_dir/snapshot/`:
- `corpus_snapshot.json.gz`
- `per_query_snapshot.json`
- `hyde_aspects.json`
- `all_targets.json`
- `gold_doc_sets.json`
- `config.json`
- `paths.json`

## Summary

Anchor-Funnel v7.5 fuses 15 retrieval channels with weighted RRF and a round-robin guarantee for 7 high-recall channels, augmented by corpus-side PRF (no train data, no hardcoded lists) and dual-pass graph expansion, producing macro mean R@50000 = 0.901 and micro union recall 0.873 across the 10 val queries (219/251 gold caught). Top contributing channels by mean recall are `statute_backprop` (0.592), `law_direct_match` (0.395), `graph_forward` (0.344), `vector_hyde` (0.340) and `per_area_bedrock` (0.330); `graph_2hop` is disabled and contributes 0.0. The multi-aspect HyDE channel (Qwen3-32B drafts 2–5 Bundesgericht-style answer paragraphs, each encoded by Qwen3-Embedding-8B, RRF-fused per query) is the single strongest dense signal — overtaking raw and enriched vector channels — and is decisive on val_007 (57.9%) and val_010 (44.0%). Recall failure modes are concentrated in three buckets that no channel surfaces: law rows of peripheral codes (StBOG, OG, ZPO civil-procedure articles), BGE/dossier paragraphs that don't share statute anchors with the LLM-named targets, and reasoning/facts paragraphs in non-DE-dominant judgments — together explaining 35 missed gold doc_ids and motivating the cascade-rerank dossier plan documented under `01_current_direction_cascade_rerank_precision`. The full pipeline takes ~12 minutes per session including model loads (Qwen3-32B 214s + Qwen3-Embedding-8B + 388s E_GPU load + ~30s per query), and the snapshot persisted under `research/anchor_funnel_val001_v7/snapshot/` lets downstream cascade work skip cells 1–46 entirely.
