# anchor_funnel_v7_part3_with_outputs.ipynb

**Path:** notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v7_part3_with_outputs.ipynb

## Configuration

### Runtime
- Python: 3.12.13
- PyTorch: 2.10.0+cu128 (CUDA)
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB
- Environment: Google Colab; runs auto-detect Drive / Kaggle / local paths via `CANDIDATE_ROOTS`.

### Models
- Query expansion: `Qwen/Qwen3-32B` (bf16, `device_map="auto"`, `enable_thinking=False`), `qwen_max_new_tokens=1024`. Deterministic first pass (`do_sample=False`), automatic retry at `temperature=0.4` if JSON parse weak.
- Query embedding: `Qwen/Qwen3-Embedding-8B` via `SentenceTransformer`, `prompt_name="query"`, `normalize_embeddings=True`. Output 4096-dim.
- Vector index: brute-force `E_GPU @ q` on cuda:0, fp16. No FAISS.
- Lexical: SQLite FTS5 in-memory, `tokenize='unicode61 remove_diacritics 2'`.

### Memory plan (one model in VRAM at a time)
- Phase 6 peak: ~65 GB (Qwen3-32B).
- Phase 7 peak: ~16 GB (Qwen3-Embedding-8B).
- Phase 8 onwards: ~22 GB (E_GPU = 27 fp16 chunks concatenated).

### CONFIG (full knob panel)
```
topk_final              = 1000
budget_law_direct       = None   (uncapped)
budget_court_statute    = 3000   (v7.1 lift from 600)
budget_concept          = 600
budget_term             = 500
budget_per_area         = 150
budget_co_citation      = 1000   (v7.1 lift from 400)
budget_bm25             = 600
budget_vector           = 800
budget_vector_enriched  = 800
budget_backprop         = 400
budget_sibling          = 2000   (v6 was 500)
budget_graph_forward    = 1500
budget_graph_reverse    = 1000
enable_graph_2hop       = True   (v7.1)
budget_graph_2hop       = 1500
rrf_k                   = 60
guarantee_channels      = ["law_direct_match", "per_area_bedrock",
                           "statute_backprop", "graph_forward",
                           "sibling_expansion"]
guarantee_per_channel   = 200    (5 x 200 = 1000)
per_area_top_n          = 100
co_citation_top_k_per_target    = 8
co_citation_min_co_count        = 50
co_citation_max_neighbour_count = 5000
concept_substring_top_k = 6
bm25_max_query_terms    = 60
bm25_min_token_len      = 3
vector_emb_model        = "Qwen/Qwen3-Embedding-8B"
vector_topk             = 800
qwen_query_model        = "Qwen/Qwen3-32B"
qwen_max_new_tokens     = 1024
enhance_top_k_codes     = 5
enhance_repeat_count    = 5
enhance_min_idf         = 1.0
noise_paragraph_roles   = {"notification","header","empty","metadata"}
lowercase_concepts      = True
lowercase_terms         = True
```

### Constants
- `CODE_ALIAS` maps French/Italian codes → Swiss German canonical: `CPP→StPO`, `CP→StGB`, `CC→ZGB`, `CO→OR`, `LTF→BGG`, `LACI→AVIG`, `LAA→UVG`, `LP→SchKG`, `LDIP→IPRG`, `Cst/Cst.→BV`, `STPO→StPO`, `OBG→OR`.
- `LEGAL_AREA_DEFAULT_CODE`: 10 legal-area phrase → default code fallbacks.
- Regex: `ART_RE = r"art\.?\s*(\d+[a-z]?)"`; `CASE_BGE_RE = r"BGE\s+(\d+)\s+([IVX]+)\s+(\d+)"`; `CASE_DOCKET_RE = r"\b(\d[A-Z]_\d+/\d{4})\b"`.
- v7.2 P2 extra statute regex `EXTRA_STATUTE_RE` covers FR/IT `art./article/articolo N CODE` with `Abs./Absatz/al./alinéa/cpv.`, `lit./let./Bst./Buchstabe`, `Ziff./Ziffer/n./no./num./cifra`. Used only on rows with no build-time anchors (~233 rows / 0.02%).

### Libraries
`sys`, `torch`, `pathlib.Path`, `pandas`, `json`, `re`, `time`, `gc`, `sqlite3`, `math`, `collections.defaultdict/Counter`, `statistics`, `numpy`, `transformers.AutoTokenizer/AutoModelForCausalLM`, `sentence_transformers.SentenceTransformer`, `google.colab.drive`.

## Data

Auto-detected `DATA_ROOT = /content/drive/MyDrive/swiss_law`. All resolved paths OK:
- `val_csv`: `/content/drive/MyDrive/swiss_law/data/val.csv` — 10 EN queries with `gold_citations` (semicolon-separated).
- `law_llm`: `/content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl` — 173,033 law rows.
- `court_v5`: `/content/drive/MyDrive/swiss_law/artifacts_v2/court_authority_cards_v5_unified.jsonl` — 2,476,315 court rows.
- `emb_dir`: `/content/drive/MyDrive/swiss_law/artifacts/embeddings` — 27 `qwen3_8b_unified_chunk*.npy` (~21 GB fp16).
- `emb_manifest`: `/content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_manifest.parquet` — 2,652,248 rows; cols `[doc_id, family, citation, row_index]`.
- `graph_db`: `/content/drive/MyDrive/swiss_law/data_insights/citation_graph_extracted.sqlite` — 4-layer graph (~2.4 GB; reads `edges WHERE dataset='court_considerations'`).
- `out_dir`: `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/`.

Flags: `EMB_AVAILABLE=True`, `GRAPH_AVAILABLE=True`.

Per-query gold counts (val.csv, 10 queries, 251 total gold citations):
- val_001: 42 (pre-trial detention, Art. 221 Abs. 1 lit. b StPO, collusion risk)
- val_002: 36 (vocational diploma, storage technician)
- val_003: 47 (Peruvian national criminal case)
- val_004: 10 (handwritten will, lakeside town near Thun, 1997)
- val_005: 11 (separated parent custody)
- val_006: 18 (homeowners, installer, March 2012)
- val_007: 19 (heirship, vintage pocket chronometer "The Meridian")
- val_008: 29 (town council member, community trust)
- val_009: 14 (divorced custodial parent, 4 children)
- val_010: 25 (Belize-registered investment vehicle, euro account)

## Pipeline

### Phase 1 — Setup
Print Python/PyTorch/GPU; mount Drive; resolve `PATHS` dict from `CANDIDATE_ROOTS`; print `CONFIG` JSON; read `val.csv` into `ALL_QUERIES` list of `{query_id, query, gold}`.

### Phase 2 — Build all corpus indexes (one streaming pass)
Stream law jsonl then court jsonl. Build all in-memory indexes:
- `cit_to_doc_ids[citation] -> [doc_id]`
- `doc_meta[did] -> {citation, family, court_base, paragraph_role, is_notification_paragraph}`
- `idx_law_direct[canon] -> {law_did}` (via `statute_anchor_canonical`)
- `idx_court_statute[canon] -> {court_did}` (canonicalized via `canonicalize_row_anchors`, fallback by `LEGAL_AREA_DEFAULT_CODE`, plus v7.2 P2 `EXTRA_STATUTE_RE` sweep on text_excerpt_original when row has no anchors)
- `idx_case_anchor[case_canon] -> {did}` (BGE / docket forms)
- `idx_court_base[base] -> {court_did}` (for sibling_expansion)
- `idx_concept_en[token] -> {did}`; `idx_term_orig[token] -> {did}`
- `legal_area_per_doc[did] -> area_lower`
- `search_text[did]` = concat of citation + summary/rule/question/conditions/concepts/terms (law) or citation + text_excerpt_original + concepts/terms/topic/anchors (court), truncated to 2000 chars
- `co_citation_pairs[(a,b)] -> count` (unordered pairs of canonical statutes within a court row)
- `tlf[token][code] -> count`, `token_doc_count[token]` (for BM25 `enhance()`)
- `doc_statute_anchors[did] -> {canon}`
- Gold mapping sanity check on val_001.

### Phase 3 — Citation graph load (v7 NEW)
Open `citation_graph_extracted.sqlite`, build `_cit_to_did` (first doc_id per citation), iterate edges from `court_considerations` dataset, map both ends to doc_ids and append to `idx_graph_out[src] / idx_graph_in[tgt]`. Synthetic targets (no doc_id) are skipped. Graph is 4-layer (intra-judgment backrefs, date-stripped aliases, E-range expansion, case-level fan-out).

### Phase 4 — Per-area bedrock + co-citation neighbours
- `per_area_canon_count[area] = Counter(canon)` from court rows' `doc_statute_anchors`, skipping `area == "law"`.
- `co_neighbours[canon]` = top neighbours from `co_citation_pairs` filtered by `min_co_count=50` and `max_neighbour_count=5000`, sorted by count, capped at `2*top_k_per_target = 16`.

### Phase 5 — BM25 (FTS5 in-memory)
Build `:memory:` FTS5 over `search_text` (2.65M rows, batched at 50k). Define `_enhance_codes(text)` (IDF-weighted token→code lookup over `tlf`) and `bm25_search(query_text, k)` which appends top-5 codes × 5 repeats, OR-joins quoted FTS terms (cap 60), executes ordered by `bm25(docs)`.

### Phase 6 — Query expansion (Qwen3-32B for all 10 queries)
Load Qwen3-32B bf16 once. Two diverse few-shots (unemployment AVIG; holographic will ZGB) — neither matches val_001 to avoid echo. System+user chat template via `apply_chat_template(..., return_tensors="pt", return_dict=True, enable_thinking=False)`. Unpack with `**_inp`; move to first non-meta device. Generate deterministic; retry once with `do_sample=True, temperature=0.4` if `parse_targets_json` returns None. `normalize_targets` coerces output to required schema (`statute_targets`, `case_targets`, `concept_targets_en`, `term_targets_de`, `term_targets_fr`, `legal_area_keywords`). Store in `ALL_TARGETS[qid]`. Free model: `del qmod; del qtok; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()`.

### Phase 7 — Encode queries (Qwen3-Embedding-8B)
Load `SentenceTransformer("Qwen/Qwen3-Embedding-8B")`. For each query encode raw query and enriched query (= raw + first 60 of `term_targets_de + term_targets_fr + concept_targets_en`). Move both to CPU. Cache `ALL_Q_EMB_RAW[qid]` and `ALL_Q_EMB_ENRICHED[qid]`. Free model.

### Phase 8 — Vector channel setup (E_GPU)
Read manifest parquet, build `row_for_did` and `my_did_for_row` mapping via `cit_to_doc_ids` + family match. Load 27 chunks `.npy`, `torch.cat` on cuda → `E_GPU`. Define `vector_search(q_emb, k)` doing `E_GPU @ q` and `torch.topk`, returning `[(did, score)]`. Set `VECTOR_OK = True`.

### Phase 9 — Channels (14 retrieval signals)
Per-query loop. Canonicalize statute targets, build `co_expanded_canons` (union of LLM canons + co-neighbours), `statute_target_codes` (set of codes from LLM canons). Expand concepts via `expand_concepts_strict` (exact + substring matches over `idx_concept_en` keys, top-K nearest by length).

Channels run in order:
1. `law_direct_match` — `idx_law_direct[canon]` over `co_expanded_canons`, Counter most_common (uncapped).
2. `court_statute` — `idx_court_statute[canon]` over `llm_statute_canons`, cap 3000.
3. `co_citation` — for each LLM statute target, follow `co_neighbours` to `idx_law_direct` + `idx_court_statute`; score = max co-count; cap 1000.
4. `per_area_bedrock` — areas matched by `legal_area_keywords` substring; canonicals top-`per_area_top_n=100` filtered to `statute_target_codes`; cap 150.
5. `bm25` — `bm25_search(query + enrich_bits, 600)`.
6. `vector_raw` and `vector_enriched` — `vector_search(q_emb, 800)`.
7. `concept_en` — `expanded_concepts → idx_concept_en`, cap 600.
8. `term_orig` — `term_targets_de/fr → idx_term_orig`, cap 500.

After topical channels, build `seed = ⋃ _court_hits(channels)` (court-family doc_ids from each).

9. `sibling_expansion` — v7.1 `channel_sibling`: score each non-seed sibling by number of distinct seeds sharing its `court_base` (replaces v7's alphabetical slice); cap 2000.

`seed_extended = seed ∪ _court_hits(ch_sibling)` — sibling outputs feed graph traversal.

10. `graph_forward` — `idx_graph_out[seed]`, count incoming votes, drop seeds; cap 1500.
11. `graph_reverse` — `idx_graph_in[seed]`, count incoming votes, drop seeds; cap 1000.
12. `graph_2hop` — 2-hop forward expansion via intermediate set; cap 1500.

`backprop_seed = seed ∪ siblings ∪ graph_fwd ∪ graph_rev`.

13. `statute_backprop` — for each canon in backprop_seed's `doc_statute_anchors`, emit law rows scored by count of distinct caught court rows citing that canon; cap 400.

### Fusion + gating
- RRF: `rrf_fuse(CHANNELS, k=60)`: `score[did] += 1/(60 + rank + 1)` summed across channels.
- Negative gate `apply_neg_gate`: drops `is_notification_paragraph` and `paragraph_role` in `{"notification","header","empty","metadata"}`.
- v7.1 round-robin guarantee: interleave 5 guarantee channels (`law_direct_match`, `per_area_bedrock`, `statute_backprop`, `graph_forward`, `sibling_expansion`) at 200 each → fills first 1000 slots fairly; RRF-gated tail appends the rest until `topk_final=1000`.

Record per-query: gold_doc_set, total_gold, CHANNELS (name+hits), channel_recalls, union_recall, final_topk, R@1000, rrf_scores, ranked_dids, guarantee, seed_extended.

### Phase 10 — Aggregate stats
Per-query table; macro mean R@1000; micro R@1000; macro mean union ceiling; per-channel mean recall ± stdev sorted descending; worst/best query.

### Phase 11 — Diagnosis
- 11.1 Helpers: `channel_rank`, `channel_score`, `_tokens`, `gold_vector_cos` (cos sim of gold row's E_GPU vector vs query vector), `concepts_pointing_to`, `terms_pointing_to`, `graph_paths_to/from`, `sibling_diagnosis`.
- 11.2 Per-gold trace on val_001 only: for each missed gold prints source row, per-channel HIT(rank)/MISS(reason), RRF score+rank, ROOT CAUSE in `{NO_CHANNEL_HIT, RRF_ZERO, GATED_OUT, RRF_RANK_TOO_LOW}`, FIX SUGGESTION.
- 11.3 Failure-mode summary: cause histogram; per-channel "caught_missed_gold" tally.
- 11.4 Cross-query miss summary: gold/caught/missed/union/best-channel per qid; union-vs-final gap table.

### Phase 12 — Save artifacts + cleanup
Per qid folder under `out_dir/<qid>/`: `final_topk.json`, `gold_in_top.json`, `targets.json`, `summary.json`. Top-level `aggregate_summary.json`, `val_001_gold_missed_diagnosis.json`, `config.json`. Cleanup: `del E_GPU; del EMB_MODEL; gc.collect(); torch.cuda.empty_cache()`.

## Results

### Phase 1 outputs
- Python 3.12.13, PyTorch 2.10.0+cu128, GPU 0 = NVIDIA RTX PRO 6000 Blackwell Server Edition (95.0 GB).
- Drive: `Mounted at /content/drive`.
- All 7 PATHS resolved OK. `EMB_AVAILABLE=True, GRAPH_AVAILABLE=True`.
- val.csv: 10 queries loaded; val_001 = 42 gold (canary).

### Phase 2 indexes
- Law: 173,033 rows in 9.9 s.
- Token→code association: 91,173 tokens, avg 11.7 codes/token.
- Court: 2,476,315 rows in 182.2 s.
- Total docs: 2,649,348. Unique citations: 2,158,211.
- Index sizes: `law_direct=49,288`, `court_statute=85,063`, `case=157,227`, `court_base=178,593`, `concept=292,946`, `term=363,331`.
- Co-citation pairs: 1,146,924.
- Gold mapping for val_001: 42/42 mapped, 42 total gold doc_ids.

### Phase 3 citation graph
- citation→doc_id map: 2,158,211 entries.
- Edges loaded: 20,494,436. Skipped (cit not in corpus): 3,155,263.
- Out-degree avg = 27.2. In-degree avg = 14.0.
- Graph load time: 120.6 s.

### Phase 4 per-area bedrock + co-citation
- per_area built in 2.0 s; 26 areas.
- `criminal law and criminal procedure` top-8: `('66 BGG', 28151), ('42 BGG', 20749), ('106 BGG', 19538), ('64 BGG', 13843), ('108 BGG', 12101), ('97 BGG', 12076), ('105 BGG', 11759), ('81 BGG', 11104)`.
- `constitutional and public law` top-8: `('66 BGG', 13218), ('29 BV', 12189), ('89 BGG', 11480), ('82 BGG', 11464), ('42 BGG', 10836), ('9 BV', 9650), ('106 BGG', 9626), ('68 BGG', 8649)`.
- `co_neighbours` indexed for 2,288 canonicals.
- `221 StPO` neighbours (top 8 after frequency filter): `31 BV (co=783, total=4339), 212 StPO (714/1337), 221 BV (646/670), 237 StPO (487/1467), 5 StPO (272/1955), 197 StPO (258/1555), 5 EMRK (246/3500), 5 CEDH (235/1234)`.

### Phase 5 BM25
- FTS5 built: 2,649,348 rows in 83.5 s.

### Phase 6 query expansion
- Qwen3-32B loaded in 197.5 s.
- Per-query parse sizes (`statute / concept / term_de / term_fr / area`):
  - val_001: 20 / 20 / 20 / 20 / 6
  - val_002: 20 / 20 / 20 / 20 / 5
  - val_003: 20 / 25 / 25 / 20 / 6
  - val_004: 20 / 20 / 20 / 19 / 6
  - val_005: weak first pass, retried at temperature=0.4 → 20 / 25 / 22 / 21 / 6
  - val_006: 20 / 24 / 20 / 20 / 6
  - val_007: 20 / 26 / 26 / 21 / 6
  - val_008: weak first pass, retried at temperature=0.4 → 20 / 20 / 19 / 17 / 5
  - val_009: 20 / 20 / 20 / 18 / 5
  - val_010: 20 / 25 / 25 / 25 / 6
- VRAM after free: `used=0.01 GB, free=94.96 GB`.
- Warning: `The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k'].`

### Phase 7 query encoding
- Qwen3-Embedding-8B loaded in 43.3 s.
- Enriched query lengths: val_001=2319, val_002=2993, val_003=2773, val_004=2210, val_005=2801, val_006=2530, val_007=2807, val_008=2637, val_009=2210, val_010=2971. Keywords appended: 60 except val_004=59, val_008=56, val_009=58.
- VRAM after free: `used=0.01 GB, free=94.96 GB`.

### Phase 8 vector setup
- Manifest: 2,652,248 rows.
- Manifest→my_did mapping: 2,649,348 / 2,652,248 (99.9%) in 40.8 s.
- E_GPU shape = (2,652,248, 4096), dtype = torch.float16.
- VRAM used = 20.2 GB, free = 74.7 GB. Chunk load 450.0 s.

### Phase 9 per-query channel recall + R@1000

| qid | gold | union_ceil | R@1000 | caught/total |
|-----|-----:|-----------:|-------:|--------------|
| val_001 | 42 | 0.738 | 0.262 | 11/42 |
| val_002 | 36 | 0.667 | 0.417 | 15/36 |
| val_003 | 47 | 0.447 | 0.298 | 14/47 |
| val_004 | 10 | 0.900 | 0.800 |  8/10 |
| val_005 | 11 | 0.909 | 0.636 |  7/11 |
| val_006 | 18 | 0.778 | 0.611 | 11/18 |
| val_007 | 19 | 0.789 | 0.737 | 14/19 |
| val_008 | 29 | 0.552 | 0.483 | 14/29 |
| val_009 | 14 | 0.643 | 0.571 |  8/14 |
| val_010 | 25 | 0.640 | 0.320 |  8/25 |

Per-query per-channel recall (selected highlights — see notebook for all 14 × 10):
- val_001 channels: law_direct=0.143, court_stat=0.071, co_cit=0.000, per_area=0.167, backprop=0.167, sibling=0.048, graph_fwd=0.190, graph_rev=0.190, graph_2h=0.000, concept=0.190, term=0.024, bm25=0.048, vector_raw=0.119, vector_enr=0.095.
- val_002 channels: backprop=0.472, graph_fwd=0.389.
- val_004 channels: backprop=0.800, graph_fwd=0.500, vector_raw=0.400.
- val_005 channels: backprop=0.545, graph_fwd=0.364, vector_raw=0.364, concept=0.364.
- val_007 channels: backprop=0.579, graph_fwd=0.526.
- val_009 channels: backprop=0.643, graph_fwd=0.357, per_area=0.286.

### Phase 10 aggregate
- **Macro mean R@1000 = 0.513** (10 queries).
- **Micro R@1000 = 0.438** (110/251 total gold).
- Macro mean union ceiling = 0.706.
- Worst query: val_001 (R@1000 = 0.262, union_ceiling = 0.738).
- Best query: val_004 (R@1000 = 0.800, union_ceiling = 0.900).

Per-channel mean recall (averaged across 10 queries, sorted):

| channel | mean recall | stdev |
|---|---:|---:|
| statute_backprop  | 0.453 | 0.201 |
| graph_forward     | 0.324 | 0.119 |
| vector_raw        | 0.153 | 0.134 |
| concept_en        | 0.145 | 0.116 |
| vector_enriched   | 0.119 | 0.090 |
| per_area_bedrock  | 0.105 | 0.098 |
| law_direct_match  | 0.096 | 0.090 |
| bm25              | 0.041 | 0.061 |
| court_statute     | 0.038 | 0.044 |
| term_orig         | 0.034 | 0.053 |
| sibling_expansion | 0.025 | 0.025 |
| graph_reverse     | 0.019 | 0.057 |
| co_citation       | 0.004 | 0.013 |
| graph_2hop        | 0.003 | 0.008 |

### Phase 11 diagnosis (val_001 only)
- val_001: 11/42 hit (26.2%), 31 missed.
- Root-cause histogram: `RRF_RANK_TOO_LOW = 16`, `NO_CHANNEL_HIT = 11`, `GATED_OUT = 4`.
- Query token count post-enhance: 192.

Selected per-gold traces:
- `1B_15/2023 E. 3.1` (doc_id=court:1035894, paragraph_role=facts): HIT vector_raw rank=146, HIT vector_enriched rank=333. RRF score 0.00737, rank 1115. cause RRF_RANK_TOO_LOW. court_statute row_anchors=`['221 StPO','212 StPO']` (no overlap with LLM statutes). graph_forward: 4 caught seeds with edges to this; graph_reverse: 42 caught seeds reachable.
- `1B_210/2023 E. 4.1` (French text, paragraph_role=legal_standard): HIT graph_reverse rank=483. cos_sim_raw=0.6763, cos_sim_enriched=0.6924. RRF score 0.00184, rank 5276.
- `1B_536/2018 E. 5.1` (French): HIT graph_reverse rank=555. cos_sim_raw=0.6606, cos_sim_enriched=0.6777. RRF rank 5868.
- `7B_496/2025 E. 3.2` (German): no row_anchors, no concept/term hits, 0 caught seed edges either direction. cos_sim_raw=0.4490, cos_sim_enriched=0.4604. RRF score 0. cause NO_CHANNEL_HIT.
- `BGE 132 I 21 E. 3.2.1`: cause RRF_RANK_TOO_LOW, RRF rank 5098.
- `1B_28/2022 E. 4.1`: 3 channels hit but cause = GATED_OUT (paragraph_role=facts).
- `Art. 422 Abs. 2 StPO` (law row): canonical `422 StPO` not in `co_expanded_canons`; seed-courts-citing=5 (below backprop threshold for ranking). graph_forward 2 caught seeds. RRF score 0. cause NO_CHANNEL_HIT.
- `Art. 428 Abs. 1 StPO`: canonical `428 StPO` not in canon set; seed-courts-citing=6. graph_forward 3 caught seeds. cause NO_CHANNEL_HIT.

Channel hit rate over val_001 missed gold (in candidate pool but ranked >1000):
- graph_reverse=7, vector_raw=5, concept_en=5, vector_enriched=4, court_statute=3, sibling_expansion=2, graph_forward=1, term_orig=1.
- 0 caught-but-missed for: law_direct_match, co_citation, per_area_bedrock, statute_backprop, graph_2hop, bm25.

### Phase 11.4 cross-query miss summary
| qid | gold | caught | missed | union | best-channel-for-query |
|---|---:|---:|---:|---:|---|
| val_001 | 42 | 11 | 31 | 0.738 | graph_forward (recall 0.190) |
| val_002 | 36 | 15 | 21 | 0.667 | statute_backprop (0.472) |
| val_003 | 47 | 14 | 33 | 0.447 | graph_forward (0.234) |
| val_004 | 10 |  8 |  2 | 0.900 | statute_backprop (0.800) |
| val_005 | 11 |  7 |  4 | 0.909 | statute_backprop (0.545) |
| val_006 | 18 | 11 |  7 | 0.778 | statute_backprop (0.500) |
| val_007 | 19 | 14 |  5 | 0.789 | statute_backprop (0.579) |
| val_008 | 29 | 14 | 15 | 0.552 | statute_backprop (0.414) |
| val_009 | 14 |  8 |  6 | 0.643 | statute_backprop (0.643) |
| val_010 | 25 |  8 | 17 | 0.640 | statute_backprop (0.280) |

Gold in union vs final top-1000 (gap = lost to RRF/gating):
| qid | in_union | in_final | gap |
|---|---:|---:|---:|
| val_001 | 31 | 11 | 20 |
| val_002 | 24 | 15 |  9 |
| val_003 | 21 | 14 |  7 |
| val_004 |  9 |  8 |  1 |
| val_005 | 10 |  7 |  3 |
| val_006 | 14 | 11 |  3 |
| val_007 | 15 | 14 |  1 |
| val_008 | 16 | 14 |  2 |
| val_009 |  9 |  8 |  1 |
| val_010 | 16 |  8 |  8 |

### Phase 12 artifacts + cleanup
- Saved 10 query folders under `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/<qid>/`.
- `aggregate_summary.json` written at `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/aggregate_summary.json`.
- Print confirms: `mean R@1000 (macro) = 0.513`, `R@1000 (micro) = 0.438`.
- VRAM after cleanup: 0.01 GB. `cleaned up.`

## Summary

The v7 (v7.1+v7.2 P2) anchor-funnel pipeline runs the full 11-phase architecture across all 10 val queries and reports a **macro mean R@1000 = 0.513 (micro = 0.438, 110/251 gold)**, with **union upper bound = 0.706**. The pass criterion stated in the title (R@1000 ≥ 0.60 for val_001) was **not met** — val_001 came in worst at 0.262, a regression versus the rest of the suite where 7/10 queries cleared 0.40 and val_004 hit 0.800. The two channels that generalize are `statute_backprop` (mean 0.453, top channel on 8/10 queries) and `graph_forward` (0.324), confirming that v7's NEW citation-graph forward channel earns its keep across queries, not just val_001. The lossy/unhelpful channels are `co_citation` (0.004), `graph_2hop` (0.003), `graph_reverse` (0.019), and `sibling_expansion` (0.025) — sibling+graph_reverse only matter on the val_001-shaped (multi-E BGer) topology. Phase 11 diagnosis on val_001 attributes its 31 misses to RRF_RANK_TOO_LOW (16), NO_CHANNEL_HIT (11), and GATED_OUT (4); the union-vs-final gap of 20 on val_001 (and 8-9 on val_010/val_002) shows ~half the missed gold is in the candidate pool but ranked >1000, suggesting the next iteration should rework guarantee allocation rather than add new channels. The negative gate dropping 4 val_001 gold whose paragraph_role is "facts" is a likely policy bug to revisit.

