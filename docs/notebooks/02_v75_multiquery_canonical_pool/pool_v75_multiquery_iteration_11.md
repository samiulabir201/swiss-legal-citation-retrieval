# pool_v75_multiquery_iteration_11.ipynb

**Path:** `notebooks/02_v75_multiquery_canonical_pool/pool_v75_multiquery_iteration_11.ipynb`

## Configuration

### Runtime / hardware
- Python 3.12.13, PyTorch 2.10.0+cu128
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB VRAM
- Colab + Google Drive mount at `/content/drive/MyDrive/swiss_law`

### Models
- **Query expansion + HyDE:** `Qwen/Qwen3-32B` (bf16, ~65 GB) — loaded once, two passes per query, then freed.
  - `qwen_max_new_tokens = 1024`
  - QEXP retry temperature: 0.4 if first pass is weak
- **Query-side embedding:** `Qwen/Qwen3-Embedding-8B` (SentenceTransformer; ~16 GB VRAM)
- **Stage 1 reranker (cells 52–54):** `Qwen/Qwen3-Reranker-8B` (bf16, ~15.3 GB VRAM, yes/no head; `token_true_id=9693`, `token_false_id=2152`; batch_size=8)

### Libraries / dependencies (imported)
`pandas`, `numpy`, `torch`, `sqlite3` (FTS5), `sentence_transformers`, `transformers`, `re`, `json`, `time`, `collections.{defaultdict,Counter}`, `gzip`, `gc`, `pathlib.Path`. Optional `google.colab.drive`.

### Top-level CONFIG (echoed in cell 8 output)
```
topk_final: 50000
budget_law_direct: null            (uncapped)
budget_court_statute: 8000
budget_concept: 3000
budget_term: 2500
budget_per_area: 1500
budget_co_citation: 2500
budget_bm25: 2000
budget_vector: 2000
budget_vector_enriched: 2000
budget_vector_hyde: 2000
budget_backprop: 2000
budget_sibling: 5000
budget_graph_forward: 5000
budget_graph_reverse: 3000
enable_graph_2hop: false
budget_graph_2hop: 1500
rrf_k: 60
guarantee_channels: [law_direct_match, per_area_bedrock, statute_backprop,
                     concept_en, graph_forward, sibling_expansion, term_orig]
guarantee_per_channel: 130    (7 × 130 = 910 round-robin slots)
channel_weights:
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
code_family_top_k: 8
enable_prf: true
prf_top_k_for_extraction: 4000
prf_new_statutes_k: 40
prf_new_concepts_k: 60
prf_new_terms_k: 60
per_area_top_n: 1000
co_citation_top_k_per_target: 50
co_citation_min_co_count: 50
co_citation_max_neighbour_count: 50000
concept_substring_top_k: 6
bm25_max_query_terms: 60
bm25_min_token_len: 3
vector_emb_model: "Qwen/Qwen3-Embedding-8B"
vector_topk: 800
qwen_query_model: "Qwen/Qwen3-32B"
qwen_max_new_tokens: 1024
enhance_top_k_codes: 5
enhance_repeat_count: 5
enhance_min_idf: 1.0
noise_paragraph_roles: {"notification", "header", "empty", "metadata"}
lowercase_concepts: true
lowercase_terms: true
```

### Constants / canonicalisers
- `CODE_ALIAS = {CPP→StPO, CP→StGB, CC→ZGB, CO→OR, LTF→BGG, LACI→AVIG, LAA→UVG, LP→SchKG, LDIP→IPRG, ...}`
- Cascade target K (Phase 12 Stage 1): top-5000 per query reranked, K-points {50, 100, 200, 500, 1000, 2000, 5000}

## Data

### Inputs (resolved by cell 6 against `DATA_ROOT = /content/drive/MyDrive/swiss_law`)
| Key | Path |
|---|---|
| `val_csv` | `/content/drive/MyDrive/swiss_law/data/val.csv` |
| `law_llm` | `/content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl` |
| `court_v5` | `/content/drive/MyDrive/swiss_law/artifacts_v2/court_authority_cards_v5_unified.jsonl` |
| `emb_dir` | `/content/drive/MyDrive/swiss_law/artifacts/embeddings` |
| `emb_manifest` | `/content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_manifest.parquet` |
| `graph_db` | `/content/drive/MyDrive/swiss_law/data_insights/citation_graph_extracted.sqlite` |
| `out_dir` | `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7` |

`CANDIDATE_ROOTS` (auto-detect order): `/content/drive/MyDrive/swiss_law`, `/content/drive/MyDrive/swiss_citation_extraction`, `/content/swiss_citation_extraction`, `E:/swiss_citation_extraction`, `Path.cwd()`.

Availability flags: `EMB_AVAILABLE = True`, `GRAPH_AVAILABLE = True`.

### val.csv contents (10 queries, gold sizes)
- val_001 gold=42 — "May a court lawfully order a three-month extension of pre-trial detention under Art. 221 ..."
- val_002 gold=36 — "A claimant holding a national vocational diploma in warehouse operations ..."
- val_003 gold=47 — "A. Rivera, a Peruvian national born in 1994 ..."
- val_004 gold=10 — "Mr. Dalton, born in 1941 and resident in a small lakeside town near Thun ..."
- val_005 gold=11 — "A parent, separated from their co-parent since 2008 ..."
- val_006 gold=18 — "On 3 March 2012, homeowners Ms. L and her partner Mr. M asked G, an installer ..."
- val_007 gold=19 — "An heirship claims title to a vintage pocket chronometer ..."
- val_008 gold=29 — "Has a member of the town council of the Borough of L., who chaired the board ..."
- val_009 gold=14 — "A divorced custodial parent lives alone with four children ..."
- val_010 gold=25 — "A Belize-registered investment vehicle (M) authorized its sole beneficial owner (P) ..."

### Index sizes built in cell 12
- Law rows: 173,033 (indexed in 12.2s)
- Court rows: 2,476,315 (indexed in 178.3s)
- Total docs: 2,649,348
- Unique citations: 2,158,211
- Index sizes: law_direct=49,288; court_statute=81,988; case=157,227; court_base=178,593; concept=292,946; term=363,331; term_lemma=521,146; term_keys=363,331
- Co-citation pairs: 1,142,790
- Token→code association: 91,173 tokens, avg codes/token = 11.7

### Embeddings (cell 24)
- Manifest rows: 2,652,248 (cols: `doc_id`, `family`, `citation`, `row_index`)
- Manifest→my_did mapping: 2,649,348/2,652,248 (99.9%) in 40.0s
- `E_GPU` shape: (2,652,248, 4096) dtype torch.float16
- VRAM used: 20.2 GB; free: 74.7 GB; load time 387.6s

### Citation graph (cell 16)
- Loaded from `data_insights/citation_graph_extracted.sqlite`
- 20,494,436 edges loaded, 3,155,263 skipped (citation not in corpus)
- Out-degree avg = 27.2; in-degree avg = 14.0
- Load time 103.2s
- Judgment importance: 178,593 judgments; median=7, p75=44, max=204686 (built in 1.1s)

### Per-area bedrock (cell 18)
- 26 legal areas indexed (built in 2.1s)
- Examples:
  - constitutional and public law: top 8 = [('66 BGG', 13212), ('29 BV', 12189), ('89 BGG', 11479), ('82 BGG', 11464), ('42 BGG', 10836), ('9 BV', 9650), ('106 BGG', 9625), ('68 BGG', 8649)]
  - administrative/tax/migration: top 8 = [('42 BGG', 20301), ('106 BGG', 19460), ('66 BGG', 17471), ('105 BGG', 16033), ('95 BGG', 15722), ('83 BGG', 15231), ('68 BGG', 14674), ('89 BGG', 12187)]
  - civil law: top 8 = [('63 OJ', 3189), ('8 ZGB', 2740), ('55 OJ', 2518), ('64 OJ', 2235), ('55 OG', 2085), ('63 OG', 1906), ('159 OG', 1876), ('9 BV', 1875)]
  - criminal law and criminal procedure: top 8 = [('66 BGG', 28149), ('42 BGG', 20746), ('106 BGG', 19537), ('64 BGG', 13843), ('108 BGG', 12101), ('97 BGG', 12075), ('105 BGG', 11759), ('81 BGG', 11102)]
- Corpus-derived code-pair stats: 54,790 directed pairs (0.7s). Example StPO related codes: [('BGG', 50026), ('BV', 34792), ('StGB', 18981), ('EMRK', 6826), ('Satz', 2812), ('CEDH', 2549), ('OR', 1961), ('ZGB', 1123)]

### Co-citation neighbours (cell 20)
- Indexed for 2,282 canonicals (after frequency filter; `co_citation_max_neighbour_count = 50,000`)
- Example for `221 StPO`: 36 BV (co=915), 31 BV (co=783), 10 BV (co=731), 212 StPO (co=714), 221 BV (co=646), 237 StPO (co=487), 5 BV (co=393), 5 StPO (co=272)

### FTS5 BM25 (cell 22)
- Legacy single-language FTS: 2,649,348 rows in 84.5s
- Per-language FTS built in 70.7s
  - fts[de]: 1,593,249 docs
  - fts[fr]:   793,023 docs
  - fts[it]:   125,583 docs
  - fts[en]:   137,493 docs

### Output paths (cell 46)
Written to `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/`:
- `summary_multiquery.json`
- `targets_multiquery.json`
- `config.json`
- `final_topk_<query_id>.json` × 10

Cell 48 saves cascade snapshot to `.../research/anchor_funnel_val001_v7/snapshot/`:
- `corpus_snapshot.json.gz` (14.8 MB)
- `per_query_snapshot.json`
- `hyde_aspects.json`
- `all_targets.json`
- `gold_doc_sets.json`
- `config.json`
- `paths.json`
- Cascade pool: 32,717 unique doc_ids

## Pipeline

### Phase 1 — Setup (cells 1–10)
- Probe Python / PyTorch / GPU.
- Mount Google Drive.
- Auto-detect `DATA_ROOT` and build `PATHS` dict; verify existence.
- Define `CONFIG` knob panel.
- Load `val.csv` into `ALL_QUERIES` (qid, query_text, gold list parsed by `;`).

### Phase 2 — Corpus indexes (cells 11–14)
One streaming pass over `law_llm_descriptors_0000000_all.jsonl` (173k rows) and `court_authority_cards_v5_unified.jsonl` (2.48M rows). Builds in-memory:
- `idx_law_direct`, `idx_court_statute`, `idx_case`, `idx_court_base`, `idx_concept_en`, `idx_term_orig`, `idx_term_lemma`, `idx_judgment_importance`
- `doc_meta`, `search_text`, `doc_statute_anchors`, `_doc_to_concepts`, `_doc_to_terms`, `legal_area_per_doc`
- `cit_to_doc_ids` (citation string → doc_id list)
- `co_citation_pairs` (Counter of (canon_a, canon_b) co-occurrences)
- Resolves each query's gold citations to doc_ids → `ALL_GOLD_DOC_SET`, `ALL_TOTAL_GOLD`. All 10 queries map all gold (mapped=gold for every query; some doc_ids slightly exceed gold count because some gold citations resolve to multiple doc_ids).

### Phase 3 — Citation graph (cells 15–16)
- Loads `citation_graph_extracted.sqlite` (4 alias passes; 23.65 M edges total in DB).
- Builds `_cit_to_did` (one did per citation) and `idx_graph_out[did]`, `idx_graph_in[did]` (doc_id → list of doc_ids).
- 20.49 M edges loaded, 3.16 M skipped because the citation is not in corpus.
- Computes `idx_judgment_importance[court_base] = degree-in-graph`.

### Phase 4 — Per-area bedrock + co-citation neighbours (cells 17–20)
- Cell 18: `per_area_canon_count[area] = Counter(canon → count)` over corpus rows tagged with that legal_area. Plus `code_family_top_k=8` corpus-derived related codes per LLM-named code.
- Cell 20: `co_neighbours[canon] = list[(neighbour_canon, co_count)]` filtered by `co_citation_min_co_count` and `co_citation_max_neighbour_count`.

### Phase 5 — BM25 / FTS5 (cells 21–22)
- Builds in-memory SQLite FTS5 over `search_text` for the legacy index (single-language) and per-language indices (de / fr / it / en).
- Defines `bm25_search(query_text, k)` returning top-k doc_ids ranked by BM25, plus `enhance(query)` adding corpus-associated codes (top_k=5, repeat_count=5, min_idf=1.0).

### Phase 6 — Vector channel (cells 23–26)
- Loads 27 fp16 chunks (`qwen3_8b_unified_chunk*.npy`) into `E_GPU = (2,652,248, 4096) fp16`.
- Builds `my_did_for_row` and `row_for_did` mapping arrays.
- `vector_search(q_emb, k)`: brute-force `E_GPU @ q` + topk on GPU.
- `encode_query(text)` via SentenceTransformer with `prompt_name="query"`, `normalize_embeddings=True`.

### Phase 7 — Query expansion + HyDE (cells 27–28)
Loads Qwen3-32B once (213.9s); two passes per query in same model load; frees model.
- **QEXP system prompt:** "You are a Swiss legal-research assistant. You enumerate broadly across the most relevant Swiss federal codes (StPO, StGB, BGG, ZGB, OR, ZPO, BV, EMRK, IPRG, IRSG, AHVG, IVG, AsylG, AIG, DBG, StHG, KVG, UVG, AVIG, PatG, MSchG, URG, FINIG, FINMAG, BankG ..."
- Targets returned per query: `statute_targets` (stat), `concept_seeds_en` (ce), `german_terms` (td), `french_terms` (tf), `legal_area` (la).
- 3-strategy JSON parser (markdown fence → brace-balanced scan → greedy). Re-sample at `temperature=0.4` if pass-1 weak (only val_005 triggered retry, taking 104.0s vs ~50s norm).
- HyDE pass: Qwen3-32B drafts 4–6 sentence Bundesgericht-style answer with inline German/French legal terms. 4–5 aspects per query.

### Phase 7.6 — Encode queries (cells 29–30)
- Loads Qwen3-Embedding-8B (after freeing Qwen3-32B).
- Builds three vectors per query: `ALL_Q_EMB_RAW[qid]`, `ALL_Q_EMB_ENRICHED[qid]` (raw + top 60 LLM-named terms/concepts), and per-aspect HyDE vectors mean-fused.

### Phase 8 — Channels (cells 31–34)
Defines 15 channel functions and `run_channels()` with **dual-pass + PRF**:

1. **Pass 1** — topical channels run with LLM-named targets + 3 vector queries + BM25.
2. **Mini-fuse pass-1 hits**; pick top-`prf_top_k_for_extraction=4000` court dids.
3. **Aggregate PRF pool** annotations: `statute_anchors` (top 40 new), `concepts_en` (top 60 new), `terms_original` (top 60 new), plus corpus-co-occurring codes.
4. **Pass 2** — canon/concept/term-dependent channels re-run with augmented targets, merged per channel.
5. **Dual-pass** for sibling_expansion / graph_forward / graph_reverse / statute_backprop (pass-1 seed + PRF-expanded seed, merged).

15 channels: `law_direct_match`, `court_statute`, `co_citation`, `per_area_bedrock`, `statute_backprop`, `sibling_expansion`, `graph_forward`, `graph_reverse`, `graph_2hop` (disabled, weight 0.0), `concept_en`, `term_orig`, `bm25`, `vector_raw`, `vector_enriched`, `vector_hyde`.

### Phase 9 — RRF fusion + master loop (cells 35–36)
- `rrf_fuse(channels, k, weights)` — weighted RRF: `score(did) = sum_ch weights[ch] / (k + rank + 1)`.
- `apply_neg_gate(doc_ids, doc_meta, noise_roles)` — drops noisy paragraph roles (notification/header/empty/metadata), but substantive roles (facts, reasoning, legal_standard, application, holding, citation, procedural_history) override the `is_notification_paragraph` flag.
- **7-channel guarantee** prepends 130 slots/channel × 7 = 910 slots via round-robin merge before the RRF tail fills the rest of `topk_final=50,000`.
- Loops over all 10 queries.

### Phase 10 — Aggregate diagnostics (cells 37–44)
- Per-query table: `gold`, `R@K`, `caught/gold`, `union`, `gate_union`.
- Macro mean R@K curve across all K-points in the union.
- Per-channel mean recall + mean channel size.
- Missed-gold diagnostic — for queries with R@K<1.0, dumps the gold docs absent from every channel with family / language / court_base / paragraph_role / 300-char text excerpt + aggregate buckets.

### Phase 11 — Save artifacts + snapshot + warm-boot (cells 45–50)
- Cell 46: per-query `final_topk_<qid>.json` + aggregate `summary_multiquery.json` + `targets_multiquery.json` + `config.json`.
- Cell 48: cascade snapshot (corpus subset of top-5000 dids per query union + per-query state).
- Cell 50: Warm-Boot reloads snapshot, reconstructing `PER_QUERY`, `search_text`, `doc_meta`, `doc_statute_anchors`, `_doc_to_concepts`, `_doc_to_terms`, `ALL_QUERIES`, `ALL_TARGETS`, `ALL_HYDE_ASPECTS`, `ALL_GOLD_DOC_SET`, `ALL_TOTAL_GOLD`, `CONFIG`, `PATHS` in ~30–60s; lets future sessions skip cells 0–46.

### Phase 12 — Hybrid 3-stage cascade (cells 51–60)
- **Stage 1** (cell 52): cross-encoder rerank with Qwen3-Reranker-8B on top-5000 per query, batch_size=8. Output `PER_QUERY[qid]['stage1_ranked']` + R@K curve at K∈{50, 100, 200, 500, 1000, 2000, 5000}.
- **Stage 1B** (cell 54, optional): translation A/B — translate top-200 DE/FR/IT candidates to English on Qwen3-32B, re-rerank, compare R@50 / R@100 / R@200.
- **Stage 2** (cell 56): listwise scoring with Qwen3-32B on top-500 reranked, batches of 20, JSON `[{id, score 1-5, why ≤25 words}, ...]`. Output top-100.
- **Stage 3** (cell 58): pointwise evidence-quoted verdict with Qwen3-32B on top-100, 5 candidates/call. Schema: `topic`, `relation`, `evidence_quote` (verbatim substring; empty → forced DROP), `confidence 1-5`, `verdict` (KEEP/MAYBE/DROP computed deterministically).
- **Stage 4** (cell 60): adaptive K + aspect coverage + F1. KEEP@conf5 → KEEP@conf4 up to `target = [max(15, n_aspects×4), max(40, n_aspects×8)]` → MAYBE if below target_min. Aspect-coverage force-promote if a HyDE aspect has zero KEEPs but a high Stage-2 candidate addresses it.

### Phase 11.2 — Cleanup (cells 61–62)
`del E_GPU; del EMB_MODEL; gc.collect(); torch.cuda.empty_cache()`.

## Results

### Per-query aggregate (cell 38, topk_final=50,000)
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
MEAN                0.901             219/251   219/251       (micro union=0.873, gate union=0.873)
```

### Per-query channel recalls (cell 36)
- **val_001** (gold=42, R@50k=0.929): law_direct 7/42 (16.7%), court_statute 12/42 (28.6%), co_citation 0/42, per_area_bedrock 17/42 (40.5%), statute_backprop 17/42 (40.5%), sibling 4/42 (9.5%), graph_forward 9/42 (21.4%), graph_reverse 2/42 (4.8%), concept_en 7/42 (16.7%), term_orig 4/42 (9.5%), bm25 5/42 (11.9%), vector_raw/enriched/hyde 13/42 (31.0%). Union 45,627 dids → 39/42. 29.8s.
- **val_002** (gold=36, R@50k=0.806): statute_backprop 19/36 (52.8%), per_area_bedrock 19/36 (52.8%), graph_forward 19/36 (52.8%), law_direct 12/36 (33.3%), vector_hyde 3/36 (8.3%). Union 42,834 → 29/36. 33.5s.
- **val_003** (gold=47, R@50k=0.766): statute_backprop 19/47 (40.4%), per_area_bedrock 15/47 (31.9%), graph_forward 13/47 (27.7%), court_statute 7/47 (14.9%), law_direct 6/47 (12.8%). Union 45,292 → 36/47. 31.6s.
- **val_004** (gold=10, R@50k=1.000): statute_backprop 9/10 (90%), law_direct 6/10 (60%), concept_en 5/10 (50%), bm25 5/10 (50%), vector_enriched 5/10 (50%), vector_hyde 5/10 (50%). Union 43,683 → 10/10. 28.3s.
- **val_005** (gold=11, R@50k=1.000): vector_raw/enriched 7/11 (63.6%), vector_hyde 6/11, statute_backprop/per_area_bedrock/law_direct 6/11 (54.5%). Union 41,914 → 11/11. 33.2s.
- **val_006** (gold=18, R@50k=0.944): statute_backprop 11/18 (61.1%), law_direct 8/18 (44.4%), vector_hyde 8/18 (44.4%), graph_forward 7/18 (38.9%), per_area_bedrock 0/18. Union 47,322 → 17/18. 29.4s.
- **val_007** (gold=19, R@50k=0.895): statute_backprop 13/19 (68.4%), vector_hyde 11/19 (57.9%), law_direct 8/19 (42.1%), graph_forward 5/19 (26.3%), per_area_bedrock 0/19. Union 41,351 → 17/19. 31.9s.
- **val_008** (gold=29, R@50k=0.862): per_area_bedrock 19/29 (65.5%), statute_backprop 18/29 (62.1%), law_direct 9/29 (31.0%), graph_forward 9/29 (31.0%), vector_hyde 5/29 (17.2%). Union 46,405 → 25/29. 30.1s.
- **val_009** (gold=14, R@50k=0.929): statute_backprop 11/14 (78.6%), law_direct 9/14 (64.3%), per_area_bedrock 9/14 (64.3%), graph_forward 6/14 (42.9%), vector_hyde 4/14 (28.6%). Union 41,066 → 13/14. 35.7s.
- **val_010** (gold=25, R@50k=0.880): statute_backprop 11/25 (44.0%), vector_hyde 11/25 (44.0%), law_direct 9/25 (36.0%), graph_forward 7/25 (28.0%), vector_raw/enriched 6/25 (24%), per_area_bedrock 0/25. Union 44,644 → 22/25. 28.3s.

### Macro R@K curve (cell 40)
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
 41310        0.894    0.745   1.000
 44138        0.896    0.766   1.000
 44440        0.901    0.766   1.000
 46812        0.901    0.766   1.000
```

### Per-channel mean recall (cell 42)
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

### Missed-gold diagnostic (cell 44)
Total missed across all queries: **35 of 251 gold doc_ids**.

By family: court 22 (62.9%), law 13 (37.1%).
By language: all "?" (100%).
By paragraph_role: (none) 13 (37.1%), reasoning 9 (25.7%), facts 6 (17.1%), legal_standard 3 (8.6%), procedural_history 2 (5.7%), costs 1, disposition 1.
By court_base prefix: (no court_base) 13 (37.1%), BGE 12 (34.3%), 6B 3 (8.6%), 8C 2 (5.7%), 7B/9C/1B/2C/5A 1 each.

Sample missed dids per query:
- **val_001 (3 missed):** `7B_496/2025 E. 3.2` (procedural_history), `Art. 37 Abs. 1 StBOG`, `Art. 39 Abs. 1 StBOG`.
- **val_002 (9 missed):** `BGE 135 V 39 E. 6.1` (facts/FR), `BGE 127 V 205 E. 4b` (costs), `BGE 132 V 93 E. 4` (legal_standard), `BGE 148 V 21 E. 5.3` (facts/FR), `BGE 140 V 193 E. 3.2` (facts), `8C_510/2020 E. 2.4` (reasoning/FR), `8C_160/2016 E. 4.1` (reasoning/FR), `9C_623/2020 E. 4.2` (procedural_history), `Art. 18d IVG`.
- **val_003 (11 missed):** `BGE 142 III 48 E. 4.1.1`, `BGE 124 III 5 E. 4`, `1B_192/2022 E. 4.1.2`, `BGE 131 III 601 E. 3.1`, `BGE 131 III 106 E. 1.1`, `2C_501/2020 E. 5.1`, `Art. 37 Abs. 1 StBOG`, `Art. 39 Abs. 1 StBOG`, `Art. 467 ZGB`, `Art. 519 Abs. 1 ZGB`, `Art. 520 Abs. 1 ZGB`.
- **val_004:** none.
- **val_005:** none.
- **val_006 (1 missed):** `BGE 128 III 419 E. 2.2` (reasoning).
- **val_007 (2 missed):** `Art. 15 OR`, `Art. 98 Abs. 2 IPRG`.
- **val_008 (5 missed):** `6B_904/2020 E. 1.1` (facts), `BGE 131 III 91 E. 5.2` (facts), `6B_1233/2016 E. 1` (reasoning), `6B_1233/2016 E. 1` (disposition), `BGE 141 IV 132 E. 3.4.1` (facts).
- **val_009 (1 missed):** `5A_954/2015 E. 3.3` (legal_standard).
- **val_010 (3 missed):** `Art. 176 Abs. 1 ZPO`, `Art. 181 Abs. 3 ZPO`, `Art. 300 ZPO`.

### Phase 12 Stage 1 — Qwen3-Reranker-8B (cell 52)
Sanity-check on val_001 ("May a court lawfully order a three-month extension of pre-trial detention under Art. 221 ..."):
- gold scores: [0.4378, 0.0191, 0.0001]
- non-gold scores: [0.0010, 0.0001, 0.0005]
- mean gold=0.1523, mean non-gold=0.0006, separation = +0.1518. "[OK] separation is reasonable; proceeding."

Per-query R@K after reranking top-5000:
```
[val_001] 312.9s  R@50=0.000 R@100=0.000 R@200=0.000 R@500=0.190 R@1000=0.286
[val_002] 329.9s  R@50=0.000 R@100=0.000 R@200=0.000 R@500=0.000 R@1000=0.083
[val_003] 334.7s  R@50=0.000 R@100=0.000 R@200=0.021 R@500=0.128 R@1000=0.149
[val_004] 311.6s  R@50=0.200 R@100=0.200 R@200=0.200 R@500=0.300 R@1000=0.500
[val_005] 323.9s  R@50=0.182 R@100=0.182 R@200=0.455 R@500=0.545 R@1000=0.636
[val_006] 325.2s  R@50=0.222 R@100=0.333 R@200=0.389 R@500=0.389 R@1000=0.444
[val_007] 336.1s  R@50=0.000 R@100=0.000 R@200=0.158 R@500=0.211 R@1000=0.632
[val_008] 329.8s  R@50=0.069 R@100=0.069 R@200=0.069 R@500=0.138 R@1000=0.276
[val_009] 304.5s  R@50=0.071 R@100=0.071 R@200=0.143 R@500=0.143 R@1000=0.500
[val_010] 332.8s  R@50=0.040 R@100=0.160 R@200=0.160 R@500=0.200 R@1000=0.240
```

Stage 1 final fusion-vs-rerank comparison block errored out:
```
ValueError: Sign not allowed in string format specifier
  File "/tmp/ipykernel_781/1299914646.py", line 181
  print(f"  {'K':>6}  {'fusion':>8}  {'stage1':>8}  {'Δ':>+7}  ...")
```
The `Δ` (Delta) column with format spec `>+7` is invalid for strings — only valid for numeric types. Stage 1B / Stage 2 / Stage 3 / Stage 4 cells (54, 56, 58, 60) have no execution outputs.

### Snapshot save (cell 48)
- `cascade pool: 32,717 unique doc_ids` (union of top-5000 per query)
- `corpus_snapshot.json.gz` 14.8 MB
- Snapshot dir: `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot`

### Warm-boot reload (cell 50)
- CONFIG loaded (topk_final=50000)
- ALL_QUERIES: 10
- corpus_snapshot: 32,717 docs
- PER_QUERY: 10; ALL_TARGETS: 10; ALL_HYDE_ASPECTS: 10
- ALL_GOLD_DOC_SET total: 254 doc_ids
- Sanity (val_001 top-2): `law:128343` "Art. 10 Abs. 1 StGB ...", `law:57057` "Art. 66 Abs. 1 BGG ...".

## Summary

The notebook is the iteration-11 build of the v7.5 multi-query anchor-funnel and pushes the 15-channel retrieval pool to **macro R@50k = 0.901** on val (219/251 gold; micro union 0.873). Pool R@K curve crosses 0.5 between K=750 and K=1000, hits 0.728 at K=5k, and saturates near 0.90 above K=35k. Per-query R_max bottoms at val_003 = 0.766 — the same structural ceiling already documented in the experiments ledger; val_002 = 0.806 and val_008 = 0.862 also fall short of 0.9. `statute_backprop` is by far the strongest individual channel (mean recall 0.592, mean size 2417), followed by `law_direct_match` (0.395), `graph_forward` (0.344), `vector_hyde` (0.340), and `per_area_bedrock` (0.330); `co_citation`, `graph_reverse`, and `graph_2hop` (disabled) contribute essentially nothing. Cell 48 saves a 32,717-doc cascade snapshot to `research/anchor_funnel_val001_v7/snapshot/` and cell 50 successfully warm-boots that snapshot. The Phase-12 cascade Stage 1 Qwen3-Reranker-8B run completed for all 10 queries in ~5 min/query but its rerank-recall numbers are low at small K (best R@50=0.222 for val_006, several queries at 0.000) and the Stage 1 vs fusion comparison print block raised `ValueError: Sign not allowed in string format specifier` (using `>+7` on a string `'Δ'`), so Stages 1B / 2 / 3 / 4 were not executed in this run. The 35-missed-gold diagnosis points to mid-paragraph FR-language BGE reasoning/facts paragraphs and orphan StBOG / IPRG / ZPO / ZGB / OR law rows as the hardest-to-recover targets across queries.
