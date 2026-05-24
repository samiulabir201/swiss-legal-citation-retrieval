# pool_v75_multiquery_hyde_test_no_lift.ipynb

**Path:** `notebooks/02_v75_multiquery_canonical_pool/pool_v75_multiquery_hyde_test_no_lift.ipynb`

Anchor-Funnel v7.5 multi-query pool with single-paragraph HyDE channel added (`vector_hyde`, 2000 budget, RRF weight 1.5). Runs the full pipeline on every query in `val.csv` with no per-id branching. `topk_final = 50,000`. Pass criterion stated in cell 0: R@1000 ≥ 0.60 for val_001 (≥ 26/42 gold), stretch ≥ 0.90 / 38+. Final macro R@50k = 0.885 (vs canonical 0.901 baseline) — HyDE did not lift macro recall in this run.

## Configuration

### Models

| Model | Use | Precision | VRAM |
|---|---|---|---|
| `Qwen/Qwen3-32B` | Query expansion (structured targets JSON) + HyDE single-paragraph answer | bfloat16 | ~65 GB |
| `Qwen/Qwen3-Embedding-8B` | Query encoding (raw / enriched / hyde) | fp16 | ~16 GB |
| `E_GPU` corpus embedding | Pre-encoded Qwen3-Embedding-8B corpus tensor (2,652,248 × 4096), brute-force GPU cosine | fp16 | 20.2 GB |

### Hardware

- Python 3.12.13, PyTorch 2.10.0+cu128
- 1 × NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB VRAM
- Colab metadata: `machine_shape: hm`, `gpuType: G4`
- Mounted: `/content/drive`

### Libraries

- `torch`, `numpy`, `pandas`, `sqlite3` (FTS5 in-memory), `gc`, `re`, `json`, `math`, `time`
- `transformers.AutoTokenizer / AutoModelForCausalLM` (Qwen3-32B load)
- `sentence_transformers.SentenceTransformer` (Qwen3-Embedding-8B query side)

### CONFIG dict (verbatim from cell 8 output)

```
topk_final:                50000
budget_law_direct:         None
budget_court_statute:      8000
budget_concept:            3000
budget_term:               2500
budget_per_area:           1500
budget_co_citation:        2500
budget_bm25:               2000
budget_vector:             2000
budget_vector_enriched:    2000
budget_vector_hyde:        2000
budget_backprop:           2000
budget_sibling:            5000
budget_graph_forward:      5000
budget_graph_reverse:      3000
enable_graph_2hop:         false
budget_graph_2hop:         1500
rrf_k:                     60
guarantee_channels: [law_direct_match, per_area_bedrock, statute_backprop,
                     concept_en, graph_forward, sibling_expansion, term_orig]
guarantee_per_channel:     130
channel_weights:
  statute_backprop:  2.5
  graph_forward:     2.0
  concept_en:        1.8
  court_statute:     1.5
  per_area_bedrock:  1.5
  vector_raw:        1.0
  vector_enriched:   1.0
  vector_hyde:       1.5
  term_orig:         1.2
  law_direct_match:  1.2
  sibling_expansion: 1.0
  bm25:              0.8
  co_citation:       0.7
  graph_reverse:     0.5
  graph_2hop:        0.0
code_family_top_k:               8
per_area_top_n:                  1000
co_citation_top_k_per_target:    50
co_citation_min_co_count:        50
co_citation_max_neighbour_count: 50000
concept_substring_top_k:         6
bm25_max_query_terms:            60
bm25_min_token_len:              3
vector_emb_model:        Qwen/Qwen3-Embedding-8B
vector_topk:             800
qwen_query_model:        Qwen/Qwen3-32B
qwen_max_new_tokens:     1024
enhance_top_k_codes:     5
enhance_repeat_count:    5
enhance_min_idf:         1.0
noise_paragraph_roles:   {notification, header, empty, metadata}
lowercase_concepts:      true
lowercase_terms:         true
```

### Key constants/regex

- `CODE_ALIAS`: CPP→StPO, CP→StGB, CC→ZGB, CO→OR, LTF→BGG, LACI→AVIG, LAA→UVG, LP→SchKG, LDIP→IPRG, Cst→BV, STPO→StPO, OBG→OR
- `ART_RE = re.compile(r"art\.?\s*(\d+[a-z]?)", re.I)`
- `CASE_BGE_RE = re.compile(r"BGE\s+(\d+)\s+([IVX]+)\s+(\d+)")`
- `CASE_DOCKET_RE = re.compile(r"\b(\d[A-Z]_\d+/\d{4})\b")`
- `LEGAL_AREA_DEFAULT_CODE`: 10 entries mapping legal areas to default codes (criminal procedure→StPO, civil law→ZGB, obligations→OR, etc.)
- `_TERM_LEMMA_SUFFIXES = ("en","es","em","er","e","n","s")` (German lemmatizer, 4-char stem floor, recursive)
- `_SUBSTANTIVE_ROLES = {"reasoning","legal_standard","application","holding"}` (paragraph_role boost: 1.5×)
- `_ROLE_W` in court_statute scoring: legal_standard/reasoning/application/holding=1.5, facts/procedural_history/citation/neutral_default=1.0, costs/disposition/notification=0.6, neutral=0.4
- `SUBSTANTIVE_ROLES` for neg-gate: `{facts, reasoning, legal_standard, application, holding, citation, procedural_history}`
- `K_SAMPLES = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 25000, 35000, 50000]`
- HyDE max_new_tokens = 400 (deterministic, `do_sample=False`); fallback to raw query if `len(hyde) < 40`

### Channel inventory (15 channels)

`law_direct_match`, `court_statute`, `co_citation`, `per_area_bedrock`, `statute_backprop`, `sibling_expansion`, `graph_forward`, `graph_reverse`, `graph_2hop` (disabled, weight 0), `concept_en`, `term_orig`, `bm25`, `vector_raw`, `vector_enriched`, `vector_hyde`.

## Data

### Auto-detected `DATA_ROOT`

`/content/drive/MyDrive/swiss_law` (resolved from candidate list: `/content/drive/MyDrive/swiss_law`, `/content/drive/MyDrive/swiss_citation_extraction`, `/content/swiss_citation_extraction`, `E:/swiss_citation_extraction`, `cwd`).

### PATHS dict (all flagged `OK`)

| Key | Path |
|---|---|
| `val_csv` | `/content/drive/MyDrive/swiss_law/data/val.csv` |
| `law_llm` | `/content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl` |
| `court_v5` | `/content/drive/MyDrive/swiss_law/artifacts_v2/court_authority_cards_v5_unified.jsonl` |
| `emb_dir` | `/content/drive/MyDrive/swiss_law/artifacts/embeddings` |
| `emb_manifest` | `/content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_manifest.parquet` |
| `graph_db` | `/content/drive/MyDrive/swiss_law/data_insights/citation_graph_extracted.sqlite` |
| `out_dir` | `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7` |

`EMB_AVAILABLE = True`, `GRAPH_AVAILABLE = True`.

### Input artifacts loaded

- **val.csv** — 10 English queries with gold citation lists (gold counts: val_001=42, val_002=36, val_003=47, val_004=10, val_005=11, val_006=18, val_007=19, val_008=29, val_009=14, val_010=25; total = 251).
- **law_llm_descriptors_0000000_all.jsonl** — streamed; 173,033 law rows indexed in 11.2 s.
- **court_authority_cards_v5_unified.jsonl** — streamed; 2,476,315 court rows indexed in 192.9 s.
- **embeddings/qwen3_8b_unified_chunk*.npy** — 27 fp16 chunks concat'd to single `E_GPU` of shape (2,652,248, 4096); load 481.5 s.
- **embeddings/qwen3_8b_unified_manifest.parquet** — 2,652,248 rows with columns `[doc_id, family, citation, row_index]`; 99.9% mapping to internal doc_ids (2,649,348/2,652,248).
- **data_insights/citation_graph_extracted.sqlite** — `edges` table, dataset='court_considerations'; 20,494,436 edges loaded, 3,155,263 skipped (citation not in corpus); out-deg avg 27.2, in-deg avg 14.0; load 108.1 s.

### Output artifacts written to `PATHS["out_dir"]`

- `final_topk_<qid>.json` × 10 (rank, doc_id, citation)
- `summary_multiquery.json` (per-query R@K, channel recalls, K-curve)
- `targets_multiquery.json` (LLM expansion outputs per query)
- `config.json` (CONFIG snapshot)

## Pipeline

### Stage 1 — Setup (cells 2–10)

- GPU detected, Drive mounted.
- DATA_ROOT auto-resolved; all 7 paths verified `OK`.
- CONFIG knob panel printed.
- 10 val queries loaded into `ALL_QUERIES` list, each `{query_id, query_text, gold}`.

### Stage 2 — Build corpus indexes (cells 11–14)

Single streaming pass over both JSONLs produces:

- `cit_to_doc_ids` — 2,158,211 unique citations
- `doc_meta[did]` — 2,649,348 docs total (law + court)
- `idx_law_direct` (49,288 canons), `idx_court_statute` (81,988), `idx_case_anchor` (157,227), `idx_court_base` (178,593), `idx_concept_en` (292,946), `idx_term_orig` (363,331), `idx_term_lemma` (521,146), `term_orig_keys` (363,331)
- `co_citation_pairs` Counter — 1,142,790 pairs
- `tlf[token][code]` — 91,173 tokens, avg 11.7 codes/token
- `doc_statute_anchors`, `legal_area_per_doc`, `doc_language`, `search_text`

Gold→doc_id resolution (cell 14): 100% of gold mapped for every query. Doc-id counts ≥ gold counts (val_002: 36 gold → 38 dids; val_008: 29 → 30) reflect cases where one gold citation appears under multiple doc_ids.

### Stage 3 — Citation graph (cells 15–16)

- Builds `idx_graph_out` / `idx_graph_in` from edges using `cit_to_doc_ids` map.
- 4 alias passes already baked into the sqlite (intra-judgment back-refs, date-stripped aliases, E.-range expansion, case-level fan-out).
- `idx_judgment_importance` per court_base: median=7, p75=44, max=204,686 (built in 1.1 s).

### Stage 4 — Per-area bedrock + co-citation (cells 17–20)

- `per_area_canon_count` over `legal_area_per_doc`: 26 areas.
  - `criminal law and criminal procedure` top 8: 66 BGG (28149), 42 BGG (20746), 106 BGG (19537), 64 BGG (13843), 108 BGG (12101), 97 BGG (12075), 105 BGG (11759), 81 BGG (11102).
  - `civil law` top 8: 63 OJ (3189), 8 ZGB (2740), 55 OJ (2518), 64 OJ (2235), 55 OG (2085), 63 OG (1906), 159 OG (1876), 9 BV (1875).
- `code_pair_count` derived from co_citation_pairs: 54,790 directed pairs. StPO top relateds: BGG (50026), BV (34792), StGB (18981), EMRK (6826), Satz (2812), CEDH (2549), OR (1961), ZGB (1123).
- `co_neighbours` after `co_citation_min_co_count = 50` and `co_citation_max_neighbour_count = 50000`: 2,282 canonicals. Sample for `221 StPO`: 36 BV (915), 31 BV (783), 10 BV (731), 212 StPO (714), 221 BV (646), 237 StPO (487), 5 BV (393), 5 StPO (272).

### Stage 5 — BM25 (cells 21–22)

- **5.A — legacy single-language FTS5** over 2,649,348 docs, `unicode61 remove_diacritics 2` tokenizer; built in 84.0 s.
- **5.B — per-language FTS5** (`de`, `fr`, `it`, `en`), routed by `doc_language[did]`; built in 70.6 s. Doc counts: de=1,593,249, fr=793,023, it=125,583, en=137,493.
- **5.C — enhance()** corpus-derived BM25 lexicon expansion using `tlf` + IDF, top-K codes appended `enhance_repeat_count=5` times.
- **5.E — bm25_search_multilang()** runs per-language with language-appropriate target enrichment terms, normalizes raw bm25 by `log(N_lang + e)` for cross-language comparability, max-merges across languages.

### Stage 6 — Vector channel setup (cells 23–26)

- Manifest mapped (39.3 s), 27 fp16 chunks concatenated to GPU: `E_GPU shape=(2652248, 4096) dtype=float16`, VRAM used 20.2 GB / free 74.7 GB.
- `vector_search(q_emb, k)`: brute-force `E_GPU @ q_normalized`, GPU topk.
- `encode_query(text)` via SentenceTransformer with `prompt_name="query"`, normalized embeddings.

### Stage 7 — Query expansion + HyDE (cells 27–30)

- Qwen3-32B loaded (~65 GB bf16) in 205.4 s.
- **Pass 1 (deterministic)** generates structured targets JSON; **Pass 2 (sampled, temperature=0.4, top_p=0.95, 1.5× max_new_tokens)** retries if first parse failed or returned <10 items. Only val_005 needed retry (108.7 s vs ~50 s normal).
- All 10 queries produced full target schemas with 20 statute_targets and 19–26 each of concept_targets_en / term_targets_de / term_targets_fr / 5–6 legal_area_keywords.
- **HyDE pass** with same Qwen3-32B: single 4–6 sentence Bundesgericht/Tribunal fédéral consideration paragraph, 400 max_new_tokens, English with parenthetical German+French legal terms. All 10 produced 1045–1468 chars (no short fallback fired).
- Qwen3-32B freed (CUDA mem returned to 20.2 GB = E_GPU only).
- Qwen3-Embedding-8B loaded; produced `ALL_Q_EMB_RAW` (raw query), `ALL_Q_EMB_ENRICHED` (raw + first 60 terms+concepts), `ALL_Q_EMB_HYDE` (HyDE paragraph encoded).

### Stage 8 — Channels (cells 31–34)

Channel implementations in cell 32 (functions: `channel_law_direct`, `channel_court_statute` with multi-match + specificity + paragraph_role boost, `channel_sibling`/`channel_graph_forward`/`channel_graph_reverse` with `_judgment_factor = sqrt(1 + log(1 + importance))` and `_role_boost`, `channel_graph_2hop`, `expand_concepts_weighted` (exact 1.0 / substring 0.85×ratio / token-overlap), `channel_concept`, `channel_term` (exact 1.0 / lemma 0.7 / substring Q⊂T = |Q|/|T| or T⊂Q = |T|/|Q|), `channel_per_area_bedrock` filtered by corpus-derived statute_target_codes, `channel_statute_backprop` with rare-canon specificity 1/log(2+global_ct), `channel_co_citation`, `channel_vector`).

`run_channels()` (cell 34) builds:
- `llm_statute_canons` and `co_expanded_canons` from targets
- `statute_target_codes` expanded via `code_family_top_k=8` corpus-derived related codes
- weighted concept expansion via `expand_concepts_weighted(top_k=25)`
- seed set = union of court hits across topical + vector channels → feeds sibling/graph_forward/graph_reverse
- `backprop_seed` = seed ∪ sibling_court_hits ∪ graph_fwd_court_hits ∪ graph_rev_court_hits → statute_backprop
- returns 15-channel CHANNELS list + `channel_recalls` + `union_size`/`union_gold`

### Stage 9 — RRF fusion + master loop (cells 35–36)

- `rrf_fuse(channels, k=60, weights=CONFIG.channel_weights)` weighted RRF: `score(did) = sum_ch weights[ch] / (60 + rank + 1)`.
- `apply_neg_gate`: keep if paragraph_role ∈ SUBSTANTIVE_ROLES; drop if `is_notification_paragraph` or role ∈ noise_paragraph_roles.
- `round_robin_guarantee(channels, guarantee_channels, per_channel_cap=130, total_cap=50000)`: 7 guarantee channels × cap 130 → ≤910 guaranteed slots prepended.
- Final list = guarantee (gated) + ranked_gated (deduped), truncated to `topk_final = 50000`.
- Master loop runs over all 10 queries; each emits the channel recall table + R@50k.

### Stage 10 — Aggregate diagnostics (cells 37–44)

- Per-query summary table.
- Macro-mean R@K curve over `K_SAMPLES`.
- Per-channel mean recall (ranked).
- Missed-gold diagnostic: for each query with R@K < 1.0, prints each gold doc_id absent from every channel union, with family/lang/court_base/role/text excerpt, plus aggregated breakdowns by family / language / paragraph_role / court_base prefix.

### Stage 11 — Save + cleanup (cells 45–48)

- Writes `final_topk_<qid>.json`, `summary_multiquery.json`, `targets_multiquery.json`, `config.json` to `out_dir`.
- Deletes `E_GPU`, `EMB_MODEL`; `torch.cuda.empty_cache()`; final VRAM 0.01 GB.

## Results

### Per-query R@50,000 (verbatim from cell 38)

```
query       gold      R@K     caught      union   gate_union
------------------------------------------------------------
val_001       42    0.929   39/42      39/42      39/42
val_002       36    0.778   28/36      28/36      28/36
val_003       47    0.745   35/47      35/47      35/47
val_004       10    1.000   10/10      10/10      10/10
val_005       11    1.000   11/11      11/11      11/11
val_006       18    0.944   17/18      17/18      17/18
val_007       19    0.895   17/19      17/19      17/19
val_008       29    0.828   24/29      24/29      24/29
val_009       14    0.857   12/14      12/14      12/14
val_010       25    0.880   22/25      22/25      22/25
------------------------------------------------------------
MEAN                0.885                  215/251    215/251
  (micro union=0.857, gate union=0.857)
```

### Macro mean R@K curve (verbatim cell 40)

```
     K   mean recall      min      max
----------------------------------------
    50        0.085    0.000   0.211
   100        0.095    0.000   0.300
   200        0.202    0.056   0.600
   300        0.280    0.128   0.600
   500        0.351    0.149   0.643
   750        0.418    0.170   0.643
  1000        0.495    0.213   0.800
  1500        0.564    0.277   0.900
  2000        0.619    0.362   0.900
  3000        0.659    0.447   1.000
  5000        0.724    0.532   1.000
  7500        0.791    0.574   1.000
 10000        0.815    0.596   1.000
 15000        0.865    0.702   1.000
 20000        0.882    0.745   1.000
 25000        0.885    0.745   1.000
```
(curve plateaus at 0.885 from K≈25,000 through 50,000)

### Per-channel mean recall across all queries (verbatim cell 42)

```
channel                 mean recall   mean size
--------------------------------------------------
statute_backprop             0.586       2000
graph_forward                0.307       5000
per_area_bedrock             0.285        982
vector_hyde                  0.273       1998
vector_enriched              0.220       2000
vector_raw                   0.213       2000
law_direct_match             0.141         84
concept_en                   0.097       3000
bm25                         0.097       2000
court_statute                0.079       4130
term_orig                    0.058       2500
co_citation                  0.022       1953
sibling_expansion            0.021       5000
graph_reverse                0.006       3000
graph_2hop                   0.000          0
```

### Per-query channel recall highlights (vector_hyde row, from cell 36 output)

| qid | vector_hyde recall | gold | hits |
|---|---|---|---|
| val_001 | 33.3% (14/42) | 42 | 2000 |
| val_002 | 0.0% (0/36) | 36 | 2000 |
| val_003 | 14.9% (7/47) | 47 | 2000 |
| val_004 | 40.0% (4/10) | 10 | 2000 |
| val_005 | 54.5% (6/11) | 11 | 1996 |
| val_006 | 16.7% (3/18) | 18 | 2000 |
| val_007 | 36.8% (7/19) | 19 | 1991 |
| val_008 | 10.3% (3/29) | 29 | 2000 |
| val_009 | 42.9% (6/14) | 14 | 2000 |
| val_010 | 24.0% (6/25) | 25 | 1995 |

### Missed-gold aggregate (verbatim cell 44)

Total missed across all queries: **39**

By family: court 26 (66.7%), law 13 (33.3%)

By language: `?` 39 (100.0%) (language metadata field absent from doc_meta enrichment used by diagnostic)

By paragraph_role: `(none)` 13 (33.3% — all 13 law rows), reasoning 10 (25.6%), facts 7 (17.9%), legal_standard 4 (10.3%), procedural_history 2 (5.1%), costs 1 (2.6%), application 1 (2.6%), disposition 1 (2.6%)

By court_base prefix: BGE 15 (38.5%), (no court_base = law row) 13 (33.3%), 6B 3 (7.7%), 8C 2 (5.1%), 5A 2 (5.1%), 7B 1 (2.6%), 9C 1 (2.6%), 1B 1 (2.6%), 2C 1 (2.6%)

### Per-query missed-gold detail (citations only, full text in cell 44)

- **val_001 (3 missed)**: `7B_496/2025 E. 3.2`, `Art. 37 Abs. 1 StBOG`, `Art. 39 Abs. 1 StBOG`
- **val_002 (10 missed)**: `BGE 135 V 39 E. 6.1`, `BGE 127 V 205 E. 4b`, `BGE 132 V 93 E. 4`, `BGE 148 V 21 E. 5.3`, `BGE 140 V 193 E. 3.2`, `8C_510/2020 E. 2.4`, `BGE 144 V 427 E. 3.2`, `8C_160/2016 E. 4.1`, `9C_623/2020 E. 4.2`, `Art. 18d IVG`
- **val_003 (12 missed)**: `BGE 142 III 48 E. 4.1.1`, `BGE 124 III 5 E. 4`, `1B_192/2022 E. 4.1.2`, `BGE 131 III 601 E. 3.1`, `BGE 131 III 106 E. 1.1`, `BGE 141 IV 360 E. 3.2`, `2C_501/2020 E. 5.1`, `Art. 37 Abs. 1 StBOG`, `Art. 39 Abs. 1 StBOG`, `Art. 467 ZGB`, `Art. 519 Abs. 1 ZGB`, `Art. 520 Abs. 1 ZGB`
- **val_004**: R@K=1.000, no missed gold
- **val_005**: R@K=1.000, no missed gold
- **val_006 (1 missed)**: `BGE 128 III 419 E. 2.2`
- **val_007 (2 missed)**: `Art. 15 OR`, `Art. 98 Abs. 2 IPRG`
- **val_008 (6 missed)**: `6B_904/2020 E. 1.1`, `BGE 149 IV 42 E. 3.5`, `BGE 131 III 91 E. 5.2`, `6B_1233/2016 E. 1` (reasoning), `6B_1233/2016 E. 1` (disposition), `BGE 141 IV 132 E. 3.4.1`
- **val_009 (2 missed)**: `5A_954/2015 E. 3.3`, `5A_561/2020 E. 5.1.1`
- **val_010 (3 missed)**: `Art. 176 Abs. 1 ZPO`, `Art. 181 Abs. 3 ZPO`, `Art. 300 ZPO`

### Pipeline timings (from outputs)

- Law index: 11.2 s; Court index: 192.9 s; Graph load: 108.1 s; Per-area bedrock: 2.1 s; Code-pair stats: 0.7 s
- FTS5 legacy build: 84.0 s; per-language FTS5: 70.6 s
- Manifest map: 39.3 s; embedding chunks → GPU: 481.5 s
- Qwen3-32B load: 205.4 s; 10 query-expansion calls: ~45–110 s each (val_005 retry: 108.7 s); 10 HyDE calls: 12.3–17.2 s each
- Per-query channel run + fusion: 9.1–17.7 s

### Pass criterion check (cell 0 stated pass = R@1000 ≥ 0.60 for val_001)

From val_001 per-K curve (the per-query curve isn't printed standalone; macro curve shows mean=0.495 at K=1000 with max=0.800). Per-query data is saved in `summary_multiquery.json["per_query"][qid]["curve"]` but is not echoed in cell outputs. The verdict relative to the stated pass criterion is not stated in the notebook itself.

### Files written

`Wrote artifacts to /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7` — `summary_multiquery.json`, `targets_multiquery.json`, `config.json`, `final_topk_<query_id>.json` × 10.

### Final VRAM

After cleanup: 0.01 GB.

## Summary

This notebook is a variant of the v7.5 canonical multi-query pool that adds a single-paragraph HyDE channel (`vector_hyde`, budget 2000, RRF weight 1.5) generated by Qwen3-32B as a 4–6 sentence Bundesgericht-style consideration with parenthetical DE/FR terms. The full 15-channel pipeline runs end-to-end on all 10 val queries, producing final macro R@50k = 0.885 vs the canonical v7.5 baseline of 0.901 — HyDE did not lift recall and the suffix "no_lift" in the filename records this verdict. The HyDE channel itself does land at mean recall 0.273 (4th overall, above raw and enriched vectors), but its hits overlap with channels already in the pool, so the union does not grow. The dominant channels remain `statute_backprop` (0.586) and `graph_forward` (0.307). Per-query R_max bottoms at val_003 = 0.745 (vs 0.766 in the canonical) — slightly worse, with 12 missed including 5 BGE/docket reasoning paragraphs and 5 law rows (StBOG and ZGB articles); missed-gold breakdown confirms the structural pattern that ~67% of irrecoverable gold is court paragraphs (predominantly BGE) and ~33% is law rows with empty `paragraph_role`. The lesson: adding a single-paragraph HyDE vector channel on top of the v7.5 stack contributes no incremental recall over the existing dense + statute + graph channels, consistent with the project finding that dense embedding has a structural ceiling (Obs 3) and that adding more channels of the same kind doesn't help once the pool already covers what their relationship-type can reach.
