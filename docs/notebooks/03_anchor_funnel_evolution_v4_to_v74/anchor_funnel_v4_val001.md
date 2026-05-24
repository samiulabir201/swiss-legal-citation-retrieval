# anchor_funnel_v4_val001.ipynb

**Path:** notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v4_val001.ipynb

Goal stated in the notebook: push val_001 R@1000 from v3's 0.286 (12/42) to >= 0.70 (>= 30/42) without hardcoding. Stretch: R@1000 >= 0.80 (>= 34/42).

## Configuration

**Runtime**
- Python: 3.12.13
- PyTorch: 2.10.0+cu128
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB VRAM
- Kernel: `python3`

**Models**
- Query expansion: `Qwen/Qwen3-32B` (bf16, device_map=auto, ~65 GB; loaded then freed)
- Embeddings (query side): `Qwen/Qwen3-Embedding-8B` (bf16); mean-pool with attention mask, L2-normalized
- Corpus embeddings (pre-computed): 27 `.npy` chunks, fp16, shape `(2_652_248, 4096)`, ~21.7 GB VRAM after concat

**Libraries**
- `torch`, `transformers` (`AutoTokenizer`, `AutoModel`, `AutoModelForCausalLM`)
- `sqlite3` (in-memory FTS5, tokenize=`unicode61 remove_diacritics 2`)
- `pyarrow.parquet`, `numpy`
- stdlib: `csv` (field_size_limit `2**31 - 1`), `json`, `re`, `collections`, `gc`, `pathlib`

**CONFIG dict (Cell 7)**

```
topk_final               = 1000
budget_law_direct        = None    # uncapped guarantee channel
budget_court_statute     = 600
budget_sibling           = 500
budget_concept           = 600
budget_term              = 500
budget_per_area          = 150
budget_co_citation       = 400
budget_bm25              = 600
budget_vector            = 800
budget_vector_enriched   = 800
rrf_k                    = 60
guarantee_channels       = ["law_direct_match", "per_area_bedrock"]
per_area_top_n           = 100
co_citation_top_k_per_target = 8
co_citation_min_co_count = 20
concept_substring_top_k  = 6
bm25_max_query_terms     = 60
bm25_min_token_len       = 3
vector_emb_model         = "Qwen/Qwen3-Embedding-8B"
vector_topk              = 800
qwen_query_model         = "Qwen/Qwen3-32B"
qwen_max_new_tokens      = 1024
noise_paragraph_roles    = {"notification","header","empty","metadata"}
lowercase_concepts       = True
lowercase_terms          = True
```

**Statute canonicalization aliases** (`CODE_ALIAS`):
`CPP -> StPO, CP -> StGB, CC -> ZGB, CO -> OR, LTF -> BGG, LACI -> AVIG, LAA -> UVG, LP -> SchKG, LDIP -> IPRG, Cst -> BV, Cst. -> BV, STPO -> StPO, OBG -> OR`.

**Legal-area default code fallback** (`LEGAL_AREA_DEFAULT_CODE`) covers criminal law/procedure -> StPO/StGB, civil law -> ZGB, obligations -> OR, civil procedure -> ZPO, constitutional/public -> BV, administrative -> VwVG, social insurance -> ATSG, tax -> DBG.

## Data

Auto-detected `DATA_ROOT = /content/drive/MyDrive/swiss_law` (Drive). Search order tried:
`/content/drive/MyDrive/swiss_law`, `/content/drive/MyDrive/swiss_citation_extraction`, `/content/swiss_citation_extraction`, `E:/swiss_citation_extraction`, `Path.cwd()`.

All inputs reported OK in the run:

- `val_csv`: `/content/drive/MyDrive/swiss_law/data/val.csv`
- `law_llm`: `/content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl` (173,033 rows)
- `court_v5`: `/content/drive/MyDrive/swiss_law/artifacts_v2/court_authority_cards_v5_unified.jsonl` (2,476,315 rows)
- `emb_dir`: `/content/drive/MyDrive/swiss_law/artifacts/embeddings` (27 `qwen3_8b_unified_chunk*.npy`)
- `emb_manifest`: `/content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_manifest.parquet` (cols: `doc_id`, `family`, `citation`, `row_index`; 2,652,248 rows)
- `out_dir`: `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v4`

**Query under test:** `val_001` — 42 gold citations.
Query (first 240 chars): *"May a court lawfully order a three-month extension of pre-trial detention under Art. 221 Abs. 1 lit. b StPO (risk of collusion) consistent with the principle of proportionality when the accused—detained after an alleged late-night assault a..."*

**Output artifacts written by Cell 18:**
- `val_001_query_targets.json`
- `val_001_v4_per_channel.json`
- `val_001_v4_per_gold_trace.tsv`
- `val_001_v4_top1000_pool.txt`
- `val_001_v4_summary.md`

## Pipeline

### Stage 1 — Environment & paths (Cells 1-3, 5)
GPU/Python check; mount Drive; auto-detect `DATA_ROOT`; resolve paths; flag `EMB_AVAILABLE` based on presence of chunk files.

### Stage 2 — Knob panel (Cell 7)
Single CONFIG dict; all knobs architectural (no target hardcoding). Budgets intentionally sum > 1000 so RRF can reorder.

### Stage 3 — Load val_001 (Cell 9)
Read `val.csv` with `csv.DictReader`, find row where `query_id == "val_001"`, split `gold_citations` on `;`. Yields 42 gold strings.

### Stage 4 — One-pass index build (Cell 11)
Streams both JSONL files once to build:
- `cit_to_doc_ids`: citation string -> list of doc_ids
- `doc_meta`: doc_id -> {citation, family, court_base, paragraph_role, is_notification_paragraph}
- `idx_law_direct`: canonical statute -> set(law doc_ids)
- `idx_court_statute`: canonical statute -> set(court doc_ids) (with legal-area-based fallback for code-less anchors)
- `idx_case_anchor`: canonical case -> set(doc_ids)
- `idx_court_base`: court_base -> set(doc_ids)
- `idx_concept_en`, `idx_term_orig`: lowercased token -> set(doc_ids)
- `legal_area_per_doc`: doc_id -> lowercased `legal_area_static`
- `search_text`: doc_id -> truncated text (<=2000 chars) for BM25
- `co_citation_pairs`: Counter over unordered canonical pairs co-occurring per court row

Doc-id scheme: `law:{i}` / `court:{i}`. Canonicalizers: `statute_anchor_canonical` (regex `art.?\s*(\d+[a-z]?)` + last code-like token), `case_anchor_canonical` (BGE pattern `BGE X Y Z` or docket `\d[A-Z]_\d+/\d{4}`).

### Stage 5 — Gold mapping (Cell 13)
Look up each of the 42 gold citation strings in `cit_to_doc_ids`. All 42 mapped to exactly one doc_id each (`total_gold = 42`).

### Stage 6 — Per-area corpus bedrock (Cell 15)
Re-scan `idx_court_statute`; for each `(canon, did)` pair count distinct `court_base` values per `legal_area_static`. Produces `per_area_canon_count[area]` -> Counter(canonical -> distinct-court-base count). 26 areas built.

### Stage 7 — Co-citation expansion index (Cell 17)
Filter `co_citation_pairs` by `co_citation_min_co_count = 20`, sort, keep top 8 per canonical -> `co_neighbours`. Frees raw `co_citation_pairs` afterward.

### Stage 8 — BM25 FTS5 index (Cell 19)
Build in-memory SQLite FTS5 (`tokenize='unicode61 remove_diacritics 2'`) over `search_text`, batches of 50k. Free `search_text` after load. `bm25_search` issues `... WHERE docs MATCH ? ORDER BY s LIMIT ?` with OR-joined quoted tokens (capped at `bm25_max_query_terms = 60`), flips bm25() sign.

### Stage 9 — Vector setup (Cell 21)
Read parquet manifest; build `manifest_row_to_did` via `(family, citation)` match (99.9% coverage). `np.load` each chunk -> `np.concatenate` -> `torch.from_numpy(...).to('cuda')`. Sets `VECTOR_OK = True`.

### Stage 10 — Query encoding (Cells 23, 27)
Lazy-load Qwen3-Embedding-8B (bf16, device_map=auto). `encode_query`: truncate to 4096, forward, mean-pool with mask, L2-normalize, return float32 numpy. `vector_search`: normalize query, matmul against `E_GPU`, `torch.topk(scores, k)`, map manifest_row -> doc_id.

Two encodings:
- `q_emb_raw`: raw QUERY only
- `q_emb_enriched`: `QUERY + "\n\nKeywords: " + " ; ".join(term_de + term_fr + concept_en + legal_area_keywords)` (74 keywords, 2705 chars)

### Stage 11 — Query expansion via Qwen3-32B (Cell 25)
Strict-JSON prompt requesting `statute_targets`, `case_targets`, `concept_targets_en`, `term_targets_de`, `term_targets_fr`, `legal_area_keywords`. Chat template with `enable_thinking=False`, greedy decode (`do_sample=False`, `temperature=0.0`, `max_new_tokens=1024`). Brace-slice the completion, `json.loads`, then delete model + `torch.cuda.empty_cache()` so embedding model can be loaded next.

### Stage 12 — Channel functions (Cell 29)
- `channel_law_direct(canon_set, idx, budget)`: counter over doc_ids in `idx[canon]` for each canon; uncapped if `budget is None`.
- `channel_court_statute`: same shape, capped.
- `channel_sibling`: for each seed court doc, gather all docs sharing `court_base`, minus seeds.
- `expand_concepts_strict`: substring containment in either direction against corpus concept vocab; sorts by abs-length-diff.
- `channel_concept`, `channel_term`: counter over `idx_concept_en` / `idx_term_orig`.
- `channel_per_area_bedrock`: legal_area_keywords -> matching corpus areas -> top-N canonicals -> law rows for those canonicals.
- `channel_co_citation`: for each LLM statute target, fetch top co-neighbours, then their law + court rows.
- `channel_vector`: thin wrapper over `vector_search`.

### Stage 13 — Run all channels (Cell 31)
1. Build `llm_statute_canons` from LLM `statute_targets`; expand with `co_neighbours` -> `co_expanded_canons`.
2. Strict-substring concept expansion against corpus vocab.
3. Run all 10 channels with their budgets; build `CHANNELS` list of `(name, hits)`.
4. Print per-channel size, gold-in-channel, recall; print union size + upper bound.

### Stage 14 — RRF fusion + negative gate + R@K curve (Cell 33)
- `rrf_fuse`: `score[did] += 1 / (k + rank + 1)` summed across channels (`k = 60`).
- `apply_neg_gate`: drops docs whose `is_notification_paragraph` is True or `paragraph_role` in `{notification, header, empty, metadata}`.
- Guarantee pool: hits from `law_direct_match` + `per_area_bedrock` (also gated), prepended.
- Final top-K filled from gated RRF tail until size = 1000.
- Print R@K curve for K in `[50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 5000]`.

### Stage 15 — Per-gold trace (Cell 35)
For each gold citation, look up best rank of its doc_id in `final_topk` plus the channel set that surfaced it. Sort by rank, print, count misses.

### Stage 16 — Save artifacts + summary (Cell 37)
Write `val_001_query_targets.json` (already saved in Cell 25), `val_001_v4_per_channel.json`, `val_001_v4_per_gold_trace.tsv`, `val_001_v4_top1000_pool.txt`, `val_001_v4_summary.md`.

### Stage 17 — Cleanup (Cell 39)
Close sqlite con; delete indexes, `E_GPU`, `EMB_MODEL`, `EMB_TOK`; `gc.collect()`; `torch.cuda.empty_cache()`.

## Results

**Index build (Cell 11):**
- Law: 173,033 rows in 6.3s
- Court: 2,476,315 rows in 179.2s
- Total docs: 2,649,348
- Unique citations: 2,158,211
- Index sizes: law_direct=49,288; court_statute=81,988; case=157,227; court_base=178,593; concept=292,946; term=363,331
- Co-citation pairs: 1,142,790

**Gold mapping:** 42/42 gold citations mapped; 42 unique gold doc_ids.

**Per-area bedrock (Cell 15):** 26 areas, built in 8.6s. Top-8 examples:
- `criminal law and criminal procedure`: `[(66 BGG, 17910), (106 BGG, 10328), (42 BGG, 10299), (64 BGG, 8388), (97 BGG, 8130), (105 BGG, 7397), (108 BGG, 7313), (9 BV, 6436)]`
- `criminal procedure and coercive measures`: `[(66 BGG, 5519), (78 BGG, 4035), (42 BGG, 3449), (81 BGG, 3442), (80 BGG, 2812), (64 BGG, 2695), (93 BGG, 2573), (108 BGG, 2404)]`
- `criminal law`: `[(278 BStP, 587), (66 BGG, 521), (121 BGG, 520), (278 PPF, 433), (42 BGG, 331), (63 StGB, 320), (269 PPF, 316), (277b BStP, 311)]`
- `criminal law and administrative criminal law`: `[(156 OG, 171), (105 OG, 117), (156 OJ, 112), (104 OG, 107), (16 SVG, 100), (104 OJ, 94), (17 SVG, 93), (114 OJ, 87)]`

**Co-citation neighbours (Cell 17):** 5,367 canonicals indexed. Neighbours of `221 StPO`:
`36 BV: 915; 31 BV: 783; 10 BV: 731; 212 StPO: 714; 221 BV: 646; 237 StPO: 487; 5 BV: 393; 5 StPO: 272`.

**BM25 (Cell 19):** FTS5 built over 2,649,348 docs in 86.0s.

**Vector setup (Cell 21):** manifest rows 2,652,248. Mapping coverage 2,649,348/2,652,248 = 99.9% in 5.5s. `E_GPU shape=(2652248, 4096) dtype=torch.float16, VRAM=21.7 GB, 487.2s`.

**Qwen3-32B query expansion (Cell 25):**
- `statute_targets` (6): `Art. 221 StPO, Art. 222 StPO, Art. 227 StPO, Art. 212 StPO, Art. 100 BGG, Art. 42 BGG`
- `case_targets` (4): `BGE 146 III 123, BGE 142 III 456, BGE 138 III 789, 1B_123/2024`
- `concept_targets_en` (29): `proportionality, pre-trial detention, risk of collusion, risk of reoffending, evidentiary integrity, ...`
- `term_targets_de` (25): `Voruntersuchungshaft, Verhältnismäßigkeitsprinzip, Zeugenbeeinflussung, Beweisbeeinträchtigung, Wiedereintrittsgefahr, ...`
- `term_targets_fr` (15): `détention préjudiciaire, principe de proportionnalité, manipulation de témoins, altération de preuves, risque de récidive, ...`
- `legal_area_keywords` (5): `criminal procedure, pre-trial detention, proportionality, evidence preservation, judicial review`

After expansion: Qwen3-32B freed, CUDA mem = 21.7 GB.

**Query encoding (Cell 27):** raw 43.8s (includes embedding model load), enriched 0.1s. Enriched query: 2705 chars, 74 keywords appended.

**Channel diagnostics (Cell 31):**
- `Statute canons: LLM=6, +co-citation=37`
- `Concepts: LLM=34 -> expanded=140`

| channel | size | gold_in_ch | recall |
|---|---:|---:|---:|
| law_direct_match | 114 | 7 | 16.7% |
| court_statute | 600 | 0 | 0.0% |
| co_citation | 400 | 0 | 0.0% |
| per_area_bedrock | 150 | 3 | 7.1% |
| sibling_expansion | 500 | 0 | 0.0% |
| concept_en | 600 | 2 | 4.8% |
| term_orig | 500 | 0 | 0.0% |
| bm25 | 600 | 3 | 7.1% |
| vector_raw | 800 | 1 | 2.4% |
| vector_enriched | 800 | 1 | 2.4% |

- Union: 4,364 unique doc_ids
- Gold in union: 10/42 (UPPER BOUND on R@K)

**Fusion + R@K curve (Cell 33):**
- Pre-gate fused: 4,364
- Post-gate: 4,204
- Guarantee pool: 203
- Final top-1000: 1,000

| K | gold/42 | recall |
|---:|---:|---:|
| 50 | 2/42 | 4.8% |
| 100 | 7/42 | 16.7% |
| 200 | 7/42 | 16.7% |
| 300 | 7/42 | 16.7% |
| 500 | 8/42 | 19.0% |
| 750 | 10/42 | 23.8% |
| 1000 | 10/42 | 23.8% |

```
PASS CRITERION: R@1000 >= 0.70 (>= 30/42)
OBSERVED:       R@1000 = 0.238  (10/42)
```

**Per-gold trace (Cell 35):** 10 found, 32 missed.

Found (rank, channels, citation):
- 13  `law_direct_match`  `Art. 393 Abs. 1 StPO`
- 19  `bm25,law_direct_match`  `Art. 227 Abs. 1 StPO`
- 62  `bm25,law_direct_match,per_area_bedrock`  `Art. 221 Abs. 1 StPO`
- 63  `law_direct_match,per_area_bedrock`  `Art. 221 Abs. 2 StPO`
- 69  `law_direct_match,per_area_bedrock`  `Art. 100 Abs. 1 BGG`
- 78  `law_direct_match`  `Art. 212 Abs. 3 StPO`
- 83  `bm25,law_direct_match`  `Art. 222 StPO`
- 390 `concept_en`  `BGE 133 I 270 E. 3.4.2`
- 627 `concept_en`  `1B_357/2022 E. 3.1`
- 743 `vector_enriched,vector_raw`  `BGE 132 I 21 E. 3.2.2`

Missed (32): `Art. 140 Abs. 1 StGB; Art. 396 Abs. 1 StPO; Art. 382 Abs. 1 StPO; Art. 385 Abs. 1 StPO; Art. 390 Abs. 2 StPO; Art. 422 Abs. 1 StPO; Art. 422 Abs. 2 StPO; Art. 428 Abs. 1 StPO; Art. 135 Abs. 4 StPO; Art. 135 Abs. 3 StPO; Art. 37 Abs. 1 StBOG; Art. 39 Abs. 1 StBOG; BGE 137 IV 122 E. 6.2; BGE 137 IV 122 E. 6.4; BGE 137 IV 122 E. 4.2; BGE 132 I 21 E. 3.2; 1B_210/2023 E. 4.1; 1B_536/2018 E. 5.1; BGE 139 IV 270 E. 3.1; BGE 133 I 168 E. 4.1; BGE 143 IV 168 E. 5.1; BGE 137 IV 122 E. 4.1; BGE 132 I 21 E. 3.2.1; 1B_90/2021 E. 2.1; 1B_90/2021 E. 2.4; 7B_496/2025 E. 3.2; 7B_231/2025 E. 4.1; 7B_69/2024 E. 3.3.2; 7B_301/2024 E. 2.4; 7B_12/2025 E. 2.2; 1B_15/2023 E. 3.1; 1B_28/2022 E. 4.1`.

**Artifacts (Cell 37):** `Saved artifacts to: /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v4`.

**Cleanup (Cell 39):** `cleaned up.`

## Summary

The v4 anchor-funnel added four new channels on top of v3 (per-legal-area corpus bedrock replacing the global BGG-dominated bedrock, corpus co-citation expansion, in-memory BM25 over enrichment text, and two Qwen3-Embedding-8B vector channels — raw and DE/FR-keyword-enriched). End-to-end execution completed on the Blackwell 95 GB GPU with all 42 gold citations mapped to corpus doc_ids, but R@1000 landed at 0.238 (10/42), well below the 0.70 pass criterion and even below v3's reported 0.286 baseline. What worked: `law_direct_match` surfaced 7/42 (the StPO/BGG articles directly named or co-cited), `bm25` and `per_area_bedrock` added redundant signal on a few statutes, and the vector channels were the only ones to surface one BGE result (`BGE 132 I 21 E. 3.2.2` at rank 743). What failed: 32 gold items — heavy in `Art. 422/428/135 StPO`, `StBOG`, and docket cases like `7B_*/2024-2025` and `1B_*/2022-2023` — were not surfaced by any channel because the LLM did not name them, co-citation neighbours did not cover them, and concept/term/vector matches did not retrieve them; `court_statute`, `co_citation`, `sibling_expansion`, and `term_orig` each contributed zero gold. Lessons recorded by the run: union recall of 10/42 (23.8%) is itself the hard ceiling — channel breadth, not RRF fusion, is the bottleneck, and reaching 0.70 will require either much richer LLM target expansion (especially case-anchor coverage of `7B_*` dockets and `Art. 135/382-396/422/428 StPO` procedural cluster) or qualitatively different channels.
