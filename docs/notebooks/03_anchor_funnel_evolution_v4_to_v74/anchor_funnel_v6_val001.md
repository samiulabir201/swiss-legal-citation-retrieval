# anchor_funnel_v6_val001.ipynb

**Path:** notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v6_val001.ipynb

## Configuration

### Pass criterion
- R@1000 >= 0.60 (>= 26/42 for val_001).

### Models
- Query expansion: `Qwen/Qwen3-32B` (bf16, ~65 GB), `max_new_tokens=1024`, `do_sample=False`, `temperature=0.0`, chat template with `enable_thinking=False`. Freed immediately after expansion.
- Embedding: `Qwen/Qwen3-Embedding-8B` loaded via SentenceTransformer, `torch_dtype=bfloat16`, `attn_implementation=sdpa`, `padding_side=left` (last-token pooling for causal LM), `max_seq_length=4096`. Canonical instruction prefix used:
  - `"Instruct: Given an English-language legal question or scenario about Swiss federal law, retrieve the Swiss statute articles or federal court decision considerations that are most directly relevant to answering it.\nQuery: "`
- Reranker: dropped in v6 (v5 measured 48 min wall-time and recall regression 0.357 -> 0.333).

### Hardware (reported by notebook)
- Python 3.12.13
- PyTorch 2.10.0+cu128
- GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB
- Memory plan: Qwen3-32B (~65 GB) loaded then freed, then Qwen3-Embedding-8B + corpus `E_GPU` (~37 GB peak, 21.7 GB after model freed).

### Libraries
- `os, sys, json, time, math, gc, re, csv`
- `torch`, `transformers` (`AutoTokenizer`, `AutoModelForCausalLM`)
- `sentence_transformers.SentenceTransformer`
- `numpy`, `pyarrow.parquet`
- `sqlite3` (in-memory FTS5, `tokenize='unicode61 remove_diacritics 2'`)
- `collections.defaultdict, Counter`
- `google.colab.drive`

### CONFIG hyperparameters
```
topk_final = 1000
budget_law_direct      = None  (uncapped, guarantee channel)
budget_court_statute   = 600
budget_sibling         = 500
budget_concept         = 600
budget_term            = 500
budget_per_area        = 150
budget_co_citation     = 400
budget_bm25            = 600
budget_vector          = 800
budget_vector_enriched = 800
budget_backprop        = 400        (v6 NEW)
rrf_k = 60
guarantee_channels = ["law_direct_match", "per_area_bedrock", "statute_backprop"]
per_area_top_n = 100
co_citation_top_k_per_target  = 8
co_citation_min_co_count      = 50
co_citation_max_neighbour_count = 5000   (v5 FIX 3)
concept_substring_top_k = 6
bm25_max_query_terms = 60
bm25_min_token_len   = 3
vector_emb_model = "Qwen/Qwen3-Embedding-8B"
vector_topk      = 800
qwen_query_model     = "Qwen/Qwen3-32B"
qwen_max_new_tokens  = 1024
enhance_top_k_codes  = 5     (v6: corpus-derived BM25 lexicon expansion)
enhance_repeat_count = 5
enhance_min_idf      = 1.0
noise_paragraph_roles = {"notification", "header", "empty", "metadata"}
lowercase_concepts = True
lowercase_terms    = True
```

### Constants / canonicalization
- `CODE_ALIAS`: `{CPP->StPO, CP->StGB, CC->ZGB, CO->OR, LTF->BGG, LACI->AVIG, LAA->UVG, LP->SchKG, LDIP->IPRG, Cst/Cst.->BV, STPO->StPO, OBG->OR}`
- `LEGAL_AREA_DEFAULT_CODE` maps legal-area phrases (criminal procedure, civil law, obligations, civil procedure, constitutional, administrative, social insurance, tax law) -> default code for code-less anchor fallback.
- Regex: `ART_RE = r"art\.?\s*(\d+[a-z]?)"`, `CODE_RE = r"\b([A-Z][A-Za-z]{1,8}\.?)\b"`, `CASE_BGE_RE = r"BGE\s+(\d+)\s+([IVX]+)\s+(\d+)"`, `CASE_DOCKET_RE = r"\b(\d[A-Z]_\d+/\d{4})\b"`.

## Data

DATA_ROOT auto-detected from candidate list. On this run, paths resolved under Drive (`/content/drive/MyDrive/swiss_law/...`); local candidate `E:/swiss_citation_extraction` also listed.

Input files (verbatim PATHS keys):
- `val_csv`:    `DATA_ROOT / data / val.csv`
- `law_llm`:    first existing of:
  - `DATA_ROOT / data / checkpoints / law_llm_descriptors_0000000_all.jsonl`
  - `DATA_ROOT / law_json_llm_output / law_llm_descriptors_0000000_all.jsonl`
- `court_v5`:   first existing of:
  - `DATA_ROOT / artifacts_v2 / court_authority_cards_v5_unified.jsonl`
  - `DATA_ROOT / artifacts / court_authority_cards_v5_unified.jsonl`
- `emb_dir`:    `DATA_ROOT / artifacts / embeddings`
- `emb_manifest`: `DATA_ROOT / artifacts / embeddings / qwen3_8b_unified_manifest.parquet`
- Embedding chunks glob: `qwen3_8b_unified_chunk*.npy` (27 chunks, ~21 GB total)

Output directory:
- `out_dir`: `DATA_ROOT / research / anchor_funnel_val001_v4`
- Output of save: `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v4`

Output files written:
- `val_001_query_targets.json`
- `val_001_v6_per_channel.json`
- `val_001_v6_per_gold_trace.tsv`
- `val_001_v6_top1000_pool.txt`
- `val_001_v6_summary.md`

Query under test: `val_001`, gold count = 42.
- Query (first 240 chars): "May a court lawfully order a three-month extension of pre-trial detention under Art. 221 Abs. 1 lit. b StPO (risk of collusion) consistent with the principle of proportionality when the accused-detained after an alleged late-night assault a..."

## Pipeline

### Stage 1 - Environment & paths (Cells 1-3)
- Print Python/PyTorch versions and GPU stats.
- Mount Google Drive (`/content/drive`).
- Auto-detect DATA_ROOT from a candidate list; resolve PATHS with `first_existing()` fallbacks.
- Determine `EMB_AVAILABLE` from presence of chunk files.

### Stage 2 - Knob panel (Cell 4)
- Define `CONFIG` dict with budgets, RRF k, guarantee channels, co-citation/concept/BM25/vector/QE hyperparameters, BM25 lexicon expansion knobs (v6), noise paragraph roles, and normalization flags.

### Stage 3 - Load val_001 query + gold (Cell 5)
- Read `val.csv` with `csv.field_size_limit(2**31 - 1)`; locate row `query_id == "val_001"`; split `gold_citations` on `;`.

### Stage 4 - One-pass index build (Cell 6)
Single streaming pass over `law_llm` and `court_v5` JSONL builds:
- `cit_to_doc_ids`: citation string -> list of doc_ids
- `doc_meta`: per-doc {citation, family, court_base, paragraph_role, is_notification_paragraph}
- `idx_law_direct`: canonical statute key -> {law doc_ids}
- `idx_court_statute`: canonical statute key -> {court doc_ids}, anchors canonicalized via `canonicalize_row_anchors()` (two-pass with `LEGAL_AREA_DEFAULT_CODE` fallback for code-less anchors - Patch B kept)
- `idx_case_anchor`: canonical case key (BGE / docket) -> {doc_ids}
- `idx_court_base`: court_base -> {doc_ids} (used for sibling expansion)
- `idx_concept_en`: lowercase concept token -> {doc_ids} (from `llm_enrichment.concepts_en`, `rag_enrichment.concepts_en`, and `terms_de_to_en.en`)
- `idx_term_orig`: normalized term -> {doc_ids} (from `terms_de_to_en.de` for law; `rag_enrichment.terms_original` for court)
- `legal_area_per_doc`: doc_id -> legal_area_static (court) or `"law"`
- `search_text`: doc_id -> short text blob for BM25 (max 2000 chars); law uses english_summary + legal_rule + legal_question + applicability_conditions + concepts_en + terms; court uses text_excerpt_original + concepts_en + terms_original + micro_topic + topic + subtopic + statute_anchors
- `co_citation_pairs`: Counter of unordered canonical pairs co-occurring in same court row
- `tlf` (v6 NEW): token -> Counter(law_code -> distinct-law-row count); built from law-row text tokens (len >= 3) tagged with the law row's canonical code
- `token_doc_count` (v6 NEW): token -> #law rows containing it (for approximate IDF)
- `doc_statute_anchors` (v6 NEW): court doc_id -> set of canonical statutes it cites (used by back-propagation channel)

### Stage 5 - Map gold to doc_ids (Cell 7)
- Lookup each gold citation in `cit_to_doc_ids` to build `gold_doc_set`; record mapping coverage.

### Stage 6 - Per-area bedrock (Cell 8)
- Construct `canon_area_bases[canon][legal_area]` = set of `court_base` values that cite `canon` within that area; collapse to `per_area_canon_count[area][canon] = #distinct court_base`. Replaces v3 global bedrock.

### Stage 7 - Co-citation expansion (Cell 9)
- Aggregate `co_citation_pairs` into bidirectional `co_neighbours[canon] -> [(neighbour, co_count), ...]`.
- v5 FIX 3 filter: drop neighbours with `canon_total_count[nb] > co_citation_max_neighbour_count (5000)` to remove universally-cited articles. Keep top `co_citation_top_k_per_target (8)`.

### Stage 8 - BM25 (Cell 10)
- Build in-memory SQLite FTS5 (`unicode61 remove_diacritics 2`), insert all docs in 50,000-row batches; free `search_text` afterwards.
- `bm25_tokens()` lowercases, splits on `[^\w\d]+`, drops <3-char tokens, strips FTS5 bad chars `["()*+\-^]`.
- v6 `enhance_query_tokens()`: for each query token, look up `tlf[tok]` (Counter of codes); score each code by `(cnt/sum_codes) * approx_idf(tok)`; append top `enhance_top_k_codes=5` codes, each repeated `enhance_repeat_count=5` times.
- `bm25_search()`: builds `" OR "`-joined quoted-term query, returns FTS5 `bm25()` top-k (sign-flipped because FTS5 score is negative-better).

### Stage 9 - Vector channel setup (Cell 11)
- Load `qwen3_8b_unified_manifest.parquet`, map manifest row -> internal doc_id by matching `(family, citation)` against `cit_to_doc_ids`.
- Load 27 `.npy` chunks, concatenate, move to GPU as fp16 tensor `E_GPU` of shape `(N, 4096)`.

### Stage 10 - Embedding model for query encoding (Cell 12)
- `_ensure_emb_model()` lazy-loads SentenceTransformer with bf16 + sdpa + left-padding + `max_seq_length=4096`.
- `encode_query()` prepends canonical Qwen3 instruct prefix, encodes with `normalize_embeddings=True`.
- `vector_search()` normalizes query, runs `E_GPU @ q`, `topk(k)`, maps manifest row -> doc_id.

### Stage 11 - Query expansion via Qwen3-32B (Cell 13)
- Prompt schema demands JSON with `statute_targets`, `case_targets`, `concept_targets_en`, `term_targets_de`, `term_targets_fr`, `legal_area_keywords` and rules including "list neighbouring articles in the same procedural cluster" and "include Art. 100 BGG and Art. 42 BGG for BGer appeals".
- Load Qwen3-32B (bf16, `device_map=auto`), apply chat template (`enable_thinking=False`), `generate(do_sample=False, temperature=0.0, max_new_tokens=1024)`, extract JSON via brace slicing.
- Save targets to `val_001_query_targets.json`. Free Qwen3-32B + tokenizer; `torch.cuda.empty_cache()`.

### Stage 12 - Encode query (raw + enriched) (Cell 14)
- Enriched query = `QUERY + "\n\nKeywords: " + " ; ".join(term_targets_de + term_targets_fr + concept_targets_en + legal_area_keywords)`.
- Encode both with `encode_query()`.

### Stage 13 - Channel functions (Cell 15)
Defined functions: `channel_law_direct`, `channel_court_statute`, `channel_sibling` (expand by `court_base` membership), `expand_concepts_strict` (substring match against corpus concept keys, top-K by length distance), `channel_concept`, `channel_term`, `channel_per_area_bedrock` (v5 FIX 4: filter to canonicals whose code is among LLM-named codes), `channel_statute_backprop` (v6 NEW: for each caught court row, pull law rows for its `statute_anchors`; score = number of distinct caught court rows citing the canon), `channel_co_citation`, `channel_vector`.

### Stage 14 - Run channels + diagnostics (Cell 16)
1. Build `llm_statute_canons` from `statute_targets`; expand via `co_neighbours` to `co_expanded_canons`.
2. Extract `statute_target_codes` (e.g., `{BGG, StPO}`).
3. `expand_concepts_strict(concept_targets_en + legal_area_keywords, ..., 6)`.
4. Run channels: `law_direct_match` (full expanded set), `court_statute` (LLM-only), `concept`, `term`, `per_area_bedrock`, `co_citation`, `bm25` (with enhance), `vector_raw`, `vector_enriched`.
5. v6 BUG FIX 5: sibling seed = court hits from ALL channels (not just statute).
6. v6 NEW: `statute_backprop` seed = sibling seed | sibling channel's court hits.
7. Print per-channel size, gold-in-channel, and union upper bound.

### Stage 15 - RRF fusion + negative gate (Cell 17)
- `rrf_fuse(channels, k=60)`: standard `1/(k+rank+1)` summation over all channels.
- `apply_neg_gate()`: drop docs with `is_notification_paragraph=True` or `paragraph_role in noise_paragraph_roles`.
- Guarantee channels (`law_direct_match`, `per_area_bedrock`, `statute_backprop`) prepended in order (deduped + gated) before RRF tail; final list capped at `topk_final=1000`.
- Diagnostic R@K curve over the full ranked-gated pool.

### Stage 16 - Final pool (Cell 18)
- `final_topk = final_topk_rrf` (reranker dropped). Print R@K table at K in {50, 100, 200, 300, 500, 750, 1000} and the pass-criterion comparison.

### Stage 17 - Per-gold trace (Cell 19)
- For each gold citation: find best rank among its doc_ids in `final_topk`; report channels that surfaced it; sort by rank; print `MISS` for unranked.

### Stage 18 - Save artifacts (Cell 20)
- Write `val_001_v6_per_channel.json`, `val_001_v6_per_gold_trace.tsv`, `val_001_v6_top1000_pool.txt`, `val_001_v6_summary.md` into `out_dir`.

### Stage 19 - Cleanup (Cell 21)
- Close FTS5 connection; pop large globals; `gc.collect()`; `torch.cuda.empty_cache()`.

## Results

### Index build sizes
- Law: 173,033 rows indexed in 10.0s
- v6 token->code association: 91,173 tokens, avg codes per token = 11.7
- Court: 2,476,315 rows indexed in 155.3s
- Total docs: 2,649,348
- Unique citations: 2,158,211
- Index sizes: law_direct=49,288, court_statute=81,988, case=157,227, court_base=178,593, concept=292,946, term=363,331
- Co-citation pairs: 1,142,790

### Gold mapping
- Mapped gold: 42/42; total gold doc_ids: 42.

### Per-area bedrock
- 26 areas built in 5.4s.
- area='criminal law and criminal procedure': top 8 = `[('66 BGG', 17910), ('106 BGG', 10328), ('42 BGG', 10299), ('64 BGG', 8388), ('97 BGG', 8130), ('105 BGG', 7397), ('108 BGG', 7313), ('9 BV', 6436)]`
- area='criminal procedure and coercive measures': top 8 = `[('66 BGG', 5519), ('78 BGG', 4035), ('42 BGG', 3449), ('81 BGG', 3442), ('80 BGG', 2812), ('64 BGG', 2695), ('93 BGG', 2573), ('108 BGG', 2404)]`
- area='criminal law': top 8 = `[('278 BStP', 587), ('66 BGG', 521), ('121 BGG', 520), ('278 PPF', 433), ('42 BGG', 331), ('63 StGB', 320), ('269 PPF', 316), ('277b BStP', 311)]`
- area='criminal law and administrative criminal law': top 8 = `[('156 OG', 171), ('105 OG', 117), ('156 OJ', 112), ('104 OG', 107), ('16 SVG', 100), ('104 OJ', 94), ('17 SVG', 93), ('114 OJ', 87)]`

### Co-citation
- Indexed for 2,282 canonicals after frequency filter (max global count = 5000).
- Neighbours of '221 StPO': `31 BV (co=783, total=4339); 212 StPO (co=714, total=1337); 221 BV (co=646, total=670); 237 StPO (co=487, total=1467); 5 StPO (co=272, total=1955); 197 StPO (co=258, total=1555); 5 EMRK (co=246, total=3500); 5 CEDH (co=235, total=1233)`.

### FTS5
- Built: 2,649,348 rows in 86.4s.

### Vector
- Manifest rows: 2,652,248; mapping coverage: 2,649,348/2,652,248 (99.9%) in 5.5s.
- E_GPU shape=(2652248, 4096) dtype=torch.float16; VRAM=21.7 GB; load time 245.2s.

### Query expansion (Qwen3-32B)
- After expansion freed: CUDA mem = 21.7 GB.
- Parsed targets:
  - statute_targets (6): `Art. 221 StPO, Art. 222 StPO, Art. 227 StPO, Art. 212 StPO, Art. 100 BGG, Art. 42 BGG`
  - case_targets (5): `BGE 146 III 387, BGE 145 III 117, BGE 143 III 215, BGE 139 III 285, 1B_123/2024`
  - concept_targets_en (20): `preventive detention, collusion risk, proportionality, evidence preservation, witness tampering, ...`
  - term_targets_de (20): `Untersuchungshaft, Kollusionsgefahr, Verhältnismässigkeit, Beweissicherung, Zeugenbeeinflussung, ...`
  - term_targets_fr (15): `détention provisoire, danger de collusion, proportionnalité, préservation des preuves, manipulation de témoins, ...`
  - legal_area_keywords (5): `criminal procedure, detention law, evidence law, proportionality principle, investigative law`

### Query encoding
- Raw query encoded in 49.1s (includes model load); enriched in 0.1s.
- Enriched query length: 2365 chars; keywords appended: 60.

### Channel build
- LLM canons = 6; co-citation expansion -> 27 canons:
  `['100 BGG', '115 BGG', '197 StPO', '212 BGG', '212 StPO', '221 BV', '221 StPO', '222 StPO', '227 StPO', '229 StPO', '237 StPO', '31 BV', '393 StPO', '42 BGG', '45 BGG', '47 BGG', '48 BGG', '5 CEDH', '5 EMRK', '5 StPO', '50 BGG', '51 StGB', '54 BGG', '61 BGG', '73 BGG', '73 StHG', '84 BGG']`
- LLM target codes: `{BGG, StPO}`.
- Concepts: LLM=25 -> expanded=113.
- BM25 enhance() boosted codes: `['stpo', 'stgb', 'or', 'mstg', 'zgb']`.
- Sibling seed: 2,691 court doc_ids -> 500 expansions.
- Backprop seed: 3,191 court doc_ids -> 400 law expansions.

### Per-channel recall
| channel | size | gold | recall |
|---|---:|---:|---:|
| `law_direct_match` | 68 | 7 | 16.7% |
| `court_statute` | 600 | 2 | 4.8% |
| `co_citation` | 400 | 0 | 0.0% |
| `per_area_bedrock` | 150 | 4 | 9.5% |
| `statute_backprop` | 400 | 8 | 19.0% |
| `sibling_expansion` | 500 | 0 | 0.0% |
| `concept_en` | 600 | 9 | 21.4% |
| `term_orig` | 500 | 3 | 7.1% |
| `bm25` | 600 | 3 | 7.1% |
| `vector_raw` | 800 | 10 | 23.8% |
| `vector_enriched` | 800 | 8 | 19.0% |

- Union: 4,222 unique doc_ids; gold in union = 23/42 (UPPER BOUND on R@K).

### Fusion + gate
- Pre-gate fused: 4,222.
- Post-gate: 4,060.
- Guarantee pool: 452 (channels: `law_direct_match, per_area_bedrock, statute_backprop`).
- Final RRF top-1000 size: 1,000.

### Diagnostic R@K over ranked-gated pool
| K | gold/42 | recall |
|---:|:---:|---:|
| 50 | 7 | 16.7% |
| 100 | 7 | 16.7% |
| 200 | 7 | 16.7% |
| 300 | 7 | 16.7% |
| 500 | 10 | 23.8% |
| 750 | 11 | 26.2% |
| 1000 | 12 | 28.6% |
| 1500 | 12 | 28.6% |
| 2000 | 14 | 33.3% |
| 3000 | 14 | 33.3% |
| 3512 | 16 | 38.1% |

### Final R@K (RRF-only top-K, with guarantee pool prepended)
| K | gold/42 | recall |
|---:|:---:|---:|
| 50 | 7 | 16.7% |
| 100 | 7 | 16.7% |
| 200 | 7 | 16.7% |
| 300 | 7 | 16.7% |
| 500 | 8 | 19.0% |
| 750 | 13 | 31.0% |
| 1000 | 15 | 35.7% |

### Pass criterion
- PASS CRITERION: R@1000 >= 0.60 (>= 26/42).
- OBSERVED: R@1000 = 0.357 (15/42). FAIL.

### Per-gold trace (caught)
| rank | citation | channels |
|---:|---|---|
| 5 | Art. 227 Abs. 1 StPO | bm25,law_direct_match,statute_backprop |
| 8 | Art. 222 StPO | bm25,law_direct_match,statute_backprop |
| 9 | Art. 221 Abs. 2 StPO | law_direct_match,per_area_bedrock,statute_backprop |
| 11 | Art. 221 Abs. 1 StPO | bm25,law_direct_match,per_area_bedrock,statute_backprop |
| 16 | Art. 212 Abs. 3 StPO | concept_en,law_direct_match,per_area_bedrock,statute_backprop |
| 23 | Art. 100 Abs. 1 BGG | law_direct_match,per_area_bedrock,statute_backprop |
| 35 | Art. 393 Abs. 1 StPO | law_direct_match,statute_backprop |
| 446 | Art. 396 Abs. 1 StPO | statute_backprop |
| 503 | BGE 137 IV 122 E. 4.2 | concept_en,term_orig,vector_enriched,vector_raw |
| 517 | 1B_357/2022 E. 3.1 | concept_en,court_statute,vector_enriched,vector_raw |
| 567 | BGE 133 I 270 E. 3.4.2 | concept_en |
| 636 | BGE 139 IV 270 E. 3.1 | concept_en,court_statute |
| 640 | BGE 132 I 21 E. 3.2.1 | concept_en |
| 766 | 1B_28/2022 E. 4.1 | concept_en,vector_enriched,vector_raw |
| 798 | 1B_90/2021 E. 2.1 | term_orig,vector_enriched,vector_raw |

### Missed gold (27/42)
Citations and (if any) channels that caught them anywhere in the union but ranked outside top-1000:
- `Art. 140 Abs. 1 StGB` — (none)
- `Art. 382 Abs. 1 StPO` — (none)
- `Art. 385 Abs. 1 StPO` — (none)
- `Art. 390 Abs. 2 StPO` — (none)
- `Art. 422 Abs. 1 StPO` — (none)
- `Art. 422 Abs. 2 StPO` — (none)
- `Art. 428 Abs. 1 StPO` — (none)
- `Art. 135 Abs. 4 StPO` — (none)
- `Art. 135 Abs. 3 StPO` — (none)
- `Art. 37 Abs. 1 StBOG` — (none)
- `Art. 39 Abs. 1 StBOG` — (none)
- `BGE 137 IV 122 E. 6.2` — (none)
- `BGE 137 IV 122 E. 6.4` — (none)
- `BGE 132 I 21 E. 3.2` — concept_en
- `1B_210/2023 E. 4.1` — (none)
- `BGE 132 I 21 E. 3.2.2` — vector_enriched,vector_raw
- `1B_536/2018 E. 5.1` — (none)
- `BGE 133 I 168 E. 4.1` — (none)
- `BGE 143 IV 168 E. 5.1` — (none)
- `BGE 137 IV 122 E. 4.1` — concept_en
- `1B_90/2021 E. 2.4` — (none)
- `7B_496/2025 E. 3.2` — (none)
- `7B_231/2025 E. 4.1` — vector_raw
- `7B_69/2024 E. 3.3.2` — term_orig,vector_enriched,vector_raw
- `7B_301/2024 E. 2.4` — vector_enriched,vector_raw
- `7B_12/2025 E. 2.2` — vector_raw
- `1B_15/2023 E. 3.1` — vector_enriched,vector_raw

### Persisted artifacts
- "Saved artifacts to: `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v4`"
- Cleanup output: "cleaned up."

## Summary

v6 of the anchor-funnel pipeline targeted R@1000 >= 0.60 on val_001 (42 gold) by (a) dropping the Qwen3-Reranker that had cost 48 min and regressed v5 from 0.357 to 0.333, (b) expanding sibling-channel seeding to court hits from ALL channels, (c) adding a corpus-derived BM25 lexicon-expansion `enhance()` that boosts BM25 with the most-associated legal codes per token, and (d) adding a statute back-propagation channel that, for every caught court row, pulls in the law rows for the article anchors it cites. The final R@1000 was 0.357 (15/42), still well below the 0.60 target and exactly matching the v5 RRF-only baseline. The new `statute_backprop` channel was the strongest single contributor at 19.0% (surfacing Art. 100 BGG, Art. 393 StPO, etc.), and `vector_raw` led at 23.8%, but the union ceiling of 23/42 = 0.548 caps any fusion strategy below pass. The hardest misses were docket cases and `Eränwägung`-level BGE references (7B/1B/StBOG citations) plus the cost/standing StPO cluster (Art. 422, 428, 135, 382-390 StPO) — none of which appeared in any channel for several entries. Lessons: the procedural/cost StPO cluster is not reachable from this query alone via co-citation or back-propagation when those law rows aren't reached through caught court rows; further recall must come from broader court-side retrieval (vector or BM25 over richer text) or from cycle-2 expansion using the v6 results as seed.
