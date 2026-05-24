# anchor_funnel_v7_4_val001.ipynb

**Path:** `notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v7_4_val001.ipynb`

## Configuration

### Models
- **Query expansion LLM:** `Qwen/Qwen3-32B` (bf16, ~65 GB VRAM, `device_map="auto"`, `enable_thinking=False`, `max_new_tokens=1024`).
- **Embedding model (query-side):** `Qwen/Qwen3-Embedding-8B` via SentenceTransformer (`prompt_name="query"`, normalized).
- **Corpus embeddings:** pre-encoded 27 fp16 chunks loaded into single GPU tensor `E_GPU` shape `(2,652,248, 4096)` fp16, ~20.2 GB VRAM.

### Libraries
- Python 3.12.13, PyTorch 2.10.0+cu128, `transformers` (AutoTokenizer/AutoModelForCausalLM), `sentence_transformers`, `sqlite3` (FTS5 in-memory), `pandas`, `numpy`, stdlib `re`, `json`, `collections`.

### Hardware
- 1× NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB VRAM (Colab "G4" gpuType, machine_shape "hm").

### Pass criterion
- `R@1000 >= 0.60` for val_001 (>= 26/42 gold). Stretch: `>= 0.90` / 38+.

### Knob panel (`CONFIG`) — printed value
The notebook source declares v7.5 budgets in `CONFIG = {...}` but the JSON dump printed by cell 8 reflects an earlier in-memory state (v7.4 values). Both are recorded below:

| Knob | v7.5 source value | v7.4 dumped value |
|---|---|---|
| `topk_final` | 1000 | 1000 |
| `budget_law_direct` | None | None |
| `budget_court_statute` | 8000 | 600 |
| `budget_concept` | 3000 | 600 |
| `budget_term` | 2500 | 500 |
| `budget_per_area` | 1500 | 150 |
| `budget_co_citation` | 2500 | 1000 |
| `budget_bm25` | 2000 | 600 |
| `budget_vector` / `budget_vector_enriched` | 2000 / 2000 | 800 / 800 |
| `budget_backprop` | 2000 | 800 |
| `budget_sibling` | 5000 | 2000 |
| `budget_graph_forward` | 5000 | 1500 |
| `budget_graph_reverse` | 3000 | 1000 |
| `enable_graph_2hop` | False | False |
| `budget_graph_2hop` | 1500 | 1000 |
| `rrf_k` | 60 | 60 |
| `guarantee_channels` | 7 channels (LDM, PAB, SBP, concept_en, graph_forward, sibling_expansion, term_orig) | 3 channels (LDM, PAB, SBP) |
| `guarantee_per_channel` | 130 | 1000 |
| `per_area_top_n` | 1000 | 100 |
| `co_citation_top_k_per_target` | 50 | 20 |
| `co_citation_min_co_count` | 50 | 50 |
| `co_citation_max_neighbour_count` | 50000 | 15000 |
| `concept_substring_top_k` | 6 | 6 |
| `bm25_max_query_terms` | 60 | 60 |
| `bm25_min_token_len` | 3 | 3 |
| `vector_topk` | 800 | 800 |
| `qwen_max_new_tokens` | 1024 | 1024 |
| `enhance_top_k_codes` | 5 | 5 |
| `enhance_repeat_count` | 5 | 5 |
| `enhance_min_idf` | 1.0 | 1.0 |
| `code_family_top_k` | 8 | — |
| `noise_paragraph_roles` | `{notification, header, empty, metadata}` | same |
| `lowercase_concepts` / `lowercase_terms` | True / True | same |

### v7.5 channel weights (RRF)
`statute_backprop=2.5`, `graph_forward=2.0`, `concept_en=1.8`, `court_statute=1.5`, `per_area_bedrock=1.5`, `vector_raw=1.0`, `vector_enriched=1.0`, `term_orig=1.2`, `law_direct_match=1.2`, `sibling_expansion=1.0`, `bm25=0.8`, `co_citation=0.7`, `graph_reverse=0.5`, `graph_2hop=0.0`.

### Constants
- `CODE_ALIAS` (CPP→StPO, CP→StGB, CC→ZGB, CO→OR, LTF→BGG, LACI→AVIG, LAA→UVG, LP→SchKG, LDIP→IPRG, Cst→BV, STPO→StPO, OBG→OR).
- `LEGAL_AREA_DEFAULT_CODE` (criminal procedure → StPO, civil law → ZGB, obligations → OR, constitutional → BV, administrative → VwVG, social insurance → ATSG, tax law → DBG, …).
- `_TERM_LEMMA_SUFFIXES = ("en","es","em","er","e","n","s")` with 4-char stem floor.
- `_ROLE_W` paragraph-role weights for `channel_court_statute`: legal_standard/reasoning/application/holding=1.5; facts/procedural_history/citation/neutral_default=1.0; costs/disposition/notification=0.6; neutral=0.4; null/empty=0.4.
- `_SUBSTANTIVE_ROLES = {reasoning, legal_standard, application, holding}` for v7.4 graph/sibling role boost (×1.5).
- `SUBSTANTIVE_ROLES` for negative-gate override: `{facts, reasoning, legal_standard, application, holding, citation, procedural_history}`.

## Data

### Auto-detected root
- `DATA_ROOT = /content/drive/MyDrive/swiss_law` (selected from candidate list including `/content/drive/MyDrive/swiss_citation_extraction`, `/content/swiss_citation_extraction`, `E:/swiss_citation_extraction`, cwd).

### PATHS dict (all flagged OK)
- `val_csv = /content/drive/MyDrive/swiss_law/data/val.csv`
- `law_llm = /content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl`
- `court_v5 = /content/drive/MyDrive/swiss_law/artifacts_v2/court_authority_cards_v5_unified.jsonl`
- `emb_dir = /content/drive/MyDrive/swiss_law/artifacts/embeddings` (27 `qwen3_8b_unified_chunk*.npy` files, ~21 GB total)
- `emb_manifest = /content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_manifest.parquet` (cols: `doc_id, family, citation, row_index`, 2,652,248 rows)
- `graph_db = /content/drive/MyDrive/swiss_law/data_insights/citation_graph_extracted.sqlite` (~2.4 GB, 4-pass extraction; reads `edges` table with `dataset='court_considerations'`)
- `out_dir = /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7`

### Query
- `val.csv` has 10 queries; selected `query_id == "val_001"` with 42 gold citations.
- Query (first 240 chars): "May a court lawfully order a three‑month extension of pre‑trial detention under Art. 221 Abs. 1 lit. b StPO (risk of collusion) consistent with the principle of proportionality when the accused—detained after an alleged late‑night assault a..."
- First 6 gold: `Art. 221 Abs. 1 StPO`, `Art. 140 Abs. 1 StGB`, `Art. 396 Abs. 1 StPO`, `Art. 222 StPO`, `Art. 393 Abs. 1 StPO`, `Art. 382 Abs. 1 StPO`.

### Output artifacts written by the notebook
- `final_topk.json`, `gold_in_top.json`, `gold_missed_diagnosis.json`, `targets.json`, `config.json`, `summary.json` → all in `out_dir`.

## Pipeline

### Phase 1 — Setup (cells 1–10)
1.1 Print Python/PyTorch/GPU info. 1.2 Mount Google Drive. 1.3 Auto-detect `DATA_ROOT`, build `PATHS` dict, check existence flags, set `EMB_AVAILABLE`, `GRAPH_AVAILABLE`. 1.4 Define `CONFIG` knob panel (budgets, RRF k, BM25 limits, vector top-k, model ids, channel weights, code-family expansion, per-area bedrock, co-citation, enhance(), negative gate). 1.5 Load `val.csv`, select `val_001`, parse 42 gold citations.

### Phase 2 — Corpus indexing (cells 11–14)
2.1–2.4 Single streaming pass over `law_llm_descriptors_*.jsonl` and `court_authority_cards_v5_unified.jsonl`. Builds in one shot:
- Canonicalizers: `statute_anchor_canonical` (regex on "Art. N CODE" plus `CODE_ALIAS`), `case_anchor_canonical` (`BGE V D P` + `\d[A-Z]_\d+/\d{4}` docket).
- German lemmatizer `term_lemma` (iterative suffix-drop with 4-char floor).
- Indices: `cit_to_doc_ids`, `doc_meta`, `idx_law_direct`, `idx_court_statute`, `idx_case_anchor`, `idx_court_base`, `idx_concept_en`, `idx_term_orig`, `idx_term_lemma`, `term_orig_keys`, `legal_area_per_doc`, `search_text`, `co_citation_pairs`, `tlf` (token→legal-code Counter), `token_doc_count`, `doc_statute_anchors`, `doc_language`.
- `doc_id` prefixes: `law:<i>`, `court:<i>`.
- 2.5 Sanity-check: every gold citation maps to a doc_id (Class A coverage).

### Phase 3 — Citation graph (cells 15–16)
Loads `citation_graph_extracted.sqlite` (4-pass build: intra-judgment backrefs, date-stripped aliases, E.-range expansion, case-level fan-out). Converts citation→citation edges to doc_id-keyed `idx_graph_out` / `idx_graph_in` via `_cit_to_did` map; drops self-edges and synthetic targets. Builds `idx_judgment_importance[court_base] = sum(len(idx_graph_in[d]) for d in idx_court_base[cb])` to break score=1 ties in graph/sibling channels (used by `_judgment_factor = sqrt(1 + log(1 + imp))`).

### Phase 4 — Per-area bedrock + co-citation (cells 17–20)
4.1 `per_area_canon_count[area] = Counter[canonical]` derived from `legal_area_per_doc` × `doc_statute_anchors`. v7.5 adds `code_pair_count` (symmetric, weighted by co-occurrence count from `co_citation_pairs`) for corpus-derived code-family expansion (no hardcoded statute table). 4.2 `co_neighbours[canon]` = top-K co-cited canonicals filtered by `co_citation_max_neighbour_count` (drop globally over-cited statutes).

### Phase 5 — BM25 / FTS5 (cells 21–22)
5.A Legacy single-index FTS5 (unicode61 remove_diacritics 2) over all `search_text`. 5.B Per-language FTS5 indices (de/fr/it/en) routed by `doc_language[did]`. 5.C `_enhance_codes(text)`: corpus-derived BM25 lexicon expansion using `tlf`+IDF; top-K codes repeated 5× and appended to query. 5.D Legacy `bm25_search(query, k)` returns `(did, -score)`. 5.E NEW `bm25_search_multilang(QUERY, targets, k)`: builds language-specific enriched query per index, runs FTS5, normalizes scores by `log(corpus_size + e)`, merges by max-score across languages, returns top-k.

### Phase 6 — Vector channel (cells 23–24)
Loads manifest parquet, maps `row_index ↔ doc_id` by `(citation, family)` lookup against `cit_to_doc_ids` + `doc_meta`. Concatenates 27 fp16 chunks via `torch.cat` to single GPU tensor `E_GPU`. `vector_search(q_emb, k)`: normalize → `E_GPU @ q` → `topk`. Cell 25–26 declare `EMB_MODEL` placeholder + `encode_query(text)` using `prompt_name="query"`.

### Phase 7 — Query expansion (cells 27–30)
7.1–7.5 Load Qwen3-32B (bf16) with `device_map="auto"`, build chat-templated prompt with `QEXP_PROMPT_SYSTEM`+`QEXP_PROMPT_USER` (asks for JSON with `statute_targets`, `case_targets`, `concept_targets_en`, `term_targets_de`, `term_targets_fr`, `legal_area_keywords`). Generate with `do_sample=False`, `max_new_tokens=1024`, `enable_thinking=False`. Decode, regex-extract `{…}`, `json.loads`. Free model + `torch.cuda.empty_cache()`. 7.6 Load `Qwen/Qwen3-Embedding-8B`, encode raw query and enriched query (raw + first 60 of `term_targets_de` + `term_targets_fr` + `concept_targets_en`).

### Phase 8 — Channels (cells 31–34)
Channel function definitions (cell 32):
- `channel_law_direct(canon_set, idx, budget)` — Counter of doc_ids by canon match, top budget.
- `channel_court_statute(...)` — specificity-weighted (1/log(2+global_count)) × multi-match bonus × `_ROLE_W[paragraph_role]`; deterministic sort.
- `channel_concept(expanded_weighted, idx, budget)` — token-overlap matcher (stopword-filtered, weighted by shared meaningful tokens).
- `channel_term(targets, idx_term_orig, idx_term_lemma, budget, corpus_keys)` — German lemmatizer + substring overlap.
- `channel_per_area_bedrock(legal_area_keywords, statute_target_codes, per_area_canon_count, idx_law_direct, budget)` — top_n=1000 per area filtered by allowed codes.
- `channel_co_citation(targets, co_neighbours, idx_law_direct, idx_court_statute, budget)` — neighbour cluster expansion.
- `channel_sibling(seed, idx_court_base, idx_judgment_importance, doc_meta, budget)` — sibling Es of caught judgments; score = `seed_count × _judgment_factor × _role_boost`.
- `channel_graph_forward(seed, idx_graph_out, idx_judgment_importance, doc_meta, budget)` — edge-count × target judgment importance × role boost.
- `channel_graph_reverse(seed, idx_graph_in, idx_judgment_importance, doc_meta, budget, landmark_imp_threshold=5)` — landmark-gated incoming-edge expansion.
- `channel_graph_2hop` (off by default).
- `channel_vector(q_emb, k)` — wraps `vector_search`.

Cell 34 runs the orchestration: build `llm_statute_canons` + `co_expanded_canons` + `statute_target_codes` (code-family-expanded via `code_pair_count`); expand concepts via `expand_concepts_weighted` (top_k=25 strict-substring against corpus); run topical channels; assemble seed pool from court hits of all topical channels; run sibling + graph_forward + graph_reverse + statute_backprop with seed; assemble `CHANNELS` list of `(name, hits)`.

### Phase 9 — RRF fusion + gating (cells 35–36)
- `rrf_fuse(CHANNELS, rrf_k=60, weights=channel_weights)` — weighted RRF: `score(did) += w / (k + rank + 1)`.
- `apply_neg_gate(doc_ids, doc_meta, noise_paragraph_roles)` — substantive paragraph_role overrides `is_notification_paragraph` flag.
- `round_robin_guarantee(channels_by_name, guarantee_channels, per_channel_cap, total_cap)` — round-robin one-per-channel-per-round until cap; deduped.
- Final pool: guarantee (gated) + RRF-ranked-gated tail truncated to `topk_final=1000`.

### Phase 10 — Diagnosis (cells 37–42)
Per-channel rank lookups (`channel_rank`, `channel_score`), token-overlap helper, gold cosine-similarity dictionary (`gold_vec_cos_raw`, `gold_vec_cos_enr`), graph path helpers, sibling diagnosis. For every gold not in top-1000: dump source row metadata + search_text + per-channel HIT/MISS reason + RRF rank → classify root cause (RRF_RANK_TOO_LOW / NO_CHANNEL_HIT / GATED_OUT) + fix suggestion. Aggregate failure-mode summary + per-channel "caught_missed_gold" rate.

### Phase 11 — Save artifacts + cleanup (cells 43–46)
Write 6 JSON files; delete `E_GPU`, `EMB_MODEL`; `gc.collect()`; `torch.cuda.empty_cache()`.

## Results

### Environment
- Python 3.12.13, PyTorch 2.10.0+cu128.
- GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB.
- Drive: `Mounted at /content/drive`.

### Path resolution
- All 6 paths flagged `OK`; `EMB_AVAILABLE = True`, `GRAPH_AVAILABLE = True`.

### val.csv
- 10 queries total; val_001 = 42 gold citations.

### Corpus indexing
- Law: 173,033 rows indexed in 11.2 s.
- Token→code association: 91,173 tokens, avg codes/token = 11.7.
- Court: 2,476,315 rows indexed in 166.2 s (progress prints at 500K/1M/1.5M/2M rows).
- Total docs: 2,649,348. Unique citations: 2,158,211.
- Index sizes: `law_direct=49,288`, `court_statute=81,988`, `case=157,227`, `court_base=178,593`, `concept=292,946`, `term=363,331`, `term_lemma=521,146`, `term_keys=363,331`.
- Co-citation pairs: 1,142,790.
- Gold→doc_id: 42/42 mapped; 42 gold doc_ids.

### Citation graph
- citation→doc_id map: 2,158,211 entries.
- 20,494,436 edges loaded, 3,155,263 skipped (cit not in corpus).
- Out-degree avg = 27.2, in-degree avg = 14.0; load time 96.2 s.
- Judgment importance: 178,593 judgments, median=7, p75=44, max=204,686; built in 1.0 s.

### Per-area bedrock
- 26 legal areas built in 2.0 s. Examples:
  - `constitutional and public law` top-8: `66 BGG(13,212)`, `29 BV(12,189)`, `89 BGG(11,479)`, `82 BGG(11,464)`, `42 BGG(10,836)`, `9 BV(9,650)`, `106 BGG(9,625)`, `68 BGG(8,649)`.
  - `administrative, tax, migration, and regulatory law` top-8: `42 BGG(20,301)`, `106 BGG(19,460)`, `66 BGG(17,471)`, `105 BGG(16,033)`, `95 BGG(15,722)`, `83 BGG(15,231)`, `68 BGG(14,674)`, `89 BGG(12,187)`.
  - `civil law` top-8: `63 OJ(3,189)`, `8 ZGB(2,740)`, `55 OJ(2,518)`, `64 OJ(2,235)`, `55 OG(2,085)`, `63 OG(1,906)`, `159 OG(1,876)`, `9 BV(1,875)`.
  - `criminal law and criminal procedure` top-8: `66 BGG(28,149)`, `42 BGG(20,746)`, `106 BGG(19,537)`, `64 BGG(13,843)`, `108 BGG(12,101)`, `97 BGG(12,075)`, `105 BGG(11,759)`, `81 BGG(11,102)`.

### Co-citation
- Neighbours indexed for 2,282 canonicals (max global count = 15,000).
- `221 StPO` neighbours after frequency filter: `36 BV (co=915, total=7,216)`, `31 BV (783, 4,339)`, `10 BV (731, 5,241)`, `212 StPO (714, 1,337)`, `221 BV (646, 670)`, `237 StPO (487, 1,467)`, `5 BV (393, 9,685)`, `5 StPO (272, 1,955)`.

### BM25 / FTS5
- Legacy single FTS5: 2,649,348 rows in 83.7 s.
- Per-language FTS5 built in 70.0 s: de=1,593,249, fr=793,023, it=125,583, en=137,493 docs.

### Vector setup
- Manifest: 2,652,248 rows; manifest→doc_id mapping 2,649,348/2,652,248 (99.9%) in 40.1 s.
- `E_GPU` shape `(2,652,248, 4096)` float16, VRAM used 20.2 GB, free 74.7 GB; load 460.8 s.

### Query expansion (Qwen3-32B)
- Model load time: 201.6 s.
- VRAM after free: 20.2 GB (E_GPU only).
- Raw response (first 400 chars): `{ "statute_targets": ["Art. 221 StPO", "Art. 222 StPO", "Art. 227 StPO", "Art. 100 BGG", "Art. 101 BGG"], "case_targets": ["BGE 147 II 123", "BGE 145 II 345", "1B_123/2024"], "concept_targets_en": ["preventive detention", "collusion risk", "proportionality", "investigative necessity", "witness tampering"], "term_targets_de": ["Untersuchungshaft", "Kollusionsgefahr", "Verhältnismässigkeit", …`
- Parsed targets (5 items each):
  - `statute_targets`: Art. 221 StPO, Art. 222 StPO, Art. 227 StPO, Art. 100 BGG, Art. 101 BGG.
  - `case_targets`: BGE 147 II 123, BGE 145 II 345, 1B_123/2024.
  - `concept_targets_en`: preventive detention, collusion risk, proportionality, investigative necessity, witness tampering.
  - `term_targets_de`: Untersuchungshaft, Kollusionsgefahr, Verhältnismässigkeit, Ermittlungsbedürftigkeit, Zeugenbeeinflussung.
  - `term_targets_fr`: détention provisoire, danger de collusion, proportionnalité, nécessité d'enquête, influence sur les témoins.
  - `legal_area_keywords`: criminal procedure, detention law, proportionality principle, evidence preservation, witness protection.
- Enriched query: 1367 chars, 15 keywords appended.

### Per-channel diagnostics (val_001)
- Statute canons: LLM=5, +co-citation=68.
- LLM target codes: `{StPO, BGG}`. (Code-family expansion declared but values not printed in this run; cell uses v7.4 codes only.)
- Concepts: LLM=10 → expanded=246.
- Seed pool: 3,533 court doc_ids.
- Sibling expansion: 2,000. Graph forward: 1,500. Graph reverse: 1,000. Backprop seed: 7,802 → 800 law expansions.

```
channel                   size   gold_in_ch  recall
------------------------------------------------------------
law_direct_match           152            7   16.7%
court_statute              600            0    0.0%
co_citation               1000            0    0.0%
per_area_bedrock           150            4    9.5%
statute_backprop           800           10   23.8%
sibling_expansion         2000            6   14.3%
graph_forward             1500           11   26.2%
graph_reverse             1000            0    0.0%
graph_2hop                   0            0    0.0%
concept_en                 600            9   21.4%
term_orig                  500            4    9.5%
bm25                       600            3    7.1%
vector_raw                 800            5   11.9%
vector_enriched            800            8   19.0%

Union: 8,958 unique doc_ids
Gold in union: 30/42  (UPPER BOUND on R@K)
```

### Fusion + final pool
- Pre-gate fused: 8,958. Post-gate: 8,770. Guarantee pool: 848 (channels: `['law_direct_match', 'per_area_bedrock', 'statute_backprop']`). Final top-1000: 1,000.

```
R@K curve:
     K  gold/42  recall
------------------------------
    50      3/42    7.1%
   100      5/42   11.9%
   200      7/42   16.7%
   300      7/42   16.7%
   500      8/42   19.0%
   750     10/42   23.8%
  1000     17/42   40.5%
  1500     21/42   50.0%
  2000     26/42   61.9%
  3000     27/42   64.3%
  5000     30/42   71.4%

Final R@1000 = 0.405  (17/42)
```

### Diagnosis (cell 40)
- R@1000 = 17/42 = 40.5%. Missed gold: 25.
- Failure-mode summary (cell 42):
  - `NO_CHANNEL_HIT`: 12
  - `RRF_RANK_TOO_LOW`: 9
  - `GATED_OUT`: 4
- Top-10 missed examples include `1B_210/2023 E. 4.1` (RRF rank 2993, 1 channel), `1B_536/2018 E. 5.1` (rank 3799), `7B_496/2025 E. 3.2` (NO_CHANNEL_HIT), `BGE 133 I 168 E. 4.1` (rank 4839), `1B_90/2021 E. 2.4` (rank 1833), `7B_231/2025 E. 4.1` (NO_CHANNEL_HIT), `1B_28/2022 E. 4.1` (GATED_OUT, paragraph_role=facts), `BGE 132 I 21 E. 3.2.2` (GATED_OUT, facts), `BGE 143 IV 168 E. 5.1` (rank 1267), `7B_69/2024 E. 3.3.2` (rank 1176).
- Channel-level hit rate on missed gold: `concept_en=7`, `vector_enriched=6`, `term_orig=3`, `vector_raw=3`, `sibling_expansion=1`, `graph_forward=1`; all other channels caught 0 missed gold.
- Query token count (post-enhance): 122.

### Artifacts saved
- "Saved 6 artifacts to `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7`".
- Cleanup: "VRAM after cleanup: 0.01 GB". "cleaned up."

## Summary

The notebook is a v7.4 production-diagnostic build of the 14-channel "anchor funnel" retrieval pipeline against val_001 (Swiss DE/FR/IT legal corpus, 2.65 M docs, 42 gold citations for a pre-trial-detention query). It indexes the full corpus, loads a 20.5 M-edge citation graph, expands the query via Qwen3-32B → structured JSON targets, runs 14 retrieval channels (statute/concept/term/BM25/dense-vector/graph/sibling), fuses via weighted RRF with a 3-channel round-robin guarantee, and produces a per-gold root-cause diagnosis for every miss. Final result: **R@1000 = 0.405 (17/42)** — below the 0.60 pass bar despite a union upper-bound of 30/42 (71.4%); fusion left 13 gold on the table relative to the union. Diagnosis attributed misses to `NO_CHANNEL_HIT=12`, `RRF_RANK_TOO_LOW=9`, `GATED_OUT=4`, with `concept_en` and `vector_enriched` catching the most missed-but-recoverable gold. The v7.5 source values written into the CONFIG dict (much larger budgets, 7-channel guarantee, code-family expansion) were not the values that actually drove this run; the printed CONFIG dump shows v7.4 budgets, suggesting the run executed under the older state before the v7.5 widening was applied. Key lessons: case-level fan-out via the citation graph closed the orphan problem (graph_forward had highest single-channel recall at 26.2%), but the 3-channel guarantee (LDM/PAB/SBP — all low- or moderate-recall on this query) starved high-recall channels (graph_forward, concept_en, vector_enriched) of top-1000 slots, motivating the v7.5 7-channel guarantee + smaller per-channel cap design.
