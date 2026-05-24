# anchor_funnel_v7_val001.ipynb

**Path:** `notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v7_val001.ipynb`

## Configuration

### Runtime
- Python 3.12.13
- PyTorch 2.10.0+cu128
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB VRAM
- Kernel: `python3` (Colab; Drive mounted at `/content/drive`)

### Models
- Query expansion: `Qwen/Qwen3-32B` (bf16, `device_map="auto"`, ~65 GB VRAM)
- Embeddings: `Qwen/Qwen3-Embedding-8B` via `sentence-transformers` (~16 GB VRAM during encode)
- No reranker, no LLM verifier.

### Libraries
`sys`, `torch`, `numpy`, `pandas`, `sqlite3` (FTS5 + graph DB), `collections.{defaultdict, Counter}`, `re`, `json`, `time`, `gc`, `math`, `statistics`, `pathlib.Path`, `transformers.{AutoTokenizer, AutoModelForCausalLM}`, `sentence_transformers.SentenceTransformer`.

### CONFIG dict (full, verbatim)
```
topk_final = 1000
budget_law_direct = None
budget_court_statute = 3000
budget_concept = 600
budget_term = 500
budget_per_area = 150
budget_co_citation = 1000
budget_bm25 = 600
budget_vector = 800
budget_vector_enriched = 800
budget_backprop = 400
budget_sibling = 2000
budget_graph_forward = 1500
budget_graph_reverse = 1000
enable_graph_2hop = False
budget_graph_2hop = 1500
rrf_k = 60
guarantee_channels = [law_direct_match, per_area_bedrock, statute_backprop, graph_forward, sibling_expansion]
guarantee_per_channel = 400
channel_weights = {statute_backprop: 2.5, graph_forward: 2.0,
                   court_statute: 0.7, bm25: 0.7, term_orig: 0.7,
                   sibling_expansion: 0.7, graph_reverse: 0.5,
                   co_citation: 0.3, graph_2hop: 0.0}
per_area_top_n = 100
co_citation_top_k_per_target = 8
co_citation_min_co_count = 50
co_citation_max_neighbour_count = 5000
concept_substring_top_k = 6
bm25_max_query_terms = 60
bm25_min_token_len = 3
vector_emb_model = Qwen/Qwen3-Embedding-8B
vector_topk = 800
qwen_query_model = Qwen/Qwen3-32B
qwen_max_new_tokens = 1024
enhance_top_k_codes = 5
enhance_repeat_count = 5
enhance_min_idf = 1.0
noise_paragraph_roles = {"notification", "header", "empty", "metadata"}
lowercase_concepts = True
lowercase_terms = True
```

### Constants / regexes
- `CODE_ALIAS`: maps FR/IT/uppercase variants to Swiss German canonical codes (CPP→StPO, CP→StGB, CC→ZGB, CO→OR, LTF→BGG, LACI→AVIG, LAA→UVG, LP→SchKG, LDIP→IPRG, Cst→BV, STPO→StPO, OBG→OR).
- `LEGAL_AREA_DEFAULT_CODE`: 10 area-name → fallback code mappings (criminal procedure→StPO, civil law→ZGB, …).
- `ART_RE = r"art\.?\s*(\d+[a-z]?)"`, `CODE_RE = r"\b([A-Z][A-Za-z]{1,8}\.?)\b"`.
- `EXTRA_STATUTE_RE` (v7.2 P2 fix): multilingual sweep over `Art./Article/Articolo N CODE` with Abs/al/cpv/lit/Ziff/cifra subclauses.
- `CASE_BGE_RE = r"BGE\s+(\d+)\s+([IVX]+)\s+(\d+)"`, `CASE_DOCKET_RE = r"\b(\d[A-Z]_\d+/\d{4})\b"`.

### Pass criterion
- Stated target: R@1000 ≥ 0.60 on val_001 (≥ 26/42 gold). Stretch: ≥ 0.90 / 38+.
- Reported "v6 → v7" baseline: v6 R@1000 was 0.357 on val_001.

## Data

Auto-detected `DATA_ROOT = /content/drive/MyDrive/swiss_law` (candidate roots include `/content/drive/MyDrive/swiss_law`, `/content/drive/MyDrive/swiss_citation_extraction`, `/content/swiss_citation_extraction`, `E:/swiss_citation_extraction`, `Path.cwd()`).

Resolved paths (all `OK`):
- `val_csv` = `/content/drive/MyDrive/swiss_law/data/val.csv`
- `law_llm` = `/content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl`
- `court_v5` = `/content/drive/MyDrive/swiss_law/artifacts_v2/court_authority_cards_v5_unified.jsonl`
- `emb_dir` = `/content/drive/MyDrive/swiss_law/artifacts/embeddings`
- `emb_manifest` = `/content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_manifest.parquet`
- `graph_db` = `/content/drive/MyDrive/swiss_law/data_insights/citation_graph_extracted.sqlite`
- `out_dir` = `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7`

Embedding chunks: 27 `qwen3_8b_unified_chunk*.npy` files (fp16, ~21 GB total) → concatenated to `E_GPU` shape `(2,652,248, 4096)`.

Flags: `EMB_AVAILABLE = True`, `GRAPH_AVAILABLE = True`, `GRAPH_OK = True`, `VECTOR_OK = True`.

## Pipeline

### Phase 1 — Setup
- 1.1 Print Python/torch/GPU.
- 1.2 `from google.colab import drive; drive.mount('/content/drive')`.
- 1.3 Auto-detect `DATA_ROOT`, build `PATHS` dict, set `EMB_AVAILABLE` / `GRAPH_AVAILABLE`.
- 1.4 Define `CONFIG` (knob panel).
- 1.5 Read `val.csv`, build `ALL_QUERIES = [{query_id, query, gold}, …]` (10 queries). Set legacy `val001`, `QUERY`, `val_gold` (42 gold) for Phase 11 detailed trace.

### Phase 2 — Corpus indexes (single streaming pass)
Built in-memory:
- `cit_to_doc_ids`, `doc_meta`, `idx_law_direct[canon]`, `idx_court_statute[canon]`, `idx_case_anchor[canon]`, `idx_court_base[base]`, `idx_concept_en[token]`, `idx_term_orig[token]`, `legal_area_per_doc[did]`, `search_text[did]` (≤2000 chars), `co_citation_pairs` Counter, `tlf[token][code]`, `token_doc_count`, `doc_statute_anchors[did]`.
- Statute canonicalization via `statute_anchor_canonical()` + `canonicalize_row_anchors()` with `LEGAL_AREA_DEFAULT_CODE` fallback.
- v7.2 P2 fix: `extract_extra_statute_canons()` runs `EXTRA_STATUTE_RE` over `text_excerpt_original` for rows with empty `statute_anchors` (catches FR/IT phrasings like `article 423 al. 1 CO`, `art 66 al. 4 LTF`).
- 2.5 Map val_001 gold citations to doc_ids via `cit_to_doc_ids`.

### Phase 3 — Citation graph (v7 NEW)
- Loads `citation_graph_extracted.sqlite`, filter `dataset='court_considerations'`.
- Maps each edge `(source_cit, target_cit)` to `(doc_id, doc_id)` via `cit_to_doc_ids`; populates `idx_graph_out[did]` (forward) and `idx_graph_in[did]` (reverse).
- 4-layer graph (per docstring): intra-judgment backrefs, date-stripped aliases, E-range expansion, case-level fan-out.

### Phase 4 — Per-area bedrock + co-citation
- 4.1 `per_area_canon_count[area][canon]` = Counter from `legal_area_per_doc` + `doc_statute_anchors`.
- 4.2 `co_neighbours[canon]` = sorted top-K (8×2) co-citation neighbours filtered by `canon_count <= 5000` and `co_count >= 50`.

### Phase 5 — BM25 (in-memory SQLite FTS5)
- `CREATE VIRTUAL TABLE docs USING fts5(did UNINDEXED, body, tokenize='unicode61 remove_diacritics 2')`, batch inserts of 50k.
- `_enhance_codes(text)`: corpus-derived BM25 lexicon expansion — for each token compute `idf = log(1 + N/df)`, accumulate `code_score[code] += tlf[token][code] * idf`, return top-5 codes. Each repeated 5× in enriched query.
- `bm25_search(q, k)` builds OR-of-quoted-tokens (max 60 terms, min len 3), executes `ORDER BY bm25(docs) LIMIT k`, flips sign.

### Phase 6 — Query expansion (Qwen3-32B, all 10 val queries)
- Load `Qwen/Qwen3-32B` bf16 once.
- For each of 10 queries: build system + user prompts (schema: `statute_targets`, `case_targets`, `concept_targets_en`, `term_targets_de`, `term_targets_fr`, `legal_area_keywords`; two diverse few-shots A=unemployment, B=holographic will; `enable_thinking=False`).
- Deterministic generate; `parse_targets_json()` (markdown fence / brace-balanced scan / greedy regex fallback); if parse weak (`raw is None` or total items < 10), retry with `do_sample=True, temperature=0.4, top_p=0.95, max_new_tokens *= 1.5`.
- Persist `ALL_TARGETS[qid]`, `ALL_RAW_RESPONSES[qid]`. Free `qmod, qtok, _inp, _out`; `gc.collect(); empty_cache(); synchronize()`.

### Phase 7 — Query encoding (Qwen3-Embedding-8B)
- Load via `SentenceTransformer`, encode raw + enriched (raw + first 60 from union of DE+FR terms + EN concepts) for all 10 queries.
- Move tensors to CPU (`ALL_Q_EMB_RAW[qid]`, `ALL_Q_EMB_ENRICHED[qid]`); free model from VRAM.

### Phase 8 — Vector setup (E_GPU)
- Read `qwen3_8b_unified_manifest.parquet` (cols: `doc_id, family, citation, row_index`), build `row_for_did` and `my_did_for_row` by matching `(citation, family)` against `cit_to_doc_ids` + `doc_meta`.
- Load 27 fp16 chunks, `torch.cat` → `E_GPU` on `cuda:0`.
- `vector_search(q_emb, k)`: cosine via L2-normalized `E_GPU @ q`, `topk` → `[(doc_id, score), …]`.

### Phase 9 — 14 channels + RRF fusion + guarantees
Channels (with budget):
1. `law_direct_match` (uncapped) — `co_expanded_canons` (LLM canons ∪ co-citation neighbours) over `idx_law_direct`.
2. `court_statute` (3000) — LLM canons over `idx_court_statute`.
3. `co_citation` (1000) — `co_neighbours` over `idx_law_direct` + `idx_court_statute`.
4. `per_area_bedrock` (150) — `legal_area_keywords` substring-match areas, top-100 canons filtered by `statute_target_codes`, fetch law rows.
5. `statute_backprop` (400) — caught court seeds → statutes they anchor → law rows.
6. `sibling_expansion` (2000) — v7.1: seed-frequency ranked (`score = |distinct seeds sharing court_base|`).
7. `graph_forward` (1500) — `idx_graph_out` from `seed_extended`.
8. `graph_reverse` (1000) — `idx_graph_in` from `seed_extended`.
9. `graph_2hop` (1500, disabled) — 2-hop forward.
10. `concept_en` (600) — `expand_concepts_strict()` substring expansion over `idx_concept_en`.
11. `term_orig` (500) — DE+FR terms over `idx_term_orig`.
12. `bm25` (600) — `bm25_search(query + " " + DE+FR+EN bits)`.
13. `vector_raw` (800) — `vector_search(q_emb_raw)`.
14. `vector_enriched` (800) — `vector_search(q_emb_enriched)`.

Seed pool: union of court hits from topical channels; `seed_extended = seed ∪ court_hits(sibling)` (feeds graph channels and `statute_backprop`).

Fusion:
- `rrf_fuse(channels, k=60, weights=CONFIG.channel_weights)`: `score(did) += w / (k + rank + 1)`.
- `round_robin_guarantee(channels_by_name, guarantee_channels, 400, 1000)` — interleaved merge, max 400 per guarantee channel.
- `apply_neg_gate` removes `is_notification_paragraph=True` and `paragraph_role in {notification, header, empty, metadata}`.
- Final list = gated guarantee items + RRF-tail-fill to topk_final=1000.

### Phase 10 — Aggregate
- Per-query R@1000 table, macro mean, micro recall, per-channel mean recall sorted by usefulness, worst/best query.

### Phase 11 — Diagnosis
- 11.1 Helpers: `channel_rank`, `channel_score`, `_tokens`, `gold_vector_cos`, `concepts_pointing_to`, `terms_pointing_to`, `graph_paths_to`, `graph_paths_from`, `sibling_diagnosis`.
- 11.2 For each missed val_001 gold: print citation, doc meta, search_text (≤240 chars), per-channel HIT (rank) or MISS (with per-channel reason: token overlap / cosine / concept hits / seed graph paths / sibling count / canonical match / row anchors / etc.), RRF score+rank, classify root cause into `NO_CHANNEL_HIT`, `GATED_OUT`, `RRF_RANK_TOO_LOW`, `RRF_ZERO`, `UNKNOWN`.
- 11.3 Failure-mode summary: cause counter, top-10 misses with fix suggestions, per-channel caught-among-missed table.
- 11.4 Cross-query brief: per-query gold/caught/missed/union/best-channel row + union-vs-final gap.

### Phase 12 — Save + cleanup
- Per-query (one folder per `qid`): `final_topk.json` (rank, doc_id, citation), `gold_in_top.json`, `targets.json`, `summary.json`.
- `aggregate_summary.json` at out_dir root with `n_queries`, `mean_R_at_1000_macro`, `R_at_1000_micro`, `mean_union_upper_bound`, per-query block, `channel_mean_recall`.
- `val_001_gold_missed_diagnosis.json`, `config.json`.
- 12.2 `del E_GPU; del EMB_MODEL; gc.collect(); empty_cache()`.

## Results

### Phase 1 sanity
- `Mounted at /content/drive`
- All 7 paths `OK`; `EMB_AVAILABLE = True`, `GRAPH_AVAILABLE = True`.

### Phase 2 — Corpus build
```
Law: 173,033 rows indexed in 10.4s
Token->code association: 91,173 tokens, avg codes/token = 11.7
court progress: 500,000 rows (24.2s)
court progress: 1,000,000 rows (46.6s)
court progress: 1,500,000 rows (72.5s)
court progress: 2,000,000 rows (100.3s)
Court: 2,476,315 rows indexed in 130.8s
Total docs:        2,649,348
Unique citations:  2,158,211
Index sizes:       law_direct=49,288, court_statute=85,063, case=157,227,
                   court_base=178,593, concept=292,946, term=363,331
Co-citation pairs: 1,146,924
```
val_001 gold mapping: `Mapped gold: 42/42`, `Total gold doc_ids: 42`.

### Phase 3 — Graph load
```
Built citation->doc_id map (2,158,211 entries)
Graph: 20,494,436 edges loaded, 3,155,263 skipped (cit not in corpus)
Graph: out-degree avg = 27.2, in-degree avg = 14.0
Graph: load time 96.6s
```

### Phase 4
- Per-area bedrock: `built in 2.1s; areas: 26`. Sample top-8:
  - `criminal law and criminal procedure`: `66 BGG (28151), 42 BGG (20749), 106 BGG (19538), 64 BGG (13843), 108 BGG (12101), 97 BGG (12076), 105 BGG (11759), 81 BGG (11104)`.
  - `constitutional and public law`: `66 BGG (13218), 29 BV (12189), 89 BGG (11480), 82 BGG (11464), 42 BGG (10836), 9 BV (9650), 106 BGG (9626), 68 BGG (8649)`.
  - `administrative, tax, migration, and regulatory law` top1: `42 BGG (20649)`.
  - `civil law`: `63 OJ (3189), 8 ZGB (2741), 55 OJ (2518), …`.
- Co-citation: `2,288 canonicals` after filtering. `221 StPO` top-8 neighbours: `31 BV (783)`, `212 StPO (714)`, `221 BV (646)`, `237 StPO (487)`, `5 StPO (272)`, `197 StPO (258)`, `5 EMRK (246)`, `5 CEDH (235)`.

### Phase 5
```
FTS5 built: 2,649,348 rows in 84.0s
```

### Phase 6 — Query expansion (Qwen3-32B)
- `model loaded in 213.7s`
- Per-query target counts (statute/concept/term_de/term_fr/area):
  - val_001: 20/20/20/20/6
  - val_002: 20/20/20/20/5
  - val_003: 20/25/25/20/6
  - val_004: 20/20/20/19/6
  - val_005: 20/25/25/24/6 (first-pass parse weak, retry with temperature=0.4)
  - val_006: 20/24/20/20/6
  - val_007: 20/26/26/21/6
  - val_008: 20/25/25/21/6 (first-pass parse weak, retry)
  - val_009: 20/20/20/18/5
  - val_010: 20/25/25/25/6
- `[free:qwen3-32b] VRAM used=0.01 GB, free=94.96 GB`

### Phase 7 — Encoding
- `Qwen3-Embedding-8B loaded in 44.1s`
- Enriched-query lengths 2210–2993 chars; keywords appended 58–60.
- `[free:qwen3-emb-8b] VRAM used=0.01 GB, free=94.96 GB`

### Phase 8 — E_GPU
```
manifest rows: 2,652,248
manifest->my_did mapping: 2,649,348/2,652,248 (99.9%) in 38.7s
E_GPU shape=(2652248, 4096) dtype=torch.float16, VRAM used=20.2 GB, free=74.7 GB, load 255.8s
```

### Phase 9 — Per-query R@1000 and per-channel recall

| qid | gold | union_ceil | R@1000 | caught |
|---|---:|---:|---:|---|
| val_001 | 42 | 0.714 | 0.310 | 13/42 |
| val_002 | 36 | 0.667 | 0.528 | 19/36 |
| val_003 | 47 | 0.532 | 0.255 | 11/47 |
| val_004 | 10 | 0.900 | 0.800 | 8/10 |
| val_005 | 11 | 1.000 | 0.545 | 6/11 |
| val_006 | 18 | 0.833 | 0.611 | 11/18 |
| val_007 | 19 | 0.789 | 0.684 | 13/19 |
| val_008 | 29 | 0.552 | 0.448 | 13/29 |
| val_009 | 14 | 0.714 | 0.643 | 9/14 |
| val_010 | 25 | 0.680 | 0.360 | 9/25 |

val_001 per-channel sizes and recall:
```
law_direct_match    size=  90  recall=0.143
court_statute       size=3000  recall=0.048
co_citation         size=1000  recall=0.024
per_area_bedrock    size= 150  recall=0.167
statute_backprop    size= 400  recall=0.190
sibling_expansion   size=2000  recall=0.024
graph_forward       size=1500  recall=0.238
graph_reverse       size=1000  recall=0.095
graph_2hop          size=   0  recall=0.000
concept_en          size= 600  recall=0.190
term_orig           size= 500  recall=0.048
bm25                size= 600  recall=0.048
vector_raw          size= 800  recall=0.119
vector_enriched     size= 800  recall=0.095
```

### Phase 10 — Aggregate
```
Macro mean R@1000:     0.518   (10 queries)
Micro R@1000:          0.446   (112/251 total gold)
Macro mean union ceil: 0.738
```
Per-channel mean recall across 10 queries (sorted):
```
statute_backprop      0.469    std 0.193
graph_forward         0.321    std 0.141
vector_raw            0.153    std 0.134
vector_enriched       0.128    std 0.091
concept_en            0.118    std 0.090
per_area_bedrock      0.105    std 0.098
law_direct_match      0.096    std 0.090
bm25                  0.041    std 0.061
court_statute         0.032    std 0.039
sibling_expansion     0.032    std 0.025
term_orig             0.027    std 0.031
graph_reverse         0.012    std 0.029
co_citation           0.009    std 0.020
graph_2hop            0.000    std 0.000
```
- Worst query: `val_003 (R@1000 = 0.255, union_ceiling = 0.532)`
- Best query: `val_004 (R@1000 = 0.800, union_ceiling = 0.900)`

### Phase 11 — val_001 diagnosis
- `R@1000 = 13/42 = 31.0%`. Missed = 29.
- Root-cause counts:
  - `RRF_RANK_TOO_LOW` — 13
  - `NO_CHANNEL_HIT` — 12
  - `GATED_OUT` — 4
- Query token count (post-enhance): 192.
- val_001 channel hit rates among missed gold:
  ```
  law_direct_match     0
  court_statute        2
  co_citation          1
  per_area_bedrock     0
  statute_backprop     1
  sibling_expansion    1
  graph_forward        1
  graph_reverse        2
  graph_2hop           0
  concept_en           6
  term_orig            2
  bm25                 0
  vector_raw           5
  vector_enriched      4
  ```
- Example traces:
  - `1B_15/2023 E. 3.1`: HIT vector_raw rank 146, vector_enriched rank 333; RRF rank 979 but **GATED_OUT** because `paragraph_role=facts`. (Note: `facts` is not in the configured noise set; per the diagnostic logic, gating dropped it post-RRF before final slot allocation.)
  - `1B_210/2023 E. 4.1`: NO_CHANNEL_HIT; cos_sim_raw=0.6763, cos_sim_enriched=0.6924 (below top-800 threshold); sibling has 15 sibs but only 1 caught; graph_forward has 0 caught seeds with outgoing edges to this row.
  - `BGE 132 I 21 E. 3.2.1`: 1 channel hit, RRF rank 3982 → `RRF_RANK_TOO_LOW`.
  - `1B_90/2021 E. 2.4`: 1 channel hit, RRF rank 9580.

### Phase 11.4 — Cross-query gap (union ∩ gold vs final ∩ gold)
```
qid       in_union  in_final   gap
val_001        30        13    17
val_002        24        19     5
val_003        25        11    14
val_004         9         8     1
val_005        11         6     5
val_006        15        11     4
val_007        15        13     2
val_008        16        13     3
val_009        10         9     1
val_010        17         9     8
```
Best-channel per query: `statute_backprop` for 8/10 (val_002 0.472, val_004 0.800, val_005 0.545, val_006 0.556, val_007 0.579, val_008 0.414, val_009 0.643, val_010 0.360); `graph_forward` for val_001 (0.238) and val_003 (0.213).

### Phase 12 — Persistence
```
Saved per-query results in /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/<qid>/  (10 query folders)
Aggregate summary at /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/aggregate_summary.json
  mean R@1000 (macro): 0.518
  R@1000        (micro): 0.446
VRAM after cleanup: 0.01 GB
cleaned up.
```

## Summary

The v7 anchor-funnel diagnostic build assembled 14 retrieval channels (statute/concept/term/BM25/dense + per-area bedrock + statute backprop + 20.5 M-edge 4-layer citation graph + sibling expansion) and fused them with weighted RRF plus a round-robin guarantee for `[law_direct_match, per_area_bedrock, statute_backprop, graph_forward, sibling_expansion]`. It ran end-to-end across the full 10-query val set (251 gold total), producing macro R@1000 = 0.518 and micro R@1000 = 0.446 against a macro union upper bound of 0.738. `statute_backprop` (mean recall 0.469) and `graph_forward` (0.321) dominated as the only consistently high-signal channels; `graph_reverse`, `co_citation`, and the disabled `graph_2hop` contributed near zero. val_001 — the historical canary — only hit R@1000 = 0.310 (13/42), missing the 0.60 pass criterion and the 0.60 v6→v7 promotion target, with diagnoses attributing misses to RRF rank too low (13), no channel hit (12), and gating (4). Key lesson surfaced by Phase 11.4: the union-vs-final gap (e.g., 17 lost gold on val_001 out of 30 in-union) indicates fusion/gating is now the binding constraint for several queries rather than candidate generation, motivating subsequent iterations to rebalance RRF weights and lift guarantee caps.
