# endgame_colab_full_pipeline.ipynb

**Path:** notebooks/04_pre_v75_pipeline_iterations/endgame_colab_full_pipeline.ipynb

## Configuration

**Runtime / hardware**
- Colab notebook (Drive mount at `/content/drive`), kernel Python 3.11.
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`, VRAM 102.0 GB.
- Env: `PYTHONIOENCODING=utf-8`, `TOKENIZERS_PARALLELISM=false`.

**Models**
- Embedding: `Qwen/Qwen3-Embedding-8B` (bf16, sdpa attention, `max_seq_length=2048`, left padding, normalize_embeddings=True). Query instruction prefix: "Instruct: Given an English-language legal question or scenario about Swiss federal law, retrieve the Swiss statute articles or federal court decision considerations that are most directly relevant to answering it.\nQuery: ".
- Reranker: `Qwen/Qwen3-Reranker-8B` (bf16, `max_seq_len=4096`, batch 8). Score = `log_softmax([no_logit, yes_logit])[..., 1].exp()`. Uses `<|im_start|>system ... <|im_end|>\n<|im_start|>user\n` prefix and `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n` suffix.
- HyDE / Judge / Agentic: `Qwen/Qwen3-8B` (bf16, device_map="auto"); judge runs greedy (`do_sample=False`, `max_new_tokens=200`, `max_seq_len=8192`, batch 4); HyDE samples (`temperature=0.3`, `max_new_tokens=220`).

**Libraries installed**
`transformers>=4.51`, `accelerate`, `sentence-transformers`, `rank_bm25`, `faiss-cpu`, `numpy`, `pandas`, `pyarrow`, `tqdm`, `deep_translator`.

**Master `CONFIG` (Cell 1) — toggles, all default ON except agentic**
- `use_enrichment_prefilter`, `use_bm25`, `use_bm25_lexicon_expansion`, `use_vector`, `use_hyde`, `use_query_german_expansion`, `use_citation_graph_expansion`, `use_legal_area_softfilter`, `use_statute_anchors`, `use_case_anchors`, `use_court_base_sibling_expansion`, `use_authority_score_boost`, `use_granularity_resolver`, `use_reranker`, `use_llm_judge`, `use_dynamic_k_calibration` = `True`
- `use_agentic_retrieval` = `False`

**Numerical knobs**
- `channel_budgets`: `{bm25: 800, vector: 800, statute: 400, case: 300, graph: 200, court_base: 200}`
- `rrf_k = 60`
- `channel_weights`: `{bm25: 1.0, vector: 0.9, hyde: 0.6, statute: 0.8, case: 0.7, graph: 0.4, court_base: 0.4}`
- `authority_alpha = 0.15`
- `rerank_top_n = 200`, `rerank_batch_size = 8`, `rerank_max_seq_len = 4096`
- `rerank_alpha_retrieval = 0.7`, `rerank_alpha_rerank = 0.3`
- `judge_auto_yes = 0.55`, `judge_auto_no = 0.25`, `judge_batch_size = 4`, `judge_max_seq_len = 8192`
- `k_sweep_for_f1 = [5, 7, 10, 13, 15, 20, 25, 30, 50, 75, 100]`
- `dynamic_k_strategy = "score_threshold"`
- `agentic_max_rounds = 2`
- `prefilter_pool_target = 1_000_000`, `prefilter_min_recall_check = True`
- `hyde_max_new_tokens = 220`, `hyde_temperature = 0.3`
- `embedding_max_seq_len = 2048`

**Constants in retrieval code**
- `BGE_FULL_RE = r"\bBGE\s+\d{3}\s+[IVX]{1,4}\s+\d+[a-z]?(?:\s+E\.\s*[\d\.]+)?\b"`
- `DOCKET_BASE_RE = r"\b\d{1,2}[A-Z]{1,4}[_\.]\d{1,5}/\d{4}\b"`
- `ART_PATTERN` regex matching `Art. <n> <law_code>` plus optional `Abs/Bst/Lit/Ziff/Ch/Cpv` qualifiers.
- `KNOWN_LAW_CODES`: 76 codes including AIG, BGG, BV, OR, CO, CC, ZGB, ZPO, StGB, StPO, BV, IVG, ATSG, LTF, etc.
- `PARAGRAPH_UNIT_RE` matches `Abs/Absatz/al/Bst/Buchstabe/lit/Ziff/Ziffer/Nr/Nummer/Satz/Unterabsatz`.
- `BARE_ARTICLE_RE` parses `Art. <article> <law_code>` with bis/ter/quater/quinquies/sexies/septies/octies/novies/decies suffixes.
- BM25 FTS5 reserved keywords filtered: `{AND, OR, NOT, NEAR}`.
- DE legal lexicon (`DE_LEX`): 28 EN->DE keyword mappings (detention, pretrial, collusion, flight risk, appeal, proportionality, fair trial, criminal/civil procedure, contract, liability, damages, compensation, court costs, attorney fees, constitution, tax, marriage, divorce, inheritance, property, bankruptcy, execution, evidence, admissibility, jurisdiction, asylum, extradition, insurance, disability).
- Enrichment fields indexed: `english_summary, legal_topic, legal_question, legal_rule, english_legal_concepts, search_keywords, natural_language_queries, court_holding, concepts_en, topics, legal_area`.
- Judge system prompt (verbatim) defines 7-category Swiss-court rubric: substantive, definitions, procedural, appeal, costs, jurisdiction, constitutional. Format `CITATION | VERDICT: YES or NO`.
- SQLite PRAGMAs: `cache_size=-200000`, `temp_store=MEMORY`.

## Data

**Drive root:** `/content/drive/MyDrive/swiss_law`

**CSV inputs (`data_dir = {DRIVE_ROOT}/data`)**
- `/content/drive/MyDrive/swiss_law/data/train.csv` — 1,139 queries
- `/content/drive/MyDrive/swiss_law/data/val.csv` — 10 queries
- `/content/drive/MyDrive/swiss_law/data/test.csv` — 40 queries
- `/content/drive/MyDrive/swiss_law/data/laws_de.csv`
- `/content/drive/MyDrive/swiss_law/data/court_considerations.csv`

**Artifacts (`artifacts_dir` and `artifacts_v2`)**
- `/content/drive/MyDrive/swiss_law/artifacts_v2/unified_retrieval.sqlite` — FTS5 corpus (`documents`, `documents_fts`, `statute_links`, `case_links`); columns include `doc_id, family, citation, vector_text, authority_score`.
- `/content/drive/MyDrive/swiss_law/artifacts_v2/citation_graph_extracted.sqlite` — edges table.
- `/content/drive/MyDrive/swiss_law/artifacts_v2/law_authority_cards_v2_unified.jsonl`
- `/content/drive/MyDrive/swiss_law/artifacts_v2/court_authority_cards_v5_unified.jsonl`
- `/content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_manifest.parquet`
- `/content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_chunk*.npy` — 27 fp16 chunks.

**Output paths**
- `submissions_dir = /content/drive/MyDrive/swiss_law/submissions`
- `cache_dir = /content/drive/MyDrive/swiss_law/cache_endgame`
- `eval_out_dir = /content/drive/MyDrive/swiss_law/artifacts/eval`
- Submission file: `{submissions_dir}/submission_endgame.csv`
- Per-query eval: `{eval_out_dir}/endgame_val_per_query.csv`
- Ablation: `{eval_out_dir}/endgame_ablation_val.csv`
- Channel recall: `{eval_out_dir}/endgame_channel_recall.csv`

**Loaded sizes (verbatim from outputs)**
- Unified corpus: court 2,476,315; law 175,933; TOTAL 2,652,248.
- Enrichment: 175,933 law cards + 2,476,315 court cards in 253.9s; indexed by citation 2,161,111.
- Citation graph: 4,599,562 edges; 1,173,383 forward sources.
- Vector manifest: 2,652,248 rows; 27 chunks cover all rows; GPU index shape `(2652248, 4096) fp16`, GPU mem 21.7 GB.

## Pipeline

### Stage 0 — Setup (Cells 1-2)
- Define `CONFIG`; `must_exist(path, label)` raises FileNotFoundError if missing.
- Mount Drive; pip-install dependencies; create `cache_dir`, `submissions_dir`, `eval_out_dir`.

### Stage 1 — Data + corpus schema check (Cell 3)
- Read `train.csv`, `val.csv`, `test.csv` via pandas.
- Open `unified_retrieval.sqlite` read-only, group counts by `family`, assert required cols `{doc_id, family, citation, vector_text, authority_score}` present.

### Stage 2 — Enrichment loading (Cell 4)
- `_load_jsonl()` reads JSONL; build `ENRICH_BY_CIT` (keyed by `citation`/`citation_canon`) and `ENRICH_BY_DOC[(family, doc_id)]`.

### Stage 3 — Granularity resolver (Cell 5)
- `is_article_level()` — true when no paragraph unit token present.
- `_parse_article_law_code()` — extracts `(article, law_code)` from `Art. <n> <code>`.
- `expand_article_to_paragraphs()` (lru_cache=200,000): looks up siblings in SQLite first via `family='law' AND law_code=? AND article=? AND granularity='paragraph'`, then falls back to `citation LIKE 'Art. <article> %<law_code>'` filtered by `PARAGRAPH_UNIT_RE`.
- `apply_granularity_post_filter()` expands the list, deduped.

### Stage 4 — Citation-graph utilities (Cell 6)
- Auto-detects edge table from `{edges, citation_edges, citations_edges, graph_edges}`.
- Builds `GRAPH_FORWARD` / `GRAPH_BACKWARD` defaultdict-of-sets.
- `expand_via_graph(seed, hops=1, max_neighbors=200)`.

### Stage 5 — BM25 / FTS5 + token-level law-abbrev expansion (Cell 7)
- Open BM25 read-only SQLite connection (`con_bm25`).
- Build `tok_to_abbrev` and `total_per_tok` Counter by iterating `train_df.gold_citations`, taking last whitespace-separated token of citation as abbreviation when citation matches `Art.\s+\d`.
- `_approx_idf(tok) = log(1 + n_docs / max(1, total_per_tok[tok]))`.
- `enhance_query_tokens()`: for each query token, score abbreviations by `(cnt/total) * idf` if `idf >= 1.0`; append top-5 abbrevs duplicated x5.
- `bm25_search(query, k=800)` runs `MATCH ?` join (`documents_fts JOIN documents USING (doc_id)`) ordered by FTS5 `bm25()` ascending.

### Stage 6 — Vector index (Cells 8-9)
- Cell 8 (CPU mmap fallback): blocked dot-product with min-heap top-K.
- Cell 9 (GPU override, "Cell 8b"): loads all 27 chunks to single fp16 GPU tensor `E_GPU` (one-time ~30s, actual 183.8s; 21.7 GB GPU mem). `vector_search` does `E_GPU @ q` then `torch.topk` on fp32.

### Stage 7 — Query encoder (Cell 10)
- Lazy-load `SentenceTransformer(Qwen3-Embedding-8B, device='cuda', torch_dtype=bf16, attn_implementation='sdpa')`.
- Disk cache: `{cache_dir}/query_embeddings.npy` + `query_embeddings_keys.json`; key = `sha1(query)`.
- `encode_query(text)` prepends `QWEN_QUERY_INSTRUCT`, encodes with `normalize_embeddings=True`, casts fp32.

### Stage 8 — Query understanding (Cell 11)
- `parse_query_anchors(query)` returns `{cases, dockets, statutes, law_codes}` using `BGE_FULL_RE`, `DOCKET_BASE_RE`, `ART_PATTERN`, and standalone tokens against `KNOWN_LAW_CODES`.

### Stage 9 — HyDE (Cell 12)
- Cached at `{cache_dir}/hyde_cache.json` (key sha1 of query).
- `HYDE_PROMPT_DE` instructs the model to write a concise 2-3 sentence GERMAN hypothetical legal answer; output truncated at `</think>`.
- Lazy-loads Qwen3-8B; generates with `do_sample=True, temperature=0.3, max_new_tokens=220`.

### Stage 10 — German keyword expansion (Cell 13)
- Cached at `{cache_dir}/german_expansion_cache.json`.
- Primary: `deep_translator.GoogleTranslator(source='auto', target='de')`.
- Fallback: lexicon-based — concat all `DE_LEX[en]` whose English key is a substring of the lowercased query.

### Stage 11 — Multi-channel retrieve (Cell 14)
Channels (each toggle-gated):
1. BM25 (query + German expansion concatenated).
2. Vector (raw query embedding).
3. HyDE vector (encoded HyDE German text), channel name `hyde`.
4. Statute anchor — SQL `statute_links` `WHERE statute LIKE ?` ordered by `authority_score DESC`.
5. Case anchor — SQL `case_links` `WHERE target_citation LIKE ? OR target_base LIKE ?` ordered by authority.
6. Court-base sibling expansion — extract BGE/docket bases from top-20 BM25+vector court seeds; SQL `family='court' AND citation LIKE ?`.
7. Graph 1-hop on seed citations from top-20 BM25 + top-20 vector.
8. Legal-area soft filter — multiplies `raw_score` by 1.1 if last token of citation equals one of the query's anchored law codes.

**Fusion** — `reciprocal_rank_fusion(channel_results, k=rrf_k, weights=channel_weights, authority_alpha=0.15)`:
`score[d] = sum_channel weights[channel] / (k + rank)`; multiplied by `(1 + authority_alpha * authority_score)` if `use_authority_score_boost`.

### Stage 12 — Enrichment pre-filter (Cell 15)
- Inverted index `ENRICHMENT_INDEX[token] -> set(doc_id)` built over all enrichment cards (token min length 3).
- `enrichment_prefilter(query)` unions `tokenise(query) | tokenise(german_expand(query))`, sums per-doc hit counts, keeps top-`prefilter_pool_target` if larger.
- Validates against val: pool size + gold recall per query.

### Stage 13 — Reranker (Cell 16)
- `QwenReranker`: tokenize `<Instruct>: ...\n<Query>: ...\n<Document>: ...` body, prefix/suffix added, left-padded, predict logits, take `log_softmax([no_logit, yes_logit])[..., 1].exp()`.
- Per-query disk cache at `{cache_dir}/rerank/{sha1(query)}.json` mapping doc_id -> score.
- `_doc_text_for_rerank(c)` concatenates citation + family + enrichment fields (`english_summary`, `legal_topic`, `legal_rule`, `court_holding`, `english_legal_concepts`) and 600-char text snippet.
- `_hydrate_text(cands)` selects `vector_text` from SQLite for missing docs in chunks of 800.
- `rerank_candidates()` reranks top-200; merges with retrieval score via `fused_score = 0.7 * minmax(retrieval) + 0.3 * rerank_score`.

### Stage 14 — LLM judge (Cell 17)
- Cached per (query, citation) under `{cache_dir}/judge/{sha1(query)}/{sha1(citation)}.json`.
- `route_and_judge()` zones candidates: `score >= 0.55` -> auto-yes, `score < 0.25` -> auto-no, else borderline → judged. Greedy gen with 200 new tokens.
- `_parse_judge_verdict()` strips `</think>` prefix, looks for `|`-delimited verdict line; defaults to `yes` if absent.

### Stage 15 — Agentic retrieval (Cell 18, OFF by default)
- Round 1 normal retrieve; uses judge model to suggest 3-8 German legal terms; appends to query; re-retrieves. Up to `agentic_max_rounds` rounds.

### Stage 16 — Granularity post-filter wrapper (Cell 19)

### Stage 17 — Dynamic-K (Cell 20)
- `select_predictions_score_threshold`: takes all `verdict=="yes"`, else top-50 fused.
- `select_predictions_static_k`: takes top-K of `k_sweep_for_f1`.

### Stage 18 — End-to-end `predict()` (Cell 21)
Order: encode_query -> enrichment_prefilter -> retrieve (or agentic) -> rerank -> route_and_judge -> select_predictions_* -> apply_granularity_to_predictions. Returns predictions + diagnostics.

### Stage 19 — Validation evaluation (Cell 22)
- `parse_gold_str()` (semicolon split, whitespace-normalised), `per_query_f1()`.
- `run_pipeline_on_split(val_df)`, then macro P/R/F1 over 10 rows.
- Saves `endgame_val_per_query.csv`; sets `VAL_BASELINE_F1`.

### Stage 20 — Ablation sweep (Cell 23)
- For each of 16 toggles in `ABLATION_TOGGLES`, flips it OFF (keeping others ON via `CONFIG.update(CONFIG_BACKUP); CONFIG[t] = False`), re-runs full pipeline on val, records F1 / delta / P / R / avg_k.
- Saves `endgame_ablation_val.csv`.

### Stage 21 — Channel-alone recall@K (Cell 24)
- For channels `{bm25, vector, statute, case}`, takes top-1000 alone, measures `R@{50, 100, 200, 500, 1000}` over val.
- Saves `endgame_channel_recall.csv`.

### Stage 22 — Test submission (Cell 25)
- Runs `predict()` over 40 test queries; writes `submissions/submission_endgame.csv` with columns `query_id, gold_citations` (semicolon-joined predictions).

### Stage 23 — Failure analysis (Cell 26)
- Picks worst-3 val queries by F1; for each prints query snippet, timings, zone counts, channel counts, false negatives (with `NEVER retrieved` / `in-retrieve, dropped before judge` / `in-judge verdict=...` tag), top-8 false positives.

### Cell 27 (markdown) — manual summary template referencing the ablation CSV.

## Results

### Cell 1 — config print
```
CONFIG toggles:
  use_enrichment_prefilter                 = True
  use_bm25                                 = True
  use_bm25_lexicon_expansion               = True
  use_vector                               = True
  use_hyde                                 = True
  use_query_german_expansion               = True
  use_citation_graph_expansion             = True
  use_legal_area_softfilter                = True
  use_statute_anchors                      = True
  use_case_anchors                         = True
  use_court_base_sibling_expansion         = True
  use_authority_score_boost                = True
  use_granularity_resolver                 = True
  use_reranker                             = True
  use_llm_judge                            = True
  use_agentic_retrieval                    = False
  use_dynamic_k_calibration                = True

Drive root: /content/drive/MyDrive/swiss_law
```

### Cell 2
```
Drive already mounted at /content/drive
Dependencies installed.
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition  VRAM: 102.0 GB
```

### Cell 3 — corpus counts
```
train: 1,139 queries
val:   10 queries
test:  40 queries

Unified corpus document counts:
  court  :  2,476,315
  law    :    175,933
  TOTAL  :  2,652,248

Loaded in 2.25s
```

### Cell 4
```
Loaded 175,933 law cards + 2,476,315 court cards in 253.9s
Indexed by citation: 2,161,111
```

### Cell 5 — granularity demo (no expansion needed — all already paragraph-level)
```
q=val_001  IN=['Art. 221 Abs. 1 StPO', 'Art. 140 Abs. 1 StGB', 'Art. 396 Abs. 1 StPO', 'Art. 222 StPO', 'Art. 393 Abs. 1 StPO', 'Art. 382 Abs. 1 StPO']
     OUT=['Art. 221 Abs. 1 StPO', 'Art. 140 Abs. 1 StGB', 'Art. 396 Abs. 1 StPO', 'Art. 222 StPO', 'Art. 393 Abs. 1 StPO', 'Art. 382 Abs. 1 StPO']
q=val_002  IN=['Art. 8 Abs. 1 ATSG', 'Art. 8 Abs. 1 IVG', 'Art. 17 Abs. 1 IVG', 'Art. 1 Abs. 1 IVG', 'Art. 56 Abs. 1 ATSG', 'Art. 69 Abs. 1 IVG']
     OUT=[...identical...]
q=val_003  IN=['Art. 29 Abs. 2 BV', 'Art. 221 Abs. 1 StPO', 'Art. 393 Abs. 1 StPO', 'Art. 222 StPO', 'Art. 384 StPO', 'Art. 396 Abs. 1 StPO']
     OUT=[...identical...]
```

### Cell 6 — citation graph
```
[graph] loaded 4,599,562 edges; 1,173,383 forward sources.
[graph demo] 'Art. 221 StPO' 1-hop: []
```

### Cell 7 — BM25 + lexicon
```
FTS5 tables: ['documents_fts']
Lexicon built in 0.2s. |vocab|=19,991 tokens.

BM25 smoke test (33.18s, top 5):
  [law] Art. 14 Abs. 1 FIDLEV    score=178.510
  [law] Art. 16 Abs. 1 FIDLEV    score=178.411
  [law] Art. 49 Abs. 1 FIDLEV    score=178.410
  [law] Art. 81 Abs. 1 FIDLEV    score=178.410
  [law] Art. 83 Abs. 1 FIDLEV    score=178.410
```

### Cell 8 — vector manifest
```
[vector] manifest rows: 2,652,248
[vector] 27 chunks, covered_rows=2,652,248
```

### Cell 9 — GPU vector index
```
[vector-gpu] loading all 27 chunks to GPU (one-time, ~30s)...
[vector-gpu] loaded shape=(2652248, 4096) dtype=torch.float16 in 183.8s; GPU mem=21.7 GB
[vector-gpu] smoke query: 10 hits in 381.4 ms
```

### Cell 11 — anchor extraction
```
q=val_001  anchors={'cases': [], 'dockets': [], 'statutes': ['Art. 221 lit'], 'law_codes': ['StPO']}
q=val_002  anchors={'cases': [], 'dockets': [], 'statutes': ['Art. 17 LAI'], 'law_codes': ['LAI']}
q=val_003  anchors={'cases': [], 'dockets': [], 'statutes': [], 'law_codes': ['StPO']}
```

### Cell 14 — retrieve smoke test (val[0])
```
fused      : 10  timings={'bm25': 39.60, 'vector': ~0, 'hyde': 0.016, 'anchor': 703.89, 'court_base': 2.10, 'graph': 0.022}
fused top 5:
  [court] BGE 128 I 149 E. 2     fused=0.0298
  [court] BGE 138 IV 92 E. 3.2   fused=0.0295
  [law]   Art. 227 Abs. 7 StPO   fused=0.0252
  [law]   Art. 788 Abs. 1 ZGB    fused=0.0218
  [law]   Art. 333 Abs. 3 StGB   fused=0.0215
```

### Cell 15 — enrichment prefilter index
```
Building enrichment inverted index over 2,161,111 cards...
  index ready in 2.7s; |vocab|=0
```
(The index built with `|vocab|=0`, meaning no English enrichment text was indexed — the `_enrich_text()` extraction yielded zero tokens. The per-query recall validation block did not print, because `ENRICHMENT_INDEX` was empty.)

### Cell 19 — granularity expansion demo
```
Granularity expansion: ['Art. 78 BV', 'Art. 221 Abs. 1 StPO', 'BGE 137 IV 122 E. 4.2']
   -> ['Art. 78 Abs. 1 BV', 'Art. 78 Abs. 2 BV', 'Art. 78 Abs. 3 BV', 'Art. 78 Abs. 4 BV', 'Art. 78 Abs. 5 BV', 'Art. 221 Abs. 1 StPO', 'BGE 137 IV 122 E. 4.2']
```

### Cell 21 — predict() smoke test on val_001
```
query_id : val_001
timings  : {'encode_query': 54.60, 'retrieve': 74.54, 'rerank': 57.26, 'judge': 168.83, 'select': ~0, 'granularity': ~0}
|preds|  : 112
first 8  : ['1B_651/2022 E. 2', 'BGE 137 IV 122 E. 5', 'BGE 128 I 149 E. 2', 'BGE 138 IV 92 E. 3.2',
           '7B_59/2025 E. 2.1', '7B_687/2024 E. 5.3', '1B_24/2022 E. 4', '1B_135/2022 E. 2.1']
```

### Cell 22 — headline val Macro F1 (all toggles ON), Done in 1636.6s
```
qid          P      R     F1   |P|   |G|    sec
--------------------------------------------------------
val_001  0.018  0.048  0.026   112    42   47.3
val_002  0.008  0.028  0.013   123    36  211.7
val_003  0.007  0.021  0.010   150    47  360.6
val_004  0.063  0.400  0.110    63    10  113.1
val_005  0.032  0.364  0.059   125    11  191.2
val_006  0.087  0.111  0.098    23    18  101.5
val_007  0.027  0.091  0.042    73    22  172.9
val_008  0.050  0.103  0.067    60    29  126.8
val_009  0.022  0.143  0.038    90    14  149.7
val_010  0.015  0.040  0.022    66    25  142.1
--------------------------------------------------------
MACRO    0.033  0.135  0.048

>>> HEADLINE val Macro F1 (all toggles ON) = 0.0485
```

### Cell 23 — Ablation table (sorted by delta)
```
toggle_off                            F1     delta      P      R   avg_k
------------------------------------------------------------------------
use_bm25                          0.0450   -0.0035  0.029  0.148   109.2
use_statute_anchors               0.0459   -0.0026  0.029  0.137    88.7
use_authority_score_boost         0.0459   -0.0025  0.030  0.146   100.4
use_court_base_sibling_expansion  0.0465   -0.0020  0.030  0.137    92.6
use_query_german_expansion        0.0470   -0.0015  0.031  0.135    81.7
use_citation_graph_expansion      0.0470   -0.0015  0.032  0.126    88.5
use_dynamic_k_calibration         0.0471   -0.0013  0.100  0.032     5.0
(baseline)                        0.0485   +0.0000  0.033  0.135    88.5
use_legal_area_softfilter         0.0485   +0.0000  0.033  0.135    88.5
use_case_anchors                  0.0485   +0.0000  0.033  0.135    88.5
use_enrichment_prefilter          0.0485   +0.0000  0.033  0.135    88.5
use_granularity_resolver          0.0486   +0.0001  0.033  0.136    88.5
use_vector                        0.0530   +0.0046  0.044  0.109    56.9
use_hyde                          0.0542   +0.0057  0.040  0.135    74.7
use_bm25_lexicon_expansion        0.0544   +0.0060  0.040  0.147    93.1
use_reranker                      0.0610   +0.0125  0.055  0.075    20.0
use_llm_judge                     0.0812   +0.0327  0.154  0.076    14.0
```

### Cell 24 — channel-alone Recall@K on val (gold expanded)
```
channel     R@50    R@100   R@200   R@500   R@1000
--------------------------------------------------
bm25        0.053   0.108   0.117   0.165   0.194
vector      0.108   0.117   0.154   0.185   0.248
statute     0.008   0.008   0.008   0.008   0.008
case        0.000   0.000   0.000   0.000   0.000
```

### Cell 25 (test submission)
No captured output (cell was not executed in this notebook copy).

### Cell 26 — worst-3 val failure analysis
**val_003** F1=0.010 P=0.007 R=0.021; |pred|=150 |gold|=47 |FP|=149 |FN|=46
- zone_counts: `{yes: 150, no: 850, borderline: 0}`
- channel_counts: `{bm25:800, vector:800, hyde:800, statute:0, case:0, court_base:200, graph:36}`
- FNs (all marked `NEVER retrieved`): `1B_192/2022 E. 4.1.2`, `1B_195/2022 E. 2.2.1`, `1B_211/2017 E. 2.1`, `1B_572/2021 E. 2.1`, `1B_581/2022 E. 2.1.2`, `1B_88/2022 E. 2.1`, `2C_501/2020 E. 5.1`, `Art. 100 Abs. 1 BGG`, `Art. 135 Abs. 3 StPO`, `Art. 135 Abs. 4 StPO`, `Art. 16 ZGB`, `Art. 214 Abs. 4 StPO`.

**val_002** F1=0.013 P=0.008 R=0.028; |pred|=123 |gold|=36 |FP|=122 |FN|=35
- zone_counts: `{yes: 127, no: 873, borderline: 0}`
- channel_counts: `{bm25:800, vector:800, hyde:800, statute:239, case:0, court_base:200, graph:34}`
- FNs (all `NEVER retrieved`): `8C_160/2016 E. 4.1`, `8C_421/2023 E. 2.2`, `8C_510/2020 E. 2.4`, `9C_623/2020 E. 4.2`, `Art. 1 Abs. 1 IVG`, `Art. 100 Abs. 1 BGG`, `Art. 113 BGG`, `Art. 16 ATSG`, `Art. 17 Abs. 1 IVG`, `Art. 18d IVG`, `Art. 21 Abs. 4 ATSG`, `Art. 28 Abs. 1 IVG`.

**val_010** F1=0.022 P=0.015 R=0.040; |pred|=66 |gold|=25 |FP|=65 |FN|=24
- zone_counts: `{yes: 66, no: 934, borderline: 0}`
- channel_counts: `{bm25:800, vector:800, hyde:800, statute:0, case:0, court_base:200, graph:16}`
- FNs (first, all `NEVER retrieved`): `4A_42/2015 E. 5.5`, `4A_42/2015 E. 6.3`, `4A_42/2015 E. 6.6`, `Art. 100 Abs. 1 BGG`, `Art. 100 Abs. 1 OR`, `Art. 100 Abs. 2 OR`, `Art. 101 Abs. 3 OR` (output truncated).

### Saved files
- `/content/drive/MyDrive/swiss_law/artifacts/eval/endgame_val_per_query.csv`
- `/content/drive/MyDrive/swiss_law/artifacts/eval/endgame_ablation_val.csv`
- `/content/drive/MyDrive/swiss_law/artifacts/eval/endgame_channel_recall.csv`

## Summary

The notebook is the "endgame" toggle-gated multi-channel pipeline: enrichment prefilter -> 8-channel retrieval (BM25+lexicon, dense vector, HyDE, statute/case/court_base anchors, graph 1-hop, legal-area soft filter) -> RRF fusion with authority boost -> Qwen3-Reranker-8B -> Qwen3-8B 7-category judge with auto-yes/no zoning -> granularity expansion -> dynamic-K, plus an ablation sweep over 16 toggles, channel-alone recall diagnostic, and worst-3 failure analysis. The headline val Macro F1 with every channel ON was **0.0485**, far below the Untitled75 v12 baseline of 0.777. The ablation showed that the heavyweight LLM stages were hurting: turning `use_llm_judge` OFF raised F1 to 0.0812 (+0.0327), `use_reranker` OFF gave 0.0610 (+0.0125), `use_bm25_lexicon_expansion`/`use_hyde`/`use_vector` OFF each gained ~+0.005, and only `use_bm25` was clearly load-bearing (-0.0035 when off); `use_enrichment_prefilter`, `use_case_anchors`, `use_legal_area_softfilter` were no-ops because the enrichment index built with |vocab|=0 and the anchors rarely fired. Channel-alone R@1000 was 0.194 (bm25), 0.248 (vector), 0.008 (statute), 0.000 (case), matching the Observation-3 ceiling, and the failure-analysis output shows the dominant failure mode is "NEVER retrieved" — the underlying recall is too low for any rerank+judge stack to recover. Key lessons: the enrichment prefilter was silently broken (empty token vocabulary, fields not present on the cards as written), the LLM judge over-accepted (avg_k ~88 with too many auto-yes promotions because most candidates exceeded the 0.55 threshold after fused scoring), and the pipeline is bottlenecked at retrieval recall rather than ranking precision.
