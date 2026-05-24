# rerank_hybrid_3model_shootout.ipynb

**Path:** notebooks/01_current_direction_cascade_rerank_precision/rerank_hybrid_3model_shootout.ipynb

## Configuration

### Models (cross-encoder rerankers)
- **Qwen/Qwen3-Reranker-8B** — loaded via vLLM; dtype `bfloat16`; `max_model_len=1024`; `gpu_memory_utilization=0.85`; `enforce_eager=False`; `trust_remote_code=True`. Scoring via yes/no logprobs (softmax over yes/no token logprobs). `SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)`. Prompt template uses `<|im_start|>system .. <|im_end|> <|im_start|>user .. <|im_end|> <|im_start|>assistant <think>\n\n</think>` wrapper.
- **BAAI/bge-reranker-v2-m3** — `AutoModelForSequenceClassification` in `torch.bfloat16` on CUDA; `BGE_MAX_LEN=1024`; `BGE_BATCH=64`; sigmoid over logits.
- **jinaai/jina-reranker-v2-base-multilingual** — `AutoModelForSequenceClassification` in `torch.bfloat16` on CUDA with `trust_remote_code=True`; `JINA_MAX_LEN=1024`; `JINA_BATCH=128`; sigmoid over logits.

### Libraries
- `vllm>=0.10.0`, `transformers>=4.51.0`, `accelerate`, `safetensors`, `huggingface_hub`
- `flashinfer-python`, `flashinfer-cubin`, `flashinfer-jit-cache` (CUDA-version-aware install)
- `torch`, `numpy`, `pandas`, `gzip`, `json`

### Hyperparameters / constants
- `RERANK_TOP_N = 50000` (full pool reranked per query)
- `K_REPORT = (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000)`
- `RECALL_FLOOR = 0.80` (both macro R AND min-per-query R must clear this)
- `K_RRF = 60` (Cormack et al. 2009 standard)
- `TEXT_TRUNC_REGEX = 2500` (chars for doc-side feature extraction)
- Enriched doc representation: 3000-char string concatenating `Citation`, `Type: {family} ({role})`, `Statute anchors` (up to 8 / 240 chars), `Concepts (English)` (up to 8 / 240 chars), `Terms` (up to 6 / 180 chars), then body truncated to 2000 chars.
- Cross-lingual instruction (passed to Qwen3): "The Query is an English question about Swiss federal law. The Document is a Swiss federal law article (German) or a Swiss court paragraph (German, French, or Italian)..."

### Hardware / environment
- Environment auto-detected: `colab` (mounts `/content/drive`); local path falls back to `E:\swiss_citation_extraction`.
- Run executed on Colab.
- GPU: RTX PRO 6000 Blackwell (per docstring). vLLM logged `Model loading took 15.26 GiB memory` for Qwen3-8B.
- FlashInfer warning: `Failed to get device capability: SM 12.x requires CUDA >= 12.9`.

### Configurations under test
| Config | Definition |
|---|---|
| F0_fusion | v7.5 RRF fusion baseline only |
| F1_qwen3 | Qwen3-Reranker-8B alone |
| F2_rerank_ensemble | RRF(Qwen3, BGE, jina) |
| F3_HYBRID | F2 + 9 dossier signals (statute_int, lead_statute_int, concept_int, term_int, co_cite, doctrinal, chamber_match, fusion_rank, -hard_neg) — 12 signals total |
| F4_HYBRID_hardneg_filter | F3 with `hard_neg==1` rows dropped pre-fusion |

### Dossier feature matrix columns (per qid × did)
`["statute_int", "lead_statute_int", "concept_int", "term_int", "co_cite", "doctrinal", "chamber_match", "hard_neg", "fusion_rank"]`

## Data

- `_DRIVE = /content/drive/MyDrive/swiss_law` (Colab) or `E:\swiss_citation_extraction` (local)
- `SNAPSHOT_DIR = /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot` (auto-detects single- or double-nested layout)
- `OUT_DIR = /content/drive/MyDrive/swiss_law/research/hybrid_rerank_final`
- `CACHE_DIR = /content/drive/MyDrive/swiss_law/research/hybrid_rerank_final/cache`

Snapshot inputs:
- `snapshot/config.json`
- `snapshot/corpus_snapshot.json.gz` — gzipped JSON keyed by did with fields `ct` (search_text), `cit` (citation), `fam` (family), `cb` (court_base), `pr` (paragraph_role), `ln` (language), `sa` (doc_statute_anchors), `cn` (_doc_to_concepts), `tm` (_doc_to_terms)
- `snapshot/per_query_snapshot.json` — `final_topk`, `curve`, `channel_recalls` per qid
- `snapshot/gold_doc_sets.json` — gold doc id lists per qid (truth source; `per_query_snapshot.gold_doc_ids` is just the count)
- `snapshot/all_targets.json` — LLM-expanded `statute_targets`, `concept_targets_en`, `term_targets_{de,fr,it}`, `legal_area_keywords`
- `snapshot/hyde_aspects.json` (optional)
- `snapshot/paths.json` — points to `val_csv`
- `val.csv` (path from `paths.json`, fallback `_DRIVE / "data" / "val.csv"`)

Cache outputs:
- `cache/dossier_features.npz` — `<qid>__dids` + `<qid>__feat (N,9) float32`
- `cache/scores_qwen3.npz` — per-qid Qwen3 score arrays
- `cache/scores_bge.npz` — per-qid BGE score arrays
- `cache/scores_jina.npz` — per-qid jina score arrays

Final output:
- `hybrid_rerank_final/hybrid_rerank_results.json`

## Pipeline

### Phase 0 — Setup
Env detection (Colab vs local). One-shot install of vLLM + transformers and (CUDA-aware) FlashInfer; raises `SystemExit("Restart runtime...")` after first install. Sets `TOKENIZERS_PARALLELISM=false`. Creates `OUT_DIR` and `CACHE_DIR`.

### Phase 1 — Warm-boot from snapshot
Loads `corpus_snapshot.json.gz` into `search_text`, `doc_meta`, `doc_statute_anchors`, `_doc_to_concepts`, `_doc_to_terms`. Loads `per_query_snapshot.json` into `PER_QUERY` (final_topk + curve + channel_recalls). Loads `gold_doc_sets.json` into `ALL_GOLD_DOC_SET`. Loads `all_targets.json`, `hyde_aspects.json`. Loads `val.csv` into `QUERY_TEXT`. Asserts query text exists for every qid.

### Phase 2 — Dossier helpers (substring path, no backtracking)
Defines:
- `canonicalize_statute(s)` — token-walk extraction of `"<NUM> <CODE>"`; LAW_CODES tuple of 40+ Swiss codes.
- `lead_codes(text_lead)` — same on first 200 chars.
- Substring marker tuples: `DE_RULE_OPENERS`, `DE_DOCTRINE`, `DE_RULE_VERBS`, `FR_RULE_OPENERS`, `FR_DOCTRINE`, `FR_RULE_VERBS`, and hard-neg signatures (dispositif / costs / facts / signature).
- `doctrinal_density(text)` — scalar score in `[-10, +20]`.
- `is_hard_neg(text, role)` — bool; role-set is `{dispositif, costs, facts, procedural_history, admissibility, signature}`.
- `chamber_of(citation)` — small regex on short citation strings only.
- `query_legal_areas(qid)` + `allowed_chambers(qid)` — 5-row legal-area-to-chamber map (criminal/civil/admin/social/labor/etc.).

### Phase 3 — Precompute per-(qid, did) dossier feature matrix
Disk cache check: if `cache/dossier_features.npz` exists, load and assert pool match.
Otherwise:
- 3a: per-doc cached features over `doc_meta` — `_doc_doctrinal`, `_doc_hardneg`, `_doc_canon_statutes` (already-canonical anchors lifted directly), `_doc_lead_canon_statutes` (lead 200 chars), `_doc_chamber`.
- 3b: co-citation index — `_statute_cite_count` (court-paras citing each canonical statute) and `_court_case_to_peers` (peers per case base) over the union of all 10 query pools. `co_cite_count(did)` returns law-side citing count or court-side peer count − 1.
- 3c: build `DOSSIER[qid] = {"did_list": pool, "feat": np.ndarray (N, 9)}` filling all 9 columns. `chamber_match` is 1 if no chamber constraint OR family == "law" OR chamber matches.
- 3-save: `np.savez_compressed` to cache.

### Phase 4 — Enriched doc representation
`repr_enriched(did)` builds the 3000-char string used by all 3 rerankers. `CROSS_LING_INSTR` defines the Qwen3 instruct prompt. Sets storage dict `ALL_RERANK_SCORES`.

### Phase 5 — Cross-encoder A: Qwen3-Reranker-8B (vLLM)
Cache-or-compute. Loads tokenizer (`padding_side="left"`) and vLLM engine. Constructs PREFIX/SUFFIX token id arrays; truncates body to `QWEN3_MAX_BODY = 1024 - len(PREFIX) - len(SUFFIX) - 8`. Scores each query × 50k pool via `qwen3_llm.generate(...)` with `logprobs=20`; converts `yes` and `no` token logprobs to `exp(y) / (exp(y) + exp(n))`. Prints quick R@2000 per query. Saves to `scores_qwen3.npz`; frees CUDA.

### Phase 6 — Cross-encoder B: BGE-reranker-v2-m3 (HF)
Cache-or-compute. Loads in `bfloat16` on CUDA; batches 64 pairs at a time; `torch.sigmoid(logits)`. Prints R@2000 per query. Saves and frees.

### Phase 7 — Cross-encoder C: jina-reranker-v2-base-multilingual (HF)
Cache-or-compute. Loads with `trust_remote_code=True` in `bfloat16` on CUDA; batches 128 pairs; `torch.sigmoid(logits)`. Saves and frees.

### Phase 8 — RRF score combination
- `rank_from_scores(scores, descending=True)` — 1-indexed stable-sort ranks.
- `rrf_of(rank_arrays)` — `sum(1/(K_RRF + rank))` over arrays.
- `evaluate(qid, ranked_pool)` — cumulative-hit R@K over `K_REPORT` per qid.
- Loops per qid: extracts `fusion_rank` from feat col 8, computes ranks for `qwen3/bge/jina` (descending) and for each dossier signal (descending; `r_notneg = rank(-hard_neg)`). Computes the 5 configs' orderings (F0..F4), applying hard-neg as a filter in F4 by removing rows with `hard_neg==1` from the F3 order.

### Phase 9 — Summary tables
For each config, finds `K_min` = smallest K in `K_REPORT` such that `macro_R >= 0.8 AND min_R >= 0.8`. Prints (a) headline summary, (b) full R@K macro/min/worst-qid table per config, (c) per-query R@K for F3, (d) macro R@K matrix. Saves consolidated `hybrid_rerank_results.json`.

### Phase 10 — Outcome interpretation
Markdown only. Maps possible result patterns (A: F3 lifts K_min; B: F3==F2; C: structural ceiling; D: F4 beats F3; E: F1==F3) to production actions.

## Results

### Phase 0 output (Setup)
```
Python 3.12
Drive already mounted at /content/drive
Environment: colab
snapshot: /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot
out:      /content/drive/MyDrive/swiss_law/research/hybrid_rerank_final
cache:    /content/drive/MyDrive/swiss_law/research/hybrid_rerank_final/cache
[setup] vLLM + FlashInfer installed; proceeding.
```

### Phase 1 output (Warm-boot)
```
[warm-boot] loading from /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot
  corpus snapshot: 255,713 docs
  PER_QUERY: 10 queries
    val_001: pool=44621  gold= 42  in_pool= 39  R_max=0.929
    val_002: pool=41254  gold= 38  in_pool= 30  R_max=0.789
    val_003: pool=44033  gold= 47  in_pool= 36  R_max=0.766
    val_004: pool=43086  gold= 10  in_pool= 10  R_max=1.000
    val_005: pool=42140  gold= 11  in_pool= 11  R_max=1.000
    val_006: pool=46925  gold= 18  in_pool= 17  R_max=0.944
    val_007: pool=40953  gold= 19  in_pool= 17  R_max=0.895
    val_008: pool=45818  gold= 30  in_pool= 25  R_max=0.833
    val_009: pool=40751  gold= 14  in_pool= 13  R_max=0.929
    val_010: pool=43903  gold= 25  in_pool= 22  R_max=0.880
[warm-boot] done in 10.2s
```

### Phase 2 output (helper smoke test)
```
[helpers] smoke test
  canon('Art. 221 Abs. 1 lit. b StPO') = '221 StPO'
  canon('Art. 100 BGG')                = '100 BGG'
  doctrinal('Nach Art. 221 StPO bestimmt ...') = 7.5
  doctrinal('Demnach erkennt das BGer: ...')   = -5.0
  val_001 legal_areas=['criminal'] chambers=['1B', '6B', '7B', 'IV']
  val_002 legal_areas=['social'] chambers=['8C', '9C', 'U']
  val_003 legal_areas=['criminal'] chambers=['1B', '6B', '7B', 'IV']
```

### Phase 3 output (Dossier features)
```
[3-cache] HIT  /content/drive/MyDrive/swiss_law/research/hybrid_rerank_final/cache/dossier_features.npz (4241.2 KB)
[3-cache] loaded DOSSIER for 10 queries — skipping Phase 3 compute

[3] DOSSIER ready in 2.3s

  Sample (val_001) — first 3 pool docs, features:
      0: law:128343 (law)
         statute_int = 0.0, lead_statute_int = 0.0, concept_int = 0.0, term_int = 0.0,
         co_cite = 0.0, doctrinal = 0.5, chamber_match = 1.0, hard_neg = 0.0, fusion_rank = 1.0
      1: law:57057 (law)
         statute_int = 0.0, lead_statute_int = 0.0, concept_int = 0.0, term_int = 0.0,
         co_cite = 0.0, doctrinal = -2.5, chamber_match = 1.0, hard_neg = 1.0, fusion_rank = 2.0
      2: law:57137 (law)
         statute_int = 0.0, lead_statute_int = 0.0, concept_int = 1.0, term_int = 0.0,
         co_cite = 0.0, doctrinal = 0.5, chamber_match = 1.0, hard_neg = 0.0, fusion_rank = 3.0
```

### Phase 4 output (Config)
```
[config] RERANK_TOP_N=50000  K_REPORT=(50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000)  floor=0.8
[config] OUT_DIR=/content/drive/MyDrive/swiss_law/research/hybrid_rerank_final
[config] CACHE_DIR=/content/drive/MyDrive/swiss_law/research/hybrid_rerank_final/cache
```

### Phase 5 output (Qwen3-Reranker-8B) — FAILED
Cache MISS. vLLM engine load progressed through:
- weights download (15.25 GiB; 39.66s)
- `Model loading took 15.26 GiB memory and 42.16s`
- `torch.compile took 18.32 s in total`
- `Initial profiling/warmup run took 1.57 s`

Then `EngineCore failed to start` during `determine_available_memory` → `profile_run` → `_dummy_sampler_run` → `flashinfer.sampling.top_k_top_p_sampling_from_logits` → `gen_sampling_module().build_and_load()` → `check_cuda_arch()`:

```
RuntimeError: FlashInfer requires GPUs with sm75 or higher
```

Preceded by warnings:
```
WARNING:flashinfer.compilation_context:Failed to get device capability: SM 12.x requires CUDA >= 12.9.
```

Phase 5 did not produce `scores_qwen3.npz`. Cells 14 (BGE), 16 (jina), 18 (RRF combination), and 20 (summary) have **no captured outputs** in the saved notebook — execution halted at Phase 5.

### Phases 6 / 7 / 8 / 9 — Not executed
No output cells captured beyond the Qwen3 traceback. No `scores_bge.npz`, `scores_jina.npz`, `hybrid_rerank_results.json` written from this run.

## Summary
The notebook implements a definitive diagnostic to find the smallest pool size K at which macro and min-per-query recall both clear 0.80, comparing five configurations (F0 fusion baseline, F1 Qwen3 alone, F2 3-reranker RRF ensemble, F3 hybrid with 9 dossier signals, F4 hybrid with hard-neg filter) over the v7.5 50k-candidate pool across 10 val queries. Phases 0–4 succeeded: snapshot of 255,713 docs loaded, dossier feature cache loaded (4.2 MB, 10 queries), helpers validated, R_max-in-pool established per query (range 0.766–1.000, with val_003 binding the achievable floor at 0.766). The Qwen3-Reranker-8B reranking phase aborted with `RuntimeError: FlashInfer requires GPUs with sm75 or higher`, traced to FlashInfer's CUDA-arch check during vLLM's dummy sampler run (companion warning indicated SM 12.x needs CUDA ≥ 12.9). Consequently no reranker scores, no RRF fusion, no R@K tables, and no `hybrid_rerank_results.json` were produced — the entire shootout (F1–F4 vs F0) is unmeasured. Lesson: the FlashInfer install path in Phase 0 selected a CUDA suffix incompatible with the Blackwell SM 12.x device + installed CUDA toolkit; the vLLM sampler needs to either disable FlashInfer top-k/top-p sampling or run on CUDA ≥ 12.9 before this notebook can produce its intended diagnostic.
