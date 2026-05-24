# precision_v1_final.ipynb

**Path:** `notebooks/01_current_direction_cascade_rerank_precision/precision_v1_final.ipynb`

## Configuration

### Hardware / Environment
- Detected GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`, compute capability 12.0 (SM 120).
- Free VRAM after vLLM load: `10.43 GiB / 94.97 GiB`.
- Environment: Colab (Kaggle flag detected). Drive mounted at `/content/drive`.
- Python `3.12`.
- `flashinfer 0.6.8.post1` importable but disabled for SM 12.x (no kernels in 0.6.x); vLLM auto-selects FlashAttention 4.

### Libraries / Install
- `vllm>=0.10.0`, `transformers>=4.51.0`, `accelerate`, `safetensors`, `huggingface_hub`.
- `flashinfer-python`, `flashinfer-cubin`, `flashinfer-jit-cache` (CUDA-version-aware index URL).
- `os.environ["TOKENIZERS_PARALLELISM"] = "false"`, `os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"`, `os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"`.

### LLM (Stages C, D, F)
- Model: `Qwen/Qwen3-32B-AWQ` via vLLM.
- `MAX_MODEL_LEN = 4096`, `MAX_NUM_SEQS = 256`, `GPU_MEM_UTIL = 0.90`.
- `quantization = "awq_marlin"`, `enable_prefix_caching = True`, `enforce_eager = False`, `disable_custom_all_reduce = True`, `tensor_parallel_size = 1`.
- Sampling: `LLM_TEMPERATURE = 0.1`, `LLM_TOP_P = 0.9`, `LLM_REPETITION_PENALTY = 1.05`.
- System+template prefix ~28 tokens (prefix-cached).
- Load time: 108.6 s; warm-up 16 short prompts: 26.33 s.
- Recursive split-on-failure batch generation wrapper (`_generate_safe`) halves the batch on OOM.

### Stage hyperparameters
- `DOSSIER_TOP_N = 50000` (full snapshot pool per query).
- Stage A: drops `cantonal_court`, `paragraph_role ∈ {notification, costs, dispositif}`, `is_dispositif` regex. `SUBSTANTIVE_ROLES = {legal_standard, reasoning, application, holding}`, `NOISE_ROLES = {notification, costs, dispositif}`.
- Stage B: `STAGE_B_TOP_N = 2000`. Walks v7.5 RRF fusion order; composite score retained only for downstream tiebreaks. Composite weights: lead-statute ×8, full-only statute ×3, concept ×2, term ×1, chamber_match +1.5, substantive role +1, doctrinal ×0.5, `min(aspect_score, 10)` ×0.1, `min(co_cite_count, 20)` ×0.3, case_peer_rule +0.5.
- Stage C: `STAGE_C_BATCH = 10`, `STAGE_C_TOP_K_KEEP = 300`, `STAGE_C_MAX_NEW_TOKENS = 600`, `STAGE_C_TEXT_CHARS = 700`. Dossier-free prompt. Selection by top-K, no threshold.
- Stage D: `STAGE_D_BATCH = 5`, `STAGE_D_MAX_CANDIDATES = 300`, `STAGE_D_MAX_NEW_TOKENS = 1500`. Full 1500-char text. Verdict deterministically derived: `conf≥4 + verbatim quote → KEEP`, `conf=3 + verbatim quote → MAYBE`, else `DROP`. Quote validated by whitespace-normalized substring match.
- Stage E: no LLM. Variable-K (no cap). Tier order: KEEP@5 → KEEP@4 → aspect-gap fill from MAYBE. Case-level dedup per `court_base`, splitting by `addresses_aspect`. Laws kept individually under `__solo__` key.
- Stage F: 1 LLM call per query, `max_new_tokens = 800`. Safety net: never drop the only candidate for an aspect (rescues from `dropped`).

### Domain primitives (helpers)
- `CODE_ALIAS = {CPP→StPO, CP→StGB, CC→ZGB, CO→OR, LTF→BGG, LACI→AVIG, LAA→UVG, LP→SchKG, LDIP→IPRG, Cst→BV, STPO→StPO, OBG→OR}`.
- `CHAMBER_TO_AREA`: maps 20 Swiss chamber codes to coarse legal areas (public_law, social_security, criminal_procedure, civil, civil_family_succession, etc.).
- `AREA_KEYWORDS`: keyword bags per area for `legal_area_for_query` inference.
- Multilingual regex bundles for doctrinal density (`_RULE_OPENER_*`, `_DOCTRINE_*`, `_RULE_VERB_*`, `_HARD_NEG_*`, `_INTERNAL_CITE`, `_STATUTE_REF`).
- `doctrinal_density` baseline 0.30 + opener/doctrine/verb/internal-cite boosts, hard-negative -0.40, ref-density up to +0.15, clipped to [0,1].

## Data

### Input — v7.5 multi-query snapshot (Drive)
- `SNAPSHOT_DIR = /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot`
- Files read:
  - `config.json` (loaded `topk_final = 50000`)
  - `paths.json`
  - `corpus_snapshot.json.gz` (gzipped; `255,713 unique docs`)
  - `per_query_snapshot.json` (`final_topk` per qid + `curve` + `channel_recalls` + `R_at_K`)
  - `all_targets.json` (10 entries — LLM-expanded statute/concept/term targets)
  - `hyde_aspects.json` (10 entries — query sub-issues)
  - `gold_doc_sets.json` (254 doc_ids total across val)

### Input — val set
- `paths.json["val_csv"]` (with local fallback to `_DRIVE / "data" / "val.csv"`).
- 10 query rows: `query_id`, `query`, `gold_citations` (split on `;`).

### Output — `OUT_DIR = /content/drive/MyDrive/swiss_law/research/precision_v1`
- `precision_v1_summary.json` — macro/micro P/R/F1, per-query metrics, selection, verdicts, Stage-E promotions, Stage-F drops, Stage-B R@2000.
- `precision_v1_predictions.json` — `{qid: [citation strings...]}` (Kaggle-style).

### Local-mode fallback path
- `_DRIVE = Path(r"E:\swiss_citation_extraction")` if not in Colab.

## Pipeline

### Phase 0 — Setup (cells 2-3)
- Detects environment (Colab vs local); mounts Drive if Colab.
- Installs vLLM + FlashInfer if missing; signals runtime restart on first install.
- Imports `flashinfer` and prints version (`flashinfer 0.6.8.post1 OK`).

### Phase 1 — Warm-Boot from snapshot (cell 5)
- Loads `CONFIG`, `ALL_QUERIES` (10), `ALL_TOTAL_GOLD`.
- Builds `search_text`, `doc_meta` (citation/family/court_base/paragraph_role/language), `doc_statute_anchors`, `_doc_to_concepts`, `_doc_to_terms` from `corpus_snapshot.json.gz`.
- Loads `PER_QUERY` (final_topk + curve + channel_recalls + R_at_K), `ALL_TARGETS`, `ALL_HYDE_ASPECTS`, `ALL_GOLD_DOC_SET`.

### Phase 2a — Dossier helpers (cell 7)
- Statute-anchor canonicalizer (`"Art. 41 OR" → "41 OR"`) — mirrors v7.5 cell 12.
- Chamber detection from BGE/BGR citations.
- Legal-area inference from query expansion blob (NOT per-query hardcoded).
- Doctrinal density + dispositif/facts detector (DE/FR/IT).

### Phase 2b — Dossier compute (cell 9)
- Per-doc cache: chamber, chamber_class, is_federal, doctrinal density, hard-negative flag, lead anchors (statutes in first 200 chars).
- Per-(qid, did) cross features:
  - `stat_overlap_full`, `stat_overlap_lead` (∩ canonicalized targets).
  - `conc_overlap`, `term_overlap` (language-specific term targets).
  - `aspect_score`, `best_aspect`, `n_aspects_addressed` (token-bag overlap on first 1500 chars + anchors + concepts).
  - `chamber_match` (chamber_class == legal_area).
  - `co_cite_count` (court paras in pool whose anchors cite this law) and `case_peer_count` / `case_peer_rule` (court_base peers).

### Stage A — Cheap rule filter (cell 11)
- Drops: cantonal court (non-federal), `paragraph_role ∈ NOISE_ROLES`, dispositif regex hits.
- The old "no_signal" rule was removed because some law-side bridge gold has zero direct signal.

### Stage B — Fusion-rank ranking (cell 13)
- Trusts the v7.5 RRF order from `PER_QUERY[qid]["final_topk"]`. Walks fusion order, keeps the first 2000 that survived Stage A.
- Composite score still computed (used only by Stage E for tiebreaks; not for reorder).

### Phase 3a — Load Qwen3-32B (cell 16)
- SM-12.x-aware vLLM load. nvidia-smi probe (no parent-process CUDA init). FlashInfer forced off on SM 12.x. Single-try load; on fail, advises restart.
- Imports `torch` only after load. Defines `_render_prompt`, `_generate_safe`, `llm_generate`, `llm_generate_batch`. Warms up with 16 short prompts.

### Stage C — Listwise LLM scoring (cell 18)
- Dossier-free prompt (the previous dossier-enriched prompt biased scores ≤ 2 for ~67 % of val gold without statute/concept overlap).
- 1-5 score + `addresses_aspect` + `≤20-word why`. Top-300 by (score desc, stage_b composite tiebreak); no threshold cutoff.

### Stage D — Pointwise judgment with evidence quote (cell 20)
- Batch of 5 candidates with full dossier signals + 1500-char text.
- Output: `topic`, `relation`, `evidence_quote` (verbatim substring), `confidence`, `addresses_aspect`, `verdict`.
- Quote is whitespace-normalized and substring-checked against `search_text[did]`. Invalid quote → DROP.

### Stage E — Adaptive selection (cell 23)
- All KEEP@5 + all KEEP@4 → aspect-gap fill from MAYBE → case-level dedup per `court_base` (split by `addresses_aspect`). Laws are kept individually.

### Stage F — Self-validation pass (cell 25)
- One LLM call per query. Presents the Stage-E set; flags items as `necessary: false` only when another item covers the same legal point with equal/greater authority.
- Safety net rescues any drop that was the sole candidate for an unaddressed aspect.

### Phase 5 — Eval + save (cell 27)
- Per-query and macro/micro P/R/F1.
- Comparison vs fusion baseline at the same K per query (fusion TP = `round(curve[K][1] * n_gold)`).
- Writes `precision_v1_summary.json` and `precision_v1_predictions.json`.

## Results

### Phase 0 — Setup
```
Python 3.12
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
Environment: colab (Kaggle)
snapshot: /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot
out:      /content/drive/MyDrive/swiss_law/research/precision_v1

[setup] vLLM + FlashInfer already installed; proceeding.
```
```
WARNING:flashinfer.compilation_context:Failed to get device capability: SM 12.x requires CUDA >= 12.9.
WARNING:flashinfer.compilation_context:Failed to get device capability: SM 12.x requires CUDA >= 12.9.
  flashinfer 0.6.8.post1 OK
```

### Phase 1 — Warm-Boot
```
[warm-boot] loading from /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot
  CONFIG loaded (topk_final=50000)
  ALL_QUERIES: 10  (val.csv)
  corpus_snapshot: 255,713 unique docs (cascade pool)
  PER_QUERY: 10
  ALL_TARGETS: 10
  ALL_HYDE_ASPECTS: 10
  ALL_GOLD_DOC_SET total: 254 doc_ids

  sanity check (val_001, top-2 = ['law:128343', 'law:57057']):
    law:128343: 'Art. 10 Abs. 1 StGB This article distinguishes between crimes and offenses based on the severity of penalties threatened. criminal acts pena'
    law:57057: 'Art. 66 Abs. 1 BGG The article states that court costs are generally imposed on the losing party, but the Federal Court may distribute them '

[warm-boot] done in 9.3s
```

### Phase 2a — Helpers
```
[helpers] chamber_for_citation('BGE 142 III 296') = III
[helpers] chamber_for_citation('1B_490/2017')      = 1B
[helpers] chamber_class('1B') = criminal_procedure
[helpers] doctrinal_density sample DE: 0.85
[helpers] doctrinal_density sample DE facts: 0.0
```

### Phase 2b — Dossier compute
```
[dossier] computing per-doc cache for 255713 unique docs...
  per-doc cache built in 5.4s
[dossier] computing per-(qid, did) cross features...
[dossier] complete for 10 queries in 25.4s

qid          pool  avg_stat  avg_lead  ch_match    disp  fed_court  legal_area
val_001     44621      0.20      0.01     10640       0      38350  criminal_procedure
val_002     41254      0.08      0.00       326       0      27087  social_security
val_003     44033      0.20      0.01     10798       0      37623  criminal_procedure
val_004     43086      0.04      0.00     13341       0      34932  civil_family_succession
val_005     42140      0.00      0.00      3298       0      33939  criminal
val_006     46925      0.18      0.02      8772       0      37660  civil
val_007     40953      0.02      0.00     11774       0      34322  civil_family_succession
val_008     45818      0.13      0.01     14872       0      37891  criminal
val_009     40751      0.03      0.00      6244       0      33530  civil
val_010     43903      0.11      0.00     16006       0      36480  civil
```

### Stage A — Cheap rule filter
```
[Stage A] applying cheap filters...
  [val_001] 44621 → 40809 ( 91.5%)  gold 39→39 (100.0%)  drops: {'cantonal_court': 2421, 'noise_role': 1391}
  [val_002] 41254 → 29418 ( 71.3%)  gold 30→30 (100.0%)  drops: {'cantonal_court': 10074, 'noise_role': 1762}
  [val_003] 44033 → 40198 ( 91.3%)  gold 36→36 (100.0%)  drops: {'cantonal_court': 2430, 'noise_role': 1405}
  [val_004] 43086 → 37644 ( 87.4%)  gold 10→10 (100.0%)  drops: {'cantonal_court': 4178, 'noise_role': 1264}
  [val_005] 42140 → 37406 ( 88.8%)  gold 11→11 (100.0%)  drops: {'cantonal_court': 3595, 'noise_role': 1139}
  [val_006] 46925 → 40168 ( 85.6%)  gold 17→17 (100.0%)  drops: {'cantonal_court': 5402, 'noise_role': 1355}
  [val_007] 40953 → 36807 ( 89.9%)  gold 17→17 (100.0%)  drops: {'cantonal_court': 2761, 'noise_role': 1385}
  [val_008] 45818 → 41157 ( 89.8%)  gold 25→25 (100.0%)  drops: {'cantonal_court': 3515, 'noise_role': 1146}
  [val_009] 40751 → 35828 ( 87.9%)  gold 13→13 (100.0%)  drops: {'cantonal_court': 3164, 'noise_role': 1759}
  [val_010] 43903 → 37228 ( 84.8%)  gold 22→22 (100.0%)  drops: {'cantonal_court': 3618, 'noise_role': 3057}
```
- Stage A retained 100 % of in-pool gold on every query while dropping 8.5-28.7 % of the pool.

### Stage B — Fusion-rank ranking
```
[Stage B] fusion-rank ranking + top-2000 per query
  Trusting the v7.5 RRF order from PER_QUERY[qid]['final_topk'].
  Dossier score still computed for downstream display/tiebreaks.

  [val_001]  fusion=44621  kept=40809  top-2000  gold  16/42   R=0.381
         gold fusion ranks: min=   58  median= 2438  max=41138  in top-2000=16/42
  [val_002]  fusion=41254  kept=29418  top-2000  gold  22/38   R=0.579
         gold fusion ranks: min=    2  median=  355  max=37532  in top-2000=21/38
  [val_003]  fusion=44033  kept=40198  top-2000  gold  14/47   R=0.298
         gold fusion ranks: min=   37  median= 3682  max=41783  in top-2000=12/47
  [val_004]  fusion=43086  kept=37644  top-2000  gold   9/10   R=0.900
         gold fusion ranks: min=   58  median=  427  max= 7617  in top-2000=9/10
  [val_005]  fusion=42140  kept=37406  top-2000  gold   9/11   R=0.818
         gold fusion ranks: min=   49  median=  182  max=11045  in top-2000=9/11
  [val_006]  fusion=46925  kept=40168  top-2000  gold  13/18   R=0.722
         gold fusion ranks: min=   30  median=  366  max=27709  in top-2000=13/18
  [val_007]  fusion=40953  kept=36807  top-2000  gold  13/19   R=0.684
         gold fusion ranks: min=   36  median=  750  max= 9204  in top-2000=13/19
  [val_008]  fusion=45818  kept=41157  top-2000  gold  17/30   R=0.567
         gold fusion ranks: min=   86  median=  790  max=24541  in top-2000=17/30
  [val_009]  fusion=40751  kept=35828  top-2000  gold  10/14   R=0.714
         gold fusion ranks: min=   58  median=  245  max=26647  in top-2000=10/14
  [val_010]  fusion=43903  kept=37228  top-2000  gold  13/25   R=0.520
         gold fusion ranks: min=  163  median= 1758  max=20623  in top-2000=13/25

[Stage B] macro mean R@2000: 0.618
          (fusion-baseline R@5000 = 0.728; R@1000 = 0.530; R@500 = 0.415)
```

### Phase 3a — vLLM load
```
[llm] GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition  (compute capability 12.0 = SM 120)
[llm] FlashInfer 0.6.8.post1 importable
[llm] SM 120 → FlashInfer DISABLED (no kernels for SM 12.x in 0.6.x).
[llm]   vLLM will auto-select FlashAttention 4 — equally fast on Blackwell.

[llm] loading Qwen/Qwen3-32B-AWQ via vLLM...
[llm] system+template prefix ≈ 28 tokens (prefix-cached)
[llm] init plan:
  attention backend  : auto-select
  quantization       : awq_marlin
  prefix caching     : on
  CUDA graphs        : on (enforce_eager=False)
  max_model_len      : 4096
  max_num_seqs       : 256
  gpu_mem_util       : 0.9
INFO 05-13 01:47:50 [utils.py:233] non-default args: {'trust_remote_code': True, 'max_model_len': 4096, 'enable_prefix_caching': True, 'gpu_memory_utilization': 0.9, 'max_num_seqs': 256, 'disable_log_stats': True, 'quantization': 'awq_marlin', 'disable_custom_all_reduce': True, 'model': 'Qwen/Qwen3-32B-AWQ'}
INFO 05-13 01:47:51 [model.py:555] Resolved architecture: Qwen3ForCausalLM
INFO 05-13 01:47:51 [model.py:1680] Using max model len 4096
INFO 05-13 01:47:52 [awq_marlin.py:252] The model is convertible to awq_marlin during runtime. Using awq_marlin kernel.
INFO 05-13 01:47:52 [scheduler.py:239] Chunked prefill is enabled with max_num_batched_tokens=16384.
INFO 05-13 01:47:52 [vllm.py:840] Asynchronous scheduling is enabled.
INFO 05-13 01:47:52 [kernel.py:205] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'])

[llm] loaded in 108.6s
  GPU 0: free=10.43 GiB / 94.97 GiB
[llm] warm-up (16 short prompts)...
[llm] warm-up done in 26.33s; sample: 'probe-0'
```

### Stages C, D, E, F, Phase 5 — Eval/save
- All Stage C, D, E, F and final evaluation cells (cells 18, 20, 23, 25, 27) have no captured outputs in this notebook file. No P / R / F1 numbers, no comparison-vs-fusion table, no save-confirmation prints were recorded. The notebook execution evidence stops after Stage B and the LLM load (Phase 3a).

## Summary

The notebook implements a six-stage variable-K precision cascade on the v7.5 multi-query snapshot (`R@50k ≈ 0.893`): cheap rule prune (Stage A) → fusion-rank top-2000 (Stage B) → listwise Qwen3-32B-AWQ score (Stage C, top-300) → pointwise evidence-quote judgment with substring verification (Stage D, KEEP/MAYBE/DROP) → adaptive aspect-coverage + case-level dedup (Stage E) → self-validation pass with aspect-rescue safety net (Stage F). What worked and is captured: warm-boot from snapshot finished in 9.3 s with 255,713 unique docs and all 10 val queries; Stage A retained 100 % of in-pool gold on every query; the SM-12.x-aware vLLM load succeeded in 108.6 s with FlashInfer correctly disabled. The binding finding from captured cells is Stage B's `macro mean R@2000 = 0.618` (per-query 0.298-0.900), meaning the cascade enters the LLM stages with substantial recall loss on val_001 (0.381), val_003 (0.298), and val_010 (0.520) — the v7.5 RRF order alone is not sharp enough at K=2000 to give the LLM the gold for most queries. Stages C through F have empty output cells, so no end-to-end F1, no fusion-baseline comparison, and no saved `precision_v1_summary.json` / `precision_v1_predictions.json` are recorded in this notebook; only the pre-LLM diagnostics are evidence. The lesson surfaced by the recorded output is that "trust the fusion order" at K=2000 caps recall at 0.618 macro before any LLM work begins, which limits the achievable F1 ceiling for the downstream cascade regardless of Stage D/F precision.
