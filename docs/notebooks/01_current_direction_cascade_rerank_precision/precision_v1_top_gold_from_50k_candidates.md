# precision_v1_top_gold_from_50k_candidates.ipynb

**Path:** `notebooks/01_current_direction_cascade_rerank_precision/precision_v1_top_gold_from_50k_candidates.ipynb`

A precision-driven 5-stage cascade (A → B → C → D → E → F) that consumes the v7.5 top-50k snapshot and compresses it to a *variable-size* gold-citation set per query, driven by LLM conviction + query-aspect coverage rather than a fixed top-K.

## Configuration

### Models
- **LLM (Stages C, D, F):** `Qwen/Qwen3-32B-AWQ` (loaded via vLLM, `quantization="awq_marlin"`).
- **Reranker / embedder:** none (this notebook does not re-encode; it reuses snapshot fusion order).

### Libraries
- `vllm>=0.10.0`, `transformers>=4.51.0`, `accelerate`, `safetensors`, `huggingface_hub`.
- `flashinfer-python==0.6.8.post1` (installed but **disabled at runtime** — no SM 12.x kernels in 0.6.x; vLLM falls back to FlashAttention 4).
- `pandas`, `json`, `gzip`, `re`, `collections.Counter/defaultdict`.

### Hardware (observed at runtime)
- GPU: **NVIDIA RTX PRO 6000 Blackwell Server Edition**, compute capability 12.0 (SM 120), VRAM 94.97 GB.
- Environment detected as `colab (Kaggle)` (Drive mounted).

### Key vLLM/LLM constants
- `MAX_MODEL_LEN = 4096`
- `MAX_NUM_SEQS = 256`
- `GPU_MEM_UTIL = 0.90`
- `LLM_TEMPERATURE = 0.1`, `LLM_TOP_P = 0.9`, `LLM_REPETITION_PENALTY = 1.05`
- `enforce_eager=False`, `enable_prefix_caching=True`, `disable_custom_all_reduce=True`
- System+template prefix ≈ 28 tokens (prefix-cached).
- Warm-up: 16 short prompts in 26.33 s after a 108.6 s load.

### Pipeline / cascade constants
- `DOSSIER_TOP_N = 50000` (full snapshot pool consumed)
- `STAGE_B_TOP_N = 2000` (was 300; uses RRF fusion order, not dossier composite)
- `STAGE_C_BATCH = 12`, `STAGE_C_KEEP_THRESHOLD = 3`, `STAGE_C_MAX_NEW_TOKENS = 700`
- `STAGE_D_BATCH = 5`, `STAGE_D_FROM_C_MIN_SCORE = 3`, `STAGE_D_MAX_CANDIDATES = 60`, `STAGE_D_MAX_NEW_TOKENS = 1500`
- Stage E: no top-K cap; tiered by verdict (KEEP@5 → KEEP@4 → aspect-gap-fill MAYBE → case dedup).
- Stage F: 1 LLM call per query, `max_new_tokens=800`.

### Env vars
- `VLLM_WORKER_MULTIPROC_METHOD = "spawn"`
- `TOKENIZERS_PARALLELISM = "false"`
- `FLASHINFER_DISABLE_VERSION_CHECK = "1"`
- `VLLM_USE_FLASHINFER_SAMPLER = "1"` (set conditionally; effectively disabled since `FLASHINFER_USABLE = False` on SM 120)

## Data

All paths quoted as in the notebook.

- **Drive root resolver:** `_DRIVE = Path("/content/drive/MyDrive/swiss_law")` on Colab; `Path(r"E:\swiss_citation_extraction")` locally.
- **Snapshot dir:** `SNAPSHOT_DIR = _DRIVE / "research" / "anchor_funnel_val001_v7" / "snapshot"`
  Resolved at runtime to `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot`.
- **Output dir:** `OUT_DIR = _DRIVE / "research" / "precision_v1"`
  Resolved to `/content/drive/MyDrive/swiss_law/research/precision_v1`.

Snapshot files read:
- `config.json` (`topk_final = 50000` observed)
- `paths.json` (provides `val_csv` path)
- `corpus_snapshot.json.gz` — 255,713 unique docs; per-doc fields `ct` (search_text), `cit` (citation), `fam`, `cb` (court_base), `pr` (paragraph_role), `ln` (language), `sa` (statute anchors set), `cn` (concepts), `tm` (terms)
- `per_query_snapshot.json` — `final_topk`, `curve`, `channel_recalls`, `gold`, `gold_doc_ids`, `R_at_K`
- `all_targets.json` (LLM-expanded statute/concept/term targets per query)
- `hyde_aspects.json`
- `gold_doc_sets.json`

Val CSV resolved at runtime via `paths.json` → falls back to `_DRIVE / "data" / "val.csv"`. 10 queries loaded.

Outputs written:
- `OUT_DIR/precision_v1_summary.json` (full per-query state + macro/micro P/R/F1, fusion-baseline comparison)
- `OUT_DIR/precision_v1_predictions.json` (Kaggle-style citation strings per query)

## Pipeline

Architecture as documented in cell 0 (markdown):
```
50k per query (R@50k ≈ 0.901)
  → Stage A: cheap rule filter (no LLM)         → ~5k-20k
  → Stage B: dossier composite / fusion order   → top-2000
  → Stage C: listwise LLM with dossier (Qwen3-32B, 12/batch, score 1-5, keep ≥3) → ~100-200
  → Stage D: pointwise judgment + EVIDENCE QUOTE (Qwen3-32B, 5/batch)            → KEEP/MAYBE/DROP
  → Stage E: adaptive selection (KEEP@5 → KEEP@4 → aspect-gap MAYBE → case dedup) [no LLM]
  → Stage F: self-validation pass (Qwen3-32B, 1 call/query)
  → variable-K final set
```

### Phase 0 — Setup (cells 1-3)
- Imports, env detection, Drive mount, snapshot/output paths.
- One-time install: vLLM + FlashInfer (CUDA-version-aware index URL: `cu{maj}{min}` ∈ `{cu126, cu128, cu129, cu130, cu131}`). Forces runtime restart if first install.
- Cell 3 separately imports `flashinfer` (logs SM 12.x kernel-missing warning).

### Phase 1 — Warm-Boot from snapshot (cell 5)
- Loads `CONFIG`, `ALL_QUERIES`, `search_text`, `doc_meta`, `doc_statute_anchors`, `_doc_to_concepts`, `_doc_to_terms`, `PER_QUERY`, `ALL_TARGETS`, `ALL_HYDE_ASPECTS`, `ALL_GOLD_DOC_SET`, `ALL_TOTAL_GOLD` in 5.5 s.

### Phase 2 — Dossier
**2a. Helpers (cell 7).** Statute-anchor canonicalizer (mirrors v7.5 cell 12: `CODE_ALIAS = {CPP→StPO, CP→StGB, CC→ZGB, CO→OR, LTF→BGG, LACI→AVIG, LAA→UVG, LP→SchKG, LDIP→IPRG, Cst→BV, ...}`), chamber regex for BGE & BGR citations, `CHAMBER_TO_AREA` mapping (1B→criminal_procedure, 4A→civil, 5A→civil_family_succession, 6B→criminal, 8C/9C→social, etc.), `AREA_KEYWORDS` for query-side legal-area inference, multilingual doctrinal-density regex (rule-openers, doctrine phrases, internal cites, hard-negatives — DE/FR/IT), `SUBSTANTIVE_ROLES = {legal_standard, reasoning, application, holding}`, `NOISE_ROLES = {notification, costs, dispositif}`.

**2b. Dossier compute (cell 9).** Per-doc cache built in 5.3 s (chamber, chamber_class, doctrinal density, hard-negative flag, is_federal, lead-anchor subset). Per-(qid, did) cross features (stat_overlap_full/lead, conc_overlap, term_overlap by lang, best_aspect, aspect_score, n_aspects_addressed, chamber_match, co_cite_count via law-citation-in-pool counter, case_peer_count, case_peer_rule, doctrinal, is_dispositif). Stored in `PER_QUERY[qid]["dossier"]`. 25.4 s total.

### Stage A — Cheap rule prune (cell 11)
Drops: cantonal courts (`family=="court" and not is_federal`), paragraph_role ∈ `NOISE_ROLES`, `is_dispositif` regex hit. **The “no_signal” rule from earlier versions has been removed** with a comment that some gold (law-side bridge articles) has zero direct signal.

### Stage B — Fusion-rank top-2000 (cell 13)
**Trusts the v7.5 RRF fusion order** in `PER_QUERY[qid]["final_topk"]`; walks that order keeping first `STAGE_B_TOP_N` candidates that survived Stage A. The composite score (lead-stat ×8 + non-lead ×3 + concept ×2 + term ×1 + chamber_match +1.5 + substantive_role +1 + doctrinal ×0.5 + aspect_score ×0.1 + co_cite ×0.3 + case_peer_rule +0.5) is computed but **only retained for Stage E tiebreaks**. Comment notes earlier dossier-ranked R@300 was 0.013 — fusion order is what works.

### Phase 3 — LLM Cascade
**3a. Load Qwen3-32B-AWQ (cell 16).** SM-aware vLLM init (avoids `torch.cuda.*` before vLLM loads; forces FLASHINFER off on SM 12.x). Single load attempt with runtime-restart guidance on failure. Generation helpers: `_render_prompt`, `_generate_safe` (recursive batch split on OOM/validation errors), `llm_generate`, `llm_generate_batch`.

**Stage C — Listwise scoring (cell 18).** System prompt teaches Swiss code aliases, chamber→area, role-noise rules, instructs the model to trust dossier signals and verify against text. Per batch of 12: dossier signals + 400-char text excerpt; LLM outputs `[{id, score 1-5, addresses_aspect, why}, ...]`. Keep score ≥ 3.

**Stage D — Pointwise judgment with verbatim evidence quote (cell 20).** Anti-hallucination gate. LLM must output `evidence_quote` that is a verbatim whitespace-normalized substring of the doc text; verdict is **computed deterministically** from `(confidence, quote_valid)`: `≥4 + quote → KEEP`, `==3 + quote → MAYBE`, otherwise `DROP`. Cap of 60 survivors per query, sorted by (-Stage-C-score, -Stage-B-composite).

### Phase 4 — Final selection
**Stage E — Adaptive K (cell 23).** No LLM. Tiers: all KEEP@5 → all KEEP@4 → for each uncovered aspect promote the highest-confidence MAYBE addressing it. Then case-level dedup keyed on `court_base`: per case, keep best by (confidence, composite score); laws are kept individually; multiple paragraphs of the same case retained iff they address different aspects.

**Stage F — Self-validation (cell 25).** 1 LLM call per query. Presents the Stage-E set; LLM marks each item `{necessary: true|false, reason: ≤15 words}`. Rules in the system prompt: BGE > parallel unpublished, statute > case that only cites it, two paragraphs of same case redundant unless different aspects, "when in doubt KEEP". **Safety net:** never drop the sole candidate for an aspect — `rescued` list re-adds them.

### Phase 5 — Eval & save (cell 27)
Per-query and macro/micro P/R/F1 table. "CASCADE vs FUSION-BASELINE F1" table at the same K per query (fusion R taken from `PER_QUERY[qid]["curve"]`). Writes `precision_v1_summary.json` (state + metrics + Stage-B R@top500 snapshot per query) and `precision_v1_predictions.json` (citation strings).

## Results

### Warm-boot & dossier
- Snapshot load: 5.5 s. `corpus_snapshot` = 255,713 unique docs. `ALL_GOLD_DOC_SET` total = 254 doc_ids across 10 queries.
- Sanity sample for val_001: top-2 docs are `law:128343` (`Art. 10 Abs. 1 StGB`) and `law:57057` (`Art. 66 Abs. 1 BGG`).
- Dossier cell helpers sample: `doctrinal_density` DE rule-recital = 0.85, DE facts paragraph = 0.0.

### Stage A — Cheap rule prune (per-query, verbatim)
```
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
Zero gold loss across all 10 queries.

### Stage B — Fusion-order top-2000 (verbatim)
```
[val_001]  fusion=44621  kept=40809  top-2000  gold  16/42   R=0.381
        gold fusion ranks: min=58    median=2438   max=41138  in top-2000=16/42
[val_002]  fusion=41254  kept=29418  top-2000  gold  22/38   R=0.579
        gold fusion ranks: min=2     median=355    max=37532  in top-2000=21/38
[val_003]  fusion=44033  kept=40198  top-2000  gold  14/47   R=0.298
        gold fusion ranks: min=37    median=3682   max=41783  in top-2000=12/47
[val_004]  fusion=43086  kept=37644  top-2000  gold   9/10   R=0.900
        gold fusion ranks: min=58    median=427    max= 7617  in top-2000=9/10
[val_005]  fusion=42140  kept=37406  top-2000  gold   9/11   R=0.818
        gold fusion ranks: min=49    median=182    max=11045  in top-2000=9/11
[val_006]  fusion=46925  kept=40168  top-2000  gold  13/18   R=0.722
        gold fusion ranks: min=30    median=366    max=27709  in top-2000=13/18
[val_007]  fusion=40953  kept=36807  top-2000  gold  13/19   R=0.684
        gold fusion ranks: min=36    median=750    max= 9204  in top-2000=13/19
[val_008]  fusion=45818  kept=41157  top-2000  gold  17/30   R=0.567
        gold fusion ranks: min=86    median=790    max=24541  in top-2000=17/30
[val_009]  fusion=40751  kept=35828  top-2000  gold  10/14   R=0.714
        gold fusion ranks: min=58    median=245    max=26647  in top-2000=10/14
[val_010]  fusion=43903  kept=37228  top-2000  gold  13/25   R=0.520
        gold fusion ranks: min=163   median=1758   max=20623  in top-2000=13/25

[Stage B] macro mean R@2000: 0.618
          (fusion-baseline R@5000 = 0.728; R@1000 = 0.530; R@500 = 0.415)
```

### Dossier summary table (verbatim)
```
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
(Note: no candidates flagged as `is_dispositif` — `disp` column is all zero — meaning Stage A's regex hard-negative rule found no hits, and the role-noise drop alone is what reduces the pool.)

### LLM load (cell 16)
- Load time: **108.6 s**. After load: free 10.43 GiB / 94.97 GiB.
- Warm-up (16 prompts): 26.33 s; sample output `'probe-0'`.
- FlashInfer disabled (SM 120 has no kernels in 0.6.x); vLLM auto-selects FlashAttention 4.

### Stage C — FAILED (cell 18 traceback)
The cell crashed with a context-length validation error:

```
[llm] batch of 167 failed (VLLMValidationError); split 83+84
[llm] batch of 83 failed (VLLMValidationError); split 41+42
[llm] batch of 41 failed (VLLMValidationError); split 20+21
[llm] batch of 21 failed (VLLMValidationError); split 10+11
[llm] batch of 10 failed (VLLMValidationError); split 5+5
[llm] batch of 5 failed (VLLMValidationError); split 2+3
[llm] batch of 3 failed (VLLMValidationError); split 1+2
```
```
VLLMValidationError: This model's maximum context length is 4096 tokens.
However, you requested 0 output tokens and your prompt contains at least 4097 input tokens,
for a total of at least 4097 tokens. Please reduce the length of the input prompt or
the number of requested output tokens. (parameter=input_tokens, value=4097)
```

Root cause: `MAX_MODEL_LEN = 4096` is too small for a 12-candidate batch with full dossier blocks + 400-char text excerpts + system prompt. The recursive `_generate_safe` split-on-failure helper repeatedly halved batches down to size 1 but still hit the limit because a single prompt was already >4096 tokens. The cell raised before producing any Stage-C scores.

### Stages D, E, F, eval/save — NOT EXECUTED
Cells 20, 23, 25, and 27 produced **no outputs** (no streams, no errors) because Stage C halted execution. No per-query F1, no macro/micro numbers, no `precision_v1_summary.json` or `precision_v1_predictions.json` were written during this run.

## Summary

The notebook implements a 6-stage precision cascade (A: rule prune, B: fusion-order top-2000, C: listwise Qwen3-32B with dossier, D: pointwise verbatim-evidence verdict, E: adaptive K + aspect coverage + case dedup, F: self-validation) on top of the v7.5 multi-query 50k snapshot, targeting variable-K precision rather than a fixed top-K. Stage A and Stage B ran cleanly (Stage A: zero gold loss across all 10 queries with 71-92% pool reduction; Stage B macro mean R@2000 = 0.618). The vLLM Qwen3-32B-AWQ load succeeded (108.6 s, FlashInfer correctly disabled on SM 120). **Stage C failed**: `MAX_MODEL_LEN=4096` is incompatible with the 12-candidate listwise prompt size — even single-candidate splits exceed 4097 tokens, so the recursive `_generate_safe` retry loop terminated with `VLLMValidationError` and Stages D/E/F/eval never ran. The next iteration must either (a) raise `MAX_MODEL_LEN` (e.g., 8192/16384), (b) shrink Stage C prompts (smaller text excerpt, more compact dossier rendering, smaller batch), or (c) both — and then re-test whether the precision cascade actually beats the fusion-baseline F1 (the comparison block in cell 27 was the intended falsifiable test, still unanswered).
