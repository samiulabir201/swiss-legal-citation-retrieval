# precision_v1_not_working_diagnostic.ipynb

**Path:** `notebooks/01_current_direction_cascade_rerank_precision/precision_v1_not_working_diagnostic.ipynb`

## Configuration

- **Goal**: Precision-driven 5-stage cascade ("Precision v1") that reads the v7.5 multi-query snapshot (top-50,000 per query, ~0.90 macro recall) and compresses it to a variable-K per-query gold-citation set driven by LLM conviction + aspect coverage (no fixed top-K).
- **Architecture**: 50k pool → Stage A (cheap rule filter, no LLM) → Stage B (composite/fusion-rank top-N) → Stage C (listwise LLM scoring, Qwen3-32B, 1-5 scale) → Stage D (pointwise verdict with verbatim-quote anti-hallucination gate) → Stage E (adaptive selection + aspect coverage + case-level dedup) → Stage F (self-validation LLM pass) → variable-K final set.
- **Anti-hallucination invariants**: Stage D `evidence_quote` must be a verbatim substring of doc text (regex-checked); verdict deterministically computed from `(confidence, quote_valid)`; case-level dedup; never drop the only candidate for an unaddressed aspect.
- **LLM**: `Qwen/Qwen3-32B-AWQ` via vLLM, `awq_marlin` quantization, `tensor_parallel_size=1`, `max_model_len=4096`, `max_num_seqs=256`, `gpu_memory_utilization=0.90`, `enable_prefix_caching=True`, `enforce_eager=False` (CUDA graphs on), `disable_custom_all_reduce=True`. Sampling: `temperature=0.1`, `top_p=0.9`, `repetition_penalty=1.05`.
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0 (SM 120), 94.97 GiB total VRAM. Colab environment (Drive mounted at `/content/drive`, IN_KAGGLE flag also true).
- **Attention backend**: vLLM auto-select (FlashAttention 4). FlashInfer 0.6.8.post1 importable but explicitly DISABLED on SM 12.x (no kernels for SM 12.x in 0.6.x — `FLASHINFER_USABLE = HAS_FLASHINFER and (GPU_SM // 10 != 12)`).
- **Libraries (install)**: `vllm>=0.10.0`, `transformers>=4.51.0`, `accelerate`, `safetensors`, `huggingface_hub`, `flashinfer-python`, `flashinfer-cubin`, `flashinfer-jit-cache` (CUDA-index-aware). `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `TOKENIZERS_PARALLELISM=false`, `FLASHINFER_DISABLE_VERSION_CHECK=1`.
- **Pool ceiling**: snapshot pool = top-50,000 per query; macro mean R@50000 ≈ 0.901 (described); R@5000 = 0.728; R@2000 ≈ 0.62; R@1000 = 0.530; R@500 = 0.415.
- **Stage hyperparameters**:
  - `DOSSIER_TOP_N = 50000` (full snapshot pool).
  - Stage A: drops cantonal courts, `NOISE_ROLES = {"notification","costs","dispositif"}`, hard-negative regex hits.
  - Stage B: `STAGE_B_TOP_N = 2000` (raised from 300); ranking trusts v7.5 RRF order (composite score retained only for tiebreaks).
  - Stage B composite weights: `8.0 * stat_overlap_lead + 3.0 * (stat_overlap_full - lead) + 2.0 * conc_overlap + 1.0 * term_overlap + 1.5 * chamber_match + 1.0 * (role in SUBSTANTIVE_ROLES) + 0.5 * doctrinal + 0.1 * min(aspect_score,10) + 0.3 * min(co_cite_count,20) + 0.5 * case_peer_rule`.
  - Stage C: `STAGE_C_BATCH=12`, `STAGE_C_KEEP_THRESHOLD=3`, `STAGE_C_MAX_NEW_TOKENS=700`. Pre-tokenizes prompts; budget = `MAX_MODEL_LEN - max_new_tokens - 64 = 3332`; bisects oversized batches recursively; falls back to 120-char text excerpt for single-candidate overflow.
  - Stage D: `STAGE_D_BATCH=5`, `STAGE_D_FROM_C_MIN_SCORE=3`, `STAGE_D_MAX_CANDIDATES=60`, `STAGE_D_MAX_NEW_TOKENS=1500`. Verdict: conf≥4 + verbatim quote → KEEP; conf=3 + quote → MAYBE; else → DROP.
  - Stage E: union(KEEP@5, KEEP@4) → aspect-gap fill from MAYBEs → case-level dedup by `court_base` keyed by addresses_aspect.
  - Stage F: 1 LLM call per query, `max_new_tokens=800`; safety rescue restores items if they are the only coverage of an aspect.
- **Domain primitives** (cell 7):
  - `CODE_ALIAS` map (CPP→StPO, CP→StGB, CC→ZGB, CO→OR, LTF→BGG, etc.).
  - `statute_anchor_canonical()` → canonical form `"41 OR"` (no `Art.` prefix) matching `doc_statute_anchors`.
  - `CHAMBER_TO_AREA` (Federal Court chambers → coarse legal area: 1B=criminal_procedure, 4A=civil, 5A=civil_family_succession, 6B=criminal, 8C/9C=social_unemployment, etc.).
  - `AREA_KEYWORDS` for legal-area inference from query expansion blob.
  - `doctrinal_density()` (DE/FR/IT regex: rule openers, doctrine phrases, rule verbs, internal-cite count, hard-neg penalty).
  - `SUBSTANTIVE_ROLES = {"legal_standard","reasoning","application","holding"}`.
  - `NOISE_ROLES = {"notification","costs","dispositif"}` — intentionally narrow (val gold includes `paragraph_role='facts'` in BGE 148 V 21, BGE 140 V 193, so blanket-dropping `facts`/`procedural_history`/`neutral_default` is forbidden).

## Data

All paths read from cells 2 & 5; quoted verbatim from cell sources/outputs:

- **Snapshot root** (Drive): `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot`
  - `config.json` — `topk_final=50000`
  - `paths.json` — points to `val_csv`
  - `corpus_snapshot.json.gz` — 255,713 unique docs; fields `ct`(text), `cit`(citation), `fam`(family), `cb`(court_base), `pr`(paragraph_role), `ln`(language), `sa`(statute anchors), `cn`(concepts), `tm`(terms)
  - `per_query_snapshot.json` — 10 queries; per-qid `final_topk`, `curve`, `channel_recalls`, `gold`, `gold_doc_ids`, `R_at_K`
  - `all_targets.json` — 10 entries (LLM-extracted statute/concept/term targets per query)
  - `hyde_aspects.json` — 10 entries (HyDE aspect paragraphs per query)
  - `gold_doc_sets.json` — total 254 doc_ids across 10 queries
- **Validation CSV**: `paths.json[val_csv]` → fallback `_DRIVE / "data" / "val.csv"`. Loaded as `val_df` → 10 query rows with fields `query_id`, `query`, `gold_citations` (semicolon-separated).
- **Output directory**: `/content/drive/MyDrive/swiss_law/research/precision_v1` (created in cell 2).
- **Local fallback root**: `E:\swiss_citation_extraction` (only used if `google.colab` import fails).
- **Expected outputs**:
  - `OUT_DIR/precision_v1_summary.json` — full state + per-query P/R/F1 + cascade-vs-fusion comparison.
  - `OUT_DIR/precision_v1_predictions.json` — citation strings per query (Kaggle-style).
- **Models** (Hub): `Qwen/Qwen3-32B-AWQ`.

## Pipeline

### Stage 0 — Setup (cells 1–3)
- Cell 1 (markdown): architecture diagram + anti-hallucination invariants + prerequisites (must have run v7.5 multi-query cell 48 with the `[:50000]` patch).
- Cell 2: imports (`sys, os, json, gzip, time, re, gc, subprocess, importlib, pathlib, collections`); environment detection (`google.colab.drive.mount` or local `E:\swiss_citation_extraction`); asserts `SNAPSHOT_DIR.exists()`; pip-installs vLLM + FlashInfer once (forces a one-shot runtime restart via `SystemExit`).
- Cell 3: `import flashinfer; print(version)` — confirms `flashinfer 0.6.8.post1 OK`, with two warnings `Failed to get device capability: SM 12.x requires CUDA >= 12.9`.

### Phase 1 — Warm-boot (cells 4–5)
- Loads `CONFIG` (topk_final=50000), `val.csv` → `ALL_QUERIES` (10), `corpus_snapshot.json.gz` → `search_text`, `doc_meta`, `doc_statute_anchors`, `_doc_to_concepts`, `_doc_to_terms` (255,713 docs); `per_query_snapshot.json` → `PER_QUERY`; `all_targets.json`, `hyde_aspects.json`, `gold_doc_sets.json` (254 gold doc_ids).
- Sanity print: top-2 docs for `val_001` = `['law:128343', 'law:57057']` (Art. 10 Abs. 1 StGB; Art. 66 Abs. 1 BGG).

### Phase 2 — Dossier (cells 6–9)
- Cell 7 (Phase 2a — helpers): canonicalizer, chamber→area map, area-keyword inference, doctrinal-density regex (DE/FR/IT), `is_federal_court`. Sanity prints confirm `chamber_for_citation('1B_490/2017')='1B'`, `chamber_class('1B')='criminal_procedure'`, doctrinal_density on DE rule sentence = 0.85, on DE Sachverhalt = 0.0.
- Cell 9 (Phase 2b — dossier compute): builds per-doc caches (`_doc_chamber`, `_doc_chamber_cls`, `_doc_doctrinal`, `_doc_hard_neg`, `_doc_is_federal`, `_doc_lead_anchors`), then per-(qid, did) cross features into `PER_QUERY[qid]["dossier"]`. Features: `stat_overlap_full/lead`, `conc_overlap`, `term_overlap`, `best_aspect` + `aspect_score` + `n_aspects_addressed`, `chamber_match`, `co_cite_count`, `case_peer_count` + `case_peer_rule`, `doctrinal`, `is_dispositif`.

### Stage A — cheap rule filter (cells 10–11)
- Drops `family=='court' and not is_federal` (cantonal courts), `paragraph_role in NOISE_ROLES`, `is_dispositif` regex hits. Writes `PER_QUERY[qid]["stage_a_kept"]`. (No-signal rule was removed — bridge-article gold has zero direct signal.)

### Stage B — ranking + top-2000 (cells 12–13)
- Walks fusion order `PER_QUERY[qid]["final_topk"]` and keeps the first `STAGE_B_TOP_N=2000` survivors of Stage A. Composite score still computed (used by Stage E as tiebreak / Stage C/D as display).

### Stage C — listwise LLM scoring (cells 14–18 setup; 17 prompt; 18 run)
- Cell 16 (Phase 3a): SM detection via `nvidia-smi` (no parent-process CUDA init), FlashInfer-disable on SM 12.x, build vLLM kwargs (`awq_marlin`, prefix caching, CUDA graphs), load Qwen3-32B-AWQ. Defines `llm_generate_batch()` + `_generate_safe()` (recursive split on OOM; returns `""` on validation/context-length errors without retry). Warm-up: 16 short prompts, sample output `'probe-0'`.
- Cell 18 (Stage C): builds 12-candidate batches; pre-counts tokens with `llm_tok.apply_chat_template(... enable_thinking=False)`; recursively bisects batches exceeding `PROMPT_TOKEN_BUDGET = 4096 - 700 - 64 = 3332` tokens; falls back to 120-char text for single-candidate overflow. Parses strict JSON array of `{id, score(1-5), addresses_aspect, why}`. Keeps `score >= 3`.

### Stage D — pointwise verdict with evidence quote (cells 19–20)
- For each Stage-C survivor (capped at top-60 per query by `(stage_c_score, stage_b_composite)`): 5-candidate batches; 1500-char text shown; LLM returns `{topic, relation, evidence_quote, confidence, addresses_aspect, verdict}`. `evidence_quote` whitespace-normalized then checked `in` whitespace-normalized text. Verdict deterministic: not-verbatim or conf≤2 → DROP; conf=3 → MAYBE; conf≥4 → KEEP.

### Stage E — adaptive selection (cells 21–23)
- Union(KEEP@5, KEEP@4) sorted by composite tiebreak → aspect-gap fill from MAYBEs (best confidence per uncovered aspect) → case-level dedup keyed by `court_base × addresses_aspect` (laws kept individually).

### Stage F — self-validation (cells 24–25)
- 1 LLM call per query showing the Stage-E set; LLM emits `{id, necessary: bool, reason}` per item. Drop `necessary=false` items, then aspect-safety rescue (restore items that are the only candidate for an aspect).

### Phase 5 — Eval + save (cells 26–27)
- Per-query and aggregate (macro & micro) P / R / F1 vs gold. Cascade vs fusion-baseline F1 at the same K per query (fusion P interpolated from `PER_QUERY[qid]["curve"]`). Writes `precision_v1_summary.json` and `precision_v1_predictions.json` (citation strings).

## Results

Pipeline did NOT run to completion. Only Phases 0–2, Stage A, and Stage B produced saved cell outputs. Stage C was interrupted mid-run on `val_003`. Stages D, E, F, and the eval cell have no outputs in this notebook.

### Phase 0 — Setup (cell 2 output)
```
Python 3.12
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
Environment: colab (Kaggle)
snapshot: /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot
out:      /content/drive/MyDrive/swiss_law/research/precision_v1

[setup] vLLM + FlashInfer already installed; proceeding.
```

### Cell 3 — FlashInfer import
```
WARNING:flashinfer.compilation_context:Failed to get device capability: SM 12.x requires CUDA >= 12.9.
WARNING:flashinfer.compilation_context:Failed to get device capability: SM 12.x requires CUDA >= 12.9.
  flashinfer 0.6.8.post1 OK
```

### Phase 1 — Warm-boot (cell 5 output)
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

[warm-boot] done in 5.6s
```

### Phase 2a — Helpers (cell 7 output)
```
[helpers] chamber_for_citation('BGE 142 III 296') = III
[helpers] chamber_for_citation('1B_490/2017')      = 1B
[helpers] chamber_class('1B') = criminal_procedure
[helpers] doctrinal_density sample DE: 0.85
[helpers] doctrinal_density sample DE facts: 0.0
```

### Phase 2b — Dossier (cell 9 output)
```
[dossier] computing per-doc cache for 255713 unique docs...
  per-doc cache built in 5.4s
[dossier] computing per-(qid, did) cross features...
[dossier] complete for 10 queries in 25.8s

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

(Pool sizes ~41-47k per query; `is_dispositif` is 0 across all queries; `avg_stat_overlap_lead` is near-zero, indicating very few candidates have a query-target statute in their first 200 chars.)

### Stage A — cheap rule prune (cell 11 output)
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
Stage A: 71–92% pool retained, 100% gold retained on every query. Only `cantonal_court` and `noise_role` rules fire (no `dispositif_regex` hits, no `no_dossier`). (Note: per-query gold counts shown here total 220, not 254 — the `_g_set & pool` intersection differs from the full `ALL_GOLD_DOC_SET` total because some gold sits outside the 50k pool.)

### Stage B — fusion-rank top-2000 (cell 13 output)
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

### Phase 3a — Qwen3-32B-AWQ load (cell 16 output, abridged)
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
INFO 05-13 01:56:21 [utils.py:233] non-default args: ...
INFO 05-13 01:56:22 [model.py:555] Resolved architecture: Qwen3ForCausalLM
INFO 05-13 01:56:22 [model.py:1680] Using max model len 4096
INFO 05-13 01:56:23 [awq_marlin.py:252] The model is convertible to awq_marlin during runtime. Using awq_marlin kernel.
INFO 05-13 01:56:23 [scheduler.py:239] Chunked prefill is enabled with max_num_batched_tokens=16384.
INFO 05-13 01:56:23 [vllm.py:840] Asynchronous scheduling is enabled.

[llm] loaded in 42.7s
  GPU 0: free=10.25 GiB / 94.97 GiB
[llm] warm-up (16 short prompts)...
[llm] warm-up done in 3.12s; sample: 'probe-0'
```

### Stage C — listwise LLM scoring (cell 18 output, INTERRUPTED)
```
[Stage C] listwise scoring on Stage-B top-2000 per query (batch=12)...
    [Stage C] val_001: 166 oversized batch(es) auto-split to fit ctx
  [val_001]  209.6s  333 batches  hist={1: 604, 2: 648, 4: 383, 3: 358, 5: 7}  keep≥3: 748  gold_in_keep: 9/42
    [Stage C] val_002: 166 oversized batch(es) auto-split to fit ctx
  [val_002]  200.4s  333 batches  hist={1: 684, 2: 551, 4: 262, 3: 495, 5: 8}  keep≥3: 765  gold_in_keep: 4/38
    [Stage C] val_003: 166 oversized batch(es) auto-split to fit ctx
```
- `val_001`: 209.6 s; 333 batches (167 original 12-batches × ~2 after oversize splits); score hist `{1:604, 2:648, 3:358, 4:383, 5:7}` (total 2000 — confirms STAGE_B_TOP_N=2000 fed in); 748 candidates kept at threshold ≥3; **gold-in-keep = 9/42 → Stage-C recall = 0.214 on val_001**, dropped from Stage-B 16/42 = 0.381.
- `val_002`: 200.4 s; 333 batches; gold-in-keep = 4/38 = 0.105 (down from Stage-B 22/38 = 0.579).
- `val_003`: started splitting batches; no completion line — execution stopped or was interrupted here. No Stage-D / E / F / eval outputs were ever produced.
- Oversized-batch indicator: every query reports `166 oversized batch(es) auto-split to fit ctx`. This means roughly every batch of 12 hits the 3332-token budget, so each 12-batch is bisected at least once — effectively running at batch size ≤6.

### Stage D / E / F / Eval
No outputs. Cells 20, 23, 25, 27 have `outputs: []` — never executed in this saved notebook state.

## Summary

This is a 5-stage precision cascade designed to compress the v7.5 RRF top-50k pool into a variable-K, evidence-quote-validated final set, but the diagnostic run **stopped in the middle of Stage C on val_003** and Stages D/E/F/Eval produced no outputs. The pipeline up to Stage B works as intended on hardware (Qwen3-32B-AWQ loads cleanly on the RTX PRO 6000 Blackwell SM 120 once FlashInfer is disabled for SM 12.x, vLLM auto-selects FlashAttention 4, warm-up completes), Stage A retains 100% of in-pool gold while dropping 8–29% of the pool by cantonal-court + noise-role rules, and Stage B reaches macro R@2000 = 0.618. The headline failure mode is Stage C: listwise scoring with Qwen3-32B over a 2000-candidate pool **destroys recall** — on val_001 the gold-keep collapses from 16/42 (Stage B) to 9/42 (Stage C), and on val_002 from 22/38 to 4/38, meaning the LLM is scoring most true gold ≤2. Secondary issues are (a) the 166-oversized-batch-per-query symptom, which means the `MAX_MODEL_LEN=4096` budget is too tight for batch=12 with full dossier blocks (every batch gets bisected, effectively halving throughput), and (b) ~200 s per query × 10 queries = >30 min of Stage C alone, likely exhausting Colab runtime before reaching Stage D. Lessons for the next iteration: (1) shrink Stage C input (smaller dossier text excerpt, smaller batch, or raise `max_model_len` to 8192) so batches don't get bisected; (2) the listwise scoring is the precision bottleneck — investigate why Qwen3-32B is rejecting true gold (prompt design, dossier signal salience, or the 1–5 scale calibration); (3) save partial state after each stage so a runtime stop doesn't lose everything; (4) Stage-B fusion-rank R@2000 = 0.618 is a hard ceiling for downstream — any Stage-C/D/E/F precision gain has to come from this 61.8% gold-retained pool.
