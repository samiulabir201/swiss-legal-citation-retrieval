# 01 — Current direction, v7.5 canonical pool, and anchor funnel evolution

Notebook-by-notebook trace of the active retrieval-funnel line for the Swiss citation extraction project.

Sources of truth used:
- Inventory `.md` files under `e:\swiss_citation_extraction\notebooks\_inventory\`
- Experiments ledger at `e:\swiss_citation_extraction\.claude\skills\swiss-citation-experiments\SKILL.md`

Scope note: the task spec listed `final_2026_pipeline_attempt.ipynb` under subfolder 01, but that file does not exist in `notebooks/01_current_direction_cascade_rerank_precision/` (verified by directory listing). Subfolder 01 actually contains 9 notebooks — the 8 named in the spec plus `precision_v1_not_working_diagnostic.ipynb` (also has an inventory). I document what is on disk.

---

## Subfolder 01 — current direction (cascade / rerank / precision)

All notebooks here warm-boot from the v7.5 multi-query snapshot at `research/anchor_funnel_val001_v7/snapshot/` and try to compress its top-50k pool into a final per-query selection (or score the rerankers on it).

---

### Notebook: cascade_legalmalr_v1_base.ipynb
**Path:** `notebooks/01_current_direction_cascade_rerank_precision/cascade_legalmalr_v1_base.ipynb`
**Inventory:** `notebooks/_inventory/cascade_legalmalr_v1_base.md`
**Mtime:** 2026-05-14 14:18
**Code cells:** 23

**Experiment goal:**
First end-to-end "LegalMALR cascade" notebook. Takes the v7.5 50k pool, runs Qwen3-Reranker-8B to rerank to top-1000, runs Qwen3-8B as a YES/NO judge on top-200, fuses rerank + judge + Phase-1 dossier (channels / statute_overlap / co-cite) into a fused per-doc score, and learns the 12-dimensional fusion policy with GRPO (group relative policy optimization) under leave-one-out across the 10 val queries.

**Data inputs (from code):**
- `research/anchor_funnel_val001_v7/snapshot/corpus_snapshot.json.gz` — 255,713 docs (cell 5)
- `research/anchor_funnel_val001_v7/snapshot/per_query_snapshot.json` — 10-query final_topk (cell 5)
- `research/anchor_funnel_val001_v7/snapshot/all_targets.json` — LLM-expanded statute/concept targets (cell 5)
- `research/anchor_funnel_val001_v7/snapshot/hyde_aspects.json` (cell 5)
- `research/anchor_funnel_val001_v7/snapshot/gold_doc_sets.json` — 254 gold dids across 10 queries (cell 5)
- `data/val.csv`, `data/train.csv`, `data/test.csv` (cell 6)

**Data outputs (from code):**
- `outputs/legalmalr_cascade_v1/loo_results.json` (cell 21)
- `outputs/legalmalr_cascade_v1/macro_f1.json` (cell 21)
- `outputs/legalmalr_cascade_v1/val_predictions.json` (cell 21)
- `cache_legalmalr/rerank/<qid>.json` per-query rerank score cache (cell 11)
- `cache_legalmalr/judge/<qid>/<sha1(did)>.json` per-pair judge cache (cell 13)

**Key approach / code patterns:**
- Stage A: `Qwen3Reranker` class (cell 10) — HF `AutoModelForCausalLM`, formats `<Instruct>/<Query>/<Document>` prompt, log-softmax over `yes`/`no` token ids; rerank pool=50000, topn=1000, batch=32, max_len=1024, dtype=bfloat16 (CONFIG cell 4).
- Stage B: `Qwen3Judge` class (cell 12) — Qwen3-8B with `enable_thinking=False`, `max_new_tokens=24`, `default_on_parse_fail="no"` (cell 4; explicit endgame §7 Move 3 fixes baked in).
- Phase-1 dossier per (qid, did): channel-of-arrival fingerprint, statute-target intersection, co-citation density over top-200 court paragraphs (cell 7).
- `render_dossier()` emits a ~250-tok block: citation, surfacing-channel list, statute-hits/targets, cocite count, leading text (cell 8); same block fed to BOTH reranker and judge.
- 12-dim discrete policy (`POLICY_DIMS` cell 9): w_stat_overlap, w_channel_cov, w_cocite, prior_law, prior_court, alpha_rerank, alpha_judge, tau_cut, tau_yes, tau_no, k_base, k_per_target; action space ~2.44e8.
- `policy_score()` (cell 14) fuses rerank * alpha_rerank + judge_signal * alpha_judge + dossier_score * (1-α-α) * family-prior. Auto-yes/auto-no zones at `tau_yes`.
- GRPO trainer (cell 15): G=8 samples/step × 60 steps, lr=0.30, entropy_bonus=0.02; advantage = standardized reward; multiplicative logit updates; LOO across 10 val queries (cell 17).
- Optuna fallback (cell 16); fixed-K sweep over [5..100] (cell 20).

**Best result (from output cells):**
- Stage A (reranker scoring) crashed with `KeyboardInterrupt` at cell 11 during the first query (val_001 scoring 50,000 candidates) — the auto-detected output ends inside `tokenizer.pad()`. The notebook never completed LOO; no F1 number was emitted. Cells 12–23 print no executed output in the inventory.
- "No metric-bearing lines auto-detected in last 5 cells" per `## End findings`.

**Verdict:** abandoned — superseded by `cascade_legalmalr_v1_optimized.ipynb` (next iteration, ~45 min later mtime, adds vLLM judge + batch 64 + bucket-by-length to avoid the same blowout).
**Position in chain:** first attempt of the LegalMALR cascade — base implementation, no speed optimizations.
**Cross-ref to experiments skill:** no separate entry; the cascade family is covered by the "Cascade dossier Phase 1 (PLANNED, not measured)" section (verdict: open).

---

### Notebook: cascade_legalmalr_v1_optimized.ipynb
**Path:** `notebooks/01_current_direction_cascade_rerank_precision/cascade_legalmalr_v1_optimized.ipynb`
**Inventory:** `notebooks/_inventory/cascade_legalmalr_v1_optimized.md`
**Mtime:** 2026-05-14 15:05
**Code cells:** 23

**Experiment goal:**
Same cascade as `_base.ipynb` but optimized for Blackwell throughput: rerank `max_len` dropped 1024→768, rerank `batch_v2`=64, sort-by-length bucketing for ~30% less padding waste, SDPA attention, intra-query checkpointing every 50 batches, AND the judge swapped from HF generate to **vLLM** with `gpu_memory_utilization=0.85`, `max_num_seqs=256`, `max_model_len=3072` — quoted ~10× speedup over the HF loop (CONFIG cell 4).

**Data inputs (from code):**
Identical to `_base.ipynb` (cells 3, 5, 6): the same v7.5 snapshot files + val/train/test CSVs.

**Data outputs (from code):**
- `outputs/legalmalr_cascade_v1/loo_results.json`, `macro_f1.json`, `val_predictions.json` (cell 21)
- `cache_legalmalr/` rerank + judge caches (per-doc; designed to be reused across runs)
- `artifacts_legalmalr/` per-stage dumps (new in optimized version, cell 4 CONFIG.artifact_dir)

**Key approach / code patterns:**
- Same 12-dim GRPO policy and Phase-1 dossier as `_base`.
- vLLM-backed judge with continuous batching: `judge_vllm_max_num_seqs=256` for high parallelism (cell 4).
- Cell 1 deliberately avoids `import torch` before vLLM init — comment documents the Blackwell SM 12.x "Failed core proc(s): {}" silent crash on parent-process CUDA init.
- Cell 20 introduces "mode-then-snap" aggregator for LOO policies (Counter-mode of bin values; falls back to mean+snap-to-bin) — fixes the `predict_fixed_k` issue where averaging non-integer scalars produced off-bin policies.

**Best result (from output cells):**
- Inventory shows code cells only (no executed outputs in steps 4–23 are captured in the `.md` other than setup). `## End findings`: "No metric-bearing lines auto-detected in last 5 cells." No executed metric.

**Verdict:** abandoned — superseded ~22 min later by `cascade_legalmalr_v1_poc.ipynb` which trims val→3 queries to make GRPO LOO tractable.
**Position in chain:** v1 cascade middle iteration; throughput fixes for `_base`.
**Cross-ref to experiments skill:** see "Cascade dossier Phase 1 (PLANNED, not measured)" — verdict open.

---

### Notebook: cascade_legalmalr_v1_poc.ipynb
**Path:** `notebooks/01_current_direction_cascade_rerank_precision/cascade_legalmalr_v1_poc.ipynb`
**Inventory:** `notebooks/_inventory/cascade_legalmalr_v1_poc.md`
**Mtime:** 2026-05-14 15:27
**Code cells:** 23

**Experiment goal:**
POC (proof-of-concept) variant of the optimized cascade. Same model stack and optimizations, but restricts the val set to **3 queries** (`val_001`, `val_004`, `val_006` — CONFIG `query_subset`, cell 4) and switches `policy_optimizer` from "grpo" to **"fixed"** to skip GRPO (only 2 train queries per LOO fold would be too noisy). Idea: get end-to-end numbers cheaply before committing to the full 10-query GRPO run.

**Data inputs (from code):**
Same v7.5 snapshot + val.csv as the other two (cells 3, 5, 6). The query_subset filter applied at iteration time.

**Data outputs (from code):**
- `outputs/legalmalr_cascade_v1/{loo_results,macro_f1,val_predictions}.json` (cell 21) — same paths as siblings, will overwrite if re-run.

**Key approach / code patterns:**
- CONFIG cell 4: `query_subset = ["val_001", "val_004", "val_006"]` — chosen because their pool R_max is 0.929 / 1.000 / 0.944 (the easiest queries; not val_003 at R_max=0.766).
- `policy_optimizer = "fixed"`: just uses `BASELINE_POLICY` from cell 9 — no learning.
- Otherwise identical optimized cascade pipeline.

**Best result (from output cells):**
- Inventory body has no executed output cells past setup. `## End findings`: "No metric-bearing lines auto-detected in last 5 cells." No executed metric.

**Verdict:** abandoned (proof-of-concept scaffolding only) — never produced a measured F1 in the inventory.
**Position in chain:** v1 cascade final iteration in this branch. Closes the cascade_legalmalr_v1 trio (base → optimized → poc).
**Cross-ref to experiments skill:** see "Cascade dossier Phase 1 (PLANNED, not measured)" — verdict open. The cascade Phase 1 plan at `research/cascade_dossier_plan.md` is the design these notebooks implement.

---

### Notebook: precision_v1_final.ipynb
**Path:** `notebooks/01_current_direction_cascade_rerank_precision/precision_v1_final.ipynb`
**Inventory:** `notebooks/_inventory/precision_v1_final.md`
**Mtime:** 2026-05-13 08:25
**Code cells:** 13

**Experiment goal:**
"Precision v1" pipeline: collapse the v7.5 50k pool down to a precision-targeted final selection per query using a 6-stage cascade A→F. Stage A = cheap filters (cantonal-court drop, noise-role drop). Stage B = trust v7.5 RRF order and keep top-N (N=2000). Stage C = Qwen3-32B-AWQ via vLLM as a relevance classifier. Stage D = verdict aggregation. Stage E = composite-score tiebreak (statute_overlap_lead/full, conc_overlap, term_overlap, chamber_match, paragraph_role, doctrinal, aspect_score, co_cite_count, case_peer_rule). Stage F = final selection vs gold.

**Data inputs (from code):**
- `research/anchor_funnel_val001_v7/snapshot/{config,paths,corpus_snapshot.json.gz,per_query_snapshot,all_targets,hyde_aspects,gold_doc_sets}.json[.gz]` (cell 3)
- `data/val.csv` resolved via `paths.json` with fallback (cell 3)

**Data outputs (from code):**
- `research/precision_v1/precision_v1_summary.json` (cell 13, line 1755) — macro/micro P/R/F1 + per-query + Stage-B R@500
- `research/precision_v1/precision_v1_predictions.json` (cell 13, line 1771) — citation strings per query

**Key approach / code patterns:**
- Stage B (cell 7) discards earlier dossier-composite ranking because "earlier dossier-based composite scoring achieved R@300 = 0.013 because most val gold has ZERO statute-target intersection (LLM expansion covers 10-15 statutes per query, but gold spans 40+)". Walks RRF order, keeps first 2000 that survived Stage A.
- Stage C uses **Qwen3-32B-AWQ via vLLM** (cell 8 from line 810 onward), with explicit Blackwell SM 12.x workarounds (no parent-process `torch.cuda.*` calls before vLLM init, FlashInfer 0.6.x has no SM 12.x kernels — detected via nvidia-smi).
- Composite-score function (cell 7, line 717) weights: stat_overlap_lead×8, stat_overlap_full-lead×3, conc_overlap×2, term_overlap×1, chamber_match×1.5, substantive role×1, doctrinal×0.5, aspect_score×0.1 capped at 10, co_cite_count×0.3 capped at 20, case_peer_rule×0.5.
- Stage A drop counts captured per query (cell 6 output, line 685): val_002 loses most (10,074 cantonal_court drops; 71.3% kept); val_001 keeps 91.5% (40,809 of 44,621); all queries preserve **100%** of in-pool gold through Stage A.
- Stage B (cell 7 output, line 803): **macro mean R@2000 = 0.618**, R@1000=0.530, R@500=0.415, R@5000=0.728 baseline.

**Best result (from output cells):**
- Stage A: gold-kept = 100% for every query (cell 6 output, lines 685–694).
- Stage B macro R@2000 = **0.618** (cell 7 output, line 803).
- Per-query Stage-B R@2000 (cell 7 output, lines 788–800): val_001=0.381, val_002=0.579, val_003=0.298, val_004=0.900, val_005 (truncated), val_006 (truncated), val_007 (truncated), val_008 (truncated), val_009=0.714, val_010=0.520.
- Stage C / D / E / F final F1: no executed metric in the inventory — `## End findings`: "No metric-bearing lines auto-detected in last 5 cells." Cells 8–13 (vLLM judge + aggregation + summary) print no outputs in the inventory.

**Verdict:** abandoned — Stage A/B ran, no end-to-end F1 produced. Superseded by the rerank-hybrid shootout (different design path — no judge, replace with RRF over rerankers + dossier signals).
**Position in chain:** the "precision compression" attempt that preceded the rerank shootout. Stage B = fusion-trust insight that fed into the F0 fusion-only baseline in the shootout.
**Cross-ref to experiments skill:** no direct entry. The R@2k fusion-baseline numbers it produced match the F0 row in the shootout (F0 R@2k = 0.611, see shootout table).

---

### Notebook: precision_v1_top_gold_from_50k_candidates.ipynb
**Path:** `notebooks/01_current_direction_cascade_rerank_precision/precision_v1_top_gold_from_50k_candidates.ipynb`
**Inventory:** `notebooks/_inventory/precision_v1_top_gold_from_50k_candidates.md`
**Mtime:** ~ contemporaneous with `precision_v1_final.ipynb`
**Code cells:** ~13 (mirror structure)

**Experiment goal:**
Sibling of `precision_v1_final.ipynb` with the same Stage A→F cascade, focused on the "top-gold from 50k candidates" framing — keeps the Qwen3-32B-AWQ relevance classifier at Stage C and the composite-score tiebreak at Stage E, but is positioned as the comparison/ablation copy.

**Data inputs (from code):**
Identical paths to `precision_v1_final.ipynb` — same v7.5 snapshot files.

**Data outputs (from code):**
- `research/precision_v1/precision_v1_summary.json`, `precision_v1_predictions.json` (cell 13) — same out-dir as sibling, overlapping artifacts.

**Key approach / code patterns:**
- Same 6-stage cascade A→F as the sibling; same composite-score weights; same Stage B trust-fusion design.
- Differences from `_final` are minor (some output formatting, the mean-policy aggregator); cell-by-cell line counts (1815 vs 1794) match within +21 lines — essentially a near-duplicate.

**Best result (from output cells):**
- Stage A and B outputs identical to `_final` (cell 6/7 outputs).
- Stage B macro R@2000 = **0.618** (cell 7 output).
- No executed end-to-end F1 in the inventory.

**Verdict:** abandoned — duplicate/diagnostic copy of `_final`. No additional metric.
**Position in chain:** twin of `precision_v1_final.ipynb`.
**Cross-ref to experiments skill:** none.

---

### Notebook: rerank_hybrid_3model_shootout.ipynb
**Path:** `notebooks/01_current_direction_cascade_rerank_precision/rerank_hybrid_3model_shootout.ipynb`
**Inventory:** `notebooks/_inventory/rerank_hybrid_3model_shootout.md`
**Mtime:** 2026-05-15 08:34
**Code cells:** 10

**Experiment goal:**
Score all 10 val queries × ~43–46k v7.5-pool candidates with **three rerankers** (Qwen3-Reranker-8B, BGE-reranker-v2-m3, jina-reranker-v2-base-multilingual) and test 5 fusion configs over the resulting score matrix: F0 fusion-only baseline, F1 Qwen3-only, F2 3-rerank RRF, F3 hybrid (3-rerank + 9 dossier signals: r_statute, r_lead_stat, r_concept, r_term, r_cocite, r_doctrinal, r_chamber, r_notneg, fusion_rank), F4 F3 + hard-neg filter.

**Data inputs (from code):**
- `research/anchor_funnel_val001_v7/snapshot/{config,corpus_snapshot.json.gz,per_query_snapshot,gold_doc_sets,all_targets,hyde_aspects,paths}.json[.gz]` (cell 2)
- `data/val.csv` (resolved from `paths.json`, cell 2)

**Data outputs (from code):**
- `research/hybrid_rerank_final/cache/scores_qwen3.npz`, `scores_bge.npz`, `scores_jina.npz` — per-(qid,did) raw scores (cells 4–6)
- `research/hybrid_rerank_final/cache/dossier_features.npz` — 9 dossier features per (qid,did) (cell 3/7)
- `research/hybrid_rerank_final/hybrid_rerank_results.json` — final macro/min tables across the 5 configs (cell 10)

**Key approach / code patterns:**
- Phase 0/1 setup with vLLM + FlashInfer install gated on `_NEEDS_RESTART` (cell 1); idempotent.
- Cell 2 warm-boots and prints per-query R_max in v7.5 pool: val_001=0.929 ... val_003=0.766 ... val_010=0.880 (lines 176–185) — confirms structural ceiling at val_003 R_max=0.766.
- Dossier feature extraction uses fixed-substring substring checks (no backtracking regex); throughput claim ~21k docs/s single-thread (cell 3, line 196).
- This file is the "scaffold" — the executed-output sibling is `rerank_hybrid_3model_shootout_with_outputs.ipynb`.

**Best result (from output cells):**
- No executed metric in this version (it's the pre-output run; numbers live in `_with_outputs.md`).

**Verdict:** reference — code-only sibling of `rerank_hybrid_3model_shootout_with_outputs.ipynb`.
**Position in chain:** companion to `_with_outputs`; same code, no captured run.
**Cross-ref to experiments skill:** see `_with_outputs` entry below.

---

### Notebook: rerank_hybrid_3model_shootout_with_outputs.ipynb
**Path:** `notebooks/01_current_direction_cascade_rerank_precision/rerank_hybrid_3model_shootout_with_outputs.ipynb`
**Inventory:** `notebooks/_inventory/rerank_hybrid_3model_shootout_with_outputs.md`
**Mtime:** 2026-05-15 (date of the shootout entry in the experiments ledger)
**Code cells:** 10

**Experiment goal:**
Same as `rerank_hybrid_3model_shootout.ipynb` above, with executed outputs preserved. This IS the headline reranker shootout cited in the experiments ledger.

**Data inputs (from code):**
Same as the no-outputs sibling.

**Data outputs (from code):**
Same paths under `research/hybrid_rerank_final/` (cell 10, line 1393).

**Key approach / code patterns:**
- Three reranker passes with **bf16, max_model_len=1024**, scoring full v7.5 pool (~43k cands/query) for each model.
- Qwen3 path: vLLM with `<Instruct>/<Query>/<Document>` prompt and `<think>\n\n</think>` suffix; reads logprobs at first generated token; max_num_seqs not throttled.
- BGE path: HF `AutoModelForSequenceClassification` bf16, batch=64 (cell 6 in diagnostic).
- jina path: same SequenceClassification head bf16, batch=128, requires xlm_roberta `create_position_ids_from_input_ids` shim for newer transformers (cell 7 in diagnostic).
- F3 hybrid fusion = RRF over `r_qwen3, r_bge, r_jina, r_statute, r_lead_stat, r_concept, r_term, r_cocite, r_doctrinal, r_chamber, r_notneg, fusion_rank` (12 signals).
- F4 = F3 + hard-neg filter (post-filter by hard-neg flag).

**Best result (from output cells):**
- **Per-model standalone R@2k (raw scoring, no fusion)** — from cell 4–7 outputs (inventory lines 1275–1284 and per-query: lines 858–930 Qwen3, 1021–1030 BGE, 1143–1152 jina):
  - Qwen3-Reranker-8B (~800 s/query) macro R@2k ≈ **0.247**
  - BGE-reranker-v2-m3 (~180 s/query) macro R@2k ≈ **0.073** — scores 0.000 on val_003/006/007/009 (4 out of 10)
  - jina-reranker-v2-base-multilingual (~145 s/query) macro R@2k ≈ **0.234**
- **Per-query R@2k (cell 10 output, line 1275–1284)** F3 hybrid:
  - val_001 F3=0.310, val_002=0.553, val_003=0.234, val_004=0.800, val_005=0.727, val_006=0.778, val_007=0.579, val_008=0.467, val_009=0.714, val_010=0.520
- **Macro R@K matrix (cell 10 output, lines 1408–1411):**
  | K | F0 fusion-only | F1 Qwen3-only | F2 3-rerank RRF | F3 hybrid | F4 F3+hardneg |
  |---|---|---|---|---|---|
  | 50 | 0.051 | 0.030 | 0.048 | **0.093** | 0.102 |
  | 100 | 0.120 | 0.051 | 0.056 | **0.224** | 0.230 |
  | 200 | 0.181 | 0.068 | 0.064 | **0.304** | 0.316 |
  | 500 | 0.415 | 0.111 | 0.132 | **0.423** | 0.447 |
  | 1000 | 0.528 | 0.187 | 0.182 | **0.504** | 0.513 |
  | 2000 | **0.611** | 0.250 | 0.268 | 0.568 | 0.570 |
  | 5000 | **0.722** | 0.436 | 0.362 | 0.683 | 0.684 |
  | 10000 | **0.816** | 0.644 | 0.505 | 0.791 | 0.742 |
  | 20000 | 0.848 | 0.816 | 0.776 | **0.868** | 0.785 |
  | 50000 | 0.897 | 0.897 | 0.897 | 0.897 | 0.796 |
- **No config reaches the strict floor** (macro≥0.8 AND min≥0.8) at any K in K_REPORT (cell 10 output, lines 1402–1407 — every config FAIL).

**Verdict:** **kept-baseline / definitive measurement.** The experiments ledger's entire "Hybrid reranker shootout (2026-05-15)" section is this notebook. Verdicts (per ledger):
- BGE rejected (R@2k=0.073, random).
- jina kept-candidate (comparable to Qwen3, 30× smaller, 5× faster; complementary on val_001 + val_004).
- Qwen3 kept-baseline (best single model on average).
- Rerankers only help at K≤200; F0 beats F3 above K=500.
- F3 lift is dossier signals, not rerankers (F2 rerank-only RRF awful: 0.268 R@2k).
- Hard-neg filter (F4) saturates at 0.796 above K=20k — drop as a filter, keep as a rank-penalty.

**Position in chain:** the definitive head-to-head test of all three rerankers + dossier hybrid against the v7.5 pool. This is the notebook to consult for any reranker-related decision.
**Cross-ref to experiments skill:** "Hybrid reranker shootout (2026-05-15)" section — full F0–F4 table and per-model verdicts.

---

### Notebook: rerank_only_diagnostic_F3_no_rerank.ipynb
**Path:** `notebooks/01_current_direction_cascade_rerank_precision/rerank_only_diagnostic_F3_no_rerank.ipynb`
**Inventory:** `notebooks/_inventory/rerank_only_diagnostic_F3_no_rerank.md`
**Mtime:** 2026-05-13 18:11
**Code cells:** 8

**Experiment goal:**
Diagnostic on **3 difficult queries** (val_001, val_003, val_010) to isolate per-reranker behavior on the v7.5 50k pool. Tests four configurations: A = Qwen3 raw doc text + baseline instruction; B = Qwen3 enriched doc representation (citation + statute_anchors + concepts EN + terms + role + text) + cross-lingual instruction; C = BGE-reranker-v2-m3 enriched; D = jina-reranker-v2-base-multilingual enriched. Designed to test if instruction-rewriting + enriched-doc representation lifts Qwen3 cross-lingually (English query, German/French/Italian doc).

**Data inputs (from code):**
- `research/anchor_funnel_val001_v7/snapshot/{config,corpus_snapshot.json.gz,per_query_snapshot,gold_doc_sets,paths}.json[.gz]` (cell 2)
- `data/val.csv` (cell 2)

**Data outputs (from code):**
- `research/rerank_only_diagnostic/rerank_only_results.json` — per-config × per-qid R@K dict (cell 8)

**Key approach / code patterns:**
- Three test qids: `val_001` (R_max=0.929), `val_003` (R_max=0.766, the worst), `val_010` (R_max=0.880) — diagnostic table at cell 3 output, lines 214–218.
- Two doc representations: `repr_raw` (text only, 3000 char cap), `repr_enriched` (citation + family + role + statute anchors + concepts EN + terms + text, 3000 char cap) — cell 3 lines 183–206.
- Two instructions: baseline (`INSTR_BASELINE`, cell 4 line 283) vs cross-lingual (`INSTR_CROSSLING`, cell 5 line 418) — the cross-lingual one explicitly tells the model "Treat language differences as a translation problem, not a mismatch".
- Qwen3 loaded via vLLM `max_model_len=1024, gpu_mem_util=0.85`, scoring via logprobs over `yes`/`no` token ids and softmax (cell 4 lines 247–310).
- BGE via HF `AutoModelForSequenceClassification` bf16, batch 64 (cell 6); jina with `xlm_roberta` shim and batch 128 (cell 7).

**Best result (from output cells):**
- Cell 3 output (lines 214–218) confirms pool ceiling: val_001 R_max=0.929, val_003=0.766, val_010=0.880.
- Configurations A–D scoring was scaffolded (vLLM engine + BGE + jina all loaded per the truncated logs at lines 336–531), but the cell 8 summary output is empty in the inventory — `## End findings`: "No metric-bearing lines auto-detected in last 5 cells."
- No final per-K table emitted in the captured outputs.

**Verdict:** open / scaffolding-complete. This notebook is the precursor to the full 10-query shootout. It established the enriched-doc + cross-lingual-instruction recipe that the shootout then applied to all 10 queries. Note: the filename includes "F3_no_rerank" but the executed content is the 4-config Qwen3/BGE/jina sweep — appears to be a re-purposed file.
**Position in chain:** the 3-query diagnostic that fed into the full 10-query `rerank_hybrid_3model_shootout_with_outputs.ipynb`.
**Cross-ref to experiments skill:** related to "F3-without-rerankers diagnostic (PLANNED, not measured)" — verdict: open. Note however that this notebook's executed content tests the rerankers themselves, not the F3-minus-rerankers ablation; the file appears mis-named relative to its body.

---

### Subfolder 01 summary
Chain progression in 01 (newest → oldest by mtime, with the executed-output anchor on each axis):

- **Cascade family** (Qwen3-Reranker + Qwen3-8B judge + GRPO over 12-dim policy):
  `cascade_legalmalr_v1_base` (2026-05-14 14:18, crashed during scoring) → `_optimized` (2026-05-14 15:05, no executed F1) → `_poc` (2026-05-14 15:27, 3-query subset, fixed policy, no executed F1).
- **Precision family** (Stage A→F with Qwen3-32B-AWQ Stage C):
  `precision_v1_final.ipynb` (2026-05-13, Stage B reaches macro R@2k=0.618, downstream stages produced no executed F1) and its sibling `precision_v1_top_gold_from_50k_candidates.ipynb` (near-duplicate).
- **Reranker shootout family** (the executed measurement):
  `rerank_only_diagnostic_F3_no_rerank.ipynb` (2026-05-13, 3-query setup) → `rerank_hybrid_3model_shootout.ipynb` (2026-05-15, code-only) and `rerank_hybrid_3model_shootout_with_outputs.ipynb` (2026-05-15, **the canonical measurement** — full F0–F4 table).

**Definitive notebook for subfolder 01: `rerank_hybrid_3model_shootout_with_outputs.ipynb`**. It is the only one with executed end-to-end metrics across all 10 val queries on the v7.5 pool, and the experiments ledger cites its numbers verbatim. Best result captured anywhere in this subfolder: F3 hybrid macro R@200=0.304, R@500=0.423, R@2k=0.568, R@20k=0.868; F0 fusion-only macro R@10k=0.816. No config reached the strict (macro≥0.8 AND min≥0.8) floor — structurally blocked by val_003 R_max=0.766.

---

## Subfolder 02 — v7.5 multi-query canonical pool

Three notebooks all run the same 15-channel multi-query funnel against the unified corpus, on Colab + RTX PRO 6000 Blackwell. The CANONICAL one is the snapshot producer for the entire downstream pipeline.

---

### Notebook: pool_v75_multiquery_iteration_11.ipynb
**Path:** `notebooks/02_v75_multiquery_canonical_pool/pool_v75_multiquery_iteration_11.ipynb`
**Inventory:** `notebooks/_inventory/pool_v75_multiquery_iteration_11.md`
**Mtime:** 2026-05-12 23:41
**Code cells:** 31

**Experiment goal:**
Iteration 11 of the v7.5 multi-query anchor-funnel design — 15 channels (statute_backprop, graph_forward, per_area_bedrock, vector_hyde, vector_enriched, vector_raw, law_direct_match, concept_en, bm25, court_statute, term_orig, sibling_expansion, co_citation, graph_reverse, graph_2hop), per-channel RRF weights tuned from v7.4 measured per-channel recall. Topk_final = 50000.

**Data inputs (from code):**
- `data/val.csv` (cell 3)
- `data/checkpoints/law_llm_descriptors_0000000_all.jsonl` (cell 3)
- `artifacts_v2/court_authority_cards_v5_unified.jsonl` (cell 3)
- `artifacts/embeddings/qwen3_8b_unified_chunk*.npy` + manifest parquet (cell 3) — 21 GB fp16 unified embeddings
- `data_insights/citation_graph_extracted.sqlite` (cell 3)

**Data outputs (from code):**
- `research/anchor_funnel_val001_v7/aggregate_summary.json` (per-query R@K + channel mean recall)

**Key approach / code patterns:**
- 15-channel `run_channels()` function with channel-specific budgets (cell ~17–18); RRF weights at cell 5: `statute_backprop=2.5, graph_forward=2.0, vector_hyde=1.5, vector_enriched=1.5, vector_raw=1.5, law_direct_match=1.5, per_area_bedrock=1.2, concept_en=1.0, bm25=1.0, court_statute=0.8, term_orig=0.7, sibling_expansion=0.6, co_citation=0.5, graph_reverse=0.4, graph_2hop=0.0`.
- Structured LLM query expansion via Qwen3-8B: `legal_domain`, `applicable_codes`, `german_terms`, `concept_seeds` (one of the v7.5 "Move 1 fix from endgame §7" pieces).
- HyDE pass (hypothetical-answer paragraph) at ~cell 12, encoded as a third vector query alongside raw and enriched queries.

**Best result (from output cells):**
- **AGGREGATE SUMMARY at topk_final=50,000 (cell 19 output, lines 2666–2683):**
  | qid | gold | R@K | caught |
  |---|---|---|---|
  | val_001 | 42 | 0.929 | 39/42 |
  | val_002 | 36 | 0.806 | 29/36 |
  | val_003 | 47 | 0.766 | 36/47 |
  | val_004 | 10 | 1.000 | 10/10 |
  | val_005 | 11 | 1.000 | 11/11 |
  | val_006 | 18 | 0.944 | 17/18 |
  | val_007 | 19 | 0.895 | 17/19 |
  | val_008 | 29 | 0.862 | 25/29 |
  | val_009 | 14 | 0.929 | 13/14 |
  | val_010 | 25 | 0.880 | 22/25 |
  | **MEAN** | | **0.901** | 219/251 |
- **Macro R@K curve (cell 20 output, lines 2710–2740):** R@50=0.052, R@100=0.121, R@500=0.415, R@1000=0.530, R@2000=0.616, R@5000=0.728, R@10000=0.822, R@20000=0.852, R@35000=0.891, R@46812=**0.901**.
- **Per-channel mean recall (cell 21 output, lines 2767–2783):** statute_backprop=0.592, law_direct_match=0.395, graph_forward=0.344, vector_hyde=0.340, per_area_bedrock=0.330, vector_enriched=0.220, vector_raw=0.213, concept_en=0.116, bm25=0.097, court_statute=0.082, term_orig=0.062, sibling_expansion=0.037, co_citation=0.031, graph_reverse=0.021, graph_2hop=0.000.

**Verdict:** **kept (but superseded by CANONICAL).** Same R@K=0.901 mean and identical per-query numbers as the CANONICAL notebook — iteration 11 is the same run with a different filename. The CANONICAL one with `_recall_0.89_at_k50k_CANONICAL` is the canonical artifact dump (mtime ~19 hours later, 2026-05-13 18:55).
**Position in chain:** v7.5 multi-query iteration #11 — the iteration that hit the 0.901 mean R@50k ceiling.
**Cross-ref to experiments skill:** "v7.5 multi-query anchor funnel (2026-05-12)" section — verdict: kept. Macro R@K curve in the ledger matches: 50k=0.893, 40k=0.892, 30k=0.880, 20k=0.848, 10k=0.816 (the ledger's numbers are slightly different because they aggregate over a different K grid, but the per-query R_max table is identical).

---

### Notebook: pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb
**Path:** `notebooks/02_v75_multiquery_canonical_pool/pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb`
**Inventory:** `notebooks/_inventory/pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.md`
**Mtime:** 2026-05-13 18:55
**Code cells:** 25

**Experiment goal:**
The canonical, snapshot-producing run of the v7.5 multi-query anchor funnel. Same 15-channel design and weights as iteration 11. Designed to emit a "snapshot" folder used by every downstream notebook (cascade, precision, rerank shootout): `corpus_snapshot.json.gz`, `per_query_snapshot.json`, `all_targets.json`, `hyde_aspects.json`, `gold_doc_sets.json`, `config.json`, `paths.json`.

**Data inputs (from code):**
Same as iteration 11 (cell 3): val.csv, law_llm descriptors, court v5 authority cards, qwen3_8b unified embeddings, citation_graph_extracted.sqlite.

**Data outputs (from code):**
- `research/anchor_funnel_val001_v7/snapshot/corpus_snapshot.json.gz` — 255,713 docs with fields ct/cit/fam/cb/pr/ln/sa/cn/tm
- `research/anchor_funnel_val001_v7/snapshot/per_query_snapshot.json` — final_topk + curve + channel_recalls per query
- `research/anchor_funnel_val001_v7/snapshot/all_targets.json` — LLM-expanded statute/concept/term targets per query
- `research/anchor_funnel_val001_v7/snapshot/hyde_aspects.json`, `gold_doc_sets.json`, `config.json`, `paths.json`
- `research/anchor_funnel_val001_v7/aggregate_summary.json`

**Key approach / code patterns:**
- Identical 15-channel pipeline to iteration 11; same RRF weights (cell 5 line 193 onwards).
- Adds the snapshot writer cells: takes the per-query state and dumps it as the warm-boot package consumed by every downstream notebook.

**Best result (from output cells):**
- **AGGREGATE SUMMARY at topk_final=50,000 (cell 19 output, lines 2666–2683)** — identical to iteration 11: val_001=0.929, val_002=0.806, val_003=0.766, val_004=1.000, val_005=1.000, val_006=0.944, val_007=0.895, val_008=0.862, val_009=0.929, val_010=0.880, **MEAN R@50000 = 0.901, micro union 219/251 = 0.873**.
- **Macro R@K curve (cell 20 output, lines 2710–2740):** R@50=0.052, R@100=0.121, R@200=0.179, R@500=0.415, R@1000=0.530, R@2000=0.616, R@5000=0.728, R@10000=0.822, R@20000=0.852, R@35000=0.891, R@46812 (full pool)=0.901.
- **Per-channel mean recall (cell 21 output, lines 2767–2783):** identical to iteration 11.

**Verdict:** **kept — canonical retrieval pool.** This IS the v7.5 snapshot that the entire downstream pipeline reads.
**Position in chain:** **the canonical v7.5 multi-query funnel** — the snapshot writer; every other notebook in subfolders 01/02 warm-boots from its output. The user-memory ledger pins the snapshot path: `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot/` (folder name is from the v7 era; v7.5 numbers live in it).
**Cross-ref to experiments skill:** "v7.5 multi-query anchor funnel (2026-05-12)" section — verdict: kept. Per-query R_max table in the ledger matches verbatim. The ledger notes the min-recall constraint is structurally blocked by val_003 R_max=0.766.

---

### Notebook: pool_v75_multiquery_hyde_test_no_lift.ipynb
**Path:** `notebooks/02_v75_multiquery_canonical_pool/pool_v75_multiquery_hyde_test_no_lift.ipynb`
**Inventory:** `notebooks/_inventory/pool_v75_multiquery_hyde_test_no_lift.md`
**Mtime:** 2026-05-11 22:51
**Code cells:** 24

**Experiment goal:**
HyDE-channel ablation: confirms that adding HyDE (hypothetical-answer paragraph) as a third vector-query channel produces no macro lift over the v7.5 baseline. Same 15-channel design but isolates the vector_hyde channel for analysis. Predecessor of iteration 11.

**Data inputs (from code):**
Same as iteration 11 / CANONICAL.

**Data outputs (from code):**
- `research/anchor_funnel_val001_v7/aggregate_summary.json` (cell 22)

**Key approach / code patterns:**
- Cell 4 CONFIG: includes `budget_vector_hyde=2000` and weight `vector_hyde=1.5` (line 200) — "HyDE — strongest vector signal expected".
- Cell ~10: Qwen3-8B emits a 1100–1500 char hypothetical-answer paragraph per query (HyDE log lines 1519–1523).
- HyDE text encoded with the same Qwen3-Embedding-8B used for raw/enriched; third query vector fed to `channel_vector(q_emb_hyde, budget_vector_hyde)` (cell ~18, line 2132).

**Best result (from output cells):**
- **AGGREGATE SUMMARY at topk_final=50,000 (cell 19 output, lines 2411–2428):** val_001=0.929, val_002=0.778, val_003=0.745, val_004=1.000, val_005=1.000, val_006=0.944, val_007=0.895, val_008=0.828, val_009=0.857, val_010=0.880, **MEAN R@50000 = 0.885, micro union 215/251 = 0.857**.
- **Macro R@K curve (cell 20 output, lines 2457–2485):** R@50=0.085, R@100=0.095, R@200=0.202, R@500=0.351, R@1000=0.495, R@2000=0.619, R@5000=0.724, R@10000=0.815, R@15000=0.865, R@20000=0.882, R@25000=0.885 (plateau).
- **Per-channel mean recall (cell 21 output, lines 2511–2527):** statute_backprop=0.586, graph_forward=0.307, per_area_bedrock=0.285, **vector_hyde=0.273**, vector_enriched=0.220, vector_raw=0.213, ..., graph_2hop=0.000. HyDE recall=0.273 — comparable to vector_enriched/raw but not lifting the macro ceiling (0.885 vs 0.901 with the canonical iteration).
- Per-query HyDE channel hit on val_010 = 44.0% recall (cell 18 output, line 2629) — the strongest single-query contribution.

**Verdict:** **kept as ablation — confirms HyDE does NOT lift macro recall**. Pool ceiling is 0.885 vs 0.901 in the canonical iteration 11; HyDE's per-channel recall (0.273) is in the middle of the pack. Notebook filename "no_lift" encodes the verdict.
**Position in chain:** earlier than iteration 11 / CANONICAL by ~24h. Tested HyDE inclusion before the canonical weight rebalancing settled.
**Cross-ref to experiments skill:** no direct entry; the "Endgame baseline" entry mentions HyDE was in the pre-v7.5 stack and its lift was disappointing; this notebook is the controlled measurement.

---

### Subfolder 02 summary
The three notebooks are the same pipeline at three points in time:
- `pool_v75_multiquery_hyde_test_no_lift.ipynb` (2026-05-11) — HyDE ablation, MEAN R@50k = 0.885.
- `pool_v75_multiquery_iteration_11.ipynb` (2026-05-12) — iteration that hit 0.901 mean.
- `pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb` (2026-05-13) — same numbers as iteration 11 plus the snapshot writer; **this is the canonical artifact**.

**Definitive notebook for subfolder 02: `pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb`**. Mean R@50k = **0.901** macro, micro union 0.873; per-query range [0.766 (val_003) ... 1.000 (val_004/005)]; the snapshot under `research/anchor_funnel_val001_v7/snapshot/` is the warm-boot for every downstream notebook in the project.

---

## Subfolder 03 — anchor funnel evolution (val_001 → v4 → v5 → v6 → v7 → v7.4)

These are the **single-query-then-scaled** evolution that led to v7.5. The chain starts with val_001-only diagnostics, expands to all 10 queries at v7, and the v7→v7.4 cleanup feeds directly into v7.5 (subfolder 02).

The v7_part2/3/4 notebooks are intermediate v7 iterations of the **all-10-queries** runs (not val_001-only); each tunes channel budgets/weights and prints a macro mean R@1000.

---

### Notebook: anchor_funnel_v1_val001_base.ipynb
**Path:** `notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v1_val001_base.ipynb`
**Inventory:** `notebooks/_inventory/anchor_funnel_v1_val001_base.md`
**Mtime:** 2026-05-09 20:28
**Code cells:** 15

**Experiment goal:**
First anchor-funnel notebook. Single query (val_001). Baseline 5–7 channel design (early form): statute_backprop, court_statute, sibling_expansion, graph_forward, etc.; RRF fusion; negative gate (drops cantonal-court + noise roles); R@1000 measurement. Pass criterion: R@1000 = 1.0 (i.e. all 42 gold caught in top-1000) — explicitly a stretch target.

**Data inputs (from code):**
- `data/val.csv`, `data/train.csv`, `data/train_granularity_expanded.csv` (cell 3)
- `data/checkpoints/law_llm_descriptors_0000000_all.jsonl` — law authority cards (cell 3)
- `artifacts_v2/court_authority_cards_v5_unified.jsonl` — court v5 cards (cell 3)

**Data outputs (from code):**
- `research/anchor_funnel_val001/val_001_per_channel_recall.json`
- `research/anchor_funnel_val001/val_001_query_targets.json`

**Key approach / code patterns:**
- Manual channel definitions (no `run_channels()` abstraction yet) — early CHANNELS list (cell ~7).
- Negative gate applied post-fusion: cannot harm recall of `paragraph_role=None` gold (cell 12, line 766).

**Best result (from output cells):**
- Cell 12 output (lines 800–814): **R@1000 = 0.119 (5/42)** for val_001. Pre-gate fused pool: 1,584; post-gate: 1,483. Pass criterion (1.0) failed.
- Per-K curve (lines 801–810): R@50=0.0, R@100=0.0, R@200=0.0, R@300=0.024, R@500=0.024, R@750=0.071, R@1000=0.119.

**Verdict:** **superseded** — baseline that established the funnel idea. F1 not measured.
**Position in chain:** v1 — the start of the anchor funnel chain.
**Cross-ref to experiments skill:** no direct entry; the v1–v6 are pre-history of "Endgame baseline (2026-05-09)" / "v7.5 multi-query anchor funnel (2026-05-12)".

---

### Notebook: anchor_funnel_v4_val001.ipynb
**Path:** `notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v4_val001.ipynb`
**Inventory:** `notebooks/_inventory/anchor_funnel_v4_val001.md`
**Mtime:** 2026-05-10 (early)
**Code cells:** 21

**Experiment goal:**
v4 — single query (val_001). Adds the "guarantee pool" idea: certain high-precision channels (`law_direct_match`, `per_area_bedrock`) bypass RRF and are pinned into the top-K regardless of fusion score. Pass criterion lowered to R@1000 ≥ 0.70.

**Data inputs (from code):**
Same val.csv + law_llm + court_v5 as v1.

**Data outputs (from code):**
- `research/anchor_funnel_val001_v4/` summary, per-channel recall, etc.

**Key approach / code patterns:**
- `Pre-gate fused: 4,364; Post-gate: 4,204; Guarantee pool: 203 (channels: ['law_direct_match', 'per_area_bedrock'])` (cell ~17 output, lines 1263–1266).

**Best result (from output cells):**
- **R@1000 = 0.238 (10/42)** for val_001 (cell 17 output, line 1280).
- R@K curve (lines 1268–1276): R@50=0.048, R@100=0.167, R@500=0.190, R@750=0.238, R@1000=0.238.
- Union upper bound: 10/42 = 0.238 — gold-in-union is already small (cell 16 output, line 1187 reports `Gold in union: 10/42 (UPPER BOUND on R@K)`).

**Verdict:** **superseded** — gold-in-union ceiling of 10/42 = 0.238 is the bottleneck; channel set too narrow.
**Position in chain:** v4 — added guarantee-pool concept. Δ vs v1: R@1000 0.119 → 0.238 (+0.119).
**Cross-ref to experiments skill:** none direct.

---

### Notebook: anchor_funnel_v5_val001.ipynb
**Path:** `notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v5_val001.ipynb`
**Inventory:** `notebooks/_inventory/anchor_funnel_v5_val001.md`
**Mtime:** 2026-05-10
**Code cells:** ~21

**Experiment goal:**
v5 — single query (val_001). Adds **reranker** (Qwen3-Reranker-8B) on top of RRF top-K. Pass criterion R@1000 ≥ 0.60. Tests whether HF-loaded reranker scoring 3000 (query, doc) pairs improves recall.

**Data inputs (from code):**
Same val.csv + law_llm + court_v5 + the same channels as v4 but with more channels added (union now 22/42 vs v4's 10/42 — line 1262).

**Data outputs (from code):**
- `research/anchor_funnel_val001_v5/` summary
- `recall_at_topk` and `rrf_only_recall_at_topk` in summary.json (line 1601–1602)

**Key approach / code patterns:**
- Loads Qwen/Qwen3-Reranker-8B via HF `AutoModelForCausalLM`, scores 3000 pairs in **2896.4s = 965 ms/pair** (cell ~17 output, lines 1509–1510). Channel-count expanded; union 22/42.

**Best result (from output cells):**
- **RRF-only baseline R@1000 = 0.357 (15/42)** for val_001 (cell 17 output, line 1525).
- **Post-rerank R@1000 = 0.333 (14/42)** — rerank made it slightly worse (line 1526).
- Post-rerank R@K curve (lines 1513–1521): R@50=0.0, R@100=0.0, R@200=0.071, R@500=0.214, R@1000=0.333.

**Verdict:** **superseded** — the v6 notebook header literally documents "v6: reranker dropped (v5 measured: 48 min, made recall worse)". The reranker hurt by 1 gold doc (-2.4%) and cost 48 minutes for a single query.
**Position in chain:** v5 — first reranker attempt. Δ vs v4: union 10/42 → 22/42 (channel expansion lifted ceiling); RRF-only R@1000 0.238 → 0.357 (+0.119); post-rerank 0.333 (rerank slightly hurts).
**Cross-ref to experiments skill:** the "Endgame baseline" verdict ("reranker + judge fixes cannot move F1 because they're downstream of the binding stage") is foreshadowed here at single-query scale.

---

### Notebook: anchor_funnel_v6_val001.ipynb
**Path:** `notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v6_val001.ipynb`
**Inventory:** `notebooks/_inventory/anchor_funnel_v6_val001.md`
**Mtime:** 2026-05-10 12:59
**Code cells:** 21

**Experiment goal:**
v6 — single query (val_001). Drops the reranker (explicit comment: "v6: reranker dropped (v5 measured: 48 min, made recall worse)", line 179). Keeps the expanded channel set from v5. RRF-only design with negative gate.

**Data inputs (from code):**
Same as v5 (val.csv, law_llm, court_v5, embeddings, graph DB).

**Data outputs (from code):**
- `research/anchor_funnel_val001_v6/` summary

**Key approach / code patterns:**
- `final_topk = RRF-only top-1000` (cell ~17 output, line 1524).
- Channel set expanded; union gold-in-union = 23/42 (cell ~16 output, line 1393).

**Best result (from output cells):**
- **R@1000 = 0.357 (15/42)** for val_001 (cell 17 output, line 1538).
- R@K curve (lines 1526–1534): R@50=0.167, R@100=0.167, R@500=0.190, R@750=0.310, R@1000=0.357.

**Verdict:** **superseded** — same R@1000 as v5's RRF-only baseline (0.357), confirming the reranker added nothing. Establishes "channels > reranker" as the operating principle.
**Position in chain:** v6 — reranker dropped. Δ vs v5: same R@1000 (0.357), 48 min saved.
**Cross-ref to experiments skill:** consistent with "Rerankers help ONLY at K≤200" finding in the hybrid shootout (2026-05-15).

---

### Notebook: anchor_funnel_v7_val001.ipynb
**Path:** `notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v7_val001.ipynb`
**Inventory:** `notebooks/_inventory/anchor_funnel_v7_val001.md`
**Mtime:** 2026-05-10 21:48
**Code cells:** 22

**Experiment goal:**
v7 — the first run that **expands beyond val_001 to all 10 val queries**. Despite the filename, the notebook iterates `PER_QUERY` across 10 queries (cell ~21 output, line 2277: "10 query folders"). v7 disables `graph_2hop` (recall 0.046 → 0.003 in v7.2 noted in v7.4 CONFIG); RRF weights are: statute_backprop=2.5, graph_forward=2.0 (line 210–213).

**Data inputs (from code):**
Same v6 baseline + graph DB + embeddings, but loops over all val queries.

**Data outputs (from code):**
- `research/anchor_funnel_val001_v7/<qid>/{final_topk,gold_in_top,targets,summary}.json` per query (10 folders)
- `research/anchor_funnel_val001_v7/aggregate_summary.json` (cell 21)

**Key approach / code patterns:**
- Per-query CHANNELS run loop (cell ~17/18), then aggregate macro mean R@1000 and union ceiling (cell 21 lines 2233–2256).
- Per-channel mean recall computed across all 10 queries (line 2249).

**Best result (from output cells):**
- **mean R@1000 (macro) = 0.518; R@1000 (micro) = 0.446** (cell 21 output, lines 2279–2280).
- val_001 specific: R@1000 = 13/42 = 31.0% (line 2315 from End findings).
- 29 of 42 gold missed on val_001; root causes documented in per-gold miss diagnosis.

**Verdict:** **superseded** — first 10-query run; macro mean R@1000 = 0.518. The v7.4 + v7.5 chain pushed this higher.
**Position in chain:** v7 — first 10-query expansion. Δ vs v6 (val_001 only): N/A (different scope).
**Cross-ref to experiments skill:** part of the lineage that becomes "v7.5 multi-query anchor funnel (2026-05-12)" — verdict kept.

---

### Notebook: anchor_funnel_v7_part2_with_outputs.ipynb
**Path:** `notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v7_part2_with_outputs.ipynb`
**Inventory:** `notebooks/_inventory/anchor_funnel_v7_part2_with_outputs.md`
**Mtime:** 2026-05-10 19:26
**Code cells:** 22

**Experiment goal:**
v7 iteration "part 2" — same 10-query loop as v7, different channel-weight / budget tuning. (The "part 2/3/4" suffix is the iteration number within the v7 family.)

**Data inputs (from code):**
Same as v7 (val.csv, embeddings, graph DB, law/court enrichments).

**Data outputs (from code):**
- `research/anchor_funnel_val001_v7/aggregate_summary.json` (cell 21, overwrites the v7 summary).

**Best result (from output cells):**
- **mean R@1000 (macro) = 0.436; R@1000 (micro) = 0.390** (cell 21 output, lines 2069–2070).
- val_001: R@1000 = 14/42 = 33.3% (line 2105 from End findings).

**Verdict:** **superseded** — macro dropped from 0.518 (v7) to 0.436; tuning regressed.
**Position in chain:** v7 iteration #2. Δ macro R@1000: 0.518 → **0.436** (regression).
**Cross-ref to experiments skill:** none direct.

---

### Notebook: anchor_funnel_v7_part3_with_outputs.ipynb
**Path:** `notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v7_part3_with_outputs.ipynb`
**Inventory:** `notebooks/_inventory/anchor_funnel_v7_part3_with_outputs.md`
**Mtime:** 2026-05-10 21:07
**Code cells:** 22

**Experiment goal:**
v7 iteration "part 3" — further tuning on the same 10-query loop. Adjustments to graph_reverse usage and vector budgets visible in the miss diagnosis output (val_001 graph_reverse rank 483, line 2264).

**Data inputs (from code):**
Same as v7.

**Data outputs (from code):**
Same as v7: aggregate_summary.json.

**Best result (from output cells):**
- **mean R@1000 (macro) = 0.513; R@1000 (micro) = 0.438** (cell 21 output, lines 2220–2221).
- val_001: R@1000 = 11/42 = 26.2% (line 2256 from End findings).

**Verdict:** **superseded** — macro 0.513 vs v7's 0.518 (essentially flat).
**Position in chain:** v7 iteration #3. Δ macro R@1000 (vs v7): 0.518 → **0.513** (flat).
**Cross-ref to experiments skill:** none direct.

---

### Notebook: anchor_funnel_v7_part4_with_outputs.ipynb
**Path:** `notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v7_part4_with_outputs.ipynb`
**Inventory:** `notebooks/_inventory/anchor_funnel_v7_part4_with_outputs.md`
**Mtime:** 2026-05-11 13:30
**Code cells:** 23

**Experiment goal:**
v7 iteration "part 4" — single-query (val_001) diagnostic that runs the same channel set with bigger budgets (court_statute=8000, concept_en=3000, term_orig=2500, bm25=2000, vector_raw=2000, vector_enriched=2000) to test "throw more candidates at it". RRF rank too low becomes the dominant failure mode (22 of 26 missed gold).

**Data inputs (from code):**
Same as v7.

**Data outputs (from code):**
- `research/anchor_funnel_val001_v7/{final_topk,gold_in_top,gold_missed_diagnosis,targets,config,summary}.json` (cell 22; single-query layout, no per-query folders).

**Best result (from output cells):**
- val_001: **R@1000 = 16/42 = 38.1%** (line 2559 from End findings); 26 missed gold.
- Failure-mode summary (cell 21 output, lines 2436–2440): NO_CHANNEL_HIT=1, RRF_RANK_TOO_LOW=22, GATED_OUT=3.
- Channel-level hit rate on missed gold (lines 2461–2471): per_area_bedrock catches 10 of 26 missed; statute_backprop catches 11; concept_en catches 9; vector_raw=9; vector_enriched=9.

**Verdict:** **superseded** — single-query diagnostic that fed into v7.4. Shows that bigger channel budgets help (val_001 R@1000 0.310→0.381) but RRF rank is now the binding failure mode.
**Position in chain:** v7 iteration #4 (val_001 deep-dive). Δ vs v7 (val_001): 31.0% → 38.1%.
**Cross-ref to experiments skill:** consistent with the v7.5 strategy of "lift channel budgets and re-tune RRF weights using v7.4 measured per-channel recall" (cited in CANONICAL CONFIG comment, line 190).

---

### Notebook: anchor_funnel_v7_4_val001.ipynb
**Path:** `notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v7_4_val001.ipynb`
**Inventory:** `notebooks/_inventory/anchor_funnel_v7_4_val001.md`
**Mtime:** 2026-05-11 12:37
**Code cells:** 23

**Experiment goal:**
v7.4 — single-query (val_001) diagnostic that **measures per-channel mean recall** across all 10 queries (the data that the v7.5 weights are derived from). graph_2hop disabled because v7.2 dropped recall 0.046 → 0.003 (line 173). RRF weights start to converge on the v7.5 set (statute_backprop=2.5, graph_forward=2.0).

**Data inputs (from code):**
Same as v7.

**Data outputs (from code):**
- `research/anchor_funnel_val001_v7/{final_topk,gold_in_top,gold_missed_diagnosis,targets,config,summary}.json` (cell 22).

**Best result (from output cells):**
- val_001: **R@1000 = 17/42 = 40.5%** (line 2548 from End findings); 25 missed gold.
- Failure modes (cell 21 output, lines 2435–2439): NO_CHANNEL_HIT=12, RRF_RANK_TOO_LOW=9, GATED_OUT=4. Note: 12 NO_CHANNEL_HIT is a regression from v7_part4's 1 — but the channel budgets in v7.4 are smaller (per_area_bedrock=150 vs v7_part4's higher), trading recall for tighter budgets.
- Channel-level hit on missed gold (lines 2447–2461): concept_en=7, vector_enriched=6, vector_raw=3, term_orig=3 — concept channels carry weight on misses.

**Verdict:** **superseded by v7.5** — but v7.4's per-channel mean recall measurements ARE the inputs to v7.5's tuned RRF weights. v7.4 is the bridge.
**Position in chain:** v7.4 — last single-query iteration before scale-up to v7.5 (subfolder 02). Δ vs v7 (val_001): 31.0% → 40.5%.
**Cross-ref to experiments skill:** comment in CANONICAL v7.5 (line 194) explicitly cites v7.4: "v7.5 weights — updated based on v7.4 per-channel mean recall". The v7.4 weights statute_backprop=0.453 mean recall and graph_forward=0.324 (line 193–194 of CANONICAL) come from this notebook.

---

### Subfolder 03 summary
Chain progression: val_001 baseline → v4 → v5 → v6 → v7 → v7.4 → v7.5 (subfolder 02). Per-version val_001 R@1000 deltas:

| version | val_001 R@1000 | macro R@1000 (10 queries) | Δ from prev (val_001) | notable change |
|---|---|---|---|---|
| v1 | 0.119 (5/42) | — (single query) | — | baseline funnel |
| v4 | 0.238 (10/42) | — | +0.119 | guarantee pool |
| v5 | 0.333 post-rerank (0.357 RRF-only) | — | -0.024 | reranker added (slower + slight regression) |
| v6 | 0.357 (15/42) | — | +0.000 (vs v5 RRF-only) | reranker dropped, 48 min saved |
| v7 | 0.310 (13/42) | **0.518** | -0.047 | first 10-query expansion |
| v7 part2 | 0.333 (14/42) | 0.436 | regression | tuning |
| v7 part3 | 0.262 (11/42) | 0.513 | regression | tuning |
| v7 part4 | 0.381 (16/42) | — (val_001 deep-dive) | +0.071 | bigger budgets diagnostic |
| v7.4 | 0.405 (17/42) | — (val_001 only) | +0.024 | per-channel recall measurement for v7.5 weights |
| **v7.5 CANONICAL** | 0.929 R@50k | **0.901 R@50k** | jump | snapshot for downstream — but R is measured at K=50k, not 1k |

Note the scale change at v7.5: earlier versions all measure at K=1000 (a precision target). v7.5 measures at K=50,000 (a recall pool the downstream cascade compresses). At K=1000 the v7.5 macro is **0.530** (not 0.901) — see the CANONICAL R@K curve. So v7.5 vs v7 at the same K=1000: 0.518 → 0.530 (+0.012); the headline jump to 0.901 is purely the pool-size lift (K=1k → K=50k).

**Definitive notebook for subfolder 03: `anchor_funnel_v7_4_val001.ipynb`** for the per-channel recall measurement that drives the v7.5 weights, BUT the actual production output (snapshot + 10-query macro) lives in subfolder 02's CANONICAL notebook. The v7→v7.4 chain is the in-vitro tuning step.

---

## Cross-subfolder overall summary

- **The canonical retrieval artifact** is the snapshot at `research/anchor_funnel_val001_v7/snapshot/` produced by `pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb`. Pool size 50k per query; macro R@50k = 0.901; structurally floored at val_003 R_max = 0.766.
- **The canonical reranker measurement** is `rerank_hybrid_3model_shootout_with_outputs.ipynb`. Best F3 hybrid macro R@2k = 0.568, R@10k = 0.791; F0 fusion-only beats F3 above K=500. No config reached the strict (macro≥0.8 AND min≥0.8) floor.
- **The cascade family (cascade_legalmalr_v1_*)** never emitted an executed F1; the most recent (`_poc`) only ran on 3 queries with fixed policy.
- **The precision_v1 family** reached Stage B with macro R@2k = 0.618 but no end-to-end F1; downstream stages did not execute or were not captured.
- The anchor funnel chain v1 → v7.4 is the in-vitro tuning history that produced the v7.5 channel weights and the snapshot.
