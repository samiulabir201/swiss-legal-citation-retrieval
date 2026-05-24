# FINDINGS SYNTHESIS — what we learned and what to try next

Generated 2026-05-15. This document ties together the four parallel analyses (data / papers / experiments / cross-cutting) into a single strategic view of:
- what we know with confidence (empirically established)
- what we tried and dropped (negative results worth not repeating)
- what we currently use (the pipeline we have)
- the gap we still face (current vs target F1)
- open levers (priority-ordered with falsifiable pass/fail criteria)
- anti-recommendations (where NOT to spend time)
- cleanup actions

For full per-finding detail, follow the citations to the section docs.

---

## 1. What we know with confidence (high evidence)

### 1a. The retrieval pool is solved at its structural ceiling
- **Macro R@50k = 0.901** on val from the v7.5 multi-query funnel.
  Evidence: `experiments_analysis/01_current_direction_v75_anchor_funnel.md`. Snapshot data: `data_analysis/DATA_INVENTORY.md` §5B.
- **val_003 R_max = 0.766** in the pool — this is the hardest-to-retrieve query.
  Evidence: per-query `gold_in_top.json` files in `drive_sync/swiss_law/v7_pool_recall_089_and_per_query/`.
- **Implication:** macro F1 ≈ 0.93 is the *theoretical max* even if the rest of the pipeline becomes perfect. Macro F1 = 0.8 is achievable; macro F1 = 0.9+ is not without changing the pool.

### 1b. Rerankers add zero value above K=500
- **F0 (plain fusion) R@10k = 0.816 > F3 (hybrid) R@10k = 0.791**
- **At K=2k: F0 = 0.611 > F3 = 0.568**
- Above K=500, every reranker config tested is worse than no reranker.
  Evidence: `experiments_analysis/01_current_direction_v75_anchor_funnel.md` reranker shootout section; `paper_analysis/03_rerankers_calibration.md`.
- **Implication:** the F3 small-K lift comes from the 9 dossier signals, not the rerankers themselves. ROADMAP Step 0 (F3-without-rerankers diagnostic) will confirm this in 15 min with no GPU.

### 1c. Dense embeddings alone cannot solve this task
- Qwen3-Embedding-8B alone on the 2.16M unified corpus: **Macro F1 = 0.044**
- Dense embedding ceiling: R@1000 = 0.289 (Obs 3)
  Evidence: `experiments_analysis/04_early_exp_funnel_misc_pdf.md` early funnel section.
- **Implication:** multi-channel retrieval (v7.5) is necessary, not optional.

### 1d. Citation graph is a tie-breaker, NOT a co-predictor
- **96% of gold citations are nodes** in the graph, BUT **99.41% of gold PAIRS have no edge** (Obs 2).
- Sister provisions in the same law don't cite each other.
  Evidence: `paper_analysis/02_legal_graph_citation.md`; `data_insights/citation_graph_db_and_edges/`.
- **Implication:** HGE/R-GCN learned link prediction (Missing Link paper) cannot work for our task. Graph is only useful as a deterministic counter channel (`statute_backprop` recall 0.453, `graph_forward` recall 0.324 in v7.5).

### 1e. Train priors are NOT transferable to val
- Train: 99% German, val: 100% English. Train Class A ceiling = 28.5% vs val Class A = 100%.
- Train median 2 gold citations vs val median 22.
  Evidence: `data_analysis/01_original_competition_data_and_gold.md` + Obs 1 + Obs 4 in `personal_observations.md`.
- **Implication:** any train-based supervised learning needs explicit cross-lingual + class-distribution adjustment, OR avoid train priors entirely.

### 1f. The endgame catastrophe was a multi-bug compound
- May 9 endgame_colab notebook: F1 < 0.10 (down from prior best 0.7082).
- Caused by 4 compounding judge bugs + unwired enrichment prefilter + reranker law-family text-feed mismatch.
  Evidence: `experiments_analysis/02_pre_v75_pipeline_embedding_reference.md`; `research/endgame_handoff_2026-05-09.md` §4.3 and §4.4.
- **Implication:** before any judge re-run, DELETE `cache_endgame/judge/` (87% rubber-stamp YES poisoning). Use default-NO on parse failure; force JSON output; no `<think>` budget.

### 1g. Qwen3-Embedding-8B + Qwen3-Reranker-8B are deployed faithfully per paper
- 4096-dim embeddings, bf16 weights / fp16 storage
- `Instruct: {...}\nQuery: {...}` template with English instructions over DE/FR/IT (paper-recommended cross-lingual convention)
- Reranker `score = exp(yes) / (exp(yes) + exp(no))` calibrated formula verbatim
  Evidence: `paper_analysis/03_rerankers_calibration.md`.

### 1h. LLM enrichment is complete and high-quality on the rows we have
- Court: 363,258 rows / 2,476,315 total = **14.7% coverage**. 99.987% ok rate. 22.85 rows/s on Blackwell.
- Law: 173,033 rows / 175,933 total = **98.4% coverage**. 98.8% usable, avg `terms_grounded` 0.902. ~11 rows/s.
  Evidence: `experiments_analysis/03_enrichment_notebooks.md`; `data_analysis/03_llm_enrichment_outputs.md`.
- **Implication:** the 85% court coverage gap is *structurally non-binding* because val_003 R_max=0.766 is set by retrieval, not by missing enrichment.

---

## 2. What we tried and dropped (negative results — do not repeat)

| Approach | Source notebook | Best result | Reason dropped |
|---|---|---|---|
| Dense-only retrieval | `early_dense_embedding_test.ipynb` | F1 = 0.044 | Dense ceiling R@1000 = 0.289 |
| segment_lattice_v3 | `early_segment_lattice_funnel_v3.ipynb` + 9 artifact files | court_recall = 0, mean R = 0.40 | Court component broken; abandoned |
| BGE-reranker-v2-m3 | Hybrid shootout | R@2k macro = 0.073 (random) | 4/10 queries at 0.000; cannot recognize Swiss legal cross-lingual |
| Legal-Swiss-RoBERTa fine-tune | `exp_A6_legal_roberta_finetune.ipynb` | stat_recall@500 = 0.101 vs BGE-M3 0.376 | Massive regression — domain fine-tune hurts at this scale (matches Stern paper observation contradicted) |
| Endgame judge (with `<think>`) | `endgame_colab_full_pipeline.ipynb` | F1 < 0.10 | 4 cascading bugs + 87% rubber-stamp YES |
| Qwen3.5-35B-A3B for authority cards | 12 `auth_cards_enrich_qwen35_*` notebooks | 80% invalid JSON (800/1000 parse-fails) | Structured outputs don't compose with this MoE under our template |
| Reranker as recall expansion (K=5000) | Anchor funnel v5 | R = 0.333 vs RRF-only 0.357 | Reranker LOST recall, dropped at v6 |
| Citation graph as co-predictor expander | Obs 2 | 99.41% gold pairs no edge | Sister provisions don't cite each other |
| Reranker above K=500 | Hybrid shootout | F0 R@10k=0.816 > F3=0.791 | Rerankers add NO recall above K=500 |
| Hard-neg filter (F4) | Hybrid shootout F4 config | Saturates at 0.796 above K=20k | Removes recoverable candidates; keep as rank penalty, not filter |
| Fixed-K prediction | Implied by gold size 10-47 | Provably suboptimal | Variable gold size requires per-query K |
| Query-specific hardcoding | Banned by `.claude/memory/feedback_no_query_specific_hardcoding.md` | N/A | Doesn't generalize |
| FP8 KV on Blackwell AWQ Marlin | Law enrichment attempt | Crashes (SM 12.0) | Use FP16 KV |
| FAISS-IVF for retrieval | Endgame §6.5 | N/A | Brute force ~50 ms/query on Blackwell; no need |

---

## 3. What we currently use (the active pipeline)

| Stage | Component | Source |
|---|---|---|
| Retrieval — first stage | Qwen3-Embedding-8B over 27 fp16 chunks (~21 GB) in `embeddings/` | Faithful per paper |
| Retrieval — fusion | v7.5 14-channel RRF (BM25 + dense + statute_backprop + graph_forward + concept + term + co-cite + …) | `v7_4_fixes/cell_bodies_extracted_from_notebook/_cell34_body.py` |
| Candidate pool | `corpus_snapshot.json.gz` per-query top-50k from v7.5 funnel | macro R@50k = 0.901 |
| Reranker | Qwen3-Reranker-8B (only useful K≤200) | `paper_analysis/03_rerankers_calibration.md` |
| Dossier features | 9 signals: statute_int, lead_statute_int, concept_int, term_int, co-cite, doctrinal, chamber, not-hardneg, fusion_rank | Hybrid F3 |
| LLM judge | Qwen3-8B (broken in endgame; needs re-design per ROADMAP Step 2) | OPEN — needs fixes from endgame §4.3 |
| Final pick | Threshold = 0.65 on fused score → val F1 = **0.498** | `threshold.json` in `drive_sync/swiss_law/threshold_and_runtime_configs/` |

---

## 4. The current gap

| Metric | Value |
|---|---|
| Target | Macro F1 0.6-0.8 |
| **Currently submitted** | **Macro F1 = 0.498** (threshold = 0.65) |
| Best historical (val-tuned, on smaller corpus) | F1 = 0.777 (Untitled75 reference) |
| Best pre-v7.5 verifiable | F1 = 0.7082 (Stage 0-6 adaptive K) |
| Theoretical max with current pool | F1 ≈ 0.93 (capped by val_003 R_max = 0.766) |
| Strict floor never reached | macro ≥ 0.8 AND min ≥ 0.8 — structurally blocked at val_003 |

**The gap is +0.10 to +0.30 macro F1.** It lives in **Stages 2-3 (dossier-aware reranking + variable-K final pick)**, NOT retrieval.

---

## 5. Open levers — priority-ordered with falsifiable pass/fail

### Step 0 — F3-without-rerankers diagnostic (15 min, no GPU)
- **What:** recompute F3's RRF dropping the 3 reranker rank columns, keep only 9 dossier signals + fusion_rank
- **Input:** cached `scores_{qwen3,bge,jina}.npz` + `dossier_features.npz` in `drive_sync/swiss_law/hybrid_rerank_shootout_cache/cache/`
- **Pass criterion:** F3-no-rerank within 2 points of F3 at every K → drop reranker stage entirely
- **Saves if pass:** ~1100 s/query of GPU time, simpler pipeline
- **Cost:** zero — uses cached scores

### Step 1 — Cascade dossier Phase 1 (the agreed plan)
- **What:** add 3 cross-features to Stage 2 LLM prompt block
  - (a) channel-of-arrival fingerprint (free — already in snapshot's `channel_hit_sets`)
  - (b) statute-target intersection (doc statute anchors ∩ query statute targets)
  - (c) co-citation density per law in top-100
- **Pass criterion:** Stage 2 mean R@100 lift from ~0.55 → ≥ 0.70
- **Source:** `research/cascade_dossier_plan.md`
- **Cost:** ~1 day to implement; runs locally on cached pool

### Step 2 — Variable-K Stage E with AdaptiveK
- **What:** sort fused scores desc, cut at largest score-gap + buffer B=5
- **Source paper:** Taguchi et al. EMNLP 2025 (`paper_analysis/04_...`)
- **Why first:** matches val's variable-gold-count range (10-47 per query); no GPU; designed for aggregation queries
- **Pass criterion:** Macro F1 ≥ 0.55 (vs current 0.498)
- **Cost:** ~2 hours to implement; replaces threshold=0.65 logic

### Step 3 — Talos top-K loss (the only learning-objective method)
- **What:** quantile-based differentiable top-K loss
- **Source paper:** arXiv 2601.19276, Jan 2026
- **Why:** only one of 7 variable-K methods that's a learning objective, not a post-hoc cutoff. Matches our actual top-K F1 task goal.
- **Cost:** ~1-2 days (needs small held-out calibration data)

### Step 4 — Granularity-aware final pick
- **What:** apply `gold_parent_link_check` rule: if `Art. 11 Abs. 2 OR` is predicted, exclude `Art. 11 OR`
- **Pass criterion:** Macro F1 ≥ 0.60 (the project target's lower bound)
- **Source:** `data_insights/gold_analysis_coverage_and_parents/gold_parent_link_check.csv`
- **Cost:** ~1 hour — pure post-processing rule

### Step 5 — CaseLink-style GAT trained on v7.5 pool (longer shot)
- **What:** small GAT learning channel-weights + role boosts from val gold supervision
- **Source paper:** UQLegalAI COLIEE 2025
- **Why:** the cascade dossier plan already enumerates the exact features a GAT would learn — could outperform the hand-tuned RRF
- **Cost:** ~2-3 days; needs careful train/val split given the 10-query val

---

## 6. Anti-recommendations — do NOT spend time on

| Activity | Why not |
|---|---|
| Improving pool recall further | val_003 R_max = 0.766 is a structural pool ceiling |
| Switching encoder away from Qwen3-Embedding-8B | At parity with best in the literature for our cross-lingual scale |
| HGE/R-GCN learned link prediction (Missing Link paper) | 99.41% gold pairs have no edge (Obs 2) |
| Stern's judgment-criticality classification task | Different task surface — not citation retrieval |
| Qwen3.5-35B-A3B for enrichment | 80% invalid JSON; abandoned with evidence |
| BGE-reranker-v2-m3 | R@2k = 0.073 — essentially random on Swiss legal |
| Legal-Swiss-RoBERTa fine-tune | stat_recall@500 = 0.101 vs BGE-M3 0.376 |
| Re-extracting citation graph | Already at 96% gold-node coverage |
| FAISS-IVF | Brute force is ~50 ms/query on Blackwell |
| FP8 KV on AWQ Marlin | Crashes on Blackwell SM 12.0; use FP16 KV |
| Test-set submission tuning | Meaningless until val F1 ≥ 0.55 |
| Court LLM enrichment for the 85% gap | Coverage isn't the bottleneck (val_003 retrieval is) |

---

## 7. Cleanup actions

| Action | Reason | Effort |
|---|---|---|
| Move 4 misfiled notebooks to `notebooks/_misfiled_other_competitions/` | `investigate_misc_notebook` (Retroviral), `pdf_extraction_marker_pymupdf_pdfminer` (Make Data Count), `pipeline_no_debug_prints` (AIMO-3), `reference_F1_0.7716_scored` (RT-activity; mislabeled — its actual score is 0.5874) | 5 min |
| Delete `cache_endgame/judge/*` | 87% rubber-stamp YES poisoning — toxic for any judge re-run | 1 min |
| Investigate `build_court_authority_cards.py:1238-1241` bug | Unconditional `_chunkNN` suffix appends produce 5.5 GB of duplicate `_chunkNN_chunkNN.jsonl` files | 15 min |
| Refresh stale Drive law-descriptors mirror | `drive_sync/swiss_law/llm_enrichment_jsonl_checkpoints/law_llm_descriptors_0000000_all.jsonl` is byte-identical to local `.bak`, NOT current | 5 min |
| Verify `final_2026_pipeline_attempt.ipynb` reference | Doesn't exist; subfolder 01 actually has `precision_v1_not_working_diagnostic.ipynb` | 1 min |

---

## 8. Confidence map (what's settled vs what's a hunch)

| Claim | Confidence | Evidence type |
|---|---|---|
| val_003 R_max = 0.766 caps macro F1 | **HIGH** | per-query `gold_in_top.json` |
| Rerankers don't help K≥500 | **HIGH** | full F0/F1/F2/F3/F4 sweep with cached scores |
| Dense alone gives F1=0.044 | **HIGH** | early funnel notebook output |
| Qwen3-Embedding-8B is at parity with literature | **MEDIUM-HIGH** | paper claims vs our deployment matches |
| Legal-Swiss-RoBERTa fine-tune fails at our scale | **MEDIUM** | one notebook (exp_A6); could be hyperparameter tuning issue |
| 87% judge YES rate is "rubber-stamp" | **MEDIUM** | rate is suspicious but not directly compared to human gold per cell |
| AdaptiveK will move us toward F1=0.55 | **LOW** (untested) | paper claims + our variable gold size fits — needs measurement |
| Talos will be best of the 7 variable-K methods | **LOW** (untested) | architectural fit only — needs measurement |
| Cascade dossier Phase 1 will hit R@100 ≥ 0.70 | **LOW** (untested) | based on plan + observed signal strength of similar features in F3 |

---

## 9. Document map

| Topic | Master | Sections |
|---|---|---|
| Data | `docs/data_analysis/DATA_INVENTORY.md` | 5 section files (01-05) |
| Papers | `docs/paper_analysis/PAPER_TO_PRACTICE.md` | 5 section files (01-05) |
| Experiments | `docs/experiments_analysis/EXPERIMENTS_TRACE.md` | 4 section files (01-04) |
| Top-level | `README.md` + `ROADMAP.md` | — |
| Skills (always loaded) | `.claude/skills/swiss-citation-{orchestrator,config,task,data,experiments,approach}/SKILL.md` | — |

---

## 10. One-paragraph TL;DR

The retrieval pool is solved (R@50k=0.901, val_003 R_max=0.766 is the structural ceiling). The final-set sizing layer is broken: current submitted F1 is **0.498**, target is 0.6-0.8, theoretical max with this pool is ~0.93. The next move is the **F3-without-rerankers diagnostic (15 min, no GPU)**, then **AdaptiveK score-gap variable-K Stage E (2 hours)**, then if needed **Talos top-K loss (1-2 days)** plus **granularity-aware final pick (1 hour)**. Stop spending time on bigger encoders, better rerankers above K=500, learned graph link prediction, fine-tuning Legal-Swiss-RoBERTa, Qwen3.5-35B for enrichment, or improving the recall pool — all proven to not move the needle. Delete `cache_endgame/judge/` and move 4 misfiled notebooks before the next run.
