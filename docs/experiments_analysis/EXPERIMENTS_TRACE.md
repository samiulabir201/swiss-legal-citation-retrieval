# EXPERIMENTS TRACE — master index

Generated 2026-05-15. 90 notebooks traced across 14 subfolders by 4 parallel agents reading the inventory `.md` files (which contain extracted code cells + outputs, not just markdown). Every claim has a cell-number citation.

## Sections

| # | Section | File | Lines | Scope (notebooks) |
|---|---|---|---|---|
| 1 | Current direction + v7.5 pool + anchor funnel evolution | [`01_current_direction_v75_anchor_funnel.md`](01_current_direction_v75_anchor_funnel.md) | 748 | Subfolders 01 (9), 02 (3), 03 (9) = 21 notebooks |
| 2 | Pre-v7.5 pipeline + embedding/retrieval base + high-scoring reference | [`02_pre_v75_pipeline_embedding_reference.md`](02_pre_v75_pipeline_embedding_reference.md) | 315 | Subfolders 04 (13), 05 (3), 06 (2) = 18 notebooks (17 actual) |
| 3 | Enrichment notebooks (law + court LLM + court concept + auth cards) | [`03_enrichment_notebooks.md`](03_enrichment_notebooks.md) | 319 | Subfolders 07 (4), 08 (7), 09 (5), 10 (12) = 28 notebooks |
| 4 | Early experiments + early funnel + misc + PDF extraction | [`04_early_exp_funnel_misc_pdf.md`](04_early_exp_funnel_misc_pdf.md) | 447 | Subfolders 11 (7), 12 (3), 13 (12), 14 (1) = 23 notebooks |

**Total: 1,829 lines across 4 section docs covering all 90 notebooks.**

## Headline progression (chronological)

### Phase 1 — exp_A1-A6 (April 21-22, 2026)
Initial benchmarking. Topped out at **stat_recall@500 = 0.376** with Qwen3-32B HyDE+Enum RRF over BGE-M3. Never hit the project's own 0.40 gate. Cross-encoder reranking (A4/A4b) only marginally beat no-rerank baseline. **A6 legal-swiss-RoBERTa fine-tune collapsed to 0.101** — the failed paper-derived experiment from `paper_analysis/01_legal_swiss.md`.

### Phase 2 — Early funnel (April 26-30)
- Killed the "dense alone" hypothesis: Qwen3-Embedding-8B on 2.16M unified corpus → Macro F1 = **0.044**
- Pivoted to LLM-planner + choice-cards architecture
- segment_lattice_v3 attempted then abandoned (court_recall = 0, mean recall = 0.40)

### Phase 3 — Kaggle infra debug + Untitled75 reference (~April 28 - May 8)
- 12 misc Kaggle notebooks: pure infrastructure for Qwen3-8B-AWQ on 2×T4. None captured a finished 363k run (Blackwell did that later).
- **Untitled75 reference reaches F1=0.777** on val (the holy grail to recreate).
- Embedding & retrieval base notebooks produce the 28-chunk fp16 embeddings + manifest + val query vectors.

### Phase 4 — pre-v7.5 pipeline iterations (April 12 - May 9)
- 5× `pipeline_end_to_end_*` variants (base / native / explicit_paths / clean_paths / no_function_imports) — progressive code cleanups
- 6× `pipeline_iteration_latest_*` variants — best verified F1 = **0.7082** in `pipeline_iteration_latest_with_all_output.ipynb` (Stage 0-6 adaptive-K, `bm25_plus_llm` w_llm=0.5)
- **May 9 endgame catastrophic regression to F1 < 0.10** in `endgame_colab_full_pipeline.ipynb`:
  - Unwired enrichment prefilter
  - 4 compounding judge bugs (default-YES + "say YES when uncertain" + `<think>` blowup + 200-token truncation)
  - Reranker law-family text-feed mismatch
  - Caused the pivot to v7.5

### Phase 5 — Enrichment runs (May 1 - May 6)
- **Court LLM descriptor: 363,258 rows, 99.987% ok, 22.85 rows/s on Blackwell**
  - Kaggle → Blackwell pivot: kaggle → qwen3_8b_awq → 10k_optimized (13.17 r/s) → blackwell_v4 (15.64 r/s) → 363k_run (22.85 r/s)
- **Law LLM descriptor: 173,033 rows, 98.8% usable, ~11 rows/s on Blackwell**
  - Uses FP16 KV (FP8 KV crashes AWQ Marlin on SM 12.0 Blackwell)
- **Qwen3.5-35B-A3B authority card branch FAILED** — 80% invalid JSON, 800/1000 parse-fails, no corpus-scale run

### Phase 6 — Anchor funnel v1→v7.4 (May 9 - May 11)
- v1: val_001-only, R@1000 = **0.119**
- v4: guarantee-pool channel expansion, R@1000 = **0.238**
- v5: reranker attempt — **reranker HURT recall** (0.333 vs 0.357 RRF-only), explicitly dropped at v6
- v6: drop the reranker
- v7 → v7.4: per-channel mean recall measurement → derived v7.5 weights

### Phase 7 — v7.5 canonical multi-query (May 12 - May 13)
- 13 iterations of the multi-query funnel notebook
- **Final: macro R@50k = 0.901** (skill said 0.893, agent measured 0.901 from latest re-run)
- val_003 R_max = **0.766** = structural floor; cannot exceed without changing the pool

### Phase 8 — Hybrid reranker shootout (May 13 - May 15)
- Qwen3-Reranker-8B + 9 dossier signals (F3 hybrid)
- macro R@2k = **0.568** ; R@20k = **0.868**
- Never breaks strict floor (macro≥0.8 AND min≥0.8) at any K
- Above K=500, plain fusion F0 wins (F0 R@10k = 0.816 > F3 R@10k = 0.791)
- Rerankers help only at K≤200

### Phase 9 — Cascade legalmalr + precision_v1 (May 13 - May 14)
- Scaffolded GRPO + Qwen3-Judge end-to-end F1 pipeline
- **NEVER emitted an executed metric**: `_base` crashed during reranker scoring; `_optimized`/`_poc`/`precision_v1_final` show no captured F1 output
- Status: incomplete / open

## Misfiled notebooks caught (NOT swiss-citation — cleanup candidates)

Four notebooks from other Kaggle competitions slipped into the organized notebooks tree:

| Notebook | Actually belongs to | Subfolder we put it in |
|---|---|---|
| `investigate_misc_notebook.ipynb` | Kaggle Retroviral Wall Challenge | 13_misc_kaggle_and_utilities |
| `pdf_extraction_marker_pymupdf_pdfminer.ipynb` | Make Data Count Challenge (PDF extraction) | 14_pdf_research_paper_extraction (entire subfolder is junk) |
| `pipeline_no_debug_prints.ipynb` | AIMO-3 Math Olympiad | 04_pre_v75_pipeline_iterations |
| `reference_F1_0.7716_scored.ipynb` | Kaggle retroviral RT-activity prediction — its actual CLS score is 0.5874, **not** 0.7716 | 06_high_scoring_reference (a misleading filename) |

→ **Recommend moving all 4 to `notebooks/_misfiled_other_competitions/`** to prevent future confusion.

## Spec discrepancy

`final_2026_pipeline_attempt.ipynb` was referenced in dispatch but does NOT exist in subfolder 01. Subfolder 01 actually contains `precision_v1_not_working_diagnostic.ipynb` instead.

## Definitive notebook per topic (single notebook to read for each result)

| Topic | Definitive notebook | Result |
|---|---|---|
| Best historical F1 (val-tuned reference) | `notebooks/06_high_scoring_reference/reference_F1_0.777_Untitled75.ipynb` | **F1 = 0.777** |
| Best pre-v7.5 verifiable F1 (Stage 0-6 adaptive K) | `notebooks/04_pre_v75_pipeline_iterations/pipeline_iteration_latest_with_all_output.ipynb` | F1 = 0.7082 |
| Canonical retrieval pool | `notebooks/02_v75_multiquery_canonical_pool/pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb` | macro R@50k = 0.901 |
| Per-channel weight derivation | `notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v7_4_val001.ipynb` | feeds v7.5 |
| Reranker shootout | `notebooks/01_current_direction_cascade_rerank_precision/rerank_hybrid_3model_shootout_with_outputs.ipynb` | F0 > F3 above K=500; rerankers dead for K≥500 |
| Endgame failure forensics | `notebooks/04_pre_v75_pipeline_iterations/endgame_colab_full_pipeline.ipynb` | F1 < 0.10 (broken; 4 judge bugs + unwired prefilter + reranker mismatch) |
| Court LLM enrichment (canonical 363k) | `notebooks/08_court_llm_descriptor_extraction/court_llm_descriptor_blackwell_optimized_363k_run.ipynb` | 363,258 rows @ 99.987% ok |
| Law LLM enrichment (canonical 173k) | `notebooks/07_law_de_enrichment/enrich_laws_de_qwen3_8b_kaggle.ipynb` | 173,033 rows |
| Failed Qwen3.5-35B-A3B branch | `notebooks/10_authority_card_enrichment/auth_cards_enrich_qwen35_vllm_fast_stable.ipynb` | 800/1000 invalid JSON, abandoned |

## Cross-cutting infrastructure findings

1. **FP8 KV consistently crashes AWQ Marlin on SM 12.0 (Blackwell)** — use FP16 KV instead. Documented in the law enrichment line.
2. **Kaggle 2×T4 is feasible but slow** for AWQ inference; Blackwell is ~2× faster.
3. **vLLM `max_model_len=1024`** is the operating point for Qwen3-Reranker-8B in the shootout; `max_length=4096` for the Untitled75 reference path.
4. **Structured-output (JSON-schema-constrained) decoding fails ~80% on Qwen3.5-35B-A3B MoE** under the project's prompt template — investigate before retrying that model line.

## Where the full per-notebook details live

For every one of the 90 notebooks: experiment goal, data inputs, data outputs, key code patterns (cell-cited), best metric, verdict, position in chain, cross-ref to experiments skill → see the four section files linked at top.
