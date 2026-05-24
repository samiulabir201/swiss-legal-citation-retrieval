# Data Analysis 05: Caches, Snapshots, Submissions, Experimental & Citation-Graph Data

_Scope: forensic caches, the canonical v7.5 recall-0.89 snapshot, per-version anchor-funnel artifacts, submission CSVs, eval metric dumps, citation-graph-derived data, abandoned/experimental side-tracks, Omnilex-pipeline data, runtime configs, and A6 fine-tuned models. Generated 2026-05-15._

This file is the inventory + provenance + verdict catalog for everything under `cache_endgame/`, the `drive_sync/swiss_law/` results subtrees, `data_insights/citation_graph_db_and_edges/`, `data_insights/citation_patterns_and_priors/`, and the experimental jsonl/csv outputs lingering under `artifacts/` and `research/stage_b_diagnostics_and_ceilings/`.

Cross-references:
- Producer/consumer relationships derived from `notebooks/_inventory/*.md` (per-notebook code dumps).
- Verdicts confirmed from `.claude/skills/swiss-citation-experiments/SKILL.md`.

---

## A. Endgame caches — `cache_endgame/`

| File | Size | Notes |
|---|---|---|
| `cache_endgame/hyde_cache.json` | 10.9 KB | 10 HyDE legal-rule answers (one per val query). Reusable. |
| `cache_endgame/german_expansion_cache.json` | 17.7 KB | German term/code expansions per val query. Reusable. |
| `cache_endgame/query_embeddings.npy` | 320 KB | 10 val query vectors (Qwen3-Embedding-8B, 4096-d fp16). Reusable. |
| `cache_endgame/query_embeddings_keys.json` | 880 B | doc_id ordering for `.npy`. |
| `cache_endgame/judge/<sha1(query)>/<sha1(citation)>.json` | 10 subdirs × ~165 files ≈ **1,650 verdicts** (sampled, claimed ~1,788) | **TOXIC — DO NOT REUSE.** Default-YES on parse failure + "say YES when uncertain" prompt + `<think>`-token blowout + 200-token truncation. Judge YES rate 87% (1546/1774). Documented in `research/endgame_handoff_2026-05-09.md` §4.3. Skill verdict: must be deleted before any future judge run. |
| `cache_endgame/rerank/<sha1>.json` | 10 files | Old rerank scores (399-529 candidates per query). **Superseded** by `drive_sync/swiss_law/hybrid_rerank_shootout_cache/cache/scores_{qwen3,bge,jina}.npz`. |
| `cache_endgame/loose_root_artifacts/` | 4 files + zip | `debug_25299.txt`, `debug_42125.txt`, `failures_repaired.jsonl` (407 KB), `failures_unrepaired.jsonl` (empty), `failures_with_text.jsonl` (286 KB), `research-20260509T142853Z-3-001.zip` (13.6 KB). Forensic; moved from repo root during 2026-05-15 reorg. |

**Producer:** `notebooks/04_pre_v75_pipeline_iterations/endgame_colab_full_pipeline.ipynb` (confirmed via `notebooks/_inventory/endgame_colab_full_pipeline.md`).
**Consumer (forensics only):** root-cause investigation, not live retrieval.
**Skill verdict (endgame baseline, 2026-05-09):** rejected. Macro F1 < 0.10, gold-in-rerank-pool 23.1%, judge auto-yes/auto-no zone hits = 0. Whole pre-v7.5 pipeline abandoned.

---

## B. v7.5 snapshots — `drive_sync/swiss_law/v7_pool_recall_089_and_per_query/`

### B.1 The canonical snapshot — `snapshot/` subfolder

`drive_sync/swiss_law/v7_pool_recall_089_and_per_query/snapshot/`

| File | Size | Notes |
|---|---|---|
| `corpus_snapshot.json.gz` | **110.7 MB** | **THE recall-0.89 pool.** Per-query top-50k candidates (~43k-46k populated), per-channel hit sets (15 channels), fusion ranks, hard-neg flags, gold doc_ids. **MOST VALUABLE SINGLE FILE IN THE REPO.** Every downstream experiment scores against this pool. |
| `per_query_snapshot.json` | 6.7 MB | Per-query metadata: query text, gold list, channel sizes, channel recalls, fusion config. |
| `all_targets.json` | 22.1 KB | LLM-extracted `statute_targets`, `concept_targets`, `term_targets` per query. |
| `hyde_aspects.json` | 32.1 KB | HyDE aspect answers used as additional query vectors. |
| `gold_doc_sets.json` | 3.8 KB | Ground-truth `doc_id` lists per val query. |
| `config.json` | 1.8 KB | Funnel config: `topk_final=50000`, budgets per channel, `rrf_k=60`, weights (`statute_backprop=2.5`, `graph_forward=2.0`, `concept_en=1.8`, etc.), PRF settings, BM25 settings, vector model = `Qwen/Qwen3-Embedding-8B`, query model = `Qwen/Qwen3-32B`. |
| `paths.json` | 146 B | Drive paths used at generation. |

**Producer:** `notebooks/02_v75_multiquery_canonical_pool/pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb` (inventory `pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.md`).
**Loaded by (consumers):**
- `pool_v75_multiquery_iteration_11.md`
- `rerank_hybrid_3model_shootout.md` / `_with_outputs.md`
- `rerank_only_diagnostic_F3_no_rerank.md`
- `precision_v1_top_gold_from_50k_candidates.md` / `_not_working_diagnostic.md` / `_final.md`
- `cascade_legalmalr_v1_{base,optimized,poc}.md`
- All Stage B diagnostic scripts in `research/stage_b_diagnostics_and_ceilings/`.

**Skill verdict (v7.5 multi-query anchor funnel, 2026-05-12):** kept. Macro R@50k = 0.893; per-query R_max range 0.766 (val_003) to 1.000 (val_004/005). val_003=0.766 is a structural pool ceiling.

### B.2 Local mirror — `research/local_v75_snapshot_mirror/snapshot/snapshot/`

Same 7 files as B.1, mirrored locally for offline/notebook use. Note the doubly nested `snapshot/snapshot/` path (artefact of `drive_pull_curated.py`).

### B.3 Root-level per-query result dumps — `drive_sync/swiss_law/v7_pool_recall_089_and_per_query/` (top folder)

| File | Size | Notes |
|---|---|---|
| `final_topk_val_001.json` ... `final_topk_val_010.json` | ~3.3-3.6 MB × 10 | Per-query ranked top-K (50k) doc_id lists with rank, citation, channels_hit, fusion_score. |
| `final_topk.json` | — | Older single-query (val_001) artefact. |
| `aggregate_summary.json` | 2.1 KB | Mean R@1000 macro = 0.518, micro = 0.446 across 10 val queries; per-query R@1000 and union upper-bound; channel mean recall (top: statute_backprop=0.469, graph_forward=0.321, vector_raw=0.153). |
| `summary.json` | 1.6 KB | val_001-specific R@1000 = 0.381 (older single-query metric — not the canonical Macro R@50k = 0.893). |
| `summary_multiquery.json` / `targets_multiquery.json` | — | Multi-query variants. |
| `targets.json` | — | LLM-extracted target lists. |
| `gold_in_top.json` | — | Gold doc_ids that landed inside top-K with their ranks. |
| `gold_missed_diagnosis.json` / `val_001_gold_missed_diagnosis.json` | — | Per-gold root-cause: `RRF_RANK_TOO_LOW` / `NO_CHANNEL_HIT` / `GATED_OUT`. |
| `val_001/` ... `val_010/` | per-query folders | Each contains `final_topk.json`, `summary.json` (query text, total_gold, R@1000, channel_sizes, channel_recalls, union_upper_bound), `gold_in_top.json`, `targets.json`. |

---

## C. Per-version anchor-funnel results — `v1_anchor_funnel_val001_initial/` + `v4_anchor_funnel_per_version_summaries/`

### C.1 v1 baseline — `drive_sync/swiss_law/v1_anchor_funnel_val001_initial/`

| File | Size | Notes |
|---|---|---|
| `val_001_top1000_pool.txt` | 33.5 KB | Initial v1 top-1000 candidate IDs with ranks. |
| `val_001_summary.md` | 2.2 KB | R@1000 = 0.286 (12/42 gold). Channel breakdown: law_direct_match 14.3%, concept_en 19.0%, court_statute 4.8%, etc. |
| `val_001_query_targets.json` | 1.9 KB | LLM target extraction. |
| `val_001_per_gold_trace.tsv` | 1.6 KB | Per-gold rank trace. |
| `val_001_per_channel_recall.json` | 1.6 KB | Per-channel recall numbers. |

### C.2 v4/v5/v6 iteration summaries — `drive_sync/swiss_law/v4_anchor_funnel_per_version_summaries/`

3 versions × 4 files each (`_top1000_pool.txt`, `_per_channel.json`, `_per_gold_trace.tsv`, `_summary.md`) + shared `val_001_query_targets.json`. Tracks val_001 anchor-funnel evolution v4 → v5 → v6 before the v7 rewrite. Sizes ~32-35 KB each pool file.

**Producer:** `notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v{1,4,5,6}_val001.ipynb` (per-version inventory MDs in `_inventory/`).
**Consumer:** historical reference only; superseded by the v7.5 snapshot at recall 0.893.

---

## D. Submissions — `drive_sync/swiss_law/submitted_predictions_csv/`

| File | Size | Notes |
|---|---|---|
| `submission_test.csv` | 34.9 KB | Predictions for 40 test queries (test_001 ... test_040). Format: `query_id,predicted_citations` (semicolon-separated). |
| `submission_val.csv` | 3.3 KB | Predictions for 10 val queries. |
| `submission_val_pertype.csv` | 8.5 KB | Per-type variant (likely separated by citation family — laws vs courts vs statute_section). |

val_001 row example shows ~31 citations including `Art. 221 Abs. 1 lit. b StPO`, `BGE 137 IV 122 E. 4.x`, `7B_*` and `1B_*` docket cites — consistent with the v7.5 pool top-K candidates after final-pick thresholding.

**Producer:** likely `precision_v1_final.ipynb` (the final-pick logic) feeding through `threshold.json` (§J).
**Consumer:** Kaggle leaderboard submission.

---

## E. Omnilex submissions K-sweep — `drive_sync/omnilex_competition/retrieval_pipeline_submissions_K_sweep/`

| File | Size | K |
|---|---|---|
| `submission_K5.csv` | 4.0 KB | 5 predictions/query |
| `submission_K10.csv` | 7.5 KB | 10 |
| `submission_K15.csv` | 11.1 KB | 15 |
| `submission_K20.csv` | 15.0 KB | 20 |
| `submission_K25.csv` | 18.4 KB | 25 |
| `submission_K30.csv` | 22.0 KB | 30 |

Each row shows steadily growing predicted_citations list. test_001 K=5 example: `Art. 100 Abs. 1 BGG;Art. 467 ZGB;Art. 16 ZGB;Art. 505 Abs. 1 ZGB;Art. 393 Abs. 1 STPO`. These are Omnilex-competition-style fixed-K predictions, distinct from the Swiss-law variable-K submission format.

**Producer:** `drive_sync/omnilex_competition/swiss_law_pipeline_source_code/submit.py` (with `stage4_llm_reranker.py` / `stage4b_cross_encoder.py` / `stage5_verify_and_score.py` upstream).

---

## F. Eval metrics & ablations — `drive_sync/swiss_law/eval_metrics_and_ablations/`

| File | Size | Notes |
|---|---|---|
| `val_summary.json` | 1.5 KB | One-shot eval: 10 queries, top_k=10000. R@50=0.170, R@100=0.204, R@500=0.367, R@1000=0.401, R@5000=0.512, R@10000=0.531. Channel counts and configs. Pre-v7.5 retrieval baseline. |
| `val_candidate_sets.jsonl` | 2.2 MB | 10 lines (one per val query) — full candidate-set dumps. |
| `val_dropped_gold.csv` | 3.0 KB | Gold citations not reachable in the eval pool — e.g. `Art. 39 173.71`, `Art. 428 312.0` (val_001), `BGE 134 V 231 E. 5.1` (val_002). |
| `endgame_ablation_val.csv` | 931 B | Toggle ablation: baseline F1=0.0485. `use_llm_judge` adds +0.033 (to 0.0812). `use_reranker` adds +0.013. `use_bm25` HURTS by -0.0035. **Best F1 reached: 0.0812 — well below the < 0.10 ceiling noted in the experiment ledger.** |
| `endgame_channel_recall.csv` | 208 B | bm25 R@1000=0.194, vector R@1000=0.248, statute R@1000=0.008, case R@1000=0.000 — confirms statute/case channels are nearly dead in this config. |
| `endgame_val_per_query.csv` | 474 B | Per-query P/R/F1/elapsed_s. F1 range 0.010 (val_003) to 0.110 (val_004). Slowest query 360s (val_003). |

**Producer:** `notebooks/04_pre_v75_pipeline_iterations/endgame_colab_full_pipeline.ipynb` and the `pipeline_iteration_latest_*` family. Confirmed in `_inventory/pipeline_iteration_latest_fixed_v{1,2,3}.md`.
**Skill verdict:** all numbers here are pre-v7.5 baselines. The Macro F1 < 0.10 figure in `experiments` SKILL is sourced from these CSVs.

---

## G. Citation-graph-derived data — `data_insights/citation_graph_db_and_edges/` + `citation_patterns_and_priors/`

### G.1 Citation-graph DB & raw edges — `data_insights/citation_graph_db_and_edges/`

| File | Size | Notes |
|---|---|---|
| `citation_graph_extracted.sqlite` | **5.81 GB** | Patched citation graph; **96.0% gold-as-node coverage** (was 92.9%). Useful as **tie-breaker only**, NOT as co-prediction expander (Obs 2: 99.41% of gold-pairs have no graph edge). |
| `citation_graph_extracted.sqlite.bak_pre_backref` | 2.03 GB | Pre-backref-patch backup. |
| `citation_graph_extracted.sqlite.bak_pre_caselevel` | 2.09 GB | Pre-case-level-patch backup. |
| `laws_de_links.json` | 2.7 MB | **45,165 edges** in laws_de — `source_to_references` mapping. |
| `court_considerations_links.json` | 159.7 MB | **4,554,397 edges** in court considerations. |
| `laws_de_citations.json` | 5.5 MB | Per-row citation extraction summary; 175,933 rows / 32,277 unique references / 45,165 edges. |
| `court_considerations_citations.json` | 69.2 MB | Court-side raw citation extractions. |
| `laws_de_classified_citations.jsonl` | 81.7 MB | One row per unique citation with `family` (statute_article / statute_section / court_bge / court_case / official_reference), `pattern`, `subfamily`, `segments`, `origin_mask` (Class A/B/C bitmask). |
| `court_considerations_classified_citations.jsonl` | 865.3 MB | Same shape for court-side. |

### G.2 Citation patterns & priors — `data_insights/citation_patterns_and_priors/`

| File | Size | Notes |
|---|---|---|
| `citation_patterns.json` | 5.9 KB | Naming-family pattern dictionary. Counts per dataset; pattern_counts: court_bge=1.1M, court_case=2.8M. |
| `posture_statute_prior.json` | 86.1 KB | Per-docket-posture statute prior. 2.44M kept edges / 4.16M scanned. Posture totals: 1B=119k, 1C=217k, 2C=443k. |
| `citation_pattern_analysis.md` | 2.5 KB | Human-readable pattern documentation: `statute_article`, `statute_section`, `court_bge`, `court_case`, `official_reference` families with segment decomposition. |
| `example.json` | 1.3 KB | Example records. |

**Producer:** `scripts/citation_extraction/` (graph extraction scripts) and the citation parser. The classified jsonl rows ground every downstream `class A/B/C` assignment in `gold_citation_coverage.csv` (covered in agent 4's scope).
**Consumer:** every retrieval channel that uses `statute_links`, `case_links`, `adjacent_law_links` in `unified_retrieval.sqlite`. Cited by `pipeline_end_to_end_*` notebooks (inventory hits in `pipeline_end_to_end_no_function_imports.md`, `..._native.md`, `..._clean_paths.md`, `..._base.md`, `..._explicit_paths.md`).

---

## H. Experimental / abandoned data (mix of locations)

### H.1 segment_lattice_v3 experiment — `artifacts/`

| File | Size | Notes |
|---|---|---|
| `segment_lattice_v3.sqlite` | — | Per-row lattice DB. |
| `val_segment_lattice_v3_candidate_sets.jsonl` | — | Per-query candidate sets. |
| `val_segment_lattice_v3_candidate_summary.csv` | — | Macro recall = ~0.13 (val_004/005/008/009/010 all 0.0; val_001=0.36, val_006=0.39). **court_recall = 0.0 across all 10 queries** — only catches laws. |
| `val_segment_lattice_v3_dropped_gold.csv` | — | Dropped gold list. |
| `val_segment_lattice_v3_planner_outputs.jsonl` | — | Planner outputs. |
| `test_segment_lattice_v3_<19-digit-id>` | 10 files | Test-side outputs (run IDs are ns-precision timestamps). |

**Producer:** `notebooks/12_early_funnel_pre_v7/early_segment_lattice_funnel_v3.ipynb` + `scripts/segment_lattice_v3.py` (referenced in cell 1 of the notebook). Inventory confirmed: `early_segment_lattice_funnel_v3.md`, plus references in `anchor_funnel_v7_*` and `INDEX.md`.
**Verdict:** abandoned. The summary CSV shows 0% court recall across val and ~0.13 macro law recall — far below the v7.5 funnel that replaced it.

### H.2 gold_bank experiment — `artifacts/`

| File | Size | Notes |
|---|---|---|
| `val_gold_bank_candidate_sets.jsonl` | — | Candidate sets. |
| `val_gold_bank_candidate_summary.csv` | — | **Recall = 1.0 on all 10 val queries** — but this is because the gold-bank contains the gold by construction (oracle-style). 2111 candidates/query (1958 law + 153 court). |
| `val_gold_bank_train_val_candidate_{sets.jsonl,summary.csv}` | — | Variant including train gold-bank. |
| `val_gold_bank_val_candidate_{sets.jsonl,summary.csv}` | — | Val-only variant. |
| `test_gold_bank_train_val_candidate_{sets.jsonl,summary.csv}` | — | Test-side. |
| `val001_gold_court_enriched.jsonl` | — | val_001-specific gold court enrichment dump. |

**Producer:** `notebooks/12_early_funnel_pre_v7/early_recall_preserving_funnel.ipynb` (inventory has `active_gold_bank_splits`, `load_gold_bank`, `run_gold_bank_candidate_funnel` functions on lines 1720-1770).
**Verdict:** abandoned as a retrieval primitive — the 100% recall is by construction (gold-bank seeded with gold). Useful only as a recall ceiling reference, not a deployable channel. Test-side gold-bank in particular would be cheating on the leaderboard.

### H.3 val_001-specific case studies — `research/` + `drive_sync/swiss_law/v7_pool_recall_089_and_per_query/`

| File | Location | Notes |
|---|---|---|
| `val_001_gold_enrichments.json` | `research/` | Per-gold-citation LLM enrichment (english_summary, legal_rule, applicability_conditions, concepts_en). |
| `val_001_static_only_court_enrichments.json` | `research/` | Static-only court enrichment field samples. |
| `val_001_gold_and_enrichment_signals.md` | `research/` | val_001 analysis writeup. |
| `val_001_field_cardinalities.md` | `research/` | Field cardinality stats. |
| `val_001_gold_missed_diagnosis.json` | `drive_sync/swiss_law/v7_pool_recall_089_and_per_query/` | Per-gold root-cause: GATED_OUT / NO_CHANNEL_HIT / RRF_RANK_TOO_LOW. |
| `gold_missed_diagnosis.json` | `drive_sync/swiss_law/v7_pool_recall_089_and_per_query/` | Aggregate gold-missed diagnosis across queries. |

**Memory rule (from `feedback_no_query_specific_hardcoding.md`):** these are case studies. **Never inject val_001-specific knowledge into the pipeline** — must generalize to any val/production query.

### H.4 Stage B diagnostics — `research/stage_b_diagnostics_and_ceilings/`

| File | Notes |
|---|---|
| `REPORT.md` | **TL;DR: best CPU-only Stage B is 3-way weighted RRF of (fusion, signal-count-boost, full-text BM25). Macro R@2000 = 0.644 (+0.033 over fusion). Min per-q R@2000 stalls at 0.255 for val_003. The 0.7-per-query floor at K=2000 is structurally unachievable on this snapshot.** Met at R@3000 (0.709 macro). |
| `final_stage_b.py` | Recommended Stage B. |
| `s0_baseline.py` ... `s12_protected_rerank.py` | 13 strategies tried (#0 fusion, #1 PRF, #2 multichannel, #3 lift-score, #4 signal-count, #5 ensemble, #6 multiplicative, #7 HyDE-BM25, #8 full-text BM25, #9 3-way RRF, #10 neighborhood, #11 union-restricted RRF, #12 protected+rerank). |
| `diag_ceiling.py`, `diag_channel_power.py`, `diag_deep_gold.py`, `diag_ensemble_ceiling.py`, `diag_targets_alignment.py`, `diag_val003.py` | Diagnostics; val_003 has 11 gold not in 50k pool + 24 at fusion ranks 2k-44k. |
| `snapshot_cache.pkl`, `full_text.pkl`, `helpers.py`, `load_snapshot.py`, `load_full_text.py` | Reusable plumbing (~150 MB + ~280 MB cached pickles). |

**Verdict:** the constraint is the snapshot, not the reranker. Unblockers per REPORT.md: (1) re-run v7.5 with `topk_final=100k`, (2) cross-encoder rerank on the 44k pool, (3) larger LLM target expansion (val_003 gold spans 40+ statutes vs ~20 in `all_targets.json`).

### H.5 LegalMALR cascade Stage A output

| File | Size | Notes |
|---|---|---|
| `drive_sync/swiss_law/legalmalr_cascade_stage_a_output/stage_a_rerank.json` | — | 3 queries (val_001, val_004, val_006); rerank_topn=1000. val_001 top-1000 gold recall = 14/42 = 0.333; val_004 top-1000 gold recall = 2/10 = 0.20. |
| `drive_sync/swiss_law/legalmalr_cascade_per_query_cache/rerank/val_001.json` | — | Per-query rerank cache. |
| `drive_sync/swiss_law/legalmalr_cascade_per_query_cache/rerank/val_004.json` | — | |
| `drive_sync/swiss_law/legalmalr_cascade_per_query_cache/rerank/val_006.json` | — | |

**Producer:** `notebooks/01_current_direction_cascade_rerank_precision/cascade_legalmalr_v1_{base,optimized,poc}.ipynb`. Inventory hits confirmed.
**Status:** Stage A proof-of-concept only ran 3 of 10 queries. Stage A's R@1000 (14/42 on val_001) is **below** the v7.5 funnel's R@1000 (15/42 = 0.357 per `summary.json`). Cascade approach was deprioritized.

---

## I. Omnilex pipeline data — `drive_sync/omnilex_competition/`

| File / folder | Size | Notes |
|---|---|---|
| `abbrev_translations.json` | 593 KB | Legal abbreviation expansions (DE/FR/IT/EN). |
| `retrieval_knowledge_base_query_translations/query_translations_trainval.json` | 1.6 MB | Train+val query translations EN→DE/FR/IT. |
| `retrieval_pipeline_qwen3_reranker_cache/reranker_Qwen_Qwen3-Reranker-8B_v3.json` | 175 KB | Cached Qwen3-Reranker-8B scores. Distinct from the `hybrid_rerank_shootout_cache/scores_qwen3.npz` cache used in agent 4's scope. |
| `retrieval_other_artifacts/` | — | bm25 v1/v2 indices, faiss v1/v2, corpus parquet v1/v2, corpus embeddings v1/v2 (`.npy`), `article_translations_de_en.json`, `eval_results_val.csv`, `predictions_{test,val}.csv`, `knowledge_base_optimized_hybrid_retrieval/`, `laws_knowledge_base.jsonl`, `pipeline_cache/`, `pipeline_output_v3/`. |
| `swiss_law_pipeline_source_code/` | — | Omnilex pipeline Python source (NOT a notebook). |

### I.1 swiss_law_pipeline_source_code/ layout

| Folder/file | Purpose |
|---|---|
| `agent/` | `llm_backend.py`, `verifier.py`, `prompts/`. |
| `configs/default.json` | Pipeline config. |
| `retrieval/` | `bm25_artifact.py`, `explicit_citations.py`, `graph_retriever.py`, `sparse_retriever.py`. |
| `scoring/` | `confidence.py`. |
| `pipeline.py`, `submit.py` | Orchestration. |
| `stage1_query_analysis.py` ... `stage5_verify_and_score.py` (+ `stage4b_cross_encoder.py`) | 5-stage pipeline: query analysis → MAS retrieval → graph expansion → LLM reranker + cross-encoder → verify+score. |
| `bm25_v2_index.pkl` + `bm25_v2_index_parts.zip` | Cached BM25 index. |
| `citation_graph.pkl`, `citation_lookup.pkl`, `citation_signal_lookup.pkl` | Cached graph artefacts. |
| `gold_cocitation_prior_train.pkl`, `gold_cocitation_prior_trainval.pkl` | Co-citation priors. |
| `colab_runtime_patches.py` | Colab-specific runtime fixes. |

**Status:** parallel pipeline from the Omnilex competition. Distinct from the Swiss-law canonical v7.5 pipeline. The K-sweep submissions (§E) come from this code.

---

## J. Configs — `drive_sync/swiss_law/threshold_and_runtime_configs/`

| File | Contents |
|---|---|
| `threshold.json` | **103 B — small but CRITICAL.** `{"threshold": 0.65, "macro_f1_val": 0.498, "tuned_on_split": "val"}`. The final-pick score threshold; whatever produces `submission_*.csv` filters by this. Macro F1 0.498 on val is the **current measured pipeline ceiling on val** — well above the < 0.10 endgame baseline but still far from the ~0.78 target seen in `reference_F1_0.777_Untitled75.ipynb`. |

**Producer:** likely `precision_v1_final.ipynb` (threshold sweep against val).
**Consumer:** submission generation. If this threshold drifts, every submitted CSV changes.

---

## K. A6 fine-tuned models — `drive_sync/swiss_law/A6_{finetuned,simcse}_legal_swiss_model/`

| Path | Files |
|---|---|
| `A6_finetuned_legal_swiss_model/` | `model.safetensors` (1.62 GB), `config.json`, `config_sentence_transformers.json`, `modules.json`, `sentence_bert_config.json`, `tokenizer.json` (9.4 MB), `tokenizer_config.json`, `1_Pooling/`, README (42 KB). |
| `A6_simcse_legal_swiss_model/` | Same set; `model.safetensors` (1.62 GB), README (32 KB). |

**Producer (confirmed via inventory grep):** `notebooks/11_early_experiments_exp_A/exp_A6_legal_roberta_finetune.ipynb`. Inventory MD (`exp_A6_legal_roberta_finetune.md`) shows the notebook installs `sentence-transformers>=3.1.0`, mounts Drive, and runs a sentence-transformers fine-tune over legal_roberta base. Both folders share identical model.safetensors size — the `simcse` variant is the SimCSE-style training, the `finetuned` variant is the supervised contrastive training; both are sentence-transformers checkpoints with a pooling head.

**Status:** early experiment (exp_A series, 2026-04-22). Not part of the current canonical pipeline (which uses Qwen3-Embedding-8B for the dense channel, not A6). No experiment-ledger entry — the exp_A line was retired before the endgame baseline.

---

## Critical-data callouts (re-emphasized)

1. **`drive_sync/swiss_law/v7_pool_recall_089_and_per_query/snapshot/corpus_snapshot.json.gz` (110 MB)** is THE recall-0.89 pool. Every downstream experiment scores against it. Losing this file forces re-running the v7.5 funnel from scratch (multi-hour Colab job).
2. **`cache_endgame/judge/*` is TOXIC** — 87% rubber-stamp YES. Must be deleted (not extended) before any future LLM-judge experiment.
3. **`threshold.json`** drives the final-pick. 103 bytes; high blast-radius. Tuned to val Macro F1 = 0.498.
4. **`research/stage_b_diagnostics_and_ceilings/REPORT.md`** quantifies the structural blocker: val_003 cannot reach R@2000 = 0.7 on this 50k snapshot, regardless of reranker. Unblockers require either widening to topk_final=100k or cross-encoder reranking.
5. **segment_lattice_v3 and gold_bank are abandoned.** Their `artifacts/*` outputs are retained as historical reference only — neither should be re-introduced into the current cascade.
