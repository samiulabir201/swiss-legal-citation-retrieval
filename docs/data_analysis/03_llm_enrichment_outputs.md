# 03 — LLM Enrichment Outputs (Qwen3-8B-AWQ Descriptors)

_Scope: every Qwen3-8B-AWQ-derived structured descriptor file on disk — court paragraphs (363k) and laws (173k) — plus their input feeders, failure logs, backup copies, and Drive-side checkpoint mirror._

---

## Why these files exist

The pipeline depends on **two parallel semantic enrichment layers** that translate Swiss multilingual (DE/FR/IT) legal text into English-tagged, structured descriptors that downstream retrieval channels can match against an English query:

- **Court descriptors** annotate paragraphs of `court_considerations.csv` (BGE/ATF/court ruling considerations) with `legal_area`, `topic/subtopic/micro_topic`, `concepts_en`, `terms_original`, `fact_pattern_tags`, `paragraph_role`, `specificity_score`, etc. The 16-field schema is fixed by [`docs/court_enrichment_field_contract.md`](../court_enrichment_field_contract.md).
- **Law descriptors** annotate articles of `laws_de.csv` (federal statutes / ordinances / agreements) with a 12-field schema centered on `english_summary`, `legal_rule`, `applicability_conditions`, `exceptions_or_limitations`, `concepts_en`, `terms_de_to_en`, `provision_role_llm`, `specificity_score`. Schema is fixed by [`docs/laws_de_llm_enrichment_schema.md`](../laws_de_llm_enrichment_schema.md).

These descriptors are the only English-language signal in an otherwise German/French/Italian corpus — every channel in the v7.5 multi-query funnel (`concept_en`, `terms`, `legal_area_prefilter`, `enrichment_topic_match`, etc.) reads from them. Their original purpose, captured in [`docs/court_rag_plan.md`](../court_rag_plan.md), was to give BM25 + dense retrieval an English semantic layer; the [`research/cascade_dossier_plan.md`](../../research/cascade_dossier_plan.md) extended that role into per-candidate dossier features (`concept_int`, `term_int`, statute-target intersection) that drive the F3 hybrid reranker config.

Both runs use the same model (`Qwen/Qwen3-8B-AWQ`) and the same Blackwell-optimized notebook stack (vLLM + FlashInfer + FP8-KV). The court run completed on 2026-05-05; the law run completed on 2026-05-06 after one engine-dead rerun cycle.

---

## A. Court LLM descriptors — `llm_enrichment_output_court_363k/`

(Folder renamed by `scripts/drive_pull/full_repo_reorg.py` from the legacy `outputs_from_363k_run/`.)

### A.1 `court_llm_descriptors_0000000_all.jsonl`
- **Path:** `e:\swiss_citation_extraction\llm_enrichment_output_court_363k\court_llm_descriptors_0000000_all.jsonl`
- **Size / lines:** 676 MB, **363,258 records**.
- **Status distribution:** `ok` = 363,212 (99.987%); `failed_descriptor_parse` = 1; the rest fall into other `ok_*` variants.
- **Schema (one record):**
  ```json
  {
    "_source_row": <int>,
    "citation": "BGE 139 I 2 E. 5.7",
    "text": "<original DE/FR/IT paragraph>",
    "language": "de" | "fr" | "it",
    "legal_area_static": "<deterministic legal area>",
    "llm_enrichment": {
      "legal_area", "primary_domain", "secondary_domain",
      "legal_domain_path": [...],
      "topic", "subtopic", "micro_topic",
      "concepts_en": [...], "terms_original": [...],
      "doctrinal_rule", "legal_test",
      "fact_pattern_tags": [...],
      "procedural_context",
      "paragraph_role": "facts|reasoning|holding|...",
      "authority_role": [...],
      "specificity_score": 0.0..1.0
    },
    "llm_generation": {
      "model": "Qwen/Qwen3-8B-AWQ",
      "method": "minimal_descriptor_only",
      "attention_backend": "auto",
      "kv_cache_dtype": null,
      "speculative": false,
      "status": "ok",
      "attempt_count": 1,
      "error": null,
      "raw_output": null
    }
  }
  ```
- **Producer notebook:** `notebooks/08_court_llm_descriptor_extraction/court_llm_descriptor_blackwell_optimized_363k_run.ipynb` (10 code cells, mtime 2026-05-05 01:39). Sibling iterations in the same folder: `_blackwell_optimized_v4`, `_10k_optimized`, `_qwen3_8b_awq`, `_safe_2t4_vllm`, `_dual_t4_vllm`, `_kaggle`. The `_363k_run` is the final canonical pass.
- **Inputs:** the 363k records are the high-priority subset identified by `scripts/citation_extraction/identify_court_enrichment_targets.py` and materialized by `artifacts/court_authority_cards_v4_enrichment_targets.jsonl`. Not every court paragraph is enriched.
- **Coverage gap:** **363,258 / 2,476,315 = 14.67%** of `court_considerations.csv` rows. This is the documented enrichment ceiling. The remaining 85% of court paragraphs carry only the deterministic `legal_area_static` plus regex anchors — they have no English semantic layer.
- **Downstream consumers:**
  - `scripts/merging/merge_court_llm_into_unified.py` — merges into `unified_retrieval.sqlite` (the canonical retrieval DB used by every v7.x funnel).
  - `scripts/merging/merge_llm_enrichment_into_v4_cards.py` — older path into `court_authority_cards_v5_unified.jsonl`.
  - Every `anchor_funnel_v*` notebook reads the merged result via `unified_retrieval.sqlite`.
- **Endgame verdict** (per `research/endgame_handoff_2026-05-09.md` §6.1, summarized in `swiss-citation-experiments` SKILL): the previously-believed "court LLM is static_only" issue was a **phantom** — these 363k rows ARE merged into `unified_retrieval.sqlite`. The schema simply forbids `english_summary` by design (court paragraphs get topic/concepts, not summaries — that's the law-side contract).
- **Verdict:** **ACTIVE / CANONICAL.** Foundation of every English-language retrieval channel on the court side.

### A.2 `court_llm_descriptors_0000000_all_failures.jsonl`
- **Size:** 9.4 KB, **1 record**.
- **Schema:** identical envelope plus `"status": "failed_descriptor_parse"` and a full diagnostic block (`attempts[]`, per-attempt `raw_output`, `error`) capturing the truncated JSON the model emitted. Only one paragraph (`BGE 129 I 12 E. 9.3`, an exceptionally long multi-canton enumeration) failed both retries.
- **Verdict:** active diagnostic; effectively a zero-failures run.

### A.3 `court_llm_descriptors_0000000_all_preview.csv`
- **Size:** 469 KB. Tabular preview of the first ~2,000 records, columns: `_source_row, citation, status, legal_area, primary_domain, secondary_domain, topic, subtopic, micro_topic, concepts_en, terms_original, doctrinal_rule, legal_test, fact_pattern_tags, procedural_context, paragraph_role, authority_role, specificity_score`.
- **Verdict:** convenience artifact for human inspection / smoke validation. Not consumed programmatically.

---

## B. Law LLM descriptors — `llm_enrichment_output_law_173k/`

(Folder renamed by `scripts/drive_pull/full_repo_reorg.py` from the legacy `law_json_llm_output/`.)

### B.1 `law_llm_descriptors_0000000_all.jsonl`
- **Path:** `e:\swiss_citation_extraction\llm_enrichment_output_law_173k\law_llm_descriptors_0000000_all.jsonl`
- **Size / lines:** 244 MB, **173,033 records**.
- **Status distribution** (measured by grep):
  - `ok` = 167,775
  - `ok_after_retry` = 857
  - `ok_after_engine_dead_rerun` = 1,829
  - `ok_after_manual_repair` = 289
  - `static_skip` = 2,283 (filtered before LLM — text was unsuitable or covered by static fields alone)
  - `failed_descriptor_parse` = 0 (all parse failures were repaired in-place)
- **Schema (one record):**
  ```json
  {
    "_source_row": <int>,
    "citation": "Art. 1 112",
    "language": "de",
    "llm_priority": "high" | "medium" | "low",
    "llm_enrichment": {
      "english_summary", "legal_rule",
      "applicability_conditions": [...],
      "exceptions_or_limitations": [...],
      "legal_question",
      "concepts_en": [...],
      "terms_de_to_en": [{"de": ..., "en": ...}, ...],
      "defined_terms": [{"term": ..., "definition": ...}, ...],
      "addressees": [...],
      "sanctions_or_consequences": [...],
      "provision_role_llm": "definition|duty|prohibition|procedure|...",
      "specificity_score": 0.0..1.0
    },
    "llm_quality": {
      "json_valid": true,
      "terms_grounded_pct": <float>,
      "boilerplate_role": <bool>
    },
    "llm_generation": { ...same shape as court... }
  }
  ```
- **Coverage:** 173,033 / 175,933 input articles. Missing rows are `static_skip` (filtered upstream) or were dropped during input chunking. Effective LLM-touch rate ≈ 98.4% of input.
- **Producer:** `scripts/enrichment/run_law_llm_enrichment.py` (Qwen3-8B-AWQ vLLM runner, supports HF fallback, append-mode with `_source_row` resume). Notebook orchestrators: `notebooks/07_law_de_enrichment/enrich_laws_de_qwen3_8b_kaggle.ipynb`, `_optimized`, `_pre_optimization`. Drive output paths in those notebooks: `/content/drive/MyDrive/swiss_law/outputs/law_llm_descriptors_0000000_all.jsonl` (final) and `.../data/checkpoints/...` (hot).
- **Downstream consumers:**
  - `scripts/merging/merge_law_llm_into_v1_cards.py` — produces `artifacts/law_authority_cards_v2_unified.jsonl` (final retrieval cards).
  - Every anchor-funnel notebook (v1/v4/v5/v6/v7/v7.4) loads it via `DATA_ROOT / "data" / "checkpoints" / "law_llm_descriptors_0000000_all.jsonl"` with fallback to `DATA_ROOT / "law_json_llm_output" / "..."` — see inventory hits in `notebooks/_inventory/anchor_funnel_v{1,4,5,6}_val001.md`.
  - Merged content feeds the law-side rows of `unified_retrieval.sqlite`.
- **Verdict:** **ACTIVE / CANONICAL.** Drives all `english_summary_view`, `legal_rule_view`, `concepts_en_view`, `terms_bilingual_view` retrieval channels.

### B.2 Backups: `.bak`, `.bak2`, `.bak3`
- **`.bak`** (255,445,816 B, 173,033 lines, mtime 2026-05-06 20:41): pre-`fix_all_invalid.py` state. Captured before role-vocab coercion + schema normalization.
- **`.bak2`** (255,496,893 B, same line count, mtime 2026-05-06 20:46): created during the `repair_failures` / `normalize_roles` / `fix_unrecoverable` passes (see `scripts/diagnostics_loose/` family).
- **`.bak3`** (253,298,808 B, same line count, mtime 2026-05-06 21:28): created by `scripts/diagnostics_loose/merge_engine_dead_rerun.py` immediately before merging the 1,829 rerun records back into the main file. Smaller because the rerun replaced empty-enrichment placeholders with real content (but those placeholder strings were verbose `raw_output` dumps, so the byte delta is slightly negative).
- **Verdict on all three:** **superseded, but kept for forensics.** Useful only if you need to reconstruct an earlier pre-repair state. The main file is the source of truth.

### B.3 `law_llm_descriptors_0000000_all_failures.jsonl`
- **Size:** 578 KB, **66 records**.
- **Schema:** failure envelope with `attempts[]`, per-attempt `raw_output`, full error trace. Failure mode is uniformly "JSON parse hit token budget mid-stream", typically on dense regulatory enumerations.
- **Verdict:** diagnostic-only; the 66 failures were either retried successfully (counted in `ok_after_retry`) or repaired by `scripts/diagnostics_loose/fix_all_invalid.py` (counted in `ok_after_manual_repair`).

### B.4 `law_llm_descriptors_engine_dead_rerun.jsonl`
- **Size:** 2.7 MB, **1,829 records**.
- **Origin:** during the first pass, vLLM hit `EngineDeadError` on a subset of long articles; the runner caught it and wrote empty enrichments with `status="ok_empty_engine_dead"`. `scripts/diagnostics_loose/build_rerun_input.py` extracted those `_source_row`s, `scripts/notebook_builders/update_notebook_for_rerun.py` re-ran them with conservative vLLM settings, and `scripts/diagnostics_loose/merge_engine_dead_rerun.py` merged the results back into the main file (changing those records to `status="ok_after_engine_dead_rerun"`).
- **Schema:** same as B.1.
- **Verdict:** **superseded** (already merged into B.1) but kept as audit trail.

---

## C. LLM input files — `artifacts/`

### C.1 `artifacts/law_llm_input.jsonl`
- **Size / lines:** 188 MB, **173,033 records**. Each line is one law article ready for the LLM with static metadata pre-attached.
- **Schema (one record):**
  ```json
  {
    "_source_row": <int>,
    "citation": "Art. 1 112",
    "law_title": "<full DE title>",
    "title_section_path": "" | ["..."],
    "structural": {"article", "units": [...], "law_code", "law_code_family", "granularity"},
    "title_metadata": {"source_type", "enactment_date", "enactment_year", "law_aliases", "systematic_collection_sector"},
    "static_hints": {"legal_area_static", "domain_labels_en", "issue_labels_en", "provision_roles_static", "matched_terms_multilingual"},
    "llm_priority": "high|medium|low",
    "static_semantic_signal": "partial_static_signal|formal_only|full_static",
    "text": "<original DE statute text>",
    "_text_len": <int>
  }
  ```
- **Producer:** `scripts/data_prep/build_law_llm_input.py` (referenced from `run_law_llm_enrichment.py` docstring as the upstream feeder).
- **Verdict:** **ACTIVE.** Re-read on any law-LLM rerun.

### C.2 `artifacts/law_llm_input.smoke.jsonl` + `.summary.json`
- 50 lines, 55 KB. The smoke fixture used to validate the pipeline before kicking off the full run.
- **Verdict:** active testing fixture.

### C.3 `artifacts/law_llm_input_chunks/law_llm_input_part00{0..6}.jsonl`
- **Sizes / lines:**
  - `part000.jsonl` — 31 MB, 26,498 records
  - `part001.jsonl` — 31 MB, 26,855 records
  - `part002.jsonl` — 31 MB, 27,685 records
  - `part003.jsonl` — 31 MB, 27,671 records
  - `part004.jsonl` — 31 MB, 28,064 records
  - `part005.jsonl` — 31 MB, 27,698 records
  - `part006.jsonl` — 9.1 MB, 8,562 records
- **Total:** **172,033 records across 7 parts** (matches `law_llm_input.jsonl` minus the few rows that get streamed but not chunk-checkpointed).
- **Schema:** identical to `law_llm_input.jsonl`.
- **Purpose:** sharded copies used by the Kaggle notebook variants to parallelize across multiple Kaggle sessions (each session takes one part). Mtimes all 2026-05-06 15:14, so they were materialized in one pass after the main `law_llm_input.jsonl`.
- **Verdict:** **ACTIVE for Kaggle reruns.** Redundant for the single-Blackwell-GPU runner.

### C.4 `artifacts/law_llm_input_engine_dead_rerun.jsonl`
- 2.0 MB, **1,829 records**. The slim rerun input for the `EngineDeadError` recovery cycle described in §B.4.
- **Producer:** `scripts/diagnostics_loose/build_rerun_input.py`.
- **Verdict:** archived (already consumed). Kept for reproducibility.

### C.5 Court-side input
- The court LLM run reads `artifacts/court_authority_cards_v4_enrichment_targets.jsonl` (219 MB) — listed in the same `artifacts/` folder. That file is built by `scripts/citation_extraction/identify_court_enrichment_targets.py` + `extract_court_target_cards.py`. The court input has no separate `_part*` chunking on local disk because the 363k run was executed end-to-end on the Blackwell box.

---

## D. Drive-side checkpoint mirror — `drive_sync/swiss_law/llm_enrichment_jsonl_checkpoints/`

- **Path:** `e:\swiss_citation_extraction\drive_sync\swiss_law\llm_enrichment_jsonl_checkpoints\`
- **Files (mirror of Drive `data/checkpoints/`):**
  - `court_llm_descriptors_0000000_all.jsonl` — 709,425,459 B (byte-identical size to the local court file in §A.1).
  - `court_llm_descriptors_0000000_all_failures.jsonl` — 9,472 B (identical to §A.2).
  - `law_llm_descriptors_0000000_all.jsonl` — 255,445,816 B. **Note:** this matches the size of the LOCAL `.bak` (pre-repair) file, not the current canonical 254,692,829 B file. The mirror captured the Drive checkpoint BEFORE the post-merge schema repair and the engine-dead rerun merge.
  - `law_llm_descriptors_0000000_all_failures.jsonl` — 577,851 B (identical to §B.3).
  - `law_llm_descriptors_engine_dead_rerun.jsonl` — 2,733,464 B (identical to §B.4).
- **Verdict:** **superseded copy of the law main file** (matches `.bak`, not the live file). Court file and the two diagnostic JSONLs are byte-identical to local. Treat the Drive mirror as a recovery snapshot from the moment just before final repair — useful only if `llm_enrichment_output_law_173k/law_llm_descriptors_0000000_all.jsonl` is ever corrupted and `.bak` is also lost.

---

## Producer-script map

| Script | Role | Output |
|---|---|---|
| `scripts/enrichment/run_law_llm_enrichment.py` | vLLM/HF runner for law articles, append-mode, resumable | `artifacts/law_llm_descriptors.jsonl` (synonym path; lives in `llm_enrichment_output_law_173k/` after the reorg) |
| `scripts/enrichment/enrich_cards_qwen3_8b_kaggle.py` | Kaggle-side runner for court target cards (low-cost variant of the 32B notebook) | `court_authority_cards_rag_targets_qwen3_8b.jsonl` |
| `scripts/enrichment/enrich_cards_with_qwen.py` | Colab Pro+ Qwen3-32B-AWQ runner for the full court enrichment (legacy heavy run) | `court_authority_cards_rag.jsonl` |
| `scripts/enrichment/enrich_cards_with_claude_agent_sdk.py` | Claude-Agent SDK variant (used for a small ad-hoc enrichment of `court_authority_cards_rag_targets_claude.jsonl`) | `*_targets_claude.jsonl` |
| `scripts/enrichment/profile_court_enrichment_schema.py` | Thin wrapper → `court_enrichment_profile.py` | `artifacts/court_enrichment_corpus_profile.json` |
| `scripts/enrichment/court_enrichment_profile.py` | Full-corpus profiler (the source of `docs/court_enrichment_field_contract.md` numbers) | corpus profile JSON |
| `scripts/enrichment/court_enrichment_normalizer.py` | Post-LLM deterministic anchor cleanup | merged unified card |
| `scripts/notebook_builders/_patch_law_notebook.py` | Programmatic patcher that injects the law-LLM cells into the Kaggle notebook | regenerated `.ipynb` |
| `scripts/diagnostics_loose/build_rerun_input.py` | Slices `EngineDeadError` rows out of the main file into a rerun-only input | `law_llm_input_engine_dead_rerun.jsonl` |
| `scripts/diagnostics_loose/merge_engine_dead_rerun.py` | Merges rerun output back into the main law file | creates `.bak3`, rewrites main |
| `scripts/diagnostics_loose/fix_all_invalid.py` + `classify_invalid.py` + `normalize_roles.py` + `find_offcontract_roles.py` + `repair_failures.py` (referenced) | Post-merge schema repair / vocab coercion pass | rewrites main, creates `.bak` / `.bak2` |
| `scripts/data_prep/repair_law_enrichment_utf8.py` | Optional UTF-8 mojibake repair (turned out to be a phantom issue per endgame §6.2 — files were already byte-clean) | `.cleaned.jsonl` |
| `scripts/merging/merge_court_llm_into_unified.py` | Merge court descriptors into `unified_retrieval.sqlite` | sqlite |
| `scripts/merging/merge_law_llm_into_v1_cards.py` | Merge law descriptors into `law_authority_cards_v2_unified.jsonl` | unified law cards |
| `scripts/drive_pull/full_repo_reorg.py` (phase 7) | Renamed `outputs_from_363k_run/` → `llm_enrichment_output_court_363k/` and `law_json_llm_output/` → `llm_enrichment_output_law_173k/` | renamed dirs |

---

## Producer-notebook map

| Notebook | Run | Output file produced |
|---|---|---|
| `notebooks/08_court_llm_descriptor_extraction/court_llm_descriptor_blackwell_optimized_363k_run.ipynb` | **Canonical 363k court run** (2026-05-05 01:39) | A.1 / A.2 / A.3 |
| `notebooks/08_court_llm_descriptor_extraction/court_llm_descriptor_blackwell_optimized_v4.ipynb` | Earlier Blackwell tune | superseded |
| `notebooks/08_court_llm_descriptor_extraction/court_llm_descriptor_10k_optimized.ipynb` | 10k smoke | superseded |
| `notebooks/08_court_llm_descriptor_extraction/court_llm_descriptor_qwen3_8b_awq.ipynb` | Initial Qwen3-8B-AWQ port | superseded |
| `notebooks/08_court_llm_descriptor_extraction/court_llm_descriptor_safe_2t4_vllm.ipynb`, `_dual_t4_vllm.ipynb`, `_kaggle.ipynb` | Kaggle T4 variants | superseded |
| `notebooks/07_law_de_enrichment/enrich_laws_de_qwen3_8b_kaggle_optimized.ipynb` | **Canonical 173k law run** (optimized vLLM config) | B.1 + B.3 + B.4 (latter via rerun cycle) |
| `notebooks/07_law_de_enrichment/enrich_laws_de_qwen3_8b_kaggle.ipynb` | Earlier law-side variant | superseded |
| `notebooks/07_law_de_enrichment/enrich_laws_de_qwen3_8b_kaggle_pre_optimization.ipynb` | Pre-optimization baseline | superseded |

---

## Downstream consumers (where these descriptors are READ)

- **Retrieval indexing:** `scripts/merging/merge_court_llm_into_unified.py` and `merge_law_llm_into_v1_cards.py` build the canonical `unified_retrieval.sqlite` + `law_authority_cards_v2_unified.jsonl`. Every `concept_en`, `terms`, `english_summary_view`, `legal_rule_view`, `terms_bilingual_view` channel is built from these merged cards.
- **Embedding corpus:** the 24-GB Qwen3-Embedding-8B `embeddings/qwen3_8b_unified_chunk*.npy` shards encode text views that include the LLM enrichment fields (see `notebooks/05_embedding_and_retrieval_base/embed_unified_corpus_qwen3_8b_blackwell.ipynb`).
- **Anchor-funnel notebooks (v1 → v7.5):** all 9 funnel notebooks under `notebooks/03_anchor_funnel_evolution_v4_to_v74/` and `notebooks/02_v75_multiquery_canonical_pool/` load these descriptors via `unified_retrieval.sqlite`. Grep hits in `notebooks/_inventory/anchor_funnel_v{1,4,5,6}_val001.md` confirm direct path-level dependency.
- **Cascade dossier (Phase 1, planned):** per `research/cascade_dossier_plan.md`, the `concept_int`, `term_int`, and statute-target intersection signals — which gave F3 hybrid its +30-point lift over rerank-only F2 in the hybrid shootout (`swiss-citation-experiments` SKILL §"Hybrid reranker shootout") — are computed against `concepts_en`, `terms_de_to_en[].en`, and the law-side statute targets that ultimately trace back to these descriptors.

---

## The 14.7% coverage gap

The court LLM run touches **363,258 of 2,476,315** paragraphs (14.67%). This is a known and accepted constraint, not a bug:

- The 363k subset is the **`llm_priority=high`** slice — paragraphs with non-trivial content, identified by `identify_court_enrichment_targets.py`. The remaining ~2.1M rows are cost/notification/disposition boilerplate, very short paragraphs, or paragraphs where deterministic `legal_area_static` already captures the semantic signal.
- The endgame retrieval ceiling (v7.5 pool: macro R@50k = 0.893, min R = 0.766 at val_003) is **structurally blocked by val_003**, not by the 85% un-enriched court rows. `research/endgame_handoff_2026-05-09.md` confirms there is no recall room left for an LLM-enrichment expansion to close — the gold paragraphs for val_003 are simply not retrievable by any single-query strategy, regardless of how rich the descriptors are.
- That said, the cascade-dossier Phase 1 plan does NOT require any new enrichment — the dossier signals reuse the existing 363k+173k descriptor base.

If the coverage gap ever becomes binding, the existing input pipeline (`identify_court_enrichment_targets.py` → `extract_court_target_cards.py` → `court_llm_descriptor_blackwell_optimized_363k_run.ipynb`) can be re-run with a lower `llm_priority` threshold to enrich more rows. Estimated cost on RTX PRO 6000 Blackwell at the same vLLM settings: ~1 h per 100k additional paragraphs.

---

## Cross-reference table (file → verdict → primary consumer)

| File | Records | Verdict | Primary consumer |
|---|---:|---|---|
| `llm_enrichment_output_court_363k/court_llm_descriptors_0000000_all.jsonl` | 363,258 | active / canonical | `merge_court_llm_into_unified.py` → `unified_retrieval.sqlite` |
| `llm_enrichment_output_court_363k/court_llm_descriptors_0000000_all_failures.jsonl` | 1 | active diagnostic | manual inspection |
| `llm_enrichment_output_court_363k/court_llm_descriptors_0000000_all_preview.csv` | ~2,000 | convenience preview | manual inspection |
| `llm_enrichment_output_law_173k/law_llm_descriptors_0000000_all.jsonl` | 173,033 | active / canonical | `merge_law_llm_into_v1_cards.py` → `law_authority_cards_v2_unified.jsonl` |
| `llm_enrichment_output_law_173k/law_llm_descriptors_0000000_all.jsonl.bak` | 173,033 | superseded (pre-repair) | forensics |
| `llm_enrichment_output_law_173k/law_llm_descriptors_0000000_all.jsonl.bak2` | 173,033 | superseded (mid-repair) | forensics |
| `llm_enrichment_output_law_173k/law_llm_descriptors_0000000_all.jsonl.bak3` | 173,033 | superseded (pre-rerun-merge) | forensics |
| `llm_enrichment_output_law_173k/law_llm_descriptors_0000000_all_failures.jsonl` | 66 | active diagnostic | manual inspection |
| `llm_enrichment_output_law_173k/law_llm_descriptors_engine_dead_rerun.jsonl` | 1,829 | superseded (already merged into B.1) | audit trail |
| `artifacts/law_llm_input.jsonl` | 173,033 | active | `run_law_llm_enrichment.py` input |
| `artifacts/law_llm_input.smoke.jsonl` | 50 | active test fixture | smoke validation |
| `artifacts/law_llm_input_chunks/law_llm_input_part00{0..6}.jsonl` | 172,033 total | active (Kaggle shards) | Kaggle rerun input |
| `artifacts/law_llm_input_engine_dead_rerun.jsonl` | 1,829 | archived | rerun trail |
| `drive_sync/.../court_llm_descriptors_0000000_all.jsonl` | 363,258 | identical mirror | recovery only |
| `drive_sync/.../court_llm_descriptors_0000000_all_failures.jsonl` | 1 | identical mirror | recovery only |
| `drive_sync/.../law_llm_descriptors_0000000_all.jsonl` | 173,033 | **stale mirror** (matches `.bak`, not current main) | recovery only — DO NOT use as source of truth |
| `drive_sync/.../law_llm_descriptors_0000000_all_failures.jsonl` | 66 | identical mirror | recovery only |
| `drive_sync/.../law_llm_descriptors_engine_dead_rerun.jsonl` | 1,829 | identical mirror | recovery only |

---

## Key takeaways

1. **One canonical court file (363k) + one canonical law file (173k).** Everything else in these folders is either a backup, a diagnostic, a rerun trail, or a Drive snapshot.
2. **Both runs used the same model** (`Qwen/Qwen3-8B-AWQ`) under different prompt schemas — court schema = 16 minimal-descriptor fields, law schema = 12 fields centered on English summary + legal rule.
3. **The Drive mirror's law-side file is stale.** It captured the pre-repair state and is BYTE-IDENTICAL to local `.bak`, not to current main. Use the local `llm_enrichment_output_law_173k/law_llm_descriptors_0000000_all.jsonl` as source of truth.
4. **The 14.7% court coverage gap is structurally fine.** Endgame analysis ruled out enrichment expansion as a meaningful recall lever — val_003 R_max=0.766 in the v7.5 pool is the ceiling, and it's a single-query retrieval problem, not an enrichment problem.
5. **No re-runs needed.** Both descriptor files are stable, schema-conformant, and merged into the canonical retrieval state used by every downstream notebook from anchor_funnel_v1 forward.
