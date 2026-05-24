# Authority Cards — Court and Law (data inventory)

Scope: all court-side and law-side authority-card files, their chunked build intermediates, the enrichment-target / target-card subsets, the Claude-side RAG output, the `choice_cards_v3.json` lattice options, and the `chunkNN.log` build logs in `e:\swiss_citation_extraction\artifacts\`.

Conventions used below:
- "Active" = currently referenced by canonical retrieval / endgame docs (`README.md`, `endgame_handoff_2026-05-09.md`, `swiss-citation-data` skill).
- "Superseded" = a later artifact in the same family supersedes it.
- "Build-time intermediate" = produced as a step in `scripts/build_pipeline/build_court_authority_cards.py` or `build_law_authority_cards.py` and not read by any notebook / downstream script.
- Grep targets: every match below was verified against `notebooks/_inventory/*.md` (notebook content extracts) and `scripts/`.

---

## Court authority cards — base v1 / v3 corpus dump

**Files:**
- `e:\swiss_citation_extraction\artifacts\court_authority_cards.jsonl` (5.45 GB, JSONL)

**Description:** Whole-corpus single-file build of deterministic court authority cards (one card per court-consideration row × distinct citation). First record reports `provenance.method = "deterministic_authority_card_v3_no_gold_no_query"`. Per-line schema includes `citation`, `court_base`, `family`, `subfamily`, `pattern` (`court_bge` / `court_case` / `unknown`), `legal_area`, `issue_labels_en`, `matched_terms_multilingual`, `law_codes`, `statutes_cited`, `court_cases_cited`, `structural.bge_division`, `retrieval_text_en`, `summary_en_proxy`, `text_excerpt_original`.

**Source / created by:** `scripts/build_pipeline/build_court_authority_cards.py` (no `--chunk-index`). Reads `data/court_considerations.csv`, `artifacts/segment_lattice_v3.sqlite`, `data_insights/citation_graph_extracted.sqlite`. Pure deterministic, no gold, no query.

**Purpose:** Single-file corpus dump used in early experiments before the chunked / v4 / v5_unified pipeline existed.

**Notebooks that use it (Grep-verified):** No notebook in `notebooks/_inventory/` references the basename `court_authority_cards.jsonl` alone (only versioned variants). It is named generically and was an early stepping stone.

**Scripts that produce/consume it:** Produced by `scripts/build_pipeline/build_court_authority_cards.py`. Not consumed by any current script; `court_authority_cards_v4.jsonl` is now the canonical pre-LLM corpus card file.

**Performance / role in pipeline:** Baseline corpus dump; replaced by `court_authority_cards_v4.jsonl` (same shape, `method` bumped to `v4_no_gold_no_query`, plus `is_notification_paragraph`).

**Verdict:** superseded.
**Successor:** `court_authority_cards_v4.jsonl` → ultimately `court_authority_cards_v5_unified.jsonl`.

---

## Court authority cards — chunked single-chunk build intermediates (chunks 00..09)

**Files:**
- `artifacts\court_authority_cards_chunk00.jsonl` ... `chunk09.jsonl` (~545 MB each, JSONL, all dated 2026-04-30 18:00)
- `artifacts\court_authority_cards_chunk00.summary.json` ... `chunk09.summary.json` (~5.8 KB each; per-chunk counts, legal-area histogram, issue-label histogram)

**Description:** 10 parallel-worker shards of the full court-card build. Each chunk holds ~247,632 cards (10 × 247.6k ≈ 2.476M, matching `court_considerations.csv` row count). First records say `method = deterministic_authority_card_v3_no_gold_no_query`, but the corresponding summary already says `v4_no_gold_no_query` — the JSONL bodies are a slightly older payload than the summaries indicate.

**Source / created by:** `scripts/build_pipeline/build_court_authority_cards.py --chunk-index N --num-chunks 10` for N = 0..9. The script appends `_chunk{N:02d}` to the output basename (lines 1238-1241 of the build script), which is how chunkNN names are formed.

**Purpose:** Parallel build sharding. Each worker reads the full CSV but only processes its `row_number % num_chunks == chunk_index` rows.

**Notebooks that use it (Grep-verified):** No notebook usage found in inventory. Build-time intermediate; consumed only by the merge step that produced `court_authority_cards.jsonl` / `court_authority_cards_v4.jsonl`.

**Scripts that produce/consume it:** Produced by `build_court_authority_cards.py`. Consumed only at concatenation time.

**Performance / role in pipeline:** Build sharding scaffolding from the April 30 build session. Once merged into the whole-corpus file, the chunks were never re-read.

**Verdict:** superseded.
**Successor:** Their concatenation = `court_authority_cards.jsonl` (and the v4 rebuild = `court_authority_cards_v4.jsonl`).

---

## Court authority cards — DOUBLE-CHUNK accidental rebuild (chunkNN_chunkNN)

**Files:**
- `artifacts\court_authority_cards_chunk00_chunk00.jsonl` ... `chunk09_chunk09.jsonl` (~546 MB each, dated 2026-04-30 18:35)
- Matching `_chunkNN_chunkNN.summary.json` for each.

**Description:** Cards re-emitted ~35 min after the singly-chunked run. JSONL records and summaries both report `method = deterministic_authority_card_v4_no_gold_no_query`. Card counts and most histograms match the v3 chunks; only minor differences (e.g. `appeal admissibility` 79,571 → 84,170; `occupational pension` moved up; `federal supreme court procedure` and `debt enforcement and bankruptcy` rows disappeared) — consistent with one-pass label-rule edits between the v3 and v4 builders.

**Source / created by:** A second invocation of `build_court_authority_cards.py --chunk-index N --num-chunks 10` where the operator passed `--output artifacts/court_authority_cards_chunk00.jsonl` (already containing `_chunk00`) alongside `--chunk-index 0`. The script then appended a second `_chunk00` to the stem (line 1239: `output.stem + f"_chunk{args.chunk_index:02d}"`), producing the doubled name.

**Purpose:** Accidental duplicate emission during the v3 → v4 rebuild. The naming is a CLI operator mistake, not a designed double-shard.

**Notebooks that use it (Grep-verified):** No notebook usage found in inventory.

**Scripts that produce/consume it:** Produced by `build_court_authority_cards.py`; consumed by nothing.

**Performance / role in pipeline:** None. Dead artifact from a misnamed rebuild.

**Verdict:** abandoned (and ~5.5 GB of redundant disk usage). Safe to delete; identical-coverage v4 cards live in `court_authority_cards_v4.jsonl`.

---

## Court authority cards — build chunk logs

**Files:**
- `artifacts\chunk00.log` ... `chunk09.log` (71 B each)

**Description:** One-line JSON each: `{"chunk": N, "written": 247631-247632, "missing": 0, "notifications": 19113-19384}`. Worker stdout from the chunked build.

**Source / created by:** Operator stdout redirection from `python build_court_authority_cards.py --chunk-index N --num-chunks 10 > artifacts/chunkNN.log`.

**Purpose:** Sanity-check counts per worker.

**Notebooks that use it (Grep-verified):** No notebook usage found in inventory.

**Scripts that produce/consume it:** No script consumes them.

**Performance / role in pipeline:** Diagnostic only. Useful to confirm the 10 workers covered the corpus exactly once (10 × 247,632 ≈ 2.476M total).

**Verdict:** abandoned (informational stub).

---

## Court authority cards — sample (5,000-row preview)

**Files:**
- `artifacts\court_authority_cards_sample.jsonl` (10.96 MB, JSONL, 5,000 cards)
- `artifacts\court_authority_cards_sample.summary.json` (2.6 KB)

**Description:** A 5,000-card preview limited to two legal areas (`constitutional and public law` 2,608 + `administrative, tax, migration, and regulatory law` 2,392), all `court_bge` pattern, all under `method = deterministic_authority_card_v1_no_gold_no_query`.

**Source / created by:** `scripts/build_pipeline/build_court_authority_cards.py --limit 5000` early in the family lineage (v1 method tag).

**Purpose:** Fast smoke / schema preview during initial design of the card schema.

**Notebooks that use it (Grep-verified):** No notebook usage found in inventory.

**Scripts that produce/consume it:** Produced by `build_court_authority_cards.py`; not consumed downstream.

**Performance / role in pipeline:** None today. Stand-in for the full corpus during the v1 schema-prototyping pass.

**Verdict:** superseded.
**Successor:** `court_authority_cards_v4.jsonl` (full corpus, v4 schema) and `court_authority_cards_v5_unified.jsonl` (full corpus, LLM-merged).

---

## Court authority cards — single-card example (BGE 137 IV 122 E. 4.2)

**Files:**
- `artifacts\court_authority_card_BGE_137_IV_122_E_4_2.jsonl` (2.9 KB, 1 record)
- `artifacts\court_authority_card_BGE_137_IV_122_E_4_2.summary.json` (652 B)

**Description:** The card for one specific BGE consideration, exhibiting the v1 schema with non-trivial fills: `issue_labels_en = [collusion risk, criminal procedure, pretrial detention, procedural rights]`, `law_codes=[StPO]`, `statutes_cited=[Art. 221 Abs. 1 Bst. b, Art. 237 Abs. 1 StPO]`, two `court_cases_cited`. Summary confirms `cards_written=1`, `citation_filter_count=1`.

**Source / created by:** `build_court_authority_cards.py --citations "BGE 137 IV 122 E. 4.2"` (the `--citations` CLI accepts a semicolon-separated filter list; see lines 1228-1235 of the script).

**Purpose:** Worked-example / reference card used while validating the v1 schema. Appears as the "well-formed card" reference in early enrichment-prompt design.

**Notebooks that use it (Grep-verified):** No notebook usage found in inventory.

**Scripts that produce/consume it:** Produced by `build_court_authority_cards.py`; not consumed by any current script.

**Performance / role in pipeline:** Documentation example only.

**Verdict:** abandoned (kept as a reference data point; not on any code path).

---

## Court authority cards — v4 (canonical pre-LLM whole-corpus)

**Files:**
- `artifacts\court_authority_cards_v4.jsonl` (5.46 GB, JSONL, 2,476,310 cards)

**Description:** Whole-corpus v4 build. Same shape as the base file but `provenance.method = "deterministic_authority_card_v4_no_gold_no_query"`. This is the input the v5_unified merger streams against.

**Source / created by:** `scripts/build_pipeline/build_court_authority_cards.py` end-to-end (no chunking) with the v4 method tag in place. The two `deterministic_authority_card_v4_no_gold_no_query` strings in the current script body (lines 1074, 1167 of `build_court_authority_cards.py`) confirm v4 is what the script emits today.

**Purpose:** The canonical pre-LLM deterministic court card corpus. Acts as the structural backbone that the 363k Qwen3-Reranker LLM enrichments are stitched into.

**Notebooks that use it (Grep-verified):** Indirectly — the v7 / v7.4 / v7.5 / endgame notebooks all reference the v5_unified successor, not v4 directly. The v4 file is the merge input, not the runtime input.

**Scripts that produce/consume it:**
- Produced by: `build_court_authority_cards.py`.
- Consumed by: `scripts/merging/merge_llm_enrichment_into_v4_cards.py` (streams v4 + LLM jsonl → v5_unified, see the v5_unified.log: `[merge] Streaming E:\...\court_authority_cards_v4.jsonl -> artifacts\court_authority_cards_v5_unified.jsonl`).
- Also consumed by: `scripts/citation_extraction/identify_court_enrichment_targets.py` and `scripts/citation_extraction/extract_court_target_cards.py` (the enrichment-targeting passes below).

**Performance / role in pipeline:** Last deterministic-only checkpoint before LLM merge. Recoverable input if the LLM merge needs to be re-run.

**Verdict:** active (as merge input).
**Successor:** `court_authority_cards_v5_unified.jsonl` is the runtime card file.

---

## Court authority cards — v4 enrichment-target priority list

**Files:**
- `artifacts\court_authority_cards_v4_enrichment_targets.jsonl` (208 MB, 363,258 records)
- `artifacts\court_authority_cards_v4_enrichment_targets.summary.json` (22 KB)

**Description:** One thin priority record per card flagged as worth LLM enrichment. Fields: `line_idx`, `csv_row`, `citation`, `court_base`, `priority` (float; top entry 289.0), `should_target` (bool), `reasons` (e.g. `missing_issue_labels`, `missing_multilingual_terms`, `thin_retrieval_text`, `noisy_law_code_detection`, `unknown_language`), `high_value_reasons` (e.g. `published_bge`, `frequently_cited_authority`, `high_court_base_source_count`, `has_statute_links`, `has_court_case_links`), various structural counts, and language. Summary reports `min_priority=65.0` and `candidate_targets_before_threshold=984,984` filtered down to `final_target_count=363,258` (matches the 363k Qwen3 enrichment run exactly).

**Source / created by:** `scripts/citation_extraction/identify_court_enrichment_targets.py` (priority scoring pass over v4 cards).

**Purpose:** Tells the enrichment runner which cards are most worth LLM processing under a fixed compute budget — the 363k target list that became the actual enrichment payload on Drive.

**Notebooks that use it (Grep-verified):** No notebook usage found in inventory. Consumed by the next pipeline step (target-card extraction), not by notebooks directly.

**Scripts that produce/consume it:** Produced by `identify_court_enrichment_targets.py`; consumed by `scripts/citation_extraction/extract_court_target_cards.py`.

**Performance / role in pipeline:** Defines the 363k subset that the Qwen3-8B / Claude enrichment runs are scoped to. Without it, the LLM run would have to process all 2.47M cards (impractical).

**Verdict:** active (as the canonical target list).
**Successor:** none; used as-is.

---

## Court authority cards — v4 target cards (full target subset)

**Files:**
- `artifacts\court_authority_cards_v4_target_cards.jsonl` (1.03 GB, 363,258 cards)
- `artifacts\court_authority_cards_v4_target_cards.summary.json` (2.8 KB)
- `artifacts\court_authority_cards_v4_target_cards_1000.jsonl` (2.97 MB, first 1,000 target cards — smoke set)
- `artifacts\court_authority_cards_v4_target_cards_1000.summary.json` (1.3 KB)

**Description:** Full v4 card payload for each of the 363,258 target rows. Carries the same v4 schema as `court_authority_cards_v4.jsonl` but materialized as a separate file so the enrichment runner only has to stream 1 GB, not 5.5 GB. The 1000-card variant is a smoke/dev slice.

**Source / created by:** `scripts/citation_extraction/extract_court_target_cards.py` (joins v4 cards with the target priority list).

**Purpose:** Direct input file for both Qwen3 and Claude court-card enrichment runs (see `enrich_cards_with_claude_agent_sdk.py` Config: `input: Path = Path("artifacts/court_authority_cards_v4_target_cards.jsonl")`).

**Notebooks that use it (Grep-verified):** Heavy notebook usage in `notebooks/10_authority_card_enrichment/` (e.g. `auth_cards_enrich_qwen3_8b_kaggle.ipynb`, `auth_cards_enrich_qwen3_8b_colab.ipynb`, `auth_cards_enrich_qwen35_*` variants) and `notebooks/13_misc_kaggle_and_utilities/kaggle_qwen3_8b_awq_vllm_text_to_json_*.ipynb` — all confirmed by Grep against `notebooks/_inventory/`.

**Scripts that produce/consume it:**
- Produced by `extract_court_target_cards.py`.
- Consumed by `scripts/enrichment/enrich_cards_qwen3_8b_kaggle.py`, `scripts/enrichment/enrich_cards_with_qwen.py`, `scripts/enrichment/enrich_cards_with_claude_agent_sdk.py`, `scripts/enrichment/enrich_val001_gold_cards.py`.

**Performance / role in pipeline:** This is the actual file the LLM enrichment fleet (Qwen3-8B on Kaggle/Colab + Claude SDK runner) consumed. Output of that run = `llm_enrichment_output_court_363k/court_llm_descriptors_0000000_all.jsonl` → merged with v4 cards = v5_unified.

**Verdict:** active (still the canonical LLM-input slice; needed if any per-target re-enrichment is re-run).
**Successor:** none.

---

## Court authority cards — Claude RAG enrichment outputs (smoke-only)

**Files:**
- `artifacts\court_authority_cards_rag_targets_claude.jsonl` (19.8 KB, 4 cards)
- `artifacts\court_authority_cards_rag_targets_claude_failed_cards.jsonl` (0 B, empty)

**Description:** 4 court target cards re-enriched via Claude Agent SDK. Records have the same v4-card prefix as `_target_cards.jsonl` (`method = deterministic_authority_card_v4_no_gold_no_query`) — they appear to be the raw input rows; Claude-side enriched fields were configured to live on the same record but the smoke output here keeps the input shape. Failures file is empty (no failures in the smoke).

**Source / created by:** `scripts/enrichment/enrich_cards_with_claude_agent_sdk.py` (docstring: "Enrich Swiss court authority target cards with Claude Agent SDK"). Config defaults: `limit = 25`, `concurrency = 2`, `model = "sonnet"`, `output = artifacts/court_authority_cards_rag_targets_claude.jsonl`, `failed = artifacts/court_authority_cards_rag_targets_claude_failed_cards.jsonl`.

**Purpose:** Claude-side equivalent of the Qwen3 enrichment fleet — used to test feasibility / quality of using Sonnet rather than Qwen3-8B for the 363k-card enrichment.

**Notebooks that use it (Grep-verified):** No notebook usage found in inventory; the Claude SDK runner is script-only.

**Scripts that produce/consume it:** Produced by `enrich_cards_with_claude_agent_sdk.py`; not consumed downstream.

**Performance / role in pipeline:** 4-card smoke test only. The full 363k enrichment was completed via Qwen3-8B, not Claude (see `llm_enrichment_output_court_363k/`).

**Verdict:** abandoned (smoke artifact; never scaled).

---

## Court authority cards — v5 unified (canonical runtime cards)

**Files:**
- `artifacts\court_authority_cards_v5_unified.jsonl` (10.44 GB, 2,476,315 cards)
- `artifacts\court_authority_cards_v5_unified.summary.json` (11.4 KB)
- `artifacts\court_authority_cards_v5_unified.log` (476 B, merge progress)
- `artifacts\court_authority_cards_v5_unified.smoke.jsonl` (10.69 MB, smoke slice)
- `artifacts\court_authority_cards_v5_unified.smoke.summary.json` (10.3 KB)

**Description:** v4 cards merged row-for-row with the 363,258 LLM enrichments. Schema dramatically different from v4: top-level `enrichment_source` ∈ {`static`, `llm+static`}, plus `rag_enrichment` (the LLM dossier — legal_area, primary_domain, topic, doctrinal_rule, legal_test, paragraph_role, outcome_signal, concepts_en, terms_original, statute_anchors, case_anchors, etc.), `normalized_anchors`, `anchor_quality_flags`, `retrieval_views` (semantic_concepts_en, topic_path, original_terms_view, statute_anchor_view, case_anchor_view, citation_view, raw_context), `enrichment_quality` flags. Summary: 2,476,315 cards (2,113,058 static + 363,257 llm+static); paragraph_role and outcome distributions; language split de=1,420k / fr=793k / it=125k / unknown=137k.

**Source / created by:** `scripts/merging/merge_llm_enrichment_into_v4_cards.py`. Log records "Reading LLM enrichment from C:\Users\samiul\Downloads\outputs_from_363k_run\court_llm_descriptors_0000000_all.jsonl" (363,257 loaded, 1 failed) then streams v4 cards. Merge runtime: 3,575.7 s ≈ 60 min; 692.5 rows/s.

**Purpose:** The canonical court-side authority-card file consumed by every downstream retrieval and reranking experiment after May 5, 2026.

**Notebooks that use it (Grep-verified):**
- `notebooks/02_v75_multiquery_canonical_pool/pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb`
- `notebooks/02_v75_multiquery_canonical_pool/pool_v75_multiquery_iteration_11.ipynb`
- `notebooks/02_v75_multiquery_canonical_pool/pool_v75_multiquery_hyde_test_no_lift.ipynb`
- `notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v4..v7..v7_4_val001.ipynb` (the v4-through-v7.4 family)
- `notebooks/04_pre_v75_pipeline_iterations/endgame_colab_full_pipeline.ipynb`
- Plus references in `research/endgame_handoff_2026-05-09.md` (sections on artifact layout and §6.1 court-LLM merge), the `swiss-citation-data` skill, and `README.md`.

**Scripts that produce/consume it:**
- Produced by: `scripts/merging/merge_llm_enrichment_into_v4_cards.py`.
- Consumed by: `scripts/build_pipeline/build_unified_retrieval_corpus.py` (folds v5 court cards into `unified_retrieval.sqlite`), `scripts/merging/merge_court_llm_into_unified.py`, the v7/v7.4/v7.5/endgame retrieval notebooks.

**Performance / role in pipeline:** This is the court half of the retrieval corpus. The static-flag misnomer is documented (endgame §6.1 + `swiss-citation-data` skill): `enrichment_source = "static"` does NOT mean un-enriched — it means the court LLM schema forbids `english_summary` by design. All cards are merged into the unified retrieval SQLite.

**Verdict:** active (canonical court runtime cards).
**Successor:** none.

The `.smoke.*` siblings are a 10 MB / 10k-card slice produced by running the merger with `--limit` for fast checks; they share the same schema.

---

## Court authority cards — test_chunk smoke

**Files:**
- `artifacts\test_chunk_chunk00.jsonl` (242 KB, 100 cards)
- `artifacts\test_chunk_chunk00.summary.json` (3.4 KB)

**Description:** A 100-card smoke build emitted via `build_court_authority_cards.py --output artifacts/test_chunk.jsonl --chunk-index 0 --num-chunks 10 --limit 100`. All cards are `court_bge` and all are in `constitutional and public law` (the first hits of the corpus iteration).

**Source / created by:** `build_court_authority_cards.py` invoked with the `test_chunk` output stem (which the script then suffixed with `_chunk00`).

**Purpose:** Build-script smoke test before kicking off the 10-shard run.

**Notebooks that use it (Grep-verified):** No notebook usage found in inventory.

**Scripts that produce/consume it:** Produced by `build_court_authority_cards.py`; not consumed downstream.

**Performance / role in pipeline:** None. Pure smoke check that the chunk-mode CLI works end-to-end.

**Verdict:** abandoned.

---

## Law authority cards — v1 static (pre-LLM)

**Files:**
- `artifacts\law_authority_cards_v1_static.jsonl` (735 MB, 175,933 cards — one per `laws_de.csv` row)
- `artifacts\law_authority_cards_v1_static.summary.json` (13.4 KB)
- `artifacts\law_authority_cards_v1_static.smoke.jsonl` (4.13 MB, 1,000-card slice)
- `artifacts\law_authority_cards_v1_static.smoke.summary.json` (7.8 KB)

**Description:** Deterministic law-side authority cards built directly from `data/laws_de.csv`. Schema mirrors the court v4 layout but for statutes: `_source_row`, `citation`, `family="law"`, `pattern="statute_article"`, `law_title`, `structural.{article, article_marker, article_series_kind, granularity, law_code, law_code_family, law_code_resolution, units}`, `title_metadata.{enactment_date, law_aliases, source_type, systematic_collection_sector}`, `normalized_anchors.{adjacent_citations.{next_in_law, previous_in_law, same_article_siblings}, statute_anchors, official_references, incoming_reference_count}`, `enrichment_quality.{llm_priority, needs_llm_for_complete_semantic_context, static_formal_complete, static_semantic_signal, text_char_count, title_char_count}`, `retrieval_views.{citation_view, law_context_view, raw_context, ...}`, `rag_enrichment` (empty stub for LLM to fill). Summary: 175,933 rows; `static_formal_complete_pct=100%`; static_semantic_signal split = strong_dictionary 138,720 / partial 23,624 / formal_only 13,589; LLM priority split = medium 150,079 / high 21,922 / low 3,932; text length p50=181, p99=1,274, max=22,348.

**Source / created by:** `scripts/build_pipeline/build_law_authority_cards.py`. Imports from `build_court_authority_cards` and `extract_citation_graph` (line 28). Reads `data/laws_de.csv`, `data_insights/laws_de_classified_citations.jsonl`, `data_insights/laws_de_links.json`. Docstring: "Build deterministic law authority cards for laws_de.csv. The output is the law-side analogue of the court v4/v5 authority-card assets… query-independent and does not read train/val/test labels."

**Purpose:** Law half of the deterministic card baseline; structural + dictionary backbone before LLM dossier merge.

**Notebooks that use it (Grep-verified):** No notebook usage found in inventory (the law-side v2_unified is what notebooks reference). Build-time intermediate used only by the merger.

**Scripts that produce/consume it:**
- Produced by: `build_law_authority_cards.py`.
- Consumed by: `scripts/merging/merge_law_llm_into_v1_cards.py` (input → v2_unified).
- Also referenced by: `scripts/data_prep/granularity_resolver.py`, `scripts/build_pipeline/build_law_llm_input.py`, `scripts/build_pipeline/build_unified_retrieval_corpus.py`.

**Performance / role in pipeline:** Pre-LLM checkpoint. Recoverable input if the law-LLM merge needs to re-run.

**Verdict:** active (as merge input).
**Successor:** `law_authority_cards_v2_unified.jsonl`.

The `.smoke.*` slice is a 1,000-row dev fixture; same schema.

---

## Law authority cards — v2 unified (canonical runtime law cards)

**Files:**
- `artifacts\law_authority_cards_v2_unified.jsonl` (883 MB, 175,933 cards)
- `artifacts\law_authority_cards_v2_unified.summary.json` (3.1 KB)

**Description:** v1 static cards merged with the 173,033-row law-side LLM enrichment (`law_json_llm_output/law_llm_descriptors_0000000_all.jsonl`). Per-record adds `enrichment_source` ∈ {`llm+static` (173,033 = 98.4%), `static` (2,900)} and fully populated `rag_enrichment`: concepts_en, english_summary, legal_question, legal_rule, doctrinal_rule, legal_test, applicability_conditions, exceptions_or_limitations, addressees, sanctions_or_consequences, terms_de_to_en, defined_terms, primary/secondary_domain, legal_area, legal_domain_path, provision_role_llm (procedure / definition / duty / right_or_entitlement / scope / competence / data_reporting / purpose / transitional_or_commencement / prohibition / principle / sanction_or_penalty / fees_or_costs / other), specificity_score. Summary records 100% legal_area, 100% legal_domain_path, 99.78% terms_de_to_en, 95.01% applicability_conditions on llm+static rows.

**Source / created by:** `scripts/merging/merge_law_llm_into_v1_cards.py` (per summary `inputs.cards = law_authority_cards_v1_static.jsonl`, `inputs.llm = law_json_llm_output/law_llm_descriptors_0000000_all.jsonl`; merge runtime 22.0 s).

**Purpose:** Canonical law-side authority-card file. Folded into `unified_retrieval.sqlite` as the law-side document rows for the retrieval pool.

**Notebooks that use it (Grep-verified):**
- `notebooks/04_pre_v75_pipeline_iterations/endgame_colab_full_pipeline.ipynb`
- `notebooks/07_law_de_enrichment/enrich_laws_de_qwen3_8b_kaggle*.ipynb` (3 variants)
- Plus references in `research/endgame_handoff_2026-05-09.md`, `swiss-citation-data` skill, `README.md`, `docs/laws_de_llm_enrichment_schema.md`, `docs/laws_de_authority_card_static_report.md`.

**Scripts that produce/consume it:**
- Produced by: `merge_law_llm_into_v1_cards.py`.
- Consumed by: `scripts/build_pipeline/build_unified_retrieval_corpus.py` (and the `unified_retrieval.build.log` trail).

**Performance / role in pipeline:** Law half of the retrieval corpus. Together with `court_authority_cards_v5_unified.jsonl` this is the structural + semantic backbone of every retrieval / rerank experiment after May 6, 2026.

**Verdict:** active (canonical law runtime cards).
**Successor:** none.

---

## Choice cards v3 — segment-lattice option set

**Files:**
- `artifacts\choice_cards_v3.json` (2.85 MB, dict-of-arrays JSON, top-level keys like `court_considerations.court.court_bge.court_base`, `court_considerations.court.court_bge.division`, etc.)
- 9 sibling copies inside `artifacts\test_segment_lattice_v3_<timestamp>\artifacts\choice_cards_v3.json`

**Description:** Each top-level key names a (dataset, family, subfamily, segment_key) tuple. The value is a list of candidate "options" — identifier-like values like `BGE 147 II 72`, with metadata: `examples` (sample child citations using the value), `meaning_en`, `pattern`, `family`, `subfamily`, `selector_kind` (`explicit_only`), `source_count`, `text_ref_count`, `co_signals`, `text_keywords`. Not a per-card file — a query-time "what values can fill this segment slot?" lookup.

**Source / created by:** `scripts/retrieval_and_rerank/segment_lattice_v3.py` (docstring: "Build and run the no-leak Segment-Lattice Funnel v3"). Co-built with `segment_lattice_v3.sqlite`. The 9 timestamped sibling folders under `artifacts/test_segment_lattice_v3_<timestamp>/artifacts/` are pytest-emitted snapshots from `tests/test_segment_lattice_v3.py`.

**Purpose:** Lookup table for the early segment-lattice funnel — for a query mentioning e.g. `division=IV`, what option-cards can selectively fill the slot. Pre-v4 funnel infrastructure.

**Notebooks that use it (Grep-verified):** Only `notebooks/_inventory/early_segment_lattice_funnel_v3.md` references it. None of the v4..v7.5 funnel notebooks use this lookup file.

**Scripts that produce/consume it:** Produced by `segment_lattice_v3.py` + `scripts/notebook_builders/create_segment_lattice_v3_notebook.py`; consumed by the same script in `audit`/`run` mode. `scripts/build_pipeline/_run_build_index.py` references it as a build target.

**Performance / role in pipeline:** Predecessor of the v4 anchor-funnel approach. Superseded once the funnel moved off the lattice-and-choice-cards design toward the channel-fusion architecture in `anchor_funnel_v4..v7_5`.

**Verdict:** superseded.
**Successor:** The 15-channel v7.4 funnel (`v7_4_fixes/_cell34_body.py`) and the v7.5 multi-query snapshot replaced this design entirely.

The 9 timestamped sibling copies under `test_segment_lattice_v3_*/artifacts/` are pytest fixtures — same content, smaller (`source_count=1`, `text_ref_count=0` per entry vs. the live file's 110/65) — abandoned test-run artifacts. Safe to delete with the test directories.

---

## Cross-references

- `README.md` and `swiss-citation-data` skill name only the two unified files (`court_authority_cards_v5_unified.jsonl`, `law_authority_cards_v2_unified.jsonl`) and the v4 family (`court_authority_cards_v4*.jsonl + chunk files — legacy v4 cards and target lists`) as the live artifacts.
- `research/endgame_handoff_2026-05-09.md` §6.1 explains why `enrichment_source = "static"` on court cards is a misnomer (the court LLM schema forbids `english_summary` by design — the cards are still LLM-merged).
- `scripts/build_pipeline/build_court_authority_cards.py` lines 1238-1241 are the reason for the `_chunkNN_chunkNN.jsonl` accidental rebuild: the script unconditionally appends `_chunk{N:02d}` to the output stem, so running it with an already-`_chunkNN`-suffixed output produces a doubled name.

## Disk-usage cleanup candidates (no code on the path)

| File | Size | Reason |
|---|---|---|
| `court_authority_cards_chunk00_chunk00.jsonl` ... `chunk09_chunk09.jsonl` (+ 10 summaries) | ~5.46 GB total | Accidental double-named rebuild. Identical-coverage v4 cards live in `court_authority_cards_v4.jsonl`. |
| `court_authority_cards_chunk00.jsonl` ... `chunk09.jsonl` (+ 10 summaries) | ~5.45 GB total | v3-era build-time intermediates. Concatenation lives in `court_authority_cards.jsonl` and v4 lives in `court_authority_cards_v4.jsonl`. |
| `court_authority_cards.jsonl` | 5.45 GB | Pre-v4 single-file dump. Superseded by `court_authority_cards_v4.jsonl` (which is itself the input to v5_unified). |
| `court_authority_cards_sample.jsonl` (+ summary) | 11 MB | v1 smoke slice. |
| `court_authority_card_BGE_137_IV_122_E_4_2.jsonl` (+ summary) | 3.5 KB | v1 single-record example. |
| `test_chunk_chunk00.jsonl` (+ summary) | 246 KB | 100-row smoke. |
| `chunk00.log` ... `chunk09.log` | 710 B total | Operator stdout. |
| `court_authority_cards_rag_targets_claude.jsonl` (+ failed) | 20 KB | 4-card Claude SDK smoke. |
| `artifacts/test_segment_lattice_v3_*` directories | unknown (9 copies of choice_cards_v3.json ≈ tens of MB) | Pytest fixture residue. |

(Cleanup is advisory — the user has not asked to delete anything; this is the visibility list for the data-analysis report.)
