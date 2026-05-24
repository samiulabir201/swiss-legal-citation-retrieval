"""Full-repo reorganization pass.

Goals:
1. Restructure drive_sync/ — flatten ugly "My Drive__..." names into nested clean paths.
2. Subcategorize scripts/ — group 58 loose .py into purpose folders.
3. Move misplaced files:
   - translation_work/laws_de_full.json → data/laws_de_full.json
   - research-run-2026-05-09/ → research/_archive_v7_initial_2026-05-09/
4. Sub-organize research/ — separate scratch from snapshot from analysis.
5. Sub-organize data_insights/ — separate citation graph from gold analysis.
6. Sub-organize v7_4_fixes/ and v7_5_precision/ internals lightly.
7. Delete obvious junk: research/New folder/.

Preserves all canonical names referenced by swiss-citation-* skills.
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(r"e:\swiss_citation_extraction")
LOG: list[dict] = []


def log(action, src, dst=None, **kw):
    e = {"action": action, "src": str(src)}
    if dst:
        e["dst"] = str(dst)
    e.update(kw)
    LOG.append(e)


def mv(src: Path, dst: Path):
    if not src.exists():
        log("missing_source", src)
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        log("target_exists_skipped", src, dst)
        return False
    shutil.move(str(src), str(dst))
    log("moved", src, dst)
    return True


def rmrf(p: Path, only_if_empty=True):
    if not p.exists():
        return False
    if only_if_empty:
        try:
            next(p.iterdir())
            log("rmrf_skipped_not_empty", p)
            return False
        except StopIteration:
            pass
    if p.is_dir():
        shutil.rmtree(str(p))
    else:
        p.unlink()
    log("removed", p)
    return True


# ─────────────────────────────────────────────────────────────────────────
# PHASE 1: drive_sync restructure
# ─────────────────────────────────────────────────────────────────────────
DS = ROOT / "drive_sync"

# Map: old flat name → new nested path under drive_sync/
DRIVE_MAP = {
    # swiss_law/*
    "My Drive__swiss_law": "swiss_law/_root_files",
    "My Drive__swiss_law__research__anchor_funnel_val001_v7": "swiss_law/v7_pool_recall_089_and_per_query",
    "My Drive__swiss_law__research__anchor_funnel_val001_v4": "swiss_law/v4_anchor_funnel_per_version_summaries",
    "My Drive__swiss_law__research__anchor_funnel_val001": "swiss_law/v1_anchor_funnel_val001_initial",
    "My Drive__swiss_law__research__hybrid_rerank_final": "swiss_law/hybrid_rerank_shootout_cache",
    "My Drive__swiss_law__research__rerank_only_diagnostic": "swiss_law/rerank_only_diagnostic_results",
    "My Drive__swiss_law__artifacts__eval": "swiss_law/eval_metrics_and_ablations",
    "My Drive__swiss_law__artifacts_legalmalr": "swiss_law/legalmalr_cascade_stage_a_output",
    "My Drive__swiss_law__cache_legalmalr": "swiss_law/legalmalr_cascade_per_query_cache",
    "My Drive__swiss_law__configs": "swiss_law/threshold_and_runtime_configs",
    "My Drive__swiss_law__scripts": "swiss_law/colab_side_scripts",
    "My Drive__swiss_law__submissions": "swiss_law/submitted_predictions_csv",
    "My Drive__swiss_law__data__checkpoints": "swiss_law/llm_enrichment_jsonl_checkpoints",
    "My Drive__swiss_law__data_insights": "swiss_law/colab_data_insights_mirror",
    "My Drive__swiss_law__data__artifacts__A6__legal_swiss_finetuned": "swiss_law/A6_finetuned_legal_swiss_model",
    "My Drive__swiss_law__data__artifacts__A6__legal_swiss_simcse": "swiss_law/A6_simcse_legal_swiss_model",
    # omnilex_local_artifacts_qwen3_flatip is just one folder (10 GB embeddings) — skipped, but path
    "My Drive__omnilex_local_artifacts_qwen3_flatip": "omnilex_local_embeddings_qwen3_flatip",
    # Omnilex competition
    "My Drive__Omnilex-Agentic-Retrieval-Competition__retrieval": "omnilex_competition/retrieval_other_artifacts",
    "My Drive__Omnilex-Agentic-Retrieval-Competition__retrieval__knowledge_base_optimized_hybrid_retrieval": "omnilex_competition/retrieval_knowledge_base_query_translations",
    "My Drive__Omnilex-Agentic-Retrieval-Competition__retrieval__pipeline_cache": "omnilex_competition/retrieval_pipeline_qwen3_reranker_cache",
    "My Drive__Omnilex-Agentic-Retrieval-Competition__retrieval__pipeline_output_v3": "omnilex_competition/retrieval_pipeline_submissions_K_sweep",
    "My Drive__Omnilex-Agentic-Retrieval-Competition__swiss-law-pipeline": "omnilex_competition/swiss_law_pipeline_source_code",
    # Loose file
    "My Drive__Omnilex-Agentic-Retrieval-Competition__abbrev-translations.json": "omnilex_competition/abbrev_translations.json",
}

print("=== PHASE 1: drive_sync restructure ===")
for old, new in DRIVE_MAP.items():
    src = DS / old
    dst = DS / new
    if src.exists():
        mv(src, dst)


# ─────────────────────────────────────────────────────────────────────────
# PHASE 2: scripts/ subcategorization
# ─────────────────────────────────────────────────────────────────────────
SCRIPTS = ROOT / "scripts"

SCRIPT_MAP = {
    # build_pipeline/ — corpus & artifact builders
    "build_pipeline": [
        "build_court_authority_cards.py",
        "build_court_retrieval_corpus.py",
        "build_gold_bank_candidates.py",
        "build_law_authority_cards.py",
        "build_law_llm_input.py",
        "build_unified_retrieval_corpus.py",
        "_run_build_index.py",
    ],
    # citation_extraction/ — citation parsing, graph extraction, alias work
    "citation_extraction": [
        "add_case_level_aliases.py",
        "add_date_stripped_aliases.py",
        "add_range_expanded_aliases.py",
        "audit_statute_anchor_regex.py",
        "extract_citation_graph.py",
        "extract_court_target_cards.py",
        "extract_intra_judgment_backrefs.py",
        "extract_retrieval_metadata.py",
        "identify_court_enrichment_targets.py",
        "verify_french_italian_aliases.py",
    ],
    # enrichment/ — LLM enrichment runners and helpers
    "enrichment": [
        "court_enrichment_normalizer.py",
        "court_enrichment_profile.py",
        "enrich_cards_qwen3_8b_kaggle.py",
        "enrich_cards_with_claude_agent_sdk.py",
        "enrich_cards_with_qwen.py",
        "enrich_val001_gold_cards.py",
        "profile_court_enrichment_schema.py",
        "run_law_llm_enrichment.py",
    ],
    # retrieval_and_rerank/ — query encoding, retrieval, reranking, judging
    "retrieval_and_rerank": [
        "encode_queries_qwen3_8b.py",
        "hybrid_retrieve.py",
        "llm_judge_qwen3.py",
        "rerank_qwen3.py",
        "precompute_dossier_cache.py",
        "segment_lattice_v3.py",
    ],
    # eval_and_audit/ — evaluation + verification + experiments
    "eval_and_audit": [
        "eval_retrieval.py",
        "check_gold_parent_links.py",
        "verify_gold_citation_coverage.py",
        "experiment_court_recall_enriched_1m.py",
        "experiment_court_recall_stress_1m.py",
        "experiment_enrichment_recall_stress.py",
    ],
    # data_prep/ — granularity resolution, splitting, UTF-8 repair
    "data_prep": [
        "granularity_resolver.py",
        "prepare_embedding_input.py",
        "split_laws_de.py",
        "repair_law_enrichment_utf8.py",
        "repair_train_csv_utf8.py",
    ],
    # merging/ — merging LLM descriptors into authority cards / unified DB
    "merging": [
        "merge_court_llm_into_unified.py",
        "merge_law_llm_into_v1_cards.py",
        "merge_llm_enrichment_into_v4_cards.py",
    ],
    # notebook_builders/ — scripts that create or patch .ipynb files
    "notebook_builders": [
        "_build_endgame_notebook.py",
        "_patch_law_notebook.py",
        "create_qwen3_8b_colab_notebook.py",
        "create_recall_funnel_notebook.py",
        "create_segment_lattice_v3_notebook.py",
        "create_v7_notebook.py",
        "patch_download_notebook_gold_bank.py",
        "patch_notebook_no_leak.py",
        "patch_qwen_checkpoint_isolation.py",
        "patch_qwen_notebook_targeted.py",
    ],
    # kaggle_upload/ — Kaggle integration
    "kaggle_upload": [
        "upload_to_kaggle.py",
    ],
}

print("\n=== PHASE 2: scripts/ subcategorization ===")
for sub, files in SCRIPT_MAP.items():
    sub_dir = SCRIPTS / sub
    sub_dir.mkdir(exist_ok=True)
    for f in files:
        src = SCRIPTS / f
        if src.exists():
            mv(src, sub_dir / f)


# ─────────────────────────────────────────────────────────────────────────
# PHASE 3: top-level cleanups + misplaced files
# ─────────────────────────────────────────────────────────────────────────
print("\n=== PHASE 3: top-level cleanups ===")

# 3a. translation_work/laws_de_full.json — DE law text → move to data/
src = ROOT / "translation_work" / "laws_de_full.json"
dst = ROOT / "data" / "laws_de_full.json"
mv(src, dst)
rmrf(ROOT / "translation_work")  # only if empty

# 3b. research-run-2026-05-09/ — early backup of v7 initial run → archive in research/
old = ROOT / "research-run-2026-05-09"
new = ROOT / "research" / "_archive_v7_initial_2026-05-09"
if old.exists():
    # Move the inner content (it's research/anchor_funnel_val001/) up one level
    inner = old / "research" / "anchor_funnel_val001"
    if inner.exists():
        mv(inner, new)
    rmrf(old / "research", only_if_empty=True)
    rmrf(old, only_if_empty=True)

# 3c. research/New folder/ — empty junk
rmrf(ROOT / "research" / "New folder", only_if_empty=True)


# ─────────────────────────────────────────────────────────────────────────
# PHASE 4: research/ sub-organization (separate concerns)
# ─────────────────────────────────────────────────────────────────────────
print("\n=== PHASE 4: research/ sub-organization ===")
RES = ROOT / "research"

# Rename underscore-prefix scratch folders (cleaner)
for src_name, dst_name in [
    ("_scratch_2026-05-12", "scratch_2026-05-12_anti_pattern_gold_text"),
    ("_scratch_2026-05-13", "scratch_2026-05-13_pair_extraction"),
]:
    mv(RES / src_name, RES / dst_name)

# Move the local v7 snapshot mirror (sample of the Drive snapshot) into a named subfolder
mv(RES / "anchor_funnel_val001_v7", RES / "local_v75_snapshot_mirror")

# Move the local hybrid rerank cache
mv(RES / "hybrid_rerank_final", RES / "local_hybrid_rerank_cache_mirror")

# stage_b_experiments has a REPORT.md + diag_*.py — rename to be descriptive
mv(RES / "stage_b_experiments", RES / "stage_b_diagnostics_and_ceilings")


# ─────────────────────────────────────────────────────────────────────────
# PHASE 5: data_insights/ sub-organization
# ─────────────────────────────────────────────────────────────────────────
print("\n=== PHASE 5: data_insights/ sub-organization ===")
DI = ROOT / "data_insights"

# Sub-folders:
#   citation_graph/ — sqlite + raw graph extracts
#   gold_analysis/  — gold coverage / parent link / posture priors
#   citation_patterns/ — pattern analysis reports
GRAPH = DI / "citation_graph_db_and_edges"
GOLD = DI / "gold_analysis_coverage_and_parents"
PATTERNS = DI / "citation_patterns_and_priors"

GRAPH.mkdir(exist_ok=True)
GOLD.mkdir(exist_ok=True)
PATTERNS.mkdir(exist_ok=True)

for f in [
    "citation_graph_extracted.sqlite",
    "citation_graph_extracted.sqlite.bak_pre_backref",
    "citation_graph_extracted.sqlite.bak_pre_caselevel",
    "laws_de_links.json",
    "court_considerations_links.json",
    "laws_de_citations.json",
    "court_considerations_citations.json",
    "laws_de_classified_citations.jsonl",
    "court_considerations_classified_citations.jsonl",
]:
    mv(DI / f, GRAPH / f)

for f in [
    "gold_citation_coverage.csv",
    "gold_citation_coverage.json",
    "gold_citation_coverage.md",
    "gold_parent_link_check.csv",
    "gold_parent_link_check.json",
    "gold_parent_link_check.md",
    "train_gold_citations.json",
    "val_gold_citations.json",
]:
    mv(DI / f, GOLD / f)

for f in [
    "citation_pattern_analysis.md",
    "citation_patterns.json",
    "posture_statute_prior.json",
    "example.json",
]:
    mv(DI / f, PATTERNS / f)


# ─────────────────────────────────────────────────────────────────────────
# PHASE 6: v7_4_fixes/ and v7_5_precision/ light sub-organization
# ─────────────────────────────────────────────────────────────────────────
print("\n=== PHASE 6: v7_4_fixes/ and v7_5_precision/ sub-organization ===")

V74 = ROOT / "v7_4_fixes"

# v7_4_fixes/ contains 4 distinct types:
#   _cellN_body.py / _cell_stageN_*.py  → notebook cell bodies extracted
#   _addX.py                            → notebook cell INSERTERS
#   <channel>_fix.py                    → per-channel patches (bm25/concept_en/etc.)
#   _verify*.py, _investigate*.py, _dump*.py, _inspect_nb.py → diagnostics / dumps
#   _v7_5_apply.py, _integrate_all.py, _lift_topk_50k.py, _make_multiquery.py, _smoke_matcher.py → integrations
#   _concepts_sample.json, missed_gold_enrichment.json → data
#   _update_markdowns.py, _robust_llm_parser.py → utilities

V74_GROUPS = {
    "cell_bodies_extracted_from_notebook": [
        "_cell28_body.py", "_cell34_body.py",
        "_cell_stage1_rerank.py", "_cell_stage1b_translation.py",
        "_cell_stage2_listwise.py", "_cell_stage3_pointwise.py",
        "_cell_stage4_final.py",
    ],
    "cell_inserters": [
        "_add_cascade.py", "_add_hyde.py", "_add_miss_diagnostic.py",
        "_add_multihyde.py", "_add_prf.py", "_add_snapshot_warmboot.py",
    ],
    "per_channel_fixes": [
        "bm25_fix.py", "co_citation_backprop_fix.py", "concept_en_fix.py",
        "court_statute_fix.py", "sibling_graph_fix.py", "term_orig_fix.py",
    ],
    "diagnostics_and_dumps": [
        "_dump_cells.py", "_dump_config.py", "_dump_exact.py",
        "_inspect_nb.py", "_investigate.py", "_investigate_cocit.py",
        "_investigate_court_stat.py", "_verify_fix.py", "_verify_patches.py",
        "_verify_val001.py",
    ],
    "integrations_and_utilities": [
        "_integrate_all.py", "_lift_topk_50k.py", "_make_multiquery.py",
        "_robust_llm_parser.py", "_smoke_matcher.py", "_update_markdowns.py",
        "_v7_5_apply.py",
    ],
    "input_data_jsons": [
        "_concepts_sample.json", "missed_gold_enrichment.json",
    ],
}

for sub, files in V74_GROUPS.items():
    sub_dir = V74 / sub
    sub_dir.mkdir(exist_ok=True)
    for f in files:
        mv(V74 / f, sub_dir / f)


V75 = ROOT / "v7_5_precision"

V75_GROUPS = {
    "pipeline_setup_and_load": [
        "_setup.py", "_load_llm.py", "_warmboot.py",
    ],
    "dossier_compute": [
        "_dossier_compute.py", "_dossier_helpers.py",
    ],
    "cascade_stages": [
        "_stage_a_prune.py", "_stage_b_composite.py", "_stage_c_listwise.py",
        "_stage_d_pointwise.py", "_stage_e_selection.py", "_stage_f_validation.py",
    ],
    "eval_and_notebook_build": [
        "_eval_and_save.py", "_build_notebook.py",
    ],
}

for sub, files in V75_GROUPS.items():
    sub_dir = V75 / sub
    sub_dir.mkdir(exist_ok=True)
    for f in files:
        mv(V75 / f, sub_dir / f)


# ─────────────────────────────────────────────────────────────────────────
# PHASE 7: rename outputs_from_363k_run and law_json_llm_output for parallelism
# ─────────────────────────────────────────────────────────────────────────
print("\n=== PHASE 7: rename LLM enrichment output folders for parallelism ===")

mv(ROOT / "outputs_from_363k_run", ROOT / "llm_enrichment_output_court_363k")
mv(ROOT / "law_json_llm_output", ROOT / "llm_enrichment_output_law_173k")


# ─────────────────────────────────────────────────────────────────────────
# PHASE 8: clean __pycache__ folders that got moved
# ─────────────────────────────────────────────────────────────────────────
print("\n=== PHASE 8: clean __pycache__ ===")
for pycache in ROOT.rglob("__pycache__"):
    if "research_repos" in str(pycache):
        continue
    rmrf(pycache, only_if_empty=False)


# ─────────────────────────────────────────────────────────────────────────
# Write the action log
# ─────────────────────────────────────────────────────────────────────────
log_path = ROOT / "scripts" / "drive_pull" / "_full_repo_reorg_log.json"
log_path.write_text(json.dumps({
    "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "actions": LOG,
    "total_actions": len(LOG),
    "moved": sum(1 for e in LOG if e["action"] == "moved"),
    "removed": sum(1 for e in LOG if e["action"] == "removed"),
    "missing_source": sum(1 for e in LOG if e["action"] == "missing_source"),
    "target_exists_skipped": sum(1 for e in LOG if e["action"] == "target_exists_skipped"),
}, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"\n{'=' * 70}")
print(f"DONE. {len(LOG)} actions logged to {log_path}")
print(f"  moved:              {sum(1 for e in LOG if e['action'] == 'moved')}")
print(f"  removed:            {sum(1 for e in LOG if e['action'] == 'removed')}")
print(f"  missing_source:     {sum(1 for e in LOG if e['action'] == 'missing_source')}")
print(f"  target_exists:      {sum(1 for e in LOG if e['action'] == 'target_exists_skipped')}")
