"""Full-repo organization pass.

1. ADD new notebook families discovered in repo + Downloads scan.
2. Clean up duplicate .ipynb copies in research/, law_json_llm_output/, outputs_from_363k_run/, repo root.
3. Move loose root-level diagnostic .py files into scripts/diagnostics_loose/.
4. Rebuild INDEX.
"""
import json, re, shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(r"e:\swiss_citation_extraction")
NB = ROOT / "notebooks"
INV = NB / "_inventory"

# ────────────────────────────────────────────────────────────────────────────
# Part 1: Add new notebook families
# ────────────────────────────────────────────────────────────────────────────
# (source path, target subfolder, canonical name, rationale)
NEW_ADDS = [
    # A. Current direction
    (r"C:\Users\samiul\Downloads\swiss_citation_precision_v1_not_working.ipynb",
     "01_current_direction_cascade_rerank_precision",
     "precision_v1_not_working_diagnostic",
     "precision_v1 attempt that failed — kept as failure record"),

    # B. v7.5 multi-query — HyDE test variant
    (r"C:\Users\samiul\Downloads\swiss_citation_anchor_funnel_v7_5_multiquery_with_hyde_result_still_same.ipynb",
     "02_v75_multiquery_canonical_pool",
     "pool_v75_multiquery_hyde_test_no_lift",
     "v7.5 + HyDE — no improvement, documented as ruled out"),

    # C. Anchor funnel evolution — earlier v7 parts
    (r"C:\Users\samiul\Downloads\swiss_citation_anchor_funnel_val001_v7_part3_output.ipynb",
     "03_anchor_funnel_evolution_v4_to_v74",
     "anchor_funnel_v7_part3_with_outputs",
     "v7 phases 1-3 with full execution outputs"),
    (r"C:\Users\samiul\Downloads\swiss_citation_anchor_funnel_val001_v7_part2_with_outputs.ipynb",
     "03_anchor_funnel_evolution_v4_to_v74",
     "anchor_funnel_v7_part2_with_outputs",
     "v7 phases 1-2 with execution outputs"),

    # D. Pre-v7.5 pipeline — many "latest_notebook" iterations + e2e variants
    (r"C:\Users\samiul\Downloads\latest__notebook_fixed_v2.ipynb",
     "04_pre_v75_pipeline_iterations",
     "pipeline_iteration_latest_fixed_v2",
     "predecessor of fixed_v3"),
    (r"C:\Users\samiul\Downloads\latest__notebook_fixed.ipynb",
     "04_pre_v75_pipeline_iterations",
     "pipeline_iteration_latest_fixed_v1",
     "first fixed iteration"),
    (r"C:\Users\samiul\Downloads\latest__notebook_with_all_output (2).ipynb",
     "04_pre_v75_pipeline_iterations",
     "pipeline_iteration_latest_with_all_output",
     "fattest iteration — 87 cells, 54 with outputs"),
    (r"C:\Users\samiul\Downloads\latest__notebook (6).ipynb",
     "04_pre_v75_pipeline_iterations",
     "pipeline_iteration_latest_base",
     "original `latest__notebook` base run"),
    (r"C:\Users\samiul\Downloads\swiss_law_pipeline_end_to_end_notebook_clean_paths.ipynb",
     "04_pre_v75_pipeline_iterations",
     "pipeline_end_to_end_clean_paths",
     "e2e pipeline variant with cleaner path config"),
    (r"C:\Users\samiul\Downloads\swiss_law_pipeline_end_to_end_notebook_explicit_paths.ipynb",
     "04_pre_v75_pipeline_iterations",
     "pipeline_end_to_end_explicit_paths",
     "e2e variant with explicit paths up-front"),
    (r"C:\Users\samiul\Downloads\swiss_law_pipeline_end_to_end_notebook_native.ipynb",
     "04_pre_v75_pipeline_iterations",
     "pipeline_end_to_end_native",
     "e2e notebook-native variant"),
    (r"C:\Users\samiul\Downloads\swiss_law_pipeline_end_to_end.ipynb",
     "04_pre_v75_pipeline_iterations",
     "pipeline_end_to_end_base",
     "original consolidated single-notebook pipeline"),

    # G. Law-DE enrichment — pre/optimized variants
    (r"C:\Users\samiul\Downloads\enrich_laws_de_qwen3_8b_kaggle_optimized (1).ipynb",
     "07_law_de_enrichment",
     "enrich_laws_de_qwen3_8b_kaggle_optimized",
     "Blackwell-optimized law-side enrichment (vLLM + FlashInfer + FP8)"),
    (r"C:\Users\samiul\Downloads\enrich_laws_de_qwen3_8b_kaggle_pre_optimization.ipynb",
     "07_law_de_enrichment",
     "enrich_laws_de_qwen3_8b_kaggle_pre_optimization",
     "law enrichment notebook before Blackwell optimization"),

    # H. Court LLM descriptor — T4 dual-GPU variants
    (r"C:\Users\samiul\Downloads\notebook_safe_2t4_vllm.ipynb",
     "08_court_llm_descriptor_extraction",
     "court_llm_descriptor_safe_2t4_vllm",
     "safe 2xT4 vLLM variant of the court LLM extractor"),
    (r"C:\Users\samiul\Downloads\notebook_dual_t4_vllm.ipynb",
     "08_court_llm_descriptor_extraction",
     "court_llm_descriptor_dual_t4_vllm",
     "dual T4 vLLM variant"),

    # I. Court concept enrichment — robust_v4 predecessor
    (r"C:\Users\samiul\Downloads\court_citation_concept_enrichment_robust_v4.ipynb",
     "09_court_citation_concept_enrichment",
     "court_concept_enrichment_robust_v4_predecessor",
     "robust v4 (predecessor to v5_t4_fixed)"),

    # J. Authority card enrichment — older qwen35 iterations + qwen3-8b-awq Kaggle variant
    (r"C:\Users\samiul\Downloads\notebook_qwen3_8b_awq_kaggle_local (1).ipynb",
     "10_authority_card_enrichment",
     "auth_cards_enrich_qwen3_8b_awq_kaggle_local",
     "Qwen3-8B-AWQ court authority card enrichment on Kaggle"),
    (r"C:\Users\samiul\Downloads\enrich_cards_qwen35_colab_fast_quality_json_optimized.ipynb",
     "10_authority_card_enrichment",
     "auth_cards_enrich_qwen35_fast_quality_json_optimized_pre_failure_capture",
     "predecessor before failure-capture refactor"),
    (r"C:\Users\samiul\Downloads\enrich_cards_qwen35_colab_updated_fast_json.ipynb",
     "10_authority_card_enrichment",
     "auth_cards_enrich_qwen35_updated_fast_json",
     "qwen35 enrichment, updated fast-json variant"),
    (r"C:\Users\samiul\Downloads\enrich_cards_qwen35_colab_vllm_fixed_pillowfix (2).ipynb",
     "10_authority_card_enrichment",
     "auth_cards_enrich_qwen35_vllm_fixed_pillowfix",
     "qwen35 enrichment with vLLM + Pillow fix"),
    (r"C:\Users\samiul\Downloads\enrich_cards_qwen35_colab_vllm_fixed.ipynb",
     "10_authority_card_enrichment",
     "auth_cards_enrich_qwen35_vllm_fixed",
     "qwen35 enrichment with vLLM (earliest)"),

    # M. Misc — qwen3-8b-awq text-to-json variants
    (r"C:\Users\samiul\Downloads\qwen3_8b_awq_vllm_fastest_accurate_text_to_json_t4_flashinfer_fixed.ipynb",
     "13_misc_kaggle_and_utilities",
     "kaggle_qwen3_8b_awq_vllm_text_to_json_t4_flashinfer_fixed",
     "T4 FlashInfer-fixed variant"),
    (r"C:\Users\samiul\Downloads\qwen3_8b_awq_vllm_fastest_accurate_text_to_json_t4_fixed.ipynb",
     "13_misc_kaggle_and_utilities",
     "kaggle_qwen3_8b_awq_vllm_text_to_json_t4_fixed",
     "T4-fixed variant"),
    (r"C:\Users\samiul\Downloads\qwen3_8b_awq_vllm_fastest_accurate_text_to_json.ipynb",
     "13_misc_kaggle_and_utilities",
     "kaggle_qwen3_8b_awq_vllm_text_to_json_base",
     "base variant"),
    (r"C:\Users\samiul\Downloads\fixed_qwen3_awq_text_to_json_no_flashinfer_with_fallback.ipynb",
     "13_misc_kaggle_and_utilities",
     "kaggle_qwen3_awq_text_to_json_no_flashinfer_with_fallback",
     "no-FlashInfer + transformers fallback"),
    (r"C:\Users\samiul\Downloads\notebook3500ee7363 (1).ipynb",
     "13_misc_kaggle_and_utilities",
     "kaggle_notebook_3500ee7363_base",
     "earlier iteration of 3500ee7363 family"),
    (r"C:\Users\samiul\Downloads\notebook0cf83146ef (1).ipynb",
     "13_misc_kaggle_and_utilities",
     "kaggle_notebook_0cf83146ef",
     "kaggle court LLM extractor variant"),
]

# Brand new subfolder: PDF research paper extraction
NEW_SUBFOLDERS = [
    "14_pdf_research_paper_extraction",
]
# Untitled49 = PDF parsing tooling
NEW_ADDS.append(
    (r"C:\Users\samiul\Downloads\Untitled49.ipynb",
     "14_pdf_research_paper_extraction",
     "pdf_extraction_marker_pymupdf_pdfminer",
     "PDF→text extraction via marker + PyPDF2 + PyMuPDF + pdfminer"),
)

# Create new subfolders
for sf in NEW_SUBFOLDERS:
    (NB / sf).mkdir(exist_ok=True)

added_log = []
for src_str, sub, canonical, why in NEW_ADDS:
    src = Path(src_str)
    if not src.exists():
        added_log.append({"src": src_str, "status": "missing_source"})
        continue
    target = NB / sub / f"{canonical}.ipynb"
    if target.exists():
        added_log.append({"src": src_str, "target": str(target), "status": "already_exists"})
        continue
    shutil.copy2(str(src), str(target))
    added_log.append({"src": src_str, "target": str(target), "status": "copied",
                      "why": why,
                      "mtime": datetime.fromtimestamp(src.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})

# ────────────────────────────────────────────────────────────────────────────
# Part 2: Remove duplicate .ipynb copies elsewhere in the repo
# ────────────────────────────────────────────────────────────────────────────
DUP_LOCATIONS = [
    ROOT / "research",
    ROOT / "law_json_llm_output",
    ROOT / "outputs_from_363k_run",
]
removed_log = []
for loc in DUP_LOCATIONS:
    if not loc.exists():
        continue
    for nb_file in loc.glob("*.ipynb"):
        nb_file.unlink()
        removed_log.append(str(nb_file))

# Root-level loose .ipynb files (single file: swiss_citation_anchor_funnel_val001.ipynb)
for nb_file in ROOT.glob("*.ipynb"):
    nb_file.unlink()
    removed_log.append(str(nb_file))

# ────────────────────────────────────────────────────────────────────────────
# Part 3: Move loose root-level diagnostic .py scripts into scripts/diagnostics_loose/
# ────────────────────────────────────────────────────────────────────────────
LOOSE_PY = [
    "_gen_rerank_final_diag_notebook.py",
    "_run_phase3_local.py",
    "audit_repaired.py",
    "build_rerun_input.py",
    "check_coverage.py",
    "check_failure_overlap.py",
    "classify_invalid.py",
    "diag_cache.py",
    "diag_judge.py",
    "diag_recall.py",
    "extract_failures.py",
    "find_offcontract_roles.py",
    "fix_all_invalid.py",
    "fix_unrecoverable.py",
    "investigate_quality.py",
    "investigate_remaining.py",
    "merge_engine_dead_rerun.py",
    "merge_repaired.py",
    "normalize_roles.py",
    "repair_failures.py",
    "spotcheck.py",
    "update_notebook_for_rerun.py",
    "verify_final.py",
]
DIAG_DIR = ROOT / "scripts" / "diagnostics_loose"
DIAG_DIR.mkdir(parents=True, exist_ok=True)
py_moves = []
for name in LOOSE_PY:
    src = ROOT / name
    if not src.exists():
        continue
    target = DIAG_DIR / name
    if target.exists():
        py_moves.append({"file": name, "status": "target_exists_skipped"})
        continue
    shutil.move(str(src), str(target))
    py_moves.append({"file": name, "status": "moved"})

# Loose junk files at root
JUNK = [
    "CUserssamiulDownloadsnb_dump.txt",
    "debug_25299.txt",
    "debug_42125.txt",
    "failures_unrepaired.jsonl",   # 0 bytes
]
ARCHIVE_DIR = ROOT / "cache_endgame" / "loose_root_artifacts"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
junk_moves = []
for name in JUNK:
    src = ROOT / name
    if not src.exists():
        continue
    target = ARCHIVE_DIR / name
    if target.exists():
        junk_moves.append({"file": name, "status": "target_exists_skipped"})
        continue
    shutil.move(str(src), str(target))
    junk_moves.append({"file": name, "status": "moved"})

# Move the large *.jsonl artifacts and the zip into archives
LARGE_ARTIFACTS = [
    "failures_repaired.jsonl",
    "failures_with_text.jsonl",
    "research-20260509T142853Z-3-001.zip",
]
artifact_moves = []
for name in LARGE_ARTIFACTS:
    src = ROOT / name
    if not src.exists():
        continue
    target = ARCHIVE_DIR / name
    if target.exists():
        artifact_moves.append({"file": name, "status": "target_exists_skipped"})
        continue
    shutil.move(str(src), str(target))
    artifact_moves.append({"file": name, "status": "moved"})

# Remove stale __pycache__
pycache = ROOT / "__pycache__"
pycache_removed = False
if pycache.exists():
    shutil.rmtree(pycache)
    pycache_removed = True

# ────────────────────────────────────────────────────────────────────────────
# Write reorg log
# ────────────────────────────────────────────────────────────────────────────
log = {
    "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "notebooks_added": added_log,
    "ipynb_removed_from_outside_notebooks_dir": removed_log,
    "py_scripts_moved_to_scripts_diagnostics_loose": py_moves,
    "junk_files_archived": junk_moves,
    "large_artifacts_archived": artifact_moves,
    "pycache_removed": pycache_removed,
}
(INV / "_full_repo_organize_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Notebooks added: {sum(1 for e in added_log if e['status'] == 'copied')}")
print(f"Already-existing targets (skipped): {sum(1 for e in added_log if e['status'] == 'already_exists')}")
print(f"Missing sources: {sum(1 for e in added_log if e['status'] == 'missing_source')}")
print(f"Dup .ipynb deleted from repo: {len(removed_log)}")
print(f"Loose .py scripts moved: {sum(1 for e in py_moves if e['status'] == 'moved')}")
print(f"Junk files archived: {sum(1 for e in junk_moves if e['status'] == 'moved')}")
print(f"Large artifacts archived: {sum(1 for e in artifact_moves if e['status'] == 'moved')}")
print(f"__pycache__ removed: {pycache_removed}")
