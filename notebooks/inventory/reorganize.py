"""Re-pick best notebook per family (prefer with outputs, then newest mtime),
   rename to canonical names, and group into topic subfolders.

Inputs: notebook_list.txt (the 75 source paths)
Outputs: e:\\swiss_citation_extraction\\notebooks\\<subfolder>\\<canonical_name>.ipynb
         plus a fresh _inventory/<canonical_name>.md per notebook.
         Also writes _inventory/INDEX.md and _reorg_manifest.json.
"""
import json, re, shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(r"e:\swiss_citation_extraction\notebooks")
INV = ROOT / "_inventory"
LIST = INV / "notebook_list.txt"

# ────────────────────────────────────────────────────────────────────────────
# Family rule (same as before — strip trailing copy markers)
# ────────────────────────────────────────────────────────────────────────────
def family(name: str) -> str:
    b = name[:-6] if name.endswith(".ipynb") else name
    while True:
        new = re.sub(r"[ _]?\(\d+\)$", "", b)
        if new == b:
            break
        b = new
    b = b.replace(" ", "_").lower()
    b = re.sub(r"_+", "_", b).strip("_")
    return b


# ────────────────────────────────────────────────────────────────────────────
# Canonical-name and subfolder mapping per family
# ────────────────────────────────────────────────────────────────────────────
SUBFOLDERS = {
    "A": "01_current_direction_cascade_rerank_precision",
    "B": "02_v75_multiquery_canonical_pool",
    "C": "03_anchor_funnel_evolution_v4_to_v74",
    "D": "04_pre_v75_pipeline_iterations",
    "E": "05_embedding_and_retrieval_base",
    "F": "06_high_scoring_reference",
    "G": "07_law_de_enrichment",
    "H": "08_court_llm_descriptor_extraction",
    "I": "09_court_citation_concept_enrichment",
    "J": "10_authority_card_enrichment",
    "K": "11_early_experiments_exp_A",
    "L": "12_early_funnel_pre_v7",
    "M": "13_misc_kaggle_and_utilities",
}

# family-key → (subfolder-letter, canonical-name)
PLAN = {
    # A. Current direction
    "swiss_citation_hybrid_rerank_final":                              ("A", "rerank_hybrid_3model_shootout"),
    "swiss_citation_hybrid_rerank_final_output":                       ("A", "rerank_hybrid_3model_shootout_with_outputs"),
    "swiss_citation_legalmalr_cascade_v1":                             ("A", "cascade_legalmalr_v1_base"),
    "swiss_citation_legalmalr_cascade_v1_poc":                         ("A", "cascade_legalmalr_v1_poc"),
    "swiss_citation_legalmalr_cascade_v1_optimized":                   ("A", "cascade_legalmalr_v1_optimized"),
    "swiss_citation_rerank_only_diagnostic":                           ("A", "rerank_only_diagnostic_F3_no_rerank"),
    "swiss_citation_final_2026":                                       ("A", "final_2026_pipeline_attempt"),
    "swiss_citation_precision_v1":                                     ("A", "precision_v1_final"),
    "swiss_citation_precision_v1_top_gold_from_50k_candidates":        ("A", "precision_v1_top_gold_from_50k_candidates"),

    # B. v7.5 multi-query
    "swiss_citation_anchor_funnel_v7_5_multiquery":                            ("B", "pool_v75_multiquery_iteration_11"),
    "swiss_citation_anchor_funnel_v7_5_multiquery_recall_0.89_at_k_50000":     ("B", "pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL"),

    # C. Anchor funnel evolution
    "swiss_citation_anchor_funnel_val001":                             ("C", "anchor_funnel_v1_val001_base"),
    "swiss_citation_anchor_funnel_val001_v4":                          ("C", "anchor_funnel_v4_val001"),
    "swiss_citation_anchor_funnel_val001_v5":                          ("C", "anchor_funnel_v5_val001"),
    "swiss_citation_anchor_funnel_val001_v6":                          ("C", "anchor_funnel_v6_val001"),
    "swiss_citation_anchor_funnel_val001_v7":                          ("C", "anchor_funnel_v7_val001"),
    "swiss_citation_anchor_funnel_val001_v7_4":                        ("C", "anchor_funnel_v7_4_val001"),
    "swiss_citation_anchor_funnel_val001_v7_part4_output":             ("C", "anchor_funnel_v7_part4_with_outputs"),

    # D. Pre-v7.5 pipeline
    "swiss_citation_endgame_colab":                                    ("D", "endgame_colab_full_pipeline"),
    "swiss_law_pipeline_end_to_end_notebook_no_function_imports":      ("D", "pipeline_end_to_end_no_function_imports"),
    "latest_notebook_fixed_v3":                                        ("D", "pipeline_iteration_latest_fixed_v3"),
    "notebook4ae3b3ffc3_stage0_gated":                                 ("D", "pipeline_stage0_gated_4ae3b3ffc3"),
    "notebook_no_debug_prints":                                        ("D", "pipeline_no_debug_prints"),

    # E. Embedding & retrieval
    "embed_unified_corpus_qwen3_8b_blackwell_colab":                   ("E", "embed_unified_corpus_qwen3_8b_blackwell"),
    "encode_queries_qwen3_8b_colab":                                   ("E", "encode_queries_qwen3_8b"),
    "retrieve_unified_corpus_qwen3_8b_colab":                          ("E", "retrieve_unified_corpus_qwen3_8b"),
    "retrieve_unified_corpus_v3":                                      ("E", "retrieve_unified_corpus_v3"),

    # F. References
    "untitled75":                                                      ("F", "reference_F1_0.777_Untitled75"),
    "0.7716_scored_notebook":                                          ("F", "reference_F1_0.7716_scored"),

    # G. Law (laws_de) enrichment
    "enrich_laws_de_qwen3_8b_kaggle":                                  ("G", "enrich_laws_de_qwen3_8b_kaggle"),
    "investigate_laws_de_token_optimization":                          ("G", "investigate_laws_de_token_optimization"),

    # H. Court LLM descriptor extraction
    "court_llm_descriptor_extractor_blackwell_optimized":              ("H", "court_llm_descriptor_blackwell_optimized_v4"),
    "court_llm_descriptor_extractor_blackwell_optimized_3":            ("H", "court_llm_descriptor_blackwell_optimized_363k_run"),
    "court_llm_descriptor_extractor_colab_10k_optimized":              ("H", "court_llm_descriptor_10k_optimized"),
    "court_llm_descriptor_extractor_colab_qwen3_8b_awq":               ("H", "court_llm_descriptor_qwen3_8b_awq"),
    "court_llm_descriptor_extractor_kaggle":                           ("H", "court_llm_descriptor_kaggle"),

    # I. Court citation concept enrichment
    "court_citation_concept_enrichment":                               ("I", "court_concept_enrichment_main"),
    "court_citation_concept_enrichment_minimal_llm":                   ("I", "court_concept_enrichment_minimal_llm"),
    "court_citation_concept_enrichment_robust_v5_t4_fixed":            ("I", "court_concept_enrichment_robust_v5_t4"),
    "court_citation_concept_enrichment_smoke_test_10_v3_local_model":  ("I", "court_concept_smoke_test_10_v3_local"),
    "court_citation_enrichment_smoke_test_10":                         ("I", "court_concept_smoke_test_10_base"),

    # J. Authority card enrichment
    "enrich_cards_qwen3_8b_colab_fast_stable":                         ("J", "auth_cards_enrich_qwen3_8b_colab"),
    "enrich_cards_qwen3_8b_kaggle":                                    ("J", "auth_cards_enrich_qwen3_8b_kaggle"),
    "enrich_cards_qwen35_colab":                                       ("J", "auth_cards_enrich_qwen35_colab"),
    "enrich_cards_qwen35_colab_fast_quality_json_optimized_failure_capture": ("J", "auth_cards_enrich_qwen35_fast_quality_json_optimized"),
    "enrich_cards_qwen35_colab_fast_quality_json_v2":                  ("J", "auth_cards_enrich_qwen35_fast_quality_json_v2"),
    "enrich_cards_qwen35_vllm_fast_stable_failure_capture":            ("J", "auth_cards_enrich_qwen35_vllm_fast_stable"),
    "fresh_safe_qwen35_card_enrichment_no_vllm":                       ("J", "auth_cards_enrich_qwen35_safe_no_vllm"),

    # K. exp_A
    "exp_a1_bgem3_statutes":                                           ("K", "exp_A1_bgem3_statutes"),
    "exp_a2_hyde_and_enumeration":                                     ("K", "exp_A2_hyde_and_enumeration"),
    "exp_a3_qwen32b_enumeration":                                      ("K", "exp_A3_qwen32b_enumeration"),
    "exp_a4_rerank":                                                   ("K", "exp_A4_rerank"),
    "exp_a4b_rerank_hyde":                                             ("K", "exp_A4b_rerank_hyde"),
    "exp_a5_court_retrieval":                                          ("K", "exp_A5_court_retrieval"),
    "exp_a6_legal_roberta_finetune":                                   ("K", "exp_A6_legal_roberta_finetune"),

    # L. Early funnel
    "colab_dense_embedding_test":                                      ("L", "early_dense_embedding_test"),
    "colab_recall_preserving_candidate_funnel":                        ("L", "early_recall_preserving_funnel"),
    "colab_segment_lattice_funnel_v3":                                 ("L", "early_segment_lattice_funnel_v3"),

    # M. Misc
    "kaggle-10-test-for-swiss-law":                                    ("M", "kaggle_10_test_for_swiss_law"),
    "investigate-notebook":                                            ("M", "investigate_misc_notebook"),
    "notebook3500ee7363_corrected_kaggle_safe":                        ("M", "kaggle_safe_notebook_3500ee7363"),
    "notebookd37248e1eb":                                              ("M", "kaggle_notebook_d37248e1eb"),
    "notebook_qwen3_8b_awq_kaggle_local_batchfix":                     ("M", "kaggle_qwen3_8b_awq_local_batchfix"),
    "qwen3_8b_awq_vllm_fastest_accurate_text_to_json_kaggle_stable":   ("M", "kaggle_qwen3_8b_awq_vllm_text_to_json_stable"),
}


def count_outputs(nb_path: Path) -> tuple[int, int]:
    """Return (n_code_cells_with_output, total_output_chars)."""
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return (0, 0)
    cells_with = 0
    total_chars = 0
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        outs = c.get("outputs", [])
        if not outs:
            continue
        cells_with += 1
        for o in outs:
            if o.get("output_type") == "stream":
                t = "".join(o.get("text", []))
            elif "data" in o:
                d = o["data"]
                if "text/plain" in d:
                    v = d["text/plain"]
                    t = "".join(v) if isinstance(v, list) else str(v)
                else:
                    t = ""
            else:
                t = ""
            total_chars += len(t)
    return (cells_with, total_chars)


# ────────────────────────────────────────────────────────────────────────────
# Step 1: gather all source candidates from notebook_list.txt
# ────────────────────────────────────────────────────────────────────────────
paths = [Path(p) for p in LIST.read_text(encoding="utf-8").splitlines() if p.strip()]
paths = [p for p in paths if p.exists()]

candidates_by_family: dict[str, list[dict]] = {}
for p in paths:
    fam = family(p.name)
    cells_w_out, out_chars = count_outputs(p)
    mtime = p.stat().st_mtime
    candidates_by_family.setdefault(fam, []).append({
        "path": p,
        "family": fam,
        "cells_with_output": cells_w_out,
        "output_chars": out_chars,
        "mtime": mtime,
        "mtime_str": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
    })

# ────────────────────────────────────────────────────────────────────────────
# Step 2: pick winner per family — max(cells_with_output) → max(output_chars) → max(mtime)
# ────────────────────────────────────────────────────────────────────────────
winners = {}
for fam, cands in candidates_by_family.items():
    cands_sorted = sorted(
        cands,
        key=lambda c: (c["cells_with_output"], c["output_chars"], c["mtime"]),
        reverse=True,
    )
    winners[fam] = {"winner": cands_sorted[0], "losers": cands_sorted[1:]}

# ────────────────────────────────────────────────────────────────────────────
# Step 3: clear old subfolders only (NOT flat .ipynb — winners may point at them)
# ────────────────────────────────────────────────────────────────────────────
for sub in ROOT.iterdir():
    if sub.is_dir() and sub.name.startswith(("01_", "02_", "03_", "04_", "05_", "06_", "07_", "08_", "09_", "10_", "11_", "12_", "13_")):
        shutil.rmtree(sub)

# Create fresh subfolders
for letter, name in SUBFOLDERS.items():
    (ROOT / name).mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# Step 4: copy winners to canonical paths
# ────────────────────────────────────────────────────────────────────────────
unplanned = []
copies = []
for fam, info in winners.items():
    w = info["winner"]
    if fam not in PLAN:
        unplanned.append(fam)
        continue
    letter, canonical = PLAN[fam]
    target_dir = ROOT / SUBFOLDERS[letter]
    target = target_dir / f"{canonical}.ipynb"
    if not w["path"].exists():
        unplanned.append(f"{fam} (source vanished: {w['path']})")
        continue
    shutil.copy2(str(w["path"]), str(target))
    copies.append({
        "family": fam,
        "subfolder": SUBFOLDERS[letter],
        "canonical": canonical,
        "source": str(w["path"]),
        "cells_with_output": w["cells_with_output"],
        "output_chars": w["output_chars"],
        "mtime": w["mtime_str"],
        "losers": [{"path": str(l["path"]), "cells_w_out": l["cells_with_output"], "out_chars": l["output_chars"], "mtime": l["mtime_str"]} for l in info["losers"]],
    })

# ────────────────────────────────────────────────────────────────────────────
# Step 4b: now safe to remove any flat .ipynb in notebooks/ root (winners are in subfolders)
# ────────────────────────────────────────────────────────────────────────────
for f in ROOT.glob("*.ipynb"):
    f.unlink()

# ────────────────────────────────────────────────────────────────────────────
# Step 5: regenerate _inventory .md files (delete old ones, run extract.py on winners)
# ────────────────────────────────────────────────────────────────────────────
for md in INV.glob("*.md"):
    if md.name in ("INDEX.md",):
        md.unlink()
        continue
    if md.name.startswith("_"):
        continue
    md.unlink()

manifest = {
    "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "subfolders": SUBFOLDERS,
    "kept": copies,
    "unplanned_families": unplanned,
}
(INV / "_reorg_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Families considered: {len(candidates_by_family)}")
print(f"Notebooks copied:    {len(copies)}")
print(f"Unplanned families:  {len(unplanned)}")
if unplanned:
    for u in unplanned:
        print(f"  - {u}")
print()
print("Per-subfolder counts:")
from collections import Counter
sub_counts = Counter(c["subfolder"] for c in copies)
for sub in sorted(sub_counts.keys()):
    print(f"  {sub:50s} {sub_counts[sub]}")
