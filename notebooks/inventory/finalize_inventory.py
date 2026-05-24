"""Walk all .ipynb in notebooks/<subfolder>/ and produce one .md per notebook.
   Then build a fresh INDEX.md that uses the subfolder structure as the grouping."""
import json, re, sys, subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(r"e:\swiss_citation_extraction\notebooks")
INV = ROOT / "_inventory"

# Reuse extract.py logic by importing it
sys.path.insert(0, str(INV))
import extract as ext  # noqa: E402

# Map subfolder → (category label, ordinal for sort)
SUB_LABELS = {
    "01_current_direction_cascade_rerank_precision":   ("A. Current direction (cascade / rerank / precision / diagnostic)", 1),
    "02_v75_multiquery_canonical_pool":                ("B. v7.5 multi-query funnel — CANONICAL recall-0.89 pool", 2),
    "03_anchor_funnel_evolution_v4_to_v74":            ("C. Anchor funnel evolution (v1 -> v7.4)", 3),
    "04_pre_v75_pipeline_iterations":                  ("D. Endgame & pre-v7.5 pipeline iterations", 4),
    "05_embedding_and_retrieval_base":                 ("E. Embedding & retrieval base", 5),
    "06_high_scoring_reference":                       ("F. High-scoring reference notebooks", 6),
    "07_law_de_enrichment":                            ("G. Law (laws_de) enrichment", 7),
    "08_court_llm_descriptor_extraction":              ("H. Court LLM descriptor extraction (363k run)", 8),
    "09_court_citation_concept_enrichment":            ("I. Court citation concept enrichment", 9),
    "10_authority_card_enrichment":                    ("J. Authority card enrichment (qwen3-8b + qwen3.5)", 10),
    "11_early_experiments_exp_A":                      ("K. Early experiments (exp_A1 - A6)", 11),
    "12_early_funnel_pre_v7":                          ("L. Early funnel experiments (pre-v7)", 12),
    "13_misc_kaggle_and_utilities":                    ("M. Misc kaggle & utility scaffolds", 13),
    "14_pdf_research_paper_extraction":                ("N. PDF / research-paper extraction tooling", 14),
}

# Clear old md files in _inventory (except INDEX, manifests, scripts)
for md in INV.glob("*.md"):
    if md.name in ("INDEX.md",):
        md.unlink()
        continue
    if md.name.startswith("_"):
        continue
    md.unlink()

# Walk subfolders, run extract, gather metadata
rows_by_sub: dict[str, list[dict]] = {}
for sub_name in sorted(SUB_LABELS.keys()):
    sub_dir = ROOT / sub_name
    if not sub_dir.exists():
        continue
    for nb in sorted(sub_dir.glob("*.ipynb")):
        mtime_str = datetime.fromtimestamp(nb.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        try:
            md, info = ext.process(nb, mtime_str)
        except Exception as e:
            print(f"FAIL extract on {nb}: {e}", file=sys.stderr)
            continue
        # Output md file (one per notebook), keep extension-less base name
        md_name = nb.stem + ".md"
        (INV / md_name).write_text(md, encoding="utf-8")
        rows_by_sub.setdefault(sub_name, []).append({
            "ipynb_rel": f"{sub_name}/{nb.name}",
            "ipynb_name": nb.name,
            "md_name": md_name,
            "mtime": mtime_str,
            "n_cells": info["n_cells"],
        })

# Build INDEX
out = []
out.append("# Swiss Citation Notebooks — Index\n")
out.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} after final reorganization. "
           f"Each notebook lives under `notebooks/<subfolder>/` with a canonical descriptive name. "
           f"Click the inventory link for the per-notebook code+output dump._\n\n")

total = sum(len(rows_by_sub.get(s, [])) for s in SUB_LABELS)
out.append(f"**Total notebooks:** {total}\n\n")
out.append("## Subfolder summary\n\n")
out.append("| # | Subfolder | Count |\n|---|---|---|\n")
for sub, (label, _) in sorted(SUB_LABELS.items(), key=lambda kv: kv[1][1]):
    cnt = len(rows_by_sub.get(sub, []))
    out.append(f"| {label.split('.')[0]} | `{sub}/` | {cnt} |\n")
out.append("\n---\n\n")

for sub, (label, _) in sorted(SUB_LABELS.items(), key=lambda kv: kv[1][1]):
    rows = rows_by_sub.get(sub, [])
    out.append(f"## {label}\n\n")
    out.append(f"`notebooks/{sub}/`\n\n")
    if not rows:
        out.append("_(empty)_\n\n")
        continue
    rows_sorted = sorted(rows, key=lambda r: r["mtime"], reverse=True)
    out.append("| Mtime | Code cells | Notebook | Inventory |\n|---|---|---|---|\n")
    for r in rows_sorted:
        out.append(f"| {r['mtime']} | {r['n_cells']} | [{r['ipynb_name']}](../{r['ipynb_rel']}) | [{r['md_name']}]({r['md_name']}) |\n")
    out.append("\n")

(INV / "INDEX.md").write_text("".join(out), encoding="utf-8")

print(f"Total notebooks: {total}")
for sub, (label, _) in sorted(SUB_LABELS.items(), key=lambda kv: kv[1][1]):
    cnt = len(rows_by_sub.get(sub, []))
    print(f"  {sub:60s} {cnt}")
print()
print("INDEX.md updated.")
