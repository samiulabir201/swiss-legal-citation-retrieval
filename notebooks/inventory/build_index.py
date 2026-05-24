"""Build a consolidated INDEX.md grouping the kept notebooks by topic."""
import json, re
from pathlib import Path
from datetime import datetime

INV = Path(r"e:\swiss_citation_extraction\notebooks\_inventory")
NB = Path(r"e:\swiss_citation_extraction\notebooks")
manifest = json.loads((INV / "_dedupe_manifest.json").read_text(encoding="utf-8"))

# Hand-curated category rules — first match wins
CATEGORIES = [
    ("A. Current direction (cascade / rerank / precision / diagnostic)", [
        r"swiss_citation_legalmalr_cascade",
        r"swiss_citation_hybrid_rerank",
        r"swiss_citation_rerank_only_diagnostic",
        r"swiss_citation_precision",
        r"swiss_citation_final_2026",
    ]),
    ("B. v7.5 multi-query funnel (canonical recall-0.89 pool)", [
        r"swiss_citation_anchor_funnel_v7_5_multiquery",
    ]),
    ("C. Anchor funnel evolution (v4 → v7.4)", [
        r"swiss_citation_anchor_funnel_val001",
    ]),
    ("D. Endgame & pre-v7.5 pipeline iterations", [
        r"swiss_citation_endgame_colab",
        r"swiss_law_pipeline_end_to_end",
        r"latest_notebook_fixed_v3",
        r"notebook4ae3b3ffc3_stage0_gated",
        r"notebook_no_debug_prints",
    ]),
    ("E. Embedding & retrieval base", [
        r"embed_unified_corpus",
        r"encode_queries_qwen3_8b_colab",
        r"retrieve_unified_corpus",
    ]),
    ("F. High-scoring reference notebooks", [
        r"Untitled75",
        r"0\.7716",
    ]),
    ("G. Law (laws_de) enrichment", [
        r"enrich_laws_de",
        r"investigate_laws_de_token_optimization",
    ]),
    ("H. Court LLM descriptor extraction (the 363k run)", [
        r"court_llm_descriptor_extractor",
    ]),
    ("I. Court citation concept enrichment", [
        r"court_citation_concept_enrichment",
        r"court_citation_enrichment_smoke_test",
    ]),
    ("J. Authority card enrichment (qwen3-8b and qwen3.5 variants)", [
        r"enrich_cards_qwen3_8b",
        r"enrich_cards_qwen35",
        r"fresh_safe_qwen35_card_enrichment",
    ]),
    ("K. Early experiments (exp_A1 → A6)", [
        r"^exp_A[1-6]",
    ]),
    ("L. Early funnel experiments (pre-v7)", [
        r"colab_segment_lattice_funnel",
        r"colab_recall_preserving_candidate_funnel",
        r"colab_dense_embedding_test",
    ]),
    ("M. Misc / utility / kaggle scaffolds", [
        r"notebook_qwen3_8b_awq_kaggle_local",
        r"qwen3_8b_awq_vllm_fastest_accurate_text_to_json",
        r"kaggle-10-test-for-swiss-law",
        r"investigate-notebook",
        r"notebook3500ee7363",
        r"notebookd37248e1eb",
    ]),
]


def categorize(target_name: str) -> str:
    for cat, patterns in CATEGORIES:
        for p in patterns:
            if re.search(p, target_name, re.I):
                return cat
    return "Z. Unclassified"


# Read each kept entry; pull a 1-line synopsis from the .md (the "End findings" first sentence)
def one_line_synopsis(md_path: Path) -> str:
    if not md_path.exists():
        return "(no inventory)"
    text = md_path.read_text(encoding="utf-8", errors="replace")
    # Look for first metric-bearing line in "End findings" section
    m = re.search(r"## End findings\s*\n(.*?)(?=\n## |\Z)", text, re.S)
    if m:
        body = m.group(1).strip()
        lines = [ln for ln in body.splitlines() if ln.strip() and not ln.startswith("_")]
        if lines:
            return lines[0][:160].lstrip("- `").rstrip("` ")
    return "(see inventory)"


groups: dict[str, list[dict]] = {}
for entry in manifest["kept"]:
    target_name = Path(entry["target"]).name
    cat = categorize(target_name)
    md_name = target_name[:-6] + ".md"
    syn = one_line_synopsis(INV / md_name)
    groups.setdefault(cat, []).append({
        "ipynb": target_name,
        "md": md_name,
        "mtime": entry["mtime"],
        "synopsis": syn,
    })

out = ["# Swiss Citation Notebooks — Index\n"]
out.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from {sum(len(v) for v in groups.values())} deduped notebooks. Each row links to the per-notebook code+output dump in `_inventory/`._\n")
out.append(f"_See also `_dedupe_manifest.json` for the kept/dropped manifest and `_extract_log.json` for cell counts & detected file paths._\n")

for cat in sorted(groups.keys()):
    rows = sorted(groups[cat], key=lambda r: r["mtime"], reverse=True)
    out.append(f"\n## {cat}\n")
    out.append(f"| Mtime | Notebook | Inventory | Synopsis (first detected metric line) |\n")
    out.append(f"|---|---|---|---|\n")
    for r in rows:
        out.append(f"| {r['mtime']} | [{r['ipynb']}](../{r['ipynb']}) | [{r['md']}]({r['md']}) | {r['synopsis']} |\n")

(INV / "INDEX.md").write_text("".join(out), encoding="utf-8")
print(f"INDEX.md written with {sum(len(v) for v in groups.values())} notebooks across {len(groups)} categories")
for cat in sorted(groups.keys()):
    print(f"  {cat}: {len(groups[cat])} notebooks")
