"""Recover the 9 of 11 lost notebook families from Downloads alternates."""
import json, shutil
from pathlib import Path
from datetime import datetime

ROOT = Path(r"e:\swiss_citation_extraction\notebooks")
INV = ROOT / "_inventory"

# (source path, target subfolder, canonical name)
RECOVERIES = [
    (r"C:\Users\samiul\Downloads\swiss_citation_legalmalr_cascade_v1.ipynb",
     "01_current_direction_cascade_rerank_precision", "cascade_legalmalr_v1_base"),

    (r"C:\Users\samiul\Downloads\swiss_citation_precision_v1_top_gold_from_50k_candidates.ipynb",
     "01_current_direction_cascade_rerank_precision", "precision_v1_top_gold_from_50k_candidates"),

    (r"C:\Users\samiul\Downloads\swiss_citation_anchor_funnel_val001_v4 (1).ipynb",
     "03_anchor_funnel_evolution_v4_to_v74", "anchor_funnel_v4_val001"),

    (r"C:\Users\samiul\Downloads\swiss_citation_anchor_funnel_val001_v5.ipynb",
     "03_anchor_funnel_evolution_v4_to_v74", "anchor_funnel_v5_val001"),

    (r"C:\Users\samiul\Downloads\swiss_citation_anchor_funnel_val001_v6.ipynb",
     "03_anchor_funnel_evolution_v4_to_v74", "anchor_funnel_v6_val001"),

    (r"C:\Users\samiul\Downloads\swiss_citation_endgame_colab.ipynb",
     "04_pre_v75_pipeline_iterations", "endgame_colab_full_pipeline"),

    (r"C:\Users\samiul\Downloads\embed_unified_corpus_qwen3_8b_blackwell_colab (3).ipynb",
     "05_embedding_and_retrieval_base", "embed_unified_corpus_qwen3_8b_blackwell"),

    (r"C:\Users\samiul\Downloads\retrieve_unified_corpus_qwen3_8b_colab (3).ipynb",
     "05_embedding_and_retrieval_base", "retrieve_unified_corpus_qwen3_8b"),
]

log = []
for src_str, sub, canonical in RECOVERIES:
    src = Path(src_str)
    if not src.exists():
        log.append({"src": src_str, "status": "missing"})
        continue
    target = ROOT / sub / f"{canonical}.ipynb"
    if target.exists():
        log.append({"src": src_str, "target": str(target), "status": "already_exists"})
        continue
    shutil.copy2(str(src), str(target))
    log.append({"src": src_str, "target": str(target), "status": "copied",
                "mtime": datetime.fromtimestamp(src.stat().st_mtime).strftime("%Y-%m-%d %H:%M")})

# Append to manifest
mf = INV / "_reorg_manifest.json"
m = json.loads(mf.read_text(encoding="utf-8"))
m["recovered"] = log
m["lost_no_recovery"] = ["swiss_citation_final_2026", "encode_queries_qwen3_8b_colab", "court_citation_concept_enrichment (0 KB original)"]
mf.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")

for e in log:
    print(e)
print(f"\nRecovered: {sum(1 for e in log if e['status'] == 'copied')}")
print(f"Already present: {sum(1 for e in log if e['status'] == 'already_exists')}")
print(f"Missing source: {sum(1 for e in log if e['status'] == 'missing')}")
