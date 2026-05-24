"""Merge the 66 repaired enrichments back into the main jsonl.

For each failed source_row, replace the corresponding line with a record that:
- Uses the repaired enrichment in llm_enrichment
- Removes the _descriptor_error key
- Sets llm_quality.json_valid = True
- Updates llm_generation.status to "ok_after_manual_repair", clears error, drops raw_output

The output is written atomically: write to a .tmp, then move to replace.
"""
import json
import os
import shutil

MAIN = r"E:\swiss_citation_extraction\law_json_llm_output\law_llm_descriptors_0000000_all.jsonl"
REPAIRED = r"E:\swiss_citation_extraction\failures_repaired.jsonl"
BACKUP = MAIN + ".bak"
TMP = MAIN + ".tmp"

# Load repaired enrichments
repairs = {}
with open(REPAIRED, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        repairs[rec["_source_row"]] = rec["repaired_enrichment"]

print(f"Loaded {len(repairs)} repairs")

# Backup the original (if no backup yet)
if not os.path.exists(BACKUP):
    shutil.copyfile(MAIN, BACKUP)
    print(f"Backed up to {BACKUP}")
else:
    print(f"Backup already exists at {BACKUP} (not overwriting)")

n_total = 0
n_replaced = 0
with open(MAIN, "r", encoding="utf-8") as src, open(TMP, "w", encoding="utf-8") as dst:
    for line in src:
        if not line.strip():
            continue
        n_total += 1
        obj = json.loads(line)
        sr = obj["_source_row"]
        if sr in repairs:
            new_enrich = dict(repairs[sr])
            obj["llm_enrichment"] = new_enrich
            obj["llm_quality"] = {
                "json_valid": True,
                "terms_grounded_pct": (obj.get("llm_quality") or {}).get("terms_grounded_pct", 1.0),
                "boilerplate_role": (obj.get("llm_quality") or {}).get("boilerplate_role", False),
            }
            gen = dict(obj.get("llm_generation") or {})
            gen["status"] = "ok_after_manual_repair"
            gen["error"] = None
            gen["raw_output"] = None
            obj["llm_generation"] = gen
            n_replaced += 1
        dst.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Read {n_total} records, replaced {n_replaced}")

# Atomic replace
os.replace(TMP, MAIN)
print(f"Wrote {MAIN}")
