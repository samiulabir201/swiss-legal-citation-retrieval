"""Re-attempt repair for the 21 ok_empty_unrecoverable records using their
original raw_output from the pre-fix backup."""
import json
import os

from repair_failures import repair_raw_output, normalize as normalize_enrichment
from fix_all_invalid import coerce_role

MAIN = r"E:\swiss_citation_extraction\law_json_llm_output\law_llm_descriptors_0000000_all.jsonl"
BAK = MAIN + ".bak2"
TMP = MAIN + ".tmp"

# 1) Find rows currently marked ok_empty_unrecoverable.
target_rows = set()
with open(MAIN, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        if (obj.get("llm_generation") or {}).get("status") == "ok_empty_unrecoverable":
            target_rows.add(obj["_source_row"])
print(f"Target rows: {len(target_rows)}")

# 2) Pull original raw_output for those rows from the backup.
raw_by_row = {}
with open(BAK, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        sr = obj["_source_row"]
        if sr in target_rows:
            raw = (obj.get("llm_generation") or {}).get("raw_output") or ""
            if raw:
                raw_by_row[sr] = raw
print(f"Raw outputs recovered: {len(raw_by_row)}")

# 3) Repair using the (now improved) repair_raw_output. Build new enrichments.
new_enr_by_row = {}
unrepaired = []
for sr, raw in raw_by_row.items():
    parsed = repair_raw_output(raw)
    if parsed:
        e = normalize_enrichment(parsed)
        e["provision_role_llm"] = coerce_role(e.get("provision_role_llm"))
        new_enr_by_row[sr] = e
    else:
        unrepaired.append(sr)
print(f"Newly repaired: {len(new_enr_by_row)}; still unrepaired: {len(unrepaired)} -> {unrepaired}")

# 4) Stream main and patch matching rows.
n_patched = 0
with open(MAIN, "r", encoding="utf-8") as src, open(TMP, "w", encoding="utf-8") as dst:
    for line in src:
        if not line.strip():
            continue
        obj = json.loads(line)
        sr = obj["_source_row"]
        if sr in new_enr_by_row:
            obj["llm_enrichment"] = new_enr_by_row[sr]
            obj["llm_quality"]["json_valid"] = True
            gen = dict(obj.get("llm_generation") or {})
            gen["status"] = "ok_after_manual_repair"
            gen["error"] = None
            gen["raw_output"] = None
            obj["llm_generation"] = gen
            n_patched += 1
        dst.write(json.dumps(obj, ensure_ascii=False) + "\n")

os.replace(TMP, MAIN)
print(f"Patched {n_patched} records in main.")
