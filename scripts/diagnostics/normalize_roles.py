"""Normalize the few off-vocabulary provision_role_llm values produced by the
manual repair pass to the main file's controlled vocabulary."""
import json
import os

MAIN = r"E:\swiss_citation_extraction\law_json_llm_output\law_llm_descriptors_0000000_all.jsonl"
TMP = MAIN + ".tmp"

ROW_TO_ROLE = {
    108469: "duty",       # was 'rule' (4% minimum reserve requirement)
    114766: "competence", # was 'function' (functions of General Secretariat)
    149945: "scope",      # was 'exemption' (lists projects exempt from plan approval)
    152303: "definition", # was 'classification' (defines vegetable categories)
    156013: "scope",      # was 'assignment' (assigns activities to security check)
}

with open(MAIN, "r", encoding="utf-8") as src, open(TMP, "w", encoding="utf-8") as dst:
    for line in src:
        if not line.strip():
            continue
        obj = json.loads(line)
        sr = obj["_source_row"]
        if sr in ROW_TO_ROLE:
            obj["llm_enrichment"]["provision_role_llm"] = ROW_TO_ROLE[sr]
        dst.write(json.dumps(obj, ensure_ascii=False) + "\n")

os.replace(TMP, MAIN)
print(f"Normalized {len(ROW_TO_ROLE)} provision_role_llm values")
