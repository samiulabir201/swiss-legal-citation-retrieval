"""Find which manually-repaired entries used non-vocabulary provision_role_llm values."""
import json

MAIN = r"E:\swiss_citation_extraction\law_json_llm_output\law_llm_descriptors_0000000_all.jsonl"

VOCAB = {
    "other", "right_or_entitlement", "duty", "definition", "prohibition",
    "transitional_or_commencement", "principle", "purpose", "scope", "procedure",
    "competence", "data_reporting", "fees_or_costs", "sanction_or_penalty",
}

with open(MAIN, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        if (obj.get("llm_generation") or {}).get("status") != "ok_after_manual_repair":
            continue
        role = (obj.get("llm_enrichment") or {}).get("provision_role_llm")
        if role not in VOCAB:
            print(f"  row {obj['_source_row']} ({obj['citation']}): role={role!r}")
