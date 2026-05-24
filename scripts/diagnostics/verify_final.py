"""Final verification: 100% json_valid, all records have schema-conformant llm_enrichment."""
import json
from collections import Counter

MAIN = r"E:\swiss_citation_extraction\law_json_llm_output\law_llm_descriptors_0000000_all.jsonl"

REQUIRED_KEYS = {
    "english_summary": str,
    "legal_rule": str,
    "applicability_conditions": list,
    "exceptions_or_limitations": list,
    "legal_question": str,
    "concepts_en": list,
    "terms_de_to_en": list,
    "defined_terms": list,
    "addressees": list,
    "sanctions_or_consequences": list,
    "provision_role_llm": str,
    "specificity_score": (int, float),
}

VOCAB = {
    "other", "right_or_entitlement", "duty", "definition", "prohibition",
    "transitional_or_commencement", "principle", "purpose", "scope", "procedure",
    "competence", "data_reporting", "fees_or_costs", "sanction_or_penalty",
}

n_total = 0
n_valid = 0
n_invalid = 0
schema_problems = Counter()
extra_keys = Counter()
status_dist = Counter()
role_dist = Counter()
off_vocab_roles = []

with open(MAIN, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        n_total += 1
        obj = json.loads(line)
        lq = obj.get("llm_quality") or {}
        gen = obj.get("llm_generation") or {}
        enr = obj.get("llm_enrichment") or {}
        if lq.get("json_valid") is True:
            n_valid += 1
        else:
            n_invalid += 1

        # Schema check
        for k, t in REQUIRED_KEYS.items():
            v = enr.get(k, "__MISSING__")
            if v == "__MISSING__":
                schema_problems[f"missing:{k}"] += 1
            elif not isinstance(v, t):
                schema_problems[f"wrong_type:{k}({type(v).__name__})"] += 1
        for k in enr:
            if k not in REQUIRED_KEYS:
                extra_keys[k] += 1
        # Items shape
        for item in enr.get("terms_de_to_en") or []:
            if not (isinstance(item, dict) and isinstance(item.get("de"), str) and isinstance(item.get("en"), str)):
                schema_problems["bad_term_item"] += 1
        for item in enr.get("defined_terms") or []:
            if not (isinstance(item, dict) and isinstance(item.get("term"), str) and isinstance(item.get("definition"), str)):
                schema_problems["bad_defined_item"] += 1
        # Vocab
        role = enr.get("provision_role_llm")
        role_dist[role] += 1
        if role not in VOCAB:
            off_vocab_roles.append((obj.get("_source_row"), obj.get("citation"), role))

        status_dist[gen.get("status")] += 1

print(f"Total records: {n_total}")
print(f"json_valid=True : {n_valid} ({n_valid / n_total * 100:.4f}%)")
print(f"json_valid=False: {n_invalid}")
print(f"Schema problems: {dict(schema_problems)}")
print(f"Extra (unexpected) keys: {dict(extra_keys)}")
print(f"Off-vocab roles: {len(off_vocab_roles)}")
if off_vocab_roles:
    for sr, cit, r in off_vocab_roles[:10]:
        print(f"  row {sr} ({cit}): {r}")
print(f"\nStatus distribution: {dict(status_dist)}")
print(f"Role distribution: {dict(role_dist)}")
