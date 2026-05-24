"""Print compact spot-check of repaired enrichments for human review."""
import json

REPAIRED = r"E:\swiss_citation_extraction\failures_repaired.jsonl"

with open(REPAIRED, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        rec = json.loads(line)
        e = rec["repaired_enrichment"]
        sr = rec["_source_row"]
        print(f"=== row {sr}: {rec['citation']} ===")
        print(f"  summary: {e['english_summary'][:200]}")
        print(f"  rule: {e['legal_rule'][:120]}")
        print(f"  applic ({len(e['applicability_conditions'])}): {e['applicability_conditions'][:3]}")
        print(f"  exc ({len(e['exceptions_or_limitations'])}): {e['exceptions_or_limitations'][:2]}")
        print(f"  concepts ({len(e['concepts_en'])}): {e['concepts_en'][:5]}")
        print(f"  terms ({len(e['terms_de_to_en'])}): {e['terms_de_to_en'][:3]}")
        print(f"  role: {e['provision_role_llm']}, score: {e['specificity_score']}")
        print()
        if i >= 5:
            break
