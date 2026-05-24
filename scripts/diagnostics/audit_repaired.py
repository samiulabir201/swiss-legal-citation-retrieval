"""Audit repaired enrichments to identify any that need manual cleanup."""
import json

REPAIRED = r"E:\swiss_citation_extraction\failures_repaired.jsonl"

bad = []
with open(REPAIRED, "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        e = rec["repaired_enrichment"]
        problems = []
        if not e.get("english_summary"):
            problems.append("no_summary")
        if not e.get("concepts_en"):
            problems.append("no_concepts")
        if not e.get("terms_de_to_en"):
            problems.append("no_terms")
        if problems:
            bad.append((rec["_source_row"], rec["citation"], problems, len(rec.get("text") or "")))

print(f"Records with empty critical fields: {len(bad)}")
for sr, cit, probs, tlen in bad:
    print(f"  row {sr}: {cit} ({probs}) text_len={tlen}")

# Print summary stats of repaired enrichments
print("\nSummary stats:")
with open(REPAIRED, "r", encoding="utf-8") as f:
    counts = {"summary": 0, "rule": 0, "applic": 0, "exc": 0, "question": 0, "concepts": 0, "terms": 0, "defs": 0, "addr": 0, "sanc": 0}
    total = 0
    for line in f:
        total += 1
        rec = json.loads(line)
        e = rec["repaired_enrichment"]
        if e.get("english_summary"): counts["summary"] += 1
        if e.get("legal_rule"): counts["rule"] += 1
        if e.get("applicability_conditions"): counts["applic"] += 1
        if e.get("exceptions_or_limitations"): counts["exc"] += 1
        if e.get("legal_question"): counts["question"] += 1
        if e.get("concepts_en"): counts["concepts"] += 1
        if e.get("terms_de_to_en"): counts["terms"] += 1
        if e.get("defined_terms"): counts["defs"] += 1
        if e.get("addressees"): counts["addr"] += 1
        if e.get("sanctions_or_consequences"): counts["sanc"] += 1
    print(f"  total: {total}")
    for k, v in counts.items():
        print(f"  {k}: {v}/{total}")
