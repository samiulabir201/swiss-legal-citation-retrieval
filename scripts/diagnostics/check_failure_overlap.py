"""Check if failures are duplicated in the main file."""
import json

MAIN = r"E:\swiss_citation_extraction\law_json_llm_output\law_llm_descriptors_0000000_all.jsonl"
FAILURES = r"E:\swiss_citation_extraction\law_json_llm_output\law_llm_descriptors_0000000_all_failures.jsonl"

failure_rows = set()
failure_records = []
with open(FAILURES, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        sr = obj["_source_row"]
        failure_rows.add(sr)
        failure_records.append({"_source_row": sr, "citation": obj.get("citation"), "language": obj.get("language"), "priority": obj.get("llm_priority")})

print(f"Total failures: {len(failure_rows)}")
print(f"First 10 failure source_rows: {sorted(failure_rows)[:10]}")
print(f"Last 5 failure source_rows: {sorted(failure_rows)[-5:]}")

# Check if these source rows exist in the main file
main_rows = set()
main_failed_rows = set()  # those in main file with json_valid=false
with open(MAIN, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        sr = obj["_source_row"]
        main_rows.add(sr)
        if not (obj.get("llm_quality") or {}).get("json_valid"):
            main_failed_rows.add(sr)

overlap = failure_rows & main_rows
not_in_main = failure_rows - main_rows
print(f"\nFailures already in main: {len(overlap)}")
print(f"Failures NOT in main: {len(not_in_main)}")
print(f"First 5 not in main: {sorted(not_in_main)[:5]}")

# Total in main with json_valid=false
print(f"\nMain file has {len(main_failed_rows)} records with json_valid=False")
print(f"Of those, {len(main_failed_rows & failure_rows)} are also in failure file")

# Print failure citations to understand scope
print("\nAll failure records (first 20):")
for rec in sorted(failure_records, key=lambda x: x["_source_row"])[:20]:
    print(f"  row {rec['_source_row']}: {rec['citation']} ({rec['priority']})")
