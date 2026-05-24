"""Build a JSONL containing only the records that need to be re-run by the
LLM agent. These are the rows where the previous run hit `EngineDeadError`
and were normalized to `ok_empty_engine_dead` status (empty enrichment).

The output file is structurally identical to law_llm_input.jsonl so the
existing notebook can ingest it without changes once the input path is
updated.
"""
import json

MAIN = r"E:\swiss_citation_extraction\law_json_llm_output\law_llm_descriptors_0000000_all.jsonl"
INPUT = r"E:\swiss_citation_extraction\artifacts\law_llm_input.jsonl"
OUT = r"E:\swiss_citation_extraction\artifacts\law_llm_input_engine_dead_rerun.jsonl"

# 1) Collect _source_row values that need rerun.
target_rows = set()
with open(MAIN, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        if (obj.get("llm_generation") or {}).get("status") == "ok_empty_engine_dead":
            target_rows.add(int(obj["_source_row"]))

print(f"Records needing rerun: {len(target_rows)}")

# 2) Walk the original input and copy matching rows verbatim.
n_written = 0
n_total_in_input = 0
with open(INPUT, "r", encoding="utf-8") as src, open(OUT, "w", encoding="utf-8") as dst:
    for line in src:
        if not line.strip():
            continue
        n_total_in_input += 1
        obj = json.loads(line)
        sr = obj.get("_source_row")
        if sr is None:
            continue
        if int(sr) in target_rows:
            dst.write(line if line.endswith("\n") else line + "\n")
            n_written += 1

print(f"Scanned input rows: {n_total_in_input:,}")
print(f"Wrote {n_written} records to {OUT}")
missing = target_rows - {int(json.loads(l)['_source_row']) for l in open(OUT, encoding='utf-8') if l.strip()}
if missing:
    print(f"WARNING: {len(missing)} target rows not found in input. First few: {sorted(missing)[:10]}")
