"""Extract all failures with their source text and raw_output for manual fixing."""
import json

FAILURES = r"E:\swiss_citation_extraction\law_json_llm_output\law_llm_descriptors_0000000_all_failures.jsonl"
INPUT = r"E:\swiss_citation_extraction\artifacts\law_llm_input.jsonl"
OUT = r"E:\swiss_citation_extraction\failures_with_text.jsonl"

# 1) Read failures and index by _source_row
failures = {}
with open(FAILURES, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        sr = obj["_source_row"]
        failures[sr] = obj

print(f"Loaded {len(failures)} failures")

# 2) Walk the input to find their texts
texts = {}
with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        sr = obj.get("_source_row")
        if sr in failures:
            texts[sr] = {
                "text": obj.get("text"),
                "law_title": obj.get("law_title"),
                "title_section_path": obj.get("title_section_path"),
                "static_hints": obj.get("static_hints"),
            }

print(f"Found texts for {len(texts)}/{len(failures)} failures")

# 3) Write a single combined file
with open(OUT, "w", encoding="utf-8") as out:
    for sr in sorted(failures.keys()):
        rec = {
            "_source_row": sr,
            "citation": failures[sr]["citation"],
            "language": failures[sr]["language"],
            "llm_priority": failures[sr]["llm_priority"],
            "text": texts.get(sr, {}).get("text"),
            "law_title": texts.get(sr, {}).get("law_title"),
            "title_section_path": texts.get(sr, {}).get("title_section_path"),
            "raw_output": failures[sr].get("llm_generation", {}).get("raw_output"),
            "error": failures[sr].get("llm_generation", {}).get("error"),
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Wrote {OUT}")
