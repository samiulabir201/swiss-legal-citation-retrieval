"""Classify the 2052 invalid records by what data is recoverable from the saved fields."""
import json
import re

MAIN = r"E:\swiss_citation_extraction\law_json_llm_output\law_llm_descriptors_0000000_all.jsonl"

categories = {
    "has_raw_output": 0,
    "valueerror_with_partial": 0,
    "syntaxerror_only_line": 0,
    "other": 0,
}
samples = {k: [] for k in categories}

with open(MAIN, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)
        lq = obj.get("llm_quality") or {}
        if lq.get("json_valid") is True:
            continue
        gen = obj.get("llm_generation") or {}
        if gen.get("raw_output"):
            categories["has_raw_output"] += 1
            if len(samples["has_raw_output"]) < 3:
                samples["has_raw_output"].append(obj)
            continue
        err = gen.get("error") or ""
        if err.startswith("ValueError("):
            # Extract partial JSON from "no balanced JSON object found: ..."
            m = re.search(r"ValueError\('no balanced JSON object found: (.*)'\)$", err, re.DOTALL)
            if m:
                categories["valueerror_with_partial"] += 1
                if len(samples["valueerror_with_partial"]) < 3:
                    samples["valueerror_with_partial"].append(obj)
                continue
        if err.startswith("SyntaxError("):
            categories["syntaxerror_only_line"] += 1
            if len(samples["syntaxerror_only_line"]) < 3:
                samples["syntaxerror_only_line"].append(obj)
            continue
        categories["other"] += 1
        if len(samples["other"]) < 3:
            samples["other"].append(obj)

print("Categories:")
for k, v in categories.items():
    print(f"  {k}: {v}")

# Display each sample
for k, recs in samples.items():
    print(f"\n=== {k} samples ===")
    for r in recs:
        gen = r.get("llm_generation") or {}
        print(f"  row {r['_source_row']}: {r['citation']}")
        err = gen.get("error") or ""
        print(f"    error[:300]: {err[:300]}")
        if gen.get("raw_output"):
            print(f"    raw_output[:300]: {gen.get('raw_output')[:300]}")
