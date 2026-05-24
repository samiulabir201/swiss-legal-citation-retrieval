"""Investigate the remaining 2052 invalid records in main file."""
import json

MAIN = r"E:\swiss_citation_extraction\law_json_llm_output\law_llm_descriptors_0000000_all.jsonl"

n_total = 0
invalid = []
status_counts = {}
has_raw = 0
has_attempts_with_raw = 0
descriptor_error_present = 0

samples_by_status = {}

with open(MAIN, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        n_total += 1
        obj = json.loads(line)
        lq = obj.get("llm_quality") or {}
        if lq.get("json_valid") is True:
            continue
        invalid.append(obj)
        gen = obj.get("llm_generation") or {}
        st = gen.get("status")
        status_counts[st] = status_counts.get(st, 0) + 1
        if gen.get("raw_output"):
            has_raw += 1
        # Check if attempts exist (with raw_output inside)
        attempts = gen.get("attempts") or []
        for a in attempts:
            if a.get("raw_output"):
                has_attempts_with_raw += 1
                break
        enr = obj.get("llm_enrichment") or {}
        if enr.get("_descriptor_error"):
            descriptor_error_present += 1
        samples_by_status.setdefault(st, []).append(obj)

print(f"Total: {n_total}")
print(f"Invalid: {len(invalid)}")
print(f"By status: {status_counts}")
print(f"Has top-level raw_output: {has_raw}")
print(f"Has attempts[*].raw_output: {has_attempts_with_raw}")
print(f"Has llm_enrichment._descriptor_error: {descriptor_error_present}")

for st, samples in samples_by_status.items():
    print(f"\n=== Status: {st} (count={len(samples)}) ===")
    s = samples[0]
    gen = s.get("llm_generation") or {}
    enr = s.get("llm_enrichment") or {}
    print(f"  citation: {s.get('citation')}")
    print(f"  source_row: {s.get('_source_row')}")
    print(f"  attempt_count: {gen.get('attempt_count')}")
    print(f"  enrichment_keys: {list(enr.keys())}")
    print(f"  raw_output present: {bool(gen.get('raw_output'))}")
    print(f"  attempts in gen: {len(gen.get('attempts') or [])}")
    print(f"  error: {(gen.get('error') or '')[:200]}")
    print(f"  descriptor_error: {(enr.get('_descriptor_error') or '')[:200]}")
