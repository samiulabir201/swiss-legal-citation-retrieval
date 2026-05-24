"""Investigate quality of law_llm_descriptors_0000000_all.jsonl"""
import json
from collections import Counter

MAIN = r"E:\swiss_citation_extraction\law_json_llm_output\law_llm_descriptors_0000000_all.jsonl"

stats = {
    "total": 0,
    "json_valid_true": 0,
    "json_valid_false": 0,
    "status_ok": 0,
    "status_other": Counter(),
    "boilerplate_role": 0,
    "empty_summary": 0,
    "missing_summary": 0,
    "empty_concepts": 0,
    "empty_terms": 0,
    "attempt_count": Counter(),
    "provision_role_llm": Counter(),
    "languages": Counter(),
    "priority": Counter(),
    "specificity_dist": Counter(),  # bucketed
    "terms_grounded_dist": Counter(),  # bucketed
    "duplicate_source_rows": 0,
}

seen_rows = set()
duplicates = []
zero_summary_rows = []

with open(MAIN, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        stats["total"] += 1
        try:
            obj = json.loads(line)
        except Exception:
            stats["status_other"]["parse_error"] += 1
            continue

        sr = obj.get("_source_row")
        if sr in seen_rows:
            stats["duplicate_source_rows"] += 1
            duplicates.append(sr)
        else:
            seen_rows.add(sr)

        lq = obj.get("llm_quality") or {}
        if lq.get("json_valid") is True:
            stats["json_valid_true"] += 1
        else:
            stats["json_valid_false"] += 1

        if lq.get("boilerplate_role"):
            stats["boilerplate_role"] += 1

        gen = obj.get("llm_generation") or {}
        status = gen.get("status")
        if status == "ok":
            stats["status_ok"] += 1
        else:
            stats["status_other"][status] += 1

        ac = gen.get("attempt_count")
        stats["attempt_count"][ac] += 1

        enr = obj.get("llm_enrichment") or {}
        summary = enr.get("english_summary")
        if summary is None:
            stats["missing_summary"] += 1
        elif summary == "":
            stats["empty_summary"] += 1
            zero_summary_rows.append(sr)
        if not enr.get("concepts_en"):
            stats["empty_concepts"] += 1
        if not enr.get("terms_de_to_en"):
            stats["empty_terms"] += 1
        stats["provision_role_llm"][enr.get("provision_role_llm")] += 1
        sp = enr.get("specificity_score")
        try:
            sp = float(sp)
            bucket = round(sp, 1)
            stats["specificity_dist"][bucket] += 1
        except Exception:
            stats["specificity_dist"]["non_numeric"] += 1

        tg = lq.get("terms_grounded_pct")
        try:
            tg = float(tg)
            bucket = round(tg, 1)
            stats["terms_grounded_dist"][bucket] += 1
        except Exception:
            stats["terms_grounded_dist"]["non_numeric"] += 1

        stats["languages"][obj.get("language")] += 1
        stats["priority"][obj.get("llm_priority")] += 1

print(f"Total records: {stats['total']}")
print(f"json_valid: True={stats['json_valid_true']} ({stats['json_valid_true']/stats['total']*100:.2f}%), False={stats['json_valid_false']}")
print(f"status ok: {stats['status_ok']} ({stats['status_ok']/stats['total']*100:.2f}%)")
print(f"status other: {dict(stats['status_other'])}")
print(f"boilerplate_role: {stats['boilerplate_role']}")
print(f"empty english_summary: {stats['empty_summary']}")
print(f"missing english_summary: {stats['missing_summary']}")
print(f"empty concepts_en: {stats['empty_concepts']} ({stats['empty_concepts']/stats['total']*100:.2f}%)")
print(f"empty terms_de_to_en: {stats['empty_terms']} ({stats['empty_terms']/stats['total']*100:.2f}%)")
print(f"duplicate_source_rows: {stats['duplicate_source_rows']}")
print(f"attempt_count distribution: {dict(stats['attempt_count'])}")
print(f"provision_role_llm distribution: {dict(stats['provision_role_llm'])}")
print(f"languages: {dict(stats['languages'])}")
print(f"priority: {dict(stats['priority'])}")
print(f"specificity_score buckets: {dict(sorted(stats['specificity_dist'].items(), key=lambda x: (str(type(x[0])), x[0])))}")
print(f"terms_grounded_pct buckets: {dict(sorted(stats['terms_grounded_dist'].items(), key=lambda x: (str(type(x[0])), x[0])))}")

# show sample _source_rows that are missing/empty to investigate
print(f"\nFirst 5 zero_summary _source_rows: {zero_summary_rows[:5]}")
