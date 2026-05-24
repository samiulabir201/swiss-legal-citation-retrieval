"""Merge the rerun enrichments back into the main file.

For each `ok_empty_engine_dead` row in main, replace it with the matching record
from the rerun output (matched by _source_row). Apply the same vocab-coercion
and schema-normalization rules as the previous fixes so the merged result stays
schema-conformant and 100% json_valid.
"""
import json
import os
import shutil
from collections import Counter

from repair_failures import normalize as normalize_enrichment
from fix_all_invalid import coerce_role, EMPTY_ENRICHMENT, VOCAB

MAIN = r"E:\swiss_citation_extraction\law_json_llm_output\law_llm_descriptors_0000000_all.jsonl"
RERUN = r"E:\swiss_citation_extraction\law_json_llm_output\law_llm_descriptors_engine_dead_rerun.jsonl"
BACKUP = MAIN + ".bak3"
TMP = MAIN + ".tmp"

REQUIRED_KEYS = list(EMPTY_ENRICHMENT.keys())


def normalize_record_enrichment(enr: dict) -> dict:
    """Trim to canonical keys, coerce types, fix role vocab."""
    keep = {k: enr.get(k, EMPTY_ENRICHMENT[k]) for k in REQUIRED_KEYS}
    norm = normalize_enrichment(keep)
    norm["provision_role_llm"] = coerce_role(norm.get("provision_role_llm"))
    return norm


def main():
    # 1) Load rerun records, indexed by _source_row.
    rerun = {}
    rerun_status_counts = Counter()
    rerun_with_summary = 0
    with open(RERUN, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            sr = int(obj["_source_row"])
            rerun[sr] = obj
            rerun_status_counts[(obj.get("llm_generation") or {}).get("status")] += 1
            if (obj.get("llm_enrichment") or {}).get("english_summary"):
                rerun_with_summary += 1
    print(f"Rerun records: {len(rerun)}")
    print(f"Rerun status distribution: {dict(rerun_status_counts)}")
    print(f"Rerun with non-empty summary: {rerun_with_summary}")

    # 2) Backup once.
    if not os.path.exists(BACKUP):
        shutil.copyfile(MAIN, BACKUP)
        print(f"Backed up to {BACKUP}")
    else:
        print(f"Backup exists at {BACKUP}; not overwriting")

    # 3) Stream main file and replace matching rows.
    n_total = 0
    n_replaced = 0
    n_eligible_rows_seen = 0
    n_eligible_without_match = 0
    rerun_rows_used = set()

    with open(MAIN, "r", encoding="utf-8") as src, open(TMP, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            n_total += 1
            obj = json.loads(line)
            sr = int(obj["_source_row"])
            cur_status = (obj.get("llm_generation") or {}).get("status")

            if cur_status == "ok_empty_engine_dead":
                n_eligible_rows_seen += 1
                rerun_rec = rerun.get(sr)
                if rerun_rec:
                    rerun_rows_used.add(sr)
                    new_enr = normalize_record_enrichment(rerun_rec.get("llm_enrichment") or {})
                    obj["llm_enrichment"] = new_enr
                    rerun_lq = rerun_rec.get("llm_quality") or {}
                    obj["llm_quality"] = {
                        "json_valid": True,
                        "terms_grounded_pct": (rerun_lq.get("terms_grounded_pct")
                                               if isinstance(rerun_lq.get("terms_grounded_pct"), (int, float))
                                               else 1.0),
                        "boilerplate_role": bool(rerun_lq.get("boilerplate_role", False)),
                    }
                    rerun_gen = rerun_rec.get("llm_generation") or {}
                    new_gen = dict(obj.get("llm_generation") or {})
                    # Carry over rerun model/method/backend metadata.
                    for k in ("model", "method", "attention_backend", "kv_cache_dtype",
                              "speculative", "attempt_count"):
                        if k in rerun_gen:
                            new_gen[k] = rerun_gen[k]
                    rerun_status = rerun_gen.get("status")
                    if rerun_status and rerun_status.startswith("ok"):
                        new_gen["status"] = "ok_after_engine_dead_rerun"
                    else:
                        new_gen["status"] = rerun_status or "failed_descriptor_parse"
                    new_gen["error"] = rerun_gen.get("error")
                    new_gen["raw_output"] = None  # stay compact in main file
                    new_gen.pop("attempts", None)
                    obj["llm_generation"] = new_gen
                    n_replaced += 1
                else:
                    n_eligible_without_match += 1
            else:
                # Defensive: re-normalize enrichment shape anyway.
                obj["llm_enrichment"] = normalize_record_enrichment(obj.get("llm_enrichment") or {})

            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")

    os.replace(TMP, MAIN)

    # 4) Report.
    print(f"\nMain records scanned: {n_total}")
    print(f"Eligible (engine_dead) rows: {n_eligible_rows_seen}")
    print(f"Replaced: {n_replaced}")
    print(f"Eligible without rerun match: {n_eligible_without_match}")
    rerun_rows_unused = set(rerun.keys()) - rerun_rows_used
    print(f"Rerun records that didn't match an eligible row: {len(rerun_rows_unused)}")
    if rerun_rows_unused:
        print(f"  first few: {sorted(rerun_rows_unused)[:10]}")


if __name__ == "__main__":
    main()
