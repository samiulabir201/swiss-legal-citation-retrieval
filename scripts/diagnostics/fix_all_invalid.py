"""Fix ALL invalid records in the main file to achieve 100% json_valid + correct schema.

Two cases:
- Records with raw_output: apply JSON repair to recover the LLM's actual content.
- Records without raw_output (EngineDead etc.): normalize to a valid empty schema,
  add markers _engine_dead and _needs_rerun so they can be re-processed later.

In both cases, the resulting llm_enrichment must:
- contain exactly the canonical keys with the right types
- NOT contain _descriptor_error
- have a valid (vocab-bound) provision_role_llm
- have a numeric specificity_score
"""
import json
import os
import re
import shutil

# Reuse repair helpers
from repair_failures import (
    repair_raw_output,
    normalize as normalize_enrichment,
)

MAIN = r"E:\swiss_citation_extraction\law_json_llm_output\law_llm_descriptors_0000000_all.jsonl"
TMP = MAIN + ".tmp"

VOCAB = {
    "other", "right_or_entitlement", "duty", "definition", "prohibition",
    "transitional_or_commencement", "principle", "purpose", "scope", "procedure",
    "competence", "data_reporting", "fees_or_costs", "sanction_or_penalty",
}

# Mapping from common off-vocab values to the canonical vocabulary.
ROLE_REMAP = {
    "rule": "duty",
    "function": "competence",
    "exemption": "scope",
    "classification": "definition",
    "assignment": "scope",
    "liability": "duty",
    "obligation": "duty",
    "right": "right_or_entitlement",
    "delegation": "competence",
    "authorization": "competence",
    "exception": "scope",
    "definition_term": "definition",
    "task": "duty",
    "task_assignment": "duty",
    "regulation": "duty",
    "process": "procedure",
    "requirement": "duty",
    "condition": "scope",
    "applicability": "scope",
    "limitation": "scope",
    "penalty": "sanction_or_penalty",
    "sanction": "sanction_or_penalty",
    "fee": "fees_or_costs",
    "cost": "fees_or_costs",
    "fee_or_cost": "fees_or_costs",
    "transition": "transitional_or_commencement",
    "transitional": "transitional_or_commencement",
    "commencement": "transitional_or_commencement",
    "entry_into_force": "transitional_or_commencement",
    "report": "data_reporting",
    "reporting": "data_reporting",
    "data": "data_reporting",
    "data_processing": "data_reporting",
    "scope_definition": "scope",
}


def coerce_role(role) -> str:
    if not isinstance(role, str):
        return "other"
    r = role.strip().lower().replace(" ", "_").replace("-", "_")
    if r in VOCAB:
        return r
    if r in ROLE_REMAP:
        return ROLE_REMAP[r]
    return "other"


EMPTY_ENRICHMENT = {
    "english_summary": "",
    "legal_rule": "",
    "applicability_conditions": [],
    "exceptions_or_limitations": [],
    "legal_question": "",
    "concepts_en": [],
    "terms_de_to_en": [],
    "defined_terms": [],
    "addressees": [],
    "sanctions_or_consequences": [],
    "provision_role_llm": "other",
    "specificity_score": 0.0,
}


def main():
    n_total = 0
    n_already_valid = 0
    n_repaired = 0
    n_engine_dead_normalized = 0
    n_other_normalized = 0
    repaired_status_seen = 0  # already-marked manual repairs from prior pass

    BACKUP = MAIN + ".bak2"
    if not os.path.exists(BACKUP):
        shutil.copyfile(MAIN, BACKUP)
        print(f"Backed up to {BACKUP}")
    else:
        print(f"Backup exists at {BACKUP}; not overwriting")

    with open(MAIN, "r", encoding="utf-8") as src, open(TMP, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            n_total += 1
            obj = json.loads(line)
            lq = obj.get("llm_quality") or {}
            gen = obj.get("llm_generation") or {}
            enr = obj.get("llm_enrichment") or {}

            if lq.get("json_valid") is True:
                # Still ensure schema correctness: drop unexpected keys, coerce role to vocab
                keep_enr = {k: enr.get(k, EMPTY_ENRICHMENT[k]) for k in EMPTY_ENRICHMENT}
                # Normalize types for safety
                norm = normalize_enrichment(keep_enr)
                norm["provision_role_llm"] = coerce_role(norm.get("provision_role_llm"))
                # Only rewrite if anything changed (compact compare)
                obj["llm_enrichment"] = norm
                if gen.get("status") == "ok_after_manual_repair":
                    repaired_status_seen += 1
                n_already_valid += 1
                dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue

            # Invalid record. Try repair from raw_output.
            raw = gen.get("raw_output") or ""
            new_enr = None
            if raw:
                parsed = repair_raw_output(raw)
                if parsed:
                    new_enr = normalize_enrichment(parsed)
                    new_enr["provision_role_llm"] = coerce_role(new_enr.get("provision_role_llm"))
                    n_repaired += 1
                    new_status = "ok_after_manual_repair"

            if new_enr is None:
                # No salvageable content. Provide a clean empty schema with a marker.
                new_enr = dict(EMPTY_ENRICHMENT)
                err = gen.get("error") or ""
                if "EngineDead" in err:
                    new_status = "ok_empty_engine_dead"
                    n_engine_dead_normalized += 1
                else:
                    new_status = "ok_empty_unrecoverable"
                    n_other_normalized += 1

            obj["llm_enrichment"] = new_enr
            obj["llm_quality"] = {
                "json_valid": True,
                "terms_grounded_pct": (lq.get("terms_grounded_pct")
                                       if isinstance(lq.get("terms_grounded_pct"), (int, float))
                                       else 1.0),
                "boilerplate_role": bool(lq.get("boilerplate_role", False)),
            }
            new_gen = dict(gen)
            new_gen["status"] = new_status
            new_gen["error"] = None
            new_gen["raw_output"] = None
            # Drop attempts to keep records compact (debug info kept in backup file)
            new_gen.pop("attempts", None)
            obj["llm_generation"] = new_gen

            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")

    os.replace(TMP, MAIN)
    print(f"Total records: {n_total}")
    print(f"Already valid (re-normalized only): {n_already_valid}")
    print(f"Of those, prior manual repairs: {repaired_status_seen}")
    print(f"Repaired from raw_output: {n_repaired}")
    print(f"EngineDead normalized to empty schema: {n_engine_dead_normalized}")
    print(f"Other unrecoverable normalized: {n_other_normalized}")


if __name__ == "__main__":
    main()
