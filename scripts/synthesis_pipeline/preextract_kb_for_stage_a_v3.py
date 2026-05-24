#!/usr/bin/env python
"""Pre-extract the slim KB needed by the synthesis pipeline.

Stage A v3 nominates ~268K unique (qid, did) candidates across 10 val queries:
~247K court paragraphs + ~21K law records. Streaming the 21.89 GB court JSONL
at pipeline runtime is wasteful — we want a small, dense lookup keyed by `did`.

Inputs:
  - research/stage_a_dossier_rerank/stage_a_v3_features.parquet
  - drive_sync/omnilex_competition/retrieval_other_artifacts/court_case_knowledge_base.patched.jsonl
  - drive_sync/omnilex_competition/retrieval_other_artifacts/corpus.parquet
  - drive_sync/omnilex_competition/retrieval_other_artifacts/laws_knowledge_base.jsonl

Outputs (written to research/stage_a_dossier_rerank/):
  - court_kb_for_stage_a_v3.parquet     (~247K rows, slim cols)
  - laws_kb_for_stage_a_v3.parquet      (~21K rows, slim cols)
  - kb_extract_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_V3 = ROOT / "research" / "stage_a_dossier_rerank" / "stage_a_v3_features.parquet"
DEFAULT_COURT_KB = (
    ROOT / "drive_sync" / "omnilex_competition" / "retrieval_other_artifacts"
    / "court_case_knowledge_base.patched.jsonl"
)
DEFAULT_LAWS_KB_PARQUET = (
    ROOT / "drive_sync" / "omnilex_competition" / "retrieval_other_artifacts"
    / "corpus.parquet"
)
DEFAULT_LAWS_KB_JSONL = (
    ROOT / "drive_sync" / "omnilex_competition" / "retrieval_other_artifacts"
    / "laws_knowledge_base.jsonl"
)
DEFAULT_OUT_DIR = ROOT / "research" / "stage_a_dossier_rerank"
KNOWN_COURT_ROWS = 2_476_315


# ─────────────────────────────────────────────────────────────────────────
# Court extraction
# ─────────────────────────────────────────────────────────────────────────

# Court record slim schema — only what the synthesis pipeline (Candidate +
# _fmt_rr + judge prompt) actually consumes.
def slim_court_record(rec: dict) -> dict:
    case = rec.get("case", {}) or {}
    semantic = rec.get("semantic", {}) or {}
    topics = rec.get("topics", {}) or {}
    refs = rec.get("references", {}) or {}
    content = rec.get("content", {}) or {}
    llm = rec.get("llm_enrichment") or {}

    text_clean = content.get("text_clean") or ""
    # Truncate text for dossier (keep 1200 chars; _fmt_rr uses 500 but judge uses 400).
    text_for_dossier = text_clean[:1200]

    return {
        "did_num": int(rec.get("metadata", {}).get("source_row_id", -1)),
        "citation_canon": rec.get("citation_canon") or "",
        "case_id_canonical": case.get("case_id_canonical") or "",
        "case_type": case.get("case_type") or "",
        "chamber": case.get("chamber") or "",
        "chamber_label": case.get("chamber_label") or "",
        "court": case.get("court") or "",
        "court_code": case.get("court_code") or "",
        "decision_year": case.get("decision_year"),
        "language_original": case.get("language_original") or "",
        "case_importance": float(case.get("case_importance") or 0.0),
        # Step 3 static fields
        "legal_area_static": rec.get("legal_area_static") or "",
        "is_leading_decision": bool(rec.get("is_leading_decision", False)),
        "has_substantive_role": bool(rec.get("has_substantive_role", False)),
        "co_citation_count_static": int(rec.get("co_citation_count_static") or 0),
        "case_peer_count": int(rec.get("case_peer_count") or 0),
        # Step 4 LLM (None if unenriched)
        "paragraph_role": semantic.get("paragraph_role") or (llm.get("paragraph_role") if llm else ""),
        "doctrinal_rule": (llm.get("doctrinal_rule") or "") if llm else "",
        "legal_test": (llm.get("legal_test") or "") if llm else "",
        "micro_topic": (llm.get("micro_topic") or "") if llm else "",
        "subtopic": (llm.get("subtopic") or "") if llm else "",
        "topic_en": topics.get("topic_en") or (llm.get("topic") if llm else "") or "",
        "legal_area_en": topics.get("legal_area_en") or (llm.get("legal_area") if llm else "") or "",
        "fact_pattern_tags": (llm.get("fact_pattern_tags") or []) if llm else [],
        "concepts_en": (llm.get("concepts_en") or []) if llm else [],
        "specificity_score": float(llm.get("specificity_score") or 0.0) if llm else 0.0,
        "references_out_count": int(refs.get("references_out_count") or 0),
        "references_in_count": int(refs.get("references_in_count") or 0),
        "text": text_for_dossier,
    }


def extract_court(court_dids: set[int], jsonl_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    target_count = len(court_dids)
    print(f"[court] streaming {jsonl_path.name} for {target_count} target dids...", file=sys.stderr)
    pbar = (tqdm(total=KNOWN_COURT_ROWS, desc="[court]", unit="rec",
                 mininterval=1.0, file=sys.stderr, smoothing=0.05)
            if HAVE_TQDM else None)
    found = 0
    t0 = time.time()
    with jsonl_path.open("r", encoding="utf-8", buffering=4 * 1024 * 1024) as f:
        for i, line in enumerate(f):
            if pbar is not None:
                pbar.update(1)
            if i not in court_dids:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(slim_court_record(rec))
            found += 1
            if pbar is not None and found % 5000 == 0:
                pbar.set_postfix(found=found, target=target_count)
            if found >= target_count:
                # All targets located — early exit
                break
    if pbar is not None:
        pbar.close()
    print(f"[court] found {found}/{target_count} in {time.time()-t0:.1f}s", file=sys.stderr)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# Law extraction
# Stage A v3 uses line-index addressing: did="law:N" == line N of
# laws_knowledge_base.jsonl (per stage_a_v3.py:67). Stream the JSONL once
# and extract only the matching line indices.
# ─────────────────────────────────────────────────────────────────────────

KNOWN_LAW_KB_LINES = 174_000  # progress hint; laws_knowledge_base.jsonl has ~370K lines


def slim_law_record(rec: dict) -> dict:
    law = rec.get("law", {}) or {}
    structure = rec.get("structure", {}) or {}
    semantic = rec.get("semantic", {}) or {}
    content = rec.get("content", {}) or {}
    topics = rec.get("topics", {}) or {}
    multilingual = rec.get("multilingual", {}) or {}

    text_clean = content.get("text_clean") or content.get("text_raw") or ""
    return {
        "did_num": -1,  # filled in by caller
        "citation_canon": rec.get("citation_canon") or "",
        "law_abbreviation": law.get("law_abbreviation") or "",
        "law_name_en": law.get("law_name_en") or multilingual.get("law_title_en") or "",
        "law_title_clean": law.get("law_title_clean") or "",
        "law_domain": law.get("law_domain") or "",
        "law_subdomain": law.get("law_subdomain") or "",
        "title": law.get("law_title_clean") or law.get("law_title_raw") or "",
        "context_heading_title": structure.get("context_heading_title") or "",
        "structure_path": structure.get("structure_path") or "",
        "provision_type": semantic.get("provision_type") or "",
        "normative_modality": semantic.get("normative_modality") or "",
        "legal_effect": semantic.get("legal_effect") or "",
        "topic_primary_de": topics.get("topic_primary_de") or "",
        "language": rec.get("metadata", {}).get("language") or "de",
        "text": text_clean[:1200],
    }


def extract_laws(
    law_dids: set[int],
    laws_kb_jsonl: Path,
) -> pd.DataFrame:
    """Stream laws_knowledge_base.jsonl using stage_a_v3.py's valid-record counter.

    Critical: did_num must match stage_a_v3.py:64-68 accounting:
        n_law = 0
        for line in f:
            try: obj = json.loads(line)
            except: continue
            if not obj.get("citation"): continue
            law_did_to_cit[f"law:{n_law}"] = cit
            n_law += 1

    Raw line index (what we used before) is WRONG — diverges whenever
    laws_knowledge_base.jsonl has unparseable lines or records with empty
    citation block.
    """
    rows: list[dict] = []
    target_count = len(law_dids)
    print(f"[laws] streaming {laws_kb_jsonl.name} for {target_count} target VALID-RECORD indices...",
          file=sys.stderr)
    pbar = (tqdm(total=KNOWN_LAW_KB_LINES, desc="[laws]", unit="rec",
                 mininterval=1.0, file=sys.stderr, smoothing=0.05)
            if HAVE_TQDM else None)
    found = 0
    skipped_parse = 0
    skipped_no_citation = 0
    n_law = 0   # MUST mirror stage_a_v3.py's counter
    t0 = time.time()
    with laws_kb_jsonl.open("r", encoding="utf-8", buffering=4 * 1024 * 1024) as f:
        for line in f:
            if pbar is not None:
                pbar.update(1)
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped_parse += 1
                continue
            # MATCH stage_a_v3.py: skip records where `citation` is falsy
            # (the `citation` *block* — a dict in this corpus).
            if not rec.get("citation"):
                skipped_no_citation += 1
                continue
            # n_law is now the stage_a_v3-style did number for THIS record.
            if n_law in law_dids:
                slim = slim_law_record(rec)
                slim["did_num"] = n_law
                rows.append(slim)
                found += 1
                if pbar is not None and found % 2000 == 0:
                    pbar.set_postfix(found=found, target=target_count)
                if found >= target_count:
                    break
            n_law += 1
    if pbar is not None:
        pbar.close()
    print(f"[laws] found {found}/{target_count} in {time.time()-t0:.1f}s "
          f"(skipped parse={skipped_parse}, no_citation={skipped_no_citation}, "
          f"n_law_total={n_law})", file=sys.stderr)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v3-features", type=Path, default=DEFAULT_V3)
    ap.add_argument("--court-jsonl", type=Path, default=DEFAULT_COURT_KB)
    ap.add_argument("--laws-parquet", type=Path, default=DEFAULT_LAWS_KB_PARQUET)
    ap.add_argument("--laws-jsonl", type=Path, default=DEFAULT_LAWS_KB_JSONL)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print(f"[v3]    {args.v3_features}", file=sys.stderr)
    df_v3 = pd.read_parquet(args.v3_features)
    print(f"[v3]    {len(df_v3)} rows, {df_v3['qid'].nunique()} qids", file=sys.stderr)

    # Split dids
    court_mask = df_v3["family_is_court"] == 1
    court_dids = set(
        df_v3.loc[court_mask, "did"].str.replace("court:", "", regex=False).astype(int).unique()
    )
    law_dids = set(
        df_v3.loc[~court_mask, "did"].str.replace("law:", "", regex=False).astype(int).unique()
    )
    print(f"[v3]    unique court dids: {len(court_dids)}", file=sys.stderr)
    print(f"[v3]    unique law dids:   {len(law_dids)}", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Court
    t_court = time.time()
    df_court = extract_court(court_dids, args.court_jsonl)
    elapsed_court = time.time() - t_court
    court_out = args.out_dir / "court_kb_for_stage_a_v3.parquet"
    df_court.to_parquet(court_out, index=False)
    print(f"[court] wrote {len(df_court)} rows -> {court_out} "
          f"({court_out.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)

    # Laws
    t_laws = time.time()
    df_laws = extract_laws(law_dids, args.laws_jsonl)
    elapsed_laws = time.time() - t_laws
    laws_out = args.out_dir / "laws_kb_for_stage_a_v3.parquet"
    df_laws.to_parquet(laws_out, index=False)
    print(f"[laws]  wrote {len(df_laws)} rows -> {laws_out} "
          f"({laws_out.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)

    # Manifest
    manifest = {
        "v3_features": str(args.v3_features),
        "court_jsonl": str(args.court_jsonl),
        "laws_parquet": str(args.laws_parquet),
        "laws_jsonl": str(args.laws_jsonl),
        "court_rows_target": len(court_dids),
        "court_rows_extracted": len(df_court),
        "laws_rows_target": len(law_dids),
        "laws_rows_extracted": len(df_laws),
        "elapsed_court_seconds": round(elapsed_court, 1),
        "elapsed_laws_seconds": round(elapsed_laws, 1),
        "outputs": {
            "court_kb_parquet": str(court_out),
            "laws_kb_parquet":  str(laws_out),
        },
    }
    (args.out_dir / "kb_extract_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[done] manifest -> {args.out_dir / 'kb_extract_manifest.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
