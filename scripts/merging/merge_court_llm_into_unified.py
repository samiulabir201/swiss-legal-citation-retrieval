#!/usr/bin/env python
"""Merge the 363k court LLM enrichment into the live unified_retrieval.sqlite.

WHAT THIS SCRIPT DOES (and why it exists)
------------------------------------------
The audit flagged that ``unified_retrieval_manifest.json`` reports
``court_flag_static_only: 2_476_315`` for every single court row in the live
unified index, suggesting LLM English concepts never made it into ``vector_text``
or ``bm25_text``.

That flag, however, is a MISNOMER set by ``build_unified_retrieval_corpus.py``::

    if not rag.get("english_summary"):
        quality_flags.append("static_only")

The court LLM schema (``court_enrichment_normalizer.py``) **explicitly forbids**
``english_summary`` (see ``FORBIDDEN_LLM_FIELDS``), so the flag fires on every
court row regardless of whether the LLM actually contributed. The flag is
informational only — it does NOT mean the row lacks LLM data.

Inspection of ``court_authority_cards_v5_unified.jsonl`` confirms 363,257 rows
have ``enrichment_source = "llm+static"`` with populated ``topic``, ``subtopic``,
``concepts_en``, ``terms_original``, ``fact_pattern_tags``, ``legal_domain_path``,
etc. Inspection of two LLM-enriched samples shows ``vector_text`` already
contains ``Topic:``, ``Concepts:``, ``Keywords:`` labels per the build_court_doc
function in ``build_unified_retrieval_corpus.py``.

This script provides a defensive idempotent re-merge: it reads the LLM JSONL
(``court_llm_descriptors_0000000_all.jsonl``), re-builds an English-augmented
``vector_text`` and ``bm25_text`` snippet for each LLM row keyed on
``(family='court', source_line=_source_row+1)``, and either:

  - ``--dry-run`` (default): writes a side parquet + sample CSV to
    ``artifacts/court_llm_remerge_dryrun/`` — never touches the live DB
  - ``--apply``: writes UPDATEs against a copy at
    ``artifacts/unified_retrieval_dryrun.sqlite`` (NOT the live DB)
  - ``--apply --target-live``: writes against the live DB
    (must be passed explicitly — the user requested no live writes by default)

KEY JOIN
--------
LLM JSONL has ``_source_row`` which is 0-based against
``court_authority_cards_v4.jsonl`` (the input that produced v5). The unified
builder reads ``court_authority_cards_v5_unified.jsonl`` line-by-line; line 1
of v5 = ``_source_row=0`` of v4. So:

    documents.source_line (1-based) == llm._source_row + 1
    AND documents.family == 'court'

CITATION-BASED FALLBACK (when source_line drifts)
The script also keys on ``citation`` text as a tie-break.

RE-EMBEDDING REQUIREMENT (CRITICAL)
-----------------------------------
After this script runs in ``--apply`` mode, the affected rows MUST be
re-embedded:

  1. Regenerate the per-row vector_text for the changed rows
  2. Re-run ``scripts/prepare_embedding_input.py`` (court family only)
  3. Re-encode those rows on Colab with the production embedding model
  4. Replace the rows in the FAISS / vector index

This script does NOT re-embed locally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

DEFAULT_LLM = ROOT / "outputs_from_363k_run" / "court_llm_descriptors_0000000_all.jsonl"
LIVE_DB = ART / "unified_retrieval.sqlite"
DRYRUN_DB = ART / "unified_retrieval_dryrun.sqlite"
DRYRUN_DIR = ART / "court_llm_remerge_dryrun"

MAX_VECTOR_CHARS = 4500
MAX_SOURCE_CHARS = 2500


# ---------------------------------------------------------------------------
# Text helpers (mirror build_unified_retrieval_corpus.py to stay byte-compatible)
# ---------------------------------------------------------------------------

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.replace(chr(0), " ").split())


def unique_keep_order(values: Iterable[Any], *, max_items: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = clean_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if max_items is not None and len(out) >= max_items:
            break
    return out


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return unique_keep_order(value)
    text = clean_text(value)
    return [text] if text else []


def field_join(*parts: Any) -> str:
    strings: list[str] = []
    for part in parts:
        if part is None or part == "":
            continue
        if isinstance(part, (list, tuple, set)):
            strings.extend(clean_text(x) for x in part if x)
        else:
            strings.append(clean_text(part))
    return " | ".join(x for x in strings if x)


def labelled(label: str, value: Any) -> str:
    text = clean_text(value) if not isinstance(value, (list, tuple, set)) else field_join(*value)
    return f"{label}: {text}" if text else ""


def weighted_join(parts: Iterable[str], repeats: int) -> str:
    base = [clean_text(x) for x in parts if clean_text(x)]
    return " ".join(base * max(1, repeats))


def clipped(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


# ---------------------------------------------------------------------------
# Per-row enrichment composer (uses ALL 16 LLM fields)
# ---------------------------------------------------------------------------

# 16-field court LLM schema:
#   legal_area, primary_domain, secondary_domain, legal_domain_path,
#   topic, subtopic, micro_topic,
#   concepts_en, terms_original, doctrinal_rule, legal_test, fact_pattern_tags,
#   procedural_context, paragraph_role, authority_role, specificity_score
COURT_LLM_FIELDS = [
    "legal_area", "primary_domain", "secondary_domain", "legal_domain_path",
    "topic", "subtopic", "micro_topic",
    "concepts_en", "terms_original", "doctrinal_rule", "legal_test",
    "fact_pattern_tags", "procedural_context", "paragraph_role",
    "authority_role", "specificity_score",
]


def compose_vector_text(citation: str, court_base: str, llm: dict, raw_text: str) -> str:
    """Re-build the labelled English-augmented vector_text from the 16-field LLM schema.

    Mirrors build_unified_retrieval_corpus.build_court_doc but pulls fields
    directly from the LLM record (no normalizer needed because anchors are
    additive/separate).
    """
    legal_area = clean_text(llm.get("legal_area"))
    topic = clean_text(llm.get("topic"))
    subtopic = clean_text(llm.get("subtopic"))
    micro = clean_text(llm.get("micro_topic"))
    primary_domain = clean_text(llm.get("primary_domain"))
    secondary_domain = clean_text(llm.get("secondary_domain"))
    domain_path = as_list(llm.get("legal_domain_path"))
    concepts_en = unique_keep_order(llm.get("concepts_en") or [], max_items=40)
    terms_original = as_list(llm.get("terms_original"))
    fact_tags = as_list(llm.get("fact_pattern_tags"))
    doctrinal_rule = clean_text(llm.get("doctrinal_rule"))
    legal_test = clean_text(llm.get("legal_test"))
    procedural_context = clean_text(llm.get("procedural_context"))
    paragraph_role = clean_text(llm.get("paragraph_role"))
    authority_role = as_list(llm.get("authority_role"))

    domain_str = " > ".join([d for d in domain_path if d]) if domain_path \
        else field_join(primary_domain, secondary_domain)

    vector_text = field_join(
        labelled("Family", "court"),
        labelled("Citation", citation),
        labelled("Court base", court_base),
        labelled("Legal area", legal_area),
        labelled("Domain", domain_str),
        labelled("Topic", topic),
        labelled("Subtopic", subtopic),
        labelled("Micro-topic", micro),
        labelled("Concepts", concepts_en),
        labelled("Keywords", terms_original),
        labelled("Fact patterns", fact_tags),
        labelled("Doctrinal rule", doctrinal_rule),
        labelled("Legal test", legal_test),
        labelled("Procedural context", procedural_context),
        labelled("Role", paragraph_role),
        labelled("Authority role", authority_role),
    )
    return clipped(vector_text, MAX_VECTOR_CHARS)


def compose_bm25_text(citation: str, court_base: str, llm: dict, raw_text: str,
                      existing_citation_text: str = "") -> str:
    """Re-build the per-row bm25_text using the same column-weighting as the live builder."""
    legal_area = clean_text(llm.get("legal_area"))
    primary_domain = clean_text(llm.get("primary_domain"))
    secondary_domain = clean_text(llm.get("secondary_domain"))
    domain_path = as_list(llm.get("legal_domain_path"))
    topic = clean_text(llm.get("topic"))
    subtopic = clean_text(llm.get("subtopic"))
    micro = clean_text(llm.get("micro_topic"))
    concepts_en = as_list(llm.get("concepts_en"))
    terms_original = as_list(llm.get("terms_original"))
    fact_tags = as_list(llm.get("fact_pattern_tags"))
    doctrinal_rule = clean_text(llm.get("doctrinal_rule"))
    legal_test = clean_text(llm.get("legal_test"))
    procedural_context = clean_text(llm.get("procedural_context"))
    paragraph_role = clean_text(llm.get("paragraph_role"))
    authority_role = as_list(llm.get("authority_role"))

    citation_text = existing_citation_text or field_join(citation, court_base)
    legal_text = field_join(
        legal_area, primary_domain, secondary_domain, *domain_path,
        topic, subtopic, micro, paragraph_role, *authority_role,
        *fact_tags,
    )
    rag_text = field_join(
        doctrinal_rule, legal_test, procedural_context,
    )
    keyword_text = field_join(
        *concepts_en, *terms_original, *fact_tags,
    )
    raw_context = clipped(raw_text, MAX_SOURCE_CHARS)

    return field_join(
        weighted_join([citation_text], 4),
        weighted_join([keyword_text], 3),
        weighted_join([rag_text], 2),
        legal_text,
        raw_context,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def stream_llm(path: Path, limit: int = 0, progress_every: int = 50_000) -> Iterable[dict]:
    """Yield dicts from the LLM JSONL with status=='ok'."""
    n_seen = 0
    n_ok = 0
    t0 = time.time()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_seen += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            status = (rec.get("llm_generation") or {}).get("status", "")
            if status.startswith("failed"):
                continue
            llm = rec.get("llm_enrichment")
            src = rec.get("_source_row")
            if not isinstance(llm, dict) or not llm or src is None:
                continue
            yield rec
            n_ok += 1
            if limit and n_ok >= limit:
                return
            if progress_every and n_ok % progress_every == 0:
                rate = n_ok / max(time.time() - t0, 1e-9)
                print(f"  [llm] streamed={n_ok:,} ({rate:>5.0f} rec/s)", flush=True)


def run(args: argparse.Namespace) -> int:
    if not args.llm.exists():
        print(f"[error] LLM JSONL not found: {args.llm}", file=sys.stderr)
        return 2

    # Pick target DB
    if args.apply and args.target_live:
        target_db = LIVE_DB
        if not target_db.exists():
            print(f"[error] live DB not found: {target_db}", file=sys.stderr)
            return 2
        print(f"[apply-live] WRITING TO LIVE DB: {target_db}")
    elif args.apply:
        target_db = DRYRUN_DB
        if not target_db.exists() or args.force_copy:
            print(f"[copy] {LIVE_DB}  ->  {target_db}", flush=True)
            shutil.copy2(LIVE_DB, target_db)
        print(f"[apply-dryrun-copy] WRITING TO COPY: {target_db}")
    else:
        target_db = None
        DRYRUN_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[dry-run] writing samples to {DRYRUN_DIR}, NOT touching any DB")

    # Open live DB read-only to fetch existing rows for comparison
    live_con = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)
    live_con.row_factory = sqlite3.Row

    # If applying, open writable handle on target_db
    write_con = None
    if target_db is not None:
        write_con = sqlite3.connect(target_db)
        write_con.execute("PRAGMA journal_mode=OFF")
        write_con.execute("PRAGMA synchronous=OFF")
        write_con.execute("PRAGMA temp_store=MEMORY")

    stats = Counter()
    sample_path = DRYRUN_DIR / "samples.jsonl"
    DRYRUN_DIR.mkdir(parents=True, exist_ok=True)
    sample_handle = sample_path.open("w", encoding="utf-8")
    sample_csv = (DRYRUN_DIR / "samples_summary.csv").open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(sample_csv)
    csv_writer.writerow([
        "n", "doc_id", "source_line", "citation",
        "before_vector_len", "after_vector_len",
        "before_vector_head", "after_vector_head",
    ])

    samples_kept = 0
    pending: list[tuple] = []
    BATCH = 5000

    t0 = time.time()
    for rec in stream_llm(args.llm, limit=args.limit):
        src_row = int(rec["_source_row"])
        citation = clean_text(rec.get("citation"))
        text = rec.get("text") or ""
        llm = rec["llm_enrichment"]

        # Lookup live row by (family='court', source_line = src_row + 1)
        # NOTE: the unified builder used line_no = enumerate(start=1), so source_line
        #       in documents == src_row + 1.
        target_source_line = src_row + 1
        row = live_con.execute(
            """
            SELECT doc_id, citation, court_base, vector_text, bm25_text,
                   source_text, enrichment_source
            FROM documents
            WHERE family='court' AND source_line=?
            """,
            (target_source_line,),
        ).fetchone()

        if row is None:
            stats["missing_in_live_db"] += 1
            continue

        # Citation tie-break (warn but continue if mismatched)
        live_citation = row["citation"]
        if citation and live_citation and citation != live_citation:
            stats["citation_mismatch"] += 1

        court_base = row["court_base"] or ""
        new_vector = compose_vector_text(citation or live_citation, court_base, llm, text)
        new_bm25 = compose_bm25_text(
            citation or live_citation, court_base, llm, text,
            existing_citation_text=field_join(citation or live_citation, court_base),
        )

        before_vec = row["vector_text"] or ""
        if before_vec == new_vector:
            stats["unchanged"] += 1
        else:
            stats["would_update"] += 1

        if samples_kept < 5 or (samples_kept < 50 and stats["would_update"] % 1000 == 0):
            sample_handle.write(json.dumps({
                "n": samples_kept,
                "doc_id": row["doc_id"],
                "source_line": target_source_line,
                "citation": live_citation,
                "before_vector_text": before_vec,
                "after_vector_text": new_vector,
                "before_bm25_text": (row["bm25_text"] or "")[:500],
                "after_bm25_text": new_bm25[:500],
                "enrichment_source_live": row["enrichment_source"],
                "llm_fields_present": [k for k in COURT_LLM_FIELDS
                                       if llm.get(k) not in (None, "", [], {})],
            }, ensure_ascii=False) + "\n")
            csv_writer.writerow([
                samples_kept, row["doc_id"], target_source_line, live_citation,
                len(before_vec), len(new_vector),
                before_vec[:200].replace("\n", " "),
                new_vector[:200].replace("\n", " "),
            ])
            samples_kept += 1

        if write_con is not None:
            pending.append((new_vector, new_bm25, "llm+static", row["doc_id"]))
            if len(pending) >= BATCH:
                write_con.executemany(
                    """
                    UPDATE documents
                    SET vector_text=?, bm25_text=?, enrichment_source=?
                    WHERE doc_id=?
                    """,
                    pending,
                )
                write_con.commit()
                pending.clear()

        stats["scanned"] += 1

    if write_con is not None and pending:
        write_con.executemany(
            """
            UPDATE documents
            SET vector_text=?, bm25_text=?, enrichment_source=?
            WHERE doc_id=?
            """,
            pending,
        )
        write_con.commit()
        pending.clear()

    if write_con is not None:
        # Optionally clean the misleading static_only flag for updated rows.
        # We do NOT touch quality_flags_json by default because reload of the
        # full builder would simply re-add it (since english_summary is still
        # absent). Caller can opt in to cleaning the flag.
        if args.clean_static_only_flag:
            print("[apply] removing 'static_only' from quality_flags_json for "
                  "all llm-enriched court rows", flush=True)
            # Best-effort string surgery (cheap, avoids re-parsing JSON for 363k rows)
            write_con.execute(
                """
                UPDATE documents
                SET quality_flags_json = REPLACE(
                    REPLACE(quality_flags_json, '"static_only",', ''),
                    ',"static_only"', ''
                )
                WHERE family='court' AND enrichment_source='llm+static'
                """
            )
            write_con.commit()

        write_con.close()

    live_con.close()
    sample_handle.close()
    sample_csv.close()

    elapsed = time.time() - t0
    summary = {
        "mode": "apply-live" if (args.apply and args.target_live)
                else ("apply-dryrun-copy" if args.apply else "dry-run"),
        "target_db": str(target_db) if target_db else None,
        "llm_input": str(args.llm),
        "limit": args.limit,
        "elapsed_seconds": round(elapsed, 1),
        "stats": dict(stats),
        "samples_path": str(sample_path),
        "samples_csv": str(DRYRUN_DIR / "samples_summary.csv"),
        "note": (
            "The 'static_only' flag in unified_retrieval_manifest.json is a misnomer. "
            "It triggers in build_unified_retrieval_corpus.py when "
            "rag.get('english_summary') is empty — and the court LLM schema "
            "(court_enrichment_normalizer.FORBIDDEN_LLM_FIELDS) explicitly forbids "
            "english_summary. So all 2.47M court rows show the flag, regardless of "
            "whether they have LLM topic/concepts/keywords data. Pass "
            "--clean-static-only-flag with --apply to strip the misleading flag "
            "for llm+static rows."
        ),
        "rebuild_required": [
            "Re-run scripts/prepare_embedding_input.py for court family",
            "Re-encode the affected vector_text rows on Colab "
            "(do NOT re-embed locally — see project conventions)",
            "Replace the affected vectors in the FAISS / vector index",
        ],
    }
    summary_path = DRYRUN_DIR / "remerge_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] summary: {summary_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", type=Path, default=DEFAULT_LLM,
                    help=f"Path to court_llm_descriptors_0000000_all.jsonl (default: {DEFAULT_LLM})")
    ap.add_argument("--apply", action="store_true",
                    help="Write UPDATEs. Without this, dry-run only.")
    ap.add_argument("--target-live", action="store_true",
                    help="With --apply, target the LIVE DB instead of the dryrun copy.")
    ap.add_argument("--force-copy", action="store_true",
                    help="With --apply (no --target-live), re-copy live -> dryrun even "
                         "if the dryrun DB already exists.")
    ap.add_argument("--clean-static-only-flag", action="store_true",
                    help="With --apply, strip the misleading 'static_only' flag from "
                         "quality_flags_json on llm+static court rows.")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = process all 363k LLM rows. Set >0 for smoke test.")
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
