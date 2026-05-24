#!/usr/bin/env python
"""Patch court_case_knowledge_base.jsonl with 4 deterministic fixes (no LLM calls).

Two-pass streaming over the 20.89 GB input:
  Pass 1 — read each record, run the enhanced citation regex over text_raw,
           build aggregate counters (case_uid -> paragraph count, statute_set hash
           -> paragraph count), stash slim per-record summary.
  Pass 2 — re-stream input, apply transforms (Steps 1-4), write patched output.

NO NEW LLM CALLS. Step 4 only does an in-memory lookup against the existing
court_llm_descriptors_*.jsonl cache.

Step 1 — Fix docket records missing chamber_label
  - Legacy EVG single-letter dockets (^[IUKHCBPMEF]\\s+\\d): set chamber, chamber_label,
    court=Eidgenössisches Versicherungsgericht, court_code=EVG.
  - Modern dockets whose chamber field stores only the digit (the classified-index
    backfill bug — e.g. chamber="1" instead of "1C"): re-parse from citation_canon
    to recover the full 2-char prefix, look up chamber_label.

Step 2 — Enhanced citation extractor over content.text_raw
  Patterns: BGE/ATF/DTF (with spaces, no-space, optional consideration in E./consid./c.),
  modern docket (with optional BGer/Urteil prefix and date), legacy EVG single-letter,
  multilingual Art./al./cpv./lit./let./lett./Abs./Bst./Ziff., § sections, SR/RS.
  Refs unioned with the existing precomputed references_out_raw.

Step 3 — 5 static derived fields, 100% coverage
  legal_area_static, is_leading_decision, has_substantive_role,
  co_citation_count_static, case_peer_count.

Step 4 — Attach cached LLM enrichment under llm_enrichment key
  Lookup-only against court_llm_descriptors_*.jsonl. Records without a cache
  hit get llm_enrichment=null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    ROOT / "drive_sync" / "omnilex_competition" / "retrieval_other_artifacts"
    / "court_case_knowledge_base.jsonl"
)
DEFAULT_OUTPUT = (
    ROOT / "drive_sync" / "omnilex_competition" / "retrieval_other_artifacts"
    / "court_case_knowledge_base.patched.jsonl"
)
DEFAULT_LLM_DESC = (
    ROOT / "drive_sync" / "swiss_law" / "llm_enrichment_jsonl_checkpoints"
    / "court_llm_descriptors_0000000_all.jsonl"
)
DEFAULT_MANIFEST = (
    ROOT / "drive_sync" / "omnilex_competition" / "retrieval_other_artifacts"
    / "court_case_knowledge_base.patched.manifest.json"
)
KNOWN_TOTAL_ROWS = 2_476_315


# ─────────────────────────────────────────────────────────────────────────
# Step 1 — chamber-label backfill
# ─────────────────────────────────────────────────────────────────────────

DOCKET_PREFIX_AREAS: dict[str, str] = {
    "1B": "criminal procedure and coercive measures",
    "1C": "constitutional and public law",
    "1D": "constitutional and public law",
    "1P": "constitutional and public law",
    "1A": "constitutional and public law",
    "1E": "constitutional and public law",
    "1F": "constitutional and public law",
    "1G": "constitutional and public law",
    "1S": "constitutional and public law",
    "2C": "administrative, tax, migration, and regulatory law",
    "2A": "administrative and tax law",
    "2D": "administrative law",
    "2E": "administrative and European law",
    "2F": "administrative law",
    "2G": "administrative, tax, migration, and regulatory law",
    "2P": "administrative and public law",
    "4A": "civil obligations, contract, commercial, and banking law",
    "4B": "civil law",
    "4C": "civil law",
    "4D": "civil law subsidiary constitutional matters",
    "4F": "civil law",
    "4G": "civil law",
    "4P": "civil law",
    "5A": "family law, inheritance, debt enforcement, and civil law",
    "5B": "family and civil law",
    "5C": "family and civil law",
    "5D": "civil law subsidiary constitutional matters",
    "5F": "civil law",
    "5E": "family and civil law",
    "5G": "family law, inheritance, debt enforcement, and civil law",
    "5N": "family and civil law",
    "5P": "family and civil law",
    "6A": "criminal law and administrative criminal law",
    "6B": "criminal law and criminal procedure",
    "6C": "criminal law and criminal procedure",
    "6F": "criminal law",
    "6G": "criminal law and criminal procedure",
    "6P": "constitutional and public law",
    "6S": "criminal law",
    "7B": "criminal law and criminal procedure",
    "7F": "criminal law",
    "7G": "criminal law and criminal procedure",
    "8C": "social insurance and public employment law",
    "8D": "social insurance law",
    "8F": "social insurance law",
    "8G": "social insurance and public employment law",
    "9C": "social insurance law",
    "9D": "social insurance law",
    "9E": "social insurance law",
    "9F": "social insurance law",
    "9G": "social insurance law",
    "9X": "social insurance law",
    "11Z": "civil law",
    "12T": "disciplinary proceedings",
    "13Y": "criminal law",
    "10Y": "criminal law",
}

EVG_AREAS = {
    "I": "invalidity insurance law",
    "U": "accident insurance law",
    "K": "health insurance law",
    "H": "accident and liability insurance law",
    "C": "unemployment insurance law",
    "B": "occupational pension and social insurance law",
    "P": "supplementary benefits law",
    "M": "military insurance law",
    "E": "social insurance law",
    "F": "social insurance law",
}

MODERN_DOCKET_PREFIX_RE = re.compile(r"^\s*(\d{1,2}[A-Z]{1,4})[_\.]")
EVG_PARSE_RE = re.compile(
    r"^\s*(?P<prefix>[IUKHCBPMEF])\s+(?P<serial>\d{1,5})/(?P<year>\d{2,4})"
    r"(?:\s+(?P<date>\d{1,2}\.\d{1,2}\.\d{2,4}))?"
    r"(?:\s+E\.\s*(?P<erw>[\w.]+))?\s*$",
)


def backfill_chamber(record: dict) -> str | None:
    """Return 'evg' / 'modern_prefix' / 'unknown' if the record was fixed, else None.

    Only touches docket records that currently have an empty chamber_label.
    """
    case = record["case"]
    if case["case_type"] != "docket":
        return None
    if case.get("chamber_label"):
        return None
    cit = record.get("citation_canon") or record["citation"]["citation_raw"] or ""

    # First try modern docket prefix (\d+[A-Z]+)
    m_mod = MODERN_DOCKET_PREFIX_RE.match(cit)
    if m_mod:
        prefix = m_mod.group(1)
        label = DOCKET_PREFIX_AREAS.get(prefix)
        if label:
            case["chamber"] = prefix
            case["chamber_label"] = label
            return "modern_prefix"

    # Then legacy EVG single-letter
    m_evg = EVG_PARSE_RE.match(cit)
    if m_evg:
        prefix = m_evg.group("prefix")
        serial = m_evg.group("serial")
        year = m_evg.group("year")
        yr_int = int(year)
        if len(year) == 2:
            yr_int = 2000 + yr_int if yr_int < 50 else 1900 + yr_int
        case["chamber"] = prefix
        case["chamber_label"] = EVG_AREAS.get(prefix)
        case["court"] = "Eidgenössisches Versicherungsgericht"
        case["court_code"] = "EVG"
        case["docket_number"] = f"{prefix} {serial}/{year}"
        case["case_id_canonical"] = case["docket_number"]
        if not case.get("decision_year"):
            case["decision_year"] = yr_int
        if m_evg.group("date") and not case.get("decision_date"):
            parts = m_evg.group("date").split(".")
            if len(parts) == 3:
                d, mo, y = parts
                if len(y) == 2:
                    y_int = int(y)
                    y = str(2000 + y_int if y_int < 50 else 1900 + y_int)
                case["decision_date"] = f"{y}-{mo.zfill(2)}-{d.zfill(2)}"
        return "evg"

    return "unknown"


# ─────────────────────────────────────────────────────────────────────────
# Step 2 — enhanced citation extractor
# ─────────────────────────────────────────────────────────────────────────

# Court case references
BGE_TEXT_RE = re.compile(
    r"\b(?P<rep>BGE|ATF|DTF)\s+(?P<vol>\d{1,4})\s+(?P<sect>[IVXLC]{1,5})\s+(?P<page>\d+[a-z]?)"
    r"(?:\s+(?:E\.|consid\.|c\.|cons\.)\s*[\w.]+)?",
    re.IGNORECASE,
)
BGE_NOSPACE_RE = re.compile(
    r"\b(BGE|ATF|DTF)(\d{2,4})([IVXLC]{1,5})(\d+[a-z]?)\b",
    re.IGNORECASE,
)
DOCKET_TEXT_RE = re.compile(
    r"\b(?:BGer\s+|Urteil\s+(?:des\s+Bundesgerichts\s+)?|arr[êe]t\s+(?:du\s+TF\s+)?)?"
    r"(\d{1,2}[A-Z]{1,4})[_\.](\d{1,6})/(\d{4})"
    r"(?:\s+\d{1,2}\.\d{1,2}\.\d{4})?"
    r"(?:\s+(?:E\.|consid\.|c\.|cons\.)\s*[\w.]+)?",
)
EVG_TEXT_RE = re.compile(
    r"(?<![A-Za-zÀ-ÖØ-öø-ÿ])([IUKHCBPMEF])\s+(\d{1,5})/(\d{2,4})\b"
    r"(?:\s+(?:E\.|consid\.|c\.|cons\.)\s*[\w.]+)?",
)

# Statute / article references — multilingual
ART_TEXT_RE = re.compile(
    r"\b(?:Art|Artikel|article|articolo)\.?\s+(\d+[a-zA-Z]*)"
    r"(?:\s+(?:Abs|al|cpv|para|par)\.?\s*\d+[a-zA-Z]*)?"
    r"(?:\s+(?:lit|let|lett|Bst|Ziff|ch|n|no)\.?\s*[a-zA-Z0-9]+)?"
    r"\s+(?P<code>[A-ZÄÖÜ][A-Za-zÄÖÜäöüçèéà0-9./_-]{1,20})",
    re.IGNORECASE,
)
SECTION_TEXT_RE = re.compile(
    r"§\s*\d+[a-zA-Z]*(?:\s+(?:Abs|al)\.?\s*\d+)?\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü0-9./_-]{1,20}",
)
SR_TEXT_RE = re.compile(r"\b(?:SR|RS)\s+\d[\d.]{2,}\b")


def extract_citations(text: str) -> tuple[list[str], list[str]]:
    """Return (law_refs, case_refs) found in text via the patched regex."""
    if not text:
        return [], []
    law_refs: list[str] = []
    case_refs: list[str] = []
    seen: set[str] = set()

    def _add(target: list[str], value: str) -> None:
        v = value.strip().rstrip(",;:.")
        v = re.sub(r"\s+", " ", v)
        if v and v not in seen:
            seen.add(v)
            target.append(v)

    for m in BGE_TEXT_RE.finditer(text):
        _add(case_refs, m.group(0))
    for m in BGE_NOSPACE_RE.finditer(text):
        normalized = f"{m.group(1).upper()} {m.group(2)} {m.group(3).upper()} {m.group(4)}"
        _add(case_refs, normalized)
    for m in DOCKET_TEXT_RE.finditer(text):
        _add(case_refs, m.group(0))
    for m in EVG_TEXT_RE.finditer(text):
        _add(case_refs, m.group(0))
    for m in ART_TEXT_RE.finditer(text):
        _add(law_refs, m.group(0))
    for m in SECTION_TEXT_RE.finditer(text):
        _add(law_refs, m.group(0))
    for m in SR_TEXT_RE.finditer(text):
        _add(law_refs, m.group(0))
    return law_refs, case_refs


# ─────────────────────────────────────────────────────────────────────────
# Step 3 — static derived fields
# ─────────────────────────────────────────────────────────────────────────

LEGAL_AREA_STATIC_MAP = {
    "constitutional and public law": "constitutional_public_law",
    "constitutional and procedural law": "constitutional_public_law",
    "constitutional procedure": "constitutional_public_law",
    "criminal procedure and coercive measures": "criminal_procedure",
    "criminal procedure": "criminal_procedure",
    "criminal law and criminal procedure": "criminal_law",
    "criminal law and administrative criminal law": "criminal_law",
    "criminal law": "criminal_law",
    "administrative, tax, migration, and regulatory law": "administrative_tax",
    "administrative and tax law": "administrative_tax",
    "administrative law": "administrative_tax",
    "administrative and European law": "administrative_tax",
    "administrative and public law": "administrative_tax",
    "civil obligations, contract, commercial, and banking law": "civil_law",
    "civil law": "civil_law",
    "civil law subsidiary constitutional matters": "civil_law",
    "family law, inheritance, debt enforcement, and civil law": "civil_law",
    "family and civil law": "civil_law",
    "social insurance and public employment law": "social_insurance",
    "social insurance law": "social_insurance",
    "social insurance general law": "social_insurance",
    "invalidity insurance law": "social_insurance",
    "accident insurance law": "social_insurance",
    "health insurance law": "social_insurance",
    "accident and liability insurance law": "social_insurance",
    "unemployment insurance law": "social_insurance",
    "occupational pension and social insurance law": "social_insurance",
    "supplementary benefits law": "social_insurance",
    "military insurance law": "social_insurance",
    "disciplinary proceedings": "disciplinary",
}

BGE_SECT_TO_STATIC = {
    "I":   "constitutional_public_law",
    "II":  "administrative_tax",
    "III": "civil_law",
    "IV":  "criminal_law",
    "V":   "social_insurance",
}

SUBSTANTIVE_ROLES = {"reasoning", "legal_standard", "application", "holding"}

LEADING_DECISION_RE = re.compile(r"^\s*(BGE|ATF|DTF)\s+", re.IGNORECASE)


def derive_legal_area_static(record: dict) -> str:
    chamber_label = record["case"].get("chamber_label")
    if chamber_label and chamber_label in LEGAL_AREA_STATIC_MAP:
        return LEGAL_AREA_STATIC_MAP[chamber_label]
    if record["case"]["case_type"] == "bge":
        sect = record["case"].get("bge_section")
        if sect in BGE_SECT_TO_STATIC:
            return BGE_SECT_TO_STATIC[sect]
    if record["case"]["case_type"] == "docket":
        ch = record["case"].get("chamber")
        if ch in EVG_AREAS:
            return "social_insurance"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────
# Step 4 — LLM enrichment cache loader (lookup only)
# ─────────────────────────────────────────────────────────────────────────

def load_llm_descriptors_full(path: Path) -> dict[str, dict]:
    """Load complete llm_enrichment dict keyed by citation."""
    if not path.exists():
        print(f"[warn] LLM descriptors missing: {path}", file=sys.stderr)
        return {}
    out: dict[str, dict] = {}
    total_bytes = path.stat().st_size
    pbar = None
    if HAVE_TQDM:
        pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
                    desc="[llm] indexing", mininterval=1.0, file=sys.stderr)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if pbar is not None:
                pbar.update(len(line.encode("utf-8", errors="replace")))
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cit = obj.get("citation")
            if not cit:
                continue
            enr = obj.get("llm_enrichment") or {}
            if not enr:
                continue
            out[cit] = enr
    if pbar is not None:
        pbar.close()
    print(f"[llm] loaded {len(out)} citation -> enrichment entries", file=sys.stderr)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",    type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output",   type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--llm-desc", type=Path, default=DEFAULT_LLM_DESC)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--limit",    type=int, default=0, help="Process only N records (smoke test)")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print(f"[paths]\n  input  = {args.input}\n  output = {args.output}\n  llm    = {args.llm_desc}",
          file=sys.stderr)

    # ── Phase 0: load LLM descriptors index (Step 4)
    llm_idx = load_llm_descriptors_full(args.llm_desc)

    total = args.limit if args.limit else KNOWN_TOTAL_ROWS

    # ── Phase 1: aggregation
    print("\n[phase1] streaming aggregation pass...", file=sys.stderr)
    t1 = time.time()
    case_paragraph_count: Counter = Counter()
    statute_set_count: Counter = Counter()
    summaries: list[tuple | None] = []

    out_ref_before_nonempty = 0
    out_ref_after_nonempty = 0
    n_records = 0
    sample_new_refs: list[tuple[str, list[str]]] = []

    pbar1 = (tqdm(total=total, desc="[phase1] read", unit="rec",
                  mininterval=1.0, file=sys.stderr, smoothing=0.05)
             if HAVE_TQDM else None)

    with args.input.open("r", encoding="utf-8", buffering=4 * 1024 * 1024) as fin:
        for line_idx, line in enumerate(fin):
            if args.limit and line_idx >= args.limit:
                break
            if pbar1 is not None:
                pbar1.update(1)
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                summaries.append(None)
                continue
            n_records += 1

            text_raw = rec.get("content", {}).get("text_raw") or ""
            law_refs, case_refs = extract_citations(text_raw)
            existing_raw = rec.get("references", {}).get("references_out_raw") or []

            had_before = bool(existing_raw)
            union = set(existing_raw) | set(law_refs) | set(case_refs)
            has_after = bool(union)
            if had_before:
                out_ref_before_nonempty += 1
            if has_after:
                out_ref_after_nonempty += 1

            # Capture a few interesting "newly captured" examples for the report
            if not had_before and has_after and len(sample_new_refs) < 5:
                sample_new_refs.append((rec.get("citation_canon", "?"),
                                        list((set(law_refs) | set(case_refs)))[:5]))

            case_uid = rec["case"].get("case_uid") or rec["citation"]["citation_group_id"]
            case_paragraph_count[case_uid] += 1

            union_laws = sorted(
                {r for r in existing_raw if r and re.search(r"\b(Art|Artikel|article|articolo|§|SR|RS)\b", r, re.IGNORECASE)}
                | set(law_refs)
            )
            if union_laws:
                h = hashlib.blake2b("|".join(union_laws).encode("utf-8"), digest_size=8).hexdigest()
                statute_set_count[h] += 1
            else:
                h = None

            summaries.append((case_uid, h, law_refs, case_refs))

    if pbar1 is not None:
        pbar1.close()

    elapsed1 = time.time() - t1
    print(f"\n[phase1] done in {elapsed1:.1f}s — {n_records} records", file=sys.stderr)
    print(f"  unique cases:               {len(case_paragraph_count)}", file=sys.stderr)
    print(f"  unique statute-set hashes:  {len(statute_set_count)}", file=sys.stderr)
    print(f"  out-ref coverage before:    {out_ref_before_nonempty}/{n_records}"
          f" = {out_ref_before_nonempty/max(n_records,1)*100:.2f}%", file=sys.stderr)
    print(f"  out-ref coverage after:     {out_ref_after_nonempty}/{n_records}"
          f" = {out_ref_after_nonempty/max(n_records,1)*100:.2f}%", file=sys.stderr)

    # ── Phase 2: transform + write
    print("\n[phase2] transforming + writing...", file=sys.stderr)
    t2 = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_llm_attached = 0
    n_step1_evg = 0
    n_step1_modern = 0
    n_step1_unfixable = 0
    n_step1_skipped_already_ok = 0
    field_coverage: Counter = Counter()
    final_case_type: Counter = Counter()
    final_legal_area: Counter = Counter()
    final_court_code: Counter = Counter()

    pbar2 = (tqdm(total=total, desc="[phase2] write", unit="rec",
                  mininterval=1.0, file=sys.stderr, smoothing=0.05)
             if HAVE_TQDM else None)

    with args.input.open("r", encoding="utf-8", buffering=4 * 1024 * 1024) as fin, \
         args.output.open("w", encoding="utf-8", buffering=4 * 1024 * 1024) as fout:
        for line_idx, line in enumerate(fin):
            if args.limit and line_idx >= args.limit:
                break
            if pbar2 is not None:
                pbar2.update(1)
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line if line.endswith("\n") else line + "\n")
                continue
            summary = summaries[line_idx]
            if summary is None:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            case_uid, statute_hash, law_refs, case_refs = summary

            # ── Step 1: chamber backfill (EVG + modern-prefix)
            outcome = backfill_chamber(rec)
            if outcome == "evg":
                n_step1_evg += 1
            elif outcome == "modern_prefix":
                n_step1_modern += 1
            elif outcome == "unknown":
                n_step1_unfixable += 1
            else:
                n_step1_skipped_already_ok += 1

            # ── Step 2: union new + existing out-refs
            refs_block = rec.setdefault("references", {})
            existing_raw = refs_block.get("references_out_raw") or []
            existing_laws = refs_block.get("references_out_laws") or []
            existing_cases = refs_block.get("references_out_cases") or []
            new_laws_union = list(dict.fromkeys(list(existing_laws) + law_refs))
            new_cases_union = list(dict.fromkeys(list(existing_cases) + case_refs))
            new_raw_union = list(dict.fromkeys(list(existing_raw) + law_refs + case_refs))
            refs_block["references_out_raw"] = new_raw_union
            refs_block["references_out_laws"] = new_laws_union
            refs_block["references_out_cases"] = new_cases_union
            refs_block["references_out_count"] = len(new_raw_union)
            refs_block["references_out_extracted_from_text"] = len(law_refs) + len(case_refs)

            # ── Step 3: 5 static fields
            rec["legal_area_static"] = derive_legal_area_static(rec)
            rec["is_leading_decision"] = bool(
                LEADING_DECISION_RE.match(rec.get("citation_canon") or "")
            )
            para_role = (rec.get("semantic") or {}).get("paragraph_role")
            rec["has_substantive_role"] = para_role in SUBSTANTIVE_ROLES
            rec["co_citation_count_static"] = (
                statute_set_count[statute_hash] - 1 if statute_hash else 0
            )
            rec["case_peer_count"] = max(0, case_paragraph_count[case_uid] - 1)

            # ── Step 4: attach cached LLM enrichment (lookup only)
            cit = rec.get("citation_canon") or rec.get("citation", {}).get("citation_raw")
            cached = llm_idx.get(cit) if cit else None
            if cached:
                rec["llm_enrichment"] = cached
                n_llm_attached += 1
            else:
                rec["llm_enrichment"] = None

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1
            final_case_type[rec["case"]["case_type"]] += 1
            final_legal_area[rec["legal_area_static"]] += 1
            final_court_code[rec["case"].get("court_code") or ""] += 1
            for k in ("legal_area_static", "is_leading_decision", "has_substantive_role",
                      "co_citation_count_static", "case_peer_count"):
                if k in rec and rec[k] is not None:
                    field_coverage[k] += 1

    if pbar2 is not None:
        pbar2.close()
    elapsed2 = time.time() - t2

    # ── Manifest + report
    manifest = {
        "input": str(args.input),
        "output": str(args.output),
        "rows_written": n_written,
        "elapsed_phase1_seconds": round(elapsed1, 1),
        "elapsed_phase2_seconds": round(elapsed2, 1),
        "step1_chamber_backfill": {
            "evg_fixed":          n_step1_evg,
            "modern_prefix_fixed": n_step1_modern,
            "unfixable":          n_step1_unfixable,
            "already_ok":         n_step1_skipped_already_ok,
        },
        "step2_out_ref_coverage": {
            "before_pct": round(out_ref_before_nonempty / max(n_written, 1) * 100, 2),
            "after_pct":  round(out_ref_after_nonempty / max(n_written, 1) * 100, 2),
            "samples_newly_captured": sample_new_refs,
        },
        "step3_field_coverage": {
            k: f"{v}/{n_written} ({v/max(n_written,1)*100:.2f}%)"
            for k, v in field_coverage.items()
        },
        "step4_llm_attached": n_llm_attached,
        "step4_llm_attached_pct": round(n_llm_attached / max(n_written, 1) * 100, 2),
        "step4_zero_new_llm_calls": True,
        "final_case_type_counts": dict(final_case_type),
        "final_legal_area_static_counts": dict(final_legal_area),
        "final_court_code_counts": {k: v for k, v in final_court_code.most_common()},
        "schema": "court_case_knowledge_base.v2_patched",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    bar = "=" * 64
    print(f"\n{bar}\nFINAL REPORT\n{bar}", file=sys.stderr)
    print(f"records written: {n_written}", file=sys.stderr)
    print(f"\nSTEP 1 — chamber-label backfill:", file=sys.stderr)
    print(f"  EVG single-letter dockets fixed:   {n_step1_evg}", file=sys.stderr)
    print(f"  modern-prefix dockets fixed:       {n_step1_modern}", file=sys.stderr)
    print(f"  still unfixable (no match):        {n_step1_unfixable}", file=sys.stderr)
    print(f"  already had chamber_label:         {n_step1_skipped_already_ok}", file=sys.stderr)
    print(f"\nSTEP 2 — out-reference coverage:", file=sys.stderr)
    print(f"  before:  {out_ref_before_nonempty / max(n_written, 1) * 100:.2f}%", file=sys.stderr)
    print(f"  after:   {out_ref_after_nonempty / max(n_written, 1) * 100:.2f}%", file=sys.stderr)
    print(f"  newly captured samples:", file=sys.stderr)
    for src, refs in sample_new_refs:
        print(f"    {src!r}  →  {refs}", file=sys.stderr)
    print(f"\nSTEP 3 — static field coverage (all should be 100%):", file=sys.stderr)
    for k in ("legal_area_static", "is_leading_decision", "has_substantive_role",
              "co_citation_count_static", "case_peer_count"):
        cov = field_coverage.get(k, 0)
        print(f"  {k}: {cov}/{n_written} ({cov/max(n_written,1)*100:.2f}%)", file=sys.stderr)
    print(f"\nSTEP 4 — LLM enrichment:", file=sys.stderr)
    print(f"  attached from cache:               {n_llm_attached} "
          f"({n_llm_attached/max(n_written,1)*100:.2f}%)", file=sys.stderr)
    print(f"  records with llm_enrichment=null:  {n_written - n_llm_attached}", file=sys.stderr)
    print(f"  new LLM API calls made:            0 (lookup-only, by design)", file=sys.stderr)
    print(f"{bar}", file=sys.stderr)
    print(f"[done] output  -> {args.output}", file=sys.stderr)
    print(f"[done] manifest-> {args.manifest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
