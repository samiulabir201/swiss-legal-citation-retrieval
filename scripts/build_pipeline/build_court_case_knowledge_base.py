#!/usr/bin/env python
"""Build court_case_knowledge_base.jsonl mirroring laws_knowledge_base.jsonl.

One record per paragraph (consideration) in data/court_considerations.csv.
Joins three enrichment sources by raw citation string:

  1. court_considerations.csv                                    (raw text)
  2. court_llm_descriptors_0000000_all.jsonl                     (LLM legal_area, topic, doctrinal_rule, legal_test,
                                                                  fact_pattern_tags, paragraph_role, concepts, terms,
                                                                  specificity_score, authority_role)
  3. court_considerations_classified_citations.jsonl             (parsed segments: docket, date, year,
                                                                  chamber, legal_area_code, consideration)
  4. court_considerations_links.json                             (citation graph: source -> out-references)

Output blocks mirror the laws schema: id, record_uid, provision_id, citation_uri,
citation_canon, citation, case, structure, content, semantic, topics, keywords,
entities, references, multilingual, search, authority, embeddings, metadata.

Court-specific extras under `case`:
    case_type ("bge" | "docket" | "cantonal"), docket_number, bge_volume,
    bge_section, bge_page, decision_date, decision_year, court, court_code,
    chamber, chamber_label (from DOCKET_PREFIX_AREAS / BGE_DIVISION_AREAS),
    legal_area_static, language_original, case_importance.

Authority score combines: BGE boost + chamber-prefix weight + in-degree
(from the inverted citation graph) + recency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

# ---------------------------------------------------------------------------
# Path defaults (override via CLI)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_COURT_CSV = ROOT / "data" / "court_considerations.csv"
DEFAULT_LLM_DESC = (
    ROOT / "drive_sync" / "swiss_law" / "llm_enrichment_jsonl_checkpoints"
    / "court_llm_descriptors_0000000_all.jsonl"
)
DEFAULT_CLASSIFIED = (
    ROOT / "data_insights" / "citation_graph_db_and_edges"
    / "court_considerations_classified_citations.jsonl"
)
DEFAULT_LINKS_JSON = (
    ROOT / "drive_sync" / "swiss_law" / "colab_data_insights_mirror"
    / "court_considerations_links.json"
)
DEFAULT_OUT = (
    ROOT / "drive_sync" / "omnilex_competition" / "retrieval_other_artifacts"
    / "court_case_knowledge_base.jsonl"
)
DEFAULT_MANIFEST = (
    ROOT / "drive_sync" / "omnilex_competition" / "retrieval_other_artifacts"
    / "court_case_knowledge_base.manifest.json"
)

# ---------------------------------------------------------------------------
# Reuse deterministic mappings from build_court_authority_cards.py
# (chamber/division -> legal area). Copied here so this script stays standalone.
# ---------------------------------------------------------------------------

BGE_DIVISION_AREAS = {
    "I":   "constitutional and public law",
    "II":  "administrative, tax, migration, and regulatory law",
    "III": "civil law",
    "IV":  "criminal law and criminal procedure",
    "V":   "social insurance law",
}

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
    "I":  "invalidity insurance law",
    "U":  "accident insurance law",
    "K":  "health insurance law",
    "H":  "accident and liability insurance law",
    "C":  "unemployment insurance law",
    "B":  "occupational pension and social insurance law",
    "P":  "supplementary benefits law",
    "M":  "military insurance law",
    "E":  "social insurance law",
    "F":  "social insurance law",
}

# ---------------------------------------------------------------------------
# Citation parsing (paragraph-level)
# ---------------------------------------------------------------------------

# BGE: "BGE 139 I 2 E. 5.7"   /   ATF (FR)  /  DTF (IT)
BGE_RE = re.compile(
    r"^\s*(?P<reporter>BGE|ATF|DTF)\s+(?P<vol>\d{1,4})\s+(?P<sect>[IVXLC]{1,5})\s+(?P<page>\d+[a-z]?)"
    r"(?:\s+E\.\s*(?P<erw>[\w.]+))?\s*$",
    re.IGNORECASE,
)
# Docket: "5A_823/2016 22.03.2017 E. 2.1"  /  "10Y.1/2003 05.11.2003 E. 4"
DOCKET_RE = re.compile(
    r"^\s*(?P<prefix>\d{1,2}[A-Z]{1,4})[_\.](?P<serial>\d{1,6})/(?P<year>\d{4})"
    r"(?:\s+(?P<date>\d{2}\.\d{2}\.\d{4}))?"
    r"(?:\s+E\.\s*(?P<erw>[\w.]+))?\s*$",
)
# Cantonal-ish: "VGE 895/05", "EGV-SZ 2006", "BVGE 2009/12"
CANTONAL_RE = re.compile(
    r"^\s*(?P<prefix>VGE|EGV|ACJC|ATA|TAF|BVGer|BVGE|TPF|SK|BB|RR|VG|KG|OGer|VB)"
    r"[-\s/_.]*(?P<rest>[A-Z0-9_./-]+)(?:\s+E\.\s*(?P<erw>[\w.]+))?\s*$",
)
# Erwägung tail tokenized: "E. 2.1", "E. A", "E. 2.1.3"
ERW_LETTER_RE = re.compile(r"^[A-Z]$")

LANG_HINTS = {
    "de": ("der ", "die ", "das ", "und ", "Bundes", "Erwägung", "Beschwerde",
           "Vorinstanz", "Recht", "Urteil"),
    "fr": ("le ", "la ", "les ", "et ", "Tribunal", "considérant", "recours",
           "instance", "arrêt", "droit"),
    "it": ("il ", "lo ", "la ", "i ", "Tribunale", "considerando", "ricorso",
           "sentenza", "diritto"),
}


def detect_language(text: str) -> str:
    if not text:
        return "de"
    sample = text[:600].lower()
    scores = {lang: sum(sample.count(h.lower()) for h in hints)
              for lang, hints in LANG_HINTS.items()}
    return max(scores, key=scores.get) or "de"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.replace("\u00A0", " ").replace("\r", " ").replace("\t", " ")
    return " ".join(value.split())


def slugify_citation(citation: str) -> str:
    s = unicodedata.normalize("NFKD", citation).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w./]+", "_", s).strip("_").lower()
    s = s.replace("/", "_").replace(".", "_")
    return re.sub(r"_+", "_", s)


def parse_citation(citation: str) -> dict[str, Any]:
    """Return parsed segments for a paragraph-level citation."""
    out = {
        "case_type": "unknown",
        "case_id_canonical": citation,
        "case_group_id": citation,
        "docket_number": None,
        "bge_volume": None,
        "bge_section": None,
        "bge_page": None,
        "bge_reporter": None,
        "decision_date": None,
        "decision_year": None,
        "chamber": None,
        "consideration_label": None,
        "consideration_root": None,
        "consideration_depth": 0,
    }

    m = BGE_RE.match(citation)
    if m:
        out["case_type"] = "bge"
        out["bge_reporter"] = m.group("reporter").upper()
        out["bge_volume"] = int(m.group("vol"))
        out["bge_section"] = m.group("sect").upper()
        out["bge_page"] = m.group("page")
        out["chamber"] = m.group("sect").upper()
        out["case_id_canonical"] = (
            f"{out['bge_reporter']} {out['bge_volume']} {out['bge_section']} {out['bge_page']}"
        )
        out["case_group_id"] = out["case_id_canonical"]
        if m.group("erw"):
            out["consideration_label"] = m.group("erw")
        return out

    m = DOCKET_RE.match(citation)
    if m:
        out["case_type"] = "docket"
        prefix = m.group("prefix")
        out["chamber"] = prefix
        out["docket_number"] = f"{prefix}_{m.group('serial')}/{m.group('year')}"
        out["case_id_canonical"] = out["docket_number"]
        out["case_group_id"] = out["docket_number"]
        out["decision_year"] = int(m.group("year"))
        if m.group("date"):
            d, mo, y = m.group("date").split(".")
            out["decision_date"] = f"{y}-{mo}-{d}"
            out["decision_year"] = int(y)
        if m.group("erw"):
            out["consideration_label"] = m.group("erw")
        return out

    m = CANTONAL_RE.match(citation)
    if m:
        out["case_type"] = "cantonal"
        out["chamber"] = m.group("prefix")
        out["case_id_canonical"] = f"{m.group('prefix')} {m.group('rest')}"
        out["case_group_id"] = out["case_id_canonical"]
        if m.group("erw"):
            out["consideration_label"] = m.group("erw")
        return out

    # Fallback: keep raw
    return out


def parse_consideration(label: str | None) -> tuple[str | None, int]:
    """Return (root_consideration, depth). 'E. 2.1.3' -> ('2', 3). 'E. A' -> ('A', 0)."""
    if not label:
        return None, 0
    label = label.strip()
    if ERW_LETTER_RE.match(label):
        # Letters mark facts paragraphs (Sachverhalt) in Swiss style
        return label, 0
    parts = label.split(".")
    return parts[0], len(parts)


# ---------------------------------------------------------------------------
# Citation alias generation (used for retrieval-time matching)
# ---------------------------------------------------------------------------

def make_aliases(parsed: dict[str, Any], raw: str) -> list[str]:
    aliases: list[str] = [raw]
    canon = parsed["case_id_canonical"]
    consid = parsed["consideration_label"]

    if parsed["case_type"] == "bge":
        vol, sect, page = parsed["bge_volume"], parsed["bge_section"], parsed["bge_page"]
        for rep in ("BGE", "ATF", "DTF"):
            base = f"{rep} {vol} {sect} {page}"
            aliases.append(base)
            if consid:
                aliases.append(f"{base} E. {consid}")
                aliases.append(f"{base} consid. {consid}")     # FR
                aliases.append(f"{base} consid {consid}")
    elif parsed["case_type"] == "docket":
        docket = parsed["docket_number"]
        aliases.append(docket)
        if consid:
            aliases.append(f"{docket} E. {consid}")
            aliases.append(f"{docket} consid. {consid}")
        if parsed["decision_date"]:
            d, mo, y = parsed["decision_date"][8:10], parsed["decision_date"][5:7], parsed["decision_date"][:4]
            aliases.append(f"{docket} {d}.{mo}.{y}")
    elif parsed["case_type"] == "cantonal":
        aliases.append(canon)

    # de-dup preserving order
    seen, out = set(), []
    for a in aliases:
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return out


# ---------------------------------------------------------------------------
# Reference classification (split out-refs into laws vs cases)
# ---------------------------------------------------------------------------

LAW_REF_RE = re.compile(r"\b(?:Art\.?|art\.?|§|SR|RS)\b", re.IGNORECASE)
CASE_REF_RE = re.compile(r"\bBGE|ATF|DTF\b|^\d{1,2}[A-Z]{1,4}[_.]\d{1,6}/\d{4}")


def classify_references(refs: list[str]) -> tuple[list[str], list[str]]:
    laws, cases = [], []
    for r in refs:
        if CASE_REF_RE.search(r):
            cases.append(r)
        elif LAW_REF_RE.search(r):
            laws.append(r)
        else:
            laws.append(r)  # safer default
    return laws, cases


# ---------------------------------------------------------------------------
# Chunking (mirror laws KB: paragraphs ~500 chars, char offsets)
# ---------------------------------------------------------------------------

def chunk_text(text: str, target_chars: int = 520, hard_max: int = 800) -> list[dict[str, Any]]:
    text = text or ""
    if not text:
        return []
    chunks: list[dict[str, Any]] = []
    # Sentence-ish split on .!? respecting whitespace
    parts = re.split(r"(?<=[.!?])\s+", text)
    buf, buf_start, offset = [], 0, 0
    cur_offset = 0
    for part in parts:
        if not buf:
            buf_start = cur_offset
        if buf and sum(len(x) for x in buf) + len(part) > target_chars:
            chunk_text_ = " ".join(buf)
            if len(chunk_text_) > hard_max:
                # split hard
                for i in range(0, len(chunk_text_), hard_max):
                    chunks.append({"text": chunk_text_[i:i + hard_max],
                                   "start_offset": buf_start + i})
            else:
                chunks.append({"text": chunk_text_, "start_offset": buf_start})
            buf, buf_start = [part], cur_offset
        else:
            buf.append(part)
        cur_offset += len(part) + 1
    if buf:
        chunk_text_ = " ".join(buf)
        chunks.append({"text": chunk_text_, "start_offset": buf_start})
    # Add chunk_id
    for i, c in enumerate(chunks, 1):
        c["chunk_id"] = str(i)
    return chunks


# ---------------------------------------------------------------------------
# Authority scoring
# ---------------------------------------------------------------------------

def authority_score(parsed: dict[str, Any], in_count: int, specificity: float | None) -> float:
    base = 0.0
    if parsed["case_type"] == "bge":
        base = 0.85
    elif parsed["case_type"] == "docket":
        base = 0.55
    elif parsed["case_type"] == "cantonal":
        base = 0.35
    else:
        base = 0.20
    # In-degree boost (saturates at 200)
    in_boost = min(in_count, 200) / 200 * 0.10
    # Recency boost (newer cases get small bump)
    rec_boost = 0.0
    yr = parsed.get("decision_year") or parsed.get("bge_volume")
    if yr:
        # BGE volume 100 ~= 1974, volume 150 ~= 2024. Docket year is real year.
        if parsed["case_type"] == "bge":
            rec_year = 1874 + yr  # approximate
        else:
            rec_year = yr
        if 1990 <= rec_year <= 2030:
            rec_boost = (rec_year - 1990) / 40 * 0.03
    # Specificity from LLM
    spec_boost = (specificity or 0.5) * 0.02
    return round(min(base + in_boost + rec_boost + spec_boost, 1.0), 4)


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def load_llm_descriptors(path: Path, verbose: bool = True) -> dict[str, dict[str, Any]]:
    """Index LLM descriptors by citation string. Keeps only fields we use."""
    if not path.exists():
        print(f"[warn] LLM descriptors missing: {path}", file=sys.stderr)
        return {}
    out: dict[str, dict[str, Any]] = {}
    t0 = time.time()
    KEEP = {
        "legal_area", "primary_domain", "secondary_domain", "legal_domain_path",
        "topic", "subtopic", "micro_topic", "concepts_en", "terms_original",
        "doctrinal_rule", "legal_test", "fact_pattern_tags", "procedural_context",
        "paragraph_role", "authority_role", "specificity_score",
    }
    total_bytes = path.stat().st_size
    pbar = None
    if HAVE_TQDM and verbose:
        pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
                    desc="[llm] indexing", mininterval=1.0, file=sys.stderr)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
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
            slim = {k: enr.get(k) for k in KEEP if k in enr}
            slim["_language"] = obj.get("language")
            slim["_legal_area_static"] = obj.get("legal_area_static")
            out[cit] = slim
            if pbar is None and verbose and (i + 1) % 200_000 == 0:
                print(f"  [llm] {i+1:>9} lines  elapsed={time.time()-t0:.1f}s  unique={len(out)}",
                      file=sys.stderr)
    if pbar is not None:
        pbar.close()
    if verbose:
        print(f"[llm] loaded {len(out)} citations in {time.time()-t0:.1f}s", file=sys.stderr)
    return out


def load_classified_citations(path: Path, verbose: bool = True) -> dict[str, dict[str, Any]]:
    """Index classified citations by citation string. Keeps `segments` + `pattern` + `subfamily`."""
    if not path.exists():
        print(f"[warn] classified citations missing: {path}", file=sys.stderr)
        return {}
    out: dict[str, dict[str, Any]] = {}
    t0 = time.time()
    total_bytes = path.stat().st_size
    pbar = None
    if HAVE_TQDM and verbose:
        pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
                    desc="[classified] indexing", mininterval=1.0, file=sys.stderr)
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
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
            out[cit] = {
                "pattern": obj.get("pattern"),
                "subfamily": obj.get("subfamily"),
                "segments": obj.get("segments") or {},
                "family": obj.get("family"),
            }
            if pbar is None and verbose and (i + 1) % 200_000 == 0:
                print(f"  [classified] {i+1:>9} lines  elapsed={time.time()-t0:.1f}s",
                      file=sys.stderr)
    if pbar is not None:
        pbar.close()
    if verbose:
        print(f"[classified] loaded {len(out)} citations in {time.time()-t0:.1f}s",
              file=sys.stderr)
    return out


def load_links_and_indegree(path: Path, verbose: bool = True) -> tuple[dict[str, list[str]], Counter]:
    """Return (out_refs[source]=refs, in_count[ref]=count)."""
    if not path.exists():
        print(f"[warn] links file missing: {path}", file=sys.stderr)
        return {}, Counter()
    t0 = time.time()
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    out_refs: dict[str, list[str]] = {}
    in_count: Counter = Counter()
    edges = obj.get("source_to_references") or []
    for edge in edges:
        src = edge.get("source")
        refs = edge.get("references") or []
        if not src:
            continue
        out_refs[src] = refs
        for r in refs:
            in_count[r] += 1
    if verbose:
        print(f"[links] {len(out_refs)} source paragraphs, "
              f"{len(in_count)} unique reference targets, "
              f"elapsed={time.time()-t0:.1f}s", file=sys.stderr)
    return out_refs, in_count


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

NOTIFICATION_RE = re.compile(
    r"(dieses urteil wird den parteien|le pr[eé]sent arr[eê]t est communiqu[eé]|"
    r"la presente sentenza [eèé] intimata|schriftlich mitgeteilt|gerichtsschreiber|greffier)",
    re.IGNORECASE,
)
HOLDING_HINT_RE = re.compile(
    r"(demnach erkennt|par ces motifs|per questi motivi|"
    r"das bundesgericht erkennt|le tribunal f[eé]d[eé]ral prononce|"
    r"il tribunale federale pronuncia|wird (abgewiesen|gutgeheissen|aufgehoben))",
    re.IGNORECASE,
)


def infer_paragraph_role(consid_label: str | None, text: str, llm_role: str | None) -> str:
    if llm_role:
        return llm_role
    if consid_label and ERW_LETTER_RE.match(consid_label):
        return "facts"
    if HOLDING_HINT_RE.search(text or ""):
        return "dispositif"
    if consid_label and consid_label.startswith("1") and "eintreten" in (text or "").lower():
        return "admissibility"
    return "reasoning"


def build_record(
    row_idx: int,
    citation: str,
    text: str,
    llm: dict[str, Any] | None,
    classified: dict[str, Any] | None,
    out_refs: list[str],
    in_count: int,
) -> dict[str, Any]:
    parsed = parse_citation(citation)
    # Backfill from classified segments if our parse missed something
    if classified and classified.get("segments"):
        seg = classified["segments"]
        if seg.get("docket") and not parsed["docket_number"]:
            parsed["docket_number"] = seg["docket"]
            parsed["case_id_canonical"] = parsed["docket_number"]
            parsed["case_group_id"] = parsed["docket_number"]
            parsed["case_type"] = "docket"
        if seg.get("decision_year") and not parsed["decision_year"]:
            try:
                parsed["decision_year"] = int(seg["decision_year"])
            except (TypeError, ValueError):
                pass
        if seg.get("decision_date") and not parsed["decision_date"]:
            try:
                d, mo, y = seg["decision_date"].split(".")
                parsed["decision_date"] = f"{y}-{mo}-{d}"
            except Exception:
                pass
        if seg.get("court_chamber") and not parsed["chamber"]:
            parsed["chamber"] = seg["court_chamber"]
        if seg.get("consideration") and not parsed["consideration_label"]:
            parsed["consideration_label"] = seg["consideration"]

    text_clean = clean_text(text)
    language = (llm or {}).get("_language") or detect_language(text_clean)
    is_boilerplate = bool(NOTIFICATION_RE.search(text_clean))

    case_slug = slugify_citation(parsed["case_group_id"])
    consid_label = parsed["consideration_label"]
    consid_root, consid_depth = parse_consideration(consid_label)
    consid_slug = slugify_citation(consid_label) if consid_label else "body"

    # Stable IDs
    case_id = f"ch-court-{case_slug}"
    paragraph_id = f"{case_id}_e_{consid_slug}" if consid_label else case_id
    record_uid = f"{paragraph_id}__{row_idx}"
    case_uid = hashlib.blake2b(parsed["case_group_id"].encode("utf-8"), digest_size=6).hexdigest()

    aliases = make_aliases(parsed, citation)
    chunks = chunk_text(text_clean)
    for c in chunks:
        c["citation"] = citation
        c["case_uid"] = case_uid
        c["paragraph_id"] = paragraph_id

    # Chamber label (deterministic from prefix tables)
    chamber_label = None
    if parsed["case_type"] == "bge" and parsed["bge_section"]:
        chamber_label = BGE_DIVISION_AREAS.get(parsed["bge_section"])
    elif parsed["case_type"] == "docket" and parsed["chamber"]:
        chamber_label = DOCKET_PREFIX_AREAS.get(parsed["chamber"])

    # Case importance (parent-level prior)
    if parsed["case_type"] == "bge":
        case_importance = 1.0
    elif parsed["case_type"] == "docket":
        case_importance = 0.7
    elif parsed["case_type"] == "cantonal":
        case_importance = 0.4
    else:
        case_importance = 0.3

    # LLM-derived enrichments
    L = llm or {}
    specificity = L.get("specificity_score")
    paragraph_role = infer_paragraph_role(consid_label, text_clean, L.get("paragraph_role"))

    laws_out, cases_out = classify_references(out_refs)

    record = {
        "id": paragraph_id,
        "record_uid": record_uid,
        "provision_id": case_id,
        "citation_uri": f"eli://ch/court/{case_slug}"
                        + (f"/e/{consid_slug}" if consid_label else ""),
        "citation_canon": citation,
        "citation": {
            "citation_raw": citation,
            "citation_match_key": citation.lower().replace(".", "").replace("  ", " ").strip(),
            "citation_group_id": case_id,
            "citation_index_key": case_slug + (f"_e_{consid_slug}" if consid_label else ""),
            "consideration_label": consid_label,
            "consideration_root": consid_root,
            "consideration_depth": consid_depth,
            "is_letter_consideration": bool(consid_label and ERW_LETTER_RE.match(consid_label)),
            "citation_patterns": aliases,
            "citation_aliases": aliases,
        },
        "case": {
            "case_uid": case_uid,
            "case_id_canonical": parsed["case_id_canonical"],
            "case_type": parsed["case_type"],
            "docket_number": parsed["docket_number"],
            "bge_reporter": parsed["bge_reporter"],
            "bge_volume": parsed["bge_volume"],
            "bge_section": parsed["bge_section"],
            "bge_page": parsed["bge_page"],
            "decision_date": parsed["decision_date"],
            "decision_year": parsed["decision_year"],
            "court": "Bundesgericht" if parsed["case_type"] in ("bge", "docket") else None,
            "court_code": "BGer" if parsed["case_type"] in ("bge", "docket") else parsed["chamber"],
            "chamber": parsed["chamber"],
            "chamber_label": chamber_label,
            "legal_area_static": L.get("_legal_area_static"),
            "jurisdiction_country": "CH",
            "jurisdiction_level": "federal" if parsed["case_type"] in ("bge", "docket") else "cantonal",
            "language_original": language,
            "case_importance": case_importance,
        },
        "structure": {
            "structure_path": (f"E. {consid_label}" if consid_label else "body"),
            "consideration_parent": (f"{case_id}_e_{consid_root}" if consid_root and consid_depth > 1 else None),
            "consideration_root_id": (f"{case_id}_e_{consid_root}" if consid_root else None),
            "consideration_number": consid_label,
            "consideration_depth": consid_depth,
            "heading_hierarchy": [],
        },
        "content": {
            "text_raw": text or "",
            "text_clean": text_clean,
            "chunks": chunks,
            "char_len": len(text_clean),
            "token_estimate": max(1, len(text_clean) // 4),
            "is_notification_boilerplate": is_boilerplate,
            "is_empty": not bool(text_clean),
        },
        "semantic": {
            "paragraph_role": paragraph_role,
            "doctrinal_rule": L.get("doctrinal_rule") or "",
            "legal_test": L.get("legal_test") or "",
            "procedural_context": L.get("procedural_context") or "",
            "specificity_score": specificity,
            "authority_role": L.get("authority_role") or [],
            "is_holding": paragraph_role in ("dispositif", "holding"),
            "is_facts": paragraph_role == "facts",
            "is_reasoning": paragraph_role == "reasoning",
        },
        "topics": {
            "legal_area_en": L.get("legal_area"),
            "primary_domain_en": L.get("primary_domain"),
            "secondary_domain_en": L.get("secondary_domain"),
            "legal_domain_path": L.get("legal_domain_path") or [],
            "topic_en": L.get("topic"),
            "subtopic_en": L.get("subtopic"),
            "micro_topic_en": L.get("micro_topic"),
            "fact_pattern_tags": L.get("fact_pattern_tags") or [],
        },
        "keywords": {
            "concepts_en": L.get("concepts_en") or [],
            "terms_original": L.get("terms_original") or [],
        },
        "entities": {
            "courts": ["Bundesgericht"] if parsed["case_type"] in ("bge", "docket") else [],
            "legal_codes_referenced": sorted({
                m.group(0) for m in re.finditer(
                    r"\b(?:[A-Z][A-Za-z]{1,8})\b", " ".join(out_refs)
                )
            }) if out_refs else [],
            "parties_roles": [],
        },
        "references": {
            "references_out_raw": out_refs,
            "references_out_laws": laws_out,
            "references_out_cases": cases_out,
            "references_out_count": len(out_refs),
            "references_in_count": in_count,
            "citation_edges": [],
        },
        "multilingual": {
            "language": language,
            "citation_aliases_de": [a for a in aliases if "BGE " in a],
            "citation_aliases_fr": [a for a in aliases if "ATF " in a or "consid" in a],
            "citation_aliases_it": [a for a in aliases if "DTF " in a],
        },
        "search": {
            "search_text_orig": " | ".join(filter(None, [
                parsed["case_id_canonical"],
                f"E. {consid_label}" if consid_label else None,
                text_clean,
            ])),
            "search_text_en": " | ".join(filter(None, [
                L.get("legal_area"), L.get("topic"), L.get("subtopic"),
                L.get("micro_topic"),
                ", ".join(L.get("concepts_en") or []),
                L.get("doctrinal_rule"), L.get("legal_test"),
            ])),
            "embedding_input": " | ".join(filter(None, [
                parsed["case_id_canonical"],
                f"E. {consid_label}" if consid_label else None,
                L.get("legal_area"),
                L.get("topic"),
                text_clean,
            ])),
        },
        "authority": {
            "in_citation_count": in_count,
            "authority_score": authority_score(parsed, in_count, specificity),
            "case_importance": case_importance,
            "is_bge": parsed["case_type"] == "bge",
            "is_leading_case": parsed["case_type"] == "bge" and in_count >= 5,
        },
        "embeddings": {
            "model": None,
            "file": None,
            "row_idx": None,
        },
        "metadata": {
            "source_dataset": "court_considerations.csv",
            "source_row_id": row_idx,
            "record_uid": record_uid,
            "language": language,
        },
    }
    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--court-csv",   type=Path, default=DEFAULT_COURT_CSV)
    ap.add_argument("--llm-desc",    type=Path, default=DEFAULT_LLM_DESC)
    ap.add_argument("--classified",  type=Path, default=DEFAULT_CLASSIFIED)
    ap.add_argument("--links",       type=Path, default=DEFAULT_LINKS_JSON)
    ap.add_argument("--out",         type=Path, default=DEFAULT_OUT)
    ap.add_argument("--manifest",    type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--limit",       type=int, default=0, help="Process only N rows (smoke test)")
    ap.add_argument("--progress-every", type=int, default=50_000)
    ap.add_argument("--no-llm", action="store_true", help="Skip loading LLM descriptors")
    ap.add_argument("--no-classified", action="store_true")
    ap.add_argument("--no-links", action="store_true")
    args = ap.parse_args()

    # CSV field-size guard (some court paragraphs are huge)
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print(f"[paths]\n  csv        = {args.court_csv}\n  llm        = {args.llm_desc}\n"
          f"  classified = {args.classified}\n  links      = {args.links}\n"
          f"  out        = {args.out}\n", file=sys.stderr)

    llm_idx = {} if args.no_llm else load_llm_descriptors(args.llm_desc)
    classified_idx = {} if args.no_classified else load_classified_citations(args.classified)
    if args.no_links:
        out_refs_map, in_count = {}, Counter()
    else:
        out_refs_map, in_count = load_links_and_indegree(args.links)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    t0 = time.time()

    case_type_counter: Counter = Counter()
    para_role_counter: Counter = Counter()
    lang_counter: Counter = Counter()

    KNOWN_TOTAL_ROWS = 2_476_315  # from data_insights/citation_patterns.json
    total = args.limit if args.limit else KNOWN_TOTAL_ROWS

    with args.court_csv.open("r", encoding="utf-8", newline="") as fin, \
         args.out.open("w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        if HAVE_TQDM:
            row_iter = tqdm(reader, total=total, desc="[build] rows",
                            unit="rec", mininterval=1.0, file=sys.stderr,
                            smoothing=0.05)
        else:
            row_iter = reader
        for row_idx, row in enumerate(row_iter):
            if args.limit and row_idx >= args.limit:
                break
            citation = (row.get("citation") or "").strip()
            text = row.get("text") or ""
            if not citation:
                continue
            llm = llm_idx.get(citation)
            classified = classified_idx.get(citation)
            out_refs = out_refs_map.get(citation, [])
            in_c = in_count.get(citation, 0)
            record = build_record(row_idx, citation, text, llm, classified, out_refs, in_c)
            fout.write(json.dumps(record, ensure_ascii=False))
            fout.write("\n")
            written += 1

            case_type_counter[record["case"]["case_type"]] += 1
            para_role_counter[record["semantic"]["paragraph_role"]] += 1
            lang_counter[record["case"]["language_original"]] += 1

            if HAVE_TQDM and written % 5000 == 0:
                row_iter.set_postfix({
                    "bge": case_type_counter.get("bge", 0),
                    "docket": case_type_counter.get("docket", 0),
                    "cant": case_type_counter.get("cantonal", 0),
                }, refresh=False)
            elif not HAVE_TQDM and written % args.progress_every == 0:
                rate = written / max(time.time() - t0, 1e-3)
                print(f"  [build] {written:>9}  {rate:.0f} rec/s  "
                      f"types={dict(case_type_counter)}", file=sys.stderr)

    manifest = {
        "output": str(args.out),
        "rows_written": written,
        "elapsed_seconds": round(time.time() - t0, 1),
        "case_type_counts": dict(case_type_counter),
        "paragraph_role_counts": dict(para_role_counter),
        "language_counts": dict(lang_counter),
        "sources": {
            "court_csv": str(args.court_csv),
            "llm_descriptors": str(args.llm_desc),
            "classified_citations": str(args.classified),
            "links_json": str(args.links),
        },
        "llm_index_size": len(llm_idx),
        "classified_index_size": len(classified_idx),
        "out_refs_index_size": len(out_refs_map),
        "schema": "court_case_knowledge_base.v1",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {written} records in {time.time()-t0:.1f}s -> {args.out}",
          file=sys.stderr)
    print(f"[done] manifest -> {args.manifest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
