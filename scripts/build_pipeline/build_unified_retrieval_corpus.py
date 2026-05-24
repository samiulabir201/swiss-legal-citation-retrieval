#!/usr/bin/env python
"""Build the unified law + court retrieval corpus.

Reads the merged unified cards produced by `merge_law_llm_into_v1_cards.py` and
`merge_llm_enrichment_into_v4_cards.py`:

  - artifacts/law_authority_cards_v2_unified.jsonl
  - artifacts/court_authority_cards_v5_unified.jsonl

Writes:

  - artifacts/unified_retrieval_documents.jsonl  (one JSON per doc, vector-DB ready)
  - artifacts/unified_retrieval.sqlite           (FTS5 BM25 + filter indexes + link tables)
  - artifacts/unified_retrieval_manifest.json    (counts + field weight recommendations)

The output combines both citation families behind a `family` column ("court" | "law")
so a single hybrid retriever can query laws and courts together. Per-family text
composition reuses the pre-built retrieval_views from the normalizers; this script
just glues them together, attaches filter columns, and computes an authority score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"

DEFAULT_COURT_INPUT = ART_DIR / "court_authority_cards_v5_unified.jsonl"
DEFAULT_LAW_INPUT = ART_DIR / "law_authority_cards_v2_unified.jsonl"
DEFAULT_JSONL_OUT = ART_DIR / "unified_retrieval_documents.jsonl"
DEFAULT_SQLITE_OUT = ART_DIR / "unified_retrieval.sqlite"
DEFAULT_MANIFEST_OUT = ART_DIR / "unified_retrieval_manifest.json"

BGE_BASE_RE = re.compile(r"\bBGE\s+\d{3}\s+[IVX]{1,4}\s+\d+[a-z]?\b")
DOCKET_BASE_RE = re.compile(r"\b\d{1,2}[A-Z]{1,4}[_\.]\d{1,5}/\d{4}\b")


# ---------------------------------------------------------------------------
# Generic helpers
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


def clipped(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


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


def weighted_join(parts: Iterable[str], repeats: int) -> str:
    base = [clean_text(x) for x in parts if clean_text(x)]
    return " ".join(base * max(1, repeats))


def labelled(label: str, value: Any) -> str:
    text = clean_text(value) if not isinstance(value, (list, tuple, set)) else field_join(*value)
    return f"{label}: {text}" if text else ""


def doc_id_for(family: str, line_no: int, citation: str, text: str) -> str:
    key = f"{family}|{line_no}|{citation}|{text[:200]}".encode("utf-8", errors="ignore")
    return f"{family}:" + hashlib.sha1(key).hexdigest()[:24]


def court_base_from_citation(citation: str) -> str:
    citation = clean_text(citation)
    bge = BGE_BASE_RE.search(citation)
    if bge:
        return bge.group(0)
    docket = DOCKET_BASE_RE.search(citation)
    if docket:
        return docket.group(0).replace(".", "_")
    if " E." in citation:
        return citation.split(" E.", 1)[0].strip()
    return citation


# ---------------------------------------------------------------------------
# Court family
# ---------------------------------------------------------------------------

COURT_BOOST_ROLES = {"holding", "reasoning", "legal_standard", "application"}
COURT_PENALTY_ROLES = {"costs", "procedural_history", "notification", "disposition"}


def court_authority_score(card: dict, rag: dict, views: dict) -> float:
    auth_view = clean_text(views.get("authority_view"))
    parts = [p.strip() for p in auth_view.split("|") if p.strip()]
    roles = {p for p in parts if not p.isdigit()}
    counts: list[int] = [int(p) for p in parts if p.isdigit()]
    src_count = counts[0] if counts else 0
    text_ref_count = counts[1] if len(counts) > 1 else 0
    paragraph_role = clean_text(rag.get("paragraph_role"))

    score = 1.0
    if "published_leading_decision" in roles:
        score += 1.5
    if "frequently_cited_authority" in roles:
        score += 1.0
    if "multi_consideration_decision" in roles:
        score += 0.2
    if paragraph_role in COURT_BOOST_ROLES:
        score += 0.35
    if paragraph_role in COURT_PENALTY_ROLES:
        score -= 0.45
    score += min(math.log1p(src_count) / 7.0, 0.8)
    score += min(math.log1p(text_ref_count) / 7.0, 0.8)
    return round(max(score, 0.05), 4)


def build_court_doc(line_no: int, card: dict, *, max_source_chars: int, max_vector_chars: int) -> dict:
    citation = clean_text(card.get("citation"))
    court_base = clean_text(card.get("court_base")) or court_base_from_citation(citation)
    language = clean_text(card.get("language"))
    legal_area = clean_text((card.get("rag_enrichment") or {}).get("legal_area")) \
        or clean_text(card.get("legal_area_static"))

    rag = card.get("rag_enrichment") or {}
    views = card.get("retrieval_views") or {}
    anchors = card.get("normalized_anchors") or {}

    paragraph_role = clean_text(rag.get("paragraph_role"))
    outcome_signal = clean_text(rag.get("outcome_signal")) or "none"
    specificity = float(rag.get("specificity_score") or 0.0)

    statute_anchors = unique_keep_order(rag.get("statute_anchors") or anchors.get("statute_anchors") or [], max_items=40)
    case_anchors = unique_keep_order(rag.get("case_anchors") or anchors.get("case_anchors") or [], max_items=40)
    concepts_en = unique_keep_order(
        (rag.get("english_legal_concepts") or []) + (rag.get("concepts_en") or []),
        max_items=40,
    )
    search_keywords = as_list(rag.get("search_keywords"))
    terms_original = as_list(rag.get("terms_original"))
    fact_pattern_tags = as_list(rag.get("fact_pattern_tags"))
    domain_path = as_list(rag.get("legal_domain_path"))
    authority_roles = [r for r in (clean_text(views.get("authority_view")).split("|"))
                       if r.strip() and not r.strip().isdigit()]
    authority_roles = unique_keep_order([r.strip() for r in authority_roles])

    raw_context = clipped(views.get("raw_context") or card.get("text_excerpt_original", ""), max_source_chars)

    citation_text = field_join(
        views.get("citation_view"),
        views.get("statute_anchor_view"),
        views.get("case_anchor_view"),
        court_base,
    )
    legal_text = field_join(
        legal_area,
        views.get("topic_path"),
        views.get("semantic_concepts_en"),
        views.get("authority_view"),
        views.get("procedural_view"),
        rag.get("topic"),
        rag.get("subtopic"),
        rag.get("micro_topic"),
        *domain_path,
        *fact_pattern_tags,
    )
    rag_text = field_join(
        rag.get("english_summary"),
        rag.get("legal_topic"),
        rag.get("legal_question"),
        views.get("legal_rule_view") or rag.get("legal_rule"),
        rag.get("court_holding"),
        rag.get("factual_context"),
        views.get("fact_pattern_view"),
        rag.get("doctrinal_rule"),
        rag.get("legal_test"),
        rag.get("procedural_context"),
    )
    keyword_text = field_join(
        *search_keywords,
        *terms_original,
        views.get("original_terms_view"),
        *concepts_en,
        *statute_anchors,
        *case_anchors,
        *fact_pattern_tags,
    )

    bm25_text = field_join(
        weighted_join([citation_text], 4),
        weighted_join([keyword_text], 3),
        weighted_join([rag_text], 2),
        legal_text,
        raw_context,
    )

    vector_text = field_join(
        labelled("Family", "court"),
        labelled("Citation", citation),
        labelled("Court base", court_base),
        labelled("Legal area", legal_area),
        labelled("Topic", rag.get("legal_topic") or rag.get("topic")),
        labelled("Question", rag.get("legal_question")),
        labelled("Rule", rag.get("legal_rule") or rag.get("doctrinal_rule")),
        labelled("Holding", rag.get("court_holding")),
        labelled("Facts", rag.get("factual_context")),
        labelled("Summary", rag.get("english_summary")),
        labelled("Concepts", concepts_en),
        labelled("Keywords", search_keywords + terms_original),
        labelled("Statutes", statute_anchors),
        labelled("Related cases", case_anchors[:10]),
        labelled("Role", paragraph_role),
        labelled("Outcome", outcome_signal if outcome_signal != "none" else ""),
    )
    vector_text = clipped(vector_text, max_vector_chars)

    quality_flags: list[str] = []
    enr_quality = card.get("enrichment_quality") or {}
    if not enr_quality.get("has_statute_anchor") and not statute_anchors:
        quality_flags.append("no_statute_anchor")
    if not enr_quality.get("has_case_anchor") and not case_anchors:
        quality_flags.append("no_case_anchor")
    if not rag.get("english_summary"):
        quality_flags.append("static_only")
    if enr_quality.get("low_value_paragraph"):
        quality_flags.append("low_value_paragraph")
    if paragraph_role in COURT_PENALTY_ROLES:
        quality_flags.append(f"role_{paragraph_role}")
    if len(vector_text.split()) < 20:
        quality_flags.append("thin_vector_text")

    score = court_authority_score(card, rag, views)

    bge_division = ""
    docket_prefix = ""
    decision_year = ""
    if BGE_BASE_RE.search(court_base):
        m = re.search(r"(\d{3})\s+([IVX]+)", court_base)
        if m:
            bge_division = m.group(2)
    docket_match = re.match(r"(\d{1,2}[A-Z]{1,4})_\d", court_base)
    if docket_match:
        docket_prefix = docket_match.group(1)
    year_match = re.search(r"/(\d{4})", court_base)
    if year_match:
        decision_year = year_match.group(1)

    filters = {
        "family": "court",
        "language": language,
        "legal_area": legal_area,
        "paragraph_role": paragraph_role,
        "outcome_signal": outcome_signal,
        "bge_division": bge_division,
        "docket_prefix": docket_prefix,
        "decision_year": decision_year,
        "authority_roles": authority_roles,
        "statutes": statute_anchors,
        "cases": case_anchors,
        "enrichment_source": clean_text(card.get("enrichment_source")),
        "specificity": specificity,
    }
    expansion = {
        "court_base": court_base,
        "source_citation": citation,
        "outgoing_court_citations": case_anchors,
        "outgoing_statute_citations": statute_anchors,
        "outgoing_court_bases": unique_keep_order(court_base_from_citation(x) for x in case_anchors),
    }

    raw_text = card.get("text_excerpt_original", "")
    doc_id = doc_id_for("court", line_no, citation, raw_text or raw_context)

    return {
        "doc_id": doc_id,
        "family": "court",
        "source_line": line_no,
        "citation": citation,
        "court_base": court_base,
        "law_code": "",
        "law_title": "",
        "retrieval": {
            "vector_text": vector_text,
            "bm25_text": bm25_text,
            "citation_text": citation_text,
            "legal_text": legal_text,
            "rag_text": rag_text,
            "keyword_text": keyword_text,
            "source_text": raw_context,
        },
        "filters": filters,
        "expansion": expansion,
        "ranking": {
            "authority_score": score,
            "specificity_score": specificity,
        },
        "rag_enrichment": rag,
        "quality_flags": quality_flags,
    }


# ---------------------------------------------------------------------------
# Law family
# ---------------------------------------------------------------------------

LAW_BOOST_ROLES = {"duty", "right_or_entitlement", "prohibition", "principle", "scope", "competence", "definition"}
LAW_PENALTY_ROLES = {"transitional_or_commencement", "fees_or_costs", "data_reporting"}


def law_authority_score(card: dict, rag: dict) -> float:
    role = clean_text(rag.get("provision_role_llm")) or clean_text(rag.get("paragraph_role"))
    anchors = card.get("normalized_anchors") or {}
    incoming = int(anchors.get("incoming_reference_count") or 0)
    outgoing = int(anchors.get("outgoing_reference_count") or 0)
    specificity = float(rag.get("specificity_score") or 0.0)

    score = 1.0
    if role in LAW_BOOST_ROLES:
        score += 0.45
    if role in LAW_PENALTY_ROLES:
        score -= 0.55
    if rag.get("legal_rule"):
        score += 0.3
    if rag.get("applicability_conditions"):
        score += 0.2
    if rag.get("english_summary"):
        score += 0.15
    score += min(math.log1p(incoming) / 5.0, 0.8)
    score += min(math.log1p(outgoing) / 8.0, 0.3)
    score += 0.3 * specificity
    quality = card.get("enrichment_quality") or {}
    if quality.get("boilerplate_role") or quality.get("rule_fields_suppressed"):
        score -= 0.4
    return round(max(score, 0.05), 4)


def build_law_doc(line_no: int, card: dict, *, max_source_chars: int, max_vector_chars: int) -> dict:
    citation = clean_text(card.get("citation"))
    language = clean_text(card.get("language")) or "de"
    law_title = clean_text(card.get("law_title"))
    structural = card.get("structural") or {}
    title_meta = card.get("title_metadata") or {}
    law_code = clean_text(structural.get("law_code"))
    article = clean_text(structural.get("article"))
    granularity = clean_text(structural.get("granularity"))
    enactment_date = clean_text(title_meta.get("enactment_date"))
    source_type = clean_text(title_meta.get("source_type"))
    sc_sector = clean_text(title_meta.get("systematic_collection_sector"))
    law_aliases = as_list(title_meta.get("law_aliases"))

    rag = card.get("rag_enrichment") or {}
    views = card.get("retrieval_views") or {}
    anchors = card.get("normalized_anchors") or {}

    legal_area = clean_text(rag.get("legal_area"))
    primary_domain = clean_text(rag.get("primary_domain"))
    secondary_domain = clean_text(rag.get("secondary_domain"))
    domain_path = as_list(rag.get("legal_domain_path"))
    provision_role = clean_text(rag.get("provision_role_llm")) or clean_text(rag.get("paragraph_role"))
    specificity = float(rag.get("specificity_score") or 0.0)

    concepts_en = as_list(rag.get("concepts_en"))
    addressees = as_list(rag.get("addressees"))
    applicability = as_list(rag.get("applicability_conditions"))
    exceptions = as_list(rag.get("exceptions_or_limitations"))
    sanctions = as_list(rag.get("sanctions_or_consequences"))
    fact_tags = as_list(rag.get("fact_pattern_tags"))
    defined_terms = rag.get("defined_terms") or []
    defined_term_strs = [
        f"{clean_text(t.get('term'))}: {clean_text(t.get('definition'))}"
        for t in defined_terms
        if isinstance(t, dict) and clean_text(t.get("term"))
    ]
    bilingual_terms = rag.get("terms_de_to_en") or []
    bilingual_pairs_de = [clean_text(t.get("de")) for t in bilingual_terms if isinstance(t, dict)]
    bilingual_pairs_en = [clean_text(t.get("en")) for t in bilingual_terms if isinstance(t, dict)]
    terms_original = as_list(rag.get("terms_original"))

    statute_anchors = unique_keep_order(anchors.get("statute_anchors") or [], max_items=40)
    case_anchors = unique_keep_order(anchors.get("court_case_anchors") or [], max_items=40)
    other_refs = unique_keep_order(anchors.get("other_reference_anchors") or [], max_items=20)
    adjacent = anchors.get("adjacent_citations") or {}
    adj_next = clean_text(adjacent.get("next_in_law"))
    adj_prev = clean_text(adjacent.get("previous_in_law"))
    adj_siblings = as_list(adjacent.get("same_article_siblings"))

    raw_context = clipped(views.get("raw_context") or "", max_source_chars)

    citation_text = field_join(
        views.get("citation_view") or citation,
        law_code,
        *law_aliases,
        views.get("statute_anchor_view"),
        adj_next,
        adj_prev,
    )
    legal_text = field_join(
        legal_area,
        primary_domain,
        secondary_domain,
        *domain_path,
        provision_role,
        source_type,
        sc_sector,
        granularity,
        views.get("structure_view"),
        views.get("title_view"),
        *fact_tags,
    )
    rag_text = field_join(
        rag.get("english_summary"),
        rag.get("legal_question"),
        views.get("legal_rule_view") or rag.get("legal_rule"),
        rag.get("doctrinal_rule"),
        rag.get("legal_test"),
        views.get("rule_components_view"),
        " applicability: " + " | ".join(applicability) if applicability else "",
        " exceptions: " + " | ".join(exceptions) if exceptions else "",
        " sanctions: " + " | ".join(sanctions) if sanctions else "",
    )
    keyword_text = field_join(
        *concepts_en,
        *addressees,
        *terms_original,
        *bilingual_pairs_de,
        *bilingual_pairs_en,
        views.get("terms_bilingual_view"),
        views.get("original_terms_view"),
        *defined_term_strs,
        law_title,
    )

    bm25_text = field_join(
        weighted_join([citation_text], 4),
        weighted_join([keyword_text], 3),
        weighted_join([rag_text], 2),
        legal_text,
        raw_context,
    )

    vector_text = field_join(
        labelled("Family", "law"),
        labelled("Citation", citation),
        labelled("Law", law_title),
        labelled("Code", law_code),
        labelled("Article", article),
        labelled("Legal area", legal_area),
        labelled("Domain", " > ".join([d for d in domain_path if d]) if domain_path else field_join(primary_domain, secondary_domain)),
        labelled("Role", provision_role),
        labelled("Question", rag.get("legal_question")),
        labelled("Rule", rag.get("legal_rule") or rag.get("doctrinal_rule")),
        labelled("Summary", rag.get("english_summary")),
        labelled("Applies when", applicability),
        labelled("Exceptions", exceptions),
        labelled("Addressees", addressees),
        labelled("Sanctions", sanctions),
        labelled("Concepts", concepts_en),
        labelled("Keywords", terms_original + bilingual_pairs_de + bilingual_pairs_en),
        labelled("Defined terms", defined_term_strs),
        labelled("Statutes referenced", statute_anchors),
        labelled("Adjacent", [adj_prev, adj_next]),
    )
    vector_text = clipped(vector_text, max_vector_chars)

    quality_flags: list[str] = []
    enr_quality = card.get("enrichment_quality") or {}
    if enr_quality.get("boilerplate_role"):
        quality_flags.append("boilerplate")
    if enr_quality.get("rule_fields_suppressed"):
        quality_flags.append("rule_suppressed")
    if not rag.get("english_summary"):
        quality_flags.append("static_only")
    if not statute_anchors and not case_anchors:
        quality_flags.append("no_anchors")
    if len(vector_text.split()) < 20:
        quality_flags.append("thin_vector_text")
    if provision_role in LAW_PENALTY_ROLES:
        quality_flags.append(f"role_{provision_role}")

    score = law_authority_score(card, rag)

    filters = {
        "family": "law",
        "language": language,
        "legal_area": legal_area,
        "primary_domain": primary_domain,
        "secondary_domain": secondary_domain,
        "provision_role": provision_role,
        "law_code": law_code,
        "law_aliases": law_aliases,
        "article": article,
        "granularity": granularity,
        "enactment_date": enactment_date,
        "source_type": source_type,
        "systematic_collection_sector": sc_sector,
        "addressees": addressees,
        "statutes": statute_anchors,
        "cases": case_anchors,
        "enrichment_source": clean_text(card.get("enrichment_source")),
        "specificity": specificity,
    }
    expansion = {
        "law_code": law_code,
        "law_title": law_title,
        "source_citation": citation,
        "adjacent_next": adj_next,
        "adjacent_previous": adj_prev,
        "adjacent_siblings": adj_siblings,
        "outgoing_statute_citations": statute_anchors,
        "outgoing_court_citations": case_anchors,
        "other_reference_anchors": other_refs,
    }

    doc_id = doc_id_for("law", line_no, citation, raw_context)

    return {
        "doc_id": doc_id,
        "family": "law",
        "source_line": line_no,
        "citation": citation,
        "court_base": "",
        "law_code": law_code,
        "law_title": law_title,
        "retrieval": {
            "vector_text": vector_text,
            "bm25_text": bm25_text,
            "citation_text": citation_text,
            "legal_text": legal_text,
            "rag_text": rag_text,
            "keyword_text": keyword_text,
            "source_text": raw_context,
        },
        "filters": filters,
        "expansion": expansion,
        "ranking": {
            "authority_score": score,
            "specificity_score": specificity,
        },
        "rag_enrichment": rag,
        "quality_flags": quality_flags,
    }


# ---------------------------------------------------------------------------
# SQLite layer
# ---------------------------------------------------------------------------

def open_sqlite(path: Path, force: bool) -> tuple[sqlite3.Connection, bool]:
    if path.exists():
        if not force:
            raise FileExistsError(f"{path} already exists; pass --force to overwrite")
        path.unlink()
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-400000")
    con.executescript(
        """
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            family TEXT NOT NULL,
            source_line INTEGER NOT NULL,
            citation TEXT NOT NULL,
            court_base TEXT,
            law_code TEXT,
            law_title TEXT,
            language TEXT,
            legal_area TEXT,
            primary_domain TEXT,
            secondary_domain TEXT,
            paragraph_role TEXT,
            provision_role TEXT,
            outcome_signal TEXT,
            bge_division TEXT,
            docket_prefix TEXT,
            decision_year TEXT,
            article TEXT,
            granularity TEXT,
            enactment_date TEXT,
            enrichment_source TEXT,
            authority_score REAL NOT NULL,
            specificity_score REAL,
            vector_text TEXT NOT NULL,
            bm25_text TEXT NOT NULL,
            source_text TEXT,
            rag_json TEXT NOT NULL,
            filters_json TEXT NOT NULL,
            expansion_json TEXT NOT NULL,
            quality_flags_json TEXT NOT NULL
        );

        CREATE TABLE statute_links (
            doc_id TEXT NOT NULL,
            family TEXT NOT NULL,
            citation TEXT NOT NULL,
            statute TEXT NOT NULL
        );

        CREATE TABLE case_links (
            doc_id TEXT NOT NULL,
            family TEXT NOT NULL,
            citation TEXT NOT NULL,
            target_citation TEXT NOT NULL,
            target_base TEXT NOT NULL
        );

        CREATE TABLE adjacent_law_links (
            doc_id TEXT NOT NULL,
            citation TEXT NOT NULL,
            neighbor_citation TEXT NOT NULL,
            kind TEXT NOT NULL
        );
        """
    )
    fts_enabled = True
    try:
        con.execute(
            """
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                doc_id UNINDEXED,
                family UNINDEXED,
                citation_text,
                legal_text,
                rag_text,
                keyword_text,
                source_text,
                tokenize = 'unicode61 remove_diacritics 2'
            )
            """
        )
    except sqlite3.OperationalError as exc:
        print(f"[warn] FTS5 unavailable: {exc}", file=sys.stderr)
        fts_enabled = False
    con.commit()
    return con, fts_enabled


def insert_batch(con: sqlite3.Connection, docs: list[dict], *, fts_enabled: bool) -> None:
    doc_rows = []
    fts_rows = []
    statute_rows = []
    case_rows = []
    adj_rows = []
    for d in docs:
        f = d["filters"]
        e = d["expansion"]
        r = d["retrieval"]
        doc_rows.append((
            d["doc_id"],
            d["family"],
            d["source_line"],
            d["citation"],
            d.get("court_base", ""),
            d.get("law_code", ""),
            d.get("law_title", ""),
            f.get("language", ""),
            f.get("legal_area", ""),
            f.get("primary_domain", ""),
            f.get("secondary_domain", ""),
            f.get("paragraph_role", ""),
            f.get("provision_role", ""),
            f.get("outcome_signal", ""),
            f.get("bge_division", ""),
            f.get("docket_prefix", ""),
            f.get("decision_year", ""),
            f.get("article", ""),
            f.get("granularity", ""),
            f.get("enactment_date", ""),
            f.get("enrichment_source", ""),
            d["ranking"]["authority_score"],
            d["ranking"].get("specificity_score", 0.0),
            r["vector_text"],
            r["bm25_text"],
            r["source_text"],
            json.dumps(d["rag_enrichment"], ensure_ascii=False),
            json.dumps(f, ensure_ascii=False),
            json.dumps(e, ensure_ascii=False),
            json.dumps(d["quality_flags"], ensure_ascii=False),
        ))
        if fts_enabled:
            fts_rows.append((
                d["doc_id"], d["family"],
                r["citation_text"], r["legal_text"], r["rag_text"],
                r["keyword_text"], r["source_text"],
            ))
        for s in f.get("statutes", []) or []:
            statute_rows.append((d["doc_id"], d["family"], d["citation"], s))
        for tgt in e.get("outgoing_court_citations", []) or []:
            case_rows.append((d["doc_id"], d["family"], d["citation"], tgt, court_base_from_citation(tgt)))
        if d["family"] == "law":
            if e.get("adjacent_next"):
                adj_rows.append((d["doc_id"], d["citation"], e["adjacent_next"], "next_in_law"))
            if e.get("adjacent_previous"):
                adj_rows.append((d["doc_id"], d["citation"], e["adjacent_previous"], "previous_in_law"))

    con.executemany(
        """
        INSERT INTO documents (
            doc_id, family, source_line, citation, court_base, law_code, law_title,
            language, legal_area, primary_domain, secondary_domain,
            paragraph_role, provision_role, outcome_signal,
            bge_division, docket_prefix, decision_year,
            article, granularity, enactment_date, enrichment_source,
            authority_score, specificity_score,
            vector_text, bm25_text, source_text,
            rag_json, filters_json, expansion_json, quality_flags_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        doc_rows,
    )
    if fts_enabled and fts_rows:
        con.executemany(
            "INSERT INTO documents_fts (doc_id, family, citation_text, legal_text, rag_text, keyword_text, source_text) VALUES (?, ?, ?, ?, ?, ?, ?)",
            fts_rows,
        )
    if statute_rows:
        con.executemany(
            "INSERT INTO statute_links (doc_id, family, citation, statute) VALUES (?, ?, ?, ?)",
            statute_rows,
        )
    if case_rows:
        con.executemany(
            "INSERT INTO case_links (doc_id, family, citation, target_citation, target_base) VALUES (?, ?, ?, ?, ?)",
            case_rows,
        )
    if adj_rows:
        con.executemany(
            "INSERT INTO adjacent_law_links (doc_id, citation, neighbor_citation, kind) VALUES (?, ?, ?, ?)",
            adj_rows,
        )


def build_indexes(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE INDEX idx_doc_family ON documents(family);
        CREATE INDEX idx_doc_citation ON documents(citation);
        CREATE INDEX idx_doc_court_base ON documents(court_base);
        CREATE INDEX idx_doc_law_code ON documents(law_code);
        CREATE INDEX idx_doc_legal_area ON documents(legal_area);
        CREATE INDEX idx_doc_role ON documents(paragraph_role);
        CREATE INDEX idx_doc_provision ON documents(provision_role);
        CREATE INDEX idx_doc_year ON documents(decision_year);
        CREATE INDEX idx_doc_docket ON documents(docket_prefix);
        CREATE INDEX idx_doc_bge_division ON documents(bge_division);
        CREATE INDEX idx_doc_authority ON documents(authority_score);
        CREATE INDEX idx_statute_links_statute ON statute_links(statute);
        CREATE INDEX idx_statute_links_doc ON statute_links(doc_id);
        CREATE INDEX idx_case_links_target ON case_links(target_citation);
        CREATE INDEX idx_case_links_target_base ON case_links(target_base);
        CREATE INDEX idx_case_links_doc ON case_links(doc_id);
        CREATE INDEX idx_adj_neighbor ON adjacent_law_links(neighbor_citation);
        CREATE INDEX idx_adj_doc ON adjacent_law_links(doc_id);
        """
    )
    con.commit()


# ---------------------------------------------------------------------------
# Streaming pipeline
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield line_no, row


def process_family(
    name: str,
    path: Path,
    *,
    builder,
    out_jsonl,
    con: sqlite3.Connection | None,
    fts_enabled: bool,
    batch_size: int,
    limit: int,
    max_source_chars: int,
    max_vector_chars: int,
    progress_every: int,
    counters: dict,
) -> None:
    pending: list[dict] = []
    t0 = time.time()
    n = 0
    for line_no, card in iter_jsonl(path):
        if limit and n >= limit:
            break
        n += 1
        try:
            doc = builder(line_no, card,
                         max_source_chars=max_source_chars,
                         max_vector_chars=max_vector_chars)
        except Exception as exc:
            counters[f"{name}_build_error"] += 1
            if counters[f"{name}_build_error"] <= 5:
                print(f"[warn] {name} build error at line {line_no}: {exc}", file=sys.stderr)
            continue

        out_jsonl.write(json.dumps(doc, ensure_ascii=False) + "\n")
        pending.append(doc)
        counters[f"{name}_documents"] += 1
        for flag in doc["quality_flags"]:
            counters[f"{name}_flag_{flag}"] += 1

        if con is not None and len(pending) >= batch_size:
            insert_batch(con, pending, fts_enabled=fts_enabled)
            con.commit()
            pending.clear()

        if progress_every and n % progress_every == 0:
            rate = n / max(time.time() - t0, 1e-9)
            print(f"  [{name}] processed={n:,}  ({rate:>5.0f} rec/s)", flush=True)

    if con is not None and pending:
        insert_batch(con, pending, fts_enabled=fts_enabled)
        con.commit()
    elapsed = time.time() - t0
    print(f"[{name}] done: {n:,} rows in {elapsed/60:.1f} min "
          f"({n/max(elapsed,1e-9):.0f} rec/s)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--court", type=Path, default=DEFAULT_COURT_INPUT,
                    help="Court v5 unified JSONL. Pass empty string to skip courts.")
    ap.add_argument("--law", type=Path, default=DEFAULT_LAW_INPUT,
                    help="Law v2 unified JSONL. Pass empty string to skip laws.")
    ap.add_argument("--output-jsonl", type=Path, default=DEFAULT_JSONL_OUT)
    ap.add_argument("--output-sqlite", type=Path, default=DEFAULT_SQLITE_OUT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_OUT)
    ap.add_argument("--limit-court", type=int, default=0)
    ap.add_argument("--limit-law", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=5000)
    ap.add_argument("--max-source-chars", type=int, default=2500)
    ap.add_argument("--max-vector-chars", type=int, default=4500)
    ap.add_argument("--no-sqlite", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--progress-every", type=int, default=50000)
    args = ap.parse_args()

    court_path = args.court if str(args.court) else None
    law_path = args.law if str(args.law) else None
    if court_path is not None and not court_path.exists():
        print(f"[warn] court input not found, skipping: {court_path}", file=sys.stderr)
        court_path = None
    if law_path is not None and not law_path.exists():
        print(f"[warn] law input not found, skipping: {law_path}", file=sys.stderr)
        law_path = None
    if court_path is None and law_path is None:
        print("ERROR: no inputs available", file=sys.stderr)
        return 2

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists() and not args.force:
        raise FileExistsError(f"{args.output_jsonl} exists; pass --force")

    con: sqlite3.Connection | None = None
    fts_enabled = False
    if not args.no_sqlite:
        con, fts_enabled = open_sqlite(args.output_sqlite, args.force)

    counters: Counter = Counter()

    print(f"[start] court={court_path} law={law_path} sqlite={'on' if con else 'off'} fts5={'on' if fts_enabled else 'off'}", flush=True)
    with args.output_jsonl.open("w", encoding="utf-8") as out_jsonl:
        if law_path is not None:
            process_family(
                "law", law_path,
                builder=build_law_doc,
                out_jsonl=out_jsonl, con=con, fts_enabled=fts_enabled,
                batch_size=args.batch_size, limit=args.limit_law,
                max_source_chars=args.max_source_chars,
                max_vector_chars=args.max_vector_chars,
                progress_every=args.progress_every,
                counters=counters,
            )
        if court_path is not None:
            process_family(
                "court", court_path,
                builder=build_court_doc,
                out_jsonl=out_jsonl, con=con, fts_enabled=fts_enabled,
                batch_size=args.batch_size, limit=args.limit_court,
                max_source_chars=args.max_source_chars,
                max_vector_chars=args.max_vector_chars,
                progress_every=args.progress_every,
                counters=counters,
            )

    if con is not None:
        print("[sqlite] creating indexes...", flush=True)
        build_indexes(con)
        print("[sqlite] VACUUM...", flush=True)
        con.execute("VACUUM")
        con.close()

    manifest = {
        "builder": "build_unified_retrieval_corpus_v1",
        "court_input": str(court_path) if court_path else None,
        "law_input": str(law_path) if law_path else None,
        "output_jsonl": str(args.output_jsonl),
        "output_sqlite": None if args.no_sqlite else str(args.output_sqlite),
        "fts_enabled": fts_enabled,
        "limits": {"court": args.limit_court, "law": args.limit_law},
        "max_source_chars": args.max_source_chars,
        "max_vector_chars": args.max_vector_chars,
        "stats": dict(counters),
        "retrieval_fields": {
            "vector_text": [
                "rag_enrichment.english_summary",
                "rag_enrichment.legal_question",
                "rag_enrichment.legal_rule / doctrinal_rule",
                "rag_enrichment.court_holding (court)",
                "rag_enrichment.applicability_conditions / exceptions / sanctions / addressees (law)",
                "concepts_en, search_keywords, terms",
                "statute_anchors, case_anchors, adjacent_law (law)",
                "labelled English schema (Citation/Court base/Law/Code/Legal area/...)",
            ],
            "bm25_columns": {
                "citation_text": "exact citation, court base, statutes, law codes",
                "legal_text": "legal area, domain path, role, topic, authority",
                "rag_text": "LLM English summary/question/rule/holding/facts/applicability",
                "keyword_text": "search keywords, multilingual terms, concepts, addressees",
                "source_text": "original German/French/Italian paragraph",
            },
            "recommended_bm25_column_weights": {
                "citation_text": 4.0,
                "keyword_text": 3.0,
                "rag_text": 2.5,
                "legal_text": 2.0,
                "source_text": 0.7,
            },
        },
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] jsonl={args.output_jsonl}", flush=True)
    if not args.no_sqlite:
        print(f"[done] sqlite={args.output_sqlite}", flush=True)
    print(f"[done] manifest={args.manifest}", flush=True)
    print(f"[done] documents (law/court): {counters.get('law_documents',0):,} / {counters.get('court_documents',0):,}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
