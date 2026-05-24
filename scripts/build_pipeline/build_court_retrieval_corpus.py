#!/usr/bin/env python
"""Build the final court retrieval corpus for hybrid search.

Input:
  - court_authority_cards_rag.jsonl, produced by LLM enrichment, or
  - court_authority_cards_v4.jsonl as a deterministic fallback.

Output:
  - JSONL documents ready for vector DB ingestion.
  - Optional SQLite database with fielded FTS5 BM25 index, metadata tables,
    statute/case links, and citation graph edges for expansion.

The script does not read train/val/test labels and does not hardcode validation
queries. It only uses court cards, their optional rag_enrichment, and the
extracted citation graph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"
DATA_INSIGHTS_DIR = ROOT / "data_insights"

DEFAULT_RAG_INPUT = ART_DIR / "court_authority_cards_rag.jsonl"
DEFAULT_V4_INPUT = ART_DIR / "court_authority_cards_v4.jsonl"
DEFAULT_JSONL_OUT = ART_DIR / "court_retrieval_documents.jsonl"
DEFAULT_SQLITE_OUT = ART_DIR / "court_retrieval.sqlite"
DEFAULT_MANIFEST_OUT = ART_DIR / "court_retrieval_manifest.json"
DEFAULT_GRAPH_DB = DATA_INSIGHTS_DIR / "citation_graph_extracted.sqlite"


BGE_BASE_RE = re.compile(r"\bBGE\s+\d{3}\s+[IVX]{1,4}\s+\d+[a-z]?\b")
DOCKET_BASE_RE = re.compile(r"\b\d{1,2}[A-Z]{1,4}[_\.]\d{1,5}/\d{4}\b")
LAW_ARTICLE_RE = re.compile(r"\bArt\.\s*\d+[A-Za-z0-9.]*")

RAG_TEXT_FIELDS = (
    "english_summary",
    "legal_topic",
    "legal_question",
    "legal_rule",
    "court_holding",
    "factual_context",
)
RAG_ARRAY_FIELDS = (
    "english_legal_concepts",
    "search_keywords",
    "natural_language_queries",
)
RAG_REQUIRED_FIELDS = (
    "english_summary",
    "legal_topic",
    "english_legal_concepts",
    "search_keywords",
    "natural_language_queries",
    "paragraph_role",
    "outcome_signal",
)


# Keep this as a broad production filter, not a val-query list. It removes
# parser artifacts while preserving Swiss German/French/Italian law-code aliases.
KNOWN_LAW_CODES = {
    "AIG", "AI", "ALC", "AMLA", "ATSG", "AVIG", "AHVG", "AsylG", "AuG",
    "BankG", "BetmG", "BGFA", "BGG", "BVG", "BV", "CC", "CEDH", "CO",
    "CPC", "CP", "CPP", "Cst", "DBG", "DSG", "EMRK", "FINMAG", "IVG",
    "IPRG", "KG", "KVG", "LAI", "LAMal", "LAVS", "LEI", "LEtr", "LIFD",
    "LP", "LPGA", "LTF", "LPP", "LStup", "LAsi", "LAA", "MSchG", "MWSTG",
    "NHG", "OJ", "OG", "OR", "PatG", "RPG", "SchKG", "StGB", "StPO",
    "SVG", "UVG", "URG", "UWG", "VwVG", "ZGB", "ZPO",
}
LAW_CODE_STOPWORDS = {
    "Art", "Abs", "Bst", "Lit", "Ziff", "Satz", "Je", "Wesentlich",
    "Ainsi", "Cette", "Dans", "Eine", "Dabei", "Zudem", "Falls", "Wenn",
    "Cette", "Selon", "Pour", "Toutefois", "Questa", "Inoltre",
}


def clean_text(value: Any) -> str:
    """Normalize whitespace while preserving the original language content."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.replace("\u0000", " ").split())


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
    if isinstance(value, list):
        return unique_keep_order(value)
    if isinstance(value, tuple):
        return unique_keep_order(value)
    if isinstance(value, set):
        return unique_keep_order(sorted(value))
    text = clean_text(value)
    return [text] if text else []


def flatten_term_map(value: Any) -> tuple[list[str], list[str]]:
    """Return (concept labels, multilingual terms) from matched_terms_multilingual."""
    if not isinstance(value, dict):
        return [], []
    concepts: list[str] = []
    terms: list[str] = []
    for concept, raw_terms in value.items():
        concepts.append(clean_text(concept))
        terms.extend(as_list(raw_terms))
    return unique_keep_order(concepts), unique_keep_order(terms)


def filter_law_codes(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for raw in values:
        code = clean_text(raw).strip(".,;:()[]{}")
        if not code or code in LAW_CODE_STOPWORDS:
            continue
        if code in KNOWN_LAW_CODES:
            out.append(code)
            continue
        # Accept unknown all-uppercase law abbreviations, but reject sentence words.
        if re.fullmatch(r"[A-Z][A-Z0-9]{1,10}", code):
            out.append(code)
    return unique_keep_order(out)


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


def citation_kind(citation: str) -> str:
    citation = clean_text(citation)
    if BGE_BASE_RE.search(citation) or DOCKET_BASE_RE.search(citation):
        return "court"
    if citation.startswith("Art.") or LAW_ARTICLE_RE.search(citation):
        return "statute"
    return "other"


def doc_id_for(line_no: int, citation: str, text: str) -> str:
    key = f"{line_no}\0{citation}\0{text[:200]}".encode("utf-8", errors="ignore")
    return "court:" + hashlib.sha1(key).hexdigest()[:24]


def clipped(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0]


def field_join(*parts: Any) -> str:
    strings: list[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, (list, tuple, set)):
            strings.extend(clean_text(x) for x in part)
        else:
            strings.append(clean_text(part))
    return " | ".join(x for x in strings if x)


def weighted_join(parts: Iterable[str], repeats: int) -> str:
    base = [clean_text(x) for x in parts if clean_text(x)]
    return " ".join(base * max(1, repeats))


def labelled(label: str, value: Any) -> str:
    text = clean_text(value)
    return f"{label}: {text}" if text else ""


def rag_quality(rag: dict[str, Any]) -> dict[str, Any]:
    if not rag:
        return {
            "has_rag": False,
            "missing_fields": list(RAG_REQUIRED_FIELDS),
            "method": "",
        }
    missing = [field for field in RAG_REQUIRED_FIELDS if field not in rag]
    return {
        "has_rag": True,
        "missing_fields": missing,
        "method": clean_text(rag.get("method")),
    }


def authority_score(card: dict[str, Any], rag: dict[str, Any]) -> float:
    roles = set(as_list(card.get("authority_role")))
    structural = card.get("structural") if isinstance(card.get("structural"), dict) else {}
    source_count = int(structural.get("court_base_source_count") or 0)
    text_ref_count = int(structural.get("court_base_text_ref_count") or 0)
    paragraph_role = clean_text(rag.get("paragraph_role")) if rag else ""

    score = 1.0
    if "published_leading_decision" in roles:
        score += 1.5
    if "frequently_cited_authority" in roles:
        score += 1.0
    if "multi_consideration_decision" in roles:
        score += 0.2
    if paragraph_role in {"holding", "reasoning", "standard_of_review"}:
        score += 0.35
    if paragraph_role in {"cost", "procedural", "disposition"}:
        score -= 0.45
    if card.get("is_notification_paragraph"):
        score -= 0.75
    score += min(math.log1p(source_count) / 7.0, 0.8)
    score += min(math.log1p(text_ref_count) / 7.0, 0.8)
    return round(max(score, 0.05), 4)


def build_document(
    line_no: int,
    card: dict[str, Any],
    *,
    max_source_chars: int,
    max_vector_chars: int,
) -> dict[str, Any]:
    citation = clean_text(card.get("citation"))
    source_text = clipped(card.get("text_excerpt_original", ""), max_source_chars)
    rag = card.get("rag_enrichment") if isinstance(card.get("rag_enrichment"), dict) else {}
    structural = card.get("structural") if isinstance(card.get("structural"), dict) else {}
    matched_concepts, matched_terms = flatten_term_map(card.get("matched_terms_multilingual"))

    issue_labels = as_list(card.get("issue_labels_en"))
    law_codes = filter_law_codes(card.get("law_codes") or [])
    statutes = unique_keep_order(card.get("statutes_cited") or [], max_items=30)
    outgoing_court_cases = unique_keep_order(card.get("court_cases_cited") or [], max_items=30)
    authority_roles = as_list(card.get("authority_role"))
    rag_concepts = unique_keep_order(
        [item for field in RAG_ARRAY_FIELDS for item in as_list(rag.get(field))],
        max_items=40,
    )
    rag_text_parts = [rag.get(field) for field in RAG_TEXT_FIELDS]
    natural_queries = as_list(rag.get("natural_language_queries"))
    search_keywords = as_list(rag.get("search_keywords"))
    english_concepts = as_list(rag.get("english_legal_concepts"))

    court_base = clean_text(card.get("court_base")) or court_base_from_citation(citation)
    legal_area = clean_text(card.get("legal_area"))
    paragraph_role = clean_text(rag.get("paragraph_role"))
    outcome_signal = clean_text(rag.get("outcome_signal"))
    language = clean_text(card.get("language"))

    citation_text = field_join(
        citation,
        court_base,
        *outgoing_court_cases,
        *statutes,
        *law_codes,
    )
    legal_text = field_join(
        legal_area,
        *issue_labels,
        *matched_concepts,
        *english_concepts,
        *authority_roles,
        paragraph_role,
        outcome_signal,
    )
    rag_text = field_join(*rag_text_parts, *natural_queries)
    keyword_text = field_join(
        *search_keywords,
        *matched_terms,
        *issue_labels,
        *law_codes,
        *statutes,
        *outgoing_court_cases,
    )

    # BM25 text intentionally preserves exact citations/statutes and source-language
    # terms. Repetition gives simple field weighting for search engines that only
    # accept a single text field.
    bm25_text = field_join(
        weighted_join([citation_text], 4),
        weighted_join([keyword_text], 3),
        weighted_join([legal_text], 2),
        rag_text,
        card.get("summary_en_proxy"),
        source_text,
    )

    # Vector text is mostly concise English legal meaning. It still includes exact
    # law/citation anchors, but avoids letting raw German/French/Italian dominate.
    vector_text = field_join(
        labelled("Citation", citation),
        labelled("Court base", court_base),
        labelled("Legal area", legal_area),
        labelled("Topic", rag.get("legal_topic")),
        labelled("Question", rag.get("legal_question")),
        labelled("Rule", rag.get("legal_rule")),
        labelled("Holding", rag.get("court_holding")),
        labelled("Facts", rag.get("factual_context")),
        labelled("Summary", rag.get("english_summary")),
        labelled("Concepts", field_join(*english_concepts, *issue_labels, *matched_concepts)),
        labelled("Keywords", field_join(*search_keywords, *matched_terms)),
        labelled("Natural queries", field_join(*natural_queries)),
        labelled("Statutes", field_join(*statutes, *law_codes)),
        labelled("Related cases", field_join(*outgoing_court_cases[:10])),
        card.get("summary_en_proxy") if not rag else "",
    )
    vector_text = clipped(vector_text, max_vector_chars)

    quality_flags = []
    if not rag:
        quality_flags.append("missing_rag_enrichment")
    elif rag_quality(rag)["missing_fields"]:
        quality_flags.append("rag_missing_required_fields")
    if len(issue_labels) == 0:
        quality_flags.append("no_deterministic_issue_labels")
    if len(vector_text.split()) < 20:
        quality_flags.append("thin_vector_text")
    if card.get("is_notification_paragraph"):
        quality_flags.append("notification_paragraph")
    if paragraph_role in {"cost", "procedural", "disposition"}:
        quality_flags.append(f"role_{paragraph_role}")

    doc_id = doc_id_for(line_no, citation, source_text)
    score = authority_score(card, rag)

    filters = {
        "family": clean_text(card.get("family")),
        "pattern": clean_text(card.get("pattern")),
        "subfamily": clean_text(card.get("subfamily")),
        "language": language,
        "legal_area": legal_area,
        "legal_area_code": clean_text(structural.get("legal_area_code")),
        "bge_division": clean_text(structural.get("bge_division")),
        "docket_prefix": clean_text(structural.get("docket_prefix")),
        "court_chamber": clean_text(structural.get("court_chamber")),
        "decision_year": clean_text(structural.get("decision_year")),
        "decision_date": clean_text(structural.get("decision_date")),
        "consideration": clean_text(structural.get("consideration")),
        "paragraph_role": paragraph_role,
        "outcome_signal": outcome_signal,
        "law_codes": law_codes,
        "statutes": statutes,
        "issue_labels": issue_labels,
        "authority_roles": authority_roles,
        "is_notification_paragraph": bool(card.get("is_notification_paragraph")),
    }

    expansion = {
        "court_base": court_base,
        "source_citation": citation,
        "outgoing_court_citations": outgoing_court_cases,
        "outgoing_statute_citations": statutes,
        "outgoing_court_bases": unique_keep_order(
            court_base_from_citation(x) for x in outgoing_court_cases
        ),
    }

    return {
        "doc_id": doc_id,
        "source_line": line_no,
        "citation": citation,
        "court_base": court_base,
        "retrieval": {
            "vector_text": vector_text,
            "bm25_text": bm25_text,
            "citation_text": citation_text,
            "legal_text": legal_text,
            "rag_text": rag_text,
            "keyword_text": keyword_text,
            "source_text": source_text,
        },
        "filters": filters,
        "expansion": expansion,
        "ranking": {
            "authority_score": score,
            "court_base_source_count": int(structural.get("court_base_source_count") or 0),
            "court_base_text_ref_count": int(structural.get("court_base_text_ref_count") or 0),
        },
        "rag_enrichment": rag,
        "rag_quality": rag_quality(rag),
        "quality_flags": quality_flags,
        "provenance": {
            "input_method": clean_text((card.get("provenance") or {}).get("method"))
            if isinstance(card.get("provenance"), dict)
            else "",
            "builder": "build_court_retrieval_corpus_v1",
        },
    }


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield line_no, row


def target_line_idx_from_card(card: dict[str, Any]) -> int | None:
    target = card.get("_enrichment_target")
    if isinstance(target, dict) and target.get("line_idx") is not None:
        try:
            return int(target["line_idx"])
        except (TypeError, ValueError):
            return None
    if card.get("source_line_idx") is not None:
        try:
            return int(card["source_line_idx"])
        except (TypeError, ValueError):
            return None
    return None


def load_rag_overrides(path: Path) -> dict[int, dict[str, Any]]:
    overrides: dict[int, dict[str, Any]] = {}
    missing_idx = 0
    missing_rag = 0
    for _, card in iter_jsonl(path):
        idx = target_line_idx_from_card(card)
        if idx is None:
            missing_idx += 1
            continue
        rag = card.get("rag_enrichment")
        if not isinstance(rag, dict) or not rag:
            missing_rag += 1
            continue
        overrides[idx] = rag
    print(
        f"[overrides] loaded={len(overrides):,} "
        f"missing_idx={missing_idx:,} missing_rag={missing_rag:,} from {path}",
        flush=True,
    )
    return overrides


def choose_input(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if DEFAULT_RAG_INPUT.exists():
        return DEFAULT_RAG_INPUT
    return DEFAULT_V4_INPUT


def ensure_overwritable(path: Path, force: bool) -> None:
    if not path.exists():
        return
    if path.is_dir():
        raise IsADirectoryError(str(path))
    if not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite")


def open_retrieval_sqlite(path: Path, force: bool) -> tuple[sqlite3.Connection, bool]:
    ensure_overwritable(path, force)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-200000")
    if force:
        con.executescript(
            """
            DROP TABLE IF EXISTS documents_fts;
            DROP TABLE IF EXISTS documents;
            DROP TABLE IF EXISTS statute_links;
            DROP TABLE IF EXISTS court_case_links;
            DROP TABLE IF EXISTS citation_graph_edges;
            """
        )

    con.executescript(
        """
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            source_line INTEGER NOT NULL,
            citation TEXT NOT NULL,
            court_base TEXT NOT NULL,
            legal_area TEXT,
            language TEXT,
            pattern TEXT,
            subfamily TEXT,
            paragraph_role TEXT,
            outcome_signal TEXT,
            decision_year TEXT,
            docket_prefix TEXT,
            bge_division TEXT,
            legal_area_code TEXT,
            is_notification INTEGER NOT NULL,
            authority_score REAL NOT NULL,
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
            citation TEXT NOT NULL,
            statute TEXT NOT NULL
        );

        CREATE TABLE court_case_links (
            doc_id TEXT NOT NULL,
            citation TEXT NOT NULL,
            target_citation TEXT NOT NULL,
            target_base TEXT NOT NULL
        );

        CREATE TABLE citation_graph_edges (
            source_citation TEXT NOT NULL,
            target_citation TEXT NOT NULL,
            source_base TEXT NOT NULL,
            target_base TEXT NOT NULL,
            target_kind TEXT NOT NULL
        );
        """
    )
    fts_enabled = True
    try:
        con.execute(
            """
            CREATE VIRTUAL TABLE documents_fts USING fts5(
                doc_id UNINDEXED,
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
        print(f"[warn] FTS5 unavailable; SQLite BM25 table skipped: {exc}", file=sys.stderr)
        fts_enabled = False
    con.commit()
    return con, fts_enabled


def insert_documents(
    con: sqlite3.Connection,
    docs: list[dict[str, Any]],
    *,
    fts_enabled: bool,
) -> None:
    doc_rows = []
    fts_rows = []
    statute_rows = []
    case_rows = []
    for doc in docs:
        filters = doc["filters"]
        retrieval = doc["retrieval"]
        expansion = doc["expansion"]
        doc_rows.append(
            (
                doc["doc_id"],
                doc["source_line"],
                doc["citation"],
                doc["court_base"],
                filters.get("legal_area", ""),
                filters.get("language", ""),
                filters.get("pattern", ""),
                filters.get("subfamily", ""),
                filters.get("paragraph_role", ""),
                filters.get("outcome_signal", ""),
                filters.get("decision_year", ""),
                filters.get("docket_prefix", ""),
                filters.get("bge_division", ""),
                filters.get("legal_area_code", ""),
                1 if filters.get("is_notification_paragraph") else 0,
                doc["ranking"]["authority_score"],
                retrieval["vector_text"],
                retrieval["bm25_text"],
                retrieval["source_text"],
                json.dumps(doc["rag_enrichment"], ensure_ascii=False),
                json.dumps(filters, ensure_ascii=False),
                json.dumps(expansion, ensure_ascii=False),
                json.dumps(doc["quality_flags"], ensure_ascii=False),
            )
        )
        if fts_enabled:
            fts_rows.append(
                (
                    doc["doc_id"],
                    retrieval["citation_text"],
                    retrieval["legal_text"],
                    retrieval["rag_text"],
                    retrieval["keyword_text"],
                    retrieval["source_text"],
                )
            )
        for statute in filters.get("statutes", []):
            statute_rows.append((doc["doc_id"], doc["citation"], statute))
        for target in expansion.get("outgoing_court_citations", []):
            case_rows.append(
                (
                    doc["doc_id"],
                    doc["citation"],
                    target,
                    court_base_from_citation(target),
                )
            )

    con.executemany(
        """
        INSERT INTO documents (
            doc_id, source_line, citation, court_base, legal_area, language,
            pattern, subfamily, paragraph_role, outcome_signal, decision_year,
            docket_prefix, bge_division, legal_area_code, is_notification,
            authority_score, vector_text, bm25_text, source_text, rag_json,
            filters_json, expansion_json, quality_flags_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        doc_rows,
    )
    if fts_enabled and fts_rows:
        con.executemany(
            """
            INSERT INTO documents_fts (
                doc_id, citation_text, legal_text, rag_text, keyword_text, source_text
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            fts_rows,
        )
    if statute_rows:
        con.executemany(
            "INSERT INTO statute_links (doc_id, citation, statute) VALUES (?, ?, ?)",
            statute_rows,
        )
    if case_rows:
        con.executemany(
            """
            INSERT INTO court_case_links (doc_id, citation, target_citation, target_base)
            VALUES (?, ?, ?, ?)
            """,
            case_rows,
        )


def create_sqlite_indexes(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE INDEX idx_documents_citation ON documents(citation);
        CREATE INDEX idx_documents_court_base ON documents(court_base);
        CREATE INDEX idx_documents_legal_area ON documents(legal_area);
        CREATE INDEX idx_documents_law_area ON documents(legal_area_code);
        CREATE INDEX idx_documents_docket ON documents(docket_prefix);
        CREATE INDEX idx_documents_bge_division ON documents(bge_division);
        CREATE INDEX idx_documents_year ON documents(decision_year);
        CREATE INDEX idx_documents_role ON documents(paragraph_role);
        CREATE INDEX idx_documents_authority_score ON documents(authority_score);
        CREATE INDEX idx_statute_links_statute ON statute_links(statute);
        CREATE INDEX idx_statute_links_doc ON statute_links(doc_id);
        CREATE INDEX idx_court_case_links_target ON court_case_links(target_citation);
        CREATE INDEX idx_court_case_links_target_base ON court_case_links(target_base);
        CREATE INDEX idx_court_case_links_doc ON court_case_links(doc_id);
        CREATE INDEX idx_graph_source ON citation_graph_edges(source_citation);
        CREATE INDEX idx_graph_target ON citation_graph_edges(target_citation);
        CREATE INDEX idx_graph_source_base ON citation_graph_edges(source_base);
        CREATE INDEX idx_graph_target_base ON citation_graph_edges(target_base);
        CREATE INDEX idx_graph_target_kind ON citation_graph_edges(target_kind);
        """
    )
    con.commit()


def copy_graph_edges(
    graph_db: Path,
    con: sqlite3.Connection,
    *,
    included_citations: set[str] | None,
    progress_every: int,
) -> int:
    if not graph_db.exists():
        print(f"[warn] graph DB not found, skipping graph copy: {graph_db}", file=sys.stderr)
        return 0
    src = sqlite3.connect(graph_db)
    cur = src.execute(
        """
        SELECT source, target
        FROM edges
        WHERE dataset = 'court_considerations'
        """
    )
    rows: list[tuple[str, str, str, str, str]] = []
    copied = 0
    seen = 0
    for source, target in cur:
        seen += 1
        if included_citations is not None and source not in included_citations and target not in included_citations:
            continue
        source = clean_text(source)
        target = clean_text(target)
        rows.append(
            (
                source,
                target,
                court_base_from_citation(source),
                court_base_from_citation(target),
                citation_kind(target),
            )
        )
        if len(rows) >= 10000:
            con.executemany(
                """
                INSERT INTO citation_graph_edges (
                    source_citation, target_citation, source_base, target_base, target_kind
                ) VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            copied += len(rows)
            rows = []
        if progress_every and seen % progress_every == 0:
            print(f"[graph] scanned={seen:,} copied={copied:,}", flush=True)
    if rows:
        con.executemany(
            """
            INSERT INTO citation_graph_edges (
                source_citation, target_citation, source_base, target_base, target_kind
            ) VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        copied += len(rows)
    con.commit()
    src.close()
    return copied


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build fielded court retrieval data for BM25 + vector + filters + graph expansion."
    )
    ap.add_argument("--input", type=Path, default=None,
                    help="Enriched authority cards JSONL. Defaults to rag file if present, else v4.")
    ap.add_argument("--output-jsonl", type=Path, default=DEFAULT_JSONL_OUT)
    ap.add_argument("--output-sqlite", type=Path, default=DEFAULT_SQLITE_OUT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_OUT)
    ap.add_argument("--graph-db", type=Path, default=DEFAULT_GRAPH_DB)
    ap.add_argument("--rag-overrides", type=Path, default=None,
                    help="Optional JSONL of enriched target cards. Their rag_enrichment is applied by _enrichment_target.line_idx while streaming the full v4 file.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N cards after parsing. 0 means full corpus.")
    ap.add_argument("--batch-size", type=int, default=5000)
    ap.add_argument("--max-source-chars", type=int, default=2500)
    ap.add_argument("--max-vector-chars", type=int, default=4500)
    ap.add_argument("--no-sqlite", action="store_true",
                    help="Only write JSONL vector-ingestion documents.")
    ap.add_argument("--skip-graph-copy", action="store_true",
                    help="Do not copy citation graph edges into the SQLite output.")
    ap.add_argument("--require-rag", action="store_true",
                    help="Fail if any processed card lacks rag_enrichment.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output files.")
    ap.add_argument("--progress-every", type=int, default=100000)
    args = ap.parse_args()

    input_path = choose_input(args.input)
    if not input_path.exists():
        print(f"ERROR: input not found: {input_path}", file=sys.stderr)
        return 2
    rag_overrides: dict[int, dict[str, Any]] = {}
    if args.rag_overrides is not None:
        if not args.rag_overrides.exists():
            print(f"ERROR: rag overrides not found: {args.rag_overrides}", file=sys.stderr)
            return 2
        rag_overrides = load_rag_overrides(args.rag_overrides)

    ensure_overwritable(args.output_jsonl, args.force)
    ensure_overwritable(args.manifest, args.force)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_sqlite.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    con: sqlite3.Connection | None = None
    fts_enabled = False
    if not args.no_sqlite:
        con, fts_enabled = open_retrieval_sqlite(args.output_sqlite, args.force)

    stats = Counter()
    roles = Counter()
    outcomes = Counter()
    areas = Counter()
    languages = Counter()
    quality = Counter()
    included_citations: set[str] | None = set() if args.limit else None

    pending: list[dict[str, Any]] = []
    with args.output_jsonl.open("w", encoding="utf-8") as out_f:
        for line_no, card in iter_jsonl(input_path):
            zero_based_line_idx = line_no - 1
            if zero_based_line_idx in rag_overrides:
                card["rag_enrichment"] = rag_overrides[zero_based_line_idx]
                stats["applied_rag_overrides"] += 1
            doc = build_document(
                line_no,
                card,
                max_source_chars=args.max_source_chars,
                max_vector_chars=args.max_vector_chars,
            )
            if args.require_rag and not doc["rag_quality"]["has_rag"]:
                print(
                    f"ERROR: missing rag_enrichment at source line {line_no}: {doc['citation']}",
                    file=sys.stderr,
                )
                return 3

            out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            pending.append(doc)
            stats["documents"] += 1
            if doc["rag_quality"]["has_rag"]:
                stats["has_rag_enrichment"] += 1
            else:
                stats["missing_rag_enrichment"] += 1
            if doc["filters"].get("statutes"):
                stats["has_statutes"] += 1
            if doc["expansion"].get("outgoing_court_citations"):
                stats["has_outgoing_court_citations"] += 1
            if doc["filters"].get("issue_labels"):
                stats["has_issue_labels"] += 1
            if doc["filters"].get("is_notification_paragraph"):
                stats["notification_paragraphs"] += 1
            roles[doc["filters"].get("paragraph_role") or ""] += 1
            outcomes[doc["filters"].get("outcome_signal") or ""] += 1
            areas[doc["filters"].get("legal_area") or ""] += 1
            languages[doc["filters"].get("language") or ""] += 1
            for flag in doc["quality_flags"]:
                quality[flag] += 1
            if included_citations is not None:
                included_citations.add(doc["citation"])

            if con is not None and len(pending) >= args.batch_size:
                insert_documents(con, pending, fts_enabled=fts_enabled)
                con.commit()
                pending = []

            if args.progress_every and stats["documents"] % args.progress_every == 0:
                print(f"[docs] processed={stats['documents']:,}", flush=True)

            if args.limit and stats["documents"] >= args.limit:
                break

    if con is not None and pending:
        insert_documents(con, pending, fts_enabled=fts_enabled)
        con.commit()

    graph_edges = 0
    if con is not None and not args.skip_graph_copy:
        print("[graph] copying citation graph edges...", flush=True)
        graph_edges = copy_graph_edges(
            args.graph_db,
            con,
            included_citations=included_citations,
            progress_every=args.progress_every,
        )
        print(f"[graph] copied={graph_edges:,}", flush=True)
        print("[sqlite] creating indexes...", flush=True)
        create_sqlite_indexes(con)
        con.execute("VACUUM")
        con.close()
    elif con is not None:
        create_sqlite_indexes(con)
        con.close()

    manifest = {
        "builder": "build_court_retrieval_corpus_v1",
        "input": str(input_path),
        "output_jsonl": str(args.output_jsonl),
        "output_sqlite": None if args.no_sqlite else str(args.output_sqlite),
        "graph_db": str(args.graph_db),
        "graph_edges_copied": graph_edges,
        "fts_enabled": fts_enabled,
        "limit": args.limit,
        "stats": dict(stats),
        "top_paragraph_roles": roles.most_common(30),
        "top_outcome_signals": outcomes.most_common(30),
        "top_legal_areas": areas.most_common(30),
        "top_languages": languages.most_common(10),
        "quality_flags": quality.most_common(50),
        "retrieval_fields": {
            "vector_text": [
                "rag_enrichment.english_summary",
                "rag_enrichment.legal_topic",
                "rag_enrichment.legal_question",
                "rag_enrichment.legal_rule",
                "rag_enrichment.court_holding",
                "rag_enrichment.factual_context",
                "rag_enrichment.english_legal_concepts",
                "rag_enrichment.search_keywords",
                "rag_enrichment.natural_language_queries",
                "deterministic issue labels",
                "statutes and citation anchors",
            ],
            "bm25_columns": {
                "citation_text": "exact citation, court base, statutes, law codes",
                "legal_text": "legal area, issue labels, concepts, roles, outcome",
                "rag_text": "LLM English summary/question/rule/holding/facts/NL queries",
                "keyword_text": "search keywords, multilingual terms, labels, statutes",
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
    write_manifest(args.manifest, manifest)

    print(f"[done] documents={stats['documents']:,}", flush=True)
    print(f"[done] jsonl={args.output_jsonl}", flush=True)
    if not args.no_sqlite:
        print(f"[done] sqlite={args.output_sqlite}", flush=True)
    print(f"[done] manifest={args.manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
