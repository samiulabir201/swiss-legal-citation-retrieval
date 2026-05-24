#!/usr/bin/env python
"""Granularity resolver for Swiss federal-law citations.

Two granularities are used by the gold sets and the corpus:

  - Article-level     : "Art. 78 BV"
  - Paragraph-level   : "Art. 78 Abs. 1 BV", "Art. 78 Abs. 2 BV", ...

The corpus indexes ONLY paragraph-level rows when an article is split.
Rule (problem statement): if "Art. 11 Abs. 2 OR" exists in the corpus,
then "Art. 11 OR" CANNOT be a valid gold citation in the corpus -- it
must be expanded to all sibling paragraph-children.

This module provides:

  expand_article_to_paragraphs(citation, conn)  -> list[str]
  collapse_paragraphs_to_article(citation, conn) -> str | None
  is_article_level(citation)                    -> bool
  apply_granularity_post_filter(predicted, conn) -> list[str]

It also exposes a CLI:

  python -m scripts.granularity_resolver \\
        --gold-csv data/train.csv \\
        --output  data/train_granularity_expanded.csv

The CLI reads a (query_id, query, gold_citations) CSV, expands every
bare-article gold citation, and writes a new CSV with the corpus-aligned
ground truth (gold_citations now contain only citations that exist as
rows in the unified retrieval corpus, modulo true article-level rows
without paragraph children).

Schema notes (verified against artifacts/unified_retrieval.sqlite produced by
scripts/build_unified_retrieval_corpus.py):

  - Table `documents` is the unified law+court row store.
  - For family='law' rows, column `granularity` is "article" or "paragraph".
  - Column `expansion_json` is a JSON blob containing
        "adjacent_siblings": [<citation>, ...]
    populated from build_law_authority_cards.add_neighbors() ->
    normalized_anchors.adjacent_citations.same_article_siblings.
  - Columns `law_code` and `article` are filter-friendly singletons.

Sibling discovery order (resolver_module is robust against schema drift):

  1. Direct corpus lookup: SELECT citations from `documents` WHERE
     family='law' AND law_code=? AND article=? AND granularity='paragraph'.
     This is the source of truth -- it is exactly what the corpus indexes.
  2. Fallback: parse the bare-article into (article, law_code) and run
     a LIKE query on the citation column to recover paragraph siblings.

The module imports parsing helpers from `extract_citation_graph` so that
its citation parsing stays consistent with the rest of the pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the project's citation parser so that "is_article_level" and the
# (article, law_code) decomposition stay consistent with the audit.
from extract_citation_graph import classify_source_citation, squash_ws  # noqa: E402

DEFAULT_DB_PATH = ROOT / "artifacts" / "unified_retrieval.sqlite"

# Markers that prove a citation is paragraph-level (or finer).
# This mirrors UNIT_RE in extract_citation_graph but is restricted to the
# unit labels the gold sets actually use.
PARAGRAPH_UNIT_RE = re.compile(
    r"\b(?:"
    r"Abs(?:\.|atz|ätze|aetze)?|"
    r"al\.?|"
    r"Bst\.?|Buchstabe(?:n)?|"
    r"lit\.?|"
    r"Ziff\.?|Ziffer|"
    r"Nr\.?|Nummer(?:n)?|"
    r"Satz|"
    r"Unterabsatz|Unterabs\.?"
    r")\b",
    re.IGNORECASE,
)

# Article + law-code bare form, e.g. "Art. 78 BV", "Art. 11a OR".
BARE_ARTICLE_RE = re.compile(
    r"^Art\.\s+(?P<article>\d+[A-Za-z]*"
    r"(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?)"
    r"\s+(?P<law_code>.+)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Citation parsing helpers
# ---------------------------------------------------------------------------

def _normalize(citation: str) -> str:
    return squash_ws(citation or "")


def is_article_level(citation: str) -> bool:
    """True iff the citation has no Abs./Bst./Ziff./lit./Satz/Nr. unit.

    A truly article-level citation is one like "Art. 78 BV" with no
    finer subdivision; "Art. 78 Abs. 1 BV" returns False.
    """
    norm = _normalize(citation)
    if not norm:
        return False
    return PARAGRAPH_UNIT_RE.search(norm) is None


def _parse_article_law_code(citation: str) -> tuple[str, str] | None:
    """Return (article_token, law_code) for a bare-article citation, else None.

    Uses classify_source_citation first (consistent with the rest of the
    codebase); falls back to a regex on the literal citation string.
    """
    norm = _normalize(citation)
    if not norm:
        return None

    parsed = classify_source_citation(norm)
    segments = parsed.segments or {}
    article = (segments.get("article") or "").strip()
    law_code = (segments.get("law_code") or "").strip()
    units = segments.get("units") or []

    # If the parser already attached units, this is paragraph-level -- not bare.
    if units:
        return None
    if article and law_code:
        return article, law_code

    match = BARE_ARTICLE_RE.match(norm)
    if not match:
        return None
    return match.group("article").strip(), match.group("law_code").strip()


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _detect_law_table(conn_id: int, db_path: str) -> tuple[bool, bool]:
    """Probe the SQLite for the columns we need.

    Returns (has_documents_table, has_granularity_column).
    The conn_id parameter is only used to make lru_cache key off the
    connection; we resolve the actual schema from db_path.
    """
    del conn_id  # only here to differentiate cache keys per connection
    probe = sqlite3.connect(db_path)
    try:
        cur = probe.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
        )
        has_table = cur.fetchone() is not None
        if not has_table:
            return False, False
        cur = probe.execute("PRAGMA table_info(documents)")
        cols = {row[1] for row in cur.fetchall()}
        return True, "granularity" in cols and "law_code" in cols and "article" in cols
    finally:
        probe.close()


def _conn_db_path(conn: sqlite3.Connection) -> str:
    """Best-effort: return the file backing this connection."""
    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
        for _, name, file_path in rows:
            if name == "main" and file_path:
                return file_path
    except sqlite3.Error:
        pass
    return ""


def _query_paragraph_siblings_via_documents(
    conn: sqlite3.Connection, article: str, law_code: str
) -> list[str]:
    """Source-of-truth lookup against the `documents` table.

    Returns the list of paragraph-level citations for (article, law_code)
    that are actually indexed in the corpus.
    """
    rows = conn.execute(
        """
        SELECT citation
          FROM documents
         WHERE family = 'law'
           AND law_code = ?
           AND article  = ?
           AND granularity = 'paragraph'
        """,
        (law_code, article),
    ).fetchall()
    return [r[0] for r in rows if r and r[0]]


def _query_paragraph_siblings_via_like(
    conn: sqlite3.Connection, article: str, law_code: str
) -> list[str]:
    """Schema-agnostic fallback that uses LIKE on the citation string.

    Pattern: "Art. <article> %<law_code>" with paragraph markers somewhere
    in the middle.  We post-filter on PARAGRAPH_UNIT_RE to keep only
    paragraph-level results.
    """
    like = f"Art. {article} %{law_code}"
    try:
        rows = conn.execute(
            "SELECT DISTINCT citation FROM documents WHERE family='law' AND citation LIKE ?",
            (like,),
        ).fetchall()
    except sqlite3.OperationalError:
        # If even the documents table is missing, give up -- caller falls through.
        return []
    siblings: list[str] = []
    for (cit,) in rows:
        if not cit:
            continue
        if PARAGRAPH_UNIT_RE.search(cit) and cit.endswith(law_code):
            siblings.append(cit)
    return siblings


def _query_via_expansion_json(
    conn: sqlite3.Connection, article_citation: str
) -> list[str]:
    """Last-resort: read expansion_json.adjacent_siblings for the bare-article row.

    Some early audits keyed off this field; we keep it as a third fallback
    in case the bare-article row is itself in the corpus and the per-row
    siblings list is more accurate than (article, law_code) lookup.
    """
    try:
        rows = conn.execute(
            "SELECT expansion_json FROM documents WHERE family='law' AND citation=?",
            (article_citation,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    siblings: list[str] = []
    for (blob,) in rows:
        if not blob:
            continue
        try:
            payload = json.loads(blob)
        except (TypeError, json.JSONDecodeError):
            continue
        sibs = payload.get("adjacent_siblings") or []
        for s in sibs:
            if isinstance(s, str) and s and PARAGRAPH_UNIT_RE.search(s):
                siblings.append(s)
    return siblings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_article_to_paragraphs(
    citation: str, conn: sqlite3.Connection
) -> list[str]:
    """Expand a bare-article citation into its paragraph-children in the corpus.

    Behavior contract:
      - paragraph-level input  -> [citation] unchanged
      - bare-article input with corpus paragraph-children -> sorted unique list
      - bare-article input with NO paragraph-children     -> [citation]
        (it is a true article-level citation; corpus indexes it as such)
      - empty / unparseable input -> [citation]
    """
    norm = _normalize(citation)
    if not norm:
        return [citation] if citation else []

    if not is_article_level(norm):
        return [norm]

    parsed = _parse_article_law_code(norm)
    if not parsed:
        return [norm]

    article, law_code = parsed

    siblings = _query_paragraph_siblings_via_documents(conn, article, law_code)
    if not siblings:
        siblings = _query_paragraph_siblings_via_like(conn, article, law_code)
    if not siblings:
        siblings = _query_via_expansion_json(conn, norm)

    if not siblings:
        # No paragraph-children indexed: this is a true article-level citation.
        return [norm]

    # De-dup, preserve a stable sort.
    return sorted({_normalize(s) for s in siblings if s})


def collapse_paragraphs_to_article(
    citation: str, conn: sqlite3.Connection | None = None
) -> str | None:
    """Inverse of expand: paragraph-level citation -> bare-article form.

    Useful for grouping paragraph-children under their article root for
    metric reporting.  Returns None if the input is not a paragraph
    citation (or cannot be parsed).
    """
    del conn  # purely string-based, kept in signature for symmetry
    norm = _normalize(citation)
    if not norm or is_article_level(norm):
        return None

    parsed = classify_source_citation(norm)
    segments = parsed.segments or {}
    article = (segments.get("article") or "").strip()
    law_code = (segments.get("law_code") or "").strip()
    if not article or not law_code:
        match = BARE_ARTICLE_RE.match(norm)
        if not match:
            return None
        article = match.group("article").strip()
        law_code = match.group("law_code").strip()
    if not article or not law_code:
        return None
    return f"Art. {article} {law_code}"


def apply_granularity_post_filter(
    predicted_citations: Sequence[str], conn: sqlite3.Connection
) -> list[str]:
    """Post-filter for the retrieval pipeline.

    For every predicted bare-article citation, REPLACE it with its
    paragraph-children present in the corpus (so the prediction matches
    the corpus granularity and is comparable to the gold).  De-dup the
    union.  Order is stable (first-occurrence wins, then expansions in
    sibling order).
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in predicted_citations:
        norm = _normalize(raw)
        if not norm:
            continue
        for expanded in expand_article_to_paragraphs(norm, conn):
            if expanded in seen:
                continue
            seen.add(expanded)
            out.append(expanded)
    return out


# ---------------------------------------------------------------------------
# CLI: expand the gold-citations column of a CSV
# ---------------------------------------------------------------------------

GOLD_SEPARATOR = ";"


def _split_gold(value: str) -> list[str]:
    return [_normalize(part) for part in (value or "").split(GOLD_SEPARATOR) if _normalize(part)]


def _join_gold(values: Iterable[str]) -> str:
    return GOLD_SEPARATOR.join(values)


def expand_csv(
    gold_csv: Path,
    output_csv: Path,
    db_path: Path,
    *,
    gold_column: str = "gold_citations",
) -> dict:
    """Expand bare-article citations in `gold_column` of a CSV.

    Returns a dict of statistics.
    """
    if not gold_csv.exists():
        raise FileNotFoundError(f"gold csv not found: {gold_csv}")
    if not db_path.exists():
        raise FileNotFoundError(f"sqlite db not found: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        # Prime the schema probe (purely informational).
        _detect_law_table(id(conn), str(db_path))

        stats = {
            "rows": 0,
            "gold_mentions": 0,
            "unique_gold_in": set(),
            "unique_gold_out": set(),
            "bare_article_inputs": 0,
            "bare_article_with_children": 0,
            "bare_article_without_children": 0,
            "expansion_fanouts": [],
        }

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with gold_csv.open("r", encoding="utf-8-sig", newline="") as src, \
             output_csv.open("w", encoding="utf-8", newline="") as dst:
            reader = csv.DictReader(src)
            if reader.fieldnames is None:
                raise ValueError(f"empty CSV: {gold_csv}")
            if gold_column not in reader.fieldnames:
                raise ValueError(
                    f"column '{gold_column}' not in {gold_csv} "
                    f"(have: {reader.fieldnames})"
                )

            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                stats["rows"] += 1
                gold_in = _split_gold(row.get(gold_column, ""))
                gold_out: list[str] = []
                seen: set[str] = set()
                for cit in gold_in:
                    stats["gold_mentions"] += 1
                    stats["unique_gold_in"].add(cit)
                    if is_article_level(cit):
                        stats["bare_article_inputs"] += 1
                        expanded = expand_article_to_paragraphs(cit, conn)
                        if len(expanded) == 1 and expanded[0] == cit:
                            stats["bare_article_without_children"] += 1
                        else:
                            stats["bare_article_with_children"] += 1
                            stats["expansion_fanouts"].append(len(expanded))
                    else:
                        expanded = [cit]
                    for ex in expanded:
                        if ex in seen:
                            continue
                        seen.add(ex)
                        gold_out.append(ex)
                        stats["unique_gold_out"].add(ex)

                row[gold_column] = _join_gold(gold_out)
                writer.writerow(row)

        # Flatten stats for JSON-friendliness.
        fanouts = stats.pop("expansion_fanouts")
        stats["unique_gold_in"] = len(stats["unique_gold_in"])
        stats["unique_gold_out"] = len(stats["unique_gold_out"])
        stats["expansion_fanout_total"] = sum(fanouts)
        stats["expansion_fanout_avg"] = (
            round(sum(fanouts) / len(fanouts), 3) if fanouts else 0.0
        )
        stats["expansion_fanout_max"] = max(fanouts) if fanouts else 0
        return stats
    finally:
        conn.close()


def report_coverage(
    gold_csv: Path,
    db_path: Path,
    *,
    gold_column: str = "gold_citations",
) -> dict:
    """Compute corpus coverage of gold citations BEFORE and AFTER expansion.

    "Coverage" here = fraction of unique gold citations whose exact string
    is a row in `documents` (family='law' OR family='court').
    """
    conn = sqlite3.connect(db_path)
    try:
        with gold_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            gold: set[str] = set()
            for row in reader:
                for cit in _split_gold(row.get(gold_column, "")):
                    gold.add(cit)

        present_before = 0
        for cit in gold:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE citation = ? LIMIT 1", (cit,)
            ).fetchone()
            if row:
                present_before += 1

        # Expanded set
        expanded: set[str] = set()
        for cit in gold:
            for ex in expand_article_to_paragraphs(cit, conn):
                expanded.add(ex)

        present_after = 0
        for cit in expanded:
            row = conn.execute(
                "SELECT 1 FROM documents WHERE citation = ? LIMIT 1", (cit,)
            ).fetchone()
            if row:
                present_after += 1

        return {
            "unique_gold_in": len(gold),
            "unique_expanded": len(expanded),
            "present_before": present_before,
            "present_after_in_expanded": present_after,
            "coverage_before_pct": round(100.0 * present_before / max(len(gold), 1), 2),
            "coverage_after_pct": round(
                100.0 * present_after / max(len(expanded), 1), 2
            ),
        }
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold-csv", type=Path, required=True,
                        help="Input gold CSV (must have a gold-citations column).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output CSV with expanded gold citations.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH,
                        help=f"unified retrieval sqlite (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--gold-column", type=str, default="gold_citations")
    parser.add_argument("--report-coverage", action="store_true",
                        help="Also print before/after corpus coverage of unique gold citations.")
    args = parser.parse_args(argv)

    stats = expand_csv(args.gold_csv, args.output, args.db,
                       gold_column=args.gold_column)
    print(json.dumps({"output": str(args.output), "stats": stats},
                     ensure_ascii=False, indent=2))

    if args.report_coverage:
        coverage = report_coverage(args.gold_csv, args.db,
                                   gold_column=args.gold_column)
        print(json.dumps({"coverage": coverage}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
