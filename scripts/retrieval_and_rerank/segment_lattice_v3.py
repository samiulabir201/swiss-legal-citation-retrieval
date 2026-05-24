#!/usr/bin/env python
"""Build and run the no-leak Segment-Lattice Funnel v3.

The index is derived from the extracted `data_insights/*_classified_citations.jsonl`
files, but source citation rows are re-parsed with the current parser so stale
segment bugs such as `OR` being nulled do not survive in the v3 source index.
Gold labels are only read by the explicit `audit` command.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from extract_citation_graph import (
    EXTRACTOR_VERSION,
    ORIGIN_REFERENCE,
    ORIGIN_SOURCE,
    Citation,
    classify_source_citation,
    extract_references,
    squash_ws,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
INSIGHTS_DIR = ROOT / "data_insights"
ART_DIR = ROOT / "artifacts"

SCHEMA_VERSION = "segment_lattice_v3_schema_2026_04_27_b_cosignals"
DEFAULT_DB = ART_DIR / "segment_lattice_v3.sqlite"
DEFAULT_CARDS = ART_DIR / "choice_cards_v3.json"

DATASET_FILES = {
    "laws_de": "laws_de_classified_citations.jsonl",
    "court_considerations": "court_considerations_classified_citations.jsonl",
}

STALE_ARTIFACT_PATTERNS = [
    "recall_funnel.duckdb",
    "choice_cards.json",
    "*candidate_sets.jsonl",
    "*planner_outputs.jsonl",
]

BGE_DIVISION_MEANINGS = {
    "I": "BGE division I: constitutional and public law decisions.",
    "II": "BGE division II: public, administrative, tax, migration, and regulatory decisions.",
    "III": "BGE division III: civil law decisions.",
    "IV": "BGE division IV: criminal law and criminal procedure decisions.",
    "V": "BGE division V: social insurance decisions.",
}

LAW_CODE_FAMILY_MEANINGS = {
    "uppercase_abbreviation": "Uppercase legal abbreviation observed as a law code.",
    "mixedcase_abbreviation": "Mixed-case legal abbreviation observed as a law code.",
    "systematic_collection_number": "Swiss systematic collection number used as the law code.",
    "compound_abbreviation": "Compound or hyphenated legal abbreviation.",
    "numbered_abbreviation": "Legal abbreviation with a trailing number.",
    "cantonal_or_slash_abbreviation": "Slash-style or cantonal-looking legal abbreviation.",
    "unresolved_law_code": "No reliable law code was resolved for this citation.",
}

STRUCTURAL_SEGMENTS = {
    "pattern",
    "subfamily",
    "fallback_pattern",
    "fallback_subfamily",
    "article_marker",
    "article_series_kind",
    "article_continuation",
    "law_code_family",
    "law_code_resolution",
    "unit_chain",
    "unit_category",
    "unit_marker",
    "section_marker",
    "reporter",
    "pinpoint_unit",
    "separator_style",
}

EXPLICIT_ONLY_SEGMENTS = {
    "article",
    "section",
    "docket",
    "court_base",
    "serial_number",
    "decision_year",
    "decision_date",
    "volume",
    "page",
    "pinpoint",
    "consideration",
    "consideration_raw",
    "raw",
    "number",
    "year",
}

SEMANTIC_SEGMENTS = {
    "law_code",
    "division",
    "docket_prefix",
    "legal_area_code",
}


def loads_json(raw: str | bytes) -> dict:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def dumps_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_metadata(data_dir: Path, insights_dir: Path) -> dict[str, str]:
    files = {
        "laws_csv": data_dir / "laws_de.csv",
        "court_csv": data_dir / "court_considerations.csv",
        "laws_classified": insights_dir / DATASET_FILES["laws_de"],
        "court_classified": insights_dir / DATASET_FILES["court_considerations"],
    }
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
    }
    for name, path in files.items():
        metadata[f"{name}_path"] = str(path)
        metadata[f"{name}_sha256"] = file_sha256(path) if path.exists() else "MISSING"
    return metadata


def existing_metadata(conn: sqlite3.Connection) -> dict[str, str]:
    try:
        rows = conn.execute("SELECT key, value FROM metadata").fetchall()
    except sqlite3.Error:
        return {}
    return {key: value for key, value in rows}


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS metadata;
        DROP TABLE IF EXISTS source_citations;
        DROP TABLE IF EXISTS citation_segments;
        DROP TABLE IF EXISTS option_registry;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE source_citations (
            dataset TEXT NOT NULL,
            citation TEXT NOT NULL,
            family TEXT NOT NULL,
            subfamily TEXT NOT NULL,
            pattern TEXT NOT NULL,
            origin_mask INTEGER NOT NULL,
            segments_json TEXT NOT NULL,
            PRIMARY KEY (dataset, citation)
        );

        CREATE TABLE citation_segments (
            dataset TEXT NOT NULL,
            citation TEXT NOT NULL,
            family TEXT NOT NULL,
            subfamily TEXT NOT NULL,
            pattern TEXT NOT NULL,
            segment_key TEXT NOT NULL,
            option_value TEXT NOT NULL,
            selector_kind TEXT NOT NULL
        );

        CREATE TABLE option_registry (
            dataset TEXT NOT NULL,
            family TEXT NOT NULL,
            subfamily TEXT NOT NULL,
            pattern TEXT NOT NULL,
            segment_key TEXT NOT NULL,
            option_value TEXT NOT NULL,
            meaning_en TEXT NOT NULL,
            source_count INTEGER NOT NULL,
            text_ref_count INTEGER NOT NULL,
            examples_json TEXT NOT NULL,
            selector_kind TEXT NOT NULL,
            co_signals_json TEXT NOT NULL DEFAULT '[]',
            text_keywords_json TEXT NOT NULL DEFAULT '[]',
            PRIMARY KEY (dataset, family, subfamily, pattern, segment_key, option_value)
        );

        CREATE INDEX idx_source_family ON source_citations(dataset, family);
        CREATE INDEX idx_source_citation ON source_citations(dataset, citation);
        CREATE INDEX idx_segments_lookup ON citation_segments(dataset, family, segment_key, option_value);
        CREATE INDEX idx_registry_lookup ON option_registry(dataset, family, segment_key, option_value);
        """
    )


def iter_classified(path: Path, dataset: str, refresh_source_parsing: bool = True) -> Iterable[dict]:
    with path.open("rb") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            row = loads_json(raw_line)
            origin_mask = int(row.get("origin_mask") or 0)
            if refresh_source_parsing and (origin_mask & ORIGIN_SOURCE):
                parsed = classify_source_citation(row["citation"])
                row = {
                    "citation": parsed.citation,
                    "family": parsed.family,
                    "subfamily": parsed.subfamily,
                    "pattern": parsed.pattern,
                    "segments": parsed.segments,
                    "origin_mask": origin_mask,
                }
            row["dataset"] = dataset
            row["origin_mask"] = origin_mask
            yield row


def stringify_option(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return dumps_json(value)
    value = str(value).strip()
    return value if value else None


def court_base(pattern: str, segments: dict) -> str | None:
    if pattern == "court_bge":
        volume = segments.get("volume")
        division = segments.get("division")
        page = segments.get("page")
        if volume and division and page:
            return f"BGE {volume} {division} {page}"
    docket = segments.get("docket")
    return str(docket) if docket else None


def docket_prefix(segments: dict) -> str | None:
    if segments.get("docket_prefix"):
        return str(segments["docket_prefix"])
    chamber = segments.get("court_chamber")
    area = segments.get("legal_area_code")
    if chamber and area:
        return f"{chamber}{area}"
    if area:
        return str(area)
    return None


def segment_options(row: dict) -> list[tuple[str, str]]:
    pattern = row.get("pattern") or "unknown"
    subfamily = row.get("subfamily") or "unknown"
    segments = row.get("segments") or {}
    out: list[tuple[str, str]] = [("pattern", pattern), ("subfamily", subfamily)]

    base = court_base(pattern, segments)
    if base:
        out.append(("court_base", base))
    prefix = docket_prefix(segments)
    if prefix:
        out.append(("docket_prefix", prefix))

    for key, value in segments.items():
        if key == "units":
            units = value or []
            if isinstance(units, list):
                chain = ">".join(str(unit.get("category")) for unit in units if unit.get("category"))
                out.append(("unit_chain", chain or "article_only"))
                for unit in units:
                    category = stringify_option(unit.get("category"))
                    marker = stringify_option(unit.get("marker"))
                    unit_value = stringify_option(unit.get("value"))
                    if category:
                        out.append(("unit_category", category))
                        if unit_value:
                            out.append((f"unit.{category}", unit_value))
                    if marker:
                        out.append(("unit_marker", marker))
            continue
        option_value = stringify_option(value)
        if option_value is not None:
            out.append((key, option_value))

    return list(dict.fromkeys(out))


def selector_kind(family: str, pattern: str, segment_key: str) -> str:
    if segment_key == "law_code" and family == "law":
        return "semantic"
    if segment_key == "division" and pattern == "court_bge":
        return "semantic"
    if segment_key in {"docket_prefix", "legal_area_code"} and family == "court":
        return "semantic"
    if segment_key in STRUCTURAL_SEGMENTS or segment_key.startswith("unit."):
        return "structural"
    if segment_key in EXPLICIT_ONLY_SEGMENTS:
        return "explicit_only"
    if segment_key in SEMANTIC_SEGMENTS:
        return "semantic"
    return "explicit_only"


def law_title_context(data_dir: Path) -> dict[str, str]:
    contexts: dict[str, Counter] = defaultdict(Counter)
    path = data_dir / "laws_de.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            citation = squash_ws(row.get("citation", ""))
            if not citation:
                continue
            parsed = classify_source_citation(citation)
            code = parsed.segments.get("law_code") if parsed.family == "law" else None
            title = squash_ws(row.get("title", ""))
            if code and title:
                contexts[str(code)][title.split(" - ")[0]] += 1
    return {code: counter.most_common(1)[0][0] for code, counter in contexts.items() if counter}


def meaning_for(
    family: str,
    pattern: str,
    segment_key: str,
    option_value: str,
    law_titles: dict[str, str],
) -> str:
    if segment_key == "law_code":
        title = law_titles.get(option_value)
        if title:
            return f"Law code {option_value}: {title}."
        return f"Law code {option_value} observed in source citations."
    if segment_key == "law_code_family":
        return LAW_CODE_FAMILY_MEANINGS.get(option_value, f"Law-code family {option_value}.")
    if pattern == "court_bge" and segment_key == "division":
        return BGE_DIVISION_MEANINGS.get(option_value, f"BGE division {option_value}.")
    if segment_key == "docket_prefix":
        return f"Federal Tribunal docket prefix {option_value}, derived from court chamber plus legal area code when available."
    if segment_key == "legal_area_code":
        return f"Court docket legal area code {option_value}, observed in Federal Tribunal docket identifiers."
    if segment_key == "subfamily":
        return f"Extracted citation subfamily {option_value}."
    if segment_key == "pattern":
        return f"Extracted citation pattern {option_value}."
    if segment_key == "separator_style":
        return f"Docket separator style {option_value}."
    if segment_key == "unit_chain":
        return f"Law unit chain {option_value}; e.g. paragraph or article-only granularity."
    if segment_key in EXPLICIT_ONLY_SEGMENTS or segment_key.startswith("unit."):
        return f"Identifier-like option {option_value} for segment {segment_key}; use mainly for explicit query mentions."
    return f"Observed option {option_value} for segment {segment_key}."


GERMAN_STOPWORDS = {
    "Der", "Die", "Das", "Den", "Dem", "Des", "Ein", "Eine", "Einem", "Einen", "Einer", "Eines",
    "Und", "Oder", "Aber", "Doch", "Sondern", "Denn", "Wenn", "Dann", "Weil", "Dass",
    "Ist", "Sind", "War", "Waren", "Sein", "Wird", "Werden", "Wurde", "Wurden", "Worden",
    "Hat", "Hatte", "Haben", "Hatten", "Habe",
    "Auf", "Aus", "Bei", "Mit", "Von", "Vom", "Zur", "Zum", "Zu", "Im", "In", "An",
    "Nach", "Vor", "Unter", "Ueber", "Über", "Durch", "Gegen", "Ohne", "Gemäss", "Gemäß",
    "Sich", "Auch", "Nur", "Noch", "Schon", "Wie", "Als", "So", "Wo", "Was",
    "Diese", "Dieser", "Dieses", "Diesem", "Diesen",
    "Es", "Er", "Sie", "Wir", "Ihr", "Ihm", "Ihn", "Ihnen", "Ihre", "Ihrer", "Ihres",
    "Art", "Abs", "Lit", "Bzw", "Vgl", "Resp", "Etc",
    "BGE", "BGer", "Urteil", "Erw", "Ziff", "Bst",
}

LAW_CODE_REGEX_BLOCKLIST = {
    # Single-letter or pure-number codes that would match too aggressively
    # against ordinary tokens or numbers in court text.
    "OR",
}


def _safe_law_code_for_regex(value: str) -> bool:
    if len(value) < 2:
        return False
    if value in LAW_CODE_REGEX_BLOCKLIST:
        return True
    return True


def compute_court_co_signals(
    conn: sqlite3.Connection,
    data_dir: Path,
    max_rows: int | None = None,
) -> tuple[dict[tuple[str, str], list[dict]], dict[tuple[str, str], list[str]]]:
    """Scan court_considerations.csv text once, return:
        co_signals[(segment_key, option_value)] = [{"law_code": str, "count": int, "freq": float}]
        text_keywords[(segment_key, option_value)] = ["term1", ...]

    Build is data-driven from corpus evidence; no gold or query labels are touched.
    """
    csv_path = data_dir / "court_considerations.csv"
    if not csv_path.exists():
        return {}, {}

    # Build lookup: citation -> {"docket_prefix": str|None, "division": str|None}
    seg_lookup: dict[str, dict[str, str]] = defaultdict(dict)
    for citation, segment_key, option_value in conn.execute(
        """
        SELECT citation, segment_key, option_value
        FROM citation_segments
        WHERE dataset = 'court_considerations'
          AND segment_key IN ('docket_prefix', 'division')
        """
    ):
        seg_lookup[citation][segment_key] = option_value

    # All semantic law codes registered.
    law_codes = [
        row[0]
        for row in conn.execute(
            """
            SELECT option_value FROM option_registry
            WHERE dataset = 'laws_de' AND family = 'law'
              AND segment_key = 'law_code' AND selector_kind = 'semantic'
            ORDER BY length(option_value) DESC, option_value
            """
        )
        if _safe_law_code_for_regex(row[0])
    ]
    if not law_codes:
        return {}, {}

    # Word-boundary regex over all known law codes; compiled once.
    code_pattern = re.compile(
        r"(?<![A-Za-z0-9])(" + "|".join(re.escape(c) for c in law_codes) + r")(?![A-Za-z0-9])"
    )
    word_pattern = re.compile(r"\b([A-ZÄÖÜ][a-zäöüß]{4,})\b")

    co_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    kw_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    option_doc_counts: Counter = Counter()

    csv.field_size_limit(2**31 - 1)
    seen_citations = 0
    matched_citations = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            seen_citations += 1
            if max_rows is not None and seen_citations > max_rows:
                break
            citation = squash_ws(row.get("citation", ""))
            if not citation:
                continue
            segs = seg_lookup.get(citation)
            if not segs:
                continue
            text = row.get("text") or ""
            if not text:
                continue
            matched_citations += 1
            codes_in_text = set(code_pattern.findall(text))
            words_in_text = set(word_pattern.findall(text))
            words_in_text = {w for w in words_in_text if w not in GERMAN_STOPWORDS}
            for skey, sval in segs.items():
                key = (skey, sval)
                option_doc_counts[key] += 1
                for code in codes_in_text:
                    co_counts[key][code] += 1
                for word in words_in_text:
                    kw_counts[key][word] += 1
            if seen_citations % 250_000 == 0:
                print(f"  co-signals scan: {seen_citations:,} rows, {matched_citations:,} matched", flush=True)

    co_signals: dict[tuple[str, str], list[dict]] = {}
    text_keywords: dict[tuple[str, str], list[str]] = {}
    for key, counter in co_counts.items():
        docs = max(option_doc_counts[key], 1)
        top = counter.most_common(8)
        co_signals[key] = [
            {"law_code": code, "count": cnt, "freq": round(cnt / docs, 4)}
            for code, cnt in top
            if cnt >= 5
        ]
    for key, counter in kw_counts.items():
        docs = max(option_doc_counts[key], 1)
        top = [
            term for term, cnt in counter.most_common(40)
            if cnt >= max(5, docs // 200)
        ][:8]
        text_keywords[key] = top
    print(
        f"co-signals built: court rows scanned={seen_citations:,}, matched_to_segments={matched_citations:,}, "
        f"options_with_signals={len(co_signals)}",
        flush=True,
    )
    return co_signals, text_keywords


def build_index(
    data_dir: Path = DATA_DIR,
    insights_dir: Path = INSIGHTS_DIR,
    db_path: Path = DEFAULT_DB,
    cards_path: Path = DEFAULT_CARDS,
    force: bool = False,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    cards_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = current_metadata(data_dir, insights_dir)

    if db_path.exists() and not force:
        conn = sqlite3.connect(db_path)
        if existing_metadata(conn) == metadata:
            print(f"Fresh v3 index already exists: {db_path}")
            export_choice_cards(conn, cards_path)
            conn.close()
            return
        conn.close()
        print("Existing v3 index is stale; rebuilding.")

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA synchronous = OFF")
    create_schema(conn)
    conn.executemany("INSERT INTO metadata (key, value) VALUES (?, ?)", sorted(metadata.items()))

    law_titles = law_title_context(data_dir)
    registry_counts: Counter = Counter()
    registry_text_counts: Counter = Counter()
    registry_examples: dict[tuple, list[str]] = defaultdict(list)
    source_batch = []
    segment_batch = []

    for dataset, file_name in DATASET_FILES.items():
        path = insights_dir / file_name
        if not path.exists():
            raise FileNotFoundError(path)
        scanned = 0
        source_rows = 0
        for row in iter_classified(path, dataset):
            scanned += 1
            origin_mask = int(row.get("origin_mask") or 0)
            is_source = bool(origin_mask & ORIGIN_SOURCE)
            is_text_ref = bool(origin_mask & ORIGIN_REFERENCE)
            citation = row["citation"]
            family = row.get("family") or "unknown"
            subfamily = row.get("subfamily") or "unknown"
            pattern = row.get("pattern") or "unknown"
            segments = row.get("segments") or {}
            options = segment_options(row)

            if is_source:
                source_rows += 1
                source_batch.append(
                    (
                        dataset,
                        citation,
                        family,
                        subfamily,
                        pattern,
                        origin_mask,
                        dumps_json(segments),
                    )
                )
                for segment_key, option_value in options:
                    kind = selector_kind(family, pattern, segment_key)
                    segment_batch.append(
                        (
                            dataset,
                            citation,
                            family,
                            subfamily,
                            pattern,
                            segment_key,
                            option_value,
                            kind,
                        )
                    )

            for segment_key, option_value in options:
                kind = selector_kind(family, pattern, segment_key)
                reg_key = (dataset, family, subfamily, pattern, segment_key, option_value, kind)
                if is_source:
                    registry_counts[reg_key] += 1
                    if len(registry_examples[reg_key]) < 5:
                        registry_examples[reg_key].append(citation)
                if is_text_ref:
                    registry_text_counts[reg_key] += 1

            if len(source_batch) >= 50_000:
                flush_source_batches(conn, source_batch, segment_batch)
            if scanned % 500_000 == 0:
                print(f"{dataset}: scanned={scanned:,} source_rows={source_rows:,}", flush=True)

        flush_source_batches(conn, source_batch, segment_batch)
        print(f"{dataset}: done scanned={scanned:,} source_rows={source_rows:,}")

    # Insert registry rows first without co-signals; we need the registry
    # populated so the co-signals scan can read law_code semantic options.
    registry_rows = []
    for reg_key, source_count in registry_counts.items():
        dataset, family, subfamily, pattern, segment_key, option_value, kind = reg_key
        registry_rows.append(
            (
                dataset,
                family,
                subfamily,
                pattern,
                segment_key,
                option_value,
                meaning_for(family, pattern, segment_key, option_value, law_titles),
                source_count,
                registry_text_counts.get(reg_key, 0),
                dumps_json(registry_examples.get(reg_key, [])),
                kind,
            )
        )
    conn.executemany(
        """
        INSERT INTO option_registry (
            dataset, family, subfamily, pattern, segment_key, option_value,
            meaning_en, source_count, text_ref_count, examples_json, selector_kind
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        registry_rows,
    )
    conn.commit()

    print("Computing court co-signals from court_considerations.csv text...")
    co_signals, text_keywords = compute_court_co_signals(conn, data_dir)

    co_updates = []
    for (segment_key, option_value), signals in co_signals.items():
        co_updates.append(
            (
                dumps_json(signals),
                dumps_json(text_keywords.get((segment_key, option_value), [])),
                "court_considerations",
                segment_key,
                option_value,
            )
        )
    # Also write empty text_keywords for options that have keywords but no co-signals.
    for (segment_key, option_value), kws in text_keywords.items():
        if (segment_key, option_value) not in co_signals and kws:
            co_updates.append(
                (
                    dumps_json([]),
                    dumps_json(kws),
                    "court_considerations",
                    segment_key,
                    option_value,
                )
            )
    if co_updates:
        conn.executemany(
            """
            UPDATE option_registry
            SET co_signals_json = ?, text_keywords_json = ?
            WHERE dataset = ? AND segment_key = ? AND option_value = ?
            """,
            co_updates,
        )
        conn.commit()

    export_choice_cards(conn, cards_path)
    conn.close()
    print(f"Built v3 index: {db_path}")
    print(f"Saved v3 cards: {cards_path}")


def flush_source_batches(conn: sqlite3.Connection, source_batch: list, segment_batch: list) -> None:
    if source_batch:
        conn.executemany(
            """
            INSERT OR REPLACE INTO source_citations (
                dataset, citation, family, subfamily, pattern, origin_mask, segments_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            source_batch,
        )
        source_batch.clear()
    if segment_batch:
        conn.executemany(
            """
            INSERT INTO citation_segments (
                dataset, citation, family, subfamily, pattern, segment_key, option_value, selector_kind
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            segment_batch,
        )
        segment_batch.clear()
    conn.commit()


def export_choice_cards(conn: sqlite3.Connection, cards_path: Path, limit_per_segment: int = 200) -> None:
    cards = defaultdict(list)
    rows = conn.execute(
        """
        SELECT dataset, family, subfamily, pattern, segment_key, option_value,
               meaning_en, source_count, text_ref_count, examples_json, selector_kind,
               co_signals_json, text_keywords_json
        FROM option_registry
        ORDER BY dataset, family, pattern, segment_key, source_count DESC, option_value
        """
    )
    seen = Counter()
    for row in rows:
        (
            dataset, family, subfamily, pattern, segment_key, option_value, meaning,
            src_count, ref_count, examples, kind, co_signals, text_keywords,
        ) = row
        card_key = f"{dataset}.{family}.{pattern}.{segment_key}"
        if seen[card_key] >= limit_per_segment and kind == "explicit_only":
            continue
        seen[card_key] += 1
        cards[card_key].append(
            {
                "id": option_value,
                "dataset": dataset,
                "family": family,
                "subfamily": subfamily,
                "pattern": pattern,
                "segment_key": segment_key,
                "meaning_en": meaning,
                "source_count": src_count,
                "text_ref_count": ref_count,
                "examples": json.loads(examples),
                "selector_kind": kind,
                "co_signals": json.loads(co_signals or "[]"),
                "text_keywords": json.loads(text_keywords or "[]"),
            }
        )
    cards_path.write_text(dumps_json(cards), encoding="utf-8")


def invalidate_stale_artifacts(art_dir: Path = ART_DIR) -> list[Path]:
    removed = []
    resolved_art = art_dir.resolve()
    for pattern in STALE_ARTIFACT_PATTERNS:
        for path in art_dir.glob(pattern):
            resolved = path.resolve()
            if resolved.parent != resolved_art:
                raise RuntimeError(f"Refusing to remove outside artifact dir: {path}")
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed


FORBIDDEN_LEAK_FILE_PATTERNS = ("_gold_citations.json", "gold_citations.jsonl", "gold_bank")


def assert_no_leak_path(path: Path) -> None:
    name = str(path).lower()
    for needle in FORBIDDEN_LEAK_FILE_PATTERNS:
        if needle in name:
            raise RuntimeError(
                f"Refusing to read potential gold-bank file during candidate generation: {path}"
            )


def load_queries_no_gold(split_csv: Path) -> list[dict]:
    assert_no_leak_path(split_csv)
    rows = []
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # Explicitly drop any gold-like columns so candidate generation
            # cannot accidentally observe them downstream.
            for forbidden in ("gold_citations", "gold", "labels"):
                row.pop(forbidden, None)
            rows.append(
                {
                    "query_id": str(row.get("query_id") or ""),
                    "query": str(row.get("query") or ""),
                }
            )
    return rows


def source_set(conn: sqlite3.Connection, dataset: str, family: str) -> set[str]:
    if family == "court_funnel":
        return {
            row[0]
            for row in conn.execute(
                "SELECT citation FROM source_citations WHERE dataset = ?",
                (dataset,),
            )
        }
    return {
        row[0]
        for row in conn.execute(
            "SELECT citation FROM source_citations WHERE dataset = ? AND family = ?",
            (dataset, family),
        )
    }


def allowed_option(conn: sqlite3.Connection, dataset: str, family: str, segment_key: str, value: str) -> bool:
    if family == "court_funnel":
        row = conn.execute(
            """
            SELECT 1 FROM option_registry
            WHERE dataset = ? AND segment_key = ? AND option_value = ?
            LIMIT 1
            """,
            (dataset, segment_key, value),
        ).fetchone()
        return row is not None
    row = conn.execute(
        """
        SELECT 1 FROM option_registry
        WHERE dataset = ? AND family = ? AND segment_key = ? AND option_value = ?
        LIMIT 1
        """,
        (dataset, family, segment_key, value),
    ).fetchone()
    return row is not None


def derive_decisions(conn: sqlite3.Connection, query: str) -> tuple[list[dict], set[str]]:
    decisions: list[dict] = []
    explicit_citations: set[str] = set()
    refs = extract_references(query)
    for ref in refs:
        if source_contains(conn, ref.citation):
            explicit_citations.add(ref.citation)
        if ref.family == "law":
            code = ref.segments.get("law_code")
            if code and allowed_option(conn, "laws_de", "law", "law_code", str(code)):
                decisions.append(
                    {
                        "funnel": "law",
                        "dataset": "laws_de",
                        "family": "law",
                        "segment_key": "law_code",
                        "selected_option_ids": [str(code)],
                        "mode": "include",
                        "confidence": 1.0,
                        "reason": "explicit law code mention in query",
                    }
                )
        elif ref.pattern == "court_bge":
            division = ref.segments.get("division")
            if division and allowed_option(conn, "court_considerations", "court_funnel", "division", str(division)):
                decisions.append(
                    {
                        "funnel": "court",
                        "dataset": "court_considerations",
                        "family": "court_funnel",
                        "segment_key": "division",
                        "selected_option_ids": [str(division)],
                        "mode": "include",
                        "confidence": 1.0,
                        "reason": "explicit BGE division mention in query",
                    }
                )
        elif ref.pattern == "court_case":
            prefix = docket_prefix(ref.segments)
            if prefix and allowed_option(conn, "court_considerations", "court_funnel", "docket_prefix", prefix):
                decisions.append(
                    {
                        "funnel": "court",
                        "dataset": "court_considerations",
                        "family": "court_funnel",
                        "segment_key": "docket_prefix",
                        "selected_option_ids": [prefix],
                        "mode": "include",
                        "confidence": 1.0,
                        "reason": "explicit docket prefix mention in query",
                    }
                )

    # Direct law-code mentions such as OR, StPO, ZGB may not be attached to an
    # Article mention in English queries; add them without using labels.
    for value, in conn.execute(
        """
        SELECT option_value FROM option_registry
        WHERE dataset = 'laws_de' AND family = 'law' AND segment_key = 'law_code'
          AND selector_kind = 'semantic'
        """
    ):
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])", query):
            decisions.append(
                {
                    "funnel": "law",
                    "dataset": "laws_de",
                    "family": "law",
                    "segment_key": "law_code",
                    "selected_option_ids": [value],
                    "mode": "include",
                    "confidence": 0.95,
                    "reason": "bare law code mention in query",
                }
            )

    # Article-number boosts: extract any "Art. N" / "Artikel N" / bare numbers
    # adjacent to a law context, and emit them as `mode=maybe` so candidates
    # within the law_code base whose `article` segment matches get +100 score.
    # This breaks alphabetical ties when the law_code base is much larger
    # than the budget.
    article_numbers = set()
    for match in re.finditer(r"(?:Art(?:ikel|\.)?|Article|article)\s*\.?\s*(\d{1,4}[A-Za-z]{0,8})", query):
        article_numbers.add(match.group(1))
    if article_numbers:
        decisions.append(
            {
                "funnel": "law",
                "dataset": "laws_de",
                "family": "law",
                "segment_key": "article",
                "selected_option_ids": sorted(article_numbers),
                "mode": "maybe",
                "confidence": 0.7,
                "reason": "article number mentions in query",
            }
        )

    # Court BGE volume mentions and docket prefixes that appear bare in the
    # query (e.g. "BGE 132 I 21" already handled above; this catches
    # references like "1B_15/2023" not attached to a recognised court_case
    # pattern). Adds maybe-boost only.
    for match in re.finditer(r"\b(\d{1,2}[A-Z]{1,2})[._/]\d", query):
        prefix = match.group(1)
        if allowed_option(conn, "court_considerations", "court_funnel", "docket_prefix", prefix):
            decisions.append(
                {
                    "funnel": "court",
                    "dataset": "court_considerations",
                    "family": "court_funnel",
                    "segment_key": "docket_prefix",
                    "selected_option_ids": [prefix],
                    "mode": "maybe",
                    "confidence": 0.6,
                    "reason": "bare docket prefix mention in query",
                }
            )

    return merge_decisions(decisions), explicit_citations


def merge_decisions(decisions: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    for decision in decisions:
        key = (decision["dataset"], decision["family"], decision["segment_key"], decision["mode"])
        if key not in grouped:
            grouped[key] = dict(decision)
            grouped[key]["selected_option_ids"] = []
        grouped[key]["selected_option_ids"].extend(decision.get("selected_option_ids") or [])
    clean = []
    for decision in grouped.values():
        decision["selected_option_ids"] = sorted(set(decision["selected_option_ids"]))
        clean.append(decision)
    return clean


def source_contains(conn: sqlite3.Connection, citation: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM source_citations WHERE citation = ? LIMIT 1",
        (citation,),
    ).fetchone()
    return row is not None


def semantic_include_decisions(conn: sqlite3.Connection, decisions: list[dict], dataset: str, family: str) -> list[dict]:
    out = []
    for decision in decisions:
        if decision.get("dataset") != dataset or decision.get("family") != family:
            continue
        if decision.get("mode") != "include":
            continue
        segment_key = decision.get("segment_key")
        values = decision.get("selected_option_ids") or []
        if not values:
            continue
        if family == "court_funnel":
            row = conn.execute(
                """
                SELECT selector_kind FROM option_registry
                WHERE dataset = ? AND segment_key = ? AND option_value = ?
                LIMIT 1
                """,
                (dataset, segment_key, values[0]),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT selector_kind FROM option_registry
                WHERE dataset = ? AND family = ? AND segment_key = ? AND option_value = ?
                LIMIT 1
                """,
                (dataset, family, segment_key, values[0]),
            ).fetchone()
        if row and row[0] == "semantic":
            out.append(decision)
    return out


def candidates_matching_segment(
    conn: sqlite3.Connection,
    dataset: str,
    family: str,
    segment_key: str,
    values: list[str],
) -> set[str]:
    if not values:
        return set()
    placeholders = ",".join("?" for _ in values)
    if family == "court_funnel":
        params = [dataset, segment_key, *values]
        return {
            row[0]
            for row in conn.execute(
                f"""
                SELECT DISTINCT citation FROM citation_segments
                WHERE dataset = ? AND segment_key = ?
                  AND option_value IN ({placeholders})
                """,
                params,
            )
        }
    params = [dataset, family, segment_key, *values]
    return {
        row[0]
        for row in conn.execute(
            f"""
            SELECT DISTINCT citation FROM citation_segments
            WHERE dataset = ? AND family = ? AND segment_key = ?
              AND option_value IN ({placeholders})
            """,
            params,
        )
    }


def hard_filter_base(conn: sqlite3.Connection, dataset: str, family: str, decisions: list[dict]) -> tuple[set[str], list[dict]]:
    """Hard-filter the source set by include+semantic decisions.

    Court funnel: alternative semantic segments (division applies to BGE rows
    only, docket_prefix applies to court_case rows only). Intersecting them
    drops everything because no row carries both keys. We therefore UNION
    across different segment_keys for the court funnel and then intersect
    that union with the source set.

    Law funnel: we keep AND semantics across segment_keys (rare in practice
    since merge_decisions collapses repeated keys, and law typically only
    fires the law_code key).
    """
    base = source_set(conn, dataset, family)
    stages = [{"stage": "source", "candidate_count": len(base)}]
    relevant = semantic_include_decisions(conn, decisions, dataset, family)
    if not relevant:
        return base, stages

    if family == "court_funnel":
        union: set[str] = set()
        per_key: dict[str, list[str]] = {}
        for decision in relevant:
            matches = candidates_matching_segment(
                conn,
                dataset,
                family,
                decision["segment_key"],
                decision["selected_option_ids"],
            )
            union |= matches
            per_key.setdefault(decision["segment_key"], []).extend(decision["selected_option_ids"])
            stages.append(
                {
                    "stage": f"hard_filter:{decision['segment_key']}",
                    "selected_option_ids": decision["selected_option_ids"],
                    "candidate_count": len(matches),
                    "matches_alone": len(matches),
                }
            )
        base &= union
        stages.append(
            {
                "stage": "hard_filter:court_union",
                "selected_options_by_key": per_key,
                "candidate_count": len(base),
            }
        )
    else:
        for decision in relevant:
            matches = candidates_matching_segment(
                conn,
                dataset,
                family,
                decision["segment_key"],
                decision["selected_option_ids"],
            )
            base &= matches
            stages.append(
                {
                    "stage": f"hard_filter:{decision['segment_key']}",
                    "selected_option_ids": decision["selected_option_ids"],
                    "candidate_count": len(base),
                }
            )
    return base, stages


def topk_with_scores(
    conn: sqlite3.Connection,
    dataset: str,
    family: str,
    base: set[str],
    decisions: list[dict],
    explicit: set[str],
    budget: int,
) -> set[str]:
    scores = Counter()
    for decision in decisions:
        if decision.get("dataset") != dataset or decision.get("family") != family:
            continue
        mode = decision.get("mode")
        weight = 500 if mode == "include" else 100 if mode == "maybe" else 0
        if weight <= 0:
            continue
        matches = candidates_matching_segment(
            conn,
            dataset,
            family,
            decision["segment_key"],
            decision.get("selected_option_ids") or [],
        )
        for citation in matches & base:
            scores[citation] += weight
    for citation in explicit & base:
        scores[citation] += 10_000

    if len(base) <= budget:
        return set(base)
    ranked = sorted(base, key=lambda citation: (-scores[citation], citation))
    return set(ranked[:budget]) | (explicit & base)


def run_candidates(
    split: str,
    data_dir: Path = DATA_DIR,
    db_path: Path = DEFAULT_DB,
    out_dir: Path = ART_DIR,
    law_budget: int = 2_000,
    court_budget: int = 2_000,
    planner_input: Path | None = None,
) -> None:
    conn = sqlite3.connect(db_path)
    queries = load_queries_no_gold(data_dir / f"{split}.csv")
    external_plans = load_external_plans(planner_input) if planner_input else {}
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / f"{split}_segment_lattice_v3_candidate_sets.jsonl"
    plan_path = out_dir / f"{split}_segment_lattice_v3_planner_outputs.jsonl"

    with candidate_path.open("w", encoding="utf-8", newline="\n") as cand_out, plan_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as plan_out:
        for row in queries:
            query_id = row["query_id"]
            query = row["query"]
            if query_id in external_plans:
                decisions = validate_external_decisions(conn, external_plans[query_id])
                _, explicit = derive_decisions(conn, query)
            else:
                decisions, explicit = derive_decisions(conn, query)
            plan = {
                "query_id": query_id,
                "segment_decisions": decisions,
                "explicit_source_mentions": sorted(explicit),
                "law_budget": law_budget,
                "court_budget": court_budget,
                "no_leak": True,
            }
            law_base, law_stages = hard_filter_base(conn, "laws_de", "law", decisions)
            court_base, court_stages = hard_filter_base(conn, "court_considerations", "court_funnel", decisions)
            law_candidates = topk_with_scores(conn, "laws_de", "law", law_base, decisions, explicit, law_budget)
            court_candidates = topk_with_scores(
                conn, "court_considerations", "court_funnel", court_base, decisions, explicit, court_budget
            )
            law_stages.append({"stage": "topk_budget", "candidate_count": len(law_candidates)})
            court_stages.append({"stage": "topk_budget", "candidate_count": len(court_candidates)})
            record = {
                "query_id": query_id,
                "candidate_count": len(law_candidates | court_candidates),
                "law_candidate_count": len(law_candidates),
                "court_candidate_count": len(court_candidates),
                "candidates": sorted(law_candidates | court_candidates),
                "law_candidates": sorted(law_candidates),
                "court_candidates": sorted(court_candidates),
                "stage_diagnostics": {
                    "law": law_stages,
                    "court": court_stages,
                },
                "planner_ref": f"{query_id}",
            }
            plan_out.write(dumps_json(plan) + "\n")
            cand_out.write(dumps_json(record) + "\n")
            print(
                query_id,
                "law=",
                len(law_candidates),
                "court=",
                len(court_candidates),
                "total=",
                record["candidate_count"],
            )
    conn.close()
    print(f"Saved candidates: {candidate_path}")
    print(f"Saved plans     : {plan_path}")


def split_gold(value: str | None) -> list[str]:
    return [squash_ws(part) for part in (value or "").split(";") if squash_ws(part)]


def load_external_plans(path: Path) -> dict[str, list[dict]]:
    plans = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = loads_json(line)
            plans[str(payload["query_id"])] = payload.get("segment_decisions") or payload.get("plan") or []
    return plans


def validate_external_decisions(conn: sqlite3.Connection, decisions: list[dict]) -> list[dict]:
    clean = []
    for decision in decisions:
        dataset = str(decision.get("dataset") or "")
        family = str(decision.get("family") or "")
        segment_key = str(decision.get("segment_key") or "")
        mode = str(decision.get("mode") or "maybe")
        if mode not in {"include", "maybe", "any"}:
            mode = "maybe"
        values = []
        for value in decision.get("selected_option_ids") or []:
            value = str(value)
            if allowed_option(conn, dataset, family, segment_key, value):
                values.append(value)
        if not values and mode != "any":
            continue
        clean.append(
            {
                "funnel": decision.get("funnel") or ("law" if dataset == "laws_de" else "court"),
                "dataset": dataset,
                "family": family,
                "segment_key": segment_key,
                "selected_option_ids": sorted(set(values)),
                "mode": mode,
                "confidence": float(decision.get("confidence") or 0.0),
                "reason": str(decision.get("reason") or "external planner"),
            }
        )
    return merge_decisions(clean)


def load_gold(split_csv: Path) -> dict[str, set[str]]:
    out = {}
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out[str(row.get("query_id") or "")] = set(split_gold(row.get("gold_citations")))
    return out


def citation_family_lookup(conn: sqlite3.Connection) -> dict[str, str]:
    lookup = {}
    for dataset, citation, family in conn.execute("SELECT dataset, citation, family FROM source_citations"):
        lookup[citation] = "law" if dataset == "laws_de" else "court" if dataset == "court_considerations" else family
    return lookup


def audit_candidates(
    split: str,
    data_dir: Path = DATA_DIR,
    db_path: Path = DEFAULT_DB,
    out_dir: Path = ART_DIR,
) -> None:
    conn = sqlite3.connect(db_path)
    gold_by_query = load_gold(data_dir / f"{split}.csv")
    family_by_citation = citation_family_lookup(conn)
    candidate_path = out_dir / f"{split}_segment_lattice_v3_candidate_sets.jsonl"
    plan_path = out_dir / f"{split}_segment_lattice_v3_planner_outputs.jsonl"
    summary_path = out_dir / f"{split}_segment_lattice_v3_candidate_summary.csv"
    dropped_path = out_dir / f"{split}_segment_lattice_v3_dropped_gold.csv"

    plans = {}
    if plan_path.exists():
        with plan_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                plan = loads_json(line)
                plans[plan["query_id"]] = plan

    summaries = []
    dropped_rows = []
    with candidate_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = loads_json(line)
            query_id = record["query_id"]
            gold = gold_by_query.get(query_id, set())
            candidates = set(record["candidates"])
            law_candidates = set(record["law_candidates"])
            court_candidates = set(record["court_candidates"])
            gold_law = {c for c in gold if family_by_citation.get(c) == "law"}
            gold_court = {c for c in gold if family_by_citation.get(c) == "court"}
            summary = {
                "query_id": query_id,
                "candidate_count": len(candidates),
                "law_candidate_count": len(law_candidates),
                "court_candidate_count": len(court_candidates),
                "gold_count": len(gold),
                "recall": recall(candidates, gold),
                "law_recall": recall(law_candidates, gold_law),
                "court_recall": recall(court_candidates, gold_court),
            }
            summaries.append(summary)
            for citation in sorted(gold - candidates):
                failed_stage, failed_options = diagnose_drop(
                    conn,
                    citation,
                    family_by_citation.get(citation),
                    plans.get(query_id, {}).get("segment_decisions", []),
                )
                dropped_rows.append(
                    {
                        "query_id": query_id,
                        "citation": citation,
                        "family": family_by_citation.get(citation, "not_in_source"),
                        "first_failed_stage": failed_stage,
                        "selected_options": failed_options,
                    }
                )

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()) if summaries else ["query_id"])
        writer.writeheader()
        writer.writerows(summaries)
    with dropped_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["query_id", "citation", "family", "first_failed_stage", "selected_options"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dropped_rows)
    conn.close()
    print(f"Saved summary: {summary_path}")
    print(f"Saved dropped: {dropped_path}")


def oracle_plan_for_split(
    split: str,
    data_dir: Path = DATA_DIR,
    db_path: Path = DEFAULT_DB,
    out_dir: Path = ART_DIR,
) -> None:
    """Diagnostic-only: build a perfect-knowledge planner output from gold.

    Reads gold citations, looks up each citation's segments in source_citations,
    and emits one segment_decisions entry per (query, segment) with mode=include.
    Output is for diagnostics; if recall after run-candidates is < 1.0, the
    bottleneck is the funnel/registry, not the LLM planner.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / f"{split}_segment_lattice_v3_oracle_planner_outputs.jsonl"

    conn = sqlite3.connect(db_path)
    gold_by_query = load_gold(data_dir / f"{split}.csv")

    # citation -> {segment_key: option_value}
    seg_lookup: dict[str, dict[str, str]] = defaultdict(dict)
    family_lookup: dict[str, str] = {}
    for citation, family, segment_key, option_value in conn.execute(
        """
        SELECT citation, family, segment_key, option_value
        FROM citation_segments
        WHERE segment_key IN ('law_code', 'division', 'docket_prefix')
        """
    ):
        seg_lookup[citation][segment_key] = option_value
        family_lookup[citation] = family

    plan_rows: list[dict] = []
    for query_id, gold in gold_by_query.items():
        law_codes: set[str] = set()
        divisions: set[str] = set()
        docket_prefixes: set[str] = set()
        for citation in gold:
            segs = seg_lookup.get(citation)
            if not segs:
                continue
            if "law_code" in segs:
                law_codes.add(segs["law_code"])
            if "division" in segs:
                divisions.add(segs["division"])
            if "docket_prefix" in segs:
                docket_prefixes.add(segs["docket_prefix"])

        decisions: list[dict] = []
        if law_codes:
            decisions.append(
                {
                    "funnel": "law",
                    "dataset": "laws_de",
                    "family": "law",
                    "segment_key": "law_code",
                    "selected_option_ids": sorted(law_codes),
                    "mode": "include",
                    "confidence": 1.0,
                    "reason": "oracle: gold law_code union",
                }
            )
        if divisions:
            decisions.append(
                {
                    "funnel": "court",
                    "dataset": "court_considerations",
                    "family": "court_funnel",
                    "segment_key": "division",
                    "selected_option_ids": sorted(divisions),
                    "mode": "include",
                    "confidence": 1.0,
                    "reason": "oracle: gold BGE division union",
                }
            )
        if docket_prefixes:
            decisions.append(
                {
                    "funnel": "court",
                    "dataset": "court_considerations",
                    "family": "court_funnel",
                    "segment_key": "docket_prefix",
                    "selected_option_ids": sorted(docket_prefixes),
                    "mode": "include",
                    "confidence": 1.0,
                    "reason": "oracle: gold docket_prefix union",
                }
            )
        plan_rows.append({"query_id": query_id, "segment_decisions": decisions})

    with plan_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in plan_rows:
            handle.write(dumps_json(row) + "\n")
    conn.close()
    print(f"Saved oracle planner output: {plan_path}")
    print(
        "Next: python scripts/segment_lattice_v3.py run-candidates "
        f"--split {split} --planner-input {plan_path}"
    )
    print(f"Then : python scripts/segment_lattice_v3.py audit --split {split}")


def diagnose_drop(
    conn: sqlite3.Connection,
    citation: str,
    family: str | None,
    decisions: list[dict],
) -> tuple[str, str]:
    if family is None:
        return "source_coverage", ""
    dataset = "laws_de" if family == "law" else "court_considerations"
    decision_family = "law" if family == "law" else "court_funnel"
    for decision in semantic_include_decisions(conn, decisions, dataset, decision_family):
        values = decision.get("selected_option_ids") or []
        if not citation_has_any_segment_value(conn, dataset, citation, decision["segment_key"], values):
            return f"hard_filter:{decision['segment_key']}", ";".join(values)
    return "topk_budget", ""


def citation_has_any_segment_value(
    conn: sqlite3.Connection,
    dataset: str,
    citation: str,
    segment_key: str,
    values: list[str],
) -> bool:
    if not values:
        return True
    placeholders = ",".join("?" for _ in values)
    row = conn.execute(
        f"""
        SELECT 1 FROM citation_segments
        WHERE dataset = ? AND citation = ? AND segment_key = ?
          AND option_value IN ({placeholders})
        LIMIT 1
        """,
        [dataset, citation, segment_key, *values],
    ).fetchone()
    return row is not None


def recall(candidates: set[str], gold: set[str]) -> float | None:
    if not gold:
        return None
    return len(candidates & gold) / len(gold)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-index")
    build.add_argument("--data-dir", type=Path, default=DATA_DIR)
    build.add_argument("--insights-dir", type=Path, default=INSIGHTS_DIR)
    build.add_argument("--db", type=Path, default=DEFAULT_DB)
    build.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    build.add_argument("--force", action="store_true")

    invalidate = sub.add_parser("invalidate-stale")
    invalidate.add_argument("--art-dir", type=Path, default=ART_DIR)

    run = sub.add_parser("run-candidates")
    run.add_argument("--split", choices=["train", "val", "test"], default="val")
    run.add_argument("--data-dir", type=Path, default=DATA_DIR)
    run.add_argument("--db", type=Path, default=DEFAULT_DB)
    run.add_argument("--out-dir", type=Path, default=ART_DIR)
    run.add_argument("--law-budget", type=int, default=2_000)
    run.add_argument("--court-budget", type=int, default=2_000)
    run.add_argument(
        "--planner-input",
        type=Path,
        default=None,
        help="Optional no-gold JSONL with query_id and segment_decisions chosen from choice_cards_v3.json.",
    )

    audit = sub.add_parser("audit")
    audit.add_argument("--split", choices=["train", "val"], default="val")
    audit.add_argument("--data-dir", type=Path, default=DATA_DIR)
    audit.add_argument("--db", type=Path, default=DEFAULT_DB)
    audit.add_argument("--out-dir", type=Path, default=ART_DIR)

    oracle = sub.add_parser(
        "run-oracle",
        help="Diagnostic-only: build perfect-knowledge planner output from gold (no candidate selection).",
    )
    oracle.add_argument("--split", choices=["train", "val"], default="val")
    oracle.add_argument("--data-dir", type=Path, default=DATA_DIR)
    oracle.add_argument("--db", type=Path, default=DEFAULT_DB)
    oracle.add_argument("--out-dir", type=Path, default=ART_DIR)

    args = parser.parse_args(argv)
    if args.command == "build-index":
        build_index(args.data_dir, args.insights_dir, args.db, args.cards, force=args.force)
    elif args.command == "invalidate-stale":
        removed = invalidate_stale_artifacts(args.art_dir)
        print("Removed stale artifacts:")
        for path in removed:
            print(path)
    elif args.command == "run-candidates":
        run_candidates(
            args.split,
            args.data_dir,
            args.db,
            args.out_dir,
            args.law_budget,
            args.court_budget,
            args.planner_input,
        )
    elif args.command == "audit":
        audit_candidates(args.split, args.data_dir, args.db, args.out_dir)
    elif args.command == "run-oracle":
        oracle_plan_for_split(args.split, args.data_dir, args.db, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
