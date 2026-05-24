#!/usr/bin/env python
"""Extract Swiss legal citation graphs from the project CSV files.

The script streams the large court CSV, stores unique citations and unique
source->reference edges in SQLite, and then exports compact JSON artifacts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ORIGIN_SOURCE = 1
ORIGIN_REFERENCE = 2
EXTRACTOR_VERSION = "segment_lattice_v3_parser_2026_04_27"

DATASETS = {
    "laws_de": "laws_de.csv",
    "court_considerations": "court_considerations.csv",
}

ORDINAL_SUFFIXES = (
    "bis",
    "ter",
    "quater",
    "quinquies",
    "sexies",
    "septies",
    "octies",
    "novies",
    "decies",
)

STOP_CODE_WORDS = {
    "",
    "Abs",
    "Absatz",
    "al",
    "lit",
    "Bst",
    "Buchstabe",
    "Buchstaben",
    "Ziff",
    "Ziffer",
    "Satz",
    "Nr",
    "Nummer",
    "Nummern",
    "Unterabsatz",
    "Unterabs",
    "und",
    "oder",
    "and",
    "or",
    "des",
    "der",
    "dem",
    "den",
    "die",
    "das",
    "du",
    "de",
    "della",
    "delle",
    "dell",
    "vom",
    "von",
    "in",
    "im",
    "i",
    "i.V.m",
    "vgl",
    "sowie",
    "mit",
    "nach",
    "zu",
    "zur",
    "zum",
    "ff",
    "f",
}

MONTHS = {
    "januar": "01",
    "janvier": "01",
    "gennaio": "01",
    "februar": "02",
    "fevrier": "02",
    "febbraio": "02",
    "marz": "03",
    "mars": "03",
    "marzo": "03",
    "april": "04",
    "avril": "04",
    "aprile": "04",
    "mai": "05",
    "maggio": "05",
    "juni": "06",
    "juin": "06",
    "giugno": "06",
    "juli": "07",
    "juillet": "07",
    "luglio": "07",
    "august": "08",
    "aout": "08",
    "agosto": "08",
    "september": "09",
    "septembre": "09",
    "settembre": "09",
    "oktober": "10",
    "octobre": "10",
    "ottobre": "10",
    "november": "11",
    "novembre": "11",
    "dezember": "12",
    "decembre": "12",
    "dicembre": "12",
}


ARTICLE_HEAD_RE = re.compile(r"\b(?:Art\.|Artikel)\s+", re.IGNORECASE)
ARTICLE_TOKEN = r"\d+[a-z]*"
ARTICLE_SERIES_RE = re.compile(
    rf"(?P<series>{ARTICLE_TOKEN}(?:(?:\s*(?:,|und|oder|and|/)\s*){ARTICLE_TOKEN})*"
    r"|"
    rf"{ARTICLE_TOKEN}\s*(?:-|bis|to)\s*{ARTICLE_TOKEN})"
    r"(?:\s*(?P<ff>ff\.|f\.))?",
    re.IGNORECASE,
)
UNIT_RE = re.compile(
    r"^\s*(?P<label>Unterabsatz|Unterabs\.?|Abs(?:a|ä)tze|Absatz|Abs\.?|al\.?|Buchstabe(?:n)?|Bst\.?|lit\.?|"
    r"Ziffer|Ziff\.?|Nummer(?:n)?|Nr\.?|Satz)\s*"
    r"(?P<value>[0-9]+[a-z]*|[a-z]|[IVX]+)"
    r"(?P<extra>(?:\s*(?:,|und|oder|and|/)\s*(?:[0-9]+[a-z]*|[a-z]|[IVX]+))*)",
    re.IGNORECASE,
)
UNIT_LABEL_RE = re.compile(
    r"^\s*(?P<label>Unterabsatz|Unterabs\.?|Abs(?:a|ä)tze|Absatz|Abs\.?|al\.?|Buchstabe(?:n)?|Bst\.?|lit\.?|"
    r"Ziffer|Ziff\.?|Nummer(?:n)?|Nr\.?|Satz)",
    re.IGNORECASE,
)
CODE_RE = re.compile(
    r"^\s*(?:\(|\[)?(?P<code>"
    r"SR\s+\d+(?:\.\d+)*"
    r"|"
    r"\d{2,}(?:\.\d+)*"
    r"|"
    r"[A-Z][A-Za-z0-9.\-/]*[A-Za-z0-9)]"
    r"(?:\s+(?:[IVX]{1,4}|\d+))?"
    r")",
)
PAREN_CODE_RE = re.compile(
    r"[\(\[](?P<code>[A-Z][A-Za-z0-9.\-/]{1,30}(?:\s+(?:[IVX]{1,4}|\d+))?)"
    r"\s*(?:;|,)\s*SR\s+(?P<sr>\d+(?:\.\d+)*)",
)
INLINE_SR_RE = re.compile(r"\bSR\s+(?P<sr>\d+(?:\.\d+)*)")

SECTION_RE = re.compile(
    r"(?<!\S)(?P<prefix>§{1,2})\s*"
    r"(?P<section>\d+(?:\.\d+)*(?:[a-z])?(?:\s*(?:-|bis|to)\s*\d+(?:\.\d+)*)?)"
    r"(?P<body>.{0,120})",
    re.IGNORECASE | re.DOTALL,
)

BGE_RE = re.compile(
    r"\b(?P<reporter>BGE|ATF|DTF)\s+"
    r"(?P<volume>\d{3})\s+"
    r"(?P<division>[IVX]{1,4})\s+"
    r"(?P<page>\d+[a-z]?)"
    r"(?:\s+(?P<unit>E\.|S\.|consid\.|cons\.|c\.)\s*(?P<pinpoint>[A-Za-zIVX0-9]+(?:[./-][A-Za-z0-9]+)*(?:\s*f{1,2}\.)?))?",
)

SOURCE_ARTICLE_RE = re.compile(
    rf"^Art\.\s+(?P<article>{ARTICLE_TOKEN})"
    r"(?:\s+Abs\.\s+(?P<paragraph>\d+[a-z]*))?"
    r"\s+(?P<law_code>.+)$",
    re.IGNORECASE,
)

MODERN_CASE_RE = re.compile(
    r"\b(?P<docket>\d{1,2}[A-Z]{1,2}[_\. ]\d{1,5}/\d{4})\b"
    r"(?P<trailing>.{0,90})",
    re.DOTALL,
)
LEGACY_CASE_RE = re.compile(
    r"\b(?P<docket>[A-Z]\s+\d{1,5}/\d{2})\b"
    r"(?P<trailing>.{0,90})",
    re.DOTALL,
)
NUMERIC_DATE_RE = re.compile(r"^\s*(?P<date>\d{2}\.\d{2}\.\d{4})")
WORD_DATE_RE = re.compile(
    r"^\s*(?:vom|du|del|of)?\s*(?P<day>\d{1,2})\.\s+"
    r"(?P<month>[A-Za-z.\u00c0-\u017f]+)\s+(?P<year>\d{4})",
    re.IGNORECASE,
)
CONSIDERATION_RE = re.compile(
    r"\b(?:consid\.|cons\.|E\.|c\.)\s*(?P<consideration>[A-Za-zIVX0-9]+(?:[./-][A-Za-z0-9]+)*)"
)
LOOSE_CONSIDERATION_RE = re.compile(
    r"\b(?:consid\.|cons\.|E\.|c\.)\s*(?P<consideration>[^\s,;)]*)"
)
COMPACT_SLASH_CASE_RE = re.compile(
    r"^(?P<prefix>\d{1,2}[A-Z]{1,2})/(?P<serial>\d{1,6})(?P<trailing>.*)$"
)

OFFICIAL_REFS = [
    (
        "systematic_collection",
        re.compile(r"\bSR\s+(?P<number>\d+(?:\.\d+)*)"),
        "SR {number}",
    ),
    (
        "federal_gazette",
        re.compile(r"\bBBl\s+(?P<year>\d{4})\s+(?P<page>\d+)"),
        "BBl {year} {page}",
    ),
    (
        "official_collection",
        re.compile(r"\bAS\s+(?P<year>\d{4})\s+(?P<page>\d+)"),
        "AS {year} {page}",
    ),
]

PATTERN_DESCRIPTIONS = {
    "statute_article": {
        "regex_family": "Art./Artikel + article + optional Abs./lit./Ziff./Satz + law code",
        "segments": [
            "article_marker",
            "article_number",
            "article_suffix_or_range",
            "paragraph_marker",
            "paragraph_number",
            "subdivision_marker",
            "subdivision_value",
            "law_code",
        ],
        "examples": [
            "Art. 221 Abs. 1 lit. b StPO",
            "Art. 69 Abs. 1bis IVG",
            "Artikel 10 TSchG",
            "Art. 30-39 EpG",
        ],
    },
    "statute_section": {
        "regex_family": "section sign + section number + optional Abs./lit. + code",
        "segments": [
            "section_marker",
            "section_number",
            "paragraph_marker",
            "paragraph_number",
            "subdivision_marker",
            "subdivision_value",
            "law_code",
        ],
        "examples": ["§ 24a Abs. 1 SHG", "§§ 25 ff. PBG/SZ", "§ 1.22"],
    },
    "court_bge": {
        "regex_family": "BGE + volume + roman division + first page + optional E./S. pinpoint",
        "segments": ["reporter", "volume", "division", "page", "pinpoint_unit", "pinpoint"],
        "examples": ["BGE 137 IV 122 E. 6.2", "BGE 139 I 2 S. 7"],
    },
    "court_case": {
        "regex_family": "federal docket + optional date + optional E. consideration",
        "segments": [
            "court_chamber",
            "legal_area_code",
            "separator_style",
            "serial_number",
            "decision_year",
            "decision_date",
            "consideration",
        ],
        "examples": [
            "1B_210/2023 E. 4.1",
            "2P.198/2006 09.05.2007 E. 2",
            "I 402/02 13.11.2002 E. 4",
        ],
    },
    "official_reference": {
        "regex_family": "SR/BBl/AS + numeric publication reference",
        "segments": ["publication", "number_or_year", "page"],
        "examples": ["SR 455", "BBl 2009 3547", "AS 2011 1199"],
    },
}


@dataclass(frozen=True)
class Citation:
    citation: str
    family: str
    subfamily: str
    pattern: str
    segments: dict


def squash_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def strip_edge_punctuation(value: str) -> str:
    return value.strip(" \t\r\n,;:.)]}")


def ascii_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def normalize_law_code(code: str | None) -> str | None:
    if not code:
        return None
    code = squash_ws(strip_edge_punctuation(code))
    if code.startswith("SR "):
        return code[3:].strip()
    if code in STOP_CODE_WORDS:
        return None
    bare = code.rstrip(".")
    first_token = bare.split()[0].rstrip(".") if bare.split() else ""
    if first_token in STOP_CODE_WORDS:
        return None
    if bare in STOP_CODE_WORDS:
        return None
    return bare


def law_code_subfamily(code: str | None) -> str:
    if not code:
        return "unresolved_law_code"
    if re.fullmatch(r"\d+(?:\.\d+)*", code):
        return "systematic_collection_number"
    if "/" in code:
        return "cantonal_or_slash_abbreviation"
    if "-" in code:
        return "compound_abbreviation"
    if re.search(r"\s+\d+$", code):
        return "numbered_abbreviation"
    if re.fullmatch(r"[A-Z]{2,6}", code):
        return "uppercase_abbreviation"
    return "mixedcase_abbreviation"


def split_article_series(series: str) -> tuple[list[str], str | None]:
    series = squash_ws(series.replace("\u2013", "-").replace("\u2014", "-"))
    range_match = re.fullmatch(r"(\d+[a-z]*)\s*(?:-|bis|to)\s*(\d+[a-z]*)", series, re.I)
    if range_match:
        return [f"{range_match.group(1)}-{range_match.group(2)}"], "range"
    if re.search(r"\b(?:und|oder|and)\b|,|/", series, re.I):
        parts = [
            p.strip()
            for p in re.split(r"\s*(?:,|/|\bund\b|\boder\b|\band\b)\s*", series, flags=re.I)
            if p.strip()
        ]
        return parts, "list"
    return [series], None


def parse_units(rest: str) -> tuple[list[dict], str]:
    units: list[dict] = []
    while True:
        label_match = UNIT_LABEL_RE.match(rest)
        if not label_match:
            break
        label = label_match.group("label").rstrip(".")
        if label.lower().startswith(("abs", "al")):
            canonical = "Abs."
            key = "paragraph"
            value_pattern = r"\d+(?:(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)|[a-z])?"
        elif label.lower().startswith(("lit", "bst", "buchst")):
            canonical = "Bst."
            key = "letter"
            value_pattern = r"[a-z]"
        elif label.lower().startswith("unterabs"):
            canonical = "Unterabs."
            key = "subparagraph"
            value_pattern = r"\d+(?:(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)|[a-z])?"
        elif label.lower().startswith(("ziff", "nummer", "nr")):
            canonical = "Ziff."
            key = "number"
            value_pattern = r"\d+(?:(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)|[a-z])?"
        else:
            canonical = "Satz"
            key = "sentence"
            value_pattern = r"\d+(?:(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)|[a-z])?"
        value_re = re.compile(
            rf"\s+(?P<value>{value_pattern})(?P<extra>(?:\s*(?:,|und|oder|and|/)\s*{value_pattern})*)",
            re.IGNORECASE,
        )
        value_match = value_re.match(rest[label_match.end() :])
        if not value_match:
            break
        value = value_match.group("value")
        extra = squash_ws(value_match.group("extra") or "")
        raw_value = value + (f" {extra}" if extra else "")
        units.append(
            {
                "marker": canonical,
                "value": raw_value,
                "category": key,
            }
        )
        rest = rest[label_match.end() + value_match.end() :]
    return units, rest


def code_from_rest(rest: str) -> str | None:
    match = CODE_RE.match(rest)
    if not match:
        return None
    code = normalize_law_code(match.group("code"))
    if not code:
        return None
    if code.endswith("("):
        code = code[:-1]
    return normalize_law_code(code)


def code_from_lookahead(window: str) -> str | None:
    match = PAREN_CODE_RE.search(window)
    if match:
        return normalize_law_code(match.group("code"))
    match = INLINE_SR_RE.search(window)
    if match:
        return normalize_law_code(match.group("sr"))
    return None


def format_units(units: list[dict]) -> str:
    if not units:
        return ""
    return " " + " ".join(f"{unit['marker']} {unit['value']}" for unit in units)


def make_statute_article(
    article: str,
    units: list[dict],
    law_code: str | None,
    resolution: str | None,
    series_kind: str | None,
    ff_marker: str | None,
) -> Citation:
    article_text = article
    if ff_marker:
        article_text = f"{article_text} {ff_marker}"
    normalized = f"Art. {article_text}{format_units(units)}"
    if law_code:
        normalized = f"{normalized} {law_code}"
    segments = {
        "article_marker": "Art.",
        "article": article,
        "article_series_kind": series_kind,
        "article_continuation": ff_marker,
        "units": units,
        "law_code": law_code,
        "law_code_family": law_code_subfamily(law_code),
        "law_code_resolution": resolution,
    }
    return Citation(
        citation=squash_ws(normalized),
        family="law",
        subfamily="statute_article",
        pattern="statute_article",
        segments=segments,
    )


def extract_statute_articles(text: str, source_law_code: str | None = None) -> list[Citation]:
    citations: list[Citation] = []
    for head in ARTICLE_HEAD_RE.finditer(text):
        window = squash_ws(text[head.end() : head.end() + 260])
        series_match = ARTICLE_SERIES_RE.match(window)
        if not series_match:
            continue
        series = series_match.group("series")
        ff_marker = series_match.group("ff")
        rest = window[series_match.end() :]
        units, rest_after_units = parse_units(rest)
        code = code_from_rest(rest_after_units)
        resolution = "inline"
        if not code:
            code = code_from_lookahead(window)
            resolution = "nearby_parenthetical_or_sr" if code else None
        if not code and source_law_code:
            code = source_law_code
            resolution = "source_law_context"

        articles, series_kind = split_article_series(series)
        for article in articles:
            citations.append(make_statute_article(article, units, code, resolution, series_kind, ff_marker))
    return citations


def extract_sections(text: str) -> list[Citation]:
    citations: list[Citation] = []
    for match in SECTION_RE.finditer(text):
        prefix = match.group("prefix")
        section = squash_ws(match.group("section").replace("\u2013", "-").replace("\u2014", "-"))
        body = squash_ws(match.group("body"))
        units, rest = parse_units(body)
        code = code_from_rest(rest)
        if not code:
            code = code_from_lookahead(body)
        normalized = f"{prefix} {section}{format_units(units)}"
        if code:
            normalized = f"{normalized} {code}"
        citations.append(
            Citation(
                citation=squash_ws(normalized),
                family="law",
                subfamily="statute_section",
                pattern="statute_section",
                segments={
                    "section_marker": prefix,
                    "section": section,
                    "units": units,
                    "law_code": code,
                    "law_code_family": law_code_subfamily(code),
                },
            )
        )
    return citations


def extract_bge(text: str) -> list[Citation]:
    citations: list[Citation] = []
    for match in BGE_RE.finditer(text):
        unit = match.group("unit")
        pinpoint = match.group("pinpoint")
        # ATF (French) and DTF (Italian) are aliases for the German BGE
        # reporter of the Swiss Federal Court official collection.
        # Canonicalize to BGE so the same decision in different languages
        # collapses to one node in the citation graph.
        reporter = "BGE"
        # Canonicalize French (c.) and Italian (consid./cons.) pinpoint
        # markers to the German E. so consideration sub-keys also dedupe
        # across languages.
        if unit in ("c.", "consid.", "cons."):
            unit_canonical = "E."
        else:
            unit_canonical = unit
        normalized = f"{reporter} {match.group('volume')} {match.group('division')} {match.group('page')}"
        if unit_canonical and pinpoint:
            normalized = f"{normalized} {unit_canonical} {strip_edge_punctuation(pinpoint)}"
        citations.append(
            Citation(
                citation=squash_ws(normalized),
                family="court",
                subfamily="bge",
                pattern="court_bge",
                segments={
                    "reporter": reporter,
                    "volume": match.group("volume"),
                    "division": match.group("division"),
                    "page": match.group("page"),
                    "pinpoint_unit": unit_canonical[:-1] if unit_canonical else None,
                    "pinpoint": strip_edge_punctuation(pinpoint) if pinpoint else None,
                },
            )
        )
    return citations


def parse_word_date(trailing: str) -> tuple[str | None, int]:
    numeric = NUMERIC_DATE_RE.match(trailing)
    if numeric:
        return numeric.group("date"), numeric.end()
    word = WORD_DATE_RE.match(trailing)
    if not word:
        return None, 0
    month_key = ascii_fold(word.group("month").rstrip("."))
    month = MONTHS.get(month_key)
    if not month:
        return None, 0
    day = int(word.group("day"))
    return f"{day:02d}.{month}.{word.group('year')}", word.end()


def extract_consideration(trailing: str) -> str | None:
    match = CONSIDERATION_RE.search(trailing[:90])
    if not match:
        return None
    return strip_edge_punctuation(match.group("consideration"))


def extract_loose_consideration(trailing: str) -> tuple[str | None, str | None]:
    match = LOOSE_CONSIDERATION_RE.search(trailing[:90])
    if not match:
        return None, None
    raw = match.group("consideration")
    if raw == "":
        return None, ""
    normalized = re.sub(r"[^A-Za-z0-9]+$", "", strip_edge_punctuation(raw))
    return normalized or None, raw


def classify_docket(docket: str) -> tuple[str, dict]:
    modern = re.fullmatch(
        r"(?P<chamber>\d{1,2})(?P<area>[A-Z]{1,2})(?P<sep>[_\. ])(?P<serial>\d{1,5})/(?P<year>\d{4})",
        docket,
    )
    if modern:
        sep = modern.group("sep")
        if sep == "_":
            subfamily = "federal_tribunal_modern_docket"
            separator_style = "underscore"
        elif sep == ".":
            subfamily = "federal_tribunal_dot_docket"
            separator_style = "dot"
        else:
            # space-separated French/Italian variant ("5A 800/2019") -- treat
            # as the same family as the underscore form so cross-language
            # citations dedupe.
            subfamily = "federal_tribunal_modern_docket"
            separator_style = "space"
        return (
            subfamily,
            {
                "docket": docket,
                "court_chamber": modern.group("chamber"),
                "legal_area_code": modern.group("area"),
                "separator_style": separator_style,
                "serial_number": modern.group("serial"),
                "decision_year": modern.group("year"),
            },
        )
    legacy = re.fullmatch(r"(?P<area>[A-Z])\s+(?P<serial>\d{1,5})/(?P<year>\d{2})", docket)
    if legacy:
        return (
            "federal_tribunal_legacy_single_letter_docket",
            {
                "docket": docket,
                "court_chamber": None,
                "legal_area_code": legacy.group("area"),
                "separator_style": "space",
                "serial_number": legacy.group("serial"),
                "decision_year": legacy.group("year"),
            },
        )
    return ("unknown_docket", {"docket": docket})


def extract_cases_with_regex(text: str, regex: re.Pattern[str]) -> list[Citation]:
    citations: list[Citation] = []
    for match in regex.finditer(text):
        docket_raw = squash_ws(match.group("docket"))
        # Canonicalize the modern federal tribunal docket so the
        # French/Italian space-separated form "5A 800/2019" collapses
        # to the German "5A_800/2019" form for graph deduplication.
        docket = re.sub(
            r"^(\d{1,2}[A-Z]{1,2}) (\d{1,5}/\d{4})$", r"\1_\2", docket_raw
        )
        trailing = match.group("trailing") or ""
        date, consumed = parse_word_date(trailing)
        consideration = extract_consideration(trailing[consumed:])
        normalized = docket
        if date:
            normalized = f"{normalized} {date}"
        if consideration:
            normalized = f"{normalized} E. {consideration}"
        subfamily, segments = classify_docket(docket_raw)
        segments = dict(segments)
        segments["docket"] = docket
        segments.update({"decision_date": date, "consideration": consideration})
        citations.append(
            Citation(
                citation=squash_ws(normalized),
                family="court",
                subfamily=subfamily,
                pattern="court_case",
                segments=segments,
            )
        )
    return citations


def extract_cases(text: str) -> list[Citation]:
    return extract_cases_with_regex(text, MODERN_CASE_RE) + extract_cases_with_regex(text, LEGACY_CASE_RE)


def fallback_unknown_court_segments(value: str) -> dict:
    segments: dict = {"raw": value}
    case_refs = extract_cases(value)
    if case_refs:
        parsed = case_refs[0]
        segments.update(parsed.segments)
        prefix = None
        if segments.get("court_chamber") and segments.get("legal_area_code"):
            prefix = f"{segments['court_chamber']}{segments['legal_area_code']}"
        elif segments.get("legal_area_code"):
            prefix = str(segments["legal_area_code"])
        if prefix:
            segments["docket_prefix"] = prefix
        _, loose_raw = extract_loose_consideration(value)
        if loose_raw is not None:
            loose_norm, loose_raw = extract_loose_consideration(value)
            segments["consideration"] = loose_norm
            segments["consideration_raw"] = loose_raw
        segments["fallback_subfamily"] = parsed.subfamily
        segments["fallback_pattern"] = parsed.pattern
        return segments

    compact = COMPACT_SLASH_CASE_RE.match(value)
    if compact:
        prefix = compact.group("prefix")
        chamber = re.match(r"\d{1,2}", prefix)
        area = prefix[len(chamber.group(0)) :] if chamber else prefix
        consideration, consideration_raw = extract_loose_consideration(compact.group("trailing") or "")
        segments.update(
            {
                "docket_prefix": prefix,
                "court_chamber": chamber.group(0) if chamber else None,
                "legal_area_code": area or None,
                "separator_style": "slash_compact",
                "serial_number": compact.group("serial"),
                "consideration": consideration,
                "consideration_raw": consideration_raw,
                "fallback_subfamily": "compact_slash_docket",
                "fallback_pattern": "court_case",
            }
        )
    return segments


def extract_official_refs(text: str) -> list[Citation]:
    citations: list[Citation] = []
    for subfamily, regex, template in OFFICIAL_REFS:
        for match in regex.finditer(text):
            values = match.groupdict()
            normalized = template.format(**values)
            citations.append(
                Citation(
                    citation=squash_ws(normalized),
                    family="official_reference",
                    subfamily=subfamily,
                    pattern="official_reference",
                    segments=values,
                )
            )
    return citations


def extract_references(text: str, source_law_code: str | None = None) -> list[Citation]:
    all_refs: list[Citation] = []
    all_refs.extend(extract_statute_articles(text, source_law_code=source_law_code))
    all_refs.extend(extract_sections(text))
    all_refs.extend(extract_bge(text))
    all_refs.extend(extract_cases(text))
    all_refs.extend(extract_official_refs(text))

    deduped: dict[str, Citation] = {}
    for citation in all_refs:
        if citation.citation:
            deduped.setdefault(citation.citation, citation)
    return list(deduped.values())


def classify_source_citation(value: str) -> Citation:
    value = squash_ws(value)
    if value.startswith("Art. "):
        refs = extract_statute_articles(value)
        if refs:
            parsed = next((ref for ref in refs if ref.citation == value), refs[0])
            return Citation(
                citation=value,
                family=parsed.family,
                subfamily=parsed.subfamily,
                pattern=parsed.pattern,
                segments=parsed.segments,
            )
        source_match = SOURCE_ARTICLE_RE.fullmatch(value)
        if source_match:
            units = []
            if source_match.group("paragraph"):
                units.append(
                    {
                        "marker": "Abs.",
                        "value": source_match.group("paragraph"),
                        "category": "paragraph",
                    }
                )
            law_code = normalize_law_code(source_match.group("law_code"))
            return Citation(
                citation=value,
                family="law",
                subfamily="statute_article",
                pattern="statute_article",
                segments={
                    "article_marker": "Art.",
                    "article": source_match.group("article"),
                    "article_series_kind": None,
                    "article_continuation": None,
                    "units": units,
                    "law_code": law_code,
                    "law_code_family": law_code_subfamily(law_code),
                    "law_code_resolution": "source_column",
                },
            )
    if value.startswith("BGE "):
        refs = extract_bge(value)
        if refs:
            return refs[0]
    case_refs = extract_cases(value)
    for ref in case_refs:
        if ref.citation == value:
            return ref
    return Citation(
        citation=value,
        family="unknown",
        subfamily="unknown",
        pattern="unknown",
        segments=fallback_unknown_court_segments(value),
    )


def source_law_code_from_citation(value: str) -> str | None:
    classified = classify_source_citation(value)
    if classified.family == "law":
        return classified.segments.get("law_code")
    return None


def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    # MEMORY avoids sidecar journal/WAL files, which is more robust in this
    # workspace sandbox on Windows.
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS citations (
            dataset TEXT NOT NULL,
            citation TEXT NOT NULL,
            origin_mask INTEGER NOT NULL,
            family TEXT NOT NULL,
            subfamily TEXT NOT NULL,
            pattern TEXT NOT NULL,
            segments_json TEXT NOT NULL,
            PRIMARY KEY (dataset, citation)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS edges (
            dataset TEXT NOT NULL,
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            PRIMARY KEY (dataset, source, target)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_dataset_source ON edges(dataset, source)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_citations_dataset ON citations(dataset, citation)")
    return conn


def reset_dataset(conn: sqlite3.Connection, dataset: str) -> None:
    conn.execute("DELETE FROM edges WHERE dataset = ?", (dataset,))
    conn.execute("DELETE FROM citations WHERE dataset = ?", (dataset,))
    conn.commit()


def upsert_citation_rows(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO citations (
            dataset, citation, origin_mask, family, subfamily, pattern, segments_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(dataset, citation) DO UPDATE SET
            origin_mask = citations.origin_mask | excluded.origin_mask
        """,
        rows,
    )


def insert_edge_rows(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    if not rows:
        return
    conn.executemany(
        "INSERT OR IGNORE INTO edges (dataset, source, target) VALUES (?, ?, ?)",
        rows,
    )


def citation_to_row(dataset: str, citation: Citation, origin_mask: int) -> tuple:
    return (
        dataset,
        citation.citation,
        origin_mask,
        citation.family,
        citation.subfamily,
        citation.pattern,
        json.dumps(citation.segments, ensure_ascii=False, sort_keys=True),
    )


def process_dataset(
    conn: sqlite3.Connection,
    dataset: str,
    csv_path: Path,
    batch_size: int,
) -> dict:
    reset_dataset(conn, dataset)
    row_count = 0
    source_counter = Counter()
    reference_counter = Counter()
    pattern_counter = Counter()
    family_counter = Counter()
    source_unique = set() if dataset == "laws_de" else None
    ref_unique = set() if dataset == "laws_de" else None
    citation_rows: list[tuple] = []
    edge_rows: list[tuple] = []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_count += 1
            source_raw = squash_ws(row.get("citation", ""))
            text = row.get("text", "") or ""
            if not source_raw:
                continue

            source = classify_source_citation(source_raw)
            source_counter[source.citation] += 1
            family_counter[source.family] += 1
            pattern_counter[source.pattern] += 1
            if source_unique is not None:
                source_unique.add(source.citation)
            citation_rows.append(citation_to_row(dataset, source, ORIGIN_SOURCE))

            source_law_code = source.segments.get("law_code") if source.family == "law" else None
            refs = extract_references(text, source_law_code=source_law_code)
            for ref in refs:
                if ref.citation == source.citation:
                    continue
                reference_counter[ref.citation] += 1
                family_counter[ref.family] += 1
                pattern_counter[ref.pattern] += 1
                if ref_unique is not None:
                    ref_unique.add(ref.citation)
                citation_rows.append(citation_to_row(dataset, ref, ORIGIN_REFERENCE))
                edge_rows.append((dataset, source.citation, ref.citation))

            if len(citation_rows) >= batch_size or len(edge_rows) >= batch_size:
                upsert_citation_rows(conn, citation_rows)
                insert_edge_rows(conn, edge_rows)
                conn.commit()
                citation_rows.clear()
                edge_rows.clear()

            if dataset == "court_considerations" and row_count % 100000 == 0:
                print(
                    f"{dataset}: scanned {row_count:,} rows; "
                    f"pending citations={len(citation_rows):,}; pending edges={len(edge_rows):,}",
                    flush=True,
                )

    upsert_citation_rows(conn, citation_rows)
    insert_edge_rows(conn, edge_rows)
    conn.commit()

    db_counts = get_dataset_counts(conn, dataset)
    summary = {
        "dataset": dataset,
        "source_file": str(csv_path),
        "rows": row_count,
        "source_unique_count": len(source_counter) if dataset != "laws_de" else len(source_unique or []),
        "reference_unique_count": len(reference_counter) if dataset != "laws_de" else len(ref_unique or []),
        "all_unique_count": db_counts["all_unique_count"],
        "edge_count": db_counts["edge_count"],
        "family_counts": dict(family_counter),
        "pattern_counts": dict(pattern_counter),
    }
    return summary


def get_dataset_counts(conn: sqlite3.Connection, dataset: str) -> dict:
    source_unique_count = conn.execute(
        "SELECT COUNT(*) FROM citations WHERE dataset = ? AND (origin_mask & ?) != 0",
        (dataset, ORIGIN_SOURCE),
    ).fetchone()[0]
    reference_unique_count = conn.execute(
        "SELECT COUNT(*) FROM citations WHERE dataset = ? AND (origin_mask & ?) != 0",
        (dataset, ORIGIN_REFERENCE),
    ).fetchone()[0]
    all_unique_count = conn.execute(
        "SELECT COUNT(*) FROM citations WHERE dataset = ?",
        (dataset,),
    ).fetchone()[0]
    edge_count = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE dataset = ?",
        (dataset,),
    ).fetchone()[0]
    return {
        "source_unique_count": source_unique_count,
        "reference_unique_count": reference_unique_count,
        "all_unique_count": all_unique_count,
        "edge_count": edge_count,
    }


def stream_json_string_array(handle, iterator: Iterable[str], indent: str = "    ") -> int:
    count = 0
    handle.write("[\n")
    first = True
    for value in iterator:
        if not first:
            handle.write(",\n")
        handle.write(indent)
        handle.write(json.dumps(value, ensure_ascii=False))
        first = False
        count += 1
    handle.write("\n  ]")
    return count


def export_unique_json(conn: sqlite3.Connection, dataset: str, output_path: Path, summary: dict) -> None:
    counts = get_dataset_counts(conn, dataset)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("{\n")
        metadata = {
            "dataset": dataset,
            "source_file": summary["source_file"],
            "rows": summary["rows"],
            **counts,
            "origin_mask": {
                "1": "citation column",
                "2": "text reference",
                "3": "both citation column and text reference",
            },
        }
        handle.write('  "metadata": ')
        handle.write(json.dumps(metadata, ensure_ascii=False, indent=2).replace("\n", "\n  "))
        handle.write(",\n")
        handle.write('  "citations": ')
        rows = conn.execute(
            "SELECT citation FROM citations WHERE dataset = ? ORDER BY citation",
            (dataset,),
        )
        stream_json_string_array(handle, (row[0] for row in rows))
        handle.write("\n}\n")


def export_links_json(conn: sqlite3.Connection, dataset: str, output_path: Path, summary: dict) -> None:
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("{\n")
        metadata = {
            "dataset": dataset,
            "source_file": summary["source_file"],
            "edge_count": get_dataset_counts(conn, dataset)["edge_count"],
            "shape": "grouped source citation to unique text reference citations",
        }
        handle.write('  "metadata": ')
        handle.write(json.dumps(metadata, ensure_ascii=False, indent=2).replace("\n", "\n  "))
        handle.write(",\n")
        handle.write('  "source_to_references": [\n')
        rows = conn.execute(
            "SELECT source, target FROM edges WHERE dataset = ? ORDER BY source, target",
            (dataset,),
        )
        current_source = None
        current_refs: list[str] = []
        first_group = True

        def flush_group() -> None:
            nonlocal first_group, current_source, current_refs
            if current_source is None:
                return
            if not first_group:
                handle.write(",\n")
            payload = {"source": current_source, "references": current_refs}
            handle.write("    ")
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            first_group = False

        for source, target in rows:
            if current_source is None:
                current_source = source
                current_refs = [target]
            elif source == current_source:
                current_refs.append(target)
            else:
                flush_group()
                current_source = source
                current_refs = [target]
        flush_group()
        handle.write("\n  ]\n}\n")


def export_classified_jsonl(conn: sqlite3.Connection, dataset: str, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        rows = conn.execute(
            """
            SELECT citation, origin_mask, family, subfamily, pattern, segments_json
            FROM citations
            WHERE dataset = ?
            ORDER BY citation
            """,
            (dataset,),
        )
        for citation, origin_mask, family, subfamily, pattern, segments_json in rows:
            payload = {
                "citation": citation,
                "origin_mask": origin_mask,
                "family": family,
                "subfamily": subfamily,
                "pattern": pattern,
                "segments": json.loads(segments_json),
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def top_pattern_examples(conn: sqlite3.Connection, dataset: str, limit: int = 8) -> dict:
    output: dict[str, list[str]] = defaultdict(list)
    rows = conn.execute(
        """
        SELECT pattern, citation
        FROM citations
        WHERE dataset = ?
        ORDER BY pattern, citation
        """,
        (dataset,),
    )
    for pattern, citation in rows:
        bucket = output[pattern]
        if len(bucket) < limit:
            bucket.append(citation)
    return dict(output)


def export_patterns(conn: sqlite3.Connection, summaries: dict, output_dir: Path) -> None:
    patterns_payload = {
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "pattern_descriptions": PATTERN_DESCRIPTIONS,
        "dataset_summaries": summaries,
        "examples": {
            dataset: top_pattern_examples(conn, dataset)
            for dataset in summaries
        },
    }
    (output_dir / "citation_patterns.json").write_text(
        json.dumps(patterns_payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "# Swiss citation pattern analysis",
        "",
        "## Counts",
        "",
        "| Dataset | Rows | Unique source citations | Unique text references | All unique citations | Unique links |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, summary in summaries.items():
        counts = get_dataset_counts(conn, dataset)
        lines.append(
            f"| {dataset} | {summary['rows']:,} | {counts['source_unique_count']:,} | "
            f"{counts['reference_unique_count']:,} | {counts['all_unique_count']:,} | "
            f"{counts['edge_count']:,} |"
        )
    lines.extend(["", "## Naming families", ""])
    for pattern, description in PATTERN_DESCRIPTIONS.items():
        lines.append(f"### {pattern}")
        lines.append("")
        lines.append(description["regex_family"])
        lines.append("")
        lines.append("Segments: " + ", ".join(description["segments"]))
        lines.append("")
        lines.append("Examples: " + "; ".join(description["examples"]))
        lines.append("")

    lines.extend(
        [
            "## Classification notes",
            "",
            "- Law citations are split into article/section marker, numeric unit, optional suffix or range, paragraph, letter/number/sentence subdivisions, and law code.",
            "- Law codes are subtyped morphologically as systematic collection numbers, uppercase abbreviations, mixed-case abbreviations, numbered abbreviations, compound abbreviations, or slash/cantonal abbreviations.",
            "- BGE court citations are split into reporter, volume, roman division, page, and optional pinpoint unit (`E.` consideration or `S.` page).",
            "- Federal tribunal docket citations are split into chamber, legal area code, separator style, serial number, year, optional decision date, and optional consideration.",
            "- References without an explicit law code in `laws_de.csv` are resolved to the source row law code. In court text, unresolved article references are kept without a law code rather than guessed.",
        ]
    )
    (output_dir / "citation_pattern_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_insights"))
    parser.add_argument("--db-name", default="citation_graph_extracted.sqlite")
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
        help="Datasets to process.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = args.output_dir / args.db_name
    conn = open_db(db_path)
    summaries: dict[str, dict] = {}
    for dataset in args.datasets:
        csv_path = args.data_dir / DATASETS[dataset]
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        print(f"Processing {dataset} from {csv_path}", flush=True)
        summary = process_dataset(conn, dataset, csv_path, batch_size=args.batch_size)
        summaries[dataset] = summary
        export_unique_json(conn, dataset, args.output_dir / f"{dataset}_citations.json", summary)
        export_links_json(conn, dataset, args.output_dir / f"{dataset}_links.json", summary)
        export_classified_jsonl(conn, dataset, args.output_dir / f"{dataset}_classified_citations.jsonl")
        print(
            f"Finished {dataset}: rows={summary['rows']:,}, "
            f"all_unique={get_dataset_counts(conn, dataset)['all_unique_count']:,}, "
            f"edges={get_dataset_counts(conn, dataset)['edge_count']:,}",
            flush=True,
        )

    export_patterns(conn, summaries, args.output_dir)
    conn.close()
    print(f"Wrote outputs to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
