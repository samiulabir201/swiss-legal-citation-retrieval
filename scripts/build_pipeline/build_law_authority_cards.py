#!/usr/bin/env python
"""Build deterministic law authority cards for laws_de.csv.

The output is the law-side analogue of the court v4/v5 authority-card assets:
formal citation structure, title metadata, citation graph anchors, static
semantic fallback signals, retrieval-safe views, and explicit LLM-needed flags.

This script is query-independent and does not read train/val/test labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_court_authority_cards import LEGAL_CONCEPTS, LAW_CODE_AREAS  # noqa: E402
from extract_citation_graph import classify_source_citation, law_code_subfamily, squash_ws  # noqa: E402


DATA_DIR = ROOT / "data"
INSIGHTS_DIR = ROOT / "data_insights"
ART_DIR = ROOT / "artifacts"

LAW_CSV = DATA_DIR / "laws_de.csv"
CLASSIFIED_JSONL = INSIGHTS_DIR / "laws_de_classified_citations.jsonl"
LINKS_JSON = INSIGHTS_DIR / "laws_de_links.json"
DEFAULT_OUTPUT = ART_DIR / "law_authority_cards_v1_static.jsonl"
DEFAULT_SUMMARY = ART_DIR / "law_authority_cards_v1_static.summary.json"


DATE_MONTHS_DE = {
    "januar": "01",
    "februar": "02",
    "marz": "03",
    "maerz": "03",
    "april": "04",
    "mai": "05",
    "juni": "06",
    "juli": "07",
    "august": "08",
    "september": "09",
    "oktober": "10",
    "november": "11",
    "dezember": "12",
}

SR_SECTORS = {
    "1": ("state, people, authorities", "constitutional and administrative law"),
    "2": ("private law, civil justice, enforcement", "private law and civil procedure"),
    "3": ("criminal law and criminal justice", "criminal law"),
    "4": ("education, science, culture", "education and culture law"),
    "5": ("national defence", "military and defence law"),
    "6": ("finance", "tax, finance, and customs law"),
    "7": ("public works, energy, transport, communications", "infrastructure and transport law"),
    "8": ("health, work, social security", "health, labour, and social insurance law"),
    "9": ("economy, agriculture, trade", "economic and agricultural law"),
}

TITLE_DOMAIN_RULES = [
    ("public procurement", "public procurement law", ["offentliche beschaffung", "beschaffungswesen", "auftraggeberin", "vergabe"]),
    ("migration and residence", "migration law", ["auslander", "integration", "asyl", "personenverkehr", "aufenthalt", "einburgerung"]),
    ("criminal law", "criminal law", ["strafgesetz", "militarstraf", "betäubungsmittel", "betaubungsmittel", "strafrecht"]),
    ("criminal procedure", "criminal procedure", ["strafprozess", "untersuchungshaft", "sicherheitshaft", "rechtshilfe in strafsachen"]),
    ("civil law", "civil law", ["zivilgesetzbuch", "zivilrecht", "erwachsenenschutz", "kindesschutz"]),
    ("obligations and contract", "private law and obligations", ["obligationenrecht", "vertrag", "haftpflicht", "kauf", "miete"]),
    ("civil procedure", "civil procedure", ["zivilprozess", "prozessordnung", "schiedsgericht"]),
    ("debt enforcement and bankruptcy", "debt enforcement", ["schuldbetreibung", "konkurs", "schkg", "pfandung"]),
    ("tax and customs", "tax law", ["steuer", "mehrwertsteuer", "zoll", "abgaben", "direkte bundessteuer"]),
    ("social insurance", "social insurance", ["sozialversicherung", "invalidenversicherung", "alters- und hinterlassenenversicherung", "arbeitslosenversicherung", "unfallversicherung", "berufliche vorsorge", "krankenversicherung"]),
    ("labour and employment", "employment law", ["arbeitsgesetz", "arbeitnehmer", "arbeitgeber", "personal", "arbeitszeit", "lohn"]),
    ("health and medicines", "health law", ["heilmittel", "krankenversicherung", "lebensmittel", "epidemien", "gesundheit", "arzneimittel", "medizinprodukte"]),
    ("animal health and welfare", "animal welfare and agriculture law", ["tierseuchen", "tierschutz", "veterinar", "veterinär"]),
    ("environmental protection", "environmental law", ["umweltschutz", "gewasserschutz", "gewässerschutz", "natur- und heimatschutz", "wald", "larm", "lärm", "klima"]),
    ("spatial planning and building", "spatial planning law", ["raumplanung", "bau", "zonenplan", "nutzungsplan"]),
    ("energy", "energy law", ["energie", "strom", "kernenergie", "elektrische leitungen"]),
    ("transport and traffic", "transport and traffic law", ["strassenverkehr", "straßenverkehr", "eisenbahn", "luftfahrt", "verkehr", "fahrzeug", "schifffahrt"]),
    ("telecommunications and media", "communications law", ["fernmelde", "radio", "fernsehen", "telekommunikation", "post"]),
    ("financial market", "financial market law", ["finanzmarkt", "finma", "bank", "kollektive kapitalanlagen", "geldwascherei", "versicherungsunternehmen"]),
    ("intellectual property", "intellectual property law", ["patent", "urheberrecht", "marken", "design", "geistiges eigentum"]),
    ("competition and antitrust", "competition law", ["kartell", "wettbewerb", "unlauterer wettbewerb"]),
    ("data protection", "data protection law", ["datenschutz", "personendaten", "informationssystem"]),
    ("education and vocational training", "education law", ["berufsbildung", "eidgenossisches fahigkeitszeugnis", "eidgenössisches fahigkeitszeugnis", "hochschule", "bildung", "efz", "eba"]),
    ("constitutional and political rights", "constitutional and public law", ["verfassung", "politische rechte", "volksabstimmung", "parlament", "bundesversammlung"]),
    ("public administration", "administrative law", ["verwaltungsverfahren", "verwaltung", "organisation", "zustandigkeit", "zuständigkeit"]),
    ("police and public security", "public security law", ["polizei", "sicherheit", "waffen", "sprengstoff", "uberwachung", "überwachung"]),
    ("international sanctions and foreign affairs", "international public law", ["massnahmen gegenuber", "massnahmen gegenüber", "sanktion", "embargo", "volkerrecht", "völkerrecht"]),
    ("agriculture and food economy", "agricultural law", ["landwirtschaft", "lebensmittel", "futtermittel", "wein", "tier", "pflanzen"]),
]

PROVISION_ROLE_RULES = [
    ("definition", ["begriffe", "bedeutet", "bedeuten", "definition"]),
    ("scope", ["geltungsbereich", "findet anwendung", "gilt fur", "gilt für", "unterstehen"]),
    ("purpose", ["zweck", "bezweckt"]),
    ("principle", ["grundsatz", "grundsatze", "grundsätze", "grundlagen"]),
    ("competence", ["zustandigkeit", "zuständigkeit", "aufgaben", "befugnisse"]),
    ("procedure", ["verfahren", "gesuch", "antrag", "beschwerde", "entscheid"]),
    ("duty", ["pflicht", "verpflichtet", "muss", "haben zu"]),
    ("right_or_entitlement", ["anspruch", "recht auf", "berechtigt"]),
    ("prohibition", ["verboten", "darf nicht", "untersagt"]),
    ("sanction_or_penalty", ["strafe", "busse", "buße", "bestraft", "sanktion"]),
    ("fees_or_costs", ["gebuhr", "gebühr", "kosten", "abgabe"]),
    ("data_reporting", ["meldet", "meldung", "register", "daten", "information"]),
    ("transitional_or_commencement", ["inkrafttreten", "ubergangsbestimmung", "übergangsbestimmung", "aufhebung", "schlussbestimmung"]),
]

SOURCE_TYPE_RULES = [
    ("federal_act", ["bundesgesetz"]),
    ("ordinance", ["verordnung"]),
    ("constitution", ["verfassung"]),
    ("regulation", ["reglement"]),
    ("federal_decree", ["bundesbeschluss", "bundesratsbeschluss"]),
    ("agreement", ["ubereinkunft", "übereinkunft", "vereinbarung", "abkommen", "vertrag"]),
    ("statutes", ["statuten"]),
]

ARTICLE_RE = re.compile(r"^Art\.\s+(?P<article>\d+[A-Za-z]*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?)(?P<rest>.*)$", re.I)
DATE_RE = re.compile(r"\b(?:vom\s+)?(?P<day>\d{1,2})\.\s+(?P<month>[A-Za-zÄÖÜäöü]+)\s+(?P<year>\d{4})\b")
PAREN_RE = re.compile(r"\(([^()]{1,100})\)")
ENUM_MARKER_RE = re.compile(r"(?:(?<=:)|(?<=;)|(?<=\n))\s*[a-z]\.\s+")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return re.sub(r"\s+", " ", value.replace("\x00", " ")).strip()


def fold(value: Any) -> str:
    text = clean_text(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.casefold()


def unique_keep_order(values: Iterable[Any], *, max_items: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_text(value)
        if not text:
            continue
        key = fold(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if max_items is not None and len(out) >= max_items:
            break
    return out


def field_join(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if isinstance(value, dict):
            for nested in value.values():
                if isinstance(nested, (list, tuple, set)):
                    parts.extend(clean_text(v) for v in nested)
                else:
                    parts.append(clean_text(nested))
        elif isinstance(value, (list, tuple, set)):
            parts.extend(clean_text(v) for v in value)
        else:
            parts.append(clean_text(value))
    return " | ".join(unique_keep_order(part for part in parts if part))


def clipped(value: Any, limit: int) -> str:
    text = clean_text(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def load_source_metadata(path: Path) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("origin_mask", 0) & 1:
                metadata[rec["citation"]] = rec
    return metadata


def load_links(path: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    outgoing: dict[str, list[str]] = {}
    incoming: dict[str, list[str]] = defaultdict(list)
    for row in payload.get("source_to_references", []):
        source = clean_text(row.get("source"))
        refs = unique_keep_order(row.get("references") or [])
        outgoing[source] = refs
        for ref in refs:
            incoming[ref].append(source)
    return outgoing, {key: unique_keep_order(value) for key, value in incoming.items()}


def fallback_classify(citation: str) -> dict:
    parsed = classify_source_citation(squash_ws(citation))
    return {
        "citation": citation,
        "family": parsed.family,
        "subfamily": parsed.subfamily,
        "pattern": parsed.pattern,
        "origin_mask": 1,
        "segments": parsed.segments,
    }


def derive_law_code_from_exact_citation(citation: str, segments: dict) -> str:
    existing = clean_text(segments.get("law_code"))
    normalized = squash_ws(citation)
    match = ARTICLE_RE.match(normalized)
    if not match:
        return existing
    rest = clean_text(match.group("rest"))
    for unit in segments.get("units") or []:
        marker = re.escape(clean_text(unit.get("marker")).rstrip("."))
        value = re.escape(clean_text(unit.get("value")))
        rest = re.sub(rf"^\s*{marker}\.?\s+{value}\b", "", rest, flags=re.I).strip()
    if not existing:
        return rest
    if rest and fold(rest) != fold(existing) and fold(rest).startswith(fold(existing)):
        return rest
    return existing


def classify_law_code(code: str) -> str:
    if not code:
        return "unresolved_law_code"
    return law_code_subfamily(code)


def title_parts(title: str) -> tuple[str, list[str]]:
    parts = [part.strip() for part in clean_text(title).split(" - ") if part.strip()]
    if not parts:
        return "", []
    return parts[0], parts[1:]


def parse_enactment_date(title: str) -> str:
    match = DATE_RE.search(title)
    if not match:
        return ""
    month_key = fold(match.group("month")).replace("ä", "a")
    month_key = month_key.replace("maerz", "marz")
    month = DATE_MONTHS_DE.get(month_key)
    if not month:
        return ""
    return f"{match.group('year')}-{month}-{int(match.group('day')):02d}"


def source_type_from_title(law_title: str) -> str:
    title_norm = fold(law_title[:180])
    for label, terms in SOURCE_TYPE_RULES:
        if any(fold(term) in title_norm for term in terms):
            return label
    return "other"


def aliases_from_title(law_title: str, law_code: str) -> list[str]:
    aliases = [law_code] if law_code else []
    for raw in PAREN_RE.findall(law_title):
        pieces = re.split(r"[,;/]", raw)
        for piece in pieces:
            piece = clean_text(piece)
            if not piece:
                continue
            tokens = piece.split()
            if len(piece) <= 30 and (
                re.search(r"[A-ZÄÖÜ]", piece)
                or re.fullmatch(r"[A-Za-zÄÖÜäöü0-9.-]+\s+\d+", piece)
            ):
                aliases.append(piece)
            if tokens:
                last = tokens[-1].strip()
                if re.search(r"[A-ZÄÖÜ]", last) and 2 <= len(last) <= 20:
                    aliases.append(last)
    return unique_keep_order(aliases, max_items=12)


def sr_sector(law_code: str) -> tuple[str, str]:
    if re.fullmatch(r"\d+(?:\.\d+)*", law_code or ""):
        return SR_SECTORS.get(law_code[0], ("", ""))
    return "", ""


def compile_concept_terms() -> list[tuple[str, re.Pattern[str] | None, str, str, str]]:
    entries: list[tuple[str, re.Pattern[str] | None, str, str, str]] = []
    for concept in LEGAL_CONCEPTS:
        label = concept["label"]
        area = concept["area"]
        for key in ("terms_de", "terms_fr", "terms_it", "terms_en"):
            for term in concept.get(key, []):
                folded = fold(term)
                if folded:
                    pattern = None
                    if re.fullmatch(r"[a-z0-9_.-]+", folded):
                        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(folded)}(?![a-z0-9])")
                    entries.append((folded, pattern, term, label, area))
    entries.sort(key=lambda row: len(row[0]), reverse=True)
    return entries


CONCEPT_TERM_ENTRIES = compile_concept_terms()


def match_static_concepts(title: str, text: str, law_code: str) -> tuple[list[str], dict[str, list[str]], list[str]]:
    haystack = f" {fold(title)} {fold(text[:5000])} "
    labels: list[str] = []
    areas: list[str] = []
    matched: dict[str, list[str]] = defaultdict(list)

    if law_code in LAW_CODE_AREAS:
        area, code_labels = LAW_CODE_AREAS[law_code]
        areas.append(area)
        labels.extend(code_labels)

    for folded, pattern, original, label, area in CONCEPT_TERM_ENTRIES:
        hit = bool(pattern.search(haystack)) if pattern is not None else bool(folded and folded in haystack)
        if hit:
            labels.append(label)
            areas.append(area)
            matched[label].append(original)

    return (
        unique_keep_order(labels, max_items=30),
        {key: unique_keep_order(values, max_items=12) for key, values in matched.items()},
        unique_keep_order(areas, max_items=20),
    )


def match_title_domains(title: str, text: str) -> tuple[list[str], list[str], dict[str, list[str]]]:
    del text
    haystack = f" {fold(title)} "
    labels: list[str] = []
    areas: list[str] = []
    matched: dict[str, list[str]] = defaultdict(list)
    for label, area, terms in TITLE_DOMAIN_RULES:
        hits = [term for term in terms if fold(term) in haystack]
        if hits:
            labels.append(label)
            areas.append(area)
            matched[label].extend(hits)
    return (
        unique_keep_order(labels, max_items=12),
        unique_keep_order(areas, max_items=12),
        {key: unique_keep_order(values, max_items=12) for key, values in matched.items()},
    )


def infer_provision_roles(title: str, text: str) -> list[str]:
    haystack = f" {fold(title)} {fold(text[:1600])} "
    roles = []
    for role, terms in PROVISION_ROLE_RULES:
        if any(fold(term) in haystack for term in terms):
            roles.append(role)
    return unique_keep_order(roles, max_items=8)


def sort_value(value: str) -> tuple[int, str]:
    match = re.match(r"(\d+)(.*)", clean_text(value))
    if not match:
        return (10**9, value)
    return (int(match.group(1)), match.group(2))


def row_sort_key(row: dict) -> tuple[Any, ...]:
    structural = row["structural"]
    units = structural.get("units") or []
    unit_parts = []
    for unit in units:
        unit_parts.append(unit.get("category", ""))
        unit_parts.extend(sort_value(part) for part in re.split(r"\s*(?:,|und|oder|/)\s*", unit.get("value", "")) if part)
    return (
        structural.get("law_code") or "",
        sort_value(structural.get("article") or ""),
        tuple(unit_parts),
        row["_source_row"],
    )


def reference_family(ref: str) -> str:
    if ref.startswith("Art.") or ref.startswith("§"):
        return "statute"
    if ref.startswith(("SR ", "AS ", "BBl ")):
        return "official_reference"
    if ref.startswith("BGE ") or re.search(r"\b\d{1,2}[A-Z]{1,4}[_\.]\d{1,6}/\d{2,4}\b", ref):
        return "court_case"
    return "other"


def split_references(refs: list[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for ref in refs:
        buckets[reference_family(ref)].append(ref)
    return {key: unique_keep_order(values, max_items=80) for key, values in buckets.items()}


def semantic_signal_level(issue_labels: list[str], title_domain_labels: list[str], roles: list[str], matched_terms: dict[str, list[str]]) -> str:
    if issue_labels and matched_terms:
        return "strong_dictionary_signal"
    if issue_labels or title_domain_labels or roles:
        return "partial_static_signal"
    return "formal_only"


def llm_priority(signal: str, roles: list[str], text_len: int) -> str:
    if "transitional_or_commencement" in roles and text_len < 120:
        return "low"
    if signal == "formal_only" or text_len > 600:
        return "high"
    if signal == "partial_static_signal":
        return "medium"
    return "medium"


def build_row_records(
    csv_path: Path,
    metadata_by_citation: dict[str, dict],
    outgoing_by_source: dict[str, list[str]],
    incoming_by_target: dict[str, list[str]],
    max_text_chars: int,
    limit: int | None,
) -> tuple[list[dict], Counter]:
    records: list[dict] = []
    stats = Counter()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            if limit is not None and row_idx >= limit:
                break
            citation = clean_text(row.get("citation"))
            text = row.get("text") or ""
            title = row.get("title") or ""
            metadata = metadata_by_citation.get(citation)
            if metadata is None:
                metadata = fallback_classify(citation)
                stats["fallback_parsed_missing_metadata"] += 1
            segments = dict(metadata.get("segments") or {})
            law_code = derive_law_code_from_exact_citation(citation, segments)
            if law_code and law_code != segments.get("law_code"):
                segments["law_code"] = law_code
                segments["law_code_family"] = classify_law_code(law_code)
                segments["law_code_resolution"] = segments.get("law_code_resolution") or "source_citation_remainder"
                stats["recovered_law_code_from_exact_citation"] += 1

            law_title, section_path = title_parts(title)
            source_type = source_type_from_title(law_title)
            aliases = aliases_from_title(law_title, law_code)
            sector_label, sector_area = sr_sector(law_code)
            issue_labels, matched_terms, concept_areas = match_static_concepts(title, text, law_code)
            title_domains, title_areas, title_matched = match_title_domains(title, text)
            provision_roles = infer_provision_roles(title, text)

            legal_area_candidates = unique_keep_order(
                [
                    *concept_areas,
                    *title_areas,
                    sector_area,
                ],
                max_items=12,
            )
            legal_area_static = legal_area_candidates[0] if legal_area_candidates else "unknown"

            static_domain_labels = unique_keep_order(
                [
                    source_type,
                    sector_label,
                    *title_domains,
                ],
                max_items=20,
            )
            all_issue_labels = unique_keep_order([*issue_labels, *title_domains], max_items=40)
            all_matched_terms: dict[str, list[str]] = dict(matched_terms)
            for key, values in title_matched.items():
                all_matched_terms.setdefault(key, [])
                all_matched_terms[key] = unique_keep_order([*all_matched_terms[key], *values], max_items=12)

            outgoing_refs = outgoing_by_source.get(citation, [])
            incoming_refs = incoming_by_target.get(citation, [])
            split_out = split_references(outgoing_refs)
            signal = semantic_signal_level(all_issue_labels, title_domains, provision_roles, all_matched_terms)
            priority = llm_priority(signal, provision_roles, len(text))

            structural = {
                "article_marker": segments.get("article_marker") or "Art.",
                "article": clean_text(segments.get("article")),
                "article_continuation": segments.get("article_continuation"),
                "article_series_kind": segments.get("article_series_kind"),
                "units": segments.get("units") or [],
                "law_code": law_code,
                "law_code_family": classify_law_code(law_code),
                "law_code_resolution": segments.get("law_code_resolution") or "source_column",
                "granularity": "paragraph" if segments.get("units") else "article",
            }

            record = {
                "_source_row": row_idx,
                "citation": citation,
                "source_family": "law",
                "family": "law",
                "pattern": metadata.get("pattern") or "statute_article",
                "subfamily": metadata.get("subfamily") or "statute_article",
                "language": "de",
                "title": title,
                "law_title": law_title,
                "title_section_path": section_path,
                "text_excerpt_original": clipped(text, max_text_chars),
                "structural": structural,
                "title_metadata": {
                    "source_type": source_type,
                    "enactment_date": parse_enactment_date(law_title),
                    "law_aliases": aliases,
                    "systematic_collection_sector": sector_label,
                },
                "static_semantics": {
                    "legal_area_static": legal_area_static,
                    "legal_area_candidates": legal_area_candidates,
                    "domain_labels_en": static_domain_labels,
                    "issue_labels_en": all_issue_labels,
                    "matched_terms_multilingual": all_matched_terms,
                    "provision_roles": provision_roles,
                    "static_semantic_signal": signal,
                },
                "normalized_anchors": {
                    "statute_anchors": split_out.get("statute", []),
                    "official_references": split_out.get("official_reference", []),
                    "court_case_anchors": split_out.get("court_case", []),
                    "other_reference_anchors": split_out.get("other", []),
                    "incoming_reference_count": len(incoming_refs),
                    "incoming_reference_examples": incoming_refs[:20],
                    "outgoing_reference_count": len(outgoing_refs),
                },
                "rag_enrichment": {
                    "legal_area": legal_area_static,
                    "primary_domain": static_domain_labels[0] if static_domain_labels else "",
                    "secondary_domain": static_domain_labels[1] if len(static_domain_labels) > 1 else "",
                    "legal_domain_path": unique_keep_order([legal_area_static, *static_domain_labels], max_items=8),
                    "topic": "",
                    "subtopic": "",
                    "micro_topic": "",
                    "concepts_en": all_issue_labels[:10],
                    "terms_original": unique_keep_order(
                        term
                        for terms in all_matched_terms.values()
                        for term in terms
                    )[:12],
                    "doctrinal_rule": "",
                    "legal_test": "",
                    "legal_question": "",
                    "legal_rule": "",
                    "english_summary": "",
                    "fact_pattern_tags": [],
                    "procedural_context": "",
                    "paragraph_role": "",
                    "authority_role": [],
                    "specificity_score": 0.0,
                },
                "retrieval_views": {},
                "enrichment_quality": {
                    "static_formal_complete": bool(structural["article"] and structural["law_code"] and title and text),
                    "static_semantic_signal": signal,
                    "needs_llm_for_complete_semantic_context": True,
                    "llm_priority": priority,
                    "has_outgoing_references": bool(outgoing_refs),
                    "has_incoming_references": bool(incoming_refs),
                    "text_char_count": len(text),
                    "title_char_count": len(title),
                },
                "provenance": {
                    "law_text": str(csv_path.relative_to(ROOT)),
                    "segments": str(CLASSIFIED_JSONL.relative_to(ROOT)),
                    "links": str(LINKS_JSON.relative_to(ROOT)),
                    "method": "deterministic_law_authority_card_v1_no_gold_no_query",
                },
            }
            records.append(record)
            stats["rows"] += 1
    return records, stats


def add_neighbors(records: list[dict]) -> None:
    by_law: dict[str, list[dict]] = defaultdict(list)
    by_article: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in records:
        structural = record["structural"]
        law_code = structural.get("law_code") or ""
        article = structural.get("article") or ""
        by_law[law_code].append(record)
        by_article[(law_code, article)].append(record["citation"])

    for law_code, law_records in by_law.items():
        law_records.sort(key=row_sort_key)
        for idx, record in enumerate(law_records):
            structural = record["structural"]
            siblings = by_article[(law_code, structural.get("article") or "")]
            adjacent = {
                "previous_in_law": law_records[idx - 1]["citation"] if idx > 0 else "",
                "next_in_law": law_records[idx + 1]["citation"] if idx + 1 < len(law_records) else "",
                "same_article_siblings": [c for c in siblings if c != record["citation"]][:30],
                "same_law_row_count": len(law_records),
                "same_article_row_count": len(siblings),
            }
            record["normalized_anchors"]["adjacent_citations"] = adjacent
            build_retrieval_views(record)


def build_retrieval_views(record: dict) -> None:
    structural = record["structural"]
    title_meta = record["title_metadata"]
    semantics = record["static_semantics"]
    anchors = record["normalized_anchors"]
    adjacent = anchors.get("adjacent_citations", {})
    record["retrieval_views"] = {
        "citation_view": field_join(record["citation"], structural.get("law_code"), title_meta.get("law_aliases")),
        "title_view": field_join(record.get("law_title"), record.get("title_section_path"), title_meta.get("source_type")),
        "structure_view": field_join(
            structural.get("article"),
            [f"{unit.get('marker')} {unit.get('value')}" for unit in structural.get("units", [])],
            structural.get("granularity"),
            structural.get("law_code_family"),
            title_meta.get("systematic_collection_sector"),
        ),
        "semantic_concepts_en": field_join(
            semantics.get("legal_area_static"),
            semantics.get("legal_area_candidates"),
            semantics.get("domain_labels_en"),
            semantics.get("issue_labels_en"),
            semantics.get("provision_roles"),
        ),
        "original_terms_view": field_join(semantics.get("matched_terms_multilingual")),
        "statute_anchor_view": field_join(anchors.get("statute_anchors"), anchors.get("official_references")),
        "law_context_view": field_join(
            adjacent.get("previous_in_law"),
            adjacent.get("next_in_law"),
            adjacent.get("same_article_siblings"),
            anchors.get("incoming_reference_examples"),
        ),
        "raw_context": record.get("text_excerpt_original", ""),
    }


def summarize(records: list[dict], base_stats: Counter, output_path: Path, limit: int | None) -> dict:
    stats = Counter(base_stats)
    law_code_family = Counter()
    legal_areas = Counter()
    source_types = Counter()
    semantic_signal = Counter()
    llm_priorities = Counter()
    provision_roles = Counter()
    issue_labels = Counter()
    domain_labels = Counter()
    outgoing_counts = Counter()
    incoming_rows = 0
    title_parts_dist = Counter()
    text_lengths: list[int] = []
    static_formal_complete = 0

    for record in records:
        structural = record["structural"]
        semantics = record["static_semantics"]
        quality = record["enrichment_quality"]
        title_meta = record["title_metadata"]
        anchors = record["normalized_anchors"]
        law_code_family[structural.get("law_code_family") or ""] += 1
        legal_areas[semantics.get("legal_area_static") or "unknown"] += 1
        source_types[title_meta.get("source_type") or ""] += 1
        semantic_signal[semantics.get("static_semantic_signal") or ""] += 1
        llm_priorities[quality.get("llm_priority") or ""] += 1
        provision_roles.update(semantics.get("provision_roles") or [])
        issue_labels.update(semantics.get("issue_labels_en") or [])
        domain_labels.update(semantics.get("domain_labels_en") or [])
        outgoing_counts[anchors.get("outgoing_reference_count") or 0] += 1
        if anchors.get("incoming_reference_count"):
            incoming_rows += 1
        title_parts_dist[1 + len(record.get("title_section_path") or [])] += 1
        text_lengths.append(quality.get("text_char_count") or 0)
        if quality.get("static_formal_complete"):
            static_formal_complete += 1

    def percentile(values: list[int], p: float) -> float:
        if not values:
            return 0.0
        values = sorted(values)
        k = (len(values) - 1) * p
        floor = math.floor(k)
        ceil = math.ceil(k)
        if floor == ceil:
            return float(values[int(k)])
        return values[floor] * (ceil - k) + values[ceil] * (k - floor)

    total = len(records)
    try:
        output_display = str(output_path.resolve().relative_to(ROOT))
    except ValueError:
        output_display = str(output_path)

    summary = {
        "builder": "deterministic_law_authority_card_v1_no_gold_no_query",
        "input": str(LAW_CSV.relative_to(ROOT)),
        "output": output_display,
        "limit": limit or 0,
        "cards_written": total,
        "static_formal_complete": static_formal_complete,
        "static_formal_complete_pct": round(static_formal_complete / max(total, 1) * 100, 2),
        "static_semantic_complete_for_english_rag": 0,
        "llm_needed_for_complete_semantic_context": total,
        "note_on_semantic_complete_count": (
            "Static code extracts formal structure and useful fallback labels, but it does not "
            "translate or legally interpret each provision's operative rule. Therefore complete "
            "English semantic context requires LLM enrichment for every row."
        ),
        "stats": dict(stats),
        "text_length": {
            "min": min(text_lengths) if text_lengths else 0,
            "median": percentile(text_lengths, 0.5),
            "p90": percentile(text_lengths, 0.9),
            "p95": percentile(text_lengths, 0.95),
            "p99": percentile(text_lengths, 0.99),
            "max": max(text_lengths) if text_lengths else 0,
        },
        "law_code_family": law_code_family.most_common(),
        "top_legal_areas_static": legal_areas.most_common(40),
        "top_source_types": source_types.most_common(30),
        "static_semantic_signal": semantic_signal.most_common(),
        "llm_priority": llm_priorities.most_common(),
        "top_provision_roles": provision_roles.most_common(40),
        "top_issue_labels": issue_labels.most_common(60),
        "top_domain_labels": domain_labels.most_common(40),
        "outgoing_reference_count_distribution": outgoing_counts.most_common(30),
        "rows_with_incoming_references": incoming_rows,
        "title_part_count_distribution": title_parts_dist.most_common(),
        "schema": {
            "identity": ["_source_row", "citation", "source_family", "language"],
            "formal_structure": ["structural.article", "structural.units", "structural.law_code", "structural.granularity"],
            "title_metadata": ["law_title", "title_section_path", "source_type", "enactment_date", "law_aliases"],
            "static_semantics": ["legal_area_static", "domain_labels_en", "issue_labels_en", "matched_terms_multilingual", "provision_roles"],
            "anchors": ["statute_anchors", "official_references", "incoming_reference_count", "adjacent_citations"],
            "retrieval_views": ["citation_view", "title_view", "structure_view", "semantic_concepts_en", "original_terms_view", "statute_anchor_view", "law_context_view", "raw_context"],
            "llm_contract": ["rag_enrichment.* empty/seeded fields", "needs_llm_for_complete_semantic_context", "llm_priority"],
        },
    }
    return summary


def write_cards(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=LAW_CSV)
    parser.add_argument("--classified", type=Path, default=CLASSIFIED_JSONL)
    parser.add_argument("--links", type=Path, default=LINKS_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--max-text-chars", type=int, default=2500)
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    metadata = load_source_metadata(args.classified)
    outgoing, incoming = load_links(args.links)
    records, stats = build_row_records(
        args.input,
        metadata,
        outgoing,
        incoming,
        args.max_text_chars,
        limit,
    )
    add_neighbors(records)
    write_cards(records, args.output)
    summary = summarize(records, stats, args.output, limit)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"cards_written": len(records), "output": str(args.output), "summary": str(args.summary)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
