#!/usr/bin/env python
"""Normalize court RAG enrichment with deterministic Swiss-law anchors.

The LLM is useful for query-neutral legal descriptors, but it is not the
source of truth for formal statute or case anchors. This module extracts those
anchors from the source paragraph and deterministic card metadata, then builds
retrieval-safe views from the cleaned fields only.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable


NORMALIZER_VERSION = "court_enrichment_normalizer_v1"


TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_./-]+")
ARTICLE_RE = re.compile(
    r"\b[Aa]rt\.\s*\d+[a-zA-Z]*"
    r"(?:\s*(?:Abs\.|al\.|para\.|par\.|lit\.|let\.|Bst\.|Ziff\.|ch\.|n\.|no\.|Nr\.)\s*[a-zA-Z0-9]+)*"
    r"(?:\s*(?:und|et|e|,|/)\s*(?:Abs\.|al\.|para\.|par\.|lit\.|let\.|Bst\.|Ziff\.|ch\.)?\s*[a-zA-Z0-9]+)*"
    r"(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü0-9./-]{1,24})?",
)
SECTION_RE = re.compile(
    r"§{1,2}\s*\d+[a-zA-Z]*"
    r"(?:\s*(?:Abs\.|al\.|lit\.|Ziff\.|Satz)\s*[a-zA-Z0-9]+)*"
    r"(?:\s*(?:ff\.|f\.))?"
    r"(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü0-9./-]{1,24})?",
)
SR_RE = re.compile(r"\b(?:SR|RS)\s*\d[\d.]{2,}\b")
BGE_RE = re.compile(r"\b(?:BGE|ATF|DTF)\s+\d{3}\s+[IVXLC]{1,5}\s+\d+[a-z]?(?:\s+E\.\s*[\w.]+)?")
BGE_BASE_RE = re.compile(r"\b(?:BGE|ATF|DTF)\s+\d{3}\s+[IVXLC]{1,5}\s+\d+[a-z]?\b")
DOCKET_RE = re.compile(r"\b\d{1,2}[A-Z]{1,4}[_\.]\d{1,6}/\d{4}\b")
URTEIL_DOCKET_RE = re.compile(
    r"\b(?:Urteil|Urteile|arr[eê]t|arr[eê]ts|sentenza|sentenze)\s+"
    r"(?:des\s+Bundesgerichts\s+|du\s+Tribunal\s+f[eé]d[eé]ral\s+|del\s+Tribunale\s+federale\s+)?"
    r"(?P<docket>\d{1,2}[A-Z]{1,4}[_\.]\d{1,6}/\d{4})",
    re.IGNORECASE,
)
CANTONAL_DECISION_RE = re.compile(
    r"\b(?:VGE|EGV|ACJC|ATA|TAF|BVGer|BVGE|TPF|SK|BB|RR|VG|KG|OGer|VB)\s*[A-Z0-9_.-]*/\d{2,4}\b"
)
PAGE_REF_RE = re.compile(r"\b(?:BGE|ATF|DTF)\s+\d{3}\s+[IVXLC]{1,5}\s+\d+[a-z]?\s+S\.\s*\d+\b")
DATE_RE = re.compile(
    r"\b\d{1,2}\.\s*(?:Januar|Februar|März|Maerz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember|"
    r"janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre|"
    r"gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}\b",
    re.IGNORECASE,
)


LEGAL_SOURCE_WORDS = {
    "de": [
        "Bundesgesetz",
        "Gesetz",
        "Verordnung",
        "Reglement",
        "Richtlinie",
        "Kreisschreiben",
        "Weisung",
        "Botschaft",
        "Erläuterungen",
    ],
    "fr": [
        "loi",
        "loi fédérale",
        "ordonnance",
        "règlement",
        "directive",
        "circulaire",
        "message",
        "instructions",
    ],
    "it": [
        "legge",
        "legge federale",
        "ordinanza",
        "regolamento",
        "direttiva",
        "circolare",
        "messaggio",
        "istruzioni",
    ],
}
DOCUMENT_WORDS = [
    "Zonenplan",
    "Nutzungsplan",
    "Richtplan",
    "Gestaltungsplan",
    "Baubewilligung",
    "Baugesuch",
    "Vertrag",
    "Gesamtarbeitsvertrag",
    "Police",
    "Gutachten",
    "Bericht",
    "Protokoll",
    "Abstimmungsunterlagen",
    "Zahlungsbefehl",
    "plan de zones",
    "plan d'affectation",
    "permis de construire",
    "contrat",
    "rapport",
    "procès-verbal",
    "proces-verbal",
    "expertise",
    "piano regolatore",
    "licenza edilizia",
    "contratto",
    "rapporto",
    "perizia",
]
EVENT_WORDS = [
    "Volksabstimmung",
    "Abstimmung",
    "Referendum",
    "Initiative",
    "Wahl",
    "votation",
    "référendum",
    "referendum",
    "initiative",
    "élection",
    "votazione",
    "iniziativa",
    "elezione",
]
SECONDARY_SOURCE_WORDS = [
    "Basler Kommentar",
    "Zürcher Kommentar",
    "Berner Kommentar",
    "Commentaire romand",
    "Kommentar zum",
    "Kommentar",
    "Commentaire",
    "Handbuch",
    "Traité",
    "Lehrbuch",
    "RDAF",
    "SJ",
    "JdT",
    "AJP",
    "ZBl",
    "Pra",
    "ASA",
    "StE",
    "SZS",
    "ZBJV",
    "ZSR",
    "BJM",
    "sic!",
    "EuGRZ",
    "SVR",
    "BlSchK",
    "PJA",
    "RtiD",
    "DTA",
    "in:",
]
EXACT_SIGNAL_TERMS = {
    "loi",
    "RDAF",
    "SJ",
    "JdT",
    "AJP",
    "ZBl",
    "Pra",
    "ASA",
    "StE",
    "SZS",
    "ZBJV",
    "ZSR",
    "BJM",
    "sic!",
    "SVR",
    "PJA",
    "RtiD",
    "DTA",
}


ROLE_ENUM = {
    "holding",
    "reasoning",
    "facts",
    "procedural_history",
    "legal_standard",
    "application",
    "citation",
    "costs",
    "notification",
    "disposition",
    "neutral",
}
ROLE_ALIASES = {
    "background": "facts",
    "fact": "facts",
    "factual": "facts",
    "procedure": "procedural_history",
    "procedural": "procedural_history",
    "standard_of_review": "legal_standard",
    "legal rule": "legal_standard",
    "cost": "costs",
    "fees": "costs",
    "notice": "notification",
    "notify": "notification",
    "obiter": "reasoning",
}
OUTCOME_ENUM = {"neutral", "dismissed", "granted", "partial", "inadmissible", "remitted", "none"}

ROLE_CUES = {
    "notification": [
        "dieses urteil wird",
        "le présent arrêt est communiqué",
        "la presente sentenza",
        "schriftlich mitgeteilt",
        "greffier",
        "gerichtsschreiber",
    ],
    "costs": [
        "gerichtskosten",
        "prozesskosten",
        "verfahrenskosten",
        "parteientschädigung",
        "frais judiciaires",
        "frais de la cause",
        "dépens",
        "spese giudiziarie",
        "ripetibili",
        "unentgeltliche rechtspflege",
        "assistance judiciaire",
    ],
    "disposition": [
        "demnach erkennt",
        "erkennt das bundesgericht",
        "par ces motifs",
        "per questi motivi",
        "die beschwerde wird",
        "le recours est",
        "il ricorso è",
    ],
    "facts": ["sachverhalt", "faits", "fatti"],
    "procedural_history": [
        "vorinstanz",
        "verwaltungsgericht",
        "kantonsgericht",
        "tribunal cantonal",
        "cour de justice",
        "replik",
        "duplik",
        "vernehmlassung",
    ],
    "legal_standard": [
        "nach art.",
        "gemäss art.",
        "selon l'art.",
        "aux termes de l'art.",
        "giusta l'art.",
        "ai sensi dell'art.",
        "rechtsprechung",
        "jurisprudence",
    ],
    "application": ["im vorliegenden fall", "en l'espèce", "nel caso concreto", "vorliegend"],
}

DISMISSED_RE = re.compile(
    r"(?:\bbeschwerde\s+wird\s+abgewiesen\b|\bbeschwerde\s+abzuweisen\b|"
    r"\ble\s+recours\s+est\s+rejet[ée]\b|\brecours\s+doit\s+[êe]tre\s+rejet[ée]\b|"
    r"\bil\s+ricorso\s+[èe]\s+respinto\b|\bricorso\s+dev'essere\s+respinto\b)",
    re.IGNORECASE,
)
GRANTED_RE = re.compile(
    r"(?:\bbeschwerde\s+wird\s+gutgeheissen\b|\bbeschwerde\s+gutzuheissen\b|"
    r"\ble\s+recours\s+est\s+admis\b|\brecours\s+doit\s+[êe]tre\s+admis\b|"
    r"\bil\s+ricorso\s+[èe]\s+accolto\b|\bricorso\s+dev'essere\s+accolto\b)",
    re.IGNORECASE,
)
PARTIAL_RE = re.compile(r"(?:\bteilweise\b|\bpartiellement\b|\bparzialmente\b)", re.IGNORECASE)
INADMISSIBLE_RE = re.compile(
    r"(?:\bauf\s+die\s+beschwerde\s+wird\s+nicht\s+eingetreten\b|\bnicht\s+einzutreten\b|"
    r"\bnichteintreten\b|\birrecevable\b|\binammissibile\b)",
    re.IGNORECASE,
)
REMITTAL_RE = re.compile(
    r"(?:\b(?:die\s+)?sache\s+wird.{0,140}\bzur(?:ue|ü|u)ckgewiesen\b|"
    r"\bzur\s+(?:neuen|erneuten)\s+(?:entscheidung|beurteilung|neubeurteilung).{0,120}\bzur(?:ue|ü|u)ckgewiesen\b|"
    r"\ban\s+die\s+(?:vorinstanz|beschwerdegegnerin|verwaltung|beh[oö]rde).{0,140}\bzur(?:ue|ü|u)ckgewiesen\b|"
    r"\brenvoie\s+la\s+cause\b|\bla\s+cause\s+est\s+renvoy[ée]e\b|"
    r"\brenvoy[ée]\s+.{0,80}\b(?:l'autorit[ée]|tribunal|instance)\b|"
    r"\bla\s+causa\s+[èe]\s+rinviata\b|\brinvia\s+la\s+causa\b)",
    re.IGNORECASE,
)


SUPPRESS_RULE_ROLES = {"facts", "procedural_history", "costs", "notification"}
TEXT_FIELDS = {
    "legal_area": 140,
    "primary_domain": 120,
    "secondary_domain": 160,
    "topic": 160,
    "subtopic": 180,
    "micro_topic": 220,
    "doctrinal_rule": 420,
    "legal_test": 320,
    "procedural_context": 240,
    "legal_topic": 160,
    "legal_rule": 420,
    "court_holding": 320,
    "factual_context": 320,
}
ARRAY_FIELDS = {
    "legal_domain_path": 8,
    "concepts_en": 10,
    "terms_original": 12,
    "fact_pattern_tags": 12,
    "authority_role": 12,
    "english_legal_concepts": 10,
    "search_keywords": 12,
}
FORBIDDEN_LLM_FIELDS = {
    "english_summary",
    "summary_en",
    "legal_question",
    "natural_language_queries",
    "query_phrases_en",
}


def normalize_enriched_court_row(
    citation: str,
    text: str,
    llm_enrichment: dict,
    deterministic_metadata: dict | None = None,
) -> dict:
    """Return production-safe enrichment for one court paragraph."""
    metadata = deterministic_metadata or {}
    source_text = clean_text(text or metadata.get("text_excerpt_original") or "")
    source_citation = clean_text(citation or metadata.get("citation") or "")
    court_base = clean_text(metadata.get("court_base")) or court_base_from_citation(source_citation)
    llm = llm_enrichment if isinstance(llm_enrichment, dict) else {}

    flags: dict[str, list[dict[str, str]] | list[str]] = {
        "moved_from_statute_anchors": [],
        "moved_from_case_anchors": [],
        "dropped_from_statute_anchors": [],
        "dropped_from_case_anchors": [],
        "self_references_removed": [],
        "page_references_removed": [],
        "role_corrections": [],
        "outcome_corrections": [],
        "suppressed_fields": [],
    }

    text_statutes = extract_statute_anchors(source_text)
    metadata_statutes = [
        item
        for item in as_list(metadata.get("statutes_cited") or metadata.get("statute_anchors"), max_items=80)
        if is_statute_anchor(item)
    ]
    statute_anchors = unique_keep_order([*text_statutes, *metadata_statutes], max_items=80)

    raw_cases = extract_case_anchor_candidates(source_text)
    raw_cases.extend(
        item
        for item in as_list(metadata.get("court_cases_cited") or metadata.get("case_anchors"), max_items=80)
        if is_case_anchor(item)
    )
    case_anchors, self_references, page_references, non_federal_decisions = clean_case_anchors(
        raw_cases,
        citation=source_citation,
        court_base=court_base,
    )
    flags["self_references_removed"].extend(self_references)
    flags["page_references_removed"].extend(page_references)

    legal_source_anchors = extract_signal_terms(source_text, all_legal_source_words())
    document_or_plan_anchors = extract_signal_terms(source_text, DOCUMENT_WORDS)
    event_anchors = unique_keep_order([*extract_signal_terms(source_text, EVENT_WORDS), *DATE_RE.findall(source_text)])
    secondary_sources = extract_signal_terms(source_text, SECONDARY_SOURCE_WORDS)

    move_llm_anchor_hints(
        llm,
        flags=flags,
        legal_source_anchors=legal_source_anchors,
        document_or_plan_anchors=document_or_plan_anchors,
        event_anchors=event_anchors,
        secondary_sources=secondary_sources,
    )

    paragraph_role = correct_paragraph_role(
        clean_text(llm.get("paragraph_role")),
        source_text,
        metadata=metadata,
        flags=flags,
    )
    outcome_signal = correct_outcome_signal(
        clean_text(llm.get("outcome_signal")),
        source_text,
        paragraph_role=paragraph_role,
        flags=flags,
    )

    rag_enrichment = clean_rag_enrichment(llm)
    seed_metadata_fields(rag_enrichment, metadata)
    drop_forbidden_llm_fields(rag_enrichment)
    rag_enrichment["paragraph_role"] = paragraph_role
    rag_enrichment["outcome_signal"] = outcome_signal

    if paragraph_role in SUPPRESS_RULE_ROLES:
        for key in ("doctrinal_rule", "legal_test", "legal_rule"):
            if clean_text(rag_enrichment.get(key)):
                flags["suppressed_fields"].append(key)
            rag_enrichment[key] = ""

    normalized_anchors = {
        "statute_anchors": statute_anchors,
        "case_anchors": case_anchors,
        "legal_source_anchors": unique_keep_order(legal_source_anchors, max_items=60),
        "secondary_sources": unique_keep_order(secondary_sources, max_items=60),
        "document_or_plan_anchors": unique_keep_order(document_or_plan_anchors, max_items=60),
        "event_anchors": unique_keep_order(event_anchors, max_items=60),
        "self_references": unique_keep_order([source_citation, court_base, *self_references], max_items=40),
        "page_references": unique_keep_order(page_references, max_items=40),
        "non_federal_decision_anchors": unique_keep_order(non_federal_decisions, max_items=40),
    }

    # Keep cleaned anchors on rag_enrichment for compatibility with older builders.
    rag_enrichment.update(
        {
            "statute_anchors": normalized_anchors["statute_anchors"],
            "case_anchors": normalized_anchors["case_anchors"],
            "legal_source_anchors": normalized_anchors["legal_source_anchors"],
            "secondary_sources": normalized_anchors["secondary_sources"],
            "document_or_plan_anchors": normalized_anchors["document_or_plan_anchors"],
            "event_anchors": normalized_anchors["event_anchors"],
        }
    )

    retrieval_views = build_retrieval_views(
        citation=source_citation,
        court_base=court_base,
        text=source_text,
        rag_enrichment=rag_enrichment,
        normalized_anchors=normalized_anchors,
        metadata=metadata,
    )
    enrichment_quality = build_enrichment_quality(
        rag_enrichment=rag_enrichment,
        normalized_anchors=normalized_anchors,
        metadata=metadata,
        flags=flags,
    )

    return {
        "rag_enrichment": rag_enrichment,
        "normalized_anchors": normalized_anchors,
        "anchor_quality_flags": flags,
        "retrieval_views": retrieval_views,
        "enrichment_quality": enrichment_quality,
    }


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return re.sub(r"\s+", " ", value.replace("\x00", " ")).strip()


def norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean_text(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.casefold()


def as_list(value: Any, *, max_items: int = 40) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        raw: Iterable[Any] = value.values()
    elif isinstance(value, (list, tuple, set)):
        raw = value
    else:
        raw = [value]
    return unique_keep_order(raw, max_items=max_items)


def unique_keep_order(values: Iterable[Any], *, max_items: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_text(value)
        if not text:
            continue
        key = norm(text)
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
        if isinstance(value, (list, tuple, set)):
            parts.extend(clean_text(item) for item in value)
        else:
            parts.append(clean_text(value))
    return " | ".join(unique_keep_order(part for part in parts if part))


def clipped(value: Any, limit: int = 1200) -> str:
    text = clean_text(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def all_legal_source_words() -> list[str]:
    out: list[str] = []
    for words in LEGAL_SOURCE_WORDS.values():
        out.extend(words)
    return out


def extract_signal_terms(text: str, terms: Iterable[str]) -> list[str]:
    normalized_text = f" {norm(text)} "
    out: list[str] = []
    for term in terms:
        n_term = norm(term)
        if not n_term:
            continue
        if n_term == "in:":
            if re.search(r"\bin\s*:", text, flags=re.IGNORECASE):
                out.append(term)
            continue
        if term in EXACT_SIGNAL_TERMS:
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, flags=re.IGNORECASE):
                out.append(term)
            continue
        if n_term in normalized_text:
            out.append(term)
    return unique_keep_order(out)


def extract_statute_anchors(text: str) -> list[str]:
    anchors: list[str] = []
    anchors.extend(ARTICLE_RE.findall(text))
    anchors.extend(SECTION_RE.findall(text))
    anchors.extend(SR_RE.findall(text))
    return unique_keep_order(clean_anchor(anchor) for anchor in anchors if clean_anchor(anchor))


def extract_case_anchor_candidates(text: str) -> list[str]:
    anchors: list[str] = []
    anchors.extend(BGE_RE.findall(text))
    anchors.extend(DOCKET_RE.findall(text))
    anchors.extend(match.group("docket") for match in URTEIL_DOCKET_RE.finditer(text))
    anchors.extend(CANTONAL_DECISION_RE.findall(text))
    anchors.extend(PAGE_REF_RE.findall(text))
    return unique_keep_order(clean_anchor(anchor) for anchor in anchors if clean_anchor(anchor))


def clean_anchor(value: Any) -> str:
    text = clean_text(value).strip(" .,;:()[]")
    return text.replace(" _", "_").replace("_ ", "_")


def is_statute_anchor(value: Any) -> bool:
    text = clean_anchor(value)
    return bool(ARTICLE_RE.fullmatch(text) or SECTION_RE.fullmatch(text) or SR_RE.fullmatch(text))


def is_case_anchor(value: Any) -> bool:
    text = clean_anchor(value)
    return bool(
        BGE_RE.search(text)
        or DOCKET_RE.search(text)
        or CANTONAL_DECISION_RE.search(text)
        or PAGE_REF_RE.search(text)
    )


def court_base_from_citation(citation: str) -> str:
    text = clean_text(citation)
    bge = BGE_BASE_RE.search(text)
    if bge:
        return bge.group(0)
    docket = DOCKET_RE.search(text)
    if docket:
        return docket.group(0)
    return re.sub(r"\s+E\.\s*[\w.]+.*$", "", text).strip()


def comparable_case_base(value: str) -> str:
    text = clean_anchor(value)
    bge = BGE_BASE_RE.search(text)
    if bge:
        return norm(bge.group(0))
    docket = DOCKET_RE.search(text)
    if docket:
        return norm(docket.group(0).replace(".", "_"))
    return norm(re.sub(r"\s+E\.\s*[\w.]+.*$", "", text))


def is_page_reference(value: str) -> bool:
    text = clean_anchor(value)
    if PAGE_REF_RE.search(text):
        return True
    return bool(BGE_BASE_RE.search(text) and re.search(r"\bS\.\s*\d+\b", text))


def clean_case_anchors(
    raw_cases: Iterable[str],
    *,
    citation: str,
    court_base: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    current_bases = {comparable_case_base(citation), comparable_case_base(court_base)}
    cases: list[str] = []
    self_refs: list[str] = []
    page_refs: list[str] = []
    non_federal: list[str] = []
    for raw in unique_keep_order(raw_cases, max_items=120):
        anchor = clean_anchor(raw)
        if not anchor:
            continue
        if is_page_reference(anchor):
            page_refs.append(anchor)
            if comparable_case_base(anchor) in current_bases:
                self_refs.append(anchor)
            continue
        if comparable_case_base(anchor) in current_bases or norm(anchor) == norm(citation):
            self_refs.append(anchor)
            continue
        if CANTONAL_DECISION_RE.search(anchor):
            non_federal.append(anchor)
            continue
        cases.append(anchor)
    return (
        unique_keep_order(cases, max_items=80),
        unique_keep_order(self_refs, max_items=40),
        unique_keep_order(page_refs, max_items=40),
        unique_keep_order(non_federal, max_items=40),
    )


def move_llm_anchor_hints(
    llm: dict[str, Any],
    *,
    flags: dict[str, list[Any]],
    legal_source_anchors: list[str],
    document_or_plan_anchors: list[str],
    event_anchors: list[str],
    secondary_sources: list[str],
) -> None:
    for field_name, flag_name in [
        ("statute_anchors", "moved_from_statute_anchors"),
        ("case_anchors", "moved_from_case_anchors"),
    ]:
        for raw in as_list(llm.get(field_name), max_items=80):
            destination = classify_nonformal_anchor(raw)
            if destination == "legal_source_anchors":
                legal_source_anchors.append(raw)
            elif destination == "document_or_plan_anchors":
                document_or_plan_anchors.append(raw)
            elif destination == "event_anchors":
                event_anchors.append(raw)
            elif destination == "secondary_sources":
                secondary_sources.append(raw)
            elif field_name == "statute_anchors" and not is_statute_anchor(raw):
                flags["dropped_from_statute_anchors"].append({"value": raw, "reason": "not_a_deterministic_statute_anchor"})
                continue
            elif field_name == "case_anchors" and not is_case_anchor(raw):
                flags["dropped_from_case_anchors"].append({"value": raw, "reason": "not_a_deterministic_case_anchor"})
                continue
            else:
                continue
            flags[flag_name].append({"value": raw, "to": destination, "reason": "anchor_like_nonformal_reference"})


def classify_nonformal_anchor(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    if extract_signal_terms(text, SECONDARY_SOURCE_WORDS):
        return "secondary_sources"
    if extract_signal_terms(text, DOCUMENT_WORDS):
        return "document_or_plan_anchors"
    if extract_signal_terms(text, EVENT_WORDS) or DATE_RE.search(text):
        return "event_anchors"
    if extract_signal_terms(text, all_legal_source_words()):
        return "legal_source_anchors"
    return ""


def clean_rag_enrichment(llm: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, limit in TEXT_FIELDS.items():
        out[key] = clipped(llm.get(key), limit)
    for key, limit in ARRAY_FIELDS.items():
        out[key] = unique_keep_order(as_list(llm.get(key), max_items=limit), max_items=limit)

    # Preserve old notebook field names, but keep them cleaned.
    for key in ("legal_topic", "legal_rule", "court_holding", "factual_context"):
        if key in llm and key not in out:
            out[key] = clipped(llm.get(key), TEXT_FIELDS.get(key, 320))
    for key in ("english_legal_concepts", "search_keywords"):
        if key in llm:
            out[key] = unique_keep_order(as_list(llm.get(key), max_items=ARRAY_FIELDS.get(key, 8)))

    specificity = llm.get("specificity_score", 0)
    try:
        out["specificity_score"] = max(0.0, min(1.0, float(specificity)))
    except (TypeError, ValueError):
        out["specificity_score"] = 0.0

    method = clean_text(llm.get("method"))
    if method:
        out["method"] = method
    return out


def drop_forbidden_llm_fields(rag: dict[str, Any]) -> None:
    for key in FORBIDDEN_LLM_FIELDS:
        rag.pop(key, None)


def seed_metadata_fields(rag: dict[str, Any], metadata: dict[str, Any]) -> None:
    if not rag.get("legal_area"):
        rag["legal_area"] = clean_text(metadata.get("legal_area"))
    if not rag.get("authority_role"):
        rag["authority_role"] = as_list(metadata.get("authority_role"), max_items=12)
    if not rag.get("concepts_en"):
        rag["concepts_en"] = unique_keep_order(
            [
                *as_list(metadata.get("issue_labels_en"), max_items=12),
                *as_list(metadata.get("english_legal_concepts"), max_items=12),
            ],
            max_items=10,
        )
    if not rag.get("terms_original") and isinstance(metadata.get("matched_terms_multilingual"), dict):
        terms: list[str] = []
        for raw_terms in metadata["matched_terms_multilingual"].values():
            terms.extend(as_list(raw_terms, max_items=20))
        rag["terms_original"] = unique_keep_order(terms, max_items=12)


def normalize_role_value(value: str) -> str:
    key = norm(value).replace(" ", "_")
    key = ROLE_ALIASES.get(key, key)
    return key if key in ROLE_ENUM else "reasoning"


def cue_hit(text_norm: str, cue: str) -> bool:
    return norm(cue) in text_norm


def correct_paragraph_role(
    llm_role: str,
    text: str,
    *,
    metadata: dict[str, Any],
    flags: dict[str, list[Any]],
) -> str:
    role = normalize_role_value(llm_role)
    text_norm = norm(text)
    detected = ""
    if metadata.get("is_notification_paragraph"):
        detected = "notification"
    else:
        for candidate in ("notification", "costs", "disposition", "facts", "procedural_history", "legal_standard", "application"):
            if any(cue_hit(text_norm, cue) for cue in ROLE_CUES[candidate]):
                detected = candidate
                break

    corrected = detected or role
    if corrected != role:
        flags["role_corrections"].append({"from": role, "to": corrected, "reason": "deterministic_role_cue"})
    return corrected


def correct_outcome_signal(
    llm_outcome: str,
    text: str,
    *,
    paragraph_role: str,
    flags: dict[str, list[Any]],
) -> str:
    original = norm(llm_outcome).replace(" ", "_")
    original = original if original in OUTCOME_ENUM else "none"
    detected = deterministic_outcome(text)
    role_allows_outcome = paragraph_role in {"disposition", "holding", "application"}

    if detected and role_allows_outcome:
        corrected = detected
    elif detected and paragraph_role == "reasoning":
        corrected = detected
    else:
        corrected = "none" if original not in {"neutral", "none"} else original

    if corrected != original:
        flags["outcome_corrections"].append({"from": original, "to": corrected, "reason": "conservative_disposition_rule"})
    return corrected


def deterministic_outcome(text: str) -> str:
    head = text[:1400]
    if REMITTAL_RE.search(head):
        return "remitted"
    if INADMISSIBLE_RE.search(head):
        return "inadmissible"
    if GRANTED_RE.search(head):
        return "partial" if PARTIAL_RE.search(head) else "granted"
    if DISMISSED_RE.search(head):
        return "partial" if PARTIAL_RE.search(head) else "dismissed"
    return ""


def build_retrieval_views(
    *,
    citation: str,
    court_base: str,
    text: str,
    rag_enrichment: dict[str, Any],
    normalized_anchors: dict[str, list[str]],
    metadata: dict[str, Any],
) -> dict[str, str]:
    topic_path = field_join(
        rag_enrichment.get("legal_area"),
        rag_enrichment.get("primary_domain"),
        rag_enrichment.get("secondary_domain"),
        rag_enrichment.get("legal_domain_path"),
        rag_enrichment.get("topic") or rag_enrichment.get("legal_topic"),
        rag_enrichment.get("subtopic"),
        rag_enrichment.get("micro_topic"),
    )
    semantic = field_join(
        topic_path,
        rag_enrichment.get("concepts_en"),
        rag_enrichment.get("english_legal_concepts"),
        rag_enrichment.get("fact_pattern_tags"),
        rag_enrichment.get("search_keywords"),
        metadata.get("issue_labels_en"),
    )
    legal_rule_view = ""
    if rag_enrichment.get("paragraph_role") not in SUPPRESS_RULE_ROLES:
        legal_rule_view = field_join(
            rag_enrichment.get("doctrinal_rule"),
            rag_enrichment.get("legal_test"),
            rag_enrichment.get("legal_rule"),
            rag_enrichment.get("court_holding"),
        )
    return {
        "semantic_concepts_en": semantic,
        "topic_path": topic_path,
        "original_terms_view": field_join(rag_enrichment.get("terms_original")),
        "statute_anchor_view": field_join(
            normalized_anchors["statute_anchors"],
            normalized_anchors["legal_source_anchors"],
            metadata.get("law_codes"),
        ),
        "case_anchor_view": field_join(court_base, normalized_anchors["case_anchors"]),
        "legal_rule_view": legal_rule_view,
        "fact_pattern_view": field_join(rag_enrichment.get("fact_pattern_tags"), rag_enrichment.get("factual_context")),
        "procedural_view": field_join(
            rag_enrichment.get("procedural_context"),
            rag_enrichment.get("paragraph_role"),
            rag_enrichment.get("outcome_signal"),
        ),
        "authority_view": field_join(
            rag_enrichment.get("authority_role"),
            metadata.get("authority_role"),
            (metadata.get("structural") or {}).get("court_base_source_count")
            if isinstance(metadata.get("structural"), dict)
            else "",
            (metadata.get("structural") or {}).get("court_base_text_ref_count")
            if isinstance(metadata.get("structural"), dict)
            else "",
        ),
        "raw_context": clipped(text, 1200),
        "citation_view": field_join(citation, court_base),
    }


def build_enrichment_quality(
    *,
    rag_enrichment: dict[str, Any],
    normalized_anchors: dict[str, list[str]],
    metadata: dict[str, Any],
    flags: dict[str, list[Any]],
) -> dict[str, Any]:
    role = rag_enrichment.get("paragraph_role")
    low_value = role in {"notification", "costs"} or bool(metadata.get("is_notification_paragraph"))
    moved_count = sum(
        len(flags[key])
        for key in ("moved_from_statute_anchors", "moved_from_case_anchors", "dropped_from_statute_anchors", "dropped_from_case_anchors")
    )
    return {
        "method": rag_enrichment.get("method", ""),
        "normalizer": NORMALIZER_VERSION,
        "has_statute_anchor": bool(normalized_anchors["statute_anchors"]),
        "has_case_anchor": bool(normalized_anchors["case_anchors"]),
        "has_original_language_terms": bool(rag_enrichment.get("terms_original")),
        "has_specific_topic": bool(
            rag_enrichment.get("micro_topic")
            or rag_enrichment.get("subtopic")
            or rag_enrichment.get("topic")
            or rag_enrichment.get("legal_topic")
        ),
        "low_value_paragraph": low_value,
        "grounded_references_only": True,
        "anchor_cleanup_count": moved_count,
        "rule_fields_suppressed": bool(flags["suppressed_fields"]),
    }


__all__ = ["normalize_enriched_court_row"]
