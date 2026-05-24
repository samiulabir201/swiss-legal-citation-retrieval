#!/usr/bin/env python
"""Profile court_considerations.csv for court enrichment schema design.

The goal is not to enrich rows with an LLM. It is to inspect the full court
corpus and produce a field contract for the LLM notebook:

  - which anchors must be deterministic
  - which anchor-like strings must not pollute statute/case anchors
  - which paragraph roles and outcomes need conservative rules
  - which legal issue labels and multilingual terms appear in the corpus

Outputs:
  artifacts/court_enrichment_corpus_profile.json
  docs/court_enrichment_field_contract.md
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
COURT_CSV = ROOT / "data" / "court_considerations.csv"
SAMPLE_OUTPUT = Path(r"C:\Users\samiul\Downloads\enriched_court_citations_10 (1).jsonl")
PROFILE_JSON = ROOT / "artifacts" / "court_enrichment_corpus_profile.json"
PROFILE_PARTIAL_JSON = ROOT / "artifacts" / "court_enrichment_corpus_profile.partial.json"
PROFILE_MD = ROOT / "docs" / "court_enrichment_field_contract.md"

# Known from the current full-corpus artifact. This avoids a slow pre-count pass.
KNOWN_TOTAL_ROWS = 2_476_315
PARTIAL_WRITE_EVERY = 250_000
PROGRESS_REFRESH_EVERY = 10_000

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_./-]+")
ARTICLE_RE = re.compile(
    r"\b[Aa]rt\.\s*\d+[a-zA-Z]*"
    r"(?:\s*(?:Abs\.|al\.|para\.|par\.|lit\.|let\.|Ziff\.|ch\.|n\.|no\.|Nr\.)\s*[a-zA-Z0-9]+)*"
    r"(?:\s*(?:und|et|e|,)\s*(?:Abs\.|al\.|para\.|par\.|lit\.|let\.|Ziff\.|ch\.)?\s*[a-zA-Z0-9]+)*"
    r"(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü0-9./-]{1,24})?",
)
SECTION_RE = re.compile(
    r"§\s*\d+[a-zA-Z]*"
    r"(?:\s*(?:Abs\.|al\.|lit\.|Ziff\.|Satz)\s*[a-zA-Z0-9]+)*"
    r"(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü0-9./-]{1,24})?",
)
SR_RE = re.compile(r"\b(?:SR|RS)\s*\d[\d.]{2,}\b")
BGE_RE = re.compile(r"\b(?:BGE|ATF|DTF)\s+\d{3}\s+[IVXLC]{1,5}\s+\d+[a-z]?(?:\s+E\.\s*[\w.]+)?")
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
PAGE_REF_RE = re.compile(r"\b(?:BGE|ATF|DTF)\s+\d{3}\s+[IVXLC]{1,5}\s+\d+\s+S\.\s*\d+\b")
DATE_RE = re.compile(r"\b\d{1,2}\.\s*(?:Januar|Februar|März|Maerz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember|janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre|gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}\b", re.IGNORECASE)

LEGAL_SOURCE_WORDS = {
    "de": [
        "Bundesgesetz", "Gesetz", "Verordnung", "Reglement", "Richtlinie",
        "Kreisschreiben", "Weisung", "Botschaft", "Erläuterungen",
    ],
    "fr": [
        "loi", "loi fédérale", "ordonnance", "règlement", "directive",
        "circulaire", "message", "instructions",
    ],
    "it": [
        "legge", "legge federale", "ordinanza", "regolamento", "direttiva",
        "circolare", "messaggio", "istruzioni",
    ],
}
DOCUMENT_WORDS = [
    "Zonenplan", "Nutzungsplan", "Richtplan", "Gestaltungsplan", "Baubewilligung",
    "Baugesuch", "Vertrag", "Gesamtarbeitsvertrag", "Police", "Gutachten",
    "Bericht", "Protokoll", "Abstimmungsunterlagen", "Zahlungsbefehl",
    "plan de zones", "plan d'affectation", "permis de construire", "contrat",
    "rapport", "procès-verbal", "proces-verbal", "expertise",
    "piano regolatore", "licenza edilizia", "contratto", "rapporto", "perizia",
]
EVENT_WORDS = [
    "Volksabstimmung", "Abstimmung", "Referendum", "Initiative", "Wahl",
    "votation", "référendum", "referendum", "initiative", "élection",
    "votazione", "iniziativa", "elezione",
]
SECONDARY_SOURCE_WORDS = [
    "Kommentar", "Basler Kommentar", "Zürcher Kommentar", "Berner Kommentar",
    "Commentaire", "Commentaire romand", "Kommentar zum", "Handbuch",
    "Traité", "Lehrbuch", "in:", "RDAF", "SJ", "JdT", "AJP", "ZBl", "Pra",
    "ASA", "StE", "SZS", "ZBJV", "ZSR", "BJM", "sic!", "EuGRZ", "SVR",
    "BlSchK", "PJA", "RtiD", "DTA",
]

ROLE_CUES = {
    "notification": [
        "dieses urteil wird", "le présent arrêt est communiqué", "la presente sentenza",
        "schriftlich mitgeteilt", "greffier", "gerichtsschreiber",
    ],
    "costs": [
        "gerichtskosten", "parteientschädigung", "frais judiciaires", "dépens",
        "spese giudiziarie", "ripetibili", "unentgeltliche rechtspflege",
    ],
    "disposition": [
        "demnach erkennt", "erkennt das bundesgericht", "par ces motifs",
        "per questi motivi", "die beschwerde wird", "le recours est", "il ricorso è",
    ],
    "facts": [
        "sachverhalt", "faits", "fatti", "a.", "b.", "c.",
    ],
    "procedural_history": [
        "vorinstanz", "verwaltungsgericht", "kantonsgericht", "tribunal cantonal",
        "cour de justice", "la cour", "replik", "duplik", "vernehmlassung",
    ],
    "legal_standard": [
        "nach art.", "gemäss art.", "selon l'art.", "aux termes de l'art.",
        "giusta l'art.", "ai sensi dell'art.", "rechtsprechung", "jurisprudence",
    ],
    "application": [
        "im vorliegenden fall", "en l'espèce", "nel caso concreto", "vorliegend",
    ],
}

OUTCOME_CUES = {
    "dismissed": [
        "beschwerde wird abgewiesen", "beschwerde abzuweisen", "recours est rejeté",
        "recours doit être rejeté", "il ricorso è respinto", "ricorso dev'essere respinto",
    ],
    "granted": [
        "beschwerde wird gutgeheissen", "beschwerde gutzuheissen", "recours est admis",
        "recours doit être admis", "il ricorso è accolto", "ricorso dev'essere accolto",
    ],
    "partial": [
        "teilweise gutgeheissen", "partiellement admis", "parzialmente accolto",
    ],
    "inadmissible": [
        "nicht einzutreten", "nichteintreten", "irrecevable", "inammissibile",
    ],
    "remitted": [
        "zurückgewiesen", "rückweisung", "renvoyée", "renvoi", "rinviata", "rinvio",
    ],
}


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.casefold()


def clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def clean_example_text(text: Any) -> str:
    """Keep profile examples useful without copying raw section signs into JSON."""
    return clean(text).replace("§", "section ")


def add_examples(store: dict[str, list[dict[str, str]]], key: str, citation: str, text: str, max_items: int = 5) -> None:
    bucket = store.setdefault(key, [])
    if len(bucket) >= max_items:
        return
    bucket.append({"citation": citation, "text": clean_example_text(text)[:420]})


def top(counter: Counter, n: int = 40) -> list[list[Any]]:
    return [[k, v] for k, v in counter.most_common(n)]


class Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self._bar = None
        self._last = 0
        try:
            from tqdm.auto import tqdm

            self._bar = tqdm(
                total=total,
                unit="row",
                desc="court schema scan",
                dynamic_ncols=True,
                mininterval=1.0,
            )
        except Exception:
            self._bar = None
            print("[progress] tqdm not installed; falling back to periodic logs.")

    def update(self, current: int, *, force: bool = False) -> None:
        if self._bar is not None:
            delta = current - self._last
            if delta > 0:
                self._bar.update(delta)
                self._last = current
            return
        if force or current % PROGRESS_REFRESH_EVERY == 0:
            pct = 100.0 * current / max(self.total, 1)
            print(f"[progress] {current:,}/{self.total:,} rows ({pct:.2f}%)", flush=True)

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def detect_language(text: str) -> str:
    n = " " + norm(text[:2000]) + " "
    scores = {
        "de": sum(n.count(w) for w in [" der ", " die ", " das ", " und ", " nicht ", " beschwerde", " gericht"]),
        "fr": sum(n.count(w) for w in [" le ", " la ", " les ", " que ", " recours ", " droit ", " tribunal", " arret"]),
        "it": sum(n.count(w) for w in [" il ", " la ", " che ", " ricorso ", " diritto ", " della ", " tribunale", " sentenza"]),
    }
    lang, score = max(scores.items(), key=lambda item: item[1])
    return lang if score else "unknown"


def extract_code_from_anchor(anchor: str) -> str | None:
    tokens = TOKEN_RE.findall(anchor)
    for token in reversed(tokens):
        stripped = token.strip(".,;:()[]")
        if len(stripped) < 2:
            continue
        if stripped.lower() in {"art", "abs", "al", "lit", "let", "para", "par", "ziff", "ch", "und", "et"}:
            continue
        if any(ch.isupper() for ch in stripped) and any(ch.isalpha() for ch in stripped):
            return stripped
    return None


def load_known_concepts() -> list[dict[str, Any]]:
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from build_court_authority_cards import LEGAL_CONCEPTS

        return list(LEGAL_CONCEPTS)
    except Exception as exc:
        print(f"[warn] could not import LEGAL_CONCEPTS: {exc}")
        return []


def match_known_concepts(text: str, concepts: list[dict[str, Any]]) -> Iterable[tuple[str, str]]:
    n = norm(text)
    for concept in concepts:
        hits = []
        for key in ("terms_de", "terms_fr", "terms_it", "terms_en"):
            for term in concept.get(key, []):
                if norm(term) in n:
                    hits.append(term)
        if hits:
            yield concept["label"], concept.get("area", "unknown")


def scan_current_sample() -> dict[str, Any]:
    if not SAMPLE_OUTPUT.exists():
        return {"exists": False}
    rows = []
    with SAMPLE_OUTPUT.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    field_counts: dict[str, Counter] = {
        "rag_enrichment": Counter(),
        "retrieval_views": Counter(),
        "enrichment_quality": Counter(),
    }
    values: dict[str, Counter] = {
        "paragraph_role": Counter(),
        "outcome_signal": Counter(),
        "authority_role": Counter(),
        "legal_area": Counter(),
    }
    for row in rows:
        for section in field_counts:
            field_counts[section].update((row.get(section) or {}).keys())
        enr = row.get("rag_enrichment") or {}
        values["paragraph_role"][enr.get("paragraph_role")] += 1
        values["outcome_signal"][enr.get("outcome_signal")] += 1
        values["legal_area"][enr.get("legal_area")] += 1
        values["authority_role"].update(enr.get("authority_role") or [])
    return {
        "exists": True,
        "path": str(SAMPLE_OUTPUT),
        "row_count": len(rows),
        "fields": {section: sorted(counter) for section, counter in field_counts.items()},
        "values": {name: top(counter, 20) for name, counter in values.items()},
    }


def write_partial_profile(stats: dict[str, Any], sample: dict[str, Any]) -> None:
    stats["elapsed_seconds"] = round(time.time() - stats.get("_start_time", time.time()), 3)
    profile = serialize_profile(stats, sample)
    profile["partial"] = True
    PROFILE_PARTIAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PARTIAL_JSON.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")


def scan_corpus(sample: dict[str, Any]) -> dict[str, Any]:
    concepts = load_known_concepts()
    start = time.time()
    stats: dict[str, Any] = {
        "row_count": 0,
        "_start_time": start,
        "language_counts": Counter(),
        "citation_kind_counts": Counter(),
        "article_anchor_count": 0,
        "section_anchor_count": 0,
        "sr_rs_anchor_count": 0,
        "bge_atf_anchor_count": 0,
        "docket_anchor_count": 0,
        "cantonal_decision_anchor_count": 0,
        "page_ref_anchor_count": 0,
        "legal_source_word_counts": Counter(),
        "document_word_counts": Counter(),
        "event_word_counts": Counter(),
        "secondary_source_word_counts": Counter(),
        "law_code_counts": Counter(),
        "known_issue_label_counts": Counter(),
        "known_issue_area_counts": Counter(),
        "role_cue_counts": Counter(),
        "outcome_cue_counts": Counter(),
        "examples": {},
    }

    progress = Progress(KNOWN_TOTAL_ROWS)
    with COURT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            citation = clean(row.get("citation"))
            text = row.get("text") or ""
            row_no = stats["row_count"] + 1
            stats["row_count"] = row_no

            if citation.startswith("BGE "):
                stats["citation_kind_counts"]["bge_consideration"] += 1
            elif DOCKET_RE.search(citation):
                stats["citation_kind_counts"]["docket_consideration"] += 1
            else:
                stats["citation_kind_counts"]["other_court_citation"] += 1

            lang = detect_language(text)
            stats["language_counts"][lang] += 1
            text_norm = norm(text)

            article_matches = ARTICLE_RE.findall(text)
            if article_matches:
                stats["article_anchor_count"] += len(article_matches)
                add_examples(stats["examples"], "article_anchors", citation, text)
                for anchor in article_matches:
                    code = extract_code_from_anchor(anchor)
                    if code:
                        stats["law_code_counts"][code] += 1

            section_matches = SECTION_RE.findall(text)
            if section_matches:
                stats["section_anchor_count"] += len(section_matches)
                add_examples(stats["examples"], "cantonal_section_anchors", citation, text)
                for anchor in section_matches:
                    code = extract_code_from_anchor(anchor)
                    if code:
                        stats["law_code_counts"][code] += 1

            sr_matches = SR_RE.findall(text)
            if sr_matches:
                stats["sr_rs_anchor_count"] += len(sr_matches)
                add_examples(stats["examples"], "sr_rs_anchors", citation, text)

            bge_matches = BGE_RE.findall(text)
            if bge_matches:
                stats["bge_atf_anchor_count"] += len(bge_matches)
                add_examples(stats["examples"], "bge_atf_case_anchors", citation, text)

            page_refs = PAGE_REF_RE.findall(text)
            if page_refs:
                stats["page_ref_anchor_count"] += len(page_refs)
                add_examples(stats["examples"], "self_or_page_references", citation, text)

            docket_matches = DOCKET_RE.findall(text)
            docket_matches.extend(m.group("docket") for m in URTEIL_DOCKET_RE.finditer(text))
            if docket_matches:
                stats["docket_anchor_count"] += len(docket_matches)
                add_examples(stats["examples"], "federal_docket_case_anchors", citation, text)

            cantonal_matches = CANTONAL_DECISION_RE.findall(text)
            if cantonal_matches:
                stats["cantonal_decision_anchor_count"] += len(cantonal_matches)
                add_examples(stats["examples"], "cantonal_or_special_decision_anchors", citation, text)

            for group in LEGAL_SOURCE_WORDS.values():
                for word in group:
                    if norm(word) in text_norm:
                        stats["legal_source_word_counts"][word] += 1
                        add_examples(stats["examples"], "legal_source_anchors", citation, text)

            for word in DOCUMENT_WORDS:
                if norm(word) in text_norm:
                    stats["document_word_counts"][word] += 1
                    add_examples(stats["examples"], "document_or_plan_anchors", citation, text)

            for word in EVENT_WORDS:
                if norm(word) in text_norm:
                    stats["event_word_counts"][word] += 1
                    add_examples(stats["examples"], "event_anchors", citation, text)

            for word in SECONDARY_SOURCE_WORDS:
                if norm(word) in text_norm:
                    stats["secondary_source_word_counts"][word] += 1
                    add_examples(stats["examples"], "secondary_sources", citation, text)

            for role, cues in ROLE_CUES.items():
                if any(norm(cue) in text_norm for cue in cues):
                    stats["role_cue_counts"][role] += 1
                    add_examples(stats["examples"], "role_" + role, citation, text)

            for outcome, cues in OUTCOME_CUES.items():
                if any(norm(cue) in text_norm for cue in cues):
                    stats["outcome_cue_counts"][outcome] += 1
                    add_examples(stats["examples"], "outcome_" + outcome, citation, text)

            for label, area in match_known_concepts(text, concepts):
                stats["known_issue_label_counts"][label] += 1
                stats["known_issue_area_counts"][area] += 1

            if row_no % PROGRESS_REFRESH_EVERY == 0:
                progress.update(row_no)

            if row_no % PARTIAL_WRITE_EVERY == 0:
                elapsed = time.time() - start
                print(f"[scan] rows={row_no:,}; elapsed={elapsed:.1f}s; rate={row_no/max(elapsed,1):,.0f} rows/s", flush=True)
                write_partial_profile(stats, sample)
                print(f"[partial] wrote {PROFILE_PARTIAL_JSON}", flush=True)

    stats["elapsed_seconds"] = round(time.time() - start, 3)
    progress.update(stats["row_count"], force=True)
    progress.close()
    stats.pop("_start_time", None)
    return stats


def serialize_profile(stats: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": str(COURT_CSV.relative_to(ROOT)),
        "sample_output": sample,
        "row_count": stats["row_count"],
        "elapsed_seconds": stats["elapsed_seconds"],
        "language_counts": top(stats["language_counts"], 20),
        "citation_kind_counts": top(stats["citation_kind_counts"], 20),
        "anchor_counts": {
            "article_anchors": stats["article_anchor_count"],
            "cantonal_section_anchors": stats["section_anchor_count"],
            "sr_rs_anchors": stats["sr_rs_anchor_count"],
            "bge_atf_case_anchors": stats["bge_atf_anchor_count"],
            "federal_docket_case_anchors": stats["docket_anchor_count"],
            "cantonal_or_special_decision_anchors": stats["cantonal_decision_anchor_count"],
            "page_ref_anchors": stats["page_ref_anchor_count"],
        },
        "top_law_codes_from_text": top(stats["law_code_counts"], 120),
        "top_legal_source_words": top(stats["legal_source_word_counts"], 80),
        "top_document_words": top(stats["document_word_counts"], 80),
        "top_event_words": top(stats["event_word_counts"], 80),
        "top_secondary_source_words": top(stats["secondary_source_word_counts"], 80),
        "paragraph_role_cue_counts": top(stats["role_cue_counts"], 40),
        "outcome_cue_counts": top(stats["outcome_cue_counts"], 40),
        "known_issue_label_counts": top(stats["known_issue_label_counts"], 120),
        "known_issue_area_counts": top(stats["known_issue_area_counts"], 80),
        "examples": stats["examples"],
    }


def md_table(rows: list[list[Any]], headers: tuple[str, str], n: int = 25) -> str:
    lines = [f"| {headers[0]} | {headers[1]} |", "|---|---:|"]
    for key, count in rows[:n]:
        lines.append(f"| `{key}` | {count:,} |")
    return "\n".join(lines)


def write_md(profile: dict[str, Any]) -> None:
    anchor_counts = profile["anchor_counts"]
    current_fields = profile.get("sample_output", {}).get("fields", {})
    text = f"""# Court Enrichment Field Contract

This document is generated from a full scan of `data/court_considerations.csv`.
It is query-independent and does not use validation/test gold labels.

## Corpus Snapshot

- Rows scanned: {profile["row_count"]:,}
- Scan time: {profile["elapsed_seconds"]:.1f}s
- Input: `{profile["source"]}`
- Current smoke output inspected: `{profile.get("sample_output", {}).get("path", "not found")}`

### Languages

{md_table(profile["language_counts"], ("language", "rows"))}

### Citation Families

{md_table(profile["citation_kind_counts"], ("family", "rows"))}

## What The LLM Should Produce

The LLM should only produce query-neutral legal descriptors. It should not be trusted
as the source of truth for formal anchors.

Current notebook `rag_enrichment` fields:

`{", ".join(current_fields.get("rag_enrichment", []))}`

Recommended production `rag_enrichment` fields:

- `legal_area`: broad normalized area, preferably seeded from deterministic v4 metadata.
- `primary_domain`: stable high-level domain, e.g. criminal procedure, social insurance, tax.
- `secondary_domain`: narrower domain, e.g. pretrial detention, invalidity assessment.
- `legal_domain_path`: 2-6 stable levels from broad to narrow.
- `topic`, `subtopic`, `micro_topic`: increasingly specific English descriptors.
- `concepts_en`: 3-8 English legal concepts that an English query might use.
- `terms_original`: exact German/French/Italian legal terms from the paragraph.
- `doctrinal_rule`: only if the paragraph itself states a rule or legal standard.
- `legal_test`: only if the paragraph states or applies a test.
- `fact_pattern_tags`: concrete facts/procedural situation, not generic law words.
- `procedural_context`: appeal type, instance, procedural posture.
- `paragraph_role`: conservative role enum.
- `authority_role`: legal value of the paragraph.
- `outcome_signal`: conservative disposition signal.
- `specificity_score`: 0-1, low for boilerplate/procedural fragments.

Additional normalized anchor fields should be built deterministically after LLM output:

- `statute_anchors`: only article/section/SR references extracted by regex or existing metadata.
- `legal_source_anchors`: names of laws/regulations without article numbers.
- `case_anchors`: only federal/cantonal case identifiers, excluding self/page refs.
- `secondary_sources`: doctrine, commentaries, journals, author names.
- `document_or_plan_anchors`: plans, permits, contracts, reports, policies, administrative documents.
- `event_anchors`: votes, initiatives, elections, dated public events.
- `self_references`: current `citation`, current `court_base`, and same-case page references.
- `anchor_quality_flags`: lists of moved/dropped anchors and why.

## Deterministic Anchor Counts In The Full Corpus

- Article anchors: {anchor_counts["article_anchors"]:,}
- Cantonal section anchors: {anchor_counts["cantonal_section_anchors"]:,}
- SR/RS anchors: {anchor_counts["sr_rs_anchors"]:,}
- BGE/ATF/DTF case anchors: {anchor_counts["bge_atf_case_anchors"]:,}
- Federal docket case anchors: {anchor_counts["federal_docket_case_anchors"]:,}
- Cantonal/special decision anchors: {anchor_counts["cantonal_or_special_decision_anchors"]:,}
- BGE page references: {anchor_counts["page_ref_anchors"]:,}

## Statute Anchor Rules

`statute_anchors` must be deterministic. Accept only:

- `Art. ... CODE`, e.g. `Art. 221 Abs. 1 lit. b StPO`, `art. 34 Cst.`
- `§ ... CODE`, e.g. `§ 25 Abs. 3 PBG/SZ`
- `SR ...` / `RS ...` numbers

Do not allow the model to put plain law names, plans, events, bibliography, or cases
inside `statute_anchors`. Move them to the correct fields.

Top code-like tokens after article/section anchors:

{md_table(profile["top_law_codes_from_text"], ("code/token", "matches"), 40)}

Legal-source words that should become `legal_source_anchors`, not `statute_anchors`:

{md_table(profile["top_legal_source_words"], ("source word", "rows"), 30)}

## Case Anchor Rules

`case_anchors` should include:

- external `BGE`, `ATF`, or `DTF` references with usable volume/division/page
- external Federal Tribunal docket references such as `1B_357/2022`
- cantonal/special decisions only in a separate `non_federal_decision_anchors` or normalized `case_anchors` subfield

Drop or move:

- current `citation`
- current `court_base`
- same-case page references such as `BGE 139 I 2 S. 8`
- bibliography/commentary entries
- author names and journal references

Secondary-source indicators found in the corpus:

{md_table(profile["top_secondary_source_words"], ("secondary source signal", "rows"), 30)}

## Document, Plan, And Event Anchors

These are useful for retrieval but must not pollute statute/case anchors.

Document/plan signals:

{md_table(profile["top_document_words"], ("document signal", "rows"), 30)}

Event signals:

{md_table(profile["top_event_words"], ("event signal", "rows"), 25)}

## Paragraph Role Rules

Recommended enum:

`holding`, `reasoning`, `facts`, `procedural_history`, `legal_standard`, `application`, `citation`, `costs`, `notification`, `disposition`, `neutral`

Corpus role cue counts:

{md_table(profile["paragraph_role_cue_counts"], ("role cue", "rows"), 25)}

Rules:

- If text is notification/cost boilerplate, do not create doctrinal rules.
- If role is `facts` or `procedural_history`, `doctrinal_rule` should be empty or descriptive only.
- If role is `legal_standard`, rule/test may be normative.
- If role is `application` or `holding`, rule/test may describe how the court applied the rule.

## Outcome Signal Rules

Recommended enum:

`neutral`, `dismissed`, `granted`, `partial`, `inadmissible`, `remitted`, `none`

Corpus outcome cue counts:

{md_table(profile["outcome_cue_counts"], ("outcome cue", "rows"), 20)}

Rules:

- Default to `neutral` or `none`.
- Set `remitted` only when the paragraph states the court remitted the matter, not merely because a party requested remand.
- Set `dismissed`, `granted`, `partial`, or `inadmissible` only on disposition/holding paragraphs with explicit court language.
- Do not infer outcome from legal argument paragraphs.

## Known Legal Issue Coverage

The current deterministic concept dictionary already detects these issue labels.
The LLM must be allowed to produce more specific `topic/subtopic/micro_topic`, but
these labels are good retrieval and QA guardrails.

{md_table(profile["known_issue_label_counts"], ("issue label", "rows"), 60)}

Areas represented by known issue labels:

{md_table(profile["known_issue_area_counts"], ("issue area", "rows"), 30)}

## Notebook Patch Requirements

1. Pass deterministic metadata into the prompt: `court_base`, `legal_area`, `law_codes`, `statutes_cited`, `court_cases_cited`, `authority_role`, and structural year/prefix/division when available.
2. Tell the model that formal anchors are advisory only; final anchors are cleaned deterministically.
3. Replace model-trusted `statute_anchors` with regex-extracted statute anchors plus metadata anchors.
4. Split anchor-like strings into `legal_source_anchors`, `document_or_plan_anchors`, `event_anchors`, and `secondary_sources`.
5. Filter `case_anchors` to remove self references, page refs, author/book names, and journals.
6. Make `outcome_signal` conservative using deterministic disposition phrases.
7. Suppress normative `doctrinal_rule` for facts/procedural-history/cost/notification paragraphs.
8. Build retrieval views from cleaned fields, not raw model fields.

## Retrieval-Safe Views

Use separate views:

- `semantic_concepts_en`: legal area/path/topic/subtopic/micro_topic/concepts/fact tags.
- `original_terms_view`: exact source-language legal terms.
- `statute_anchor_view`: cleaned statute anchors plus statute-related concepts.
- `case_anchor_view`: cleaned external case anchors plus court base.
- `legal_rule_view`: doctrinal rule and legal test only when role permits.
- `procedural_view`: procedural context, paragraph role, outcome.
- `authority_view`: v4 authority role, source count, published/frequently-cited flags.
- `raw_context`: clipped original text.

The retrieval system should never index one uncleaned giant blob as the only view.
"""
    PROFILE_MD.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_MD.write_text(text, encoding="utf-8")


def main() -> int:
    print(f"[start] scanning {COURT_CSV}")
    sample = scan_current_sample()
    stats = scan_corpus(sample)
    profile = serialize_profile(stats, sample)
    PROFILE_JSON.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_JSON.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(profile)
    print(f"[write] {PROFILE_JSON}")
    print(f"[write] {PROFILE_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
