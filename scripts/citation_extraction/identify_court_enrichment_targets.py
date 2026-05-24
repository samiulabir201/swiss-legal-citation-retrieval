#!/usr/bin/env python
"""Select court authority cards that deserve LLM enrichment first.

This script is deliberately query/gold independent. It scans
artifacts/court_authority_cards_v4.jsonl and identifies rows that are:

  1. not reliably covered by deterministic v4 metadata, and/or
  2. high-value authorities worth enriching even when deterministic labels exist.

Output is a priority-sorted JSONL manifest consumed by the Colab enrichment
notebook. Row identifiers use the notebook's zero-based JSONL line index.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"
DEFAULT_INPUT = ART_DIR / "court_authority_cards_v4.jsonl"
DEFAULT_OUTPUT = ART_DIR / "court_authority_cards_v4_enrichment_targets.jsonl"
DEFAULT_SUMMARY = ART_DIR / "court_authority_cards_v4_enrichment_targets.summary.json"


KNOWN_LAW_CODES = {
    "AIG", "AI", "ALC", "AMLA", "ATSG", "AVIG", "AHVG", "AsylG", "AuG",
    "BankG", "BetmG", "BGFA", "BGG", "BVG", "BV", "CC", "CEDH", "CO",
    "CPC", "CP", "CPP", "Cst", "DBG", "DSG", "EMRK", "FINMAG", "IVG",
    "IPRG", "KG", "KVG", "LAI", "LAMal", "LAVS", "LEI", "LEtr", "LIFD",
    "LP", "LPGA", "LTF", "LPP", "LStup", "LAsi", "LAA", "MSchG", "MWSTG",
    "NHG", "OJ", "OG", "OR", "PatG", "RPG", "SchKG", "StGB", "StPO",
    "SVG", "UVG", "URG", "UWG", "VwVG", "ZGB", "ZPO",
}

COST_RE = re.compile(
    r"("
    r"\bgerichtskosten\b|\bprozesskosten\b|\bverfahrenskosten\b|"
    r"\bfrais judiciaires\b|\bfrais de la cause\b|\bd[eé]pens\b|"
    r"\bspese giudiziarie\b|\bripetibili\b|"
    r"\bparteientsch[äa]digung\b|\bhonoraire\b|"
    r"\bunentgeltliche rechtspflege\b|\bassistance judiciaire\b|"
    r"\bpatrocinio gratuito\b"
    r")",
    re.IGNORECASE,
)

REMITTAL_RE = re.compile(
    r"("
    r"\b(?:die\s+)?sache\s+wird.{0,140}\bzur(?:[üu?]ck|ueck)gewiesen\b|"
    r"\bzur\s+(?:neuen|erneuten)\s+(?:entscheidung|beurteilung|neubeurteilung).{0,120}\bzur(?:[üu?]ck|ueck)gewiesen\b|"
    r"\ban\s+die\s+(?:vorinstanz|beschwerdegegnerin|verwaltung|beh[öo?]rde).{0,140}\bzur(?:[üu?]ck|ueck)gewiesen\b|"
    r"\brenvoie\s+la\s+cause\b|\bla\s+cause\s+est\s+renvoy[ée]e\b|"
    r"\brenvoy[ée]\s+.{0,80}\b(?:l'autorit[ée]|tribunal|instance)\b|"
    r"\bla\s+causa\s+[èe]\s+rinviata\b|\brinvia\s+la\s+causa\b"
    r")",
    re.IGNORECASE,
)

DISPOSITION_RE = re.compile(
    r"("
    r"\bbeschwerde\s+wird\s+(?:abgewiesen|gutgeheissen)\b|"
    r"\bauf\s+die\s+beschwerde\s+wird\s+nicht\s+eingetreten\b|"
    r"\ble\s+recours\s+est\s+(?:rejet[ée]|admis|irrecevable)\b|"
    r"\bil\s+ricorso\s+[èe]\s+(?:respinto|accolto|inammissibile)\b"
    r")",
    re.IGNORECASE,
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\x00", " ").split())


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [clean(v) for v in value if clean(v)]
    if value:
        return [clean(value)]
    return []


def invalid_law_codes(values: Iterable[Any]) -> list[str]:
    bad = []
    for value in values:
        code = clean(value).strip(".,;:()[]{}")
        if not code:
            continue
        if code not in KNOWN_LAW_CODES:
            bad.append(code)
    return sorted(set(bad))


def score_card(line_idx: int, card: dict[str, Any]) -> dict[str, Any]:
    text = clean(card.get("text_excerpt_original"))
    retrieval = clean(card.get("retrieval_text_en"))
    labels = as_list(card.get("issue_labels_en"))
    matched_terms = card.get("matched_terms_multilingual")
    matched_terms_count = 0
    if isinstance(matched_terms, dict):
        for terms in matched_terms.values():
            matched_terms_count += len(as_list(terms))
    law_codes = as_list(card.get("law_codes"))
    statutes = as_list(card.get("statutes_cited"))
    cited_cases = as_list(card.get("court_cases_cited"))
    roles = set(as_list(card.get("authority_role")))
    structural = card.get("structural") if isinstance(card.get("structural"), dict) else {}
    source_count = int(structural.get("court_base_source_count") or 0)
    text_ref_count = int(structural.get("court_base_text_ref_count") or 0)

    reasons: list[str] = []
    reliability_flags: list[str] = []

    if card.get("is_notification_paragraph"):
        reliability_flags.append("notification_paragraph")
    if len(text) < 80:
        reliability_flags.append("short_or_fragment")
    if COST_RE.search(text[:500]):
        reliability_flags.append("cost_or_fee_paragraph")
    if REMITTAL_RE.search(text[:800]):
        reliability_flags.append("remittal_disposition")
    elif DISPOSITION_RE.search(text[:500]):
        reliability_flags.append("disposition_paragraph")

    if not labels:
        reasons.append("missing_issue_labels")
    if matched_terms_count == 0:
        reasons.append("missing_multilingual_terms")
    if len(retrieval.split()) < 20:
        reasons.append("thin_retrieval_text")
    if card.get("language") in (None, "", "unknown"):
        reasons.append("unknown_language")
    bad_laws = invalid_law_codes(law_codes)
    if bad_laws:
        reasons.append("noisy_law_code_detection")

    high_value_reasons: list[str] = []
    if "published_leading_decision" in roles:
        high_value_reasons.append("published_bge")
    if "frequently_cited_authority" in roles:
        high_value_reasons.append("frequently_cited_authority")
    if source_count >= 20:
        high_value_reasons.append("high_court_base_source_count")
    elif source_count >= 10:
        high_value_reasons.append("medium_court_base_source_count")
    if text_ref_count >= 10:
        high_value_reasons.append("high_court_base_text_ref_count")
    elif text_ref_count >= 3:
        high_value_reasons.append("medium_court_base_text_ref_count")
    if statutes:
        high_value_reasons.append("has_statute_links")
    if cited_cases:
        high_value_reasons.append("has_court_case_links")

    non_trivial = (
        len(text) >= 120
        and not card.get("is_notification_paragraph")
        and "short_or_fragment" not in reliability_flags
        and "cost_or_fee_paragraph" not in reliability_flags
        and "remittal_disposition" not in reliability_flags
        and "disposition_paragraph" not in reliability_flags
    )

    priority = 0.0
    priority += min(len(text) / 120.0, 10.0)
    priority += min(source_count, 60) * 0.9
    priority += min(text_ref_count, 60) * 1.3
    if "published_bge" in high_value_reasons:
        priority += 45
    if "frequently_cited_authority" in high_value_reasons:
        priority += 30
    if statutes:
        priority += 12
    if cited_cases:
        priority += 10
    if "missing_issue_labels" in reasons:
        priority += 25
    if "missing_multilingual_terms" in reasons:
        priority += 15
    if "thin_retrieval_text" in reasons:
        priority += 18
    if "unknown_language" in reasons:
        priority += 8
    if "noisy_law_code_detection" in reasons:
        priority += 10
    if not non_trivial:
        priority -= 70

    should_target = non_trivial and (
        bool(reasons)
        or "published_bge" in high_value_reasons
        or "frequently_cited_authority" in high_value_reasons
        or source_count >= 20
        or text_ref_count >= 10
    )

    return {
        "line_idx": line_idx,
        "csv_row": line_idx + 2,
        "citation": clean(card.get("citation")),
        "court_base": clean(card.get("court_base")),
        "priority": round(priority, 4),
        "should_target": should_target,
        "reasons": reasons,
        "high_value_reasons": high_value_reasons,
        "reliability_flags": reliability_flags,
        "text_chars": len(text),
        "retrieval_words": len(retrieval.split()),
        "issue_label_count": len(labels),
        "matched_term_count": matched_terms_count,
        "statute_count": len(statutes),
        "cited_case_count": len(cited_cases),
        "source_count": source_count,
        "text_ref_count": text_ref_count,
        "bad_law_codes": bad_laws,
        "legal_area": clean(card.get("legal_area")),
        "language": clean(card.get("language")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--min-priority", type=float, default=65.0)
    ap.add_argument("--max-targets", type=int, default=0, help="0 means no cap.")
    ap.add_argument("--progress-every", type=int, default=250000)
    args = ap.parse_args()

    targets: list[dict[str, Any]] = []
    stats = Counter()
    reasons = Counter()
    high_value = Counter()
    reliability = Counter()
    areas = Counter()
    languages = Counter()

    with args.input.open(encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                card = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_json"] += 1
                continue

            stats["cards"] += 1
            scored = score_card(line_idx, card)
            if scored["should_target"]:
                stats["candidate_targets_before_threshold"] += 1
            if scored["should_target"] and scored["priority"] >= args.min_priority:
                targets.append(scored)
                stats["selected_targets_before_cap"] += 1
                for reason in scored["reasons"]:
                    reasons[reason] += 1
                for reason in scored["high_value_reasons"]:
                    high_value[reason] += 1
                for flag in scored["reliability_flags"]:
                    reliability[flag] += 1
                areas[scored["legal_area"]] += 1
                languages[scored["language"]] += 1

            if args.progress_every and stats["cards"] % args.progress_every == 0:
                print(f"[scan] cards={stats['cards']:,} selected={len(targets):,}", flush=True)

    targets.sort(key=lambda x: (-x["priority"], x["line_idx"]))
    if args.max_targets:
        targets = targets[: args.max_targets]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for row in targets:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "min_priority": args.min_priority,
        "max_targets": args.max_targets,
        "stats": dict(stats),
        "final_target_count": len(targets),
        "top_reasons": reasons.most_common(50),
        "top_high_value_reasons": high_value.most_common(50),
        "top_reliability_flags": reliability.most_common(50),
        "top_legal_areas": areas.most_common(30),
        "top_languages": languages.most_common(10),
        "top_targets": targets[:20],
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] cards={stats['cards']:,} targets={len(targets):,}")
    print(f"[done] output={args.output}")
    print(f"[done] summary={args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
