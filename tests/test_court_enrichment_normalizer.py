from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from court_enrichment_normalizer import normalize_enriched_court_row


def test_normalizer_moves_bad_llm_anchors_and_removes_self_refs() -> None:
    text = (
        "Gemäss Art. 221 Abs. 1 lit. b StPO und § 25 Abs. 3 PBG/SZ ist die "
        "Rechtsprechung zu beachten. Vgl. BGE 139 I 2 S. 8, BGE 137 IV 122 E. 4.2 "
        "und Urteil 1B_575/2021 vom 8. November 2021. Der Zonenplan und die "
        "Volksabstimmung werden erwähnt; siehe auch Basler Kommentar und ZBl 112/2011."
    )
    llm = {
        "legal_area": "criminal procedure",
        "topic": "collusion risk",
        "concepts_en": ["pretrial detention"],
        "paragraph_role": "reasoning",
        "outcome_signal": "granted",
        "statute_anchors": ["Art. 221 Abs. 1 lit. b StPO", "Zonenplan", "Bundesgesetz"],
        "case_anchors": ["BGE 139 I 2 S. 8", "Basler Kommentar", "BGE 137 IV 122 E. 4.2"],
        "doctrinal_rule": "A supported legal standard.",
        "legal_test": "A supported legal test.",
    }

    result = normalize_enriched_court_row(
        "BGE 139 I 2 E. 7.1",
        text,
        llm,
        {"court_base": "BGE 139 I 2", "statutes_cited": ["SR 312.0"]},
    )

    anchors = result["normalized_anchors"]
    assert "Art. 221 Abs. 1 lit. b StPO" in anchors["statute_anchors"]
    assert "§ 25 Abs. 3 PBG/SZ" in anchors["statute_anchors"]
    assert "SR 312.0" in anchors["statute_anchors"]
    assert "Zonenplan" not in anchors["statute_anchors"]
    assert "BGE 137 IV 122 E. 4.2" in anchors["case_anchors"]
    assert "BGE 137 IV 122 E. 4.2." not in anchors["case_anchors"]
    assert "1B_575/2021" in anchors["case_anchors"]
    assert all("BGE 139 I 2" not in case for case in anchors["case_anchors"])
    assert "BGE 139 I 2 S. 8" in anchors["page_references"]
    assert "Basler Kommentar" in anchors["secondary_sources"]
    assert "Zonenplan" in anchors["document_or_plan_anchors"]
    assert "Volksabstimmung" in anchors["event_anchors"]
    assert "Bundesgesetz" in anchors["legal_source_anchors"]
    assert result["rag_enrichment"]["outcome_signal"] == "none"
    assert "Zonenplan" not in result["retrieval_views"]["statute_anchor_view"]
    assert "Basler Kommentar" not in result["retrieval_views"]["case_anchor_view"]


def test_normalizer_suppresses_rules_for_cost_rows() -> None:
    text = "Die Gerichtskosten von Fr. 1'000.-- werden dem Beschwerdeführer auferlegt."
    result = normalize_enriched_court_row(
        "1B_28/2022 E. 6",
        text,
        {
            "paragraph_role": "holding",
            "outcome_signal": "dismissed",
            "doctrinal_rule": "Costs follow the event.",
            "legal_test": "Cost allocation test.",
            "legal_rule": "A cost rule.",
        },
    )

    rag = result["rag_enrichment"]
    assert rag["paragraph_role"] == "costs"
    assert rag["outcome_signal"] == "none"
    assert rag["doctrinal_rule"] == ""
    assert rag["legal_test"] == ""
    assert rag["legal_rule"] == ""
    assert result["retrieval_views"]["legal_rule_view"] == ""
    assert result["enrichment_quality"]["low_value_paragraph"] is True


def test_normalizer_drops_generated_summaries_and_questions() -> None:
    result = normalize_enriched_court_row(
        "1B_28/2022 E. 4.1",
        "Gemäss Art. 221 Abs. 1 lit. b StPO gilt ein strenger Massstab.",
        {
            "english_summary": "Generated summary that must not ship.",
            "summary_en": "Another generated summary.",
            "legal_question": "Generated question that must not ship?",
            "natural_language_queries": ["generated query"],
            "query_phrases_en": ["generated phrase"],
            "legal_topic": "collusion risk",
            "paragraph_role": "reasoning",
            "outcome_signal": "none",
        },
    )

    rag = result["rag_enrichment"]
    for key in {
        "english_summary",
        "summary_en",
        "legal_question",
        "natural_language_queries",
        "query_phrases_en",
    }:
        assert key not in rag

    views_blob = "\n".join(result["retrieval_views"].values())
    assert "Generated summary" not in views_blob
    assert "Generated question" not in views_blob
    assert "generated query" not in views_blob
    assert "generated phrase" not in views_blob
