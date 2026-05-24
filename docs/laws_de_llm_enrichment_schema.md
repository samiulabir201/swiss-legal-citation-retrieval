# Laws-DE LLM Enrichment Schema (v2)

Companion to `laws_de_authority_card_static_report.md`. This document fixes the
contract for the law-side LLM run that produces `artifacts/law_authority_cards_v2_unified.jsonl`.

## 1. What is decided STATICALLY (no LLM)

These fields are deterministic from `citation`, `title`, the citation graph
artifacts, and dictionary heuristics. The LLM must NEVER overwrite them.

| Field | Source | Coverage |
|---|---|---|
| `_source_row`, `citation`, `language` ("de"), `family` ("law"), `pattern` | builder | 100% |
| `structural.article`, `structural.units`, `structural.law_code`, `structural.law_code_family`, `structural.granularity` | citation parser, with Unicode repair from exact citation text (4,117 rows) | 100% |
| `law_title`, `title_section_path` | CSV `title` column split on " - " | 100% / 95.7% |
| `title_metadata.source_type` (federal_act / ordinance / constitution / regulation / agreement / federal_decree / statutes / other) | title-prefix rules | 100% |
| `title_metadata.enactment_date`, `title_metadata.enactment_year` | regex `vom DD. MMMM YYYY` on title | 99.9% (175,728 / 175,933) |
| `title_metadata.law_aliases` | parenthetical short-form parsing | 100% of acts with parentheses |
| `title_metadata.systematic_collection_sector` (1..9 SR sector + label) | first SR-number digit | 100% of SR-numbered citations |
| `legal_area_static`, `domain_labels_en`, `issue_labels_en`, `matched_terms_multilingual`, `provision_roles_static` | dictionary + cue rules | 100% (formal labels), 92% non-empty |
| `statute_anchors`, `official_references`, `court_case_anchors`, `other_reference_anchors` | citation graph extractor | 31,219 rows have outgoing refs |
| `incoming_reference_count`, `incoming_reference_examples` | back-edges in `laws_de_links.json` | 10,409 rows have incoming refs |
| `adjacent_citations` (next/prev/same-article siblings/same-law count) | corpus pass | 100% |

Anything LLM produces that **conflicts** with these is dropped at merge time.

## 2. What MUST come from the LLM

The LLM is the only reliable way to translate, interpret, and disambiguate the
operative semantic content of multilingual statutory text into English, while
preserving the original-language legal terms. It is **not** a translation task —
the model must produce *legal* English equivalents (e.g. `Bewilligung` →
`permit / authorisation`, not "approval"; `Rechtsbegehren` → `prayer for relief`,
not "legal request").

**LLM output schema (12 fields, JSON only, all keys present):**

| Field | Type | Constraints |
|---|---|---|
| `english_summary` | string | ≤ 2 sentences, ≤ 60 words. What the provision says, in English. |
| `legal_rule` | string | ≤ 25 words. The operative rule in form "X must/may/shall/is forbidden to Y". Empty for transitional / fee-schedule / annex-list / commencement provisions. |
| `applicability_conditions` | list[string] | 0–5 items, each ≤ 14 words. When this rule applies (triggers, prerequisites). |
| `exceptions_or_limitations` | list[string] | 0–5 items, each ≤ 14 words. Express exceptions, "unless", "except where", time limits. |
| `legal_question` | string | ≤ 18 words. One English question this provision answers. Empty for boilerplate. |
| `concepts_en` | list[string] | 3–8 broad English legal concepts. |
| `terms_de_to_en` | list[{de, en}] | 3–10 pairs. `de` MUST be a verbatim substring of source text. `en` is the **legal English equivalent**, not a literal translation. |
| `defined_terms` | list[{term, definition}] | 0–4 items. Only when the article explicitly defines a term ("Im Sinne …", "gilt als …", "bedeutet"). `term` verbatim from text. |
| `addressees` | list[string] | 0–6 English labels: who is bound (e.g. "federal authority", "competent cantonal authority", "employer", "data subject", "taxpayer", "carrier"). |
| `sanctions_or_consequences` | list[string] | 0–4 English items. Penalties, denials, revocations, fees **explicitly named in the text**. |
| `provision_role_llm` | enum | One of: `definition`, `purpose`, `scope`, `principle`, `right_or_entitlement`, `duty`, `prohibition`, `procedure`, `competence`, `sanction_or_penalty`, `data_reporting`, `fees_or_costs`, `transitional_or_commencement`, `other`. |
| `specificity_score` | float | 0..1. 0 = generic boilerplate, 1 = very specific operative rule. |

**Hard rules baked into the prompt:**
1. JSON only, no prose.
2. English for semantic fields. Original German for `terms_de_to_en[].de` and `defined_terms[].term` (must be substrings of source text).
3. Do **not** invent statute citations, BGE numbers, dates, or party names that are not in the text.
4. Do **not** translate literally — choose the recognised Swiss legal English equivalent.
5. For `transitional_or_commencement`, `fees_or_costs`, `data_reporting`, and pure annex-list rows: keep `legal_rule`, `applicability_conditions`, `exceptions_or_limitations`, `legal_question` empty. Provide only `english_summary`, `concepts_en`, `terms_de_to_en`, `provision_role_llm`, `specificity_score`.
6. Never copy the law title into `english_summary`.
7. Never put `Art.` / `Abs.` / SR numbers into `terms_de_to_en` — those are anchors, not terms.

## 3. Final unified card schema (v2)

```jsonc
{
  "_source_row": 140478,
  "citation": "Art. 30f Abs. 3 USG",
  "language": "de",
  "family": "law",
  "law_title": "Bundesgesetz vom 7. Oktober 1983 über den Umweltschutz (Umweltschutzgesetz, USG)",
  "title_section_path": "1. Abschnitt: Vermeidung und Entsorgung von Abfällen",

  "structural": {
    "article": "30f",
    "units": ["Abs. 3"],
    "law_code": "USG",
    "law_code_family": "uppercase_abbreviation",
    "granularity": "article_paragraph"
  },
  "title_metadata": {
    "source_type": "federal_act",
    "enactment_date": "1983-10-07",
    "enactment_year": "1983",
    "law_aliases": ["Umweltschutzgesetz", "USG"],
    "systematic_collection_sector": null
  },

  "enrichment_source": "llm+static",          // or "static" if LLM skipped/failed

  "rag_enrichment": {
    // ── static ─────────────────────────────────────────────────────────
    "legal_area": "environmental law",
    "primary_domain": "environmental protection",
    "secondary_domain": "waste management",
    "legal_domain_path": ["environmental law", "federal_act", "environmental protection"],
    "domain_labels_en": ["environmental protection", "federal_act"],
    "issue_labels_en": ["environmental protection", "waste management"],
    "matched_terms_multilingual": {"de": ["Bewilligung", "Entsorgung"]},
    "provision_roles_static": ["competence"],

    // ── LLM (12 fields) ────────────────────────────────────────────────
    "english_summary": "Permits for waste disposal facilities are granted only when environmentally sound disposal is guaranteed.",
    "legal_rule": "Permits are granted only if environmentally sound disposal is assured.",
    "applicability_conditions": ["application for waste disposal permit"],
    "exceptions_or_limitations": [],
    "legal_question": "Under what conditions is a waste-disposal permit issued?",
    "concepts_en": ["Waste disposal", "Environmental compatibility", "Permit", "Environmental law"],
    "terms_de_to_en": [
      {"de": "Bewilligungen", "en": "permits / authorisations"},
      {"de": "umweltverträgliche Entsorgung", "en": "environmentally sound disposal"},
      {"de": "Abfälle", "en": "waste"}
    ],
    "defined_terms": [],
    "addressees": ["competent authority", "permit applicant"],
    "sanctions_or_consequences": [],
    "provision_role_llm": "competence",
    "specificity_score": 0.7
  },

  "normalized_anchors": { "...static..." },
  "retrieval_views": {
    "citation_view": "Art. 30f Abs. 3 USG | USG | SR 814.01",
    "title_view": "Bundesgesetz ... Umweltschutzgesetz | 1. Abschnitt: Vermeidung und Entsorgung von Abfällen",
    "structure_view": "art:30f abs:3 code:USG family:uppercase_abbreviation",
    "english_summary_view": "<english_summary>",
    "legal_rule_view": "<legal_rule>",
    "rule_components_view": "applies: ... | exceptions: ...",
    "concepts_en_view": "Waste disposal | Environmental compatibility | Permit | Environmental law",
    "terms_bilingual_view": "Bewilligungen→permits / authorisations | umweltverträgliche Entsorgung→environmentally sound disposal",
    "addressees_view": "competent authority | permit applicant",
    "statute_anchor_view": "<from static>",
    "law_context_view": "<adjacent>",
    "raw_context": "<original German text>"
  },

  "enrichment_quality": {
    "static_formal_complete": true,
    "needs_llm_for_complete_semantic_context": true,
    "llm_priority": "high",
    "llm_status": "ok",
    "llm_json_valid": true,
    "llm_attempt_count": 1,
    "terms_grounded_pct": 1.0,           // fraction of terms_de_to_en[].de found verbatim in text
    "boilerplate_role": false,
    "rule_fields_suppressed": false,     // true if role is boilerplate → rule fields blanked
    "text_char_count": 92,
    "title_char_count": 91
  }
}
```

## 4. Volume

Process **all 175,933 rows**. Estimated 60–90 min on RTX PRO 6000 Blackwell with
Qwen3-8B-AWQ at the same settings used for the 363k court run. Median text is
181 chars vs court p50 ~600 chars, so per-row latency is lower; the smaller
schema (12 fields vs 16) further reduces decode tokens.

The post-merge normalizer suppresses `legal_rule` / `applicability_conditions` /
`exceptions_or_limitations` / `legal_question` for any row whose merged
`provision_role` resolves to `transitional_or_commencement`, `fees_or_costs`,
`data_reporting`, or where the source text matches "Aufgehoben" / "Tritt … in
Kraft" — even if the LLM produced something. This prevents hallucinated
operative rules on boilerplate.
