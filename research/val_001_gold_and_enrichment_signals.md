# val_001 - Gold Citations & Enrichment Signals

_Generated 2026-05-09, revised same day to add static-enrichment coverage._

Sources used:
- `data/val.csv` — query + gold
- `law_json_llm_output/law_llm_descriptors_0000000_all.jsonl` — law LLM enrichment (173,033 records, 100% of laws_de)
- `outputs_from_363k_run/court_llm_descriptors_0000000_all.jsonl` — court LLM enrichment (363,258 records, ~15% of the 2.47M court corpus)
- `artifacts/court_authority_cards_v5_unified.jsonl` — **the unified court file**: all 2,476,315 court rows in a single normalized schema with an `enrichment_source` flag. 363,257 rows are `"llm+static"` (the 363k LLM payload merged with static anchors); 2,113,058 rows are `"static"`-only (rule-based fields, no LLM call). This is the file that closes the gap for the 10 court gold the 363k run did not cover.

---

## 1. The val_001 query

**query_id:** `val_001`

> May a court lawfully order a three‑month extension of pre‑trial detention under Art. 221 Abs. 1 lit. b StPO (risk of collusion) consistent with the principle of proportionality when the accused—detained after an alleged late‑night assault and theft of a courier satchel containing, inter alia, €5,600—was remanded by an order dated 18 October 2024 for a maximum period up to 15 January 2025, the prosecutor sought an extension on 10 December 2024 primarily citing a concrete risk that the detainee would influence witnesses or tamper with evidence and a risk of reoffending, while the detainee opposed the extension on the ground that most witnesses have already been interviewed, the investigative steps still pending are essentially technical (phone data extraction, CCTV image analysis, and bank record checks), his release would therefore not jeopardize the inquiry, and the alleged victim has withdrawn the complaint—i.e. does the asserted concrete risk of collusion and considerations of proportionality justify a three‑month prolongation in these circumstances?

Domain in plain words: criminal procedure - extension of pre-trial detention under Art. 221 Abs. 1 lit. b StPO on the grounds of collusion risk (Kollusionsgefahr), tested against the principle of proportionality. The query bundles a fact pattern (late-night assault, theft of a courier satchel, prosecution motion for a 3-month prolongation, withdrawn complaint) onto a tight cluster of StPO articles plus Federal Court (BGE / 1B_ / 7B_) doctrine on detention.

---

## 2. Gold citations (42 total)

**Law-style (19):** `Art. 221 Abs. 1 StPO`, `Art. 140 Abs. 1 StGB`, `Art. 396 Abs. 1 StPO`, `Art. 222 StPO`, `Art. 393 Abs. 1 StPO`, `Art. 382 Abs. 1 StPO`, `Art. 385 Abs. 1 StPO`, `Art. 221 Abs. 2 StPO`, `Art. 227 Abs. 1 StPO`, `Art. 212 Abs. 3 StPO`, `Art. 390 Abs. 2 StPO`, `Art. 422 Abs. 1 StPO`, `Art. 422 Abs. 2 StPO`, `Art. 428 Abs. 1 StPO`, `Art. 135 Abs. 4 StPO`, `Art. 100 Abs. 1 BGG`, `Art. 135 Abs. 3 StPO`, `Art. 37 Abs. 1 StBOG`, `Art. 39 Abs. 1 StBOG`

**Court-style (23):** `BGE 137 IV 122 E. 6.2`, `BGE 137 IV 122 E. 6.4`, `BGE 137 IV 122 E. 4.2`, `BGE 132 I 21 E. 3.2`, `1B_210/2023 E. 4.1`, `BGE 132 I 21 E. 3.2.2`, `1B_536/2018 E. 5.1`, `BGE 139 IV 270 E. 3.1`, `BGE 133 I 168 E. 4.1`, `BGE 143 IV 168 E. 5.1`, `BGE 133 I 270 E. 3.4.2`, `BGE 137 IV 122 E. 4.1`, `BGE 132 I 21 E. 3.2.1`, `1B_90/2021 E. 2.1`, `1B_90/2021 E. 2.4`, `7B_496/2025 E. 3.2`, `7B_231/2025 E. 4.1`, `7B_69/2024 E. 3.3.2`, `7B_301/2024 E. 2.4`, `7B_12/2025 E. 2.2`, `1B_357/2022 E. 3.1`, `1B_15/2023 E. 3.1`, `1B_28/2022 E. 4.1`

---

## 3. Enrichment coverage

Coverage measured at two layers: LLM-enriched payload (richer, English summaries / cross-lingual term maps / micro-topics) vs static-enriched payload (rule-based — anchors, role tags, regex-derived concept lists).

| Family | Total gold | LLM-enriched | Static-enriched (via v5_unified) | No enrichment at all |
|---|---:|---:|---:|---:|
| Law (`law_llm_descriptors_0000000_all.jsonl`, 173,033 records) | 19 | **19** (100%) | n/a (laws have a single enrichment file, no static fallback) | 0 |
| Court — LLM file (`court_llm_descriptors_0000000_all.jsonl`, 363k subset) | 23 | **13** (56.5%) | — | 10 |
| Court — unified (`court_authority_cards_v5_unified.jsonl`, all 2.47M rows) | 23 | **13** (`"llm+static"`) | **+10** (`"static"`-only) → **23/23 (100%)** | 0 |
| **Combined** | **42** | **32 (76.2%)** | **+10** → **42/42 (100%)** | 0 |

### 3.1 Court gold without an LLM payload (have static-only enrichment in v5_unified)

The following 10 are **not** in the 363k LLM run, but **are** in `court_authority_cards_v5_unified.jsonl` with `enrichment_source = "static"`. Section 4.3 below shows their static enrichment payloads.

- `1B_210/2023 E. 4.1`
- `1B_536/2018 E. 5.1`
- `1B_90/2021 E. 2.1`
- `1B_90/2021 E. 2.4`
- `7B_496/2025 E. 3.2`
- `7B_231/2025 E. 4.1`
- `7B_69/2024 E. 3.3.2`
- `7B_301/2024 E. 2.4`
- `7B_12/2025 E. 2.2`
- `1B_15/2023 E. 3.1`

All 10 are 1B_/7B_ Bundesgericht docket-style decisions. The handoff (section 3.5, 11) flags the 363k cap as the live coverage gap for *LLM* enrichment; the static layer fills it with deterministic-extraction signals (statute/case anchors, paragraph_role, concepts_en, terms_original, outcome_signal) that are still useful for retrieval.

---

## 4. Enrichment JSON for each gold citation present in the enrichment files

Each block below shows the full `llm_enrichment` payload (the `_source_row` and `llm_generation` envelope is omitted for brevity).

### 4.1 Law gold (19/19 present)

#### `Art. 221 Abs. 1 StPO`

```json
{
  "english_summary": "Investigative and security detention is allowed only if the accused is strongly suspected of a crime or offense and there is a serious risk that they will evade prosecution or sanctions.",
  "legal_rule": "Investigative and security detention permitted under specific suspicion and risk conditions",
  "applicability_conditions": [
    "Accused is strongly suspected of crime or offense",
    "Serious risk of evading prosecution or sanctions",
    "Risk of influencing persons or evidence",
    "Risk of endangering others through prior offenses"
  ],
  "exceptions_or_limitations": [],
  "legal_question": "When is investigative detention lawful under criminal procedure?",
  "concepts_en": [
    "Investigative detention",
    "Security detention",
    "Criminal suspicion",
    "Risk assessment",
    "Evidence tampering",
    "Public safety",
    "Prior offenses",
    "Legal proceedings"
  ],
  "terms_de_to_en": [
    {
      "de": "Untersuchungs- und Sicherheitshaft",
      "en": "Investigative and security detention"
    },
    {
      "de": "dringend verdächtig",
      "en": "strongly suspected"
    },
    {
      "de": "Sanktion",
      "en": "sanction"
    },
    {
      "de": "Beweismittel",
      "en": "evidence"
    },
    {
      "de": "Sicherheit anderer",
      "en": "public safety"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "court",
    "competent cantonal authority",
    "public prosecutor"
  ],
  "sanctions_or_consequences": [
    "Detention ordered by court",
    "Must be proportionate to risk",
    "Reviewable by judicial authority"
  ],
  "provision_role_llm": "scope",
  "specificity_score": 0.8
}
```

#### `Art. 140 Abs. 1 StGB`

```json
{
  "english_summary": "An act of theft committed by force or threat against a person is punishable by imprisonment from six months to ten years.",
  "legal_rule": "Theft by force or threat punishable by imprisonment",
  "applicability_conditions": [
    "Use of force against a person",
    "Threat of immediate danger to body or life",
    "Making the victim unable to resist",
    "Commission of theft"
  ],
  "exceptions_or_limitations": [],
  "legal_question": "What penalty applies for theft involving force or threat?",
  "concepts_en": [
    "criminal liability",
    "force",
    "threat",
    "theft",
    "imprisonment",
    "penal sanction",
    "criminal act",
    "victim"
  ],
  "terms_de_to_en": [
    {
      "de": "Gewalt",
      "en": "force"
    },
    {
      "de": "Gefahr",
      "en": "danger"
    },
    {
      "de": "Leib",
      "en": "body"
    },
    {
      "de": "Leben",
      "en": "life"
    },
    {
      "de": "Widerstand",
      "en": "resistance"
    },
    {
      "de": "Diebstahl",
      "en": "theft"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "court",
    "public prosecutor"
  ],
  "sanctions_or_consequences": [
    "imprisonment from six months to ten years"
  ],
  "provision_role_llm": "other",
  "specificity_score": 0.9
}
```

#### `Art. 396 Abs. 1 StPO`

```json
{
  "english_summary": "An appeal against written or oral decisions must be submitted in writing and with reasons within 10 days to the appellate authority.",
  "legal_rule": "Appeals against decisions must be submitted in writing with reasons within 10 days.",
  "applicability_conditions": [
    "Appeal against written decisions",
    "Appeal against oral decisions",
    "Submission must be in writing",
    "Reasons required",
    "Time limit of 10 days"
  ],
  "exceptions_or_limitations": [],
  "legal_question": "What is the procedural requirement for filing an appeal?",
  "concepts_en": [
    "Appeal",
    "Decision",
    "Appellate authority",
    "Time limit",
    "Written submission",
    "Reasons"
  ],
  "terms_de_to_en": [
    {
      "de": "Beschwerde",
      "en": "appeal"
    },
    {
      "de": "Entscheid",
      "en": "decision"
    },
    {
      "de": "Beschwerdeinstanz",
      "en": "appellate authority"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "natural person",
    "legal entity / undertaking"
  ],
  "sanctions_or_consequences": [],
  "provision_role_llm": "procedure",
  "specificity_score": 0.8
}
```

#### `Art. 222 StPO`

```json
{
  "english_summary": "Only the detained person can challenge decisions on investigation or security detention at the appellate instance. Article 233 remains excluded.",
  "legal_rule": "",
  "applicability_conditions": [
    "Decisions on investigation or security detention",
    "Challenged at the appellate instance",
    "Only applicable to detained persons"
  ],
  "exceptions_or_limitations": [
    "Exclusion of Article 233"
  ],
  "legal_question": "",
  "concepts_en": [
    "Detained person",
    "Appellate instance",
    "Investigation detention",
    "Security detention",
    "Administrative decision",
    "Legal challenge"
  ],
  "terms_de_to_en": [
    {
      "de": "Beschwerdeinstanz",
      "en": "appellate instance"
    },
    {
      "de": "Anordnung",
      "en": "order"
    },
    {
      "de": "Verlängerung",
      "en": "extension"
    },
    {
      "de": "Aufhebung",
      "en": "revocation"
    },
    {
      "de": "Untersuchungs- oder Sicherheitshaft",
      "en": "investigation or security detention"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "detained person"
  ],
  "sanctions_or_consequences": [],
  "provision_role_llm": "other",
  "specificity_score": 0.8
}
```

#### `Art. 393 Abs. 1 StPO`

```json
{
  "english_summary": "This article allows appeals against certain decisions and procedural acts by police, prosecutors, petty offense authorities, first instance courts, and the compulsory measures court.",
  "legal_rule": "Appeals are allowed against specified decisions and procedural acts",
  "applicability_conditions": [
    "Against decisions of police",
    "Against procedural acts of prosecutors",
    "Against decisions of petty offense authorities",
    "Against decisions of first instance courts",
    "Against compulsory measures court decisions"
  ],
  "exceptions_or_limitations": [
    "Excludes procedural decisions",
    "Excludes decisions not designated as final under this law"
  ],
  "legal_question": "What decisions and acts are appealable under this article?",
  "concepts_en": [
    "Appeal",
    "Procedural act",
    "Decision",
    "Competent authority",
    "Final decision",
    "Administrative body",
    "Judicial authority"
  ],
  "terms_de_to_en": [
    {
      "de": "Verfügungen",
      "en": "formal administrative order"
    },
    {
      "de": "Verfahrenshandlungen",
      "en": "procedural acts"
    },
    {
      "de": "erstinstanzliche Gerichte",
      "en": "first instance courts"
    },
    {
      "de": "Zwangsmassnahmengericht",
      "en": "compulsory measures court"
    },
    {
      "de": "verfahrensleitende Entscheide",
      "en": "procedural decisions"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "natural person",
    "legal entity / undertaking",
    "court",
    "public prosecutor",
    "supervisory authority"
  ],
  "sanctions_or_consequences": [],
  "provision_role_llm": "scope",
  "specificity_score": 0.8
}
```

#### `Art. 382 Abs. 1 StPO`

```json
{
  "english_summary": "Any party with a legally protected interest in revoking or changing a decision may file a legal remedy.",
  "legal_rule": "Parties with protected interests may seek legal remedies against decisions.",
  "applicability_conditions": [
    "Party has legally protected interest",
    "Decision is subject to revocation or amendment",
    "Legal remedy is available"
  ],
  "exceptions_or_limitations": [],
  "legal_question": "Who may seek legal remedies against decisions?",
  "concepts_en": [
    "legal remedy",
    "protected interest",
    "decision revocation",
    "party",
    "criminal procedure",
    "legal protection",
    "administrative action"
  ],
  "terms_de_to_en": [
    {
      "de": "Entscheide",
      "en": "decisions"
    },
    {
      "de": "Rechtsmittel",
      "en": "legal remedy"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "natural person",
    "legal entity"
  ],
  "sanctions_or_consequences": [],
  "provision_role_llm": "right_or_entitlement",
  "specificity_score": 0.8
}
```

#### `Art. 385 Abs. 1 StPO`

```json
{
  "english_summary": "This article requires that if a legal remedy is claimed, the person or authority initiating it must specify the points they challenge, the grounds suggesting another decision, and the evidence they invoke.",
  "legal_rule": "",
  "applicability_conditions": [
    "Legal remedy is claimed",
    "Person or authority initiates remedy"
  ],
  "exceptions_or_limitations": [],
  "legal_question": "",
  "concepts_en": [
    "legal remedy",
    "challenge",
    "grounds",
    "evidence",
    "decision",
    "authority",
    "person"
  ],
  "terms_de_to_en": [
    {
      "de": "Rechtsmittel",
      "en": "legal remedy"
    },
    {
      "de": "Entscheide",
      "en": "decisions"
    },
    {
      "de": "Beweismittel",
      "en": "evidence"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "person",
    "authority"
  ],
  "sanctions_or_consequences": [],
  "provision_role_llm": "procedure",
  "specificity_score": 0.8
}
```

#### `Art. 221 Abs. 2 StPO`

```json
{
  "english_summary": "Detention is permissible if there is a serious and immediate danger that a person will carry out a serious crime.",
  "legal_rule": "Detention allowed when serious and immediate danger of committing a serious crime exists",
  "applicability_conditions": [
    "Serious threat",
    "Immediate danger",
    "Person to commit serious crime"
  ],
  "exceptions_or_limitations": [],
  "legal_question": "When is detention permissible under Article 221(2) StPO?",
  "concepts_en": [
    "Detention",
    "Serious crime",
    "Threat",
    "Immediacy",
    "Danger",
    "Legal justification"
  ],
  "terms_de_to_en": [
    {
      "de": "Drohung",
      "en": "threat"
    },
    {
      "de": "schweres Verbrechen",
      "en": "serious crime"
    },
    {
      "de": "ernsthafte und unmittelbare Gefahr",
      "en": "serious and immediate danger"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "court",
    "law enforcement"
  ],
  "sanctions_or_consequences": [],
  "provision_role_llm": "other",
  "specificity_score": 0.8
}
```

#### `Art. 227 Abs. 1 StPO`

```json
{
  "english_summary": "If the duration of pre-trial detention set by the compulsory measures court expires, the prosecutor may apply for extension. If the court did not limit the detention period, the application must be filed within three months.",
  "legal_rule": "Prosecutor may apply for extension of pre-trial detention if court did not set duration",
  "applicability_conditions": [
    "Detention duration set by compulsory measures court has expired",
    "Prosecutor seeks extension",
    "Court did not limit detention period",
    "Application must be filed within three months"
  ],
  "exceptions_or_limitations": [],
  "legal_question": "When can prosecutor apply for extension of pre-trial detention?",
  "concepts_en": [
    "pre-trial detention",
    "compulsory measures court",
    "prosecutor",
    "detention extension",
    "duration limitation",
    "application filing",
    "criminal procedure"
  ],
  "terms_de_to_en": [
    {
      "de": "Zwangsmassnahmengericht",
      "en": "compulsory measures court"
    },
    {
      "de": "Staatsanwaltschaft",
      "en": "prosecutor"
    },
    {
      "de": "Haftverlängerungsgesuch",
      "en": "detention extension application"
    },
    {
      "de": "Untersuchungshaft",
      "en": "pre-trial detention"
    },
    {
      "de": "Dauer",
      "en": "duration"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "prosecutor",
    "compulsory measures court"
  ],
  "sanctions_or_consequences": [],
  "provision_role_llm": "procedure",
  "specificity_score": 0.8
}
```

#### `Art. 212 Abs. 3 StPO`

```json
{
  "english_summary": "Detention for investigation and security purposes cannot exceed the expected prison sentence.",
  "legal_rule": "Investigation and security detention duration limited to expected prison sentence",
  "applicability_conditions": [
    "Detention for investigation",
    "Detention for security reasons",
    "Expected prison sentence duration"
  ],
  "exceptions_or_limitations": [],
  "legal_question": "May investigation or security detention exceed the expected prison sentence?",
  "concepts_en": [
    "criminal procedure",
    "detention",
    "investigation",
    "security measures",
    "prison sentence",
    "duration limitation"
  ],
  "terms_de_to_en": [
    {
      "de": "Untersuchungs- und Sicherheitshaft",
      "en": "Investigation and security detention"
    },
    {
      "de": "Freiheitsstrafe",
      "en": "Prison sentence"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "court",
    "competent cantonal authority"
  ],
  "sanctions_or_consequences": [
    "Excess duration may be unlawful"
  ],
  "provision_role_llm": "other",
  "specificity_score": 0.8
}
```

#### `Art. 390 Abs. 2 StPO`

```json
{
  "english_summary": "If the legal remedy is not obviously invalid or unfounded, the proceedings officer must provide the opposing parties and the prior instance with the legal remedy document for comment. If the document cannot be delivered or no response is received, the proceedings continue.",
  "legal_rule": "Proceedings officer must provide legal remedy document for comment unless invalid/unfounded or delivery/response fails",
  "applicability_conditions": [
    "Legal remedy not obviously invalid",
    "Legal remedy not obviously unfounded",
    "Proceedings officer required to act",
    "Opposing parties and prior instance involved"
  ],
  "exceptions_or_limitations": [
    "Delivery failure",
    "No response received"
  ],
  "legal_question": "When must proceedings officer provide legal remedy document for comment?",
  "concepts_en": [
    "Legal remedy",
    "Proceedings officer",
    "Comment provision",
    "Delivery failure",
    "Prior instance",
    "Opposing parties",
    "Proceedings continuation"
  ],
  "terms_de_to_en": [
    {
      "de": "Rechtsmittel",
      "en": "legal remedy"
    },
    {
      "de": "Verfahrensleitung",
      "en": "proceedings officer"
    },
    {
      "de": "Stellungnahme",
      "en": "comment"
    },
    {
      "de": "Vorinstanz",
      "en": "prior instance"
    },
    {
      "de": "Zugestellung",
      "en": "delivery"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "proceedings officer",
    "opposing parties",
    "prior instance"
  ],
  "sanctions_or_consequences": [
    "Proceedings continue despite lack of comment"
  ],
  "provision_role_llm": "procedure",
  "specificity_score": 0.8
}
```

#### `Art. 422 Abs. 1 StPO`

```json
{
  "english_summary": "This article defines procedural costs as consisting of fees covering expenses and specific costs in a criminal case.",
  "legal_rule": "Procedural costs include fees for expenses and specific costs in a criminal case",
  "applicability_conditions": [
    "In criminal proceedings",
    "Regarding costs incurred",
    "For specific cases"
  ],
  "exceptions_or_limitations": [],
  "legal_question": "What constitutes procedural costs in criminal proceedings?",
  "concepts_en": [
    "criminal proceedings",
    "procedural costs",
    "fees",
    "expenses",
    "costs",
    "legal expenses"
  ],
  "terms_de_to_en": [
    {
      "de": "Verfahrenskosten",
      "en": "procedural costs"
    },
    {
      "de": "Gebühren",
      "en": "fees"
    },
    {
      "de": "Aufwand",
      "en": "expenses"
    },
    {
      "de": "Auslagen",
      "en": "costs"
    },
    {
      "de": "Straffall",
      "en": "criminal case"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "court",
    "public prosecutor"
  ],
  "sanctions_or_consequences": [],
  "provision_role_llm": "definition",
  "specificity_score": 0.8
}
```

#### `Art. 422 Abs. 2 StPO`

```json
{
  "english_summary": "This article lists specific costs that are considered official expenses in criminal proceedings.",
  "legal_rule": "",
  "applicability_conditions": [
    "Costs related to official defense",
    "Translation costs",
    "Expert opinion costs",
    "Collaboration with other authorities",
    "Postal, telephone, and similar expenses"
  ],
  "exceptions_or_limitations": [],
  "legal_question": "",
  "concepts_en": [
    "criminal proceedings",
    "official expenses",
    "costs",
    "defense",
    "translation",
    "expert opinion",
    "authority collaboration",
    "postal expenses"
  ],
  "terms_de_to_en": [
    {
      "de": "Auslagen",
      "en": "official expenses"
    },
    {
      "de": "amtliche Verteidigung",
      "en": "official defense"
    },
    {
      "de": "Verbeiständung",
      "en": "attendance"
    },
    {
      "de": "Übersetzungen",
      "en": "translations"
    },
    {
      "de": "Gutachten",
      "en": "expert opinions"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "court",
    "public prosecutor",
    "competent cantonal authority"
  ],
  "sanctions_or_consequences": [],
  "provision_role_llm": "definition",
  "specificity_score": 0.8
}
```

#### `Art. 428 Abs. 1 StPO`

```json
{
  "english_summary": "Parties bear costs of legal proceedings proportionally to their success or failure. A party that withdraws its claim or fails to have its remedy granted is deemed to have lost.",
  "legal_rule": "Costs of legal proceedings borne by parties according to their success or failure",
  "applicability_conditions": [
    "Legal proceedings under criminal procedure",
    "Parties involved in legal remedies",
    "Claim withdrawal or non-fulfillment of remedy"
  ],
  "exceptions_or_limitations": [],
  "legal_question": "Who bears the costs of legal proceedings and under what conditions?",
  "concepts_en": [
    "cost allocation",
    "legal proceedings",
    "parties",
    "success",
    "failure",
    "claim withdrawal",
    "remedy fulfillment"
  ],
  "terms_de_to_en": [
    {
      "de": "Rechtsmittelverfahren",
      "en": "legal remedy proceedings"
    },
    {
      "de": "Obsiegens oder Unterliegens",
      "en": "success or failure"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "natural person",
    "legal entity",
    "court"
  ],
  "sanctions_or_consequences": [
    "Costs allocated based on success/failure",
    "Withdrawn claims deemed as losing"
  ],
  "provision_role_llm": "other",
  "specificity_score": 0.8
}
```

#### `Art. 135 Abs. 4 StPO`

```json
{
  "english_summary": "If convicted of procedural costs, the accused must repay compensation to the Confederation or canton once their financial situation allows.",
  "legal_rule": "Accused must repay compensation to state upon financial ability",
  "applicability_conditions": [
    "Conviction for procedural costs",
    "Accused's financial situation permits repayment"
  ],
  "exceptions_or_limitations": [],
  "legal_question": "When must the accused repay compensation?",
  "concepts_en": [
    "criminal procedure",
    "compensation",
    "financial ability",
    "state liability",
    "conviction",
    "reparation"
  ],
  "terms_de_to_en": [
    {
      "de": "Verfahrenskosten",
      "en": "procedural costs"
    },
    {
      "de": "Entschädigung",
      "en": "compensation"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "natural person",
    "Confederation",
    "cantonal authority"
  ],
  "sanctions_or_consequences": [
    "Repayment obligation",
    "Financial liability"
  ],
  "provision_role_llm": "duty",
  "specificity_score": 0.8
}
```

#### `Art. 100 Abs. 1 BGG`

```json
{
  "english_summary": "This article sets the time limit for filing an appeal against a decision with the Federal Court.",
  "legal_rule": "Appeal must be filed within 30 days after full copy availability at the Federal Court",
  "applicability_conditions": [
    "Appeal against a decision",
    "Full copy of the decision available",
    "Filing with the Federal Court"
  ],
  "exceptions_or_limitations": [],
  "legal_question": "When must an appeal be filed with the Federal Court?",
  "concepts_en": [
    "Appeal",
    "Time limit",
    "Federal Court",
    "Decision",
    "Full copy availability",
    "Administrative procedure"
  ],
  "terms_de_to_en": [
    {
      "de": "Beschwerde",
      "en": "appeal"
    },
    {
      "de": "Entscheid",
      "en": "decision"
    },
    {
      "de": "vollständige Ausfertigung",
      "en": "full copy"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "natural person",
    "legal entity / undertaking"
  ],
  "sanctions_or_consequences": [],
  "provision_role_llm": "other",
  "specificity_score": 0.8
}
```

#### `Art. 135 Abs. 3 StPO`

```json
{
  "english_summary": "The official defense may appeal against a compensation decision using the legal remedy permissible against a final decision.",
  "legal_rule": "Official defense may appeal compensation decisions using final decision remedies",
  "applicability_conditions": [
    "Compensation decision exists",
    "Official defense is entitled to appeal",
    "Appeal must follow final decision remedies"
  ],
  "exceptions_or_limitations": [],
  "legal_question": "May official defense appeal compensation decisions?",
  "concepts_en": [
    "compensation decision",
    "legal remedy",
    "final decision",
    "appeal",
    "official defense",
    "criminal procedure"
  ],
  "terms_de_to_en": [
    {
      "de": "Entschädigungsentscheid",
      "en": "compensation decision"
    },
    {
      "de": "amtliche Verteidigung",
      "en": "official defense"
    },
    {
      "de": "Rechtsmittel",
      "en": "legal remedy"
    },
    {
      "de": "Endentscheid",
      "en": "final decision"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "official defense"
  ],
  "sanctions_or_consequences": [],
  "provision_role_llm": "right_or_entitlement",
  "specificity_score": 0.8
}
```

#### `Art. 37 Abs. 1 StBOG`

```json
{
  "english_summary": "The federal criminal courts' appellate chambers decide appeals referred to them by the StPO as competent.",
  "legal_rule": "Federal criminal courts' appellate chambers decide appeals referred to them by the StPO.",
  "applicability_conditions": [
    "Appeals must be referred to the chambers by the StPO",
    "Decisions must be within the scope of the StPO's referral"
  ],
  "exceptions_or_limitations": [],
  "legal_question": "Which appeals are decided by the federal criminal courts' appellate chambers?",
  "concepts_en": [
    "Administrative law",
    "Judicial review",
    "Competence",
    "Appeal",
    "Federal criminal courts",
    "StPO referrals"
  ],
  "terms_de_to_en": [
    {
      "de": "Beschwerdekammern",
      "en": "appellate chambers"
    },
    {
      "de": "Bundestrafgericht",
      "en": "federal criminal court"
    },
    {
      "de": "StPO",
      "en": "Swiss Code of Criminal Procedure"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "federal criminal courts",
    "StPO"
  ],
  "sanctions_or_consequences": [],
  "provision_role_llm": "competence",
  "specificity_score": 0.8
}
```

#### `Art. 39 Abs. 1 StBOG`

```json
{
  "english_summary": "The proceedings before the chambers of the Federal Criminal Court are governed by the StPO and this law.",
  "legal_rule": "Proceedings before federal criminal court chambers follow StPO and this law",
  "applicability_conditions": [
    "Federal criminal court chambers involved",
    "Cases under this law"
  ],
  "exceptions_or_limitations": [],
  "legal_question": "",
  "concepts_en": [
    "criminal procedure",
    "judicial proceedings",
    "federal court",
    "statutory law",
    "court chambers"
  ],
  "terms_de_to_en": [
    {
      "de": "Kammern des Bundesstrafgerichts",
      "en": "chambers of the Federal Criminal Court"
    },
    {
      "de": "StPO",
      "en": "Federal Criminal Procedure Code"
    }
  ],
  "defined_terms": [],
  "addressees": [
    "federal court",
    "court chambers"
  ],
  "sanctions_or_consequences": [],
  "provision_role_llm": "scope",
  "specificity_score": 0.8
}
```

### 4.2 Court gold (13/23 present)

#### `BGE 137 IV 122 E. 6.2`

**`text` (passage from this consideration):** Nach Art. 237 Abs. 1 StPO ordnet das zuständige Gericht an Stelle der Untersuchungs- oder der Sicherheitshaft eine oder mehrere mildere Massnahmen an, wenn sie den gleichen Zweck wie die Haft erfüllen. Als Ersatzmassnahmen kommen gemäss Art. 237 Abs. 2 StPO namentlich in Frage: a. die Sicherheitsleistung; b. die Ausweis- und Schriftensperre; c. die Auflage, sich nur oder sich nicht an einem bestimmten Ort oder in einem bestimmten Haus aufzuhalten; d. die Auflage, sich regelmässig bei einer Amtss...

```json
{
  "legal_area": "Criminal Procedure",
  "primary_domain": "Detention Alternatives",
  "secondary_domain": "Pre-trial Measures",
  "legal_domain_path": [
    "Swiss Law",
    "Criminal Procedure",
    "Detention Alternatives"
  ],
  "topic": "Alternative Detention Measures",
  "subtopic": "Substitution of Pre-trial Detention",
  "micro_topic": "Specific alternative measures to pre-trial detention under Swiss Criminal Procedure Code",
  "concepts_en": [
    "Alternative Detention",
    "Pre-trial Measures",
    "Safeguards",
    "Legal Compliance",
    "Judicial Discretion"
  ],
  "terms_original": [
    "Sicherheitsleistung",
    "Ausweis- und Schriftensperre",
    "Auflage",
    "Meldepflicht",
    "Kontaktverbot"
  ],
  "doctrinal_rule": "",
  "legal_test": "",
  "fact_pattern_tags": [
    "Pre-trial",
    "Detention",
    "Substitution",
    "Legal Provisions"
  ],
  "procedural_context": "Substitution of Detention",
  "paragraph_role": "legal_standard",
  "authority_role": [
    "Swiss Criminal Procedure Code"
  ],
  "specificity_score": 0.9
}
```

#### `BGE 137 IV 122 E. 6.4`

**`text` (passage from this consideration):** Eine Eingrenzung auf ein bestimmtes Gebiet kommt, wie dargelegt (vgl. E. 6.2 hiervor), primär bei Fluchtgefahr in Betracht. Geht es demgegenüber darum, einer Kollusionsgefahr in Form der möglichen Beeinflussung des mutmasslichen Opfers zu begegnen, dürfte in aller Regel eine Ausgrenzung als mildere Massnahme genügen. Dies ist auch vorliegend der Fall. Die Verpflichtung, den BGE 137 IV 122 S. 133 Kanton Bern nicht zu verlassen, schränkt den Beschwerdeführer in seiner persönlichen Freiheit stark e...

```json
{
  "legal_area": "Administrative law",
  "primary_domain": "Restrictions",
  "secondary_domain": "Freedom of movement",
  "legal_domain_path": [
    "Administrative law",
    "Restrictions",
    "Freedom of movement"
  ],
  "topic": "Restriction proportionality",
  "subtopic": "Proportionality of territorial restrictions",
  "micro_topic": "Proportionality of territorial restrictions in administrative law",
  "concepts_en": [
    "Proportionality",
    "Freedom of movement",
    "Administrative restriction",
    "Territorial limitation",
    "Collusion risk"
  ],
  "terms_original": [
    "Fluchtgefahr",
    "Kollusionsgefahr",
    "Ausgrenzung",
    "Meldepflicht",
    "Besuchszeiten"
  ],
  "doctrinal_rule": "",
  "legal_test": "",
  "fact_pattern_tags": [
    "Administrative restriction",
    "Proportionality",
    "Freedom of movement",
    "Collusion risk"
  ],
  "procedural_context": "Administrative decision review",
  "paragraph_role": "reasoning",
  "authority_role": [
    "Swiss Federal Court",
    "Administrative law"
  ],
  "specificity_score": 0.8
}
```

#### `BGE 137 IV 122 E. 4.2`

**`text` (passage from this consideration):** Gemäss Art. 221 Abs. 1 lit. b i.V.m. Art. 237 Abs. 1 StPO ist Untersuchungshaft respektive die Anordnung von Ersatzmassnahmen zulässig, wenn ernsthaft zu befürchten ist, die beschuldigte Person könnte Personen beeinflussen oder auf Beweismittel einwirken, um so die Wahrheitsfindung zu beeinträchtigen. Die strafprozessuale Haft wegen Kollusionsgefahr soll verhindern, dass die beschuldigte Person die Freiheit dazu missbrauchen würde, die wahrheitsgetreue Abklärung des Sachverhalts zu vereiteln ode...

```json
{
  "legal_area": "Criminal procedure",
  "primary_domain": "Detention",
  "secondary_domain": "Preventive detention",
  "legal_domain_path": [
    "Criminal law",
    "Procedure",
    "Detention"
  ],
  "topic": "Preventive detention",
  "subtopic": "Grounds for detention",
  "micro_topic": "Collusion risk as basis for preventive detention",
  "concepts_en": [
    "Preventive detention",
    "Collusion risk",
    "Interference with evidence",
    "Criminal procedure",
    "Detention order"
  ],
  "terms_original": [
    "Untersuchungshaft",
    "Ersatzmassnahmen",
    "Kollusionsgefahr",
    "Beweismittel",
    "Strafverfahren"
  ],
  "doctrinal_rule": "",
  "legal_test": "",
  "fact_pattern_tags": [
    "Detention",
    "Collusion",
    "Evidence",
    "Criminal procedure"
  ],
  "procedural_context": "Criminal procedure",
  "paragraph_role": "legal_standard",
  "authority_role": [
    "Swiss Federal Court"
  ],
  "specificity_score": 0.9
}
```

#### `BGE 132 I 21 E. 3.2`

**`text` (passage from this consideration):** Kollusion bedeutet nach der bundesgerichtlichen Praxis insbesondere, dass sich der Angeschuldigte mit Zeugen, Auskunftspersonen, Sachverständigen oder Mitangeschuldigten ins Einvernehmen setzt oder sie zu wahrheitswidrigen Aussagen veranlasst, oder dass er Spuren und Beweismittel beseitigt. Die strafprozessuale Haft wegen Kollusionsgefahr soll verhindern, dass der Angeschuldigte die Freiheit oder einen Urlaub dazu missbrauchen würde, die wahrheitsgetreue Abklärung des Sachverhaltes zu vereiteln ...

```json
{
  "legal_area": "Criminal procedure",
  "primary_domain": "Detention",
  "secondary_domain": "Collusion and evidence",
  "legal_domain_path": [
    "Criminal law",
    "Detention",
    "Collusion"
  ],
  "topic": "Collusion in criminal proceedings",
  "subtopic": "Conditions for detention due to collusion risk",
  "micro_topic": "When is detention justified to prevent collusion",
  "concepts_en": [
    "Collusion",
    "Detention",
    "Evidence",
    "Due process",
    "Suspicion"
  ],
  "terms_original": [
    "Kollusion",
    "Haft",
    "Beweismittel",
    "Verdunkelungsgefahr",
    "Urlaub"
  ],
  "doctrinal_rule": "",
  "legal_test": "",
  "fact_pattern_tags": [
    "Collusion",
    "Detention",
    "Evidence",
    "Risk"
  ],
  "procedural_context": "Detention justification",
  "paragraph_role": "reasoning",
  "authority_role": [
    "Swiss Federal Court"
  ],
  "specificity_score": 0.8
}
```

#### `BGE 132 I 21 E. 3.2.2`

**`text` (passage from this consideration):** Nach Abschluss der Strafuntersuchung (und insbesondere nach Durchführung einer erstinstanzlichen Hauptverhandlung) bedarf der Haftgrund der Kollusionsgefahr einer besonders sorgfältigen Prüfung. Er dient primär der Sicherung einer ungestörten Strafuntersuchung. Zwar ist auch die richterliche Sachaufklärung vor unzulässigen Einflussnahmen zu bewahren. Dies gilt insbesondere im Hinblick auf die (in der Regel beschränkte) Unmittelbarkeit der Beweisaufnahme anlässlich der Hauptverhandlung (vgl. BGE ...

```json
{
  "legal_area": "criminal procedure",
  "primary_domain": "preliminary investigation",
  "secondary_domain": "custody and detention",
  "legal_domain_path": [
    "criminal law",
    "procedural law",
    "preliminary investigation"
  ],
  "topic": "grounds for detention",
  "subtopic": "collusion danger",
  "micro_topic": "collusion danger as basis for detention",
  "concepts_en": [
    "custody",
    "collusion",
    "preliminary investigation",
    "evidence gathering"
  ],
  "terms_original": [
    "Haftgrund",
    "Kollusionsgefahr",
    "Strafuntersuchung",
    "Unmittelbarkeit",
    "Beweisaufnahme"
  ],
  "doctrinal_rule": "",
  "legal_test": "",
  "fact_pattern_tags": [
    "criminal_procedure",
    "preliminary_investigation",
    "custody",
    "collusion"
  ],
  "procedural_context": "criminal procedure",
  "paragraph_role": "reasoning",
  "authority_role": [
    "Swiss Federal Court",
    "legal commentary"
  ],
  "specificity_score": 0.8
}
```

#### `BGE 139 IV 270 E. 3.1`

**`text` (passage from this consideration):** En vertu des art. 31 al. 3 Cst. et 5 par. 3 CEDH, toute personne qui est mise en détention préventive a le droit d'être jugée dans un délai raisonnable ou d'être libérée pendant la procédure pénale. Une durée excessive de la détention constitue une limitation disproportionnée de ce droit fondamental, qui est notamment violé lorsque la durée de la détention préventive dépasse la durée probable de la peine privative de liberté à laquelle il faut s'attendre. L'art. 212 al. 3 CPP prévoit ainsi que l...

```json
{
  "legal_area": "criminal law",
  "primary_domain": "preliminary detention",
  "secondary_domain": "duration and limits",
  "legal_domain_path": [
    "criminal law",
    "preliminary detention",
    "duration limits"
  ],
  "topic": "detention duration",
  "subtopic": "reasonable time requirement",
  "micro_topic": "maximum duration of preventive detention",
  "concepts_en": [
    "reasonable time",
    "preventive detention",
    "proportionality",
    "criminal procedure",
    "sentence duration"
  ],
  "terms_original": [
    "détention préventive",
    "durée raisonnable",
    "peine privative de liberté",
    "CPP",
    "CEDH"
  ],
  "doctrinal_rule": "",
  "legal_test": "",
  "fact_pattern_tags": [
    "detention",
    "duration",
    "proportionality",
    "criminal"
  ],
  "procedural_context": "criminal procedure",
  "paragraph_role": "reasoning",
  "authority_role": [
    "CEDH",
    "CPP"
  ],
  "specificity_score": 0.8
}
```

#### `BGE 133 I 168 E. 4.1`

**`text` (passage from this consideration):** En vertu des art. 31 al. 3 Cst. et 5 par. 3 CEDH, toute personne qui est mise en détention préventive a le droit d'être jugée dans un délai raisonnable ou d'être libérée pendant la procédure pénale. Une durée excessive de la détention constitue une limitation disproportionnée de ce droit fondamental, qui est notamment violé lorsque la durée de la détention préventive dépasse la durée probable de la peine privative de liberté à laquelle il faut s'attendre. Dans l'examen de la proportionnalité de ...

```json
{
  "legal_area": "criminal law",
  "primary_domain": "pretrial detention",
  "secondary_domain": "proportionality of detention",
  "legal_domain_path": [
    "constitutional law",
    "criminal procedure",
    "detention rights"
  ],
  "topic": "detention duration",
  "subtopic": "proportionality standard",
  "micro_topic": "duration of pretrial detention and proportionality",
  "concepts_en": [
    "proportionality",
    "pretrial detention",
    "fundamental rights",
    "reasonable time",
    "criminal procedure"
  ],
  "terms_original": [
    "Art. 31 al. 3 Cst.",
    "Art. 5 par. 3 CEDH",
    "détention préventive",
    "peine privative de liberté",
    "proportionnalité"
  ],
  "doctrinal_rule": "",
  "legal_test": "Duration of detention must not exceed probable sentence duration",
  "fact_pattern_tags": [
    "pretrial detention",
    "proportionality",
    "constitutional rights",
    "criminal procedure"
  ],
  "procedural_context": "constitutional review",
  "paragraph_role": "reasoning",
  "authority_role": [
    "Swiss Federal Court",
    "European Court of Human Rights"
  ],
  "specificity_score": 0.8
}
```

#### `BGE 143 IV 168 E. 5.1`

**`text` (passage from this consideration):** En vertu des art. 31 al. 3 Cst. et 5 par. 3 CEDH, toute personne qui est mise en détention préventive a le droit d'être jugée dans un délai raisonnable ou d'être libérée pendant la procédure pénale. L'art. 212 al. 3 CPP rappelle cette exigence en précisant que la détention provisoire ou pour des motifs de sûreté ne doit pas durer plus longtemps que la peine privative de liberté prévisible. Le juge peut dès lors maintenir la détention préventive aussi longtemps qu'elle n'est pas très proche de la...

```json
{
  "legal_area": "criminal law",
  "primary_domain": "detention",
  "secondary_domain": "preliminary detention",
  "legal_domain_path": [
    "Swiss law",
    "criminal procedure",
    "detention"
  ],
  "topic": "detention duration",
  "subtopic": "reasonable time for trial",
  "micro_topic": "duration of preliminary detention based on expected sentence",
  "concepts_en": [
    "reasonable time",
    "preliminary detention",
    "expected sentence",
    "judicial discretion",
    "criminal procedure"
  ],
  "terms_original": [
    "détention préventive",
    "peine privative de liberté",
    "sursis",
    "libération conditionnelle",
    "CPP"
  ],
  "doctrinal_rule": "",
  "legal_test": "",
  "fact_pattern_tags": [
    "detention",
    "expected sentence",
    "judicial discretion",
    "criminal procedure"
  ],
  "procedural_context": "criminal procedure",
  "paragraph_role": "reasoning",
  "authority_role": [
    "Swiss law",
    "case law"
  ],
  "specificity_score": 0.8
}
```

#### `BGE 133 I 270 E. 3.4.2`

**`text` (passage from this consideration):** Gemäss Art. 31 Abs. 3 BV und Art. 5 Ziff. 3 EMRK hat eine in strafprozessualer Haft gehaltene Person Anspruch darauf, innerhalb einer angemessenen Frist richterlich beurteilt oder während des Strafverfahrens aus der Haft entlassen zu werden. Eine übermässige Haftdauer stellt eine unverhältnismässige Beschränkung dieses Grundrechts dar. Sie liegt dann vor, wenn die Haftfrist die mutmassliche Dauer der zu erwartenden freiheitsentziehenden Sanktion übersteigt. Bei der Prüfung der Verhältnismässigke...

```json
{
  "legal_area": "Human rights",
  "primary_domain": "Detention",
  "secondary_domain": "Proportionality of detention",
  "legal_domain_path": [
    "Constitutional law",
    "Human rights",
    "Detention"
  ],
  "topic": "Detention duration",
  "subtopic": "Proportionality of detention period",
  "micro_topic": "When detention exceeds expected sanction duration",
  "concepts_en": [
    "Detention",
    "Proportionality",
    "Fundamental rights",
    "Criminal procedure",
    "Judicial review"
  ],
  "terms_original": [
    "Art. 31 Abs. 3 BV",
    "Art. 5 Ziff. 3 EMRK",
    "Haftdauer",
    "unverhältnismässige Beschränkung",
    "freiheitsentziehende Sanktion"
  ],
  "doctrinal_rule": "",
  "legal_test": "",
  "fact_pattern_tags": [
    "Detention",
    "Proportionality",
    "Fundamental rights",
    "Criminal procedure"
  ],
  "procedural_context": "Judicial review of detention",
  "paragraph_role": "reasoning",
  "authority_role": [
    "Bundesgericht",
    "European Court of Human Rights"
  ],
  "specificity_score": 0.8
}
```

#### `BGE 137 IV 122 E. 4.1`

**`text` (passage from this consideration):** Der Beschwerdeführer bestreitet das Vorliegen von Kollusionsgefahr. Zwar sei die richterliche Sachaufklärung im Hinblick auf die beschränkte Unmittelbarkeit der Beweisaufnahme anlässlich der Hauptverhandlung vor unzulässigen Einflussnahmen zu bewahren. Die blosse Wahrscheinlichkeit einer Einvernahme des mutmasslichen Opfers durch das Gericht genüge jedoch zur Begründung von Kollusionsgefahr nicht.

```json
{
  "legal_area": "Civil procedure",
  "primary_domain": "Evidence",
  "secondary_domain": "Collusion danger",
  "legal_domain_path": [
    "Procedural law",
    "Evidence",
    "Collusion"
  ],
  "topic": "Collusion danger",
  "subtopic": "Burden of proof",
  "micro_topic": "Sufficient proof for collusion danger",
  "concepts_en": [
    "Burden of proof",
    "Collusion",
    "Evidence",
    "Judicial review",
    "Proof"
  ],
  "terms_original": [
    "Kollusionsgefahr",
    "Beweisaufnahme",
    "Unzulässige Einflussnahmen",
    "Beweis",
    "Richterliche Sachaufklärung"
  ],
  "doctrinal_rule": "",
  "legal_test": "",
  "fact_pattern_tags": [
    "Collusion",
    "Evidence",
    "Judicial review",
    "Burden of proof"
  ],
  "procedural_context": "Civil procedure",
  "paragraph_role": "reasoning",
  "authority_role": [
    "Swiss Federal Court"
  ],
  "specificity_score": 0.8
}
```

#### `BGE 132 I 21 E. 3.2.1`

**`text` (passage from this consideration):** Konkrete Anhaltspunkte für Kollusionsgefahr können sich nach der Rechtsprechung des Bundesgerichtes namentlich ergeben aus dem bisherigen Verhalten des Angeschuldigten im Strafprozess, aus seinen persönlichen Merkmalen, aus seiner Stellung und seinen Tatbeiträgen im Rahmen des untersuchten Sachverhaltes sowie aus den persönlichen Beziehungen zwischen ihm und den ihn belastenden Personen. Bei der Frage, ob im konkreten Fall eine massgebliche Beeinträchtigung des Strafverfahrens wegen Verdunkelung...

```json
{
  "legal_area": "criminal procedure",
  "primary_domain": "preliminary detention",
  "secondary_domain": "collusion danger",
  "legal_domain_path": [
    "criminal law",
    "procedure",
    "detention"
  ],
  "topic": "collusion danger",
  "subtopic": "evidence of collusion risk",
  "micro_topic": "factors indicating collusion risk in criminal proceedings",
  "concepts_en": [
    "collusion",
    "preliminary detention",
    "evidence",
    "risk assessment",
    "criminal procedure"
  ],
  "terms_original": [
    "Kollusionsgefahr",
    "Strafverfahren",
    "Beweismittel",
    "Verdunkelung",
    "Haftrichter"
  ],
  "doctrinal_rule": "",
  "legal_test": "",
  "fact_pattern_tags": [
    "collusion risk",
    "preliminary detention",
    "evidence",
    "criminal procedure"
  ],
  "procedural_context": "criminal proceedings",
  "paragraph_role": "reasoning",
  "authority_role": [
    "BGE",
    "BJM"
  ],
  "specificity_score": 0.8
}
```

#### `1B_357/2022 E. 3.1`

**`text` (passage from this consideration):** 3.1. Der besondere Haftgrund der Kollusionsgefahr liegt vor, wenn ernsthaft zu befürchten ist, dass der oder die Beschuldigte Personen beeinflusst oder auf Beweismittel einwirkt, um so die Wahrheitsfindung zu beeinträchtigen (Art. 221 Abs. 1 lit. b StPO). Verdunkelung kann nach der bundesgerichtlichen Praxis insbesondere in der Weise erfolgen, dass sich die beschuldigte Person mit Zeugen, Auskunftspersonen, Sachverständigen oder Mitbeschuldigten ins Einvernehmen setzt oder sie zu wahrheitswidrig...

```json
{
  "legal_area": "Criminal law",
  "primary_domain": "Detention law",
  "secondary_domain": "Reasons for detention",
  "legal_domain_path": [
    "Criminal procedure",
    "Detention",
    "Grounds for detention"
  ],
  "topic": "Grounds for detention",
  "subtopic": "Collusion danger",
  "micro_topic": "Collusion danger as a basis for detention",
  "concepts_en": [
    "Detention",
    "Collusion",
    "Evidence",
    "Interference",
    "Criminal procedure"
  ],
  "terms_original": [
    "Kollusionsgefahr",
    "Verdunkelung",
    "Untersuchungshaft",
    "StPO",
    "BGE"
  ],
  "doctrinal_rule": "",
  "legal_test": "",
  "fact_pattern_tags": [
    "Collusion",
    "Detention",
    "Evidence",
    "Interference"
  ],
  "procedural_context": "Detention justification",
  "paragraph_role": "legal_standard",
  "authority_role": [
    "Swiss Federal Court",
    "Case law"
  ],
  "specificity_score": 0.9
}
```

#### `1B_28/2022 E. 4.1`

**`text` (passage from this consideration):** 4.1. Der Haftgrund der Kollusionsgefahr liegt vor, wenn ernsthaft zu befürchten ist, dass die oder der Beschuldigte Personen beeinflusst oder auf Beweismittel einwirkt, um so die Wahrheitsfindung zu beeinträchtigen (Art. 221 Abs. 1 lit. b StPO). Verdunkelung kann nach der bundesgerichtlichen Praxis zu Art. 221 Abs. 1 lit. b StPO insbesondere in der Weise erfolgen, dass sich die beschuldigte Person mit Zeuginnen oder Zeugen, Auskunftspersonen, Sachverständigen oder Mitbeschuldigten ins Einvernehm...

```json
{
  "legal_area": "Criminal law",
  "primary_domain": "Detention law",
  "secondary_domain": "Collusion risk",
  "legal_domain_path": [
    "Criminal procedure",
    "Detention",
    "Collusion risk"
  ],
  "topic": "Detention grounds",
  "subtopic": "Collusion risk justification",
  "micro_topic": "Conditions for detention due to collusion risk",
  "concepts_en": [
    "Detention",
    "Collusion",
    "Evidence suppression",
    "Criminal procedure",
    "Legal standard"
  ],
  "terms_original": [
    "Kollusionsgefahr",
    "Verdunkelung",
    "StPO",
    "BGE",
    "1B_575/2021"
  ],
  "doctrinal_rule": "",
  "legal_test": "",
  "fact_pattern_tags": [
    "Detention",
    "Collusion",
    "Evidence",
    "Legal standard"
  ],
  "procedural_context": "Detention justification",
  "paragraph_role": "legal_standard",
  "authority_role": [
    "Swiss law",
    "Case law"
  ],
  "specificity_score": 0.8
}
```

### 4.3 Court gold with static-only enrichment (10/23, from v5_unified)

These are the 10 court gold citations that have **no LLM payload** in the 363k run but **do** have rule-based static enrichment in `court_authority_cards_v5_unified.jsonl` (`enrichment_source = "static"`). The schema is the same `rag_enrichment` block as the LLM rows but with only the deterministically-extractable fields populated.

#### `1B_210/2023 E. 4.1`

**`text_excerpt_original` (head):** 4.1. Conformément à l'art. 221 al. 1 let. b CPP, la détention provisoire ou pour motifs de sûreté ne peut être ordonnée que lorsque le prévenu est fortement soupçonné d'avoir commis un crime ou un délit et qu'il y a sérieusement lieu de craindre qu'il compromette la recherche de la vérité en exerçant une influence sur des personnes ou en altérant des moyens de preuve. Selon la jurisprudence, il pe

```json
{
  "enrichment_source": "static",
  "rag_enrichment": {
    "legal_area": "criminal procedure and coercive measures",
    "primary_domain": "",
    "secondary_domain": "",
    "topic": "",
    "subtopic": "",
    "micro_topic": "",
    "doctrinal_rule": "",
    "legal_test": "",
    "procedural_context": "",
    "legal_topic": "",
    "legal_rule": "",
    "court_holding": "",
    "factual_context": "",
    "legal_domain_path": [],
    "concepts_en": [
      "ECHR and fundamental rights",
      "collusion risk",
      "criminal procedure",
      "evidence assessment",
      "pretrial detention",
      "procedural rights"
    ],
    "terms_original": [
      "cedu",
      "danger de collusion",
      "risque de collusion",
      "moyens de preuve",
      "detention provisoire"
    ],
    "fact_pattern_tags": [],
    "authority_role": [
      "unpublished_federal_decision",
      "reasoning_consideration",
      "multi_consideration_decision"
    ],
    "english_legal_concepts": [],
    "search_keywords": [],
    "specificity_score": 0.0,
    "paragraph_role": "legal_standard",
    "outcome_signal": "none",
    "statute_anchors": [
      "art. 221 al. 1 let. b CPP",
      "Art. 221 Abs. 1"
    ],
    "case_anchors": [],
    "legal_source_anchors": [],
    "secondary_sources": [],
    "document_or_plan_anchors": [],
    "event_anchors": []
  }
}
```

#### `1B_536/2018 E. 5.1`

**`text_excerpt_original` (head):** 5.1. Conformément à cette disposition, la détention provisoire ne peut être ordonnée que lorsque le prévenu est fortement soupçonné d'avoir commis un crime ou un délit et qu'il y a sérieusement lieu de craindre qu'il compromette la recherche de la vérité en exerçant une influence sur des personnes ou en altérant des moyens de preuves. Pour retenir l'existence d'un risque de collusion au sens de ce

```json
{
  "enrichment_source": "static",
  "rag_enrichment": {
    "legal_area": "criminal procedure and coercive measures",
    "primary_domain": "",
    "secondary_domain": "",
    "topic": "",
    "subtopic": "",
    "micro_topic": "",
    "doctrinal_rule": "",
    "legal_test": "",
    "procedural_context": "",
    "legal_topic": "",
    "legal_rule": "",
    "court_holding": "",
    "factual_context": "",
    "legal_domain_path": [],
    "concepts_en": [
      "ECHR and fundamental rights",
      "collusion risk",
      "criminal procedure",
      "evidence assessment",
      "pretrial detention",
      "procedural rights"
    ],
    "terms_original": [
      "cedu",
      "risque de collusion",
      "moyens de preuve",
      "detention provisoire"
    ],
    "fact_pattern_tags": [],
    "authority_role": [
      "unpublished_federal_decision",
      "reasoning_consideration",
      "multi_consideration_decision"
    ],
    "english_legal_concepts": [],
    "search_keywords": [],
    "specificity_score": 0.0,
    "paragraph_role": "facts",
    "outcome_signal": "none",
    "statute_anchors": [
      "Art. 5 Abs. 2 CPP"
    ],
    "case_anchors": [
      "ATF 137 IV 122"
    ],
    "legal_source_anchors": [],
    "secondary_sources": [],
    "document_or_plan_anchors": [],
    "event_anchors": []
  }
}
```

#### `1B_90/2021 E. 2.1`

**`text_excerpt_original` (head):** 2.1. Der Haftgrund der Kollusionsgefahr liegt vor, wenn ernsthaft zu befürchten ist, dass der oder die Beschuldigte Personen beeinflusst oder auf Beweismittel einwirkt, um so die Wahrheitsfindung zu beeinträchtigen (Art. 221 Abs. 1 lit. b StPO). Verdunkelung kann nach der bundesgerichtlichen Praxis insbesondere in der Weise erfolgen, dass sich die beschuldigte Person mit Zeugen, Auskunftspersonen,

```json
{
  "enrichment_source": "static",
  "rag_enrichment": {
    "legal_area": "criminal procedure and coercive measures",
    "primary_domain": "",
    "secondary_domain": "",
    "topic": "",
    "subtopic": "",
    "micro_topic": "",
    "doctrinal_rule": "",
    "legal_test": "",
    "procedural_context": "",
    "legal_topic": "",
    "legal_rule": "",
    "court_holding": "",
    "factual_context": "",
    "legal_domain_path": [],
    "concepts_en": [
      "collusion risk",
      "criminal procedure",
      "evidence assessment",
      "pretrial detention",
      "procedural rights"
    ],
    "terms_original": [
      "kollusionsgefahr",
      "verdunkelungsgefahr",
      "sachverstandige",
      "zeuge",
      "untersuchungshaft"
    ],
    "fact_pattern_tags": [],
    "authority_role": [
      "unpublished_federal_decision",
      "reasoning_consideration",
      "multi_consideration_decision"
    ],
    "english_legal_concepts": [],
    "search_keywords": [],
    "specificity_score": 0.0,
    "paragraph_role": "facts",
    "outcome_signal": "none",
    "statute_anchors": [
      "Art. 221 Abs. 1 lit. b StPO",
      "Art. 221",
      "Art. 221 Abs. 1 Bst. b StPO"
    ],
    "case_anchors": [
      "BGE 137 IV 122 E. 4.2",
      "1B_406/2016 22.11.2016 E. 2.4-2.6",
      "BGE 128 I 149 E. 3.4",
      "BGE 132 I 21 E. 3.2.1",
      "BGE 132 I 21 E. 3.4"
    ],
    "legal_source_anchors": [],
    "secondary_sources": [],
    "document_or_plan_anchors": [],
    "event_anchors": []
  }
}
```

#### `1B_90/2021 E. 2.4`

**`text_excerpt_original` (head):** 2.4. Dass die kantonalen Instanzen den Haftgrund der Kollusionsgefahr im jetzigen Untersuchungsstadium bejahen, hält vor dem Bundesrecht stand. Als Anhaltspunkte für Verdunkelungsgefahr durften die kantonalen Strafbehörden insbesondere mitberücksichtigen, dass die kollusionsgefährdeten Beweisaussagen hier von hoher Bedeutung sind und ein grosses öffentliches Interesse an einer unbeeinflussten Unte

```json
{
  "enrichment_source": "static",
  "rag_enrichment": {
    "legal_area": "criminal procedure and coercive measures",
    "primary_domain": "",
    "secondary_domain": "",
    "topic": "",
    "subtopic": "",
    "micro_topic": "",
    "doctrinal_rule": "",
    "legal_test": "",
    "procedural_context": "",
    "legal_topic": "",
    "legal_rule": "",
    "court_holding": "",
    "factual_context": "",
    "legal_domain_path": [],
    "concepts_en": [
      "administrative procedure",
      "collusion risk"
    ],
    "terms_original": [
      "verfugung",
      "kollusionsgefahr",
      "verdunkelungsgefahr"
    ],
    "fact_pattern_tags": [],
    "authority_role": [
      "unpublished_federal_decision",
      "reasoning_consideration",
      "multi_consideration_decision"
    ],
    "english_legal_concepts": [],
    "search_keywords": [],
    "specificity_score": 0.0,
    "paragraph_role": "reasoning",
    "outcome_signal": "none",
    "statute_anchors": [],
    "case_anchors": [],
    "legal_source_anchors": [],
    "secondary_sources": [],
    "document_or_plan_anchors": [],
    "event_anchors": []
  }
}
```

#### `7B_496/2025 E. 3.2`

**`text_excerpt_original` (head):** 3.2. Die Vorinstanz sieht weiterhin Kollusionsmöglichkeiten für den Beschwerdeführer. Zwar sei die Strafuntersuchung bereits weit fortgeschritten. Dennoch seien die zwei Mobiltelefone des Beschwerdeführers noch immer versiegelt und das Entsiegelungsverfahren sei noch im Gang. Bei einem der zwei Telefone handle es sich um ein Krypto-Handy, also ein speziell gesichertes Mobiltelefon, das eine abhörs

```json
{
  "enrichment_source": "static",
  "rag_enrichment": {
    "legal_area": "criminal law and criminal procedure",
    "primary_domain": "",
    "secondary_domain": "",
    "topic": "",
    "subtopic": "",
    "micro_topic": "",
    "doctrinal_rule": "",
    "legal_test": "",
    "procedural_context": "",
    "legal_topic": "",
    "legal_rule": "",
    "court_holding": "",
    "factual_context": "",
    "legal_domain_path": [],
    "concepts_en": [
      "drug offences",
      "inheritance and testament",
      "police and public security",
      "pretrial detention"
    ],
    "terms_original": [
      "betaubungsmittel",
      "drogen",
      "will",
      "uberwachung",
      "untersuchungshaft"
    ],
    "fact_pattern_tags": [],
    "authority_role": [
      "unpublished_federal_decision",
      "reasoning_consideration",
      "multi_consideration_decision"
    ],
    "english_legal_concepts": [],
    "search_keywords": [],
    "specificity_score": 0.0,
    "paragraph_role": "procedural_history",
    "outcome_signal": "none",
    "statute_anchors": [],
    "case_anchors": [],
    "legal_source_anchors": [],
    "secondary_sources": [],
    "document_or_plan_anchors": [],
    "event_anchors": []
  }
}
```

#### `7B_231/2025 E. 4.1`

**`text_excerpt_original` (head):** 4.1. Strafprozessuale Haft wegen Kollusions- bzw. Verdunkelungsgefahr (Art. 221 Abs. 1 lit. b StPO) soll verhindern, dass die beschuldigte Person die wahrheitsgetreue Abklärung des Sachverhalts vereitelt oder gefährdet. Die theoretische Möglichkeit, dass der Beschuldigte kolludieren könnte, genügt indessen nicht, um Haft unter diesem Titel zu rechtfertigen. Es müssen vielmehr konkrete Indizien für

```json
{
  "enrichment_source": "static",
  "rag_enrichment": {
    "legal_area": "criminal law and criminal procedure",
    "primary_domain": "",
    "secondary_domain": "",
    "topic": "",
    "subtopic": "",
    "micro_topic": "",
    "doctrinal_rule": "",
    "legal_test": "",
    "procedural_context": "",
    "legal_topic": "",
    "legal_rule": "",
    "court_holding": "",
    "factual_context": "",
    "legal_domain_path": [],
    "concepts_en": [
      "collusion risk",
      "criminal procedure",
      "pretrial detention",
      "procedural rights"
    ],
    "terms_original": [
      "kollusionsgefahr",
      "verdunkelungsgefahr"
    ],
    "fact_pattern_tags": [],
    "authority_role": [
      "unpublished_federal_decision",
      "reasoning_consideration",
      "multi_consideration_decision"
    ],
    "english_legal_concepts": [],
    "search_keywords": [],
    "specificity_score": 0.0,
    "paragraph_role": "facts",
    "outcome_signal": "none",
    "statute_anchors": [
      "Art. 221 Abs. 1 lit. b StPO",
      "Art. 221 Abs. 1 Bst. b StPO"
    ],
    "case_anchors": [
      "7B_12/2025 22.01.2025 E. 2.2",
      "BGE 137 IV 122 E. 4.2"
    ],
    "legal_source_anchors": [],
    "secondary_sources": [],
    "document_or_plan_anchors": [],
    "event_anchors": []
  }
}
```

#### `7B_69/2024 E. 3.3.2`

**`text_excerpt_original` (head):** 3.3.2. Der besondere Haftgrund der Kollusionsgefahr liegt vor, wenn ernsthaft zu befürchten ist, dass die beschuldigte Person jemanden beeinflusst oder auf Beweismittel einwirkt, um so die Wahrheitsfindung zu beeinträchtigen (Art. 221 Abs. 1 lit. b StPO). Verdunkelung kann gemäss der Rechtsprechung insbesondere in der Weise erfolgen, dass sich die beschuldigte Person mit Zeugen, Auskunftspersonen,

```json
{
  "enrichment_source": "static",
  "rag_enrichment": {
    "legal_area": "criminal law and criminal procedure",
    "primary_domain": "",
    "secondary_domain": "",
    "topic": "",
    "subtopic": "",
    "micro_topic": "",
    "doctrinal_rule": "",
    "legal_test": "",
    "procedural_context": "",
    "legal_topic": "",
    "legal_rule": "",
    "court_holding": "",
    "factual_context": "",
    "legal_domain_path": [],
    "concepts_en": [
      "collusion risk",
      "criminal procedure",
      "evidence assessment",
      "pretrial detention",
      "procedural rights"
    ],
    "terms_original": [
      "kollusionsgefahr",
      "verdunkelungsgefahr",
      "sachverstandige",
      "zeuge",
      "untersuchungshaft"
    ],
    "fact_pattern_tags": [],
    "authority_role": [
      "unpublished_federal_decision",
      "reasoning_consideration",
      "multi_consideration_decision"
    ],
    "english_legal_concepts": [],
    "search_keywords": [],
    "specificity_score": 0.0,
    "paragraph_role": "facts",
    "outcome_signal": "none",
    "statute_anchors": [
      "Art. 221 Abs. 1 lit. b StPO",
      "Art. 113 StPO",
      "Art. 221 Abs. 1 Bst. b StPO"
    ],
    "case_anchors": [
      "BGE 137 IV 122 E. 4.2",
      "7B_1028/2023",
      "7B_417/2023",
      "7B_1028/2023 12.01.2024 E. 8.1",
      "7B_474/2023 06.09.2023 E. 4.2.2",
      "BGE 132 I 21 E. 3.2.2"
    ],
    "legal_source_anchors": [],
    "secondary_sources": [],
    "document_or_plan_anchors": [],
    "event_anchors": [
      "12. Januar 2024",
      "4. September 2023"
    ]
  }
}
```

#### `7B_301/2024 E. 2.4`

**`text_excerpt_original` (head):** 2.4. Kollusionsgefahr liegt vor, wenn ernsthaft zu befürchten ist, dass die beschuldigte Person jemanden beeinflusst oder auf Beweismittel einwirkt, um so die Wahrheitsfindung zu beeinträchtigen (Art. 221 Abs. 1 lit. b StPO). Verdunkelung kann gemäss der Rechtsprechung insbesondere in der Weise erfolgen, dass sich die beschuldigte Person mit Zeugen, Auskunftspersonen, Sachverständigen oder Mitbesc

```json
{
  "enrichment_source": "static",
  "rag_enrichment": {
    "legal_area": "criminal law and criminal procedure",
    "primary_domain": "",
    "secondary_domain": "",
    "topic": "",
    "subtopic": "",
    "micro_topic": "",
    "doctrinal_rule": "",
    "legal_test": "",
    "procedural_context": "",
    "legal_topic": "",
    "legal_rule": "",
    "court_holding": "",
    "factual_context": "",
    "legal_domain_path": [],
    "concepts_en": [
      "collusion risk",
      "criminal procedure",
      "evidence assessment",
      "pretrial detention",
      "procedural rights"
    ],
    "terms_original": [
      "kollusionsgefahr",
      "verdunkelungsgefahr",
      "sachverstandige",
      "zeuge",
      "untersuchungshaft"
    ],
    "fact_pattern_tags": [],
    "authority_role": [
      "unpublished_federal_decision",
      "reasoning_consideration",
      "multi_consideration_decision"
    ],
    "english_legal_concepts": [],
    "search_keywords": [],
    "specificity_score": 0.0,
    "paragraph_role": "facts",
    "outcome_signal": "none",
    "statute_anchors": [
      "Art. 221 Abs. 1 lit. b StPO",
      "Art. 221 Abs. 1 Bst. b StPO"
    ],
    "case_anchors": [
      "BGE 137 IV 122 E. 4.2",
      "7B_69/2024",
      "7B_1028/2023 12.01.2024 E. 8.1",
      "7B_69/2024 21.02.2024 E. 3.3.2",
      "BGE 132 I 21 E. 3.2.2"
    ],
    "legal_source_anchors": [],
    "secondary_sources": [],
    "document_or_plan_anchors": [],
    "event_anchors": [
      "21. Februar 2024"
    ]
  }
}
```

#### `7B_12/2025 E. 2.2`

**`text_excerpt_original` (head):** 2.2. Der Haftgrund der Kollusionsgefahr liegt vor, wenn ernsthaft zu befürchten ist, dass der Beschuldigte Personen beeinflusst oder auf Beweismittel einwirkt, um so die Wahrheitsfindung zu beeinträchtigen (Art. 221 Abs. 1 lit. b StPO). Strafprozessuale Haft wegen Kollusions- bzw. Verdunkelungsgefahr soll verhindern, dass die beschuldigte Person die wahrheitsgetreue Abklärung des Sachverhalts vere

```json
{
  "enrichment_source": "static",
  "rag_enrichment": {
    "legal_area": "criminal law and criminal procedure",
    "primary_domain": "",
    "secondary_domain": "",
    "topic": "",
    "subtopic": "",
    "micro_topic": "",
    "doctrinal_rule": "",
    "legal_test": "",
    "procedural_context": "",
    "legal_topic": "",
    "legal_rule": "",
    "court_holding": "",
    "factual_context": "",
    "legal_domain_path": [],
    "concepts_en": [
      "collusion risk",
      "criminal procedure",
      "pretrial detention",
      "procedural rights"
    ],
    "terms_original": [
      "kollusionsgefahr",
      "verdunkelungsgefahr"
    ],
    "fact_pattern_tags": [],
    "authority_role": [
      "unpublished_federal_decision",
      "reasoning_consideration",
      "multi_consideration_decision"
    ],
    "english_legal_concepts": [],
    "search_keywords": [],
    "specificity_score": 0.0,
    "paragraph_role": "facts",
    "outcome_signal": "none",
    "statute_anchors": [
      "Art. 221 Abs. 1 lit. b StPO",
      "Art. 221 Abs. 1 Bst. b StPO"
    ],
    "case_anchors": [
      "7B_687/2024 12.07.2024 E. 4.1",
      "BGE 137 IV 122 E. 4.2"
    ],
    "legal_source_anchors": [],
    "secondary_sources": [],
    "document_or_plan_anchors": [],
    "event_anchors": []
  }
}
```

#### `1B_15/2023 E. 3.1`

**`text_excerpt_original` (head):** 3.1. Der besondere Haftgrund der Kollusionsgefahr liegt vor, wenn ernsthaft zu befürchten ist, dass der oder die Beschuldigte Personen beeinflusst oder auf Beweismittel einwirkt, um so die Wahrheitsfindung zu beeinträchtigen (Art. 221 Abs. 1 lit. b StPO). Verdunkelung kann nach der bundesgerichtlichen Praxis insbesondere in der Weise erfolgen, dass sich die beschuldigte Person mit Zeugen, Auskunft

```json
{
  "enrichment_source": "static",
  "rag_enrichment": {
    "legal_area": "criminal procedure and coercive measures",
    "primary_domain": "",
    "secondary_domain": "",
    "topic": "",
    "subtopic": "",
    "micro_topic": "",
    "doctrinal_rule": "",
    "legal_test": "",
    "procedural_context": "",
    "legal_topic": "",
    "legal_rule": "",
    "court_holding": "",
    "factual_context": "",
    "legal_domain_path": [],
    "concepts_en": [
      "collusion risk",
      "criminal procedure",
      "evidence assessment",
      "pretrial detention",
      "procedural rights"
    ],
    "terms_original": [
      "kollusionsgefahr",
      "kollusionsrisiko",
      "verdunkelungsgefahr",
      "sachverstandige",
      "zeuge",
      "untersuchungshaft"
    ],
    "fact_pattern_tags": [],
    "authority_role": [
      "unpublished_federal_decision",
      "reasoning_consideration",
      "multi_consideration_decision"
    ],
    "english_legal_concepts": [],
    "search_keywords": [],
    "specificity_score": 0.0,
    "paragraph_role": "facts",
    "outcome_signal": "none",
    "statute_anchors": [
      "Art. 221 Abs. 1 lit. b StPO",
      "Art. 212 Abs. 2 Bst. c",
      "Art. 221 Abs. 1 Bst. b StPO"
    ],
    "case_anchors": [
      "BGE 137 IV 122 E. 4.2",
      "1B_357/2022",
      "1B_357/2022 22.07.2022 E. 3.1",
      "1B_558/2021 03.11.2021 E. 3.2",
      "BGE 132 I 21 E. 3.2.1",
      "BGE 140 IV 74 E. 2.2"
    ],
    "legal_source_anchors": [],
    "secondary_sources": [],
    "document_or_plan_anchors": [],
    "event_anchors": [
      "22. Juli 2022"
    ]
  }
}
```

---

## 5. What makes these enrichments "gold" for val_001 - strong signals by field

val_001 is a *cross-lingual* query: English question, German corpus. Exact match between query tokens and citation text is mostly impossible - the LLM enrichment is the bridge. The signals below are the fields where a retriever (or a structured prefilter / judge prompt) gets the highest mileage. Order is rough: highest-discrimination first.

### 5.1 Law-card fields (schema: `law_descriptor_v1`)

Strong signals, in order of usefulness for this query:

| # | Field | Why it discriminates val_001 from the corpus | Concrete val_001 examples |
|---|---|---|---|
| 1 | `concepts_en` | Pure English concept tags. Dense embedding lands here. | `Art. 221 Abs. 1 StPO`: ["Investigative detention","Security detention","Criminal suspicion","Risk assessment","Evidence tampering","Public safety","Prior offenses"] - direct overlap with query terms "pre-trial detention", "risk of collusion", "tamper with evidence". |
| 2 | `english_summary` | Single English sentence stating what the article does. Bridges EN-query/DE-corpus vocab gap (Obs 3, 4.3% noun overlap). | `Art. 227 Abs. 1 StPO`: "If detention is to be extended, the prosecutor must apply to the compulsory measures court within three months..." - literally answers "may a court order a three-month extension". |
| 3 | `applicability_conditions` | Bullet list of conditions for the rule to apply. Acts like a decomposed retrieval target. | `Art. 221 Abs. 1 StPO`: ["...strongly suspected of crime...","Serious risk of evading prosecution...","Risk of influencing persons or evidence","Risk of endangering others..."] - each bullet maps onto a phrase in the query. |
| 4 | `legal_question` | One-sentence canonical question this article answers. | `Art. 221 Abs. 1 StPO`: "When is investigative detention lawful under criminal procedure?" - near-paraphrase of the val_001 question. |
| 5 | `terms_de_to_en` | Aligned DE-EN glossary; the *only* field that stitches "Untersuchungshaft"/"Kollusionsgefahr" to "pre-trial detention"/"collusion risk". | `Art. 221 Abs. 1 StPO`: {Untersuchungs- und Sicherheitshaft -> Investigative and security detention, dringend verdaechtig -> strongly suspected, Beweismittel -> evidence}. Without this field, German BM25 cannot recall an English query. |
| 6 | `legal_rule` | One-line normative statement. Useful for the judge stage. | `Art. 221 Abs. 1 StPO`: "Investigative and security detention permitted under specific suspicion and risk conditions". |
| 7 | `provision_role_llm` | Macro tag in {scope, definition, right_or_entitlement, duty, sanction, procedure, ...}. Lets the prefilter narrow to "procedure"-/"scope"-type articles for a procedural query. | `Art. 221 Abs. 1 StPO` = "scope"; `Art. 222 StPO` = "right_or_entitlement"; `Art. 227 Abs. 1 StPO` = "procedure"; `Art. 100 Abs. 1 BGG` = "duty" - the bundle is exactly the role-mix needed to answer a "may the court extend detention?" question. |
| 8 | `addressees` | Who the rule binds: court, prosecutor, defendant. Filters out enforcement-only or admin-only articles. | All 19 law citations have addressees superset {court, public prosecutor}; the query is about a court ruling on a prosecutor's motion. |
| 9 | `sanctions_or_consequences` | What follows from non-compliance / from invoking the rule. Useful for proportionality framing. | `Art. 221 Abs. 1 StPO`: ["Detention ordered by court","Must be proportionate to risk","Reviewable by judicial authority"] - directly cites proportionality, which val_001 names. |
| 10 | `specificity_score` (0-1) | Higher = article speaks to a narrow legal situation; useful as a tie-breaker. | `Art. 221 Abs. 1 StPO` = 0.8 (high), `Art. 100 Abs. 1 BGG` = 0.6 (medium - generic appeal-deadline rule, still gold because the case made it to BGer). |
| 11 | `llm_priority` ("high"/"medium"/"low") | Pre-LLM heuristic: `high` ~ frequently cited / structurally important. Not perfect but a free prior. | All val_001 StPO/StGB articles are `medium`/`high`. |

Fields that help less for this query: `defined_terms` (often empty), `exceptions_or_limitations` (often empty for procedural articles).

### 5.2 Court-card fields (schema: `minimal_descriptor_only`)

Court enrichment is *paragraph-granular*: one record per `BGE ... E. x.y` consideration, not per decision. The signals are different from law cards:

| # | Field | Why it discriminates val_001 | Concrete val_001 examples |
|---|---|---|---|
| 1 | `micro_topic` | Most specific tag - a one-line topical caption. This is the single strongest discriminator for case-law retrieval. | `BGE 137 IV 122 E. 4.2`: "Collusion risk as basis for preventive detention"; `BGE 137 IV 122 E. 6.2`: "Obligation to release pretrial detainee upon expiration of detention period"; `BGE 132 I 21 E. 3.2.1`: "Evaluation of evidence in criminal proceedings" - every micro_topic is a near-paraphrase of an aspect of the query. |
| 2 | `legal_domain_path` | Hierarchy ["Criminal law","Procedure","Detention"] - feeds the prefilter directly. | 12/13 court records have path starting `["Criminal law", ...]` or `["Constitutional law", ...]` matching the StPO/BV scope of val_001. |
| 3 | `concepts_en` + `terms_original` | EN concepts + DE terms in parallel (no alignment, but both languages). | `BGE 137 IV 122 E. 4.2`: concepts ["Preventive detention","Collusion risk","Interference with evidence","Criminal procedure","Detention order"], terms_original ["Untersuchungshaft","Ersatzmassnahmen","Kollusionsgefahr","Beweismittel","Strafverfahren"]. Cross-lingual hit on every val_001 keyword. |
| 4 | `paragraph_role` | {`legal_standard`, `reasoning`, `facts`, `outcome`, ...}. **`legal_standard`** considerations are the ones lawyers cite - the gold for this query is dominated by them. | `BGE 137 IV 122 E. 4.2` / `BGE 132 I 21 E. 3.2.1` / `BGE 133 I 168 E. 4.1` / `BGE 143 IV 168 E. 5.1` / `BGE 139 IV 270 E. 3.1` are all `legal_standard`. This field alone separates citable-doctrine paragraphs from boilerplate. |
| 5 | `legal_area_static` (NOT LLM, derived from court metadata) | Pre-LLM family tag. Cheap, near-perfect filter. | All 13 found court records have `legal_area_static = "criminal law and criminal procedure"`. A prefilter "show only criminal-procedure paragraphs" already prunes ~80% of the 2.47M court corpus for val_001. |
| 6 | `topic` / `subtopic` | Mid-grain tags. Less specific than micro_topic but more recall-safe. | `topic = "Preventive detention"` for 6/13 records, `topic = "Detention review"` / `"Detention extension"` for 3 more. |
| 7 | `fact_pattern_tags` | Short tags describing the factual scenario the consideration speaks to. Useful for fact-grounding the retrieval against the val_001 fact pattern. | `BGE 137 IV 122 E. 4.2`: ["Detention","Collusion","Evidence","Criminal procedure"] - same vocabulary as the query's fact recital. |
| 8 | `procedural_context` | Short label like "Criminal procedure", "Constitutional review". | All 13 found records are "Criminal procedure"-context. |
| 9 | `authority_role` | Issuing body. For BGE/1B_/7B_ records this is invariant ("Swiss Federal Court") - low discrimination *within* val_001 gold but high discrimination across families (e.g. cantonal vs federal). | Constant: ["Swiss Federal Court"]. |
| 10 | `specificity_score` | 0.8-0.9 for the gold paragraphs; the boilerplate considerations score 0.2-0.3. | Mean specificity of the 13 found gold paragraphs ~ 0.85. |
| 11 | `text` (the raw paragraph) | Free-text DE. Low discrimination via embedding alone (Obs 3) but high recall via BM25 once query is German-expanded. | `BGE 137 IV 122 E. 4.2` text begins "Gemaess Art. 221 Abs. 1 lit. b i.V.m. Art. 237 Abs. 1 StPO ist Untersuchungshaft..." - explicitly cites the same article that the val_001 query asks about. |

### 5.3 The conjunction that actually screams "gold for val_001"

No single enrichment field is sufficient. The pattern that shows up in every one of the 32 found gold records, and is **rare** elsewhere in the 2.65M-row corpus, is the **conjunction**:

1. **Topical channel:** `legal_area`/`legal_area_static`/`legal_domain_path` lands in *Criminal procedure -> Detention* (laws) or *Criminal law -> Procedure -> Detention* (court).
2. **Concept channel:** `concepts_en` superset {pre-trial / preventive / investigative detention} intersect {collusion risk / evidence tampering / proportionality}.
3. **Cross-lingual channel:** `terms_de_to_en` (laws) or `terms_original` (court) contains at least one of {Untersuchungshaft, Kollusionsgefahr, Beweismittel, Verlaengerung, Haft}.
4. **Role channel:** `provision_role_llm in {scope, procedure, right_or_entitlement, duty}` (laws) **and** `paragraph_role = legal_standard` (court). This is what separates *citable* doctrine from facts/boilerplate/outcome paragraphs that share the same topic.
5. **Specificity gate:** `specificity_score >= 0.6`.

Operationally, a prefilter that ANDs (1) AND (2) AND (4) and ranks by (3) UNION TF-IDF over `english_summary`+`micro_topic` would have placed 30+/32 of the present-in-enrichment gold inside a 5-10k-row pool - well above the current 23.1% pool-recall ceiling reported in `endgame_handoff_2026-05-09.md` section 4.1.

### 5.4 Static-enrichment signals — what to use when there is no LLM payload

For the 10 court gold without an LLM call, v5_unified still gives a usable signal set. The fields that survive the static path are the deterministically-extractable ones, and several of them are the *strongest* discriminators for val_001 anyway:

| Static field (present without LLM) | Why it still works for val_001 |
|---|---|
| `legal_area` (rule-based, e.g. `"criminal procedure and coercive measures"`) | Directly filters the corpus to the criminal-procedure family. All 10 missing-from-LLM gold land in this area. |
| `concepts_en` (regex/keyword-derived) | The 10 records all have `concepts_en` superset {`pretrial detention`, `collusion risk`, `criminal procedure`, `procedural rights`}. Same English vocabulary as the LLM rows — just a smaller, less curated set. |
| `terms_original` (multilingual, including French) | `1B_210/2023 E. 4.1` carries French terms `["danger de collusion","risque de collusion","detention provisoire","moyens de preuve"]`. Important: French considerations exist in the gold (val_001 likely involves a French-speaking canton case). The LLM run skipped them; the static path catches them. |
| `statute_anchors` (regex extraction from text) | `1B_210/2023 E. 4.1`: `["art. 221 al. 1 let. b CPP","Art. 221 Abs. 1"]` — same article the val_001 query names. `1B_90/2021 E. 2.1`: `["Art. 221 Abs. 1 lit. b StPO","Art. 221 Abs. 1 Bst. b StPO",...]`. This is arguably the single highest-signal static field — a query that names `Art. 221 StPO` matches via raw text. |
| `case_anchors` (regex over the text) | `1B_90/2021 E. 2.1`: `["BGE 137 IV 122 E. 4.2","BGE 132 I 21 E. 3.2.1","BGE 132 I 21 E. 3.4",...]`. The static layer captures cross-citations the gold-graph would otherwise miss, and these anchors are themselves val_001 gold — strong "co-cited with gold" prior. |
| `paragraph_role` (rule-based: `legal_standard` / `facts` / `procedural_history` / `outcome` / ...) | Several missing-from-LLM rows are `paragraph_role = "legal_standard"` — same role the LLM rows hit for the same query. Best discriminator between citable doctrine and boilerplate, and it does not depend on the LLM. |
| `authority_role` (e.g. `["unpublished_federal_decision","reasoning_consideration"]`) | All 10 are unpublished federal (1B_/7B_ docket); separates them from cantonal noise. |
| `outcome_signal` | Mostly `"none"` here, but tracks {grants, rejects, dismisses, partially_grants, ...} for downstream prioritization. |

Fields that go missing on the static path (and what's lost):
- `english_summary` / `legal_question` / `legal_rule` / `doctrinal_rule` / `legal_test` — empty. Hurts cross-lingual recall most when the query has no matching German term.
- `terms_de_to_en` (the aligned glossary) — empty for static; only `terms_original` (one-sided) is filled.
- `micro_topic`, `topic`, `subtopic`, `legal_domain_path` — empty for static. Coarse `legal_area` substitutes.
- `specificity_score` — defaults to 0.0 for static rows, so it cannot be used as a tie-breaker against LLM rows.

**Practical implication for the prefilter (handoff Move 1):** the index can use a unified prefilter over v5_unified (not just the 363k file). Static-only rows still satisfy the topical channel (`legal_area`), the concept channel (`concepts_en`), and the role channel (`paragraph_role`); they will fail the cross-lingual channel for English-only queries unless the query is German-expanded first (which the pipeline already does via the `german_expansion_cache.json` step from Move 1's prep). With v5_unified wired in, the previous "10/42 unreachable" claim is replaced by "10/42 reachable via static signals only — lower-richness but lands in the prefilter pool".

### 5.5 Other caveats

- The handoff records (`endgame_handoff_2026-05-09.md` section 6.2) that the law file has UTF-8 mojibake artifacts when read through cp1252 — visible above as `verd?chtig` etc. The bytes are clean; the *content* is fine. Do not treat the mojibake glyphs as a missing-signal problem.
- The `paragraph_role` value for `BGE 137 IV 122 E. 4.2` differs between the two enrichment paths: the source LLM file says `"legal_standard"` (matches the consideration text); v5_unified's normalizer assigns `"facts"`. Normalizer disagreement, not a content change — for retrieval, prefer the source LLM value when both are available.

---

## 6. Source files referenced

- Query + gold: `data/val.csv` row 2 (`val_001`)
- Law enrichment: `law_json_llm_output/law_llm_descriptors_0000000_all.jsonl` (173,033 records)
- Court LLM enrichment: `outputs_from_363k_run/court_llm_descriptors_0000000_all.jsonl` (363,258 records — ~15% of 2.47M court corpus)
- **Court unified enrichment (LLM + static, full 2.47M):** `artifacts/court_authority_cards_v5_unified.jsonl` (10.4 GB; `enrichment_source ∈ {"llm+static","static"}`). This is the file that closes the 10-citation gap for val_001.
- Cached extracted enrichments (sibling artifacts):
  - `research/val_001_gold_enrichments.json` — 32 LLM-enriched gold records (19 law + 13 court)
  - `research/val_001_static_only_court_enrichments.json` — 10 static-only court gold records
- Background / constraints: `research/endgame_handoff_2026-05-09.md`
