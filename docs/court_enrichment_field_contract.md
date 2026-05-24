# Court Enrichment Field Contract

This document is generated from a full scan of `data/court_considerations.csv`.
It is query-independent and does not use validation/test gold labels.

## Corpus Snapshot

- Rows scanned: 2,476,315
- Scan time: 6293.7s
- Input: `data\court_considerations.csv`
- Current smoke output inspected: `C:\Users\samiul\Downloads\enriched_court_citations_10 (1).jsonl`

### Languages

| language | rows |
|---|---:|
| `de` | 1,443,322 |
| `fr` | 823,868 |
| `it` | 138,151 |
| `unknown` | 70,974 |

### Citation Families

| family | rows |
|---|---:|
| `docket_consideration` | 2,218,564 |
| `other_court_citation` | 161,286 |
| `bge_consideration` | 96,465 |

## What The LLM Should Produce

The LLM should only produce query-neutral legal descriptors. It should not be trusted
as the source of truth for formal anchors.

Current notebook `rag_enrichment` fields:

`authority_role, case_anchors, concepts_en, doctrinal_rule, fact_pattern_tags, legal_area, legal_domain_path, legal_test, micro_topic, outcome_signal, paragraph_role, procedural_context, specificity_score, statute_anchors, subtopic, terms_original, topic`

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

- Article anchors: 3,271,050
- Cantonal section anchors: 102,586
- SR/RS anchors: 94,037
- BGE/ATF/DTF case anchors: 1,088,471
- Federal docket case anchors: 1,011,420
- Cantonal/special decision anchors: 1,366
- BGE page references: 44,253

## Statute Anchor Rules

`statute_anchors` must be deterministic. Accept only:

- `Art. ... CODE`, e.g. `Art. 221 Abs. 1 lit. b StPO`, `art. 34 Cst.`
- `§ ... CODE`, e.g. `§ 25 Abs. 3 PBG/SZ`
- `SR ...` / `RS ...` numbers

Do not allow the model to put plain law names, plans, events, bibliography, or cases
inside `statute_anchors`. Move them to the correct fields.

Top code-like tokens after article/section anchors:

| code/token | matches |
|---|---:|
| `BGG` | 569,710 |
| `LTF` | 395,866 |
| `BV` | 102,874 |
| `OG` | 84,019 |
| `StGB` | 75,603 |
| `Cst` | 70,287 |
| `StPO` | 61,891 |
| `CP` | 59,959 |
| `ZGB` | 52,480 |
| `CO` | 51,527 |
| `OJ` | 47,407 |
| `CPP` | 45,988 |
| `OR` | 42,662 |
| `EMRK` | 42,053 |
| `ATSG` | 40,978 |
| `CC` | 39,512 |
| `Satz` | 34,496 |
| `ZPO` | 33,578 |
| `IVG` | 29,080 |
| `SchKG` | 25,116 |
| `CPC` | 23,089 |
| `CEDH` | 23,066 |
| `AuG` | 16,801 |
| `LP` | 16,760 |
| `DBG` | 14,645 |
| `SVG` | 13,962 |
| `AVIG` | 13,593 |
| `UVG` | 12,814 |
| `IVV` | 11,490 |
| `BVG` | 11,090 |
| `LPGA` | 11,082 |
| `RPG` | 10,431 |
| `LEtr` | 10,394 |
| `KVG` | 9,925 |
| `LIFD` | 9,626 |
| `AHVG` | 9,467 |
| `ANAG` | 9,430 |
| `LDIP` | 9,429 |
| `AIG` | 9,051 |
| `LAT` | 8,827 |

Legal-source words that should become `legal_source_anchors`, not `statute_anchors`:

| source word | rows |
|---|---:|
| `Gesetz` | 221,264 |
| `loi` | 140,099 |
| `Weisung` | 104,670 |
| `ordonnance` | 48,295 |
| `Bundesgesetz` | 36,899 |
| `Reglement` | 35,000 |
| `règlement` | 35,000 |
| `Verordnung` | 29,758 |
| `loi fédérale` | 23,216 |
| `Botschaft` | 13,845 |
| `legge` | 11,919 |
| `message` | 8,457 |
| `Richtlinie` | 4,565 |
| `directive` | 4,192 |
| `instructions` | 3,583 |
| `legge federale` | 3,382 |
| `Erläuterungen` | 2,730 |
| `Kreisschreiben` | 2,094 |
| `regolamento` | 2,091 |
| `ordinanza` | 2,021 |
| `messaggio` | 1,546 |
| `circulaire` | 1,479 |
| `istruzioni` | 510 |
| `direttiva` | 341 |
| `circolare` | 315 |

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

| secondary source signal | rows |
|---|---:|
| `StE` | 1,236,047 |
| `Pra` | 396,326 |
| `in:` | 104,789 |
| `Traité` | 51,962 |
| `Kommentar` | 44,422 |
| `SJ` | 23,503 |
| `SVR` | 21,362 |
| `Commentaire` | 20,204 |
| `Basler Kommentar` | 19,389 |
| `Kommentar zum` | 12,335 |
| `ASA` | 11,915 |
| `Commentaire romand` | 7,386 |
| `Berner Kommentar` | 5,037 |
| `ZBl` | 4,777 |
| `RDAF` | 4,147 |
| `Handbuch` | 3,470 |
| `Zürcher Kommentar` | 3,109 |
| `AJP` | 3,010 |
| `SZS` | 2,462 |
| `PJA` | 2,125 |
| `DTA` | 2,017 |
| `ZBJV` | 1,534 |
| `RtiD` | 1,404 |
| `JdT` | 1,368 |
| `sic!` | 1,225 |
| `ZSR` | 910 |
| `BlSchK` | 721 |
| `EuGRZ` | 606 |
| `BJM` | 366 |
| `Lehrbuch` | 205 |

## Document, Plan, And Event Anchors

These are useful for retrieval but must not pollute statute/case anchors.

Document/plan signals:

| document signal | rows |
|---|---:|
| `rapport` | 86,867 |
| `Bericht` | 81,756 |
| `Gutachten` | 62,457 |
| `Vertrag` | 56,234 |
| `expertise` | 39,439 |
| `contrat` | 38,617 |
| `Police` | 23,775 |
| `Protokoll` | 12,879 |
| `Baubewilligung` | 11,660 |
| `Baugesuch` | 5,444 |
| `rapporto` | 5,430 |
| `procès-verbal` | 5,406 |
| `proces-verbal` | 5,406 |
| `permis de construire` | 4,649 |
| `contratto` | 4,641 |
| `Zahlungsbefehl` | 3,757 |
| `perizia` | 2,926 |
| `Nutzungsplan` | 2,676 |
| `Zonenplan` | 2,174 |
| `Gestaltungsplan` | 2,016 |
| `licenza edilizia` | 1,670 |
| `Richtplan` | 1,557 |
| `plan d'affectation` | 1,500 |
| `piano regolatore` | 1,325 |
| `Gesamtarbeitsvertrag` | 878 |
| `plan de zones` | 168 |
| `Abstimmungsunterlagen` | 109 |

Event signals:

| event signal | rows |
|---|---:|
| `Wahl` | 19,619 |
| `Initiative` | 5,818 |
| `initiative` | 5,818 |
| `Abstimmung` | 4,933 |
| `élection` | 2,608 |
| `Referendum` | 2,105 |
| `référendum` | 2,105 |
| `referendum` | 2,105 |
| `Volksabstimmung` | 1,402 |
| `votation` | 1,192 |
| `iniziativa` | 536 |
| `votazione` | 472 |
| `elezione` | 256 |

## Paragraph Role Rules

Recommended enum:

`holding`, `reasoning`, `facts`, `procedural_history`, `legal_standard`, `application`, `citation`, `costs`, `notification`, `disposition`, `neutral`

Corpus role cue counts:

| role cue | rows |
|---|---:|
| `procedural_history` | 801,507 |
| `facts` | 756,389 |
| `legal_standard` | 391,372 |
| `costs` | 377,561 |
| `application` | 193,373 |
| `disposition` | 181,320 |
| `notification` | 165,893 |

Rules:

- If text is notification/cost boilerplate, do not create doctrinal rules.
- If role is `facts` or `procedural_history`, `doctrinal_rule` should be empty or descriptive only.
- If role is `legal_standard`, rule/test may be normative.
- If role is `application` or `holding`, rule/test may describe how the court applied the rule.

## Outcome Signal Rules

Recommended enum:

`neutral`, `dismissed`, `granted`, `partial`, `inadmissible`, `remitted`, `none`

Corpus outcome cue counts:

| outcome cue | rows |
|---|---:|
| `inadmissible` | 172,348 |
| `remitted` | 103,441 |
| `dismissed` | 101,814 |
| `granted` | 18,046 |
| `partial` | 16,340 |

Rules:

- Default to `neutral` or `none`.
- Set `remitted` only when the paragraph states the court remitted the matter, not merely because a party requested remand.
- Set `dismissed`, `granted`, `partial`, or `inadmissible` only on disposition/holding paragraphs with explicit court language.
- Do not infer outcome from legal argument paragraphs.

## Known Legal Issue Coverage

The current deterministic concept dictionary already detects these issue labels.
The LLM must be allowed to produce more specific `topic/subtopic/micro_topic`, but
these labels are good retrieval and QA guardrails.

| issue label | rows |
|---|---:|
| `appeal admissibility` | 435,500 |
| `evidence assessment` | 318,889 |
| `administrative procedure` | 303,729 |
| `ECHR and fundamental rights` | 272,226 |
| `invalidity insurance` | 263,510 |
| `inheritance and testament` | 211,435 |
| `migration and residence` | 150,819 |
| `criminal sentencing` | 130,525 |
| `liability and damages` | 110,433 |
| `employment and dismissal` | 109,710 |
| `debt enforcement and bankruptcy` | 108,163 |
| `asylum and refugee status` | 93,098 |
| `police and public security` | 75,976 |
| `right to be heard` | 71,470 |
| `maintenance and child support` | 61,519 |
| `tax assessment` | 41,915 |
| `health insurance` | 37,357 |
| `spatial planning and zoning` | 37,189 |
| `pretrial detention` | 36,791 |
| `direct democracy and political rights` | 35,035 |
| `old age and survivors insurance` | 33,474 |
| `proportionality` | 33,337 |
| `contract interpretation` | 27,378 |
| `occupational pension` | 22,847 |
| `international mutual legal assistance` | 21,491 |
| `drug offences` | 19,393 |
| `road traffic and accidents` | 18,130 |
| `competition and antitrust` | 16,658 |
| `environmental protection` | 12,960 |
| `recidivism risk` | 9,396 |
| `epidemic and public health` | 9,197 |
| `flight risk` | 6,270 |
| `data protection and privacy` | 5,084 |
| `banking duty of care` | 4,040 |
| `collusion risk` | 3,412 |

Areas represented by known issue labels:

| issue area | rows |
|---|---:|
| `procedure` | 754,389 |
| `administrative law` | 303,729 |
| `social insurance` | 296,984 |
| `human rights law` | 272,226 |
| `inheritance law` | 211,435 |
| `migration law` | 150,819 |
| `criminal law` | 149,918 |
| `private law` | 137,811 |
| `employment law` | 109,710 |
| `debt enforcement` | 108,163 |
| `asylum law` | 93,098 |
| `public security law` | 75,976 |
| `constitutional procedure` | 71,470 |
| `family law` | 61,519 |
| `criminal procedure` | 55,869 |
| `tax law` | 41,915 |
| `health insurance law` | 37,357 |
| `spatial planning law` | 37,189 |
| `constitutional and public law` | 35,035 |
| `constitutional and procedural law` | 33,337 |
| `pension law` | 22,847 |
| `international law` | 21,491 |
| `traffic law` | 18,130 |
| `competition law` | 16,658 |
| `environmental law` | 12,960 |
| `public health law` | 9,197 |
| `data protection law` | 5,084 |
| `banking and obligations law` | 4,040 |

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
