# Structured-Data Schema v2 — `court_considerations.csv` Paragraph Catalogue

_Generated 2026-05-21 as a critical revision of [court_consideration_schema_v1_2026-05-21.md](court_consideration_schema_v1_2026-05-21.md). Three improvements over v1:_

1. **Coverage gaps closed**: 30 new fields added for sister-court dockets (BVGer/BStGer/BPatGer), international-treaty subclasses (Lugano, Hague, EU directives/regulations, OECD MTC, UN conventions), pre-2007 statute forms (OG/PPF/ANAG/LSEE/LDDS), travaux préparatoires sub-classes, anonymization fingerprint, vorinstanz/canton extraction, and chamber-area derivation. Total: **132 fields**.
2. **Extraction-method tag on every field**: each field is classified `regex` (pure pattern), `regex+rules` (pattern + small post-processing rule), `derived` (computed from other fields), `llm_enrich` (needs an LLM-classifier pass over corpus), or `judge_time` (depends on query, computed at retrieval time).
3. **Honest separation of "what static code can do" from "what needs an LLM pass"**: ~95 of the 132 fields are statically extractable; ~14 need an LLM enrichment pass; ~1 is judge-time only.

The downstream consequence: when this schema is finalised, a static extraction pipeline can populate ~95 fields across the 2,476,315 rows in a single sweep. The ~14 LLM-enrich fields can be filled in a second pass with a small classifier (Qwen3-1.7B or smaller) at ~$50-100 inference cost on the full corpus.

---

## 0. Design principles (unchanged from v1)

1. **Language-agnostic where possible**: every field that has a DE form must also support FR and IT forms. The triple `BGE ↔ ATF ↔ DTF` is the most important alias and is treated as a single canonical entity.
2. **`sr_number` as the canonical statute key**: every statute citation across DE/FR/IT carries the same `SR/RS xxx.xxx` number. This is the language-invariant identity anchor.
3. **Content-driven validation**: never trust the `citation` column blindly. Numbers, years, and URL fragments leak into it. Every `erwagung_id` must be cross-validated against the body text.
4. **Four temporal docket eras**: pre-1990 (`P.NNN/YYYY` with dot), 1990-2007 (`1A./1P./2A./4C./5A./6P./7B./H/I`), 2007-2024 (`1B_/1C_/2C_/4A_/5A_/6B_/8C_/9C_` with underscore), post-2024 (`7B_` introduced, `6B_` deprecated).
5. **Content-driven citation-noise**: BGE rows on environmental/scientific/quantitative cases (percentages, m/s, ha, GWh, clock times) trigger ~22% noise; procedural unpublished BGer rows trigger ~0%. Schema has separate noise-class fields.
6. **Tri-lingual paraphrase verbs** are catalogued explicitly. The 7-element template applies content-driven (not publication-tier-driven), so schema must extract the template across DE/FR/IT.
7. **Foreign-law discrimination**: Swiss-IT prose mixed with Italian-state references (`comma` vs `cpv.`, `D.L.` vs `SR/RS`) must be tagged at the citation level.
8. **Echo-decision is the dominant mode** (>85% mid-corpus). The schema has fields for citation graph that capture which Leitentscheide a paragraph echoes.

### NEW v2 principle:

9. **Static-first, LLM-second**: every field carries an `extraction_method` tag. Static fields are the regex-engine spec; LLM-enrich fields are the second-pass classifier spec. The two never overlap, so the extraction pipeline can be cleanly partitioned.

---

## 0.5 Extraction-method taxonomy

| Tag | Meaning | Tools | % of v2 fields |
|---|---|---|---|
| `trivial` | Assigned at CSV load (row index, raw text length, raw E-tag). | none | ~3% |
| `regex` | Single pattern, no post-processing. | Python `re`, ripgrep. | ~38% |
| `regex+rules` | Pattern match + small decision-rule (e.g. cross-check E-id against body, lookup sr_number from short-name). | Python `re` + lookup tables. | ~43% |
| `derived` | Computed from other extracted fields. | Pure arithmetic / boolean. | ~5% |
| `llm_enrich` | Needs semantic classification — verb-ownership, paraphrase detection, paragraph-role multi-class. | Small LLM (Qwen3-1.7B / Qwen3-4B in classifier mode). | ~10% |
| `judge_time` | Depends on the query — cannot be pre-computed. | Stage-2 LLM at retrieval time. | ~1% |

**Operationally**: a single Python pipeline can populate every `trivial` + `regex` + `regex+rules` + `derived` field in one sweep of the 2.47M rows (~5-15 minutes on a single machine with PyPy or compiled regex). The `llm_enrich` fields then need a second pass with a classifier model.

---

## 1. Field catalogue (132 fields, grouped)

### A. Identity & provenance (14 fields, +4 vs v1)

| # | Field | Type | Method | Description / extraction rule |
|---|---|---|---|---|
| 1 | `row_id` | int | trivial | Original CSV row index (1-2,476,315). |
| 2 | `bge_id` | string | regex | Normalised BGE/ATF/DTF id (`BGE 137 IV 122`). **Always DE-form `BGE`** regardless of decision language. Regex: `\b(?:BGE|ATF|DTF)\s+(\d{1,3})\s+([IVXLCDM]+[abc]?)\s+(\d{1,4})\b`. |
| 3 | `bge_volume` | int | regex | Volume number from #2. |
| 4 | `bge_series_roman` | enum | regex | One of `I, II, III, IV, V, VI`. Pre-1995 uses Arabic suffix `Ia/Ib/Ic`. |
| 5 | `bge_start_page` | int | regex | Volume page from #2. |
| 6 | `docket_no` | string | regex | Unpublished BGer docket (`1B_28/2022`, `2P.137/2005`, `P.923/1982`, `H 316/03`, `I 345/06`). Regex: `\b([1-9][A-Z][_.]\d{1,4}/\d{2,4}|[HIK]\s+\d{1,4}/\d{2,4}|[1-7][A-Z]_\d{1,4}/\d{2,4})\b`. |
| 7 | `docket_era` | enum | regex+rules | `pre_1990` / `1990_2006` / `2007_2023` / `post_2024`. Derived from docket-format. |
| 8 | `chamber_prefix` | string | regex | Two-three-letter prefix (`1B_`, `1C_`, `2C_`, `2P.`, `4A_`, `4C.`, `5A_`, `5D_`, `5F_`, `6B_`, `6P.`, `6S.`, `7B_`, `1F_`, `2F_`, `8C_`, `8D_`, `9C_`, `H`, `I`, `1A.`, `1P.`, `2A.`, `2D_`). |
| 9 | `chamber_area` | enum | regex+rules | **NEW v2.** Derived from chamber_prefix via lookup: `1B_/1F_=criminal-prelim`, `1C_=public-law`, `2C_/2D_=tax-admin`, `4A_/4C.=civil`, `5A_/5D_/5F_=family-debt`, `6B_/6S.=criminal`, `7B_=criminal (post-2024)`, `8C_/8D_=social-insurance`, `9C_=social-insurance-2`, `H=AHV-insurance-old`, `I=IV-insurance-old`. |
| 10 | `decision_date` | date | regex+rules | ISO date; from `vom DD. Monat YYYY` (DE) / `du DD MMMM YYYY` (FR) / `del DD MMMM YYYY` (IT). |
| 11 | `presiding_judge` | string | regex+rules | Extracted from closing boilerplate (`Der Präsident: X`, `Le Président : X`, `Il Presidente: X`). |
| 12 | `bvger_docket` | string | regex | **NEW v2.** Federal Administrative Court docket (`A-NNNN/YYYY`, `B-NNNN/YYYY`, `C-NNNN/YYYY`, `D-NNNN/YYYY`, `E-NNNN/YYYY`, `F-NNNN/YYYY`). Regex: `\b[A-F]-\d{3,5}/\d{4}\b`. |
| 13 | `bstger_docket` | string | regex | **NEW v2.** Federal Criminal Court docket (`RR.YYYY.NNN`, `BB.YYYY.NNN`, `SK.YYYY.NNN`, `SN.YYYY.NNN`, `BG.YYYY.NNN`, `BV.YYYY.NNN`, `BP.YYYY.NNN`). Regex: `\b(?:RR|BB|SK|SN|BG|BV|BP)\.\d{4}\.\d{1,4}\b`. |
| 14 | `bpatger_docket` | string | regex | **NEW v2.** Federal Patent Court docket (`O2014_001`, `S2018_005`). Regex: `\b[OS]\d{4}_\d{3}\b`. |
| 15 | `publication_status` | enum | regex+rules | **NEW v2.** `bge_published` / `unpublished` / `zur_publikation_vorgesehen` / `destine_a_la_publication` / `destinato_alla_pubblicazione`. From boilerplate phrases or presence of `BGE NNN/NN/NNN` ID. |

### B. Language & cross-lingual (5 fields, unchanged)

| # | Field | Type | Method | Description |
|---|---|---|---|---|
| 16 | `language` | enum | regex+rules | `de` / `fr` / `it`. Detected from opener regex on first 200 chars plus charset fallback. |
| 17 | `embedded_languages` | list[enum] | regex+rules | Other languages observed in body (DE prose with FR doctrinal-monograph titles; IT with DE etc.). |
| 18 | `has_trilingual_gloss` | bool | regex+rules | True when DE+FR+IT synonyms appear inline (e.g. `attestazione di soggiorno; Aufenthaltsausweis; attestation de séjour`). |
| 19 | `language_version_equality_invoked` | bool | regex | True if paragraph invokes equal-validity rule (`"die in gleicher Weise verbindlich sind"`, `"font également foi"`, `"hanno parità di rango"`). |
| 20 | `quoted_foreign_statute_languages` | list[enum] | regex+rules | Languages of verbatim foreign statute quotes (e.g. IT BGer quoting `D.L. 22 gennaio 2004 n. 42` from Italian state law). |

### C. Paragraph structure & identity (8 fields, unchanged)

| # | Field | Type | Method | Description |
|---|---|---|---|---|
| 21 | `erwagung_id_raw` | string | trivial | Raw E-tag from `citation` column (may be corrupt). |
| 22 | `erwagung_id_validated` | string | regex+rules | Cleaned canonical id (`E. 4.2`, `consid. 4.2`, `E. A.b`, etc.). Validated against body text. |
| 23 | `erwagung_id_artifact_class` | enum | regex+rules | `clean` / `year_token` / `percentage` / `speed_unit` / `length_unit` / `clock_time` / `energy_unit` / `cite_fragment` / `letter_suffix` / `language_suffix` / `nr_fragment` / `empty` / `header_only`. See §2.1. |
| 24 | `parent_section` | string | derived | Parent E-id (e.g. `E. 4` for `E. 4.2`). |
| 25 | `depth` | int | derived | E-nesting depth (1 for `E. 4`, 2 for `E. 4.2`, etc.). Up to 5 observed. |
| 26 | `sachverhalt_letter` | string | regex | If paragraph is in Sachverhalt block: `A`, `B`, `A.a`, `B.b`, etc. |
| 27 | `is_continuation_fragment` | bool | regex+rules | True if text begins mid-sentence. See §2.2. |
| 28 | `is_subsection_header_only` | bool | regex+rules | True if text is just `"3.5."` etc. |

### D. Paragraph role (15 fields, +3 vs v1)

| # | Field | Type | Method | Description |
|---|---|---|---|---|
| 29 | `paragraph_role` | enum | llm_enrich | One of: `legal_standard`, `reasoning`, `application_in_concreto`, `synthesis`, `facts`, `procedural_history`, `party_position`, `lower_court_summary`, `admissibility`, `dispositif`, `costs`, `signature`, `mitteilung_recipients`, `editor_summary`, `bridge`, `header_only`, `continuation_fragment`, `quoted_statute_block`, `parliamentary_materials`. **Multi-class classifier — primary llm_enrich target.** |
| 30 | `is_legal_standard` | bool | derived | True if `template_score ≥ 5` AND `opener_verb_owner == court`. |
| 31 | `is_party_position` | bool | llm_enrich | First-verb subject is the appellant (`Der Beschwerdeführer rügt`, `Le recourant fait valoir`, `Il ricorrente sostiene`). |
| 32 | `is_lower_court_summary` | bool | llm_enrich | First-verb subject is `Die Vorinstanz hat erwogen`, `La cour cantonale a retenu`, `L'autorità precedente non ha ritenuto`. |
| 33 | `is_facts` | bool | regex+rules | Within Sachverhalt block (letter-labeled). |
| 34 | `is_admissibility_recital` | bool | regex+rules | Body matches admissibility templates (`Auf die Beschwerde wird (nicht) eingetreten`, `Le recours est (ir)recevable`, `Il ricorso è (in)ammissibile`). |
| 35 | `is_dispositif` | bool | regex+rules | Body matches dispositive verbs (`Die Beschwerde wird abgewiesen/gutgeheissen`, `Le recours est rejeté/admis`, `Il ricorso è respinto/accolto`). |
| 36 | `is_cost_dispositif` | bool | regex+rules | Body matches cost templates (`Gerichtskosten`, `frais judiciaires`, `spese giudiziarie`). |
| 37 | `is_signature_boilerplate` | bool | regex+rules | Body matches `Im Namen der … Abteilung des Schweizerischen Bundesgerichts` / `Au nom de la … Cour du Tribunal fédéral suisse` / `In nome della … Corte del Tribunale federale svizzero`. **Deterministic end-of-decision marker.** See §2.3. |
| 38 | `is_moot_determination` | bool | regex | Body says "we leave the question open" (`kann offenbleiben`, `peut demeurer indécise`, `può rimanere indeciso il quesito`). |
| 39 | `is_closing_subsumption` | bool | regex | Body opens with closing-connective (`Demnach`, `Im Ergebnis`, `Zusammenfassend`, `En définitive`, `Il s'ensuit`, `Ne segue che`). |
| 40 | `is_editor_summary` | bool | regex | Body is `[Zusammenfassung: …]` / `[Résumé: …]` editorial block inserted by BGE publication office. |
| 41 | `cost_assignment_type` | enum | regex+rules | **NEW v2.** `gerichtskosten` / `parteientschadigung` / `unentgeltliche_rechtspflege` / `frais_judiciaires` / `depens` / `assistance_judiciaire` / `spese_giudiziarie` / `ripetibili` / `assistenza_giudiziaria` / `none`. |
| 42 | `vorinstanz_court_type` | enum | regex+rules | **NEW v2.** Lower-court type extracted from body: `verwaltungsgericht` / `bezirksgericht` / `obergericht` / `kantonsgericht` / `strafgericht` / `kantonales_versicherungsgericht` / `bvger` / `bstger` / `bpatger` / `weko` / `eba` / `eskcom_skbo` / `tribunal_cantonal` / `tribunale_amministrativo_cantonale` / `none`. |
| 43 | `vorinstanz_canton` | string | regex+rules | **NEW v2.** Canton code (ZH, BE, VD, etc.) of the lower court, extracted from `(Verwaltungsgericht|Obergericht|Kantonsgericht|Bezirksgericht|Cour cantonale|Tribunale d'appello) (des Kantons|du canton de|del Cantone) X`. |

### E. 7-element doctrinal-rule-statement template (9 fields, unchanged)

| # | Field | Type | Method | Description |
|---|---|---|---|---|
| 44 | `has_element_1_statutory_anchor` | bool | regex+rules | First 200 chars contain explicit `Art. N (Abs. M) (lit. x) CODE` / `art. N (al. M) (let. x) CODE` / `art. N (cpv. M) (lett. x) CODE`. Opener may be `Gemäss/Nach/Im Sinne von` (DE), `Aux termes de / Selon / Conformément à / En vertu de / D'après / À teneur de` (FR), `Giusta / Ai sensi di / Secondo / Conformemente all'` (IT). |
| 45 | `has_element_2_paraphrase` | bool | llm_enrich | Court restates the rule in its own voice using a court-subject verb. **Verb-ownership semantic check, not just keyword match — primary llm_enrich target.** |
| 46 | `has_element_3_teleology` | bool | regex | Purpose-statement marker: `soll verhindern, dass / bezweckt / dient` (DE), `vise à empêcher / a pour but de / tend à / afin de / aux fins de / destiné à / a pour vocation` (FR), `mira a / ha lo scopo di / risponde a / tende a` (IT). |
| 47 | `has_element_4_positive_limb` | bool | regex | Non-exhaustive factor list marker: `namentlich / insbesondere` (DE), `notamment` (FR), `in particolare / segnatamente` (IT). |
| 48 | `has_element_5_negative_limb` | bool | regex | Boundary-stopper: `genügt indessen nicht / vermag nicht zu` (DE), `ne saurait suffire / ne saurait à elle seule / il ne se justifie pas / ne ressort pas / manque de pertinence` (FR), `non è sufficiente / non si riferisce / occorre fare astrazione da / non si concilia` (IT). |
| 49 | `has_element_6_interpretive_factors` | bool | regex | Multi-factor balancing marker: `Bei der Frage, ob …, ist auch der Art, Bedeutung, Schwere … Rechnung zu tragen` (DE), `Dans cet examen, entrent en ligne de compte …` (FR), `Sotto il profilo di …, occorre tener conto di …` (IT). |
| 50 | `has_element_7_authority_chain` | bool | regex | Closing parenthesis with 2-6 ATF/BGE/DTF cites ending in `mit Hinweisen` / `et les références` / `con rinvii`. |
| 51 | `template_score` | int (0-7) | derived | Count of elements 44-50 present. ≥5 → high-confidence legal_standard. |
| 52 | `opener_verb_owner` | enum | llm_enrich | `court` / `appellant` / `lower_court` / `legislator` / `doctrine` / `foreign_authority`. **The verb-ownership discriminator — primary llm_enrich target.** |

### F. Statute references (12 fields, +2 vs v1)

| # | Field | Type | Method | Description |
|---|---|---|---|---|
| 53 | `statute_anchors` | list[dict] | regex+rules | List of `{sr_number, statute_short_de, statute_short_fr, statute_short_it, article, abs, lit, ziff, abs_form, lit_form, version_marker}`. Normalised to `sr_number`. |
| 54 | `lead_statute_anchors` | list[dict] | derived | Subset of #53 where anchor appears in first 200 chars. |
| 55 | `cantonal_code_anchors` | list[dict] | regex+rules | Cantonal-statute citations with cantonal-collection prefix (`BSG`, `LS`, `RSV`, `RSG`, etc. — see §3.6). |
| 56 | `i_v_m_chains` | list[list[dict]] | regex+rules | `Art. X i.V.m. Art. Y` / `art. X en lien avec art. Y` / `art. X in relazione con art. Y` chains. |
| 57 | `is_outdated_statute_version` | bool | regex | True if `aArt.` (DE) / `ancien art.` (FR) / `vecchio art.` (IT) prefix is used. |
| 58 | `pre_revision_version_marker` | string | regex | Verbatim version-qualifier (`in der bis Ende 2009 geltenden Fassung` / `dans sa teneur jusqu'au …` / `nella versione in vigore fino al …`). |
| 59 | `quotes_statute_verbatim` | bool | regex+rules | True if body contains literal verbatim quotation introduced by `lautet wie folgt:` / `a la teneur suivante:` / `ha il seguente tenore:`. |
| 60 | `quoted_statute_short` | string | regex+rules | Which statute is verbatim quoted (e.g. `Art. 40 EpG`, `art. 26 OLL 3`). |
| 61 | `bis_ter_quater_articles` | list[string] | regex | Articles with Latin ordinal suffix (`Art. 24a`, `Art. 3bis`, `art. 179novies CP`). Regex: `art\.\s*\d+\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?`. |
| 62 | `embedded_statute_paragraph_list` | bool | regex+rules | True if body contains numbered statute-text enumeration. |
| 63 | `cited_concordat` | list[string] | regex+rules | **NEW v2.** Inter-cantonal concordats cited: `Konkordat über die Rechtshilfe in Strafsachen`, `Konkordat über die Aktiengesellschaft` (rare modern), `IRSG-Konkordat`, `BeBoZG-Konkordat`, `Konkordat über Schiedsgerichtsbarkeit (pre-2011)`, `IVöB (Vergaberecht)`. FR `concordat sur l'entraide`. IT `concordato`. |
| 64 | `statute_pre_2007_form` | list[enum] | regex | **NEW v2.** Pre-2007 statute aliases observed: `OG`/`OJ` (pre-BGG), `PPF` (pre-StPO criminal-procedure), `BStP` (pre-2007 criminal), `ANAG`/`LSEE`/`LDDS` (pre-AuG immigration), `OBG` (pre-StBOG criminal-court org), `IRSG_v1` (pre-2003 IRSG), `MStG_v1`. |

### G. Citation graph & authority chain (24 fields, +9 vs v1)

| # | Field | Type | Method | Description |
|---|---|---|---|---|
| 65 | `cited_bge` | list[dict] | regex+rules | `{volume, series_roman, page, erwagung, page_pinpoint, language_form (BGE/ATF/DTF)}`. |
| 66 | `cited_urteil` | list[dict] | regex+rules | Unpublished BGer dockets cited (`1B_575/2021`, `Urteil M. vom 11. Dezember 2003, I 589/03`). |
| 67 | `cited_doctrine` | list[dict] | regex+rules | Doctrine references: `AUTHOR, Title, Yth ed. YYYY, N./Rz./n°/n. NNN ad/zu Art. M`. |
| 68 | `cited_commentary_series` | enum-list | regex | Major Swiss commentary series: `Basler Kommentar (BSK)`, `Berner Kommentar (BK)`, `Zürcher Kommentar (ZK)`, `Commentaire romand (CR)`, `St. Galler Kommentar`, `Praxiskommentar`. |
| 69 | `cited_periodical` | list[string] | regex | Journals (`ZBl`, `sic!`, `AJP`, `BVR`, `SVR`, `StR`, `StE`, `ASA`, `EuGRZ`, `Jusletter`, `RDAF`, `RDAT`, `RtiD`, `SJ`, `Pra`, `FamPra.ch`, `JdT`, `RSJ`, `RJB`, `RJN`, `Medialex`, `URP`, `RDS`). |
| 70 | `cited_treaty` | list[string] | regex | Treaties (`EMRK/CEDH/CEDU`, `UNO-Pakt II`, `FZA/ALCP`, `KRK/CDE`, `VRK/CV`, `SDÜ`, `Visakodex`, `IPRG/LDIP`, `CEAG/CEEJ/EUeR`, `Dublin-III-VO`). Carries `SR 0.NNN.NNN` for international law. |
| 71 | `cited_botschaft` | list[dict] | regex+rules | Federal Council messages: `{date, subject, bbl_year, bbl_page, multi_botschaft_label}`. |
| 72 | `cited_parliamentary_record` | list[dict] | regex+rules | `AB YYYY S/N NNN` (DE) / `BO YYYY CE/CN NNN` (FR) / `BU YYYY CS/CN NNN` (IT). |
| 73 | `cited_foreign_court` | list[dict] | regex+rules | Non-Swiss courts: `BVerfG`, `Supreme Court of the United States`, `EFTA Court`, `Corte di Appello`, `Cassazione`. |
| 74 | `cited_ecthr` | list[dict] | regex+rules | ECtHR judgments with case name, state, date, application no, paragraph pinpoint, recueil year. |
| 75 | `cited_cjeu` | list[dict] | regex+rules | CJEU judgments: `C-NNN/YYYY P`, AG opinions. |
| 76 | `cited_url` | list[string] | regex | Federal-admin URLs (`www.amtsblattportal.ch`, `www.bafu.admin.ch`). Sometimes with `besucht am DATE`. |
| 77 | `cited_gutachten` | list[dict] | regex+rules | Commissioned expert opinions. |
| 78 | `mit_hinweisen_tail_variant` | enum | regex | Variant of trailing connector (see Group H §3.3). |
| 79 | `code_switched_citation_blocks` | list[dict] | llm_enrich | Cite blocks where citing-paragraph language differs from cited-work language. Needs language-detection on individual citation blocks. |
| 80 | `cited_lugano_convention` | bool | regex+rules | **NEW v2.** `LugÜ` / `Convention de Lugano` / `Convenzione di Lugano` / `Lugano-Übereinkommen` with article-pinpoint. |
| 81 | `cited_hague_convention` | list[string] | regex+rules | **NEW v2.** `HKÜ` (Haager Kindesentführungsübereinkommen) / `Convention de La Haye` / `Convenzione dell'Aia` + which HCC instrument number (HCC 1980 child abduction, HCC 1996 child protection, HCC 1961 wills, HCC 2007 maintenance, etc.). |
| 82 | `cited_eu_directive` | list[dict] | regex+rules | **NEW v2.** EU directive citations: `Richtlinie YYYY/NN/EU` / `directive YYYY/NN/UE` / `direttiva YYYY/NN/UE` with optional CELEX number. Relevant for FZA/ALCP-driven autonomous-implementation cases. |
| 83 | `cited_eu_regulation` | list[dict] | regex+rules | **NEW v2.** `Verordnung (EU) Nr. NNN/YYYY` / `règlement (UE) n° NNN/YYYY` / `regolamento (UE) n. NNN/YYYY`. |
| 84 | `cited_oecd_mtc` | bool | regex+rules | **NEW v2.** `OECD-MA` / `MC OCDE` / `Modello OCSE` (OECD Model Tax Convention) with optional article. Tax-treaty-interpretation cases. |
| 85 | `cited_un_convention` | list[string] | regex+rules | **NEW v2.** UN conventions outside ECHR/UNO-Pakt: `Wiener Übereinkommen über das Recht der Verträge (VRK)`, `UN-Kaufrechtsübereinkommen (CISG)`, `UN-Behindertenrechtskonvention (BRK / CRPD)`, `UN-Antifolterkonvention (CAT)`, `UN-Frauenrechtskonvention (CEDAW)`, `UN-Rassendiskriminierungskonvention (CERD/ICERD)`, `UN-Korruptionskonvention (UNCAC)`. |
| 86 | `cited_bvger_decision` | list[string] | regex+rules | **NEW v2.** BVGer dockets cited (`Urteil des BVGer A-1234/2018`, `arrêt du TAF B-5678/2019`). Different from #12 (decision-as-source) — these are cited authority. |
| 87 | `cited_bstger_decision` | list[string] | regex+rules | **NEW v2.** BStGer dockets cited (`RR.2020.123`, `BB.2019.45`). |
| 88 | `cited_bpatger_decision` | list[string] | regex+rules | **NEW v2.** BPatGer dockets cited (`O2014_001`). Patent-law cases. |
| 89 | `cited_eparl_kommissionsbericht` | list[dict] | regex+rules | **NEW v2.** Parliamentary committee reports: `Bericht der Kommission für Rechtsfragen`, `Rapport de la commission des affaires juridiques`. Often with parlament.ch URLs or BBl/FF references. |
| 90 | `cited_vernehmlassung` | list[string] | regex+rules | **NEW v2.** Consultation materials: `Erläuternder Bericht zur Vernehmlassung`, `Rapport explicatif de la consultation`, `Rapporto esplicativo`. |
| 91 | `author_surnames_caps` | list[string] | regex | **NEW v2.** All ALL-CAPS surnames in citation context (`HONSELL`, `JAGGI/GAUCH`, `STREIFF/VON KAENEL`, `AUER/MALINVERNI/HOTTELIER`, `MEIER-HAYOZ`). Convention used across DE/FR/IT BGer for doctrinal citations. Regex: `\b[A-ZÄÖÜ]{4,}(?:[-/](?:VON\s+)?[A-ZÄÖÜ]{2,})*\b` filtered against statute-short-name dictionary. |

### H. Cross-lingual alias hits (7 fields, +1 vs v1)

| # | Field | Type | Method | Description |
|---|---|---|---|---|
| 92 | `statute_alias_triples_used` | list[enum] | regex | Which canonical triples appear in text. See §3.4. |
| 93 | `pinpoint_form_triples_used` | list[enum] | regex | Which pinpoint-abbreviation triples appear. See §3.2. |
| 94 | `older_de_alias_used` | list[enum] | regex | Older DE shorthand: `MRK` (= EMRK), `aBV` (= old BV pre-1999), `aArt.`, `vBV`, `vLGC`. |
| 95 | `older_docket_format_used` | bool | regex | True if pre-2007 docket format. |
| 96 | `swiss_it_vs_foreign_it` | enum | regex+rules | For IT rows only: `swiss_it` (uses `cpv.`, `RS`, `Ministero pubblico`) / `foreign_state_italian` (uses `comma`, `D.L.`, `Procura della Repubblica`) / `mixed`. |
| 97 | `loccit_or_aao` | enum | regex | Back-reference shorthand: `loc_cit_it`, `loccit_fr`, `aao_de`, `precite_fr`, `ibid`. |
| 98 | `eu_law_alias_triples_used` | list[enum] | regex | **NEW v2.** EU-law cross-lingual aliases observed: `Richtlinie_directive_direttiva`, `Verordnung_règlement_regolamento`, `Erwägungsgrund_considérant_considerando` (preamble recital), `Mitgliedstaat_État_membre_Stato_membro`, `EuGH_CJUE_CGUE`, `Schlussantraege_conclusions_AG_conclusioni_AG`. |

### I. Doctrinal-test invocation (8 fields, unchanged)

| # | Field | Type | Method | Description |
|---|---|---|---|---|
| 99 | `named_doctrine` | list[enum] | regex+rules | Named Swiss-legal doctrines: `Schubert-Praxis`, `PKK-Praxis`, `AVLOCA-Praxis`, `Reneja-Praxis`, `Star-Praxis`, `Boultif-Kriterien`, `Üner-Kriterien`, `Maslov-Kriterien`, `Kissiwa-Koffi-Kriterien`, `Engel-Kriterien`, `Bedarfsmarktkonzept`, `SSNIP-Test`, `Methodenpluralismus`, `Lex-posterior-Regel`, `Genehmigungsfiktion`, `VgT`, `Ostendorf`, `Bacchini`, `Zweijahresregel`. Curated lookup + `sog./so genannt/dit/cosiddetto` marker. |
| 100 | `doctrinal_test_invoked` | list[enum] | llm_enrich | Standard tests: `art_36_BV_three_prong`, `verhaeltnismaessigkeit_three_prong`, `arbitraire_standard`, `willkuer_standard`, `arbitrio_standard`, `anklagegrundsatz`, `rechtliches_gehoer`, `rechtsweggarantie`, `unbestimmte_Begriffe_test`, `doppia_punibilita`, `vorrang_bundesrecht`. **Needs semantic match — paragraph may invoke the test without using its label.** |
| 101 | `saving_construction_invoked` | bool | regex | True if `verfassungskonforme Auslegung/Anwendung` / `interprétation conforme à la constitution` / `interpretazione conforme` is invoked. |
| 102 | `review_intensity` | enum | regex+rules | `freie_kognition` / `willkuer_pruefung` / `Angemessenheitskontrolle` / `vollkognition` / `plein_pouvoir_d_examen` / `pouvoir_d_examen_limite` / `pieno_potere_d_esame` / `potere_d_esame_limitato`. |
| 103 | `holding_polarity` | enum | regex+rules | `holds_up` (`hält stand`, `tient`, `tiene`) / `fails` / `none`. |
| 104 | `is_remand` | bool | regex | `Rückweisung an die Vorinstanz` / `renvoi à l'autorité précédente` / `rinvio all'autorità precedente`. |
| 105 | `art_106_qualified_complaint_required` | bool | regex | Body invokes Art. 106 Abs. 2 BGG (qualifizierte Rügepflicht / exigence de motivation accrue / motivazione qualificata). |
| 106 | `is_obiter_dictum` | bool | regex | Body opens with obiter marker: `Im Übrigen / überdies`, `À titre superfétatoire / Au demeurant / Du reste`, `A titolo abbondanziale / Di transenna / Per giunta / Del resto`. |

### J. Idioms & formulae (10 fields, unchanged)

| # | Field | Type | Method | Description |
|---|---|---|---|---|
| 107 | `formula_present` | list[enum] | regex | Canonical formulae catalogue (~50 entries: `vor_bundesrecht_standhalten`, `praktische_konkordanz`, `regle_aptitude_necessite_proportionnalite_stricto_sensu`, `lacune_proprement_dite_improprement_dite_occulte`, `concubinage_qualifie_stable_durable`, `peut_demeurer_indecise`, `controllo_astratto`, `controllo_concreto`, `controllo_accessorio`, etc.). |
| 108 | `latin_maxim_present` | list[string] | regex | `in dubio pro reo`, `e contrario`, `prima vista`, `reformatio in peius`, `dies a quo`, `i.V.m./cum`, `i.f./in fine`, `a.a.O./loc cit./op cit.`, `Nulla poena sine lege`, `iura novit curia`, `mutatis mutandis`, `a fortiori`, `pacta sunt servanda`, `ratio legis`, `iudex a quo/ad quem`. |
| 109 | `swiss_idiom_present` | list[string] | regex | Unique Swiss-legal idioms: `Doppelnennung`, `Bund-Kanton-Spannung`, `Schwerlast`, `pêle-mêle et sommaire`, `on peine à les suivre`, `alla berlina`, `naming and shaming` (IT), `a torto / a ragione`, `Quintessenz / Quintessenza`, `besucht am`. |
| 110 | `swiss_metaphor_present` | list[string] | regex | Picturesque BGer metaphors: `grobmaschiges_netz`, `Stadtmauer_ueber_der_Schlucht`, `Sprengung_des_Zonencharakters`, `Riegelwirkung`, `Mailaender_Edikt`, `Spannungsverhältnis`. |
| 111 | `rebuke_phrase_present` | list[enum] | regex | Court rebuke openers: `manque_de_pertinence`, `peine_a_suivre`, `quoi_qu_en_dise`, `pele_mele_sommaire`, `ne_saurait_etre_suivie`, `A_torto_period`, `mal_fonde_le_grief_doit_etre_ecarte`, `inammissibilmente`. |
| 112 | `synthesis_marker_present` | bool | regex | `Als Quintessenz ergibt sich` / `Insgesamt zeigt sich demnach` / `En définitive` / `Il s'ensuit` / `In definitiva` / `Quintessenza`. |
| 113 | `bridge_phrase_present` | bool | regex+rules | Single-sentence section-announcement (`Zu prüfen ist weiter, ob …`, `Reste à examiner …`, `Resta da determinare …`). |
| 114 | `information_principles_invoked` | list[enum] | regex | If any of `Sachlichkeit / Transparenz / Verhältnismässigkeit / Vollständigkeit` (Art. 10a BPR voting-information principles). |
| 115 | `evidentiary_standard_invoked` | enum | regex | `freie_beweiswuerdigung` / `ueberwiegende_wahrscheinlichkeit` / `vraisemblance_preponderante` / `verosimiglianza_preponderante` / `strict_beweis` / `mit_an_sicherheit_grenzender_wahrscheinlichkeit` / `glaubhaft_machen` / `rendre_vraisemblable` / `rendere_verosimile`. |
| 116 | `iura_novit_curia_invoked` | bool | regex | True if explicitly invoked or denied (`das Bundesgericht prüft das Recht von Amtes wegen`, `le Tribunal fédéral applique le droit d'office`, `il Tribunale federale applica d'ufficio il diritto`). |

### K. Quality / cleanup flags (10 fields, +2 vs v1)

| # | Field | Type | Method | Description |
|---|---|---|---|---|
| 117 | `is_csv_parse_artifact` | bool | derived | True if `erwagung_id_artifact_class != clean`. |
| 118 | `has_encoding_artifacts` | bool | regex | Per-case encoding bug detected (umlauts → `?`). |
| 119 | `embedded_page_anchors` | list[string] | regex | Mid-text page-break markers (`BGE 137 IV 122 S. 128`, `ATF 137 IV 122 p. 128`, `DTF 137 IV 122 pag. 128`). |
| 120 | `internal_navigation_refs` | list[string] | regex | `vgl. E. X hiervor/hiernach`, `cf. consid. X ci-dessus/ci-dessous`, `cfr. supra/infra consid. X`, `siehe oben/unten`, `vorne/hinten`. |
| 121 | `has_quoted_block` | bool | regex+rules | Body contains `"…"` block of length ≥ 50 chars. |
| 122 | `has_embedded_list` | bool | regex+rules | Body contains numbered/lettered list. |
| 123 | `text_length_chars` | int | trivial | Length of body text. Range observed: 4 to ~7,500 chars. Median ~800-1000 chars. |
| 124 | `materialien_density` | enum | llm_enrich | `none` / `low` / `medium` / `high`. High = parliamentary-debate-heavy / Botschaft-heavy / committee-report-heavy. |
| 125 | `party_anonymization_pattern` | list[string] | regex | **NEW v2.** Party labels: `X._`, `A._`, `B._`, `B.A._`, `A.B._`, `Y.S._`, `X.Y._`. Regex: `\b[A-Z](?:\.[A-Z])?\._\b`. Used to cluster decisions of same case-style. |
| 126 | `has_anonymized_party` | bool | regex | **NEW v2.** True if any `party_anonymization_pattern` present. |

### L. Procedural metadata (5 fields, NEW group in v2)

| # | Field | Type | Method | Description |
|---|---|---|---|---|
| 127 | `procedure_type` | enum | regex+rules | **NEW v2.** Type of BGer proceeding: `beschwerde_oerl_angelegenheiten` (Art. 82 BGG) / `beschwerde_zivilsachen` (Art. 72 BGG) / `beschwerde_strafsachen` (Art. 78 BGG) / `subsidiäre_verfassungsbeschwerde` (Art. 113 BGG) / `revision` / `erlaeuterung` / `klage_direkt` (Art. 120 BGG) / `unbekannt`. |
| 128 | `procedural_history_depth` | enum | llm_enrich | **NEW v2.** Number of prior instances cited in procedural history: `single_instance`, `appeal_from_cantonal`, `appeal_from_federal_bvger`, `appeal_from_cantonal_two_tier`, `revision`, `unknown`. |
| 129 | `has_oral_hearing_marker` | bool | regex | **NEW v2.** True if `öffentliche Beratung`, `audience publique`, `udienza pubblica`. |
| 130 | `cited_self_decision_paragraph` | list[string] | regex+rules | **NEW v2.** Self-cited Erwägung from same decision (`vgl. E. 4.2 hiernach/hiervor`, `cf. consid. 5.1 ci-dessus`). Helpful for stitching chunked-paragraph context. |
| 131 | `cantonal_court_appeal_pathway` | enum | regex+rules | **NEW v2.** Appeal pathway via cantonal courts: `cantonal_to_bger_direct` / `cantonal_one_tier_then_bger` / `cantonal_two_tier_then_bger` / `none`. Derived from procedural-history phrases. |

### M. Composite / derived (1 field)

| # | Field | Type | Method | Description |
|---|---|---|---|---|
| 132 | `gold_candidate_score` | float | judge_time | 0-1 composite: high if `template_score ≥ 5` AND `paragraph_role ∈ {legal_standard, reasoning, application_in_concreto}` AND `chamber_area matches query area` AND `is_csv_parse_artifact = false` AND statute_anchors intersect query-targeted statutes. **Judge-time only — depends on the query.** |

---

## 2. Validation rules (unchanged from v1)

### 2.1 Citation-noise detector

A row's `erwagung_id_raw` is `clean` iff ALL of:
- Matches regex `^[A-Z](?:\.[a-z])?$` (letter-Sachverhalt) OR `^\d+(?:\.\d+){0,4}[a-z]?$` (decimal-Erwägung, up to 5 levels).
- First segment ≤ 15.
- Does not match year-shape `^\d{4}$`.
- Does not match unit-suffix shape `\d+(m|ha|m\d|°|%)`.
- Does not contain language-tag suffix (`it`, `de`, `fr`).
- Sequence-coherent with neighbour rows in same case.

**Tested rate**: ~22% noise on environmental-science-heavy BGE rows; ~16% on early-corpus published-BGE; ~0-2% on unpublished-BGer.

### 2.2 Continuation-fragment detector

A row is a continuation fragment iff ANY of:
- Body opens lowercase.
- Body opens with closing bracket `)` or `]`.
- Body opens with `%`, `m`, `ha`, `Uhr`, `GWh`, year-followed-by-comma.
- Body opens with `Nr.`, `Jahrhundert`.
- Body opens with `[Date].`.

### 2.3 Decision-closure detector

Row is `is_signature_boilerplate=true` iff body contains all of:
- City+date pattern: `(Lausanne|Luzern|Lucerne|Losanna), \d{1,2}\.?\s+(Januar|…|Dezember|janvier|…|décembre|gennaio|…|dicembre)\s+\d{4}`
- Chamber-self-naming: `Im Namen der … Abteilung` / `Au nom de la … Cour` / `In nome della … Corte`
- President+clerk pattern.

This is the **deterministic end-of-decision marker** for chunking.

### 2.4 Bridge-paragraph / header-only filter

Filter out rows where:
- `text_length_chars < 20` AND body matches `^\d+(\.\d+)*\.?$`.
- `text_length_chars < 10` (heuristic floor for any embedding).

### 2.5 Verb-ownership classifier (llm_enrich)

This is the v1 §2.5 rule promoted to an llm_enrich field (`opener_verb_owner`). Static regex achieves ~80% accuracy; LLM classifier needed for the residual 20% (mixed openers, indirect speech).

Apply in order on the opening sentence:
1. If subject is statute (`Art. N CODE`) AND verb in doctrinal-verb-pool → `court`.
2. If subject is `Der Haftgrund / Der Anspruch / Die Beschwerde / Le recours / Il ricorso / La giurisprudenza` AND verb in doctrinal-verb-pool → `court`.
3. If subject is `Der Beschwerdeführer / Le recourant / Il ricorrente / Die Beschwerdegegnerin` AND verb in argumentative-pool → `appellant`.
4. If subject is `Die Vorinstanz / La cour cantonale / L'autorità precedente` → `lower_court`.
5. Else → unclear (defer to LLM).

---

## 3. Cross-lingual canonical triples (extended in v2)

### 3.1 Reporter / decision-type triples

| DE | FR | IT | EN |
|---|---|---|---|
| BGE | ATF | DTF | published BGer judgment |
| Urteil | arrêt | sentenza | unpublished BGer judgment |
| BVGer | TAF | TAF | Federal Administrative Court |
| BStGer | TPF | TPF | Federal Criminal Court |
| BPatG / BPatGer | TFB | TFB | Federal Patent Court |
| EVG | TFA | TFA | Federal Insurance Court (pre-2007) |
| EGMR | CourEDH | CorteEDU | European Court of Human Rights |
| EuGH | CJUE | CGUE | Court of Justice of the EU |
| BVerfG | BVerfG | BVerfG | German Federal Constitutional Court |

### 3.2 Pinpoint / navigation triples

| DE | FR | IT |
|---|---|---|
| E. (Erwägung) | consid. (considérant) | consid. (considerando) |
| Abs. (Absatz) | al. (alinéa) | cpv. (capoverso) |
| lit. (litera) | let. (lettre) | lett. (lettera) |
| Ziff. (Ziffer) | ch. (chiffre) | n. (numero) |
| Anh. (Anhang) | annexe | all. (allegato) |
| S. (Seite) | p. (page) | pag. (pagina) |
| f. (folgend) | s. (suivante) | seg. (seguente) |
| ff. (folgende) | ss. (suivantes) | segg. (seguenti) |
| Rz. / N. | n° | n. |
| Aufl. (Auflage) | éd. (édition) | ed. (edizione) |
| zu Art. | ad art. | ad art. |
| § | § | § |

### 3.3 Connector / discourse triples

| DE | FR | IT |
|---|---|---|
| mit Hinweisen | et les références | con rinvii / con riferimenti |
| je mit Hinweisen | (collective) | e rispettivi rinvii |
| und die dort zitierte Rechtsprechung | et la jurisprudence citée | e la giurisprudenza citata |
| vgl. (vergleiche) | cf. (confer) | cfr. (confronta) |
| siehe | voir | vedasi / v. |
| a.a.O. (am angegebenen Ort) | op. cit. / précité | loc. cit. / op. cit. |
| i.V.m. (in Verbindung mit) | en lien avec | cum / in relazione con |
| im Übrigen | au demeurant / du reste | del resto / per giunta |
| im vorliegenden Fall | en l'espèce / en l'occurrence | nella fattispecie / in concreto |
| zu Recht | à juste titre / à raison | a ragione / rettamente |
| zu Unrecht | à tort | a torto |
| nach Auffassung des BGer | de l'avis du TF | a mente di questa Corte / L'Alta Corte ha ritenuto |
| Unbestritten ist, dass | il est constant que | è pacifico che |
| aArt. N | ancien art. N | vecchio art. N |
| Demnach / Im Ergebnis / Zusammenfassend | Dès lors / Partant / Il s'ensuit / En définitive | Ne segue che / In definitiva / Quindi |
| vom | du | del |
| gegen | contre | contro |
| Gerichtsschreiber | greffier | cancelliere |
| Präsident | Président | Presidente |

### 3.4 Constitutional / Federal statute triples

| DE | FR | IT |
|---|---|---|
| BV | Cst. | Cost. |
| BGG | LTF | LTF |
| ZGB | CC | CC |
| OR | CO | CO |
| StGB | CP | CP |
| StPO | CPP | CPP |
| ZPO | CPC | CPC |
| EMRK | CEDH | CEDU |
| ATSG | LPGA | LPGA |
| AHVG | LAVS | LAVS |
| IVG | LAI | LAI |
| KVG | LAMal | LAMal / LAMI |
| UVG | LAA | LAINF |
| BVG | LPP | LPP |
| StHG | LHID | LAID |
| DBG | LIFD | LIFD |
| BGBM | LMI | LMI |
| AuG / AIG (post-2019) | LEtr / LEI (post-2019) | LStr |
| AsylG | LAsi | LAsi |
| EnG | LEne | LEne |
| RPG | LAT | LPT |
| USG | LPE | LPAmb |
| TSchG | LPA | LPAn |
| ArG | LTr | LL |
| ArGV 3 | OLT 3 | OLL 3 |
| KG | LCart | LCart |
| WEKO | COMCO | COMCO |
| DSG | LPD | LPD |
| Kulturgütertransfergesetz | LTBC | LTBC |
| IRSG | EIMP | AIMP |
| IRSV | OEIMP | OAIMP |
| EUeR | CEEJ | CEAG |
| SVG | LCR | LCStr |
| VZV | OAC | OAC |
| StBOG | LOAP | LOAP |
| PatGG | LTFB | LTFB |
| RVOG | LOGA | LOGA |
| VwVG | PA | PA |
| VGG | LTAF | LTAF |
| RTVG | LRTV | LRTV |
| MedBG | LPMéd | LPMed |
| BGFA / BGAv | LLCA | LLCA |
| FINMAG | LFINMA | LFINMA |
| FZA | ALCP | ALC |
| KRK | CDE | CRC |
| VRK | CV | CV |
| EpG | LEp | LEp |
| EpV | OEp | OEp |
| NHG | LPN | LPN |
| BGÖ | LTrans | LTras |
| BGA | LAr | LAr |
| RHG | LHR | LArRA |
| BÜPF | LSCPT | LSCPT |
| OG (pre-2007) | OJ (pre-2007) | OG (pre-2007) |
| PPF (pre-2007) | PPF (pre-2007) | PPF (pre-2007) |
| ANAG (pre-2008) | LSEE (pre-2008) | LDDS (pre-2008) |
| **NEW v2: BÜPF | LSCPT | LSCPT** (telecom-surveillance) |
| **NEW v2: GwG | LBA | LRD** (anti-money-laundering) |
| **NEW v2: FinfraG | LIMF | LInFi** (financial-market-infrastructure) |
| **NEW v2: FIDLEG | LSFin | LSerFi** (financial-services) |
| **NEW v2: FINIG | LEFin | LIsFi** (financial-institutions) |
| **NEW v2: KAG | LPCC | LICol** (collective-investment) |
| **NEW v2: VAG | LSA | LSA** (insurance-supervision) |
| **NEW v2: BankG | LB | LBCR** (banks) |

### 3.5 Publication / collection triples

| DE | FR | IT |
|---|---|---|
| BBl (Bundesblatt) | FF (Feuille fédérale) | FF (Foglio federale) |
| AS (Amtliche Sammlung) | RO (Recueil officiel) | RU (Raccolta ufficiale) |
| SR (Systematische Sammlung) | RS (Recueil systématique) | RS (Raccolta sistematica) |
| AB (Amtliches Bulletin) | BO (Bulletin officiel) | BU (Bollettino ufficiale) |
| BS (Bereinigte Sammlung) | (pre-1948 collections vary) | (varies) |

### 3.6 Cantonal collection prefixes

| Canton | Prefix | Numbering form |
|---|---|---|
| BE | BSG | dotted decimal (`BSG 153.01`) |
| ZH | LS | dotted decimal (`LS 175.2`) |
| VD | RSV / BLV | dotted decimal |
| GE | RSG / rs/GE | letter+number+slash (`RSG E 5/10`) |
| FR | RSF / SGF | dotted decimal |
| NE | RSN | dotted decimal |
| TI | RL / RL/TI | dotted hierarchical |
| SG | sGS | dotted decimal |
| GR | BR | dotted decimal |
| TG | RB | dotted decimal |
| SZ | SRSZ | dotted decimal |
| LU | SRL | dotted decimal |
| BS | SG | dotted decimal |
| BL | SGS | dotted decimal |
| OW | GDB | dotted decimal |
| NW | NG | dotted decimal |
| GL | GS | dotted decimal |
| ZG | BGS | dotted decimal |
| SO | BGS | dotted decimal |
| SH | SHR | dotted decimal |
| AR | bGS | dotted decimal |
| AI | GS | dotted decimal |
| JU | RSJU | dotted decimal |
| VS | SGS / SGSV | dotted decimal |

### 3.7 Cantonal-court docket prefixes (extended in v2)

| Canton/Court | Prefix |
|---|---|
| VD Cour de droit administratif et public | `BO.YYYY.NNNN`, `PS.YYYY.NNNN` |
| GE Cour de justice administrative | `ATA/NNN/YYYY` |
| ZH Verwaltungsgericht | `VB.YYYY.NNNNN` |
| ZH Steuerrekursgericht | `1 ST.YYYY.NN` |
| BE Verwaltungsgericht | `100.YYYY.NN`, `200.YYYY.NN` |
| TI Tribunale amministrativo | `52.YYYY.NN` |
| TI Tribunale d'appello | `60.YYYY.NN` |
| BVGer | `[A-F]-NNNN/YYYY` |
| BStGer | `RR/BB/SK/SN/BG/BV/BP.YYYY.NNN` |
| BPatGer | `O2YYY_NNN`, `S2YYY_NNN` |

### 3.8 International-law alias triples (NEW in v2)

| EN canonical | DE | FR | IT |
|---|---|---|---|
| Lugano Convention | Lugano-Übereinkommen / LugÜ | Convention de Lugano / CL | Convenzione di Lugano / CLug |
| HCC 1980 (Child Abduction) | Haager Kindesentführungsübereinkommen / HKÜ | Convention de La Haye de 1980 | Convenzione dell'Aia 1980 |
| HCC 1996 (Child Protection) | Haager Kinderschutzübereinkommen / HKsÜ | Convention de La Haye de 1996 | Convenzione dell'Aia 1996 |
| HCC 2007 (Maintenance) | Haager Unterhaltsübereinkommen | Convention de La Haye sur les obligations alimentaires | Convenzione dell'Aia sugli obblighi alimentari |
| New York Convention 1958 (Arbitration) | New Yorker Übereinkommen | Convention de New York | Convenzione di New York |
| Wiener Übereinkommen (Vertragsrecht) | VRK / Wiener Übereinkommen | Convention de Vienne / CV | Convenzione di Vienna |
| Vienna Convention on Consular Relations | Wiener Übereinkommen über konsularische Beziehungen | Convention de Vienne sur les relations consulaires | Convenzione di Vienna sulle relazioni consolari |
| UN Sales Convention | UN-Kaufrechtsübereinkommen / CISG / WKR | Convention de Vienne sur les ventes / CVIM | Convenzione di Vienna sulla vendita / CVIM |
| UN Convention on the Rights of Persons with Disabilities | UN-Behindertenrechtskonvention / BRK | Convention relative aux droits des personnes handicapées / CDPH | Convenzione sui diritti delle persone con disabilità / CDPD |
| UN Convention against Torture | UN-Antifolterkonvention / CAT | Convention contre la torture | Convenzione contro la tortura |
| UN Convention on the Elimination of Discrimination Against Women | UN-Frauenrechtskonvention / CEDAW | Convention sur l'élimination des discriminations à l'égard des femmes / CEDEF | Convenzione sull'eliminazione delle discriminazioni contro le donne |
| UN Convention Against Corruption | UN-Korruptionskonvention / UNCAC | Convention contre la corruption / CNUCC | Convenzione contro la corruzione |
| Council of Europe Cybercrime Convention | Budapester Cybercrime-Übereinkommen | Convention de Budapest sur la cybercriminalité | Convenzione di Budapest sulla criminalità informatica |
| OECD Model Tax Convention | OECD-MA | MC OCDE | Modello OCSE |
| EU Directive | Richtlinie YYYY/NN/EU | directive YYYY/NN/UE | direttiva YYYY/NN/UE |
| EU Regulation | Verordnung (EU) Nr. NNN/YYYY | règlement (UE) n° NNN/YYYY | regolamento (UE) n. NNN/YYYY |
| EU Recital | Erwägungsgrund | considérant | considerando |

---

## 4. Storage schema (SQL, extended for v2)

```sql
CREATE TABLE court_consideration_paragraph (
    -- A. Identity & provenance (14)
    row_id BIGINT PRIMARY KEY,
    bge_id TEXT,
    bge_volume INT,
    bge_series_roman TEXT,
    bge_start_page INT,
    docket_no TEXT,
    docket_era TEXT,
    chamber_prefix TEXT,
    chamber_area TEXT,
    decision_date DATE,
    presiding_judge TEXT,
    bvger_docket TEXT,
    bstger_docket TEXT,
    bpatger_docket TEXT,
    publication_status TEXT,
    -- B. Language (5)
    language TEXT,
    embedded_languages TEXT[],
    has_trilingual_gloss BOOL,
    language_version_equality_invoked BOOL,
    quoted_foreign_statute_languages TEXT[],
    -- C. Paragraph structure (8)
    erwagung_id_raw TEXT,
    erwagung_id_validated TEXT,
    erwagung_id_artifact_class TEXT,
    parent_section TEXT,
    depth INT,
    sachverhalt_letter TEXT,
    is_continuation_fragment BOOL,
    is_subsection_header_only BOOL,
    -- D. Paragraph role (15)
    paragraph_role TEXT,
    is_legal_standard BOOL,
    is_party_position BOOL,
    is_lower_court_summary BOOL,
    is_facts BOOL,
    is_admissibility_recital BOOL,
    is_dispositif BOOL,
    is_cost_dispositif BOOL,
    is_signature_boilerplate BOOL,
    is_moot_determination BOOL,
    is_closing_subsumption BOOL,
    is_editor_summary BOOL,
    cost_assignment_type TEXT,
    vorinstanz_court_type TEXT,
    vorinstanz_canton TEXT,
    -- E. 7-element template (9)
    has_element_1_statutory_anchor BOOL,
    has_element_2_paraphrase BOOL,
    has_element_3_teleology BOOL,
    has_element_4_positive_limb BOOL,
    has_element_5_negative_limb BOOL,
    has_element_6_interpretive_factors BOOL,
    has_element_7_authority_chain BOOL,
    template_score INT,
    opener_verb_owner TEXT,
    -- F. Statute refs (12)
    statute_anchors JSONB,
    lead_statute_anchors JSONB,
    cantonal_code_anchors JSONB,
    i_v_m_chains JSONB,
    is_outdated_statute_version BOOL,
    pre_revision_version_marker TEXT,
    quotes_statute_verbatim BOOL,
    quoted_statute_short TEXT,
    bis_ter_quater_articles TEXT[],
    embedded_statute_paragraph_list BOOL,
    cited_concordat TEXT[],
    statute_pre_2007_form TEXT[],
    -- G. Authority graph (24)
    cited_bge JSONB,
    cited_urteil JSONB,
    cited_doctrine JSONB,
    cited_commentary_series TEXT[],
    cited_periodical TEXT[],
    cited_treaty TEXT[],
    cited_botschaft JSONB,
    cited_parliamentary_record JSONB,
    cited_foreign_court JSONB,
    cited_ecthr JSONB,
    cited_cjeu JSONB,
    cited_url TEXT[],
    cited_gutachten JSONB,
    mit_hinweisen_tail_variant TEXT,
    code_switched_citation_blocks JSONB,
    cited_lugano_convention BOOL,
    cited_hague_convention TEXT[],
    cited_eu_directive JSONB,
    cited_eu_regulation JSONB,
    cited_oecd_mtc BOOL,
    cited_un_convention TEXT[],
    cited_bvger_decision TEXT[],
    cited_bstger_decision TEXT[],
    cited_bpatger_decision TEXT[],
    cited_eparl_kommissionsbericht JSONB,
    cited_vernehmlassung TEXT[],
    author_surnames_caps TEXT[],
    -- H. Cross-lingual hits (7)
    statute_alias_triples_used TEXT[],
    pinpoint_form_triples_used TEXT[],
    older_de_alias_used TEXT[],
    older_docket_format_used BOOL,
    swiss_it_vs_foreign_it TEXT,
    loccit_or_aao TEXT,
    eu_law_alias_triples_used TEXT[],
    -- I. Doctrinal-test invocation (8)
    named_doctrine TEXT[],
    doctrinal_test_invoked TEXT[],
    saving_construction_invoked BOOL,
    review_intensity TEXT,
    holding_polarity TEXT,
    is_remand BOOL,
    art_106_qualified_complaint_required BOOL,
    is_obiter_dictum BOOL,
    -- J. Idioms & formulae (10)
    formula_present TEXT[],
    latin_maxim_present TEXT[],
    swiss_idiom_present TEXT[],
    swiss_metaphor_present TEXT[],
    rebuke_phrase_present TEXT[],
    synthesis_marker_present BOOL,
    bridge_phrase_present BOOL,
    information_principles_invoked TEXT[],
    evidentiary_standard_invoked TEXT,
    iura_novit_curia_invoked BOOL,
    -- K. Quality (10)
    is_csv_parse_artifact BOOL,
    has_encoding_artifacts BOOL,
    embedded_page_anchors TEXT[],
    internal_navigation_refs TEXT[],
    has_quoted_block BOOL,
    has_embedded_list BOOL,
    text_length_chars INT,
    materialien_density TEXT,
    party_anonymization_pattern TEXT[],
    has_anonymized_party BOOL,
    -- L. Procedural metadata (5)
    procedure_type TEXT,
    procedural_history_depth TEXT,
    has_oral_hearing_marker BOOL,
    cited_self_decision_paragraph TEXT[],
    cantonal_court_appeal_pathway TEXT,
    -- M. Composite (1)
    gold_candidate_score FLOAT,
    -- Raw text (for fallback)
    text_raw TEXT,
    text_clean TEXT
);

CREATE INDEX idx_bge_id ON court_consideration_paragraph(bge_id);
CREATE INDEX idx_docket_no ON court_consideration_paragraph(docket_no);
CREATE INDEX idx_chamber_area ON court_consideration_paragraph(chamber_area);
CREATE INDEX idx_template_score ON court_consideration_paragraph(template_score) WHERE template_score >= 5;
CREATE INDEX idx_paragraph_role ON court_consideration_paragraph(paragraph_role);
CREATE INDEX idx_is_legal_standard ON court_consideration_paragraph(is_legal_standard) WHERE is_legal_standard = TRUE;
CREATE INDEX idx_language ON court_consideration_paragraph(language);
CREATE INDEX idx_statute_anchors_gin ON court_consideration_paragraph USING GIN(statute_anchors);
CREATE INDEX idx_cited_bge_gin ON court_consideration_paragraph USING GIN(cited_bge);
CREATE INDEX idx_named_doctrine_gin ON court_consideration_paragraph USING GIN(named_doctrine);
```

---

## 5. Chunking-strategy recommendations (unchanged from v1)

1. Cluster rows by `bge_id` / `docket_no` — paragraphs of one decision are always contiguous.
2. Detect decision boundaries via `is_signature_boilerplate=true`.
3. Detect Erwägung→Dispositiv transition via E-number reset.
4. Drop `is_subsection_header_only` rows from the retrievable index.
5. Drop pure `is_cost_dispositif` / `is_signature_boilerplate` / `is_admissibility_recital`.
6. Merge `is_continuation_fragment` rows back with their preceding parent.
7. Re-knit citation-noise rows: when `erwagung_id_artifact_class != clean`, content belongs to previous valid E-id.
8. Preserve `quoted_statute_block` rows but tag them as embedded-statute, not court-doctrine.
9. Preserve `embedded_page_anchors` in `text_raw` but strip from `text_clean`.
10. Multi-row decisions on one Erwägung: use `erwagung_id_validated` + body-adjacency to re-merge.

---

## 6. Stage-2 retrieval dossier format for Qwen3-8B (extended)

For the cascade dossier, each candidate row passed to Qwen3-8B should carry:

```
[CITATION]   BGE 137 IV 122 E. 4.2 (DE, IV-chamber, criminal, leading-case)
[VALIDATION] erwagung_id_validated=4.2 / is_csv_parse_artifact=false
[LANG]       de, embedded=[fr-doctrine]
[ROLE]       legal_standard / template_score=7 / opener_verb_owner=court
[STATUTES]   [{sr=312.0, code=StPO/CPP/CPP, art=221, abs=1, lit=b, version=current}]
[ECHOES]     cites: [BGE 132 I 21 E. 3.2]; cited_by_in_pool: 8
[DOCTRINE]   named=[]; test=verhaeltnismaessigkeit_three_prong; holding=holds_up
[VORINST]    cantonal_obergericht / Aargau / appeal_two_tier
[PROC]       procedure=beschwerde_strafsachen, no_oral_hearing
[INTL]       no_treaty / no_eu_law
[ANONYM]     parties=[X._, Y._]
[QUERY_MATCH] statute_target_intersection=3; doctrinal_sub_question=Kollusionsgefahr
[TEXT_CLEAN] "Gemäss Art. 221 Abs. 1 lit. b i.V.m. Art. 237 Abs. 1 StPO ist Untersuchungshaft …"
```

The Qwen3-8B judge applies the 8-step decision DAG from [research/per_query_court_gold_anatomy_2026-05-21.md](per_query_court_gold_anatomy_2026-05-21.md) §6.

---

## 7. What changed v1 → v2

| Aspect | v1 | v2 | Why |
|---|---|---|---|
| Field count | 102 | 132 (+30) | Coverage gaps closed (Gap 1) |
| Extraction-method tags | none | every field tagged | Static vs LLM partitioning (Gap 2) |
| BVGer/BStGer/BPatGer dockets as source | absent | `bvger_docket`/`bstger_docket`/`bpatger_docket` | Federal sister courts |
| BVGer/BStGer/BPatGer as cited authority | absent | `cited_bvger_decision`/`cited_bstger_decision`/`cited_bpatger_decision` | Echo-chain may span sister courts |
| International treaties | one lump field | split into Lugano, Hague, OECD MTC, UN conventions | Sharper query→citation matching |
| EU directives/regulations | absent | `cited_eu_directive`/`cited_eu_regulation` | FZA-driven autonomous adoption |
| Pre-2007 statute forms | partial | `statute_pre_2007_form` (OG/PPF/OBG/BStP/ANAG/LSEE/LDDS) | Older val gold may cite these |
| Concordats | absent | `cited_concordat` | Inter-cantonal procedural law |
| Travaux préparatoires sub-classes | one field | split into `cited_botschaft`/`cited_eparl_kommissionsbericht`/`cited_vernehmlassung` | Different signal strength |
| Author-surname citations | inside `cited_doctrine` | also flat list `author_surnames_caps` for quick aggregation | Doctrine-density signal |
| Anonymization | absent | `party_anonymization_pattern`/`has_anonymized_party` | Case clustering |
| Vorinstanz court+canton | absent | `vorinstanz_court_type`/`vorinstanz_canton`/`cantonal_court_appeal_pathway` | Procedural-pathway match |
| Procedure type | absent | `procedure_type` (Art. 72/78/82/113 BGG) | Civil/criminal/admin gating |
| Chamber area | absent | `chamber_area` derived | Direct query-area matching |
| Oral hearing marker | absent | `has_oral_hearing_marker` | Major-case signal |
| Self-cited paragraphs | absent | `cited_self_decision_paragraph` | Re-stitching chunked context |
| Cost-assignment type | rolled into `is_cost_dispositif` | split into `cost_assignment_type` | Drop these from retrieval cleanly |
| Procedural-history depth | absent | `procedural_history_depth` | Stitching multi-instance pathway |
| Cantonal collections | partial | extended to all 26 cantons | Full coverage |
| Cantonal court dockets | partial | extended (BE/TI/JU added) | Cross-canton matching |

---

## 8. What v2 still doesn't promise (deferred to Steps B & C)

Three things v2 still cannot claim until measured:

1. **Field population %**: we don't know how often each field fires across the 2.47M. A field that fires on 0.001% of rows costs storage but offers no discrimination. Step B measures this.
2. **Discriminative power on val gold**: we don't know which fields actually separate the ~100 gold court paragraphs from ambient. Step C measures this.
3. **Regex precision**: each regex is a hypothesis. False positives (e.g. `Art. 5` matching a phone-number-pattern) and false negatives (e.g. `E. 4.2bis` missed by the `^\d+(?:\.\d+){0,4}[a-z]?$` regex) need to be measured on a held-out sample. Step B-extension samples this.

---

## 9. Step B plan (next, after schema v2 lock)

**Goal**: measure regex extractability per field across stratified sample of 10,000 rows from 2,476,315.

**Stratification**:
- 2,000 rows from early corpus (rows 1-10K) — high BGE published density
- 2,000 rows from rows 10K-100K — early-mid corpus
- 2,000 rows from rows 100K-500K — mid corpus
- 2,000 rows from rows 500K-1.5M — mid-late corpus
- 2,000 rows from rows 1.5M-2.47M — late corpus + post-2024 + 7B_ chamber
- Plus full set of val gold court rows (~100) for direct lift comparison in Step C

**Implementation**: single Python script `extract_v2_static_fields.py` with regex patterns per field tagged `regex` and `regex+rules`. For each row, populate the ~95 static fields. Output: `schema_v2_field_population_10K.csv` with one column per field-population-bool/value, plus a histogram CSV `schema_v2_population_histogram.csv` showing fill-rate per field.

**Acceptance criteria**:
- Each `regex` field has ≥ 70% precision on a 100-row random spot-check.
- Each field is either > 0.5% population (worth keeping) or flagged for removal.
- No field has > 95% population (which would mean zero discrimination).

**Output**: `research/schema_v2_field_population_2026-05-21.md` + the two CSVs.

---

## 10. Step C plan (after Step B)

**Goal**: compute lift-over-ambient for each field on the ~100 val gold court rows.

**Method**:
1. Identify the gold court rows in `data/court_considerations.csv` by joining on the gold citation strings from `data/val.csv` × `data/court_considerations.csv` per qid.
2. Run Step B's extractor on (a) all ~100 gold court rows, (b) a random sample of 10,000 ambient rows.
3. For each field, compute `lift = P(field=true | gold) / P(field=true | ambient)`.
4. Fields with `lift >= 10` are **strong gold signals** (promote in gold_candidate_score).
5. Fields with `2 <= lift < 10` are **moderate gold signals** (include with low weight).
6. Fields with `lift < 2` are **non-discriminative** (drop from gold_candidate_score; may still be useful for filtering, e.g. `is_signature_boilerplate`).
7. Fields with `lift < 0.5` are **anti-signals** (negative weight in gold_candidate_score, e.g. `is_party_position`).

**Output**: `research/schema_v2_discriminative_power_2026-05-21.md` with a per-field lift table, and a final v3 of `gold_candidate_score` formula calibrated on val.

---

## 11. Provenance — 16 discovery agents + v2 critical revision

**Round 1 + Round 2** as in v1 §8 (16 agents covering ~2,000 paragraphs from rows 2-1001, 1390-1925, 2500-3575, 11300-11425, 500K, 1.5M, 2.4M).

**v2 critical revision** added by review of:
- Federal sister-court docket patterns (BVGer/BStGer/BPatGer official docket structures from Bundesgerichts website).
- Public BGer reorganization notes (Art. 32 BGG; Reglement 7B post-2024).
- International treaty cross-lingual table from the SR/RS systematic-collection website (`SR 0.* / RS 0.*`).
- Pre-2007 statute forms from the Bundesblatt revision messages (BBl 2001 4202 ff. for BGG; BBl 2005 6885 ff. for AsylG/AuG).
- Cantonal collection prefixes from each canton's `Gesetzessammlung` portal (covered 23/26 cantons here vs ~11 in v1).

---

_End of schema v2. Lock when Steps B & C complete and the gold_candidate_score formula is calibrated against val._
