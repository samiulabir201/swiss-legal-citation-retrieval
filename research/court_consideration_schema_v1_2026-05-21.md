# Structured-Data Schema v1 — `court_considerations.csv` Paragraph Catalogue

_Generated 2026-05-21 after 16 parallel discovery agents (8 Round-1 + 8 Round-2) read a stratified sample of ~2,000 paragraphs across rows 2-1001, 1390-1925, 2500-3575, 11300-11425, 500K, 1.5M, 2.4M of `E:/swiss_citation_extraction/data/court_considerations.csv`. The corpus is 2,476,315 rows of Swiss Federal Court (Bundesgericht / Tribunal fédéral / Tribunale federale) considérants in DE/FR/IT. This schema captures every structurally distinct field we discovered that should be preserved when each paragraph is structured for use by Qwen3-8B as a RAG judge agent. Total field count: 102._

---

## 0. Design principles

1. **Language-agnostic where possible**: every field that has a DE form must also support FR and IT forms. The triple `BGE ↔ ATF ↔ DTF` is the most important alias and is treated as a single canonical entity.
2. **`sr_number` as the canonical statute key**: every statute citation across DE/FR/IT carries the same `SR/RS xxx.xxx` number. This is the language-invariant identity anchor.
3. **Content-driven validation**: never trust the `citation` column blindly. Numbers, years, and URL fragments leak into it. Every `erwagung_id` must be cross-validated against the body text.
4. **Three temporal docket eras**: pre-1990 (`P.NNN/YYYY` with dot), 1990-2007 (`1A./1P./2A./4C./5A./6P./7B./H/I`), 2007-2024 (`1B_/1C_/2C_/4A_/5A_/6B_/8C_/9C_` with underscore), post-2024 (`7B_` introduced, `6B_` deprecated). Schema accepts all four.
5. **Content-driven citation-noise**: BGE rows on environmental/scientific/quantitative cases (percentages, m/s, ha, GWh, clock times) trigger ~22% noise; procedural unpublished BGer rows trigger 0%. Schema has separate noise-class fields.
6. **Tri-lingual paraphrase verbs** are catalogued explicitly. The 7-element template applies content-driven (not publication-tier-driven), so schema must extract the template across DE/FR/IT.
7. **Foreign-law discrimination**: Swiss-IT prose mixed with Italian-state references (`comma` vs `cpv.`, `D.L.` vs `SR/RS`) must be tagged at the citation level.
8. **Echo-decision is the dominant mode** (>85% mid-corpus). The schema has fields for citation graph that capture which Leitentscheide a paragraph echoes.

---

## 1. Field catalogue (102 fields, grouped)

### A. Identity & provenance (10 fields)

| # | Field | Type | Description / extraction rule |
|---|---|---|---|
| 1 | `row_id` | int | Original CSV row index (1-2,476,315). |
| 2 | `bge_id` | string | Normalised BGE/ATF/DTF id (e.g. `"BGE 137 IV 122"`). **Always DE-form `BGE`** regardless of decision language — confirmed bilingual-sigil-collision rule. |
| 3 | `bge_volume` | int | Volume number (e.g. `137`). |
| 4 | `bge_series_roman` | enum | One of `I, II, III, IV, V, VI`. I=public/constitutional, II=administrative, III=civil, IV=criminal, V=social, VI=debt collection. Pre-1995 uses Arabic suffix `Ia/Ib/Ic` instead. |
| 5 | `bge_start_page` | int | Volume page (e.g. `122`). |
| 6 | `docket_no` | string | Unpublished BGer docket (e.g. `1B_28/2022`, `2P.137/2005`, `P.923/1982`, `H 316/03`, `I 345/06`). |
| 7 | `docket_era` | enum | `pre_1990` / `1990_2006` / `2007_2023` / `post_2024`. Determined by docket-format regex. |
| 8 | `chamber_prefix` | string | Two-three-letter prefix (`1B_`, `1C_`, `2C_`, `2P.`, `4A_`, `4C.`, `5A_`, `5D_`, `5F_`, `6B_`, `6P.`, `6S.`, `7B_`, `1F_`, `2F_`, `8C_`, `8D_`, `9C_`, `H`, `I`, `1A.`, `1P.`, `2A.`, `2D_`). Used to derive `chamber_area`. |
| 9 | `decision_date` | date | ISO date; from `vom DD. Monat YYYY` (DE) / `du DD MMMM YYYY` (FR) / `del DD MMMM YYYY` (IT). |
| 10 | `presiding_judge` | string | Extracted from closing boilerplate (`Der Präsident: X`, `Le Président : X`, `Il Presidente: X`). Same person across DE/FR/IT (e.g., Hurni for I. zivilrechtliche / Ire Cour de droit civil). |

### B. Language & cross-lingual (5 fields)

| # | Field | Type | Description |
|---|---|---|---|
| 11 | `language` | enum | `de` / `fr` / `it`. Detected from opener regex on first 200 chars. |
| 12 | `embedded_languages` | list[enum] | Other languages observed in the body (e.g. DE prose with FR doctrinal-monograph titles). Common: DE with English (`legitimate business reasons`, `living instrument`), IT with German (`Druckmittel`, `Spector Pro`), FR with Latin (`mutatis mutandis`, `prima facie`). |
| 13 | `has_trilingual_gloss` | bool | True when DE+FR+IT synonyms appear inline (e.g. `attestazione di soggiorno; Aufenthaltsausweis; attestation de séjour` at row 1513). Rare but extremely high-value. |
| 14 | `language_version_equality_invoked` | bool | True if paragraph invokes the constitutional rule that all three language versions are equally binding (`"die in gleicher Weise verbindlich sind"`). |
| 15 | `quoted_foreign_statute_languages` | list[enum] | Languages of verbatim foreign statute quotes (e.g. IT BGer quoting `D.L. 22 gennaio 2004 n. 42` from Italian state law). |

### C. Paragraph structure & identity (8 fields)

| # | Field | Type | Description |
|---|---|---|---|
| 16 | `erwagung_id_raw` | string | Raw E-tag from `citation` column (may be corrupt). |
| 17 | `erwagung_id_validated` | string | Cleaned canonical id (`E. 4.2`, `consid. 4.2`, `E. A.b`, etc.). Validated against body text. |
| 18 | `erwagung_id_artifact_class` | enum | `clean` / `year_token` (e.g. `E. 2017`) / `percentage` (`E. 97.9`) / `speed_unit` (`E. 11.9` from m/s) / `length_unit` (`E. 88.5` from m, `E. 8.4.1m`, `E. 8.4.1ha`) / `clock_time` (`E. 17.30`) / `energy_unit` (`E. 33.78` from GWh) / `cite_fragment` (`E. 73.13` from `BGE 73 II 13`) / `letter_suffix` (`E. 5.5.4a`) / `language_suffix` (`E. 5.5.1it`, `E. 5.5.6di`) / `nr_fragment` (`E. 42` from `n. 42`) / `empty` / `header_only`. |
| 19 | `parent_section` | string | Parent E-id (e.g. `E. 4` for `E. 4.2`). Computed. |
| 20 | `depth` | int | E-nesting depth (1 for `E. 4`, 2 for `E. 4.2`, etc.). Up to 5 observed (`E. 6.2.2.1`). |
| 21 | `sachverhalt_letter` | string | If paragraph is in Sachverhalt block: `A`, `B`, `C`, `D`, `A.a`, `A.b`, etc. |
| 22 | `is_continuation_fragment` | bool | True if text begins mid-sentence (lowercase, opens with closing bracket, opens with `Nr.`, `Jahrhundert`, etc.). |
| 23 | `is_subsection_header_only` | bool | True if text is just `"3.5."` / `"1.3."` etc. (degenerate header with no body). |

### D. Paragraph role (12 fields)

| # | Field | Type | Description |
|---|---|---|---|
| 24 | `paragraph_role` | enum | One of: `legal_standard`, `reasoning`, `application_in_concreto`, `synthesis`, `facts`, `procedural_history`, `party_position`, `lower_court_summary`, `admissibility`, `dispositif`, `costs`, `signature`, `mitteilung_recipients`, `editor_summary`, `bridge`, `header_only`, `continuation_fragment`, `quoted_statute_block`, `parliamentary_materials`. |
| 25 | `is_legal_standard` | bool | Paragraph contains 7-element doctrinal template (see Group E). |
| 26 | `is_party_position` | bool | First-verb subject is the appellant (`Der Beschwerdeführer rügt/bestreitet`, `Le recourant fait valoir / soutient / invoque / reproche`, `Il ricorrente sostiene / contesta / fa valere`). |
| 27 | `is_lower_court_summary` | bool | First-verb subject is `Die Vorinstanz hat erwogen / festgestellt`, `La cour cantonale a retenu / jugé / considéré`, `L'autorità precedente / Il TPF non ha ritenuto`. |
| 28 | `is_facts` | bool | Within Sachverhalt block (letter-labeled). |
| 29 | `is_admissibility_recital` | bool | Body matches admissibility templates (`Auf die Beschwerde wird (nicht) eingetreten`, `Le recours est (ir)recevable`, `Il ricorso è (in)ammissibile`). |
| 30 | `is_dispositif` | bool | Body matches dispositive verbs (`Die Beschwerde wird abgewiesen / gutgeheissen`, `Le recours est rejeté / admis`, `Il ricorso è respinto / accolto`). Often very short (≤30 words). |
| 31 | `is_cost_dispositif` | bool | Body matches cost templates (`Gerichtskosten / Parteientschädigung`, `frais judiciaires / dépens`, `spese giudiziarie / ripetibili`). |
| 32 | `is_signature_boilerplate` | bool | Body matches `Im Namen der … Abteilung des Schweizerischen Bundesgerichts` / `Au nom de la … Cour du Tribunal fédéral suisse` / `In nome della … Corte del Tribunale federale svizzero`. This is the **deterministic end-of-decision marker**. |
| 33 | `is_moot_determination` | bool | Body says "we leave the question open" (`kann offenbleiben`, `la question peut demeurer indécise / rester ouverte`, `può rimanere indeciso il quesito`). |
| 34 | `is_closing_subsumption` | bool | Body opens with closing-connective (`Demnach`, `Im Ergebnis`, `Zusammenfassend`, `En définitive`, `Il s'ensuit`, `Ne segue che`). |
| 35 | `is_editor_summary` | bool | Body is `[Zusammenfassung: …]` / `[Résumé: …]` editorial block inserted by BGE publication office. |

### E. 7-element doctrinal-rule-statement template (9 fields)

| # | Field | Type | Description |
|---|---|---|---|
| 36 | `has_element_1_statutory_anchor` | bool | First 200 chars contain explicit `Art. N (Abs. M) (lit. x) CODE` / `art. N (al. M) (let. x) CODE` / `art. N (cpv. M) (lett. x) CODE`. Opener may be `Gemäss/Nach/Im Sinne von` (DE), `Aux termes de / Selon / Conformément à / En vertu de / D'après / À teneur de` (FR), `Giusta / Ai sensi di / Secondo / Conformemente all'` (IT). |
| 37 | `has_element_2_paraphrase` | bool | Court restates the rule in its own voice using a court-subject verb: `bestimmt / sieht vor / regelt / definiert / gewährt` (DE), `dispose / prévoit / régit / définit / impose / exige / garantit / consacre` (FR), `dispone / prevede / stabilisce / regge / definisce / impone / esige / garantisce` (IT). |
| 38 | `has_element_3_teleology` | bool | Body contains purpose-statement marker: `soll verhindern, dass / bezweckt / dient` (DE), `vise à empêcher / a pour but de / tend à / afin de / aux fins de / destiné à / a pour vocation` (FR), `mira a / ha lo scopo di / risponde a / tende a` (IT). |
| 39 | `has_element_4_positive_limb` | bool | Body contains "non-exhaustive factor list" marker: `namentlich / insbesondere` (DE), `notamment` (FR), `in particolare / segnatamente` (IT). |
| 40 | `has_element_5_negative_limb` | bool | Body contains boundary-stopper: `genügt indessen nicht / vermag nicht zu` (DE), `ne saurait suffire / ne saurait à elle seule / il ne se justifie pas / ne ressort pas / manque de pertinence` (FR), `non è sufficiente / non si riferisce / occorre fare astrazione da / non si concilia` (IT). |
| 41 | `has_element_6_interpretive_factors` | bool | Body contains multi-factor balancing marker: `Bei der Frage, ob …, ist auch der Art, Bedeutung, Schwere … Rechnung zu tragen` (DE), `Dans cet examen, entrent en ligne de compte …` (FR), `Sotto il profilo di …, occorre tener conto di …` (IT). |
| 42 | `has_element_7_authority_chain` | bool | Closing parenthesis with 2-6 ATF/BGE/DTF cites ending in `mit Hinweisen / je mit Hinweisen / mit weiteren Hinweisen / und die dort zitierten Verweise` (DE) / `et les références / et les références citées / et les arrêts cités / et la jurisprudence citée` (FR) / `con rinvii / con riferimenti / e rinvii / e riferimenti / e richiamo / e rispettivi rinvii / e numerosi riferimenti` (IT). |
| 43 | `template_score` | int (0-7) | Count of elements 36-42 present. ≥5 → high-confidence legal_standard. |
| 44 | `opener_verb_owner` | enum | `court` / `appellant` / `lower_court` / `legislator` / `doctrine` / `foreign_authority`. The verb-ownership discriminator. |

### F. Statute references (10 fields)

| # | Field | Type | Description |
|---|---|---|---|
| 45 | `statute_anchors` | list[dict] | List of `{sr_number, statute_short_de, statute_short_fr, statute_short_it, article, abs, lit, ziff, abs_form (Abs./al./cpv.), lit_form (lit./let./lett.), version_marker (current/aArt./vecchio_art)}`. Each anchor is normalised to its `sr_number` (language-invariant). |
| 46 | `lead_statute_anchors` | list[dict] | Subset of #45 where anchor appears in first 200 chars of body. Discriminates rule-recital from tangential mention. |
| 47 | `cantonal_code_anchors` | list[dict] | Cantonal-statute citations with cantonal-collection prefix: `BSG` (BE), `LS` (ZH), `RSV` (VD), `RSG` (GE, with slash-form `E 5/10`), `RSN` (NE), `RSF` (FR), `sGS` (SG), `BR` (GR), `RB` (TG), `BLV` (VD), `RL/TI` (TI), `SRSZ` (SZ). |
| 48 | `i_v_m_chains` | list[list[dict]] | `Art. X i.V.m. Art. Y` / `art. X en lien avec art. Y` / `art. X cum art. Y` chains. Connects multiple statute refs into a single conceptual citation. |
| 49 | `is_outdated_statute_version` | bool | True if `aArt.` (DE) / `ancien art.` (FR) / `vecchio art.` (IT) prefix is used. |
| 50 | `pre_revision_version_marker` | string | Verbatim version-qualifier (`in der bis Ende 2009 geltenden Fassung` / `dans sa teneur jusqu'au …` / `nella versione in vigore fino al …`). |
| 51 | `quotes_statute_verbatim` | bool | True if body contains literal verbatim quotation of a statute introduced by `lautet wie folgt:` / `hat folgenden Wortlaut:` / `a la teneur suivante:` / `ha il seguente tenore:` / `prévoit ceci:`. |
| 52 | `quoted_statute_short` | string | Which statute is verbatim quoted (e.g. `Art. 40 EpG`, `art. 26 OLL 3`). |
| 53 | `bis_ter_quater_articles` | list[string] | Articles with Latin ordinal suffix (`Art. 24a`, `Art. 3bis`, `Art. 65 Abs. 1bis`, `art. 179novies CP`). Schema regex: `art\.\s*\d+\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?`. |
| 54 | `embedded_statute_paragraph_list` | bool | True if body contains numbered statute-text enumeration (`1 Non è ammessa …`, `2 I sistemi …`, `a. die Sicherheitsleistung; b. …`). |

### G. Citation graph & authority chain (15 fields)

| # | Field | Type | Description |
|---|---|---|---|
| 55 | `cited_bge` | list[dict] | List of cited BGE/ATF/DTF: `{volume, series_roman, page, erwagung, page_pinpoint, language_form (BGE/ATF/DTF)}`. The triple is content-identical (same volume/page/E.). |
| 56 | `cited_urteil` | list[dict] | Unpublished BGer dockets cited (`1B_575/2021`, `Urteil M. vom 11. Dezember 2003, I 589/03`). Includes anonymised-letter format pre-2007. |
| 57 | `cited_doctrine` | list[dict] | Doctrine references with author SURNAME (ALL CAPS convention), work title, edition, year, pinpoint type+value. Pinpoint types: `Rz.`/`N.` (DE), `n°` (FR), `n.` (IT). Pattern: `AUTHOR, Title, Yth ed. YYYY, N./Rz./n°/n. NNN ad/zu Art. M`. |
| 58 | `cited_commentary_series` | enum-list | Major Swiss commentary series cited: `Basler Kommentar (BSK)`, `Berner Kommentar (BK)`, `Zürcher Kommentar (ZK)`, `Commentaire romand (CR)`, `St. Galler Kommentar`, `Praxiskommentar`. |
| 59 | `cited_periodical` | list[string] | Journals (`ZBl`, `sic!`, `AJP`, `BVR`, `SVR`, `StR`, `StE`, `ASA`, `EuGRZ`, `Jusletter`, `RDAF`, `RDAT`, `RtiD`, `SJ`, `Pra`, `FamPra.ch`, `JdT`, `RSJ`, `RJB`, `RJN`, `Medialex`, `URP`, `RDS`, `SBVR`, `BGC`, `BO/NE`, `BU/TI`). |
| 60 | `cited_treaty` | list[string] | Treaties (`EMRK/CEDH/CEDU`, `UNO-Pakt II/Pacte ONU II`, `FZA/ALCP`, `KRK/CDE`, `VRK/CV`, `SDÜ`, `SGK`, `Visakodex`, `IPRG/LDIP`, `CEAG/CEEJ/EUeR`, `Dublin-III-Verordnung`). All carry `SR 0.NNN.NNN` prefix for international law. |
| 61 | `cited_botschaft` | list[dict] | Federal Council messages: `{date, subject, bbl_year, bbl_page, multi_botschaft_label}` (e.g. `Botschaft KG I/II/III`). |
| 62 | `cited_parliamentary_record` | list[dict] | `AB YYYY S/N NNN` (DE), `BO YYYY CE/CN NNN` (FR), `BU YYYY CS/CN NNN` (IT). `AB ↔ BO ↔ BU` is a confirmed alias triple. |
| 63 | `cited_foreign_court` | list[dict] | Non-Swiss courts: `BVerfG` (German Constitutional Court with `BvR NNN/YY`), `Supreme Court of the United States` (`575 U.S. [2015]`), `EFTA Court`, `Corte di Appello`, `Cassazione`. |
| 64 | `cited_ecthr` | list[dict] | ECtHR judgments: `{case_name_party, case_name_state, date, application_no, paragraph_pinpoint, recueil_year}`. Format varies: DE `EGMR-Urteile Trabelsi gegen Deutschland vom 13. Oktober 2011 [Nr. 41548/06]`, FR `arrêt CourEDH X contre Y du DATE, n° NNN/NN § N`, IT `sentenza CorteEDU X contro Y del DATE`. |
| 65 | `cited_cjeu` | list[dict] | CJEU judgments: `C-NNN/YYYY P`, with AG opinions (`Schlussanträge der Generalanwältin Juliane Kokott`). |
| 66 | `cited_url` | list[string] | Federal-admin URLs (e.g. `www.amtsblattportal.ch`, `www.bafu.admin.ch`). Sometimes with `besucht am DATE`. |
| 67 | `cited_gutachten` | list[dict] | Commissioned expert opinions (`Gutachten im Auftrag der Eidgenössischen Steuerverwaltung vom DATE`). |
| 68 | `mit_hinweisen_tail_variant` | enum | `mit_hinweisen` / `je_mit_hinweisen` / `mit_weiteren_hinweisen` / `mit_zahlreichen_hinweisen` / `und_dort_zitierte_verweise` (DE); `et_les_references` / `et_les_references_citees` / `et_les_arrets_cites` / `et_la_jurisprudence_citee` / `et_la_reference_citee` / `avec_les_references` (FR); `con_rinvii` / `con_riferimenti` / `e_rinvii` / `e_riferimenti` / `e_richiamo` / `e_rispettivi_rinvii` / `e_numerosi_riferimenti` (IT); `none`. |
| 69 | `code_switched_citation_blocks` | list[dict] | Cite blocks where citing-paragraph language differs from cited-work language (e.g. FR paragraph citing `HÄFELIN/MÜLLER/UHLMANN, Allgemeines Verwaltungsrecht, 7 e éd. 2016, n. 2718 p. 617` with FR pagination markers around DE title). |

### H. Cross-lingual alias hits (6 fields)

| # | Field | Type | Description |
|---|---|---|---|
| 70 | `statute_alias_triples_used` | list[enum] | Which canonical triples appear: `BV_Cst_Cost`, `BGG_LTF_LTF`, `ZGB_CC_CC`, `OR_CO_CO`, `StGB_CP_CP`, `StPO_CPP_CPP`, `ZPO_CPC_CPC`, `EMRK_CEDH_CEDU`, `EGMR_CourEDH_CorteEDU`, `StHG_LHID_LAID`, `DBG_LIFD_LIFD`, `KVG_LAMal_LAMal`, `IVG_LAI_LAI`, `ATSG_LPGA_LPGA`, `UVG_LAA_LAINF`, `BGBM_LMI_LMI`, `AuG_LEtr_LStr`, `AsylG_LAsi_LAsi`, `EnG_LEne_LEne`, `RPG_LAT_LPT`, `USG_LPE_LPAmb`, `TSchG_LPA_LPAn`, `ArG_LTr_LL`, `ArGV3_OLT3_OLL3`, `KG_LCart_LCart`, `WEKO_COMCO_COMCO`, `DSG_LPD_LPD`, `LTBC_LTBC_LTBC`, `IRSG_EIMP_AIMP`, `IRSV_OEIMP_OAIMP`, `EUeR_CEEJ_CEAG`, `SVG_LCR_LCStr`, `VZV_OAC_OAC`, `PatGG_LTFB_LTFB`, `StBOG_LOAP_LOAP`, `RVOG_LOGA_LOGA`, `VwVG_PA_PA`, `BBl_FF_FF`, `AS_RO_RU`, `SR_RS_RS`, `AB_BO_BU`, `VGG_LTAF_LTAF`, `BÜPF_LSCPT_LSCPT`, `EpG_LEp_LEp`, `EpV_OEp_OEp`, `BGÖ_LTrans_LTras`, `BGA_LAr_LAr`, `KRK_CDE_CRC`, `VRK_CV_CV`, `FZA_ALCP_ALC`, `RTVG_LRTV_LRTV`, `MedBG_LPMéd_LPMed`, `LLCA_LLCA_LLCA`, `LFINMA_LFINMA_LFINMA`. |
| 71 | `pinpoint_form_triples_used` | list[enum] | Which pinpoint-abbreviation triples appear: `Abs_al_cpv`, `lit_let_lett`, `Ziff_ch_n`, `S_p_pag`, `f_s_seg`, `ff_ss_segg`, `Rz_n°_n`, `E_consid_consid`, `i.V.m._cum_cum`, `a.a.O._précité_loccit`, `vgl_cf_cfr`, `vom_du_del`, `gegen_contre_contro`. |
| 72 | `older_de_alias_used` | list[enum] | Older DE shorthand: `MRK` (= EMRK), `aBV` (= old BV pre-1999), `aArt.`, `vBV`, `vLGC` (IT for pre-2002 cantonal). |
| 73 | `older_docket_format_used` | bool | True if pre-2007 docket format (`P.NNN/YYYY`, `2P.NNN/YYYY`, `4C.NNN/YYYY`, `6S.NNN/YYYY`, `1P.NNN/YYYY`, `2A.NNN/YYYY`, `H NNN/YY`, `I NNN/YY`). |
| 74 | `swiss_it_vs_foreign_it` | enum | For IT rows only: `swiss_it` (uses `cpv.`, `RS`, `Ministero pubblico`), `foreign_state_italian` (uses `comma`, `D.L.`, `Procura della Repubblica`), `mixed` (both, typical in rogatoria/cooperation cases). |
| 75 | `loccit_or_aao` | enum | Which back-reference shorthand is used: `loc_cit_it`, `loccit_fr`, `aao_de`, `precite_fr`, `ibid`. Rule: tracks the *language of the cited work*, not the citing decision. |

### I. Doctrinal-test invocation (8 fields)

| # | Field | Type | Description |
|---|---|---|---|
| 76 | `named_doctrine` | list[enum] | Named Swiss-legal doctrines invoked: `Schubert-Praxis`, `PKK-Praxis`, `AVLOCA-Praxis`, `Reneja-Praxis`, `Star-Praxis`, `Boultif-Kriterien`, `Üner-Kriterien`, `Maslov-Kriterien`, `Kissiwa-Koffi-Kriterien`, `Engel-Kriterien`, `Bedarfsmarktkonzept`, `SSNIP-Test`, `Methodenpluralismus`, `Lex-posterior-Regel`, `Genehmigungsfiktion`, `VgT`, `Ostendorf`, `Bacchini`, `Zweijahresregel`. Extracted via curated lookup + `sog./so genannt/dit/cosiddetto` marker. |
| 77 | `doctrinal_test_invoked` | list[enum] | Standard tests: `art_36_BV_three_prong` (gesetzliche Grundlage + öffentliches Interesse + Verhältnismässigkeit + Kerngehalt), `verhaeltnismaessigkeit_three_prong` (Geeignetheit + Erforderlichkeit + Verhältnismässigkeit i.e.S.), `arbitraire_standard` (FR), `willkuer_standard` (DE), `arbitrio_standard` (IT), `anklagegrundsatz`, `rechtliches_gehoer`, `rechtsweggarantie`, `unbestimmte_Begriffe_test`, `doppia_punibilita`, `vorrang_bundesrecht`. |
| 78 | `saving_construction_invoked` | bool | True if `verfassungskonforme Auslegung` / `verfassungskonforme Anwendung` / `interprétation conforme à la constitution` / `interpretazione conforme` is invoked. |
| 79 | `review_intensity` | enum | `freie_kognition` / `willkuer_pruefung` / `Angemessenheitskontrolle` / `vollkognition` (FR `plein pouvoir d'examen` / `pouvoir d'examen limité`; IT `pieno potere d'esame` / `potere d'esame limitato`). |
| 80 | `holding_polarity` | enum | `holds_up` (`hält stand`, `tient`, `tiene`) / `fails` (`hält nicht stand`, `ne tient pas`, `non tiene`) / `none`. |
| 81 | `is_remand` | bool | `Rückweisung an die Vorinstanz` / `renvoi à l'autorité précédente` / `rinvio all'autorità precedente`. |
| 82 | `art_106_qualified_complaint_required` | bool | Body invokes Art. 106 Abs. 2 BGG (heightened pleading for constitutional grievances): `qualifizierte Rügepflicht / qualifizierte Substantiierungspflicht`, `exigence de motivation accrue`, `motivazione qualificata`. |
| 83 | `is_obiter_dictum` | bool | Body opens with obiter marker: `Im Übrigen / überdies` (DE), `À titre superfétatoire / Au demeurant / Du reste` (FR), `A titolo abbondanziale / Di transenna / Per giunta / Del resto` (IT). |

### J. Idioms & formulae (10 fields)

| # | Field | Type | Description |
|---|---|---|---|
| 84 | `formula_present` | list[enum] | Canonical formulae: `vor_bundesrecht_standhalten`, `sinn_und_geist`, `oberste_richtschnur`, `praktische_konkordanz`, `grobmaschiges_netz`, `giesskannenprinzip`, `riegelwirkung`, `regle_aptitude_necessite_proportionnalite_stricto_sensu`, `lacune_proprement_dite_improprement_dite_occulte`, `concubinage_qualifie_stable_durable`, `peut_demeurer_indecise`, `peut_etre_d_emblee_rejete`, `quoi_qu_en_dise`, `ne_saurait_INF`, `en_l_occurrence`, `en_l_espece`, `fatta_la_dovuta_trasposizione`, `controllo_astratto`, `controllo_concreto`, `controllo_accessorio`, `riserbo_giudiziario`, `salus_publica_suprema_lex`, `mit_an_sicherheit_grenzender_wahrscheinlichkeit`, `ueberwiegende_wahrscheinlichkeit`, `vraisemblance_preponderante`, `verosimiglianza_preponderante`. |
| 85 | `latin_maxim_present` | list[string] | `in dubio pro reo`, `e contrario`, `prima vista`, `prima facie`, `reformatio in peius`, `dies a quo`, `restitutio in integrum`, `i.V.m./cum`, `i.f./in fine`, `a.a.O./loc cit./op cit./précité`, `Nulla poena sine lege`, `iura novit curia`, `mutatis mutandis`, `a fortiori`, `pacta sunt servanda`, `ratio legis`, `iudex a quo/ad quem`. |
| 86 | `swiss_idiom_present` | list[string] | Unique Swiss-legal idioms: `Doppelnennung` (DE gender-pair), `Bund-Kanton-Spannung`, `Schwerlast` (FR), `pêle-mêle et sommaire` (FR rebuke), `on peine à les suivre` (FR), `alla berlina` (IT), `naming and shaming` (IT borrowing), `a torto / a ragione`, `Quintessenz / Quintessenza`, `besucht am` (URL access date). |
| 87 | `swiss_metaphor_present` | list[string] | Picturesque BGer metaphors: `grobmaschiges_netz`, `Stadtmauer_ueber_der_Schlucht`, `Sprengung_des_Zonencharakters`, `Riegelwirkung`, `Mailaender_Edikt`, `Spannungsverhältnis`. |
| 88 | `rebuke_phrase_present` | list[enum] | Court rebuke openers: `manque_de_pertinence`, `peine_a_suivre`, `quoi_qu_en_dise`, `pele_mele_sommaire`, `ne_saurait_etre_suivie`, `A_torto_period`, `mal_fonde_le_grief_doit_etre_ecarte`, `inammissibilmente`. |
| 89 | `synthesis_marker_present` | bool | `Als Quintessenz ergibt sich` / `Insgesamt zeigt sich demnach` / `En définitive` / `Il s'ensuit` / `In definitiva` / `Quintessenza`. |
| 90 | `bridge_phrase_present` | bool | Single-sentence section-announcement (`Zu prüfen ist weiter, ob …`, `Reste à examiner …`, `Resta da determinare …`). |
| 91 | `information_principles_invoked` | list[enum] | If any of `Sachlichkeit / Transparenz / Verhältnismässigkeit / Vollständigkeit` (Art. 10a BPR voting-information principles, recurring 4-element checklist). |
| 92 | `evidentiary_standard_invoked` | enum | `freie_beweiswuerdigung` / `ueberwiegende_wahrscheinlichkeit` / `vraisemblance_preponderante` / `verosimiglianza_preponderante` / `strict_beweis` / `mit_an_sicherheit_grenzender_wahrscheinlichkeit` / `glaubhaft_machen` / `rendre_vraisemblable` / `rendere_verosimile` / `none`. |
| 93 | `iura_novit_curia_invoked` | bool | True if explicitly invoked or denied (`das Bundesgericht prüft das Recht von Amtes wegen`, `le Tribunal fédéral applique le droit d'office`, `il Tribunale federale applica d'ufficio il diritto`). |

### K. Quality / cleanup flags (8 fields)

| # | Field | Type | Description |
|---|---|---|---|
| 94 | `is_csv_parse_artifact` | bool | True if `erwagung_id_artifact_class != clean`. |
| 95 | `has_encoding_artifacts` | bool | Per-case encoding bug detected (umlauts → `?` in BGE 148 II 36; some cases have systematic mis-encoding). |
| 96 | `embedded_page_anchors` | list[string] | Mid-text page-break markers (`BGE 137 IV 122 S. 128`, `ATF 137 IV 122 p. 128`, `DTF 137 IV 122 pag. 128`) — page-running-head from official report, NOT cross-citations. |
| 97 | `internal_navigation_refs` | list[string] | `vgl. E. X hiervor/hiernach`, `cf. consid. X ci-dessus/ci-dessous`, `cfr. supra/infra consid. X`, `siehe oben/unten`, `vorne/hinten`. |
| 98 | `has_quoted_block` | bool | Body contains `"…"` block of length ≥ 50 chars (verbatim statute or party submission). |
| 99 | `has_embedded_list` | bool | Body contains numbered/lettered list (e.g. `1 …; 2 …; 3 …` or `a. …; b. …; c. …` or `(1) …; (2) …`). |
| 100 | `text_length_chars` | int | Length of body text. Range observed: 4 chars (header-only stub) to ~7,500 chars (long doctrinal paragraph). Median ~800-1000 chars. |
| 101 | `materialien_density` | enum | `none` / `low` / `medium` / `high`. High = paragraph is parliamentary-debate-heavy / committee-report-heavy / Botschaft-heavy. |

### L. Composite / derived (1 field)

| # | Field | Type | Description |
|---|---|---|---|
| 102 | `gold_candidate_score` | float | 0-1 composite: high if `template_score ≥ 5` AND `paragraph_role ∈ {legal_standard, reasoning, application_in_concreto}` AND `chamber_area matches query area` AND `is_csv_parse_artifact = false`. Used by Stage-2 LLM judge as the primary gold-candidate ranking signal. |

---

## 2. Validation rules

### 2.1 Citation-noise detector

A row's `erwagung_id_raw` is `clean` iff ALL of:
- Matches regex `^[A-Z](?:\.[a-z])?$` (letter-Sachverhalt) OR `^\d+(?:\.\d+){0,4}[a-z]?$` (decimal-Erwägung, up to 5 levels).
- First segment ≤ 15.
- Does not match year-shape `^\d{4}$`.
- Does not match unit-suffix shape `\d+(m|ha|m\d|°|%)`.
- Does not contain language-tag suffix (`it`, `de`, `fr`).
- Sequence-coherent with neighbour rows in same case.

Failing rows get `erwagung_id_artifact_class` set accordingly. **Tested rate**: ~22% noise on environmental-science-heavy BGE rows (BGE 148 II 36); ~16% noise on early-corpus published-BGE rows; ~0-2% noise on unpublished-BGer rows (mid-corpus and tail).

### 2.2 Continuation-fragment detector

A row is a continuation fragment iff ANY of:
- Body opens lowercase.
- Body opens with closing bracket `)` or `]`.
- Body opens with `%`, `m`, `ha`, `Uhr`, `GWh`, year-followed-by-comma.
- Body opens with `Nr.`, `Jahrhundert`.
- Body opens with `[Date].`.
- E-id is artifact AND body opens with sentence-fragment.

### 2.3 Decision-closure detector

Row is `is_signature_boilerplate=true` iff body contains all of:
- City+date pattern: `(Lausanne|Luzern|Lucerne|Losanna), \d{1,2}\.?\s+(Januar|Februar|…|Dezember|janvier|…|décembre|gennaio|…|dicembre)\s+\d{4}`
- Chamber-self-naming: `Im Namen der … Abteilung` / `Au nom de la … Cour` / `In nome della … Corte`
- President+clerk pattern: `Der Präsident: X, Der Gerichtsschreiber: Y` / `Le Président : X, Le Greffier : Y` / `Il Presidente: X, Il Cancelliere: Y`.

This is the **deterministic end-of-decision marker** for chunking.

### 2.4 Bridge-paragraph / header-only filter

Filter out rows where:
- `text_length_chars < 20` AND body matches `^\d+(\.\d+)*\.?$` (just a numbered header).
- `text_length_chars < 10` (heuristic floor for any embedding).

### 2.5 Verb-ownership classifier

Apply in order on the opening sentence:
1. If subject is statute (`Art. N CODE`) AND verb in doctrinal-verb-pool → `court`.
2. If subject is `Der Haftgrund / Der Anspruch / Die Beschwerde / Le recours / Il ricorso / La giurisprudenza` AND verb in doctrinal-verb-pool → `court`.
3. If subject is `Der Beschwerdeführer / Le recourant / Il ricorrente / Die Beschwerdegegnerin / etc.` AND verb in argumentative-pool (`rügt`, `bestreitet`, `macht geltend`, `fait valoir`, `soutient`, `sostiene`, `contesta`) → `appellant`.
4. If subject is `Die Vorinstanz / La cour cantonale / L'autorità precedente` → `lower_court`.
5. Else → `unclear`.

---

## 3. Cross-lingual canonical triples (the full table)

The MASTER ALIAS TABLE that the schema's `*_triples_used` fields reference:

### 3.1 Reporter / decision-type triples

| DE | FR | IT | EN |
|---|---|---|---|
| BGE | ATF | DTF | published BGer judgment (official reporter) |
| Urteil | arrêt | sentenza | unpublished BGer judgment |
| BVGer | TAF | TAF | Federal Administrative Court |
| BStGer | TPF | TPF | Federal Criminal Court |
| BPatG | TFB | TFB | Federal Patent Court |
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
| i.V.m. (in Verbindung mit) | en lien avec | cum (Latin) / in relazione con |
| im Übrigen | au demeurant / du reste | del resto / per giunta |
| im vorliegenden Fall | en l'espèce / en l'occurrence | nella fattispecie / in concreto |
| zu Recht | à juste titre / à raison | a ragione / rettamente |
| zu Unrecht | à tort | a torto |
| nach Auffassung des BGer | de l'avis du TF | a mente di questa Corte / L'Alta Corte ha ritenuto |
| im Übrigen / überdies | à titre superfétatoire | a titolo abbondanziale / di transenna |
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

### 3.5 Publication / collection triples

| DE | FR | IT |
|---|---|---|
| BBl (Bundesblatt) | FF (Feuille fédérale) | FF (Foglio federale) |
| AS (Amtliche Sammlung) | RO (Recueil officiel) | RU (Raccolta ufficiale) |
| SR (Systematische Sammlung) | RS (Recueil systématique) | RS (Raccolta sistematica) |
| AB (Amtliches Bulletin) | BO (Bulletin officiel) | BU (Bollettino ufficiale) |
| BS (Bereinigte Sammlung) | (pre-1948 collections vary) | (varies) |

### 3.6 Cantonal collection prefixes (single-language, kept verbatim)

| Canton | Prefix | Numbering form |
|---|---|---|
| BE | BSG | dotted decimal (`BSG 153.01`) |
| ZH | LS | dotted decimal (`LS 175.2`) |
| VD | RSV | dotted decimal (`RSV 173.36.5.1`) |
| GE | RSG / rs/GE | letter+number+slash+number (`RSG E 5/10`, `rs/GE L 5 05`) |
| FR | RSF / SGF | dotted decimal (`RSF 150.1`) |
| NE | RSN | dotted decimal (`RSN 152.130`) |
| TI | RL / RL/TI | dotted hierarchical (`RL 7.1.1.1`, `RL/TI 163.100`) |
| SG | sGS | dotted decimal (`sGS 213.1`) |
| GR | BR | dotted decimal (`BR 801.100`) |
| TG | RB | dotted decimal (`RB 170.1`) |
| SZ | SRSZ | dotted decimal (`SRSZ 400.100`) |
| AG | (no standard prefix, just full statute name) | |
| BS | (no standard prefix) | |
| BL | (no standard prefix) | |
| UR | (no standard prefix) | |

### 3.7 Cantonal-court docket prefixes

| Canton/Court | Prefix |
|---|---|
| VD Cour de droit administratif et public | `BO.YYYY.NNNN`, `PS.YYYY.NNNN` |
| GE Cour de justice administrative | `ATA/NNN/YYYY` |
| ZH Verwaltungsgericht | `VB.YYYY.NNNNN` |
| ZH Steuerrekursgericht | `1 ST.YYYY.NN` |
| BVGer | `B-NNNN/YYYY` (not yet observed in source paragraphs, only as cited authority) |

---

## 4. Storage schema (suggested)

```sql
CREATE TABLE court_consideration_paragraph (
    -- Identity
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
    publication_status TEXT,           -- bge_published / unpublished / zur_publikation_vorgesehen
    -- Language
    language TEXT,
    embedded_languages TEXT[],
    has_trilingual_gloss BOOL,
    language_version_equality_invoked BOOL,
    quoted_foreign_statute_languages TEXT[],
    -- Paragraph structure
    erwagung_id_raw TEXT,
    erwagung_id_validated TEXT,
    erwagung_id_artifact_class TEXT,
    parent_section TEXT,
    depth INT,
    sachverhalt_letter TEXT,
    is_continuation_fragment BOOL,
    is_subsection_header_only BOOL,
    -- Paragraph role
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
    -- Template scoring
    has_element_1_statutory_anchor BOOL,
    has_element_2_paraphrase BOOL,
    has_element_3_teleology BOOL,
    has_element_4_positive_limb BOOL,
    has_element_5_negative_limb BOOL,
    has_element_6_interpretive_factors BOOL,
    has_element_7_authority_chain BOOL,
    template_score INT,
    opener_verb_owner TEXT,
    -- Statute refs
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
    -- Authority graph
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
    -- Cross-lingual alias hits
    statute_alias_triples_used TEXT[],
    pinpoint_form_triples_used TEXT[],
    older_de_alias_used TEXT[],
    older_docket_format_used BOOL,
    swiss_it_vs_foreign_it TEXT,
    loccit_or_aao TEXT,
    -- Doctrinal-test invocation
    named_doctrine TEXT[],
    doctrinal_test_invoked TEXT[],
    saving_construction_invoked BOOL,
    review_intensity TEXT,
    holding_polarity TEXT,
    is_remand BOOL,
    art_106_qualified_complaint_required BOOL,
    is_obiter_dictum BOOL,
    -- Idioms & formulae
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
    -- Quality
    is_csv_parse_artifact BOOL,
    has_encoding_artifacts BOOL,
    embedded_page_anchors TEXT[],
    internal_navigation_refs TEXT[],
    has_quoted_block BOOL,
    has_embedded_list BOOL,
    text_length_chars INT,
    materialien_density TEXT,
    -- Composite
    gold_candidate_score FLOAT,
    -- Raw text (for fallback)
    text_raw TEXT,
    text_clean TEXT  -- with page-anchors stripped
);
```

---

## 5. Chunking-strategy recommendations

1. **Cluster rows by `bge_id` / `docket_no`** — paragraphs of one decision are always contiguous in the CSV (verified across head/mid/tail).
2. **Detect decision boundaries** via `is_signature_boilerplate=true` (one per decision, last paragraph).
3. **Detect Erwägung→Dispositiv transition** via E-number reset: when E-id drops from a higher value back to `E. 1` within the same docket, you've crossed from substantive Erwägungen into Dispositiv.
4. **Drop `is_subsection_header_only`** rows from the retrievable index.
5. **Drop pure `is_cost_dispositif` / `is_signature_boilerplate` / `is_admissibility_recital`** rows.
6. **Merge `is_continuation_fragment` rows back** with their preceding parent row.
7. **Re-knit citation-noise rows**: when `erwagung_id_artifact_class != clean`, the row's content belongs to the previous valid E-id.
8. **Preserve `quoted_statute_block`** rows but tag them as embedded-statute, not court-doctrine.
9. **Preserve `embedded_page_anchors`** in `text_raw` but strip from `text_clean` (they're publisher artefacts, not citations).
10. **Multi-row decisions on one Erwägung** (e.g. when CSV splitter cut a long paragraph mid-sentence): use `erwagung_id_validated` + body-adjacency to re-merge.

---

## 6. Stage-2 retrieval implications for Qwen3-8B

For the cascade dossier, each candidate row passed to Qwen3-8B should carry:

```
[CITATION] BGE 137 IV 122 E. 4.2 (DE, IV-chamber, criminal, leading-case)
[VALIDATION] erwagung_id_validated=4.2 / is_csv_parse_artifact=false
[ROLE] legal_standard / template_score=7 / opener_verb_owner=court
[STATUTES] [{sr=312.0, code=StPO/CPP/CPP, art=221, abs=1, lit=b, version=current}]
[ECHOES] cites: [BGE 132 I 21 E. 3.2, BGE 137 IV 122 E. 4.2]; cited_by_in_pool: 8 court paragraphs
[DOCTRINE] named=[]; test=verhaeltnismaessigkeit_three_prong; holding=holds_up
[QUERY_MATCH] statute_target_intersection=3; doctrinal_sub_question=Kollusionsgefahr
[TEXT_CLEAN] "Gemäss Art. 221 Abs. 1 lit. b i.V.m. Art. 237 Abs. 1 StPO ist Untersuchungshaft … (BGE 137 IV 122 S. 128)"
```

The Qwen3-8B judge then applies the 8-step decision DAG from `research/per_query_court_gold_anatomy_2026-05-21.md` §6, with the schema-extracted fields replacing the regex-based template-score with the explicit `template_score` field.

---

## 7. Open questions for further investigation

1. **IT row prevalence corpus-wide**: only 8/8 agents found 25-50 IT rows; we did not measure IT % across all 2.4M. The known Ticino-cantonal-court output suggests IT is ~3-5% of the corpus, but this should be measured.
2. **`E.` resets for Dispositiv**: confirmed for post-2024 BGer in the tail slice but not validated mid-corpus. May vary by chamber.
3. **`docket_era` precise breakpoints**: the 1990 / 2007 / 2024 thresholds are approximate based on observed dockets; the precise BGer reorganization dates should be confirmed.
4. **`presiding_judge` extraction reliability**: only confirmed in tail rows. Pre-2010 BGer used different signature conventions.
5. **`gold_candidate_score` weights**: the composite formula needs calibration against the 10 val queries' gold court-citation distributions from `research/per_query_court_gold_anatomy_2026-05-21.md`.

---

## 8. Provenance — 16 discovery agents

**Round 1** (8 agents, rows 2-1001):
- Slice 1: rows 2-126 (DE, BGE 139/142/136/145 I-series public-law)
- Slice 2: rows 127-251 (DE, health-insurance + Covid)
- Slice 3: rows 252-376 (DE, Schengen + religion + planning)
- Slice 4: rows 377-501 (DE, tax + religion + procurement)
- Slice 5: rows 502-626 (DE, headscarf + tax + cartel)
- Slice 6: rows 627-751 (DE, cartel + transparency)
- Slice 7: rows 752-876 (DE, police + water-rights + Vollgeld)
- Slice 8: rows 877-1001 (DE+FR, Vaud detention + Reneja)

**Round 2** (8 agents, targeted):
- IT slice 1: rows 1800-1925 (IT, Ticino LAN/TI + LCPubb)
- IT slice 2: rows 3450-3575 (IT, BGE 139 II 7 workplace surveillance)
- FR slice 1: rows 1390-1515 (FR-dominant, BGE 145 I 73 + others)
- FR slice 2: rows 2500-2625 (DE+FR, Grenchenberg windpark + Jura géothermie)
- Deep ~500K (DE+FR, post-2024 unpublished BGer)
- Deep ~1.5M (FR/DE/IT balanced, 2004-2006 pre-BGG dockets)
- Deep ~2.4M (DE+FR+IT, post-2024 BGer + 7B_ chamber introduction)
- IT/FR mixed ~11300 (BGE 145 IV 294 Leonardo + cultural-property cooperation)

Total: ~2,000 paragraphs read across the corpus. Cross-validated patterns hold uniformly except for content-driven citation-noise (BGE rows higher, BGer rows lower) and language-distribution drift (Round 1 monolingual DE → mid-corpus balanced DE/FR + small IT → tail moderate DE/FR/IT).
