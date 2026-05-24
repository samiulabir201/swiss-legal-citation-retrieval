# val_001 — Enrichment field cardinalities & judge-filter feasibility

_Generated 2026-05-09. One linear pass over both LLM enrichment files; counts of unique values per field, plus the actual value lists or top-K._

---

## 1. Coverage check (revisited)

| Group | Citations | Source | Has rich categorical fields? |
|---|---:|---|---|
| Group A: LLM-enriched gold | 32 (19 law + 13 court) | `law_llm_descriptors_*` + `court_llm_descriptors_*` | **Yes** — all of `legal_area` / `topic` / `concepts_en` / `terms_*` / `provision_role_llm` etc. |
| Group B: static-only gold | 10 court (1B_/7B_ docket) | `court_authority_cards_v5_unified.jsonl` (`enrichment_source = "static"`) | Partial — only `legal_area` / `concepts_en` / `terms_original` / `statute_anchors` / `paragraph_role` / `authority_role`. Missing `topic` / `micro_topic` / `legal_domain_path` / `english_summary`. |

Per the user instruction (_"the ones we can retrieve from jsonl, lets focus on them"_) the analysis below is over the 32 Group-A gold and the categorical fields populated by the LLM. Group B is parked for a separate handler.

---

## 2. Cardinality summary tables

### Cardinality summary for **law (`law_llm_descriptors_0000000_all.jsonl`)** (173,033 records)

| Field | Unique values | Coverage (% of records w/ value) | Mean values per record (for list fields) |
|---|---:|---:|---:|
| `specificity_score` | 10 | 100.0% | - |
| `provision_role_llm` | 14 | 100.0% | - |
| `defined_terms [list-item.term]` | 23,719 | - | 0.28 |
| `exceptions_or_limitations [list-item]` | 31,170 | - | 0.19 |
| `addressees [list-item]` | 32,225 | - | 2.24 |
| `sanctions_or_consequences [list-item]` | 34,554 | - | 0.33 |
| `defined_terms [list-item.definition]` | 36,533 | - | 0.28 |
| `legal_rule` | 81,321 | 49.8% | - |
| `legal_question` | 82,819 | 51.4% | - |
| `terms_de_to_en [list-item.de]` | 137,083 | - | 3.63 |
| `terms_de_to_en [list-item.en]` | 158,358 | - | 3.63 |
| `english_summary` | 161,026 | 100.0% | - |
| `concepts_en [list-item]` | 189,380 | - | 5.76 |
| `applicability_conditions [list-item]` | 414,624 | - | 2.84 |


### Cardinality summary for **court (`court_llm_descriptors_0000000_all.jsonl`)** (363,258 records)

| Field | Unique values | Coverage (% of records w/ value) | Mean values per record (for list fields) |
|---|---:|---:|---:|
| `_descriptor_error` | 1 | 0.0% | - |
| `paragraph_role` | 10 | 100.0% | - |
| `specificity_score` | 12 | 100.0% | - |
| `doctrinal_rule` | 112 | 0.0% | - |
| `legal_test` | 1,015 | 0.3% | - |
| `legal_area` | 2,916 | 100.0% | - |
| `primary_domain` | 11,719 | 100.0% | - |
| `authority_role [list-item]` | 18,556 | - | 1.62 |
| `legal_domain_path [list-item]` | 48,864 | - | 2.97 |
| `procedural_context` | 50,076 | 99.8% | - |
| `secondary_domain` | 68,911 | 100.0% | - |
| `topic` | 69,915 | 100.0% | - |
| `concepts_en [list-item]` | 116,517 | - | 4.30 |
| `fact_pattern_tags [list-item]` | 152,787 | - | 3.56 |
| `subtopic` | 197,998 | 100.0% | - |
| `terms_original [list-item]` | 280,271 | - | 4.22 |
| `micro_topic` | 297,752 | 100.0% | - |


---

## 3. Full unique-value lists per field

Fields with ≤ 80 unique values get the full list; others get top-40 + a tail count.

### Unique values per field — law

#### `addressees [list-item]` — 32,225 unique values

_Top 40 (of 32,225):_

- `supervisory authority` — 28,996
- `competent cantonal authority` — 27,995
- `natural person` — 24,199
- `federal authority` — 18,206
- `legal entity / undertaking` — 14,728
- `Federal Council` — 9,289
- `cantonal authority` — 8,346
- `market participant` — 6,512
- `employer` — 5,825
- `court` — 5,019
- `legal entity` — 4,931
- `Competent cantonal authority` — 4,700
- `employee` — 3,922
- `Federal authority` — 3,312
- `competent authority` — 2,786
- `federal department or office` — 2,744
- `public authority` — 1,936
- `federal department` — 1,887
- `taxpayer` — 1,867
- `financial institution` — 1,858
- `training provider` — 1,822
- `manufacturer` — 1,779
- `educational institution` — 1,685
- `Supervisory authority` — 1,644
- `applicant` — 1,622
- `tax authority` — 1,513
- `regulatory authority` — 1,433
- `cantons` — 1,370
- `cantonal authorities` — 1,342
- `federal government` — 1,278
- `public prosecutor` — 1,246
- `Federal government` — 1,148
- `insured person` — 1,136
- `learner` — 1,108
- `federal office` — 1,086
- `federal authorities` — 961
- `Federal authorities` — 943
- `military personnel` — 940
- `Cantons` — 936
- `data controller` — 930
- _(... 32,185 more values, total tail occurrences = 182,548)_

#### `applicability_conditions [list-item]` — 414,624 unique values

_Top 40 (of 414,624):_

- `Federal Council has authority` — 531
- `Qualification procedures` — 233
- `Training sites` — 187
- `The law is subject to a facultative referendum` — 181
- `Intentional act` — 174
- `Federal Council's authority` — 173
- `Training providers are bound` — 169
- `Cantons have authority` — 164
- `Federal Council's discretion` — 162
- `Vocational trainer is responsible` — 158
- `The commission is established` — 134
- `Completed vocational training under this ordinance` — 133
- `Vocational training commencement` — 130
- `Business has trained trainees with above-average success` — 127
- `Employment relationship exists` — 126
- `Training in professional practice` — 126
- `Qualification procedure is in effect` — 117
- `General education instruction applies` — 116
- `Annual reporting requirement` — 114
- `Notification must be in writing` — 114
- `Businesses limited to one trainee` — 110
- `Cantonal authorities` — 108
- `Every five years` — 106
- `Acquiring action competencies` — 106
- `Ordinance enters into force` — 104
- `Vocational training context` — 103
- `Tasks are listed in the curriculum annex` — 101
- `Federal Council has discretion` — 97
- `Providers of cross-company courses` — 97
- `Commission established under this article` — 96
- `Municipalities` — 94
- `Cantons` — 94
- `A qualification area must be repeated` — 94
- `Assessment is conducted by exam experts` — 92
- `Coordination of training content` — 91
- `Federal Council's authority to issue implementing regulations` — 90
- `Completion of a qualification procedure` — 90
- `Training occurs at all training sites` — 88
- `Successful outcome of the procedure` — 86
- `Cantonal authority considers the case` — 85
- _(... 414,584 more values, total tail occurrences = 485,579)_

#### `concepts_en [list-item]` — 189,380 unique values

_Top 40 (of 189,380):_

- `Regulatory compliance` — 8,114
- `regulatory compliance` — 5,369
- `Administrative procedure` — 4,976
- `vocational training` — 4,591
- `administrative procedure` — 3,848
- `supervisory authority` — 3,445
- `Cantonal authority` — 2,816
- `Federal Council` — 2,784
- `Competent authority` — 2,720
- `Authorization` — 2,515
- `entry into force` — 2,475
- `commencement` — 2,416
- `Administrative law` — 2,404
- `Data protection` — 2,265
- `Administrative authority` — 2,200
- `Supervisory authority` — 2,195
- `compensation` — 2,146
- `regulatory framework` — 2,109
- `Federal authority` — 2,083
- `Public administration` — 1,973
- `cantonal authority` — 1,917
- `public administration` — 1,870
- `jurisdiction` — 1,868
- `competent authority` — 1,694
- `employment law` — 1,550
- `data protection` — 1,533
- `administrative law` — 1,493
- `Administrative discretion` — 1,428
- `supervision` — 1,355
- `regulation` — 1,347
- `federal authority` — 1,277
- `compliance` — 1,246
- `reporting obligation` — 1,245
- `regulatory authority` — 1,226
- `authorization` — 1,212
- `competence` — 1,183
- `criminal liability` — 1,171
- `administrative authority` — 1,123
- `Administrative regulation` — 1,106
- `Regulatory oversight` — 1,104
- _(... 189,340 more values, total tail occurrences = 905,513)_

#### `defined_terms [list-item.definition]` — 36,533 unique values

_Top 40 (of 36,533):_

- `Federal Vocational Qualification Certificate` — 148
- `placing on the market` — 139
- `Occupational Safety and Health Ordinance` — 118
- `Swiss Financial Market Supervisory Authority` — 103
- `vocational trainer` — 93
- `federal vocational certificate` — 77
- `vocational training` — 76
- `training plan` — 74
- `Federal Office of Public Health` — 74
- `Federal Veterinary Office` — 73
- `Swiss Federal Institute of Technology` — 69
- `qualification procedure with final examination` — 69
- `Federal Vocational Baccalaureate` — 66
- `Federal Council` — 61
- `federal qualification certificate` — 59
- `individual undergoing vocational training` — 59
- `federal vocational qualification certificate` — 57
- `trainees` — 56
- `permit / authorisation` — 55
- `personal data` — 52
- `Training curriculum` — 51
- `Swiss Code of Obligations` — 51
- `Federal Office of Civil Aviation` — 47
- `permit` — 45
- `qualified vocational trainer` — 42
- `qualification procedure` — 42
- `Swiss Federal Tax Administration` — 40
- `Federal Department of Foreign Affairs` — 38
- `Federal Office of Communications` — 37
- `Cantonal council` — 36
- `competent organization of the labor world` — 36
- `Federal Tax Administration` — 36
- `applicant` — 35
- `Federal Office of Agriculture` — 35
- `Federal Office of Energy` — 34
- `implementing regulations` — 34
- `Specialist with relevant vocational qualification` — 34
- `financial intermediary` — 34
- `employee` — 33
- `compensation` — 32
- _(... 36,493 more values, total tail occurrences = 46,881)_

#### `defined_terms [list-item.term]` — 23,719 unique values

_Top 40 (of 23,719):_

- `Bildungsplan` — 319
- `EFZ` — 232
- `EBA` — 201
- `eidgenössisches Fähigkeitszeugnis` — 191
- `Bewilligung` — 176
- `Inverkehrbringen` — 143
- `ArGV 5` — 141
- `FINMA` — 136
- `Qualifikationsverfahren` — 125
- `Berufsbildnerin` — 117
- `BLV` — 97
- `Gesuch` — 96
- `Fachkraft` — 96
- `BAG` — 96
- `SBFI` — 93
- `Berufliche Grundbildung` — 92
- `Berufsbildner` — 91
- `SEM` — 89
- `Lernende` — 89
- `ESTV` — 85
- `BLW` — 83
- `Personendaten` — 82
- `Anhang 1` — 81
- `Qualifikationsverfahren mit Abschlussprüfung` — 80
- `BAZG` — 80
- `Bundesrat` — 76
- `PUBLICA` — 75
- `AHV-Nummer` — 74
- `Abschlussprüfung` — 73
- `Handlungskompetenzen` — 72
- `SECO` — 66
- `überbetriebliche Kurse` — 66
- `EDA` — 61
- `SICAV` — 61
- `SchKG` — 60
- `Beiträge` — 59
- `Anhang 2` — 59
- `Finanzintermediär` — 59
- `KVG` — 55
- `Zulassung` — 54
- _(... 23,679 more values, total tail occurrences = 45,150)_

#### `english_summary` — 161,026 unique values

_Top 40 (of 161,026):_

- `This law is subject to a facultative referendum.` — 198
- `These provisions and recommendations are taught at all training sites and considered in qualificatio...` — 172
- `At least two exam experts assess performance in each qualification area.` — 147
- `The commission constitutes itself.` — 132
- `If a qualification area must be repeated, it must be repeated in its entirety.` — 110
- `The trainer must inspect and sign the training documentation at least once per semester and discuss ...` — 91
- `The trainee is taught knowledge about sustainable development, including balancing societal, ecologi...` — 91
- `This article specifies the effective date of the ordinance.` — 90
- `The qualification procedure requires demonstrating that the competencies from Article 4 have been ac...` — 90
- `The Federal Council issues implementing regulations.` — 80
- `The start of vocational training aligns with the school year of the relevant vocational school.` — 77
- `If training objectives are not met despite agreed measures or if training success is at risk, the vo...` — 76
- `This law is subject to optional referendum.` — 71
- `The vocational school documents the learner's performance in taught competencies and general educati...` — 65
- `If the final exam is retaken without repeating vocational knowledge instruction, the previous experi...` — 61
- `The language regions must be adequately represented.` — 59
- `This article defines the calculation method for the experience grade in vocational training.` — 57
- `This article defines the structure of professional basic training objectives and requirements as com...` — 56
- `This article refers to the general education instruction regulations for vocational training.` — 55
- `This article states that competency certificates are expressed in grades and contribute to the exper...` — 55
- `This article sets the effective date for qualification procedures, certificates, and titles regulati...` — 55
- `This article defines the competencies that include professional, methodological, social, and self-co...` — 54
- `Cantonal authorities have continuous access to training courses.` — 54
- `A second trainee may start training if the first trainee enters the final year of vocational trainin...` — 54
- `The trainer and learner agree on measures to achieve educational goals and set deadlines. They docum...` — 53
- `The cantonal authority may grant a permit for a business to exceed the maximum number of trainees if...` — 52
- `The start of vocational training is determined by the school year of the competent vocational school...` — 52
- `This article allows trainees to be used for specific tasks listed in the curriculum annex, deviating...` — 51
- `This article requires collaboration between all training sites in building practical competencies. T...` — 50
- `The instruction language is generally the language of the school location.` — 50
- `A second trainee may start training if the first trainee enters the final year of vocational basic t...` — 49
- `A candidate's experience grade is waived if they have acquired required competencies outside regulat...` — 49
- `Bilingual instruction in the language of the school location and another regional language or Englis...` — 47
- `A specialist is defined as someone with a federal vocational qualification or equivalent in the lear...` — 46
- `Cantons may transfer oversight of cross-employer courses to other providers if quality or implementa...` — 46
- `This article refers to the general education curriculum regulations for vocational training.` — 46
- `If the final exam is retaken without re-attending cross-enterprise courses, the previous grade is re...` — 45
- `The educational plan includes a directory of instruments for ensuring and implementing vocational tr...` — 43
- `The learner maintains a learning documentation during vocational training, recording all essential t...` — 43
- `A specialist is someone with a federal vocational qualification or equivalent in the learner's field...` — 43
- _(... 160,986 more values, total tail occurrences = 170,218)_

#### `exceptions_or_limitations [list-item]` — 31,170 unique values

_Top 40 (of 31,170):_

- `Permission is not automatic` — 47
- `Conditions must be met` — 36
- `Only in special cases` — 29
- `Only applies to businesses with training capacity for one trainee` — 29
- `Cantonal authority discretion applies` — 27
- `Conditions must be met for exceptional cases` — 24
- `Cantons may allow other instruction languages` — 23
- `Subject to Article 4a Abs. 15 ArGV 5 guidelines` — 23
- `Training must be completed under previous rules` — 22
- `Cantons have discretion to allow other languages` — 22
- `Permit is not automatic` — 21
- `Only applies to specified tasks in the curriculum annex` — 17
- `Subject to the provisions of Article 4a Abs. 17 ArGV 5` — 16
- `Limited to exceptional cases` — 14
- `Subject to ArGV 5 provisions` — 14
- `Cantons may permit additional instruction languages` — 13
- `Subject to Article 20 Abs. 1` — 13
- `Subject to Article 4a Abs. 17 ArGV 5 guidelines` — 13
- `Final semester excludes cross-company courses` — 13
- `Exceptions not specified in the text` — 12
- `Measures must be proportionate` — 12
- `No exception explicitly stated` — 11
- `No explicit exceptions listed` — 10
- `Experience grade is excluded in this case` — 10
- `international organizations` — 10
- `Authorization is exceptional` — 10
- `No exceptions explicitly stated` — 9
- `Subject to cantonal regulations` — 9
- `Emergency situations` — 8
- `Must follow ArGV 5 guidelines` — 8
- `Previous grade retained upon retake without re-attending cross-enterprise courses` — 8
- `Only applies to specified tasks` — 8
- `New grades only count for experience grade upon retake` — 8
- `Only in exceptional cases` — 7
- `Not applicable to all cases` — 7
- `Only for specified tasks` — 7
- `Must be granted by cantonal authority` — 7
- `Only new grades count for experience grade upon retake of last two assessed cross-enterprise courses` — 7
- `Previous grade is retained if no re-attendance` — 7
- `Subject to Article 21 Abs. 1` — 7
- _(... 31,130 more values, total tail occurrences = 32,444)_

#### `legal_question` — 82,819 unique values

_Top 40 (of 82,819):_

- `What competencies must be demonstrated in the qualification procedure?` — 111
- `What measures must trainer and learner agree on for educational goals?` — 85
- `What obligations do training providers have regarding safety and health regulations?` — 80
- `What obligations do training providers have regarding safety regulations?` — 79
- `When may a business train an additional apprentice?` — 78
- `Who may assess performance in a qualification area?` — 77
- `How is the grade for cross-enterprise courses determined?` — 69
- `When can a cantonal authority permit exceeding trainee limits?` — 66
- `What obligations does the trainer have regarding training documentation?` — 63
- `What must a learner document during vocational training?` — 62
- `What grades are required to pass the qualification procedure?` — 62
- `When does the experience grade not apply?` — 56
- `Who assesses performance in qualification areas?` — 55
- `When must a vocational trainer notify authorities about training risks?` — 54
- `How is the experience grade calculated in vocational training?` — 54
- `How are individual grades weighted in the overall grade calculation?` — 54
- `How is the experience grade calculated for vocational training?` — 54
- `What access rights do cantonal authorities have to training courses?` — 52
- `Who is eligible for qualification procedures under this ordinance?` — 51
- `Do competency certificates contribute to experience grade calculation?` — 48
- `What must be included in the training plan?` — 47
- `How are individual grades weighted in the overall assessment?` — 45
- `When may an additional trainee be trained?` — 44
- `May a second trainee start training when first trainee enters final year?` — 43
- `What is the default instruction language for vocational training?` — 42
- `What must be included in the vocational training plan?` — 42
- `What qualifies as a specialist under this provision?` — 42
- `When may cantons transfer cross-employer course oversight?` — 39
- `What tasks must be recorded in the learning documentation?` — 36
- `When does vocational training commence?` — 36
- `What entitlement arises from successful completion of a qualification procedure?` — 34
- `Can lesson numbers be adjusted between training years?` — 33
- `What grading criteria determine passing the qualification procedure?` — 33
- `What qualifications qualify as vocational trainers?` — 32
- `What safety measures are required for learners in hazardous conditions?` — 32
- `May second trainee start training when first trainee enters final year?` — 30
- `May external courses occur in final semester of vocational training?` — 29
- `How is the overall grade calculated in the final examination?` — 28
- `May business train learner with specified trainer employment?` — 27
- `How is the experience grade calculated upon retaking the final exam?` — 26
- _(... 82,779 more values, total tail occurrences = 86,925)_

#### `legal_rule` — 81,321 unique values

_Top 40 (of 81,321):_

- `Assessment of performance by at least two exam experts per qualification area` — 114
- `Learner must maintain learning documentation during vocational training.` — 97
- `Second trainee may commence training if first trainee enters final year` — 87
- `Competencies from Article 4 must be demonstrated in the qualification procedure.` — 84
- `Trainer must inspect and sign training documentation at least once per semester.` — 69
- `Cantonal authority may permit exceeding maximum trainee numbers under specific conditions.` — 62
- `Grade calculation for cross-enterprise courses based on competency certificates` — 61
- `Specialist defined as having federal vocational qualification or equivalent in learner's field` — 59
- `Experience grade is waived under specified conditions` — 55
- `Providers must provide and explain safety, health, and environmental protection regulations.` — 53
- `Trainer and learner must agree on measures and set deadlines for educational goals, documenting deci...` — 47
- `Qualification procedure with final examination is passed based on grades in practical work and overa...` — 47
- `Instruction language is generally the language of the school location` — 47
- `Vocational trainer must inform contract parties and cantonal authority in writing if training object...` — 46
- `Cantonal authorities have access to training courses.` — 45
- `Cantons may transfer cross-employer course oversight to other providers under certain conditions.` — 43
- `Cantonal authority may permit exceeding maximum trainees if trainees have above-average success over...` — 38
- `Grades are weighted as 50% practical work, 30% professional knowledge, and 20% general education.` — 38
- `Competency certificates are expressed in grades and contribute to experience grade calculation` — 38
- `Minor adjustments to lesson numbers allowed with cantonal authorities and workplace organizations if...` — 37
- `Providers must provide and explain safety, health, and environmental protection regulations to train...` — 37
- `Vocational trainer must inform contract parties and cantonal authority in writing if training object...` — 35
- `Trainer and learner must agree on measures and deadlines to achieve educational goals and document t...` — 34
- `Repeat candidates assessed under old and new rules upon written request` — 33
- `Learners must be trained, guided, and supervised for increased hazards` — 31
- `Prohibition of external courses in final semester of vocational training` — 29
- `Training participants must be trained, guided, and supervised according to increased hazards` — 29
- `Learner must maintain ongoing documentation of work, skills, and experiences.` — 29
- `Instruction language is school location's language; cantons may allow others` — 28
- `Competencies include professional, methodological, social, and self-competencies.` — 27
- `Business may train learner under specified trainer employment conditions` — 26
- `Qualification procedure with final exam is passed based on grades in practical work and overall grad...` — 24
- `Trainer must document learner's educational progress in a report at semester end` — 23
- `Providers must document learners' achievements and issue certificates at semester end.` — 23
- `Trainer must document learner's educational progress in a report at semester end.` — 21
- `Competency certificates are graded and contribute to experience grade calculation` — 21
- `Successful completion of a qualification procedure grants an EFZ.` — 20
- `Certificate grants right to use protected title` — 20
- `Trainer and learner agree on measures and deadlines for educational goals, documenting decisions.` — 20
- `Vocational training begins in the school year of the competent vocational school.` — 19
- _(... 81,281 more values, total tail occurrences = 84,554)_

#### `provision_role_llm` — 14 unique values

- `other` — 65,156
- `procedure` — 24,739
- `definition` — 24,432
- `duty` — 22,260
- `right_or_entitlement` — 9,330
- `scope` — 8,128
- `competence` — 4,611
- `data_reporting` — 3,517
- `purpose` — 3,229
- `transitional_or_commencement` — 3,088
- `prohibition` — 2,490
- `principle` — 1,349
- `sanction_or_penalty` — 689
- `fees_or_costs` — 15

#### `sanctions_or_consequences [list-item]` — 34,554 unique values

_Top 40 (of 34,554):_

- `Non-compliance may result in regulatory action` — 1,393
- `Non-compliance may result in administrative penalties` — 752
- `Non-compliance may lead to regulatory action` — 706
- `Administrative penalties` — 688
- `fine` — 517
- `Legal liability` — 464
- `Penalties for non-compliance` — 370
- `Non-compliance may lead to administrative penalties` — 264
- `Fine` — 252
- `imprisonment up to three years` — 251
- `Non-compliance may lead to regulatory scrutiny` — 184
- `Potential regulatory action` — 132
- `penalties for non-compliance` — 128
- `Disciplinary action` — 123
- `Non-compliance may result in penalties` — 122
- `Legal liability for non-compliance` — 120
- `Imprisonment up to three years` — 117
- `Revocation of permit` — 117
- `No explicit sanctions mentioned` — 113
- `criminal liability` — 111
- `Market withdrawal` — 111
- `Operational restrictions` — 99
- `administrative penalties` — 97
- `Financial liability` — 96
- `Administrative fines` — 89
- `Revocation of authorization` — 89
- `Criminal liability` — 88
- `No sanctions specified` — 85
- `Non-compliance may result in enforcement actions` — 85
- `Non-compliance may result in legal consequences` — 84
- `Potential legal liability` — 82
- `Non-compliance may result in administrative sanctions` — 80
- `Non-compliance risks` — 78
- `imprisonment up to five years` — 76
- `Non-compliance may lead to enforcement actions` — 73
- `Administrative penalties for non-compliance` — 70
- `non-compliance may result in regulatory action` — 68
- `non-compliance may lead to regulatory action` — 68
- `Civil liability` — 67
- `Potential administrative penalties` — 66
- _(... 34,514 more values, total tail occurrences = 47,813)_

#### `specificity_score` — 10 unique values

- `0.8` — 117,011
- `0.5` — 33,473
- `0.9` — 11,400
- `0.2` — 3,392
- `0.05` — 2,283
- `0.6` — 1,979
- `0.0` — 1,709
- `0.3` — 1,432
- `0.7` — 313
- `1.0` — 41

#### `terms_de_to_en [list-item.de]` — 137,083 unique values

_Top 40 (of 137,083):_

- `Bundesrat` — 5,158
- `Bewilligung` — 3,349
- `Inkrafttreten` — 2,757
- `Kantone` — 2,410
- `Gesuch` — 2,035
- `Massnahmen` — 1,873
- `Zuständigkeit` — 1,584
- `Qualifikationsverfahren` — 1,472
- `Verfügung` — 1,376
- `Inverkehrbringen` — 1,347
- `Verordnung` — 1,334
- `FINMA` — 1,309
- `Kanton` — 1,195
- `Aufsichtsbehörde` — 1,177
- `Handlungskompetenzen` — 1,120
- `Verfahren` — 1,082
- `Berufsbildnerin` — 1,053
- `BAZG` — 1,052
- `Erfahrungsnote` — 1,035
- `Arbeitgeber` — 1,019
- `Beiträge` — 996
- `zuständige Behörde` — 983
- `Bildungsplan` — 911
- `Bund` — 873
- `Behörde` — 820
- `Zustimmung` — 804
- `BAG` — 804
- `Anforderungen` — 801
- `Frist` — 794
- `Zulassung` — 771
- `Vorschriften` — 764
- `Beschwerde` — 764
- `BAFU` — 762
- `BLV` — 760
- `Betrieb` — 758
- `Antrag` — 747
- `BLW` — 741
- `Abschlussprüfung` — 733
- `Genehmigung` — 711
- `Personendaten` — 708
- _(... 137,043 more values, total tail occurrences = 576,510)_

#### `terms_de_to_en [list-item.en]` — 158,358 unique values

_Top 40 (of 158,358):_

- `Federal Council` — 5,435
- `entry into force` — 2,539
- `cantons` — 2,445
- `application` — 2,087
- `measures` — 1,814
- `permit` — 1,619
- `applicant` — 1,479
- `placing on the market` — 1,435
- `formal administrative order` — 1,379
- `request` — 1,371
- `competent authority` — 1,360
- `supervisory authority` — 1,322
- `compensation` — 1,242
- `permit / authorisation` — 1,233
- `approval` — 1,192
- `cantonal authority` — 1,181
- `fine` — 1,125
- `employer` — 1,125
- `jurisdiction` — 1,088
- `contributions` — 1,014
- `authorization` — 988
- `vocational trainer` — 988
- `experience grade` — 978
- `requirements` — 954
- `authority` — 910
- `vocational training` — 898
- `regulations` — 889
- `Swiss Financial Market Supervisory Authority` — 864
- `conditions` — 835
- `documents` — 835
- `Federal Office of Public Health` — 811
- `consent` — 806
- `ordinance` — 771
- `qualification procedure` — 771
- `personal data` — 758
- `registration` — 729
- `decision` — 728
- `costs` — 713
- `learner` — 675
- `employment relationship` — 665
- _(... 158,318 more values, total tail occurrences = 577,201)_


---

### Unique values per field — court

#### `_descriptor_error` — 1 unique values

- `ValueError('no balanced JSON object found: {\n  "legal_area": "Education Law",\n  "primary_domain": ...` — 1

#### `authority_role [list-item]` — 18,556 unique values

_Top 40 (of 18,556):_

- `Swiss Federal Court` — 40,432
- `Swiss law` — 34,998
- `Court` — 22,229
- `Bundesgericht` — 20,402
- `Swiss Federal Supreme Court` — 19,610
- `BGE` — 17,503
- `Appellate court` — 12,362
- `Cantonal court` — 11,735
- `Administrative court` — 8,982
- `court` — 7,914
- `Administrative authority` — 7,537
- `Federal court` — 7,287
- `Swiss Federal Tribunal` — 6,665
- `Case law` — 6,274
- `BGG` — 5,329
- `Legal commentary` — 5,278
- `Administrative law` — 4,855
- `Appellant` — 4,574
- `Constitution` — 4,556
- `Federal Tribunal` — 4,056
- `Applicant` — 3,942
- `Federal Supreme Court` — 3,931
- `Federal law` — 3,732
- `Judicial` — 3,547
- `tax authority` — 3,518
- `Judicial precedent` — 3,441
- `Cantonal Court` — 3,295
- `Constitutional court` — 3,272
- `Switzerland` — 3,225
- `Constitutional Court` — 3,198
- `Obergericht` — 3,052
- `Swiss Supreme Court` — 2,933
- `Tribunal fédéral` — 2,930
- `Lower court` — 2,890
- `Tax authority` — 2,873
- `Swiss civil law` — 2,829
- `Swiss cantonal court` — 2,742
- `Vorinstanz` — 2,685
- `Constitutional law` — 2,553
- `Federal Court` — 2,490
- _(... 18,516 more values, total tail occurrences = 273,926)_

#### `concepts_en [list-item]` — 116,517 unique values

_Top 40 (of 116,517):_

- `Administrative law` — 39,653
- `Judicial review` — 29,216
- `Appeal` — 18,569
- `Administrative procedure` — 14,068
- `Administrative review` — 13,017
- `Administrative appeal` — 11,468
- `Evidence` — 10,906
- `Compensation` — 9,022
- `Burden of proof` — 8,960
- `Administrative decision` — 7,944
- `Jurisdiction` — 7,833
- `Public law` — 7,732
- `Legal remedy` — 7,147
- `Criminal liability` — 6,774
- `Procedural fairness` — 6,064
- `Legal interpretation` — 5,936
- `Constitutional law` — 5,592
- `Admissibility` — 5,513
- `Proportionality` — 5,243
- `Criminal procedure` — 5,012
- `Administrative Law` — 4,982
- `Criminal law` — 4,837
- `Due process` — 4,825
- `Appeals` — 4,712
- `administrative law` — 4,709
- `Annulment` — 4,533
- `Legal standards` — 4,523
- `Social Security` — 4,505
- `Costs` — 4,362
- `Constitutional review` — 4,151
- `Procedural law` — 4,081
- `Notification` — 4,007
- `Regulatory compliance` — 3,994
- `Constitutional rights` — 3,951
- `Taxation` — 3,918
- `Revocation` — 3,910
- `Legal aid` — 3,904
- `Public interest` — 3,770
- `Contractual obligations` — 3,738
- `judicial review` — 3,599
- _(... 116,477 more values, total tail occurrences = 1,252,679)_

#### `doctrinal_rule` — 112 unique values

_Top 40 (of 112):_

- `Interpretation must consider purpose and context` — 7
- `Proceedings may be consolidated if they are closely related and based on similar factual and legal g...` — 3
- `Losses are tied to the taxpayer, not the independent business` — 2
- `Mental illness consequences qualify for dental coverage` — 2
- `Proportionality requires balancing public safety and personal liberty.` — 2
- `Duration of marriage affects spousal support obligation` — 2
- `Gesetzeswortlaut determines insurance coverage start` — 1
- `Cantonal decision must consider the unity of divorce judgment under new divorce law` — 1
- `Medical reports have probative value if comprehensive, based on thorough examination, consider the c...` — 1
- `Violation of fundamental rights may contravene public policy` — 1
- `New evidence only admissible if prior decision provides basis` — 1
- `Constitutional complaints must be precise and detailed, with evidence of constitutional rights viola...` — 1
- `Federal Tribunal must rely on prior facts unless they are manifestly incorrect or violate law.` — 1
- `Medical reports have proof value if comprehensive, based on thorough examination, consider claims, a...` — 1
- `Minusklassenentscheid requires deviation from wage system disadvantaging gender-specific functions.` — 1
- `Eingliederung vor Rente requires assessable reintegration measures` — 1
- `Restrictions on property rights require legal basis and must be proportionate.` — 1
- `Creation of pledge occurs at transfer of possession or written disposition contract` — 1
- `Sickness benefits take precedence over unemployment benefits` — 1
- `Competence objection must be raised before substantive defense.` — 1
- `Imprévisibilité required for noise damage indemnification` — 1
- `Concurrent sentences are calculated based on total duration` — 1
- `Art. 23 Abs. 4 zweiter Satz StHG applies sinngemäss to juridical persons` — 1
- `Constitutional complaints require reformative legal claims` — 1
- `Failure to provide immediate notice excludes delay extension` — 1
- `In case of conflicting statements, prosecution is preferred unless objective evidence exists.` — 1
- `Review requires specific allegations of legal violation` — 1
- `Private use exceeding 20% disqualifies VAT deduction` — 1
- `Non-recoverable legal disadvantage required for state liability` — 1
- `Prohibition of double taxation and fiscal discrimination` — 1
- `Interpretation by text and purpose` — 1
- `Art. 22 DBG is lex specialis over Art. 23 lit. b DBG` — 1
- `Therapeutic measures required over detention if risk reduction is probable` — 1
- `Compensation is calculated in two stages: first for material expropriation, then for formal expropri...` — 1
- `New evidence is admissible only if the prior instance's decision provides a basis for it.` — 1
- `In dubio pro duriore applies in authorization decisions` — 1
- `Revocation of measures when conditions no longer met` — 1
- `Factual findings may only be corrected if they are obviously wrong or based on legal violation.` — 1
- `Textual interpretation is primary, deviations allowed only on strong grounds.` — 1
- `Correction of factual findings is allowed only if they are obviously incorrect or based on legal vio...` — 1
- _(... 72 more values, total tail occurrences = 72)_

#### `fact_pattern_tags [list-item]` — 152,787 unique values

_Top 40 (of 152,787):_

- `Appeal` — 14,715
- `Administrative law` — 14,436
- `Judicial review` — 14,381
- `Administrative appeal` — 13,526
- `Administrative decision` — 13,089
- `Swiss law` — 7,825
- `Switzerland` — 7,077
- `Administrative review` — 6,882
- `Evidence` — 5,688
- `Compensation` — 5,651
- `Burden of proof` — 4,814
- `Insurance` — 4,707
- `Cantonal court` — 4,650
- `Rejection` — 4,338
- `Administrative procedure` — 4,287
- `Public law` — 4,268
- `Legal interpretation` — 4,174
- `Federal court` — 4,071
- `Federal law` — 3,934
- `Revocation` — 3,898
- `Jurisdiction` — 3,746
- `Costs` — 3,646
- `Legal remedy` — 3,600
- `Notification` — 3,457
- `Proportionality` — 3,255
- `Reconsideration` — 3,200
- `Annulment` — 3,198
- `appeal` — 2,999
- `Constitutional law` — 2,977
- `Dismissal` — 2,905
- `Social Insurance` — 2,878
- `Appeal dismissed` — 2,850
- `Disability` — 2,823
- `Court decision` — 2,821
- `Criminal case` — 2,807
- `Admissibility` — 2,730
- `public law` — 2,702
- `Criminal law` — 2,694
- `Cantonal law` — 2,614
- `Constitutional review` — 2,549
- _(... 152,747 more values, total tail occurrences = 1,085,518)_

#### `legal_area` — 2,916 unique values

_Top 40 (of 2,916):_

- `Administrative law` — 59,167
- `Civil law` — 43,624
- `Criminal law` — 30,449
- `Civil procedure` — 23,578
- `Constitutional law` — 19,514
- `Family law` — 15,642
- `Tax law` — 15,018
- `Social Security Law` — 10,890
- `Social Security` — 6,262
- `Insurance law` — 5,983
- `Labor law` — 4,994
- `criminal law` — 4,670
- `Environmental law` — 4,261
- `Public Law` — 4,113
- `Public law` — 3,819
- `Intellectual Property` — 3,616
- `Administrative Law` — 3,133
- `Social security law` — 3,009
- `Corporate law` — 2,890
- `Private law` — 2,787
- `International law` — 2,739
- `criminal procedure` — 2,710
- `Criminal procedure` — 2,702
- `Human rights` — 2,652
- `Criminal Procedure` — 2,621
- `Social security` — 2,380
- `public law` — 2,374
- `Consumer protection` — 2,316
- `Property law` — 2,147
- `Private Law` — 2,081
- `Immigration law` — 1,958
- `Commercial law` — 1,715
- `Public procurement` — 1,603
- `International private law` — 1,537
- `Social Insurance Law` — 1,426
- `Immigration and asylum` — 1,268
- `Healthcare regulation` — 1,252
- `Civil Procedure` — 1,160
- `Real property law` — 1,124
- `Private international law` — 1,077
- _(... 2,876 more values, total tail occurrences = 56,993)_

#### `legal_domain_path [list-item]` — 48,864 unique values

_Top 40 (of 48,864):_

- `Administrative law` — 62,356
- `Criminal law` — 31,963
- `Swiss law` — 25,521
- `Civil law` — 21,092
- `Family law` — 17,311
- `Judicial review` — 16,900
- `Procedural law` — 15,934
- `Administrative procedure` — 15,703
- `Appeals` — 15,167
- `Constitution` — 14,128
- `Tax law` — 13,834
- `Social Security` — 13,796
- `Civil procedure` — 11,869
- `Contract law` — 11,662
- `Evidence` — 11,010
- `Public law` — 10,235
- `Administrative review` — 9,496
- `Insurance` — 9,371
- `Administrative Law` — 7,723
- `criminal law` — 7,431
- `Constitutional law` — 7,179
- `Criminal procedure` — 6,227
- `Social security` — 5,833
- `Administrative appeal` — 5,372
- `Labor law` — 5,125
- `Procedure` — 5,033
- `Federal law` — 4,811
- `Healthcare` — 4,712
- `International law` — 4,494
- `Swiss Law` — 4,465
- `Social Insurance` — 4,421
- `Costs` — 4,139
- `Sentencing` — 3,969
- `Immigration` — 3,951
- `Compensation` — 3,905
- `Public Law` — 3,816
- `Real estate` — 3,771
- `Environmental law` — 3,581
- `Jurisdiction` — 3,539
- `Taxation` — 3,518
- _(... 48,824 more values, total tail occurrences = 643,449)_

#### `legal_test` — 1,015 unique values

_Top 40 (of 1,015):_

- `Proportionality test` — 25
- `Proportionality test under Article 8 ECHR` — 13
- `Legitimate purpose and proportionality test` — 9
- `Reasonable legal differentiation test` — 8
- `Legitimate purpose and proportionality` — 7
- `Factual findings can be corrected if they are obviously wrong or based on legal violation` — 6
- `Objective appearance of bias` — 5
- `Four cumulative conditions for hidden profit distribution` — 5
- `Balancing of public and private interests` — 5
- `Necessity in a democratic society` — 5
- `Proportionality test under Article 8 EMRK` — 5
- `whether re-examination is necessary to affect the outcome` — 4
- `Proportionality test under Art. 8 ECHR` — 4
- `Facts must be established inexactely or violate the right` — 4
- `Confiscatory tax test` — 4
- `Abuse of discretion test` — 4
- `Reasonable prospects of success test` — 3
- `Decision is arbitrary if it is clearly untenable` — 3
- `Irreparable harm test` — 3
- `Likelihood of confusion test` — 3
- `Proportionality test under Art. 5 Abs. 2 BV` — 3
- `Proportionality test under Art. 36 Abs. 3 BV` — 3
- `Willkür in der Rechtsanwendung liegt vor, wenn der angefochtene Entscheid offensichtlich unhaltbar i...` — 3
- `Similar factual and legal grounds with same parties and similar legal questions.` — 3
- `Balancing of interests test` — 3
- `Willfulness of factual findings` — 3
- `Arbitrariness in factual findings` — 3
- `Decision is arbitrary if it violates clear legal norms or shocks the sense of justice` — 2
- `Serious and clear violation of fundamental rights` — 2
- `Factual findings may only be corrected if they are obviously wrong or based on legal violation` — 2
- `Willfulness in application of law` — 2
- `Whether re-examination is necessary to affect the outcome` — 2
- `Unreasonable exercise of discretion` — 2
- `Evidence is admissible if it was lawfully obtained by private individuals and meets the interest bal...` — 2
- `Serious and objective reasons for change of jurisprudence` — 2
- `Whether punishment is necessary and proportionate` — 2
- `Decision is arbitrary if manifestly unsustainable` — 2
- `Important regulation test` — 2
- `Whether the measure can achieve its goal` — 2
- `50% causation threshold for occupational disease` — 2
- _(... 975 more values, total tail occurrences = 1,035)_

#### `micro_topic` — 297,752 unique values

_Top 40 (of 297,752):_

- `Notification of judgment to parties and authorities` — 836
- `Compensation for damages in civil proceedings` — 817
- `Dismissal of insurance appeal by cantonal court` — 502
- `Appeal dismissed by social insurance court` — 453
- `Notification to parties and authorities` — 424
- `Dismissal of Social Insurance Appeal` — 371
- `Notification of Judgment to Parties and Authorities` — 339
- `Rejection of Social Insurance Claim` — 322
- `Notification of court decision to parties and authorities` — 314
- `Reimbursement of legal expenses from the federal court fund` — 292
- `Administrative appeal dismissed` — 283
- `Administrative appeal dismissed by administrative court` — 258
- `Judicial review of admissibility of legal remedies` — 252
- `Administrative review of a decision` — 252
- `Rejection of administrative appeal` — 223
- `Rejection of criminal appeal by appellate court` — 206
- `Reimbursement of legal representation costs` — 201
- `Grounds for judicial review under Art. 95 and 96 BGG` — 194
- `Grounds for appeal in civil cases` — 194
- `Compensation for legal representation` — 185
- `Notification of court decisions to parties and authorities` — 180
- `Correction of factual findings by federal court` — 176
- `Notification of court decisions to parties` — 173
- `Notification of counsel and court` — 172
- `Compensation for damages in court proceedings` — 168
- `Rejection of insurance claim appeal by cantonal court` — 167
- `Notification of Judgment to Parties and Court` — 166
- `Compensation for legal proceedings` — 162
- `Right to be heard in criminal proceedings` — 156
- `Rejection of Social Insurance Appeal` — 154
- `Compensation for legal representation costs in federal court proceedings` — 154
- `Grounds for complaint under Art. 95 BGG` — 147
- `Administrative appeal against a decision` — 147
- `Dismissal of administrative appeal` — 136
- `Rejection of insurance claim by insurer` — 133
- `Notification of court decisions` — 133
- `Public Law` — 132
- `Notification of Health Authorities` — 131
- `Filing of administrative appeal` — 126
- `Correction of factual findings by higher court` — 123
- _(... 297,712 more values, total tail occurrences = 353,300)_

#### `paragraph_role` — 10 unique values

- `reasoning` — 180,824
- `facts` — 114,575
- `legal_standard` — 30,731
- `procedural_history` — 14,714
- `disposition` — 10,849
- `notification` — 4,269
- `holding` — 2,816
- `neutral` — 2,489
- `citation` — 1,360
- `application` — 631

#### `primary_domain` — 11,719 unique values

_Top 40 (of 11,719):_

- `Administrative procedure` — 17,868
- `Contract law` — 12,517
- `Administrative law` — 11,987
- `Judicial review` — 9,845
- `Administrative review` — 8,411
- `Appeals` — 7,037
- `Evidence` — 7,034
- `Procedural law` — 6,329
- `Criminal procedure` — 6,158
- `Taxation` — 4,904
- `Property law` — 3,960
- `Criminal liability` — 3,647
- `Sentencing` — 3,561
- `Administrative Law` — 2,888
- `Social Insurance` — 2,601
- `Disability Benefits` — 2,566
- `Fundamental rights` — 2,474
- `Administrative remedies` — 2,440
- `Public law` — 2,362
- `Insurance law` — 2,338
- `Tax assessment` — 2,257
- `Procedure` — 2,205
- `Insurance claims` — 2,120
- `Jurisdiction` — 2,074
- `Constitutional review` — 2,057
- `administrative law` — 2,027
- `criminal procedure` — 1,987
- `Compensation` — 1,945
- `Child custody` — 1,831
- `Evidence law` — 1,789
- `Costs` — 1,770
- `Marriage` — 1,764
- `Personal injury` — 1,751
- `Administrative Procedure` — 1,731
- `Legal aid` — 1,712
- `Land use` — 1,675
- `Family law` — 1,624
- `Social Security` — 1,583
- `Appeals process` — 1,553
- `Civil procedure` — 1,511
- _(... 11,679 more values, total tail occurrences = 205,361)_

#### `procedural_context` — 50,076 unique values

_Top 40 (of 50,076):_

- `Administrative appeal` — 14,925
- `Legal interpretation` — 13,448
- `Legal reasoning` — 9,498
- `Appeal review` — 8,208
- `Appellate review` — 7,571
- `Administrative review` — 5,567
- `Administrative procedure` — 5,054
- `Administrative appeal process` — 4,127
- `Constitutional review` — 3,934
- `Judicial review` — 3,698
- `Constitutional interpretation` — 3,466
- `Appeal decision` — 3,385
- `Legal Interpretation` — 2,931
- `Appeal procedure` — 2,900
- `Court decision` — 2,861
- `Judicial interpretation` — 2,713
- `Administrative decision` — 2,708
- `Constitutional complaint` — 2,702
- `Administrative decision review` — 2,326
- `Appeal proceedings` — 2,148
- `Administrative court decision` — 2,118
- `Civil litigation` — 2,042
- `Cantonal court decision` — 1,790
- `Factual background` — 1,760
- `Criminal proceedings` — 1,660
- `Administrative appeal procedure` — 1,596
- `Legal definition` — 1,389
- `Legal analysis` — 1,377
- `Judicial reasoning` — 1,355
- `Legal standard` — 1,348
- `Federal court review` — 1,280
- `Court proceedings` — 1,195
- `Regulatory framework` — 1,097
- `Cantonal court ruling` — 1,084
- `Judicial decision` — 1,044
- `Filing of appeal` — 1,034
- `criminal proceedings` — 1,013
- `Tax dispute` — 1,000
- `Notification of judgment` — 979
- `Appeal reasoning` — 960
- _(... 50,036 more values, total tail occurrences = 231,320)_

#### `secondary_domain` — 68,911 unique values

_Top 40 (of 68,911):_

- `Administrative appeal` — 5,718
- `Administrative procedure` — 4,075
- `Administrative review` — 3,071
- `Burden of proof` — 2,933
- `Right to be heard` — 2,858
- `Judicial review` — 2,351
- `Grounds for appeal` — 2,304
- `Appeal procedure` — 2,131
- `Appeals and review` — 2,012
- `Grounds for review` — 1,790
- `Administrative law` — 1,660
- `Dispute Resolution` — 1,574
- `Evidence and proof` — 1,333
- `Administrative procedures` — 1,281
- `Annulment of decisions` — 1,268
- `Costs and expenses` — 1,224
- `Administrative remedies` — 1,216
- `Statute of limitations` — 1,188
- `Contractual obligations` — 1,160
- `judicial review` — 1,103
- `Breach of contract` — 1,093
- `Admissibility of evidence` — 1,074
- `Review of administrative decisions` — 1,040
- `Procedural fairness` — 921
- `Administrative Procedure` — 914
- `Insurance claims` — 880
- `Regulatory compliance` — 870
- `Judicial review of administrative acts` — 837
- `Civil liability` — 801
- `Constitutional complaints` — 783
- `Cost allocation` — 775
- `Administrative Law` — 765
- `Abuse of discretion` — 760
- `Public administration` — 738
- `Compensation for damages` — 733
- `Access to justice` — 727
- `Benefit Calculation` — 712
- `Legal interpretation` — 685
- `Administrative decision review` — 683
- `Judicial review of administrative decisions` — 678
- _(... 68,871 more values, total tail occurrences = 304,535)_

#### `specificity_score` — 12 unique values

- `0.8` — 283,121
- `0.2` — 32,742
- `0.7` — 14,439
- `0.9` — 12,099
- `0.5` — 7,941
- `0.0` — 5,394
- `0.3` — 5,342
- `0.6` — 1,704
- `0.85` — 266
- `0.1` — 206
- `0.75` — 2
- `0.4` — 2

#### `subtopic` — 197,998 unique values

_Top 40 (of 197,998):_

- `Grounds for appeal` — 2,124
- `Administrative appeal` — 1,441
- `Compensation for damages` — 1,306
- `Correction of factual findings` — 1,115
- `Revocation of administrative decision` — 992
- `Rejection of appeal` — 849
- `Notification to parties` — 836
- `Notification of judgment` — 825
- `Review of administrative decisions` — 773
- `Grounds for dismissal` — 763
- `Grounds for review` — 742
- `Dismissal of appeal` — 690
- `Appeal Dismissal` — 679
- `Right to be heard` — 624
- `Annulment of administrative decisions` — 623
- `Dismissal of appeals` — 587
- `Cancellation of administrative decision` — 552
- `Filing of appeal` — 512
- `Administrative review of decisions` — 446
- `Administrative appeal procedure` — 445
- `Appeal Rejected` — 444
- `Grounds for annulment` — 441
- `Legal representation costs` — 435
- `Grounds for judicial review` — 424
- `Appeal against administrative decision` — 424
- `Notification of Judgment` — 413
- `Administrative review` — 412
- `Annulment of administrative decision` — 406
- `Revocation of decision` — 396
- `Admissibility of new facts` — 393
- `Administrative decision review` — 378
- `Distribution of Judgments` — 373
- `Judicial review of admissibility` — 370
- `Admissibility of new evidence` — 366
- `Violation of federal law` — 354
- `Administrative appeal dismissed` — 342
- `Grounds for complaint` — 339
- `Statute of limitations` — 331
- `Administrative Law` — 328
- `Compensation for legal proceedings` — 315
- _(... 197,958 more values, total tail occurrences = 338,646)_

#### `terms_original [list-item]` — 280,271 unique values

_Top 40 (of 280,271):_

- `Beschwerde` — 30,230
- `BGE` — 14,283
- `Vorinstanz` — 12,724
- `Entscheid` — 11,177
- `BGG` — 11,137
- `Bundesgericht` — 9,387
- `Beschwerdeführer` — 7,888
- `Verwaltungsgericht` — 7,534
- `Urteil` — 6,758
- `Beschwerdeführerin` — 5,389
- `Entschädigung` — 5,366
- `Vernehmlassung` — 4,897
- `Verwaltungsgerichtsbeschwerde` — 4,872
- `Obergericht` — 4,658
- `Recours` — 4,407
- `Abweisung` — 3,656
- `Verfahren` — 3,622
- `Bundesrecht` — 3,550
- `Sozialversicherungsgericht` — 3,445
- `Beschwerdegegnerin` — 3,200
- `recours` — 3,124
- `Tribunal fédéral` — 3,097
- `Berufung` — 2,980
- `Kantonsgericht` — 2,975
- `StPO` — 2,834
- `Art. 95 BGG` — 2,518
- `Aufhebung` — 2,507
- `StGB` — 2,506
- `Beschwerde in öffentlich-rechtlichen Angelegenheiten` — 2,504
- `LTF` — 2,186
- `Beschluss` — 2,118
- `Rechtsmittel` — 2,045
- `Versicherungsgericht` — 2,032
- `Bundesgerichtskasse` — 1,995
- `ATF` — 1,958
- `Kosten` — 1,950
- `Beschwerdegegner` — 1,929
- `Art. 97 Abs. 1 BGG` — 1,896
- `Comunicazione` — 1,842
- `Art. 105 Abs. 1 BGG` — 1,781
- _(... 280,231 more values, total tail occurrences = 1,328,599)_

#### `topic` — 69,915 unique values

_Top 40 (of 69,915):_

- `Administrative appeal` — 9,258
- `Administrative review` — 7,102
- `Judicial review` — 3,080
- `Burden of proof` — 2,789
- `Compensation` — 2,499
- `Appeals` — 2,480
- `Cost allocation` — 2,003
- `Appeal process` — 1,879
- `Appeal grounds` — 1,810
- `Appeal procedure` — 1,800
- `Right to be heard` — 1,692
- `Contractual obligations` — 1,565
- `Administrative procedure` — 1,503
- `Cost recovery` — 1,459
- `Statute of limitations` — 1,296
- `Procedural compliance` — 1,197
- `Administrative Review` — 1,191
- `Grounds for review` — 1,117
- `Appeal review` — 1,105
- `Costs recovery` — 1,087
- `Criminal liability` — 1,076
- `Legal interpretation` — 1,024
- `Constitutional review` — 994
- `Benefit Calculation` — 993
- `Insurance claims` — 973
- `Constitutional complaints` — 918
- `Procedural fairness` — 880
- `Judicial Communication` — 819
- `Benefit Eligibility` — 749
- `Jurisdiction` — 729
- `Administrative Dispute` — 724
- `Appeal dismissal` — 707
- `Duty of care` — 701
- `Judicial impartiality` — 700
- `Taxable income` — 693
- `Benefit entitlement` — 688
- `Case consolidation` — 673
- `Legal representation` — 661
- `Suspensive effect` — 657
- `Court communication` — 642
- _(... 69,875 more values, total tail occurrences = 299,341)_

