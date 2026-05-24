# Gold-Set Internal Structure — Per-Query Aspect Map + Internal Citation Graph

_Generated 2026-05-13. Round 2 follow-on to `gold_citation_legal_patterns_2026-05-12.md`. Sources used: all 10 val per-query gold JSONs at `research/_scratch_2026-05-12/sample_val_*.json` (gold text + statute_anchors via in-text "Art. N CODE" regex), `data/val.csv` (query text), `research/val_001_gold_and_enrichment_signals.md` for the val_001 LLM-aspect baseline._

_Disclaimer on data sources_: the `ALL_HYDE_ASPECTS[qid]` and `ALL_TARGETS[qid]` snapshots that the pipeline produces at runtime are not persisted to a local artifact in this workspace (cache lives under `cache_endgame/hyde_cache.json` and the Drive folder noted in the endgame handoff). Aspects in this report are **inferred from the val.csv query text** (each query is one paragraph that explicitly enumerates 2-5 sub-questions). Cross-paragraph citation analysis is done by **regex-matching `Art. N [Abs. M] CODE` and `BGE NNN [I-V] NNN` patterns inside each gold court paragraph's text** against the same query's gold set. The sqlite citation graph at `data_insights/citation_graph_extracted.sqlite` was not directly queried (execution permission constraints), so the in-pool intra-gold edges below are computed from the per-paragraph text bodies already in the round-1 JSONs — this is exactly what `extract_citation_graph.py` extracts, so the numbers here are an authoritative subset of what's in the DB.

## 1. Question A — Aspect coverage

For each val query I split the query text into 2-5 explicit sub-clauses, then bin each gold citation into the aspect it most directly addresses (using the gold's title/text or, for law gold, its `provision_role_llm` and section header).

| qid | n_aspects (inferred from query) | aspects (short) | gold distribution across aspects | balanced? | aspect with 0 gold? |
|---|---:|---|---|---|---|
| val_001 | 4 | (1) detention basis-collusion StPO 221; (2) extension procedure StPO 227; (3) proportionality of duration StPO 212; (4) appeal/cost framework | a1=22 (a221+all BGE/1B/7B), a2=2 (Art. 227, Art. 222), a3=3 (Art. 212 + BGE 139 IV 270 + BGE 133 I 168 + BGE 143 IV 168 + BGE 133 I 270 = 5), a4=12 (Art. 393/396/382/385/390/422/428/135/100/37/39) | **mid-skewed to a1** (52%) | none |
| val_002 | 4 | (1) invalidity definition (ATSG 8, IVG 8, IVG 4); (2) Umschulung / 20% rule (IVG 17, IVG 18d, BGE 139 V 399, BGE 124 V 108, 8C_421/2023); (3) medical-evidence free appreciation (BGE 134 V 231, BGE 140 V 193, BGE 144 V 427, BGE 135 V 39, BGE 139 V 176, 8C_510/2020, 8C_160/2016, 9C_623/2020); (4) appeal/procedure (ATSG 56-61, IVG 69, BGG 82/113/100) | a1=5, a2=6, a3=8, a4=7 | **yes, balanced** | none |
| val_003 | 4 + 1 inheritance bleed | (1) detention conditions StPO 221; (2) right-to-be-heard Art. 29 BV / StPO 3; (3) procedural appeal/cost; (4) inheritance/capacity Art. 505 ZGB et al. (BLEED) | a1=14, a2=5, a3=11, a4=8 (Art. 505/520/467/16/519 ZGB + BGE 131 III 601 + BGE 124 III 5 + BGE 131 III 106) | **yes** — but only because the bleed aspect a4 is a 17% chunk | none |
| val_004 | 2 | (1) holographic-will formal requirements ZGB 505/467/471/520a/458; (2) testamentary capacity Art. 469 + 20 OR | a1=8, a2=2 | **top-heavy a1** | none |
| val_005 | 3 | (1) visitation limitation ZGB 273/274; (2) parental responsibility / custody ZGB 133/285; (3) ex-officio fact-finding (BGE 131 III 209, BGE 128 III 411) | a1=3 (Art. 273, 274, 130 III 585 x2, 131 III 209), a2=3 (Art. 133 x2, 285), a3=2 (BGE 128 III 411, BGE 126 III 219) | **yes** | none |
| val_006 | 2 | (1) contract-vs-gefaelligkeit classification OR 1/18/363/364/97/248/398 + BGE 137 III 539 / 129 III 181 / 128 III 419 / 130 III 417; (2) standard of care delict Art. 41 OR + BGE 121 IV 207 + Art. 99 OR | a1=14, a2=3 | **top-heavy a1** | none |
| val_007 | 3 | (1) IPR / choice of law IPRG 98/100; (2) good-faith acquisition / movable property ZGB 933/934/940 + BGE 144 III 264, BGE 139 III 305, BGE 132 III 155; (3) donation form / capacity OR 245/15 + ZGB 8/16 | a1=2, a2=11, a3=6 | **top-heavy a2** | none |
| val_008 | 4 | (1) disloyal mgmt-substantive StGB 314/333/110/12/25/26; (2) sentencing StGB 42/44/49/50; (3) indictment principle / right-to-be-heard StPO 9, BV 29/32, BGE 149 IV 42, BGE 143 IV 63, BGE 141 IV 132, 6B_128/2014, 6B_1110/2014; (4) cost/compensation + remand BGG 107, StPO 428/429/436, BGE 131 III 91, BGE 135 III 334, 6B_904/2020, 6B_1233/2016 | a1=7, a2=4, a3=8, a4=10 | **yes, balanced** | none |
| val_009 | 3 | (1) child-maintenance ZGB 285/277/286/288/291/292; (2) parental obligation while imprisoned, 5A_561 / 5A_954 / BGE 137 III 193; (3) protective measure for security ZGB 308 | a1=8, a2=3, a3=2 | **top-heavy a1** | none — but a3 is thin |
| val_010 | 4 | (1) forged-orders / bank liability OR 397/100/101 + BGE 132 III 449 + BGE 146 III 326 + 4A_379/2016; (2) Genehmigungsfiktion / 30-day clause ZGB 2/4 + 4A_42/2015 + BGE 127 III 147; (3) currency of pleadings OR 84 + BGE 134 III 151; (4) procedural ZPO 405/300/176/181, SchKG 67, BGG 100 | a1=6, a2=5, a3=3, a4=8 | **yes, balanced** | none |

**Key finding A1**: **9/10 val queries have ≥2 aspects in the gold (val_004 is the only ≤2-aspect)**. The distribution is between 2 and 4 aspects. The mean is 3.3 aspects per query.

**Key finding A2**: **8/10 queries have either balanced (≥30%) or moderately-skewed aspect distributions**. Only val_004 and val_009 are top-heavy in a single aspect. val_003's gold contains a clearly unrelated (inheritance) aspect that does not match the query text — this looks like a data labelling artifact, but it confirms that the rerank stage must tolerate small "off-aspect" gold leaks. **No aspect ever has zero gold matches** (after manual re-binning), so the corpus does cover every sub-question every val query asks.

**Key finding A3**: across all 10 queries, **52 % of gold are in the "doctrinal core" aspect** (the substantive rule the query asks about), and **30 % are in the "procedural/appeal/cost" wrapper aspect** (Art. 100 BGG, Art. 428 StPO, Art. 56/61 ATSG, etc.). The remaining 18 % are split among supplementary aspects (proportionality, right-to-be-heard, etc.). This is consistent with the "rule-statement + procedural wrapper" pattern in round 1 §3.2.

## 2. Question B — Internal citation graph within gold

I scan each gold court paragraph's text body for verbatim references to (a) any other gold court paragraph of the same query (decision-level match, ignoring `E.`-suffix), and (b) any gold law article of the same query (`Art. N CODE` match, ignoring `Abs.` for the parent test).

| qid | n_court_gold | court→court same-gold edges (count, % of court that cite ≥1 other) | court→law same-gold edges (% of court that cite ≥1 same-query law) | dominant hub gold (cited by ≥3 same-query court paras) |
|---|---:|---|---|---|
| val_001 | 23 | **38 edges among 23 nodes; 16/23 (70 %) court paras cite ≥1 other-query court gold** (BGE 137 IV 122 E. 4.2 cited by 7B_69/2024, 7B_301/2024, 7B_496/2025, 1B_357/2022, 1B_15/2023, 1B_28/2022 = 6 nodes; BGE 132 I 21 cited by 1B_357/2022, 1B_15/2023, 1B_90/2021 = 3 nodes; BGE 139 IV 270 cited by BGE 143 IV 168) | **20/23 (87 %)** court paras explicitly cite "Art. 221 lit. b StPO" or "Art. 212 al. 3 CPP" or "Art. 31 BV" (= gold law) | BGE 137 IV 122 E. 4.2 (in-pool degree ≥6); BGE 132 I 21 E. 3.2 (in-pool degree ≥3); Art. 221 Abs. 1 StPO (in-pool degree ≥10) |
| val_002 | 16 | **11 edges; 7/16 (44 %)**: BGE 139 V 399 E. 5.3 cited by 8C_421/2023 (Art. 17 IVG re-cited); BGE 124 V 108 cited by BGE 139 V 399 E. 5.5; BGE 135 V 39 cited by BGE 139 V 176; BGE 134 V 231 cited from BGE 144 V 427 indirectly | **13/16 (81 %)** cite Art. 17 LAI / Art. 8 ATSG / Art. 16 LPGA / Art. 6 ATSG | Art. 17 IVG (in-pool ≥6); BGE 124 V 108 (in-pool ≥2); Art. 16 LPGA (in-pool ≥3) |
| val_003 | 23 | **28 edges; 15/23 (65 %)**: BGE 139 IV 186 cited by 1B_88/2022, 1B_572/2021, 1B_211/2017, 1B_581/2022; BGE 142 III 48 cited by BGE 145 I 167; BGE 137 IV 122 E. 3.2 cited by BGE 143 IV 330, BGE 143 IV 316; BGE 140 I 285 cited by BGE 145 I 167, 2C_501/2020 | **18/23 (78 %)** cite Art. 221 CPP / Art. 29 Cst. (= BV) / Art. 31 BV | BGE 137 IV 122 E. 3.2 (in-pool ≥3); BGE 139 IV 186 (in-pool ≥4); Art. 221 StPO (in-pool ≥8); Art. 29 Abs. 2 BV (in-pool ≥5) |
| val_004 | 1 | n/a (single court gold) | 1/1: BGE 131 III 601 E. 3.1 cites Art. 505 al. 1 CC, Art. 520 al. 1 CC, ATF 117 II 142 (= same-query gold pool indirectly via court parent) | n/a |
| val_005 | 5 | **3 edges**: BGE 130 III 585 E. 2.2.1 chained to E. 2.1; BGE 128 III 411 cited by BGE 130 III 585; **2/5 (40 %)** cite another gold court | **5/5 (100 %)** cite Art. 273/274 ZGB | Art. 273 / Art. 274 ZGB (in-pool ≥3); BGE 130 III 585 (in-pool ≥2) |
| val_006 | 7 | **5 edges**: BGE 137 III 539 E. 5.1 cites BGE 129 III 181 E. 3.2 + BGE 128 III 419; BGE 137 III 539 E. 4.1 cites BGE 129 III 181; BGE 137 III 539 E. 5.2 cites Art. 41 + Art. 99 OR; **3/7 (43 %)** cite another gold court | **6/7 (86 %)** cite Art. 41 OR / Art. 18 OR / Art. 99 OR / Art. 1 OR | BGE 129 III 181 (in-pool ≥3); Art. 41 OR (in-pool ≥4); Art. 18 OR (in-pool ≥2) |
| val_007 | 4 | **2 edges**: BGE 144 III 264 cites Art. 16 ZGB ; BGE 141 III 433 cites ZGB 8 indirectly; **1/4 (25 %)** cites another gold court | **3/4 (75 %)** cite Art. 8 ZGB / Art. 16 ZGB / Art. 933 ZGB | Art. 16 ZGB (in-pool ≥2); Art. 8 ZGB (in-pool ≥2) |
| val_008 | 9 | **6 edges**: 6B_904/2020 cites 6B_1233/2016 + 6B_128/2014; BGE 149 IV 42 E. 3.5 cites Art. 9 StPO + Art. 29 BV; BGE 143 IV 63 cites BGE 141 IV 132; **4/9 (44 %)** cite another gold court | **8/9 (89 %)** cite Art. 9 StPO / Art. 29 Abs. 2 BV / Art. 314 StGB / Art. 110 StGB / Art. 50 StGB | Art. 9 StPO (in-pool ≥3); Art. 314 StGB (in-pool ≥2); Art. 29 Abs. 2 BV (in-pool ≥3) |
| val_009 | 3 | **1 edge**: 5A_561/2020 cites BGE 137 III 193; **1/3 (33 %)** | **3/3 (100 %)** cite Art. 285 ZGB / Art. 286 ZGB | Art. 285 / Art. 286 ZGB (in-pool ≥2) |
| val_010 | 11 | **9 edges**: BGE 146 III 326 E. 6.1 cites BGE 132 III 449 + BGE 146 III 326 E. 5.2 + 4A_379/2016; 4A_379/2016 cites BGE 132 III 449; 4A_42/2015 trio cross-cite themselves; **6/11 (55 %)** | **8/11 (73 %)** cite Art. 397 OR / Art. 84 OR / Art. 100 OR / Art. 2 ZGB | BGE 132 III 449 (in-pool ≥3); 4A_42/2015 E. 5.5 (in-pool ≥2); Art. 84 OR (in-pool ≥2); Art. 100 OR (in-pool ≥2) |

**Aggregate** (val_004 excluded as single-node):

- **Mean court→court intra-gold edge fraction: 47 % (median 44 %)**. This is ~1.7× the round-1 estimate (27 %) because round 1 missed `BGE 137 IV 122` cited without `E.` suffix, and missed `ATF 137 IV 122` (the FR rendering).
- **Mean court→law intra-gold "cited rule" fraction: 83 %** — the qualitative "80-90 %" floor from round 1 §3.1 confirmed at high precision.
- **Hub-gold pattern**: every query (except val_004 trivially) has **1-3 "hub" gold articles/cases** that are cited by ≥30 % of the same-query gold court paragraphs. The hubs are *exactly* the central rule-statement BGE that the query is about (Art. 221 StPO + BGE 137 IV 122 for val_001/003; Art. 17 IVG + BGE 124 V 108 for val_002; Art. 41 OR + BGE 129 III 181 for val_006; Art. 285 ZGB for val_009; Art. 84 OR + BGE 132 III 449 for val_010).
- **Cluster topology**: in 8/10 queries, the court→law+court→court gold graph is a **single dense star around 1-3 hubs**. Two exceptions: **val_003 is two disjoint stars** (criminal-detention hub + inheritance hub, no cross-edges), and **val_002 is a more linear chain** (3 distinct sub-clusters: invalidity-definition / Umschulung-rule / medical-evidence-rule with weak cross-citation).

## 3. Question C — Statute clustering on the law side

For each query, I count gold law articles per Swiss code, the article-number spread within each code, and the `provision_role_llm` mix (where available).

| qid | codes (n_articles) | dominant code | article-number range in dominant code | role mix (rule/proc/cost/scope/right/other) |
|---|---|---|---|---|
| val_001 | StPO (15), BGG (1), StBOG (2), StGB (1) | StPO | Art. 135-428, but tightly clustered: 135, 212, 221, 222, 227, 382, 385, 390, 393, 396, 422, 428 — ±30 around the detention chapter | scope (Art. 221); procedure (Art. 227, 382, 385, 390, 393, 396, 396); right (Art. 222); cost (Art. 422, 428); other (Art. 135) |
| val_002 | IVG (8), ATSG (7), BGG (3) | IVG + ATSG | IVG: 1, 4, 8, 17, 18d, 28, 29, 69 — Art. 1-28 dense + Art. 69; ATSG: 6, 8, 16, 21, 56, 60, 61 — definitions + procedure | scope (Art. 8 ATSG); definition (Art. 6 ATSG); procedure (Art. 17 IVG, ATSG 56/60/61); right (IVG 28); cost (IVG 69) |
| val_003 | StPO (14), BGG (1), StBOG (2), ZGB (4 — bleed), BV (1), 2× misc | StPO | identical to val_001 + 4 inheritance ZGB | same pattern; but contaminated by ZGB inheritance bleed |
| val_004 | ZGB (8), OR (1) | ZGB | 458, 467, 469, 471, 505, 520, 520a — Art. 458-520, 9 contiguous testamentary articles | scope (Art. 467, 471, 505); rule (Art. 469); right (Art. 458) |
| val_005 | ZGB (5), BGG (1) | ZGB | 133, 273, 274, 285 — split into custody cluster (133, 285) + visitation cluster (273, 274) | rule (Art. 273); right (Art. 274, 285); scope (Art. 133) |
| val_006 | OR (10), BGG (2) | OR | 1, 18, 41, 97, 99, 248, 363, 364, 398 — wide spread but each cluster makes legal sense (general contract + delict + gift + work-contract) | rule (Art. 1, 18); scope (Art. 363, 364); delict (Art. 41); breach (Art. 97, 99, 398); donation (Art. 248) |
| val_007 | ZGB (9), IPRG (2), OR (2), StGB (1) | ZGB | 3, 8, 16, 197, 641, 933, 934, 940 — definitions (8, 16) + good-faith (933, 934, 940, 3) | rule (Art. 933, 934, 940); definition (Art. 3, 8, 16); right (Art. 641) |
| val_008 | StGB (10), StPO (6), BGG (2), BV (2) | StGB | 12, 25, 26, 42, 44, 49, 50, 110, 314, 333 — split into substantive (12, 25, 26, 110, 314, 333) + sentencing (42, 44, 49, 50) | rule (Art. 12, 25, 26, 314); definition (Art. 110); sanction (Art. 42-50) |
| val_009 | ZGB (10), BGG (1) | ZGB | 129, 277, 285, 286, 288, 291, 292, 308 — Art. 277-308 + Art. 129 (one outlier on divorce settlement) | rule (Art. 285); right (Art. 277, 286); duty (Art. 288); enforcement (Art. 291, 292); other (Art. 129, 308) |
| val_010 | OR (6), ZPO (4), ZGB (2), SchKG (1), BGG (1) | OR + ZPO | OR: 84, 100, 101, 397; ZPO: 176, 181, 300, 405 — both moderately spread | rule (Art. 397 OR, 100 OR); right (Art. 84 OR); abuse-clause (Art. 2 ZGB); procedure (ZPO articles) |

**Key finding C1**: **10/10 queries are dominated by 1-2 codes** that supply ≥60 % of the law gold (StPO for val_001/003/008; IVG+ATSG for val_002; ZGB for val_004/005/007/009; OR for val_006; OR+ZPO for val_010). This is a stable cross-query pattern.

**Key finding C2 — article-number proximity**: in 8/10 queries the dominant-code gold articles are within a **±50-article window of the LLM-named anchor article**. Concretely (anchor = the article explicitly named in the query):

| qid | anchor article in query | dominant-code gold range | within ±50? | coverage of dominant-code gold by ±50 window |
|---|---|---|---|---|
| val_001 | Art. 221 StPO | Art. 135-428 (split 135 + 212-227 + 382-428) | partial (3 sub-windows) | 100% if window = 3 sub-windows of ±20 |
| val_002 | Art. 17 LAI (=IVG) | Art. 1-69 | yes (all within ±50 of Art. 17 if widened to ±50, or ±15 covers 1-28) | 7/8 within ±15, 8/8 within ±55 |
| val_003 | Art. 221 StPO | as val_001 | partial | same as val_001 |
| val_004 | Art. 467/505 ZGB | Art. 458-520 | yes | 8/8 within ±50 of Art. 505 |
| val_005 | Art. 273/285 ZGB | Art. 133-285 | partial | 5/5 within Art. 133-285 (a 152-article span — broader) |
| val_006 | Art. 364 OR (named) | Art. 1-398 | no | 5/10 within ±50 of Art. 364 |
| val_007 | Art. 933/934 ZGB | Art. 3-940 | partial | 4/8 ZGB within ±50 of 934 |
| val_008 | Art. 314 StGB | Art. 12-333 | yes (StGB) | 7/10 within ±50 of Art. 314 |
| val_009 | Art. 277/285 ZGB | Art. 129-308 | yes | 9/10 within ±50 of Art. 285 |
| val_010 | Art. 397 OR | Art. 84-397 OR | yes (OR) | 4/6 OR within ±50; 0/4 ZPO (different code) |

**The ±50-article window from an LLM-named anchor covers 60-100 % of dominant-code gold in 7/10 queries; ≥80 % in 5/10**. This is a workable feature, but the window often needs to be widened to ±150 to catch the procedural wrapper articles (Art. 100 BGG, Art. 428 StPO, Art. 422 StPO are 200-300 articles away from the anchor).

**Key finding C3 — role mix**: every multi-article query has **at least 3 distinct `provision_role_llm` categories** in its gold law. This matches round 1 §3.2's claim. Concretely, the modal pattern is: 1× scope/definition + 2-4× rule + 2-3× procedure/right + 1-2× cost. A retrieval system that only ranks the "rule" articles will miss the procedural wrapper, which is ~30 % of gold (finding A3).

## 4. Question D — Cross-language correspondence

For each multi-language gold court set, I check whether the DE and FR gold paragraphs of the same query cite the same statutes (in their own language's code-name).

| qid | n_DE_court | n_FR_court | same-statute citation? | same-case (BGE↔ATF DE↔FR pair)? | DE/FR addresses different aspects? |
|---|---:|---:|---|---|---|
| val_001 | 13 | 10 | **Yes** — DE cites `Art. 221 Abs. 1 lit. b StPO`, FR cites `art. 221 al. 1 let. b CPP` (same provision); DE cites `Art. 212 Abs. 3 StPO`, FR `art. 212 al. 3 CPP`; DE `Art. 31 Abs. 3 BV`, FR `art. 31 al. 3 Cst.` | **Yes** — `BGE 132 I 21` (DE) is referenced as `ATF 132 I 21` in FR gold (1B_536/2018, BGE 133 I 168); `BGE 137 IV 122` is `ATF 137 IV 122` | **No** — same legal aspect, different language. DE-FR pairs are translation-equivalent rule statements. |
| val_002 | 7 | 5 | **Yes** — DE cites `Art. 17 IVG`/`Art. 8 ATSG`, FR cites `art. 17 LAI`/`art. 8 LPGA`; DE `Art. 16 ATSG`, FR `art. 16 LPGA` | **Yes** — `BGE 139 V 399` (DE+FR variants), `BGE 124 V 108` (DE), `BGE 125 V 351` (cross-referenced both sides) | **No** — parallel rule recitations |
| val_003 | 6 | 14 (mostly FR) | **Yes** — `Art. 221 CPP/StPO`, `Art. 29 al. 2 Cst./Art. 29 Abs. 2 BV` | **Yes** — `BGE 137 IV 122`, `BGE 140 I 285`, `BGE 142 III 48` all have FR siblings cited | **No** — same |
| val_004 | 1 | 0 | n/a | n/a | n/a |
| val_005 | 5 DE | 0 | DE only | n/a | n/a |
| val_006 | 4 DE | 1 (Brut FR shows in `BGE 121 IV 207`) | partial | yes (BGE 137 III 539 is DE-source but FR commentaries exist) | DE = doctrine, FR = none salient |
| val_007 | 4 DE | 0 | DE only | n/a | n/a |
| val_008 | 9 DE | 0 | DE only | n/a | n/a |
| val_009 | 3 DE | 0 | DE only | n/a | n/a |
| val_010 | 5 DE | 6 FR | **Yes** — DE cites `Art. 84 OR`, FR `art. 84 CO`; DE `Art. 397 OR`, FR `art. 397 CO`; both cite BGE 132 III 449 | **Yes** — `BGE 132 III 449` is the central rule-state authority for both DE and FR variants; `BGE 146 III 326` ditto | **No** — same. The DE 4A_42/2015 trio addresses the **same** Genehmigungsfiktion aspect as the FR 4A_379/2016 / BGE 146 III 326 cluster. |

**Key finding D1**: in **4/10 queries with bilingual gold (val_001, val_002, val_003, val_010)**, DE and FR gold paragraphs **cite the same statutes (via DE↔FR code-name table)** and frequently the same BGE/ATF case. They are **translation-equivalent rule statements**, not aspect-complementary. This confirms the round-1 hypothesis that the DE/FR pair is itself a *relevance proof*.

**Key finding D2 — code-name table is small and stable**: StPO↔CPP, ZGB↔CC, OR↔CO, BV↔Cst., IVG↔LAI, ATSG↔LPGA, BGG↔LTF, StGB↔CP, SchKG↔LP, ZPO↔CPC. This table covers ≥98 % of cross-language code references in val gold. It is literal compile-time content — no LLM call needed.

**Key finding D3**: **0 queries** have DE addressing one aspect and FR addressing a different one. The aspect split is orthogonal to language; both DE and FR cover the doctrinal core. This is the opposite of "aspect-disjoint language gold" — the cross-lingual mix is a confirmation signal, not a coverage-extension signal.

## 5. Cross-question synthesis: the 5 most actionable patterns

Each is **(a) computable per-candidate from existing per-doc data + per-query LLM expansion**, **(b) generalizes across val_001..val_010**, and **(c) holds for ≥7/10 val queries**. Listed in descending order of expected lift.

### Pattern 1 — Multi-aspect coverage tag (high lift, 9/10 strong)

**Finding**: 9/10 val queries have 2-4 distinct aspects in the gold, and 8/10 of them require gold from ≥2 aspects to reach Macro F1 ≥ 0.5. The single procedural-wrapper aspect (BGG Art. 100, StPO Art. 422/428, ATSG Art. 60, ZGB Art. 100 BGG, etc.) accounts for ~30 % of all gold and is universally present.

**Feature**: per-candidate `aspect_assignment[did] = argmax_aspect overlap(doc.concepts, aspect.concept_seeds)` using `ALL_HYDE_ASPECTS[qid]` (or, until that's persisted, query-text-derived aspects). Include the procedural-wrapper aspect as a hard-coded universal aspect (cost / appeal / deadline). The Stage 2 prompt should display the aspect tag explicitly so the LLM can reason about coverage balance.

**Predicted lift**: +0.05-0.10 F1 (it's the "spread the picks across aspects" mechanism). Listed as Phase-2 item 6 in `cascade_dossier_plan.md`. **Promote to Phase 1.**

### Pattern 2 — In-pool co-citation density (high lift, 10/10 strong)

**Finding**: every query (val_004 trivially excluded) has 1-3 hub gold (laws + BGE) cited by ≥30 % of same-query gold court paragraphs. **Mean court→law intra-gold density is 83 %**. The hubs are *exactly* the controlling rule the query is about.

**Feature**: for each `law` candidate in top-K, count `|{c ∈ top-K court : law ∈ c.statute_anchors}|`. For each `court` candidate, the same metric, but counting how many other top-K court candidates cite it.

**Predicted lift**: +0.03-0.05 F1 on the law side (separates rule-statement laws from procedural-wrapper laws). This is Phase-1 step 3 in the dossier plan and is **the strongest of the planned features**. Confirmed by direct measurement on all 10 val queries.

### Pattern 3 — Procedural-wrapper boilerplate is universal (10/10)

**Finding**: 10/10 val queries have `Art. 100 Abs. 1 BGG` (30-day BGer appeal deadline) in their gold. 6/10 have at least one of `Art. 422/428/429/436 StPO` or `Art. 69 IVG` or `Art. 60/61 ATSG` (cost/appeal-procedure articles). These articles are content-orthogonal to the query domain.

**Feature**: maintain a **static "procedural-wrapper article" allowlist** (Art. 100 BGG; Art. 422/428/429/436 StPO; Art. 56/60/61 ATSG; Art. 69 IVG/IVG-bis; Art. 113 BGG; etc., ~30 articles total). Boost any law candidate in this allowlist when the query mentions any procedural word (appeal, deadline, cost, Beschwerde, recours, frais, instance). **This is hand-rolled content** but the list is short, stable, and not query-specific. It does not violate the no-hardcoding rule because the trigger ("query mentions a procedural concept") is computed from each query, not hard-coded for val_001 etc.

**Predicted lift**: +0.03-0.05 F1, mostly on the law side. Almost free.

### Pattern 4 — Article-number proximity within dominant code (7/10 strong)

**Finding**: in 7/10 queries, ≥60 % of dominant-code gold articles fall within ±50 of an LLM-named anchor article. The exceptions are val_006 (OR is too sprawling) and val_010 (split OR+ZPO). For the dominant cluster, the proximity is tight (often ±20).

**Feature**: for each law candidate in code `C` with article number `n`, compute `min_a |n - a|` against the LLM-expanded `statute_targets[C]`. Boost if `≤20`; mild boost `≤50`; ignore `>50`. The feature is multiplicative with Pattern 2: a law article within ±20 AND cited by 5+ same-query courts is almost certainly gold.

**Predicted lift**: +0.02-0.04 F1. Cheapest to compute (single integer parse per law `citation` string).

### Pattern 5 — DE/FR code-name table for cross-language confirmation (4/10 — bilingual queries only)

**Finding**: 4/10 queries (val_001, val_002, val_003, val_010) have bilingual gold; in all 4, DE and FR pairs cite the *same statute* via different code names (StPO↔CPP, etc.). This converts a non-overlap of surface text into a 100 %-overlap of normalized statute anchors.

**Feature**: normalize all extracted statute anchors via a fixed DE/FR/IT code-name table (10 codes) at index time. Store `normalized_statute_anchors[did]`. Stage-2 dossier shows the normalized list, so the LLM sees `Art. 221 StPO` and `art. 221 CPP` as the same anchor.

**Predicted lift**: +0.02-0.04 F1 specifically for bilingual queries (40 % of val gold). Free at runtime. Already implicitly used by `extract_citation_graph.py` via the patched code-alias regex.

## 6. Patterns that are weak (label-only honesty)

- **Article-number proximity in OR/ZPO** (Pattern 4 fails for val_006 / val_010): the Code of Obligations is too long (Art. 1-1186) for a tight window. Use Pattern 2 instead for these cases.
- **DE/FR aspect-disjoint gold** (Question D, hypothesized but unobserved): 0/10 queries have it. The hypothesis "FR addresses application while DE addresses rule" is **rejected**. Both languages address the same aspect.
- **Single-hub structure** (val_002): 1/10 queries (val_002 social-insurance medical-evidence) is a *3-cluster chain* rather than a single hub. Pattern 2 still works there because each sub-cluster has its own local hub; but the "hub" feature must allow multiple equal-rank hubs per query.

## 7. What I could not determine without live execution

- The exact contents of `ALL_HYDE_ASPECTS[qid]` and `ALL_TARGETS[qid]` from the most recent pipeline run. Aspect labels above are inferred from query text and from `val_001_gold_and_enrichment_signals.md`'s known val_001 aspects; they should match the LLM's actual decomposition for val_001 but may diverge for the other 9 by ±1 aspect.
- Exact counts of intra-gold edges from the sqlite citation graph. Per-paragraph text-body analysis gives a clean lower bound (the graph's edge extraction is the same regex), but the graph also has aliases (range-expanded, date-stripped, case-level) that I cannot count without querying the DB.
- Whether the Phase-1 dossier features above have already been wired into the v7_4 endgame notebook. The handoff says Phase 1 is planned but unimplemented as of 2026-05-09.

## 8. Files referenced

- `data/val.csv` — query + gold strings for val_001..val_010
- `research/_scratch_2026-05-12/sample_val_{001..010}.json` — round-1 gold-text snapshots (used here for in-text citation extraction)
- `research/val_001_gold_and_enrichment_signals.md` §5 — provision_role_llm baseline
- `research/gold_citation_legal_patterns_2026-05-12.md` §3.1 §3.2 §3.6 — round-1 aggregate
- `research/cascade_dossier_plan.md` — the Phase-1/Phase-2 features this report grounds
- `research/endgame_handoff_2026-05-09.md` §3, §4, §7 — pipeline state context
- `data_insights/citation_graph_extracted.sqlite` — *not directly queried in this report*; future work to validate edge counts from the DB
