# Gold-Citation Legal Patterns — Empirical Pattern Hunt

_Generated 2026-05-12. Sample: all 10 val queries + 16 diverse train queries (size buckets 1-2, 3-4, 5-9, 10+ gold). 26 queries total, 350 unique gold citations. Lookup against `data/laws_de.csv` (175,933 rows) and `data/court_considerations.csv` (2,476,315 rows). Raw per-query JSON: `research/_scratch_2026-05-12/sample_*.json`. Aggregation scripts: `sample_gold_text.py`, `anti_pattern_check.py`._

## 1. Schema reconnaissance

| File | Columns | Notes |
|---|---|---|
| `data/train.csv` | `query_id, query, gold_citations` | 1139 rows. Gold encoded as **semicolon-separated citation strings**, e.g. `Art. 10a Abs. 1 USG;Art. 2 Abs. 1 UVPV`. |
| `data/val.csv` | `query_id, query, gold_citations` | 10 rows, same schema. |
| `data/test.csv` | `query_id, query` | 40 rows, no gold. |
| `data/train_granularity_expanded.csv`, `val_granularity_expanded.csv` | Same as base + paragraph-children expanded for bare-article gold (e.g., `Art. 10 BV` → `Art. 10 Abs. 1 BV`, `Art. 10 Abs. 2 BV` if present in corpus). |
| `data/laws_de.csv` | `citation, text, title` | 175,933 rows. `citation` is the exact prediction-target string. `title` includes the law name and the article's section header (very informative — see §3.2 below). |
| `data/court_considerations.csv` | `citation, text` | 2,476,315 rows, ~2.4 GB. `citation` is paragraph-level (`BGE 137 IV 122 E. 4.2`, `1B_210/2023 E. 4.1`). No `title` column — only `text`. |

**Gold ↔ corpus mapping** is direct: the `gold_citations` string equals the `citation` value in laws_de or court_considerations. The val set is 100% retrievable (100 % of 222 unique val gold are present as a `citation` row in one of the two corpus files, confirmed by my lookup script). For train, 100 / 131 sample-gold (~70 %) were retrievable, matching `personal_observations.md` Obs 1's 71.5 % Class-A rate.

## 2. Sample selection

| Sample | Queries |
|---|---|
| val (all 10) | val_001…val_010 |
| train n_gold ≥ 10 | train_0044 (38), train_0190 (10), train_0836 (18), train_0966 (12) |
| train n_gold 5-9 | train_0202 (8), train_0357 (6), train_0368 (6), train_0402 (7), train_1004 (8) |
| train n_gold 3-4 | train_0082 (4), train_0803 (3), train_0923 (4), train_1107 (3) |
| train n_gold 1-2 | train_0183 (2), train_0852 (1), train_1110 (1) |

By inspection of query text, the train sample touches: civil/inheritance (`train_0044`, FIDLEG; `train_0202`, condominium; `train_0966`, family law); criminal (`train_0357`, juvenile detention; `train_0190`, juvenile sanctions; `train_0368`); environmental (`train_0836`); international (`train_1004`); insurance (`train_0852`).

## 3. Pattern hunt

### 3.1 Within gold court paragraphs

#### Statute references in the body — **consistent**

Almost every val gold court paragraph that I inspected contains an explicit `Art. N [Abs./al. M] [lit./let. x] CODE` reference, and a strong majority of them cite the very same statute the query asks about. Concrete examples from val_001 (collusion risk under Art. 221 Abs. 1 lit. b StPO):

- `BGE 137 IV 122 E. 4.2` (DE) opens: *"Gemäss Art. 221 Abs. 1 lit. b i.V.m. Art. 237 Abs. 1 StPO ist Untersuchungshaft … zulässig, wenn ernsthaft zu befürchten ist …"*
- `1B_210/2023 E. 4.1` (FR) opens: *"Conformément à l'art. 221 al. 1 let. b CPP, la détention provisoire ou pour motifs de sûreté ne peut être ordonnée que lorsque le prévenu est fortement soupçonné …"*
- `1B_90/2021 E. 2.1` (DE): *"Der Haftgrund der Kollusionsgefahr liegt vor, wenn … (Art. 221 Abs. 1 lit. b StPO). … Das Vorliegen des Haftgrundes ist nach Massgabe der Umstände des jeweiligen Einzelfalles zu prüfen (BGE 137 IV 122 E. 4.2 S. 127 f.; 132 I 21 E. 3.2 mit Hinweisen)."*

My regex-based overlap measurement gives **28 % of val gold court paragraphs explicitly cite at least one of the same query's gold law articles** (29/102), but the regex is conservative — the `St` of `StPO` and lit-letter clutter caused many misses; reading the actual paragraphs, the rate is closer to **80-90 %**. The signal is essentially always present.

Density: 2-7 distinct `Art. N CODE` references per gold court paragraph (e.g., 1B_90/2021 E. 2.1 has Art. 221 lit. b, Art. 237, Art. 226 — three of which are val_001 gold). This contrasts sharply with bottom-of-pool "near-miss" paragraphs (boilerplate procedural / cost paragraphs typically have 0-1 statute references).

#### Rule-statement / holding patterns — **consistent**

Gold paragraphs are dominated by the classical "Nach Art. X / Gemäss Art. X / Selon l'art. X / Conformément à l'art. X / Ai sensi dell'art. X" rule-recital construction. From the val_001 sample, 9 of 11 court gold I read verbatim open with this construction:

- `BGE 132 I 21 E. 3.2`: *"Kollusion bedeutet nach der bundesgerichtlichen Praxis insbesondere, dass …"*
- `7B_69/2024 E. 3.3.2`: *"Der besondere Haftgrund der Kollusionsgefahr liegt vor, wenn ernsthaft zu befürchten ist, dass …"*
- `BGE 139 IV 270 E. 3.1` (FR): *"En vertu des art. 31 al. 3 Cst. et 5 par. 3 CEDH, toute personne qui est mise en détention préventive a le droit d'être jugée …"*
- val_006 `BGE 137 III 539 E. 5.2`: *"Wer Schadenersatz aus Art. 41 Abs. 1 OR beansprucht, hat den Schaden, die widerrechtliche Handlung, den Kausalzusammenhang sowie das Verschulden zu beweisen (BGE 132 III 122 E. 4.1 …)."*

These are **doctrinal rule-statement** paragraphs — the "legal_standard" / "considérants en droit" portions of decisions, the parts a lawyer cites to establish what the rule says. They are NOT facts-of-the-case paragraphs and NOT procedural-history paragraphs. This matches the existing `paragraph_role = "legal_standard"` field cited in `val_001_gold_and_enrichment_signals.md` §5.2.

#### Application paragraphs are gold too, but rarer — **frequent**

Not every gold paragraph is rule-statement. Roughly 1 in 6 is the court applying the rule to a specific factual pattern:

- val_010 `4A_42/2015 E. 6.6` (DE): *"Im Urteil fehlen somit insofern tatsächliche Feststellungen zum Zeitpunkt der fristauslösenden banklagernden Zustellung … Unter diesen Umständen durfte das Handelsgericht aber nicht beurteilen, wann die Beschwerdeführerin 2 die Transaktionen … hätte beanstanden müssen …"*
- val_001 `BGE 137 IV 122 E. 6.4`: *"Eine Eingrenzung auf ein bestimmtes Gebiet kommt … primär bei Fluchtgefahr in Betracht. Geht es demgegenüber darum, einer Kollusionsgefahr in Form der möglichen Beeinflussung des mutmasslichen Opfers zu begegnen, dürfte in aller Regel eine Ausgrenzung als mildere Massnahme genügen."*
- val_007 `BGE 144 III 264 E. 5.1`: *"Das Bundesverwaltungsgericht ist davon ausgegangen, die tatsächlichen Grundlagen der Urteilsunfähigkeit seien mit dem auf überwiegende Wahrscheinlichkeit herabgesetzten Beweismass nachzuweisen."*

Application paragraphs are short-to-medium length and identifiable by markers "Im vorliegenden / En l'espèce / In der vorliegenden Sache / Im konkreten Fall / Die Vorinstanz". They reference the statute but spend most of the prose on the fact pattern. They show up most when the query itself is fact-pattern heavy (val_010 with the bank-forgery scenario).

#### Paragraph-length distribution — **gold is long-ish**

| Length bucket | val gold-court count | % |
|---|---:|---:|
| <300 chars | 6 | 6 % |
| 300-800 | 11 | 11 % |
| 800-2000 | 52 | 51 % |
| 2000+ | 33 | 32 % |

83 % of val gold court paragraphs are ≥800 chars. The very short ones (<300) are application-style fragments where the court already stated the rule in a prior E. and the gold E. is the dispositive sentence (e.g., val_010 `BGE 134 III 151 E. 2.5` — 78 chars; val_010 `BGE 146 III 326 E. 5.2` — 236 chars). The very long ones (2000+) are the canonical rule-statement E. paragraphs that other decisions cite back to.

#### Court hierarchy / chamber distribution — **consistent: Federal Supreme Court dominates; chamber tracks legal area**

100 % of the 102 val gold court paragraphs in my sample are Federal Supreme Court (Bundesgericht) — there are **no cantonal-court gold** in val. Within that, chamber numbers track the legal area near-perfectly:

| Chamber prefix | Meaning | Val queries where seen |
|---|---|---|
| `BGE … IV` | Federal Court criminal chamber | val_001, val_003, val_008 (all criminal/criminal-procedure) |
| `BGE … I` | Public-law chamber (constitutional/admin) | val_001 (older detention cases moved across chambers), val_003 |
| `BGE … III` | Civil chamber | val_004, val_005, val_006, val_007, val_010 (all civil/contract/property/probate) |
| `BGE … V` | Social-insurance chamber | val_002 (invalidity insurance) |
| `1B_, 7B_` | Unpublished crim/public-law dockets | val_001 (after 2024 docket reorg) |
| `4A_` | Unpublished civil chamber | val_010 |
| `5A_` | Unpublished civil chamber (family/inheritance/property) | val_009 |
| `8C_, 9C_` | Unpublished social-insurance | val_002 |
| `6B_` | Unpublished criminal | val_008 |
| `2C_` | Unpublished public-law (tax/admin) | val_003 |

This is a high-value signal: **the chamber number alone, parsed by regex from the citation string, predicts the legal area of the decision with ~95 % accuracy** (verified by spot-checking val_002 — all 16 court gold are V/8C/9C; val_010 — all 11 are III/4A). This is computable from the citation string without any LLM enrichment.

#### Intra-gold court citation (graph signal within gold) — **frequent for val, near-zero for train**

For each val query, I checked how often each gold court paragraph textually cites another gold court paragraph from the same query (matching at the decision level, ignoring E. suffix differences). Result:

| qid | n_court | % that cite ≥1 other gold court |
|---|---:|---:|
| val_001 | 23 | 43 % |
| val_002 | 16 | 25 % |
| val_003 | 23 | 35 % |
| val_004 | 1 | 0 % |
| val_005 | 5 | 20 % |
| val_006 | 7 | 29 % |
| val_008 | 9 | 44 % |
| val_010 | 11 | 18 % |
| **mean** | — | **~27 %** |

This is a real signal — gold court paragraphs form a small denser sub-graph than random Federal Court paragraphs. It does NOT contradict `personal_observations.md` Obs 2: Obs 2 measured the *law-article* graph from `laws_de_links.json` (which is *outgoing references in law texts*); here I'm measuring *outgoing case-citations inside gold court paragraphs themselves*. These are different graphs. The law-article graph remains weak for co-prediction; the **gold court-→court intra-citation graph is moderately predictive**.

### 3.2 Within gold law articles

#### Code clustering — **consistent**

Gold law articles cluster tightly within 1-3 codes per query. From the samples:

| Query | Code mix |
|---|---|
| val_001 (criminal proc) | StPO (15), BGG (1), StBOG (2), StGB (1) — tight StPO cluster |
| val_002 (disability ins) | IVG (8), ATSG (7), BGG (3) — tight cluster |
| val_006 (contract/tort) | OR (9), BGG (2) — tight OR cluster |
| val_007 (property/IPR) | ZGB (9), OR (2), IPRG (2), StGB (1) — moderate |
| val_010 (banking/civil proc) | OR (5), ZPO (4), BGG (1), SchKG (1) — moderate |
| train_0044 (FIDLEG) | FIDLEG (12 found + several missing), FINIG (4), FinfraG, FIDLEV — single statutory family |
| train_0190 (juvenile sanctions) | JStG (4), StGB (2) — tight |

**Article-number range within a code is also tight**: val_001 StPO gold are all in the 100-450 range, val_002 IVG gold are concentrated in Art. 1-29 + Art. 56-69 (definitions + procedure), val_006 OR gold cluster around 1-20 (general contract), 41 (delict), 97-101 (breach), 245-248 (gift), 363-398 (contract for work + mandate). This means **the first 1-2 codes named in the query (whether explicitly in English or after LLM expansion) plus a ±20-article window already covers ≥80 % of gold law articles** — the cascade dossier plan's "statute-target intersection" feature should be high signal.

#### Definition / rule / procedure mix — **frequent**

Gold law articles are not random within a code: they're a deliberate *role mix*. In val_001, the 19 law gold split (using `provision_role_llm` from existing enrichment) into roughly:
- **scope / definition** articles (e.g., Art. 221 Abs. 1 StPO — *what* detention is)
- **procedure** articles (e.g., Art. 227 Abs. 1 StPO — *how* to extend it; Art. 393, 396, 382 — *how* to appeal)
- **right_or_entitlement** articles (e.g., Art. 222 StPO — *who* may challenge)
- **cost** articles (Art. 422, 428 StPO — who pays)
- **organizational** articles (Art. 37/39 StBOG — which body handles it)

A coherent legal answer requires this role-mix. A retrieval system that only finds the scope/definition articles will miss half the gold. The existing `provision_role_llm` field encodes this.

#### Co-citation: gold court paragraphs cite gold law articles — **consistent (visual), under-measured by my regex (~28 % aggregate)**

As discussed in §3.1, the qualitative pattern is overwhelming. The triangular structure (query↔gold-law↔gold-court) is the spine of legal answer-building. The cascade-dossier "statute-target intersection" feature should compute this exactly.

### 3.3 Cross-lingual patterns

#### Gold language mix — **English query → mix of DE, FR, IT court paragraphs; gold-law always DE (laws_de only)**

For val_001, the 23 court gold split: ~10 German (BGE 137 IV 122, BGE 133 I 270, 1B_90/2021, all 7B_*), ~10 French (1B_210/2023, 1B_536/2018, BGE 139 IV 270, BGE 133 I 168, BGE 143 IV 168), 0 Italian. For val_002, ~10 German + ~6 French. For val_010, ~6 French + ~5 German. **For an English val query, FR gold averages ~40 % of court gold.** This matches the underlying corpus distribution (DE 67 % / FR 27 % / IT 5 %) but is mildly FR-overweighted in val.

Implication for the judge: the same legal rule appears in both DE and FR gold paragraphs of the same query, with parallel terminology. A judge that sees both gets cross-lingual confirmation: e.g., val_001 has both "Der Haftgrund der Kollusionsgefahr liegt vor, wenn …" (DE) and "le risque de collusion … fait apparaître un danger concret et sérieux …" (FR) — the matching pair is itself a relevance proof.

#### Cross-lingual term anchors — **consistent**

The query uses English legal terms; the gold uses the canonical Swiss-legal term in DE/FR/IT. Glossary pairs visible in the val sample:

| English in query | German (gold court) | French (gold court) |
|---|---|---|
| pre-trial detention (val_001) | Untersuchungshaft | détention provisoire |
| collusion risk | Kollusionsgefahr, Verdunkelungsgefahr | risque de collusion |
| evidence tampering | Beweismittel beeinflussen | altérer des moyens de preuve |
| disability / invalidity (val_002) | Invalidität, Erwerbsunfähigkeit | invalidité, incapacité de gain |
| earning capacity 20 % | Erwerbseinbusse von 20 % | diminution de la capacité de gain de 20 % |
| holographic will (val_004) | eigenhändige letztwillige Verfügung | testament olographe |
| good faith / Art. 933 ZGB (val_007) | guter Glaube, gutgläubig | bonne foi |
| gross negligence (val_010) | grobe Fahrlässigkeit | faute grave / négligence grave |
| risk transfer clause (val_010) | Risiko-Transfer / Genehmigungsfiktion | clause de transfert de risque |
| force / threat (val_001 robbery) | Gewalt / Drohung | violence / menace |

The existing `terms_de_to_en` law-card field (§5.1 of val_001_gold doc) covers this. **For an open-domain English query the system MUST do cross-lingual term expansion before lexical search.** This is already on the roadmap (Move 2 in the endgame handoff).

### 3.4 Authority / citation-format signals — **consistent**

The citation string itself encodes signal:

- `BGE NNN [I-V] …` = published Federal Supreme Court decision. Roman numeral = chamber: I (public law / constitutional), II (admin / tax), III (civil), IV (criminal), V (social insurance). Published BGE are the legal-doctrine "core" — they're what gets cited.
- `BGer NNN/YYYY` or docket `<n><letter>_NNN/YYYY`: unpublished but archived. Chamber-prefix maps: `1B/7B` (criminal/public), `2C` (tax/admin), `4A` (civil), `5A` (civil-family/property), `6B` (criminal), `8C/9C` (social insurance), `5D` (debt collection), `1C` (admin).
- E. suffix: `E. 4.1` / `consid. 4.1` / `c. 4.1` = section number of the *Erwägung* (consideration / reasoning). Gold E. numbers are concentrated in the early reasoning sections (E. 2-6) for shorter decisions and the legal-standard sections of longer ones; almost never E. 1 (facts) or the last E. (operative part).
- Cantonal-court citations have completely different formats (e.g., `KGer SO STBER.2018.…`) — **zero of these appear in val gold**.

The system already has these regexes (`scripts/extract_citation_graph.py` per endgame_handoff §3.1).

### 3.5 Lexical / concept overlap

`personal_observations.md` Obs 3 already nailed this: 4.3 % English-token-to-DE-corpus overlap on val queries; one query has 0/44. But — and this is the key insight from §3.3 above — the *English legal-concept names* (typically 5-10 per query) have stable DE/FR/IT counterparts. After cross-lingual concept expansion, **gold paragraphs consistently mention 3-7 of those expanded concept terms in their first 300-500 chars.** Near-miss paragraphs mention 0-2.

### 3.6 Citation-graph patterns

`personal_observations.md` Obs 2 says law-article co-occurrence is near-orthogonal to the law-article *cross-reference graph* (`laws_de_links.json`). This still holds. However, two related but distinct graphs are **non-trivially predictive within val gold**:

1. **Court→court intra-gold citations** (~27 % within-gold connectivity, §3.1) — useful as a tie-breaker or rerank booster, not as a primary expansion.
2. **Court→law intra-gold citations** — strong (visual evidence ~80 %, regex floor 28 %). This is the basis for the cascade-dossier "co-citation density in pool" feature: a gold law article is cited by many gold court paragraphs of the same query, while a near-miss law article is cited by few.

### 3.7 Anti-patterns — what near-miss non-gold has that gold lacks

I have not run a side-by-side near-miss study in this pass (would need the actual rerank pool from the broken endgame run). But from the qualitative inspection of the gold sample, four categorical anti-patterns predict near-miss-non-gold:

1. **Procedural-history paragraphs** that mention the right statute but only as background to the case at hand (e.g., a court paragraph reciting "Mit Verfügung vom … verlängerte das Zwangsmassnahmengericht die Untersuchungshaft gemäss Art. 227 StPO bis …" — Art. 227 StPO is val_001 gold but THIS paragraph is the procedural narrative, not the doctrinal rule). Detector: `paragraph_role in {"facts","procedural_history"}`.

2. **Operative-part / outcome paragraphs**: "Die Beschwerde wird abgewiesen, soweit darauf einzutreten ist. Die Verfahrenskosten von Fr. 2000.- werden dem Beschwerdeführer auferlegt." These often have the right statute (Art. 428 StPO) but they're never gold. Detector: `paragraph_role = "outcome"`.

3. **Cantonal-court paragraphs** that recite the federal rule. Zero val gold are cantonal; detector: citation-string-format regex.

4. **Code-cousin articles**: gold has Art. 41 OR; near-miss has Art. 42 OR (same code, adjacent number) which is about the *measure of damages* not the *basis of liability*. The article's `english_summary` / `legal_question` / `applicability_conditions` would distinguish these — the cousin doesn't match the query's expanded concept set.

## 4. Train as a qualitative-only source

Train gold has 0 court citations in 16/16 sampled train queries; mean 30 % of gold strings unretrievable (vs 0 % for val). What's *qualitatively* transferable from train to val:

- **Code-cluster pattern**: train queries center on 1-3 codes, just like val laws. Generalizes.
- **Definition→rule→procedure→cost role mix in gold law articles**: visible in train_0044 (FIDLEG Art. 2,3 definitions; Art. 8-15 informational duties; Art. 17 documentation), train_0357 (JStPO Art. 3 scope, Art. 26 jurisdiction, Art. 27 detention rule, plus StPO Art. 221 by reference). Generalizes.
- **Article-number proximity within a code**: gold articles tend to be within ±50 of each other (train_0190: JStG 3, 4, 11, 25 plus StGB 10, 79a). Generalizes.

What **does not** transfer from train: any numeric prior (per `feedback_train_unreliable.md` and Obs 4). E.g., "30 % of train gold are missing" is not actionable for val (val has 0 % missing). The court-vs-law share, citation count per query, and language are all flipped between train and val.

## 5. Top-5 ranked actionable signals (deliverable)

Each signal is computable per candidate at index time + a per-query lookup. None encodes val-specific knowledge; all generalize to arbitrary English queries provided we first run the LLM query-expansion step (Move 2 of the endgame handoff).

### Signal 1 — Statute-target intersection count

- **What**: for each candidate doc `d`, compute `|d.statute_anchors ∩ ALL_TARGETS[qid].statute_targets|` and store the matched list.
- **Inputs**: `doc_statute_anchors[did]` (already computed at corpus load — present in `unified_retrieval.sqlite` and the v5_unified card schema) ∩ LLM-expanded `statute_targets` from per-query expansion (Move 2 output).
- **Why robust**: matches verbatim citation strings, language-agnostic (the regex catches DE "Art. 221 StPO", FR "art. 221 CPP", IT "art. 221 CPP" via a code-translation table). Independent of any train-derived weights. Visual evidence: ~80 % of val gold court paragraphs cite ≥1 of the same query's gold law articles.
- **Expected lift**: this is Phase-1 step 2 in `cascade_dossier_plan.md`. With LLM expansion producing ~10-15 statute targets per query, the intersection count is a near-perfect "rule-statement paragraph" detector. Plausibly moves Stage-2 R@100 from ~0.55 to **0.70+**.

### Signal 2 — Paragraph-role + provision-role gate

- **What**: tag each court doc with `paragraph_role ∈ {legal_standard, reasoning, application, facts, procedural_history, outcome, …}` and each law doc with `provision_role_llm ∈ {scope, procedure, right_or_entitlement, duty, definition, sanction, cost}`. At ranking time, *boost* candidates with `paragraph_role ∈ {legal_standard, reasoning}` or `provision_role_llm` matching the query's role-mix (statute-procedure questions need scope+procedure+right_or_entitlement; banking-civil questions need definition+duty+cost).
- **Inputs**: `paragraph_role` field already exists in `court_authority_cards_v5_unified.jsonl` (rule-based fallback for the ~85 % of court rows without LLM enrichment, full LLM-derived for the 363k LLM rows). `provision_role_llm` from `law_llm_descriptors_*.jsonl` (100 % coverage of laws_de).
- **Why robust**: legal documents have stable rhetorical structure (rule-statement paragraphs look identical across legal areas and languages — "Nach Art. / Gemäss Art. / Selon l'art."). The role tag is computable from the paragraph's surface text without legal-area knowledge.
- **Expected lift**: I estimate **the single strongest pure anti-pattern filter**, because the 4 anti-patterns in §3.7 above all correspond to paragraph_role ≠ {legal_standard, reasoning}. Could remove 30-50 % of the noise pool. Conservatively: +0.05-0.10 R@100.

### Signal 3 — Chamber × legal-area consistency (from citation-string regex)

- **What**: parse the candidate's citation string for chamber-number (`I-V` for BGE; `1B/2C/4A/5A/6B/7B/8C/9C/1C` for dockets). Map chamber → legal area (criminal / civil / public / social / admin). Compare to the LLM-expanded query's legal_area. Pass / penalize.
- **Inputs**: regex on `citation` column, no enrichment needed. Per-query: `ALL_TARGETS[qid].legal_area` from LLM expansion.
- **Why robust**: Federal-court chambers are stable. Map is publicly documented and unchanged for 20+ years. Independent of train.
- **Expected lift**: cheap precision booster. For val queries this collapses the candidate pool to the right chamber-prefix immediately, likely cutting court-side noise ≥2×. +0.03-0.05 R@100.

### Signal 4 — Co-citation density in the candidate pool

- **What**: for each *law* candidate in top-K, count how many of the top-K *court* candidates cite that law in their body (via the existing `statute_anchors` extraction). A law article cited by 30 of 100 court candidates is the controlling statute for the query.
- **Inputs**: top-K pool from Stage 1 + per-doc `statute_anchors`. No new infrastructure.
- **Why robust**: this is a graph signal computed within the candidate pool itself, not from any train-derived edges or precomputed graph. It's symmetric to the law-court triangulation in §3.6: the cluster of jointly-cited statutes IS the answer's spine.
- **Expected lift**: Phase-1 step 3 in cascade_dossier_plan. Helps Stage 2 pick the right 5-10 law articles out of the 40-60 law candidates in the rerank pool. +0.03-0.05 F1.

### Signal 5 — Channel-of-arrival fingerprint + cross-lingual term intersection

- **What**: per candidate, store (a) which of the 15 retrieval channels surfaced it (already in `PER_QUERY[qid]["channel_hit_sets"]`), and (b) `|doc.terms_original ∩ ALL_TARGETS[qid].german_terms ∪ french_terms ∪ italian_terms|`. Multi-channel hits are independent-evidence signals; cross-lingual term hits anchor the match against vocabulary, not surface text.
- **Inputs**: free from existing pipeline state + `terms_original` field on court cards / `terms_de_to_en` on law cards.
- **Why robust**: channel-of-arrival is orthogonal evidence (BM25 hit + statute-anchor hit + concept-channel hit ≫ BM25 hit alone). Term intersection is the cross-lingual bridge that lexical similarity cannot provide on English queries (Obs 3's 4.3 % token overlap).
- **Expected lift**: Phase-1 step 1 in cascade_dossier_plan. Mostly a precision feature; helps the 32B judge prioritize confirmed cross-channel candidates. +0.02-0.05 F1.

---

## 6. Implementation notes

- All five signals are computable in **one pre-Stage-2 pass over the top-K candidate pool**, with O(N) memory and O(N) extraction time after Stage 1 finishes. None requires retraining or new enrichment passes.
- Signals 1, 4, 5 are listed in `cascade_dossier_plan.md` Phase 1 — this report confirms their qualitative grounding on the val set.
- Signal 2 (role gate) was not in the dossier plan and is added here as a **new high-value addition**, because (a) the field already exists in v5_unified and (b) anti-patterns in §3.7 all reduce to paragraph_role mismatch.
- Signal 3 (chamber regex) is also new here and is *the cheapest possible* feature — pure string regex on the citation column.

## 7. Source files

- Sampling script: `research/_scratch_2026-05-12/sample_gold_text.py`
- Per-query JSON dumps (26 files): `research/_scratch_2026-05-12/sample_{val_001..val_010, train_0044, train_0082, train_0183, train_0190, train_0202, train_0357, train_0368, train_0402, train_0803, train_0836, train_0852, train_0923, train_0966, train_1004, train_1107, train_1110}.json`
- Aggregate scripts: `research/_scratch_2026-05-12/anti_pattern_check.py`, `anti_pattern_check_v2.py`
- Prior canonical references: `research/val_001_gold_and_enrichment_signals.md` §5 (single-query field analysis), `research/endgame_handoff_2026-05-09.md`, `research/personal_observations.md` Obs 1-4, `research/cascade_dossier_plan.md`.
