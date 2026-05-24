# Near-Miss Anti-Pattern Forensics — Round 2

_Generated 2026-05-13. Companion to `gold_citation_legal_patterns_2026-05-12.md` (round 1: what gold looks like). This round characterizes the **high-rank non-gold population** the LLM judge has to discriminate gold from. Evidence: per-query ranked candidate pools (200-1000 per query) for 5 val queries, cross-referenced against the round-1 gold dossier. Per-candidate features (citation, retrieval_preview with concept tags + statute anchors, fused_score, is_gold) read from `artifacts/eval/val_candidate_sets.jsonl` (5 queries × top-200) and `artifacts/enrichment_recall_stress_val_001_top1000.csv` (val_001 × 1000 with concept previews)._

## 0. Available evidence and its limits

I have ranked, scored candidate pools for the following val queries with `is_gold` flags computable against the round-1 gold lists:

| Query | Pool source | Pool depth | R@200 | Notes |
|---|---|---:|---:|---|
| val_001 | `enrichment_recall_stress_val_001_top1000.csv` | 1000 | n/a | Rich: includes concept tags, source_family, statute anchors in `retrieval_preview` |
| val_001-005 | `artifacts/eval/val_candidate_sets.jsonl` | 200 each | 0.07-0.12 | Citation + family + channel_ranks; no preview text |
| val_006-010 | not available in current snapshot | – | – | The newer `eval/val_summary.json` only ran 5 of 10 |

The `eval/val_candidate_sets.jsonl` run is a **degraded** pipeline (BM25 + statute only, no vector, R@200 ≈ 0.05-0.12); but for an anti-pattern study this is **better than the fully-recovered pool** because every rank-1 to rank-50 hit is **a near-miss by construction** — gold is sparse, near-miss is everywhere. Where I cite a fully-recovered candidate (val_001 enrichment_stress pool), `is_gold=True` candidates are explicitly distinguished from `is_gold=False` near-misses; the rank ordering used is the model's, not retrospective filtering. The dropped_gold CSV (`artifacts/eval/val_dropped_gold.csv`) gives the complement (gold that fell out).

I will not extrapolate to val_006-010 numerically. Patterns are abstracted from val_001/002/003/005 with cross-checks against round-1 gold dossiers (which cover all 10).

## 1. The corpus the judge faces

### 1.1 Round-1 baseline (gold)

From round 1: gold court paragraphs are (i) Federal Supreme Court only, (ii) chamber-aligned with the legal area, (iii) ≥800 chars in 83% of cases, (iv) opened by "Gemäss/Nach Art. X / Conformément à l'art. X" rule-statement boilerplate, (v) cite 2-7 distinct `Art. N CODE` references in body, (vi) cite ≥1 of the same query's gold law articles at the regex-floor rate of 28% (visual rate ~80%). Gold law articles cluster in 1-3 codes and ±50-article window.

### 1.2 What the pool actually surfaces

Looking at the val_001 top-1000 (criminal procedure / collusion-risk query, gold = 19 StPO+BGG+StBOG laws, 23 BGer criminal court paragraphs):

- Rank **1-4** are all `4A_*` (civil chamber) and `5F_*` (civil revision) court paragraphs: `4A_490/2017 E. 19`, `5F_8/2018 E. 3`, `4A_94/2009 E. B`, `4A_166/2021 E. 4.4.2`. Their `retrieval_preview` shows topic tag "civil obligations, contract, commercial, and banking law". Chamber is wrong, score is 210 (gold rank-5 `7B_301/2024` scores 163).
- Ranks **5-9** mix gold (`7B_301/2024`, `7B_69/2024`, `7B_12/2025`, `1B_357/2022`, `1B_90/2021`) with non-gold of similar shape (`4P.162/2003 E. A`, `BGE 133 I 234 E. 08.25`).
- Rank **207**: `BGE 137 IV 122 E. 2` — **same case** as four val_001 gold (E. 4.1, 4.2, 6.2, 6.4) but the wrong E.; E. 2 is the facts/procedural summary.
- Many `7B_*` and `1B_*` mid-rank entries (`7B_984/2023 E. 3.3.2`, `7B_448/2023 E. 3.3.2`, `1B_540/2022 E. 5.1`) are **chamber-correct, topic-correct, and cite Art. 221 StPO**, yet are not gold — they are downstream citing decisions, not the doctrinal source.

So the high-rank non-gold breaks into four very different shapes that the Stage 2/3 LLM has to disambiguate.

## 2. Anti-pattern catalogue

Each entry: name → diagnostic test in plain English → concrete example from real candidates → cheap detector.

### 2.1 Chamber-mismatch civil-spillover (most prevalent)

The single largest near-miss class in val_001 (criminal procedure) is **civil-chamber paragraphs (`4A_`, `5A_`, `5F_`, `4P.`)** that surface high because their text contains BM25-matching English-translatable legal vocabulary (`Beweis`, `Sachverhalt`, `Verfahren`, `Beschwerdeführer`) and/or because some upstream concept tagger tagged them with "collusion risk" or "evidence assessment" generically.

Real high-rank examples in val_001 top-1000 (score / rank / gold flag):

- `4A_490/2017 E. 19` — score 210.4 / rank 1 / not gold. Retrieval preview: *"4A_490/2017 | civil obligations, contract, commercial, and banking law | …"*
- `5F_8/2018 E. 3` — score 187.3 / rank 2 / not gold. *"civil law | Holds (a) that …"*
- `4A_94/2009 09.06.2009 E. B` — score 175.2 / rank 3 / not gold.
- `4P.162/2003 21.11.2003 E. A` — score 159.2 / rank 7 / not gold. *"Schiedsklausel | Art. 22"*

Same pattern appears at the head of val_003 (criminal): rank 1-2 are `4A_221/2024 E. I` and `4A_223/2024 E. I` — purely-civil decisions, ranked above all `7B_*` criminal candidates. Same in val_002 (social insurance, expected chamber 8C/9C/V): top of the BM25-only run has `4A_490/2017 E. 2`, `4A_258/2023 E. 5.3`, `4A_388/2012 E. B`, `4A_232/2022 E. 4` — all civil — interleaved with the chamber-correct `9C_801/2012`, `9C_308/2021`, `9C_81/2013`, etc.

**Diagnostic**: chamber prefix from the citation regex does not intersect with the query's legal-area chamber set.

**Cheap detector**:
```
chamber_class(d) ∈ ALLOWED_CHAMBERS[legal_area(q)]
```
where `chamber_class()` is a 5-line regex over the citation column (already in round 1 §3.4) and `ALLOWED_CHAMBERS` is a fixed lookup from LLM-derived query legal_area → chamber set (criminal→{IV,1B,6B,7B,BStrK}, civil→{III,4A,5A,5F}, public→{I,1C,2C,2D}, social→{V,8C,9C,K,I 999/YY pre-2007}, admin→{II,1C,2C}, …). No train priors.

**False-positive risk**: low. Cross-area citations (criminal procedure citing civil-law BGE) exist but are rare in the round-1 gold sample (0/102 val gold court paragraphs were cross-chamber). The detector should be a **boost / penalty**, not a hard filter, to avoid dropping the 1-in-100 valid cross-area gold.

**Prevalence**: very high in our top-50 non-gold population — roughly half of the val_001 top-20 non-gold are chamber-wrong.

### 2.2 Wrong paragraph within the right case (very high precision, high frequency)

A gold court citation is at the granularity `BGE 137 IV 122 E. 4.2`. The same decision has E. 1 (parties), E. 2 (Sachverhalt), E. 3 (procedural admissibility), E. 5, E. 7 (operative part), etc. — typically 6-10 paragraphs per decision. The retrieval system surfaces **all** paragraphs of a relevant case because they share the same case-level vocabulary and the same case-level concept tags. Only one or two are doctrine; the rest are recital.

Real example in val_001:

- Gold: `BGE 137 IV 122 E. 4.1`, `E. 4.2`, `E. 6.2`, `E. 6.4` (four E.'s of the same case).
- Near-miss in top-1000: `BGE 137 IV 122 E. 2` at rank 207 (score 69.3). Body of E. 2 (from round-1 sample inspection of analogous cases) is purely Sachverhalt/Verfahrensgeschichte recital — it does *not* state the Art. 221 lit. b doctrine; it merely says the lower court ordered detention. Indistinguishable from gold at the case level; differentiable at the E.-section level.

Another example: val_002 top-200 contains `9C_177/2015 E. 1`, `E. 2`, `E. 4.1` ranked 83/86/88. Within the same case the E. numbers and chamber are identical; the body content (rule-statement vs facts) is the discriminator.

**Diagnostic**: identical case-base citation as a gold candidate but different E. number; OR `paragraph_role ∈ {facts, procedural_history, operative_part, costs, admissibility}`.

**Cheap detector**:
```
case_base(d) = citation_regex_strip_E(d.citation)   # "BGE 137 IV 122"
peer_count(d) = |{d' in pool : case_base(d')==case_base(d)}|
role_score(d) = +1 if paragraph_role(d) in {legal_standard, reasoning, application} else -1
```
A candidate that is one of N≥3 paragraphs of the same case in the pool, and has `paragraph_role` not in the doctrine set, is overwhelmingly likely to be a near-miss sibling.

**False-positive risk**: medium-low. Some gold E.'s ARE application or facts (round-1 §3.1 — about 1 in 6); the detector must be a soft signal that the LLM can override when the text genuinely states the rule.

**Prevalence**: ubiquitous. Every case-level gold draws 3-9 sibling paragraphs into the pool. Concretely, if 23 val_001 gold court paragraphs cover ~15 distinct decisions, the pool likely contains 90-150 sibling-paragraphs from those same decisions, of which 23 are gold and the rest are near-miss.

### 2.3 Procedural / cost / dispositif near-miss (frequent and high-confidence drop)

Specific instances of 2.2 where the role classification is mechanical:

- Cost-allocation paragraphs at end of decision: cite Art. 428 Abs. 1 StPO (which IS val_001 gold as a *law*), but the **court paragraph** that applies the rule ("Die Verfahrenskosten von Fr. 2'000.- werden dem Beschwerdeführer auferlegt …") is never gold itself.
- Admissibility / Eintreten paragraphs: cite Art. 100 Abs. 1 BGG (val_001 gold law) in the formula "Die Beschwerde wurde frist- und formgerecht erhoben …" — short, formulaic, never gold.
- Dispositif: "Die Beschwerde wird abgewiesen, soweit darauf einzutreten ist." — 30-150 chars, always non-gold.

Real example in val_001 top-1000:

- `4A_446/2022 E. 2` at rank 11 — preview: *"Art. 190 291 Art. 77 Abs. 1 Bst. a BGG | Swiss Federal Supreme Court consideration in civil oblig…"*. Hits because it contains a BGG admissibility statute string; substance is generic admissibility.
- `4A_152/2024 E. 1` at rank 6 in val_003's pool — admissibility paragraph (Eintreten).
- `BGE 137 IV 122 E. 8.25` (val_001 rank 8) — looks like a near-end E., almost certainly cost or dispositive.

**Diagnostic**: very short paragraph length AND citation of "admin / cost" statutes (BGG admissibility, StPO 422-428, ZPO 95-110, OG 64) without any rule-statement opening.

**Cheap detector**:
```
len(text) < 350  AND  startswith_admin_pattern(text)
```
where `startswith_admin_pattern` matches "Die Beschwerde …", "Le recours …", "Il ricorso …", "Die Kosten …", "Auf die Beschwerde …", etc.

**Combined with 2.2**: this is a paragraph_role tag of `admissibility`, `costs`, `dispositif`.

**False-positive risk**: very low. None of the 102 val gold court paragraphs is short admin/dispositif.

### 2.4 Code-cousin law article (most insidious; high LLM-confusion risk)

Gold is `Art. 273 Abs. 1 ZGB`; pool surfaces `Art. 270a Abs. 3 ZGB`, `Art. 272 ZGB`, `Art. 274 Abs. 1 ZGB`, `Art. 275a Abs. 1 ZGB`. All are in the same code (ZGB), same article window (270-275), same subject area (Eltern-Kind-Beziehungen / parent-child relations). They look identical at every coarse feature (chamber n/a, code match, paragraph_role n/a, language DE). The discriminator is the article's *operative content*.

Real examples — val_005 top-200 (gold: ZGB Art. 133, 273 Abs. 1, 274 Abs. 2, 285 Abs. 1; near-misses):

- Rank 14: `Art. 298b Abs. 3ter ZGB` — joint parental authority (verwandte Bestimmung, but Abs. 3ter is the procedural extension, not the visitation rule).
- Rank 17: `Art. 270a Abs. 3 ZGB` — surname rules for child.
- Rank 20: `Art. 301a Abs. 3 ZGB` — domicile change rules.
- Rank 32: `Art. 134 Abs. 1 ZGB` — modification of arrangements (adjacent article to gold Art. 133).
- Rank 34: `Art. 275a Abs. 1 ZGB` — information rights (adjacent to gold Art. 274; subject = adjacent right).
- Rank 50: `Art. 318 Abs. 2 ZGB` — child property management (different doctrine within parent-child).

Same pattern val_001 (gold = many StPO articles): top-50 of any pool has dozens of "Art. NNN StPO" entries in the 100-450 range, of which 15 are gold and 30+ are near-miss neighbours.

**Diagnostic**: same code as a gold candidate, same ±50 article window as a gold candidate, but the article's controlling concept (from `provision_role_llm` + `legal_question` + `english_summary`) does not match the query's expanded concept set.

**Cheap detector**:
```
concept_overlap(d, q) = |concepts_en(d) ∩ concept_targets_en(q)|   # uses Move-2 LLM expansion
                       / max(1, |concepts_en(d)|)
provision_role_match(d, q) = provision_role_llm(d) ∈ required_roles(q)
```
The `concepts_en` field already exists per law card. Both `concept_targets_en` and `required_roles` come from the per-query LLM-expansion step (no train priors). A code-cousin shows up as same-code, same-range, but **low concept_overlap** and/or wrong `provision_role_llm`.

**False-positive risk**: medium. Code-cousins sometimes ARE gold (round-1 §3.2: val_006 OR gold spans Art. 1-20, 41, 97-101, 245-248, 363-398 — a wide spread within OR, so a candidate at Art. 50 OR is plausibly gold). The detector must be probabilistic — a low concept-overlap is a *down*-weight, not a kill.

**Prevalence**: dominant in the law-side of the pool. For any query with N gold laws clustered in code C, the pool of "Art. * C" candidates is typically 50-200, of which only N are gold.

### 2.5 Tangential statute mention in a court paragraph (the "citation regex false positive")

The retrieval pipeline surfaces a court paragraph because regex/BM25 finds "Art. 221 StPO" inside it. But the paragraph cites Art. 221 only in passing — as a one-line reference inside a longer paragraph about a different doctrine, OR as part of a verbatim quotation of an opposing party's brief.

Real example — val_001 top-1000:

- `6F_5/2022 E. 2.3.2` (rank 42, score 122.9) — preview: *"6F_5/2022 | criminal law | Swiss Federal Supreme Court consideration in criminal law. | unpublished_federal_decision reasoning_consideration multi_consideration_decision …"*. Likely cites Art. 221 because the procedural posture mentioned it, but `6F_*` is a revision docket — its substantive doctrine is res judicata, not detention.
- `BGE 138 IV 92 E. 14.21` (rank 34) — published BGE IV ✓ but the E. number "14.21" is far down the reasoning, suggesting a specific factual sub-issue rather than the rule recital.

In val_003 (Rivera/criminal case, gold heavily StPO): rank 32 is `Art. 224 Abs. 1 StPO` — about admission of evidence — and rank 45 is `Art. 197 Abs. 1 StPO` — about coercive-measure proportionality. Both are StPO and tangentially-mentioned in detention jurisprudence; neither is gold for this specific Rivera-fact-pattern query.

**Diagnostic**: statute anchor exists (`Art. 221 StPO` appears in candidate's body) BUT the paragraph's lead sentence and `paragraph_role` indicate a *different* doctrine.

**Cheap detector**: this is precisely what the **statute-target intersection count** dossier feature (Phase 1 step 2 in `cascade_dossier_plan.md`) was designed to catch — except it must be *weighted by lead position*. A statute anchor in the first 100 chars (rule recital) is high signal; an anchor at char 1500+ of a 2000-char paragraph is a tangential mention.

```
lead_statute_anchors(d) = statute_anchors found in first 200 chars of d.text
intersection_lead(d, q) = |lead_statute_anchors(d) ∩ statute_targets(q)|
```

**False-positive risk**: low when measured against lead position. Application paragraphs (round-1 §3.1) do cite the statute later, not in the lead — these would be missed; but they're a minority of gold and round-1 §3.1 says "the statute reference is usually present in the lead even for application paragraphs."

### 2.6 Wrong-language sibling of a gold case (rare but observed)

Same federal case, both DE and FR versions of the same E. exist as separate corpus rows. The DE version may be gold; the FR is then a sibling near-miss (or vice versa), depending on which gold list one is matching.

Cross-checked the val_001 dropped_gold list against the round-1 gold sample: FR gold for val_001 includes `1B_210/2023 E. 4.1`, `1B_536/2018 E. 5.1`, `BGE 139 IV 270 E. 3.1`, `BGE 133 I 168 E. 4.1`, `BGE 143 IV 168 E. 5.1` — so the FR siblings of these cases (different E. numbers, same case, same language) would be near-misses; the DE-language equivalent decision text (where it exists in the corpus as the "Italian/French/German" parallel) would also be a near-miss.

This pattern is mostly a corollary of 2.2 + a language axis. Round-1 gold contains both DE and FR; the discriminator is again `paragraph_role` and `case_base`, not language alone.

**Detector**: not worth a separate feature. Subsumed by 2.2.

### 2.7 Concept-tag bleed (LLM-tagged "collusion risk" on civil decisions)

A high-rank class of mismatch: the upstream concept tagger (used to tag corpus paragraphs with concepts like "collusion risk", "pretrial detention", "due process") has applied "collusion risk" to civil-procedure paragraphs that mention witnesses or evidence in a *civil* setting. The retrieval channel that uses concept-overlap then surfaces these as high-scoring matches.

Real example — val_001 top-1000:

- `BGE 133 I 234 E. 08.25` (rank 8) — preview tags: *"ECHR and fundamental rights asylum and refugee status evidence assessment international mutual …"*. Domain is asylum/refugee, not criminal procedure. Surfaces because "evidence assessment" overlaps with the query's concept set.
- `BGE 148 III 330 E. 5.159` (rank 202) — preview: *"civil law | … ECHR and fundamental rights asylum and refugee status constitutional rights criminal sentencing due process fundamental rights"* — tags are aspirational, content is civil arbitration.
- `BGE 134 II 10 E. 2004` (rank 216) — *"administrative, tax, migration … migration and residence pretrial detention recidivism"* — uses "pretrial detention" tag but applied to immigration detention, not criminal procedure.

**Diagnostic**: concept-tag overlap is high but the candidate's source_family / topic_area / chamber is incompatible with the query's legal area.

**Cheap detector**: cross-check legal_area derived from chamber regex (§2.1 detector) against legal_area derived from concept tags. A divergence between the two means the concept tagger is contaminating across domains.

**False-positive risk**: low — when the chamber says "civil" and the concept tag set claims "criminal procedure", the chamber wins for almost-all queries.

**Prevalence**: substantial — this is what's putting `4A_*` and asylum BGE on the val_001 top-10.

### 2.8 Operative-part / outcome paragraph (specific sub-case)

Round-1 §3.7 anti-pattern 2. Confirmed in pool data: `4A_446/2022 E. 2` (val_001 rank 11) preview lead is *"Art. 190 291 Art. 77 Abs. 1 Bst. a BGG"* — a list of admissibility article references, no rule prose. Discriminator: lead text matches an "outcome/admissibility/cost" sentence template, OR text is ≤300 chars with citations-only-no-prose density.

Detector: collapses into 2.3 (procedural near-miss) + paragraph_role tag.

### 2.9 Cantonal-court paragraph (very rare in val pool)

Round-1 confirmed zero cantonal gold in val. Spot-checking the candidate pools, cantonal-court citation strings (`KGer …`, `OGer …`, `BezGer …`, `Kantonsgericht …`) appear at low rank or not at all. The pre-filter that limits the corpus to BGer + BVGer is doing its job.

Detector: trivial citation-regex flag. Penalize aggressively.

**Prevalence**: low in current pool (corpus-level filter handled it). Keep as a hard-filter for safety against future corpus changes.

## 3. Aggregate prevalence ranking and discriminator priorities

For the 50 highest-ranked non-gold candidates per val_001-005 pool (~250 near-misses total, cross-tabulated against round-1 gold lists), the qualitative ranking is:

| Anti-pattern | Prevalence | Best discriminator | False-pos risk |
|---|---|---|---|
| 2.4 Code-cousin (law side) | very high | `concept_overlap` ∩ `provision_role_match` (LLM-expanded) | medium |
| 2.2 Wrong E. of right case | very high | `paragraph_role` + `peer_count(case_base)` | medium-low |
| 2.1 Chamber spillover | high | `chamber_class(citation) ∈ ALLOWED_CHAMBERS[legal_area(q)]` | low |
| 2.7 Concept-tag bleed | medium-high | chamber vs concept legal-area cross-check | low |
| 2.3/2.8 Procedural/cost/dispositif | medium-high | `paragraph_role ∈ {admissibility, costs, dispositif}` OR short+admin-template | very low |
| 2.5 Tangential statute mention | medium | `intersection_lead` (anchors in first 200c) | low |
| 2.6 Wrong-language sibling | low | subsumed by 2.2 | n/a |
| 2.9 Cantonal-court | very low | citation-format regex hard filter | very low |

## 4. The 5 features most likely to discriminate

These are computable from per-doc data + per-query LLM-expansion ONLY. No train priors. No query-specific hardcoding.

1. **`chamber_class_match`** — bool. Citation regex extracts chamber from `BGE NNN [I-V] …` or `<n><letter>_NNN/YYYY` docket prefix; compared to `ALLOWED_CHAMBERS[legal_area(q)]`. **Targets**: 2.1, 2.7. Computed from citation string only. (Already in round-1 signal 3.)

2. **`paragraph_role_alignment`** — categorical. Existing `paragraph_role` field on court cards (`legal_standard`, `reasoning`, `application`, `facts`, `procedural_history`, `admissibility`, `costs`, `dispositif`). Strong positive for `{legal_standard, reasoning, application}`; strong negative for `{facts, admissibility, costs, dispositif}`. **Targets**: 2.2, 2.3, 2.8. (Round-1 signal 2.)

3. **`case_peer_count` + `is_peer_to_gold_candidate`** — int. For each pool candidate, count how many other pool candidates share the same `case_base` (citation stripped of E.); flag whether one of those peers is itself a top-K candidate with `paragraph_role = legal_standard`. Many peers + I'm-the-facts-E. ⇒ near-miss; few peers OR I'm-the-rule-E. ⇒ gold-likely. **Targets**: 2.2, 2.6.

4. **`lead_statute_intersection`** — int. `statute_anchors found in first 200 chars(d.text) ∩ statute_targets(q)`. Distinguishes rule-recital ("Gemäss Art. 221 StPO ist…") from tangential mention ("…ferner ist auf Art. 221 StPO zu verweisen…"). **Targets**: 2.5. Strict subset of Phase 1 step 2 in `cascade_dossier_plan.md` with the lead-position refinement.

5. **`concept_overlap_normalized` × `provision_role_match`** — float × bool, only for law candidates. `|concepts_en(d) ∩ concept_targets_en(q)| / |concepts_en(d)|` (avoid penalising laws with many concepts) AND `provision_role_llm(d) ∈ required_roles(q)`. Together they distinguish a same-code, same-window article that addresses the right doctrine from one that addresses an adjacent doctrine. **Targets**: 2.4 (the dominant law-side anti-pattern, currently unprotected).

These five, added per-candidate to the Stage-2 dossier block, give the 32B judge a structured discriminator triple:
- *chamber* (do I cite from the right body of law?)
- *role* (am I rule, fact, or boilerplate?)
- *case-peer* (am I the canonical E. of the case, or its sibling?)
- *lead anchor* (is the statute hit in my rule-recital lead, or buried?)
- *concept-role* (does my doctrine match the query, not just my code/article number?)

Cost: ~30 extra tokens per candidate. Computable in one O(N²/2) pass over top-100 + O(N) anchor scan on first 200 chars of each candidate. Same logic for any val/production query — no per-query priors.

## 5. Calibration notes & open questions

- **Asymmetry**: features 1-3 are court-side; 4 is mostly court-side; 5 is law-side. The dossier needs to cleanly tag a candidate as `family=law` or `family=court` (already present in candidate_sets) so the Stage-2 prompt applies the right discriminator block. A `family=law` candidate with paragraph_role missing is normal; a `family=court` candidate with provision_role missing is normal. The LLM gets distracted by missing fields if not signposted.
- **Application paragraphs**: round-1 §3.1 noted application paragraphs ("Im vorliegenden Fall …") are ~17% of gold court — they violate detector 4's lead-position rule because they recite facts first, statute later. Solution: paragraph_role tag `application` should soften the lead-anchor penalty. The role tag carries this.
- **Code coverage of `provision_role_llm`**: round-1 §5 stated 100% of `laws_de` has LLM-derived `provision_role_llm`. Court paragraphs only have rule-based fallback for ~85% (val_001 doc §5.2). For the 15% lacking LLM-derived `paragraph_role`, detector 2 will be soft; detectors 1, 3, 4 stay strong. Acceptable.
- **No-hardcoding compliance**: `ALLOWED_CHAMBERS` is a single static table (Federal-court chamber → legal-area, unchanged for 20+ years). It is not query-specific or train-derived; it is a corpus-structural fact. `required_roles(q)` and `legal_area(q)` come from per-query LLM expansion — no train data touches the system. This satisfies feedback_no_query_specific_hardcoding.md and feedback_train_unreliable.md.

## 6. Source files

- Round-1 dossier: `research/gold_citation_legal_patterns_2026-05-12.md`
- Cascade plan: `research/cascade_dossier_plan.md`
- Per-query ranked pools used in this round:
  - `artifacts/enrichment_recall_stress_val_001_top1000.csv` (val_001, depth 1000, with retrieval_preview)
  - `artifacts/eval/val_candidate_sets.jsonl` (val_001-005, depth 200, with channel_ranks)
  - `artifacts/eval/val_dropped_gold.csv` (gold absent from pool)
- Gold dossiers per query: `research/_scratch_2026-05-12/sample_val_001.json` … `sample_val_010.json`
