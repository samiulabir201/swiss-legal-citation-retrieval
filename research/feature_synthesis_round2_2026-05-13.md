# Feature Synthesis — What Makes a Citation Gold (Round 2 Close-Out)

**Date**: 2026-05-13
**Reports synthesized**:
- `gold_citation_legal_patterns_2026-05-12.md` (round 1 — what gold looks like)
- `near_miss_antipatterns_2026-05-13.md` (round 2 agent 1 — what near-miss looks like)
- `gold_internal_structure_2026-05-13.md` (round 2 agent 2 — aspect & graph structure)
- `linguistic_signatures_2026-05-13.md` (round 2 agent 3 — regex signatures)

## 1. The crystallized picture — what makes a citation gold

A **gold court paragraph** is a paragraph that satisfies ALL of:
1. **Federal Supreme Court** (Bundesgericht — chamber matches the query's legal area at ~95% accuracy from the citation string alone). Cantonal is never gold (102/102 val).
2. **Substantive paragraph role** — `legal_standard` / `reasoning` / `application`. Facts, procedural history, admissibility, costs, and dispositif paragraphs are never gold.
3. **Opens with an explicit rule statement** — ~80% match `^(Nach|Gemäss|Selon|Conformément à|Aux termes de) (l')?Art\. \d+`. The rule statement cites a statute that is one of the query's expanded targets (the **first 200 chars** carry the discriminating anchor — not anywhere in the paragraph).
4. **Cited by, or cites, other top-pool members** — 83% of val gold court paragraphs cite at least one gold law article of the same query; 47% cite another gold court paragraph. The gold set is a single dense star around 1-3 hub statutes for 8/10 queries.
5. **Doctrinal-language density** — `Rechtsprechung` / `jurisprudence` / "ständige Praxis" / "il est constant" markers appear in ~30% of gold vs <1% ambient. Internal cite clusters (≥3 ATF/BGE references in a parenthesis) appear in ~70% of gold vs ~6% ambient.

A **gold law article** is an article that:
1. Belongs to a code the query LLM-expanded (covers most gold).
2. Falls in a tight article-number window — ±20 covers 5/10 queries, ±50 covers 7/10. (Weak for OR/ZPO which sprawl.)
3. Has a `provision_role_llm` aligned with the query type — mix of scope/definition + procedure + rights + cost.
4. Is **cited by ≥30% of same-query gold court paragraphs** — hub statutes are central.

**Universal procedural-wrapper backbone** — `Art. 100 Abs. 1 BGG` appears in 10/10 val queries; cost/appeal articles draw from a fixed ~30-article set. Triggered when the query mentions appeal / deadline / cost.

## 2. The anti-pattern catalog — what near-miss looks like

Ranked by prevalence (Agent 1 evidence, val_001-005 pools):

| # | Anti-pattern | What it looks like | Discriminator |
|---|---|---|---|
| 1 | **Code-cousin law** | Art. 42 OR retrieved when Art. 41 OR is gold | Concept overlap normalized + provision_role_match |
| 2 | **Wrong E. within right case** | Procedural E. of a case whose E. 4 is the gold paragraph | `case_peer_count` + peer-role check (does any peer have rule-recital role?) |
| 3 | **Chamber spillover** | val_001 rank-1 is `4A_490/2017 E. 19` (civil), score 210 vs gold rank-5 at 163 (criminal-procedure query) | Chamber-class match against `legal_area(q)` |
| 4 | **Concept-tag bleed** | Asylum/civil decision tagged with "collusion risk"/"pretrial detention" | Lead-200char statute intersection (not full-text) |
| 5 | **Procedural/cost/dispositif boilerplate** | Admissibility / `Verfahrenskosten` / "wird abgewiesen" | Hard-negative regex (Agent 3 feature #5) + paragraph_role |
| 6 | **Tangential statute mention** | Statute hit, but buried mid-paragraph, not in rule-recital lead | Lead-200char statute intersection |
| 7 | Wrong-language sibling | (Subsumed by #2 — case-level dedup) | — |
| 8 | Cantonal court | (Very rare — already corpus-filtered) | court_base regex |

**The dominant noise classes** in the 50k pool are #1, #2, #5, #6. Together these likely account for >60% of high-ranked non-gold. Phase 1 of the cascade dossier must hit all four.

## 3. Consolidated feature inventory

All features below are: (a) cheap (set ops / regex / counts — no LLM call), (b) generalize across queries, (c) derive from existing pipeline state. Grouped by what they detect.

### A. Pool-context features (need ALL of top-K, computed once after Stage 1)
- **`channel_fingerprint`** — list of channels in `PER_QUERY[qid]["channel_hit_sets"]` that include this did. Free.
- **`channel_count`** — `len(channel_fingerprint)`. Orthogonal-evidence proxy.
- **`co_citation_count`** — for law `did`, count how many top-K court paragraphs have `did` in their statute_anchors. Picks hub statutes. **Empirically 83% intra-gold density; mean ≥30%-of-pool for hubs.**
- **`case_peer_count`** — for court `did`, count how many other top-K dids share `case_base`. High = leading case.
- **`case_peer_has_rule_role`** — bool: does any peer of this case in the pool have `paragraph_role ∈ {legal_standard, reasoning}`? Discriminates "wrong E. within right case."
- **`graph_neighbor_count`** — siblings (same case) + parent-court of this did in the pool.

### B. Cross-query features (query × candidate)
- **`statute_target_intersection`** — `doc_statute_anchors[did] ∩ ALL_TARGETS[qid]["statute_targets"]` (count + list).
- **`lead_statute_intersection`** — same as above but restricted to statutes appearing in **first 200 chars** of `search_text[did]`. Refinement from Agent 1; specifically attacks tangential-mention anti-pattern.
- **`concept_target_intersection`** — `_doc_to_concepts[did] ∩ ALL_TARGETS[qid]["concept_targets_en"]`.
- **`term_target_intersection_lang_matched`** — `_doc_to_terms[did] ∩ ALL_TARGETS[qid]["term_targets_<doc.lang>"]`. Language-filtered.
- **`aspect_best_match`** — for each aspect in `ALL_HYDE_ASPECTS[qid]`, score keyword overlap with doc anchors/concepts; tag with best aspect. **Promoted to Phase 1 — 9/10 queries need this.**
- **`procedural_wrapper_flag`** — bool: is this did in the fixed ~30-article procedural backbone (`Art. 100 BGG`, `Art. 422 StPO`, etc.) AND query mentions appeal/deadline/cost? Free.

### C. Per-doc structural features (computed once at corpus load)
- **`chamber`** — regex on citation string. `BGE \d+ (?P<chamber>[IVX]+)` or `^(?P<docket_prefix>\d[A-Z])_\d+`.
- **`chamber_class_match`** — `chamber ∈ ALLOWED_CHAMBERS[legal_area(q)]`. `ALLOWED_CHAMBERS` is built FROM the LLM-expanded `legal_area`, NOT hardcoded.
- **`is_federal`** — bool, from `court_base`. (Already on `doc_meta`.)
- **`paragraph_role`** — already on `doc_meta`; pass to LLM. The LLM will weight `legal_standard / reasoning > facts / procedural / costs / dispositif`.
- **`provision_role_llm`** — already on `doc_meta` for laws.

### D. Linguistic features (regex on search_text, computed once at corpus load)
Combine into a single `doctrinal_density_score`:

```python
RULE_OPENER_DE = r'^(Nach|Gemäss|Im Sinne von)\s+Art\.\s+\d+'
RULE_OPENER_FR = r'^(Selon|Conformément à|Aux termes de|En vertu de)\s+l?\'?art\.\s+\d+'
DOCTRINE_DE   = r'\b(ständige\s+Rechtsprechung|nach\s+der\s+Rechtsprechung|Lehre\s+und\s+Rechtsprechung)\b'
DOCTRINE_FR   = r'\b(la\s+jurisprudence|selon\s+la\s+doctrine|il\s+est\s+constant)\b'
INTERNAL_CITE = r'(?:ATF|BGE)\s+\d+\s+[IVX]+\s+\d+\s+(?:consid|E)\.\s+\d'   # ≥3 in one parens = cluster
RULE_VERB_DE  = r'Art\.\s+\d+.{0,40}\b(bestimmt|sieht\s+vor|regelt)\b'
RULE_VERB_FR  = r'art\.\s+\d+.{0,40}\b(dispose|prévoit|précise)\b'

# Hard negatives (subtract):
HARD_NEG_DE = r'\b(Sachverhalt|Verfahrensgeschichte|wird\s+abgewiesen|Gerichtskosten|der\s+Präsident)\b'
HARD_NEG_FR = r'\b(en\s+fait|le\s+recourant\s+fait\s+valoir|rejeté|frais\s+judiciaires|le\s+greffier)\b'
```

Score = `+rule_opener +doctrine +rule_verb +internal_cite_cluster +statute_density - hard_neg`. Estimated gold-vs-ambient AUC ≈ 0.80-0.88. Single-core 50s for the full 2.47M corpus.

## 4. Revised cascade dossier plan — Phase 1 promotion

The plan now promotes seven features to Phase 1 (was three). All are cheap; the discriminative power compounds:

1. `channel_fingerprint` + `channel_count`
2. `statute_target_intersection` (count + list) — Phase 1 confirmed.
3. **`lead_statute_intersection`** — NEW. The single sharpest refinement; attacks tangential-mention.
4. `co_citation_count` — Phase 1 confirmed. Picks hub statutes.
5. **`paragraph_role` passed forward** — NEW. Most direct anti-pattern filter (#5 procedural).
6. **`chamber_class_match`** — NEW. Free; ~2× court-side noise reduction.
7. **`aspect_best_match`** — NEW (promoted from Phase 2). Universal — 9/10 queries need it.

Phase 2 (layer if Phase 1 lifts R@100 ≥ 0.7):
8. `case_peer_count` + `case_peer_has_rule_role` — attacks wrong-E within right case.
9. `concept_target_intersection` + `term_target_intersection_lang_matched`.
10. `doctrinal_density_score` — precision booster, not primary signal.
11. `graph_neighbor_count`.
12. `procedural_wrapper_flag`.

## 5. What's still uncertain

- **Agent 1 only had pool-level data for val_001-005**; the prevalence ratios for anti-patterns #1-6 across val_006-010 are extrapolations. Worth running the same diagnostic on the missing 5 once the warm-boot snapshot lands.
- **Combined-feature AUC** — each feature has individual stats; we don't have a measurement of how they stack. Best path: implement Phase 1 (7 features) and measure Stage 2 R@100 lift before adding more.
- **The Untitled75.ipynb F1=0.777 reference** — what its 7-category judge + zone routing did differently is not fully extracted. Worth a focused dig if Phase 1 doesn't hit ≥0.6 macro F1.
- **`provision_role_llm` distribution on the law side** — we know it exists; we haven't confirmed it's populated densely enough on the val gold law articles for it to be a strong feature.

## 6. Bottom line

We now have:
- A coherent anti-pattern taxonomy (8 categories) with named, cheap discriminators for each.
- A coherent gold-structure model (chamber, role, rule-opener, in-pool co-citation, aspect coverage).
- A 7-feature Phase-1 plan with empirical grounding for every feature.

The plan is ready to implement. The bottleneck moves from "do we know enough about the data?" to "does the LLM judge use the dossier well?" — which is an implementation + measurement question, not a research question.
