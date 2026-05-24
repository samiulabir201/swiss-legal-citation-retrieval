# Cascade Forensic Case-File Plan

**Status**: agreed approach — not yet implemented.
**Date**: 2026-05-12
**Context**: top-50k fusion pool reaches 0.893 macro recall. Need to discriminate gold from a 50k pool down to 20-40 final picks with F1 = 0.6-0.8.

## The Problem

Stage 2/3 LLM currently sees the same evidence the cross-encoder reranker sees: citation + 5 metadata fields + a text excerpt + ~10 concepts + ~12 statute anchors. The 32B model has no fundamentally new evidence to work with — it's asked to do the same matching task with the same fields, just slower. Result: Stage 2/3 don't materially lift over Stage 1.

## The Fix: Forensic Case File per Candidate

Pre-compute a per-candidate evidence dossier from existing pipeline state. Show the LLM the *pre-digested signal*, not the raw fields. The LLM's task changes from "guess relevance from text" to "confirm the dossier against the text" — much harder to hallucinate around.

## Phase 1 (immediate — measure before layering more): 3 highest-signal additions

1. **Channel-of-arrival fingerprint** — list which of 15 channels surfaced this doc + how many. Already in `PER_QUERY[qid]["channel_hit_sets"]`. Free. Orthogonal channels = orthogonal evidence; multi-channel hits are structurally stronger than single-channel.

2. **Statute-target intersection** — `doc_statute_anchors[did] ∩ ALL_TARGETS[qid]["statute_targets"]` with count + list per candidate. Cheap set intersection. Tells the LLM "this doc cites 2 of your 12 expanded query statutes" instead of making it scan-and-match.

3. **Co-citation density in pool** — for each law article in top-100, count how many of the top-100 *court paragraphs* cite it (via their `statute_anchors`). O(N²) one-pass over top-100. A law cited by 40/100 top court paras is the controlling statute for this question.

After adding these three to the Stage 2 prompt block, measure mean R@100 before/after. If lifts current ~0.55 fusion-baseline to ≥0.7, the dossier hypothesis is validated and we layer the rest.

## Phase 2 (layer on if Phase 1 validates)

4. Concept-target intersection (`_doc_to_concepts[did] ∩ ALL_TARGETS[qid]["concept_targets_en"]`)
5. Term-target intersection by document language
6. Multi-aspect coverage tag — which decomposed aspect this candidate best addresses
7. Stage 1 reranker score passed forward to Stage 2 (currently dropped)
8. Graph-neighbor density in pool (siblings, parent court, co-citations of same case)
9. Citation-format flags from regex on citation string (BGE prefix, court level, article magnitude)
10. Paragraph-role × query-type alignment (LLM uses priors implicitly — just pass forward)

## Implementation notes

- **Instrument `run_channels()`** in [v7_4_fixes/_cell34_body.py](v7_4_fixes/_cell34_body.py) to also persist per-channel *rank* (currently only doc-id sets survive RRF). ~5 lines.
- **Pre-compute per-candidate cross-features ONCE** after Stage 1 finishes, store on `PER_QUERY[qid]`. Both Stage 2 and Stage 3 read from there — no recomputation.
- **Evidence-quote invariant stays** at Stage 3. The cross-features guide attention; the verbatim-quote requirement still gates against hallucination.

## What the new Stage 2 candidate block looks like

```
[1] BGE 142 III 296 E. 4.1  (DE, court, BGer, role=reasoning)
    surfaced by: law_direct_match, statute_backprop, concept_en — 3/15 channels
    cites: {Art. 41 OR, Art. 44 OR, Art. 99 BGG}   ← 2 of your 12 query targets
    concepts: {damages calculation, contributory negligence}   ← 2 of your 6 query concepts
    addresses aspect #2 ("measure of damages")
    cited by 14 of your top-100 court paragraphs
    stage1 reranker: 0.87
    text (400c): "Bei der Bemessung des Schadens ist nach Art. 41 OR …"
```

~120 tokens vs current ~250, and denser in evidence.

## No-hardcoding guarantee

Every signal in the dossier comes from per-query LLM expansion (statute_targets, concept_targets, HyDE aspects) intersected with per-doc data computed once at corpus load. No code constants (no DE_LEX, no statute clusters). No train-derived weights. Same logic for any val/production query.

## Decision pending

Phase 1 only after the user runs the upstream pipeline once (cells 0-46) and saves the snapshot. Then Phase 1 enrichments slot in between Stage 1 and Stage 2 as a new pre-compute cell.
