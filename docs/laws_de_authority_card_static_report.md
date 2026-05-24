# Law Authority Card Static Extraction Report

Generated from a repo scan and a full static run over `data/laws_de.csv`.
No train, validation, test, or gold labels were used.

## 1. Repo And Court Pipeline Findings

The current plan is enrichment-first RAG. The old segment-lattice funnel used coarse filters such as law code, BGE division, and docket prefix; this was not enough to rank inside very large court buckets. The new approach turns every law and court row into a structured authority card, then retrieves by a union of fielded BM25, vector text, exact anchors, graph/adjacent expansion, and authority boosts.

The court pipeline already implements this shape:

1. `scripts/extract_citation_graph.py` extracts canonical source citations, text references, classified segments, and source-to-reference edges into `data_insights/*`.
2. `scripts/build_court_authority_cards.py` builds deterministic court v4 cards from `data/court_considerations.csv`, `artifacts/segment_lattice_v3.sqlite`, and `data_insights/citation_graph_extracted.sqlite`.
3. `scripts/identify_court_enrichment_targets.py` selects 363,258 high-value or weakly covered court rows from 2,476,315 v4 cards.
4. `outputs_from_363k_run/court_llm_descriptor_extractor_blackwell_optimized__3__(1).ipynb` runs Qwen3-8B-AWQ with vLLM over the target cards and emits compact query-neutral legal descriptors.
5. `scripts/merge_llm_enrichment_into_v4_cards.py` merges the LLM descriptors with static v4 cards and normalizes both populations into `artifacts/court_authority_cards_v5_unified.jsonl`.

Important court v5 counts:

- Total unified court cards: 2,476,315
- LLM plus static cards: 363,257
- Static-only cards: 2,113,058
- Normalized retrieval views include: `semantic_concepts_en`, `topic_path`, `original_terms_view`, `statute_anchor_view`, `case_anchor_view`, `legal_rule_view`, `fact_pattern_view`, `procedural_view`, `authority_view`, `raw_context`, and `citation_view`.

The key design lesson for law cards is the same: deterministic metadata is the source of truth for citations, law codes, article units, outgoing references, and adjacency; the LLM should only add query-neutral English legal meaning.

## 2. laws_de.csv Analysis And Final Schema

Input profile:

- Rows: 175,933
- Columns: `citation`, `text`, `title`
- Empty text rows: 0
- Empty title rows: 0
- Text length: median 181 chars, p90 428, p95 594, p99 1,274, max 22,348
- Citation family: 175,933 statute-article law rows
- Original classified rows with unresolved or Unicode-truncated law codes were repaired from the exact citation string during card building.

Static extraction reliability:

- Formally complete rows: 175,933 / 175,933
- Law-code repairs from exact citation text: 4,117
- Rows with outgoing text references: 31,219 source citations, 45,165 edges
- Rows with incoming references: 10,409
- Title has law title plus section path in 168,315 rows; 7,618 rows have only the base title.

Static semantic tiers:

- Strong dictionary/title/code signal: 138,720
- Partial static signal: 23,624
- Formal-only signal: 13,589
- Complete English legal semantic context by static code alone: 0
- Rows needing LLM for complete semantic context: 175,933

The zero count is intentional. Static code can extract structure, labels, reference anchors, and cue-based roles, but it cannot reliably translate and legally interpret each provision's operative meaning into English legal questions, rules, applicability conditions, exceptions, and concepts. The static fields are useful fallback retrieval signals, not a replacement for LLM semantic enrichment.

Final static law-card schema:

- Identity: `_source_row`, `citation`, `source_family`, `language`
- Formal structure: `structural.article`, `structural.units`, `structural.law_code`, `structural.law_code_family`, `structural.granularity`
- Title metadata: `law_title`, `title_section_path`, `title_metadata.source_type`, `title_metadata.enactment_date`, `title_metadata.law_aliases`, `title_metadata.systematic_collection_sector`
- Static semantics: `legal_area_static`, `legal_area_candidates`, `domain_labels_en`, `issue_labels_en`, `matched_terms_multilingual`, `provision_roles`, `static_semantic_signal`
- Anchors: `statute_anchors`, `official_references`, `court_case_anchors`, `other_reference_anchors`, `incoming_reference_count`, `incoming_reference_examples`, `outgoing_reference_count`, `adjacent_citations`
- Static `rag_enrichment`: seeded placeholders matching the future LLM contract
- Retrieval views: `citation_view`, `title_view`, `structure_view`, `semantic_concepts_en`, `original_terms_view`, `statute_anchor_view`, `law_context_view`, `raw_context`
- Quality/provenance: `static_formal_complete`, `needs_llm_for_complete_semantic_context`, `llm_priority`, `text_char_count`, `title_char_count`, input artifact paths

Why this helps downstream RAG:

- Exact citation and law-code fields support hard filters when a query names an article or statute.
- Title and section paths expose legal domain context that often is not repeated in the article text.
- Article/paragraph granularity and sibling/adjacent links support expansion from `Art. X Abs. 1` to related paragraphs without inventing parent citations.
- Outgoing and incoming anchors preserve deterministic graph links without treating graph co-citation as relevance.
- Separate retrieval views prevent one noisy blob from dominating search. BM25/vector channels can weight citation, title, English static concepts, original terms, references, and raw text differently.
- `needs_llm_for_complete_semantic_context` keeps the boundary honest: retrieval can use static fallback now, while later LLM fields can be merged without schema churn.

## 3. Static Extraction Run

Added script:

```text
scripts/build_law_authority_cards.py
```

Full run:

```text
python scripts\build_law_authority_cards.py --output artifacts\law_authority_cards_v1_static.jsonl --summary artifacts\law_authority_cards_v1_static.summary.json
```

Outputs:

- `artifacts/law_authority_cards_v1_static.jsonl`
- `artifacts/law_authority_cards_v1_static.summary.json`
- Smoke outputs: `artifacts/law_authority_cards_v1_static.smoke.jsonl`, `artifacts/law_authority_cards_v1_static.smoke.summary.json`

Validation:

- JSONL records written: 175,933
- Required-key scan: 175,933 records, 0 missing required schema groups
- Output size: about 700.7 MB
- Unicode repair checked on examples including `Art. 1 BöB` and `Art. 10 Org-VöB`.
- `python -m pytest ...` could not run because `pytest` is not installed.
- `python tests\test_segment_lattice_v3.py` passed: 3 tests OK.

## 4. LLM Law Enrichment Plan

Run law enrichment as the law-side equivalent of the 363k court descriptor run, but because the law corpus is only 175,933 rows and text is short for most rows, the best production target is all rows.

Recommended law LLM descriptor schema:

- `legal_area`
- `primary_domain`
- `secondary_domain`
- `legal_domain_path`
- `topic`
- `subtopic`
- `micro_topic`
- `concepts_en`
- `terms_original`
- `english_summary`
- `legal_question`
- `legal_rule`
- `legal_test`
- `applicability_conditions`
- `exceptions_or_limitations`
- `rights_obligations`
- `actors`
- `regulated_action`
- `fact_pattern_tags`
- `provision_role`
- `specificity_score`

Prompt rules:

- Use English for semantic fields.
- Preserve exact original-language legal terms in `terms_original`.
- Do not invent citations, statute anchors, or related provisions.
- Do not change deterministic fields such as citation, law code, title, article, units, outgoing references, or adjacent citations.
- If the provision is repeal, commencement, fee schedule, transitional, or administrative boilerplate, keep rule/test fields conservative.

Pipeline:

1. Build compact LLM input JSONL from `law_authority_cards_v1_static.jsonl` with `_source_row`, `citation`, `law_title`, `title_section_path`, structural fields, static labels, references, and clipped source text.
2. Run a vLLM notebook cloned from `court_llm_descriptor_extractor_blackwell_optimized__3__(1).ipynb`, using Qwen3-8B-AWQ or the best available local model.
3. Process by priority if compute is limited: high 21,922 first, medium 150,079 next, low 3,932 last. For final production, enrich all 175,933.
4. Validate JSON strictly; retry only invalid JSON rows.
5. Normalize with a new `law_enrichment_normalizer.py`: remove generated anchors, seed deterministic metadata, suppress overconfident rules for low-value provision roles, and build retrieval-safe views.
6. Merge static plus LLM into `artifacts/law_authority_cards_v2_unified.jsonl`.
7. Build unified law plus court retrieval documents and evaluate recall@1000 before reranking.

Quality checks:

- JSON validity and field fill rates
- Original-term grounding against source text
- No generated citation anchors
- No generated summaries/questions in deterministic-only fallback records
- Sample review by source type and legal area
- Retrieval ablation: static-only law cards vs LLM-enriched law cards vs unified law+court v5
