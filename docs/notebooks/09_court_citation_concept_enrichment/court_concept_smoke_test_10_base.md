# court_concept_smoke_test_10_base

**Path:** e:\swiss_citation_extraction\notebooks\09_court_citation_concept_enrichment\court_concept_smoke_test_10_base.ipynb

## Configuration

- Model: `Qwen/Qwen3-8B-AWQ` (vLLM, AWQ-quantized).
- vLLM init: `tensor_parallel_size=1`, `gpu_memory_utilization=0.70`, `max_model_len=4096`, `enforce_eager=True`, `trust_remote_code=True`, `disable_custom_all_reduce=True`.
- Attention backend: `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (Kaggle-safe).
- Generation: `temperature=0.0`, `top_p=1.0`, `max_new_tokens=768`, `batch_size=10`.
- Qwen3 thinking mode: `enable_thinking=False`.
- Input budget: `text_chars=5500` per citation; truncated with `...[TRUNCATED]` marker.
- Row selection: `n_rows=10`, `sample_random=False`, `random_seed=None` (first 10 valid rows; valid = both citation/text non-null and stripped text > 50 chars).
- Input CSV resolution order: `/kaggle/input/your-dataset/court_consideration.csv` → `court_consideration.csv` (cwd) → `/kaggle/working/court_consideration.csv` → `/mnt/data/court_consideration.csv` → glob `/kaggle/input/**/court_consideration.csv`.
- Output dir: `/kaggle/working/`. Artifacts: `enriched_court_citations_10.jsonl`, `enriched_court_citations_10_preview.csv`.
- Guided decoding: attempts `vllm.sampling_params.GuidedDecodingParams(json=ENRICHMENT_SCHEMA)`; falls back to prompt-only JSON control if unsupported.
- Env: `TOKENIZERS_PARALLELISM=false`.

## Data

- Source file: `court_consideration.csv` (path resolved at runtime; not bundled).
- Column auto-detection via `pick_column`:
  - citation column preferred names: `citation`, `cite`, `court_citation`, `authority_citation`, `consideration_citation`; substring match on `citation`, `cite`, `bge`.
  - text column preferred names: `text`, `consideration_text`, `paragraph_text`, `content`, `raw_text`, `body`; substring match on `text`, `content`, `paragraph`, `consideration`, `body`.
- Filtering: drop rows where citation or text is null; cast text to string; keep only rows with `text.strip().length > 50`.
- Selection: first 10 valid rows (`valid_10`), reset index, original row index preserved as `_source_row`.

## Pipeline

1. **Config + env setup**: instantiate `CONFIG` dataclass; set vLLM/tokenizer env vars; create output dir.
2. **Input resolution**: `resolve_input_path` walks candidate paths and globs Kaggle inputs; load CSV with pandas.
3. **Column inference**: `pick_column` identifies citation and text columns; raises if either is missing.
4. **Row filtering and sampling**: drop nulls, filter short texts, take first 10 (or random sample if `sample_random=True`).
5. **Schema + prompt definition**:
   - `ENRICHMENT_SCHEMA` (JSON Schema): required fields are `legal_area`, `legal_domain_path` (2-6), `topic`, `subtopic`, `micro_topic`, `concepts_en` (4-14), `terms_original` (4-18), `statute_anchors` (≤10), `case_anchors` (≤10), `doctrinal_rule`, `legal_test`, `fact_pattern_tags` (≤10), `procedural_context`, `paragraph_role` (enum: holding/reasoning/facts/procedural_history/citation/dissent/neutral), `authority_role` (array of enum incl. leading_decision, legal_test, constitutional_standard, statutory_interpretation, standard_of_review, application_of_rule, distinguishing_case, background, procedural_rule, evidentiary_standard, official_intervention_limit, voting_rights_jurisprudence, neutral; ≤5), `outcome_signal` (enum: granted/dismissed/remanded/partially_granted/inadmissible/neutral), `specificity_score` (0-1). `additionalProperties=False`.
   - `SYSTEM_PROMPT`: Swiss legal indexing assistant; no synthetic questions, no fact invention, grounded descriptors only, English for taxonomic fields, original-language for `terms_original`, JSON only.
   - `make_user_prompt(citation, text)`: truncates text to `text_chars` and frames the request.
6. **Output parsing utilities**:
   - `extract_json_object`: strips markdown fences, removes `<think>...</think>`, parses JSON or falls back to first `{`...last `}` slice.
   - `normalize_list`: dedupes case-insensitively, caps to `max_items`.
   - `normalize_enrichment`: coerces string fields, normalizes list fields, validates enums (falls back to `neutral`), clamps `specificity_score` to [0,1].
   - `build_retrieval_views`: produces seven view strings - `semantic_concepts_en`, `topic_path` (joined with " > "), `original_terms_view`, `statute_anchor_view`, `case_anchor_view`, `legal_rule_view`, `fact_pattern_view`, plus `raw_context` truncated to `text_chars`.
   - `validate_enrichment`: emits soft-error tags like `missing:<field>`, `weak:concepts_en`, `weak:terms_original`, `weak:topic`, `weak:rule_or_test`, `too_generic:concepts_en` (against blacklist {law, court, case, decision, legal, rights, appeal}).
7. **vLLM load**: `LLM(**llm_kwargs)`; tries to import `GuidedDecodingParams`.
8. **Prompt rendering**: `build_messages` (system + user); `render_prompt` calls `tokenizer.apply_chat_template` with `enable_thinking=False`, with TypeError fallback for older tokenizers.
9. **Batch generation**: `llm.generate(prompts, sampling_params)` over the 10 prompts.
10. **Per-row record assembly**: parse → normalize → validate → build retrieval views → wrap in record with `citation`, `source_row`, `source_family="court"`, `rag_enrichment`, `retrieval_views`, `enrichment_quality` (method, flags, validation_errors, low_value_paragraph), `_raw_model_output`. On parse failure: record with `rag_enrichment=None`, `json_valid=False`, `low_value_paragraph=True`, and a minimal retrieval view containing only `raw_context` and `case_anchor_view`.
11. **Persist**: write JSONL (one record per line, ensure_ascii=False) and a flattened preview CSV with pipe-joined list fields; display preview DataFrame.
12. **Qualitative inspection**: re-display a subset of columns (`citation`, `topic`, `subtopic`, `micro_topic`, `concepts_en`, `terms_original`, `doctrinal_rule`, `validation_errors`) with `max_colwidth=300`.

## Results

No output cells are present in the notebook (all cells are unexecuted).

## Summary

This is a Kaggle-targeted smoke test that enriches 10 Swiss court-citation rows with query-neutral legal descriptors using Qwen3-8B-AWQ on vLLM. It defines a strict JSON Schema covering taxonomic fields (legal_area through micro_topic), grounded concept/term/anchor lists, doctrinal rule/test, procedural and authority roles, an outcome signal, and a specificity score, and forces JSON-only output via guided decoding when available. Auxiliary utilities normalize and lightly validate the parsed JSON, then derive seven retrieval-view strings (semantic, topic-path, original-terms, statute-anchor, case-anchor, legal-rule, fact-pattern) plus a truncated raw context. Outputs are written to JSONL and a flattened preview CSV under `/kaggle/working/`. The notebook ships unexecuted, so no measurements, parse-failure counts, or sample enrichments are recorded.
