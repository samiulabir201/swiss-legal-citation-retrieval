# court_concept_smoke_test_10_v3_local

**Path:** e:\swiss_citation_extraction\notebooks\09_court_citation_concept_enrichment\court_concept_smoke_test_10_v3_local.ipynb

## Configuration

- Input CSV: `/kaggle/input/competitions/llm-agentic-legal-information-retrieval/court_considerations.csv` (fallback `court_considerations.csv`).
- Output directory: `/kaggle/working`.
  - JSONL: `enriched_court_citations_10.jsonl`
  - Preview CSV: `enriched_court_citations_10_preview.csv`
  - Failures JSONL: `enriched_court_citations_10_failures.jsonl`
- Row selection: `n_rows=10`, `min_text_chars=600`, `sample_random=False`, `random_seed=42`. Substantive rows only (avoids tiny fragments like "BGE ... S. 7").
- Input text budget: `text_chars=3200` per citation (trimmed at last whitespace, " ..." appended).
- Model: local Kaggle path `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1` (Qwen3-8B AWQ).
- vLLM engine: `tensor_parallel_size=1`, `gpu_memory_utilization=0.70`, `max_model_len=4096`, `max_num_seqs=16`, `enforce_eager=True`, `quantization=awq_marlin`, `disable_custom_all_reduce=True`, `trust_remote_code=True`. Explicit `AttentionConfig(backend="TRITON_ATTN")` attempted, with env-var fallback `VLLM_ATTENTION_BACKEND=TRITON_ATTN`.
- Generation: `batch_size=5`, `max_new_tokens=512`, `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.02`, `enable_thinking=False`.
- Structured output: prefers `StructuredOutputsParams(json=ENRICHMENT_SCHEMA)`, falls back to `GuidedDecodingParams`, finally to prompt-only JSON control.
- Env: `TOKENIZERS_PARALLELISM=false`.

## Data

- Source dataframe: read via `pd.read_csv(input_path)` after `resolve_input_path` probes Kaggle/working/mnt locations and globs `/kaggle/input/**/court_considerations.csv` (or `court_consideration.csv`).
- Column inference via `pick_column`:
  - `citation_col`: preferred names `citation, cite, court_citation, authority_citation, consideration_citation`; substring fallback `citation, cite, bge`.
  - `text_col`: preferred names `text, consideration_text, paragraph_text, content, raw_text, body`; substring fallback `text, content, paragraph, consideration, body`.
- Filtering: rows with non-null citation and text; text cast to `str`; `_text_len = str.strip().str.len()`; keep rows where `_text_len >= 600`. Raises if fewer than 10 substantive rows remain.
- Working set: first 10 substantive rows (or random sample with seed 42 when `sample_random=True`), with original index preserved as `_source_row`.

## Pipeline

1. **Config + env setup** — instantiate `Config` dataclass; set tokenizer parallelism off; optionally set Triton attention backend env var; create output dir.
2. **Input path resolution** — `resolve_input_path` walks candidate paths and globs `/kaggle/input`.
3. **CSV load + column inference + row filtering** — selects substantive rows above `min_text_chars` threshold.
4. **Schema + prompt construction** — `ENRICHMENT_SCHEMA` defines a JSON Schema (object, `additionalProperties=False`) with required keys: `legal_area`, `legal_domain_path` (array), `topic`, `subtopic`, `micro_topic`, `concepts_en` (array, max 12), `terms_original` (array, max 16), `statute_anchors`, `case_anchors`, `doctrinal_rule`, `legal_test`, `fact_pattern_tags`, `procedural_context`, `paragraph_role` (enum: holding/reasoning/facts/procedural_history/citation/dissent/neutral), `authority_role` (array of enums incl. `leading_decision`, `legal_test`, `constitutional_standard`, `statutory_interpretation`, `standard_of_review`, `application_of_rule`, `distinguishing_case`, `background`, `procedural_rule`, `evidentiary_standard`, `official_intervention_limit`, `voting_rights_jurisprudence`, `neutral`), `outcome_signal` (enum: granted/dismissed/remanded/partially_granted/inadmissible/neutral), `specificity_score` (0..1). System prompt forbids question generation, summaries, and invented references.
5. **JSON parser + normalizer** — `extract_json_object` strips Markdown fences and locates a balanced top-level `{...}`; `normalize_enrichment` fills missing keys, coerces array/string fields, clamps enums to allowed sets (`paragraph_role`→`neutral`, `outcome_signal`→`neutral`, invalid `authority_role` entries dropped, default `["neutral"]`), and clamps `specificity_score` to `[0,1]`.
6. **Retrieval-view builder** — `build_retrieval_views` produces `semantic_concepts_en`, `topic_path`, `original_terms_view`, `statute_anchor_view`, `case_anchor_view`, `legal_rule_view`, `raw_context` (truncated to `text_chars`).
7. **vLLM model load** — verifies local model path exists, instantiates `LLM(**build_llm_kwargs())`, fetches tokenizer.
8. **Sampling params** — `build_sampling_params` tries `StructuredOutputsParams(json=schema)`, then `GuidedDecodingParams(json=schema)`, otherwise plain `SamplingParams`.
9. **Batch generation loop** — iterate `work_df` in chunks of 5; build chat messages (system + user with citation + truncated text + schema text); render via `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`; call `llm.generate(prompts, sampling_params)`; for each output, parse JSON, normalize, build retrieval views, attach `enrichment_quality` metadata (`method=llm_legal_descriptor_enrichment`, `question_generation_used=False`, `summary_generation_used=False`, `grounded_references_only=True`, `low_value_paragraph=False`); on parse failure, append to `failures` with raw output.
10. **Persist outputs** — write `records` to JSONL, `failures` to JSONL (if any), build a flattened pandas `preview_df` (pipe-joined arrays) and write CSV; print first successful record (truncated to 5000 chars) or first failure for inspection.
11. **Markdown checklist** — final cell lists qualitative inspection criteria (good signs: narrow `micro_topic`, meaningful `concepts_en`, preserved original terms, short grounded rule/test, specific fact tags; bad signs: generic labels, invented references, synthetic questions, summary prose).

## Results

No executed cell outputs are present in the notebook (no `cfg` dump, no shape/columns print, no model load logs, no `Generated N OK, M failed` line, no preview dataframe, no sample record dump). All code cells contain only source and no `outputs` payloads.

## Summary

This is a 10-row local smoke test for a query-neutral LLM-based enrichment pipeline over `court_considerations.csv`, designed to attach compact retrieval metadata (legal area, topic hierarchy, concepts, original-language terms, statute/case anchors, doctrinal rule, legal test, fact-pattern tags, paragraph and authority roles, outcome signal, specificity score) to Swiss court citation paragraphs. The v3-local variant runs a locally-mounted Kaggle Qwen3-8B AWQ checkpoint through vLLM with Triton attention and AWQ-Marlin quantization, using vLLM's structured-outputs / guided-decoding JSON-schema enforcement (with prompt-only fallback). Row selection is restricted to substantive paragraphs (`_text_len >= 600`) to avoid degenerate one-line BGE fragments, and the prompt explicitly bans synthetic questions, generic summaries, and invented references. Outputs comprise an enriched JSONL with `rag_enrichment` + `retrieval_views` + `enrichment_quality`, a flattened preview CSV, and a separate failures JSONL for unparsable model responses. The notebook is unexecuted, so no empirical timing, parse-failure counts, or example enrichments are recorded.
