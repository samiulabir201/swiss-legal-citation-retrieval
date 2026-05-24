# court_concept_enrichment_robust_v5_t4

**Path:** e:\swiss_citation_extraction\notebooks\09_court_citation_concept_enrichment\court_concept_enrichment_robust_v5_t4.ipynb

## Configuration

- Engine: vLLM (fallback `transformers`); GPU mode `single` (CUDA_VISIBLE_DEVICES=0, tensor_parallel_size=1); alternative `tp2` shards one model across both T4s.
- Model: `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1` (Qwen3-8B-AWQ).
- vLLM settings: `gpu_memory_utilization=0.78`, `max_model_len=4096`, `enforce_eager=True`, `max_num_seqs=8`, `quantization="awq_marlin"`, `disable_custom_all_reduce=True`, `force_triton_attention=True` (sets `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, also tries explicit `AttentionConfig(backend="TRITON_ATTN")`).
- Structured outputs: disabled by default (`use_structured_outputs=False`) because Kaggle vLLM v0.20 raised `AttributeError("'dict' object has no attribute '_backend'")`; engine auto-detects support for `guided_json`, `guided_decoding`, or `structured_outputs` if enabled.
- Generation: `batch_size=4`, `max_new_tokens=768`, `retry_max_new_tokens=1024`, `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.02`, `enable_thinking=False`, `max_retries=2`.
- Selection: `n_rows=10`, `start=0`, `limit=10`, `sample_random=False`, `random_seed=42`, `min_text_chars=250`, `max_text_chars=4500`; deterministic sharding via `start`/`limit`, with fallback to all rows >30 chars if not enough substantive rows.
- Output dir `/kaggle/working`; files `enriched_court_citations_10.jsonl`, `enriched_court_citations_10_preview.csv`, `enriched_court_citations_10_failures.jsonl`.

## Data

- Input CSV: `/kaggle/input/competitions/llm-agentic-legal-information-retrieval/court_considerations.csv` (with fallbacks `court_considerations.csv`, `/kaggle/working/court_considerations.csv`, `/mnt/data/court_considerations.csv`, `/mnt/data/court_consideration.csv`, and a `/kaggle/input/**/court_considerations.csv` glob).
- Columns inferred by `pick_column`: citation column from preferred `[citation, cite, court_citation, authority_citation, consideration_citation]` or token match `[citation, cite, bge]`; text column from preferred `[text, consideration_text, paragraph_text, content, raw_text, body]` or token match `[text, content, paragraph, consideration, body]`.
- Rows with NaN citation/text are dropped; text trimmed by length; substantive subset filtered by `min_text_chars=250` then sliced `iloc[start:start+limit]`.
- Manual mode: if `cfg.manual_records` is non-empty the CSV is bypassed and a DataFrame is built directly from `{citation, text}` dicts (defaults to empty list, so CSV is used).

## Pipeline

1. Cell 1 – Imports (`pandas`, `tqdm`, `torch`); prints CUDA visibility and per-GPU memory.
2. Cell 2 – `Config` dataclass; applies `gpu_mode`; aligns `n_rows` with `limit`; verifies model path exists.
3. Cell 3 – `resolve_input_path`, `pick_column`, `build_work_df_from_csv` / `build_work_df_from_manual` produce `work_df`, `citation_col`, `text_col`.
4. Cell 4 – Defines enums `ROLE_VALUES`, `OUTCOME_VALUES`, `AUTHORITY_VALUES`, generic-concept blacklist, and `ENRICHMENT_JSON_SCHEMA` (17 required fields: legal_area, legal_domain_path, topic, subtopic, micro_topic, concepts_en, terms_original, statute_anchors, case_anchors, doctrinal_rule, legal_test, fact_pattern_tags, procedural_context, paragraph_role, authority_role, outcome_signal, specificity_score). System prompt forbids user-question/summary generation; `USER_TEMPLATE` and `trim_text` (head+tail truncation with `[TRUNCATED]` marker) build prompts. `build_prompt` supports a repair variant that feeds the bad output and parser error back in.
5. Cell 5 – Robust parsing/normalization: `find_balanced_json_object` (handles code fences, brace depth, strings), `parse_json_lenient` (strict JSON → trailing-comma + smart-quote cleanup → `ast.literal_eval`), `dedupe_keep_order`, `clamp_string`, `normalize_enum` with alias map (e.g. `remanded→remitted`, `partially_granted→partial`, `analysis→reasoning`, `application→application_of_rule`). `extract_statutes_from_text` (regex for `Art. N Abs. N lit. x <Code>`) and `extract_case_citations_from_text` (BGE + `Urteil X_Y/YYYY`) deterministically supplement model anchors. `remove_self_case_anchor` strips the input citation and its `citation_base` from anchors. `normalize_enrichment` clamps and fills defaults, drops generic concepts, falls back to topic fields if too sparse, and removes forbidden fields. `validate_enrichment` checks required fields, enum membership, non-empty concepts/topic, and absence of forbidden keys. `deterministic_fallback` builds a fingerprint from regex extractions only. `build_retrieval_views` produces `semantic_concepts_en`, `topic_path`, `original_terms_view`, `statute_anchor_view`, `case_anchor_view`, `legal_rule_view`, `fact_pattern_view`, and a 1200-char `raw_context`.
6. Cell 6 – `VllmJsonEngine` (with `_build_attention_config` and `_detect_structured_mode`) and `TransformersJsonEngine` (HF generate with `do_sample=False`) both apply chat template (Qwen-style, `enable_thinking=False`) and expose `generate_prompts`. `load_engine` selects per `cfg.engine`.
7. Cell 7 – Main loop: mini-batches of `batch_size=4`; for each row `try_enrich_single` parses, normalizes, validates; on failure issues a repair prompt up to `max_retries` (structured then unstructured), else emits `deterministic_fallback` with `_fallback_reason`. `build_output_record` wraps each row with `court_base`, `source_family="court"`, `rag_enrichment`, `retrieval_views`, and an `enrichment_quality` block (`method`, `generation_status`, flags, `attempt_count`). Outputs are written to JSONL, a flattened preview CSV, and a failures JSONL (cleaned up if empty). Prints elapsed time, rows/s, `Counter` of statuses, and displays the preview DataFrame.
8. Cell 8 – Pretty-prints the first record (first 8000 chars) for inspection.
9. Cell 9 – QC pass: `check_no_forbidden_fields` walks each record; reports forbidden fields, concept count, original-term count, self-case-anchor presence, specificity score, status; prints aggregate counts.

## Results

The notebook is not executed. All code cells have empty output streams, so no generated row counts, timings, status `Counter`, preview DataFrame, full-record dump, or QC frame are available in the notebook artifact.

## Summary

This is a smoke-test enrichment notebook (10 rows) that turns Swiss court consideration rows into a compact, query-neutral legal fingerprint using Qwen3-8B-AWQ on vLLM, targeted at Kaggle dual-T4 sessions. The design deliberately avoids generated user questions and summaries, instead emitting a 17-field schema (topic hierarchy, English concepts, original-language terms, statute/case anchors, doctrinal rule, legal test, fact-pattern tags, procedural context, paragraph/authority roles, outcome signal, specificity score). Robustness comes from prompt-constrained JSON plus a brace-balanced lenient parser, alias-driven enum normalization, regex-based deterministic statute/case anchor extraction, a one-by-one repair retry loop, and a clearly-marked deterministic fallback record so a row is always produced. Structured decoding (`guided_json`/`guided_decoding`/`structured_outputs`) is auto-detected but disabled by default due to a reported `_backend` AttributeError on Kaggle vLLM v0.20. Outputs include a JSONL, a flattened preview CSV, a failures JSONL, and an in-notebook QC frame; this saved notebook contains no execution outputs.
