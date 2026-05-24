# court_concept_enrichment_robust_v4_predecessor

**Path:** e:\swiss_citation_extraction\notebooks\09_court_citation_concept_enrichment\court_concept_enrichment_robust_v4_predecessor.ipynb

## Configuration

Defined in Cell 2 via a `Config` dataclass:

- Input: `input_csv=/kaggle/input/competitions/llm-agentic-legal-information-retrieval/court_considerations.csv` with `fallback_input_csv=court_considerations.csv`.
- Model: `model_name=/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1` (Qwen3-8B AWQ).
- Output: `output_dir=/kaggle/working`, `output_jsonl=enriched_court_citations_10.jsonl`, `output_preview_csv=enriched_court_citations_10_preview.csv`, `output_failures_jsonl=enriched_court_citations_10_failures.jsonl`.
- Input selection: `n_rows=10`, `sample_random=False`, `random_seed=42`, `min_text_chars=250`, `max_text_chars=4500`, `manual_records=[]` (CSV path is used when empty).
- Engine: `engine=vllm`, `tensor_parallel_size=1`, `gpu_memory_utilization=0.72`, `max_model_len=4096`, `enforce_eager=True`, `max_num_seqs=8`, `quantization=awq`, `disable_custom_all_reduce=True`, `force_triton_attention=True` (sets `VLLM_ATTENTION_BACKEND=TRITON_ATTN`).
- Generation: `batch_size=4`, `max_new_tokens=768`, `retry_max_new_tokens=1024`, `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.02`, `enable_thinking=False`, `max_retries=2`.

## Data

- Source: `court_considerations.csv` resolved by searching a list of candidate paths and `/kaggle/input/**/court_considerations.csv` (or `court_consideration.csv`).
- Column inference via `pick_column`: a citation column (`citation`/`cite`/`court_citation`/`authority_citation`/`consideration_citation`, or any column containing `citation`/`cite`/`bge`) and a text column (`text`/`consideration_text`/`paragraph_text`/`content`/`raw_text`/`body`, or any column containing `text`/`content`/`paragraph`/`consideration`/`body`).
- Row filtering: drop NaN in citation/text, compute `_text_len`, prefer rows with `_text_len >= min_text_chars=250`; if fewer than `n_rows` qualify, fall back to all rows with `_text_len > 30`.
- When `sample_random=False`, take the first `n_rows` substantive rows (head). When True, random sample with `random_state=random_seed`.
- Alternate input: if `cfg.manual_records` is non-empty, the CSV is ignored and a DataFrame is built from those dicts (`citation`, `text`).

## Pipeline

- Cell 1 - Imports `os`, `re`, `gc`, `ast`, `json`, `time`, `traceback`, `dataclasses`, `pathlib`, `typing`, `collections.Counter`, `pandas`, `tqdm.auto`; conditionally imports `torch` and prints CUDA/GPU info.
- Cell 2 - Defines `Config` dataclass; instantiates `cfg = Config(manual_records=[])`; asserts the local model path exists.
- Cell 3 - Resolves input CSV via `resolve_input_path`, picks citation/text columns via `pick_column`, builds `work_df` via `build_work_df_from_csv` (or `build_work_df_from_manual` when manual mode is active).
- Cell 4 - Defines enrichment schema and prompt:
  - Enum sets `ROLE_VALUES` (holding/reasoning/facts/procedural_history/citation/dissent/neutral), `OUTCOME_VALUES` (granted/dismissed/inadmissible/remitted/partial/neutral/none), `AUTHORITY_VALUES` (leading_decision/settled_rule/legal_test/standard_of_review/constitutional_standard/statutory_interpretation/application_of_rule/distinguishing_case/procedural_background/factual_background/background/none).
  - `GENERIC_CONCEPTS` blocklist of low-signal words.
  - `ENRICHMENT_JSON_SCHEMA` with 17 required fields: `legal_area`, `legal_domain_path` (2-6 items), `topic`, `subtopic`, `micro_topic`, `concepts_en` (3-8), `terms_original` (<=10), `statute_anchors` (<=8), `case_anchors` (<=8), `doctrinal_rule`, `legal_test`, `fact_pattern_tags` (<=6), `procedural_context`, `paragraph_role` (enum), `authority_role` (array enum, 1-4), `outcome_signal` (enum), `specificity_score` (0..1).
  - `SYSTEM_PROMPT` instructs the model to act as a "deterministic Swiss legal citation enrichment engine", emitting a query-neutral legal fingerprint (English classification, original-language terms, only-grounded anchors, no synthetic queries, no summary).
  - `USER_TEMPLATE` injects citation, trimmed text, and the JSON schema.
  - `trim_text` reduces long inputs by keeping head and tail joined with `... [TRUNCATED] ...`.
  - `build_prompt` produces either a fresh prompt or a repair prompt that includes the previous bad output and parser error.
- Cell 5 - Parsing, normalization, validation, fallback, and retrieval views:
  - `find_balanced_json_object` strips code fences and extracts the first balanced JSON object.
  - `parse_json_lenient` tries `json.loads`, then trailing-comma + smart-quote cleanup, then `ast.literal_eval`.
  - Helpers: `dedupe_keep_order`, `clamp_string`, `normalize_enum` (with alias map), `normalize_authority_roles`.
  - `citation_base` extracts e.g. `BGE 145 I 1` from `BGE 145 I 1 E. 6.5.1`; `remove_self_case_anchor` drops self-references.
  - `extract_statutes_from_text` regex `Art\.\s*\d+...` for Swiss articles; `extract_case_citations_from_text` regex for `BGE` and `Urteil ...` patterns.
  - `normalize_enrichment` fills defaults, applies generic-concept filter, merges deterministic statute/case extractions with model output, clamps lengths, normalizes enums, clamps `specificity_score` to [0,1], and strips forbidden fields.
  - `validate_enrichment` checks required keys, enum validity, non-empty concepts/topic, and forbidden fields.
  - `deterministic_fallback` builds a "manual review required" record with extracted statutes/cases and capitalized term heuristics.
  - `build_retrieval_views` returns `semantic_concepts_en`, `topic_path`, `original_terms_view`, `statute_anchor_view`, `case_anchor_view`, `legal_rule_view`, `fact_pattern_view`, and `raw_context` (trimmed to 1200 chars).
- Cell 6 - Engines:
  - `VllmJsonEngine` loads tokenizer + `vllm.LLM`, optionally sets an explicit `AttentionConfig(backend=TRITON_ATTN)` if available, and auto-detects structured-decoding mode (`structured_outputs` -> `guided_decoding` -> `guided_json` -> `none`). `_sampling_params` injects the JSON schema accordingly. `_chat_to_prompt` applies the chat template with `enable_thinking=False` (TypeError fallback for older tokenizers). `generate_prompts` runs `llm.generate` with `use_tqdm=False`.
  - `TransformersJsonEngine` fallback using `AutoModelForCausalLM` with `do_sample=False`, manual loop over prompts.
  - `load_engine` dispatches; the cell finally calls `engine = load_engine(cfg)`.
- Cell 7 - Runner:
  - `make_row_dict`, `enrich_one_from_raw` (parse + normalize + validate), `try_enrich_single` (loops up to `max_retries+1`; on failure prepares a repair prompt and retries first in structured then unstructured mode; final failure returns `deterministic_fallback`).
  - `build_output_record` packages `_source_row`, `citation`, `court_base`, `source_family=court`, `text`, `rag_enrichment`, `retrieval_views`, and `enrichment_quality` (method tag, status, flags `has_specific_topic`/`has_original_language_terms`/`has_statute_anchor`/`has_case_anchor`/`low_value_paragraph`, `attempt_count`); attaches `_debug_attempts` on fallback.
  - Main loop iterates `work_df` in `batch_size=4` chunks via `tqdm`, calls `engine.generate_prompts`; on batch exception falls back to per-row generation. Writes JSONL, flattens to a preview CSV with key fields, writes a separate failures JSONL (or deletes stale one), prints elapsed time, rows/s, and a Counter of generation statuses, then `display(preview_df)`.
- Cell 8 - Prints `json.dumps(records[0], ...)` truncated to 8000 chars.
- Cell 9 - Quality checks: `check_no_forbidden_fields` recursively walks each record for `query_phrases_en`/`summary_en`/`natural_language_queries`/`english_summary`; builds `qc_df` with citation, status, forbidden-field hits, concept count, original-term count, self-case-anchor flag, specificity score; prints counts of forbidden-field rows, self-case-anchor rows, and fallback rows.

## Results

The notebook contains no executed output cells (all cells are unexecuted source code only).

## Summary

This is a self-contained, single-pass smoke-test notebook for enriching Swiss court citation rows (from `court_considerations.csv`) into a compact, query-neutral legal-descriptor JSON via a Qwen3-8B-AWQ model served by vLLM with schema-guided decoding. The pipeline emphasises grounded outputs: regex-based statute/case anchor extraction is merged with model output, generic concepts are filtered, self-references are removed, enums are normalised against fixed allow-sets, and lengths are clamped. Robustness features include lenient JSON parsing (balanced-brace extraction, smart-quote/trailing-comma cleanup, `ast.literal_eval` fallback), up to two repair retries with the previous bad output included in the prompt, structured-then-unstructured retry sampling, and a deterministic fallback record so every input row always produces an output. Outputs are a full JSONL, a flattened preview CSV, an optional failures JSONL, and a QC table flagging forbidden fields, self case anchors, and fallback rows. As shipped, the notebook has no recorded execution; it is intended to run on Kaggle with the listed model dataset attached.
