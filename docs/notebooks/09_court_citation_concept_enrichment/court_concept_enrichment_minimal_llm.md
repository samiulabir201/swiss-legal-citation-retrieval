# court_concept_enrichment_minimal_llm

**Path:** `e:\swiss_citation_extraction\notebooks\09_court_citation_concept_enrichment\court_concept_enrichment_minimal_llm.ipynb`

## Configuration

Defined in Cell 1 via a `Config` dataclass:

- Input data
  - `input_csv = "/kaggle/input/competitions/llm-agentic-legal-information-retrieval/court_considerations.csv"`
  - `fallback_input_csv = "court_considerations.csv"`
- Normalizer module discovery
  - `normalizer_search_paths = (".", "./scripts", "/kaggle/working", "/kaggle/working/scripts", "/kaggle/input/code", "/kaggle/input/scripts")`
- Model
  - `model_name = "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1"` (Qwen3-8B-AWQ)
- Run controls
  - `output_dir = "/kaggle/working"`, `start = 0`, `limit = 50`
  - `sample_random = False`, `random_seed = 42`
  - `min_text_chars = 120`, `max_text_chars = 3200`
- GPU mode
  - `gpu_mode = "single"` (sets `CUDA_VISIBLE_DEVICES=0`, `tensor_parallel_size = 1`); alternative `tp2` for tensor-parallel across 2 T4s
- vLLM settings
  - `gpu_memory_utilization = 0.78`, `max_model_len = 4096`, `max_num_seqs = 8`
  - `batch_size = 4`, `enforce_eager = True`, `quantization = "awq_marlin"`
  - `disable_custom_all_reduce = True`, `force_triton_attention = True` (sets `VLLM_ATTENTION_BACKEND=TRITON_ATTN`)
  - `use_structured_outputs = False` (kept off because Kaggle vLLM v0.20 hit `AttributeError("'dict' object has no attribute '_backend'")` with structured outputs)
- Generation
  - `max_new_tokens = 384`, `retry_max_new_tokens = 512`, `max_retries = 1`
  - `temperature = 0.0`, `top_p = 1.0`, `repetition_penalty = 1.02`, `enable_thinking = False`
- Env overrides
  - `TOKENIZERS_PARALLELISM=false`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256`
- Shard-safe output names (derived from `start`/`limit`)
  - `court_enriched_{start:07d}_{end:07d}.jsonl`
  - `court_enriched_{start:07d}_{end:07d}_preview.csv`
  - `court_enriched_{start:07d}_{end:07d}_failures.jsonl`
  - `court_enriched_{start:07d}_{end:07d}_metrics.json`

## Data

- Primary input: `court_considerations.csv` resolved via `resolve_input_path` searching:
  - `cfg.input_csv`
  - `cfg.fallback_input_csv`
  - `/kaggle/working/court_considerations.csv`
  - `/mnt/data/court_considerations.csv`, `/mnt/data/court_consideration.csv`
  - Recursive `/kaggle/input/**/court_consideration[s].csv`
- Column inference via `pick_column`:
  - `citation_col` from preferred names `["citation", "cite", "court_citation"]` or substring `["citation", "cite", "bge"]`
  - `text_col` from `["text", "consideration_text", "paragraph_text", "content", "raw_text"]` or substring `["text", "content", "paragraph", "consideration"]`
- Row selection:
  - Keep rows where citation and text are non-null
  - Filter by stripped text length `>= cfg.min_text_chars`
  - Slice `[start : start+limit]` (or random sample if `sample_random=True`)
- Deterministic metadata builder (`build_deterministic_metadata`) also reads optional columns when present: `court_base`, `legal_area`, `law_codes`, `statutes_cited`, `court_cases_cited`, `authority_role`, `issue_labels_en`, `matched_terms_multilingual`, `is_notification_paragraph`, `structural`.

## Pipeline

1. **Cell 1 — Imports and configuration.** Builds the `Config`, applies GPU env vars based on `gpu_mode`, sets vLLM/PyTorch env, computes shard-safe output paths, and prints CUDA/GPU info.
2. **Cell 2 — Import deterministic normalizer.** Searches `cfg.normalizer_search_paths` for `court_enrichment_normalizer.py` and imports `normalize_enriched_court_row`. Raises `FileNotFoundError` if missing.
3. **Cell 3 — Load CSV and select rows.** Resolves the input path, infers `citation_col`/`text_col`, filters by `min_text_chars`, slices or random-samples into `work_df`, and displays a preview of up to 20 rows.
4. **Cell 4 — Minimal LLM descriptor schema and prompt.** Defines:
   - `LLM_DESCRIPTOR_SCHEMA` (legal_area, primary_domain, secondary_domain, legal_domain_path, topic, subtopic, micro_topic, concepts_en, terms_original, doctrinal_rule, legal_test, fact_pattern_tags, procedural_context, paragraph_role, authority_role, specificity_score)
   - `FORBIDDEN_LLM_FIELDS` (statute_anchors, case_anchors, normalized_anchors, retrieval_views, query_phrases_en, natural_language_queries, legal_question, summary_en, english_summary, outcome_signal)
   - `SYSTEM_PROMPT` instructing JSON-only descriptor extraction, no anchors/summaries/questions/outcome
   - `USER_TEMPLATE` and `REPAIR_TEMPLATE` (passes previous bad output + error)
   - Helpers `trim_text`, `build_user_prompt`, `build_repair_prompt`
5. **Cell 5 — JSON parsing and minimal descriptor cleanup.** `find_balanced_json_object` strips Markdown fences and balances braces; `parse_json_lenient` tries strict `json.loads`, trailing-comma cleanup, smart-quote replacement, then `ast.literal_eval`. `as_clean_list` and `clean_descriptor` enforce length/dedup/forbidden-field stripping; `empty_descriptor` provides fallback values with `_descriptor_error`.
6. **Cell 6 — vLLM engine.** `VllmDescriptorEngine` loads the AWQ Qwen3-8B with the configured vLLM kwargs, optionally injects an explicit `AttentionConfig(backend='TRITON_ATTN')`, builds prompts via `apply_chat_template` (with `enable_thinking=False`), and exposes `generate(prompts, max_tokens)`. Asserts the model path exists.
7. **Cell 7 — Enrich rows.** For each batch of `batch_size` rows:
   - Build user prompts and batch-generate descriptor JSON
   - For each row: parse + clean via `generate_descriptor_for_row` (with up to `max_retries` repair attempts using `REPAIR_TEMPLATE`); on persistent failure return `empty_descriptor`
   - `build_final_record` calls `normalize_enriched_court_row(citation, text, llm_enrichment, deterministic_metadata)` and stamps `enrichment_quality` with `llm_descriptor_status`, `question_generation_used=False`, `summary_generation_used=False`, `llm_generated_anchors=False`
   - Writes final records to the JSONL, builds a `preview_df`, writes preview CSV, writes a failures JSONL (only if any), and writes `metrics.json` containing: `start`, `limit`, `processed`, `elapsed_seconds`, `rows_per_second`, `descriptor_status_counts`, `forbidden_rag_field_rows`, `self_anchor_rows`, `fallback_rows`, plus output paths.
8. **Cell 8 — QC checks.** Builds a `qc_df` with per-row checks (forbidden RAG fields, concept/term counts, statute/case anchor counts, self-citation in case anchors, rule suppression, anchor cleanup count, specificity score) and prints aggregate stats.
9. **Cell 9 — Inspect one final JSON record.** Pretty-prints the first 9000 characters of `records[0]`.

## Results

All code cells in the notebook show no output cells (empty). No execution artifacts, no logs, no displayed dataframes, no error tracebacks, and no metrics dictionary are present in the notebook as saved.

## Summary

This notebook is the canonical Kaggle runner for "minimal LLM" court citation enrichment, designed so that a deterministic Python normalizer (`court_enrichment_normalizer.py`) owns all anchors, outcome signals, and retrieval views, while a quantized Qwen3-8B-AWQ served via vLLM produces only non-static semantic descriptors (legal area/domain hierarchy, topics, concepts_en, original-language terms, optional doctrinal rule/test, fact tags, procedural context, paragraph/authority roles, specificity). It hard-strips a forbidden-field set from any LLM output and uses lenient JSON parsing with a single repair retry to keep failure modes tight. The pipeline reads `court_considerations.csv`, filters by min text length, slices by `start`/`limit` for shard-safe parallel runs, and emits four artifacts per shard (final JSONL, preview CSV, failures JSONL when any, and a metrics JSON). vLLM is configured for a single T4 with AWQ-Marlin, Triton attention, eager mode, and `use_structured_outputs=False` to dodge a known v0.20 backend bug; an alternate `tp2` mode exists for dual-GPU testing. The notebook as saved contains no executed outputs.
