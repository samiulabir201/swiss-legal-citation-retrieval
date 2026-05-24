# court_llm_descriptor_kaggle

**Path:** e:\swiss_citation_extraction\notebooks\08_court_llm_descriptor_extraction\court_llm_descriptor_kaggle.ipynb

## Configuration

Kaggle GPU notebook running only the expensive LLM step. The downstream local CPU scripts (`scripts/court_enrichment_profile.py`, `scripts/court_enrichment_normalizer.py`, `scripts/finalize_court_enrichment_from_llm.py`) handle anchor normalization and final retrieval views.

`Config` dataclass values:

- Input CSV: `/kaggle/input/competitions/llm-agentic-legal-information-retrieval/court_considerations.csv`
- Fallback input: `court_considerations.csv`
- Model: `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1` (Qwen3-8B-AWQ)
- Row selection: `start=0`, `limit=50`, `sample_random=False`, `random_seed=42`
- Text length filter: `min_text_chars=80`, `max_text_chars=3200`
- GPU/vLLM: `gpu_mode='single'`, `tensor_parallel_size=1`, `gpu_memory_utilization=0.78`, `max_model_len=4096`, `max_num_seqs=8`, `batch_size=4`, `enforce_eager=True`, `quantization='awq_marlin'`, `disable_custom_all_reduce=True`, `force_triton_attention=True`
- Structured outputs: disabled (`use_structured_outputs=False`) due to Kaggle vLLM v0.20 bug `AttributeError("'dict' object has no attribute '_backend'")`
- Generation: `max_new_tokens=384`, `retry_max_new_tokens=512`, `max_retries=1`, `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.02`, `enable_thinking=False`
- Output dir: `/kaggle/working`; `include_raw_output_on_success=False`
- Env vars: `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, `TOKENIZERS_PARALLELISM=false`, `CUDA_VISIBLE_DEVICES=0` (single mode)

Output filenames pattern: `court_llm_descriptors_{start:07d}_{end:07d}{.jsonl,_preview.csv,_failures.jsonl,_metrics.json}`.

## Data

- Source CSV resolved via priority list: `cfg.input_csv` -> `cfg.fallback_input_csv` -> `/kaggle/working/court_considerations.csv` -> glob search `**/court_considerations.csv` or `**/court_consideration.csv` under `/kaggle/input`.
- Columns auto-detected: `citation` (or first column) and `text` (or second column).
- Filtering: drop rows with null `citation`/`text`, cast text to str, compute `_text_len` (stripped length), keep rows with `_text_len >= 80`.
- Row selection: sequential slice `[start:start+limit]` (or random sample if `sample_random=True`). Reset index, keep original index as `_source_row`.

## Pipeline

1. **Imports (Cell 1):** stdlib (`pathlib`, `dataclasses`, `typing`, `collections.Counter`, `os`, `re`, `gc`, `ast`, `json`, `time`, `traceback`), `pandas`, `tqdm.auto`, optional `torch`. Prints CUDA device info if available.
2. **Config (Cell 2):** instantiates `Config`, applies GPU mode env vars, validates model path exists, prepares output filenames.
3. **Load CSV (Cell 3):** resolve input path, filter valid rows, slice, display first 20 rows.
4. **Schema and prompt (Cell 4):** defines 16 `DESCRIPTOR_KEYS` (`legal_area`, `primary_domain`, `secondary_domain`, `legal_domain_path`, `topic`, `subtopic`, `micro_topic`, `concepts_en`, `terms_original`, `doctrinal_rule`, `legal_test`, `fact_pattern_tags`, `procedural_context`, `paragraph_role`, `authority_role`, `specificity_score`), allowed `ROLE_VALUES` set (`holding`, `reasoning`, `facts`, `procedural_history`, `legal_standard`, `application`, `citation`, `costs`, `notification`, `disposition`, `neutral`), `LLM_SCHEMA_HINT` dict, system prompt forbidding anchors/questions/summaries/outcomes/retrieval views, `USER_TEMPLATE` with `trim_text()` helper (head+tail truncation with `... [TRUNCATED] ...` marker).
5. **JSON parsing + normalization (Cell 5):** `extract_json_object()` strips markdown fences and `<think>` tags, finds balanced `{...}` block, falls back through `json.loads`, comma-cleaning, then `ast.literal_eval`. `clean_str()`/`clean_list()` trim/dedup. `normalize_descriptor()` coerces every key, validates `paragraph_role` against `ROLE_VALUES` (default `'neutral'`), clamps `specificity_score` to `[0.0, 1.0]`, deletes any forbidden keys. `empty_descriptor(error)` returned on failure with `_descriptor_error`.
6. **Load vLLM (Cell 6):** load tokenizer and `vllm.LLM` with kwargs from config. Optionally wraps `AttentionConfig(backend='TRITON_ATTN')` (with fallback to lowercase), tolerates `ImportError`.
7. **Generation helpers (Cell 7):** `render_prompt()` builds chat messages via `tokenizer.apply_chat_template` (with `enable_thinking` argument fallback). `generate_raw()` calls `llm.generate` with `SamplingParams`. `generate_descriptor()` runs up to `max_retries+1` attempts, sending repair prompts that include parser error and previous bad output.
8. **Run extraction (Cell 8):** batched generation (`batch_size=4`), per-row parse+normalize, collects `records` and `failures`, writes outputs: `output_jsonl` (always), `output_failures_jsonl` (only if failures), `output_preview_csv`, `output_metrics_json`. Metrics include `selected_rows`, `written_rows`, `failures`, `elapsed_seconds`, `rows_per_second`, `status_counts`, output paths.
9. **QC (Cell 9):** `find_forbidden()` recurses through records looking for `FORBIDDEN` keys; produces `qc_df` with `forbidden_fields`, `concept_count`, `terms_original_count`, `has_topic`, `specificity_score`; prints forbidden field rows count and failed rows count.

## Results

The notebook contains no execution outputs (all 9 code cells have 0 stored outputs). No metrics, no preview tables, no generated JSONL, no QC dataframe, no Kaggle GPU prints are persisted in the file. No errors are recorded either.

## Summary

This notebook is the Kaggle-side GPU stage of the court-considerations enrichment pipeline: it loads Qwen3-8B-AWQ via vLLM and extracts a strict 16-field semantic descriptor schema from `court_considerations.csv`, writing raw JSONL plus a preview CSV, a failures JSONL, and a metrics JSON to `/kaggle/working`. The descriptor is intentionally minimal and query-neutral: no anchors, outcomes, summaries, natural-language questions, or retrieval views are produced here, with forbidden keys actively stripped during normalization and verified by a final QC cell. Robustness measures include `awq_marlin` quantization, forced Triton attention, disabled custom all-reduce, disabled structured outputs (Kaggle vLLM v0.20 bug), repair-prompt retries, balanced-brace JSON extraction with markdown/`<think>` stripping, and trailing-comma/`ast.literal_eval` fallbacks. Downstream deterministic CPU scripts (`court_enrichment_profile.py`, `court_enrichment_normalizer.py`, `finalize_court_enrichment_from_llm.py`) consume the JSONL to build `rag_enrichment`, `normalized_anchors`, `anchor_quality_flags`, `retrieval_views`, and `enrichment_quality`. The notebook as stored has not been executed (no cells contain outputs).
