# court_llm_descriptor_10k_optimized

**Path:** e:\swiss_citation_extraction\notebooks\08_court_llm_descriptor_extraction\court_llm_descriptor_10k_optimized.ipynb

## Configuration

- Runtime: Google Colab with a single NVIDIA RTX PRO 6000 Blackwell Server Edition GPU (~94.97 GiB VRAM); Drive mounted at `/content/drive`.
- Packages installed in Cell 0: `vllm>=0.8.5`, `transformers>=4.51.0`, `accelerate`, `safetensors`, `pandas`, `tqdm`, `huggingface_hub`.
- `BASE_DIR = /content/drive/MyDrive/swiss_law`; `DATA_DIR = .../data`; `MODEL_DIR = .../models`; `OUTPUT_DIR = .../outputs`.
- `Config` dataclass (Cell 2):
  - `model_name = 'Qwen/Qwen3-8B-AWQ'`
  - `model_download_dir = .../models/huggingface`
  - `input_csv = .../data/court_considerations.csv`; `fallback_input_csv = 'court_considerations.csv'`
  - Slice: `start=0`, `limit=10_000`, `sample_random=False`, `random_seed=42`, `min_text_chars=80`
  - Prompt budget: `max_text_chars=2400`
  - GPU/vLLM: `gpu_mode='single'`, `tensor_parallel_size=1`, `gpu_memory_utilization=0.82`, `max_model_len=3072`, `max_num_seqs=96`, `batch_size=64`, `enforce_eager=False`, `quantization='awq_marlin'`, `disable_custom_all_reduce=True`, `force_triton_attention=False`, `use_structured_outputs=False`
  - Generation: `max_new_tokens=320`, `retry_max_new_tokens=448`, `max_retries=1`, `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.02`, `enable_thinking=False`
  - Output: `include_raw_output_on_success=False`
- Env tweaks: `CUDA_VISIBLE_DEVICES=0` (single-GPU mode), `TOKENIZERS_PARALLELISM=false`.
- Output artifacts (built from `start`/`limit` suffix `0000000_0009999`):
  - `court_llm_descriptors_0000000_0009999.jsonl`
  - `court_llm_descriptors_0000000_0009999_preview.csv`
  - `court_llm_descriptors_0000000_0009999_failures.jsonl` (deleted when no failures)
  - `court_llm_descriptors_0000000_0009999_metrics.json`

## Data

- Input CSV: `/content/drive/MyDrive/swiss_law/data/court_considerations.csv`.
- Reported shape: `(2476315, 2)` with columns `['citation', 'text']`.
- Row selection logic in Cell 3:
  - Auto-detects `citation`/`text` columns (falls back to first two columns).
  - Drops rows with NaN in either column and rows where stripped `text` length < `min_text_chars` (80).
  - With `sample_random=False`, slices `valid.iloc[0:10000]`.
  - Adds `_source_row` (the original DataFrame index) before reset.
- Selected rows after filtering: 10,000. The first 20 rows displayed are German-language Swiss Federal Supreme Court considerations from `BGE 139 I 2` (e.g., E. 2, E. 5.1, E. 5.2, ...) with text lengths between 118 and 2,916 characters.

## Pipeline

1. **Cell 0 – Colab setup**: detects Colab, mounts Drive, installs `vllm`, `transformers`, `accelerate`, `safetensors`, `pandas`, `tqdm`, `huggingface_hub`.
2. **Cell 1 – Imports / runtime check**: imports `pandas`, `tqdm`, `torch`; resolves `BASE_DIR` (Drive on Colab, `..` locally); prints CUDA device info.
3. **Cell 2 – Config**: defines the `Config` dataclass, applies GPU mode by setting `CUDA_VISIBLE_DEVICES` and `tensor_parallel_size`, ensures output dirs exist, and builds output paths via `start`/`end_idx` suffix.
4. **Cell 3 – Load CSV / select rows**: `resolve_input_path` walks candidate locations and globs `**/court_considerations.csv`; loads via `pandas`; filters by non-null citation/text and `min_text_chars`; slices or random-samples; stores original index as `_source_row`.
5. **Cell 4 – Schema and prompt**: declares `DESCRIPTOR_KEYS` (16 keys: `legal_area`, `primary_domain`, `secondary_domain`, `legal_domain_path`, `topic`, `subtopic`, `micro_topic`, `concepts_en`, `terms_original`, `doctrinal_rule`, `legal_test`, `fact_pattern_tags`, `procedural_context`, `paragraph_role`, `authority_role`, `specificity_score`); allowed `ROLE_VALUES` set (11 roles incl. `holding`, `reasoning`, `facts`, `procedural_history`, `legal_standard`, `application`, `citation`, `costs`, `notification`, `disposition`, `neutral`); a compact `LLM_SCHEMA_HINT` dict, a `SYSTEM_PROMPT` instructing the model to return one compact JSON with only semantic descriptors (no statute anchors, case anchors, outcomes, retrieval views, questions, summaries, or quality flags) using English classification fields and exact source-language `terms_original`; a `USER_TEMPLATE`; and `trim_text` that collapses whitespace and head/tail-splits to `max_text_chars` with a `[TRUNCATED]` marker.
6. **Cell 5 – JSON parsing / normalization**: `extract_json_object` strips code fences and `<think>` tags, then scans for a balanced `{...}` (string-aware), parses via `json.loads`, retries after removing trailing commas, and falls back to `ast.literal_eval`. `clean_str` and `clean_list` cap field lengths and dedupe (casefold). `normalize_descriptor` coerces every key to the declared shape, clamps `specificity_score` into `[0,1]`, snaps `paragraph_role` into `ROLE_VALUES` (default `neutral`), and pops a `forbidden` set of fields (`statute_anchors`, `case_anchors`, `normalized_anchors`, `retrieval_views`, `outcome_signal`, `query_phrases_en`, `natural_language_queries`, `legal_question`, `summary_en`, `english_summary`, `enrichment_quality`, `anchor_quality_flags`). `empty_descriptor` produces a fallback record carrying a truncated `_descriptor_error`.
7. **Cell 6 – Load vLLM**: loads tokenizer via `AutoTokenizer` (with `cache_dir=model_download_dir`), then iterates over `quantization` candidates `[cfg.quantization, 'awq_marlin', 'awq', None]`; instantiates `vllm.LLM` with `tensor_parallel_size`, `gpu_memory_utilization=0.82`, `max_model_len=3072`, `max_num_seqs=96`, `enforce_eager=False`, `disable_custom_all_reduce=True`, `disable_log_stats=True`, `download_dir`, and `enable_prefix_caching=True` when supported. On failure, runs `gc.collect()` and `torch.cuda.empty_cache()` and tries the next quantization; raises if all fail.
8. **Cell 7 – Generation helpers**: `render_prompt` builds a chat message list (`system` + `user`) and applies the tokenizer chat template with `enable_thinking=False`, with a repair variant that prepends the previous parse error and bad output. `generate_raw` calls `llm.generate` with `SamplingParams(temperature=0.0, top_p=1.0, max_tokens, repetition_penalty=1.02)`. `generate_raw_safe` recursively halves the batch on exception. `generate_descriptor` runs up to `max_retries+1` attempts; first attempt can reuse a pre-generated `first_raw`; subsequent attempts call `generate_raw_safe` with the repair prompt and `retry_max_new_tokens=448`; on terminal failure returns an `empty_descriptor` with status `failed_descriptor_parse`.
9. **Cell 8 – Batch loop**: iterates `work_df` in steps of `batch_size=64`, renders all prompts in the batch, calls `generate_raw_safe` once, then per-row invokes `generate_descriptor(citation, text, first_raw=raw)`. Streams each record (`_source_row`, `citation`, `text`, `llm_enrichment`, `llm_generation`) as a JSON line to `output_jsonl`, flushing per batch. Keeps up to 1000 records in memory for preview and collects all failures separately. After the loop, writes (or removes) the failures file, dumps the in-memory preview as `output_preview_csv` (with 17 selected columns including pipe-joined lists), and saves a `metrics` JSON containing timing, row counts, status counter, output paths, and the full config.
10. **Cell 9 – QC**: re-reads the JSONL and walks each record looking for any keys in the `FORBIDDEN` set; counts concepts/terms, flags presence of any topic field, and prints a status summary.
11. **Final markdown cell**: documents the downstream local step (`normalize_enriched_court_row`) that produces `rag_enrichment`, `normalized_anchors`, `anchor_quality_flags`, `retrieval_views`, and `enrichment_quality`.

## Results

- Cell 0: `Running in Colab: True`; `Drive already mounted at /content/drive`; `Setup done. If this was the first package install in a fresh Colab runtime, restart runtime once, then continue from Cell 1.`
- Cell 1: `Imports OK`; `Running in Colab: True`; `BASE_DIR: /content/drive/MyDrive/swiss_law`; `DATA_DIR: /content/drive/MyDrive/swiss_law/data`; `CUDA devices: 1`; `GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition; free=94.43 GiB / total=94.97 GiB`.
- Cell 2: Prints the full `Config` JSON (values as listed above); `Output JSONL: /content/drive/MyDrive/swiss_law/outputs/court_llm_descriptors_0000000_0009999.jsonl`.
- Cell 3: `Using input: /content/drive/MyDrive/swiss_law/data/court_considerations.csv`; `Shape: (2476315, 2)`; `Columns: ['citation', 'text']`; `Selected rows: 10000`. Head-20 preview shows German `BGE 139 I 2` considerations with `_text_len` ranging 118–2916.
- Cell 6: `Before vLLM load GPU 0: free=94.43 GiB total=94.97 GiB`; `Initializing vLLM with quantization='awq_marlin'`; `INFO 05-03 08:39:08 [model.py:555] Resolved architecture: Qwen3ForCausalLM`; `INFO 05-03 08:39:08 [model.py:1680] Using max model len 3072`; `INFO 05-03 08:39:08 [awq_marlin.py:252] The model is convertible to awq_marlin during runtime. Using awq_marlin kernel.`; `INFO 05-03 08:39:08 [scheduler.py:239] Chunked prefill is enabled with max_num_batched_tokens=16384.`; `INFO 05-03 08:39:09 [vllm.py:840] Asynchronous scheduling is enabled.`; `WARNING 05-03 08:39:12 [system_utils.py:157] We must use the spawn multiprocessing start method.`; `vLLM loaded`; `After vLLM load GPU 0: free=17.14 GiB total=94.97 GiB`.
- Cell 8 metrics JSON:

```json
{
  "start": 0,
  "limit": 10000,
  "selected_rows": 10000,
  "written_rows": 10000,
  "failures": 0,
  "elapsed_seconds": 759.1911044120789,
  "rows_per_second": 13.171914083139905,
  "status_counts": {"ok": 9998, "ok_after_retry": 2},
  "output_jsonl": "/content/drive/MyDrive/swiss_law/outputs/court_llm_descriptors_0000000_0009999.jsonl",
  "output_preview_csv": "/content/drive/MyDrive/swiss_law/outputs/court_llm_descriptors_0000000_0009999_preview.csv",
  "output_failures_jsonl": null
}
```

- Cell 8 preview head (selected columns): rows are `BGE 139 I 2 E. *` and `BGE 145 I 1 E. *` with `status='ok'`; `legal_area` values include `Administrative law`, `Constitutional Law`, `Local Government`, `Land use planning`, `Public Law`, `public administration`, `Freedom of expression`, `Federalism`; `primary_domain` values include `Administrative review`, `Federal Constitution`, `Initiatives`, `Planning law`, `Voting Rights`, `Federal Legislation`. The earlier in-memory snapshot in the notebook also shows specificity scores around 0.7–0.8 and 3–5 entries each for `concepts_en` and `terms_original`.
- Cell 9 QC: prints `Rows checked: 10000`; `Forbidden field rows: 0`; `Failed rows: 0`; `Status counts: {'ok': 9998, 'ok_after_retry': 2}`. Per-row QC head shows `forbidden_fields=[]`, `has_topic=True`, and `specificity_score` between 0.7 and 0.8 for the first 100 rows.

## Summary

This notebook runs the GPU-only LLM step of the Swiss court-considerations enrichment pipeline, generating compact "minimal semantic descriptors" for a 10,000-row pre-production shard of `court_considerations.csv` using Qwen/Qwen3-8B-AWQ served by vLLM on a single ~96 GiB Blackwell GPU. The schema is intentionally narrow (16 descriptor fields) and explicitly excludes anchors, outcomes, retrieval views, queries, summaries, and quality flags, which are deferred to a downstream local normalizer. Prompts are built from a strict JSON-only system message plus a per-row user template that trims text to 2,400 chars and embeds the schema hint; outputs are parsed with a hand-written balanced-brace JSON extractor, normalized, and any forbidden keys are stripped before writing. The batch loop streams JSONL with prefix caching enabled, performs at most one repair retry with a longer token budget, and recursively splits batches on OOM/runtime errors. The run processed all 10,000 rows in ~759 s (~13.17 rows/s) with zero failures (9,998 `ok`, 2 `ok_after_retry`), and Cell 9 QC confirmed no forbidden fields and full topic coverage across all rows.
