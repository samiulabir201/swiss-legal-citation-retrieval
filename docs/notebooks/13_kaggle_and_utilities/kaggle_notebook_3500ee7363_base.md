# kaggle_notebook_3500ee7363_base

**Path:** e:\swiss_citation_extraction\notebooks\13_misc_kaggle_and_utilities\kaggle_notebook_3500ee7363_base.ipynb

## Configuration

- Target runtime: Kaggle T4 with Qwen3-8B-AWQ.
- Install cell upgrades `pandas`, `tqdm`, `transformers`, `accelerate`, `safetensors`, `gptqmodel`; installs `vllm`; then uninstalls `flashinfer`, `flashinfer-python`, `flashinfer-python-cu12` to avoid `cannot find -lcuda` JIT/linker crashes.
- Environment defaults: `TOKENIZERS_PARALLELISM=false`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7`, `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, `OMP_NUM_THREADS=1`.
- Model resolution: tries local `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1`, else `Qwen/Qwen3-8B-AWQ`.
- `Config` dataclass (effective at runtime, per output):
  - `input_csv=/kaggle/input/competitions/llm-agentic-legal-information-retrieval/court_considerations.csv`
  - `fallback_input_csv=court_consideration.csv`
  - `output_dir=/kaggle/working`
  - `output_jsonl=enriched_court_citations_10.jsonl`
  - `output_preview_csv=enriched_court_citations_10_preview.csv`
  - `n_rows=10`, `random_seed=None`, `sample_random=False`, `text_chars=3200`
  - `model_name=Qwen/Qwen3-8B-AWQ`
  - `engine=auto`
  - vLLM: `tensor_parallel_size=1`, `gpu_memory_utilization=0.62`, `max_model_len=3072`, `enforce_eager=True`, `max_num_seqs=32`, `quantization=awq`, `disable_custom_all_reduce=True`, `force_triton_attention=True`
  - Generation: `batch_size=8`, `max_new_tokens=384`, `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.02`, `enable_thinking=False`
- GPUs detected: 2x Tesla T4, 14.56 GiB each, ~14.46 GiB free.
- Torch 2.11.0+cu130, CUDA available.

## Data

- Input file (resolved): `/kaggle/input/competitions/llm-agentic-legal-information-retrieval/court_considerations.csv`.
- Shape: `(2476315, 2)`; columns: `['citation', 'text']`.
- Column inference selects `citation_col=citation`, `text_col=text`.
- Filtering: requires both fields non-null and stripped text length > 30 chars.
- Sampling: deterministic head; `sample_random=False`, `n_rows=10`. Selected 10 rows.
- First selected citations: `BGE 139 I 2 E. 1.12.2011`, `BGE 139 I 2 E. 2`, `BGE 139 I 2 E. 5.1`, `BGE 139 I 2 E. 5.2`, `BGE 139 I 2 E. 5.3`, `BGE 139 I 2 E. 7.1`, `BGE 139 I 2 E. 5.4`, `BGE 139 I 2 E. 7.1`, `BGE 139 I 2 E. 5.5`, `BGE 139 I 2 E. 5.6`.

## Pipeline

- Cell 1 - Install dependencies; force-remove FlashInfer.
- Cell 2 - Imports, env vars, GPU probe, `Config` dataclass; prints effective config dict.
- Cell 3 - `resolve_input_path` searches Kaggle competition path then fallbacks; loads CSV with pandas.
- Cell 4 - `pick_column` heuristic to infer `citation`/`text` columns; filters NaN and short rows; sample by head or random.
- Cell 5 - Defines `REQUIRED_FIELDS` (17), enum sets (`ROLE_VALUES`, `OUTCOME_VALUES`, `AUTHORITY_VALUES`), `JSON_SCHEMA_TEXT`, `SYSTEM_PROMPT` (deterministic Swiss legal text-to-JSON), `USER_TEMPLATE`; helpers `make_messages`, `extract_json_object` (strips markdown/think-tags, balanced-brace scan), `clean_str`, `clean_list`, `normalize_enrichment` (clamps fields, enforces enums, fills defaults).
- Cell 6 - `JsonEngine` class:
  - Loads tokenizer via `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`.
  - Engine `auto`: tries vLLM, falls back to Transformers + GPTQModel AWQ.
  - `_attention_config()` imports `vllm.config.AttentionConfig` and forces `TRITON_ATTN`.
  - `_load_vllm()` instantiates `LLM` with config kwargs incl. `quantization=awq`, `enforce_eager=True`, `disable_custom_all_reduce=True`, `attention_config`; retries without `attention_config` on `TypeError`.
  - `_load_transformers()` loads `AutoModelForCausalLM` with `device_map=auto`, `dtype=fp16`.
  - `_render_prompts` calls `apply_chat_template(..., enable_thinking=False)`.
  - `generate()` uses `SamplingParams(temperature=0, top_p=1, max_tokens=384, repetition_penalty=1.02, stop=[<|im_end|>, </s>])` on vLLM, or HF `generate(do_sample=False)` on fallback.
- Cell 7 - Iterates `work_df` in batches of `cfg.batch_size=8`, invokes `engine.generate(rows)`, parses via `extract_json_object`, normalizes, accumulates `records`/`failures`; writes JSONL, preview CSV, and failures JSONL.
- Cell 8 - Pretty-prints first record (or first failure if none).

## Results

- Cell 2 print:
  - `Python imports OK`
  - `Torch: 2.11.0+cu130`
  - `CUDA available: True`
  - `GPU 0: Tesla T4 total=14.56 GiB free=14.46 GiB`
  - `GPU 1: Tesla T4 total=14.56 GiB free=14.46 GiB`
  - Effective config printed as a dict.
- Cell 3: `Using input: /kaggle/input/competitions/llm-agentic-legal-information-retrieval/court_considerations.csv`; `Shape: (2476315, 2)`; `Columns: ['citation', 'text']`.
- Cell 4: `citation_col: citation`; `text_col: text`; `Selected rows: 10`.
- Cell 6 vLLM init log highlights:
  - `[vLLM] using explicit AttentionConfig backend=TRITON_ATTN`
  - `WARNING ... Unknown vLLM environment variable detected: VLLM_ATTENTION_BACKEND`
  - `Resolved architecture: Qwen3ForCausalLM`; `Using max model len 3072`
  - `Detected that the model can run with awq_marlin, however you specified quantization=awq explicitly, so forcing awq.`
  - `Initializing a V1 LLM engine (v0.20.0)` with `quantization=awq`, `enforce_eager=True`, `disable_custom_all_reduce=True`.
  - `Using AttentionBackendEnum.TRITON_ATTN backend.`
  - Weight load: 2 shards, 5.68 GiB checkpoint, took 5.76 s, model loaded `5.69 GiB` in `6.611704 s`.
  - `Available KV cache memory: 2.37 GiB`; `GPU KV cache size: 17,280 tokens`; `Maximum concurrency for 3,072 tokens per request: 5.62x`.
  - `init engine (profile, create kv cache, warmup model) took 7.84 s`.
  - `Engine: vLLM`.
- Cell 7:
  - `Generated 0 OK, 10 failed in 55.1s`
  - `Rows/s: 0.182`
  - `Failures written to: /kaggle/working/enriched_court_citations_10_failures.jsonl`
  - `JSONL: /kaggle/working/enriched_court_citations_10.jsonl`
  - `Preview CSV: /kaggle/working/enriched_court_citations_10_preview.csv`
  - Preview DataFrame: empty (no columns, no rows).
- Cell 8: `No successful records. Check failures.` First failure (row 0, `BGE 139 I 2 E. 1.12.2011`) printed; error is `ValueError('No balanced JSON object found: ...')` because the raw output is truncated mid-`summary_en` field (cut off by `max_new_tokens=384`). The truncated raw begins `{ "legal_area": "Civil Law", "legal_domain_path": ["Procedural Law", "Evidence", "Judicial Conduct", "Litigation"], ...` and ends mid-string at `"summary_en`.

## Summary

This base-revision notebook builds a Qwen3-8B-AWQ pipeline on Kaggle T4 to enrich Swiss court considerations into a 17-field JSON schema, with deterministic generation (`temperature=0`), Triton attention, and a Transformers fallback if vLLM fails. The install cell removes FlashInfer to avoid the `cannot find -lcuda` linker crash, and the vLLM loader forces `attention_config=AttentionConfig(backend=TRITON_ATTN)` and `quantization=awq`. On the 10-row head sample from the 2,476,315-row competition CSV, the engine loads cleanly (5.69 GiB model, 17,280-token KV cache, vLLM v0.20.0) but generation fails on every row: `max_new_tokens=384` is too small for the 17-field schema, so outputs are truncated mid-JSON and `extract_json_object` rejects all 10 with `No balanced JSON object found`. End state: 0 successful records, 10 failures, 55.1 s wall (0.182 rows/s), failures JSONL written; preview CSV is empty. The fix is implied: raise `max_new_tokens`, which later revisions in this folder address.
