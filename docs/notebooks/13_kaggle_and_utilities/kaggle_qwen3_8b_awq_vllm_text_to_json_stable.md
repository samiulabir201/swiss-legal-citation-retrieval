# kaggle_qwen3_8b_awq_vllm_text_to_json_stable

**Path:** e:\swiss_citation_extraction\notebooks\13_misc_kaggle_and_utilities\kaggle_qwen3_8b_awq_vllm_text_to_json_stable.ipynb

## Configuration

- Runtime: Kaggle 2×T4 with vLLM offline inference.
- Install (Cell 1): `pip -q install -U --no-cache-dir vllm transformers accelerate safetensors tqdm`.
- Environment (Cell 2) sets `TOKENIZERS_PARALLELISM=false`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7`, `OMP_NUM_THREADS=1`, `VLLM_ATTENTION_BACKEND=TRITON_ATTN`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- Model: local Kaggle path `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1`, fallback `Qwen/Qwen3-8B-AWQ`.
- `Config` dataclass (Cell 3) fields:
  - `quantization="awq"`, `dtype="float16"`, `max_model_len=3072`, `gpu_memory_utilization=0.62`.
  - `tensor_parallel_size=2`, `disable_custom_all_reduce=True`, `attention_backend="TRITON_ATTN"`, `enforce_eager=True`.
  - `auto_shrink_memory_request=True`, `auto_fallback_to_single_gpu=False`.
  - `batch_size=64`, `max_num_seqs=64`.
  - `text_chars=1400`, `max_new_tokens=224`, `retry_max_new_tokens=384`.
  - `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.03`.
  - `main_enable_thinking=False`, `retry_enable_thinking=True`, `retry_invalid=True`, `use_structured_outputs=True`.
  - `limit=500`, `start=None`, `reset_output=False`, `flush_every_batches=1`.
- Output paths: `/kaggle/working/court_authority_cards_rag_targets_qwen3_8b_vllm.jsonl`, failed at `..._failed.jsonl`, checkpoint at `rag_targets_qwen3_8b_vllm_checkpoint.txt`.

## Data

- Input dataset slug: `swiss-court-authority-cards-rag-targets`.
- Input filename: `court_authority_cards_v4_target_cards.jsonl`.
- Candidate input lookups search `/kaggle/input/{slug}/...`, the `samiulislam180041221` dataset path, generic `/kaggle/input/datasets`, `/kaggle/input`, current dir, and a recursive glob.
- Per-card fields consumed by the prompt builder: `text_excerpt_original`, `citation`, `language`, `court_base`, `authority_role`, `legal_area`, `family`, `subfamily`, `issue_labels_en`, `law_codes`, `statutes_cited`, `court_cases_cited`, `summary_en_proxy`, `retrieval_text_en`, `structural`.

## Pipeline

- Cell 1 installs vLLM + supporting libs.
- Cell 2 imports, sets env vars (forces Triton attention before importing vLLM), prints Torch/CUDA info.
- Cell 3 defines `Config`, resolves input path via `first_existing_input_file`, resolves model via `first_existing_model_source`, instantiates `CONFIG`, prints resolved paths and key options.
- Cell 4 smoke test: asserts input exists, prints size and line count, peeks first JSON card, verifies local model files (`config.json`, `tokenizer_config.json`, safetensors shards), loads tokenizer, applies chat template (with `enable_thinking=False`), tests `/kaggle/working` write.
- Cell 5 defines:
  - `RAG_SCHEMA_HINT`, `REQUIRED_FIELDS`, enum sets `ROLE_VALUES` and `OUTCOME_VALUES`, full `JSON_SCHEMA` (with caps `english_legal_concepts` ≤8, `search_keywords` ≤12, `natural_language_queries` ≤5, `additionalProperties=False`).
  - `SYSTEM_PROMPT` (deterministic Swiss legal text-to-JSON engine, no markdown/CoT) and `REPAIR_GUARD` for retries.
  - Helpers: `safe_str`, `safe_list`, `build_user_prompt`, `strip_think_blocks`, `extract_first_json_object` (fast path + balanced-brace parser), `normalize_and_validate` (truncates strings to 1200 chars, lowercases enums, defaults bad role→`reasoning`/bad outcome→`none`, enforces non-empty core/retrieval fields), `parse_validate_raw`.
- Cell 6 defines vLLM generator:
  - `gpu_free_fraction` reads min free/total VRAM across visible CUDA devices.
  - `shrink_memory_request_if_needed` caps `gpu_memory_utilization` to `free_fraction - 0.08`, floor 0.45.
  - `VLLMJsonGenerator` loads tokenizer, clamps TP to device count, calls `_load_llm_with_fallback` which builds `LLM(**kwargs)` with `enable_prefix_caching=True`, `disable_log_stats=True`, `enforce_eager`, `disable_custom_all_reduce`, and `attention_backend` when supported; retries without `attention_backend` on `TypeError`; on full failure, optional fallback to `tensor_parallel_size=1`, `gpu_memory_utilization=0.55`, `max_num_seqs=48`, `batch_size=48`, `max_model_len=3072`.
  - Creates `fast_params` and `retry_params` with `StructuredOutputsParams(json=JSON_SCHEMA)` when available, stop tokens `["<|im_end|>", "</s>"]`.
  - `render_prompt` applies chat template with `enable_thinking` toggle; `generate_raw` runs `llm.generate` with `use_tqdm=False`.
- Cell 7 defines checkpoint IO (`read_checkpoint`, `write_checkpoint`), `iter_cards` streaming reader with start/stop bounds and per-line JSON error capture, `make_output_record` (embeds source card, `_source_line_idx`, `_rag_generation` block with model/engine/source/elapsed/errors, attaches `rag_targets_qwen3_8b_awq` payload, retains `_raw_model_output` truncated to 4000 chars on errors), `run_one_batch` (generates main pass, retries invalid items with `REPAIR_GUARD` and `retry_enable_thinking`, keeps retry result only when it validates or improves parseability), `append_jsonl` (with `f.flush` + `os.fsync`), and `main` (computes start from checkpoint or `config.start`, computes `stop_before` from `limit`, batches `pending`, flushes per `batch_size`, writes OK/failed jsonl, updates checkpoint to `last_line_idx`, prints per-batch and final stats including ETA).
- Cell 8 sets conservative T4 defaults and calls `main(CONFIG)` for a 500-card run.
- Cell 9 reads tail of output JSONL and prints citation, `_rag_generation`, and indented `rag_targets_qwen3_8b_awq`.
- Cell 10 commented-out full production run (`limit=0`).
- Cell 11 commented-out tuning presets (safer TP=1 mode, faster post-validation settings, higher-accuracy slow settings, faster low-accuracy settings, full run).

## Results

No cells produced executed outputs in the notebook (no stdout, no errors, no display data recorded).

## Summary

The notebook is a Kaggle 2×T4 production pipeline that uses vLLM offline inference with a local Qwen3-8B-AWQ model to convert Swiss court paragraph cards from `court_authority_cards_v4_target_cards.jsonl` into deterministic JSON RAG targets matching a fixed 11-field schema (summary, topic, question, rule, holding, factual context, concepts, keywords, NL queries, paragraph_role enum, outcome_signal enum). Reliability is engineered around known Kaggle 2×T4 failure modes: `disable_custom_all_reduce=True`, forced `TRITON_ATTN` backend (to dodge FlashInfer JIT `-lcuda` errors), `enforce_eager=True`, auto-shrinking `gpu_memory_utilization` against live free VRAM, and an optional TP=2 → TP=1 fallback with reduced batch/memory settings. Generation is deterministic (`temperature=0.0`) with optional vLLM `StructuredOutputsParams` for schema-constrained decoding, a non-thinking fast pass, and a thinking-enabled repair pass with `REPAIR_GUARD` for items that fail JSON parsing or schema validation. The loop is resume-safe via a line-index checkpoint, writes valid outputs and failures to separate JSONL files with `fsync`, and ships preset Cells 8/10/11 for a 500-card smoke run, the full production run, and tuning. None of the cells were executed in this saved notebook, so no measured throughput or sample output is recorded.
