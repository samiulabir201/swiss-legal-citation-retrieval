# kaggle_qwen3_8b_awq_vllm_text_to_json_base

**Path:** e:\swiss_citation_extraction\notebooks\13_misc_kaggle_and_utilities\kaggle_qwen3_8b_awq_vllm_text_to_json_base.ipynb

## Configuration

Runtime targets Kaggle 2×T4 GPUs with the locally mounted model at `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1`, falling back to HuggingFace `Qwen/Qwen3-8B-AWQ`. Environment defaults set `TOKENIZERS_PARALLELISM=false`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7`. Installed packages: `vllm`, `transformers`, `accelerate`, `safetensors`, `tqdm`.

The `Config` dataclass (instance `CONFIG`) controls the run:

- Files: `input_file` resolved via `first_existing_input_file()` from candidate Kaggle paths; `output_file=/kaggle/working/court_authority_cards_rag_targets_qwen3_8b_vllm.jsonl`; `failed_file=…_failed.jsonl`; `checkpoint_file=rag_targets_qwen3_8b_vllm_checkpoint.txt`.
- Model: `quantization=awq`, `dtype=float16`, `max_model_len=4096`, `gpu_memory_utilization=0.90`, `tensor_parallel_size=2`.
- Throughput: `batch_size=128`, `max_num_seqs=128`, `enable_prefix_caching=True`.
- Prompt/output: `text_chars=1600`, `max_new_tokens=224`, `retry_max_new_tokens=384`.
- Sampling: `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.03`, stop tokens `["<|im_end|>", "</s>"]`.
- Reliability: `main_enable_thinking=False`, `retry_enable_thinking=True`, `retry_invalid=True`, `use_structured_outputs=True`.
- Run controls: `limit=500` (default test), `start=None`, `reset_output=False`, `flush_every_batches=1`.

Cell 8 overrides `CONFIG.limit=500`, `CONFIG.batch_size=128`, `CONFIG.max_num_seqs=128`, `CONFIG.reset_output=False` before calling `main(CONFIG)`. Cell 11 contains commented tuning presets for higher-accuracy, faster, and TP=1 fallback configurations.

## Data

- Input dataset slug: `swiss-court-authority-cards-rag-targets`, file `court_authority_cards_v4_target_cards.jsonl`. Each record is a court-authority card containing fields like `citation`, `language`, `court_base`, `authority_role`, `legal_area`, `family`, `subfamily`, `issue_labels_en`, `law_codes`, `statutes_cited`, `court_cases_cited`, `summary_en_proxy`, `retrieval_text_en`, `structural`, and `text_excerpt_original` (German, French, or Italian paragraph).
- Output: append-mode JSONL with each input card preserved plus `_source_line_idx`, `_rag_generation` (model, engine, source, elapsed_s, errors), and `rag_targets_qwen3_8b_awq` (the validated JSON). Failed records also include `_raw_model_output` truncated to 4000 chars.

## Pipeline

1. Cell 1 installs vLLM + transformers + accelerate + safetensors + tqdm.
2. Cell 2 imports and prints Torch/CUDA/GPU info.
3. Cell 3 builds the `CONFIG` dataclass with file resolvers (`first_existing_input_file`, `first_existing_model_source`).
4. Cell 4 smoke test: asserts input exists, counts lines, parses first card, checks model path/config/tokenizer_config/safetensors shards, loads tokenizer, renders a chat template with `enable_thinking=False`, verifies writable working dir.
5. Cell 5 defines `RAG_SCHEMA_HINT`, `REQUIRED_FIELDS`, enum value sets `ROLE_VALUES` (holding/reasoning/background/cost/procedural/disposition/standard_of_review/obiter) and `OUTCOME_VALUES` (granted/dismissed/inadmissible/remitted/partial/none), the `JSON_SCHEMA` (with `additionalProperties=False` and `maxItems` caps), `SYSTEM_PROMPT` (deterministic Swiss-legal extractor, English RAG terminology, no invention), `REPAIR_GUARD` repair instruction, helpers `safe_str`/`safe_list`, `build_user_prompt` (assembles schema + metadata + paragraph), `strip_think_blocks`, `extract_first_json_object` (fast `json.loads` then balanced-brace scan), `normalize_and_validate` (clamps strings to 1200 chars, lists with dedup, enforces enums with defaults `reasoning`/`none`, checks core-field non-emptiness and retrieval-term presence), and `parse_validate_raw`.
6. Cell 6 defines `VLLMJsonGenerator`: loads `LLM(model=…, quantization=awq, dtype=float16, trust_remote_code=True, tensor_parallel_size=…, gpu_memory_utilization=0.90, max_model_len=4096, max_num_seqs=128, enable_prefix_caching=True)`, downgrades TP if fewer CUDA devices are present, builds `fast_params` and `retry_params` with optional `StructuredOutputsParams(json=JSON_SCHEMA)`, renders chat templates with toggleable `enable_thinking`, and exposes `generate_raw` (uses `llm.generate` with `use_tqdm=False`).
7. Cell 7 defines `read_checkpoint`/`write_checkpoint`, `iter_cards` (skips to `start`, stops at `stop_before`, yields parse-error stubs `{_read_error, _raw_line}`), `make_output_record`, `run_one_batch` (renders prompts, generates main outputs, parses/validates, retries invalid items with `REPAIR_GUARD` + thinking enabled, keeps retry only if it validates or improves parseability, computes per-item elapsed), `append_jsonl` (with `f.flush()` + `os.fsync`), and `main(config)` which reads checkpoint, prints run summary, instantiates the generator, iterates cards into `pending` batches, flushes per batch (writes ok/failed JSONL, updates checkpoint, prints batch rate/avg/ETA via tqdm), and prints a final summary.
8. Cell 8 sets test parameters and calls `main(CONFIG)`.
9. Cell 9 reads the last 3 lines from the output JSONL and prints citation, `_rag_generation`, and `rag_targets_qwen3_8b_awq` for each.
10. Cell 10 (commented) shows the full production run invocation (`CONFIG.limit=0`).
11. Cell 11 (commented) lists tuning presets.

## Results

The notebook has no executed outputs. All code cells are unexecuted; no install logs, smoke-test prints, batch progress, sample dumps, or final-summary numbers are present in the file.

## Summary

This Kaggle-targeted notebook converts Swiss court-authority-card paragraphs (German/French/Italian) into validated English RAG-target JSON using `Qwen3-8B-AWQ` served by vLLM offline inference on 2×T4 GPUs. The pipeline applies deterministic decoding (temperature 0, repetition penalty 1.03), enforces a strict JSON schema via vLLM `StructuredOutputsParams` (with a robust balanced-brace parser fallback), and uses a retry pass with `enable_thinking=True` plus a repair-guard suffix to recover invalid items. State is durable: it appends to JSONL, fsyncs each batch, separates failed records, and resumes from a checkpoint file. The notebook is shipped unexecuted, so no throughput, validation, or ETA numbers are recorded; Cell 8 is configured for a 500-card test run and Cell 10 (commented) is the full production trigger.
