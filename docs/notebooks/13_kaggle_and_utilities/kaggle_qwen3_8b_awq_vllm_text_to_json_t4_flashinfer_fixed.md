# kaggle_qwen3_8b_awq_vllm_text_to_json_t4_flashinfer_fixed

**Path:** e:\swiss_citation_extraction\notebooks\13_misc_kaggle_and_utilities\kaggle_qwen3_8b_awq_vllm_text_to_json_t4_flashinfer_fixed.ipynb

## Configuration

- Environment: Kaggle (detected via `/kaggle/working`), targeting 2×T4 GPUs.
- Runtime install (Cell 1): `pip -q install -U --no-cache-dir vllm transformers accelerate safetensors tqdm`.
- Environment variables (set before importing vLLM):
  - `TOKENIZERS_PARALLELISM=false`
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID`
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7`
  - `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (forces Triton attention to avoid FlashInfer JIT linker failures on Kaggle, e.g. `cannot find -lcuda`).
  - `VLLM_WORKER_MULTIPROC_METHOD=spawn`
- `Config` dataclass fields:
  - Files: `input_file` resolved via `first_existing_input_file()` (searches Kaggle dataset paths for `court_authority_cards_v4_target_cards.jsonl` under slug `swiss-court-authority-cards-rag-targets`); `output_file=/kaggle/working/court_authority_cards_rag_targets_qwen3_8b_vllm.jsonl`; `failed_file=...qwen3_8b_vllm_failed.jsonl`; `checkpoint_file=...qwen3_8b_vllm_checkpoint.txt`.
  - Model: `model_id` = local Kaggle path `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1` if present, else HF fallback `Qwen/Qwen3-8B-AWQ`; `quantization="awq"`; `dtype="float16"`; `max_model_len=4096`; `gpu_memory_utilization=0.90`.
  - Tensor parallelism: `tensor_parallel_size=2`; `disable_custom_all_reduce=True`; `attention_backend="TRITON_ATTN"`; `auto_fallback_to_single_gpu=True`.
  - Throughput: `batch_size=128`; `max_num_seqs=128`; `enable_prefix_caching=True`; `disable_log_stats=True`.
  - Prompts/output: `text_chars=1600`; `max_new_tokens=224`; `retry_max_new_tokens=384`.
  - Sampling/accuracy: `temperature=0.0`; `top_p=1.0`; `repetition_penalty=1.03`; `main_enable_thinking=False`; `retry_enable_thinking=True`; `retry_invalid=True`; `use_structured_outputs=True`.
  - Run controls: `limit=500`; `start=None`; `reset_output=False`; `flush_every_batches=1`.
- Stop tokens: `["<|im_end|>", "</s>"]`.
- Structured outputs use `vllm.sampling_params.StructuredOutputsParams(json=JSON_SCHEMA)` when importable.

## Data

- Input JSONL: `court_authority_cards_v4_target_cards.jsonl` from Kaggle dataset slug `swiss-court-authority-cards-rag-targets`.
- Each input card carries Swiss court paragraph metadata: `citation`, `language`, `court_base`, `authority_role`, `legal_area`, `family`, `subfamily`, `issue_labels_en`, `law_codes`, `statutes_cited`, `court_cases_cited`, `summary_en_proxy`, `retrieval_text_en`, `structural`, and the original paragraph text in `text_excerpt_original` (truncated to `text_chars=1600`).
- Outputs (written by `main`):
  - `court_authority_cards_rag_targets_qwen3_8b_vllm.jsonl` — successful records.
  - `court_authority_cards_rag_targets_qwen3_8b_vllm_failed.jsonl` — invalid/incomplete records.
  - `rag_targets_qwen3_8b_vllm_checkpoint.txt` — next line index for resume.
- Each output record copies the original card and adds `_source_line_idx`, `_rag_generation` (`model`, `engine="vllm"`, `source` in `{main, retry}`, `elapsed_s`, `errors`), `rag_targets_qwen3_8b_awq` (normalized JSON), and `_raw_model_output` (truncated 4000 chars) when errors are present.

## Pipeline

- Cell 1: install vLLM + Transformers + Accelerate + Safetensors + tqdm.
- Cell 2 (imports/env): seeds environment variables before importing vLLM, prints torch/CUDA info and per-GPU VRAM.
- Cell 3 (Configuration): instantiates `CONFIG = Config()`, resolves input file and model source, prints paths and settings.
- Cell 4 (Smoke Test): asserts input exists; counts lines; loads first JSONL record and prints `citation` and fields; checks local model directory for `config.json`, `tokenizer_config.json`, and `*.safetensors` shards; loads tokenizer; probes chat-template rendering with `enable_thinking=False`; verifies the working directory is writable.
- Cell 5 (Schema/Prompt/Validators):
  - `RAG_SCHEMA_HINT` defines required fields: `english_summary`, `legal_topic`, `legal_question`, `legal_rule`, `court_holding`, `factual_context`, `english_legal_concepts[]`, `search_keywords[]`, `natural_language_queries[]`, `paragraph_role` (enum `holding|reasoning|background|cost|procedural|disposition|standard_of_review|obiter`), `outcome_signal` (enum `granted|dismissed|inadmissible|remitted|partial|none`).
  - `JSON_SCHEMA` enforces types, enums, and `maxItems` (8/12/5) for the three arrays.
  - `SYSTEM_PROMPT` instructs deterministic English-terminology JSON-only output, prohibits invention of statutes/cases, and forbids markdown or chain-of-thought.
  - `build_user_prompt` packages metadata + paragraph text into the user turn.
  - `strip_think_blocks` removes `<think>…</think>` and markdown fences.
  - `extract_first_json_object` does a fast `json.loads` then balanced-brace extraction over the response.
  - `normalize_and_validate` enforces presence of required keys, clamps strings (≤1200 chars) and arrays (8/12/5 with dedup), and falls back to `paragraph_role="reasoning"` / `outcome_signal="none"` on enum violations; appends errors for `missing:*`, `bad_paragraph_role:*`, `bad_outcome_signal:*`, `empty_core_fields`, `empty_retrieval_terms`.
- Cell 6 (`VLLMJsonGenerator`):
  - Sets `VLLM_ATTENTION_BACKEND` again at import time.
  - Loads tokenizer; clamps `tensor_parallel_size` to available CUDA devices.
  - `_load_llm_with_fallback`: tries TP=requested first; on exception, when `auto_fallback_to_single_gpu` is true, calls `gc.collect()` + `torch.cuda.empty_cache()` and retries with `tensor_parallel_size=1`, `max_num_seqs/batch_size` capped at 96.
  - LLM kwargs: `quantization`, `dtype`, `trust_remote_code=True`, `tensor_parallel_size`, `gpu_memory_utilization`, `max_model_len`, `max_num_seqs`, `enable_prefix_caching=True`, `disable_log_stats=True`, `disable_custom_all_reduce`.
  - Builds two `SamplingParams` (fast/retry) optionally with `StructuredOutputsParams(json=JSON_SCHEMA)`.
  - `render_prompt` applies the Qwen chat template with `enable_thinking` toggled per pass; `generate_raw` runs `llm.generate(prompts, sampling_params=…, use_tqdm=False)` and returns first-completion text per item.
- Cell 7 (IO + main loop):
  - `read_checkpoint` / `write_checkpoint` track next line index.
  - `iter_cards` yields `(idx, card)` honoring `start` and `stop_before`; surfaces JSON read errors as a `_read_error` record.
  - `run_one_batch`: main generation pass; for items with errors, when `retry_invalid=True`, re-runs them with `retry_enable_thinking=True`, `retry_max_new_tokens`, and a `REPAIR_GUARD` user-message suffix; replaces results when the retry validates or improves parseability.
  - `make_output_record` merges card + `rag_targets_qwen3_8b_awq` + `_rag_generation` metadata; includes `_raw_model_output` only on errors.
  - `append_jsonl` writes with `flush()` + `os.fsync()`.
  - `main`: counts lines, optionally resets outputs, computes `start`/`stop_before`, instantiates `VLLMJsonGenerator`, iterates with a `tqdm` bar, calls `flush_batch()` every `batch_size`; logs per-batch `ok/failed/retried/cards_s/avg/eta_h/checkpoint`; final summary prints processed/ok/failed/retried/elapsed/average and a full-run ETA.
- Cell 8 (Run a quality/throughput test): sets `CONFIG.limit=500`, `batch_size=128`, `max_num_seqs=128`, `tensor_parallel_size=2`, `disable_custom_all_reduce=True`, `attention_backend="TRITON_ATTN"`, `auto_fallback_to_single_gpu=True`, `reset_output=False`, then calls `main(CONFIG)`.
- Cell 9 (Inspect output quality): `read_jsonl_tail(CONFIG.output_file, 3)` prints the last three citations, their `_rag_generation` metadata, and pretty-printed `rag_targets_qwen3_8b_awq`.
- Cell 10 (Full production run): commented-out template with `CONFIG.limit=0`, `CONFIG.reset_output=False`, `main(CONFIG)`.
- Cell 11 (Optional tuning presets): commented presets for higher-accuracy (`text_chars=2200`, `max_new_tokens=320`, etc.), faster (`text_chars=1200`, `max_new_tokens=192`, batch 192), TP=1 fallback (batch 96), Kaggle 2×T4 stable startup, and an `awq_marlin` speed experiment.

## Results

The notebook has no executed cell outputs (verbatim: empty — no stdout, no errors, no figures, no displayed values for any cell).

## Summary

This Kaggle notebook is a production-style text-to-JSON enrichment pipeline that uses vLLM to drive a locally hosted Qwen3-8B-AWQ model over Swiss court paragraph cards, producing a fixed RAG-target schema (`english_summary`, `legal_question`, `legal_rule`, `court_holding`, etc., plus enums `paragraph_role` and `outcome_signal`). The Kaggle 2×T4 reliability story is central: `VLLM_ATTENTION_BACKEND=TRITON_ATTN` is forced before vLLM is imported to dodge FlashInfer JIT `-lcuda` linker failures, `disable_custom_all_reduce=True` avoids the `custom_all_reduce.cuh` startup crash, and an automatic TP=2→TP=1 fallback path is built into the LLM loader. Generation is deterministic (`temperature=0.0`) and uses vLLM `StructuredOutputsParams` when available, with a two-pass design: a fast main pass without thinking and a retry pass with `enable_thinking=True`, longer `max_new_tokens`, and an extra repair guard for any item that fails JSON parsing or schema/enum validation. The IO layer is resume-safe via a line-index checkpoint, splits outputs into success/failed JSONL files, and emits per-batch throughput and ETA logs. The notebook ships with a smoke test, a 500-card quality probe, a tail-inspection cell, and commented tuning presets for higher-accuracy, faster, TP=1 fallback, and `awq_marlin` speed experimentation.
