# kaggle_notebook_d37248e1eb

**Path:** `e:\swiss_citation_extraction\notebooks\13_misc_kaggle_and_utilities\kaggle_notebook_d37248e1eb.ipynb`

## Configuration

Kaggle vLLM offline-inference notebook titled "Fastest Reliable Text -> JSON Enrichment with Qwen3-8B-AWQ + vLLM". Target model is the local Kaggle AWQ build at `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1` with HF fallback `Qwen/Qwen3-8B-AWQ`.

Cell 1 installs `vllm transformers accelerate safetensors tqdm` via pip.

Cell 2 sets environment defaults before importing vLLM:
- `TOKENIZERS_PARALLELISM=false`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7`
- `OMP_NUM_THREADS=1`
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (forces Triton attention to avoid FlashInfer JIT/linker failures on Kaggle)
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`

Cell 3 `Config` dataclass:
- Files: input `court_authority_cards_v4.jsonl` discovered via `first_existing_input_file()`; output `/kaggle/working/court_authority_cards_v4_enriched.jsonl`; failed `/kaggle/working/court_authority_cards_v4_enriched_failed.jsonl`; checkpoint `/kaggle/working/court_authority_cards_v4_enriched_checkpoint.txt`.
- Model: `quantization='awq'`, `dtype='float16'`, `max_model_len=3072`, `gpu_memory_utilization=0.62`.
- TP/Stability: `tensor_parallel_size=2`, `disable_custom_all_reduce=True`, `attention_backend='TRITON_ATTN'`, `enforce_eager=True`, `auto_shrink_memory_request=True`, `auto_fallback_to_single_gpu=False`.
- Throughput: `batch_size=64`, `max_num_seqs=64`.
- Prompt/output: `text_chars=1400`, `max_new_tokens=224`, `retry_max_new_tokens=384`.
- Sampling: `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.03`, `main_enable_thinking=False`, `retry_enable_thinking=True`, `retry_invalid=True`, `use_structured_outputs=True`.
- Run: `limit=500` (smoke test), `start=None` (auto from checkpoint), `reset_output=False`, `flush_every_batches=1`.
- `TOTAL_V4_LINES = 2_476_315` is hard-coded to skip counting passes.

## Data

- Expected input artifact: `court_authority_cards_v4.jsonl` (a v4 Swiss court-authority cards corpus, full nominal size 2,476,315 lines).
- Cell 4 smoke-test actually resolves the input to `/kaggle/input/datasets/samiulislam180041221/swiss-court-authority-cards-rag-targets/court_authority_cards_v4_target_cards.jsonl` (0.96 GB, 363,258 lines).
- First record citation `BGE 139 I 2 E. 5.7`; fields: `_enrichment_target, authority_role, citation, court_base, court_cases_cited, family, is_notification_paragraph, issue_labels_en, language, law_codes, legal_area, matched_terms_multilingual, pattern, provenance, retrieval_text_en, statutes_cited, structural, subfamily, summary_en_proxy, text_excerpt_original`.
- Model assets verified: `config.json`, `tokenizer_config.json` present; 2 safetensors shards.

## Pipeline

Cell 5 defines the RAG enrichment schema and validators:
- `RAG_SCHEMA_HINT` / `REQUIRED_FIELDS`: `english_summary, legal_topic, legal_question, legal_rule, court_holding, factual_context, english_legal_concepts, search_keywords, natural_language_queries, paragraph_role, outcome_signal`.
- `ROLE_VALUES = {holding, reasoning, background, cost, procedural, disposition, standard_of_review, obiter}`.
- `OUTCOME_VALUES = {granted, dismissed, inadmissible, remitted, partial, none}`.
- `JSON_SCHEMA`: full JSON Schema with maxItems caps (concepts=8, keywords=12, queries=5), `additionalProperties=False`.
- `SYSTEM_PROMPT`: deterministic Swiss legal text-to-JSON engine; no invention; empty string/array/`'none'` for unsupported; no markdown/CoT.
- `REPAIR_GUARD`: strict repair instruction for retries.
- Helpers: `safe_str` (whitespace-collapse + truncate), `safe_list` (dedupe + cap), `build_user_prompt` (assembles metadata + paragraph excerpt), `strip_think_blocks` (removes `<think>` and code fences), `extract_first_json_object` (balanced-brace JSON extraction), `normalize_and_validate` (fills missing keys, normalizes enums, checks core/retrieval emptiness), `parse_validate_raw` (combined pipeline returning `(norm, errors)`).

Cell 6 `VLLMJsonGenerator`:
- `gpu_free_fraction()` reads `torch.cuda.mem_get_info` across visible devices.
- `shrink_memory_request_if_needed()` lowers `gpu_memory_utilization` to `min(requested, free_frac - margin)` floored at 0.45.
- `_llm_kwargs` builds vLLM kwargs: `quantization=awq`, `dtype=float16`, `tensor_parallel_size`, `gpu_memory_utilization`, `max_model_len`, `max_num_seqs`, `enable_prefix_caching=True`, `disable_log_stats=True`, `enforce_eager`, `disable_custom_all_reduce`, optional `attention_backend`.
- `_load_once` retries without `attention_backend` if older vLLM build rejects the kwarg.
- `_load_llm_with_fallback` optionally falls back to TP=1 with shrunk memory/batch/seq/model-len if initial load fails (guarded by `auto_fallback_to_single_gpu`).
- `_make_params` constructs `SamplingParams` with stop tokens `<|im_end|>`, `</s>`, and attaches `StructuredOutputsParams(json=JSON_SCHEMA)` when available.
- `render_prompt` applies the Qwen chat template with `enable_thinking` toggle and optional `extra_guard` (REPAIR_GUARD for retries).
- `generate_raw` batches prompts and returns raw model texts.

Cell 7 IO and main loop:
- `read_checkpoint` / `write_checkpoint` for `checkpoint_file`.
- `iter_cards(start, stop_before)` streams JSONL; malformed lines yield a `_read_error` stub.
- `make_output_record` clones the card, attaches `_source_line_idx`, `_rag_generation` metadata (model, engine `vllm`, method `qwen3_8b_vllm`, source `main|retry`, elapsed_s, errors), and `rag_enrichment` (the normalized schema record consumed by `experiment_enrichment_recall_stress.py` and BM25/vector pipelines). Failed records also carry `_raw_model_output` (first 4000 chars).
- `run_one_batch` performs a main-pass generation, validates each output, retries invalid ones with thinking enabled and `REPAIR_GUARD`, keeps the better of the two, and returns ok/failed records plus stats.
- `append_jsonl` writes with `flush()` + `os.fsync()`.
- `main(config)` reads checkpoint, computes `stop_before`, loads the generator, iterates input, calls `flush_batch()` per `batch_size`, updates a tqdm bar, prints per-batch stats and ETA (hours + Kaggle 9 h sessions), final summary at the end.

Cell 8 runs a 500-card smoke test setting `limit=500, batch_size=64, max_num_seqs=64, max_model_len=3072, gpu_memory_utilization=0.62, tensor_parallel_size=2`, with `disable_custom_all_reduce=True`, `enforce_eager=True`, `attention_backend='TRITON_ATTN'`, no reset.

Cell 9 reads the tail 3 records from the output JSONL and prints citation, `_rag_generation`, and `rag_enrichment`.

Cell 10 is the full 2.4 M production run (`limit=0`, `reset_output=False`); resume protocol documented in comments (Restart Session, re-run Cells 1-7, run Cell 10).

Cell 11 `check_progress()` reports checkpoint position, output/failed line counts, remaining, and ETA at 3/5/8/12 cards/s; also documents tuning presets (SAFEST, FASTER, HIGHER ACCURACY, LEAN) as commented config snippets.

## Results

Only Cells 2 and 4 produced visible output in the notebook; Cells 8-11 were not executed.

Cell 2 environment probe:
```
Python OK
VLLM_ATTENTION_BACKEND: TRITON_ATTN
Torch: 2.11.0+cu130
CUDA: True
GPU 0: Tesla T4 VRAM=14.6 GB
GPU 1: Tesla T4 VRAM=14.6 GB
```

Cell 4 smoke test:
```
Input exists: /kaggle/input/datasets/samiulislam180041221/swiss-court-authority-cards-rag-targets/court_authority_cards_v4_target_cards.jsonl
Input size GB: 0.9582597967237234
Input lines: 363258
First citation: BGE 139 I 2 E. 5.7
Fields: ['_enrichment_target', 'authority_role', 'citation', 'court_base', 'court_cases_cited', 'family', 'is_notification_paragraph', 'issue_labels_en', 'language', 'law_codes', 'legal_area', 'matched_terms_multilingual', 'pattern', 'provenance', 'retrieval_text_en', 'statutes_cited', 'structural', 'subfamily', 'summary_en_proxy', 'text_excerpt_original']
Local model path exists: /kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1
config.json OK
tokenizer_config.json OK
Safetensors shards: 2
Tokenizer OK. Probe tokens: 4
Chat template OK. Rendered chars: 121
Working dir writable
Smoke test passed
```

No generation outputs, batch stats, sample tail records, or progress reports are present (Cells 8/9/10/11 were not run in this snapshot).

## Summary

This Kaggle notebook is a production-oriented vLLM offline-inference pipeline that enriches Swiss court-authority paragraph cards (v4 corpus) into a strict 11-field RAG JSON schema using Qwen3-8B-AWQ on dual T4 GPUs. It bakes in Kaggle-specific stability fixes (Triton attention backend, `disable_custom_all_reduce=True`, `enforce_eager=True`, auto memory-shrink, optional TP=2 -> TP=1 fallback) and uses vLLM structured outputs when available, with a retry pass that enables thinking mode and a repair guard for any record failing JSON parse or schema validation. Resume-safe append-only JSONL output and a checkpoint file allow the 2.47 M card full run to span multiple 9 h Kaggle sessions. The captured run only executed environment setup and the smoke test, which resolved the input to a 363,258-line `court_authority_cards_v4_target_cards.jsonl` (not the full 2.47 M v4 artifact) and confirmed both T4s, the local AWQ model, and the tokenizer/chat template were ready. No generation, batch throughput, or output samples were produced in this notebook snapshot.
