# kaggle_qwen3_8b_awq_vllm_text_to_json_t4_fixed

**Path:** e:\swiss_citation_extraction\notebooks\13_misc_kaggle_and_utilities\kaggle_qwen3_8b_awq_vllm_text_to_json_t4_fixed.ipynb

## Configuration

- Runtime: Kaggle with 2x T4 GPUs.
- Install: `pip install -U vllm transformers accelerate safetensors tqdm`.
- Environment vars: `TOKENIZERS_PARALLELISM=false`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7`.
- Model source: local Kaggle path `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1`; HF fallback `Qwen/Qwen3-8B-AWQ`.
- vLLM settings: `quantization="awq"`, `dtype="float16"`, `max_model_len=4096`, `gpu_memory_utilization=0.90`, `tensor_parallel_size=2`, `max_num_seqs=128`, `enable_prefix_caching=True`, `disable_log_stats=True`.
- Kaggle 2xT4 reliability fix: `disable_custom_all_reduce=True`, `auto_fallback_to_single_gpu=True` (falls back to TP=1, `max_num_seqs=96`, `batch_size=96` if initial load fails).
- Throughput: `batch_size=128`, `text_chars=1600`, `max_new_tokens=224`, `retry_max_new_tokens=384`.
- Sampling: `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.03`, stop tokens `["<|im_end|>", "</s>"]`.
- Thinking modes: `main_enable_thinking=False`, `retry_enable_thinking=True`, `retry_invalid=True`, `use_structured_outputs=True`.
- Run controls: `limit=500` (test) or `0` (full), `reset_output=False`, `flush_every_batches=1`.
- Input file (resolved via candidate search): `/kaggle/input/swiss-court-authority-cards-rag-targets/court_authority_cards_v4_target_cards.jsonl`.
- Outputs in `/kaggle/working/`:
  - `court_authority_cards_rag_targets_qwen3_8b_vllm.jsonl`
  - `court_authority_cards_rag_targets_qwen3_8b_vllm_failed.jsonl`
  - `rag_targets_qwen3_8b_vllm_checkpoint.txt`

## Data

- Input: JSONL of court-authority cards (`court_authority_cards_v4_target_cards.jsonl`) from the `swiss-court-authority-cards-rag-targets` Kaggle dataset.
- Per-card fields read for prompt metadata: `citation`, `language`, `court_base`, `authority_role`, `legal_area`, `family`, `subfamily`, `issue_labels_en`, `law_codes`, `statutes_cited`, `court_cases_cited`, `summary_en_proxy`, `retrieval_text_en`, `structural`, plus original paragraph `text_excerpt_original` (truncated to `text_chars`).
- Smoke test reads first line, prints citation, fields, file size in GB, line count; verifies model `config.json`/`tokenizer_config.json`/`*.safetensors` shards if local; loads tokenizer; renders a chat template probe; checks working dir is writable.

## Pipeline

1. Install vLLM + HF stack (Cell 1).
2. Import torch/transformers/tqdm, set CUDA env (Cell 2), print torch/CUDA/GPU info.
3. Build `Config` dataclass; resolve input file via candidate paths and glob, resolve model from local Kaggle path or HF fallback (Cell 3).
4. Smoke test: existence of input, line count, first record, model artifacts, tokenizer, chat template, write-test in working dir (Cell 4).
5. Define schema and prompt artifacts (Cell 5):
   - `RAG_SCHEMA_HINT` with 11 required keys: `english_summary`, `legal_topic`, `legal_question`, `legal_rule`, `court_holding`, `factual_context`, `english_legal_concepts` (list), `search_keywords` (list, max 12), `natural_language_queries` (list, max 5), `paragraph_role` (enum of 8), `outcome_signal` (enum of 6).
   - `JSON_SCHEMA` (JSON-Schema dict with enums and max-item caps) for vLLM structured outputs.
   - `SYSTEM_PROMPT`: deterministic Swiss legal text-to-JSON extractor; English RAG terminology; no inventions; empty string/array/`none` for unsupported fields; no markdown/CoT.
   - `REPAIR_GUARD` appended on retry to force compact JSON only.
   - `build_user_prompt`: serializes metadata JSON + paragraph and demands JSON-only output.
   - Parsing helpers: `strip_think_blocks` (removes `<think>...</think>` and code fences), `extract_first_json_object` (balanced-brace scan), `normalize_and_validate` (per-field length caps, list dedup, enum clamping with `paragraph_role` default `reasoning` and `outcome_signal` default `none`, `empty_core_fields` and `empty_retrieval_terms` checks), `parse_validate_raw`.
6. `VLLMJsonGenerator` class (Cell 6): loads tokenizer + vLLM `LLM`; attempts `StructuredOutputsParams(json=JSON_SCHEMA)` and falls back to parser-only if unavailable; renders chat with `enable_thinking` flag; `generate_raw` builds prompts and calls `llm.generate` with `use_tqdm=False`. Auto-fallback to TP=1 on initial load failure with cache cleanup.
7. Main loop (Cell 7):
   - `iter_cards` streams JSONL with start offset and stop-before bound.
   - `run_one_batch` runs fast pass, parses+validates, collects retry indices, runs retry pass with thinking enabled and repair guard, prefers retry output when it fully validates or recovers a previously unparseable response.
   - `make_output_record` clones the card and adds `_source_line_idx`, `_rag_generation` (model, engine, source `main`/`retry`, elapsed_s, errors), `rag_targets_qwen3_8b_awq` payload, and `_raw_model_output` (truncated to 4000 chars) when errors occurred.
   - `append_jsonl` writes with `fsync`; checkpoint file holds the next line index.
   - `main` orchestrates: line count, optional reset, checkpoint load, tqdm bar, per-batch flush with stats (`batch_rate`, avg cards/s, ETA hours), final summary.
8. Cell 8 runs a 500-card quality/throughput test with TP=2 and the Kaggle stability flags.
9. Cell 9 prints the last 3 output records (citation, `_rag_generation`, and the generated `rag_targets_qwen3_8b_awq` JSON).
10. Cell 10 is the gated full production run (commented out).
11. Cell 11 lists optional tuning presets: higher-accuracy (larger context/tokens, batch 96), faster (batch 192, no retry thinking), TP=1 fallback (batch 96), Kaggle 2xT4 stable preset (TP=2 + `disable_custom_all_reduce=True`), and an `awq_marlin` speed experiment after a successful test.

## Results

All cells in the notebook have no captured execution output (no stdout, results, or errors recorded for any cell).

## Summary

Kaggle 2xT4 notebook that runs Qwen3-8B-AWQ under vLLM offline inference to enrich Swiss court-authority paragraph cards with an 11-field English RAG-target JSON (summary, topic, question, rule, holding, facts, concepts, keywords, NL queries, paragraph_role enum, outcome_signal enum). The pipeline uses chat-template prompting with optional `enable_thinking`, vLLM structured outputs against a JSON Schema, balanced-brace JSON extraction with `<think>` stripping, schema/enum normalization with retry-on-invalid using a repair guard, and resume-safe JSONL checkpointing (success + failure files + line-index checkpoint). It specifically targets the Kaggle 2xT4 custom-all-reduce startup crash by defaulting `disable_custom_all_reduce=True` and providing an automatic fallback to `tensor_parallel_size=1` with reduced `batch_size`/`max_num_seqs`. Cell 8 is configured for a 500-card test run with batch 128 before flipping to the full run, and Cell 11 supplies higher-accuracy, faster, TP=1, and `awq_marlin` tuning presets. The notebook ships with no captured execution outputs.
