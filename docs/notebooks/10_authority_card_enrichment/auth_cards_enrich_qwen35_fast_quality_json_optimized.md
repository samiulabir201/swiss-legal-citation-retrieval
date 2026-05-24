# auth_cards_enrich_qwen35_fast_quality_json_optimized

**Path:** `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen35_fast_quality_json_optimized.ipynb`

## Configuration

- Model: `Qwen/Qwen3.5-35B-A3B` (BF16, no quantization), loaded via vLLM offline inference (`enforce_eager=True`, `enable_prefix_caching=True`, `tensor_parallel_size=1`, `max_num_seqs=128`, `max_num_batched_tokens=16384`).
- Inference params: `GPU_MEMORY_UTIL=0.94`, `MAX_MODEL_LEN=2048`, `BATCH_SIZE=128`, `TEMPERATURE=0.0`, `TOP_P=1.0`, `REPETITION_PENALTY=1.0`, `MAX_TOKENS=256`, `RETRY_MAX_TOKENS=512`, `SMOKE_MAX_TOKENS=1024`, `TEXT_CHARS=900`.
- Structured decoding: prefers `StructuredOutputsParams(json=RAG_SCHEMA)`, falls back to `GuidedDecodingParams(json=...)`. Raises `RuntimeError` if neither is available (no unconstrained generation allowed).
- Env vars set pre-vLLM-import: `VLLM_LOGGING_LEVEL=DEBUG`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`. Stale `VLLM_WORKER_MULTIPROC_METHOD`, `VLLM_MOE_BACKEND`, `VLLM_FLASHINFER_MOE_BACKEND` are popped.
- Qwen "thinking" disabled at chat-template render via `enable_thinking=False` (with fallback to a post-prompt guard if the kwarg is unsupported).
- Paths (Colab): `BASE_DIR=/content/drive/MyDrive/swiss_law`, `ART_DIR=BASE_DIR/artifacts`.
- Input/output mode: `ENRICH_TARGET_CARD_FILE_ONLY=True` -> input `court_authority_cards_v4_target_cards.jsonl`, output `court_authority_cards_rag_targets.jsonl`. Fallback mode uses full `court_authority_cards_v4.jsonl` with `TARGETED_ENRICHMENT=True` driven by `court_authority_cards_v4_enrichment_targets.jsonl`.
- Failure capture files: `court_authority_cards_rag_failed_cards.jsonl`, `..._failed_retry_output.jsonl`, `..._failed_still_failed.jsonl`. Checkpoint: `rag_checkpoint.txt`.
- Quality guardrails: `RETRY_BAD_JSON=False` (fast main pass captures failures for a separate retry-only stage), `ABORT_IF_ERROR_RATE_ABOVE=1.00` (do not abort), `VALIDATE_REFERENCE_GROUNDING=True`, `WRITE_DETERMINISTIC_FALLBACK_FOR_UNTARGETED=True`.
- `LIMIT=0` / `TARGET_LIMIT=0` (process all targeted rows).

## Data

- Primary input: `/content/drive/MyDrive/swiss_law/artifacts/court_authority_cards_v4_target_cards.jsonl` (pre-selected compact subset of v4 authority cards).
- Optional/fallback input: `court_authority_cards_v4.jsonl` (the full ~2.47M-card corpus), gated by `court_authority_cards_v4_enrichment_targets.jsonl` of `line_idx` rows.
- Each card carries fields including `citation`, `court_base`, `legal_area`, `authority_role`, `issue_labels_en`, `law_codes`, `statutes_cited`, `court_cases_cited`, `matched_terms_multilingual`, `text_excerpt_original`, `is_notification_paragraph`, `summary_en_proxy`.
- Outputs append a `rag_enrichment` object onto each card containing 11 RAG_SCHEMA fields plus a `method` tag (e.g. `qwen35_35b_a3b_vllm`, `qwen35_35b_a3b_vllm_retry`, `auto_notification`, `auto_remittal`, `auto_inadmissible`, `auto_dismissed`, `auto_granted`, `auto_short`, `auto_cost`, `deterministic_v4_fallback`).

## Pipeline

1. Environment setup (cells 2-4): detect Colab, query `nvidia-smi`, run a defensive uninstall/reinstall of `PIL/pillow/numpy/scipy/torchvision/vllm/transformers` pinned via `pillow==11.3.0`, `numpy==2.3.5`, `scipy==1.16.3` to a `/tmp/vllm_constraints.txt`, then `uv pip install --system vllm --torch-backend=auto -c ...`. Mounts Google Drive.
2. Configuration (cell 6): builds all paths and prints them; honors `RESET_OUTPUT` to wipe outputs/checkpoints.
3. Schema and prompt (cell 8): defines `COST_PROC_RE`, `REMITTAL_RE`, `INADMISSIBLE_RE`, `DISMISSED_RE`, `GRANTED_RE`, `PARTIAL_RE` deterministic pre-filters. Defines `RAG_SCHEMA` with 11 typed fields (`english_summary`, `legal_topic`, `legal_question`, `legal_rule`, `court_holding`, `factual_context`, `english_legal_concepts`, `search_keywords`, `natural_language_queries`, `paragraph_role` enum, `outcome_signal` enum), all required, with per-field max lengths and per-array `maxItems`. `SYSTEM_PROMPT` instructs deterministic Swiss-legal JSON extraction with grounding-only references.
4. Helpers (cell 10): text/list normalizers, `load_target_line_indices`, `build_user_message` (assembles XML-tagged metadata + paragraph excerpt clipped to `TEXT_CHARS`), `_stub` for deterministic enrichments, `deterministic_disposition` (regex-based remittal/inadmissibility/dismissal/grant detection), `deterministic_fallback` (builds enrichment from labels/statutes/cases for untargeted rows), `auto_classify` (notification, short fragment, cost, disposition shortcuts), `stream_input`, `render_prompt` (applies tokenizer chat template with `enable_thinking=False`), `clean_model_json` (strips think tags and code fences), `validate_and_normalize_enrichment` (clips strings, dedupes/clips arrays, snaps enums), `validate_reference_grounding` (rejects ungrounded `Art. N` / `BGE` / docket references using `BGE_REF_RE`, `DOCKET_REF_RE`, `ART_HEAD_RE`, `ART_GENERATED_RE`, with a tolerance rule for compact `art. 6f` vs source `art. 6 f.`), `parse_output_text`, `append_jsonl`, `stream_failed_cards`.
5. Model load (cell 12): constructs `build_llm_kwargs()`, instantiates `LLM(...)`, gets tokenizer, builds three `SamplingParams` (main, retry, smoke) bound to `RAG_SCHEMA`.
6. Smoke test (cell 14, opt-in `RUN_SMOKE_TEST=True`, `SMOKE_N=8`): streams target input, skips rows the deterministic `auto_classify` already covers, runs `llm.generate` with `smoke_sampling_params`, validates JSON + grounding, prints GREEN/RED. Raises on any failure.
7. Main enrichment (cells 16-17): optional reset, resume from `CHECKPOINT_FILE`, `count_lines(INPUT_FILE)`, then for each row: if not in LLM target set write deterministic fallback (`auto_classify` else `deterministic_fallback`); else try `auto_classify` shortcut, otherwise queue. When queue hits `BATCH_SIZE`, `flush_batch()` calls `llm.generate(prompts, sampling_params)`, parses + validates each output; failures are captured to `FAILED_FILE` (no retry, since `RETRY_BAD_JSON=False`). Per-batch tqdm + throughput/eta logging every 5 batches or on errors. Whole-batch `llm.generate` exceptions are also captured per-row to `FAILED_FILE`. Checkpoint written after each batch and each fallback row.
8. Retry-only pass (cell 18): re-reads `FAILED_FILE` minus already-recovered indices, runs guided JSON with a stronger `retry_guard` system addendum and `RETRY_MAX_TOKENS`, writes successes to `FAILED_RETRY_FILE` and remaining failures to `FAILED_STILL_FILE`. Note: this cell references a `guided` variable that is not defined in this notebook (would `NameError` if run as-is).
9. Verification (cells 20-21): counts `total_out`, `missing_required`, distributions over `paragraph_role`, `outcome_signal`, and `method`; flags `deterministic_fallback_after_parse_failed` and `json_parse_failed` methods; prints 5 bad examples and 3 random LLM-enriched samples with key fields.

## Results

- Cell 2 (env probe): `Running in Colab: True`; GPU `NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB`.
- Cell 3 (installer): completed; `Pillow OK: 11.3.0`, `NumPy OK: 2.3.5`, `SciPy OK: 1.16.3`, `vLLM import OK`.
- Cell 4: `Mounted at /content/drive`.
- Cell 6 (config print): `BASE_DIR: /content/drive/MyDrive/swiss_law`; `INPUT_FILE: .../artifacts/court_authority_cards_v4_target_cards.jsonl`; `OUTPUT_FILE: .../court_authority_cards_rag_targets.jsonl`; `MODEL_ID: Qwen/Qwen3.5-35B-A3B`; `BATCH_SIZE=128, MAX_TOKENS=256, RETRY_MAX_TOKENS=512, SMOKE_MAX_TOKENS=1024`; `ENRICH_TARGET_CARD_FILE_ONLY=True, TARGETED_ENRICHMENT=True, TARGET_LIMIT=0, WRITE_FALLBACK=True`.
- Cell 8: schema and required field lists printed (11 fields each).
- Cell 10: `Helpers loaded.`
- Cell 12 (load model): printed final vLLM kwargs; vLLM resolved architecture as `Qwen3_5MoeForConditionalGeneration`, max model len 2048, chunked prefill on, MoE prepare/finalize via `MoEPrepareAndFinalizeNoDPEPModular`, attention backend `FLASH_ATTN` (FlashAttention v2), MoE backend `FlashInfer CUTLASS Unquantized MoE`. Weight download took 201.16 s (66.97 GiB); safetensors load took 14.03 s; model loading took 64.69 GiB GPU memory and 216.92 s. Then failed with:
  - `RuntimeError: No supported CUDA architectures found for major versions [12].`
  - Preceded by `WARNING:flashinfer.compilation_context:Failed to get device capability: SM 12.x requires CUDA >= 12.9.` (twice).
  - Traceback origin: `LLM(**build_llm_kwargs())` in cell 12, going through `vllm/entrypoints/llm.py` -> `LLMEngine.from_engine_args` -> `vllm/v1/engine/llm_engine.py`.
- Cells 14, 16, 17, 18, 20, 21, 22: have `execution_count=None` and zero outputs (never run because cell 12 raised).

## Summary

This notebook is a vLLM-based batch enrichment job that adds an 11-field English RAG schema (summary, topic, question, rule, holding, factual context, concepts, keywords, NL queries, paragraph_role enum, outcome_signal enum) to Swiss Federal Tribunal authority cards using `Qwen/Qwen3.5-35B-A3B` with structured JSON decoding, deterministic temperature, prefix caching, and `enforce_eager`. A regex-driven pre-classifier (notifications, costs, dispositions, short fragments) short-circuits the LLM for trivial paragraphs, and a separate `deterministic_fallback` writes labels-only enrichments for untargeted rows so the output covers the full input. Reference grounding is enforced post-hoc by parsing `Art. N`, `BGE`, and docket-number tokens out of the model JSON and rejecting any that do not appear in the source metadata/paragraph. The fast main pass captures any JSON/grounding failures to a side file rather than retrying inline; a follow-up retry-only cell re-runs them with stronger guard text (note: that cell references an undefined `guided` symbol). In this run on an RTX PRO 6000 Blackwell (SM 12.x) the model weights loaded successfully but the engine then aborted with `RuntimeError: No supported CUDA architectures found for major versions [12]` because FlashInfer requires CUDA >= 12.9, so cells 14 onward (smoke test, main enrichment, retry, verification) did not execute and produced no output.
