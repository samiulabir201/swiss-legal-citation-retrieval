# auth_cards_enrich_qwen35_updated_fast_json

**Path:** `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen35_updated_fast_json.ipynb`

Batch-enriches `court_authority_cards_v4.jsonl` with English legal RAG metadata using `Qwen/Qwen3.5-35B-A3B` via vLLM offline inference. Targets a Google Compute Engine / Colab GPU setup with an RTX PRO 6000 Blackwell-class 96 GB GPU. Uses deterministic schema-constrained JSON generation, disables Qwen thinking at chat-template render time, forces Triton MoE kernels (to avoid FlashInfer CUTLASS SM120 JIT failures on Blackwell), uses checkpointed JSONL append output, keeps deterministic auto-classification for trivial notification/cost/short paragraphs, and adds batch throughput logging, JSON validation, retry, and high-error-rate abort protection.

## Configuration

- **Model:** `Qwen/Qwen3.5-35B-A3B` (BF16, official post-trained MoE checkpoint), `QUANTIZATION = None`
- **vLLM runtime knobs:**
  - `GPU_MEMORY_UTIL = 0.90`
  - `MAX_MODEL_LEN = 4096`
  - `TENSOR_PARALLEL = 1`
  - `enforce_eager = True`
  - `enable_prefix_caching = True`
  - `kernel_config = KernelConfig(moe_backend='triton')` (Blackwell fix)
  - `max_num_seqs = max(BATCH_SIZE, 64)`
  - `limit_mm_per_prompt = {'image': 0, 'video': 0}`
  - `language_model_only = True` (when supported)
- **Sampling / batching:**
  - `BATCH_SIZE = 32`
  - `TEMPERATURE = 0.0`, `TOP_P = 1.0`
  - `MAX_TOKENS = 384` (first pass), `RETRY_MAX_TOKENS = 768` (retries)
  - `TEXT_CHARS = 1800` (truncate source paragraph)
- **Quality guardrails:**
  - `RETRY_BAD_JSON = True`
  - `ABORT_IF_ERROR_RATE_ABOVE = 0.10`
  - `ERROR_RATE_CHECK_AFTER = 200`
- **Environment variables (set before vLLM init):**
  - `VLLM_LOGGING_LEVEL=DEBUG` (avoids Jupyter `sys.stdout.fileno()` crash)
  - `VLLM_ENABLE_V1_MULTIPROCESSING=0` (keep engine in-process for tracebacks)
  - Pops stale `VLLM_WORKER_MULTIPROC_METHOD`, `VLLM_MOE_BACKEND`, `VLLM_FLASHINFER_MOE_BACKEND`
- **Paths (Colab):**
  - `BASE_DIR = /content/drive/MyDrive/swiss_law` (else `..` resolved)
  - `INPUT_FILE = {BASE_DIR}/data/court_authority_cards_v4.jsonl`
  - `OUTPUT_FILE = {BASE_DIR}/artifacts/court_authority_cards_rag.jsonl`
  - `CHECKPOINT_FILE = {BASE_DIR}/artifacts/rag_checkpoint.txt`
- **Other controls:** `LIMIT = 0` (process all), `RESET_OUTPUT = False`
- **Install path (Colab only):** uninstalls + hard-deletes stale `PIL/pillow/numpy/scipy/torchvision/vllm/transformers`, reinstalls with constraints (`pillow==11.3.0`, `numpy==2.3.5`, `scipy==1.16.3`), then `uv pip install vllm --torch-backend=auto`.
- **Structured decoding:** prefers `StructuredOutputsParams(json=schema)`; falls back to `GuidedDecodingParams(json=schema)`. Raises if neither is available (no unconstrained generation).

## Data

- **Input:** `court_authority_cards_v4.jsonl` — one Swiss Federal Tribunal paragraph per line. Card fields consumed: `text_excerpt_original` (truncated to 1800 chars), `citation`, `legal_area`, `issue_labels_en` (first 8), `is_notification_paragraph`.
- **Output:** `court_authority_cards_rag.jsonl` — input cards with a new `rag_enrichment` object appended (JSONL append + checkpoint at `rag_checkpoint.txt`).
- **Auto-classification short-circuits (no LLM call):**
  - `auto_notification` — if `is_notification_paragraph` is truthy.
  - `auto_short` — if `len(text) < 50`.
  - `auto_cost` — if `COST_PROC_RE` matches the first 400 chars (matches DE/FR/IT terms for court costs, procedural fees, legal aid, remittal, and `^N. N. ...Fr. N`-style cost lines).
- **JSON schema (`RAG_SCHEMA`) — required fields:**
  - `english_summary`, `legal_topic`, `legal_question`, `legal_rule`, `court_holding`, `factual_context` (strings)
  - `english_legal_concepts` (array, maxItems 8)
  - `search_keywords` (array, maxItems 10)
  - `natural_language_queries` (array, maxItems 5)
  - `paragraph_role` enum: `holding | reasoning | background | cost | procedural | disposition | standard_of_review | obiter`
  - `outcome_signal` enum: `granted | dismissed | inadmissible | remitted | partial | none`
  - `method` is appended in Python after parsing/validation.

## Pipeline

1. **Environment setup** (`detect-colab`, `install-deps`): detects Colab, runs `nvidia-smi`, conditionally rebuilds the Python env via uv + pip with pinned constraints, then sanity-imports `PIL`, `numpy`, `scipy`, `numpy._core.umath._center`, and `vllm`.
2. **Mount Drive** (Colab only).
3. **Configuration** (`config`): builds paths, sets model / batching / guardrail knobs, optionally resets output + checkpoint.
4. **Schema + prompts** (`schema-and-prompts`): defines `COST_PROC_RE`, `RAG_SCHEMA`, and `SYSTEM_PROMPT` (deterministic Swiss legal JSON extractor; returns one JSON object matching the schema; no markdown, no chain-of-thought, no continuation).
5. **Helpers** (`helpers`):
   - `build_user_message` — wraps the truncated paragraph + citation + legal area + existing labels in `<record>...<paragraph_original_language>...</record>` tags.
   - `auto_classify` — returns a deterministic stub for notification / short / cost paragraphs, else `None`.
   - `stream_input`, `count_lines` — streaming JSONL reader with offset and line counter.
   - `render_prompt` — applies the tokenizer chat template with `enable_thinking=False` (with a TypeError fallback that appends an inline "no chain-of-thought" reminder).
   - `clean_model_json` — strips `</think>` prefix, fenced ``` markers, and trims to the outermost `{...}` if needed.
   - `validate_and_normalize_enrichment` — enforces required keys, normalizes strings/arrays/enums, and stamps `method`.
   - `parse_output_text` — cleans, `json.loads`, then validates+normalizes.
6. **Load model** (`load-model`): `LLM(...)` with the kwargs above; resolves structured-output API (`StructuredOutputsParams` first, else `GuidedDecodingParams`); builds `sampling_params` (max_tokens=384) and `retry_sampling_params` (max_tokens=768).
7. **Optional smoke test** (`RUN_SMOKE_TEST = False` by default): pulls 8 non-auto cards, generates, parses each, raises `RuntimeError` if any fail.
8. **Run enrichment** (`run-enrichment`):
   - Reads resume offset from `CHECKPOINT_FILE`, counts total lines, opens output in append mode, starts a tqdm bar.
   - For each input line: applies `auto_classify`; if non-None, writes the stub and advances. Otherwise queues into `pending`.
   - When `pending` reaches `BATCH_SIZE`, calls `flush_batch`: renders prompts, runs `llm.generate` with the constrained `sampling_params`, parses each output; on parse/validation failure with `RETRY_BAD_JSON=True`, retries that card with an "invalid generation" guard and `retry_sampling_params`; if retry still fails, writes a `json_parse_failed` stub carrying `parse_error` and `raw_output` (first 400 chars).
   - After each flush: writes one record per card to output, flushes, updates checkpoint to `last_line_idx + 1`, advances pbar.
   - **Throughput logging:** every batch 1 and every 5th batch (or any batch with errors) prints `batch_time`, `batch_rate`, `avg_rate`, `eta`, error counts.
   - **High-error abort:** after `llm_attempts >= 200`, raises `RuntimeError` if cumulative JSON error rate exceeds 10%.
9. **Verify output** (`stats`): counts `paragraph_role`, `outcome_signal`, `method` distributions across the output file; reports rows missing required fields or with `method == json_parse_failed`; prints up to 5 bad examples.
10. **Sample** (`sample`): loads all cards whose `method` starts with `qwen35` (and is not `json_parse_failed`), prints `random.sample(..., 3)` of them with the main RAG fields.

## Results

No cell outputs are captured in the notebook (all code cells have empty outputs); the notebook has not been executed in this saved copy. No printed banners, GPU info, install logs, throughput batch logs, role/outcome/method distributions, or sample-card prints are present.

## Summary

This notebook is the LLM-enrichment stage for Swiss authority cards, taking `court_authority_cards_v4.jsonl` and emitting `court_authority_cards_rag.jsonl` with an English-legal `rag_enrichment` object per card. It runs Qwen3.5-35B-A3B in vLLM offline mode with structured JSON decoding, Triton MoE kernels (Blackwell workaround), and `enable_thinking=False` to prevent `<think>` tokens from corrupting guided decoding. Trivial paragraphs (notifications, very short text, court-cost / legal-aid / remittal templates) bypass the LLM via deterministic regex auto-classification, while substantive paragraphs are batched at size 32, parsed with strict schema validation, and retried once on failure with a larger token budget before being marked `json_parse_failed`. The pipeline is checkpointed line-by-line, supports resume, and aborts after 200 attempts if the JSON error rate exceeds 10%. The saved notebook contains no execution outputs, so no quantitative results (rates, distributions, samples) are available.
