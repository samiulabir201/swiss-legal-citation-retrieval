# auth_cards_enrich_qwen35_safe_no_vllm

**Path:** `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen35_safe_no_vllm.ipynb`

## Configuration

- Engine: `transformers_safe` (HuggingFace Transformers with `attn_implementation="sdpa"`, `torch_dtype=torch.bfloat16`, `device_map="auto"`, `low_cpu_mem_usage=True`)
- Model: `Qwen/Qwen3.5-35B-A3B` (`MODEL_ID`)
- Rationale: avoids the repeated vLLM / FlashInfer `CUTLASS Unquantized MoE` startup crash on Colab SM 12.x GPUs.
- Environment cleanup: removes `VLLM_ATTENTION_BACKEND`, `VLLM_MOE_BACKEND`, `VLLM_FLASHINFER_MOE_BACKEND`, `VLLM_WORKER_MULTIPROC_METHOD` from `os.environ`.
- Dependencies (installed via `pip`): `transformers`, `accelerate`, `safetensors`, `jsonschema`, `tqdm`.
- Paths:
  - `INPUT_FILE`: `/content/court_authority_cards.jsonl`
  - `OUT_DIR`: `/content/drive/MyDrive/qwen35_card_enrichment_output`
  - `OUTPUT_FILE`: `OUT_DIR / enriched_cards.jsonl`
  - `FAILED_FILE`: `OUT_DIR / failed_cards.jsonl`
  - `FAILED_RETRY_FILE`: `OUT_DIR / failed_cards_retry_output.jsonl`
  - `FAILED_STILL_FILE`: `OUT_DIR / failed_cards_still_failed.jsonl`
  - `CHECKPOINT_FILE`: `OUT_DIR / checkpoint_processed_line_idxs.json`
- Generation params:
  - `BATCH_SIZE = 1`
  - `MAX_NEW_TOKENS = 256`, `RETRY_MAX_NEW_TOKENS = 512`
  - `TEXT_CHARS = 900` (card text truncation)
  - `TEMPERATURE = 0.0`, `TOP_P = 1.0`, `DO_SAMPLE = False`
  - Tokenizer max length: 2048 (input encoding `truncation=True, max_length=2048`).
- Flow flags:
  - `RESET_OUTPUT = False` (preserve existing outputs/checkpoint)
  - `INLINE_RETRY = False` (failed rows go to `FAILED_FILE` for later retry)
  - `TARGET_LIMIT = 100` (smoke/testing cap; set `None` for full run)

## Data

- Input: JSONL at `INPUT_FILE` (`/content/court_authority_cards.jsonl`), one card object per line. The notebook does not specify a fixed schema; it extracts text via `extract_card_text` from the first present of: `text`, `content`, `paragraph`, `body`, `quote`, `passage`, `source_text`, `card_text`, `full_text`, `raw_text`. Falls back to compact JSON of the card if none are present. Truncated to `TEXT_CHARS = 900`.
- Card identifier resolved from first present of `id`, `card_id`, `_id`, `uuid`, `source_id`.
- Source metadata hint appended from any of: `case_name`, `court`, `jurisdiction`, `year`, `citation`, `source`, `url`.
- Smoke-test card (cell 11): `{"id": "smoke-test", "text": "The court held that the claimant failed to prove causation and dismissed the appeal."}`.
- Output JSONL schema (`RAG_SCHEMA`, strict, no additional properties):
  - `english_summary` (string, maxLength 220)
  - `legal_topic` (string, maxLength 100)
  - `legal_question` (string, maxLength 180)
  - `legal_rule` (string, maxLength 220)
  - `court_holding` (string, maxLength 180)
  - `factual_context` (string, maxLength 180)
  - `english_legal_concepts` (array of string maxLength 60, maxItems 5)
  - `search_keywords` (array of string maxLength 60, maxItems 6)
  - `natural_language_queries` (array of string maxLength 140, maxItems 2)

## Pipeline

1. Cell 0 — Environment preflight: print Python/torch/CUDA info, list GPUs and VRAM, pop noisy vLLM env vars.
2. Cell 1 — Install dependencies (`transformers accelerate safetensors jsonschema tqdm`).
3. Cell 2 — Config (paths, model, generation params, `TARGET_LIMIT=100`).
4. Cell 3 — Optional reset of outputs/checkpoint when `RESET_OUTPUT=True`.
5. Cell 4 — Define output JSON schema `RAG_SCHEMA`.
6. Cell 5 — JSONL/checkpoint helpers: `append_jsonl`, `stream_jsonl`, `load_checkpoint`, `save_checkpoint`, `chunked`.
7. Cell 6 — Card text extraction: `get_first_present`, `card_identifier`, `extract_card_text`, `make_failure_obj`.
8. Cell 7 — Prompt renderer: `SYSTEM_INSTRUCTION` plus schema, source metadata, card text; instructs model to return one JSON object only.
9. Cell 8 — JSON parsing: `extract_json_object` (strips ```` ```json ```` fences, tries direct then first `{...}` substring), `normalize_enriched` (coerces types and truncates to schema limits), `parse_model_output` (extract + normalize + `jsonschema.validate`).
10. Cell 9 — Safe Transformers engine loader: builds `tokenizer` and `model` with SDPA attention, bf16, auto device map. Sets `ENGINE_READY` based on success and never re-raises — on failure prints recommended next actions instead.
11. Cell 10 — Generation helpers: `generate_texts_transformers` (batched tokenize, pad/truncate to 2048, greedy decode, slice off the prompt before decoding) and `generate_batch` wrapper (only `transformers_safe` enabled).
12. Cell 11 — Smoke test: render prompt for `smoke-test` card, generate, parse and print parsed JSON; parsing failure is logged but does not raise.
13. Cell 12 — Main pass:
    - Loads `processed` checkpoint set.
    - Streams `INPUT_FILE` and skips already-processed line indices.
    - Buffers rows up to `BATCH_SIZE`, generates with `MAX_NEW_TOKENS`.
    - On batch-generation exception: appends each row to `FAILED_FILE` with `stage="batch_generation_failed"` and marks processed.
    - On per-row parse/validate exception: appends to `FAILED_FILE` with `stage="parse_or_validation_failed"`; successes appended to `OUTPUT_FILE` with `status="ok"`, `engine`, `model_id`, `card`, `enriched`, `raw_output`.
    - Saves checkpoint after every batch; prints `processed/ok/failed/rate` line. Stops at `TARGET_LIMIT` if set. Flushes any trailing partial batch.
14. Cell 13 — Retry-only pass: streams `FAILED_FILE`, regenerates with `RETRY_MAX_NEW_TOKENS=512`; recovered rows go to `FAILED_RETRY_FILE` (`status="recovered"`), still-failed rows to `FAILED_STILL_FILE` with both current and previous error and stage labels (`retry_batch_generation_failed` or `retry_parse_or_validation_failed`).

## Results

(Empty — the notebook contains no executed cell outputs.)

## Summary

This is a Colab-oriented enrichment notebook that replaces vLLM with a Transformers-only path for `Qwen/Qwen3.5-35B-A3B` to sidestep a FlashInfer MoE crash on SM 12.x GPUs. It enriches court authority cards into a strict 9-field RAG schema (English summary, legal topic/question/rule, court holding, factual context, concepts, keywords, natural-language queries), validated via `jsonschema`. The control flow is failure-tolerant: model load is wrapped to set `ENGINE_READY=False` rather than raise, per-batch and per-row exceptions are captured to `failed_cards.jsonl`, and a separate retry pass writes recovered rows to `failed_cards_retry_output.jsonl` and persistent failures to `failed_cards_still_failed.jsonl`. A line-index checkpoint enables resumable runs without re-processing prior successes or failures. The notebook in its committed state has no executed outputs and uses `TARGET_LIMIT=100`, `BATCH_SIZE=1`, greedy decoding (`do_sample=False`), bf16 SDPA attention.
