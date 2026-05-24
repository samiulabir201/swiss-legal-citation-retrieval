# auth_cards_enrich_qwen35_fast_quality_json_optimized_pre_failure_capture

**Path:** e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen35_fast_quality_json_optimized_pre_failure_capture.ipynb

## Configuration

- Environment: Google Colab on NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB). Python 3.12.13.
- Install pins: `pillow==11.3.0`, `numpy==2.3.5`, `scipy==1.16.3`. vLLM installed via `uv pip install --system vllm --torch-backend=auto` with constraints file `/tmp/vllm_constraints.txt`.
- Drive mount: `/content/drive`; base dir `BASE_DIR = /content/drive/MyDrive/swiss_law`.
- Subdirectories: `DATA_DIR`, `INSIGHTS_DIR`, `ART_DIR = BASE_DIR/artifacts`, `SCRIPT_DIR`.
- Model: `MODEL_ID = 'Qwen/Qwen3.5-35B-A3B'`, `QUANTIZATION = None`, `dtype='bfloat16'`.
- vLLM runtime: `GPU_MEMORY_UTIL=0.94`, `MAX_MODEL_LEN=2048`, `TENSOR_PARALLEL=1`, `ENABLE_PREFIX_CACHE=True`, `max_num_seqs=max(BATCH_SIZE,256)`, `max_num_batched_tokens=32768`, `kernel_config=KernelConfig(moe_backend='triton')`.
- Sampling: `BATCH_SIZE=128`, `TEMPERATURE=0.0`, `TOP_P=1.0`, `REPETITION_PENALTY=1.03`, `MAX_TOKENS=320`, `RETRY_MAX_TOKENS=512`, `SMOKE_MAX_TOKENS=1024`, `TEXT_CHARS=900` (paragraph truncation).
- Env vars set before vLLM init: `VLLM_LOGGING_LEVEL=DEBUG`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`; explicitly pops `VLLM_WORKER_MULTIPROC_METHOD`, `VLLM_MOE_BACKEND`, `VLLM_FLASHINFER_MOE_BACKEND`.
- Structured decoding: prefers `StructuredOutputsParams(json=schema)`, falls back to `GuidedDecodingParams(json=schema)`; raises `RuntimeError` if neither is available (no unconstrained generation).
- Targeted enrichment flags: `ENRICH_TARGET_CARD_FILE_ONLY=True`, `TARGETED_ENRICHMENT=True`, `WRITE_DETERMINISTIC_FALLBACK_FOR_UNTARGETED=True`, `TARGET_LIMIT=0`, `LIMIT=0` (full run).
- Quality guards: `RETRY_BAD_JSON=True`, `ABORT_IF_ERROR_RATE_ABOVE=0.03`, `ERROR_RATE_CHECK_AFTER=200`, `VALIDATE_REFERENCE_GROUNDING=True`.
- Chat template: thinking explicitly disabled via `enable_thinking=False`; falls back to appending a JSON-only instruction if `TypeError`.
- Reset flag `RESET_OUTPUT=True` deletes `OUTPUT_FILE` and `CHECKPOINT_FILE` before run.

## Data

- Input (fast path): `INPUT_FILE = ART_DIR / 'court_authority_cards_v4_target_cards.jsonl'` (the compact target file; avoids scanning the 2.47M v4 cards).
- Fallback target index: `TARGET_FILE = ART_DIR / 'court_authority_cards_v4_enrichment_targets.jsonl'` (only used if `ENRICH_TARGET_CARD_FILE_ONLY=False`).
- Full corpus (fallback only): `ART_DIR / 'court_authority_cards_v4.jsonl'`.
- Output: `OUTPUT_FILE = ART_DIR / 'court_authority_cards_rag_targets.jsonl'` (target-only path; the alternative is `court_authority_cards_rag.jsonl`).
- Checkpoint: `CHECKPOINT_FILE = ART_DIR / 'rag_checkpoint.txt'` (resumable; stores next line index).
- Per-card source fields consumed: `text_excerpt_original`, `citation`, `court_base`, `legal_area`, `authority_role`, `issue_labels_en`, `law_codes`, `statutes_cited`, `court_cases_cited`, `matched_terms_multilingual`, `summary_en_proxy`, `is_notification_paragraph`.

## Pipeline

1. **Environment setup** (cells `detect-colab`, `install-deps`): detect Colab; uninstall and re-install `PIL/pillow/numpy/scipy/torchvision/vllm/transformers` with constraints; verify import of `vllm`, `numpy._core.umath._center`.
2. **Drive mount** (`mount-drive`): `google.colab.drive.mount('/content/drive')`.
3. **Configuration** (`config`): set paths, model, sampling, target flags; create dirs; conditionally reset output and checkpoint.
4. **Schema + prompts** (`schema-and-prompts`): defines deterministic regexes (`COST_PROC_RE`, `REMITTAL_RE`, `INADMISSIBLE_RE`, `DISMISSED_RE`, `GRANTED_RE`, `PARTIAL_RE`); defines `RAG_SCHEMA` with 11 required fields: `english_summary`, `legal_topic`, `legal_question`, `legal_rule`, `court_holding`, `factual_context`, `english_legal_concepts`, `search_keywords`, `natural_language_queries`, `paragraph_role` (enum: holding/reasoning/background/cost/procedural/disposition/standard_of_review/obiter), `outcome_signal` (enum: granted/dismissed/inadmissible/remitted/partial/none); defines `SYSTEM_PROMPT` enforcing grounded English JSON extraction.
5. **Helpers** (`helpers`): `build_user_message` wraps `<record><metadata>…</metadata><paragraph_original_language>…</paragraph_original_language></record>` (text capped at `TEXT_CHARS=900`); `auto_classify` handles notifications, disposition signals, very short fragments, and cost paragraphs without LLM; `deterministic_disposition` matches remittal/inadmissible/dismissed/granted/partial; `deterministic_fallback` synthesizes English fields from labels/area/statutes/cases/terms; `render_prompt` applies chat template with `enable_thinking=False`; `clean_model_json` strips `</think>`, ```json fences, and isolates the outermost `{…}`; `validate_and_normalize_enrichment` enforces field-length clipping, list dedup/cap, and enum membership; `validate_reference_grounding` checks generated `Art.`/`BGE`/docket references against the metadata+paragraph blob (with tolerance for `Art. N f.` ↔ `Art. Nf`).
6. **Model load** (`load-model`): builds vLLM `LLM(**kwargs)` (filtered by `inspect.signature`), obtains tokenizer, builds `sampling_params`, `retry_sampling_params`, and `smoke_sampling_params` via `make_sampling_params`.
7. **Smoke test** (`Q7LmYo7Z7Fv5`): `RUN_SMOKE_TEST=True`, `SMOKE_N=8`; selects 8 non-auto-classifiable target cards, calls `llm.generate` with `smoke_sampling_params`, parses/validates each, prints GREEN/RED, raises `RuntimeError` on failure.
8. **Reset** (`y-RnVQp9AXOe`): re-deletes `OUTPUT_FILE` and `CHECKPOINT_FILE` for a clean production run.
9. **Run enrichment** (`run-enrichment`): streams `INPUT_FILE` from checkpoint; for each card, decides `should_llm` (always true under `ENRICH_TARGET_CARD_FILE_ONLY`); applies `auto_classify` short-circuit, else queues for batch; `flush_batch` generates with `sampling_params`, parses, retries once with `retry_sampling_params` and a guard suffix on parse/grounding failure; on retry failure writes `deterministic_fallback_after_parse_failed` with `parse_error` + clipped `raw_output`; appends JSONL, flushes, updates checkpoint after each batch; logs `[batch N]` lines with batch time, rates, ETA, retries, errors, fallbacks, autos; raises `RuntimeError` if `json_errors/llm_attempts > ABORT_IF_ERROR_RATE_ABOVE` once `llm_attempts >= ERROR_RATE_CHECK_AFTER`.
10. **Stats** (`stats`): reads `OUTPUT_FILE`, counts `paragraph_role`, `outcome_signal`, and `method` distributions; lists up to 10 cards with missing required fields or `deterministic_fallback_after_parse_failed`; prints GREEN/RED verdict.
11. **Sample inspection** (`sample`): random-samples 3 cards whose `rag_enrichment.method` contains `qwen35`, prints citation, target line idx, role, outcome, topic, question, rule, holding, summary, concepts, keywords, NL queries.

## Results

Verbatim cell outputs captured:

- `detect-colab`:
  - `Running in Colab: True`
  - `GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB`
- `install-deps`:
  - `Python executable: /usr/bin/python3`
  - `Python version: 3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0]`
  - pip uninstall/install/uv commands echoed.
  - `Pillow OK: 11.3.0 /usr/local/lib/python3.12/dist-packages/PIL/__init__.py`
  - `NumPy OK: 2.3.5 /usr/local/lib/python3.12/dist-packages/numpy/__init__.py`
  - `SciPy OK: 1.16.3 /usr/local/lib/python3.12/dist-packages/scipy/__init__.py`
  - `NumPy internal symbol OK`
  - `vLLM import OK`
- `mount-drive`: `Mounted at /content/drive`.
- `config`, `schema-and-prompts`, `helpers`: no outputs captured in the notebook JSON for these print statements.
- `load-model`: outputs flagged as too large to inline (the notebook states they were truncated; only the directive `Use Bash with: cat <notebook_path> | jq '.cells[12].outputs'` is recorded).
- `Q7LmYo7Z7Fv5` (smoke test): no captured outputs.
- `y-RnVQp9AXOe` (reset):
  - `Deleted old output: /content/drive/MyDrive/swiss_law/artifacts/court_authority_cards_rag.jsonl`
  - `Deleted old checkpoint: /content/drive/MyDrive/swiss_law/artifacts/rag_checkpoint.txt`
  - `Clean reset done. Now rerun the enrichment cell from line 0.`
- `run-enrichment`: no captured outputs (cell did not produce stored results in this snapshot — consistent with the notebook filename `pre_failure_capture`).
- `stats`: no captured outputs.
- `sample`: no captured outputs.
- Final empty cell `EIj_PYsj_5hJ`: empty.

## Summary

This notebook is the Qwen3.5-35B-A3B + vLLM authority-card enrichment pipeline, configured for an RTX PRO 6000 Blackwell GPU and tuned for deterministic, schema-constrained JSON generation with Triton MoE kernels and disabled thinking mode. It enriches the compact `court_authority_cards_v4_target_cards.jsonl` (fast path) into `court_authority_cards_rag_targets.jsonl`, producing 11-field English RAG metadata per card with strict reference-grounding validation against the source paragraph and metadata. The pipeline short-circuits via deterministic regex auto-classification for notifications, dispositions (remitted/inadmissible/dismissed/granted/partial), very short fragments, and cost paragraphs, reserving the LLM only for non-trivial reasoning content. It includes a smoke test, batch-level checkpointing, single retry with an extended token cap and guard prompt, deterministic fallback after retry failure, and a 3% error-rate abort. The `pre_failure_capture` suffix is reflected in the captured outputs: environment setup, drive mount, and the pre-run reset succeeded, but the model-load, smoke-test, enrichment, stats, and sample cells contain no stored outputs in this snapshot.
