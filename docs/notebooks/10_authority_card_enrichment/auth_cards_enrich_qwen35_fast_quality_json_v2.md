# auth_cards_enrich_qwen35_fast_quality_json_v2

**Path:** e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen35_fast_quality_json_v2.ipynb

Batch-enriches `court_authority_cards_v4.jsonl` with English legal RAG metadata using Qwen/Qwen3.5-35B-A3B served by vLLM offline inference. Tuned for an RTX PRO 6000 Blackwell 96 GB GPU on Google Compute Engine / Colab, with deterministic schema-constrained JSON generation, Qwen thinking disabled, Triton MoE kernels (to avoid FlashInfer CUTLASS SM120 JIT failures), checkpointed JSONL append output, deterministic auto-classification for trivial paragraphs, JSON validation, retry, and a high-error-rate abort guard. Target output: `/content/drive/MyDrive/swiss_law/artifacts/court_authority_cards_rag.jsonl`.

## Configuration

Environment detection: runs in Colab (`IN_COLAB=True`); GPU reported as `NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB`; Python `3.12.13`.

Dependency install (cell `install-deps`): uninstalls `PIL pillow numpy scipy torchvision vllm transformers`, hard-deletes their stale site-packages directories, then installs `uv`, `tqdm`, and reinstalls Pillow 11.3.0, NumPy 2.3.5, SciPy 1.16.3 from a `/tmp/vllm_constraints.txt`, followed by `uv pip install --system vllm --torch-backend=auto -c <constraints>`. Sanity imports report: `Pillow OK: 11.3.0`, `NumPy OK: 2.3.5`, `SciPy OK: 1.16.3`, `vLLM import OK`. Google Drive is mounted at `/content/drive`.

Paths (cell `config`):
- `BASE_DIR = /content/drive/MyDrive/swiss_law` (Colab) or `..` resolved
- `DATA_DIR = BASE_DIR/data`, `INSIGHTS_DIR = BASE_DIR/data_insights`, `ART_DIR = BASE_DIR/artifacts`, `SCRIPT_DIR = BASE_DIR/scripts`
- `INPUT_FILE = ART_DIR/court_authority_cards_v4_target_cards.jsonl` (when `ENRICH_TARGET_CARD_FILE_ONLY=True`) else `ART_DIR/court_authority_cards_v4.jsonl`
- `OUTPUT_FILE = ART_DIR/court_authority_cards_rag_targets.jsonl` (target-card mode) else `ART_DIR/court_authority_cards_rag.jsonl`
- `TARGET_CARD_FILE = ART_DIR/court_authority_cards_v4_target_cards.jsonl`
- `TARGET_FILE = ART_DIR/court_authority_cards_v4_enrichment_targets.jsonl`
- `CHECKPOINT_FILE = ART_DIR/rag_checkpoint.txt`

Model and inference parameters:
- `MODEL_ID = 'Qwen/Qwen3.5-35B-A3B'`, `QUANTIZATION = None`, dtype `bfloat16`
- `GPU_MEMORY_UTIL = 0.90`, `MAX_MODEL_LEN = 4096`, `TENSOR_PARALLEL = 1`, `enforce_eager=True`, `enable_prefix_caching=True`, `disable_log_stats=True`, `limit_mm_per_prompt={'image': 0, 'video': 0}`, `kernel_config=KernelConfig(moe_backend='triton')`, `max_num_seqs=max(BATCH_SIZE, 64)`, `language_model_only=True` (if supported)
- `BATCH_SIZE = 32`, `TEMPERATURE = 0.0`, `TOP_P = 1.0`, `REPETITION_PENALTY = 1.03`
- `MAX_TOKENS = 640`, `RETRY_MAX_TOKENS = 1024`, `SMOKE_MAX_TOKENS = 1024`
- `TEXT_CHARS = 1500` per paragraph excerpt
- Structured decoding via `StructuredOutputsParams(json=RAG_SCHEMA)` if available, else `GuidedDecodingParams(json=...)`; otherwise raises.
- Env vars: `VLLM_LOGGING_LEVEL=DEBUG`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`; pops `VLLM_WORKER_MULTIPROC_METHOD`, `VLLM_MOE_BACKEND`, `VLLM_FLASHINFER_MOE_BACKEND`.

Targeting and quality guardrails:
- `ENRICH_TARGET_CARD_FILE_ONLY = True` (fast path, only enrich the compact target-card file)
- `TARGETED_ENRICHMENT = True`, `WRITE_DETERMINISTIC_FALLBACK_FOR_UNTARGETED = True`
- `TARGET_LIMIT = 1000`, `LIMIT = TARGET_LIMIT` when target-card-only
- `RETRY_BAD_JSON = True`, `ABORT_IF_ERROR_RATE_ABOVE = 0.03`, `ERROR_RATE_CHECK_AFTER = 200`, `VALIDATE_REFERENCE_GROUNDING = True`
- `RESET_OUTPUT = False` in config cell; reset cell sets `RESET_OUTPUT = True` later and deletes the old output and checkpoint.

JSON schema `RAG_SCHEMA` (`additionalProperties=False`) required keys: `english_summary` (<=320), `legal_topic` (<=140), `legal_question` (<=260), `legal_rule` (<=320), `court_holding` (<=260), `factual_context` (<=260), `english_legal_concepts` (array <=6 strings <=80), `search_keywords` (array <=8 strings <=80), `natural_language_queries` (array <=3 strings <=180), `paragraph_role` enum `[holding, reasoning, background, cost, procedural, disposition, standard_of_review, obiter]`, `outcome_signal` enum `[granted, dismissed, inadmissible, remitted, partial, none]`.

System prompt instructs the model to act as a deterministic Swiss legal JSON extraction engine, return one schema-matching JSON object using concise English legal terminology, use only statutes/articles/citations present in the paragraph or metadata, leave empty/`"none"` when unsupported, and not repeat the source text or emit markdown, bibliography, tables, or chain-of-thought.

## Data

Input: `INPUT_FILE` resolved to `court_authority_cards_v4_target_cards.jsonl` (compact target-card file inside `ART_DIR`); the fallback full file would be `court_authority_cards_v4.jsonl` (~2.47M cards, not used here). Each card is a JSON line with fields including `citation`, `court_base`, `legal_area`, `authority_role`, `issue_labels_en`, `law_codes`, `statutes_cited`, `court_cases_cited`, `text_excerpt_original`, `summary_en_proxy`, `matched_terms_multilingual`, `is_notification_paragraph`.

Target index file: `court_authority_cards_v4_enrichment_targets.jsonl` (used only when not in `ENRICH_TARGET_CARD_FILE_ONLY` mode), loaded via `load_target_line_indices`, each row having `line_idx` referencing positions in the full v4 file.

Output: JSONL append of the input card plus a `rag_enrichment` field containing the validated schema fields and a `method` tag (`qwen35_35b_a3b_vllm`, `qwen35_35b_a3b_vllm_retry`, `auto_notification`, `auto_remittal`, `auto_inadmissible`, `auto_dismissed`, `auto_granted`, `auto_short`, `auto_cost`, `deterministic_v4_fallback`, `deterministic_fallback_after_parse_failed`); failed parses additionally carry `parse_error` and `raw_output` (first 400 chars of cleaned text). Checkpoint file holds the next line index to resume from.

## Pipeline

1. Environment detection and GPU verification via `nvidia-smi` query (cell `detect-colab`).
2. Forced clean install of Pillow/NumPy/SciPy/vLLM with pinned versions; runtime import sanity (cell `install-deps`).
3. Drive mount (cell `mount-drive`).
4. Configuration: paths, model, inference, targeting, and guardrail constants (cell `config`).
5. Schema and prompts: defines `RAG_SCHEMA`, `SYSTEM_PROMPT`, plus regex pre-filters `COST_PROC_RE`, `REMITTAL_RE`, `INADMISSIBLE_RE`, `DISMISSED_RE`, `GRANTED_RE`, `PARTIAL_RE` (cell `schema-and-prompts`).
6. Helpers (cell `helpers`):
   - Text/list normalization (`_txt`, `_list`, `_term_values`, `_join`).
   - `load_target_line_indices` reads target JSONL into a set/dict keyed by `line_idx`.
   - `build_user_message` packs metadata and the truncated paragraph (up to 1500 chars) into `<record><metadata>...<paragraph_original_language>...</record>` XML-style envelope and appends terse JSON-only instructions.
   - `_stub`, `deterministic_disposition`, `deterministic_fallback`, `auto_classify` synthesize JSON without the LLM for notifications, remittals, inadmissibility, dismissal, grants (with partial detection), short text, and cost/legal-aid paragraphs.
   - `stream_input` yields `(line_idx, card)` from offset; `count_lines` counts raw lines.
   - `render_prompt` calls `tokenizer.apply_chat_template` with `enable_thinking=False`, falling back to a manual JSON-only suffix on `TypeError`.
   - `clean_model_json` strips `</think>` content, fenced code blocks, and trims to the outermost `{...}`.
   - `validate_and_normalize_enrichment` enforces required fields, clips string lengths, deduplicates/clips array items, and resets `paragraph_role`/`outcome_signal` to defaults if out-of-enum; stamps `method`.
   - `validate_reference_grounding` ensures any `Art.`/`Article` IDs in the generated payload are present in the source metadata/text via `_article_ids_from_blob` (tolerates `Art. 6f` <-> `Art. 6 f.` compaction), and that all `BGE ... ...` references and Federal Tribunal docket numbers like `1A_123/2024` appear in the source blob. Raises `ValueError` listing the first eight bad references.
   - `parse_output_text` chains `clean_model_json` -> `json.loads` -> `validate_and_normalize_enrichment` -> `validate_reference_grounding`.
7. Model load (cell `load-model`): builds `LLM(**kwargs)` with `KernelConfig(moe_backend='triton')`; pulls tokenizer; constructs `sampling_params`, `retry_sampling_params`, `smoke_sampling_params` via `make_sampling_params`, each backed by `StructuredOutputsParams` when available.
8. Smoke test (cell `Q7LmYo7Z7Fv5`, `RUN_SMOKE_TEST=True`, `SMOKE_N=8`): selects target rows, runs deterministic fallback checks on up to three auto-classifiable cards, generates with `smoke_sampling_params`, parses each output with `parse_output_text`. Failures raise `RuntimeError('RED: ...')`; success prints GREEN.
9. Reset cell (`y-RnVQp9AXOe`) deletes `OUTPUT_FILE` and `CHECKPOINT_FILE` for a clean run.
10. Enrichment loop (cell `run-enrichment`):
    - Loads target indices (skipped in `ENRICH_TARGET_CARD_FILE_ONLY` mode).
    - Resumes from the checkpoint file; counts total lines; reports targeted LLM rows in range.
    - For each card up to `LIMIT`: decides `should_llm`. Non-target cards either get auto-classified or `deterministic_fallback` (when `WRITE_DETERMINISTIC_FALLBACK_FOR_UNTARGETED`). LLM-targeted cards first try `auto_classify`; if unmatched they enter `pending` for batched generation at `BATCH_SIZE=32`.
    - `flush_batch` calls `llm.generate(...)` with `sampling_params`, parses each output, retries a single failure with stricter guard text via `retry_one` (using `retry_sampling_params`), and writes the augmented card; if both attempts fail, writes a `deterministic_fallback_after_parse_failed` stub with `parse_error` and `raw_output`. After each batch it updates the checkpoint, logs throughput (`batch_rate`, `avg_rate`, `eta_min`, retry/error counts) every batch or on errors, and aborts if `json_errors / llm_attempts > ABORT_IF_ERROR_RATE_ABOVE` after `ERROR_RATE_CHECK_AFTER` attempts.
    - Final flush, closes `OUTPUT_FILE`, prints totals.
11. Verification (cell `stats`): re-reads `OUTPUT_FILE`, counts totals, distributions of `paragraph_role`, `outcome_signal`, and `method`, plus required-field completeness; prints up to five problem cards. Prints RED/GREEN.
12. Sample (cell `sample`): filters to `method` containing `'qwen35'`, prints three random LLM-enriched cards.

## Results

- `detect-colab` outputs: `Running in Colab: True`; `GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB`.
- `install-deps` outputs: prints `Python executable: /usr/bin/python3`, `Python version: 3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0]`, the five pip/uv install command invocations, and confirms `Pillow OK: 11.3.0`, `NumPy OK: 2.3.5`, `NumPy internal symbol OK`, `SciPy OK: 1.16.3`, `vLLM import OK`.
- `mount-drive` output: `Mounted at /content/drive`.
- `config`, `schema-and-prompts`, `helpers` cells: no captured outputs displayed for inspection in this notebook snapshot (the cells produce print statements but no outputs were saved).
- `load-model` cell output: notebook records `Outputs are too large to include. Use Bash with: cat <notebook_path> | jq '.cells[12].outputs'` — the vLLM load logs were truncated from the saved notebook.
- `Q7LmYo7Z7Fv5` (smoke test): no outputs saved in the notebook.
- `y-RnVQp9AXOe` (reset cell) output:
  ```
  Deleted old output: /content/drive/MyDrive/swiss_law/artifacts/court_authority_cards_rag.jsonl
  Deleted old checkpoint: /content/drive/MyDrive/swiss_law/artifacts/rag_checkpoint.txt
  Clean reset done. Now rerun the enrichment cell from line 0.
  ```
- `run-enrichment`, `stats`, `sample` cells: no outputs saved in this notebook snapshot (the production run did not leave persisted output cells).

## Summary

This notebook is the production enrichment pipeline that turns Swiss Federal Tribunal authority cards into English RAG metadata via Qwen3.5-35B-A3B on vLLM, configured for Blackwell-class 96 GB GPUs with Triton MoE, BF16, structured JSON decoding, and a 4096-token context. It pairs schema-constrained generation with deterministic regex pre-classifiers for procedural, cost, disposition, and remittal paragraphs, and a reference-grounding check that rejects generated `Art.`, `BGE`, or docket citations that are absent from the source card. The loop streams a compact target-card JSONL, batches at 32, retries failed parses once with stricter guidance, falls back deterministically on second failure, checkpoints to disk, and aborts if the post-retry JSON error rate exceeds 3% after 200 attempts. Captured outputs confirm a Blackwell GPU was available, the patched Pillow/NumPy/SciPy/vLLM stack imported cleanly, Drive mounted, and the output/checkpoint were reset prior to running; the saved notebook does not include the vLLM load logs (marked truncated), the smoke test verdict, the enrichment progress logs, or the verification statistics. Outputs land at `ART_DIR/court_authority_cards_rag_targets.jsonl` with one input card per line augmented by a `rag_enrichment` block tagged by `method`.
