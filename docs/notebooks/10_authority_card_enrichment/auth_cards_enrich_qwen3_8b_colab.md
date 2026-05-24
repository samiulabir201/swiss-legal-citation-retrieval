# auth_cards_enrich_qwen3_8b_colab

**Path:** `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen3_8b_colab.ipynb`

## Configuration

- Model: `Qwen/Qwen3-8B-AWQ` via vLLM offline inference, `quantization='awq'`, `dtype='float16'`, `enforce_eager=True`, `enable_prefix_caching=True`, `trust_remote_code=True`, `limit_mm_per_prompt={'image':0,'video':0}`, `max_num_seqs=max(BATCH_SIZE,32)`, `max_num_batched_tokens=8192`, optional `language_model_only=True`.
- Sampling: `TEMPERATURE=0.0`, `TOP_P=1.0`, `REPETITION_PENALTY=1.05`, `MAX_TOKENS=224`, `RETRY_MAX_TOKENS=384`, `SMOKE_MAX_TOKENS=384`, schema-constrained via `StructuredOutputsParams(json=RAG_SCHEMA)` (fallback `GuidedDecodingParams`).
- Throughput: `GPU_MEMORY_UTIL=0.90`, `MAX_MODEL_LEN=2048`, `BATCH_SIZE=16`, `TENSOR_PARALLEL=1`, `TEXT_CHARS=700`.
- Chat template applied with `enable_thinking=False` (Qwen thinking disabled).
- Env vars set before vLLM init: `VLLM_LOGGING_LEVEL=INFO`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`; pops `VLLM_WORKER_MULTIPROC_METHOD`, `VLLM_MOE_BACKEND`, `VLLM_FLASHINFER_MOE_BACKEND`.
- Paths (Colab): `BASE_DIR=/content/drive/MyDrive/swiss_law`, `ART_DIR=BASE_DIR/artifacts`.
- Targeted enrichment: `ENRICH_TARGET_CARD_FILE_ONLY=True`, `TARGETED_ENRICHMENT=True`, `WRITE_DETERMINISTIC_FALLBACK_FOR_UNTARGETED=True`, `TARGET_LIMIT=1000` (1000 for first smoke/quality run; set 0 for full 363,258-card target run).
- Quality guardrails: `RETRY_BAD_JSON=False`, `ABORT_IF_ERROR_RATE_ABOVE=1.00`, `ERROR_RATE_CHECK_AFTER=200`, `VALIDATE_REFERENCE_GROUNDING=True`.
- Input file: `ART_DIR/court_authority_cards_v4_target_cards.jsonl`.
- Output file: `ART_DIR/court_authority_cards_rag_targets_qwen3_8b.jsonl`.
- Failed file: `ART_DIR/court_authority_cards_rag_targets_qwen3_8b_failed_cards.jsonl`.
- Failed retry/still-failed files: `..._failed_cards_recovered.jsonl` / `..._failed_cards_still_failed.jsonl`.
- Checkpoint: `ART_DIR/rag_targets_qwen3_8b_checkpoint.txt`.
- Install cell (Colab): uninstalls and force-reinstalls `pillow==11.3.0`, `numpy==2.3.5`, `scipy==1.16.3` under `/tmp/vllm_constraints.txt`; installs `vllm --torch-backend=auto` via `uv pip install --system`; hard-deletes stale `PIL/pillow/numpy/scipy/torchvision/vllm/transformers` site-package directories first.
- GPU detected: `NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB`; Python `3.12.13`.

## Data

Input: `court_authority_cards_v4_target_cards.jsonl` (compact pre-targeted file, full run 363,258 cards). Each card is a Swiss Federal Tribunal paragraph in German/French/Italian with metadata fields `citation`, `court_base`, `legal_area`, `authority_role`, `issue_labels_en`, `law_codes`, `statutes_cited`, `court_cases_cited`, `text_excerpt_original`, `is_notification_paragraph`, `matched_terms_multilingual`, `summary_en_proxy`.

Output: each input card is rewritten with a `rag_enrichment` sub-object conforming to `RAG_SCHEMA`. Required fields: `english_summary` (<=320 chars), `legal_topic` (<=140), `legal_question` (<=260), `legal_rule` (<=320), `court_holding` (<=260), `factual_context` (<=260), `english_legal_concepts` (<=6, items <=80), `search_keywords` (<=8, items <=80), `natural_language_queries` (<=3, items <=180), `paragraph_role` enum (`holding|reasoning|background|cost|procedural|disposition|standard_of_review|obiter`), `outcome_signal` enum (`granted|dismissed|inadmissible|remitted|partial|none`). A `method` tag records provenance.

## Pipeline

1. **Environment setup.** Detect Colab, print GPU, run conservative install (pins Pillow/NumPy/SciPy, installs vLLM via `uv` with torch-backend resolver), import sanity-check `PIL`/`numpy`/`scipy`/`numpy._core.umath._center`/`vllm`. Mount Google Drive.
2. **Configuration.** Define paths, model, throughput knobs, target-card-only flags, file paths, optional `RESET_OUTPUT` cleanup of output/failed/checkpoint files.
3. **JSON schema + prompts.** Define multilingual regexes for deterministic short-circuiting: `COST_PROC_RE` (cost/legal-aid DE/FR/IT terms), `REMITTAL_RE` (remittal phrasings), `INADMISSIBLE_RE`, `DISMISSED_RE`, `GRANTED_RE`, `PARTIAL_RE`. Define `RAG_SCHEMA` and `SYSTEM_PROMPT` (deterministic Swiss legal JSON extraction engine instructions: concise English terminology, no invention, no chain-of-thought, no markdown).
4. **Helper functions.**
   - Reference regexes `BGE_REF_RE`, `DOCKET_REF_RE`, `ART_REF_RE`, `ART_HEAD_RE`, `ART_GENERATED_RE`, `ART_NUM_RE`.
   - Text normalizers `_txt`, `_list`, `_term_values`, `_join`.
   - `load_target_line_indices` loads `TARGET_FILE` indices when not in card-only mode.
   - `build_user_message` wraps the card into `<record><metadata>…</metadata><paragraph_original_language>…</paragraph_original_language></record>` plus three instruction lines, truncating `text_excerpt_original` to `TEXT_CHARS`.
   - `_stub`/`deterministic_disposition`/`deterministic_fallback`/`auto_classify` produce schema-conforming dicts directly from metadata for notification paragraphs, remittal/inadmissible/dismissed/granted/partial outcomes, very short text (`len<50`), and cost paragraphs.
   - `stream_input`, `count_lines`.
   - `render_prompt` applies the chat template with `enable_thinking=False` (falls back to plain template + manual guard line on `TypeError`).
   - `clean_model_json` strips `</think>`, fenced blocks, and surrounding noise to isolate the JSON object.
   - `validate_and_normalize_enrichment` enforces required keys, clips string lengths, dedupes/truncates arrays, normalizes enums, sets `method`.
   - `validate_reference_grounding` re-checks every `Art. N`, `BGE …`, and docket reference in the generated JSON against a lowercased reference blob built from the card; rejects ungrounded references (tolerates "Art. 6 f." → "Art. 6f" when the source uses the spaced form). Disabled by `VALIDATE_REFERENCE_GROUNDING=False`.
   - `parse_output_text` chains `clean_model_json` -> `json.loads` -> `validate_and_normalize_enrichment` -> `validate_reference_grounding`.
5. **Load Qwen3-8B-AWQ.** Build `LLM(**build_llm_kwargs())`, fetch tokenizer, construct three `SamplingParams` (`sampling_params`, `retry_sampling_params`, `smoke_sampling_params`) all using structured JSON decoding against `RAG_SCHEMA`.
6. **Smoke test** (`RUN_SMOKE_TEST=True`, `SMOKE_N=8`). Stream input, skip cards auto-classified by `auto_classify`, run vLLM on 8 LLM-eligible target cards with `smoke_sampling_params`, parse + ground-validate, print GREEN/RED.
7. **Run enrichment.** Validate input exists; resume from `CHECKPOINT_FILE` (reset if stale); open append handles for `OUTPUT_FILE` + `FAILED_FILE`; iterate `stream_input` and for each card either (a) write deterministic auto-classification, (b) write deterministic fallback (untargeted rows when fallback flag set), or (c) accumulate into `pending` until `BATCH_SIZE` then `flush_batch`. `flush_batch` calls `llm.generate(prompts, sampling_params=sampling_params)`, parses each output via `parse_output_text(method='qwen35_35b_a3b_vllm', card=card)`, writes `rag_enrichment` on success, writes a failed-row record on parse/grounding failure (`RETRY_BAD_JSON=False`), flushes files, updates the checkpoint to `last_line_idx+1`, and logs per-batch throughput/error stats every 5 batches. Optional inline `retry_one` exists but is off in the main run.
8. **Verify output.** Walk `OUTPUT_FILE`, tally `paragraph_role`/`outcome_signal`/`method` distributions, count missing-required cards, flag parse-fallback method, print GREEN/RED.
9. **Inspect samples.** Random-sample three LLM-enriched cards (`method` contains `qwen35`) and print key enrichment fields.
10. **Empty placeholder cell.**
11. **Failed-row retry pass** (`RUN_FAILED_RETRY_PASS=False` by default). Streams `FAILED_FILE`, regenerates each card with an extra-guard prompt and `retry_sampling_params`, writes recoveries to `FAILED_RETRY_FILE` and still-failed cards to `FAILED_STILL_FILE`.

## Results

Cell 2 (env setup): `Running in Colab: True`; `GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB`.

Cell 3 (install): `Pillow OK: 11.3.0`, `NumPy OK: 2.3.5`, `SciPy OK: 1.16.3`, `NumPy internal symbol OK`, `vLLM import OK`.

Cell 4: `Mounted at /content/drive`.

Cell 6 (configuration print, run with stale values from the Qwen3.5-35B notebook session — does not match the configuration source above):
```
BASE_DIR   : /content/drive/MyDrive/swiss_law
INPUT_FILE : /content/drive/MyDrive/swiss_law/artifacts/court_authority_cards_v4_target_cards.jsonl
OUTPUT_FILE: /content/drive/MyDrive/swiss_law/artifacts/court_authority_cards_rag_targets.jsonl
FAILED_FILE: /content/drive/MyDrive/swiss_law/artifacts/court_authority_cards_rag_failed_cards.jsonl
TARGET_CARD_FILE: /content/drive/MyDrive/swiss_law/artifacts/court_authority_cards_v4_target_cards.jsonl
TARGET_FILE: /content/drive/MyDrive/swiss_law/artifacts/court_authority_cards_v4_enrichment_targets.jsonl
MODEL_ID   : Qwen/Qwen3.5-35B-A3B
BATCH_SIZE=64, MAX_TOKENS=256, RETRY_MAX_TOKENS=512, SMOKE_MAX_TOKENS=512
ENRICH_TARGET_CARD_FILE_ONLY=True, TARGETED_ENRICHMENT=True, TARGET_LIMIT=0, WRITE_FALLBACK=True
```

Cell 8 (schema fields): `['english_summary', 'legal_topic', 'legal_question', 'legal_rule', 'court_holding', 'factual_context', 'english_legal_concepts', 'search_keywords', 'natural_language_queries', 'paragraph_role', 'outcome_signal']` (all required).

Cell 10: `Helpers loaded.`

Cell 12 (model load, stale Qwen3.5-35B-A3B session): `MODEL_ID=Qwen/Qwen3.5-35B-A3B`, `quantization: None`, `dtype: bfloat16`, `max_model_len=2048`, `gpu_memory_utilization=0.9`, `batch_size=64`, `max_tokens=256`, `retry_max_tokens=512`; vLLM v0.20.0; resolved architecture `Qwen3_5MoeForConditionalGeneration`; weights download 200.79s (66.97 GiB); load 217.80s using 64.69 GiB; KV cache 18.0 GiB / 235,488 tokens / max concurrency 111.62x at 2048 tokens; FlashAttention v2; FlashInfer Autotuner ran; `init engine took 44.36 s`; three `StructuredOutputsParams(json=...)` sampling-params built (`max_tokens=256/512/512`); `Model loaded successfully.`

Cell 14 (smoke test): `GREEN: smoke test passed. 8 target LLM outputs valid and reference-grounded; deterministic fallback callable.` All 8 samples parsed cleanly; example BGE citations include `BGE 139 I 2 E. 5.7`, `BGE 139 I 2 E. 6.3`, `BGE 145 I 1 E. 1.1.3`, `BGE 145 I 1 E. 6`, `BGE 145 I 1 E. 6.5.1`, `BGE 145 I 1 E. 8.3`, `BGE 142 I 1 E. 7.2.1`, `BGE 136 I 1 E. 4.2`. `Deterministic fallback checks: 0` (no auto-eligible cards encountered before 8 LLM samples). One sample (line 7) was assigned `paragraph_role: "cost"` despite being substantive equality-of-law reasoning; one sample (line 4) emitted `legal_rule: "Art. 34 Abs. 2 BV"` as a bare citation. Sample 2 finished with truncated `english_summary` ending mid-word and a truncated `natural_language_queries` array (a single open-quoted entry). Sample 4 was truncated mid-array.

Cell 16 (reset): no output.

Cell 17 (run enrichment, stale checkpoint session — the run executed against the Qwen3.5-35B output file path with a checkpoint past end of file, producing a no-op):
```
Resuming at line 2,238,067
Counting lines in court_authority_cards_v4_target_cards.jsonl ...
Total=363,258  To process=-1,874,809
Using BATCH_SIZE=64, MAX_TOKENS=256, RETRY_BAD_JSON=False
Output file: /content/drive/MyDrive/swiss_law/artifacts/court_authority_cards_rag_targets.jsonl
Failed rows file: /content/drive/MyDrive/swiss_law/artifacts/court_authority_cards_rag_failed_cards.jsonl
Targeted LLM rows in processing range: -1,874,809
enrich: 2238067card [00:00, ?card/s]
Done. Failed rows captured: 0
Retries used inline: 0
LLM target rows attempted: 0
Deterministic fallback rows: 0
Auto-classified rows: 0
Output ? /content/drive/MyDrive/swiss_law/artifacts/court_authority_cards_rag_targets.jsonl
Failures ? /content/drive/MyDrive/swiss_law/artifacts/court_authority_cards_rag_failed_cards.jsonl
```
Note: `Total=363,258` but checkpoint `2,238,067` so `target_total - start = -1,874,809`; the stale-checkpoint reset branch did not trigger because `start < total_lines` evaluation only resets when `start >= total_lines` — here `2,238,067 >= 363,258` so it should have reset, yet the output shows `Resuming at line 2,238,067`, indicating the print statement labeled "Resuming at validated line" was suppressed and the stale-checkpoint message likewise did not fire in the captured output. The progress bar opened initialized at 2,238,067 and immediately closed with zero rows processed.

Cells 19, 20, 22: no recorded output. Cell 21 is empty.

## Summary

This Colab notebook is the low-cost Qwen3-8B-AWQ variant of the Swiss authority-card enrichment pipeline; it consumes the pre-targeted `court_authority_cards_v4_target_cards.jsonl` (363,258 rows) and writes a `rag_enrichment` block per card via vLLM structured-JSON decoding against a strict `RAG_SCHEMA`, with deterministic short-circuits for notification, cost, remittal, and disposition paragraphs plus regex-based reference grounding that rejects fabricated `Art.`/`BGE`/docket citations. The notebook's configuration source declares `Qwen/Qwen3-8B-AWQ`, `quantization='awq'`, `dtype='float16'`, `BATCH_SIZE=16`, `MAX_TOKENS=224`, and `TARGET_LIMIT=1000`. The captured outputs, however, are from a stale Qwen3.5-35B-A3B kernel session (`MODEL_ID=Qwen/Qwen3.5-35B-A3B`, `quantization=None`, `dtype=bfloat16`, `BATCH_SIZE=64`, `MAX_TOKENS=256`, `TARGET_LIMIT=0`), so the model-load logs and method tag `qwen35_35b_a3b_vllm` are from the 35B run, not from an 8B run. The smoke test in that session passed 8/8 schema-valid grounded outputs; the production-run cell executed against a stale checkpoint (`2,238,067` vs. total `363,258`) and processed zero rows. The verify and inspect cells produced no output in this notebook.
