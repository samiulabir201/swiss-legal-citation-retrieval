# auth_cards_enrich_qwen3_8b_kaggle

**Path:** `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen3_8b_kaggle.ipynb`

## Configuration

Kaggle notebook settings (Edit -> Notebook Settings):

| Setting | Value |
|---|---|
| Accelerator | GPU T4 x 2 (or P100) |
| Internet | ON (needed to download Qwen3-8B-AWQ from HuggingFace) |
| Dataset | `samiulislam180041221/swiss-court-authority-cards-rag-targets` |

Install (Cell 2):
- `transformers>=4.51.0`, `accelerate`, `tqdm`
- Optional fallback: `autoawq` if Qwen3-8B-AWQ fails to load

`Config` dataclass (Cell 4):

- Files:
  - `input_file`: resolved by `first_existing_target_file()` (Kaggle mounts or local `artifacts/`)
  - `output_file`: `/kaggle/working/court_authority_cards_rag_targets_qwen3_8b.jsonl`
  - `failed_file`: `/kaggle/working/court_authority_cards_rag_targets_qwen3_8b_failed_cards.jsonl`
  - `checkpoint_file`: `/kaggle/working/rag_targets_qwen3_8b_checkpoint.txt`
- Model:
  - `model_id`: `Qwen/Qwen3-8B-AWQ`
  - `trust_remote_code`: True
  - `torch_dtype`: `float16`
  - `device_map`: `auto`
  - `attn_implementation`: None
  - `use_bnb_4bit`: False (set True with `Qwen/Qwen3-8B` if AWQ unavailable)
- Run control:
  - `limit`: 50 (default smoke; set 0 for full 363,258-card run)
  - `reset_output`: True
  - `batch_size`: 8
  - `progress_every_batches`: 10
- Generation:
  - `text_chars`: 700
  - `max_input_tokens`: 1600
  - `max_new_tokens`: 224
  - `retry_max_new_tokens`: 384
  - `do_sample`: False, `temperature`: 0.0, `top_p`: 1.0, `repetition_penalty`: 1.05
- Reliability:
  - `auto_classify`: True
  - `retry_bad_outputs`: True
  - `fallback_on_fail`: False
  - `validate_reference_grounding`: True

Dataset slug: `swiss-court-authority-cards-rag-targets`.

## Data

- Input: `court_authority_cards_v4_target_cards.jsonl` (363,258 target cards).
- Lookup paths (in order): `/kaggle/input/swiss-court-authority-cards-rag-targets/`, `/kaggle/input/datasets/`, `/kaggle/input/swiss-law/`, `/kaggle/input/swiss-law-artifacts/`, `/kaggle/input/swiss-citation-extraction/`, local `artifacts/`, then a recursive glob under `/kaggle/input/`.
- Per-card fields consumed: `text_excerpt_original`, `citation`, `court_base`, `legal_area`, `authority_role`, `issue_labels_en`, `law_codes`, `statutes_cited`, `court_cases_cited`, `language`, `matched_terms_multilingual`, `summary_en_proxy`, `is_notification_paragraph`, `_enrichment_target`.
- Outputs (JSONL):
  - Enriched cards with appended `rag_enrichment` object (12 fields incl. `method`).
  - Failed-card sidecar carrying `line_idx`, `citation`, `court_base`, `stage`, `error`, truncated `raw_output`, and original `card`.
  - Checkpoint file storing the next line index to resume from.

## Pipeline

Cell 2 (Install): pip-installs `transformers`, `accelerate`, `tqdm`; optional `autoawq` left commented.

Cell 3 (Smoke Test): seven checks gated by P/F/W markers. Verifies Python >= 3.9; CUDA available and per-GPU VRAM (>= 14 GB hard, 10-14 GB warn); package versions (`transformers>=4.51.0`, `accelerate>=0.26.0`, `tqdm>=4.0.0`); presence of input JSONL and `_enrichment_target` flag; HuggingFace reachability via `urllib`; tokenizer loads for `Qwen/Qwen3-8B-AWQ` and renders the chat template (with `enable_thinking=False`); `/kaggle/working` writability. Issues are accumulated and summarized.

Cell 4 (Configuration): defines paths, model id, run-control, generation, and reliability switches; instantiates `CONFIG` and prints input/output/limit/model summary.

Cell 5 (Helpers + Model):

- Schema/prompt: `RAG_SCHEMA_HINT` defines 11 target fields (`english_summary`, `legal_topic`, `legal_question`, `legal_rule`, `court_holding`, `factual_context`, `english_legal_concepts`, `search_keywords`, `natural_language_queries`, `paragraph_role`, `outcome_signal`). `SYSTEM_PROMPT` enforces deterministic Swiss legal JSON extraction, no chain-of-thought, no markdown, no invented references. `ROLE_VALUES` and `OUTCOME_VALUES` enumerate allowed categorical labels.
- Regex patterns (DE/FR/IT-aware): `COST_PROC_RE`, `REMITTAL_RE`, `INADMISSIBLE_RE`, `DISMISSED_RE`, `GRANTED_RE`, `PARTIAL_RE`, plus `BGE_REF_RE`, `DOCKET_REF_RE`, `ART_HEAD_RE`, `ART_GENERATED_RE`, `ART_NUM_RE`.
- Utilities: `_txt`, `_list`, `_term_values`, `_stub` build trimmed, de-duplicated payloads.
- Deterministic auto-classification (`auto_classify`): short-circuits the model when the card is a notification paragraph, a court-cost/legal-aid passage, a short procedural fragment (< 50 chars), or a clear disposition (remittal, inadmissibility, dismissal, granted/partial). Each branch returns a method-tagged stub.
- Validation: `validate_and_normalize_enrichment` enforces required keys, per-field char/item caps, role/outcome whitelists. `validate_reference_grounding` cross-checks every generated `art.` number, BGE citation, and docket against a normalized blob of source metadata + paragraph text.
- JSON parsing: `extract_first_json_object` walks balanced braces; `clean_model_json` strips `</think>` prefix and ```` ```json ```` fences; `parse_output_text` chains parsing + validation + grounding.
- Prompt builder (`build_user_prompt`): wraps trimmed `text_excerpt_original` (<= `text_chars`) and a 9-key metadata JSON in `<record>/<metadata_json>/<paragraph_original_language>` XML-like tags, then appends the schema and rules.
- I/O: `stream_input` yields `(line_idx, card)` between start-offset and stop-before; `count_lines`, `read_checkpoint`, `write_jsonl`, `write_failed` provide append/resume primitives.
- `QwenGenerator`: loads tokenizer (left-padded, eos as pad if missing), then model with `torch_dtype=float16`, `device_map="auto"`, `low_cpu_mem_usage=True`, optional `BitsAndBytesConfig` 4-bit (NF4 + double-quant) when `use_bnb_4bit=True`. AWQ load failure raises a `RuntimeError` instructing to install `autoawq` or switch to bnb 4-bit. `render_prompt` applies the chat template with `enable_thinking=False`, falling back to a `/no_think` suffix. `generate_raw` tokenizes a batch (max length `max_input_tokens`), moves to model device, runs `model.generate` under `torch.inference_mode()` with `do_sample`, `repetition_penalty`, and EOS/pad ids, and decodes only the new tokens.
- `validate_config`: bounds checks on `batch_size`, `limit`, `text_chars`, `max_new_tokens`, and rejects bnb-4bit + AWQ.

Cell 6 (`main()` Run):

1. Validates config; aborts with FileNotFoundError if input missing (listing `/kaggle/input` for diagnostics).
2. Creates parent dirs; if `reset_output`, deletes prior output/failed/checkpoint.
3. Reads checkpoint; resets if stale (>= total lines). Computes `stop_before = min(total, start + limit)` when `limit` set, else full file.
4. Logs paths, model, start/total/to-process counts, batch size.
5. Constructs `QwenGenerator`; opens output and failed handles in append mode; starts tqdm.
6. Iterates `stream_input`. Each card runs `auto_classify` first; on hit, the pending LLM batch is flushed, the stub is validated, the card is written, and `auto_count` increments. Otherwise the card is queued; when `len(pending) >= batch_size`, `flush_batch` runs.
7. `flush_batch`: `generate_raw` -> per-card `parse_output_text`; on parse/validation/grounding failure with `retry_bad_outputs=True`, retries once with an added "previous output was invalid" guard and `retry_max_new_tokens`. Persistent failures either trigger `deterministic_fallback` (when `fallback_on_fail=True`) and write the card with `fallback_count++`, or write the original to the failed sidecar with `fail_count++` while still advancing the checkpoint.
8. Every `progress_every_batches` batches (and on the first), prints batch size, batch/avg rate, ETA, and counters.
9. On exit, flushes/closes handles, prints final processed / LLM-ok / auto / failed / fallback counts, elapsed minutes, and overall rate.

## Results

The notebook has no executed cell outputs (no `outputs` arrays / no execution counts captured in the .ipynb). No metrics, smoke-test output, batch logs, or final-summary numbers are present.

## Summary

Kaggle GPU notebook that enriches 363,258 Swiss Federal Tribunal target authority cards with `Qwen/Qwen3-8B-AWQ` via HuggingFace `transformers`, producing an 11-field RAG JSON per card (summary, topic, question, rule, holding, facts, concepts, keywords, NL queries, paragraph role, outcome signal) plus a `method` tag. A deterministic `auto_classify` short-circuits notifications, court-cost paragraphs, short procedural fragments, and disposition lines (remittal / inadmissible / dismissed / granted+partial) using multilingual DE/FR/IT regexes, reserving the model for substantive reasoning. Outputs are validated against a strict schema, role/outcome whitelists, char/item caps, and a reference-grounding check that rejects any generated `art.`, BGE, or docket reference absent from the source card. The pipeline is resumable via a `/kaggle/working` checkpoint, retries malformed model outputs once with a stricter guard, optionally falls back to a deterministic stub on persistent failures, and streams metrics through tqdm with periodic batch ETA logs. The captured `.ipynb` contains source only; no execution outputs are recorded.
