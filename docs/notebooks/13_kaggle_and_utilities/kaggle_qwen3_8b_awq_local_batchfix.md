# kaggle_qwen3_8b_awq_local_batchfix

**Path:** `e:\swiss_citation_extraction\notebooks\13_misc_kaggle_and_utilities\kaggle_qwen3_8b_awq_local_batchfix.ipynb`

## Configuration

Notebook header (Cell 0, markdown) describes a Qwen3-8B-AWQ enrichment job for the Swiss Federal Tribunal `court_authority_cards_v4_target_cards.jsonl` corpus (363,258 target cards) running on a Kaggle GPU session. Author intent: load the AWQ-quantised model from the Kaggle-mounted dataset `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1`, write outputs to `/kaggle/working/`, and use a checkpoint file to resume across sessions. Cell 0 also documents the patch reflected in the notebook name: "fixed batching bug where `(line_idx, card)` tuples were passed to the AWQ generator instead of card dictionaries."

Recommended Kaggle settings (from Cell 0):
- Accelerator: GPU T4 x 2 (or P100)
- Internet: ON for the Cell 2 package install
- Input dataset: `samiulislam180041221/swiss-court-authority-cards-rag-targets`
- Model dataset: Qwen3-8B-AWQ mounted at `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1`

Cell 1 (Cell 2 in notebook numbering) installs packages once per session:
```
!pip install -q -U --no-cache-dir transformers accelerate safetensors tqdm gptqmodel
```
A comment warns: "Do NOT install autoawq for this notebook path unless you intentionally switch away from AutoModelForCausalLM + Transformers AWQ loading."

Cell 3 (`Config` dataclass) — key fields:
- `input_file`: resolved via `first_existing_target_file()` (searches `/kaggle/input/...` paths plus a recursive glob).
- `output_file`: `/kaggle/working/court_authority_cards_rag_targets_qwen3_8b.jsonl`
- `failed_file`: `/kaggle/working/court_authority_cards_rag_targets_qwen3_8b_failed_cards.jsonl`
- `checkpoint_file`: `/kaggle/working/rag_targets_qwen3_8b_checkpoint.txt`
- `model_id`: `first_existing_model_source()` — local Kaggle AWQ path if present, else HF fallback `Qwen/Qwen3-8B-AWQ`.
- `trust_remote_code=True`, `torch_dtype="float16"` (T4/P100-safe), `device_map="auto"`, `attn_implementation=None`, `use_bnb_4bit=False`.
- Run control: `limit=50` (quality check default; 0 = full run), `reset_output=True`, `batch_size=8`, `progress_every_batches=10`.
- Generation: `text_chars=700`, `max_input_tokens=1600`, `max_new_tokens=224`, `retry_max_new_tokens=384`, `do_sample=False`, `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.05`.
- Reliability: `auto_classify=True`, `retry_bad_outputs=True`, `fallback_on_fail=False`, `validate_reference_grounding=True`.

Cell 5 (sanity check) verifies the AWQ backend through Transformers + gptqmodel, printing `Transformers:` and `gptqmodel: OK`.

## Data

Input: `court_authority_cards_v4_target_cards.jsonl` (per smoke test: 1029 MB, 363,258 lines). First-card field set as printed:
`['_enrichment_target', 'authority_role', 'citation', 'court_base', 'court_cases_cited', 'family', 'is_notification_paragraph', 'issue_labels_en', 'language', 'law_codes', 'legal_area', 'matched_terms_multilingual', 'pattern', 'provenance', 'retrieval_text_en', 'statutes_cited', 'structural', 'subfamily', 'summary_en_proxy', 'text_excerpt_original']`. First card citation: `BGE 139 I 2 E. 5.7`.

Outputs written to `/kaggle/working/`:
- `court_authority_cards_rag_targets_qwen3_8b.jsonl` — enriched cards with new `rag_enrichment` field.
- `court_authority_cards_rag_targets_qwen3_8b_failed_cards.jsonl` — failure rows with stage, error, raw_output, card.
- `rag_targets_qwen3_8b_checkpoint.txt` — last processed line index (resume marker).

## Pipeline

Cell 2 (Cell 3, Smoke Test): Python/CUDA/GPU check, library version gates (`transformers>=4.51.0`, `accelerate>=0.26.0`, `tqdm>=4.0.0`), input dataset resolution (with fallback recursive glob), Kaggle AWQ model path existence + shard count, `gptqmodel` import probe, tokenizer + chat-template render probe (with `enable_thinking=False` when supported), `/kaggle/working` writability probe. Each check prints a tick/cross/warn symbol; aggregated `issues` list is printed at the end.

Cell 3 (Cell 4, Configuration): defines `Config` dataclass, resolves input file and model source, builds the singleton `CONFIG` instance, prints input/output/limit/model.

Cell 4 (Cell 5, Helpers and pipeline) defines:
- `RAG_SCHEMA_HINT` — eleven required fields (`english_summary`, `legal_topic`, `legal_question`, `legal_rule`, `court_holding`, `factual_context`, `english_legal_concepts`, `search_keywords`, `natural_language_queries`, `paragraph_role`, `outcome_signal`).
- `SYSTEM_PROMPT` — deterministic Swiss legal JSON extraction prompt; English-only, "Do not invent article numbers, statutes, laws, or case citations", "No markdown. No commentary. No chain-of-thought."
- Multilingual regex set (DE/FR/IT) for cost/procedural paragraphs, remittal, inadmissibility, dismissed, granted, partial outcomes; BGE and docket reference patterns; article-head and generated-article patterns.
- Utilities `_txt`, `_list`, `_term_values`, `_stub`.
- Deterministic classifiers: `deterministic_disposition` (remitted / inadmissible / dismissed / granted+partial), `deterministic_fallback`, `auto_classify` (handles notification paragraphs, dispositions, short fragments <50 chars, cost/procedural paragraphs).
- `validate_and_normalize_enrichment` — enforces field presence, length caps (`english_summary`<=320, `legal_topic`<=140, `legal_question`<=260, `legal_rule`<=320, `court_holding`<=260, `factual_context`<=260, lists capped at 6/8/3 items), forces `paragraph_role` into `ROLE_VALUES` (`holding|reasoning|background|cost|procedural|disposition|standard_of_review|obiter`) and `outcome_signal` into `OUTCOME_VALUES` (`granted|dismissed|inadmissible|remitted|partial|none`).
- `validate_reference_grounding` — rejects model output containing article ids, BGE refs, or docket refs not appearing in the reference blob (built from citation/court_base/text/legal_area/law_codes/statutes_cited/court_cases_cited).
- `extract_first_json_object` / `clean_model_json` / `parse_output_text` — strip `</think>` and fenced code, recover the first balanced JSON object, then validate + ground-check.
- `build_user_prompt` — wraps the card in `<record><metadata_json>...</metadata_json><paragraph_original_language>...</paragraph_original_language></record>` plus a schema echo and short rule list, truncating text to `text_chars`.
- I/O helpers `stream_input`, `count_lines`, `read_checkpoint`, `write_jsonl`, `write_failed`.
- `QwenGenerator` class: loads tokenizer + `AutoModelForCausalLM` with `dtype=float16`, `device_map="auto"`, `low_cpu_mem_usage=True`, left-padding; supports optional `BitsAndBytesConfig` 4-bit only when `use_bnb_4bit` is set and the model isn't AWQ. On load failure, raises a `RuntimeError` instructing the user to run Cell 2 (`!pip install -U transformers accelerate safetensors tqdm gptqmodel`) and confirm the local AWQ model input. `render_prompt` calls `apply_chat_template(..., enable_thinking=False)` with a TypeError fallback that appends `/no_think`. `generate_raw` explicitly type-checks that all items are dicts and raises `TypeError(f"generate_raw expected card dictionaries, got {bad}")` if not — this is the batchfix in the file name.
- `validate_config` — sanity gates on batch_size, limit, text_chars, max_new_tokens, and refuses `use_bnb_4bit=True` with an AWQ model_id.
- `main()` — sets up output/failed/checkpoint files (clears them when `reset_output`), reads checkpoint (resetting if stale), computes stop_before, instantiates `QwenGenerator`, opens `tqdm` progress, iterates `stream_input`:
  - For each card, if `auto_classify` returns a stub, flush pending batch, normalise the stub, write success, and continue.
  - Otherwise append `(line_idx, card)` to `pending`; when full, `flush_batch` strips line indices and calls `generator.generate_raw(batch_cards, max_new_tokens=...)`. Per row: parse + validate + ground-check; on failure, optionally retry with `retry_max_new_tokens` and a guard message; on terminal failure, either write a `deterministic_fallback` (when `fallback_on_fail`) or record into the failed JSONL.
  - Updates checkpoint after each successful write; logs batch throughput every `progress_every_batches`.
- Cell prints `All helpers and model class defined — ready.`

Cell 6 (Cell 6): invokes `main()`.

Cell 7: empty.

## Results

Verbatim outputs captured in the notebook:

Cell 2 (Smoke Test) stream output:
```
================================================================
Smoke Test
================================================================
[ok]  Python 3.12.12
[ok]  CUDA available
[ok]  GPU 0: Tesla T4  VRAM=15.6 GB
[ok]  GPU 1: Tesla T4  VRAM=15.6 GB
[ok]  transformers==5.7.0
[ok]  accelerate==1.13.0
[ok]  tqdm==4.67.3
[ok]  Input file: /kaggle/input/datasets/samiulislam180041221/swiss-court-authority-cards-rag-targets/court_authority_cards_v4_target_cards.jsonl
     size=1029 MB  lines=363,258
     first card citation: BGE 139 I 2 E. 5.7
     fields: ['_enrichment_target', 'authority_role', 'citation', 'court_base', 'court_cases_cited', 'family', 'is_notification_paragraph', 'issue_labels_en', 'language', 'law_codes', 'legal_area', 'matched_terms_multilingual', 'pattern', 'provenance', 'retrieval_text_en', 'statutes_cited', 'structural', 'subfamily', 'summary_en_proxy', 'text_excerpt_original']
[warn]  _enrichment_target flag present
[ok]  HuggingFace reachable (Internet is ON)

Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

[ok]  Tokenizer OK -- 'Swiss Federal Tribunal' -> 4 tokens
[ok]  Chat template renders OK (123 chars)
[ok]  /kaggle/working is writable

================================================================
[ok]  All checks passed -- ready to run enrichment!
================================================================
```

Cell 3 (Configuration) stream output:
```
Input  : /kaggle/input/datasets/samiulislam180041221/swiss-court-authority-cards-rag-targets/court_authority_cards_v4_target_cards.jsonl
Output : /kaggle/working/court_authority_cards_rag_targets_qwen3_8b.jsonl
Limit  : 50 cards (set 0 for full run)
Model  : Qwen/Qwen3-8B-AWQ
```
Note: model resolved to the HF fallback id, indicating the local Kaggle AWQ model path was not attached in this run.

Cell 4 (helpers) stream output: `All helpers and model class defined -- ready.`

Cell 5 (AWQ backend sanity) stream output: this output actually shows a long `pip install autoawq` "Requirement already satisfied" log for `autoawq 0.2.9`, `torch 2.10.0+cu128`, `transformers 5.7.0`, etc. — this is leftover output from a previous version of the cell, not the new `transformers/gptqmodel` import probe currently in the source.

Cell 6 (Run): partial stream output then a fatal error:
```
Input      : /kaggle/input/datasets/samiulislam180041221/swiss-court-authority-cards-rag-targets/court_authority_cards_v4_target_cards.jsonl
Output     : /kaggle/working/court_authority_cards_rag_targets_qwen3_8b.jsonl
Failures   : /kaggle/working/court_authority_cards_rag_targets_qwen3_8b_failed_cards.jsonl
Checkpoint : /kaggle/working/rag_targets_qwen3_8b_checkpoint.txt
Model      : Qwen/Qwen3-8B-AWQ
Start      : 0
Total      : 363,258
To process : 50
Batch size : 8
[model] loading tokenizer: Qwen/Qwen3-8B-AWQ
[model] loading weights: Qwen/Qwen3-8B-AWQ

[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
```
Then `RuntimeError: AWQ load failed. Run: !pip install autoawq  or set CONFIG.model_id='Qwen/Qwen3-8B' + CONFIG.use_bnb_4bit=True.` chained from `ImportError: Loading an AWQ quantized model requires gptqmodel. Please install it with pip install gptqmodel` raised by `transformers/quantizers/quantizer_awq.py:50`. Traceback frames cover `QwenGenerator.__init__` -> `AutoModelForCausalLM.from_pretrained` -> `get_hf_quantizer` -> `validate_environment`.

Cell 7: no output (empty cell).

## Summary

This Kaggle notebook enriches the 363,258-row `court_authority_cards_v4_target_cards.jsonl` corpus with deterministic JSON RAG fields (`english_summary`, `legal_topic`, etc.) using Qwen3-8B-AWQ loaded via Transformers + gptqmodel, with an auto-classifier short-circuit for procedural/disposition/cost paragraphs and strict reference-grounding validation that rejects invented article/BGE/docket citations. The variant's name signals its fix: `QwenGenerator.generate_raw` now asserts every batched item is a `dict` and `flush_batch` strips `(line_idx, card)` tuples before generation, so the prompt builder cannot receive tuples. Smoke test passed on a 2x Tesla T4 Kaggle session with the expected dataset present. The configuration cell resolved `Model` to the HF fallback `Qwen/Qwen3-8B-AWQ` rather than the local `/kaggle/input/models/...` path, indicating the AWQ model dataset was not attached for this run. Cell 6 (`main()`) loaded the tokenizer but failed at weight load with `ImportError: Loading an AWQ quantized model requires gptqmodel` from `quantizer_awq.py`, surfaced as the notebook's own `RuntimeError` from `QwenGenerator.__init__`; no enrichment rows were written in this session.
