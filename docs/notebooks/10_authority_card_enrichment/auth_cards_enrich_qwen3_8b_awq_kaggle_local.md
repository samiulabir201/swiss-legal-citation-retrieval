# auth_cards_enrich_qwen3_8b_awq_kaggle_local

**Path:** `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen3_8b_awq_kaggle_local.ipynb`

## Configuration

Notebook header (Cell 0, markdown):

- Title: "Qwen3-8B-AWQ — Swiss Court Authority Card Enrichment"
- Purpose: enrich `court_authority_cards_v4_target_cards.jsonl` (363,258 target cards) with Qwen3-8B-AWQ on a Kaggle GPU; outputs land in `/kaggle/working/` with a checkpoint file for resume.
- Model dataset path: `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1`.
- Kaggle notebook settings table:
  - Accelerator: GPU T4 × 2 (or P100)
  - Internet: ON for Cell 2 package install
  - Input dataset: `samiulislam180041221/swiss-court-authority-cards-rag-targets`
  - Model dataset: Qwen3 8B AWQ mounted at `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1`
- Expected outputs in `/kaggle/working/`:
  - `court_authority_cards_rag_targets_qwen3_8b.jsonl`
  - `court_authority_cards_rag_targets_qwen3_8b_failed_cards.jsonl`
  - `rag_targets_qwen3_8b_checkpoint.txt`

Cell 1 — Install (run once per Kaggle session):

```python
!pip install -q -U --no-cache-dir transformers accelerate safetensors tqdm gptqmodel
```

The model weights load from the attached Kaggle model dataset; `gptqmodel` is the AWQ backend that current Transformers uses. The notebook comment explicitly says: do NOT install `autoawq` unless intentionally switching away from `AutoModelForCausalLM` + Transformers AWQ loading.

Cell 3 — `Config` dataclass (key fields):

- Files (Kaggle paths):
  - `input_file`: resolved by `first_existing_target_file()` (probes fixed Kaggle locations then glob fallback under `/kaggle/input`)
  - `output_file`: `/kaggle/working/court_authority_cards_rag_targets_qwen3_8b.jsonl`
  - `failed_file`: `/kaggle/working/court_authority_cards_rag_targets_qwen3_8b_failed_cards.jsonl`
  - `checkpoint_file`: `/kaggle/working/rag_targets_qwen3_8b_checkpoint.txt`
- Model:
  - `model_id`: resolved by `first_existing_model_source()` — prefers local `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1`, falls back to HF id `Qwen/Qwen3-8B-AWQ`
  - `trust_remote_code = True`
  - `torch_dtype = "float16"` (T4/P100-safe; avoid bf16 on Kaggle T4)
  - `device_map = "auto"`
  - `attn_implementation = None`
  - `use_bnb_4bit = False` (the comment notes bitsandbytes 4-bit is for the non-AWQ base model only)
- Run control:
  - `limit = 50` (50 for smoke/quality check; 0 for the full 363,258-card run)
  - `reset_output = True`
  - `batch_size = 8`
  - `progress_every_batches = 10`
- Generation:
  - `text_chars = 700`
  - `max_input_tokens = 1600`
  - `max_new_tokens = 224`
  - `retry_max_new_tokens = 384`
  - `do_sample = False`, `temperature = 0.0`, `top_p = 1.0`, `repetition_penalty = 1.05`
- Reliability:
  - `auto_classify = True`
  - `retry_bad_outputs = True`
  - `fallback_on_fail = False`
  - `validate_reference_grounding = True`
- Env: `os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")`; `IS_KAGGLE = Path("/kaggle/working").exists()`; `DATASET_SLUG = "swiss-court-authority-cards-rag-targets"`.

Schema / prompt constants (Cell 4):

- `RAG_SCHEMA_HINT` keys: `english_summary`, `legal_topic`, `legal_question`, `legal_rule`, `court_holding`, `factual_context`, `english_legal_concepts` (list), `search_keywords` (list), `natural_language_queries` (list), `paragraph_role` (one of `holding|reasoning|background|cost|procedural|disposition|standard_of_review|obiter`), `outcome_signal` (one of `granted|dismissed|inadmissible|remitted|partial|none`).
- `ROLE_VALUES` = the 8 paragraph_role values above.
- `OUTCOME_VALUES` = the 6 outcome_signal values above.
- `SYSTEM_PROMPT`: instructs a deterministic Swiss legal JSON extraction engine over a German/French/Italian Swiss Federal Tribunal paragraph plus deterministic metadata; output exactly one JSON object using concise English legal terminology, only using statutes/articles/citations present in paragraph or metadata, empty fields preferred over invention, no markdown / commentary / chain-of-thought.

## Data

- Input file (target-card JSONL): `court_authority_cards_v4_target_cards.jsonl`, resolved at runtime to `/kaggle/input/datasets/samiulislam180041221/swiss-court-authority-cards-rag-targets/court_authority_cards_v4_target_cards.jsonl` (size 1029 MB, 363,258 lines per smoke-test output).
- Card fields observed on first record (smoke output):

```
['_enrichment_target', 'authority_role', 'citation', 'court_base',
 'court_cases_cited', 'family', 'is_notification_paragraph',
 'issue_labels_en', 'language', 'law_codes', 'legal_area',
 'matched_terms_multilingual', 'pattern', 'provenance',
 'retrieval_text_en', 'statutes_cited', 'structural', 'subfamily',
 'summary_en_proxy', 'text_excerpt_original']
```

- First card citation observed: `BGE 139 I 2 E. 5.7`.
- `_enrichment_target` flag check marked with the warning symbol (warn_only) — present on the first card.
- Outputs (Kaggle working dir, written per `Config`):
  - `court_authority_cards_rag_targets_qwen3_8b.jsonl` (each input card with appended `rag_enrichment` dict)
  - `court_authority_cards_rag_targets_qwen3_8b_failed_cards.jsonl` (failures: `{line_idx, citation, court_base, stage, error, raw_output, card}`)
  - `rag_targets_qwen3_8b_checkpoint.txt` (resume offset)

## Pipeline

Cells (in execution order; cell-number labels in source comments are off by one because Cell 0 is markdown):

1. Cell 0 (markdown) — README/quick-start as listed under Configuration above.
2. Cell 1 — `pip install -q -U --no-cache-dir transformers accelerate safetensors tqdm gptqmodel`.
3. Cell 2 — Smoke Test. Source-labelled `# Cell 3 — Smoke Test`. Checks Python version (>=3.9); CUDA + per-GPU VRAM (warn if <14 GB, fail if <10 GB); package versions (`transformers>=4.51.0`, `accelerate>=0.26.0`, `tqdm>=4.0.0`); locates the input JSONL via fixed paths then recursive glob, prints size/lines, reads first card to print citation/fields and verifies `_enrichment_target` flag (warn_only); checks Kaggle AWQ model path existence and reports `config.json`/`tokenizer_config.json` and safetensors shard count, else warns; verifies `gptqmodel` import; loads tokenizer from `MODEL_SOURCE` (Kaggle path or `Qwen/Qwen3-8B-AWQ` fallback), tokenizes `"Swiss Federal Tribunal"`, renders a chat template with `enable_thinking=False` (with `TypeError` fallback); verifies `/kaggle/working` writable. Prints a final pass/fail summary listing every issue.
4. Cell 3 — `Config` dataclass + helpers `first_existing_target_file()`, `first_existing_model_source()`. Instantiates `CONFIG` and prints `Input`, `Output`, `Limit`, `Model`; warns if the Kaggle local model path is missing.
5. Cell 4 — Helpers, model, and enrichment pipeline. Subsystems:
   - Schema/prompt constants (`RAG_SCHEMA_HINT`, `REQUIRED_FIELDS`, `ROLE_VALUES`, `OUTCOME_VALUES`, `SYSTEM_PROMPT`).
   - Regex patterns: `COST_PROC_RE`, `REMITTAL_RE`, `INADMISSIBLE_RE`, `DISMISSED_RE`, `GRANTED_RE`, `PARTIAL_RE`, `BGE_REF_RE`, `DOCKET_REF_RE`, `ART_HEAD_RE`, `ART_GENERATED_RE`, `ART_NUM_RE` — multilingual DE/FR/IT triggers for procedural costs, remittal, inadmissibility, dismissal, grant, partial wording, plus BGE / docket / article-number capture.
   - Utilities: `_txt`, `_list`, `_term_values`, `_stub` (canonical stub dict with length caps for each schema field).
   - Deterministic auto-classifiers (no model call): `deterministic_disposition` (remittal/inadmissible/dismissed/granted; partial detection inside granted), `deterministic_fallback` (composes summary from `summary_en_proxy`/`legal_area`/`issue_labels_en`/`law_codes`/`statutes_cited`/`court_cases_cited`/`matched_terms_multilingual`), `auto_classify` (notification paragraphs, then disposition, then short fragments <50 chars, then court-cost paragraphs via `COST_PROC_RE`).
   - Validation: `validate_and_normalize_enrichment` (enforces required keys, per-field length caps `english_summary=320`, `legal_topic=140`, `legal_question=260`, `legal_rule=320`, `court_holding=260`, `factual_context=260`; list-field caps `english_legal_concepts=6@80`, `search_keywords=8@80`, `natural_language_queries=3@180`; coerces `paragraph_role` and `outcome_signal` to vocabulary; sets `method`).
   - Reference grounding: `_reference_blob`, `_article_ids_from_blob`, `_generated_article_ids`, `validate_reference_grounding` — rejects any generated `art. N`, `BGE …`, or docket reference not present in the card's metadata/text blob; allows `…f` only when the compact form `art N f` appears in source.
   - JSON parsing: `extract_first_json_object` (depth-tracking with string/escape handling), `clean_model_json` (strips `</think>`, ```` ```json ```` / ```` ``` ```` fences), `parse_output_text` (combines cleanup, schema validation, reference grounding).
   - Prompt builder: `build_user_prompt` wraps card metadata + truncated paragraph (`text_chars=700`) in `<record>`/`<metadata_json>`/`<paragraph_original_language>` blocks and appends the JSON schema hint plus the four rules ("English legal terminology", "Use only references present in metadata/paragraph", "Empty is better than inventing", "No markdown").
   - I/O helpers: `stream_input`, `count_lines`, `read_checkpoint`, `write_jsonl`, `write_failed`.
   - `QwenGenerator` class: loads tokenizer (left padding; pad_token = eos if needed) and `AutoModelForCausalLM` from `config.model_id` with `dtype=torch.float16`, `device_map="auto"`, `low_cpu_mem_usage=True`, optional `attn_implementation` and `BitsAndBytesConfig` for `use_bnb_4bit`. On AWQ load failure, raises `RuntimeError` instructing to re-run Cell 2 (`gptqmodel`) and confirm the local model input path. `render_prompt` applies chat template with `enable_thinking=False` (or appends `/no_think` on `TypeError`). `generate_raw` batches prompts, truncates to `max_input_tokens`, moves to model device, calls `model.generate` greedy by default (returns only continuation text).
   - `validate_config`: enforces `batch_size>=1`, `limit>=0`, `text_chars>=200`, `max_new_tokens>=80`, and forbids `use_bnb_4bit=True` with an AWQ `model_id`.
   - `main`: validates config; checks input exists; creates output/failed/checkpoint parents; if `reset_output`, deletes the three output files. Reads checkpoint, counts input lines, resets stale checkpoint, computes `stop_before = min(total, start+limit)` (or `total`). Constructs `QwenGenerator`, opens output/failed files in append mode, opens optional `tqdm` bar. Streams input from offset; per card calls `auto_classify` and, if hit, flushes any pending batch and writes the deterministic enrichment; otherwise appends to `pending` and calls `flush_batch` when `batch_size` reached. `flush_batch` runs `generate_raw`, parses each output, on parse failure runs a single retry with `retry_max_new_tokens` and an extra guard message; on retry failure either writes a deterministic fallback (if `fallback_on_fail`) or records the failure JSON and advances the checkpoint. Counters: `llm_ok`, `auto_count`, `fail_count`, `fallback_count`. Prints batch stats every `progress_every_batches`. End of run prints `Done processed / LLM ok / Auto-classified / Failed captured / Fallback written / Elapsed / Rate / Output / Failures`.
6. Cell 5 — AWQ backend sanity check: imports `transformers` and `gptqmodel`, prints `Transformers:` and `gptqmodel: OK`.
7. Cell 6 — Run enrichment: `main()`.
8. Cell 7 — empty.

## Results

Cell 2 (Smoke Test) output (verbatim):

```
================================================================
Smoke Test
================================================================
✓  Python 3.12.12
✓  CUDA available
✓  GPU 0: Tesla T4  VRAM=15.6 GB
✓  GPU 1: Tesla T4  VRAM=15.6 GB
✓  transformers==5.7.0
✓  accelerate==1.13.0
✓  tqdm==4.67.3
✓  Input file: /kaggle/input/datasets/samiulislam180041221/swiss-court-authority-cards-rag-targets/court_authority_cards_v4_target_cards.jsonl
     size=1029 MB  lines=363,258
     first card citation: BGE 139 I 2 E. 5.7
     fields: ['_enrichment_target', 'authority_role', 'citation', 'court_base', 'court_cases_cited', 'family', 'is_notification_paragraph', 'issue_labels_en', 'language', 'law_codes', 'legal_area', 'matched_terms_multilingual', 'pattern', 'provenance', 'retrieval_text_en', 'statutes_cited', 'structural', 'subfamily', 'summary_en_proxy', 'text_excerpt_original']
⚠  _enrichment_target flag present
✓  HuggingFace reachable (Internet is ON)
```

Followed by a Hub auth warning:

```
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

And the remainder:

```
✓  Tokenizer OK — 'Swiss Federal Tribunal' → 4 tokens
✓  Chat template renders OK (123 chars)
✓  /kaggle/working is writable

================================================================
✓  All checks passed — ready to run enrichment!
================================================================
```

Cell 3 (Config) output:

```
Input  : /kaggle/input/datasets/samiulislam180041221/swiss-court-authority-cards-rag-targets/court_authority_cards_v4_target_cards.jsonl
Output : /kaggle/working/court_authority_cards_rag_targets_qwen3_8b.jsonl
Limit  : 50 cards (set 0 for full run)
Model  : Qwen/Qwen3-8B-AWQ
```

(Note: `Model` resolved to the HF id `Qwen/Qwen3-8B-AWQ`, indicating the Kaggle local AWQ model path was not attached at the time of this run.)

Cell 4 (Helpers + model class) output:

```
All helpers and model class defined — ready.
```

Cell 5 (AWQ backend sanity) output is a `pip` resolution log showing `autoawq 0.2.9` already installed alongside `transformers 5.7.0`, `tokenizers 0.22.2`, `accelerate 1.13.0`, `huggingface-hub 1.13.0`, `safetensors 0.7.0`, `gptqmodel`/`torch 2.10.0+cu128`/`triton 3.6.0`, etc. (full dependency tree of `autoawq`'s install). The cell's own `print` lines for `Transformers:` and `gptqmodel: OK` are not visible in the captured outputs.

Cell 6 (`main()`) output:

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

Then it raised:

```
ImportError: Loading an AWQ quantized model requires gptqmodel. Please install it with `pip install gptqmodel`
```

re-raised by `QwenGenerator.__init__` as:

```
RuntimeError: AWQ load failed. Run: !pip install autoawq  or set CONFIG.model_id='Qwen/Qwen3-8B' + CONFIG.use_bnb_4bit=True.
```

Traceback frames hit `AutoModelForCausalLM.from_pretrained` → `quantizer_awq.validate_environment` → `is_gptqmodel_available()` returning False inside the live kernel even though Cell 5's `pip` log shows `autoawq 0.2.9` present. No enrichment cards were processed; the output JSONL, failed JSONL, and checkpoint file from this run are not produced beyond the file creation that happens before `QwenGenerator` is instantiated (`out_f` and `failed_f` are opened only after the model loads successfully).

Cell 7 has no source and no outputs.

## Summary

Configures a Kaggle GPU notebook to enrich 363,258 Swiss Federal Tribunal authority-card paragraphs (`court_authority_cards_v4_target_cards.jsonl`) with a Qwen3-8B-AWQ model loaded via Transformers + `gptqmodel`, writing per-card RAG dicts (English summary, topic, question, rule, holding, facts, concepts, keywords, NL queries, paragraph_role, outcome_signal) into `/kaggle/working/court_authority_cards_rag_targets_qwen3_8b.jsonl` with checkpoint and failed-card files. The pipeline combines deterministic auto-classification (notifications, dispositions via DE/FR/IT regex, short fragments, court-cost paragraphs) with batched LLM generation (batch_size=8, greedy, `max_new_tokens=224`, retry at 384 with grounding guard), strict schema/length validation, and a reference-grounding check that rejects any generated BGE, docket, or article number not present in the card metadata or paragraph text. The Cell-2 smoke test on Kaggle confirmed Python 3.12.12, dual Tesla T4 (15.6 GB each), `transformers==5.7.0` / `accelerate==1.13.0` / `tqdm==4.67.3`, the 1029 MB / 363,258-line input, working tokenizer + chat template (`Swiss Federal Tribunal` → 4 tokens, 123-char render), and writable working dir. The actual run in Cell 6 failed at model load with `ImportError: Loading an AWQ quantized model requires gptqmodel`, re-raised by the notebook as `RuntimeError: AWQ load failed`, despite Cell 5's pip log showing `autoawq 0.2.9` already installed — the AWQ quantizer in `transformers 5.7.0` looks specifically for `gptqmodel`, and the model id used was the HF fallback `Qwen/Qwen3-8B-AWQ` (the Kaggle local path under `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1` was not detected). No enriched cards or failure records were written from this execution.
