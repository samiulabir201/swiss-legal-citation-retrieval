# kaggle_10_test_for_swiss_law

**Path:** `e:\swiss_citation_extraction\notebooks\13_misc_kaggle_and_utilities\kaggle_10_test_for_swiss_law.ipynb`

## Configuration

Engine and model:
- `engine = "vllm"` (fallback `"transformers"` class also defined).
- `model_name = "/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1"` (Qwen3-8B AWQ).
- `quantization = "awq_marlin"`, `gpu_memory_utilization = 0.78`, `max_model_len = 4096`, `enforce_eager = True`, `max_num_seqs = 8`, `disable_custom_all_reduce = True`, `force_triton_attention = True` (sets `VLLM_ATTENTION_BACKEND=TRITON_ATTN` and passes an explicit `AttentionConfig(backend="TRITON_ATTN")` when supported).
- `use_structured_outputs = False` (documented to avoid `'dict' object has no attribute '_backend'` failure observed on Kaggle vLLM v0.20; prompt-constrained JSON + repair is used instead).

GPU mode:
- `gpu_mode = "single"` -> sets `CUDA_VISIBLE_DEVICES=0`, `tensor_parallel_size = 1`.
- Alternative `"tp2"` documented for sharding one model across both T4s (slower in practice; production recommendation is two independent Kaggle sessions sharded by `start`/`limit`).

Generation:
- `batch_size = 4`, `max_new_tokens = 768`, `retry_max_new_tokens = 1024`, `temperature = 0.0`, `top_p = 1.0`, `repetition_penalty = 1.02`, `enable_thinking = False`, `max_retries = 2`.

Input selection:
- `input_csv = "/kaggle/input/competitions/llm-agentic-legal-information-retrieval/court_considerations.csv"` (fallback `court_considerations.csv`).
- `n_rows = 10`, `start = 0`, `limit = 10`, `sample_random = False`, `random_seed = 42`, `min_text_chars = 250`, `max_text_chars = 4500`.
- `manual_records: List[Dict[str, str]] = None` — if non-empty, CSV is ignored and rows come from `(citation, text)` pairs supplied inline.

Outputs (under `/kaggle/working`):
- `output_jsonl = "enriched_court_citations_10.jsonl"`.
- `output_preview_csv = "enriched_court_citations_10_preview.csv"`.
- `output_failures_jsonl = "enriched_court_citations_10_failures.jsonl"` (deleted if no fallbacks occurred).

Setup cell installs `transformers>=4.45.0 accelerate safetensors pandas tqdm gptqmodel` and `vllm>=0.6.0`, then uninstalls `flashinfer` / `flashinfer-python`.

## Data

Source: `court_considerations.csv` from the Kaggle competition `llm-agentic-legal-information-retrieval`, resolved via `resolve_input_path` which checks `/kaggle/input/...`, `/kaggle/working/`, `/mnt/data/` and a globbed fallback under `/kaggle/input`.

- Reported shape: `(2476315, 2)`, columns `['citation', 'text']`.
- `pick_column` infers `citation_col = "citation"` and `text_col = "text"`.
- Rows are filtered to non-null citation+text, text trimmed length >= `min_text_chars=250`; if too few survive, falls back to texts longer than 30 chars.
- Deterministic shard: `iloc[start:start+limit]`, reset and stamped with `_source_row` (original DataFrame index).
- The 10 selected rows are all from `BGE 139 I 2` (various Erwägung subsections E. 2, 5.1, 5.3, 5.5, 5.6, 5.7, 5.7.1, 5.7.2, 7.1 ×2), text lengths 278–2405 chars.

## Pipeline

1. Cell 0 (install): `transformers/accelerate/safetensors/pandas/tqdm/gptqmodel`, `vllm`, uninstall `flashinfer*`. Restart of kernel is requested.
2. Cell 1 (imports + GPU probe): stdlib + `pandas`, `tqdm`, `torch`; prints CUDA availability and per-GPU memory.
3. Cell 2 (config): defines `Config` dataclass and applies `gpu_mode`; checks `model_path` exists.
4. Cell 3 (input): `build_work_df_from_csv` or `build_work_df_from_manual` to produce `work_df` plus inferred `citation_col`, `text_col`.
5. Cell 4 (schema + prompt): defines `ROLE_VALUES`, `OUTCOME_VALUES`, `AUTHORITY_VALUES`, `GENERIC_CONCEPTS`, and a JSON schema `ENRICHMENT_JSON_SCHEMA` with required fields `legal_area`, `legal_domain_path`, `topic`, `subtopic`, `micro_topic`, `concepts_en`, `terms_original`, `statute_anchors`, `case_anchors`, `doctrinal_rule`, `legal_test`, `fact_pattern_tags`, `procedural_context`, `paragraph_role`, `authority_role`, `outcome_signal`, `specificity_score`. Schema enforces lengths/maxima and `additionalProperties: False`. System prompt instructs the model to produce a deterministic, query-neutral fingerprint, forbids generated user questions and summaries, requires `terms_original` in source language, and grounds `statute_anchors`/`case_anchors` only on what is explicit in the text. `trim_text` keeps head+tail halves around `[TRUNCATED]`. `build_prompt` produces both a first-pass user prompt and a repair prompt that includes the prior invalid output and error.
6. Cell 5 (parsing/normalization): `find_balanced_json_object` (strips ```` ```json ```` fences, depth-balanced scan), `parse_json_lenient` (json -> trailing-comma cleanup + smart-quote normalization -> `ast.literal_eval`). `normalize_enrichment` clamps strings, dedupes, drops generic concepts, augments `statute_anchors`/`case_anchors` with regex extraction (`Art. <n> [Abs. ...] [lit. ...] [ABBR]`, `BGE <n> <roman> <n> [E. ...]`, `Urteil 1C_xxx/YYYY`), removes self-reference of the input citation/court_base, normalizes enums with alias map (`remanded->remitted`, `partially_granted->partial`, etc.), and produces a deterministic 17-key record. `validate_enrichment` reports missing required keys / bad enums / forbidden fields. `deterministic_fallback` builds a minimal record marked `manual review required` carrying extracted statutes/cases/capitalized terms and `_fallback_reason`. `build_retrieval_views` flattens 8 views: `semantic_concepts_en`, `topic_path`, `original_terms_view`, `statute_anchor_view`, `case_anchor_view`, `legal_rule_view`, `fact_pattern_view`, `raw_context` (trimmed to 1200 chars).
7. Cell 6 (engine): `VllmJsonEngine` loads tokenizer + vLLM with `AttentionConfig(backend=TRITON_ATTN)` when available, prints `Engine: vLLM` and a structured-decoding mode (`disabled` here). `_detect_structured_mode` chooses between `guided_json`, `guided_decoding`, or `structured_outputs` if/when enabled. `_chat_to_prompt` uses `apply_chat_template` with `enable_thinking=False`. `TransformersJsonEngine` mirrors the same interface using `AutoModelForCausalLM.generate`. `load_engine(cfg)` returns the requested engine.
8. Cell 7 (runner): mini-batches of `batch_size=4`. For each batch, `engine.generate_prompts` runs first-pass; per-row `try_enrich_single` parses+normalizes+validates and, on failure, generates a repair prompt (up to `max_retries=2`); after final retry it emits `deterministic_fallback`. `build_output_record` wraps `{_source_row, citation, court_base, source_family="court", text, rag_enrichment, retrieval_views, enrichment_quality}` and tags `enrichment_quality.method` as either `llm_compact_legal_descriptor_enrichment` or `deterministic_fallback`. Records are written to `enriched_court_citations_10.jsonl`, flattened into `enriched_court_citations_10_preview.csv`; failure-only JSONL is removed when empty.
9. Cell 8 (inspect): prints the first record's full JSON (truncated to 8000 chars).
10. Cell 9 (QC): `check_no_forbidden_fields` recursively walks each record looking for any of `{query_phrases_en, summary_en, natural_language_queries, english_summary}`; per-row `concept_count`, `terms_original_count`, self-case-anchor presence, `specificity_score`.

## Results

Cell 0 (install): logs many pip dependency conflicts (protobuf 7.34.1 vs google-cloud-* / tensorflow, numpy 2.2.6 vs tensorflow 2.19.0, `vllm 0.20.0 requires flashinfer-python==0.6.8.post1, which is not installed`), then prints `Setup done. Now restart the Kaggle session/kernel, then run from the next cell.`

Cell 1 (imports + GPU probe):
```
Python imports OK
Torch: 2.11.0+cu130
CUDA available: True
GPU 0: Tesla T4 total=14.56 GiB free=14.46 GiB
GPU 1: Tesla T4 total=14.56 GiB free=14.46 GiB
```

Cell 2 (config): full `Config` dict printed; `Model path OK: /kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1`.

Cell 3 (input):
```
Using input: /kaggle/input/competitions/llm-agentic-legal-information-retrieval/court_considerations.csv
Shape: (2476315, 2)
Columns: ['citation', 'text']
citation_col: citation
text_col: text
Selected rows: 10
```
The 10 selected rows are `BGE 139 I 2 E. 2`, `E. 5.1`, `E. 5.3`, `E. 7.1`, `E. 7.1`, `E. 5.5`, `E. 5.6`, `E. 5.7`, `E. 5.7.1`, `E. 5.7.2` with text lengths 885 / 437 / 286 / 1493 / 690 / 904 / 2405 / 278 / 1529 / 1367.

Cell 4 and Cell 5 produce no stdout (definition-only).

Cell 6 (engine load): `[vLLM] using explicit AttentionConfig backend=TRITON_ATTN`. vLLM v0.20.0 log shows resolved architecture `Qwen3ForCausalLM`, `max_model_len=4096`, `quantization=awq_marlin`, `enforce_eager=True`, chunked-prefill `max_num_batched_tokens=8192`, async scheduling enabled, cudagraph disabled under eager, custom fusions `norm_quant, act_quant`. NIXL warnings emitted but harmless. Spawn multiprocessing was forced because CUDA was initialized.

Cell 7 (runner):
```
first-pass batches: 0/3
Generated 10 rows in 91.9s
Rows/s: 0.109
Status counts: {'ok': 10}
JSONL: /kaggle/working/enriched_court_citations_10.jsonl
Preview CSV: /kaggle/working/enriched_court_citations_10_preview.csv
```
Preview DataFrame (truncated print): all 10 rows have `generation_status = ok`. Inferred `legal_area` mix: Administrative Law x6, Constitutional Law x3, Land Use and Planning x1. Topics include `Administrative Review`, `Constitutional Review of Administrative Acts`, `Initiative Validity`, `Planning Initiatives`, `Municipal Decision-Making`, `Initiative Implementation`, `Land Use Regulation`, `Zoning and Land Use Regulation`. `concepts_en` and `terms_original` show 3–8 short entries per row, e.g. row 0 `terms_original = ["Rueckweisung an die Vorinstanz", "Neubehandlung", "Sachverhaltsabklaerung", "Neubeurteilung", "Verwaltungsgericht", "Beschwerde"]`. Doctrinal rules and legal tests are present per row.

Cell 8 (full first record): includes for `BGE 139 I 2 E. 2` —
- `rag_enrichment.legal_area = "Administrative Law"`, `topic = "Administrative Review"`, `subtopic = "Remand for Further Proceedings"`, `paragraph_role = "reasoning"`, `authority_role = ["legal_test", "standard_of_review"]`, `outcome_signal = "remitted"`, `specificity_score = 0.9`.
- `statute_anchors = ["Verwaltungsverfahrensgesetz"]`, `case_anchors = []`.
- `retrieval_views` populated for all 8 views; `case_anchor_view = "BGE 139 I 2 E. 2 BGE 139 I 2"` (self-citation plus court_base).
- `enrichment_quality = {method: llm_compact_legal_descriptor_enrichment, generation_status: ok, question_generation_used: false, summary_generation_used: false, grounded_references_only: true, has_specific_topic: true, has_original_language_terms: true, has_statute_anchor: true, has_case_anchor: false, low_value_paragraph: false, attempt_count: 1}`.

Cell 9 (QC): all 10 rows are `status=ok`, `forbidden_fields=[]`, `self_case_anchor_present=False`, `concept_count` 3–4, `terms_original_count` 5–8, `specificity_score` between 0.80 and 0.90 (mostly 0.85–0.90). Totals: `Forbidden field rows: 0`, `Self case anchor rows: 0`, `Fallback rows: 0`.

## Summary

This Kaggle smoke-test notebook enriches 10 Swiss Bundesgericht consideration paragraphs from `court_considerations.csv` (all sampled from `BGE 139 I 2`) into a 17-field "query-neutral legal fingerprint" using Qwen3-8B AWQ via vLLM 0.20 on a single Kaggle T4. The pipeline deliberately avoids structured/guided decoding (a documented Kaggle vLLM bug causing `'dict' object has no attribute '_backend'`) and instead drives the model with a strict JSON schema in the prompt plus a hardened parser, normalizer, regex-based statute/case-anchor extractor, alias-aware enum normalizer, retry-with-repair loop (`max_retries=2`), and a deterministic fallback record. The first-pass batch (size 4) completed all 10 rows in 91.9 s (~0.11 rows/s) with status `ok` for every row, no forbidden fields, no self-case anchors, and `specificity_score` 0.80–0.90; outputs are JSONL + flattened preview CSV in `/kaggle/working`. Cell 8 confirms the structured per-row product (rag_enrichment + 8 retrieval_views + enrichment_quality flags) for `BGE 139 I 2 E. 2`. The notebook also documents production-throughput guidance (two independent `gpu_mode="single"` Kaggle sessions sharded by `start`/`limit`) rather than `tp2`.
