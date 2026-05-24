# enrich_laws_de_qwen3_8b_kaggle_optimized

**Path:** `e:\swiss_citation_extraction\notebooks\07_law_de_enrichment\enrich_laws_de_qwen3_8b_kaggle_optimized.ipynb`

## Configuration

- **Model:** `Qwen/Qwen3-8B-AWQ` loaded via vLLM (AWQ-Marlin quantization, FP16 KV cache).
- **Hardware:** Single NVIDIA RTX PRO 6000 Blackwell Server Edition (95.59 GiB; compute capability 12.0). Detected as Colab+Kaggle hybrid runtime with `BASE_DIR = /content/drive/MyDrive/swiss_law`.
- **Attention backend:** Auto-selected (vLLM picks FA4 on SM 12.x). FlashInfer 0.6.10 imports successfully but is marked `FLASHINFER_USABLE = False` because FlashInfer 0.6.x has no SM 12.x cubins; install logic targets `cu130`. `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `FLASHINFER_DISABLE_VERSION_CHECK=1`.
- **vLLM init (as reported by Cell 6 output):** `max_model_len=3072`, `max_num_seqs=480`, `gpu_memory_utilization=0.92`, `enable_prefix_caching=True`, `disable_custom_all_reduce=True`, `enforce_eager=False`, `kv_cache_dtype=None`, n-gram speculation off. System+schema prefix measured at ~1497 tokens (prefix-cached).
- **Config class values declared in code (Cell 2):** `max_model_len=2560`, `max_num_seqs=320`, `gpu_memory_utilization=0.95`, `max_text_chars=2500`, `submit_chunk=4096`, `temperature=0.1`, `top_p=0.9`, `repetition_penalty=1.0`, `max_new_tokens=512`, `retry_max_new_tokens=600`, `max_retries=1`, `enable_thinking=False`, `use_structured_outputs=False`, `min_text_chars=20`, `start=0`, `limit=0`, `sample_random=False`, `random_seed=42`. (The Cell 9 dump shows the values that actually ran: `max_model_len=3072`, `max_num_seqs=480`, `gpu_memory_utilization=0.92`, `max_text_chars=0`, `submit_chunk=2048`.)
- **Per-row max_tokens (`TIER_MAX_TOKENS`):** `<100 chars -> 192`, `<300 -> 256`, `<600 -> 320`, `<1500 -> 416`, `>=1500 -> 512`.
- **Schema (12 LLM fields, JSON only):** `english_summary`, `legal_rule`, `applicability_conditions`, `exceptions_or_limitations`, `legal_question`, `concepts_en`, `terms_de_to_en`, `defined_terms`, `addressees`, `sanctions_or_consequences`, `provision_role_llm`, `specificity_score`. Caps: 10 term pairs, 4 defined terms, 6 addressees, 5 conditions, 5 exceptions, 8 concepts.
- **Provision roles vocabulary:** `definition, purpose, scope, principle, right_or_entitlement, duty, prohibition, procedure, competence, sanction_or_penalty, data_reporting, fees_or_costs, transitional_or_commencement, other`. `BOILERPLATE_ROLES = {transitional_or_commencement, fees_or_costs, data_reporting}` are forced to empty rule fields.
- **Addressees controlled vocabulary** (system prompt): Federal Council, federal department or office, supervisory authority, competent cantonal authority, competent communal authority, court, public prosecutor, natural person, legal entity / undertaking, employer, employee, taxpayer, data subject, data controller or processor, market participant / operator, service provider, consumer / customer, foreign authority or international organisation.
- **Forbidden fields stripped from descriptors:** `statute_anchors`, `case_anchors`, `normalized_anchors`, `retrieval_views`, `outcome_signal`, `enrichment_quality`, `anchor_quality_flags`, `incoming_references`, `outgoing_references`, `adjacent_citations`, `law_code`, `law_title`, `enactment_date`, `enactment_year`.

## Data

- **Input:** `/content/drive/MyDrive/swiss_law/data/law_llm_input.jsonl` (188.6 MB), the slim per-row payload produced by `scripts/build_law_llm_input.py` against the full 175,933-row `laws_de.csv` corpus (Swiss federal law in DE/FR/IT — Bundesgesetze / Verordnungen / Bundesverfassung / Verträge, SR-numbered statutes).
- **Rows loaded:** 173,033 valid rows (0 skipped for missing fields, 0 below `min_text_chars=20`).
- **Text length distribution (chars):** count 173,033 / mean 242.75 / std 237.95 / min 20 / median 184 / p90 432 / p95 599 / p99 1,277.68 / max 2,500.
- **llm_priority mix:** `medium 147,941`, `high 21,490`, `low 3,602`.
- **Work slice:** all 173,033 rows (`start=0`, `limit=all`). 100% of rows flagged as exceeding `max_text_chars=0` (head+tail truncation marker applied per the deliberate "no truncation" run config).
- **Per-row payload fields kept:** `_source_row` (checkpoint key), `citation`, `law_title`, `title_section_path`, `structural` (law_code, article, units), `title_metadata` (source_type, enactment_year, enactment_date), `static_hints` (legal_area_static, domain_labels_en, provision_roles_static), `llm_priority`, `text`, `_text_len`.

## Pipeline

1. **Cell 0 (setup):** Detects Colab/Kaggle, mounts `/content/drive`, installs `vllm>=0.10.0`, `transformers>=4.51.0`, `accelerate`, `safetensors`, `pandas`, `tqdm`, `huggingface_hub`. Detects CUDA suffix (here `cu130`) and installs `flashinfer-python` + `flashinfer-cubin` + `flashinfer-jit-cache`. Advises restart-runtime-once after the first install.
2. **Cell 1 (imports + GPU probe):** Reads GPU info via `nvidia-smi`, captures compute capability, decides `FLASHINFER_USABLE` (off for SM 12.x).
3. **Cell 2 (Config + tiering):** Dataclass `Config` plus `TIER_MAX_TOKENS` and `max_tokens_for()` helper.
4. **Cell 3 (load input):** Locates `law_llm_input.jsonl` via candidate paths and recursive glob, validates each line (non-empty citation+text, length >= `min_text_chars`), prints length distribution, builds `work_df`.
5. **Cell 4 (prompts):** Defines `DESCRIPTOR_KEYS`, `PROVISION_ROLES`, `BOILERPLATE_ROLES`, `LLM_SCHEMA_HINT`. The `SYSTEM_PROMPT` enforces JSON-only output, embeds DE/FR/IT -> Swiss-legal English glossaries (e.g. `Bewilligung -> permit / authorisation`, `Verfügung -> formal administrative order`, `Beschwerde -> appeal`, French `recours -> appeal`, Italian `ricorso -> appeal`), keeps acronyms verbatim (EDI/DFI/SBFI/SEFRI/FINMA/ESTV/AFC/EJPD/DFJP/DEFR/IFSN/FF/RS/SR), prohibits invented citations, forbids `Art.`/`Abs.`/`al.`/`ch.` etc. in `terms_de_to_en`, and rules that boilerplate (repeals, commencement, fee tables, annex code lists, transitional provisions) gets empty rule fields. `USER_TEMPLATE` injects citation, law_title, section_path, law_code, article, units, source_type, year, static legal area hint, domain hints, and the article text. `trim_text()` does head+tail truncation, `_units_to_str()` coerces structural.units list/dict shapes into a flat label string.
6. **Cell 5 (parsing + boilerplate detection):** `extract_json_object()` strips Markdown code fences and `<think>` blocks then balance-walks braces; on parse failure it tries trailing-comma fix, `_repair_terms_arrays()` (handles `{"de":"X":"Y"}` -> `{"de":"X","en":"Y"}` and `{"de":"X","de":"Y"}` -> two entries), then `ast.literal_eval` as a last resort. `normalize_descriptor()` clips strings/lists to caps, coerces `provision_role_llm` into the controlled vocabulary, clamps `specificity_score` to [0,1], strips forbidden fields, and forces boilerplate rule fields to empty. `detect_boilerplate()` regex-matches `Aufgehoben|Abrog[ée]e?s?|Abrogata?` (repeals, text<220 chars), `Tritt am ... in Kraft|Inkrafttreten|Entre en vigueur|Entra in vigore` (commencement, text<300), and `provision_roles_static = {transitional_or_commencement}` with text<300 chars. `build_static_stub()` emits a templated stub descriptor for these rows (provision_role_llm='transitional_or_commencement', specificity_score=0.05).
7. **Cell 6 (vLLM load):** Loads `AutoTokenizer`, applies the chat template to a probe message to measure the prefix size, inspects `LLM` signature for `attention_backend` / `attention_config` kwargs, calls `LLM(**llm_kwargs)` once with no fallback re-imports. Falls through to diagnostics on failure.
8. **Cell 7 (generation helpers + warm-up):** `render_prompt()` (system + user, optional repair wrapper), `generate_raw()` (single SamplingParams batch), `generate_raw_safe()` (recursive batch-split on OOM), `parse_or_retry()` (extract -> normalize, on failure issue a repair prompt with the previous bad output + parser error). Warm-up: 16 short prompts run through `generate_raw_safe(..., max_tokens=64)` to capture CUDA graphs (3.49 s).
9. **Cell 8 (run config + output paths):** Reasserts `cfg = Config()`, fixes `CUDA_VISIBLE_DEVICES=0`, sets `tensor_parallel_size=1`, builds `output_jsonl`, `output_preview_csv`, `output_failures_jsonl`, `output_metrics_json`, plus a `local_scratch` checkpoint copy.
10. **Cell 9 (extraction loop):** Reads existing checkpoint `_source_row` set, classifies remaining rows via `detect_boilerplate()` and writes static stubs directly (no LLM call). The LLM workload is sorted by length tier descending then by `law_code` ascending (mergesort/stable) so long-prefill batches go first and consecutive articles of the same law share user-prompt header prefix in vLLM's prefix cache. Submits chunks of `submit_chunk` prompts to `llm.generate()` with per-row `SamplingParams` (per-row `max_tokens` from `TIER_MAX_TOKENS`), recursively halves on failure, parses each output via `parse_or_retry`, computes `grounded_terms_pct` (fraction of `terms_de_to_en[].de` strings that are substrings of source text), appends to `local_jsonl`, then copies to Drive `output_jsonl` at the end. Writes preview CSV (first 1,000 rows), `failures.jsonl` if any, and `metrics.json` with status counts, role distribution, prompt/output token totals, rows/sec, tokens/sec, and the full config dump.
11. **Cell 10 (QC pass):** Walks the final JSONL, flags forbidden-field leaks under `llm_enrichment`, counts failed-status rows, summarises status counts, provision-role distribution, mean `terms_grounded_pct`, median concept and term-pair counts.

## Results

Cell 0 (setup):
```
Running in Colab : True
Running in Kaggle: True
Mounted at /content/drive
Installing/upgrading core packages...
Detected CUDA index suffix for FlashInfer: cu130
Step A: installing flashinfer-python + flashinfer-cubin (pinned together)...
Step B: installing flashinfer-jit-cache for cu130...
WARNING:flashinfer.compilation_context:Failed to get device capability: SM 12.x requires CUDA >= 12.9.
WARNING:flashinfer.compilation_context:Failed to get device capability: SM 12.x requires CUDA >= 12.9.
  flashinfer 0.6.10 importable
FlashInfer available: True
============================================================
Setup complete.
If this was the first install in a fresh runtime,
  RESTART RUNTIME ONCE, then resume from Cell 1.
============================================================
```

Cell 1 (imports + GPU probe):
```
Imports OK
Running in Colab : True
Running in Kaggle: True
BASE_DIR        : /content/drive/MyDrive/swiss_law
CHECKPOINT_DIR  : /content/drive/MyDrive/swiss_law/data/checkpoints
GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition (CC 12.0); free=94.97 GiB / total=95.59 GiB; driver 580.82.07
WARNING:flashinfer.compilation_context:Failed to get device capability: SM 12.x requires CUDA >= 12.9.
WARNING:flashinfer.compilation_context:Failed to get device capability: SM 12.x requires CUDA >= 12.9.
FlashInfer importable: 0.6.10

FlashInfer installed but SM 12.0 (NVIDIA RTX PRO 6000 Blackwell Server Edition) is consumer/workstation Blackwell.
FlashInfer 0.6.x has no SM 12.x cubins; vLLM will auto-select FA4 (equally fast).
FlashInfer effective: False
```

Cell 3 (data load):
```
Using input: /content/drive/MyDrive/swiss_law/data/law_llm_input.jsonl
  size: 188.6 MB
Loaded: 173,033 valid rows  (skipped: 0 missing, 0 below min_chars)

Text length distribution (chars):
count    173033.000000
mean        242.750614
std         237.947206
min          20.000000
50%         184.000000
90%         432.000000
95%         599.000000
99%        1277.680000
max        2500.000000

Rows exceeding max_text_chars=0: 173,033 / 173,033 (100.00%)
(those rows get head+tail truncation; quality preserved.)

llm_priority mix:
llm_priority
medium    147941
high       21490
low         3602

Work slice: 173,033 rows (start=0, limit=all)
```

Cell 6 (vLLM load):
```
Loading tokenizer: Qwen/Qwen3-8B-AWQ
System+schema prefix size: ~1497 tokens (prefix-cached portion).
vLLM LLM signature (37 explicit params); attention_config=True

vLLM init plan:
  attention backend: auto-select (vLLM picks FA4 on Blackwell SM 12.x)
  KV cache dtype:    default fp16
  prefix caching:    True
  speculative dec.:  off
  max_model_len:     3072
  max_num_seqs:      480
  gpu_mem_util:      0.92

Initializing vLLM (this can take a minute)...
INFO 05-06 08:45:37 [utils.py:233] non-default args: {'trust_remote_code': True, 'download_dir': '/content/drive/MyDrive/swiss_law/models/huggingface', 'max_model_len': 3072, 'enable_prefix_caching': True, 'max_num_seqs': 480, 'disable_log_stats': True, 'quantization': 'awq_marlin', 'disable_custom_all_reduce': True, 'model': 'Qwen/Qwen3-8B-AWQ'}
INFO 05-06 08:45:38 [model.py:555] Resolved architecture: Qwen3ForCausalLM
INFO 05-06 08:45:38 [model.py:1680] Using max model len 3072
INFO 05-06 08:45:38 [nixl_utils.py:20] Setting UCX_RCACHE_MAX_UNRELEASED to '1024' to avoid a rare memory leak in UCX when using NIXL.
WARNING 05-06 08:45:38 [nixl_utils.py:34] NIXL is not available
WARNING 05-06 08:45:38 [nixl_utils.py:44] NIXL agent config is not available
INFO 05-06 08:45:38 [awq_marlin.py:252] The model is convertible to awq_marlin during runtime. Using awq_marlin kernel.
INFO 05-06 08:45:38 [scheduler.py:239] Chunked prefill is enabled with max_num_batched_tokens=16384.
INFO 05-06 08:45:39 [vllm.py:840] Asynchronous scheduling is enabled.
INFO 05-06 08:45:39 [kernel.py:205] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'])

============================================================
vLLM loaded successfully.
  attention backend (requested): auto
  quantization:                  awq_marlin
  KV cache dtype:                None
  speculative decoding:          False
============================================================
After vLLM load GPU 0: free=6.82 GiB total=94.97 GiB
```

Cell 7 (warm-up):
```
Warm-up generation (16 short prompts)...
  done in 3.49s
```

Cell 8 (config dump):
```
{
  "base_dir": "/content/drive/MyDrive/swiss_law",
  "data_dir": "/content/drive/MyDrive/swiss_law/data",
  "output_dir": "/content/drive/MyDrive/swiss_law/outputs",
  "model_download_dir": "/content/drive/MyDrive/swiss_law/models/huggingface",
  "local_scratch_dir": "/content/drive/MyDrive/swiss_law/data/checkpoints",
  "input_jsonl": "/content/drive/MyDrive/swiss_law/data/law_llm_input.jsonl",
  "fallback_input_jsonl": "law_llm_input.jsonl",
  "model_name": "Qwen/Qwen3-8B-AWQ",
  "start": 0,
  "limit": 0,
  "sample_random": false,
  "random_seed": 42,
  "min_text_chars": 20,
  "max_text_chars": 0,
  "gpu_mode": "single",
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.92,
  "max_model_len": 3072,
  "max_num_seqs": 480,
  "submit_chunk": 2048,
  "enforce_eager": false,
  "quantization": "awq_marlin",
  "disable_custom_all_reduce": true,
  "kv_cache_dtype": null,
  "enable_ngram_speculation": false,
  "speculative_num_tokens": 5,
  "speculative_ngram_min": 2,
  "speculative_ngram_max": 4,
  "use_structured_outputs": false,
  "max_new_tokens": 512,
  "retry_max_new_tokens": 600,
  "max_retries": 1,
  "temperature": 0.1,
  "top_p": 0.9,
  "repetition_penalty": 1.0,
  "enable_thinking": false,
  "include_raw_output_on_success": false
}
Output JSONL (final): /content/drive/MyDrive/swiss_law/outputs/law_llm_descriptors_0000000_all.jsonl
Output JSONL (hot)  : /content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl
```

Cell 9 (extraction run — terminated by error):
```
Checkpoint found: /content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl
  -> 33,003 rows already done; will skip.
Rows remaining: 140,030 / 173,033 total (33,003 skipped).
Boilerplate skip-LLM rows: 0 (0.00%)
LLM workload after sort: 140,030 rows.
Processing 140,030 LLM rows in chunks of 2048 (max_num_seqs=480; static skips already written: 0).
Law LLM enrichment:   0%|          | 0/140030 [00:00<?, ?it/s]
```
Then:
```
ERROR VLLMValidationError: This model's maximum context length is 3072 tokens. However, you requested 0 output tokens and your prompt contains at least 3073 input tokens, for a total of at least 3073 tokens. Please reduce the length of the input prompt or the number of requested output tokens. (parameter=input_tokens, value=3073)
```
The traceback shows the failure path: `parse_or_retry` -> `extract_json_object` raised `ValueError: no balanced JSON object found` on a long truncated descriptor (`{"english_summary": "Aerosol packages must display specific information..." ... "legal_question": "leg` — output ran out mid-string), and the retry path called `generate_raw_safe([render_prompt(row, repair=True, bad_output=raw, error=err)], cfg.retry_max_new_tokens)`. The repair prompt (system prompt + previous bad output + parser error + original user prompt) tokenized to >=3073 tokens, which exceeds `max_model_len=3072`, so vLLM raised `VLLMValidationError` and the cell aborted. No Cell 10 (QC) output was captured.

## Summary

Notebook is the law-side counterpart to the court-considerations LLM-descriptor extractor, targeting all 173,033 valid rows of `law_llm_input.jsonl` (Swiss federal DE/FR/IT law) and emitting a 12-field JSON descriptor per article via Qwen3-8B-AWQ on vLLM with prefix caching, length-tier sorting, per-row `max_tokens` from `TIER_MAX_TOKENS`, and append-mode JSONL checkpointing. The descriptor schema enforces Swiss-legal English equivalents (not literal translation) with verbatim DE/FR/IT terms grounded as source substrings, boilerplate roles forced to empty rule fields, and a controlled addressees vocabulary; parsing is hardened with a custom JSON extractor that repairs duplicate-key and inline-value malformations seen from Qwen3. The run resumed from a 33,003-row checkpoint, found 0 boilerplate skips after classification, sorted 140,030 remaining rows by length tier and `law_code`, and warm-up + vLLM load completed (FA4 backend auto-selected; ~1,497-token cached prefix; 6.82 GiB free post-load). Cell 9 aborted at the very start of the main loop with `VLLMValidationError`: a Qwen3 output was truncated mid-JSON, parsing failed, and the constructed repair prompt (system + bad-output + error + user prompt) exceeded `max_model_len=3072` (>=3,073 input tokens). The Cell 10 markdown documents seven optimisations applied (skip-LLM boilerplate filter, no truncation, tightened `max_model_len`, raised `max_num_seqs`, length-sort descending, within-tier `law_code` sort, tiered per-row `max_tokens`) and notes that the next step after a clean completion is `python scripts/merge_law_llm_into_v1_cards.py --llm <path>`.
