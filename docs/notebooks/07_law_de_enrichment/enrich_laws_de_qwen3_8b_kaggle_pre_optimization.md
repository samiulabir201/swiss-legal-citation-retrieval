# enrich_laws_de_qwen3_8b_kaggle_pre_optimization

**Path:** `e:\swiss_citation_extraction\notebooks\07_law_de_enrichment\enrich_laws_de_qwen3_8b_kaggle_pre_optimization.ipynb`

## Configuration

- **Model:** `Qwen/Qwen3-8B-AWQ` (AWQ-quantized; `awq_marlin` kernel).
- **Runtime:** Colab/Kaggle, GPU = NVIDIA RTX PRO 6000 Blackwell Server Edition (CC 12.0), 95.59 GiB VRAM, driver 580.82.07.
- **Inference engine:** vLLM (`>=0.10.0`) with `enable_prefix_caching=True`, `disable_custom_all_reduce=True`, `disable_log_stats=True`, `enforce_eager=False`.
- **Attention backend:** requested `FLASHINFER` (FlashInfer 0.6.10 importable) but `FLASHINFER_USABLE=False` on SM 12.0 (no SM 12.x cubins in FlashInfer 0.6.x); vLLM auto-selects FA4 on Blackwell.
- **KV cache dtype:** `None` (default fp16). Code comment says "FP8 KV — same as court" but `kv_cache_dtype` is left `None` in this config.
- **Speculative decoding:** `enable_ngram_speculation=False` (off on SM 12.x due to same incompatibility found in the court run).
- **GPU layout:** `gpu_mode='single'`, `tensor_parallel_size=1`, `gpu_memory_utilization=0.92`.
- **Context / batching:** `max_model_len=3072`, `max_num_seqs=384`, `submit_chunk=2048`.
- **Sampling:** `temperature=0.1`, `top_p=0.9`, `repetition_penalty=1.0`, `max_new_tokens=450`, `retry_max_new_tokens=600`, `max_retries=1`, `use_structured_outputs=False`, `enable_thinking=False`.
- **Slice:** `start=0`, `limit=0` (all rows), `sample_random=False`, `random_seed=42`.
- **Text bounds:** `min_text_chars=20`, `max_text_chars=1500` (head+tail truncation above; covers ~99% untruncated per law p99 = 1274 chars).
- **CUDA/FlashInfer install:** detects CUDA suffix (`cu130` in this run) and installs `flashinfer-python` + `flashinfer-cubin` + `flashinfer-jit-cache` from `https://flashinfer.ai/whl/{cuda_idx}`. `FLASHINFER_DISABLE_VERSION_CHECK=1`. `VLLM_WORKER_MULTIPROC_METHOD=spawn`. `TOKENIZERS_PARALLELISM=false`.
- **Paths (Colab):** `BASE_DIR=/content/drive/MyDrive/swiss_law`; checkpoints in `BASE_DIR/data/checkpoints/`; outputs in `BASE_DIR/outputs/`; HF cache in `BASE_DIR/models/huggingface/`.

## Data

- **Input file:** `/content/drive/MyDrive/swiss_law/data/law_llm_input.jsonl` (188.6 MB), produced by `scripts/build_law_llm_input.py`.
- **Source corpus:** `laws_de.csv` (175,933 rows total). After `min_char` filter the slim JSONL has 173,033 rows.
- **Loaded:** 173,033 valid rows; 0 skipped missing, 0 below `min_chars`.
- **Text length (chars):** count=173,033; mean=242.75; std=237.95; min=20; median=184; p90=432; p95=599; p99=1277.68; max=2500.
- **Rows exceeding `max_text_chars=1500`:** 1,205 / 173,033 (0.70%) — receive head+tail truncation.
- **`llm_priority` mix:** medium=147,941; high=21,490; low=3,602.
- **Work slice:** 173,033 rows (start=0, limit=all).
- **Per-row payload fields:** `_source_row`, `citation`, `law_title`, `title_section_path`, `structural`, `title_metadata`, `static_hints`, `llm_priority`, `text`. Static fields (citation, structural article/units/law_code, title metadata, enactment year, anchors, adjacency, dictionary labels) are NEVER re-derived by the LLM and are merged back later by `scripts/merge_law_llm_into_v1_cards.py`.

## Pipeline

1. **Cell 0 (markdown):** scope statement — law-side counterpart of the court enrichment notebook; 12 LLM fields; Qwen3-8B-AWQ on RTX PRO 6000 Blackwell with FlashInfer-or-FA4 + FP8 KV + continuous batching; ~60–90 min target for 173k rows; quality contract (verbatim DE/FR/IT substrings; Swiss-legal English equivalents not literal translations; empty rule fields for boilerplate; no invented citations).
2. **Cell 1 (Colab/Kaggle setup):** detect Colab/Kaggle, mount Drive, `pip install` vLLM>=0.10.0, transformers>=4.51.0, accelerate, safetensors, pandas, tqdm, huggingface_hub. Detect CUDA via `torch.version.cuda` -> `cu126|cu128|cu129|cu130|cu131`; install `flashinfer-python`, `flashinfer-cubin`, and `flashinfer-jit-cache` from `https://flashinfer.ai/whl/{cuda_idx}`. Set `FLASHINFER_DISABLE_VERSION_CHECK=1`. Note: requires runtime restart once after first install.
3. **Cell 2 (imports + GPU probe):** standard imports (`pandas`, `tqdm`, `pathlib`, `dataclasses`, `re`, `gc`, `ast`, `json`, `time`, `shutil`, `subprocess`, `traceback`). Set `VLLM_WORKER_MULTIPROC_METHOD=spawn`. Run `nvidia-smi --query-gpu=...` to record name, compute capability, memory, driver. Try `import flashinfer` and on SM 12.x set `FLASHINFER_USABLE=False`.
4. **Cell 3 (Config dataclass):** `@dataclass Config` holding all knobs above; instantiate `cfg = Config()`.
5. **Cell 4 (load input JSONL):** `resolve_input_path()` searches Drive, BASE_DIR, `/content`, `/kaggle/input` and CWD for `law_llm_input.jsonl`. Parse line-by-line, drop rows missing `citation`/`text` or below `min_text_chars`. Build `valid` DataFrame; print length distribution and `llm_priority` mix; compute `work_df` slice. Truncation logic uses head+tail when `len(text) > cfg.max_text_chars`.
6. **Cell 5 (schema + prompts):** define `DESCRIPTOR_KEYS` (12 fields), `PROVISION_ROLES` (14 enum values), `BOILERPLATE_ROLES = {'transitional_or_commencement', 'fees_or_costs', 'data_reporting'}`. Build `LLM_SCHEMA_HINT` JSON and inject into `SYSTEM_PROMPT` (~1497 tokens prefix-cached) describing Swiss federal law context (DE/FR/IT), JSON-only contract, verbatim original-language substrings in `terms_de_to_en[].de` and `defined_terms[].term`, mapping to Swiss-legal English equivalents (NOT literal translations), and rules forbidding invented citations/dates/BGE numbers/party names.
7. **Cell 6 (parsing & normalization):** `_repair_terms_arrays()` fixes two observed Qwen3 malformations — (A) `{"de":"X":"Y"}` → `{"de":"X","en":"Y"}`, (B) `{"de":"X","de":"Y"}` → `{"de":"X"},{"de":"Y"}`. `extract_json_object()` strips code fences, `<think>` blocks, parses the first balanced `{...}`, removes trailing commas. Normalization mirrors `scripts/run_law_llm_enrichment.py`. Boilerplate roles get empty rule fields and a low specificity score; `grounded_terms_pct()` measures the fraction of `terms_de_to_en[].de` that are exact substrings of source text.
8. **Cell 7 (vLLM load):** load `AutoTokenizer` (`trust_remote_code=True`, cache_dir = `MODEL_DIR/huggingface`); probe system+schema prefix size (~1497 tokens); inspect `LLM` kwarg set (37 explicit params; `attention_config=True`); construct `llm_kwargs` (model, TP=1, gpu_mem_util=0.92, max_model_len=3072, max_num_seqs=384, enforce_eager=False, awq_marlin quantization, prefix caching True); choose attention backend (`FLASHINFER` if usable, else auto/FA4). Optionally enable n-gram speculation (off here). After load: GPU 0 free = 6.90 GiB / 94.97 GiB.
9. **Cell 8 (helpers + warm-up):** `render_prompt()` applies chat template with `enable_thinking=False`; supports repair mode (re-prompts with previous-bad-output + parser error). `generate_raw_safe()` recursively halves the batch on exceptions, with `gc.collect()` + `torch.cuda.empty_cache()`. `parse_or_retry()` calls `extract_json_object` + `normalize_descriptor`; on failure retries once at `retry_max_new_tokens=600`. Warm-up runs 16 short prompts (done in 3.49s).
10. **Cell 9 (paths + config dump):** rebuild `cfg`, create output dirs, derive output filenames (`law_llm_descriptors_{start}_{end}.jsonl|_preview.csv|_failures.jsonl|_metrics.json`) on Drive + a local hot-path checkpoint copy.
11. **Cell 10 (run extraction):** load any existing checkpoint by `_source_row`; submit `submit_chunk=2048` prompts per `llm.generate()` call (vLLM continuous-batches across `max_num_seqs=384`); per row compute descriptor + `terms_grounded_pct`; append JSONL line with `{_source_row, citation, language='de', llm_priority, llm_enrichment, llm_quality, llm_generation}`; flush after each chunk; on exception the chunk falls back to per-row. At end: copy local hot JSONL to Drive output, write failures JSONL if any, write preview CSV (first 1000 rows) and metrics JSON (status counts, provision-role distribution, throughput, terms-grounded, etc.).
12. **Cell 11 (QC):** read final `output_jsonl` and verify schema integrity, detect forbidden fields (`statute_anchors`, `case_anchors`, `normalized_anchors`, `retrieval_views`, `outcome_signal`, `enrichment_quality`, `anchor_quality_flags`, `incoming_references`, `outgoing_references`, `adjacent_citations`, `law_code`, `law_title`, `enactment_date`, `enactment_year`), tally statuses, provision roles, term grounding, concept/term/addressee counts, boilerplate-vs-substantive rule presence.
13. **Cell 12 (markdown tuning notes):** discusses bumping `max_text_chars` to 2000 if TRUNCATED markers appear; bumping `max_new_tokens` to 800 and retry to 950 on truncated JSON; rationale for `temperature=0.1` (vs court's 0.0) to break literal-translation attractor; resume protocol; next step `python scripts/merge_law_llm_into_v1_cards.py --llm <path>` → `artifacts/law_authority_cards_v2_unified.jsonl`.

## Results

Cell 1 (setup) output:
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

Cell 2 (imports + GPU probe) output:
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

Cell 4 (data load) output:
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

Rows exceeding max_text_chars=1500: 1,205 / 173,033 (0.70%)
(those rows get head+tail truncation; quality preserved.)

llm_priority mix:
llm_priority
medium    147941
high       21490
low         3602

Work slice: 173,033 rows (start=0, limit=all)
```

Cell 7 (vLLM load) output:
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
  max_num_seqs:      384
  gpu_mem_util:      0.92

Initializing vLLM (this can take a minute)...
INFO 05-06 05:38:15 [utils.py:233] non-default args: {'trust_remote_code': True, 'download_dir': '/content/drive/MyDrive/swiss_law/models/huggingface', 'max_model_len': 3072, 'enable_prefix_caching': True, 'max_num_seqs': 384, 'disable_log_stats': True, 'quantization': 'awq_marlin', 'disable_custom_all_reduce': True, 'model': 'Qwen/Qwen3-8B-AWQ'}
INFO 05-06 05:38:16 [model.py:555] Resolved architecture: Qwen3ForCausalLM
INFO 05-06 05:38:16 [model.py:1680] Using max model len 3072
INFO 05-06 05:38:16 [nixl_utils.py:20] Setting UCX_RCACHE_MAX_UNRELEASED to '1024' to avoid a rare memory leak in UCX when using NIXL.
WARNING 05-06 05:38:16 [nixl_utils.py:34] NIXL is not available
WARNING 05-06 05:38:16 [nixl_utils.py:44] NIXL agent config is not available
INFO 05-06 05:38:16 [awq_marlin.py:252] The model is convertible to awq_marlin during runtime. Using awq_marlin kernel.
INFO 05-06 05:38:16 [scheduler.py:239] Chunked prefill is enabled with max_num_batched_tokens=16384.
Parse safetensors files:   0%|          | 0/2 [00:00<?, ?it/s]
INFO 05-06 05:38:17 [vllm.py:840] Asynchronous scheduling is enabled.
INFO 05-06 05:38:17 [kernel.py:205] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'])

============================================================
vLLM loaded successfully.
  attention backend (requested): auto
  quantization:                  awq_marlin
  KV cache dtype:                None
  speculative decoding:          False
============================================================
After vLLM load GPU 0: free=6.90 GiB total=94.97 GiB
```

Cell 8 (warm-up) output:
```
Warm-up generation (16 short prompts)...
  done in 3.49s
```

Cell 9 (config dump) output:
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
  "max_text_chars": 1500,
  "gpu_mode": "single",
  "tensor_parallel_size": 1,
  "gpu_memory_utilization": 0.92,
  "max_model_len": 3072,
  "max_num_seqs": 384,
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
  "max_new_tokens": 450,
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

Cell 10 (run extraction) output:
```
No checkpoint found — starting fresh.
Rows remaining: 173,033 / 173,033 total (0 skipped).
Processing 173,033 remaining rows in chunks of 2048 (max_num_seqs=384; 0 done).
Law LLM enrichment:   0%|          | 0/173033 [00:00<?, ?it/s]
```

No throughput, status-counter, role-distribution, or completion log is captured in this notebook — the extraction loop's progress bar shows 0/173033 at start and no further lines were saved. Cell 11 (QC) has no recorded output.

## Summary

This is the law-side LLM-enrichment notebook (pre-optimization variant), targeting the full 173,033-row slim `law_llm_input.jsonl` derived from `laws_de.csv` (175,933 rows) on a single RTX PRO 6000 Blackwell with Qwen3-8B-AWQ via vLLM. Configuration uses `awq_marlin` quantization, `max_model_len=3072`, `max_num_seqs=384`, `submit_chunk=2048`, `temperature=0.1`, `top_p=0.9`, `max_new_tokens=450`, prefix caching on; FlashInfer 0.6.10 imports but is unusable on SM 12.0 so vLLM auto-selects FA4. The 12-field schema (`english_summary`, `legal_rule`, `applicability_conditions`, `exceptions_or_limitations`, `legal_question`, `concepts_en`, `terms_de_to_en`, `defined_terms`, `addressees`, `sanctions_or_consequences`, `provision_role_llm`, `specificity_score`) is enforced as JSON-only with verbatim DE/FR/IT term substrings mapped to Swiss-legal English equivalents, with bespoke regex repairs for two observed Qwen3 malformations. Setup, data-load, vLLM-load and warm-up cells executed cleanly (system prefix ~1497 tokens, GPU free=6.90 GiB after load, warm-up 3.49s); the extraction cell shows only the start-of-run banner with 0/173,033 — no final throughput, status counts, role distribution, or QC metrics are recorded in this saved notebook. Downstream step (out of scope here) is `scripts/merge_law_llm_into_v1_cards.py` producing `artifacts/law_authority_cards_v2_unified.jsonl`.
