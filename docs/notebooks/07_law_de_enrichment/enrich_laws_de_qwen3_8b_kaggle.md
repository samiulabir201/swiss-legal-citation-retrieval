# enrich_laws_de_qwen3_8b_kaggle

**Path:** `e:\swiss_citation_extraction\notebooks\07_law_de_enrichment\enrich_laws_de_qwen3_8b_kaggle.ipynb`

## Configuration

- Model: `Qwen/Qwen3-8B-AWQ`, quantization `awq_marlin`, KV cache dtype default fp16 (`kv_cache_dtype=None`).
- vLLM: `tensor_parallel_size=1`, `gpu_memory_utilization=0.92`, `max_model_len=2560`, `max_num_seqs=320`, `submit_chunk=2048`, `enforce_eager=False`, `disable_custom_all_reduce=True`, `enable_prefix_caching=True`, `enable_thinking=False`.
- Attention backend: requested `FLASHINFER` if available, but FlashInfer 0.6.x has no SM 12.x cubins so on the actual run `FLASHINFER_USABLE=False` and vLLM auto-selects FA4. N-gram speculative decoding off (`enable_ngram_speculation=False`, crashes on AWQ-Marlin + FP8-KV path).
- Sampling: `temperature=0.1` (vs court's 0.0 — small sampling avoids literal-translation attractor), `top_p=0.9`, `repetition_penalty=1.05`, `max_new_tokens=700`, `retry_max_new_tokens=850`, `max_retries=1`, `use_structured_outputs=False`.
- Text limits: `min_text_chars=20`, `max_text_chars=1500` (law p99 = 1274 chars; rows beyond get head+tail truncation with ` ... [TRUNCATED] ... ` marker).
- Slice: `start=0`, `limit=0` (process all), `sample_random=False`, `random_seed=42`.
- Paths: `BASE_DIR=/content/drive/MyDrive/swiss_law` (Colab) or `/kaggle/working/swiss_law` (Kaggle); input `data/law_llm_input.jsonl`; checkpoints persisted to Drive under `data/checkpoints/`; outputs to `outputs/law_llm_descriptors_<suffix>.jsonl` plus `_preview.csv`, `_failures.jsonl`, `_metrics.json`. Notebook code references `law_llm_input_engine_dead_rerun.jsonl` and suffix `engine_dead_rerun`, but the actual run resolved `law_llm_input.jsonl` and emitted suffix `0000000_all`.
- Environment overrides: `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `FLASHINFER_DISABLE_VERSION_CHECK=1`, `CUDA_VISIBLE_DEVICES=0`, `TOKENIZERS_PARALLELISM=false`, optional `VLLM_USE_FLASHINFER_SAMPLER=1`.
- FlashInfer install path: `flashinfer-python` + `flashinfer-cubin` + `flashinfer-jit-cache` from `https://flashinfer.ai/whl/cu130` (detected via `torch.version.cuda`).

## Data

- Source: slim per-row JSONL `law_llm_input.jsonl` produced by `scripts/build_law_llm_input.py` (~189 MB, derived from the 175,933-row `laws_de.csv` corpus after min-char filtering).
- Loaded 173,033 valid rows (skipped 0 missing, 0 below `min_chars=20`); file size 188.6 MB.
- Each record carries: `_source_row` (checkpoint key), `citation`, `law_title`, `title_section_path`, `structural` (law_code/article/units), `title_metadata` (source_type, enactment_year/date), `static_hints` (legal_area_static, domain_labels_en), `llm_priority`, `text`.
- Text length distribution (chars): mean 242.75, std 237.95, min 20, median 184, p90 432, p95 599, p99 1277.68, max 2500. 1,205 / 173,033 rows (0.70%) exceed `max_text_chars=1500` and receive head+tail truncation.
- `llm_priority` mix: medium 147,941; high 21,490; low 3,602.
- Work slice: 173,033 rows (all). On the rerun reflected by the outputs, 102,296 rows were already in the checkpoint and skipped; 70,737 rows remained.
- Static fields (citation, structural article/units/law_code, title metadata, enactment year, anchors, adjacency, dictionary labels) are never re-derived by the LLM — they are already on each input row and merged back in by `scripts/merge_law_llm_into_v1_cards.py`.

## Pipeline

1. **Cell 0 setup (`enrich_laws_de_qwen3_8b_kaggle.ipynb` Cell 0).** Detects Colab vs Kaggle runtime; mounts Drive if Colab. Installs `vllm>=0.10.0`, `transformers>=4.51.0`, `accelerate`, `safetensors`, `pandas`, `tqdm`, `huggingface_hub`. Detects CUDA major.minor via `torch.version.cuda`, maps to `cu126/cu128/cu129/cu130/cu131` index; installs `flashinfer-python` + `flashinfer-cubin` + `flashinfer-jit-cache` from `https://flashinfer.ai/whl/<cu_idx>`. Requires one runtime restart after the first install.
2. **Cell 1 imports + GPU probe.** Sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`. Runs `nvidia-smi` to capture compute capability and free/total memory. Imports `flashinfer` and flips `FLASHINFER_USABLE=False` when major SM is 12 (consumer/workstation Blackwell), so vLLM falls back to FA4.
3. **Cell 2 Config dataclass.** Defines all runtime knobs as a `Config` dataclass; sets `CUDA_VISIBLE_DEVICES=0`; creates `BASE_DIR`, `DATA_DIR`, `OUTPUT_DIR`, `MODEL_DIR/huggingface`, `LOCAL_SCRATCH`; computes output filenames (`law_llm_descriptors_<suffix>.jsonl`, `_preview.csv`, `_failures.jsonl`, `_metrics.json`) and a local hot-disk twin.
4. **Cell 3 input loader.** `resolve_input_path()` searches DATA_DIR, BASE_DIR, `/content`, `/kaggle/input`, CWD, then globs `**/law_llm_input_engine_dead_rerun.jsonl` and `**/law_llm_input.jsonl`. Streams the JSONL, skipping rows missing `citation`/`text` or shorter than `min_text_chars`. Builds a `pandas` DataFrame with `_source_row`, `citation`, `law_title`, `title_section_path`, `structural`, `title_metadata`, `static_hints`, `llm_priority`, `text`, `_text_len`. Prints length percentiles, count over `max_text_chars`, and `llm_priority` mix. Optional random sampling via `sample_random` + `random_seed`.
5. **Cell 4 schema + prompts.** Defines `DESCRIPTOR_KEYS` (12 LLM fields), the `PROVISION_ROLES` set (14 roles incl. `definition`, `purpose`, `scope`, `principle`, `right_or_entitlement`, `duty`, `prohibition`, `procedure`, `competence`, `sanction_or_penalty`, `data_reporting`, `fees_or_costs`, `transitional_or_commencement`, `other`), and `BOILERPLATE_ROLES = {transitional_or_commencement, fees_or_costs, data_reporting}`. Serializes a `LLM_SCHEMA_HINT` shape into the `SYSTEM_PROMPT`, which embeds 11 hard rules: JSON only, English semantic fields except `terms_de_to_en[].de` / `defined_terms[].term` (verbatim DE/FR/IT substrings), explicit DE→EN / FR→EN / IT→EN Swiss-legal mapping tables (Bewilligung→permit/authorisation; autorisation→permit/authorisation; autorizzazione→permit/authorisation; Verfügung→formal administrative order; etc.), language-invariant acronyms (EDI/DFI/SBFI/SEFRI/FINMA/ESTV/AFC/EJPD/DFJP/DEFR/IFSN/FF/RS/SR), no invented anchors/dates/BGE/parties, no literal translation, do not echo the law title, empty rule fields + low `specificity_score` for boilerplate roles, no anchor fragments (`Art.`, `Abs.`, `Buchstabe`, `Ziffer`, `al.`, `let.`, `ch.`, `cpv.`, `lett.`, `n.`, SR numbers) in `terms_de_to_en`, controlled vocabulary of 18 `addressees` labels, caps (10 terms_de_to_en / 4 defined_terms / 6 addressees / 5 conditions / 5 exceptions / 8 concepts_en), all keys always present. `USER_TEMPLATE` fills citation, law title, section path, law_code, article, units, source_type, enactment year, legal_area hint, domain hints, and the truncated article text. `trim_text` applies head+tail truncation with marker; `_units_to_str` flattens the structural units list (handles dict shapes with `marker/value`, `unit/number`, `type/value`, `label`).
6. **Cell 5 parsing/normalization.** `extract_json_object()` strips markdown fences and `<think>` tags, walks the first balanced `{...}` block, retries with trailing-comma stripping, then `ast.literal_eval` as a last resort. `clean_str`/`clean_list`/`clean_pair_list` apply caps, dedupe casefold-equivalent items, and char limits. `normalize_descriptor()` enforces field caps (summary 400, rule 260, applicability/exceptions 5×160, question 220, concepts 8×80, terms_de_to_en 10×(120/200), defined_terms 4×(120/300), addressees 6×80, sanctions 4×160), coerces `provision_role_llm` to the controlled vocabulary (default `other`), clamps `specificity_score` to [0,1], strips a forbidden-key set (`statute_anchors`, `case_anchors`, `normalized_anchors`, `retrieval_views`, `outcome_signal`, `enrichment_quality`, `anchor_quality_flags`, `incoming_references`, `outgoing_references`, `adjacent_citations`, `law_code`, `law_title`, `enactment_date`, `enactment_year`), and blanks the rule fields when the role is boilerplate. `empty_descriptor(error)` returns a zeroed schema with `_descriptor_error`. `grounded_terms_pct(terms, text)` measures the fraction of `terms_de_to_en[].de` strings that occur verbatim in the source text.
7. **Cell 6 vLLM load.** Loads `AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)`, probes the system+schema prefix length, inspects `LLM.__init__` signature for `attention_backend` / `attention_config`, builds `llm_kwargs` (model, TP, gpu_mem_util, max_model_len, max_num_seqs, enforce_eager, disable_custom_all_reduce, disable_log_stats, download_dir, quantization, prefix-caching, optional `kv_cache_dtype`), wires FlashInfer via kwarg / `AttentionConfig` / `VLLM_ATTENTION_BACKEND` env in that order, and constructs `LLM(**llm_kwargs)`. Catches load failures and prints SM/FP8/spec/OOM-specific remediation hints. On success, reads free/total GPU memory via `torch.cuda.mem_get_info`.
8. **Cell 7 generation helpers + warm-up.** `render_prompt(row, repair, bad_output, error)` builds the user prompt via `build_user_prompt`, optionally wraps it in a repair-mode prefix (parser error + truncated bad output + repeat schema demand), and renders the chat template with `apply_chat_template(add_generation_prompt=True, enable_thinking=False)`. Guards repair prompts against overflow: if `n_tokens > max_model_len - retry_max_new_tokens - 32`, falls back to a minimal repair prompt that drops bad_output/error. `generate_raw` calls `llm.generate(prompts, SamplingParams(...), use_tqdm=False)`. `generate_raw_safe` halves the batch and retries on any exception, with `gc.collect()` + `torch.cuda.empty_cache()` between halves; never propagates a per-row failure. `parse_or_retry(row, raw)` attempts parse → normalize → return; on failure invokes a repair generation with `retry_max_new_tokens`; on a second failure returns `empty_descriptor` with status `failed_descriptor_parse`. Warm-up issues 16 short prompts to trigger CUDA-graph capture.
9. **Cell 8 mega-batch run.** Reads `local_jsonl` to collect already-done `_source_row` ids; filters `work_df` to `remaining_df`. Iterates `range(0, N, SUBMIT)` where `SUBMIT=cfg.submit_chunk=2048`. For each chunk: builds row_obj dicts (with the slim card fields), renders prompts, calls `_generate_with_metrics_safe(prompts, max_new_tokens)`. For each output: extracts raw text, accumulates `prompt_token_ids` and `outputs[0].token_ids` counts, calls `parse_or_retry`, computes `grounded_terms_pct`, builds a record `{_source_row, citation, language='de', llm_priority, llm_enrichment, llm_quality (json_valid, terms_grounded_pct, boilerplate_role), llm_generation (model, method='law_descriptor_v1', attention_backend, kv_cache_dtype, speculative, status, attempt_count, error, raw_output)}`, appends one JSON line to `local_jsonl`, updates the tqdm postfix (chunk RPS, avg RPS, ok count, failed count). After the loop: copies `local_jsonl` to `output_jsonl` on Drive, writes failures to `output_failures_jsonl`, builds a preview DataFrame (up to 1000 rows) flattening enrichment fields and writes `output_preview_csv`, and serializes a metrics dict (config + token totals + rates + status distributions + role distribution) to `output_metrics_json`.
10. **Cell 9 QC.** Streams `output_jsonl`, computes `has_summary`, `has_rule`, `has_question`, list counts, `specificity_score`, `terms_grounded_pct`, and runs `find_forbidden` against the `llm_enrichment` subtree to assert the forbidden-key set is absent. Prints rows checked, forbidden-field count, failed-row count, status / role distributions, average grounded pct, median concept and term-pair counts, boilerplate-role row count, and substantive-role rows with non-empty rule.
11. **Cell 10 markdown — tuning notes.** Documents knobs for `max_text_chars`, `max_new_tokens`, `temperature`, double boilerplate suppression (in `normalize_descriptor` and again in `scripts/merge_law_llm_into_v1_cards.py`), and the resume protocol via `_source_row`. Next step: `python scripts/merge_law_llm_into_v1_cards.py --llm <path>` to produce `artifacts/law_authority_cards_v2_unified.jsonl`.

## Results

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
  "max_model_len": 2560,
  "max_num_seqs": 320,
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
  "max_new_tokens": 700,
  "retry_max_new_tokens": 850,
  "max_retries": 1,
  "temperature": 0.1,
  "top_p": 0.9,
  "repetition_penalty": 1.05,
  "enable_thinking": false,
  "include_raw_output_on_success": false
}
Output JSONL (final): /content/drive/MyDrive/swiss_law/outputs/law_llm_descriptors_0000000_all.jsonl
Output JSONL (hot)  : /content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl
```

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

```
Loading tokenizer: Qwen/Qwen3-8B-AWQ
System+schema prefix size: ~1497 tokens (prefix-cached portion).
vLLM LLM signature (37 explicit params); attention_config=True

vLLM init plan:
  attention backend: auto-select (vLLM picks FA4 on Blackwell SM 12.x)
  KV cache dtype:    default fp16
  prefix caching:    True
  speculative dec.:  off
  max_model_len:     2560
  max_num_seqs:      320
  gpu_mem_util:      0.92

Initializing vLLM (this can take a minute)...
INFO 05-06 12:40:24 [utils.py:233] non-default args: {'trust_remote_code': True, 'download_dir': '/content/drive/MyDrive/swiss_law/models/huggingface', 'max_model_len': 2560, 'enable_prefix_caching': True, 'max_num_seqs': 320, 'disable_log_stats': True, 'quantization': 'awq_marlin', 'disable_custom_all_reduce': True, 'model': 'Qwen/Qwen3-8B-AWQ'}
INFO 05-06 12:40:25 [model.py:555] Resolved architecture: Qwen3ForCausalLM
INFO 05-06 12:40:25 [model.py:1680] Using max model len 2560
INFO 05-06 12:40:26 [nixl_utils.py:20] Setting UCX_RCACHE_MAX_UNRELEASED to '1024' to avoid a rare memory leak in UCX when using NIXL.
WARNING 05-06 12:40:26 [nixl_utils.py:34] NIXL is not available
WARNING 05-06 12:40:26 [nixl_utils.py:44] NIXL agent config is not available
INFO 05-06 12:40:26 [awq_marlin.py:252] The model is convertible to awq_marlin during runtime. Using awq_marlin kernel.
INFO 05-06 12:40:26 [scheduler.py:239] Chunked prefill is enabled with max_num_batched_tokens=16384.

INFO 05-06 12:40:26 [vllm.py:840] Asynchronous scheduling is enabled.
INFO 05-06 12:40:26 [kernel.py:205] Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['native'])

============================================================
vLLM loaded successfully.
  attention backend (requested): auto
  quantization:                  awq_marlin
  KV cache dtype:                None
  speculative decoding:          False
============================================================
After vLLM load GPU 0: free=7.05 GiB total=94.97 GiB
```

```
Warm-up generation (16 short prompts)...
  done in 0.66s
```

```
Checkpoint found: /content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl
  → 102,296 rows already done; will skip.
Rows remaining: 70,737 / 173,033 total (102,296 skipped).
Processing 70,737 remaining rows in chunks of 2048 (max_num_seqs=320; 102,296 done).

Finished 70737 rows in 6337.4s = 11.16 rows/sec
Prompt tokens total:  119,821,790  (avg 1694/row)
Output tokens total:  20,596,323  (avg 291/row)
Avg terms_grounded_pct: 0.909

Copying /content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl → /content/drive/MyDrive/swiss_law/outputs/law_llm_descriptors_0000000_all.jsonl ...
  done.
{
  "start": 0,
  "limit": 0,
  "selected_rows": 173033,
  "written_rows_this_session": 70737,
  "total_rows_in_file": 173033,
  "failures": 66,
  "elapsed_seconds": 6337.447025775909,
  "rows_per_second": 11.1617501041501,
  "prompt_tokens_total": 119821790,
  "output_tokens_total": 20596323,
  "tokens_per_second": 22156.889427065194,
  "output_tokens_per_second": 3249.940065176062,
  "avg_terms_grounded_pct": 0.9087373368959325,
  "attention_backend": "auto",
  "quantization": "awq_marlin",
  "kv_cache_dtype": null,
  "speculative": false,
  "status_counts": {
    "ok": 70617,
    "failed_descriptor_parse": 66,
    "ok_after_retry": 54
  },
  "provision_role_distribution": {
    "principle": 364,
    "other": 27062,
    "prohibition": 1084,
    "scope": 3254,
    "competence": 1841,
    "procedure": 11489,
    "purpose": 1015,
    "definition": 8695,
    "duty": 9565,
    "data_reporting": 1327,
    "right_or_entitlement": 4469,
    "sanction_or_penalty": 360,
    "transitional_or_commencement": 209,
    "fees_or_costs": 3
  },
  "output_jsonl": "/content/drive/MyDrive/swiss_law/outputs/law_llm_descriptors_0000000_all.jsonl",
  "output_preview_csv": "/content/drive/MyDrive/swiss_law/outputs/law_llm_descriptors_0000000_all_preview.csv",
  "output_failures_jsonl": "/content/drive/MyDrive/swiss_law/outputs/law_llm_descriptors_0000000_all_failures.jsonl",
  "config": {
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
    "max_model_len": 2560,
    "max_num_seqs": 320,
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
    "max_new_tokens": 700,
    "retry_max_new_tokens": 850,
    "max_retries": 1,
    "temperature": 0.1,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
    "enable_thinking": false,
    "include_raw_output_on_success": false
  }
}
```

Preview DataFrame head (first 8 of 50 displayed rows):

```
    _source_row                citation status            role  specificity_score
0        102325    Art. 19 Abs. 3 MessG     ok       principle                0.5
1        102326    Art. 19 Abs. 4 MessG     ok           other                0.8
2        102327    Art. 19 Abs. 5 MessG     ok           other                0.5
3        102328    Art. 20 Abs. 1 MessG     ok     prohibition                0.8
4        102329    Art. 20 Abs. 2 MessG     ok           other                0.8
5        102330    Art. 21 Abs. 1 MessG     ok           other                0.8
6        102331    Art. 21 Abs. 2 MessG     ok           other                0.8
7        102332           Art. 22 MessG     ok           scope                0.8
```

Cell 9 QC pass over the full output JSONL:

```
Rows checked: 173033
Forbidden field rows: 0
Failed rows: 2118
Status counts: {'ok': 167775, 'static_skip': 2283, 'failed_descriptor_parse': 2118, 'ok_after_retry': 857}
Provision role distribution: {'other': 66303, 'procedure': 24573, 'definition': 24125, 'duty': 21997, 'right_or_entitlement': 9263, 'scope': 8029, 'competence': 4510, 'data_reporting': 3491, 'purpose': 3161, 'transitional_or_commencement': 3072, 'prohibition': 2468, 'principle': 1340, 'sanction_or_penalty': 686, 'fees_or_costs': 15}
Avg terms_grounded_pct: 0.902
Median concept_count: 6.0
Median term_pair_count: 4.0
Boilerplate-role rows: 6578
Substantive rows with non-empty rule: 85080
```

## Summary

The notebook enriches the full 173,033-row Swiss `laws_de` corpus with a 12-field JSON descriptor (`english_summary`, `legal_rule`, `applicability_conditions`, `exceptions_or_limitations`, `legal_question`, `concepts_en`, `terms_de_to_en`, `defined_terms`, `addressees`, `sanctions_or_consequences`, `provision_role_llm`, `specificity_score`) by running Qwen3-8B-AWQ under vLLM on a single RTX PRO 6000 Blackwell (SM 12.0, 95 GiB), with FA4 attention (FlashInfer 0.6.x has no SM 12.x cubins so it auto-falls back), default fp16 KV cache, prefix caching, no speculative decoding, and continuous batching at `max_num_seqs=320` / `submit_chunk=2048`. The system prompt embeds the full Swiss-legal DE/FR/IT→EN equivalence tables and 11 hard rules so original-language statutory terms remain verbatim while semantic fields are emitted in Swiss-legal English; `normalize_descriptor` enforces field caps, controlled `provision_role_llm` and `addressees` vocabularies, and a forbidden-key denylist; boilerplate roles (`transitional_or_commencement`, `fees_or_costs`, `data_reporting`) get blanked rule fields. The session-as-run resumed from a checkpoint with 102,296 rows already done and processed the remaining 70,737 rows in 6,337.4 s (11.16 rows/s; ~22,157 tokens/s total, ~3,250 output tokens/s; avg 1,694 prompt and 291 output tokens/row), with 70,617 `ok` / 54 `ok_after_retry` / 66 `failed_descriptor_parse` and an average `terms_grounded_pct` of 0.909. The final Cell 9 QC pass over all 173,033 output rows reports 0 forbidden-field violations, status counts `{ok: 167775, static_skip: 2283, failed_descriptor_parse: 2118, ok_after_retry: 857}`, average grounded pct 0.902, median 6 concepts and 4 term pairs per row, and 85,080 substantive rows carrying a non-empty `legal_rule`, ready to feed `scripts/merge_law_llm_into_v1_cards.py` to build `artifacts/law_authority_cards_v2_unified.jsonl`.
