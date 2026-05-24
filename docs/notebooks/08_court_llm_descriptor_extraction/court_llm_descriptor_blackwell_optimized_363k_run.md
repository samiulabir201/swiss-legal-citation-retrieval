# court_llm_descriptor_blackwell_optimized_363k_run

**Path:** e:\swiss_citation_extraction\notebooks\08_court_llm_descriptor_extraction\court_llm_descriptor_blackwell_optimized_363k_run.ipynb

## Configuration

- Runtime: Google Colab, Blackwell GPU (NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0, 95.59 GiB VRAM, driver 580.82.07).
- Base directory: `/content/drive/MyDrive/swiss_law`; checkpoint dir `/content/drive/MyDrive/swiss_law/data/checkpoints`.
- Model: `Qwen/Qwen3-8B-AWQ`, quantization `awq_marlin`, `enable_thinking=False`.
- vLLM engine: `tensor_parallel_size=1`, `gpu_memory_utilization=0.92`, `max_model_len=2048`, `max_num_seqs=384`, `enforce_eager=False`, `disable_custom_all_reduce=True`, `enable_prefix_caching=True`.
- KV cache dtype: `None` (default fp16; FP8 KV is documented as a knob but left off in this run).
- Attention backend: auto (FlashInfer was installed and importable as `0.6.9` but auto-suppressed because FlashInfer 0.6.x lacks pre-built kernels for SM 12.x; vLLM auto-selected FlashAttention 4).
- Speculative decoding: `enable_ngram_speculation=False` (documented incompatibility with SM 12.x + AWQ Marlin path).
- Submission strategy: `submit_chunk=2048` prompts per `llm.generate()` call, continuous batching across 384 slots.
- Sampling: `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.0`, `max_new_tokens=320`, `retry_max_new_tokens=448`, `max_retries=1`, `use_structured_outputs=False`.
- Slice: `start=0`, `limit=0` (entire input), `sample_random=False`, `min_text_chars=80`, `max_text_chars=2400` (head+tail truncation if exceeded).
- Schema-in-system-prompt optimization: the 16-key descriptor schema is embedded in the system message so the constant prefix (~315 tokens) is prefix-cached across all rows.
- Forbidden output fields stripped during normalization: `statute_anchors`, `case_anchors`, `normalized_anchors`, `retrieval_views`, `outcome_signal`, `query_phrases_en`, `natural_language_queries`, `legal_question`, `summary_en`, `english_summary`, `enrichment_quality`, `anchor_quality_flags`.
- Output paths (suffix `0000000_all`): `court_llm_descriptors_0000000_all.jsonl`, `_preview.csv`, `_failures.jsonl`, `_metrics.json` under `outputs/`; local hot-path JSONL written under `data/checkpoints/` and copied to Drive at end.

## Data

- Input: `court_authority_cards_v4_target_cards.jsonl` (981.3 MB, 363,258 records) — the LLM-enrichment-target subset (~15%) of the 2.4M-row v4 court authority cards corpus, produced by `scripts/identify_court_enrichment_targets.py` and `scripts/extract_court_target_cards.py`.
- Per-card fields used: `citation`, `text_excerpt_original`, `language`, `legal_area`, `_enrichment_target.line_idx` (stable checkpoint key copied into `_source_row`).
- Loading result: 363,258 valid cards; 0 skipped for missing text/citation; 0 skipped below `min_text_chars`.
- Text length distribution (chars): count 363,258; mean 712.39; std 403.28; min 120; p50 693; p90 1,203; p95 1,203; p99 1,203; max 1,203. 0 rows exceeded `max_text_chars=2400`.
- Language mix: de 247,255; fr 99,097; it 14,955; unknown 1,951.
- Selected rows: 363,258 (entire dataset).
- Output schema keys (16): `legal_area`, `primary_domain`, `secondary_domain`, `legal_domain_path`, `topic`, `subtopic`, `micro_topic`, `concepts_en`, `terms_original`, `doctrinal_rule`, `legal_test`, `fact_pattern_tags`, `procedural_context`, `paragraph_role`, `authority_role`, `specificity_score`.
- `paragraph_role` validated against `{holding, reasoning, facts, procedural_history, legal_standard, application, citation, costs, notification, disposition, neutral}` (defaults to `neutral`).

## Pipeline

1. Cell 0 — Colab setup, mounts Drive, installs `vllm>=0.10.0`, `transformers>=4.51.0`, plus FlashInfer triplet (`flashinfer-python`, `flashinfer-cubin`, `flashinfer-jit-cache`) keyed on detected CUDA suffix `cu130`; sets `FLASHINFER_DISABLE_VERSION_CHECK=1`.
2. Cell 1 — Imports, sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`, queries GPU via `nvidia-smi` without initializing CUDA, probes FlashInfer importability, decides `FLASHINFER_USABLE=False` for SM 12.x GPUs.
3. Cell 2 — Builds `Config` dataclass, creates Drive/scratch directories, defines output paths with suffix `{start:07d}_{end_idx:07d}` or `{start:07d}_all`.
4. Cell 3 — Resolves input JSONL via candidate-path search, streams records keeping only required fields, applies start/limit slicing, prints text-length distribution and language mix.
5. Cell 4 — Defines `DESCRIPTOR_KEYS`, `ROLE_VALUES`, `LLM_SCHEMA_HINT`, `SYSTEM_PROMPT` (with schema inline), `USER_TEMPLATE` (variable per-row), and `trim_text` (head+tail middle-elision when over `max_text_chars`).
6. Cell 5 — JSON parsing (`extract_json_object`: strips fences/`<think>`, brace-balance scan, fallback `ast.literal_eval`) and `normalize_descriptor` (clamps lengths, dedupes lists, normalizes `paragraph_role`, clamps `specificity_score` to `[0,1]`, strips forbidden fields); `empty_descriptor` builds zero-valued payload with `_descriptor_error`.
7. Cell 6 — Loads tokenizer, measures prefix (315 tokens), inspects `LLM.__init__` signature, builds `llm_kwargs` with `enable_prefix_caching=True`, instantiates `LLM` in a single attempt (no fallback reloads — explicitly designed to avoid the `LayerName already registered` crash), then imports `torch` post-load for VRAM metric.
8. Cell 7 — Generation helpers: `render_prompt` (with repair-mode template), `generate_raw` (single `llm.generate` call with `SamplingParams`), `generate_raw_safe` (OOM-resilient recursive batch split), `parse_or_retry` (per-row retry on parse failure), and a 16-prompt warm-up to pay CUDA-graph capture cost.
9. Cell 8 — Main loop: scans local JSONL for prior `_source_row` checkpoints, processes `remaining_df` in chunks of `submit_chunk=2048`, accumulates prompt/output token totals, parses or retries each output, writes per-row records to local-disk JSONL with one flush per chunk, copies final JSONL to Drive, writes failures JSONL (or unlinks if zero), writes preview CSV (first 1000 records), writes metrics JSON.
10. Cell 9 — QC: scans output JSONL, recursively searches for any forbidden field name in records, reports `forbidden_fields` per row, status counts, and failed-row count.

## Results

- Cell 0 (FlashInfer install): detected `cu130`; pip install of `flashinfer-python` and `flashinfer-cubin` succeeded; `flashinfer 0.6.9` importable; warnings logged: `Failed to get device capability: SM 12.x requires CUDA >= 12.9`.
- Cell 1 (runtime check): `GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition (compute capability 12.0); free=94.97 GiB / total=95.59 GiB; driver 580.82.07`. FlashInfer importable but `FLASHINFER_USABLE=False` because SM 12.0 lacks pre-built kernels in FlashInfer 0.6.x.
- Cell 2 (config): dumped full config JSON. Output JSONL final path: `/content/drive/MyDrive/swiss_law/outputs/court_llm_descriptors_0000000_all.jsonl`; local hot-path: `/content/drive/MyDrive/swiss_law/data/checkpoints/court_llm_descriptors_0000000_all.jsonl`.
- Cell 3 (data load): `Loaded: 363,258 valid cards (skipped: 0 missing text/citation, 0 below min_text_chars)`; printed length and language distributions; previewed first 10 rows (all DE constitutional/public-law BGE citations).
- Cell 4 (prompt) — no output.
- Cell 5 (parsing) — no output.
- Cell 6 (vLLM load): `System+schema prefix size: ~315 tokens`; `vLLM LLM signature (37 explicit params); attention_config=True`; plan reported `attention backend: auto-select (vLLM picks FA4 on Blackwell SM 12.x)`, `KV cache dtype: default fp16`, `prefix caching: True`, `speculative dec.: off`, `max_model_len=2048`, `max_num_seqs=384`, `gpu_mem_util=0.92`. vLLM logs: `Resolved architecture: Qwen3ForCausalLM`, `Using max model len 2048`, `Using awq_marlin kernel`, `Chunked prefill is enabled with max_num_batched_tokens=16384`, `Asynchronous scheduling is enabled`. Load succeeded; `After vLLM load GPU 0: free=6.98 GiB total=94.97 GiB`.
- Cell 7 (warm-up): `Warm-up generation (16 short prompts)... done in 23.59s`.
- Cell 8 (main run):
  - `No checkpoint found — starting fresh.` `Rows remaining: 363,258 / 363,258 total (0 skipped).` `Processing 363,258 remaining rows in chunks of 2048 (max_num_seqs=384; 0 already done).`
  - `Finished 363258 rows in 15895.0s = 22.85 rows/sec`.
  - `Prompt tokens total: 207,978,225 (avg 573/row)`.
  - `Output tokens total: 80,692,838 (avg 222/row)`.
  - Metrics JSON: `selected_rows=363258`, `written_rows_this_session=363258`, `total_rows_in_file=363258`, `failures=1`, `elapsed_seconds=15895.011837005615`, `rows_per_second=22.853584742496952`, `tokens_per_second=18161.11028794184`, `output_tokens_per_second=5076.613897961169`, `attention_backend="auto"`, `quantization="awq_marlin"`, `kv_cache_dtype=null`, `speculative=false`, `status_counts={"ok": 363212, "ok_after_retry": 45, "failed_descriptor_parse": 1}`.
  - Preview-table head (first row): `BGE 139 I 2 E. 5.7 | ok | Constitutional law | Local planning | Land use regulation | Zoning changes | Initiative for new zoning | Initiative to rezone land for public parks | Zoning | Initiative | Land use | Public parks ... | Volksabstimmung | Umzonung | Parkanlagen | Haf...` with `paragraph_role=facts`.
- Cell 9 (QC): `Rows checked: 363258`, `Forbidden field rows: 0`, `Failed rows: 1`, `Status counts: {'ok': 363212, 'ok_after_retry': 45, 'failed_descriptor_parse': 1}`. Preview-head specificity scores mostly 0.8 (first row 0.2); `concept_count` 4–5; `terms_original_count` 5; `has_topic` True for all sampled rows.
- Cell 10 (tuning notes) — markdown, no output.
- Cell 11 (Colab session footer): `Python 3 Google Compute Engine backend (GPU); Showing resources from 1:02 AM to 1:28 AM; System RAM 17.1 / 176.9 GB; GPU RAM 89.8 / 95.6 GB; Disk 72.2 / 235.7 GB`.

## Summary

The notebook is the Blackwell-optimized full-corpus run of the court-card LLM descriptor extractor, applying the 16-key descriptor schema from `Qwen/Qwen3-8B-AWQ` to all 363,258 LLM-enrichment-target rows of `court_authority_cards_v4_target_cards.jsonl` in a single Colab session. Speed optimizations vs. the prior 10k notebook include shrinking `max_model_len` to 2048, lifting `max_num_seqs` to 384, raising `gpu_memory_utilization` to 0.92, mega-batching 2048 prompts per `llm.generate()` call, moving the schema into the system prompt for full-prefix caching (315 tokens), removing the temperature-0 `repetition_penalty`, and writing JSONL to local SSD before a final copy to Drive. FlashInfer and FP8 KV are configured but disabled here: FlashInfer 0.6.x lacks SM 12.x kernels so vLLM auto-selected FlashAttention 4, and `kv_cache_dtype` was left at `None`. The run completed 363,258 rows in 15,895.0s (22.85 rows/sec, 18,161 tok/sec end-to-end), with 363,212 `ok`, 45 `ok_after_retry`, and 1 `failed_descriptor_parse`; QC found zero forbidden fields across the entire output.
