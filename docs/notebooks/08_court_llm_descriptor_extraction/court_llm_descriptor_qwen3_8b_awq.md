# court_llm_descriptor_qwen3_8b_awq

**Path:** e:\swiss_citation_extraction\notebooks\08_court_llm_descriptor_extraction\court_llm_descriptor_qwen3_8b_awq.ipynb

## Configuration

- Runtime: Google Colab / Kaggle GPU. Target runtime advertised in header: NVIDIA RTX PRO 6000 Blackwell Server Edition, ~96 GiB VRAM. Actual executed run (captured outputs) used Kaggle dual Tesla T4 (CUDA devices: 2; GPU 0/1 each ~14.56 GiB total, ~14.46 GiB free), with `CUDA_VISIBLE_DEVICES=0` (single-GPU mode).
- Model: `Qwen/Qwen3-8B-AWQ` (as declared in `Config`); executed run loaded local path `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1`.
- Inference engine: vLLM (`vllm>=0.20.0`, requires `transformers>=4.51.0`), AWQ quantization with `awq` -> `awq_marlin` -> `None` fallback chain. Executed run used `awq_marlin` kernel via `MarlinLinearKernel`/`AWQMarlinLinearMethod`, `TRITON_ATTN` attention backend.
- `Config` dataclass fields:
  - `base_dir`/`data_dir`/`output_dir`/`model_download_dir` derived from `BASE_DIR` (`/content/drive/MyDrive/swiss_law` in Colab, else `..`).
  - `input_csv`: `<DATA_DIR>/court_considerations.csv`; `fallback_input_csv`: `court_considerations.csv`.
  - Row selection: `start=0`, `limit=50`, `sample_random=False`, `random_seed=42`, `min_text_chars=80`, `max_text_chars=3200`.
  - GPU/vLLM: `gpu_mode='single'`, `tensor_parallel_size=1`, `gpu_memory_utilization=0.90`, `max_model_len=4096`, `max_num_seqs=16`, `batch_size=8`, `enforce_eager=False`, `quantization='awq'`, `disable_custom_all_reduce=True`, `force_triton_attention=False`, `use_structured_outputs=False`.
  - Generation: `max_new_tokens=384`, `retry_max_new_tokens=512`, `max_retries=1`, `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.02`, `enable_thinking=False`.
  - Output: `include_raw_output_on_success=False`.
- Executed `Config` overrides observed in output: `input_csv='/kaggle/input/competitions/llm-agentic-legal-information-retrieval/court_considerations.csv'`, `model_name='/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1'`, `gpu_memory_utilization=0.78`, `max_num_seqs=8`, `batch_size=4`, `enforce_eager=True`, `quantization='awq_marlin'`, `force_triton_attention=True`, `output_dir='/kaggle/working'`.
- Environment toggles: `TOKENIZERS_PARALLELISM=false`; when `force_triton_attention`, `VLLM_ATTENTION_BACKEND=TRITON_ATTN`.
- Output artifacts pattern: `court_llm_descriptors_{start:07d}_{end:07d}.jsonl` plus `_preview.csv`, `_failures.jsonl` (deleted if no failures), `_metrics.json`.

## Data

- Input CSV: `court_considerations.csv`. `resolve_input_path()` searches `cfg.input_csv`, `DATA_DIR/court_considerations.csv`, `BASE_DIR/court_considerations.csv`, `/content/court_considerations.csv`, `cwd/court_considerations.csv`, then recursive glob under `DATA_DIR`, `BASE_DIR`, `/content`.
- Executed run: `Using input: /kaggle/input/competitions/llm-agentic-legal-information-retrieval/court_considerations.csv`; `Shape: (2476315, 2)`; `Columns: ['citation', 'text']`.
- Filtering: drop NaN citation/text; require `_text_len >= 80` chars (stripped). Selected slice: `iloc[0:50]` since `sample_random=False`. `Selected rows: 50`.
- Head shows Swiss Federal Tribunal court considerations such as `BGE 139 I 2 E. 2`, `BGE 139 I 2 E. 5.1`, ..., `BGE 145 I 1 E. 5.1.1` with German-language reasoning paragraphs (e.g., "Eventualiter sei die Rückweisung an die Vorinstanz...", "Art. 34 Abs. 1 BV gewährleistet...").
- Texts are normalized via `trim_text()` (collapse whitespace, truncate to `max_text_chars=3200` keeping head+tail with `[TRUNCATED]` marker).

## Pipeline

- Cell 0 (markdown): scope statement. Notebook produces raw minimal semantic descriptors only; no anchor normalization, outcomes, summaries, questions, or retrieval views. Downstream local CPU scripts (`court_enrichment_profile.py`, `court_enrichment_normalizer.py`, `finalize_court_enrichment_from_llm.py`) build final views.
- Cell 1 (Cell 0): Colab detection, Drive mount, install of `vllm`, `transformers`, `accelerate`, `safetensors`, `pandas`, `tqdm`, `huggingface_hub`.
- Cell 2 (Cell 1): imports, `BASE_DIR` resolution, CUDA device report.
- Cell 3 (Cell 2): `Config` dataclass instantiation, GPU mode environment setup, output directories created, output paths derived.
- Cell 4 (Cell 3): `resolve_input_path()`, CSV load, NaN/length filtering, deterministic row slice, preview display.
- Cell 5 (Cell 4): `DESCRIPTOR_KEYS` (16 fields), `ROLE_VALUES` whitelist (11 values), `LLM_SCHEMA_HINT`, `SYSTEM_PROMPT` (Swiss legal descriptor extractor, JSON-only, query-neutral, no anchors/outcomes/views), `USER_TEMPLATE`, `trim_text()`, `build_user_prompt()`.
- Cell 6 (Cell 5): `extract_json_object()` (strips fences and `<think>` blocks, balanced-brace scan, trailing-comma repair, `ast.literal_eval` fallback), `clean_str()`/`clean_list()` (whitespace, length, dedupe casefold), `normalize_descriptor()` enforces per-field caps and `ROLE_VALUES` whitelist for `paragraph_role`, drops a hard-coded `forbidden` set (`statute_anchors`, `case_anchors`, `normalized_anchors`, `retrieval_views`, `outcome_signal`, `query_phrases_en`, `natural_language_queries`, `legal_question`, `summary_en`, `english_summary`, `enrichment_quality`, `anchor_quality_flags`), and `empty_descriptor(error)` placeholder.
- Cell 7 (Cell 6): vLLM/tokenizer load with `awq` -> `awq_marlin` -> `None` quantization fallback loop and CUDA cache clear on failure.
- Cell 8 (Cell 7): `render_prompt()` applies chat template (`enable_thinking=False`), includes a repair branch that pastes the prior bad output and parser error; `generate_raw()` uses `SamplingParams(temperature=0.0, top_p=1.0, repetition_penalty=1.02, max_tokens=...)`; `generate_descriptor()` does up to `max_retries+1` attempts and records per-attempt errors.
- Cell 9 (Cell 8): batch loop over `work_df` in `batch_size` slices, batch generation with fallback to per-row generation on failure, writes `output_jsonl`, `output_preview_csv`, optionally `output_failures_jsonl`, and `output_metrics_json`.
- Cell 10 (Cell 9): QC scan that recursively searches each record for any forbidden field keys, plus per-row counts of `concepts_en`, `terms_original`, `has_topic`, and `specificity_score`.
- Cell 11 (markdown): pointer to the local finalizer which consumes the JSONL via `normalize_enriched_court_row(...)` and produces final `rag_enrichment`, `normalized_anchors`, `anchor_quality_flags`, `retrieval_views`, `enrichment_quality`.

## Results

- Cell 1 (Cell 0) install output: pip dependency-resolver conflict warnings (e.g., `vllm 0.20.0 requires flashinfer-python==0.6.8.post1, which is not installed`, multiple `protobuf 7.34.1` incompatibilities with `google-cloud-*` and `tensorflow 2.19.0`, `numpy 2.2.6` vs `tensorflow 2.19.0` requirement `<2.2.0`, `gptqmodel 7.0.0 requires protobuf>=7.34.0, but you have protobuf 6.33.6`). Final lines: `WARNING: Skipping flashinfer as it is not installed.`, `Found existing installation: flashinfer-python 0.6.8.post1`, `Uninstalling flashinfer-python-0.6.8.post1: Successfully uninstalled flashinfer-python-0.6.8.post1`, `Setup done. Now restart the Kaggle session/kernel, then run from the next cell.`
- Cell 2 (Cell 1): `Imports OK`, `CUDA devices: 2`, `GPU 0: Tesla T4; free=14.46 GiB / total=14.56 GiB`, `GPU 1: Tesla T4; free=14.46 GiB / total=14.56 GiB`.
- Cell 3 (Cell 2): printed JSON of effective `Config` (Kaggle paths, `awq_marlin`, `enforce_eager=True`, `gpu_memory_utilization=0.78`, `max_num_seqs=8`, `batch_size=4`, `force_triton_attention=True`, `output_dir='/kaggle/working'`), and `Output JSONL: /kaggle/working/court_llm_descriptors_0000000_0000049.jsonl`.
- Cell 4 (Cell 3): `Using input: /kaggle/input/competitions/llm-agentic-legal-information-retrieval/court_considerations.csv`, `Shape: (2476315, 2)`, `Columns: ['citation', 'text']`, `Selected rows: 50`; preview DataFrame of 20 rows from `BGE 139 I 2 E. 2` ... `BGE 139 I 2 E. 5.7.6` with `_text_len` 183-2405.
- Cell 5 (Cell 4) and Cell 6 (Cell 5): no outputs.
- Cell 7 (Cell 6) vLLM load: `[vLLM] using explicit TRITON attention_config`, `WARNING ... Unknown vLLM environment variable detected: VLLM_ATTENTION_BACKEND`, `Resolved architecture: Qwen3ForCausalLM`, `Using max model len 4096`, `The model is convertible to awq_marlin during runtime. Using awq_marlin kernel.`, `Chunked prefill is enabled with max_num_batched_tokens=8192.`, `Enforce eager set, disabling torch.compile and CUDAGraphs.`, `Initializing a V1 LLM engine (v0.20.0)` with `dtype=torch.float16`, `quantization=awq_marlin`, weight loading from NFS (`Checkpoint size: 5.68 GiB. Available RAM: 19.38 GiB.`), safetensors shards `100% Completed | 2/2 [00:55<00:00, 22.92s/it]`, `Loading weights took 55.10 seconds`, `Model loading took 5.71 GiB memory and 57.270029 seconds`, `Available KV cache memory: 4.79 GiB`, `GPU KV cache size: 34,848 tokens`, `Maximum concurrency for 4,096 tokens per request: 8.51x`, `init engine (profile, create kv cache, warmup model) took 20.42 s`, final `vLLM loaded`.
- Cell 8 (Cell 7): no outputs.
- Cell 9 (Cell 8) run metrics (printed JSON):
  ```
  {
    "start": 0,
    "limit": 50,
    "selected_rows": 50,
    "written_rows": 50,
    "failures": 0,
    "elapsed_seconds": 211.5400824546814,
    "rows_per_second": 0.23636182523806848,
    "status_counts": { "ok": 50 },
    "output_jsonl": "/kaggle/working/court_llm_descriptors_0000000_0000049.jsonl",
    "output_preview_csv": "/kaggle/working/court_llm_descriptors_0000000_0000049_preview.csv",
    "output_failures_jsonl": null
  }
  ```
  Preview DataFrame shows 50 rows, all `status=ok`. Sample `legal_area` values: `Administrative law`, `Constitutional law`, `Land use and planning`, `public law`. Example row 0 (`BGE 139 I 2 E. 2`): `primary_domain=Public law`, `secondary_domain=Administrative procedure`, `topic=Administrative review`, `subtopic=Remand for reconsideration`, `micro_topic=Remand for re-examination and re-evaluation`, `paragraph_role=reasoning`, `authority_role=application_of_rule | legal_standard`, `specificity_score=0.8`, `terms_original=Rückweisung | Neubehandlung | Sachverhaltsabkl...`. Example row 1 (`BGE 139 I 2 E. 5.1`): `primary_domain=Public law`, `secondary_domain=Local government law`, `topic=Constitutional compliance`, `subtopic=Compatibility of local decisions with referendums`.
- Cell 10 (Cell 9) QC: per-row `forbidden_fields=[]` for all rows, `concept_count` mostly 4 (range 3-6), `terms_original_count` 3-6, `has_topic=True` for sampled rows, `specificity_score` ranges 0.3-0.8 in the visible head. Final summary lines: `Forbidden field rows: 0`, `Failed rows: 0`.
- Cell 11 (markdown): no executed output.

## Summary

This notebook runs the GPU-only LLM descriptor extraction step of the Swiss court enrichment pipeline, calling `Qwen/Qwen3-8B-AWQ` via vLLM on Colab/Kaggle to convert rows of `court_considerations.csv` into raw minimal semantic descriptors (16 fields covering legal area, domain path, topic/subtopic/micro_topic, English concepts, source-language terms, doctrinal rule, legal test, fact pattern, procedural context, paragraph and authority roles, and specificity score). The prompt and `normalize_descriptor` enforce a strict whitelist and drop forbidden post-processing fields (anchors, outcomes, retrieval views, queries, summaries), since those are produced by separate local CPU scripts (`court_enrichment_profile.py`, `court_enrichment_normalizer.py`, `finalize_court_enrichment_from_llm.py`). The captured run processed 50 rows (`start=0`, `limit=50`) from the Kaggle copy of the CSV (`2476315 x 2`) on a single Tesla T4 with `awq_marlin` + `TRITON_ATTN`, completing in 211.54 s (0.236 rows/s) with 50/50 `ok` status, zero failures, and zero forbidden fields detected by the QC pass. Outputs written: `court_llm_descriptors_0000000_0000049.jsonl`, its `_preview.csv`, and `_metrics.json` under `/kaggle/working/`; no failures JSONL was produced.
