# kaggle_notebook_0cf83146ef

**Path:** `e:\swiss_citation_extraction\notebooks\13_misc_kaggle_and_utilities\kaggle_notebook_0cf83146ef.ipynb`

Court LLM Descriptor Extractor — Kaggle GPU Only. The notebook runs only the expensive LLM step on Kaggle GPU and writes raw minimal semantic descriptors from `court_considerations.csv` via Qwen3-8B-AWQ into `court_llm_descriptors_*.jsonl`. Anchor normalization, role/outcome correction, and final retrieval views are explicitly deferred to local CPU scripts (`scripts/court_enrichment_profile.py`, `scripts/court_enrichment_normalizer.py`, `scripts/finalize_court_enrichment_from_llm.py`).

## Configuration

Setup cell installs `transformers>=4.45.0`, `accelerate`, `safetensors`, `pandas`, `tqdm`, `gptqmodel`, then `vllm>=0.6.0`, and uninstalls `flashinfer`/`flashinfer-python`. Output flags Kaggle kernel restart before next cell.

`Config` dataclass values:
- `input_csv`: `/kaggle/input/competitions/llm-agentic-legal-information-retrieval/court_considerations.csv`
- `fallback_input_csv`: `court_considerations.csv`
- `model_name`: `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1`
- Row selection: `start=0`, `limit=50`, `sample_random=False`, `random_seed=42`, `min_text_chars=80`, `max_text_chars=3200`
- GPU/vLLM: `gpu_mode='single'`, `tensor_parallel_size=1`, `gpu_memory_utilization=0.78`, `max_model_len=4096`, `max_num_seqs=8`, `batch_size=4`, `enforce_eager=True`, `quantization='awq_marlin'`, `disable_custom_all_reduce=True`, `force_triton_attention=True`, `use_structured_outputs=False` (disabled because Kaggle vLLM v0.20 crashes with `AttributeError("'dict' object has no attribute '_backend'")`)
- Generation: `max_new_tokens=384`, `retry_max_new_tokens=512`, `max_retries=1`, `temperature=0.0`, `top_p=1.0`, `repetition_penalty=1.02`, `enable_thinking=False`
- Output: `output_dir='/kaggle/working'`, `include_raw_output_on_success=False`

Single-GPU mode sets `CUDA_VISIBLE_DEVICES=0`. `VLLM_ATTENTION_BACKEND=TRITON_ATTN` and `TOKENIZERS_PARALLELISM=false` are exported. Output paths are derived as `court_llm_descriptors_0000000_0000049.jsonl` plus `_preview.csv`, `_failures.jsonl`, `_metrics.json` siblings.

## Data

Input is the Kaggle competition CSV `court_considerations.csv` resolved through a candidate-path/glob lookup (`/kaggle/input/**/court_considerations.csv` fallback). Loaded shape: `(2476315, 2)` with columns `['citation', 'text']`. Rows are filtered to non-null citation/text and text length >= 80 chars, then the contiguous slice `[start:start+limit]` is taken (random sampling disabled), yielding 50 selected rows (`_source_row` 1-50) drawn from `BGE 139 I 2` and `BGE 145 I 1` subsection IDs. Sample preview includes German-language paragraphs (e.g., starting with "Eventualiter sei...", "Art. 34 Abs. 1 BV gewährleistet...") with `_text_len` ranging 118-2916 chars.

## Pipeline

1. Imports (`torch`, `pandas`, `tqdm`, etc.); CUDA reports 2x Tesla T4 with ~14.46 GiB free each.
2. `Config` is materialized; output filenames are derived from `start`/`limit`.
3. CSV resolution and row selection (length filter + slice).
4. Schema definitions: `DESCRIPTOR_KEYS` (16 fields covering taxonomy, concepts, terms, doctrinal rule/test, fact tags, procedural context, paragraph_role, authority_role, specificity_score), `ROLE_VALUES` whitelist, `LLM_SCHEMA_HINT` skeleton, and `SYSTEM_PROMPT` enforcing "JSON only" and explicit prohibitions on questions/summaries/statute anchors/case anchors/outcome inference/retrieval views.
5. `USER_TEMPLATE` injects citation, trimmed text (`trim_text` collapses whitespace and middle-truncates to `max_text_chars`), and the JSON schema hint.
6. JSON extraction (`extract_json_object`) strips code fences and `<think>` blocks, finds the balanced `{...}` span, and falls back to trailing-comma cleanup then `ast.literal_eval`.
7. `normalize_descriptor` coerces, length-caps, and dedupes each field; validates `paragraph_role` against `ROLE_VALUES` (default `neutral`); clamps `specificity_score` to `[0,1]`; deletes a forbidden field set (`statute_anchors`, `case_anchors`, `normalized_anchors`, `retrieval_views`, `outcome_signal`, `query_phrases_en`, `natural_language_queries`, `legal_question`, `summary_en`, `english_summary`, `enrichment_quality`, `anchor_quality_flags`). `empty_descriptor` provides the failure record.
8. vLLM load: `AutoTokenizer.from_pretrained(...)`, then `LLM(...)` with the configured kwargs; `AttentionConfig(backend='TRITON_ATTN')` is attempted explicitly. Engine resolves architecture `Qwen3ForCausalLM`, uses `awq_marlin` kernel, allocates 5.71 GiB for weights and 4.79 GiB for KV cache (34,848-token cache, ~8.5x concurrency at 4,096-token contexts).
9. `render_prompt` builds a Qwen chat-templated prompt (with optional repair payload). `generate_raw` calls `llm.generate` with the configured `SamplingParams`. `generate_descriptor` parses/normalizes the raw output and retries once with a repair prompt on parse failure.
10. Cell 8 iterates the working DataFrame in batches of 4 (13 batches for 50 rows), runs batched generation, parses each row, writes per-row records to `court_llm_descriptors_*.jsonl`, dumps failures (none), writes a flat `_preview.csv`, and serializes a metrics summary.
11. Cell 9 QC walks each record for forbidden field paths and reports per-row concept/term/topic counts and status.

## Results

Setup cell output: pip dependency-conflict warnings (vllm/bigframes/google-cloud/google-adk/protobuf/numpy/tensorflow incompatibilities, `flashinfer-python 0.6.8.post1` uninstalled), then `Setup done. Now restart the Kaggle session/kernel, then run from the next cell.`

Cell 1: `Imports OK`, `CUDA devices: 2`, `GPU 0: Tesla T4; free=14.46 GiB / total=14.56 GiB`, `GPU 1: Tesla T4; free=14.46 GiB / total=14.56 GiB`.

Cell 2: `Config` dict echoed (matches the values above), `Output JSONL: /kaggle/working/court_llm_descriptors_0000000_0000049.jsonl`.

Cell 3: `Using input: /kaggle/input/competitions/llm-agentic-legal-information-retrieval/court_considerations.csv`, `Shape: (2476315, 2)`, `Columns: ['citation', 'text']`, `Selected rows: 50`, plus a 20-row head display showing `_source_row` 1-20 (citations `BGE 139 I 2 E. 2` through `BGE 139 I 2 E. 6.3`) with text lengths from 118 to 2916.

Cell 6: vLLM logs report engine init with `Resolved architecture: Qwen3ForCausalLM`, `Using max model len 4096`, `awq_marlin` quantization, chunked prefill at `max_num_batched_tokens=8192`, async scheduling enabled, eager mode (CUDA graphs disabled), `Using AttentionBackendEnum.TRITON_ATTN backend`, NFS checkpoint of 5.68 GiB prefetched, weights loaded in 55.10 s (5.71 GiB), KV cache 4.79 GiB / 34,848 tokens / 8.51x concurrency, init engine took 20.42 s, final line `vLLM loaded`. A warning `Unknown vLLM environment variable detected: VLLM_ATTENTION_BACKEND` is emitted.

Cell 8 metrics:

```
{
  "start": 0,
  "limit": 50,
  "selected_rows": 50,
  "written_rows": 50,
  "failures": 0,
  "elapsed_seconds": 211.5400824546814,
  "rows_per_second": 0.23636182523806848,
  "status_counts": {"ok": 50},
  "output_jsonl": "/kaggle/working/court_llm_descriptors_0000000_0000049.jsonl",
  "output_preview_csv": "/kaggle/working/court_llm_descriptors_0000000_0000049_preview.csv",
  "output_failures_jsonl": null
}
```

Cell 8 preview table covers all 50 rows with `status='ok'` throughout. `legal_area` is predominantly `Constitutional law`/`Administrative law` (with some `Land use and planning`, `public law`, `public administration` variants and casing inconsistencies the local normalizer is meant to fix).

Cell 9 QC: per-row table shows `forbidden_fields: []` for every row; `concept_count` ranges 3-6 (mostly 4), `terms_original_count` ranges 3-8, `has_topic=True` for all rows, `specificity_score` ranges 0.3-0.8. Summary lines: `Forbidden field rows: 0`, `Failed rows: 0`.

Final markdown cell points to the local finalizer step that should call `normalize_enriched_court_row(citation, text, llm_enrichment, deterministic_metadata)` to produce the production JSONL with `rag_enrichment`, `normalized_anchors`, `anchor_quality_flags`, `retrieval_views`, and `enrichment_quality` fields.

## Summary

This Kaggle notebook is the GPU-only descriptor-extraction stage of a two-stage court-enrichment pipeline: it loads `court_considerations.csv`, slices 50 rows, and runs Qwen3-8B-AWQ on Tesla T4 via vLLM (TRITON attention, AWQ-Marlin quantization, eager mode, single-GPU, `gpu_memory_utilization=0.78`) to produce 16-field query-neutral JSON descriptors. The schema deliberately excludes statute/case anchors, outcomes, summaries, and retrieval views — those are deferred to local CPU finalizers; a forbidden-field deny list and QC pass enforce this contract. All 50 rows succeeded (`status=ok`, no failures, no forbidden fields) in 211.5 s (~0.24 rows/s), writing `court_llm_descriptors_0000000_0000049.jsonl` plus a preview CSV and metrics JSON to `/kaggle/working`. The run demonstrates the pipeline contract on `BGE 139 I 2` and `BGE 145 I 1` German-language paragraphs, with `legal_area` casing/synonym inconsistencies expected to be normalized downstream.
