# auth_cards_enrich_qwen35_vllm_fixed_pillowfix

**Path:** `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen35_vllm_fixed_pillowfix.ipynb`

## Configuration

- Model: `Qwen/Qwen3.5-35B-A3B` (BF16, `quantization=None`).
- Runtime: vLLM `0.20.0` offline inference (`LLM.generate`), no Transformers `model.generate`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB), driver 580.82.07, CUDA 13.0.
- Engine settings: `dtype=bfloat16`, `max_model_len=4096`, `gpu_memory_utilization=0.90`, `tensor_parallel_size=1`, `trust_remote_code=False`, `enforce_eager=True`, `enable_prefix_caching=False`, `disable_log_stats=True`, `reasoning_parser="qwen3"`, `limit_mm_per_prompt={"image": 0, "video": 0}`, `kernel_config=KernelConfig(moe_backend="triton")` (Blackwell FlashInfer CUTLASS MoE bypass), `language_model_only=True` when the signature exposes it.
- Sampling: `temperature=0.15`, `max_tokens=384`, `BATCH_SIZE=32`, structured output via `StructuredOutputsParams(json=RAG_SCHEMA)` with fallback to `GuidedDecodingParams(json=...)`.
- Thinking disabled at chat-template render time with `enable_thinking=False`; chain-of-thought disclaimer appended if the template rejects that kwarg.
- Env vars: `VLLM_LOGGING_LEVEL=DEBUG`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, `VLLM_MOE_BACKEND` / `VLLM_FLASHINFER_MOE_BACKEND` / `VLLM_WORKER_MULTIPROC_METHOD` popped.
- Recipe flags declared but not actually wired into the loader: `USE_MTP=True`, `MTP_TOKENS=1` (no `speculative_config` in `build_llm_kwargs`).
- Dependency stack pinned: `pillow==11.3.0`, `numpy==2.3.5`, `scipy==1.16.3`; vLLM installed via `uv pip install --system vllm --torch-backend=auto`. Resulting versions reported by sanity check: Python 3.12.13, Torch 2.11.0+cu130, Torchvision 0.26.0+cu130, NumPy 2.3.5, SciPy 1.16.3, Pillow 11.3.0, Transformers 5.7.0.
- Paths: `BASE_DIR=/content/drive/MyDrive/swiss_law`; `INPUT_FILE=BASE_DIR/artifacts/court_authority_cards_v4.jsonl`; `OUTPUT_FILE=BASE_DIR/artifacts/court_authority_cards_rag.jsonl`; `CHECKPOINT_FILE=BASE_DIR/artifacts/rag_checkpoint.txt`.
- Run cap: `LIMIT=1000`.

## Data

- Input: `court_authority_cards_v4.jsonl` on Google Drive (`/content/drive/MyDrive/swiss_law/artifacts/`), reported `Total=2,476,315` lines.
- Output: `court_authority_cards_rag.jsonl` (append mode), one input card per line with an added `rag_enrichment` object.
- Checkpoint: `rag_checkpoint.txt` storing the next line offset to resume from. Resumed at line 166 in this run.
- Per-card input fields consumed: `text_excerpt_original` (truncated to 2500 chars), `citation`, `legal_area`, `issue_labels_en` (first 8), `is_notification_paragraph`.
- Pre-filter `COST_PROC_RE` matches German/French/Italian boilerplate for court costs, procedural fees, legal aid, remand-to-instance, and short cost-line patterns.
- Run scope: 1,000 cards processed in the loop window (`To process=1,000`, final output count 1,166 due to the resumed offset).

## Pipeline

1. Detect Colab; print GPU info.
2. Clean and reinstall numeric/image stack (`pillow`, `numpy`, `scipy`) under pinned constraints, then install `vllm` via `uv pip install --system ... --torch-backend=auto -c constraints`; run dependency and import sanity checks.
3. Mount Google Drive at `/content/drive`.
4. Define paths, model id, inference knobs, and file locations.
5. Define `RAG_SCHEMA` (11 fields, 7 required, `paragraph_role` enum of 8, `outcome_signal` enum of 6) and `SYSTEM_PROMPT` for a Swiss legal analyst emitting English RAG JSON.
6. Helpers: `build_user_message` (citation + legal_area + existing labels + paragraph), `_stub` (default enrichment), `auto_classify` (notification, length < 50, cost/procedural regex), `stream_input`, `count_lines`.
7. Set env vars, import vLLM, build `LLM` kwargs (forcing Triton MoE), build `SamplingParams` with structured outputs, instantiate `LLM`, fetch tokenizer.
8. Enrichment loop: read checkpoint, stream input from offset, fast-path via `auto_classify`, otherwise buffer to `pending` and flush at `BATCH_SIZE=32`.
9. Per flush: render Qwen chat prompts (`enable_thinking=False`), call `llm.generate(prompts, sampling_params, use_tqdm=False)`, strip Markdown fences and any `</think>` prefix, `json.loads`, normalize via `_stub` defaults and tag `method="qwen35_35b_a3b_vllm"`; on parse failure record stub with `method="json_parse_failed"`, `raw_output` (400-char prefix), and `parse_error`.
10. Append each enriched line to `OUTPUT_FILE`, flush, advance the checkpoint to `last_line + 1`, update tqdm.
11. Verification cell: iterate the output, count `paragraph_role`, `outcome_signal`, `method`, and rows missing required schema fields.
12. Sample cell: print three randomly chosen records whose `method` starts with `qwen35`.

## Results

Install / import sanity:

```
Python: 3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0]
Pillow OK: 11.3.0
NumPy OK: 2.3.5
SciPy OK: 1.16.3
NumPy internal symbol OK
Torch OK: 2.11.0+cu130 CUDA: 13.0 available: True
Torchvision OK: 0.26.0+cu130
vLLM import OK
vLLM version: 0.20.0
```

`pip check` flagged conflicts (none fatal for this run):

```
ipython 7.34.0 requires jedi, which is not installed.
libraft-cu12 26.2.0 has requirement cuda-toolkit[cublas,curand,cusolver,cusparse]==12.*, but you have cuda-toolkit 13.0.2.
pylibraft-cu12 26.2.0 has requirement cuda-python<13.0,>=12.9.2, but you have cuda-python 13.2.0.
pylibcudf-cu12 26.2.1 has requirement cuda-python<13.0,>=12.9.2, but you have cuda-python 13.2.0.
cudf-cu12 26.2.1 has requirement cuda-python<13.0,>=12.9.2, but you have cuda-python 13.2.0.
cudf-cu12 26.2.1 has requirement cuda-toolkit[nvcc,nvrtc]==12.*, but you have cuda-toolkit 13.0.2.
cudf-cu12 26.2.1 has requirement numba<0.62.0,>=0.60.0, but you have numba 0.65.0.
cuml-cu12 26.2.0 has requirement cuda-python<13.0,>=12.9.2, but you have cuda-python 13.2.0.
cuml-cu12 26.2.0 has requirement cuda-toolkit[cublas,cufft,curand,cusolver,cusparse]==12.*, but you have cuda-toolkit 13.0.2.
cuml-cu12 26.2.0 has requirement numba<0.62.0,>=0.60.0, but you have numba 0.65.0.
libcuml-cu12 26.2.0 has requirement cuda-toolkit[cublas,cufft,curand,cusolver,cusparse]==12.*, but you have cuda-toolkit 13.0.2.
cuvs-cu12 26.2.0 has requirement cuda-python<13.0,>=12.9.2, but you have cuda-python 13.2.0.
libcuvs-cu12 26.2.0 has requirement cuda-toolkit[cublas,curand,cusolver,cusparse]==12.*, but you have cuda-toolkit 13.0.2.
rmm-cu12 26.2.0 has requirement cuda-python<13.0,>=12.9.2, but you have cuda-python 13.2.0.
```

Loader notes:

- `Resolved architecture: Qwen3_5MoeForConditionalGeneration`, `max_model_len=4096`, chunked prefill with `max_num_batched_tokens=16384`, asynchronous scheduling enabled, eager mode (no CUDA graphs / torch.compile).
- Text-only mode confirmed: `All limits of multimodal modalities supported by the model are set to 0, running in text-only mode.`
- `Using TRITON Unquantized MoE backend out of potential backends: ['FlashInfer TRTLLM', 'FlashInfer CUTLASS', 'TRITON', 'BATCHED_TRITON'].`
- FlashInfer compilation warnings: `Failed to get device capability: SM 12.x requires CUDA >= 12.9.` (twice).
- Weight download: `Time spent downloading weights for Qwen/Qwen3.5-35B-A3B: 201.004344 seconds`, checkpoint size 66.97 GiB, OVERLAY filesystem, auto-prefetch disabled.
- `Loading weights took 7.44 seconds`; `Model loading took 64.69 GiB memory and 210.156106 seconds`.
- `Using default MoE config. Performance might be sub-optimal! Config file not found at .../E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition.json`.
- KV cache: `Available KV cache memory: 16.03 GiB`, `GPU KV cache size: 209,088 tokens`, `Maximum concurrency for 4,096 tokens per request: 113.57x`.
- `init engine (profile, create kv cache, warmup model) took 45.09 s`.
- Attention backend: `FLASH_ATTN` (FlashAttention v2).

Enrichment run (resume offset 166, limit 1000):

```
Resuming at line 166
Total=2,476,315  To process=1,000
Using BATCH_SIZE=32, MAX_TOKENS=384

[batch  1] size=32, batch_time=77.1s, batch_rate=0.42 cards/s, avg_rate=0.41 cards/s, eta~39.2 min, json_errors=29
[batch  5] size=32, batch_time=14.2s, batch_rate=2.25 cards/s, avg_rate=1.21 cards/s, eta~11.5 min, json_errors=129
[batch 10] size=32, batch_time=14.1s, batch_rate=2.26 cards/s, avg_rate=1.62 cards/s, eta~6.9 min, json_errors=269
[batch 15] size=32, batch_time=14.4s, batch_rate=2.23 cards/s, avg_rate=1.80 cards/s, eta~4.6 min, json_errors=400
[batch 20] size=32, batch_time=14.0s, batch_rate=2.28 cards/s, avg_rate=1.93 cards/s, eta~2.9 min, json_errors=535
[batch 25] size=32, batch_time=14.1s, batch_rate=2.27 cards/s, avg_rate=2.00 cards/s, eta~1.4 min, json_errors=661
[batch 30] size=32, batch_time=14.1s, batch_rate=2.26 cards/s, avg_rate=2.05 cards/s, eta~0.0 min, json_errors=800
Done. JSON parse errors: 801
Output -> /content/drive/MyDrive/swiss_law/artifacts/court_authority_cards_rag.jsonl
```

Verification of the output file (`Total output cards : 1,166`, `Missing required : 0`):

```
paragraph_role distribution:
  reasoning                    933  (80.0%)
  procedural                   144  (12.3%)
  disposition                   52  (4.5%)
  cost                          29  (2.5%)
  holding                        5  (0.4%)
  obiter                         1  (0.1%)
  standard_of_review             1  (0.1%)
  background                     1  (0.1%)

outcome_signal distribution:
  none                        1019  (87.4%)
  dismissed                     69  (5.9%)
  remitted                      56  (4.8%)
  inadmissible                  10  (0.9%)
  partial                        6  (0.5%)
  granted                        6  (0.5%)

method distribution:
  json_parse_failed            879  (75.4%)
  qwen35_35b_a3b_vllm          243  (20.8%)
  auto_short                    34  (2.9%)
  auto_cost                     10  (0.9%)
```

Sample LLM-enriched cards (`method` starts with `qwen35`):

```
Citation     : BGE 145 IV 23 E. 3.4
Role         : disposition
Outcome      : granted
Topic        : criminal law and criminal procedure
Summary      : Public dissemination of written materials satisfies the objective elements of the offense.
Keywords     : ['public dissemination', 'written materials', 'objective elements', 'infraction', 'constituent elements']
NL queries   : ['What constitutes the objective elements of an offense involving public written dissemination?', 'Does public action through writing satisfy the objective requirements of an infraction?']

Citation     : BGE 139 I 265 E. 3
Role         : disposition
Outcome      : dismissed
Topic        : Constitutional and Public Law
Summary      : The dispute and matter for adjudication is whether the Social Welfare Office of the City of St. Gallen correctly declined to enter upon the applicant's request for the disbursement of emergency assist
Keywords     : ['emergency assistance', 'social welfare office', 'refusal to enter upon', 'administrative discretion', 'St. Gallen', 'BGE 139 I 265']
NL queries   : ['When can a Swiss social welfare office refuse to enter upon a request for emergency assistance?', 'Judicial review of refusal to grant emergency assistance in Switzerland', 'BGE 139 I 265 emergency assistance St. Gallen']

Citation     : BGE 136 III 65 E. 2.1
Role         : reasoning
Outcome      : dismissed
Topic        : Civil Law
Summary      : The appellants argued that their lease could not be terminated prior to their exclusion from the cooperative society. The cantonal court held that, in this case, the two legal relationships were indep
Keywords     : ['lease termination', 'cooperative society exclusion', 'independent legal relationships', 'mixed contract', 'proportionality', 'Swiss Civil Code', 'BGE 136 III 65']
NL queries   : ['Can a lease be terminated before a member is excluded from a cooperative?', 'Are lease and cooperative membership independent legal relationships?', 'When does interference between lease termination and cooperative exclusion violate proportionality?']
```

GPU snapshot before load (`nvidia-smi`): RTX PRO 6000 Blackwell Server Edition, 0 MiB used of 97,887 MiB, 28C, 46W of 600W cap, no running processes.

## Summary

The notebook reinstalls a pinned numeric/image stack, installs vLLM 0.20.0 via `uv` with the auto torch backend, then loads `Qwen/Qwen3.5-35B-A3B` in BF16 on a single RTX PRO 6000 Blackwell (97 GiB) using vLLM offline inference with Triton MoE forced via `KernelConfig(moe_backend="triton")` to sidestep Blackwell FlashInfer CUTLASS issues. It enriches Swiss court authority cards from `court_authority_cards_v4.jsonl` into `court_authority_cards_rag.jsonl` by fast-path classifying notification, short, and cost paragraphs, then asking the LLM for an English RAG JSON object (`paragraph_role`, `outcome_signal`, summary, keywords, NL queries, etc.) under structured-output constraints. The run resumed at line 166 and processed 1,000 cards in roughly 7 minutes wall time per batch metrics, yielding 1,166 total output rows. JSON parsing failed on 801 of those (75.4% `method=json_parse_failed`; only 20.8% `qwen35_35b_a3b_vllm`, 2.9% `auto_short`, 0.9% `auto_cost`), so the structured-output path is not reliably emitting valid JSON in this configuration despite a default MoE config warning and the FlashInfer SM 12.x capability warnings. All 1,166 records still satisfy the required-field check because failed parses are filled with `_stub` defaults.
