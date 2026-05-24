# pipeline_no_debug_prints.ipynb

**Path:** notebooks/04_pre_v75_pipeline_iterations/pipeline_no_debug_prints.ipynb

> Note: Despite living in the Swiss citation extraction repo under `04_pre_v75_pipeline_iterations/`, this notebook's actual contents target the **AI Mathematical Olympiad – Progress Prize 3 (AIMO3)** Kaggle competition using the Nemotron Nano 30B A3B BF16 model served via vLLM. It has no Swiss legal / retrieval logic.

## Configuration

### Model
- Model: Nemotron Nano 30B A3B (BF16 shards), assembled at `/kaggle/working/nemotron-model` via symlinks
- `served_model_name`: `nemotron-nano`
- `dtype`: `bfloat16`
- `kv_cache_dtype`: `auto`
- `mamba-ssm-cache-dtype`: `float32`
- Reasoning parser plugin: custom `nano_v3` (subclasses `DeepSeekR1ReasoningParser`), written to `/kaggle/working/nano_v3_reasoning_parser.py`
- Tool-call parser: `qwen3_coder`
- `enable-auto-tool-choice`: on
- `trust-remote-code`: on

### Serving (vLLM)
- Required vLLM version: `>= 0.19.0` (notebook also enforces `>= 0.12.0` at runtime)
- Server: `vllm.entrypoints.openai.api_server` on `127.0.0.1:8000`, OpenAI client at `/v1`
- `tensor_parallel_size`: auto-detected from `torch.cuda.device_count()` (override via `AIMO_TP_SIZE`)
- `max_num_seqs`: 8 (fallback 4 via `AIMO_FALLBACK_MAX_NUM_SEQS`)
- `gpu_memory_utilization`: 0.90 (`AIMO_GPU_MEM_UTIL`)
- `max-model-len` / `context_tokens`: 262144 (`AIMO_CONTEXT_TOKENS`)
- `stream-interval`: 200
- `disable-log-stats`: on
- `enable_prefix_caching`: off by default (`AIMO_ENABLE_PREFIX_CACHING=1` to enable)
- `enable_async_scheduling`: on by default (`AIMO_ENABLE_ASYNC_SCHEDULING=0` to disable)
- FP8 MoE explicitly disabled: `VLLM_USE_FLASHINFER_MOE_FP8=0`, `VLLM_FLASHINFER_MOE_BACKEND` popped
- `server_timeout`: 1200s; `session_timeout`: 1200s

### Sampling / Reasoning
- `seed`: 42 (`set_seed`); per-attempt seed = `(seed + attempt_index + 1) ** 2`
- `turns`: 24 multi-turn loops per attempt (`AIMO_TURNS`)
- `attempts`: 32 attempt plans, all tool-enabled (`T01`–`T32`), `temperature=0.6`, `top_p=0.95`, `min_p=None`
- `max_tokens_reasoning`: 131072
- `max_tokens_tool`: 131072
- `context_buffer_tokens`: 2048
- `min_generation_tokens`: 128
- `tool_turn_buffer_tokens`: 1536
- `chat_turn_buffer_tokens`: 768
- `finalization_max_tokens`: 256 (final pass uses `temperature=0.0`, `top_p=1.0`, `enable_thinking=False`)
- `high_diversity_min_p`: 0.02
- `logprobs`: False (entropy weighting code is present but inactive)
- `top_logprobs`: 5
- `chat_template_kwargs`: `{"enable_thinking": True}` for solving turns
- Answer range: `min_answer=0`, `max_answer=999`

### Concurrency / Budgets
- `workers`: 8 (`AIMO_WORKERS`) — used as ThreadPoolExecutor size and sandbox pool size
- `early_stop`: 999
- `notebook_limit`: `5*3600 - 10*60` seconds = 17400s (`AIMO_NOTEBOOK_LIMIT_SEC`)
- `high_problem_timeout`: 780s
- `base_problem_timeout`: 180s
- `sandbox_timeout`: 60s
- `jupyter_timeout`: 120s
- `problems_remaining`: 50 (countdown)

### Prompts (verbatim from CFG)
- `system_prompt`: elite olympiad-level reasoner; 8-step core protocol; final boxed integer in `\boxed{N}`, N in 0–99999
- `tool_suffix`: "stateful Python verification tool ... exact checker, not a substitute for reasoning"
- `progress_prompt`: continue carefully; finalize with single `\boxed{N}`
- `finalization_prompt`: deterministic finalization, no new assumptions, no tool calls
- `tool_prompt`: stateful notebook tool; available modules `math, itertools, collections, fractions, functools, statistics, numpy, sympy, mpmath`

### Libraries / Env
- Uninstalled before install: `keras matplotlib scikit-learn tensorflow`
- Installed (offline, `--no-index`): `vllm==0.19.0`, `openai`, `polars`, `transformers`, `jupyter_client`, `packaging`
- Imports: `torch`, `pandas`, `polars`, `openai.OpenAI`, `jupyter_client.KernelManager`, `transformers.set_seed`, `kaggle_evaluation.aimo_3_inference_server`, `vllm`, `cbor2`, `msgspec`, `google.protobuf` (`< 6` required)
- Env vars set: `TRANSFORMERS_NO_TF=1`, `TRANSFORMERS_NO_FLAX=1`, `TOKENIZERS_PARALLELISM=false`, `PYTHONWARNINGS=ignore`, `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`, `VLLM_LOGGING_LEVEL=INFO`
- Sandbox kernel env: `PYDEVD_DISABLE_FILE_VALIDATION=1`, `PYDEVD_WARN_EVALUATION_TIMEOUT=0`, `JUPYTER_PLATFORM_DIRS=1`, `MPLBACKEND=Agg`
- Sandbox bootstrap: `import math, numpy, sympy, itertools, collections, fractions, functools, statistics, mpmath; mpmath.mp.dps = 80`

### Hardware
- `GPU_COUNT = torch.cuda.device_count()` (Kaggle GPU runtime)
- Multi-GPU TP supported; falls back to single-GPU if only one device

## Data

### Model weights (Kaggle dataset paths)
- `/kaggle/input/datasets/samiulislam180041221/nemotron-nano-bf16-b1`
- `/kaggle/input/datasets/samiulislam180041221/nemotron-nano-bf16-b2`
- `/kaggle/input/datasets/samiulislam180041221/nemotron-nano-bf16-b3`
- `/kaggle/input/datasets/samiulislam180041221/nemotron-nano-bf16-b4`
- `/kaggle/input/datasets/samiulislam180041221/nemotron-nano-bf16-b5`

Assembled (symlink farm) at: `/kaggle/working/nemotron-model`
Required files validated: `config.json`, `tokenizer.json`

### Wheels / utilities (offline install sources, preferred order)
- vLLM wheels: `/kaggle/input/datasets/samiulislam180041221/vllm-latest-wheels`, then `/kaggle/input/vllm-latest-wheels`, then any `vllm*.whl` under `/kaggle/input`
- Utils archive (`wheels.tar.gz`) extracted to `/kaggle/working/aimo_setup/utils_wheels`, preferred paths:
  - `/kaggle/input/notebooks/andreasbis/aimo-3-utils/wheels.tar.gz`
  - `/kaggle/input/aimo-3-utils/wheels.tar.gz`
  - `/kaggle/input/aimo3-utils/wheels.tar.gz`
  - `/kaggle/input/aimo-utils/wheels.tar.gz`
  - `/kaggle/input/utils/wheels.tar.gz`
- Optional: `TIKTOKEN_ENCODINGS_BASE` set to `<UTILS_DIR>/tiktoken_encodings` if present

### Competition input
- Local gateway CSV: `AIMO_LOCAL_CSV` env override, else `/kaggle/input/competitions/ai-mathematical-olympiad-progress-prize-3/test.csv`

### Runtime artifacts
- vLLM server log: `/kaggle/working/vllm_server.log`
- Reasoning parser plugin: `/kaggle/working/nano_v3_reasoning_parser.py`
- Setup dir: `/kaggle/working/aimo_setup`

## Pipeline

### Stage 1 — Environment cleanup
- `pip uninstall --yes keras matplotlib scikit-learn tensorflow`.
- Set `VLLM_USE_FLASHINFER_MOE_FP8=0` and pop `VLLM_FLASHINFER_MOE_BACKEND` BEFORE importing vLLM to keep BF16 MoE path.

### Stage 2 — Offline dependency install
- Resolve vLLM wheel directory (`_find_vllm_wheels`) and utils archive (`_find_utils_archive`).
- If `vllm < 0.19.0` or missing: `pip install --no-index --find-links <vllm_dir> --no-deps vllm==0.19.0`.
- Install other wheels in `vllm_dir` (excluding `vllm-*` itself and OpenTelemetry / protobuf wheels) with `--no-deps`.
- If utils archive present: extract and `pip install --no-index --find-links <UTILS_DIR>` for `openai polars transformers jupyter_client packaging`.
- Sanity-check `protobuf < 6`; import `cbor2`, `msgspec`, `vllm`.

### Stage 3 — Imports and runtime env
- Standard library + `torch`, `pandas`, `polars`, `IPython.display`, `packaging`, `openai`, `jupyter_client`, `transformers.set_seed`, `kaggle_evaluation.aimo_3_inference_server`, `vllm`.
- Sets `TRANSFORMERS_NO_TF`, `TRANSFORMERS_NO_FLAX`, `TOKENIZERS_PARALLELISM=false`, `PYTHONWARNINGS=ignore`, `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas`.
- Asserts `vllm.__version__ >= 0.12.0`. Reads `GPU_COUNT`.

### Stage 4 — Custom reasoning parser plugin
- Write `/kaggle/working/nano_v3_reasoning_parser.py` registering a `NanoV3ReasoningParser` (extends `DeepSeekR1ReasoningParser`) under the `nano_v3` name in `ReasoningParserManager`. When `enable_thinking=False` and `final_content is None`, swap reasoning and final content.

### Stage 5 — Model directory assembly
- `mkdir /kaggle/working/nemotron-model`.
- Symlink every file from each `nemotron-nano-bf16-bN` dataset part (skipping `dataset-metadata.json`).
- Verify `config.json` and `tokenizer.json` present.
- Detect TP size: `AIMO_TP_SIZE` env, else `torch.cuda.device_count()`.
- Define `CFG` dataclass with all hyperparameters and 32 attempt plans (`T01`..`T32`).

### Stage 6 — Sandbox + Tool definitions
- `AIMO3Sandbox`: spawns a Jupyter kernel via `KernelManager` with explicit ports (50000+ in increments of 5), captures stdout/stderr/errors, supports `execute` with timeout (kernel interrupt on timeout), `reset` via `%reset -f` plus bootstrap, and a teardown that intentionally avoids `shutdown_kernel()` to dodge Kaggle/Papermill async teardown bugs.
- `AIMO3Tool`: exposes a single OpenAI-tool function `stateful_python_code_exec` (description = `CFG.tool_prompt`). Auto-wraps the final line in `print(...)` when needed. Uses a per-tool execution lock.

### Stage 7 — Solver construction (`AIMO3Solver.__init__`)
1. `set_seed(seed)`.
2. `_preload_model_weights`: thread-pool reads every file in `model_path` (1 GB chunks) to warm the page cache.
3. `_launch_server_with_retries`: try `primary` plan, then `fallback` (smaller `max_num_seqs`, no `--async-scheduling`, no `--enable-prefix-caching`).
4. `_wait_for_server`: poll `client.models.list()` until ready or `server_timeout` exceeded; on death, raise with tail of `vllm_server.log`.
5. `_initialize_kernels`: launch `workers` `AIMO3Sandbox` instances in parallel into a `queue.Queue` pool.
6. Record `notebook_start_time`, `problems_remaining = 50`.

### Stage 8 — Per-problem solve (`solve_problem`)
1. Compute per-problem budget: `time_left = notebook_limit - elapsed`; reserve `base_problem_timeout` per remaining problem; clamp to `[base_problem_timeout, high_problem_timeout]`. Set `deadline`.
2. Launch all 32 attempt plans concurrently via `ThreadPoolExecutor(max_workers=workers)`.
3. Each attempt (`_process_attempt`):
   - Acquire a sandbox from the pool; wrap as `AIMO3Tool` if `use_tools`.
   - Build messages: `[system_prompt, user_problem + plan.prompt_suffix]`.
   - Up to `turns=24` chat-completion rounds:
     - Compute `turn_max_tokens` from remaining context.
     - Call `_chat_completion` with `tools=tool.tools`, `tool_choice="auto"`, `chat_template_kwargs={"enable_thinking": True}`.
     - If tool calls returned: execute each via the sandbox; append assistant message + tool messages; continue.
     - Else: scan content for `\boxed{N}`, `final answer is/=/:`, or a bare integer in `[0, 999]`. On valid answer, break. Otherwise append a `progress_prompt` user turn.
   - On exhaustion: `_finalize_from_messages` performs a deterministic finalization call (`temperature=0.0`, `enable_thinking=False`, max 256 tokens).
   - Score: `2.0 * has_answer + entropy_weight + 0.15 * python_calls - 0.35 * python_errors + 1/max(completion_tokens, 1)`.
4. `_select_answer`: tally votes across attempts; tiebreak by summed weighted score; display vote dataframe; return top.
5. Default to `0` if no valid answer.

### Stage 9 — Inference server entrypoint
- `predict(id_, question, answer=None)`: disable GC, call `solver.solve_problem(problem_text)`, re-enable GC and `gc.collect()`. Returns `pl.DataFrame({"id": row_id, "answer": int(final_answer)})`.
- `AIMO3InferenceServer(predict)`.
- If `KAGGLE_IS_COMPETITION_RERUN` set: `inference_server.serve()`. Else: `inference_server.run_local_gateway((local_csv,))` using `AIMO_LOCAL_CSV` or `/kaggle/input/competitions/ai-mathematical-olympiad-progress-prize-3/test.csv`.

## Results

The notebook contains no executed model-outputs, metric tables, vote dataframes, server-readiness traces, or competition scores. The only captured output is from the very first cell (Stage 1 pip uninstall), and it is itself truncated:

```
Found existing installation: keras 3.10.0
Uninstalling keras-3.10.0:
  Successfully uninstalled keras-3.10.0
Found existing installation: matplotlib 3.10.0
Uninstalling matplotlib-3.10.0:
```

(Output cuts off mid-uninstall; no further cells have stored outputs.)

No AIMO accuracy, no per-problem vote DataFrames, no `vllm_server.log` excerpts, no kernel timings, no `_preload_model_weights` elapsed values, no errors visible in the notebook.

## Summary

The notebook is a self-contained Kaggle submission template for AIMO Progress Prize 3 that serves a BF16 Nemotron Nano 30B A3B via vLLM 0.19.0 with a custom `nano_v3` reasoning parser, runs 32 parallel tool-enabled reasoning attempts per problem against a per-attempt stateful Jupyter sandbox, and aggregates answers with vote + weighted-score voting. The structural pieces — offline wheel install, symlink-based model assembly, primary/fallback vLLM launch with log capture, per-attempt deadlines tied to a global notebook budget, defensive sandbox teardown that skips `shutdown_kernel()` — show iteration against Kaggle-specific failure modes. There is no evidence of execution: only the pip-uninstall cell has output, so neither functional success nor accuracy can be confirmed from this notebook alone. Despite the file's location under `04_pre_v75_pipeline_iterations/`, the contents are unrelated to the Swiss DE/FR/IT citation retrieval / Macro F1 work; it appears the file was misfiled in this repo. Lesson: the project tree includes at least one off-topic artifact that should be relocated to an AIMO-specific directory to avoid confusing future readers.
