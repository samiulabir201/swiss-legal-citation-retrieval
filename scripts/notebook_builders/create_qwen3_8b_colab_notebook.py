"""Create a Qwen3-8B-AWQ Colab version of the heavy Qwen3.5 notebook."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def cell_text(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_cell_text(cell: dict, text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def patch_notebook(src_path: Path, out_path: Path) -> None:
    nb = json.loads(src_path.read_text(encoding="utf-8"))

    # Title/overview.
    if nb["cells"] and nb["cells"][0]["cell_type"] == "markdown":
        set_cell_text(
            nb["cells"][0],
            """# Enrich Court Authority Cards — Qwen3-8B-AWQ + vLLM

Batch-enriches the compact `court_authority_cards_v4_target_cards.jsonl` file with English legal RAG metadata.

This is the low-cost Colab version of the Qwen3.5-35B notebook:

- Uses `Qwen/Qwen3-8B-AWQ` via vLLM offline inference.
- Uses deterministic, schema-constrained JSON generation.
- Disables Qwen thinking at chat-template render time.
- Keeps deterministic auto-classification for notification/cost/remittal/disposition/short paragraphs.
- Keeps reference-grounding validation so the model cannot invent statutes, article numbers, or case citations unchecked.
- Uses target-card-only input, checkpointed output, and failure capture.

Start with `TARGET_LIMIT = 1000`. After quality looks good, set `TARGET_LIMIT = 0` for the full 363,258-card target run.
""",
        )

    # Config cell.
    config_src = cell_text(nb["cells"][6])
    replacements = {
        "MODEL_ID     = 'Qwen/Qwen3.5-35B-A3B'": "MODEL_ID     = 'Qwen/Qwen3-8B-AWQ'",
        "QUANTIZATION = None": "QUANTIZATION = 'awq'",
        "BATCH_SIZE           = 64    # safe fast default for 95GB VRAM; try 96 then 128 only after stable load": (
            "BATCH_SIZE           = 16    # T4/L4-safe start; try 24/32 after smoke test"
        ),
        "REPETITION_PENALTY   = 1.0   # deterministic JSON; avoids extra decoding penalty": (
            "REPETITION_PENALTY   = 1.05  # helps small model avoid repeated JSON fragments"
        ),
        "MAX_TOKENS           = 256   # major throughput win; schema is concise enough for this": (
            "MAX_TOKENS           = 224   # small-model fast path; raise to 256 only if truncated"
        ),
        "RETRY_MAX_TOKENS     = 512": "RETRY_MAX_TOKENS     = 384",
        "SMOKE_MAX_TOKENS     = 512": "SMOKE_MAX_TOKENS     = 384",
        "TEXT_CHARS           = 900": "TEXT_CHARS           = 700",
    }
    for old, new in replacements.items():
        if old in config_src:
            config_src = config_src.replace(old, new)
    config_src = config_src.replace(
        "# First test: 1000. Full targeted run: 0.\nTARGET_LIMIT         = 0     # 0 means full run; set 1000 for a quick test",
        "# First test: 1000. Full targeted run: 0.\nTARGET_LIMIT         = 1000  # 0 means full run; use 1000 for first smoke/quality run",
    )
    config_src = config_src.replace(
        "print('MODEL_ID   :', MODEL_ID)",
        "print('MODEL_ID   :', MODEL_ID)\nprint('QUANTIZATION:', QUANTIZATION)",
    )
    set_cell_text(nb["cells"][6], config_src)

    # Markdown heading before model load.
    if len(nb["cells"]) > 11 and nb["cells"][11]["cell_type"] == "markdown":
        set_cell_text(
            nb["cells"][11],
            """## 5 · Load Qwen3-8B-AWQ

This loads the dense 8B AWQ model in FP16. It should be much faster and lighter than Qwen3.5-35B-A3B, but use the smoke test before the full run because quality is lower.
""",
        )

    # Model load cell: remove MoE-specific kernel config and use FP16/AWQ-friendly vLLM kwargs.
    load_cell = r'''import os
import inspect

# Must be set before initializing vLLM.
os.environ['VLLM_LOGGING_LEVEL'] = 'INFO'
os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'

# Remove stale/invalid env vars from prior experiments.
os.environ.pop('VLLM_WORKER_MULTIPROC_METHOD', None)
os.environ.pop('VLLM_MOE_BACKEND', None)
os.environ.pop('VLLM_FLASHINFER_MOE_BACKEND', None)

from vllm import LLM, SamplingParams

try:
    from vllm.sampling_params import StructuredOutputsParams
except Exception:
    StructuredOutputsParams = None

try:
    from vllm.sampling_params import GuidedDecodingParams
except Exception:
    GuidedDecodingParams = None


def build_llm_kwargs():
    kwargs = dict(
        model=MODEL_ID,
        quantization=QUANTIZATION,
        dtype='float16',
        gpu_memory_utilization=GPU_MEMORY_UTIL,
        max_model_len=MAX_MODEL_LEN,
        tensor_parallel_size=TENSOR_PARALLEL,
        trust_remote_code=True,
        enforce_eager=True,
        enable_prefix_caching=ENABLE_PREFIX_CACHE,
        disable_log_stats=True,
        limit_mm_per_prompt={'image': 0, 'video': 0},

        # Dense 8B model: no MoE kernel config needed. Keep batch knobs conservative.
        max_num_seqs=max(BATCH_SIZE, 32),
        max_num_batched_tokens=8192,
    )

    sig = inspect.signature(LLM.__init__)
    if 'language_model_only' in sig.parameters:
        kwargs['language_model_only'] = True

    print('Final vLLM kwargs actually passed:')
    for k, v in kwargs.items():
        print(f'  {k}: {v}')

    return kwargs


def make_sampling_params(schema, *, max_tokens: int):
    base = dict(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        max_tokens=max_tokens,
    )

    if StructuredOutputsParams is not None:
        try:
            params = SamplingParams(
                **base,
                structured_outputs=StructuredOutputsParams(json=schema),
            )
            print(f'Using StructuredOutputsParams(json=...) with max_tokens={max_tokens}')
            return params
        except TypeError as e:
            print('StructuredOutputsParams failed; trying GuidedDecodingParams:', repr(e))

    if GuidedDecodingParams is not None:
        params = SamplingParams(
            **base,
            guided_decoding=GuidedDecodingParams(json=schema),
        )
        print(f'Using GuidedDecodingParams(json=...) with max_tokens={max_tokens}')
        return params

    raise RuntimeError('No vLLM structured JSON decoding support found. Do not run unconstrained generation.')


print('Loading Qwen3-8B-AWQ with vLLM offline inference...')
print(f'MODEL_ID={MODEL_ID}, QUANTIZATION={QUANTIZATION}')
print(
    f'max_model_len={MAX_MODEL_LEN}, gpu_memory_utilization={GPU_MEMORY_UTIL}, '
    f'batch_size={BATCH_SIZE}, max_tokens={MAX_TOKENS}, retry_max_tokens={RETRY_MAX_TOKENS}'
)

llm = LLM(**build_llm_kwargs())
tokenizer = llm.get_tokenizer()

sampling_params = make_sampling_params(RAG_SCHEMA, max_tokens=MAX_TOKENS)
retry_sampling_params = make_sampling_params(RAG_SCHEMA, max_tokens=RETRY_MAX_TOKENS)
smoke_sampling_params = make_sampling_params(RAG_SCHEMA, max_tokens=SMOKE_MAX_TOKENS)

print('Sampling params:', sampling_params)
print('Model loaded successfully.')
'''
    set_cell_text(nb["cells"][12], load_cell)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python scripts/create_qwen3_8b_colab_notebook.py <source.ipynb> <output.ipynb>")
    patch_notebook(Path(sys.argv[1]), Path(sys.argv[2]))
