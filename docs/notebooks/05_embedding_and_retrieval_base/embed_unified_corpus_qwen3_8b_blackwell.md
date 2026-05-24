# embed_unified_corpus_qwen3_8b_blackwell.ipynb

**Path:** `notebooks/05_embedding_and_retrieval_base/embed_unified_corpus_qwen3_8b_blackwell.ipynb`

## Configuration

### Model
- `MODEL_NAME = 'Qwen/Qwen3-Embedding-8B'`
- `MODEL_DIM = 4096`
- `MAX_SEQ_LEN = 768`
- Weights loaded in `torch.bfloat16`
- Tokenizer `padding_side='left'` (last-token pooling)
- `model.max_seq_length = 768`
- Document side: no instruction prefix (`document_prompt_prefix: None`)

### Embedding pipeline hyperparameters
- `BATCH_SIZE = 256`
- `CHUNK_SIZE = 100_000` rows per `.npy` file
- `OUTPUT_DTYPE = np.float16`
- `USE_TORCH_COMPILE = False`
- L2-normalized embeddings (`normalize_embeddings=True`)
- Length-sorted batching within each chunk (sort indices by `len(texts[i])`, restore original order after encode)

### Libraries / runtime
- `sentence-transformers>=3.3`
- `transformers>=4.51`
- `accelerate>=0.34`
- `pyarrow>=15`
- `einops`
- `flash-attn-4` (pure-Python CuTeDSL JIT wheel; spec `"flash-attn-4"` for CUDA major < 13, `"flash-attn-4[cu13]"` otherwise)
- `torch 2.10.0+cu128`
- CUDA 12.8
- `os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'`
- `torch.set_float32_matmul_precision('high')`
- `torch.backends.cuda.matmul.allow_tf32 = True`
- `torch.backends.cudnn.allow_tf32 = True`
- SDPA kernels enabled when SDPA selected: `enable_flash_sdp`, `enable_cudnn_sdp`, `enable_mem_efficient_sdp`

### Attention backend selection
Hardware-aware preference chain:
- Blackwell (sm_100+): `flash_attention_4` → `flash_attention_2` → `sdpa`
- Hopper (sm_90): `flash_attention_3` → `flash_attention_2` → `sdpa`
- Older (sm_8x): `flash_attention_2` → `sdpa`

A candidate is selected only if its symbol imports AND its string appears in `transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS` valid keys. FA2 probe requires BOTH `flash_attn.flash_attn_func` AND `flash_attn.bert_padding.unpad_input` (since `flash-attn-4` owns the `flash_attn` namespace too).

### Hardware (detected at runtime)
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- VRAM: `102.0 GB`
- Compute capability: `sm_120` (Blackwell)
- bf16 supported: `True`
- TF32 matmul (pre-config): `False`

## Data

### Input
- `DRIVE_INPUT = '/content/drive/MyDrive/swiss_law/data/unified_embedding_input.parquet'`
- Produced locally by `scripts/prepare_embedding_input.py`
- Columns: `['doc_id', 'family', 'citation', 'vector_text', 'char_len']`
- Row count: `2,652,248`
- Family counts: `{'court': 2476315, 'law': 175933}`
- `char_len` stats: mean `373`, p50 `309`, p90 `612`, p99 `1414`, max `2,845`
- Load time: `9.7s`

### Output directory and prefixes
- `DRIVE_OUT_DIR = '/content/drive/MyDrive/swiss_law/artifacts/embeddings'`
- `OUT_PREFIX = 'qwen3_8b_unified'`
- Chunk files: `qwen3_8b_unified_chunk000.npy` … `qwen3_8b_unified_chunk026.npy` (27 chunks; last chunk 52,248 rows)
- Manifest: `/content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_manifest.parquet` (2,652,248 rows; columns `doc_id, family, citation, row_index`)
- Summary: `/content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_summary.json`

### Drive mount
- `drive.mount('/content/drive', force_remount=False)` → `Mounted at /content/drive`

## Pipeline

### Stage 1 — Install dependencies
Upgrade pip, install `sentence-transformers`, `transformers`, `accelerate`, `pyarrow`, `einops`. Detect CUDA major; select `flash-attn-4` spec; install (stable, fall back to `--pre`). Post-install probe imports `flash_attn.cute.flash_attn_func`. FA2 (`flash-attn` 2.x) is intentionally skipped on Blackwell (no PyPI wheel, source build is 30 min+, SDPA cuDNN flash already used).

### Stage 2 — GPU + library check
Assert `torch.cuda.is_available()`. Print device properties (name, VRAM, sm cap, torch/CUDA version, bf16). Enable TF32 paths and set high-precision matmul.

### Stage 3 — Configuration block
Bind `MODEL_NAME`, `MODEL_DIM`, `MAX_SEQ_LEN`, `BATCH_SIZE`, `CHUNK_SIZE`, `OUTPUT_DTYPE`, `USE_TORCH_COMPILE`, `DRIVE_INPUT`, `DRIVE_OUT_DIR`, `OUT_PREFIX`. `os.makedirs(DRIVE_OUT_DIR, exist_ok=True)`.

### Stage 4 — Mount Drive
`from google.colab import drive; drive.mount('/content/drive', force_remount=False)`.

### Stage 5 — Load parquet
`df = pd.read_parquet(DRIVE_INPUT)` into pandas. Report row count, columns, family counts, `char_len` distribution.

### Stage 6 — Load model with best attention backend
Detect compute capability; build per-arch preference list. Probe FA4/FA2/FA3 imports. Read `ALL_ATTENTION_FUNCTIONS` valid keys. Pick the first viable candidate; fall back to `sdpa`. On SDPA selection, enable flash + cuDNN + mem-efficient SDPA kernels.

```
model = SentenceTransformer(
    MODEL_NAME, device='cuda',
    model_kwargs={'torch_dtype': torch.bfloat16, 'attn_implementation': ATTN_IMPL},
    tokenizer_kwargs={'padding_side': 'left'},
)
model.max_seq_length = MAX_SEQ_LEN
model.eval()
```

If `USE_TORCH_COMPILE=True`, wrap `model[0].auto_model` with `torch.compile(mode='reduce-overhead', fullgraph=False)`; on exception print fallback message. Print allocated/reserved VRAM.

### Stage 7 — Define encoder
- `encode_chunk(model, texts, batch_size)`: sort indices by text length, encode with `model.encode(..., show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)` under `torch.inference_mode()`, build inverse-permutation, return fp32 normalized embeddings in original order.
- `encode_corpus(df, out_dir, prefix, chunk_size, batch_size)`: iterate `n_chunks = ceil(n/chunk_size)`. For each chunk, skip if existing `.npy` matches expected row count, else encode, cast to `OUTPUT_DTYPE` (fp16), `np.save`, free memory via `gc.collect()` and `torch.cuda.empty_cache()`. Collect per-chunk timings.

### Stage 8 — Run encoding job
Call `encode_corpus(...)` with the configured chunk/batch sizes. Print per-chunk progress and totals (total minutes, avg/chunk, first chunk, steady-state avg from chunk 2 onward).

### Stage 9 — Write manifest + summary
Build manifest from `df[['doc_id', 'family', 'citation']]` with `row_index = np.arange(len(df))`, write parquet. Build summary dict (model name/dim/dtype, normalization flag, attention impl, seq/batch/chunk, compile flag, chunk count, row count, family counts, GPU/torch/CUDA versions); write `qwen3_8b_unified_summary.json`.

### Stage 10 — Sanity check chunks
Glob `OUT_PREFIX_chunk*.npy`, mmap-load each. For first 2000 rows of each: compute L2 norms (cast fp32), count NaNs. Print shape/dtype/norm stats/NaN count. Assert total rows equals `len(df)`.

### Stage 11 — Smoke retrieval test
Build query template:
```
QWEN_INSTRUCT = (
    'Instruct: Given an English-language legal question or scenario about Swiss federal law, '
    'retrieve the Swiss statute articles or federal court decision considerations that are most '
    'directly relevant to answering it.\nQuery: '
)
```
Encode three synthetic English queries through the instruct template (batch 8, normalized, fp32). Load chunk 0 fp32, compute scores `q_emb @ doc_emb.T`, print top-5 per query.

## Results

### Stage 2 — GPU probe output
```
GPU            : NVIDIA RTX PRO 6000 Blackwell Server Edition
VRAM           : 102.0 GB
Compute cap.   : sm_120
Torch          : 2.10.0+cu128  CUDA: 12.8
bf16 supported : True
TF32 matmul    : False
```

### Stage 1 — Flash-attention install probe
```
installing flash-attn-4 (cuda major detected: 12, spec: "flash-attn-4") ...
flash-attn-4   : OK (flash_attn.cute available)
flash-attn (FA2): skipped (not needed on Blackwell — SDPA cuDNN flash is used)
```

### Stage 5 — Parquet load
```
rows           : 2,652,248
load time      : 9.7s
columns        : ['doc_id', 'family', 'citation', 'vector_text', 'char_len']
family counts  : {'court': 2476315, 'law': 175933}
avg chars      : 373
p50/p90/p99    : {0.5: 309.0, 0.9: 612.0, 0.99: 1414.0}
max chars      : 2,845
```

### Stage 6 — Backend selection and model load
```
compute cap.   : sm_120  (blackwell=True, hopper=False)
flash-attn-4   : available
flash-attn (FA2): missing
flash-attn-3   : missing
attention      : sdpa
sdpa kernels   : flash + cudnn + mem-efficient enabled
load time      : 7.6s
allocated VRAM : 15.13 GB
reserved VRAM  : 15.14 GB
```
Deprecation warning emitted: `The 'tokenizer_kwargs' argument was renamed and is now deprecated. Please use 'processor_kwargs' instead.`

Despite `flash-attn-4` being importable, the registry test for `flash_attention_4` failed (HF integration pending PR #42405), so backend selected = `sdpa`. FA2 absent because the bare `flash_attn` namespace was claimed by FA4 but `flash_attn.flash_attn_func` / `flash_attn.bert_padding.unpad_input` did not import.

### Stage 8 — Encoding run
All 27 chunks were already cached from a previous run; no encode timings were produced this run.
```
corpus rows    : 2,652,248
chunks         : 27  (chunk_size=100000, batch=256)
  [cached] chunk 1/27  rows=100,000  shape=(100000, 4096)
  ...
  [cached] chunk 27/27  rows=52,248  shape=(52248, 4096)
```
No per-chunk encode times printed (cache hits skip timing).

### Stage 9 — Manifest + summary
```
manifest       : /content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_manifest.parquet  (2,652,248 rows)
summary        : /content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_summary.json
```
Summary JSON written:
```json
{
  "model_name": "Qwen/Qwen3-Embedding-8B",
  "embedding_dim": 4096,
  "output_dtype": "float16",
  "normalized": true,
  "document_prompt_prefix": null,
  "attention_impl": "sdpa",
  "max_seq_length": 768,
  "batch_size": 256,
  "chunk_size": 100000,
  "torch_compile": false,
  "chunk_count": 27,
  "corpus_rows": 2652248,
  "family_counts": {"court": 2476315, "law": 175933},
  "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "torch_version": "2.10.0+cu128",
  "cuda_version": "12.8"
}
```

### Stage 10 — Chunk sanity check
All 27 chunks valid: shapes `(100000, 4096)` (last `(52248, 4096)`), dtype `float16`, no NaNs, L2 `norm_mean` ≈ 1.0012–1.0014 (norm_std 0.0017–0.0018). Total rows across chunks: `2,652,248` (matches corpus).

### Stage 11 — Smoke retrieval against chunk 0
```
Q: When can Swiss courts extend pretrial detention based on collusion risk?
  1. [court] 1B_227/2017 E. C                        score=0.780
  2. [court] 1B_541/2020 E. 3.2                      score=0.771
  3. [court] 1B_246/2007 20.11.2007 E. 2             score=0.758
  4. [court] 1P.482/2006 17.08.2006 E. B             score=0.746
  5. [court] 1B_228/2008 02.09.2008 E. A             score=0.745

Q: What principles bind persons performing public tasks under Swiss public law?
  1. [court] 2C_1023/2021 E. 2.2.1                   score=0.675
  2. [court] BGE 144 II 281 E. 4.1                   score=0.663
  3. [court] 2C_94/2018 E. 4.1                       score=0.662
  4. [court] BGE 137 II 409 E. 7.3.1                 score=0.641
  5. [court] 2C_492/2022 E. 7                        score=0.640

Q: Dublin transfer detention proportionality test
  1. [court] 2C_199/2018 E. 4.2                      score=0.735
  2. [court] 2C_142/2023 E. 3.3.5                    score=0.672
  3. [court] BGE 150 II 57 E. 3.3.2                  score=0.663
  4. [court] 2C_207/2016 E. 2.3                      score=0.660
  5. [court] BGE 139 II 121 E. 6.5.1                 score=0.631
```

### Errors
- Stage 0 (`flash-attn` 2.x install): `ERROR: Operation cancelled by user` while `Building wheel for flash-attn (pyproject.toml)`. This is the intentional cancel — FA2 build was abandoned in favor of `flash-attn-4` (see Stage 1).
- Cell 7 source contains an extraneous `except Exception: pass` block after the FA2 probe (orphaned `except` without matching `try`); did not raise at runtime in the captured output.

## Summary

The notebook embeds a unified Swiss legal corpus of 2,652,248 documents (2,476,315 court + 175,933 law) with `Qwen/Qwen3-Embedding-8B` at bf16, 4096-dim, max sequence 768, batch 256, into 27 fp16 `.npy` chunks of 100k rows on Google Drive (last chunk 52,248). Runs on an NVIDIA RTX PRO 6000 Blackwell (sm_120, 102 GB VRAM, torch 2.10.0+cu128) and uses a hardware-aware attention backend chain (FA4 → FA2 → SDPA on Blackwell); FA4 was installed but not yet wired through HF transformers, so SDPA with flash/cuDNN/mem-efficient kernels was selected. Documents are encoded without an instruction prefix (only queries use the `Instruct: ... \nQuery: ...` template per Qwen3-Embedding conventions); embeddings are L2-normalized then cast to fp16, and a per-doc manifest plus a JSON summary are written alongside the chunks. On this run all 27 chunks were already cached, so no encode timings were produced; sanity checks confirmed correct shapes, dtype, no NaNs, and norms ≈ 1.0013. A smoke retrieval test against chunk 0 produced sensible top-5 hits for three English Swiss-law queries (top scores 0.780 / 0.675 / 0.735).
