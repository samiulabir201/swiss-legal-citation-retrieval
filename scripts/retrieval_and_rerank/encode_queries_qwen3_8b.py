#!/usr/bin/env python
"""Encode val/test/train query strings with Qwen3-Embedding-8B.

Standalone (no Colab dependency). Replaces the equivalent step in
`notebooks/encode_queries_qwen3_8b_colab.ipynb`.

Why this exists: the notebook encoded queries with `MAX_SEQ_LEN=256`, which
empirically truncates 10/10 val queries (mean -98 tok, worst -197 tok). At
`MAX_SEQ_LEN=2048` no val query truncates. The doc side was embedded with
`max_seq_length=768`, but the model is the same so dot-products remain valid;
query and doc inputs do not need matching seq lengths.

Output (one .npy + one .parquet per split):
    qwen3_8b_query_<split>.npy        (n, 4096) fp32 L2-normalized
    qwen3_8b_query_<split>_ids.parquet  query_id ordering
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

QWEN_INSTRUCT = (
    "Instruct: Given an English-language legal question or scenario about "
    "Swiss federal law, retrieve the Swiss statute articles or federal court "
    "decision considerations that are most directly relevant to answering it."
    "\nQuery: "
)


def encode_split(
    *,
    split: str,
    csv_path: Path,
    out_dir: Path,
    model,
    batch_size: int,
) -> int:
    if not csv_path.exists():
        print(f"[skip] {csv_path} not found", file=sys.stderr)
        return 0
    df = pd.read_csv(csv_path)
    if "query" not in df.columns:
        raise SystemExit(f"{csv_path}: expected a 'query' column, got {list(df.columns)}")
    if "query_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "query_id"})

    queries = [QWEN_INSTRUCT + str(q) for q in df["query"].tolist()]
    print(f"[{split}] n={len(queries)}  longest_chars={max(len(q) for q in queries)}", flush=True)

    t0 = time.time()
    with torch.inference_mode():
        emb = model.encode(
            queries,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
    print(f"[{split}] shape={emb.shape}  norm_mean={np.linalg.norm(emb,axis=1).mean():.4f}  in {time.time()-t0:.1f}s", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    np_path = out_dir / f"qwen3_8b_query_{split}.npy"
    np.save(np_path, emb)
    ids = df[["query_id"]].copy()
    ids["row_index"] = np.arange(len(ids), dtype=np.int64)
    ids["query"] = df["query"].astype(str)
    ids_path = out_dir / f"qwen3_8b_query_{split}_ids.parquet"
    ids.to_parquet(ids_path, index=False)
    print(f"[{split}] saved {np_path}  ({emb.nbytes/1e6:.1f} MB)")
    print(f"[{split}] saved {ids_path}")
    return len(queries)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/embeddings"))
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--max-seq-len", type=int, default=2048,
                    help="Was 256 in the original notebook; 256 truncates 10/10 val queries.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--splits", nargs="+", default=["val"],
                    help="One or more of {val, test, train}. Default just val "
                         "for the truncation-fix sanity check.")
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available; this model needs ~16 GB VRAM in bf16.",
              file=sys.stderr)
        return 2

    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  VRAM: {p.total_memory/1e9:.1f} GB")
    print(f"model={args.model}  max_seq_len={args.max_seq_len}  batch={args.batch_size}  "
          f"attn={args.attn}  dtype={args.dtype}", flush=True)

    from sentence_transformers import SentenceTransformer
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    model = SentenceTransformer(
        args.model, device="cuda",
        model_kwargs={"torch_dtype": dtype, "attn_implementation": args.attn},
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = args.max_seq_len
    model.eval()
    print(f"loaded; allocated {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

    total = 0
    for split in args.splits:
        total += encode_split(
            split=split,
            csv_path=args.data_dir / f"{split}.csv",
            out_dir=args.out_dir,
            model=model,
            batch_size=args.batch_size,
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\n[done] encoded {total} queries across {len(args.splits)} split(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
