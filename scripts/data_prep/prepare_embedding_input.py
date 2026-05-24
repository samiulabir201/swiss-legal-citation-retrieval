#!/usr/bin/env python
"""Extract embedding input from the unified retrieval SQLite.

Reads the per-document `vector_text` (a labelled, English-leaning composition
already produced by `build_unified_retrieval_corpus.py`) and writes a compact
parquet with just the columns the embedding job needs:

    doc_id           (str, primary key)
    family           ('court' | 'law')
    citation         (str, for sanity / debugging only)
    vector_text      (str, the input to encode)
    char_len         (int, used for length-sorted batching on the GPU)

Uploaded to Drive, this is what the Colab embedding notebook consumes.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"

DEFAULT_SQLITE = ART_DIR / "unified_retrieval.sqlite"
DEFAULT_OUT = ART_DIR / "unified_embedding_input.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap rows for smoke runs. 0 = full corpus.")
    ap.add_argument("--batch-size", type=int, default=200_000,
                    help="Rows per parquet row-group.")
    ap.add_argument("--compression", default="snappy",
                    choices=["snappy", "zstd", "gzip", "none"])
    ap.add_argument("--family", default="all",
                    choices=["all", "court", "law"],
                    help="Restrict to one family for partial runs.")
    args = ap.parse_args()

    if not args.sqlite.exists():
        print(f"ERROR: sqlite not found: {args.sqlite}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    where = "" if args.family == "all" else f" WHERE family = '{args.family}'"
    limit_sql = "" if args.limit <= 0 else f" LIMIT {int(args.limit)}"
    sql = (
        "SELECT doc_id, family, citation, vector_text, LENGTH(vector_text) AS char_len "
        f"FROM documents{where} ORDER BY family, doc_id{limit_sql}"
    )

    schema = pa.schema([
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("family", pa.string(), nullable=False),
        pa.field("citation", pa.string(), nullable=False),
        pa.field("vector_text", pa.string(), nullable=False),
        pa.field("char_len", pa.int32(), nullable=False),
    ])
    compression = None if args.compression == "none" else args.compression
    writer = pq.ParquetWriter(args.output, schema, compression=compression)

    con = sqlite3.connect(args.sqlite)
    con.execute("PRAGMA temp_store=MEMORY")
    cur = con.execute(sql)

    total = 0
    by_family: dict[str, int] = {}
    char_total = 0
    t0 = time.time()
    print(f"[query] {sql}", flush=True)

    while True:
        batch = cur.fetchmany(args.batch_size)
        if not batch:
            break
        doc_ids, families, citations, texts, char_lens = zip(*batch)
        char_total += sum(char_lens)
        for f in families:
            by_family[f] = by_family.get(f, 0) + 1
        table = pa.Table.from_arrays(
            [
                pa.array(doc_ids, type=pa.string()),
                pa.array(families, type=pa.string()),
                pa.array(citations, type=pa.string()),
                pa.array(texts, type=pa.string()),
                pa.array(char_lens, type=pa.int32()),
            ],
            schema=schema,
        )
        writer.write_table(table)
        total += len(batch)
        rate = total / max(time.time() - t0, 1e-9)
        print(f"  rows={total:>10,}  ({rate:>6.0f} rows/s)", flush=True)

    writer.close()
    con.close()
    elapsed = time.time() - t0

    out_size_mb = args.output.stat().st_size / 1e6
    print()
    print(f"[done] rows={total:,}  by_family={by_family}", flush=True)
    print(f"[done] avg_chars={char_total/max(total,1):.0f}  total_chars={char_total:,}", flush=True)
    print(f"[done] output={args.output} ({out_size_mb:.1f} MB)", flush=True)
    print(f"[done] elapsed={elapsed/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
