#!/usr/bin/env python
"""Extract slim retrieval metadata from `unified_retrieval.sqlite` to a single
ZIP for upload to Colab.

Contents (~400-500 MB total):

    docs_meta.parquet            documents columns minus heavy text/JSON fields
    statute_links.parquet        statute_links table (raw, aliasing applied at query-time)
    case_links.parquet           case_links table
    adjacent_law_links.parquet   adjacent_law_links table
    law_code_aliases.json        Swiss law-code <-> SC-number alias map

The notebook applies aliasing at query time so the parquets stay small. We
exclude `vector_text`, `bm25_text`, `source_text`, `rag_json`, etc. to drop
~95% of the SQLite size.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"

DEFAULT_SQLITE = ART_DIR / "unified_retrieval.sqlite"
DEFAULT_OUT = ART_DIR / "retrieval_metadata.zip"

# Each German abbreviation maps to a list of equivalent forms used across the
# corpus and queries. The notebook applies these at retrieval time so we don't
# have to materialize an expanded statute_links table.
LAW_CODE_ALIASES: dict[str, list[str]] = {
    "StPO":  ["312.0",   "CPP"],
    "StGB":  ["311.0",   "CP", "CPS"],
    "ZGB":   ["210",     "CC", "CCS"],
    "ZPO":   ["272",     "CPC"],
    "OR":    ["220",     "CO"],
    "BV":    ["101",     "Cst"],
    "BGG":   ["173.110", "LTF"],
    "AsylG": ["142.31",  "LAsi"],
    "AuG":   ["142.20",  "LEtr", "LEI", "AIG"],
    "AIG":   ["142.20",  "LEtr", "LEI", "AuG"],
    "AVIG":  ["837.0",   "LACI"],
    "AHVG":  ["831.10",  "LAVS"],
    "IVG":   ["831.20",  "LAI"],
    "BVG":   ["831.40",  "LPP"],
    "KVG":   ["832.10",  "LAMal"],
    "UVG":   ["832.20",  "LAA"],
    "DBG":   ["642.11",  "LIFD"],
    "MWSTG": ["641.20",  "LIVA"],
    "SVG":   ["741.01",  "LCR"],
    "SchKG": ["281.1",   "LP"],
    "BetmG": ["812.121", "LStup"],
    "BankG": ["952.0",   "LB"],
    "EMRK":  ["0.101",   "CEDH"],
    "PatG":  ["232.14",  "LBI"],
    "KG":    ["251",     "LCart"],
    "IPRG":  ["291",     "LDIP"],
    "DSG":   ["235.1",   "LPD"],
    "NHG":   ["451",     "LPN"],
    "RPG":   ["700",     "LAT"],
    "BGFA":  ["935.61",  "LLCA"],
    "FINMAG":["956.1",   "LFINMA"],
    "AMLA":  ["955.0",   "LBA"],
    "URG":   ["231.1",   "LDA"],
    "UWG":   ["241",     "LCD"],
    "MSchG": ["232.11",  "LPM"],
    "VwVG":  ["172.021", "PA"],
    "StBOG": ["173.71",  "LOAP"],
    "ATSG":  ["830.1",   "LPGA"],
    "VRG":   ["173.110.131"],
    "OG":    ["173.110.131"],
    "OJ":    ["173.110"],
}

# Heavy text and serialized-JSON fields to drop.
DROP_COLS = {
    "vector_text", "bm25_text", "source_text",
    "rag_json", "filters_json", "expansion_json", "quality_flags_json",
}


def extract(sqlite_path: Path, output_zip: Path) -> None:
    if not sqlite_path.exists():
        raise FileNotFoundError(sqlite_path)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    print(f"[extract] reading from {sqlite_path}", flush=True)
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        # Discover columns and drop the heavy ones
        cols = pd.read_sql_query("PRAGMA table_info(documents)", con).name.tolist()
        keep = [c for c in cols if c not in DROP_COLS]
        col_list = ", ".join(keep)

        # documents metadata
        t0 = time.time()
        docs = pd.read_sql_query(f"SELECT {col_list} FROM documents", con)
        out = tmp / "docs_meta.parquet"
        docs.to_parquet(out, compression="snappy")
        print(f"  docs_meta.parquet         {len(docs):>10,} rows  "
              f"{out.stat().st_size/1e6:>7.1f} MB  ({time.time()-t0:.1f}s)", flush=True)

        # statute_links
        t0 = time.time()
        slinks = pd.read_sql_query("SELECT * FROM statute_links", con)
        out = tmp / "statute_links.parquet"
        slinks.to_parquet(out, compression="snappy")
        print(f"  statute_links.parquet     {len(slinks):>10,} rows  "
              f"{out.stat().st_size/1e6:>7.1f} MB  ({time.time()-t0:.1f}s)", flush=True)

        # case_links
        t0 = time.time()
        clinks = pd.read_sql_query("SELECT * FROM case_links", con)
        out = tmp / "case_links.parquet"
        clinks.to_parquet(out, compression="snappy")
        print(f"  case_links.parquet        {len(clinks):>10,} rows  "
              f"{out.stat().st_size/1e6:>7.1f} MB  ({time.time()-t0:.1f}s)", flush=True)

        # adjacent_law_links
        t0 = time.time()
        alinks = pd.read_sql_query("SELECT * FROM adjacent_law_links", con)
        out = tmp / "adjacent_law_links.parquet"
        alinks.to_parquet(out, compression="snappy")
        print(f"  adjacent_law_links.parquet {len(alinks):>10,} rows  "
              f"{out.stat().st_size/1e6:>7.1f} MB  ({time.time()-t0:.1f}s)", flush=True)

        # aliases
        out = tmp / "law_code_aliases.json"
        out.write_text(json.dumps(LAW_CODE_ALIASES, indent=2), encoding="utf-8")

        con.close()

        # zip everything (deflated to keep the upload small)
        t0 = time.time()
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for name in ("docs_meta.parquet", "statute_links.parquet",
                         "case_links.parquet", "adjacent_law_links.parquet",
                         "law_code_aliases.json"):
                zf.write(tmp / name, arcname=name)
        size_mb = output_zip.stat().st_size / 1e6
        print(f"\n[done] {output_zip} ({size_mb:.1f} MB)  zip-time {time.time()-t0:.1f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    try:
        extract(args.sqlite, args.output)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
