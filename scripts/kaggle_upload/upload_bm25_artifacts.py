#!/usr/bin/env python
"""Upload Swiss Law BM25 artifacts (+ val all_targets) to Kaggle as a new dataset.

Usage:
    python scripts/kaggle_upload/upload_bm25_artifacts.py

Files uploaded:
  - bm25_v2_index.pkl   (~104 MB)
  - bm25_v2_ids.pkl     (~3.7 MB)
  - corpus.parquet      (~131 MB)
  - all_targets.json    (~22 KB, val_001..val_010 only)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

KAGGLE_USERNAME = "samiulislam180041221"
DATASET_SLUG    = "swiss-law-bm25-artifacts"
DATASET_TITLE   = "Swiss Law BM25 Artifacts (Omnilex)"
PUBLIC          = True

ARTIFACT_DIR = ROOT / "drive_sync" / "omnilex_competition" / "retrieval_other_artifacts"
TARGETS_SRC  = ROOT / "drive_sync" / "swiss_law" / "v7_pool_recall_089_and_per_query" / "snapshot" / "all_targets.json"

FILES_TO_UPLOAD = [
    ARTIFACT_DIR / "bm25_v2_index.pkl",
    ARTIFACT_DIR / "bm25_v2_ids.pkl",
    ARTIFACT_DIR / "corpus.parquet",
    TARGETS_SRC,
]


def _require_tqdm() -> None:
    try:
        import tqdm  # noqa: F401
    except ImportError:
        import subprocess
        print("Installing tqdm…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])


def _copy_with_progress(src: Path, dest: Path) -> None:
    from tqdm import tqdm
    size = src.stat().st_size
    with tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024,
              desc=f"  copy {src.name}") as pbar:
        with src.open("rb") as fin, dest.open("wb") as fout:
            while True:
                chunk = fin.read(4 * 1024 * 1024)
                if not chunk:
                    break
                fout.write(chunk)
                pbar.update(len(chunk))


def _check_files() -> None:
    missing = [f for f in FILES_TO_UPLOAD if not f.exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(str(f) for f in missing))
    for f in FILES_TO_UPLOAD:
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:30s}  {size_mb:>8.2f} MB   {f}")


def main() -> int:
    if "KAGGLE_API_TOKEN" not in os.environ:
        raise RuntimeError(
            "KAGGLE_API_TOKEN env var is required. "
            "Get a token at https://www.kaggle.com/settings/account and run: "
            "`export KAGGLE_API_TOKEN=<token>` (or `$env:KAGGLE_API_TOKEN='<token>'` on PowerShell)."
        )

    _require_tqdm()

    print("=" * 70)
    print("Kaggle Dataset Upload — Swiss Law BM25 Artifacts")
    print(f"  username : {KAGGLE_USERNAME}")
    print(f"  slug     : {DATASET_SLUG}")
    print(f"  title    : {DATASET_TITLE}")
    print(f"  public   : {PUBLIC}")
    print("=" * 70)

    print("\nChecking input files…")
    _check_files()

    try:
        from kaggle import api  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("kaggle package not found. Run: pip install kaggle") from exc

    api.authenticate()
    print("Authenticated OK.\n")

    staging = Path(tempfile.mkdtemp(prefix="kaggle_bm25_"))
    print(f"Staging dir: {staging}\n")

    try:
        metadata = {
            "title": DATASET_TITLE,
            "id": f"{KAGGLE_USERNAME}/{DATASET_SLUG}",
            "licenses": [{"name": "CC0-1.0"}],
        }
        (staging / "dataset-metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print("Copying files to staging directory…")
        for src in FILES_TO_UPLOAD:
            _copy_with_progress(src, staging / src.name)
        print()

        print(f"Uploading to Kaggle as {'public' if PUBLIC else 'private'} dataset…\n")
        api.dataset_create_new(
            folder=str(staging),
            public=PUBLIC,
            quiet=False,
            convert_to_csv=False,
            dir_mode="skip",
        )

        url = f"https://www.kaggle.com/datasets/{KAGGLE_USERNAME}/{DATASET_SLUG}"
        print(f"\nDataset created:  {url}")

    finally:
        shutil.rmtree(staging, ignore_errors=True)
        print("Staging dir cleaned up.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
