#!/usr/bin/env python
"""Upload AWQ wheels (autoawq, autoawq-kernels, rank_bm25) to Kaggle for offline install.

Usage:
    python scripts/kaggle_upload/upload_awq_wheels.py

Files uploaded:
  - autoawq-0.2.6-cp310-cp310-manylinux2014_x86_64.whl
  - autoawq_kernels-0.0.9-cp310-cp310-manylinux2014_x86_64.whl
  - rank_bm25-0.2.2-py3-none-any.whl
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
DATASET_SLUG    = "swiss-law-awq-wheels"
DATASET_TITLE   = "Swiss Law AWQ Wheels"
PUBLIC          = False  # private — only for the Swiss-law notebook

SOURCE_DIR = ROOT / "kaggle_wheels_gptqmodel"

FILES_TO_UPLOAD = sorted(SOURCE_DIR.glob("*.whl"))


def _require_tqdm() -> None:
    try:
        import tqdm  # noqa: F401
    except ImportError:
        import subprocess
        print("Installing tqdm ...")
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


def main() -> int:
    if "KAGGLE_API_TOKEN" not in os.environ:
        raise RuntimeError(
            "KAGGLE_API_TOKEN env var is required. "
            "Get a token at https://www.kaggle.com/settings/account and run: "
            "`export KAGGLE_API_TOKEN=<token>` (or `$env:KAGGLE_API_TOKEN='<token>'` on PowerShell)."
        )

    _require_tqdm()

    print("=" * 70)
    print("Kaggle Dataset Upload — AWQ Wheels")
    print(f"  username : {KAGGLE_USERNAME}")
    print(f"  slug     : {DATASET_SLUG}")
    print(f"  title    : {DATASET_TITLE}")
    print(f"  public   : {PUBLIC}")
    print("=" * 70)

    if not FILES_TO_UPLOAD:
        raise FileNotFoundError(f"No .whl files in {SOURCE_DIR}")

    print(f"\nFiles to upload ({len(FILES_TO_UPLOAD)}):")
    for f in FILES_TO_UPLOAD:
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:60s}  {size_mb:>7.2f} MB")

    try:
        from kaggle import api  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("kaggle package not found. Run: pip install kaggle") from exc

    api.authenticate()
    print("\nAuthenticated OK.\n")

    staging = Path(tempfile.mkdtemp(prefix="kaggle_awq_"))
    print(f"Staging dir: {staging}\n")

    try:
        metadata = {
            "title": DATASET_TITLE,
            "id": f"{KAGGLE_USERNAME}/{DATASET_SLUG}",
            "licenses": [{"name": "MIT"}],
        }
        (staging / "dataset-metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        print("Copying wheels to staging directory ...")
        for src in FILES_TO_UPLOAD:
            _copy_with_progress(src, staging / src.name)
        print()

        print(f"Uploading to Kaggle as {'public' if PUBLIC else 'private'} dataset ...\n")
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
