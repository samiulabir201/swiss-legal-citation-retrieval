#!/usr/bin/env python
"""Upload Swiss court authority cards to Kaggle as a new dataset.

Usage:
    python scripts/upload_to_kaggle.py

Requires: kaggle>=2.1.0 (pip install kaggle), tqdm

Files uploaded:
  - artifacts/court_authority_cards_v4_target_cards.jsonl  (~981 MB)
  - research/enrich_cards_qwen3_8b_kaggle.ipynb            (~44 KB)

The kaggle package's upload_complete() already wraps GCS PUT with tqdm,
so progress bars appear automatically when quiet=False.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── Configuration ────────────────────────────────────────────────────────────

KAGGLE_USERNAME = "samiulislam180041221"
DATASET_SLUG    = "swiss-court-authority-cards-rag-targets"
DATASET_TITLE   = "Swiss Court Authority Cards - RAG Targets"
PUBLIC          = True   # Set False to keep the dataset private

FILES_TO_UPLOAD = [
    ROOT / "artifacts" / "court_authority_cards_v4_target_cards.jsonl",
    ROOT / "research"  / "enrich_cards_qwen3_8b_kaggle.ipynb",
]

# ── Helpers ──────────────────────────────────────────────────────────────────


def _require_tqdm() -> None:
    try:
        import tqdm  # noqa: F401
    except ImportError:
        import subprocess
        print("Installing tqdm…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])


def _copy_with_progress(src: Path, dest: Path) -> None:
    try:
        from tqdm import tqdm
    except ImportError:
        shutil.copy2(src, dest)
        return

    size = src.stat().st_size
    with tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024,
              desc=f"  copy {src.name}") as pbar:
        with src.open("rb") as fin, dest.open("wb") as fout:
            while True:
                chunk = fin.read(4 * 1024 * 1024)  # 4 MB chunks
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
        print(f"  {f.name:60s}  {size_mb:>8.1f} MB")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    if "KAGGLE_API_TOKEN" not in os.environ:
        raise RuntimeError(
            "KAGGLE_API_TOKEN env var is required. "
            "Get a token at https://www.kaggle.com/settings/account and run: "
            "`export KAGGLE_API_TOKEN=<token>` (or `$env:KAGGLE_API_TOKEN='<token>'` on PowerShell)."
        )

    _require_tqdm()

    print("=" * 70)
    print("Kaggle Dataset Upload")
    print(f"  username : {KAGGLE_USERNAME}")
    print(f"  slug     : {DATASET_SLUG}")
    print(f"  title    : {DATASET_TITLE}")
    print(f"  public   : {PUBLIC}")
    print("=" * 70)

    print("\nChecking input files…")
    _check_files()

    # Import kaggle *after* env var is set
    try:
        from kaggle import api  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SystemExit("kaggle package not found. Run: pip install kaggle") from exc

    api.authenticate()
    print("Authenticated OK.\n")

    # ── Build staging directory ───────────────────────────────────────────────
    staging = Path(tempfile.mkdtemp(prefix="kaggle_upload_"))
    print(f"Staging dir: {staging}\n")

    try:
        # Write dataset-metadata.json
        metadata = {
            "title": DATASET_TITLE,
            "id": f"{KAGGLE_USERNAME}/{DATASET_SLUG}",
            "licenses": [{"name": "CC0-1.0"}],
        }
        (staging / "dataset-metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Copy files into staging with local progress bars
        print("Copying files to staging directory…")
        for src in FILES_TO_UPLOAD:
            _copy_with_progress(src, staging / src.name)
        print()

        # ── Upload ────────────────────────────────────────────────────────────
        print(f"Uploading to Kaggle as {'public' if PUBLIC else 'private'} dataset…")
        print("(tqdm progress bars will appear per file during GCS upload)\n")

        api.dataset_create_new(
            folder=str(staging),
            public=PUBLIC,
            quiet=False,        # ← tqdm progress bars from kaggle's upload_complete()
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
