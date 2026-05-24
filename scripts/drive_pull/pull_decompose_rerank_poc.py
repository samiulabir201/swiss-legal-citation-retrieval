"""pull_decompose_rerank_poc.py — pull val_004 phase outputs from Drive.

Pulls the artifacts produced by `decompose_rerank_poc_val004.ipynb` from:
    Drive: My Drive/swiss_law/research/decompose_rerank_poc/

into:
    Local: e:/swiss_citation_extraction/research/decompose_rerank_poc/run_outputs_val_004/

Files pulled (all that exist — missing ones are skipped, not errors):
    val_004_bundle.json                 - Phase 2 output: sub-issues + targets + DE
    val_004_candidates.parquet          - Phase 3 output: union of sub-issue funnel pools
    val_004_reranked.parquet            - Phase 4 output: top-K per sub-issue after reranker
    val_004_reranked.partial.parquet    - Phase 4 in-progress checkpoint (if Phase 4 didn't finish)
    val_004_f1.json                     - Phase 5 final summary

Uses the same token.json + credentials.json as drive_pull_folders.py.

USAGE
-----
    python scripts/drive_pull/pull_decompose_rerank_poc.py
    python scripts/drive_pull/pull_decompose_rerank_poc.py --qid val_004
"""
from __future__ import annotations
import argparse
import io
import sys
from pathlib import Path

try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
except ImportError:
    sys.exit(
        "Missing libs. Run: pip install google-api-python-client "
        "google-auth-httplib2 google-auth-oauthlib"
    )

SCOPES        = ["https://www.googleapis.com/auth/drive.readonly"]
HERE          = Path(__file__).resolve().parent
TOKEN_PATH    = HERE / "token.json"
PROJECT_ROOT  = Path(r"e:\swiss_citation_extraction")

FOLDER_MIME = "application/vnd.google-apps.folder"


def get_service():
    if not TOKEN_PATH.exists():
        sys.exit("token.json not found. Run drive_pull.py first to authenticate.")
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def find_folder(svc, path_parts: list[str]) -> str | None:
    """Walk My Drive/<part1>/<part2>/... and return the leaf folder id."""
    parent = "root"
    for part in path_parts:
        q = (
            f"'{parent}' in parents and "
            f"mimeType = '{FOLDER_MIME}' and "
            f"name = '{part}' and trashed = false"
        )
        resp = svc.files().list(
            q=q,
            pageSize=10,
            fields="files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files = resp.get("files", [])
        if not files:
            print(f"  [missing] folder '{part}' under parent {parent}")
            return None
        parent = files[0]["id"]
        print(f"  resolved: {part} -> {parent}")
    return parent


def list_files_in_folder(svc, folder_id: str) -> list[dict]:
    out: list[dict] = []
    page_token = None
    while True:
        resp = svc.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            pageSize=500,
            fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        out.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out


def download_one(svc, file_id: str, dest: Path, chunk_mb: int = 16) -> None:
    request = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with io.FileIO(str(dest), "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=chunk_mb * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def human_mb(n: int) -> str:
    return f"{n / 1024 ** 2:.2f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default="val_004")
    args = ap.parse_args()
    qid = args.qid

    expected_names = {
        f"{qid}_bundle.json":                "Phase 2 — sub-issues + targets",
        f"{qid}_candidates.parquet":         "Phase 3 — funnel candidates union",
        f"{qid}_reranked.parquet":           "Phase 4 — top-K per sub-issue after reranker",
        f"{qid}_reranked.partial.parquet":   "Phase 4 — partial checkpoint",
        f"{qid}_f1.json":                    "Phase 5 — final F1 summary",
    }

    dest_dir = PROJECT_ROOT / "research" / "decompose_rerank_poc" / f"run_outputs_{qid}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    svc = get_service()

    print("Resolving folder: My Drive/swiss_law/research/decompose_rerank_poc/ ...")
    leaf_id = find_folder(svc, ["swiss_law", "research", "decompose_rerank_poc"])
    if leaf_id is None:
        sys.exit("Could not resolve the target folder on Drive.")

    print("\nListing files in folder...")
    files = list_files_in_folder(svc, leaf_id)
    by_name = {f["name"]: f for f in files}
    print(f"  {len(files)} files on Drive in that folder")

    pulled, missing = [], []
    for fname, label in expected_names.items():
        f = by_name.get(fname)
        if f is None:
            missing.append(fname)
            continue
        dest = dest_dir / fname
        size_b = int(f.get("size", 0))
        if dest.exists() and dest.stat().st_size == size_b and size_b > 0:
            print(f"  [skip cached] {fname}  {human_mb(size_b)}")
            pulled.append((fname, dest, size_b, label))
            continue
        print(f"  [download   ] {fname}  {human_mb(size_b)}  ({label})")
        try:
            download_one(svc, f["id"], dest)
            pulled.append((fname, dest, size_b, label))
        except Exception as e:
            print(f"    FAIL: {e}")

    print()
    print("=" * 70)
    print(f"Pulled {len(pulled)} files into: {dest_dir}")
    print("=" * 70)
    for fname, dest, size_b, label in pulled:
        print(f"  {fname:40s}  {human_mb(size_b):>10s}  {label}")
    if missing:
        print()
        print("Missing (not on Drive — phase did not run or saved elsewhere):")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
