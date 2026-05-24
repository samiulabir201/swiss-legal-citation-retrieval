"""pull_unified_retrieval.py — pull the ~24 GB unified_retrieval.sqlite from Drive.

Uses the same OAuth setup as `drive_pull_folders.py` (token.json + credentials.json
in this directory). Downloads to:

    e:\\swiss_citation_extraction\\artifacts\\unified_retrieval.sqlite

Designed for a single big file:
  - 64 MB chunks (vs the 8 MB default in folder-walker)
  - Skip-if-already-complete (size check vs Drive)
  - Resumable manifest entry (same `_downloaded_ids.json` as the other pullers)
  - Progress every chunk so you can see throughput on a 24 GB transfer

USAGE
-----
    python pull_unified_retrieval.py                 # default behaviour
    python pull_unified_retrieval.py --dry-run       # find file + print size, don't download
    python pull_unified_retrieval.py --file-id <id>  # skip search, pull this Drive id
    python pull_unified_retrieval.py --dest <path>   # override local dest
    python pull_unified_retrieval.py --chunk-mb 128  # larger chunks if your network is fast
"""
from __future__ import annotations
import argparse
import io
import json
import sys
import time
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
MANIFEST_PATH = HERE / "_downloaded_ids.json"

PROJECT_ROOT  = Path(r"e:\swiss_citation_extraction")
DEFAULT_DEST  = PROJECT_ROOT / "artifacts" / "unified_retrieval.sqlite"

# Search tokens to find the file on Drive
NAME_TOKENS = [
    "unified_retrieval.sqlite",
    "unified_retrieval",
]


# ── auth ──────────────────────────────────────────────────────────────────
def get_service():
    if not TOKEN_PATH.exists():
        sys.exit("token.json not found. Run drive_pull.py first to authenticate.")
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ── search ────────────────────────────────────────────────────────────────
def search_for_file(svc, tokens):
    by_id = {}
    for tok in tokens:
        # Exact-name match first
        for q in (f"name = '{tok}'", f"name contains '{tok}'"):
            page_token = None
            while True:
                resp = svc.files().list(
                    q=f"{q} and trashed = false",
                    pageSize=200,
                    fields="nextPageToken, files(id, name, size, modifiedTime, parents, mimeType)",
                    pageToken=page_token,
                    supportsAllDrives=True, includeItemsFromAllDrives=True,
                ).execute()
                for f in resp.get("files", []):
                    by_id[f["id"]] = f
                page_token = resp.get("nextPageToken")
                if not page_token:
                    break
    items = list(by_id.values())
    items.sort(key=lambda f: f.get("modifiedTime", ""), reverse=True)
    return items


def make_path_resolver(svc):
    cache = {}
    def resolve(file_id):
        if file_id in cache:
            return cache[file_id]
        try:
            f = svc.files().get(
                fileId=file_id, fields="id,name,parents",
                supportsAllDrives=True,
            ).execute()
        except Exception:
            cache[file_id] = "?"; return "?"
        parents = f.get("parents") or []
        full = f["name"] if not parents else f"{resolve(parents[0])}/{f['name']}"
        cache[file_id] = full
        return full
    return resolve


# ── manifest ──────────────────────────────────────────────────────────────
def load_manifest():
    if MANIFEST_PATH.exists():
        try: return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}


def save_manifest(m):
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(MANIFEST_PATH)


def _fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:6.1f} {unit}"
        n /= 1024
    return f"{n:6.1f} PB"


# ── download ──────────────────────────────────────────────────────────────
def download_with_progress(svc, file_id, dest: Path, expected_size: int, chunk_mb: int):
    chunk_bytes = chunk_mb * 1024 * 1024
    request = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    last_print = started
    with io.FileIO(str(dest), "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=chunk_bytes)
        done = False
        chunk_idx = 0
        while not done:
            status, done = downloader.next_chunk()
            chunk_idx += 1
            now = time.time()
            if now - last_print >= 1.0 or done:
                resp_size = status.resumable_progress if status else fh.tell()
                pct = (resp_size / expected_size * 100) if expected_size else 0
                mb = resp_size / 1024**2
                rate = mb / max(0.01, now - started)
                eta = (expected_size - resp_size) / (1024**2 * max(0.01, rate)) if rate else 0
                print(f"  {pct:>6.2f}%   {mb:>9,.0f} MB   {rate:>6.1f} MB/s   "
                      f"ETA {eta/60:>5.1f} min   chunk={chunk_idx}", flush=True)
                last_print = now
    return time.time() - started


# ── main ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--file-id", help="Drive file ID. Skips search.")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Local destination path")
    ap.add_argument("--chunk-mb", type=int, default=64, help="Chunk size in MB (8-128)")
    ap.add_argument("--dry-run", action="store_true", help="Find + print only, don't download")
    args = ap.parse_args()

    svc = get_service()
    resolve = make_path_resolver(svc)

    # ── locate file ──
    if args.file_id:
        try:
            file_rec = svc.files().get(
                fileId=args.file_id,
                fields="id,name,size,modifiedTime,parents,mimeType",
                supportsAllDrives=True,
            ).execute()
            file_rec["drive_path"] = resolve(file_rec["id"])
        except Exception as e:
            sys.exit(f"Could not fetch file ID {args.file_id}: {e}")
    else:
        print(f"Searching Drive for {NAME_TOKENS} ...")
        hits = search_for_file(svc, NAME_TOKENS)
        if not hits:
            sys.exit("Nothing found. Pass --file-id <id> with the Drive web URL's ID fragment.")
        for i, h in enumerate(hits[:10]):
            sz = int(h.get("size", 0))
            print(f"  [{i}] {h.get('modifiedTime', '?')}  {_fmt_bytes(sz)}  "
                  f"mime={h.get('mimeType', '?'):<35}  {resolve(h['id'])}")
        # Pick by size — the real one will be ~24 GB
        big_hits = [h for h in hits if int(h.get("size", 0)) > 1024**3]   # > 1 GB
        if not big_hits:
            sys.exit("No candidate > 1 GB. Pass --file-id explicitly.")
        file_rec = big_hits[0]
        file_rec["drive_path"] = resolve(file_rec["id"])
        print(f"\nChose: {file_rec['drive_path']}  ({_fmt_bytes(int(file_rec['size']))})")

    expected_size = int(file_rec.get("size", 0))
    if expected_size == 0:
        sys.exit("Drive returned size=0; cannot verify download. Aborting.")

    # ── pre-flight ──
    print(f"\nTarget local path: {args.dest}")
    print(f"Drive size:        {_fmt_bytes(expected_size)} ({expected_size:,} bytes)")
    print(f"Chunk size:        {args.chunk_mb} MB")

    if args.dest.exists():
        cur = args.dest.stat().st_size
        if cur == expected_size:
            print(f"\nLocal file already complete ({_fmt_bytes(cur)}). Skipping.")
            manifest = load_manifest()
            manifest[file_rec["id"]] = {"path": str(args.dest), "size": expected_size}
            save_manifest(manifest)
            return 0
        else:
            print(f"\nLocal file exists but size differs: local={_fmt_bytes(cur)} vs drive={_fmt_bytes(expected_size)}")
            print("Will overwrite (download restarts from byte 0 — Drive API doesn't support range-resume).")

    if args.dry_run:
        print("\n--dry-run set. Not downloading.")
        return 0

    # ── go ──
    print("\nDownloading ...")
    duration = download_with_progress(
        svc, file_rec["id"], args.dest, expected_size, args.chunk_mb,
    )

    # Verify
    final_size = args.dest.stat().st_size
    if final_size != expected_size:
        print(f"\n[WARN] Size mismatch after download: local={final_size:,} vs drive={expected_size:,}")
        return 1
    print(f"\nDone. {_fmt_bytes(final_size)} in {duration/60:.1f} min "
          f"({final_size / 1024**2 / max(1, duration):.1f} MB/s avg)")

    # Update manifest
    manifest = load_manifest()
    manifest[file_rec["id"]] = {"path": str(args.dest), "size": expected_size}
    save_manifest(manifest)
    print(f"Manifest updated: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
