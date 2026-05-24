"""pull_v75_chan16_run.py — pull the v7.5 channel-16 run outputs from Drive to local disk.

Uses the same OAuth setup as `drive_pull_folders.py` (token.json + credentials.json
in this directory). Downloads three things:

1. The `multi_aspect/` folder — channel-16 source artifacts (parquet, npy, json).
2. The `anchor_funnel_val001_v7/snapshot/` folder — v7.5 Phase-11 run outputs.
   By default only the small files (JSON / Parquet / sub-500-MB) are pulled — the
   cascade-snapshot NPY chunks are skipped. Pass --include-corpus to override.
3. The most-recently-modified v7.5 notebook on Drive (the executed one with outputs).

Output lands under:
    e:\\swiss_citation_extraction\\research\\v75_chan16_download\\
        v75_run\\          <-- contents of anchor_funnel_val001_v7/snapshot
        channel_16\\       <-- contents of multi_aspect
        pool_v75_*.ipynb   <-- the notebook

Resume / dedup uses the shared `_downloaded_ids.json` manifest.

USAGE
-----
    python pull_v75_chan16_run.py               # default: skip big corpus files
    python pull_v75_chan16_run.py --include-corpus     # also download big NPYs
    python pull_v75_chan16_run.py --dry-run            # walk + print, no download
    python pull_v75_chan16_run.py --notebook-id <id>   # skip search, pull this notebook id
"""
from __future__ import annotations

import argparse
import io
import json
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

# ── config ──────────────────────────────────────────────────────────────────
SCOPES        = ["https://www.googleapis.com/auth/drive.readonly"]
HERE          = Path(__file__).resolve().parent
TOKEN_PATH    = HERE / "token.json"
CRED_PATH     = HERE / "credentials.json"
MANIFEST_PATH = HERE / "_downloaded_ids.json"

PROJECT_ROOT  = Path(r"e:\swiss_citation_extraction")
OUT_ROOT      = PROJECT_ROOT / "research" / "v75_chan16_download"

# Folder name patterns to find on Drive. Each maps to a local subdir under OUT_ROOT.
FOLDER_TARGETS = [
    {"name": "multi_aspect",             "local": "channel_16",
     "path_must_contain": "concept_embedding_path"},
    {"name": "snapshot",                 "local": "v75_run",
     "path_must_contain": "anchor_funnel_val001_v7"},
]

# Notebook search — tries exact match first, then substring on each token.
# Add more variants here if the file gets renamed.
NOTEBOOK_NAME_TOKENS = [
    "pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb",  # exact
    "pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL",        # name without ext
    "pool_v75_multiquery_recall",                               # token
    "pool_v75_multiquery",                                      # token
    "v75_multiquery",                                           # broader fallback
]

# Size policy
MAX_FILE_MB_SMALL_DEFAULT = 500          # default cap when --include-corpus is OFF
MAX_FILE_MB_FULL          = 10_000       # cap when --include-corpus is ON

# File-extension whitelist for the "small" / analysis-only mode
SMALL_MODE_EXTS = {".json", ".parquet", ".ipynb", ".md", ".csv", ".pkl"}
# We also accept .npy if its size is below MAX_FILE_MB_SMALL_DEFAULT.

FOLDER_MIME           = "application/vnd.google-apps.folder"
GOOGLE_NATIVE_PREFIX  = "application/vnd.google-apps."


# ── auth ────────────────────────────────────────────────────────────────────
def get_service():
    if not TOKEN_PATH.exists():
        sys.exit("token.json not found. Run drive_pull.py first to authenticate.")
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ── path resolver (cached) ──────────────────────────────────────────────────
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
            cache[file_id] = "?"
            return "?"
        parents = f.get("parents", [])
        if not parents:
            cache[file_id] = f["name"]
            return f["name"]
        p = resolve(parents[0])
        full = f"{p}/{f['name']}"
        cache[file_id] = full
        return full
    return resolve


# ── search ──────────────────────────────────────────────────────────────────
def find_folders_by_name(svc, name_token: str):
    items, page_token = [], None
    while True:
        q = (f"mimeType = '{FOLDER_MIME}' "
             f"and name contains '{name_token}' "
             f"and trashed = false")
        resp = svc.files().list(
            q=q, pageSize=200,
            fields="nextPageToken, files(id, name, parents)",
            pageToken=page_token,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def _search_by_name(svc, q_extra: str, mime_clause: str = ""):
    """Run a single Drive query, return all matching files."""
    items, page_token = [], None
    while True:
        q = f"{q_extra} and trashed = false"
        if mime_clause:
            q += f" and {mime_clause}"
        resp = svc.files().list(
            q=q, pageSize=200,
            fields="nextPageToken, files(id, name, parents, modifiedTime, size, mimeType)",
            pageToken=page_token,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def find_notebook_by_name(svc, name_tokens: list[str]):
    """Search Drive for an .ipynb matching any of the tokens. No MIME filter
    because Colab notebooks register as `application/vnd.google.colaboratory`,
    `application/x-ipynb+json`, `application/json`, or `application/octet-stream`
    depending on how they were uploaded."""
    by_id = {}
    # 1. Exact-name match — fastest path if you know the file name.
    for tok in name_tokens:
        for f in _search_by_name(svc, f"name = '{tok}'"):
            by_id[f["id"]] = f
    # 2. Substring match for each token.
    for tok in name_tokens:
        for f in _search_by_name(svc, f"name contains '{tok}'"):
            if f["name"].lower().endswith(".ipynb"):
                by_id[f["id"]] = f
    items = list(by_id.values())
    items.sort(key=lambda f: f.get("modifiedTime", ""), reverse=True)
    return items


def list_children(svc, folder_id: str):
    children, page_token = [], None
    while True:
        resp = svc.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            pageSize=1000,
            fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
            pageToken=page_token,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        children.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return children


def walk_recursive(svc, folder_id: str, rel_path: str = ""):
    stack = [(folder_id, rel_path)]
    while stack:
        fid, rp = stack.pop()
        for c in list_children(svc, fid):
            cp = f"{rp}/{c['name']}" if rp else c["name"]
            if c["mimeType"] == FOLDER_MIME:
                stack.append((c["id"], cp))
            else:
                yield cp, c


# ── download ────────────────────────────────────────────────────────────────
def safe_pathname(name: str) -> str:
    bad = '<>:"|?*\n\r'
    return "".join("_" if c in bad else c for c in name)


def download_one(svc, file_id: str, dest: Path, mime_type: str) -> bool:
    if mime_type.startswith(GOOGLE_NATIVE_PREFIX):
        return False
    request = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with io.FileIO(str(dest), "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return True


# ── manifest ────────────────────────────────────────────────────────────────
def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_manifest(m: dict) -> None:
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(MANIFEST_PATH)


def already_downloaded(file_id, size_bytes, manifest) -> bool:
    rec = manifest.get(file_id)
    if not rec:
        return False
    p = Path(rec["path"])
    if not p.exists():
        return False
    if size_bytes and p.stat().st_size != size_bytes:
        return False
    return True


# ── policy ──────────────────────────────────────────────────────────────────
def keep_in_small_mode(name: str, size_bytes: int, max_mb: int) -> bool:
    ext = Path(name).suffix.lower()
    size_mb = size_bytes / 1024**2 if size_bytes else 0
    if ext in SMALL_MODE_EXTS:
        return size_mb <= max_mb
    if ext == ".npy":
        return size_mb <= max_mb
    return False


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--include-corpus", action="store_true",
                    help="Also pull large NPY corpus chunks (default: only small files).")
    ap.add_argument("--max-mb", type=int, default=None,
                    help="Per-file cap in MB. Overrides the default 500 (small mode) "
                         "or 10000 (with --include-corpus).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Walk and print only — no download.")
    ap.add_argument("--notebook-id", type=str, default=None,
                    help="Skip notebook search, pull this Drive file ID instead.")
    ap.add_argument("--out", type=Path, default=OUT_ROOT,
                    help="Local output root.")
    args = ap.parse_args()

    max_mb = args.max_mb or (
        MAX_FILE_MB_FULL if args.include_corpus else MAX_FILE_MB_SMALL_DEFAULT
    )

    svc = get_service()
    manifest = load_manifest()
    resolve = make_path_resolver(svc)
    args.out.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: find target folders ────────────────────────────────────────
    print("=" * 72)
    print("Phase 1 — locating target folders on Drive")
    print("=" * 72)
    folder_assignments = []   # list of {id, name, drive_path, local_subdir}
    for tgt in FOLDER_TARGETS:
        candidates = find_folders_by_name(svc, tgt["name"])
        # Filter by path requirement
        keep = []
        for c in candidates:
            full = resolve(c["id"])
            if tgt["path_must_contain"].lower() in full.lower():
                keep.append({**c, "drive_path": full})
        if not keep:
            print(f"  [WARN] no folder named '{tgt['name']}' under '{tgt['path_must_contain']}'")
            continue
        # If multiple match, take the deepest path (most specific)
        keep.sort(key=lambda x: -len(x["drive_path"]))
        chosen = keep[0]
        if len(keep) > 1:
            print(f"  multiple matches for '{tgt['name']}' — using deepest:")
            for k in keep:
                tag = "  ← chosen" if k is chosen else ""
                print(f"      {k['drive_path']}{tag}")
        folder_assignments.append({
            "id": chosen["id"],
            "name": chosen["name"],
            "drive_path": chosen["drive_path"],
            "local_subdir": tgt["local"],
        })
        print(f"  ✓ {tgt['name']:<14} → {chosen['drive_path']}")

    # ── Phase 2: find the notebook ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Phase 2 — locating v7.5 notebook on Drive")
    print("=" * 72)
    nb_record = None
    if args.notebook_id:
        try:
            f = svc.files().get(
                fileId=args.notebook_id,
                fields="id,name,size,modifiedTime",
                supportsAllDrives=True,
            ).execute()
            nb_record = {**f, "drive_path": resolve(f["id"])}
            print(f"  using --notebook-id: {nb_record['drive_path']}")
        except Exception as e:
            print(f"  [ERROR] fetching notebook id {args.notebook_id}: {e}")
    else:
        hits = find_notebook_by_name(svc, NOTEBOOK_NAME_TOKENS)
        if not hits:
            print(f"  [WARN] no .ipynb matching any of {NOTEBOOK_NAME_TOKENS} on Drive.")
            print("  Try `--notebook-id <id>`. Find the id by opening the file in Drive web "
                  "and copying the URL fragment after `/d/`.")
        else:
            print(f"  found {len(hits)} candidates (sorted by modifiedTime desc):")
            for i, f in enumerate(hits[:10]):
                size_mb = int(f.get("size", 0)) / 1024**2 if f.get("size") else 0
                print(f"    [{i}] {f.get('modifiedTime')}  {size_mb:>7.1f} MB  "
                      f"mime={f.get('mimeType','?'):<45}  {resolve(f['id'])}")
            nb_record = {**hits[0], "drive_path": resolve(hits[0]["id"])}
            print(f"  ✓ picking the most recent: {nb_record['drive_path']}")

    # ── Phase 3: walk and tally ─────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"Phase 3 — walking targets  (size cap: {max_mb} MB/file, "
          f"include_corpus={args.include_corpus})")
    print("=" * 72)
    plan = []   # list of (file_dict, dest_path)
    for fa in folder_assignments:
        local_root = args.out / fa["local_subdir"]
        total = kept = skipped = 0
        skipped_reasons = []
        for rel, item in walk_recursive(svc, fa["id"]):
            total += 1
            mime = item["mimeType"]
            if mime == FOLDER_MIME or mime.startswith(GOOGLE_NATIVE_PREFIX):
                skipped += 1
                continue
            size_bytes = int(item.get("size", 0)) if item.get("size") else 0
            size_mb = size_bytes / 1024**2
            if not args.include_corpus:
                if not keep_in_small_mode(item["name"], size_bytes, max_mb):
                    skipped += 1
                    skipped_reasons.append(f"{rel} ({size_mb:.0f} MB)")
                    continue
            else:
                if size_mb > max_mb:
                    skipped += 1
                    skipped_reasons.append(f"{rel} ({size_mb:.0f} MB > cap)")
                    continue
            dest = local_root / safe_pathname(rel)
            plan.append((item, dest, size_mb))
            kept += 1
        print(f"  {fa['drive_path']}")
        print(f"     total files: {total}   selected: {kept}   skipped: {skipped}")
        if skipped_reasons and not args.include_corpus:
            print(f"     skipped (too big or non-whitelist ext):")
            for r in skipped_reasons[:6]:
                print(f"        - {r}")
            if len(skipped_reasons) > 6:
                print(f"        ... and {len(skipped_reasons) - 6} more")

    # Notebook
    if nb_record:
        dest_nb = args.out / safe_pathname(nb_record["name"])
        plan.append((nb_record, dest_nb, int(nb_record.get("size", 0)) / 1024**2))

    total_mb = sum(sz for _, _, sz in plan)
    print(f"\nPlan: {len(plan)} files, ~{total_mb:.1f} MB total")
    if args.dry_run:
        print("--dry-run set — exiting before download.")
        return 0

    # ── Phase 4: download ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Phase 4 — downloading")
    print("=" * 72)
    args.out.mkdir(parents=True, exist_ok=True)
    ok = skip_resume = skip_manifest = fail = 0
    SAVE_EVERY, pending = 25, 0

    for item, dest, size_mb in plan:
        fid       = item["id"]
        size_b    = int(item.get("size", 0)) if item.get("size") else 0
        rel_disp  = dest.relative_to(args.out)

        if already_downloaded(fid, size_b, manifest):
            skip_manifest += 1
            continue
        if dest.exists() and size_b > 0 and dest.stat().st_size == size_b:
            manifest[fid] = {"path": str(dest), "size": size_b}
            pending += 1
            skip_resume += 1
            if pending >= SAVE_EVERY:
                save_manifest(manifest); pending = 0
            continue

        try:
            download_one(svc, fid, dest, item["mimeType"])
            manifest[fid] = {"path": str(dest), "size": size_b}
            ok += 1; pending += 1
            print(f"  ok  {size_mb:>7.1f} MB  {rel_disp}")
            if pending >= SAVE_EVERY:
                save_manifest(manifest); pending = 0
        except Exception as e:
            fail += 1
            print(f"  FAIL  {rel_disp}: {e}")

    save_manifest(manifest)
    print()
    print(f"Done.  downloaded={ok}  resume={skip_resume}  "
          f"manifest_dup={skip_manifest}  fail={fail}")
    print(f"Output: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
