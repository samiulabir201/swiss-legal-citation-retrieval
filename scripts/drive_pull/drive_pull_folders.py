"""drive_pull_folders.py — pull whole folder trees from Google Drive.

Companion to drive_pull.py (which only pulls .ipynb by filename).
This one finds FOLDERS matching keywords, walks them recursively, prints a
size summary, then downloads everything below configurable caps.

Reuses the same token.json — no re-auth needed.

Output: e:\\swiss_citation_extraction\\drive_sync\\<flattened-drive-path>\\...
"""

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
    sys.exit("Missing libs. Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

# ─── config ──────────────────────────────────────────────────────────────
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
HERE = Path(__file__).resolve().parent
TOKEN_PATH = HERE / "token.json"
CRED_PATH = HERE / "credentials.json"
SYNC_ROOT = Path(r"e:\swiss_citation_extraction\drive_sync")
MANIFEST_PATH = HERE / "_downloaded_ids.json"  # shared dedup manifest (file_id → local path)

# Keywords to find candidate folders (case-insensitive substring match on folder name)
FOLDER_KEYWORDS = [
    "swiss", "omnilex", "legal_rag", "legal-rag", "legalrag",
    "swiss_law", "swiss-law", "citation", "anchor_funnel",
    "court_authority", "court_llm", "retrieval",
]

# Excluded path tokens — third-party tutorial code we don't want
EXCLUDE_PATH_TOKENS = [
    "research_repos", "FlagEmbedding", "ColBERT", "LEXTREME",
    ".ipynb_checkpoints", "RLT4Reranking", "missing_link",
]

# Caps — skip individual files or whole folders above these
MAX_FILE_MB = 6000     # individual file > this MB → skip
MAX_FOLDER_GB = 20     # whole folder > this GB → flag & ask before pull

FOLDER_MIME = "application/vnd.google-apps.folder"
GOOGLE_NATIVE_PREFIX = "application/vnd.google-apps."


# ─── auth ────────────────────────────────────────────────────────────────
def get_service():
    if not TOKEN_PATH.exists():
        sys.exit(f"token.json not found. Run drive_pull.py first to authenticate.")
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ─── search ──────────────────────────────────────────────────────────────
def find_candidate_folders(svc):
    """Return [(id, name)] for every folder whose name contains a keyword."""
    by_id = {}
    for kw in FOLDER_KEYWORDS:
        page_token = None
        while True:
            q = (f"mimeType = '{FOLDER_MIME}' "
                 f"and name contains '{kw}' "
                 f"and trashed = false")
            resp = svc.files().list(
                q=q, pageSize=200,
                fields="nextPageToken, files(id, name, parents)",
                pageToken=page_token,
                supportsAllDrives=True, includeItemsFromAllDrives=True,
            ).execute()
            for f in resp.get("files", []):
                by_id[f["id"]] = f
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    return list(by_id.values())


def make_path_resolver(svc):
    cache = {}
    def resolve(file_id):
        if file_id in cache:
            return cache[file_id]
        try:
            f = svc.files().get(fileId=file_id, fields="id,name,parents", supportsAllDrives=True).execute()
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


# ─── walk ────────────────────────────────────────────────────────────────
def list_children(svc, folder_id):
    children = []
    page_token = None
    while True:
        resp = svc.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            pageSize=1000,
            fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, parents)",
            pageToken=page_token,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        for f in resp.get("files", []):
            children.append(f)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return children


def walk_recursive(svc, folder_id, rel_path="", progress_every=200):
    """Yield (rel_path, file_dict) for every non-folder under folder_id."""
    stack = [(folder_id, rel_path)]
    yielded = 0
    while stack:
        fid, rp = stack.pop()
        for c in list_children(svc, fid):
            cp = f"{rp}/{c['name']}" if rp else c["name"]
            if c["mimeType"] == FOLDER_MIME:
                stack.append((c["id"], cp))
            else:
                yielded += 1
                if yielded % progress_every == 0:
                    print(f"    walked {yielded} files so far…", flush=True)
                yield cp, c


# ─── download ────────────────────────────────────────────────────────────
def download_one(svc, file_id, dest, mime_type):
    if mime_type.startswith(GOOGLE_NATIVE_PREFIX):
        return False  # native Google formats need export, skip for now
    request = svc.files().get_media(fileId=file_id, supportsAllDrives=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with io.FileIO(str(dest), "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return True


def safe_pathname(name):
    bad = '<>:"|?*\n\r'
    out = "".join("_" if c in bad else c for c in name)
    return out


# ─── dedup manifest (resume + cross-run) ─────────────────────────────────
def load_manifest():
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_manifest(m):
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(MANIFEST_PATH)


def already_downloaded(file_id, size_bytes, manifest):
    """True if manifest records this id at a still-existing path with matching size."""
    rec = manifest.get(file_id)
    if not rec:
        return False
    p = Path(rec["path"])
    if not p.exists():
        return False
    if size_bytes and p.stat().st_size != size_bytes:
        return False
    return True


# ─── main ────────────────────────────────────────────────────────────────
def main():
    svc = get_service()

    # Phase 1 — find candidate folders
    print("Phase 1: finding folders matching keywords…")
    raw = find_candidate_folders(svc)
    print(f"  {len(raw)} candidates by name match.")

    resolve = make_path_resolver(svc)
    for f in raw:
        f["path"] = resolve(f["id"])
    folders = [f for f in raw if not any(tok in f["path"] for tok in EXCLUDE_PATH_TOKENS)]
    print(f"  {len(folders)} after excluding third-party paths.")

    # Phase 2 — walk each to compute totals
    print("\nPhase 2: walking each candidate folder to compute size + file count…")
    summaries = []
    for i, f in enumerate(folders, 1):
        print(f"  [{i}/{len(folders)}] {f['path']}")
        items = list(walk_recursive(svc, f["id"]))
        total_bytes = sum(int(it[1].get("size", 0)) for it in items if it[1].get("size"))
        native_count = sum(1 for _, it in items if it["mimeType"].startswith(GOOGLE_NATIVE_PREFIX))
        n_files = len(items)
        big_files = sorted(
            [(it[1]["name"], int(it[1].get("size", 0))) for it in items if it[1].get("size")],
            key=lambda x: x[1], reverse=True,
        )[:3]
        summaries.append({
            "id": f["id"], "drive_path": f["path"],
            "n_files": n_files,
            "total_bytes": total_bytes,
            "total_gb": round(total_bytes / 1024 ** 3, 3),
            "native_count": native_count,
            "top3_big": [(n, round(s / 1024 ** 2, 1)) for n, s in big_files],
            "items": items,
        })
    summaries.sort(key=lambda s: s["total_gb"], reverse=True)

    # Print table
    print()
    print(f"{'#':>3}  {'GB':>9}  {'files':>7}  {'native':>7}  Drive path")
    print("-" * 130)
    for i, s in enumerate(summaries):
        flag = "  ⚠ >20GB" if s["total_gb"] > MAX_FOLDER_GB else ""
        print(f"{i:>3}  {s['total_gb']:>9.3f}  {s['n_files']:>7}  {s['native_count']:>7}  {s['drive_path']}{flag}")
        for name, mb in s["top3_big"]:
            print(f"        biggest: {name}  ({mb} MB)")

    SYNC_ROOT.mkdir(parents=True, exist_ok=True)
    meta_to_save = [{k: v for k, v in s.items() if k != "items"} for s in summaries]
    (SYNC_ROOT / "_folder_summary.json").write_text(
        json.dumps(meta_to_save, indent=2, ensure_ascii=False), encoding="utf-8")

    # Phase 3 — pick what to pull
    print(f"\nCaps: skip files > {MAX_FILE_MB} MB. Per-folder size is informational only.")
    print(f"Dedup manifest: {MANIFEST_PATH.name} ({len(load_manifest())} files recorded)")
    print("\nChoices:")
    print(f"  'all'   → every folder ≤ {MAX_FOLDER_GB} GB total")
    print("  'big'   → EVERY folder including >20GB ones (per-file cap still applies)")
    print("  '0,2,5' → comma-separated indices from the table (no folder cap)")
    print("  'q'     → quit (table above + JSON saved)")
    ans = input("\nChoice: ").strip().lower()
    if ans in ("q", ""):
        print("Quit. Summary saved to drive_sync/_folder_summary.json")
        return

    if ans == "all":
        to_pull = [s for s in summaries if s["total_gb"] <= MAX_FOLDER_GB]
        excluded_big = [s for s in summaries if s["total_gb"] > MAX_FOLDER_GB]
        if excluded_big:
            print(f"\n  (excluded {len(excluded_big)} folder(s) > {MAX_FOLDER_GB} GB — type 'big' to include them)")
    elif ans == "big":
        to_pull = list(summaries)  # everything, no folder-size filter
    else:
        try:
            idxs = [int(x.strip()) for x in ans.split(",")]
            to_pull = [summaries[i] for i in idxs]
        except Exception as e:
            print(f"Bad input: {e}")
            return

    # Dedupe: drop a selected folder if a selected ANCESTOR is also chosen
    # (avoids walking & writing the same content twice)
    paths_to_pull = sorted({s["drive_path"] for s in to_pull}, key=len)
    deduped = []
    for s in to_pull:
        is_descendant = any(
            s["drive_path"] != p and s["drive_path"].startswith(p + "/")
            for p in paths_to_pull
        )
        if is_descendant:
            print(f"skipping (covered by ancestor): {s['drive_path']}")
            continue
        deduped.append(s)

    # Phase 4 — download
    print(f"\nDownloading {len(deduped)} folder(s)…")
    manifest = load_manifest()
    SAVE_EVERY = 25
    pending_saves = 0

    seen_ids = set()
    ok = skip_resume = skip_manifest = skip_native = skip_dup = fail = 0
    skipped_too_big = []
    for s in deduped:
        print(f"\n>>> {s['drive_path']}  ({s['n_files']} files, {s['total_gb']} GB)")
        out_root = SYNC_ROOT / safe_pathname(s["drive_path"].replace("/", "__"))
        for rel, item in s["items"]:
            fid = item["id"]
            if fid in seen_ids:
                skip_dup += 1
                continue
            seen_ids.add(fid)
            if item["mimeType"] == FOLDER_MIME:
                continue
            if item["mimeType"].startswith(GOOGLE_NATIVE_PREFIX):
                skip_native += 1
                continue
            size_bytes = int(item.get("size", 0)) if item.get("size") else 0
            size_mb = size_bytes / 1024 ** 2
            if size_mb > MAX_FILE_MB:
                skipped_too_big.append((rel, size_mb))
                skip_resume += 1
                continue
            dest = out_root / safe_pathname(rel)

            # Cross-run dedup: file_id manifest takes precedence
            if already_downloaded(fid, size_bytes, manifest):
                skip_manifest += 1
                continue

            # Same-run / interrupted-resume: dest path exists with right size
            if dest.exists() and dest.stat().st_size == size_bytes and size_bytes > 0:
                # Register so future runs see it via manifest too
                manifest[fid] = {"path": str(dest), "size": size_bytes}
                pending_saves += 1
                skip_resume += 1
                if pending_saves >= SAVE_EVERY:
                    save_manifest(manifest)
                    pending_saves = 0
                continue

            try:
                download_one(svc, fid, dest, item["mimeType"])
                print(f"  ok {rel}  ({size_mb:.1f} MB)")
                manifest[fid] = {"path": str(dest), "size": size_bytes}
                pending_saves += 1
                ok += 1
                if pending_saves >= SAVE_EVERY:
                    save_manifest(manifest)
                    pending_saves = 0
            except Exception as e:
                print(f"  FAIL {rel}: {e}")
                fail += 1

    save_manifest(manifest)
    print(f"\nDone. ok={ok}  skip_resume={skip_resume}  skip_manifest={skip_manifest}  "
          f"skip_native={skip_native}  skip_dup={skip_dup}  fail={fail}")
    print(f"  resume        = already at dest path with matching size")
    print(f"  manifest      = same Drive file already downloaded elsewhere this/prior run")
    print(f"  native        = Google native format (Docs/Sheets) — needs export, skipped")
    print(f"  dup           = same file_id appears in two selected folder trees")
    if skipped_too_big:
        print(f"\nFiles skipped because > {MAX_FILE_MB} MB:")
        for r, m in skipped_too_big[:30]:
            print(f"  {m:>8.1f} MB  {r}")
        if len(skipped_too_big) > 30:
            print(f"  ... and {len(skipped_too_big) - 30} more")
    print(f"\nFiles in: {SYNC_ROOT}")
    print("Next: tell me 'organize the drive sync' and I'll fold these into the project tree.")


if __name__ == "__main__":
    main()
