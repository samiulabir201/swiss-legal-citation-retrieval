"""drive_pull_curated.py — download a hand-picked list of decision-relevant Drive paths.

The INCLUDES list below is the curated ~2 GB set covering:
  - Tier 1: recall-0.89 v7.5 snapshot + per-query analysis + cached reranker scores
  - Tier 2: LLM enrichment outputs (court + law descriptors) + data_insights
  - Tier 3: Omnilex pipeline source + submission outputs

Skip-rules:
  - Files > per-include max_mb cap are skipped
  - Google-native files (Docs/Sheets) skipped
  - Files already in the cross-script manifest (_downloaded_ids.json) are skipped
  - Files already on disk at the right size are skipped

Edit INCLUDES below to add/remove paths. Each entry is either:
  - "Drive/path/to/folder/"    → pull every file under it (recursive)
  - "Drive/path/to/file.ext"   → pull just that file
  - ("Drive/path/", max_mb)    → pull folder but cap individual files at max_mb
"""

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
    sys.exit("Missing libs. Run drive_pull.py once first.")

# ─── config ──────────────────────────────────────────────────────────────
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
HERE = Path(__file__).resolve().parent
TOKEN_PATH = HERE / "token.json"
MANIFEST_PATH = HERE / "_downloaded_ids.json"
SYNC_ROOT = Path(r"e:\swiss_citation_extraction\drive_sync")

DEFAULT_MAX_FILE_MB = 1500  # per-file cap when none specified

FOLDER_MIME = "application/vnd.google-apps.folder"
NATIVE_PREFIX = "application/vnd.google-apps."

# Substrings in file/folder NAME to exclude (case-sensitive)
EXCLUDE_NAME_TOKENS = [
    "__pycache__", ".ipynb_checkpoints", ".no_exist", ".locks", "xet",
    # huge files we explicitly don't want
    "citation_graph_extracted.sqlite",  # 5.68 GB — regeneratable
    "segment_lattice_v3.sqlite",        # 5.50 GB — abandoned experiment
    "court_considerations_classified_citations.jsonl",  # 816 MB — large
]

# ─── the curated path list ──────────────────────────────────────────────
INCLUDES = [
    # ─────── TIER 1: decision-grade analysis artifacts (~150 MB) ───────
    "My Drive/swiss_law/research/anchor_funnel_val001_v7/",
    "My Drive/swiss_law/research/anchor_funnel_val001_v4/",
    "My Drive/swiss_law/research/anchor_funnel_val001/",
    "My Drive/swiss_law/research/hybrid_rerank_final/",
    "My Drive/swiss_law/research/rerank_only_diagnostic/",
    "My Drive/swiss_law/artifacts/eval/",
    "My Drive/swiss_law/submissions/",
    "My Drive/swiss_law/configs/",
    "My Drive/swiss_law/scripts/",
    "My Drive/swiss_law/artifacts_legalmalr/",
    "My Drive/swiss_law/cache_legalmalr/",

    # ─────── TIER 2: LLM enrichment outputs (~1 GB) ───────
    "My Drive/swiss_law/data/checkpoints/",
    "My Drive/swiss_law/outputs/legalmalr_cascade_v1/",
    ("My Drive/swiss_law/data_insights/", 1500),  # cap to skip the 5.68 GB sqlite (we exclude by name too)

    # ─────── TIER 3: Omnilex (~1 GB, source code mostly) ───────
    "My Drive/Omnilex-Agentic-Retrieval-Competition/retrieval/pipeline_cache/",
    "My Drive/Omnilex-Agentic-Retrieval-Competition/retrieval/pipeline_output_v3/",
    "My Drive/Omnilex-Agentic-Retrieval-Competition/retrieval/knowledge_base_optimized_hybrid_retrieval/",
    "My Drive/Omnilex-Agentic-Retrieval-Competition/swiss-law-pipeline/",
    "My Drive/Omnilex-Agentic-Retrieval-Competition/abbrev-translations.json",
]


# ─── auth + manifest ─────────────────────────────────────────────────────
def get_service():
    if not TOKEN_PATH.exists():
        sys.exit(f"token.json missing. Run drive_pull.py once first.")
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


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


# ─── path resolution ─────────────────────────────────────────────────────
def resolve_path(svc, path: str):
    """Resolve 'My Drive/folder/sub' → {id, mimeType, name} or None."""
    parts = path.strip("/").split("/")
    if parts[0] != "My Drive":
        return None
    parent_id = "root"
    last = {"id": "root", "mimeType": FOLDER_MIME, "name": "My Drive"}
    for p in parts[1:]:
        # Escape single quotes in name for the query
        p_esc = p.replace("'", "\\'")
        q = f"'{parent_id}' in parents and name = '{p_esc}' and trashed = false"
        resp = svc.files().list(
            q=q, pageSize=10,
            fields="files(id, name, mimeType)",
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        files = resp.get("files", [])
        if not files:
            return None
        last = files[0]
        parent_id = last["id"]
    return last


def list_children(svc, folder_id):
    out = []
    pt = None
    while True:
        resp = svc.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            pageSize=1000,
            fields="nextPageToken, files(id, name, mimeType, size)",
            pageToken=pt,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        out.extend(resp.get("files", []))
        pt = resp.get("nextPageToken")
        if not pt:
            break
    return out


def walk(svc, node, rel_path=""):
    """Yield (rel_path, item) for every non-folder under node."""
    if node["mimeType"] != FOLDER_MIME:
        yield rel_path or node["name"], node
        return
    stack = [(node["id"], "")]
    while stack:
        fid, rp = stack.pop()
        for c in list_children(svc, fid):
            if any(tok in c["name"] for tok in EXCLUDE_NAME_TOKENS):
                continue
            cp = f"{rp}/{c['name']}" if rp else c["name"]
            if c["mimeType"] == FOLDER_MIME:
                stack.append((c["id"], cp))
            else:
                yield cp, c


# ─── download ────────────────────────────────────────────────────────────
def download_one(svc, fid, dest, mime):
    if mime.startswith(NATIVE_PREFIX):
        return False
    request = svc.files().get_media(fileId=fid, supportsAllDrives=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with io.FileIO(str(dest), "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return True


def safe(s: str):
    return "".join("_" if c in '<>:"|?*\n\r' else c for c in s)


# ─── main ────────────────────────────────────────────────────────────────
def main():
    svc = get_service()
    manifest = load_manifest()
    print(f"Manifest entries already on disk: {len(manifest)}")

    ok = skip_resume = skip_manifest = skip_native = skip_cap = skip_excl = fail = 0
    pending = 0
    SAVE_EVERY = 25

    for entry in INCLUDES:
        max_mb = DEFAULT_MAX_FILE_MB
        if isinstance(entry, tuple):
            inc_path, max_mb = entry
        else:
            inc_path = entry
        inc_path = inc_path.rstrip("/")

        print(f"\n>>> {inc_path}  (cap: {max_mb} MB / file)")
        node = resolve_path(svc, inc_path)
        if not node:
            print(f"    NOT FOUND on Drive")
            continue

        # Output root: flattened path
        flat_name = safe(inc_path.replace("/", "__"))

        if node["mimeType"] != FOLDER_MIME:
            # Single file include
            files_iter = [("", node)]
        else:
            files_iter = walk(svc, node)

        for rel, item in files_iter:
            fid = item["id"]
            mime = item["mimeType"]
            size_bytes = int(item.get("size", 0)) if item.get("size") else 0
            size_mb = size_bytes / 1024 ** 2

            if mime.startswith(NATIVE_PREFIX):
                skip_native += 1
                continue
            if size_mb > max_mb and size_bytes > 0:
                print(f"    SKIP (>{max_mb} MB): {rel}  ({size_mb:.1f} MB)")
                skip_cap += 1
                continue

            # Determine dest path
            if node["mimeType"] == FOLDER_MIME:
                dest = SYNC_ROOT / flat_name / safe(rel)
            else:
                dest = SYNC_ROOT / flat_name  # single file: write to flat name

            # Manifest dedup
            if fid in manifest:
                p = Path(manifest[fid]["path"])
                if p.exists() and (size_bytes == 0 or p.stat().st_size == size_bytes):
                    skip_manifest += 1
                    continue

            # On-disk resume check
            if dest.exists() and dest.stat().st_size == size_bytes and size_bytes > 0:
                manifest[fid] = {"path": str(dest), "size": size_bytes}
                pending += 1
                skip_resume += 1
                if pending >= SAVE_EVERY:
                    save_manifest(manifest)
                    pending = 0
                continue

            try:
                t0 = time.time()
                download_one(svc, fid, dest, mime)
                dt = time.time() - t0
                print(f"    ok {rel or item['name']}  ({size_mb:.1f} MB in {dt:.1f}s)")
                manifest[fid] = {"path": str(dest), "size": size_bytes}
                pending += 1
                ok += 1
                if pending >= SAVE_EVERY:
                    save_manifest(manifest)
                    pending = 0
            except Exception as e:
                print(f"    FAIL {rel}: {e}")
                fail += 1

    save_manifest(manifest)
    print()
    print("=" * 70)
    print(f"DONE.  ok={ok}  skip_resume={skip_resume}  skip_manifest={skip_manifest}")
    print(f"       skip_native={skip_native}  skip_cap={skip_cap}  fail={fail}")
    print(f"Output: {SYNC_ROOT}")


if __name__ == "__main__":
    main()
