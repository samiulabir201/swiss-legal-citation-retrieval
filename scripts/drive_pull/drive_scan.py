"""drive_scan.py — scan-only inventory of Swiss/Omnilex folders on Google Drive.

NO DOWNLOADS. Builds a recursive tree of every matched folder and writes:
  e:\\swiss_citation_extraction\\drive_sync\\_scan_tree.txt   (pretty-printed)
  e:\\swiss_citation_extraction\\drive_sync\\_scan_tree.json  (structured)

Reuses token.json from drive_pull.py — no re-auth.

Run after drive_pull.py has created token.json:
    python drive_scan.py
"""

import json
import sys
from pathlib import Path

try:
    from googleapiclient.discovery import build
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
except ImportError:
    sys.exit("Missing libs. Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

# ─── config ──────────────────────────────────────────────────────────────
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
HERE = Path(__file__).resolve().parent
TOKEN_PATH = HERE / "token.json"
SYNC_ROOT = Path(r"e:\swiss_citation_extraction\drive_sync")

FOLDER_KEYWORDS = [
    "swiss", "omnilex", "legal_rag", "legal-rag", "legalrag",
    "swiss_law", "swiss-law", "citation", "anchor_funnel",
    "court_authority", "court_llm", "retrieval",
]
EXCLUDE_PATH_TOKENS = [
    "research_repos", "FlagEmbedding", "ColBERT", "LEXTREME",
    ".ipynb_checkpoints", "RLT4Reranking", "missing_link",
]

MAX_DEPTH = 10              # don't recurse deeper than this
MAX_FILES_PER_DIR = 8       # cap files printed per folder in pretty tree

FOLDER_MIME = "application/vnd.google-apps.folder"
GOOGLE_NATIVE_PREFIX = "application/vnd.google-apps."


# ─── auth ────────────────────────────────────────────────────────────────
def get_service():
    if not TOKEN_PATH.exists():
        sys.exit(f"token.json not found at {TOKEN_PATH}. Run drive_pull.py first to authenticate.")
    creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_PATH.write_text(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ─── search ──────────────────────────────────────────────────────────────
def find_candidate_folders(svc):
    by_id = {}
    for kw in FOLDER_KEYWORDS:
        page_token = None
        while True:
            q = f"mimeType = '{FOLDER_MIME}' and name contains '{kw}' and trashed = false"
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
    def resolve(fid):
        if fid in cache:
            return cache[fid]
        try:
            f = svc.files().get(fileId=fid, fields="id,name,parents", supportsAllDrives=True).execute()
        except Exception:
            cache[fid] = "?"
            return "?"
        parents = f.get("parents", [])
        if not parents:
            cache[fid] = f["name"]
            return f["name"]
        cache[fid] = resolve(parents[0]) + "/" + f["name"]
        return cache[fid]
    return resolve


# ─── walk ────────────────────────────────────────────────────────────────
def list_children(svc, folder_id):
    out = []
    page_token = None
    while True:
        resp = svc.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            pageSize=1000,
            fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
            pageToken=page_token,
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        out.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out


def build_tree(svc, folder_id, name, depth=0, progress=None):
    """Recursively build a tree node: {name, type, size, file_count, folder_count, children}."""
    if progress is not None:
        progress["n_folders"] += 1
        if progress["n_folders"] % 10 == 0:
            print(f"    walked {progress['n_folders']} folders…", flush=True)

    node = {
        "name": name, "type": "folder",
        "size": 0, "file_count": 0, "folder_count": 0,
        "children": [], "depth": depth,
    }

    if depth > MAX_DEPTH:
        node["truncated"] = True
        return node

    try:
        children = list_children(svc, folder_id)
    except Exception as e:
        node["error"] = str(e)
        return node

    subfolders, files = [], []
    for c in children:
        if c["mimeType"] == FOLDER_MIME:
            subfolders.append(c)
        else:
            size = int(c.get("size", 0)) if c.get("size") else 0
            is_native = c["mimeType"].startswith(GOOGLE_NATIVE_PREFIX)
            files.append({
                "name": c["name"], "type": "file",
                "size": size, "mime": c["mimeType"],
                "native": is_native,
            })
            node["size"] += size
            node["file_count"] += 1

    for sf in subfolders:
        cn = build_tree(svc, sf["id"], sf["name"], depth + 1, progress)
        node["size"] += cn.get("size", 0)
        node["file_count"] += cn.get("file_count", 0)
        node["folder_count"] += 1 + cn.get("folder_count", 0)
        node["children"].append(cn)

    # Sort: folders first (biggest), then files (biggest)
    files.sort(key=lambda x: x["size"], reverse=True)
    node["children"].extend(files)
    return node


# ─── render ──────────────────────────────────────────────────────────────
def fmt_size(b):
    if b < 1024: return f"{b} B"
    if b < 1024 ** 2: return f"{b / 1024:.1f} KB"
    if b < 1024 ** 3: return f"{b / 1024 ** 2:.1f} MB"
    return f"{b / 1024 ** 3:.2f} GB"


def render(node, lines, indent=0):
    pad = "  " * indent
    if node["type"] == "folder":
        info = f"({fmt_size(node['size'])}, {node['file_count']} files, {node['folder_count']} subfolders)"
        if node.get("truncated"):
            info += " [TRUNCATED — too deep]"
        if node.get("error"):
            info += f" [ERROR: {node['error']}]"
        lines.append(f"{pad}[DIR] {node['name']}/   {info}")

        folders = [c for c in node["children"] if c["type"] == "folder"]
        files = [c for c in node["children"] if c["type"] == "file"]
        # folders already big-first since we extended children from list-comprehension order;
        # be explicit anyway
        folders.sort(key=lambda x: x["size"], reverse=True)
        files.sort(key=lambda x: x["size"], reverse=True)

        for f in folders:
            render(f, lines, indent + 1)
        for f in files[:MAX_FILES_PER_DIR]:
            render(f, lines, indent + 1)
        if len(files) > MAX_FILES_PER_DIR:
            lines.append(f"{pad}  [+{len(files) - MAX_FILES_PER_DIR} more files in this folder]")
    else:
        flag = "  [Google native]" if node.get("native") else ""
        lines.append(f"{pad}      {node['name']}   ({fmt_size(node['size'])}){flag}")


# ─── main ────────────────────────────────────────────────────────────────
def main():
    svc = get_service()

    print("Phase 1: finding folders matching keywords…")
    raw = find_candidate_folders(svc)
    print(f"  {len(raw)} candidates by name match.")

    resolve = make_path_resolver(svc)
    for f in raw:
        f["path"] = resolve(f["id"])
    folders = [f for f in raw if not any(t in f["path"] for t in EXCLUDE_PATH_TOKENS)]
    print(f"  {len(folders)} after excluding third-party.")

    # Collapse: keep only top-level matches (drop folders whose ancestor is also matched)
    paths = sorted({f["path"] for f in folders}, key=len)
    top_level = []
    for f in folders:
        is_desc = any(f["path"] != p and f["path"].startswith(p + "/") for p in paths)
        if not is_desc:
            top_level.append(f)
    print(f"  {len(top_level)} top-level (after collapsing nested duplicates).\n")

    SYNC_ROOT.mkdir(parents=True, exist_ok=True)
    progress = {"n_folders": 0}

    all_lines = []
    all_trees = []
    for i, f in enumerate(top_level, 1):
        print(f"[{i}/{len(top_level)}] walking {f['path']}…")
        tree = build_tree(svc, f["id"], f["path"], depth=0, progress=progress)
        all_trees.append(tree)
        all_lines.append("\n" + "=" * 90)
        all_lines.append(f"# {f['path']}    "
                         f"({fmt_size(tree['size'])}, {tree['file_count']} files, "
                         f"{tree['folder_count']} subfolders)")
        all_lines.append("=" * 90)
        render(tree, all_lines, indent=0)

    # Write outputs
    txt_path = SYNC_ROOT / "_scan_tree.txt"
    json_path = SYNC_ROOT / "_scan_tree.json"
    txt_path.write_text("\n".join(all_lines), encoding="utf-8")
    json_path.write_text(json.dumps(all_trees, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console summary table
    print()
    print("=" * 90)
    print(f"{'#':>3}  {'GB':>9}  {'files':>7}  {'subdirs':>8}  Drive path")
    print("-" * 90)
    for i, t in enumerate(sorted(all_trees, key=lambda x: x["size"], reverse=True)):
        print(f"{i:>3}  {t['size'] / 1024 ** 3:>9.3f}  {t['file_count']:>7}  "
              f"{t['folder_count']:>8}  {t['name']}")
    print("=" * 90)
    total_size = sum(t["size"] for t in all_trees)
    total_files = sum(t["file_count"] for t in all_trees)
    print(f"\nTOTAL across {len(all_trees)} folders: {fmt_size(total_size)}, {total_files} files")
    print(f"\nFull tree written to:")
    print(f"  {txt_path}")
    print(f"  {json_path}")
    print(f"\nReview the tree, then re-run `drive_pull_folders.py` with the indices you want.")


if __name__ == "__main__":
    main()
