"""drive_pull.py — list & download Swiss-citation notebooks from your Google Drive.

Runs on your local Windows machine. NO password is ever typed into this script.
First run opens a browser for one-time OAuth consent, saves a token, then exits.
Subsequent runs use the saved token (no browser, no auth).

----------------------------------------------------------------------------
ONE-TIME SETUP (3-5 minutes, done once, never again)
----------------------------------------------------------------------------
You need a `credentials.json` file in the same folder as this script. Get it like this:

1. Go to https://console.cloud.google.com/projectcreate
   - Create a new project (any name, e.g. "swiss-drive-pull"). Click CREATE.

2. After it's created, go to: https://console.cloud.google.com/apis/library/drive.googleapis.com
   - Click ENABLE.

3. Go to: https://console.cloud.google.com/apis/credentials/consent
   - User Type: External   →  CREATE
   - App name: swiss_drive_pull
   - User support email: <your Gmail>
   - Developer contact: <your Gmail>
   - SAVE AND CONTINUE through "Scopes" (no changes)
   - On "Test users": ADD USERS → enter your Gmail (the one whose Drive you'll read)
   - SAVE AND CONTINUE → BACK TO DASHBOARD

4. Go to: https://console.cloud.google.com/apis/credentials
   - + CREATE CREDENTIALS  →  OAuth client ID
   - Application type: **Desktop app**
   - Name: anything
   - CREATE
   - In the popup, click DOWNLOAD JSON.

5. Save that downloaded file as `credentials.json` in this folder:
   `e:\\swiss_citation_extraction\\scripts\\drive_pull\\credentials.json`

----------------------------------------------------------------------------
INSTALL THE LIBRARIES (once)
----------------------------------------------------------------------------
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

----------------------------------------------------------------------------
RUN
----------------------------------------------------------------------------
    cd e:\\swiss_citation_extraction\\scripts\\drive_pull
    python drive_pull.py

On first run a browser window opens, you sign in with your Gmail, click Allow.
A `token.json` is written next to this script. Don't share it; it's like a
session cookie.

The script will:
  - search your Drive for *.ipynb files matching swiss/citation/omnilex keywords
  - print a list with size + modified time + path
  - prompt before downloading
  - save matches to e:\\swiss_citation_extraction\\notebooks\\_drive_pull\\
"""

import io
import json
import re
import sys
from pathlib import Path

# ---- dependency check with friendly error -------------------------------
try:
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError:
    sys.exit(
        "Missing libraries. Install with:\n"
        "  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
    )

# ---- config --------------------------------------------------------------
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
HERE = Path(__file__).resolve().parent
CRED_PATH = HERE / "credentials.json"
TOKEN_PATH = HERE / "token.json"
OUT_DIR = Path(r"e:\swiss_citation_extraction\notebooks\_drive_pull")

# Keywords used for filename filter. Add/remove as you like.
KEYWORDS = [
    "swiss", "citation", "omnilex", "anchor_funnel",
    "court_citation", "court_llm", "enrich_laws_de", "enrich_cards",
    "legalmalr", "law_pipeline", "qwen3_8b_awq",
]

# Folder names to *exclude* (third-party tutorial repos cloned into Drive, etc.)
EXCLUDE_PATH_TOKENS = [
    "research_repos", "FlagEmbedding", "ColBERT", "LEXTREME",
    ".ipynb_checkpoints",
]


# ---- auth ----------------------------------------------------------------
def get_service():
    creds = None
    if TOKEN_PATH.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
        except Exception:
            creds = None
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CRED_PATH.exists():
                sys.exit(
                    f"\nERROR: {CRED_PATH.name} not found in {HERE}\n"
                    f"Read the setup steps at the top of this file to obtain it (3-5 minutes).\n"
                )
            print("First-run OAuth: a browser window will open. Sign in with your Gmail and click Allow.")
            flow = InstalledAppFlow.from_client_secrets_file(str(CRED_PATH), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json())
        print(f"Token saved to {TOKEN_PATH} (subsequent runs skip the browser).")
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ---- search --------------------------------------------------------------
def list_matches(service):
    """Search Drive for .ipynb files containing any keyword in the name.
       Drive API doesn't support OR on `name contains`, so loop per keyword and dedupe."""
    by_id = {}
    for kw in KEYWORDS:
        page_token = None
        while True:
            q = f"name contains '{kw}' and name contains '.ipynb' and trashed = false"
            resp = service.files().list(
                q=q,
                pageSize=200,
                fields="nextPageToken, files(id, name, size, modifiedTime, parents, mimeType)",
                pageToken=page_token,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            ).execute()
            for f in resp.get("files", []):
                by_id[f["id"]] = f
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    return list(by_id.values())


# ---- parent-path resolver -----------------------------------------------
def make_path_resolver(service):
    cache = {}

    def resolve(file_id):
        if file_id in cache:
            return cache[file_id]
        try:
            f = service.files().get(
                fileId=file_id,
                fields="id, name, parents",
                supportsAllDrives=True,
            ).execute()
        except Exception:
            cache[file_id] = "?"
            return "?"
        parents = f.get("parents", [])
        if not parents:
            cache[file_id] = f["name"]
            return f["name"]
        parent_path = resolve(parents[0])
        full = f"{parent_path}/{f['name']}"
        cache[file_id] = full
        return full

    return resolve


# ---- download ------------------------------------------------------------
def download_one(service, file_id: str, dest: Path):
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with io.FileIO(str(dest), "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=2 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()


# ---- main ----------------------------------------------------------------
def main():
    svc = get_service()
    print("Searching Drive for matching .ipynb files…")
    matches = list_matches(svc)
    if not matches:
        print("No matches found.")
        return

    resolve = make_path_resolver(svc)

    enriched = []
    for f in matches:
        parents = f.get("parents", [])
        parent_path = resolve(parents[0]) if parents else ""
        rel = f"{parent_path}/{f['name']}" if parent_path else f["name"]
        # Exclude obvious third-party tutorial paths
        if any(tok in rel for tok in EXCLUDE_PATH_TOKENS):
            continue
        size = int(f["size"]) if f.get("size") else None
        enriched.append({
            "id": f["id"],
            "name": f["name"],
            "size_kb": round(size / 1024, 1) if size is not None else None,
            "mtime": f.get("modifiedTime", "")[:16].replace("T", " "),
            "rel_path": rel,
        })
    enriched.sort(key=lambda x: (x["mtime"], x["rel_path"]), reverse=True)

    print(f"\nFound {len(enriched)} candidates after exclusions.\n")
    print(f"{'mtime':<18} {'KB':>9}  rel-path")
    print("-" * 110)
    for e in enriched:
        kb = f"{e['size_kb']:>9.1f}" if e['size_kb'] is not None else "      n/a"
        print(f"{e['mtime']:<18} {kb}  {e['rel_path']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_out = OUT_DIR / "_drive_matches.json"
    meta_out.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMetadata saved to: {meta_out}")
    print(f"Download target:   {OUT_DIR}")

    ans = input(f"\nDownload all {len(enriched)} files? [y/N] ").strip().lower()
    if ans != "y":
        print("Skipped download. Re-run and answer y to fetch.")
        return

    ok, fail, skipped = 0, 0, 0
    for e in enriched:
        # Flatten path into filename (so we don't have to mkdir trees)
        safe = re.sub(r"[<>:\"/\\|?*\n\r]", "_", e["rel_path"]).replace("/", "__")
        dest = OUT_DIR / safe
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        print(f"  ↓ {safe}")
        try:
            download_one(svc, e["id"], dest)
            ok += 1
        except Exception as ex:
            print(f"    FAILED: {ex}")
            fail += 1

    print(f"\nDone. ok={ok}  skipped={skipped}  fail={fail}")
    print(f"Files in: {OUT_DIR}")
    print("\nNext: tell me 'organize the drive pull' and I'll run the same dedupe + categorize")
    print("pipeline on the new files and slot them into notebooks/<subfolder>/.")


if __name__ == "__main__":
    main()
