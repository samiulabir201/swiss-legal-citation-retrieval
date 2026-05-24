# drive_pull

Pulls Swiss-citation / Omnilex `.ipynb` files from your Google Drive into the repo.

Runs locally on Windows. No password. OAuth handles auth on first run via your browser.

## TL;DR

```powershell
cd e:\swiss_citation_extraction\scripts\drive_pull
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
# Get credentials.json from Google Cloud Console (see setup below), put it here.
python drive_pull.py
```

## One-time setup (3-5 min, done once)

1. **Create a Google Cloud project**
   <https://console.cloud.google.com/projectcreate> → name it anything → Create.

2. **Enable the Drive API**
   <https://console.cloud.google.com/apis/library/drive.googleapis.com> → Enable.

3. **Configure the OAuth consent screen**
   <https://console.cloud.google.com/apis/credentials/consent>
   - User type: **External** → Create
   - App name: `swiss_drive_pull` (any name)
   - User support email + Developer contact: your Gmail
   - Save → through "Scopes" (no edits)
   - **Test users** → Add Users → add your Gmail (the account whose Drive you'll read)
   - Save → Back to Dashboard

4. **Create the OAuth client**
   <https://console.cloud.google.com/apis/credentials>
   - Create Credentials → OAuth client ID
   - Application type: **Desktop app**
   - Name: anything → Create
   - Click **DOWNLOAD JSON** in the popup

5. **Place the file**
   Save the downloaded JSON as `credentials.json` in this folder
   (`scripts/drive_pull/credentials.json`).

## What the script does

1. First run: opens a browser for OAuth consent → saves `token.json` here (gitignored).
2. Searches your Drive for `*.ipynb` whose name contains any of:
   `swiss`, `citation`, `omnilex`, `anchor_funnel`, `court_citation`, `court_llm`,
   `enrich_laws_de`, `enrich_cards`, `legalmalr`, `law_pipeline`, `qwen3_8b_awq`.
3. Excludes third-party paths (`research_repos`, `FlagEmbedding`, etc.).
4. Prints the match list with size + modified-time + Drive path.
5. Saves metadata to `notebooks/_drive_pull/_drive_matches.json`.
6. Asks `Download all N files? [y/N]` — answer `y` to fetch.
7. Files land in `e:\swiss_citation_extraction\notebooks\_drive_pull\` with flattened
   names like `swiss_law__research__anchor_funnel_val001_v7_4.ipynb`.

## After it finishes

Tell me **"organize the drive pull"** and I'll run the same dedupe + content-classify
pipeline that produced the existing 90 organized notebooks and slot the new ones into
the correct `notebooks/<subfolder>/`.

## Files in this folder

- `drive_pull.py` — the script
- `credentials.json` — your OAuth client (you provide, gitignored)
- `token.json` — written after first auth, do not share (gitignored)
- `.gitignore` — keeps secrets out of git
- `README.md` — this file
