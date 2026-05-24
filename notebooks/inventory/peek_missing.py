"""Find every .ipynb in the repo + Downloads that is not yet in notebooks/<subfolder>/.
   Write a concise content peek (first 1500 chars of first 4 code cells + import lines)
   per missing-family winner. Helps with content-based classification.
"""
import json, re
from pathlib import Path
from datetime import datetime

ROOT = Path(r"e:\swiss_citation_extraction")
ORGANIZED = ROOT / "notebooks"
INV = ORGANIZED / "_inventory"
DOWNLOADS = Path(r"C:\Users\samiul\Downloads")

# Names indicating notebooks NOT swiss-citation related — skip these
NON_SWISS_PATTERNS = [
    r"aimo", r"AIMO", r"retroviral", r"cross-lineage", r"rt-prediction",
    r"44-50", r"data-leakage", r"submission-a", r"submission-b", r"Rt Prime Editing",
]
NON_SWISS_RE = re.compile("|".join(NON_SWISS_PATTERNS), re.I)


def is_swiss(name: str) -> bool:
    if NON_SWISS_RE.search(name):
        return False
    return True


def family(name: str) -> str:
    b = name[:-6] if name.endswith(".ipynb") else name
    while True:
        new = re.sub(r"[ _]?\(\d+\)$", "", b)
        if new == b:
            break
        b = new
    b = b.replace(" ", "_").lower()
    b = re.sub(r"_+", "_", b).strip("_")
    return b


# Already-placed families: walk the subfolders and collect basenames
already_placed_basenames = set()
already_placed_families = set()
for sub in ORGANIZED.iterdir():
    if sub.is_dir() and sub.name[:3] in [f"{n:02d}_" for n in range(1, 20)]:
        for nb in sub.glob("*.ipynb"):
            already_placed_basenames.add(nb.stem)


def get_imports_and_first_cells(nb_path: Path, n_code: int = 4, char_budget: int = 1500) -> dict:
    """Return imports + first N code-cell source (concatenated, truncated)."""
    try:
        nb = json.loads(nb_path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        return {"error": str(e)}

    code_cells_src = []
    imports = []
    n_total = 0
    n_with_output = 0
    n_md = 0
    md_first = []
    for c in nb.get("cells", []):
        ct = c.get("cell_type")
        if ct == "markdown":
            n_md += 1
            if len(md_first) < 2:
                txt = "".join(c.get("source", [])) if isinstance(c.get("source"), list) else str(c.get("source", ""))
                md_first.append(txt[:300])
            continue
        if ct != "code":
            continue
        n_total += 1
        src = c.get("source", [])
        src = "".join(src) if isinstance(src, list) else str(src)
        if c.get("outputs"):
            n_with_output += 1
        # Pick up imports from anywhere
        for line in src.splitlines():
            if re.match(r"^\s*(from\s+\S+\s+import|import\s+\S+)", line):
                imports.append(line.strip())
        if len(code_cells_src) < n_code and src.strip():
            code_cells_src.append(src.strip()[:char_budget // n_code])

    return {
        "n_code_total": n_total,
        "n_code_with_output": n_with_output,
        "n_md": n_md,
        "imports_uniq": sorted(set(imports))[:25],
        "first_md_cells": md_first,
        "first_code_cells": code_cells_src,
    }


# Gather all .ipynb candidates
candidates = []
for d in [ROOT, DOWNLOADS]:
    if d == ROOT:
        for p in d.rglob("*.ipynb"):
            # Exclude already-organized, third-party, hidden, etc.
            sp = str(p).lower()
            if "research_repos" in sp:
                continue
            if "\\notebooks\\" in sp and re.search(r"\\notebooks\\\d{2}_", sp):
                continue
            if "\\.ipynb_checkpoints\\" in sp:
                continue
            if not is_swiss(p.name):
                continue
            candidates.append(p)
    else:
        for p in d.glob("*.ipynb"):
            if not is_swiss(p.name):
                continue
            candidates.append(p)

# Group by family
fams = {}
for p in candidates:
    f = family(p.name)
    fams.setdefault(f, []).append(p)

# For each family, pick winner = max(code_with_output) then max(mtime)
winners = []
for f, ps in fams.items():
    enriched = []
    for p in ps:
        info = get_imports_and_first_cells(p, n_code=1, char_budget=400)  # tiny peek for ranking
        if "error" in info:
            n_out = 0
        else:
            n_out = info["n_code_with_output"]
        enriched.append({"path": p, "n_with_out": n_out, "mtime": p.stat().st_mtime})
    enriched.sort(key=lambda x: (x["n_with_out"], x["mtime"]), reverse=True)
    w = enriched[0]
    winners.append({"family": f, "winner_path": w["path"], "n_candidates_in_family": len(ps),
                    "winner_n_with_out": w["n_with_out"]})

# Filter to only families NOT in already-placed
def is_placed(fam: str) -> bool:
    # Build a normalized version of placed basenames and check if family matches any
    fam_keywords = fam.split("_")
    for placed in already_placed_basenames:
        plc = placed.lower()
        # If most fam keywords appear in placed basename, treat as placed
        # Use exact substring match on a few key tokens
        if fam in plc.replace("_", "_"):
            return True
    return False

# More robust: rebuild family from already-placed canonical names, but those are arbitrary
# Just produce the list of all families and let me categorize
report_lines = []
report_lines.append(f"# Repo-wide notebook scan\n")
report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
report_lines.append(f"Candidates scanned: {len(candidates)}\n")
report_lines.append(f"Families found: {len(fams)}\n")
report_lines.append(f"Already-placed basenames (in notebooks/<sub>/): {len(already_placed_basenames)}\n\n")

# Sort by mtime (newest first) using the winner's mtime
winners.sort(key=lambda w: w["winner_path"].stat().st_mtime, reverse=True)

for w in winners:
    p = w["winner_path"]
    info = get_imports_and_first_cells(p)
    mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    report_lines.append(f"\n## FAMILY: `{w['family']}`\n")
    report_lines.append(f"- Winner path: `{p}`\n")
    report_lines.append(f"- Mtime: {mtime}\n")
    report_lines.append(f"- Candidates in family: {w['n_candidates_in_family']}\n")
    report_lines.append(f"- Total code cells: {info.get('n_code_total', '?')}; with output: {info.get('n_code_with_output', '?')}; markdown cells: {info.get('n_md', '?')}\n")
    if info.get("imports_uniq"):
        report_lines.append(f"- Imports: {info['imports_uniq']}\n")
    if info.get("first_md_cells"):
        report_lines.append(f"- First markdown:\n")
        for m in info["first_md_cells"]:
            report_lines.append(f"  > {m[:200]}\n")
    if info.get("first_code_cells"):
        report_lines.append(f"- First code cells (truncated):\n")
        for i, c in enumerate(info["first_code_cells"], 1):
            report_lines.append(f"  ```python\n  # cell {i}\n  {c[:350]}\n  ```\n")

(INV / "_repo_scan.md").write_text("".join(report_lines), encoding="utf-8")
print(f"Wrote {INV / '_repo_scan.md'}")
print(f"Candidates: {len(candidates)}, Families: {len(fams)}")
