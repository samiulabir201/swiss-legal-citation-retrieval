"""Filter the 100-family scan to ONLY families not already placed under notebooks/<sub>/.
   Output a compact-decision report with: family name, winner path, mtime, n_outputs, first 200 chars
   of first markdown + 300 chars of first code cell. That's enough to classify by content."""
import json, re
from pathlib import Path
from datetime import datetime

ROOT = Path(r"e:\swiss_citation_extraction")
ORGANIZED = ROOT / "notebooks"
INV = ORGANIZED / "_inventory"
DOWNLOADS = Path(r"C:\Users\samiul\Downloads")

NON_SWISS_RE = re.compile(r"aimo|retroviral|cross-lineage|rt-prediction|44-50|data-leakage|submission-a|submission-b|Rt Prime Editing", re.I)


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


# Already-placed family names — derive from current canonical filenames in subfolders
# We'll match by the original-source family used to populate them via the manifest
mf = INV / "_reorg_manifest.json"
placed_families = set()
if mf.exists():
    m = json.loads(mf.read_text(encoding="utf-8"))
    for k in m.get("kept", []):
        placed_families.add(k["family"])

# Also placed via recovery: those went into canonical files; build family-from-source for them too
if mf.exists():
    for r in m.get("recovered", []):
        if "src" in r:
            placed_families.add(family(Path(r["src"]).name))


def peek(p: Path) -> dict:
    try:
        nb = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        return {"error": str(e)}
    n_code_tot = 0
    n_code_w_out = 0
    n_md = 0
    md_first = ""
    code_first = ""
    imports = []
    for c in nb.get("cells", []):
        ct = c.get("cell_type")
        if ct == "markdown":
            n_md += 1
            if not md_first:
                src = c.get("source", [])
                txt = "".join(src) if isinstance(src, list) else str(src)
                md_first = txt[:300]
        elif ct == "code":
            n_code_tot += 1
            if c.get("outputs"):
                n_code_w_out += 1
            src = c.get("source", [])
            txt = "".join(src) if isinstance(src, list) else str(src)
            if not code_first and txt.strip():
                code_first = txt.strip()[:500]
            for line in txt.splitlines():
                ls = line.strip()
                if re.match(r"^(from\s+\S+\s+import|import\s+\S+)", ls):
                    imports.append(ls)
    return {
        "n_code_total": n_code_tot,
        "n_code_with_output": n_code_w_out,
        "n_md": n_md,
        "first_md_chunk": md_first,
        "first_code_chunk": code_first,
        "imports": sorted(set(imports))[:15],
    }


# Gather all candidates, group by family, drop families already placed
candidates = []
for d in [ROOT, DOWNLOADS]:
    if d == ROOT:
        for p in d.rglob("*.ipynb"):
            sp = str(p).lower()
            if "research_repos" in sp:
                continue
            if "\\notebooks\\" in sp and re.search(r"\\notebooks\\\d{2}_", sp):
                continue
            if "\\.ipynb_checkpoints\\" in sp:
                continue
            if NON_SWISS_RE.search(p.name):
                continue
            candidates.append(p)
    else:
        for p in d.glob("*.ipynb"):
            if NON_SWISS_RE.search(p.name):
                continue
            candidates.append(p)

fams = {}
for p in candidates:
    f = family(p.name)
    fams.setdefault(f, []).append(p)

new_families = [(f, ps) for f, ps in fams.items() if f not in placed_families]
print(f"Placed families: {len(placed_families)}")
print(f"Total families: {len(fams)}")
print(f"New (unplaced) families: {len(new_families)}")

# For each new family pick winner: max(n_code_with_output) then max(mtime)
winners = []
for f, ps in new_families:
    best = None
    for p in ps:
        info = peek(p)
        score = (info.get("n_code_with_output", 0), p.stat().st_mtime)
        rec = {"path": p, "info": info, "score": score}
        if best is None or rec["score"] > best["score"]:
            best = rec
    winners.append({"family": f, "best": best, "n_in_family": len(ps)})

winners.sort(key=lambda w: w["best"]["path"].stat().st_mtime, reverse=True)

# Write compact report
lines = []
lines.append(f"# New family decisions\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
lines.append(f"Total new families: {len(winners)}\n\n")
for w in winners:
    p = w["best"]["path"]
    info = w["best"]["info"]
    mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    lines.append(f"\n---\n\n## `{w['family']}`  ({w['n_in_family']} copies)\n")
    lines.append(f"- **Winner:** `{p}`\n")
    lines.append(f"- **Mtime:** {mtime}  |  **Code cells:** {info.get('n_code_total','?')}  |  **With output:** {info.get('n_code_with_output','?')}  |  **MD cells:** {info.get('n_md','?')}\n")
    if info.get("first_md_chunk"):
        clean_md = info["first_md_chunk"].replace("\n", " ").replace("`", "'")
        lines.append(f"- **First markdown:** `{clean_md[:200]}`\n")
    if info.get("first_code_chunk"):
        lines.append(f"- **First code cell:**\n```python\n{info['first_code_chunk'][:500]}\n```\n")
    if info.get("imports"):
        lines.append(f"- **Imports:** {info['imports']}\n")

(INV / "_new_families.md").write_text("".join(lines), encoding="utf-8")
print(f"Wrote {INV / '_new_families.md'}")
