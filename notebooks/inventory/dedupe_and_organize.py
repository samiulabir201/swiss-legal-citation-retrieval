"""Dedupe notebooks by name family and copy the survivors into notebooks/.

Family rule:
- Strip trailing copy-marker patterns like ` (1)`, `_(1)`, ` (1).ipynb`, `_1` (but only when at the end and preceded by a non-digit).
- Group by remaining basename (lowercased, spaces → underscores).
- Keep the file with the latest mtime in each group.
- Also delete the corresponding _inventory .md files for dropped notebooks.
"""
import re, shutil, json
from pathlib import Path
from datetime import datetime

LIST = Path(r"e:\swiss_citation_extraction\notebooks\_inventory\notebook_list.txt")
NB_OUT = Path(r"e:\swiss_citation_extraction\notebooks")
INV = Path(r"e:\swiss_citation_extraction\notebooks\_inventory")

paths = [Path(p) for p in LIST.read_text(encoding="utf-8").splitlines() if p.strip() and Path(p).exists()]

def safe_basename_md(name: str) -> str:
    base = name[:-6] if name.endswith(".ipynb") else name
    base = base.replace(" ", "_").replace("(", "_").replace(")", "_")
    base = re.sub(r"_+", "_", base).strip("_")
    return base + ".md"


def family(name: str) -> str:
    """Strip trailing copy-marker patterns to identify a name family."""
    b = name[:-6] if name.endswith(".ipynb") else name
    # Strip patterns: ` (1)`, `_(1)`, `(1)`, `_1` (digit suffixes that look like Drive copy markers)
    while True:
        new = re.sub(r"[ _]?\(\d+\)$", "", b)
        if new == b:
            break
        b = new
    # Normalize spaces and parens
    b = b.replace(" ", "_").lower()
    b = re.sub(r"_+", "_", b).strip("_")
    return b


# Group by family; keep newest mtime
groups: dict[str, list[Path]] = {}
for p in paths:
    f = family(p.name)
    groups.setdefault(f, []).append(p)

kept: list[tuple[str, Path]] = []
dropped: list[tuple[str, Path, Path]] = []  # (family, dropped path, kept-instead path)

for fam, ps in groups.items():
    ps_sorted = sorted(ps, key=lambda p: p.stat().st_mtime, reverse=True)
    keep = ps_sorted[0]
    kept.append((fam, keep))
    for d in ps_sorted[1:]:
        dropped.append((fam, d, keep))


# Choose final filename for each kept notebook
# Prefer the keep file's existing name but strip the dedup markers for cleanliness
def clean_target_name(name: str) -> str:
    b = name[:-6] if name.endswith(".ipynb") else name
    # Strip dedup markers
    while True:
        new = re.sub(r"[ _]?\(\d+\)$", "", b)
        if new == b:
            break
        b = new
    b = b.replace(" ", "_")
    b = re.sub(r"_+", "_", b).strip("_")
    return b + ".ipynb"


# Copy survivors into notebooks/
NB_OUT.mkdir(parents=True, exist_ok=True)
copy_log = []
for fam, src in kept:
    target_name = clean_target_name(src.name)
    target = NB_OUT / target_name
    # Don't re-copy a file from notebooks/ to itself
    if src.resolve() == target.resolve():
        copy_log.append({"family": fam, "src": str(src), "target": str(target), "action": "in_place"})
        continue
    # If target exists with same content, skip
    if target.exists() and target.stat().st_mtime >= src.stat().st_mtime and target.stat().st_size == src.stat().st_size:
        copy_log.append({"family": fam, "src": str(src), "target": str(target), "action": "already_present"})
        continue
    try:
        shutil.copy2(str(src), str(target))
        copy_log.append({"family": fam, "src": str(src), "target": str(target), "action": "copied"})
    except Exception as e:
        copy_log.append({"family": fam, "src": str(src), "target": str(target), "action": f"FAILED: {e}"})


# Delete _inventory .md files for dropped notebooks (only if there's a kept md with same family)
inv_dropped = []
for fam, drop_src, kept_src in dropped:
    drop_md = INV / safe_basename_md(drop_src.name)
    kept_md = INV / safe_basename_md(kept_src.name)
    if drop_md.exists() and drop_md.resolve() != kept_md.resolve():
        try:
            drop_md.unlink()
            inv_dropped.append(str(drop_md.name))
        except Exception as e:
            inv_dropped.append(f"FAILED to delete {drop_md.name}: {e}")


# Also delete _inventory files for sources that we're keeping but had multiple writes (collision)
# Actually leave them — the kept file's md is what we want.

# Write a manifest of what was kept and dropped
manifest = {
    "kept": [
        {"family": f, "src": str(p), "target": str(NB_OUT / clean_target_name(p.name)), "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")}
        for f, p in sorted(kept, key=lambda x: x[0])
    ],
    "dropped": [
        {"family": f, "dropped": str(d), "kept_instead": str(k), "drop_mtime": datetime.fromtimestamp(d.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if d.exists() else "?"}
        for f, d, k in sorted(dropped, key=lambda x: x[0])
    ],
    "copy_log": copy_log,
    "inv_md_deleted": inv_dropped,
}
(INV / "_dedupe_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Families: {len(groups)}")
print(f"Kept: {len(kept)}")
print(f"Dropped: {len(dropped)}")
print(f"Copies: {sum(1 for c in copy_log if c['action']=='copied')}")
print(f"Already present: {sum(1 for c in copy_log if c['action']=='already_present')}")
print(f"In place (notebook already in notebooks/): {sum(1 for c in copy_log if c['action']=='in_place')}")
print(f"Inventory md deleted: {len(inv_dropped)}")
print()
print("=== Kept (one per family) ===")
for f, p in sorted(kept, key=lambda x: x[0]):
    print(f"  [{f}] -> {clean_target_name(p.name)}  ({datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M')})")
print()
print("=== Dropped (older duplicates) ===")
for f, d, k in sorted(dropped, key=lambda x: x[0]):
    print(f"  [{f}] {d.name}  (kept: {k.name})")
