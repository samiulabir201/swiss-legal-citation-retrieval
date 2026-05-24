"""Rename the inventory .md files to match the cleaned .ipynb basenames."""
import json, re
from pathlib import Path

INV = Path(r"e:\swiss_citation_extraction\notebooks\_inventory")
manifest = json.loads((INV / "_dedupe_manifest.json").read_text(encoding="utf-8"))


def safe_basename_md(name: str) -> str:
    base = name[:-6] if name.endswith(".ipynb") else name
    base = base.replace(" ", "_").replace("(", "_").replace(")", "_")
    base = re.sub(r"_+", "_", base).strip("_")
    return base + ".md"


renames = []
for entry in manifest["kept"]:
    src_name = Path(entry["src"]).name
    target_name = Path(entry["target"]).name  # cleaned .ipynb name
    md_src = INV / safe_basename_md(src_name)
    md_target = INV / (target_name[:-6] + ".md")  # .ipynb -> .md
    if md_src == md_target:
        continue
    if md_src.exists() and not md_target.exists():
        md_src.rename(md_target)
        renames.append({"from": md_src.name, "to": md_target.name})
    elif md_target.exists():
        renames.append({"from": md_src.name if md_src.exists() else "?", "to": md_target.name, "note": "target already exists"})
    else:
        renames.append({"from": md_src.name, "to": md_target.name, "note": "src missing"})

for r in renames:
    print(r)
print(f"\nTotal renames: {sum(1 for r in renames if 'note' not in r)}")
