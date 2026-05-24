"""Convert concept_embedding_path_experiment.py (jupytext percent format)
to a Colab-ready .ipynb. Run once."""
import json, re
from pathlib import Path

SRC = Path("e:/swiss_citation_extraction/notebooks/stage_b_grounded_llm.py")
DST = Path("e:/swiss_citation_extraction/notebooks/stage_b_grounded_llm.ipynb")

text = SRC.read_text(encoding="utf-8")

# Split on percent-cell delimiters; preserve markdown vs code
# Pattern: "# %% [markdown]" or "# %%"
parts = re.split(r"\n(?=# %%(?: \[markdown\])?(?:\n|$))", text)

cells = []
for p in parts:
    if not p.strip():
        continue
    lines = p.splitlines(keepends=True)
    header = lines[0].rstrip("\n")
    body_lines = lines[1:]

    if header.startswith("# %% [markdown]"):
        # Markdown cells: body lines start with '#' — strip the leading '# '
        md_lines = []
        for ln in body_lines:
            if ln.startswith("# "):
                md_lines.append(ln[2:])
            elif ln.rstrip() == "#":
                md_lines.append("\n")
            else:
                md_lines.append(ln)
        # Strip jupytext frontmatter cell if present
        joined = "".join(md_lines).rstrip() + "\n"
        if joined.strip().startswith("---") and "jupytext:" in joined:
            continue
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": joined.splitlines(keepends=True) or [""],
        })
    elif header.startswith("# %%"):
        code = "".join(body_lines).rstrip("\n")
        if not code.strip():
            continue
        cells.append({
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": (code + "\n").splitlines(keepends=True),
        })

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "name": "python", "version": "3.11",
            "mimetype": "text/x-python", "file_extension": ".py",
            "pygments_lexer": "ipython3",
        },
        "colab": {"provenance": [], "machine_shape": "hm", "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {DST}  ({len(cells)} cells, {DST.stat().st_size:,} bytes)")
