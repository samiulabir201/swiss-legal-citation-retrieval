"""Extract code cells + outputs from .ipynb files; write one .md per notebook.

- SKIP markdown cells (user wants code+output only).
- SKIP image outputs; replace with [image omitted].
- TRUNCATE single outputs > 1500 chars (first 800 + ...[truncated]... + last 400).
- Detect file paths used.
- Cap: if > 50 code cells, detail first 25 + last 5, summarize middle as "Cells X-Y: ...".
"""
import json, re, sys
from pathlib import Path
from datetime import datetime

OUT_DIR = Path(r"e:\swiss_citation_extraction\notebooks\_inventory")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DRIVE_RE = re.compile(r"(/content/drive/[A-Za-z0-9_./()\-]+|MyDrive/[A-Za-z0-9_./()\-]+|/content/[A-Za-z0-9_./()\-]+)")
LOCAL_PAT = re.compile(
    r"([eEcC]:[\\/][A-Za-z0-9_./\\()\-]+|"
    r"\b(?:data|artifacts|embeddings|outputs_from_363k_run|law_json_llm_output|cache_endgame|notebooks|research|data_insights)/[A-Za-z0-9_./\-]+)"
)


def truncate(s: str, lim: int = 1500) -> str:
    if len(s) <= lim:
        return s
    return s[:800] + "\n...[truncated]...\n" + s[-400:]


def extract_output(o) -> str:
    t = o.get("output_type")
    if t == "stream":
        return "".join(o.get("text", []))
    if t == "error":
        return "ERROR: " + "\n".join(o.get("traceback", []))
    if t in ("execute_result", "display_data"):
        d = o.get("data", {})
        if "text/plain" in d:
            v = d["text/plain"]
            return "".join(v) if isinstance(v, list) else str(v)
        if "image/png" in d or "image/jpeg" in d or "image/svg+xml" in d:
            return "[image omitted]"
        if "text/html" in d:
            v = d["text/html"]
            return "[html output] " + (("".join(v) if isinstance(v, list) else str(v))[:300])
    return ""


def render_cell(idx: int, src: str, outs: list[str]) -> str:
    md = f"## Step {idx}\n```python\n{src.rstrip()}\n```\n"
    if outs:
        joined = "\n---\n".join([truncate(o) for o in outs if o])
        if joined.strip():
            md += "**Output:**\n```\n" + joined + "\n```\n"
    return md


def detect_paths(text: str) -> tuple[list, list]:
    drives = set(DRIVE_RE.findall(text))
    locals_ = set(m if isinstance(m, str) else m[0] for m in LOCAL_PAT.findall(text))
    locals_ = {p for p in locals_ if not p.startswith("/content/")}
    return sorted(drives), sorted(locals_)


def process(nb_path: Path, mtime: str) -> tuple[str, dict]:
    raw = nb_path.read_text(encoding="utf-8", errors="replace")
    nb = json.loads(raw)
    code_cells = []
    all_src = []
    all_out = []
    for c in nb.get("cells", []):
        if c.get("cell_type") != "code":
            continue
        src = c.get("source", [])
        src = "".join(src) if isinstance(src, list) else str(src)
        if not src.strip():
            continue
        outs = []
        for o in c.get("outputs", []):
            txt = extract_output(o)
            if txt:
                outs.append(txt)
        code_cells.append((src, outs))
        all_src.append(src)
        all_out.append("\n".join(outs))

    drive_paths, local_paths = detect_paths("\n".join(all_src) + "\n" + "\n".join(all_out))

    n = len(code_cells)
    parts = []
    parts.append(f"# {nb_path.name}\n")
    parts.append(f"**Path:** `{nb_path}`")
    parts.append(f"**Mtime:** {mtime}")
    parts.append(f"**Total code cells:** {n}\n")

    if n <= 50:
        for i, (src, outs) in enumerate(code_cells, 1):
            parts.append(render_cell(i, src, outs))
    else:
        for i, (src, outs) in enumerate(code_cells[:25], 1):
            parts.append(render_cell(i, src, outs))
        mid_n = n - 30
        parts.append(f"\n## Cells 26..{n-5} (compressed — {mid_n} cells summarized)\n")
        mid_summary = []
        for i, (src, outs) in enumerate(code_cells[25:-5], 26):
            first_line = src.splitlines()[0][:120] if src.strip() else "(empty)"
            mid_summary.append(f"- Cell {i}: `{first_line}`")
        parts.append("\n".join(mid_summary) + "\n")
        for j, (src, outs) in enumerate(code_cells[-5:]):
            parts.append(render_cell(n - 4 + j, src, outs))

    parts.append("\n## End findings\n_(Heuristic from final cells. Review the last cells above for the canonical numbers.)_\n")
    last_outputs = [outs for _, outs in code_cells[-5:] if outs]
    final_metric_lines = []
    for outs in last_outputs:
        for o in outs:
            for line in o.splitlines():
                if re.search(r"(F1|recall|R@|macro|MRR|precision|accuracy|score|gold|hit|coverage)\b", line, re.I):
                    final_metric_lines.append(line.strip())
    final_metric_lines = final_metric_lines[:15]
    if final_metric_lines:
        parts.append("Notable lines from final outputs:\n")
        for ln in final_metric_lines:
            parts.append(f"- `{truncate(ln, 200)}`")
        parts.append("")
    else:
        parts.append("(No metric-bearing lines auto-detected in last 5 cells.)\n")

    parts.append("\n## Local files used")
    if local_paths:
        for p in local_paths:
            parts.append(f"- `{p}`")
    else:
        parts.append("- (none detected)")

    parts.append("\n## Google Drive files used")
    if drive_paths:
        for p in drive_paths:
            parts.append(f"- `{p}`")
    else:
        parts.append("- (none detected)")

    return "\n".join(parts) + "\n", {
        "name": nb_path.name,
        "path": str(nb_path),
        "n_cells": n,
        "drive_paths": drive_paths,
        "local_paths": local_paths,
        "mtime": mtime,
    }


def safe_basename(name: str) -> str:
    base = name[:-6] if name.endswith(".ipynb") else name
    base = base.replace(" ", "_").replace("(", "_").replace(")", "_")
    base = re.sub(r"_+", "_", base).strip("_")
    return base + ".md"


def main(paths: list[Path]):
    summaries = []
    for p in paths:
        if not p.exists():
            print(f"MISSING: {p}", file=sys.stderr)
            continue
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            md, info = process(p, mtime)
            out_name = safe_basename(p.name)
            (OUT_DIR / out_name).write_text(md, encoding="utf-8")
            summaries.append({"out": out_name, **info})
            print(f"OK   {p.name} -> {out_name}  ({info['n_cells']} cells)")
        except Exception as e:
            print(f"FAIL {p}: {e}", file=sys.stderr)
    (OUT_DIR / "_extract_log.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]]
    main(paths)
