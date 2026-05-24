"""Update enrich_laws_de_qwen3_8b_kaggle.ipynb to re-run only the 1829
EngineDead records.

Changes:
- Cell 3 (Config): point `input_jsonl` and `fallback_input_jsonl` at the rerun
  JSONL.
- Cell 3 (suffix): change the output filename suffix from
  `<start>_<end>` / `<start>_all` to a fixed `engine_dead_rerun` suffix so the
  rerun outputs don't overwrite the existing 173k file.

The body of every other cell is left untouched, so the LLM/vLLM logic, schema,
prompt, parsing, and checkpoint-resume all behave the same — they just
operate on the smaller input.
"""
import json

NB = r"E:\swiss_citation_extraction\law_json_llm_output\enrich_laws_de_qwen3_8b_kaggle.ipynb"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell = nb["cells"][3]  # Config cell
assert cell["cell_type"] == "code"
src = "".join(cell["source"])
assert "input_jsonl" in src and "fallback_input_jsonl" in src and "suffix =" in src

# 1) Swap input file paths.
old_input = "    input_jsonl: str = str(DATA_DIR / 'law_llm_input.jsonl')\n"
new_input = "    input_jsonl: str = str(DATA_DIR / 'law_llm_input_engine_dead_rerun.jsonl')\n"
assert old_input in src, "input_jsonl line not found"
src = src.replace(old_input, new_input)

old_fallback = "    fallback_input_jsonl: str = 'law_llm_input.jsonl'\n"
new_fallback = "    fallback_input_jsonl: str = 'law_llm_input_engine_dead_rerun.jsonl'\n"
assert old_fallback in src, "fallback_input_jsonl line not found"
src = src.replace(old_fallback, new_fallback)

# 2) Swap the output suffix to a fixed string so rerun results don't overwrite
#    the existing law_llm_descriptors_0000000_all.jsonl. Replace the entire
#    suffix-building block with a single fixed-suffix line.
old_suffix_block = (
    "end_idx = cfg.start + cfg.limit - 1 if cfg.limit else -1\n"
    "suffix = f'{cfg.start:07d}_{end_idx:07d}' if cfg.limit else f'{cfg.start:07d}_all'\n"
)
new_suffix_block = (
    "end_idx = cfg.start + cfg.limit - 1 if cfg.limit else -1\n"
    "suffix = 'engine_dead_rerun'\n"
)
assert old_suffix_block in src, "suffix block not found"
src = src.replace(old_suffix_block, new_suffix_block)

# Re-split into a list of lines preserving newlines (Jupyter format).
new_source_lines = src.splitlines(keepends=True)
cell["source"] = new_source_lines

# 3) Add a small helpful comment in cell 4 that resolve_input_path also tries
#    common Kaggle dataset mountpoints. (No code change — but verify the search
#    pattern picks up the new filename. The existing glob is
#    '**/law_llm_input.jsonl', which would NOT match our new file. Update it.)
cell4 = nb["cells"][4]
src4 = "".join(cell4["source"])
old_glob = "            for pat in ['**/law_llm_input.jsonl']:\n"
new_glob = "            for pat in ['**/law_llm_input_engine_dead_rerun.jsonl', '**/law_llm_input.jsonl']:\n"
if old_glob in src4:
    src4 = src4.replace(old_glob, new_glob)
    cell4["source"] = src4.splitlines(keepends=True)

# 4) Bump the failure-collection so the LLM crash retry doesn't quietly mask
#    the same failure mode again. We keep behavior identical, but if vLLM
#    again hits EngineDeadError on a rerun, surface it loudly.
# (No structural change required; the existing exception handler already
# stores the full error in llm_generation.error. Skipping.)

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Updated notebook cells:")
print(f"  cfg.input_jsonl     -> law_llm_input_engine_dead_rerun.jsonl")
print(f"  cfg.fallback_input  -> law_llm_input_engine_dead_rerun.jsonl")
print(f"  output suffix       -> engine_dead_rerun (was {{start}}_{{end}})")
print(f"  search glob updated -> matches new filename first")
print()
print("Output files will be:")
print("  law_llm_descriptors_engine_dead_rerun.jsonl")
print("  law_llm_descriptors_engine_dead_rerun_failures.jsonl")
print("  law_llm_descriptors_engine_dead_rerun_metrics.json")
print("  law_llm_descriptors_engine_dead_rerun_preview.csv")
