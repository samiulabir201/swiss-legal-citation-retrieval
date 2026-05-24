"""Apply Channel 16 (multi-vector concept-cosine) patch to the v7.5 canonical notebook.
Idempotent — bails if patch markers are already present."""
import json
from pathlib import Path
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

NB_PATH = Path(
    "notebooks/02_v75_multiquery_canonical_pool/"
    "pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb"
)

# ---------- load ----------
nb = json.load(open(NB_PATH, encoding="utf-8"))

def get_src(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

def set_src(cell, s):
    cell["source"] = s.splitlines(keepends=True)

# ---------- idempotency check ----------
all_text = "".join(get_src(c) for c in nb["cells"])
if "Phase 8.5: Concept-cosine channel" in all_text:
    print("Patch already applied (Phase 8.5 marker found). Aborting.")
    sys.exit(0)

# ---------- (A) Cell 8: CONFIG additions ----------
cell8_src = get_src(nb["cells"][8])

# A1: add budget after `budget_vector_hyde`
budget_anchor = '    "budget_vector_hyde":    2000,   # HyDE — hypothetical-answer vector\n'
budget_replacement = (
    budget_anchor
    + '    "budget_concept_cosine": 2000,   # v7.5 channel 16 (multi-vector aspect cosine)\n'
)
assert budget_anchor in cell8_src, "[CELL 8] budget_anchor not found"
cell8_src = cell8_src.replace(budget_anchor, budget_replacement, 1)

# A2: add channel weight after `vector_hyde`
weight_anchor = '        "vector_hyde":       1.5,   # HyDE — strongest vector signal expected\n'
weight_replacement = (
    weight_anchor
    + '        "concept_cosine":    1.5,   # v7.5 channel 16 (multi-vector aspect cosine)\n'
)
assert weight_anchor in cell8_src, "[CELL 8] weight_anchor not found"
cell8_src = cell8_src.replace(weight_anchor, weight_replacement, 1)

# A3: append to guarantee_channels
guarantee_anchor = (
    '        "term_orig",             # v7.5 NEW\n'
    '    ],'
)
guarantee_replacement = (
    '        "term_orig",             # v7.5 NEW\n'
    '        "concept_cosine",        # v7.5 channel 16\n'
    '    ],'
)
assert guarantee_anchor in cell8_src, "[CELL 8] guarantee_anchor not found"
cell8_src = cell8_src.replace(guarantee_anchor, guarantee_replacement, 1)

set_src(nb["cells"][8], cell8_src)
print("[CELL 8] CONFIG patched")

# ---------- (B) Insert Phase 8.5 cells after Cell 34 ----------
phase85_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Phase 8.5 — Concept-cosine channel (multi-vector, channel 16)\n",
        "\n",
        "Loads the multi-vector aspect-cosine channel produced by "
        "`v75_channel_16_aspect_cosine.ipynb`. Resolves citation strings → v7.5 doc IDs "
        "via `cit_to_doc_ids` so the channel plugs into RRF fusion alongside the other 15.\n",
    ],
}

phase85_code_src = (
    '# Phase 8.5: Concept-cosine channel (multi-vector, max-pool over aspects)\n'
    'import json as _json_85\n'
    'from pathlib import Path as _Path_85\n'
    'import numpy as _np_85\n'
    '\n'
    '_CONCEPT_MULTI_DIR = _Path_85("/content/drive/MyDrive/swiss_law/research/concept_embedding_path/multi_aspect")\n'
    '_CHANNEL_JSON      = _CONCEPT_MULTI_DIR / "v75_concept_cosine_channel_multi.json"\n'
    '\n'
    'assert "cit_to_doc_ids" in globals(), "Run v7.5 Phase 2 first (builds cit_to_doc_ids)"\n'
    'assert _CHANNEL_JSON.exists(), f"Missing: {_CHANNEL_JSON} — run v75_channel_16_aspect_cosine.ipynb first"\n'
    '\n'
    'with open(_CHANNEL_JSON, encoding="utf-8") as _f:\n'
    '    _channel_cit = _json_85.load(_f)\n'
    '\n'
    'CONCEPT_COSINE_CHANNEL     = {}   # qid -> list[(did, score)]\n'
    'CONCEPT_COSINE_BEST_ASPECT = {}   # qid -> dict[did -> aspect_id]\n'
    'for _qid, _triples in _channel_cit.items():\n'
    '    _best, _aspects = {}, {}\n'
    '    for (_cit, _score, _best_aspect) in _triples:\n'
    '        for _did in cit_to_doc_ids.get(_cit, []):\n'
    '            if _did not in _best or _score > _best[_did]:\n'
    '                _best[_did]    = _score\n'
    '                _aspects[_did] = _best_aspect\n'
    '    CONCEPT_COSINE_CHANNEL[_qid]     = sorted(_best.items(), key=lambda kv: (-kv[1], kv[0]))\n'
    '    CONCEPT_COSINE_BEST_ASPECT[_qid] = _aspects\n'
    '\n'
    'print(f"Channel 16 loaded: {len(CONCEPT_COSINE_CHANNEL)} queries, "\n'
    '      f"avg candidates/query = {_np_85.mean([len(v) for v in CONCEPT_COSINE_CHANNEL.values()]):.0f}")\n'
    'for _qid in sorted(CONCEPT_COSINE_CHANNEL.keys()):\n'
    '    print(f"  {_qid}: {len(CONCEPT_COSINE_CHANNEL[_qid])} candidates")\n'
    '\n'
    'def concept_cosine_for(qid, budget=None):\n'
    '    hits = CONCEPT_COSINE_CHANNEL.get(qid, [])\n'
    '    return hits if budget is None else hits[:budget]\n'
)
phase85_code = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": phase85_code_src.splitlines(keepends=True),
}

# Insert at index 35 (right after Cell 34: run_channels def, before Cell 35: Phase 9 header)
nb["cells"].insert(35, phase85_md)
nb["cells"].insert(36, phase85_code)
print("[PHASE 8.5] 2 cells inserted at index 35-36")

# ---------- (C) Master-loop patch (originally Cell 36, now Cell 38) ----------
master_idx = 38
cell_ml_src = get_src(nb["cells"][master_idx])

ml_anchor = (
    '    ch_out = run_channels(q["query_text"], ALL_TARGETS[qid],\n'
    '                          ALL_Q_EMB_RAW[qid], ALL_Q_EMB_ENRICHED[qid],\n'
    '                          ALL_Q_EMB_HYDE_LIST[qid],\n'
    '                          g_set, g_tot, verbose=True)\n'
    '    fuse_out = fuse_for_query(ch_out["channels"], g_set, g_tot)'
)
ml_replacement = (
    '    ch_out = run_channels(q["query_text"], ALL_TARGETS[qid],\n'
    '                          ALL_Q_EMB_RAW[qid], ALL_Q_EMB_ENRICHED[qid],\n'
    '                          ALL_Q_EMB_HYDE_LIST[qid],\n'
    '                          g_set, g_tot, verbose=True)\n'
    '    # ---- channel 16: concept-cosine (multi-vector, aspect max-pool) ----\n'
    '    _cc_budget = CONFIG.get("budget_concept_cosine", 2000)\n'
    '    _cc_hits   = concept_cosine_for(qid, _cc_budget)\n'
    '    ch_out["channels"].append(("concept_cosine", _cc_hits))\n'
    '    _cc_did_set = {d for d, _ in _cc_hits}\n'
    '    ch_out.setdefault("channel_recalls", {})["concept_cosine"] = (\n'
    '        len(_cc_did_set & g_set) / max(1, g_tot)\n'
    '    )\n'
    '    fuse_out = fuse_for_query(ch_out["channels"], g_set, g_tot)'
)
assert ml_anchor in cell_ml_src, f"[CELL {master_idx}] master loop anchor not found"
cell_ml_src = cell_ml_src.replace(ml_anchor, ml_replacement, 1)
set_src(nb["cells"][master_idx], cell_ml_src)
print(f"[CELL {master_idx}] master loop patched (6 lines inserted)")

# ---------- save ----------
json.dump(nb, open(NB_PATH, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"\nSaved → {NB_PATH}")
print(f"Total cells: {len(nb['cells'])}")
