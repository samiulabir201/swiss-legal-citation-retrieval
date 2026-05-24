"""Builder for swiss_citation_legalmalr_cascade_v1.ipynb.

Turns the cascade dossier (research/cascade_dossier_plan.md Phase 1) + LegalMALR-
style multi-agent decomposition + Qwen3-Reranker-8B (zero-shot) + Qwen3-8B judge
+ GRPO-optimized cascade policy into a single warm-boot notebook that consumes
the snapshot/ folder produced by the recall-0.89 anchor-funnel run.

Target: macro F1 0.6-0.7 on val (10 queries). No guarantee; design maximizes odds.

Run this script to regenerate the .ipynb in place. Cell ordering and CONFIG
defaults are checked-in here so iteration is reviewable in git.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).with_name("swiss_citation_legalmalr_cascade_v1.ipynb")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS: list[dict] = []

# =============================================================================
# Phase 0 — Title
# =============================================================================
CELLS.append(md("""# Swiss Citation — LegalMALR Cascade v1
*(Multi-Agent Query Understanding + Qwen3-Reranker-8B + Qwen3-8B LLM Judge + GRPO policy, warm-booted from the recall-0.89 v7.5 snapshot)*

**Goal.** Lift macro F1 from <0.10 (broken endgame run) to **0.6-0.7** on `val.csv` (10 queries) by
discriminating gold from the 50k-candidate fusion pool that the v7.5 anchor-funnel notebook produces.

**Inputs (warm-boot).** `snapshot/` folder from the recall-0.89 run:
- `corpus_snapshot.json.gz` — per-doc state (text, family, anchors, concepts, terms)
- `per_query_snapshot.json` — `final_topk` (50k), curves, channel recalls
- `all_targets.json` — LLM-extracted statute/concept/keyword targets per query
- `hyde_aspects.json` — decomposed query aspects
- `gold_doc_sets.json` — gold doc_id sets per val query (for scoring only)

**Stages.**
1. Cascade dossier per candidate (channel fingerprint + statute-target intersection + co-citation density) — per `research/cascade_dossier_plan.md` Phase 1.
2. Stage A: Qwen3-Reranker-8B over 50k → top-1000 with **dossier-as-document** (zero-shot, fixes law-family text-feed bug from `research/endgame_handoff_2026-05-09.md` §4.4).
3. Stage B: Qwen3-8B judge with fixed prompt (endgame §7 Move 3 patches: `enable_thinking=False`, default-NO, no "say YES" instruction, `max_new_tokens=24`).
4. Multi-agent decomposition (LegalMALR): 4 agents (statute focus, family balancer, aspect router, K predictor) parameterized by a 12-scalar policy.
5. GRPO leave-one-out across val (10 splits) — discretized policy, group relative advantage. Optuna as fallback.
6. Train (1139 queries) used **only as a sanity check** — memory flags train distribution shift as a trap.

**Hardware.** Colab Pro+ with NVIDIA RTX PRO 6000 Blackwell, 95.6 GB VRAM. Reranker (~16 GB) + judge (~16 GB) co-resident.
"""))

# =============================================================================
# Phase 1 — Setup
# =============================================================================
CELLS.append(md("""# Phase 1 — Setup
"""))

CELLS.append(code("""import sys, os, torch, platform
print(f"Python:  {sys.version.split()[0]}")
print(f"Torch:   {torch.__version__}")
print(f"CUDA:    {torch.version.cuda}")
print(f"Platform: {platform.platform()}")
if torch.cuda.is_available():
    print(f"GPU:     {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM:    {props.total_memory/1024**3:.1f} GB")
    print(f"Compute: {props.major}.{props.minor}")
else:
    print("No CUDA. The notebook will not run.")
"""))

CELLS.append(code("""# Mount Drive + auto-detect paths.
from pathlib import Path

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IN_COLAB = True
except Exception as e:
    print(f"[skip] not in Colab: {e}")
    IN_COLAB = False

CANDIDATE_ROOTS = [
    Path("/content/drive/MyDrive/swiss_law"),
    Path("/content/drive/MyDrive/swiss_citation"),
    Path("/content/drive/MyDrive/swiss-citation"),
    Path("/content/swiss_law"),
    Path("E:/swiss_citation_extraction"),   # local dev
]
DATA_ROOT = next((p for p in CANDIDATE_ROOTS if p.exists()), None)
if DATA_ROOT is None:
    raise SystemExit("Could not find DATA_ROOT. Edit CANDIDATE_ROOTS.")
print(f"DATA_ROOT = {DATA_ROOT}")

# Snapshot path — produced by the recall-0.89 v7.5 notebook (cell 48).
SNAPSHOT_DIRS = [
    DATA_ROOT / "outputs/anchor_funnel_v7_5_multiquery/snapshot",
    DATA_ROOT / "snapshot",
    Path("/content/snapshot"),
]
SNAPSHOT_DIR = next((p for p in SNAPSHOT_DIRS if p.exists() and (p / "corpus_snapshot.json.gz").exists()), None)
if SNAPSHOT_DIR is None:
    raise SystemExit(
        "Snapshot not found. Run cell 48 of swiss_citation_anchor_funnel_v7_5_multiquery_recall_0.89_at_k_50000.ipynb "
        "and place the resulting `snapshot/` folder in one of: " + str(SNAPSHOT_DIRS))
print(f"SNAPSHOT_DIR = {SNAPSHOT_DIR}")

PATHS = {
    "snapshot_dir":   SNAPSHOT_DIR,
    "val_csv":        DATA_ROOT / "data" / "val.csv",
    "train_csv":      DATA_ROOT / "data" / "train.csv",
    "test_csv":       DATA_ROOT / "data" / "test.csv",
    "out_dir":        DATA_ROOT / "outputs" / "legalmalr_cascade_v1",
    "cache_dir":      DATA_ROOT / "cache_legalmalr",
}
PATHS["out_dir"].mkdir(parents=True, exist_ok=True)
PATHS["cache_dir"].mkdir(parents=True, exist_ok=True)
for k, v in PATHS.items():
    print(f"  {k:14s} = {v}")
"""))

CELLS.append(code("""# CONFIG — single dict of all knobs. Every behavior toggle lives here.
CONFIG = {
    # ---- Stage A: reranker ----
    "rerank_pool":          50000,   # how many of the snapshot final_topk to feed the reranker
    "rerank_topn":          1000,    # how many to keep for the judge
    "rerank_batch":         32,
    "rerank_max_len":       1024,
    "rerank_model":         "Qwen/Qwen3-Reranker-8B",
    "rerank_use_dossier":   True,    # use cascade dossier as the "doc" text (vs raw corpus text)
    "rerank_dossier_chars": 1100,    # ~250 tokens per candidate
    # ---- Stage B: judge ----
    "judge_topn":           200,     # judge top-N after rerank (per query)
    "judge_model":          "Qwen/Qwen3-8B",
    "judge_max_new_tokens": 24,      # endgame §7 Move 3: tight cap; verdict only
    "judge_enable_thinking": False,  # endgame §7 Move 3: disable <think> tokens
    "judge_temperature":    0.0,
    "judge_default_on_parse_fail": "no",  # endgame §7 Move 3: NOT "yes"
    # ---- Cascade dossier ----
    "dossier_phase":        1,       # Phase 1 enrichments only; per cascade plan
    "dossier_top_for_cocite": 200,   # co-citation density: count over top-N of pool
    # ---- Multi-agent / policy ----
    "policy_optimizer":     "grpo",  # "grpo" | "optuna" | "fixed"
    "grpo_groups":          8,       # group size G
    "grpo_steps":           60,      # rollout steps per LOO fold
    "grpo_lr":              0.30,    # learning rate on log-prob updates
    "grpo_entropy_bonus":   0.02,    # exploration term
    "optuna_trials":        80,      # fallback BO trial budget per LOO fold
    "k_min":                5,       # variable-K predictor bounds (val gold range = 10-47)
    "k_max":                60,
    # ---- Eval ----
    "k_sweep":              [5,7,10,13,15,18,20,22,25,28,30,35,40,50,75,100],
    "loo_seed":             42,
    # ---- Train sanity ----
    "train_sanity_n":       100,
    "train_sanity_skip_grpo": True,  # only score with LOO-best policy
}
import json as _json
print(_json.dumps({k: v for k, v in CONFIG.items() if not isinstance(v, set)}, indent=2, default=str))
"""))

# =============================================================================
# Phase 2 — Warm-boot
# =============================================================================
CELLS.append(md("""# Phase 2 — Warm-boot from snapshot

Loads the corpus subset + per-query state produced by the recall-0.89 v7.5 run.
No retrieval re-execution. All ~3M-row indexing is skipped.

**If `channel_hit_sets` is missing from `per_query_snapshot.json`**, the cascade
dossier's channel-fingerprint signal degrades to "channels that hit at all per
query" (less informative than per-doc per-channel). To fix this permanently,
patch the recall-0.89 notebook's cell 48 to also save:
```python
"channel_hit_sets": {ch: sorted(dids) for ch, dids in _r.get("channel_hit_sets", {}).items()},
```
The dossier code below handles either case.
"""))

CELLS.append(code("""import gzip, json, time
from collections import defaultdict

_t = time.time()
with gzip.open(PATHS["snapshot_dir"] / "corpus_snapshot.json.gz", "rt", encoding="utf-8") as f:
    CORPUS = json.load(f)
print(f"  corpus_snapshot: {len(CORPUS):,} docs  ({time.time()-_t:.1f}s)")

PER_QUERY_SNAP = json.loads((PATHS["snapshot_dir"] / "per_query_snapshot.json").read_text(encoding="utf-8"))
print(f"  per_query_snap:  {len(PER_QUERY_SNAP)} queries")

ALL_TARGETS = json.loads((PATHS["snapshot_dir"] / "all_targets.json").read_text(encoding="utf-8"))
print(f"  all_targets:     {len(ALL_TARGETS)} queries")

try:
    ALL_HYDE_ASPECTS = json.loads((PATHS["snapshot_dir"] / "hyde_aspects.json").read_text(encoding="utf-8"))
    print(f"  hyde_aspects:    {len(ALL_HYDE_ASPECTS)} queries")
except FileNotFoundError:
    ALL_HYDE_ASPECTS = {}
    print(f"  hyde_aspects:    (absent — aspect routing will use targets only)")

GOLD_DOC_SETS = json.loads((PATHS["snapshot_dir"] / "gold_doc_sets.json").read_text(encoding="utf-8"))
GOLD_DOC_SETS = {qid: set(dids) for qid, dids in GOLD_DOC_SETS.items()}
print(f"  gold_doc_sets:   {sum(len(s) for s in GOLD_DOC_SETS.values())} total gold docs across {len(GOLD_DOC_SETS)} queries")

# Has per-doc per-channel info?
HAS_CHANNEL_HIT_SETS = all(
    "channel_hit_sets" in PER_QUERY_SNAP[qid] and PER_QUERY_SNAP[qid]["channel_hit_sets"]
    for qid in PER_QUERY_SNAP
)
print(f"  channel_hit_sets present per query: {HAS_CHANNEL_HIT_SETS}")
"""))

CELLS.append(code("""# Load val + train; parse gold; sanity-check that snapshot covers them.
import pandas as pd

val_df   = pd.read_csv(PATHS["val_csv"])
train_df = pd.read_csv(PATHS["train_csv"])
print(f"val.csv:   {len(val_df)} queries")
print(f"train.csv: {len(train_df)} queries")

def parse_gold(s):
    return [c.strip() for c in str(s).split(';') if c.strip()]

VAL_QUERIES = [{"query_id": r.query_id, "query": r.query, "gold": parse_gold(r.gold_citations)} for r in val_df.itertuples()]
print()
print(f"  val gold count per query: {[len(q['gold']) for q in VAL_QUERIES]}")
print(f"  snapshot gold counts:     {[len(GOLD_DOC_SETS.get(q['query_id'], [])) for q in VAL_QUERIES]}")

# Distribution audit (the train-trap reminder).
import statistics as st
train_gold_n = [len(parse_gold(g)) for g in train_df['gold_citations']]
val_gold_n   = [len(q['gold']) for q in VAL_QUERIES]
print()
print(f"  train  median gold = {st.median(train_gold_n):.1f} (mean {sum(train_gold_n)/len(train_gold_n):.1f})")
print(f"  val    median gold = {st.median(val_gold_n):.1f}  (mean {sum(val_gold_n)/len(val_gold_n):.1f})")
print(f"  ⇒ ~6x more cites per val query than train. Train is NOT representative.")
"""))

# =============================================================================
# Phase 3 — Cascade dossier
# =============================================================================
CELLS.append(md("""# Phase 3 — Cascade dossier (Phase 1 enrichments)

Per `research/cascade_dossier_plan.md`. Three signals only:
1. **Channel-of-arrival fingerprint** — which channels surfaced this doc + count.
2. **Statute-target intersection** — `doc_statute_anchors[did] ∩ statute_targets[qid]`.
3. **Co-citation density in pool** — for each law article in top-N, count how many court paragraphs in the same top-N cite it. A law cited by 40/100 top court paras is the controlling statute for this question.

No train-derived weights. Per-query LLM targets ∩ per-doc anchors. The dossier text becomes the **document field fed to the reranker** and the **evidence block fed to the judge** — both stages see pre-digested signal instead of raw German legal text.
"""))

CELLS.append(code("""from collections import Counter, defaultdict

# Helpers over the warm-booted CORPUS dict.
# Per-doc fields: ct, cit, fam, cb, pr, ln, sa (statute anchors), cn (concepts), tm (terms)

def statute_targets_for(qid):
    t = ALL_TARGETS.get(qid, {}) or {}
    out = set()
    for raw in t.get("statute_targets", []) or []:
        out.add(str(raw).strip())
    return out

def concept_targets_for(qid):
    t = ALL_TARGETS.get(qid, {}) or {}
    return {str(c).strip().lower() for c in (t.get("concept_targets_en") or []) + (t.get("legal_area_keywords") or [])}

# Per-query dossier features cache.
DOSSIER = {}   # qid -> { did -> {channels:[...], stat_hits:[...], cocite_count:int, fam, cit, text} }

for qid in PER_QUERY_SNAP:
    pq = PER_QUERY_SNAP[qid]
    final_topk = pq["final_topk"][:CONFIG["rerank_pool"]]
    pool = set(final_topk)

    stat_targets = statute_targets_for(qid)

    # 1) channel fingerprint
    chan_hits = defaultdict(list)   # did -> [channel names]
    if HAS_CHANNEL_HIT_SETS:
        for ch, dids in pq.get("channel_hit_sets", {}).items():
            dids_set = set(dids) & pool
            for d in dids_set:
                chan_hits[d].append(ch)
    else:
        # degrade: mark which channels had ANY hit for this query; can't attribute per-doc.
        # leave chan_hits empty — dossier will show "channel info unavailable"
        pass

    # 2) statute-target intersection per doc + 3) co-citation density in pool
    top_cocite = final_topk[:CONFIG["dossier_top_for_cocite"]]
    cocite_count = Counter()
    for d in top_cocite:
        doc = CORPUS.get(d)
        if not doc or doc.get("fam") != "court":
            continue
        for sa in doc.get("sa", []):
            cocite_count[sa] += 1

    qd = {}
    for d in final_topk:
        doc = CORPUS.get(d)
        if not doc:
            continue
        sa = set(doc.get("sa", []))
        stat_hits = sorted(sa & stat_targets)
        # For this law/article doc itself, how many top-N court paras cite it?
        # (Use the canonical anchor that best matches its citation string.)
        doc_anchor = None
        for a in sa:
            if a in cocite_count and (doc_anchor is None or cocite_count[a] > cocite_count[doc_anchor]):
                doc_anchor = a
        cocite = cocite_count[doc_anchor] if doc_anchor else 0
        qd[d] = {
            "channels":     chan_hits.get(d, []),
            "stat_hits":    stat_hits,
            "stat_overlap": len(stat_hits),
            "cocite":       cocite,
            "fam":          doc.get("fam", "?"),
            "cit":          doc.get("cit", d),
            "cb":           doc.get("cb", ""),
            "pr":           doc.get("pr", ""),
            "ln":           doc.get("ln", "?"),
            "text":         doc.get("ct", ""),
        }
    DOSSIER[qid] = qd

print(f"Built dossier for {len(DOSSIER)} queries.")
print("Sample (first val query, first 3 candidates):")
sample_qid = next(iter(DOSSIER))
for d, info in list(DOSSIER[sample_qid].items())[:3]:
    print(f"  {d[:25]:25s}  fam={info['fam']:5s}  stat_overlap={info['stat_overlap']}  cocite={info['cocite']}  channels={info['channels']}")
"""))

CELLS.append(code("""# Dossier text formatter — produces the ~250-token block fed to reranker and judge.
def render_dossier(qid, did, max_chars=None):
    info = DOSSIER[qid][did]
    targets = ALL_TARGETS.get(qid, {}) or {}
    n_targets = len(targets.get("statute_targets") or [])
    n_concepts = len(targets.get("concept_targets_en") or [])
    parts = [
        f"[{info['cit']}]  ({info['ln']}, {info['fam']}, {info['cb'] or '-'}, role={info['pr'] or '-'})"
    ]
    if info["channels"]:
        parts.append(f"surfaced by: {', '.join(info['channels'][:6])} — {len(info['channels'])}/15 channels")
    if info["stat_hits"]:
        parts.append(f"cites: {{{', '.join(info['stat_hits'][:10])}}}  ← {info['stat_overlap']} of your {n_targets} query targets")
    elif n_targets:
        parts.append(f"cites: none of your {n_targets} query targets")
    if info["cocite"] > 0:
        parts.append(f"cited by {info['cocite']} of your top-{CONFIG['dossier_top_for_cocite']} court paragraphs")
    txt = (info["text"] or "")[:600]
    parts.append(f"text: \\"{txt}\\"")
    out = "\\n".join(parts)
    if max_chars and len(out) > max_chars:
        out = out[:max_chars]
    return out

sample = render_dossier(sample_qid, list(DOSSIER[sample_qid].keys())[0])
print(sample)
print()
print(f"length: {len(sample)} chars")
"""))

# =============================================================================
# Phase 4 — Multi-agent decomposition
# =============================================================================
CELLS.append(md("""# Phase 4 — LegalMALR multi-agent decomposition

Four "agents" — these are policy-controlled scoring/filtering moves, not separate LLMs. The agents compose a unified scoring function whose 12 scalars are the policy parameters optimized by GRPO.

| Agent | Role | Policy param(s) |
|---|---|---|
| A. **Statute focus** | weight candidates by how many query statute targets they cite | `w_stat_overlap` (cont) |
| B. **Family balancer** | up/down-weight law vs court family | `prior_law`, `prior_court` |
| C. **Aspect router** | up-weight candidates whose channel set covers more distinct aspects | `w_channel_cov`, `w_cocite` |
| D. **K predictor** | per-query K = clip(`k_base` + `k_per_target` · |statute_targets|, k_min, k_max) | `k_base`, `k_per_target` |

Plus stage-fusion weights `α_rerank`, `α_judge`, cut threshold `τ`, and the "auto-yes / auto-no" zone routing thresholds `τ_yes`, `τ_no`. Total: 12 scalars.

Each scalar is discretized into 5 bins → GRPO's policy is a categorical distribution over each dim. Action space size: 5^12 ≈ 244M but GRPO never enumerates — it samples groups.
"""))

CELLS.append(code("""import numpy as np

POLICY_DIMS = {
    # (name, [discrete values, low→high])
    "w_stat_overlap": [0.0, 0.25, 0.5, 1.0, 2.0],
    "w_channel_cov":  [0.0, 0.1,  0.25, 0.5, 1.0],
    "w_cocite":       [0.0, 0.05, 0.1,  0.25, 0.5],
    "prior_law":      [0.6, 0.8,  1.0,  1.2,  1.5],
    "prior_court":    [0.6, 0.8,  1.0,  1.2,  1.5],
    "alpha_rerank":   [0.0, 0.25, 0.5,  0.75, 1.0],
    "alpha_judge":    [0.0, 0.25, 0.5,  0.75, 1.0],
    "tau_cut":        [0.30,0.45, 0.55, 0.65, 0.75],
    "tau_yes":        [0.70,0.80, 0.85, 0.90, 0.95],
    "tau_no":         [0.05,0.15, 0.25, 0.35, 0.45],
    "k_base":         [10,  15,   20,   25,   30],
    "k_per_target":   [0.0, 0.3,  0.5,  0.8,  1.2],
}
N_DIMS = len(POLICY_DIMS)
DIM_NAMES = list(POLICY_DIMS.keys())
DIM_NBINS = [len(POLICY_DIMS[d]) for d in DIM_NAMES]
print(f"policy dims: {N_DIMS}")
print(f"action space: {np.prod(DIM_NBINS):.2e}")

def sample_policy(probs, rng):
    '''probs: list of length N_DIMS, each a np.ndarray of bin probabilities.
    Returns: indices (length N_DIMS), values (dict).'''
    idx = [int(rng.choice(DIM_NBINS[i], p=probs[i])) for i in range(N_DIMS)]
    vals = {DIM_NAMES[i]: POLICY_DIMS[DIM_NAMES[i]][idx[i]] for i in range(N_DIMS)}
    return idx, vals

def fixed_policy_values(values_dict):
    '''Get bin indices from a dict of policy values (for the "fixed" optimizer baseline).'''
    idx = []
    for i, d in enumerate(DIM_NAMES):
        v = values_dict.get(d, POLICY_DIMS[d][2])
        bins = POLICY_DIMS[d]
        # snap to nearest bin
        j = int(np.argmin([abs(v - b) for b in bins]))
        idx.append(j)
    return idx, {DIM_NAMES[i]: POLICY_DIMS[DIM_NAMES[i]][idx[i]] for i in range(N_DIMS)}

# Sensible baseline (used when optimizer=="fixed" and as GRPO init).
BASELINE_POLICY = {
    "w_stat_overlap": 1.0,
    "w_channel_cov":  0.25,
    "w_cocite":       0.1,
    "prior_law":      1.0,
    "prior_court":    1.0,
    "alpha_rerank":   0.5,
    "alpha_judge":    0.5,
    "tau_cut":        0.55,
    "tau_yes":        0.85,
    "tau_no":         0.25,
    "k_base":         20,
    "k_per_target":   0.5,
}
print(f"baseline:    {BASELINE_POLICY}")
"""))

# =============================================================================
# Phase 5 — Stage A: 8B reranker
# =============================================================================
CELLS.append(md("""# Phase 5 — Stage A: Qwen3-Reranker-8B over 50k → top-1000

Zero-shot pairwise yes/no scoring. Document text = cascade dossier (`render_dossier`), so the model scores against pre-digested evidence (channel fingerprint + statute hits + co-citation density + text excerpt) instead of raw German court paragraphs.

Per the endgame doc §4.4 the bottom-5 in the broken run were `law:*` with score 0.0. By feeding the dossier (which always contains the statute hits + cocite + a normalized text excerpt), the law family gets a fair shot.

Cached to `cache_legalmalr/rerank/<qid>.json`. Cache is **reusable across policy rollouts** — this is the expensive step (~minutes per query); GRPO rollouts after this are cheap (~milliseconds).
"""))

CELLS.append(code("""import json, hashlib, time
from pathlib import Path

class Qwen3Reranker:
    def __init__(self, model_name, max_len, dtype="bfloat16"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, dtype),
            device_map="cuda:0",
        ).eval()
        self.max_len = max_len
        # Per Qwen3-Reranker HF card.
        self.prefix = "<|im_start|>system\\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \\\"yes\\\" or \\\"no\\\".<|im_end|>\\n<|im_start|>user\\n"
        self.suffix = "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"
        self.prefix_ids = self.tok(self.prefix, return_tensors=None)["input_ids"]
        self.suffix_ids = self.tok(self.suffix, return_tensors=None)["input_ids"]
        self.token_true = self.tok("yes", add_special_tokens=False).input_ids[0]
        self.token_false = self.tok("no",  add_special_tokens=False).input_ids[0]
        print(f"  yes_id={self.token_true}  no_id={self.token_false}")

    def _format_pair(self, query, doc, instruction):
        return f"<Instruct>: {instruction}\\n<Query>: {query}\\n<Document>: {doc}"

    @torch.inference_mode()
    def score_batch(self, query, docs, instruction, batch_size=16):
        import torch
        scores = []
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i+batch_size]
            texts = [self._format_pair(query, d, instruction) for d in batch_docs]
            inputs = self.tok(
                texts, padding=False, truncation=True,
                max_length=self.max_len - len(self.prefix_ids) - len(self.suffix_ids),
                return_attention_mask=False, return_token_type_ids=False, add_special_tokens=False,
            )
            for j in range(len(inputs["input_ids"])):
                inputs["input_ids"][j] = self.prefix_ids + inputs["input_ids"][j] + self.suffix_ids
            inputs = self.tok.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_len)
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
            logits = self.model(**inputs).logits[:, -1, :]
            true_l = logits[:, self.token_true]
            false_l = logits[:, self.token_false]
            stacked = torch.stack([false_l, true_l], dim=1)
            probs = torch.log_softmax(stacked, dim=1)[:, 1].exp().cpu().tolist()
            scores.extend(probs)
        return scores

print("(reranker class defined; load deferred to next cell to make this cell cheap)")
"""))

CELLS.append(code("""# Load reranker + score top-K candidates per query with dossier-as-doc.
import json, time
from pathlib import Path

RERANK_CACHE = PATHS["cache_dir"] / "rerank"
RERANK_CACHE.mkdir(parents=True, exist_ok=True)

INSTRUCTION = "Given a legal research question (in English), determine whether the Swiss legal source described in the Document is one of the citations a Swiss legal expert would cite when answering the question."

# Load reranker once.
reranker = Qwen3Reranker(CONFIG["rerank_model"], CONFIG["rerank_max_len"])

RERANK_SCORES = {}   # qid -> {did: score}

for q in VAL_QUERIES:
    qid = q["query_id"]
    cache = RERANK_CACHE / f"{qid}.json"
    if cache.exists():
        RERANK_SCORES[qid] = json.loads(cache.read_text(encoding="utf-8"))
        print(f"  [cached] {qid}: {len(RERANK_SCORES[qid])} scores")
        continue
    final_topk = PER_QUERY_SNAP[qid]["final_topk"][:CONFIG["rerank_pool"]]
    if CONFIG["rerank_use_dossier"]:
        docs = [render_dossier(qid, d, max_chars=CONFIG["rerank_dossier_chars"]) for d in final_topk]
    else:
        docs = [(DOSSIER[qid][d]["text"] or "")[:CONFIG["rerank_dossier_chars"]] for d in final_topk]
    t = time.time()
    scores = reranker.score_batch(q["query"], docs, INSTRUCTION, batch_size=CONFIG["rerank_batch"])
    elapsed = time.time() - t
    RERANK_SCORES[qid] = {did: float(s) for did, s in zip(final_topk, scores)}
    cache.write_text(json.dumps(RERANK_SCORES[qid]), encoding="utf-8")
    # Gold recall in top-1000 after rerank
    sorted_dids = sorted(final_topk, key=lambda d: -RERANK_SCORES[qid][d])
    gold = GOLD_DOC_SETS.get(qid, set())
    hit_top1000 = sum(1 for d in sorted_dids[:CONFIG["rerank_topn"]] if d in gold)
    print(f"  {qid}: {len(final_topk)} scored in {elapsed:.1f}s  |  gold in top-{CONFIG['rerank_topn']} = {hit_top1000}/{len(gold)}")

# Free reranker VRAM before loading judge.
import torch, gc
del reranker
gc.collect(); torch.cuda.empty_cache()
print("\\nReranker unloaded.")
"""))

# =============================================================================
# Phase 6 — Stage B: LLM judge
# =============================================================================
CELLS.append(md("""# Phase 6 — Stage B: Qwen3-8B judge over top-1000 (per query) → verdict + confidence

Endgame §7 Move 3 patches applied verbatim:
- `enable_thinking=False` — no `<think>` tokens eating budget.
- Default to **NO** on parse failure (was default-YES; caused 87% rubber-stamp YES on the broken run).
- Drop the "say YES when uncertain" instruction.
- `max_new_tokens=24` — verdict only, no rationale.
- Strict output: `VERDICT: YES|NO  CONF: 0-1` regex.

Cached to `cache_legalmalr/judge/<qid>/<sha>.json`. Verdicts are policy-independent, so this cache is also reusable across GRPO rollouts.
"""))

CELLS.append(code("""import re, json, hashlib, time
import torch

JUDGE_CACHE = PATHS["cache_dir"] / "judge"
JUDGE_CACHE.mkdir(parents=True, exist_ok=True)

JUDGE_SYSTEM = (
    "You are a Swiss legal-citation auditor. Given a legal research question and a candidate "
    "Swiss legal source (with structured evidence about where it was retrieved from, which "
    "query statute targets it cites, and how often it is co-cited by top court paragraphs), "
    "you must decide whether a Swiss legal expert would cite this exact source when answering "
    "the question. Be strict: only YES if there is concrete, on-point evidence."
)
JUDGE_USER_TMPL = (
    "QUESTION:\\n{query}\\n\\n"
    "CANDIDATE SOURCE (dossier):\\n{dossier}\\n\\n"
    "Output exactly two lines, no other text, no thinking:\\n"
    "VERDICT: YES or NO\\n"
    "CONF: a number 0.00 to 1.00\\n"
)
VERDICT_RE = re.compile(r"VERDICT:\\s*(YES|NO)\\b", re.IGNORECASE)
CONF_RE    = re.compile(r"CONF:\\s*([0-9.]+)", re.IGNORECASE)

class Qwen3Judge:
    def __init__(self, model_name, enable_thinking):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
        ).eval()
        self.enable_thinking = enable_thinking

    @torch.inference_mode()
    def judge(self, query, dossier_text):
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": JUDGE_USER_TMPL.format(query=query, dossier=dossier_text)},
        ]
        prompt = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.tok(prompt, return_tensors="pt").to("cuda:0")
        out = self.model.generate(
            **inputs,
            max_new_tokens=CONFIG["judge_max_new_tokens"],
            do_sample=False,
            temperature=CONFIG["judge_temperature"],
            pad_token_id=self.tok.eos_token_id,
        )
        text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        m_v = VERDICT_RE.search(text)
        m_c = CONF_RE.search(text)
        if m_v is None:
            verdict = CONFIG["judge_default_on_parse_fail"].lower()
            conf = 0.5
        else:
            verdict = m_v.group(1).lower()
            conf = float(m_c.group(1)) if m_c else (0.7 if verdict == "yes" else 0.3)
        return verdict, max(0.0, min(1.0, conf)), text

print("Loading judge (Qwen3-8B)...")
judge = Qwen3Judge(CONFIG["judge_model"], enable_thinking=CONFIG["judge_enable_thinking"])
print("  ready.")
"""))

CELLS.append(code("""# Run judge on top-N reranked candidates per query. Cached per (qid, did).
JUDGE_SCORES = {}   # qid -> {did: {"verdict": "yes/no", "conf": float}}

for q in VAL_QUERIES:
    qid = q["query_id"]
    qcache = JUDGE_CACHE / qid
    qcache.mkdir(parents=True, exist_ok=True)
    final_topk = PER_QUERY_SNAP[qid]["final_topk"][:CONFIG["rerank_pool"]]
    sorted_by_rerank = sorted(final_topk, key=lambda d: -RERANK_SCORES[qid][d])[:CONFIG["judge_topn"]]
    out = {}
    t = time.time(); n_new = 0
    for did in sorted_by_rerank:
        cache_file = qcache / f"{hashlib.sha1(did.encode()).hexdigest()}.json"
        if cache_file.exists():
            out[did] = json.loads(cache_file.read_text(encoding="utf-8"))
            continue
        dossier_text = render_dossier(qid, did, max_chars=1400)
        v, c, raw = judge.judge(q["query"], dossier_text)
        rec = {"verdict": v, "conf": c}
        cache_file.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        out[did] = rec
        n_new += 1
    JUDGE_SCORES[qid] = out
    elapsed = time.time() - t
    yes_n = sum(1 for r in out.values() if r["verdict"] == "yes")
    gold = GOLD_DOC_SETS.get(qid, set())
    gold_in_judged = sum(1 for d in out if d in gold)
    gold_in_yes    = sum(1 for d, r in out.items() if r["verdict"] == "yes" and d in gold)
    print(f"  {qid}: judged {len(out)} in {elapsed:.1f}s ({n_new} new)  |  YES={yes_n}/{len(out)}  gold_in_judged={gold_in_judged}/{len(gold)}  gold_in_YES={gold_in_yes}/{len(gold)}")

# Free judge.
import gc
del judge
gc.collect(); torch.cuda.empty_cache()
print("\\nJudge unloaded.")
"""))

# =============================================================================
# Phase 7 — Policy + GRPO
# =============================================================================
CELLS.append(md("""# Phase 7 — Cascade policy + GRPO

The policy fuses the cached reranker score, judge verdict+confidence, and the dossier signals into a single per-(qid, did) score. Then it applies a cut threshold + variable K. The 12 scalars are learned per LOO fold by **Generalized Reinforcement Policy Optimization (GRPO)**:

1. Initialize each dim's bin distribution uniformly (or biased by `BASELINE_POLICY`).
2. Repeat for `grpo_steps`:
   - Sample `grpo_groups` policies from the current distribution.
   - Run each on the 9 training queries → reward = mean F1.
   - Compute group-normalized advantage `A_g = (R_g - μ_R) / σ_R`.
   - Update each dim's log-probs with `lr · A_g · 1{a_g}` + entropy bonus.
3. Final policy = argmax of each dim's distribution → evaluate on held-out 10th query.

All rerank/judge scores are **already cached**, so each rollout is just a re-weighting — fast.
"""))

CELLS.append(code("""# Cascade scorer + reward function.
import math

def policy_score(qid, did, p):
    '''Single per-doc fused score under policy `p`.'''
    info = DOSSIER[qid][did]
    rerank = RERANK_SCORES[qid].get(did, 0.0)
    judge_rec = JUDGE_SCORES[qid].get(did)
    if judge_rec is not None:
        judge_signal = judge_rec["conf"] if judge_rec["verdict"] == "yes" else (1.0 - judge_rec["conf"])
    else:
        judge_signal = 0.5   # un-judged: neutral
    fam_prior = p["prior_law"] if info["fam"] == "law" else p["prior_court"] if info["fam"] == "court" else 1.0
    dossier_score = (
        p["w_stat_overlap"] * (info["stat_overlap"] / 5.0) +
        p["w_channel_cov"]  * (len(info["channels"]) / 6.0) +
        p["w_cocite"]       * (info["cocite"] / 20.0)
    )
    fused = (
        p["alpha_rerank"] * rerank +
        p["alpha_judge"]  * judge_signal +
        (1.0 - p["alpha_rerank"] - p["alpha_judge"]) * dossier_score
    ) * fam_prior
    return fused, judge_signal

def predict(qid, p):
    '''Return list of selected doc_ids for this query under policy p.'''
    candidates = list(JUDGE_SCORES[qid].keys())   # only judged candidates are considered
    n_targets = len(ALL_TARGETS.get(qid, {}).get("statute_targets") or [])
    k = int(p["k_base"] + p["k_per_target"] * n_targets)
    k = max(CONFIG["k_min"], min(CONFIG["k_max"], k))
    scored = []
    for d in candidates:
        s, j = policy_score(qid, d, p)
        # auto-no zone: judge said NO with conf >= tau_yes → drop
        rec = JUDGE_SCORES[qid].get(d, {})
        if rec.get("verdict") == "no" and rec.get("conf", 0.0) >= p["tau_yes"]:
            continue
        # auto-yes shortcut: judge said YES with conf >= tau_yes → keep regardless of score
        force_keep = (rec.get("verdict") == "yes" and rec.get("conf", 0.0) >= p["tau_yes"])
        scored.append((d, s, force_keep))
    # auto-yes wins; then score; then threshold cut + top-K
    forced = [d for d, s, fk in scored if fk]
    rest = [(d, s) for d, s, fk in scored if not fk]
    rest.sort(key=lambda t: -t[1])
    rest_kept = [d for d, s in rest if s >= p["tau_cut"]]
    selected = list(dict.fromkeys(forced + rest_kept))[:k]
    return selected

def f1_for(qid, predicted_dids):
    gold = GOLD_DOC_SETS.get(qid, set())
    if not gold or not predicted_dids:
        return 0.0
    pset = set(predicted_dids)
    tp = len(pset & gold)
    p = tp / max(1, len(pset))
    r = tp / max(1, len(gold))
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

# Baseline F1 across val with BASELINE_POLICY.
baseline_f1s = []
for q in VAL_QUERIES:
    preds = predict(q["query_id"], BASELINE_POLICY)
    f1 = f1_for(q["query_id"], preds)
    baseline_f1s.append(f1)
    print(f"  {q['query_id']}: baseline F1 = {f1:.3f}  (selected={len(preds)}, gold={len(q['gold'])})")
import statistics as st
print(f"\\nBaseline macro F1 (no GRPO): {sum(baseline_f1s)/len(baseline_f1s):.4f}  ±{st.stdev(baseline_f1s):.3f}")
"""))

CELLS.append(code("""# GRPO over discrete bin policies (per-dim categorical).
import numpy as np

def init_probs(biased_to=None):
    probs = []
    for i, d in enumerate(DIM_NAMES):
        if biased_to is not None:
            bins = POLICY_DIMS[d]
            j = int(np.argmin([abs(biased_to[d] - b) for b in bins]))
            p = np.ones(DIM_NBINS[i]) * (1.0 / DIM_NBINS[i])
            p[j] += 1.5
            p = p / p.sum()
        else:
            p = np.ones(DIM_NBINS[i]) / DIM_NBINS[i]
        probs.append(p)
    return probs

def policy_logprob(probs, idx):
    return sum(math.log(probs[i][idx[i]] + 1e-12) for i in range(N_DIMS))

def grpo_train(train_qids, n_steps, n_groups, lr, entropy_bonus, seed, verbose=False):
    rng = np.random.default_rng(seed)
    probs = init_probs(BASELINE_POLICY)
    best_reward = -1e9; best_idx = None; best_vals = None
    history = []
    for step in range(n_steps):
        # sample group
        group_idx = []
        group_vals = []
        for _ in range(n_groups):
            idx, vals = sample_policy(probs, rng)
            group_idx.append(idx); group_vals.append(vals)
        # evaluate
        rewards = []
        for vals in group_vals:
            f1s = []
            for qid in train_qids:
                preds = predict(qid, vals)
                f1s.append(f1_for(qid, preds))
            rewards.append(float(np.mean(f1s)))
        r = np.array(rewards)
        if r.std() > 1e-6:
            adv = (r - r.mean()) / (r.std() + 1e-8)
        else:
            adv = r - r.mean()
        # update probs: log-prob ascent with advantage; multiplicative on logits.
        logits = [np.log(p + 1e-12) for p in probs]
        for g in range(n_groups):
            idx = group_idx[g]
            for i in range(N_DIMS):
                logits[i][idx[i]] += lr * adv[g] / n_groups
        # entropy bonus: pull a bit toward uniform
        for i in range(N_DIMS):
            uniform = np.log(np.ones(DIM_NBINS[i]) / DIM_NBINS[i])
            logits[i] = (1 - entropy_bonus) * logits[i] + entropy_bonus * uniform
            m = logits[i].max()
            probs[i] = np.exp(logits[i] - m); probs[i] /= probs[i].sum()
        max_r = r.max()
        if max_r > best_reward:
            j = int(np.argmax(r))
            best_reward = max_r; best_idx = group_idx[j]; best_vals = group_vals[j]
        history.append({"step": step, "mean_r": float(r.mean()), "max_r": float(r.max()), "std_r": float(r.std())})
        if verbose and (step+1) % 10 == 0:
            print(f"    step {step+1:3d}: mean_r={r.mean():.4f}  max_r={r.max():.4f}  std={r.std():.4f}")
    # argmax policy (deterministic readout)
    argmax_idx = [int(np.argmax(p)) for p in probs]
    argmax_vals = {DIM_NAMES[i]: POLICY_DIMS[DIM_NAMES[i]][argmax_idx[i]] for i in range(N_DIMS)}
    return {"argmax_vals": argmax_vals, "best_vals": best_vals, "best_reward": best_reward, "history": history, "probs": probs}

print("GRPO trainer defined.")
"""))

CELLS.append(code("""# Optuna fallback (Bayesian optimization) for when GRPO is noisy on 9-query folds.
def optuna_train(train_qids, n_trials, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        vals = {}
        for d in DIM_NAMES:
            vals[d] = trial.suggest_categorical(d, POLICY_DIMS[d])
        f1s = [f1_for(qid, predict(qid, vals)) for qid in train_qids]
        return float(np.mean(f1s))
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"argmax_vals": study.best_params, "best_vals": study.best_params, "best_reward": study.best_value}

print("Optuna trainer defined.")
"""))

CELLS.append(code("""# Leave-one-out across 10 val queries.
val_qids = [q["query_id"] for q in VAL_QUERIES]
LOO_RESULTS = []
for held in val_qids:
    train = [qid for qid in val_qids if qid != held]
    print(f"\\n=== LOO fold: hold out {held} ===")
    if CONFIG["policy_optimizer"] == "grpo":
        res = grpo_train(train, CONFIG["grpo_steps"], CONFIG["grpo_groups"],
                         CONFIG["grpo_lr"], CONFIG["grpo_entropy_bonus"],
                         seed=CONFIG["loo_seed"] + hash(held) % 1000,
                         verbose=True)
    elif CONFIG["policy_optimizer"] == "optuna":
        res = optuna_train(train, CONFIG["optuna_trials"], seed=CONFIG["loo_seed"] + hash(held) % 1000)
    else:
        res = {"argmax_vals": BASELINE_POLICY, "best_vals": BASELINE_POLICY, "best_reward": 0.0}
    # Score held-out
    p_argmax = res["argmax_vals"]
    p_best   = res.get("best_vals", p_argmax)
    preds_argmax = predict(held, p_argmax)
    preds_best   = predict(held, p_best)
    f1_argmax = f1_for(held, preds_argmax)
    f1_best   = f1_for(held, preds_best)
    print(f"  argmax policy F1 on {held}: {f1_argmax:.4f}  (preds={len(preds_argmax)})")
    print(f"  best-seen    F1 on {held}: {f1_best:.4f}  (preds={len(preds_best)})")
    LOO_RESULTS.append({"held": held, "policy_argmax": p_argmax, "policy_best": p_best,
                        "f1_argmax": f1_argmax, "f1_best": f1_best,
                        "preds_argmax": preds_argmax, "preds_best": preds_best,
                        "train_reward": res.get("best_reward", 0.0)})

macro_argmax = np.mean([r["f1_argmax"] for r in LOO_RESULTS])
macro_best   = np.mean([r["f1_best"]   for r in LOO_RESULTS])
print(f"\\nLOO macro F1 (argmax policy):    {macro_argmax:.4f}")
print(f"LOO macro F1 (best-seen policy): {macro_best:.4f}")
print(f"Target: 0.6 - 0.7")
"""))

# =============================================================================
# Phase 8 — Train sanity
# =============================================================================
CELLS.append(md("""# Phase 8 — Train sanity check

Run the LOO-best policy on a sample of train queries with **no GRPO** — just to confirm the policy doesn't collapse outright on the out-of-distribution short German queries.

Expect: train F1 will be *substantially lower than val* (train median gold = 2 vs val median = 22; train is 99% German vs val 100% English). If train F1 ≪ 0.05 the policy might still be fine on val. If train F1 is *higher* than val F1, that's a red flag — likely overfit to dossier features that don't transfer.

This is a diagnostic only, not a fitness signal. **Do not** tune policy on train. (See memory entry `feedback_train_unreliable.md`.)
"""))

CELLS.append(code("""# Sample N train queries; build a mini-snapshot for them; predict.
# NOTE: full train evaluation requires re-running the recall-0.89 pipeline on those queries.
# Here we do a cheap sanity check: pick train queries that the existing snapshot ALSO covered
# (if any), else skip. If your snapshot was val-only (typical), this cell prints a notice.

train_in_snapshot = [r for r in train_df.itertuples() if r.query_id in PER_QUERY_SNAP]
if not train_in_snapshot:
    print("[notice] no train queries appear in the warm-booted snapshot.")
    print("         To run a real train sanity check, re-run the recall-0.89 pipeline on a")
    print("         sample of train queries first, then warm-boot here. Skipping.")
else:
    n = min(CONFIG["train_sanity_n"], len(train_in_snapshot))
    print(f"  evaluating {n} train queries with LOO median policy...")
    # use LOO median policy across the 10 folds
    median_policy = {}
    for d in DIM_NAMES:
        vals = [r["policy_argmax"][d] for r in LOO_RESULTS]
        median_policy[d] = float(np.median(vals)) if isinstance(vals[0], (int, float)) else max(set(vals), key=vals.count)
    train_f1s = []
    for r in train_in_snapshot[:n]:
        gold = set(parse_gold(r.gold_citations))
        preds = predict(r.query_id, median_policy)
        # Note: train gold uses citation strings, not doc_ids; would need citation→doc_id map for true F1
        # As a proxy, count how many predicted citations match by string fragment
        pred_cits = {DOSSIER[r.query_id][d]["cit"] for d in preds if d in DOSSIER[r.query_id]}
        tp = len(pred_cits & gold)
        f1 = (2 * tp / (len(pred_cits) + len(gold))) if (len(pred_cits) + len(gold)) > 0 else 0.0
        train_f1s.append(f1)
    print(f"  train sanity F1 (string-overlap proxy): {np.mean(train_f1s):.4f}  ± {np.std(train_f1s):.4f}")
    print(f"  val LOO macro F1 (argmax):              {macro_argmax:.4f}")
    if np.mean(train_f1s) > macro_argmax:
        print(f"  [RED FLAG] train F1 > val F1. Investigate: may indicate dossier features that overfit to short-cite queries.")
    else:
        print(f"  [OK] train F1 lower than val F1, as expected given distribution shift.")
"""))

# =============================================================================
# Phase 9 — Final predictions on val + K-sweep
# =============================================================================
CELLS.append(md("""# Phase 9 — Final predictions on val + K-sweep

Report per-query F1 + macro F1 + F1 K-sweep at fixed K, so the LOO-policy variable-K can be compared against a tuned fixed-K baseline.
"""))

CELLS.append(code("""# Per-query LOO-best predictions table.
print(f"{'query':<10}  {'gold':>4}  {'pred':>4}  {'TP':>3}  {'P':>6}  {'R':>6}  {'F1':>6}")
print("-" * 60)
for r in LOO_RESULTS:
    qid = r["held"]
    gold = GOLD_DOC_SETS.get(qid, set())
    preds = set(r["preds_argmax"])
    tp = len(preds & gold)
    P = tp / max(1, len(preds)); R = tp / max(1, len(gold))
    print(f"{qid:<10}  {len(gold):>4}  {len(preds):>4}  {tp:>3}  {P:>6.3f}  {R:>6.3f}  {r['f1_argmax']:>6.3f}")
print("-" * 60)
print(f"{'MACRO':<10}                            {macro_argmax:>6.4f}")
"""))

CELLS.append(code("""# Fixed-K sweep using ONE global policy (mean over LOO).
def predict_fixed_k(qid, p, K):
    candidates = list(JUDGE_SCORES[qid].keys())
    scored = []
    for d in candidates:
        s, _ = policy_score(qid, d, p)
        scored.append((d, s))
    scored.sort(key=lambda t: -t[1])
    return [d for d, _ in scored[:K]]

# Use mean policy across LOO (just for the K-sweep diagnostic).
mean_policy = {}
for d in DIM_NAMES:
    vals = [r["policy_argmax"][d] for r in LOO_RESULTS]
    if isinstance(vals[0], (int, float)):
        mean_policy[d] = float(np.mean(vals))
    else:
        mean_policy[d] = max(set(vals), key=vals.count)

print(f"\\nK-sweep with mean-LOO policy (fixed K, ignoring k_base/k_per_target):")
print(f"  {'K':>4}  {'macro F1':>10}")
for K in CONFIG["k_sweep"]:
    f1s = []
    for q in VAL_QUERIES:
        preds = predict_fixed_k(q["query_id"], mean_policy, K)
        f1s.append(f1_for(q["query_id"], preds))
    print(f"  {K:>4}  {np.mean(f1s):>10.4f}")
"""))

# =============================================================================
# Phase 10 — Save + (optional) test inference
# =============================================================================
CELLS.append(md("""# Phase 10 — Save artifacts + (optional) test inference

Persists per-fold policies, per-query predictions, macro F1. Test inference is **only meaningful if val F1 ≥ 0.5** — otherwise we'd be submitting noise.
"""))

CELLS.append(code("""import json as _j
out = PATHS["out_dir"]
out.mkdir(parents=True, exist_ok=True)

# Strip non-JSON-serializable bits.
loo_dump = []
for r in LOO_RESULTS:
    loo_dump.append({
        "held":          r["held"],
        "policy_argmax": r["policy_argmax"],
        "policy_best":   r["policy_best"],
        "f1_argmax":     r["f1_argmax"],
        "f1_best":       r["f1_best"],
        "preds_argmax":  r["preds_argmax"],
        "train_reward":  r["train_reward"],
    })
(out / "loo_results.json").write_text(_j.dumps(loo_dump, indent=2, ensure_ascii=False), encoding="utf-8")
(out / "macro_f1.json").write_text(_j.dumps({
    "macro_argmax": float(macro_argmax),
    "macro_best":   float(macro_best),
    "policy_optimizer": CONFIG["policy_optimizer"],
    "n_val": len(VAL_QUERIES),
}, indent=2), encoding="utf-8")

# Per-query final predictions in submission-ready format (citation strings).
per_query_preds = {}
for r in LOO_RESULTS:
    qid = r["held"]
    cits = []
    for did in r["preds_argmax"]:
        info = DOSSIER.get(qid, {}).get(did)
        if info:
            cits.append(info["cit"])
    per_query_preds[qid] = list(dict.fromkeys(cits))
(out / "val_predictions.json").write_text(_j.dumps(per_query_preds, indent=2, ensure_ascii=False), encoding="utf-8")

print(f"Saved to {out}")
print(f"  loo_results.json   ({len(loo_dump)} folds)")
print(f"  macro_f1.json      macro_argmax={macro_argmax:.4f}")
print(f"  val_predictions.json  ({len(per_query_preds)} queries)")
if macro_argmax >= 0.6:
    print(f"\\n[OK] macro F1 = {macro_argmax:.4f} is in the target range 0.6 - 0.7.")
elif macro_argmax >= 0.4:
    print(f"\\n[close] macro F1 = {macro_argmax:.4f}. Try: increasing CONFIG['grpo_steps'], or switch policy_optimizer='optuna'.")
else:
    print(f"\\n[gap] macro F1 = {macro_argmax:.4f}. Likely cause: judge YES-rate too high or too low; inspect JUDGE_SCORES distribution.")
"""))

CELLS.append(code("""# Test inference scaffold. Requires the recall-0.89 pipeline to have been run on test.csv
# AND its snapshot folded into the warm-boot. If not, this cell prints a notice and stops.
test_qids_in_snapshot = [r for r in pd.read_csv(PATHS["test_csv"]).itertuples()
                         if r.query_id in PER_QUERY_SNAP and r.query_id in DOSSIER]
if not test_qids_in_snapshot:
    print("[notice] no test queries in warm-boot. To produce a test submission:")
    print("  1. Re-run swiss_citation_anchor_funnel_v7_5_multiquery on test.csv.")
    print("  2. Merge the resulting per_query_snapshot.json + corpus_snapshot.json.gz into the snapshot folder.")
    print("  3. Re-run THIS notebook from Phase 2.")
else:
    # Mean policy across LOO is the best fixed prediction policy for test.
    submission_rows = []
    for r in test_qids_in_snapshot:
        preds = predict(r.query_id, mean_policy)
        cits = []
        for did in preds:
            info = DOSSIER.get(r.query_id, {}).get(did)
            if info: cits.append(info["cit"])
        submission_rows.append({"query_id": r.query_id, "citations": ";".join(dict.fromkeys(cits))})
    sub_df = pd.DataFrame(submission_rows)
    sub_path = out / "submission_test.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"Wrote {sub_path}  ({len(sub_df)} rows)")
"""))

CELLS.append(code("""# Final teardown.
import gc, torch
for name in ["reranker", "judge"]:
    if name in globals():
        del globals()[name]
gc.collect(); torch.cuda.empty_cache()
if torch.cuda.is_available():
    print(f"VRAM after teardown: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated")
print("Done.")
"""))

# =============================================================================
# Assemble notebook
# =============================================================================
notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        "colab": {"gpuType": "RTX_PRO_6000_BLACKWELL", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Wrote {NB_PATH}")
print(f"  cells: {len(CELLS)}")
print(f"  size:  {NB_PATH.stat().st_size/1024:.1f} KB")
