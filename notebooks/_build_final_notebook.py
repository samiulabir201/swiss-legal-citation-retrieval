"""Build the final notebook `swiss_citation_final_2026.ipynb`.

Same architecture as before, but ported to run on Google Drive (Colab) with vLLM,
mirroring the setup pattern of `swiss_citation_precision_v1_(3).ipynb`.

Pipeline:
  Phase 0 - environment detection + vLLM/FlashInfer install + paths
  Phase 0b - flashinfer sanity import
  Phase 1 - warm-boot from snapshot
  Phase 2 - Qwen3-Reranker-8B rerank via vLLM (top-5000 -> calibrated probs)
  Phase 3 - score-distribution diagnostics
  Phase 4 - six variable-K estimators
  Phase 5 - K ensemble + asymmetric clamp
  Phase 6 - build final candidate sets
  Phase 7 - (optional) evidence-quote verifier
  Phase 8 - evaluation + ablation

All citations / hyperparameters traceable to research_papers/INDEX.md.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).parent / "swiss_citation_final_2026.ipynb"

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def code(text: str) -> None:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    })


# =============================================================================
# Title + rationale
# =============================================================================
md(r'''# Swiss Citation Extraction - Final Notebook (2026-05-13, Colab + vLLM)

End-to-end pipeline from the v7.5 snapshot to per-query variable-K answer sets,
targeting macro F1 in the 0.6-0.8 band. All design choices grounded in
2025-2026 literature; see `FINAL_NOTEBOOK_RATIONALE.md` for the evidence base.

**Setup pattern (Colab + vLLM)**: same as `swiss_citation_precision_v1_(3).ipynb` -
auto-detect Colab vs local; mount Drive on Colab; install vLLM + FlashInfer once.

**Pipeline:**

| Phase | Stage | Output | Hardware |
|---|---|---|---|
| 0  | Environment + vLLM/FlashInfer install + paths | `_DRIVE`, `SNAPSHOT_DIR`, `OUT_DIR` | CPU |
| 0b | flashinfer sanity import | binding check | CPU |
| 1  | Warm-boot from snapshot | `PER_QUERY`, `ALL_*` globals | CPU, ~10s |
| 2  | Qwen3-Reranker-8B rerank top-5000 via **vLLM logprobs** -> calibrated probs | `stage1_ranked[qid]` | GPU |
| 3  | Score-distribution diagnostics | sanity printouts | CPU |
| 4  | Six variable-K estimators (Adaptive-k, CAR, PSI-Rank, AcuRank, Conformal, AspectCover) | `K_est[qid][method]` | mostly CPU |
| 5  | K ensemble + asymmetric clamp | `K_final[qid]` | CPU |
| 6  | Build final answer sets | `answers[qid]` | CPU |
| 7  | (optional) Qwen3-32B evidence-quote verifier | swap-not-drop | GPU |
| 8  | Macro/micro P/R/F1 + per-query + ablation | summary table | CPU |

**Key invariants** (no hardcoding):
- Every K estimator reads only the reranker's score distribution OR the doc's own fields (paragraph_role, family, language, etc.). No val-specific lexicon.
- Train.csv is used only as a conformal **calibration** set (a quantile threshold), never to derive Swiss-statute priors.
- K varies per query in [3, n_with_p>0.05]. Never a fixed cap.

**Reference numbers** (from `research/stage_b_experiments/REPORT.md` and 2025-2026 papers):
- v7.5 RRF baseline: macro R@5000 = 0.728, R@2000 = 0.611, R@50000 = 0.89-0.90
- Per-query in-pool ceiling: val_003 = 0.766 (binding); macro = 0.866
- Perfect-oracle macro F1 = 0.944
- COLIEE 2025 SOTA F1 = 0.335 (legal case retrieval); Untitled75 reference on this val = 0.777
''')


# =============================================================================
# Phase 0 - Setup (Colab/local + vLLM + FlashInfer)
# =============================================================================
md(r'''# Phase 0 - Setup (imports + path resolution + vLLM/FlashInfer install)

Detects Colab vs local, mounts Drive on Colab, installs `vllm` and
`flashinfer` once. **First install requires a runtime restart** - the cell
will raise SystemExit; re-run it after restart. Pattern verbatim from
`swiss_citation_precision_v1_(3).ipynb`.''')

code(r'''# Phase 0 - Setup: imports + path resolution + vLLM/FlashInfer install
# (Blackwell-optimized - pattern from swiss_citation_precision_v1_(3).ipynb)

import sys, os, json, gzip, time, re, gc, subprocess
import importlib.util
from pathlib import Path
from collections import defaultdict, Counter

print(f"Python {sys.version_info.major}.{sys.version_info.minor}")

# Environment detection
try:
    from google.colab import drive as _drv
    _drv.mount("/content/drive", force_remount=False)
    _DRIVE = Path("/content/drive/MyDrive/swiss_law")
    ENV = "colab"
except (ImportError, ModuleNotFoundError):
    _DRIVE = Path(r"E:\swiss_citation_extraction")
    ENV = "local"
IN_KAGGLE = bool(os.environ.get("KAGGLE_URL_BASE") or Path("/kaggle").exists())
print(f"Environment: {ENV}{' (Kaggle)' if IN_KAGGLE else ''}")

# Snapshot path: try canonical first, fall back to doubled-nested (local layout)
_snap_a = _DRIVE / "research" / "anchor_funnel_val001_v7" / "snapshot"
_snap_b = _DRIVE / "research" / "anchor_funnel_val001_v7" / "snapshot" / "snapshot"
if (_snap_a / "config.json").exists():
    SNAPSHOT_DIR = _snap_a
elif (_snap_b / "config.json").exists():
    SNAPSHOT_DIR = _snap_b
else:
    SNAPSHOT_DIR = _snap_a  # leave the canonical so the assert below fires with the canonical message
OUT_DIR      = _DRIVE / "research" / "final_2026"
OUT_DIR.mkdir(parents=True, exist_ok=True)
assert SNAPSHOT_DIR.exists() and (SNAPSHOT_DIR / "config.json").exists(), (
    f"Snapshot dir missing: {SNAPSHOT_DIR}\n"
    f"Run cell 48 of swiss_citation_anchor_funnel_v7_5_multiquery first."
)
print(f"snapshot: {SNAPSHOT_DIR}")
print(f"out:      {OUT_DIR}")

# vLLM + FlashInfer install (Blackwell-optimized, one-time)
_NEEDS_RESTART = False

if importlib.util.find_spec("vllm") is None:
    print("\n[setup] installing core packages (~3 min)...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "-U",
        "vllm>=0.10.0", "transformers>=4.51.0",
        "accelerate", "safetensors", "huggingface_hub",
    ], check=True)
    _NEEDS_RESTART = True

if importlib.util.find_spec("flashinfer") is None:
    print("[setup] installing FlashInfer (CUDA-version-aware)...")
    def _cuda_suffix():
        try:
            import torch as _t
            cuda = (_t.version.cuda or "").strip()
            if not cuda: return None
            parts = cuda.split(".")
            major, minor = int(parts[0]), int(parts[1])
            supported = {"cu126", "cu128", "cu129", "cu130", "cu131"}
            s = f"cu{major}{minor}"
            if s in supported: return s
            cands = sorted(supported, key=lambda x: int(x[2:]))
            det = major * 10 + minor
            best = None
            for c in cands:
                if int(c[2:]) <= det: best = c
            return best or cands[0]
        except Exception:
            return None
    _cidx = _cuda_suffix()
    print(f"  CUDA index suffix: {_cidx}")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                        "flashinfer-python", "flashinfer-cubin"], check=False)
        if _cidx:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "--pre",
                            "flashinfer-jit-cache",
                            "--index-url", f"https://flashinfer.ai/whl/{_cidx}"], check=False)
        os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
        importlib.invalidate_caches()
        import flashinfer
        print(f"  flashinfer {getattr(flashinfer, '__version__', '?')} OK")
        _NEEDS_RESTART = True
    except Exception as e:
        print(f"  flashinfer install skipped ({e}); vLLM auto-selects backend (still fast).")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if _NEEDS_RESTART:
    print("\n" + "=" * 64)
    print("IMPORTANT: this was a FIRST install. RESTART THE RUNTIME ONCE")
    print("(Runtime -> Restart runtime in Colab; Kernel -> Restart in Jupyter),")
    print("then re-run this cell. The runtime restart is needed for vLLM's")
    print("and FlashInfer's CUDA extensions to bind cleanly. Cell 2 onwards")
    print("will fail without the restart.")
    print("=" * 64)
    raise SystemExit("Restart runtime and re-run from this cell.")
else:
    print("\n[setup] vLLM + FlashInfer already installed; proceeding.")
''')


# Phase 0b
md(r'''## Phase 0b - flashinfer sanity import

If this errors, FlashInfer isn't bound to the current CUDA. vLLM will still
run but slower; revisit the install if speed matters.''')

code(r'''import flashinfer''')


# =============================================================================
# Phase 1 - Warm-boot from snapshot
# =============================================================================
md(r'''# Phase 1 - Warm-Boot from snapshot

Loads `CONFIG`, `ALL_QUERIES`, `search_text`, `doc_meta`, `doc_statute_anchors`,
`_doc_to_concepts`, `_doc_to_terms`, `PER_QUERY`, `ALL_TARGETS`,
`ALL_HYDE_ASPECTS`, `ALL_GOLD_DOC_SET`, `ALL_TOTAL_GOLD` from the snapshot in
about 10-60s.

The snapshot is the output of `swiss_citation_anchor_funnel_v7_5_multiquery`
cell 48: top-50k candidates per val query with macro recall about 0.89.
''')

code(r'''# Phase 1 - Warm-boot from snapshot
import json as _j, gzip as _gz
import pandas as _pd

print(f"[warm-boot] loading from {SNAPSHOT_DIR}")
_t0 = time.time()

# CONFIG
CONFIG = _j.loads((SNAPSHOT_DIR / "config.json").read_text(encoding="utf-8"))
print(f"  CONFIG loaded (topk_final={CONFIG.get('topk_final')})")

# Resolve val.csv - paths.json may reference Drive; fall back to local
_paths_meta = _j.loads((SNAPSHOT_DIR / "paths.json").read_text(encoding="utf-8"))
VAL_CSV = Path(_paths_meta["val_csv"])
if not VAL_CSV.exists():
    VAL_CSV = _DRIVE / "data" / "val.csv"
assert VAL_CSV.exists(), f"val.csv not found at {VAL_CSV}"
val_df = _pd.read_csv(VAL_CSV)
ALL_QUERIES = [
    {"query_id":   str(r["query_id"]),
     "query_text": str(r["query"]),
     "gold":       [c.strip() for c in str(r["gold_citations"]).split(";") if c.strip()]}
    for _, r in val_df.iterrows()
]
ALL_TOTAL_GOLD = {q["query_id"]: len(q["gold"]) for q in ALL_QUERIES}
print(f"  ALL_QUERIES: {len(ALL_QUERIES)}  ({VAL_CSV.name})")

# Corpus snapshot
with _gz.open(SNAPSHOT_DIR / "corpus_snapshot.json.gz", "rt", encoding="utf-8") as _f:
    _c = _j.load(_f)
search_text         = {_d: _r["ct"]  for _d, _r in _c.items()}
doc_meta            = {_d: {"citation":       _r["cit"],
                            "family":         _r["fam"],
                            "court_base":     _r["cb"],
                            "paragraph_role": _r["pr"],
                            "language":       _r["ln"]}
                       for _d, _r in _c.items()}
doc_statute_anchors = {_d: set(_r["sa"]) for _d, _r in _c.items()}
_doc_to_concepts    = {_d: set(_r["cn"]) for _d, _r in _c.items()}
_doc_to_terms       = {_d: set(_r["tm"]) for _d, _r in _c.items()}
print(f"  corpus_snapshot: {len(_c):,} unique docs (cascade pool)")
del _c

# PER_QUERY
_pq = _j.loads((SNAPSHOT_DIR / "per_query_snapshot.json").read_text(encoding="utf-8"))
PER_QUERY = {
    _qid: {
        "final_topk":      _r["final_topk"],
        "curve":           {int(_k): tuple(_v) for _k, _v in _r.get("curve", {}).items()},
        "channel_recalls": {_ch: tuple(_v) for _ch, _v in _r.get("channel_recalls", {}).items()},
        "gold":            _r.get("gold", 0),
        "gold_doc_ids":    _r.get("gold_doc_ids", 0),
        "R_at_K":          _r.get("R_at_K", 0.0),
    }
    for _qid, _r in _pq.items()
}
print(f"  PER_QUERY: {len(PER_QUERY)}")

# Targets, aspects, gold
ALL_TARGETS = (_j.loads((SNAPSHOT_DIR / "all_targets.json").read_text(encoding="utf-8"))
               if (SNAPSHOT_DIR / "all_targets.json").exists() else {})
ALL_HYDE_ASPECTS = (_j.loads((SNAPSHOT_DIR / "hyde_aspects.json").read_text(encoding="utf-8"))
                    if (SNAPSHOT_DIR / "hyde_aspects.json").exists() else {})
_g = _j.loads((SNAPSHOT_DIR / "gold_doc_sets.json").read_text(encoding="utf-8"))
ALL_GOLD_DOC_SET = {_qid: set(_lst) for _qid, _lst in _g.items()}
print(f"  ALL_TARGETS: {len(ALL_TARGETS)}")
print(f"  ALL_HYDE_ASPECTS: {len(ALL_HYDE_ASPECTS)}")
print(f"  ALL_GOLD_DOC_SET total: {sum(len(_s) for _s in ALL_GOLD_DOC_SET.values()):,} doc_ids")
print(f"\n[warm-boot] done in {time.time()-_t0:.1f}s")

# Per-query in-pool ceiling table (binding constraint diagnostic)
print()
print("=" * 76)
print(f"  Per-query in-pool gold (the ceiling on R for this snapshot)")
print("=" * 76)
print(f"  {'qid':<10}{'gold':>6}{'in_pool':>9}{'R_max':>8}{'pool_size':>11}")
_macro_ceiling = []
for q in ALL_QUERIES:
    qid = q["query_id"]
    topk_set = set(PER_QUERY[qid]["final_topk"])
    gold_set = ALL_GOLD_DOC_SET.get(qid, set())
    in_pool  = len(topk_set & gold_set)
    R_max    = in_pool / max(1, len(gold_set))
    _macro_ceiling.append(R_max)
    print(f"  {qid:<10}{len(gold_set):>6}{in_pool:>9}{R_max:>8.3f}{len(PER_QUERY[qid]['final_topk']):>11}")
print(f"  {'MACRO':<10}{'':>6}{'':>9}{sum(_macro_ceiling)/len(_macro_ceiling):>8.3f}")
''')


# =============================================================================
# Phase 2 - Qwen3-Reranker-8B rerank via vLLM
# =============================================================================
md(r'''# Phase 2 - Qwen3-Reranker-8B rerank (vLLM, top-5000 -> calibrated probs)

**Scoring contract** (Qwen3 TR Eq. p.4, verified against
`Qwen3-Embedding/examples/qwen3_reranker_transformers.py`):

```
score(q, d) = softmax([logit_no, logit_yes])[:, 1]
```

The output is already a calibrated probability in (0, 1).

**vLLM path**: we run `LLM.generate` with `temperature=0, max_tokens=1,
logprobs=20`. From the returned top-20 logprobs for the first generated
position we pick out the `yes` and `no` token logprobs, softmax them, and
take the `P(yes)` entry. If either token is missing from the top-20 (very
rare with `logprobs=20`), we treat its logprob as `-1e9`.

**Three non-obvious requirements**:
1. Pad prompts with the canonical Qwen3-Reranker prefix (system prompt with
   the "yes/no only" instruction) and suffix (empty `<think></think>` block).
2. Use `convert_tokens_to_ids("yes")` / `convert_tokens_to_ids("no")` - not
   `tokenizer("yes")` which BPE-tokenizes and may yield `" yes"` (leading space).
3. Truncate the body so prefix + body + suffix <= `_RRK_MAX_LEN`.

Sanity check first (gold > non-gold on val_001), then full rerank.
''')

code(r'''# Phase 2 - vLLM Qwen3-Reranker-8B (load once)
import math
import gc as _gc
import torch as _torch_s1
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Free anything left from prior cells
_gc.collect()
if _torch_s1.cuda.is_available():
    _torch_s1.cuda.empty_cache()

RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"

# Tokenizer first (cheap; always (re)load — needed for prompt assembly)
rrk_tok = AutoTokenizer.from_pretrained(
    RERANKER_MODEL, trust_remote_code=True, padding_side="left"
)

# Canonical prompt format (verbatim from Qwen3 TR p.3)
_RRK_INSTRUCT = (
    "Given an English legal question about Swiss federal law, determine "
    "whether the provided document (a Swiss law article OR a court paragraph "
    "in German / French / Italian) is a RELEVANT CITATION - meaning it "
    "answers, supports, or directly relates to the question."
)
_RRK_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Note that the answer can only be \"yes\" or "
    "\"no\".<|im_end|>\n<|im_start|>user\n"
)
_RRK_SUFFIX = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
_RRK_PREFIX_IDS = rrk_tok.encode(_RRK_PREFIX, add_special_tokens=False)
_RRK_SUFFIX_IDS = rrk_tok.encode(_RRK_SUFFIX, add_special_tokens=False)
_RRK_MAX_LEN  = 1024
# `-8` reserves room for the 1 output token + 7-token safety margin (BOS edge
# cases, multi-byte token boundaries). vLLM requires
# prompt_len + max_tokens <= max_model_len; without the margin, prompts that
# fill the body budget hit exactly max_model_len and vLLM rejects them.
_RRK_MAX_BODY = _RRK_MAX_LEN - len(_RRK_PREFIX_IDS) - len(_RRK_SUFFIX_IDS) - 8

# Yes/no token IDs - convert_tokens_to_ids, NOT tokenizer("yes")
_RRK_YES_ID = rrk_tok.convert_tokens_to_ids("yes")
_RRK_NO_ID  = rrk_tok.convert_tokens_to_ids("no")
assert _RRK_YES_ID != rrk_tok.unk_token_id, "yes not in vocab"
assert _RRK_NO_ID  != rrk_tok.unk_token_id, "no not in vocab"
print(f"[stage1] token_yes_id={_RRK_YES_ID}, token_no_id={_RRK_NO_ID}")

# Load vLLM engine — IDEMPOTENT. If `rrk_llm` already exists in globals
# (cell was re-run after a code edit), skip the reload. Re-loading would
# spawn a second engine and conflict with the alive one on GPU memory /
# multiprocessing → "Engine core initialization failed" with empty Failed
# proc dict. To force a clean reload: Runtime → Restart, re-run from Phase 0.
if "rrk_llm" not in globals() or globals().get("rrk_llm") is None:
    print(f"[stage1] loading {RERANKER_MODEL} via vLLM ...")
    _t0 = time.time()
    # vLLM 0.10 + Jupyter/Colab workaround: ipykernel's OutStream lacks
    # fileno(), which vLLM's worker subprocess calls. Restore the real
    # file-descriptor-backed streams; Colab still captures via its kernel.
    import sys as _sys_vllmfix
    _sys_vllmfix.stdout = _sys_vllmfix.__stdout__
    _sys_vllmfix.stderr = _sys_vllmfix.__stderr__
    rrk_llm = LLM(
        model=RERANKER_MODEL,
        dtype="bfloat16",
        max_model_len=_RRK_MAX_LEN,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        trust_remote_code=True,
    )
    print(f"[stage1] loaded in {time.time()-_t0:.1f}s")
else:
    print(f"[stage1] reusing existing vLLM engine (already in VRAM)")

# Sampling: greedy, one token, top-20 logprobs so we can extract yes/no
_RRK_SP = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)


def _build_body_ids(query: str, doc: str) -> list[int]:
    """Tokenize one (query, doc) pair body, truncated to fit _RRK_MAX_BODY."""
    body = f"<Instruct>: {_RRK_INSTRUCT}\n<Query>: {query}\n<Document>: {doc}"
    ids = rrk_tok.encode(body, add_special_tokens=False)
    if len(ids) > _RRK_MAX_BODY:
        ids = ids[:_RRK_MAX_BODY]
    return ids


def _rrk_score_pairs(pairs):
    """Score (query, doc) pairs via vLLM. Returns list[float] in [0,1].

    Builds full prompt token IDs (prefix + body + suffix) manually for each
    pair, passes them to vLLM as `prompt_token_ids`, then extracts the
    yes/no logprobs from the first generated step.
    """
    if not pairs:
        return []
    prompt_token_ids = [
        _RRK_PREFIX_IDS + _build_body_ids(q, d) + _RRK_SUFFIX_IDS
        for (q, d) in pairs
    ]
    # vLLM 0.10+: wrap token-id lists as TokensPrompt / dict and pass via `prompts`
    _prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
    outs = rrk_llm.generate(
        _prompts,
        sampling_params=_RRK_SP,
        use_tqdm=False,
    )
    scores = []
    for out in outs:
        # out.outputs[0].logprobs is list[dict[int, Logprob]] of length max_tokens
        lp_step = out.outputs[0].logprobs[0]
        yes_lp = lp_step.get(_RRK_YES_ID)
        no_lp  = lp_step.get(_RRK_NO_ID)
        yes_l = yes_lp.logprob if yes_lp is not None else -1e9
        no_l  = no_lp.logprob  if no_lp  is not None else -1e9
        # softmax over (no, yes), take P(yes)
        m = max(yes_l, no_l)
        e_yes = math.exp(yes_l - m)
        e_no  = math.exp(no_l  - m)
        scores.append(e_yes / (e_yes + e_no))
    return scores


# === sanity check on first query ===
print()
# STAGE1_TOP_N controls how deep into the v7.5 fusion pool we rerank.
# 50000 = use the full pool (~44k actual after channel union dedup).
# Macro recall ceiling: top-5000 = 0.728, top-50000 = 0.866 → +9pp macro F1.
# Cost: ~10x more pairs but vLLM continuous batching makes wall-time ~6-8x.
# Set to 5000 only if you want a fast first pass / debugging.
STAGE1_TOP_N = 50000
_qid0 = sorted(PER_QUERY.keys())[0]
_q0   = next(q["query_text"] for q in ALL_QUERIES if q["query_id"] == _qid0)
_pool = PER_QUERY[_qid0]["final_topk"][:STAGE1_TOP_N]
_g    = ALL_GOLD_DOC_SET[_qid0]
_gold_in_pool    = [d for d in _pool if d in _g][:3]
_nongold_in_pool = [d for d in _pool if d not in _g][:3]
if _gold_in_pool and _nongold_in_pool:
    _sanity_pairs = [(_q0, (search_text.get(d, "") or doc_meta.get(d, {}).get("citation", d))[:2000])
                     for d in _gold_in_pool + _nongold_in_pool]
    _s = _rrk_score_pairs(_sanity_pairs)
    _mg = sum(_s[:len(_gold_in_pool)]) / len(_gold_in_pool)
    _mn = sum(_s[len(_gold_in_pool):]) / len(_nongold_in_pool)
    print(f"[stage1 sanity] {_qid0}  gold_mean={_mg:.4f}  nongold_mean={_mn:.4f}  sep={_mg-_mn:+.4f}")
    if _mg <= _mn:
        raise RuntimeError(
            "Reranker is NOT separating gold from non-gold. Check token IDs / prompt."
        )
    elif _mg - _mn < 0.10:
        print(f"  [WARN] separation < 0.10 - reranker may not help much")
    else:
        print(f"  [OK] separation looks good; proceeding")

# === Full Stage 1 rerank ===
print(f"\n[stage1] reranking top-{STAGE1_TOP_N} per query for {len(PER_QUERY)} queries ...")
for q in ALL_QUERIES:
    qid   = q["query_id"]
    qtext = q["query_text"]
    cands = PER_QUERY[qid]["final_topk"][:STAGE1_TOP_N]
    pairs = []
    for did in cands:
        text = search_text.get(did, "") or doc_meta.get(did, {}).get("citation", did) or did
        pairs.append((qtext, text[:3000]))

    t_q = time.time()
    scores = _rrk_score_pairs(pairs)
    ranked = sorted(zip(cands, scores), key=lambda kv: -kv[1])
    PER_QUERY[qid]["stage1_ranked"] = ranked

    # Per-query R@K curve
    g = ALL_GOLD_DOC_SET[qid]
    g_tot = ALL_TOTAL_GOLD[qid]
    PER_QUERY[qid]["stage1_R_at_K"] = {}
    for K in (50, 100, 200, 500, 1000, 2000, 5000):
        K_use = min(K, len(ranked))
        hit = len({d for d, _ in ranked[:K_use]} & g)
        PER_QUERY[qid]["stage1_R_at_K"][K] = (hit, hit / max(1, g_tot))

    print(f"  [{qid}] {time.time()-t_q:6.1f}s  "
          f"R@50={PER_QUERY[qid]['stage1_R_at_K'][50][1]:.3f} "
          f"R@200={PER_QUERY[qid]['stage1_R_at_K'][200][1]:.3f} "
          f"R@500={PER_QUERY[qid]['stage1_R_at_K'][500][1]:.3f} "
          f"R@2000={PER_QUERY[qid]['stage1_R_at_K'][2000][1]:.3f}")

# Comparison vs fusion baseline
print()
print("=" * 84)
print(f"  Stage-1 vs fusion-baseline (macro mean R@K)")
print("=" * 84)
print(f"  {'K':>6}  {'fusion':>8}  {'stage1':>8}  {'D':>+8}  {'min':>8}  {'max':>8}")
for K in (50, 100, 200, 500, 1000, 2000, 5000):
    fus = []
    s1 = []
    for qid in PER_QUERY:
        c = PER_QUERY[qid]["curve"]
        keys = sorted(c.keys())
        kk = max((k for k in keys if k <= K), default=None)
        fus.append(c[kk][1] if kk is not None else 0.0)
        s1.append(PER_QUERY[qid]["stage1_R_at_K"][K][1])
    print(f"  {K:>6}  {sum(fus)/len(fus):>8.3f}  {sum(s1)/len(s1):>8.3f}  "
          f"{(sum(s1)-sum(fus))/len(fus):>+8.3f}  {min(s1):>8.3f}  {max(s1):>8.3f}")
''')


# =============================================================================
# Phase 2b - RRF defense (recovers gold the reranker demoted)
# =============================================================================
md(r'''# Phase 2b - RRF defense: combine reranker rank with fusion rank

The reranker can demote gold that v7.5 fusion correctly surfaced (cross-lingual
legal text is a hard domain). Standard production fix: **RRF-combine reranker
rank with original fusion rank**.

```
rrf_score(d) = 1/(60 + rerank_rank(d)) + 1/(60 + fusion_rank(d))
```

Any doc strong in EITHER signal stays high. Gold that fusion's 14 channels
liked but Qwen3-Reranker disagreed about gets a second chance.

Overwrites `PER_QUERY[qid]["stage1_ranked"]` with the RRF-defended list so
Phases 3-8 consume the better ranking transparently. Raw Qwen3-Reranker
outputs are preserved at `stage1_ranked_raw` for diagnostics and at
`rerank_scores` for phases that need the absolute probability (PSI-Rank,
Conformal).
''')

code(r'''# Phase 2b - RRF defense
# Uses RRF scores themselves (monotonic in RRF order, by construction) as the
# `stage1_ranked` score values. This keeps the variable-K estimators happy
# (they assume monotonic-decreasing scores along the list) while reordering
# the candidates to combine reranker + fusion signal.
RRF_K = 60
RRF_W_RERANK = 1.0
RRF_W_FUSION = 1.0

print("Phase 2b - RRF(stage1, fusion) defense")
print(f"{'qid':<10}{'fus_R@2k':>10}{'rrk_R@2k':>10}{'rrf_R@2k':>10}{'d_vs_rrk':>+10}{'d_vs_fus':>+10}")
for q in ALL_QUERIES:
    qid = q["query_id"]
    rerank_ranked = PER_QUERY[qid]["stage1_ranked"]
    fusion_order  = PER_QUERY[qid]["final_topk"]
    rerank_rank = {d: i for i, (d, _) in enumerate(rerank_ranked)}
    fusion_rank = {d: i for i, d in enumerate(fusion_order)}

    all_dids = set(rerank_rank) | set(fusion_rank)
    rrf = {}
    big = 10**9
    for d in all_dids:
        s  = RRF_W_RERANK * 1.0 / (RRF_K + rerank_rank.get(d, big))
        s += RRF_W_FUSION * 1.0 / (RRF_K + fusion_rank.get(d, big))
        rrf[d] = s
    # Sort by RRF descending; the RRF scores themselves are monotonic in this order.
    rrf_sorted = sorted(rrf.keys(), key=lambda d: -rrf[d])
    rrf_ranked = [(d, rrf[d]) for d in rrf_sorted]

    # Keep raw Qwen3 P(yes) per doc so PSI-Rank and Conformal can use absolute
    # probability thresholds (their pivot/quantile inputs are P(yes) values).
    rerank_score_lookup = {d: s for d, s in rerank_ranked}
    PER_QUERY[qid]["rerank_scores"]     = rerank_score_lookup
    PER_QUERY[qid]["stage1_ranked_raw"] = rerank_ranked   # raw Qwen3 order + scores
    PER_QUERY[qid]["stage1_ranked"]     = rrf_ranked       # RRF order, monotonic RRF scores

    g = ALL_GOLD_DOC_SET[qid]
    g_tot = ALL_TOTAL_GOLD[qid]
    fus_r2k = len(set(fusion_order[:2000]) & g) / max(1, g_tot)
    rrk_r2k = len({d for d, _ in rerank_ranked[:2000]} & g) / max(1, g_tot)
    rrf_r2k = len({d for d, _ in rrf_ranked[:2000]} & g) / max(1, g_tot)
    print(f"  {qid:<10}{fus_r2k:>10.3f}{rrk_r2k:>10.3f}{rrf_r2k:>10.3f}"
          f"{rrf_r2k - rrk_r2k:>+10.3f}{rrf_r2k - fus_r2k:>+10.3f}")
''')


# =============================================================================
# Phase 3 - Score-distribution diagnostics
# =============================================================================
md(r'''# Phase 3 - Score-distribution diagnostics

A quick statistical check on the rerank score distribution before running any
K-estimator. We want to see:
- Gold scores stand higher than non-gold scores (the reranker actually works)
- Scores spread across [0, 1] (not collapsed to 0/1)
- The distribution has a recognizable elbow (the variable-K methods rely on this)
''')

code(r'''# Phase 3 - Score-distribution diagnostics
import numpy as np

print(f"{'qid':<10}{'n>0.9':>8}{'n>0.5':>8}{'n>0.1':>8}{'mean':>9}{'std':>8}{'gold@top10':>12}")
for q in ALL_QUERIES:
    qid = q["query_id"]
    ranked = PER_QUERY[qid]["stage1_ranked"]
    scores = np.array([s for _, s in ranked])
    g = ALL_GOLD_DOC_SET[qid]
    top10_gold = len({d for d, _ in ranked[:10]} & g)
    print(f"  {qid:<10}{(scores>0.9).sum():>8}{(scores>0.5).sum():>8}{(scores>0.1).sum():>8}"
          f"{scores.mean():>9.4f}{scores.std():>8.4f}{top10_gold:>12}")
''')


# =============================================================================
# Phase 4 - Six variable-K estimators
# =============================================================================
md(r'''# Phase 4 - Six variable-K estimators (no fixed K, no hardcoding)

Each estimator returns a single integer K per query, computed from:
- The Stage 1 rerank score distribution, OR
- The candidate doc's own fields (paragraph_role, family, language) + hyde_aspects, OR
- A train.csv calibration quantile (Conformal only - no statute priors)

Estimators are then ensembled (median + clamp) in Phase 5.''')


# 4a Adaptive-k
md(r'''## 4a. Adaptive-k (EMNLP 2025, Taguchi/Maekawa/Bhutani)

```
sort scores descending -> s
gaps[i] = s[i] - s[i+1]
search window: i in [0, 0.9*N)   # ignore the bottom 10% noise tail
k* = argmax(gaps within window)
return k* + B                    # B = 5 buffer
```

Reference: `research_papers/variable_k_truncation/AdaptiveK_NoTuning_EMNLP2025.pdf`
Code: `research_repos/adaptive-k-retrieval/adaptive-k-retrieval/retriever.py:196`
''')

code(r'''# Phase 4a - Adaptive-k (EMNLP 2025)
import torch as _torch_ak

def adaptive_k(scores_desc, B=5, top_frac=0.9, ignore_head=0):
    """Largest-gap cutoff on descending-sorted scores. Returns int K in [1, N]."""
    if len(scores_desc) <= 2:
        return len(scores_desc)
    s = _torch_ak.as_tensor(scores_desc, dtype=_torch_ak.float32)
    n = len(s)
    head_cut = int(ignore_head) if isinstance(ignore_head, int) else int(ignore_head * (n - 1))
    tail_cut = int((1.0 - top_frac) * (n - 1))
    gaps = s[:-1] - s[1:]
    if tail_cut > 0:
        sub = gaps[head_cut:-tail_cut]
    else:
        sub = gaps[head_cut:]
    k_star = int(_torch_ak.argmax(sub).item()) + head_cut
    return min(n, max(1, k_star + 1 + B))


K_adaptive = {}
for q in ALL_QUERIES:
    qid = q["query_id"]
    scores = [s for _, s in PER_QUERY[qid]["stage1_ranked"]]
    K_adaptive[qid] = adaptive_k(scores, B=5, top_frac=0.9)

print(f"{'qid':<10}{'K_adaptive':>12}{'gold':>6}{'top_K_recall':>14}")
for qid in sorted(K_adaptive):
    K = K_adaptive[qid]
    ranked = PER_QUERY[qid]["stage1_ranked"]
    hit = len({d for d, _ in ranked[:K]} & ALL_GOLD_DOC_SET[qid])
    g_tot = ALL_TOTAL_GOLD[qid]
    print(f"  {qid:<10}{K:>12}{g_tot:>6}{hit/max(1,g_tot):>14.3f}")
''')


# 4b CAR
md(r'''## 4b. CAR - Cluster-based Adaptive Retrieval (Coinbase + USC, Nov 2025)

```
distances = 1 - scores                 # min-max normalised
X = column_stack([dist, rank_index])   # 2-D feature
for each clustering theta in a grid:
    C_theta = Cluster(X, theta)
    s_theta = silhouette_score(X, C_theta)
theta* = argmax silhouette
S = positions where C*[i] != C*[i-1]   # cluster boundaries
for each i in S:
    g_i = dist[i] - dist[i-1]
    score_i = g_i / max_g + i/N        # gap-magnitude + rank penalty
return argmax_i score_i - 1
```

Reference: `research_papers/variable_k_truncation/CAR_Cluster_Based_Adaptive_Retrieval_arXiv2511.14769.pdf`
''')

code(r'''# Phase 4b - CAR (Coinbase + USC, arXiv 2511.14769)
import numpy as _np_car
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def car_cutoff(scores_desc, window=300, k_range=range(2, 11)):
    """Cluster-based adaptive retrieval. Returns int K.

    Restricts to top-`window` candidates because silhouette grid search is
    O(N) per k. The reranker's gold density is overwhelmingly in the top few
    hundred, so 300 is sufficient context.
    """
    s = _np_car.asarray(scores_desc[:window], dtype=_np_car.float32)
    n = len(s)
    if n < 4:
        return n
    d = 1.0 - s
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-9)
    X = _np_car.column_stack([d_norm, _np_car.arange(n) / n])

    best_k, best_sil, best_labels = None, -1.0, None
    for k in k_range:
        if k >= n: break
        try:
            labels = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(X)
            if len(set(labels)) < 2: continue
            sil = silhouette_score(X, labels)
        except Exception:
            continue
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels
    if best_labels is None:
        return n

    boundaries = [i for i in range(1, n) if best_labels[i] != best_labels[i-1]]
    if not boundaries:
        return n
    gaps = [d_norm[i] - d_norm[i-1] for i in boundaries]
    gmax = max(gaps) + 1e-9
    cand_scores = [(i, gaps[j] / gmax + i / n) for j, i in enumerate(boundaries)]
    i_star, _ = max(cand_scores, key=lambda t: t[1])
    return max(1, i_star)


K_car = {}
for q in ALL_QUERIES:
    qid = q["query_id"]
    scores = [s for _, s in PER_QUERY[qid]["stage1_ranked"]]
    K_car[qid] = car_cutoff(scores, window=300)

print(f"{'qid':<10}{'K_car':>8}{'gold':>6}{'top_K_recall':>14}")
for qid in sorted(K_car):
    K = K_car[qid]
    ranked = PER_QUERY[qid]["stage1_ranked"]
    hit = len({d for d, _ in ranked[:K]} & ALL_GOLD_DOC_SET[qid])
    g_tot = ALL_TOTAL_GOLD[qid]
    print(f"  {qid:<10}{K:>8}{g_tot:>6}{hit/max(1,g_tot):>14.3f}")
''')


# 4c PSI-Rank
md(r'''## 4c. PSI-Rank-Dyn - LLM-generated pivot (arXiv 2604.09492, 2026)

```
D* = LLM.generate(prompt(Q, tau=2))   # one borderline pseudo-doc per query
s*  = reranker(Q, D*)                  # score with the SAME reranker
K   = #candidates with score > s*
```

We re-use the already-loaded reranker. For the pivot, we use the median
`hyde_aspects[qid]` paragraph (by length) - LLM-generated "marginally
relevant" text for the query, perfect as a borderline pivot.

Reference: `research_papers/variable_k_truncation/Dynamic_RLT_LLM_Pivot_arXiv2604.09492.pdf`
''')

code(r'''# Phase 4c - PSI-Rank-Dyn (substitute pivot from hyde_aspects)
def psi_rank_pivot_text(qid, hyde_aspects, fallback_query_text=""):
    aspects = hyde_aspects.get(qid, []) or []
    paragraphs = []
    for a in aspects:
        if isinstance(a, dict):
            paragraphs.append(a.get("paragraph") or a.get("answer") or str(a))
        else:
            paragraphs.append(str(a))
    if not paragraphs:
        return fallback_query_text
    paragraphs.sort(key=len)
    return paragraphs[len(paragraphs) // 2][:1500]


K_psi = {}
print(f"{'qid':<10}{'pivot_score':>13}{'K_psi':>8}{'gold':>6}{'top_K_recall':>14}")
# Build all pivot pairs once, score in one vLLM batch
_pivot_pairs = []
_qid_order = []
for q in ALL_QUERIES:
    qid = q["query_id"]
    qtext = q["query_text"]
    pivot_text = psi_rank_pivot_text(qid, ALL_HYDE_ASPECTS, fallback_query_text=qtext)
    _pivot_pairs.append((qtext, pivot_text))
    _qid_order.append(qid)
_pivot_scores = _rrk_score_pairs(_pivot_pairs)

for qid, pivot_score in zip(_qid_order, _pivot_scores):
    ranked = PER_QUERY[qid]["stage1_ranked"]
    # PSI-Rank: count how many candidates have raw Qwen3 P(yes) > pivot_score.
    # We compare against the raw rerank probability (not the RRF combined
    # score) because the pivot was scored on the same Qwen3 P(yes) scale.
    # Then we slice top-K of the RRF order so RRF-defended gold is kept.
    rerank_scores = PER_QUERY[qid].get("rerank_scores", {})
    K = sum(1 for d, _ in ranked if rerank_scores.get(d, 0.0) > pivot_score)
    K = max(1, K)
    K_psi[qid] = K
    hit = len({d for d, _ in ranked[:K]} & ALL_GOLD_DOC_SET[qid])
    g_tot = ALL_TOTAL_GOLD[qid]
    print(f"  {qid:<10}{pivot_score:>13.4f}{K:>8}{g_tot:>6}{hit/max(1,g_tot):>14.3f}")
''')


# 4d AcuRank-light
md(r'''## 4d. AcuRank-light - uncertainty stopping (NeurIPS 2025)

```
Initialise per-doc Bayesian relevance: mu_i = score_i, sigma_i = score_i / 3
For each target K_candidate:
    t(K) = threshold s.t. sum_i P(x_i > t) = K     # binary search
    uncertain_i = (tol < P(x_i > t) < 1-tol)
Pick K minimising |uncertain pool|  (most decisive K)
```

We don't run the iterative listwise reranking loop (it needs an LLM in the
loop). Instead we use the **uncertainty-decisive K**: the K where the most
candidates are clearly in/out.

Reference: `research_papers/variable_k_truncation/AcuRank_Uncertainty_Aware_Listwise_arXiv2505.18512.pdf`
''')

code(r'''# Phase 4d - AcuRank-light
from scipy.stats import norm as _norm

def acurank_light(scores_desc, K_min=3, K_max=80, tol=0.02):
    s = _np_car.asarray(scores_desc, dtype=_np_car.float32)
    mu = s
    sigma = _np_car.maximum(s / 3.0, 1e-3)
    K_max = min(K_max, len(s))

    def n_uncertain_at_K(K):
        lo, hi = 0.0, 1.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            p_above = 1.0 - _norm.cdf((mid - mu) / sigma)
            total = p_above.sum()
            if total > K:
                lo = mid
            else:
                hi = mid
        t = 0.5 * (lo + hi)
        p_above = 1.0 - _norm.cdf((t - mu) / sigma)
        return int(((p_above > tol) & (p_above < 1 - tol)).sum())

    best_K, best_count = K_min, float("inf")
    for K in range(K_min, K_max + 1, 1):
        cnt = n_uncertain_at_K(K)
        if cnt < best_count:
            best_count = cnt
            best_K = K
    return best_K


K_acurank = {}
for q in ALL_QUERIES:
    qid = q["query_id"]
    scores = [s for _, s in PER_QUERY[qid]["stage1_ranked"][:200]]
    K_acurank[qid] = acurank_light(scores)

print(f"{'qid':<10}{'K_acurank':>10}{'gold':>6}{'top_K_recall':>14}")
for qid in sorted(K_acurank):
    K = K_acurank[qid]
    ranked = PER_QUERY[qid]["stage1_ranked"]
    hit = len({d for d, _ in ranked[:K]} & ALL_GOLD_DOC_SET[qid])
    g_tot = ALL_TOTAL_GOLD[qid]
    print(f"  {qid:<10}{K:>10}{g_tot:>6}{hit/max(1,g_tot):>14.3f}")
''')


# 4e Conformal
md(r'''## 4e. Conformal calibration on train.csv (IJCAI 2025)

```
1. Score train.csv (query, gold) pairs with the SAME reranker
2. For each train query: tau_i = min reranker_score over its gold
3. tau_alpha = (|cal|*alpha)-th smallest of {tau_i}
4. At inference: K = #candidates with score > tau_alpha
```

Distribution-free probability >= 1 - alpha that all gold scores are above
tau_alpha (under exchangeability) - i.e., recall >= 1 - alpha in expectation.

Train.csv is used ONLY for the quantile (a single scalar threshold). No
per-query priors are baked in. Compatible with `feedback_train_unreliable`.

The first run scores ~1,000 train queries x ~5 gold docs each (~5 minutes on
RTX PRO 6000). Re-runs read the pickle cache.

Reference: `research_papers/variable_k_truncation/TwoStage_Risk_Control_Ranked_Retrieval_IJCAI2025.pdf`
''')

code(r'''# Phase 4e - Conformal calibration on train.csv
import pickle

# Find train.csv: prefer the granularity-expanded version if present
DATA_DIR = _DRIVE / "data"
TRAIN_CSV = DATA_DIR / "train_granularity_expanded.csv"
if not TRAIN_CSV.exists():
    TRAIN_CSV = DATA_DIR / "train.csv"
assert TRAIN_CSV.exists(), f"train.csv not found in {DATA_DIR}"

CACHE_FILE = OUT_DIR / "train_gold_scores.pkl"
ALPHA = 0.20  # 80% recall guarantee per query in expectation

def _need_compute():
    if not CACHE_FILE.exists(): return True
    return CACHE_FILE.stat().st_mtime < TRAIN_CSV.stat().st_mtime


if _need_compute():
    print(f"[conformal] computing train-gold scores from {TRAIN_CSV.name} ...")
    train_df = _pd.read_csv(TRAIN_CSV)
    # Build citation_string -> doc_id lookup
    cit_to_doc_ids = {}
    for d, m in doc_meta.items():
        cit_to_doc_ids.setdefault(m.get("citation", ""), set()).add(d)
    print(f"  cit_to_doc_ids: {len(cit_to_doc_ids):,} unique citation strings")

    train_gold_scores_min = []
    skipped = 0
    _t0 = time.time()
    # Batch all pairs across train queries for vLLM efficiency
    _batch_pairs, _batch_owner = [], []
    BATCH_FLUSH = 256

    def _flush_batch():
        if not _batch_pairs: return
        scores = _rrk_score_pairs(_batch_pairs)
        # Aggregate per train query (owner)
        cur_owner = None
        cur_scores = []
        for owner, s in zip(_batch_owner, scores):
            if cur_owner is not None and owner != cur_owner:
                train_gold_scores_min.append(min(cur_scores))
                cur_scores = []
            cur_owner = owner
            cur_scores.append(s)
        if cur_scores:
            train_gold_scores_min.append(min(cur_scores))
        _batch_pairs.clear()
        _batch_owner.clear()

    for i, row in train_df.iterrows():
        qtext = str(row["query"])
        gold_cits = [c.strip() for c in str(row["gold_citations"]).split(";") if c.strip()]
        gold_doc_ids = []
        for cit in gold_cits:
            dids = cit_to_doc_ids.get(cit, set())
            if dids:
                gold_doc_ids.append(next(iter(dids)))  # one doc per citation string
        if not gold_doc_ids:
            skipped += 1
            continue
        gold_doc_ids = gold_doc_ids[:5]  # cap at 5 per query
        for did in gold_doc_ids:
            text = (search_text.get(did, "") or doc_meta.get(did, {}).get("citation", did) or did)[:3000]
            _batch_pairs.append((qtext, text))
            _batch_owner.append(i)
        if len(_batch_pairs) >= BATCH_FLUSH:
            _flush_batch()
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(train_df)}  elapsed={time.time()-_t0:.0f}s  cached={len(train_gold_scores_min)}")
    _flush_batch()

    print(f"[conformal] resolved {len(train_gold_scores_min)} train queries, skipped {skipped}")
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(train_gold_scores_min, f)
else:
    with open(CACHE_FILE, "rb") as f:
        train_gold_scores_min = pickle.load(f)
    print(f"[conformal] loaded {len(train_gold_scores_min)} cached tau_i from {CACHE_FILE}")

sorted_tau = sorted(train_gold_scores_min)
n_cal = len(sorted_tau)
idx = max(0, min(n_cal - 1, int((n_cal + 1) * ALPHA) - 1))
tau_alpha = sorted_tau[idx]
print(f"\n[conformal] alpha={ALPHA}, tau_alpha = {tau_alpha:.4f}")
print(f"  -> at inference: keep candidates with rerank_score > {tau_alpha:.4f}")

K_conformal = {}
print(f"\n{'qid':<10}{'K_conformal':>13}{'gold':>6}{'top_K_recall':>14}")
for q in ALL_QUERIES:
    qid = q["query_id"]
    ranked = PER_QUERY[qid]["stage1_ranked"]
    # Compare raw Qwen3 P(yes) vs tau_alpha (which was learned from raw scores
    # on train.csv), not the RRF combined score.
    rerank_scores = PER_QUERY[qid].get("rerank_scores", {})
    K = sum(1 for d, _ in ranked if rerank_scores.get(d, 0.0) > tau_alpha)
    K = max(1, K)
    K_conformal[qid] = K
    hit = len({d for d, _ in ranked[:K]} & ALL_GOLD_DOC_SET[qid])
    g_tot = ALL_TOTAL_GOLD[qid]
    print(f"  {qid:<10}{K:>13}{g_tot:>6}{hit/max(1,g_tot):>14.3f}")
''')


# 4f Aspect coverage
md(r'''## 4f. Aspect coverage (legal-domain anchor)

For each `hyde_aspects[qid]` (4-7 LLM-generated sub-aspects of the query),
require that at least one candidate covers it. A candidate "covers" an aspect
if its rerank score > 0.5 AND it has substantive paragraph_role
(legal_standard, reasoning, application, holding) OR it is a law-family doc.

Returns the **smallest K** such that all aspects are covered.

Reference: `research/feature_synthesis_round2_2026-05-13.md`
''')

code(r'''# Phase 4f - Aspect coverage K
SUBSTANTIVE_ROLES = {"legal_standard", "reasoning", "application", "holding"}

_TOK = re.compile(r"[A-Za-zÀ-ſ]{4,}")
def _toks(text): return set(m.group(0).lower() for m in _TOK.finditer(text or ""))


def aspect_coverage_K(qid, ranked, hyde_aspects, doc_meta, doc_to_concepts, doc_to_terms,
                      search_text, score_floor=0.5, hard_max=200):
    aspects = hyde_aspects.get(qid, []) or []
    if not aspects:
        return 5
    aspect_kw = []
    for a in aspects:
        atxt = a.get("paragraph") if isinstance(a, dict) else str(a)
        if isinstance(a, dict) and not atxt: atxt = a.get("answer") or ""
        aspect_kw.append(_toks(atxt))

    n_aspects = len(aspect_kw)
    covered = [False] * n_aspects
    K = 0
    for K_pos, (did, score) in enumerate(ranked, start=1):
        if K_pos > hard_max: break
        if score < score_floor:
            m = doc_meta.get(did, {})
            if m.get("family") != "law":
                continue
        m = doc_meta.get(did, {})
        role = m.get("paragraph_role", "")
        if m.get("family") == "court" and role not in SUBSTANTIVE_ROLES and role != "facts":
            continue
        text = (search_text.get(did, "") or "")[:1500]
        cand_kw = _toks(text.lower()) | _toks(
            " ".join(doc_to_concepts.get(did, set())) + " " + " ".join(doc_to_terms.get(did, set()))
        )
        for i in range(n_aspects):
            if covered[i]: continue
            ovl = len(aspect_kw[i] & cand_kw)
            if ovl >= 2:
                covered[i] = True
        if all(covered):
            K = K_pos
            break
    return K or hard_max


K_aspect = {}
print(f"{'qid':<10}{'K_aspect':>10}{'n_aspects':>11}{'gold':>6}{'top_K_recall':>14}")
for q in ALL_QUERIES:
    qid = q["query_id"]
    ranked = PER_QUERY[qid]["stage1_ranked"]
    K = aspect_coverage_K(
        qid, ranked, ALL_HYDE_ASPECTS, doc_meta, _doc_to_concepts, _doc_to_terms,
        search_text, score_floor=0.5, hard_max=200,
    )
    K_aspect[qid] = K
    n_asp = len(ALL_HYDE_ASPECTS.get(qid, []))
    hit = len({d for d, _ in ranked[:K]} & ALL_GOLD_DOC_SET[qid])
    g_tot = ALL_TOTAL_GOLD[qid]
    print(f"  {qid:<10}{K:>10}{n_asp:>11}{g_tot:>6}{hit/max(1,g_tot):>14.3f}")
''')


# =============================================================================
# Phase 5 - Ensemble + clamp
# =============================================================================
md(r'''# Phase 5 - Ensemble + asymmetric clamp

```
K_hat(q) = median( K_adaptive, K_car, K_psi, K_acurank, K_conformal )
K_final(q) = clip(
    K_hat,
    lower = max(K_aspect, n_hyde_aspects, 3),     # never under-predict
    upper = count of candidates with score > 0.05
)
```
''')

code(r'''# Phase 5 - Ensemble + asymmetric clamp
import statistics as _stats

K_final = {}
K_estimators = {
    "adaptive-k":  K_adaptive,
    "car":         K_car,
    "psi-rank":    K_psi,
    "acurank":     K_acurank,
    "conformal":   K_conformal,
    "aspect":      K_aspect,
}

print(f"{'qid':<10}" + "".join(f"{name[:9]:>10}" for name in K_estimators) +
      f"{'med':>6}{'clamp_lo':>10}{'clamp_hi':>10}{'K_final':>9}")

for q in ALL_QUERIES:
    qid = q["query_id"]
    Ks_for_median = [K_adaptive[qid], K_car[qid], K_psi[qid], K_acurank[qid], K_conformal[qid]]
    K_median = int(_stats.median(Ks_for_median))

    n_asp = len(ALL_HYDE_ASPECTS.get(qid, []))
    lower = max(K_aspect[qid], n_asp, 3)

    ranked = PER_QUERY[qid]["stage1_ranked"]
    upper = sum(1 for _, s in ranked if s > 0.05)
    upper = max(upper, lower)

    K_f = min(upper, max(lower, K_median))
    K_final[qid] = K_f

    row = "".join(f"{K_estimators[name][qid]:>10}" for name in K_estimators)
    print(f"  {qid:<10}" + row + f"{K_median:>6}{lower:>10}{upper:>10}{K_f:>9}")
''')


# =============================================================================
# Phase 6 - Final answer sets
# =============================================================================
md(r'''# Phase 6 - Build final answer sets''')

code(r'''# Phase 6 - Build final answer sets
answers = {}
for q in ALL_QUERIES:
    qid = q["query_id"]
    K = K_final[qid]
    ranked = PER_QUERY[qid]["stage1_ranked"]
    answers[qid] = [d for d, _ in ranked[:K]]

print(f"{'qid':<10}{'K_final':>9}{'gold_total':>12}{'in_pool':>10}{'tp':>5}{'fp':>5}{'fn':>5}")
for q in ALL_QUERIES:
    qid = q["query_id"]
    ans = answers[qid]
    g   = ALL_GOLD_DOC_SET[qid]
    in_pool = len(set(PER_QUERY[qid]["final_topk"]) & g)
    tp = len(set(ans) & g)
    fp = len(ans) - tp
    fn = len(g) - tp
    print(f"  {qid:<10}{len(ans):>9}{len(g):>12}{in_pool:>10}{tp:>5}{fp:>5}{fn:>5}")
''')


# =============================================================================
# Phase 7 - (Optional) Evidence-quote verifier
# =============================================================================
md(r'''# Phase 7 - (Optional) Qwen3-32B evidence-quote verifier

For each candidate in the final answer set, ask Qwen3-32B to provide a
**verbatim substring of the doc text** that supports relevance. If the
quoted substring is NOT found in the original doc text (regex-checked), the
candidate fails the verifier. We then **swap** the failed candidate with the
next-highest scored Stage-1 candidate not yet in the answer set - this
preserves K.

Costs ~30 min on RTX PRO 6000. Set `RUN_EVIDENCE_VERIFIER = True` to enable.
The verifier loads Qwen3-32B via vLLM in a *separate* engine (after freeing
the reranker), so do this only if memory allows.

Reference: precision_v1.ipynb Stage D.
''')

code(r'''# Phase 7 - Optional evidence-quote verifier (DISABLED by default)
RUN_EVIDENCE_VERIFIER = False

if RUN_EVIDENCE_VERIFIER:
    print("[verifier] freeing Qwen3-Reranker-8B engine to make room for Qwen3-32B ...")
    try:
        del rrk_llm
    except NameError:
        pass
    _gc.collect()
    if _torch_s1.cuda.is_available():
        _torch_s1.cuda.empty_cache()
    # See notebooks/swiss_citation_precision_v1.ipynb Stage D for the prompt
    # template and parsing code. Port the same vLLM-based scoring loop here.
    print("[verifier] port the Stage-D prompt from precision_v1.ipynb here.")
else:
    print("[verifier] SKIPPED (set RUN_EVIDENCE_VERIFIER = True to enable)")
''')


# =============================================================================
# Phase 8 - Evaluation
# =============================================================================
md(r'''# Phase 8 - Evaluation

Macro / micro precision, recall, F1 over the 10 val queries + ablation table
comparing each K-estimator's standalone F1 vs the ensemble.''')

code(r'''# Phase 8 - Evaluation
def prf(predicted_ids, gold_ids):
    p_set = set(predicted_ids); g_set = set(gold_ids)
    tp = len(p_set & g_set); fp = len(p_set) - tp; fn = len(g_set) - tp
    P = tp / max(1, len(p_set))
    R = tp / max(1, len(g_set))
    F1 = 2 * P * R / max(1e-9, P + R) if (P + R) > 0 else 0.0
    return P, R, F1, tp, fp, fn


print("=" * 80)
print("  FINAL ANSWER SETS - per-query P/R/F1")
print("=" * 80)
print(f"  {'qid':<10}{'K':>5}{'gold':>5}{'tp':>4}{'fp':>4}{'fn':>4}{'P':>9}{'R':>9}{'F1':>9}")
macro = {"P": [], "R": [], "F1": []}
total_tp = total_fp = total_fn = 0
for q in ALL_QUERIES:
    qid = q["query_id"]
    P, R, F1, tp, fp, fn = prf(answers[qid], ALL_GOLD_DOC_SET[qid])
    macro["P"].append(P); macro["R"].append(R); macro["F1"].append(F1)
    total_tp += tp; total_fp += fp; total_fn += fn
    print(f"  {qid:<10}{K_final[qid]:>5}{ALL_TOTAL_GOLD[qid]:>5}{tp:>4}{fp:>4}{fn:>4}"
          f"{P:>9.4f}{R:>9.4f}{F1:>9.4f}")

macro_P = sum(macro["P"])/len(macro["P"])
macro_R = sum(macro["R"])/len(macro["R"])
macro_F1 = sum(macro["F1"])/len(macro["F1"])
micro_P = total_tp / max(1, total_tp + total_fp)
micro_R = total_tp / max(1, total_tp + total_fn)
micro_F1 = 2 * micro_P * micro_R / max(1e-9, micro_P + micro_R)
print(f"\n  MACRO   P={macro_P:.4f}  R={macro_R:.4f}  F1={macro_F1:.4f}")
print(f"  MICRO   P={micro_P:.4f}  R={micro_R:.4f}  F1={micro_F1:.4f}")

# Ablation
print()
print("=" * 80)
print(f"  ABLATION - macro F1 if we used each estimator's K alone (no clamp)")
print("=" * 80)
ablation = {}
for name, K_est in K_estimators.items():
    f1s = []
    for q in ALL_QUERIES:
        qid = q["query_id"]
        K = K_est[qid]
        ranked = PER_QUERY[qid]["stage1_ranked"]
        ans = [d for d, _ in ranked[:K]]
        _, _, F1, *_ = prf(ans, ALL_GOLD_DOC_SET[qid])
        f1s.append(F1)
    ablation[name] = sum(f1s) / len(f1s)
    print(f"  {name:<12}  macro_F1 = {ablation[name]:.4f}")
print(f"  {'final-clamp':<12}  macro_F1 = {macro_F1:.4f}  <- ensemble + clamp")

# Save
_save = {
    "macro_P": macro_P, "macro_R": macro_R, "macro_F1": macro_F1,
    "micro_P": micro_P, "micro_R": micro_R, "micro_F1": micro_F1,
    "K_final": K_final,
    "K_estimators": {name: K_est for name, K_est in K_estimators.items()},
    "ablation": ablation,
    "answers": answers,
}
(OUT_DIR / "final_results.json").write_text(json.dumps(_save, indent=2), encoding="utf-8")
print(f"\n[save] results -> {OUT_DIR / 'final_results.json'}")

predictions = {}
for q in ALL_QUERIES:
    qid = q["query_id"]
    cits = []
    seen = set()
    for did in answers[qid]:
        cit = doc_meta.get(did, {}).get("citation", "")
        if cit and cit not in seen:
            cits.append(cit); seen.add(cit)
    predictions[qid] = cits
(OUT_DIR / "predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
print(f"[save] predictions -> {OUT_DIR / 'predictions.json'}")
''')


# Notes
md(r'''# Notes for further improvement

If macro F1 < 0.6 after this notebook:

1. **Widen the snapshot pool** - re-run v7.5 with `topk_final = 100000`.
   val_002/val_003 each have 8-11 gold not in the 50k pool. Pool recall is the
   binding constraint there.

2. **Enable the evidence-quote verifier** (Phase 7) - port Stage-D from
   `precision_v1.ipynb`. Costs ~30 min but typically lifts F1 by +0.03-0.07.

3. **Tune `ALPHA` for Phase 4e (conformal)** - alpha=0.20 targets 80% recall.
   Try alpha=0.10 (90% recall, larger K) or alpha=0.30 (70% recall, smaller K)
   and see which the median tracks.

4. **Add a Stage 1b listwise rerank** using `castorini/rank_llm` on the top-200
   of Stage 1 (the local repo at `research_repos/rank_llm/`).

5. **Implement the Missing Link GNN** (research_repos/missing_link/) for an
   additional graph-aware score channel. ~1 hour of training on RTX PRO 6000.

All grounded by 2025-2026 papers; see `research_papers/INDEX.md`.
''')


# =============================================================================
# Write the notebook
# =============================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells, {OUT.stat().st_size/1024:.1f} KB)")
