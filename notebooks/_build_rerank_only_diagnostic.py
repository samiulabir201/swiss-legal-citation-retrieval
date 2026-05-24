"""Build the rerank-only diagnostic notebook.

Goal: prove (or disprove) that PURE cross-encoder reranking can hit R@2000 >= 0.7
on the 3 hardest val queries — no RRF defense, no ensembles, no variable-K tricks.

3 hardest val queries (by fusion R@2000):
  val_003: 47 gold, 36 in pool (R_max=0.766), fusion R@2k = 0.255  -- HARDEST
  val_001: 42 gold, 39 in pool (R_max=0.929), fusion R@2k = 0.381
  val_010: 25 gold, 22 in pool (R_max=0.880), fusion R@2k = 0.520

4 reranker configurations to test (one at a time; load -> score -> free):
  A. Qwen3-Reranker-8B  baseline  (raw text + original instruction)
  B. Qwen3-Reranker-8B  FIXED     (enriched doc + cross-lingual instruction)
  C. BGE-reranker-v2-m3           (enriched doc; 568M; multilingual-first; hard-neg trained)
  D. jina-reranker-v2-base-mul    (enriched doc; 278M; cross-lingual specialist)

Each config scores the FULL fusion top-50k for each query, then we compute
R@500 / R@1000 / R@2000.

Hardware budget on RTX PRO 6000 Blackwell:
  Qwen3-8B  ~12 min/query x 3 queries x 2 configs = ~72 min
  BGE-v2-m3 ~3-5 min/query x 3 = ~12 min
  jina-v2   ~2-3 min/query x 3 = ~9 min
  Total: ~90 minutes for the complete diagnostic.

Output: swiss_citation_rerank_only_diagnostic.ipynb
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).parent / "swiss_citation_rerank_only_diagnostic.ipynb"
cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(keepends=True)})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(keepends=True)})


# =============================================================================
# Title + plan
# =============================================================================
md(r'''# Rerank-Only Diagnostic — can reranking alone reach R@2000 >= 0.7?

**Goal**: prove or disprove that pure cross-encoder reranking on the v7.5
top-50k pool can hit **R@2000 >= 0.7** on the 3 hardest val queries — no RRF
defense, no ensembles, no variable-K tricks, no post-processing. Just:

```
rerank top-50k -> take top-2000 -> measure recall
```

**The 3 hardest val queries** (by fusion R@2000):

| qid | gold | in pool | R_max (pool ceiling) | fusion R@2k | gold needed for R=0.7 |
|---|---:|---:|---:|---:|---:|
| val_003 | 47 | 36 | 0.766 | **0.255** | 33  (= 92% of in-pool) |
| val_001 | 42 | 39 | 0.929 | **0.381** | 30  (= 77% of in-pool) |
| val_010 | 25 | 22 | 0.880 | **0.520** | 18  (= 82% of in-pool) |

**4 configurations under test** (run sequentially; load one model at a time):

| # | Model | Doc representation | Instruction |
|---|---|---|---|
| A | Qwen3-Reranker-8B | raw `search_text[:3000]` | original Swiss-legal phrasing |
| B | Qwen3-Reranker-8B | **enriched** (cit + statutes + concepts_en + role + text) | **cross-lingual** (explicit query-English / doc-DE/FR/IT framing) |
| C | BAAI/bge-reranker-v2-m3 | enriched | "Represent this sentence for searching relevant passages: " |
| D | jinaai/jina-reranker-v2-base-multilingual | enriched | (model has no instruction support; doc-only) |

**What we conclude**:

- If **any** config hits **R@2000 >= 0.7 on all three queries** -> reranking alone IS sufficient at top-2000; use that config in production.
- If config B/C/D hit it on **val_001 and val_010 but not val_003** -> reranking works, but val_003 needs pool widening (`topk_final=100k`) or other support.
- If **no** config beats fusion baseline -> reranker is structurally the wrong tool for Swiss legal cross-lingual; fall back to fusion or fine-tune.

Hardware budget (RTX PRO 6000 Blackwell, 96 GB VRAM): ~90 min for the full diagnostic.
''')


# =============================================================================
# Phase 0 - Setup
# =============================================================================
md(r'''# Phase 0 — Setup

Same Colab/local + vLLM/FlashInfer install pattern as `swiss_citation_precision_v1_(3).ipynb`.
First install will SystemExit; restart runtime and re-run.''')

code(r'''# Phase 0 - Setup: imports + path resolution + vLLM/FlashInfer install
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
print(f"Environment: {ENV}")

# Snapshot path: try canonical first, fall back to doubled-nested
_snap_a = _DRIVE / "research" / "anchor_funnel_val001_v7" / "snapshot"
_snap_b = _DRIVE / "research" / "anchor_funnel_val001_v7" / "snapshot" / "snapshot"
SNAPSHOT_DIR = _snap_a if (_snap_a / "config.json").exists() else _snap_b
OUT_DIR = _DRIVE / "research" / "rerank_only_diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)
assert SNAPSHOT_DIR.exists() and (SNAPSHOT_DIR / "config.json").exists(), \
    f"Snapshot dir missing: {SNAPSHOT_DIR}"
print(f"snapshot: {SNAPSHOT_DIR}")
print(f"out:      {OUT_DIR}")

# vLLM + FlashInfer install (once)
_NEEDS_RESTART = False
if importlib.util.find_spec("vllm") is None:
    print("\n[setup] installing vllm + transformers ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "vllm>=0.10.0", "transformers>=4.51.0",
                    "accelerate", "safetensors", "huggingface_hub"], check=True)
    _NEEDS_RESTART = True

if importlib.util.find_spec("flashinfer") is None:
    print("[setup] installing FlashInfer (CUDA-version-aware)...")
    def _cuda_suffix():
        try:
            import torch as _t
            cuda = (_t.version.cuda or "").strip()
            if not cuda: return None
            major, minor = map(int, cuda.split(".")[:2])
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
        _NEEDS_RESTART = True
    except Exception as e:
        print(f"  flashinfer install skipped ({e})")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if _NEEDS_RESTART:
    print("\n" + "=" * 64)
    print("RESTART RUNTIME and re-run from this cell.")
    print("=" * 64)
    raise SystemExit("Restart runtime and re-run from this cell.")
else:
    print("\n[setup] vLLM + FlashInfer installed; proceeding.")
''')


# =============================================================================
# Phase 1 - Warm-boot
# =============================================================================
md(r'''# Phase 1 — Warm-boot from snapshot

Loads only what we need for reranking: `search_text`, `doc_meta`,
`doc_statute_anchors`, `_doc_to_concepts`, `_doc_to_terms`, `PER_QUERY`,
`ALL_GOLD_DOC_SET`, `ALL_QUERIES`, `ALL_TOTAL_GOLD`. Skip aspects/targets
(not needed for pure rerank diagnostic).''')

code(r'''# Phase 1 - Warm-boot
import pandas as _pd
import json as _j, gzip as _gz

print(f"[warm-boot] loading from {SNAPSHOT_DIR}")
_t0 = time.time()
CONFIG = _j.loads((SNAPSHOT_DIR / "config.json").read_text(encoding="utf-8"))

# val.csv
_paths = _j.loads((SNAPSHOT_DIR / "paths.json").read_text(encoding="utf-8"))
VAL_CSV = Path(_paths["val_csv"])
if not VAL_CSV.exists():
    VAL_CSV = _DRIVE / "data" / "val.csv"
val_df = _pd.read_csv(VAL_CSV)
ALL_QUERIES = [
    {"query_id": str(r["query_id"]), "query_text": str(r["query"]),
     "gold": [c.strip() for c in str(r["gold_citations"]).split(";") if c.strip()]}
    for _, r in val_df.iterrows()
]
ALL_TOTAL_GOLD = {q["query_id"]: len(q["gold"]) for q in ALL_QUERIES}
print(f"  ALL_QUERIES: {len(ALL_QUERIES)}")

# Corpus
with _gz.open(SNAPSHOT_DIR / "corpus_snapshot.json.gz", "rt", encoding="utf-8") as f:
    _c = _j.load(f)
search_text         = {d: r["ct"] for d, r in _c.items()}
doc_meta            = {d: {"citation": r["cit"], "family": r["fam"],
                            "court_base": r["cb"], "paragraph_role": r["pr"],
                            "language": r["ln"]} for d, r in _c.items()}
doc_statute_anchors = {d: set(r["sa"]) for d, r in _c.items()}
_doc_to_concepts    = {d: set(r["cn"]) for d, r in _c.items()}
_doc_to_terms       = {d: set(r["tm"]) for d, r in _c.items()}
del _c

# PER_QUERY
_pq = _j.loads((SNAPSHOT_DIR / "per_query_snapshot.json").read_text(encoding="utf-8"))
PER_QUERY = {qid: {"final_topk": r["final_topk"],
                    "curve": {int(k): tuple(v) for k, v in r.get("curve", {}).items()}}
             for qid, r in _pq.items()}
_g = _j.loads((SNAPSHOT_DIR / "gold_doc_sets.json").read_text(encoding="utf-8"))
ALL_GOLD_DOC_SET = {qid: set(lst) for qid, lst in _g.items()}
print(f"  PER_QUERY: {len(PER_QUERY)}  doc_meta: {len(doc_meta):,}")
print(f"[warm-boot] done in {time.time()-_t0:.1f}s")
''')


# =============================================================================
# Phase 2 - Define test set + doc enrichment
# =============================================================================
md(r'''# Phase 2 — Define hardest queries + enriched doc representation

We test 3 queries and compare 2 doc representations (raw vs enriched).
Enrichment prepends: Citation, family/role, Statute anchors, Concepts (English),
Terms — then the body text truncated to 2000 chars.

The English `concepts` field is the **cross-lingual bridge** — these are
LLM-tagged English concepts attached to each DE/FR/IT doc, giving the
reranker English tokens to match against the English query.''')

code(r'''# Phase 2 - test set + doc representations
TEST_QIDS = ["val_001", "val_003", "val_010"]
STAGE1_TOP_N = 50000   # rerank the full v7.5 pool — no shortcuts
K_REPORT = (50, 100, 200, 500, 1000, 2000, 5000)

# Sanity: print per-query targets
print(f"{'qid':<10}{'gold':>6}{'in_pool':>9}{'R_max':>8}"
      f"{'fus_R@2k':>10}{'gold_needed_for_0.7':>22}")
for qid in TEST_QIDS:
    g = ALL_GOLD_DOC_SET[qid]
    pool_set = set(PER_QUERY[qid]["final_topk"])
    in_pool = len(g & pool_set)
    R_max = in_pool / max(1, len(g))
    fus_r2k = PER_QUERY[qid]["curve"].get(2000, (0, 0))[1]
    need = int(0.7 * len(g)) + 1
    pct_of_pool = need / max(1, in_pool) * 100
    print(f"{qid:<10}{len(g):>6}{in_pool:>9}{R_max:>8.3f}"
          f"{fus_r2k:>10.3f}{need:>12} ({pct_of_pool:5.1f}% of in-pool)")

def _join_short(values, max_items=8, max_chars=240):
    return ", ".join(list(values)[:max_items])[:max_chars]

def repr_raw(did):
    """Baseline doc representation — raw text only."""
    text = search_text.get(did, "") or doc_meta.get(did, {}).get("citation", did) or did
    return text[:3000]

def repr_enriched(did):
    """Enriched doc representation — cit + statutes + concepts (English) + role + text.
    The English concepts are the cross-lingual bridge tokens."""
    m     = doc_meta.get(did, {})
    cit   = (m.get("citation") or "").strip()
    fam   = m.get("family") or "?"
    role  = m.get("paragraph_role") or ""
    body  = (search_text.get(did, "") or "").strip()
    sa    = sorted(doc_statute_anchors.get(did, set()))
    concs = sorted(_doc_to_concepts.get(did, set()))
    terms = sorted(_doc_to_terms.get(did, set()))
    parts = []
    if cit:    parts.append(f"Citation: {cit}")
    parts.append(f"Type: {fam} ({role})" if role else f"Type: {fam}")
    if sa:     parts.append(f"Statute anchors: {_join_short(sa)}")
    if concs:  parts.append(f"Concepts (English): {_join_short(concs)}")
    if terms:  parts.append(f"Terms: {_join_short(terms, max_items=6, max_chars=180)}")
    parts.append(f"Text: {body[:2000]}")
    return "\n".join(parts)[:3000]


# Storage for results: per (config, qid) -> R@K dict
RESULTS = {}  # {"A_qwen3_raw": {"val_001": {500: 0.19, 2000: 0.38}, ...}, ...}
''')


# =============================================================================
# Phase 3 - Config A: Qwen3-Reranker-8B baseline (raw)
# =============================================================================
md(r'''# Phase 3 — Config A: Qwen3-Reranker-8B baseline (raw text, original instruction)

This is the **current Phase 2 behavior** — the reference baseline. Reranks the
full top-50k pool for each of the 3 queries.

Expected wall-time: ~12 min/query × 3 queries = ~36 min on RTX PRO 6000 Blackwell.''')

code(r'''# Phase 3 - Config A: Qwen3-Reranker-8B BASELINE
import math
import torch as _torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

QWEN3_MODEL = "Qwen/Qwen3-Reranker-8B"

# Load Qwen3-Reranker-8B (idempotent)
if "qwen3_llm" not in globals() or globals().get("qwen3_llm") is None:
    print(f"[A] loading {QWEN3_MODEL} via vLLM ...")
    _t0 = time.time()
    qwen3_tok = AutoTokenizer.from_pretrained(QWEN3_MODEL, trust_remote_code=True, padding_side="left")
    # vLLM 0.10 + Jupyter/Colab workaround: ipykernel's OutStream lacks
    # fileno(). Swap to real fd-backed streams during LLM init, then RESTORE
    # so subsequent print() calls remain visible in the cell output.
    import sys as _sys_vllmfix
    _orig_stdout, _orig_stderr = _sys_vllmfix.stdout, _sys_vllmfix.stderr
    _sys_vllmfix.stdout = _sys_vllmfix.__stdout__
    _sys_vllmfix.stderr = _sys_vllmfix.__stderr__
    try:
        qwen3_llm = LLM(
            model=QWEN3_MODEL, dtype="bfloat16",
            max_model_len=1024, gpu_memory_utilization=0.85, enforce_eager=False,
            trust_remote_code=True,
        )
    finally:
        _sys_vllmfix.stdout = _orig_stdout
        _sys_vllmfix.stderr = _orig_stderr
    print(f"[A] loaded in {time.time()-_t0:.1f}s", flush=True)
else:
    print(f"[A] reusing loaded qwen3_llm", flush=True)

QWEN3_YES_ID = qwen3_tok.convert_tokens_to_ids("yes")
QWEN3_NO_ID  = qwen3_tok.convert_tokens_to_ids("no")

QWEN3_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Note that the answer can only be \"yes\" or "
    "\"no\".<|im_end|>\n<|im_start|>user\n"
)
QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN3_PREFIX_IDS = qwen3_tok.encode(QWEN3_PREFIX, add_special_tokens=False)
QWEN3_SUFFIX_IDS = qwen3_tok.encode(QWEN3_SUFFIX, add_special_tokens=False)
QWEN3_MAX_LEN  = 1024
QWEN3_MAX_BODY = QWEN3_MAX_LEN - len(QWEN3_PREFIX_IDS) - len(QWEN3_SUFFIX_IDS) - 8

QWEN3_SP = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

INSTR_BASELINE = (
    "Given an English legal question about Swiss federal law, determine "
    "whether the provided document (a Swiss law article OR a court paragraph "
    "in German / French / Italian) is a RELEVANT CITATION - meaning it "
    "answers, supports, or directly relates to the question."
)

def _qwen3_score(query, dids, instruction, repr_fn, show_tqdm=True):
    """Score (query, did) pairs via Qwen3-Reranker via vLLM.

    show_tqdm=True prints a vLLM progress bar so you can see scoring is alive
    (without it, the cell goes silent for 5-15 minutes per query).
    """
    prompt_ids = []
    for d in dids:
        body = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {repr_fn(d)}"
        ids = qwen3_tok.encode(body, add_special_tokens=False)
        if len(ids) > QWEN3_MAX_BODY: ids = ids[:QWEN3_MAX_BODY]
        prompt_ids.append(QWEN3_PREFIX_IDS + ids + QWEN3_SUFFIX_IDS)
    _prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]
    outs = qwen3_llm.generate(_prompts, sampling_params=QWEN3_SP, use_tqdm=show_tqdm)
    scores = []
    for out in outs:
        lp = out.outputs[0].logprobs[0]
        y = lp.get(QWEN3_YES_ID)
        n = lp.get(QWEN3_NO_ID)
        yl = y.logprob if y else -1e9
        nl = n.logprob if n else -1e9
        m = max(yl, nl)
        e_y = math.exp(yl - m); e_n = math.exp(nl - m)
        scores.append(e_y / (e_y + e_n))
    return scores


CONFIG_NAME = "A_qwen3_raw"
print(f"\n[{CONFIG_NAME}] reranking top-{STAGE1_TOP_N} for {TEST_QIDS} ...")
RESULTS[CONFIG_NAME] = {}
for qid in TEST_QIDS:
    qtext = next(q["query_text"] for q in ALL_QUERIES if q["query_id"] == qid)
    cands = PER_QUERY[qid]["final_topk"][:STAGE1_TOP_N]
    print(f"  [{qid}] starting ({len(cands)} candidates) ...", flush=True)
    t_q = time.time()
    scores = _qwen3_score(qtext, cands, INSTR_BASELINE, repr_raw)
    ranked = sorted(zip(cands, scores), key=lambda kv: -kv[1])
    g = ALL_GOLD_DOC_SET[qid]
    g_tot = ALL_TOTAL_GOLD[qid]
    R = {}
    for K in K_REPORT:
        K_use = min(K, len(ranked))
        R[K] = len({d for d, _ in ranked[:K_use]} & g) / max(1, g_tot)
    RESULTS[CONFIG_NAME][qid] = R
    print(f"  [{qid}] {time.time()-t_q:6.1f}s  "
          f"R@50={R[50]:.3f} R@200={R[200]:.3f} R@500={R[500]:.3f} "
          f"R@1000={R[1000]:.3f} R@2000={R[2000]:.3f}  "
          + ("PASS" if R[2000] >= 0.7 else "fail"), flush=True)
''')


# =============================================================================
# Phase 4 - Config B: Qwen3-Reranker-8B FIXED (enriched + cross-lingual)
# =============================================================================
md(r'''# Phase 4 — Config B: Qwen3-Reranker-8B FIXED

Same model, two changes:
1. **Doc representation**: `repr_enriched` instead of `repr_raw`
2. **Instruction**: cross-lingual aware (explicit query-English / doc-DE-FR-IT framing)

Engine reused — no reload. Same ~36 min budget.''')

code(r'''# Phase 4 - Config B: Qwen3-Reranker-8B FIXED
INSTR_CROSSLING = (
    "The Query is an English question about Swiss federal law. The Document is "
    "a Swiss federal law article (German) or a Swiss court paragraph (German, "
    "French, or Italian) — it may include English-language metadata (citation, "
    "statutes referenced, legal concepts). Decide YES if the Document is a "
    "relevant citation for the Query — i.e., the Query's legal question can be "
    "answered, supported, or directly discussed by this Document. Decide NO "
    "otherwise. Treat language differences as a translation problem, not a "
    "mismatch."
)

CONFIG_NAME = "B_qwen3_fixed"
print(f"\n[{CONFIG_NAME}] reranking top-{STAGE1_TOP_N} for {TEST_QIDS} ...")
RESULTS[CONFIG_NAME] = {}
for qid in TEST_QIDS:
    qtext = next(q["query_text"] for q in ALL_QUERIES if q["query_id"] == qid)
    cands = PER_QUERY[qid]["final_topk"][:STAGE1_TOP_N]
    print(f"  [{qid}] starting ({len(cands)} candidates) ...", flush=True)
    t_q = time.time()
    scores = _qwen3_score(qtext, cands, INSTR_CROSSLING, repr_enriched)
    ranked = sorted(zip(cands, scores), key=lambda kv: -kv[1])
    g = ALL_GOLD_DOC_SET[qid]
    g_tot = ALL_TOTAL_GOLD[qid]
    R = {}
    for K in K_REPORT:
        K_use = min(K, len(ranked))
        R[K] = len({d for d, _ in ranked[:K_use]} & g) / max(1, g_tot)
    RESULTS[CONFIG_NAME][qid] = R
    print(f"  [{qid}] {time.time()-t_q:6.1f}s  "
          f"R@50={R[50]:.3f} R@200={R[200]:.3f} R@500={R[500]:.3f} "
          f"R@1000={R[1000]:.3f} R@2000={R[2000]:.3f}  "
          + ("PASS" if R[2000] >= 0.7 else "fail"), flush=True)
''')


# =============================================================================
# Phase 5 - Free Qwen3, load BGE-reranker-v2-m3
# =============================================================================
md(r'''# Phase 5 — Free Qwen3, load BGE-reranker-v2-m3

BGE-reranker-v2-m3 is 568M params, ~1.5 GB BF16. Multilingual-first
training with hard negatives across 100+ languages.

To make room (Qwen3 holds ~30 GB with KV cache), we destroy the vLLM engine
first. This is permanent within the kernel — to test more Qwen3 configs after,
restart the runtime.''')

code(r'''# Phase 5 - Free Qwen3, load BGE-reranker-v2-m3
print("[teardown] freeing Qwen3-Reranker-8B vLLM engine ...")
try:
    # Stop vLLM workers cleanly if available
    if "qwen3_llm" in globals() and qwen3_llm is not None:
        del qwen3_llm
    if "qwen3_tok" in globals():
        del qwen3_tok
except Exception:
    pass
gc.collect()
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"  CUDA mem after free = {torch.cuda.memory_allocated()/1024**3:.1f} GB")

BGE_MODEL = "BAAI/bge-reranker-v2-m3"
print(f"\n[bge] loading {BGE_MODEL} ...")
_t0 = time.time()
from transformers import AutoModelForSequenceClassification, AutoTokenizer
bge_tok = AutoTokenizer.from_pretrained(BGE_MODEL)
bge_mod = AutoModelForSequenceClassification.from_pretrained(
    BGE_MODEL, dtype=torch.bfloat16
).cuda().eval()
print(f"[bge] loaded in {time.time()-_t0:.1f}s")
BGE_MAX_LEN = 1024
BGE_BATCH   = 64   # 568M model can handle large batches on RTX PRO 6000

def _bge_score(query, dids, repr_fn, max_len=BGE_MAX_LEN, batch_size=BGE_BATCH):
    scores = []
    pairs = [(query, repr_fn(d)) for d in dids]
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        with torch.no_grad():
            inp = bge_tok(batch, padding=True, truncation=True,
                          max_length=max_len, return_tensors="pt")
            inp = {k: v.cuda() for k, v in inp.items()}
            logits = bge_mod(**inp).logits.view(-1).float()
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    return scores

CONFIG_NAME = "C_bge_v2_m3"
print(f"\n[{CONFIG_NAME}] reranking top-{STAGE1_TOP_N} for {TEST_QIDS} ...")
RESULTS[CONFIG_NAME] = {}
for qid in TEST_QIDS:
    qtext = next(q["query_text"] for q in ALL_QUERIES if q["query_id"] == qid)
    cands = PER_QUERY[qid]["final_topk"][:STAGE1_TOP_N]
    t_q = time.time()
    scores = _bge_score(qtext, cands, repr_enriched)
    ranked = sorted(zip(cands, scores), key=lambda kv: -kv[1])
    g = ALL_GOLD_DOC_SET[qid]
    g_tot = ALL_TOTAL_GOLD[qid]
    R = {}
    for K in K_REPORT:
        K_use = min(K, len(ranked))
        R[K] = len({d for d, _ in ranked[:K_use]} & g) / max(1, g_tot)
    RESULTS[CONFIG_NAME][qid] = R
    print(f"  [{qid}] {time.time()-t_q:6.1f}s  "
          f"R@50={R[50]:.3f} R@200={R[200]:.3f} R@500={R[500]:.3f} "
          f"R@1000={R[1000]:.3f} R@2000={R[2000]:.3f}  "
          + ("PASS" if R[2000] >= 0.7 else "fail"), flush=True)
''')


# =============================================================================
# Phase 6 - Free BGE, load jina-reranker-v2-multilingual
# =============================================================================
md(r'''# Phase 6 — Free BGE, load jina-reranker-v2-base-multilingual

Jina v2 multilingual is 278M params, ~0.6 GB BF16. Specifically trained for
cross-lingual reranking. ~15x faster than BGE-v2-m3 in published benchmarks.''')

code(r'''# Phase 6 - Free BGE, load jina-reranker-v2-multilingual
print("[teardown] freeing BGE ...")
try:
    if "bge_mod" in globals(): del bge_mod
    if "bge_tok" in globals(): del bge_tok
except Exception:
    pass
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"  CUDA mem after free = {torch.cuda.memory_allocated()/1024**3:.1f} GB")

JINA_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
print(f"\n[jina] loading {JINA_MODEL} ...")
_t0 = time.time()
from transformers import AutoModelForSequenceClassification, AutoTokenizer
jina_tok = AutoTokenizer.from_pretrained(JINA_MODEL, trust_remote_code=True)
jina_mod = AutoModelForSequenceClassification.from_pretrained(
    JINA_MODEL, dtype=torch.bfloat16, trust_remote_code=True
).cuda().eval()
print(f"[jina] loaded in {time.time()-_t0:.1f}s")
JINA_MAX_LEN = 1024
JINA_BATCH   = 128   # tiny model, big batches

def _jina_score(query, dids, repr_fn, max_len=JINA_MAX_LEN, batch_size=JINA_BATCH):
    # jina-reranker-v2 supports compute_score directly on a list of pairs
    pairs = [[query, repr_fn(d)] for d in dids]
    scores = []
    # Use raw transformers path for explicit batching control
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        with torch.no_grad():
            inp = jina_tok(
                [b[0] for b in batch], [b[1] for b in batch],
                padding=True, truncation=True, max_length=max_len, return_tensors="pt",
            )
            inp = {k: v.cuda() for k, v in inp.items()}
            logits = jina_mod(**inp).logits.view(-1).float()
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    return scores

CONFIG_NAME = "D_jina_v2_mul"
print(f"\n[{CONFIG_NAME}] reranking top-{STAGE1_TOP_N} for {TEST_QIDS} ...")
RESULTS[CONFIG_NAME] = {}
for qid in TEST_QIDS:
    qtext = next(q["query_text"] for q in ALL_QUERIES if q["query_id"] == qid)
    cands = PER_QUERY[qid]["final_topk"][:STAGE1_TOP_N]
    t_q = time.time()
    scores = _jina_score(qtext, cands, repr_enriched)
    ranked = sorted(zip(cands, scores), key=lambda kv: -kv[1])
    g = ALL_GOLD_DOC_SET[qid]
    g_tot = ALL_TOTAL_GOLD[qid]
    R = {}
    for K in K_REPORT:
        K_use = min(K, len(ranked))
        R[K] = len({d for d, _ in ranked[:K_use]} & g) / max(1, g_tot)
    RESULTS[CONFIG_NAME][qid] = R
    print(f"  [{qid}] {time.time()-t_q:6.1f}s  "
          f"R@50={R[50]:.3f} R@200={R[200]:.3f} R@500={R[500]:.3f} "
          f"R@1000={R[1000]:.3f} R@2000={R[2000]:.3f}  "
          + ("PASS" if R[2000] >= 0.7 else "fail"), flush=True)
''')


# =============================================================================
# Phase 7 - Final summary table
# =============================================================================
md(r'''# Phase 7 — Summary

Final summary table: per-config R@K on each of the 3 hardest queries.
Pass/fail is at R@2000 >= 0.7.

**Interpretation guide**:
- If **B (qwen3_fixed)** beats **A (qwen3_raw)** by > 0.10 on R@2000 → enrichment + cross-lingual instruction is the fix; deploy B.
- If **C (bge_v2_m3)** clearly beats both Qwen3 configs → Qwen3 is just the wrong model; switch to BGE-v2-m3.
- If **D (jina_v2_mul)** matches C → use jina (15x faster than BGE).
- If **no config hits R@2000 >= 0.7 on val_003** → val_003 is structurally hard (pool ceiling 0.766; we'd need 91.6% of in-pool gold in top-2000, basically perfect ranking). Mitigation: widen pool to top-100k.
- If **no config beats fusion baseline** anywhere → reranking is the wrong strategy; stick with fusion + variable-K methods.
''')

code(r'''# Phase 7 - Summary
print()
print("=" * 100)
print(f"  RERANK-ONLY DIAGNOSTIC — R@K per query × config")
print("=" * 100)
fus_R = {}
for qid in TEST_QIDS:
    c = PER_QUERY[qid]["curve"]
    fus_R[qid] = {K: c.get(K, c.get(str(K), (0, 0)))[1] for K in K_REPORT}

# Header
print(f"  {'config':<18}{'qid':<10}{'R@50':>8}{'R@200':>8}{'R@500':>8}"
      f"{'R@1000':>8}{'R@2000':>9}{'PASS_0.7':>10}")
print("  " + "-" * 78)
# Fusion baseline (reference row)
for qid in TEST_QIDS:
    R = fus_R[qid]
    print(f"  {'fusion (baseline)':<18}{qid:<10}{R[50]:>8.3f}{R[200]:>8.3f}{R[500]:>8.3f}"
          f"{R[1000]:>8.3f}{R[2000]:>9.3f}{'PASS' if R[2000]>=0.7 else 'fail':>10}")
print("  " + "-" * 78)
# Each tested config
for cfg in ["A_qwen3_raw", "B_qwen3_fixed", "C_bge_v2_m3", "D_jina_v2_mul"]:
    if cfg not in RESULTS: continue
    for qid in TEST_QIDS:
        R = RESULTS[cfg][qid]
        print(f"  {cfg:<18}{qid:<10}{R[50]:>8.3f}{R[200]:>8.3f}{R[500]:>8.3f}"
              f"{R[1000]:>8.3f}{R[2000]:>9.3f}{'PASS' if R[2000]>=0.7 else 'fail':>10}")
    print()

# Best per query at R@2000
print("=" * 100)
print(f"  Best config per query at R@2000")
print("=" * 100)
for qid in TEST_QIDS:
    candidates = [(cfg, RESULTS[cfg][qid][2000]) for cfg in RESULTS]
    candidates.append(("fusion", fus_R[qid][2000]))
    best = max(candidates, key=lambda kv: kv[1])
    print(f"  {qid:<10}  best = {best[0]:<22}  R@2000 = {best[1]:.3f}  "
          + ("PASS" if best[1] >= 0.7 else "fail"))

# Save results to disk
import json
with open(OUT_DIR / "rerank_only_results.json", "w", encoding="utf-8") as f:
    json.dump({
        "fusion_baseline": {qid: {str(k): v for k, v in fus_R[qid].items()} for qid in TEST_QIDS},
        "configs": {cfg: {qid: {str(k): v for k, v in R.items()}
                          for qid, R in qid_R.items()}
                    for cfg, qid_R in RESULTS.items()},
        "test_qids": TEST_QIDS,
        "stage1_top_n": STAGE1_TOP_N,
    }, f, indent=2)
print(f"\n[save] results -> {OUT_DIR / 'rerank_only_results.json'}")
''')


# =============================================================================
# Write notebook
# =============================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells, {OUT.stat().st_size/1024:.1f} KB)")
