"""Build the FINAL hybrid reranking diagnostic notebook.

Goal: Definitively answer — given the v7.5 50k-candidate pool (macro R = 0.89),
what is the smallest K where we can compress the pool while keeping
macro R >= 0.8 AND min-per-query R >= 0.8, using the best-of-the-best
state-of-the-art Swiss-legal multilingual reranker stack AUGMENTED with
all our data-derived signals.

ARSENAL (data-derived signals fused with cross-encoders):
  Cross-encoder relevance (3 SOTA multilingual rerankers, RRF-ensembled)
    1. Qwen3-Reranker-8B   (MMTEB-R 72.94, top at 8B; calibrated yes/no probs)
    2. BGE-reranker-v2-m3  (568M, multilingual + hard-neg trained, 100+ langs)
    3. jina-reranker-v2-base-multilingual (278M, cross-lingual specialist)

  Dossier signals (from snapshot + LLM query expansion + corpus regex):
    4. Statute-target intersection (canonicalized; doc_statute_anchors n statute_targets)
    5. Lead-statute intersection (first 200 chars; 14-88x gold lift per linguistic_signatures)
    6. Concept-target intersection EN (cross-lingual bridge tokens)
    7. Term-target intersection (DE + FR + IT; per-language counts summed)
    8. Co-citation density in pool (laws cited by court-paras in top-50k)
    9. Doctrinal density regex (rule-opener / doctrine / cite-cluster; 10 features)
   10. Hard-negative regex (dispositif / facts / cost / procedural; soft penalty)
   11. Chamber-class match (criminal/civil/admin via docket regex)
   12. Channel hit count (channel_recalls per-channel breadth via fusion-rank proxy)
   13. Fusion-baseline prior (v7.5 RRF rank; orthogonal to cross-encoders)
   14. Paragraph-role weighting (legal_standard/reasoning > facts/procedural)

ENSEMBLING:
  Reciprocal Rank Fusion (RRF) with k_rrf=60 — rank-based combination is
  robust to scale heterogeneity across probabilities (0-1), counts (0-N),
  and booleans (0/1), and needs no train-derived weight tuning.

CONFIGS UNDER TEST:
  F0  fusion_baseline       (v7.5 RRF only)                    -- reference
  F1  qwen3_only            (single best SOTA reranker)
  F2  rerank_ensemble       (RRF of 3 rerankers)
  F3  HYBRID                (RRF of 3 rerankers + 9 dossier signals)
  F4  HYBRID+hardneg_filter (drop dispositif/cost/facts upfront)

EVAL: per-config, per-query R@K for K in {50,100,200,500,1k,2k,5k,10k,20k,30k,40k,50k}.
Then find smallest K such that macro_R >= 0.8 AND min_R >= 0.8.

GPU BUDGET (RTX PRO 6000 Blackwell, 96 GB):
  Qwen3-Reranker-8B: ~12 min/q x 10 = ~2h (full 50k per query)
  BGE-v2-m3:         ~3  min/q x 10 = ~30 min
  jina-v2-mul:       ~2  min/q x 10 = ~20 min
  Total:             ~3 hours sequential (free between models).

Output notebook: swiss_citation_hybrid_rerank_final.ipynb
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).parent / "swiss_citation_hybrid_rerank_final.ipynb"
cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(keepends=True)})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.splitlines(keepends=True)})


# =============================================================================
# Title + design rationale
# =============================================================================
md(r'''# Swiss Citation Hybrid Reranking — Final Diagnostic (2026-05-14)

> **Question we are answering**: starting from the v7.5 50k-candidate pool
> (macro R = 0.89, min-per-q = 0.766 on val_003), how far can we compress
> the pool while keeping **macro R >= 0.8 AND min-per-query R >= 0.8**,
> using the BEST cross-encoder reranker stack we can build, augmented with
> every data-derived signal in our arsenal?
>
> This is the definitive diagnostic. After this run we will know:
> (a) the smallest K achievable at the 0.8-recall floor, (b) whether
> reranking gives lift over fusion at any K, and (c) which signals (cross-
> encoder vs. dossier) carry the lift.

---

## The arsenal — what we will fuse

### Cross-encoder relevance (3 SOTA multilingual rerankers)

| Model | Size | Why |
|---|---|---|
| **Qwen3-Reranker-8B** | 8 B  | MMTEB-R **72.94** (top at 8B); calibrated yes/no logprobs; cross-lingual EN->DE/FR/IT confirmed on this val by `Untitled75.ipynb` (F1=0.777) |
| **BGE-reranker-v2-m3** | 568 M | Multilingual-first, 100+ langs, hard-negative trained; sigmoid logit head |
| **jina-reranker-v2-base-multilingual** | 278 M | Cross-lingual specialist; 15x faster than BGE-v2-m3 in published benchmarks |

These 3 are blended via **rank-based RRF** (`k_rrf=60`) — no probability-scale tuning needed.

### Dossier signals (from snapshot + LLM expansion + corpus regex)

| # | Signal | Source | Strength evidence |
|---|---|---|---|
| 4 | **Statute-target intersection** | `doc_statute_anchors` n LLM `statute_targets` (canonicalized) | Direct rule citation; high precision but skips paragraphs that don't carry the anchor |
| 5 | **Lead-statute intersection** (first 200 chars) | Same, restricted to lead | Sharpest single dossier refinement (`feature_synthesis_round2_2026-05-13.md`) — 26-74x gold lift |
| 6 | **Concept-target intersection (EN)** | `_doc_to_concepts` n `concept_targets_en` | Cross-lingual bridge — English concepts attached to DE/FR/IT docs |
| 7 | **Term-target intersection (DE/FR/IT)** | `_doc_to_terms` n per-language `term_targets_*` | Catches docs whose statute anchors miss but content keywords match |
| 8 | **Co-citation density in pool** | For laws: # court-paras in top-50k citing this law. For courts: # peer paragraphs from same case in top-50k | Hub-statute detector; 83% intra-gold density per `linguistic_signatures` |
| 9 | **Doctrinal density (regex)** | 10 linguistic features: DE/FR rule-openers, doctrine phrases, internal cite clusters | Gold rate ~30-70% vs ambient ~0.4-6%; AUC 0.80-0.88 on gold-vs-ambient |
| 10 | **Hard-negative regex** | dispositif / cost / facts-header / procedural-boilerplate | Anti-pattern — gold rate ~0% in these roles |
| 11 | **Chamber-class match** | Criminal / civil / admin chamber regex on docket, matched to query `legal_area_keywords` | ~2x court-side noise reduction |
| 12 | **Channel hit count** | `channel_recalls` per-channel breadth (proxied by which retrieval channels recovered this doc) | Orthogonal-evidence ranker — multi-channel hits = stronger consensus |
| 13 | **Fusion-baseline prior** | v7.5 RRF rank itself | Independent of every reranker; preserves the 0.89 R@50k investment |
| 14 | **Paragraph-role weighting** | `doc_meta[did]['paragraph_role']` -> in {legal_standard, reasoning, application} = +; in {facts, procedural_history, dispositif, costs, admissibility} = - | Direct from the 8-way role classification |

### Ensembling: RRF over all 12 signals

For each candidate `d` in a query `q`, define rank under each signal `s` (lower = better).
Then:

```
hybrid_score(d, q) = sum_{s in signals} 1 / (k_rrf + rank_s(d, q))
```

This is the canonical Reciprocal Rank Fusion (Cormack et al. 2009) — no scale tuning,
no train-derived weights, robust to one signal being a probability and another being an integer.

Hard-negative signal (#10) is applied as a **rank penalty** (push to bottom 1/4) rather than a deletion in F3, and as a **filter** in F4 (drop entirely if matched).

---

## Configurations under test

| Config | Definition | What we learn |
|---|---|---|
| **F0** | Fusion baseline — v7.5 RRF only | Reference; this is where we are today |
| **F1** | Qwen3-Reranker-8B alone | Best single SOTA reranker |
| **F2** | RRF(Qwen3, BGE, jina) | Does ensembling rerankers add lift? |
| **F3** | RRF(rerankers) + RRF(9 dossier signals) | Full hybrid — the "best" we can build |
| **F4** | F3 with hard-neg filter applied first | Does pruning anti-pattern roles upfront help? |

---

## Pass criteria

For each config, compute R@K for K in `{50, 100, 200, 500, 1k, 2k, 5k, 10k, 20k, 30k, 40k, 50k}` per query, then macro R and min-per-query R per K.

**Floor: 0.8 macro AND 0.8 min-per-query.**

Find the smallest K where both hold. Report:
- The K_min per config
- The compression ratio (50k / K_min)
- The macro and min R at K_min
- The hardest val query (lowest R at K_min)

---

## No-hardcoding contract

- All signals are computed per-query from `ALL_TARGETS[qid]` (LLM-expanded) and `doc_meta[did]` (corpus regex/enrichment) — **no fixed Swiss-statute lists, no train priors**.
- All weights are RRF (rank-only) — **no train-derived signal coefficients**.
- The chamber-match table is a 5-row legal-domain classification (criminal/civil/admin/social/labor) that derives from query `legal_area_keywords`, not from val gold.
- This architecture is identical for any held-out query — `val_007` would receive the same processing as `test_023`.
''')


# =============================================================================
# Phase 0 - Setup
# =============================================================================
md(r'''# Phase 0 — Setup (env detection + vLLM/FlashInfer install)

Auto-detects Colab vs local. First run will install vLLM + FlashInfer, then
`raise SystemExit("Restart runtime...")` — restart kernel and re-run from
the top.''')

code(r'''# Phase 0 - Setup
import sys, os, json, gzip, time, re, gc, math, subprocess
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

# Snapshot path: handle both single- and double-nested layouts
_snap_a = _DRIVE / "research" / "anchor_funnel_val001_v7" / "snapshot"
_snap_b = _DRIVE / "research" / "anchor_funnel_val001_v7" / "snapshot" / "snapshot"
SNAPSHOT_DIR = _snap_a if (_snap_a / "config.json").exists() else _snap_b
OUT_DIR = _DRIVE / "research" / "hybrid_rerank_final"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
assert SNAPSHOT_DIR.exists() and (SNAPSHOT_DIR / "config.json").exists(), \
    f"Snapshot dir missing: {SNAPSHOT_DIR}"
print(f"snapshot: {SNAPSHOT_DIR}")
print(f"out:      {OUT_DIR}")
print(f"cache:    {CACHE_DIR}")

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
        import flashinfer  # noqa
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
# Phase 1 - Warm-boot from snapshot
# =============================================================================
md(r'''# Phase 1 — Warm-boot from snapshot

Loads the 50k-candidate pool + corpus enrichments + per-query LLM targets
for all 10 val queries. All RAM-resident — no DB calls during reranking.''')

code(r'''# Phase 1 - Warm-boot
import json as _j, gzip as _gz

print(f"[warm-boot] loading from {SNAPSHOT_DIR}")
_t0 = time.time()
CONFIG = _j.loads((SNAPSHOT_DIR / "config.json").read_text(encoding="utf-8"))

# Corpus snapshot (top-50k-per-query union; field codes: ct/cit/fam/cb/pr/ln/sa/cn/tm)
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
print(f"  corpus snapshot: {len(search_text):,} docs")

# Per-query pool (final_topk + recall curve). NOTE: this file's "gold_doc_ids"
# is the count (int), not the list — the actual gold list lives in
# gold_doc_sets.json (validated against val.csv at snapshot time).
_pq = _j.loads((SNAPSHOT_DIR / "per_query_snapshot.json").read_text(encoding="utf-8"))
PER_QUERY = {qid: {"final_topk": r["final_topk"],
                    "curve": {int(k) if str(k).isdigit() else k: tuple(v) if isinstance(v, list) else v
                              for k, v in r.get("curve", {}).items()},
                    "channel_recalls": r.get("channel_recalls", {})}
             for qid, r in _pq.items()}
_gold_sets = _j.loads((SNAPSHOT_DIR / "gold_doc_sets.json").read_text(encoding="utf-8"))
ALL_GOLD_DOC_SET = {qid: set(lst) for qid, lst in _gold_sets.items()}
QUERY_IDS = sorted(PER_QUERY.keys())
print(f"  PER_QUERY: {len(PER_QUERY)} queries")
for qid in QUERY_IDS:
    pool = PER_QUERY[qid]["final_topk"]
    g = ALL_GOLD_DOC_SET[qid]
    in_pool = len(set(pool) & g)
    print(f"    {qid}: pool={len(pool):>5}  gold={len(g):>3}  "
          f"in_pool={in_pool:>3}  R_max={in_pool/max(1,len(g)):.3f}")

# Per-query targets (LLM expansion)
ALL_TARGETS = _j.loads((SNAPSHOT_DIR / "all_targets.json").read_text(encoding="utf-8"))
# Per-query HyDE aspects (used for diagnostic logs only)
try:
    ALL_HYDE_ASPECTS = _j.loads((SNAPSHOT_DIR / "hyde_aspects.json").read_text(encoding="utf-8"))
except Exception:
    ALL_HYDE_ASPECTS = {qid: [] for qid in QUERY_IDS}

# Query text — from val.csv (snapshot's paths.json points to it)
import pandas as _pd
_paths = _j.loads((SNAPSHOT_DIR / "paths.json").read_text(encoding="utf-8"))
VAL_CSV = Path(_paths.get("val_csv", str(_DRIVE / "data" / "val.csv")))
if not VAL_CSV.exists():
    VAL_CSV = _DRIVE / "data" / "val.csv"
val_df = _pd.read_csv(VAL_CSV)
QUERY_TEXT = {str(r["query_id"]): str(r["query"]) for _, r in val_df.iterrows()}
for qid in QUERY_IDS:
    assert qid in QUERY_TEXT, f"query text missing for {qid}"

print(f"[warm-boot] done in {time.time()-_t0:.1f}s")
''')


# =============================================================================
# Phase 2 - Dossier helpers (statute canon, doctrinal density, hard-neg)
# =============================================================================
md(r'''# Phase 2 — Dossier helpers (fast substring path)

Pure-Python substring helpers — **no backtracking regex** on long text.
All cheap; throughput ~21k docs/sec single-thread (measured on full 255k snapshot).

- `canonicalize_statute(s)`: token-walk extraction of `"<NUM> <CODE>"` from LLM
  query targets like `"Art. 221 Abs. 1 lit. b StPO"` -> `"221 StPO"`. Matches the
  format of `doc_statute_anchors` in the snapshot (which is already canonical, so
  we use those anchors directly — no per-doc canonicalize call).
- `doctrinal_density(text)`: lowercased-substring score in [-10, +20] over fixed
  marker tuples (DE/FR rule-openers, doctrine phrases, rule-verbs, cite-cluster
  via `(bge ` / `(atf ` counts, statute density via `art.` count).
- `is_hard_neg(text, role)`: lowercased-substring check against dispositif / cost
  / facts / signature markers + the categorical role flag.
- `lead_codes(text_lead)`: token-walk extraction of `"<NUM> <CODE>"` from first
  200 chars (for the **lead_statute_int** signal — sharpest per `linguistic_signatures_2026-05-13.md`).
- `chamber_of(citation)` + `query_legal_areas(qid)`: small regex on short citation
  strings (no backtracking risk) + the universal Swiss-federal chamber-to-area map.

**Why this redesign**: the first version used a 40-way alternation regex with
nested optional groups (catastrophic backtracking — 5 docs/sec on Swiss legal
paragraphs). The substring approach is ~4000x faster with no semantic loss.''')

code(r'''# Phase 2 - dossier helpers (FAST: substring checks, no backtracking regex)
# Reasoning: doc-side statute anchors in the snapshot are ALREADY canonical
# strings ("104 OG", "717 OR"); query-side LLM-expanded targets are normalized
# by token-walking. All doctrinal/hard-neg signals use small fixed-substring
# tuples — no alternation regex on long text. Throughput: ~21,000 docs/sec
# single-thread (validated locally on full 255k corpus snapshot).

LAW_CODES = (
    "StPO", "StGB", "ZGB", "OR", "BGG", "ZPO", "BV", "IVG", "ATSG", "SchKG",
    "MWSTG", "UWG", "UVG", "AHVG", "BVG", "AVIG", "AsylG", "BGE", "ATF", "StG",
    "VStrR", "EleG", "EnG", "FINIG", "FinfraG", "GwG", "KVG", "VPG", "MEDBG",
    "SR", "CPC", "CC", "CO", "CP", "CPP", "LP", "LCart", "LTF", "LPGA", "LAI",
)


def canonicalize_statute(s: str) -> str:
    """'Art. 221 Abs. 1 lit. b StPO' -> '221 StPO'. Token-walking; no regex."""
    if not s: return s
    tokens = s.replace(",", " ").replace("(", " ").replace(")", " ").split()
    for i, tok in enumerate(tokens):
        if tok and tok[0].isdigit():
            num = tok.rstrip(".,;:")
            if not num or not num[0].isdigit():
                continue
            for j in range(i + 1, min(i + 6, len(tokens))):
                code = tokens[j].rstrip(".,;:)")
                if code in LAW_CODES:
                    return f"{num} {code}"
    return s.strip()


def canon_set(values) -> set:
    out = set()
    for v in values:
        c = canonicalize_statute(v)
        if c: out.add(c)
    return out


def lead_codes(text_lead: str) -> set:
    """Return canonical 'NUM CODE' pairs found in the first 200 chars (no regex)."""
    if not text_lead: return set()
    tokens = text_lead.replace(",", " ").replace("(", " ").replace(")", " ").split()
    out = set()
    for i, tok in enumerate(tokens):
        if tok and tok[0].isdigit():
            num = tok.rstrip(".,;:")
            if not num or not num[0].isdigit():
                continue
            for j in range(i + 1, min(i + 5, len(tokens))):
                code = tokens[j].rstrip(".,;:)")
                if code in LAW_CODES:
                    out.add(f"{num} {code}")
                    break
    return out


# Doctrinal markers (lowercased substring tuples — no regex)
DE_RULE_OPENERS = (
    "nach art.", "nach artikel", "gemäss art.", "gemäss artikel",
    "gemaess art.", "gemaess artikel", "im sinne von art.", "im sinne der art.",
    "laut art.", "laut artikel", "aufgrund von art.", "aufgrund des art.",
    "bei art.", "entsprechend art.",
)
DE_DOCTRINE = (
    "rechtsprechung", "ständige rechtsprechung", "staendige rechtsprechung",
    "bundesgerichtliche rechtsprechung", "bundesgerichtlichen rechtsprechung",
    "nach bundesgerichtlicher praxis", "nach herrschender lehre",
    "nach herrschender auffassung", "herrschende lehre",
    "lehre und rechtsprechung",
)
DE_RULE_VERBS = ("bestimmt", "sieht vor", "verlangt", "setzt voraus", "erfordert",
                 "ordnet an", "statuiert")
FR_RULE_OPENERS = (
    "selon l'art.", "selon l'article", "selon art.",
    "conformément à l'art.", "conformément à l'article",
    "conformement a l'art.", "conformement a l'article",
    "aux termes de l'art.", "aux termes de l'article",
    "en vertu de l'art.", "en vertu de l'article",
    "d'après l'art.", "d'apres l'art.", "suivant l'art.",
)
FR_DOCTRINE = (
    "selon la jurisprudence", "d'après la jurisprudence", "d'apres la jurisprudence",
    "jurisprudence constante", "selon la doctrine", "la doctrine et",
    "la doctrine admet", "la doctrine considère", "la doctrine considere",
)
FR_RULE_VERBS = ("dispose", "prévoit", "prevoit", "exige", "requiert", "ordonne", "stipule")

# Hard-neg signatures (lowercased substring tuples)
DE_DISPOSITIV_SIGS = ("demnach erkennt", "wird erkannt",
                      "das bundesgericht erkennt", "das bundesgericht hat entschieden",
                      "dispositiv", "urteilsdispositiv",
                      "die beschwerde wird gutgeheissen",
                      "die beschwerde wird abgewiesen",
                      "die beschwerde wird nicht eingetreten")
FR_DISPOSITIV_SIGS = ("par ces motifs", "le tribunal fédéral prononce",
                      "le tribunal federal prononce",
                      "le recours est admis", "le recours est rejeté",
                      "le recours est rejete", "le recours est irrecevable")
DE_FACTS_SIGS = ("sachverhalt", "in sachen ", "a.- ", "b.- ", "c.- ",
                 "gegen die verfügung", "gegen den entscheid")
DE_COSTS_SIGS = ("gerichtskosten", "verfahrenskosten", "parteientschädigung",
                 "parteientschadigung", "kostenfolge")
FR_COSTS_SIGS = ("frais judiciaires", "dépens", "depens", "émolument judiciaire",
                 "emolument judiciaire")
FR_SIG_SIGS = ("le président", "le presidént", "le greffier", "la greffière", "la greffiere")

_HARDNEG_ROLES = {"dispositif", "costs", "facts", "procedural_history",
                  "admissibility", "signature"}


def _count_any(text_lower: str, needles: tuple) -> int:
    return sum(1 for n in needles if n in text_lower)


def doctrinal_density(text: str, lang: str = "?") -> float:
    """[-10, +20]; positive = doctrinal rule statement, negative = anti-pattern."""
    if not text: return 0.0
    tl = text.lower()
    score = 0.0
    score += 3 * _count_any(tl, DE_RULE_OPENERS)
    score += 3 * _count_any(tl, DE_DOCTRINE)
    score += 1 * _count_any(tl, DE_RULE_VERBS)
    score += 3 * _count_any(tl, FR_RULE_OPENERS)
    score += 3 * _count_any(tl, FR_DOCTRINE)
    score += 1 * _count_any(tl, FR_RULE_VERBS)
    paren_cites = sum(tl.count(p) for p in ("(bge ", "(atf ", "; bge ", "; atf "))
    if paren_cites >= 3:
        score += 4
    elif paren_cites == 2:
        score += 2
    art_count = tl.count("art.")
    density = art_count / max(1, len(text) / 1000.0)
    score += min(density, 6.0) * 0.5
    if any(s in tl for s in DE_DISPOSITIV_SIGS) or any(s in tl for s in FR_DISPOSITIV_SIGS):
        score -= 5
    if any(s in tl for s in DE_FACTS_SIGS):
        score -= 3
    if any(s in tl for s in DE_COSTS_SIGS) or any(s in tl for s in FR_COSTS_SIGS):
        score -= 3
    if any(s in tl for s in FR_SIG_SIGS):
        score -= 3
    return max(-10.0, min(20.0, score))


def is_hard_neg(text: str, role: str) -> bool:
    if role and role.lower() in _HARDNEG_ROLES:
        return True
    if not text: return False
    tl = text.lower()
    if any(s in tl for s in DE_DISPOSITIV_SIGS) or any(s in tl for s in FR_DISPOSITIV_SIGS):
        return True
    if any(s in tl for s in DE_COSTS_SIGS) or any(s in tl for s in FR_COSTS_SIGS):
        return True
    if any(s in tl for s in FR_SIG_SIGS):
        return True
    return False


# Chamber classification — small regex on short citation strings only (no backtracking risk)
_QUERY_AREA_TO_CHAMBERS = {
    "criminal":      {"1B", "6B", "7B", "IV"},
    "civil":         {"4A", "5A", "I", "II", "III"},
    "administrative":{"2C", "8C", "9C"},
    "public":        {"1C", "2C"},
    "social":        {"8C", "9C", "U"},
    "labor":         {"4A", "8C"},
    "tax":           {"2C"},
    "asylum":        {"E", "F"},
    "intellectual":  {"4A"},
}
_CHAMBER_DOCKET = re.compile(
    r"BGE\s+\d+\s+([IVX]+)|(\d[A-Z]+)[._ ]?\d|\b([1-9][A-Z])\b")


def chamber_of(citation: str) -> str:
    if not citation: return ""
    m = _CHAMBER_DOCKET.search(citation)
    if not m: return ""
    for g in m.groups():
        if g: return g.upper()
    return ""


_LEGAL_AREA_PATTERNS = {
    "criminal":      re.compile(r"\b(criminal|strafe|strafrecht|penal|p[eé]nal|"
                                 r"d[eé]tention|untersuchungshaft|fluchtgefahr|"
                                 r"verdachts|stpo|strafverfahren)\b", re.IGNORECASE),
    "civil":         re.compile(r"\b(civil|zivil|contract|obligation|schadenersatz|"
                                 r"ehe|scheidung|erbrecht|ZGB|inheritance|tort)\b",
                                 re.IGNORECASE),
    "administrative":re.compile(r"\b(administrative|verwalt|publique|asyl|"
                                 r"ausl[aä]nder|migration|naturalisation)\b",
                                 re.IGNORECASE),
    "tax":           re.compile(r"\b(tax|steuer|mwst|imp[oô]t|imposition|"
                                 r"taxation|vat)\b", re.IGNORECASE),
    "social":        re.compile(r"\b(social|invalidity|pension|versicherung|"
                                 r"ahv|iv|uvg|bvg|disability|sozialversicher)\b",
                                 re.IGNORECASE),
    "labor":         re.compile(r"\b(labor|labour|arbeitsrecht|travail|"
                                 r"employment|k[uü]ndigung|dismissal|"
                                 r"arbeitsvertrag)\b", re.IGNORECASE),
    "intellectual":  re.compile(r"\b(intellectual|trademark|marke|patent|"
                                 r"urheber|copyright|design)\b", re.IGNORECASE),
}


def query_legal_areas(qid: str) -> set:
    kws = ALL_TARGETS.get(qid, {}).get("legal_area_keywords", []) or []
    text = " ".join(str(k) for k in kws)
    areas = {area for area, rx in _LEGAL_AREA_PATTERNS.items() if rx.search(text)}
    if not areas:
        areas.add("any")
    return areas


def allowed_chambers(qid: str) -> set:
    areas = query_legal_areas(qid)
    if "any" in areas:
        return set()
    chambers = set()
    for a in areas:
        chambers |= _QUERY_AREA_TO_CHAMBERS.get(a, set())
    return chambers


# Smoke test
print("[helpers] smoke test")
print(f"  canon('Art. 221 Abs. 1 lit. b StPO') = '{canonicalize_statute('Art. 221 Abs. 1 lit. b StPO')}'")
print(f"  canon('Art. 100 BGG')                = '{canonicalize_statute('Art. 100 BGG')}'")
print(f"  doctrinal('Nach Art. 221 StPO bestimmt ...') = "
      f"{doctrinal_density('Nach Art. 221 StPO bestimmt die Vorinstanz nach der Rechtsprechung des Bundesgerichts.'):.1f}")
print(f"  doctrinal('Demnach erkennt das BGer: ...')   = "
      f"{doctrinal_density('Demnach erkennt das Bundesgericht: 1. Die Beschwerde wird abgewiesen.'):.1f}")
for qid in QUERY_IDS[:3]:
    print(f"  {qid} legal_areas={sorted(query_legal_areas(qid))} chambers={sorted(allowed_chambers(qid))}")
''')


# =============================================================================
# Phase 3 - Precompute per-(qid, did) dossier features (vectorized)
# =============================================================================
md(r'''# Phase 3 — Precompute per-(qid, did) dossier feature matrix

For each query, build a `(N_cands, 9)` feature matrix that we can rank-sort
in pure NumPy later. This runs ONCE; the 9 columns are:

| col | name | semantic |
|---|---|---|
| 0 | `statute_int` | int — count of canonical statute-anchor n statute_targets |
| 1 | `lead_statute_int` | int — same restricted to first 200 chars of text |
| 2 | `concept_int` | int — count of doc.concepts n concept_targets_en |
| 3 | `term_int` | int — count of doc.terms n (term_targets_de u fr u it) |
| 4 | `co_cite` | int — for laws: # court-paras citing this in pool; for courts: peer count |
| 5 | `doctrinal` | float — doctrinal density score [-10, +20] |
| 6 | `chamber_match` | int 0/1 — chamber-class match (1 if pass or no constraint) |
| 7 | `hard_neg` | int 0/1 — hard-neg role / regex hit (we want this LOW) |
| 8 | `fusion_rank` | int — original v7.5 fusion rank (lower = better) |

We also cache `_doc_doctrinal[did]` so doctrinal score is computed per-doc only once,
then reused across queries.''')

code(r'''# Phase 3 - Precompute dossier feature matrix per query
import numpy as np

t_p3 = time.time()
DOSSIER_CACHE = CACHE_DIR / "dossier_features.npz"
COL_NAMES = ["statute_int", "lead_statute_int", "concept_int", "term_int",
             "co_cite", "doctrinal", "chamber_match", "hard_neg", "fusion_rank"]

# --- 3-CACHE. Disk-cache: skip Phase 3 entirely if cache exists ---
# Cache format (np.savez_compressed):
#   key "<qid>__dids" -> object array of doc-id strings (length N)
#   key "<qid>__feat" -> float32 array (N, 9), columns match COL_NAMES
# Cache is portable; produced once locally (CPU-only, ~3-5 min) and reused.
if DOSSIER_CACHE.exists():
    print(f"[3-cache] HIT  {DOSSIER_CACHE} ({DOSSIER_CACHE.stat().st_size/1024:.1f} KB)")
    _z = np.load(DOSSIER_CACHE, allow_pickle=True)
    DOSSIER = {}
    for qid in QUERY_IDS:
        dids = list(_z[f"{qid}__dids"])
        feat = _z[f"{qid}__feat"]
        DOSSIER[qid] = {"did_list": dids, "feat": feat}
        # sanity: pool must match the snapshot's final_topk
        snap_pool = PER_QUERY[qid]["final_topk"]
        assert len(dids) == len(snap_pool) and dids[0] == snap_pool[0] and dids[-1] == snap_pool[-1], \
            f"DOSSIER cache pool mismatch for {qid} — delete {DOSSIER_CACHE} to rebuild"
    print(f"[3-cache] loaded DOSSIER for {len(DOSSIER)} queries — skipping Phase 3 compute")
else:
    print(f"[3-cache] MISS — building DOSSIER ({DOSSIER_CACHE})")

    # --- 3a. Per-doc cached features (query-independent; FAST substring path) ---
    # Doc-side statute anchors in the snapshot are ALREADY canonical strings
    # ("104 OG", "717 OR"), so we use doc_statute_anchors[did] directly — no
    # canonicalize_statute call needed on the doc side.
    TEXT_TRUNC_REGEX = 2500
    print(f"[3a] per-doc cached features over {len(doc_meta):,} docs "
          f"(text trunc={TEXT_TRUNC_REGEX}, fast substring path) ...", flush=True)
    t0 = time.time()
    _doc_doctrinal = {}
    _doc_hardneg   = {}
    _doc_canon_statutes = {}      # already-canonical anchor set per doc
    _doc_lead_canon_statutes = {} # canonical 'NUM CODE' set in first 200 chars
    _doc_chamber = {}
    _n = 0
    for did, m in doc_meta.items():
        text = (search_text.get(did, "") or "")[:TEXT_TRUNC_REGEX]
        role = m.get("paragraph_role", "")
        cit  = m.get("citation", "") or ""
        _doc_doctrinal[did] = doctrinal_density(text)
        _doc_hardneg[did]   = 1 if is_hard_neg(text, role) else 0
        # Snapshot anchors are already canonical — just lift the set
        _doc_canon_statutes[did] = doc_statute_anchors.get(did, set())
        _doc_lead_canon_statutes[did] = lead_codes(text[:200])
        _doc_chamber[did] = chamber_of(cit)
        _n += 1
        if _n % 50_000 == 0:
            el = time.time() - t0
            rate = _n / max(el, 0.001)
            print(f"  ...{_n:,}/{len(doc_meta):,}  ({el:.1f}s, {rate:.0f} docs/s)",
                  flush=True)
    print(f"  done in {time.time()-t0:.1f}s "
          f"({len(_doc_doctrinal)/max(time.time()-t0, 0.001):.0f} docs/s)", flush=True)

    # --- 3b. Co-citation index: for each canonical statute, count court-paras citing it
    #         (restricted to the union of all pools across queries) ---
    print("[3b] co-citation index ...")
    t0 = time.time()
    _statute_cite_count = Counter()
    _court_case_to_peers = defaultdict(set)
    all_did_union = set()
    for qid in QUERY_IDS:
        all_did_union.update(PER_QUERY[qid]["final_topk"])
    for did in all_did_union:
        m = doc_meta.get(did, {})
        if m.get("family") == "court":
            for s in _doc_canon_statutes.get(did, set()):
                _statute_cite_count[s] += 1
            cb = m.get("court_base") or ""
            if cb:
                _court_case_to_peers[cb].add(did)
    print(f"  statute-cite index: {len(_statute_cite_count):,} statutes  "
          f"case-peer index: {len(_court_case_to_peers):,} cases  "
          f"in {time.time()-t0:.1f}s")

    def co_cite_count(did: str) -> int:
        """LAW: # court-paras citing this article. COURT: # peers in pool from same case."""
        m = doc_meta.get(did, {})
        if m.get("family") == "law":
            canon = _doc_canon_statutes.get(did, set())
            if not canon:
                return 0
            cit_canon = canonicalize_statute(m.get("citation", ""))
            if cit_canon in _statute_cite_count:
                return _statute_cite_count[cit_canon]
            return max((_statute_cite_count.get(s, 0) for s in canon), default=0)
        else:
            cb = m.get("court_base") or ""
            peers = _court_case_to_peers.get(cb, set())
            return max(0, len(peers) - 1)

    # --- 3c. Per-(qid, did) features ---
    print("[3c] per-(qid, did) feature matrix ...")
    DOSSIER = {}  # {qid: {"did_list": [...], "feat": np.ndarray (N,9)}}
    for qid in QUERY_IDS:
        targets = ALL_TARGETS.get(qid, {}) or {}
        stat_t = canon_set(targets.get("statute_targets", []))
        conc_t = set(targets.get("concept_targets_en", []))
        term_t = (set(targets.get("term_targets_de", [])) |
                  set(targets.get("term_targets_fr", [])) |
                  set(targets.get("term_targets_it", [])))
        chambers_ok = allowed_chambers(qid)

        pool = PER_QUERY[qid]["final_topk"]
        N = len(pool)
        feat = np.zeros((N, 9), dtype=np.float32)

        for i, did in enumerate(pool):
            feat[i, 0] = len(_doc_canon_statutes.get(did, set()) & stat_t)
            feat[i, 1] = len(_doc_lead_canon_statutes.get(did, set()) & stat_t)
            feat[i, 2] = len(_doc_to_concepts.get(did, set()) & conc_t)
            feat[i, 3] = len(_doc_to_terms.get(did, set()) & term_t)
            feat[i, 4] = co_cite_count(did)
            feat[i, 5] = _doc_doctrinal.get(did, 0.0)
            ch = _doc_chamber.get(did, "")
            fam = doc_meta.get(did, {}).get("family")
            if not chambers_ok or fam == "law" or not ch:
                feat[i, 6] = 1
            else:
                feat[i, 6] = 1 if ch in chambers_ok else 0
            feat[i, 7] = _doc_hardneg.get(did, 0)
            feat[i, 8] = i + 1

        DOSSIER[qid] = {"did_list": pool, "feat": feat}

    # --- 3-save. Persist to cache ---
    _save_payload = {}
    for qid in QUERY_IDS:
        _save_payload[f"{qid}__dids"] = np.array(DOSSIER[qid]["did_list"], dtype=object)
        _save_payload[f"{qid}__feat"] = DOSSIER[qid]["feat"]
    np.savez_compressed(DOSSIER_CACHE, **_save_payload)
    print(f"[3-cache] saved DOSSIER -> {DOSSIER_CACHE} "
          f"({DOSSIER_CACHE.stat().st_size/1024:.1f} KB)")

# --- 3-report. Quick sanity print (runs whether we loaded or computed) ---
print(f"\n[3] DOSSIER ready in {time.time()-t_p3:.1f}s")
print("\n  Sample (val_001) — first 3 pool docs, features:")
qid = "val_001"
for i, did in enumerate(DOSSIER[qid]["did_list"][:3]):
    row = DOSSIER[qid]["feat"][i]
    gold_flag = "[GOLD]" if did in ALL_GOLD_DOC_SET[qid] else ""
    fam = doc_meta.get(did, {}).get("family", "?")
    print(f"   {i:>4}: {did} ({fam}) {gold_flag}")
    for name, v in zip(COL_NAMES, row):
        print(f"         {name:<18} = {v}")
''')


# =============================================================================
# Phase 4 - Doc representations + Phase 5 - Cross-encoder A: Qwen3-Reranker-8B
# =============================================================================
md(r'''# Phase 4 — Enriched doc representation for cross-encoders

The single biggest reranker lift in `_build_rerank_only_diagnostic.py` came from
**enriched doc representation**: prepending Citation + family/role + Statute
anchors + Concepts (English) + Terms, then the body text truncated to 2000 chars.

The English concepts are the cross-lingual bridge — they give the reranker English
tokens to match against the English query, even when the body is German / French /
Italian.''')

code(r'''# Phase 4 - Enriched doc representation
def _join_short(values, max_items=8, max_chars=240):
    return ", ".join(list(values)[:max_items])[:max_chars]


def repr_enriched(did: str) -> str:
    """Citation + family/role + Statute anchors + Concepts (EN) + Terms + body (2000c)."""
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


CROSS_LING_INSTR = (
    "The Query is an English question about Swiss federal law. The Document is "
    "a Swiss federal law article (German) or a Swiss court paragraph (German, "
    "French, or Italian) — it may include English-language metadata (citation, "
    "statutes referenced, legal concepts). Decide YES if the Document is a "
    "relevant citation for the Query — i.e., the Query's legal question can be "
    "answered, supported, or directly discussed by this Document. Decide NO "
    "otherwise. Treat language differences as a translation problem, not a "
    "mismatch."
)


# Rerank budget per query (full pool by default)
RERANK_TOP_N = 50000        # adjust down for smoke test (e.g. 5000) before full run
K_REPORT = (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000)
RECALL_FLOOR = 0.80
print(f"[config] RERANK_TOP_N={RERANK_TOP_N}  K_REPORT={K_REPORT}  floor={RECALL_FLOOR}")
print(f"[config] OUT_DIR={OUT_DIR}")
print(f"[config] CACHE_DIR={CACHE_DIR}")

# Storage: ALL_RERANK_SCORES[model][qid] = np.array shape (len(pool),)
ALL_RERANK_SCORES = {}
''')


md(r'''# Phase 5 — Cross-encoder A: Qwen3-Reranker-8B (cross-lingual + enriched)

Loads `Qwen/Qwen3-Reranker-8B` via vLLM. Scores all 10 queries × top-50k pairs
with the cross-lingual instruction + enriched representation. Scores saved to
`{CACHE_DIR}/scores_qwen3.npz` — re-running this cell skips work if cache exists.

Expected wall-time on RTX PRO 6000 Blackwell: ~10-15 min/query × 10 = ~2 hours.''')

code(r'''# Phase 5 - Qwen3-Reranker-8B
import torch as _torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

QWEN3_MODEL = "Qwen/Qwen3-Reranker-8B"
QWEN3_CACHE = CACHE_DIR / "scores_qwen3.npz"

if QWEN3_CACHE.exists():
    print(f"[A] cache HIT  {QWEN3_CACHE} — loading and skipping rerank")
    _z = np.load(QWEN3_CACHE, allow_pickle=True)
    ALL_RERANK_SCORES["qwen3"] = {qid: _z[qid] for qid in _z.files}
    print(f"  loaded scores for {len(ALL_RERANK_SCORES['qwen3'])} queries")
else:
    print(f"[A] cache MISS — loading {QWEN3_MODEL} via vLLM ...")
    _t0 = time.time()
    qwen3_tok = AutoTokenizer.from_pretrained(QWEN3_MODEL, trust_remote_code=True,
                                              padding_side="left")
    import sys as _sysfix
    _orig_o, _orig_e = _sysfix.stdout, _sysfix.stderr
    _sysfix.stdout = _sysfix.__stdout__; _sysfix.stderr = _sysfix.__stderr__
    try:
        qwen3_llm = LLM(
            model=QWEN3_MODEL, dtype="bfloat16",
            max_model_len=1024, gpu_memory_utilization=0.85,
            enforce_eager=False, trust_remote_code=True,
        )
    finally:
        _sysfix.stdout = _orig_o; _sysfix.stderr = _orig_e
    print(f"[A] loaded in {time.time()-_t0:.1f}s", flush=True)

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

    def _qwen3_score_batch(query, dids):
        prompts = []
        for d in dids:
            body = (f"<Instruct>: {CROSS_LING_INSTR}\n"
                    f"<Query>: {query}\n"
                    f"<Document>: {repr_enriched(d)}")
            ids = qwen3_tok.encode(body, add_special_tokens=False)
            if len(ids) > QWEN3_MAX_BODY:
                ids = ids[:QWEN3_MAX_BODY]
            prompts.append({"prompt_token_ids": QWEN3_PREFIX_IDS + ids + QWEN3_SUFFIX_IDS})
        outs = qwen3_llm.generate(prompts, sampling_params=QWEN3_SP, use_tqdm=True)
        scores = np.empty(len(outs), dtype=np.float32)
        for i, out in enumerate(outs):
            lp = out.outputs[0].logprobs[0]
            y = lp.get(QWEN3_YES_ID); n = lp.get(QWEN3_NO_ID)
            yl = y.logprob if y else -1e9
            nl = n.logprob if n else -1e9
            mx = max(yl, nl)
            e_y = math.exp(yl - mx); e_n = math.exp(nl - mx)
            scores[i] = e_y / (e_y + e_n)
        return scores

    ALL_RERANK_SCORES["qwen3"] = {}
    for qid in QUERY_IDS:
        cands = PER_QUERY[qid]["final_topk"][:RERANK_TOP_N]
        t_q = time.time()
        print(f"  [{qid}] scoring {len(cands)} candidates ...", flush=True)
        scores = _qwen3_score_batch(QUERY_TEXT[qid], cands)
        ALL_RERANK_SCORES["qwen3"][qid] = scores
        # Quick recall@2k peek
        order = np.argsort(-scores)
        g = ALL_GOLD_DOC_SET[qid]
        g_tot = len(g)
        top2k = set(cands[j] for j in order[:2000])
        r2k = len(top2k & g) / max(1, g_tot)
        print(f"  [{qid}] {time.time()-t_q:6.1f}s   R@2000={r2k:.3f}", flush=True)

    np.savez_compressed(QWEN3_CACHE,
                        **{qid: s for qid, s in ALL_RERANK_SCORES["qwen3"].items()})
    print(f"[A] saved to {QWEN3_CACHE}")

    # Free Qwen3 to make room for next model
    print("[A] freeing Qwen3 vLLM engine ...")
    try:
        del qwen3_llm, qwen3_tok
    except Exception:
        pass
    gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
        print(f"   CUDA mem after free = {_torch.cuda.memory_allocated()/1024**3:.1f} GB")
''')


# =============================================================================
# Phase 6 - Cross-encoder B: BGE-reranker-v2-m3
# =============================================================================
md(r'''# Phase 6 — Cross-encoder B: BGE-reranker-v2-m3

568 M params, BF16 ~1.5 GB VRAM. Multilingual hard-neg trained. Sigmoid head.
Same enriched representation. Cached to `scores_bge.npz`.

Expected wall-time: ~3 min/query × 10 = ~30 min.''')

code(r'''# Phase 6 - BGE-reranker-v2-m3
BGE_MODEL = "BAAI/bge-reranker-v2-m3"
BGE_CACHE = CACHE_DIR / "scores_bge.npz"

if BGE_CACHE.exists():
    print(f"[B] cache HIT  {BGE_CACHE} — loading")
    _z = np.load(BGE_CACHE, allow_pickle=True)
    ALL_RERANK_SCORES["bge"] = {qid: _z[qid] for qid in _z.files}
else:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print(f"[B] cache MISS — loading {BGE_MODEL} ...")
    _t0 = time.time()
    bge_tok = AutoTokenizer.from_pretrained(BGE_MODEL)
    bge_mod = AutoModelForSequenceClassification.from_pretrained(
        BGE_MODEL, dtype=torch.bfloat16).cuda().eval()
    print(f"[B] loaded in {time.time()-_t0:.1f}s")
    BGE_MAX_LEN = 1024
    BGE_BATCH = 64

    def _bge_score_arr(query, dids):
        scores = np.empty(len(dids), dtype=np.float32)
        pairs = [(query, repr_enriched(d)) for d in dids]
        for i in range(0, len(pairs), BGE_BATCH):
            batch = pairs[i:i+BGE_BATCH]
            with torch.no_grad():
                inp = bge_tok(batch, padding=True, truncation=True,
                              max_length=BGE_MAX_LEN, return_tensors="pt")
                inp = {k: v.cuda() for k, v in inp.items()}
                logits = bge_mod(**inp).logits.view(-1).float()
                s = torch.sigmoid(logits).cpu().numpy()
            scores[i:i+len(batch)] = s
        return scores

    ALL_RERANK_SCORES["bge"] = {}
    for qid in QUERY_IDS:
        cands = PER_QUERY[qid]["final_topk"][:RERANK_TOP_N]
        t_q = time.time()
        scores = _bge_score_arr(QUERY_TEXT[qid], cands)
        ALL_RERANK_SCORES["bge"][qid] = scores
        order = np.argsort(-scores)
        g = ALL_GOLD_DOC_SET[qid]
        top2k = set(cands[j] for j in order[:2000])
        r2k = len(top2k & g) / max(1, len(g))
        print(f"  [{qid}] {time.time()-t_q:6.1f}s   R@2000={r2k:.3f}", flush=True)

    np.savez_compressed(BGE_CACHE,
                        **{qid: s for qid, s in ALL_RERANK_SCORES["bge"].items()})
    print(f"[B] saved to {BGE_CACHE}")

    print("[B] freeing BGE ...")
    try:
        del bge_mod, bge_tok
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"   CUDA mem after free = {torch.cuda.memory_allocated()/1024**3:.1f} GB")
''')


# =============================================================================
# Phase 7 - Cross-encoder C: jina-reranker-v2-multilingual
# =============================================================================
md(r'''# Phase 7 — Cross-encoder C: jina-reranker-v2-base-multilingual

278 M params, BF16 ~0.6 GB VRAM. Cross-lingual specialist. Same enriched
representation. Cached to `scores_jina.npz`.

Expected wall-time: ~2 min/query × 10 = ~20 min.''')

code(r'''# Phase 7 - jina-reranker-v2-base-multilingual
JINA_MODEL = "jinaai/jina-reranker-v2-base-multilingual"
JINA_CACHE = CACHE_DIR / "scores_jina.npz"

if JINA_CACHE.exists():
    print(f"[C] cache HIT  {JINA_CACHE} — loading")
    _z = np.load(JINA_CACHE, allow_pickle=True)
    ALL_RERANK_SCORES["jina"] = {qid: _z[qid] for qid in _z.files}
else:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Compat patch: jina-reranker-v2's custom modeling_xlm_roberta.py imports
    # create_position_ids_from_input_ids, which transformers removed in >=4.45.
    # Restore it before trust_remote_code triggers the import.
    from transformers.models.xlm_roberta import modeling_xlm_roberta as _xlmrm
    if not hasattr(_xlmrm, "create_position_ids_from_input_ids"):
        def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
            mask = input_ids.ne(padding_idx).int()
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)
                                   + past_key_values_length) * mask
            return incremental_indices.long() + padding_idx
        _xlmrm.create_position_ids_from_input_ids = create_position_ids_from_input_ids
        print("[C] patched transformers.xlm_roberta with create_position_ids_from_input_ids")

    print(f"[C] cache MISS — loading {JINA_MODEL} ...")
    _t0 = time.time()
    jina_tok = AutoTokenizer.from_pretrained(JINA_MODEL, trust_remote_code=True)
    jina_mod = AutoModelForSequenceClassification.from_pretrained(
        JINA_MODEL, dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()
    print(f"[C] loaded in {time.time()-_t0:.1f}s")
    JINA_MAX_LEN = 1024
    JINA_BATCH = 128

    def _jina_score_arr(query, dids):
        scores = np.empty(len(dids), dtype=np.float32)
        pairs = [[query, repr_enriched(d)] for d in dids]
        for i in range(0, len(pairs), JINA_BATCH):
            batch = pairs[i:i+JINA_BATCH]
            with torch.no_grad():
                inp = jina_tok(
                    [b[0] for b in batch], [b[1] for b in batch],
                    padding=True, truncation=True, max_length=JINA_MAX_LEN,
                    return_tensors="pt",
                )
                inp = {k: v.cuda() for k, v in inp.items()}
                logits = jina_mod(**inp).logits.view(-1).float()
                s = torch.sigmoid(logits).cpu().numpy()
            scores[i:i+len(batch)] = s
        return scores

    ALL_RERANK_SCORES["jina"] = {}
    for qid in QUERY_IDS:
        cands = PER_QUERY[qid]["final_topk"][:RERANK_TOP_N]
        t_q = time.time()
        scores = _jina_score_arr(QUERY_TEXT[qid], cands)
        ALL_RERANK_SCORES["jina"][qid] = scores
        order = np.argsort(-scores)
        g = ALL_GOLD_DOC_SET[qid]
        top2k = set(cands[j] for j in order[:2000])
        r2k = len(top2k & g) / max(1, len(g))
        print(f"  [{qid}] {time.time()-t_q:6.1f}s   R@2000={r2k:.3f}", flush=True)

    np.savez_compressed(JINA_CACHE,
                        **{qid: s for qid, s in ALL_RERANK_SCORES["jina"].items()})
    print(f"[C] saved to {JINA_CACHE}")

    print("[C] freeing jina ...")
    try:
        del jina_mod, jina_tok
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"   CUDA mem after free = {torch.cuda.memory_allocated()/1024**3:.1f} GB")
''')


# =============================================================================
# Phase 8 - Score combination (F0..F4) via RRF
# =============================================================================
md(r'''# Phase 8 — Hybrid score combination (RRF over all signals)

For each candidate `d` in query `q`, the score under config X is

```
score_X(d, q) = sum_{s in signals_X} 1 / (k_rrf + rank_s(d, q))
```

where `rank_s` is the 1-indexed rank under signal `s` (lower = better; ties broken by stable sort).

Signals by config:

- **F0**: `fusion_rank` only (1 signal, baseline).
- **F1**: `qwen3_score` only (1 signal).
- **F2**: `qwen3_score`, `bge_score`, `jina_score` (3 rerankers).
- **F3**: F2 + 9 dossier signals (statute_int, lead_statute_int, concept_int, term_int,
  co_cite, doctrinal, chamber_match, fusion_rank, -hard_neg).
  - `chamber_match` and `(-hard_neg)` are 0/1 — for these we use a soft penalty: rows with `chamber_match=0` get rank-bottom in that signal; rows with `hard_neg=1` get rank-bottom in `(-hard_neg)`.
  - For F3 the hard-neg is a *signal*, not a filter — it pushes down but doesn't remove.
- **F4**: F3 with hard-neg as a FILTER (drop `hard_neg==1` rows BEFORE fusion).
  Useful test: does deletion vs penalty change the K_min?

k_rrf = 60 (standard).''')

code(r'''# Phase 8 - RRF score combination
K_RRF = 60

def rank_from_scores(scores: np.ndarray, descending: bool = True) -> np.ndarray:
    """Return 1-indexed rank of each element. Lower = better."""
    if descending:
        order = np.argsort(-scores, kind="stable")
    else:
        order = np.argsort(scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def rrf_of(rank_arrays: list[np.ndarray]) -> np.ndarray:
    """Sum 1/(k_rrf + rank) across the provided rank arrays. Higher = better."""
    if not rank_arrays:
        raise ValueError("no ranks given")
    out = np.zeros_like(rank_arrays[0], dtype=np.float32)
    for r in rank_arrays:
        out += 1.0 / (K_RRF + r.astype(np.float32))
    return out


def evaluate(qid: str, ranked_pool: list[str], K_list=K_REPORT) -> dict[int, float]:
    """Compute R@K against gold for a list of dids ordered best->worst.
    Assumes ranked_pool has unique dids (true for v7.5 final_topk)."""
    g = ALL_GOLD_DOC_SET[qid]
    g_tot = max(1, len(g))
    if not ranked_pool:
        return {K: 0.0 for K in K_list}
    is_gold = np.array([1 if d in g else 0 for d in ranked_pool], dtype=np.int32)
    cumhit = np.cumsum(is_gold)
    out = {}
    for K in K_list:
        K_use = min(K, len(ranked_pool))
        out[K] = float(cumhit[K_use - 1]) / g_tot if K_use > 0 else 0.0
    return out


CONFIGS = ["F0_fusion", "F1_qwen3", "F2_rerank_ensemble", "F3_HYBRID",
           "F4_HYBRID_hardneg_filter"]
RESULTS: dict[str, dict[str, dict[int, float]]] = {c: {} for c in CONFIGS}

for qid in QUERY_IDS:
    pool = DOSSIER[qid]["did_list"]
    feat = DOSSIER[qid]["feat"]
    N = len(pool)
    # Fusion rank is just 1..N (pool already in fusion order)
    fusion_rank = feat[:, COL_NAMES.index("fusion_rank")].astype(np.int32)

    # Reranker scores -> ranks
    s_qwen3 = ALL_RERANK_SCORES["qwen3"][qid][:N]
    s_bge   = ALL_RERANK_SCORES["bge"][qid][:N]
    s_jina  = ALL_RERANK_SCORES["jina"][qid][:N]
    r_qwen3 = rank_from_scores(s_qwen3, descending=True)
    r_bge   = rank_from_scores(s_bge,   descending=True)
    r_jina  = rank_from_scores(s_jina,  descending=True)

    # Dossier signal ranks (descending for "more = better"; ascending for hard_neg)
    r_statute    = rank_from_scores(feat[:, 0], descending=True)
    r_lead_stat  = rank_from_scores(feat[:, 1], descending=True)
    r_concept    = rank_from_scores(feat[:, 2], descending=True)
    r_term       = rank_from_scores(feat[:, 3], descending=True)
    r_cocite     = rank_from_scores(feat[:, 4], descending=True)
    r_doctrinal  = rank_from_scores(feat[:, 5], descending=True)
    r_chamber    = rank_from_scores(feat[:, 6], descending=True)   # 1=match, 0=no
    r_notneg     = rank_from_scores(-feat[:, 7], descending=True)  # 1 if NOT hard-neg

    # ---- F0: fusion baseline ----
    f0_order = np.argsort(fusion_rank)
    RESULTS["F0_fusion"][qid] = evaluate(qid, [pool[i] for i in f0_order])

    # ---- F1: qwen3-only ----
    f1_order = np.argsort(r_qwen3)
    RESULTS["F1_qwen3"][qid] = evaluate(qid, [pool[i] for i in f1_order])

    # ---- F2: 3-reranker RRF ----
    f2_score = rrf_of([r_qwen3, r_bge, r_jina])
    f2_order = np.argsort(-f2_score)
    RESULTS["F2_rerank_ensemble"][qid] = evaluate(qid, [pool[i] for i in f2_order])

    # ---- F3: HYBRID (3 rerankers + 9 dossier signals) ----
    f3_signals = [r_qwen3, r_bge, r_jina,
                  r_statute, r_lead_stat, r_concept, r_term,
                  r_cocite, r_doctrinal, r_chamber, r_notneg,
                  fusion_rank]
    f3_score = rrf_of(f3_signals)
    f3_order = np.argsort(-f3_score)
    RESULTS["F3_HYBRID"][qid] = evaluate(qid, [pool[i] for i in f3_order])

    # ---- F4: HYBRID with hard-neg FILTER ----
    not_hardneg_mask = feat[:, 7] == 0
    keep_idx = np.where(not_hardneg_mask)[0]
    if len(keep_idx) == 0:
        # impossible safety net
        RESULTS["F4_HYBRID_hardneg_filter"][qid] = RESULTS["F3_HYBRID"][qid]
    else:
        # RRF over the kept subset using the same signals (but recomputed ranks on subset)
        # For speed: just take the F3 order and remove hard-negs
        f4_order_all = f3_order
        keep_set = set(keep_idx.tolist())
        f4_order = [i for i in f4_order_all if i in keep_set]
        RESULTS["F4_HYBRID_hardneg_filter"][qid] = evaluate(qid, [pool[i] for i in f4_order])

    # Console summary
    g = len(ALL_GOLD_DOC_SET[qid])
    print(f"[{qid}] gold={g:>3}  "
          + "  ".join(
              f"{cfg.split('_',1)[0]}:R@2k={RESULTS[cfg][qid][2000]:.3f}"
              for cfg in CONFIGS))
''')


# =============================================================================
# Phase 9 - Summary tables + smallest-K-at-floor analysis
# =============================================================================
md(r'''# Phase 9 — Summary: smallest K achieving R >= 0.8

For each config and each K in `K_REPORT`:
- `macro_R@K`  = mean over the 10 val queries of `R@K`
- `min_R@K`    = min over the 10 val queries (binding for the floor)

Then find `K_min(config) = min{K : macro_R@K >= 0.8 AND min_R@K >= 0.8}` and report:
- the compression ratio `50000 / K_min`
- which val query is the binding (worst) one at `K_min`
- a full R@K table per config''')

code(r'''# Phase 9 - Summary tables
import pandas as _pd

def macro_min_at(cfg: str, K: int) -> tuple[float, float, str]:
    vals = [(qid, RESULTS[cfg][qid][K]) for qid in QUERY_IDS]
    macro = sum(v for _, v in vals) / len(vals)
    worst_qid, worst_R = min(vals, key=lambda kv: kv[1])
    return macro, worst_R, worst_qid


def smallest_K_at_floor(cfg: str, floor: float = RECALL_FLOOR) -> tuple[int | None, dict]:
    for K in K_REPORT:
        macro, minR, worst = macro_min_at(cfg, K)
        if macro >= floor and minR >= floor:
            return K, {"macro": macro, "min": minR, "worst_qid": worst, "K": K}
    return None, {}


print("=" * 110)
print(f"  HYBRID RERANK — Smallest K such that macro R >= {RECALL_FLOOR} AND min-per-query R >= {RECALL_FLOOR}")
print("=" * 110)
print(f"  {'config':<28}{'K_min':>10}{'compression':>14}"
      f"{'macro_R':>10}{'min_R':>9}{'worst_query':>15}")
print("  " + "-" * 100)
summary_rows = []
for cfg in CONFIGS:
    Kmin, info = smallest_K_at_floor(cfg)
    if Kmin is None:
        print(f"  {cfg:<28}{'FAIL':>10}{'-':>14}{'-':>10}{'-':>9}{'-':>15}  "
              f"(no K in K_REPORT reaches the floor)")
        summary_rows.append({"config": cfg, "K_min": None,
                             "macro_R": None, "min_R": None, "worst_query": None})
    else:
        comp = 50000 / Kmin
        print(f"  {cfg:<28}{Kmin:>10}{comp:>13.1f}x"
              f"{info['macro']:>10.3f}{info['min']:>9.3f}"
              f"{info['worst_qid']:>15}")
        summary_rows.append({"config": cfg, "K_min": Kmin, "compression": comp,
                             "macro_R": info["macro"], "min_R": info["min"],
                             "worst_query": info["worst_qid"]})

# Full R@K table per config
print()
print("=" * 110)
print(f"  FULL R@K TABLE — macro / min across {len(QUERY_IDS)} val queries")
print("=" * 110)
for cfg in CONFIGS:
    print(f"\n  {cfg}")
    print(f"    {'K':>7}{'macro_R':>10}{'min_R':>9}{'worst_query':>15}")
    for K in K_REPORT:
        macro, minR, worst = macro_min_at(cfg, K)
        floor_flag = "  ***" if (macro >= RECALL_FLOOR and minR >= RECALL_FLOOR) else ""
        print(f"    {K:>7}{macro:>10.3f}{minR:>9.3f}{worst:>15}{floor_flag}")

# Per-query R@K table for the best config (F3)
print()
print("=" * 110)
print(f"  PER-QUERY R@K — F3_HYBRID (the headline config)")
print("=" * 110)
print(f"  {'qid':<10}{'gold':>6}" + "".join(f"{f'R@{K}':>9}" for K in K_REPORT))
for qid in QUERY_IDS:
    g = len(ALL_GOLD_DOC_SET[qid])
    row = "  " + f"{qid:<10}{g:>6}"
    for K in K_REPORT:
        row += f"{RESULTS['F3_HYBRID'][qid][K]:>9.3f}"
    print(row)

# Macro R@K matrix across configs (for plotting)
print()
print("=" * 110)
print(f"  MACRO R@K MATRIX (configs as rows, K as columns)")
print("=" * 110)
print(f"  {'config':<28}" + "".join(f"{f'R@{K}':>9}" for K in K_REPORT))
for cfg in CONFIGS:
    row = f"  {cfg:<28}"
    for K in K_REPORT:
        macro, _, _ = macro_min_at(cfg, K)
        row += f"{macro:>9.3f}"
    print(row)

# Save everything
results_out = {
    "config": {
        "RERANK_TOP_N": RERANK_TOP_N,
        "K_REPORT": list(K_REPORT),
        "RECALL_FLOOR": RECALL_FLOOR,
        "K_RRF": K_RRF,
    },
    "configs_tested": CONFIGS,
    "summary": summary_rows,
    "macro_min_table": {
        cfg: {K: {"macro": macro_min_at(cfg, K)[0],
                  "min":   macro_min_at(cfg, K)[1],
                  "worst": macro_min_at(cfg, K)[2]}
              for K in K_REPORT}
        for cfg in CONFIGS
    },
    "per_query": {
        cfg: {qid: {str(K): RESULTS[cfg][qid][K] for K in K_REPORT}
              for qid in QUERY_IDS}
        for cfg in CONFIGS
    },
}
out_path = OUT_DIR / "hybrid_rerank_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results_out, f, indent=2)
print(f"\n[save] results -> {out_path}")
''')


# =============================================================================
# Phase 10 - Interpretation
# =============================================================================
md(r'''# Phase 10 — How to read the result

The numbers above give us the definitive answer to the diagnostic. Possible outcomes:

### Outcome A — F3 hybrid lifts K_min meaningfully below 50k
e.g. F3 hits the floor at K=10000 while F0 needs 40000.
**Action**: deploy F3 ordering as the cascade Stage 1.5 (after fusion, before LLM cascade). Use top-10k from F3 as the input to the existing Stage 2 listwise LLM.

### Outcome B — F3 == F2 (rerankers carry it; dossier adds nothing)
**Action**: drop the dossier signals from production; rely on the 3-reranker RRF only. Simpler is better.

### Outcome C — F3 cannot reach the floor at any K above the fusion baseline
**Action**: reranking is structurally unable to compress this pool. The candidate selection is the bottleneck. Two paths:
  1. Pool widening (push to top-100k or top-200k) and re-measure.
  2. Move investment to query-side: better LLM-expanded statute/concept/term targets (more diverse seeds), at the cost of pool size.

### Outcome D — F4 (hard-neg filter) noticeably beats F3
**Action**: dispositif / cost / facts paragraphs are systematically dragging the ranking down. Apply the hard-neg filter at the corpus level before fusion (free; offline; saves rerank compute).

### Outcome E — F1 (Qwen3 alone) ~= F3
**Action**: Qwen3-Reranker-8B is already capturing all the signal. Drop BGE / jina / dossier from the production cascade. Saves ~50 min of compute per run.

---

The full per-query R@K is dumped to `{OUT_DIR}/hybrid_rerank_results.json`, the cached scores are at `{CACHE_DIR}/scores_{qwen3,bge,jina}.npz` so the analysis can be replayed without re-running the GPU work.
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
