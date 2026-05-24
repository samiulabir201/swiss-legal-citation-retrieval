"""Generate the FINAL state-of-the-art reranking diagnostic notebook.

Produces: notebooks/swiss_citation_reranker_final_diagnostic.ipynb

Question the notebook answers (on the same 10 val queries, top-50k snapshot
per query, fusion baseline R@5000 = 0.728):
  At what smallest K can we hold macro-mean R@K >= 0.80, using
   (1) the v7.5 fusion order alone (baseline),
   (2) Qwen3-Reranker-8B alone (raw),
   (3) Qwen3-Reranker-8B with enriched repr + cross-lingual instruction (SOTA),
   (4) Hybrid: (3) fused with ALL data derivatives we have built ?
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(r"E:\swiss_citation_extraction\notebooks\swiss_citation_reranker_final_diagnostic.ipynb")

def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}

def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}


CELLS = []

# ---------------------------------------------------------------------------
# Cell 0 - Title + arsenal inventory
# ---------------------------------------------------------------------------
CELLS.append(md("""# Swiss Citation - Reranker FINAL Diagnostic (top-50k -> smallest K @ R >= 0.80)

**Question this notebook answers**: starting from the v7.5 multi-query snapshot
(top-50,000 candidates per val query, macro R@50k = 0.89, macro R@5k = 0.728),
at what smallest K can we hold **macro-mean R@K >= 0.80** on the 10 val queries?

We compare four configurations:

| # | Config | Description |
|---|---|---|
| 1 | `fusion_baseline` | The v7.5 14-channel RRF fusion order from the snapshot. No reranking. |
| 2 | `qwen3_raw` | Qwen3-Reranker-8B on raw doc text, generic instruction. |
| 3 | `qwen3_sota` | Qwen3-Reranker-8B + **enriched repr** + **cross-lingual instruction**. State-of-the-art single-model config. |
| 4 | `hybrid` | Late-fusion of `qwen3_sota` score with every data derivative we have built (full arsenal). |

**Why these four**: the rerank-only diagnostic already showed Qwen3-Reranker-8B
> BGE-v2-m3 > Jina-v2-mul on this data. So we stop the model bake-off and lock
to Qwen3-Reranker-8B, then attack the gap with hybrid scoring.

---

## Arsenal - data derivatives we exploit in the hybrid score

These are all built upstream and live in the snapshot. We do NOT re-compute them.

### A. From v7.5 retrieval pipeline (snapshot)
1. **`fusion_rank`** - position in the 14-channel RRF order. Strongest single non-LLM signal (R@5k = 0.728).
2. **`final_topk`** - the 50k candidate set per query (already structurally filtered).

### B. From corpus enrichment (LLM-tagged once at build time)
3. **`doc_statute_anchors`** - canonical statute references per doc (`"41 OR"` form).
4. **`_doc_to_concepts`** - English concept tags per doc. **Primary cross-lingual bridge** for EN query -> DE/FR/IT doc.
5. **`_doc_to_terms`** - DE/FR/IT terms tagged per doc (language-matched).
6. **`paragraph_role`** - LLM-classified role (`legal_standard` / `reasoning` / `application` / `facts` / `procedural_history` / `notification` / `costs` / `dispositif`).
7. **`court_base`** - case-base citation (lets us count case peers).
8. **`family` + `language`** - law vs court, DE/FR/IT.

### C. From query expansion (per-query LLM dossier)
9. **`statute_targets`** - LLM-expanded statute hits the query maps to.
10. **`concept_targets_en`** - LLM-expanded English concept targets.
11. **`term_targets_de` / `_fr` / `_it`** - LLM-expanded language-matched terms.
12. **`legal_area_keywords`** - LLM-tagged legal area.
13. **`ALL_HYDE_ASPECTS`** - HyDE-style query sub-aspects (each with a keyword bag).

### D. Computed-here-from-arsenal (deterministic, no LLM)
14. **`lead_statute_intersection`** - statutes appearing in the first 200 chars of doc text. Sharpest discriminator per Round-2 feature synthesis (kills tangential-mention anti-pattern).
15. **`chamber`** - regex over citation (`BGE \\d+ [IVX]+ \\d+` / `^\\d[A-Z]_\\d+`). 7B/1B = criminal, 4A/5A = civil, 8C/9C = social, 2C/2D = tax, etc.
16. **`chamber_class_match`** - `chamber_class(d) == legal_area(q)`. Free, attacks chamber-spillover.
17. **`doctrinal_density`** - regex score for rule-openers (`Nach Art.`, `Selon l'art.`), doctrine phrases (`stAndige Rechtsprechung`, `la jurisprudence`), internal-cite clusters. Gold-vs-ambient AUC = 0.80-0.88.
18. **`is_dispositif_or_facts`** - hard-negative regex (`Sachverhalt`, `wird abgewiesen`, `en fait`, `le greffier`, ...). Never gold.
19. **`co_citation_count`** - for laws: how many top-K court paragraphs cite it. Picks hub statutes (83% intra-gold density).
20. **`case_peer_count`** + **`case_peer_has_rule_role`** - for courts: discriminates wrong-E within right case.
21. **`aspect_best_match`** + **`n_aspects_addressed`** - keyword-bag overlap against each HyDE aspect.

### E. SOTA cross-encoder
22. **`Qwen3-Reranker-8B`** - 8B, 119-language, instruction-tuned. Proven feasible on this exact task (Untitled75.ipynb F1=0.777). With enriched repr (Citation + statutes + concepts EN + terms + role + body) and explicit cross-lingual instruction.

---

## Pipeline (top -> bottom)

```
50,000 per query     (snapshot, macro R@50k = 0.89)
        |
        V  Stage A: cheap rule pre-prune (cantonal-court, dispositif regex, noise-role drops; <1% gold loss)
        |
~20-30k per query
        |
        V  Phase 3: Qwen3-Reranker-8B  (vLLM, Blackwell, enriched repr + cross-lingual instruction)
        |
ranked + scored
        |
        V  Phase 4: Hybrid late-fusion (z-score sum of reranker + 11 handcrafted experts)
        |
ranked + scored (hybrid)
        |
        V  Phase 5: R@K sweep at K = 50, 100, 200, 500, 1k, 2k, 5k, 10k, 20k, 30k, 40k, 50k
        |
diagnostic table + smallest-K-for-R>=0.80 per config + save
```

---

## Configuration knobs

All at top of Phase 0 cell for one-line override:

| Knob | Default | Meaning |
|---|---|---|
| `STAGE1_TOP_N` | 50000 | rerank the full v7.5 pool |
| `K_REPORT` | [50,100,200,500,1k,2k,5k,10k,20k,30k,40k,50k] | sweep grid |
| `RECALL_TARGET` | 0.80 | the minimum recall we must hold |
| `RERANK_CACHE` | `OUT_DIR/rerank_scores/{qid}.json.gz` | per-query cache so re-runs of Phase 4-5 are free |
| `MAX_LEN` | 1024 | reranker prompt context |

---

## Expected runtime (RTX PRO 6000 Blackwell, 96 GB)
- Phase 0-2: ~2 min (warm-boot + dossier).
- Stage A: ~10 sec.
- Phase 3 rerank: ~12 min/query x 10 = ~2 h (CACHED - re-runs are seconds).
- Phase 4-6: ~30 sec.

Total cold run ~2 h; warm run (cache hit) ~3 min.
"""))


# ---------------------------------------------------------------------------
# Cell 1 - Phase 0 markdown
# ---------------------------------------------------------------------------
CELLS.append(md("""---

# Phase 0 - Setup (Blackwell-optimized vLLM + FlashInfer install)

Same install pattern as the precision-v1 notebook. First-run installs vLLM and
FlashInfer, then `raise SystemExit` to force a runtime restart so CUDA bindings
register cleanly. Subsequent runs skip the install.
"""))


# ---------------------------------------------------------------------------
# Cell 2 - Setup code
# ---------------------------------------------------------------------------
CELLS.append(code("""# Phase 0 - Setup: imports + path resolution + vLLM/FlashInfer install
import sys, os, json, gzip, time, re, gc, subprocess, math, hashlib
import importlib.util
from pathlib import Path
from collections import defaultdict, Counter

print(f"Python {sys.version_info.major}.{sys.version_info.minor}")

# ---- Environment detection ----
try:
    from google.colab import drive as _drv
    _drv.mount("/content/drive", force_remount=False)
    _DRIVE = Path("/content/drive/MyDrive/swiss_law")
    ENV = "colab"
except (ImportError, ModuleNotFoundError):
    _DRIVE = Path(r"E:\\swiss_citation_extraction")
    ENV = "local"
IN_KAGGLE = bool(os.environ.get("KAGGLE_URL_BASE") or Path("/kaggle").exists())
print(f"Environment: {ENV}{' (Kaggle)' if IN_KAGGLE else ''}")

# The v7.5 multi-query snapshot lives at the legacy val001_v7 folder (intentional).
SNAPSHOT_DIR = _DRIVE / "research" / "anchor_funnel_val001_v7" / "snapshot"
OUT_DIR      = _DRIVE / "research" / "reranker_final_diagnostic"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RERANK_CACHE = OUT_DIR / "rerank_scores"
RERANK_CACHE.mkdir(parents=True, exist_ok=True)
assert SNAPSHOT_DIR.exists(), f"Snapshot dir missing: {SNAPSHOT_DIR}"
print(f"snapshot: {SNAPSHOT_DIR}")
print(f"out:      {OUT_DIR}")
print(f"cache:    {RERANK_CACHE}")

# ---- Master knobs ----
STAGE1_TOP_N   = 50000
K_REPORT       = (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000)
RECALL_TARGET  = 0.80
MAX_LEN        = 1024  # Qwen3-Reranker prompt context

# ---- vLLM + FlashInfer install ----
_NEEDS_RESTART = False
if importlib.util.find_spec("vllm") is None:
    print("\\n[setup] installing vLLM + transformers ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                    "vllm>=0.10.0", "transformers>=4.51.0",
                    "accelerate", "safetensors", "huggingface_hub"], check=True)
    _NEEDS_RESTART = True

if importlib.util.find_spec("flashinfer") is None:
    print("[setup] installing FlashInfer ...")
    def _cuda_suffix():
        try:
            import torch as _t
            cuda = (_t.version.cuda or "").strip()
            if not cuda: return None
            parts = cuda.split("."); major, minor = int(parts[0]), int(parts[1])
            supported = {"cu126","cu128","cu129","cu130","cu131"}
            s = f"cu{major}{minor}"
            if s in supported: return s
            cands = sorted(supported, key=lambda x: int(x[2:]))
            det = major*10 + minor
            best = None
            for c in cands:
                if int(c[2:]) <= det: best = c
            return best or cands[0]
        except Exception:
            return None
    _cidx = _cuda_suffix()
    print(f"  CUDA index suffix: {_cidx}")
    try:
        subprocess.run([sys.executable,"-m","pip","install","-q","-U",
                        "flashinfer-python","flashinfer-cubin"], check=False)
        if _cidx:
            subprocess.run([sys.executable,"-m","pip","install","-q","-U","--pre",
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
    print("\\n" + "=" * 64)
    print("FIRST INSTALL DONE. RESTART RUNTIME (Runtime->Restart) then re-run.")
    print("=" * 64)
    raise SystemExit("Restart runtime and re-run from this cell.")
print("\\n[setup] OK")
"""))


# ---------------------------------------------------------------------------
# Cell 3 - Phase 1 markdown
# ---------------------------------------------------------------------------
CELLS.append(md("""---

# Phase 1 - Warm-boot from snapshot

Reconstruct in ~30-60 s from the on-disk v7.5 snapshot:

- `ALL_QUERIES` (10 val queries, English text + gold list)
- `search_text[did]`, `doc_meta[did]` (citation, family, court_base, paragraph_role, language)
- `doc_statute_anchors[did]`, `_doc_to_concepts[did]`, `_doc_to_terms[did]`
- `PER_QUERY[qid]["final_topk"]` (the 50,000 candidates from the v7.5 14-channel RRF)
- `PER_QUERY[qid]["curve"]` (the fusion-baseline R@K curve - reference row in every table)
- `ALL_TARGETS[qid]` (LLM-expanded statute/concept/term targets)
- `ALL_HYDE_ASPECTS[qid]` (HyDE sub-aspects)
- `ALL_GOLD_DOC_SET[qid]` (gold doc IDs - what we measure recall against)
"""))


# ---------------------------------------------------------------------------
# Cell 4 - Warm-boot code
# ---------------------------------------------------------------------------
CELLS.append(code("""# Phase 1 - Warm-boot
import pandas as _pd

print(f"[warm-boot] loading from {SNAPSHOT_DIR}")
_t0 = time.time()
CONFIG = json.loads((SNAPSHOT_DIR / "config.json").read_text(encoding="utf-8"))

# val.csv (with local fallback if paths.json points to Drive)
_paths = json.loads((SNAPSHOT_DIR / "paths.json").read_text(encoding="utf-8"))
VAL_CSV = Path(_paths["val_csv"])
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
print(f"  ALL_QUERIES: {len(ALL_QUERIES)}")

# Corpus snapshot (~15 MB gzipped -> ~150 MB in memory)
with gzip.open(SNAPSHOT_DIR / "corpus_snapshot.json.gz", "rt", encoding="utf-8") as f:
    _c = json.load(f)
search_text         = {d: r["ct"] for d, r in _c.items()}
doc_meta            = {d: {"citation": r["cit"], "family": r["fam"],
                            "court_base": r["cb"], "paragraph_role": r["pr"],
                            "language": r["ln"]} for d, r in _c.items()}
doc_statute_anchors = {d: set(r["sa"]) for d, r in _c.items()}
_doc_to_concepts    = {d: set(r["cn"]) for d, r in _c.items()}
_doc_to_terms       = {d: set(r["tm"]) for d, r in _c.items()}
del _c
print(f"  doc_meta: {len(doc_meta):,}")

# PER_QUERY
_pq = json.loads((SNAPSHOT_DIR / "per_query_snapshot.json").read_text(encoding="utf-8"))
PER_QUERY = {
    qid: {"final_topk":      r["final_topk"],
          "curve":           {int(k): tuple(v) for k, v in r.get("curve", {}).items()},
          "channel_recalls": {ch: tuple(v) for ch, v in r.get("channel_recalls", {}).items()},
          "gold":            r.get("gold", 0),
          "gold_doc_ids":    r.get("gold_doc_ids", 0),
          "R_at_K":          r.get("R_at_K", 0.0)}
    for qid, r in _pq.items()
}

# Targets, aspects, gold
ALL_TARGETS = (json.loads((SNAPSHOT_DIR / "all_targets.json").read_text(encoding="utf-8"))
               if (SNAPSHOT_DIR / "all_targets.json").exists() else {})
ALL_HYDE_ASPECTS = (json.loads((SNAPSHOT_DIR / "hyde_aspects.json").read_text(encoding="utf-8"))
                    if (SNAPSHOT_DIR / "hyde_aspects.json").exists() else {})
_g = json.loads((SNAPSHOT_DIR / "gold_doc_sets.json").read_text(encoding="utf-8"))
ALL_GOLD_DOC_SET = {qid: set(lst) for qid, lst in _g.items()}
print(f"  PER_QUERY: {len(PER_QUERY)}  ALL_TARGETS: {len(ALL_TARGETS)}  HYDE: {len(ALL_HYDE_ASPECTS)}")
print(f"  gold doc_ids total: {sum(len(s) for s in ALL_GOLD_DOC_SET.values()):,}")

# Sanity - fusion baseline R@K macro mean
def _macro_recall_at_K(K, per_query_rankings):
    rs = []
    for q in ALL_QUERIES:
        qid = q["query_id"]
        ranked = per_query_rankings.get(qid, [])[:K]
        g = ALL_GOLD_DOC_SET.get(qid, set())
        rs.append(len(set(ranked) & g) / max(1, len(g)))
    return sum(rs) / len(rs)

FUSION_RANKING = {q["query_id"]: PER_QUERY[q["query_id"]]["final_topk"] for q in ALL_QUERIES}
print("\\n[warm-boot] fusion-baseline R@K (macro mean over 10 val queries):")
for K in K_REPORT:
    r = _macro_recall_at_K(K, FUSION_RANKING)
    flag = "*** PASS >=0.80 ***" if r >= RECALL_TARGET else ""
    print(f"  K={K:>6}  R={r:.3f}  {flag}")

print(f"\\n[warm-boot] done in {time.time()-_t0:.1f}s")
"""))


# ---------------------------------------------------------------------------
# Cell 5 - Phase 2 markdown
# ---------------------------------------------------------------------------
CELLS.append(md("""---

# Phase 2 - Build the FULL feature dossier per (qid, did)

Every feature listed in the Arsenal table above is computed here. The output
is `PER_QUERY[qid]["dossier"][did] = {feature_dict}`. All arithmetic;
no LLM call. Total compute time: ~30 s for 10 queries x ~50k candidates.

### Phase 2a - helpers (chamber regex, doctrinal-density regex, area inference, canonicalizer)
These are general Swiss-legal-system primitives, not query-specific knowledge.
"""))


# ---------------------------------------------------------------------------
# Cell 6 - Dossier helpers
# ---------------------------------------------------------------------------
CELLS.append(code("""# Phase 2a - Dossier helpers
import re as _re_d

# ---- Statute-anchor canonicalizer ----
# Corpus stores anchors as "41 OR" (no "Art."). We must canonicalize the
# LLM-produced ALL_TARGETS[qid]["statute_targets"] before intersecting.
CODE_ALIAS = {
    "CPP": "StPO", "CP": "StGB", "CC": "ZGB", "CO": "OR",
    "LTF": "BGG", "LACI": "AVIG", "LAA": "UVG",
    "LP": "SchKG", "LDIP": "IPRG", "Cst": "BV", "Cst.": "BV",
    "STPO": "StPO", "OBG": "OR",
}
_ART_RE  = _re_d.compile(r"art\\.?\\s*(\\d+[a-z]?)", _re_d.I)
_CODE_RE = _re_d.compile(r"\\b([A-Z][A-Za-z]{1,8}\\.?)\\b")

def statute_anchor_canonical(raw):
    if not raw: return None
    s = str(raw).strip()
    m = _ART_RE.search(s)
    if not m: return None
    cands = [c.strip(".") for c in _CODE_RE.findall(s)
             if c.strip(".") not in ("Art","Abs","Ziff","lit","let","al","Bst")]
    if not cands: return None
    code = CODE_ALIAS.get(cands[-1], cands[-1])
    return f"{m.group(1)} {code}"

def canonicalize_target_statutes(targets):
    out = set()
    for raw in (targets.get("statute_targets") or []):
        c = statute_anchor_canonical(raw)
        if c: out.add(c)
    return out

# ---- Chamber -> coarse legal area (Swiss Federal Court structure) ----
CHAMBER_TO_AREA = {
    "I":"public_law","II":"public_law","III":"social_security","IV":"social_security",
    "V":"civil","VI":"criminal",
    "1B":"criminal_procedure","1C":"public_administrative",
    "2C":"tax_administrative","2D":"tax_administrative",
    "4A":"civil","4F":"civil",
    "5A":"civil_family_succession","5D":"civil","5F":"civil",
    "6B":"criminal","6F":"criminal","7B":"criminal",
    "8C":"social_unemployment","8D":"social_unemployment",
    "9C":"social_unemployment","9F":"social_unemployment",
}
_RE_CHAMBER_BGE = _re_d.compile(r"^BGE\\s+\\d+\\s+([IVX]+)\\s+\\d+")
_RE_CHAMBER_BGR = _re_d.compile(r"^(\\d[A-Z])_\\d+")

def chamber_for_citation(citation):
    if not citation: return None
    m = _RE_CHAMBER_BGE.match(citation)
    if m: return m.group(1)
    m = _RE_CHAMBER_BGR.match(citation)
    if m: return m.group(1)
    return None

def chamber_class(chamber):
    return CHAMBER_TO_AREA.get(chamber) if chamber else None

# ---- Legal area inference from query expansion (NOT hardcoded per query) ----
AREA_KEYWORDS = {
    "criminal_procedure": ["pretrial","detention","kollusion","kollusionsgefahr","stpo","cpp","untersuchung","haft","strafverfahren","procedure penale","detention provisoire"],
    "criminal": ["strafgesetz","criminal law","stgb","diritto penale","strafrecht","code penal"],
    "civil": ["civil law","obligationenrecht","art. or"," or ","contract","obligation","schaden","haftung","responsabilite","kaufvertrag","code des obligations"],
    "civil_family_succession": ["zgb","cc ","civil code","succession","erbschaft","scheidung","familienrecht","marriage","code civil"],
    "social_security": ["ahv","iv ","atsg","lpga","lai","invalidenversicherung","invalidity","rente","pension","ivg","social insurance"],
    "social_unemployment": ["unemployment","alv","lacl","arbeitslosen","assurance chomage"],
    "tax_administrative": ["tax","steuer","dbg","lifd","impot","imposta"],
    "public_administrative": ["verwaltungsverfahren","administrative procedure","rvog"],
    "public_law": ["constitutional","verfassung","bv ","cst ","grundrecht","fundamental right","freiheit"],
}

def legal_area_for_query(qid, targets, query_text):
    parts = [(query_text or "").lower()]
    for k in ("legal_area_keywords","concept_targets_en","statute_targets"):
        v = (targets or {}).get(k) or []
        if v: parts.extend(str(x).lower() for x in v)
    blob = " ".join(parts)
    scores = {a: sum(1 for kw in kws if kw in blob) for a, kws in AREA_KEYWORDS.items()}
    best = max(scores, key=scores.get) if scores else "unknown"
    return best if scores.get(best, 0) > 0 else "unknown"

# ---- Doctrinal-density regex (gold-vs-ambient AUC = 0.80-0.88) ----
_RULE_OPENER_DE = _re_d.compile(r"(?:^|\\s)(?:Nach|GemAss|Im\\s+Sinne\\s+von)\\s+Art\\.\\s*\\d+")
_RULE_OPENER_FR = _re_d.compile(r"(?:^|\\s)(?:Selon|ConformEment\\s+A|Aux\\s+termes\\s+de|En\\s+vertu\\s+de)\\s+l?'?\\s*art\\.\\s*\\d+", _re_d.I)
_RULE_OPENER_IT = _re_d.compile(r"(?:^|\\s)(?:Conformemente\\s+all'|Ai\\s+sensi\\s+dell')\\s*art\\.\\s*\\d+", _re_d.I)
_DOCTRINE_DE    = _re_d.compile(r"\\b(?:stAndige\\s+Rechtsprechung|nach\\s+der\\s+Rechtsprechung|Lehre\\s+und\\s+Rechtsprechung|Praxis\\s+des\\s+Bundesgerichts)\\b", _re_d.I)
_DOCTRINE_FR    = _re_d.compile(r"\\b(?:la\\s+jurisprudence|selon\\s+la\\s+doctrine|il\\s+est\\s+constant|jurisprudence\\s+constante)\\b", _re_d.I)
_DOCTRINE_IT    = _re_d.compile(r"\\b(?:la\\s+giurisprudenza|secondo\\s+la\\s+dottrina)\\b", _re_d.I)
_INTERNAL_CITE  = _re_d.compile(r"(?:ATF|BGE)\\s+\\d+\\s+[IVX]+\\s+\\d+\\s+(?:consid|E)\\.\\s*\\d")
_RULE_VERB_DE   = _re_d.compile(r"Art\\.\\s*\\d+[\\w\\s.,]{0,40}\\b(?:bestimmt|sieht\\s+vor|regelt|verlangt)\\b", _re_d.I)
_RULE_VERB_FR   = _re_d.compile(r"art\\.\\s*\\d+[\\w\\s.,]{0,40}\\b(?:dispose|prEvoit|prEcise|exige)\\b", _re_d.I)
_HARD_NEG_DE    = _re_d.compile(r"\\b(?:Sachverhalt|Verfahrensgeschichte|wird\\s+abgewiesen|Gerichtskosten|der\\s+PrAsident|der\\s+Gerichtsschreiber)\\b", _re_d.I)
_HARD_NEG_FR    = _re_d.compile(r"\\b(?:en\\s+fait|le\\s+recourant\\s+fait\\s+valoir|est\\s+rejetE|frais\\s+judiciaires|le\\s+greffier)\\b", _re_d.I)
_HARD_NEG_IT    = _re_d.compile(r"\\b(?:in\\s+fatto|spese\\s+giudiziarie|E\\s+respinto|il\\s+cancelliere)\\b", _re_d.I)
_STATUTE_REF    = _re_d.compile(r"\\bart(?:icle)?\\.?\\s*\\d+", _re_d.I)

def doctrinal_density(text, lang="de"):
    if not text or len(text) < 40: return 0.0
    lang = (lang or "de").lower()[:2]
    score = 0.30
    if lang == "de":
        if _RULE_OPENER_DE.search(text): score += 0.30
        if _DOCTRINE_DE.search(text):    score += 0.20
        if _RULE_VERB_DE.search(text):   score += 0.15
        if _HARD_NEG_DE.search(text):    score -= 0.40
    elif lang == "fr":
        if _RULE_OPENER_FR.search(text): score += 0.30
        if _DOCTRINE_FR.search(text):    score += 0.20
        if _RULE_VERB_FR.search(text):   score += 0.15
        if _HARD_NEG_FR.search(text):    score -= 0.40
    elif lang == "it":
        if _RULE_OPENER_IT.search(text): score += 0.30
        if _DOCTRINE_IT.search(text):    score += 0.20
        if _HARD_NEG_IT.search(text):    score -= 0.40
    n_internal = len(_INTERNAL_CITE.findall(text))
    if n_internal >= 3:   score += 0.20
    elif n_internal >= 1: score += 0.05
    n_refs = len(_STATUTE_REF.findall(text))
    refs_per_1000 = n_refs / max(1, len(text) / 1000)
    score += min(0.15, refs_per_1000 * 0.05)
    return max(0.0, min(1.0, score))

def is_dispositif_or_facts(text, lang="de"):
    if not text: return False
    lang = (lang or "de").lower()[:2]
    if lang == "de" and _HARD_NEG_DE.search(text): return True
    if lang == "fr" and _HARD_NEG_FR.search(text): return True
    if lang == "it" and _HARD_NEG_IT.search(text): return True
    return False

def is_federal_court(court_base):
    if not court_base: return False
    cb = str(court_base).strip().upper()
    if cb.startswith("BGE") or cb.startswith("BGER"): return True
    if _RE_CHAMBER_BGR.match(cb): return True
    return False

SUBSTANTIVE_ROLES = {"legal_standard","reasoning","application","holding"}
NOISE_ROLES       = {"notification","costs","dispositif"}

print("[helpers] sanity:")
print("  chamber('BGE 142 III 296') =", chamber_for_citation("BGE 142 III 296"))
print("  chamber('1B_490/2017')      =", chamber_for_citation("1B_490/2017"))
print("  doctrinal_density(DE rule) =", round(doctrinal_density("Nach Art. 41 OR bestimmt das Bundesgericht in stAndiger Rechtsprechung. Vgl. BGE 142 III 296 E. 4.1.", "de"), 3))
print("  doctrinal_density(DE facts)=", round(doctrinal_density("Sachverhalt: Die BeschwerdefUhrerin macht geltend, die Vorinstanz habe Art. 95 BGG verletzt.", "de"), 3))
"""))


# ---------------------------------------------------------------------------
# Cell 7 - Dossier compute markdown
# ---------------------------------------------------------------------------
CELLS.append(md("""### Phase 2b - Per-(qid, did) feature compute

Walk every (qid, did) in the 50k pools and populate `PER_QUERY[qid]["dossier"][did]`
with the full feature set. Per-doc features (chamber, doctrinal density, lead
anchors) are cached once; per-query features (statute intersection, concept
overlap, co-citation, case peers, aspect match) are computed per query."""))


# ---------------------------------------------------------------------------
# Cell 8 - Dossier compute
# ---------------------------------------------------------------------------
CELLS.append(code("""# Phase 2b - Dossier compute
DOSSIER_TOP_N = STAGE1_TOP_N

print(f"[dossier] computing per-doc cache for {len(doc_meta):,} unique docs ...")
_t0 = time.time()
_doc_chamber, _doc_chamber_cls, _doc_doctrinal = {}, {}, {}
_doc_hard_neg, _doc_is_federal, _doc_lead_anchors = {}, {}, {}

for did, meta in doc_meta.items():
    cit  = meta.get("citation") or did
    ch   = chamber_for_citation(cit)
    _doc_chamber[did]     = ch
    _doc_chamber_cls[did] = chamber_class(ch)
    _doc_is_federal[did]  = is_federal_court(meta.get("court_base","")) or (ch is not None)
    text = (search_text.get(did,"") or "")
    lang = (meta.get("language") or "de").lower()[:2]
    _doc_doctrinal[did]   = doctrinal_density(text, lang)
    _doc_hard_neg[did]    = is_dispositif_or_facts(text, lang)
    lead = text[:200].lower()
    anchors = doc_statute_anchors.get(did, set())
    lead_set = set()
    for a in anchors:
        al = a.lower()
        if al in lead:
            lead_set.add(a); continue
        m = re.match(r"art\\.?\\s*\\d+\\w*", al)
        if m and m.group(0) in lead:
            lead_set.add(a)
    _doc_lead_anchors[did] = lead_set
print(f"  per-doc cache built in {time.time()-_t0:.1f}s")

# ---- per-(qid, did) cross features ----
print("[dossier] per-query cross features ...")
for q in ALL_QUERIES:
    qid     = q["query_id"]
    qtext   = q["query_text"]
    targets = ALL_TARGETS.get(qid, {}) or {}
    aspects = ALL_HYDE_ASPECTS.get(qid, []) or []
    pool    = PER_QUERY[qid].get("final_topk", [])[:DOSSIER_TOP_N]

    target_statutes  = canonicalize_target_statutes(targets)
    target_concepts  = set(targets.get("concept_targets_en", []) or [])
    target_terms_de  = set(targets.get("term_targets_de", []) or [])
    target_terms_fr  = set(targets.get("term_targets_fr", []) or [])
    target_terms_it  = set(targets.get("term_targets_it", []) or [])

    aspect_kw = []
    for a in aspects:
        atxt = a.get("paragraph") or a.get("answer") or str(a) if isinstance(a, dict) else str(a)
        toks = set(t.lower() for t in re.findall(r"[A-Za-zAOUaouss]{4,}", atxt))
        aspect_kw.append(toks)

    legal_area = legal_area_for_query(qid, targets, qtext)

    # Co-citation: for each law in pool, count court paras in pool whose anchors cite it.
    law_cit_to_did = {}
    for did in pool:
        m = doc_meta.get(did, {})
        if m.get("family") == "law":
            cit = (m.get("citation") or "").strip()
            if cit: law_cit_to_did[cit] = did
    co_cite = Counter()
    for did in pool:
        m = doc_meta.get(did, {})
        if m.get("family") != "court": continue
        for sa in doc_statute_anchors.get(did, set()):
            tgt = law_cit_to_did.get(sa)
            if tgt: co_cite[tgt] += 1

    # Case peers
    case_peer = defaultdict(list)
    for did in pool:
        m = doc_meta.get(did, {})
        if m.get("family") == "court":
            cb = (m.get("court_base") or "").strip()
            if cb: case_peer[cb].append(did)

    dossier = {}
    for did in pool:
        m       = doc_meta.get(did, {})
        family  = m.get("family", "?")
        lang    = (m.get("language") or "?").lower()[:2]
        anchors = doc_statute_anchors.get(did, set())
        concs   = _doc_to_concepts.get(did, set())

        stat_full = anchors & target_statutes
        stat_lead = _doc_lead_anchors.get(did, set()) & target_statutes
        conc_ovl  = concs & target_concepts

        if   lang == "de": term_ovl = _doc_to_terms.get(did, set()) & target_terms_de
        elif lang == "fr": term_ovl = _doc_to_terms.get(did, set()) & target_terms_fr
        elif lang == "it": term_ovl = _doc_to_terms.get(did, set()) & target_terms_it
        else:              term_ovl = set()

        if aspect_kw:
            doc_text_low = (search_text.get(did,"") or "")[:1500].lower()
            anchor_blob  = " ".join(anchors).lower() + " " + " ".join(concs).lower()
            blob_toks    = set(t for t in re.findall(r"[A-Za-zAOUaouss]{4,}", doc_text_low + " " + anchor_blob))
            ascores      = [len(akw & blob_toks) for akw in aspect_kw]
            best_aspect  = max(range(len(ascores)), key=lambda i: ascores[i])
            aspect_score = ascores[best_aspect]
            n_addressed  = sum(1 for s in ascores if s >= 2)
        else:
            best_aspect, aspect_score, n_addressed = 0, 0, 0

        ch_cls   = _doc_chamber_cls.get(did)
        ch_match = (ch_cls is not None and ch_cls == legal_area)

        cb    = (m.get("court_base") or "").strip()
        peers = case_peer.get(cb, [])
        cp_n  = max(0, len(peers) - 1)
        cp_rule = any(doc_meta.get(p, {}).get("paragraph_role") in SUBSTANTIVE_ROLES
                      for p in peers if p != did)

        dossier[did] = {
            "family": family, "language": lang,
            "chamber": _doc_chamber.get(did), "chamber_class": ch_cls,
            "is_federal": _doc_is_federal.get(did, False),
            "paragraph_role": m.get("paragraph_role",""),
            "stat_overlap_full_n": len(stat_full),
            "stat_overlap_lead_n": len(stat_lead),
            "conc_overlap_n":      len(conc_ovl),
            "term_overlap_n":      len(term_ovl),
            "best_aspect":         int(best_aspect),
            "aspect_score":        int(aspect_score),
            "n_aspects_addressed": int(n_addressed),
            "chamber_match":       bool(ch_match),
            "legal_area":          legal_area,
            "co_cite_count":       int(co_cite.get(did, 0)),
            "case_peer_count":     int(cp_n),
            "case_peer_rule":      bool(cp_rule),
            "doctrinal":           round(float(_doc_doctrinal.get(did, 0.0)), 3),
            "is_dispositif":       bool(_doc_hard_neg.get(did, False)),
            "fusion_rank":         -1,  # filled next loop
        }
    # fusion-rank annotation
    for rank, did in enumerate(pool):
        if did in dossier:
            dossier[did]["fusion_rank"] = rank

    PER_QUERY[qid]["dossier"]    = dossier
    PER_QUERY[qid]["legal_area"] = legal_area

print(f"[dossier] complete in {time.time()-_t0:.1f}s")
print(f"\\n{'qid':<10}{'pool':>7}{'avg_stat':>10}{'avg_lead':>10}{'ch_match':>10}{'disp':>8}{'fed_court':>11}  legal_area")
for qid in sorted(PER_QUERY):
    d = PER_QUERY[qid].get("dossier", {})
    n = len(d)
    if not n: continue
    avg_stat = sum(x["stat_overlap_full_n"] for x in d.values()) / n
    avg_lead = sum(x["stat_overlap_lead_n"] for x in d.values()) / n
    n_chm    = sum(1 for x in d.values() if x["chamber_match"])
    n_disp   = sum(1 for x in d.values() if x["is_dispositif"])
    n_fed_ct = sum(1 for x in d.values() if x["family"]=="court" and x["is_federal"])
    print(f"{qid:<10}{n:>7}{avg_stat:>10.2f}{avg_lead:>10.2f}{n_chm:>10}{n_disp:>8}{n_fed_ct:>11}  {PER_QUERY[qid].get('legal_area','?')}")
"""))


# ---------------------------------------------------------------------------
# Cell 9 - Stage A markdown
# ---------------------------------------------------------------------------
CELLS.append(md("""---

# Stage A - Cheap rule pre-prune (recall-safe; < 1% gold loss expected)

Drop candidates that are *structurally* unable to be gold based on cheap signals:

- **Cantonal courts** - 102/102 val gold are Federal Supreme Court.
- **Noise paragraph roles** - `notification` / `costs` / `dispositif` only (kept `facts`/`procedural_history` because some val gold has those roles).
- **Hard-negative regex hits** - dispositif/facts/notification language.

This reduces 50k -> ~20-30k per query and makes the cross-encoder phase ~2x cheaper.
We do NOT prune on signal-strength (e.g. zero statute overlap) because some
gold law-side bridge articles have zero direct LLM-target overlap and rely on
channel-of-arrival.

Reported per query: pool size before/after, gold-before/after, drops by reason.
"""))


# ---------------------------------------------------------------------------
# Cell 10 - Stage A code
# ---------------------------------------------------------------------------
CELLS.append(code("""# Stage A - cheap rule-based pre-prune (no LLM)
print("[Stage A] applying cheap filters ...")
macro_gold_before = []
macro_gold_after  = []
for q in ALL_QUERIES:
    qid     = q["query_id"]
    pool    = PER_QUERY[qid].get("final_topk", [])[:STAGE1_TOP_N]
    dossier = PER_QUERY[qid].get("dossier", {})

    kept   = []
    drops  = Counter()
    for did in pool:
        d = dossier.get(did)
        if d is None:
            drops["no_dossier"] += 1; continue
        if d["family"] == "court" and not d["is_federal"]:
            drops["cantonal_court"] += 1; continue
        if d["paragraph_role"] in NOISE_ROLES:
            drops["noise_role"] += 1; continue
        if d["is_dispositif"]:
            drops["dispositif_regex"] += 1; continue
        kept.append(did)

    PER_QUERY[qid]["stage_a_kept"] = kept

    g = ALL_GOLD_DOC_SET.get(qid, set())
    gb = len(set(pool) & g)
    ga = len(set(kept) & g)
    macro_gold_before.append(gb / max(1, len(g)))
    macro_gold_after.append(ga / max(1, len(g)))
    pct_kept = len(kept) / max(1, len(pool)) * 100
    pct_gold = ga / max(1, gb) * 100 if gb else 100.0
    print(f"  [{qid}]  {len(pool):>5}->{len(kept):>5}  ({pct_kept:5.1f}%)  "
          f"gold {gb}->{ga} ({pct_gold:5.1f}%)  drops: {dict(drops)}")

print(f"\\n[Stage A] macro pool R before prune = {sum(macro_gold_before)/len(macro_gold_before):.3f}")
print(f"           macro pool R after  prune = {sum(macro_gold_after)/len(macro_gold_after):.3f}")
"""))


# ---------------------------------------------------------------------------
# Cell 11 - Phase 3 markdown
# ---------------------------------------------------------------------------
CELLS.append(md("""---

# Phase 3 - Cross-encoder reranking (Qwen3-Reranker-8B, SOTA config)

**Model choice locked**: prior rerank-only diagnostic (`swiss_citation_rerank_only_diagnostic.ipynb`)
ranked Qwen3-Reranker-8B > BGE-reranker-v2-m3 > Jina-reranker-v2-base-multilingual
on this exact data. We use Qwen3-Reranker-8B and stop the model bake-off.

**Two key fixes** vs the rerank-only baseline (`A_qwen3_raw`):

1. **Enriched doc representation**: instead of raw text only, we prepend
   `Citation`, `Type/role`, `Statute anchors`, `Concepts (English)`, `Terms`,
   then the body. The English concepts are the cross-lingual bridge tokens -
   they let the reranker score an EN query against a DE/FR/IT doc via a
   shared concept vocabulary.

2. **Cross-lingual instruction**: tells the reranker explicitly that the
   query is English and the document may be DE/FR/IT, and to treat language
   differences as a translation problem, not a mismatch. The Qwen team's own
   recommendation for cross-lingual reranking is to write the instruction in
   English even when the documents are in another language.

**Cache**: per-query scores saved to `OUT_DIR/rerank_scores/{qid}.json.gz`.
First run is ~12 min/query x 10 = ~2 h on Blackwell. Re-runs are seconds.
"""))


# ---------------------------------------------------------------------------
# Cell 12 - Load Qwen3-Reranker
# ---------------------------------------------------------------------------
CELLS.append(code("""# Phase 3a - Load Qwen3-Reranker-8B (vLLM, Blackwell-optimized)
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

QWEN3_MODEL = "Qwen/Qwen3-Reranker-8B"

if "qwen3_llm" not in globals() or globals().get("qwen3_llm") is None:
    print(f"[load] {QWEN3_MODEL} ...")
    _t0 = time.time()
    qwen3_tok = AutoTokenizer.from_pretrained(QWEN3_MODEL, trust_remote_code=True, padding_side="left")
    # vLLM 0.10 + Jupyter ipykernel workaround
    import sys as _sys_fix
    _o_out, _o_err = _sys_fix.stdout, _sys_fix.stderr
    _sys_fix.stdout = _sys_fix.__stdout__
    _sys_fix.stderr = _sys_fix.__stderr__
    try:
        qwen3_llm = LLM(model=QWEN3_MODEL, dtype="bfloat16",
                        max_model_len=MAX_LEN, gpu_memory_utilization=0.85,
                        enforce_eager=False, trust_remote_code=True)
    finally:
        _sys_fix.stdout, _sys_fix.stderr = _o_out, _o_err
    print(f"[load] done in {time.time()-_t0:.1f}s", flush=True)
else:
    print("[load] reusing qwen3_llm", flush=True)

QWEN3_YES_ID = qwen3_tok.convert_tokens_to_ids("yes")
QWEN3_NO_ID  = qwen3_tok.convert_tokens_to_ids("no")

QWEN3_PREFIX = ("<|im_start|>system\\n"
                "Judge whether the Document meets the requirements based on the Query "
                "and the Instruct provided. Note that the answer can only be "
                "\\"yes\\" or \\"no\\".<|im_end|>\\n<|im_start|>user\\n")
QWEN3_SUFFIX = "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"
QWEN3_PREFIX_IDS = qwen3_tok.encode(QWEN3_PREFIX, add_special_tokens=False)
QWEN3_SUFFIX_IDS = qwen3_tok.encode(QWEN3_SUFFIX, add_special_tokens=False)
QWEN3_MAX_BODY   = MAX_LEN - len(QWEN3_PREFIX_IDS) - len(QWEN3_SUFFIX_IDS) - 8
QWEN3_SP = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)

INSTR_CROSSLING = (
    "The Query is an English question about Swiss federal law. The Document is "
    "a Swiss federal law article (German) or a Swiss court paragraph (German, "
    "French, or Italian) - it may include English-language metadata (citation, "
    "statutes referenced, legal concepts). Decide YES if the Document is a "
    "relevant citation for the Query - i.e., the Query's legal question can be "
    "answered, supported, or directly discussed by this Document. Decide NO "
    "otherwise. Treat language differences as a translation problem, not a mismatch."
)

def _join_short(values, max_items=8, max_chars=240):
    return ", ".join(list(values)[:max_items])[:max_chars]

def repr_enriched(did):
    m = doc_meta.get(did, {})
    cit  = (m.get("citation") or "").strip()
    fam  = m.get("family") or "?"
    role = m.get("paragraph_role") or ""
    body = (search_text.get(did, "") or "").strip()
    sa    = sorted(doc_statute_anchors.get(did, set()))
    concs = sorted(_doc_to_concepts.get(did, set()))
    terms = sorted(_doc_to_terms.get(did, set()))
    parts = []
    if cit:   parts.append(f"Citation: {cit}")
    parts.append(f"Type: {fam} ({role})" if role else f"Type: {fam}")
    if sa:    parts.append(f"Statute anchors: {_join_short(sa)}")
    if concs: parts.append(f"Concepts (English): {_join_short(concs)}")
    if terms: parts.append(f"Terms: {_join_short(terms, max_items=6, max_chars=180)}")
    parts.append(f"Text: {body[:2000]}")
    return "\\n".join(parts)[:3000]

def _qwen3_score_batch(query, dids, instruction, repr_fn, show_tqdm=True):
    prompt_ids = []
    for d in dids:
        body = f"<Instruct>: {instruction}\\n<Query>: {query}\\n<Document>: {repr_fn(d)}"
        ids = qwen3_tok.encode(body, add_special_tokens=False)
        if len(ids) > QWEN3_MAX_BODY: ids = ids[:QWEN3_MAX_BODY]
        prompt_ids.append(QWEN3_PREFIX_IDS + ids + QWEN3_SUFFIX_IDS)
    prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]
    outs = qwen3_llm.generate(prompts, sampling_params=QWEN3_SP, use_tqdm=show_tqdm)
    scores = []
    for out in outs:
        lp = out.outputs[0].logprobs[0]
        y = lp.get(QWEN3_YES_ID); n = lp.get(QWEN3_NO_ID)
        yl = y.logprob if y else -1e9
        nl = n.logprob if n else -1e9
        mx = max(yl, nl)
        ey = math.exp(yl - mx); en = math.exp(nl - mx)
        scores.append(ey / (ey + en))
    return scores

print("[rerank] Qwen3-Reranker-8B ready (max_len={}, cross-lingual instr loaded)".format(MAX_LEN))
"""))


# ---------------------------------------------------------------------------
# Cell 13 - Rerank loop with cache
# ---------------------------------------------------------------------------
CELLS.append(code("""# Phase 3b - Rerank all 10 val queries (cached per-query)
RERANK_RESULTS = {}  # {qid: {did: rerank_score}}

# Cache key combines instruction text + STAGE1_TOP_N so changes invalidate.
_instr_hash = hashlib.sha1(INSTR_CROSSLING.encode("utf-8")).hexdigest()[:8]
CACHE_TAG   = f"qwen3sota_{_instr_hash}_top{STAGE1_TOP_N}"
print(f"[rerank] cache tag = {CACHE_TAG}")

def _cache_path(qid): return RERANK_CACHE / f"{qid}__{CACHE_TAG}.json.gz"

for q in ALL_QUERIES:
    qid   = q["query_id"]
    qtext = q["query_text"]
    cp    = _cache_path(qid)
    if cp.exists():
        with gzip.open(cp, "rt", encoding="utf-8") as f:
            RERANK_RESULTS[qid] = json.load(f)
        print(f"  [{qid}] cache HIT ({len(RERANK_RESULTS[qid])} scores)", flush=True)
        continue

    # Rerank on Stage-A survivors; everything pruned by Stage A gets score=0
    cands = PER_QUERY[qid].get("stage_a_kept", []) or PER_QUERY[qid]["final_topk"][:STAGE1_TOP_N]
    print(f"  [{qid}] reranking {len(cands)} cands ...", flush=True)
    t0 = time.time()
    scores = _qwen3_score_batch(qtext, cands, INSTR_CROSSLING, repr_enriched)
    sm = {d: float(s) for d, s in zip(cands, scores)}
    RERANK_RESULTS[qid] = sm
    with gzip.open(cp, "wt", encoding="utf-8") as f:
        json.dump(sm, f)
    print(f"  [{qid}] {time.time()-t0:6.1f}s  cache -> {cp.name}", flush=True)

# Build ranked-only lists (descending rerank score) for downstream use.
RERANK_RANKING = {}
for qid, sm in RERANK_RESULTS.items():
    RERANK_RANKING[qid] = [d for d, _ in sorted(sm.items(), key=lambda kv: -kv[1])]

print("\\n[rerank] R@K (macro, reranker only, Stage-A-survivors ordered by rerank score):")
for K in K_REPORT:
    r = _macro_recall_at_K(K, RERANK_RANKING)
    flag = "*** PASS >=0.80 ***" if r >= RECALL_TARGET else ""
    print(f"  K={K:>6}  R={r:.3f}  {flag}")
"""))


# ---------------------------------------------------------------------------
# Cell 14 - Phase 4 markdown
# ---------------------------------------------------------------------------
CELLS.append(md("""---

# Phase 4 - Hybrid late-fusion scoring (full arsenal)

For each (qid, did), compute a hybrid score that fuses the cross-encoder
output with all data derivatives. Two variants computed and reported in
parallel:

### Variant A - Weighted z-score sum
```
hybrid_A = w_r  * z(rerank_score)
         + w_f  * z(-fusion_rank)        # smaller rank = better
         + w_ls * z(lead_stat_overlap)   # SHARPEST single discriminator
         + w_c  * z(concept_overlap)     # cross-lingual bridge
         + w_t  * z(term_overlap)        # language-matched
         + w_a  * z(aspect_score)
         + w_ch * I[chamber_match]       # legal-area chamber alignment
         + w_pr * I[substantive_role]    # paragraph_role bonus
         - w_neg* I[noise_role|dispositif]  # hard-negative penalty
         + w_dd * doctrinal_density
         + w_cc * z(co_cite_count)       # hub-statute boost (laws)
         + w_cp * I[case_peer_rule]      # leading-case boost (courts)
```

### Variant B - Reciprocal Rank Fusion (RRF) over 4 expert rankings
- Expert 1: rerank_score (desc)
- Expert 2: fusion_rank (asc)
- Expert 3: composite_handcrafted (desc) - linear sum of structural + content features
- Expert 4: doctrinal_density + lead_stat (desc) - precision booster
```
rrf_score(d) = sum_i 1 / (k + rank_i(d)),  k = 60
```

Both variants are deterministic; weights default to research-grounded values
and can be re-tuned downstream. Reported: macro R@K curve for each variant.
"""))


# ---------------------------------------------------------------------------
# Cell 15 - Hybrid scoring
# ---------------------------------------------------------------------------
CELLS.append(code("""# Phase 4 - Hybrid late-fusion
import numpy as np

# ---- Weight tables (research-grounded; tune later if needed) ----
W = {
    "rerank":      2.0,
    "fusion":      1.5,
    "lead_stat":   1.5,
    "concept":     1.0,
    "term":        0.5,
    "aspect":      0.5,
    "chamber":     1.0,
    "subst_role":  0.5,
    "hard_neg":   -2.0,
    "doctrinal":   0.5,
    "co_cite":     0.5,
    "case_peer":   0.3,
}

def _z(vec):
    a = np.array(vec, dtype=np.float64)
    s = a.std()
    if s == 0: return np.zeros_like(a)
    return (a - a.mean()) / s

HYBRID_A_RANKING = {}  # weighted z-score
HYBRID_B_RANKING = {}  # RRF
HYBRID_HANDCRAFT_RANKING = {}  # rerank-free baseline (handcrafted only)
RRF_K = 60

print("[hybrid] computing per-query scores ...")
for q in ALL_QUERIES:
    qid     = q["query_id"]
    dossier = PER_QUERY[qid].get("dossier", {})
    # Population we score: Stage-A survivors (matches the rerank pool).
    pop = PER_QUERY[qid].get("stage_a_kept", []) or PER_QUERY[qid]["final_topk"][:STAGE1_TOP_N]
    rerank_sm = RERANK_RESULTS.get(qid, {})

    rerank_v   = [rerank_sm.get(d, 0.0)              for d in pop]
    fus_v      = [-(dossier.get(d, {}).get("fusion_rank", STAGE1_TOP_N)) for d in pop]
    lead_v     = [dossier.get(d, {}).get("stat_overlap_lead_n", 0)        for d in pop]
    full_v     = [dossier.get(d, {}).get("stat_overlap_full_n", 0)        for d in pop]
    conc_v     = [dossier.get(d, {}).get("conc_overlap_n", 0)             for d in pop]
    term_v     = [dossier.get(d, {}).get("term_overlap_n", 0)             for d in pop]
    asp_v      = [dossier.get(d, {}).get("aspect_score", 0)               for d in pop]
    chm_v      = [1.0 if dossier.get(d, {}).get("chamber_match")    else 0.0 for d in pop]
    sub_v      = [1.0 if dossier.get(d, {}).get("paragraph_role","") in SUBSTANTIVE_ROLES else 0.0 for d in pop]
    neg_v      = [1.0 if (dossier.get(d, {}).get("paragraph_role","") in NOISE_ROLES
                          or dossier.get(d, {}).get("is_dispositif", False)) else 0.0 for d in pop]
    doc_v      = [dossier.get(d, {}).get("doctrinal", 0.0)                 for d in pop]
    cc_v       = [dossier.get(d, {}).get("co_cite_count", 0)               for d in pop]
    cp_v       = [1.0 if dossier.get(d, {}).get("case_peer_rule") else 0.0 for d in pop]

    # ----- Variant A: weighted z-score sum -----
    hybrid_a = (W["rerank"]    * _z(rerank_v)
              + W["fusion"]    * _z(fus_v)
              + W["lead_stat"] * _z(lead_v)
              + W["concept"]   * _z(conc_v)
              + W["term"]      * _z(term_v)
              + W["aspect"]    * _z(asp_v)
              + W["chamber"]   * np.array(chm_v)
              + W["subst_role"]* np.array(sub_v)
              + W["hard_neg"]  * np.array(neg_v)
              + W["doctrinal"] * np.array(doc_v)
              + W["co_cite"]   * _z(cc_v)
              + W["case_peer"] * np.array(cp_v))
    order_a = np.argsort(-hybrid_a)
    HYBRID_A_RANKING[qid] = [pop[i] for i in order_a]

    # ----- Handcrafted-only (no reranker) -----
    handcraft = (W["fusion"]    * _z(fus_v)
               + W["lead_stat"] * _z(lead_v)
               + W["concept"]   * _z(conc_v)
               + W["term"]      * _z(term_v)
               + W["aspect"]    * _z(asp_v)
               + W["chamber"]   * np.array(chm_v)
               + W["subst_role"]* np.array(sub_v)
               + W["hard_neg"]  * np.array(neg_v)
               + W["doctrinal"] * np.array(doc_v)
               + W["co_cite"]   * _z(cc_v)
               + W["case_peer"] * np.array(cp_v))
    order_h = np.argsort(-handcraft)
    HYBRID_HANDCRAFT_RANKING[qid] = [pop[i] for i in order_h]

    # ----- Variant B: RRF of 4 expert rankings -----
    expert_orderings = []
    expert_orderings.append(np.argsort(-np.array(rerank_v)))        # rerank
    expert_orderings.append(np.argsort(-np.array(fus_v)))           # fusion (already negated)
    expert_orderings.append(np.argsort(-handcraft))                  # handcrafted composite
    precision_v = (np.array(doc_v) + _z(lead_v) * 0.5)
    expert_orderings.append(np.argsort(-precision_v))                # precision booster
    rrf = np.zeros(len(pop), dtype=np.float64)
    for ranking in expert_orderings:
        rank_of = np.empty_like(ranking)
        for r, idx in enumerate(ranking): rank_of[idx] = r
        rrf += 1.0 / (RRF_K + rank_of)
    order_b = np.argsort(-rrf)
    HYBRID_B_RANKING[qid] = [pop[i] for i in order_b]

print("\\n[hybrid] macro R@K per variant:")
print(f"  {'K':>6}{'fusion':>9}{'rerank':>9}{'hybrid_A':>10}{'hybrid_B':>10}{'handcraft':>11}")
for K in K_REPORT:
    r_fus = _macro_recall_at_K(K, FUSION_RANKING)
    r_rer = _macro_recall_at_K(K, RERANK_RANKING)
    r_hA  = _macro_recall_at_K(K, HYBRID_A_RANKING)
    r_hB  = _macro_recall_at_K(K, HYBRID_B_RANKING)
    r_hc  = _macro_recall_at_K(K, HYBRID_HANDCRAFT_RANKING)
    print(f"  {K:>6}{r_fus:>9.3f}{r_rer:>9.3f}{r_hA:>10.3f}{r_hB:>10.3f}{r_hc:>11.3f}")
"""))


# ---------------------------------------------------------------------------
# Cell 16 - Phase 5 markdown
# ---------------------------------------------------------------------------
CELLS.append(md("""---

# Phase 5 - R@K diagnostic sweep + smallest-K-for-R>=0.80 per config

Final summary tables:

1. **Per-query R@K** for each config (rows: 10 queries; columns: K = 50, 100, 200, 500, 1k, 2k, 5k, 10k, 20k, 30k, 40k, 50k).
2. **Macro mean R@K** for each config.
3. **Smallest K such that macro R@K >= 0.80** for each config - the headline number.
4. **Compression ratio** = `50_000 / smallest_K_at_R>=0.80` (how much we shrunk the pool).
"""))


# ---------------------------------------------------------------------------
# Cell 17 - R@K sweep + final tables
# ---------------------------------------------------------------------------
CELLS.append(code("""# Phase 5 - R@K diagnostic
CONFIGS = [
    ("fusion_baseline",   FUSION_RANKING),
    ("qwen3_sota",        RERANK_RANKING),
    ("hybrid_A_zsum",     HYBRID_A_RANKING),
    ("hybrid_B_rrf",      HYBRID_B_RANKING),
    ("handcraft_only",    HYBRID_HANDCRAFT_RANKING),
]

# --- 1) Per-query R@K table per config ---
print("=" * 110)
print("  PER-QUERY R@K (rows = 10 val queries)")
print("=" * 110)
for cfg_name, ranking in CONFIGS:
    print(f"\\n  [{cfg_name}]")
    hdr = f"  {'qid':<10}" + "".join(f"{('R@'+str(K)):>10}" for K in K_REPORT)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for q in ALL_QUERIES:
        qid = q["query_id"]
        g   = ALL_GOLD_DOC_SET.get(qid, set())
        gt  = ALL_TOTAL_GOLD.get(qid, len(g))
        row = f"  {qid:<10}"
        for K in K_REPORT:
            r = len(set(ranking.get(qid, [])[:K]) & g) / max(1, gt)
            row += f"{r:>10.3f}"
        print(row)

# --- 2) Macro mean R@K (single comparison table) ---
print("\\n" + "=" * 110)
print(f"  MACRO MEAN R@K (10 val queries, target R >= {RECALL_TARGET})")
print("=" * 110)
print(f"  {'config':<22}" + "".join(f"{('R@'+str(K)):>10}" for K in K_REPORT))
print("  " + "-" * 100)
macro_table = {}
for cfg_name, ranking in CONFIGS:
    row = f"  {cfg_name:<22}"
    macro_table[cfg_name] = {}
    for K in K_REPORT:
        r = _macro_recall_at_K(K, ranking)
        macro_table[cfg_name][K] = r
        row += f"{r:>10.3f}"
    print(row)

# --- 3) Smallest K such that R >= 0.80 ---
print("\\n" + "=" * 110)
print(f"  SMALLEST K such that macro R@K >= {RECALL_TARGET}  (lower is better; compression vs 50,000)")
print("=" * 110)
print(f"  {'config':<22}{'smallest_K':>14}{'R_at_that_K':>14}{'compression':>14}")
print("  " + "-" * 64)
smallest_at = {}
for cfg_name, _ in CONFIGS:
    smallest = None
    smallest_r = 0.0
    for K in K_REPORT:
        r = macro_table[cfg_name][K]
        if r >= RECALL_TARGET:
            smallest = K
            smallest_r = r
            break
    smallest_at[cfg_name] = (smallest, smallest_r)
    if smallest is None:
        print(f"  {cfg_name:<22}{'NONE':>14}{'-':>14}{'-':>14}    "
              f"(max R = {max(macro_table[cfg_name].values()):.3f})")
    else:
        comp = 50000 / smallest
        print(f"  {cfg_name:<22}{smallest:>14}{smallest_r:>14.3f}{comp:>13.1f}x")

# --- 4) Conclusion line ---
best_cfg = min((s for s in smallest_at.items() if s[1][0] is not None),
               key=lambda kv: kv[1][0], default=None)
print("\\n" + "=" * 110)
if best_cfg:
    name, (K, r) = best_cfg
    comp = 50000 / K
    print(f"  CONCLUSION: best config = {name}; can compress 50,000 -> {K:,} "
          f"({comp:.1f}x) while holding macro R = {r:.3f} >= {RECALL_TARGET}")
else:
    print(f"  CONCLUSION: NO config holds R >= {RECALL_TARGET} on the 10-query macro mean. "
          f"Reranking cannot lower the pool below 50k without breaching the recall floor.")
print("=" * 110)
"""))


# ---------------------------------------------------------------------------
# Cell 18 - Phase 6 markdown
# ---------------------------------------------------------------------------
CELLS.append(md("""---

# Phase 6 - Save final results

Saves:

- `rerank_final_diagnostic_results.json` - macro R@K curves, per-query R@K curves, smallest-K-for-R>=0.80 per config.
- `rerank_scores/{qid}__qwen3sota_<hash>_top<N>.json.gz` - per-query reranker scores (already saved during Phase 3).
- `hybrid_rankings/{qid}.json.gz` - ranked doc_id lists for each config.

These files let you (a) reproduce the diagnostic table without re-running the
reranker, (b) feed any of the rankings into downstream precision stages.
"""))


# ---------------------------------------------------------------------------
# Cell 19 - Save
# ---------------------------------------------------------------------------
CELLS.append(code("""# Phase 6 - Save
out = OUT_DIR / "rerank_final_diagnostic_results.json"
per_query_rk = {}
for cfg_name, ranking in CONFIGS:
    per_query_rk[cfg_name] = {}
    for q in ALL_QUERIES:
        qid = q["query_id"]
        g   = ALL_GOLD_DOC_SET.get(qid, set())
        gt  = ALL_TOTAL_GOLD.get(qid, len(g))
        per_query_rk[cfg_name][qid] = {
            str(K): len(set(ranking.get(qid, [])[:K]) & g) / max(1, gt)
            for K in K_REPORT
        }

payload = {
    "config": {
        "stage1_top_n":    STAGE1_TOP_N,
        "K_report":        list(K_REPORT),
        "recall_target":   RECALL_TARGET,
        "model":           "Qwen/Qwen3-Reranker-8B",
        "max_len":         MAX_LEN,
        "instruction":     INSTR_CROSSLING,
        "weights":         W,
        "rrf_k":           60,
    },
    "macro_R_at_K": {cfg: {str(K): v for K, v in d.items()} for cfg, d in macro_table.items()},
    "smallest_K_at_R_target": {
        cfg: ({"K": K, "R": r} if K is not None else None)
        for cfg, (K, r) in smallest_at.items()
    },
    "per_query_R_at_K": per_query_rk,
}
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"[save] -> {out}")

ranks_dir = OUT_DIR / "hybrid_rankings"
ranks_dir.mkdir(parents=True, exist_ok=True)
for cfg_name, ranking in CONFIGS:
    cfg_dir = ranks_dir / cfg_name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for qid, ord_list in ranking.items():
        with gzip.open(cfg_dir / f"{qid}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(ord_list, f)
print(f"[save] rankings -> {ranks_dir}")
print("\\n[done] Diagnostic complete. See macro_R_at_K and smallest_K_at_R_target in the JSON above.")
"""))


# ---------------------------------------------------------------------------
# Build the notebook
# ---------------------------------------------------------------------------
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"WROTE: {OUT}")
print(f"  cells: {len(CELLS)}  size: {OUT.stat().st_size:,} bytes")
