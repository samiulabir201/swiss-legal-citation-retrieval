"""Generate notebooks/swiss_citation_endgame_colab.ipynb.

This script writes ONE comprehensive Colab notebook that consolidates every
retrieval technique with manual on/off toggles for ablation. Each technique
is inlined (no .py imports of project modules).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "notebooks" / "swiss_citation_endgame_colab.ipynb"


# ---------------------------------------------------------------------------
# Helpers to build cells
# ---------------------------------------------------------------------------

def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Cell sources
# ---------------------------------------------------------------------------

CELL_0_MD = """# Swiss Citation Retrieval — Endgame Pipeline (Ablation-Ready)

**Goal.** Macro F1 in the **0.6 – 0.8** band on val (10 English queries) for the
Swiss-law citation set-retrieval task (closed vocabulary, exact match).

**Empirical ceiling we are fighting.**
- Dense embedding alone (Qwen3-Embedding-8B, full 2.16M corpus, max enrichment)
  caps at **R@1000 = 0.289 / Macro F1 = 0.041** on val. So dense is ONE channel
  among many, never the primary stage. (`research/personal_observations.md`,
  Observation 3.)
- Train gold has a **28.5% unrecoverable ceiling** (Class B+C). Calibrate only
  on val. (Observation 1.)
- The citation graph has **0.59% overlap** with gold co-occurrence pairs;
  it is not a co-prediction signal but is useful for context expansion.
  (Observation 2.)
- Train -> val shifts on three independent axes simultaneously: language
  (DE 99% -> EN 100%), citation count (median 2 -> 22), court share
  (1.2% -> 40.6%). (Observation 4.)

**Design principle (enrichment-first multi-channel).**
1. **Pre-filter** the 2.6M corpus with enrichment-derived inverted indices on
   English concepts / topics / legal area, narrowing to a ~500k-1M working pool.
2. **Multi-channel retrieval** (BM25 with law-abbrev expansion, dense vector,
   HyDE, statute anchors, case anchors, court_base sibling expansion, citation
   graph 1-hop, legal-area soft filter) fused with reciprocal-rank fusion +
   authority boost.
3. **Reranker** (Qwen3-Reranker-8B, official template, log_softmax([no, yes])).
4. **LLM judge** (Qwen3-8B with the 7-category Swiss-court rubric) on
   borderline candidates only. Auto-yes / auto-no / borderline zoning.
5. **Granularity post-filter** to expand bare-article predictions into corpus
   paragraph-children.
6. **Dynamic K** by score threshold OR train-tuned static K-sweep.

**Reference baselines to beat.**
- `research/Untitled75.ipynb` v12 (BM25 + Reranker + Judge): **val Macro F1 =
  0.777** with avg_k=13.0. This notebook ports its core stages verbatim.
- `research/colab_dense_embedding_test.ipynb`: dense alone Macro F1 = 0.041.

**Toggle architecture.** Every technique is gated by a `CONFIG['use_X']` flag
(default ON). Cell 22 sweeps each toggle OFF in isolation and reports the
val Macro F1 delta — the headline deliverable.
"""

CELL_1_CODE = '''# ============================================================================
# CELL 1 — Master CONFIG. Every toggle in one place. Default: every toggle ON.
# ============================================================================
#
# Required Drive files (upload manually before running):
#   {drive_root}/data/train.csv
#   {drive_root}/data/val.csv
#   {drive_root}/data/test.csv
#   {drive_root}/data/laws_de.csv
#   {drive_root}/data/court_considerations.csv
#   {drive_root}/artifacts/unified_retrieval.sqlite          (FTS5 + filter indexes)
#   {drive_root}/artifacts/citation_graph_extracted.sqlite   (optional, graph expansion)
#   {drive_root}/artifacts/law_authority_cards_v2_unified.jsonl
#   {drive_root}/artifacts/court_authority_cards_v5_unified.jsonl
#   {drive_root}/artifacts/embeddings/qwen3_8b_unified_chunk*.npy   (27 fp16 chunks)
#   {drive_root}/artifacts/embeddings/qwen3_8b_unified_manifest.parquet

from pathlib import Path

DRIVE_ROOT = "/content/drive/MyDrive/swiss_citation"

CONFIG = {
    # ---- pipeline toggles (default ON) ----
    "use_enrichment_prefilter":       True,
    "use_bm25":                       True,
    "use_bm25_lexicon_expansion":     True,   # Untitled75 token-level law-abbrev expansion
    "use_vector":                     True,
    "use_hyde":                       True,
    "use_query_german_expansion":     True,   # English -> German keyword expansion
    "use_citation_graph_expansion":   True,
    "use_legal_area_softfilter":      True,
    "use_statute_anchors":            True,
    "use_case_anchors":               True,
    "use_court_base_sibling_expansion": True,
    "use_authority_score_boost":      True,
    "use_granularity_resolver":       True,
    "use_reranker":                   True,
    "use_llm_judge":                  True,
    "use_agentic_retrieval":          False,  # opt-in only (slow)
    "use_dynamic_k_calibration":      True,

    # ---- numerical knobs ----
    "channel_budgets": {
        "bm25": 800, "vector": 800, "statute": 400, "case": 300,
        "graph": 200, "court_base": 200,
    },
    "rrf_k": 60,
    "channel_weights": {
        "bm25": 1.0, "vector": 0.9, "hyde": 0.6,
        "statute": 0.8, "case": 0.7, "graph": 0.4, "court_base": 0.4,
    },
    "authority_alpha": 0.15,
    "rerank_top_n": 200,
    "rerank_batch_size": 8,
    "rerank_max_seq_len": 4096,
    "rerank_alpha_retrieval": 0.7,
    "rerank_alpha_rerank": 0.3,
    "judge_auto_yes": 0.55,
    "judge_auto_no": 0.25,
    "judge_batch_size": 4,
    "judge_max_seq_len": 8192,
    "k_sweep_for_f1": [5, 7, 10, 13, 15, 20, 25, 30, 50, 75, 100],
    "dynamic_k_strategy": "score_threshold",  # or "static_K_sweep"
    "agentic_max_rounds": 2,

    # ---- prefilter knobs ----
    "prefilter_pool_target": 1_000_000,
    "prefilter_min_recall_check": True,

    # ---- HyDE ----
    "hyde_max_new_tokens": 220,
    "hyde_temperature": 0.3,

    # ---- paths (Drive) ----
    "drive_root":             DRIVE_ROOT,
    "data_dir":               f"{DRIVE_ROOT}/data",
    "artifacts_dir":          f"{DRIVE_ROOT}/artifacts",
    "embeddings_dir":         f"{DRIVE_ROOT}/artifacts/embeddings",
    "submissions_dir":        f"{DRIVE_ROOT}/submissions",
    "cache_dir":              f"{DRIVE_ROOT}/cache_endgame",
    "eval_out_dir":           f"{DRIVE_ROOT}/artifacts/eval",
    "unified_sqlite":         f"{DRIVE_ROOT}/artifacts/unified_retrieval.sqlite",
    "citation_graph_sqlite":  f"{DRIVE_ROOT}/artifacts/citation_graph_extracted.sqlite",
    "law_enrichment_jsonl":   f"{DRIVE_ROOT}/artifacts/law_authority_cards_v2_unified.jsonl",
    "court_enrichment_jsonl": f"{DRIVE_ROOT}/artifacts/court_authority_cards_v5_unified.jsonl",
    "embedding_manifest":     f"{DRIVE_ROOT}/artifacts/embeddings/qwen3_8b_unified_manifest.parquet",
    "embedding_chunk_glob":   "qwen3_8b_unified_chunk*.npy",
    "val_csv":                f"{DRIVE_ROOT}/data/val.csv",
    "train_csv":              f"{DRIVE_ROOT}/data/train.csv",
    "test_csv":               f"{DRIVE_ROOT}/data/test.csv",
    "laws_de_csv":            f"{DRIVE_ROOT}/data/laws_de.csv",
    "court_csv":              f"{DRIVE_ROOT}/data/court_considerations.csv",

    # ---- model IDs ----
    "embedding_model": "Qwen/Qwen3-Embedding-8B",
    "reranker_model":  "Qwen/Qwen3-Reranker-8B",
    "judge_model":     "Qwen/Qwen3-8B",
    "embedding_max_seq_len": 2048,
}

# Helper: assert path exists or raise loud error
def must_exist(path_str: str, label: str):
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(
            f"[{label}] expected path does not exist: {p}\\n"
            f"Upload it to Drive at the location set in CONFIG, then re-run."
        )
    return p

# Pretty-print the toggle table.
print("CONFIG toggles:")
for k, v in CONFIG.items():
    if k.startswith("use_"):
        print(f"  {k:40s} = {v}")
print()
print(f"Drive root: {DRIVE_ROOT}")
'''

CELL_2_CODE = '''# ============================================================================
# CELL 2 — Drive mount + dependency install
# ============================================================================
import sys, subprocess, os

IN_COLAB = False
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    IN_COLAB = True
    print("Drive mounted.")
except Exception:
    print("Not in Colab; assuming local paths still resolve.")

# Install pinned versions. Re-runs are no-ops.
PKGS = [
    "transformers>=4.51",
    "accelerate",
    "sentence-transformers",
    "rank_bm25",
    "faiss-cpu",
    "numpy",
    "pandas",
    "pyarrow",
    "tqdm",
    "deep_translator",
]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PKGS], check=False)
print("Dependencies installed.")

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Make cache dirs.
for k in ("cache_dir", "submissions_dir", "eval_out_dir"):
    Path(CONFIG[k]).mkdir(parents=True, exist_ok=True)

# GPU sanity check (every GPU-required cell echoes its requirement).
try:
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}  VRAM: {p.total_memory/1e9:.1f} GB")
    else:
        print("WARNING: no GPU detected. Reranker / Judge / HyDE cells will fail.")
except ImportError:
    print("torch not yet importable. After install, re-run this cell.")
'''

CELL_3_CODE = '''# ============================================================================
# CELL 3 — Data loading + corpus schema check
# ============================================================================
import time, sqlite3, pandas as pd, numpy as np

t0 = time.time()

train_csv = must_exist(CONFIG["train_csv"], "train_csv")
val_csv   = must_exist(CONFIG["val_csv"],   "val_csv")
test_csv  = must_exist(CONFIG["test_csv"],  "test_csv")

train_df = pd.read_csv(train_csv)
val_df   = pd.read_csv(val_csv)
test_df  = pd.read_csv(test_csv)

print(f"train: {len(train_df):,} queries")
print(f"val:   {len(val_df):,} queries")
print(f"test:  {len(test_df):,} queries")

# Corpus schema check via unified_retrieval.sqlite
sqlite_path = must_exist(CONFIG["unified_sqlite"], "unified_sqlite")
con_check = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
con_check.row_factory = sqlite3.Row
families = con_check.execute(
    "SELECT family, COUNT(*) AS n FROM documents GROUP BY family"
).fetchall()
print("\\nUnified corpus document counts:")
for r in families:
    print(f"  {r['family']:6s} : {r['n']:>10,}")
total_docs = sum(r["n"] for r in families)
print(f"  {'TOTAL':6s} : {total_docs:>10,}")

# Sanity-check key columns
cols = [r[1] for r in con_check.execute("PRAGMA table_info(documents)").fetchall()]
required = {"doc_id", "family", "citation", "vector_text", "authority_score"}
missing = required - set(cols)
if missing:
    raise RuntimeError(f"unified_retrieval.sqlite missing required cols: {missing}")
con_check.close()
print(f"\\nLoaded in {time.time() - t0:.2f}s")
'''

CELL_4_CODE = '''# ============================================================================
# CELL 4 — Enrichment loading (toggle-gated)
# ============================================================================
import json, time
from collections import defaultdict

ENRICH_BY_DOC: dict[tuple[str, str], dict] = {}
ENRICH_BY_CIT: dict[str, dict] = {}

def _load_jsonl(p):
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out

if CONFIG["use_enrichment_prefilter"]:
    t0 = time.time()
    law_p   = must_exist(CONFIG["law_enrichment_jsonl"],   "law_enrichment_jsonl")
    court_p = must_exist(CONFIG["court_enrichment_jsonl"], "court_enrichment_jsonl")
    law_cards   = _load_jsonl(law_p)
    court_cards = _load_jsonl(court_p)
    for fam, cards in (("law", law_cards), ("court", court_cards)):
        for c in cards:
            cit = (c.get("citation") or c.get("citation_canon") or "").strip()
            if not cit:
                continue
            ENRICH_BY_CIT[cit] = c
            doc_id = c.get("doc_id") or c.get("_doc_id") or f"{fam}::{cit}"
            ENRICH_BY_DOC[(fam, doc_id)] = c
    print(f"Loaded {len(law_cards):,} law cards + {len(court_cards):,} court cards "
          f"in {time.time() - t0:.1f}s")
    print(f"Indexed by citation: {len(ENRICH_BY_CIT):,}")
else:
    print("use_enrichment_prefilter=False -> skipping enrichment load. "
          "Cell 14 will become a pass-through.")
'''

CELL_5_CODE = r'''# ============================================================================
# CELL 5 — Granularity resolver (ported from scripts/granularity_resolver.py)
# ============================================================================
# Two granularities exist:
#   article-level   : "Art. 78 BV"
#   paragraph-level : "Art. 78 Abs. 1 BV"
# If paragraph children exist, the bare-article form is NOT in the corpus.
# This module expands bare-article predictions into corpus paragraph-children.

import re, sqlite3
from functools import lru_cache

PARAGRAPH_UNIT_RE = re.compile(
    r"\b(?:Abs(?:\.|atz|aetze|atze)?|al\.?|Bst\.?|Buchstabe(?:n)?|lit\.?|"
    r"Ziff\.?|Ziffer|Nr\.?|Nummer(?:n)?|Satz|Unterabsatz|Unterabs\.?)\b",
    re.IGNORECASE,
)
BARE_ARTICLE_RE = re.compile(
    r"^Art\.\s+(?P<article>\d+[A-Za-z]*"
    r"(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies)?)"
    r"\s+(?P<law_code>.+)$",
    re.IGNORECASE,
)

def _norm_cit(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def is_article_level(c: str) -> bool:
    n = _norm_cit(c)
    return bool(n) and PARAGRAPH_UNIT_RE.search(n) is None

def _parse_article_law_code(c: str):
    n = _norm_cit(c)
    m = BARE_ARTICLE_RE.match(n)
    if not m: return None
    return m.group("article").strip(), m.group("law_code").strip()

@lru_cache(maxsize=200_000)
def expand_article_to_paragraphs(citation: str, db_path: str) -> tuple:
    """Expand a bare-article citation into corpus paragraph-children.
    Returns a tuple (immutable for lru_cache).
    """
    n = _norm_cit(citation)
    if not n:
        return (citation,) if citation else ()
    if not is_article_level(n):
        return (n,)
    parsed = _parse_article_law_code(n)
    if not parsed:
        return (n,)
    article, law_code = parsed
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # Direct lookup if granularity column is present.
        cols = {r[1] for r in con.execute("PRAGMA table_info(documents)").fetchall()}
        if {"law_code", "article", "granularity"} <= cols:
            rows = con.execute(
                "SELECT citation FROM documents "
                "WHERE family='law' AND law_code=? AND article=? AND granularity='paragraph'",
                (law_code, article),
            ).fetchall()
            sibs = [r[0] for r in rows if r and r[0]]
            if sibs:
                return tuple(sorted({_norm_cit(s) for s in sibs}))
        # LIKE fallback.
        like = f"Art. {article} %{law_code}"
        try:
            rows = con.execute(
                "SELECT DISTINCT citation FROM documents WHERE family='law' AND citation LIKE ?",
                (like,),
            ).fetchall()
        except sqlite3.OperationalError:
            return (n,)
        sibs = []
        for (cit,) in rows:
            if cit and PARAGRAPH_UNIT_RE.search(cit) and cit.endswith(law_code):
                sibs.append(cit)
        if sibs:
            return tuple(sorted({_norm_cit(s) for s in sibs}))
    finally:
        con.close()
    return (n,)

def apply_granularity_post_filter(predicted: list, db_path: str) -> list:
    seen, out = set(), []
    for raw in predicted:
        n = _norm_cit(raw)
        if not n: continue
        for ex in expand_article_to_paragraphs(n, db_path):
            if ex in seen: continue
            seen.add(ex); out.append(ex)
    return out

# Quick validation on val.csv: 3-row before/after.
if CONFIG["use_granularity_resolver"]:
    print("Granularity resolver — 3-row before/after on val.csv:")
    for i in range(min(3, len(val_df))):
        gold_str = str(val_df.iloc[i].get("gold_citations", "") or "")
        gold_list = [_norm_cit(c) for c in gold_str.split(";") if c.strip()][:6]
        expanded = apply_granularity_post_filter(gold_list, str(sqlite_path))
        print(f"  q={val_df.iloc[i]['query_id']}  IN={gold_list}\n     OUT={expanded[:8]}{'...' if len(expanded) > 8 else ''}")
else:
    print("use_granularity_resolver=False -> skipping post-filter.")
'''

CELL_6_CODE = '''# ============================================================================
# CELL 6 — Citation graph utilities (1-hop expansion)
# ============================================================================
# NOTE: Observation 2 says graph edges have 0.59% overlap with gold pairs.
# Use the graph for CONTEXT (filling in nearby reference text) only — never as
# a co-prediction signal in isolation. Channel weight is intentionally low.

import sqlite3
from collections import defaultdict

GRAPH_FORWARD: dict = defaultdict(set)   # citation -> {referenced_citations}
GRAPH_BACKWARD: dict = defaultdict(set)  # citation -> {citations_that_reference_it}
GRAPH_AVAILABLE = False

if CONFIG["use_citation_graph_expansion"]:
    graph_path = Path(CONFIG["citation_graph_sqlite"])
    if not graph_path.exists():
        print(f"[graph] {graph_path} not found; expansion disabled this run.")
    else:
        gcon = sqlite3.connect(f"file:{graph_path}?mode=ro", uri=True)
        try:
            tables = [r[0] for r in gcon.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            edge_table = None
            for cand in ("edges", "citation_edges", "citations_edges", "graph_edges"):
                if cand in tables:
                    edge_table = cand; break
            if edge_table is None:
                print(f"[graph] no recognised edge table among {tables}; expansion disabled.")
            else:
                cols = [r[1] for r in gcon.execute(f"PRAGMA table_info({edge_table})").fetchall()]
                src_col = "source" if "source" in cols else cols[0]
                tgt_col = "target" if "target" in cols else (cols[1] if len(cols) > 1 else cols[0])
                edge_rows = gcon.execute(
                    f"SELECT {src_col}, {tgt_col} FROM {edge_table}"
                ).fetchall()
                for s, t in edge_rows:
                    if not s or not t: continue
                    GRAPH_FORWARD[s].add(t)
                    GRAPH_BACKWARD[t].add(s)
                GRAPH_AVAILABLE = True
                print(f"[graph] loaded {len(edge_rows):,} edges; "
                      f"{len(GRAPH_FORWARD):,} forward sources.")
        finally:
            gcon.close()

def expand_via_graph(seed_citations, hops: int = 1, max_neighbors: int = 200):
    """Return citations reachable via outgoing edges within `hops` from seeds.
    Per Observation 2 we cap to a small budget; downstream channel weight is low.
    """
    if not GRAPH_AVAILABLE:
        return []
    frontier = set(seed_citations)
    seen = set(seed_citations)
    out = []
    for _ in range(hops):
        next_front = set()
        for c in frontier:
            for t in GRAPH_FORWARD.get(c, ()):
                if t in seen: continue
                seen.add(t); next_front.add(t); out.append(t)
                if len(out) >= max_neighbors:
                    return out
        frontier = next_front
        if not frontier: break
    return out

# Demo: 1-hop from "Art. 221 StPO".
if GRAPH_AVAILABLE:
    demo = expand_via_graph(["Art. 221 StPO"], hops=1, max_neighbors=10)
    print(f"[graph demo] 'Art. 221 StPO' 1-hop: {demo[:10]}")
'''

CELL_7_CODE = r'''# ============================================================================
# CELL 7 — BM25 (FTS5 over unified_retrieval.sqlite) + token-level law-abbrev
#         expansion (ported verbatim from research/Untitled75.ipynb `enhance()`)
# ============================================================================

import re, sqlite3, time, math
from collections import Counter, defaultdict

con_bm25 = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
con_bm25.row_factory = sqlite3.Row
con_bm25.execute("PRAGMA cache_size=-200000")
con_bm25.execute("PRAGMA temp_store=MEMORY")

# Confirm FTS5 table exists.
fts_tables = [r[0] for r in con_bm25.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_fts'"
).fetchall()]
if "documents_fts" not in fts_tables:
    raise RuntimeError(
        "documents_fts not in unified sqlite. "
        "scripts/build_unified_retrieval_corpus.py must have produced FTS5."
    )
print(f"FTS5 tables: {fts_tables}")

FTS5_RESERVED = {"AND", "OR", "NOT", "NEAR"}
FTS5_BAD_CHARS = re.compile(r'[\"\(\)\*\+\-\^]')

def fts5_match_string(query: str) -> str:
    cleaned = FTS5_BAD_CHARS.sub(" ", query)
    toks = []
    for tok in re.findall(r"[A-Za-zA-Za-z\xc0-\xff0-9_]+", cleaned):
        if len(tok) < 2: continue
        if tok.upper() in FTS5_RESERVED: continue
        toks.append(f'"{tok}"')
    return " OR ".join(toks)

# ---- token-level law-abbrev expansion (Untitled75 enhance()) ----------------
# Build a train-derived lexicon: for each English/German query token, count
# co-occurrence with each law abbreviation in train gold. At query time, we
# inject the top-5 most associated abbrevs as repeated tokens.

_TOKEN_RE = re.compile(r"[^\w\d]+", re.UNICODE)

def tokenise(t: str) -> list:
    if not t: return []
    return [w for w in _TOKEN_RE.split(t.lower().strip()) if len(w) >= 2 or w.isdigit()]

def _abbrev_of_citation(cit: str) -> str:
    parts = (cit or "").strip().split()
    return parts[-1].lower() if parts else ""

print("Building train-derived token->law-abbrev lexicon...")
t0 = time.time()
tok_to_abbrev = defaultdict(Counter)
total_per_tok = Counter()
for _, r in train_df.iterrows():
    raw_gold = str(r.get("gold_citations", "") or "")
    gold = [g.strip() for g in raw_gold.split(";") if g.strip()]
    abbrevs = {_abbrev_of_citation(g) for g in gold if re.match(r"^Art\.\s+\d", g)}
    abbrevs.discard("")
    if not abbrevs: continue
    toks = set(tokenise(str(r.get("query", ""))))
    for tk in toks:
        for ab in abbrevs:
            tok_to_abbrev[tk][ab] += 1
            total_per_tok[tk] += 1

# IDF approx from FTS5: get column counts.
n_docs = con_bm25.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

def _approx_idf(tok: str) -> float:
    # Rough: 1 / log(1 + total_per_tok). Untitled75 used BM25Okapi.idf; with
    # SQLite FTS5 we approximate via co-occurrence count.
    n = total_per_tok.get(tok, 0)
    if n <= 0: return 0.0
    return math.log(1.0 + n_docs / max(1, n))

def enhance_query_tokens(tokens: list) -> list:
    """Append top-5 train-associated law abbreviations to a token list.
    Mirrors Untitled75 enhance(). With CONFIG['use_bm25_lexicon_expansion']
    OFF, returns tokens unchanged.
    """
    base = list(dict.fromkeys(tokens))
    if not CONFIG["use_bm25_lexicon_expansion"]:
        return base
    sc = {}
    for tk in set(tokens):
        if tk not in tok_to_abbrev: continue
        idf = _approx_idf(tk)
        if idf < 1.0: continue
        tot = max(1, total_per_tok.get(tk, 1))
        for ab, cnt in tok_to_abbrev[tk].items():
            sc[ab] = sc.get(ab, 0.0) + (cnt / tot) * idf
    for ab, _ in sorted(sc.items(), key=lambda x: -x[1])[:5]:
        base.extend([ab] * 5)
    return base

print(f"Lexicon built in {time.time()-t0:.1f}s. "
      f"|vocab|={len(tok_to_abbrev):,} tokens.")

def bm25_search(query: str, k: int = 800) -> list:
    if not CONFIG["use_bm25"]:
        return []
    toks = tokenise(query)
    enhanced = enhance_query_tokens(toks)
    match = " OR ".join(f'"{t}"' for t in enhanced if t)
    if not match: return []
    rows = con_bm25.execute(
        "SELECT documents_fts.doc_id AS doc_id, "
        "       documents.family AS family, documents.citation AS citation, "
        "       documents.authority_score AS authority_score, "
        "       bm25(documents_fts) AS s "
        "FROM documents_fts JOIN documents USING (doc_id) "
        "WHERE documents_fts MATCH ? "
        "ORDER BY s LIMIT ?",
        (match, k),
    ).fetchall()
    # bm25() is more-negative-is-better; flip sign.
    return [
        {"doc_id": r["doc_id"], "family": r["family"], "citation": r["citation"],
         "authority_score": r["authority_score"], "channel": "bm25",
         "raw_score": -float(r["s"])}
        for r in rows
    ]

# Smoke-test
t0 = time.time()
demo = bm25_search("pretrial detention collusion risk Switzerland", k=20)
print(f"\nBM25 smoke test ({time.time()-t0:.2f}s, top 5):")
for c in demo[:5]:
    print(f"  [{c['family']}] {c['citation']:40s} score={c['raw_score']:.3f}")
'''

CELL_8_CODE = r'''# ============================================================================
# CELL 8 — Vector index (memory-mapped fp16 chunks; brute-force blocked dot)
# ============================================================================
# Ported from scripts/hybrid_retrieve.py: load chunked fp16 .npy files, mmap,
# compute top-K via blocked matmul. With L2-normalised fp16 vectors, dot ==
# cosine. The manifest gives us the row -> (doc_id, family, citation) mapping.

import numpy as np, time, re
import pyarrow.parquet as pq

VECTOR_AVAILABLE = False
VECTOR_CHUNKS = []   # list of (path, start_row, n_rows)
VECTOR_MANIFEST = None

if CONFIG["use_vector"]:
    manifest_path = Path(CONFIG["embedding_manifest"])
    emb_dir = Path(CONFIG["embeddings_dir"])
    chunks = sorted(emb_dir.glob(CONFIG["embedding_chunk_glob"]))
    if not manifest_path.exists():
        print(f"[vector] manifest missing at {manifest_path}; vector channel disabled.")
    elif not chunks:
        print(f"[vector] no chunks matching {CONFIG['embedding_chunk_glob']} in {emb_dir}")
    else:
        VECTOR_MANIFEST = pq.read_table(manifest_path).to_pandas()
        for path in chunks:
            arr = np.load(path, mmap_mode="r")
            n = arr.shape[0]
            m = re.search(r"chunk(\d+)\.npy$", path.name)
            chunk_idx = int(m.group(1)) if m else None
            start = chunk_idx * 100_000 if chunk_idx is not None else \
                    (VECTOR_CHUNKS[-1][1] + VECTOR_CHUNKS[-1][2] if VECTOR_CHUNKS else 0)
            VECTOR_CHUNKS.append((path, start, n))
        VECTOR_AVAILABLE = True
        total_rows = sum(c[2] for c in VECTOR_CHUNKS)
        print(f"[vector] manifest rows: {len(VECTOR_MANIFEST):,}")
        print(f"[vector] {len(VECTOR_CHUNKS)} chunks, covered_rows={total_rows:,}")
        if total_rows < len(VECTOR_MANIFEST):
            print(f"[vector] WARNING: chunks cover only {total_rows:,} of "
                  f"{len(VECTOR_MANIFEST):,} manifest rows.")
else:
    print("use_vector=False -> vector channel disabled.")

def vector_search(q_emb: np.ndarray, k: int = 800) -> list:
    if not VECTOR_AVAILABLE or q_emb is None:
        return []
    q = np.asarray(q_emb, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(q))
    if n <= 0: return []
    q = q / n
    heap_scores = np.full(k, -np.inf, dtype=np.float32)
    heap_rows   = np.full(k, -1, dtype=np.int64)
    block = 50_000
    for path, start, n_rows in VECTOR_CHUNKS:
        arr = np.load(path, mmap_mode="r")
        for s in range(0, n_rows, block):
            e = min(s + block, n_rows)
            a = np.asarray(arr[s:e], dtype=np.float32)
            scores = a @ q
            if scores.shape[0] >= k:
                idx_local = np.argpartition(-scores, k - 1)[:k]
                cand_scores = scores[idx_local]
                cand_rows = (start + s) + idx_local.astype(np.int64)
            else:
                cand_scores = scores
                cand_rows = (start + s) + np.arange(scores.shape[0], dtype=np.int64)
            combined_s = np.concatenate([heap_scores, cand_scores])
            combined_r = np.concatenate([heap_rows, cand_rows])
            top_idx = np.argpartition(-combined_s, k - 1)[:k]
            heap_scores = combined_s[top_idx]
            heap_rows = combined_r[top_idx]
    order = np.argsort(-heap_scores)
    heap_scores = heap_scores[order]; heap_rows = heap_rows[order]
    mask = (heap_rows >= 0) & np.isfinite(heap_scores)
    heap_scores = heap_scores[mask]; heap_rows = heap_rows[mask]
    rows = VECTOR_MANIFEST.iloc[heap_rows]
    out = []
    for s, row in zip(heap_scores, rows.itertuples(index=False)):
        out.append({
            "doc_id": getattr(row, "doc_id"),
            "family": getattr(row, "family", "?"),
            "citation": getattr(row, "citation", ""),
            "authority_score": None,
            "channel": "vector",
            "raw_score": float(s),
        })
    return out
'''

CELL_9_CODE = '''# ============================================================================
# CELL 9 — Query encoder (Qwen3-Embedding-8B, lazy GPU load)
# Requires: A100 / L4 24GB+ recommended. ~16 GB VRAM in bf16.
# ============================================================================
# Ported from scripts/encode_queries_qwen3_8b.py:
#   instruction prefix + last-token pool + L2 norm.

QWEN_QUERY_INSTRUCT = (
    "Instruct: Given an English-language legal question or scenario about "
    "Swiss federal law, retrieve the Swiss statute articles or federal court "
    "decision considerations that are most directly relevant to answering it.\\n"
    "Query: "
)

_query_encoder = None

def _load_query_encoder():
    """Lazy-load the SentenceTransformer-wrapped Qwen3-Embedding-8B."""
    global _query_encoder
    if _query_encoder is not None:
        return _query_encoder
    if not CONFIG["use_vector"]:
        return None
    print("[query-encoder] loading Qwen/Qwen3-Embedding-8B (~16 GB VRAM, bf16)")
    import torch
    from sentence_transformers import SentenceTransformer
    if not torch.cuda.is_available():
        raise RuntimeError("Query encoder requires CUDA.")
    m = SentenceTransformer(
        CONFIG["embedding_model"],
        device="cuda",
        model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"},
        tokenizer_kwargs={"padding_side": "left"},
    )
    m.max_seq_length = CONFIG["embedding_max_seq_len"]
    m.eval()
    _query_encoder = m
    return m

# Per-query embedding cache (Drive).
QUERY_EMB_CACHE_PATH = Path(CONFIG["cache_dir"]) / "query_embeddings.npy"
QUERY_EMB_CACHE_KEYS = Path(CONFIG["cache_dir"]) / "query_embeddings_keys.json"

import json
import hashlib
import numpy as np
def _q_hash(q: str) -> str:
    return hashlib.sha1(q.encode("utf-8")).hexdigest()

_query_emb_cache = {}
if QUERY_EMB_CACHE_KEYS.exists() and QUERY_EMB_CACHE_PATH.exists():
    keys = json.loads(QUERY_EMB_CACHE_KEYS.read_text(encoding="utf-8"))
    arr = np.load(QUERY_EMB_CACHE_PATH)
    if len(keys) == arr.shape[0]:
        for k, v in zip(keys, arr):
            _query_emb_cache[k] = v

def _save_query_emb_cache():
    if not _query_emb_cache: return
    keys = list(_query_emb_cache.keys())
    arr = np.stack([_query_emb_cache[k] for k in keys])
    np.save(QUERY_EMB_CACHE_PATH, arr)
    QUERY_EMB_CACHE_KEYS.write_text(json.dumps(keys), encoding="utf-8")

def encode_query(text: str) -> np.ndarray:
    h = _q_hash(text)
    if h in _query_emb_cache:
        return _query_emb_cache[h]
    enc = _load_query_encoder()
    if enc is None:
        return None
    full = QWEN_QUERY_INSTRUCT + str(text)
    import torch
    with torch.inference_mode():
        emb = enc.encode([full], convert_to_numpy=True, normalize_embeddings=True)
    v = emb[0].astype(np.float32)
    _query_emb_cache[h] = v
    _save_query_emb_cache()
    return v

print("Query encoder helpers ready (lazy-loaded on first call to encode_query).")
'''

CELL_10_CODE = r'''# ============================================================================
# CELL 10 — Query understanding (anchor extraction)
# Ported from scripts/hybrid_retrieve.py: parse_query_anchors.
# ============================================================================

BGE_FULL_RE = re.compile(
    r"\bBGE\s+\d{3}\s+[IVX]{1,4}\s+\d+[a-z]?(?:\s+E\.\s*[\d\.]+)?\b"
)
DOCKET_BASE_RE = re.compile(r"\b\d{1,2}[A-Z]{1,4}[_\.]\d{1,5}/\d{4}\b")
ART_PATTERN = re.compile(
    r"\bArt\.?\s*(\d+[a-z]?)(?:\s+(?:Abs|Bst|Lit|Ziff|Ch|Cpv)\.?\s*\d+\w*)*"
    r"\s+([A-Z][A-Za-z]{1,8}|\d{3}\.\d+)",
    re.IGNORECASE,
)
KNOWN_LAW_CODES = {
    "AIG", "AI", "ALC", "AMLA", "ATSG", "AVIG", "AHVG", "AsylG", "AuG",
    "BankG", "BetmG", "BGFA", "BGG", "BVG", "BV", "CC", "CEDH", "CO",
    "CPC", "CP", "CPP", "Cst", "DBG", "DSG", "EMRK", "FINMAG", "IVG",
    "IPRG", "KG", "KVG", "LAI", "LAMal", "LAVS", "LEI", "LEtr", "LIFD",
    "LP", "LPGA", "LTF", "LPP", "LStup", "LAsi", "LAA", "MSchG", "MWSTG",
    "NHG", "OJ", "OG", "OR", "PatG", "RPG", "SchKG", "StGB", "StPO",
    "SVG", "UVG", "URG", "UWG", "VwVG", "ZGB", "ZPO",
}

def parse_query_anchors(query: str) -> dict:
    cases   = sorted(set(BGE_FULL_RE.findall(query)))
    dockets = sorted(set(DOCKET_BASE_RE.findall(query)))
    statutes_raw = ART_PATTERN.findall(query)
    statutes, law_codes = [], set()
    for art_num, code in statutes_raw:
        statutes.append(f"Art. {art_num} {code}")
        if code in KNOWN_LAW_CODES:
            law_codes.add(code)
        if re.fullmatch(r"\d{3}\.\d+", code):
            law_codes.add(code)
    for tok in re.findall(r"\b([A-Z][A-Za-z]{1,8})\b", query):
        if tok in KNOWN_LAW_CODES:
            law_codes.add(tok)
    return {
        "cases": cases,
        "dockets": dockets,
        "statutes": sorted(set(statutes)),
        "law_codes": sorted(law_codes),
    }

# Demo on val.
print("Anchor extraction — first 3 val queries:")
for i in range(min(3, len(val_df))):
    q = str(val_df.iloc[i]["query"])
    a = parse_query_anchors(q)
    print(f"  q={val_df.iloc[i]['query_id']}  anchors={a}")
'''

CELL_11_CODE = '''# ============================================================================
# CELL 11 — HyDE: hypothetical-answer expansion (toggle-gated)
# GPU required: shares the Qwen3-8B judge model when present, else lazy loads.
# ============================================================================
import json, hashlib

HYDE_CACHE_PATH = Path(CONFIG["cache_dir"]) / "hyde_cache.json"
_hyde_cache = {}
if HYDE_CACHE_PATH.exists():
    try:
        _hyde_cache = json.loads(HYDE_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        _hyde_cache = {}

_hyde_model = None  # lazy
_hyde_tokenizer = None

HYDE_PROMPT_DE = (
    "You are a Swiss legal scholar. Given the English question below, write a "
    "concise 2-3 sentence GERMAN-language hypothetical legal answer that would "
    "match the wording of relevant Swiss statutes (StPO, StGB, OR, ZGB, BV, etc.) "
    "and BGE court decisions. Use Swiss legal vocabulary "
    "(Untersuchungshaft, Kollusionsgefahr, Verhaeltnismaessigkeit, etc.). "
    "Output German text only."
)

def _load_hyde_model():
    """Lazy-load Qwen3-8B for HyDE generation (~16 GB bf16)."""
    global _hyde_model, _hyde_tokenizer
    if _hyde_model is not None:
        return _hyde_model, _hyde_tokenizer
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("[hyde] loading Qwen/Qwen3-8B for HyDE generation (~16 GB)")
    tok = AutoTokenizer.from_pretrained(CONFIG["judge_model"], trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        CONFIG["judge_model"], torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    ).eval()
    _hyde_model, _hyde_tokenizer = mdl, tok
    return mdl, tok

def hyde_for_query(query: str) -> str:
    if not CONFIG["use_hyde"]:
        return ""
    h = hashlib.sha1(query.encode("utf-8")).hexdigest()
    if h in _hyde_cache:
        return _hyde_cache[h]
    import torch
    mdl, tok = _load_hyde_model()
    msgs = [
        {"role": "system", "content": HYDE_PROMPT_DE},
        {"role": "user",   "content": str(query)},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(
            **enc, max_new_tokens=CONFIG["hyde_max_new_tokens"],
            do_sample=True, temperature=CONFIG["hyde_temperature"],
            pad_token_id=tok.eos_token_id,
        )
    new = out[0][enc["input_ids"].shape[1]:]
    raw = tok.decode(new, skip_special_tokens=True).strip()
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    _hyde_cache[h] = raw
    HYDE_CACHE_PATH.write_text(json.dumps(_hyde_cache, ensure_ascii=False), encoding="utf-8")
    return raw

print("HyDE helper ready. First call lazy-loads Qwen3-8B (~16 GB).")
'''

CELL_12_CODE = '''# ============================================================================
# CELL 12 — Query German-keyword expansion (toggle-gated)
# Strategy: cached deep_translator GoogleTranslator. If offline / blocked,
# falls back to a small built-in German legal lexicon.
# ============================================================================
import json, hashlib

GERMAN_EXPANSION_CACHE_PATH = Path(CONFIG["cache_dir"]) / "german_expansion_cache.json"
_de_cache = {}
if GERMAN_EXPANSION_CACHE_PATH.exists():
    try:
        _de_cache = json.loads(GERMAN_EXPANSION_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        _de_cache = {}

# Built-in fallback lexicon (English -> German legal terms).
DE_LEX = {
    "detention": "Untersuchungshaft Sicherheitshaft",
    "pretrial": "Untersuchungshaft",
    "collusion": "Kollusionsgefahr Verdunkelungsgefahr",
    "flight risk": "Fluchtgefahr",
    "appeal": "Beschwerde Berufung",
    "proportionality": "Verhaeltnismaessigkeit",
    "fair trial": "rechtliches Gehoer",
    "criminal procedure": "Strafprozessordnung",
    "civil procedure": "Zivilprozessordnung",
    "contract": "Vertrag Obligationenrecht",
    "liability": "Haftung",
    "damages": "Schadenersatz",
    "compensation": "Entschaedigung",
    "court costs": "Gerichtskosten",
    "attorney fees": "Anwaltskosten",
    "constitution": "Bundesverfassung",
    "tax": "Steuer",
    "marriage": "Ehe",
    "divorce": "Scheidung",
    "inheritance": "Erbrecht",
    "property": "Eigentum",
    "bankruptcy": "Konkurs",
    "execution": "Betreibung Vollstreckung",
    "evidence": "Beweis",
    "admissibility": "Zulaessigkeit",
    "jurisdiction": "Zustaendigkeit",
    "asylum": "Asyl",
    "extradition": "Auslieferung",
    "insurance": "Versicherung",
    "disability": "Invaliditaet",
}

def _lex_expand(query: str) -> str:
    q_low = query.lower()
    bits = []
    for en, de in DE_LEX.items():
        if en in q_low:
            bits.append(de)
    return " ".join(bits)

def german_expand(query: str) -> str:
    if not CONFIG["use_query_german_expansion"]:
        return ""
    h = hashlib.sha1(query.encode("utf-8")).hexdigest()
    if h in _de_cache:
        return _de_cache[h]
    out = ""
    try:
        from deep_translator import GoogleTranslator
        out = GoogleTranslator(source="auto", target="de").translate(str(query)[:5000])
    except Exception as e:
        out = ""
    if not out:
        out = _lex_expand(query)
    _de_cache[h] = out
    GERMAN_EXPANSION_CACHE_PATH.write_text(json.dumps(_de_cache, ensure_ascii=False), encoding="utf-8")
    return out

print("German-expansion helper ready (Google translate w/ lexicon fallback).")
'''

CELL_13_CODE = r'''# ============================================================================
# CELL 13 — Multi-channel retrieve(): the core retriever
# ============================================================================
# Channels (each toggle-gated):
#   1) BM25 (with lexicon expansion if on)
#   2) Vector (raw query)
#   3) Vector HyDE (if on)
#   4) Statute anchors
#   5) Case anchors
#   6) Court_base sibling expansion (if on)
#   7) Citation graph 1-hop (if on)
#   8) Legal-area soft filter
# Fusion: reciprocal-rank fusion (k=rrf_k) with channel weights and
#         authority-score boost.

import time
from collections import defaultdict

# Helpers reusing con_bm25.

def statute_anchor_search(statutes: list, k: int = 400) -> list:
    if not CONFIG["use_statute_anchors"] or not statutes:
        return []
    like_clauses = " OR ".join(["statute LIKE ?"] * len(statutes))
    params = [f"%{st}%" for st in statutes] + [k]
    try:
        rows = con_bm25.execute(
            f"SELECT statute_links.doc_id AS doc_id, statute_links.statute AS hit, "
            f"       documents.family AS family, documents.citation AS citation, "
            f"       documents.authority_score AS authority_score "
            f"FROM statute_links JOIN documents USING (doc_id) "
            f"WHERE {like_clauses} "
            f"ORDER BY documents.authority_score DESC LIMIT ?",
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {"doc_id": r["doc_id"], "family": r["family"], "citation": r["citation"],
         "authority_score": r["authority_score"], "channel": "statute",
         "raw_score": float(r["authority_score"] or 0.0)}
        for r in rows
    ]

def case_anchor_search(cases: list, dockets: list, k: int = 300) -> list:
    if not CONFIG["use_case_anchors"]:
        return []
    targets = list(cases) + list(dockets)
    if not targets:
        return []
    like_clauses = " OR ".join(
        ["target_citation LIKE ?"] * len(targets)
        + ["target_base LIKE ?"] * len(targets)
    )
    params = [f"%{t}%" for t in targets] * 2 + [k]
    try:
        rows = con_bm25.execute(
            f"SELECT case_links.doc_id AS doc_id, case_links.target_citation AS hit, "
            f"       documents.family AS family, documents.citation AS citation, "
            f"       documents.authority_score AS authority_score "
            f"FROM case_links JOIN documents USING (doc_id) "
            f"WHERE {like_clauses} "
            f"ORDER BY documents.authority_score DESC LIMIT ?",
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {"doc_id": r["doc_id"], "family": r["family"], "citation": r["citation"],
         "authority_score": r["authority_score"], "channel": "case",
         "raw_score": float(r["authority_score"] or 0.0)}
        for r in rows
    ]

def court_base_sibling_expansion(top_court_cands: list, k: int = 200) -> list:
    """For court hits, expand to siblings sharing the same court_base
    (so we recover other considerations of the same decision)."""
    if not CONFIG["use_court_base_sibling_expansion"] or not top_court_cands:
        return []
    bases = []
    for c in top_court_cands[:20]:
        cit = (c.get("citation") or "")
        m = BGE_FULL_RE.search(cit)
        if m: bases.append(m.group(0).split(" E.")[0])
        else:
            m2 = DOCKET_BASE_RE.search(cit)
            if m2: bases.append(m2.group(0))
    bases = list(dict.fromkeys(bases))
    if not bases: return []
    like = " OR ".join(["citation LIKE ?"] * len(bases))
    params = [f"%{b}%" for b in bases] + [k]
    try:
        rows = con_bm25.execute(
            f"SELECT doc_id, family, citation, authority_score FROM documents "
            f"WHERE family='court' AND ({like}) "
            f"ORDER BY authority_score DESC LIMIT ?",
            params,
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {"doc_id": r["doc_id"], "family": r["family"], "citation": r["citation"],
         "authority_score": r["authority_score"], "channel": "court_base",
         "raw_score": float(r["authority_score"] or 0.0)}
        for r in rows
    ]

def graph_expansion_channel(seed_cands: list, k: int = 200) -> list:
    if not (CONFIG["use_citation_graph_expansion"] and GRAPH_AVAILABLE):
        return []
    seed_citations = [c.get("citation") for c in seed_cands[:20] if c.get("citation")]
    expanded = expand_via_graph(seed_citations, hops=1, max_neighbors=k)
    if not expanded:
        return []
    placeholders = ",".join("?" * len(expanded))
    try:
        rows = con_bm25.execute(
            f"SELECT doc_id, family, citation, authority_score FROM documents "
            f"WHERE citation IN ({placeholders}) LIMIT ?",
            list(expanded) + [k],
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {"doc_id": r["doc_id"], "family": r["family"], "citation": r["citation"],
         "authority_score": r["authority_score"], "channel": "graph",
         "raw_score": float(r["authority_score"] or 0.0)}
        for r in rows
    ]

def reciprocal_rank_fusion(channel_results: dict, *, k: int = 60,
                            weights: dict = None,
                            authority_alpha: float = 0.15) -> list:
    weights = weights or {}
    apply_authority = bool(CONFIG.get("use_authority_score_boost", True))
    scores = defaultdict(float)
    meta = {}
    per_channel = defaultdict(dict)
    for channel, results in channel_results.items():
        w = weights.get(channel, 1.0)
        for rank, item in enumerate(results, start=1):
            doc_id = item["doc_id"]
            scores[doc_id] += w / (k + rank)
            per_channel[channel][doc_id] = rank
            if doc_id not in meta:
                meta[doc_id] = {
                    "doc_id": doc_id,
                    "family": item.get("family"),
                    "citation": item.get("citation"),
                    "authority_score": item.get("authority_score"),
                }
    out = []
    for doc_id, sc in scores.items():
        entry = dict(meta[doc_id])
        authority = entry.get("authority_score") or 0.0
        try: authority = float(authority)
        except (TypeError, ValueError): authority = 0.0
        boost = (1.0 + authority_alpha * authority) if apply_authority else 1.0
        entry["fused_score"] = sc * boost
        entry["channel_ranks"] = {
            c: per_channel[c][doc_id] for c in per_channel if doc_id in per_channel[c]
        }
        out.append(entry)
    out.sort(key=lambda r: r["fused_score"], reverse=True)
    return out

def retrieve(query: str, query_emb=None, top_k: int = 1000, verbose: bool = False) -> dict:
    """Run the multi-channel retrieval pipeline. Toggle-aware."""
    timings = {}
    anchors = parse_query_anchors(query)
    results = {}

    # Optional German keyword expansion concatenated to query for BM25/vector
    de_expand = german_expand(query) if CONFIG["use_query_german_expansion"] else ""
    bm25_query = (query + " " + de_expand).strip() if de_expand else query

    t0 = time.time()
    results["bm25"] = bm25_search(bm25_query, CONFIG["channel_budgets"]["bm25"])
    timings["bm25"] = time.time() - t0

    t0 = time.time()
    if CONFIG["use_vector"] and query_emb is not None:
        results["vector"] = vector_search(query_emb, CONFIG["channel_budgets"]["vector"])
    else:
        results["vector"] = []
    timings["vector"] = time.time() - t0

    t0 = time.time()
    if CONFIG["use_hyde"]:
        try:
            hyde_text = hyde_for_query(query)
            if hyde_text:
                hyde_emb = encode_query(hyde_text)
                if hyde_emb is not None:
                    hits = vector_search(hyde_emb, CONFIG["channel_budgets"]["vector"])
                    for h in hits: h["channel"] = "hyde"
                    results["hyde"] = hits
                else:
                    results["hyde"] = []
            else:
                results["hyde"] = []
        except Exception as e:
            print(f"[hyde] failed: {e}; channel skipped")
            results["hyde"] = []
    else:
        results["hyde"] = []
    timings["hyde"] = time.time() - t0

    t0 = time.time()
    results["statute"] = statute_anchor_search(
        anchors["statutes"], CONFIG["channel_budgets"]["statute"]
    )
    results["case"] = case_anchor_search(
        anchors["cases"], anchors["dockets"], CONFIG["channel_budgets"]["case"]
    )
    timings["anchor"] = time.time() - t0

    # Seed-driven expansions (graph + court_base) use top BM25 + vector hits.
    seeds = (results["bm25"][:20] + results["vector"][:20])[:30]

    t0 = time.time()
    court_seeds = [c for c in seeds if c.get("family") == "court"]
    results["court_base"] = court_base_sibling_expansion(
        court_seeds, CONFIG["channel_budgets"]["court_base"]
    )
    timings["court_base"] = time.time() - t0

    t0 = time.time()
    results["graph"] = graph_expansion_channel(
        seeds, CONFIG["channel_budgets"]["graph"]
    )
    timings["graph"] = time.time() - t0

    # Legal-area soft filter (if on): re-weight top of each channel.
    if CONFIG["use_legal_area_softfilter"]:
        # Heuristic: extract law codes from anchors. Boost candidates whose
        # citation ends with one of those codes.
        boost_codes = set(anchors.get("law_codes") or [])
        if boost_codes:
            for ch_name, ch_results in results.items():
                for c in ch_results:
                    cit = (c.get("citation") or "")
                    last = cit.split()[-1] if cit else ""
                    if last in boost_codes:
                        c["raw_score"] = float(c.get("raw_score", 0.0)) * 1.1

    active = {k: v for k, v in results.items() if v}
    fused = reciprocal_rank_fusion(
        active, k=CONFIG["rrf_k"],
        weights=CONFIG["channel_weights"],
        authority_alpha=CONFIG["authority_alpha"],
    )[:top_k]

    if verbose:
        for k, v in results.items():
            print(f"  {k:12s}: {len(v):4d}")
        print(f"  fused      : {len(fused)}  timings={timings}")

    return {
        "query": query,
        "anchors": anchors,
        "channel_counts": {k: len(v) for k, v in results.items()},
        "channel_timings_s": timings,
        "candidates": fused,
        "raw_channel_results": results,
    }

# Smoke-test on one val query.
print("Multi-channel retrieve() smoke test on val[0]:")
v0 = val_df.iloc[0]
demo = retrieve(str(v0["query"]), query_emb=None, top_k=10, verbose=True)
print(f"  query: {v0['query'][:120]}...")
print(f"  fused top 5:")
for c in demo["candidates"][:5]:
    print(f"    [{c['family']}] {c['citation']:40s} fused={c['fused_score']:.4f}")
'''

CELL_14_CODE = r'''# ============================================================================
# CELL 14 — Enrichment-based pre-filter (the user's specific ask)
# ============================================================================
# Build an inverted index from concept tokens (English) -> {doc_id} using
# enrichment fields: english_summary, legal_topic, legal_question, legal_rule,
# english_legal_concepts, search_keywords, natural_language_queries.
# At query time, tokenise the query, union-lookup, narrow 2.4M -> ~500k-1M.

from collections import defaultdict
import time

ENRICHMENT_INDEX: dict = defaultdict(set)   # token -> {doc_id}
DOC_TO_FAM_CIT: dict = {}                    # doc_id -> (family, citation)

ENRICH_FIELDS = [
    "english_summary", "legal_topic", "legal_question", "legal_rule",
    "english_legal_concepts", "search_keywords", "natural_language_queries",
    "court_holding", "concepts_en", "topics", "legal_area",
]

def _enrich_text(card: dict) -> str:
    bits = []
    for f in ENRICH_FIELDS:
        v = card.get(f)
        if not v: continue
        if isinstance(v, (list, tuple, set)):
            bits.extend(str(x) for x in v if x)
        else:
            bits.append(str(v))
    return " ".join(bits)

def _enrich_tokenise(t: str) -> list:
    return tokenise(t)

if CONFIG["use_enrichment_prefilter"] and ENRICH_BY_DOC:
    t0 = time.time()
    print(f"Building enrichment inverted index over {len(ENRICH_BY_DOC):,} cards...")
    for (fam, doc_id), card in ENRICH_BY_DOC.items():
        cit = (card.get("citation") or card.get("citation_canon") or "").strip()
        if not cit: continue
        DOC_TO_FAM_CIT[doc_id] = (fam, cit)
        toks = _enrich_tokenise(_enrich_text(card))
        for t in set(toks):
            if len(t) < 3: continue
            ENRICHMENT_INDEX[t].add(doc_id)
    print(f"  index ready in {time.time()-t0:.1f}s; |vocab|={len(ENRICHMENT_INDEX):,}")

def enrichment_prefilter(query: str, target_pool: int = None) -> set:
    """Return a set of candidate doc_ids that share at least one concept token
    with the query. If pool exceeds target_pool, prune by token overlap count.
    Returns empty set when the index is unavailable (signals 'no filter')."""
    target_pool = target_pool or CONFIG["prefilter_pool_target"]
    if not CONFIG["use_enrichment_prefilter"] or not ENRICHMENT_INDEX:
        return set()
    q_toks = set(_enrich_tokenise(query))
    de_toks = set(_enrich_tokenise(german_expand(query))) if CONFIG["use_query_german_expansion"] else set()
    all_toks = q_toks | de_toks
    if not all_toks:
        return set()
    # Score doc_ids by token-overlap count.
    counter = defaultdict(int)
    for t in all_toks:
        if t in ENRICHMENT_INDEX:
            for d in ENRICHMENT_INDEX[t]:
                counter[d] += 1
    if len(counter) <= target_pool:
        return set(counter.keys())
    # Top-N by overlap count.
    sorted_docs = sorted(counter.items(), key=lambda x: -x[1])[:target_pool]
    return {d for d, _ in sorted_docs}

# Validation: per-query candidate-pool size + recall on val gold.
if CONFIG["use_enrichment_prefilter"] and ENRICHMENT_INDEX:
    print("\nValidation: enrichment prefilter pool size + gold recall on val:")
    cit_to_doc = {cit: doc_id for doc_id, (_, cit) in DOC_TO_FAM_CIT.items()}
    pool_sizes, recalls = [], []
    for _, r in val_df.iterrows():
        pool = enrichment_prefilter(str(r["query"]))
        gold = [c.strip() for c in str(r.get("gold_citations") or "").split(";") if c.strip()]
        gold_doc_ids = [cit_to_doc.get(g) for g in gold]
        gold_doc_ids = [g for g in gold_doc_ids if g]
        if not gold_doc_ids:
            recall = float("nan")
        else:
            recall = sum(1 for g in gold_doc_ids if g in pool) / len(gold_doc_ids)
        pool_sizes.append(len(pool))
        recalls.append(recall)
        print(f"  {r['query_id']:8s} pool={len(pool):>8,}  gold_recall={recall:.3f}")
    if pool_sizes:
        avg_pool = sum(pool_sizes)/len(pool_sizes)
        avg_rec  = sum(r for r in recalls if r == r) / max(1, sum(1 for r in recalls if r == r))
        print(f"  AVG pool={avg_pool:,.0f}  AVG recall={avg_rec:.3f}")
        if CONFIG["prefilter_min_recall_check"] and avg_rec < 0.95:
            print(f"  WARNING: avg recall {avg_rec:.3f} < 0.95 target. "
                  "Consider raising prefilter_pool_target or disabling prefilter.")
'''

CELL_15_CODE = r'''# ============================================================================
# CELL 15 — Reranker (Qwen3-Reranker-8B, ported from scripts/rerank_qwen3.py)
# GPU REQUIRED: ~22-30 GB VRAM at batch 8, max_seq_len 4096. A100 / L4-24GB ok.
# ============================================================================
# Score = log_softmax([no_logit, yes_logit])[..., 1].exp()  in [0, 1]
#
# Cite: scripts/rerank_qwen3.py:80-355 + research/Untitled75.ipynb cell 5
#       (rerank_batch + load_reranker, lines 1493-1561 of nb_dump.txt).

import os, hashlib, json, math, sys
from pathlib import Path

DEFAULT_RERANK_INSTRUCTION = (
    "Given an English-language Swiss legal question, judge whether the "
    "Document is a relevant authority (statute provision or court "
    "consideration) that the question's answer would cite."
)
QWEN_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)
QWEN_PREFIX = (
    "<|im_start|>system\n" + QWEN_SYSTEM_PROMPT
    + "<|im_end|>\n<|im_start|>user\n"
)
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

RERANK_CACHE_DIR = Path(CONFIG["cache_dir"]) / "rerank"
RERANK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _q_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

class QwenReranker:
    def __init__(self, model_id: str, dtype: str = "bf16", max_seq_len: int = 4096):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype, torch.bfloat16)
        self.max_seq_len = int(max_seq_len)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, padding_side="left", trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True,
        ).eval()
        # yes/no token IDs
        yes_ids = self.tokenizer("yes", add_special_tokens=False)["input_ids"]
        no_ids  = self.tokenizer("no",  add_special_tokens=False)["input_ids"]
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise RuntimeError(f"Unexpected yes/no tokenisation: yes={yes_ids}, no={no_ids}")
        self.yes_id, self.no_id = int(yes_ids[0]), int(no_ids[0])
        self.prefix_ids = self.tokenizer(QWEN_PREFIX, add_special_tokens=False)["input_ids"]
        self.suffix_ids = self.tokenizer(QWEN_SUFFIX, add_special_tokens=False)["input_ids"]

    def _build_inputs(self, queries, documents, instruction):
        import torch
        bodies = [
            f"<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {d}"
            for q, d in zip(queries, documents)
        ]
        body_budget = max(8, self.max_seq_len - len(self.prefix_ids) - len(self.suffix_ids))
        body_enc = self.tokenizer(
            bodies, add_special_tokens=False, truncation=True,
            max_length=body_budget, return_attention_mask=False,
        )
        full_ids = [list(self.prefix_ids) + list(ids) + list(self.suffix_ids)
                    for ids in body_enc["input_ids"]]
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        max_len = max(len(s) for s in full_ids)
        input_ids, attention_mask = [], []
        for s in full_ids:
            pad_n = max_len - len(s)
            input_ids.append([pad_id] * pad_n + s)
            attention_mask.append([0] * pad_n + [1] * len(s))
        device = next(self.model.parameters()).device
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        }

    def score_batch(self, query, documents, instruction=None,
                    batch_size=8, doc_ids=None):
        import torch
        instruction = instruction or DEFAULT_RERANK_INSTRUCTION
        n = len(documents)
        if n == 0: return []
        # Cache keyed by doc_id
        cache_path = RERANK_CACHE_DIR / f"{_q_hash(query)}.json"
        cache = {}
        if cache_path.exists():
            try: cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception: cache = {}
        scores = [None] * n
        todo = []
        for i in range(n):
            d = doc_ids[i] if doc_ids else None
            if d is not None and d in cache:
                scores[i] = float(cache[d])
            else:
                todo.append(i)
        for s in range(0, len(todo), batch_size):
            chunk = todo[s:s+batch_size]
            batch_docs = [documents[i] for i in chunk]
            inputs = self._build_inputs([query]*len(chunk), batch_docs, instruction)
            with torch.inference_mode():
                logits = self.model(**inputs).logits
                last = logits[:, -1, :]
                yes = last[:, self.yes_id]
                no_ = last[:, self.no_id]
                stacked = torch.stack([no_, yes], dim=-1)
                probs = torch.log_softmax(stacked, dim=-1)[:, 1].exp().float().cpu().tolist()
            for j, i in enumerate(chunk):
                p = float(probs[j]) if math.isfinite(probs[j]) else 0.0
                scores[i] = p
                if doc_ids is not None and doc_ids[i]:
                    cache[doc_ids[i]] = p
        if doc_ids is not None:
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        return [float(x) for x in scores]

_reranker_singleton = None
def _load_reranker():
    global _reranker_singleton
    if _reranker_singleton is not None:
        return _reranker_singleton
    if not CONFIG["use_reranker"]:
        return None
    print(f"[reranker] loading {CONFIG['reranker_model']} (~22-30 GB VRAM, bf16)")
    _reranker_singleton = QwenReranker(
        CONFIG["reranker_model"], dtype="bf16",
        max_seq_len=CONFIG["rerank_max_seq_len"],
    )
    return _reranker_singleton

def _doc_text_for_rerank(c: dict) -> str:
    """Compose a compact judging document from candidate metadata."""
    parts = [
        c.get("citation", ""),
        f"Family: {c.get('family','?')}",
    ]
    cit = (c.get("citation") or "")
    e = ENRICH_BY_CIT.get(cit) or {}
    if e:
        for k in ("english_summary", "legal_topic", "legal_rule",
                  "court_holding", "english_legal_concepts"):
            v = e.get(k)
            if not v: continue
            if isinstance(v, (list, tuple, set)):
                v = "; ".join(str(x) for x in v)
            parts.append(f"{k}: {str(v)[:400]}")
    txt = c.get("text") or ""
    if txt:
        parts.append(f"Text: {str(txt)[:600]}")
    return "\n".join(parts)

def _hydrate_text(cands: list):
    missing = [c for c in cands if not c.get("text")]
    if not missing: return
    ids = [c["doc_id"] for c in missing]
    chunk = 800
    text_by_id = {}
    for s in range(0, len(ids), chunk):
        sub = ids[s:s+chunk]
        q = ("SELECT doc_id, vector_text FROM documents "
             f"WHERE doc_id IN ({','.join('?' * len(sub))})")
        for r in con_bm25.execute(q, sub).fetchall():
            text_by_id[r["doc_id"]] = r["vector_text"] or ""
    for c in missing:
        c["text"] = text_by_id.get(c["doc_id"], "")

def _minmax(values):
    vals = [float(v) for v in values]
    if not vals: return []
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12: return [0.0]*len(vals)
    return [(v - lo) / (hi - lo) for v in vals]

def rerank_candidates(query: str, candidates: list, top_n: int = None) -> list:
    """Rerank top-N with Qwen3-Reranker-8B and emit fused_score (0.7/0.3)."""
    if not CONFIG["use_reranker"] or not candidates:
        return candidates
    top_n = top_n or CONFIG["rerank_top_n"]
    reranker = _load_reranker()
    if reranker is None:
        return candidates
    head = list(candidates[:top_n])
    tail = list(candidates[top_n:])
    _hydrate_text(head)
    documents = [_doc_text_for_rerank(c) for c in head]
    doc_ids = [str(c.get("doc_id","")) for c in head]
    rr = reranker.score_batch(
        query, documents,
        batch_size=CONFIG["rerank_batch_size"],
        doc_ids=doc_ids if all(doc_ids) else None,
    )
    retrieval_scores = [float(c.get("fused_score") or c.get("score") or 0.0) for c in head]
    norm = _minmax(retrieval_scores)
    out = []
    for c, n, r in zip(head, norm, rr):
        item = dict(c)
        item["rerank_score"] = float(r)
        item["retrieval_score_norm"] = float(n)
        item["fused_score"] = float(
            CONFIG["rerank_alpha_retrieval"] * n + CONFIG["rerank_alpha_rerank"] * r
        )
        out.append(item)
    out.sort(key=lambda x: x["fused_score"], reverse=True)
    for c in tail:
        item = dict(c)
        item["rerank_score"] = None
        out.append(item)
    return out

print("Reranker helpers ready (lazy-loaded on first call to rerank_candidates).")
'''

CELL_16_CODE = r'''# ============================================================================
# CELL 16 — LLM judge (Qwen3-8B, ported from scripts/llm_judge_qwen3.py)
# GPU REQUIRED: ~16 GB bf16. Greedy decoding for reproducibility.
# ============================================================================
# Verbatim 7-category JUDGE_SYSTEM prompt from research/Untitled75.ipynb
# (lines 1585-1608 of nb_dump.txt).

import json, hashlib, re

JUDGE_SYSTEM = """You are a Swiss Federal Court (Bundesgericht) legal citation expert.

DOMAIN KNOWLEDGE - Swiss Legal Citation Practice:
Swiss court decisions (BGE) and legal briefs cite provisions across multiple categories:

1. SUBSTANTIVE LAW: The core articles governing the legal issue (e.g., StGB for criminal offenses, OR for contracts, ZGB for civil matters)
2. DEFINITIONS: Articles that define key legal terms used in the case (e.g., Art. 8 ATSG defines invalidity)
3. PROCEDURAL RULES: Articles governing how the case is processed (StPO for criminal procedure, ZPO for civil procedure)
4. APPEAL PROVISIONS: Articles about legal remedies - Beschwerde (Art. 393ff StPO), Berufung, appeal deadlines
5. COST ALLOCATION: Articles about who pays court costs and attorney fees (Art. 422, 428 StPO; Art. 64 BGG)
6. COURT JURISDICTION: Articles defining which court decides (Art. 37/39 StBOG, Art. 100 BGG for Federal Court)
7. CONSTITUTIONAL PRINCIPLES: Fair trial (Art. 29 BV), proportionality, good faith (Art. 2 ZGB)

A query about pre-trial detention will cite detention rules AND appeal rules AND cost rules AND court jurisdiction.
A query about disability insurance will cite insurance provisions AND definitions AND procedural rules.

YOUR TASK: For each candidate article, read the German text carefully and decide YES or NO.
Say YES if the article belongs in ANY of the 7 categories above for this specific legal query.
Say NO only if the article is from a completely unrelated legal domain.

When uncertain, say YES - it is better to include a marginally relevant article than to miss one.

For each candidate, respond with exactly:
CITATION | VERDICT: YES or NO"""

JUDGE_CACHE_DIR = Path(CONFIG["cache_dir"]) / "judge"
JUDGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

class QwenJudge:
    def __init__(self, model_id: str, dtype: str = "bf16",
                 max_seq_len: int = 8192, do_sample: bool = False):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype, torch.bfloat16)
        self.max_seq_len = int(max_seq_len)
        self.do_sample = do_sample
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True,
        ).eval()

    def _cache(self, query, citation):
        sub = JUDGE_CACHE_DIR / hashlib.sha1(query.encode("utf-8")).hexdigest()
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{hashlib.sha1(citation.encode('utf-8')).hexdigest()}.json"

    def _generate(self, prompts):
        import torch
        chats = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": JUDGE_SYSTEM},
                 {"role": "user",   "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        enc = self.tokenizer(
            chats, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_seq_len,
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=200, do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return [
            self.tokenizer.decode(out[i][enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip()
            for i in range(out.shape[0])
        ]

    def judge_batch(self, query, candidates, batch_size=4):
        if not candidates: return []
        results = [None] * len(candidates)
        pending = []
        for i, c in enumerate(candidates):
            cit = str(c.get("citation",""))
            cp = self._cache(query, cit)
            if cp.exists():
                try:
                    results[i] = json.loads(cp.read_text(encoding="utf-8"))
                    continue
                except Exception:
                    pass
            pending.append(i)
        for s in range(0, len(pending), batch_size):
            chunk = pending[s:s+batch_size]
            prompts = [_build_judge_user_prompt(query, candidates[i]) for i in chunk]
            try:
                raws = self._generate(prompts)
            except Exception:
                raws = ["<error>"] * len(chunk)
            for i, raw in zip(chunk, raws):
                cit = str(candidates[i].get("citation",""))
                payload = _parse_judge_verdict(raw, cit)
                payload["raw_response"] = raw
                results[i] = payload
                cp = self._cache(query, cit)
                cp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return [r if r is not None else {"verdict":"yes","category":"unspecified","raw_response":""}
                for r in results]

def _build_judge_user_prompt(query: str, cand: dict) -> str:
    parts = [f"LEGAL QUERY: {str(query)[:600]}", "\n1 CANDIDATE to judge:\n"]
    cit = cand.get("citation","")
    txt = cand.get("text") or "(kein Text)"
    parts.append(f"[1] {cit}\n    German text: {str(txt)[:400]}")
    e = ENRICH_BY_CIT.get(cit) or {}
    if e.get("english_summary"):
        parts.append(f"    Summary: {str(e['english_summary'])[:300]}")
    parts.append("\nRespond with exactly: CITATION | VERDICT: YES or NO")
    return "\n".join(parts)

def _parse_judge_verdict(raw: str, citation: str) -> dict:
    text = raw or ""
    if "</think>" in text: text = text.split("</think>")[-1]
    text_low = text.lower()
    cat = "unspecified"
    for kw, label in (("substantive","substantive"),("definition","definitions"),
                      ("procedural","procedural"),("appeal","appeal"),
                      ("cost","costs"),("jurisdiction","jurisdiction"),
                      ("constitutional","constitutional")):
        if kw in text_low: cat = label; break
    verdict = None
    for line in text.splitlines():
        if "|" not in line: continue
        right = line.split("|")[-1].strip().upper()
        if "YES" in right: verdict="yes"; break
        if "NO" in right: verdict="no"; break
    if verdict is None:
        if re.search(r"\bYES\b", text, re.I): verdict="yes"
        elif re.search(r"\bNO\b", text, re.I): verdict="no"
        else: verdict="yes"  # default-YES on parse failure (notebook behaviour)
    return {"verdict": verdict, "category": cat}

_judge_singleton = None
def _load_judge():
    global _judge_singleton
    if _judge_singleton is not None: return _judge_singleton
    if not CONFIG["use_llm_judge"]: return None
    print(f"[judge] loading {CONFIG['judge_model']} (~16 GB VRAM, bf16)")
    _judge_singleton = QwenJudge(
        CONFIG["judge_model"], dtype="bf16",
        max_seq_len=CONFIG["judge_max_seq_len"], do_sample=False,
    )
    return _judge_singleton

def route_and_judge(query: str, candidates: list,
                    auto_yes_thresh: float = None, auto_no_thresh: float = None) -> list:
    """Three-zone router: auto-YES / auto-NO / LLM-judge borderline."""
    if not CONFIG["use_llm_judge"]:
        # Threshold-only fallback.
        ay = auto_yes_thresh or CONFIG["judge_auto_yes"]
        out = []
        for c in candidates:
            sc = float(c.get("fused_score") or c.get("score") or 0.0)
            item = dict(c); item["verdict"] = "yes" if sc >= ay else "no"
            item["category"] = "auto"
            out.append(item)
        return out
    ay = auto_yes_thresh or CONFIG["judge_auto_yes"]
    an = auto_no_thresh  or CONFIG["judge_auto_no"]
    if an > ay:
        raise ValueError("auto_no_thresh > auto_yes_thresh")
    enriched, bidx, bcands = [], [], []
    for i, c in enumerate(candidates):
        out = dict(c)
        score = float(out.get("fused_score") or out.get("score") or 0.0)
        if score >= ay:
            out["verdict"] = "yes"; out["category"] = "auto"
        elif score < an:
            out["verdict"] = "no"; out["category"] = "auto"
        else:
            out["verdict"] = None; bidx.append(i); bcands.append(out)
        enriched.append(out)
    if bcands:
        judge = _load_judge()
        if judge is None:
            for i in bidx: enriched[i]["verdict"] = "yes"; enriched[i]["category"] = "auto-fallback"
        else:
            _hydrate_text(bcands)
            verdicts = judge.judge_batch(query, bcands, batch_size=CONFIG["judge_batch_size"])
            for i, v in zip(bidx, verdicts):
                enriched[i]["verdict"] = v["verdict"]
                enriched[i]["category"] = v.get("category","unspecified")
    return enriched

print("LLM-judge helpers ready (lazy-loaded on first call to route_and_judge).")
'''

CELL_17_CODE = '''# ============================================================================
# CELL 17 — Multi-step agentic retrieval (toggle-gated, OFF by default)
# ============================================================================
# Round 1: standard retrieve(). Sample top-N candidates, ask the judge to spot
# missing concepts (e.g., "candidates are about general detention; user asks
# specifically about COLLUSION risk -> add Kollusionsgefahr").
# Round 2: re-retrieve with merged anchors. Cap CONFIG.agentic_max_rounds.
#
# Time/quality tradeoff: each extra round adds ~40-60s on A100. Most val
# queries hit ceiling within 1 round; reserve for the failing ones.

AGENTIC_PROMPT = """You are a Swiss legal research agent reviewing initial retrieval results.
LEGAL QUERY: {query}

The system returned these top candidates:
{snippet}

The user expects a comprehensive answer (median ~22 citations on val).
Identify 2-3 SPECIFIC sub-questions or German legal terms that the current
candidates miss but the query implies. Output ONLY a comma-separated list of
3-8 German legal keywords / terms. No prose.
"""

def agentic_retrieve(query: str, query_emb=None, top_k: int = 1000,
                     max_rounds: int = None) -> dict:
    if not CONFIG["use_agentic_retrieval"]:
        return retrieve(query, query_emb=query_emb, top_k=top_k)
    max_rounds = max_rounds or CONFIG["agentic_max_rounds"]
    cur_query = query
    last_result = retrieve(cur_query, query_emb=query_emb, top_k=top_k)
    for round_i in range(1, max_rounds):
        # Build snippet of top 8 candidates for the agent.
        snippet_lines = []
        for c in last_result["candidates"][:8]:
            snippet_lines.append(f"- [{c.get('family')}] {c.get('citation')}")
        snippet = "\\n".join(snippet_lines)
        prompt = AGENTIC_PROMPT.format(query=query[:500], snippet=snippet)
        try:
            judge = _load_judge()
            if judge is None: break
            import torch
            tok = judge.tokenizer; mdl = judge.model
            chat = tok.apply_chat_template(
                [{"role": "system", "content": "You are a focused legal research agent."},
                 {"role": "user",   "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            enc = tok(chat, return_tensors="pt", truncation=True, max_length=4096).to(mdl.device)
            with torch.no_grad():
                out = mdl.generate(**enc, max_new_tokens=120, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            new = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            if "</think>" in new: new = new.split("</think>")[-1].strip()
        except Exception as e:
            print(f"[agentic] round {round_i} failed: {e}")
            break
        # Merge new keywords into the query for next round.
        new_terms = [t.strip() for t in new.replace("\\n", ",").split(",") if t.strip()]
        if not new_terms: break
        cur_query = query + " " + " ".join(new_terms[:8])
        # Re-encode if vector channel.
        new_emb = encode_query(cur_query) if (CONFIG["use_vector"] and query_emb is not None) else query_emb
        last_result = retrieve(cur_query, query_emb=new_emb, top_k=top_k)
        last_result["agentic_round"] = round_i
        last_result["agentic_added_terms"] = new_terms
    return last_result

print("Agentic helper ready. Default off; flip CONFIG['use_agentic_retrieval']=True to enable.")
'''

CELL_18_CODE = '''# ============================================================================
# CELL 18 — Granularity post-filter wrapper
# ============================================================================
# Cell 5 already defined apply_granularity_post_filter. This cell wires it
# into the pipeline so predictions become corpus-comparable.

def apply_granularity_to_predictions(predicted_citations: list) -> list:
    """Expand bare-article predictions into corpus paragraph-children.
    Returns a deduplicated list preserving original order."""
    if not CONFIG["use_granularity_resolver"]:
        return list(dict.fromkeys(predicted_citations))
    return apply_granularity_post_filter(predicted_citations, str(sqlite_path))

# Smoke test.
sample_preds = ["Art. 78 BV", "Art. 221 Abs. 1 StPO", "BGE 137 IV 122 E. 4.2"]
expanded = apply_granularity_to_predictions(sample_preds)
print(f"Granularity expansion: {sample_preds}")
print(f"   -> {expanded}")
'''

CELL_19_CODE = '''# ============================================================================
# CELL 19 — Dynamic K calibration
# ============================================================================
# (a) score_threshold: cut by reranker/judge zone thresholds (variable size).
# (b) static_K_sweep:  pick the K from CONFIG.k_sweep_for_f1 that maximises
#     val Macro F1 (after the rest of the pipeline runs at top_k=max(K_sweep)).

def select_predictions_score_threshold(judged_candidates: list,
                                       fallback_top_k: int = 50) -> list:
    """Take all 'yes' verdicts; if empty, fall back to top-N by fused_score."""
    yes_cits = [c.get("citation") for c in judged_candidates
                if c.get("verdict") == "yes" and c.get("citation")]
    if yes_cits:
        return list(dict.fromkeys(yes_cits))
    # Fallback: top-N by fused_score regardless of verdict.
    sorted_c = sorted(judged_candidates,
                      key=lambda x: float(x.get("fused_score") or x.get("score") or 0.0),
                      reverse=True)
    return [c.get("citation") for c in sorted_c[:fallback_top_k] if c.get("citation")]

def select_predictions_static_k(candidates: list, k: int) -> list:
    sorted_c = sorted(candidates,
                      key=lambda x: float(x.get("fused_score") or x.get("score") or 0.0),
                      reverse=True)
    return [c.get("citation") for c in sorted_c[:k] if c.get("citation")]

print("Dynamic K calibration helpers ready.")
print(f"  Strategy: {CONFIG['dynamic_k_strategy']!r}")
print(f"  K-sweep:  {CONFIG['k_sweep_for_f1']}")
'''

CELL_20_CODE = '''# ============================================================================
# CELL 20 — END-TO-END predict() pipeline
# ============================================================================
# Wires Cells 11-19 together respecting toggles. Returns predicted citation
# set + per-stage diagnostics.

import time

def predict(query: str, query_id: str = None, return_intermediates: bool = False) -> dict:
    timings = {}
    diag = {"query_id": query_id, "query": query}

    # 1. Query embedding
    t0 = time.time()
    q_emb = encode_query(query) if CONFIG["use_vector"] else None
    timings["encode_query"] = time.time() - t0

    # 2. Enrichment prefilter (informational only — soft signal; the channels
    #    do their own retrieval. We log the pool size.)
    if CONFIG["use_enrichment_prefilter"] and ENRICHMENT_INDEX:
        t0 = time.time()
        pool = enrichment_prefilter(query)
        timings["enrichment_prefilter"] = time.time() - t0
        diag["enrichment_pool_size"] = len(pool)

    # 3. Retrieval (with optional agentic loop)
    t0 = time.time()
    if CONFIG["use_agentic_retrieval"]:
        ret = agentic_retrieve(query, query_emb=q_emb, top_k=1000)
    else:
        ret = retrieve(query, query_emb=q_emb, top_k=1000)
    timings["retrieve"] = time.time() - t0
    diag["channel_counts"] = ret["channel_counts"]
    diag["anchors"] = ret["anchors"]
    candidates = ret["candidates"]
    diag["n_after_retrieve"] = len(candidates)

    # 4. Reranker
    t0 = time.time()
    if CONFIG["use_reranker"] and candidates:
        candidates = rerank_candidates(query, candidates, top_n=CONFIG["rerank_top_n"])
    timings["rerank"] = time.time() - t0
    diag["n_after_rerank"] = len(candidates)

    # 5. LLM judge zone routing
    t0 = time.time()
    judged = route_and_judge(query, candidates) if candidates else []
    timings["judge"] = time.time() - t0
    if judged:
        diag["zone_counts"] = {
            "yes":   sum(1 for c in judged if c.get("verdict") == "yes"),
            "no":    sum(1 for c in judged if c.get("verdict") == "no"),
            "borderline": sum(1 for c in judged if c.get("verdict") not in ("yes", "no")),
        }

    # 6. K-selection
    t0 = time.time()
    if CONFIG["use_dynamic_k_calibration"] and CONFIG["dynamic_k_strategy"] == "score_threshold":
        preds_raw = select_predictions_score_threshold(judged or candidates, fallback_top_k=20)
    else:
        # Static-K fallback uses the SMALLEST K in the sweep (caller may rerank
        # via the K-sweep evaluation in Cell 22).
        preds_raw = select_predictions_static_k(judged or candidates, k=CONFIG["k_sweep_for_f1"][0])
    timings["select"] = time.time() - t0
    diag["n_before_granularity"] = len(preds_raw)

    # 7. Granularity post-filter (corpus paragraph expansion)
    t0 = time.time()
    preds = apply_granularity_to_predictions(preds_raw)
    timings["granularity"] = time.time() - t0
    diag["n_final"] = len(preds)
    diag["timings"] = timings

    out = {"predictions": preds, "diagnostics": diag}
    if return_intermediates:
        out["candidates_after_retrieve"] = ret["candidates"][:200]
        out["candidates_after_judge"]    = judged[:200]
        out["raw_predictions"]           = preds_raw
    return out

# Smoke test on val[0] without GPU-heavy stages (force toggles off temporarily).
print("predict() smoke test — first val query (HEAVY: GPU stages on if toggles on)")
print("=" * 80)
v0 = val_df.iloc[0]
res = predict(str(v0["query"]), query_id=v0["query_id"], return_intermediates=False)
print(f"query_id : {res['diagnostics']['query_id']}")
print(f"timings  : {res['diagnostics']['timings']}")
print(f"|preds|  : {len(res['predictions'])}")
print(f"first 8  : {res['predictions'][:8]}")
'''

CELL_21_CODE = r'''# ============================================================================
# CELL 21 — Validation run on val.csv (all toggles ON, the headline number)
# ============================================================================
import time, csv, json
from pathlib import Path

def parse_gold_str(s) -> set:
    if not isinstance(s, str): return set()
    return {re.sub(r"\s+", " ", c.strip()) for c in s.split(";") if c.strip()}

def per_query_f1(pred: list, gold: set) -> tuple:
    pset = {re.sub(r"\s+"," ",p.strip()) for p in pred if p.strip()}
    if not gold and not pset: return 1.0, 1.0, 1.0
    if not gold: return 0.0, 1.0, 0.0
    if not pset: return 1.0, 0.0, 0.0
    tp = len(pset & gold)
    p = tp / len(pset); r = tp / len(gold)
    f = (2*p*r/(p+r)) if (p+r) > 0 else 0.0
    return p, r, f

def run_pipeline_on_split(df, split_label: str = "val") -> list:
    rows = []
    for _, q in df.iterrows():
        t0 = time.time()
        out = predict(str(q["query"]), query_id=str(q["query_id"]),
                      return_intermediates=False)
        elapsed = time.time() - t0
        # Apply granularity to gold for fair comparison only when use_granularity is on.
        gold_raw = parse_gold_str(q.get("gold_citations", ""))
        if CONFIG["use_granularity_resolver"]:
            gold_expanded = set(apply_granularity_to_predictions(list(gold_raw)))
        else:
            gold_expanded = gold_raw
        p, r, f = per_query_f1(out["predictions"], gold_expanded)
        rows.append({
            "query_id": q["query_id"],
            "n_pred": len(out["predictions"]),
            "n_gold": len(gold_expanded),
            "P": p, "R": r, "F1": f,
            "elapsed_s": elapsed,
            "diagnostics": out["diagnostics"],
        })
    return rows

print("Running full pipeline on val.csv (all toggles per CONFIG)...")
t_run = time.time()
val_rows = run_pipeline_on_split(val_df, "val")
print(f"\\nDone in {time.time()-t_run:.1f}s")

# Per-query table.
print(f"\\n{'qid':<10} {'P':>6} {'R':>6} {'F1':>6} {'|P|':>5} {'|G|':>5} {'sec':>6}")
print("-" * 56)
for r in val_rows:
    print(f"{r['query_id']:<10} {r['P']:6.3f} {r['R']:6.3f} {r['F1']:6.3f} "
          f"{r['n_pred']:5d} {r['n_gold']:5d} {r['elapsed_s']:6.1f}")

macro_p  = sum(r["P"]  for r in val_rows) / max(1, len(val_rows))
macro_r  = sum(r["R"]  for r in val_rows) / max(1, len(val_rows))
macro_f1 = sum(r["F1"] for r in val_rows) / max(1, len(val_rows))
print("-" * 56)
print(f"{'MACRO':<10} {macro_p:6.3f} {macro_r:6.3f} {macro_f1:6.3f}")
print(f"\\n>>> HEADLINE val Macro F1 (all toggles ON) = {macro_f1:.4f}")

# Save per-query CSV.
out_dir = Path(CONFIG["eval_out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "endgame_val_per_query.csv"
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["query_id","n_pred","n_gold","P","R","F1","elapsed_s"])
    for r in val_rows:
        w.writerow([r["query_id"], r["n_pred"], r["n_gold"],
                    f"{r['P']:.4f}", f"{r['R']:.4f}", f"{r['F1']:.4f}",
                    f"{r['elapsed_s']:.2f}"])
print(f"Saved per-query CSV -> {out_csv}")

VAL_BASELINE_F1 = macro_f1
'''

CELL_22_CODE = r'''# ============================================================================
# CELL 22 — Ablation table (the deliverable)
# ============================================================================
# Sweep: flip each toggle OFF (one at a time, everything else ON), measure
# val Macro F1 delta. Save artifacts/eval/endgame_ablation_val.csv.

import csv, copy, time

ABLATION_TOGGLES = [
    "use_bm25",
    "use_bm25_lexicon_expansion",
    "use_vector",
    "use_hyde",
    "use_query_german_expansion",
    "use_citation_graph_expansion",
    "use_legal_area_softfilter",
    "use_statute_anchors",
    "use_case_anchors",
    "use_court_base_sibling_expansion",
    "use_authority_score_boost",
    "use_enrichment_prefilter",
    "use_granularity_resolver",
    "use_reranker",
    "use_llm_judge",
    "use_dynamic_k_calibration",
]

CONFIG_BACKUP = copy.deepcopy(CONFIG)

def _eval_macro_f1():
    rows = run_pipeline_on_split(val_df, "val")
    p = sum(r["P"]  for r in rows)/max(1,len(rows))
    r_ = sum(r["R"] for r in rows)/max(1,len(rows))
    f = sum(r["F1"] for r in rows)/max(1,len(rows))
    avg_k = sum(r["n_pred"] for r in rows)/max(1,len(rows))
    return p, r_, f, avg_k, rows

baseline_p, baseline_r, baseline_f1, baseline_k, _ = _eval_macro_f1()
print(f"Baseline (all toggles ON): F1={baseline_f1:.4f}  P={baseline_p:.3f}  "
      f"R={baseline_r:.3f}  avg_k={baseline_k:.1f}")

ablation_rows = [{
    "toggle_off": "(baseline)", "F1": baseline_f1, "delta": 0.0,
    "P": baseline_p, "R": baseline_r, "avg_k": baseline_k,
}]

for t in ABLATION_TOGGLES:
    if t not in CONFIG:
        print(f"  skip {t} — not in CONFIG"); continue
    print(f"\n--- ablation: OFF {t} ---")
    CONFIG.update(CONFIG_BACKUP)
    CONFIG[t] = False
    t0 = time.time()
    p, r_, f, ak, _ = _eval_macro_f1()
    print(f"   F1={f:.4f}  delta={f - baseline_f1:+.4f}  ({time.time()-t0:.1f}s)")
    ablation_rows.append({
        "toggle_off": t, "F1": f, "delta": f - baseline_f1,
        "P": p, "R": r_, "avg_k": ak,
    })

# Restore baseline config.
CONFIG.update(CONFIG_BACKUP)

# Sort by delta (most-helpful toggles at the top, biggest hurts at bottom).
ablation_rows.sort(key=lambda x: x["delta"])

# Print table.
print(f"\n\n{'toggle_off':<40} {'F1':>7} {'delta':>8} {'P':>6} {'R':>6} {'avg_k':>6}")
print("-" * 80)
for r in ablation_rows:
    print(f"{r['toggle_off']:<40} {r['F1']:>7.4f} {r['delta']:>+8.4f} "
          f"{r['P']:>6.3f} {r['R']:>6.3f} {r['avg_k']:>6.1f}")

# Save
out_csv = Path(CONFIG["eval_out_dir"]) / "endgame_ablation_val.csv"
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["toggle_off","F1","delta","P","R","avg_k"])
    for r in ablation_rows:
        w.writerow([r["toggle_off"], f"{r['F1']:.4f}", f"{r['delta']:.4f}",
                    f"{r['P']:.3f}", f"{r['R']:.3f}", f"{r['avg_k']:.1f}"])
print(f"\nSaved -> {out_csv}")
'''

CELL_23_CODE = r'''# ============================================================================
# CELL 23 — Channel recall@K@pool diagnostic
# ============================================================================
# For each retrieval channel ALONE, measure R@K on val.csv. Echoes the
# Observation 3 ceiling: dense alone caps at R@1000 = 0.289.

CHANNEL_FUNCS = {
    "bm25":     lambda q, e: bm25_search(q, k=1000),
    "vector":   lambda q, e: vector_search(e, k=1000) if (CONFIG["use_vector"] and e is not None) else [],
    "statute":  lambda q, e: statute_anchor_search(parse_query_anchors(q)["statutes"], k=1000),
    "case":     lambda q, e: case_anchor_search(parse_query_anchors(q)["cases"], parse_query_anchors(q)["dockets"], k=1000),
}

CUTOFFS = (50, 100, 200, 500, 1000)

def channel_recall_at_K(df) -> dict:
    out = {ch: {k: [] for k in CUTOFFS} for ch in CHANNEL_FUNCS}
    for _, q in df.iterrows():
        gold = parse_gold_str(q.get("gold_citations","") or "")
        if CONFIG["use_granularity_resolver"]:
            gold = set(apply_granularity_to_predictions(list(gold)))
        if not gold: continue
        emb = encode_query(str(q["query"])) if CONFIG["use_vector"] else None
        for ch, fn in CHANNEL_FUNCS.items():
            try: results = fn(str(q["query"]), emb)
            except Exception: results = []
            cits = [c.get("citation") for c in results if c.get("citation")]
            for k in CUTOFFS:
                hit = sum(1 for g in gold if g in set(cits[:k])) / max(1, len(gold))
                out[ch][k].append(hit)
    summary = {}
    for ch in CHANNEL_FUNCS:
        summary[ch] = {k: sum(out[ch][k]) / max(1, len(out[ch][k])) for k in CUTOFFS}
    return summary

print("Channel-alone recall@K on val (gold expanded to corpus granularity):")
ch_summary = channel_recall_at_K(val_df)
print(f"\n{'channel':<10}" + "".join(f"  R@{k:>4}" for k in CUTOFFS))
print("-" * 56)
for ch, scores in ch_summary.items():
    line = f"{ch:<10}" + "".join(f"  {scores[k]:.3f}" for k in CUTOFFS)
    print(line)

# Save.
import csv as _csv
out_csv = Path(CONFIG["eval_out_dir"]) / "endgame_channel_recall.csv"
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w = _csv.writer(f)
    w.writerow(["channel"] + [f"R@{k}" for k in CUTOFFS])
    for ch, scores in ch_summary.items():
        w.writerow([ch] + [f"{scores[k]:.4f}" for k in CUTOFFS])
print(f"\nSaved -> {out_csv}")
'''

CELL_24_CODE = '''# ============================================================================
# CELL 24 — Test-set submission (with WINNING config from Cell 22)
# ============================================================================
# Run predict() over test.csv, write submission.csv to Drive.

import csv, time

print("Running pipeline on test.csv with current CONFIG (winning ablation choice)...")
t0 = time.time()
test_rows = []
for _, q in test_df.iterrows():
    out = predict(str(q["query"]), query_id=str(q["query_id"]),
                  return_intermediates=False)
    test_rows.append({
        "query_id": q["query_id"],
        "predictions": ";".join(out["predictions"]),
        "n_pred": len(out["predictions"]),
    })
print(f"Done in {time.time()-t0:.1f}s; {len(test_rows)} test queries.")

submissions_dir = Path(CONFIG["submissions_dir"])
submissions_dir.mkdir(parents=True, exist_ok=True)
sub_path = submissions_dir / "submission_endgame.csv"
with sub_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["query_id", "gold_citations"])
    for r in test_rows:
        w.writerow([r["query_id"], r["predictions"]])
print(f"\\nWrote test submission -> {sub_path}")
print(f"Median |pred| per test query: "
      f"{sorted([r['n_pred'] for r in test_rows])[len(test_rows)//2]}")
'''

CELL_25_CODE = '''# ============================================================================
# CELL 25 — Failure analysis: worst-3 val queries
# ============================================================================
import pprint

worst3 = sorted(val_rows, key=lambda r: r["F1"])[:3]
for r in worst3:
    qid = r["query_id"]
    q_row = val_df[val_df["query_id"] == qid].iloc[0]
    gold = parse_gold_str(q_row.get("gold_citations","") or "")
    if CONFIG["use_granularity_resolver"]:
        gold = set(apply_granularity_to_predictions(list(gold)))
    out = predict(str(q_row["query"]), query_id=qid, return_intermediates=True)
    pred = set(out["predictions"])
    fp = pred - gold
    fn = gold - pred
    print("=" * 80)
    print(f"FAIL  qid={qid}  F1={r['F1']:.3f}  P={r['P']:.3f}  R={r['R']:.3f}")
    print(f"      |pred|={len(pred)}  |gold|={len(gold)}  "
          f"|FP|={len(fp)}  |FN|={len(fn)}")
    print(f"\\nQuery (truncated):\\n  {str(q_row['query'])[:240]}")
    print(f"\\nDiagnostics:")
    pprint.pprint(out["diagnostics"]["timings"])
    pprint.pprint(out["diagnostics"].get("zone_counts", {}))
    pprint.pprint(out["diagnostics"].get("channel_counts", {}))
    print(f"\\nFalse negatives (gold not in pred), first 12:")
    for c in sorted(fn)[:12]:
        # Track: was it ever in retrieval? rerank? judged 'no'?
        seen_retrieve = any(x.get("citation") == c for x in out.get("candidates_after_retrieve") or [])
        seen_judged = next((x for x in (out.get("candidates_after_judge") or [])
                            if x.get("citation") == c), None)
        if seen_judged is not None:
            tag = f"in-judge verdict={seen_judged.get('verdict')}"
        elif seen_retrieve:
            tag = "in-retrieve, dropped before judge"
        else:
            tag = "NEVER retrieved"
        print(f"  - {c:<40s} ({tag})")
    print(f"\\nFalse positives (pred not in gold), first 8:")
    for c in sorted(fp)[:8]:
        print(f"  + {c}")
'''

CELL_26_MD = """## Cell 26 — Final summary

After running the ablation in Cell 22, fill in this template by hand
(or auto-generate from the rows in `endgame_ablation_val.csv`).

**Headline numbers (all toggles ON, val Macro F1):** see Cell 21 print-out
(`VAL_BASELINE_F1`).

**Reference baselines:**
- `research/Untitled75.ipynb` v12 (BM25 + Reranker + Judge): **0.777**
- Dense-alone (Qwen3-Embedding-8B): **0.041** (Observation 3)

**Toggles that helped (positive delta when OFF means: keeping it ON helps):**
The Cell-22 ablation table is sorted with the biggest *positive* delta-at-top
(turning the toggle off helps least and may hurt). Those rows are the
load-bearing components.

**Toggles that hurt (negative delta when OFF means: turning it OFF was BETTER):**
Bottom of the Cell-22 table. Strong candidates for permanent removal in the
final config.

**Recommended final config:** keep every toggle whose ablation delta is at
least `-0.005` (or 0). Disable any toggle whose ablation delta is positive
(the pipeline is worse with that channel on, and removing it both speeds
things up and improves F1).

**Residual gaps (from `endgame_val_per_query.csv` and Cell 25):**
- Per-query F1 floor (worst 3): see Cell 25 dump.
- Channel-alone ceiling (Cell 23): expect dense ~ 0.29 R@1000;
  BM25 + lexicon expansion typically ~0.85 R@100 on val.
- Granularity expansion: small but consistent uplift; mandatory unless
  ablation says otherwise.

**What we have NOT solved:**
- Class B (text-reference-only) gold: ~21% of train gold, 0% of val (val is
  100% retrievable). On the hidden test set we expect a similar ~0% — but if
  the host added *new* held-out queries, this could become a hard ceiling.
- Court considerations (val 40.6% of gold): if the court_base sibling
  expansion ablation shows a large negative delta when off, this is the
  channel that recovers them.

Update this cell after running the full ablation.
"""


# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------

cells = [
    md(CELL_0_MD),
    code(CELL_1_CODE),
    code(CELL_2_CODE),
    code(CELL_3_CODE),
    code(CELL_4_CODE),
    code(CELL_5_CODE),
    code(CELL_6_CODE),
    code(CELL_7_CODE),
    code(CELL_8_CODE),
    code(CELL_9_CODE),
    code(CELL_10_CODE),
    code(CELL_11_CODE),
    code(CELL_12_CODE),
    code(CELL_13_CODE),
    code(CELL_14_CODE),
    code(CELL_15_CODE),
    code(CELL_16_CODE),
    code(CELL_17_CODE),
    code(CELL_18_CODE),
    code(CELL_19_CODE),
    code(CELL_20_CODE),
    code(CELL_21_CODE),
    code(CELL_22_CODE),
    code(CELL_23_CODE),
    code(CELL_24_CODE),
    code(CELL_25_CODE),
    md(CELL_26_MD),
]

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
            "mimetype": "text/x-python",
            "file_extension": ".py",
        },
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Wrote: {OUT_PATH}  ({OUT_PATH.stat().st_size/1024:.1f} KB)")
print(f"Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
