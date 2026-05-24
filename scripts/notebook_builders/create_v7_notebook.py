#!/usr/bin/env python
"""Build v7 notebook from scratch — production-grade with rigorous diagnostics.

Design principles:
  1. Every code cell preceded by a markdown cell stating WHAT it does, WHY we
     need it, what to EXPECT in outputs, and what FAILURE MODES to watch for.
  2. v6's proven retrieval logic kept (channels, RRF, gating).
  3. v7 graph wiring added (intra-judgment backrefs + date aliases + range
     expansion + case-level fan-out; 23.65M edges).
  4. PHASE 10 — per-gold diagnosis: for every gold not in top-1000, trace
     through every channel and every signal source to identify the exact
     drop point. No guessing — every diagnostic prints concrete numbers
     (rank, score, token overlap, cosine similarity, graph in/out degree).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

V7 = Path(r"C:\Users\samiul\Downloads\swiss_citation_anchor_funnel_val001_v7.ipynb")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_lines(text: str) -> list[str]:
    text = text.lstrip("\n")
    lines = text.splitlines(keepends=True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    return lines


def add_md(cells: list, text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": to_lines(text)})


def add_code(cells: list, text: str) -> None:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_lines(text),
    })


# ---------------------------------------------------------------------------
# Cell builders — each phase grouped together
# ---------------------------------------------------------------------------

def phase_intro(cells: list) -> None:
    add_md(cells, """
# Swiss Citation — Anchor-Funnel v7 (production diagnostic build)

**Pass criterion: R@1000 ≥ 0.60** for val_001 (≥ 26/42 gold). Stretch: ≥ 0.90 / 38+.

This notebook is structured as **11 phases** with a markdown header before every code cell.
Each markdown header states:
- **What** the next cell computes
- **Why** we need that signal (which retrieval gap it closes)
- **Expected outputs** — numeric ranges so anomalies pop out
- **Failure modes** — what to check if a number looks off

The final phase (Phase 10) does **per-gold diagnosis**: for every gold citation
NOT captured in top-1000, the notebook traces through every channel and every
signal source and prints a concrete root-cause report (with rank, score, token
overlap, cosine similarity, graph degrees).

## Phase map

| Phase | Cells | Purpose |
|---|---|---|
| 1. Setup | env, drive, paths, knobs | runtime + IO + global config |
| 2. Load val + corpus indexes | val.csv, law-llm jsonl, court-v5 jsonl | build all in-memory indexes |
| 3. Citation graph | sqlite → idx_graph_out / idx_graph_in | 4-layer graph (23.65M edges) |
| 4. Per-area bedrock + co-citation | corpus statistics | universal articles per legal area |
| 5. BM25 (FTS5 in-memory) | query-side lexical match | catches code names + Swiss terminology |
| 6. **Query expansion (Qwen3-32B)** | load → run → **FREE** | structured JSON targets |
| 7. **Encode queries (Qwen3-Embedding-8B)** | load → encode → **FREE** | q_emb_raw + q_emb_enriched |
| 8. **Vector setup (E_GPU)** | load 27 fp16 chunks to GPU | dense semantic search |
| 9. Run channels | 14 retrieval signals | each with per-channel R@K |
| 10. RRF fusion + gating | reciprocal rank fusion + neg-gate | top-1000 final pool |
| 11. **Diagnosis** | per-gold trace + miss attribution | which signal failed for which gold |
| 12. Save artifacts + cleanup | persist to Drive | reproducibility |

## Memory plan — only ONE model in VRAM at a time

The Blackwell GPU has 95.6 GB. We enforce a strict order so peak VRAM never
exceeds ~65 GB:

| When | What's resident | Peak VRAM |
|---|---|---:|
| End of Phase 5 | nothing on GPU | 0 GB |
| **Phase 6** | Qwen3-32B (bf16) | **~65 GB** |
| End of Phase 6 | nothing (freed) | 0 GB |
| **Phase 7** | Qwen3-Embedding-8B | **~16 GB** |
| End of Phase 7 | nothing (freed); q_emb_raw + q_emb_enriched cached on CPU | 0 GB |
| **Phase 8 onwards** | E_GPU (corpus embeddings) | **~22 GB** |

Each model-loading phase ends with: `del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()` and prints VRAM-after-free as a sanity check.

## Constraints respected (recall from prior chat)

- **No hardcoded lists.** No DE_LEX, no fixed BGE list, no statute cluster table.
- **No query-specific knowledge.** Architecture must generalize to any val/test/production query.
- **No train data.** Train.csv is never read.
- **Open-source only.** Qwen3-32B (query expansion), Qwen3-Embedding-8B (vectors).

## v7 changes vs v6 (R@1000 was 0.357)

| Change | Why |
|---|---|
| Sibling budget 500 → 2000 | v6's 500 cap was a non-deterministic `set→list[:budget]` slice; with 2691 typical seeds × ~5 siblings, the slice dropped val_001 sibling Es. |
| Graph forward channel (NEW) | Loads corpus-derived citation graph (4 alias passes, 23.65M edges) and follows outgoing edges — surfaces text-cited targets PLUS all sibling Es of cited judgments via case-level fan-out. Closes 6/11 originally-orphan val_001 gold. |
| Graph reverse channel (NEW) | Follows incoming edges — finds rows that text-cite seeds. Co-citation expansion via the actual corpus graph, not LLM. |
| Phase 10 diagnostics | Every missed gold gets a per-channel trace + recommended fix. |
""")


def phase_1_setup(cells: list) -> None:
    add_md(cells, """
# Phase 1 — Setup

## 1.1 Environment & GPU check

**What:** Print Python version, PyTorch version, GPU name + VRAM.

**Why:** All our heavy work (Qwen3-32B, Qwen3-Embedding-8B, full-corpus dense
matmul) requires a single high-VRAM GPU. The Blackwell instance has 95.6 GB —
Qwen3-32B in bf16 (~65 GB) + Qwen3-Embedding-8B + corpus `E_GPU` (~37 GB) just
fit if loaded sequentially.

**Expected:** Python 3.12+, PyTorch 2.x with CUDA, 1× GPU with ≥ 80 GB.

**Failure modes:**
- "0 GPUs" → Colab session lost GPU; restart runtime.
- VRAM < 80 GB → Qwen3-32B will OOM. Drop to `qwen_query_model = "Qwen/Qwen3-8B"`
  in CONFIG and accept thinner JSON targets.
""")
    add_code(cells, """
import sys, torch
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}, {p.total_memory / 1024**3:.1f} GB")
else:
    print("[WARN] No CUDA GPU available — vector channels will be skipped.")
""")

    add_md(cells, """
## 1.2 Mount Google Drive

**What:** Mount Drive so we can read the corpus, embeddings, and citation
graph from `/content/drive/MyDrive/swiss_law/`.

**Why:** The 21 GB of fp16 embeddings, 24 GB unified retrieval SQLite, and
2.4 GB citation graph all live on Drive — too large to download per run.

**Expected:** "Mounted at /content/drive". Skip silently if running locally.

**Failure modes:**
- "Drive not authorized" → click the OAuth link Colab prints.
- Mount succeeds but `MyDrive/swiss_law` is empty → wrong account; remount.
""")
    add_code(cells, """
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as e:
    print(f"[skip] Not on Colab or drive already mounted: {e}")
""")

    add_md(cells, """
## 1.3 Resolve all data paths

**What:** Auto-detect `DATA_ROOT` (could be Drive, local, or Kaggle) and
build a `PATHS` dict pointing at every required artifact.

**Why:** Same notebook should run on Colab, Kaggle, or locally without code
changes. Each artifact is checked for existence with a clear OK/MISSING flag.

**Required inputs:**
- `data/val.csv` — 10 English queries with gold citations.
- `law_llm_descriptors_*.jsonl` — LLM enrichment of all 175k laws.
- `court_authority_cards_v5_unified.jsonl` — court enrichment (10.4 GB).
- `embeddings/qwen3_8b_unified_chunk*.npy` — 27 fp16 chunks (21 GB total).
- `embeddings/qwen3_8b_unified_manifest.parquet` — doc_id ↔ row_index.
- **v7 NEW:** `data_insights/citation_graph_extracted.sqlite` — 4-layer graph (~2.4 GB).

**Expected:** All flags `OK`. If `graph_db` is `MISSING`, graph channels
silently skip (recall drops back to v6 levels).

**Failure modes:**
- `MISSING law_llm` or `court_v5` → notebook can't build any index. Stop.
- `MISSING emb_dir` → vector channels skipped; BM25 + anchors still run.
- `MISSING graph_db` → graph channels skipped; sibling_expansion (court_base) only.
""")
    add_code(cells, """
from pathlib import Path

CANDIDATE_ROOTS = [
    Path("/content/drive/MyDrive/swiss_law"),
    Path("/content/drive/MyDrive/swiss_citation_extraction"),
    Path("/content/swiss_citation_extraction"),
    Path(r"E:/swiss_citation_extraction"),
    Path.cwd(),
]

DATA_ROOT = None
for root in CANDIDATE_ROOTS:
    if (root / "data" / "val.csv").exists():
        DATA_ROOT = root; break
if DATA_ROOT is None:
    print("[warn] Could not auto-detect DATA_ROOT — defaulting to /content/drive/MyDrive/swiss_law")
    DATA_ROOT = Path("/content/drive/MyDrive/swiss_law")
print(f"DATA_ROOT = {DATA_ROOT}")

def first_existing(*paths):
    for p in paths:
        if p.exists(): return p
    return paths[0]

PATHS = {
    "val_csv": DATA_ROOT / "data" / "val.csv",
    "law_llm": first_existing(
        DATA_ROOT / "data" / "checkpoints" / "law_llm_descriptors_0000000_all.jsonl",
        DATA_ROOT / "law_json_llm_output" / "law_llm_descriptors_0000000_all.jsonl",
    ),
    "court_v5": first_existing(
        DATA_ROOT / "artifacts_v2" / "court_authority_cards_v5_unified.jsonl",
        DATA_ROOT / "artifacts" / "court_authority_cards_v5_unified.jsonl",
    ),
    "emb_dir":      DATA_ROOT / "artifacts" / "embeddings",
    "emb_manifest": DATA_ROOT / "artifacts" / "embeddings" / "qwen3_8b_unified_manifest.parquet",
    "graph_db": first_existing(
        DATA_ROOT / "data_insights" / "citation_graph_extracted.sqlite",
        DATA_ROOT / "citation_graph_extracted.sqlite",
    ),
    "out_dir":      DATA_ROOT / "research" / "anchor_funnel_val001_v7",
}
PATHS["out_dir"].mkdir(parents=True, exist_ok=True)

for k, p in PATHS.items():
    if k == "out_dir": continue
    flag = "OK     " if p.exists() else "MISSING"
    print(f"  {flag}  {k:<14} {p}")

EMB_AVAILABLE   = PATHS["emb_dir"].exists() and any(PATHS["emb_dir"].glob("qwen3_8b_unified_chunk*.npy"))
GRAPH_AVAILABLE = PATHS["graph_db"].exists()
print()
print(f"  EMB_AVAILABLE   = {EMB_AVAILABLE}")
print(f"  GRAPH_AVAILABLE = {GRAPH_AVAILABLE}")
""")

    add_md(cells, """
## 1.4 Knob panel (CONFIG)

**What:** All architecture knobs live in a single `CONFIG` dict — budgets,
RRF k, BM25 limits, vector top-k, query-expansion model, etc.

**Why:** Editing budgets without touching channel code is essential when
diagnosis suggests "channel X dropped this gold due to truncation, lift
budget". Every budget is rationalized in a comment.

**Sensitive knobs:**
- `budget_sibling = 2000` — was 500 in v6, dropped sibling Es non-deterministically.
- `budget_graph_forward = 1500` — graph 1-hop forward; case-level fan-out adds
  up to ~30 expansions per seed, 1500 covers 50 seeds × 30 siblings.
- `budget_graph_reverse = 1000` — co-citing rows; bounded to keep ranking signal.
- `enable_graph_2hop = False` — 2-hop tends to dump procedural articles already
  caught elsewhere; flip to True only after Phase 10 says so.

**Expected:** dict prints cleanly; nothing should be `None` except `budget_law_direct`.
""")
    add_code(cells, """
import json

CONFIG = {
    "topk_final": 1000,

    # --- Channel budgets ---------------------------------------------------
    # law_direct_match: every law row whose canonical citation matches a
    # (LLM-named OR co-cited) statute target. Tiny per canon; uncapped is safe.
    "budget_law_direct":     None,

    # court_statute: court rows annotated with one of the LLM-named statutes.
    # v7.1 LIFT: was 600. With ~5 LLM statute canonicals × ~2000 court rows
    # per canonical, the 600 cap dropped val gold (which often score only 1-2
    # on the Counter, tied with thousands of others). 3000 keeps more of the
    # tied "score=1" tail.
    "budget_court_statute":  3000,

    # concept_en: rows whose concepts_en token overlaps with LLM concept_targets.
    "budget_concept":        600,

    # term_orig: rows whose terms_original (DE/FR/IT) overlap with LLM term_targets.
    "budget_term":           500,

    # per_area_bedrock: most-cited canonical statutes within the LLM-named legal_area.
    # Per-area_top_n is internal cap; this budget caps the bedrock channel output.
    "budget_per_area":       150,

    # co_citation: for each LLM statute target, fetch top-K co-citation neighbours
    # and pull their law + court rows.
    # v7.1 LIFT: was 400. Same reasoning as court_statute — ties at score=1
    # were dropping gold rows that are real co-cited neighbours.
    "budget_co_citation":    1000,

    # bm25: lexical match on enriched query text. enhance() adds top-K corpus-
    # associated codes to query.
    "budget_bm25":           600,

    # vector_raw / vector_enriched: dense semantic match using Qwen3-Embedding-8B
    # against full-corpus E_GPU.
    "budget_vector":         800,
    "budget_vector_enriched":800,

    # statute_backprop: each caught court row contributes its cited statutes;
    # score = number of distinct caught court rows citing that article.
    # Surfaces procedural cluster (Art. 100 BGG, Art. 422 StPO, etc.).
    "budget_backprop":       400,

    # sibling_expansion (court_base): caught court row → all Es of same judgment.
    # v6 had 500 → non-deterministic slice dropped val_001 sibling Es. 2000 fits
    # 100 seeds × 20 Es each.
    "budget_sibling":        2000,

    # graph_forward / graph_reverse / graph_2hop: 1-hop and 2-hop traversal of
    # the citation graph (intra-judgment backrefs + date aliases + range +
    # case-level fan-out). budget_forward sized for case-level fan-out from
    # ~50 seeds × 30 sibling-fanout ≈ 1500.
    "budget_graph_forward":  1500,
    "budget_graph_reverse":  1000,
    # v7.3 DISABLE: 2-hop crashed in v7.2 from 0.046 → 0.003 mean recall after
    # the seed-extension change. The sibling-extended seed makes 2-hop's
    # combinatorial expansion blow up; budget=1500 truncation becomes pure
    # noise. Re-enable only after a smarter 2-hop scoring scheme.
    "enable_graph_2hop":     False,
    "budget_graph_2hop":     1500,

    # --- RRF + guarantee ---------------------------------------------------
    "rrf_k": 60,
    # v7.1: guarantee channels merged ROUND-ROBIN (interleaved) instead of
    # concatenated, so each gets fair share even when total > topk_final.
    # Promoted graph_forward + sibling_expansion to guarantees — they had
    # gold at low ranks that RRF was killing.
    "guarantee_channels": [
        "law_direct_match",
        "per_area_bedrock",
        "statute_backprop",
        "graph_forward",       # v7.1 NEW: best topical channel (10/42 in val_001)
        "sibling_expansion",   # v7.1 NEW: own-judgment Es; also re-ranked by seed-freq
    ],
    # Each guarantee channel contributes at most this many slots before
    # RRF tail kicks in.
    # v7.3 LIFT: was 200. With 5 guarantees but 2 of them (LDM, PAB) often
    # smaller than 200, the round-robin saturates at total_cap=1000 with
    # statute_backprop / graph_forward / sibling each capped at 200. v7.2 lost
    # gold deeper in graph_forward (e.g., BGE 137 IV 122 E. 6.2 at fused rank
    # 11758 → its in-channel rank was beyond 200). Lifting the cap to 400 lets
    # the high-recall channels contribute up to ~253 items each (LDM + PAB
    # exhausted earlier free slots for the others).
    "guarantee_per_channel": 400,

    # v7.3 NEW: per-channel RRF weight. Reweight RRF contributions by channel
    # mean recall observed in v7.2. statute_backprop (0.453) and graph_forward
    # (0.324) carry the most gold-per-budget; reweight them up so deep-rank
    # gold survives RRF accumulation against shallow-rank non-gold from
    # weaker channels.
    "channel_weights": {
        "statute_backprop":  2.5,   # v7.2 mean recall 0.453 (universal best)
        "graph_forward":     2.0,   # v7.2 mean recall 0.324 (consistent #2)
        # Default 1.0 for mid-tier (vector_raw/enriched, concept_en,
        # per_area_bedrock, law_direct_match) — recall 0.10–0.15
        "court_statute":     0.7,
        "bm25":              0.7,
        "term_orig":         0.7,
        "sibling_expansion": 0.7,
        "graph_reverse":     0.5,
        "co_citation":       0.3,
        "graph_2hop":        0.0,   # disabled in v7.3
    },

    # --- Per-area bedrock --------------------------------------------------
    "per_area_top_n": 100,

    # --- Co-citation -------------------------------------------------------
    "co_citation_top_k_per_target":    8,
    "co_citation_min_co_count":        50,
    # drop neighbours that are TOO globally common (Art. 36 BV cited everywhere
    # would pollute court_statute). 5000 = ~0.2% of 2.5M corpus.
    "co_citation_max_neighbour_count": 5000,

    # --- Concept matching --------------------------------------------------
    "concept_substring_top_k": 6,

    # --- BM25 --------------------------------------------------------------
    "bm25_max_query_terms": 60,    # cap to stop token-explosion from enriched query
    "bm25_min_token_len":   3,

    # --- Vector ------------------------------------------------------------
    "vector_emb_model": "Qwen/Qwen3-Embedding-8B",
    "vector_topk":      800,

    # --- Query expansion ---------------------------------------------------
    "qwen_query_model":    "Qwen/Qwen3-32B",
    "qwen_max_new_tokens": 1024,

    # --- enhance() — corpus-derived BM25 lexicon expansion -----------------
    "enhance_top_k_codes":   5,
    "enhance_repeat_count":  5,
    "enhance_min_idf":       1.0,

    # --- Negative gate -----------------------------------------------------
    "noise_paragraph_roles": {"notification", "header", "empty", "metadata"},

    "lowercase_concepts": True,
    "lowercase_terms":    True,
}

print(json.dumps({k: v for k, v in CONFIG.items() if not isinstance(v, set)}, indent=2, default=str))
""")

    add_md(cells, """
## 1.5 Load all val queries + gold

**What:** Read `val.csv`, build a per-query dict so Phases 6-9 can loop
over all 10 queries (not just val_001).

**Why generalization matters:** v5/v6/v7 were all single-query optimizations
on val_001. We don't yet know if the patterns generalize. Running on the
full val set (10 queries) tells us:
- Which queries hit > 0.6 R@1000 (target).
- Which channels carry their weight on average (vs. only on val_001).
- Whether sibling_expansion / graph_forward are universally useful or
  idiosyncratic to val_001's structure (multi-E BGer detention case).

**Expected:** 10 queries, gold counts 10–47 per query.

**Failure modes:**
- "0 gold for query X" → semicolon parser issue or the column name drifted.
- val.csv missing → check Phase 1.3 paths.
""")
    add_code(cells, """
import pandas as pd

val_df = pd.read_csv(PATHS["val_csv"])
print(f"val.csv has {len(val_df)} queries\\n")

# Per-query gold lists, parsed once and used by every downstream phase.
ALL_QUERIES = []
for _, vrow in val_df.iterrows():
    qid = str(vrow["query_id"])
    qtext = str(vrow["query"])
    gold = [c.strip() for c in str(vrow["gold_citations"]).split(";") if c.strip()]
    ALL_QUERIES.append({"query_id": qid, "query": qtext, "gold": gold})
    print(f"  {qid}: {len(gold)} gold | {qtext[:120]}{'...' if len(qtext)>120 else ''}")

# val_001 sanity-print kept (it remains the diagnostic anchor in Phase 11).
val001 = next(q for q in ALL_QUERIES if q["query_id"] == "val_001")
QUERY = val001["query"]            # legacy var used by Phase 11 detailed diagnosis
val_gold = val001["gold"]           # legacy var
print(f"\\nval_001 (canary for detailed diagnosis): {len(val_gold)} gold")
""")


def phase_2_index_build(cells: list) -> None:
    add_md(cells, """
# Phase 2 — Build all corpus indexes (one pass)

## 2.1 What this big cell does

This cell is the heart of the retrieval pipeline. It **streams both JSONL files**
(`law_llm_descriptors` + `court_authority_cards_v5`) and builds **all** in-memory
indexes the channels need:

| Index | Type | What it maps |
|---|---|---|
| `cit_to_doc_ids[citation]` | dict[str, list[str]] | citation string → list of doc_ids |
| `doc_meta[did]` | dict[str, dict] | doc_id → {citation, family, court_base, paragraph_role} |
| `idx_law_direct[canonical]` | dict[str, set] | "100 BGG" → law doc_ids |
| `idx_court_statute[canonical]` | dict[str, set] | "100 BGG" → court rows whose anchors include this |
| `idx_court_base[base]` | dict[str, set] | "137 IV 122" → all Es of judgment (used by sibling_expansion) |
| `idx_concept_en[token]` | dict[str, set] | English concept → doc_ids |
| `idx_term_orig[token]` | dict[str, set] | DE/FR/IT term → doc_ids |
| `search_text[did]` | dict[str, str] | doc_id → BM25-search text (concatenated enrichment) |
| `legal_area_per_doc[did]` | dict[str, str] | court doc → its `legal_area_static` |
| `co_citation_pairs` | Counter[(canon_a, canon_b)] | unordered pairs of statutes co-cited within same court row |
| `tlf[token][code]` | dict[str, Counter] | corpus-wide token → law-code association (for `enhance()`) |
| `doc_statute_anchors[did]` | dict[str, set] | court doc_id → set of canonical statutes it cites |

## 2.2 Why we need each one (channel attribution)

- `idx_law_direct` → channels: `law_direct_match`, `per_area_bedrock`, `statute_backprop`, `co_citation`
- `idx_court_statute` → channels: `court_statute`, `co_citation`
- `idx_court_base` → channel: `sibling_expansion` (own-judgment Es)
- `idx_concept_en` / `idx_term_orig` → channels: `concept_en`, `term_orig`
- `search_text` → channel: `bm25` (FTS5 indexed in Phase 5)
- `co_citation_pairs` → Phase 4 builds `co_neighbours` from this; channel: `co_citation`
- `tlf` → BM25 query enhancement in Phase 5
- `doc_statute_anchors` → channel: `statute_backprop`

## 2.3 Expected outputs
- Law: ~175k rows in ~10 s
- Court: ~2.47M rows in ~150 s
- Token→code association: ~90k tokens, avg ~10 codes/token
- Co-citation pairs: ~1.1M

## 2.4 Failure modes
- **Slower than 200 s for court** → Drive throttling; retry.
- **`Total docs ≠ ~2.65M`** → JSONL truncation; check file size matches local.
- **`tlf` very small (< 50k tokens)** → law tokenizer pattern wrong; check the
  regex split below.
""")
    add_code(cells, """
from collections import defaultdict, Counter
import re, json, time

# --- Statute / case canonicalizers --------------------------------------------
CODE_ALIAS = {
    "CPP": "StPO", "CP": "StGB", "CC": "ZGB", "CO": "OR",
    "LTF": "BGG", "LACI": "AVIG", "LAA": "UVG",
    "LP": "SchKG", "LDIP": "IPRG", "Cst": "BV", "Cst.": "BV",
    "STPO": "StPO", "OBG": "OR",
}
ART_RE = re.compile(r"art\\.?\\s*(\\d+[a-z]?)", re.I)
CODE_RE = re.compile(r"\\b([A-Z][A-Za-z]{1,8}\\.?)\\b")

def statute_anchor_canonical(raw):
    if not raw: return None
    s = raw.strip()
    m = ART_RE.search(s)
    if not m: return None
    cands = [c.strip(".") for c in CODE_RE.findall(s)
             if c.strip(".") not in ("Art","Abs","Ziff","lit","let","al","Bst")]
    if not cands: return None
    code = CODE_ALIAS.get(cands[-1], cands[-1])
    return f"{m.group(1)} {code}"

def article_num(raw):
    if not raw: return None
    m = ART_RE.search(raw.strip())
    return m.group(1) if m else None

LEGAL_AREA_DEFAULT_CODE = {
    "criminal law and criminal procedure": "StPO",
    "criminal procedure":                  "StPO",
    "criminal law":                        "StGB",
    "civil law":                           "ZGB",
    "obligations":                         "OR",
    "civil procedure":                     "ZPO",
    "constitutional and public law":       "BV",
    "constitutional law":                  "BV",
    "administrative law":                  "VwVG",
    "social insurance":                    "ATSG",
    "tax law":                             "DBG",
}

def canonicalize_row_anchors(raw_anchors, legal_area_static):
    canons = set()
    primary_code = None
    for sa in raw_anchors:
        c = statute_anchor_canonical(sa)
        if c:
            primary_code = c.split()[1]; break
    fallback = primary_code
    if fallback is None and legal_area_static:
        la = legal_area_static.lower()
        for k, v in LEGAL_AREA_DEFAULT_CODE.items():
            if k in la:
                fallback = v; break
    for sa in raw_anchors:
        c = statute_anchor_canonical(sa)
        if c:
            canons.add(c); continue
        n = article_num(sa)
        if n and fallback:
            canons.add(f"{n} {fallback}")
    return canons

# v7.2 P2 FIX: French/Italian formatting variants the build-time regex missed.
# Audit (audit_statute_anchor_regex.py) found 233 rows (0.02%) with empty
# statute_anchors but text containing French-style citations like:
#   "article 423 al. 1 CO"   (lowercase 'article', full word, no period)
#   "art 66 al. 4 LTF"       (no period after 'art')
#   "art. 99 al. 2 LTF"      (already caught by build, but safer here)
# We map French/Italian aliases (CO→OR, CP→StGB, CPP→StPO, CPC→ZPO, CC→ZGB,
# LTF→BGG, Cst→BV) into their canonical Swiss German equivalents using the
# existing CODE_ALIAS map.
EXTRA_STATUTE_RE = re.compile(
    r'(?<!\\w)'
    r'(?:Art(?:icle)?\\.?|Artikel|art\\.?|article|articolo|articoli)\\s+'
    r'(\\d+(?:[a-z]+)?)'                                  # article number
    r'(?:\\s+(?:Abs\\.?|Absatz|al\\.?|alin[ée]a|cpv\\.?)\\s*\\d+(?:[a-z]+)?)?'
    r'(?:\\s+(?:lit\\.?|let\\.?|Bst\\.?|Buchstabe)\\s*[a-z])?'
    r'(?:\\s+(?:Ziff\\.?|Ziffer|n\\.|no\\.|num\\.|cifra)\\s*\\d+(?:[a-z]+)?)?'
    r'\\s+([A-Z][A-Za-z]{1,9}\\.?)',                      # code
    re.IGNORECASE,
)

def extract_extra_statute_canons(text):
    \"\"\"v7.2: regex sweep over text for any 'art./article/articolo N CODE'
    citation. Returns canonical 'N CODE' strings (using French/Italian aliases
    mapped to Swiss German via CODE_ALIAS).\"\"\"
    if not text: return set()
    out = set()
    for m in EXTRA_STATUTE_RE.finditer(text):
        n = m.group(1).strip()
        code = m.group(2).strip().rstrip('.')
        if not n or not code: continue
        # Filter out obvious noise (section markers, internal cross-refs)
        if code in ("Abs", "Ziff", "lit", "let", "Bst", "al", "cpv", "n", "no"):
            continue
        # Apply alias (CO→OR, CP→StGB, ...) — same dict the build-time path uses
        code = CODE_ALIAS.get(code, code)
        out.add(f"{n} {code}")
    return out

CASE_BGE_RE    = re.compile(r"BGE\\s+(\\d+)\\s+([IVX]+)\\s+(\\d+)")
CASE_DOCKET_RE = re.compile(r"\\b(\\d[A-Z]_\\d+/\\d{4})\\b")

def case_anchor_canonical(raw):
    if not raw: return None
    s = raw.strip()
    m = CASE_BGE_RE.search(s)
    if m: return f"BGE {m.group(1)} {m.group(2)} {m.group(3)}"
    m = CASE_DOCKET_RE.search(s)
    if m: return m.group(1)
    return None

TOKEN_NORM_RE = re.compile(r"\\s+")
def norm_token(s, lower):
    if not s: return None
    s = TOKEN_NORM_RE.sub(" ", s.strip())
    if not s: return None
    return s.lower() if lower else s

# --- Indexes -----------------------------------------------------------------
cit_to_doc_ids       = defaultdict(list)
doc_meta             = {}
idx_law_direct       = defaultdict(set)
idx_court_statute    = defaultdict(set)
idx_case_anchor      = defaultdict(set)
idx_court_base       = defaultdict(set)
idx_concept_en       = defaultdict(set)
idx_term_orig        = defaultdict(set)
legal_area_per_doc   = {}
search_text          = {}
co_citation_pairs    = Counter()
tlf                  = defaultdict(Counter)
token_doc_count      = Counter()
doc_statute_anchors  = {}

DOC_ID_LAW   = lambda i: f"law:{i}"
DOC_ID_COURT = lambda i: f"court:{i}"

def _take_text(*parts, max_chars=2000):
    out = []
    for p in parts:
        if not p: continue
        if isinstance(p, list):
            for x in p:
                if isinstance(x, str): out.append(x)
                elif isinstance(x, dict):
                    for v in x.values():
                        if isinstance(v, str): out.append(v)
        elif isinstance(p, str):
            out.append(p)
    return (" ".join(out))[:max_chars]

# --- Stream law jsonl --------------------------------------------------------
t0 = time.time(); n_law = 0
with open(PATHS["law_llm"], encoding="utf-8") as f:
    for line in f:
        try: obj = json.loads(line)
        except Exception: continue
        cit = obj.get("citation","")
        if not cit: continue
        did = DOC_ID_LAW(n_law)
        cit_to_doc_ids[cit].append(did)
        doc_meta[did] = {"citation": cit, "family": "law", "court_base": None,
                         "paragraph_role": None, "is_notification_paragraph": False}
        canon = statute_anchor_canonical(cit)
        if canon: idx_law_direct[canon].add(did)

        enr = obj.get("llm_enrichment") or {}
        terms_de = []; terms_en = []
        for t in enr.get("terms_de_to_en") or []:
            if isinstance(t, dict):
                de = norm_token(t.get("de",""), CONFIG["lowercase_terms"])
                en = norm_token(t.get("en",""), CONFIG["lowercase_terms"])
                if de:
                    idx_term_orig[de].add(did); terms_de.append(de)
                if en:
                    idx_concept_en[en].add(did); terms_en.append(en)
        for c in enr.get("concepts_en") or []:
            tok = norm_token(c, CONFIG["lowercase_concepts"])
            if tok: idx_concept_en[tok].add(did)
        search_text[did] = _take_text(
            cit, enr.get("english_summary",""), enr.get("legal_rule",""),
            enr.get("legal_question",""), enr.get("applicability_conditions"),
            enr.get("concepts_en"), terms_de, terms_en,
        )
        legal_area_per_doc[did] = "law"

        # token -> code association (only law rows have a clean canonical code).
        if canon and " " in canon:
            row_code = canon.split()[1].lower()
            row_text = search_text[did].lower()
            row_tokens = set()
            for tok in re.split(r"[^\\w\\d]+", row_text, flags=re.UNICODE):
                if len(tok) >= 3:
                    row_tokens.add(tok)
            for tok in row_tokens:
                tlf[tok][row_code] += 1
                token_doc_count[tok] += 1

        n_law += 1

print(f"Law: {n_law:,} rows indexed in {time.time()-t0:.1f}s")
print(f"Token->code association: {len(tlf):,} tokens, "
      f"avg codes/token = {sum(len(c) for c in tlf.values())/max(1,len(tlf)):.1f}")

# --- Stream court jsonl ------------------------------------------------------
t1 = time.time(); n_court = 0
with open(PATHS["court_v5"], encoding="utf-8") as f:
    for line in f:
        try: obj = json.loads(line)
        except Exception: continue
        cit = obj.get("citation","")
        if not cit: continue
        did = DOC_ID_COURT(n_court)
        cit_to_doc_ids[cit].append(did)
        cb  = obj.get("court_base") or ""
        rag = obj.get("rag_enrichment") or {}
        legal_area_static = obj.get("legal_area_static") or rag.get("legal_area") or ""
        doc_meta[did] = {"citation": cit, "family": "court", "court_base": cb,
                         "paragraph_role": rag.get("paragraph_role"),
                         "is_notification_paragraph": bool(obj.get("is_notification_paragraph"))}
        legal_area_per_doc[did] = (legal_area_static or "").lower()

        if cb:
            idx_court_base[cb].add(did)
            cb_canon = case_anchor_canonical(cb)
            if cb_canon: idx_case_anchor[cb_canon].add(did)

        row_canons = canonicalize_row_anchors(rag.get("statute_anchors") or [], legal_area_static)
        # v7.2 P2: if no anchors, run our augmented regex over text_excerpt_original
        # to catch French/Italian variants the build-time pass missed. Empirically
        # this affects ~233 of 1.09M empty-anchor rows (0.02%) but adds robustness
        # for any unseen test/production query that hits those rows.
        if not row_canons:
            extra = extract_extra_statute_canons(obj.get("text_excerpt_original") or "")
            if extra:
                row_canons = extra
        for canon in row_canons:
            idx_court_statute[canon].add(did)
        if row_canons:
            doc_statute_anchors[did] = row_canons
        rc = sorted(row_canons)
        for i in range(len(rc)):
            for j in range(i+1, len(rc)):
                co_citation_pairs[(rc[i], rc[j])] += 1

        for ca in rag.get("case_anchors") or []:
            canon = case_anchor_canonical(ca)
            if canon: idx_case_anchor[canon].add(did)
        for c in rag.get("concepts_en") or []:
            tok = norm_token(c, CONFIG["lowercase_concepts"])
            if tok: idx_concept_en[tok].add(did)
        for t in rag.get("terms_original") or []:
            tok = norm_token(t, CONFIG["lowercase_terms"])
            if tok: idx_term_orig[tok].add(did)

        search_text[did] = _take_text(
            cit, obj.get("text_excerpt_original",""),
            rag.get("concepts_en"), rag.get("terms_original"),
            rag.get("micro_topic",""), rag.get("topic",""), rag.get("subtopic",""),
            rag.get("statute_anchors"),
        )

        n_court += 1
        if n_court % 500_000 == 0:
            print(f"  court progress: {n_court:,} rows ({time.time()-t1:.1f}s)")

print(f"Court: {n_court:,} rows indexed in {time.time()-t1:.1f}s")
print(f"Total docs:        {len(doc_meta):,}")
print(f"Unique citations:  {len(cit_to_doc_ids):,}")
print(f"Index sizes:       law_direct={len(idx_law_direct):,}, court_statute={len(idx_court_statute):,}, "
      f"case={len(idx_case_anchor):,}, court_base={len(idx_court_base):,}, "
      f"concept={len(idx_concept_en):,}, term={len(idx_term_orig):,}")
print(f"Co-citation pairs: {len(co_citation_pairs):,}")
""")

    add_md(cells, """
## 2.5 Map gold to doc_ids (sanity check)

**What:** For each val_001 gold citation, look up its doc_ids in `cit_to_doc_ids`.

**Why:** If a gold citation has NO doc_id in our corpus, no channel can ever
catch it — it's a Class C miss (gold not in corpus at all). val should be
100% Class A, so all 42 gold should map. Any drop here is a corpus-build
problem, not retrieval.

**Expected:** "Mapped gold: 42/42, Total gold doc_ids: 42".

**Failure modes:**
- < 42 mapped → corpus build is missing those rows (granularity mismatch?
  maybe the gold uses paragraph-level form like "Art. 100 Abs. 1 BGG" but
  corpus has only "Art. 100 BGG"). Investigate before continuing.
""")
    add_code(cells, """
gold_doc_set = set()
unmapped_gold = []
for g in val_gold:
    g = g.strip()
    if not g: continue
    dids = cit_to_doc_ids.get(g, [])
    if not dids:
        unmapped_gold.append(g)
    else:
        gold_doc_set.update(dids)

total_gold = len(val_gold)
print(f"Mapped gold: {total_gold - len(unmapped_gold)}/{total_gold}")
print(f"Total gold doc_ids: {len(gold_doc_set)}")
if unmapped_gold:
    print(f"\\n[WARN] {len(unmapped_gold)} gold citations have NO matching doc_id in corpus:")
    for g in unmapped_gold:
        print(f"  - {g}")
    print("These cannot be retrieved by any channel. Investigate corpus build before continuing.")
""")


def phase_3_graph(cells: list) -> None:
    add_md(cells, """
# Phase 3 — Citation graph (v7 NEW)

## 3.1 What this cell does

Loads `data_insights/citation_graph_extracted.sqlite` (built locally via 4
alias passes; ~2.4 GB; 23.65 M edges) and converts edge tuples (citation_str
→ citation_str) into doc_id-keyed in-memory dicts:
- `idx_graph_out[did]` → list of doc_ids the row text-cites or fans-out to
- `idx_graph_in[did]` → list of doc_ids whose text cites this row

## 3.2 Why we need this — the 4 layers

The graph encodes signal that's **invisible to BM25, vector, and concepts**:

1. **Intra-judgment back-references**
   - Court text uses bare back-refs like `(vgl. E. 6.2 hiervor)` — references
     to siblings of the same judgment. Original `extract_citation_graph.py`
     missed all of these (its CONSIDERATION_RE only fires after a docket).
   - Pass 1 (`extract_intra_judgment_backrefs.py`) handles 4 patterns × 3
     languages: `vgl./siehe E. N`, `E. N hiervor`, `E. N ci-dessus`, `cf. supra
     consid. N`, including `siehe oben E. N`, `hiervor E. N`, `vorstehend`.
   - Self-tested with 15 real-world cases; 4-layer audit on 500 marker-rows
     dropped zero-target rate from 335→175 (90% of remaining are TRUE false
     markers like `nach oben` = "to the top").

2. **Date-stripped aliases**
   - Corpus extraction stores dated form `1B_210/2023 12.05.2023 E. 3`, but
     val gold uses un-dated form `1B_210/2023 E. 3`.
   - Pass 2 adds alias edges from dated → un-dated forms (only when the
     un-dated form exists as a real corpus row).

3. **E.-range expansion**
   - Corpus stores ranges as one citation: `1B_90/2021 E. 2.1-2.4`. Gold uses
     individual Es: `E. 2.1`, `E. 2.2`, `E. 2.3`, `E. 2.4`.
   - Pass 3 enumerates ranges and adds aliases.

4. **Case-level fan-out**
   - When a source cites one E. of a judgment, gold may include OTHER Es of
     the same judgment that nobody text-cites by exact pinpoint. Treats
     "citing one E." as "this case is relevant".
   - Pass 4 fans out: edge → BASE E. X spawns alias edges to all `BASE E. Y`
     where Y exists as a real corpus row.
   - Largest pass: +18.77 M edges.

After all 4 passes: **0/42 val_001 gold orphan** (was 11 originally).

## 3.3 Why doc_id mapping skips synthetic targets

Some graph nodes are synthetic (e.g., a backref `E. 4 hiervor` resolves to
`{base} E. 4` which may not be a real corpus row). Such targets have no
`cit_to_doc_ids` entry and are skipped here — graph channels only retrieve
real corpus rows.

## 3.4 Expected outputs
- ~24 M edges loaded (some skipped because synthetic targets have no doc_id)
- Out-degree avg ~10–15 (reflects case-level fan-out per cited judgment)
- In-degree avg ~10–15
- Load time: 30–60 s

## 3.5 Failure modes
- `graph_db MISSING` → graph channels skipped; sibling_expansion still works.
- `0 edges loaded` → all citations failed to map; check `cit_to_doc_ids`
  was built before this cell, and citations strings are exact (case, spacing).
- `Out-degree avg < 3` → mapping mostly failed; check date format normalization.
""")
    add_code(cells, """
import sqlite3 as _sqlite3

idx_graph_out = defaultdict(list)
idx_graph_in  = defaultdict(list)
GRAPH_OK = bool(GRAPH_AVAILABLE)

if GRAPH_OK:
    _t = time.time()
    _cit_to_did = {cit: dids[0] for cit, dids in cit_to_doc_ids.items() if dids}
    print(f"Built citation->doc_id map ({len(_cit_to_did):,} entries)")

    _g = _sqlite3.connect(str(PATHS["graph_db"]))
    n_loaded = 0; n_skipped = 0
    for _src, _tgt in _g.execute(
        "SELECT source, target FROM edges WHERE dataset='court_considerations'"
    ):
        _s = _cit_to_did.get(_src)
        _t2 = _cit_to_did.get(_tgt)
        if _s is None or _t2 is None:
            n_skipped += 1; continue
        if _s == _t2: continue
        idx_graph_out[_s].append(_t2)
        idx_graph_in[_t2].append(_s)
        n_loaded += 1
    _g.close()
    print(f"Graph: {n_loaded:,} edges loaded, {n_skipped:,} skipped (cit not in corpus)")
    print(f"Graph: out-degree avg = {n_loaded/max(1,len(idx_graph_out)):.1f}, "
          f"in-degree avg = {n_loaded/max(1,len(idx_graph_in)):.1f}")
    print(f"Graph: load time {time.time()-_t:.1f}s")
else:
    print("[skip] Graph DB missing — graph channels will return empty lists.")
""")


def phase_4_bedrock_cocit(cells: list) -> None:
    add_md(cells, """
# Phase 4 — Per-area bedrock + co-citation neighbours

## 4.1 Per-area bedrock — why we need it

For a query in legal area "criminal procedure", certain articles are
**universally cited** by every BGer detention decision (Art. 100 BGG,
Art. 42 BGG, Art. 66 BGG — the procedural/cost cluster). The LLM rarely
names these because they're "implicit" to lawyers but they're often gold.

Per-area bedrock is **corpus-derived**: from `legal_area_per_doc`, count
which canonical statutes appear most often in court rows of each area.

**Filter:** v6 added a critical fix — restrict per-area bedrock to canonicals
whose code matches one of the LLM-named codes (e.g., StPO + BGG for val_001).
v4 returned BGG-dominated lists across ALL areas because BGG appeal articles
are cited everywhere.

**Expected:** ~26 distinct legal areas; `criminal procedure and coercive measures`
top-8 should include 66 BGG, 78 BGG, 81 BGG.

**Failure modes:**
- `0 areas` → `legal_area_per_doc` is empty; check court enrichment has
  `legal_area_static` set.
""")
    add_code(cells, """
print("Building per-area bedrock index...")
_t = time.time()
per_area_canon_count = defaultdict(Counter)
for did, area in legal_area_per_doc.items():
    if not area or area == "law": continue
    canons = doc_statute_anchors.get(did, ())
    for canon in canons:
        per_area_canon_count[area][canon] += 1
print(f"  built in {time.time()-_t:.1f}s; areas: {len(per_area_canon_count)}")
for area in list(per_area_canon_count.keys())[:4]:
    print(f"  area={area!r}: top 8 = {per_area_canon_count[area].most_common(8)}")
""")

    add_md(cells, """
## 4.2 Co-citation neighbours — why we need it

For each statute target the LLM names (e.g., `Art. 221 StPO`), find the top-K
canonical statutes that are **cited together** in the same court rows most
often. Surfaces statute clusters that move together in legal practice.

**Filter:** drop neighbours that are TOO globally common (Art. 36 BV cited
in nearly every criminal case). 5000 = ~0.2% of 2.5 M corpus.

**Expected:** for `221 StPO` neighbours: `212 StPO`, `237 StPO`, `5 StPO`,
`5 EMRK` (the detention statute cluster). NOT `36 BV` (filtered out).

**Failure modes:**
- All-empty neighbours → `co_citation_pairs` was empty; check court enrichment
  contained statute_anchors.
""")
    add_code(cells, """
co_neighbours = defaultdict(list)
canon_count = Counter()
for did, canons in doc_statute_anchors.items():
    for c in canons: canon_count[c] += 1

for (a, b), cnt in co_citation_pairs.items():
    if cnt < CONFIG["co_citation_min_co_count"]: continue
    if canon_count[b] > CONFIG["co_citation_max_neighbour_count"]: pass  # may filter b
    if canon_count[a] > CONFIG["co_citation_max_neighbour_count"]: pass  # may filter a
    co_neighbours[a].append((b, cnt))
    co_neighbours[b].append((a, cnt))

# Apply frequency filter on neighbour side and keep top-K per source
co_neighbours = {
    src: sorted(
        ((nb, n) for nb, n in nbs if canon_count[nb] <= CONFIG["co_citation_max_neighbour_count"]),
        key=lambda x: -x[1]
    )[:CONFIG["co_citation_top_k_per_target"] * 2]
    for src, nbs in co_neighbours.items()
}
print(f"Co-citation neighbours indexed for {len(co_neighbours):,} canonicals "
      f"(after frequency filter: max global count = {CONFIG['co_citation_max_neighbour_count']}).")
print("Sample - neighbours of '221 StPO' AFTER frequency filter:")
for nb, n in co_neighbours.get("221 StPO", [])[:8]:
    print(f"  {nb}: co={n}, total_in_corpus={canon_count[nb]}")
""")


def phase_5_bm25(cells: list) -> None:
    add_md(cells, """
# Phase 5 — BM25 (FTS5 in-memory)

## 5.1 What this cell does

Build SQLite FTS5 over `search_text[did]` for all 2.65 M docs. Provides
`bm25_search(query_text, k)` which returns top-k doc_ids ranked by BM25.

## 5.2 Why FTS5 (and not Whoosh / Lucene)

- Pure stdlib; no extra install.
- ~80 s build for 2.6 M short docs.
- Returns BM25-scored top-K in <50 ms.
- We can pass any expanded query text and get a stable ranking.

## 5.3 enhance() — corpus-derived query enrichment

Untitled75's reference notebook used a `enhance()` from train data that we
forbid. We replicate the IDEA (boost query with code names most associated
with query tokens) but train it on the **corpus** (laws_de) instead of train.

For each token in query, look up `tlf[token]` (a Counter mapping legal codes
to row counts). The top-K codes with highest score get appended to the query
multiple times, biasing BM25 toward law rows of those codes.

Example: query contains "detention" → boost `stpo` (high) more than `or`.

**Expected:** FTS5 build ~80 s. enhance() boost typically adds 5×5=25 token
repetitions to the query.

**Failure modes:**
- Slow build (>180 s) → swap MEMORY journal for OFF, or use a temp file.
""")
    add_code(cells, """
import sqlite3, math

print(f"Building in-memory FTS5 over {len(search_text):,} docs...")
_t = time.time()
_fts = sqlite3.connect(":memory:")
_fts.execute("PRAGMA journal_mode = MEMORY")
_fts.execute("PRAGMA synchronous = OFF")
_fts.execute("CREATE VIRTUAL TABLE docs USING fts5(did UNINDEXED, body, tokenize = 'unicode61 remove_diacritics 2')")
_inserted = 0
_batch = []
for did, txt in search_text.items():
    _batch.append((did, txt))
    if len(_batch) >= 50000:
        _fts.executemany("INSERT INTO docs(did, body) VALUES (?, ?)", _batch)
        _inserted += len(_batch); _batch.clear()
        if _inserted % 500000 == 0:
            print(f"  inserted {_inserted:,} ({time.time()-_t:.1f}s)")
if _batch:
    _fts.executemany("INSERT INTO docs(did, body) VALUES (?, ?)", _batch)
    _inserted += len(_batch)
_fts.commit()
print(f"FTS5 built: {_inserted:,} rows in {time.time()-_t:.1f}s")

# enhance(): given a query string, return enriched query with corpus-associated codes
def _enhance_codes(text):
    text_lc = text.lower()
    tokens = set()
    for tok in re.split(r"[^\\w\\d]+", text_lc, flags=re.UNICODE):
        if len(tok) >= CONFIG["bm25_min_token_len"]:
            tokens.add(tok)
    code_score = Counter()
    for tok in tokens:
        if tok not in tlf: continue
        n_docs = max(1, token_doc_count[tok])
        idf = math.log(1 + (max(1, len(search_text)) / n_docs))
        if idf < CONFIG["enhance_min_idf"]: continue
        for code, cnt in tlf[tok].most_common():
            code_score[code] += cnt * idf
    return [c for c, _ in code_score.most_common(CONFIG["enhance_top_k_codes"])]

def bm25_search(query_text, k):
    boosted = _enhance_codes(query_text)
    enriched = query_text + " " + " ".join(c * CONFIG["enhance_repeat_count"] for c in boosted)
    # tokenize for FTS5: keep only alpha tokens >= min_len, cap to budget
    fts_q = []
    for tok in re.split(r"[^\\w\\d]+", enriched, flags=re.UNICODE):
        if len(tok) >= CONFIG["bm25_min_token_len"]:
            fts_q.append(tok)
            if len(fts_q) >= CONFIG["bm25_max_query_terms"]: break
    if not fts_q: return []
    fts_query = " OR ".join(f'"{t}"' for t in fts_q)
    rows = _fts.execute(
        "SELECT did, bm25(docs) FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?",
        (fts_query, k),
    ).fetchall()
    # FTS5 returns negative scores (lower = more relevant); flip sign for downstream
    return [(did, -score) for did, score in rows]
""")


def phase_6_query_expansion(cells: list) -> None:
    add_md(cells, """
# Phase 6 — Query expansion for ALL 10 val queries (Qwen3-32B)

## 6.1 Memory contract

This phase is the FIRST GPU-resident phase. Before it: 0 GB on GPU.
- Load Qwen3-32B in bf16 → ~65 GB VRAM (loaded ONCE)
- **Loop over ALL 10 val queries**, run generate() per query, parse JSON
- After all 10 queries done: delete model + tokenizer + inputs/outputs
- `gc.collect()` → `torch.cuda.empty_cache()` → `torch.cuda.synchronize()`
- Print VRAM-after-free as proof of release

After this phase, NO model is on GPU. Phase 7 then loads Qwen3-Embedding-8B.
Total wall-time: ~210 s load + 10 × ~3 s generate = ~240 s.

## 6.2 What this cell does

Loads Qwen3-32B once. For each of the 10 val queries, runs the same
structured prompt and parses the JSON to a per-query `targets` dict.
All targets are stored in `ALL_TARGETS[query_id]`.

- `statute_targets`: "Art. N CODE" strings → fed to `law_direct_match`,
  `court_statute`, `co_citation`, `per_area_bedrock` filter.
- `concept_targets_en`: English legal concepts → fed to `concept_en`.
- `term_targets_de` / `term_targets_fr`: original-language terms → fed to
  `term_orig` and BM25.
- `legal_area_keywords`: 1-5 legal-area phrases → fed to `per_area_bedrock`.
- `case_targets`: BGE/docket case targets (NOT used as anchor channel —
  v5 measured: LLM hallucinates BGE numbers; dropped).

## 6.3 Why structured JSON expansion (not HyDE)

Plain HyDE (generate fake answer, embed it) was tried in v5 and added zero
measurable recall. Structured targets give explicit signals to non-vector
channels (BM25 query enrichment, statute matching, concept overlap).

## 6.4 The `apply_chat_template` invocation — known pitfall

The tokenizer's `apply_chat_template` returns a `BatchEncoding` (dict-like)
object when `return_dict=True`. Without `return_dict`, it can return a bare
tensor in some transformer versions; passing that to `generate()` as
`input_ids=_inp` triggers `KeyError: 'shape'`. We use `return_dict=True`
and unpack with `**_inp`. We also explicitly find the model's first real
device (some weights may be on `meta` under `device_map="auto"`).

## 6.5 Expected outputs

For each query: 4-8 statute_targets, 5-25 concepts/terms (variable; see
Phase 11 for diagnosis if too few), 5 legal_area_keywords. v7 saw 5 each
for val_001 — short. v7.1 prompt asks explicitly for >=10 per category.

After free: `[free:qwen3-32b] VRAM used=0.00 GB, free=~95 GB`

## 6.6 Failure modes

- `KeyError: 'shape'` → `return_dict=True` not set in `apply_chat_template`.
- `<think>` chain-of-thought eats tokens → `enable_thinking=False` set.
- JSON parse fails → `targets` defaults to empty lists; subsequent channels
  degrade gracefully (concept_en, term_orig return 0 hits).
- Free reports VRAM > 5 GB → some tensor still alive; check kernel didn't
  retain `_resp` referencing GPU memory (decode returns CPU str so this is
  rarely the cause).
""")
    add_code(cells, """
QEXP_PROMPT_SYSTEM = (
    "You are a Swiss legal-research assistant. You enumerate broadly across "
    "the most relevant Swiss federal codes (StPO, StGB, BGG, ZGB, OR, ZPO, BV, "
    "EMRK, IPRG, IRSG, AHVG, IVG, AsylG, AIG, DBG, StHG, KVG, UVG, AVIG, PatG, "
    "MSchG, URG, FINIG, FINMAG, BankG, KKG, SchKG, VwVG, FZG, BVG, BPV, BetmG, "
    "RPG, NHG, USG, GSchG, SVG, RAG, RPG, OBG, GwG, BankG, etc). You output "
    "ONLY a single JSON object — no prose, no markdown, no code fences."
)

# Two diverse few-shots to avoid example-echo failures, neither of which
# matches val_001 (so val_001 doesn't degenerate into echo-the-example).
QEXP_PROMPT_USER = '''Schema (exact keys, all required):

{
  "statute_targets":      [10-20 "Art. N CODE" strings, broad coverage of relevant codes],
  "case_targets":         [list of "BGE V D P" or "1B_N/Y" dockets — empty if uncertain],
  "concept_targets_en":   [15-25 English legal concepts (technical AND procedural)],
  "term_targets_de":      [15-25 German legal terms in original spelling],
  "term_targets_fr":      [10-20 French legal terms],
  "legal_area_keywords":  [3-6 short legal-area phrases]
}

Hard rules:
- Output ONLY the JSON object. No surrounding text, no markdown, no code fences.
- Each list MUST have at least the minimum number of items.
- Do NOT invent specific BGE volumes or dockets you are uncertain about.

Example A for "Does an unemployed worker keep insurance benefits when he refuses a job offer that is below his prior wage?":
{"statute_targets":["Art. 16 AVIG","Art. 17 AVIG","Art. 30 AVIG","Art. 30 Abs. 1 AVIG","Art. 16 Abs. 2 AVIG","Art. 22 AVIG","Art. 23 AVIG","Art. 8 AVIG","Art. 95 AVIG","Art. 11 AVIG"],
"case_targets":[],
"concept_targets_en":["unemployment insurance","suitable employment","wage protection","willingness to work","refusal of suitable work","reduction of benefits","good cause","unemployment compensation","insured earnings","obligation to accept","sanction","placement","right to compensation","intermediate earnings","admissibility threshold"],
"term_targets_de":["Arbeitslosenversicherung","zumutbare Arbeit","Lohnvergleich","Vermittlungsfähigkeit","Ablehnung","Einstellung in der Anspruchsberechtigung","Versicherungsleistungen","Arbeitslosenentschädigung","Zwischenverdienst","Vermittlungsbemühungen","Arbeitsbemühungen","versicherter Verdienst","Selbstverschulden","Sanktion","Verfügung"],
"term_targets_fr":["assurance-chômage","emploi convenable","comparaison de salaire","aptitude au placement","refus","suspension du droit à l'indemnité","indemnisation","gain intermédiaire","obligation","sanction"],
"legal_area_keywords":["unemployment insurance","social insurance","labour market","social security law"]}

Example B for "Is a will written on lined notebook paper and signed only on the last page valid under Swiss inheritance law?":
{"statute_targets":["Art. 505 ZGB","Art. 498 ZGB","Art. 499 ZGB","Art. 519 ZGB","Art. 520 ZGB","Art. 6 ZGB","Art. 467 ZGB","Art. 468 ZGB","Art. 522 ZGB","Art. 540 ZGB"],
"case_targets":[],
"concept_targets_en":["holographic will","testator","handwriting","signature","testamentary capacity","formal validity","formal requirements","invalidity","challenge of will","disposition mortis causa","heirship","forced heirship","compulsory portion","inheritance","reduction action"],
"term_targets_de":["eigenhändige Verfügung","Testament","Erblasser","Handschrift","Unterschrift","Verfügungsfähigkeit","Formvorschriften","Ungültigerklärung","Anfechtung","letztwillige Verfügung","Erbe","Pflichtteil","Erbschaft","Herabsetzung","Verfügung von Todes wegen"],
"term_targets_fr":["testament olographe","testateur","écriture","signature","capacité de disposer","conditions de forme","nullité","action en réduction","disposition pour cause de mort","héritage","réserve héréditaire"],
"legal_area_keywords":["inheritance law","succession","testamentary law","civil law"]}

Now produce the JSON for this query (JSON object only, no other text):
{QUERY}
'''

import time, gc, re as _re, json as _json
import torch as _torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_targets_json(text: str):
    \"\"\"Robust JSON extractor: tries (a) markdown-fenced JSON, (b) brace-balanced
    object scan, (c) every {...} substring as a last resort. Returns dict or None.\"\"\"
    if not text:
        return None
    # 1) ```json {...} ``` or ``` {...} ```
    fence = _re.search(r'```(?:json|JSON)?\\s*(\\{.*?\\})\\s*```', text, _re.S)
    if fence:
        try: return _json.loads(fence.group(1))
        except Exception: pass
    # 2) brace-balanced scan starting from each '{'
    n = len(text)
    i = 0
    while i < n:
        i = text.find('{', i)
        if i < 0: break
        depth = 0; in_str = False; esc = False; j = i
        while j < n:
            c = text[j]
            if esc: esc = False
            elif c == '\\\\': esc = True
            elif c == '\"': in_str = not in_str
            elif not in_str:
                if c == '{': depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[i:j+1]
                        try: return _json.loads(candidate)
                        except Exception: break
            j += 1
        i += 1
    # 3) very greedy fallback
    m = _re.search(r'\\{.*\\}', text, _re.S)
    if m:
        try: return _json.loads(m.group(0))
        except Exception: pass
    return None


def normalize_targets(d):
    \"\"\"Coerce whatever the model returned into the expected schema. Keys it
    doesn't fill default to []. Lists of dicts get flattened to strings.\"\"\"
    out = {k: [] for k in (
        "statute_targets","case_targets","concept_targets_en",
        "term_targets_de","term_targets_fr","legal_area_keywords",
    )}
    if not isinstance(d, dict): return out
    for k in out.keys():
        v = d.get(k)
        if v is None: continue
        if isinstance(v, str):
            out[k] = [v.strip()] if v.strip() else []
        elif isinstance(v, list):
            flat = []
            for item in v:
                if isinstance(item, str) and item.strip():
                    flat.append(item.strip())
                elif isinstance(item, dict):
                    # take any string value as the term
                    for vv in item.values():
                        if isinstance(vv, str) and vv.strip():
                            flat.append(vv.strip()); break
            out[k] = flat
        # silently drop other types
    return out


print(f"[qexp] loading {CONFIG['qwen_query_model']} (~65 GB bf16)...")
_t = time.time()
qtok = AutoTokenizer.from_pretrained(CONFIG["qwen_query_model"])
qmod = AutoModelForCausalLM.from_pretrained(
    CONFIG["qwen_query_model"],
    dtype=_torch.bfloat16,
    device_map="auto",
)
qmod.eval()
print(f"[qexp] model loaded in {time.time()-_t:.1f}s")

ALL_TARGETS = {}
ALL_RAW_RESPONSES = {}  # kept for diagnosis even when parse succeeds

for q in ALL_QUERIES:
    qid = q["query_id"]
    qtext = q["query"]
    print(f"\\n[qexp] {qid} ...")
    _messages = [
        {"role": "system", "content": QEXP_PROMPT_SYSTEM},
        {"role": "user",   "content": QEXP_PROMPT_USER.replace("{QUERY}", qtext)},
    ]
    _inp = qtok.apply_chat_template(
        _messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    )
    _input_device = next(p.device for p in qmod.parameters() if p.device.type != "meta")
    _inp = {k: v.to(_input_device) for k, v in _inp.items()}

    # First attempt: deterministic.
    with _torch.no_grad():
        _out = qmod.generate(
            **_inp,
            max_new_tokens=CONFIG["qwen_max_new_tokens"],
            do_sample=False,
            pad_token_id=qtok.eos_token_id,
        )
    _prompt_len = _inp["input_ids"].shape[1]
    _resp = qtok.decode(_out[0][_prompt_len:], skip_special_tokens=True)

    raw = parse_targets_json(_resp)
    targets = normalize_targets(raw) if raw is not None else {k: [] for k in (
        "statute_targets","case_targets","concept_targets_en",
        "term_targets_de","term_targets_fr","legal_area_keywords")}

    # If first attempt produced an empty dict OR very short lists, retry with
    # a small temperature to break out of any deterministic failure mode.
    if (raw is None) or (sum(len(v) for v in targets.values()) < 10):
        print(f"  [qexp] {qid} first-pass parse weak (raw={raw is not None}); retrying with temperature=0.4")
        with _torch.no_grad():
            _out2 = qmod.generate(
                **_inp,
                max_new_tokens=int(CONFIG["qwen_max_new_tokens"] * 1.5),
                do_sample=True,
                temperature=0.4,
                top_p=0.95,
                pad_token_id=qtok.eos_token_id,
            )
        _resp2 = qtok.decode(_out2[0][_prompt_len:], skip_special_tokens=True)
        raw2 = parse_targets_json(_resp2)
        targets2 = normalize_targets(raw2) if raw2 is not None else None
        if targets2 and sum(len(v) for v in targets2.values()) > sum(len(v) for v in targets.values()):
            targets = targets2
            _resp = _resp2

    # Persist raw for post-mortem.
    ALL_RAW_RESPONSES[qid] = _resp
    ALL_TARGETS[qid] = targets
    summary = (f"statute={len(targets['statute_targets'])} "
               f"concept={len(targets['concept_targets_en'])} "
               f"term_de={len(targets['term_targets_de'])} "
               f"term_fr={len(targets['term_targets_fr'])} "
               f"area={len(targets['legal_area_keywords'])}")
    print(f"  {summary}")
    if sum(len(v) for v in targets.values()) == 0:
        # Print raw response so we can see what the model actually said.
        print(f"  [qexp][WARN] {qid}: empty targets after retry. "
              f"Raw response (first 600 chars):\\n  {_resp[:600]!r}")

# legacy var for the val_001 detailed-diagnosis path in Phase 11.
targets = ALL_TARGETS["val_001"]

# --- FREE Qwen3-32B from VRAM ----------------------------------------------
del qmod, qtok, _inp, _out
gc.collect()
if _torch.cuda.is_available():
    _torch.cuda.empty_cache()
    _torch.cuda.synchronize()
    _used = _torch.cuda.memory_allocated() / 1024**3
    _free = (_torch.cuda.get_device_properties(0).total_memory
             - _torch.cuda.memory_allocated()) / 1024**3
    print(f"\\n[free:qwen3-32b] VRAM used={_used:.2f} GB, free={_free:.2f} GB")
else:
    print("\\n[free:qwen3-32b] no CUDA")
""")


def phase_7_encode_queries(cells: list) -> None:
    add_md(cells, """
# Phase 7 — Encode queries (Qwen3-Embedding-8B → q_emb_raw + q_emb_enriched)

## 7.1 Memory contract

Before this phase: 0 GB on GPU (Qwen3-32B was freed at end of Phase 6).
- Load Qwen3-Embedding-8B → ~16 GB VRAM
- Encode raw query AND enriched query (~2 forward passes, fast)
- **Move embeddings to CPU** so they survive after model is freed
- Delete model → gc → empty_cache → synchronize
- Print VRAM-after-free as proof of release

After this phase: 0 GB models on GPU. Two CPU tensors `q_emb_raw` and
`q_emb_enriched` (4096-dim each, ~16 KB total) cached for use by
`vector_search()` in Phase 9. Phase 8 then loads E_GPU.

## 7.2 What this cell does

- Loads Qwen3-Embedding-8B via SentenceTransformer (handles instruction
  prefix internally with `prompt_name="query"`).
- Encodes the raw query (the verbatim English question).
- Encodes the **enriched query** = raw + ` ` + first 60 tokens from the
  union of `term_targets_de`, `term_targets_fr`, `concept_targets_en`.
- Moves both tensors to CPU (`.cpu()`) so they survive past `del EMB_MODEL`.

## 7.3 Why two embeddings

Different recall surfaces. Raw captures literal sentence semantics; enriched
shifts the vector toward the Swiss legal domain and matches more
domain-aligned corpus rows. RRF fuses both — multi-channel hits bubble up.

## 7.4 Expected outputs

- Two 4096-dim normalized fp32 (or fp16) vectors on CPU
- Enriched query: ~2300 chars, ~60 keywords appended
- VRAM after free: ~0 GB

## 7.5 Failure modes

- VRAM OOM during model load → Qwen3-32B not fully freed in Phase 6;
  re-run Phase 6 cleanup.
- `prompt_name="query"` not supported → SentenceTransformer version
  mismatch; downgrade or pass instruction text manually.
""")
    add_code(cells, """
import time, gc
import torch as _torch
from sentence_transformers import SentenceTransformer

print("Loading Qwen3-Embedding-8B...")
_t = time.time()
EMB_MODEL = SentenceTransformer(CONFIG["vector_emb_model"])
print(f"  loaded in {time.time()-_t:.1f}s")

def encode_query(text):
    return EMB_MODEL.encode(
        [text], prompt_name="query",
        convert_to_tensor=True, normalize_embeddings=True,
    )[0]

ALL_Q_EMB_RAW = {}
ALL_Q_EMB_ENRICHED = {}
ALL_ENRICHED_TEXT = {}

for q in ALL_QUERIES:
    qid = q["query_id"]
    qtext = q["query"]
    targets_q = ALL_TARGETS.get(qid, {})
    enriched_bits = (
        (targets_q.get("term_targets_de") or [])
      + (targets_q.get("term_targets_fr") or [])
      + (targets_q.get("concept_targets_en") or [])
    )
    enriched_query = qtext + " " + " ".join(enriched_bits[:60])
    ALL_ENRICHED_TEXT[qid] = enriched_query
    ALL_Q_EMB_RAW[qid]      = encode_query(qtext).detach().cpu()
    ALL_Q_EMB_ENRICHED[qid] = encode_query(enriched_query).detach().cpu()
    print(f"  {qid}: enriched_len={len(enriched_query)} "
          f"keywords_appended={min(60, len(enriched_bits))}")

# legacy vars used by the detailed val_001 diagnostic in Phase 11
q_emb_raw      = ALL_Q_EMB_RAW["val_001"]
q_emb_enriched = ALL_Q_EMB_ENRICHED["val_001"]

# --- FREE Qwen3-Embedding-8B from VRAM -------------------------------------
del EMB_MODEL
gc.collect()
if _torch.cuda.is_available():
    _torch.cuda.empty_cache()
    _torch.cuda.synchronize()
    _used = _torch.cuda.memory_allocated() / 1024**3
    _free = (_torch.cuda.get_device_properties(0).total_memory
             - _torch.cuda.memory_allocated()) / 1024**3
    print(f"\\n[free:qwen3-emb-8b] VRAM used={_used:.2f} GB, free={_free:.2f} GB")
else:
    print("\\n[free:qwen3-emb-8b] no CUDA")
""")


def phase_8_vector_setup(cells: list) -> None:
    add_md(cells, """
# Phase 8 — Vector channel setup (E_GPU)

## 8.1 Memory contract

Before this phase: 0 GB on GPU (both Qwen models were freed earlier).
- Load 27 fp16 embedding chunks → concat to single `E_GPU` tensor of shape
  (2.65 M, 4096) → ~22 GB VRAM **sustained** until Phase 12 cleanup.
- Define `vector_search(q_emb, k)` doing brute-force `E_GPU @ q` on GPU.

This is the only GPU-resident object for Phase 9-11. Q_emb_raw / q_emb_enriched
were moved to CPU in Phase 7; `vector_search` moves them to E_GPU.device on call.

## 8.2 What this cell does

- Reads `qwen3_8b_unified_manifest.parquet` to map row_index ↔ doc_id.
- Loads 27 chunk .npy files, concatenates to `E_GPU` on cuda:0.
- Defines `vector_search(q_emb, k)` and sets `VECTOR_OK=True`.

## 8.3 Why brute-force GPU and not FAISS-IVF

- 22 GB fits comfortably in 95 GB Blackwell.
- Matmul `E_GPU @ q` ≈ 50 ms; FAISS-IVF query is comparable but adds index-
  build time and quantization recall loss.
- Per `personal_observations.md` Obs 3: dense embedding alone caps at
  R@1000 = 0.289 on val regardless of index. Bottleneck is "wrong kind of
  relationship for cosine similarity", not retrieval algorithm.

## 8.4 Expected outputs

- manifest rows: ~2.65 M
- mapping coverage: ~99.9 %
- `E_GPU shape=(2652248, 4096)` dtype=`torch.float16`
- VRAM after load: ~22 GB

## 8.5 Failure modes

- `EMB_AVAILABLE = False` → vector channels skip; phase prints `[skip]` and
  Phase 9 returns empty hits for vector_raw / vector_enriched.
- VRAM OOM during `torch.cat` → cat allocates 2× transiently; use in-place
  copy_ pattern if chunk count grows.
- mapping coverage < 95 % → manifest doc_id form drifted from corpus build;
  spot-check a few citations.
""")
    add_code(cells, """
VECTOR_OK = False
E_GPU = None
my_did_for_row = None
row_for_did = None

if EMB_AVAILABLE:
    import torch as _torch, numpy as _np, pandas as _pd, time as _time
    print("Loading manifest...")
    _t = _time.time()
    _man = _pd.read_parquet(PATHS["emb_manifest"])
    print(f"  manifest rows: {len(_man):,}, cols: {list(_man.columns)}")
    row_for_did = {}; my_did_for_row = [None] * len(_man)
    for _, r in _man.iterrows():
        cit = r.get("citation") or ""
        fam = r.get("family") or ""
        ridx = int(r.get("row_index", -1))
        if ridx < 0: continue
        candidates = cit_to_doc_ids.get(cit, [])
        for d in candidates:
            if doc_meta.get(d, {}).get("family") == fam:
                row_for_did[d] = ridx
                if ridx < len(my_did_for_row):
                    my_did_for_row[ridx] = d
                break
    n_mapped = sum(1 for x in my_did_for_row if x is not None)
    print(f"  manifest->my_did mapping: {n_mapped:,}/{len(_man):,} "
          f"({100*n_mapped/len(_man):.1f}%) in {_time.time()-_t:.1f}s")

    print("  loading 27 chunks to GPU (~21 GB)...")
    _t = _time.time()
    _chunks = sorted(PATHS["emb_dir"].glob("qwen3_8b_unified_chunk*.npy"))
    arrs = []
    for cp in _chunks:
        arrs.append(_torch.from_numpy(_np.load(cp)).to("cuda", non_blocking=True))
    E_GPU = _torch.cat(arrs, dim=0); del arrs
    free_vram = (_torch.cuda.get_device_properties(0).total_memory
                 - _torch.cuda.memory_allocated()) / 1024**3
    print(f"  E_GPU shape={tuple(E_GPU.shape)} dtype={E_GPU.dtype}, "
          f"VRAM used={_torch.cuda.memory_allocated()/1024**3:.1f} GB, "
          f"free={free_vram:.1f} GB, "
          f"load {_time.time()-_t:.1f}s")

    def vector_search(q_emb, k):
        if E_GPU is None: return []
        with _torch.no_grad():
            q = q_emb.to(E_GPU.device, dtype=E_GPU.dtype)
            q = q / (q.norm(dim=-1, keepdim=True) + 1e-9)
            scores = E_GPU @ q
            top_v, top_i = _torch.topk(scores, k=min(k, scores.shape[0]))
        out = []
        for s, i in zip(top_v.cpu().tolist(), top_i.cpu().tolist()):
            d = my_did_for_row[i]
            if d is not None: out.append((d, float(s)))
        return out

    VECTOR_OK = True
else:
    print("[skip] EMB_AVAILABLE = False; vector channels will be skipped.")
    def vector_search(q_emb, k): return []
""")


def phase_8_channels(cells: list) -> None:
    add_md(cells, """
# Phase 9 — Channels

## 9.1 What this section does

Defines all retrieval channels as standalone functions, runs them, prints
per-channel R@K (recall against val_001 gold).

## 9.2 The 14 channels and which gap each closes

| # | Channel | Index used | Gap closed |
|---|---|---|---|
| 1 | `law_direct_match` | `idx_law_direct` | Statute targets named by LLM → matching law rows. Uncapped. |
| 2 | `court_statute` | `idx_court_statute` | LLM-named statute appears in row's `statute_anchors`. |
| 3 | `co_citation` | `co_neighbours` + `idx_law_direct` + `idx_court_statute` | Statute cluster co-cited with LLM target (e.g., 221 StPO + 212 StPO). |
| 4 | `per_area_bedrock` | `per_area_canon_count` filtered by LLM codes | Universal procedural articles per legal area. |
| 5 | `statute_backprop` | `doc_statute_anchors` | Caught court rows → law articles they cite (procedural cluster). |
| 6 | `sibling_expansion` | `idx_court_base` | All Es of caught judgments (own-judgment fan-out). v6 budget bug fixed. |
| 7 | **`graph_forward`** (v7 NEW) | `idx_graph_out` | Caught row → text-cited targets + case-level fan-out across judgments. |
| 8 | **`graph_reverse`** (v7 NEW) | `idx_graph_in` | Caught row ← rows that text-cite it (co-citing peers). |
| 9 | **`graph_2hop`** (v7 NEW, off by default) | `idx_graph_out` | 2-hop forward expansion. Disable unless Phase 10 says to enable. |
| 10 | `concept_en` | `idx_concept_en` | LLM concepts → English-tagged rows. |
| 11 | `term_orig` | `idx_term_orig` | LLM DE/FR terms → original-language-tagged rows. |
| 12 | `bm25` | FTS5 + enhance() | Lexical match including code-name boosts. |
| 13 | `vector_raw` | `E_GPU` brute-force | Dense semantic match on raw query. |
| 14 | `vector_enriched` | `E_GPU` brute-force | Dense semantic match on keyword-enriched query. |

## 9.3 Channel function definitions
""")
    add_code(cells, """
from collections import Counter

def channel_law_direct(canon_set, idx, budget):
    counter = Counter()
    for canon in canon_set:
        for did in idx[canon]: counter[did] += 1
    items = counter.most_common()
    return items if budget is None else items[:budget]

def channel_court_statute(statute_canons, idx, budget):
    counter = Counter()
    for canon in statute_canons:
        for did in idx[canon]: counter[did] += 1
    return counter.most_common(budget)

def channel_sibling(seed_doc_ids, idx_court_base, doc_meta, budget):
    \"\"\"v7.1 FIX: rank siblings by seed-frequency, not alphabetical.

    A sibling's score = number of distinct seeds whose court_base it shares.
    Judgments with multiple caught Es (high score) rank above judgments with
    one caught E. (low score). Within a tie, ordering is Counter-stable.

    v7 used `sorted(out)[:budget]` — a non-prioritized alphabetical slice
    that dropped val_001 sibling Es non-deterministically.
    \"\"\"
    seed_set = set(seed_doc_ids)
    cb_seed_count = Counter()
    for did in seed_doc_ids:
        m = doc_meta.get(did) or {}
        cb = m.get("court_base")
        if cb:
            cb_seed_count[cb] += 1
    counter = Counter()
    for cb, score in cb_seed_count.items():
        for s in idx_court_base.get(cb, ()):
            if s in seed_set:
                continue
            if counter[s] < score:
                counter[s] = score
    return counter.most_common(budget)

def channel_graph_forward(seed_doc_ids, idx_graph_out, budget):
    counter = Counter()
    seed_set = set(seed_doc_ids)
    for did in seed_doc_ids:
        for t in idx_graph_out.get(did, ()):
            counter[t] += 1
    for d in seed_set:
        counter.pop(d, None)
    return counter.most_common(budget)

def channel_graph_reverse(seed_doc_ids, idx_graph_in, budget):
    counter = Counter()
    seed_set = set(seed_doc_ids)
    for did in seed_doc_ids:
        for s in idx_graph_in.get(did, ()):
            counter[s] += 1
    for d in seed_set:
        counter.pop(d, None)
    return counter.most_common(budget)

def channel_graph_2hop(seed_doc_ids, idx_graph_out, budget):
    seed_set = set(seed_doc_ids)
    intermediate = set()
    for did in seed_doc_ids:
        intermediate.update(idx_graph_out.get(did, ()))
    intermediate -= seed_set
    counter = Counter()
    for x in intermediate:
        for t in idx_graph_out.get(x, ()):
            if t in seed_set: continue
            counter[t] += 1
    for d in intermediate:
        counter.pop(d, None)
    return counter.most_common(budget)

def expand_concepts_strict(llm_concepts, corpus_concept_keys, top_k):
    expanded = []; seen = set()
    keys = list(corpus_concept_keys)
    for raw in llm_concepts:
        c = (raw or "").lower().strip()
        if not c or len(c) < 4: continue
        if c in corpus_concept_keys and c not in seen:
            expanded.append(c); seen.add(c)
        cands = []
        for cv in keys:
            if c in cv or cv in c:
                cands.append((abs(len(cv) - len(c)), cv))
        cands.sort()
        for _, cv in cands[:top_k]:
            if cv not in seen: expanded.append(cv); seen.add(cv)
    return expanded

def channel_concept(expanded_concepts, idx, budget):
    counter = Counter()
    for tok in expanded_concepts:
        for did in idx.get(tok, ()): counter[did] += 1
    return counter.most_common(budget)

def channel_term(targets, idx, budget):
    counter = Counter()
    for key in ("term_targets_de", "term_targets_fr"):
        for raw in targets.get(key, []) or []:
            tok = norm_token(raw, CONFIG["lowercase_terms"])
            if tok:
                for did in idx[tok]: counter[did] += 1
    return counter.most_common(budget)

def channel_per_area_bedrock(legal_area_keywords, statute_target_codes,
                             per_area_canon_count, idx_law_direct, budget):
    if not legal_area_keywords: return []
    keys = [k.lower() for k in legal_area_keywords]
    selected_areas = set()
    for area in per_area_canon_count.keys():
        for k in keys:
            if k in area: selected_areas.add(area); break
    if not selected_areas: return []
    canon_score = Counter()
    for area in selected_areas:
        for canon, n in per_area_canon_count[area].most_common(CONFIG["per_area_top_n"]):
            canon_score[canon] = max(canon_score[canon], n)
    out = []; seen = set()
    for canon, _ in canon_score.most_common():
        canon_code = canon.split()[1] if canon and " " in canon else None
        if statute_target_codes and canon_code not in statute_target_codes:
            continue
        for did in idx_law_direct.get(canon, set()):
            if did not in seen:
                out.append((did, canon_score[canon])); seen.add(did)
        if len(out) >= budget: break
    return out[:budget]

def channel_statute_backprop(seed_court_dids, doc_statute_anchors, idx_law_direct, budget):
    canon_to_courts = defaultdict(set)
    for did in seed_court_dids:
        for canon in doc_statute_anchors.get(did, set()):
            canon_to_courts[canon].add(did)
    counter = Counter()
    for canon, court_set in canon_to_courts.items():
        score = len(court_set)
        for law_did in idx_law_direct.get(canon, set()):
            if counter[law_did] < score: counter[law_did] = score
    return counter.most_common(budget)

def channel_co_citation(targets, co_neighbours, idx_law_direct, idx_court_statute, budget):
    counter = Counter()
    for raw in targets.get("statute_targets", []) or []:
        canon = statute_anchor_canonical(raw)
        if not canon: continue
        for nb, n in co_neighbours.get(canon, []):
            for did in idx_law_direct.get(nb, set()):
                counter[did] = max(counter[did], n)
            for did in idx_court_statute.get(nb, set()):
                counter[did] = max(counter[did], n)
    return counter.most_common(budget)

def channel_vector(q_emb, k):
    return vector_search(q_emb, k)
""")

    add_md(cells, """
## 9.4 Per-query retrieval loop (channels + fusion + R@1000)

**What:** For each of the 10 val queries, runs every channel, builds the
seed pool, runs sibling + graph channels, fuses via RRF with round-robin
guarantee, computes R@1000.

**Per query, prints:** seed size, channel sizes, per-channel recall, union
ceiling, R@1000.

**Stores in `PER_QUERY[qid]`** for the aggregate phase + diagnosis:
- `gold_doc_set`: mapped gold doc_ids
- `total_gold`: count of gold citations
- `CHANNELS`: list of (name, hits) tuples
- `channel_recalls`: name → recall
- `union_recall`: upper bound on R@K
- `final_topk`: top-1000 doc_ids
- `R@1000`: final recall
- `rrf_scores`, `ranked_dids`: needed by Phase 11 diagnostic for val_001

**Memory:** no model on GPU; only E_GPU (~22 GB) is resident.

**Failure modes:**
- A channel returns 0 across ALL queries → its index is empty, go back to
  Phase 2 build.
- val_005 (or any specific query) R@1000 << others → check that query's
  ALL_TARGETS for empty lists.
""")
    add_code(cells, """
def _court_hits(hits):
    return {d for d, _ in hits if doc_meta.get(d, {}).get("family") == "court"}

def rrf_fuse(channels, k, weights=None):
    \"\"\"Weighted Reciprocal Rank Fusion.

    score(did) = sum over channels of  weights[ch] / (k + rank_in_ch + 1)

    `weights` defaults to 1.0 per channel. v7.3 introduces non-uniform weights
    so high-recall channels (statute_backprop, graph_forward) contribute more
    per rank — this surfaces single-channel deep-rank gold that v7.2 lost in
    fusion.
    \"\"\"
    if weights is None: weights = {}
    score = defaultdict(float)
    for name, hits in channels:
        w = weights.get(name, 1.0)
        if w == 0: continue
        for rank, (did, _) in enumerate(hits):
            score[did] += w / (k + rank + 1)
    return score

def apply_neg_gate(doc_ids, doc_meta, noise_roles):
    keep = []
    for did in doc_ids:
        m = doc_meta.get(did) or {}
        if m.get("is_notification_paragraph"): continue
        pr = (m.get("paragraph_role") or "").lower()
        if pr in noise_roles: continue
        keep.append(did)
    return keep

def round_robin_guarantee(channels_by_name, guarantee_channel_names, per_channel_cap, total_cap):
    \"\"\"v7.1 NEW: round-robin merge so each guarantee channel gets a fair share
    even when sum(guarantees) > total_cap. Each channel contributes at most
    `per_channel_cap` items in round-robin order.\"\"\"
    iters = {cn: iter(channels_by_name.get(cn, [])) for cn in guarantee_channel_names}
    counts = {cn: 0 for cn in guarantee_channel_names}
    out = []; seen = set()
    while iters and len(out) < total_cap:
        exhausted = []
        for cn in list(iters.keys()):
            if counts[cn] >= per_channel_cap:
                exhausted.append(cn); continue
            try:
                did, _ = next(iters[cn])
                while did in seen:
                    did, _ = next(iters[cn])
                out.append(did); seen.add(did); counts[cn] += 1
                if len(out) >= total_cap: break
            except StopIteration:
                exhausted.append(cn)
        for cn in exhausted:
            if cn in iters: del iters[cn]
    return out

PER_QUERY = {}

for q in ALL_QUERIES:
    qid = q["query_id"]
    QUERY_LOCAL = q["query"]
    val_gold_q = q["gold"]
    targets_q = ALL_TARGETS.get(qid)
    if targets_q is None:
        print(f"[skip] {qid}: no targets")
        continue
    q_emb_raw_q      = ALL_Q_EMB_RAW[qid]
    q_emb_enriched_q = ALL_Q_EMB_ENRICHED[qid]

    # Map gold
    gold_doc_set_q = set()
    for g in val_gold_q:
        gold_doc_set_q.update(cit_to_doc_ids.get(g, []))
    total_gold_q = len(val_gold_q)

    # Canonicalize statutes
    llm_statute_canons = set()
    for raw in targets_q.get("statute_targets", []) or []:
        c = statute_anchor_canonical(raw)
        if c: llm_statute_canons.add(c)
    co_expanded_canons = set(llm_statute_canons)
    for canon in llm_statute_canons:
        for nb, _ in co_neighbours.get(canon, []):
            co_expanded_canons.add(nb)
    statute_target_codes = {c.split()[1] for c in llm_statute_canons if " " in c}

    # Concept expansion
    llm_concepts = (targets_q.get("concept_targets_en") or []) + (targets_q.get("legal_area_keywords") or [])
    expanded_concepts = expand_concepts_strict(
        llm_concepts, set(idx_concept_en.keys()), CONFIG["concept_substring_top_k"])

    # Run channels
    ch_law_direct = channel_law_direct(co_expanded_canons, idx_law_direct, CONFIG["budget_law_direct"])
    ch_court_stat = channel_court_statute(llm_statute_canons, idx_court_statute, CONFIG["budget_court_statute"])
    ch_concept    = channel_concept(expanded_concepts, idx_concept_en, CONFIG["budget_concept"])
    ch_term       = channel_term(targets_q, idx_term_orig, CONFIG["budget_term"])
    ch_per_area   = channel_per_area_bedrock(targets_q.get("legal_area_keywords", []),
                                              statute_target_codes, per_area_canon_count,
                                              idx_law_direct, CONFIG["budget_per_area"])
    ch_cocit      = channel_co_citation(targets_q, co_neighbours, idx_law_direct, idx_court_statute,
                                         CONFIG["budget_co_citation"])
    enrich_bits = ((targets_q.get("term_targets_de") or [])
                 + (targets_q.get("term_targets_fr") or [])
                 + (targets_q.get("concept_targets_en") or []))
    bm25_query_text_q = QUERY_LOCAL + " " + " ".join(enrich_bits)
    ch_bm25 = bm25_search(bm25_query_text_q, CONFIG["budget_bm25"])
    ch_vector  = channel_vector(q_emb_raw_q,      CONFIG["budget_vector"])           if VECTOR_OK else []
    ch_venrich = channel_vector(q_emb_enriched_q, CONFIG["budget_vector_enriched"])  if VECTOR_OK else []

    # Topical seed (no sibling/graph yet)
    seed = (
        _court_hits(ch_court_stat) | _court_hits(ch_law_direct)
      | _court_hits(ch_concept)    | _court_hits(ch_term)
      | _court_hits(ch_per_area)   | _court_hits(ch_cocit)
      | _court_hits(ch_bm25)
      | _court_hits(ch_vector)     | _court_hits(ch_venrich)
    )

    # Sibling first (now seed-frequency ranked thanks to v7.1 fix)
    ch_sibling = channel_sibling(seed, idx_court_base, doc_meta, CONFIG["budget_sibling"])

    # v7.1: extend seed with sibling outputs before graph traversal so 1-hop
    # graph from sibling-Es reaches procedural-cluster law articles.
    seed_extended = seed | _court_hits(ch_sibling)

    if GRAPH_OK:
        ch_graph_fwd = channel_graph_forward(seed_extended, idx_graph_out, CONFIG["budget_graph_forward"])
        ch_graph_rev = channel_graph_reverse(seed_extended, idx_graph_in,  CONFIG["budget_graph_reverse"])
        if CONFIG["enable_graph_2hop"]:
            ch_graph_2h = channel_graph_2hop(seed_extended, idx_graph_out, CONFIG["budget_graph_2hop"])
        else:
            ch_graph_2h = []
    else:
        ch_graph_fwd = []; ch_graph_rev = []; ch_graph_2h = []

    backprop_seed = seed | _court_hits(ch_sibling) | _court_hits(ch_graph_fwd) | _court_hits(ch_graph_rev)
    ch_backprop = channel_statute_backprop(backprop_seed, doc_statute_anchors,
                                            idx_law_direct, CONFIG["budget_backprop"])

    CHANNELS_q = [
        ("law_direct_match",  ch_law_direct),
        ("court_statute",     ch_court_stat),
        ("co_citation",       ch_cocit),
        ("per_area_bedrock",  ch_per_area),
        ("statute_backprop",  ch_backprop),
        ("sibling_expansion", ch_sibling),
        ("graph_forward",     ch_graph_fwd),
        ("graph_reverse",     ch_graph_rev),
        ("graph_2hop",        ch_graph_2h),
        ("concept_en",        ch_concept),
        ("term_orig",         ch_term),
        ("bm25",              ch_bm25),
        ("vector_raw",        ch_vector),
        ("vector_enriched",   ch_venrich),
    ]

    # RRF + round-robin guarantee + neg-gate
    rrf_scores_q = rrf_fuse(CHANNELS_q, CONFIG["rrf_k"],
                             weights=CONFIG.get("channel_weights"))
    ranked_q = sorted(rrf_scores_q.items(), key=lambda x: x[1], reverse=True)
    ranked_dids_q = [d for d, _ in ranked_q]
    ranked_gated_q = apply_neg_gate(ranked_dids_q, doc_meta, CONFIG["noise_paragraph_roles"])
    ch_by_name_q = dict(CHANNELS_q)
    guarantee_q = round_robin_guarantee(
        ch_by_name_q, CONFIG["guarantee_channels"],
        CONFIG.get("guarantee_per_channel", 200),
        CONFIG["topk_final"],
    )
    guarantee_q = apply_neg_gate(guarantee_q, doc_meta, CONFIG["noise_paragraph_roles"])
    final_topk_q = list(guarantee_q); seen_f = set(final_topk_q)
    for did in ranked_gated_q:
        if len(final_topk_q) >= CONFIG["topk_final"]: break
        if did not in seen_f:
            final_topk_q.append(did); seen_f.add(did)
    final_topk_q = final_topk_q[:CONFIG["topk_final"]]

    # Per-channel recall + union upper bound
    union_did_q = set()
    channel_recalls_q = {}
    for name, hits in CHANNELS_q:
        found = {d for d, _ in hits}
        union_did_q.update(found)
        channel_recalls_q[name] = (len(found & gold_doc_set_q) / max(1, total_gold_q)) if total_gold_q else 0.0
    R_at_K_q = (len(set(final_topk_q) & gold_doc_set_q) / max(1, total_gold_q)) if total_gold_q else 0.0

    PER_QUERY[qid] = {
        "query": QUERY_LOCAL,
        "gold": val_gold_q,
        "gold_doc_set": gold_doc_set_q,
        "total_gold": total_gold_q,
        "CHANNELS": CHANNELS_q,
        "channel_recalls": channel_recalls_q,
        "union_did": union_did_q,
        "union_recall": (len(union_did_q & gold_doc_set_q) / max(1, total_gold_q)) if total_gold_q else 0.0,
        "final_topk": final_topk_q,
        "R@1000": R_at_K_q,
        "rrf_scores": rrf_scores_q,
        "ranked_dids": ranked_dids_q,
        "guarantee": guarantee_q,
        "seed_extended": seed_extended,
    }

    print(f"\\n=== {qid} ({total_gold_q} gold) ===")
    print(f"  union_ceiling = {PER_QUERY[qid]['union_recall']:.3f}")
    print(f"  R@1000        = {R_at_K_q:.3f}")
    print(f"  per-channel recall:")
    for name, recall in channel_recalls_q.items():
        ch_size = len(dict(CHANNELS_q)[name])
        print(f"    {name:<22} size={ch_size:>5}  recall={recall:.3f}")

# val_001 legacy variables for the detailed Phase 11 diagnostic
_v1 = PER_QUERY["val_001"]
gold_doc_set = _v1["gold_doc_set"]; total_gold = _v1["total_gold"]
CHANNELS = _v1["CHANNELS"]; final_topk = _v1["final_topk"]
rrf_scores = _v1["rrf_scores"]; ranked_dids = _v1["ranked_dids"]
seed = _v1["seed_extended"]
val_gold = ALL_QUERIES[0]["gold"] if ALL_QUERIES else []
QUERY = ALL_QUERIES[0]["query"] if ALL_QUERIES else ""
bm25_query_text = (
    QUERY + " "
  + " ".join(
        (ALL_TARGETS["val_001"].get("term_targets_de") or [])
      + (ALL_TARGETS["val_001"].get("term_targets_fr") or [])
      + (ALL_TARGETS["val_001"].get("concept_targets_en") or [])
    )
)
""")


def phase_9_fusion(cells: list) -> None:
    add_md(cells, """
# Phase 10 — Aggregate stats across the 10 val queries

## 10.1 What this cell does

Aggregates `PER_QUERY` into:
- **Mean R@1000** across 10 queries — the true generalization metric.
- **Per-query R@1000 table** — shows which queries we hit and which fail.
- **Per-channel mean recall** — which channels carry their weight on
  average (across queries) vs. only on val_001.
- **Channel "ranking" by stable usefulness** — sorted by mean recall.
- **Union ceiling** — upper bound of any RRF strategy on this corpus.

## 10.2 Why this matters

v5/v6/v7 were single-query optimizations. v7 hit 0.452 on val_001 but
we don't know if that's representative. If mean R@1000 across 10 queries
is, say, 0.30, then the gains we measured on val_001 don't generalize and
we've been chasing val_001-specific patterns. If it's, say, 0.55+, the
graph + sibling + guarantee changes are real.

## 10.3 Expected outputs

- 10 rows of per-query R@1000.
- Channel mean-recall table (14 rows).
- One headline: `Mean R@1000 = X.XXX`.

## 10.4 Failure modes

- One query at R@1000 = 0.0 → that query's targets dict is empty (LLM
  failed for that prompt). Check `ALL_TARGETS[qid]` directly.
- All queries < 0.4 → graph wiring or budgets are wrong (not generalization).
""")
    add_code(cells, """
print("=" * 78)
print("AGGREGATE ACROSS 10 VAL QUERIES")
print("=" * 78)

# Per-query table
print(f"\\n{'query':<10}{'gold':>5}{'union_ceil':>12}{'R@1000':>10}")
print("-" * 40)
total_R = 0.0; total_gold_sum = 0; total_caught_sum = 0
for qid, r in PER_QUERY.items():
    caught = int(r["R@1000"] * r["total_gold"])
    total_R += r["R@1000"]
    total_gold_sum += r["total_gold"]
    total_caught_sum += caught
    print(f"{qid:<10}{r['total_gold']:>5}{r['union_recall']:>12.3f}{r['R@1000']:>10.3f}  ({caught}/{r['total_gold']})")

n_q = len(PER_QUERY)
mean_R       = total_R / max(1, n_q)
micro_R      = total_caught_sum / max(1, total_gold_sum)
mean_union   = sum(r["union_recall"] for r in PER_QUERY.values()) / max(1, n_q)
print()
print(f"  Macro mean R@1000:     {mean_R:.3f}   ({n_q} queries)")
print(f"  Micro R@1000:          {micro_R:.3f}   ({total_caught_sum}/{total_gold_sum} total gold)")
print(f"  Macro mean union ceil: {mean_union:.3f}")

# Per-channel mean recall (sorted)
print()
print("=" * 78)
print("PER-CHANNEL MEAN RECALL (averaged over 10 queries)")
print("=" * 78)
all_channel_names = [n for n, _ in PER_QUERY[next(iter(PER_QUERY))]["CHANNELS"]]
channel_mean = {}
for cn in all_channel_names:
    rs = [PER_QUERY[qid]["channel_recalls"].get(cn, 0.0) for qid in PER_QUERY]
    channel_mean[cn] = sum(rs) / max(1, len(rs))

print(f"\\n{'channel':<22}{'mean recall':>15}{'std':>10}")
print("-" * 50)
import statistics as _stat
ranked = sorted(channel_mean.items(), key=lambda x: -x[1])
for cn, m in ranked:
    rs = [PER_QUERY[qid]["channel_recalls"].get(cn, 0.0) for qid in PER_QUERY]
    sd = _stat.pstdev(rs) if len(rs) >= 2 else 0.0
    bar = "#" * int(round(m * 30))
    print(f"  {cn:<20}  {m:>10.3f}    {sd:>5.3f}  {bar}")

# Worst-performing query
print()
worst = min(PER_QUERY.items(), key=lambda kv: kv[1]["R@1000"])
best  = max(PER_QUERY.items(), key=lambda kv: kv[1]["R@1000"])
print(f"Worst query: {worst[0]} (R@1000 = {worst[1]['R@1000']:.3f}, "
      f"union_ceiling = {worst[1]['union_recall']:.3f})")
print(f"Best  query: {best[0]} (R@1000 = {best[1]['R@1000']:.3f}, "
      f"union_ceiling = {best[1]['union_recall']:.3f})")
""")


def phase_10_diagnosis(cells: list) -> None:
    add_md(cells, """
# Phase 11 — DIAGNOSIS

This phase runs in two modes:
- **11.1–11.3 detailed trace**: for **val_001 only** (the canary query).
  Per-gold root cause + per-channel signal explanation. ~1500 lines of
  output but tells you exactly which signal failed for which gold.
- **11.4 cross-query brief**: for every val query, a one-line summary of
  missed gold counts and a top-3 root-cause table. Tells you whether
  val_001 patterns generalize.

For each missed gold (val_001), we run a per-channel signal trace and
print a structured report:

1. **Source row inspection** — is this gold present as a `citation` column
   value in laws_de or court_considerations? If yes, we print:
   - The full `search_text` (BM25 input)
   - The enrichment fields (concepts_en, terms, statute_anchors)
   - The doc's paragraph_role, court_base, legal_area_static

2. **Per-channel hit/miss + reason**:
   - For each of the 14 channels, report:
     - `hit (rank N)` — gold is in channel's output at rank N
     - `miss (no signal)` — gold's doc_id never appeared as a candidate
     - `miss (truncated)` — gold appeared in raw scoring but got cut by budget
     - `miss (zero overlap)` — for token channels, no token-level overlap
   - For **bm25**: which query tokens (if any) appear in gold's `search_text`?
   - For **vector**: cosine similarity between query embedding and gold's
     row embedding (requires gold to be in the manifest).
   - For **concept_en / term_orig**: which expanded LLM concepts/terms map
     to this gold? If none, it's a coverage gap in query expansion.
   - For **graph_forward**: from each seed, list outgoing edges that target
     this gold. If empty, no caught seed text-cites this gold.
   - For **graph_reverse**: list incoming edges from this gold. If a source
     in the seed set, why wasn't its edge followed? (Should never happen.)
   - For **sibling_expansion**: gold's `court_base`. List other Es of the
     same judgment. Were any caught? If none caught, sibling can't fire.

3. **Recommended fix** — one of:
   - Enrichment gap: gold's `search_text` doesn't contain query keywords.
     Fix: extend law/court LLM enrichment prompt.
   - Query expansion gap: LLM didn't produce a relevant concept/term that
     matches this gold's enrichment. Fix: improve query-expansion prompt
     or use a stronger model.
   - Graph extraction gap: gold's referencing source row exists but no
     edge to it in the graph. Fix: add a regex pattern or run another
     extraction pass.
   - RRF rank gap: gold has hits in N channels but score is below cutoff.
     Fix: lift channel budget OR add channel to guarantee_channels.
   - True orphan: nothing in the corpus references this gold by exact
     pinpoint. Fix: case-level fan-out (already in graph) — verify it fired.

## 11.1 Helper functions for diagnosis
""")
    add_code(cells, """
# --- per-channel rank lookup helpers -----------------------------------------
def channel_rank(did, hits):
    \"\"\"Return rank (0-indexed) of did in hits, or None if absent.\"\"\"
    for i, (d, _) in enumerate(hits):
        if d == did: return i
    return None

def channel_score(did, hits):
    for d, s in hits:
        if d == did: return s
    return None

# --- token overlap for BM25 --------------------------------------------------
def _tokens(text):
    return set(t for t in re.split(r"[^\\w\\d]+", (text or "").lower(), flags=re.UNICODE)
               if len(t) >= CONFIG["bm25_min_token_len"])

query_tokens = _tokens(bm25_query_text)
print(f"Query token count (post-enhance): {len(query_tokens)}")

# --- vector cosine helper (requires manifest mapping) ------------------------
def gold_vector_cos(did, q_emb):
    if not VECTOR_OK: return None
    if did not in row_for_did: return None
    ridx = row_for_did[did]
    with torch.no_grad():
        v = E_GPU[ridx]
        v = v / (v.norm() + 1e-9)
        q = q_emb.to(E_GPU.device, dtype=E_GPU.dtype)
        q = q / (q.norm() + 1e-9)
        return float((v * q).sum().item())

import torch
# Compute gold vector cos for all gold (those in manifest)
gold_vec_cos_raw = {}
gold_vec_cos_enr = {}
if VECTOR_OK:
    for did in gold_doc_set:
        cr = gold_vector_cos(did, q_emb_raw)
        ce = gold_vector_cos(did, q_emb_enriched)
        if cr is not None: gold_vec_cos_raw[did] = cr
        if ce is not None: gold_vec_cos_enr[did] = ce

# --- which expanded concepts/terms point to a given did ----------------------
def concepts_pointing_to(did, expanded_concepts, idx):
    return [c for c in expanded_concepts if did in idx.get(c, ())]
def terms_pointing_to(did, targets, idx):
    out = []
    for key in ("term_targets_de", "term_targets_fr"):
        for raw in targets.get(key, []) or []:
            tok = norm_token(raw, CONFIG["lowercase_terms"])
            if tok and did in idx.get(tok, ()):
                out.append((key, raw, tok))
    return out

# --- find caught seeds with edges to a given did -----------------------------
def graph_paths_to(did, seed, idx_graph_in):
    \"\"\"Return list of seeds that have an outgoing edge to did (i.e., did is
    reachable forward from these seeds).\"\"\"
    incoming_to_did = set(idx_graph_in.get(did, ()))
    return [s for s in seed if s in incoming_to_did]

def graph_paths_from(did, seed, idx_graph_out):
    \"\"\"Return list of seeds that have an incoming edge from did (i.e., this
    did's outgoing edges include some seed = caught).\"\"\"
    outgoing_from_did = set(idx_graph_out.get(did, ()))
    return [s for s in seed if s in outgoing_from_did]

# --- court_base sibling diagnosis --------------------------------------------
def sibling_diagnosis(did, doc_meta, idx_court_base, seed):
    m = doc_meta.get(did) or {}
    cb = m.get("court_base")
    if not cb: return None, [], []
    siblings = idx_court_base.get(cb, set()) - {did}
    caught_siblings = [s for s in siblings if s in seed]
    return cb, list(siblings), caught_siblings
""")

    add_md(cells, """
## 11.2 Per-gold trace (the main diagnostic loop)

For each gold not in top-1000, prints a structured report. If a gold IS in
top-1000, we still record at what rank for the summary table.
""")
    add_code(cells, """
top1000_set = set(final_topk)
gold_in_top = gold_doc_set & top1000_set
gold_missed = gold_doc_set - top1000_set

# Reverse map: doc_id -> citation
did_to_cit = {did: m["citation"] for did, m in doc_meta.items()}

print(f"R@1000 = {len(gold_in_top)}/{total_gold} = {100*len(gold_in_top)/total_gold:.1f}%")
print(f"Missed gold: {len(gold_missed)} of {total_gold}")
print()
print("=" * 80)
print("PER-GOLD MISS DIAGNOSIS")
print("=" * 80)

miss_summary = []  # (gold_cit, root_cause, fix_suggestion)
for did in sorted(gold_missed):
    cit = did_to_cit.get(did, did)
    m = doc_meta.get(did, {})
    fam = m.get("family")
    cb = m.get("court_base")
    pr = m.get("paragraph_role")
    print()
    print("-" * 80)
    print(f"GOLD: {cit}")
    print(f"  doc_id={did}  family={fam}  court_base={cb}  paragraph_role={pr}")

    # --- Source row inspection ---
    st = search_text.get(did, "")
    print(f"  search_text ({len(st)} chars): {st[:240]}{'...' if len(st)>240 else ''}")

    # --- Per-channel rank ---
    print("  Per-channel:")
    channel_hit_count = 0
    channel_miss_reasons = []
    for cname, hits in CHANNELS:
        r = channel_rank(did, hits)
        if r is not None:
            print(f"    HIT  {cname:<22} rank={r}  (size={len(hits)})")
            channel_hit_count += 1
        else:
            # Probe why miss
            reason = None
            if cname == "bm25":
                doc_tokens = _tokens(st)
                overlap = query_tokens & doc_tokens
                reason = f"token overlap with query: {len(overlap)} ({sorted(overlap)[:6]})"
            elif cname == "vector_raw" and VECTOR_OK:
                cr = gold_vec_cos_raw.get(did)
                reason = f"cos_sim_raw={cr:.4f}" if cr is not None else "not in manifest"
            elif cname == "vector_enriched" and VECTOR_OK:
                ce = gold_vec_cos_enr.get(did)
                reason = f"cos_sim_enriched={ce:.4f}" if ce is not None else "not in manifest"
            elif cname == "concept_en":
                ptrs = concepts_pointing_to(did, expanded_concepts, idx_concept_en)
                reason = f"expanded concepts hitting this row: {ptrs[:5]}" if ptrs else "no concept hits this row"
            elif cname == "term_orig":
                tptrs = terms_pointing_to(did, targets, idx_term_orig)
                reason = f"terms hitting this row: {tptrs[:3]}" if tptrs else "no term hits this row"
            elif cname == "graph_forward":
                paths = graph_paths_to(did, seed, idx_graph_in)
                reason = f"caught seeds with edges -> {len(paths)}: {paths[:3]}"
            elif cname == "graph_reverse":
                paths = graph_paths_from(did, seed, idx_graph_out)
                reason = f"caught seeds reachable from this -> {len(paths)}: {paths[:3]}"
            elif cname == "sibling_expansion":
                cb_, sibs, caught_sibs = sibling_diagnosis(did, doc_meta, idx_court_base, seed)
                reason = f"court_base={cb_} | sibs={len(sibs)} | caught_sibs={len(caught_sibs)}"
            elif cname == "law_direct_match":
                canon = statute_anchor_canonical(cit)
                reason = f"canonical={canon} | in_canon_set={canon in co_expanded_canons}"
            elif cname == "court_statute":
                row_canons = doc_statute_anchors.get(did, set())
                hits_named = row_canons & llm_statute_canons
                reason = f"row_anchors={list(row_canons)[:3]} | overlap with LLM names={list(hits_named)}"
            elif cname == "statute_backprop":
                row_canons = doc_statute_anchors.get(did, set())
                # for law gold, count caught court rows whose anchors include this canon
                gcanon = statute_anchor_canonical(cit) if fam == "law" else None
                if gcanon:
                    n_courts = sum(1 for s in seed if gcanon in doc_statute_anchors.get(s, set()))
                    reason = f"law canonical={gcanon} | seed-courts-citing={n_courts}"
                else:
                    reason = f"family={fam}, statute_backprop only emits law rows"
            elif cname == "co_citation":
                gcanon = statute_anchor_canonical(cit) if fam == "law" else None
                ndid_in_neighbours = False
                for raw in targets.get("statute_targets", []) or []:
                    tcanon = statute_anchor_canonical(raw)
                    if not tcanon: continue
                    for nb, _ in co_neighbours.get(tcanon, []):
                        if did in idx_law_direct.get(nb, set()) or did in idx_court_statute.get(nb, set()):
                            ndid_in_neighbours = True; break
                    if ndid_in_neighbours: break
                reason = f"reachable via co-cited neighbour={ndid_in_neighbours}"
            elif cname == "per_area_bedrock":
                gcanon = statute_anchor_canonical(cit) if fam == "law" else None
                in_pab = False
                if gcanon:
                    for area, cnts in per_area_canon_count.items():
                        if any(k.lower() in area for k in (targets.get("legal_area_keywords") or [])):
                            if gcanon in dict(cnts.most_common(CONFIG["per_area_top_n"])):
                                in_pab = True; break
                reason = f"law canonical={gcanon} | in selected-area top-{CONFIG['per_area_top_n']}={in_pab}"
            elif cname == "graph_2hop":
                reason = "disabled by config" if not CONFIG["enable_graph_2hop"] else "miss"
            else:
                reason = "miss"
            print(f"    MISS {cname:<22} ({reason})")
            channel_miss_reasons.append((cname, reason))

    # --- RRF rank if any score ---
    rrf_s = rrf_scores.get(did, 0.0)
    rrf_pos = ranked_dids.index(did) if did in rrf_scores else None
    print(f"  RRF: score={rrf_s:.5f}  rank={rrf_pos}  (top-1000 cutoff is rank 999)")

    # --- root cause heuristic ---
    if channel_hit_count == 0:
        cause = "NO_CHANNEL_HIT"
        fix = "no channel produced this as a candidate; check enrichment, regex, graph"
    elif rrf_s == 0:
        cause = "RRF_ZERO"
        fix = "channel sizes vs guarantees mismatch (should not happen)"
    elif rrf_pos is not None and rrf_pos < 1000 and did not in top1000_set:
        cause = "GATED_OUT"
        fix = f"negative gate dropped this; paragraph_role={pr}"
    elif rrf_pos is not None and rrf_pos >= 1000:
        cause = "RRF_RANK_TOO_LOW"
        fix = f"in {channel_hit_count} channel(s) but RRF rank {rrf_pos}; consider promoting a channel to guarantee or lifting budget"
    else:
        cause = "UNKNOWN"
        fix = "investigate above trace"

    print(f"  ROOT CAUSE: {cause}")
    print(f"  FIX SUGGESTION: {fix}")
    miss_summary.append((cit, cause, channel_hit_count, fix))
""")

    add_md(cells, """
## 11.3 Failure-mode summary (aggregated)

Aggregate the per-gold root causes into buckets so we know where to invest
the next iteration's effort.
""")
    add_code(cells, """
print("=" * 80)
print("FAILURE-MODE SUMMARY")
print("=" * 80)

from collections import Counter as _Counter
cause_counter = _Counter(c for _, c, _, _ in miss_summary)
print(f"Missed gold: {len(miss_summary)}")
print(f"  by root cause:")
for cause, n in cause_counter.most_common():
    print(f"    {cause:<22}  {n}")

# Top fixes
print()
print("Top 10 missed gold with their fix suggestions:")
for cit, cause, n_hits, fix in miss_summary[:10]:
    print(f"  - {cit:<35}  cause={cause:<22}  channels_hit={n_hits}  fix={fix}")

# Quick yes/no diagnostic table
print()
print("=" * 80)
print("CHANNEL-LEVEL HIT RATE ON val_001 MISSED GOLD")
print("=" * 80)
print("(a channel that 'caught' missed gold means the gold was in its candidate")
print(" pool but ranked below top-1000 after fusion)")
print()
print(f"{'channel':<22}  caught_missed_gold")
print("-" * 50)
for cname, hits in CHANNELS:
    hit_set = {d for d, _ in hits}
    n_caught = len(hit_set & gold_missed)
    print(f"  {cname:<20}  {n_caught}")

# === 11.4 Cross-query brief miss summary ===================================
print()
print("=" * 80)
print("11.4  CROSS-QUERY MISS SUMMARY (one row per val query)")
print("=" * 80)
print(f"{'qid':<10}{'gold':>5}{'caught':>8}{'missed':>8}{'union':>10}  best-channel-for-query")
print("-" * 80)
for qid, r in PER_QUERY.items():
    caught = int(r["R@1000"] * r["total_gold"])
    missed = r["total_gold"] - caught
    best_ch = max(r["channel_recalls"].items(), key=lambda kv: kv[1])
    print(f"{qid:<10}{r['total_gold']:>5}{caught:>8}{missed:>8}{r['union_recall']:>10.3f}  "
          f"{best_ch[0]} (recall {best_ch[1]:.3f})")

# Show RRF/budget-waste gap per query (gold in union but not final)
print()
print("Gold in candidate union vs final top-1000 (gap = lost to RRF/gating):")
print(f"{'qid':<10}{'in_union':>10}{'in_final':>10}{'gap':>8}")
print("-" * 40)
for qid, r in PER_QUERY.items():
    union_caught = len(r["union_did"] & r["gold_doc_set"])
    final_caught = int(r["R@1000"] * r["total_gold"])
    print(f"{qid:<10}{union_caught:>10}{final_caught:>10}{(union_caught-final_caught):>8}")
""")


def phase_11_save(cells: list) -> None:
    add_md(cells, """
# Phase 12 — Save artifacts + cleanup

## 12.1 What we save

- `final_topk.json` — list of (citation, doc_id, rank) for top-1000.
- `gold_in_top.json` — gold citations captured.
- `gold_missed_diagnosis.json` — per-gold root-cause + fix suggestion.
- `targets.json` — LLM query expansion output (for repro).
- `config.json` — exact CONFIG used.
- `summary.json` — R@K numbers + channel sizes.

These persist in `out_dir` so a future run can compare.
""")
    add_code(cells, """
import json as _json
out_dir = PATHS["out_dir"]
out_dir.mkdir(parents=True, exist_ok=True)

# --- Per-query artifacts (one folder per query_id) -------------------------
for qid, r in PER_QUERY.items():
    qdir = out_dir / qid
    qdir.mkdir(parents=True, exist_ok=True)

    final_records = [
        {"rank": i, "doc_id": did, "citation": did_to_cit.get(did, did)}
        for i, did in enumerate(r["final_topk"])
    ]
    (qdir / "final_topk.json").write_text(
        _json.dumps(final_records, ensure_ascii=False, indent=2), encoding="utf-8")

    gold_in_top_q = set(r["final_topk"]) & r["gold_doc_set"]
    gold_top_records = sorted(
        [{"doc_id": d, "citation": did_to_cit.get(d, d),
          "rank": r["final_topk"].index(d) if d in set(r["final_topk"]) else None}
         for d in gold_in_top_q],
        key=lambda x: x["rank"] if x["rank"] is not None else 1e9,
    )
    (qdir / "gold_in_top.json").write_text(
        _json.dumps(gold_top_records, ensure_ascii=False, indent=2), encoding="utf-8")

    (qdir / "targets.json").write_text(
        _json.dumps(ALL_TARGETS.get(qid, {}), ensure_ascii=False, indent=2), encoding="utf-8")

    qsummary = {
        "query_id": qid,
        "query": r["query"][:500],
        "total_gold": r["total_gold"],
        "R_at_1000": r["R@1000"],
        "union_upper_bound": r["union_recall"],
        "channel_sizes":   {n: len(h) for n, h in r["CHANNELS"]},
        "channel_recalls": r["channel_recalls"],
    }
    (qdir / "summary.json").write_text(
        _json.dumps(qsummary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

# --- Aggregate cross-query summary (top of out_dir) ------------------------
all_channel_names = [n for n, _ in PER_QUERY[next(iter(PER_QUERY))]["CHANNELS"]]
mean_R = sum(r["R@1000"] for r in PER_QUERY.values()) / max(1, len(PER_QUERY))
mean_union = sum(r["union_recall"] for r in PER_QUERY.values()) / max(1, len(PER_QUERY))
total_gold_sum = sum(r["total_gold"] for r in PER_QUERY.values())
total_caught_sum = sum(int(r["R@1000"] * r["total_gold"]) for r in PER_QUERY.values())

aggregate = {
    "n_queries": len(PER_QUERY),
    "mean_R_at_1000_macro": mean_R,
    "R_at_1000_micro": total_caught_sum / max(1, total_gold_sum),
    "mean_union_upper_bound": mean_union,
    "per_query": {
        qid: {"R@1000": r["R@1000"], "total_gold": r["total_gold"],
              "union": r["union_recall"]}
        for qid, r in PER_QUERY.items()
    },
    "channel_mean_recall": {
        cn: sum(PER_QUERY[qid]["channel_recalls"].get(cn, 0.0) for qid in PER_QUERY)
            / max(1, len(PER_QUERY))
        for cn in all_channel_names
    },
}
(out_dir / "aggregate_summary.json").write_text(
    _json.dumps(aggregate, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

# Detailed val_001 diagnostic (the one Phase 11 produced)
miss_records = [
    {"citation": cit, "root_cause": cause, "channels_hit": n_hits, "fix_suggestion": fix}
    for cit, cause, n_hits, fix in miss_summary
]
(out_dir / "val_001_gold_missed_diagnosis.json").write_text(
    _json.dumps(miss_records, ensure_ascii=False, indent=2), encoding="utf-8")

(out_dir / "config.json").write_text(_json.dumps(
    {k: v for k, v in CONFIG.items() if not isinstance(v, set)},
    ensure_ascii=False, indent=2, default=str), encoding="utf-8")

print(f"Saved per-query results in {out_dir}/<qid>/  (10 query folders)")
print(f"Aggregate summary at {out_dir}/aggregate_summary.json")
print(f"  mean R@1000 (macro): {mean_R:.3f}")
print(f"  R@1000        (micro): {total_caught_sum / max(1, total_gold_sum):.3f}")
""")

    add_md(cells, """
## 12.2 Cleanup (free GPU memory)

**What:** Delete large GPU tensors so a notebook re-run starts clean.

**Why:** Colab keeps state across cells; without explicit cleanup, re-running
Phase 6 will OOM.
""")
    add_code(cells, """
import gc
try:
    del E_GPU
except NameError:
    pass
try:
    del EMB_MODEL
except NameError:
    pass
gc.collect()
import torch as _torch2
if _torch2.cuda.is_available():
    _torch2.cuda.empty_cache()
    print(f"VRAM after cleanup: {_torch2.cuda.memory_allocated()/1024**3:.2f} GB")
print("cleaned up.")
""")


# ---------------------------------------------------------------------------
# Build notebook
# ---------------------------------------------------------------------------

def main() -> int:
    cells: list = []
    phase_intro(cells)
    phase_1_setup(cells)
    phase_2_index_build(cells)
    phase_3_graph(cells)
    phase_4_bedrock_cocit(cells)
    phase_5_bm25(cells)
    phase_6_query_expansion(cells)   # Qwen3-32B: load → use → free (peak ~65 GB)
    phase_7_encode_queries(cells)    # Qwen3-Embedding-8B: load → encode → free (peak ~16 GB)
    phase_8_vector_setup(cells)      # E_GPU: ~22 GB sustained for the rest
    phase_8_channels(cells)          # markdown header is "Phase 9 — Channels"
    phase_9_fusion(cells)            # markdown header is "Phase 10 — RRF fusion"
    phase_10_diagnosis(cells)        # markdown header is "Phase 11 — Diagnosis"
    phase_11_save(cells)             # markdown header is "Phase 12 — Save artifacts"

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    V7.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote v7 -> {V7}")
    print(f"  cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='markdown')} markdown, "
          f"{sum(1 for c in cells if c['cell_type']=='code')} code)")
    print(f"  size: {V7.stat().st_size:,} bytes")

    # Syntax check
    errors = 0
    for i, c in enumerate(cells):
        if c["cell_type"] != "code": continue
        try:
            compile("".join(c["source"]), f"<cell {i}>", "exec")
        except SyntaxError as e:
            errors += 1
            print(f"  [{i}] SyntaxError: line {e.lineno}: {e.msg}")
    print(f"  syntax: {'PASS' if errors == 0 else f'{errors} ERRORS'}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
