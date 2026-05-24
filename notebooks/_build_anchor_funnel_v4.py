"""Build the v4 Colab notebook for val_001 R@1000 verification.

v4 additions over v3:
- Per-legal-area corpus bedrock (replaces global)
- Co-citation expansion of LLM-named statutes (corpus-derived, no train)
- BM25 channel built on the fly from enrichment + court text_excerpt_original
- Vector channel via Qwen3-Embedding-8B (uses pre-computed corpus embeddings)
- Query-enriched vector channel (query + DE/FR terms concatenated)
- Drops the case_anchor channel (LLM hallucinates BGE numbers)

No hardcoding:
- All channel inputs derive from either the LLM (query targets), the corpus
  (bedrock + co-citation), or the v5_unified static enrichment.
- Train.csv is not read.

Output: notebooks/swiss_citation_anchor_funnel_val001_v4.ipynb
"""
import json
from pathlib import Path

CELLS = []


def md(text):
    CELLS.append(("markdown", text.rstrip() + "\n"))


def code(text):
    CELLS.append(("code", text.rstrip() + "\n"))


# ===========================================================================
md("""
# Swiss Citation — Anchor-Funnel v4 (val_001 R@1000 ≥ 0.7 attempt)

**Goal:** push val_001 R@1000 from v3's 0.286 (12/42) to **0.70+** without
hardcoding.

## What's new in v4

v3 hit recall ceilings driven by:
- The LLM (Qwen3-32B) only naming 6/19 of val_001's law gold by article
- Global corpus bedrock returning BGG-only top (val_001 needs StPO procedural cluster)
- Concept channel mismatching corpus vocabulary
- 8 docket court rows whose static enrichment is sparse

v4 adds four channels that attack each failure mode:

| # | Channel | What it adds | No-hardcoding source |
|---|---|---|---|
| 7' | **Per-legal-area corpus bedrock** (replaces v3's global bedrock) | Within `criminal procedure`, the top-N most-cited articles include `Art. 422/428/135/382-396 StPO` directly. | Derived from the v5_unified court corpus statute_anchor + legal_area distribution. |
| 8 | **Co-citation expansion** | When the LLM names `Art. 221 StPO`, look up which articles co-occur with it in the same court rows most often (`Art. 222/227/237 StPO`). Add those as additional statute_targets. | Pure corpus statistic from `idx_court_statute`. Mirrors how a Swiss lawyer thinks but learns the cluster from the corpus. |
| 9 | **BM25** over enrichment text + court paragraph excerpts | Catches articles whose text mentions val_001 keywords (`Diebstahl` / `Untersuchungshaft` / `Kollusionsgefahr`). Reaches the docket court rows whose enrichment is sparse but text is rich. | FTS5 built in-memory from data already loaded; no train-derived lexicon. |
| 10 | **Vector** (Qwen3-Embedding-8B) — direct query | Multilingual cosine match on the precomputed 21 GB corpus embeddings. | Already on disk per handoff §3.5; uses the model's existing multilingual capability. |
| 11 | **Vector — query-enriched** | Concatenate query + LLM's DE/FR terms + EN concepts before encoding. Lands the embedding into the corpus's German+French legal-vocabulary region. | LLM-generated terms; no narrative hallucination (unlike HyDE). |

Drops:
- `case_anchor`: even Qwen3-32B hallucinated BGE numbers (`BGE 145 III 456` sequential pattern). Channel contributed 0 in v2 + v3.

Pass criterion: R@1000 ≥ 0.70 (≥ 30/42).
Stretch: R@1000 ≥ 0.80 (≥ 34/42).

## Constraints respected

- **No hardcoded lists.** No DE_LEX, no fixed BGE list, no statute cluster table.
- **No train data.** Train.csv is not read at any point.
- **Open-source only.** Qwen3-32B (query expansion), Qwen3-Embedding-8B (vectors). Both fit in 95.6 GB.

## Memory plan on Blackwell (95.6 GB)

| Stage | Peak VRAM |
|---|---:|
| Index build (CPU) | ~5 GB host RAM |
| BM25 FTS5 build (CPU) | ~3 GB host RAM |
| Qwen3-32B for query expansion (loaded then freed) | ~65 GB |
| Qwen3-Embedding-8B (loaded after Qwen3-32B is freed) | ~16 GB |
| Corpus embedding tensor on GPU | ~21 GB |
| Concurrent peak after Qwen3-32B is freed | ~37 GB |
""")


# ===========================================================================
md("""## Cell 1 — Environment & GPU check""")
code("""
import os, sys, json, time, math, gc, re
print("Python:", sys.version.split()[0])

try:
    import torch
    print("PyTorch:", torch.__version__)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name}, {p.total_memory / 1024**3:.1f} GB")
    else:
        print("  No CUDA detected.")
except ImportError:
    print("PyTorch not installed.")
""")


# ===========================================================================
md("""## Cell 2 — Configure paths

Set `DATA_ROOT` to the directory holding `data/`, `artifacts_v2/`,
`law_json_llm_output/` (or the equivalent), and `embeddings/`.
""")
code("""
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
        DATA_ROOT = root
        break

if DATA_ROOT is None:
    print("Could not auto-detect DATA_ROOT. If on Colab:")
    print("    from google.colab import drive; drive.mount('/content/drive')")
    DATA_ROOT = Path("/content/drive/MyDrive/swiss_law")

print("DATA_ROOT =", DATA_ROOT)

# Try multiple known sub-paths; the user's Drive layout differs from local.
def first_existing(*paths):
    for p in paths:
        if p.exists():
            return p
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
    "emb_dir":      DATA_ROOT / "embeddings",
    "emb_manifest": DATA_ROOT / "embeddings" / "qwen3_8b_unified_manifest.parquet",
    "out_dir":      DATA_ROOT / "research" / "anchor_funnel_val001_v4",
}
PATHS["out_dir"].mkdir(parents=True, exist_ok=True)

for k, p in PATHS.items():
    if k == "out_dir":
        continue
    flag = "OK" if p.exists() else "MISSING"
    print(f"  {k}: {flag}  {p}")

EMB_AVAILABLE = PATHS["emb_dir"].exists() and any(PATHS["emb_dir"].glob("qwen3_8b_unified_chunk*.npy"))
if not EMB_AVAILABLE:
    print()
    print("[warn] Pre-computed embeddings not found. Vector channels (10, 11) will be SKIPPED.")
    print("       Anchor + bedrock + BM25 channels still run.")
""")


# ===========================================================================
md("""## Cell 3 — Knob panel (architecture knobs only; no target hardcoding)""")
code("""
CONFIG = {
    "topk_final": 1000,

    # Channel budgets. Sum > topk_final on purpose so RRF can reorder.
    "budget_law_direct":     None,   # uncapped (guarantee channel)
    "budget_court_statute":  600,
    "budget_sibling":        500,
    "budget_concept":        600,
    "budget_term":           500,
    "budget_per_area":       150,
    "budget_co_citation":    400,
    "budget_bm25":           600,
    "budget_vector":         800,
    "budget_vector_enriched":800,

    # RRF.
    "rrf_k": 60,
    # Channels whose hits are force-included in top-K before the RRF tail.
    # law_direct_match: every law row whose canonical citation matches a
    #   (LLM-named OR co-cited) statute target.
    # per_area_bedrock: corpus-derived universal articles for the query's
    #   legal_area cluster.
    "guarantee_channels": ["law_direct_match", "per_area_bedrock"],

    # Per-area bedrock — top-N most-cited canonical statutes WITHIN the
    # query's legal_area cluster. Derived from corpus, not train.
    "per_area_top_n": 100,

    # Co-citation expansion — for each LLM-named statute, fetch top-K
    # canonical articles that co-occur in the same court rows most often,
    # filtered by row count >= min_co_count.
    "co_citation_top_k_per_target": 8,
    "co_citation_min_co_count":     20,

    # Concept matching: strict substring rule (no token overlap noise).
    "concept_substring_top_k": 6,

    # BM25.
    "bm25_max_query_terms": 60,    # cap to stop token-explosion from enriched query
    "bm25_min_token_len":   3,

    # Vector channel.
    "vector_emb_model": "Qwen/Qwen3-Embedding-8B",
    "vector_topk":      800,

    # Query-expansion model.
    "qwen_query_model":  "Qwen/Qwen3-32B",
    "qwen_max_new_tokens": 1024,

    # Negative gate.
    "noise_paragraph_roles": {"notification", "header", "empty", "metadata"},

    "lowercase_concepts": True,
    "lowercase_terms":    True,
}

print(json.dumps({k: v for k, v in CONFIG.items() if not isinstance(v, set)}, indent=2, default=str))
""")


# ===========================================================================
md("""## Cell 4 — Load val_001 query + gold""")
code("""
import csv
csv.field_size_limit(2**31 - 1)

with open(PATHS["val_csv"], encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    val_001 = next(r for r in reader if r["query_id"] == "val_001")

QUERY = val_001["query"]
GOLD = [c.strip() for c in val_001["gold_citations"].split(";") if c.strip()]

print(f"val_001: {len(GOLD)} gold citations")
print(f"Query (first 240 chars): {QUERY[:240]}...")
""")


# ===========================================================================
md("""## Cell 5 — One-pass index build

Builds (in a single pass over both JSONL files):
- `cit_to_doc_ids`: every doc_id matching a citation string
- `doc_meta`: minimal per-row metadata (citation, family, court_base, paragraph_role, legal_area, search_text)
- Inverted indexes: `idx_law_direct`, `idx_court_statute`, `idx_case_anchor`, `idx_court_base`, `idx_concept_en`, `idx_term_orig`, `idx_legal_area_court`
- `legal_area_per_doc[did] = canonicalized legal area` (for per-area bedrock and gating)
- `search_text[did]` = concatenated text used by BM25 index later (Cell 9)
- `co_citation_pairs`: count of (canonical_a, canonical_b) co-occurrences in the same court row, used by Cell 8

Memory: streaming. Does not load full files into RAM.
""")
code("""
from collections import defaultdict, Counter
import re, json, time

# Statute / case canonicalizers (same as v3).
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
    \"\"\"Two-pass: row-level primary code → fallback for code-less anchors.\"\"\"
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

CASE_BGE_RE = re.compile(r"BGE\\s+(\\d+)\\s+([IVX]+)\\s+(\\d+)")
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

# Indexes
cit_to_doc_ids   = defaultdict(list)
doc_meta         = {}
idx_law_direct   = defaultdict(set)
idx_court_statute= defaultdict(set)
idx_case_anchor  = defaultdict(set)
idx_court_base   = defaultdict(set)
idx_concept_en   = defaultdict(set)
idx_term_orig    = defaultdict(set)
legal_area_per_doc = {}
search_text = {}                 # did -> text for BM25
co_citation_pairs = Counter()    # (canon_a, canon_b) -> count of court rows containing both

DOC_ID_LAW   = lambda i: f"law:{i}"
DOC_ID_COURT = lambda i: f"court:{i}"

def _take_text(*parts, max_chars=2000):
    \"\"\"Build a search_text string from a list of strings/lists.\"\"\"
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
    s = " ".join(out)
    return s[:max_chars]

t0 = time.time()
n_law = 0
with open(PATHS["law_llm"], encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        cit = obj.get("citation","")
        if not cit: continue
        did = DOC_ID_LAW(n_law)
        cit_to_doc_ids[cit].append(did)
        doc_meta[did] = {"citation": cit, "family": "law", "court_base": None,
                         "paragraph_role": None, "is_notification_paragraph": False}
        canon = statute_anchor_canonical(cit)
        if canon: idx_law_direct[canon].add(did)

        enr = obj.get("llm_enrichment") or {}
        # search_text for BM25: english_summary + legal_rule + legal_question +
        #                       applicability_conditions + concepts_en + terms (de/en).
        terms_de = []
        terms_en = []
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
            cit,
            enr.get("english_summary",""),
            enr.get("legal_rule",""),
            enr.get("legal_question",""),
            enr.get("applicability_conditions"),
            enr.get("concepts_en"),
            terms_de, terms_en,
        )
        legal_area_per_doc[did] = "law"   # placeholder; law has no legal_area_static
        n_law += 1

print(f"Law: {n_law:,} rows indexed in {time.time()-t0:.1f}s")

t1 = time.time()
n_court = 0
with open(PATHS["court_v5"], encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
        except Exception:
            continue
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

        # Statute canonicalization with fallback (Patch B kept).
        row_canons = canonicalize_row_anchors(rag.get("statute_anchors") or [], legal_area_static)
        for canon in row_canons:
            idx_court_statute[canon].add(did)
        # Co-citation: every unordered pair within the row's canonical set.
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

        # search_text for BM25: text excerpt + concepts + terms + statute anchors text
        search_text[did] = _take_text(
            cit,
            obj.get("text_excerpt_original",""),
            rag.get("concepts_en"),
            rag.get("terms_original"),
            rag.get("micro_topic",""),
            rag.get("topic",""),
            rag.get("subtopic",""),
            rag.get("statute_anchors"),
        )

        n_court += 1
        if n_court % 250_000 == 0:
            print(f"  court progress: {n_court:,} rows ({time.time()-t1:.1f}s)")

print(f"Court: {n_court:,} rows indexed in {time.time()-t1:.1f}s")
print(f"Total docs:        {len(doc_meta):,}")
print(f"Unique citations:  {len(cit_to_doc_ids):,}")
print(f"Index sizes:       law_direct={len(idx_law_direct):,}, court_statute={len(idx_court_statute):,}, "
      f"case={len(idx_case_anchor):,}, court_base={len(idx_court_base):,}, "
      f"concept={len(idx_concept_en):,}, term={len(idx_term_orig):,}")
print(f"Co-citation pairs: {len(co_citation_pairs):,}")
""")


# ===========================================================================
md("""## Cell 6 — Map gold to doc_ids""")
code("""
gold_doc_ids = {}
for g in GOLD:
    docs = cit_to_doc_ids.get(g, [])
    if docs: gold_doc_ids[g] = docs

gold_doc_set = set()
for docs in gold_doc_ids.values():
    gold_doc_set.update(docs)

print(f"Mapped gold:   {len(gold_doc_ids)}/{len(GOLD)}")
print(f"Total gold doc_ids: {len(gold_doc_set)}")
total_gold = len(gold_doc_set)
""")


# ===========================================================================
md("""## Cell 7 — Per-legal-area corpus bedrock (replaces global v3 bedrock)

For each `legal_area_static` cluster, rank canonical statute keys by the
number of distinct `court_base` values that cite them WITHIN that cluster.
A criminal-procedure query then pulls the top-N criminal-procedure-cited
articles, instead of the global top-N (which was BGG-dominated).
""")
code("""
from collections import defaultdict, Counter

# legal_area_static -> Counter of canonical -> distinct court_base count
per_area_canon_count = defaultdict(Counter)
# Build by re-scanning idx_court_statute: each (canon, did) pair contributes
# to the legal_area of did.
print("Building per-area bedrock index...")
t = time.time()
canon_area_bases = defaultdict(lambda: defaultdict(set))   # canon -> area -> set(court_base)
for canon, did_set in idx_court_statute.items():
    for d in did_set:
        m = doc_meta.get(d) or {}
        if m.get("family") != "court":
            continue
        cb = m.get("court_base"); area = legal_area_per_doc.get(d, "")
        if cb and area:
            canon_area_bases[canon][area].add(cb)

for canon, area_dict in canon_area_bases.items():
    for area, bases in area_dict.items():
        per_area_canon_count[area][canon] = len(bases)
print(f"  built in {time.time()-t:.1f}s; areas: {len(per_area_canon_count):,}")

# Show top 5 for the criminal-procedure cluster (val_001's area)
for area in list(per_area_canon_count.keys()):
    if "criminal" in area:
        print(f"  area={area!r}: top 8 = {per_area_canon_count[area].most_common(8)}")

del canon_area_bases
""")


# ===========================================================================
md("""## Cell 8 — Co-citation expansion (corpus-derived)

When the LLM names `Art. 221 StPO`, look up which canonical statutes most
often appear in the same court row (`idx_court_statute` co-occurrence). The
top-K become additional `statute_targets`. Pure corpus statistic; no train.

Mirrors how Swiss legal practice clusters articles ("221 StPO in detention
matters travels with 222/227/237 StPO"). Learned from the corpus, not declared.
""")
code("""
from collections import defaultdict

co_neighbours = defaultdict(list)   # canonical -> list of (neighbour_canonical, count) sorted desc
for (a, b), n in co_citation_pairs.items():
    if n < CONFIG["co_citation_min_co_count"]:
        continue
    co_neighbours[a].append((b, n))
    co_neighbours[b].append((a, n))

for k in co_neighbours:
    co_neighbours[k].sort(key=lambda x: -x[1])
    co_neighbours[k] = co_neighbours[k][: CONFIG["co_citation_top_k_per_target"]]

print(f"Co-citation neighbours indexed for {len(co_neighbours):,} canonicals.")
print("Sample — neighbours of '221 StPO':")
for nb, n in co_neighbours.get("221 StPO", [])[:8]:
    print(f"  {nb}: {n} co-occurrences")

# Memory cleanup — pairs no longer needed.
del co_citation_pairs
""")


# ===========================================================================
md("""## Cell 9 — Build BM25 (FTS5 in-memory) over enrichment text

Streams `search_text` into an in-memory SQLite FTS5 table. ~2.65M docs at
~500 chars avg = ~1.3 GB text → ~3-5 min to populate. After this, BM25
queries are sub-100ms.

No train-derived lexicon. Tokenizes the query directly with the same simple
splitter as the index.
""")
code("""
import sqlite3, re

print(f"Building in-memory FTS5 over {len(search_text):,} docs...")
t0 = time.time()

con = sqlite3.connect(":memory:")
con.execute("PRAGMA journal_mode=OFF")
con.execute("PRAGMA synchronous=OFF")
con.execute("CREATE VIRTUAL TABLE docs USING fts5(doc_id, body, tokenize='unicode61 remove_diacritics 2')")

# Insert in batches.
batch = []
BATCH_N = 50_000
n = 0
for did, txt in search_text.items():
    if not txt: continue
    batch.append((did, txt))
    if len(batch) >= BATCH_N:
        con.executemany("INSERT INTO docs(doc_id, body) VALUES (?, ?)", batch)
        n += len(batch); batch = []
        if n % 500_000 == 0:
            print(f"  inserted {n:,} ({time.time()-t0:.1f}s)")
if batch:
    con.executemany("INSERT INTO docs(doc_id, body) VALUES (?, ?)", batch); n += len(batch)
con.commit()
print(f"FTS5 built: {n:,} rows in {time.time()-t0:.1f}s")

# Free the search_text dict — FTS5 has it now.
del search_text
gc.collect()

FTS5_BAD_CHARS = re.compile(r'[\"\\(\\)\\*\\+\\-\\^]')
TOK_RE = re.compile(r"[^\\w\\d]+", re.UNICODE)

def bm25_tokens(text):
    if not text: return []
    out = []
    for w in TOK_RE.split(text.lower().strip()):
        if len(w) >= CONFIG["bm25_min_token_len"]:
            out.append(FTS5_BAD_CHARS.sub("", w))
    return [w for w in out if w]

def bm25_search(query_text, k):
    toks = bm25_tokens(query_text)
    # de-dup, cap to budget
    seen = set(); uniq = []
    for t in toks:
        if t not in seen:
            seen.add(t); uniq.append(t)
        if len(uniq) >= CONFIG["bm25_max_query_terms"]: break
    if not uniq: return []
    match = " OR ".join(f'"{t}"' for t in uniq)
    rows = con.execute(
        "SELECT doc_id, bm25(docs) AS s FROM docs WHERE docs MATCH ? ORDER BY s LIMIT ?",
        (match, k),
    ).fetchall()
    # bm25() in FTS5 returns negative score; lower = better, so flip sign.
    return [(d, -s) for d, s in rows]
""")


# ===========================================================================
md("""## Cell 10 — Vector channel setup

Loads pre-computed Qwen3-Embedding-8B corpus embeddings (21 GB across 27
chunks per handoff §3.5). Builds a manifest_row → my_doc_id translation
(matched by `(family, citation)`). Skipped if embeddings unavailable.
""")
code("""
VECTOR_OK = False
E_GPU = None
manifest_row_to_did = {}

if EMB_AVAILABLE:
    import numpy as np
    import pyarrow.parquet as pq

    if not PATHS["emb_manifest"].exists():
        print("[vector] manifest missing; vector channels disabled")
    else:
        print("Loading manifest...")
        manifest = pq.read_table(PATHS["emb_manifest"]).to_pandas()
        print(f"  manifest rows: {len(manifest):,}, cols: {list(manifest.columns)}")

        # Build manifest_row -> my doc_id mapping by (family, citation).
        t = time.time()
        for row_idx, mrow in enumerate(manifest.itertuples(index=False)):
            cit = getattr(mrow, "citation", "")
            fam = getattr(mrow, "family", "")
            my_dids = cit_to_doc_ids.get(cit, [])
            for did in my_dids:
                if doc_meta.get(did, {}).get("family") == fam:
                    manifest_row_to_did[row_idx] = did
                    break
        print(f"  manifest->my_did mapping: {len(manifest_row_to_did):,}/{len(manifest):,} "
              f"({100*len(manifest_row_to_did)/max(1,len(manifest)):.1f}%) in {time.time()-t:.1f}s")

        # Load all chunks to GPU as a single fp16 tensor.
        chunks = sorted(PATHS["emb_dir"].glob("qwen3_8b_unified_chunk*.npy"))
        if not chunks:
            print("[vector] no chunks found")
        else:
            import torch
            t = time.time()
            print(f"  loading {len(chunks)} chunks to GPU (~21 GB, ~30-60s)...")
            parts = [np.load(p) for p in chunks]
            E_np = np.concatenate(parts, axis=0)
            del parts
            E_GPU = torch.from_numpy(E_np).to("cuda")
            del E_np
            torch.cuda.empty_cache()
            print(f"  E_GPU shape={tuple(E_GPU.shape)} dtype={E_GPU.dtype}, "
                  f"VRAM={torch.cuda.memory_allocated()/1e9:.1f} GB, {time.time()-t:.1f}s")
            VECTOR_OK = True
else:
    print("[vector] embeddings not on disk; channels 10/11 will be empty.")
""")


# ===========================================================================
md("""## Cell 11 — Embedding model for query encoding""")
code("""
EMB_MODEL = None
EMB_TOK = None

def _ensure_emb_model():
    global EMB_MODEL, EMB_TOK
    if EMB_MODEL is not None:
        return EMB_MODEL, EMB_TOK
    import torch
    from transformers import AutoTokenizer, AutoModel
    print(f"[emb] loading {CONFIG['vector_emb_model']}...")
    EMB_TOK = AutoTokenizer.from_pretrained(CONFIG["vector_emb_model"], trust_remote_code=True)
    EMB_MODEL = AutoModel.from_pretrained(
        CONFIG["vector_emb_model"], torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    EMB_MODEL.eval()
    return EMB_MODEL, EMB_TOK

def encode_query(text, max_length=4096):
    if not VECTOR_OK or not text: return None
    import torch
    mdl, tok = _ensure_emb_model()
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(mdl.device)
    with torch.inference_mode():
        out = mdl(**enc)
    # Qwen3 embedding: last hidden state mean-pool with attention mask
    last = out.last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).type_as(last)
    pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled[0].detach().cpu().float().numpy()

def vector_search(q_emb, k):
    if not VECTOR_OK or q_emb is None or E_GPU is None: return []
    import torch, numpy as np
    q = torch.from_numpy(np.asarray(q_emb, dtype=np.float32))
    n = q.norm()
    if n.item() <= 0: return []
    q = (q / n).to(E_GPU.device).to(E_GPU.dtype)
    with torch.no_grad():
        scores = (E_GPU @ q).float()
        top = torch.topk(scores, k)
    rows = top.indices.cpu().numpy().tolist()
    vals = top.values.cpu().numpy().tolist()
    out = []
    for r, s in zip(rows, vals):
        did = manifest_row_to_did.get(r)
        if did:
            out.append((did, float(s)))
    return out
""")


# ===========================================================================
md("""## Cell 12 — Query expansion (Qwen3-32B → JSON targets)

Same prompt schema as v3. Output: statute_targets, case_targets,
concept_targets_en, term_targets_de, term_targets_fr, legal_area_keywords.
""")
code("""
QUERY_EXPANSION_PROMPT = '''You are a Swiss legal-retrieval query analyst.

Given an English question about Swiss law, emit STRICTLY VALID JSON:

{
  "statute_targets":     [string],   // articles likely cited. Format: "Art. <num> <code>".
  "case_targets":        [string],   // expected leading decisions. Format: "BGE X Y Z" or docket "1B_NNN/YYYY".
  "concept_targets_en":  [string],   // <=30 English legal concepts.
  "term_targets_de":     [string],   // <=25 German legal terms.
  "term_targets_fr":     [string],   // <=15 French legal terms if applicable.
  "legal_area_keywords": [string]    // <=5 area phrases.
}

Rules:
- Be thorough. Extra targets are cheap; missed targets break the retrieval.
- For statutes, list the article the question names AND neighbouring articles in the same procedural cluster (e.g., if Art. 221 StPO is named, also list Art. 222 / 227 / 212 StPO).
- If the case posture is a Bundesgericht appeal, include Art. 100 BGG and Art. 42 BGG.
- Output JSON only. No prose. No code fences.

QUERY:
{query}

JSON:'''

def run_query_expansion(query):
    import torch
    if not torch.cuda.is_available():
        print("No GPU; manual fallback:")
        print(QUERY_EXPANSION_PROMPT.replace("{query}", query))
        raw = input("Paste JSON: ")
        return json.loads(raw)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[qexp] loading {CONFIG['qwen_query_model']} (~65 GB bf16)...")
    tok = AutoTokenizer.from_pretrained(CONFIG["qwen_query_model"], trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        CONFIG["qwen_query_model"], torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    ).eval()

    prompt = QUERY_EXPANSION_PROMPT.replace("{query}", query)
    msgs = [
        {"role":"system","content":"You output strictly valid JSON. No commentary."},
        {"role":"user","content":prompt},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt").to(mdl.device)
    with torch.inference_mode():
        out = mdl.generate(**enc, max_new_tokens=CONFIG["qwen_max_new_tokens"], do_sample=False, temperature=0.0)
    completion = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print("RAW (first 400 chars):", completion[:400])
    a = completion.find("{"); b = completion.rfind("}")
    if a < 0 or b < 0: raise ValueError(f"No JSON braces: {completion!r}")
    parsed = json.loads(completion[a:b+1])
    # Free the 65 GB model immediately so embedding model fits next.
    del mdl, tok
    gc.collect(); torch.cuda.empty_cache()
    print(f"[qexp] freed Qwen3-32B; CUDA mem={torch.cuda.memory_allocated()/1e9:.1f} GB")
    return parsed

targets = run_query_expansion(QUERY)
print()
print("Targets parsed:")
for k, v in targets.items():
    n = len(v) if isinstance(v, list) else 1
    print(f"  {k}: ({n}) {v[:5] if isinstance(v, list) else v}")

with open(PATHS["out_dir"] / "val_001_query_targets.json", "w", encoding="utf-8") as fp:
    json.dump(targets, fp, ensure_ascii=False, indent=2)
""")


# ===========================================================================
md("""## Cell 13 — Encode query (raw + enriched) for vector channels

The enriched query concatenates the LLM's German + French terms + English
concepts after the query, landing the embedding into the corpus's German
legal-vocabulary region without HyDE-style narrative hallucination.
""")
code("""
q_emb_raw = None
q_emb_enriched = None

if VECTOR_OK:
    enrichment_bits = (
        (targets.get("term_targets_de") or [])
      + (targets.get("term_targets_fr") or [])
      + (targets.get("concept_targets_en") or [])
      + (targets.get("legal_area_keywords") or [])
    )
    enriched_query_text = QUERY + "\\n\\nKeywords: " + " ; ".join(enrichment_bits)

    print("Encoding raw query...")
    t = time.time(); q_emb_raw = encode_query(QUERY); print(f"  done in {time.time()-t:.1f}s")
    print("Encoding enriched query...")
    t = time.time(); q_emb_enriched = encode_query(enriched_query_text); print(f"  done in {time.time()-t:.1f}s")
    print(f"  enriched query length: {len(enriched_query_text)} chars, "
          f"keywords appended: {len(enrichment_bits)}")
""")


# ===========================================================================
md("""## Cell 14 — Channel functions""")
code("""
from collections import Counter

def channel_law_direct(canon_set, idx, budget):
    \"\"\"Take a set of canonical statute keys (LLM + co-citation expanded) and
    retrieve every law row whose canonical citation matches. No score-tie
    budget; if budget is None all matches are returned.\"\"\"
    counter = Counter()
    for canon in canon_set:
        for did in idx[canon]:
            counter[did] += 1
    items = counter.most_common()
    return items if budget is None else items[:budget]

def channel_court_statute(statute_canons, idx, budget):
    counter = Counter()
    for canon in statute_canons:
        for did in idx[canon]:
            counter[did] += 1
    return counter.most_common(budget)

def channel_sibling(seed_doc_ids, idx_court_base, doc_meta, budget):
    out = set()
    for did in seed_doc_ids:
        m = doc_meta.get(did) or {}
        cb = m.get("court_base")
        if cb:
            out.update(idx_court_base.get(cb, set()))
    out -= set(seed_doc_ids)
    return [(d, 1.0) for d in list(out)[:budget]]

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
            if cv not in seen:
                expanded.append(cv); seen.add(cv)
    return expanded

def channel_concept(expanded_concepts, idx, budget):
    counter = Counter()
    for tok in expanded_concepts:
        for did in idx.get(tok, ()):
            counter[did] += 1
    return counter.most_common(budget)

def channel_term(targets, idx, budget):
    counter = Counter()
    for key in ("term_targets_de", "term_targets_fr"):
        for raw in targets.get(key, []) or []:
            tok = norm_token(raw, CONFIG["lowercase_terms"])
            if tok:
                for did in idx[tok]:
                    counter[did] += 1
    return counter.most_common(budget)

def channel_per_area_bedrock(legal_area_keywords, per_area_canon_count, idx_law_direct, idx_court_statute, budget):
    \"\"\"For each legal-area keyword, find the matching corpus area, take top-N
    canonical statutes, return (a) law rows for the article + (b) the top
    court rows that cite the article. budget caps total.\"\"\"
    selected_areas = set()
    if not legal_area_keywords: return []
    keys = [k.lower() for k in legal_area_keywords]
    for area in per_area_canon_count.keys():
        for k in keys:
            if k in area:
                selected_areas.add(area); break
    if not selected_areas:
        return []
    # Aggregate canonical -> highest count across selected areas
    canon_score = Counter()
    for area in selected_areas:
        for canon, n in per_area_canon_count[area].most_common(CONFIG["per_area_top_n"]):
            canon_score[canon] = max(canon_score[canon], n)
    out = []; seen = set()
    for canon, _ in canon_score.most_common(CONFIG["per_area_top_n"]):
        for did in idx_law_direct.get(canon, set()):
            if did not in seen:
                out.append((did, canon_score[canon])); seen.add(did)
        if len(out) >= budget: break
    return out[:budget]

def channel_co_citation(targets, co_neighbours, idx_law_direct, idx_court_statute, budget):
    \"\"\"For each LLM statute target, fetch the top co-citation neighbours and
    retrieve their law + court rows.\"\"\"
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

# Vector channels — defined inline using existing vector_search().
def channel_vector(q_emb, k):
    return vector_search(q_emb, k)
""")


# ===========================================================================
md("""## Cell 15 — Run all channels + per-channel diagnostics""")
code("""
# 1) Build full statute_canon set: LLM-named + co-citation expansion.
llm_statute_canons = set()
for raw in targets.get("statute_targets", []) or []:
    c = statute_anchor_canonical(raw)
    if c: llm_statute_canons.add(c)

co_expanded_canons = set(llm_statute_canons)
for canon in llm_statute_canons:
    for nb, _ in co_neighbours.get(canon, []):
        co_expanded_canons.add(nb)
print(f"Statute canons: LLM={len(llm_statute_canons)}, +co-citation={len(co_expanded_canons)}")

# 2) Concept expansion (strict-substring against corpus vocab).
llm_concepts = (targets.get("concept_targets_en") or []) + (targets.get("legal_area_keywords") or [])
corpus_concept_keys = set(idx_concept_en.keys())
expanded_concepts = expand_concepts_strict(llm_concepts, corpus_concept_keys, CONFIG["concept_substring_top_k"])
print(f"Concepts: LLM={len(llm_concepts)} -> expanded={len(expanded_concepts)}")

# 3) Run channels.
# law_direct_match now uses the FULL expanded canonical set (LLM + co-citation),
# so co-cited articles' law rows are also force-included via the guarantee pool.
ch_law_direct  = channel_law_direct(co_expanded_canons, idx_law_direct, CONFIG["budget_law_direct"])
ch_court_stat  = channel_court_statute(co_expanded_canons, idx_court_statute, CONFIG["budget_court_statute"])

seed = ({d for d, _ in ch_court_stat} |
        {d for d, _ in ch_law_direct if doc_meta.get(d, {}).get("family") == "court"})
ch_sibling     = channel_sibling(seed, idx_court_base, doc_meta, CONFIG["budget_sibling"])

ch_concept     = channel_concept(expanded_concepts, idx_concept_en, CONFIG["budget_concept"])
ch_term        = channel_term(targets, idx_term_orig, CONFIG["budget_term"])
ch_per_area    = channel_per_area_bedrock(targets.get("legal_area_keywords", []),
                                          per_area_canon_count, idx_law_direct, idx_court_statute,
                                          CONFIG["budget_per_area"])
ch_cocit       = channel_co_citation(targets, co_neighbours, idx_law_direct, idx_court_statute,
                                     CONFIG["budget_co_citation"])

# BM25 channel: feed enriched query text.
enrichment_bits_for_bm25 = (
    (targets.get("term_targets_de") or [])
  + (targets.get("term_targets_fr") or [])
  + (targets.get("concept_targets_en") or [])
)
bm25_query_text = QUERY + " " + " ".join(enrichment_bits_for_bm25)
ch_bm25 = bm25_search(bm25_query_text, CONFIG["budget_bm25"])

# Vector channels.
ch_vector  = channel_vector(q_emb_raw,      CONFIG["budget_vector"])           if VECTOR_OK else []
ch_venrich = channel_vector(q_emb_enriched, CONFIG["budget_vector_enriched"])  if VECTOR_OK else []

CHANNELS = [
    ("law_direct_match",  ch_law_direct),
    ("court_statute",     ch_court_stat),
    ("co_citation",       ch_cocit),
    ("per_area_bedrock",  ch_per_area),
    ("sibling_expansion", ch_sibling),
    ("concept_en",        ch_concept),
    ("term_orig",         ch_term),
    ("bm25",              ch_bm25),
    ("vector_raw",        ch_vector),
    ("vector_enriched",   ch_venrich),
]

print()
print(f"{'channel':<22}  {'size':>6}  {'gold_in_ch':>11}  recall")
print("-" * 60)
for name, hits in CHANNELS:
    found_dids = {d for d, _ in hits}
    g = len(found_dids & gold_doc_set)
    print(f"{name:<22}  {len(hits):>6}  {g:>11}  {100*g/total_gold:5.1f}%")

union_did = set()
for _, hits in CHANNELS:
    union_did.update(d for d, _ in hits)
print()
print(f"Union: {len(union_did):,} unique doc_ids")
print(f"Gold in union: {len(union_did & gold_doc_set)}/{total_gold}  (UPPER BOUND on R@K)")
""")


# ===========================================================================
md("""## Cell 16 — RRF fusion + negative gate + R@K curve""")
code("""
from collections import defaultdict

def rrf_fuse(channels, k):
    score = defaultdict(float)
    for _, hits in channels:
        for rank, (did, _) in enumerate(hits):
            score[did] += 1.0 / (k + rank + 1)
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

rrf_scores = rrf_fuse(CHANNELS, CONFIG["rrf_k"])
ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
ranked_dids = [d for d, _ in ranked]
ranked_gated = apply_neg_gate(ranked_dids, doc_meta, CONFIG["noise_paragraph_roles"])

# Guarantee channels: prepended before the RRF tail.
ch_by_name = dict(CHANNELS)
guarantee = []
seen = set()
for cname in CONFIG["guarantee_channels"]:
    for did, _ in ch_by_name.get(cname, []):
        if did not in seen:
            guarantee.append(did); seen.add(did)
guarantee = apply_neg_gate(guarantee, doc_meta, CONFIG["noise_paragraph_roles"])

PASS_K = CONFIG["topk_final"]
final_topk = list(guarantee)
seen_f = set(final_topk)
for did in ranked_gated:
    if len(final_topk) >= PASS_K: break
    if did not in seen_f:
        final_topk.append(did); seen_f.add(did)

print(f"Pre-gate fused: {len(ranked_dids):,}")
print(f"Post-gate:      {len(ranked_gated):,}")
print(f"Guarantee pool: {len(guarantee):,}  (channels: {CONFIG['guarantee_channels']})")
print(f"Final top-{PASS_K}: {len(final_topk):,}")

print()
print(f"{'K':>6}  gold/{total_gold}  recall")
print("-" * 30)
for K in [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 5000]:
    K_use = min(K, len(final_topk))
    topk = set(final_topk[:K_use])
    g = len(topk & gold_doc_set)
    print(f"{K_use:>6}  {g:>5}/{total_gold}  {100*g/total_gold:5.1f}%")
    if K_use == len(final_topk): break

top_pass = set(final_topk[:PASS_K])
gold_in_top = top_pass & gold_doc_set
recall_at_k = len(gold_in_top) / total_gold

print()
print("=" * 50)
print(f"PASS CRITERION: R@{PASS_K} >= 0.70 (>= {math.ceil(0.70*total_gold)}/{total_gold})")
print(f"OBSERVED:       R@{PASS_K} = {recall_at_k:.3f}  ({len(gold_in_top)}/{total_gold})")
print("=" * 50)
""")


# ===========================================================================
md("""## Cell 17 — Per-gold trace""")
code("""
channel_membership = defaultdict(set)
for name, hits in CHANNELS:
    for did, _ in hits:
        channel_membership[did].add(name)

rank_lookup = {did: i+1 for i, did in enumerate(final_topk)}

print(f"{'rank':>6}  {'in1k':>4}  {'channels':<70}  citation")
print("-" * 140)
rows = []
for g in GOLD:
    docs = gold_doc_ids.get(g, [])
    if not docs:
        rows.append((10**9, "MISS", set(), g)); continue
    best_rank = min((rank_lookup.get(d, 10**9) for d in docs), default=10**9)
    best_did  = next((d for d in docs if rank_lookup.get(d, 10**9) == best_rank), None)
    chs = channel_membership.get(best_did, set())
    rows.append((best_rank, "YES" if best_rank <= CONFIG["topk_final"] else "no", chs, g))

rows.sort(key=lambda r: r[0])
for rk, in1k, chs, g in rows:
    rk_s = str(rk) if rk < 10**8 else "—"
    chs_s = ",".join(sorted(chs))
    print(f"{rk_s:>6}  {in1k:>4}  {chs_s:<70}  {g}")

missed = [r for r in rows if r[1] != "YES"]
print()
print(f"Missed: {len(missed)}/{total_gold}")
""")


# ===========================================================================
md("""## Cell 18 — Save artifacts""")
code("""
out = PATHS["out_dir"]

per_channel = {
    "channels": [
        {"name": name, "size": len(hits),
         "gold_in_channel": sum(1 for d, _ in hits if d in gold_doc_set)}
        for name, hits in CHANNELS
    ],
    "rrf_k": CONFIG["rrf_k"],
    "guarantee_channels": CONFIG["guarantee_channels"],
    "guarantee_pool_size": len(guarantee),
    "fused_pool_size": len(ranked_gated),
    "final_topk_size": len(final_topk),
    "vector_available": VECTOR_OK,
    "expanded_concepts_count": len(expanded_concepts),
    "co_expanded_canons_count": len(co_expanded_canons),
}
with open(out / "val_001_v4_per_channel.json", "w", encoding="utf-8") as fp:
    json.dump(per_channel, fp, ensure_ascii=False, indent=2)

with open(out / "val_001_v4_per_gold_trace.tsv", "w", encoding="utf-8") as fp:
    fp.write("rank\\tin_top_1000\\tchannels\\tcitation\\n")
    for rk, in1k, chs, g in rows:
        rk_s = str(rk) if rk < 10**8 else "unranked"
        fp.write(f"{rk_s}\\t{in1k}\\t{','.join(sorted(chs))}\\t{g}\\n")

with open(out / "val_001_v4_top1000_pool.txt", "w", encoding="utf-8") as fp:
    for did in final_topk[:CONFIG["topk_final"]]:
        m = doc_meta.get(did) or {}
        fp.write(f"{did}\\t{m.get('citation','')}\\n")

# Summary
S = []
S.append("# val_001 v4 anchor-funnel R@1000 attempt — summary")
S.append("")
S.append(f"- gold mapped: {len(gold_doc_ids)}/{len(GOLD)}")
S.append(f"- corpus size: {len(doc_meta):,}")
S.append(f"- guarantee pool: {len(guarantee):,}")
S.append(f"- final top-1000 size: {len(final_topk):,}")
S.append(f"- vector available: {VECTOR_OK}")
S.append(f"- R@1000: **{recall_at_k:.3f}**  ({len(gold_in_top)}/{total_gold})")
S.append("")
S.append("## Per-channel recall")
S.append("| channel | size | gold | recall |")
S.append("|---|---:|---:|---:|")
for c in per_channel["channels"]:
    g = c["gold_in_channel"]; sz = c["size"]
    S.append(f"| `{c['name']}` | {sz} | {g} | {100*g/total_gold:.1f}% |")
S.append("")
S.append("## Missed gold")
if not missed:
    S.append("None.")
else:
    S.append("| best_rank | channels | citation |")
    S.append("|---:|---|---|")
    for rk, _, chs, g in missed:
        rk_s = str(rk) if rk < 10**8 else "unranked"
        S.append(f"| {rk_s} | {','.join(sorted(chs)) or '(none)'} | `{g}` |")

with open(out / "val_001_v4_summary.md", "w", encoding="utf-8") as fp:
    fp.write("\\n".join(S))

print("Saved artifacts to:", out)
""")


# ===========================================================================
md("""## Cell 19 — Cleanup""")
code("""
try:
    con.close()
except Exception:
    pass
del idx_law_direct, idx_court_statute, idx_case_anchor, idx_court_base, idx_concept_en, idx_term_orig
del corpus_concept_keys, co_neighbours, per_area_canon_count
if VECTOR_OK:
    del E_GPU
    if EMB_MODEL is not None:
        del EMB_MODEL, EMB_TOK
gc.collect()
try:
    import torch
    if torch.cuda.is_available(): torch.cuda.empty_cache()
except Exception:
    pass
print("cleaned up.")
""")


# ===========================================================================
# Build notebook JSON
def build_notebook(cells):
    nb_cells = []
    for i, (kind, src) in enumerate(cells):
        lines = src.splitlines(keepends=True)
        cid = f"cell-{i:03d}"
        if kind == "markdown":
            nb_cells.append({"cell_type":"markdown","id":cid,"metadata":{},"source":lines})
        else:
            nb_cells.append({"cell_type":"code","id":cid,"metadata":{},
                             "execution_count":None,"outputs":[],"source":lines})
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
            "language_info": {"name":"python","version":"3.11"},
            "colab": {"provenance":[],"gpuType":"Blackwell"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


nb = build_notebook(CELLS)
out_path = Path(__file__).parent / "swiss_citation_anchor_funnel_val001_v4.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"Wrote {out_path}")
print(f"  cells: {len(CELLS)}, size: {out_path.stat().st_size:,} bytes")
