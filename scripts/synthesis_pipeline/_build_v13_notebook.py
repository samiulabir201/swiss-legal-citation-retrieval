"""Generate notebooks/v13_court_aware_adaptive_k/synthesis_v13_final.ipynb."""

from __future__ import annotations
import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


cells = []

# ── Cell 1 ─ title
cells.append(md("""# Swiss Legal Citation Retrieval — v13 Court-Aware Pipeline with Adaptive K

**Target:** Kaggle *LLM Agentic Legal Information Retrieval*

Starts from the **v7.5 funnel pool** (~45K candidates per query, recall ≈ 0.93)
covering both `law:*` and `court:*` candidates. Re-ranks via a court-aware
two-pass pipeline and selects predictions using an **adaptive K predictor**
derived from procedural-cascade signals.

## Pipeline
```
[v7.5 pool 50K/query, recall 0.93]
    ↓  Stage 1   — pool-restricted BM25 → top-500
    ↓  Stage 2   — Qwen3-Reranker-8B  (court-aware _fmt_rr branched on family)
    ↓  Stage 3   — fused = 0.7·norm(BM25_pool) + 0.3·P(yes); zone-split
    ↓  Stage 4   — Qwen3-8B judge for borderline only (per-family system prompt)
    ↓  Stage 5   — adaptive-K cutoff (procedural-cascade regex predictor)
    ↓
Submissions: adaptive, adaptive×1.3, fixed K=5 (train-honest fallback)
```

## Empirical priors driving this notebook
- Train queries: median gold = 2, ~1 cite per issue, mostly German substantive
- Val queries: median gold = 22, ~5 cites per issue, English procedural cascades
- Real test ≈ val-like, so adaptive K is the dominant precision lever
- v12 reference notebook hit F1=0.777 val but F1=0.296 train (Δ=+0.48). Adaptive K closes most of that gap (oracle K honest avg = 0.715 vs v12 honest = 0.537)

## GPU target
NVIDIA RTX Pro 6000 Blackwell — **95.6 GB VRAM**. Plenty of room to load both
Qwen3-Reranker-8B (~16 GB bf16) and Qwen3-8B (~16 GB bf16) concurrently and run
reranker batch 16, max_len 4096.
"""))

# ── Cell 2 ─ install
cells.append(code("""# ── Install dependencies (run once per Colab session)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "rank_bm25", "transformers>=4.51", "accelerate",
    "tqdm", "pandas", "pyarrow", "numpy"], check=False)
"""))

# ── Cell 3 ─ setup paths
cells.append(code("""# ── Imports, Drive mount, paths
import os, re, json, pickle, math, gc, time, datetime, sys
from pathlib import Path
from dataclasses import dataclass, field, fields as dc_fields
from collections import defaultdict, Counter
import numpy as np, pandas as pd
from tqdm.auto import tqdm

# Mount Drive (skipped if already mounted)
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except ImportError:
    pass

DRIVE_BASE = Path("/content/drive/MyDrive/Omnilex-Agentic-Retrieval-Competition")
DATA_DIR   = DRIVE_BASE / "retrieval" / "pipeline_inputs_v13"
OUT_DIR    = DRIVE_BASE / "retrieval" / "pipeline_output_v13"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Inputs — upload to DATA_DIR. See README cell at top.
# Court channel: corpus_snapshot.json.gz + per_query_snapshot.json + gold_doc_sets.json
# Law channel:   corpus.parquet + bm25_v2_index.pkl + bm25_v2_ids.pkl  (v12 setup)
POOL_SNAP_PATH      = DATA_DIR / "per_query_snapshot.json"
CORPUS_SNAP_PATH    = DATA_DIR / "corpus_snapshot.json.gz"
GOLD_DOC_SETS_PATH  = DATA_DIR / "gold_doc_sets.json"
VAL_CSV_PATH        = DATA_DIR / "val.csv"
TRAIN_CSV_PATH      = DATA_DIR / "train.csv"
QUERY_TRANS_PATH    = DATA_DIR / "query_translations_trainval.json"  # optional
# Law-channel inputs (v12)
CORPUS_PARQUET_PATH = DATA_DIR / "corpus.parquet"
BM25_INDEX_PATH     = DATA_DIR / "bm25_v2_index.pkl"
BM25_IDS_PATH       = DATA_DIR / "bm25_v2_ids.pkl"
LAW_KB_PATH         = DATA_DIR / "laws_knowledge_base.jsonl"  # optional, for kb fields
COURT_CARDS_PATH    = DATA_DIR / "court_cards_slim_for_v13.jsonl"  # slim v5_unified extract

# Output caches (per channel — keyed by did so two channels write to one file safely)
RERANKER_CACHE = OUT_DIR / "reranker_v13_cache.json"
JUDGE_CACHE    = OUT_DIR / "judge_v13_cache.json"

# Pipeline knobs
TOP_K_LAW         = 100     # Law channel: BM25+TLF top-K  (v12 setup)
TOP_K_COURT       = 500     # Court channel: v7.5 final_topk filtered top-K
RERANKER_MODEL    = "Qwen/Qwen3-Reranker-8B"
JUDGE_MODEL       = "Qwen/Qwen3-8B"
RERANKER_BATCH    = 16
# Per-family zone thresholds (law uses v12's tighter LOW; court uses v13's looser LOW)
HIGH_THRESH       = 0.55
LOW_THRESH_LAW    = 0.25
LOW_THRESH_COURT  = 0.15
FUSED_W_RECALL    = 0.7
FUSED_W_RERANK    = 0.3

print(f"DATA_DIR = {DATA_DIR}")
print(f"OUT_DIR  = {OUT_DIR}")
for p in (POOL_SNAP_PATH, CORPUS_SNAP_PATH, GOLD_DOC_SETS_PATH, VAL_CSV_PATH,
          CORPUS_PARQUET_PATH, BM25_INDEX_PATH, BM25_IDS_PATH, LAW_KB_PATH,
          COURT_CARDS_PATH):
    print(f"  exists  {p.exists()}  {p}")
"""))

# ── Cell 4 ─ adaptive K predictor
cells.append(md("""## Adaptive K predictor

Derived from the procedural-cascade analysis on train+val:

- `cascade_score` ≥ 8 → val-style heavy cascade (val_001/002/003) → K = `n_codes × 10`
- `cascade_score` ≥ 3 → moderate procedural framing → K ≈ `n_codes × 5`
- otherwise → train-style substantive → K = `max(2, n_codes × 2)`

Composite cascade score from query text alone (no LLM call):
- 2.0 × litigant references (`the prosecutor`, `der Beschuldigte`…)
- 1.5 × adversarial verbs (`sought`, `opposed`, `—i.e.`, `geltend gemacht`…)
- 1.5 × procedural-posture cues (`by order dated`, `mit Verfügung vom`…)
- 1.0 × concrete calendar dates
- 5.0 × `—i.e. does X justify Y?` closing pattern
- 3.0 × English language (val is 100% EN, train is 98% DE)
"""))

cells.append(code('''# ── Adaptive K predictor (regex-only, deterministic)

KNOWN_CODES = {
    # civil
    "ZGB","CC","OR","CO","ZPO","CPC","IPRG","LDIP","SchKG","LP","LEF",
    # criminal
    "StGB","CP","StPO","CPP","BetmG","LStup",
    # constitutional / public
    "BV","Cst","Cost","EMRK","CEDH",
    # federal procedure
    "BGG","LTF","OG","OJ","VwVG","PA","StBOG",
    # tax / finance
    "DBG","LIFD","StHG","LHID","MWSTG","LTVA","VStG",
    # migration
    "AIG","AuG","LEI","LEtr","AsylG","LAsi",
    # social insurance
    "ATSG","LPGA","AHVG","LAVS","IVG","LAI","UVG","LAA",
    "AVIG","BVG","LPP","KVG","LAMal","ELG","EOG",
    # environment / planning
    "USG","LPE","RPG","LAT","UVPV","OEIE","GSchG","LEaux",
    # other
    "KG","LCart","MSchG","PatG","URG","FINMAG","DSG","LPD",
    "GBV","FusG","BoeB","LMP","WaG","NHG","VRV",
}
CODE_RE = re.compile(r"\\b(" + "|".join(re.escape(c) for c in KNOWN_CODES) + r")\\b")

LITIGANT_RE = re.compile(
    r"\\b(the\\s+(?:prosecutor|detainee|defendant|plaintiff|accused|claimant|appellant|respondent|insured|insurer)"
    r"|der\\s+(?:Beschuldigte|Beklagte|Kläger|Angeschuldigte|Versicherte|Beschwerdeführer)"
    r"|der\\s+Staatsanwalt|le\\s+procureur)\\b", re.IGNORECASE)

ADVERSARIAL_RE = re.compile(
    r"(sought\\s+(?:an\\s+)?extension|opposed|argued|requested|denied|"
    r"submitted|while\\s+the|whereas|—i\\.e\\.|i\\.e\\.\\s+does|"
    r"beantragte|wehrte\\s+sich|widersetzte\\s+sich|geltend\\s+gemacht|verlangte)",
    re.IGNORECASE)

POSTURE_RE = re.compile(
    r"(by\\s+order\\s+dated|by\\s+(?:decision|order)\\s+of|remanded|appeal|extension|sought|"
    r"mit\\s+Verfügung\\s+vom|mit\\s+(?:Entscheid|Beschluss)\\s+vom|"
    r"wurde\\s+(?:zur|in)\\s+(?:Untersuchungshaft|Sicherheitshaft)|erhob\\s+Beschwerde|"
    r"par\\s+ordonnance\\s+du|par\\s+décision\\s+du|a\\s+formé\\s+recours)",
    re.IGNORECASE)

DATE_DDMMYY_RE = re.compile(r"\\b\\d{1,2}\\.\\s?\\d{1,2}\\.\\s?\\d{4}\\b")
DATE_WORD_DE_RE = re.compile(r"\\b\\d{1,2}\\.\\s*(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\\s+\\d{4}\\b", re.IGNORECASE)
DATE_WORD_EN_RE = re.compile(r"\\b\\d{1,2}\\s+(January|February|March|April|May|June|July|August|September|October|November|December)\\s+\\d{4}\\b", re.IGNORECASE)

CLOSING_IIE_RE = re.compile(r"—\\s*i\\.e\\.\\s+(?:does|may|can|is|do|are|should)", re.IGNORECASE)


def detect_lang_is_english(s: str) -> bool:
    s = (s or "").lower()
    en = sum(s.count(w) for w in (" the ", " is ", " a ", " an ", " does ", " may ", " under "))
    de = sum(s.count(w) for w in (" der ", " die ", " das ", " ist ", " ein ", " eine ", " für ", " mit "))
    return en > de


def predict_k(query: str) -> dict:
    q = query or ""
    n_codes = max(1, len(set(CODE_RE.findall(q))))
    n_lit  = len(LITIGANT_RE.findall(q))
    n_adv  = len(ADVERSARIAL_RE.findall(q))
    n_pos  = len(POSTURE_RE.findall(q))
    n_dates = (len(DATE_DDMMYY_RE.findall(q))
               + len(DATE_WORD_DE_RE.findall(q))
               + len(DATE_WORD_EN_RE.findall(q)))
    has_iie = bool(CLOSING_IIE_RE.search(q))
    is_en = detect_lang_is_english(q)

    cascade = (2.0 * n_lit
               + 1.5 * n_adv
               + 1.5 * n_pos
               + 1.0 * n_dates
               + 5.0 * has_iie
               + 3.0 * (1 if is_en else 0))

    # TWO INDEPENDENT predictors — K_law and K_court calibrated separately
    # to per-family gold counts observed on val.
    #
    # Val gold breakdown:
    #   cascade ≥ 10  (val_001-003):  19-24 law, 16-23 court (~50/50 split)
    #   cascade  3-10 (val_004-010):  9-20 law, 1-11 court  (law dominant ~70/30)
    #   cascade  < 3  (train-style):  median 2 law, ~0 court
    if cascade >= 10:
        # Heavy procedural cascade — roughly balanced law/court gold
        K_law   = max(15, min(40, int(cascade * 1.5)))   # 15-40
        K_court = max(15, min(40, int(cascade * 1.5)))   # 15-40
        tier = "cascade_heavy"
    elif cascade >= 3:
        # Moderate cascade — law-dominant on val
        K_law   = max(8,  min(25, int(cascade * 2.0)))   # 8-25
        K_court = max(3,  min(15, int(cascade * 1.0)))   # 3-15
        tier = "cascade_moderate"
    else:
        # Train-style substantive question — mostly law, court rare
        K_law   = max(2, int(cascade * 1.5))
        K_court = max(0, int(cascade * 0.3))
        tier = "substantive_only"

    K_law   = max(2, min(40, K_law))
    K_court = max(0, min(40, K_court))
    K_total = K_law + K_court    # for backwards-compat with eval / diagnostics

    return {
        "K_pred": K_total, "K_law": K_law, "K_court": K_court,
        "tier": tier, "cascade_score": round(cascade, 2),
        "n_codes_in_query": n_codes, "n_litigants": n_lit, "n_adversarial": n_adv,
        "n_posture": n_pos, "n_dates": n_dates, "has_iie_closing": has_iie,
        "is_english": is_en,
    }


# Smoke test on val_001 — should report cascade_heavy
demo_query = ("May a court lawfully order a three-month extension of pre-trial "
              "detention under Art. 221 Abs. 1 lit. b StPO consistent with proportionality "
              "when the accused was remanded by an order dated 18 October 2024 and the "
              "prosecutor sought an extension on 10 December 2024, while the detainee "
              "opposed on the ground that—i.e. does the asserted risk justify a prolongation?")
print(json.dumps(predict_k(demo_query), indent=2))
'''))

# ── Cell 5 ─ Candidate + _fmt_rr
cells.append(md("""## Candidate dataclass + court-aware `_fmt_rr`

`Candidate` carries both law and court fields. `_fmt_rr` branches on
`candidate.family`:
- `"law"`   → 5-line law dossier (abbreviation / title / heading / type / text)
- `"court"` → 7-line court dossier with chamber, paragraph role, authority
              (case_importance, cited_by ≈ N), and optional doctrinal_rule /
              legal_test from the 16.55% LLM-enriched paragraphs
"""))

cells.append(code('''# ── Candidate dataclass (law + court) and _fmt_rr (branched)
@dataclass
class Candidate:
    citation_canon: str
    did: str                                    # "law:N" | "court:N"
    family: str                                  # "law" | "court"
    recall_rank: int = 0                        # rank in v7.5 final_topk
    pool_bm25_score: float = 0.0                # Stage 1 BM25-over-pool
    bm25_norm_score: float = 0.0                # per-query min-max norm
    reranker_score: float = 0.0
    fused_score: float = 0.0
    zone: str = ""                              # "yes" | "no" | "borderline"
    # shared
    text: str = ""
    # law-only
    law_abbreviation: str = ""
    law_name_en: str = ""
    title: str = ""
    context_heading_title: str = ""
    provision_type: str = ""
    # court-only
    chamber: str = ""
    chamber_label: str = ""
    court_code: str = ""
    paragraph_role: str = ""
    case_importance: float = 0.0
    is_leading_decision: bool = False
    has_substantive_role: bool = False
    co_citation_count_static: int = 0
    case_peer_count: int = 0
    legal_area_static: str = ""
    doctrinal_rule: str = ""
    legal_test: str = ""
    topic_en: str = ""
    statute_anchors: list = field(default_factory=list)  # statutes this case interprets


RERANKER_INSTRUCTION_LAW = (
    "Given a query about Swiss law, determine whether the provided law article "
    "is related to or applicable to the legal issue."
)
RERANKER_INSTRUCTION_COURT = (
    "Given a query about Swiss law, determine whether the provided Swiss Federal "
    "Court paragraph is a binding or persuasive precedent for the legal issue, "
    "considering the chamber, paragraph role, and case authority."
)


def _fmt_rr(query: str, c: Candidate) -> str:
    if c.family == "court":
        leading     = " (leading decision)" if c.is_leading_decision else ""
        substantive = " [substantive]" if c.has_substantive_role else " [non-substantive]"
        parts = [
            f"{c.citation_canon}{leading}",
            f"Court: {c.court_code} — {c.chamber_label}",
            f"Area: {c.legal_area_static}",
            f"Paragraph role: {c.paragraph_role}{substantive}",
            f"Authority: importance={c.case_importance:.2f}, "
            f"cited_by≈{c.co_citation_count_static}, case_peers={c.case_peer_count}",
        ]
        if c.doctrinal_rule:
            parts.append(f"Doctrinal rule: {c.doctrinal_rule[:200]}")
        if c.legal_test:
            parts.append(f"Legal test: {c.legal_test[:200]}")
        if c.topic_en:
            parts.append(f"Topic: {c.topic_en}")
        if c.statute_anchors:
            # Statutes this case interprets — direct match signal when the query
            # references the same statute. Show up to ~8 to keep dossier compact.
            anchors_str = "; ".join(str(s) for s in c.statute_anchors[:8])
            parts.append(f"Statute anchors: {anchors_str}")
        parts.append(f"Text: {c.text[:800]}")
        doc = "\\n".join(parts)
        instruct = RERANKER_INSTRUCTION_COURT
    else:
        doc = (f"{c.citation_canon}\\n"
               f"Law: {c.law_abbreviation} — {c.law_name_en}\\n"
               f"Title: {c.title}\\nHeading: {c.context_heading_title}\\n"
               f"Type: {c.provision_type}\\nText: {c.text[:500]}")
        instruct = RERANKER_INSTRUCTION_LAW
    return f"<Instruct>: {instruct}\\n<Query>: {query}\\n<Document>: {doc}"
'''))

# ── Cell 6 ─ judge prompts
cells.append(code('''# ── Judge prompts (per family)

JUDGE_SYSTEM_LAW = """You are a Swiss Federal Court (Bundesgericht) legal citation expert.

DOMAIN KNOWLEDGE — Swiss Legal Citation Practice:
Swiss court decisions (BGE) and legal briefs cite provisions across multiple categories:

1. SUBSTANTIVE LAW: The core articles governing the legal issue (StGB for criminal offenses, OR for contracts, ZGB for civil matters)
2. DEFINITIONS: Articles that define key legal terms used in the case (e.g., Art. 8 ATSG defines invalidity)
3. PROCEDURAL RULES: Articles governing how the case is processed (StPO for criminal procedure, ZPO for civil procedure)
4. APPEAL PROVISIONS: Articles about legal remedies — Beschwerde (Art. 393ff StPO), Berufung, appeal deadlines
5. COST ALLOCATION: Articles about who pays court costs and attorney fees (Art. 422, 428 StPO; Art. 64 BGG)
6. COURT JURISDICTION: Articles defining which court decides (Art. 37/39 StBOG, Art. 100 BGG)
7. CONSTITUTIONAL PRINCIPLES: Fair trial (Art. 29 BV), proportionality, good faith (Art. 2 ZGB)

A query about pre-trial detention will cite detention rules AND appeal rules AND cost rules AND court jurisdiction.

YOUR TASK: For each candidate article, read the German text carefully and decide YES or NO.
Say YES if the article belongs in ANY of the 7 categories above for this specific legal query.
Say NO only if the article is from a completely unrelated legal domain.
When uncertain, say YES.

For each candidate, respond with exactly:
CITATION | VERDICT: YES or NO"""


JUDGE_SYSTEM_COURT = """You are a Swiss Federal Court (Bundesgericht) precedent expert.

For a query about a Swiss legal issue, decide whether a candidate court-decision paragraph is a binding or persuasive precedent for the issue.

KEY SIGNALS:
- Same legal area (social insurance for IV/AHV/ATSG; criminal procedure for StPO detention; civil law for OR/ZGB)
- Same procedural context (admissibility, merits, sentencing, appeal, costs)
- Authority: BGE (leading decision) > docket judgment > cantonal
- Substantive paragraphs (reasoning, legal_standard, application, holding) are usually citable; facts, procedural_history, notification rarely are
- Doctrinal rule or legal test in the paragraph matches the question
- Chamber match (5A for family/SchKG, 6B for criminal, 8C/9C for social insurance, 1B/1C for public/criminal procedure)

A precedent is RELEVANT if:
- It establishes or applies the rule that governs the issue
- It interprets the same statute/article the query is about
- It addresses the same fact pattern

When uncertain, err on YES.

For each candidate, respond with exactly:
CITATION | VERDICT: YES or NO"""


def build_judge_prompt(query: str, cands, family: str) -> str:
    parts = [f"LEGAL QUERY: {query[:800]}"]
    label = "BORDERLINE COURT-PARAGRAPH CANDIDATES" if family == "court" else "BORDERLINE LAW-ARTICLE CANDIDATES"
    parts.append(f"\\n{len(cands)} {label} to judge:\\n")
    for i, c in enumerate(cands, 1):
        if family == "court":
            extra = (f"\\n    Chamber: {c.chamber_label or c.chamber}  "
                     f"Role: {c.paragraph_role or 'unknown'}  "
                     f"BGE={'yes' if c.is_leading_decision else 'no'}  "
                     f"cited_by≈{c.co_citation_count_static}")
            if c.doctrinal_rule:
                extra += f"\\n    Doctrinal rule: {c.doctrinal_rule[:200]}"
            txt = c.text[:400] if c.text else "(kein Text)"
            parts.append(f"[{i}] {c.citation_canon}{extra}\\n    Text: {txt}\\n")
        else:
            parts.append(
                f"[{i}] {c.citation_canon}\\n"
                f"    Law: {c.law_abbreviation} — {c.law_name_en}\\n"
                f"    German text: {(c.text or '(kein Text)')[:400]}\\n"
            )
    parts.append("\\nFor each candidate, output: CITATION | VERDICT: YES or NO")
    return "\\n".join(parts)
'''))

# ── Cell 7 ─ Stage 0 load
cells.append(md("""## Stage 0 — Load v7.5 pool + slim KBs + val/train"""))

cells.append(code('''# ── Stage 0: load all inputs (canonical v7.5 snapshot)
import gzip

print("Loading v7.5 pool snapshot ...")
with POOL_SNAP_PATH.open("r", encoding="utf-8") as f:
    pool_snap = json.load(f)
print(f"  qids: {len(pool_snap)}")
for qid, snap in pool_snap.items():
    print(f"    {qid}: {len(snap['final_topk'])} candidates, "
          f"R_at_K={snap.get('R_at_K', '?'):.3f}, gold={snap.get('gold', '?')}")

print("\\nLoading corpus_snapshot.json.gz (CANONICAL did → cit/text/role/family) ...")
t0 = time.time()
with gzip.open(CORPUS_SNAP_PATH, "rt", encoding="utf-8") as f:
    corpus_snap = json.load(f)
print(f"  loaded {len(corpus_snap):,} dids  ({time.time()-t0:.1f}s)")

print("\\nLoading gold_doc_sets.json (CANONICAL gold dids per qid) ...")
with GOLD_DOC_SETS_PATH.open("r", encoding="utf-8") as f:
    gold_doc_sets = json.load(f)
print(f"  qids with gold: {len(gold_doc_sets)}")

val = pd.read_csv(VAL_CSV_PATH)
train = pd.read_csv(TRAIN_CSV_PATH) if TRAIN_CSV_PATH.exists() else None
print(f"\\nval: {len(val)} queries, train: {len(train) if train is not None else 'N/A'}")

# Law channel inputs (v12)
print("\\nLoading v12 law-side artefacts (corpus.parquet + BM25 index) ...")
t0 = time.time()
corpus_law = pd.read_parquet(CORPUS_PARQUET_PATH)
with BM25_INDEX_PATH.open("rb") as f:
    bm25_law = pickle.load(f)
with BM25_IDS_PATH.open("rb") as f:
    bm25_law_ids = pickle.load(f)["citation_canon"]
print(f"  corpus_law:    {len(corpus_law):,} rows   ({time.time()-t0:.1f}s)")
print(f"  bm25 ids:      {len(bm25_law_ids):,}")

# v12 lookup: corpus.parquet indexed by citation_canon (NOT by positional iloc)
# This is critical — corpus_law row order may NOT match bm25_law_ids order, so
# `corpus_law.iloc[idx]` would return metadata for the wrong citation.
if "citation_canon" not in corpus_law.columns:
    raise RuntimeError(
        "corpus.parquet missing 'citation_canon' column. "
        "Cannot align with bm25 ids — would scramble dossier metadata."
    )
cd = corpus_law.set_index("citation_canon").to_dict("index")
print(f"  cd (corpus-by-citation): {len(cd):,} entries")

# Optional law KB for heading/provision_type fields (v12's dossier had these)
kb = {}
lne = {}      # law_abbreviation -> law_name_en (fallback when kb missing)
if LAW_KB_PATH.exists():
    t1 = time.time()
    with LAW_KB_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line); c = r.get("citation_canon", "")
                if c:
                    kb[c] = r
                    law_info = r.get("law", {}) or {}
                    a = law_info.get("law_abbreviation", "")
                    n = law_info.get("law_name_en", "")
                    if a and n and a not in lne:
                        lne[a] = n
            except Exception:
                continue
    print(f"  kb (laws_knowledge_base.jsonl): {len(kb):,} entries  ({time.time()-t1:.1f}s)")
else:
    print(f"  WARNING: {LAW_KB_PATH.name} not found — dossier "
          f"context_heading_title/provision_type will be empty (degrades reranker).")

# Query translations (optional, matches v12's bilingual trick)
qt = {}
if QUERY_TRANS_PATH.exists():
    qt = json.loads(QUERY_TRANS_PATH.read_text(encoding="utf-8"))
    print(f"  query_translations: {len(qt)} qids")

# Court authority cards — replace the slim corpus_snapshot.ct with the actual
# paragraph text (~800 chars), plus topic / statute_anchors signals. This is
# what gives the court reranker something real to discriminate on. Without
# this the court F1 sits near zero.
court_cards = {}
if COURT_CARDS_PATH.exists():
    t1 = time.time()
    with COURT_CARDS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            cit = r.get("citation") or ""
            if not cit:
                continue
            # Dedup: prefer the variant with more populated enrichment fields,
            # then the one with longer text_excerpt_original.
            en = r.get("rag_enrichment") or {}
            score = sum(1 for v in en.values() if v)
            text_len = len(r.get("text_excerpt_original") or "")
            existing = court_cards.get(cit)
            if existing is None or (score, text_len) > existing["_rank"]:
                r["_rank"] = (score, text_len)
                court_cards[cit] = r
    for v in court_cards.values():
        v.pop("_rank", None)
    print(f"  court_cards: {len(court_cards):,} entries  ({time.time()-t1:.1f}s)")
else:
    print(f"  WARNING: {COURT_CARDS_PATH.name} not found — court dossier "
          f"will use slim corpus_snapshot.ct only (degrades court reranker).")
'''))


# ── Cell 7b ─ Gold tracking helper (uses gold_doc_sets.json directly)
cells.append(md("""## Gold tracking — recall report at every stage

`gold_doc_sets.json` already has the canonical gold did set per qid (the same
set the v7.5 funnel's recall=0.93 was computed against). No fuzzy string
matching needed — gold is already at the did level and aligned with
`corpus_snapshot.json.gz`'s indexing.

`recall_funnel` accumulates per-stage stats so we get one summary table at the end.
"""))

cells.append(code('''# ── Gold did sets (already canonical) + per-stage recall report

gold_map = {qid: {
    "gold_dids": set(dids),
    "n_resolved": len(dids),
    "n_total_strings": len(dids),
} for qid, dids in gold_doc_sets.items()}

print(f"  {'qid':<10} {'gold_dids':>9} {'in_pool':>7}")
total_gold = total_in_pool = 0
for qid in val["query_id"]:
    if qid not in gold_map:
        print(f"  {qid:<10} (no gold_doc_sets entry)")
        continue
    pool = set(pool_snap[qid]["final_topk"]) if qid in pool_snap else set()
    in_pool = len(gold_map[qid]["gold_dids"] & pool)
    print(f"  {qid:<10} {gold_map[qid]['n_resolved']:>9} {in_pool:>7}")
    total_gold += gold_map[qid]["n_resolved"]
    total_in_pool += in_pool
print(f"  TOTAL gold dids: {total_gold}, in pool: {total_in_pool} "
      f"({total_in_pool/max(total_gold,1)*100:.1f}% — should match v7.5 ~0.93)")

# Per-stage recall funnel
recall_funnel = []


def stage_recall_report(stage_name, qid_to_surviving_dids, label="dids"):
    print(f"\\n=== Recall @ {stage_name} ===")
    print(f"  {'qid':<10} {'gold':>6} {'survived':>9} {'dropped':>9} {'recall':>8}")
    macro_recalls = []
    for qid in val["query_id"]:
        gold = gold_map.get(qid, {}).get("gold_dids", set())
        surv_set = qid_to_surviving_dids.get(qid, set())
        survived = len(gold & surv_set)
        dropped  = len(gold - surv_set)
        rec = survived / max(len(gold), 1)
        macro_recalls.append(rec)
        recall_funnel.append({
            "stage": stage_name, "qid": qid,
            "gold_total": len(gold), "survived": survived,
            "dropped": dropped, "recall": rec,
        })
        print(f"  {qid:<10} {len(gold):>6} {survived:>9} {dropped:>9} {rec:>7.3f}")
    print(f"  MACRO recall @ {stage_name}: {sum(macro_recalls)/max(len(macro_recalls),1):.3f}")


def report_dropped_gold(stage_name, qid_to_surviving_dids, max_per_qid=5):
    print(f"\\n--- Gold dropped @ {stage_name} (sample) ---")
    for qid in val["query_id"]:
        gold = gold_map.get(qid, {}).get("gold_dids", set())
        surv = qid_to_surviving_dids.get(qid, set())
        dropped_dids = gold - surv
        if not dropped_dids: continue
        print(f"  {qid}: dropped {len(dropped_dids)} gold:")
        for did in sorted(dropped_dids)[:max_per_qid]:
            rec = corpus_snap.get(did, {})
            cit = rec.get("cit", "(unknown)")
            print(f"    {did:<22} -> {cit}")
        if len(dropped_dids) > max_per_qid:
            print(f"    ... +{len(dropped_dids) - max_per_qid} more")
'''))

# ── Cell 8 ─ Build candidates (TWO CHANNELS — law via BM25+TLF, court via v7.5)
cells.append(md("""## Stage 1a — Build candidates per query, **two independent channels**

- **Law channel** — BM25 over `corpus.parquet` with TLF expansion mined from
  `train.csv` (v12's exact recall stage). Top-`TOP_K_LAW`=100 law candidates.
  Recall@100 on LAW gold ≈ 0.90 (proven by v12).

- **Court channel** — v7.5 funnel's `final_topk` filtered to `court:*`
  candidates. Top-`TOP_K_COURT`=500 court candidates. Recall@500 on COURT
  gold ≈ 0.30-0.35.

The two channels never compete for ranking. They each have their own
fused-score normalization, zone-split, judge call, and adaptive-K
selection. **Only the final output is unioned** (law_predictions ∪
court_predictions).
"""))

cells.append(code('''# ── Build Candidate objects per query from pool
# All metadata is either from corpus_snapshot or derived from the citation string.

# Deterministic mappings (from build_court_authority_cards.py)
BGE_DIVISION_AREAS = {
    "I":   "constitutional and public law",
    "II":  "administrative, tax, migration, and regulatory law",
    "III": "civil law",
    "IV":  "criminal law and criminal procedure",
    "V":   "social insurance law",
}
DOCKET_PREFIX_AREAS = {
    "1A": "constitutional and public law", "1B": "criminal procedure and coercive measures",
    "1C": "constitutional and public law", "1D": "constitutional and public law",
    "1E": "constitutional and public law", "1F": "constitutional and public law",
    "1G": "constitutional and public law", "1P": "constitutional and public law",
    "1S": "constitutional and public law", "2A": "administrative and tax law",
    "2C": "administrative, tax, migration, and regulatory law", "2D": "administrative law",
    "2E": "administrative and European law", "2F": "administrative law",
    "2G": "administrative, tax, migration, and regulatory law", "2P": "administrative and public law",
    "4A": "civil obligations, contract, commercial, and banking law", "4B": "civil law",
    "4C": "civil law", "4D": "civil law subsidiary constitutional matters",
    "4F": "civil law", "4G": "civil law", "4P": "civil law",
    "5A": "family law, inheritance, debt enforcement, and civil law", "5B": "family and civil law",
    "5C": "family and civil law", "5D": "civil law subsidiary constitutional matters",
    "5E": "family and civil law", "5F": "civil law",
    "5G": "family law, inheritance, debt enforcement, and civil law",
    "5N": "family and civil law", "5P": "family and civil law",
    "6A": "criminal law and administrative criminal law", "6B": "criminal law and criminal procedure",
    "6C": "criminal law and criminal procedure", "6F": "criminal law",
    "6G": "criminal law and criminal procedure", "6P": "constitutional and public law",
    "6S": "criminal law", "7B": "criminal law and criminal procedure",
    "7F": "criminal law", "7G": "criminal law and criminal procedure",
    "8C": "social insurance and public employment law", "8D": "social insurance law",
    "8F": "social insurance law", "8G": "social insurance and public employment law",
    "9C": "social insurance law", "9D": "social insurance law",
    "9E": "social insurance law", "9F": "social insurance law",
    "9G": "social insurance law", "9X": "social insurance law",
    "11Z": "civil law", "12T": "disciplinary proceedings",
    "13Y": "criminal law", "10Y": "criminal law",
}
EVG_AREAS = {
    "I": "invalidity insurance law", "U": "accident insurance law",
    "K": "health insurance law", "H": "accident and liability insurance law",
    "C": "unemployment insurance law", "B": "occupational pension and social insurance law",
    "P": "supplementary benefits law", "M": "military insurance law",
    "E": "social insurance law", "F": "social insurance law",
}
LEGAL_AREA_STATIC = {
    "constitutional and public law":                                "constitutional_public_law",
    "criminal procedure and coercive measures":                     "criminal_procedure",
    "criminal law and criminal procedure":                          "criminal_law",
    "criminal law and administrative criminal law":                 "criminal_law",
    "criminal law":                                                 "criminal_law",
    "administrative, tax, migration, and regulatory law":           "administrative_tax",
    "administrative and tax law":                                   "administrative_tax",
    "administrative law":                                           "administrative_tax",
    "administrative and European law":                              "administrative_tax",
    "administrative and public law":                                "administrative_tax",
    "civil obligations, contract, commercial, and banking law":    "civil_law",
    "civil law":                                                    "civil_law",
    "civil law subsidiary constitutional matters":                 "civil_law",
    "family law, inheritance, debt enforcement, and civil law":    "civil_law",
    "family and civil law":                                         "civil_law",
    "social insurance and public employment law":                  "social_insurance",
    "social insurance law":                                         "social_insurance",
    "invalidity insurance law":                                     "social_insurance",
    "accident insurance law":                                       "social_insurance",
    "health insurance law":                                         "social_insurance",
    "accident and liability insurance law":                        "social_insurance",
    "unemployment insurance law":                                   "social_insurance",
    "occupational pension and social insurance law":               "social_insurance",
    "supplementary benefits law":                                   "social_insurance",
    "military insurance law":                                       "social_insurance",
    "disciplinary proceedings":                                     "disciplinary",
}

BGE_PARSE_RE = re.compile(r"^\\s*(?:BGE|ATF|DTF)\\s+(\\d{1,4})\\s+([IVXLC]{1,5})\\s+(\\d+[a-z]?)", re.IGNORECASE)
DOCKET_PARSE_RE = re.compile(r"^\\s*(\\d{1,2}[A-Z]{1,4})[_\\.](\\d{1,6})/(\\d{4})")
EVG_PARSE_RE = re.compile(r"^\\s*([IUKHCBPMEF])\\s+(\\d{1,5})/(\\d{2,4})")
ART_PARSE_RE = re.compile(r"^\\s*Art\\.\\s+(\\d+[a-zA-Z]*).*\\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöü0-9./_-]+)\\s*$")
SUBSTANTIVE_ROLES = {"reasoning", "legal_standard", "application", "holding"}


def derive_court_metadata(cit: str, paragraph_role: str):
    """From a court citation string, derive chamber/chamber_label/court_code/...."""
    out = dict(chamber="", chamber_label="", court_code="BGer", case_importance=0.7,
               is_leading_decision=False, legal_area_static="unknown")
    m = BGE_PARSE_RE.match(cit)
    if m:
        section = m.group(2).upper()
        out["chamber"] = section
        out["chamber_label"] = BGE_DIVISION_AREAS.get(section, "")
        out["case_importance"] = 1.0
        out["is_leading_decision"] = True
        out["court_code"] = "BGer"
    else:
        m = DOCKET_PARSE_RE.match(cit)
        if m:
            prefix = m.group(1)
            out["chamber"] = prefix
            out["chamber_label"] = DOCKET_PREFIX_AREAS.get(prefix, "")
            out["court_code"] = "BGer"
        else:
            m = EVG_PARSE_RE.match(cit)
            if m:
                prefix = m.group(1)
                out["chamber"] = prefix
                out["chamber_label"] = EVG_AREAS.get(prefix, "")
                out["court_code"] = "EVG"
    out["legal_area_static"] = LEGAL_AREA_STATIC.get(out["chamber_label"], "unknown")
    out["has_substantive_role"] = (paragraph_role in SUBSTANTIVE_ROLES)
    return out


def derive_law_metadata(cit: str):
    """From a law citation string, extract law_abbreviation (last token)."""
    parts = (cit or "").split()
    abbrev = parts[-1] if parts else ""
    return dict(law_abbreviation=abbrev, law_name_en="", title="",
                context_heading_title="", provision_type="")


def make_candidate(did: str, rank: int):
    rec = corpus_snap.get(did)
    if rec is None:
        return None
    cit = rec.get("cit") or ""
    text = rec.get("ct") or ""
    family = rec.get("fam") or ("court" if did.startswith("court:") else "law")
    paragraph_role = rec.get("pr") or ""
    if family == "court":
        meta = derive_court_metadata(cit, paragraph_role)
        # Enrich from court_cards if available — gives the reranker an actual
        # paragraph text (~800 chars), English topic descriptors, and statute
        # anchors instead of the slim corpus_snapshot.ct field.
        card = court_cards.get(cit, {}) if court_cards else {}
        en   = card.get("rag_enrichment", {}) or {}
        card_text = card.get("text_excerpt_original") or ""
        # Prefer card text (richer, real paragraph) over slim snapshot text.
        text_use = card_text or text
        topic_en = ""
        if en.get("topic"):
            topic_en = en["topic"]
            if en.get("subtopic"):
                topic_en += f" / {en['subtopic']}"
        elif en.get("legal_topic"):
            topic_en = en["legal_topic"]
        return Candidate(
            citation_canon=cit, did=did, family="court", recall_rank=rank,
            text=text_use, paragraph_role=paragraph_role,
            chamber=meta["chamber"], chamber_label=meta["chamber_label"],
            court_code=meta["court_code"], case_importance=meta["case_importance"],
            is_leading_decision=meta["is_leading_decision"],
            has_substantive_role=meta["has_substantive_role"],
            legal_area_static=meta["legal_area_static"],
            doctrinal_rule=en.get("doctrinal_rule", "") or en.get("legal_rule", "") or "",
            legal_test=en.get("legal_test", "") or en.get("court_holding", "") or "",
            topic_en=topic_en,
            statute_anchors=list(en.get("statute_anchors") or []),
        )
    else:
        meta = derive_law_metadata(cit)
        return Candidate(
            citation_canon=cit, did=did, family="law", recall_rank=rank,
            text=text,
            law_abbreviation=meta["law_abbreviation"], law_name_en=meta["law_name_en"],
            title=meta["title"], context_heading_title=meta["context_heading_title"],
            provision_type=meta["provision_type"],
        )


# ───────────────────────────────────────────────────────────────────────
# COURT channel — top-K_COURT from v7.5 final_topk (court family only)
# ───────────────────────────────────────────────────────────────────────
print(f"\\nCourt channel: top-{TOP_K_COURT} from v7.5 final_topk (court only)")
court_cands_by_qid = {}
for qid, snap in pool_snap.items():
    court_dids = [d for d in snap["final_topk"] if d.startswith("court:")]
    cands = []
    for rank, did in enumerate(court_dids[:TOP_K_COURT], 1):
        c = make_candidate(did, rank)
        if c is not None and c.family == "court":
            cands.append(c)
    court_cands_by_qid[qid] = cands
    print(f"  {qid}: court_cands={len(cands)}  (from {len(court_dids)} court dids in pool)")

# ───────────────────────────────────────────────────────────────────────
# LAW channel — BM25 + TLF over corpus.parquet (v12's exact recall stage)
# ───────────────────────────────────────────────────────────────────────
print(f"\\nLaw channel: BM25+TLF → top-{TOP_K_LAW} per query")

# Tokenization (matches v12)
_TOKEN_RE = re.compile(r"[^\\w\\d]+", re.UNICODE)
def tokenise(t):
    return [w for w in _TOKEN_RE.split((t or "").lower().strip())
            if len(w) >= 2 or w.isdigit()]

def build_query_tokens(q, de=None):
    en = tokenise(q)
    if not de: return en
    seen, c = set(), []
    for t in en + tokenise(de):
        if t not in seen:
            seen.add(t); c.append(t)
    return c

# TLF mining from train.csv (v12 method)
print("  Mining TLF from train.csv gold ...")
tlf = defaultdict(Counter)
COURT_GOLD_RE = re.compile(r"^(BGE |ATF |DTF |\\d{1,2}[A-Z]{1,4}[_.]|[IUKHCBPMEF]\\s+\\d)")
if train is not None:
    for _, r in train.iterrows():
        gold_strs = [c.strip() for c in str(r.get("gold_citations") or "").split(";") if c.strip()]
        law_codes = set()
        for g in gold_strs:
            if COURT_GOLD_RE.match(g): continue
            parts = g.split()
            if len(parts) >= 2 and parts[0] in ("Art.", "Artikel"):
                law_codes.add(parts[-1].lower())
        if not law_codes: continue
        for t in set(tokenise(r["query"])):
            for code in law_codes:
                tlf[t][code] += 1
ttc = {t: sum(c.values()) for t, c in tlf.items()}
print(f"  TLF: {len(tlf):,} tokens, {sum(ttc.values()):,} (token,code) pairs")


def enhance(tokens, bm25_idf, tlf, ttc):
    """v12's enhance(): boost top-5 correlated law codes ×5 each."""
    base = list(dict.fromkeys(tokens))
    sc = {}
    for t in set(tokens):
        if t not in tlf: continue
        idf = bm25_idf.get(t, 0.0)
        if idf < 1.0: continue
        tot = ttc.get(t, 1)
        for code, cnt in tlf[t].items():
            sc[code] = sc.get(code, 0.0) + (cnt / tot) * idf
    for code, _ in sorted(sc.items(), key=lambda x: -x[1])[:5]:
        base.extend([code] * 5)
    return base


law_cands_by_qid = {}
# v12's law dossier fields come from corpus.parquet (cd, citation-keyed) + kb
for qid in val["query_id"]:
    qtxt = val[val["query_id"] == qid].iloc[0]["query"]
    raw_tokens = build_query_tokens(qtxt, qt.get(qid))
    enhanced_tokens = enhance(raw_tokens, bm25_law.idf, tlf, ttc)
    scores = bm25_law.get_scores(enhanced_tokens)
    top_idx = scores.argsort()[::-1][:TOP_K_LAW]
    cands = []
    for rank_local, idx in enumerate(top_idx, 1):
        idx = int(idx)
        cit = bm25_law_ids[idx]
        # CITATION-keyed lookup (v12-style) — NOT corpus_law.iloc[idx] which
        # silently scrambled metadata when corpus_law row order != bm25 id order.
        rd = cd.get(cit, {})
        k  = kb.get(cit, {})
        law_info = (k.get("law", {}) or {})
        sem      = (k.get("semantic", {}) or {})
        st       = (k.get("structure", {}) or {})
        la       = str(rd.get("law_abbrev", "") or "")
        did = f"lawbm25:{cit}"
        c = Candidate(
            citation_canon=cit, did=did, family="law", recall_rank=rank_local,
            pool_bm25_score=float(scores[idx]),
            text=str(rd.get("text", "") or "")[:1500],
            title=str(rd.get("title", "") or ""),
            law_abbreviation=la,
            law_name_en=law_info.get("law_name_en", "") or lne.get(la, ""),
            context_heading_title=st.get("context_heading_title", "") or "",
            provision_type=sem.get("provision_type", "") or "",
        )
        cands.append(c)
    law_cands_by_qid[qid] = cands
print(f"  built law candidates for {len(law_cands_by_qid)} queries "
      f"({sum(len(v) for v in law_cands_by_qid.values())} total)")

# ───────────────────────────────────────────────────────────────────────
# Combine into ONE list per qid (for reranker batching efficiency).
# Family stays attached on each Candidate so downstream stages process
# per-family independently.
# ───────────────────────────────────────────────────────────────────────
cands_by_qid = {}
for qid in val["query_id"]:
    cands_by_qid[qid] = list(law_cands_by_qid.get(qid, [])) + list(court_cands_by_qid.get(qid, []))
    n_law = len(law_cands_by_qid.get(qid, []))
    n_court = len(court_cands_by_qid.get(qid, []))
    print(f"  {qid}: total={n_law + n_court}  (law={n_law}, court={n_court})")

# ── Recall check after Stage 1a (combined)
stage_recall_report("Stage 1a — combined candidates (law BM25 + court v7.5)",
                    {q: {c.did for c in cs} for q, cs in cands_by_qid.items()})
# Per-channel recall reports
stage_recall_report("Stage 1a — LAW channel only",
                    {q: {c.did for c in cs if c.family == "law"} for q, cs in cands_by_qid.items()})
stage_recall_report("Stage 1a — COURT channel only",
                    {q: {c.did for c in cs if c.family == "court"} for q, cs in cands_by_qid.items()})
'''))

# ── Cell 9 ─ Stage 1b normalize recall scores per channel
cells.append(md("""## Stage 1b — Per-channel recall-score normalization

Court candidates carry `pool_bm25_score = (n - rank)` from Stage 1a. We also
need that filled for law candidates (BM25 raw score already populated). This
cell just ensures court candidates have proper scores; law candidates already
have `pool_bm25_score = bm25_raw_score`.

Per-family fused-score normalization happens in **Stage 3**, not here.
"""))

cells.append(code('''# ── Stage 1b: set rank-based pool_bm25_score for court candidates.
# Law candidates already have raw BM25 score from Stage 1a; we keep that.
for qid, cands in cands_by_qid.items():
    court_cs = [c for c in cands if c.family == "court"]
    n = len(court_cs)
    for i, c in enumerate(court_cs):
        # rank-derived score so deeper court ranks decay linearly
        c.pool_bm25_score = float(n - i)
    n_law = sum(1 for c in cands if c.family == "law")
    print(f"  {qid}: law={n_law} (raw BM25 scores), court={n} (rank-derived)")
'''))

# ── Cell 10 ─ Reranker
cells.append(md("""## Stage 2 — Court-aware Qwen3-Reranker

Load Qwen3-Reranker-8B in bf16 (~16 GB VRAM on 95.6 GB). Batch 16. Each
candidate is scored as `softmax([logit_no, logit_yes])[1]` from the model's
last-token logits. Cached to `reranker_v13_cache.json` so re-runs skip done qids.
"""))

cells.append(code('''# ── Stage 2: Qwen3-Reranker-8B with court/law branched dossier
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_reranker(model_name=RERANKER_MODEL, dev="auto"):
    print(f"Loading reranker: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
                                        padding_side="left")
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=dev,
        trust_remote_code=True)
    mdl.eval()
    pfx = tok.encode(
        '<|im_start|>system\\nJudge whether the Document meets the '
        'requirements based on the Query and the Instruct provided. '
        'Note that the answer can only be "yes" or "no".'
        '<|im_end|>\\n<|im_start|>user\\n', add_special_tokens=False)
    sfx = tok.encode('<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n',
                     add_special_tokens=False)
    yi = tok.convert_tokens_to_ids("yes")
    ni = tok.convert_tokens_to_ids("no")
    print(f"  yes={yi}  no={ni}  VRAM={torch.cuda.memory_allocated()/1e9:.1f} GB")
    return tok, mdl, pfx, sfx, yi, ni


def rerank_batch(query, cands, tok, mdl, pfx, sfx, yi, ni, bs=16, ml=4096):
    pairs = [_fmt_rr(query, c) for c in cands]
    all_sc = []
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i + bs]
        inp = tok(batch, padding=False, truncation="longest_first",
                  return_attention_mask=False,
                  max_length=ml - len(pfx) - len(sfx))
        for j in range(len(inp["input_ids"])):
            inp["input_ids"][j] = pfx + inp["input_ids"][j] + sfx
        inp = tok.pad(inp, padding=True, return_tensors="pt")
        inp = {k: v.to(mdl.device) for k, v in inp.items()}
        with torch.no_grad():
            logits = mdl(**inp).logits[:, -1, :]
        st = torch.stack([logits[:, ni], logits[:, yi]], dim=1)
        sc = torch.nn.functional.log_softmax(st, dim=1)[:, 1].exp().cpu().tolist()
        all_sc.extend(sc)
    for c, s in zip(cands, all_sc):
        c.reranker_score = float(s)


# Cache + run
rcache = {}
if RERANKER_CACHE.exists():
    try:
        rcache = json.loads(RERANKER_CACHE.read_text(encoding="utf-8"))
        print(f"  loaded cache: {len(rcache)} qids")
    except Exception:
        pass

# Coverage check: invalidate cache entries that don't cover the current
# candidate set (prevents the stale-cache silent-failure mode).
STALE_THRESHOLD = 0.80   # require >=80% candidate coverage to trust cache
stale_qids = []
for qid, cands in cands_by_qid.items():
    sm = rcache.get(qid)
    if sm is None: continue
    coverage = sum(1 for c in cands if c.did in sm) / max(len(cands), 1)
    if coverage < STALE_THRESHOLD:
        stale_qids.append((qid, coverage))
if stale_qids:
    print(f"  WARNING: {len(stale_qids)} qids have stale cache "
          f"(<{int(STALE_THRESHOLD*100)}% candidate coverage):")
    for qid, cov in stale_qids:
        print(f"    {qid}: {cov*100:.1f}% covered — invalidating")
        rcache.pop(qid, None)

need = [q for q in cands_by_qid if q not in rcache]
if need:
    avg_per_q = sum(len(cands_by_qid[q]) for q in need) / max(len(need), 1)
    print(f"  scoring {len(need)} queries × ~{int(avg_per_q)} candidates "
          f"(law+court per qid) = ~{int(len(need)*avg_per_q):,} pairs ...")
    rtok, rmdl, rpfx, rsfx, ryi, rni = load_reranker()
    t0 = time.time()
    for qid in tqdm(need, desc="reranking"):
        qtxt = val[val["query_id"] == qid].iloc[0]["query"]
        rerank_batch(qtxt, cands_by_qid[qid], rtok, rmdl,
                     rpfx, rsfx, ryi, rni, bs=RERANKER_BATCH)
        rcache[qid] = {c.did: c.reranker_score for c in cands_by_qid[qid]}
        RERANKER_CACHE.write_text(json.dumps(rcache, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(f"  reranker done in {time.time()-t0:.1f}s")
    del rmdl, rtok
    gc.collect(); torch.cuda.empty_cache()
else:
    print("  all qids cached ✓ (coverage validated)")

# Apply cache
for qid, cands in cands_by_qid.items():
    sm = rcache.get(qid, {})
    for c in cands:
        c.reranker_score = float(sm.get(c.did, 0.0))
'''))

# ── Cell 11 ─ Fused + zone
cells.append(md("""## Stage 3 — Fused score + zone split

`fused_score = 0.7 · norm(pool_BM25) + 0.3 · P(yes)` per v12. Zones: ≥0.55 →
auto-YES, <0.25 → auto-NO, else borderline. Cheap mass cleanup; the judge only
sees the borderline pool.
"""))

cells.append(code('''# ── Stage 3: fused score + zone split, PER FAMILY (no cross-family mixing)

stats = Counter()
for qid, cands in cands_by_qid.items():
    if not cands: continue
    # Per-family normalization — law BM25 scores and court rank scores have
    # different scales, so we min-max each family separately.
    for family in ("law", "court"):
        fam_cands = [c for c in cands if c.family == family]
        if not fam_cands: continue
        bsc = [c.pool_bm25_score for c in fam_cands]
        bmin, bmax = min(bsc), max(bsc)
        brng = max(bmax - bmin, 1e-9)
        low_thresh = LOW_THRESH_LAW if family == "law" else LOW_THRESH_COURT
        for c in fam_cands:
            c.bm25_norm_score = (c.pool_bm25_score - bmin) / brng
            c.fused_score = (FUSED_W_RECALL * c.bm25_norm_score
                             + FUSED_W_RERANK * c.reranker_score)
            if c.fused_score >= HIGH_THRESH:
                c.zone = "yes"
            elif c.fused_score < low_thresh:
                c.zone = "no"
            else:
                c.zone = "borderline"
            stats[c.zone] += 1
            stats[f"{c.zone}_{c.family}"] += 1

print(f"Total: auto-YES={stats['yes']}  borderline={stats['borderline']}  auto-NO={stats['no']}")
for z in ("yes", "borderline", "no"):
    print(f"  {z}: court={stats.get(f'{z}_court',0):>5}  law={stats.get(f'{z}_law',0):>5}")

# ── Recall check after Stage 3: gold in auto-YES / borderline / lost to auto-NO
print("\\n--- Stage 3 recall breakdown (gold by zone) ---")
auto_yes_dids = {q: {c.did for c in cs if c.zone == "yes"} for q, cs in cands_by_qid.items()}
borderline_dids = {q: {c.did for c in cs if c.zone == "borderline"} for q, cs in cands_by_qid.items()}
yes_or_border = {q: auto_yes_dids[q] | borderline_dids[q] for q in cands_by_qid}

stage_recall_report("Stage 3 — auto-YES zone alone", auto_yes_dids)
stage_recall_report("Stage 3 — auto-YES + borderline (max possible after Stage 4)", yes_or_border)

# Gold lost to auto-NO is the most painful — un-recoverable downstream
auto_no_lost = {}
for qid in val["query_id"]:
    gold = gold_map[qid]["gold_dids"]
    lost = gold - yes_or_border.get(qid, set())
    auto_no_lost[qid] = lost
total_lost = sum(len(v) for v in auto_no_lost.values())
total_gold = sum(len(gold_map[q]["gold_dids"]) for q in val["query_id"])
print(f"\\n  Gold DROPPED to auto-NO (unrecoverable): {total_lost}/{total_gold} "
      f"= {total_lost/max(total_gold,1)*100:.1f}%")
if total_lost > 0:
    report_dropped_gold("Stage 3 auto-NO loss", yes_or_border, max_per_qid=3)
'''))

# ── Cell 12 ─ Judge
cells.append(md("""## Stage 4 — Court-aware Qwen3-8B judge on borderline only

Per-query, two judge calls (one per family) using `JUDGE_SYSTEM_LAW` /
`JUDGE_SYSTEM_COURT`. Default to YES on parse failure (recall-preserving
fallback). Cached so retries skip done qids.
"""))

cells.append(code('''# ── Stage 4: Qwen3-8B judge (per-family) for borderline only

def load_judge(model_name=JUDGE_MODEL, dev="auto"):
    print(f"Loading judge: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=dev,
        trust_remote_code=True)
    mdl.eval()
    print(f"  VRAM={torch.cuda.memory_allocated()/1e9:.1f} GB")
    return tok, mdl


def judge_one_family(query, cands, family, tok, mdl):
    if not cands: return {}
    system = JUDGE_SYSTEM_COURT if family == "court" else JUDGE_SYSTEM_LAW
    user = build_judge_prompt(query, cands, family)
    messages = [{"role": "system", "content": system},
                {"role": "user",   "content": user}]
    text = tok.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=True)
    inputs = tok(text, return_tensors="pt", truncation=True,
                 max_length=20000).to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=3000,
                           do_sample=True, temperature=0.5, top_k=20, top_p=0.95,
                           pad_token_id=tok.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    raw = tok.decode(gen, skip_special_tokens=True).strip()
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    verdicts = {}
    valid = {c.citation_canon for c in cands}
    valid_lower = {c.lower(): c for c in valid}
    for line in raw.split("\\n"):
        line = line.strip()
        if "|" not in line: continue
        parts = line.split("|")
        cit_part = parts[0].strip().lstrip("[0-9] ").strip()
        verdict_part = parts[-1].strip().upper()
        v = "YES" if "YES" in verdict_part else "NO"
        if cit_part in valid:
            verdicts[cit_part] = v
        elif cit_part.lower() in valid_lower:
            verdicts[valid_lower[cit_part.lower()]] = v
    for c in cands:
        verdicts.setdefault(c.citation_canon, "YES")
    return verdicts


jcache = {}
if JUDGE_CACHE.exists():
    try:
        jcache = json.loads(JUDGE_CACHE.read_text(encoding="utf-8"))
        print(f"  loaded cache: {len(jcache)} qids")
    except Exception:
        pass

# Coverage check: invalidate cached qids whose borderline count differs.
# (If upstream pool / thresholds changed, the borderline set is different,
# so the verdicts are stale even if the qid is "cached".)
JUDGE_COVERAGE_TOL = 0.20   # allow ±20% drift in borderline counts
stale_jqids = []
for qid, cands in cands_by_qid.items():
    j = jcache.get(qid)
    if j is None: continue
    cur_bl_court = sum(1 for c in cands if c.zone == "borderline" and c.family == "court")
    cur_bl_law   = sum(1 for c in cands if c.zone == "borderline" and c.family == "law")
    old_bl_court = j.get("n_borderline_court", 0)
    old_bl_law   = j.get("n_borderline_law",   0)
    drift_c = abs(cur_bl_court - old_bl_court) / max(cur_bl_court, 1)
    drift_l = abs(cur_bl_law   - old_bl_law)   / max(cur_bl_law,   1)
    if drift_c > JUDGE_COVERAGE_TOL or drift_l > JUDGE_COVERAGE_TOL:
        stale_jqids.append((qid, cur_bl_court, old_bl_court, cur_bl_law, old_bl_law))
if stale_jqids:
    print(f"  WARNING: {len(stale_jqids)} qids have stale judge cache "
          f"(borderline drift > {int(JUDGE_COVERAGE_TOL*100)}%):")
    for qid, cc, oc, cl, ol in stale_jqids:
        print(f"    {qid}: court {oc}->{cc}, law {ol}->{cl} — invalidating")
        jcache.pop(qid, None)

need = [q for q in cands_by_qid if q not in jcache]
if need:
    jtok, jmdl = load_judge()
    t0 = time.time()
    for qid in tqdm(need, desc="judging"):
        cands = cands_by_qid[qid]
        bl_court = [c for c in cands if c.zone == "borderline" and c.family == "court"]
        bl_law   = [c for c in cands if c.zone == "borderline" and c.family == "law"]
        qtxt = val[val["query_id"] == qid].iloc[0]["query"]
        v_court = judge_one_family(qtxt, bl_court, "court", jtok, jmdl) if bl_court else {}
        v_law   = judge_one_family(qtxt, bl_law,   "law",   jtok, jmdl) if bl_law   else {}
        jcache[qid] = {
            "verdicts_court": v_court, "verdicts_law": v_law,
            "n_borderline_court": len(bl_court), "n_borderline_law": len(bl_law),
        }
        JUDGE_CACHE.write_text(json.dumps(jcache, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print(f"  judge done in {time.time()-t0:.1f}s")
    del jmdl, jtok
    gc.collect(); torch.cuda.empty_cache()
else:
    print("  all qids cached ✓ (coverage validated)")
'''))

# ── Cell 13 ─ Stage 5 adaptive K predictions
cells.append(md("""## Stage 5 — Adaptive-K final predictions

For each query:
1. Run `predict_k(query)` → `K_total` (val-cascade → 30-50, train-substantive → 2-5)
2. Collect auto-YES + borderline-YES (from judge verdicts)
3. Sort by `fused_score` descending, take top-K_total
4. Emit three submissions: `adaptive`, `adaptive × 1.3`, `fixed K=5`

The adaptive predictor is the dominant precision lever discovered in the
analysis. v12 used fixed K=12 (val-tuned), which is good for val_001/002/003
but ruinous for substantive-only queries.
"""))

cells.append(code('''# ── Stage 5: TWO-CHANNEL adaptive K — per-family selection + UNION

predictions_adaptive  = {}     # union of law-top-K_law + court-top-K_court
predictions_x13       = {}     # same but K × 1.3
predictions_k5        = {}     # fixed K=5 union (3 law + 2 court)
predictions_law_only  = {}     # law channel alone (for ablation)
predictions_court_only= {}     # court channel alone (for ablation)
diagnostics = []

for qid, cands in cands_by_qid.items():
    qtxt = val[val["query_id"] == qid].iloc[0]["query"]
    k_info = predict_k(qtxt)
    K_pred = k_info["K_pred"]
    jc = jcache.get(qid, {})
    v_court = jc.get("verdicts_court", {})
    v_law   = jc.get("verdicts_law",   {})

    # Per-family selected pool (auto-YES + borderline-judged-YES), each sorted by fused
    law_pool = []
    for c in sorted([x for x in cands if x.family == "law"], key=lambda x: -x.fused_score):
        if c.zone == "yes":
            law_pool.append(c.citation_canon)
        elif c.zone == "borderline" and v_law.get(c.citation_canon, "YES") == "YES":
            law_pool.append(c.citation_canon)

    court_pool = []
    for c in sorted([x for x in cands if x.family == "court"], key=lambda x: -x.fused_score):
        if c.zone == "yes":
            court_pool.append(c.citation_canon)
        elif c.zone == "borderline" and v_court.get(c.citation_canon, "YES") == "YES":
            court_pool.append(c.citation_canon)

    # Per-family K — INDEPENDENT predictors from predict_k().
    # K_law calibrated to expected law-gold count for this query's cascade tier.
    # K_court calibrated to expected court-gold count for this query's cascade tier.
    K_law   = k_info["K_law"]
    K_court = k_info["K_court"]
    sel_law   = law_pool[:K_law]
    sel_court = court_pool[:K_court]

    # ── Variant A: adaptive UNION (K_law law preds ∪ K_court court preds)
    sel_A = list(dict.fromkeys(sel_law + sel_court))
    # ── Variant B: adaptive × 1.3 UNION
    K_law_hi   = int(K_law * 1.3)
    K_court_hi = int(K_court * 1.3)
    sel_B = list(dict.fromkeys(law_pool[:K_law_hi] + court_pool[:K_court_hi]))
    # ── Variant C: fixed K=5 UNION (3 law + 2 court)
    sel_C = list(dict.fromkeys(law_pool[:3] + court_pool[:2]))

    # Channel-only variants (for ablation/debugging)
    predictions_law_only[qid]   = list(sel_law)
    predictions_court_only[qid] = list(sel_court)

    # Safety: never emit empty
    sorted_all = sorted(cands, key=lambda x: -x.fused_score)
    if not sel_A and sorted_all: sel_A = [sorted_all[0].citation_canon]
    if not sel_B and sorted_all: sel_B = [sorted_all[0].citation_canon]
    if not sel_C and sorted_all: sel_C = [sorted_all[0].citation_canon]

    predictions_adaptive[qid] = sel_A
    predictions_x13[qid]      = sel_B
    predictions_k5[qid]       = sel_C
    diagnostics.append({
        "qid": qid, **k_info,
        "n_law_yes": sum(1 for c in cands if c.zone == "yes" and c.family == "law"),
        "n_law_border": sum(1 for c in cands if c.zone == "borderline" and c.family == "law"),
        "n_court_yes": sum(1 for c in cands if c.zone == "yes" and c.family == "court"),
        "n_court_border": sum(1 for c in cands if c.zone == "borderline" and c.family == "court"),
        "law_pool_size":   len(law_pool),
        "court_pool_size": len(court_pool),
        "K_law":            len(sel_law),
        "K_court":          len(sel_court),
        "union_size":       len(sel_A),
    })

diag = pd.DataFrame(diagnostics)
print(diag.to_string(index=False))


# ── Oracle-K sweep — what F1 ceiling is achievable from the cached fused scores?
# Pure analysis: no GPU. Reveals whether the new predict_k targets are right.
val_gold_sets = {r["query_id"]: set(c.strip() for c in str(r["gold_citations"]).split(";") if c.strip())
                 for _, r in val.iterrows()}
K_grid = [5, 10, 15, 20, 25, 30, 35, 40, 50]
print("\\nOracle K sweep (F1 at each K, best K* per query):")
print(f"  {'qid':<10} {'gold':>5} " + " ".join(f"K={k:<2}" for k in K_grid) + "    K*  F1*")
print("  " + "-" * 92)
oracle_f1s = []
for qid in val["query_id"]:
    cands = cands_by_qid[qid]
    jc = jcache.get(qid, {})
    v_court = jc.get("verdicts_court", {})
    v_law   = jc.get("verdicts_law",   {})
    pool = []
    for c in sorted(cands, key=lambda x: -x.fused_score):
        if c.zone == "yes":
            pool.append(c.citation_canon)
        elif c.zone == "borderline":
            pv = v_court if c.family == "court" else v_law
            if pv.get(c.citation_canon, "YES") == "YES":
                pool.append(c.citation_canon)
    gold = val_gold_sets[qid]
    best_f1, best_k = 0.0, 0
    f1s = []
    for k in K_grid:
        pred = set(pool[:k])
        tp = len(pred & gold)
        p = tp / max(len(pred), 1); rec = tp / max(len(gold), 1)
        ff = 2 * p * rec / max(p + rec, 1e-9)
        f1s.append(ff)
        if ff > best_f1:
            best_f1, best_k = ff, k
    oracle_f1s.append(best_f1)
    print(f"  {qid:<10} {len(gold):>5} " + " ".join(f"{v:>4.2f}" for v in f1s) +
          f"   {best_k:>3} {best_f1:>5.3f}")
print(f"\\n  MACRO oracle F1 (best K* per query): "
      f"{sum(oracle_f1s)/max(len(oracle_f1s),1):.3f}")


# ── Recall check after Stage 5: each variant's selected dids
# Reconstruct dids per variant for the recall report (uses corpus_snap)
_cit_to_did = {}
for d, r in corpus_snap.items():
    c = r.get("cit")
    if c:
        _cit_to_did.setdefault(c, d)


def cit_set_to_did_set(qid, cit_set):
    """Map predicted citation_canon strings back to dids via corpus_snap."""
    return {_cit_to_did[c] for c in cit_set if c in _cit_to_did}


for preds, name in [(predictions_adaptive, "Stage 5 — adaptive K"),
                    (predictions_x13,      "Stage 5 — adaptive × 1.3"),
                    (predictions_k5,       "Stage 5 — fixed K=5")]:
    pred_dids = {q: cit_set_to_did_set(q, set(preds.get(q, []))) for q in val["query_id"]}
    stage_recall_report(name, pred_dids)
'''))

# ── Stage 5.5 ─ case-peer + article-sibling expansion (post-processing)
cells.append(md("""## Stage 5.5 — Case-peer + article-sibling expansion

Single highest-leverage post-processing step. When the reranker finds ONE
paragraph of a multi-paragraph gold case (e.g. `BGE 137 IV 122 E. 4.1`),
expansion pulls 2-3 sibling Es of the same case (`E. 4.2`, `E. 6.2`, `E. 6.4`)
from `corpus_snap`. Same idea for law: `Art. 221 Abs. 1 StPO` → also predict
`Art. 221 Abs. 2 StPO`, `Art. 221 Abs. 3 StPO`.

Builds two extra submission variants: `adaptive_expanded` and `adaptive_x1.3_expanded`.
"""))

cells.append(code('''# ── Stage 5.5: case-peer + article-sibling expansion (no GPU, ~20 s)

# Parse a court citation into its case base ("BGE 137 IV 122" or "1B_210/2023")
COURT_CASE_BASE_RE = re.compile(
    r"^\\s*(BGE\\s+\\d{1,4}\\s+[IVXLC]{1,5}\\s+\\d+[a-z]?"
    r"|\\d{1,2}[A-Z]{1,4}[_.]\\d{1,6}/\\d{4}(?:\\s+\\d{1,2}\\.\\d{1,2}\\.\\d{4})?"
    r"|[IUKHCBPMEF]\\s+\\d{1,5}/\\d{2,4})"
)
# Parse a law citation into (article, code) — siblings share both
LAW_ARTICLE_RE = re.compile(
    r"^\\s*Art\\.\\s+(\\d+[a-zA-Z]*)"
    r"(?:\\s+Abs\\.\\s+\\d+[a-zA-Z]*)?"
    r"(?:\\s+lit\\.\\s+\\w+)?"
    r"(?:\\s+Ziff\\.\\s+\\w+)?"
    r"\\s+([A-ZÄÖÜ][A-Za-zÄÖÜäöüçèéà0-9./_-]+)\\s*$"
)


def _is_court_cit(cit: str) -> bool:
    return (cit.startswith(("BGE ", "ATF ", "DTF "))
            or bool(re.match(r"^\\d{1,2}[A-Z]{1,4}[_.]", cit))
            or bool(re.match(r"^[IUKHCBPMEF]\\s+\\d", cit)))


print("Building expansion indices from corpus_snap ...")
t0 = time.time()
case_index = defaultdict(list)        # case_base -> list[(did, cit)]
article_index = defaultdict(list)     # (article_num, code) -> list[(did, cit)]
for did, rec in corpus_snap.items():
    cit = rec.get("cit") or ""
    if not cit: continue
    if _is_court_cit(cit):
        m = COURT_CASE_BASE_RE.match(cit)
        if m:
            base = m.group(1)
            case_index[base].append((did, cit))
    else:
        m = LAW_ARTICLE_RE.match(cit)
        if m:
            key = (m.group(1), m.group(2))
            article_index[key].append((did, cit))
print(f"  case_index:    {len(case_index):,} cases  "
      f"({sum(len(v) for v in case_index.values()):,} paragraphs)")
print(f"  article_index: {len(article_index):,} articles  "
      f"({sum(len(v) for v in article_index.values()):,} Abs.-rows)")
print(f"  built in {time.time()-t0:.1f}s")


def expand_predictions(preds_for_qid: list[str],
                       max_case_peers: int = 3,
                       max_article_peers: int = 2) -> list[str]:
    """Add sibling court-paragraph and law-Abs. paragraphs to predictions."""
    expanded = list(preds_for_qid)
    seen = set(preds_for_qid)
    # Court-side case-peer expansion
    for cit in preds_for_qid:
        if not _is_court_cit(cit): continue
        m = COURT_CASE_BASE_RE.match(cit)
        if not m: continue
        peers = case_index.get(m.group(1), [])
        added = 0
        for _, peer_cit in peers:
            if peer_cit != cit and peer_cit not in seen and added < max_case_peers:
                expanded.append(peer_cit); seen.add(peer_cit); added += 1
    # Law-side article-sibling expansion
    for cit in preds_for_qid:
        if _is_court_cit(cit): continue
        m = LAW_ARTICLE_RE.match(cit)
        if not m: continue
        peers = article_index.get((m.group(1), m.group(2)), [])
        added = 0
        for _, peer_cit in peers:
            if peer_cit != cit and peer_cit not in seen and added < max_article_peers:
                expanded.append(peer_cit); seen.add(peer_cit); added += 1
    return expanded


# Build expanded variants
predictions_adaptive_expanded = {qid: expand_predictions(preds)
                                  for qid, preds in predictions_adaptive.items()}
predictions_x13_expanded      = {qid: expand_predictions(preds)
                                  for qid, preds in predictions_x13.items()}

# Per-query expansion stats
print(f"\\n  {'qid':<10} {'base_K':>7} {'+peers':>7} {'final':>6}  expansion details")
for qid in val["query_id"]:
    base = predictions_adaptive.get(qid, [])
    expd = predictions_adaptive_expanded.get(qid, [])
    print(f"  {qid:<10} {len(base):>7} {len(expd)-len(base):>7} {len(expd):>6}")

# Recall check on expanded variants
print()
for preds, name in [(predictions_adaptive_expanded,  "Stage 5.5 — adaptive expanded"),
                    (predictions_x13_expanded,       "Stage 5.5 — adaptive × 1.3 expanded")]:
    pred_dids = {q: cit_set_to_did_set(q, set(preds.get(q, []))) for q in val["query_id"]}
    stage_recall_report(name, pred_dids)
'''))

# ── Cell 14 ─ Eval + submission
cells.append(md("""## Stage 6 — Eval on VAL + submission CSVs

Eval uses citation-string match (semicolon-separated gold from `val.csv`).
Output five CSV files: adaptive, adaptive_expanded, ×1.3, ×1.3_expanded, K=5 fallback.
"""))

cells.append(code('''# ── Eval + submission

def compute_f1(pred, gold):
    if not pred and not gold: return 1., 1., 1.
    if not pred or not gold: return 0., 0., 0.
    pred, gold = set(pred), set(gold)
    tp = len(pred & gold)
    p = tp / len(pred); r = tp / len(gold)
    return p, r, (2*p*r/(p+r) if p+r > 0 else 0.)


def evaluate(preds, label):
    print(f"\\n=== {label} ===")
    print(f"  {'qid':<10} {'pred':>5} {'gold':>5} {'P':>6} {'R':>6} {'F1':>6}")
    f1s = []
    for _, r in val.iterrows():
        qid = r["query_id"]
        if qid not in preds: continue
        gold = set(c.strip() for c in str(r.get("gold_citations") or "").split(";") if c.strip())
        pred = set(preds.get(qid, []))
        p, rec, f = compute_f1(pred, gold)
        f1s.append(f)
        print(f"  {qid:<10} {len(pred):>5} {len(gold):>5} {p:>6.3f} {rec:>6.3f} {f:>6.3f}")
    if f1s:
        print(f"  {'MACRO':<10} {'':>5} {'':>5} {'':>6} {'':>6} {sum(f1s)/len(f1s):>6.3f}")


evaluate(predictions_law_only,           "law channel only (ablation)")
evaluate(predictions_court_only,         "court channel only (ablation)")
evaluate(predictions_adaptive,           "UNION (adaptive K)")
evaluate(predictions_adaptive_expanded,  "UNION (adaptive K) + expansion")
evaluate(predictions_x13,                "UNION (adaptive K × 1.3)")
evaluate(predictions_x13_expanded,       "UNION (adaptive K × 1.3) + expansion")
evaluate(predictions_k5,                 "UNION fixed K=5 (fallback)")


def save_submission(preds, name):
    rows = []
    for _, r in val.iterrows():
        qid = r["query_id"]
        rows.append({"query_id": qid,
                     "predicted_citations": ";".join(preds.get(qid, []))})
    df = pd.DataFrame(rows)
    p = OUT_DIR / f"submission_v13_{name}.csv"
    df.to_csv(p, index=False)
    return p


for preds, name in [(predictions_law_only,          "law_only_ablation"),
                    (predictions_court_only,        "court_only_ablation"),
                    (predictions_adaptive,          "union_adaptive"),
                    (predictions_adaptive_expanded, "union_adaptive_expanded"),
                    (predictions_x13,               "union_adaptive_x1.3"),
                    (predictions_x13_expanded,      "union_adaptive_x1.3_expanded"),
                    (predictions_k5,                "union_k5_fallback")]:
    p = save_submission(preds, name)
    print(f"  saved: {p}")

# Also save diagnostics + summary
diag.to_csv(OUT_DIR / "v13_diagnostics.csv", index=False)
print(f"  saved: {OUT_DIR / 'v13_diagnostics.csv'}")
'''))

# ── Cell 14b ─ Final funnel summary
cells.append(md("""## Recall funnel summary — where gold is lost

One table per qid showing recall at every stage. Where recall drops sharply
between two stages, that's the stage to fix:

- LAW channel recall drop: BM25+TLF missing gold — bump `TOP_K_LAW` from 100 to 200, or check TLF coverage on the val_001 codes
- COURT channel recall drop: v7.5 pool too narrow — bump `TOP_K_COURT` from 500 to 1000
- Stage 1a → Stage 3 auto-YES drop: reranker is missing relevant candidates — relax `HIGH_THRESH` to 0.45 or check dossier rendering
- Auto-YES + borderline → Stage 5 drop: adaptive K is cutting too aggressively — try `× 1.3` variant
- Stage 5 adaptive < Stage 5 × 1.3: predict_k under-estimates; check K_law / K_court formula vs gold counts
"""))

cells.append(code('''# ── Recall funnel summary

funnel_df = pd.DataFrame(recall_funnel)
# Pivot to qid × stage matrix
pivot = funnel_df.pivot_table(index="qid", columns="stage", values="recall",
                                aggfunc="mean").round(3)
# Reorder columns by typical stage order. The Stage 1a / Stage 5 stage names
# come from stage_recall_report() calls earlier — match those strings exactly.
stage_order = [
    "Stage 1a — combined candidates (law BM25 + court v7.5)",
    "Stage 1a — LAW channel only",
    "Stage 1a — COURT channel only",
    "Stage 3 — auto-YES zone alone",
    "Stage 3 — auto-YES + borderline (max possible after Stage 4)",
    "Stage 5 — adaptive K",
    "Stage 5 — adaptive × 1.3",
    "Stage 5 — fixed K=5",
    "Stage 5.5 — adaptive expanded",
    "Stage 5.5 — adaptive × 1.3 expanded",
]
present = [s for s in stage_order if s in pivot.columns]
pivot = pivot[present]
print("Recall funnel (rows = qid, cols = stage):")
print(pivot.to_string())

# Macro per stage
print("\\nMacro recall per stage:")
for stage in present:
    macro = funnel_df[funnel_df["stage"] == stage]["recall"].mean()
    print(f"  {stage:<60} {macro:.3f}")

# Save
funnel_df.to_csv(OUT_DIR / "v13_recall_funnel.csv", index=False)
pivot.to_csv(OUT_DIR / "v13_recall_funnel_pivot.csv")
print(f"\\n  saved: {OUT_DIR / 'v13_recall_funnel.csv'}")
print(f"  saved: {OUT_DIR / 'v13_recall_funnel_pivot.csv'}")
'''))

# ── Cell 15 ─ closing
cells.append(md("""## Notes

### Three-variant hedge

You're submitting three CSVs. Read the leaderboard pattern:
- If `adaptive_x1.3` wins → real test is more cascade-heavy than val (rare)
- If `adaptive` wins → real ≈ val (likely if the test is auto-translated to English)
- If `k5_fallback` wins → real ≈ train (German, single-issue) — revisit K formula

### Auto-resume

`reranker_v13_cache.json` and `judge_v13_cache.json` are written incrementally.
If the Colab session dies mid-run, re-run the notebook — it skips done qids.

### If you want to test on training queries

Generate a v7.5 pool snapshot for train queries (same funnel run, write to
`per_query_snapshot_train.json`). Swap `POOL_SNAP_PATH` and `VAL_CSV_PATH`
accordingly. Be aware: train queries trigger train-style K predictions
(K=2-5), so don't expect val-level F1.

### Test set extension

For the real Kaggle test set, you'll need a v7.5 pool snapshot computed on the
test queries (`test.csv`). Run the funnel on test queries to produce
`per_query_snapshot_test.json`, then point this notebook at it.
"""))


# ────────────────────────────────────────────────────────────────────────
# Cell 16 — v12 reference architecture (what we're building on)
# ────────────────────────────────────────────────────────────────────────
cells.append(md(r"""# Appendix A — v12 reference pipeline (F1 = 0.777 on law-only val)

The v12 hybrid pipeline (from `swiss_legal_citation_pipeline_experiments.ipynb`,
Experiment 3) is the baseline this notebook extends. It hit **F1 = 0.777 on
val** measured against **law-only gold** (court citations filtered out via
`re.match(r"^Art\.\s+\d", c)`). The same pipeline scores **F1 = 0.296 on
train** (Δ = +0.481 — val is leaked / distribution-shifted).

## Inputs

| Artefact | Purpose |
|---|---|
| `data/val.csv`, `data/train.csv` | queries + gold citation strings |
| `retrieval/corpus.parquet` (171,654 rows) | LAW corpus — one row per article/paragraph, with `bm25_text` curated field |
| `retrieval/laws_knowledge_base.jsonl` | extra law metadata (law_name_en, provision_type, context_heading_title) |
| `retrieval/bm25_v2_index.pkl` (108 MB) | pre-built BM25Okapi index over `corpus["bm25_text"]` |
| `retrieval/bm25_v2_ids.pkl` (4 MB) | citation_canon ordering parallel to BM25 index |
| `retrieval/knowledge_base_optimized_hybrid_retrieval/query_translations_trainval.json` | DE translation of val queries (bilingual token augmentation) |

## Models

| Stage | Model | Precision | VRAM |
|---|---|---|---:|
| Reranker | `Qwen/Qwen3-Reranker-8B` | bf16 | ~16 GB |
| LLM Judge | `Qwen/Qwen3-8B` | bf16 | ~16 GB |

## Stage-by-stage mechanism

### Stage 1 — BM25 retrieval (top-100)

1. **Tokenization** of English query with `_TOKEN_RE = r"[^\w\d]+"`, lowercased, ≥2 chars.
2. **Bilingual augmentation**: concatenate tokens from `query_translations_trainval.json` German translation.
3. **TLF mining** (one-time, from train.csv):
   ```python
   for each train row:
     resolve gold → set of corpus citation_canon
     abbreviations = {last token of each gold}
     for each token t in query: tlf[t][abbrev] += 1
   ```
4. **Enhance** (per-query):
   - For each query token `t` present in TLF, score each correlated abbreviation `a` by `(tlf[t][a]/ttc[t]) * idf[t]`.
   - Take top-5 abbreviations by aggregated score; **repeat each ×5** in the query token list.
5. **BM25Okapi.get_scores(enhanced_tokens)** over the full 171K corpus → take top-100 ranked docs.
6. **Recall claim**: `recall@100 ≈ 0.90` on val-law-only gold.

### Stage 2 — Qwen3-Reranker-8B (pointwise yes/no scoring)

For each (query, candidate) pair:
- Build a document string: `citation_canon\nLaw: abbrev — name_en\nTitle: ...\nHeading: ...\nType: ...\nText: ...[:500]`.
- Wrap in instruction template:
  ```
  <Instruct>: Given a query about Swiss law, determine whether the provided
              law article is related to or applicable to the legal issue.
  <Query>: <user query>
  <Document>: <document>
  ```
- Score = `softmax([logit(no), logit(yes)])[1]` from the model's last-token logits.

### Stage 3 — Fused score + 3-zone split

```python
fused_score = 0.7 * norm_minmax(bm25_score) + 0.3 * P(yes)
```
Zone boundaries: `HIGH = 0.55`, `LOW = 0.25`.
- `fused ≥ 0.55` → **auto-YES** (skip LLM judge)
- `fused < 0.25` → **auto-NO** (skip LLM judge)
- else → **borderline** → LLM judge

Empirically ≈ 5 auto-YES + ≈ 80 auto-NO + ≈ 15 borderline per query.

### Stage 4 — Qwen3-8B Judge (borderline only)

System prompt embeds the **Swiss 7-category citation framework**:
1. Substantive law (StGB / OR / ZGB articles)
2. Definitions
3. Procedural rules (StPO / ZPO articles)
4. Appeal provisions (Beschwerde)
5. Cost allocation (Art. 422, 428 StPO; Art. 64 BGG)
6. Court jurisdiction (Art. 37/39 StBOG, Art. 100 BGG)
7. Constitutional principles (Art. 29 BV, proportionality)

Borderline candidates rendered in one prompt:
```
[1] Art. 221 Abs. 1 StPO
    Law: StPO — Schweizerische Strafprozessordnung
    German text: ...

[2] ...
```
Generation: `temperature=0.5, top_k=20, top_p=0.95, max_new_tokens=3000`, thinking enabled.

Verdicts parsed via `CITATION | VERDICT: YES/NO` lines. Default **YES** on parse failure (recall-preserving fallback).

### Stage 5 — Final predictions

```python
predictions = {c for c in cands if c.zone == "yes"} ∪ {c for c in borderline if verdict == "YES"}
```
**No K cutoff** in v12 — it submits everything that passed.

### Performance

| Method | VAL F1 | TRAIN F1 | Δ | Honest avg |
|---|---:|---:|---:|---:|
| Hybrid v12 (full) | **0.777** | 0.296 | +0.481 | 0.537 |
| Threshold-only (no judge) | 0.681 | 0.570 | +0.111 ✓ | 0.625 |
| BM25 K=12 (no reranker) | 0.714 | 0.146 | +0.568 ⚠ | 0.430 |
| BM25 K=5 (train-honest) | 0.550 | 0.255 | +0.295 ⚠ | 0.403 |

**Honest read** (from the reference notebook's own conclusion): "Val is leaked.
The honest pipeline floor is F1 ≈ 0.34 (train-tuned, BM25 top-30 + reranker, K=5)."
"""))


# ────────────────────────────────────────────────────────────────────────
# Cell 17 — v13 final plan + diagram
# ────────────────────────────────────────────────────────────────────────
cells.append(md(r"""# Appendix B — v13 final plan: two parallel channels, merge at output

v13 is structurally **v12 done twice in parallel** — once over law (using
v12's exact recall stage) and once over court (using the v7.5 funnel pool as
the recall stage, since no equivalent of TLF exists for court paragraphs).
Both channels feed the **same reranker and judge models**, but with
family-aware dossier formatting and family-specific system prompts. Output
is the **union** of the two channels' final picks.

## Why two channels

| Reason | Explanation |
|---|---|
| Law gold has lexical anchors | "Art. 221 Abs. 1 StPO" appears literally in `corpus.parquet`'s `bm25_text` → BM25+TLF nails recall@100 ≈ 0.90 |
| Court gold has only semantic anchors | "BGE 137 IV 122 E. 6.2" is a paragraph of reasoning; its own ID isn't in the query. BM25 can't find specific court paragraphs lexically; needs the v7.5 funnel's 16-channel ensemble |
| Different score scales | Law's BM25 raw score and court's v7.5 RRF rank have incomparable distributions; per-family normalization avoids cross-family interference |
| Different K calibrations | Cascade-heavy val queries: ~50/50 law:court. Moderate cascade: ~70/30 law:court. Train-style: ~99/1. Separate `K_law` and `K_court` predictors capture this. |

## Diagram

```text
                ┌─────────────────────────────────────────────────────────┐
                │                       User Query                        │
                └────────────────────────┬────────────────────────────────┘
                                         │
                                         ▼
                ┌─────────────────────────────────────────────────────────┐
                │ Adaptive-K predictor  (regex-only, no LLM)              │
                │   cascade = 2·litigants + 1.5·adversarial + 1.5·posture │
                │           + 1·dates + 5·iie + 3·is_english              │
                │ → tier ∈ {heavy ≥10, moderate ≥3, substantive <3}       │
                │ → K_law, K_court                                        │
                └────────────────────────┬────────────────────────────────┘
                                         │
                  ┌──────────────────────┴──────────────────────┐
                  │                                              │
                  ▼                                              ▼
    ╔════════════════════════════════╗            ╔════════════════════════════════╗
    ║         LAW CHANNEL            ║            ║        COURT CHANNEL           ║
    ║  (v12's exact recall stage)    ║            ║  (v7.5 funnel as recall stage) ║
    ╠════════════════════════════════╣            ╠════════════════════════════════╣
    ║                                ║            ║                                ║
    ║  Stage 1a:                     ║            ║  Stage 1a:                     ║
    ║    BM25 + TLF                  ║            ║    v7.5 final_topk             ║
    ║    over corpus.parquet (171K)  ║            ║    filtered to court:*         ║
    ║    → top-100 law candidates    ║            ║    → top-500 court candidates  ║
    ║                                ║            ║                                ║
    ║  Stage 1b: keep raw BM25 score ║            ║  Stage 1b: rank-derived score  ║
    ║                                ║            ║    (n - rank)                  ║
    ║                                ║            ║                                ║
    ║  Stage 2: Qwen3-Reranker-8B    ║            ║  Stage 2: Qwen3-Reranker-8B    ║
    ║    _fmt_rr branch:LAW          ║            ║    _fmt_rr branch: COURT       ║
    ║    Instruction: "is article    ║            ║    Instruction: "is paragraph  ║
    ║    applicable to issue"        ║            ║    binding or persuasive       ║
    ║                                ║            ║    precedent"                  ║
    ║                                ║            ║                                ║
    ║  Stage 3: fused + zone         ║            ║  Stage 3: fused + zone         ║
    ║    fused = 0.7·norm(BM25)      ║            ║    fused = 0.7·norm(rank)      ║
    ║          + 0.3·P(yes)          ║            ║          + 0.3·P(yes)          ║
    ║    HIGH=0.55, LOW=0.25 (tight) ║            ║    HIGH=0.55, LOW=0.15 (loose) ║
    ║                                ║            ║                                ║
    ║  Stage 4: Qwen3-8B judge       ║            ║  Stage 4: Qwen3-8B judge       ║
    ║    JUDGE_SYSTEM_LAW            ║            ║    JUDGE_SYSTEM_COURT          ║
    ║    (Swiss 7-category prompt)   ║            ║    (precedent-relevance        ║
    ║                                ║            ║     prompt: chamber, role,     ║
    ║                                ║            ║     authority, doctrinal rule) ║
    ║                                ║            ║                                ║
    ║  Stage 5: select top-K_law     ║            ║  Stage 5: select top-K_court   ║
    ║    from law_pool by fused      ║            ║    from court_pool by fused    ║
    ║                                ║            ║                                ║
    ║  Stage 5.5: article-sibling    ║            ║  Stage 5.5: case-peer          ║
    ║    expansion (Abs./lit. peers  ║            ║    expansion (sibling Es of    ║
    ║    of same Art. N CODE)        ║            ║    same case_base, e.g. BGE    ║
    ║                                ║            ║    137 IV 122 E.4.1/4.2/6.2)   ║
    ╚════════════════╤═══════════════╝            ╚════════════════╤═══════════════╝
                     │                                              │
                     │            ┌──────────────────────┐          │
                     └───────────►│  Stage 6: MERGE      │◄─────────┘
                                  │  predictions =       │
                                  │    sel_law ∪         │
                                  │    sel_court         │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │ Final predictions    │
                                  │ + evaluate on full   │
                                  │   val.csv gold       │
                                  └──────────────────────┘
```

## K predictor table

| Cascade tier | K_law | K_court |
|---|---|---|
| **cascade ≥ 10** (heavy procedural cascade) | `clip(cascade × 1.5, 15, 40)` | `clip(cascade × 1.5, 15, 40)` |
| **cascade ∈ [3, 10]** (moderate cascade) | `clip(cascade × 2.0, 8, 25)` | `clip(cascade × 1.0, 3, 15)` |
| **cascade < 3** (substantive) | `max(2, cascade × 1.5)` | `max(0, cascade × 0.3)` |

## Submission variants emitted

| File | What it is |
|---|---|
| `submission_v13_law_only_ablation.csv` | law channel alone (debug) |
| `submission_v13_court_only_ablation.csv` | court channel alone (debug) |
| `submission_v13_union_adaptive.csv` | **primary** — `sel_law ∪ sel_court` at predicted K |
| `submission_v13_union_adaptive_expanded.csv` | primary + Stage 5.5 expansion |
| `submission_v13_union_adaptive_x1.3.csv` | recall-leaning hedge |
| `submission_v13_union_adaptive_x1.3_expanded.csv` | recall-leaning + expansion |
| `submission_v13_union_k5_fallback.csv` | train-honest fallback |

## Per-stage recall accounting

| Stage | Macro recall target | Notes |
|---|---:|---|
| Stage 1a — LAW BM25+TLF top-100 | ~0.85-0.90 | v12 ceiling on law gold |
| Stage 1a — COURT v7.5 top-500 | ~0.30-0.35 | court gold density limit |
| Stage 1a — combined | ~0.55-0.60 | union recall on full gold |
| Stage 2 — reranker | unchanged | re-orders, doesn't drop |
| Stage 3 — auto-YES ∪ borderline | ~0.50-0.55 | zone-split ceiling |
| Stage 4 — judge survivors | ~0.50-0.55 | judge defaults YES on parse-fail |
| Stage 5 — adaptive K cutoff | depends on K_law/K_court calibration |
| Stage 5.5 — expansion | small recall lift on cluster-heavy queries |

## Realistic F1 forecast

- **Law channel F1 on law gold**: 0.70-0.85 (matches v12)
- **Court channel F1 on court gold**: 0.15-0.25 (the hard half)
- **Combined F1 on full val.csv gold (macro)**: **0.30-0.45 realistic, 0.50+ aspirational**

To go beyond 0.45 in a single pipeline requires either:
1. A **listwise reranker** (e.g., Qwen3-32B-AWQ sliding-window) to fix the precision-on-many-similar-court-paragraphs problem
2. A **LoRA-finetuned Qwen3-Reranker** on Swiss legal triplets
3. A **dense retriever specifically trained on Swiss legal data** to lift court recall@K
"""))

# ── Save
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

ROOT = Path(__file__).resolve().parents[2]
OUT_NB = ROOT / "notebooks" / "v13_court_aware_adaptive_k" / "synthesis_v13_final.ipynb"
OUT_NB.parent.mkdir(parents=True, exist_ok=True)
OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT_NB}  ({OUT_NB.stat().st_size/1024:.1f} KB)")
print(f"Cells: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
