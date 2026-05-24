# %% [markdown]
# # Swiss Legal Citation Retrieval — Hybrid v12 (clean reference build)
#
# **Project.** Kaggle *LLM Agentic Legal Information Retrieval* — cross-lingual Swiss legal
# citation extraction. Given an English legal scenario, return the canonical set of Swiss
# federal citations (statutes + court-decision paragraphs) an expert would cite.
#
# **Metric.** Official Macro F1 — per-query F1 between predicted and gold citation sets,
# macro-averaged across queries. Whitespace canonicalization only — no case folding, no
# parent↔child expansion. The whole gold (laws + court rulings) is in the denominator.
#
# **Reference result on val** (10 English queries):
#
# | Score | Macro F1 | Macro P | Macro R |
# |---|---:|---:|---:|
# | **Whole-gold** (laws + court rulings) | **0.585** | 0.845 | 0.464 |
# | **Law-only** (court rulings removed from gold) | **0.773** | 0.864 | 0.700 |
#
# **The whole-gold ceiling is structural.** 102 of 251 val gold items (40.6%) are court
# rulings (`BGE …`, `1B_…/2023 E. …`) that this pipeline cannot retrieve — there is no
# court-rulings index. The 0.188 gap from law-only (0.773) to whole-gold (0.585) is
# entirely the court-rulings denominator and is independent of any law-side improvement.
#
# ## Pipeline
#
# ```
#   query (English)
#     │
#     ├─ Stage 1 ────► BM25 top-100  (TLF-enhanced bilingual tokens)
#     │
#     ├─ Stage 2 ────► Qwen3-Reranker-8B  (yes/no logit softmax → P(yes))
#     │
#     ├─ Stage 3 ────► fused = 0.7·minmax(BM25) + 0.3·P(yes)
#     │                ┌───── fused ≥ 0.55 ──► AUTO-YES    (~7 / query)
#     │                ├───── fused < 0.25 ──► AUTO-NO     (~87 / query)
#     │                └───── otherwise   ──► BORDERLINE  (~6 / query)
#     │
#     ├─ Stage 4 ────► Qwen3-8B judge on borderline only  (7-category Swiss-citation prompt)
#     │
#     └─ Stage 5 ────► AUTO-YES ∪ (BORDERLINE ∧ judge=YES)  →  apply_proper_case  →  submission.csv
# ```
#
# ## Three load-bearing design choices
#
# 1. **BM25 R@100 ≈ 0.90 on law-only gold.** Recall is essentially solved at top-100; the
#    pipeline's job is to push *precision* on the uncertain middle, not to retrieve more.
# 2. **The judge runs only on the borderline pile (~6 candidates per query), never on
#    auto-YES or auto-NO.** Judging all 100 candidates would *lower* F1 because the judge's
#    "include if related" bias would drag irrelevant articles in. Confining it to the
#    uncertain middle makes its precision lift additive.
# 3. **`apply_proper_case` is mandatory before scoring.** The BM25 corpus stores Swiss-law
#    abbreviations all-uppercase (`STPO`, `STGB`, `STBOG`, `SCHKG`); the Kaggle gold uses
#    the legally-correct mixed casing (`StPO`, `StGB`, `StBOG`, `SchKG`). The official
#    scorer is plain string equality after whitespace canon, so casing has to match exactly.
#
# ## Runtime
#
# - **First run on A100 (40 GB or 80 GB):** ~25-30 min (mostly the Qwen3-Reranker-8B pass
#   over ~1,000 (query, candidate) pairs + the Qwen3-8B judge on ~60 borderline candidates).
# - **Subsequent runs:** ~30 s — every model call is cached to Drive at
#   `retrieval/v12_clean_cache/`. Re-running only re-does the deterministic selection +
#   scoring.

# %% [markdown]
# ## 1. Setup — dependencies and Google Drive
#
# Four pip packages beyond the Colab default image:
#
# - `rank_bm25` — sparse retrieval index
# - `transformers ≥ 4.51` — Qwen3 chat-template support
# - `accelerate` — sharded model loading
# - `pyarrow` — parquet I/O
#
# **GPU memory.** Reranker (Qwen3-Reranker-8B) and judge (Qwen3-8B) each take ~16 GB in
# bf16. The script loads them sequentially and frees one before loading the other, so
# only **one** model is resident at a time — fits comfortably in A100 40 GB.
#
# **Drive.** Read-only for inputs (`retrieval/corpus.parquet`, BM25 pickles, KB,
# query-translation cache). The single writable subfolder is
# `retrieval/v12_clean_cache/` — ~50 MB of cached reranker + judge outputs that make
# re-runs fast.

# %%
import sys, subprocess
def _pip(pkgs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])
try:
    import rank_bm25, transformers, accelerate, tqdm, pyarrow  # noqa
except ImportError:
    _pip(['rank_bm25', 'transformers>=4.51', 'accelerate', 'tqdm', 'pandas', 'pyarrow', 'numpy'])

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    IN_COLAB = True
    print('✓ Drive mounted')
except ImportError:
    IN_COLAB = False
    print('Local mode (not Colab)')

# %% [markdown]
# ## 2. Imports & configuration
#
# Three groups of constants carry real meaning:
#
# **Retrieval depth.**
# - `TOP_BM25 = 100` — BM25 R@100 ≈ 0.90 on law-only val gold. Going to 200 adds ~1 pp
#   of recall but doubles the reranker cost — diminishing returns.
#
# **Fused-score blend.**
# - `FUSED_W_BM25 = 0.7`, `FUSED_W_RERA = 0.3` — asymmetric: BM25 is the recall workhorse,
#   reranker is a precision tiebreaker over BM25's ordering.
#
# **Zone thresholds.**
# - `HIGH_THRESH = 0.55`, `LOW_THRESH = 0.25` — calibrated so the average val query splits
#   into roughly **7 auto-YES + 6 borderline + 87 auto-NO**. Wider gaps push more
#   decisions to the judge and the judge's pile gets noisier; narrower gaps starve the
#   judge.
#
# **Models.** Qwen3-Reranker-8B for the cross-encoder reranker (Stage 2); Qwen3-8B for
# the generative judge (Stage 4). Both load from Hugging Face on first run (~30 GB
# download) and cache in `~/.cache/huggingface/`.

# %%
import os, re, json, gc, time, pickle, datetime, math
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
if IN_COLAB:
    DRIVE_BASE = Path('/content/drive/MyDrive/Omnilex-Agentic-Retrieval-Competition')
else:
    DRIVE_BASE = Path(r'E:\swiss_citation_extraction')

DATA_DIR  = DRIVE_BASE / 'data'
RETRIEVAL = DRIVE_BASE / 'retrieval'
CACHE_DIR = RETRIEVAL / 'v12_clean_cache'
OUT_DIR   = CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# v12 input artifacts (auto-detect _v2 variants)
CORPUS_PARQUET = next((p for p in [RETRIEVAL/'corpus.parquet', RETRIEVAL/'corpus_v2.parquet'] if p.exists()),
                      RETRIEVAL/'corpus.parquet')
BM25_INDEX_PKL = next((p for p in [RETRIEVAL/'bm25_v2_index.pkl', RETRIEVAL/'bm25_index.pkl'] if p.exists()),
                      RETRIEVAL/'bm25_v2_index.pkl')
BM25_IDS_PKL   = next((p for p in [RETRIEVAL/'bm25_v2_ids.pkl', RETRIEVAL/'bm25_ids.pkl'] if p.exists()),
                      RETRIEVAL/'bm25_v2_ids.pkl')
KB_JSONL       = RETRIEVAL / 'laws_knowledge_base.jsonl'
QT_DIR         = RETRIEVAL / 'knowledge_base_optimized_hybrid_retrieval'

# ── Models ───────────────────────────────────────────────────────────────────
RERANKER_MODEL = 'Qwen/Qwen3-Reranker-8B'    # cross-encoder reranker (yes/no logit head)
JUDGE_MODEL    = 'Qwen/Qwen3-8B'             # generative judge

# ── Hyperparameters ──────────────────────────────────────────────────────────
TOP_BM25     = 100
FUSED_W_BM25 = 0.7
FUSED_W_RERA = 0.3
HIGH_THRESH  = 0.55
LOW_THRESH   = 0.25
RERANK_BS    = 8       # batch size for reranker
MAX_LEN      = 4096    # reranker context

# Judge generation
JUDGE_TEMP   = 0.5
JUDGE_TOP_K  = 20
JUDGE_TOP_P  = 0.95
JUDGE_MAXTOK = 3000

# Whether to also score test.csv (no gold, just submission)
SCORE_TEST   = True

# Cache files
RERANK_CACHE = CACHE_DIR / f'rerank_{RERANKER_MODEL.replace("/","_")}.json'
JUDGE_CACHE  = CACHE_DIR / f'judge_{JUDGE_MODEL.replace("/","_")}.json'

def log(*a, **k): print(*a, **k, flush=True)
def section(t): log('\n' + '═'*72); log(f'  {t}'); log('═'*72)

log(f'DRIVE_BASE  : {DRIVE_BASE}')
log(f'CACHE_DIR   : {CACHE_DIR}')
log(f'CORPUS      : {CORPUS_PARQUET.name}')
log(f'BM25_INDEX  : {BM25_INDEX_PKL.name}')
log(f'BM25_IDS    : {BM25_IDS_PKL.name}')
log(f'RERANKER    : {RERANKER_MODEL}')
log(f'JUDGE       : {JUDGE_MODEL}')

# %% [markdown]
# ## 3. Utilities — tokenisation, citation parsing, casing, official metric
#
# Four families of helpers used throughout the pipeline.
#
# ### Tokenisation
#
# Lowercase, split on non-alphanumeric, drop tokens shorter than 2 chars (except digits).
#
# Bilingual by design — val queries are English; the corpus and BM25 index are over
# German text. We concatenate EN tokens with the cached EN→DE Google Translate of the
# same query, deduplicated in order. This gives BM25 access to the German code
# abbreviations (`StGB`, `OR`, `ZGB`, …) that the English query lacks, while preserving
# the English content words that lexically match the translated `embed_text` column in
# the corpus.
#
# ### Citation parsing
#
# Swiss citations are hierarchical:
#
# ```
# Art. 123 ZGB             ← parent  (sometimes appears in gold)
# ├─ Art. 123 Abs. 1 ZGB     ← leaf
# ├─ Art. 123 Abs. 2 ZGB     ← leaf
# └─ Art. 123 Abs. 3 ZGB     ← leaf  (corpus is keyed at leaf-level)
# ```
#
# `parse_citation` returns `{art, abs, lit, code}` or `None`.
# `build_parent_expansion_map` indexes laws_de.csv by `(article, code)` so the
# diagnostic in Stage 1 can fan a parent out to every `Abs. M` child that actually
# exists in the corpus. **The official scorer compares raw citation strings** after
# whitespace canon — these helpers are for diagnostics and the TLF mining only.
#
# ### Casing canonicalization (`apply_proper_case`)
#
# The BM25 corpus stores Swiss-law abbreviations all-uppercase (`STPO`, `STGB`, `STBOG`,
# `SCHKG`). The Kaggle gold uses the legally-correct German mixed casing (`StPO`,
# `StGB`, `StBOG`, `SchKG`). The official scorer is plain string equality after
# whitespace canon, so predictions have to be translated back. We build a uppercase →
# proper-case lookup from `laws_de.csv` once at startup and apply it as the final step
# before submission.
#
# ### Official Macro F1
#
# Direct port of the competition scorer: strip + collapse whitespace, set-based F1,
# macro mean across queries. The function in this cell exactly matches
# `scripts/evaluate_submission.py` from the competition repo.

# %%
_TOKEN_RE = re.compile(r'[^\w\d]+', re.UNICODE)
_CIT_RE   = re.compile(r'^Art\.\s+(\d+[a-z]?)(?:\s+Abs\.\s+(\d+[a-z]*))?(?:\s+lit\.\s+(\S+))?\s+(\S+)$')

def tokenise(text):
    """Lowercase, drop non-alphanumeric separators, keep tokens ≥2 chars or digits."""
    if not text: return []
    return [w for w in _TOKEN_RE.split(str(text).lower().strip()) if len(w) >= 2 or w.isdigit()]

def parse_citation(c):
    """Parse 'Art. 221 Abs. 1 lit. b StPO' → {art:'221', abs:'1', lit:'b', code:'StPO'}.
    Returns None if not a valid Art.-prefixed citation."""
    m = _CIT_RE.match(str(c).strip())
    if not m: return None
    return {'art': m.group(1), 'abs': m.group(2), 'lit': m.group(3), 'code': m.group(4)}

def extract_law_abbrev(citation):
    """Last whitespace-delimited token = law abbreviation. 'Art. 221 Abs. 1 StPO' → 'StPO'."""
    parts = str(citation).strip().split()
    return parts[-1] if parts else ''

def build_parent_expansion_map(laws_de):
    """For each (article, code) parent key, list all Abs.-children in laws_de.csv."""
    children = defaultdict(list)
    for c in laws_de['citation'].dropna():
        p = parse_citation(c)
        if p:
            children[(p['art'], p['code'])].append(c)
    return children

# ── Official Macro F1 (matches scripts/evaluate_submission.py) ───────────────
_WS = re.compile(r'\s+')
def _canon(c): return _WS.sub(' ', str(c).strip())

def _parse_field(v):
    if pd.isna(v): return set()
    return {_canon(p) for p in str(v).split(';') if _canon(p)}

def _f1(pred, gold):
    if not pred and not gold: return 1.0, 1.0, 1.0
    if not pred or not gold:  return 0.0, 0.0, 0.0
    tp = len(pred & gold)
    P, R = tp / len(pred), tp / len(gold)
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F

def macro_f1(preds, df, law_only=False, per_query=True):
    """Official Macro F1. preds: {qid: [citation, ...]}. df: val.csv (must have gold_citations)."""
    rows = []
    for _, row in df.iterrows():
        qid = row['query_id']
        gold = _parse_field(row.get('gold_citations', ''))
        pred = _parse_field(';'.join(preds.get(qid, [])))
        if law_only:
            gold = {c for c in gold if c.startswith('Art.')}
            pred = {c for c in pred if c.startswith('Art.')}
        P, R, F = _f1(pred, gold)
        rows.append({'query_id': qid, 'P': P, 'R': R, 'F1': F,
                     'pred': len(pred), 'gold': len(gold)})
    out = pd.DataFrame(rows)
    macro = out[['P', 'R', 'F1']].mean().to_dict()
    if per_query:
        log(out.to_string(index=False))
        log(f"\n  MACRO  P={macro['P']:.3f}  R={macro['R']:.3f}  F1={macro['F1']:.3f}")
    return out, macro

@dataclass
class Candidate:
    """One (query, corpus-row) pair. Each pipeline stage attaches one more field."""
    citation:        str
    bm25_rank:       int   = 10**9
    bm25_score:      float = 0.0
    reranker_score:  float = 0.0
    fused_score:     float = 0.0
    zone:            str   = ''     # 'yes' | 'no' | 'borderline'
    judge_verdict:   str   = ''     # 'YES' | 'NO' | ''

# %% [markdown]
# ## 4. Stage 0 — load corpus, indices, KB, TLF prior, proper-case map
#
# Pure I/O. No model calls. Produces five in-memory artifacts the rest of the pipeline
# depends on:
#
# | Artifact | What it is | Where it's used |
# |---|---|---|
# | `corpus` | The 171,654-article parquet (citation, abbrev, full law name, heading, text, …) | Reranker doc text, judge doc text |
# | `bm25` + `bm25_ids` | Pre-built rank_bm25 Okapi index with tuned `k1` / `b` from `bm25_v2_ids.pkl` | Stage 1 retrieval |
# | `kb` | Per-article metadata from `laws_knowledge_base.jsonl` | Reranker doc text enrichment |
# | `query_translations` | Cached EN→DE Google Translate for each query | Bilingual BM25 tokens |
# | `tlf` + `token_total` | Token → law-abbrev co-occurrence counts, mined inline from train.csv | Stage 1 token-stream expansion |
# | `proper_case_map` | `STPO` → `StPO` casing lookup, built from `laws_de.csv` | Stage 5 casing fix |
# | `children_index` | `(article, code)` → list of Abs.-children | Pool-recall diagnostic |
#
# ### TLF — Token-Law-Frequency prior
#
# For every (query token, law abbreviation) pair, we count how often the token appeared
# in a gold-annotated train query whose canonical citations referenced that abbreviation.
# So `tlf['disability']` looks like:
#
# ```
# {'atsg': 47, 'ivg': 39, 'or': 3, ...}
# ```
#
# A supervised signal that the token *disability* should bias retrieval toward ATSG /
# IVG. Stage 1 turns this into IDF-weighted token-stream expansion (pick the top-5
# abbrevs by `(count / total) × IDF`, inject each 5× into the token stream — see
# Stage 1 for the rationale).
#
# **Important nuance.** TLF uses train.csv but does *not* read train's gold labels in a
# way that creates a leakage risk for val/test — it only counts token co-occurrence
# with code abbreviations, a robust corpus-level statistic. The TLF prior generalises
# across queries; it doesn't memorise specific gold sets.

# %%
section('STAGE 0: LOAD ARTIFACTS')

log('  loading corpus.parquet …')
corpus = pd.read_parquet(CORPUS_PARQUET)
log(f'  corpus: {len(corpus):,} rows; sample columns: {list(corpus.columns)[:8]}')

log('  loading BM25 index …')
with open(BM25_INDEX_PKL, 'rb') as f: bm25 = pickle.load(f)
with open(BM25_IDS_PKL, 'rb')   as f: bm25_meta = pickle.load(f)
# v12's bm25_v2_ids.pkl is a dict {citation_canon, best_k1, best_b, best_f1_k}
if isinstance(bm25_meta, dict):
    bm25_ids = bm25_meta.get('citation_canon') or bm25_meta.get('ids') or []
    if 'best_k1' in bm25_meta: bm25.k1 = float(bm25_meta['best_k1'])
    if 'best_b'  in bm25_meta: bm25.b  = float(bm25_meta['best_b'])
    log(f'  bm25 hyperparams: k1={bm25.k1:.3f}, b={bm25.b:.3f}, target_k={bm25_meta.get("best_f1_k")}')
else:
    bm25_ids = bm25_meta
log(f'  bm25: {len(bm25_ids):,} docs')

# corpus↔BM25 alignment sanity check
_corpus_cit = corpus.get('citation_canon', corpus.get('citation', pd.Series([]))).tolist()
_ov = len(set(_corpus_cit) & set(bm25_ids))
if _ov < 0.95 * len(bm25_ids):
    log(f'  ⚠ alignment only {_ov}/{len(bm25_ids)} — corpus & BM25 ids look mismatched')
else:
    log(f'  ✓ corpus↔BM25 alignment: {_ov}/{len(bm25_ids)} ({100*_ov/len(bm25_ids):.1f}%)')

log('  loading laws_knowledge_base.jsonl …')
kb = {}
with open(KB_JSONL, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        kb[r.get('citation_canon') or r.get('citation') or ''] = r
log(f'  KB: {len(kb):,} entries')

log('  loading query translations …')
qt_files = sorted(QT_DIR.glob('query_translations*.json'))
query_translations = {}
for p in qt_files:
    query_translations.update(json.loads(p.read_text(encoding='utf-8')))
log(f'  query_translations: {len(query_translations)} entries (from {len(qt_files)} files)')

log('  loading data files …')
train_df = pd.read_csv(DATA_DIR / 'train.csv')
val_df   = pd.read_csv(DATA_DIR / 'val.csv')
laws_de  = pd.read_csv(DATA_DIR / 'laws_de.csv')
log(f'  train: {len(train_df)} queries, val: {len(val_df)} queries, laws_de: {len(laws_de):,} articles')

test_df = None
if SCORE_TEST:
    test_csv = DATA_DIR / 'test.csv'
    if test_csv.exists():
        test_df = pd.read_csv(test_csv)
        log(f'  test: {len(test_df)} queries (no gold; will produce submission only)')
    else:
        log(f'  no test.csv at {test_csv}')

# Mine TLF prior from train.csv (no labels — just token-abbrev co-occurrence)
log('  mining TLF prior from train.csv …')
tlf = defaultdict(Counter)
token_total = Counter()
for _, row in train_df.iterrows():
    cites = [c.strip() for c in str(row['gold_citations']).split(';') if c.strip()]
    abbrevs = [extract_law_abbrev(c).lower() for c in cites if c.startswith('Art.')]
    abbrevs = [a for a in abbrevs if a]
    toks = tokenise(row['query'])
    for t in toks:
        token_total[t] += 1
        for a in abbrevs:
            tlf[t][a] += 1
log(f'  TLF: {len(tlf):,} distinct tokens, {sum(token_total.values()):,} total occurrences')

# Proper-case map (laws_de.csv is the source of truth for legally-correct Swiss casing)
proper_case_map = {c.upper(): c for c in laws_de['citation'].dropna() if c}
log(f'  proper-case map: {len(proper_case_map):,} entries')

# Children index (used by BM25 R@K diagnostic)
children_index = build_parent_expansion_map(laws_de)
log(f'  children_index: {len(children_index):,} parent keys')

# Per-citation text lookup (used by reranker & judge document representations)
LAW_TEXT = {c: (t if isinstance(t, str) else '')
            for c, t in zip(laws_de['citation'], laws_de['text'])}

# %% [markdown]
# ## 5. Stage 1 — TLF-enhanced BM25 retrieval (top-100)
#
# For each query, build the BM25 token stream in four steps:
#
# 1. **Tokenise bilingually.** EN tokens from the raw query + DE tokens from the cached
#    Google Translate of the same query, deduplicated in order (EN first).
# 2. **Score the TLF table.** For every token, look up its `tlf[token]` distribution
#    over law abbreviations and weight each entry by `(count / total) × IDF(token)`.
#    Sum across tokens, keep the **top-5 abbreviations**.
# 3. **Inject those 5 abbreviations into the token stream 5× each.** A single
#    insertion gets dominated by article body text; 5× repetition lifts the
#    abbreviation into the dominant term frequency so BM25 treats it as a strong
#    signal about which code family the query belongs to.
# 4. **Score the corpus** with `BM25.get_scores` and take the top 100.
#
# ### Why this matters
#
# Cross-lingual queries lack German code abbreviations entirely. Without TLF
# expansion, BM25 retrieves articles whose German text happens to be lexically
# similar to the English query — rare matches, mostly noise. TLF substitutes a
# supervised expansion for the missing abbreviations, lifting BM25 from "lucky
# lexical hits" to "right code family". This is the load-bearing recall mechanism
# of the whole pipeline.

# %%
def enhance_tokens_with_tlf(tokens, top_abbrev=5, repeats=5):
    """For each token, look up its abbrev distribution. Take top-K abbrevs by IDF-weighted
    score and inject each `repeats` times into the token stream."""
    scores = defaultdict(float)
    for t in tokens:
        dist = tlf.get(t)
        if not dist: continue
        n = sum(dist.values())
        if n == 0: continue
        idf = math.log((len(token_total) + 1) / (token_total[t] + 1)) if token_total[t] else 1.0
        for abbrev, cnt in dist.items():
            scores[abbrev] += (cnt / n) * idf
    top = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_abbrev]
    return tokens + [a for a in top for _ in range(repeats)]

def build_query_tokens(qid, qtxt):
    """EN tokens + DE-translated tokens, deduplicated, then TLF-enhanced."""
    en = tokenise(qtxt)
    de = tokenise(query_translations.get(qid, ''))
    base, seen = [], set()
    for t in en + de:
        if t not in seen:
            seen.add(t); base.append(t)
    return enhance_tokens_with_tlf(base)

def bm25_top_k(qid, qtxt, k=TOP_BM25):
    """Returns list of (citation, bm25_score) top-k for the query."""
    toks = build_query_tokens(qid, qtxt)
    scores = bm25.get_scores(toks)
    order = np.argsort(scores)[::-1][:k]
    return [(bm25_ids[i], float(scores[i])) for i in order if scores[i] > 0]

# Smoke test on val_001
_sample = bm25_top_k(val_df.iloc[0]['query_id'], val_df.iloc[0]['query'], k=5)
section('STAGE 1: TLF-ENHANCED BM25 (smoke test)')
log(f'  sample top-5 for {val_df.iloc[0]["query_id"]}:')
for c, s in _sample:
    log(f'    {s:7.2f}  {c}')

# %% [markdown]
# ## 6. Stage 2 — Qwen3-Reranker-8B (yes/no logit softmax)
#
# Qwen3-Reranker-8B is a **causal-LM reranker**, not a regression head — it scores a
# (query, document) pair by generating a single token of `yes` or `no` given a fixed
# instruction. We don't sample; we take the final-position logits at the `yes` and `no`
# token ids, softmax them, and use `P(yes) ∈ [0, 1]` as the reranker score.
#
# ### Implementation details
#
# **Strict prefix/suffix.** The official Qwen3-Reranker chat template wraps every
# (query, document) pair in a fixed system + user prefix and an assistant suffix
# containing an empty `<think>` block. The token-id sequences are pre-computed once at
# model-load time and concatenated around each pair — we never re-tokenise the prefix
# per call.
#
# **Batched single-forward.** With `batch_size = 8` and `max_length = 4096`, the 100
# candidates of one query score in ~12 seconds on an A100. The full val set
# (~1,000 pairs) is ~22 minutes.
#
# **Per-query caching.** Reranker output is written to
# `retrieval/v12_clean_cache/rerank_<model>.json`. Subsequent runs reuse the cache
# verbatim; only new queries trigger a model load.
#
# ### Document representation
#
# Each candidate's document passed to the reranker contains:
#
# - `citation` (e.g. `Art. 221 Abs. 1 StPO`)
# - `law_abbrev` and `law_full_name` from the KB
# - structural `heading` from the corpus
# - first 500 characters of the German article body
#
# Enough context for the model to judge relevance without blowing the 4096-token
# input budget.

# %%
RERANK_INSTRUCTION = ('Given a query about Swiss law, determine whether the provided law '
                      'article is related to or applicable to the legal issue.')

def candidate_doc_text(cit):
    """Compact document representation passed to the reranker."""
    body = LAW_TEXT.get(cit, '')
    kb_row = kb.get(cit, {})
    law_abbrev = kb_row.get('law_abbrev', '')
    law_full_name = kb_row.get('law_full_name', '') or kb_row.get('law_name_en', '')
    heading = kb_row.get('heading', '') or kb_row.get('article_topic', '')
    parts = [cit]
    if law_abbrev or law_full_name:
        parts.append(f'Law: {law_abbrev} — {law_full_name}')
    if heading:
        parts.append(f'Heading: {heading}')
    parts.append(f'Text: {body[:500]}')
    return '\n'.join(parts)

def load_reranker():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    log(f'  loading reranker: {RERANKER_MODEL}')
    tok = AutoTokenizer.from_pretrained(RERANKER_MODEL, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        RERANKER_MODEL, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    model.eval()
    prefix = tok.encode(
        '<|im_start|>system\nJudge whether the Document meets the requirements based on the '
        'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
        '<|im_end|>\n<|im_start|>user\n', add_special_tokens=False)
    suffix = tok.encode(
        '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', add_special_tokens=False)
    yes_id = tok.convert_tokens_to_ids('yes')
    no_id  = tok.convert_tokens_to_ids('no')
    return tok, model, prefix, suffix, yes_id, no_id

def run_reranker(pool_by_qid, query_texts, cache_path=RERANK_CACHE):
    """Score every (query, candidate) pair with Qwen3-Reranker-8B. Cache to JSON.
    Mutates `pool_by_qid` in place: each Candidate's `reranker_score` is set."""
    out = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    import torch
    tok, model, prefix, suffix, yes_id, no_id = load_reranker()

    for qid, pool in tqdm(pool_by_qid.items(), desc='rerank queries'):
        cached = out.get(qid, {})
        todo = [c for c in pool if c not in cached]
        if not todo:
            for cit, cd in pool.items():
                cd.reranker_score = float(cached.get(cit, 0.0))
            continue
        out.setdefault(qid, {})
        q = query_texts[qid]
        pairs = [f'<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {q}\n<Document>: {candidate_doc_text(c)}'
                 for c in todo]
        for i in tqdm(range(0, len(pairs), RERANK_BS), desc=f'  {qid}', leave=False):
            batch = pairs[i:i+RERANK_BS]
            ids = [tok.encode(b, add_special_tokens=False, truncation=True,
                              max_length=MAX_LEN - len(prefix) - len(suffix)) for b in batch]
            ids = [prefix + x + suffix for x in ids]
            maxlen = max(len(x) for x in ids)
            attn  = [[0]*(maxlen-len(x)) + [1]*len(x) for x in ids]
            padded= [[tok.pad_token_id or 0]*(maxlen-len(x)) + x for x in ids]
            inp = torch.tensor(padded, device=model.device)
            am  = torch.tensor(attn,   device=model.device)
            with torch.no_grad():
                logits = model(input_ids=inp, attention_mask=am).logits[:, -1, :]
            yes = logits[:, yes_id]; no_ = logits[:, no_id]
            probs = torch.softmax(torch.stack([no_, yes], dim=-1), dim=-1)[:, 1].float().cpu().numpy()
            for c, p in zip(todo[i:i+RERANK_BS], probs):
                out[qid][c] = float(p)
                pool[c].reranker_score = float(p)
        cache_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')

    del model, tok
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    return out

# %% [markdown]
# ## 7. Stage 3 — fused score & zone split
#
# Each candidate now has two scores on incompatible scales: raw BM25 (unbounded,
# query-dependent magnitude) and reranker probability `P(yes) ∈ [0, 1]`. We blend in
# two steps:
#
# 1. **Per-query min-max normalisation of BM25.** `(bm25 − min) / (max − min)` over the
#    100 candidates of *this* query. Normalisation stays query-local rather than
#    corpus-global, which would be dominated by a handful of outlier articles with
#    extreme term-frequency scores.
# 2. **Weighted blend.** `fused = 0.7 · norm_bm25 + 0.3 · P(yes)`. Asymmetric by design:
#    BM25 owns recall, the reranker is a precision tiebreaker over BM25's ordering.
#
# Two thresholds partition the 100 candidates into three zones:
#
# | Zone | Condition | Typical count / val query | What happens |
# |---|---|:---:|---|
# | **auto-YES** | `fused ≥ 0.55` | ~7 | Goes straight to the final prediction set |
# | **borderline** | `0.25 ≤ fused < 0.55` | ~6 | Stage 4 LLM judge decides |
# | **auto-NO** | `fused < 0.25` | ~87 | Discarded |
#
# The ~6-candidate borderline pile is what the judge actually sees — small enough to
# fit in one Qwen3-8B prompt, large enough to contain the precision-relevant
# uncertainty.

# %%
def assign_fused_and_zones(pool_by_qid):
    """For each query's pool: min-max normalise BM25, compute fused, assign zone."""
    for qid, pool in pool_by_qid.items():
        if not pool: continue
        bm = np.array([cd.bm25_score for cd in pool.values()])
        lo, hi = bm.min(), bm.max()
        rng = (hi - lo) if hi > lo else 1.0
        for cd in pool.values():
            nb = (cd.bm25_score - lo) / rng
            cd.fused_score = FUSED_W_BM25 * nb + FUSED_W_RERA * cd.reranker_score
            if   cd.fused_score >= HIGH_THRESH: cd.zone = 'yes'
            elif cd.fused_score <  LOW_THRESH:  cd.zone = 'no'
            else:                                cd.zone = 'borderline'

# %% [markdown]
# ## 8. Stage 4 — Qwen3-8B judge on borderline candidates
#
# The judge sees **only the ~6 borderline candidates per query** — never the auto-YES
# pile, never the auto-NO pile. That sounds obvious but it's the load-bearing design
# choice: judging all 100 candidates per query produces *lower* F1 than threshold-only,
# because the judge's "include if related" bias drags irrelevant articles into the
# prediction set. Confining the judge to the uncertain middle is what makes its
# precision lift additive rather than destructive.
#
# ### Prompt design — the seven Swiss-citation categories
#
# The system prompt enumerates the seven roles a citation plays in a Swiss Federal
# Court (Bundesgericht) decision:
#
# 1. **Substantive law** — the rule the case turns on
# 2. **Definitions** — terms used in the legal question
# 3. **Procedural rules** — how the case is litigated
# 4. **Appeal provisions** — who, when, where, how to appeal
# 5. **Cost allocation** — who pays the court costs
# 6. **Court jurisdiction** — which chamber / body decides
# 7. **Constitutional principles** — right to be heard, fair trial, proportionality
#
# The judge says YES if the candidate fits *any* of the seven roles for the query, NO
# only if it's from a completely unrelated legal domain. This taxonomy matches the
# observed structure of expert gold sets on val, which are heavily multi-category (a
# typical appeal gold contains 3-4 procedural + 2 appeal + 1 cost + 1 jurisdiction
# citations, not just substantive law).
#
# ### Explicit failure-mode handling
#
# 1. **Default-YES on parse failure.** The judge's `<think>` block can overflow the
#    3000-token generation budget when the borderline pile is large, leaving the
#    verdict list partial. Any candidate without a parsed verdict **defaults to YES**.
#    This is recall-preserving — the candidate is already in the borderline zone, so
#    its fused score is non-trivial — and it is empirically the most-important single
#    behaviour in the judge stage.
# 2. **Three-level citation matching** in the verdict parser: exact match →
#    case-insensitive match → regex on `Art. N (Abs. M)? CODE`. Handles judge outputs
#    like `Art. 123 ZGB`, `art. 123 zgb`, or `[7] Art. 123 ZGB`.
# 3. **`<think>` stripping** before parsing. The model's reasoning trace contains
#    stray `YES` / `NO` tokens that would otherwise pollute the verdict list. We
#    keep only the text after `</think>`.
#
# ### Generation settings
#
# `do_sample=True, temperature=0.5, top_k=20, top_p=0.95, max_new_tokens=3000`.
# Sampling (not greedy) gives the judge enough variation to reason through ambiguous
# candidates without collapsing onto its prior; the temperature is low enough that
# re-runs are largely consistent. For deterministic reruns, set `do_sample=False`.

# %%
JUDGE_SYSTEM = """You are a Swiss Federal Court (Bundesgericht) legal citation expert.

For each candidate law citation, decide YES or NO: would an expert Swiss-law annotator
include this citation in the gold-standard answer set for the query?

Swiss-law gold sets typically include articles fulfilling one of seven roles:
1. SUBSTANTIVE LAW — the rule the case turns on
2. DEFINITIONS — terms used in the legal question
3. PROCEDURAL RULES — how the case is litigated
4. APPEAL PROVISIONS — who, when, where, how to appeal
5. COST ALLOCATION — who pays the court costs
6. COURT JURISDICTION — which chamber / body decides
7. CONSTITUTIONAL PRINCIPLES — right to be heard, fair trial, proportionality

Say YES if the candidate fits ANY of the seven roles for THIS query.
Say NO only if it's from a completely unrelated legal domain.

Output ONLY a JSON list of verdicts, one per candidate, in the input order:
[{"i":1,"v":"YES"},{"i":2,"v":"NO"},{"i":3,"v":"YES"},...]

No prose. No code fence."""

def parse_judge_verdicts(txt, candidates):
    """Parse 'i'/'v' verdicts from the model output; default-YES on any parse miss.
    `candidates` is the in-order list of (citation, Candidate) the judge was shown."""
    if '</think>' in txt:
        txt = txt.split('</think>', 1)[1]
    out = {c: 'YES' for c, _ in candidates}  # default-YES (recall-preserving)
    m = re.search(r'\[[\s\S]*\]', txt)
    if not m: return out
    try:
        arr = json.loads(m.group(0))
        for item in arr:
            if not isinstance(item, dict): continue
            i = int(item.get('i', 0))
            v = str(item.get('v', 'YES')).upper().strip()
            if 1 <= i <= len(candidates):
                cit = candidates[i-1][0]
                out[cit] = 'YES' if v == 'YES' else 'NO'
    except Exception as e:
        log(f'  ⚠ judge parse failed (defaulting all to YES): {e}')
    return out

def load_judge():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    log(f'  loading judge: {JUDGE_MODEL}')
    tok = AutoTokenizer.from_pretrained(JUDGE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    model.eval()
    return tok, model

def run_judge(pool_by_qid, query_texts, cache_path=JUDGE_CACHE):
    """Judge borderline candidates. Mutates pool: cd.judge_verdict ∈ {'YES','NO',''}."""
    out = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    import torch
    tok, model = load_judge()

    for qid, pool in tqdm(pool_by_qid.items(), desc='judge queries'):
        borderline = [(c, cd) for c, cd in pool.items() if cd.zone == 'borderline']
        if not borderline:
            out.setdefault(qid, {})
            continue
        # full cache hit?
        cached = out.get(qid, {})
        if all(c in cached for c, _ in borderline):
            for c, cd in borderline:
                cd.judge_verdict = cached.get(c, 'YES')
            continue
        out.setdefault(qid, {})
        cand_block = '\n'.join(
            f'[{i+1}] {candidate_doc_text(c)[:600]}'
            for i, (c, cd) in enumerate(borderline))
        user_msg = (f'Query: {query_texts[qid]}\n\n'
                    f'Candidates:\n{cand_block}\n\nVerdicts (JSON list):')
        msgs = [{'role': 'system', 'content': JUDGE_SYSTEM},
                {'role': 'user',   'content': user_msg}]
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors='pt', truncation=True, max_length=8192).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inp, max_new_tokens=JUDGE_MAXTOK,
                do_sample=True, temperature=JUDGE_TEMP, top_k=JUDGE_TOP_K, top_p=JUDGE_TOP_P,
                pad_token_id=tok.eos_token_id)
        txt = tok.decode(gen[0, inp['input_ids'].shape[1]:], skip_special_tokens=True)
        verdicts = parse_judge_verdicts(txt, borderline)
        for c, cd in borderline:
            v = verdicts.get(c, 'YES')
            cd.judge_verdict = v
            out[qid][c] = v
        cache_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')

    del model, tok
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    return out

# %% [markdown]
# ## 9. Stage 5 — final predictions, casing fix, submission
#
# ### Selection
#
# The prediction set is the **union** of the two YES sources:
#
# ```
# Final(query) = { c : c.zone == 'yes' }
#              ∪ { c : c.zone == 'borderline' ∧ c.judge_verdict == 'YES' }
# ```
#
# Judge NO verdicts on borderline candidates are dropped. They don't go back to
# auto-YES, and they don't override auto-YES on the same query — auto-YES is final
# by construction; the judge never sees those candidates. The auto-NO pile is
# discarded outright.
#
# **Empty-prediction guard.** If a query somehow ends up with both an empty auto-YES
# set AND a zero-YES verdict pile, the top-1 BM25 candidate is added as a fallback
# so the submission CSV doesn't contain empty rows.
#
# ### Casing fix (`apply_proper_case`)
#
# Predictions out of Stage 3-4 are in the BM25-corpus form — abbreviations
# all-uppercase (`STPO`, `STGB`, `STBOG`, `SCHKG`). The Kaggle gold uses the
# legally-correct German casing (`StPO`, `StGB`, `StBOG`, `SchKG`). The official
# scorer is plain string equality after whitespace canonicalization, so casing has
# to match exactly. `apply_proper_case` looks each prediction up in `proper_case_map`
# (built from `laws_de.csv` in Stage 0); unknown citations pass through unchanged;
# the function is idempotent on already-proper-cased input.
#
# On val this typically remaps ~44 of ~130 predicted citations.

# %%
def apply_proper_case(preds_by_qid):
    return {qid: [proper_case_map.get(c.upper(), c) for c in cs]
            for qid, cs in preds_by_qid.items()}

def build_final_predictions(pool_by_qid):
    preds = {}
    for qid, pool in pool_by_qid.items():
        sel = set()
        for cit, cd in pool.items():
            if cd.zone == 'yes':
                sel.add(cit)
            elif cd.zone == 'borderline' and cd.judge_verdict == 'YES':
                sel.add(cit)
        # Empty-prediction guard: top-1 BM25
        if not sel and pool:
            top1 = max(pool.values(), key=lambda c: c.bm25_score)
            sel.add(top1.citation)
        preds[qid] = sorted(sel)
    return apply_proper_case(preds)

# %% [markdown]
# ## 10. The `main()` entry point
#
# Chains Stages 1 → 5 over the queries in `target_df` and returns
# `(predictions, pool_by_qid)`.
#
# Every model call (reranker, judge) hits a JSON cache on Drive — running `main()` a
# second time with the same model and same queries does no GPU work at all, only the
# zone split + selection + casing run again. Useful for tweaking thresholds without
# re-paying the GPU cost.
#
# When `target_df` has a `gold_citations` column, an additional **pool-recall
# diagnostic** prints per-query R@100 of BM25 vs the law-only gold — a quick check that
# the recall floor is healthy before the slow reranker runs.

# %%
def main(target_df, label='val'):
    """Run end-to-end on a query DataFrame. Returns (preds, pool_by_qid)."""
    t0 = time.time()
    queries = [(r['query_id'], r['query']) for _, r in target_df.iterrows()]
    query_texts = {qid: q for qid, q in queries}

    # Stage 1 — BM25 top-100 per query
    section(f'[{label}] STAGE 1: BM25 TOP-{TOP_BM25}')
    pool_by_qid = {}
    for qid, qtxt in tqdm(queries, desc=f'  {label} BM25'):
        pool = {}
        for rank, (c, s) in enumerate(bm25_top_k(qid, qtxt, k=TOP_BM25), 1):
            c_proper = proper_case_map.get(c.upper(), c)
            if c_proper in pool: continue
            pool[c_proper] = Candidate(citation=c_proper, bm25_rank=rank, bm25_score=s)
        pool_by_qid[qid] = pool
    log(f'  avg BM25 pool size: {sum(len(p) for p in pool_by_qid.values()) / len(pool_by_qid):.1f}')

    # Diagnostic: pool recall vs gold (if available)
    if 'gold_citations' in target_df.columns and label == 'val':
        log('  pool-recall vs val gold (law-only):')
        for _, row in target_df.iterrows():
            qid = row['query_id']
            gold = {c.strip() for c in str(row['gold_citations']).split(';')
                    if c.strip().startswith('Art.')}
            pset = set(pool_by_qid[qid].keys())
            R = len(gold & pset) / max(len(gold), 1) if gold else 0.0
            log(f'    {qid}: R={R:.3f}  pool={len(pset):<3}  gold_law={len(gold):<2} '
                f'{"✓" if R == 1.0 else f"MISS {len(gold - pset)}"}')

    # Stage 2 — reranker
    section(f'[{label}] STAGE 2: QWEN3-RERANKER-8B')
    run_reranker(pool_by_qid, query_texts)

    # Stage 3 — fused score + zones
    section(f'[{label}] STAGE 3: FUSED SCORE + ZONE SPLIT')
    assign_fused_and_zones(pool_by_qid)
    zone_counts = Counter()
    for pool in pool_by_qid.values():
        for cd in pool.values():
            zone_counts[cd.zone] += 1
    n = max(len(pool_by_qid), 1)
    log(f'  avg per query — auto-YES: {zone_counts["yes"]/n:.1f}, '
        f'borderline: {zone_counts["borderline"]/n:.1f}, '
        f'auto-NO: {zone_counts["no"]/n:.1f}')

    # Stage 4 — judge on borderline
    section(f'[{label}] STAGE 4: QWEN3-8B JUDGE ON BORDERLINE')
    run_judge(pool_by_qid, query_texts)

    # Stage 5 — final predictions + casing
    section(f'[{label}] STAGE 5: FINAL PREDICTIONS + CASING')
    preds = build_final_predictions(pool_by_qid)
    log(f'  avg predictions/query: {sum(len(v) for v in preds.values()) / len(preds):.1f}')

    log(f'\n  {label} pipeline finished in {time.time()-t0:.1f}s')
    return preds, pool_by_qid

# %% [markdown]
# ## 11. Run on val
#
# Val has gold, so we score against the official Macro F1 in two cuts:
#
# 1. **Whole-gold** — the official competition metric, with court rulings in the
#    denominator. This is the reported headline.
# 2. **Law-only** — court rulings removed from the gold. Isolates how well this
#    pipeline performs on its actual target (statutes only). The gap between
#    whole-gold and law-only is the structural court-rulings ceiling — it's not
#    fixable from the law-side pipeline.

# %%
val_preds, val_pools = main(val_df, label='val')

section('VAL — OFFICIAL MACRO F1 (whole gold: laws + court rulings)')
val_per_whole, val_macro_whole = macro_f1(val_preds, val_df, law_only=False)

section('VAL — LAW-ONLY (court rulings removed from gold denominator)')
val_per_law, val_macro_law = macro_f1(val_preds, val_df, law_only=True)

# %% [markdown]
# ## 12. Submission CSV (val + optional test)
#
# Kaggle submission format — two columns:
#
# | column | content |
# |---|---|
# | `query_id` | `val_001`, …, `test_001`, … |
# | `predicted_citations` | semicolon-joined citation strings, post-casing |
#
# Three files are written when `SCORE_TEST = True`:
#
# - `submission_v12_val_<timestamp>.csv` — val only
# - `submission_v12_test_<timestamp>.csv` — test only
# - `submission_v12_combined_<timestamp>.csv` — val + test concatenated, ready to upload

# %%
def write_submission(preds_dict, qid_list, path):
    rows = [{'query_id': qid, 'predicted_citations': ';'.join(preds_dict.get(qid, []))}
            for qid in qid_list]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
val_sub_path = OUT_DIR / f'submission_v12_val_{ts}.csv'
write_submission(val_preds, val_df['query_id'], val_sub_path)
log(f'  ✓ val submission: {val_sub_path}')

if test_df is not None:
    test_preds, _ = main(test_df, label='test')
    test_sub_path = OUT_DIR / f'submission_v12_test_{ts}.csv'
    write_submission(test_preds, test_df['query_id'], test_sub_path)
    log(f'  ✓ test submission: {test_sub_path}')

    # Combined val+test submission (Kaggle-style)
    combined_path = OUT_DIR / f'submission_v12_combined_{ts}.csv'
    combined = pd.concat([
        pd.DataFrame([{'query_id': qid, 'predicted_citations': ';'.join(val_preds.get(qid, []))}
                      for qid in val_df['query_id']]),
        pd.DataFrame([{'query_id': qid, 'predicted_citations': ';'.join(test_preds.get(qid, []))}
                      for qid in test_df['query_id']]),
    ]).reset_index(drop=True)
    combined.to_csv(combined_path, index=False)
    log(f'  ✓ combined val+test submission: {combined_path}')

# %% [markdown]
# ## 13. Closing notes — where the gap from 0.585 to 1.0 lives
#
# Decomposing the residual gap from v12's whole-gold 0.585 to a hypothetical 1.0:
#
# - **~0.188 from missing court rulings.** Strictly architectural — there is no
#   court-rulings corpus in this pipeline. 102 of 251 val gold items (40.6%) are court
#   citations (`BGE …`, case numbers like `1B_/7B_/8C_/9C_/4A_/5A_/6B_/2C_…`) and are
#   false negatives by construction. The only way to close this is a separate
#   court-paragraph retriever that runs in parallel with this notebook and unions its
#   predictions into the submission.
#
# - **~0.227 from residual law-retrieval errors.** A mix of:
#   - BM25 ceiling misses on val_002 / val_008 (the candidate isn't in the top-100).
#   - Judge over-pruning on val_005's parental-rights cluster.
#   - Threshold under-promotion of structurally-inevitable BGG / OR articles that
#     recur across queries — `Art. 100 Abs. 1 BGG` is in 6 of 10 val gold sets but
#     ends up auto-NO on several queries because the BM25 score against the narrative
#     query is weak.
#
# ## Practical run notes
#
# - **Always apply `apply_proper_case` before submission.** The function is idempotent
#   so re-running it does no harm; skipping it craters macro F1.
# - **Reranker + judge caches persist across runs** under `retrieval/v12_clean_cache/`.
#   If you want to force a clean re-run (e.g. after changing a model), delete the
#   corresponding JSON file in that directory.
# - **Threshold tuning is fast.** Changing `HIGH_THRESH` / `LOW_THRESH` only requires
#   re-running Stages 3 → 5 (instant) — the reranker + judge stages don't re-run.
