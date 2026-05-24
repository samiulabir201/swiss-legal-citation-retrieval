# %% [markdown]
# # Beat v12 — law-side enhanced pipeline
#
# Stacks on top of `swiss_legal_citation_official_scoring_aligned_markdowns` (Hybrid v12, F1=0.585).
# Adds, in order, the research-derived channels and filters from
# `research/law_gold_retrieval_mechanism_2026-05-23.md`:
#
# * **Stage 0.5 — LLM expander** (Qwen3-8B). Predicts `legal_area`, `needed_roles`, and
#   `statute_mentions` for each query. Cached.
# * **Channel A — Statute parser** (deterministic regex; Bridge-A). Catches explicit
#   `Art. N (Abs. M)? CODE` mentions in DE/FR/IT/EN forms.
# * **Channel C — Procedural-apparatus injection** (Bridge-C). For the predicted
#   `legal_area`, inject every `laws_de` article whose code is in the area's code-set
#   AND whose `title` heading matches a procedural pattern (`Beschwerde`, `Berufung`,
#   `Verfahrenskosten`, …).
# * **Channel D — One-hop reference expansion** (Bridge-D). For each of v12's top-100
#   BM25 hits, regex `Art. N (Abs. M)? CODE` in the body text and add the referenced
#   articles to the pool.
# * **Specificity floor** — drop pool candidates with `specificity_score < 0.30`
#   (generic-header articles like `Allgemeine Bestimmungen`).
# * **Role-set filter on borderline** — drop borderline-zone candidates whose
#   `provision_role_llm` is not in the expander-predicted `needed_roles`.
# * **Enrichment-augmented judge text** — pass `english_summary + concepts_en` to the
#   judge instead of raw German body text (cross-lingual signal).
#
# Everything else (v12's TLF-enhanced BM25, Qwen3-Reranker, fused score, zone split,
# auto-YES / borderline / auto-NO, `apply_proper_case`, the 7-category judge prompt,
# the macro-F1 scorer) is unchanged.
#
# Run order: 1 → 2 → … → 13. Each cell prints progress and caches its output, so
# you can resume from the last completed stage.

# %%
# 1. Install deps + mount Drive
import sys, subprocess
def pip_install(pkgs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *pkgs])
try:
    import rank_bm25, transformers, accelerate, tqdm, pyarrow  # noqa
except ImportError:
    pip_install(['rank_bm25', 'transformers>=4.51', 'accelerate', 'tqdm', 'pandas', 'pyarrow', 'numpy'])

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print('Local mode (not Colab).')

# %%
# 2. Imports + configuration
import os, re, json, gc, time, pickle, datetime
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
np.random.seed(42)

# Paths — match v12's layout on Drive.
if IN_COLAB:
    DRIVE_BASE = Path('/content/drive/MyDrive/Omnilex-Agentic-Retrieval-Competition')
else:
    DRIVE_BASE = Path(r'E:\swiss_citation_extraction')

DATA_DIR    = DRIVE_BASE / 'data'
RETRIEVAL   = DRIVE_BASE / 'retrieval'
CACHE_DIR   = RETRIEVAL / 'beat_v12_law_cache'
OUT_DIR     = CACHE_DIR
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Auto-pick corpus parquet: prefer canonical "corpus.parquet" but accept "corpus_v2.parquet"
CORPUS_PARQUET = next(
    (p for p in [RETRIEVAL / 'corpus.parquet', RETRIEVAL / 'corpus_v2.parquet'] if p.exists()),
    RETRIEVAL / 'corpus.parquet')
BM25_INDEX_PKL = next(
    (p for p in [RETRIEVAL / 'bm25_v2_index.pkl', RETRIEVAL / 'bm25_index.pkl'] if p.exists()),
    RETRIEVAL / 'bm25_v2_index.pkl')
BM25_IDS_PKL   = next(
    (p for p in [RETRIEVAL / 'bm25_v2_ids.pkl', RETRIEVAL / 'bm25_ids.pkl'] if p.exists()),
    RETRIEVAL / 'bm25_v2_ids.pkl')
KB_JSONL       = RETRIEVAL / 'laws_knowledge_base.jsonl'
QT_DIR         = RETRIEVAL / 'knowledge_base_optimized_hybrid_retrieval'
ENRICHMENT_JL  = DRIVE_BASE / 'llm_enrichment_output_law_173k' / 'law_llm_descriptors_0000000_all.jsonl'

# Models
EXPANDER_MODEL = 'Qwen/Qwen3-8B'
RERANKER_MODEL = 'Qwen/Qwen3-Reranker-8B'
JUDGE_MODEL    = 'Qwen/Qwen3-8B'

# v12 thresholds — unchanged
TOP_BM25       = 100
HIGH_THRESH    = 0.55
LOW_THRESH     = 0.25
RERANK_BS      = 8
MAX_LEN        = 4096

# Research-derived filters
SPECIFICITY_FLOOR = 0.30
APPLY_ROLE_FILTER = True
APPLY_SPECIFICITY = True

# Channel toggles (turn off for ablation)
ADD_STATUTE_PARSER     = True
ADD_PROCEDURAL_CHANNEL = True
ADD_REF_EXPANSION      = True

# Cache files
EXPANSIONS_CACHE = CACHE_DIR / 'llm_expansions.json'
POOL_CACHE       = CACHE_DIR / 'enhanced_pool.json'
RERANK_CACHE     = CACHE_DIR / f'rerank_{RERANKER_MODEL.replace("/","_")}.json'
JUDGE_CACHE      = CACHE_DIR / f'judge_{JUDGE_MODEL.replace("/","_")}.json'
ENRICHMENT_PQ    = CACHE_DIR / 'enrichment_index.parquet'

def log(*a, **k): print(*a, **k, flush=True)
def section(t): log('\n' + '='*72); log(f'  {t}'); log('='*72)

log(f'DRIVE_BASE   : {DRIVE_BASE}')
log(f'CACHE_DIR    : {CACHE_DIR}')
log(f'CORPUS       : {CORPUS_PARQUET.name}')
log(f'BM25_INDEX   : {BM25_INDEX_PKL.name}')
log(f'BM25_IDS     : {BM25_IDS_PKL.name}')
# Sanity-check the BM25↔corpus alignment after both load (in Stage 0).

# %%
# 3. Utilities — citation parsing, tokenisation, casing, official metric
_TOKEN_RE = re.compile(r'[^\w\d]+', re.UNICODE)
_CIT_RE   = re.compile(r'^Art\.\s+(\d+[a-z]?)(?:\s+Abs\.\s+(\d+[a-z]*))?(?:\s+lit\.\s+(\S+))?\s+(\S+)$')

# DE/FR/IT/EN code aliases — canonicalize to DE
CODE_ALIASES = {
    'CPP':'StPO','CPC':'ZPO','LP':'SchKG','LEF':'SchKG','LTF':'BGG',
    'CC':'ZGB','CO':'OR','CP':'StGB','LAI':'IVG','LPGA':'ATSG','LDIP':'IPRG',
    'DPMin':'JStG','PPMin':'JStPO','PA':'VwVG','LAA':'UVG','LAVS':'AHVG',
    'LAMAL':'KVG','LPP':'BVG','CST':'BV','EMRK':'EMRK',
}

# Procedural-heading regex — broad cover of appeal / costs / defence patterns
PROCEDURAL_HEADING_PATTERNS = [
    r'\bbeschwerde\b', r'beschwerdefrist', r'beschwerdeverfahren', r'rechtsmittel',
    r'\bberufung\b', r'\brevision\b', r'verfahrenskosten', r'\bkosten\b',
    r'unentgeltlich', r'verteidigung', r'rechtspflegeverfahren', r'gerichtsverfahren',
    r'\bgrundrechte\b', r'rechtliches\s+geh', r'\bzust', r'kammern\b',
]
_PROC_RE = re.compile('|'.join(PROCEDURAL_HEADING_PATTERNS), re.IGNORECASE)

# Legal-area → code set (built from the procedural codes themselves; NOT val-tuned)
LEGAL_AREA_CODES = {
    'criminal-procedure':   {'StPO','StGB','StBOG','BGG','BV','MStG','MStP','JStG','JStPO'},
    'criminal-substantive': {'StGB','StPO','StBOG','BGG','BV'},
    'civil-contract':       {'OR','ZGB','ZPO','BGG'},
    'civil-family':         {'ZGB','ZPO','BGG'},
    'civil-inheritance':    {'ZGB','ZPO','BGG','OR'},
    'civil-property':       {'ZGB','OR','ZPO','BGG','IPRG'},
    'civil-tort':           {'OR','ZGB','BGG','SVG','UVG'},
    'social-insurance':     {'ATSG','IVG','UVG','KVG','AHVG','BVG','BGG'},
    'public-law':           {'BGG','BV','VwVG'},
    'constitutional':       {'BV','BGG'},
    'debt-enforcement':     {'SchKG','BGG','ZPO','OR','ZGB'},
    'tax':                  {'DBG','MWSTG','BGG'},
    'banking':              {'OR','BGG','FINMAG','SchKG'},
    'environment':          {'USG','UVPV','RPG','GschG','BGG'},
    'international-private':{'IPRG','BGG','ZGB','OR'},
}

def tokenise(text):
    if not text: return []
    return [w for w in _TOKEN_RE.split(text.lower().strip()) if len(w) >= 2 or w.isdigit()]

def fold_umlauts(s):
    return (s.lower().replace('ä','a').replace('ö','o').replace('ü','u').replace('ß','ss'))

def extract_heading(title):
    if not isinstance(title, str) or ' - ' not in title: return ''
    h = title.split(' - ', 1)[1].strip()
    h = re.sub(r'^[0-9]+\.\s+', '', h)
    h = re.sub(r'^[IVXLCDM]+\.\s+', '', h)
    h = re.sub(r'^[A-Za-z]\.\s+', '', h)
    h = re.sub(r'^\d+\s+(Kapitel|Abschnitt|Teil|Titel):\s+', '', h)
    h = re.sub(r'\d+$', '', h).strip()
    return h

def parse_citation(c):
    m = _CIT_RE.match(str(c).strip())
    if not m: return None
    return {'art': m.group(1), 'abs': m.group(2), 'lit': m.group(3), 'code': m.group(4)}

# Statute-mention regex covering DE / FR / IT / EN forms in a query
_STATUTE_QRE = re.compile(
    r'(?:Art(?:icle|\.?)?|art(?:\.|icolo)?)\s*'
    r'(\d+[a-z]?)'
    r'(?:\s*(?:Abs\.?|al\.?|cpv\.?|\()\s*(\d+[a-z]*))?'
    r'(?:\s*\)?\s*(?:lit\.?|let\.?|lett\.?|\()\s*([a-z]+))?'
    r'(?:\s*\)?)?'
    r'\s+(?:de\s+la\s+|du\s+|della?\s+|of\s+the\s+)?'
    r'([A-Z][A-Za-z]{1,8})', re.IGNORECASE)

def canonical_code(c):
    c = c.strip().strip('.,;:()[]')
    return CODE_ALIASES.get(c.upper(), c)

def parse_statute_mentions(text):
    out = set()
    for m in _STATUTE_QRE.finditer(text or ''):
        art, abs_, _lit, code = m.groups()
        code = canonical_code(code)
        if abs_:
            out.add(f'Art. {art} Abs. {abs_} {code}')
        out.add(f'Art. {art} {code}')  # also include parent — fan out later
    return out

# Official scorer (v12-aligned)
_WS = re.compile(r'\s+')
def _canon(c): return _WS.sub(' ', str(c).strip())
def _parse_field(v):
    if pd.isna(v): return set()
    if not isinstance(v, str): v = str(v)
    return {_canon(p) for p in v.split(';') if _canon(p)}
def macro_f1(pred_by_qid, df):
    rows = []
    for _, r in df.iterrows():
        qid = r['query_id']
        gold = _parse_field(r.get('gold_citations',''))
        pred = _parse_field(';'.join(pred_by_qid.get(qid, [])))
        if not pred and not gold:
            P=R=F=1.0
        elif not pred or not gold:
            P=R=F=0.0
        else:
            tp = len(pred & gold)
            P = tp/len(pred); R = tp/len(gold)
            F = 2*P*R/(P+R) if (P+R) else 0.0
        rows.append({'query_id':qid,'P':P,'R':R,'F1':F,'pred':len(pred),'gold':len(gold)})
    out = pd.DataFrame(rows)
    return out, out[['P','R','F1']].mean().to_dict()

# %%
# 4. Stage 0 — Load v12 artifacts (corpus, BM25, KB, query_translations, TLF, proper_case map)
section('STAGE 0: LOAD v12 ARTIFACTS')

log('  loading corpus.parquet ...')
corpus = pd.read_parquet(CORPUS_PARQUET)
log(f'  corpus: {len(corpus):,} rows; columns: {list(corpus.columns)}')

log('  loading bm25_v2_index.pkl ...')
with open(BM25_INDEX_PKL,'rb') as f: bm25 = pickle.load(f)
with open(BM25_IDS_PKL,'rb')   as f: bm25_meta = pickle.load(f)
# v12's bm25_v2_ids.pkl is a dict {citation_canon: [...], best_k1, best_b, best_f1_k}
# rather than a plain list. Unwrap it and apply the tuned hyperparams.
if isinstance(bm25_meta, dict):
    bm25_ids = bm25_meta.get('citation_canon') or bm25_meta.get('ids') or []
    if 'best_k1' in bm25_meta:
        bm25.k1 = float(bm25_meta['best_k1'])
    if 'best_b' in bm25_meta:
        bm25.b  = float(bm25_meta['best_b'])
    log(f'  bm25 hyperparams: k1={bm25.k1:.3f}, b={bm25.b:.3f}, target_k={bm25_meta.get("best_f1_k")}')
else:
    bm25_ids = bm25_meta
log(f'  bm25: {len(bm25_ids):,} docs')

log('  loading laws_knowledge_base.jsonl ...')
kb = {}
with open(KB_JSONL,'r',encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        kb[r.get('citation_canon') or r.get('citation') or ''] = r
log(f'  KB: {len(kb):,} entries')

# query translations (cached EN→DE per query)
qt_files = sorted(QT_DIR.glob('query_translations*.json'))
log(f'  query_translations files: {[p.name for p in qt_files]}')
query_translations = {}
for p in qt_files:
    query_translations.update(json.loads(p.read_text(encoding='utf-8')))
log(f'  query_translations: {len(query_translations)} entries')

# Build TLF prior from train.csv (v12 logic)
log('  mining TLF prior from train.csv ...')
train_df = pd.read_csv(DATA_DIR / 'train.csv')
val_df   = pd.read_csv(DATA_DIR / 'val.csv')
laws_de  = pd.read_csv(DATA_DIR / 'laws_de.csv')

def extract_law_abbrev(citation):
    parts = str(citation).strip().split()
    if not parts: return ''
    return parts[-1]

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
log(f'  TLF: {len(tlf):,} distinct tokens')

# Build proper-case map from laws_de.csv (the canonical Swiss-German case)
proper_case_map = {c.upper(): c for c in laws_de['citation'].dropna() if c}
log(f'  proper-case map: {len(proper_case_map):,} entries')

# Children index for parent → Abs.-children fanout
children_index = defaultdict(list)
laws_set = set(laws_de['citation'].dropna())
for c in laws_set:
    p = parse_citation(c)
    if p:
        children_index[(p['art'], p['code'])].append(c)
log(f'  children_index: {len(children_index):,} parent keys')

# Corpus citation lookup (v12's corpus uses citation_canon; map to/from laws_de canonical)
corpus_cit_col = 'citation_canon' if 'citation_canon' in corpus.columns else 'citation'
corpus_cit = corpus[corpus_cit_col].tolist()
corpus_cit_upper_to_orig = {c.upper(): c for c in corpus_cit}
log(f'  corpus uses citation column: {corpus_cit_col}')

# Sanity-check: BM25 IDs MUST align with the corpus we loaded.
# If they don't match the BM25 retrieval would point to wrong articles.
_corpus_set = set(corpus_cit)
_bm25_set   = set(bm25_ids)
_overlap = len(_corpus_set & _bm25_set)
if len(bm25_ids) != len(corpus):
    log(f'  ⚠ row-count mismatch: corpus={len(corpus)}, bm25={len(bm25_ids)} — '
        f'check that corpus.parquet matches the BM25 build')
if _overlap < 0.95 * len(bm25_ids):
    log(f'  ⚠ citation overlap only {_overlap}/{len(bm25_ids)} ({100*_overlap/max(len(bm25_ids),1):.1f}%) — '
        f'corpus and BM25 ids look misaligned')
else:
    log(f'  ✓ corpus↔BM25 alignment: {_overlap}/{len(bm25_ids)} citations overlap '
        f'({100*_overlap/max(len(bm25_ids),1):.1f}%)')

# %%
# 5. Build/load enrichment index (provision_role + specificity_score per article)
section('STAGE 0b: ENRICHMENT INDEX')

if ENRICHMENT_PQ.exists():
    enrich = pd.read_parquet(ENRICHMENT_PQ)
    log(f'  loaded enrichment cache: {len(enrich):,} rows')
else:
    log(f'  parsing {ENRICHMENT_JL.name} (this is one-time, ~10s) ...')
    rows = []
    with open(ENRICHMENT_JL, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=175000, desc='enrichment', unit='line'):
            if not line.strip(): continue
            obj = json.loads(line)
            enr = obj.get('llm_enrichment') or {}
            rows.append({
                'citation':        obj.get('citation') or '',
                'english_summary': enr.get('english_summary') or '',
                'concepts_en':     ' | '.join(enr.get('concepts_en') or []),
                'legal_question':  enr.get('legal_question') or '',
                'applicability':   ' | '.join(enr.get('applicability_conditions') or []),
                'provision_role':  enr.get('provision_role_llm') or '',
                'specificity':     float(enr.get('specificity_score') or 0.0),
            })
    enrich = pd.DataFrame(rows)
    enrich.to_parquet(ENRICHMENT_PQ, index=False)
    log(f'  built + saved {len(enrich):,} rows -> {ENRICHMENT_PQ}')

# lookup: cit -> dict of (specificity, provision_role, english_summary, concepts_en)
enrich_lookup = {r['citation']: r for _, r in enrich.iterrows()}
log(f'  enrich_lookup: {len(enrich_lookup):,}')

# Provision-role frequency to inform the expander prompt
role_counts = enrich['provision_role'].value_counts()
log(f'  top provision_role values: {list(role_counts.head(15).items())}')

# %%
# 6. Stage 0.5 — LLM expander (Qwen3-8B). Predicts legal_area + needed_roles + statute_mentions per query.
section('STAGE 0.5: LLM EXPANDER (Qwen3-8B)')

EXPANDER_SYSTEM = '''You are a Swiss Federal Court (Bundesgericht) legal-citation expert.

Given an English legal scenario, output ONE JSON object with these exact fields:

{
  "legal_area": one of ["criminal-procedure","criminal-substantive","civil-contract",
    "civil-family","civil-inheritance","civil-property","civil-tort","social-insurance",
    "public-law","constitutional","debt-enforcement","tax","banking","environment",
    "international-private"],
  "secondary_areas": [optional list, same vocabulary, empty if none],
  "statute_mentions": [list of explicit statute references appearing in the query, normalised
    to "Art. N Abs. M CODE" using the Swiss-German abbreviation. CODE must be one of
    StPO, StGB, ZGB, OR, IVG, ATSG, BGG, BV, StBOG, ZPO, SchKG, IPRG, UVG, KVG, AHVG,
    BVG, VwVG, USG, RPG, JStG, JStPO, DBG, MWSTG, MStG, MStP, AsylG, SVG.
    Convert FR/IT codes: LAI→IVG, CPP→StPO, CC→ZGB, CO→OR, LPGA→ATSG, LTF→BGG, CP→StGB,
    CPC→ZPO, LP→SchKG, LDIP→IPRG.],
  "needed_roles": [subset of the role vocabulary describing what kinds of provisions
    are needed to answer the query — pick all that apply. Vocabulary:
    "substantive_rule" (the actual legal rule the case turns on),
    "definition" (defines a term used in the question),
    "procedural_remedy" (how the affected party challenges or seeks review),
    "appeal_route" (which court hears the appeal, time-limit, format),
    "cost_apportionment" (who pays what costs),
    "jurisdiction_rule" (which body has authority),
    "evidentiary_rule" (burden of proof, admissibility),
    "constitutional_principle" (right to be heard, fair trial),
    "sanction_or_penalty"]
}

Rules:
- Output ONLY the JSON object. No prose. No <think>. No code-fence.
- legal_area: pick the single most-specific area.
- needed_roles: criminal/civil appeal queries almost always need at least
  ["substantive_rule","procedural_remedy","appeal_route","cost_apportionment","jurisdiction_rule"].
'''

def load_expander():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    log(f'  loading expander: {EXPANDER_MODEL}')
    tok = AutoTokenizer.from_pretrained(EXPANDER_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        EXPANDER_MODEL, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    model.eval()
    return tok, model

def run_expander(queries, cache_path):
    if cache_path.exists():
        out = json.loads(cache_path.read_text(encoding='utf-8'))
        log(f'  loaded cached expansions: {len(out)} queries')
        # check if all needed queries are present; if so, return
        if all(qid in out for qid,_ in queries):
            return out
        log(f'  cache missing some queries, will fill in')
    else:
        out = {}

    import torch
    tok, model = load_expander()
    for qid, qtxt in tqdm(queries, desc='expander'):
        if qid in out: continue
        msgs = [
            {'role':'system','content':EXPANDER_SYSTEM},
            {'role':'user','content':qtxt}
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=600, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        txt = tok.decode(gen[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        # strip <think>...</think>
        if '</think>' in txt:
            txt = txt.split('</think>',1)[1]
        # find the first JSON object
        m = re.search(r'\{[\s\S]*\}', txt)
        parsed = {'legal_area':'', 'secondary_areas':[], 'statute_mentions':[], 'needed_roles':[]}
        if m:
            try:
                parsed.update(json.loads(m.group(0)))
            except Exception as e:
                log(f'  WARN parse fail for {qid}: {e}')
        out[qid] = parsed
        cache_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    # free model
    del model, tok
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    return out

# Run on val + test (test has no gold but produces submission)
test_df = None
try:
    test_df = pd.read_csv(DATA_DIR / 'test.csv')
except Exception as e:
    log(f'  no test.csv: {e}')

QUERIES_FOR_EXPANSION = [(r['query_id'], r['query']) for _, r in val_df.iterrows()]
if test_df is not None:
    QUERIES_FOR_EXPANSION += [(r['query_id'], r['query']) for _, r in test_df.iterrows()]

expansions = run_expander(QUERIES_FOR_EXPANSION, EXPANSIONS_CACHE)
log(f'  expansions: {len(expansions)} cached')
# show one for sanity
sample_qid = QUERIES_FOR_EXPANSION[0][0]
log(f'  sample {sample_qid}: {json.dumps(expansions[sample_qid], indent=2, ensure_ascii=False)[:400]}')

# %%
# 7. v12 Stage 1 — TLF-enhanced BM25 (unchanged from v12)
section('STAGE 1: TLF-ENHANCED BM25 (v12 spine)')

def enhance_tokens_with_tlf(tokens, top_abbrev=5, repeats=5):
    scores = defaultdict(float)
    for t in tokens:
        dist = tlf.get(t)
        if not dist: continue
        n = sum(dist.values())
        if n == 0: continue
        idf = math.log((len(token_total)+1) / (token_total[t]+1)) if token_total[t] else 1.0
        for abbrev, cnt in dist.items():
            scores[abbrev] += (cnt/n) * idf
    top = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_abbrev]
    return tokens + [a for a in top for _ in range(repeats)]

import math

def query_tokens_v12(qid, qtxt):
    en = tokenise(qtxt)
    de = tokenise(query_translations.get(qid, ''))
    base = []
    seen = set()
    for t in en + de:
        if t in seen: continue
        seen.add(t); base.append(t)
    enhanced = enhance_tokens_with_tlf(base)
    return enhanced

def bm25_top_k(qid, qtxt, k=TOP_BM25):
    toks = query_tokens_v12(qid, qtxt)
    scores = bm25.get_scores(toks)
    order = np.argsort(scores)[::-1][:k]
    return [(bm25_ids[i], float(scores[i])) for i in order if scores[i] > 0]

# Test on val_001
sample_top = bm25_top_k(*QUERIES_FOR_EXPANSION[0], k=10)
log(f'  sample top-10 for {QUERIES_FOR_EXPANSION[0][0]}: '
    + ', '.join([c for c,_ in sample_top[:5]]) + ' ...')

# %%
# 8. NEW Channel A — Statute parser (deterministic Bridge-A)
def channel_statute(qid, qtxt, expansion):
    mentions = parse_statute_mentions(qtxt + ' ' + query_translations.get(qid,''))
    for s in expansion.get('statute_mentions', []) or []:
        mentions.update(parse_statute_mentions(s))
        if s in laws_set: mentions.add(s)
    hits = set()
    for m in mentions:
        if m in laws_set:
            hits.add(m); continue
        # parent fanout
        p = parse_citation(m)
        if p and not p['abs']:
            for ch in children_index.get((p['art'], p['code']), []):
                hits.add(ch)
    return hits

# 9. NEW Channel C — procedural-apparatus injection (Bridge-C)
# Build once: heading-normalised view of laws_de
laws_de_clean = laws_de.dropna(subset=['citation']).copy()
laws_de_clean['heading_norm'] = laws_de_clean['title'].fillna('').map(
    lambda t: fold_umlauts(extract_heading(t)))
laws_de_clean['code'] = laws_de_clean['citation'].map(lambda c: (parse_citation(c) or {}).get('code',''))

def channel_procedural(qid, expansion):
    areas = [expansion.get('legal_area','')] + (expansion.get('secondary_areas', []) or [])
    codes = set()
    for a in areas:
        if a in LEGAL_AREA_CODES:
            codes.update(LEGAL_AREA_CODES[a])
    if not codes:
        codes = {'BGG'}
    hits = set()
    for code in codes:
        sub = laws_de_clean[laws_de_clean['code']==code]
        for _, r in sub.iterrows():
            if r['heading_norm'] and _PROC_RE.search(r['heading_norm']):
                hits.add(r['citation'])
    return hits

# 10. NEW Channel D — one-hop reference expansion (Bridge-D)
# Build a lookup: cit -> body text (from laws_de.csv)
LAW_TEXT = {c: (t if isinstance(t,str) else '')
            for c, t in zip(laws_de['citation'], laws_de['text'])}

def channel_ref_expansion(seed_citations):
    out = set()
    for sc in seed_citations:
        body = LAW_TEXT.get(sc, '')
        if not body: continue
        for m in parse_statute_mentions(body):
            if m in laws_set:
                out.add(m); continue
            p = parse_citation(m)
            if p and not p['abs']:
                for ch in children_index.get((p['art'], p['code']), []):
                    out.add(ch)
    return out

log('  channel functions defined (statute / procedural / ref-expansion)')

# %%
# 11. Stage 1.5 — Build the enhanced candidate pool (union BM25 + new channels)
section('STAGE 1.5: ENHANCED POOL CONSTRUCTION')

@dataclass
class Candidate:
    citation: str
    bm25_score: float = 0.0
    bm25_rank: int = 10**9
    channels: set = field(default_factory=set)
    reranker_score: float = 0.0
    fused_score: float = 0.0
    zone: str = 'no'
    judge_verdict: str = ''
    specificity: float = 0.0
    provision_role: str = ''

def build_enhanced_pool(qid, qtxt):
    expansion = expansions.get(qid, {})
    pool = {}
    def get(cit):
        if cit not in pool: pool[cit] = Candidate(citation=cit)
        return pool[cit]

    # v12 BM25 top-100 — the spine
    bm25_hits = bm25_top_k(qid, qtxt, k=TOP_BM25)
    for rank, (c, s) in enumerate(bm25_hits, 1):
        # corpus citations may be uppercase; normalize via proper_case_map
        c_proper = proper_case_map.get(c.upper(), c)
        cd = get(c_proper); cd.channels.add('bm25_v12')
        cd.bm25_score = max(cd.bm25_score, s); cd.bm25_rank = min(cd.bm25_rank, rank)

    if ADD_STATUTE_PARSER:
        for c in channel_statute(qid, qtxt, expansion):
            cd = get(c); cd.channels.add('statute_parser')
    if ADD_PROCEDURAL_CHANNEL:
        for c in channel_procedural(qid, expansion):
            cd = get(c); cd.channels.add('procedural')
    if ADD_REF_EXPANSION:
        seeds = list(pool.keys())
        for c in channel_ref_expansion(seeds):
            cd = get(c); cd.channels.add('ref_expansion')

    # Attach enrichment metadata
    for cit, cd in pool.items():
        r = enrich_lookup.get(cit)
        if r is not None:
            cd.specificity = float(r.get('specificity') or 0.0)
            cd.provision_role = str(r.get('provision_role') or '')
    return pool

# Build pools for all val + test
pools = {}
all_queries = [(r['query_id'], r['query']) for _, r in val_df.iterrows()]
if test_df is not None:
    all_queries += [(r['query_id'], r['query']) for _, r in test_df.iterrows()]

for qid, qtxt in tqdm(all_queries, desc='pool'):
    pools[qid] = build_enhanced_pool(qid, qtxt)

avg_pool = sum(len(p) for p in pools.values()) / max(len(pools),1)
log(f'  avg pool size: {avg_pool:.1f}')

# Diagnostic: pool-recall vs val gold
log('\n  Pool-recall vs val gold (law-only):')
for _, row in val_df.iterrows():
    qid = row['query_id']
    gold = {c.strip() for c in str(row['gold_citations']).split(';')
            if c.strip().startswith('Art.')}
    pset = set(pools[qid].keys())
    R = len(gold & pset) / max(len(gold),1) if gold else 0.0
    miss = sorted(gold - pset)
    tag = '✓' if R==1.0 else f'MISS {len(miss)}'
    log(f'    {qid}: R={R:.3f} pool={len(pset)} gold={len(gold)} {tag}')
    for m in miss[:5]:
        log(f'        MISSING: {m}')

# Apply specificity floor BEFORE reranking
if APPLY_SPECIFICITY:
    n_dropped = 0
    for qid, pool in pools.items():
        for cit in list(pool):
            if pool[cit].specificity > 0 and pool[cit].specificity < SPECIFICITY_FLOOR:
                # only drop if NOT a statute_parser hit (those are always-include)
                if 'statute_parser' not in pool[cit].channels:
                    pool.pop(cit, None); n_dropped += 1
    log(f'  specificity floor dropped {n_dropped} candidates total')
    avg_pool = sum(len(p) for p in pools.values()) / max(len(pools),1)
    log(f'  avg pool after floor: {avg_pool:.1f}')

# Save pool snapshot
pool_dump = {qid: [{'c':cd.citation, 'bm25':cd.bm25_score, 'rank':cd.bm25_rank,
                    'channels':sorted(cd.channels), 'spec':cd.specificity,
                    'role':cd.provision_role}
                   for cd in pool.values()]
             for qid, pool in pools.items()}
POOL_CACHE.write_text(json.dumps(pool_dump, indent=2, ensure_ascii=False), encoding='utf-8')
log(f'  saved pool snapshot: {POOL_CACHE}')

# %%
# 12. Stage 2 — Qwen3-Reranker-8B (v12 spine)
section('STAGE 2: QWEN3-RERANKER-8B')

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

RERANK_INSTRUCTION = ('Given a query about Swiss law, determine whether the provided law '
                      'article is related to or applicable to the legal issue.')

def candidate_doc_text_for_reranker(cit):
    # v12 uses citation + abbrev + English name + heading + provision_type + first 500 chars body
    body = LAW_TEXT.get(cit, '')
    h = ''
    r = laws_de_clean.loc[laws_de_clean['citation']==cit]
    if not r.empty:
        h = r.iloc[0]['title'] or ''
    return f'{cit}\nTitle: {h}\nText: {body[:500]}'

def run_reranker(pools, query_texts, cache_path):
    if cache_path.exists():
        out = json.loads(cache_path.read_text(encoding='utf-8'))
    else:
        out = {}
    import torch
    tok, model, prefix, suffix, yes_id, no_id = load_reranker()

    for qid, pool in tqdm(pools.items(), desc='rerank queries'):
        if qid in out and len(out[qid]) >= len(pool):
            # apply cached
            for cit, cd in pool.items():
                cd.reranker_score = float(out[qid].get(cit, 0.0))
            continue
        out.setdefault(qid, {})
        q = query_texts[qid]
        cits = [c for c in pool.keys() if c not in out[qid]]
        if not cits:
            for cit, cd in pool.items():
                cd.reranker_score = float(out[qid].get(cit, 0.0))
            continue
        pairs = [
            f'<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {q}\n<Document>: {candidate_doc_text_for_reranker(c)}'
            for c in cits]
        for i in tqdm(range(0, len(pairs), RERANK_BS), desc=f'  {qid}', leave=False):
            batch = pairs[i:i+RERANK_BS]
            ids = [tok.encode(b, add_special_tokens=False, truncation=True,
                              max_length=MAX_LEN - len(prefix) - len(suffix))
                   for b in batch]
            ids = [prefix + x + suffix for x in ids]
            maxlen = max(len(x) for x in ids)
            attn  = [[0]*(maxlen-len(x)) + [1]*len(x) for x in ids]
            padded= [[tok.pad_token_id or 0]*(maxlen-len(x)) + x for x in ids]
            inp = torch.tensor(padded, device=model.device)
            am  = torch.tensor(attn,   device=model.device)
            with torch.no_grad():
                logits = model(input_ids=inp, attention_mask=am).logits[:,-1,:]
            yes = logits[:, yes_id]
            no_ = logits[:, no_id]
            probs = torch.softmax(torch.stack([no_, yes], dim=-1), dim=-1)[:,1].float().cpu().numpy()
            for c, p in zip(cits[i:i+RERANK_BS], probs):
                out[qid][c] = float(p)
                pool[c].reranker_score = float(p)
        cache_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    del model, tok
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    return out

query_texts = {qid:q for qid,q in all_queries}
rerank_scores = run_reranker(pools, query_texts, RERANK_CACHE)
log(f'  reranked {sum(len(v) for v in rerank_scores.values())} (query,doc) pairs')

# %%
# 13. Stage 3 — Fused score + zone split (v12 spine)
section('STAGE 3: FUSED SCORE + ZONE SPLIT')

def assign_zones(pools):
    for qid, pool in pools.items():
        if not pool: continue
        bm = np.array([cd.bm25_score for cd in pool.values()])
        lo, hi = bm.min(), bm.max()
        rng = hi - lo if hi > lo else 1.0
        for cd in pool.values():
            nb = (cd.bm25_score - lo) / rng
            cd.fused_score = 0.7*nb + 0.3*cd.reranker_score
            if cd.fused_score >= HIGH_THRESH: cd.zone = 'yes'
            elif cd.fused_score < LOW_THRESH: cd.zone = 'no'
            else: cd.zone = 'borderline'
        # statute_parser candidates are ALWAYS auto-YES (deterministic Bridge-A)
        for cd in pool.values():
            if 'statute_parser' in cd.channels:
                cd.zone = 'yes'

assign_zones(pools)
for qid in [qq[0] for qq in QUERIES_FOR_EXPANSION[:3]]:
    pool = pools[qid]
    by_zone = Counter(cd.zone for cd in pool.values())
    log(f'  {qid}: {dict(by_zone)}')

# %%
# 14. NEW filter — role-set filter on borderline pile (research add)
if APPLY_ROLE_FILTER:
    n_dropped = 0
    for qid, pool in pools.items():
        needed = set(expansions.get(qid,{}).get('needed_roles', []) or [])
        if not needed:
            continue
        for cit, cd in list(pool.items()):
            if cd.zone != 'borderline': continue
            if cd.provision_role and cd.provision_role not in needed:
                # drop from borderline (move to 'no' = won't reach judge)
                cd.zone = 'no'; n_dropped += 1
    log(f'  role filter pruned {n_dropped} borderline candidates')

# %%
# 15. Stage 4 — Qwen3-8B judge on borderline (with enrichment-augmented doc text)
section('STAGE 4: QWEN3-8B JUDGE ON BORDERLINE')

JUDGE_SYSTEM = '''You are a Swiss Federal Court (Bundesgericht) citation expert.

For each candidate citation, decide YES or NO: would an expert Swiss-law annotator
include this citation in the gold-standard answer set for the query?

Swiss-law gold sets typically include articles from the seven roles:
1. SUBSTANTIVE LAW — the rule the case turns on
2. DEFINITIONS — terms used in the legal question
3. PROCEDURAL RULES — how the case is litigated
4. APPEAL PROVISIONS — who, when, where, how to appeal (always include if a BGer appeal)
5. COST ALLOCATION — who pays the court costs
6. COURT JURISDICTION — which chamber / body decides
7. CONSTITUTIONAL PRINCIPLES — right to be heard, proportionality, fair trial

Say YES if the candidate fits ANY of the seven roles for THIS query.
Say NO only if it's from a completely unrelated legal domain.

Output ONLY a JSON list, one verdict per candidate, in the input order:
[{"i":1,"v":"YES"},{"i":2,"v":"NO"},...]
'''

def judge_doc_text(cit, cd):
    # research add — use enrichment english fields where available
    r = enrich_lookup.get(cit)
    if r is not None and (r.get('english_summary') or r.get('concepts_en')):
        en = r.get('english_summary','')
        cc = r.get('concepts_en','')
        return f"{cit}\nEnglish summary: {en[:400]}\nConcepts: {cc[:200]}"
    # fallback to German body
    return candidate_doc_text_for_reranker(cit)[:600]

def run_judge(pools, query_texts, cache_path):
    if cache_path.exists():
        out = json.loads(cache_path.read_text(encoding='utf-8'))
    else:
        out = {}
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    log(f'  loading judge: {JUDGE_MODEL}')
    tok = AutoTokenizer.from_pretrained(JUDGE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
    model.eval()

    for qid, pool in tqdm(pools.items(), desc='judge'):
        borderline = [(c,cd) for c,cd in pool.items() if cd.zone == 'borderline']
        if not borderline:
            out.setdefault(qid, {})
            continue
        if qid in out and all(c in out[qid] for c,_ in borderline):
            for c, cd in borderline:
                cd.judge_verdict = out[qid].get(c, 'YES')
            continue
        out.setdefault(qid, {})
        # build prompt
        cand_block = '\n'.join(
            f'[{i+1}] {judge_doc_text(c, cd)}'
            for i,(c,cd) in enumerate(borderline))
        user_msg = f'Query: {query_texts[qid]}\n\nCandidates:\n{cand_block}\n\nVerdicts (JSON list):'
        msgs = [{'role':'system','content':JUDGE_SYSTEM},
                {'role':'user','content':user_msg}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors='pt', truncation=True, max_length=8192).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=3000, do_sample=True,
                                 temperature=0.5, top_k=20, top_p=0.95,
                                 pad_token_id=tok.eos_token_id)
        txt = tok.decode(gen[0, inp['input_ids'].shape[1]:], skip_special_tokens=True)
        if '</think>' in txt: txt = txt.split('</think>',1)[1]
        # parse JSON list
        m = re.search(r'\[[\s\S]*\]', txt)
        verdicts = {}
        if m:
            try:
                arr = json.loads(m.group(0))
                for item in arr:
                    if not isinstance(item, dict): continue
                    i = int(item.get('i', 0))
                    v = str(item.get('v','YES')).upper().strip()
                    if 1 <= i <= len(borderline):
                        verdicts[borderline[i-1][0]] = 'YES' if v=='YES' else 'NO'
            except Exception as e:
                log(f'  WARN judge parse {qid}: {e}')
        # default-YES on missing (recall-preserving)
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

judge_verdicts = run_judge(pools, query_texts, JUDGE_CACHE)
log(f'  judge verdicts: {sum(len(v) for v in judge_verdicts.values())} total')

# %%
# 16. Stage 5 — Build final predictions + apply_proper_case + score
section('STAGE 5: FINAL PREDICTIONS')

def apply_proper_case(preds_by_qid):
    out = {}
    for qid, cs in preds_by_qid.items():
        out[qid] = [proper_case_map.get(c.upper(), c) for c in cs]
    return out

preds = {}
for qid, pool in pools.items():
    pred = set()
    for cit, cd in pool.items():
        if cd.zone == 'yes':
            pred.add(cit)
        elif cd.zone == 'borderline' and cd.judge_verdict == 'YES':
            pred.add(cit)
    # empty-prediction guard: top-1 BM25
    if not pred and pool:
        top = sorted(pool.values(), key=lambda cd: cd.bm25_score, reverse=True)[0]
        pred.add(top.citation)
    preds[qid] = sorted(pred)

preds = apply_proper_case(preds)

# Show val macro F1
val_per, val_macro = macro_f1(preds, val_df)
section('VAL MACRO F1 — OFFICIAL SCORER (whole gold, laws + court)')
log(val_per.to_string(index=False))
log(f"\n  MACRO P={val_macro['P']:.3f}  R={val_macro['R']:.3f}  F1={val_macro['F1']:.3f}")

# Law-only F1 — restrict gold to Art.-prefixed citations
def macro_f1_lawonly(preds_by_qid, df):
    df2 = df.copy()
    df2['gold_citations'] = df2['gold_citations'].fillna('').map(
        lambda v: ';'.join(c for c in str(v).split(';') if c.strip().startswith('Art.')))
    return macro_f1({qid:[p for p in cs if p.startswith('Art.')] for qid,cs in preds_by_qid.items()}, df2)

val_per_law, val_macro_law = macro_f1_lawonly(preds, val_df)
section('VAL MACRO F1 — LAW-ONLY (court citations removed from gold)')
log(val_per_law.to_string(index=False))
log(f"\n  MACRO P={val_macro_law['P']:.3f}  R={val_macro_law['R']:.3f}  F1={val_macro_law['F1']:.3f}")

# Write submission
SUB_PATH = OUT_DIR / f'submission_val_beat_v12_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv'
sub_rows = []
for qid in val_df['query_id']:
    cs = preds.get(qid, [])
    sub_rows.append({'query_id': qid, 'predicted_citations': ';'.join(cs)})
if test_df is not None:
    for qid in test_df['query_id']:
        cs = preds.get(qid, [])
        sub_rows.append({'query_id': qid, 'predicted_citations': ';'.join(cs)})
pd.DataFrame(sub_rows).to_csv(SUB_PATH, index=False)
log(f'\n  wrote {SUB_PATH}')

# %% [markdown]
# # Ablations
#
# To turn off any single research add and re-measure, restart from cell 11 with:
#
# ```
# ADD_STATUTE_PARSER     = False   # disable Channel A
# ADD_PROCEDURAL_CHANNEL = False   # disable Channel C
# ADD_REF_EXPANSION      = False   # disable Channel D
# APPLY_SPECIFICITY      = False   # disable specificity floor
# APPLY_ROLE_FILTER      = False   # disable role-filter on borderline
# ```
#
# Re-run cells 11 → 16. The cached expander + reranker + judge caches will be reused
# where possible; only the affected stages re-compute.
