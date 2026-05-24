# %% [markdown]
# # Beat v12 — val-only, count-calibrated, fresh run (self-contained)
#
# Single notebook. No dependency on v1 caches. Val-only.
#
# Pipeline:
#  Stage 0  — Load v12 artifacts (corpus, BM25, query_translations, TLF, proper-case)
#  Stage 0b — Build enrichment parquet (provision_role + specificity + english_summary)
#  Stage 0.5 — Qwen3-14B expander → legal_area + statute_mentions + needed_roles + expected_citation_count
#  Stage 1  — TLF-enhanced BM25 top-100 (v12 spine)
#  Stage 1.5 — Add channels A (statute parser), C (procedural apparatus), D (ref expansion)
#  Stage 2  — Qwen3-Reranker-8B over the enhanced pool
#  Stage 3  — combined_score = blend of BM25 + reranker + channels + role + specificity
#  Stage 4  — STRICT TOP-K SELECTION (K = expected_count, clamped to [4, 30])
#  Stage 5  — apply_proper_case + macro F1 (whole-gold and law-only)
#
# Runtime on A100: ~15-20 min total. Caches every stage to Drive for fast re-runs.

# %%
# 1. Install + Drive
import sys, subprocess
try:
    import rank_bm25, transformers, accelerate, tqdm, pyarrow  # noqa
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
        'rank_bm25', 'transformers>=4.51', 'accelerate', 'tqdm', 'pandas', 'pyarrow', 'numpy'])
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# %%
# 2. Imports + config
import os, re, json, gc, time, pickle, datetime, math, ast
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
np.random.seed(42)

if IN_COLAB:
    DRIVE_BASE = Path('/content/drive/MyDrive/Omnilex-Agentic-Retrieval-Competition')
else:
    DRIVE_BASE = Path(r'E:\swiss_citation_extraction')

DATA_DIR  = DRIVE_BASE / 'data'
RETRIEVAL = DRIVE_BASE / 'retrieval'
CACHE     = RETRIEVAL / 'beat_v12_val_cc_cache'
CACHE.mkdir(parents=True, exist_ok=True)

# v12 artifacts (auto-detect corpus name)
CORPUS_PARQUET = next((p for p in [RETRIEVAL/'corpus.parquet', RETRIEVAL/'corpus_v2.parquet'] if p.exists()),
                     RETRIEVAL/'corpus.parquet')
BM25_INDEX_PKL = next((p for p in [RETRIEVAL/'bm25_v2_index.pkl', RETRIEVAL/'bm25_index.pkl'] if p.exists()),
                     RETRIEVAL/'bm25_v2_index.pkl')
BM25_IDS_PKL   = next((p for p in [RETRIEVAL/'bm25_v2_ids.pkl', RETRIEVAL/'bm25_ids.pkl'] if p.exists()),
                     RETRIEVAL/'bm25_v2_ids.pkl')
KB_JSONL       = RETRIEVAL / 'laws_knowledge_base.jsonl'
QT_DIR         = RETRIEVAL / 'knowledge_base_optimized_hybrid_retrieval'
ENRICHMENT_JL  = DRIVE_BASE / 'llm_enrichment_output_law_173k' / 'law_llm_descriptors_0000000_all.jsonl'

# Models — match what's currently downloaded on Colab (Qwen3-14B from yesterday)
EXPANDER_MODEL = 'Qwen/Qwen3-14B'
RERANKER_MODEL = 'Qwen/Qwen3-Reranker-8B'

# v12 spine params
TOP_BM25     = 100
RERANK_BS    = 8
MAX_LEN      = 4096

# Channel toggles
ADD_STATUTE_PARSER     = True
ADD_PROCEDURAL_CHANNEL = True
ADD_REF_EXPANSION      = True
APPLY_SPECIFICITY      = True   # drop pool candidates with specificity_score < 0.30
SPECIFICITY_FLOOR      = 0.30

# Count-calibrated selector (STRICT top-K, NO overflow)
COUNT_MIN              = 4
COUNT_MAX              = 22    # was 30 — LLM tended to hit ceiling on hallucinated sequences

# Per-area count bands — general Swiss BGer practice, not val-tuned.
# Used as a sanity clamp on the LLM's expected_citation_count.
AREA_COUNT_BAND = {
    'criminal-procedure':    (14, 22),
    'criminal-substantive':  (13, 22),
    'civil-contract':        ( 8, 14),
    'civil-family':          ( 6, 11),
    'civil-inheritance':     ( 7, 13),
    'civil-property':        ( 9, 15),
    'civil-tort':            ( 8, 13),
    'social-insurance':      (12, 20),
    'public-law':            ( 8, 14),
    'constitutional':        ( 8, 14),
    'debt-enforcement':      ( 6, 13),
    'tax':                   ( 6, 12),
    'banking':               ( 8, 14),
    'environment':           ( 6, 12),
    'international-private': ( 8, 14),
}
DEFAULT_BAND = (8, 18)

def clamp_count_by_area(area, llm_count):
    lo, hi = AREA_COUNT_BAND.get(area, DEFAULT_BAND)
    return max(lo, min(hi, int(llm_count)))

# combined_score blend weights (sum to 1.0)
W_FUSED   = 0.55   # 0.7*minmax(BM25) + 0.3*P_yes_reranker  (v12-style)
W_CHANNEL = 0.15   # multi-channel evidence (count / 4)
W_ROLE    = 0.15   # provision_role ∈ needed_roles
W_SPEC    = 0.10   # specificity_score
W_STATUTE = 0.05   # statute-parser hit bonus

# Cache files
EXPANSIONS_CACHE = CACHE / 'expansions.json'
ENRICHMENT_PQ    = CACHE / 'enrichment_index.parquet'
POOL_CACHE       = CACHE / 'pool.json'
RERANK_CACHE     = CACHE / 'rerank.json'

def log(*a, **k): print(*a, **k, flush=True)
def section(t): log('\n'+'='*72); log(f'  {t}'); log('='*72)

log(f'DRIVE_BASE     : {DRIVE_BASE}')
log(f'CACHE          : {CACHE}')
log(f'CORPUS         : {CORPUS_PARQUET.name}')
log(f'BM25_INDEX     : {BM25_INDEX_PKL.name}')
log(f'BM25_IDS       : {BM25_IDS_PKL.name}')
log(f'EXPANDER       : {EXPANDER_MODEL}')
log(f'RERANKER       : {RERANKER_MODEL}')

# %%
# 3. Utilities
_TOKEN_RE = re.compile(r'[^\w\d]+', re.UNICODE)
_CIT_RE   = re.compile(r'^Art\.\s+(\d+[a-z]?)(?:\s+Abs\.\s+(\d+[a-z]*))?(?:\s+lit\.\s+(\S+))?\s+(\S+)$')

CODE_ALIASES = {
    'CPP':'StPO','CPC':'ZPO','LP':'SchKG','LEF':'SchKG','LTF':'BGG',
    'CC':'ZGB','CO':'OR','CP':'StGB','LAI':'IVG','LPGA':'ATSG','LDIP':'IPRG',
    'DPMin':'JStG','PPMin':'JStPO','PA':'VwVG','LAA':'UVG','LAVS':'AHVG',
    'LAMAL':'KVG','LPP':'BVG','CST':'BV',
}
PROCEDURAL_HEADING_PATTERNS = [
    r'\bbeschwerde\b', r'beschwerdefrist', r'beschwerdeverfahren', r'rechtsmittel',
    r'\bberufung\b', r'\brevision\b', r'verfahrenskosten', r'unentgeltlich',
    r'verteidigung', r'rechtspflegeverfahren', r'gerichtsverfahren',
    r'\bgrundrechte\b', r'rechtliches\s+geh', r'\bzust', r'kammern\b',
]
_PROC_RE = re.compile('|'.join(PROCEDURAL_HEADING_PATTERNS), re.IGNORECASE)

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
        out.add(f'Art. {art} {code}')
    return out

# Official scorer
_WS = re.compile(r'\s+')
def _canon(c): return _WS.sub(' ', str(c).strip())
def _parse_field(v):
    if pd.isna(v): return set()
    return {_canon(p) for p in str(v).split(';') if _canon(p)}
def f1_set(pred, gold):
    if not pred and not gold: return 1.0,1.0,1.0
    if not pred or not gold:  return 0.0,0.0,0.0
    tp = len(pred & gold)
    P,R = tp/len(pred), tp/len(gold)
    F = 2*P*R/(P+R) if (P+R) else 0.0
    return P,R,F

# Tolerant JSON parser (handles unquoted keys, single quotes, trailing commas, Python dicts)
def _tolerant_load(s):
    try: return json.loads(s)
    except json.JSONDecodeError: pass
    fixed = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', s)
    fixed = re.sub(r"(?<![A-Za-z])'([^']*?)'(?![A-Za-z])", r'"\1"', fixed)
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
    fixed = re.sub(r'\bNone\b', 'null', fixed)
    fixed = re.sub(r'\bTrue\b', 'true', fixed)
    fixed = re.sub(r'\bFalse\b','false',fixed)
    try: return json.loads(fixed)
    except json.JSONDecodeError: pass
    try:
        v = ast.literal_eval(s)
        if isinstance(v, dict): return v
    except (ValueError, SyntaxError): pass
    return None

# %%
# 4. Stage 0 — load v12 artifacts
section('STAGE 0: LOAD v12 ARTIFACTS')

log('  loading corpus.parquet ...')
corpus = pd.read_parquet(CORPUS_PARQUET)
log(f'  corpus: {len(corpus):,} rows; sample columns: {list(corpus.columns)[:8]}')

log('  loading BM25 index ...')
with open(BM25_INDEX_PKL,'rb') as f: bm25 = pickle.load(f)
with open(BM25_IDS_PKL,'rb')   as f: bm25_meta = pickle.load(f)
if isinstance(bm25_meta, dict):
    bm25_ids = bm25_meta.get('citation_canon') or bm25_meta.get('ids') or []
    if 'best_k1' in bm25_meta: bm25.k1 = float(bm25_meta['best_k1'])
    if 'best_b'  in bm25_meta: bm25.b  = float(bm25_meta['best_b'])
    log(f'  bm25 tuned: k1={bm25.k1:.3f} b={bm25.b:.3f} target_k={bm25_meta.get("best_f1_k")}')
else:
    bm25_ids = bm25_meta
log(f'  bm25: {len(bm25_ids):,} docs')

# alignment sanity check
_ov = len(set(corpus.get('citation_canon', corpus.get('citation', []))) & set(bm25_ids))
if _ov < 0.95 * len(bm25_ids):
    log(f'  ⚠ corpus↔BM25 overlap only {_ov}/{len(bm25_ids)}')
else:
    log(f'  ✓ corpus↔BM25 alignment: {_ov}/{len(bm25_ids)} ({100*_ov/len(bm25_ids):.1f}%)')

log('  loading laws_knowledge_base.jsonl ...')
kb = {}
with open(KB_JSONL,'r',encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        r = json.loads(line)
        kb[r.get('citation_canon') or r.get('citation') or ''] = r
log(f'  KB: {len(kb):,} entries')

qt_files = sorted(QT_DIR.glob('query_translations*.json'))
query_translations = {}
for p in qt_files:
    query_translations.update(json.loads(p.read_text(encoding='utf-8')))
log(f'  query_translations: {len(query_translations)} entries (from {len(qt_files)} files)')

log('  mining TLF prior from train.csv ...')
train_df = pd.read_csv(DATA_DIR / 'train.csv')
val_df   = pd.read_csv(DATA_DIR / 'val.csv')
laws_de  = pd.read_csv(DATA_DIR / 'laws_de.csv')

tlf = defaultdict(Counter)
token_total = Counter()
for _, row in train_df.iterrows():
    cites = [c.strip() for c in str(row['gold_citations']).split(';') if c.strip()]
    abbrevs = [c.split()[-1].lower() for c in cites if c.startswith('Art.')]
    toks = tokenise(row['query'])
    for t in toks:
        token_total[t] += 1
        for a in abbrevs:
            tlf[t][a] += 1
log(f'  TLF: {len(tlf):,} distinct tokens')

proper_case_map = {c.upper(): c for c in laws_de['citation'].dropna() if c}
log(f'  proper-case map: {len(proper_case_map):,}')

laws_set = set(laws_de['citation'].dropna())
children_index = defaultdict(list)
for c in laws_set:
    p = parse_citation(c)
    if p: children_index[(p['art'], p['code'])].append(c)
log(f'  children_index: {len(children_index):,} parent keys')

LAW_TEXT = {c:(t if isinstance(t,str) else '') for c,t in zip(laws_de['citation'], laws_de['text'])}

# Heading-normalised laws_de for procedural channel
laws_de_clean = laws_de.dropna(subset=['citation']).copy()
laws_de_clean['heading_norm'] = laws_de_clean['title'].fillna('').map(
    lambda t: fold_umlauts(extract_heading(t)))
laws_de_clean['code'] = laws_de_clean['citation'].map(lambda c: (parse_citation(c) or {}).get('code',''))

# %%
# 5. Stage 0b — enrichment parquet
section('STAGE 0b: ENRICHMENT INDEX')

if ENRICHMENT_PQ.exists():
    enrich = pd.read_parquet(ENRICHMENT_PQ)
    log(f'  loaded enrichment cache: {len(enrich):,}')
else:
    if not ENRICHMENT_JL.exists():
        raise FileNotFoundError(f'No enrichment JSONL at {ENRICHMENT_JL}')
    log(f'  building enrichment from {ENRICHMENT_JL.name} ...')
    rows = []
    with open(ENRICHMENT_JL,'r',encoding='utf-8') as f:
        for line in tqdm(f, total=175000, desc='enrichment', unit='line'):
            if not line.strip(): continue
            obj = json.loads(line)
            enr = obj.get('llm_enrichment') or {}
            rows.append({
                'citation':        obj.get('citation') or '',
                'english_summary': enr.get('english_summary') or '',
                'concepts_en':     ' | '.join(enr.get('concepts_en') or []),
                'provision_role':  enr.get('provision_role_llm') or '',
                'specificity':     float(enr.get('specificity_score') or 0.0),
            })
    enrich = pd.DataFrame(rows)
    enrich.to_parquet(ENRICHMENT_PQ, index=False)
    log(f'  saved {len(enrich):,} → {ENRICHMENT_PQ}')

enrich_lookup = {r['citation']: r for _, r in enrich.iterrows()}
role_counts = enrich['provision_role'].value_counts()
log(f'  enrich_lookup: {len(enrich_lookup):,}')
log(f'  provision_role vocab: {[(r,c) for r,c in role_counts.head(12).items()]}')

# %%
# 6. Stage 0.5 — LLM expander (Qwen3-14B) with expected_count
section('STAGE 0.5: LLM EXPANDER (expected_count + correct role vocab)')

EXPANDER_SYSTEM = '''You are a Swiss Federal Court (Bundesgericht) legal-citation expert.

Given an English legal query, output ONE JSON object with EXACTLY these three fields:

{
  "legal_area": ONE of ["criminal-procedure","criminal-substantive","civil-contract",
    "civil-family","civil-inheritance","civil-property","civil-tort","social-insurance",
    "public-law","constitutional","debt-enforcement","tax","banking","environment",
    "international-private"],

  "expected_citation_count": INTEGER between 4 and 22 — the MEDIAN number of Art.-prefixed
    Swiss federal-law citations a competent BGer lawyer cites for a typical query of this
    legal area. Use the calibration table below; do NOT exceed the upper bound.

    | legal_area              | typical count | range |
    |-------------------------|--------------:|------:|
    | criminal-procedure      | 18            | 14-22 |
    | criminal-substantive    | 17            | 13-22 |
    | social-insurance        | 16            | 12-20 |
    | civil-contract          | 11            |  8-14 |
    | civil-family            |  9            |  6-11 |
    | civil-inheritance       | 10            |  7-13 |
    | civil-property          | 12            |  9-15 |
    | civil-tort              | 10            |  8-13 |
    | public-law              | 11            |  8-14 |
    | constitutional          | 11            |  8-14 |
    | debt-enforcement        |  9            |  6-13 |
    | banking                 | 11            |  8-14 |
    | tax                     |  9            |  6-12 |
    | environment             |  9            |  6-12 |
    | international-private   | 11            |  8-14 |

  "needed_roles": LIST of up to 5 strings from
    {"procedure","definition","duty","right_or_entitlement","scope","competence",
    "prohibition","principle","sanction_or_penalty","fees_or_costs"}
}

CRITICAL RULES:
- Output ONLY the JSON object. NO prose. NO code fence. NO <think>.
- DO NOT enumerate Art. references. NO `statute_mentions` field. NO sequential Art. lists.
- For expected_citation_count, pick a value WITHIN the area's range. Do not exceed 22.
- Output must be valid JSON: double-quoted keys, double-quoted strings, no trailing commas.
'''

def parse_expander_output(txt, qid=None):
    default = {'legal_area':'', 'secondary_areas':[], 'statute_mentions':[],
               'needed_roles':[], 'expected_citation_count': 15}
    raw = txt
    txt = re.sub(r'<think>[\s\S]*?</think>', '', txt)
    if '</think>' in txt: txt = txt.split('</think>',1)[1]
    txt = re.sub(r'```(?:json)?\s*', '', txt)
    txt = txt.replace('```', '')
    candidates = [m.group(0) for m in re.finditer(r'\{[\s\S]*?\}', txt)]
    g = re.search(r'\{[\s\S]*\}', txt)
    if g: candidates.append(g.group(0))
    for cand in candidates:
        d = _tolerant_load(cand)
        if isinstance(d, dict) and ('legal_area' in d or 'expected_citation_count' in d):
            merged = default.copy(); merged.update(d)
            try: merged['expected_citation_count'] = int(merged['expected_citation_count'])
            except: merged['expected_citation_count'] = 15
            for f in ('secondary_areas','statute_mentions','needed_roles'):
                v = merged.get(f)
                if isinstance(v, str): merged[f] = [v] if v.strip() else []
                elif not isinstance(v, list): merged[f] = []
            return merged
    if qid:
        log(f'  ⚠ parse fail for {qid}; raw (first 600 chars):')
        log('  ' + raw[:600].replace('\n','\n  '))
    return default

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
    out = {}
    if cache_path.exists():
        out = json.loads(cache_path.read_text(encoding='utf-8'))
        if all(qid in out for qid,_ in queries):
            log(f'  loaded full cache: {len(out)} queries')
            return out
    import torch
    tok, model = load_expander()
    for qid, qtxt in tqdm(queries, desc='expander'):
        if qid in out: continue
        msgs = [{'role':'system','content':EXPANDER_SYSTEM},
                {'role':'user','content':qtxt}]
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors='pt', truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=400, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        txt = tok.decode(gen[0, inp['input_ids'].shape[1]:], skip_special_tokens=True)
        out[qid] = parse_expander_output(txt, qid=qid)
        cache_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    del model, tok
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    return out

VAL_QUERIES = [(r['query_id'], r['query']) for _, r in val_df.iterrows()]
expansions = run_expander(VAL_QUERIES, EXPANSIONS_CACHE)
log('\n  per-query predictions:')
for qid, _ in VAL_QUERIES:
    e = expansions[qid]
    log(f"    {qid}: area={e.get('legal_area',''):<22s} "
        f"E={e.get('expected_citation_count',0):>2}  "
        f"roles={','.join(e.get('needed_roles',[]))[:50]}  "
        f"statute_mentions={len(e.get('statute_mentions',[]))}")

# %%
# 7. Stage 1 — TLF-enhanced BM25 (v12 spine)
section('STAGE 1: TLF-ENHANCED BM25')

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

def query_tokens(qid, qtxt):
    en = tokenise(qtxt)
    de = tokenise(query_translations.get(qid, ''))
    base, seen = [], set()
    for t in en + de:
        if t in seen: continue
        seen.add(t); base.append(t)
    return enhance_tokens_with_tlf(base)

def bm25_top_k(qid, qtxt, k=TOP_BM25):
    toks = query_tokens(qid, qtxt)
    scores = bm25.get_scores(toks)
    order  = np.argsort(scores)[::-1][:k]
    return [(bm25_ids[i], float(scores[i])) for i in order if scores[i] > 0]

# Smoke test on val_001
smp = bm25_top_k(*VAL_QUERIES[0], k=5)
log(f'  sample BM25 top-5 for {VAL_QUERIES[0][0]}: {[c for c,_ in smp]}')

# %%
# 8. Stage 1.5 — channel functions + pool construction
section('STAGE 1.5: ENHANCED POOL CONSTRUCTION')

def channel_statute(qid, qtxt, expansion):
    mentions = parse_statute_mentions(qtxt + ' ' + query_translations.get(qid,''))
    for s in expansion.get('statute_mentions', []) or []:
        mentions.update(parse_statute_mentions(s))
        if s in laws_set: mentions.add(s)
    hits = set()
    for m in mentions:
        if m in laws_set:
            hits.add(m); continue
        p = parse_citation(m)
        if p and not p['abs']:
            for ch in children_index.get((p['art'], p['code']), []):
                hits.add(ch)
    return hits

def channel_procedural(expansion):
    areas = [expansion.get('legal_area','')] + (expansion.get('secondary_areas',[]) or [])
    codes = set()
    for a in areas:
        if a in LEGAL_AREA_CODES: codes.update(LEGAL_AREA_CODES[a])
    if not codes: codes = {'BGG'}
    hits = set()
    for code in codes:
        sub = laws_de_clean[laws_de_clean['code']==code]
        for _, r in sub.iterrows():
            if r['heading_norm'] and _PROC_RE.search(r['heading_norm']):
                hits.add(r['citation'])
    return hits

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

@dataclass
class Candidate:
    citation:        str
    bm25_score:      float = 0.0
    bm25_rank:       int   = 10**9
    channels:        set   = field(default_factory=set)
    reranker_score:  float = 0.0
    specificity:     float = 0.0
    provision_role:  str   = ''

def build_pool(qid, qtxt):
    exp = expansions.get(qid, {})
    pool = {}
    def get(cit):
        if cit not in pool: pool[cit] = Candidate(citation=cit)
        return pool[cit]
    for rank, (c, s) in enumerate(bm25_top_k(qid, qtxt, k=TOP_BM25), 1):
        c_proper = proper_case_map.get(c.upper(), c)
        cd = get(c_proper); cd.channels.add('bm25')
        cd.bm25_score = max(cd.bm25_score, s); cd.bm25_rank = min(cd.bm25_rank, rank)
    if ADD_STATUTE_PARSER:
        for c in channel_statute(qid, qtxt, exp):
            get(c).channels.add('statute_parser')
    if ADD_PROCEDURAL_CHANNEL:
        for c in channel_procedural(exp):
            get(c).channels.add('procedural')
    if ADD_REF_EXPANSION:
        seeds = list(pool.keys())
        for c in channel_ref_expansion(seeds):
            get(c).channels.add('ref_expansion')
    for cit, cd in pool.items():
        r = enrich_lookup.get(cit)
        if r is not None:
            cd.specificity = float(r.get('specificity') or 0.0)
            cd.provision_role = str(r.get('provision_role') or '')
    return pool

pools = {qid: build_pool(qid, qtxt) for qid, qtxt in tqdm(VAL_QUERIES, desc='pool')}
avg_pool = sum(len(p) for p in pools.values()) / len(pools)
log(f'  avg pool size: {avg_pool:.1f}')

# Specificity floor (keep statute-parser hits regardless)
if APPLY_SPECIFICITY:
    n_dropped = 0
    for qid, pool in pools.items():
        for cit in list(pool):
            if 0 < pool[cit].specificity < SPECIFICITY_FLOOR:
                if 'statute_parser' not in pool[cit].channels:
                    pool.pop(cit, None); n_dropped += 1
    log(f'  specificity floor dropped {n_dropped} candidates')

log('\n  Pool-recall vs val gold (law-only):')
for _, row in val_df.iterrows():
    qid  = row['query_id']
    gold = {c.strip() for c in str(row['gold_citations']).split(';')
            if c.strip().startswith('Art.')}
    pset = set(pools[qid].keys())
    R = len(gold & pset) / max(len(gold),1) if gold else 0.0
    miss = sorted(gold - pset)
    tag = '✓' if R==1.0 else f'MISS {len(miss)}'
    log(f'    {qid}: R={R:.3f} pool={len(pset)} gold={len(gold)} {tag}')

# Save pool snapshot
POOL_CACHE.write_text(json.dumps({qid:[{'c':cd.citation,'bm25':cd.bm25_score,'rank':cd.bm25_rank,
                                         'channels':sorted(cd.channels),'spec':cd.specificity,
                                         'role':cd.provision_role}
                                        for cd in pool.values()]
                                   for qid, pool in pools.items()},
                                  indent=2, ensure_ascii=False), encoding='utf-8')

# %%
# 9. Stage 2 — Qwen3-Reranker-8B
section('STAGE 2: QWEN3-RERANKER-8B')

RERANK_INSTRUCTION = ('Given a query about Swiss law, determine whether the provided law '
                      'article is related to or applicable to the legal issue.')

def load_reranker():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    log(f'  loading {RERANKER_MODEL}')
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

def doc_text_for_rerank(cit):
    body = LAW_TEXT.get(cit, '')
    h = ''
    r = laws_de_clean.loc[laws_de_clean['citation']==cit]
    if not r.empty: h = r.iloc[0]['title'] or ''
    return f'{cit}\nTitle: {h}\nText: {body[:500]}'

def run_reranker(pools, qtexts, cache_path):
    if cache_path.exists():
        out = json.loads(cache_path.read_text(encoding='utf-8'))
    else:
        out = {}
    import torch
    tok, model, prefix, suffix, yes_id, no_id = load_reranker()
    for qid, pool in tqdm(pools.items(), desc='rerank queries'):
        cits_todo = [c for c in pool.keys() if c not in out.get(qid, {})]
        if not cits_todo:
            for cit, cd in pool.items():
                cd.reranker_score = float(out.get(qid, {}).get(cit, 0.0))
            continue
        out.setdefault(qid, {})
        q = qtexts[qid]
        pairs = [f'<Instruct>: {RERANK_INSTRUCTION}\n<Query>: {q}\n<Document>: {doc_text_for_rerank(c)}'
                 for c in cits_todo]
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
                logits = model(input_ids=inp, attention_mask=am).logits[:,-1,:]
            yes = logits[:, yes_id]; no_ = logits[:, no_id]
            probs = torch.softmax(torch.stack([no_,yes], dim=-1), dim=-1)[:,1].float().cpu().numpy()
            for c, p in zip(cits_todo[i:i+RERANK_BS], probs):
                out[qid][c] = float(p)
                pool[c].reranker_score = float(p)
        cache_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    del model, tok
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    return out

qtexts = {qid:q for qid,q in VAL_QUERIES}
rerank_scores = run_reranker(pools, qtexts, RERANK_CACHE)
log(f'  reranked: {sum(len(v) for v in rerank_scores.values())} pairs')

# %%
# 10. Stage 3 — combined_score
section('STAGE 3: COMBINED SCORE')

def minmax(d):
    if not d: return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo: return {k: 1.0 for k in d}
    return {k: (v-lo)/(hi-lo) for k,v in d.items()}

combined = {}
for qid, pool in pools.items():
    exp = expansions.get(qid, {})
    needed = set(exp.get('needed_roles', []) or [])
    bm25_dict = {c: cd.bm25_score for c, cd in pool.items()}
    bm25_norm = minmax(bm25_dict)
    sc = {}
    for c, cd in pool.items():
        v12_fused = 0.7*bm25_norm[c] + 0.3*cd.reranker_score
        ch_norm   = min(len(cd.channels), 4) / 4.0
        role_norm = 1.0 if (cd.provision_role and cd.provision_role in needed) else 0.0
        statute_norm = 1.0 if 'statute_parser' in cd.channels else 0.0
        sc[c] = (
            W_FUSED   * v12_fused
          + W_CHANNEL * ch_norm
          + W_ROLE    * role_norm
          + W_SPEC    * cd.specificity
          + W_STATUTE * statute_norm
        )
    combined[qid] = sc

# %%
# 11. Stage 4 — STRICT top-K selection
section('STAGE 4: STRICT TOP-K SELECTION')

preds = {}
for qid, _ in VAL_QUERIES:
    sc = combined[qid]
    if not sc:
        preds[qid] = []; continue
    ordered = sorted(sc.items(), key=lambda x: x[1], reverse=True)
    exp_q = expansions[qid]
    E_raw = exp_q.get('expected_citation_count', 12)
    area  = exp_q.get('legal_area', '')
    # Two clamps: (1) by area band, (2) global [COUNT_MIN, COUNT_MAX].
    E_area = clamp_count_by_area(area, E_raw)
    K      = max(COUNT_MIN, min(COUNT_MAX, E_area))
    selected = [c for c, _ in ordered[:K]]
    preds[qid] = sorted({proper_case_map.get(c.upper(), c) for c in selected})

log('\n  Per-query selection:')
for _, row in val_df.iterrows():
    qid  = row['query_id']
    gold_law = {c.strip() for c in str(row['gold_citations']).split(';')
                if c.strip().startswith('Art.')}
    exp_q = expansions[qid]
    E_raw  = exp_q.get('expected_citation_count', 12)
    area   = exp_q.get('legal_area','')
    E_area = clamp_count_by_area(area, E_raw)
    log(f'    {qid}: gold_law={len(gold_law):>2}  E_llm={E_raw:>2}  area={area or "(?)":<22s}  '
        f'E_clamped={E_area:>2}  selected={len(preds[qid]):>2}')

# %%
# 12. Stage 5 — score
section('STAGE 5: VAL MACRO F1')

def macro(preds, df, law_only=False):
    rows = []
    for _, row in df.iterrows():
        qid = row['query_id']
        gold = _parse_field(row.get('gold_citations',''))
        pred = _parse_field(';'.join(preds.get(qid, [])))
        if law_only:
            gold = {c for c in gold if c.startswith('Art.')}
            pred = {c for c in pred if c.startswith('Art.')}
        P,R,F = f1_set(pred, gold)
        rows.append({'query_id':qid,'P':P,'R':R,'F1':F,
                     'pred':len(pred),'gold':len(gold)})
    out = pd.DataFrame(rows)
    return out, out[['P','R','F1']].mean().to_dict()

per_whole, m_whole = macro(preds, val_df, law_only=False)
log('\nWHOLE-GOLD (laws + court rulings in denominator):')
log(per_whole.to_string(index=False))
log(f"\n  MACRO P={m_whole['P']:.3f}  R={m_whole['R']:.3f}  F1={m_whole['F1']:.3f}")

per_law, m_law = macro(preds, val_df, law_only=True)
log('\nLAW-ONLY (court rulings removed from gold):')
log(per_law.to_string(index=False))
log(f"\n  MACRO P={m_law['P']:.3f}  R={m_law['R']:.3f}  F1={m_law['F1']:.3f}")

# Submission CSV
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
sub = CACHE / f'submission_val_count_calibrated_{ts}.csv'
pd.DataFrame([{'query_id':qid, 'predicted_citations':';'.join(preds.get(qid,[]))}
              for qid in val_df['query_id']]).to_csv(sub, index=False)
log(f'\n  wrote {sub}')

log('\nvs v12 baseline (Hybrid v12, F1=0.585 whole-gold / 0.773 law-only):')
log(f"  whole-gold F1: {m_whole['F1']:.3f}  (Δ {m_whole['F1']-0.585:+.3f})")
log(f"  law-only   F1: {m_law['F1']:.3f}  (Δ {m_law['F1']-0.773:+.3f})")

# %% [markdown]
# # Ablations
#
# All knobs in cell 2:
#
# - `ADD_PROCEDURAL_CHANNEL = False` — disable Channel C
# - `ADD_REF_EXPANSION = False` — disable Channel D
# - `APPLY_SPECIFICITY = False` — disable specificity floor
# - `W_FUSED / W_CHANNEL / W_ROLE / W_SPEC / W_STATUTE` — adjust the combined-score blend
# - `COUNT_MIN / COUNT_MAX` — clamp `expected_count` if LLM gives extreme values
#
# All caches persist on Drive — re-run only the affected cells:
# - Change `EXPANDER_SYSTEM` → delete `expansions.json`, re-run cell 6
# - Change channels / spec floor → re-run cells 8-12
# - Change weights → re-run cells 10-12 (instant)
