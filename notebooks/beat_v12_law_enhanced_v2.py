# %% [markdown]
# # Beat v12 — v2: VAL-ONLY, count-calibrated selector
#
# Successor to `beat_v12_law_enhanced.py`. Three differences from v1:
#
# 1. **Val-only.** No test predictions, no test in the expander pool.
# 2. **Expected count predicted by LLM.** New `expected_citation_count` field — the
#    LLM estimates how many articles a Swiss BGer expert would cite for the query,
#    based on legal area, query length, sub-issues, and explicit statute mentions.
# 3. **Count-calibrated selector replaces v12's zone-split.** Final predictions =
#    top-`expected_count` per query, ranked by `combined_score` (multi-signal blend).
#    Optional: drop any candidate the v1 judge explicitly said NO to.
#
# **Reuses all v1 caches** — the slow reranker stage from yesterday is NOT re-run.
# Total runtime ~5 min: new expander (3 min) + selector (instant) + scoring (instant).

# %%
# 1. Setup
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
# 2. Imports + config (match v1 paths so caches line up)
import os, re, json, gc, time, pickle, datetime, math
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

if IN_COLAB:
    DRIVE_BASE = Path('/content/drive/MyDrive/Omnilex-Agentic-Retrieval-Competition')
else:
    DRIVE_BASE = Path(r'E:\swiss_citation_extraction')

DATA_DIR  = DRIVE_BASE / 'data'
RETRIEVAL = DRIVE_BASE / 'retrieval'
V1_CACHE  = RETRIEVAL / 'beat_v12_law_cache'
V2_CACHE  = RETRIEVAL / 'beat_v12_law_cache_v2'
V2_CACHE.mkdir(parents=True, exist_ok=True)

# v1 cache files (REUSED)
V1_POOL          = V1_CACHE / 'enhanced_pool.json'
V1_RERANK_CACHE  = next(V1_CACHE.glob('rerank_*'), None)   # found by glob
V1_JUDGE_CACHE   = next(V1_CACHE.glob('judge_*'), None)
V1_ENRICH_PQ     = V1_CACHE / 'enrichment_index.parquet'

# v2 cache files
V2_EXPANSIONS    = V2_CACHE / 'llm_expansions_v2.json'

# Models
EXPANDER_MODEL = 'Qwen/Qwen3-14B'   # match v1 (user upgraded to 14B)

# Hyperparameters
COMBINED_W_FUSED   = 0.55
COMBINED_W_CHANNEL = 0.15
COMBINED_W_ROLE    = 0.15
COMBINED_W_SPEC    = 0.10
COMBINED_W_STATUTE = 0.05   # statute-parser bonus
USE_JUDGE_FILTER   = True    # drop candidates the v1 judge said NO to
COUNT_OVERFLOW_PCT = 0.0     # STRICT top-K (no overflow). v1 set 0.10 and exploded.
COUNT_MIN          = 4       # don't predict less than this even if LLM says so
COUNT_MAX          = 30      # don't predict more than this even if LLM says so

def log(*a, **k): print(*a, **k, flush=True)
def section(t): log('\n'+'='*72); log(f'  {t}'); log('='*72)

log(f'DRIVE_BASE     : {DRIVE_BASE}')
log(f'V1_POOL        : {V1_POOL}                     exists={V1_POOL.exists()}')
log(f'V1_RERANK_CACHE: {V1_RERANK_CACHE}             exists={V1_RERANK_CACHE is not None and V1_RERANK_CACHE.exists()}')
log(f'V1_JUDGE_CACHE : {V1_JUDGE_CACHE}              exists={V1_JUDGE_CACHE is not None and V1_JUDGE_CACHE.exists()}')
log(f'V1_ENRICH_PQ   : {V1_ENRICH_PQ}                exists={V1_ENRICH_PQ.exists()}')

# %%
# 3. Load val + corpus + enrichment + caches
section('LOAD VAL + CORPUS + CACHES')

val_df = pd.read_csv(DATA_DIR / 'val.csv')
laws_de = pd.read_csv(DATA_DIR / 'laws_de.csv')
log(f'  val: {len(val_df)} queries')
log(f'  laws_de: {len(laws_de):,} rows')

# Proper-case map (Kaggle canonical form: StPO not STPO)
proper_case_map = {c.upper(): c for c in laws_de['citation'].dropna() if c}
log(f'  proper-case map: {len(proper_case_map):,}')

# Enrichment: try v1 cache → v2 cache → build from source JSONL
ENRICHMENT_JL = DRIVE_BASE / 'llm_enrichment_output_law_173k' / 'law_llm_descriptors_0000000_all.jsonl'
V2_ENRICH_PQ  = V2_CACHE / 'enrichment_index.parquet'

enrich = None
for candidate in [V1_ENRICH_PQ, V2_ENRICH_PQ]:
    if candidate.exists():
        enrich = pd.read_parquet(candidate)
        log(f'  loaded enrichment from {candidate.name} ({len(enrich):,} rows)')
        break
if enrich is None:
    if not ENRICHMENT_JL.exists():
        raise FileNotFoundError(
            f'Enrichment JSONL not found at {ENRICHMENT_JL}. '
            f'Upload the 254 MB law_llm_descriptors_0000000_all.jsonl to that path.')
    log(f'  building enrichment parquet from {ENRICHMENT_JL.name} (one-time, ~10-30s) ...')
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
    enrich.to_parquet(V2_ENRICH_PQ, index=False)
    log(f'  built + saved {len(enrich):,} rows -> {V2_ENRICH_PQ}')

enrich_lookup = {r['citation']: r for _, r in enrich.iterrows()}
log(f'  enrich_lookup: {len(enrich_lookup):,}')

# Show actual provision_role vocabulary (we'll use this in the new prompt)
role_counts = enrich['provision_role'].value_counts()
role_vocab = [r for r in role_counts.index if r and r != 'other'][:12]
log(f'  provision_role vocab to use in expander: {role_vocab}')

# v1 pool snapshot (citation -> {bm25, rank, channels, spec, role})
if not V1_POOL.exists():
    raise FileNotFoundError(
        f'v1 pool snapshot not found at {V1_POOL}. '
        f'Run v1 (beat_v12_law_enhanced) through Stage 1.5 before running v2.')
pool_snapshot = json.loads(V1_POOL.read_text(encoding='utf-8'))
pool_snapshot = {qid: pool_snapshot[qid] for qid in val_df['query_id'] if qid in pool_snapshot}
log(f'  pool snapshot: {len(pool_snapshot)} val queries')

# v1 reranker scores  (qid -> {cit: p_yes})
if V1_RERANK_CACHE is None or not V1_RERANK_CACHE.exists():
    raise FileNotFoundError(
        f'v1 reranker cache not found in {V1_CACHE}. '
        f'Run v1 through Stage 2 (Qwen3-Reranker) before running v2.')
rerank_scores = json.loads(V1_RERANK_CACHE.read_text(encoding='utf-8'))
log(f'  reranker cache: {V1_RERANK_CACHE.name} - {len(rerank_scores)} queries '
    f'({sum(len(v) for v in rerank_scores.values())} pairs)')

# v1 judge verdicts (optional filter)
judge_verdicts = {}
if V1_JUDGE_CACHE and V1_JUDGE_CACHE.exists():
    judge_verdicts = json.loads(V1_JUDGE_CACHE.read_text(encoding='utf-8'))
    log(f'  judge cache: {V1_JUDGE_CACHE.name} - {len(judge_verdicts)} queries, '
        f'{sum(len(v) for v in judge_verdicts.values())} verdicts')
else:
    log('  no judge cache (USE_JUDGE_FILTER will be a no-op)')

# %%
# 4. NEW expander — predicts expected_citation_count + correct-vocabulary needed_roles
section('NEW LLM EXPANDER (expected_count + correct role vocab)')

EXPANDER_SYSTEM_V2 = '''You are a Swiss Federal Court (Bundesgericht) legal-citation expert.

Given an English legal scenario, output ONE JSON object with these exact fields:

{
  "legal_area": one of ["criminal-procedure","criminal-substantive","civil-contract",
    "civil-family","civil-inheritance","civil-property","civil-tort","social-insurance",
    "public-law","constitutional","debt-enforcement","tax","banking","environment",
    "international-private"],
  "secondary_areas": [optional list from the same vocabulary, empty if none],
  "statute_mentions": [list of explicit statute references in the query, normalised to
    "Art. N Abs. M CODE" with the Swiss-German abbreviation. CODE in: StPO, StGB, ZGB,
    OR, IVG, ATSG, BGG, BV, StBOG, ZPO, SchKG, IPRG, UVG, KVG, AHVG, BVG, VwVG, USG,
    RPG, JStG, JStPO, DBG, MWSTG, MStG, MStP, AsylG, SVG.
    Convert FR/IT: LAI→IVG, CPP→StPO, CC→ZGB, CO→OR, LPGA→ATSG, LTF→BGG, CP→StGB,
    CPC→ZPO, LP→SchKG, LDIP→IPRG.],
  "needed_roles": [subset of {"procedure","definition","duty","right_or_entitlement",
    "scope","competence","prohibition","principle","sanction_or_penalty","fees_or_costs"}
    — which provision roles answer this query],
  "expected_citation_count": INTEGER — your estimate of how many Art.-prefixed federal-law
    citations a Swiss BGer expert lawyer would cite to resolve this scenario.

    Calibration heuristics (general Swiss BGer practice, NOT specific to any query set):
    - A typical Federal Court (BGer) appeal cites 12-25 articles total
      (substantive rule + appeal apparatus + cost rules + jurisdiction).
    - Simple standalone substantive questions cite 6-10 articles.
    - Social-insurance / disability cases cite 12-20 articles.
    - Constitutional / right-to-be-heard questions cite 8-15 articles.
    - Civil inheritance with capacity question: 8-12 articles.
    - Multi-issue queries with explicit statute mentions: scale up by 1-2 per mention.
    - Output a single integer between 4 and 30.
}

Rules:
- Output ONLY the JSON object. No prose. No <think>. No code fence.
- legal_area: pick the single most-specific area.
- needed_roles: for a BGer appeal, typical sets are {procedure, competence, fees_or_costs}
  plus 1-2 of {duty, right_or_entitlement, prohibition, principle} for the substantive part.
- expected_citation_count: be precise. Don't default to a round number.
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

def _tolerant_load(s):
    """Try strict json.loads, then progressively tolerant transforms."""
    # 1. strict
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # 2. quote unquoted keys: {foo: -> {"foo":   (and after commas)
    fixed = re.sub(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', s)
    # 3. single → double quotes for string values
    fixed = re.sub(r"(?<![A-Za-z])'([^']*?)'(?![A-Za-z])", r'"\1"', fixed)
    # 4. trailing commas before } or ]
    fixed = re.sub(r',(\s*[}\]])', r'\1', fixed)
    # 5. Python None/True/False → JSON null/true/false
    fixed = re.sub(r'\bNone\b', 'null', fixed)
    fixed = re.sub(r'\bTrue\b', 'true', fixed)
    fixed = re.sub(r'\bFalse\b','false',fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    # 6. ast literal_eval (handles Python dicts directly)
    try:
        import ast
        v = ast.literal_eval(s)
        if isinstance(v, dict):
            return v
    except (ValueError, SyntaxError):
        pass
    return None

def parse_expander_output(txt, qid=None):
    default = {'legal_area':'', 'secondary_areas':[], 'statute_mentions':[],
               'needed_roles':[], 'expected_citation_count': 15}
    raw = txt
    # Strip <think>...</think> blocks (Qwen3 reasoning) entirely
    txt = re.sub(r'<think>[\s\S]*?</think>', '', txt)
    if '</think>' in txt:
        txt = txt.split('</think>', 1)[1]
    # Strip markdown code fences
    txt = re.sub(r'```(?:json)?\s*', '', txt)
    txt = txt.replace('```', '')

    # Try non-greedy candidates first (each {...} block independently), then greedy.
    candidates = [m.group(0) for m in re.finditer(r'\{[\s\S]*?\}', txt)]
    m_greedy = re.search(r'\{[\s\S]*\}', txt)
    if m_greedy: candidates.append(m_greedy.group(0))

    for cand in candidates:
        d = _tolerant_load(cand)
        if isinstance(d, dict) and ('legal_area' in d or 'expected_citation_count' in d):
            merged = default.copy()
            merged.update(d)
            try: merged['expected_citation_count'] = int(merged['expected_citation_count'])
            except Exception: merged['expected_citation_count'] = 15
            # coerce list fields if the LLM returned a string
            for f in ('secondary_areas','statute_mentions','needed_roles'):
                v = merged.get(f)
                if isinstance(v, str): merged[f] = [v] if v.strip() else []
                elif not isinstance(v, list): merged[f] = []
            return merged

    # All strategies failed — dump raw for diagnostic
    if qid:
        log(f'  ⚠ WARN parse fail for {qid}; raw output (first 800 chars):')
        log('  ' + raw[:800].replace('\n', '\n  '))
    return default

def run_expander_v2(queries, cache_path):
    out = {}
    if cache_path.exists():
        out = json.loads(cache_path.read_text(encoding='utf-8'))
        if all(qid in out for qid,_ in queries):
            log(f'  loaded full expander cache: {len(out)} queries')
            return out
    import torch
    tok, model = load_expander()
    for qid, qtxt in tqdm(queries, desc='expander-v2'):
        if qid in out: continue
        msgs = [{'role':'system','content':EXPANDER_SYSTEM_V2},
                {'role':'user','content':qtxt}]
        # Disable Qwen3's thinking mode so output is the JSON directly (no <think> preamble).
        try:
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        except TypeError:
            # Older transformers / tokenizers without the kwarg
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=600, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        txt = tok.decode(gen[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        out[qid] = parse_expander_output(txt, qid=qid)
        cache_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    del model, tok
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception: pass
    return out

VAL_QUERIES = [(r['query_id'], r['query']) for _, r in val_df.iterrows()]
expansions_v2 = run_expander_v2(VAL_QUERIES, V2_EXPANSIONS)
log('\n  per-query predictions (legal_area · expected_count · roles):')
for qid, _ in VAL_QUERIES:
    e = expansions_v2[qid]
    log(f"    {qid}: area={e.get('legal_area',''):<24s}  "
        f"E={e.get('expected_citation_count',0):>2}  "
        f"roles={','.join(e.get('needed_roles',[]) or [])[:60]}")

# %%
# 5. Compute combined_score per (qid, candidate)
section('COMBINED SCORE')

def normalize_dict_minmax(d):
    if not d: return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    if hi <= lo: return {k: 1.0 for k in d}
    return {k: (v-lo)/(hi-lo) for k,v in d.items()}

def compute_combined_scores(pool_snapshot, rerank_scores, expansions, judge_verdicts):
    out = {}
    for qid, candidates in pool_snapshot.items():
        exp = expansions.get(qid, {})
        needed = set(exp.get('needed_roles', []) or [])

        # 1) BM25 score (norm)
        bm25_dict = {c['c']: float(c.get('bm25', 0.0)) for c in candidates}
        bm25_norm = normalize_dict_minmax(bm25_dict)

        # 2) Reranker P(yes)
        rerank_dict = rerank_scores.get(qid, {})
        # already 0-1 from softmax
        rerank_norm = {c: float(rerank_dict.get(c, 0.0)) for c in bm25_dict}

        # 3) Channel-hit count (normalised by max=4 channels: bm25, statute, procedural, ref)
        channel_norm = {c['c']: min(len(c.get('channels',[])), 4) / 4.0
                        for c in candidates}

        # 4) Role match (1.0 if provision_role ∈ needed_roles)
        role_norm = {}
        for c in candidates:
            r = c.get('role','') or ''
            role_norm[c['c']] = 1.0 if (r and r in needed) else 0.0

        # 5) Specificity floor (already 0-1)
        spec_norm = {c['c']: float(c.get('spec', 0.0) or 0.0) for c in candidates}

        # 6) Statute-parser bonus
        statute_norm = {c['c']: 1.0 if 'statute_parser' in (c.get('channels') or [])
                        else 0.0 for c in candidates}

        # Blend
        fused = {}
        for c in bm25_dict:
            v12_fused = 0.7*bm25_norm[c] + 0.3*rerank_norm[c]
            combined = (
                COMBINED_W_FUSED   * v12_fused
              + COMBINED_W_CHANNEL * channel_norm[c]
              + COMBINED_W_ROLE    * role_norm[c]
              + COMBINED_W_SPEC    * spec_norm[c]
              + COMBINED_W_STATUTE * statute_norm[c]
            )
            fused[c] = combined
        out[qid] = fused
    return out

combined = compute_combined_scores(pool_snapshot, rerank_scores, expansions_v2, judge_verdicts)
log(f'  computed combined_score for {len(combined)} queries')
# Show top-10 for val_001
qid0 = VAL_QUERIES[0][0]
top10 = sorted(combined[qid0].items(), key=lambda x: x[1], reverse=True)[:10]
log(f'\n  val_001 top-10 by combined_score:')
for c, s in top10:
    log(f'    {s:.3f}  {c}')

# %%
# 6. Count-calibrated selection per query
section('COUNT-CALIBRATED SELECTION')

def select_predictions(combined_q, expected_count, judge_verdicts_q=None):
    # Sort candidates by combined_score desc
    ordered = sorted(combined_q.items(), key=lambda x: x[1], reverse=True)
    if not ordered:
        return []
    K = max(COUNT_MIN, min(COUNT_MAX, int(expected_count)))
    take = ordered[:K]
    # Overflow (default off): include any candidate within COUNT_OVERFLOW_PCT of K-th score
    if COUNT_OVERFLOW_PCT > 0 and K < len(ordered):
        kth_score = ordered[K-1][1]
        threshold = kth_score * (1 - COUNT_OVERFLOW_PCT)
        for c, s in ordered[K:]:
            if s >= threshold:
                take.append((c, s))
            else:
                break
    selected = [c for c,_ in take]
    # Optional: drop any explicit judge=NO
    if USE_JUDGE_FILTER and judge_verdicts_q:
        selected = [c for c in selected if judge_verdicts_q.get(c) != 'NO']
    return selected

preds = {}
for qid, _ in VAL_QUERIES:
    E = expansions_v2[qid].get('expected_citation_count', 15)
    jvq = judge_verdicts.get(qid, {})
    sel = select_predictions(combined[qid], E, jvq)
    # Apply proper case
    sel = [proper_case_map.get(c.upper(), c) for c in sel]
    preds[qid] = sorted(sel)

for qid, _ in VAL_QUERIES:
    E = expansions_v2[qid].get('expected_citation_count', 15)
    log(f'    {qid}: expected={E:>2}  selected={len(preds[qid]):>2}')

# %%
# 7. Score on val
section('VAL SCORING')

_WS = re.compile(r'\s+')
def _canon(c): return _WS.sub(' ', str(c).strip())
def _parse_field(v):
    if pd.isna(v): return set()
    if not isinstance(v,str): v = str(v)
    return {_canon(p) for p in v.split(';') if _canon(p)}
def f1_set(pred, gold):
    if not pred and not gold: return 1.0,1.0,1.0
    if not pred or not gold:  return 0.0,0.0,0.0
    tp = len(pred & gold)
    P = tp/len(pred); R = tp/len(gold)
    F = 2*P*R/(P+R) if (P+R) else 0.0
    return P,R,F

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
    m = out[['P','R','F1']].mean().to_dict()
    return out, m

per_whole, macro_whole = macro(preds, val_df, law_only=False)
section('VAL — WHOLE-GOLD (laws + court rulings in denominator)')
log(per_whole.to_string(index=False))
log(f"\n  MACRO P={macro_whole['P']:.3f}  R={macro_whole['R']:.3f}  F1={macro_whole['F1']:.3f}")

per_law, macro_law = macro(preds, val_df, law_only=True)
section('VAL — LAW-ONLY (court rulings removed from gold)')
log(per_law.to_string(index=False))
log(f"\n  MACRO P={macro_law['P']:.3f}  R={macro_law['R']:.3f}  F1={macro_law['F1']:.3f}")

# %%
# 8. Submission CSV + comparison vs v12 baseline
section('SUBMISSION CSV + COMPARISON')

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
sub_path = V2_CACHE / f'submission_val_beat_v12_v2_{ts}.csv'
pd.DataFrame([{'query_id':qid, 'predicted_citations':';'.join(preds.get(qid,[]))}
              for qid in val_df['query_id']]).to_csv(sub_path, index=False)
log(f'  wrote {sub_path}')

log('\n  vs v12 baseline:')
log('  Metric              v12      v1       v2 (this run)')
log(f"  whole-gold F1       0.585    0.533    {macro_whole['F1']:.3f}")
log(f"  whole-gold P        0.845    0.790    {macro_whole['P']:.3f}")
log(f"  whole-gold R        0.464    0.429    {macro_whole['R']:.3f}")
log(f"  law-only   F1       0.773    0.699    {macro_law['F1']:.3f}")

# %% [markdown]
# # Ablations (rerun cells 5-7 only)
#
# All these are top-of-file constants; edit and re-run cells 5, 6, 7:
#
# - `USE_JUDGE_FILTER = False` — disable the v1 judge=NO filter
# - `COUNT_OVERFLOW_PCT = 0.0` — strict top-K (no overflow)
# - `COUNT_OVERFLOW_PCT = 0.20` — generous overflow (more recall, less precision)
# - `COMBINED_W_FUSED = 1.0, others=0` — pure v12-fused ranking (debug)
#
# # If `expected_citation_count` looks systematically off
#
# Read the per-query log line in cell 4 output. If the LLM is way off (e.g. predicts 8
# when gold=22 consistently), the prompt's calibration heuristics need adjusting.
# Look at cell 6's print: `{qid}: expected=E  selected=N` — if E is too small across all
# queries, lift the heuristic floors in `EXPANDER_SYSTEM_V2`. If E is too large, lower
# the BGer-appeal range.
