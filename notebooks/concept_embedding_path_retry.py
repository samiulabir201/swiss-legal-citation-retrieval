# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Concept-Embedding Path — Retry & Re-Analyze
#
# **Why this notebook exists**: the previous run hit a 38.3 % JSON-truncation rate
# in Cell 4 because `max_tokens=900` was too small for the trilingual extraction.
# Two val queries (incl. val_001 — the 42-gold canonical query) returned empty
# concepts, contaminating the lift numbers.
#
# **What this does**:
# 1. Reuses every cached artifact on Drive (doc embeddings, partial extraction).
# 2. Re-extracts ONLY the 440 broken queries with `max_tokens=2048`.
# 3. Re-encodes query embeddings (because concept strings change).
# 4. Re-runs Cells 8-12 of the original experiment with the fixed data.
#
# **Expected wall-clock**: ~25-30 min on G4 / RTX PRO 6000 Blackwell.

# %% [markdown]
# ## Cell 1 — Setup (mount + install)

# %%
import os, sys, subprocess, json, time, gc
from pathlib import Path

IS_COLAB = "google.colab" in sys.modules
print(f"Colab: {IS_COLAB}")

if IS_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

INSTALL_CMD = [
    "pip", "install", "-q", "-U",
    "vllm>=0.8.5", "transformers>=4.51.0", "pandas==2.2.3", "pyarrow==16.1.0",
    "numpy==1.26.4", "tqdm",
]
if IS_COLAB:
    subprocess.run(INSTALL_CMD, check=True)

# %% [markdown]
# ## Cell 2 — Paths (same as original notebook)

# %%
DRIVE_ROOT = Path("/content/drive/MyDrive/swiss_law")

TRAIN_CSV    = DRIVE_ROOT / "data" / "train.csv"
VAL_CSV      = DRIVE_ROOT / "data" / "val.csv"
LAW_ENRICH   = DRIVE_ROOT / "llm_enrichment_output_law_173k"   / "law_llm_descriptors_0000000_all.jsonl"
COURT_ENRICH = DRIVE_ROOT / "llm_enrichment_output_court_363k" / "court_llm_descriptors_0000000_all.jsonl"

OUT_DIR = DRIVE_ROOT / "research" / "concept_embedding_path"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_QUERY_CONCEPTS = OUT_DIR / "query_concepts.parquet"
CACHE_DOC_EMB_LAW    = OUT_DIR / "doc_concept_emb_law.fp16.npy"
CACHE_DOC_EMB_LAW_M  = OUT_DIR / "doc_concept_emb_law_manifest.parquet"
CACHE_DOC_EMB_COURT  = OUT_DIR / "doc_concept_emb_court.fp16.npy"
CACHE_DOC_EMB_COURT_M= OUT_DIR / "doc_concept_emb_court_manifest.parquet"
CACHE_QRY_EMB        = OUT_DIR / "query_concept_emb.fp16.npy"
CACHE_QRY_EMB_M      = OUT_DIR / "query_concept_emb_manifest.parquet"
REPORT_MD            = OUT_DIR / "report.md"
LIFT_JSON            = OUT_DIR / "lift_table.json"

LLM_MODEL = "Qwen/Qwen3-32B-AWQ"
EMB_MODEL = "Qwen/Qwen3-Embedding-8B"

NEG_PER_GOLD = 5
SEED = 13
COSINE_THRESHOLDS = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
TOP_K_TO_RECORD = 1000

import random
random.seed(SEED)

# Sanity: confirm everything we need is on Drive
print("Cache status:")
for f in [CACHE_QUERY_CONCEPTS, CACHE_DOC_EMB_LAW, CACHE_DOC_EMB_LAW_M,
          CACHE_DOC_EMB_COURT, CACHE_DOC_EMB_COURT_M]:
    print(f"  {'OK' if f.exists() else 'MISSING'}  {f}")

# Drop stale query embeddings — we'll re-encode after fixing concepts
for f in [CACHE_QRY_EMB, CACHE_QRY_EMB_M]:
    if f.exists():
        os.remove(f); print(f"  deleted stale → {f}")

# %% [markdown]
# ## Cell 3 — Reload train/val + enrichment (needed for retry + analysis)

# %%
import pandas as pd

def _load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass
    return rows

def _law_concept_string(rec):
    e = rec.get("llm_enrichment") or {}
    bits = []
    bits += [str(c) for c in (e.get("concepts_en") or [])]
    bits += [str(tp.get("en", "")) for tp in (e.get("terms_de_to_en") or []) if isinstance(tp, dict)]
    if e.get("legal_question"): bits.append(str(e["legal_question"]))
    if e.get("english_summary"): bits.append(str(e["english_summary"])[:300])
    bits += [str(c) for c in (e.get("applicability_conditions") or [])][:5]
    return " | ".join(b for b in bits if b)

def _court_concept_string(rec):
    e = rec.get("llm_enrichment") or {}
    bits = []
    bits += [str(c) for c in (e.get("concepts_en") or [])]
    bits += [str(t) for t in (e.get("terms_original") or [])]
    bits += [str(t) for t in (e.get("fact_pattern_tags") or [])]
    bits += [str(t) for t in (e.get("legal_domain_path") or [])]
    for f_ in ("topic", "subtopic", "micro_topic", "doctrinal_rule", "legal_test", "procedural_context"):
        if e.get(f_): bits.append(str(e[f_]))
    return " | ".join(b for b in bits if b)

print("Loading train + val ...")
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
print(f"  train: {len(train_df)}  val: {len(val_df)}")

print("Loading law enrichment ...")
law_rows = _load_jsonl(LAW_ENRICH)
law_concepts = {r["citation"]: _law_concept_string(r) for r in law_rows if r.get("citation")}
print(f"  law: {len(law_concepts):,}")
del law_rows; gc.collect()

print("Loading court enrichment ...")
court_rows = _load_jsonl(COURT_ENRICH)
court_concepts = {r["citation"]: _court_concept_string(r) for r in court_rows if r.get("citation")}
print(f"  court: {len(court_concepts):,}")
del court_rows; gc.collect()

# %% [markdown]
# ## Cell 4 — Audit cached extraction
#
# Shows you exactly how many queries need re-extraction.

# %%
df_existing = pd.read_parquet(CACHE_QUERY_CONCEPTS)
print(f"Cached extractions: {len(df_existing)}")

needs_retry_mask = df_existing['legal_concepts_en'].apply(lambda v: v is None or len(v) == 0)
needs_retry_qids = df_existing.loc[needs_retry_mask, 'query_id'].tolist()

print(f"  parse_ok=True: {int(df_existing['parse_ok'].sum())}")
print(f"  need retry (empty concepts): {int(needs_retry_mask.sum())}")
print(f"  by split: {df_existing[needs_retry_mask].split.value_counts().to_dict()}")

# %% [markdown]
# ## Cell 5 — Re-extraction with `max_tokens=2048`
#
# Only retries the broken queries. Preserves the 709 working extractions.

# %%
QUERY_EXTRACT_PROMPT = """You are a Swiss legal analyst. Extract structured concepts from the question.

Switzerland is trilingual (DE / FR / IT). The corpus contains decisions in all three.
For each legal concept produced in English, ALSO provide the canonical legal-term
translations in German, French, and Italian, drawn from the actual Swiss legal
vocabulary (the same terms a Swiss lawyer would write in a Bundesgericht /
Tribunal federal / Tribunale federale decision in that language).

Output **strict JSON** with exactly these keys:
{
  "legal_area": "<one of: criminal procedure | criminal substantive | civil obligations | family/inheritance | property | social insurance | tax | administrative | constitutional | international private | other>",
  "legal_concepts_en": ["concept1", "concept2", ...],          // 5-15 English concepts
  "terms_de":  ["Untersuchungshaft", "Kollusionsgefahr", ...], // 5-15 German legal terms
  "terms_fr":  ["detention provisoire", "risque de collusion", ...], // 5-15 French
  "terms_it":  ["detenzione preventiva", "rischio di collusione", ...], // 5-15 Italian. NEVER leave empty.
  "named_statutes": ["Art. 221 Abs. 1 StPO", ...],
  "predicted_codes": ["StPO", "BGG", ...],
  "is_appeal": true|false,
  "is_fundamental_right_issue": true|false
}

All three of terms_de, terms_fr, terms_it MUST be populated with at least 5 items each.

QUESTION:
{QUESTION}

JSON:"""

def parse_json_lenient(s):
    s = s.strip()
    # Strip ```json fence if present
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    a = s.find("{"); b = s.rfind("}")
    if a == -1 or b == -1:
        # Try closing an open JSON by appending '}'
        if a != -1:
            s2 = s[a:] + "}" * 5  # close up to 5 levels
            for n_close in range(5, 0, -1):
                try:
                    return json.loads(s[a:] + "}" * n_close)
                except Exception:
                    continue
        return None
    try: return json.loads(s[a:b+1])
    except Exception:
        cleaned = s[a:b+1].replace("'", '"').replace(",}", "}").replace(",]", "]")
        try: return json.loads(cleaned)
        except Exception: return None

if needs_retry_mask.sum() == 0:
    print("Nothing to retry — every cached extraction already has concepts.")
    df_merged = df_existing
else:
    # Pull the original query text from train_df / val_df
    src = pd.concat([
        train_df[['query_id', 'query']].assign(split='train'),
        val_df[['query_id', 'query']].assign(split='val'),
    ], ignore_index=True)
    retry_src = src[src.query_id.isin(needs_retry_qids)].copy()
    print(f"Retrying {len(retry_src)} queries with max_tokens=2048 ...")

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=LLM_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.55,
        max_model_len=32768,
        enforce_eager=False,
        trust_remote_code=True,
    )
    sp = SamplingParams(
        temperature=0.0, top_p=1.0,
        max_tokens=2048,
        stop=["\n```", "QUESTION:", "</json>"],
    )

    prompts = [QUERY_EXTRACT_PROMPT.replace("{QUESTION}", str(q)) for q in retry_src["query"].tolist()]
    outs = llm.generate(prompts, sp)

    retry_rows = []
    for q_row, out in zip(retry_src.itertuples(), outs):
        text = out.outputs[0].text if out.outputs else ""
        parsed = parse_json_lenient(text) or {}
        retry_rows.append({
            "split": q_row.split,
            "query_id": q_row.query_id,
            "raw_llm": text[:4000],
            "legal_area": parsed.get("legal_area"),
            "legal_concepts_en": parsed.get("legal_concepts_en", []) or [],
            "terms_de": parsed.get("terms_de", []) or [],
            "terms_fr": parsed.get("terms_fr", []) or [],
            "terms_it": parsed.get("terms_it", []) or [],
            "named_statutes": parsed.get("named_statutes", []) or [],
            "predicted_codes": parsed.get("predicted_codes", []) or [],
            "is_appeal": bool(parsed.get("is_appeal", False)),
            "is_fundamental_right_issue": bool(parsed.get("is_fundamental_right_issue", False)),
            "parse_ok": bool(parsed.get("legal_concepts_en")),
        })

    retry_df = pd.DataFrame(retry_rows)
    print(f"  retry parse_ok rate: {retry_df['parse_ok'].mean():.1%}")
    print(f"  still empty after retry: {(~retry_df['parse_ok']).sum()}")

    # Merge: drop old broken rows, append retried
    df_merged = pd.concat(
        [df_existing[~needs_retry_mask], retry_df],
        ignore_index=True,
    )
    df_merged.to_parquet(CACHE_QUERY_CONCEPTS, index=False)
    print(f"Saved → {CACHE_QUERY_CONCEPTS}  total={len(df_merged)}  parse_ok={df_merged['parse_ok'].mean():.1%}")

    del llm; gc.collect()
    import torch; torch.cuda.empty_cache()

query_concepts_df = df_merged

# %% [markdown]
# ## Cell 6 — Spot-check the retried val queries
#
# val_001 should now have real concepts. If it's still empty, the prompt or budget needs another bump.

# %%
for qid in ["val_001", "val_002", "val_003", "val_004", "val_010"]:
    row = query_concepts_df[query_concepts_df.query_id == qid]
    if len(row) == 0: continue
    r = row.iloc[0]
    print(f"--- {qid} (parse_ok={r['parse_ok']}) ---")
    print(f"  legal_area: {r['legal_area']}")
    print(f"  concepts_en[:8]: {list(r['legal_concepts_en'])[:8]}")
    print(f"  terms_de[:6]: {list(r['terms_de'])[:6]}")
    print(f"  terms_fr[:6]: {list(r['terms_fr'])[:6]}")
    print(f"  terms_it[:6]: {list(r['terms_it'])[:6]}")
    print(f"  named_statutes: {list(r['named_statutes'])}")
    print(f"  is_appeal={r['is_appeal']}  is_fund_right={r['is_fundamental_right_issue']}")
    print()

# %% [markdown]
# ## Cell 7 — Load cached doc embeddings + re-encode queries
#
# Doc embeddings: free (cached). Query embeddings: re-encode (concepts changed).

# %%
import numpy as np

print("Loading cached doc embeddings ...")
law_emb = np.load(CACHE_DOC_EMB_LAW)
law_man = pd.read_parquet(CACHE_DOC_EMB_LAW_M)
court_emb = np.load(CACHE_DOC_EMB_COURT)
court_man = pd.read_parquet(CACHE_DOC_EMB_COURT_M)
print(f"  law:   {law_emb.shape} {law_emb.dtype}")
print(f"  court: {court_emb.shape} {court_emb.dtype}")

def encode_qwen3_embed(texts, batch_size=64, max_length=256):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(EMB_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        EMB_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda").eval()
    def last_token_pool(last_hidden, attn):
        left_padded = (attn[:, -1].sum() == attn.shape[0])
        if left_padded:
            return last_hidden[:, -1]
        seqlens = attn.sum(dim=1) - 1
        return last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seqlens]
    out = np.empty((len(texts), 4096), dtype=np.float16)
    from tqdm.auto import tqdm
    for i in tqdm(range(0, len(texts), batch_size), desc="encode_q"):
        chunk = texts[i:i+batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        v = last_token_pool(h, enc["attention_mask"])
        v = torch.nn.functional.normalize(v, p=2, dim=-1)
        out[i:i+v.size(0)] = v.cpu().to(torch.float16).numpy()
    del model, tok; gc.collect()
    torch.cuda.empty_cache()
    return out

def build_query_string(r):
    bits = []
    bits += list(r.get("legal_concepts_en") or [])
    bits += list(r.get("terms_de") or [])
    bits += list(r.get("terms_fr") or [])
    bits += list(r.get("terms_it") or [])
    if r.get("legal_area"): bits.append(str(r["legal_area"]))
    return " | ".join(str(b) for b in bits if b)

query_concepts_df["concept_string"] = query_concepts_df.apply(build_query_string, axis=1)
print("Re-encoding query concept strings (with fixed val_001) ...")
q_texts = query_concepts_df["concept_string"].tolist()
q_emb = encode_qwen3_embed(q_texts)
np.save(CACHE_QRY_EMB, q_emb)
query_concepts_df[["split", "query_id", "concept_string"]].to_parquet(CACHE_QRY_EMB_M, index=False)
print(f"Saved → {CACHE_QRY_EMB}  shape={q_emb.shape}")

# %% [markdown]
# ## Cell 8 — Gold vs control: cosine on the FIXED concept embeddings

# %%
import re

law_cit_to_row   = {c: i for i, c in enumerate(law_man["citation"].tolist())}
court_cit_to_row = {c: i for i, c in enumerate(court_man["citation"].tolist())}

BGE_RE    = re.compile(r"^(?:BGE|ATF|DTF)\s+\d+\s+[IVX]+\s+\d+")
DOCKET_RE = re.compile(r"^[1-9][A-Z]_\d+/\d{4}")
def is_court_cit(c):
    return bool(BGE_RE.match(c) or DOCKET_RE.match(c) or re.match(r"^\d+[A-Z]\.\s*\d", c))

def cosine_for(query_idx, doc_idx, doc_emb):
    return float(np.dot(q_emb[query_idx].astype(np.float32),
                        doc_emb[doc_idx].astype(np.float32)))

qid_to_idx = {qid: i for i, qid in enumerate(query_concepts_df["query_id"].tolist())}
eval_df = pd.concat([train_df.assign(split="train"), val_df.assign(split="val")], ignore_index=True)
all_law_cits   = list(law_cit_to_row.keys())
all_court_cits = list(court_cit_to_row.keys())

records = []
n_skipped_query = 0
n_skipped_no_enr = 0
for _, r in eval_df.iterrows():
    qid = r["query_id"]
    if qid not in qid_to_idx:
        n_skipped_query += 1; continue
    qix = qid_to_idx[qid]
    gold = [c.strip() for c in str(r["gold_citations"]).split(";") if c.strip()] if isinstance(r.get("gold_citations"), str) else []
    gold_set = set(gold)
    for g in gold:
        if is_court_cit(g):
            row_idx = court_cit_to_row.get(g)
            if row_idx is None:
                n_skipped_no_enr += 1
                records.append({"qid": qid, "split": r["split"], "kind": "gold",
                                "family": "court", "citation": g, "cosine": None, "found": False})
                continue
            cos = cosine_for(qix, row_idx, court_emb)
            records.append({"qid": qid, "split": r["split"], "kind": "gold",
                            "family": "court", "citation": g, "cosine": cos, "found": True})
        else:
            row_idx = law_cit_to_row.get(g)
            if row_idx is None:
                n_skipped_no_enr += 1
                records.append({"qid": qid, "split": r["split"], "kind": "gold",
                                "family": "law", "citation": g, "cosine": None, "found": False})
                continue
            cos = cosine_for(qix, row_idx, law_emb)
            records.append({"qid": qid, "split": r["split"], "kind": "gold",
                            "family": "law", "citation": g, "cosine": cos, "found": True})
        is_court_g = is_court_cit(g)
        for _ in range(NEG_PER_GOLD):
            if is_court_g:
                c = random.choice(all_court_cits); tries = 0
                while c in gold_set and tries < 5:
                    c = random.choice(all_court_cits); tries += 1
                cos = cosine_for(qix, court_cit_to_row[c], court_emb)
                records.append({"qid": qid, "split": r["split"], "kind": "ctrl",
                                "family": "court", "citation": c, "cosine": cos, "found": True})
            else:
                c = random.choice(all_law_cits); tries = 0
                while c in gold_set and tries < 5:
                    c = random.choice(all_law_cits); tries += 1
                cos = cosine_for(qix, law_cit_to_row[c], law_emb)
                records.append({"qid": qid, "split": r["split"], "kind": "ctrl",
                                "family": "law", "citation": c, "cosine": cos, "found": True})

eval_records_df = pd.DataFrame(records)
print(f"records: {len(eval_records_df):,}")
print(f"  queries skipped (no extraction): {n_skipped_query}")
print(f"  gold skipped (no enrichment):    {n_skipped_no_enr}")
print(eval_records_df.groupby(["split", "kind", "family"]).size().unstack(fill_value=0))

# %% [markdown]
# ## Cell 9 — Updated lift table

# %%
def lift_at_threshold(df, tau):
    sub = df.dropna(subset=["cosine"])
    gold = sub[sub.kind == "gold"]
    ctrl = sub[sub.kind == "ctrl"]
    g_rate = (gold["cosine"] >= tau).mean() if len(gold) else 0.0
    c_rate = (ctrl["cosine"] >= tau).mean() if len(ctrl) else 0.0
    lift = (g_rate / c_rate) if c_rate > 0 else float("inf")
    return g_rate, c_rate, lift

rows = []
for scope, sub in [
    ("all",   eval_records_df),
    ("law",   eval_records_df[eval_records_df.family == "law"]),
    ("court", eval_records_df[eval_records_df.family == "court"]),
    ("val",   eval_records_df[eval_records_df.split == "val"]),
    ("train", eval_records_df[eval_records_df.split == "train"]),
]:
    print(f"\n=== Cosine threshold sweep — scope: {scope.upper()} ===")
    print(f"{'tau':>6} {'gold%':>8} {'ctrl%':>8} {'lift':>10}")
    for tau in COSINE_THRESHOLDS:
        g, c, l = lift_at_threshold(sub, tau)
        lift_s = f"{l:.1f}x" if l != float('inf') else "inf"
        print(f"{tau:>6.2f} {100*g:>7.1f}% {100*c:>7.1f}% {lift_s:>9}")
        rows.append({"scope": scope, "tau": tau, "gold_rate": g, "ctrl_rate": c, "lift": l})

print("\n=== Cosine percentiles ===")
for label, df in [
    ("all gold",   eval_records_df[eval_records_df.kind == "gold"]),
    ("all ctrl",   eval_records_df[eval_records_df.kind == "ctrl"]),
    ("law gold",   eval_records_df[(eval_records_df.kind == "gold") & (eval_records_df.family == "law")]),
    ("court gold", eval_records_df[(eval_records_df.kind == "gold") & (eval_records_df.family == "court")]),
    ("val gold",   eval_records_df[(eval_records_df.kind == "gold") & (eval_records_df.split == "val")]),
]:
    s = df["cosine"].dropna().to_numpy()
    if len(s) == 0: continue
    p = np.percentile(s, [5, 25, 50, 75, 95])
    print(f"  {label:<11} n={len(s):>6}  p5={p[0]:.3f} p25={p[1]:.3f} p50={p[2]:.3f} p75={p[3]:.3f} p95={p[4]:.3f}")

# %% [markdown]
# ## Cell 10 — Compare to token-overlap baseline + prior buggy run

# %%
TOKEN_BASELINE = {
    "D_concept_law":         {"gold": 0.118, "lift": 11.1},
    "D_concept_law_strong":  {"gold": 0.019, "lift": 63.6},
}
PRIOR_BUGGY_RUN = {
    "@tau=0.45": {"gold": 0.515, "ctrl": 0.047, "lift": 10.9},
    "@tau=0.55": {"gold": 0.309, "ctrl": 0.004, "lift": 75.1},
}

# Pick best operating points from current run
best = None; best_hp = None
for r in rows:
    if r["scope"] != "all": continue
    if r["lift"] >= 10 and (best is None or r["gold_rate"] > best["gold_rate"]):
        best = r
    if r["lift"] >= 50 and (best_hp is None or r["gold_rate"] > best_hp["gold_rate"]):
        best_hp = r

print("=== Bottom-line comparison ===")
print(f"Token D_concept_law:                gold=11.8% lift=11.1x")
print(f"Token D_concept_law_strong:         gold= 1.9% lift=63.6x")
print(f"Embedding cosine (buggy, lift~11):  gold=51.5% lift=10.9x")
print(f"Embedding cosine (buggy, lift~75):  gold=30.9% lift=75.1x")
if best:
    print(f"Embedding cosine (FIXED, lift>=10): tau={best['tau']:.2f}  gold={100*best['gold_rate']:.1f}%  ctrl={100*best['ctrl_rate']:.1f}%  lift={best['lift']:.1f}x")
if best_hp:
    print(f"Embedding cosine (FIXED, lift>=50): tau={best_hp['tau']:.2f}  gold={100*best_hp['gold_rate']:.1f}%  ctrl={100*best_hp['ctrl_rate']:.1f}%  lift={best_hp['lift']:.1f}x")

# %% [markdown]
# ## Cell 11 — Save report + JSON

# %%
import json as _json

eval_records_df.to_parquet(OUT_DIR / "eval_records.parquet", index=False)

with open(LIFT_JSON, "w", encoding="utf-8") as f:
    _json.dump({
        "config": {
            "neg_per_gold": NEG_PER_GOLD,
            "thresholds": COSINE_THRESHOLDS,
            "emb_model": EMB_MODEL, "llm_model": LLM_MODEL,
            "n_train_queries": int(len(train_df)),
            "n_val_queries": int(len(val_df)),
            "extraction_parse_ok_rate": float(query_concepts_df["parse_ok"].mean()),
            "fixed_run": True,
        },
        "rows": rows,
        "token_baseline": TOKEN_BASELINE,
        "prior_buggy_run": PRIOR_BUGGY_RUN,
    }, f, indent=2)

def fmt_section(label, key):
    out = [f"### {label}", "", "| tau | gold% | ctrl% | lift |", "|---|---|---|---|"]
    for r in rows:
        if r["scope"] != key: continue
        lift_s = f"{r['lift']:.1f}x" if r['lift'] != float('inf') else "inf"
        out.append(f"| {r['tau']:.2f} | {100*r['gold_rate']:.1f}% | {100*r['ctrl_rate']:.1f}% | {lift_s} |")
    return "\n".join(out)

md = [
    "# Concept-Embedding Path — FIXED run", "",
    f"**LLM**: {LLM_MODEL} | **Embedding**: {EMB_MODEL}",
    f"**Extraction parse_ok**: {query_concepts_df['parse_ok'].mean():.1%}",
    f"**Train queries**: {len(train_df)} | **Val queries**: {len(val_df)} | **Neg/gold**: {NEG_PER_GOLD}",
    "",
    "## Baselines for comparison", "",
    "| Path | Gold% | Lift |", "|---|---|---|",
    "| Token D_concept_law (>=2 overlap) | 11.8% | 11.1x |",
    "| Token D_concept_law_strong (>=4)  |  1.9% | 63.6x |",
    "| Embedding cosine (buggy, tau=0.45)| 51.5% | 10.9x |",
    "| Embedding cosine (buggy, tau=0.55)| 30.9% | 75.1x |",
    "",
    "## Fixed embedding-cosine path",
    "", fmt_section("ALL train+val", "all"),
    "", fmt_section("LAW only", "law"),
    "", fmt_section("COURT only", "court"),
    "", fmt_section("VAL only", "val"),
    "", fmt_section("TRAIN only", "train"),
    "",
]
if best:
    md.append(f"**Best lift>=10 operating point**: tau={best['tau']:.2f}, gold={100*best['gold_rate']:.1f}%, lift={best['lift']:.1f}x.")
if best_hp:
    md.append(f"**Best lift>=50 operating point**: tau={best_hp['tau']:.2f}, gold={100*best_hp['gold_rate']:.1f}%, lift={best_hp['lift']:.1f}x.")
REPORT_MD.write_text("\n".join(md), encoding="utf-8")
print(f"Saved → {REPORT_MD}")
print(f"        {LIFT_JSON}")

# %% [markdown]
# ## Cell 12 — Updated R@K over full corpus
#
# Same as before but now with the fixed val_001 etc. Expect the VAL recall to improve.

# %%
import torch
from tqdm.auto import tqdm

print("GPU brute-force cosine: each query × full law + court corpus ...")
device = "cuda"
law_t   = torch.from_numpy(law_emb).to(device, dtype=torch.float16)
court_t = torch.from_numpy(court_emb).to(device, dtype=torch.float16)
q_t_all = torch.from_numpy(q_emb).to(device, dtype=torch.float16)
print(f"  law  matrix: {tuple(law_t.shape)}  ({law_t.element_size() * law_t.nelement() / 1e9:.2f} GB)")
print(f"  court matrix: {tuple(court_t.shape)} ({court_t.element_size() * court_t.nelement() / 1e9:.2f} GB)")

law_cit_arr   = law_man["citation"].to_numpy()
court_cit_arr = court_man["citation"].to_numpy()

all_df = pd.concat([train_df.assign(split="train"), val_df.assign(split="val")], ignore_index=True)
qid_to_gold = {r["query_id"]: [g.strip() for g in str(r.get("gold_citations") or "").split(";") if g.strip()]
               for _, r in all_df.iterrows()}

TOPK_KEEP = 1000
RANK_BUCKETS = (1, 5, 10, 25, 50, 100, 200, 500, 1000)

def topk_per_query(doc_t, q_t):
    N_q = q_t.shape[0]
    N_d = doc_t.shape[0]
    top_idx_all = np.empty((N_q, TOPK_KEEP), dtype=np.int32)
    top_sim_all = np.empty((N_q, TOPK_KEEP), dtype=np.float16)
    BATCH = 16
    for i in tqdm(range(0, N_q, BATCH), desc="  cosine"):
        qb = q_t[i:i+BATCH]
        sims = qb @ doc_t.T
        vals, idx = torch.topk(sims, k=min(TOPK_KEEP, N_d), dim=1)
        top_idx_all[i:i+vals.size(0)] = idx.cpu().numpy()
        top_sim_all[i:i+vals.size(0)] = vals.cpu().numpy()
        del sims, vals, idx
    return top_idx_all, top_sim_all

print("\n--- LAW corpus brute force ---")
law_topk_idx, _ = topk_per_query(law_t, q_t_all)
print("\n--- COURT corpus brute force ---")
court_topk_idx, _ = topk_per_query(court_t, q_t_all)

def build_rank_map(idx_row, citations_arr):
    return {citations_arr[int(idx)]: rk for rk, idx in enumerate(idx_row)}

gold_ranks = []
for q_row in range(len(query_concepts_df)):
    qid = query_concepts_df.iloc[q_row]["query_id"]
    split = query_concepts_df.iloc[q_row]["split"]
    if qid not in qid_to_gold: continue
    law_rmap   = build_rank_map(law_topk_idx[q_row], law_cit_arr)
    court_rmap = build_rank_map(court_topk_idx[q_row], court_cit_arr)
    for g in qid_to_gold[qid]:
        if is_court_cit(g):
            rk = court_rmap.get(g, -1); fam = "court"
            in_corpus = g in court_cit_to_row
        else:
            rk = law_rmap.get(g, -1); fam = "law"
            in_corpus = g in law_cit_to_row
        gold_ranks.append({"qid": qid, "split": split, "family": fam, "gold": g,
                           "rank": rk, "in_corpus": in_corpus})

rank_df = pd.DataFrame(gold_ranks)
rank_df.to_parquet(OUT_DIR / "gold_rank_full_corpus.parquet", index=False)

print("\n=== R@K over full enriched corpus (FIXED run) ===")
rk_rows = []
def report_rk(df, label):
    df2 = df[df.in_corpus]; n = len(df2)
    if n == 0: print(f"  {label}: n=0"); return
    print(f"  {label}  (n={n})")
    for k in RANK_BUCKETS:
        hit = ((df2["rank"] >= 0) & (df2["rank"] < k)).mean()
        rk_rows.append({"label": label, "k": k, "recall": float(hit), "n": int(n)})
        print(f"    R@{k:>4}: {hit:.1%}")

report_rk(rank_df, "ALL")
report_rk(rank_df[rank_df.family == "law"], "LAW")
report_rk(rank_df[rank_df.family == "court"], "COURT")
report_rk(rank_df[rank_df.split == "val"], "VAL")
report_rk(rank_df[rank_df.split == "train"], "TRAIN")

with open(OUT_DIR / "rank_table.json", "w", encoding="utf-8") as f:
    _json.dump(rk_rows, f, indent=2)
print(f"\nSaved → {OUT_DIR / 'rank_table.json'}")

del law_t, court_t, q_t_all
gc.collect(); torch.cuda.empty_cache()
