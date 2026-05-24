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
# # Concept-Embedding Path Experiment
#
# **Goal**: Does cosine-similarity between *concept embeddings* (query vs. doc enrichment) lift the D_concept path from 11.8 % gold-fire / 1.1 % control-fire (token-overlap, lift 11×) to something stronger — strong enough to cover the 65 % of train gold that fires zero paths under the regex/token framework?
#
# **Prior empirical pass** (`research/stage_b_diagnostics_and_ceilings/path_empirical_pass.md`):
# - 1,139 train queries, 4,659 gold (4,602 law / 57 court), 23,295 matched negatives.
# - 64.8 % of gold fires zero paths.
# - Token-overlap D_concept_law: gold 11.8 %, ctrl 1.1 %, lift 11×.
#
# **Hypothesis**: Replacing the token-overlap test with cosine on Qwen3-Embedding-8B vectors over `concepts_en` (+ `terms`, +`domain_path`) will raise D's gold rate above 50 % while keeping ctrl rate below 5 % (i.e., lift ≥ 10× at much higher recall).
#
# **Compute budget**: ~1 h on G4 / RTX PRO 6000 Blackwell (95.6 GB VRAM).
#
# **Outputs** (all to Drive):
# - `query_concepts_<split>.parquet` — extracted concepts for every train/val query
# - `doc_concept_embeddings_law.npy` + `_court.npy` + manifests
# - `query_concept_embeddings_<split>.npy` + manifest
# - `report.md` and `lift_table.json` — the bottom-line numbers

# %% [markdown]
# ## Cell 1 — Environment, Drive mount, install

# %%
import os, sys, subprocess, json, time, gc
from pathlib import Path

IS_COLAB = "google.colab" in sys.modules
print(f"Colab: {IS_COLAB}")

if IS_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

# Install deps. vLLM 0.6+ has stable embedding-mode for Qwen3-Embedding.
# Pin versions known to work together with the existing pipeline.
INSTALL_CMD = [
    "pip", "install", "-q",
    "vllm==0.7.3", "transformers==4.49.0", "pandas==2.2.3", "pyarrow==16.1.0",
    "numpy==1.26.4", "tqdm",
]
if IS_COLAB:
    subprocess.run(INSTALL_CMD, check=True)

# %% [markdown]
# ## Cell 2 — Paths & config
#
# Edit `DRIVE_ROOT` if your Drive layout differs. Everything else derives from it.

# %%
DRIVE_ROOT = Path("/content/drive/MyDrive/swiss_law")

# Inputs (existing artifacts on Drive)
TRAIN_CSV   = DRIVE_ROOT / "data" / "train.csv"
VAL_CSV     = DRIVE_ROOT / "data" / "val.csv"
LAW_ENRICH  = DRIVE_ROOT / "llm_enrichment_output_law_173k"   / "law_llm_descriptors_0000000_all.jsonl"
COURT_ENRICH= DRIVE_ROOT / "llm_enrichment_output_court_363k" / "court_llm_descriptors_0000000_all.jsonl"

# Outputs (new artifacts this notebook produces)
OUT_DIR     = DRIVE_ROOT / "research" / "concept_embedding_path"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_QUERY_CONCEPTS = OUT_DIR / "query_concepts.parquet"      # 1149 queries
CACHE_DOC_EMB_LAW    = OUT_DIR / "doc_concept_emb_law.fp16.npy"
CACHE_DOC_EMB_LAW_M  = OUT_DIR / "doc_concept_emb_law_manifest.parquet"
CACHE_DOC_EMB_COURT  = OUT_DIR / "doc_concept_emb_court.fp16.npy"
CACHE_DOC_EMB_COURT_M= OUT_DIR / "doc_concept_emb_court_manifest.parquet"
CACHE_QRY_EMB        = OUT_DIR / "query_concept_emb.fp16.npy"
CACHE_QRY_EMB_M      = OUT_DIR / "query_concept_emb_manifest.parquet"
REPORT_MD            = OUT_DIR / "report.md"
LIFT_JSON            = OUT_DIR / "lift_table.json"

# Models (already in the project's stack)
LLM_MODEL  = "Qwen/Qwen3-8B-AWQ"      # query concept extraction
EMB_MODEL  = "Qwen/Qwen3-Embedding-8B" # 4096-dim sentence embedding

# Experiment knobs
NEG_PER_GOLD = 5
SEED = 13
COSINE_THRESHOLDS = [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
TOP_K_TO_RECORD  = 1000   # for each query, store cosine of all gold + top-N candidates (diagnostic)

import random
random.seed(SEED)

# %% [markdown]
# ## Cell 3 — Load queries + enrichments
#
# Builds `(citation → concept_string)` lookups for laws and courts. The concept_string is what we'll embed.

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
    """Build the canonical concept string for a law row that we will embed."""
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
print(f"  law: {len(law_concepts):,} with concept_string")
del law_rows; gc.collect()

print("Loading court enrichment ...")
court_rows = _load_jsonl(COURT_ENRICH)
court_concepts = {r["citation"]: _court_concept_string(r) for r in court_rows if r.get("citation")}
print(f"  court: {len(court_concepts):,} with concept_string")
del court_rows; gc.collect()

# %% [markdown]
# ## Cell 4 — Query concept extraction via Qwen3-8B-AWQ (vLLM)
#
# Cached. Skip if `query_concepts.parquet` already exists.

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
  "legal_concepts_en": ["concept1", "concept2", ...],          // 5-15 English concepts (e.g., "pretrial detention", "collusion risk")
  "terms_de":  ["Untersuchungshaft", "Kollusionsgefahr", ...], // 5-15 German legal terms; the canonical Swiss-German vocabulary
  "terms_fr":  ["detention provisoire", "risque de collusion", ...], // 5-15 French; Swiss-French legal vocabulary (CPP / CC / CO terms)
  "terms_it":  ["detenzione preventiva", "rischio di collusione", ...], // 5-15 Italian; Swiss-Italian legal vocabulary (CPP / CC / CO terms used in Ticino BGer decisions). NEVER leave empty.
  "named_statutes": ["Art. 221 Abs. 1 StPO", ...],              // verbatim from question
  "predicted_codes": ["StPO", "BGG", ...],                      // 1-6 codes likely to contain the answer
  "is_appeal": true|false,                                      // does the question concern an appeal / Beschwerde / recours / ricorso?
  "is_fundamental_right_issue": true|false                      // does it implicate BV, EMRK, or fundamental rights?
}

All three of terms_de, terms_fr, terms_it MUST be populated with at least 5 items each,
because Swiss court paragraphs that are gold can be in any of the three languages.

QUESTION:
{QUESTION}

JSON:"""

def build_prompts(df):
    return [QUERY_EXTRACT_PROMPT.replace("{QUESTION}", str(q)) for q in df["query"].tolist()]

def parse_json_lenient(s):
    """Extract the first JSON object from a possibly-noisy LLM response."""
    s = s.strip()
    # find first '{' and matching last '}'
    a = s.find("{"); b = s.rfind("}")
    if a == -1 or b == -1: return None
    try: return json.loads(s[a:b+1])
    except Exception:
        # Try common cleanups: stray trailing commas, smart quotes
        cleaned = s[a:b+1].replace("'", '"').replace(",}", "}").replace(",]", "]")
        try: return json.loads(cleaned)
        except Exception: return None

def run_concept_extraction():
    if CACHE_QUERY_CONCEPTS.exists():
        print(f"Cached → {CACHE_QUERY_CONCEPTS}")
        return pd.read_parquet(CACHE_QUERY_CONCEPTS)

    from vllm import LLM, SamplingParams
    print(f"Loading {LLM_MODEL} on vLLM ...")
    llm = LLM(
        model=LLM_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.55,   # leaves room for the embedding model later
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,
    )
    sp = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=900,
        stop=["\n\n\n", "</json>", "QUESTION:"],
    )

    rows = []
    for split, df in [("train", train_df), ("val", val_df)]:
        print(f"Extracting concepts for {split} ({len(df)} queries) ...")
        prompts = build_prompts(df)
        outs = llm.generate(prompts, sp)
        for q_row, out in zip(df.itertuples(), outs):
            text = out.outputs[0].text if out.outputs else ""
            parsed = parse_json_lenient(text) or {}
            rows.append({
                "split": split,
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
                "parse_ok": parsed is not {} and bool(parsed.get("legal_concepts_en")),
            })

    df_out = pd.DataFrame(rows)
    df_out.to_parquet(CACHE_QUERY_CONCEPTS, index=False)
    print(f"Saved → {CACHE_QUERY_CONCEPTS}  ({len(df_out)} rows, parse_ok rate = {df_out['parse_ok'].mean():.1%})")

    # Drop LLM to free VRAM before loading embedding model
    del llm; gc.collect()
    import torch; torch.cuda.empty_cache()
    return df_out

query_concepts_df = run_concept_extraction()
query_concepts_df.head(3)

# %% [markdown]
# ## Cell 5 — Sanity check on extraction
#
# Spot-check 3 val queries against their known gold structure (val_001 should produce "pretrial detention", "collusion risk", "Untersuchungshaft", named_statutes={Art. 221 Abs. 1 lit. b StPO, ...}, is_appeal=True).

# %%
for qid in ["val_001", "val_002", "val_004", "val_010"]:
    row = query_concepts_df[query_concepts_df.query_id == qid]
    if len(row) == 0: continue
    r = row.iloc[0]
    print(f"--- {qid} ---")
    print(f"  legal_area: {r['legal_area']}")
    print(f"  concepts_en[:8]: {r['legal_concepts_en'][:8]}")
    print(f"  terms_de[:6]: {r['terms_de'][:6]}")
    print(f"  named_statutes: {r['named_statutes']}")
    print(f"  predicted_codes: {r['predicted_codes']}")
    print(f"  is_appeal={r['is_appeal']}  is_fund_right={r['is_fundamental_right_issue']}")
    print()

# %% [markdown]
# ## Cell 6 — Doc concept embeddings (Qwen3-Embedding-8B, ~530 k docs)
#
# Encodes the concept string of every enriched law + court doc. Result is one fp16 matrix per family.

# %%
def encode_qwen3_embed(texts, batch_size=128, max_length=512):
    """Stream-encode `texts` with Qwen3-Embedding-8B, returning fp16 numpy [N, 4096]."""
    import torch, numpy as np
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(EMB_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        EMB_MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda").eval()

    # Qwen3-Embedding uses last-token pooling per official guidance.
    def last_token_pool(last_hidden, attn):
        # left-padded? right-padded?
        left_padded = (attn[:, -1].sum() == attn.shape[0])
        if left_padded:
            return last_hidden[:, -1]
        seqlens = attn.sum(dim=1) - 1
        return last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seqlens]

    out = np.empty((len(texts), 4096), dtype=np.float16)
    from tqdm.auto import tqdm
    for i in tqdm(range(0, len(texts), batch_size), desc="encode"):
        chunk = texts[i:i+batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        v = last_token_pool(h, enc["attention_mask"])
        v = torch.nn.functional.normalize(v, p=2, dim=-1)
        out[i:i+v.size(0)] = v.cpu().to(torch.float16).numpy()
    del model, tok; gc.collect()
    import torch; torch.cuda.empty_cache()
    return out

def encode_and_cache(name, items_dict, emb_path, manifest_path):
    """items_dict: {citation: concept_string}. Cache fp16 matrix + manifest parquet."""
    import numpy as np
    if emb_path.exists() and manifest_path.exists():
        print(f"[{name}] cached → {emb_path}")
        return np.load(emb_path), pd.read_parquet(manifest_path)
    items = [(cit, s) for cit, s in items_dict.items() if s and s.strip()]
    print(f"[{name}] encoding {len(items):,} ...")
    citations, texts = zip(*items)
    mat = encode_qwen3_embed(list(texts))
    np.save(emb_path, mat)
    pd.DataFrame({"row": range(len(citations)), "citation": citations}).to_parquet(manifest_path, index=False)
    print(f"[{name}] saved → {emb_path}  shape={mat.shape}")
    return mat, pd.DataFrame({"row": range(len(citations)), "citation": citations})

import numpy as np
law_emb,   law_man   = encode_and_cache("law",   law_concepts,   CACHE_DOC_EMB_LAW,   CACHE_DOC_EMB_LAW_M)
court_emb, court_man = encode_and_cache("court", court_concepts, CACHE_DOC_EMB_COURT, CACHE_DOC_EMB_COURT_M)
print(f"law:   {law_emb.shape} {law_emb.dtype}")
print(f"court: {court_emb.shape} {court_emb.dtype}")

# %% [markdown]
# ## Cell 7 — Query concept embeddings
#
# Concatenate per-query `legal_concepts_en + terms_de + terms_fr + terms_it` and embed once. Per-query cost is negligible.

# %%
def build_query_string(r):
    bits = []
    bits += list(r.get("legal_concepts_en") or [])
    bits += list(r.get("terms_de") or [])
    bits += list(r.get("terms_fr") or [])
    bits += list(r.get("terms_it") or [])
    if r.get("legal_area"): bits.append(str(r["legal_area"]))
    return " | ".join(str(b) for b in bits if b)

query_concepts_df["concept_string"] = query_concepts_df.apply(build_query_string, axis=1)
q_texts = query_concepts_df["concept_string"].tolist()
print(f"Encoding {len(q_texts)} query concept strings ...")
q_emb = encode_qwen3_embed(q_texts, batch_size=64, max_length=256)
np.save(CACHE_QRY_EMB, q_emb)
query_concepts_df[["split", "query_id", "concept_string"]].to_parquet(CACHE_QRY_EMB_M, index=False)
print(f"Saved → {CACHE_QRY_EMB}  shape={q_emb.shape}")

# %% [markdown]
# ## Cell 8 — Gold vs control: cosine on concept embeddings
#
# For each (query, gold) pair and (query, random non-gold) pair, record the cosine. Family-matched negatives (law gold → law negative; court gold → court negative), mirroring the prior empirical pass exactly.

# %%
import numpy as np
import re

# Build (citation → row) lookups for fast indexing
law_cit_to_row   = {c: i for i, c in enumerate(law_man["citation"].tolist())}
court_cit_to_row = {c: i for i, c in enumerate(court_man["citation"].tolist())}

BGE_RE    = re.compile(r"^(?:BGE|ATF|DTF)\s+\d+\s+[IVX]+\s+\d+")
DOCKET_RE = re.compile(r"^[1-9][A-Z]_\d+/\d{4}")
def is_court_cit(c):
    return bool(BGE_RE.match(c) or DOCKET_RE.match(c) or re.match(r"^\d+[A-Z]\.\s*\d", c))

def cosine_for(query_idx, doc_idx, doc_emb):
    return float(np.dot(q_emb[query_idx].astype(np.float32),
                        doc_emb[doc_idx].astype(np.float32)))  # already normalized

# Index queries
qid_to_idx = {qid: i for i, qid in enumerate(query_concepts_df["query_id"].tolist())}

# Concat train + val for evaluation
eval_df = pd.concat([train_df.assign(split="train"), val_df.assign(split="val")], ignore_index=True)

all_law_cits   = list(law_cit_to_row.keys())
all_court_cits = list(court_cit_to_row.keys())

records = []  # (qid, split, kind, family, citation, cosine, found_in_enrich)
n_skipped_query = 0
n_skipped_no_enr = 0

for _, r in eval_df.iterrows():
    qid = r["query_id"]
    if qid not in qid_to_idx:
        n_skipped_query += 1
        continue
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
        # Family-matched negatives (NEG_PER_GOLD per gold)
        is_court_g = is_court_cit(g)
        for _ in range(NEG_PER_GOLD):
            if is_court_g:
                c = random.choice(all_court_cits); tries = 0
                while c in gold_set and tries < 5: c = random.choice(all_court_cits); tries += 1
                cos = cosine_for(qix, court_cit_to_row[c], court_emb)
                records.append({"qid": qid, "split": r["split"], "kind": "ctrl",
                                "family": "court", "citation": c, "cosine": cos, "found": True})
            else:
                c = random.choice(all_law_cits); tries = 0
                while c in gold_set and tries < 5: c = random.choice(all_law_cits); tries += 1
                cos = cosine_for(qix, law_cit_to_row[c], law_emb)
                records.append({"qid": qid, "split": r["split"], "kind": "ctrl",
                                "family": "law", "citation": c, "cosine": cos, "found": True})

eval_records_df = pd.DataFrame(records)
print(f"records: {len(eval_records_df):,}")
print(f"  queries skipped (no extraction): {n_skipped_query}")
print(f"  gold skipped (no enrichment):    {n_skipped_no_enr}")
print(eval_records_df.groupby(["split", "kind", "family"]).size().unstack(fill_value=0))

# %% [markdown]
# ## Cell 9 — Lift table at multiple cosine thresholds + cosine distribution

# %%
import numpy as np

def lift_at_threshold(df, tau):
    """Compute path-D-cosine fire rate for gold vs ctrl at threshold tau."""
    sub = df.dropna(subset=["cosine"])
    gold = sub[sub.kind == "gold"]
    ctrl = sub[sub.kind == "ctrl"]
    g_rate = (gold["cosine"] >= tau).mean() if len(gold) else 0.0
    c_rate = (ctrl["cosine"] >= tau).mean() if len(ctrl) else 0.0
    lift = (g_rate / c_rate) if c_rate > 0 else float("inf")
    return g_rate, c_rate, lift

print("=== Cosine threshold sweep (ALL train+val) ===")
print(f"{'tau':>6} {'gold%':>8} {'ctrl%':>8} {'lift':>10}")
rows = []
for tau in COSINE_THRESHOLDS:
    g, c, l = lift_at_threshold(eval_records_df, tau)
    print(f"{tau:>6.2f} {100*g:>7.1f}% {100*c:>7.1f}% {l:>9.1f}x")
    rows.append({"tau": tau, "gold_rate": g, "ctrl_rate": c, "lift": l, "scope": "all"})

print("\n=== LAW family only ===")
print(f"{'tau':>6} {'gold%':>8} {'ctrl%':>8} {'lift':>10}")
for tau in COSINE_THRESHOLDS:
    g, c, l = lift_at_threshold(eval_records_df[eval_records_df.family == "law"], tau)
    print(f"{tau:>6.2f} {100*g:>7.1f}% {100*c:>7.1f}% {l:>9.1f}x")
    rows.append({"tau": tau, "gold_rate": g, "ctrl_rate": c, "lift": l, "scope": "law"})

print("\n=== COURT family only ===")
print(f"{'tau':>6} {'gold%':>8} {'ctrl%':>8} {'lift':>10}")
for tau in COSINE_THRESHOLDS:
    g, c, l = lift_at_threshold(eval_records_df[eval_records_df.family == "court"], tau)
    print(f"{tau:>6.2f} {100*g:>7.1f}% {100*c:>7.1f}% {l:>9.1f}x")
    rows.append({"tau": tau, "gold_rate": g, "ctrl_rate": c, "lift": l, "scope": "court"})

print("\n=== VAL only (10 queries) ===")
print(f"{'tau':>6} {'gold%':>8} {'ctrl%':>8} {'lift':>10}")
for tau in COSINE_THRESHOLDS:
    g, c, l = lift_at_threshold(eval_records_df[eval_records_df.split == "val"], tau)
    print(f"{tau:>6.2f} {100*g:>7.1f}% {100*c:>7.1f}% {l:>9.1f}x")
    rows.append({"tau": tau, "gold_rate": g, "ctrl_rate": c, "lift": l, "scope": "val"})

# Cosine distribution percentiles
print("\n=== Cosine percentiles (informational) ===")
for scope, df in [
    ("all gold",  eval_records_df[eval_records_df.kind == "gold"]),
    ("all ctrl",  eval_records_df[eval_records_df.kind == "ctrl"]),
    ("law gold",  eval_records_df[(eval_records_df.kind == "gold") & (eval_records_df.family == "law")]),
    ("court gold",eval_records_df[(eval_records_df.kind == "gold") & (eval_records_df.family == "court")]),
]:
    s = df["cosine"].dropna().to_numpy()
    if len(s) == 0: continue
    p = np.percentile(s, [5, 25, 50, 75, 95])
    print(f"  {scope:<11} n={len(s):>6}  p5={p[0]:.3f} p25={p[1]:.3f} p50={p[2]:.3f} p75={p[3]:.3f} p95={p[4]:.3f}")

# %% [markdown]
# ## Cell 10 — Comparison vs prior token-overlap baseline + recommendation

# %%
TOKEN_BASELINE = {
    "D_concept_law":           {"gold": 0.118, "ctrl": 0.011, "lift": 11.1},
    "D_concept_law_strong":    {"gold": 0.019, "ctrl": 0.0003, "lift": 63.6},
}

# Pick the best operating point on the embedding curve: highest gold% subject to lift ≥ 10
best = None
for r in rows:
    if r["scope"] != "all": continue
    if r["lift"] >= 10 and (best is None or r["gold_rate"] > best["gold_rate"]):
        best = r

# Also report the "lift ≥ 50" operating point (precision-tilted)
best_high_prec = None
for r in rows:
    if r["scope"] != "all": continue
    if r["lift"] >= 50 and (best_high_prec is None or r["gold_rate"] > best_high_prec["gold_rate"]):
        best_high_prec = r

print("=== Bottom line ===")
print(f"Token-overlap D_concept_law:          gold={TOKEN_BASELINE['D_concept_law']['gold']*100:.1f}% lift={TOKEN_BASELINE['D_concept_law']['lift']:.1f}x")
print(f"Token-overlap D_concept_law_strong:   gold={TOKEN_BASELINE['D_concept_law_strong']['gold']*100:.1f}% lift={TOKEN_BASELINE['D_concept_law_strong']['lift']:.1f}x")
if best:
    print(f"Embedding cosine (best lift≥10):      tau={best['tau']:.2f}  gold={100*best['gold_rate']:.1f}%  ctrl={100*best['ctrl_rate']:.1f}%  lift={best['lift']:.1f}x")
if best_high_prec:
    print(f"Embedding cosine (best lift≥50):      tau={best_high_prec['tau']:.2f}  gold={100*best_high_prec['gold_rate']:.1f}%  ctrl={100*best_high_prec['ctrl_rate']:.1f}%  lift={best_high_prec['lift']:.1f}x")

# %% [markdown]
# ## Cell 11 — Save report + JSON
#
# All artifacts to Drive so the analysis is reproducible offline.

# %%
import json as _json

# Per-query cosines (for downstream debugging — keeps it lightweight)
eval_records_df.to_parquet(OUT_DIR / "eval_records.parquet", index=False)

with open(LIFT_JSON, "w", encoding="utf-8") as f:
    _json.dump({
        "config": {
            "neg_per_gold": NEG_PER_GOLD,
            "thresholds": COSINE_THRESHOLDS,
            "emb_model": EMB_MODEL,
            "llm_model": LLM_MODEL,
            "n_train_queries": int(len(train_df)),
            "n_val_queries":   int(len(val_df)),
        },
        "rows": rows,
        "token_baseline": TOKEN_BASELINE,
    }, f, indent=2)

# Compact markdown report
def fmt_section(scope_label, scope_key):
    out = [f"### {scope_label}", "", "| tau | gold% | ctrl% | lift |", "|---|---|---|---|"]
    for r in rows:
        if r["scope"] != scope_key: continue
        lift_s = f"{r['lift']:.1f}x" if r['lift'] != float('inf') else "inf"
        out.append(f"| {r['tau']:.2f} | {100*r['gold_rate']:.1f}% | {100*r['ctrl_rate']:.1f}% | {lift_s} |")
    return "\n".join(out)

md = ["# Concept-Embedding Path — Empirical Results", "",
      f"**Models**: LLM={LLM_MODEL}, Embedding={EMB_MODEL}",
      f"**Train queries**: {len(train_df)} | **Val queries**: {len(val_df)} | **Neg/gold**: {NEG_PER_GOLD}",
      "",
      "## Baseline (from prior token-overlap pass)", "",
      "| Path | Gold% | Ctrl% | Lift |", "|---|---|---|---|",
      f"| D_concept_law (≥2 token overlap)        | 11.8% | 1.1%  | 11.1x |",
      f"| D_concept_law_strong (≥4 token overlap) |  1.9% | 0.03% | 63.6x |",
      "", "## Embedding-cosine path lift",
      "", fmt_section("ALL train+val", "all"),
      "", fmt_section("LAW family only", "law"),
      "", fmt_section("COURT family only", "court"),
      "", fmt_section("VAL split only", "val"),
      ""]
if best:
    md.append(f"**Best lift≥10 operating point**: tau={best['tau']:.2f}, gold={100*best['gold_rate']:.1f}%, lift={best['lift']:.1f}x.")
if best_high_prec:
    md.append(f"**Best lift≥50 operating point**: tau={best_high_prec['tau']:.2f}, gold={100*best_high_prec['gold_rate']:.1f}%, lift={best_high_prec['lift']:.1f}x.")

REPORT_MD.write_text("\n".join(md), encoding="utf-8")
print(f"Saved: {REPORT_MD}")
print(f"       {LIFT_JSON}")
print(f"       {OUT_DIR / 'eval_records.parquet'}")

# %% [markdown]
# ## Cell 12 — Concept-cosine recall over the full law + court corpus
#
# For *every* train + val gold, compute its cosine rank against the full enriched
# corpus of its family. This answers a different question than Cell 9: not "can
# we discriminate gold from random?" but "could concept-cosine alone *retrieve*
# gold from the full 173k laws / 356k courts without any BM25/vector funnel?"
#
# If R@100 ≥ 0.5 here, concept-cosine becomes a viable third retrieval channel
# alongside BM25 and full-text vector search.

# %%
import numpy as np
import torch
from tqdm.auto import tqdm

print("GPU brute-force cosine: each query × full law + court corpus ...")
device = "cuda"

# Move to GPU as fp16 (already L2-normalized, so dot product == cosine)
law_t   = torch.from_numpy(law_emb).to(device, dtype=torch.float16)    # [N_law, 4096]
court_t = torch.from_numpy(court_emb).to(device, dtype=torch.float16)  # [N_court, 4096]
q_t_all = torch.from_numpy(q_emb).to(device, dtype=torch.float16)      # [N_q, 4096]
print(f"  law  matrix: {tuple(law_t.shape)}  ({law_t.element_size() * law_t.nelement() / 1e9:.2f} GB)")
print(f"  court matrix: {tuple(court_t.shape)} ({court_t.element_size() * court_t.nelement() / 1e9:.2f} GB)")
print(f"  q    matrix: {tuple(q_t_all.shape)}")

law_cit_arr   = law_man["citation"].to_numpy()
court_cit_arr = court_man["citation"].to_numpy()

# Gold lookup once
all_df = pd.concat([train_df.assign(split="train"), val_df.assign(split="val")], ignore_index=True)
qid_to_gold = {r["query_id"]: [g.strip() for g in str(r.get("gold_citations") or "").split(";") if g.strip()]
               for _, r in all_df.iterrows()}

TOPK_KEEP = 1000  # store top-1000 per query (per family) for rank lookup
RANK_BUCKETS = (1, 5, 10, 25, 50, 100, 200, 500, 1000)

# Compute per-query top-K for each family
def topk_per_query(doc_t, citations_arr, q_t):
    """Return per-query top-K (citations, ranks) — runs query batches to control VRAM."""
    N_q = q_t.shape[0]
    N_d = doc_t.shape[0]
    top_idx_all = np.empty((N_q, TOPK_KEEP), dtype=np.int32)
    top_sim_all = np.empty((N_q, TOPK_KEEP), dtype=np.float16)
    BATCH = 16  # 16 queries × 356k docs × 4096 × 2B ≈ 47 GB — safe
    for i in tqdm(range(0, N_q, BATCH), desc="  cosine"):
        qb = q_t[i:i+BATCH]                   # [B, 4096]
        sims = qb @ doc_t.T                   # [B, N_d]
        vals, idx = torch.topk(sims, k=min(TOPK_KEEP, N_d), dim=1)
        top_idx_all[i:i+vals.size(0)] = idx.cpu().numpy()
        top_sim_all[i:i+vals.size(0)] = vals.cpu().numpy()
        del sims, vals, idx
    return top_idx_all, top_sim_all

print("\n--- LAW corpus brute force ---")
law_topk_idx, law_topk_sim = topk_per_query(law_t, law_cit_arr, q_t_all)
print("\n--- COURT corpus brute force ---")
court_topk_idx, court_topk_sim = topk_per_query(court_t, court_cit_arr, q_t_all)

# Build per-query (citation → rank) maps from the top-K
def build_rank_map(topk_idx_row, citations_arr):
    return {citations_arr[int(idx)]: rk for rk, idx in enumerate(topk_idx_row)}

# Evaluate ranks for every gold
print("\nMeasuring per-gold rank ...")
gold_ranks = []
for q_row in range(len(query_concepts_df)):
    qid = query_concepts_df.iloc[q_row]["query_id"]
    split = query_concepts_df.iloc[q_row]["split"]
    if qid not in qid_to_gold: continue
    law_rmap   = build_rank_map(law_topk_idx[q_row],   law_cit_arr)
    court_rmap = build_rank_map(court_topk_idx[q_row], court_cit_arr)
    for g in qid_to_gold[qid]:
        if is_court_cit(g):
            rk = court_rmap.get(g, -1)
            fam = "court"
            in_corpus = g in court_cit_to_row
        else:
            rk = law_rmap.get(g, -1)
            fam = "law"
            in_corpus = g in law_cit_to_row
        gold_ranks.append({"qid": qid, "split": split, "family": fam, "gold": g,
                           "rank": rk, "in_corpus": in_corpus})

rank_df = pd.DataFrame(gold_ranks)
rank_df.to_parquet(OUT_DIR / "gold_rank_full_corpus.parquet", index=False)
print(f"saved → {OUT_DIR / 'gold_rank_full_corpus.parquet'}  ({len(rank_df):,} rows)")

# Report R@K (only counting gold that's in the enriched corpus)
print("\n=== R@K over full enriched corpus (concept-cosine only) ===")
def report_rk(df, label):
    df2 = df[df.in_corpus]
    n = len(df2)
    if n == 0:
        print(f"  {label}: n=0"); return []
    print(f"  {label}  (n={n})")
    out = []
    for k in RANK_BUCKETS:
        hit = ((df2["rank"] >= 0) & (df2["rank"] < k)).mean()
        out.append({"label": label, "k": k, "recall": float(hit), "n": int(n)})
        print(f"    R@{k:>4}: {hit:.1%}")
    return out

rk_rows = []
rk_rows += report_rk(rank_df, "ALL")
rk_rows += report_rk(rank_df[rank_df.family == "law"],   "LAW")
rk_rows += report_rk(rank_df[rank_df.family == "court"], "COURT")
rk_rows += report_rk(rank_df[rank_df.split == "val"],    "VAL")
rk_rows += report_rk(rank_df[rank_df.split == "train"],  "TRAIN")

# Persist + free VRAM
import json as _json
with open(OUT_DIR / "rank_table.json", "w", encoding="utf-8") as f:
    _json.dump(rk_rows, f, indent=2)
print(f"\nSaved → {OUT_DIR / 'rank_table.json'}")

del law_t, court_t, q_t_all
gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## Summary of what this experiment tells you
#
# 1. **If embedding-cosine at the best lift≥10 tau raises D's gold rate above ~50 %** while keeping ctrl <5 %, **the concept path becomes self-sufficient** — it alone covers half the gold the regex paths missed. Promote it to the primary D feature in `score()` and reduce reliance on the LLM judge.
#
# 2. **If embedding-cosine plateaus around 25-35 % gold** with lift in the 10-30× range, the path is a useful *evidence boost* but not a primary recall mechanism. Keep the LLM judge in Stage 5 and use cosine as a Stage-3 score feature.
#
# 3. **If embedding-cosine performs worse than token overlap** (unlikely given the model's MMTEB ranking) — that flags either an extraction failure (Cell 5 sanity check would show bad concepts) or a doc-side concept-string problem (Cell 6 build).
#
# 4. **Optional Cell 12** answers the related question: "could embedding-cosine *retrieve* the 65 % no-path gold from the full law corpus?" If R@100 ≥ 0.50 here, concept-cosine becomes a viable third retrieval channel alongside BM25 and full-text vector.
