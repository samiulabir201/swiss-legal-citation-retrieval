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
# # v7.5 — Channel 16: Aspect-Cosine (Multi-Vector Late Interaction)
#
# Builds the 16th retrieval channel for the v7.5 anchor-funnel: per-aspect query
# embeddings × cached doc concept embeddings, scored by **max-pool over aspects**
# (ColBERT-style late interaction).
#
# ## Why this channel exists
#
# The single-vector concept-cosine baseline reached **VAL R@1000 = 39 %** vs
# **TRAIN R@1000 = 62 %**. The 23-point gap is the aspect-averaging penalty: val
# queries have median K = 22 spread across 3-5 independent legal aspects, so a
# single query embedding becomes a centroid that under-ranks aspect-specific
# gold (e.g. `Art. 100 BGG` as the appeal-wrapper aspect for a substantive
# criminal-procedure query).
#
# Decomposing the query into independent aspects and scoring each candidate by
# the **best** matching aspect closes that gap. Expected VAL R@1000 lift:
# **39 % → 55-70 %**.
#
# ## Inputs (cached on Drive — nothing re-encoded)
#
# | File | Provenance |
# |---|---|
# | `doc_concept_emb_law.fp16.npy` + manifest (173 k laws)    | `concept_embedding_path_retry.ipynb`, Cell 6 |
# | `doc_concept_emb_court.fp16.npy` + manifest (356 k courts)| same |
#
# ## Outputs (this notebook produces)
#
# | File | Purpose |
# |---|---|
# | `val_aspects.parquet`                       | per-query aspect decomposition (LLM extraction) |
# | `val_aspect_embeddings.fp16.npy` + manifest | one vector per `(qid, aspect_id)` |
# | `v75_concept_cosine_channel_multi.json`     | drop-in channel for v7.5 |
# | `per_query_aspect_scores.parquet`           | per-(qid, citation) score + `best_aspect_id` — reused by Stage 4 stratified selection |
#
# ## Wall-clock on G4 / RTX PRO 6000 Blackwell
#
# ~5 minutes total: aspect LLM ~1 min, aspect encoding ~10 s, max-pool cosine ~1 min, IO ~30 s.
#
# ## Phase map
#
# | Phase | Purpose |
# |---|---|
# | 0 — Setup | Drive, install, paths |
# | 1 — Query decomposition | Qwen3-32B-AWQ emits per-aspect `{label, weight, concepts_en, terms_de/fr/it}` |
# | 2 — Multi-vector embedding | Qwen3-Embedding-8B → one vector per `(qid, aspect)` |
# | 3 — Late-interaction scoring | Per-doc `max(cosine(doc, aspect_i))` + `best_aspect_id` |
# | 4 — Persistence | Channel JSON + per-aspect score table |
# | 5 — v7.5 integration | Three paste-ready insertions |

# %% [markdown]
# ---
# ## Phase 0 — Setup

# %% [markdown]
# ### 0.1 Mount Drive + install pinned deps

# %%
import os, sys, subprocess, json, time, gc, io
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

IS_COLAB = "google.colab" in sys.modules
print(f"Colab runtime: {IS_COLAB}")

if IS_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    subprocess.run([
        "pip", "install", "-q", "-U",
        "vllm>=0.8.5", "transformers>=4.51.0", "pandas==2.2.3",
        "pyarrow==16.1.0", "numpy==1.26.4", "tqdm",
    ], check=True)

# %% [markdown]
# ### 0.2 Paths
#
# Outputs land in a `multi_aspect/` subdir so they don't clobber the
# single-vector cache from `concept_embedding_path_retry.ipynb`.

# %%
DRIVE_ROOT = Path("/content/drive/MyDrive/swiss_law")

VAL_CSV     = DRIVE_ROOT / "data" / "val.csv"
CONCEPT_DIR = DRIVE_ROOT / "research" / "concept_embedding_path"
MULTI_DIR   = CONCEPT_DIR / "multi_aspect"
MULTI_DIR.mkdir(parents=True, exist_ok=True)

# Inputs (already on Drive — reused, NOT recomputed)
CACHE_DOC_EMB_LAW     = CONCEPT_DIR / "doc_concept_emb_law.fp16.npy"
CACHE_DOC_EMB_LAW_M   = CONCEPT_DIR / "doc_concept_emb_law_manifest.parquet"
CACHE_DOC_EMB_COURT   = CONCEPT_DIR / "doc_concept_emb_court.fp16.npy"
CACHE_DOC_EMB_COURT_M = CONCEPT_DIR / "doc_concept_emb_court_manifest.parquet"

# Outputs (new artifacts)
ASPECTS_PARQUET     = MULTI_DIR / "val_aspects.parquet"
ASPECT_EMB          = MULTI_DIR / "val_aspect_embeddings.fp16.npy"
ASPECT_EMB_M        = MULTI_DIR / "val_aspect_manifest.parquet"
CHANNEL_JSON        = MULTI_DIR / "v75_concept_cosine_channel_multi.json"
PER_Q_ASPECT_SCORES = MULTI_DIR / "per_query_aspect_scores.parquet"

LLM_MODEL    = "Qwen/Qwen3-32B-AWQ"
EMB_MODEL    = "Qwen/Qwen3-Embedding-8B"
PER_FAMILY_K = 1500   # candidates kept per family per query (same scale as v7.5 budgets)

print("Required inputs on Drive:")
for f in [CACHE_DOC_EMB_LAW, CACHE_DOC_EMB_LAW_M, CACHE_DOC_EMB_COURT, CACHE_DOC_EMB_COURT_M]:
    print(f"  {'OK' if f.exists() else 'MISSING'}  {f}")

# %% [markdown]
# ---
# ## Phase 1 — Per-aspect query decomposition

# %% [markdown]
# ### 1.1 Load val.csv

# %%
import pandas as pd

val_df = pd.read_csv(VAL_CSV)
val_qids = val_df["query_id"].tolist()
print(f"val queries: {len(val_df)}\n")
for _, r in val_df.iterrows():
    g_count = len([c for c in str(r["gold_citations"]).split(";") if c.strip()])
    head = str(r["query"])[:90] + ("..." if len(str(r["query"])) > 90 else "")
    print(f"  {r['query_id']:<8}  K={g_count:>2}   {head}")

# %% [markdown]
# ### 1.2 Aspect extraction prompt
#
# The schema below forces the LLM to emit **independent** aspects (different
# statutes / different court paragraphs), trilingual terms for each, and a
# weight that estimates how much of the gold falls on that aspect. The weights
# normalise to 1.0 downstream.

# %%
ASPECT_PROMPT = """You are a Swiss legal analyst. Decompose this question into its independent legal aspects.

A Swiss legal question typically has 2-5 aspects. Aspects are INDEPENDENT legal
issues that would be answered by DIFFERENT statutes or court paragraphs:
- Criminal procedure: (i) substantive basis (collusion/flight/reoffending risk); (ii) procedural mechanics (extension, review); (iii) proportionality; (iv) appeal/cost wrapper.
- Civil obligations: (i) substantive rule (contract/delict); (ii) standard of care; (iii) damages; (iv) appeal wrapper.
- "What does Art. X say?" — 1 aspect.

Per aspect emit COMPACTLY:
- "id":     "a1" | "a2" | ...
- "label":  2-5 word description
- "weight": fraction of gold on this aspect (weights sum to ~1.0)
- "concepts_en": 4-6 English concepts SPECIFIC to this aspect
- "terms_de":    3-5 Swiss-German legal terms
- "terms_fr":    3-5 Swiss-French legal terms
- "terms_it":    3-5 Swiss-Italian legal terms

All three of terms_de, terms_fr, terms_it MUST be populated for EVERY aspect.

Output ONLY the JSON object — no prose before or after. Begin with `{` and end with `}`.

{
  "aspects": [
    {"id": "a1", "label": "...", "weight": 0.45,
     "concepts_en": [...], "terms_de": [...], "terms_fr": [...], "terms_it": [...]},
    {"id": "a2", "label": "...", "weight": 0.30, ...}
  ],
  "named_statutes": ["Art. 221 Abs. 1 StPO", ...],
  "expected_codes": ["StPO", "BGG"],
  "is_appeal": true,
  "is_fundamental_right_issue": false
}

QUESTION:
{QUESTION}

JSON:"""


import re as _re

def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    return s.strip()

def _extract_top_level_field(text: str, field: str):
    """Best-effort regex extraction of a top-level scalar/list/string field."""
    m = _re.search(rf'"{field}"\s*:\s*(\[[^\]]*\]|true|false|"[^"]*"|\d+)', text, _re.DOTALL)
    if m is None:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

def _walk_aspect_blocks(text: str):
    """Yield each balanced `{...}` block found inside the aspects:[...] array."""
    arr_start = _re.search(r'"aspects"\s*:\s*\[', text)
    if arr_start is None:
        return
    i = arr_start.end()
    n = len(text)
    while i < n:
        # skip whitespace + commas
        while i < n and text[i] in " \t\n\r,":
            i += 1
        if i >= n or text[i] == "]":
            return
        if text[i] != "{":
            i += 1
            continue
        # balanced-brace walk, respecting strings
        depth = 0
        start = i
        in_str = False
        esc = False
        while i < n:
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[start:i + 1]
                        i += 1
                        break
            i += 1
        else:
            return  # unbalanced — give up

def parse_aspects_lenient(text: str):
    """Robust extraction: try full JSON; fall back to per-aspect brace extraction.
    Returns dict with at least 'aspects' (possibly empty). Returns None only if
    nothing could be salvaged at all."""
    s = _strip_fences(text)

    # 1) Full parse
    a, b = s.find("{"), s.rfind("}")
    if a != -1 and b != -1 and b > a:
        for candidate in [s[a:b + 1],
                          s[a:b + 1].replace(",}", "}").replace(",]", "]")]:
            try:
                obj = json.loads(candidate)
                if isinstance(obj.get("aspects"), list):
                    return obj
            except Exception:
                pass

    # 2) Per-aspect salvage
    aspects = []
    for block in _walk_aspect_blocks(s):
        try:
            aspect = json.loads(block)
            if isinstance(aspect, dict) and aspect.get("id"):
                aspects.append(aspect)
        except Exception:
            try:
                aspect = json.loads(block.replace(",}", "}").replace(",]", "]"))
                if isinstance(aspect, dict) and aspect.get("id"):
                    aspects.append(aspect)
            except Exception:
                continue

    if not aspects:
        return None

    return {
        "aspects":                    aspects,
        "named_statutes":             _extract_top_level_field(s, "named_statutes") or [],
        "expected_codes":             _extract_top_level_field(s, "expected_codes") or [],
        "is_appeal":                  bool(_extract_top_level_field(s, "is_appeal")),
        "is_fundamental_right_issue": bool(_extract_top_level_field(s, "is_fundamental_right_issue")),
    }

# Single-aspect retry prompt — used for queries the main pass couldn't parse.
# Much smaller per-call output (one aspect at a time), almost impossible to truncate.
SINGLE_ASPECT_PROMPT = """You are a Swiss legal analyst. Identify ONE specific legal aspect of this question (the {WHICH} aspect by importance: e.g., 1st = main substantive rule, 2nd = procedural mechanics, 3rd = proportionality / fundamental rights, 4th = appeal/cost wrapper).

Return a JSON object for ONE aspect ONLY. Do not return an array.

{
  "id": "a{N}",
  "label": "<2-5 words>",
  "weight": <0.0-1.0>,
  "concepts_en": ["concept1", "concept2", "concept3", "concept4"],
  "terms_de":    ["German term 1", "German term 2", "German term 3"],
  "terms_fr":    ["French term 1", "French term 2", "French term 3"],
  "terms_it":    ["Italian term 1", "Italian term 2", "Italian term 3"]
}

If this question does NOT have a {WHICH} distinct aspect, return:
  {"id": null}

QUESTION:
{QUESTION}

JSON:"""

# %% [markdown]
# ### 1.3 Run aspect extraction (two-pass, parse-tolerant)
#
# **Pass 1** — main extraction with `max_tokens=4096` (multi-aspect trilingual output is
# verbose; the earlier `max_tokens=2048` truncated half of val).
#
# **Pass 2** — per-aspect retry for queries that pass 1 couldn't parse: asks for ONE
# aspect at a time with a tiny output budget. Almost impossible to truncate.
#
# **Parser** — full JSON parse first; on failure, falls back to walking balanced
# `{aspect}` blocks individually. Salvages partial output (e.g. complete aspect a1
# even if a2 is truncated).
#
# **Cache guard** — only honors `val_aspects.parquet` if its parse_ok rate is ≥ 90%.
# If a previous run produced a broken cache, it's deleted and re-extracted.
#
# **Diagnostic** — for any query that still has 0 aspects after both passes, the head
# and tail of `raw_llm` are printed so the failure mode is visible, not silent.

# %%
def _normalise_aspects(aspects):
    """Normalise weights to sum to 1.0; ensure required keys exist."""
    if not aspects:
        return []
    out = []
    for a in aspects:
        if not isinstance(a, dict):
            continue
        out.append({
            "id":          a.get("id") or f"a{len(out) + 1}",
            "label":       a.get("label") or "",
            "weight":      float(a.get("weight", 0) or 0),
            "concepts_en": list(a.get("concepts_en") or []),
            "terms_de":    list(a.get("terms_de") or []),
            "terms_fr":    list(a.get("terms_fr") or []),
            "terms_it":    list(a.get("terms_it") or []),
        })
    tot_w = sum(a["weight"] for a in out) or 1.0
    for a in out:
        a["weight"] = a["weight"] / tot_w
    return out

_use_cache = False
if ASPECTS_PARQUET.exists():
    _df = pd.read_parquet(ASPECTS_PARQUET)
    _rate = _df["parse_ok"].mean()
    if _rate >= 0.90:
        print(f"Cached (parse_ok={_rate:.0%}) → {ASPECTS_PARQUET}")
        aspects_df = _df
        _use_cache = True
        print(f"  mean aspects/query: {aspects_df['n_aspects'].mean():.1f}  "
              f"|  range: [{aspects_df['n_aspects'].min()}-{aspects_df['n_aspects'].max()}]")
    else:
        print(f"Cached parse_ok={_rate:.0%} < 90% — re-extracting from scratch.")
        ASPECTS_PARQUET.unlink()
        # Also drop the downstream embedding cache so it gets rebuilt for the new aspects
        for f in [ASPECT_EMB, ASPECT_EMB_M]:
            if f.exists():
                f.unlink(); print(f"  deleted stale → {f}")
    del _df

if not _use_cache:
    from vllm import LLM, SamplingParams
    print(f"Loading {LLM_MODEL} on vLLM ...")
    llm = LLM(
        model=LLM_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.55,
        max_model_len=32768,
        enforce_eager=False,
        trust_remote_code=True,
    )
    sp_main = SamplingParams(
        temperature=0.0, top_p=1.0,
        max_tokens=4096,                                  # was 2048; gives full multi-aspect room
        stop=["\nQUESTION:", "</json>"],                  # drop "\n```" which was firing too early
    )

    # ---------- Pass 1: main extraction ----------
    queries = val_df["query"].tolist()
    qids    = val_df["query_id"].tolist()
    prompts = [ASPECT_PROMPT.replace("{QUESTION}", str(q)) for q in queries]
    print(f"\nPass 1 — main extraction on {len(prompts)} queries (max_tokens=4096) ...")
    outs = llm.generate(prompts, sp_main)

    raw_texts = [(out.outputs[0].text if out.outputs else "") for out in outs]
    parsed_list = [parse_aspects_lenient(t) for t in raw_texts]

    pass1_parse_rate = sum(1 for p in parsed_list if p and p.get("aspects")) / max(1, len(parsed_list))
    print(f"  pass-1 parse_ok: {pass1_parse_rate:.0%}")

    # ---------- Pass 2: per-aspect retry for failures ----------
    failed = [i for i, p in enumerate(parsed_list) if not (p and p.get("aspects"))]
    if failed:
        print(f"\nPass 2 — single-aspect retry for {len(failed)} failures ...")
        sp_retry = SamplingParams(
            temperature=0.0, top_p=1.0,
            max_tokens=1024,                              # single aspect — small output
            stop=["\n\n", "QUESTION:", "</json>"],
        )
        ORDINALS = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}
        for orig_i in failed:
            q_text = str(queries[orig_i])
            collected = []
            # Try up to 5 aspect positions; stop once we get "null" back
            for n in range(1, 6):
                prompt = (SINGLE_ASPECT_PROMPT
                          .replace("{N}", str(n))
                          .replace("{WHICH}", ORDINALS[n])
                          .replace("{QUESTION}", q_text))
                out = llm.generate([prompt], sp_retry)[0]
                t = out.outputs[0].text if out.outputs else ""
                # Parse a single object
                ts = _strip_fences(t)
                a_idx, b_idx = ts.find("{"), ts.rfind("}")
                if a_idx == -1 or b_idx == -1:
                    break
                try:
                    obj = json.loads(ts[a_idx:b_idx + 1])
                except Exception:
                    try:
                        obj = json.loads(ts[a_idx:b_idx + 1].replace(",}", "}").replace(",]", "]"))
                    except Exception:
                        break
                if not obj.get("id") or obj["id"] in (None, "null"):
                    break
                collected.append(obj)
            if collected:
                parsed_list[orig_i] = {
                    "aspects":                    collected,
                    "named_statutes":             [],
                    "expected_codes":             [],
                    "is_appeal":                  False,
                    "is_fundamental_right_issue": False,
                }
                print(f"  {qids[orig_i]}: recovered {len(collected)} aspects via retry")
            else:
                print(f"  {qids[orig_i]}: still 0 aspects after retry — see raw_llm")

    # ---------- Build DataFrame ----------
    rows = []
    for i, (qid, text, parsed) in enumerate(zip(qids, raw_texts, parsed_list)):
        parsed = parsed or {}
        aspects = _normalise_aspects(parsed.get("aspects") or [])
        rows.append({
            "query_id":       qid,
            "raw_llm":        text[:8000],
            "aspects":        aspects,
            "named_statutes": parsed.get("named_statutes", []) or [],
            "expected_codes": parsed.get("expected_codes", []) or [],
            "is_appeal":      bool(parsed.get("is_appeal", False)),
            "is_fund_right":  bool(parsed.get("is_fundamental_right_issue", False)),
            "n_aspects":      len(aspects),
            "parse_ok":       bool(aspects),
        })

    aspects_df = pd.DataFrame(rows)
    aspects_df.to_parquet(ASPECTS_PARQUET, index=False)
    print(f"\nSaved → {ASPECTS_PARQUET}")
    print(f"  parse_ok: {aspects_df['parse_ok'].mean():.0%}  "
          f"|  mean aspects/query: {aspects_df['n_aspects'].mean():.1f}  "
          f"|  range: [{aspects_df['n_aspects'].min()}-{aspects_df['n_aspects'].max()}]")

    # Diagnostic dump for any remaining failures
    fails = aspects_df[~aspects_df["parse_ok"]]
    if len(fails):
        print(f"\n[!] {len(fails)} queries still have 0 aspects. raw_llm head/tail for each:")
        for _, r in fails.iterrows():
            raw = r["raw_llm"]
            print(f"\n--- {r['query_id']} (raw_llm len={len(raw)}) ---")
            print(f"HEAD: {raw[:400]!r}")
            print(f"TAIL: {raw[-400:]!r}")

    del llm; gc.collect()
    import torch; torch.cuda.empty_cache()

# %% [markdown]
# ### 1.4 Spot-check aspects
#
# Verify trilingual decomposition is plausible before paying for embeddings.
# val_001 should produce ~4 aspects (collusion / proportionality / extension / appeal-wrapper).
# val_004 should produce ~2 (holographic-will formality + testamentary capacity).

# %%
for qid in ["val_001", "val_004", "val_010"]:
    row = aspects_df[aspects_df.query_id == qid]
    if len(row) == 0: continue
    r = row.iloc[0]
    print(f"\n=== {qid} | n_aspects={r['n_aspects']} | is_appeal={r['is_appeal']} ===")
    for a in r["aspects"]:
        w = a.get("weight", 0)
        print(f"  [{a.get('id')}] w={w:.2f}  {a.get('label')}")
        print(f"      concepts_en: {list(a.get('concepts_en') or [])[:5]}")
        print(f"      terms_de:    {list(a.get('terms_de')    or [])[:5]}")
        print(f"      terms_fr:    {list(a.get('terms_fr')    or [])[:5]}")
        print(f"      terms_it:    {list(a.get('terms_it')    or [])[:5]}")

# %% [markdown]
# ---
# ## Phase 2 — Multi-vector embedding

# %% [markdown]
# ### 2.1 Build per-aspect concept strings
#
# Each aspect becomes one input string for the embedding model. The string
# concatenates `concepts_en | terms_de | terms_fr | terms_it | label` with a
# delimiter that survives tokenization cleanly. Aspects are flattened across
# all val queries — typically ~40-50 rows total.

# %%
import numpy as np

def build_aspect_string(aspect: dict) -> str:
    bits = []
    bits += [str(c) for c in (aspect.get("concepts_en") or [])]
    bits += [str(c) for c in (aspect.get("terms_de")    or [])]
    bits += [str(c) for c in (aspect.get("terms_fr")    or [])]
    bits += [str(c) for c in (aspect.get("terms_it")    or [])]
    if aspect.get("label"):
        bits.append(str(aspect["label"]))
    return " | ".join(b for b in bits if b)

aspect_rows = []
for _, r in aspects_df.iterrows():
    qid = r["query_id"]
    for a in r["aspects"]:
        s = build_aspect_string(a)
        if s.strip():
            aspect_rows.append({
                "query_id":       qid,
                "aspect_id":      a.get("id"),
                "label":          a.get("label"),
                "weight":         float(a.get("weight", 0)),
                "concept_string": s,
            })
aspect_manifest = pd.DataFrame(aspect_rows)
print(f"Aspect rows: {len(aspect_manifest)}  "
      f"(mean {aspect_manifest.groupby('query_id').size().mean():.1f}/query)")

# %% [markdown]
# ### 2.2 Encode aspects with Qwen3-Embedding-8B
#
# Last-token pool + L2 normalise (the official Qwen3-Embedding recipe). ~50 inputs → seconds.

# %%
def encode_qwen3_embed(texts, batch_size=64, max_length=256):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok   = AutoTokenizer.from_pretrained(EMB_MODEL, trust_remote_code=True)
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
    for i in tqdm(range(0, len(texts), batch_size), desc="encode_aspect"):
        chunk = texts[i:i+batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = model(**enc).last_hidden_state
        v = last_token_pool(h, enc["attention_mask"])
        v = torch.nn.functional.normalize(v, p=2, dim=-1)
        out[i:i+v.size(0)] = v.cpu().to(torch.float16).numpy()
    del model, tok; gc.collect()
    torch.cuda.empty_cache()
    return out

if ASPECT_EMB.exists() and ASPECT_EMB_M.exists():
    print(f"Cached → {ASPECT_EMB}")
    aspect_emb      = np.load(ASPECT_EMB)
    aspect_manifest = pd.read_parquet(ASPECT_EMB_M)
else:
    aspect_emb = encode_qwen3_embed(aspect_manifest["concept_string"].tolist())
    np.save(ASPECT_EMB, aspect_emb)
    aspect_manifest.to_parquet(ASPECT_EMB_M, index=False)
    print(f"Saved → {ASPECT_EMB}  shape={aspect_emb.shape}")

# %% [markdown]
# ---
# ## Phase 3 — Late-interaction scoring
#
# For each `(query, doc)` pair: `score = max_i cos(doc, aspect_i)`.
# Also records `best_aspect_id` per doc — directly consumable by Stage 4
# aspect-stratified selection without recomputation.

# %% [markdown]
# ### 3.1 Load cached doc embeddings to GPU

# %%
import torch
from tqdm.auto import tqdm

print("Loading cached doc embeddings ...")
law_emb   = np.load(CACHE_DOC_EMB_LAW)
law_man   = pd.read_parquet(CACHE_DOC_EMB_LAW_M)
court_emb = np.load(CACHE_DOC_EMB_COURT)
court_man = pd.read_parquet(CACHE_DOC_EMB_COURT_M)
print(f"  law:   {law_emb.shape}  ({law_emb.nbytes / 1e9:.2f} GB)")
print(f"  court: {court_emb.shape}  ({court_emb.nbytes / 1e9:.2f} GB)")

device     = "cuda"
law_t      = torch.from_numpy(law_emb).to(device, dtype=torch.float16)
court_t    = torch.from_numpy(court_emb).to(device, dtype=torch.float16)
aspect_t   = torch.from_numpy(aspect_emb).to(device, dtype=torch.float16)
law_cits   = law_man["citation"].to_numpy()
court_cits = court_man["citation"].to_numpy()

# qid -> array of row indices in aspect_emb / aspect_manifest
qid_to_aspect_rows = aspect_manifest.groupby("query_id").indices

# %% [markdown]
# ### 3.2 Max-pool over aspects (per query × per family)
#
# Top-K per aspect within each family, then merge → take max. This keeps the
# per-aspect winners distinct so a doc that's a strong match on aspect a3
# can't get squeezed out by docs that are mediocre on every aspect.

# %%
PER_FAM_TOPK_INTERNAL = max(PER_FAMILY_K, 2000)

per_query_doc_scores = {}     # qid -> { citation -> {"score": max_cos, "best_aspect": id, "aspect_scores": {...}} }
for qid in val_qids:
    rows = qid_to_aspect_rows.get(qid, np.array([], dtype=int))
    if len(rows) == 0:
        per_query_doc_scores[qid] = {}
        print(f"  {qid:<8}  no aspects extracted (parse failed?)")
        continue
    q_aspect_t  = aspect_t[rows]                                    # [n_aspects, 4096]
    q_aspect_ids = aspect_manifest.iloc[rows]["aspect_id"].tolist()

    doc_best = {}    # citation -> [max_score, best_aspect_id, {aspect_id: per_aspect_score}]
    for fam_t, cits in [(law_t, law_cits), (court_t, court_cits)]:
        sims = q_aspect_t @ fam_t.T                                 # [n_aspects, N_fam]
        for ai in range(sims.size(0)):
            vals, idx = torch.topk(sims[ai], k=min(PER_FAM_TOPK_INTERNAL, sims.size(1)))
            a_id = q_aspect_ids[ai]
            for v, i in zip(vals.cpu().tolist(), idx.cpu().tolist()):
                cit = str(cits[i])
                cur = doc_best.get(cit)
                if cur is None:
                    doc_best[cit] = [float(v), a_id, {a_id: float(v)}]
                else:
                    if v > cur[0]:
                        cur[0], cur[1] = float(v), a_id
                    cur[2][a_id] = max(cur[2].get(a_id, -1.0), float(v))
        del sims

    # Keep top 2× PER_FAMILY_K combined across both families (channel will downsample at injection)
    items = sorted(doc_best.items(), key=lambda kv: -kv[1][0])[:PER_FAMILY_K * 2]
    per_query_doc_scores[qid] = {
        cit: {"score": s[0], "best_aspect": s[1], "aspect_scores": s[2]}
        for cit, s in items
    }
    print(f"  {qid:<8}  candidates after max-pool: {len(per_query_doc_scores[qid])}")

del aspect_t, law_t, court_t
gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ---
# ## Phase 4 — Persistence
#
# Two artifacts:
# 1. **Channel JSON** — `(citation, score, best_aspect)` triples per query.
#    v7.5 patch resolves citation → doc_id at injection time via `cit_to_doc_ids`.
# 2. **Per-aspect score table** — full `(qid, citation, score, best_aspect, aspect_scores)`
#    for downstream stages that want aspect attribution (Stage 4 stratified selection,
#    Stage 5 dossier features).

# %%
# 4.1 Channel JSON (citation-keyed)
CONCEPT_COSINE_MULTI_BY_CIT = {}
for qid, doc_scores in per_query_doc_scores.items():
    hits = sorted(
        ((cit, info["score"], info["best_aspect"]) for cit, info in doc_scores.items()),
        key=lambda x: -x[1],
    )
    CONCEPT_COSINE_MULTI_BY_CIT[qid] = hits

serializable = {
    qid: [(cit, float(sc), str(ba)) for (cit, sc, ba) in hits]
    for qid, hits in CONCEPT_COSINE_MULTI_BY_CIT.items()
}
with open(CHANNEL_JSON, "w", encoding="utf-8") as f:
    json.dump(serializable, f)
print(f"Saved channel → {CHANNEL_JSON}")
print(f"  per-query candidate counts: {[len(v) for v in CONCEPT_COSINE_MULTI_BY_CIT.values()]}")

# 4.2 Per-aspect score table (for Stage 4 / dossier features)
score_rows = []
for qid, doc_scores in per_query_doc_scores.items():
    for cit, info in doc_scores.items():
        score_rows.append({
            "query_id":      qid,
            "citation":      cit,
            "score":         info["score"],
            "best_aspect":   info["best_aspect"],
            "aspect_scores": json.dumps(info["aspect_scores"]),
        })
pd.DataFrame(score_rows).to_parquet(PER_Q_ASPECT_SCORES, index=False)
print(f"Saved per-aspect score table → {PER_Q_ASPECT_SCORES}")

# Spot-check
print("\nTop-5 candidates per query (cosine score, best_aspect, citation):")
for qid in val_qids[:4]:
    print(f"  {qid}")
    for cit, sc, ba in CONCEPT_COSINE_MULTI_BY_CIT.get(qid, [])[:5]:
        print(f"     {sc:.3f}  [{ba}]  {cit}")

# %% [markdown]
# ---
# ## Phase 5 — v7.5 integration
#
# Three paste-ready cells for your v7.5 notebook. Run THIS notebook first so the
# Drive artifacts exist, then copy each cell below into v7.5 at the indicated location.
#
# These cells are **not meant to execute in this notebook** — they reference
# `cit_to_doc_ids`, `CONFIG`, and `ch_out` which only exist inside v7.5. If you
# Run-All here, they'll raise `AssertionError` / `NameError` — that's expected,
# skip them. They're here as proper code cells so you can copy them syntax-highlit
# and indented correctly into v7.5.

# %% [markdown]
# ### 5.1 Insertion A — paste as a NEW CELL **after** v7.5 Cell 34 (end of Phase 8)
#
# Loads the multi-vector channel from Drive and resolves citations → v7.5 doc IDs.
# Defines `concept_cosine_for(qid, budget)` used by the master-loop patch (5.2).

# %%
# === COPY THIS CELL INTO v7.5 ===
# Phase 8.5: Concept-cosine channel (MULTI-VECTOR, max-pool over aspects)
import json
from pathlib import Path
import numpy as np

CONCEPT_MULTI_DIR = Path("/content/drive/MyDrive/swiss_law/research/concept_embedding_path/multi_aspect")
CHANNEL_JSON      = CONCEPT_MULTI_DIR / "v75_concept_cosine_channel_multi.json"

assert "cit_to_doc_ids" in globals(), "Run v7.5 Phase 2 first (builds cit_to_doc_ids)"

with open(CHANNEL_JSON, encoding="utf-8") as _f:
    _channel_cit = json.load(_f)

CONCEPT_COSINE_CHANNEL     = {}   # qid -> list[(did, score)]
CONCEPT_COSINE_BEST_ASPECT = {}   # qid -> dict[did -> aspect_id]
for qid, triples in _channel_cit.items():
    best, aspects = {}, {}
    for (cit, score, best_aspect) in triples:
        for did in cit_to_doc_ids.get(cit, []):
            if did not in best or score > best[did]:
                best[did]    = score
                aspects[did] = best_aspect
    CONCEPT_COSINE_CHANNEL[qid]     = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    CONCEPT_COSINE_BEST_ASPECT[qid] = aspects

print(f"Loaded multi-vector channel: {len(CONCEPT_COSINE_CHANNEL)} queries, "
      f"avg candidates/query = {np.mean([len(v) for v in CONCEPT_COSINE_CHANNEL.values()]):.0f}")

def concept_cosine_for(qid, budget=None):
    hits = CONCEPT_COSINE_CHANNEL.get(qid, [])
    return hits if budget is None else hits[:budget]

# %% [markdown]
# ### 5.2 Insertion B — paste INSIDE the master loop in v7.5 Cell 36
#
# Locate these two lines in v7.5 Cell 36:
#
# ```python
# ch_out   = run_channels(q["query_text"], ALL_TARGETS[qid], ...)
# fuse_out = fuse_for_query(ch_out["channels"], g_set, g_tot)
# ```
#
# Paste the six lines below **between them** (at the same indent level as `ch_out =`).

# %%
# === PASTE BETWEEN `ch_out = run_channels(...)` AND `fuse_out = fuse_for_query(...)` IN v7.5 CELL 36 ===
_cc_budget = CONFIG.get("channel_budgets", {}).get("concept_cosine", 2000)
_cc_hits   = concept_cosine_for(qid, _cc_budget)
ch_out["channels"].append(("concept_cosine", _cc_hits))
_cc_did_set = {d for d, _ in _cc_hits}
ch_out.setdefault("channel_recalls", {})["concept_cosine"] = (
    len(_cc_did_set & g_set) / max(1, g_tot)
)

# %% [markdown]
# ### 5.3 Insertion C — paste at the END of v7.5 Cell 8 (Phase 1.4 knob panel)
#
# Three assignments adding the channel to v7.5's `CONFIG`. Run *after* `CONFIG = {...}`
# is defined, so the new keys land in the existing dict.

# %%
# === PASTE AT THE END OF v7.5 CELL 8 (after CONFIG is built) ===
CONFIG["channel_weights"]["concept_cosine"] = 1.0   # equal weight to start; tune after first measurement
CONFIG["channel_budgets"]["concept_cosine"] = 2000
# Optional: include in guarantee_channels so the channel always contributes to the final pool.
# CONFIG["guarantee_channels"].append("concept_cosine")

# %% [markdown]
# ### 5.4 What to expect when v7.5 re-runs
#
# - **Phase 10.3 "Per-channel mean recall"** will print a new row `concept_cosine`. That row
#   IS the standalone R@2000 of the channel on val.
# - **Phase 10.2 "Macro R@K curve"** reflects the fused pool with the new channel mixed in.
# - **Decision rule**: if `concept_cosine` standalone R@2000 ≥ 0.50 on val, the
#   channel is doing its job. If the fused pool's R@2000 lifts by ≥ 0.04 over the
#   pre-channel baseline (current 0.611), ship it.
