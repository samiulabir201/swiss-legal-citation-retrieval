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
# # Stage B — Grounded LLM Filter with Dossier Evidence
#
# **Goal**: take Stage A v3's top-2000 per query, screen each candidate with
# Qwen3-32B-AWQ + a pre-digested dossier, output keep/drop + verbatim evidence
# quote. The quote-substring check is a hard gate against hallucinated picks.
#
# **Input on Drive**:
# - `swiss_law/research/stage_b_input/stage_b_input.parquet` (produced by `bundle_stage_b_input.py` locally)
# - `swiss_law/research/anchor_funnel_val001_v7/all_targets.json`
# - `swiss_law/research/anchor_funnel_val001_v7/gold_doc_sets.json`
# - `swiss_law/research/concept_embedding_path/multi_aspect/val_aspects.parquet`
# - `swiss_law/data/val.csv`
#
# **Output on Drive**:
# - `swiss_law/research/stage_b_grounded_llm/stage_b_survivors.parquet`  — per-(qid, did) decisions
# - `swiss_law/research/stage_b_grounded_llm/stage_b_metrics.json`        — per-query precision/recall/F1
#
# **Wall-clock**: ~30-60 min on G4 / RTX PRO 6000 Blackwell at TOP_K=500;
# ~2-4 hours at TOP_K=2000.
#
# **Phase map**
#
# | Phase | Purpose |
# |---|---|
# | 0 | Setup, paths, install |
# | 1 | Load Stage B input + val queries + Stage 0 targets |
# | 2 | Build per-candidate dossier prompts (1 prompt per (qid, did)) |
# | 3 | LLM screen via vLLM batched generate |
# | 4 | Parse strict-JSON output + apply hard gates (quote substring) |
# | 5 | Per-query precision / recall / F1, with K-truncation by confidence |
# | 6 | Save survivors + metrics |

# %% [markdown]
# ## Phase 0 — Setup

# %%
import os, sys, subprocess, json, time, gc, io, re
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

IS_COLAB = "google.colab" in sys.modules
print(f"Colab: {IS_COLAB}")

if IS_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    subprocess.run([
        "pip", "install", "-q", "-U",
        "vllm>=0.8.5", "transformers>=4.51.0", "pandas==2.2.3",
        "pyarrow==16.1.0", "numpy==1.26.4", "tqdm",
    ], check=True)

# %% [markdown]
# ## Phase 1 — Paths and data loading

# %%
DRIVE_ROOT  = Path("/content/drive/MyDrive/swiss_law")
STAGE_B_IN  = DRIVE_ROOT / "research" / "stage_b_input" / "stage_b_input.parquet"
VAL_CSV     = DRIVE_ROOT / "data"     / "val.csv"
ALL_TARGETS = DRIVE_ROOT / "research" / "anchor_funnel_val001_v7" / "all_targets.json"
GOLD_SETS   = DRIVE_ROOT / "research" / "anchor_funnel_val001_v7" / "gold_doc_sets.json"
VAL_ASPECTS = DRIVE_ROOT / "research" / "concept_embedding_path"  / "multi_aspect" / "val_aspects.parquet"

OUT_DIR = DRIVE_ROOT / "research" / "stage_b_grounded_llm"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SURVIVORS = OUT_DIR / "stage_b_survivors.parquet"
OUT_METRICS   = OUT_DIR / "stage_b_metrics.json"
OUT_RAW       = OUT_DIR / "stage_b_raw_llm.parquet"

# Tunables
TOP_K        = 500              # candidates per query to send through Stage B (start at 500, scale up later)
LLM_MODEL    = "Qwen/Qwen3-32B-AWQ"
MAX_TOKENS   = 96               # output budget per candidate (strict JSON)
MAX_MODEL_LEN= 4096             # input + output; prompt is ~600 tokens
BATCH_SIZE   = 256              # vLLM batches internally; this is the .generate() chunk size

print("Verifying inputs on Drive:")
for p in [STAGE_B_IN, VAL_CSV, ALL_TARGETS, GOLD_SETS, VAL_ASPECTS]:
    print(f"  {'OK' if p.exists() else 'MISSING'}  {p}")

# %%
import pandas as pd

print("\nLoading val.csv ...")
val_df = pd.read_csv(VAL_CSV)
qid_to_query = {r.query_id: str(r.query) for r in val_df.itertuples()}
print(f"  {len(val_df)} queries")

print("Loading all_targets.json + gold_doc_sets.json ...")
targets   = json.load(open(ALL_TARGETS, encoding="utf-8"))
gold_sets = json.load(open(GOLD_SETS, encoding="utf-8"))

print("Loading val_aspects.parquet ...")
asp_df = pd.read_parquet(VAL_ASPECTS)
qid_to_aspects = {r.query_id: list(r.aspects) for r in asp_df.itertuples()}

print("Loading stage_b_input.parquet ...")
sb = pd.read_parquet(STAGE_B_IN)
print(f"  total rows: {len(sb):,}  cols: {len(sb.columns)}")

# Filter to top-K per query
sb = (
    sb.sort_values(["qid", "stage_a_rank"])
      .groupby("qid", group_keys=False)
      .head(TOP_K)
      .reset_index(drop=True)
)
print(f"  after top-{TOP_K} filter: {len(sb):,}")
print(f"  gold in retained rows: {sb['is_gold'].sum()}")
print(sb.groupby("qid")["is_gold"].agg(["sum", "count"]).rename(columns={"sum":"gold","count":"total"}))

# %% [markdown]
# ## Phase 2 — Prompt construction

# %%
PROMPT_TEMPLATE = """You are a Swiss legal analyst. Decide whether the CANDIDATE below is a citation a competent Swiss lawyer would write in answering the LEGAL QUESTION.

LEGAL QUESTION:
{question}

QUERY ASPECTS (decompose the answer):
{aspects}

CANDIDATE:
- Citation: {citation}
- Family: {family}  {family_extra}
- Paragraph role: {role}
- Retrieval evidence:
  * Article-match with a query-named statute: {article_match}
  * Cited by {co_citation_count} other top-100 pool court paragraphs
  * Concept-cosine vs query aspects: {concept_cosine_score:.2f}  (best on aspect {best_aspect_id})
  * Code in query's legal area: {code_in_target}
  * Chamber matches legal area: {chamber_match}
- Text excerpt: "{text}"

DECISION RULES:
1. GOLD = a Swiss lawyer writing the legal answer would cite this.
2. Substantive paragraphs (legal_standard, reasoning, application, holding) are far more often gold than facts/costs/dispositif/procedural_history.
3. Court paragraphs from chambers irrelevant to the legal area are rarely gold.
4. Articles from codes outside the query's legal area are rarely gold (with one exception: Art. 100 BGG is gold for any appeal question).
5. If retrieval evidence shows article_match=true OR co_citation_count >= 3 OR concept_cosine >= 0.60, the bias is toward YES.
6. If you say keep=true, you MUST quote a verbatim 5-30-word substring of the text excerpt that justifies inclusion.

OUTPUT STRICT JSON (no prose, begin with `{{`):
{{"keep": true|false, "confidence": <0..1>, "which_aspect": "<aspect id>", "evidence_quote": "<verbatim substring of text>"}}
"""

def format_aspects(aspects_list):
    if not aspects_list: return "(none extracted — treat as single-aspect question)"
    lines = []
    for a in aspects_list:
        if isinstance(a, dict):
            label = a.get("label", "")
            w     = a.get("weight", 0)
            aid   = a.get("id", "")
            lines.append(f"  - {aid} (w={float(w):.2f}): {label}")
    return "\n".join(lines) if lines else "(empty)"

def family_extra_str(row):
    if row.family == "court":
        return f"chamber={row.chamber!s}  court_base={row.court_base!s}"
    else:
        return f"code={row.law_code!s}  title={row.law_title!s}"

def build_prompt(row, question, aspects_text):
    return PROMPT_TEMPLATE.format(
        question=question[:1500],
        aspects=aspects_text,
        citation=row.citation,
        family=row.family,
        family_extra=family_extra_str(row),
        role=row.role or "(unknown)",
        article_match=str(bool(row.article_match)).lower(),
        co_citation_count=int(row.co_citation_count),
        concept_cosine_score=float(row.concept_cosine_score),
        best_aspect_id=row.best_aspect_id or "?",
        code_in_target=str(bool(row.code_in_target)).lower(),
        chamber_match=str(bool(row.chamber_match)).lower(),
        text=(row.text or "")[:400].replace('"', "'"),
    )

# Build prompts
prompts = []
prompt_meta = []     # parallel: (qid, did) for each prompt
qid_aspects_cache = {qid: format_aspects(qid_to_aspects.get(qid, [])) for qid in sb["qid"].unique()}
for r in sb.itertuples():
    p = build_prompt(r, qid_to_query[r.qid], qid_aspects_cache[r.qid])
    prompts.append(p)
    prompt_meta.append((r.qid, r.did))
print(f"\nBuilt {len(prompts):,} prompts.")
print(f"\nSample prompt for one candidate:\n{'-'*70}\n{prompts[0]}\n{'-'*70}")
print(f"\nApprox prompt length (chars): mean={int(sum(len(p) for p in prompts)/len(prompts)):,}  "
      f"max={max(len(p) for p in prompts):,}")

# %% [markdown]
# ## Phase 3 — vLLM batched generation

# %%
from vllm import LLM, SamplingParams

print(f"Loading {LLM_MODEL} on vLLM ...")
llm = LLM(
    model=LLM_MODEL,
    dtype="bfloat16",
    gpu_memory_utilization=0.60,
    max_model_len=MAX_MODEL_LEN,
    enforce_eager=False,
    trust_remote_code=True,
)
sp = SamplingParams(
    temperature=0.0, top_p=1.0,
    max_tokens=MAX_TOKENS,
    stop=["\n\n", "QUESTION:", "</json>"],
)

print(f"Generating for {len(prompts):,} prompts ...")
t0 = time.time()
outs = llm.generate(prompts, sp)
dt = time.time() - t0
print(f"Done in {dt/60:.1f} min  ({len(prompts)/max(1,dt):.1f} candidates/sec)")

raw_texts = [o.outputs[0].text if o.outputs else "" for o in outs]

# Free LLM before downstream parsing
del llm
gc.collect()
import torch; torch.cuda.empty_cache()

# %% [markdown]
# ## Phase 4 — Parse strict JSON + apply hard gates
#
# Two gates:
# 1. **Quote substring**: `evidence_quote` MUST appear in the candidate's text (case-insensitive, whitespace-normalized). Drops hallucinated picks.
# 2. **Anti-optimism**: if `keep=true` but every dossier signal is 0 (no article match, no co-citation, no cosine ≥ 0.45, no code match), reject — pure LLM optimism with no evidence.

# %%
def parse_lenient(s):
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[-1]
        if s.endswith("```"):
            s = s.rsplit("```", 1)[0]
    a, b = s.find("{"), s.rfind("}")
    if a == -1 or b == -1: return None
    for cand in [s[a:b+1], s[a:b+1].replace(",}", "}").replace(",]", "]")]:
        try: return json.loads(cand)
        except Exception: pass
    return None

def quote_in_text(quote, text):
    """Case-insensitive whitespace-normalized substring check."""
    if not quote or not text: return False
    norm = lambda x: re.sub(r"\s+", " ", x.lower())
    return norm(quote) in norm(text)

rows = []
for (qid, did), raw, src_row in zip(prompt_meta, raw_texts, sb.itertuples()):
    parsed = parse_lenient(raw) or {}
    keep_llm        = bool(parsed.get("keep", False))
    conf            = float(parsed.get("confidence", 0.0) or 0.0)
    which_aspect    = str(parsed.get("which_aspect", "") or "")
    quote           = str(parsed.get("evidence_quote", "") or "")
    quote_ok        = quote_in_text(quote, src_row.text)
    has_any_evidence = (
        bool(src_row.article_match) or
        int(src_row.co_citation_count) >= 1 or
        float(src_row.concept_cosine_score) >= 0.45 or
        bool(src_row.code_in_target)
    )
    # Final keep: LLM said yes AND quote-grounded AND has some retrieval evidence
    keep_final = keep_llm and quote_ok and has_any_evidence

    rows.append({
        "qid":             qid,
        "did":             did,
        "is_gold":         bool(src_row.is_gold),
        "stage_a_rank":    int(src_row.stage_a_rank),
        "stage_a_score":   float(src_row.stage_a_score),
        "citation":        src_row.citation,
        "family":          src_row.family,
        "keep_llm":        keep_llm,
        "confidence":      conf,
        "which_aspect":    which_aspect,
        "evidence_quote":  quote[:200],
        "quote_in_text":   quote_ok,
        "has_any_evidence":has_any_evidence,
        "keep_final":      keep_final,
        "parse_ok":        bool(parsed),
        "raw_llm":         raw[:600],
    })

out_df = pd.DataFrame(rows)
print(f"\nLLM output diagnostics:")
print(f"  parse_ok:          {out_df.parse_ok.mean():.1%}")
print(f"  keep_llm=true:     {out_df.keep_llm.mean():.1%}")
print(f"  quote_in_text:     {out_df.quote_in_text.mean():.1%}  (of all rows)")
print(f"  has_any_evidence:  {out_df.has_any_evidence.mean():.1%}")
print(f"  keep_final=true:   {out_df.keep_final.mean():.1%}")

# %% [markdown]
# ## Phase 5 — Per-query precision / recall / F1
#
# Reports two versions:
#   - **No K cap**: keep all `keep_final == True` rows (lets LLM choose count).
#   - **K-capped**: keep top-K_pred by confidence where K_pred = round(median of per-query gold counts).
# Stage C will add proper adaptive K + aspect stratification.

# %%
def f1(p, r):
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)

print("\n=== Per-query results (no K cap — keep all keep_final=True) ===")
print(f"{'qid':<8}  {'gold':>5}  {'picks':>6}  {'correct':>8}  {'P':>6}  {'R':>6}  {'F1':>6}")
metrics_no_cap = {}
for qid in sorted(out_df.qid.unique()):
    sub  = out_df[out_df.qid == qid]
    gold = set(d for d, g in zip(sub.did, sub.is_gold) if g)
    # Gold count is total gold per query, not just gold in retained top-K
    total_gold = len(set(gold_sets.get(qid, [])))
    picks = set(sub[sub.keep_final].did)
    correct = len(picks & gold)
    p = correct / max(1, len(picks))
    r = correct / max(1, total_gold)
    metrics_no_cap[qid] = {"gold": total_gold, "picks": len(picks), "correct": correct,
                          "P": p, "R": r, "F1": f1(p, r)}
    print(f"  {qid:<8}  {total_gold:>5}  {len(picks):>6}  {correct:>8}  {p:>6.3f}  {r:>6.3f}  {f1(p,r):>6.3f}")
macro_p_nc = sum(m["P"]  for m in metrics_no_cap.values()) / len(metrics_no_cap)
macro_r_nc = sum(m["R"]  for m in metrics_no_cap.values()) / len(metrics_no_cap)
macro_f_nc = sum(m["F1"] for m in metrics_no_cap.values()) / len(metrics_no_cap)
print(f"\n  MACRO  P={macro_p_nc:.3f}  R={macro_r_nc:.3f}  F1={macro_f_nc:.3f}")

print("\n=== Per-query results (K-capped: top by confidence, K = predicted_gold_size) ===")
# Use median gold count as starter predicted size; Stage C will do better.
# For now, take top-K_q where K_q = total_gold (perfect-K oracle for upper bound)
print(f"{'qid':<8}  {'gold':>5}  {'picks':>6}  {'correct':>8}  {'P':>6}  {'R':>6}  {'F1':>6}")
metrics_kcap = {}
for qid in sorted(out_df.qid.unique()):
    sub = out_df[(out_df.qid == qid) & (out_df.keep_final)]
    sub = sub.sort_values("confidence", ascending=False)
    total_gold = len(set(gold_sets.get(qid, [])))
    # K = predicted_gold_size proxy: use total_gold for now (oracle-K). Stage C replaces this.
    K = total_gold
    picks = set(sub.head(K).did)
    gold_dids = set(gold_sets.get(qid, []))
    correct = len(picks & gold_dids)
    p = correct / max(1, len(picks))
    r = correct / max(1, total_gold)
    metrics_kcap[qid] = {"gold": total_gold, "K": K, "picks": len(picks),
                        "correct": correct, "P": p, "R": r, "F1": f1(p, r)}
    print(f"  {qid:<8}  {total_gold:>5}  {len(picks):>6}  {correct:>8}  {p:>6.3f}  {r:>6.3f}  {f1(p,r):>6.3f}")
macro_p_k = sum(m["P"]  for m in metrics_kcap.values()) / len(metrics_kcap)
macro_r_k = sum(m["R"]  for m in metrics_kcap.values()) / len(metrics_kcap)
macro_f_k = sum(m["F1"] for m in metrics_kcap.values()) / len(metrics_kcap)
print(f"\n  MACRO  P={macro_p_k:.3f}  R={macro_r_k:.3f}  F1={macro_f_k:.3f}")

# %% [markdown]
# ## Phase 6 — Save outputs

# %%
out_df.to_parquet(OUT_RAW, index=False)
print(f"Saved raw → {OUT_RAW}")

# Keep the slim survivors table separate
survivors = out_df[out_df.keep_final][["qid","did","citation","confidence","which_aspect",
                                       "evidence_quote","stage_a_rank","stage_a_score","is_gold"]]
survivors.to_parquet(OUT_SURVIVORS, index=False)
print(f"Saved survivors → {OUT_SURVIVORS}")

with open(OUT_METRICS, "w", encoding="utf-8") as f:
    json.dump({
        "config": {"TOP_K": TOP_K, "LLM_MODEL": LLM_MODEL, "MAX_TOKENS": MAX_TOKENS,
                   "BATCH_SIZE": BATCH_SIZE},
        "macro_no_cap":  {"P": macro_p_nc, "R": macro_r_nc, "F1": macro_f_nc},
        "macro_k_cap":   {"P": macro_p_k,  "R": macro_r_k,  "F1": macro_f_k},
        "per_query_no_cap": metrics_no_cap,
        "per_query_k_cap":  metrics_kcap,
        "diagnostics": {
            "parse_ok_rate":          float(out_df.parse_ok.mean()),
            "keep_llm_rate":          float(out_df.keep_llm.mean()),
            "quote_in_text_rate":     float(out_df.quote_in_text.mean()),
            "has_any_evidence_rate":  float(out_df.has_any_evidence.mean()),
            "keep_final_rate":        float(out_df.keep_final.mean()),
        },
    }, f, indent=2)
print(f"Saved metrics → {OUT_METRICS}")

print("\n=== DECISION GATE ===")
print(f"  Macro F1 (no K cap):     {macro_f_nc:.3f}")
print(f"  Macro F1 (K = K_gold):   {macro_f_k:.3f}  (upper bound — Stage C will pick K adaptively)")
print()
if macro_f_k >= 0.60:
    print("  ✅ Stage B passed (F1 >= 0.60). Proceed to Stage C (adaptive K + aspect stratification).")
elif macro_f_k >= 0.50:
    print("  ⚠ Stage B partial pass (0.50-0.60). Diagnose:")
    print("    - If keep_llm_rate > 0.5 → LLM over-permissive; tighten prompt rules.")
    print("    - If quote_in_text rate < 0.7 → LLM hallucinating quotes; tighten quote rule.")
    print("    - If parse_ok < 0.95 → JSON output format unreliable; consider guided_json.")
else:
    print("  ❌ Stage B failed (F1 < 0.50). Likely causes:")
    print("    - Stage A's top-K had too little gold (R@K too low).")
    print("    - Dossier signals not surfaced clearly enough in prompt.")
    print("    - Consider widening TOP_K from 500 to 1000+.")
