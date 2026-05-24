"""Diagnostic — five reranker variants on val_001.

Tests the four hypothesized fixes (and a control) in ONE pass. Use this to
verify each lever's individual contribution before committing to the production
configuration.

Variants:
  V0  baseline                 — raw text, original instruction (current Phase 2)
  V1  enriched doc only        — V0 + cit + statutes + concepts + role
  V2  cross-lingual instruction only — V0 with the explicit cross-lingual instruction
  V3  enriched + cross-ling    — both fixes (= the production "Phase 2 fixed")
  V4  V3 + BGE-v2-m3 ensemble  — V3 averaged with BGE-reranker-v2-m3 scores

Reports per-variant: gold_mean, nongold_mean, separation, R@500/1000 on val_001.
Costs: ~6-8 minutes total (V0-V3 each ~1.5 min on 1000 candidates; V4 ~2 min on top-500).

Requires Phase 0/1 already run (i.e., snapshot loaded), and rrk_llm + rrk_tok
already loaded in globals.
"""
import math
import time
import torch

# Settings
TEST_QID = "val_001"
N_CANDS  = 1000
N_BGE    = 300  # only top-N for the BGE ensemble in V4 (cost control)


assert "rrk_llm" in globals() and rrk_llm is not None, \
    "Run Phase 2 (Qwen3 reranker) first so rrk_llm is loaded."
assert TEST_QID in PER_QUERY

qtext = next(q["query_text"] for q in ALL_QUERIES if q["query_id"] == TEST_QID)
pool  = PER_QUERY[TEST_QID]["final_topk"][:N_CANDS]
gold  = ALL_GOLD_DOC_SET[TEST_QID]
print(f"[diagnostic] {TEST_QID}  N={N_CANDS}  in-pool gold={len(set(pool)&gold)}/{len(gold)}")


# ─── Build doc representations ─────────────────────────────────────────────
def _join_short(values, max_items=8, max_chars=240):
    return ", ".join(list(values)[:max_items])[:max_chars]


def repr_raw(did):
    return (search_text.get(did, "") or doc_meta.get(did, {}).get("citation", did) or did)[:3000]


def repr_enriched(did):
    m     = doc_meta.get(did, {})
    cit   = (m.get("citation") or "").strip()
    fam   = m.get("family") or "?"
    role  = m.get("paragraph_role") or ""
    body  = (search_text.get(did, "") or "").strip()
    sa    = sorted(doc_statute_anchors.get(did, set()))
    concs = sorted(_doc_to_concepts.get(did, set()))
    parts = []
    if cit:    parts.append(f"Citation: {cit}")
    parts.append(f"Type: {fam} ({role})" if role else f"Type: {fam}")
    if sa:     parts.append(f"Statutes: {_join_short(sa)}")
    if concs:  parts.append(f"Concepts (English): {_join_short(concs)}")
    parts.append(f"Text: {body[:2000]}")
    return "\n".join(parts)[:3000]


# ─── Build (instruction, doc_repr) ────────────────────────────────────────
INSTR_BASELINE = (
    "Given an English legal question about Swiss federal law, determine "
    "whether the provided document (a Swiss law article OR a court paragraph "
    "in German / French / Italian) is a RELEVANT CITATION - meaning it "
    "answers, supports, or directly relates to the question."
)
INSTR_CROSSLING = (
    "The Query is an English question about Swiss federal law. The Document is "
    "a Swiss federal law article (German) or a Swiss court paragraph (German, "
    "French, or Italian) — it may include English-language metadata (citation, "
    "statutes referenced, legal concepts). Decide YES if the Document is a "
    "relevant citation for the Query — i.e., the Query's legal question can be "
    "answered, supported, or directly discussed by this Document. Decide NO "
    "otherwise. Treat language differences as a translation problem, not a "
    "mismatch."
)

_RRK_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Note that the answer can only be \"yes\" or "
    "\"no\".<|im_end|>\n<|im_start|>user\n"
)
_RRK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_RRK_PREFIX_IDS = rrk_tok.encode(_RRK_PREFIX, add_special_tokens=False)
_RRK_SUFFIX_IDS = rrk_tok.encode(_RRK_SUFFIX, add_special_tokens=False)
_RRK_MAX_BODY   = 1024 - len(_RRK_PREFIX_IDS) - len(_RRK_SUFFIX_IDS) - 8

_RRK_YES_ID = rrk_tok.convert_tokens_to_ids("yes")
_RRK_NO_ID  = rrk_tok.convert_tokens_to_ids("no")


def _score_with(query, dids, instruction, repr_fn):
    """Score (query, did) pairs with given instruction + doc representation."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    body_strs = [
        f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {repr_fn(d)}"
        for d in dids
    ]
    prompt_ids = []
    for bs in body_strs:
        ids = rrk_tok.encode(bs, add_special_tokens=False)
        if len(ids) > _RRK_MAX_BODY: ids = ids[:_RRK_MAX_BODY]
        prompt_ids.append(_RRK_PREFIX_IDS + ids + _RRK_SUFFIX_IDS)
    _prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]
    outs = rrk_llm.generate(_prompts, sampling_params=sp, use_tqdm=False)
    scores = []
    for out in outs:
        lp_step = out.outputs[0].logprobs[0]
        y = lp_step.get(_RRK_YES_ID)
        n = lp_step.get(_RRK_NO_ID)
        yl = y.logprob if y is not None else -1e9
        nl = n.logprob if n is not None else -1e9
        m = max(yl, nl)
        e_y = math.exp(yl - m); e_n = math.exp(nl - m)
        scores.append(e_y / (e_y + e_n))
    return scores


# ─── Run variants ─────────────────────────────────────────────────────────
def _summary(name, scores, dids):
    g_set = gold
    g_s  = [s for d, s in zip(dids, scores) if d in g_set]
    ng_s = [s for d, s in zip(dids, scores) if d not in g_set]
    gmean  = sum(g_s) / max(1, len(g_s))
    ngmean = sum(ng_s) / max(1, len(ng_s))
    ranked = sorted(zip(dids, scores), key=lambda kv: -kv[1])
    g_tot = max(1, len(g_set))
    R500  = len({d for d, _ in ranked[:500]} & g_set) / g_tot
    R1000 = len({d for d, _ in ranked[:1000]} & g_set) / g_tot
    return {"name": name, "gold_mean": gmean, "nongold_mean": ngmean,
            "sep": gmean - ngmean, "R500": R500, "R1000": R1000}


results = []

t0 = time.time()
print("\n[V0] baseline (raw doc, baseline instruction) ...")
v0_scores = _score_with(qtext, pool, INSTR_BASELINE, repr_raw)
results.append(_summary("V0_baseline", v0_scores, pool))
print(f"   done in {time.time()-t0:.0f}s")

t0 = time.time()
print("[V1] enriched doc, baseline instruction ...")
v1_scores = _score_with(qtext, pool, INSTR_BASELINE, repr_enriched)
results.append(_summary("V1_enrich_only", v1_scores, pool))
print(f"   done in {time.time()-t0:.0f}s")

t0 = time.time()
print("[V2] raw doc, cross-lingual instruction ...")
v2_scores = _score_with(qtext, pool, INSTR_CROSSLING, repr_raw)
results.append(_summary("V2_instr_only", v2_scores, pool))
print(f"   done in {time.time()-t0:.0f}s")

t0 = time.time()
print("[V3] enriched doc + cross-lingual instruction (= production fix) ...")
v3_scores = _score_with(qtext, pool, INSTR_CROSSLING, repr_enriched)
results.append(_summary("V3_enrich+instr", v3_scores, pool))
print(f"   done in {time.time()-t0:.0f}s")

# ─── V4: V3 + BGE-v2-m3 ensemble (top-300 only for cost) ─────────────────
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if "bge_mod" not in globals():
        print("[V4] loading BGE-reranker-v2-m3 (~1.5 GB) ...")
        _t = time.time()
        bge_tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        bge_mod = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-v2-m3", dtype=torch.bfloat16
        ).cuda().eval()
        print(f"   loaded in {time.time()-_t:.0f}s")
    t0 = time.time()
    print(f"[V4] V3 + BGE ensemble on top-{N_BGE} ...")
    # Take V3's top-N for the BGE pass
    v3_ranked = sorted(zip(pool, v3_scores), key=lambda kv: -kv[1])
    bge_top = [d for d, _ in v3_ranked[:N_BGE]]
    bge_pairs = [(qtext, repr_enriched(d)) for d in bge_top]
    bge_scores = []
    BS = 16
    for i in range(0, len(bge_pairs), BS):
        with torch.no_grad():
            inp = bge_tok(bge_pairs[i:i+BS], padding=True, truncation=True,
                          max_length=1024, return_tensors="pt")
            inp = {k: v.cuda() for k, v in inp.items()}
            logits = bge_mod(**inp).logits.view(-1).float()
            bge_scores.extend(torch.sigmoid(logits).cpu().tolist())
    bge_score_map = dict(zip(bge_top, bge_scores))
    v3_score_map  = dict(zip(pool, v3_scores))
    # Ensemble on top-N only; below that, Qwen3-only
    v4_scores = []
    for d in pool:
        if d in bge_score_map:
            v4_scores.append(0.5 * v3_score_map[d] + 0.5 * bge_score_map[d])
        else:
            v4_scores.append(0.5 * v3_score_map[d])  # downweighted: only one vote
    results.append(_summary("V4_ensemble", v4_scores, pool))
    print(f"   done in {time.time()-t0:.0f}s")
except Exception as e:
    print(f"[V4] SKIPPED: {e}")


# ─── Report ────────────────────────────────────────────────────────────────
print()
print("=" * 96)
print(f"  RERANKER FIX DIAGNOSTIC — {TEST_QID}  (N_CANDS={N_CANDS}, gold in pool={len(set(pool)&gold)})")
print("=" * 96)
print(f"  {'variant':<22}{'gold_mean':>11}{'ng_mean':>10}{'sep':>9}"
      f"{'R@500':>9}{'R@1000':>9}{'d_sep_vs_V0':>+13}{'d_R1k_vs_V0':>+13}")
v0 = results[0]
for r in results:
    print(f"  {r['name']:<22}{r['gold_mean']:>11.4f}{r['nongold_mean']:>10.4f}"
          f"{r['sep']:>+9.4f}{r['R500']:>9.3f}{r['R1000']:>9.3f}"
          f"{r['sep']-v0['sep']:>+13.4f}{r['R1000']-v0['R1000']:>+13.3f}")
print()
# Quick interpretation
best = max(results, key=lambda r: r["R1000"])
print(f"  Best R@1000: {best['name']}  ({best['R1000']:.3f})")
print(f"  Δ vs baseline: sep {best['sep']-v0['sep']:+.4f},  R@1000 {best['R1000']-v0['R1000']:+.3f}")
