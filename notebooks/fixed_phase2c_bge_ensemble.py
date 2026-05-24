"""Phase 2c - BGE-reranker-v2-m3 ensemble (cross-lingual hard-negative-trained backbone).

OPTIONAL but high-leverage. Run AFTER the fixed Phase 2. Combines Qwen3-Reranker-8B
scores (already computed in PER_QUERY[qid]["stage1_ranked"]) with BGE-reranker-v2-m3
scores via mean — gives a second opinion from a model that was trained with hard
negatives across 100+ languages.

Evidence base:
  - "What Drives Cross-lingual Ranking?" arXiv 2511.19324 Nov 2025:
    hard-negative training takes XLM-R from 16.1% → 83.4% Recall@10 on mMARCO
  - BGE-reranker-v2-m3 explicitly trained with hard negatives + multilingual data
  - At 568M params, runs in ~1.5GB VRAM alongside the loaded Qwen3-Reranker-8B
  - Ensemble (mean of two calibrated probabilities) is the standard 2025 hardening
    pattern (Anthropic Contextual Retrieval + reranker combo, +67% failures cut)

Score:
   final_rerank_score(d) = 0.5 * qwen3_P_yes(d) + 0.5 * bge_score(d)

Then PER_QUERY[qid]["stage1_ranked"] is OVERWRITTEN with the ensemble order.
The raw Qwen3 ranking is preserved at PER_QUERY[qid]["stage1_ranked_qwen3"].

Cost: ~3-5 minutes per query on top-500 (BGE has 8k context, not 1k like Qwen3,
so we restrict to top-500 of Qwen3's ranking — that's where the precision matters).
"""
import time
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BGE_MODEL = "BAAI/bge-reranker-v2-m3"
BGE_MAX_LEN = 1024     # match Qwen3's context for fair comparison
BGE_BATCH   = 16
BGE_TOP_N   = 500      # only re-score Qwen3's top-N; deep candidates are settled
ENSEMBLE_W_QWEN = 0.5
ENSEMBLE_W_BGE  = 0.5

# Load BGE-reranker-v2-m3 (load once)
if "bge_mod" not in globals():
    print(f"[bge] loading {BGE_MODEL} (~570M params, ~1.5 GB bf16) ...")
    _t0 = time.time()
    bge_tok = AutoTokenizer.from_pretrained(BGE_MODEL)
    bge_mod = AutoModelForSequenceClassification.from_pretrained(
        BGE_MODEL, dtype=torch.bfloat16
    ).cuda().eval()
    print(f"[bge] loaded in {time.time()-_t0:.1f}s")
else:
    print(f"[bge] reusing existing BGE engine")


def _bge_score(pairs, max_len=BGE_MAX_LEN, batch_size=BGE_BATCH):
    """Score (query, doc_text) pairs via BGE. Returns list[float] in [0,1]."""
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i+batch_size]
        with torch.no_grad():
            inputs = bge_tok(
                [(q, d) for q, d in batch],
                padding=True, truncation=True, max_length=max_len, return_tensors="pt",
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}
            logits = bge_mod(**inputs).logits.view(-1).float()
            # BGE outputs unbounded scores; sigmoid for [0, 1]
            probs = torch.sigmoid(logits).cpu().tolist()
        scores.extend(probs)
    return scores


# === Run BGE on top-N of Qwen3's ranking, ensemble ===
print(f"\n[bge] re-scoring top-{BGE_TOP_N} of Qwen3 ranking with BGE-reranker-v2-m3 ...")
for q in ALL_QUERIES:
    qid   = q["query_id"]
    qtext = q["query_text"]
    qwen3_ranked = PER_QUERY[qid]["stage1_ranked"]
    PER_QUERY[qid]["stage1_ranked_qwen3"] = list(qwen3_ranked)   # backup

    top_n = qwen3_ranked[:BGE_TOP_N]
    qwen3_scores = {d: s for d, s in top_n}

    # Build enriched docs — same enrichment as Qwen3 used for fairness
    pairs = []
    for did, _ in top_n:
        # Reuse the enriched representation from fixed_phase2_rerank.py
        m       = doc_meta.get(did, {})
        cit     = (m.get("citation") or "").strip()
        family  = m.get("family") or "?"
        role    = m.get("paragraph_role") or ""
        body    = (search_text.get(did, "") or "").strip()
        sa      = sorted(doc_statute_anchors.get(did, set()))
        concs   = sorted(_doc_to_concepts.get(did, set()))
        parts = []
        if cit:    parts.append(f"Citation: {cit}")
        parts.append(f"Type: {family} ({role})" if role else f"Type: {family}")
        if sa:     parts.append(f"Statutes: {', '.join(sa[:6])}")
        if concs:  parts.append(f"Concepts: {', '.join(concs[:8])}")
        parts.append(f"Text: {body[:2200]}")
        doc_repr = "\n".join(parts)[:3500]
        pairs.append((qtext, doc_repr))

    t_q = time.time()
    bge_scores_list = _bge_score(pairs)
    bge_scores = dict(zip([d for d, _ in top_n], bge_scores_list))

    # Ensemble: mean of qwen3 P(yes) and bge sigmoid score
    ensemble = []
    seen = set()
    for did, _ in top_n:
        s = ENSEMBLE_W_QWEN * qwen3_scores[did] + ENSEMBLE_W_BGE * bge_scores[did]
        ensemble.append((did, s))
        seen.add(did)
    # Append the rest of Qwen3 ranking (below top-N) with Qwen3 scores only
    for did, s in qwen3_ranked[BGE_TOP_N:]:
        if did not in seen:
            ensemble.append((did, s * ENSEMBLE_W_QWEN))  # downweighted: only one model voted

    ensemble.sort(key=lambda kv: -kv[1])
    PER_QUERY[qid]["stage1_ranked"] = ensemble  # OVERWRITE

    g = ALL_GOLD_DOC_SET[qid]
    g_tot = ALL_TOTAL_GOLD[qid]
    PER_QUERY[qid]["stage1_R_at_K"] = {}
    for K in (50, 100, 200, 500, 1000, 2000, 5000):
        K_use = min(K, len(ensemble))
        hit = len({d for d, _ in ensemble[:K_use]} & g)
        PER_QUERY[qid]["stage1_R_at_K"][K] = (hit, hit / max(1, g_tot))

    # Compare per-query: Qwen3 alone vs Ensemble at R@2000
    qwen3_r2k = sum(1 for d, _ in qwen3_ranked[:2000] if d in g) / max(1, g_tot)
    ens_r2k   = PER_QUERY[qid]["stage1_R_at_K"][2000][1]
    print(f"  [{qid}] {time.time()-t_q:5.1f}s  "
          f"qwen3 R@2k={qwen3_r2k:.3f}  ensemble R@2k={ens_r2k:.3f}  "
          f"Δ={ens_r2k-qwen3_r2k:+.3f}")

print()
print("=" * 80)
print(f"  FIXED + ENSEMBLED vs FUSION (macro mean R@K)")
print("=" * 80)
print(f"  {'K':>6}  {'fusion':>8}  {'ensemble':>10}  {'d':>+8}  {'min':>8}  {'max':>8}")
for K in (50, 100, 200, 500, 1000, 2000, 5000):
    fus = []
    s1  = []
    for qid in PER_QUERY:
        c = PER_QUERY[qid]["curve"]
        keys = sorted(c.keys())
        kk = max((k for k in keys if k <= K), default=None)
        fus.append(c[kk][1] if kk is not None else 0.0)
        s1.append(PER_QUERY[qid]["stage1_R_at_K"][K][1])
    print(f"  {K:>6}  {sum(fus)/len(fus):>8.3f}  {sum(s1)/len(s1):>10.3f}  "
          f"{(sum(s1)-sum(fus))/len(fus):>+8.3f}  {min(s1):>8.3f}  {max(s1):>8.3f}")
