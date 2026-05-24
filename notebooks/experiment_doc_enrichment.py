"""Experiment: which doc representation makes Qwen3-Reranker-8B work on Swiss legal?

Run this as a cell in your Colab session AFTER Phase 1 (warm-boot) and Phase 2
(vLLM rerank engine loaded). It uses `rrk_llm`, `_rrk_score_pairs`, and the
globals already populated.

Compares 4 doc representations on val_001 (the documented failure case):

  P1 baseline      — search_text[:3000]                           (current)
  P2 cit-prepended — citation + search_text[:2800]
  P3 cross-bridge  — citation + english concepts + search_text[:2500]
  P4 SAC-enriched  — citation + family/role + statutes + concepts + terms + search_text[:2000]

For each: gold_mean, nongold_mean, sep, R@500, R@2000 on the top-5000 fusion pool.

Hypothesis: P3 (cross-bridge) and P4 (SAC-enriched) should lift gold_mean by
exposing English concept tokens that match the English query AND the
canonical statute anchors that match the query's statute mentions.

References:
  - Anthropic Contextual Retrieval (Sept 2024) — metadata cuts failures 49%
  - "Towards Reliable Retrieval in RAG Systems for Large Legal Datasets" (NLLP 2025)
  - "Best Reranker for Cross-Lingual Search 2026" — modality gap evidence
"""

import time

# Pick the test query
TEST_QID = "val_001"
N_CANDS  = 1000   # top-1000 of fusion pool; ~5x the size needed for a stable separation estimate

assert TEST_QID in PER_QUERY, f"{TEST_QID} not loaded — run warm-boot first"
qtext = next(q["query_text"] for q in ALL_QUERIES if q["query_id"] == TEST_QID)
pool  = PER_QUERY[TEST_QID]["final_topk"][:N_CANDS]
gold  = ALL_GOLD_DOC_SET[TEST_QID]

print(f"[experiment] query: {TEST_QID}  (gold={len(gold)}, in pool={len(set(pool)&gold)})")
print(f"[experiment] testing {len(pool)} candidates × 4 patterns = {4*len(pool)} pairs")


def _join_short(values, max_items=5, max_chars=200):
    out = ", ".join(list(values)[:max_items])
    return out[:max_chars]


def repr_p1_baseline(did):
    """Current behavior — raw text only."""
    text = search_text.get(did, "") or doc_meta.get(did, {}).get("citation", did) or did
    return text[:3000]


def repr_p2_cit_prepend(did):
    """Citation explicitly prepended."""
    cit  = (doc_meta.get(did, {}).get("citation") or "").strip()
    body = (search_text.get(did, "") or "").strip()
    if body and cit and not body.startswith(cit):
        return f"{cit}\n{body}"[:3000]
    return (body or cit or did)[:3000]


def repr_p3_cross_bridge(did):
    """Cross-lingual bridge — English concepts + raw text. The point: query is
    English, doc text is German/French/Italian; concepts_en are English tokens
    drawn from the doc's own enrichment, acting as bridge tokens."""
    cit   = (doc_meta.get(did, {}).get("citation") or "").strip()
    body  = (search_text.get(did, "") or "").strip()
    concs = sorted(_doc_to_concepts.get(did, set()))
    parts = []
    if cit: parts.append(f"Citation: {cit}")
    if concs: parts.append(f"Concepts (English): {_join_short(concs, max_items=8)}")
    parts.append(f"Text: {body[:2500]}")
    return "\n".join(parts)[:3000]


def repr_p4_sac_enriched(did):
    """SAC-style full enrichment — every field the corpus has on this doc."""
    m       = doc_meta.get(did, {})
    cit     = (m.get("citation") or "").strip()
    family  = m.get("family") or "?"
    role    = m.get("paragraph_role") or ""
    cb      = m.get("court_base") or ""
    lang    = m.get("language") or "?"
    body    = (search_text.get(did, "") or "").strip()
    sa      = sorted(doc_statute_anchors.get(did, set()))
    concs   = sorted(_doc_to_concepts.get(did, set()))
    terms   = sorted(_doc_to_terms.get(did, set()))

    parts = []
    if cit:    parts.append(f"Citation: {cit}")
    type_str = family
    if role:   type_str = f"{family} ({role})"
    parts.append(f"Type: {type_str}")
    if cb and family == "court":
        parts.append(f"Court: {cb}")
    parts.append(f"Language: {lang}")
    if sa:     parts.append(f"Statute anchors: {_join_short(sa, max_items=6)}")
    if concs:  parts.append(f"Concepts (English): {_join_short(concs, max_items=8)}")
    if terms:  parts.append(f"Terms: {_join_short(terms, max_items=8)}")
    parts.append(f"Text: {body[:2000]}")
    return "\n".join(parts)[:3000]


REPRS = {
    "P1_baseline":     repr_p1_baseline,
    "P2_cit_prepend":  repr_p2_cit_prepend,
    "P3_cross_bridge": repr_p3_cross_bridge,
    "P4_sac_enriched": repr_p4_sac_enriched,
}


# === score each pattern ===
results = {}
for name, fn in REPRS.items():
    print(f"\n[experiment] scoring {name} ...")
    t0 = time.time()
    pairs = [(qtext, fn(d)) for d in pool]
    scores = _rrk_score_pairs(pairs)
    print(f"  scored {len(scores)} pairs in {time.time()-t0:.1f}s")

    # Compute metrics
    g_scores  = [s for d, s in zip(pool, scores) if d in gold]
    ng_scores = [s for d, s in zip(pool, scores) if d not in gold]
    gmean = sum(g_scores) / max(1, len(g_scores))
    ngmean = sum(ng_scores) / max(1, len(ng_scores))
    # R@K from this pattern's ranking
    ranked = sorted(zip(pool, scores), key=lambda t: -t[1])
    g_set = gold
    g_in_pool = len(set(pool) & g_set)
    results[name] = {
        "gold_mean":  gmean,
        "nongold_mean": ngmean,
        "sep": gmean - ngmean,
        "n_gold_scored": len(g_scores),
        "R_at_500":   len({d for d, _ in ranked[:500]} & g_set) / max(1, len(g_set)),
        "R_at_1000":  len({d for d, _ in ranked[:1000]} & g_set) / max(1, len(g_set)),
        "top_K_gold_in_pool_ceil": g_in_pool / max(1, len(g_set)),
    }


# === Report ===
print()
print("=" * 90)
print(f"  RERANKER DOC-REPRESENTATION EXPERIMENT — {TEST_QID}")
print(f"  candidates: top-{N_CANDS} of fusion pool;  in-pool gold: "
      f"{len(set(pool) & gold)}/{len(gold)}")
print("=" * 90)
print(f"  {'pattern':<18} {'gold_mean':>10} {'ng_mean':>10} {'sep':>8} "
      f"{'R@500':>8} {'R@1000':>8}")
for name, r in results.items():
    print(f"  {name:<18} {r['gold_mean']:>10.4f} {r['nongold_mean']:>10.4f} "
          f"{r['sep']:>+8.4f} {r['R_at_500']:>8.3f} {r['R_at_1000']:>8.3f}")
print()
# Interpretation hint
best_sep = max(results.values(), key=lambda r: r["sep"])
best_R   = max(results.values(), key=lambda r: r["R_at_1000"])
best_sep_name = [k for k, v in results.items() if v == best_sep][0]
best_R_name   = [k for k, v in results.items() if v == best_R][0]
print(f"  best separation: {best_sep_name}  ({best_sep['sep']:+.4f})")
print(f"  best R@1000:     {best_R_name}    ({best_R['R_at_1000']:.3f})")
print()
print("  Interpretation guide:")
print("  - If P3 or P4 lifts gold_mean by > 0.10 over P1: cross-lingual gap is the cause")
print("  - If P2 alone lifts: pure citation prefix is the issue; metadata isn't needed")
print("  - If no variant lifts: reranker may not be the bottleneck — consider")
print("    switching to bge-reranker-v2-m3 or jina-reranker-v2-multilingual,")
print("    or fine-tuning Qwen3-Reranker-0.6B on Swiss legal pairs (GPL).")
