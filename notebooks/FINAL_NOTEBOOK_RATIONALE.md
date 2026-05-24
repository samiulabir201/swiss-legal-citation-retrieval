# Final Notebook Rationale — Swiss Citation Extraction (2026-05-13)

A consolidated assessment of what worked / didn't in the existing pipeline and the evidence base for the final notebook's design.

---

## 1. What worked in `swiss_citation_anchor_funnel_v7_5_multiquery (10).ipynb`

The current notebook is the **retriever** half — it produces the 50k-candidate pool and saves a snapshot. The agents verified:

| Component | Result | Evidence |
|---|---|---|
| 14-channel RRF fusion (statute_backprop + concept_en + court_statute + per_area_bedrock + sibling_expansion + co_citation + graph_{forward,reverse,2hop} + law_direct_match + term_orig + bm25 + vector_{raw,enriched,hyde}) | **Macro R@5000 = 0.728, R@2000 = 0.611, R@50000 ≈ 0.89-0.90** | `research/stage_b_experiments/REPORT.md` |
| v7.5 budget lifts + 7-channel guarantee (130 each) + code-family expansion (top-8 via corpus-derived `code_pair_count`) | Raised macro R@50k from ~0.7 (v7.4) to **0.89** | `_v7_5_apply.py`, `_warmboot.py` |
| Role-gate (drops cantonal court, notification/costs/dispositif paragraphs) | Confirmed: 102/102 val gold court paragraphs are federal | `gold_citation_legal_patterns_2026-05-12.md` |
| Manual Stage 1 cell — **Qwen3-Reranker-8B with `softmax([logit_no, logit_yes])[:,1]`** scoring top-5000 from snapshot's `final_topk` | Already correct per Qwen3 TR canonical: uses `convert_tokens_to_ids("yes")`, `padding_side='left'`, `<think>\n\n</think>` suffix | Cell visible in current notebook; matches `Qwen3-Embedding/examples/qwen3_reranker_transformers.py` |
| Sanity check (gold pair score > non-gold pair score before full run) | Catches token-ID bugs that destroyed the prior endgame run | Cell in current notebook |

**Per-query gold-in-pool ceilings** (verified from `per_query_snapshot.json`):

| qid | gold_total | gold_in_pool (of 50k) | R_max |
|---|---:|---:|---:|
| val_001 | 42 | 39 | 0.929 |
| val_002 | 38 | 30 | 0.789 |
| val_003 | 47 | 36 | **0.766** (binding) |
| val_004 | 10 | 10 | 1.000 |
| val_005 | 11 | 11 | 1.000 |
| val_006 | 18 | 17 | 0.944 |
| val_007 | 19 | 17 | 0.895 |
| val_008 | 30 | 25 | 0.833 |
| val_009 | 14 | 13 | 0.929 |
| val_010 | 25 | 22 | 0.880 |

**Macro oracle ceiling** (if every in-pool gold is selected with `K = in_pool_gold_count`): macro F1 ≈ **0.944**. So F1 0.7-0.8 is achievable — it's a selection problem, not a retrieval problem.

---

## 2. What did NOT work

| Failure | Where | Why |
|---|---|---|
| **Dossier-only ranking (no fusion)** | Stage B's original composite scorer | Achieved R@300 = 0.013 because most val gold has ZERO statute-target intersection (LLM expands 10-15 statutes; gold spans 40+ across val_003). Most gold has NO match against the canonical (`Art. NUM CODE`) statute target list. |
| **All 12 CPU-only Stage B strategies** | `stage_b_experiments/REPORT.md` | Best macro R@2000 = 0.644 (+0.033 over fusion baseline), min per-query R@2000 = 0.255 for val_003 across every strategy. Structural ceiling, not algorithmic. |
| **Old endgame run** (Qwen3-Reranker-8B + Qwen3-8B judge on a smaller candidate pool) | `swiss_citation_endgame_colab.ipynb` | F1 < 0.10 because rerank pool recall was 23.1%. Pool-recall ceiling dominates the cascade output. |
| **Fixed top-K = 5 (COLIEE 2025 style)** | UQLegalAI@COLIEE2025 paper, 2nd place | Recall capped at K=5 lost gold in queries with >5 true positives. Their F1 = 0.296. Lesson: **never hard-cap K**. |
| **Cohere reranker on legal text** | LegalBench-RAG paper | Reranker hurt vs no-reranker on niche legal data. Lesson: validate per-domain. (Qwen3-Reranker is trained more broadly, so this risk is lower for us — but still validate.) |
| **HyDE alone** | 2026 RAG benchmarks | Often UNDERPERFORMS dense retrieval (Recall@5 0.544 vs 0.587). HyDE is useful as a query-expansion signal, not as primary retrieval. |
| **Train-derived priors** | User memory `feedback_train_unreliable.md` | Train has 3-axis distribution shift: language (99% DE → 100% EN), citation count (median 2 → 22), court share (1.2% → 40.6% federal). Anything baked from train statistics doesn't transfer. |

---

## 3. Evidence base from 2025-2026 papers (in `research_papers/`)

| Paper | Key finding | What we use |
|---|---|---|
| **Qwen3-Embedding TR (June 2025)** | Score = `softmax([P(no), P(yes)])[:,1]` is already a calibrated probability — no additional sigmoid / Platt needed | Stage 1 scoring (current implementation correct) |
| **Adaptive-k (EMNLP 2025)** | Largest score gap in sorted descending scores + buffer B=5. Search window = top 90%. Pure function of score distribution. | Variable-K estimator 1 |
| **CAR (Nov 2025)** | Cluster (score, rank_index) pairs; cut at silhouette-best clustering's biggest gap, weighted by rank_index/N. | Variable-K estimator 2 |
| **PSI-Rank (2026)** | LLM generates a "borderline" pseudo-doc; cut at its rerank score. Each query gets its own pivot. | Variable-K estimator 3 |
| **AcuRank (NeurIPS 2025)** | TrueSkill `Rating(mu=score, sigma=score/3)`. Stop when uncertain pool (`tol < P(>t) < 1-tol`) drops below U=10. | Variable-K estimator 4 |
| **Two-Stage Risk Control (IJCAI 2025)** | Conformal calibration on held-out set: `λ̃ = inf{λ : Σ L⁽¹⁾(λ) ≤ (|I|+1)α−1}` gives distribution-free 1−α recall guarantee. | Variable-K estimator 5 (uses train.csv as calibration set — quantile thresholds only, not priors) |
| **Missing Link (DEXA 2025)** | Heterogeneous R-GCN for joint case+law citation, AP 87-88% on German legal graph. Schema: case + law + court + lawbook nodes; case-case + case-law + enrichment edges. | Optional Stage 2.5 graph re-score (deferred to v2 — significant build cost) |
| **From Citations to Criticality (ACL 2025)** | Fine-tuned multilingual encoders beat zero-shot LLMs on Swiss legal classification. Best macro F1 ≈ 37 on a 4-way ordinal — calibrates expectations. | We're doing retrieval, not classification, but the message stands: pure zero-shot LLM judges are not enough; supplement with calibrated cheap signals. |
| **UQLegalAI@COLIEE2025** | 2nd place F1 = 0.296 with CaseLink+GAT. Recall constrained by hard-cap at K=5 ≈ train mean. | Lesson: don't cap K at train mean; use score-distribution. |
| **GRLStop (SIGIR 2025)** | RL stopping for TAR; needs labels at inference. Not directly applicable to our offline setting. | Skip; deferred to future work. |
| **Talos (Jan 2026)** | Training-time loss for per-user K threshold. Requires retraining. Skip — we're not retraining the reranker. | Skip. |

---

## 4. Final notebook architecture

8 phases, no fixed K, no val-specific hardcoding, no train-derived priors.

```
Phase 0  Warm-boot from snapshot                                        CPU,  ~10s
Phase 1  Qwen3-Reranker-8B rerank on top-5000 per query                 GPU,  ~3-5 min × 10 = 30-50 min
              → calibrated probabilities ∈ (0,1) via softmax([no,yes])

Phase 2  Score-distribution diagnostics + sigmoid sanity                CPU,  <1s

Phase 3  SIX independent variable-K estimators (all from score curve)   CPU,  ~5s/query
              3a. Adaptive-k       (largest-gap, top 90%, buffer B=5)
              3b. CAR              (k-means clusters + silhouette + gap·rank score)
              3c. PSI-Rank-Dyn     (Qwen3-32B generates a borderline pseudo-doc;
                                    score with same reranker; cut at pivot)
              3d. AcuRank-light    (TrueSkill posterior; stop when |uncertain| < 10)
              3e. Conformal        (train.csv as cal set → quantile threshold τ_α
                                    for 1-α recall guarantee)
              3f. Aspect-coverage  (each hyde_aspect must be covered by ≥1
                                    candidate with score > 0.5 AND substantive role)

Phase 4  Ensemble + asymmetric clamp                                    CPU
              K̂(q) = round( median(K_3a, K_3b, K_3c, K_3d, K_3e) )
              K_final(q) = clip(
                  K̂,
                  lower = max(K_aspect, n_hyde_aspects, 3),
                  upper = #candidates with p > 0.05
              )

Phase 5  Build final candidate sets per query (top-K_final by rerank)   CPU,  <1s

Phase 6  Optional precision pass: Qwen3-32B evidence-quote verifier     GPU,  ~2 min/query
              For each finalist, require a verbatim substring of doc text
              that supports relevance. Replace, don't drop, on failure
              (preserves K).

Phase 7  Evaluation: macro / micro P / R / F1 vs val gold + per-query   CPU
              breakdown + ablation table (Adaptive-k only, CAR only,
              ensemble) + fusion-baseline comparison at same K
```

### Why no hardcoding

- Stages 3a-3e read **only the rerank score distribution** — pure functions of sorted scores
- Stage 3f reads **hyde_aspects[qid]** + candidate doc text — query-driven, doc-driven, no statute priors
- Stage 3e (Conformal) uses **train.csv** as a calibration set — but it only learns a quantile threshold τ_α, not which Swiss statutes are popular. The threshold transfers as a probability cutoff.
- Stage 4 ensembling is **median**, not weighted-by-train-performance

### Per-query F1 cost asymmetry (the F1 lever)

For perfect-ranking F1 with gold count g:
- predict K=g-3 on a g=5 query → F1 = 2·2/(2+5) ≈ 0.57
- predict K=g+3 on a g=30 query → F1 = 2·30/(30+33) ≈ 0.95

**Under-predicting is fatal for small-g queries; over-predicting is forgiving for large-g queries.** The lower-clamp at `max(K_aspect, 3)` enforces this.

### Estimated F1 contribution per stage

(Based on per-stage gains reported in source papers, scaled to our pool's 0.89 macro recall)

| Phase | Expected macro F1 contribution | Confidence |
|---|---|---|
| Phase 1 (Qwen3-Reranker-8B rerank) | +0.40 over fusion order baseline | Very high — direct evidence from MMTEB-R 72.94 |
| Phase 3 ensemble variable-K | +0.10-0.20 over fixed K=mean(gold)=22 | High — converged 2025-2026 result |
| Phase 3f aspect-coverage (lower clamp) | +0.05 by protecting small-g queries | Medium-high |
| Phase 6 evidence-quote verifier | +0.03-0.07 via precision swap | Medium (already validated in precision_v1) |

Target macro F1: **0.60-0.78** (probably mid-range; val_003 ceiling at 0.86 caps the macro). The Untitled75 reference of 0.777 used a tuned 7-category judge — that's the realistic upper bound we're aiming for.

---

Created 2026-05-13 from agent reports + 2025-2026 papers. See INDEX.md in `research_papers/` for paper-level details.
