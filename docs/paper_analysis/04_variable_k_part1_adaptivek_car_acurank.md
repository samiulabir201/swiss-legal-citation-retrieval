# Variable-K Truncation — Part 1: AdaptiveK / CAR / AcuRank

**Scope:** three papers from `research_papers/variable_k_truncation/` that propose **per-query K-selection** (variable retrieval depth or variable rerank cutoff) by reading the *shape* of the score distribution rather than picking a fixed K.

**Why this matters for this project.** Val gold size ranges 10–47 (median 22). `ROADMAP.md` lists "Fixed K prediction" as a banned move and identifies **Step 3 — Calibrate K per query** as the open work that unlocks the macro-F1 0.6–0.8 target. The current variable-K mechanism (precision-v1 Stage E) is driven by *LLM verdict confidence* + *aspect coverage* + *case-dedup*, not by score-distribution geometry or uncertainty over relevance. So all three papers below describe families of mechanism that this project has **not** instantiated.

Conventions:
- STATUS: `OPEN` (never tried), `PARTIAL` (related mechanism tried, not this one), `TRIED` (this specific algorithm was implemented).
- "Tried" means evidence of the algorithm's signature operation appearing in a project notebook or `v7_*` pipeline cell, **not** just the word "variable K" being present.

---

## Paper 1 — Adaptive-k: Largest Similarity Gap (Taguchi, Maekawa, Bhutani — Megagon Labs, EMNLP 2025)

### Approaches in the paper

#### Approach 1.A — Largest score-gap cut with buffer B
- **Core algorithm:** Compute similarity `s = fsim(q, C)`. Sort descending. Compute first differences `g[i] = s[i] - s[i+1]`. Pick `k* = argmax(g)`. Then return top-`(k* + B)` documents.
- **Key hyperparameters:**
  - Buffer `B = 5` (extra docs beyond the gap, hedge against borderline relevants).
  - Search restricted to the **top 90% of the sorted list** (a "head-only" guard so the algorithm doesn't pick a gap among the tail of irrelevants).
- **Claimed result:**
  - HoloBench (aggregation QA): Adaptive-k achieves ≈70% context recall across info_amount = {5k, 10k, 25k, 50k} tokens (vs. SELF-ROUTE's 21–66% recall). +9 points answer accuracy over SELF-ROUTE on high-information aggregation tasks.
  - HotpotQA / NQ / TriviaQA (factoid QA): matches the *oracle* fixed-n SubEM at **up to 99% input-token reduction** vs full context, and up to 90% vs SELF-ROUTE.
  - "Diff-k" (|k_pred − k_oracle|) closely tracks the true oracle on HoloBench-50k (Adaptive-k ≈ 992 vs oracle fixed-50k ≈ 1680). Note diff-k is large in absolute terms because the contexts are 100k tokens.
- **Code reference:** https://github.com/megagonlabs/adaptive-k-retrieval (mirrored locally at `research_repos/adaptive-k-retrieval/`). Algorithm 1, `solve.py`.
- **Caveats called out by the paper itself:**
  - Embedding-model sensitive — BGE wins on factoid; GTE wins on aggregation; Contriever in between.
  - The "largest gap" can fall among the *least* relevant docs without the top-90% guard.
  - Fails for summarization-style queries that lack a sharp relevance frontier.

### Code-level keywords searched
`score_gap`, `gap`, `np.diff(scores)`, `argmax(diff)`, `argmax(g)`, `buffer`, `adaptive_k`, `adaptive-k`, `largest gap`, `largest_gap`, `score_diff`, `B = 5`.

### Project search results

| keyword family | matches in project code |
|---|---|
| `adaptive-k`, `adaptive_k`, `largest_gap`, `score_gap` | **0 matches** in project source/notebooks (only matches are inside `research_papers/` text and `research_repos/adaptive-k-retrieval/` upstream code) |
| `np.diff(scores)`, `argmax(diff)`, `argmax(g)` | **0 matches** in project notebooks for this score-gap pattern |
| `gap`, `score gap`, `score-gap` | only matches in unrelated contexts (e.g. enrichment field names, training/val distribution gap discussions) |

### Project-side experiments tried

**None.** No project notebook implements "sort scores descending → take first diff → argmax → cut + buffer". The closest mechanic in the actual pipeline is in `v7_5_precision/cascade_stages/_stage_e_selection.py`, but Stage E does *not* operate on a similarity distribution; it sorts by `stage_b_score_map` purely as a tie-breaker after LLM verdicts have been collected. The K is set by LLM verdict tiering (KEEP@5, KEEP@4, MAYBE), not by a discontinuity in the score curve. See `notebooks/_inventory/precision_v1_final.md` line 1644: "PRECISION-V1 — FINAL RESULTS (variable K, evidence-quote-gated)" — variable K, but evidence-gated, not gap-gated.

**Best result for this approach on this project:** N/A — never attempted.

**Verdict:** **STATUS = OPEN.** Strong candidate for the project's ROADMAP Step 3 because:
1. The Stage 2 fusion already produces a ranked-by-score list at the K=50–500 range where rerankers are useful (per `swiss-citation-experiments` SKILL §"Hybrid reranker shootout").
2. The val gold-size variance (10–47) is exactly the variability Adaptive-k is built for.
3. The method is fully cached-cost (one `np.diff` + `argmax` per query — milliseconds), so it can be evaluated against the existing F3 / F3-no-rerank score arrays without any new GPU work.
4. Buffer `B` is the only hyperparameter; pick on a train-leave-one-out cross-val if needed.

**Risk to flag if attempted:** the paper's own §5.4 warning — the largest gap can appear among the tail; the top-90% guard is essential. On our 43–46k-deep pool this guard matters a lot.

---

## Paper 2 — CAR: Cluster-based Adaptive Retrieval (Xu, Gupta, Aggarwal, Mahadevan, Krishnamachari — Coinbase + USC, arXiv 2511.14769, Nov 2025)

### Approaches in the paper

#### Approach 2.A — 1-D clustering on sorted similarity, cut at the best cluster boundary
- **Core algorithm (Algorithm 1):**
  1. Retrieve top-N (N chosen so that 90+% of queries' golds are inside).
  2. Min-max normalise the sorted distances to [0,1].
  3. Grid-search over a clustering backbone's hyperparameters; pick the configuration that maximises **silhouette score**.
  4. Identify contiguous cluster regions along the rank, take the rank indices `b` where the cluster label changes.
  5. For each boundary `b`, compute `score_b = (d̃_b − d̃_{b−1}) / max gap  +  α / b` (the second term is a **position penalty** that discourages cutting too high in the list).
  6. Cut at `k = argmax_b score_b − 1`.
- **Key hyperparameters:**
  - Clustering backbone — paper tests K-Means, DBSCAN, HDBSCAN, OPTICS, Agglomerative, Spectral, BIRCH, Bisecting K-Means (Table 4). Results 0.58–0.63 TES on CDP; 0.30–0.34 on MultiHop-RAG. **Choice of backbone barely moves the TES — the adaptive-cutoff itself is what matters.**
  - K-Means `n_clusters` ∈ [2, N/2] capped by √N.
  - DBSCAN `eps` ∈ {0.1, 0.325, 0.55, 0.775, 1.0}, `min_samples` ∈ [2, 5].
  - HDBSCAN `min_cluster_size` ∈ {2,3,4,5}, `min_samples` ∈ {1,2,3}.
  - α (position-penalty weight) is implicit in the equation — not separately swept in the paper.
- **Claimed result (production deployment):**
  - CDP RAG: average **15.57** docs per query (vs Top-40 baseline → 60% token reduction).
  - Hallucination: **0.10** (CAR) vs **0.12** (Top-40) vs **0.16** (Top-16).
  - End-to-end latency: **5.46 s** vs **6.99 s** (Top-40) → 22% latency cut.
  - User engagement: **+200%** week-over-week post-deployment.
  - Cross-encoder benchmark (MultiHop-RAG, Table 2): CAR TES = 0.34 on text-embedding-3-small, beats every fixed top-k.
- **Code reference:** none released in the paper. Algorithm 1 is fully self-contained pseudocode.

### Code-level keywords searched
`KMeans`, `from sklearn.cluster`, `cluster_score`, `n_clusters`, `elbow`, `1d kmeans`, `agglomerative`, `hierarchical`, `HDBSCAN`, `DBSCAN`, `silhouette`, `BisectingKMeans`, `cluster_cut`, `cluster.*cut`.

### Project search results

| keyword family | matches in project code |
|---|---|
| `KMeans`, `sklearn.cluster`, `n_clusters`, `HDBSCAN`, `DBSCAN`, `silhouette`, `agglomerative`, `BIRCH`, `elbow` | **0 matches** in project pipeline code (`v7_5_precision/`, `v7_4_fixes/`, project notebooks). Hits in `research_repos/` are upstream library code (FlagEmbedding, LEXTREME, etc.). Hits in `artifacts/*.jsonl` are LLM-enrichment fields like `clustering_themes` — unrelated to retrieval-side clustering. |
| `cluster_cut`, `cluster.*cut`, `cut.*gap` | **0 matches** |

### Project-side experiments tried

**None.** No project code clusters a sorted similarity distribution. The Stage E selection in `v7_5_precision/cascade_stages/_stage_e_selection.py` does perform a *form* of grouping (per `case_base` for dedup, per `addresses_aspect` for coverage), but this is **case-level / aspect-level grouping over LLM verdicts**, not clustering over the score distribution.

**Best result for this approach on this project:** N/A — never attempted.

**Verdict:** **STATUS = OPEN.** Reasons CAR is a plausible variant for Step 3 of ROADMAP:
- The paper's own Table 4 shows the clustering backbone choice is nearly irrelevant (0.58–0.63 on CDP), which means CAR is essentially "Adaptive-k with a smoother cutoff signal and a position penalty". The position-penalty term `α / b` is what differentiates it from a naive gap-cut.
- Same as AdaptiveK: zero new GPU cost — operates on cached score arrays.

Reasons CAR might be **worse than AdaptiveK for this project specifically:**
- CAR was tuned for *single-gold* or 2–4-gold queries. Our val has 10–47 golds. The "tight cluster of low-distance docs followed by a sharp increase" motif (paper §1) is the *factoid* signature, not the *aggregation* signature. Adaptive-k explicitly tested both regimes; CAR explicitly tested only the small-gold regime.
- The position penalty `α / b` biases the cut *toward smaller k*, which is the opposite of what our 10–47-gold queries need.

Recommendation if attempted: re-tune the penalty to `α · log(b)` or drop it; otherwise CAR will over-truncate on the high-gold queries (e.g. val_002 with 47 golds).

---

## Paper 3 — AcuRank: Uncertainty-Aware Listwise Reranking (Yoon et al. — SNU + UCSB, NeurIPS 2025)

### Approaches in the paper

#### Approach 3.A — TrueSkill-based per-document Bayesian rating with uncertainty-gated stopping
- **Core algorithm (Algorithm 1):**
  1. Initialise each candidate's relevance as a Gaussian `xᵢ ~ N(μᵢ, σᵢ²)`, with `μᵢ` = first-stage retrieval score (normalised) and `σᵢ = μᵢ/3`.
  2. Compute, for each doc, `sᵢ = P(xᵢ > t(k))` where `t(k)` is the relevance threshold s.t. `Σᵢ P(xᵢ > t(k)) = k`. (`t(k)` found by binary search.)
  3. Pick the **uncertain set** `C = {Dᵢ : ε < sᵢ < 1 − ε}`, default `ε = 0.01`.
  4. Sort `C` descending by `μᵢ`, partition into ordered groups of size `m = 20`.
  5. Apply the listwise LLM reranker to each group; update TrueSkill `(μᵢ, σᵢ)` from each group's permutation (treat the result as a multi-player game outcome).
  6. Stop when `|C| < τ` (default) or top-k unchanged for `T` iterations or compute budget exhausted.
  7. Final ranking = sort all docs by `μᵢ`.
- **Key hyperparameters:**
  - `k` = 10 (target rank cutoff for evaluating top-k probabilities; **this is the boundary the uncertainty is computed around**, not the final output K).
  - `ε` = 0.01 (uncertainty tolerance band).
  - `m` = 20 (reranker group size).
  - `τ` (stopping threshold on |C|) and `T` (top-k stability iterations).
  - Observation noise `σ̃` (global).
- **Claimed result (TREC-DL + BEIR, NDCG@10):**
  - Matches **sliding-window-1** in reranker-call budget (so same wall-clock when calls are sequential).
  - **+1.2 NDCG@10** average gain over SW-1 across benchmarks.
  - Scales better with compute than fixed-computation baselines (SW-x, TourRank-x).
  - Default reranker: RankZephyr; also evaluated with RankGPT (gpt-4.1-mini).
- **Code reference:** https://github.com/soyoung97/AcuRank (mirrored locally at `research_repos/AcuRank/`).
- **Important note for this project:** AcuRank is a *reranking budget allocator*, NOT a K-selector. It uses uncertainty to decide *which* candidates to spend reranker compute on. The final K is still a hyperparameter. The "K-stopping" framing in the task brief is a slight misread — AcuRank stops the **reranking** when uncertainty drops, not the **retrieval**. The relevance-mass-around-k signal could be re-purposed as a K-selector in principle, but the paper doesn't claim that use.

### Code-level keywords searched
`trueskill`, `TrueSkill`, `from trueskill`, `Bayesian`, `mu`, `sigma`, `uncertainty`, `listwise rerank`, `acurank`, `AcuRank`, `epistemic`, `silhouette` (not relevant), `setwise`, `tournament`, `sliding window`.

### Project search results

| keyword family | matches in project code |
|---|---|
| `trueskill`, `TrueSkill`, `acurank`, `AcuRank` | **0 matches** in project code. Hits are entirely in `research_repos/AcuRank/` (upstream code and the BM25 data files referencing dataset names like "dl19"/"dl23"). |
| `Bayesian`, `epistemic`, `mu/sigma` per-doc state | **0 matches** in project code |
| `listwise rerank` | matches only in `research_repos/rank_llm/` and `research_repos/AcuRank/` (upstream). Project rerankers (Qwen3-Reranker-8B, BGE-v2-m3, jina-v2-base) are **pointwise** scorers per the hybrid rerank shootout, not listwise. |

### Project-side experiments tried

**None.** This project does not currently use a listwise reranker at all — the hybrid rerank shootout (`swiss-citation-experiments` SKILL) used three pointwise cross-encoders. AcuRank's TrueSkill update step requires a **listwise** reranker output (a permutation over m=20 docs), which the project's pipeline does not produce.

**Best result for this approach on this project:** N/A — never attempted, and would require either (a) switching to a listwise LLM reranker like RankZephyr/RankGPT, or (b) synthesising a listwise outcome from pointwise scores (degenerate — defeats the purpose).

**Verdict:** **STATUS = OPEN, but lower priority than AdaptiveK / CAR.** Reasons:
1. AcuRank is a *reranker-budget* technique; the project's actual bottleneck is variable K **after** the rank, not reranker budget. Per the hybrid rerank shootout: at K ≥ 500, rerankers underperform plain fusion, and the F3-no-rerank diagnostic (ROADMAP Step 0) may drop rerankers entirely. If rerankers are dropped, AcuRank has nothing to allocate.
2. The TrueSkill-based **uncertainty signal** itself (`sᵢ = P(xᵢ > t(k))`) is interesting as a *standalone K-selector* — feed the F0 fusion scores in as the prior, pick the K' such that the cumulative-confidence on the inclusion mass passes a threshold. But this is a *re-purposing*, not the paper's claim.
3. Listwise reranking with a 100B-parameter judge (RankZephyr ≈ Llama-3-8B-tuned, RankGPT = GPT-4.1) is a different beast from this project's Qwen3-8B-judge cascade; integrating it is a multi-day rebuild.

Recommendation: revisit AcuRank only after (a) ROADMAP Step 0 confirms rerankers stay in the pipeline, and (b) ROADMAP Steps 1-2 confirm the cascade dossier doesn't already saturate the per-candidate evidence available.

---

## Aggregate summary

| paper | mechanism | hyperparams | project status | priority for Step 3 |
|---|---|---|---|---|
| Adaptive-k (EMNLP 2025) | sort scores desc → `argmax(np.diff)` cut + buffer B | B=5; restrict gap-search to top 90% | **OPEN** (never tried) | **HIGH** — cached-cost, designed for aggregation-style variable-gold workloads, matches val's 10–47 spread |
| CAR (Coinbase, Nov 2025) | 1-D cluster on sorted scores; silhouette-tuned; cut at best cluster boundary + position-penalty | clustering backbone (any of 8); α position-penalty implicit | **OPEN** (never tried) | **MEDIUM** — paper tuned for 1–4-gold queries; position penalty biases toward small K which hurts val_002 (47 golds). Worth testing with penalty disabled. |
| AcuRank (NeurIPS 2025) | TrueSkill per-doc Gaussian; uncertainty-gated listwise reranker calls | k=10, ε=0.01, m=20, τ, σ̃ | **OPEN, blocked** — needs a listwise reranker, which the project doesn't run | **LOW** — wrong fit; revisit only if cascade switches to listwise rerank |

### Cross-references checked
- `.claude/skills/swiss-citation-experiments/SKILL.md` — no entry for any of the three approaches. The closest entry is "Hybrid reranker shootout (2026-05-15)" which is a *pointwise* exercise.
- `.claude/skills/swiss-citation-approach/SKILL.md` — Thread 3 (line 44): "Final pick (variable K, F1-optimal sizing). Val gold size ranges 10-47; the system must choose K per query. **Not the current focus**, but ultimately the F1 lever once Threads 1+2 produce a sharp ranked list." — confirms variable K is an open thread and not implemented in any score-gap / clustering / TrueSkill form.
- `research/cascade_dossier_plan.md` — no mention of score-gap, clustering, or TrueSkill K-selection. The plan is about *signal enrichment*, not *K-selection geometry*.
- `ROADMAP.md` Step 3 (line 58): "Calibrate K per query from cheap features. Features: count of `statute_targets`, # codes mentioned, query length, court/law ratio." — confirms the **planned** K-selector is a *cheap-feature regression*, not any of the three paper methods. AdaptiveK / CAR are complementary alternatives to this plan; AcuRank is orthogonal.
- `v7_5_precision/cascade_stages/_stage_e_selection.py` — the *only* variable-K mechanism currently in the pipeline. It is **confidence-tier + aspect-coverage + case-dedup**, fully decoupled from the score distribution. The header comment is explicit: "Variable-size final selection driven by confidence + aspect coverage. NO TOP-K CAP."
- `notebooks/01_current_direction_cascade_rerank_precision/precision_v1_final.ipynb` (per inventory) — uses Stage E above + a Stage F "evidence-quote-gate" LLM pass. Both LLM-driven, neither geometric.

### Honest answer to the brief

For all three papers in this batch, the answer is the same: **STATUS = OPEN.** The Swiss citation pipeline has variable-K in the sense that the final selection size varies per query, but the *mechanism* is "LLM verdict tiering + aspect coverage" — none of (score-gap-cut, sorted-distance clustering, TrueSkill uncertainty) appears anywhere in `v7_*` code or in the project notebooks. The paper-side keyword grep confirms this: every match for `KMeans` / `silhouette` / `trueskill` / `score_gap` / `adaptive-k` resolves to either upstream library code under `research_repos/`, data files under `artifacts/`, or the paper texts themselves.
