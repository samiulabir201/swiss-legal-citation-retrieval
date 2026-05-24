# PAPER → PRACTICE — master inventory

Generated 2026-05-15 from a 5-agent parallel sweep. Every paper in `research_papers/` was matched against actual Python code cells in `notebooks/_inventory/*.md` (not just markdown) and cross-referenced with `.claude/skills/swiss-citation-experiments/SKILL.md`. Best result per approach is reported when multiple notebook attempts exist.

## Sections (read each for full per-paper detail)

| # | Section | File | Lines | Papers |
|---|---|---|---|---|
| 1 | Legal-Swiss precedent | [`01_legal_swiss.md`](01_legal_swiss.md) | 149 | Stern et al. ACL 2025 (Criticality); UQLegalAI COLIEE 2025 (CaseLink) |
| 2 | Legal citation graph / GNN | [`02_legal_graph_citation.md`](02_legal_graph_citation.md) | 239 | Wendlinger et al. DEXA 2025 (Missing Link / R-GCN) |
| 3 | Rerankers + calibration | [`03_rerankers_calibration.md`](03_rerankers_calibration.md) | 285 | Qwen3 Embedding Tech Report (Qwen team, June 2025) |
| 4 | Variable-K part 1 (geometry-based) | [`04_variable_k_part1_adaptivek_car_acurank.md`](04_variable_k_part1_adaptivek_car_acurank.md) | 184 | AdaptiveK (Megagon EMNLP 2025); CAR (Coinbase+USC); AcuRank (NeurIPS 2025) |
| 5 | Variable-K part 2 (pivot / RL / conformal / quantile) | [`05_variable_k_part2_pivot_rl_conformal_quantile.md`](05_variable_k_part2_pivot_rl_conformal_quantile.md) | 205 | PSI-Rank; GRLStop; Two-stage Risk Control; Talos |

**Total: 11 papers across 1,062 lines of paper-to-practice analysis.**

## Implementation status across all 11 papers

| Paper | Adopted | Tried & Dropped | Open / Never Tried |
|---|---|---|---|
| Stern et al. — Criticality | — | Legal-Swiss-RoBERTa fine-tune in `exp_A6` (stat_recall@500=0.101 vs BGE-M3 0.376; FAILED) | Stern's own classification task (LD-Label, Citation-Label on `rcds/swiss_criticality_prediction`); 11 other paper baselines (XLM-R, SwissBERT, mDeBERTa, MiniLM, DistilmBERT, X-MOD, mT5, BLOOM, Legal-Swiss-LF) |
| UQLegalAI — CaseLink | BM25+dense two-stage spirit (v7.5 multi-language cascade); peer-scale LLM embedder (Qwen3-Embedding-8B substituted for e5-mistral-7b); InfoNCE-style contrastive loss (in `exp_A6` only) | — | CaseLink GAT itself (zero matches for caselink/casegnn/GAT/GCN/GraphSAGE/torch_geometric/dgl); degree regularization on RRF hubs; year-filter |
| Wendlinger — Missing Link R-GCN | Citation graph as deterministic-counter channel (`statute_backprop` recall 0.453; `graph_forward` recall 0.324); meta-features (`court_base`, `law_code`, `paragraph_role`) as channel weights / role boosts; `r_cocite` as 1 of 9 dossier rank features in F3 hybrid | — | HGE/R-GCN learned link prediction (no torch_geometric/DGL in any of our code); joint Case→Case + Case→Law training |
| Qwen3 Embedding/Reranker tech report | Qwen3-Embedding-8B as canonical first-stage retriever, EXACTLY per paper (4096-dim, bf16/fp16, `Instruct: {...}\nQuery: {...}` template, English instructions over DE/FR/IT, left-padded last-token pooling); Qwen3-Reranker-8B `score = exp(yes)/(exp(yes)+exp(no))` formula verbatim in hybrid shootout (vLLM `max_model_len=1024`) and Untitled75 F1=0.777 reference (transformers `max_length=4096`) | — | — (paper followed faithfully) |
| AdaptiveK | — | — | Score-gap-cut + buffer B=5 (no `score_gap` / `argmax(np.diff)` / `adaptive_k` matches anywhere in our code) |
| CAR | — | — | 1-D K-means / hierarchical clustering of sorted scores (no `KMeans` / `silhouette` matches) |
| AcuRank | — | — | TrueSkill Bayesian rating per candidate (no `trueskill` / `mu` / `sigma` matches; also needs a listwise reranker our pipeline doesn't run) |
| PSI-Rank | — | — | LLM-generated boundary pseudo-doc + pivot rerank (no `pseudo_doc` / `pivot` / `PSI` matches) |
| GRLStop | — | — | RL stop/continue policy (RLStop repo cloned at `research_repos/RLStop/` but UNMODIFIED) |
| Two-stage Risk Control | — | — | Conformal prediction with calibration set (no `conformal` / `coverage_guarantee` matches) |
| Talos | — | — | Quantile-based learned threshold for top-K accuracy (no `quantile_threshold` / `topk_loss` matches) — **flagged as most natural fit for our top-K F1 objective** |

## Current variable-K mechanism (what we actually use)

- **Threshold-based final pick** with fused-score cutoff `0.65`, stored at `drive_sync/swiss_law/threshold_and_runtime_configs/threshold.json` (103 B). Submitted val Macro F1 = **0.498**.
- **`_stage_e_selection.py` heuristic** (in `v7_5_precision/cascade_stages/`) driven by LLM verdict tiers (KEEP@5 ∪ KEEP@4 ∪ aspect-fill, case-dedup, no cap). Fully decoupled from score-distribution geometry — does NOT use any of the 7 variable-K papers' techniques.

## Cross-cutting findings

### Followed faithfully (high-confidence)
- **Qwen3-Embedding-8B + Qwen3-Reranker-8B** are deployed exactly as the Qwen team's June 2025 paper prescribes. English-instructions-over-DE/FR/IT convention confirmed in code; calibrated reranker score formula confirmed in two notebooks.

### Adopted in spirit (paper insight → custom implementation)
- **CaseLink BM25+dense two-stage** → realized as our v7.5 14-channel RRF fusion with `bm25` channel + `vector` channel
- **Missing Link meta-features** (court-type, code, court-state) → realized as channel-weight heuristics + paragraph-role boosts + `r_cocite` as a dossier feature
- **UQLegalAI's e5-mistral-7b** → substituted by larger Qwen3-Embedding-8B (4096-dim vs 4096 also, multilingual)

### Tried and dropped (negative-result)
- **Legal-Swiss-RoBERTa fine-tune** (Stern's Table 2 baseline) — `exp_A6` notebook. SimCSE + MultipleNegativesRankingLoss. stat_recall@500 = 0.101 vs BGE-M3 baseline 0.376. Dropped.

### Never attempted — the largest open surface
**7 of 11 papers (all of variable-K)** have **zero implementation in any of our 90 notebooks**:
- AdaptiveK, CAR, AcuRank (geometry-based stopping)
- PSI-Rank, GRLStop, Two-stage Risk Control, Talos (pivot / RL / conformal / quantile)

Plus from Stern + UQLegalAI: HGE/R-GCN learned link prediction, CaseLink GAT, degree regularization on RRF hubs, Stern's algorithmic LD-citation score as authority prior.

## Priority recommendation for the open work

Per ROADMAP.md Step 3 (variable-K calibration is the final F1 lever), priority order from agent analyses:

1. **AdaptiveK** (Megagon EMNLP 2025) — highest priority. Score-gap + buffer B=5. Cached cost (no GPU). Designed for variable-gold-count aggregation queries that match val's 10-47 gold range. Drop-in at `_stage_e_selection.py` post-LLM-tier-merge.
2. **Talos** (arXiv 2601.19276, Jan 2026) — second priority. The only one of the 7 that's a learning objective, not a post-hoc cutoff. Differentiable top-K loss matches our top-K F1 task goal.
3. **CAR** (Coinbase production) — medium priority. 1-D K-means cut at first elbow. Position-penalty would hurt val_002's 47 golds.
4. **PSI-Rank** — medium-low. LLM-generated pivot pseudo-doc. 66% inference speedup but our pipeline already runs an LLM verdict — adds complexity.
5. **GRLStop** — low. RL policy needs training; RLStop repo is cloned but unmodified.
6. **AcuRank** — low. Needs listwise reranker our pipeline doesn't run; ROADMAP Step 0 may drop rerankers entirely.
7. **Two-stage Risk Control** — low for now. Distribution-free coverage guarantee is appealing but needs a calibration set; val is only 10 queries.

Also re-consider:
8. **CaseLink GAT** trained on v7.5 pool — explicit gap. The cascade dossier plan already enumerates the exact features a small GAT would learn. Time investment: 1-2 days.

## Anti-recommendations (don't waste time)

- HGE/R-GCN as co-predictor — ruled out by Obs 2 (99.4% of gold pairs have no graph edge). Sister provisions in same law don't cite each other.
- Stern's classification task (LD-Label / Citation-Label) — different task surface (judgment criticality, not citation retrieval).
- 11 alternative encoders from Stern's Table 2 — already proven inferior to Qwen3-Embedding-8B at this scale.

## Files map

For full per-approach detail (paper claim quote, code excerpt, output metric, evidence cell + .md filename):

```
docs/paper_analysis/
├── PAPER_TO_PRACTICE.md           ← this file
├── 01_legal_swiss.md              ← Stern + UQLegalAI
├── 02_legal_graph_citation.md     ← Missing Link R-GCN
├── 03_rerankers_calibration.md    ← Qwen3 tech report
├── 04_variable_k_part1_adaptivek_car_acurank.md
└── 05_variable_k_part2_pivot_rl_conformal_quantile.md
```
