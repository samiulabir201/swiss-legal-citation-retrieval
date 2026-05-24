# Research Index — Swiss Citation Extraction

Curated **2025-2026** literature + code repositories for the variable-K, no-hardcoding macro-F1 retrieval goal.

Location:
- Papers: [`E:\swiss_citation_extraction\research_papers\`](.)
- Code:   [`E:\swiss_citation_extraction\research_repos\`](../research_repos/)

All papers strictly 2025-2026 (earlier work is referenced inside the papers themselves). Repos may include some foundational pre-2025 codebases that are still the standard implementation for a technique.

---

## Goal — what we are trying to do

From a 50,000-candidate pool per validation query (macro recall ≈ 0.893; per-query range 0.76 to 1.00), produce a final **variable-size** answer set (1 to ~60 citations depending on the query) that maximizes **macro F1 ≥ 0.7** without:

- Any val-query-specific hardcoding
- Any fixed top-K cutoff
- Any train-derived priors (per project memory `feedback_train_unreliable`)

---

## Folder map

```
research_papers/
├── INDEX.md                          (this file)
├── legal_swiss/                      Direct Swiss-jurisprudence precedent
├── legal_graph_citation/             Joint case+law citation prediction
├── rerankers_calibration/            Cross-encoder with calibrated scores
└── variable_k_truncation/            Adaptive-K / stopping-rule methods
```

---

## 1. Swiss legal precedent

| Paper | Why it matters | File |
|---|---|---|
| **From Citations to Criticality: Predicting Legal Decision Influence in the Multilingual Swiss Jurisprudence** — Stern, Kawamura, Stürmer, Chalkidis, Niklaus. **ACL 2025**. arXiv 2410.13460. | Direct domain match — multilingual Swiss federal court decisions with citation-derived labels. Key finding: **fine-tuned multilingual models consistently outperform zero-shot LLMs** on Swiss legal tasks when training data is sufficient. Supports a measured-fine-tuning step on top of a multilingual base. | [legal_swiss/From_Citations_to_Criticality_ACL2025_arXiv2410.13460.pdf](legal_swiss/From_Citations_to_Criticality_ACL2025_arXiv2410.13460.pdf) |
| **UQLegalAI@COLIEE2025: Advancing Legal Case Retrieval with Large Language Models and Graph Neural Networks** — UQLegalAI team. arXiv 2505.20743. | Closest published benchmark to the user's task: legal citation retrieval, F1 evaluation, variable-K. SOTA F1=0.2962 (2nd place); 1st place 0.3353. Anchors a realistic comparison point. Uses CaseLink GNN with BM25 + e5-mistral-7b features. | [legal_swiss/UQLegalAI_COLIEE2025_arXiv2505.20743.pdf](legal_swiss/UQLegalAI_COLIEE2025_arXiv2505.20743.pdf) |

## 2. Legal citation graph / GNN

| Paper | Why it matters | File |
|---|---|---|
| **The Missing Link: Joint Legal Citation Prediction using Heterogeneous Graph Enrichment** — Wendlinger et al. **Springer DEXA 2025**. arXiv 2506.22165. | Heterogeneous R-GCN for joint case-case + case-law citation prediction. **Achieves macro AP 87-88%** on 200k-case German legal graph (similar scale to user's data). Jointly predicting both edge types adds +4.7 AP synergy. Enrichment with court-type, code, court-state meta-features adds +8.5 AP in sparsity conditions. Code released. | [legal_graph_citation/Missing_Link_Joint_Legal_Citation_GNN_arXiv2506.22165.pdf](legal_graph_citation/Missing_Link_Joint_Legal_Citation_GNN_arXiv2506.22165.pdf) |

## 3. Rerankers + calibration

| Paper | Why it matters | File |
|---|---|---|
| **Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models** — Qwen team. **June 2025**. arXiv 2506.05176. | Defines Qwen3-Reranker-{0.6B, 4B, 8B}. Multilingual (100+ languages incl. DE/FR/IT). **32k context**. Score = `sigmoid(logit_yes − logit_no)` = calibrated P(relevant). MMTEB-R: 64.6 / 72.7 / 72.9. Apache 2.0. **The default reranker for our pipeline.** | [rerankers_calibration/Qwen3_Embedding_Technical_Report_arXiv2506.05176.pdf](rerankers_calibration/Qwen3_Embedding_Technical_Report_arXiv2506.05176.pdf) |

## 4. Variable-K / truncation / stopping (the F1 lever)

| Paper | Method | Key result | File |
|---|---|---|---|
| **Adaptive-k Retrieval (No Tuning, No Iteration)** — Taguchi, Maekawa, Bhutani (Megagon Labs). **EMNLP 2025**. | Sort scores descending, cut at largest score-gap + buffer B=5 | Up to 99% token reduction vs fixed K; matches best fixed-K. Plug-and-play. | [variable_k_truncation/AdaptiveK_NoTuning_EMNLP2025.pdf](variable_k_truncation/AdaptiveK_NoTuning_EMNLP2025.pdf) |
| **Cluster-based Adaptive Retrieval (CAR)** — Coinbase + USC. arXiv 2511.14769. **Nov 2025**. | Cluster sorted scores (1-D K-means or hierarchical), cut at first elbow | Production at Coinbase: 60% token reduction, 22% latency reduction, 10% hallucination cut, 200% engagement | [variable_k_truncation/CAR_Cluster_Based_Adaptive_Retrieval_arXiv2511.14769.pdf](variable_k_truncation/CAR_Cluster_Based_Adaptive_Retrieval_arXiv2511.14769.pdf) |
| **AcuRank: Uncertainty-Aware Adaptive Computation for Listwise Reranking** — Yoon et al. **NeurIPS 2025**. arXiv 2505.18512. | TrueSkill Bayesian rating per candidate; stop when uncertain candidates drop below τ | Matches sliding-window-1 budget with higher accuracy; +1.2 NDCG@10 over best fixed baseline | [variable_k_truncation/AcuRank_Uncertainty_Aware_Listwise_arXiv2505.18512.pdf](variable_k_truncation/AcuRank_Uncertainty_Aware_Listwise_arXiv2505.18512.pdf) |
| **Two-stage Risk Control with Application to Ranked Retrieval** — Xu, Guo, Wei. **IJCAI 2025**. | Conformal prediction for ranked retrieval — calibrate on held-out set to get distribution-free 1-α coverage guarantee | Variable-K with statistical recall guarantee | [variable_k_truncation/TwoStage_Risk_Control_Ranked_Retrieval_IJCAI2025.pdf](variable_k_truncation/TwoStage_Risk_Control_Ranked_Retrieval_IJCAI2025.pdf) |
| **Dynamic Ranked List Truncation via LLM-generated Reference-Documents (PSI-Rank)** — arXiv 2604.09492. **2026**. | LLM generates a "boundary pseudo-doc"; rerank it with the same scorer; cut everything below pivot. Per-query (PSI-RankDyn) or average pivot (PSI-RankAvg). | 66% inference speedup with no quality drop on TREC-DL | [variable_k_truncation/Dynamic_RLT_LLM_Pivot_arXiv2604.09492.pdf](variable_k_truncation/Dynamic_RLT_LLM_Pivot_arXiv2604.09492.pdf) |
| **A Generalised and Adaptable Reinforcement Learning Stopping Method (GRLStop)** — Bin-Hezam & Stevenson. **SIGIR 2025**. arXiv 2505.01907. | RL policy that decides stop / continue at each rank position. Single model handles any target recall (improvement over RLStop 2024). | Outperforms classical TAR stopping methods; legal-domain-tested | [variable_k_truncation/GRLStop_Generalised_RL_Stopping_SIGIR2025_arXiv2505.01907.pdf](variable_k_truncation/GRLStop_Generalised_RL_Stopping_SIGIR2025_arXiv2505.01907.pdf) |
| **Talos: Optimizing Top-K Accuracy in Recommender Systems** — arXiv 2601.19276. **Jan 2026**. | Quantile-based learned threshold for top-K accuracy; converts ranking-dependent operations into score-vs-threshold comparisons | Differentiable, plug-into-loss top-K optimization | [variable_k_truncation/Talos_TopK_Accuracy_Recommender_arXiv2601.19276.pdf](variable_k_truncation/Talos_TopK_Accuracy_Recommender_arXiv2601.19276.pdf) |

**Total: 11 papers, all 2025-2026.**

---

## Code repositories ([`E:\swiss_citation_extraction\research_repos\`](../research_repos/))

| Repo | Purpose | Pipeline stage |
|---|---|---|
| [`Qwen3-Embedding/`](../research_repos/Qwen3-Embedding/) | Qwen3-Reranker-{0.6, 4, 8}B model + inference examples | **Stage B — cross-encoder rerank** (calibrated probabilities) |
| [`adaptive-k-retrieval/`](../research_repos/adaptive-k-retrieval/) | EMNLP 2025 Adaptive-k reference implementation | **Stage E — score-gap K estimator** |
| [`AcuRank/`](../research_repos/AcuRank/) | NeurIPS 2025 AcuRank implementation + RankLLM baselines | **Stage E — uncertainty-stop K estimator** |
| [`RLStop/`](../research_repos/RLStop/) | SIGIR 2024 RLStop + SIGIR 2025 GRLStop generalisation | **Stage E — RL stopping K estimator** |
| [`rank_llm/`](../research_repos/rank_llm/) | castorini RankLLM (SIGIR 2025 resource paper) — production listwise reranker toolkit | **Stage B alt — listwise LLM reranker** |
| [`RLT4Reranking/`](../research_repos/RLT4Reranking/) | Meng SIGIR 2024 truncation toolkit (BiCut / Choppy / AttnCut / MtCut / LeCut) | **Stage E — supervised K estimators** |
| [`missing_link/`](../research_repos/missing_link/) | DEXA 2025 heterogeneous R-GCN joint citation predictor | **Stage C alt — graph link prediction** |
| [`Qwen3-Embedding/`](../research_repos/Qwen3-Embedding/) | (already listed; same repo provides embedding model too) | **Stage A — first-stage retrieval (you already use Qwen3-Embedding-8B)** |
| [`FlagEmbedding/`](../research_repos/FlagEmbedding/) | BGE-M3 + BGE-reranker-v2-m3 + finetune scripts | **Stage B fallback** |
| [`gpl/`](../research_repos/gpl/) | UKPLab GPL unsupervised domain adaptation | **Phase C — domain adaptation without train labels** |
| [`ColBERT/`](../research_repos/ColBERT/) | Stanford ColBERT late-interaction | Alternative to cross-encoder when truncation matters |
| [`LEXTREME/`](../research_repos/LEXTREME/) | Niklaus group multilingual legal NLP benchmark | Reference for Swiss legal evaluation |
| [`legalbenchrag/`](../research_repos/legalbenchrag/) | ZeroEntropy legal RAG benchmark — flags Cohere reranker hurts on legal | Reference for validation protocol; reminder to validate per-domain |

**Total: 13 repos.**

---

## Pipeline mapping — paper → pipeline stage

```
Stage A — Retrieval pool (50k or 100k from v7.5 RRF)
   Tools: Qwen3-Embedding-8B (already in use) + BM25 + RRF fusion
   Repo: Qwen3-Embedding/, FlagEmbedding/

Stage B — Cross-encoder rerank (RTX PRO 6000 Blackwell)
   Tool: Qwen3-Reranker-4B with sigmoid → calibrated P(rel)
   Repo: Qwen3-Embedding/, optional rank_llm/ for listwise variant
   Paper: Qwen3_Embedding_Technical_Report

Stage C — Graph-aware score fusion
   Tool: Heterogeneous R-GCN on (cases, laws, chambers, codes)
   Repo: missing_link/
   Paper: Missing_Link_Joint_Legal_Citation_GNN

Stage D — LightGBM feature ranker (your dossier features)
   Tool: LightGBM lambdarank on the 30 features
   No external repo needed
   Reference: (your existing research/feature_synthesis_round2_2026-05-13.md)

Stage E — Variable-K ensemble (the F1 lever)
   6 independent K estimators, ensembled:
   - K_adaptive-k    : adaptive-k-retrieval/    EMNLP 2025
   - K_CAR           : CAR algorithm (paper only, implement from arXiv 2511.14769)
   - K_acurank       : AcuRank/                 NeurIPS 2025
   - K_pivot         : implement PSI-Rank from arXiv 2604.09492
   - K_conformal     : implement from IJCAI 2025 Two-stage Risk Control
   - K_mtcut         : RLT4Reranking/rlt/supervised_rlt/models/mmoecut.py
   Then ensemble (median) + clamp (max(K_aspect, n_hyde_aspects, 3) ≤ K ≤ count with p > 0.05)

Stage F — Precision pass
   Your existing Qwen3-32B evidence-quote verifier; swap instead of drop

Optional Phase X — GPL domain adaptation (if F1 < 0.65)
   Repo: gpl/  → adapt the reranker to Swiss legal vocabulary using only corpus
```

---

## What's NOT included and why

- **GPL paper (Wang et al. 2022, NAACL)** — too old per the 2025-2026 restriction; code in [`gpl/`](../research_repos/gpl/) is still current.
- **Lipton et al. 2014 (F1-optimal thresholding)** — foundational result but pre-2025; the principle is embedded in every 2025-2026 paper here.
- **LegalBench-RAG paper (2024)** — important cautionary result (Cohere reranker hurts on legal) but pre-2025; code in [`legalbenchrag/`](../research_repos/legalbenchrag/) is the eval framework.
- **mGTE paper (EMNLP 2024)** — pre-2025; model is referenced as a comparison baseline only.
- **Surprise (SIGIR 2023)** — pre-2025; method is implemented inside [`RLT4Reranking/rlt/unsupervised_rlt.py`](../research_repos/RLT4Reranking/rlt/unsupervised_rlt.py).
- **CAR code** — Coinbase has not released code as of Nov 2025; the algorithm is short and reproducible from the paper.

---

Created 2026-05-13. All cloned via personal GitHub account `samiulabir201` over SSH (`github-personal` host alias, never the office key).
