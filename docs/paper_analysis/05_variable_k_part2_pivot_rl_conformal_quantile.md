# Paper analysis 05 — Variable-K truncation (part 2): LLM-pivot, RL-stopping, conformal, quantile

**Scope:** Four 2025-2026 papers proposing distinct ways to choose *how many* documents to keep / when to stop / where to cut. All four are candidate solutions for the project's Thread 3 ("variable K, F1-optimal sizing") — the open final-pick layer documented in `swiss-citation-approach`. As of 2026-05-15, **none of the four methods has been implemented in the project**.

**Project context — current variable-K baseline:**
- `drive_sync/swiss_law/threshold_and_runtime_configs/threshold.json` holds a single fused-score threshold `0.6500…` tuned on val, yielding `macro_f1_val = 0.49753` (the "reference baseline").
- `v7_5_precision/cascade_stages/_stage_e_selection.py` is the live cascade's adaptive K: it takes whatever survives the Stage-D LLM verdicts (KEEP@5 ∪ KEEP@4 ∪ aspect-gap-fill MAYBEs), then deduplicates by case. K is not capped or learned — it's whatever LLM conviction supports.
- `v7_5_precision/cascade_stages/_stage_f_validation.py` then asks a second LLM pass to flag REDUNDANT items.
- `notebooks/_inventory/reference_F1_0.777_Untitled75.md` uses a **two-zone threshold** (`HIGH_THRESH=0.55`, `LOW_THRESH=0.25`) + LLM-judged borderline zone, reaching F1=0.777 on val (but train-eval leakage flagged in the notebook itself).
- v7.5 pool R@50k = 0.893 on val (per `swiss-citation-experiments`); pool ceiling at val_003 = 0.766 R_max. Anything below the threshold layer can't recover what the pool dropped.

The current selection logic is therefore a **heuristic confidence tier + aspect coverage + LLM dedup** — explicitly not any of the four formal methods reviewed below.

---

## 1. PSI-Rank — Dynamic RLT via LLM-generated reference document
**Source:** `research_papers/variable_k_truncation/Dynamic_RLT_LLM_Pivot_arXiv2604.09492.txt`
**Authors:** Sinhababu, Bharati, Ganguly, Mitra (IIT Kharagpur / Glasgow), arXiv Apr 2026.
**Repo cited in paper:** `https://github.com/nilanjansb/DynamicRLT`.

### Method
Truncate the first-stage ranked list at a **semantic pivot** — an LLM-generated reference document whose target relevance grade is exactly "marginally relevant" (grade 2 of 0-3 in NIST/UMBRELA conventions). The pivot acts as a boundary between relevant and non-relevant docs.

**Prompt template (Figure 1 of the paper):**
- System: *"You are an expert information retrieval judge. Your task is to generate a single document that matches a specific relevance grade for the given query."*
- User: *"Given the query: '{Q}' Generate a document that would be scored exactly: '{relevance_grade}'."*
- Relevance grades: 0=irrelevant, 1=minimally, 2=marginally (the pivot), 3=highly. Length is controlled to match the collection's `|D̄|` (typically 200-512 tokens).

**Once the pivot d* is generated, the same first-stage retriever R(q, d) is used to score d* in-place inside the ranked list.** That position becomes the cut-off.

### Variants
- **PSI-RankDyn** (per-query hard boundary): cut at the pivot's position in *this* query's ranked list. D⁺ = {d : R(q,d) > R(q,d*)}, then rerank D⁺.
- **PSI-RankAvg** (query-set statistics): use a calibration set C and cut at the mean pivot score across queries `μ_t = (1/|C|) Σ R(q,d*_q)`. More stable, less query-adaptive.

The paper extends the same pivot to listwise reranking (SNOW = shared non-overlapping window with d* injected as a propagation anchor; VS-Sliding = adaptive-stride sliding window using d* density; GPTD-Part = swap TDPart's noisy internal pivot for the LLM-generated d*).

### Claims
- TREC-DL: outperforms existing RLT methods.
- LLM-listwise reranking inference **66% faster** vs prior baselines, with no quality drop.
- No training, no hyperparameter tuning, retriever-agnostic.

### Have we tried it on the Swiss citation pipeline?
**No.** Grep across `notebooks/_inventory/`, `v7_5_precision/`, and `research_repos/` for `pseudo_doc`, `pivot`, `boundary`, `PSI`, `RankAvg`, `RankDyn` → zero hits matching this construct. The codebase has HyDE-style pseudo-document generation (in Stage 1 retrieval, e.g., `pool_v75_multiquery_hyde_test_no_lift.md`) but HyDE generates a **maximally** relevant pseudo-doc to inject into the *query* embedding — it's not a marginally-relevant pivot scored back through the retriever to find a cut-off. Different mechanism.

### Where to try it
Most natural drop-in point: between Stage A (prune) and Stage B (composite) in `v7_5_precision/cascade_stages/_stage_a_prune.py` → `_stage_b_composite.py`. Generate one grade-2 pivot per query with Qwen3-8B (already loaded for HyDE), score it with the v7.5 fused signal exactly the way every candidate is scored, then keep only candidates above the pivot's fused score before sending to Stage C/D.

**Falsifiable pass criterion (suggested):** macro R@K_pivot ≥ macro R@500 of current fusion, with K_pivot < 500 on average. If the pivot lands at K_pivot > 1000 routinely it's not adding signal vs the existing top-cuts.

### Risks specific to this project
- The paper validates on TREC-DL (English MS-MARCO / passages, single-best-answer setup). Our task is multi-citation per query (gold size 10-47), cross-lingual EN→DE/FR/IT legal corpus. Whether a grade-2 LLM pivot lands at a stable position in a noisy 50k-doc legal pool is empirically unknown.
- LLM pivot quality on Swiss legal cases is the unknown. Qwen3-8B *can* generate plausible Swiss legal text (we use it for HyDE and structured query expansion), but generating *moderately* relevant content with controlled length is harder than generating maximally relevant.

---

## 2. GRLStop — Generalised reinforcement learning stopping
**Source:** `research_papers/variable_k_truncation/GRLStop_Generalised_RL_Stopping_SIGIR2025_arXiv2505.01907.txt`
**Authors:** Bin-Hezam & Stevenson (Sheffield), SIGIR '25.
**Repo:** https://github.com/ReemBinHezam/GRLStop
**Local clone present:** `research_repos/RLStop/` — contains `README.md`, `data/`, `rl_utils/`, `rlstop_experiments.ipynb` (RLStop SIGIR-2024 code; GRLStop extends it).

### Method
Sequential decision problem: walk a ranked list in fixed-size batches; at each batch, choose **STOP** or **CONTINUE**.

- **State (S_i):** vector of length `T+2` where T is the number of batches (paper uses T=100). First T elements = proportion of relevant docs in each batch (known for batches ≤ i, **estimated by a classifier** for batches > i — this is the key extension over RLStop). Last 2 elements = batch index i and target recall.
- **Action (A):** {STOP, CONTINUE}.
- **Reward R(S_i):** piecewise function (Eq. 1 of the paper) parameterised by α, β. Positive reward before the target-recall batch t', negative after. Parameters α/β tune the balance:
  - α<1, β>1 → "Recall-Objective" (prefer overshoot)
  - α>1, β<1 → "Cost-Objective" (prefer undershoot)
  - α=β=1 → "Balanced-Objective"
- **Target-recall input:** scalar in the state vector. **One trained model serves any target recall** — this is the "generalised" claim and the headline win over RLStop, which required a separate model per target.
- **Algorithm:** PPO (Stable-Baselines3), 2-hidden-layer feed-forward policy, softmax over 2 actions. Classifier on unexamined batches = TF-IDF + LogisticRegression with cost-sensitive weighting.

### Claims
Outperforms classical TAR stopping baselines (SCAL, AutoStop, SD-training/sampling, IP-H, Knee, TM-adapted, QBCB) and RLStop itself on 6 datasets (CLEF 2017/18/19, TREC-TR, TREC-Legal, RCV1).

### Have we tried it?
**No.** No notebook, script, or import in the project references `RLStop`, `GRLStop`, `policy_net`, `Q-learning`, `target_recall`, or PPO/Stable-Baselines3. `research_repos/RLStop/` exists as a clone but is unmodified — the repo was pulled, not used.

### Where to try it
This is a **stopping** method designed for TAR (technology-assisted review for systematic reviews), which is structurally identical to "scan a ranked candidate list and stop when you've collected enough gold citations." The Swiss task fits: 50k v7.5-pool candidates, gold size 10-47 per query.

**Plug-in plan (if pursued):**
1. Use `_stage_b_composite.py` fused score as the "ranking" (or any of the 9 dossier signals from F3).
2. Train a classifier on train-set queries to predict "is gold" from candidate features (would reuse the LLM enrichments we already have).
3. Wire up the GRLStop PPO loop in `research_repos/RLStop/` against this corpus.
4. Train one model on the 1139 train queries; evaluate on the 10 val queries.

**Falsifiable pass criterion (suggested):** GRLStop's stopping point on val must beat the static threshold baseline F1=0.498 by ≥3 points. Otherwise the PPO overhead isn't worth it vs the threshold.

### Risks specific to this project
- GRLStop expects "target recall" as an explicit input, but our metric is **F1**, not recall. We'd need a wrapper that sweeps target recall over a grid and picks the F1-optimal stopping point per query — or rewrite the reward to be F1-based, which voids the paper's optimality argument.
- 1139 train queries is small for a PPO policy that previously trained on TREC-Legal-scale data, but train/val distribution shift on this project (99% German train vs 100% English val, court share 1.2% vs 40.6%) is the bigger structural concern noted in `swiss-citation-experiments`.

---

## 3. Two-stage Risk Control — conformal prediction for ranked retrieval
**Source:** `research_papers/variable_k_truncation/TwoStage_Risk_Control_Ranked_Retrieval_IJCAI2025.txt`
**Authors:** Xu, Ying, Guo, Wei (NJIT + Rutgers), IJCAI '25.

### Method
Apply learn-then-test (LTT) and conformal risk control (CRC) to a **two-stage** retrieval-then-rank pipeline. The output is a distribution-free, finite-sample guarantee that the expected risk at each stage is below user-chosen levels α₁, α₂.

**Mechanism in words:**
1. Held-out calibration set with i.i.d. samples (X_i, Y_i, Z_i): query + gold-relevant docs (stage 1) + gold-ordered top-r₀ docs (stage 2).
2. Two finite parameter grids Λ (retrieval tuning) and Φ (ranking tuning).
3. Compute empirical risks `R̂_n^(1)(λ)` and `R̂_n^(2)(λ, φ)` on the calibration set, with monotone-in-λ loss functions (e.g., loss = 0 when λ=1, loss = 1 when λ=0).
4. For each λ ∈ Λ, compute a Hoeffding-Bentkus p-value for the null `H_i^(1): R^(1)(λ_i) > α₁`. Apply Bonferroni at level δ → keep λ rejecting this.
5. For each surviving λ, fixed-sequence-test `H_{i,j}^(2): R^(2)(λ_i, φ_j) > α₂` at level δ/m → keep φ.
6. Returned set R of (λ, φ) pairs jointly controls FWER at level δ (Theorem 1) and provides **1-δ probability of meeting both risk constraints simultaneously**.
7. A separate CRC extension (Theorem 2) gives `E[R^(1)(λ̂)] ≤ α₁` and `lim sup E[R^(2)(λ̂, φ̂)] ≤ α₂` in expectation rather than high-prob.

**Custom loss functions for ranked retrieval:** the paper defines retrieval risk = 1 - recall@K (with K determined by λ) and ranking risk based on listwise loss over the r₀-relevant subset.

### Claims
- Distribution-free coverage guarantee at user-chosen α₁, α₂.
- Validated on MSLR-Web10k and Yahoo LTRC.
- Computational saving vs treating multiple risks jointly: exploits monotonicity of the loss in (λ, φ) to do fixed-sequence rather than full grid.

### Have we tried it?
**No.** No code in this project mentions `conformal`, `calibration_set`, `coverage_guarantee`, `risk_control`, `LTT`, or `CRC`. The closest analogue is the trivial calibration in `threshold.json` (one threshold tuned on val, no calibration/test split, no statistical guarantee).

### Where to try it
The held-out calibration step requires a held-out set of (query, gold) pairs disjoint from where we tune. We have 1139 train + 10 val + test (no labels). A clean split would be:
- **Calibration:** ~100-200 train queries sampled randomly.
- **Test of the guarantee:** remaining train + val.
- **Risk function R^(1)(λ):** the retrieval cut-off — e.g., 1 - recall@λ where λ ∈ {500, 1k, 2k, 5k, 10k, 20k, 50k} indexes the v7.5 pool.
- **Risk function R^(2)(λ, φ):** the final-pick stage — e.g., 1 - F1 where φ is the fused-score cut-off threshold ∈ {0.4, 0.45, …, 0.85}.

**Falsifiable pass criterion (suggested):** when calibrated to deliver 1-α coverage of recall ≥ 0.8 in stage 1 and F1 ≥ 0.6 in stage 2, the empirical coverage on the test split must be ≥ 1-α. If it is and the resulting (λ, φ) pair gives val F1 ≥ 0.55, this is a strict improvement over the heuristic threshold.

### Risks specific to this project
- The i.i.d. calibration assumption is the binding constraint. Our train/val are *not* drawn i.i.d. from the same distribution (massive distribution shift, see Obs 4 / `swiss-citation-experiments`). The conformal guarantee will hold *if and only if* we draw the calibration set from the same distribution we'll evaluate on — meaning we'd need to use val queries to calibrate val performance, which then has nothing to test on. The 10-query val set is anyway too small for any non-trivial 1-α guarantee (α ≥ 1/11 ≈ 0.09 is the best possible).
- **Best fit:** to deliver F1 guarantees on **test** (where we have 100s of queries and no labels), we'd calibrate on val. That bounds the marginal F1 of the test deliverable, which is exactly the right framing for a competition submission. This is the strongest reason to consider this method.

---

## 4. Talos — quantile-based top-K accuracy loss
**Source:** `research_papers/variable_k_truncation/Talos_TopK_Accuracy_Recommender_arXiv2601.19276.txt`
**Authors:** Zhang, Yang, Chen et al. (Zhejiang U), WWW '26, Jan 2026.
**Repo:** https://github.com/cynthia-shengjia/WWW-2026-Talos

### Method
A **training-time loss function** that directly optimises Precision@K / Recall@K for recommender systems, replacing the ranking-dependent truncation indicator `I(r_i ≤ K)` with a comparison `I(score_i ≥ τ_u)` where `τ_u` is a learned per-user score threshold.

**Three components:**
1. **Quantile reformulation (Eq. 3.1-3.2):** τ_u = the score of the item exactly at rank K for user u. Precision@K reduces to `Σ_{i ∈ P} I(s_{u,i} ≥ τ_u) / K`, which is differentiable once we replace I(·) with a sigmoid surrogate.
2. **Sampling-based quantile regression (Eq. 3.3):** instead of evaluating τ_u over all items (O(|I|)), sample a negative set N̄ and use importance-weighted pinball loss `ρ_β`. Estimation error reported as <0.02 (Table A.4).
3. **Constraint term for score-inflation (Eq. 3.4-3.7):** without a constraint, both item scores and τ_u inflate together. Add denominator `Σ_{i ∈ I} φ(s_i - τ)` (Heaviside replaced with sigmoid surrogate) to penalise score inflation and keep training stable.

**Final Talos loss (Eq. 3.7):**
`L_Talos = -(1/|P|) Σ_{i∈P} log[ σ((s_i - τ_u)^{1/η}) / Σ_{j∈N̄} σ((s_j - τ_u)^{1/η}) ]`

Reads like Softmax Loss (SL) but with **score - threshold** instead of raw scores, and a single temperature hyperparameter η.

### Claims
- Provably tight upper bound on -log Precision@K (Advantage 2 in §3.2).
- Equivalent to Distributionally Robust Optimisation → robust to distribution shift (Advantage 3).
- Convergence guarantees (Advantage 4).
- O(η̄|U||N̄|) time, same as SL.
- Beats SL, BPR, LLPAUC, OPAUC, SL@K on 4 datasets × 3 backbones (LightGCN, SimpleX, BPR-MF).

### Have we tried it?
**No.** No code in this project references `quantile`, `topk_loss`, `differentiable_topk`, `learned_threshold`, `pinball_loss`. The project has not done any *training* of a final-pick layer — every model in the cascade is used as-is (BM25, Qwen3-Embedding-8B, Qwen3-Reranker-8B, jina, BGE, Qwen3-8B judge). No supervised top-K objective.

### Why this is the most natural fit for the project goal
Stated in the brief and confirmed: **Talos is the only one of the four that targets top-K accuracy as a learning objective**. Macro F1 over a per-query variable K is what we are trying to maximise. The other three methods choose K post-hoc on top of frozen models; Talos lets you *train a scoring head* whose argmax-K is the F1 you want.

### Where to try it
Most plausible plug-in: train a small head on top of the cascade's per-candidate dossier vector (9 dossier features + 3 reranker scores from the F3 hybrid + fusion rank = ~13-dim per (query, candidate) pair). Loss = Talos with K = expected gold size for the query (which we could pre-estimate from a query-side classifier, or set as a learned per-query τ following the paper exactly).

**Training data:** 1139 train queries × (50k v7.5 candidates) = up to 57M (q, d) pairs with binary gold/not-gold. Negative sampling per the paper keeps this manageable.

**Falsifiable pass criterion (suggested):** train Talos head on train; evaluate macro F1 on val with no further tuning. Must beat val F1 = 0.498 (threshold baseline) by ≥5 points to be worth the training cost.

### Risks specific to this project
- Same train-to-val distribution-shift concern that hangs over option 2c (reranker fine-tuning) in `swiss-citation-approach`. If Talos overfits the German-language training prior, it won't generalise to English val.
- The "K" in Talos is fixed during training. Our K varies 10-47 per query. We'd need to either condition the head on a per-query K estimate, or train a separate model per K bucket, or use the paper's per-user variant which learns τ per query (most natural).
- Talos was validated for recommender systems with thousands of items per user — our setting has 50k candidates per query, similar order of magnitude. Compute should be fine on Blackwell.

---

## Cross-method comparison

| Method | What it controls | When chosen | Training needed? | Best fit signal in our pipeline | Have we tried it? |
|---|---|---|---|---|---|
| PSI-Rank | Cut-off K | Test time | None (uses existing LLM + retriever) | Stage-A → Stage-B handoff | No |
| GRLStop | Stopping point on ranked list | Test time | Yes — PPO policy | Stage-D verdict ordering | No |
| Two-stage Risk Control | (λ, φ) tuning pair with statistical guarantee | Calibration time | None — pure calibration | Pool cut-off + final-pick threshold | No |
| Talos | Score of each candidate (learned head) | Training time | Yes — gradient training | New head on F3 dossier vector | No |

**The four methods are largely orthogonal and could compose:** Talos trains a scoring head; PSI-Rank uses that head to pick a per-query cut-off via an LLM pivot; Two-stage Risk Control calibrates the (cut-off, threshold) pair to deliver a 1-α F1 floor; GRLStop is the alternative to PSI-Rank if you want stop-as-you-scan instead of cut-at-pivot.

**Single most natural fit per the project's stated goal (top-K accuracy as objective):** **Talos**. The other three are downstream of any scoring head you have; Talos *is* the scoring head.

---

## Pointers to the project's existing variable-K state

- `e:\swiss_citation_extraction\drive_sync\swiss_law\threshold_and_runtime_configs\threshold.json` — single threshold 0.65, F1=0.498 baseline.
- `e:\swiss_citation_extraction\v7_5_precision\cascade_stages\_stage_e_selection.py` — current adaptive K (KEEP@5 ∪ KEEP@4 ∪ aspect-fill, case dedup, no cap).
- `e:\swiss_citation_extraction\v7_5_precision\cascade_stages\_stage_f_validation.py` — LLM-based redundancy drop (precision booster).
- `e:\swiss_citation_extraction\notebooks\_inventory\reference_F1_0.777_Untitled75.md` — historical reference: two-zone threshold (0.55 high / 0.25 low) + LLM borderline judge, F1=0.777 val (leakage flagged in notebook).
- `e:\swiss_citation_extraction\notebooks\01_current_direction_cascade_rerank_precision\precision_v1_final.ipynb` — current cascade entry point.
- `e:\swiss_citation_extraction\research_repos\RLStop\` — unmodified clone of the SIGIR-2024 predecessor repo. GRLStop is the 2025 follow-up; same repo family.
