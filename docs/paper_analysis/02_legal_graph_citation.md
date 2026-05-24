# Paper Analysis 02 — Missing Link: Joint Legal Citation Prediction via Heterogeneous GNN

## Paper meta

| field | value |
|---|---|
| Title | The Missing Link: Joint Legal Citation Prediction using Heterogeneous Graph Enrichment |
| Authors | Lorenz Wendlinger, Simon A. Nonn, Abdullah Al Zubaer, Michael Granitzer (Univ. Passau / IT-U Linz) |
| Venue | Springer DEXA 2025 |
| arXiv | 2506.22165v1 (27 Jun 2025) |
| Local txt | `e:\swiss_citation_extraction\research_papers\legal_graph_citation\Missing_Link_Joint_Legal_Citation_GNN_arXiv2506.22165.txt` |
| Local pdf | `e:\swiss_citation_extraction\research_papers\legal_graph_citation\Missing_Link_Joint_Legal_Citation_GNN_arXiv2506.22165.pdf` |
| Reference code repo (pulled) | `e:\swiss_citation_extraction\research_repos\missing_link\src\graph_models.py` (DGL `RelGraphConv`, `HeteroGraphConv`, `GraphConv`, `GATConv`) |

### One-paragraph summary
HGE (Heterogeneous Graph Enrichment) is an R-GCN-style link-prediction model trained jointly on Case→Case and Case→Law edges in the OLD201k German citation graph (201k Cases, 50k Laws, ~1M+90k edges). Two innovations: (a) a general residual replaces the RGCN per-relation self-loop, (b) categorical meta-features (Case Court Type, Law Code, Case Court State) are exposed as discrete graph nodes with their own relations rather than as concatenated features. The encoder produces case/law embeddings; the decoder is an asymmetric inner product trained with BCE on balanced negatives. Time-based splitting (train on old, test on new) is the evaluation rigour.

### Headline numbers (Tables 2-5 of the paper)
- Macro AP on **OLD201k**: **88.1 ± 1.3** (HGE) vs 85.9 (RGCN) vs 82.9 (VGAE) vs 80.4 (GCN) vs 72.1 (SGD on text-only).
- Macro AP on **OLD36k** (sparser, ~10× smaller): **87.5 ± 1.05** (HGE) vs 80.3 (RGCN); HGE delivers **+7.2 AP / +8.5 AUC-ROC** over RGCN in this sparser regime.
- Joint vs separate prediction: joint gives **+4.7 AP** on the harder CC (Case-Case) edges of OLD36k, **+0.6 AP** on OLD201k. Runtime roughly halves (1.79–2.15× speed-up).
- Initial node features: Jina-V2-base-de (8192 ctx) ~+1.3 macro AP over `all-mpnet-base-v2`. Chunking parameters basically don't matter.
- Inference cost: encode-then-dot-product all reference nodes, **<5 s for one case on OLD201k**; full retrain "<1 hour".

---

## Step 1 — All technical approaches the paper proposes

| # | Approach | Paper section |
|---|---|---|
| A1 | **Heterogeneous R-GCN encoder** with separate weight matrices `W_r` per relation `r ∈ {CC, CL}` (and the additional enrichment relations) | §3.1 Eq. 1 |
| A2 | **General residual** replacing the per-relation self-loop of RGCN — `h_i^(l) = h_i^(l-1) + Σ_r Σ_j (1/c) W_r h_j^(l)` | §3.1 Eq. 2 |
| A3 | **3 layers × 256 hidden dim** model size; 2 heads for GAT baselines; 200 epochs, lr 1e-4, Adam, 0.2 dropout | §3.5 |
| A4 | **Joint multi-edge-type learning** (CC + CL) with shared encoder, separate decoders | §3.2, Table 3 |
| A5 | **Meta-feature exposure as discrete nodes** — for `m: V→{0,1}^M`, add one new node per expression and an edge `(u, m(u))` for each node bearing that meta value. Avoids hand-crafted categorical embeddings. | §3.3 Eq. 3 |
| A6 | Specific meta-features chosen: **Case Court Type** (level of appeal × jurisdiction), **Law Code** (the law book), **Case Court State** | §3.3 |
| A7 | **Reverse edges + self-loops as ordinary relation types** rather than special-cased | §3.3 |
| A8 | **Initial node features = mean-pooled sentence-transformer embeddings** (Jina-V2-base-de, 8192-token context, ~512 overlap) of full doc text with citations stripped to prevent leakage | §3.4 |
| A9 | **Asymmetric inner-product decoder** ([23] Yu&Wang 2014) trained with cross-entropy on balanced (uniform per-source) negatives | §3.5, §5.1 |
| A10 | **Time-based splitting** with cumulative training: train on all cases before cut-off, test on later cases; produces train graph + semi-inductive inference graph + indicator graph for prediction | §5.1 Fig. 4 |
| A11 | **Macro-averaged AP + AUC-ROC** as the metric (so the minority CC edge type isn't drowned by CL) | §5.1 |
| A12 | **Fully inductive ablation** at test_size=100% (no inference edges); HGE still >80 AP via meta-feature topology alone | §5.6 Fig. 5b |

---

## Step 2 — Search terms used against our code

Searches performed (Grep, glob across the repo):

| Query | Where searched | Hits in our code | Conclusion |
|---|---|---|---|
| `torch_geometric` | full repo | 1 (in `research_repos/FlagEmbedding/research/old-examples/search_demo/requirements.txt` — not our pipeline) | We do not use PyG. |
| `RGCNConv`, `HeteroData`, `RelGCN`, `HeteroGNN`, `RelGraphConv`, `HeteroGraphConv` | full repo | Only in `research_repos/missing_link/src/graph_models.py` (the reference repo we pulled for context) | No GNN model has ever been instantiated in our pipeline. |
| `dgl` | full repo | Only in `research_repos/missing_link/` | We do not use DGL. |
| `RGCN`, `R-GCN`, `HGE` | full repo | Only in the paper text + reference repo | No port of HGE was attempted. |
| `link_prediction`, `joint_citation`, `case_law_edge` | full repo | None in our notebooks/scripts | No joint case+law edge model. |
| `negative_sampling`, `neg_sampling` | notebooks/_inventory | None | No negative-sampling training loop. |
| `missing_link`, `wendlinger` | full repo | Only as a directory name under `research_repos/` | The paper has been read; nothing implemented. |
| `co_citation`, `cocite` | notebooks/_inventory | 28 files (anchor_funnel v4-v7.5, pool_v75, rerank_hybrid, cascade_legalmalr) | Co-citation is used as a **counter-based retrieval channel and a dossier rank signal**, not as a learned GNN edge type. See Step 3. |
| `citation_graph_extracted.sqlite` | full repo | Loaded in every anchor-funnel v4-v7.5 notebook plus the canonical v7.5 pool notebook | Confirmed as a tie-breaker/counter source — not as a learned graph. |
| `graph_forward`, `graph_reverse`, `graph_2hop`, `sibling_graph`, `sibling_expansion` | full repo | Used in 22+ notebooks as **deterministic retrieval channels** scoring by edge-count × judgment-importance, not by a GNN. | Channel-based, not learned. |

---

## Step 3 — How our codebase actually uses the citation graph

This is the section the brief asked for specifically.

### 3.1 The local citation graph (file + schema)

| | |
|---|---|
| Path | `e:\swiss_citation_extraction\data_insights\citation_graph_db_and_edges\citation_graph_extracted.sqlite` |
| Size | **5.68 GiB** (post back-ref patch + case-level patch; built 2026-05-10 15:49) |
| Companion JSONs | `laws_de_links.json` (45,021 undirected law-law edges, 31,217 source nodes), `court_considerations_links.json` (4.1 M court edges) |
| Gold-as-node coverage | **96.0 %** (up from 92.9 % pre-patch). 4 % residual is corpus gaps (LugÜ, FIDLEG, FINIG, FinfraG, GBV, parts of URG/ZGB) — not patchable from the graph side. |
| Pre-patch backups (kept) | `.bak_pre_backref` (2.13 GB, pre-back-reference), `.bak_pre_caselevel` (2.19 GB, pre case-level-aggregation) |

Documented in `docs/data_analysis/04_embeddings_and_indices.md` §B.4 and §B.5.

The schema is **edge-count-style**, not nodes-with-features. Documented loaders in `v7_4_fixes/cell_bodies_extracted_from_notebook/_cell34_body.py` (the channel runner) and the per-channel fixes in `v7_4_fixes/per_channel_fixes/` build these in-RAM indices from the sqlite:

| Index | Built in | Used by channel |
|---|---|---|
| `idx_graph_out` | cell 16 of every anchor-funnel notebook | `channel_graph_forward` (forward 1-hop) |
| `idx_graph_in` | cell 16 | `channel_graph_reverse` (reverse 1-hop) |
| `idx_court_base` | cell 16 | `channel_sibling` (court-base sibling expansion) |
| `idx_law_direct` | cell 16 | `channel_law_direct`, `channel_co_citation`, `channel_statute_backprop` |
| `idx_court_statute` | cell 16/20 | `channel_court_statute`, `channel_co_citation` |
| `idx_court_statute_count` | court_statute_fix.py | specificity weighting |
| `co_neighbours[canon]` | cell 20 | `channel_co_citation` (top-K co-cited canons per LLM-named statute) |
| `idx_judgment_importance` | cell 16 patch (`sibling_graph_fix.py`) | sibling/forward/reverse scoring |
| `canon_count` | cell 20 | specificity weighting in co_citation_backprop_fix.py |

### 3.2 How v7.5 multi-channel funnel uses the graph (channel-by-channel)

Source: `v7_4_fixes/cell_bodies_extracted_from_notebook/_cell34_body.py` (channel orchestrator) + `v7_4_fixes/per_channel_fixes/co_citation_backprop_fix.py` + `v7_4_fixes/per_channel_fixes/sibling_graph_fix.py`.

| Channel | What it does | Scoring | v7.5 budget | v7.5 weight | Mean recall (v7.4 meas.) |
|---|---|---|---|---|---|
| `graph_forward` | for each seed did, emit `t` for each outgoing edge | `edge_count(t) × sqrt(1+log(1+importance(judgment(t)))) × role_boost(t)` | 5000 | **2.0** | **0.324** (per cell-8 comment in canonical v7.5 notebook) |
| `graph_reverse` | for each landmark seed (importance ≥ 5), emit `s` for each incoming edge | `factor(seed) × role_boost(s)` summed | 3000 | 0.5 | — (low, gated to landmark seeds only) |
| `graph_2hop` | **disabled** (`enable_graph_2hop: False`) | — | 1500 | 0.0 | disabled |
| `sibling_expansion` | for each seed, fetch every paragraph row sharing the same `court_base` | `seed_count(cb) × judgment_factor(cb) × role_boost(s)` | 5000 | 1.0 | — |
| `co_citation` | for each LLM statute target canon, fetch top-K=50 co-cited canon neighbours (cap `corpus_count ≤ 15000`), pull their law+court rows | `n_cocite × 1/log(2 + canon_count[nb])` (specificity-weighted) | 2500 | 0.7 | — |
| `statute_backprop` | each caught court row contributes its statutes; score = caught-row-count, specificity-weighted | `n_caught × 1/log(2 + global_court_count[canon])` | 2000 | **2.5** | **0.453** (best single channel) |

Key shape: **all of these are counter-based with weighting heuristics**. No learned propagation, no learnt per-relation matrix W_r, no negative sampling, no BCE loss. The graph is consumed read-only.

### 3.3 Why "graph as co-predictor" was rejected (Obs 2)

From `research/personal_observations.md` Observation 2 and `data_insights/citation_graph_db_and_edges/laws_de_links.json` analysis:

| Metric | Count | Percentage |
|---|---|---|
| Gold co-occurrence pairs also in graph | 97 | **0.59 %** of gold pairs |
| Gold pairs **NOT** in graph | 16,449 | **99.41 %** |
| Graph edges that appear as gold pair | 97 | 0.22 % of all graph edges |
| Strong gold pairs (≥3 co-occurrences) also in graph | 6 / 465 | 1.3 % |

> "A model that expands predictions by following citation graph edges will almost always add irrelevant citations, hurting precision without recovering the missed gold citations. The citation graph cannot be used as a co-prediction signal."
> — `research/personal_observations.md` Obs 2

Codified in three skill files:
- `.claude/skills/swiss-citation-task/SKILL.md` (Banned moves table): *"Treating citation graph as a co-prediction expander — Obs 2: 99.4% of gold pairs have no edge. OK as a tie-breaker only."*
- `.claude/skills/swiss-citation-experiments/SKILL.md` (Resolved phantom issues): *"Citation graph can predict gold co-occurrence — 99.41% of gold-citation pairs have no graph edge. Useful only as a tie-breaker."*
- `.claude/skills/swiss-citation-orchestrator/SKILL.md`: included in the list of "things already tried and ruled out".

### 3.4 Did we attempt joint case+law link prediction (the paper's main claim)?

**No.** Searched the full repo for `link_prediction`, `joint_citation`, `case_law_edge`, `negative_sampling`, `RGCNConv`, `RelGraphConv`, `HeteroData`, `HeteroGraphConv`, `R-GCN`, `RGCN`, `HGE`, `wendlinger`, `missing_link`. Only matches are in:

1. `research_repos/missing_link/src/graph_models.py` — the **reference repo** cloned from the paper's GitHub for context. Not wired into our pipeline.
2. `research_repos/FlagEmbedding/research/old-examples/search_demo/requirements.txt` — third-party FlagEmbedding example, irrelevant.
3. Inventory file `exp_A2_hyde_and_enumeration.md` — false hit (the substring "GCN" inside the word "stat_recall@500 ... gate 0.40").

No supervised GNN, no negative sampling, no BCE link-prediction loss has ever been trained in this project's notebooks (`notebooks/_inventory/*.md`).

### 3.5 The 96 %-coverage vs 99.4 %-no-edge discrepancy (the brief's "explain this")

Both numbers are true and measure different things:

| Property | Value | Source |
|---|---|---|
| **Gold-as-node coverage** | **96.0 %** of gold citations appear as nodes (extractable citation strings) in `citation_graph_extracted.sqlite` | endgame §6.3, `docs/data_analysis/04_embeddings_and_indices.md` §B.4 |
| **Gold-pair-as-edge coverage** | **0.59 %** of (gold_a, gold_b) co-occurrence pairs from train queries are connected by an edge | Obs 2, `research/personal_observations.md` |

Why the gap:
- Coverage is measured **per node** — "does the gold citation string exist as a target in any edge anywhere in the graph?". The patches (ATF/DTF/c./consid./space-docket) lifted this from 92.9 % to 96.0 %.
- Pair-edge presence is measured **per (gold_i, gold_j) pair** — "do the two gold citations of the same query cite each other?". The empirical answer is: nearly never. The intersection between train-query gold-co-occurrence and the graph's edge set is 0.59 %.
- The structural reason: queries usually concern a **legal situation**, and the gold citations are the legal building blocks for that situation. Those building blocks (e.g. Art. 963 ZGB and Art. 965 ZGB; Art. 10 Abs. 1 BV and Art. 10 Abs. 2 BV) are **thematically related** but their statute texts **don't textually cite each other** — sister provisions inside the same law don't reciprocally reference. Court decisions cite individual statutes in their reasoning; they don't generate a graph edge "(Art. 963 ZGB, Art. 965 ZGB)" just because both appear in the same paragraph.
- Hence: 96 % of golds are reachable as nodes in graph traversal; <1 % of gold *co-occurrence* relations are encoded as edges. Edge-following expansion adds noise; node-anchored retrieval (the v7.5 channels) works.

This is exactly the regime where HGE's joint CL prediction would in theory help **except** that the Swiss-citation task is not link prediction at all — see §4 below.

---

## Step 4 — Code-verified attempts of HGE-like approaches

| Paper approach | Attempted in our code? | Evidence | Best measured result | Verdict |
|---|---|---|---|---|
| A1 R-GCN encoder | **No** | No `torch_geometric`/`dgl`/`RGCN` usage in any notebook | — | Not pursued. |
| A2 General residual replacing self-loop | **No** | No GNN model exists | — | Not applicable. |
| A3 3×256-dim GNN training | **No** | No GNN training loop | — | Not pursued. |
| A4 Joint case + law link prediction | **No** | No link-prediction objective anywhere | — | Not pursued (task mismatch — see §6). |
| A5 Meta-feature exposure as discrete nodes | **No** (as a *learned* GNN feature) | But we **do** consume the same meta-info (court_type, law_code, court_state via court_base / docket prefix) as channel-routing heuristics and as dossier rank features (`r_chamber`, `r_doctrinal`) | F3 hybrid macro R@500 = 0.423 (rerank_hybrid_3model_shootout) | Channel-form: kept. GNN-form: not pursued. |
| A6 Court Type / Law Code / Court State features | **Yes — as static fields, not graph nodes** | `doc_meta` carries `court_base`, `law_code`, `paragraph_role`. The `_role_boost` and `judgment_factor` functions in `sibling_graph_fix.py` are the analogues. | Sibling/graph channel weights tuned in v7.5; per-channel mean recall = 0.324 (graph_forward) | Kept as channel weighting. |
| A7 Reverse edges + self-loops as relations | **Partially** — `idx_graph_in` provides reverse traversal but as a counter, not a learned relation type | `channel_graph_reverse` in `sibling_graph_fix.py` (landmark-gated, threshold imp ≥ 5) | low recall, weight 0.5 in v7.5 | Kept at small weight. |
| A8 Sentence-transformer node features (Jina V2 8192) | **Different stack** — we use Qwen3-Embedding-8B (full corpus, 4096-dim fp16, 2.65M docs, max_seq_len 768) | `embeddings/qwen3_8b_unified_chunk*.npy`, 21 GB, see `docs/data_analysis/04_embeddings_and_indices.md` §A | R@1000 = 0.289 on val (Obs 3 ceiling) — **structurally insufficient as a sole channel** | Kept as one of 15 channels. |
| A9 Asymmetric inner-product decoder + BCE | **No** | No discriminative training | — | Not pursued. |
| A10 Time-based splitting | **No** (train/val/test are language-stratified, not date-stratified) | — | — | Not applicable to Swiss leaderboard task. |
| A11 Macro AP / AUC-ROC | **No — we use Macro F1** (set retrieval, exact-string match) | `research/problem_statement.md` | Target Macro F1 0.6–0.8; v7.5 + F3 best macro F1 < 0.45 at any K | Different metric; not directly comparable. |
| A12 Fully inductive prediction | **No** | — | — | Not applicable. |

### 4.1 Best result on the closest analogue we *did* run
Graph-derived signals enter our pipeline two ways:
1. **As retrieval channels** (graph_forward, graph_reverse, sibling_expansion, co_citation, statute_backprop): contribute to v7.5 pool R@50k = **0.893 macro**, with `statute_backprop` (mean recall 0.453) and `graph_forward` (0.324) as the top two. The pool problem is closed at this level (`swiss-citation-experiments` SKILL.md).
2. **As a dossier rank signal**: `r_cocite` is one of 9 dossier features fused in F3 hybrid (rerank_hybrid_3model_shootout_with_outputs line 1225-1246). F3 macro R@500 = **0.423** vs F0 fusion-only 0.415 (i.e. ~marginal). F2 (rerank-only RRF without dossier) collapses to R@2k = 0.268 — confirming the lift is from the dossier signals collectively, not the cocite/graph features alone.

---

## Step 5 — Aggregate (best-result + POC-counts rule)

- **Best result of HGE itself in our project: not measured** (never attempted).
- **Closest in-house analogue (graph-channel funnel + cocite dossier)**:
  - v7.5 pool: **R@50k = 0.893 macro** (canonical).
  - F3 hybrid: **macro R@500 = 0.423**, **macro R@2000 = 0.568** (no rerank baseline F0 hits 0.415 / 0.611 — rerankers add nothing above K=500).
  - These are *retrieval recall*, not the paper's macro AP. The metrics are not interchangeable.
- **POC count for HGE-style learned link prediction in our pipeline: zero notebooks**. The only HGE source on disk is the cloned reference repo at `research_repos/missing_link/src/`, never imported by any of our notebooks.

---

## Step 6 — Cross-paper synthesis (why we did not adopt HGE)

### 6.1 Task mismatch
HGE solves **link prediction within a known graph** — "given a corpus where some Case→Law edges are missing, predict the missing edges". Our task per `research/problem_statement.md` is **closed-vocabulary set retrieval** — "given a *new* English query (not a graph node), return the set of citation strings in the closed corpus". The new query is not a node in the citation graph. Even an inductive HGE setting still requires the new node to be added to the graph; an arbitrary English question does not naturally have outgoing edges to seed the propagation. The mismatch is structural, not just engineering.

### 6.2 Empirical block: Obs 2
Even if we did frame queries as graph nodes, Obs 2 shows that gold-citation **pair-coverage** in our citation graph is 0.59 %. HGE's joint case+law prediction relies on the assumption that nearby co-cited citations form graph neighbourhoods; the Swiss corpus structure does not satisfy that assumption — sister provisions in the same law book do not cite each other, and gold sets for a query are largely defined by **thematic similarity** rather than **explicit citation edges**.

### 6.3 What the paper *does* contribute to our project
Even without porting HGE, the paper validates two design choices already in our v7.5 funnel:

| Paper finding | Our existing implementation |
|---|---|
| Meta-features (court type, law code, court state) materially improve retrieval (+8.5 AP in sparsity) | We use court_base, law_code, paragraph_role as deterministic channel-weight / role-boost / `r_chamber` / `r_doctrinal` features (sibling_graph_fix.py, dossier features 0-8 in the F3 hybrid). |
| Joint learning of case + law citations regularises and prevents shortcut collapse | We retrieve case (court_considerations) and law (laws_de) jointly through the same channels; we never split them into two separate pipelines. The 15-channel funnel handles both edge types in one pass. |
| Sentence-transformer features for node embedding are sufficient; chunking parameters matter little | We use Qwen3-Embedding-8B (2025 SOTA, MMTEB #1) at max_seq 768; per Obs 3 this caps at R@1000 = 0.289, lower than the paper's HGE but the paper is on AP not recall and on a denser graph. The Jina-V2-base-de result in Table 5 confirms that for our scale (2.65 M docs vs 250 k cases) we have made the right encoder choice. |

### 6.4 What would have to change to adopt HGE here
1. Build a `(query, citation)` node-set rather than a `(case, law)` node-set. Queries are not currently graph nodes.
2. Mine gold (query, citation) pairs from val/train as positive edges; mine balanced uniform-per-source negatives. (Train has 99 % German vs val/test 100 % English — distribution shift would dominate, see Obs 4.)
3. Train an RGCN with relations `{(query, citation_law), (query, citation_court)}` plus the existing CC and CL edges as background. 3 layers × 256 dim is feasible on Blackwell (`swiss-citation-config` SKILL.md confirms 95.6 GB VRAM is more than enough; DGL on PyTorch 2.10 / CUDA 12.8 works).
4. Decode with asymmetric inner product, eval with macro AP **then** convert top-K to a Macro F1 score with variable-K thresholding — at which point the bottleneck shifts back to bottleneck 2 (F1 K-pick).

This is a multi-week effort competing with the **already-committed** Cascade Dossier Phase 1 plan (`research/cascade_dossier_plan.md`) which uses the same graph signals as a much cheaper dossier feed to a Stage-2 LLM. Given that:
- The pool is already at R@50k = 0.893 (above the val_003=0.766 ceiling).
- The bottleneck is **candidate selection + final-set sizing**, not graph-feature quality.
- Reranker fine-tuning is already on hold pending the F3-no-rerank diagnostic.

…HGE is not a productive next move on this codebase.

---

## Final verdict

- **Approaches the paper proposes that we already do (in deterministic channel form, not learned GNN form): A5, A6, A7** — meta-feature exposure, court-type / law-code / court-state routing, reverse-edge traversal. These are baked into the v7.5 funnel via `_cell34_body.py` and `sibling_graph_fix.py` / `co_citation_backprop_fix.py`.
- **Approaches the paper proposes that we do NOT do, and have explicitly ruled out: A1-A4, A9-A12** — learned R-GCN encoder, general residual, joint multi-edge BCE training, time-based splitting, fully inductive eval. The ruling out is anchored by Obs 2 (graph cannot co-predict gold pairs) and a task-shape mismatch (closed-vocabulary set retrieval, not link prediction).
- **Approaches the paper proposes that we replaced with a stronger 2025 alternative: A8** — Qwen3-Embedding-8B in place of Jina-V2-base-de.
- **No POCs of HGE itself exist in this repo.** The reference code at `research_repos/missing_link/src/graph_models.py` is cloned for documentation only.

Cross-references:
- `.claude/skills/swiss-citation-experiments/SKILL.md` (Resolved phantom issues row: "Citation graph can predict gold co-occurrence — 99.41 % of gold-citation pairs have no graph edge. Useful only as a tie-breaker. (Obs 2)").
- `.claude/skills/swiss-citation-task/SKILL.md` (Banned moves: "Treating citation graph as a co-prediction expander").
- `research/personal_observations.md` Observation 2.
- `docs/data_analysis/04_embeddings_and_indices.md` §B.4 (citation_graph_extracted.sqlite role: tie-breaker only).
- `v7_4_fixes/per_channel_fixes/co_citation_backprop_fix.py` (specificity-weighted co_citation channel).
- `v7_4_fixes/per_channel_fixes/sibling_graph_fix.py` (judgment-importance-weighted sibling / graph_forward / graph_reverse channels).
- `v7_4_fixes/cell_bodies_extracted_from_notebook/_cell34_body.py` (the multi-channel orchestrator that fuses all the graph-derived counters into the v7.5 pool).
