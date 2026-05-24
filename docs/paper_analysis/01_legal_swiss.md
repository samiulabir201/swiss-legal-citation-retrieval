# Paper analysis — legal_swiss

Scope: the two papers in `research_papers/legal_swiss/`.
Method: extract every technical approach, grep the inventory of code cells under `notebooks/_inventory/*.md`, attribute each match to a notebook + cell + output, then cross-reference the experiment ledger at `.claude/skills/swiss-citation-experiments/SKILL.md`.

---

## Paper 1: From Citations to Criticality (ACL 2025)

**File:** `research_papers/legal_swiss/From_Citations_to_Criticality_ACL2025_arXiv2410.13460.pdf` (text extract used: `.txt`)
**Venue / year:** ACL 2025 / arXiv 2410.13460
**Authors:** Ronja Stern, Ken Kawamura, Matthias Stürmer, Ilias Chalkidis, Joel Niklaus
**One-line summary:** Introduces the Swiss "Criticality Prediction" dataset (LD-Label binary + Citation-Label 4-class) over 138 k multilingual SFSC cases (de/fr/it) and benchmarks small fine-tuned multilingual encoders against zero-shot GPT-3.5 / LLaMA-2; small fine-tuned encoders win.

**Core technical claims:**
- Fine-tuned XLM-RoBERTa-Large achieves the best aggregate macro-F1 (37.1) across LD-F / LD-C / C-F / C-C inputs; SwissBERT (34.8) and Legal-Swiss-LongformerBase (29.5 / 38.6 val) follow.
- Small fine-tuned encoders consistently beat zero-shot GPT-3.5 (Agg 28.1) and LLaMA-2 (Agg 12.5) despite having 1000× fewer parameters.
- Swiss-specific pretraining (SwissBERT, Legal-Swiss-RoBERTa, Legal-Swiss-Longformer by Rasiah et al. 2023) gives a clear lift on LD-labels over generic multilingual baselines.
- Citation-Label task is hard even for domain models — the 4-class citation-frequency task caps around 25–29 macro-F1 (vs. 25 random baseline).
- Algorithmically derived labels (LD-flag and citation-count × recency weight) scale to 138 k cases without manual annotation.

---

### Approach 1 — Multilingual encoder fine-tune on Swiss legal text (XLM-R / Legal-Swiss-RoBERTa / SwissBERT family)

- **Paper's claim/result:** "XLM-R_Large … 37.1 … SwissBERT … 34.8 … Legal-Swiss-LFBase … 38.6 (val 47.0)" — Table 2/3 + Table 6 (val). The paper argues that fine-tuning small multilingual encoders on the Swiss legal task beats zero-shot LLMs.
- **What we tried in our code (Grep-verified):**
  - `notebooks/11_early_experiments_exp_A/exp_A6_legal_roberta_finetune.ipynb` — we used **the exact Stern-family model `joelniklaus/legal-swiss-roberta-large` (Rasiah et al. 2023)** as a sentence encoder and ran two-stage training: SimCSE pre-train (1 epoch) → MultipleNegativesRankingLoss fine-tune (1 epoch, bs=32, lr=2e-5, 1024-dim mean-pool) on (court paragraph anchor, statute/BGE positive) pairs mined from `rcds/swiss_citation_extraction` IOB tags. Inventory: `exp_A6_legal_roberta_finetune.md`, cells 6, 8–10.
    - Code excerpt: `BASE = 'joelniklaus/legal-swiss-roberta-large'` ... `model = SentenceTransformer(modules=[word, pool]) ... train_loss = MultipleNegativesRankingLoss(simcse_model)`.
    - Output (cell 10, val statute-recall on `laws_de.csv`, 175 933 docs, 10 val queries):
      - pre-train (no FT) DE-query stat_recall@500 = **0.060** (sanity floor)
      - A6 FT DE-query stat_recall@500 = 0.067; HyDE stat_recall@500 = 0.101; Enum stat_recall@500 = 0.074
      - **Best A6 stat_recall@500 = 0.101 (RRF de+hyde+enum)** vs. Exp-3 BGE-M3-based baseline 0.376 (cell 12 verdict).
      - Hard-negative margin sanity (cell 21): Pos mean 0.5993 / HN mean -0.0597 → margin 0.6591, so the encoder *did* learn the task — it just doesn't outrank our enrichment-driven BGE-M3 baseline.
    - **Best result: stat_recall@500 = 0.101 (Legal-Swiss-RoBERTa-Large + SimCSE + MNR-FT + 3-query RRF on val).**
- **How it compares to paper:** Not directly comparable — the paper measures macro-F1 on a 4-class classification (LD-Label / Citation-Label) over de/fr/it test sets, while we use this encoder for *retrieval* of statute citations against a 175 k-doc German legal corpus on 10 val queries. The paper does not report retrieval recall. The Stern paper is therefore evidence for the *base model's* multilingual legal pretraining quality, not for the downstream retrieval task we ran.
- **Verdict:** dropped. The gate "stat_recall@500 > 0.376" FAILED, and the court re-encoding step was skipped. The fine-tuned encoder is `data/artifacts/A6/legal_swiss_finetuned/` on Drive but is not used by the v7.5 multi-query pool (which won with BGE-M3 + Qwen3-Embedding-8B instead).
- **Notes:** The base model is Rasiah et al. 2023 (`Legal-Swiss-RoBERTa-Large`), one of the four Stern-paper baselines. The poor downstream retrieval result here is a *task-mismatch* signal (sentence-pair retrieval ≠ classification head fine-tune), not a refutation of the paper. The XLM-R-base/large, SwissBERT, mDeBERTa-v3, MiniLM, DistilmBERT, X-MOD, mT5, and BLOOM models from the paper were NOT tested.

### Approach 2 — Zero-shot multilingual LLM (paper baseline)

- **Paper's claim/result:** GPT-3.5 Agg = 28.1, LLaMA-2 Agg = 12.5 on the test set — small fine-tunes win.
- **What we tried in our code (Grep-verified):** We never reproduced the paper's zero-shot LLM baselines (no Criticality-Prediction LLM prompting). We do use Qwen3-8B-Instruct heavily for query expansion (HyDE, enumeration, structured fields) — but for a *retrieval* task, not for LD/Citation classification.
- **Verdict:** not attempted on Stern's task. Re-running the paper's GPT-3.5 / LLaMA-2 baseline is out of scope for our retrieval pipeline.

### Approach 3 — Criticality Prediction task (LD-Label, Citation-Label) and the `rcds/swiss_criticality_prediction` dataset

- **Paper's claim/result:** Two-tier labels derived algorithmically from leading-decision membership + citation-frequency × recency weighting; 138 531 cases (85 167 de / 45 451 fr / 7 913 it); train 2002–15 / dev 2016–17 / test 2018–22.
- **What we tried in our code (Grep-verified):** We did NOT attempt the Criticality Prediction classification task. Greps for `criticality`, `swiss_criticality`, `LD-Label`, `citation-label`, `swiss_judgment_prediction` return *zero* hits in our notebook inventory. The only "leading_decision" hits in our code are unrelated enum values inside the court-paragraph descriptor schema (`court_concept_enrichment_robust_v5_t4.md` line 265 — an `AUTHORITY_VALUES` set used by Qwen3 for per-paragraph role labelling).
- **Verdict:** not attempted. We use a different dataset (`rcds/swiss_citation_extraction`, IOB-tagged citations) and a different task (retrieve cited laws/BGE cases per court query) — see `exp_A6_legal_roberta_finetune.md` cell 3. Stern's dataset is a *classification* benchmark over case importance; our task is *citation retrieval* (defined in `swiss-citation-task` SKILL).

### Approach 4 — Multilingual training & evaluation (de/fr/it splits)

- **Paper's claim/result:** Multilingual training + per-language test sets; harmonic-mean macro-F1 across {de, fr, it} × {facts, considerations}; SwissBERT pretrained de-heavy still tops Italian C-F at 39.4 (Table 2).
- **What we tried in our code (Grep-verified):** Our val set is 100% English query translations (per experiments SKILL §"Reranker fine-tuning … train is 99% German, val is 100% English"). We do retrieval *into* a multilingual corpus (de/fr/it law text), and v7.5 uses per-language BM25 indexes (`bm25_search_multilang` in `pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.md` cell containing "Phase 5.E — NEW multi-language BM25 search"). We DO NOT train on multilingual data per the paper's protocol.
- **Verdict:** partially adopted (multilingual *retrieval* via multi-lang BM25). The paper's *training* and *evaluation* protocol on de/fr/it text classification is not run.

### Approach 5 — Hyperparameters from the paper (fixed lr=1e-5, batch 64, max-seq 2048/4096, mixed-precision AMP)

- **Paper's claim/result:** Appendix F — fixed lr=1e-5 (no tuning), batch 64 (gradient accumulation as needed), 2048 token cap on Facts, 4096 on Considerations, early-stop patience 5.
- **What we tried in our code:** exp_A6 uses lr=2e-5, batch 32, max_seq 512, 1 epoch, MultipleNegativesRankingLoss — *not* matched to the paper's classification recipe (because A6 trains a contrastive sentence encoder, not a sequence-classification head). The 512-cap (vs. paper's 2048/4096) is dictated by the base RoBERTa-XL position embeddings (514 incl. specials).
- **Verdict:** N/A — different task, different head, different loss.

---

## Paper 2: UQLegalAI@COLIEE2025 (CaseLink, 2nd place)

**File:** `research_papers/legal_swiss/UQLegalAI_COLIEE2025_arXiv2505.20743.txt`
**Venue / year:** COLIEE 2025 workshop / arXiv 2505.20743
**Authors:** Yanran Tang, Ruihong Qiu, Zi Huang (UQ)
**One-line summary:** CaseLink — a GNN-based legal-case retrieval system that builds a Global Case Graph (Case–Case via BM25 TopK, Case–Charge via charge-name occurrence, Charge–Charge via cosine threshold), encodes nodes with `e5-mistral-7b-instruct`, then trains a GAT/GCN/GraphSAGE with InfoNCE + degree-regularisation. Ranked 2nd in Task 1 with F1 = 0.2962 (1st = JNLP 0.3353).

**Core technical claims:**
- Case-Case BM25 TopK={3,5,10} edges + Charge-Charge cosine-threshold {0.85,0.90,0.95} edges → undirected unweighted GCG.
- LLM text-embedding (`e5-mistral-7b-instruct`, top of MTEB legal) → 4096-token truncation → node features.
- GNN (GAT default; alternatives GCN/GraphSAGE) on the GCG; InfoNCE contrastive loss with 1 positive + 1 easy + {1,5,10} hard negatives mined from BM25 TopK; degree regulariser coefficient {0, 5e-4, 1e-3, 5e-3}; Adam {1e-2, 1e-3, 1e-4}.
- Two-stage inference: BM25 pre-rank to 10 candidates → CaseLink similarity → fix top-5 retrieval per query.
- Year-filter post-processing — exclude candidates whose representative trial date is *after* the query date (precedent must be prior).

---

### Approach 1 — CaseLink Global Case Graph + GNN retrieval (the core method)

- **Paper's claim/result:** F1 = 0.2962 (UQLegalAI Run 3) on COLIEE 2025 Task 1 Canadian Federal Court case-retrieval test set (400 queries, 2 159 candidates).
- **What we tried in our code (Grep-verified):** Greps for `caselink`, `case_link`, `casegnn`, `case_gnn`, `Global Case Graph`, `GAT`, `GCN`, `GraphSAGE`, `GATConv`, `GCNConv`, `torch_geometric`, `import dgl` over the entire `notebooks/` tree return *zero* hits inside any of our notebooks. The only GNN code in the repo is `research_repos/missing_link/src/graph_models.py` — a *third-party* DEXA 2025 paper (Wendlinger et al.) checked out for reference, not run by any of our 90 notebooks.
- **How it compares to paper:** Not comparable — we have no GNN retrieval implementation.
- **Verdict:** **not attempted.** To try: build a heterogeneous graph (BGE case nodes + statute nodes + judgment-chamber/concept nodes) over the unified corpus; encode node text with our existing Qwen3-Embedding-8B fp16 chunks; train a small GAT with InfoNCE + DegReg using the v7.5 pool gold pairs as supervision. The biggest risk is data shape: the COLIEE Task-1 dataset (1 678 train queries, ~7 350 candidates) is small enough for full-batch GNN training, whereas our corpus is 363 k court paragraphs + 175 k law articles, which would need a subgraph-sampling strategy (GraphSAGE-style).

### Approach 2 — `e5-mistral-7b-instruct` as the node-embedding LLM

- **Paper's claim/result:** `e5-mistral-7b-instruct` (Wang et al. 2024) chosen as top legal-retrieval model on MTEB; 4096-token cap; cosine-similarity Sim function.
- **What we tried in our code (Grep-verified):** Greps for `e5-mistral`, `intfloat/e5-mistral`, `e5_mistral` over `notebooks/` return *zero* hits. The model strings appear only inside the paper file itself and inside third-party `research_repos/FlagEmbedding/`. Our embedding stack is:
  - `BAAI/bge-m3` (XLM-RoBERTa-base under the hood) — `exp_A1_bgem3_statutes.md` cell 4: `model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)` → best result stat_recall@500 = 0.215 (10 val queries, 175 k laws_de corpus).
  - `Qwen/Qwen3-Embedding-8B` — used by v7.5 multi-query funnel (canonical pool, R@50 k = 0.893, see experiments SKILL).
- **How it compares to paper:** Different LLM-embedder, similar idea (top-of-MTEB legal model used as the node/document encoder). Qwen3-Embedding-8B is 8 B params (≈ same scale as e5-mistral-7b) and ranks higher than e5-mistral on current MMTEB-R (per memory `reference_qwen3_reranker_multilingual.md`).
- **Verdict:** functional substitute already in place. e5-mistral-7b not separately tested.

### Approach 3 — BM25 two-stage pre-ranking (BM25 → top-K → semantic rerank)

- **Paper's claim/result:** "candidate lists are initially reduced to ten cases using BM25 … top-5 selected by CaseLink score" — §4.4.1.
- **What we tried in our code (Grep-verified):** BM25 is everywhere in our pipeline (39 inventory files match). Most directly comparable to CaseLink's two-stage pattern:
  - `notebooks/02_v75_multiquery_canonical_pool/pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb` — multi-language BM25 (`bm25_search_multilang`, budget 2000 per query split across de/fr/it) feeds the 15-channel pool. Best fused recall: macro R@50 k = 0.893 (experiments SKILL).
  - `notebooks/01_current_direction_cascade_rerank_precision/rerank_hybrid_3model_shootout_with_outputs.ipynb` — three rerankers (Qwen3-Reranker-8B, BGE-reranker-v2-m3, jina-reranker-v2) scored over the v7.5 BM25+vector+enrichment pool. Best macro R@2 k = 0.611 (F0 plain fusion) > 0.568 (F3 hybrid).
- **How it compares to paper:** Our cascade is BM25-feeds-pool *and* dense-feeds-pool *and* enrichment-channels-feed-pool (15 channels, RRF-fused) — strictly broader than CaseLink's BM25-→-CaseLink-rerank cascade. The CaseLink K=10 → top-5 cutoff is much narrower than our budgets (BM25 budget = 2000, final pool ~471 candidates / query at endgame, 46 k / query post-v7.5).
- **Verdict:** broadly adopted (BM25 as the lexical channel in a fusion-and-rerank cascade), in a form that subsumes the paper's two-stage.

### Approach 4 — InfoNCE + Degree-Regularisation contrastive training

- **Paper's claim/result:** L = InfoNCE(positives, easy-neg, hard-neg from BM25 TopK) + α·DegReg(candidate degrees on pseudo-adjacency `A_hat = cos(h_i, h_j)`); α ∈ {0, 5e-4, 1e-3, 5e-3}.
- **What we tried in our code (Grep-verified):** We use InfoNCE-style training only once: `exp_A6_legal_roberta_finetune.md` cell 9 uses `MultipleNegativesRankingLoss` (sentence-transformers in-batch InfoNCE), with hard-negative mining and a positive-vs-HN margin diagnostic. **Degree regularisation is NOT implemented anywhere** (greps for `degreg`, `DegReg`, `degree_reg`, `degree regulariz` return no matches). We have no candidate-graph at training time.
- **How it compares to paper:** Partial — InfoNCE-style contrastive loss tested once (A6) with a 0.659 mean margin between positives and BM25-hard negatives, but the degree term and the graph-conditioning are absent.
- **Verdict:** InfoNCE: tested-and-dropped (A6 contrastive FT failed the stat_recall@500 gate). DegReg: not attempted.

### Approach 5 — Year-filter post-processing (exclude future-dated candidates)

- **Paper's claim/result:** §4.4.2 — given a query, drop candidates whose latest date appearing in the case is after the query date.
- **What we tried in our code (Grep-verified):** Greps for `year_filter`, `date_filter`, `date_before`, `year_before`, `temporal`, `precedent` over the inventory return only `early_recall_preserving_funnel.md`, `investigate_misc_notebook.md`, and `court_llm_descriptor_blackwell_optimized_363k_run.md` — none of these implement a precedence year-filter. The closest signal in our pipeline is `chamber_match` and `doctrinal` features in `rerank_hybrid_3model_shootout_with_outputs.md` cell 4 (Phase 3), but no date-cutoff filter.
- **How it compares to paper:** Not adopted.
- **Verdict:** not attempted. Cheap to add (we have a `year` field on court cases per `auth_cards_*` notebooks and `data_manager` ingestion) but its Task-1 relevance is COLIEE-specific (precedent is a strict legal concept — only prior cases can be cited). For Swiss val/test our gold citations are statutes + BGE cases; statutes don't have a precedent-date semantics, so a year-filter would only help on the BGE case-citation channel.

### Approach 6 — Charge-Charge / Case-Charge edges (heterogeneous graph augmentation)

- **Paper's claim/result:** Charge nodes from the Canadian Federal Courts Act, connected by cosine ≥ θ, and to cases by name-occurrence.
- **What we tried in our code:** No heterogeneous graph. The Swiss analogue would be (statute article ↔ statute article, court paragraph ↔ statute article, court paragraph ↔ court paragraph). Our enrichment system (`court_concept_enrichment_*`, `auth_cards_*`) does produce per-paragraph metadata (concepts, statute-anchors, chamber, authority), but we use these as RRF channels, not as graph edges or GNN-training signals.
- **Verdict:** not attempted as a graph; partial coverage via 15-channel RRF.

---

## Cross-paper synthesis

**What we adopted from these 2 papers:**
- Use of a Swiss-legal-pretrained encoder (`joelniklaus/legal-swiss-roberta-large`, from Stern et al.'s baseline family — Rasiah et al. 2023) inside `exp_A6_legal_roberta_finetune.ipynb`.
- BM25-as-lexical-channel inside a multi-channel cascade (CaseLink §4.4.1's two-stage idea, but broadened to 15 channels and RRF-fused in `pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb`).
- LLM-as-text-embedder for legal retrieval (CaseLink picks e5-mistral-7b; we picked Qwen3-Embedding-8B — same scale, current-MTEB peer).
- InfoNCE-style contrastive sentence-pair training (`MultipleNegativesRankingLoss` in `exp_A6`).

**What we did NOT try and why:**
- Stern's classification task itself (LD-Label / Citation-Label on `rcds/swiss_criticality_prediction`) — out of scope: our task is citation *retrieval*, not case-importance classification.
- The Stern model menagerie (XLM-R Base/Large, SwissBERT, mDeBERTa-v3, MiniLM, DistilmBERT, X-MOD, mT5, BLOOM, Legal-Swiss-LF-Base) — we only tested the largest Swiss-legal RoBERTa.
- The Stern paper's de/fr/it train + per-language evaluation protocol — our val is 100% English queries against a multilingual corpus.
- CaseLink's GNN training stack (GAT/GCN/GraphSAGE, torch_geometric / dgl, Global Case Graph, degree regularisation) — *no graph-based retrieval anywhere in our 90 notebooks*.
- Year-filter post-processing.
- `e5-mistral-7b-instruct` as a direct comparator to Qwen3-Embedding-8B.

**Open gaps where a paper offers an idea we haven't tested:**
1. **GNN over the v7.5 pool (CaseLink-style).** Build a small graph: nodes = (53 k pool unique docs ∪ statute-article nodes ∪ concept nodes); edges = (BM25 TopK case-case, statute-anchor case-statute, concept-co-occurrence statute-statute); features = Qwen3-Embedding-8B vectors; train a GAT with InfoNCE + DegReg on the v7.5 gold pairs. The cascade dossier plan (`research/cascade_dossier_plan.md`) already enumerates "co-citation density" and "statute-target intersection" — these are exactly the *features* the GCG would learn. Worth comparing a learned GNN vs. our hand-crafted RRF channels.
2. **Stern's de/fr/it train-time multilingualism.** Our val is English, our train is 99% German (per experiments SKILL "Reranker fine-tuning"). If we ever LoRA-fine-tune Qwen3-Reranker on the train pool, Stern's Table 2 evidence that XLM-R-Large handles all three languages robustly says the distribution shift may be manageable on the *language* axis (less so on the *citation-count* axis — train median 2 vs. val median 22).
3. **CaseLink's degree-regularisation term.** Even without a full GNN, the DegReg idea ("don't over-rank high-degree hubs") could be applied to our RRF fusion by penalising candidates that get high RRF scores from many channels but are corpus-wide hubs (e.g., Art. 41 OR, BGE 130 III 257). Cheap to bolt on; would target the precision tail.
4. **Stern's algorithmic LD-Label as an authority prior.** The criticality score (`count × (year - 2002 + 1) / 22`) could be precomputed for our 363 k court corpus and used as an additional channel-of-arrival weight in the v7.5 fusion — a "BGE is more likely to be cited if it was historically heavily cited" prior. This is *similar* in spirit to our existing `authority_score` / `frequently_cited_authority` enrichment (`docs/court_rag_plan.md` §line 184).
