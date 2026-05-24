# 04 — Early experiments, early funnel, misc Kaggle utilities, PDF extraction

Scope: 23 notebooks across `11_early_experiments_exp_A/`, `12_early_funnel_pre_v7/`,
`13_misc_kaggle_and_utilities/`, `14_pdf_research_paper_extraction/`.

All paths are absolute. Per-notebook source paths:
`e:\swiss_citation_extraction\notebooks\<subfolder>\<file>.ipynb`. Inventory MDs that
back this analysis: `e:\swiss_citation_extraction\notebooks\_inventory\<basename>.md`.

Metric note: Subfolder 11 reports `stat_recall@K` on **statute-only gold** (BGE/docket gold
filtered out by `is_statute()`). Subfolder 12 reports against full gold (statutes ∪
BGE ∪ docket). These are NOT directly comparable, but the trajectory is what matters.

---

## Subfolder 11 — early experiments exp_A1..A6 (April 21–22, 2026)

This is the **earliest** work on the project. Files load CSVs from
`/content/drive/MyDrive/swiss_law/data/` (`val.csv` = 10 queries, `laws_de.csv` =
175 933 statute articles, `court_considerations.csv` = 2 476 315 rows, plus
`val_translated_de.pkl`). The reproducible artifact path each experiment writes to is
`/content/drive/MyDrive/swiss_law/data/artifacts/exp_A?_report.json`.

### exp_A1_bgem3_statutes.ipynb — 2026-04-21 18:55
- **Goal:** Establish a baseline for BGE-M3 dense retrieval over the 175 933-row Swiss
  statute corpus (`laws_de.csv`). Kill criterion declared in the notebook:
  `stat_recall@500 ≥ 0.40` (either query form). If failed, move to HyDE.
- **Approach:** Encode `citation | title | text` with `BAAI/bge-m3` (fp16, dense only,
  `max_length=512`), build full (10 × 175 933) cosine matrix on GPU, evaluate
  statute-only recall at K∈{50,200,500,1000} for EN queries and German-translated
  queries.
- **Best result:** EN `stat_recall@500 = 0.215`; DE `0.168`; EN @1000 = 0.302.
- **Verdict:** FAIL the 0.40 gate. Recorded conclusion: *"need HyDE"*. This explicit
  failure is what triggered exp_A2.
- This notebook produces `laws_bgem3.npy` (175 933 × 1024 fp16, ~360 MB),
  which is reused as input by exp_A2..A6.

### exp_A2_hyde_and_enumeration.ipynb — 2026-04-21 19:42
- **Goal:** Test whether LLM-based query expansion (HyDE + statute Enumeration) +
  RRF fusion lifts BGE-M3 retrieval past the 0.40 statute@500 gate.
- **Approach:** Qwen2.5-7B-Instruct via vLLM bf16, two prompts per val query:
  HyDE writes 3 hypothetical German `Bundesgesetz`-style paragraphs; Enum writes
  15 expected Swiss federal citations in `Art. X Abs. Y ABBR` format. Embed each
  expansion with BGE-M3, then RRF (k=60) over {en, de, hyde, enum} pair combinations.
- **Best result:** `hyde+enum -> stat_recall@500 = 0.295`, @1000 = 0.362, @50 = 0.107.
  Improvement over A1 (+0.080 at @500). Hallucination check: only **7/251 = 2.8 %** of
  enumerated citations appear verbatim in gold.
- **Verdict:** BELOW the 0.40 gate. Improvement is real but insufficient. Two unsolved
  problems flagged: (a) the LLM hallucinates citation strings, (b) recall@500 below
  the candidate-budget target.

### exp_A3_qwen32b_enumeration.ipynb — 2026-04-21 21:13
- **Goal:** Does scaling the expansion LLM (Qwen2.5-7B → **Qwen3-32B**, thinking off)
  fix the HyDE+Enum quality and push past 0.40?
- **Approach:** Identical prompts to A2, same RRF pipeline, only the LLM changes.
  Required 95 GB VRAM; ran on RTX PRO 6000 Blackwell. Total generation time 319.5 s.
- **Best result:** `hyde+enum -> stat_recall@500 = 0.376`, @1000 = **0.456**,
  hyde-only @200 = 0.262, enum-only @1000 = 0.403. Verbatim gold-in-enum
  hit-rate **6/149 = 4.0 %** (vs 2.8 % for 7B).
- **Verdict:** STILL BELOW the 0.40 gate, but @1000 = 0.456 is the strongest signal
  yet. Larger LLM helps materially (+0.08 over A2). Crucially this notebook produced
  `query_vecs_A3.npz` and `exp_A3_expansions.json` which are the inputs every later
  early experiment depends on.

### exp_A4_rerank.ipynb — 2026-04-21 21:36
- **Goal:** Test whether a cross-encoder rerank (`BAAI/bge-reranker-v2-m3`) on top
  of the A3 hyde+enum 1000-candidate pool can lift early-K precision to the
  `stat_recall@50 ≥ 0.35` floor (sub-25 prediction budget).
- **Approach:** For each query, score 1000 candidates with the cross-encoder using
  (a) the raw DE query and (b) the raw EN query, then re-rank.
- **Best result:** rerank-DE @50 = **0.114**; rerank-EN @50 = **0.067**. The
  unranked baseline (hyde+enum RRF) is 0.161 @50. Reranking with the raw query
  **degrades** the order.
- **Verdict:** FAIL the 0.35 floor and FAIL the "must not be worse than baseline"
  check. The diagnostic finding: a raw legal query and a single statute article
  have asymmetric surface forms; the cross-encoder cannot match them. This motivates
  A4b.

### exp_A4b_rerank_hyde.ipynb — 2026-04-21 21:53
- **Goal:** Same reranker, but rerank using the **HyDE passages** as the query
  side (not the raw question), aggregating with mean/max across 3 HyDE passages.
- **Approach:** Pair each HyDE passage with each of the 1000 candidates; compute
  rerank score; aggregate `mean` and `max` across passages.
- **Best result:** rerank-HyDE-mean @50 = 0.174, @30 = 0.134; rerank-HyDE-max @50
  = 0.174. Baseline RRF @50 = 0.161.
- **Verdict:** PASS the "beat baseline" gate (+0.013 over RRF). FAIL the 0.30
  "useful" floor. This is the **canonical 'cross-encoder is dead for this task'
  measurement**: even with the best query form, rerank only marginally beats no-rerank.
  Findings: the bottleneck is recall, not late-stage reordering.

### exp_A5_court_retrieval.ipynb — 2026-04-22 01:11
- **Goal:** First-ever attempt at the case-citation half of the task. Encode the
  full 2.47 M `court_considerations` corpus with BGE-M3 (fp16, 24 chunks of 100 k
  + one of 76 315), then evaluate (a) case recall, (b) trilingual statute mining
  from retrieved case text.
- **Approach:** 39.5-minute encode → 5.07 GB `court_bgem3.npy`. Top-K retrieval
  done as chunked fp16 GPU matmul (the user invented the trick later reused by
  the v7 pool). RRF over {q_de, q_hyde, q_enum}. Statute mining: DE/FR/IT regex
  over top-200 retrieved case texts, FR/IT abbrevs (`CO`/`CC`/`CP`/...)
  normalized to DE (`OR`/`ZGB`/`StGB`/...).
- **Best result:**
  - Case retrieval: RRF `case_recall@200 = 0.108`, @500 = 0.157, hyde-only
    @500 = 0.176 (best single form). Gate `case_recall@200 ≥ 0.30` → FAIL.
  - Mined statutes: `mined_stat_recall = 0.060` (9/149) at top-200 cases.
    Gate `≥ 0.20` → FAIL.
  - Combined statute (Exp-3 candidates ∪ mined): @500 = **0.409** (target 0.45 →
    still FAIL but the best statute number yet recorded).
- **Verdict:** Dense retrieval over the court corpus is a dead end on its own
  (case_recall@200 ~10 %). Mining statute mentions out of retrieved case text
  yields almost nothing because the cases themselves are not being retrieved.
  Produced reusable artifacts: `court_bgem3.npy`, `court_citations.json`,
  `court_topk_A5.npz`.

### exp_A6_legal_roberta_finetune.ipynb — 2026-04-22 13:11
- **Goal:** Replace BGE-M3 with a fine-tuned **`joelniklaus/legal-swiss-roberta-large`**
  sentence encoder, trained on the `rcds/swiss_citation_extraction` IOB dataset.
  Gate: `stat_recall@500 > 0.376` (must beat A3).
- **Approach:** (1) Wrap RoBERTa-large + mean pooling as a SentenceTransformer.
  (2) Sample 466 833 unlabeled texts (statutes + 300 k court considerations) and
  do 1 epoch SimCSE pre-training (MultipleNegativesRankingLoss). (3) Mine 117 453
  (anchor, positive, hard-negative) triplets from IOB-tagged spans: NER tags
  resolve to statute prefixes (`B-LAW`/`I-LAW` → `Art. N ABBR`) or BGE volumes
  (`B-CITATION`/`I-CITATION` → `BGE V P R`), looked up in `laws_de.csv` and
  `court_considerations.csv`. Hard negatives mined via cached BGE-M3.
  (4) Train 1 epoch, batch 32, lr 2e-5, MNRL, AMP. 35.3 min.
- **Best result:** A6 best `stat_recall@500 = 0.101` (RRF of 4 query forms);
  best @200 = 0.074. Pre-FT baseline was 0.060 @500 (sanity floor). Triplet
  margin in the FT model is clean: positives mean cos=0.5993, hard-negs cos=−0.0597
  (margin 0.66). Yet val performance is **3.7× WORSE than A3 (0.376)**.
- **Verdict:** FAIL both gates. Decision in-notebook: "statute gate FAILED — skipping
  court re-encode." This is the **failed legal-swiss-roberta-large experiment**
  referenced in paper analysis 01. Root cause hypothesis (visible in cell ordering):
  IOB pair quality is poor — 258 088 of 379 565 citations are unresolved against
  the corpus, the resolved pairs use the abstract `Art. N ABBR` parent article
  even when gold is specific (`Abs. 1 lit. b`), and val queries are EN scenarios
  but every training anchor is German legal prose. The encoder optimises a different
  retrieval surface than the val needs.

### Subfolder 11 summary — what fed into v7.5, what was dropped
- **Fed forward:**
  - BGE-M3 as the dense retriever for statutes (A1/A2/A3 establish this).
  - Multi-query expansion (en, de, hyde, enum) + RRF fusion is the basis for the
    later "v7.5 multi-query" pool.
  - Qwen-class LLM for enumeration (A2/A3) — later swapped for Qwen3-8B-AWQ on
    Kaggle for cost.
  - The 4-query-form fp16 chunked GPU matmul pattern (A5) is reused at v7-scale.
  - Trilingual DE/FR/IT statute-mention mining regex (A5 cell 11) is reused.
- **Dropped:**
  - Cross-encoder rerank on raw query (A4) — proven to under-perform baseline.
  - Cross-encoder rerank on HyDE (A4b) — only marginally beats baseline, dropped
    in favour of later Qwen3-Reranker work.
  - **Fine-tuned legal-swiss-RoBERTa (A6) — categorically abandoned.** This is the
    failed fine-tune the paper-analysis docs reference.
  - The "encode 2.47 M court rows once with BGE-M3" approach (A5) is later
    superseded by a Qwen3-8B descriptor enrichment that operates on richer text.
- **Chronology of failure-to-gate:** A1 0.215 → A2 0.295 → A3 0.376 → all under the
  declared 0.40 floor. The implication, accepted later, is that the bottleneck is
  not the retriever; it's the citation/query lexical asymmetry. This is what
  motivated the funnel/lattice approaches in subfolder 12.

---

## Subfolder 12 — early funnel pre-v7 (April 26–30, 2026)

### early_dense_embedding_test.ipynb — 2026-04-26 18:16
- **Goal:** "Maximum-context ceiling" test — settle the question of whether ANY
  dense embedding model with EVERY auxiliary signal (citation header + structural
  metadata + outgoing-reference graph + 1600 char text) can hit Macro F1 in the
  0.60–0.80 target window on val.
- **Approach:** Concat `laws_de.csv` (175 933) + `court_considerations.csv`
  (1 985 178 dedup'd) → **2 161 111-row unified corpus**. Build "enriched passage"
  = `[CITATION] x | [STRUCTURE] family;subfamily;pattern;segments | [REFERENCES] up
  to 10 outgoing refs | [TEXT] up to 1600 chars`. Encode with
  **`Qwen/Qwen3-Embedding-8B`** (bf16, sdpa, `max_seq=768`, batch 256, torch.compile
  reduce-overhead), 4096-dim, FAISS IndexFlatIP. Apply Qwen3-Embedding instruct
  prefix only to queries. 22 chunks × 100 k passages (35.4 GB of doc embeddings).
- **Best result:**
  - Macro F1 oracle-k = **0.041**, F1@20 = **0.044**, F1@30 = 0.034.
  - Mean Recall@100 = 0.117, Recall@500 = 0.247, Recall@1000 = 0.289.
  - val_006 (heating-installer fact pattern) is the strongest at F1=0.111, R@500=0.500.
  - val_001 (Untersuchungshaft) is at zero for K≤100.
- **Verdict:** Hard FAIL of the 0.60 target. Notebook's printed conclusion: *"Dense
  embedding INSUFFICIENT — Observation 3 CONFIRMED with maximum context. … Even SOTA
  + every available signal cannot bridge the legal-conceptual gap."* This is the
  empirical death certificate for "just embed it better" approaches and motivates
  the structured funnel that follows.
- This is the **first dense-embedding test on this task** — Qwen3-Embedding-8B on
  the unified 2.16 M corpus.

### early_recall_preserving_funnel.ipynb — 2026-04-27 02:56
- **Goal:** Build a recall-preserving structural funnel: instead of one giant
  embedding match, parse each val query into law/BGE/docket mentions and use
  pre-computed citation-graph indexes + an LLM planner to pick law codes and
  docket prefixes. Recall is the only allowed loss.
- **Approach:** DuckDB-backed pipeline. Build `laws_de_links.json` and
  `court_considerations_links.json` into a `citation_edges` table. Plan stages:
  Qwen3-32B planner emits `{law_codes, law_unit_chains, court_families,
  bge_divisions, docket_prefixes, routes, time_mode, law_budget, court_budget}`.
  Hard recall floors enforced (EARLY ≥ 0.98, MID ≥ 0.95, FINAL gate).
  Routes: `bge_base_route`, `law_code_route`, `statute_citing_cases`. Up to
  2000 law + 2000 court candidates.
- **Best result (on val):**
  - Median candidate pool = 1879–4000 rows.
  - Per-query recall (full gold, statute+case): val_001=0.381, val_002=0.25,
    val_003=0.383, val_004=0.30, val_005=0.182, val_006=0.333, val_007=0.579,
    val_008=0.25 (visible from "gold law/court … recall …" lines). law-side
    routinely 0.45–0.84; court-side typically 0.00.
- **Verdict:** Demonstrates the funnel CAN preserve recall on the statute side
  cheaply (no embedding needed), but court-side recall is essentially 0. This is
  the architectural predecessor of `anchor_funnel_v1` — both share the LLM
  planner, the "law codes + docket prefixes" stage, the recall-floor gating
  pattern, and the `choice_cards.json` artifact name.

### early_segment_lattice_funnel_v3.ipynb — 2026-04-30 15:10
- **Goal:** Generalise the funnel to a **segment lattice**: instead of hand-coded
  routes, all citation segments (law_code, article, BGE division, docket_prefix,
  legal_area, year, …) become "choice cards" the LLM (or an oracle) selects from.
  Build a SQLite-backed normalised index + co-signal stats so the LLM has full
  evidence per choice. Re-use `segment_lattice_v3.py` script.
- **Approach:** `build_index` scans 197 945 law rows + 2 416 056 court rows;
  builds 48 card groups across `laws_de.*` and `court_considerations.*` axes.
  Court co-signals computed from 2 476 315 court rows scanned. Shrinks to
  `choice_cards_v3_compact.json` with caps per segment (laws law_code top-80,
  court docket_prefix top-30, etc.). Both an **oracle** (cheats with gold to set
  ceiling) and an **LLM planner** run.
- **Best result:** On val (10 queries):
  - Oracle: mean recall 0.396 (median 0.295), law_recall 0.613, court_recall 0.034.
  - LLM planner: mean recall 0.402 (median 0.338), law_recall 0.623,
    court_recall 0.000. Per-stage diag: `topk_budget` stage drops most of the
    law-side recall (mean=0.623 from 1.0 source).
- **Verdict:** ABANDONED. The lattice approach is more general and the LLM
  *matches* oracle on the law side (~0.62), but **court_recall is still 0**, mean
  recall is 0.40 (gate is much higher), and the per-stage trace shows even the
  oracle plateaus at law_recall=0.613 inside the 2000-row budget. The conclusion
  (consistent with later choices) is that lexical/structural funnels can fix
  statute retrieval but not case retrieval, and the budget structure that the
  lattice enforces is too rigid. The successor work pivots to anchor_funnel_v1
  (which keeps the planner+choice-cards pattern but loosens routing) and then
  v7+v7.5 (which gives up trying to be recall-preserving on courts and instead
  uses multi-query dense pooling at K=50 k).

### Subfolder 12 summary
- **Chronology:** dense-embedding test (4/26) → recall-preserving funnel (4/27) →
  segment-lattice v3 (4/30) → (next, in another subfolder) anchor_funnel_v1.
- **Definitive version:** none — every notebook here is a probe that informed the
  next architecture. The lineage from "recall_preserving_funnel" to
  `anchor_funnel_v1` is the load-bearing transition.
- **Abandoned items:** segment-lattice v3 (too rigid budgets, court_recall=0).
  The "encode unified corpus once and retrieve" approach (Qwen3-Embedding-8B on
  2.16 M docs, F1=0.04). The Qwen3-32B-planner-as-router pattern itself survives
  into later notebooks but with relaxed budgets and additional anchor signals.

---

## Subfolder 13 — misc Kaggle utilities (May 2–3, 2026)

These are infrastructure notebooks for running Qwen3-8B-AWQ on Kaggle 2×T4 GPUs to
produce the LLM enrichment ("`rag_enrichment`"/"`court_authority_cards`"). They are
**all the same problem**: get vLLM + AWQ + (optionally Triton or FlashInfer) running
stably on Kaggle inside 9-hour sessions for the 363 258-row enrichment job. Most
produce no measurable retrieval metric — they're build-system iterations.

### investigate_misc_notebook.ipynb — 2026-04-29 13:11
- **Misnamed file**: this is actually a **Kaggle Retroviral Challenge (FoldSeek + RT)**
  notebook investigating Prime-Editing reverse-transcriptase activity using LOFO
  validation, FoldSeek TM scores, ESM embeddings, family-bootstrap stability tests.
  References `/kaggle/input/competitions/retroviral-challenge-predict/`. Has nothing
  to do with Swiss legal citation extraction. Should be flagged as
  out-of-scope for this project (likely cross-contaminated when reorganising
  notebooks). No legal-retrieval signal can be extracted from it.

### kaggle_10_test_for_swiss_law.ipynb — 2026-05-02 21:44
- **Goal:** 10-row smoke test of the Qwen3-8B-AWQ enrichment schema on Kaggle 2×T4.
- **Schema being tested:** `rag_enrichment` with fields `concepts_en`,
  `terms_original`, `legal_area`, `topic`/`subtopic`/`micro_topic`,
  `statute_anchors`, `case_anchors`, `specificity_score`, `enrichment_quality`,
  `paragraph_role`, `authority_role`. QC enforces NO forbidden fields
  (`query_phrases_en`, `summary_en`, `natural_language_queries`, `english_summary`).
  All 10 rows passed `enrichment_quality.generation_status = ok`, 0 forbidden
  fields, all `specificity_score ≥ 0.85`.
- **Verdict:** PASS smoke test. Locks the enrichment schema that the 363 k full
  run uses.

### kaggle_notebook_3500ee7363_base.ipynb — 2026-05-02 20:54
- **Goal:** Earlier Kaggle iteration on the same 10-row smoke test. Uses vLLM with
  `gptqmodel` + AWQ.
- **Result:** All 10 rows **failed** with `ValueError('No balanced JSON object
  found …')` — JSON parsing of model output ended mid-record. 0 OK / 10 failed in
  55.1s. Produced no useful enrichment.
- **Verdict:** Debug iteration; pre-FlashInfer-removal. Superseded by the
  `_safe_` and `_kaggle_stable` variants.

### kaggle_safe_notebook_3500ee7363.ipynb — 2026-05-02 21:07
- **Goal:** Safer Kaggle T4 preset of the prior notebook (`enforce_eager=True`,
  `gpu_memory_utilization=0.62`, `max_model_len=3072`, batch 4, Triton attention,
  FlashInfer uninstalled). 10-row smoke test.
- **Verdict:** Run never completed inference in the captured outputs (cells
  defined but the executed evidence in the inventory stops at the engine setup).
  Schema flattening logic enriches every field downstream consumers want.

### kaggle_qwen3_awq_text_to_json_no_flashinfer_with_fallback.ipynb — 2026-05-02 20:49
- **Goal:** Same 10-row enrichment, but with explicit Transformers-fallback path if
  vLLM fails to start.
- **Verdict:** Setup-only — no measurable inference output in captured cells.

### kaggle_notebook_0cf83146ef.ipynb — 2026-05-03 13:38
- **Goal:** "Raw descriptor only" enrichment variant of the smoke test. QC forbids
  retrieval-side fields (`statute_anchors`, `case_anchors`, `retrieval_views`,
  `outcome_signal`, …) — this notebook explicitly produces only the descriptor
  layer, not the retrieval-anchor layer.
- **Result:** 50 BGE rows enriched, all 50 status=ok, 0 forbidden fields, 0 failed,
  `concept_count` 3–5, `specificity_score` ∈ {0.3, 0.8}.
- **Verdict:** This separates the descriptor schema from the anchor schema — a
  schema decision that survives into the full run.

### kaggle_notebook_d37248e1eb.ipynb — 2026-05-02 19:55
- **Goal:** Full-corpus runner skeleton (TOTAL_V4_LINES = **2 476 315**, NOT the
  363 k subset). Forces `VLLM_ATTENTION_BACKEND=TRITON_ATTN` to bypass FlashInfer
  bug, uses spawn multiproc, checkpointed resume-on-restart loop. 4 tuning presets
  for cards/s.
- **Verdict:** Engineering infrastructure — no measurement, but the ETA cell
  ("at 3/5/8/12 cards/s") shows this is designed for the full 2.47 M-row v4 run
  across multiple Kaggle sessions. Likely became the basis for the actual
  363 k-row production runner.

### kaggle_qwen3_8b_awq_local_batchfix.ipynb — 2026-05-02 05:20
- **Goal:** Pure-Transformers (no vLLM) AWQ enrichment on the **363 258**-row
  `court_authority_cards_v4_target_cards.jsonl` subset. Uses `gptqmodel` for AWQ.
- **Result:** Failed with `RuntimeError: AWQ load failed. Run: !pip install
  autoawq …` — autoawq backend mismatch on Kaggle.
- **Verdict:** FAILED iteration. The 363 258 number is the canonical
  subset size that the eventual production run targets (≠ the full 2.47 M).

### kaggle_qwen3_8b_awq_vllm_text_to_json_base.ipynb — 2026-05-02 05:30
- **Goal:** First vLLM-based runner for the 363 258 RAG-targets subset. No
  Triton override.
- **Verdict:** Setup-only in captured outputs.

### kaggle_qwen3_8b_awq_vllm_text_to_json_t4_fixed.ipynb — 2026-05-02 05:40
- **Goal:** Same as `_base`, with 2×T4 TP=2 + `disable_custom_all_reduce=True`
  preset for Kaggle multi-GPU stability.
- **Verdict:** Setup-only in captured outputs.

### kaggle_qwen3_8b_awq_vllm_text_to_json_t4_flashinfer_fixed.ipynb — 2026-05-02 05:44
- **Goal:** Adds the FlashInfer workaround (`VLLM_ATTENTION_BACKEND=TRITON_ATTN`
  before vLLM import). Same 363 258-card target.
- **Verdict:** Setup-only in captured outputs.

### kaggle_qwen3_8b_awq_vllm_text_to_json_stable.ipynb (≡ "_kaggle_stable") — 2026-05-02 05:48
- **Goal:** "Stable" Kaggle preset combining everything: Triton attention before
  vLLM import, `spawn` multiproc, `OMP_NUM_THREADS=1`, plus the SAFEST/FASTER
  tuning presets at the end (gpu_util 0.55→0.70, batch 48→96).
- **Verdict:** Setup-only in captured outputs. This file is the most likely
  starting point for the production 363 k run.

### Subfolder 13 summary — chronology and which one produced the 363 k enrichment
- **Two parallel debug threads:**
  1. `3500ee7363_base → 0cf83146ef → d37248e1eb → kaggle_safe → corrected_kaggle_safe → kaggle_qwen3_8b_awq_kaggle_local` — the original schema iteration. Notebook 3500ee7363 visibly failed (JSON parse errors); kaggle_safe locked the safe T4 preset; the kaggle_local one (not in this scope but in subfolder 13's `auth_cards_enrich_qwen3_8b_*` cluster) is the production target.
  2. `vllm_text_to_json_base → t4_fixed → t4_flashinfer_fixed → stable` — the vLLM stability thread, all 4 share `court_authority_cards_v4_target_cards.jsonl` as input.
- **Which one actually produced the 363 k enrichment:** None of these 12 notebooks
  show a successful 363 k run in their captured outputs — they're all the
  prerequisite debug iterations. The actual production run lives in the
  `auth_cards_enrich_qwen3_8b_*` cluster (out of scope here; covered by another
  agent's analysis of `notebooks/_inventory/auth_cards_enrich_qwen3_8b_kaggle.md`
  and siblings). The most likely *direct ancestor* of the production runner is
  `kaggle_qwen3_8b_awq_vllm_text_to_json_stable.ipynb`, which is the only one
  combining ALL the Kaggle fixes (Triton attention, spawn, AWQ via vLLM, TP=1
  fallback, batch-tuning presets).
- **Out-of-scope file:** `investigate_misc_notebook.ipynb` belongs to a different
  Kaggle competition (Retroviral Challenge) and was misfiled into this folder.

---

## Subfolder 14 — PDF research-paper extraction (May 12, 2026)

### pdf_extraction_marker_pymupdf_pdfminer.ipynb — 2026-05-12 00:32
- **Misclassified file:** Folder name says "PDF research paper extraction", but the
  notebook content is for the **Make Data Count Kaggle Challenge** — extracting
  in-text DOI / dataset-ID spans from research-paper PDFs in
  `/content/drive/MyDrive/Make_data_count_challenge/Data/train/PDF/`.
  Schema produced: `(article_id, dataset_id, dataset_doi_norm, found,
  matched_doi, n_sections_with_any_doi, markdown_path,
  Section_reference_is_coming_from)`.
- **Goal:** Mine DOI mentions from PDFs using a layered toolchain — Marker (deep
  ML PDF→markdown), PyMuPDF (`fitz`), pdfminer.six, PyPDF2 — with section-bucket
  inference (introduction / methods / data-availability / references) per match.
  Uses 4-way threaded shards on `(article_id, dataset_id)` pairs from
  `train_labels_cleaned.csv`.
- **Approach (skim of the 9 code cells):**
  - Cell 1: install `PyPDF2`.
  - Cell 2 (large, commented-out): an in-text-span miner with DA-block detection,
    DOI canonicalisation (`https://doi.org/<core>`), unicode-safe canon,
    pdfminer + PyPDF2 page-view comparison.
  - Cell 3+: Marker pipeline that converts PDF → markdown into `marker_md/`,
    then matches dataset DOIs against the markdown sections to populate
    `Section_reference_is_coming_from`.
  - Final cell runs 2 parallel GPU threads (`gpu_threads=2`).
- **Result captured:** Marker attempted to download models (`text_recognition`,
  `vocab_math.json`, …). At least one PDF failed with `PdfiumError: Failed to
  load document (PDFium: Data format error)`. No final precision/recall numbers
  emitted; the notebook ends mid-execution.
- **Verdict:** Out of scope for the Swiss citation extraction project. This is
  competition-side tooling for the Make Data Count challenge, accidentally
  organised under `14_pdf_research_paper_extraction/`. There is no link to
  the Swiss `research_papers/` directory (paper-analysis 01/02 references), no
  Swiss `.txt` output, and no produced artifact under
  `e:\swiss_citation_extraction\`.

### Subfolder 14 summary
- Single notebook, mislabelled folder. Workflow: Marker (PDF→markdown) +
  pdfminer/PyPDF2 fallback + DOI regex + section-bucket inference + 4-way thread
  shard runner.
- No Swiss legal output produced.

---

## Cross-cutting findings

1. **Original (exp_A1..A6) approach** = BGE-M3 + Qwen LLM query expansion (HyDE +
   Enum) + RRF fusion. **Best end-to-end statute result was A3 at 0.376 @500**,
   still under the project's own 0.40 gate. Cross-encoder rerank (A4, A4b) does
   not solve it. Fine-tuning legal-swiss-RoBERTa-large (A6) makes things ~3.7×
   WORSE — this is the recorded failed-fine-tune in `paper_analysis/01_*`.
2. **What carried into v7.5:**
   - BGE-M3 (later replaced by Qwen3-Embedding-8B per the unified-corpus test).
   - 4-query-form generation (en / de / HyDE / Enumeration) — the multi-query
     pool's core idea.
   - RRF fusion over rankings.
   - Trilingual DE/FR/IT statute mining regex (A5 cell 11).
   - LLM planner + choice-cards pattern (from recall_preserving_funnel).
   - Chunked GPU fp16 top-K matmul over court corpus (A5).
3. **What was retired:**
   - Cross-encoder rerank using the raw query.
   - Fine-tuning the encoder.
   - "Maximum-context dense embedding" theory of the problem (early_dense_embedding_test
     killed it).
   - Hard-budget segment-lattice planning (early_segment_lattice_funnel_v3).
4. **Misc Kaggle thread** is purely infrastructure (no retrieval metric).
   The production 363 258-row enrichment is NOT inside this batch of 12
   notebooks — they are the prerequisite debug stack. The most-complete preset
   is `kaggle_qwen3_8b_awq_vllm_text_to_json_stable.ipynb`.
5. **Two misfiled notebooks** in this scope:
   - `investigate_misc_notebook.ipynb` is from the Kaggle Retroviral Challenge.
   - `pdf_extraction_marker_pymupdf_pdfminer.ipynb` is from the Make Data Count
     Challenge.
   Neither contributes to Swiss citation extraction; both should be flagged for
   re-organisation.
