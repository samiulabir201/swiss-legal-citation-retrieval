# Swiss Legal Citation Retrieval

Cross-lingual retrieval over **2 652 248 Swiss legal documents** in German, French and Italian — given an English fact-pattern, return the exact set of statute articles and federal-court considerations that a legal expert would cite.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Task definition

| Property | Value |
| --- | --- |
| **Input** | One English natural-language fact-pattern per query |
| **Output** | A set of exact citation strings drawn from a closed vocabulary |
| **Corpus** | 175 933 Swiss law articles (`laws_de.csv`, German) + 2 476 315 federal-court paragraphs (`court_considerations.csv`, German / French / Italian) — total 2 652 248 documents |
| **Splits** | `train.csv` (1 139 queries, 99 % German, median 2 gold) · `val.csv` (10 queries, 100 % English, median 22 gold, range 10–47) · `test.csv` (40 queries, 100 % English, no gold released) |
| **Metric** | Macro F1 over the per-query gold set |
| **Target** | Macro F1 ∈ [0.6, 0.8] on `val.csv` |

A naïve baseline (e.g. BM25 on the English query → German corpus) scores roughly zero because the EN ↔ DE noun overlap is only 4.3 %. A single-channel dense embedding (Qwen3-Embedding-8B, #1 MMTEB-Multilingual at the time of measurement) caps at **R@1000 = 0.289** on `val.csv` — measured, not assumed. This corpus is therefore not a pure dense-retrieval problem; the architecture has to inject explicit legal-domain structure.

---

## Headline results

Each result below is a self-contained card stating **what was measured · on which split · with which method · in which notebook**, plus any caveat that materially affects interpretation. Results are grouped into three sections: retrieval-pool recall (A), end-to-end Macro F1 (B), and enrichment / infrastructure (C).

### A — Retrieval-pool recall (does the candidate pool *contain* the gold?)

#### A1. Macro recall of gold citations in the top-50 000 candidate pool

- **Value:** R@50K = **0.893** (macro)
- **Eval split:** `val.csv`, n = 10
- **Method:** 15-channel anchor / BM25 / dense / graph funnel + weighted Reciprocal Rank Fusion + 7-channel round-robin guarantee (130 slots × 7 channels = 910 reserved slots before RRF fills the tail of `topk_final = 50 000`).
- **Caveat:** Per-query recall ranges from **0.766 (val_003)** to **1.000 (val_004, val_005)**. The 0.766 floor on val_003 is a structural ceiling — gold citations not in the union of all 15 channels cannot be reached by any downstream reranker or judge.
- **Source notebook:** [`pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb`](notebooks/02_v75_multiquery_canonical_pool/)

#### A2. Dense-only recall ceiling (R@1000)

- **Value:** R@1000 = **0.289** (macro)
- **Eval split:** `val.csv`, n = 10
- **Method:** Qwen3-Embedding-8B brute-force cosine similarity over the full 2 652 248-document corpus. No fusion, no rerank, no judge.
- **Caveat:** Establishes that single-channel dense retrieval *cannot* solve this task — a 4096-dim embedding cannot encode the kind of inter-citation relationship that an expert annotator uses. This is the empirical evidence that motivates the multi-channel design in A1.
- **Source notebook:** [`notebooks/05_embedding_and_retrieval_base/`](notebooks/05_embedding_and_retrieval_base/)

### B — End-to-end Macro F1 (the headline numbers, with full attribution)

#### B1. Law-only hybrid pipeline (the "v12 hybrid" baseline)

- **Value:** Macro F1 = **0.777** on val (P = 0.845, R = 0.748, avg K = 13)
- **Eval split:** `val.csv`, n = 10 — **law-only**, court considerations excluded
- **Method:** BM25 top-100 → Qwen3-Reranker-8B → fused score `0.7·BM25 + 0.3·rerank` → zone-split (`HIGH = 0.55`, `LOW = 0.25`) → Qwen3-8B judge on the borderline zone only.
- **Caveat — this is val-overfit, not a clean result:** the same pipeline scores Macro F1 = **0.296 on `train.csv`**, a delta of +0.481 between val and train. The notebook itself flags this in cell 7 as `warning`. Only the `thresh_only` and `bm25_K3` variants generalise across splits.
- **Source notebook:** [`law_only_hybrid_v12_val_macroF1_0.777_overfit_warning.ipynb`](notebooks/06_law_only_bm25_rerank_judge_baseline/law_only_hybrid_v12_val_macroF1_0.777_overfit_warning.ipynb)

#### B2. End-to-end pipeline on the full 2.65 M-doc corpus (the "endgame" run)

- **Value:** Macro F1 < **0.10** on val
- **Eval split:** `val.csv`, n = 10 — full DE + FR + IT corpus including court considerations
- **Method:** The multi-query candidate pool from A1 → Qwen3-Reranker-8B → Qwen3-8B judge.
- **Caveat — failure-mode diagnosis, not a model-quality result:** The judge had two compounding bugs — default-YES on parse failure, and `<think>` chain-of-thought consuming most of the token budget before reaching `VERDICT:`. The combined effect was ≈ 87 % rubber-stamp YES verdicts. Recall, not precision, is the binding bottleneck downstream. Full diagnosis in [`research/endgame_handoff_2026-05-09.md`](research/endgame_handoff_2026-05-09.md) §4.
- **Source notebook:** [`notebooks/04_pre_v75_pipeline_iterations/`](notebooks/04_pre_v75_pipeline_iterations/)

### C — Enrichment & infrastructure (the building blocks)

#### C1. Law-article LLM-enrichment coverage

- **Value:** **173 033 / 175 933 articles enriched (98.3 %)**
- **Dataset:** All law articles in `laws_de.csv`
- **Method:** Qwen3-8B-AWQ generating an English-aligned descriptor schema per article: `english_summary`, `concepts_en`, `legal_question`, `applicability_conditions`, `provision_role_llm`, `specificity_score`.
- **Source code:** [`scripts/enrichment/`](scripts/enrichment/) + [`notebooks/07_law_de_enrichment/`](notebooks/07_law_de_enrichment/)

#### C2. Court-paragraph LLM-enrichment coverage

- **Value:** 363 258 / 2 476 315 paragraphs enriched (**≈ 14.7 %**, partial coverage)
- **Dataset:** Court considerations in `court_considerations.csv`
- **Method:** Same Qwen3-8B-AWQ model; per-paragraph schema includes a 15-role taxonomy + 7-element doctrinal template (rule-statement, application, lower-court summary, party-position, etc.).
- **Source notebook:** [`notebooks/08_court_llm_descriptor_extraction/`](notebooks/08_court_llm_descriptor_extraction/)

#### C3. Citation-graph gold-node coverage

- **Value:** **92.9 % → 96.0 %** (+3.1 pp)
- **Dataset:** All gold-citation entities across `train.csv` + `val.csv`
- **Method:** Patched the citation-extraction regex with ATF / DTF / `c.` / `consid.` / space-tolerant docket forms. The remaining 4 % gap is corpus-side (codes not present in `laws_de.csv`, e.g. LugÜ, FIDLEG, FINIG, FinfraG), not regex-side.
- **Source code:** [`scripts/citation_extraction/extract_citation_graph.py`](scripts/citation_extraction/extract_citation_graph.py)

#### C4. Granularity-resolver gold recovery

- **Value:** **71.5 % → 98.1 %** gold-in-corpus on train (+26.6 pp); val unchanged at 100 %
- **Dataset:** `train.csv` gold-citation set
- **Method:** Bare `Art. N LAW` ↔ paragraph-children (`Abs.`) fanout, validated against `laws_de.csv`. Critical because train gold cites bare articles whereas the corpus stores Absatz-granular rows.
- **Source code:** [`scripts/data_prep/granularity_resolver.py`](scripts/data_prep/granularity_resolver.py)

#### C5. Dense-vector query latency

- **Value:** **≈ 50 ms per query**
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell, 95.6 GB VRAM
- **Method:** Full-GPU brute-force `E @ q` against the 2 652 248 × 4096 fp16 corpus matrix (21 GB), all resident in VRAM. No FAISS, no IVF — Blackwell makes the naive matmul fast enough that ANN structures aren't needed.
- **Source code:** [`scripts/retrieval_and_rerank/encode_queries_qwen3_8b.py`](scripts/retrieval_and_rerank/encode_queries_qwen3_8b.py)

---

## Architecture

```
              English query (10–47-citation gold set)
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
  ┌────▼─────┐      ┌─────▼──────┐     ┌─────▼─────┐
  │ Statute  │      │  HyDE +    │     │ Structured│
  │ + case   │      │  German    │     │ query     │
  │ anchors  │      │  expansion │     │ expansion │
  └────┬─────┘      └─────┬──────┘     └─────┬─────┘
       │                  │                  │
       └────────┬─────────┴───────┬──────────┘
                │                 │
        ┌───────▼────┐    ┌───────▼────────┐
        │ BM25 FTS5  │    │ Qwen3-Embed-8B │
        │ (DE/FR/IT) │    │ brute-force GPU│
        └───────┬────┘    └───────┬────────┘
                │                 │
                └────────┬────────┘
                         │
              ┌──────────▼───────────┐
              │   Reciprocal Rank    │
              │   Fusion (RRF)       │
              └──────────┬───────────┘
                         │
             ┌───────────▼────────────┐
             │ Qwen3-Reranker-8B      │
             │ (cross-encoder, K≤500) │
             └───────────┬────────────┘
                         │
             ┌───────────▼────────────┐
             │ Qwen3-8B AWQ judge     │
             │ (default-NO, 7-cat)    │
             └───────────┬────────────┘
                         │
                    Citation set
```

**Why each piece is there:**

- **Anchor channels (statute / case)** — when a query names a code ("Art. 95 BGG") the answer is partly deterministic; the anchor channel bypasses the noisier embedding step.
- **HyDE + structured query expansion** — Qwen3-8B generates a German hypothetical answer and a structured `{legal_domain, applicable_codes, german_terms, concept_seeds}` JSON, then 3–5 sub-queries are issued in parallel. Bridges the 4.3 % EN ↔ DE vocab gap.
- **Sparse (BM25) + dense (Qwen3-Embedding-8B) union** — neither channel alone hits 50 % recall on `val.csv`; their RRF fusion does.
- **Cross-encoder rerank** — only applied at K ≤ 500. The reranker shootout (Qwen3-Reranker-8B vs BGE-v2-m3 vs jina-v2-base-multilingual) showed plain RRF beats every reranker above K = 500.
- **LLM judge with default-NO** — the law-only v12 baseline (row B above) defaulted to YES on parse failure and produced ≈ 87 % rubber-stamp YES verdicts when applied at full corpus scale. The current judge disables `<think>`, caps `max_new_tokens = 24`, and defaults to NO on any parse failure.

---

## Engineering decisions worth highlighting

These came out of empirical work, not literature search.

1. **The embedding-similarity ceiling was measured, not assumed.** Qwen3-Embedding-8B on the full 2.65 M-doc corpus capped at R@1000 = 0.289 on `val.csv`. The pipeline therefore treats dense retrieval as one channel of fifteen, not the spine.
2. **The citation graph is a tie-breaker, not a co-predictor.** Measured: **99.41 %** of gold-citation pairs have *no graph edge*. The graph is used downstream of scoring, not to expand the candidate pool.
3. **Variable-K is mandatory.** Gold set sizes on `val.csv` range 10–47; fixed-K is provably sub-optimal. K is calibrated per query from cheap features (statute-target count, query length, court/law ratio).
4. **No training-set-derived priors.** `train.csv` is 99 % German with median gold size 2 and 1.2 % court share; `val.csv` and `test.csv` are 100 % English with median gold 22 and 40 % court. The three axes of distribution shift make any train-derived weight a leakage hazard. The `train_val_law_gold_research_2026-05-23` study quantifies this.
5. **Granularity-aware gold matching.** Swiss articles fan out into Absatz children (`Art. 11 OR` → `Art. 11 Abs. 1 OR`, `Abs. 2`, …). A bare-article ↔ paragraph-child resolver lifts gold-in-corpus from 71.5 % to 98.1 % on `train.csv`.
6. **The law-only F1 = 0.777 result is val-overfit.** The same v12 thresholds (`HIGH = 0.55, LOW = 0.25`) score F1 = 0.296 on `train.csv`. This is documented in the source notebook itself; the README quotes it explicitly rather than presenting 0.777 as a clean win.

The [research/](research/) directory contains the canonical write-ups of these findings, notably [`endgame_handoff_2026-05-09.md`](research/endgame_handoff_2026-05-09.md), [`personal_observations.md`](research/personal_observations.md), and [`law_gold_retrieval_mechanism_2026-05-23.md`](research/law_gold_retrieval_mechanism_2026-05-23.md). Supporting analyses live under [`analysis/`](analysis/) (citation graph, citation patterns, gold-citation coverage).

---

## Data & infrastructure (what was built and why)

The pipeline runs on a small number of purpose-built indexes. Each one is described below — what's inside, how it was built, the analyzer / schema decisions, and the recall / latency it enables.

### 1. Unified retrieval store — SQLite + FTS5 (24 GB)

A single SQLite file (`artifacts/unified_retrieval.sqlite`) holds the entire corpus and every sparse signal needed at query time. Schema:

- **`documents`** — 2 652 248 rows (175 933 laws + 2 476 315 court paragraphs). Each row carries `doc_id`, `family` (`law` / `court`), `language` (`de` / `fr` / `it`), `text`, structural metadata (chapter / heading / paragraph role), and a `text_for_fts` field that concatenates the searchable fields.
- **`documents_fts`** — FTS5 virtual table over `text_for_fts` with the **`unicode61` tokenizer + Snowball stemming** for German / French / Italian. Custom legal-lexicon expansion is applied at index time (canonical abbreviation forms `StPO`, `ZGB`, `BGG` mapped to their variants, ATF ↔ BGE ↔ DTF, Absatz / al. / cpv. paragraph markers normalised).
- **`statute_links`** — 5.30 M extracted statute-citation edges (`doc_id → cited statute`).
- **`case_links`** — 1.81 M case-citation edges (`doc_id → BGE/ATF docket`).
- **`adjacent_law_links`** — 348 K sibling links (within-code neighbours).

**Why SQLite + FTS5 over Elasticsearch / OpenSearch.** The deciding factor was operational simplicity: a single file is reproducible, fits the offline-Kaggle inference constraint (≤ 12 h runtime, no external services), and FTS5 supports the language-specific stemming and custom-tokeniser logic the task needs. The same schema would port directly to Postgres `tsvector` with `pg_trgm` if a production deployment required it; nothing about the index design is SQLite-specific.

**Build path.** [`scripts/build_pipeline/build_unified_retrieval_corpus.py`](scripts/build_pipeline/build_unified_retrieval_corpus.py) constructs the rows from the three input CSVs and merges in the LLM-enrichment JSONL files (described below). One full rebuild = ~3 hours; incremental rebuilds of single families = ~20 minutes.

### 2. Dense-vector index — Qwen3-Embedding-8B fp16 (21 GB)

The entire corpus is encoded once with `Qwen/Qwen3-Embedding-8B` (4096-dim, the #1 model on MMTEB-Multilingual at the time of build) and stored as 27 fp16 chunks under `embeddings/qwen3_8b_unified_chunk000..026.npy`. A `manifest.parquet` maps `(chunk_id, row_index)` → `doc_id`.

At query time the matrix lives entirely in GPU VRAM and dense retrieval is a single matmul: `E_GPU @ q`. Brute-force search **on Blackwell ≈ 50 ms / query** — fast enough that a FAISS-IVF approximate index was tested and rejected: the recall loss from quantisation / centroid mismatch wasn't worth the 5–10× latency saving.

The embedder runs with the recommended `Instruct: ... Query: ...` template; query and document side use distinct prompts. Pooling is last-token + L2-normalised. Encoding cost: ~6 hours on one Blackwell for the full 2.65 M-doc encode.

### 3. LLM enrichment pipeline — Qwen3-8B-AWQ on Kaggle GPUs

Both families are enriched with structured English-aligned descriptors generated by `Qwen/Qwen3-8B-AWQ`. The AWQ quantisation drops VRAM use from 16 GB → ~5 GB and roughly halves wall-clock, letting the entire 173 K-row law enrichment finish in a single 12-hour Kaggle T4 session.

- **Law enrichment** ([`scripts/enrichment/`](scripts/enrichment/) + [`notebooks/07_law_de_enrichment/`](notebooks/07_law_de_enrichment/)) — 173 033 / 175 933 articles = **98.3 % coverage**. Each article produces a JSON record with `english_summary`, `concepts_en` (list of English doctrinal terms), `legal_question`, `applicability_conditions`, `provision_role_llm` (one of {definition, substantive_rule, procedural, appeal, cost_allocation, jurisdiction, constitutional_principle}), and `specificity_score` ∈ [0, 1].
- **Court enrichment** ([`notebooks/08_court_llm_descriptor_extraction/`](notebooks/08_court_llm_descriptor_extraction/)) — 363 258 / 2 476 315 paragraphs = **14.7 % coverage** (partial). Each paragraph carries a 15-role taxonomy (rule-statement, application, lower-court summary, party-position, dispositif, costs, …) and a 7-element doctrinal template (rule, fact-pattern, application, holding, citation_anchor, qualification, dispositif).

**Why generate English summaries when queries are English and docs are German.** The 4.3 % EN ↔ DE noun overlap measured on val (see [Engineering decisions](#engineering-decisions-worth-highlighting) §1) means raw English BM25 against the German corpus returns essentially nothing. Enrichment-EN BM25 turns this into a same-language match: on val law-gold, raw query → enrichment-EN BM25 hits **R@1000 = 0.232** vs ~0 % for raw query → DE-body BM25. This single design choice unlocks the cross-lingual problem.

**Prompt + schema engineering.** The law-enrichment prompt is in [`docs/laws_de_llm_enrichment_schema.md`](docs/laws_de_llm_enrichment_schema.md); the court contract is in [`docs/court_enrichment_field_contract.md`](docs/court_enrichment_field_contract.md). Both prompts use structured JSON output with explicit type validation and a "default to abstain" rule for fields the article doesn't support.

### 4. Citation graph — patched, validated, 96 % gold-node coverage

Built by parsing every document's body text with a citation-extraction regex and recording `(citing_doc, cited_citation_string)` edges into a separate SQLite (`analysis/citation_graph/citation_graph_extracted.sqlite`, 5.8 GB).

The first version of the regex missed ~7 % of gold citation entities. The shape of the misses (ATF / DTF dotted forms, French `c.` / `consid.` markers, space-tolerant docket numbers) was diagnosed and patched. After re-extraction: **+398 K edges (+9.6 %), +16 K nodes (+0.7 %)**, gold-as-node coverage rose from **92.9 % → 96.0 %**. The remaining 4 % is corpus-side (codes like LugÜ, FIDLEG, FINIG, FinfraG, GBV that aren't present in `laws_de.csv` at all), not regex-side.

**The graph is used as a tie-breaker, not a co-predictor.** Measured: **99.41 %** of gold-citation co-occurrence pairs have no graph edge. Top-frequent gold pairs (e.g. Art. 963 + 965 ZGB, Art. 10 Abs. 1 + 2 BV) are thematically adjacent but not textually cross-referenced. Even pairs that co-occur 3+ times in gold sets show only 1.3 % graph overlap. This negative result is what kept the pipeline from over-investing in graph expansion.

Source: [`scripts/citation_extraction/extract_citation_graph.py`](scripts/citation_extraction/extract_citation_graph.py) for the parser; [`analysis/citation_patterns/`](analysis/citation_patterns/) for the diagnostic.

### 5. Granularity resolver — 71.5 % → 98.1 % gold-in-corpus on train

Swiss statute articles fan out into Absatz children (`Art. 11 OR` is the parent of `Art. 11 Abs. 1 OR`, `Abs. 2`, …). Train gold cites bare-article forms; the corpus stores paragraph-granular rows. A regex-driven resolver expands `Art. N CODE` to its paragraph children when the parent is gold and the children are in corpus.

Validated: train coverage **71.54 % → 98.06 %** gold-in-corpus, val unchanged at 100 %. Source: [`scripts/data_prep/granularity_resolver.py`](scripts/data_prep/granularity_resolver.py).

### 6. Authority cards — per-document feature bundles (10 GB)

Authority cards are pre-computed per-document JSON bundles that join `documents`, statute-link counts, case-link counts, citation-graph in-degree, paragraph-role flags, and enrichment fields into one record. They exist as `law_authority_cards_v2_unified.jsonl` (842 MB, 175 K rows) and `court_authority_cards_v5_unified.jsonl` (9.95 GB, 2.48 M rows).

Why pre-compute. At rerank time, the model needs structured features alongside the candidate text. Loading per-candidate features from SQLite on each call adds ~10 ms × 500 candidates = 5 s of latency to a per-query rerank. Pre-bundling drops this to ~0 (the cards are mmap-loaded once per session).

---

## Experiments conducted (what worked, what didn't, with numbers)

Every experiment in the table below was actually run on `val.csv` (n = 10) and produced the reported numbers; verdicts and the rationale that drove the next step are in the linked source. The pattern across the project is **measure first, decide second** — multiple intuitions-from-the-literature got rejected after measurement.

### A. Sparse retrieval

| Experiment | Setup | Verdict | Key numbers | Source |
| --- | --- | --- | --- | --- |
| **A1. BM25 (FTS5) over German body text** | Raw English val query → `documents_fts` over `text_for_fts` (DE body + heading) | **Rejected as primary recall channel** | ≈ 0 % R@200 — the 4.3 % EN↔DE noun overlap leaves BM25 with nothing to match | [`research/personal_observations.md`](research/personal_observations.md) Obs 3 |
| **A2. Title-only BM25 with oracle headings** | Restrict FTS to `title` + `heading` fields only | Kept as one channel | Oracle ceiling **R@200 = 0.567**, **R@1000 = 0.575** on val law-gold | [`research/law_gold_retrieval_mechanism_2026-05-23.md`](research/law_gold_retrieval_mechanism_2026-05-23.md) §4.3bis |
| **A3. Enrichment-EN BM25 (the bridge channel)** | BM25 over concatenated `english_summary + concepts_en + legal_question + applicability_conditions` | **Kept — mandatory** | **R@200 = 0.050, R@1000 = 0.232** on val law-gold from a *raw* English query. With LLM expansion (A4) this rises substantially | [`research/law_gold_retrieval_mechanism_2026-05-23.md`](research/law_gold_retrieval_mechanism_2026-05-23.md) §4.3 |
| **A4. Multi-query BM25 expansion (HyDE + structured)** | Qwen3-8B generates per-query `{german_terms, applicable_codes, doctrine_concepts}`; issue 3–5 sub-BM25 searches, union pools | Kept | Not isolated per-channel; folds into the v7.5 multi-query pool (B1) | [`notebooks/02_v75_multiquery_canonical_pool/`](notebooks/02_v75_multiquery_canonical_pool/) |

### B. Dense retrieval & hybrid fusion

| Experiment | Setup | Verdict | Key numbers | Source |
| --- | --- | --- | --- | --- |
| **B1. Qwen3-Embedding-8B over the full 2.65 M corpus (dense ceiling)** | Brute-force cosine, oracle K selection per query | **Confirmed structural ceiling — rejected as primary spine** | **R@1000 = 0.289 (macro)**, oracle-K Macro F1 = 0.041. Per-query: 4 of 10 queries scored F1 = 0.000 | [`research/personal_observations.md`](research/personal_observations.md) Obs 3 |
| **B2. Bilingual embedding (MarianMT DE→EN prepended)** | Re-encode the corpus with `[LAW_NAME_EN] [PROVISION_TYPE] [TEXT_EN] [TEXT_DE]` | **Rejected** | Mini-index (1 064 docs) showed gains; **full-index delta = −0.026** vs DE-only at K = 1500. The MT signal helps small-K precision but doesn't survive at scale | [`notebooks/06_law_only_bm25_rerank_judge_baseline/`](notebooks/06_law_only_bm25_rerank_judge_baseline/) |
| **B3. Qwen3-Embedding-4B alternative** | Same setup as B1 with the 4B model | Rejected | Similar recall profile to 8B at lower latency, but the 8B model already fits VRAM with headroom; no reason to trade quality for unused budget | [`notebooks/05_embedding_and_retrieval_base/`](notebooks/05_embedding_and_retrieval_base/) |
| **B4. v7.5 multi-query 50K candidate pool (the canonical pool)** | 15-channel fan-out (anchor / BM25 / dense / graph / concept) → weighted RRF (`k = 60`) → 7-channel round-robin guarantee (910 reserved slots) → fill tail to `topk_final = 50 000` | **Kept — solves the recall problem** | **Macro R@50K = 0.893** on val; per-query range **0.766 (val_003) — 1.000 (val_004, val_005)**. The 0.766 floor on val_003 is the binding structural ceiling | [`notebooks/02_v75_multiquery_canonical_pool/pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb`](notebooks/02_v75_multiquery_canonical_pool/) |
| **B5. HyDE-only ablation on the multi-query pool** | Same pool as B4 but with the HyDE channel removed | Rejected — no lift | HyDE adds latency without measurable pool-recall gain at the K = 50 K horizon | [`notebooks/02_v75_multiquery_canonical_pool/pool_v75_multiquery_hyde_test_no_lift.ipynb`](notebooks/02_v75_multiquery_canonical_pool/) |

### C. Reranking shootout (the most informative experiment in the project)

Three cross-encoders were each run over the entire v7.5 50K-pool snapshot (255 K candidate-rows across 10 val queries) and fused four ways. The point of the experiment was to find out whether modern multilingual rerankers actually lift recall over plain RRF in a hybrid stack.

| Fusion strategy | R@500 | R@2K | R@10K | R@50K | What it tells us |
| --- | --- | --- | --- | --- | --- |
| **F0 — RRF only (no reranker)** | 0.415 | **0.611** | **0.816** | **0.897** | Plain fusion is the floor *and* the ceiling above K = 500 |
| F1 — Qwen3-Reranker-8B only | low | 0.250 | 0.644 | 0.897 | Single reranker hurts at every K |
| F2 — RRF of all three rerankers (Qwen3 + BGE-v2-m3 + Jina-v2-base-multilingual) | low | 0.268 | 0.505 | 0.897 | Stacking rerankers compounds their failure modes |
| F3 — Qwen3 + 9-signal dossier (the hybrid) | low | 0.568 | 0.791 | 0.897 | Dossier signals do most of the work; the reranker contributes mainly in the top-50/100 band |

**Conclusion drawn from this table:** rerankers are useful only at small K (top-50 / top-100), where they sharpen ordering. Above K ≈ 500, the heavy lift is done by the **9-signal dossier feature matrix** (statute-target intersection, lead-statute-intersection, concept intersection, term intersection, co-citation density, doctrinal density, chamber-match, hard-negative flag, fusion-rank). This is why the architecture caps reranking at K ≤ 500 instead of running it over the whole pool.

Source: [`drive_sync/swiss_law/hybrid_rerank_shootout_cache/hybrid_rerank_results.json`](drive_sync/swiss_law/) (cached scores + dossier features); methodology in [`docs/notebooks/01_current_direction_cascade_rerank_precision/rerank_hybrid_3model_shootout.md`](docs/notebooks/01_current_direction_cascade_rerank_precision/).

### D. The 9-signal dossier feature matrix

The dossier is a per-(query, candidate) feature record computed once per query and cached as `dossier_features.npz` (4.2 MB). The nine signals:

1. **`statute_intersection`** — count of doc statute-anchors ∩ query statute-targets.
2. **`lead_statute_intersection`** — same intersection, restricted to the first 200 characters of the doc (catches statutes named in rule-statement openers).
3. **`concept_intersection`** — doc `concepts_en` ∩ query expansion concepts.
4. **`term_intersection`** — doc terms ∩ query `term_targets_de/fr/it`.
5. **`co_cite_count`** — for laws: number of court paragraphs in the pool citing this law; for courts: peer count from the same case.
6. **`doctrinal_density`** — substring matches against rule-opener, doctrine, cost / fact / dispositif signatures (substring-only, no backtracking regex → ~21 K docs/sec).
7. **`chamber_match`** — 1 if the doc's chamber ∈ the allowed-chamber set for the query's legal area, else 0.
8. **`hard_neg_flag`** — 1 if the doc's role is `dispositif`, `costs`, `signature`, `procedural_history`, or `admissibility`.
9. **`fusion_rank`** — position in the RRF order (cheap, but useful as a calibration anchor).

This feature matrix is the difference between F1 (single reranker, R@10K = 0.644) and F3 (reranker + dossier, R@10K = 0.791) in the table above.

### E. LLM judge — failure-mode diagnosis

The endgame-pipeline judge (Qwen3-8B with a 7-category zone-routing prompt) produced ≈ 87 % rubber-stamp YES verdicts and collapsed F1 to < 0.10. Diagnosis identified four compounding bugs, all in the same prompt:

1. **Recall-biased system prompt** — explicit instruction "say YES when uncertain" (appropriate for some legal-research contexts but wrong for an F1 metric).
2. **Default-YES on parse failure** — when the response didn't contain `VERDICT:`, the parser inferred YES.
3. **`<think>` chain-of-thought consumed the token budget** — Qwen3-8B emits reasoning by default; with `max_new_tokens = 200`, 78 of 115 verdicts on one query were truncated mid-thinking.
4. **Threshold mismatch** — auto-YES / auto-NO thresholds were calibrated on the F1 = 0.777 baseline's `0.7·BM25 + 0.3·rerank` score scale (0–1); the endgame pipeline used raw RRF scores (typically 0.01–0.1). Every candidate fell into the borderline zone, so every candidate went to the judge.

The fix (in the current pipeline): disable `<think>`, cap `max_new_tokens = 24`, restrict response to `VERDICT: YES|NO` exactly, default to NO on any parse failure, recalibrate thresholds against the actual fused-score distribution per run. Source: [`research/endgame_handoff_2026-05-09.md`](research/endgame_handoff_2026-05-09.md) §4.3 + §7 Move 3.

### F. Query understanding

- **Statute-mention parser** (regex) — extracts `Art. N CODE` and `Art. N Abs. M CODE` from the query, canonicalises to `"NUM CODE"`. Hit rate on val: ~4 % of law-gold come in via this deterministic bridge; their precision is ~100 %.
- **Legal-area classifier** (regex over LLM-emitted German keywords) — maps query → one of {criminal, civil, administrative, tax, social, labor, intellectual_property, constitutional}. Used to gate the **chamber-match** dossier signal (e.g., criminal queries only credit chambers `1B / 6B / 7B / IV`).
- **Doctrine-concept expansion** (Qwen3-8B JSON prompt) — emits `{doctrine_concepts_de, english_concepts, legal_area_keywords, term_targets_de/fr/it}` per query, which feed channels A3 / A4 / D3 / D4.

### G. Evaluation methodology

- **Macro F1 with per-query oracle K** — sweep `K ∈ [5, 7, 10, 13, 15, 20, 25, 30, 35, 50, 75, 100]`, take the K maximising F1 per query, macro-average across val. This is the canonical metric, given the val gold-set size range of 10–47.
- **Stage-by-stage recall diagnostic** — per query, log `(gold_in_corpus, gold_in_50K_pool, gold_in_rerank_top_500, gold_in_judge_yes)`. This is what surfaced the 23.1 % rerank-pool recall in the broken endgame run.
- **Granularity-tolerant matching** — accept parent prediction for an Absatz-granular gold (and vice versa) when comparing to gold. Implemented in [`scripts/data_prep/granularity_resolver.py`](scripts/data_prep/granularity_resolver.py).
- **Train-vs-val calibration discipline** — all hyperparameters and thresholds are tuned on val (n = 10), never on train, because of the three-axis distribution shift documented in [Engineering decisions](#engineering-decisions-worth-highlighting) §4.

### H. Cost / latency optimisations actually applied

- **AWQ quantisation** for the judge / enrichment LLM — Qwen3-8B from 16 GB → ~5 GB VRAM, ~2× faster at comparable quality.
- **Brute-force GPU dense search** instead of FAISS-IVF — ~50 ms / query on Blackwell, no quantisation loss.
- **Pre-computed dossier features** (`.npz`, 4.2 MB) — loaded once per session, eliminates ~5 s of per-query SQLite joins.
- **HyDE / German-expansion / query-embedding caches** in [`reference_caches/`](reference_caches/) — 10 cached val queries are reused across every re-run, so iteration cost on the val set is dominated by index loads not LLM calls.
- **`<think>`-disabled, 24-token-capped judge calls** — the fixed judge prompt is ~10× cheaper per call than the broken one.

---

## Tech stack

- **Models** — `Qwen/Qwen3-Embedding-8B` (dense retrieval), `Qwen/Qwen3-Reranker-8B` (cross-encoder rerank), `Qwen/Qwen3-8B` and its AWQ-quantised variant (HyDE / structured expansion / judge), all served via Hugging Face Transformers and vLLM.
- **Sparse retrieval** — SQLite FTS5 with German / French / Italian Snowball stemming and custom legal-lexicon expansion; `rank_bm25` for ablations.
- **Dense retrieval** — fp16 matrix-multiply on a single NVIDIA RTX PRO 6000 Blackwell (95.6 GB VRAM); the full 2.65 M × 4096 corpus fits entirely in VRAM.
- **Storage layer** — 24 GB unified SQLite (`documents` + FTS5 + `statute_links` + `case_links` + `adjacent_law_links`), 21 GB fp16 embedding chunks, 10 GB v5 authority-card JSONL.
- **Auxiliary indexes** — citation-graph SQLite (96 % gold-node coverage), DuckDB FTS over court paragraphs, BM25 pickles.
- **Orchestration** — Python 3.10 + PyTorch 2.x; experimentation in Jupyter / Google Colab Pro+; long-running enrichment jobs on Kaggle GPUs.
- **Evaluation** — F1 K-sweep harness ([`scripts/eval_and_audit/eval_retrieval.py`](scripts/eval_and_audit/eval_retrieval.py)), per-query stage-by-stage recall diagnostic, granularity-aware gold matcher.

---

## Repository layout

```text
.
├── data/                              ← train / val / test splits + corpus CSVs (not in git, see "Data acquisition")
├── notebooks/                         ← 90+ Jupyter notebooks across 14 thematic groups
│   ├── 01_current_direction_cascade_rerank_precision/
│   ├── 02_v75_multiquery_canonical_pool/        ← R@50K = 0.89 multi-query funnel notebooks
│   ├── 03_anchor_funnel_evolution_v4_to_v74/
│   ├── 04_pre_v75_pipeline_iterations/          ← end-to-end "endgame" runs (F1 < 0.10, recall-bound)
│   ├── 05_embedding_and_retrieval_base/         ← dense-only ceiling measurement (R@1000 = 0.289)
│   ├── 06_law_only_bm25_rerank_judge_baseline/  ← the law-only v12 hybrid (Macro F1 = 0.777 on val, overfit)
│   ├── 07_law_de_enrichment/                    ← Qwen3-8B-AWQ enrichment of 173k law articles
│   ├── 08_court_llm_descriptor_extraction/      ← same model, 363k court paragraphs
│   ├── 09_court_citation_concept_enrichment/
│   ├── 10_authority_card_enrichment/
│   ├── 11_early_experiments/
│   ├── 12_early_funnel_pre_v7/
│   ├── 13_kaggle_and_utilities/
│   ├── 14_pdf_research_paper_extraction/
│   └── inventory/                               ← per-notebook code + output dumps
├── scripts/                           ← pipeline source code, organised by purpose
│   ├── build_pipeline/                ← corpus + authority-card builders
│   ├── citation_extraction/           ← citation parsing, graph extraction, aliases
│   ├── enrichment/                    ← LLM enrichment runners
│   ├── retrieval_and_rerank/          ← encode_queries, hybrid_retrieve, rerank_qwen3, llm_judge_qwen3
│   ├── eval_and_audit/                ← F1 K-sweep, recall diagnostics, gold checkers
│   ├── data_prep/                     ← granularity_resolver, split_laws_de
│   ├── merging/                       ← merge LLM enrichment into unified index
│   ├── notebook_builders/             ← code-generated notebook assembly
│   ├── kaggle_upload/                 ← Kaggle dataset publishing
│   ├── diagnostics/                   ← one-off diagnostic + repair scripts
│   ├── drive_pull/                    ← Drive sync (OAuth + scan + folder pullers)
│   └── synthesis_pipeline/
├── analysis/                          ← citation graph, citation patterns, gold-coverage analyses
│   ├── citation_graph/
│   ├── citation_patterns/
│   └── gold_coverage/
├── reference_caches/                  ← reusable caches (HyDE answers, query embeddings, German expansion)
├── research/                          ← experimental notes, anatomy studies, scratch
├── research_papers/                   ← PDFs referenced in the design
├── docs/                              ← written schemas + RAG plans + per-notebook code-dump mirrors
├── tests/                             ← pytest suite for parsers + normalisers
├── README.md
├── ROADMAP.md                         ← measured state-of-the-world + next steps
└── LICENSE
```

Large generated artefacts (`artifacts/`, `embeddings/*.npy`, `analysis/citation_graph/*.sqlite`, `llm_enrichment/`, `drive_sync/`, `pipeline/`) are excluded from version control — see [Data acquisition](#data-acquisition).

---

## Reproducing the pipeline

### Environment

```bash
python -m venv .venv
source .venv/bin/activate                   # Windows: .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install torch transformers accelerate sentence-transformers \
            rank_bm25 duckdb pandas numpy scipy scikit-learn \
            tqdm jupyter ipykernel matplotlib seaborn pyarrow
# AWQ-quantised judge:
pip install autoawq autoawq-kernels
```

Tested on Python 3.10 with PyTorch 2.x and CUDA 12.x. The judge and the reranker each need a single 24 GB+ GPU; the brute-force dense channel needs the full corpus matrix in VRAM (≈ 21 GB fp16) — either an 80 GB A100 / H100 / RTX PRO 6000, or chunked iteration on smaller cards.

### Data acquisition

The corpus and the gold splits come from the public Swiss legal-citation competition. Drop the following files into `data/`:

- `train.csv`, `val.csv`, `test.csv` — the official splits.
- `laws_de.csv` — 175 933 Swiss statute articles.
- `court_considerations.csv` — 2 476 315 federal-court paragraphs (DE / FR / IT).

The pipeline then generates everything else into `artifacts/` (canonical SQLite + FTS5 index), `embeddings/` (fp16 chunks), and `analysis/citation_graph/` (citation graph). Build steps in order:

```bash
python scripts/build_pipeline/build_unified_retrieval_corpus.py
python scripts/build_pipeline/prepare_embedding_input.py
python scripts/retrieval_and_rerank/encode_queries_qwen3_8b.py
python scripts/citation_extraction/extract_citation_graph.py
```

### Running retrieval

```bash
python scripts/retrieval_and_rerank/hybrid_retrieve.py \
    --query-file data/val.csv \
    --enable-rerank --enable-judge \
    --apply-granularity-post-filter \
    --output predictions.csv

python scripts/eval_and_audit/eval_retrieval.py \
    --predictions predictions.csv \
    --gold data/val.csv
```

For interactive runs, the canonical notebooks are:

| Goal | Notebook |
| --- | --- |
| Reproduce the **R@50K = 0.89** multi-query candidate pool on `val.csv` | [`notebooks/02_v75_multiquery_canonical_pool/pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb`](notebooks/02_v75_multiquery_canonical_pool/) |
| Reproduce the **law-only Macro F1 = 0.777 / overfit** baseline on `val.csv` | [`notebooks/06_law_only_bm25_rerank_judge_baseline/law_only_hybrid_v12_val_macroF1_0.777_overfit_warning.ipynb`](notebooks/06_law_only_bm25_rerank_judge_baseline/law_only_hybrid_v12_val_macroF1_0.777_overfit_warning.ipynb) |
| Measure the dense-only **R@1000 = 0.289** ceiling | [`notebooks/05_embedding_and_retrieval_base/`](notebooks/05_embedding_and_retrieval_base/) |
| Run the end-to-end "endgame" pipeline (F1 < 0.10 baseline + bug diagnosis) | [`notebooks/04_pre_v75_pipeline_iterations/`](notebooks/04_pre_v75_pipeline_iterations/) |

---

## Roadmap & status

The current open work is tracked in [`ROADMAP.md`](ROADMAP.md):

- **Solved** — multi-query pool recall (R@50K = 0.893 macro on `val.csv`); reranker shootout (Qwen3 vs BGE vs Jina above K = 500); citation-graph extraction (96.0 % gold-node coverage); granularity resolver (98.1 % on `train.csv`); LLM-judge failure-mode diagnosis.
- **Open** — Stage 2 dossier-aware rerank using channel-fingerprints + statute-target intersection + co-citation density; variable-K LLM gate with default-NO semantics; per-query K calibration from cheap features; granularity-aware final-pick.
- **Target** — Macro F1 ∈ [0.6, 0.8] on `val.csv`.

---

## Engineering capabilities demonstrated in this codebase

This project, end-to-end, exercises the following areas. Each item points at where in the repo the work lives.

- **Hybrid retrieval design at scale** — 15-channel funnel combining sparse (BM25 over FTS5), dense (Qwen3-Embedding-8B brute-force), anchors (regex statute / case parsers), citation-graph 1-hop, and concept channels; fused via weighted Reciprocal Rank Fusion with a round-robin reserved-slot guarantee. See [`research/cascade_dossier_plan.md`](research/cascade_dossier_plan.md) and [`notebooks/02_v75_multiquery_canonical_pool/`](notebooks/02_v75_multiquery_canonical_pool/).
- **Cross-encoder reranking with empirical trade-off analysis** — three rerankers (Qwen3-Reranker-8B, BGE-v2-m3, jina-v2-base-multilingual) compared head-to-head across K ∈ {500, 2K, 10K, 50K}; rerank-vs-fuse decision driven by measurement, not intuition. See the reranker-shootout table above and [`docs/notebooks/01_current_direction_cascade_rerank_precision/`](docs/notebooks/01_current_direction_cascade_rerank_precision/).
- **Query understanding** — regex statute-mention parser, regex-over-LLM legal-area classifier, structured JSON expansion (`{legal_domain, applicable_codes, german_terms, concept_seeds}`) via Qwen3-8B, HyDE German hypothetical-answer generation. See [`scripts/retrieval_and_rerank/`](scripts/retrieval_and_rerank/).
- **Knowledge-graph engineering** — citation extraction regex tuned through diagnosis-driven patches (ATF / DTF / `c.` / `consid.` / dockets), validated to **96.0 %** gold-node coverage; the negative co-prediction result (99.41 % of gold pairs have no edge) is what kept the design from over-investing in graph expansion. See [`scripts/citation_extraction/extract_citation_graph.py`](scripts/citation_extraction/extract_citation_graph.py) and [`analysis/citation_patterns/`](analysis/citation_patterns/).
- **Production-grade LLM systems** — AWQ-quantised judge (16 GB → ~5 GB VRAM, ~2× faster), structured-JSON prompt engineering with type validation, failure-mode diagnosis (87 %-rubber-stamp-YES debugging across four compounding bugs in one prompt), default-NO parse semantics, `<think>`-disabled inference for latency control. See [`research/endgame_handoff_2026-05-09.md`](research/endgame_handoff_2026-05-09.md) §4.3.
- **Search infrastructure & analyser design** — SQLite + FTS5 schema with `unicode61` tokeniser, Snowball stemming for DE / FR / IT, custom legal-lexicon expansion at index time, separate small tables for statute / case / sibling links. The schema is portable to Postgres `tsvector` + `pg_trgm` without changes to the retrieval logic. See [`scripts/build_pipeline/build_unified_retrieval_corpus.py`](scripts/build_pipeline/build_unified_retrieval_corpus.py).
- **Evaluation pipelines that drive shipping** — macro-F1 K-sweep harness, stage-by-stage recall diagnostic (which is how the 23.1 % rerank-pool ceiling surfaced), granularity-tolerant gold matching, train-vs-val calibration discipline that explicitly bans train-derived weights as leakage. See [`scripts/eval_and_audit/`](scripts/eval_and_audit/).
- **Authority / feature engineering** — per-document pre-computed feature bundles with statute / case link counts, citation-graph in-degree, paragraph-role flags, doctrinal-density signatures, chamber-match priors, hard-negative role flags. Nine such signals make up the dossier feature matrix that carries the bulk of ranking weight above K = 500. See [`docs/notebooks/01_current_direction_cascade_rerank_precision/`](docs/notebooks/01_current_direction_cascade_rerank_precision/) for the design write-up.
- **Cost & latency engineering** — AWQ quantisation, `<think>`-disabled 24-token judge calls, pre-computed `.npz` feature caches, GPU brute-force matmul instead of FAISS-IVF (justified by measurement, not assumption), per-query HyDE / German-expansion / query-embedding caches for fast iteration. See [`reference_caches/`](reference_caches/).
- **Distributed batch processing** — long enrichment jobs split across Kaggle T4 sessions (12-hour cap) and Colab Pro+ Blackwell sessions; Drive-sync layer for state persistence ([`scripts/drive_pull/`](scripts/drive_pull/)); Kaggle dataset publishing for cross-platform reuse ([`scripts/kaggle_upload/`](scripts/kaggle_upload/)).
- **Empirical rigour and honest reporting** — every headline number is attributed to a notebook and dataset; rejected experiments (bilingual embedding, citation-graph co-prediction, single-channel dense) are documented with the measurements that rejected them; the F1 = 0.777 baseline is published with its train-collapse caveat rather than as a clean win.

The recurring theme across the project: **measure first, decide second.** Multiple intuitions-from-the-literature (graph expansion, bigger reranker, bilingual re-embedding) were tested against val and rejected with numbers; the architecture is shaped by those rejections.

---

## License

Released under the [MIT License](LICENSE).

## Acknowledgments

- The **Qwen3** model family — Alibaba Cloud / Tongyi Lab. Used for embedding (`Qwen3-Embedding-8B`), reranking (`Qwen3-Reranker-8B`) and judge / HyDE (`Qwen3-8B`, `Qwen3-8B-AWQ`).
- **rank_bm25** — Dorian Brown.
- The law-only v12 hybrid pipeline (BM25 + cross-encoder + LLM judge with zone-routing) was prototyped from a public competition notebook layout and adapted to this corpus; the val-set overfit caveat described above was discovered during reproduction.
