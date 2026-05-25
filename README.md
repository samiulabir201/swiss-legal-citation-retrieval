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

Every row below states **what was measured · on which split · with which method · in which notebook**, plus any caveats that materially affect interpretation.

### A — Retrieval-pool recall (does the candidate pool *contain* the gold?)

| What is measured | Value | Eval split | Method | Caveat | Source notebook |
| --- | --- | --- | --- | --- | --- |
| Macro recall of gold citations in the top-50 000 candidate pool | **R@50K = 0.893** | `val.csv`, n = 10 | 15-channel anchor / BM25 / dense / graph funnel + weighted RRF + 7-channel round-robin guarantee | Per-query recall ranges 0.766 (val_003) — 1.000 (val_004, val_005); val_003 R_max = 0.766 is a structural ceiling | [`notebooks/02_v75_multiquery_canonical_pool/pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb`](notebooks/02_v75_multiquery_canonical_pool/) |
| Dense-only recall ceiling (R@1000) | **0.289** | `val.csv`, n = 10 | Qwen3-Embedding-8B brute-force cosine over the full 2.65 M-doc corpus | Establishes that dense retrieval alone *cannot* solve this task; motivates the multi-channel design | [`notebooks/05_embedding_and_retrieval_base/`](notebooks/05_embedding_and_retrieval_base/) |

### B — End-to-end Macro F1 (the headline number, with full attribution)

| What is measured | Value | Eval split | Method | Caveat | Source notebook |
| --- | --- | --- | --- | --- | --- |
| Macro F1 of the law-only hybrid pipeline (the "v12 hybrid" baseline) | **0.777** (P = 0.845, R = 0.748, avg K = 13) | `val.csv`, n = 10 | BM25 top-100 → Qwen3-Reranker-8B (re-fused as `0.7·BM25 + 0.3·rerank`) → zone-split with `HIGH = 0.55, LOW = 0.25` → Qwen3-8B judge on the borderline zone only | **Law-only — no court considerations.** **Val-overfit:** the same pipeline scores Macro F1 = **0.296 on train**, a +0.481 delta the notebook itself flags as `warning`. Only the `thresh_only` and `bm25_K3` variants generalise. | [`notebooks/06_law_only_bm25_rerank_judge_baseline/law_only_hybrid_v12_val_macroF1_0.777_overfit_warning.ipynb`](notebooks/06_law_only_bm25_rerank_judge_baseline/law_only_hybrid_v12_val_macroF1_0.777_overfit_warning.ipynb) |
| End-to-end Macro F1 using the full 2.65 M corpus + v7.5 multi-query pool + LLM judge | < 0.10 | `val.csv`, n = 10 | The "endgame" pipeline — multi-channel pool from row A above + Qwen3-Reranker-8B + Qwen3-8B judge | **The judge's default-YES-on-parse-failure and `<think>` token-budget bug compounded to ≈ 87 % rubber-stamp YES verdicts.** Recall, not precision, is the binding bottleneck — see [`research/endgame_handoff_2026-05-09.md`](research/endgame_handoff_2026-05-09.md) §4 for the full diagnosis | [`notebooks/04_pre_v75_pipeline_iterations/`](notebooks/04_pre_v75_pipeline_iterations/) |

### C — Enrichment & infrastructure (the building blocks)

| What is measured | Value | Sample / dataset | Method | Source code |
| --- | --- | --- | --- | --- |
| Law-article LLM-enrichment coverage | **173 033 / 175 933 = 98.3 %** | All law articles in `laws_de.csv` | Qwen3-8B-AWQ with an English-aligned descriptor schema (`english_summary`, `concepts_en`, `legal_question`, `applicability_conditions`, `provision_role_llm`, `specificity_score`) | [`scripts/enrichment/`](scripts/enrichment/) + [`notebooks/07_law_de_enrichment/`](notebooks/07_law_de_enrichment/) |
| Court-paragraph LLM-enrichment coverage | 363 258 / 2 476 315 ≈ 14.7 % | Court considerations in `court_considerations.csv` | Same model, 15-role taxonomy + 7-element doctrinal template | [`notebooks/08_court_llm_descriptor_extraction/`](notebooks/08_court_llm_descriptor_extraction/) |
| Citation-graph gold-as-node coverage | **92.9 % → 96.0 %** (+3.1 pp) | All gold-citation entities in train + val | Regex patches for ATF / DTF / `c.` / `consid.` / space-tolerant docket forms | [`scripts/citation_extraction/extract_citation_graph.py`](scripts/citation_extraction/extract_citation_graph.py) |
| Granularity-resolver gold recovery | **71.5 % → 98.1 %** (+26.6 pp) | `train.csv` gold | Bare `Art. N LAW` ↔ paragraph-children fanout against `laws_de.csv` | [`scripts/data_prep/granularity_resolver.py`](scripts/data_prep/granularity_resolver.py) |
| Dense-vector query latency | ≈ 50 ms / query | 2.65 M × 4096 fp16 = 21 GB matrix in VRAM | Brute-force `E @ q` on NVIDIA RTX PRO 6000 Blackwell (95.6 GB VRAM) | [`scripts/retrieval_and_rerank/encode_queries_qwen3_8b.py`](scripts/retrieval_and_rerank/encode_queries_qwen3_8b.py) |

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

## License

Released under the [MIT License](LICENSE).

## Acknowledgments

- The **Qwen3** model family — Alibaba Cloud / Tongyi Lab. Used for embedding (`Qwen3-Embedding-8B`), reranking (`Qwen3-Reranker-8B`) and judge / HyDE (`Qwen3-8B`, `Qwen3-8B-AWQ`).
- **rank_bm25** — Dorian Brown.
- The law-only v12 hybrid pipeline (BM25 + cross-encoder + LLM judge with zone-routing) was prototyped from a public competition notebook layout and adapted to this corpus; the val-set overfit caveat described above was discovered during reproduction.
