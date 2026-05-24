# Swiss Legal Citation Retrieval

Cross-lingual retrieval over **2.65 M Swiss legal documents** in German, French and Italian — given an English fact-pattern, return the exact set of statute articles and federal-court considerations a legal expert would cite.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## What this project does

Swiss federal jurisprudence cites two kinds of authorities: **statute articles** (laws_de, ~176 K articles) and **court considerations** (paragraphs of Bundesgericht rulings, ~2.48 M). Legal-tech competitions provide English fact-patterns and ask for the citation set an expert would attach. The corpus is multilingual (DE / FR / IT); the queries are English; gold sets contain **10–47 citations each across ~5 legal domains** per query.

Off-the-shelf retrieval doesn't work here:

| Naïve approach                                      | What it scores              | Why it fails                                                                             |
| --------------------------------------------------- | --------------------------- | ---------------------------------------------------------------------------------------- |
| BM25 directly on EN query → DE corpus               | ≈ 0 % recall                | EN noun overlap with the DE corpus is **4.3 %**                                          |
| Qwen3-Embedding-8B (#1 MMTEB-Multilingual) brute-force | **R@1000 ≈ 0.29** on val | "Cosine similarity is the wrong relationship": citation ≠ topical similarity             |
| Single-channel LLM judge                            | F1 ≈ 0.04                   | 87 % rubber-stamp YES; recall-side problem masquerading as a precision problem           |

The system in this repository combines **anchor parsing, sparse BM25 over FTS5, GPU brute-force dense vectors, HyDE + structured query expansion, citation-graph 1-hop, cross-encoder reranking, and an LLM judge with default-NO semantics** into a multi-channel funnel that achieves **macro pool recall ≈ 0.89 at K = 50 K** on the validation set.

---

## Headline results

| Metric                              | Result                              | Source                                                     |
| ----------------------------------- | ----------------------------------- | ---------------------------------------------------------- |
| **Macro pool recall (R@50K)**       | **0.893**                           | v7.5 multi-query anchor funnel                             |
| Per-query R_max range               | 0.766 (val_003) — 1.000 (val_004/5) | Characterizes the binding ceiling                          |
| Reference notebook F1 reproduced    | 0.777                               | Untitled75 public reference, matched on val                |
| Law enrichment coverage             | **173 033 / 175 933 (98.3 %)**      | Qwen3-8B AWQ with English-aligned descriptor schema        |
| Court enrichment coverage           | 363 258 / 2 476 315 (≈ 15 %)        | Same model + 15-role taxonomy + 7-element template          |
| Citation-graph gold-node coverage   | **92.9 % → 96.0 %**                 | After ATF / DTF / c. / consid. regex patches               |
| Granularity-resolver gold recovery  | **71.5 % → 98.1 %** on train        | Article ↔ Absatz fanout                                    |
| Dense-vector throughput             | **≈ 50 ms / query**                 | 2.65 M × 4096 fp16 brute-force on NVIDIA RTX PRO 6000      |

---

## Architecture

```
              English query (10–47-citation gold)
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
             │ (cross-encoder)        │
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

- **Anchor channels** — when a query names a code ("Art. 95 BGG") the answer is partly deterministic; bypass the noisy embedding step.
- **HyDE + structured expansion** — bridge the 4.3 % EN ↔ DE vocab gap by hallucinating German legal vocabulary from an English fact-pattern.
- **Sparse + dense union** — neither alone hits 50 % recall; their fusion (RRF) does.
- **Cross-encoder rerank** — applied at K ≤ 500 only; ablations showed RRF dominates above K = 500.
- **LLM judge with default-NO** — the public reference judge defaulted to YES on parse failure and produced 87 % rubber-stamps. Inverted to default-NO + disabled `<think>` + 24-token cap.

---

## Engineering decisions worth highlighting

These came out of empirical work, not literature search:

1. **Embedding-similarity ceiling was measured, not assumed.** Qwen3-Embedding-8B on the full corpus capped at R@1000 = 0.289. The system therefore treats dense retrieval as one channel among many, not the spine.
2. **Citation graph is a tie-breaker, not a co-predictor.** Measured: 99.41 % of gold-citation pairs have **no graph edge**. The graph is used downstream of scoring, not to expand the candidate pool.
3. **Variable-K is mandatory.** Gold set sizes range 10–47; fixed-K is provably suboptimal. K is calibrated per query from cheap features (statute target count, query length, court/law ratio).
4. **No training-set priors.** Train is 99 % German with median gold size 2 and 1.2 % court share; val/test are 100 % English with median gold 22 and 40 % court. Training-set-derived weights are explicitly banned to avoid distribution-shift leakage.
5. **Granularity-aware gold matching.** Swiss articles fan out into Absatz children (Art. 11 → Art. 11 Abs. 1, Abs. 2, …). A bare-article ↔ paragraph-child resolver lifts gold-in-corpus from 71.5 % to 98.1 % on train.

The [research/](research/) directory contains the canonical write-ups of these findings, including [`endgame_handoff_2026-05-09.md`](research/endgame_handoff_2026-05-09.md), [`personal_observations.md`](research/personal_observations.md), and [`law_gold_retrieval_mechanism_2026-05-23.md`](research/law_gold_retrieval_mechanism_2026-05-23.md). Supporting analyses live under [analysis/](analysis/) (citation graph, citation patterns, gold-citation coverage).

---

## Tech stack

- **Models** — Qwen3-Embedding-8B, Qwen3-Reranker-8B, Qwen3-8B-AWQ (judge), all via Hugging Face Transformers / vLLM.
- **Sparse retrieval** — SQLite FTS5 with German / French / Italian Snowball stemming + custom lexicon expansion; rank_bm25 for ablations.
- **Dense retrieval** — fp16 matrix-multiply on NVIDIA RTX PRO 6000 Blackwell (95.6 GB VRAM); the full 2.65 M × 4096 corpus fits in VRAM.
- **Storage** — 24 GB unified SQLite (`documents` + FTS5 + statute_links + case_links + adjacent_law_links), 21 GB fp16 embedding chunks, 10 GB v5 authority-card jsonl.
- **Auxiliary indexes** — citation-graph SQLite, DuckDB FTS for court paragraphs, BM25 pickles.
- **Orchestration** — Python 3.10 + PyTorch 2.x; experimentation in Jupyter / Colab Pro+; long-running enrichment on Kaggle.
- **Eval** — F1 K-sweep harness, per-query recall diagnostic, granularity-aware gold matcher.

---

## Repository layout

```text
.
├── data/                              ← train / val / test + corpus CSVs (not in git, see "Data acquisition" below)
├── notebooks/                         ← 90+ Jupyter notebooks across 14 topic areas
│   ├── 01_current_direction_cascade_rerank_precision/
│   ├── 02_v75_multiquery_canonical_pool/      ← the recall-0.89 generator
│   ├── 03_anchor_funnel_evolution_v4_to_v74/
│   ├── 04_pre_v75_pipeline_iterations/
│   ├── 05_embedding_and_retrieval_base/
│   ├── 06_high_scoring_reference/             ← F1=0.777 reference + study
│   ├── 07_law_de_enrichment/
│   ├── 08_court_llm_descriptor_extraction/
│   ├── 09_court_citation_concept_enrichment/
│   ├── 10_authority_card_enrichment/
│   ├── 11_early_experiments/
│   ├── 12_early_funnel_pre_v7/
│   ├── 13_kaggle_and_utilities/
│   ├── 14_pdf_research_paper_extraction/
│   └── inventory/                              ← per-notebook code + output dumps
├── scripts/                           ← pipeline source code, organized by purpose
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
├── analysis/                          ← citation graph, citation patterns, gold coverage
│   ├── citation_graph/
│   ├── citation_patterns/
│   └── gold_coverage/
├── reference_caches/                  ← reusable caches (HyDE answers, query embeddings, German expansion)
├── research/                          ← experimental notes, anatomy studies, scratch
├── research_papers/                   ← PDFs referenced in the design
├── docs/                              ← written schemas + RAG plans
├── tests/                             ← pytest suite for parsers + normalizers
├── README.md
├── ROADMAP.md                         ← measured-state-of-the-world, next steps
└── LICENSE
```

Large generated artifacts (`artifacts/`, `embeddings/*.npy`, `analysis/citation_graph/*.sqlite`, `llm_enrichment/`, `drive_sync/`, `pipeline/`) are excluded from version control — see the [Data acquisition](#data-acquisition) section.

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
# AWQ-quantized judge:
pip install autoawq autoawq-kernels
```

Tested on Python 3.10 with PyTorch 2.x and CUDA 12.x. The judge and reranker need a single 24 GB+ GPU; brute-force dense retrieval needs the full corpus in VRAM (~21 GB fp16) — either an 80 GB A100 / H100 / RTX PRO 6000, or chunked iteration on smaller cards.

### Data acquisition

The corpus and gold splits come from the public Swiss legal competition. Drop the following into `data/`:

- `train.csv`, `val.csv`, `test.csv` — the official splits.
- `laws_de.csv` — 175 933 statute articles.
- `court_considerations.csv` — 2 476 315 federal-court paragraphs.

The pipeline then generates everything else under `artifacts/` (canonical SQLite + FTS5 index), `embeddings/` (fp16 chunks), and `data_insights/` (citation graph). Build scripts:

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

For end-to-end interactive runs, open the notebooks in [`notebooks/06_high_scoring_reference/`](notebooks/06_high_scoring_reference/) (F1 = 0.777 reference) or [`notebooks/02_v75_multiquery_canonical_pool/`](notebooks/02_v75_multiquery_canonical_pool/) (recall 0.89 multi-query funnel).

---

## Roadmap & status

The current open work is on [`ROADMAP.md`](ROADMAP.md):

- **Solved** — pool recall (R@50K = 0.893), reranker shootout (Qwen3 vs BGE vs Jina), citation-graph extraction, granularity resolver, LLM-judge failure-mode diagnosis.
- **Open** — Stage 2 dossier-aware rerank, variable-K LLM gate with default-NO semantics, K calibration from cheap features, granularity-aware final pick.
- **Target** — Macro F1 = 0.6 — 0.8 on val.

---

## License

Released under the [MIT License](LICENSE).

## Acknowledgments

- **Qwen3** model family — Alibaba Cloud / Tongyi Lab.
- **rank_bm25** — Dorian Brown.
- Public competition reference notebooks for the F1 = 0.777 starting point.
