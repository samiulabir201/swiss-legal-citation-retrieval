# 01 — Original Competition Data + Gold-Citation Analysis Files

**Scope of this document:** the Kaggle-shipped competition files (queries + corpus + sample
submission), the locally-derived granularity-expanded gold CSVs, the `laws_de` split-segment
CSVs, the full DE law-text JSON, and the gold-citation analysis outputs in
`data_insights/gold_analysis_coverage_and_parents/`.

All paths are absolute on the local Windows machine: `e:\swiss_citation_extraction\…`.

---

## Official competition description (excerpt)

The Kaggle competition page
(<https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval/data>)
is gated behind login and cannot be fetched anonymously — `WebFetch` against both `/data` and
`/overview` returned only the page title. The authoritative competition description is
mirrored locally at `e:\swiss_citation_extraction\research\problem_statement.md`. Verbatim
excerpt:

> This work addresses the task of **cross-lingual legal citation retrieval** for Swiss law.
> Given an English-language natural language query describing a legal question or scenario,
> the system must identify and return a set of canonical citation identifiers from a fixed
> Swiss legal corpus that constitute the authoritative legal basis for answering that query.
> This is formally a **set retrieval problem** — not a ranking problem, not a generative
> problem — where the output is an unordered set of citation strings drawn from a closed
> vocabulary, and correctness is judged by exact string match against a human-annotated
> gold set. **Target performance:** Macro F1 score of **0.6 to 0.8** on the public
> leaderboard (50% of test queries).
>
> The corpus is composed of two sources:
>
> - `laws_de.csv` — Swiss federal laws, German, ~175,933 rows, one row per law snippet
>   (article or paragraph), citation key e.g. `Art. 11 Abs. 2 OR`, with fields `text` and
>   `title`.
> - `court_considerations.csv` — Swiss federal court decisions, primarily German + some
>   French/Italian, ~2,476,315 rows, one row per consideration (Erwägung), citation
>   structures `BGE [volume] [series] [page] E. [consideration]` (leading) or
>   `[chamber_code]_[serial]/[year] E [consideration]` (non-leading).
>
> Splits: `train.csv` — German queries from the LEXam benchmark (Fan et al., 2025, CC BY 4.0),
> distribution **does not** match the test distribution; `val.csv` — 10 English queries,
> matches test distribution; `test.csv` — 40 English queries (20 public / 20 private);
> `HIDDEN.csv` — a private re-evaluation set held by the organizers (not shipped).
>
> Evaluation: citation-level **Macro F1** averaged uniformly across queries; exact string
> match required.

The Kaggle file shipping list `train.csv` + `val.csv` + `test.csv` + `sample_submission.csv`
+ `laws_de.csv` + `court_considerations.csv` is fixed; everything else under `data/` was
derived locally.

---

## Family A — Query splits (competition-provided)

### A.1 `data/train.csv`

**Files:**
- `e:\swiss_citation_extraction\data\train.csv` — 1,970,552 bytes, CSV

**Description:**
1,139 training queries. Columns: `query_id, query, gold_citations`. Queries are German
(99%) Swiss-law exam-style scenarios sourced from the LEXam benchmark. `gold_citations` is
a `;`-separated list of citation strings (laws + court considerations). Sample row
`train_0001` references `Art. 10a Abs. 1 USG;Art. 2 Abs. 1 UVPV;Art. 10a Abs. 1 UVG`.

**Source:** Kaggle competition `llm-agentic-legal-information-retrieval`. Gold extracted from
the LEXam open-question split (`problem_statement.md` §4.1).

**Purpose:** Supervised training signal. Carries the structural ceiling discussion: 28.5%
of train gold is unreachable in the corpus (Obs 1; see `swiss-citation-data` SKILL line 14:
"71.5% Class A, 20.9% text-only, 7.6% absent"). 99% German vs val/test 100% English is the
core distribution-shift axis.

**Notebooks that use it (verified by Grep):**
- `notebooks\_inventory\reference_F1_0.777_Untitled75.md` — loads in F1=0.777 reference
  (`train=pd.read_csv(bd/"data"/"train.csv")` at line 1387 of the inventory dump)
- `notebooks\_inventory\anchor_funnel_v7_val001.md`, `anchor_funnel_v7_part2_with_outputs.md`,
  `anchor_funnel_v7_part3_with_outputs.md`, `anchor_funnel_v7_part4_with_outputs.md`,
  `anchor_funnel_v7_4_val001.md`, `anchor_funnel_v6_val001.md`, `anchor_funnel_v5_val001.md`,
  `anchor_funnel_v4_val001.md`, `anchor_funnel_v1_val001_base.md` — anchor funnel iteration
  loads train for query-expansion / prior tuning
- `notebooks\_inventory\pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.md`,
  `pool_v75_multiquery_iteration_11.md`, `pool_v75_multiquery_hyde_test_no_lift.md` — v7.5
  multi-query funnel
- `notebooks\_inventory\rerank_hybrid_3model_shootout.md`,
  `rerank_hybrid_3model_shootout_with_outputs.md`,
  `rerank_only_diagnostic_F3_no_rerank.md`,
  `precision_v1_top_gold_from_50k_candidates.md`,
  `precision_v1_final.md`, `precision_v1_not_working_diagnostic.md` — rerank/precision
  notebooks
- `notebooks\_inventory\cascade_legalmalr_v1_*.md` (poc/base/optimized) — cascade dossier
- `notebooks\_inventory\endgame_colab_full_pipeline.md` — endgame stack (abandoned)
- `notebooks\_inventory\pipeline_end_to_end_*.md`, `pipeline_iteration_latest_*.md`,
  `pipeline_no_debug_prints.md`, `pipeline_stage0_gated_4ae3b3ffc3.md` — pre-v7.5 iterations
- `notebooks\_inventory\exp_A1_bgem3_statutes.md` through `exp_A6_legal_roberta_finetune.md`,
  `early_dense_embedding_test.md`, `early_recall_preserving_funnel.md` — early experiments
- `notebooks\_inventory\reference_F1_0.7716_scored.md` — second high-scoring reference
- `notebooks\_inventory\investigate_misc_notebook.md` — misc

(Match count across `notebooks/_inventory/`: **49 files**.)

**Scripts that produce/consume it:**
- `scripts\data_prep\granularity_resolver.py` — reads `data/train.csv`, writes
  `data/train_granularity_expanded.csv` (CLI in module docstring, lines 21-25)
- `scripts\data_prep\repair_train_csv_utf8.py` — UTF-8 repair (verified phantom; file is
  already clean per `swiss-citation-experiments` line 148)
- `scripts\eval_and_audit\verify_gold_citation_coverage.py` (line 28: `data_dir / f"{dataset}.csv"`)
- `scripts\eval_and_audit\check_gold_parent_links.py` (line 28)
- `scripts\eval_and_audit\eval_retrieval.py`, `experiment_court_recall_enriched_1m.py`,
  `experiment_court_recall_stress_1m.py`, `experiment_enrichment_recall_stress.py`
- `scripts\enrichment\enrich_val001_gold_cards.py`
- `scripts\notebook_builders\create_v7_notebook.py`, `_build_endgame_notebook.py`,
  `create_recall_funnel_notebook.py`
- `scripts\diagnostics_loose\diag_recall.py`, `_gen_rerank_final_diag_notebook.py`

**Performance / role in pipeline:**
Used for query-side priors and for the granularity resolver. **Not** used as a held-out
metric: per `swiss-citation-experiments` and Obs 4, train priors do not transfer reliably to
val (train DE / val EN, train citation-count median 2 vs val median 22, train 1.2% court
share vs val 40.6%). The user's 2026-05-15 note explicitly withdrew the older "train is
unreliable" stance, but train remains a secondary signal.

**Verdict:** **active** — canonical training queries; cannot be replaced.

---

### A.2 `data/val.csv`

**Files:**
- `e:\swiss_citation_extraction\data\val.csv` — 19,775 bytes, CSV

**Description:**
10 validation queries (`val_001` … `val_010`). Columns: `query_id, query, gold_citations`.
100% English. 222 unique gold citations (251 mentions; see gold_citation_coverage table
below). Per-query gold-set sizes range 10–47, median 22 (`swiss-citation-task` §Output).

**Source:** Kaggle competition file.

**Purpose:** The only reliable in-distribution dev signal (`swiss-citation-data` line 15:
"100% Class A — all gold retrievable. The only reliable in-distribution dev signal.").
Every Macro-F1 / R@K number cited in `swiss-citation-experiments` is computed on this set.

**Notebooks that use it (verified by Grep):**
- Same 49-file set as `train.csv` above — the matched grep pattern catches both. Notable
  call sites confirmed by content grep:
  - `notebooks\_inventory\reference_F1_0.777_Untitled75.md` line 94 (`VAL_PATH = DATA / "val.csv"`)
  - `notebooks\_inventory\anchor_funnel_v7_val001.md` line 68
    (`"val_csv": DATA_ROOT / "data" / "val.csv"`) and line 326 (`print(f"val.csv has {len(val_df)} queries")` → `val.csv has 10 queries`)
  - `notebooks\_inventory\pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.md` line 68
    and line 325 (same)
  - `notebooks\_inventory\anchor_funnel_v1_val001_base.md` line 71

**Scripts that produce/consume it:**
- Same set as `train.csv`. Producer: none (shipped). Consumers (all under
  `scripts/eval_and_audit/`, `scripts/data_prep/`, `scripts/enrichment/`,
  `scripts/notebook_builders/`, `scripts/diagnostics_loose/`) — see A.1 list.

**Performance / role in pipeline:**
The 10 val queries are the binding evaluation set. v7.5 multi-query funnel reaches
**Macro R@50k = 0.893** on val; val_003 R_max=0.766 is the structural pool ceiling
(`swiss-citation-experiments` v7.5 section).

**Verdict:** **active** — the single source of truth for every reported number.

---

### A.3 `data/test.csv`

**Files:**
- `e:\swiss_citation_extraction\data\test.csv` — 56,724 bytes, CSV

**Description:**
40 test queries (20 public LB / 20 private LB per `problem_statement.md` §4.3). Columns:
`query_id, query` — **no `gold_citations`** column. Queries are English Swiss-law scenarios
(`test_001` involves cross-border IP / Lumen Sàrl; `test_002` involves SVG cyclist tort).

**Source:** Kaggle competition file.

**Purpose:** Final submission target. Not used for any current development run
(`swiss-citation-approach`: "Test-set submission — meaningless until val F1 is in target
range").

**Notebooks that use it (verified by Grep):**
- Same 49-file set returned by the combined `train|val|test` pattern; in practice the
  v7.5 + cascade notebooks read it only for the eventual submission cell. Confirmed reads:
  `reference_F1_0.777_Untitled75.md`, `pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.md`,
  `anchor_funnel_v7_*` family, `rerank_hybrid_3model_shootout*` (via the inventory
  filename list).

**Scripts that produce/consume it:**
- Same `verify_gold_citation_coverage.py` / `check_gold_parent_links.py` /
  `eval_retrieval.py` consumers via the `f"{dataset}.csv"` pattern, though the verify
  scripts treat train+val only (`CSV_SPLITS = ("train", "val")` in `check_gold_parent_links.py`
  line 17). Test is referenced by notebook builders for submission.

**Performance / role in pipeline:**
Pending. No leaderboard submission since the val pool was solved on 2026-05-12.

**Verdict:** **active** (the final-submission input) — held in reserve.

---

### A.4 `data/sample_submission.csv`

**Files:**
- `e:\swiss_citation_extraction\data\sample_submission.csv` — 104 bytes, CSV

**Description:**
Tiny competition-format reference. Columns: `query_id, predicted_citations`. Two example
rows: `test_0001,Art. 111 ZGB;Art. 114 ZGB` and `test_0002,Art. 457 ZGB;Art. 462 ZGB`. The
`test_0001` IDs (4-digit) differ from the `test.csv` IDs (`test_001`, 3-digit) — the sample
is illustrative of format only, not a row-aligned template.

**Source:** Kaggle competition file.

**Purpose:** Documents the submission column layout (`predicted_citations` semicolon-joined).

**Notebooks that use it (verified by Grep):**
- **No notebook usage found in inventory.** Grep `sample_submission` against
  `notebooks/_inventory/` returned zero matches. The format is so simple notebooks build
  the submission CSV inline rather than loading the sample.

**Scripts that produce/consume it:**
- **No script usage found** (Grep against `scripts/` returned zero matches).

**Performance / role in pipeline:**
Reference only — never loaded, format learned by inspection.

**Verdict:** **active** as a competition contract reference, but **unused at runtime**.

---

## Family B — Corpus (competition-provided)

### B.1 `data/laws_de.csv`

**Files:**
- `e:\swiss_citation_extraction\data\laws_de.csv` — 72,970,627 bytes (≈ 70 MB), CSV

**Description:**
175,933 German law snippets. Columns: `citation, text, title`. Citation strings cover both
article-level (`Art. 1 ZGB`) and paragraph-level (`Art. 11 Abs. 2 OR`). The text column is
the full German article/paragraph body; title is the law's German title.

**Source:** Kaggle competition file.

**Purpose:** The law leg of the closed-vocabulary corpus — half of the universe of valid
predictions.

**Notebooks that use it (verified by Grep):**
- 26 files matched in `notebooks/_inventory/`. Key examples:
  - `notebooks\_inventory\reference_F1_0.777_Untitled75.md` — the F1=0.777 reference loads
    the law corpus
  - `notebooks\_inventory\retrieve_unified_corpus_v3.md`,
    `retrieve_unified_corpus_qwen3_8b.md` — corpus build for embeddings
  - `notebooks\_inventory\endgame_colab_full_pipeline.md`
  - `notebooks\_inventory\pipeline_end_to_end_*.md` family (8 notebooks)
  - `notebooks\_inventory\pipeline_iteration_latest_*.md` family (5 notebooks)
  - `notebooks\_inventory\exp_A1_bgem3_statutes.md`, `exp_A4_rerank.md`,
    `exp_A4b_rerank_hyde.md`, `exp_A3_qwen32b_enumeration.md`,
    `exp_A2_hyde_and_enumeration.md`, `exp_A5_court_retrieval.md`,
    `exp_A6_legal_roberta_finetune.md`
  - `notebooks\_inventory\early_recall_preserving_funnel.md`,
    `early_dense_embedding_test.md`

**Scripts that produce/consume it:**
- Consumers (13 hits): `scripts\citation_extraction\extract_citation_graph.py`,
  `extract_intra_judgment_backrefs.py`; `scripts\build_pipeline\build_law_authority_cards.py`;
  `scripts\enrichment\run_law_llm_enrichment.py`;
  `scripts\data_prep\repair_law_enrichment_utf8.py`, `repair_train_csv_utf8.py`;
  `scripts\diagnostics_loose\check_coverage.py`;
  `scripts\retrieval_and_rerank\segment_lattice_v3.py`;
  `scripts\eval_and_audit\experiment_enrichment_recall_stress.py`,
  `check_gold_parent_links.py`;
  `scripts\notebook_builders\_build_endgame_notebook.py`,
  `create_recall_funnel_notebook.py`.
- Producer of derivatives: `scripts\data_prep\split_laws_de.py` consumes `laws_de.csv` and
  writes the three `laws_de_segments/laws_de_part0*.csv` files (line 6-7 of the script).

**Performance / role in pipeline:**
Foundation of every retrieval pipeline. Indexed into the unified retrieval SQLite,
embedded by Qwen3-Embedding-8B, BM25-indexed, enriched into 173,033 LLM cards
(`llm_enrichment_output_law_173k/`).

**Verdict:** **active** — irreplaceable closed-vocabulary half.

---

### B.2 `data/court_considerations.csv`

**Files:**
- `e:\swiss_citation_extraction\data\court_considerations.csv` — 2,425,780,697 bytes
  (≈ 2.26 GB), CSV

**Description:**
2,476,315 court considerations (Erwägungen). Columns: `citation, text`. Languages: primarily
German, some French and Italian. Citation strings include both BGE leading-decision format
(`BGE 139 I 2 E. 2`) and docket format (`5A_800/2019 E 2.`).

**Source:** Kaggle competition file.

**Purpose:** The court leg of the closed-vocabulary corpus — the other half of valid
predictions. Mandatory because court citations make up ~40% of val gold (`swiss-citation-task`
"Court ~40% on val").

**Notebooks that use it (verified by Grep):**
- 36 files matched in `notebooks/_inventory/`. Includes the same large set as `laws_de.csv`
  plus court-specific notebooks:
  - `notebooks\_inventory\court_concept_enrichment_*.md` family (5)
  - `notebooks\_inventory\court_llm_descriptor_*.md` family (7)
  - `notebooks\_inventory\court_concept_smoke_test_10_v3_local.md`
  - `notebooks\_inventory\kaggle_*.md` (4)
  - `notebooks\_inventory\exp_A5_court_retrieval.md`
  - `notebooks\_inventory\early_segment_lattice_funnel_v3.md`
  - `pipeline_*` and `endgame_colab_full_pipeline.md` families

**Scripts that produce/consume it:**
- Consumers (10 hits): `scripts\citation_extraction\add_case_level_aliases.py`,
  `extract_intra_judgment_backrefs.py`, `extract_citation_graph.py`,
  `verify_french_italian_aliases.py`;
  `scripts\enrichment\court_enrichment_profile.py`;
  `scripts\build_pipeline\build_court_authority_cards.py`;
  `scripts\retrieval_and_rerank\segment_lattice_v3.py`;
  `scripts\eval_and_audit\check_gold_parent_links.py`;
  `scripts\notebook_builders\_build_endgame_notebook.py`,
  `create_recall_funnel_notebook.py`.

**Performance / role in pipeline:**
The dominant corpus by row count; the slowest to enrich (only ~15% / 363k rows enriched so
far per `swiss-citation-data` line 34). The remaining 85% un-enriched is the live coverage
gap deferred per endgame §11.

**Verdict:** **active** — irreplaceable. Re-extraction is on the banned-moves list
(`swiss-citation-task` table).

---

### B.3 `data/laws_de_full.json`

**Files:**
- `e:\swiss_citation_extraction\data\laws_de_full.json` — 52,049,860 bytes (≈ 52 MB), JSON

**Description:**
Full DE law text as a JSON document. File is too large to read in one shot (25M-token
overflow on the Read tool). Per `swiss-citation-data` SKILL line 19: "Full DE law text JSON
(moved from translation_work/)." Treated as a denser parsed view of `laws_de.csv`.

**Source:** Derived. The reorg log records the move:
`scripts\drive_pull\_full_repo_reorg_log.json` lines 401-402 — `src: ...\translation_work\laws_de_full.json` → `dst: ...\data\laws_de_full.json`. Originally produced
during the translation/enrichment work stream (not by a single canonical script in the
current repo layout — the producer pre-dates the current organization and is no longer in
`scripts/`).

**Purpose:** Convenience JSON snapshot of the law text used by some translation/enrichment
prototypes.

**Notebooks that use it (verified by Grep):**
- **No notebook usage found in inventory.** Grep `laws_de_full.json` against
  `notebooks/_inventory/` returned zero matches. The only references in the entire repo are
  in `.claude\skills\swiss-citation-data\SKILL.md` and the reorg log/script
  (`scripts\drive_pull\_full_repo_reorg_log.json`, `full_repo_reorg.py` lines 215-217).

**Scripts that produce/consume it:**
- Only the reorg-related scripts reference it (`full_repo_reorg.py`,
  `_full_repo_reorg_log.json`). No active consumer in `scripts/build_pipeline/`,
  `scripts/enrichment/`, or `scripts/retrieval_and_rerank/`.

**Performance / role in pipeline:**
Effectively cold storage — the canonical law-text source for downstream work is
`laws_de.csv`, indexed into `artifacts/unified_retrieval.sqlite` and embedded.

**Verdict:** **superseded / archival**. Useful as a parsed reference if a translation
project resumes, but not on the active retrieval path.
**Successor:** `data\laws_de.csv` + `artifacts\unified_retrieval.sqlite` +
`artifacts\law_authority_cards_v2_unified.jsonl`.

---

### B.4 `data/laws_de_segments/laws_de_part01.csv` + `_part02.csv` + `_part03.csv`

**Files:**
- `e:\swiss_citation_extraction\data\laws_de_segments\laws_de_part01.csv` — 31,457,552 bytes
- `e:\swiss_citation_extraction\data\laws_de_segments\laws_de_part02.csv` — 31,457,327 bytes
- `e:\swiss_citation_extraction\data\laws_de_segments\laws_de_part03.csv` — 10,055,790 bytes

**Description:**
`laws_de.csv` split into three ~30 MB CSV segments with the header preserved in each.
Same `citation, text, title` schema. Created so that environments with file-size or memory
ceilings (Kaggle attached datasets, mobile/Colab T4 chunked loops) can stream the law
corpus without loading 70 MB at once.

**Source:** Derived. Producer: `scripts\data_prep\split_laws_de.py` — script docstring
(line 1): *"Split data/laws_de.csv into ~30 MB CSV segments, preserving the header in each."*
Output directory hardcoded to `data/laws_de_segments/` (line 7).

**Purpose:** Chunked-processing convenience for the F1=0.777 reference notebook which
reads the corpus segment-by-segment.

**Notebooks that use it (verified by Grep):**
- `notebooks\_inventory\reference_F1_0.777_Untitled75.md` — line 1401:
  `fp=loc/f"laws_de_part{i}.csv"` and line 2175 (same pattern). The notebook iterates `i`
  over the three parts.

**Scripts that produce/consume it:**
- Producer: `scripts\data_prep\split_laws_de.py`.
- Consumer: none in `scripts/` — usage is notebook-only (the F1=0.777 reference).

**Performance / role in pipeline:**
Indirectly load-bearing — the reference notebook that achieved Macro F1 = 0.777 depends on
this chunked layout. Per `swiss-citation-data` line 96, the reference is preserved at
`notebooks/06_high_scoring_reference/reference_F1_0.777_Untitled75.ipynb`.

**Verdict:** **active** — kept because the F1=0.777 reference still reads them, and the
split costs nothing to keep on disk.

---

## Family C — Granularity-expanded gold (derived)

### C.1 `data/train_granularity_expanded.csv`

**Files:**
- `e:\swiss_citation_extraction\data\train_granularity_expanded.csv` — 2,040,082 bytes, CSV

**Description:**
Train queries with article-level gold citations expanded to corpus-aligned siblings.
Columns parallel `train.csv` but `gold_citations` now contains only citations that are
present as rows in the unified retrieval corpus (paragraph children substituted for bare
articles when paragraph-level rows exist).

**Source:** Derived. Producer: `scripts\data_prep\granularity_resolver.py` — CLI in
docstring (lines 23-25):
`python -m scripts.granularity_resolver --gold-csv data/train.csv --output data/train_granularity_expanded.csv`.
The resolver applies the rule: *"if Art. 11 Abs. 2 OR exists in the corpus, then Art. 11 OR
CANNOT be a valid gold citation in the corpus -- it must be expanded to all sibling
paragraph-children"* (script docstring lines 9-12).

**Purpose:** Lift train gold from the **71.5%** in-corpus rate to **98.06%** after
resolver (`swiss-citation-data` SKILL line 17). Makes train gold trainable against the
exact corpus rows.

**Notebooks that use it (verified by Grep):**
- `notebooks\_inventory\anchor_funnel_v1_val001_base.md` — line 73:
  `"train_expanded_csv": DATA_ROOT / "data" / "train_granularity_expanded.csv"`,
  line 90 (`OK` status check), line 461 (`Bedrock source: train_granularity_expanded.csv`),
  lines 1009 + 1024 (path references).
- `notebooks\_inventory\_extract_log.json` lines 1523, 1536 (build-log references).

**Scripts that produce/consume it:**
- Producer: `scripts\data_prep\granularity_resolver.py`.
- Consumer scripts: none direct (loaded only inside the anchor-funnel v1 notebook).

**Performance / role in pipeline:**
Foundational for any approach that uses train as a prior source — without expansion, 28.5%
of train gold is structurally unreachable, contaminating retrieval recall metrics measured
on train. Used as the "bedrock" in the anchor-funnel v1 base notebook.

**Verdict:** **active** — cheap, correct, no successor.

---

### C.2 `data/val_granularity_expanded.csv`

**Files:**
- `e:\swiss_citation_extraction\data\val_granularity_expanded.csv` — 19,860 bytes, CSV

**Description:**
Val queries with the same granularity resolver applied. The val change is minimal because
val gold is already 100% Class A (`swiss-citation-data` line 18: "Val gold post-resolver;
still 100%.") — every paragraph-level gold already exists as a corpus row.

**Source:** Derived. Producer: `scripts\data_prep\granularity_resolver.py` invoked with
`--gold-csv data/val.csv --output data/val_granularity_expanded.csv`.

**Purpose:** Pipeline symmetry — every train operation that consumes the expanded train CSV
can apply the same operation to val using the parallel file.

**Notebooks that use it (verified by Grep):**
- **No direct notebook usage found in inventory** (the only `granularity_expanded` matches
  in `notebooks/_inventory/` are train-side in `anchor_funnel_v1_val001_base.md` and
  `_extract_log.json`). In practice val is loaded directly from `val.csv` because the
  expanded form is identical for val.

**Scripts that produce/consume it:**
- Producer: `scripts\data_prep\granularity_resolver.py`.
- Consumers: none active.

**Performance / role in pipeline:**
Acts as a symmetric placeholder. Confirmed empirically that the resolver is a no-op on val
gold.

**Verdict:** **active but trivial** — kept for symmetry; no measurable downstream
consumer.

---

## Family D — Gold-citation analysis (under `data_insights/gold_analysis_coverage_and_parents/`)

Note: every file in this family was produced by a script under
`scripts/eval_and_audit/`. None are competition-shipped.

### D.1 `gold_citation_coverage.{csv,json,md}`

**Files:**
- `e:\swiss_citation_extraction\data_insights\gold_analysis_coverage_and_parents\gold_citation_coverage.csv` — 322,038 bytes
- `…\gold_citation_coverage.json` — 6,283,954 bytes
- `…\gold_citation_coverage.md` — 14,586 bytes

**Description:**
Per-gold-citation classification: which split mentioned it, how many times, family
(`law` / court), subfamily, pattern, whether it appears as the `citation` column in
`laws_de.csv` / `court_considerations.csv` ("Class A" source rows), whether only as a
text reference ("Class B"), or absent ("Class C"). The MD is the human summary.
Summary (from `gold_citation_coverage.md`):

| Split | Gold mentions | Unique gold | Pattern covered | Present in extracted graph | Missing from extracted graph | Text-reference only |
|---|---:|---:|---:|---:|---:|---:|
| train | 4,659 | 2,695 | 2,695 | 2,578 | 117 | 650 |
| val | 251 | 222 | 222 | 222 | 0 | 0 |
| combined | 4,910 | 2,878 | 2,878 | 2,761 | 117 | 650 |

The MD then enumerates the 117 train gold citations absent from the extracted graph
(LugÜ, FIDLEG/FIDLEV, FINIG/FinfraG, GBV, parts of URG/ZGB, etc.) — these are the
4% corpus gap acknowledged in `swiss-citation-experiments` "Resolved phantom issues" §6.3.

**Source:** Derived. Producer: `scripts\eval_and_audit\verify_gold_citation_coverage.py`
(reads `data/train.csv` + `data/val.csv` and the citation graph, classifies each gold,
writes the three files).

**Purpose:** Codify the **Class A / B / C** structural ceiling that pins train at 28.5%
reachable and val at 100% reachable. Drives the Obs 1 numbers in `swiss-citation-data`.

**Notebooks that use it (verified by Grep):**
- **No notebook usage found in inventory.** Grep `gold_citation_coverage` against
  `notebooks/_inventory/` returned zero matches. The artifact is consumed at the research
  / planning layer, not inside notebooks.

**Scripts that produce/consume it:**
- Producer: `scripts\eval_and_audit\verify_gold_citation_coverage.py`.
- Listed in `scripts\drive_pull\_full_repo_reorg_log.json` (path movement only) and
  `full_repo_reorg.py`.

**Performance / role in pipeline:**
**Foundational diagnostic.** It is the basis for the "structural ceiling" framing that
killed pre-v7.5 recall-fix attempts. Cited directly in `swiss-citation-task` Hard rules
table ("Class A only on val | `gold_citation_coverage.csv`").

**Verdict:** **active** — canonical evidence file.

---

### D.2 `gold_parent_link_check.{csv,json,md}`

**Files:**
- `…\gold_parent_link_check.csv` — 514,461 bytes
- `…\gold_parent_link_check.json` — 1,302,435 bytes
- `…\gold_parent_link_check.md` — 2,943 bytes

**Description:**
For every gold citation that has **no own text row**, check whether a parent citation is
present in the extracted citation graph and whether that parent appears in the same
query's gold. Columns:
`split, query_id, citation, present_as_text_reference, parent_count_global, parent_in_same_query_gold, same_query_gold_parents, sample_global_parents, status`.

Summary (from `gold_parent_link_check.md`):

| Split | No-own-text gold mentions | Same-query gold parent found | Has parents, none in same-query gold | No parent edge found |
|---|---:|---:|---:|---:|
| train | 1,340 | 73 | 1,105 | 162 |
| val | 0 | 0 | 0 | 0 |

**Source:** Derived. Producer: `scripts\eval_and_audit\check_gold_parent_links.py` (lines
17, 25-40 confirm it reads `train.csv` + `val.csv` and queries `extract_citation_graph`).

**Purpose:** Establish the **granularity / parent-child rule** that informs the granularity
resolver and the prediction-time constraint *"when a paragraph-level citation exists, the
article-level parent is NOT valid gold"* (`swiss-citation-task` Hard rules row). Quantifies
how often a no-own-text gold has a usable parent edge.

**Notebooks that use it (verified by Grep):**
- **No notebook usage found in inventory.** Grep `gold_parent_link_check` against
  `notebooks/_inventory/` returned zero matches.

**Scripts that produce/consume it:**
- Producer: `scripts\eval_and_audit\check_gold_parent_links.py`.
- Other matches in `scripts\drive_pull\_full_repo_reorg_log.json` are reorg path-move records
  only.

**Performance / role in pipeline:**
Same diagnostic role as D.1 — feeds the granularity resolver design and confirms val has
zero parent-link issues (val gold is fully reachable at the row level).

**Verdict:** **active** — canonical evidence file.

---

### D.3 `train_gold_citations.json`

**Files:**
- `…\train_gold_citations.json` — 62,617 bytes, JSON

**Description:**
A flat sorted JSON array of every distinct train gold citation string (first few entries
`Art. 1 Abs. 1 AVO`, `Art. 1 Abs. 1 HVUV`, …). Convenience dump of the unique-gold column
from `gold_citation_coverage.csv` for the train split.

**Source:** Derived. Same producer family as D.1 (the verify script writes this dump as a
companion artifact). Listed in the reorg log alongside the coverage / parent-link files.

**Purpose:** Convenience list — `set(train_gold_citations.json)` is the queryable gold
vocabulary for train.

**Notebooks that use it (verified by Grep):**
- **No notebook usage found in inventory.** Grep `train_gold_citations` against
  `notebooks/_inventory/` returned zero matches.

**Scripts that produce/consume it:**
- Producer: `scripts\eval_and_audit\verify_gold_citation_coverage.py` (companion output).
- Consumer: `scripts\notebook_builders\patch_notebook_no_leak.py` (verified hit). This is
  a notebook-builder utility ensuring gold isn't leaked into a candidate-generation cell.

**Performance / role in pipeline:**
Used for anti-leak checks during notebook construction (e.g., verifying that prediction
code does not accidentally hardcode gold strings). Not on the retrieval hot path.

**Verdict:** **active** — small artifact, useful for guard-rail scripts.

---

### D.4 `val_gold_citations.json`

**Files:**
- `…\val_gold_citations.json` — 5,673 bytes, JSON

**Description:**
Same shape as D.3 but for val. A flat JSON array; first entries are court IDs
(`1B_15/2023 E. 3.1`, `1B_192/2022 E. 4.1.2`, …) reflecting val's 40.6% court share.

**Source:** Derived. Same producer family as D.1 / D.3.

**Purpose:** Convenience list of every val gold string. Used as the truth-set lookup when
computing R@K and Macro F1 inside evaluation utilities.

**Notebooks that use it (verified by Grep):**
- **No notebook usage found in inventory.** Grep `val_gold_citations` against
  `notebooks/_inventory/` returned zero matches.

**Scripts that produce/consume it:**
- Producer: `scripts\eval_and_audit\verify_gold_citation_coverage.py`.
- Consumer: `scripts\notebook_builders\patch_notebook_no_leak.py` (same anti-leak utility
  as D.3).

**Performance / role in pipeline:**
Same role as D.3 but on val. Notebooks evaluate against val by re-parsing
`val.csv`'s `gold_citations` column rather than loading this JSON, which is why it has no
notebook hits.

**Verdict:** **active** — small artifact retained for guard-rail and offline auditing.

---

## Cross-cutting verdicts

| File | Verdict | Drives |
|---|---|---|
| `train.csv` | active | training prior; recall ceiling diagnostic |
| `val.csv` | active | the only reliable in-distribution eval |
| `test.csv` | active (held-out) | final submission |
| `sample_submission.csv` | active reference, unused at runtime | format contract |
| `laws_de.csv` | active | half of closed-vocabulary corpus |
| `court_considerations.csv` | active | other half (40% of val gold) |
| `laws_de_full.json` | superseded / archival | translation-prototype only |
| `laws_de_segments/laws_de_part0{1,2,3}.csv` | active | F1=0.777 reference notebook |
| `train_granularity_expanded.csv` | active | 71.5% → 98.06% in-corpus train gold |
| `val_granularity_expanded.csv` | active but trivial | symmetry placeholder |
| `gold_citation_coverage.{csv,json,md}` | active | Class A/B/C diagnostic |
| `gold_parent_link_check.{csv,json,md}` | active | parent-link granularity diagnostic |
| `train_gold_citations.json` | active | anti-leak guard-rails |
| `val_gold_citations.json` | active | anti-leak guard-rails |

## Sources of evidence

- Notebook references: `Grep` against `e:\swiss_citation_extraction\notebooks\_inventory\`
  with `output_mode="files_with_matches"`. Counts and call sites quoted above are direct
  Grep hits.
- Script references: `Grep` against `e:\swiss_citation_extraction\scripts\`. Producer
  scripts confirmed by `Read` against their docstrings and CLI patterns.
- Verdicts grounded in `.claude\skills\swiss-citation-experiments\SKILL.md` and
  `.claude\skills\swiss-citation-data\SKILL.md`.
- Official competition description quoted from `research\problem_statement.md` because the
  live Kaggle page requires login (WebFetch returned title-only).
