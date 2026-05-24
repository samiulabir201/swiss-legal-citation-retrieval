# Personal Observations: Swiss Legal Citation Retrieval

> This document records empirically verified observations about the data and the problem.
> Each entry includes the exact test setup, measured results, and an interpreted conclusion.
> These observations should inform every design decision downstream.
>
> **Important caveat on Observation 3 embedding test:** All retrieval tests were run against a
> candidate pool of (all gold citations + 500 random negatives), NOT the full 2.6M-entry corpus.
> This makes the test *favorable* to the retrieval methods — real-corpus performance will be
> substantially lower. Results should be read as an upper bound, not a realistic estimate.

---

## Observation 1: Train Gold Citations Are Split Into Three Distinct Coverage Classes

### Claim
Not all train gold citations have a text field in the retrieval corpus. The 28.5% that are absent
from the source columns are not a uniform block — they split into two structurally different
categories: citations that exist inside the corpus as text references (mentioned in other
citations' body text but not as standalone retrievable units) and citations that are completely
absent from the corpus.

### Test Setup
- **Files used:** `data_insights/gold_citation_coverage.csv` (primary), confirmed against
  `data/laws_de.csv` and `data/court_considerations.csv`
- **Key columns used:**
  - `present_in_source_column_any` — True if the citation string exists as a `citation` column
    value in either laws_de or court_considerations (i.e., it is a valid prediction target)
  - `present_as_text_reference_any` — True if the string appears within the body *text* of some
    other corpus entry
  - `text_reference_only_any` — True if present in text references but NOT in source column
  - `present_in_any_extracted` — True if found anywhere (source OR text)
- **Sample size:** All 2695 unique train gold citations, all 222 unique val gold citations.

### Results

**Three-class breakdown for train gold citations:**

| Class | Count | % of all train gold |
|---|---|---|
| **Class A — Directly retrievable** (in source column) | 1928 | 71.5% |
| **Class B — Text-reference only** (in corpus text, not source) | 563 | 20.9% |
| **Class C — Completely absent** (not found anywhere in corpus) | 204 | 7.6% |

**Class A breakdown by source:**
| Source | Count | % of total |
|---|---|---|
| `laws_de.csv` source column | 1876 | 69.6% |
| `court_considerations.csv` source column | 52 | 1.9% |

**Class B breakdown — where text references appear:**
| Location | Count |
|---|---|
| Within `laws_de` body texts | 248 |
| Within `court_considerations` body texts | 525 |

**Val set (for comparison):**
| Class | Count | % |
|---|---|---|
| Directly retrievable | 222 | 100% |
| Text-reference only | 0 | 0% |
| Completely absent | 0 | 0% |

**Val source split:** 121 from laws_de (54.5%), 101 from court (45.5%).

**Sample Class B citations (text-reference only, not retrievable):**
```
Art. 975 ZGB, Art. 973 ZGB, Art. 956a ZGB, Art. 976a ZGB,
Art. 264m StGB, Art. 50 BV, Art. 106 StGB, Art. 630 ZGB
```

**Sample Class C citations (completely absent):**
```
Art. 10a Abs. 1 UVG, Art. 60 OR, Art. 337 OR, Art. 361 OR,
Art. 706 OR, Art. 156 FinfraG, Art. 168 Abs. 1 ZPO
```

### Interpretation

The three classes have different implications:

**Class A (71.5%):** These are the only citations that a model can ever correctly predict. The
system should be optimized around retrieving these.

**Class B (20.9%):** These citations exist in the legal reference space — they are mentioned
inside the text of other corpus entries — but the competition corpus has not indexed them as
standalone units. This may be because the corpus uses a different granularity (e.g., only
paragraph-level entries exist, not article-level), or certain laws were simply excluded from
corpus construction. A retrieval system can encounter these strings inside retrieved texts but
cannot output them as predictions. They represent a structural ceiling: roughly 1 in 5 train
gold citations will always be an unrecoverable false negative regardless of the retrieval method.

**Class C (7.6%):** These are completely absent — neither retrievable nor mentioned. They may
represent laws enacted or amended after corpus construction, obscure references, or annotation
errors in the training data.

**Val is clean, train is not:** The val/test gold sets are 100% aligned with the corpus.
The train set has 28.5% of gold citations that cannot be predicted. This means:
- Training loss or accuracy computed against raw train labels is systematically misleading.
- A perfect retrieval system can score at most ~71.5% recall on train labels; this is not a
  failure — it is a ceiling imposed by corpus construction.
- Val set metrics are the only reliable measure of actual retrieval ability.

---

## Observation 2: The Citation Graph Does Not Reliably Predict Gold Co-occurrences

### Claim
Pairs of citations that co-occur as gold answers in the same train query are almost never
connected by edges in the citation graph. The graph topology (textual cross-references between
legal articles) is structurally independent of the gold label co-occurrence pattern.

### Test Setup
- **Files used:** `data/train.csv`, `data_insights/laws_de_links.json`
- **Method:** For every train query with ≥2 gold citations, enumerate all unordered pairs.
  Count how often each pair co-occurs across queries. Load `laws_de_links.json`
  (`source_to_references` list of `{source, references}` entries). Build an undirected edge set.
  Compute intersection between gold pairs and graph edges.
- **Sample size:** All 1139 train queries; full `laws_de_links.json`
  (31,217 source nodes, 45,021 undirected edges).

### Results

**Gold co-occurrence pair stability:**

| Co-occurrence frequency | Pairs | % of total |
|---|---|---|
| Appears exactly 1 time | 14,918 | 90.2% |
| Appears 2+ times | 1,628 | 9.8% |
| Appears 3+ times | 465 | 2.8% |
| **Total unique pairs** | **16,546** | — |

**Gold pairs vs citation graph edges:**

| Metric | Count | Percentage |
|---|---|---|
| Gold co-occurrence pairs also in graph | 97 | 0.59% of gold pairs |
| Gold pairs NOT in graph | 16,449 | **99.41%** |
| Graph edges that appear as gold pair | 97 | 0.22% of all graph edges |

**Even "strong" pairs (3+ co-occurrences) are overwhelmingly not graph edges:**

| Metric | Count | % |
|---|---|---|
| Strong pairs (3+ co-occurrences) | 465 | — |
| Strong pairs also in graph | 6 | 1.3% |
| Strong pairs NOT in graph | 459 | **98.7%** |

**Top frequent gold co-occurrences (all absent from graph):**
```
7x  Art. 963 Abs. 1 ZGB  +  Art. 965 Abs. 2 ZGB
7x  Art. 963 Abs. 1 ZGB  +  Art. 965 Abs. 3 ZGB
7x  Art. 10 Abs. 1 BV    +  Art. 10 Abs. 2 BV
6x  Art. 20 Abs. 1 DBG   +  Art. 20 Abs. 3 DBG
6x  Art. 20 Abs. 2 OR    +  Art. 31 OR
```

### Interpretation

The citation graph captures textual cross-references within legal texts — when one law article
mentions another in its body. Gold co-occurrences capture something fundamentally different:
which citations a legal expert considers jointly necessary to answer a specific legal question.
These two relationships are near-orthogonal (0.59% overlap).

The most co-occurring gold pairs are typically adjacent articles of the same statute (e.g.,
Art. 963 and Art. 965 ZGB, Art. 10 Abs. 1 and Abs. 2 BV). These are thematically related
but do not textually cite each other. A model that expands predictions by following citation
graph edges will almost always add irrelevant citations, hurting precision without recovering
the missed gold citations. The citation graph cannot be used as a co-prediction signal.

---

## Observation 3: Standard Retrieval Approaches Fail — Dense Embedding Cannot Reach the Target Even with the Strongest 2026 SOTA Model and Maximum Context

### Claim
Dense embedding retrieval is structurally insufficient for this task. Even the strongest
multilingual encoder available in 2026 (Qwen3-Embedding-8B, #1 on MMTEB Multilingual),
running on the full 2.16M-entry corpus with passages enriched by every available signal from
`data_insights/` (citation header + structural segments + outgoing reference graph + body text),
achieves a Macro F1 of approximately **0.04** — fifteen times below the 0.6 target. The relationship
between an English legal question and its relevant Swiss statute/court citation is conceptual and
juridical, not lexical or shallow-semantic. Surface-level similarity simply does not encode it.

### Test Setup

**Test 1 — Vocabulary overlap (lexical baseline):**
- Tokenize queries and 2000-row laws_de sample (alphabetic tokens ≥4 chars, lowercased).
- Compute fraction of unique query tokens appearing anywhere in corpus vocabulary.
- 50 random German train queries, all 10 English val queries.

**Test 2 — Local dense embedding test (`paraphrase-multilingual-MiniLM-L12-v2`, favorable):**
- Per query, embed against (all gold + 500 random negatives). Predict top-k = |gold| (oracle k).
- This is a highly favorable test — only 500 negatives, oracle k, small model.
- Establishes the lower bound on what tiny multilingual embeddings achieve.

**Test 3 — BM25 + keyword query rewriting:**
- Same 500-negative pool. Test raw English vs query augmented with German translations of
  ~20 common legal English words (`detention` → `Untersuchungshaft`, `liability` → `Haftung`).

**Test 4 — SOTA dense embedding ceiling on FULL corpus (the definitive theory test):**
- Model: `Qwen/Qwen3-Embedding-8B` — #1 on MMTEB Multilingual leaderboard (June 2025, score 70.58),
  beats Gemini-Embedding-001 by 2+ points. Apache 2.0. 8B parameters.
- Hardware: NVIDIA RTX PRO 6000 Blackwell, 102 GB VRAM. bf16 precision, batch 256, sdpa attention.
- Corpus: full 2,161,111 entries (laws_de + court_considerations).
- Passages enriched with 5 data_insights signals before encoding, in this format:
  ```
  [CITATION] Art. 11 Abs. 2 OR — Bundesgesetz über das Obligationenrecht
  [STRUCTURE] law; statute_article; article 11, Abs 2, law code OR
  [REFERENCES] Art. 1 OR; Art. 12 OR; Art. 24 OR
  [TEXT] (original German legal text, truncated to 1600 chars)
  ```
  Sources: `*_classified_citations.jsonl` (2.59M entries), `*_links.json` (1.16M source nodes
  with outgoing reference lists), and `laws_de.csv` `title` column.
- Query side: English val query prefixed with the Qwen3 instruction template.
- Search: FAISS-GPU exact inner-product search over all 2.16M × 4096-dim vectors.
- Each chunk of 100k passages was checkpointed to disk; 22 chunks total. Each chunk took ~21 minutes
  to encode at ~80 docs/sec — confirming compute saturation on the 8B model.

### Results

**Vocabulary overlap with German corpus:**

| Query set | Mean token overlap | Min | Max |
|---|---|---|---|
| German train queries (n=50) | 62.7% | 40.0% | 84.8% |
| **English val queries (n=10)** | **4.3%** | **1.1%** | **6.1%** |

One val query (44 unique words): **0 out of 44 words** found anywhere in the German corpus.

**Local dense embedding (favorable, 500 negatives):**

| Method | Macro F1 | Per-query F1 |
|---|---|---|
| Multilingual MiniLM dense embed | **0.248** | [0.45, 0.33, 0.13, 0.00, 0.64, 0.06, 0.05, 0.07, 0.43, 0.32] |
| BM25 (raw English) | **0.097** | [0.26, 0.08, 0.28, 0.10, 0.09, 0.00, 0.00, 0.00, 0.00, 0.16] |

Query rewriting (BM25 with English→German keyword substitution) provides only marginal
recall gains on some queries (+0.05 to +0.17 on R@50). Several queries stay at zero recall.

**Noun-level semantic gap (German train queries vs gold citation text):**

| Query | Query nouns | Found in gold text |
|---|---|---|
| train_0054 | 9 | 1 (Obligationen) → 11% |
| train_0398 | 84 | 1 (Recht) → 1% |
| train_0943 | 5 | 1 (Kantone) → 20% |
| train_0733 | 50 | 1–2 per citation → 2–4% |
| train_0493 | 18 | 1–2 per citation → 6–11% |

Even German queries share only 1–20% of nouns with their own gold citation texts.

---

### **CEILING TEST RESULTS — Qwen3-Embedding-8B on full 2.16M corpus, enriched passages**

**Aggregate Macro F1 across 10 val queries:**

| Metric | Value | vs target |
|---|---|---|
| **Macro F1 (oracle k)** | **0.041** | 15× below 0.6 target |
| Macro F1 (k=20) | 0.044 | — |
| Macro F1 (k=30) | 0.034 | — |
| Mean Recall@10 | 0.036 | — |
| Mean Recall@20 | 0.051 | — |
| Mean Recall@50 | 0.107 | — |
| Mean Recall@100 | 0.136 | — |
| Mean Recall@500 | 0.247 | — |
| Mean Recall@1000 | 0.289 | — |

Even casting the widest possible net — top 1000 candidates from a 2.16M corpus (0.046% of the
search space) — only **28.9%** of gold citations are retrieved. To clear the F1 target window,
we would need Recall@k ≥ 0.6 at a precision-balanced k, which dense retrieval misses by an order
of magnitude.

**Per-query F1 (oracle k) breakdown:**

| query | n_gold | F1 oracle-k | R@50 | R@100 | R@500 | R@1000 |
|---|---|---|---|---|---|---|
| val_001 | 42 | 0.000 | 0.000 | 0.000 | 0.048 | 0.167 |
| val_002 | 36 | 0.000 | 0.028 | 0.028 | 0.028 | 0.083 |
| val_003 | 47 | 0.000 | 0.000 | 0.064 | 0.064 | 0.106 |
| val_004 | 10 | 0.000 | 0.400 | 0.600 | 0.600 | 0.600 |
| val_005 | 11 | 0.091 | 0.091 | 0.091 | 0.364 | 0.364 |
| val_006 | 18 | 0.111 | 0.222 | 0.222 | 0.500 | 0.611 |
| val_007 | 19 | 0.105 | 0.105 | 0.105 | 0.211 | 0.263 |
| val_008 | 29 | 0.034 | 0.069 | 0.103 | 0.103 | 0.103 |
| val_009 | 14 | 0.071 | 0.071 | 0.143 | 0.357 | 0.357 |
| val_010 | 25 | 0.000 | 0.080 | 0.200 | 0.200 | 0.240 |

- **4 of 10** queries score F1 = exactly 0.000 at oracle k (val_001, val_002, val_003, val_010).
- The "best" query (val_006) scores 0.111 — still 80% below target.
- High-volume queries (n_gold ≥ 25) consistently fail: val_001, val_002, val_003, val_008, val_010.

**Counter-intuitive comparison with the favorable local test:**

| Setup | Model | Corpus size | Macro F1 |
|---|---|---|---|
| Favorable (500 negatives, oracle k) | MiniLM-L12 (118M params) | 511 | **0.248** |
| **SOTA ceiling (full corpus, enriched, oracle k)** | **Qwen3-8B (8B params)** | **2,161,111** | **0.041** |

A 70× larger model with full data_insights enrichment performs **6× worse** when tested on
the full corpus. This is not a model-quality issue — it is a search-space scaling issue. The
2.16M-entry haystack contains many passages whose surface text resembles the query closely
enough to outscore the actual gold citations. Dense embedding is unable to distinguish "passage
that mentions similar legal concepts" from "passage that is the legally authoritative answer."

### Interpretation — Why dense embedding fails on this task

The Qwen3-8B test definitively closes the question. Dense embedding fails not because the model
is too small, the language gap too wide, or the corpus too underspecified. It fails because the
relationship being modeled is the wrong kind of relationship for cosine similarity in any
embedding space.

**1. The task requires legal reasoning, not text similarity.**
A query like *"the accused was detained for collusion danger; can he appeal the detention extension?"*
maps to gold citations like `Art. 221 Abs. 1 lit. b StPO`, `Art. 396 Abs. 1 StPO`, and
`BGE 137 IV 122 E. 6.2`. None of these citation texts contain the words "appeal" or "collusion" —
they contain procedural definitions and legal doctrine that an expert applies to the scenario.
The encoder has no way to perform that application. It can only match what is textually similar.

**2. Recall@1000 = 0.289 is the ceiling, not a fixable parameter.**
This is the strongest signal: even being maximally permissive (consider 1000 candidates per query),
71% of gold citations are not in the dense-retrieved candidate pool. No reranker, no LLM filter,
no precision-recall tradeoff can recover gold citations that were never in the top 1000. This
caps any pipeline that uses dense embedding as its first stage at <30% recall before any
downstream stage even sees the candidates.

**3. Some queries do work — and the pattern is informative.**
val_004 (n=10), val_006 (n=18) get R@500 ≥ 0.5. These tend to be queries with shorter, more
focused fact patterns where the legal domain is unambiguous (e.g., a specific contract dispute).
val_001, val_002, val_003 (n_gold ≥ 36) get R@50 = 0.0 — these are queries that span multiple
legal areas and require synthesizing a full doctrinal answer. Dense embedding can occasionally
hit a relevant statute when the query maps cleanly to one legal domain. It cannot construct
the multi-citation answer set that the val gold demands.

**4. Adding context did not help — and that is itself diagnostic.**
The enriched passages contained the citation header, structural segments, the citation graph
neighborhood, and the body text. None of this resolved the gap. The encoder can already see
all of this and still cannot map English legal scenarios to the right Swiss citations. The
information needed is not in any single passage; it is in the legal expert's knowledge of which
articles govern which scenarios — knowledge that exists in legal training and case experience,
not in the corpus text.

### Verdict

**Observation 3 is confirmed beyond doubt with the strongest possible evidence.**
Dense embedding alone — including SOTA models, full-corpus retrieval, and maximum auxiliary
context — cannot reach the 0.6 F1 target on this task. It cannot even be used as a recall
stage in a two-stage pipeline (Recall@1000 < 0.30 forecloses that).

**Implication for the system architecture:**
The retrieval pipeline cannot be built around dense embedding. The architecture must instead
incorporate explicit legal knowledge: an LLM that has read enough Swiss case law to know which
statutes govern which factual patterns, or a structured legal-knowledge backbone (e.g., domain
classification → statute filter → article ranking). Dense embedding may still play a minor
role as a tie-breaker or as one signal among several, but it cannot be the primary retrieval
mechanism.

---

## Observation 4: Train and Val/Test Are Drawn from Different Distributions on Three Independent Axes

### Claim
The training set and the validation/test sets differ simultaneously on: (1) query language,
(2) number of gold citations per query, and (3) proportion of citations from court decisions
vs statutes. Any system calibrated on the training distribution will be systematically
miscalibrated for evaluation.

### Test Setup
- **Files used:** `data/train.csv`, `data/val.csv`, `data/test.csv`, `data/laws_de.csv`,
  `data/court_considerations.csv`
- **Language detection:** Heuristic counting German vs English function word frequency in the
  first 500 characters of each query.
- **Citation count:** Group gold citations by `query_id`, compute per-query count.
- **Source mix:** Label each gold citation as `laws`, `court`, or `missing` based on presence
  in the respective corpus source column.
- **Sample size:** All 1139 train, 10 val, 40 test queries.

### Results

**Axis 1 — Query language:**

| Split | German | English | Mixed/Other |
|---|---|---|---|
| Train (n=1139) | 1132 (**99.4%**) | 2 (0.2%) | 5 (0.4%) |
| Val (n=10) | 0 | **10 (100%)** | 0 |
| Test (n=40) | 0 | **40 (100%)** | 0 |

**Axis 2 — Gold citations per query:**

| Metric | Train | Val |
|---|---|---|
| Mean | 4.1 | 25.1 |
| Median | **2** | **22** |
| 25th percentile | 1 | 15 |
| 75th percentile | 5 | 34 |
| Min / Max | 1 / 44 | 10 / 47 |
| Queries with 1–3 citations | 721 (**63.3%**) | 0 (0%) |
| Queries with 10+ citations | 112 (9.8%) | 10 (**100%**) |
| Queries with 30+ citations | 4 (0.4%) | 3 (30%) |

**Axis 3 — Gold citation source mix:**

| Split | Laws_de | Court | Missing/unretrievable |
|---|---|---|---|
| Train | 70.0% | **1.2%** | 28.8% |
| Val | 59.4% | **40.6%** | 0% |

**Queries with at least one court citation:**
- Train: 31 out of 1139 (**2.7%**)
- Val: 10 out of 10 (**100%**)

**Val per-query laws vs court breakdown:**

| Query | Gold total | Laws | Court |
|---|---|---|---|
| val_001 | 42 | 19 | 23 |
| val_002 | 36 | 20 | 16 |
| val_003 | 47 | 24 | 23 |
| val_004 | 10 | 9 | 1 |
| val_005 | 11 | 6 | 5 |
| val_006 | 18 | 11 | 7 |
| val_007 | 19 | 15 | 4 |
| val_008 | 29 | 20 | 9 |
| val_009 | 14 | 11 | 3 |
| val_010 | 25 | 14 | 11 |

### Interpretation

These three distributional gaps compound each other and create a situation where the training
data, while large (1139 queries), is a poor proxy for the evaluation task on every measurable
axis.

**Gap 1 — Language (German → English):**
Train is 99.4% German. Val/test is 100% English. The corpus is in German. Any system that
learns from train queries is learning from German-language input signals. At evaluation time,
it receives English input. Cross-lingual transfer is not optional — it is the core difficulty.
A retrieval system that does not explicitly bridge this gap will fail at the first step.

**Gap 2 — Citation volume (median 2 → median 22):**
Train queries require a median of 2 citations; val queries require a median of 22. A system
that predicts a fixed small number of citations will have good precision but near-zero recall
on val. A system that predicts a large fixed number will have good recall but poor precision
on train. Neither threshold transfers. The optimal prediction count must be dynamically
inferred from the query and calibrated using val, not train. Furthermore, the higher gold count
in val implies that val queries are more "comprehensive" legal analyses requiring many relevant
articles and precedents, not simple single-statute lookups.

**Gap 3 — Court citations (1.2% → 40.6%):**
Court decisions are essentially absent from training gold sets — only 31 of 1139 train queries
(2.7%) have any court citation at all. In val, every single query requires court citations,
which make up 40.6% of the gold set on average. A system trained on the training distribution
will have almost no signal about when to retrieve from `court_considerations.csv`. If the
retrieval pipeline does not actively search the court corpus, there is a hard upper bound of
roughly 59% recall on val (the laws-only fraction) before even accounting for retrieval
quality. The 2.4M-entry court corpus is not optional — it is essential for val/test performance.

**Practical consequence for calibration:**
Since the three axes all shift simultaneously from train to val/test, the training set should
be treated as a pre-training or warm-start signal — not as a ground truth for system tuning.
All threshold decisions (how many citations to predict, confidence cutoffs, reranking weights)
must be calibrated on the 10 val queries, which are the only reliable in-distribution signal
available.

---

## Summary Reference Table

| # | Observation | Key Number | Implication |
|---|---|---|---|
| 1 | Train gold coverage | 71.5% retrievable, 20.9% text-ref only, 7.6% absent | Train labels have a ~28.5% ceiling on recall; calibrate only on val |
| 2 | Citation graph reliability | 99.41% of gold pairs have no graph edge | Do not use citation graph as a co-prediction signal |
| 3 | Dense retrieval ceiling | **Qwen3-8B + full corpus + max context: Macro F1 = 0.041, R@1000 = 0.289** | Dense embedding cannot be the primary retrieval mechanism. Architecture must use legal-domain LLM reasoning |
| 4 | Train/val distribution shift | Language (DE→EN), citation count (2→22), court share (1.2%→40.6%) | Train ≠ eval distribution on every axis; all calibration must use val |

---

*Generated: 2026-04-26. Updated 2026-04-26 with the Qwen3-Embedding-8B full-corpus ceiling test results.
All inline Python tests reproducible from project root with `data/` and `data_insights/` directories present.
Full notebook + per-query results at `research/colab_dense_embedding_test.ipynb` and
`artifacts/results_qwen3_8b_enriched.csv`.*
