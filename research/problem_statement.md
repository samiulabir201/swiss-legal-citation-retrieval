# Swiss Legal Citation Retrieval: Formal Research Problem Statement

---

## 1. Problem Definition

This work addresses the task of **cross-lingual legal citation retrieval** for Swiss law. Given an English-language natural language query describing a legal question or scenario, the system must identify and return a set of canonical citation identifiers from a fixed Swiss legal corpus that constitute the authoritative legal basis for answering that query.

This is formally a **set retrieval problem** — not a ranking problem, not a generative problem — where the output is an unordered set of citation strings drawn from a closed vocabulary, and correctness is judged by exact string match against a human-annotated gold set.

**Target performance:** Macro F1 score of **0.6 to 0.8** on the public leaderboard (50% of test queries).

---

## 2. Task Specification

### 2.1 Input

A natural language query `q` written in English, representing a legal question or scenario grounded in Swiss federal law. Queries are short to medium length and may reference specific legal concepts, parties, or factual situations.

### 2.2 Output

A set of citation strings `C = {c₁, c₂, ..., cₙ}` where each `cᵢ` is a canonical citation identifier drawn exactly from the retrieval corpus. Order within the set is irrelevant. The set may be empty.

### 2.3 Constraint: Closed Vocabulary

All valid predictions must be exact string matches to citation strings present in the retrieval corpus (`laws_de.csv` or `court_considerations.csv`). Any predicted string not present in the corpus is automatically a false positive, regardless of semantic proximity.

---

## 3. Retrieval Corpus

The corpus constitutes the full "vocabulary" of valid predictions. It is composed of two distinct sources:

### 3.1 Swiss Federal Laws — `laws_de.csv`

| Property | Detail |
|---|---|
| Language | German |
| Granularity | One row per law snippet (article or paragraph) |
| Citation key | Canonical string, e.g., `Art. 11 Abs. 2 OR` |
| Additional fields | `text` (full German article text), `title` (law title) |
| Size | ~175,933 citations, ~45,052 internal citation edges |

**Citation structure for laws:**
- Article-level (unsplit): `Art. 1 ZGB`, `Art. 117 StGB`
- Paragraph-level (split articles): `Art. 11 Abs. 2 OR`, `Art. 45 Abs. 2 AHVG`
- Critical constraint: when a paragraph-level citation exists (e.g., `Art. 11 Abs. 2 OR`), the article-level equivalent (`Art. 11 OR`) is **not** a valid gold citation. The corpus operates at the most granular level of available legislation.

### 3.2 Swiss Federal Court Decisions — `court_considerations.csv`

| Property | Detail |
|---|---|
| Language | Primarily German; some French and Italian |
| Granularity | One row per consideration (Erwägung) of a decision |
| Citation key | Canonical string identifying the specific consideration |
| Additional fields | `text` (full consideration text) |
| Size | ~2,476,315 citations, ~4,155,115 internal citation edges |
| Temporal coverage | Approximately 30 years back; older decisions may be absent |

**Citation structure for court decisions:**

Leading decisions (BGE — Bundesgerichtsentscheid):
```
BGE 116 Ia 56 E 1.
BGE 121 III 38 E. 2b
BGE 145 II 32 E. 3.1
```
Structure: `BGE [volume] [series] [page] E. [consideration number]`

Non-leading public decisions (docket-style):
```
5A_800/2019 E 2.
2C_123/2020 E 1.2.3
```
Structure: `[chamber_code]_[serial]/[year] E [consideration number]`

**Important:** Even when multiple rows originate from the same decision (e.g., multiple considerations of `BGE 139 I 2`), each consideration is a **distinct retrievable unit** and a distinct valid prediction ID.

---

## 4. Dataset Splits

### 4.1 Training Set — `train.csv`

| Property | Detail |
|---|---|
| Language | Non-English (German) queries |
| Source | LEXam benchmark (Fan et al., 2025), CC BY 4.0 |
| Gold citations | Extracted from "answer" fields of the LEXam open-question split |
| Distribution | Does **not** match the test/validation distribution |

The training set provides supervised signal for learning citation retrieval, but represents a **distribution shift** relative to evaluation — training queries are German, while evaluation queries are English.

### 4.2 Validation Set — `val.csv`

| Property | Detail |
|---|---|
| Size | 10 queries |
| Language | English |
| Gold citations | Provided |
| Distribution | Matches test distribution |

The validation set is small by design (annotation cost and inference time constraints). It is the primary in-distribution development signal.

### 4.3 Test Set — `test.csv`

| Property | Detail |
|---|---|
| Total size | 40 English queries |
| Public leaderboard | 20 queries (50%) |
| Private leaderboard | 20 queries (50%) |
| Gold citations | Hidden |
| Distribution | Matches validation distribution |

### 4.4 Hidden Set — `HIDDEN.csv`

A completely private set of queries, not in `test.csv`, reserved by the competition host to verify that solutions generalize and have not been trained or annotated on test data. Prize eligibility depends on re-evaluation on this set.

---

## 5. Evaluation Metric

### 5.1 Macro F1

Scoring is **citation-level Macro F1** computed per-query and averaged across queries.

For a single query with predicted set `P` and gold set `G`:

```
Precision = |P ∩ G| / |P|
Recall    = |P ∩ G| / |G|
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

The per-query F1 scores are then averaged uniformly across all queries (macro average — not weighted by citation count).

### 5.2 Implications of This Metric

**Macro averaging punishes query-level failures equally regardless of citation set size.** A query with 1 gold citation and a query with 20 gold citations contribute equally to the final score. This means the system must be consistently accurate across the full test distribution, not merely good on high-citation queries.

**Precision-Recall tradeoff is explicit.** Predicting more citations increases recall but decreases precision. The optimal number of predictions per query is not fixed — it is a function of how many citations a given query truly requires. A system that always returns the same number of citations will be suboptimal.

**Over-prediction is penalized.** Any citation in `P` that is not in `G` is a false positive that directly lowers precision for that query. Since the gold set size varies per query, there is no universally safe number of citations to return.

**Exact string matching.** An LLM-generated citation that differs from the corpus string by even one character (spacing, punctuation, case) scores as a false positive. All valid predictions must come from the exact strings present in the corpus.

### 5.3 Target Score Interpretation

| Macro F1 Range | Interpretation |
|---|---|
| < 0.3 | Baseline retrieval with limited relevance |
| 0.3 – 0.5 | Reasonable retrieval, significant gaps |
| **0.6 – 0.8** | **Research target: strong cross-lingual retrieval with good precision-recall balance** |
| > 0.8 | Expert-level; likely requires sophisticated multi-step reasoning |

The target range of **0.6 – 0.8** represents the regime where the system correctly identifies the majority of relevant citations while maintaining precision — a non-trivial objective given the cross-lingual setting, the size of the corpus (~2.6M court considerations alone), and the inherent ambiguity of legal interpretation.

---

## 6. Core Challenges

### 6.1 Cross-Lingual Retrieval Gap

Queries are in English. The corpus is in German (and partially French/Italian). There is no bilingual alignment provided. Bridging this gap — whether through translation, multilingual embeddings, or cross-lingual transfer — is a fundamental challenge.

### 6.2 Distribution Shift Between Train and Test

Training data (LEXam) consists of German-language queries from a law exam benchmark. Test/validation data consists of English legal scenarios of a different provenance. Directly fitting to the training distribution may not generalize to the evaluation distribution.

### 6.3 Corpus Scale

The court considerations corpus contains over 2.4 million entries. Exhaustive scoring of all candidate citations per query is computationally infeasible at runtime. Efficient retrieval — indexing, approximate nearest neighbor search, cascaded filtering — is a prerequisite.

### 6.4 Variable Gold Set Size

The number of relevant citations per query is not fixed and is not known in advance. Some queries require a single statute; others may require a dozen statutes and several court decisions. The system must implicitly or explicitly model this variability to avoid systematic over- or under-prediction.

### 6.5 Citation Granularity and Hierarchy

The corpus contains both article-level and paragraph-level statute citations, and both leading (BGE) and non-leading court decisions at the consideration level. Understanding which granularity is appropriate — and that a parent citation is not always valid if a child citation exists — requires structural awareness of the citation taxonomy.

### 6.6 Exact String Constraint

Since scoring requires exact string match, any generative component of the pipeline that produces citations (e.g., an LLM generating candidate citations) introduces the risk of format mismatches. The safest strategy is to constrain all predictions to the set of strings observed in the corpus.

### 6.7 Label Noise and Expert Disagreement

Gold citations represent the opinion of a domain expert annotator, not a provably complete or correct answer. Legal interpretation is inherently ambiguous — different experts may select different citation sets for the same query. This means a theoretical upper bound on F1 is below 1.0 even for a perfect retrieval system. The target range of 0.6–0.8 accounts for this ceiling.

---

## 7. Operational Constraints

These constraints are imposed by the competition and represent realistic production-adjacent requirements:

| Constraint | Specification |
|---|---|
| Execution environment | Offline Kaggle notebook (no internet access at inference time) |
| Maximum runtime | 12 hours per full submission |
| Inference cost budget | ≤ $10 per query (for scalability) |
| Model availability | All models and data must be pre-loaded as Kaggle datasets |
| Reproducibility | Host must be able to recreate submission from notebook |
| Generalization | Solution must perform on the hidden private set, not just public test |

The offline constraint eliminates external API calls (OpenAI, Anthropic, etc.) at inference time. Any LLM used must be locally hosted, either as a quantized open-weight model or as a preloaded asset.

---

## 8. What the Data Provides — A Complete Inventory

Given any citation from the retrieval corpus, the following information is available:

| Information Need | Source | Notes |
|---|---|---|
| The citation string itself (canonical ID) | `laws_de.csv` / `court_considerations.csv` — `citation` column | The prediction target |
| Full text associated with the citation | `laws_de.csv` / `court_considerations.csv` — `text` column | In German/French/Italian |
| Structured breakdown of citation components | `data_insights/*_classified_citations.jsonl` | Named segments: article number, paragraph, law code, volume, series, etc. |
| Other citations referenced within the text of a citation | `data_insights/laws_de_links.json` / `data_insights/court_considerations_links.json` | 45K + 4.1M directed citation edges respectively |
| Coverage of gold citations across train/val splits | `data_insights/gold_citation_coverage.csv` | Indicates whether gold citations are present in corpus |
| Parent-child citation relationships | `data_insights/gold_parent_link_check.csv` | Maps citations to the citations that reference them |

---

## 9. Research Objective Statement

The objective of this research is to develop a **cross-lingual legal citation retrieval system** for Swiss federal law that:

1. Accepts an English-language legal query as input
2. Retrieves a variable-size set of citation identifiers from a fixed corpus of ~2.6 million Swiss legal source documents
3. Achieves a **Macro F1 score between 0.6 and 0.8** on the public test set of 20 English queries
4. Operates fully offline within a 12-hour compute budget
5. Generalizes to unseen queries from the same distribution (the hidden evaluation set)

The research must contend with cross-lingual retrieval (English → German/French/Italian), massive corpus scale, variable gold set size per query, exact string prediction requirements, and inherent label noise arising from expert disagreement in legal interpretation.

---

*This document serves as the canonical problem statement. All research directions, experiments, and design decisions should be evaluated against the objectives and constraints recorded here.*
