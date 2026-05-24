# Enrichment-First RAG Plan for Swiss Legal Citations

## Summary

The goal is to retrieve the right Swiss legal citations for English legal queries from two very large citation families:

- law citations from `data/laws_de.csv`
- court citations from `data/court_considerations.csv`

The old segment-lattice funnel tried to reduce candidates mostly with coarse metadata such as law code, BGE division, or docket prefix. That is not enough for court citations because even correct court buckets can contain tens or hundreds of thousands of rows. The new plan is enrichment-first: every law and court citation row becomes a structured RAG card with English legal concepts, multilingual search terms, legal questions, rules, holdings, and deterministic metadata. Retrieval then uses a multi-channel union over those enriched JSONL files to reduce roughly `~178k` law citations and `~2.4M` court citations to fewer than `1000` combined candidates while targeting `0.8-0.9` recall. A separate reranker then optimizes precision and macro F1.

Core principle:

```text
Retriever optimizes recall.
Reranker optimizes precision and macro F1.
```

Enrichment is query-independent. It must not use validation, test, or gold labels.

## 1. What We Are Enriching

Every citation row should become an authority card / RAG card. The model is not being asked to search millions of rows at query time. Instead, we preprocess each row into a compact structured record that future retrieval can search efficiently.

### Law Citation Input

Rows from `data/laws_de.csv` look like this:

```text
citation: Art. 5 Abs. 1 131.221
title: Verfassung des Kantons Solothurn, vom 8. Juni 1986 - I. Allgemeines
text: Wer öffentliche Aufgaben wahrnimmt, ist an Verfassung und Gesetz gebunden...
```

Example enriched law output:

```json
{
  "source_family": "law",
  "citation": "Art. 5 Abs. 1 131.221",
  "english_summary": "Persons performing public tasks are bound by the constitution and law and must act in the public interest while respecting equality and proportionality.",
  "legal_topic": "public law - legality, public interest, equality, proportionality",
  "legal_question": "What principles bind persons performing public tasks?",
  "legal_rule": "Public actors must act under constitutional and statutory authority, exclusively in the public interest, and respect equality and proportionality.",
  "english_legal_concepts": ["legality", "public interest", "equal treatment", "proportionality", "public duties"],
  "search_keywords": ["öffentliche Aufgaben", "Verfassung", "Gesetz", "öffentliches Interesse", "Rechtsgleichheit", "Verhältnismässigkeit"],
  "natural_language_queries": [
    "What principles apply to public authorities in Swiss public law?",
    "When must public actors respect proportionality and equality?"
  ]
}
```

For laws, enrichment should preserve deterministic metadata such as law code, article number, title, statute family, outgoing law references, and adjacent article information. The LLM should generate English summaries and search aliases, not invent citations.

### Court Citation Input

Rows from `data/court_considerations.csv` look like this:

```text
citation: BGE 148 II 169 E. 2.2
text: Nach Art. 28 Abs. 1 der Dublin-III-Verordnung dürfen die Mitgliedstaaten eine Person nicht allein deshalb in Haft nehmen...
```

Example enriched court output:

```json
{
  "source_family": "court",
  "citation": "BGE 148 II 169 E. 2.2",
  "court_base": "BGE 148 II 169",
  "english_summary": "Dublin detention requires an individualized assessment, significant flight risk, proportionality, and the ineffectiveness of less restrictive measures.",
  "legal_topic": "Dublin detention - flight risk and proportionality",
  "legal_question": "When may a person be detained to secure a Dublin transfer?",
  "legal_rule": "Detention for a Dublin transfer requires significant flight risk, proportionality, and no effective less restrictive measure.",
  "court_holding": "Detention is unlawful if national law lacks objective criteria for flight risk.",
  "english_legal_concepts": ["Dublin detention", "flight risk", "proportionality", "less restrictive measures"],
  "search_keywords": ["Dublin-Haft", "Fluchtgefahr", "Verhältnismässigkeit", "mildere Massnahmen"],
  "natural_language_queries": [
    "When is Dublin transfer detention lawful in Switzerland?",
    "What flight risk criteria are required for Dublin detention?"
  ],
  "paragraph_role": "reasoning",
  "outcome_signal": "none"
}
```

For courts, enrichment should preserve deterministic metadata such as `court_base`, `legal_area`, `authority_role`, `law_codes`, `statutes_cited`, `court_cases_cited`, BGE division, docket prefix, and citation-frequency signals. The LLM can summarize the legal issue and produce search terms, but deterministic metadata remains the source of truth for legal references.

## 2. Why `court_authority_cards_v4.jsonl` Was Created

`data/court_considerations.csv` has about `2.4M` court consideration rows. It is too large to send to an LLM at query time, and it is too noisy to search only with coarse docket or BGE division filters.

`artifacts/court_authority_cards_v4.jsonl` was created to make every court citation row usable as a deterministic retrieval object before LLM enrichment. It stores:

- citation identity: `citation`, `court_base`
- deterministic legal area from BGE division, docket prefix, law codes, and text signals
- `authority_role`, including leading, published, and frequently cited signals
- `law_codes`, `statutes_cited`, `court_cases_cited`
- structural citation-frequency signals, including source count and text-reference count
- original paragraph text
- deterministic fallback retrieval text

This v4 file solves the identity and metadata problem. It tells us what each court row is, what larger decision it belongs to, which law/court references appear around it, and whether it looks like an important authority.

It does not solve the semantic matching problem. English queries like "pretrial detention based on collusion risk" still do not naturally match German/French/Italian paragraphs containing terms such as `Untersuchungshaft`, `Kollusionsgefahr`, `Verdunkelungsgefahr`, or `détention provisoire`. That is why we need enrichment.

Current court enrichment artifacts:

- `artifacts/court_authority_cards_v4.jsonl`: deterministic court authority registry
- `artifacts/court_authority_cards_v4_enrichment_targets.jsonl`: selected high-value / weakly covered rows
- `artifacts/court_authority_cards_v4_target_cards.jsonl`: compact target-card file for LLM enrichment
- `artifacts/court_authority_cards_rag_targets_qwen3_8b.jsonl`: planned low-cost enriched target output
- `artifacts/court_retrieval_documents.jsonl`: final retrieval documents after applying enrichment
- `artifacts/court_retrieval.sqlite`: fielded BM25 / metadata retrieval database

The law side needs the equivalent deterministic/enriched JSONL asset for `data/laws_de.csv`.

## 3. Why The Old Segment-Lattice Funnel Failed

The failure was observed in:

```text
C:\Users\samiul\Downloads\colab_segment_lattice_funnel_v3 (1).ipynb
```

The old pipeline tried to reduce candidates using coarse hard filters:

- laws by statute or law-code segments
- courts by BGE division or docket prefix
- final top-k budget by equal scores and alphabetical tie-breaks

For courts this failed structurally:

- the planner often emitted no court decisions
- no court hard filter meant about `1.98M` court citations went straight to top-k
- even oracle division/docket filters left pools like `20k-400k` rows
- all candidates inside a bucket scored equally
- top-k dropped the gold court citations before reranking could help

The validation candidate summary showed court recall at `0.0` for all 10 validation queries in that run, even when the law side sometimes recovered part of the gold set. This means the failure was not only bad LLM planning. The deeper problem was that coarse legal taxonomy cannot rank within huge court buckets.

Example: selecting "criminal procedure" or docket prefix `1B` may be directionally correct, but it still leaves far more than `1000` court considerations. Without enriched issue-level fields, the system cannot know which rows discuss pretrial detention, collusion risk, concrete indicators, proportionality, admissibility, or remedy.

## 4. How Enriched JSONL Gives High Recall Under 1000 Candidates

For each query, the system starts with roughly:

```text
~178k law citations
~2.4M court citations
```

Using enriched JSONL, the retrieval target becomes:

```text
<1000 combined law + court candidates
target recall: 0.8-0.9
```

This should not be one brittle hard filter. It should be a multi-channel union where each channel contributes different recall signals, followed by dedupe and fusion.

### Law Retrieval Channels

Use the enriched law JSONL to retrieve from:

- BM25 over `english_summary`, `legal_question`, `legal_rule`, `english_legal_concepts`, and `search_keywords`
- vector search over the same enriched English fields
- deterministic statute and law-code parsing from the query
- adjacent article and same-statute expansion
- graph expansion from cited law references

Law retrieval should be strongest when the query names or implies a statute, law code, article family, or doctrinal topic. For example, a query about pretrial detention should surface `Art. 221 StPO` even if the English query never says `StPO` explicitly.

### Court Retrieval Channels

Use the enriched court JSONL to retrieve from:

- BM25 over enriched court RAG fields
- vector search over enriched court RAG fields
- legal-area soft filters
- statute-linked court retrieval using `statutes_cited`
- case-linked retrieval using `court_cases_cited`
- authority-score boosting using `published_leading_decision`, `frequently_cited_authority`, source count, and text-reference count
- narrow same-`court_base` expansion for highly ranked decisions

Court retrieval should be strongest when a query describes a legal test, holding, factual pattern, procedural posture, or doctrinal issue. The enriched fields let English concepts match multilingual legal text and known Swiss legal terminology.

### Example Query

```text
When can Swiss courts extend pretrial detention based on collusion risk, and what proportionality test applies?
```

Query understanding should produce:

```json
{
  "legal_area": "criminal procedure",
  "law_codes": ["StPO"],
  "candidate_statutes": ["Art. 221 StPO"],
  "english_concepts": ["pretrial detention", "collusion risk", "proportionality"],
  "multilingual_terms": ["Untersuchungshaft", "Kollusionsgefahr", "Verdunkelungsgefahr", "Verhältnismässigkeit"]
}
```

Expected retrieval behavior:

- law JSONL returns `Art. 221 StPO` and related detention/proportionality statutes
- court JSONL returns cases whose enriched fields mention pretrial detention, collusion risk, flight/collusion danger, concrete indicators, and proportionality
- authority boost moves leading BGE cases above ordinary single-mention rows
- final union and dedupe give fewer than `1000` candidates with high recall

This is how the enriched JSONL fixes the old failure. Instead of asking, "Which huge court bucket should we keep?", the retriever asks, "Which law and court authority cards are semantically and legally about this issue?"

## 5. Candidate Budget Design

Initial channel budgets:

```text
law BM25: 250
law vector: 250
law statute/code expansion: 150
court BM25: 300
court vector: 300
court statute/case linked: 200
court authority/graph expansion: 150
dedupe + fusion: <=1000 final candidates
```

The raw channel totals can exceed `1000` because the same citation should appear in multiple channels. After union, dedupe, and fusion, the final candidate list is capped at `<=1000`.

Use reciprocal-rank fusion or weighted score fusion. The score should combine:

- BM25 rank
- vector rank
- exact statute/law-code match
- legal-area match
- statute/case link match
- authority score
- paragraph-role boost or penalty
- same-`court_base` expansion score

Filters should be soft unless the query explicitly names a statute, law code, court citation, docket, or BGE. A soft filter changes score; a hard filter deletes candidates. Hard filters are dangerous because many queries describe legal scenarios without naming the exact authority.

## 6. How To Get Good Macro F1 From Top 1000

High recall top-1000 is not the final answer. Macro F1 needs reranking and selection.

The reranker should score each candidate against the query using both enriched JSON and original source text. It can be a stronger cross-encoder, a small LLM judge, or a feature-rich learning-to-rank model.

Reranking features:

- semantic relevance to the query
- legal-area match
- statute/court-case grounding
- authority score
- paragraph role
- exact citation or law-code match
- redundancy with already selected citations
- same-`court_base` grouping
- whether the candidate answers a distinct part of the query

Court reranking should group by `court_base` so the final prediction does not contain many irrelevant paragraphs from the same case. If multiple considerations from one decision rank highly, the system should select the best consideration citations and optionally retain the base decision as a grouping signal.

Final selection should be dynamic, not a fixed top-k for every query. The selector should use calibrated score thresholds and an estimated expected citation count. It should preserve diversity across:

- statutes
- court cases
- legal issues
- procedural roles
- authority types

The intended division of labor is:

```text
Retriever optimizes recall.
Reranker optimizes precision and macro F1.
```

## 7. Evaluation Plan

Evaluation should happen in stages.

First, compare enrichment quality:

- no-enrichment deterministic fallback vs enriched JSONL
- Qwen3-8B vs Qwen3.5-35B on the same 25-card sample
- JSON validity rate
- ungrounded statute/case hallucination rate
- empty or low-value enrichment rate
- multilingual quality across German, French, and Italian

Second, evaluate retrieval before reranking:

- recall@1000
- recall@2000
- law recall vs court recall
- recall by legal area
- recall by source family: law vs court
- candidate count after dedupe/fusion

Third, run retrieval ablations:

- BM25 only
- vector only
- BM25 + vector
- plus statute/legal-area filters
- plus graph expansion
- plus authority boosting
- plus same-`court_base` expansion
- plus reranker

Fourth, evaluate final task metrics:

- macro F1 after reranking
- precision/recall tradeoff by threshold
- per-query failure analysis
- dropped-gold analysis: whether gold was missing from retrieval or lost during reranking

Success criteria:

```text
pre-rerank candidate budget: <=1000 combined law + court candidates
pre-rerank recall target: 0.8-0.9
post-rerank goal: strong macro F1 through calibrated precision selection
no val/test/gold leakage
low ungrounded-reference rate
reproducible JSONL and SQLite retrieval artifacts
```

## Implementation Status And Next Steps

Already available:

- `artifacts/court_authority_cards_v4.jsonl`
- `artifacts/court_authority_cards_v4_enrichment_targets.jsonl`
- `artifacts/court_authority_cards_v4_target_cards.jsonl`
- court enrichment notebooks/scripts for Qwen3-8B, Qwen3.5-35B, and Claude experiments
- `scripts/build_court_retrieval_corpus.py` for court retrieval documents and SQLite indexing

Still needed:

- deterministic law authority-card builder for `data/laws_de.csv`
- law enrichment target selection and law enriched JSONL
- unified law + court retrieval corpus
- hybrid retrieval runner that produces `<=1000` candidates per query
- reranker and final selector
- ablation/evaluation notebook for recall@1000, recall@2000, and macro F1

The final production shape should be:

```text
data/laws_de.csv
  -> law_authority_cards_v1.jsonl
  -> law_authority_cards_rag.jsonl

data/court_considerations.csv
  -> court_authority_cards_v4.jsonl
  -> court_authority_cards_rag_targets_*.jsonl + deterministic fallback

law + court enriched JSONL
  -> unified retrieval documents
  -> BM25 index + vector DB + metadata tables
  -> top <=1000 recall candidates per query
  -> reranker/final selector
  -> final citations
```

## Assumptions

- Enrichment covers both `laws_de.csv` and `court_considerations.csv`.
- Court v4 cards already exist.
- Law-side enriched cards still need an equivalent deterministic/enriched JSONL asset.
- Enrichment is query-independent and must not use gold labels.
- Deterministic metadata remains the source of truth for citations, statutes, law codes, and court links.
- The recall target is `0.8-0.9` with fewer than `1000` combined law + court candidates before reranking.
- Reranking is required for macro F1; high-recall retrieval alone is not enough.
