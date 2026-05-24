# Structured Law Retrieval v1

This prototype turns `laws_de.csv` into structured retrieval fields and tests fielded BM25 channels over exact laws_de-source law gold.

## Structured Data

- Structured table: `E:\swiss_citation_extraction\research\_structured_law_v1\structured_laws_v1.parquet`
- BM25 index cache: `E:\swiss_citation_extraction\research\_structured_law_v1\structured_bm25_indices_v1.pkl`
- Fields: citation/code/article/paragraph, law title, doctrine heading, German text, English concepts, bilingual terms, legal rule/conditions, procedural role, sibling and statute anchors.

## Retrieval Channels

1. `citation`: citation/code/article/paragraph aliases, for exact statute references.
2. `heading_de`: law title and doctrine heading, using German query translation.
3. `terms_de`: isolated German legal terms from enrichment and headings.
4. `terms_en`: English translated terms and concepts from enrichment.
5. `rule_en`: English summary, legal question/rule, applicability conditions, addressees, consequences.
6. `body_de`: raw law body plus statute-anchor text.
7. `all`: combined fallback field.

## Results

| split | n | median gold | recall@100 | recall@250 | recall@500 | queries R@250=1 | F1@gold_count | F1@gapK | oracle F1 | median oracle K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 42 | 2.0 | 0.246 | 0.275 | 0.280 | 10 | 0.097 | 0.089 | 0.151 | 1.0 |
| val | 10 | 14.5 | 0.169 | 0.233 | 0.271 | 0 | 0.096 | 0.114 | 0.151 | 3.0 |

## Mechanism Implication

The structured channels should be used as a recall pool, not as a final top-K ranker. Exact law citation output requires a calibrated inclusion layer over these channel features. The feature vector should include every channel rank/score, explicit reference flag, same-code flag, provision role, legal domain, graph-sibling flag, and one-hop statute-anchor flag.

The current prototype already exposes where the remaining misses are via `missing_at_250`. Those misses identify which structured channels need another signal, usually a procedural apparatus prior or a better doctrine-heading translation.

Per-query diagnostics: `E:\swiss_citation_extraction\research\_structured_law_v1\structured_law_retrieval_v1_per_query.csv`