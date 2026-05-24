# Train + Val Law-Gold Research

_Generated 2026-05-23. Focus: identify query/gold inventory, laws_de-source gold, law enrichment/KB files, and retrieval/cardinality implications._

## Files Identified

| Role | Path |
|---|---|
| train queries + gold | `E:\swiss_citation_extraction\data\train.csv` |
| val queries + gold | `E:\swiss_citation_extraction\data\val.csv` |
| raw laws_de exact target corpus | `E:\swiss_citation_extraction\data\laws_de.csv` |
| law enrichment file | `E:\swiss_citation_extraction\artifacts\law_authority_cards_v2_unified.jsonl` |
| law enrichment summary | `E:\swiss_citation_extraction\artifacts\law_authority_cards_v2_unified.summary.json` |
| laws knowledge base file | `E:\swiss_citation_extraction\drive_sync\omnilex_competition\retrieval_other_artifacts\laws_knowledge_base.jsonl` |
| BM25 law corpus | `E:\swiss_citation_extraction\drive_sync\omnilex_competition\retrieval_other_artifacts\corpus_v2.parquet` |
| BM25 law index + ids | `E:\swiss_citation_extraction\drive_sync\omnilex_competition\retrieval_other_artifacts\bm25_v2_index.pkl + E:\swiss_citation_extraction\drive_sync\omnilex_competition\retrieval_other_artifacts\bm25_v2_ids.pkl` |

## Split Summary

`data/laws_de.csv` has **175,933 rows** and **175,933 unique citation strings**.
`law_authority_cards_v2_unified.jsonl` scanned/wrote **175,933** records; source split: {'llm+static': 173033, 'static': 2900}.

| split | queries | all gold | law gold | laws_de-source law gold | laws_de coverage | law gold/query min/median/max | explicit query hits | explicit coverage |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| train | 1139 | 4659 | 4602 | 3262 | 0.709 | 0/2.0/43 | 405 | 0.088 |
| val | 10 | 251 | 149 | 149 | 1.000 | 6/14.5/24 | 9 | 0.060 |

Interpretation:

- `laws_de-source law gold` means the exact prediction target is present as a row in `data/laws_de.csv`, not merely mentioned inside another law/court text.
- Val is clean for law retrieval: every val law gold citation is a `laws_de.csv` row.
- Train has a non-trivial missing/source-mismatch tail, so a strict laws_de-only system cannot reach law F1=1.0 on all train law gold unless those missing targets are repaired or mapped.

Top missing exact laws_de-source codes:

| split | missing code counts |
|---|---|
| train | ZGBx218; ORx184; StGBx129; BVx75; DSGx54; LugÜx43; URGx41; ZPOx40; IPRGx39; JStGx36 |
| val | (none) |

## Per-Query Law Gold Counts

| query_id | split | total gold | law gold | laws_de-source | top code mix | explicit query match | cascade tier | K_law_pred |
|---|---|---:|---:|---:|---|---:|---|---:|
| val_001 | val | 42 | 19 | 19 | BGGx1;StBOGx2;StGBx1;StPOx15 | 1 | cascade_heavy | 40 |
| val_002 | val | 36 | 20 | 20 | ATSGx7;BGGx3;IVGx10 | 1 | cascade_heavy | 16 |
| val_003 | val | 47 | 24 | 24 | BGGx1;BVx1;StBOGx2;StPOx15;ZGBx5 | 1 | cascade_heavy | 23 |
| val_004 | val | 10 | 9 | 9 | BGGx1;ORx1;ZGBx7 | 0 | cascade_moderate | 10 |
| val_005 | val | 11 | 6 | 6 | BGGx1;ZGBx5 | 0 | cascade_moderate | 9 |
| val_006 | val | 18 | 11 | 11 | BGGx2;ORx9 | 3 | cascade_moderate | 8 |
| val_007 | val | 19 | 15 | 15 | IPRGx2;ORx2;StGBx1;ZGBx10 | 3 | cascade_moderate | 10 |
| val_008 | val | 29 | 20 | 20 | BGGx2;BVx2;StGBx10;StPOx6 | 0 | cascade_moderate | 8 |
| val_009 | val | 14 | 11 | 11 | BGGx1;ZGBx10 | 0 | cascade_moderate | 9 |
| val_010 | val | 25 | 14 | 14 | BGGx1;ORx6;SchKGx1;ZGBx2;ZPOx4 | 0 | cascade_moderate | 9 |
| train_0001 | train | 3 | 3 | 2 | USGx1;UVPVx1 | 0 | substantive_only | 2 |
| train_0002 | train | 11 | 11 | 6 | ZGBx6 | 0 | substantive_only | 2 |
| train_0003 | train | 1 | 1 | 0 |  | 0 | substantive_only | 2 |
| train_0004 | train | 4 | 4 | 4 | IPRGx4 | 0 | substantive_only | 2 |
| train_0005 | train | 6 | 6 | 5 | BGGx5 | 0 | substantive_only | 2 |
| train_0006 | train | 3 | 3 | 2 | StGBx2 | 0 | substantive_only | 2 |
| train_0007 | train | 3 | 3 | 2 | ZGBx2 | 0 | substantive_only | 2 |
| train_0008 | train | 1 | 1 | 1 | DBGx1 | 0 | substantive_only | 2 |
| train_0009 | train | 3 | 3 | 1 | ORx1 | 0 | substantive_only | 2 |
| train_0010 | train | 14 | 14 | 8 | ORx1;StGBx7 | 0 | substantive_only | 2 |
| train_0011 | train | 2 | 2 | 1 | StGBx1 | 0 | substantive_only | 2 |
| train_0012 | train | 3 | 3 | 1 | FinfraGx1 | 0 | substantive_only | 2 |
| train_0013 | train | 5 | 5 | 2 | DSGx2 | 0 | substantive_only | 2 |
| train_0014 | train | 2 | 2 | 0 |  | 0 | substantive_only | 2 |
| train_0015 | train | 7 | 7 | 6 | AIGx4;AsylGx1;BVx1 | 0 | substantive_only | 2 |
| train_0016 | train | 9 | 9 | 7 | BVx1;StromVGx5;StromVVx1 | 1 | substantive_only | 2 |
| train_0017 | train | 4 | 4 | 4 | FIDLEGx3;FIDLEVx1 | 0 | substantive_only | 2 |
| train_0018 | train | 5 | 5 | 3 | BGGx3 | 1 | substantive_only | 2 |
| train_0019 | train | 4 | 4 | 4 | ORx4 | 0 | substantive_only | 2 |
| train_0020 | train | 5 | 5 | 3 | AsylGx3 | 0 | substantive_only | 2 |

_The CSV contains every train row; the table above shows all val rows plus the first 20 train rows for scanability._

## Existing Law-Only Retrieval Baseline

Using the already-built BM25+TLF law channel from `artifacts/law_only_oracle_k_full.*`:

| split | fixed F1@5 | fixed F1@10 | fixed F1@12 | F1 at exact law-gold count | oracle-K F1 | median oracle K | K_pred F1 | K_pred MAE vs law count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 0.316 | 0.266 | 0.247 | 0.429 | 0.544 | 3.0 | 0.327 | 2.27 |
| val | 0.550 | 0.705 | 0.714 | 0.781 | 0.864 | 11.0 | 0.739 | 5.70 |

Key finding: knowing the exact number of laws to output is not enough with the current BM25 ranking. Even `K = exact law-gold count` averages well below 1.0, and even oracle-K over the top-60 BM25 list is below 1.0. The missing piece is candidate-pool recall plus a stronger inclusion scorer, not only cardinality.

## Retrieval Mechanism Toward Law-Gold F1 = 1.0

The mechanism should be channel-based, with cardinality decided after scoring rather than by a fixed K:

1. **Exact statute parser**: parse explicit `Art./Article/Artikel` mentions in the query; normalize aliases such as `LAI -> IVG`, `CPP -> StPO`, `LTF -> BGG`; lookup in `laws_de.csv`; expand parent articles to existing paragraph rows when needed.
2. **Doctrine-heading channel**: mine German doctrine headings from `laws_de.csv.title` and `law_authority_cards_v2_unified.jsonl.retrieval_views.title_view`; map English query concepts to those headings; retrieve exact article rows by heading/title BM25.
3. **Procedural apparatus channel**: classify legal area/posture from the query and add the Swiss procedural bundle implied by that area, e.g. BGG appeal deadline, StPO Beschwerde/cost/jurisdiction articles for criminal-procedure cascades, ATSG/IVG procedural provisions for social insurance.
4. **One-hop law graph channel**: from the candidates above, parse outbound statute references in `laws_de.csv.text` and `normalized_anchors.statute_anchors`; add foundational definitions and sibling paragraphs only one hop deep.
5. **Precision/cardinality scorer**: train/calibrate an inclusion model over channel flags, BM25 ranks, enrichment fields (`concepts_en`, `terms_de_to_en`, `legal_rule`, `applicability_conditions`, `provision_role_llm`, `law_context_view`, `statute_anchor_view`), and query features. Select every candidate with calibrated `P(gold | query,candidate)` above a split-safe threshold; use predicted count only as a soft prior or tie-breaker.

For exact macro F1, two conditions must hold: candidate recall must be 1.0 for law gold, and the inclusion scorer must have no false positives. The existing files provide the raw material for this, but the current BM25+TLF channel alone does not meet those conditions.

## Artifacts Written

- `E:\swiss_citation_extraction\research\train_val_law_gold_inventory_2026-05-23.csv`: per-query inventory, query text, gold counts, laws_de-source gold list, explicit matches, cascade/cardinality features, BM25 F1 columns.
- `E:\swiss_citation_extraction\research\train_val_law_gold_citation_rows_2026-05-23.csv`: one row per gold citation with `is_law_citation`, `is_laws_de_source`, code, family, and query.
- `E:\swiss_citation_extraction\research\train_val_law_gold_summary_2026-05-23.json`: aggregate summary and file map.
- `E:\swiss_citation_extraction\research\train_val_law_gold_research_2026-05-23.md`: this report.
