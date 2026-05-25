# law_only_hybrid_v12_val_macroF1_0.777_overfit_warning

**Path:** `e:\swiss_citation_extraction\notebooks\06_law_only_bm25_rerank_judge_baseline\law_only_hybrid_v12_val_macroF1_0.777_overfit_warning.ipynb`

**Summary (one line):** Law-only (`laws_de.csv`, 171 654 articles) hybrid pipeline — BM25 top-100 → Qwen3-Reranker-8B → fused score `0.7·BM25 + 0.3·rerank` → zone-split (`HIGH = 0.55`, `LOW = 0.25`) → Qwen3-8B LLM judge on the borderline zone. **Macro F1 = 0.777 on `val.csv` (n = 10)** but **0.296 on `train.csv`** — the notebook's own cell-7 output flags this as `warning`: val-set overfit, only the `thresh_only` and `bm25_K3` variants generalise.

## Configuration

Colab Pro (A100 80GB), Python 3.12. 9 cells total (2 empty trailing cells).
Drive base: `/content/drive/MyDrive/Omnilex-Agentic-Retrieval-Competition`.

Notebook is a multi-script pipeline executed in sequence:

1. Cell 0–1: pip installs (`faiss-gpu-cu12`, `sacremoses`).
2. Cell 2: `option_a_translate_and_embed.py` — full DE→EN article translation + Qwen3-Embedding-4B re-embedding of all 171,654 corpus articles into a bilingual FAISS index.
3. Cell 3: `test_better_translation.py` — mini-index ablation comparing German-only / MarianMT / NLLB-3.3B translations on 5 worst val queries.
4. Cell 4: pip install `rank_bm25 deep_translator tqdm pandas pyarrow`.
5. **Cell 5 (the F1=0.777 producer): `bm25_hybrid_v12_colab.py`** — BM25 top-100 → Qwen3-Reranker-8B → fused score zone split → Qwen3-8B LLM judge on borderline only.
6. Cell 7: `submission_v3.py` — train-tuned BM25 top-30 → Qwen3-Reranker-8B → pick best K, generates test submissions for K=5,10,15,20,25,30.

### Models

| Role | Model |
|------|-------|
| MT (DE→EN, full corpus) | `Helsinki-NLP/opus-mt-de-en` (MarianMT), `num_beams=2`, max_len 512, batch 512 |
| MT ablation only | `facebook/nllb-200-3.3B` (fp16), `num_beams=5`, max_input 768, max_new 384, batch 8 |
| Embedder | `Qwen/Qwen3-Embedding-4B` (bfloat16), dim 2560, max_len 1024 (docs) / 512 (queries), batch 64/10, last-token pooling, L2-normalized |
| FAISS | `IndexFlatIP`, 171,654 × 2560, 1757.7 MB |
| BM25 | `rank_bm25.BM25Okapi`, prebuilt index `bm25_v2_index.pkl` (171,654 docs) |
| Reranker | `Qwen/Qwen3-Reranker-8B` (bfloat16), causal-LM yes/no scoring, batch 8, max_len 4096 |
| Judge (v12) | `Qwen/Qwen3-8B` (bfloat16), thinking enabled, temperature 0.5, top_k 20, top_p 0.95, max_new_tokens 3000, max input 20000 |

### Thresholds (v12 hybrid, the F1=0.777 setting)

- BM25 candidates per query: 100
- Fused score: `0.7 * normalised_BM25 + 0.3 * reranker_score`
- `HIGH_THRESH = 0.55` → auto-YES
- `LOW_THRESH  = 0.25` → auto-NO
- In between → borderline, sent to Qwen3-8B judge
- Default verdict for any borderline citation the judge fails to mention: YES (err on inclusion)

### Reranker prompt (v12)

System: `Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".`

Instruct: `Given a query about Swiss law, determine whether the provided law article is related to or applicable to the legal issue.`

Document format passed per candidate:
```
{citation_canon}
Law: {law_abbreviation} — {law_name_en}
Title: {title}
Heading: {context_heading_title}
Type: {provision_type}
Text: {text[:500]}
```
Scoring: extract logits for `"yes"`/`"no"` tokens at the final position, take `softmax([no, yes])[:,1]`.

### LLM judge prompt (v12, drives the +0.10 F1 lift over thresh-only)

System prompt embeds explicit Swiss Federal Court (Bundesgericht) domain knowledge — the 7 citation categories:
1. SUBSTANTIVE LAW (StGB, OR, ZGB…)
2. DEFINITIONS (e.g., Art. 8 ATSG defines invalidity)
3. PROCEDURAL RULES (StPO, ZPO)
4. APPEAL PROVISIONS (Beschwerde Art. 393ff StPO, Berufung, deadlines)
5. COST ALLOCATION (Art. 422, 428 StPO; Art. 64 BGG)
6. COURT JURISDICTION (Art. 37/39 StBOG, Art. 100 BGG)
7. CONSTITUTIONAL PRINCIPLES (Art. 29 BV, Art. 2 ZGB)

Explicit reminder examples: "A query about pre-trial detention will cite detention rules AND appeal rules AND cost rules AND court jurisdiction." Rule: when uncertain → YES. Forced output format: `CITATION | VERDICT: YES or NO`. The system reads German text directly.

User prompt: `LEGAL QUERY: {query_en[:600]}` + `ANFRAGE (Deutsch): {query_de[:400]}` + numbered list of borderline candidates with `Law / German text[:400]`. Generation uses chat template with `enable_thinking=True`; `</think>` split is stripped before parsing verdicts. Citations matched by exact / lowercase / regex fallback against borderline set.

### Embedding instruction (cell 2 / option A)

```
Instruct: Given a legal fact pattern, retrieve the most relevant
Swiss statutory provisions (law article citations).
Query: {query}
```

### Bilingual embed_text v2 structure (cell 2)

```
[DOC_TYPE] Swiss law article
[CITATION] {cit}
[LAW_ABBREV] {abbrev}
[GROUP] / [SUBGROUP]
[LAW_NAME_EN] ...
[PROVISION_TYPE] ...
[TEXT_EN]\n{translated EN}
[LAW_NAME_DE] {de[:150]}
[ARTICLE_TOPIC_DE] ...
[TEXT_DE]\n{original DE}
```
English block first (anchors EN-query embedding space), German block second.

### Token-law-frequency BM25 expansion (cells 5 and 7)

Per-token law-abbreviation prior built from train gold: `tlf[token][law_abbr] += 1`. At query time, for each query token, score laws by `(count / total) * BM25_idf`. Top-5 laws are appended to the query token list 5× each as boosters before calling `bm25.get_scores`.

### Gold expansion logic

A bare `Art. N LAW` (no `Abs.`) gold citation that doesn't match corpus directly is expanded to all `Art. N Abs. * LAW` child paragraphs found in `laws_de.csv` and present in corpus.

## Data

- `retrieval/corpus.parquet` — 171,654 Swiss law articles, columns include `citation_canon`, `law_abbrev`, `text` (German), `group`, `subgroup`, `law_full_name`, `title`, `article_topic`, `embed_text`.
- `retrieval/laws_knowledge_base.jsonl` — 171,654 entries; supplies `law.law_abbreviation`, `law.law_name_en` (1,125 unique), `semantic.provision_type` (171,654), `structure.context_heading_title`.
- `retrieval/article_translations_de_en.json` — 171,654 cached MarianMT DE→EN article texts.
- `retrieval/corpus_v2.parquet` — corpus with bilingual `embed_text`.
- `retrieval/corpus_embeddings_v2.npy` — (171654, 2560) float32 memmap.
- `retrieval/faiss_index_v2.bin` — FlatIP, 1757.7 MB.
- `retrieval/bm25_v2_index.pkl` + `bm25_v2_ids.pkl` — prebuilt BM25 over the same 171,654 corpus.
- `retrieval/knowledge_base_optimized_hybrid_retrieval/query_translations_trainval.json` — EN→DE cached query translations.
- `data/train.csv` (1,139), `data/val.csv` (10), `data/test.csv` (40), `data/laws_de.csv` (179,641 rows, used for parent→Abs. expansion).
- Caches written: `pipeline_cache/reranker_Qwen_Qwen3-Reranker-8B_v12.json` (20 queries), `pipeline_cache/llm_judge_v12_borderline.json`, `pipeline_output_v12/summary_*.json`.

## Pipeline

### Cell 2 — Option A (build the bilingual FAISS index used by other notebooks; not used by v12 in this run)

Step 1: MarianMT DE→EN translates all 171,654 article texts (cached, all hits in this run).
Step 2: Build `corpus_v2.parquet` with bilingual `embed_text`.
Step 3: Embed with Qwen3-Embedding-4B (bf16, batch 64, max_len 1024, last-token pool, L2-norm). Resumed from doc 125,440. Peak VRAM 32.95 GB.
Step 4: FAISS `IndexFlatIP` over (171654, 2560).
Step 5: Embed 10 val queries with `Instruct: ...\nQuery: ...` prefix, max_len 512, batch 10.

### Cell 3 — translator ablation on 5 worst val queries (mini-index)

`WORST_QIDS = ["val_008","val_001","val_003","val_010","val_004"]`. Mini-index = 64 gold + 1,000 random distractors = 1,064 docs. Compares German-only / MarianMT bilingual / NLLB-3.3B bilingual using same Qwen3-Embedding-4B + IndexFlatIP search.

### Cell 5 — `bm25_hybrid_v12_colab.py` (the F1=0.777 path)

Eval set: val (10 queries) + 10 train queries (`train.sample(n=10, random_state=42)`), 20 total.

Stage 0: Load corpus, KB, BM25 index, laws_de, query-translations cache, build `tlf`/`ttc` from full train (1,139).
Stage 1: BM25 retrieval, top-100 per query, with tlf-enhanced tokens.
Stage 2: Qwen3-Reranker-8B over 20 × 100 = 2,000 (query, doc) pairs, score = `softmax([no, yes])[:,1]`. Tokens: yes=9693, no=2152.
Stage 3: Fused score = `0.7 * normalized_BM25 + 0.3 * reranker`. Split into auto-YES (≥0.55), borderline (0.25–0.55), auto-NO (<0.25). Across 20 queries: 104 auto-YES, 180 borderline (~9/query), 1,716 auto-NO.
Stage 4: Qwen3-8B judges only the 180 borderline candidates with the 7-category Swiss-Bundesgericht system prompt. Unparsed borderline → default YES.
Stage 5: Final predictions = `auto_YES ∪ borderline_YES`. Fallback (no predictions): top-1 BM25.

### Cell 7 — `submission_v3.py` (test submissions)

Author's framing: "Val is leaked. Train is honest. Tune on train only." 100 train queries sampled, top-30 BM25 → Qwen3-Reranker-8B with a more permissive reranker instruction (mentions substantive/procedural/definitions/penalties/exceptions explicitly). Sorts candidates by fused score, picks top-K, sweeps K ∈ {5,10,15,20,25,30}. Generates `submission_K{K}.csv` for all six K values for the 40-query test set.

## Results

### Cell 2 — Bilingual Qwen3-Embedding-4B recall on full 171,654 corpus (val, 10 queries)

| QID | R@30 | R@100 | R@300 | R@500 | R@1000 | R@1500 | gold |
|---|---:|---:|---:|---:|---:|---:|---:|
| val_001 | 0.211 | 0.263 | 0.263 | 0.263 | 0.263 | 0.263 | 19 |
| val_002 | 0.105 | 0.263 | 0.421 | 0.474 | 0.579 | 0.579 | 19 |
| val_003 | 0.083 | 0.083 | 0.167 | 0.250 | 0.292 | 0.333 | 24 |
| val_004 | 0.222 | 0.556 | 0.556 | 0.556 | 0.556 | 0.556 | 9 |
| val_005 | 0.333 | 0.500 | 0.667 | 0.667 | 0.833 | 0.833 | 6 |
| val_006 | 0.273 | 0.364 | 0.545 | 0.545 | 0.636 | 0.636 | 11 |
| val_007 | 0.143 | 0.214 | 0.357 | 0.571 | 0.571 | 0.643 | 14 |
| val_008 | 0.050 | 0.050 | 0.050 | 0.050 | 0.100 | 0.100 | 20 |
| val_009 | 0.182 | 0.545 | 0.636 | 0.636 | 0.818 | 0.818 | 11 |
| val_010 | 0.000 | 0.071 | 0.143 | 0.214 | 0.357 | 0.357 | 14 |
| **MACRO AVG** | **0.160** | **0.291** | **0.380** | **0.423** | **0.501** | **0.512** | |

Author's reference comparison (printed verbatim in the cell):
```
OLD (German-only embed_text, with Instruct: prefix):
  R@30=0.325  R@100=0.402  R@300=0.458  R@1500=0.564
OPTION C (+ English metadata, same German text):
  R@30=0.471  R@100=0.605  R@300=0.732  R@1500=0.888  (mini-index)
  Full-index result: WORSE than old (avg delta = -0.026)
OPTION A (this run — bilingual EN+DE): see table above.
BM25 baseline (reference):
  R@30=0.845  R@1500=0.981
```
Bilingual full-corpus dense retrieval underperforms BM25 by a wide margin → v12 abandons dense, uses BM25 top-100 only.

### Cell 3 — Translator ablation mini-index (5 worst val queries, 1,064-doc mini-index)

A. German-only baseline:

| QID | R@10 | R@30 | R@100 | R@300 | R@500 |
|---|---:|---:|---:|---:|---:|
| val_001 | 0.263 | 0.579 | 0.947 | 0.947 | 1.000 |
| val_003 | 0.167 | 0.458 | 0.792 | 0.875 | 0.958 |
| val_004 | 0.556 | 0.667 | 0.889 | 0.889 | 0.889 |
| val_008 | 0.100 | 0.200 | 0.650 | 0.850 | 0.950 |
| val_010 | 0.286 | 0.500 | 0.929 | 1.000 | 1.000 |
| **MACRO** | **0.274** | **0.481** | **0.841** | **0.912** | **0.959** |

B. MarianMT bilingual: MACRO 0.297 / 0.429 / 0.686 / 0.888 / 0.951.
C. NLLB-3.3B bilingual: MACRO 0.306 / 0.429 / 0.655 / 0.870 / 0.931.

Delta vs German-only (averaged across K): MarianMT −0.043, NLLB-3.3B −0.056. Verdict: better MT does NOT fix dense recall.

### Cell 5 — VAL hybrid v12 (the headline F1=0.777 result)

| QID | P | R | F1 | Pred | Gold |
|---|---:|---:|---:|---:|---:|
| val_001 | 1.000 | 0.895 | 0.944 | 17 | 19 |
| val_002 | 1.000 | 0.632 | 0.774 | 12 | 19 |
| val_003 | 0.957 | 0.917 | 0.936 | 23 | 24 |
| val_004 | 1.000 | 0.667 | 0.800 | 6 | 9 |
| val_005 | 0.500 | 0.833 | 0.625 | 10 | 6 |
| val_006 | 0.857 | 0.545 | 0.667 | 7 | 11 |
| val_007 | 0.600 | 0.643 | 0.621 | 15 | 14 |
| val_008 | 0.923 | 0.600 | 0.727 | 13 | 20 |
| val_009 | 0.750 | 0.818 | 0.783 | 12 | 11 |
| val_010 | 0.867 | 0.929 | 0.897 | 15 | 14 |
| **MACRO** | **0.845** | **0.748** | **0.777** | **avg_k=13.0** | |

BM25 Recall@100 (val): val_001 0.947, val_002 0.789, val_003 0.958, val_004 1.000, val_005 1.000, val_006 0.818, val_007 0.857, val_008 0.750, val_009 1.000, val_010 0.929. Author note: BM25 R@100 ≈ 0.90 — "Don't let the LLM destroy it."

Zone distribution per val query: val_001 (yes=11, bl=6, no=83), val_002 (10/2/88), val_003 (6/17/77), val_004 (4/2/94), val_005 (4/6/90), val_006 (5/2/93), val_007 (6/9/85), val_008 (9/4/87), val_009 (8/4/88), val_010 (11/4/85).

### Cell 5 — VAL threshold-only (no judge) and BM25-K baselines

| Method | P | R | F1 | AvgK |
|---|---:|---:|---:|---:|
| VAL_hybrid_v12 | 0.845 | 0.748 | **0.777** | 13.0 |
| VAL_bm25_K12   | 0.800 | 0.693 | 0.714 | 12.0 |
| VAL_bm25_K15   | 0.713 | 0.753 | 0.705 | 15.0 |
| VAL_bm25_K8    | 0.938 | 0.577 | 0.687 | 8.0 |
| VAL_thresh_only | 1.000 | 0.531 | 0.681 | 7.4 |
| VAL_bm25_K20   | 0.585 | 0.808 | 0.654 | 20.0 |
| VAL_bm25_K5    | 1.000 | 0.400 | 0.550 | 5.0 |
| VAL_bm25_K3    | 1.000 | 0.240 | 0.376 | 3.0 |

The Qwen3-8B borderline judge contributes +0.096 F1 over threshold-only (0.777 vs 0.681).

### Cell 5 — TRAIN hybrid v12 (10 sampled train queries, no leakage)

| QID | P | R | F1 | Pred | Gold |
|---|---:|---:|---:|---:|---:|
| train_0789 | 0.158 | 0.545 | 0.245 | 38 | 11 |
| train_0905 | 0.095 | 1.000 | 0.174 | 21 | 2 |
| train_0290 | 0.000 | 0.000 | 0.000 | 24 | 2 |
| train_1041 | 0.167 | 1.000 | 0.286 | 6 | 1 |
| train_0333 | 0.065 | 0.400 | 0.111 | 31 | 5 |
| train_0110 | 0.333 | 1.000 | 0.500 | 3 | 1 |
| train_0527 | 0.111 | 1.000 | 0.200 | 9 | 1 |
| train_0057 | 0.250 | 0.800 | 0.381 | 16 | 5 |
| train_0753 | 0.500 | 1.000 | 0.667 | 2 | 1 |
| train_0241 | 0.250 | 1.000 | 0.400 | 4 | 1 |
| **MACRO** | **0.193** | **0.775** | **0.296** | **avg_k=15.4** | |

Consistency table (in-cell):

| Method | VAL | TRAIN | Delta | AvgF1 | Flag |
|---|---:|---:|---:|---:|---|
| hybrid_v12 | 0.777 | 0.296 | +0.481 | 0.537 | warning |
| thresh_only | 0.681 | 0.570 | +0.111 | 0.625 | consistent |
| bm25_K3 | 0.376 | 0.334 | +0.042 | 0.355 | consistent |

Author's printed insight in cell 7: the v12 hybrid F1=0.777 is val-overfit; train F1 collapses to 0.296. Only `thresh_only` and `bm25_K3` generalize. Total runtime cell 5: 1,849 s.

### Cell 7 — Train-tuned submission (100 train queries, BM25 top-30 → Qwen3-Reranker-8B → top-K)

| K | P | R | F1 | AvgK |
|---:|---:|---:|---:|---:|
| 5 | 0.3571 | 0.4956 | **0.3364** | 5.0 |
| 10 | 0.2235 | 0.5831 | 0.2695 | 10.0 |
| 15 | 0.1599 | 0.6050 | 0.2175 | 15.0 |
| 20 | 0.1245 | 0.6164 | 0.1822 | 20.0 |
| 25 | 0.1004 | 0.6186 | 0.1544 | 25.0 |
| 30 | 0.0840 | 0.6197 | 0.1339 | 30.0 |

Best K = 5, train F1 = 0.3364. Six test submissions written (`submission_K{5,10,15,20,25,30}.csv`, 40 queries each). Top-5 test_001 predictions: `Art. 100 Abs. 1 BGG; Art. 467 ZGB; Art. 16 ZGB; Art. 505 Abs. 1 ZGB; Art. 393 Abs. 1 STPO`. Runtime cell 7: 2,341 s.

## Summary

The notebook's F1=0.777 (cell 5, `bm25_hybrid_v12_colab.py`) is produced by a four-stage pipeline that deliberately abandons dense retrieval after cell 2 demonstrated that bilingual Qwen3-Embedding-4B (R@100 = 0.291 on val) is dominated by BM25 (R@100 ≈ 0.90 on val). BM25 over the 171,654-article corpus with token-law-frequency expansion (top-5 law-abbreviations appended 5× each, derived from train gold) supplies the top-100 candidate pool. Qwen3-Reranker-8B then assigns yes/no probabilities; a fused score `0.7·BM25_norm + 0.3·reranker` triages each candidate into auto-YES (≥0.55), auto-NO (<0.25), or borderline (≈9 per query). Qwen3-8B with a Swiss-Bundesgericht system prompt enumerating seven citation categories (substantive / definitions / procedure / appeal / costs / jurisdiction / constitutional) judges only the borderline ~20% — final = auto_YES ∪ borderline_YES, with default-YES for any unparsed borderline. The unique features that drive the score are the explicit Swiss-domain category taxonomy in the judge prompt (+0.096 F1 over threshold-only), the conservative fused thresholds that protect BM25 recall, the tlf-IDF law-abbreviation booster, and parent-to-`Abs.` gold expansion. Cell 7 explicitly flags v12 as val-overfit (train F1 drops to 0.296, delta +0.481) and falls back to a simpler train-tuned BM25 top-30 + reranker + top-K=5 for the actual test submissions.
