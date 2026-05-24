# Swiss Citation Retrieval — Endgame Handoff (2026-05-09)

> Read this top-to-bottom before responding to new instructions. It contains
> the full context of what's been built, what was actually measured on val,
> what works vs. what's broken, and the agreed plan for the next phase.
> Do not redo any item under §10. Do not propose anything blocked by §3.

---

## 0. TL;DR

We built and ran an end-to-end Colab pipeline (`notebooks/swiss_citation_endgame_colab.ipynb`)
combining BM25 + Qwen3-Embedding-8B vectors + HyDE + Qwen3-Reranker-8B +
Qwen3-8B LLM judge + 5 auxiliary channels. Result on val: **best Macro F1 < 0.10**,
target 0.6-0.8.

**Root cause is structural, not a bug:** retrieval pool recall is locked at
**23.1% (58/251 gold)** before the reranker even sees candidates. This matches
the user's prior empirical ceiling (`research/personal_observations.md` Obs 3:
"Qwen3-8B + full corpus + max context: R@1000 = 0.289").

**Decision:** drop trying to break the ceiling with bigger models or better
prompts. Go back to the user's original architectural ask — enrichment-first
prefilter to narrow 2.65M → 500k before BM25/vector, plus structured query
expansion to add diversity. The current notebook BUILT the prefilter as a
function but never wired it into `retrieve()`. That's the first fix.

---

## 1. Project goal (from `research/problem_statement.md`)

| Property | Value |
|---|---|
| Task | Cross-lingual legal citation retrieval (English query → Swiss legal citations) |
| Output | Set of citation strings, exact match against closed vocab |
| Corpus | 175,933 law (de) + 2,476,315 court (de/fr/it) = ~2,652,248 |
| Splits | train (1139, 99% German), val (10, English), test (40, English) |
| Metric | Macro F1 |
| Target | **0.6-0.8** |
| Best public reference | F1=0.777 on val from `research/Untitled75.ipynb` (BM25 + Qwen3-Reranker-8B + Qwen3-8B judge zone routing — but pre-tuned on this val set with smaller corpus) |
| Constraint | Offline Kaggle, ≤12h, ≤$10/query |

---

## 2. Hard empirical constraints (the user's prior research, all canonical)

These are measured, not theoretical. **Any plan that ignores them will fail.**

### 2.1 Train gold has a hard 28.5% recall ceiling
(`personal_observations.md` Obs 1)
- 71.5% of train gold is in corpus as a row (Class A — retrievable)
- 20.9% appears only inside other rows' text (Class B — text-ref only)
- 7.6% absent entirely (Class C — not in corpus)
- **Val gold is 100% Class A.** Calibration must use val, never train.

### 2.2 Citation graph is NOT a co-prediction signal
(`personal_observations.md` Obs 2)
- 99.41% of gold-citation pairs have no graph edge
- Graph captures textual cross-references; gold captures expert co-selection
- **Don't use graph as a co-prediction expander.** OK as a tie-breaker.

### 2.3 Dense embedding alone caps at R@1000 = 0.289 on val
(`personal_observations.md` Obs 3)
- Qwen3-Embedding-8B (#1 MMTEB Multilingual)
- Full 2.16M-row corpus, max enrichment in passage text
- Macro F1 (oracle k) = **0.041**
- This is a structural ceiling, not a model-quality issue
- "The relationship being modeled is the wrong kind of relationship for cosine similarity"
- **Implication:** dense cannot be the primary recall stage. Architecture must add explicit legal knowledge.

### 2.4 Train/val distribution shift on 3 axes
(`personal_observations.md` Obs 4)
- Language: train 99.4% DE → val 100% EN
- Citation count: train median 2 → val median 22 (range 10-47)
- Court share: train 1.2% → val 40.6%
- **Court searching is mandatory** for val/test (~40% of gold)
- **Variable K** is mandatory; fixed K fails

---

## 3. Pipeline state — what's built (with file paths)

### 3.1 Local Python modules (DO NOT recreate)
- `scripts/granularity_resolver.py` — bare-article → paragraph-children expansion. Validated: train coverage 71.54% → 98.06%, val unchanged at 100%.
- `scripts/rerank_qwen3.py` — Qwen3-Reranker-8B class with HF model card prompt + `log_softmax([no, yes])[..., 1].exp()` scoring.
- `scripts/llm_judge_qwen3.py` — Qwen3-8B 7-category judge + zone routing. **The version in this file has the same bugs as the notebook.** Will need fixes per §6.4.
- `scripts/hybrid_retrieve.py` — multi-channel retriever with reranker + judge hooks. **Vector channel is brute-force CPU** — replaced by GPU code in the notebook.
- `scripts/eval_retrieval.py` — F1-tuned K-scan eval, with `--enable-rerank` / `--enable-judge` / `--apply-granularity-post-filter`.
- `scripts/extract_citation_graph.py` — patched (ATF/DTF/c./consid./space-docket). Re-extracted DB at `data_insights/citation_graph_extracted.sqlite` (1.99 GB). +398k edges (+9.6%) and +16k nodes (+0.7%) vs old graph. Gold-as-node coverage: 92.9% → 96.0%.

### 3.2 Reference notebooks (read for ideas, don't import)
- `research/Untitled75.ipynb` (532 KB) + `research/nb_dump.txt` (119 KB plaintext) — **the F1=0.777 reference**. Contains the canonical reranker prompt, the 7-category judge prompt (lines 1585-1608 of nb_dump), the `enhance()` token-level lexicon expansion, train-tuned K-scan.
- `research/retrieve_unified_corpus_v3.ipynb` — user's prior 8-channel retrieval notebook.
- `research/colab_dense_embedding_test.ipynb` — the canonical Obs 3 measurement.
- `research/colab_recall_preserving_candidate_funnel.ipynb`, `colab_segment_lattice_funnel_v3.ipynb` — superseded predecessors.
- `research/personal_observations.md`, `problem_statement.md` — canonical truth.

### 3.3 Endgame notebook (the one that produced the bad F1)
- `notebooks/swiss_citation_endgame_colab.ipynb` (27 cells, 17 toggles, 1999 lines)
- 17 CONFIG toggles, all default ON except `use_agentic_retrieval`
- Run completed end-to-end on Colab (Blackwell, 95.6 GB VRAM)

### 3.4 Cache from the broken run (preserved for forensics)
- `cache_endgame/hyde_cache.json` — 10 HyDE answers
- `cache_endgame/german_expansion_cache.json` — 10 German expansions
- `cache_endgame/query_embeddings.npy` (327 KB) + keys — 10 val query embeddings
- `cache_endgame/rerank/<sha1(query)>.json` × 10 — 399-529 rerank scores per query
- `cache_endgame/judge/<sha1(query)>/<sha1(citation)>.json` — 1788 judge verdicts total
- **Reranker + HyDE caches are reusable** for the next run. **Judge cache must be deleted** before re-running because verdicts are toxic.

### 3.5 Data assets
- `data/{train,val,test,laws_de,court_considerations}.csv`
- `data/train_granularity_expanded.csv`, `data/val_granularity_expanded.csv` — gold sets after granularity resolver. Coverage: train 98.06%, val 100%.
- `artifacts/unified_retrieval.sqlite` (~24 GB) — `documents` (2.65M rows) + FTS5 + `statute_links` (5.30M) + `case_links` (1.81M) + `adjacent_law_links` (348k)
- `artifacts/law_authority_cards_v2_unified.jsonl` — law cards
- `artifacts/court_authority_cards_v5_unified.jsonl` — court cards (10.4 GB)
- `law_json_llm_output/law_llm_descriptors_0000000_all.jsonl` — 173,033 law enrichments
- `outputs_from_363k_run/court_llm_descriptors_0000000_all.jsonl` — 363,258 court enrichments
- `embeddings/qwen3_8b_unified_chunk*.npy` (27 fp16 chunks, ~21 GB) + `qwen3_8b_unified_manifest.parquet` (2,652,248 rows)

### 3.6 Forensic scripts (kept under project root; can be deleted later)
- `diag_cache.py` — dumps cache distributions
- `diag_recall.py` — per-query stage-by-stage recall
- `diag_judge.py` — judge verdict raw-response analysis

---

## 4. Empirical results from the broken endgame run (val.csv, 10 queries)

### 4.1 Stage-by-stage recall (run `python diag_recall.py` to reproduce)

| qid | gold | gold_in_corpus | rerank_pool | gold_in_rerank | judged_yes | judged_no | auto_yes | auto_no |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val_001 | 42 | 42 | 512 | 11 | 212 | 28 | 0 | 0 |
| val_002 | 36 | 36 | 427 | 5 | 178 | 32 | 0 | 0 |
| val_003 | 47 | 47 | 529 | 4 | 257 | 18 | 0 | 0 |
| val_004 | 10 | 10 | 426 | 6 | 100 | 4 | 0 | 0 |
| val_005 | 11 | 11 | 481 | 7 | 201 | 6 | 0 | 0 |
| val_006 | 18 | 18 | 399 | 4 | 52 | 29 | 0 | 0 |
| val_007 | 19 | 19 | 419 | 8 | 115 | 1 | 0 | 0 |
| val_008 | 29 | 29 | 524 | 4 | 117 | 60 | 0 | 0 |
| val_009 | 14 | 14 | 509 | 4 | 188 | 11 | 0 | 0 |
| val_010 | 25 | 25 | 480 | 5 | 126 | 39 | 0 | 0 |
| **sum** | **251** | **251 (100%)** | **avg 471** | **58 (23.1%)** | 1546 | 228 | 0 | 0 |

### 4.2 Direct read of these numbers

- **100% of gold is in the corpus.** Granularity is fine (val gold is paragraph-granular). The system isn't predicting non-existent strings.
- **23.1% of gold is in the retrieval pool.** This is the binding ceiling. Even a perfect downstream maxes F1 at ~0.23. The reference pool size (~471 candidates) is generous; the problem is which candidates.
- **Auto-yes/auto-no zones never fired** (`auto_yes=0`, `auto_no=0` for every query). Thresholds 0.55/0.25 were calibrated for Untitled75's `0.7*minmax(retrieval) + 0.3*rerank` scale. The endgame notebook uses raw RRF scores (typically 0.01-0.1 range), so no candidate ever crossed a threshold. **Every candidate went through the judge.**
- **Judge said YES to 87%** (1546 / 1774). With ~22 gold per query, this is ~70 false positives and ~6-11 true positives per query → P ≈ 0.10, R ≈ 0.30, F1 ≈ 0.04.

### 4.3 Why the judge is so YES-heavy (4 compounding bugs in Cell 16)

1. **System prompt bias:** "When uncertain, say YES — it is better to include a marginally relevant article than to miss one." Recall-oriented; competition is F1.
2. **Parser default-YES on parse failure:** `if verdict is None: verdict = "yes"`.
3. **Qwen3-8B emits `<think>` chain-of-thought by default**, consuming most of the token budget before reaching `VERDICT:`.
4. **`max_new_tokens=200`** truncates 78/115 of query 4's YES responses mid-thinking; the parser falls through to default-YES.

Evidence: `python diag_judge.py` shows for query 487b (115 YES / 1 NO), only 37/115 YES had a literal `VERDICT: YES` in the response; 78 had `<think>` but no VERDICT line.

### 4.4 Reranker has a family bias
Looking at the top-5 of the first val query:
```
court:1cc193b985ff4efd484999bd  0.9844
court:5b898168fbd0e5b2e7936387  0.9805
court:1577eaa5b83fb2cf24abd32b  0.9766
court:b06c4b8d3168f288ae8ebca3  0.9727
court:9c4c43f509211043a8657c3c  0.9727
```
Bottom-5 are all `law:*` with score 0.0000. Either:
- The law candidates fed to the reranker have empty/wrong text fields, or
- The Qwen3-Reranker prompt format isn't matching law text well

This is **secondary** to the recall bottleneck. Don't address until the pool recall is fixed.

---

## 5. Per-technique status (the "what works what doesn't" table)

| Technique | Built | Wired into retrieve() | Measurable contribution | Notes |
|---|---|---|---|---|
| BM25 (FTS5, lexicon expansion) | ✅ | ✅ | Some — drives most of the pool | Vocab gap (val EN vs corpus DE = 4.3% noun overlap, Obs 3) caps it |
| Vector (Qwen3-Embedding-8B) | ✅ | ✅ | Capped at R@1000 = 0.29 (Obs 3) | Brute-force GPU works (~50 ms/query); not the bottleneck |
| HyDE (Qwen3-8B → embed) | ✅ | ✅ | **Unmeasured** | 10 hyde answers cached. No per-channel R@K logged. |
| Statute anchors (LIKE) | ✅ | ✅ | Helps when query names a code | Val queries rarely name codes |
| Case anchors (BGE/docket) | ✅ | ✅ | Same | |
| Citation graph 1-hop | ✅ | ✅ | Near-zero per Obs 2 | Don't oversell |
| Court_base sibling | ✅ | ✅ | Marginal | |
| German keyword expansion | ✅ | ✅ | **Unmeasured** | 10 expansions cached |
| **Enrichment prefilter** | ✅ (Cell 14) | **❌ NOT WIRED** | **N/A — never called by `retrieve()`** | The user's specific architectural ask. Index built, function defined, never invoked. **This is the #1 fix.** |
| Granularity post-filter | ✅ | ✅ (eval-side) | Validated 71.54% → 98.06% on train, val unchanged 100% | Can only help if gold is already in pool |
| Reranker (Qwen3-Reranker-8B) | ✅ | ✅ | Top-5 all-court / bottom-5 all-law=0.0 | Suspected text-feeding bug for law candidates |
| LLM judge (Qwen3-8B) | ✅ | ✅ | **Broken** — 87% rubber-stamp YES | 4 bugs in Cell 16 |
| Auto-yes / auto-no zone routing | ✅ | ❌ (thresholds wrong) | All candidates judged | Score scale mismatch |
| Multi-step agentic retrieval | ✅ | toggle off | Untested | Not the bottleneck |
| F1 K-sweep | ✅ | ✅ | Working | Best K=7 in last smoke run |

---

## 6. Phantom issues — already diagnosed and resolved (DO NOT redo)

These looked like problems but turned out not to be. Don't waste a turn re-investigating them.

### 6.1 "Court LLM enrichment is static_only"
**Wrong.** The 363,257 court LLM rows ARE merged into `unified_retrieval.sqlite`. The `static_only` quality flag fires because the court schema deliberately forbids `english_summary` (`scripts/court_enrichment_normalizer.py:324-330`) but `build_unified_retrieval_corpus.py:269-270` checks for it. The flag is a misnomer.

The real gap is that only 363k of 2.47M court rows ever got LLM enrichment (85% never ran). That's a Colab job, deferred to §11.

### 6.2 "UTF-8 mojibake in law enrichment / train.csv"
**Wrong.** Both files are byte-clean UTF-8. The audit's `Aufsichtsbeh�rde` was a viewing-side artifact (terminal reading UTF-8 through cp1252).

### 6.3 "Gold citations missing from graph (regex too narrow)"
**Patched.** ATF/DTF/c./consid./space-docket added. Re-extracted graph at `data_insights/citation_graph_extracted.sqlite` (1.99 GB). Coverage 92.9% → 96.0%. Remaining 4% missing (LugÜ, FIDLEG, FINIG, FinfraG, GBV, parts of URG/ZGB) are corpus gaps, not regex gaps.

### 6.4 "LLM judge is broken"
**Confirmed and diagnosed.** §4.3 above. **Not fixed yet.** Per §7 plan, defer until §7 Move 1 fixes the recall pool.

### 6.5 "No real ANN index for vector"
**Replaced with full-GPU brute force in the notebook.** `E_GPU = (2.65M, 4096) fp16` lives in VRAM, `vector_search` is `E_GPU @ q + topk`. ~50 ms/query. Not a bottleneck anymore. Don't waste time on FAISS-IVF.

### 6.6 "Stale 36 GB fp32 embedding chunks"
Deleted. The 21 GB fp16 unified chunks at `embeddings/qwen3_8b_unified_chunk*.npy` are the canonical set.

---

## 7. Agreed plan — Moves 1, 2, 3 (in order)

### Move 1 — Wire the enrichment prefilter into BM25/vector

**Cost:** ~1 hour. CPU-only measurement; no GPU needed for the pass/fail check.

**Change:** in `notebooks/swiss_citation_endgame_colab.ipynb` Cell 13's `retrieve()`:
1. Call `enrichment_prefilter(query, target_pool=500_000)` first.
2. Constrain BM25 with `... WHERE doc_id IN (?, ?, ...)` against the pool.
3. Constrain vector search to the pool's row indices on GPU (`E_GPU[pool_idx] @ q`).
4. Anchor channels (statute / case) NOT constrained — they're already small.
5. Graph / court_base expansion NOT constrained — they expand from anchor seeds.

**Pass criterion:** average pool R@500k ≥ 0.95 on val.
- If yes: BM25 over a 5× smaller corpus is ~5× more discriminative; expect pool R@1000 to jump from 0.23 to 0.4-0.6.
- If no: prefilter strategy is wrong (not enough enrichment coverage). Drop it; revisit per §11.

**Measurement cell to add to notebook:**
```python
# pre-Move-1 baseline (no GPU): for each val query, measure
# (a) prefilter pool size, (b) gold recall in pool, (c) breakdown by family
import json
results = []
for _, r in val_df.iterrows():
    pool = enrichment_prefilter(str(r["query"]), target_pool=500_000)
    gold = [c.strip() for c in str(r["gold_citations"]).split(";") if c.strip()]
    hits = sum(1 for g in gold if cit_to_doc.get(g) in pool)
    results.append({"qid": r["query_id"], "pool": len(pool),
                    "gold": len(gold), "in_pool": hits,
                    "recall": hits/max(1, len(gold))})
import pandas as pd; print(pd.DataFrame(results))
```

### Move 2 — Structured query expansion via LLM (NOT HyDE — different)

**Only if Move 1 passes.** HyDE generates a fake answer; this generates a structured query.

**Change:** add a new cell that calls Qwen3-8B with this output schema:
```json
{
  "legal_domain": "criminal procedure",
  "applicable_codes": ["StPO", "BGG"],
  "german_terms": ["Untersuchungshaft", "Kollusionsgefahr", "Verlängerung"],
  "concept_seeds": ["pretrial detention", "collusion risk", "proportionality"]
}
```

Run 3-5 targeted searches per query (one per code, one per concept) and union the pools. Each sub-search uses the existing BM25+vector stack but with a strict per-search budget (e.g., 200) so the union doesn't blow up.

**Pass criterion:** per-channel union R@5000 ≥ 0.50 on val.

**Why this and not bigger LLM judge:** the bottleneck is recall, not precision. A larger LLM at the end stage cannot recover gold that was never retrieved. Structured query expansion attacks the diversity problem directly — every val query needs ~22 citations across ~5 legal areas; one query embedding can't cover that span.

### Move 3 — Reranker + judge cleanup (only after recall is unstuck)

**Only if Move 2 lifts pool R to ≥ 0.5.** Otherwise reranking a 25%-recall pool is rearranging deck chairs.

Three changes to apply together:
1. **Reranker text-feed audit:** for the law family, log what `text` field is being passed to the reranker on the bottom-5 (currently scoring 0.0). Likely the law card has the German legal text in a different field than `text`. Fix the feed.
2. **Judge prompt (replace Cell 16 verbatim):** disable thinking (`enable_thinking=False`), default-NO on parse failure, drop the "say YES when uncertain" instruction, raise `max_new_tokens=24`, restrict response to `VERDICT: YES|NO` exactly. (Patch already drafted in §6 of `notebooks/swiss_citation_endgame_colab.ipynb` discussion thread — see `docs/notebook_judge_v2_patch.md` if extracted, otherwise reconstruct from the diff in the chat.)
3. **Recalibrate zone thresholds against actual fused-score distribution:** print a histogram of `fused_score` on val candidates, set `auto_yes_thresh` to the 90th percentile, `auto_no_thresh` to the 30th percentile.

**Pass criterion:** Macro F1 on val ≥ 0.30 with judge OFF (reranker + threshold-cut alone). Adding the judge should lift to ≥ 0.40.

---

## 8. Numerical knobs that matter (current values + thinking)

| Knob | Current | Notes |
|---|---|---|
| `prefilter_pool_target` | 1_000_000 | Lower to 500k after Move 1 if recall holds |
| `channel_budgets.bm25` | 800 | Probably too low after prefilter; raise to 2000 |
| `channel_budgets.vector` | 800 | Same |
| `rrf_k` | 60 | Standard; don't change |
| `judge_auto_yes` | 0.55 | Calibrated for fused score scale 0-1; current RRF is 0-0.1, so threshold is unreachable |
| `judge_auto_no` | 0.25 | Same problem |
| `rerank_top_n` | 200 | OK |
| `rerank_alpha_retrieval` / `rerank_alpha_rerank` | 0.7 / 0.3 | From Untitled75; keep |
| `k_sweep_for_f1` | [5, 7, 10, 13, 15, 20, 25, 30, 50, 75, 100] | OK; val gold range is 10-47 so include 35 too |

---

## 9. File inventory (paths the new chat may need to reference)

```
research/
  problem_statement.md            ← canonical task spec
  personal_observations.md        ← Obs 1-4, the empirical truth
  Untitled75.ipynb                ← F1=0.777 reference notebook
  nb_dump.txt                     ← plaintext dump of Untitled75
  retrieve_unified_corpus_v3.ipynb ← user's prior 8-channel retrieval
  colab_dense_embedding_test.ipynb ← canonical Obs 3 measurement
  endgame_handoff_2026-05-09.md   ← THIS FILE

notebooks/
  swiss_citation_endgame_colab.ipynb  ← the broken run; will be patched per §7

scripts/
  granularity_resolver.py
  rerank_qwen3.py
  llm_judge_qwen3.py
  hybrid_retrieve.py              ← reranker/judge hooks wired
  eval_retrieval.py               ← F1 K-scan + --enable-rerank/--enable-judge
  extract_citation_graph.py       ← patched (ATF/DTF/c./consid./space-docket)
  build_unified_retrieval_corpus.py
  encode_queries_qwen3_8b.py
  prepare_embedding_input.py
  merge_court_llm_into_unified.py ← defensive remerge tool (already-merged)

docs/
  court_rag_plan.md               ← enrichment-first multi-channel plan
  laws_de_authority_card_static_report.md
  laws_de_llm_enrichment_schema.md
  court_enrichment_field_contract.md

data/
  train.csv, val.csv, test.csv, laws_de.csv, court_considerations.csv
  train_granularity_expanded.csv  ← 71.54% → 98.06% gold-in-corpus
  val_granularity_expanded.csv

artifacts/
  unified_retrieval.sqlite        ← 24 GB; documents + FTS5 + statute_links + case_links
  law_authority_cards_v2_unified.jsonl
  court_authority_cards_v5_unified.jsonl

embeddings/
  qwen3_8b_unified_chunk000..026.npy  (27 chunks, fp16, ~21 GB)
  qwen3_8b_unified_manifest.parquet   (2,652,248 rows)
  qwen3_8b_unified_summary.json

law_json_llm_output/
  law_llm_descriptors_0000000_all.jsonl  (173,033 records)

outputs_from_363k_run/
  court_llm_descriptors_0000000_all.jsonl  (363,258 records — only 15% of court corpus)

data_insights/
  citation_graph_extracted.sqlite (1.99 GB; +398k edges, +16k nodes vs old)
  gold_citation_coverage.{md,csv,json}
  gold_parent_link_check.{md,csv,json}

cache_endgame/                    ← preserved forensic evidence
  rerank/<sha1(q)>.json × 10
  judge/<sha1(q)>/<sha1(c)>.json × 1788  (DELETE before re-run)
  hyde_cache.json                  (10 entries, REUSABLE)
  german_expansion_cache.json      (10 entries, REUSABLE)
  query_embeddings.npy             (10 vectors, REUSABLE)

tests/
  test_extract_citation_graph_aliases.py  (passes)
  test_segment_lattice_v3.py              (passes)

diag_*.py                          ← forensic scripts (root level)
```

---

## 10. DO NOT redo (one-line each)

- Don't re-run UTF-8 repair scripts. Files are clean.
- Don't re-merge court LLM enrichment into `unified_retrieval.sqlite`. Already merged.
- Don't waste cycles on FAISS-IVF for the vector index. Full-GPU brute force is fine on Blackwell.
- Don't re-extract the citation graph. Just done with patches.
- Don't tune the judge before fixing recall. F1 is recall-bottlenecked.
- Don't propose a bigger reranker / judge model. Same reason.
- Don't try to fix the train-distribution; calibrate on val (Obs 4).
- Don't expect graph expansion to lift recall meaningfully (Obs 2: 0.59% co-prediction overlap).
- Don't redo the granularity resolver. Validated.
- Don't over-engineer the agentic retrieval before Move 1 + Move 2 land.

---

## 11. Deferred / parked items

- **Extending court LLM enrichment from 363k to all 2.47M rows** — biggest live coverage gap. Colab GPU job, ~30+ hours. Park until Move 1+2 land; if structured query expansion gets us to F1 0.4, this becomes the next ~0.1 lift.
- **Reranker law-family bias diagnosis** — needs a one-cell experiment that logs the exact text fed to the reranker for 5 law candidates. Defer to Move 3.
- **MMR / diversity selection** — could replace top-K with rank-diversified selection. Defer until base recall is unstuck.
- **Token-level lexicon expansion learned from train** — Untitled75's `enhance()`. Implemented in Cell 7. Effect not measured in isolation. Worth ablating only after Move 1.
- **Multi-step agentic retrieval** — toggle is off. Defer.
- **Submission to test set** — meaningless until val F1 is in a respectable range.

---

## 12. Open questions for the user (ask before starting Move 1)

1. **Drive vs local execution**: Move 1's measurement cell is CPU-only and runs in seconds. Would you prefer to run it locally first (no GPU needed) so we have the prefilter recall numbers before touching Colab?
2. **Prefilter coverage**: enrichment cards exist for 173k laws + 363k courts = 536k rows out of 2.65M. The prefilter index has cards only for those. Roughly half the corpus has no enrichment to filter on. We need to decide: do unenriched docs (a) bypass the prefilter (always pass through, defeats the purpose), (b) get filtered out (loses recall on unenriched gold), or (c) get a fallback inverted index built from raw German text?
3. **Structured query expansion model**: Qwen3-8B is on the GPU already; we can reuse it for query expansion. OK to proceed with that, or do you want to pull in a different LLM?

---

## 13. Quick start for the new chat

If the user says "let's do Move 1":
1. Read this file end-to-end first.
2. Open `notebooks/swiss_citation_endgame_colab.ipynb`, find Cell 13 (`def retrieve(...)`), and Cell 14/15 (the prefilter `enrichment_prefilter()` definition).
3. Decide on §12 question 2 (unenriched-doc fallback strategy) before patching `retrieve()`.
4. Patch `retrieve()` to call `enrichment_prefilter()` first and constrain BM25 + vector to the pool.
5. Add the measurement cell from §7 Move 1.
6. Tell the user to re-run those cells; the measurement is CPU-only so it's fast.
7. If pool R@500k ≥ 0.95, proceed to BM25/vector inside the pool and remeasure rerank-pool recall.

If the user says "let's debug judge / reranker":
1. Reread §4.3 and §4.4. Both are downstream of recall.
2. Push back: "Per the handoff §7, judge/reranker are blocked on Move 1 + 2. Do you want to override or proceed with Move 1 first?"

If the user proposes a bigger model:
1. Reread Obs 3. Bigger models don't break the embedding-similarity ceiling.
2. Suggest the prefilter + structured query expansion path instead.

---

*Document state: written 2026-05-09 after the broken endgame run. Update this
file as Moves 1-3 produce numbers; do not let it go stale.*
