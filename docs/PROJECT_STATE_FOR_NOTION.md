# Swiss Legal Citation Extraction — Project State

**Snapshot: 2026-05-15** · paste this whole doc into Notion as a private page.

---

## TL;DR

We are building a system that takes an English legal question and returns the set of Swiss legal citations (laws + court decisions) that answer it, scored by **macro F1** against gold. The corpus has ~2.65M citation targets across German, French and Italian; val has 10 English queries with 222 gold citations total.

| Where we are | Number |
|---|---|
| Macro F1 target (competition) | **0.6 – 0.8** |
| Macro F1 currently submitted | **0.498** |
| Macro F1 of the best historical reference (`Untitled75` notebook) | **0.777** |
| Theoretical maximum given the retrieval pool we have | **~0.93** |
| Strict floor that has **never** been reached (macro ≥ 0.8 AND min ≥ 0.8) | structurally blocked at val_003 |
| Notebooks produced over ~3 months | **90** |
| Pipeline iterations (v1 → v7.5) | 8 major versions + 13 sub-iterations |

The retrieval problem is solved. The final-set-sizing problem isn't. Our next +0.10 to +0.30 in F1 lives in **how we cut the top-K**, not in pulling more candidates.

---

## 1. What we have done so far

This section enumerates everything we built and tried. The "Helpful or not" verdict is the subject of §2 and §3.

### 1.1 Data we built (on top of competition inputs)

The competition shipped 6 raw files:

| File | Rows | Description |
|---|---|---|
| `train.csv` | 1,139 | Query + gold-citation list; 99% German queries |
| `val.csv` | 10 | Same schema; 100% English queries, 222 gold total |
| `test.csv` | 40 | Same schema; 20 public + 20 private leaderboard |
| `sample_submission.csv` | — | Format reference only |
| `laws_de.csv` | 175,933 | One row per German law snippet (Article / paragraph) |
| `court_considerations.csv` | 2,476,315 | One row per court Erwägung (consideration) in DE / FR / IT |

From these we derived every artifact below. Total derived data ≈ 200 GB across local disk + Google Drive.

#### Splits, granularity, and gold structure
- **`train_granularity_expanded.csv` / `val_granularity_expanded.csv`** — gold expanded to handle the granularity ambiguity (a query asking for "Art. 11 OR" is satisfied by either the article-level or any of its paragraph-level rows). After expansion, gold-in-corpus rose from **71.54% → 98.06%** on train.
- **`data_insights/gold_analysis_coverage_and_parents/gold_citation_coverage.{csv,json,md}`** — Class A (directly retrievable from corpus column), Class B (text-only), Class C (absent). Val is 100% Class A. Train is 71.5% A + 20.9% B + 7.6% C → train ceiling 28.5%.
- **`gold_parent_link_check.{csv,json,md}`** — encodes the rule: if a paragraph-level citation (`Art. 11 Abs. 2 OR`) is gold, the article-level parent (`Art. 11 OR`) is NOT also valid.

#### Authority cards (per-document feature blocks)
- **`court_authority_cards_v5_unified.jsonl`** (10.4 GB, 2,476,315 cards) — canonical court-side store. Derived from `court_considerations.csv` + 363k LLM enrichments + citation graph.
- **`law_authority_cards_v2_unified.jsonl`** (883 MB, 175,933 cards) — canonical law-side store. Derived from `laws_de.csv` + 173k LLM enrichments + segment-lattice features.

#### LLM-derived enrichments (Qwen3-8B-AWQ via vLLM)
- **`court_llm_descriptors_0000000_all.jsonl`** — 363,258 court rows enriched (14.7% of 2.47M corpus). 99.987% `ok` status. 22.85 rows/s on Blackwell.
- **`law_llm_descriptors_0000000_all.jsonl`** — 173,033 law rows enriched (98.4% of 175,933 corpus). 98.8% usable, average `terms_grounded` 0.902. ~11 rows/s on Blackwell.
- Schema enforced by `docs/court_enrichment_field_contract.md` and `docs/laws_de_llm_enrichment_schema.md`.

#### Dense embeddings
- **`embeddings/qwen3_8b_unified_chunk000..026.npy`** — 27 chunks of fp16 (~21 GB total), 2,652,248 rows × 4096-dim, encoded with Qwen3-Embedding-8B.
- **`embeddings/qwen3_8b_unified_manifest.parquet`** — `(chunk, row) → doc_id` mapping.
- Loads as one tensor `E_GPU ∈ ℝ^(2.65M × 4096) fp16` (~21 GB) into Blackwell VRAM. Brute-force `E_GPU @ q + topk` ≈ 50 ms/query — no need for FAISS.

#### SQLite indices
- **`unified_retrieval.sqlite`** (22.4 GB) — 2.65M-row document table + FTS5 BM25 (with per-language scoring) + `statute_links` (5.30M edges) + `case_links` (1.81M) + `adjacent_law_links` (348k). The canonical retrieval substrate.
- **`citation_graph_extracted.sqlite`** (5.68 GB) — patched citation graph; 96% of val gold appears as a node. Useful as a tie-breaker, not a co-predictor (see §3).

#### Retrieval pool snapshot — the canonical artifact
- **`corpus_snapshot.json.gz`** (110 MB on Drive) — for each of the 10 val queries: top-50,000 candidates from the v7.5 funnel, with per-channel hit sets across 15 channels, fusion ranks, statute / concept / term targets from LLM query expansion, hard-negative role signals, and gold-doc-ids.
- **Macro R@50k = 0.901** on val. Per-query R_max ranges 0.766 (val_003) → 1.000 (val_004, val_005).

#### Cached reranker scores
- `scores_qwen3.npz`, `scores_bge.npz`, `scores_jina.npz` — full per-query × per-pool-doc scores for Qwen3-Reranker-8B, BGE-reranker-v2-m3, jina-reranker-v2-base-multilingual. Enables any fusion / sweep without GPU.
- `dossier_features.npz` — 9-dim per-(qid, did) cross-feature cache (statute_int, lead_statute_int, concept_int, term_int, co-citation, doctrinal, chamber, not-hardneg, fusion_rank).

### 1.2 Pipeline iterations we built

The chronological progression of pipelines we wrote, with the headline metric we captured for each:

| Phase | Date | Pipeline / approach | Headline measurement |
|---|---|---|---|
| 1 | Apr 21-22 | `exp_A1-A6` benchmarking — BGE-M3, HyDE+Enum, cross-encoder rerank, Legal-Swiss-RoBERTa fine-tune | Best: stat_recall@500 = **0.376** (A2: Qwen3-32B HyDE+Enum RRF over BGE-M3). A6 fine-tune collapsed to **0.101** |
| 2 | Apr 26-30 | Early funnel — dense-only test, recall-preserving funnel, segment_lattice_v3 | Dense alone: F1 = **0.044**. segment_lattice_v3: court_recall = 0 (broken). Both abandoned. |
| 3 | Apr 28 – May 8 | Kaggle infra debug (12 notebooks) + `Untitled75` historical reference run | `Untitled75` F1 = **0.777** on val (val-tuned, smaller corpus) |
| 4 | Apr 12 – May 9 | `pipeline_end_to_end_*` (5 variants), `pipeline_iteration_latest_*` (6 variants), `endgame_colab` | Best pre-v7.5 verifiable: F1 = **0.7082** (`latest_with_all_output`, Stage 0-6 adaptive K). Endgame regression to F1 < 0.10. |
| 5 | May 1 – May 6 | LLM enrichment runs (court 363k + law 173k) | 363,258 court rows @ 99.987% ok. 173,033 law rows @ 98.8% usable. |
| 6 | May 9 – May 11 | Anchor funnel v1 → v4 → v5 → v6 → v7 → v7.4 | v1 R@1000 = 0.119 → v4 = 0.238 → v7.5 R@50k = 0.901 |
| 7 | May 12 – May 13 | v7.5 multi-query canonical pool (13 iterations) | Macro R@50k = **0.901** ✓ |
| 8 | May 13 – May 15 | Hybrid reranker shootout (Qwen3 / BGE / jina + 9 dossier signals; F0–F4 fusion sweep) | F0 R@10k = **0.816** > F3 = 0.791. Reranker dead above K=500. |
| 9 | May 13 – May 14 | `cascade_legalmalr_v1` + `precision_v1` (GRPO + Qwen3-Judge end-to-end) | **Never produced an executed F1.** `_base` crashed; others have no captured output. Open. |

### 1.3 New approaches we innovated (on top of papers)

These are not direct paper implementations — they are project-specific designs:

1. **15-channel multi-query RRF fusion (v7.5)** — extends the standard BM25 + dense two-stage by adding 13 more channels: statute_backprop, graph_forward, concept_de / concept_en, term_de / term_orig, court_statute, sibling_graph, doctrinal, hard_negative, paragraph_role, court_base, law_code, fusion_rank. Each gets a per-query weight derived from v7.4 per-channel recall measurement.
2. **Granularity-expanded gold + parent-link rule** — handles the article/paragraph dual-gold ambiguity. Lifted train coverage from 71.54% → 98.06%.
3. **9-dimensional dossier feature cache** — pre-computed per-(qid, did) feature dictionary that lets the LLM judge / Stage 2 prompt see structured cross-features instead of just raw text.
4. **Endgame-judge bug-fix protocol** — explicit defensive design for the LLM judge after the May-9 catastrophe: delete previous cache, default-NO on parse failure (not default-YES), force JSON output, no `<think>` budget. (Designed; not yet measured.)
5. **Authority card cascade** — v5_unified merges raw court text + LLM descriptors + citation graph edges + paragraph role tags into a single per-doc card consumed by all downstream channels.
6. **Drive-pull tooling** — OAuth + curated + folder + scan scripts under `scripts/drive_pull/` for selectively syncing the 200 GB Drive tree to local without re-downloading (dedup manifest by Drive file ID).

---

## 2. What turned out helpful and why — evidence-based

For every claim below, the "Evidence" column points to the file or notebook where the measurement was captured.

### 2.1 Qwen3-Embedding-8B as first-stage retriever — **WORKS**
- **What we did:** encoded all 2,652,248 corpus rows with Qwen3-Embedding-8B (4096-dim, fp16, English instructions prepended even for DE/FR/IT docs).
- **Why it works:** the paper's `Instruct: {…}\nQuery: {…}` template was followed verbatim; the multilingual training covers DE/FR/IT; the 4096-dim representation captures legal nuance.
- **Evidence:** every notebook from v5 onward loads `qwen3_8b_unified_chunk*.npy`. The v7.5 dense channel alone contributes ~30% of pool recall. Brute-force search is ~50 ms/query on Blackwell (no need for FAISS, confirmed empirically).
- **How helpful (1–5):** ⭐⭐⭐⭐⭐ — load-bearing.

### 2.2 15-channel multi-query RRF fusion (v7.5) — **THE CORE WIN**
- **What we did:** built a funnel that issues multiple sub-queries per val question (from LLM expansion: legal_domain, applicable_codes, german_terms, concept_seeds) and fuses 15 channels into a single ranked top-50k.
- **Why it works:** no single channel reaches 0.5 recall; the union is 0.901 macro because each channel covers different gold (statute_backprop catches direct citations, graph_forward catches procedural neighbors, BM25 catches exact-word matches, dense catches paraphrase-level matches, etc.).
- **Evidence:** `pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.ipynb` reports per-channel recall; top contributors are statute_backprop (0.453), graph_forward (0.324), bm25 (0.31x), dense (0.30x). Combined R@50k = 0.901.
- **How helpful:** ⭐⭐⭐⭐⭐ — solved the retrieval problem within the structural ceiling.

### 2.3 LLM enrichment of court + law rows (Qwen3-8B-AWQ) — **WORKS**
- **What we did:** ran Qwen3-8B-AWQ over 363,258 court rows + 173,033 law rows on Blackwell, extracting structured descriptors (statute_anchors, concept_seeds, paragraph_role, etc.).
- **Why it works:** AWQ + vLLM + FP16 KV + FlashInfer on Blackwell gives 22.85 rows/s (court) / 11 rows/s (law). 99.987% / 98.8% valid output rate. Schemas enforced in `docs/court_enrichment_field_contract.md`.
- **Evidence:** the resulting `*_llm_descriptors_0000000_all.jsonl` files exist on disk; the unified authority cards merge them at build time; downstream channels (statute_anchors, concept_*, doctrinal) directly index into these.
- **How helpful:** ⭐⭐⭐⭐ — necessary for several channels but only 14.7% court coverage (structural ceiling sits in retrieval, not coverage).

### 2.4 Citation-graph extraction (96% gold-as-node coverage) — **HELPFUL AS TIE-BREAKER**
- **What we did:** parsed every citation across `laws_de.csv` + `court_considerations.csv`, applied alias resolution (ATF / DTF / case-level / date-stripped / range-expanded), patched 92.9% → 96.0% gold-node coverage, persisted into `citation_graph_extracted.sqlite`.
- **Why it works:** powers two of the strongest v7.5 channels — `statute_backprop` (per-doc recall 0.453) and `graph_forward` (0.324) — by following citation edges from query-targeted laws back to citing court considerations.
- **Why limited:** 99.41% of gold-citation PAIRS have no graph edge (sister provisions in the same law don't cite each other). So graph as a *co-predictor expander* (predict X because Y was predicted) fails. As a *tie-breaker* / *channel contributor*, it works.
- **Evidence:** `personal_observations.md` Obs 2; `v7_4_fixes/cell_bodies_extracted_from_notebook/_cell34_body.py` channel implementation.
- **How helpful:** ⭐⭐⭐⭐ as channel, ⭐ as co-predictor.

### 2.5 Granularity-expanded gold + parent-link rule — **WORKS**
- **What we did:** for each train query whose gold contains a paragraph-level citation, expanded to also accept the article-level row in corpus. The parent-link rule prevents double-counting at prediction time.
- **Why it works:** without this fix, train gold-in-corpus is 71.54%; with it, 98.06%. Val was already 100%.
- **Evidence:** `data_insights/gold_analysis_coverage_and_parents/gold_citation_coverage.csv` documents the deltas.
- **How helpful:** ⭐⭐⭐⭐ — unblocks train-side experimentation entirely. Less critical for val.

### 2.6 9-dimensional dossier feature cache — **WORKS AT SMALL K**
- **What we did:** for each (qid, did) in the v7.5 top-50k, pre-computed: statute_int, lead_statute_int, concept_int, term_int, co-citation density, doctrinal match, chamber match, not-hardneg flag, fusion_rank. Saved as `dossier_features.npz`.
- **Why it works (at small K):** at K=200, F3 hybrid (3 rerankers + 9 dossier signals) reaches macro R@200 = 0.304 vs F0 plain fusion R@200 = 0.181 — a +12 point lift. At K=500: F3 = 0.423 vs F0 = 0.415.
- **Why it doesn't work above K=500:** F0 catches up and overtakes. At K=10k, F0 = 0.816 > F3 = 0.791.
- **Evidence:** `hybrid_rerank_results.json` in the cached cache; full sweep documented in the experiments skill.
- **How helpful:** ⭐⭐⭐⭐ for K≤500 use cases (Stage 2 prompting, final pick); ⭐⭐ for K>500.

### 2.7 Untitled75 reference notebook (F1 = 0.777) — **PROOF THAT THE TASK IS DOABLE**
- **What it does:** Qwen3-Reranker-8B + Qwen3-8B judge over a smaller candidate set; val-tuned threshold sweep.
- **Why it matters:** caveat is that it was tuned on the same val and on a smaller corpus, but architecturally it proves a Qwen3-only stack can hit F1=0.777 on this exact val.
- **Evidence:** `notebooks/06_high_scoring_reference/reference_F1_0.777_Untitled75.ipynb`. Plaintext dump in `research/nb_dump.txt`.
- **How helpful:** ⭐⭐⭐⭐⭐ — sets the architectural target.

### 2.8 Qwen3-Reranker-8B at K ≤ 200 — **PARTIALLY HELPFUL**
- **What we did:** cross-encoder reranking with `score = exp(yes) / (exp(yes) + exp(no))` per the Qwen team's calibration formula.
- **Why it works (at K ≤ 200):** the calibrated probabilities are usable as a confidence signal for the LLM judge's "auto-yes / auto-no" zoning.
- **Why it doesn't help recall:** above K=500, plain fusion already has everything the reranker finds.
- **Evidence:** standalone Qwen3-Reranker-8B at K=2000: R@2k = 0.247 (worse than F0 R@2k = 0.611).
- **How helpful:** ⭐⭐⭐ as score head for K≤200; ⭐ for recall above K=500.

### 2.9 Drive-pull tooling — **WORKS, INFRASTRUCTURE**
- **What we did:** OAuth-based Google Drive sync scripts (`scripts/drive_pull/`) with file-ID-based dedup manifest. Pulls only what isn't already local, resumable across runs.
- **Why it works:** lets us iterate on the recall-0.89 snapshot + cached scores without re-downloading the 200 GB Drive tree.
- **Evidence:** `_downloaded_ids.json` manifest now tracks N files; `drive_pull_curated.py` lets us pick the 2 GB of analysis-critical files only.
- **How helpful:** ⭐⭐⭐⭐ — unblocks local-machine work without losing access to the canonical Drive artifacts.

---

## 3. What didn't work, and why future attempts also won't work

These are documented failures with structural reasons not to retry.

### 3.1 BGE-reranker-v2-m3 on Swiss legal — **DEAD**
- **What we tried:** as one of three rerankers in the May-15 hybrid shootout.
- **Result:** R@2k macro = **0.073** (essentially random). 4 of 10 val queries scored exactly 0.000.
- **Why it failed:** BGE-reranker-v2-m3 is not trained on Swiss/German legal text; cannot recognize statute citation patterns like `Art. 11 Abs. 2 OR` or `BGE 142 III 296`.
- **Why retry won't help:** the failure is a domain-knowledge gap in the model weights. Without fine-tuning on Swiss legal (which we already tried with Legal-Swiss-RoBERTa, see §3.4), it can't learn this.
- **Verdict:** dropped. Don't include in any future reranker config.

### 3.2 Cross-encoder rerankers above K=500 — **DEAD**
- **What we tried:** F1 = Qwen3-only, F2 = 3-rerank RRF, F3 = 3-rerank + 9 dossier, F4 = F3 + hard-neg filter. All evaluated at K = 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000.
- **Result:** F0 (plain fusion, no reranker) wins at K ≥ 500. At K = 10k: F0 = 0.816 > F3 = 0.791. At K = 2k: F0 = 0.611 > F3 = 0.568.
- **Why it failed:** plain RRF fusion already aggregates 15 channels and surfaces nearly every gold the rerankers find. The reranker's per-pair scoring is noisy below the fusion signal.
- **Why retry won't help:** this is a property of the candidate distribution — gold is spread across 50k candidates, but rerankers can only meaningfully sort the top few hundred. Above that K, sorting accuracy decays faster than fusion does.
- **Verdict:** rerank stage = use ONLY at K ≤ 200, or drop entirely. ROADMAP Step 0 (the F3-without-rerankers diagnostic, 15 min, no GPU) will confirm we can drop it.

### 3.3 Dense embeddings alone (Qwen3-Embedding-8B) as a complete retriever — **DEAD**
- **What we tried:** retrieve top-K with cosine similarity over the full corpus, predict that set.
- **Result:** Macro F1 = **0.044**. R@1000 = 0.289 (Obs 3 ceiling).
- **Why it failed:** semantic similarity captures topical relevance but legal citation requires exact granularity matching. Two laws on the same topic still have different gold sets.
- **Why retry won't help:** even with a bigger encoder, the failure mode is task-mismatch, not capacity. Set retrieval ≠ similarity retrieval.
- **Verdict:** keep as ONE channel in the 15-channel fusion; do not use alone.

### 3.4 Legal-Swiss-RoBERTa fine-tune (exp_A6) — **DEAD**
- **What we tried:** loaded `joelniklaus/legal-swiss-roberta-large` (Rasiah et al. 2023, the Legal-Swiss-RoBERTa baseline in Stern's ACL 2025 Table 2), fine-tuned with SimCSE + MultipleNegativesRankingLoss.
- **Result:** stat_recall@500 = **0.101** vs BGE-M3 zero-shot baseline of **0.376**. A regression of −0.275.
- **Why it failed:** the base model is monolingual on Swiss legal; our val is English; the fine-tune data was insufficient or wrong-format; the contrastive loss probably collapsed under low-batch-size pressure.
- **Why retry won't help:** the model class is wrong for cross-lingual EN→DE/FR/IT retrieval. Even if we fix the fine-tuning recipe, the upper bound is monolingual.
- **Verdict:** dropped. Multilingual encoders (Qwen3-Embedding-8B) are the right base.

### 3.5 segment_lattice_v3 — **DEAD**
- **What we tried:** a deterministic segmentation lattice over court paragraphs, intended to surface paragraph-level matches.
- **Result:** court_recall = 0, mean recall = 0.40. The court component was broken.
- **Why it failed:** the segmentation didn't align with how court citations are structured (`E. [consideration_number]` pointers don't map to text segments cleanly).
- **Why retry won't help:** the citation structure is hierarchical text-positional, not text-segment-based. A lattice approach has the wrong topology.
- **Verdict:** abandoned. The 5.5 GB sqlite file still on disk (`artifacts/segment_lattice_v3.sqlite`) can be deleted.

### 3.6 Qwen3.5-35B-A3B for authority card enrichment — **DEAD**
- **What we tried:** 12 notebook iterations of `auth_cards_enrich_qwen35_*` exploring the larger MoE model under structured outputs.
- **Result:** ~80% invalid JSON output (800/1000 parse failures observed). No corpus-scale run completed.
- **Why it failed:** Qwen3.5-35B-A3B under JSON-schema-constrained decoding consistently breaks structure under our prompt template (the MoE attention misroutes when the structured-output token mask is tight).
- **Why retry won't help:** the failure is a model × constrained-decoding compatibility issue, not a prompting issue. Until vLLM's structured-output backend handles MoE routing better, this is blocked.
- **Verdict:** stick with Qwen3-8B-AWQ (which gets 99.987% valid output).

### 3.7 The "endgame" judge configuration — **DEAD AS-IS**
- **What we tried:** Qwen3-8B as LLM judge over 471 candidates per query with `<think>` budget, "say YES when uncertain" prompt, default-YES on parse failure, 200-token output truncation.
- **Result:** F1 < 0.10. 87% rubber-stamp YES rate. 1788 verdicts cached at `cache_endgame/judge/*` (all toxic).
- **Why it failed:** four compounding bugs:
  1. Default-YES on parse failure (should be default-NO)
  2. "Say YES when uncertain" prompt (wrong incentive)
  3. `<think>` token budget gets consumed by long Swiss legal text reasoning before the answer token
  4. 200-token output truncation cuts off the JSON output mid-key
- **Why retry won't help AS-IS:** these are systemic design errors. The cached verdicts are poisoned and must be deleted before any judge re-run.
- **Verdict:** delete `cache_endgame/judge/*` (1788 files). Re-design the judge with: default-NO on parse failure, force JSON via grammar, no `<think>` budget, 32-token output, "say NO when uncertain".

### 3.8 HGE / R-GCN learned link prediction (Missing Link paper) as a co-predictor — **DEAD**
- **What the paper claims:** 87-88% macro AP on 200k-case German legal graph with joint case→case + case→law learned link prediction. Code at `research_repos/missing_link/` (unused by our pipeline).
- **Why it won't work for us:** Obs 2 shows that 99.41% of our val gold-citation PAIRS have no edge in the extracted graph. Sister provisions in the same Swiss law don't cite each other — the graph topology that the GNN would learn from doesn't contain the prediction signal we need.
- **Verdict:** ruled out as co-predictor. The graph already contributes maximum value as a deterministic channel (§2.4); a GNN on top adds noise.

### 3.9 Stern's judgment-criticality classification task — **OUT OF SCOPE**
- **What the paper does:** classifies Swiss federal court decisions on `rcds/swiss_criticality_prediction` for LD-Label / Citation-Label.
- **Why we shouldn't try:** different task — judgment criticality classification ≠ citation retrieval. Even succeeding on it doesn't move our F1 because our metric is set retrieval, not classification.
- **Verdict:** off-task. Don't pursue.

### 3.10 Improving recall beyond R@50k = 0.901 — **STRUCTURALLY BLOCKED**
- **What it means:** val_003 R_max in the v7.5 pool = 0.766. This is the worst per-query recall in the pool. We cannot exceed val_003's contribution to macro F1 unless we find more of val_003's gold.
- **Why it's blocked:** the gold for val_003 includes citations that are not text-searchable, not graph-reachable from the query's expansion, and not in any of the 15 channels' output. They're effectively invisible to our current funnel.
- **Implication for F1:** even if everything downstream (rerank, dossier, judge, K-pick) is perfect, the maximum achievable macro F1 ≈ 0.93 (capped by val_003's 0.766 × perfect-precision contribution). F1 = 0.8 is reachable; F1 = 0.9+ is not without a fundamentally new retrieval channel.
- **Verdict:** don't spend more time chasing pool recall. Focus on Stages 2-3.

### 3.11 Adding more LLM enrichment coverage (the 85% un-enriched court gap) — **NOT THE BOTTLENECK**
- **What it sounds like:** we've only enriched 363k of 2.47M court rows; surely more enrichment helps?
- **Why it doesn't:** the val_003 R_max = 0.766 ceiling is set by RETRIEVAL, not enrichment. The candidate pool already contains the val_003 gold that's findable; enriching more rows widens the haystack but doesn't reveal more needles. And enrichment of 30+ hours of Colab compute would be needed.
- **Verdict:** parked. Revisit only if val_003 R_max moves AND we have evidence missing enrichment is the cause.

---

## 4. What we haven't tried yet that should help us reach the goal

Priority-ordered. Each item has a falsifiable pass criterion so we know if it worked.

### 4.1 PRIORITY 1 — F3-without-rerankers diagnostic (15 minutes, no GPU)
**What:** recompute F3's RRF by dropping the 3 reranker rank columns. Keep only the 9 dossier signals + fusion_rank.

**Why it should help:** if dossier alone matches F3 at every K, we can drop the entire reranker stage. That saves ~1100 seconds/query of GPU time and simplifies the pipeline.

**Pass criterion:** F3-no-rerank within 2 points of F3 at every K from K=50 to K=50000.

**Data needed:** `scores_qwen3.npz`, `dossier_features.npz` — both already cached locally in `drive_sync/swiss_law/hybrid_rerank_shootout_cache/cache/`.

**Why it's first:** zero-cost, fast, and the result determines whether Step 4.2 and 4.3 need reranker integration.

### 4.2 PRIORITY 2 — AdaptiveK score-gap variable-K final pick (2 hours)
**What:** replace the threshold = 0.65 logic in `threshold.json` with score-gap cut + buffer B=5 per the Megagon EMNLP 2025 paper. Concretely: sort fused scores descending, compute `np.diff()`, find the largest negative gap below the top-N, cut there + B more.

**Why it should help:**
1. Val gold size ranges 10-47 with median 22 — fixed K is provably suboptimal.
2. The fixed threshold 0.65 currently gives F1 = 0.498, with the K count varying per query but the threshold not. We're throwing away the per-query shape of the score distribution.
3. The paper claims up to 99% token reduction with no F1 drop on aggregation-style retrieval. Our task is exactly an aggregation-style retrieval (gather all relevant citations).

**Pass criterion:** Macro F1 ≥ 0.55 (vs current 0.498). Stretch: ≥ 0.60.

**Data needed:** existing fused scores from F0 or F3. No new data, no GPU.

**Why it's second:** highest leverage per hour of work. The score-gap approach is a one-line algorithmic change applied to data we already have.

### 4.3 PRIORITY 3 — Cascade Dossier Phase 1 in Stage 2 prompt (1 day)
**What:** the agreed plan from `research/cascade_dossier_plan.md`. Add three structured features into the LLM judge prompt's per-candidate block:
- **Channel-of-arrival fingerprint** — list of which of the 15 v7.5 channels surfaced this candidate (already in `channel_hit_sets`).
- **Statute-target intersection** — list of statute anchors the candidate has in common with the query expansion's `statute_targets`.
- **Co-citation density** — for each law in the top-100, how many top-100 court paragraphs cite it.

**Why it should help:** the LLM judge currently sees raw text and has to infer relevance. Pre-digesting these cross-features changes the task from "guess relevance" to "confirm dossier against text" — a much easier verification task. Untitled75 (F1=0.777) essentially does something similar with a smaller candidate pool; we're recreating that conditioning at our pool scale.

**Pass criterion:** Stage 2 mean R@100 lifts from current ~0.55 → ≥ 0.70.

**Data needed:** existing v7.5 snapshot + dossier features. No new data, no extra GPU (LLM judge runs anyway).

**Why third:** depends on a working judge; needs Step 4.2 to feed it the right K.

### 4.4 PRIORITY 4 — Granularity-aware final pick post-filter (1 hour)
**What:** apply the `gold_parent_link_check` rule at submission time: if the predicted set contains a paragraph-level citation (`Art. 11 Abs. 2 OR`), remove its article-level parent (`Art. 11 OR`) from the same set if also predicted.

**Why it should help:** the grader marks parent + child as a double-count error in some cases. Our current pipeline doesn't apply this rule at output time. Even small precision wins compound across 10 val queries.

**Pass criterion:** Macro F1 ≥ 0.60 (vs current 0.498). Stretch: ≥ 0.65.

**Data needed:** `data_insights/gold_analysis_coverage_and_parents/gold_parent_link_check.csv` already encodes the rule.

**Why fourth:** trivial implementation, modest expected gain. Doesn't require anything from Steps 4.1-4.3 to land.

### 4.5 PRIORITY 5 — Talos top-K F1 differentiable loss (1-2 days)
**What:** the only one of the 7 variable-K papers that's a learning objective rather than a post-hoc cutoff. From arXiv 2601.19276 (Jan 2026). Defines a quantile-based learned threshold that's differentiable end-to-end and directly optimizes top-K accuracy.

**Why it should help:** we're trying to optimize macro F1 (a top-K F1 metric). Every other variable-K method (AdaptiveK, CAR, AcuRank, PSI-Rank, GRLStop, Risk Control) is post-hoc — they cut after scoring. Talos lets us TRAIN with the cut-objective baked in.

**Pass criterion:** Macro F1 ≥ 0.65 (the project target's lower bound).

**Data needed:** train + val splits; a small held-out calibration set. We have train (1139 queries) and val (10) — the train/val distribution shift is the risk (train is 99% DE, val is 100% EN). May need to construct a held-out calibration partition from val itself.

**Why fifth:** highest implementation cost; most powerful if it works. Best done after Step 4.2 confirms variable-K is the lever.

### 4.6 PRIORITY 6 — CaseLink-style GAT on the v7.5 pool (2-3 days)
**What:** from UQLegalAI COLIEE 2025 (F1=0.2962 SOTA). Small graph attention network trained on the v7.5 top-50k pool with val gold supervision. Predicts a per-candidate score combining structural features (graph position, channel weights, paragraph role).

**Why it should help:** our current channel weights and role boosts are hand-tuned in `v7_4_fixes/`. A GAT can learn the optimal combination from val gold rather than relying on hand-tuned per-query weights. The cascade dossier plan already enumerates the features a GAT would learn.

**Pass criterion:** Macro F1 ≥ 0.65 (matching Step 4.5's bar).

**Data needed:** v7.5 snapshot, dossier features, val gold. PyTorch Geometric or DGL.

**Why sixth:** highest unknown — training a GAT on 10 val queries with proper cross-validation is hard. Don't start until Steps 4.1-4.4 have moved F1 to ≥ 0.55.

### 4.7 Lower-priority opportunities

| Item | Why it might help | Why it's lower priority |
|---|---|---|
| **CAR (1-D K-means on score distribution)** | First-elbow cut. Could outperform AdaptiveK if val score distributions are multi-modal. | Position-penalty would likely hurt val_002 (47 golds). |
| **AcuRank (TrueSkill uncertainty stopping)** | Bayesian principled stopping. | Needs a listwise reranker our pipeline doesn't run; if Step 4.1 drops rerankers, AcuRank doesn't fit. |
| **PSI-Rank (LLM-generated pivot pseudo-doc)** | 66% inference speedup at no quality loss on TREC-DL. | We already run an LLM judge; adding a pivot LLM call is incremental cost without clear gain. |
| **GRLStop (RL stopping policy)** | Adaptive to target recall. | Needs training data; `research_repos/RLStop/` is cloned but unmodified. |
| **Two-stage Risk Control (conformal prediction)** | Distribution-free coverage guarantee. | Val is only 10 queries — calibration set too small. |

### 4.8 Why this list, and not something else

Our `FINDINGS_SYNTHESIS.md` analysis surfaced that the gap is exclusively in the K-selection / score-aggregation layer. Every priority 1-5 item directly attacks that layer. The two anti-patterns we avoid:
- Spending more time on retrieval (already at structural ceiling)
- Spending time on bigger rerankers / encoders (proven inferior or already at parity)

The expected cumulative F1 trajectory if Steps 4.1-4.4 all pass their criteria:
- After Step 4.1 (just diagnostic): F1 unchanged but pipeline simpler
- After Step 4.2 (AdaptiveK): F1 ≥ 0.55
- After Step 4.3 (Cascade Dossier Phase 1): F1 ≥ 0.60 (target lower bound)
- After Step 4.4 (Granularity post-filter): F1 ≥ 0.65
- After Step 4.5 OR 4.6 (Talos OR GAT): F1 ≥ 0.70 (mid-target)

To reach F1 = 0.8 we likely need both 4.5 AND 4.6, plus all of 4.1-4.4.

---

## 5. Operational reminders for the next run

| Action | Why | Time |
|---|---|---|
| **DELETE** `cache_endgame/judge/*` (1788 files, ~860 KB) | 87% rubber-stamp YES; poisons any judge re-run | 1 min |
| **Move** 4 misfiled notebooks (`investigate_misc_notebook`, `pdf_extraction_marker_*`, `pipeline_no_debug_prints`, `reference_F1_0.7716_scored`) to `notebooks/_misfiled_other_competitions/` | They belong to Retroviral / Make Data Count / AIMO-3 / RT-activity competitions — not this project | 5 min |
| **Refresh** stale Drive law-descriptors mirror (`drive_sync/swiss_law/llm_enrichment_jsonl_checkpoints/law_llm_descriptors_0000000_all.jsonl` is byte-identical to local `.bak`, not current) | Wrong source-of-truth | 5 min |
| **Investigate** `scripts/build_pipeline/build_court_authority_cards.py:1238-1241` bug | Producing 5.5 GB of duplicate `_chunkNN_chunkNN.jsonl` files | 15 min |
| **Use FP16 KV, not FP8 KV** when running AWQ Marlin inference on Blackwell SM 12.0 | FP8 KV crashes deterministically; documented in enrichment runs | n/a (process rule) |
| **First markdown cell** of every new experimental notebook MUST state pass / fail criterion | Project convention; carries from endgame_handoff_2026-05-09.md | n/a (process rule) |

---

## 6. Document map for deep-dive

If any section of this doc raises a question, the deeper-dive lives at:

- **`README.md`** — repo layout map (every folder, what it contains)
- **`ROADMAP.md`** — concrete 4-step next plan with pass/fail criteria
- **`docs/FINDINGS_SYNTHESIS.md`** — the strategic synthesis behind this doc
- **`docs/data_analysis/`** — DATA_INVENTORY.md + 5 section files (~2,200 lines)
- **`docs/paper_analysis/`** — PAPER_TO_PRACTICE.md + 5 section files (~1,062 lines)
- **`docs/experiments_analysis/`** — EXPERIMENTS_TRACE.md + 4 section files (~1,829 lines)
- **`.claude/skills/swiss-citation-orchestrator/`** + sibling skills — always-loaded project context
- **`research/personal_observations.md`** — empirically verified Obs 1-4 (the binding constraints)
- **`research/cascade_dossier_plan.md`** — the agreed Phase 1 plan (input to Step 4.3)
- **`research/endgame_handoff_2026-05-09.md`** — pre-v7.5 forensic record + "DO NOT redo" list
