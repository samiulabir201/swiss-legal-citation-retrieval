# Swiss Citation Extraction — Consolidated Notebook Summary
_Generated 2026-05-15. One-document survey of every dataset used, every process attempted, and every measured result across the 90 notebooks in `notebooks/`. Sources: per-notebook code+output dumps in [_inventory/](_inventory/), [FINAL_NOTEBOOK_RATIONALE.md](FINAL_NOTEBOOK_RATIONALE.md), and [research/endgame_handoff_2026-05-09.md](../research/endgame_handoff_2026-05-09.md)._

**Project goal:** macro F1 0.6–0.8 on `data/val.csv` (10 English queries) over a closed corpus of 175,933 Swiss law articles (DE) + 2,476,315 court considerations (DE/FR/IT) — **≈2.65M documents total**.

---

## Part 1 — Every dataset used (and what for)

### 1.1 Competition / raw data

| File | Size / shape | Used in | Purpose |
|---|---|---|---|
| `data/train.csv` | 1,139 queries (99.4 % DE, median 2 citations) | every pipeline run, fine-tune (exp_A6), enhance() lexicon (Untitled75), conformal calibration (final notebook plan) | training queries + gold; **explicit project rule: train is unreliable** (3-axis shift vs val: language, citation-count, court share) |
| `data/val.csv` | 10 English queries, 10–47 gold citations each, 251 gold total | every retrieval/eval run | the held-out target metric set |
| `data/test.csv` | 40 English queries | cascade_legalmalr_v1_*, planned submission | unseen target |
| `data/laws_de.csv` | 175,933 statute rows (citation, title, text, law_code, language=de) | exp_A1, exp_A2, exp_A3, exp_A5, early_dense_embedding_test, exp_A6, enrich_laws_de_*, all pipeline/funnel notebooks | primary law corpus |
| `data/court_considerations.csv` | 2,476,315 court paragraph rows (citation, text, decision_id, year, law_area, chamber, language, region) | exp_A5, court_concept_*, court_llm_descriptor_*, every pipeline notebook | primary court corpus |
| `data/train_granularity_expanded.csv` | gold after article→Abs. expansion (train coverage 71.54 % → 98.06 %) | pipeline_iteration_latest_*, endgame_colab | granularity-aware gold |
| `data/val_granularity_expanded.csv` | val gold post-resolver (still 100 %) | pipeline_iteration_latest_*, endgame_colab | granularity-aware gold |
| `data/checkpoints/law_llm_descriptors_0000000_all.jsonl` (mirror in `law_json_llm_output/`) | 173,033 records | every retrieval notebook (concept_en channel, LLM target expansion) | **full** Qwen3-8B enrichment of `laws_de` |
| `outputs_from_363k_run/court_llm_descriptors_0000000_all.jsonl` | 363,258 records (only ~15 % of 2.47 M court rows) | every retrieval notebook | partial Qwen3-8B enrichment of court considerations — the known 85 % coverage gap |
| `data/court_considerations_classified_citations.jsonl` / `laws_de_classified_citations.jsonl` | IOB-tagged spans | exp_A6, citation-graph extraction | NER labels for law/citation spans |
| `data/court_considerations_links.json` / `laws_de_links.json` | citation→reference link tables | citation graph build | seeds for `citation_graph_extracted.sqlite` |
| `val_translated_de.pkl`, `test_translated_de.pkl` | German translations of EN queries | exp_A1…A4, exp_A5 | DE-side dense retrieval / HyDE |
| HF `rcds/swiss_citation_extraction` (87k+12k+27k) | IOB-tagged paragraphs | exp_A6 | RoBERTa fine-tune labels |
| `research_papers/` (PDFs) | external academic corpus | pdf_extraction_marker_pymupdf_pdfminer | feeds the 2025–2026 research-paper review |

### 1.2 Authority cards (intermediate hand-built / static enrichment)

| File | Used in | Purpose |
|---|---|---|
| `artifacts/law_authority_cards_v2_unified.jsonl` | endgame_colab, pipeline_end_to_end_*, all anchor_funnel | law cards with title/text/anchors/language |
| `artifacts/court_authority_cards_v5_unified.jsonl` (10.4 GB) | endgame_colab, all anchor_funnel + pool_v75, all auth_cards_enrich_* | court cards with citation/text/role/anchors |
| `court_authority_cards_v4_target_cards.jsonl` (363,258 cards, 1029 MB) | all 12 `auth_cards_enrich_*` notebooks | the RAG-target subset that was put through Qwen3-8B / Qwen3.5-35B for English summary + concepts + topics |

### 1.3 Unified retrieval store

| File | Used in | Purpose |
|---|---|---|
| `artifacts/unified_retrieval.sqlite` (~24 GB) | endgame_colab, all `pipeline_end_to_end_*`, all `pipeline_iteration_latest_*` | `documents` (2.65 M) + FTS5 BM25 + `statute_links` (5.30 M) + `case_links` (1.81 M) + `adjacent_law_links` (348 k) |
| `unified_embedding_input.parquet` (2,652,248 rows) | embed_unified_corpus_qwen3_8b_blackwell, retrieve_unified_corpus_qwen3_8b / v3 | the text fed to the encoder (doc_id, family, citation, vector_text, char_len) |
| `docs_meta.parquet` | retrieve_unified_corpus_* | doc_id, family, citation, law_code, authority_score |

### 1.4 Embeddings

| File | Shape | Used in | Purpose |
|---|---|---|---|
| `embeddings/qwen3_8b_unified_chunk000..026.npy` (27 chunks) | (2,652,248, 4096) fp16, ~21 GB | every retrieval notebook from v4 onward; loaded straight to GPU as `E_GPU` | the canonical dense embedding bank (Qwen3-Embedding-8B) |
| `embeddings/qwen3_8b_unified_manifest.parquet` | 2,652,248 rows | every retrieval notebook | doc_id ↔ row-index map |
| `embeddings/qwen3_8b_unified_summary.json` | scalar | sanity checks | norms, NaN counts |
| `laws_bgem3.npy`, `court_bgem3.npy` | (~2.65 M, 1024) fp16 | exp_A1, exp_A2, exp_A3, exp_A4, exp_A5, early_dense_embedding_test, exp_A6 | BGE-M3 dense baseline (superseded by Qwen3-8B) |
| `query_vecs_A3.npz`, `query_vecs_A6.npz` | (10, 1024, 4 forms) | exp_A3, exp_A4, exp_A6 | EN / DE / HyDE / Enum query encodings |
| `laws_swissroberta.npy`, `court_swissroberta.npy` | (~2.65 M, 1024) | exp_A6 | legal-swiss-roberta-large fine-tuned vectors |

### 1.5 Citation graph

| File | Size | Used in | Purpose |
|---|---|---|---|
| `data_insights/citation_graph_extracted.sqlite` (patched) | 1.99 GB, 20.49 M edges, out-deg avg 27.2, in-deg avg 14.0 | anchor_funnel_v6 onward, every pool_v75_*, endgame_colab | 1-hop, 2-hop, judgment-importance ranking (median 7, p75 44, max 204 k incoming) |
| `gold_cocitation_prior_train.pkl` / `gold_cocitation_prior_trainval.pkl` | dict | pipeline_iteration_latest_fixed_v2 / v3 | co-citation priors (tested, marginal lift) |
| `numbered_cocitation.pkl`, `citation_signal_lookup.pkl`, `reference_graph_v2.pkl`, `statue_to_numbered.pkl`, `numbered_to_statutes.pkl` | dicts | pipeline_iteration_latest_* | canonical-citation resolution + co-occurrence signals |

### 1.6 v7.5 snapshot (canonical recall-0.89 pool — the "warm-boot" for everything in folder 01)

`research/anchor_funnel_val001_v7/snapshot/` contains:

| File | Used in | Purpose |
|---|---|---|
| `corpus_snapshot.json.gz` | every folder-01 cascade / rerank / precision notebook | 255,713-doc corpus snapshot (all union-merged channel hits across 10 queries) |
| `per_query_snapshot.json` | every folder-01 notebook | per-query `final_topk` ranking + R@K curve + channel_hit_sets |
| `gold_doc_sets.json` | every folder-01 notebook | per-query gold doc ids (for in-pool / not-in-pool diagnostics) |
| `all_targets.json` | every folder-01 notebook | LLM-named `statute_targets`, `concept_targets`, `term_targets`, `legal_area` |
| `hyde_aspects.json` | every folder-01 notebook | HyDE aspects per query (used for aspect-coverage stage) |
| `config.json`, `paths.json` | warm-boot | config of the producing v7.5 pool |

### 1.7 Endgame run forensic caches

| File | Used in | Notes |
|---|---|---|
| `cache_endgame/hyde_cache.json` | endgame_colab | 10 HyDE answers — **reusable** |
| `cache_endgame/german_expansion_cache.json` | endgame_colab | 10 DE expansions — **reusable** |
| `cache_endgame/query_embeddings.npy` | endgame_colab | 10 query embeddings — **reusable** |
| `cache_endgame/rerank/<sha1(q)>.json` × 10 | endgame_colab | 399–529 rerank scores per query — **reusable** |
| `cache_endgame/judge/<sha1(q)>/<sha1(c)>.json` × 1788 | endgame_colab | judge verdicts — **MUST be deleted before re-run** (toxic) |

### 1.8 Models referenced

`Qwen3-Embedding-8B` (encoder), `Qwen3-Reranker-8B` (cross-encoder reranker), `Qwen3-8B-AWQ` (judge + LLM enrichment, the workhorse), `Qwen3-32B` (query enumeration / HyDE / planner), `Qwen3.5-35B-A3B` MoE (auth-card enrichment exploration), `BGE-reranker-v2-m3` and `jina-reranker-v2-base-multilingual` (3-model shootout), `BGE-M3` (early dense baseline in exp_A1–A5), `legal-swiss-roberta-large` (exp_A6 fine-tune), `Qwen2.5-7B-Instruct` (exp_A2).

### 1.9 Research-paper PDF corpus

`research_papers/` (Adaptive-k, CAR, PSI-Rank, AcuRank, Two-Stage Risk Control, Missing Link, From Citations to Criticality, UQLegalAI@COLIEE2025, GRLStop, Talos, Qwen3-Embedding TR, etc.) — ingested via the single PDF-extraction notebook (folder 14) and synthesized in `research_papers/INDEX.md` and `FINAL_NOTEBOOK_RATIONALE.md`.

---

## Part 2 — Every process tried (technique catalogue)

Each entry: what it does → where it was tried → what came of it.

### 2.1 LLM enrichment of the corpus (precondition for everything else)

| Process | Notebooks | Outcome |
|---|---|---|
| Qwen3-8B AWQ via vLLM, JSON-mode, descriptor schema (english_summary, legal_rule, concepts_en, terms_de_to_en, defined_terms, addressees, sanctions, specificity_score, provision_role_llm…) over **laws_de** | folder 07: `enrich_laws_de_qwen3_8b_kaggle{,_optimized,_pre_optimization}`, `investigate_laws_de_token_optimization` | **~100 % coverage** of 173,033 laws; 11.16 rows/sec on Blackwell; flashinfer / max_num_seqs / max_model_len tuning; bucketed token-length config |
| Qwen3-8B AWQ via vLLM minimal descriptor schema (legal_area, primary/secondary_domain, topic/sub/micro, concepts_en, terms_original, doctrinal_rule, legal_test, fact_pattern_tags, procedural_context, paragraph_role, authority_role, specificity_score) over **court_considerations** | folder 08: 7 notebooks: 10k_optimized, qwen3_8b_awq, kaggle, safe_2t4_vllm, dual_t4_vllm, blackwell_optimized_v4, blackwell_optimized_363k_run | Throughput evolved 0.24 rows/s (dual T4) → 13.17 (Blackwell v4) → **22.85 rows/s** (363k run); **363,258 / 2,476,315 = 15 % coverage** — the known gap |
| Court "concept" enrichment with full anchor schema (statute_anchors, case_anchors, paragraph_role, authority_role, outcome_signal, etc.) | folder 09: `court_concept_smoke_test_10_{base,v3_local}`, `court_concept_enrichment_robust_v4_predecessor`, `court_concept_enrichment_robust_v5_t4`, `court_concept_enrichment_minimal_llm` | smoke → robust v4 → robust v5 (awq_marlin) → minimal (LLM emits descriptors only; deterministic normalizer owns anchors). v5 + minimal are the canonical ones for downstream |
| Authority-card enrichment loop (363,258 court "RAG target" cards → english_summary + topic + question + rule + holding + factual_context + concepts + search_keywords + natural_language_queries) | folder 10: 12 notebooks (auth_cards_enrich_qwen35_* and auth_cards_enrich_qwen3_8b_*) | Long stability/JSON-validity grind; Qwen3.5-35B-A3B (MoE) explored but rolled back to Qwen3-8B-AWQ kaggle_local as canonical; vLLM Pillow / flashinfer / pillowfix workarounds |
| Kaggle vLLM text-to-JSON infrastructure scaffolds (T4 stability, FlashInfer JIT `-lcuda` fix, TRITON_ATTN fallback, custom_all_reduce disable, enforce_eager, TP=1/2, checkpoint/resume) | folder 13: 11 kaggle scaffolds (`kaggle_qwen3_8b_awq_vllm_text_to_json_{base,stable,t4_fixed,t4_flashinfer_fixed}`, `kaggle_qwen3_awq_text_to_json_no_flashinfer_with_fallback`, `kaggle_qwen3_8b_awq_local_batchfix`, etc.) | Pure engineering — no algorithm change; got Kaggle 2×T4 to run JSON-mode batch inference reliably |

### 2.2 Embedding / encoding

| Process | Notebooks | Outcome |
|---|---|---|
| BGE-M3 dense encoding over `laws_de` + `court_considerations` | folder 11: exp_A1, exp_A2, exp_A3, exp_A4, exp_A5; folder 12: early_dense_embedding_test | baseline; abandoned for Qwen3-8B |
| **Qwen3-Embedding-8B** chunked encoding of the unified 2.65 M-row corpus on Blackwell, 4096-dim fp16, length-sorted batches, SDPA / cuDNN flash backend, 27 chunks ×100 k rows | folder 05: `embed_unified_corpus_qwen3_8b_blackwell` | All 27 chunks cached, L2-norm std ≈ 0.0018, no NaN — the canonical embedding bank used everywhere downstream |
| Legal-swiss-RoBERTa-large SimCSE pre-train (466 k pairs) + triplet fine-tune (117 k pairs, BGE-M3 hard negatives) | folder 11: exp_A6_legal_roberta_finetune | best ft_rrf@500 = **0.101** → fail; abandoned |
| Last-token pooling + Qwen3 instruction-prefix encoding bug fix (corpus encoded with prefix, queries weren't) | anchor_funnel_v5 | took recall@1000 on val_001 from 0.238 → **0.357** |

### 2.3 Retrieval — channel inventory (15 channels total at v7.5 maturity)

| Channel | First built in | Description |
|---|---|---|
| BM25 (FTS5 on enriched text, multi-language) | retrieve_unified_corpus_v3, exp_A1 | per-language FTS5 indices (DE/FR/IT/EN) by v7.4; ~91 k token→code association per lang |
| `enhance()` token→law-code lexicon expansion (corpus-derived) | reference_F1_0.777_Untitled75; built into anchor_funnel_v6 | boosts BM25 queries with codes statistically associated with the query tokens |
| vector_raw (Qwen3-Embedding-8B, brute-force GPU `E_GPU @ q`, top-k) | anchor_funnel_v4 onward | ~50 ms/query; replaces FAISS-IVF entirely |
| vector_enriched (vector with keyword/code boost mixed into the query text) | anchor_funnel_v4 | |
| vector_hyde (HyDE — Qwen3-8B generates a hypothetical answer, embed it, retrieve) | anchor_funnel_v7.5 (pool_v75_hyde_test); exp_A2/A3; endgame_colab | "no lift" verdict in pool_v75_multiquery_hyde_test |
| statute_anchor (regex `Art. N CODE` parsing of the query) | retrieve_unified_corpus_v3, exp_A1 | helps only when query names a code (rare on val) |
| case_anchor (BGE/DTF/docket regex) | retrieve_unified_corpus_v3 | rarely fires on val |
| law_direct_match (every law row whose canonical citation matches an LLM-named or co-cited statute target — uncapped, guaranteed) | anchor_funnel_v4 | per-channel mean recall best in v7.4 |
| court_statute (court rows annotated with one of the LLM-named statutes; multi-match + role boost in v7.4) | anchor_funnel_v4 | |
| concept_en (rows whose `concepts_en` overlaps with LLM concept_targets) | anchor_funnel_v1 | strong in v5 (combined with vector → 0.238 on val_001) |
| term_orig (DE/FR/IT term overlap on `terms_original`; lemmatization-aware in v7.4 — 521 k-key lemma index) | anchor_funnel_v1 | |
| per_area_bedrock (most-cited canonical statutes within the LLM-named legal_area, code-filtered to LLM list) | anchor_funnel_v4 + filter in v5 | covers the rank-100–500 procedural cluster |
| co_citation (top-K co-citation neighbours per LLM statute target; global-frequency filter ≤ 5,000) | anchor_funnel_v4 + filter in v5 | unfiltered = noise dominated; filtered = stable mid-tail |
| sibling_expansion / court_base (caught court row → all `E`s of the same judgment; non-deterministic slice fixed; **judgment-importance** breaks ties) | anchor_funnel_v1 → v7 (importance scoring) | sized for 100 seeds × 20 Es = 2 k–5 k |
| graph_forward / graph_reverse / graph_2hop (BFS/DFS over the patched 20.49 M-edge graph) | anchor_funnel_v7 | graph_forward per-channel mean recall 0.324 on val_001; 2hop is mostly empty |
| statute_backprop (every caught court row contributes its cited statutes — surfaces Art. 100 BGG / Art. 422 StPO etc. without query naming them) | anchor_funnel_v6 | **universal best per-channel mean recall 0.453** in v7.4 |

### 2.4 Fusion / pool construction

| Process | First seen | Notes |
|---|---|---|
| Reciprocal Rank Fusion (`k_const=60`) | retrieve_unified_corpus_v3 | the universal fusion |
| Authority amplification (α=0.20 on authority_score) | retrieve_unified_corpus_v3 | |
| Channel weights tuned per-channel mean recall + structural role | anchor_funnel_v7.5 | weights {statute_backprop 2.5, graph_forward 2.0, concept_en 1.8, court_statute 1.5, per_area_bedrock 1.5, vector_hyde 1.5, vector_raw/enriched 1.0, ...} |
| Guarantee channels (high-precision channels' hits prepended before RRF tail; round-robin) | anchor_funnel_v4 (2-channel) → v7.5 (**7-channel**, 130 each → 910 guarantee slots) | the v7.5 lift mechanism |
| Code-family expansion (top-8 via corpus-derived `code_pair_count`) | v7.5 | bumped macro R@50k from 0.7 (v7.4) to 0.89 |
| Statute/case canonicalization (StPO→312.0, CPP→312.0 etc., 2,147 known forms) | retrieve_unified_corpus_v3 | |
| Enrichment prefilter (token-inverted index over 2.16 M enrichment cards → narrow 2.65 M → 500 k–1 M before BM25/vector) | endgame_colab Cell 14 | **built but never wired into `retrieve()` in the endgame run** — the #1 open item in `endgame_handoff_2026-05-09.md` Move 1 |
| Role gate (drop notification/cost/dispositif paragraphs, drop cantonal court paragraphs) | v7.5 | verified: 102/102 val gold court paragraphs are federal |
| Snapshot export (corpus_snapshot.json.gz + per_query_snapshot.json + gold_doc_sets + all_targets + hyde_aspects + config + paths) | pool_v75_multiquery | the canonical warm-boot for folder 01 |

### 2.5 Query expansion / understanding

| Process | Notebooks | Result |
|---|---|---|
| Qwen2.5-7B-Instruct HyDE + Enumeration (15 citations per query) | exp_A2 | best 0.295 @500 — fails 0.40 gate |
| **Qwen3-32B** HyDE + Enumeration | exp_A3 | 0.376 @500 — still fails 0.40 gate, but +8 % over 7B |
| Structured query expansion (legal_domain, applicable_codes, german_terms, concept_seeds) | endgame_handoff §7 Move 2 (plan); built into all anchor_funnel via `all_targets.json` | the LLM-emitted `statute_targets / concept_targets / term_targets / legal_area` |
| Per-query HyDE aspect list (≥1 candidate must cover each aspect at p > 0.5) | final notebook plan (3f) | aspect-coverage lower clamp |
| German keyword expansion (PMI over corpus) | endgame_colab | cached 10 expansions; effect not isolated |

### 2.6 Cross-encoder rerank

| Process | Notebooks | Result |
|---|---|---|
| BGE-reranker-v2-m3 on top-1000 RRF (EN vs DE forms) | exp_A4 | rerank @50 = **0.114** (vs RRF 0.161) — **hurts** |
| BGE-reranker on HyDE-mean / HyDE-max aggregation | exp_A4b | @50 = 0.174 — marginal win, dropped |
| Qwen3-Reranker-8B with HF-canonical prompt, `convert_tokens_to_ids("yes")`, `padding_side='left'`, `<think>\n\n</think>` suffix, `softmax([logit_no, logit_yes])[:,1]` calibrated probability | endgame_colab; pool_v75 canonical Stage-1 cell; precision_v1_final | top-5 in endgame ran all court; bottom-5 all law with score 0.0 — suspected law-card text-feed bug (Move 3) |
| 3-model cross-encoder shootout (Qwen3-Reranker-8B vs BGE-reranker-v2-m3 vs jina-reranker-v2-base-multilingual) on top-50 k from snapshot | rerank_hybrid_3model_shootout(_with_outputs) | per-config R@K matrix; no single reranker hit R@2000 = 0.7 alone |
| Rerank-only "no-cascade" diagnostic | rerank_only_diagnostic_F3_no_rerank | fusion baseline R@2000 0.381–0.520 per query (val_001–val_010) |
| List-wise reranker prompt with 7-category judgments (Very Strong/Strong/Moderate/Weak/None/Irrelevant/Non-Swiss) on top-30, mini-index (65 k docs) | reference_F1_0.777_Untitled75 | **F1=0.777 on val** (5-query subset, train-tuned K=5) — the headline reference |
| `0.7 × normalized BM25 + 0.3 × reranker_score`, zone routing (auto-yes / auto-no thresholds) | Untitled75; endgame_colab | thresholds 0.55/0.25 calibrated for that fused scale; in endgame the RRF was 0.01–0.10 so thresholds never fired (every candidate went to judge) |

### 2.7 LLM judge

| Process | Notebooks | Result |
|---|---|---|
| Qwen3-8B 7-category judge + zone routing | endgame_colab Cell 16 | **broken: 4 compounding bugs** — recall-biased prompt ("say YES when uncertain"), parser default-YES on parse failure, `<think>` chain-of-thought consuming the 200-token budget, max_new_tokens=200 truncation. Judge said YES to **87 %** of 1,774 candidates → F1 ≈ 0.04 |
| Qwen3-8B judge with patches (`enable_thinking=False`, default-NO on parse failure, drop "say YES when uncertain", max_new_tokens=24, response = `VERDICT: YES\|NO`) | planned in endgame_handoff Move 3; partially in pipeline_iteration_latest_fixed_v3 | 4 bugs documented as closed in v3 |
| Qwen3-8B binary judge (24 tokens, no thinking, T=0) on top-200 | cascade_legalmalr_v1_{base,optimized,poc} | Stage 2 of 2-stage cascade |
| Qwen3-8B judge (64 tokens, T=0) on top-2000 with confidence-gated promotion (auto-yes if YES & conf≥0.85, auto-no if NO & conf≥0.85) | precision_v1_final | Stage D |
| Co-citation prior + dual prior train/trainval variants | pipeline_iteration_latest_fixed_v2 / v3 | marginal lift, kept in v3 |
| Evidence-quote verifier (require verbatim substring of doc text; replace, don't drop, on failure) | final notebook plan Phase 6 | not yet executed |

### 2.8 Variable-K / stopping (the F1 lever)

| Process | Source | Notebooks |
|---|---|---|
| Fixed K=mean(gold)=22 | UQLegalAI@COLIEE2025 baseline | rejected; fixed-cap K=5 documented as **F1 cap 0.296** |
| Oracle K (per-query best K from gold) | pipeline_iteration_latest_fixed_v1 | macro F1 **0.6970** at oracle adaptive K |
| Train-tuned K-scan | Untitled75 | K=5 chosen on train, F1=0.777 on val |
| Adaptive-k (largest score gap in sorted scores, buffer B=5, top 90 %) | EMNLP 2025 paper | final notebook Phase 3a |
| CAR (cluster (score, rank) pairs, cut at silhouette-best clustering's biggest gap, weighted by rank/N) | Nov 2025 paper | final notebook Phase 3b |
| PSI-Rank-Dyn (Qwen3-32B generates a borderline pseudo-doc; cut at its rerank score) | 2026 paper | final notebook Phase 3c |
| AcuRank-light (TrueSkill `Rating(mu=score, sigma=score/3)`; stop when uncertain pool < 10) | NeurIPS 2025 | final notebook Phase 3d |
| Two-Stage Risk Control / Conformal (use `train.csv` as calibration set, learn quantile τ_α for distribution-free 1-α recall guarantee) | IJCAI 2025 | final notebook Phase 3e |
| Aspect-coverage lower clamp (each `hyde_aspect` covered by ≥1 candidate p>0.5 + substantive role) | bespoke | final notebook Phase 3f |
| Median ensemble of 3a-3e, asymmetric clamp `max(K_aspect, n_hyde_aspects, 3)` upper `#{p > 0.05}` | bespoke | final notebook Phase 4 |
| GRPO over a 12-dim discrete policy (w_stat_overlap, w_channel_cov, w_cocite, prior_law, prior_court, α_rerank, α_judge, τ_cut, τ_yes, τ_no, k_base, k_per_target), LOO across 10 val queries | cascade_legalmalr_v1_base (60 steps) / _optimized (120 steps) / _poc | optimization wired; tested with Optuna fallback for noisy folds |

### 2.9 Diagnostics / utilities

| Process | Notebooks |
|---|---|
| Per-query stage-by-stage recall table (gold_in_corpus / rerank_pool / gold_in_rerank / judged_yes/no / auto_yes/no) | endgame_colab → diag_recall.py |
| Judge raw-response dissection (literal `VERDICT: YES` vs `<think>` truncation) | endgame_colab → diag_judge.py |
| Cache dump (rerank/judge/hyde distributions) | endgame_colab → diag_cache.py |
| Channel-alone recall audit (per-channel R@200 / R@1000 isolated) | endgame_colab, all anchor_funnel |
| F1 K-sweep over `[5, 7, 10, 13, 15, 20, 25, 30, 50, 75, 100, 35]` | endgame_colab, every pipeline_end_to_end_* |
| Sanity check (gold-pair vs non-gold-pair reranker score before full run) | pool_v75 canonical Stage 1 |
| Multi-engine PDF→text extraction (pdfminer LAParams, PyPDF2, pdftotext -layout) + data-driven engine picker on spacing/word-boundary | pdf_extraction_marker_pymupdf_pdfminer |

### 2.10 Things explicitly ruled out (so they're not retried)

From `endgame_handoff §10` and `FINAL_NOTEBOOK_RATIONALE §2`: UTF-8 mojibake repair, re-merge of court LLM enrichment into the unified store, FAISS-IVF, citation-graph re-extraction (now patched), bigger reranker / judge models, train-derived priors, fixed-cap K, dossier-only ranking with no fusion (R@300 = 0.013), Cohere reranker on legal text (LegalBench-RAG: hurts vs no-reranker), GRLStop (needs inference-time labels), Talos (needs retraining).

---

## Part 3 — Key results, by notebook family

### 3.1 The headline numbers

| Pipeline / experiment | Macro F1 | Macro R@K | Notebooks |
|---|---|---|---|
| Untitled75 reference (mini 65 k-doc index, train-tuned K=5, 5-query val) | **0.777** | R@30 0.16; R@1000 0.50 | reference_F1_0.777_Untitled75 |
| Endgame run (full corpus, all toggles ON, broken judge) | **0.0485** (P 0.033 / R 0.135) | rerank-pool recall **23.1 %** | endgame_colab_full_pipeline |
| pipeline_iteration_latest_fixed_v1 with **oracle adaptive K** | 0.6970 | enrichment prefilter pool recall ≥ 0.95 | pipeline_iteration_latest_fixed_v1 |
| pipeline_iteration_latest_base (no enrichment prefilter, binary judge, no HyDE) | ~0.45–0.52 | — | pipeline_iteration_latest_base |
| pipeline_iteration_latest_fixed_v3 (judge bugs closed) | ~0.60–0.65 | — | pipeline_iteration_latest_fixed_v3 |
| precision_v1_final cascade (Stage A→F over snapshot) | macro **fusion** F1 0.586; macro **cascade** F1 in saved JSON | Stage A fusion R@5000 = 0.728; Stage B Qwen3-Reranker-8B R@2000 = 0.618 | precision_v1_final |
| cascade_legalmalr_v1_base (Stage A reranker + Stage B Qwen3-8B judge + GRPO 60 steps, LOO) | baseline (pre-GRPO) **0.3516 ± 0.2899** | — | cascade_legalmalr_v1_base |
| pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL | retrieval only | **macro R@50k = 0.880** (range 0.745–1.0) — the warm-boot for all of folder 01; verified ceilings per query (val_003 binding at 0.766) | pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL |
| pool_v75_multiquery_hyde_test_no_lift | retrieval only | HyDE channel **no lift** over base v7.5; excluded | pool_v75_multiquery_hyde_test_no_lift |
| Macro oracle ceiling on v7.5 pool (every in-pool gold selected at K=in_pool_gold_count) | macro F1 **~0.944** | — | FINAL_NOTEBOOK_RATIONALE §1 |

### 3.2 Anchor funnel evolution (val_001 single-query R@1000 → multi-query macro R@1000)

| Version | Channels (cumulative) | val_001 R@1000 | macro R@1000 (10 val) | Notes |
|---|---|---|---|---|
| v1 base | statute, case, sibling, concept, term, bedrock | **0.119** | — | bedrock returns 0 articles |
| v4 | + law_direct_match, co_citation (unfiltered), bm25, vector_raw, vector_enriched | **0.238** | — | guarantee = law_direct + per_area |
| v5 | same 10 channels, but: embedding instruction-prefix fix, co-citation freq filter ≤5 k, per_area code-filter, court_statute LLM-only | **0.357** | — | reranker hurt → dropped |
| v6 | + statute_backprop (3rd guarantee channel), + `enhance()` corpus lexicon | 0.357 | **0.518** | multi-query loop opened |
| v7 | + graph_forward / reverse / 2hop, judgment_importance | 0.310 | 0.518 | graph_forward = 2nd-best channel on multi-query |
| v7.4 | + per-language BM25 (DE/FR/IT/EN), lemma index, budgets lifted | ~0.31 | 0.518 | deep-tail tuning |
| **v7.5 (canonical)** | code-family expansion (top-8), 7-channel guarantee 130 each | snapshot — | **R@50k = 0.880**; R@5000 = 0.728; R@2000 = 0.611 | per-query gold-in-pool ceiling 0.766 (val_003) — 1.000 (val_004/005) |

### 3.3 Early experiments (exp_A1–A6) gate sequence — all dense paths failed the 0.40 stat_recall@500 gate

| Notebook | stat_recall@500 |
|---|---|
| exp_A1 BGE-M3 baseline (EN / DE) | 0.215 / 0.168 |
| exp_A2 Qwen2.5-7B HyDE + Enum (RRF over en/de/hyde/enum) | best 0.295 |
| exp_A3 Qwen3-32B HyDE + Enum | best 0.376 |
| exp_A4 BGE-reranker on top-1000 | **hurts** (0.114 @50 vs 0.161 baseline) |
| exp_A4b BGE-reranker on HyDE-mean | 0.174 @50 — marginal |
| exp_A5 court BGE-M3 + statute mining (DE/FR/IT regex) | case_recall@200 = 0.108; stat_recall_mined = 0.060; combined = 0.349 — fail |
| exp_A6 legal-swiss-roberta fine-tune (SimCSE + triplet, 117 k pairs) | ft_rrf@500 = 0.101 — fail |
| early_dense_embedding_test (Qwen3-Embedding-8B enriched + 2.16 M corpus) | F1_oracle = 0.041, R@500 = 0.247 |
| early_segment_lattice_funnel_v3 (Qwen3-32B option selector) | oracle 0.396 / LLM 0.402; **LLM planner collapses court recall to 0.000** |

**Takeaway:** dense + reranker + enumeration alone tops out at ~0.38 R@500. This is the empirical justification for the structured anchor-funnel pivot.

### 3.4 Enrichment-coverage gap (the single largest live coverage hole)

| Corpus | Total rows | LLM-enriched rows | Coverage |
|---|---|---|---|
| Laws (DE) | 175,933 | **173,033** | ~100 % |
| Court considerations (DE/FR/IT) | 2,476,315 | **363,258** | ~15 % |

The 85 % court-enrichment gap is documented as deferred (endgame_handoff §11); the canonical v7.5 pool achieves R@50k = 0.88 anyway by leaning on graph + statute_backprop + per-area_bedrock + sibling_expansion.

### 3.5 The 4 documented judge bugs (resolved status)

1. System-prompt bias "When uncertain, say YES" — closed in v3.
2. Parser default-YES on parse failure — closed in v3.
3. `<think>` chain-of-thought consumes the token budget — closed by `enable_thinking=False`.
4. `max_new_tokens=200` truncates YES responses mid-thinking — closed by reducing to 24 with `VERDICT: YES|NO` constrained output.

### 3.6 Hard empirical ceilings (do not propose plans that ignore these)

- Train gold has a 28.5 % retrieval ceiling (71.5 % Class A + 20.9 % Class B + 7.6 % Class C). **Val gold is 100 % Class A** → calibrate on val only.
- Citation graph and gold co-prediction overlap is **0.59 %** → graph is OK as a tie-breaker, not as a recall expander.
- Qwen3-Embedding-8B alone on the full corpus caps at **R@1000 = 0.289** on val — dense cannot be the primary recall stage.
- Train→val distribution shift on 3 axes: language (99.4 % DE → 100 % EN), citation count (median 2 → 22), court share (1.2 % → 40.6 %) → **variable K mandatory**, court search mandatory, no train-derived priors.

---

## Part 4 — Where the work currently stands (2026-05-15)

- **Retrieval is solved.** v7.5 multi-query canonical pool achieves macro R@50k = 0.88 (oracle ceiling 0.944). The 50 k-doc snapshot is at `research/anchor_funnel_val001_v7/snapshot/` and feeds every folder-01 cascade.
- **Selection is the remaining problem.** Stage-1 reranker on the snapshot gives R@2000 ≈ 0.62 (precision_v1_final Stage B). The bottleneck is converting that into F1 0.6–0.8.
- **Final-notebook design is locked** (see `FINAL_NOTEBOOK_RATIONALE.md`): warm-boot from snapshot → Qwen3-Reranker-8B rerank top-5000 → 6 independent variable-K estimators (Adaptive-k + CAR + PSI-Rank + AcuRank + Conformal + Aspect-coverage) → median ensemble with asymmetric clamp → optional Qwen3-32B evidence-quote verifier → eval.
- **Confirmed open work item from endgame handoff**: enrichment prefilter is built (Cell 14) but never wired into `retrieve()` in `endgame_colab_full_pipeline.ipynb`. Folder 01 / pool_v75 work supersedes this lineage, so Move 1 from the handoff is effectively absorbed into the v7.5 channel architecture rather than patched directly.
- **Known deferred items**: extending court LLM enrichment from 363 k → 2.47 M (the 85 % gap), reranker law-family text-feed bug audit, MMR / diversity selection, multi-step agentic retrieval, submission to test.

---

_End of document. For deeper code-level detail on any notebook, follow the [_inventory/](_inventory/) link in [_inventory/INDEX.md](_inventory/INDEX.md)._
