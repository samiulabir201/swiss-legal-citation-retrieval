# Pre-v7.5 Pipelines, Embedding Base & High-Scoring Reference — Notebook Trace

Author note: this file traces 17 notebooks across subfolders `04_pre_v75_pipeline_iterations/`, `05_embedding_and_retrieval_base/`, `06_high_scoring_reference/`. The task spec asked for 18 (with `encode_queries_qwen3_8b.ipynb`), but that artifact only exists in the repo as the script `scripts/retrieval_and_rerank/encode_queries_qwen3_8b.py` — there is no matching `.ipynb`. Two reference-folder notebooks turned out to be misfiled (see flags under subfolder F). All paths are absolute. Sources: per-notebook inventories at `notebooks/_inventory/<canonical>.md`, the canonical `research/endgame_handoff_2026-05-09.md`, and `.claude/skills/swiss-citation-experiments/SKILL.md`.

---

## Subfolder 04 — pre-v7.5 pipeline iterations (13 notebooks)

This subfolder contains **three distinct architectural eras**, in chronological order:

1. **Apr 10 — `pipeline_no_debug_prints.ipynb`**: NOT a Swiss-citation notebook. Misfiled.
2. **Apr 12 — `pipeline_end_to_end_*` family (5 notebooks) + `pipeline_stage0_gated`**: a multi-stage (Stage 0 → Stage 5 + Stage 4b cross-encoder) pipeline rooted in `E:/swiss_law_pipeline_revised/` source. The "0.5400 → 0.6520 macro-F1" lift came from this family's Stage 4b cross-encoder upgrade.
3. **Apr 18 → Apr 20 — `pipeline_iteration_latest_*` family (5 notebooks)**: a re-architected Stage 0 → Stage 6 pipeline that reached **MACRO F1 = 0.7082** on val (best of any pre-v7.5 work in this folder).
4. **May 9 — `endgame_colab_full_pipeline.ipynb`**: the BIG ONE — 17-toggle Colab pipeline that catastrophically regressed to **F1 < 0.10** and produced `research/endgame_handoff_2026-05-09.md`.

### 04.0 `pipeline_no_debug_prints.ipynb` — Apr 10 — MISFILED

- **Path:** `e:/swiss_citation_extraction/notebooks/04_pre_v75_pipeline_iterations/pipeline_no_debug_prints.ipynb`
- **Inventory:** `e:/swiss_citation_extraction/notebooks/_inventory/pipeline_no_debug_prints.md`
- **Goal (actual):** AIMO-3 (AI Mathematical Olympiad Progress Prize 3) Kaggle inference server. Uses Nemotron-Nano (bf16, ~1×102 GB GPU) under vLLM 0.19, 32-attempt majority voting with stateful Python sandbox tool, scoring with entropy-weighted voting. Final-answer output is `\boxed{N}` with `N` in [0, 99999].
- **Data:** `/kaggle/input/competitions/ai-mathematical-olympiad-progress-prize-3/test.csv`. Zero relation to Swiss law data.
- **Best result:** N/A. The inventory's "End findings" auto-extraction confirms: "No metric-bearing lines auto-detected in last 5 cells."
- **Verdict:** Misfiled in this subfolder. Should not be considered part of the Swiss citation pipeline lineage. Possible cause: similar filename hygiene at upload time.

### 04.1 The `pipeline_end_to_end_*` family — Apr 12 (5 notebooks) + `pipeline_stage0_gated`

All five `pipeline_end_to_end_*` notebooks and `pipeline_stage0_gated_4ae3b3ffc3.ipynb` import paths from `data_paths.py` and run a **multi-stage Stage-0..Stage-5 + Stage-4b cross-encoder** pipeline that was developed in a separate repo at `E:/swiss_law_pipeline_revised/swiss-law-pipeline/`. They are functionally near-identical and form a progressive cleanup. Common features across all six:

- **Corpus:** `corpus.parquet` (171,654 law articles), `court_considerations.csv` (2.47M court rows), `laws_knowledge_base.jsonl`.
- **Stage 0:** BM25 (rank-bm25, ~171k docs), citation graph, citation lookup.
- **Stage 1-2-3:** query analysis → multi-agent retrieval (MAS) → graph expansion.
- **Stage 4:** LLM reranker (Qwen3-4B local backend).
- **Stage 4b:** Qwen3-Reranker-4B NF4 cross-encoder scoring top-500 candidates. **This stage is the documented `0.5400 → 0.6520` macro-F1 lift** (replaces binary llm_reranker/tier3 signals with continuous cross-encoder × 0.18 weight). Comment block in stage4b appears verbatim in all six.
- **Stage 5:** verify-and-score with threshold sweep + CE-weight sweep. The end-of-Stage-5 block prints `Best CE weight: <w>, macro-F1=<f1>` and `Best macro-F1: <f1>` but the inventories don't have the final printed F1 numbers in the captured outputs (only the printf format strings).

Progressive differences within the family (from mtime + cell counts):

| Notebook | Mtime | Cells | Distinctive change |
|---|---|---|---|
| `pipeline_end_to_end_base.ipynb` | Apr 12 12:37 | 24 | Earliest. `data_paths.py` central registry. Inline `BASE = "E:/swiss-law-pipeline-revised/swiss-law-pipeline"`. |
| `pipeline_end_to_end_native.ipynb` | Apr 12 12:46 | 37 | Adds inline native module bodies (no external imports). Stage 1–5 self-contained. |
| `pipeline_end_to_end_explicit_paths.ipynb` | Apr 12 12:52 | 39 | Same as `native` but with explicit absolute-path declarations per stage; +2 cells of path debugging. |
| `pipeline_end_to_end_clean_paths.ipynb` | Apr 12 12:57 | 37 | Cleanup: removes the explicit-path debug prints; canonicalizes path constants. |
| `pipeline_stage0_gated_4ae3b3ffc3.ipynb` | Apr 12 16:10 | 53 | Adds Stage 0 gating logic + KB-load failure guard (avoids the Sub 37 F1=0.058 regression). |
| `pipeline_end_to_end_no_function_imports.ipynb` | Apr 12 16:31 | 53 | Final cleanup; removes all `from X import Y` for swiss_law_pipeline modules — every function pasted in-cell so notebook runs without the parent repo. |

**Best result observed (per inventory grep):** the macro-F1 numbers shown in all six are computed but not visibly printed in captured outputs. The inline doc references **`0.5400 → 0.6520`** as the Stage 4b uplift. The notebooks' own evaluation cells call `Best macro-F1: {best_f1:.4f}` but the corresponding output is not captured in the inventory dump for this family.

**Verdict:** progressive cleanup track from a separately-developed pipeline. The clearest "shipping form" is `pipeline_end_to_end_no_function_imports.ipynb` (last mtime in the family, no external imports). All superseded by the `pipeline_iteration_latest_*` family one week later, then again by the `endgame_colab_full_pipeline.ipynb`. None retained for current direction.

### 04.2 The `pipeline_iteration_latest_*` family — Apr 18 → Apr 20 (5 notebooks)

This is a **re-architected 6-stage pipeline** (Stage 0 → Stage 6 adaptive-K) using BM25 + cross-encoder + Qwen3-32B (bf16, no quantization) as ranker. Adds `STAGE6_K_STRATEGY="score_gap"` adaptive-K selection. **Verifiable best macro-F1 = 0.7082** on val from this lineage. Chronological order:

| Notebook | Mtime | Cells | Verdict / measured F1 |
|---|---|---|---|
| `pipeline_iteration_latest_base.ipynb` | Apr 18 13:38 | 80 | Baseline of the new architecture. Comment: `0.5400 → 0.6520 macro-F1` carries forward from the end_to_end family. No final F1 printed. |
| `pipeline_iteration_latest_with_all_output.ipynb` | Apr 18 17:57 | 86 | Same notebook but with all output cells captured. **Final printed result: MACRO F1 (adaptive-K) = 0.7082, P=0.8962, R=0.6047, mean preds/query=16.3.** Per-query F1: val_001=0.895, val_002=0.820, val_003=0.537, val_004=0.522, val_005=0.667, val_006=0.643, val_007=0.690, val_008=0.711, val_009=0.880, val_010=0.718. Diagnostic: `BM25 oracle-K macro-F1 = 0.6023`, `Stage-5 oracle-K macro-F1 = 0.5456`. Score source: `bm25_plus_llm` (w_llm=0.5). |
| `pipeline_iteration_latest_fixed_v1.ipynb` | Apr 20 01:03 | 76 | (Inventory dump shows only printf format strings — no final F1 captured. F1=0.058 KB-load failure guard added.) |
| `pipeline_iteration_latest_fixed_v2.ipynb` | Apr 20 02:49 | 76 | **Printed: MACRO F1 (adaptive-K) = 0.6777, P=0.8831, R=0.5655.** Per-query F1 lower than `with_all_output` on val_001 (0.727 vs 0.895) and val_008 (0.682 vs 0.711); other queries match. BM25 oracle = 0.6786, Stage-5 oracle = 0.5168. |
| `pipeline_iteration_latest_fixed_v3.ipynb` | Apr 20 21:20 | 76 | **Printed: MACRO F1 (adaptive-K) = 0.6970, P=0.8713, R=0.5967.** Per-query: val_001=0.895, val_002=0.820, val_003=0.571, val_004=0.522, val_005=0.571, val_006=0.643, val_007=0.690, val_008=0.711, val_009=0.833, val_010=0.714. BM25 oracle = 0.6767, Stage-5 oracle = 0.5232. Adds HF cache path setup before `from transformers import`. `STAGE3_BUILD_STATUTE_TO_BGE=True`, `STAGE6_NUMBERED_REQUIRE_GRAPH_SOURCE=False` (reverted from v2 — the gate had rejected gold Numbered like 7B_xxx/yyyy that only exist via text-mined indexes). Pre-diag panel reports val codes: gold = `{ZGB:39, StPO:36, OR:18, BGG:13, StGB:12, IVG:10, ATSG:7, ZPO:4, BV:3, IPRG:2, SchKG:1}`. |

**Key per-query gold-in-pool / gold-in-BM25 stats (from `fixed_v2`/`fixed_v3` diagnostic):**

| qid | gold | in_pool | in_bm25 | med_rank | max_rank |
|---|---|---|---|---|---|
| val_001 | 42 | 42 | 35 | 18 | 51 |
| val_002 | 36 | 34 | 29 | 15 | 84 |
| val_003 | 47 | 33 | 21 | 11 | 61 |
| val_004 | 10 | 10 | 6 | 3.5 | 55 |
| val_005 | 11 | 10 | 8 | 4.5 | 56 |
| val_006 | 18 | 16 | 11 | 6 | 65 |
| val_007 | 19 | 17 | 11 | 6 | 11 |
| val_008 | 29 | 26 | 18 | 9.5 | 51 |
| val_009 | 14 | 14 | 12 | 6.5 | 51 |
| val_010 | 25 | 21 | 16 | 8.5 | 50 |

**Verdict for the iteration_latest family:** The `with_all_output` 0.7082 is the highest observed F1 in this entire folder, but it relies on `bm25_plus_llm` score with w_llm=0.5 ("best on val" per inline comment). All retired when the team pivoted to the endgame Colab pipeline two weeks later.

### 04.3 `endgame_colab_full_pipeline.ipynb` — May 9 (CANONICAL FAILURE — F1 < 0.10)

- **Path:** `e:/swiss_citation_extraction/notebooks/04_pre_v75_pipeline_iterations/endgame_colab_full_pipeline.ipynb`
- **Inventory:** `e:/swiss_citation_extraction/notebooks/_inventory/endgame_colab_full_pipeline.md` (26 code cells, 2528 lines)
- **Goal:** unified end-to-end pipeline on the full **2,652,248-row** unified corpus (175,933 law + 2,476,315 court) — vs the earlier `corpus.parquet` 171k. All 17 toggles ON except `use_agentic_retrieval`. Output: per-query predictions + adaptive-K. Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition (95.6 GB VRAM).
- **Stack:**
  - **Embeddings:** Qwen/Qwen3-Embedding-8B, 4096-dim, fp16, 27 chunks at `embeddings/qwen3_8b_unified_chunk*.npy` + `qwen3_8b_unified_manifest.parquet`.
  - **Reranker:** Qwen/Qwen3-Reranker-8B (`rerank_top_n=200`, `rerank_alpha_retrieval=0.7`, `rerank_alpha_rerank=0.3`, `max_seq_len=4096`).
  - **Judge:** Qwen/Qwen3-8B with 7-category zone routing (`judge_auto_yes=0.55`, `judge_auto_no=0.25`, `max_seq_len=8192`).
  - **Channels (all toggled ON):** enrichment prefilter (BUILT BUT NOT WIRED into `retrieve()` — the #1 root cause), BM25 (`channel_budget=800`, lexicon expansion), vector (`budget=800`, brute-force GPU `E_GPU @ q`), HyDE (`max_new_tokens=220`, T=0.3), German keyword expansion, citation-graph 1-hop, legal-area soft-filter, statute anchors, case anchors, court-base sibling expansion, authority-score boost (`alpha=0.15`), granularity resolver, dynamic-K calibration.
  - **Fusion:** RRF (`rrf_k=60`) with per-channel weights `{bm25:1.0, vector:0.9, hyde:0.6, statute:0.8, case:0.7, graph:0.4, court_base:0.4}`.
  - **F1 K-sweep:** `[5, 7, 10, 13, 15, 20, 25, 30, 50, 75, 100]`.
- **Data:** Drive root `/content/drive/MyDrive/swiss_law`. `train.csv` (1139), `val.csv` (10), `test.csv` (40), `laws_de.csv`, `court_considerations.csv`. SQLite `artifacts_v2/unified_retrieval.sqlite` (~24 GB, FTS5 + `statute_links` 5.30M + `case_links` 1.81M + `adjacent_law_links` 348k).
- **Best measured result (val, 10 queries):**
  - **Macro F1: < 0.10** (target 0.6–0.8). Per-toggle ablation grid (from inventory):

    | Ablation | F1 | delta |
    |---|---|---|
    | Baseline (all toggles ON) | **0.0485** | — |
    | Disable enrichment_prefilter | 0.0450 | −0.0035 |
    | Disable BM25 | 0.0544 | +0.0060 |
    | Disable BM25_lexicon_expansion | 0.0530 | +0.0046 |
    | Disable vector | 0.0542 | +0.0057 |
    | Disable HyDE | 0.0470 | −0.0015 |
    | Disable query_german_expansion | 0.0470 | −0.0015 |
    | Disable citation_graph_expansion | 0.0485 | +0.0000 |
    | Disable legal_area_softfilter | 0.0459 | −0.0026 |
    | Disable statute_anchors | 0.0485 | +0.0000 |
    | Disable case_anchors | 0.0465 | −0.0020 |
    | Disable court_base_sibling_expansion | 0.0459 | −0.0025 |
  - **Avg pool size:** 471 candidates per query.
  - **Gold in rerank pool:** **58/251 = 23.1%** (the binding ceiling).
  - **Judge YES rate:** 1546/1774 = 87% (rubber-stamp).
  - **Auto-yes / auto-no zone fires:** 0 / 0 (thresholds in wrong scale — RRF is 0–0.1, thresholds were 0.25/0.55).
  - **Per-query rerank pool / gold-in-rerank:** val_001 (512/11), val_002 (427/5), val_003 (529/4), val_004 (426/6), val_005 (481/7), val_006 (399/4), val_007 (419/8), val_008 (524/4), val_009 (509/4), val_010 (480/5).

- **Documented root causes (`research/endgame_handoff_2026-05-09.md`):**
  1. **Retrieval recall ceiling (binding):** pool R = 23.1%. The enrichment prefilter (Cell 14) was defined but never invoked inside `retrieve()` — the user's specifically-asked-for architectural piece was dead code. This is the #1 fix item.
  2. **Judge `<think>` token-budget blowout (4 compounding bugs in Cell 16):** (a) system prompt biased toward YES — "When uncertain, say YES — it is better to include a marginally relevant article than to miss one"; (b) parser defaults to `verdict="yes"` on parse failure; (c) Qwen3-8B emits `<think>` chain-of-thought by default, consuming budget before reaching `VERDICT:`; (d) `max_new_tokens=200` truncates 78/115 of query-487b's YES responses mid-thinking. Only 37/115 YES responses had a literal `VERDICT: YES` line; 78 had `<think>` but no VERDICT.
  3. **Reranker family bias:** top-5 of val_001 are all `court:*` with score 0.97–0.98; bottom-5 are all `law:*` with score 0.0000. Suspected: law candidates have empty/wrong text field fed to the reranker, OR Qwen3-Reranker prompt format mismatches law text. Secondary to recall.
  4. **Zone thresholds at wrong scale:** `judge_auto_yes=0.55` / `judge_auto_no=0.25` were calibrated for Untitled75's `0.7*minmax(retrieval) + 0.3*rerank` scale (0–1). The endgame uses raw RRF scores (0.01–0.1). Every candidate fell into the "judge it" zone.
- **Cache state (preserved in `cache_endgame/`):** `hyde_cache.json` (10 entries, REUSABLE), `german_expansion_cache.json` (10 entries, REUSABLE), `query_embeddings.npy` (10 vectors × 4096, REUSABLE — this is the val-query embedding artifact normally attributed to a separate `encode_queries_qwen3_8b` step), `rerank/<sha1>.json` × 10 (REUSABLE), `judge/<sha1>/<sha1>.json` × 1788 (TOXIC — must delete before any re-run).
- **Verdict (per `swiss-citation-experiments` skill, "Endgame baseline 2026-05-09"):** **REJECTED.** The whole pre-v7.5 pipeline was abandoned in favor of the v7.5 multi-query funnel (R@50k=0.893). Caches preserved for forensics; judge cache must be deleted before any future run. The "dossier-plan" Phase 1 enrichments (`research/cascade_dossier_plan.md`) explicitly inherit fixes for this notebook's structural problems.

### 04-subfolder summary

- **Definitive single notebook:** `endgame_colab_full_pipeline.ipynb` is the canonical pre-v7.5 endgame artifact. The v7.5 multi-query funnel (in subfolder 02) and the cascade/dossier work (subfolder 01) are direct responses to its failure modes.
- **Best historical F1 in this folder:** `pipeline_iteration_latest_with_all_output.ipynb` at **F1=0.7082** on val (Apr 18). That was on the 171k-row `corpus.parquet`, not the 2.65M unified corpus.
- **Progression:** end_to_end (Apr 12) → iteration_latest (Apr 18-20) → endgame (May 9). Each rebuild added more capacity but the May 9 endgame regressed because (a) the prefilter wasn't wired and (b) the judge's compounding bugs poisoned every prediction.
- **Misfiled artifact:** `pipeline_no_debug_prints.ipynb` is an AIMO-3 math notebook — exclude from any swiss-citation lineage analysis.

---

## Subfolder 05 — Embedding & retrieval base (3 notebooks)

### 05.1 `embed_unified_corpus_qwen3_8b_blackwell.ipynb` — PRODUCER OF THE fp16 CHUNKS

- **Path:** `e:/swiss_citation_extraction/notebooks/05_embedding_and_retrieval_base/embed_unified_corpus_qwen3_8b_blackwell.ipynb`
- **Inventory:** `e:/swiss_citation_extraction/notebooks/_inventory/embed_unified_corpus_qwen3_8b_blackwell.md`
- **Mtime:** 2026-05-07 20:37, 12 code cells.
- **Goal:** encode the full 2.65M-row unified corpus (court + law) with Qwen3-Embedding-8B, output 27 fp16 chunks + manifest. **This is the producer of the canonical 21 GB fp16 embeddings** referenced everywhere downstream.
- **Data:** `/content/drive/MyDrive/swiss_law/data/unified_embedding_input.parquet` — 2,652,248 rows × `[doc_id, family, citation, vector_text, char_len]`. Family counts `{court: 2,476,315, law: 175,933}`. Char stats: avg=373, p50=309, p90=612, p99=1414, max=2845.
- **Approach:**
  - **Model:** `Qwen/Qwen3-Embedding-8B`, bf16, MODEL_DIM=4096, MAX_SEQ_LEN=768 (covers >99% of chars).
  - **Attention backend:** auto-probe chain `flash_attention_4 → flash_attention_2 → sdpa`. On Blackwell sm_120, flash_attn-4 was importable but the transformers registry didn't accept it; final selection was `sdpa` with cuDNN flash + mem-efficient kernels enabled.
  - **Hardware:** RTX PRO 6000 Blackwell Server Edition, 102.0 GB VRAM, Torch 2.10.0+cu128, CUDA 12.8.
  - **Batching:** `BATCH_SIZE=256`, `CHUNK_SIZE=100_000`, `OUTPUT_DTYPE=float16`. Length-sorted batching for throughput; original order restored via permutation index. `USE_TORCH_COMPILE=False` (compile crashed on Blackwell).
  - **Normalization:** L2-normalized embeddings (norm_mean=1.0013 ± 0.0017 across all 27 chunks).
  - **Output:** 27 chunks at `/content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_chunk000..026.npy` (chunk000-025 each 100k×4096 fp16, chunk026 52,248×4096). Manifest at `qwen3_8b_unified_manifest.parquet` (2,652,248 rows × `[doc_id, family, citation, row_index]`). Summary at `qwen3_8b_unified_summary.json`.
  - **Sanity smoke retrieval (chunk000 only, English queries):**

    | Query | Top-1 doc | Score |
    |---|---|---|
    | "When can Swiss courts extend pretrial detention based on collusion risk?" | court 1B_227/2017 E. C | 0.780 |
    | "What principles bind persons performing public tasks under Swiss public law?" | court 2C_1023/2021 E. 2.2.1 | 0.675 |
    | "Dublin transfer detention proportionality test" | court 2C_199/2018 E. 4.2 | 0.735 |
  - Query prompt template: `"Instruct: Given an English-language legal question or scenario about Swiss federal law, retrieve the Swiss statute articles or federal court decision considerations that are most directly relevant to answering it.\nQuery: "`.
- **Best result:** N/A (utility notebook, not a retrieval experiment). Smoke retrieval shows model works cross-lingually (EN query → DE court docs).
- **Verdict:** **kept — canonical embedding artifact.** Endgame handoff §6.6 declares these the "canonical set" and the previous 36 GB fp32 stale chunks were deleted. All v7.5 pool and downstream rerank work consumes these chunks.

### 05.2 `retrieve_unified_corpus_qwen3_8b.ipynb` / 05.3 `retrieve_unified_corpus_v3.ipynb` — TWIN HYBRID RETRIEVERS

- **Paths:**
  - `e:/swiss_citation_extraction/notebooks/05_embedding_and_retrieval_base/retrieve_unified_corpus_qwen3_8b.ipynb` (mtime 2026-05-08 01:01)
  - `e:/swiss_citation_extraction/notebooks/05_embedding_and_retrieval_base/retrieve_unified_corpus_v3.ipynb` (mtime 2026-05-08 21:11)
- **Inventories:** `retrieve_unified_corpus_qwen3_8b.md` and `retrieve_unified_corpus_v3.md` are **byte-for-byte identical through Step 12** and produce **identical output numbers**.
- **Goal:** hybrid 9-channel retrieval over the 2.65M-row unified corpus; measure R@K up to K=10000 on val.
- **Approach:**
  - **9 channels (RRF-fused, `rrf_k=60`, `authority_alpha=0.20`):**

    | Channel | Budget | Weight | Description |
    |---|---|---|---|
    | vector | 5000 | 1.5 | Qwen3-8B brute-force GPU `E_GPU @ q` |
    | law_card_direct | 400 | 2.5 | Direct law-card citation match |
    | statute | 800 | 1.0 | `statute_links` table lookup |
    | case | 600 | 1.4 | `case_links` table lookup |
    | statute_co_occurrence | 800 | 1.6 | Docs that co-cite our statute |
    | citation_graph | 1500 | 1.2 | Graph 1-hop neighbors (CG_PER_DOC=8 from 500 vector seeds) |
    | court_base_x | 1500 | 0.9 | Court-base sibling expansion (CB_PER_BASE=30 from 1000 vector seeds) |
    | adjacent_law | 400 | 0.7 | `adjacent_law_links` table |
    | same_law_code | 300 | 0.5 | Same law-code group |
  - **Citation normalization:** alias map combines hardcoded `law_code_aliases.json` (126 forms) + every law-code in the corpus (2063 forms) → 2147 known forms. Cross-language aliases collapse to SC numbers: e.g. `StPO`/`CPP` → `312.0`, `BGG` → `173.110`. Suffix stripper drops `Abs/Bst/Lit/Ziff/Ch/Cpv` for statute-side fairness.
  - **Indexes built:** court_base_groups (178,593), unique normalized statutes (171,498), law_cit_to_docs, law_code_to_docs, case_to_docs, case_base_to_docs, neighbor_to_docs, adj_links 347,740, statute_links 5,295,519, case_links 1,813,789.
  - **Model load:** `Qwen/Qwen3-Embedding-8B` via sentence-transformers, bf16. Model load: 54.9s, VRAM 36.86 GB. Encoded 10 val queries in 5.4s → saved `qwen3_8b_query_val.npy` (10, 4096) — **this is the val-query embedding artifact referenced as `cache_endgame/query_embeddings.npy`**.
- **Best measured result on val (10 queries, top_k=10000):**

  | K | macro R@K |
  |---|---|
  | 50 | 0.1696 |
  | 100 | 0.2038 |
  | 200 | 0.2535 |
  | 500 | 0.3673 |
  | 1000 | 0.4013 |
  | 2000 | 0.4468 |
  | 5000 | 0.5117 |
  | 10000 | **0.5307** |

  **Per-query R@10000 (worst-first):** val_008=0.107, val_003=0.200, val_010=0.435, val_002=0.471, val_009=0.500, val_001=0.821; others not shown explicitly. val_003=0.022 at R@200, val_001=0.103 at R@200 (the same gold-anchor problem that later v7.5 multi-query solves).
- **Most-frequently missed gold (from dropped-gold CSV head):** several BGE-formatted citations and 9C-series dockets (e.g. `BGE 144 V 427 E. 3.2`, `BGE 132 V 93 E. 4`, `BGE 148 V 21 E. 5.3`, `9C_623/2020 E. 4.2`, `BGE 139 V 399 E. 5.3`, `BGE 135 V 39 E. 6.1`).
- **Channel totals (over 10 queries):** vector 50k, law_card_direct 18, statute 2147, statute_co_occurrence 1466, citation_graph 15k, court_base_x 15k, same_law_code 850, case 0 (no val query had a parsed case anchor). N statutes parsed per query: val_001/val_002=1, others=0.
- **Verdict:** **superseded but useful baseline.** R@10k=0.5307 was the pre-funnel ceiling; the v7.5 multi-query funnel (with structured LLM expansion + the enrichment prefilter fix) lifts the same pool to R@50k=0.893. These twin notebooks demonstrate the structural limit of single-query 9-channel retrieval and explain why structured query expansion was the next move. The two notebooks (`qwen3_8b` and `v3`) are duplicates; `v3` is 20 hours later but has the same outputs, suggesting it was a rerun with no functional change.

### 05-subfolder summary

- **Definitive embedder:** `embed_unified_corpus_qwen3_8b_blackwell.ipynb` is the **producer of all canonical fp16 chunks** used downstream. Re-running it would re-produce the same artifacts.
- **Definitive query encoder:** the val-query embeddings (`cache_endgame/query_embeddings.npy`) were produced by step 10 of `retrieve_unified_corpus_qwen3_8b.ipynb` (cell loads Qwen3-Embedding-8B, encodes 10 val queries, saves `qwen3_8b_query_val.npy`). There is no standalone `encode_queries_qwen3_8b.ipynb` in the repo — only `scripts/retrieval_and_rerank/encode_queries_qwen3_8b.py` and this notebook's cell 10.
- **Definitive baseline retrieval recall:** R@10000 = **0.5307** macro on val. This is the ceiling that motivated v7.5 multi-query.
- **Twin notebooks:** `retrieve_unified_corpus_qwen3_8b.ipynb` and `retrieve_unified_corpus_v3.ipynb` are byte-identical in code and output — keep only one for future work.

---

## Subfolder 06 — High-scoring reference notebooks (2 notebooks)

### 06.1 `reference_F1_0.777_Untitled75.ipynb` — THE F1=0.777 HOLY GRAIL (partial)

- **Path:** `e:/swiss_citation_extraction/notebooks/06_high_scoring_reference/reference_F1_0.777_Untitled75.ipynb`
- **Inventory:** `e:/swiss_citation_extraction/notebooks/_inventory/reference_F1_0.777_Untitled75.md` (7 code cells, 2530 lines)
- **Mtime:** 2026-05-08 12:15
- **Status flag:** the file shipped under this name contains **three distinct scripts in one notebook**, only the third of which is the F1-relevant pipeline. The "F1=0.777" figure cited everywhere (problem_statement.md, endgame_handoff §1) is for the val-tuned version of this third script. In the captured run inside the inventory the train-tuned variant prints **F1=0.3364** at K=5 (on a 100-query train sample, not val). The endgame handoff describes "F1=0.777 on val from `research/Untitled75.ipynb` — but pre-tuned on this val set with smaller corpus".
- **What's actually in the cells:**
  - **Cells 1–2:** `pip install faiss-gpu-cu12` + `sacremoses`.
  - **Cell 3 — `option_a_translate_and_embed.py`:** Translates 171,654 DE article texts to EN with MarianMT (`Helsinki-NLP/opus-mt-de-en`, batch=512, beam=2), rebuilds bilingual `embed_text` (English block first, then German), re-embeds with **Qwen3-Embedding-4B** (NOT 8B), dim=2560, max_len=1024, batch=64 on A100 80GB, FAISS IndexFlatIP (171,654 vectors, 1757.7 MB). Val recall (with `Instruct: Given a legal fact pattern, retrieve the most relevant Swiss statutory provisions...` template): **R@30=0.160, R@100=0.291, R@300=0.380, R@1500=0.512 macro** — WORSE than the old German-only baseline (R@30=0.325, R@1500=0.564). BM25 baseline reference cited inline: R@30=0.845, R@1500=0.981.
  - **Cell 4 — `test_better_translation.py`:** Mini-index test (gold for 5 worst val queries + 1000 distractors) comparing German-only / MarianMT / NLLB-200-3.3B bilingual. NLLB is the high-quality translation candidate. Result inline: "Option C (full-index): WORSE than old (delta=−0.026)".
  - **Cells 5–7 — the F1=0.777 lineage** (compressed-syntax, single-letter variable code by a different author):
    - **Loader:** corpus.parquet (171,654), train.csv (1139), test.csv (40), translations (1189), laws_de.csv (179,641), pre-built BM25 (171,654).
    - **`enhance()` lexicon expansion:** token-level — for each query token, score every law-abbrev with `(count/total_in_token) × idf`, multiply each of the top-5 articles' tokens by 5 (oversample). This is the Untitled75 token-level lexicon expansion referenced in `endgame_handoff §3.1` and §5.
    - **`retr()`:** BM25 top-`TOP_BM25=30` over the enhanced token list. Per candidate, populate `Cand(cn, rank, bm25, title, text, law, law_en, heading, ptype)`.
    - **`rerank()`:** **Qwen3-Reranker-8B** as `AutoModelForCausalLM` (NOT a sequence-classification head). Prompt:
      ```
      <|im_start|>system
      Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
      <|im_start|>user
      <Instruct>: {RR_INST}
      <Query>: {q}
      <Document>: {doc}
      <|im_end|>
      <|im_start|>assistant
      <think>

      </think>

      ```
      Document prompt template: `"{cn}\nLaw: {law} — {law_en}\nTitle: {title}\nHeading: {heading}\nType: {ptype}\nText: {text[:500]}"`. Scoring: `log_softmax([no_logit, yes_logit])[..., 1].exp()`. Batch=8, max_len=4096. Cache layered across suffixes `_v3, _final, _v2, _v12, _v7, _v6b, _v5b, _v5, _full_v4, ""`.
    - **Fusion (canonical 7/3 blend):** `c.fused = 0.7 * minmax(c.bm25) + 0.3 * c.rr`, then sort descending.
    - **K-scan F1 on train sample (100 queries, K=5,10,15,20,25,30):** P=0.357/0.224/0.160/0.124/0.100/0.084, R=0.496/0.583/0.605/0.616/0.619/0.620, **F1 best = 0.3364 at K=5** (on this 100-query train sample). For test, generates `submission_K{5,10,15,20,25,30}.csv` at the recommended K=5.
- **Best result (canonical):** **val Macro F1 = 0.777** (per endgame_handoff and problem_statement.md), produced by a val-tuned version of the same pipeline. The exact tuning that lifted train-F1 0.336 → val-F1 0.777 is **the K-scan was tuned on val instead of train**, exploiting the smaller corpus and the 10-query val set's specific gold distribution.
- **Verdict:** **reference-only.** Recipe: BM25 (top-30 with `enhance()` token-level lexicon expansion) + Qwen3-Reranker-8B yes/no LLM-prompted scoring + 0.7-minmax(bm25) + 0.3-rerank fused + K-scan tuned on val. This is **what the v7.5 + cascade-dossier track is trying to beat structurally** (not by training on val). The dossier plan attacks the underlying assumption that the small val K-scan + 30-candidate window can be generalized.

### 06.2 `reference_F1_0.7716_scored.ipynb` — MISFILED

- **Path:** `e:/swiss_citation_extraction/notebooks/06_high_scoring_reference/reference_F1_0.7716_scored.ipynb`
- **Inventory:** `e:/swiss_citation_extraction/notebooks/_inventory/reference_F1_0.7716_scored.md` (9 code cells, 340 lines)
- **Mtime:** 2026-04-29 20:38
- **Status flag:** **NOT a Swiss legal citation notebook.** This is a Kaggle "retroviral-challenge-predict" notebook — protein/reverse-transcriptase activity prediction.
- **Goal (actual):** predict reverse-transcriptase prime-editing efficiency from biophysical features + ESM-2 embeddings.
- **Approach:**
  - Inputs: `/kaggle/input/competitions/retroviral-challenge-predict/{train.csv, test.csv, esm2_embeddings.npz}` (57 train × 71 cols, 57 test × 69 cols).
  - Features: 11 biophysical (`foldseek_best_TM, foldseek_best_fident, foldseek_best_LDDT, triad_best_rmsd, D1_D2_dist, pocket_hydrophobic_per_res, camsol, perplexity, instability_index, net_charge, protein_length_aa`) + 5 PCA components of ESM-2 (1280 → 5 dims via PCA) + 2 engineered (`is_broken_enzyme`, `length_to_TM_ratio`).
  - Hurdle models (XGBoost + CatBoost, both `n_estimators=100, max_depth=1, lr=0.05`): classify active (`y_bin`) → if active, regress efficiency (`y_eff`) → multiply → ensemble average of XGB and CatBoost.
  - Cross-validation: leave-one-family-out on `rt_family`.
  - Metric: `CLS = 2*PR_AUC*W_Spearman / (PR_AUC + W_Spearman)` (weighted Spearman with weights `pe_efficiency_pct + 0.01`).
- **Best result:** **OOF CLS = 0.5874** (PR-AUC=0.6432, W-Spearman=0.5405). The "0.7716" in the filename does **not** appear anywhere in the printed outputs of the captured run. The "F1" in the filename is also a misnomer — the score is a custom CLS, not F1.
- **Verdict:** misfiled — exclude from any swiss-citation lineage analysis. Likely uploaded by accident when reorganizing high-scoring Kaggle artifacts.

### 06-subfolder summary

- **Definitive reference:** `reference_F1_0.777_Untitled75.ipynb` cells 5–7 (the BM25-top30 → Qwen3-Reranker-8B → 0.7/0.3 fuse → val-tuned K-scan recipe). The other content in the same `.ipynb` (cells 1–4: translation/embedding experiments with Qwen3-Embedding-4B) does NOT produce the 0.777 figure; those measure recall, not F1.
- **What we're trying to recreate (structurally):** the cascade-dossier plan in `research/cascade_dossier_plan.md` is the principled successor to the 0.7-bm25 + 0.3-rerank heuristic, designed so the lift is not dependent on val-tuning the K-scan.
- **Misfiled artifact:** `reference_F1_0.7716_scored.ipynb` is a retroviral biology notebook — exclude.

---

## Cross-folder summary

### What was achieved before v7.5 (best of each lineage)

| Lineage | Best val F1 / R | Provenance | Status |
|---|---|---|---|
| Untitled75 reference (BM25+Qwen3-Rerank-8B+0.7/0.3 fuse, val K-tuned) | F1 = **0.777** | `research/Untitled75.ipynb` + `nb_dump.txt` | reference-only; the target to beat structurally |
| `pipeline_iteration_latest_with_all_output` (Stage 0-6 adaptive-K) | F1 = **0.7082**, P=0.8962, R=0.6047 | Apr 18 17:57 | superseded by endgame which failed |
| `pipeline_iteration_latest_fixed_v3` | F1 = **0.6970** | Apr 20 21:20 | superseded |
| `pipeline_end_to_end_*` family (Stage 4b CE upgrade) | F1 lift 0.5400 → 0.6520 | Apr 12 | superseded |
| `retrieve_unified_corpus_qwen3_8b` (9-channel hybrid, no rerank, no judge) | R@10000 = **0.5307**, R@200 = 0.2535 | May 8 | baseline for v7.5 multi-query (now R@50k=0.893) |
| `endgame_colab_full_pipeline` | F1 = **< 0.10**, gold-in-pool = 23.1% | May 9 (CANONICAL FAILURE) | rejected (`endgame_handoff §0`) |

### What was kept

- The 27 fp16 corpus embeddings at `embeddings/qwen3_8b_unified_chunk000..026.npy` (producer: `embed_unified_corpus_qwen3_8b_blackwell.ipynb`) — canonical.
- The val-query embeddings at `cache_endgame/query_embeddings.npy` (producer: `retrieve_unified_corpus_qwen3_8b.ipynb` cell 10) — REUSABLE.
- HyDE cache (10 entries) and German expansion cache (10 entries) at `cache_endgame/*.json` — REUSABLE.
- Rerank cache `cache_endgame/rerank/<sha1>.json` × 10 — REUSABLE.
- Citation-graph DB at `data_insights/citation_graph_extracted.sqlite` (1.99 GB; gold-coverage 92.9% → 96.0%) — used as tie-breaker only (graph is NOT a co-prediction signal — 99.41% of gold-citation pairs have no edge, per Obs 2).

### What was rejected

- `endgame_colab_full_pipeline.ipynb` and its judge cache `cache_endgame/judge/` (TOXIC — must delete before re-run).
- The all-five `pipeline_end_to_end_*` family (Apr 12) and all-five `pipeline_iteration_latest_*` family (Apr 18-20): superseded by v7.5 multi-query funnel; not part of current direction.
- The 36 GB fp32 embedding chunks (deleted; fp16 chunks are canonical — `endgame_handoff §6.6`).

### Misfiled artifacts (to be re-classified)

- `notebooks/04_pre_v75_pipeline_iterations/pipeline_no_debug_prints.ipynb` — AIMO-3 Nemotron-Nano math notebook.
- `notebooks/06_high_scoring_reference/reference_F1_0.7716_scored.ipynb` — retroviral RT activity prediction notebook (Kaggle "retroviral-challenge-predict").

### Path to current direction

The lineage from this folder set into the current direction (subfolder 01 cascade/rerank/precision) goes:

1. **embed_unified_corpus_qwen3_8b_blackwell** → produces canonical fp16 chunks.
2. **retrieve_unified_corpus_qwen3_8b** → 9-channel hybrid baseline (R@10k=0.5307).
3. **endgame_colab_full_pipeline** → adds HyDE / judge / reranker — **fails at F1<0.10** because the enrichment prefilter is dead code and the judge is broken.
4. **v7.5 multi-query funnel** (subfolder 02) → wires the prefilter, adds structured LLM expansion → R@50k=0.893 canonical pool.
5. **rerank_hybrid_3model_shootout** (subfolder 01, May 15) → uses the v7.5 pool, measures Qwen3-Reranker-8B (R@2k=0.247) vs BGE-v2-m3 (0.073, rejected) vs jina-v2-base (0.234, kept-candidate). F3 hybrid (3-rerank + 9 dossier signals) at R@200=0.304 — still structurally blocked by val_003 R_max=0.766 in the pool.
6. **Cascade dossier plan** (`research/cascade_dossier_plan.md`) — Phase 1: channel-of-arrival fingerprint + statute-target intersection + co-citation density. Open.

The val_003 R_max=0.766 ceiling in the v7.5 pool, observed first as `val_003: in_pool=33 / gold=47 → 0.702` in this folder's `pipeline_iteration_latest_with_all_output`, is the same ceiling that still binds today. No work in this folder solved it — that's what the dossier plan and reranker fine-tuning proposals (subfolder 01) target.
