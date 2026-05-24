# 03 — Enrichment Notebooks Analysis

Scope: `notebooks/07_law_de_enrichment/`, `notebooks/08_court_llm_descriptor_extraction/`, `notebooks/09_court_citation_concept_enrichment/`, `notebooks/10_authority_card_enrichment/`.

Source data:
- `notebooks/_inventory/<canonical>.md` for each notebook (run-time outputs preserved by the inventory extractor)
- `docs/court_enrichment_field_contract.md` (court LLM schema contract; 16-field descriptor + deterministic anchors)
- `docs/laws_de_llm_enrichment_schema.md` (laws-DE 12-field schema; LLM outputs semantic only, static block deterministic)
- `docs/enrichment_optimization.md` (Kaggle 2×T4 throughput analysis that motivated Blackwell migration)
- `.claude/skills/swiss-citation-experiments/SKILL.md`

Definitive enriched artifacts produced by this family:
1. `outputs/court_llm_descriptors_0000000_all.jsonl` — **363,258 rows; 363,212 ok + 45 ok_after_retry + 1 failed (99.987 % ok); 15,895 s (≈ 4.4 h) on RTX PRO 6000 Blackwell.**
2. `outputs/law_llm_descriptors_0000000_all.jsonl` — **173,033 rows; 167,775 ok + 857 ok_after_retry + 2,283 static_skip + 2,118 failed (98.8 % usable); avg terms_grounded_pct 0.902.**

The Qwen3.5-35B-A3B authority-card line never produced a usable artifact.

---

## Subfolder 07 — law-de LLM enrichment (Qwen3-8B-AWQ, Blackwell)

The law-de pipeline takes `artifacts/law_llm_input.jsonl` (175,933 rows of Swiss federal law articles in German), enriches each article with the 12-field semantic schema in `docs/laws_de_llm_enrichment_schema.md`, and writes `law_llm_descriptors_0000000_all.jsonl`. After deterministic deduplication / filtering, 173,033 rows reach the LLM stage. All three production variants below use `Qwen/Qwen3-8B-AWQ` with `awq_marlin` quantization, `enable_prefix_caching=True`, FP16 KV (FP8 KV was rejected; see verdict below).

### 1. `enrich_laws_de_qwen3_8b_kaggle_pre_optimization.ipynb`

- **Goal:** Baseline law-de enrichment on Blackwell, mirroring the court 363k run's settings exactly.
- **Data in:** `law_llm_input.jsonl` (173,033 LLM rows after filtering).
- **Approach:** vLLM 0.10+, `tensor_parallel_size=1`, `gpu_memory_utilization=0.92`, `max_model_len=2560`, `max_num_seqs=320`, `quantization=awq_marlin`, `enforce_eager=False`, `kv_cache_dtype=None` (FP16, after FP8 KV crashed AWQ Marlin on SM 12.0), `enable_thinking=False`, `max_new_tokens=700`, `retry_max_new_tokens=850`, `repetition_penalty=1.05`, `submit_chunk=2048`. Two-pass: main + thinking retry on parse failure.
- **Result:** Ran cells fine but the captured session shows **70,737 rows finished in 6,337.4 s = 11.16 rows/s** (note: 102,296 rows had already been done in a prior session, so cumulative file = 173,033). Prompt tokens 119.8 M (avg 1,694/row), output tokens 20.6 M (avg 291/row), avg terms_grounded_pct = 0.909.
- **Best result:** This run rolled directly into the final output file. Status counts across all 173,033 rows: `{ok: 167,775, ok_after_retry: 857, static_skip: 2,283, failed_descriptor_parse: 2,118}`. Median concept_count = 6.0, median term_pair_count = 4.0. Forbidden-field count = 0 (schema contract enforced post-merge).
- **Verdict:** Kept — the file `outputs/law_llm_descriptors_0000000_all.jsonl` produced here is the canonical law-side artifact.

### 2. `enrich_laws_de_qwen3_8b_kaggle.ipynb`

- **Goal:** Continuation of the same job (checkpoint resume), executing the remaining slice on the Blackwell. This is the script that **closed out** the law enrichment file.
- **Approach:** Same `Qwen3-8B-AWQ` stack with `max_model_len=3072`, `max_num_seqs=384` (raised from 320 because the FP16 KV budget on 95 GB allowed a bigger slot count), `enable_ngram_speculation=False`, all 175,933 rows in scope, `start=0`, `limit=0`.
- **Result:** Session captured shows **173,033 rows checked at end, 70,737 written this session** (the other 102,296 came from prior runs). Status counts: `{ok: 167,775, ok_after_retry: 857, static_skip: 2,283, failed_descriptor_parse: 2,118}`. Avg terms_grounded_pct = 0.902. Failed rows file present (2,118 rows). 6,578 boilerplate-role rows correctly suppress legal_rule. 85,080 substantive rows have non-empty rule.
- **Verdict:** Kept — together with the pre-optimization run, this is the production trace that produced the 173k law artifact.

### 3. `enrich_laws_de_qwen3_8b_kaggle_optimized.ipynb`

- **Goal:** Further tuning to push throughput. Tighter prompts (max_text_chars audit), larger concurrency.
- **Approach:** Same stack with `max_model_len=2560` (tightened to fit real prompt p99 ≈ 1,820 tok + 480 output), `max_num_seqs=480` (raised again because the tighter model_len freed KV budget), `submit_chunk=2048`, `max_new_tokens=512`, `retry_max_new_tokens=600`, `repetition_penalty=1.0`. Boilerplate filter writes static stubs without LLM call. Length-tier descending sort to keep continuous batches full.
- **Result:** Attempted to resume from the checkpoint (33,003 rows already done, 140,030 to go) but **crashed mid-batch** with `VLLMValidationError: This model's maximum context length is 3072 tokens. However, you requested 0 output tokens and your prompt contains at least 3073 input tokens` — i.e. some retry prompt overflowed when `max_model_len` was reduced to 2560 then the next chunk pushed past it.
- **Verdict:** Rejected as the closing run, but instructive: the `max_num_seqs=480` setting works on Blackwell at FP16 KV, and the boilerplate stub path saves ~2,283 LLM calls. The throughput proof remains the 11.16 rows/s baseline from variant 1.

### 4. `investigate_laws_de_token_optimization.ipynb`

- **Goal:** Notebook-only analysis (no production output): measure prompt-token distributions across `law_llm_input.jsonl`, plan multi-bucket inference configs, optionally run a small vLLM pilot to estimate per-bucket `max_new_tokens` and `max_num_seqs`.
- **Approach:** Tokenize a stratified 10k-row sample with `Qwen/Qwen3-8B-AWQ` tokenizer (both at `max_text_chars=1500` truncation and untruncated); compute p50/p99/max in 17 char buckets; derive `rec_max_model_len`, `rec_max_text_chars`, and `rec_max_num_seqs` per bucket using a KV-pool memory model (95 GB × 0.92 minus weights/activations). Optional pilot: 120 rows/bucket × ~10 buckets through vLLM to capture real output-token p99.
- **Result:** Writes `artifacts/laws_de_bucketed_config.json` (bucket schedule). Pilot output not captured in the inventory dump but the heuristic fallback (256/320/384/480 max_new_tokens by prompt length tier) is in the config writer.
- **Verdict:** Diagnostic only. Its findings fed the variant-3 settings (`max_model_len=2560`, `max_num_seqs=480`) and confirmed `text_chars=1500` truncation costs ~1500 × median 0.27 tok/char ≈ 405 tok at most.

### Subfolder 07 summary

The canonical law-DE enrichment is the file produced by variants 1 + 2 together: **173,033 rows, 98.8 % usable, 0.902 avg terms-grounding**. Variants 3 (`_optimized`) and 4 (`investigate_*`) are stepping stones — 3 explored the bigger `max_num_seqs=480` envelope (worked at the engine level but the resume run hit a context-length mismatch on a tail prompt), 4 derived the empirical token statistics those settings rest on. Throughput: ~11 rows/s on Blackwell at FP16 KV (FP8 KV is structurally blocked by AWQ Marlin on SM 12.0).

---

## Subfolder 08 — court LLM descriptor extraction (Qwen3-8B-AWQ)

This is the long progression that produced **`outputs/court_llm_descriptors_0000000_all.jsonl` (363,258 rows; 99.987 % ok)** on Colab Blackwell. The schema is "minimal descriptor only" per `docs/court_enrichment_field_contract.md` §"Recommended production rag_enrichment": 16 semantic fields (legal_area, primary_domain, secondary_domain, legal_domain_path, topic, subtopic, micro_topic, concepts_en, terms_original, doctrinal_rule, legal_test, fact_pattern_tags, procedural_context, paragraph_role, authority_role, specificity_score). It explicitly excludes statute/case anchors, outcomes, summaries, questions, and retrieval views — those are built deterministically post-merge.

Chronological evolution (mtimes):

| # | Date | Notebook | Stage | Hardware | Status |
|---|---|---|---|---|---|
| 1 | 2026-05-03 13:17 | `court_llm_descriptor_kaggle.ipynb` | Initial Kaggle scaffold (Qwen3-8B-AWQ on /kaggle/input/models path) | Kaggle 2×T4 | Code only — no captured run output |
| 2 | 2026-05-03 13:41 | `court_llm_descriptor_dual_t4_vllm.ipynb` | Alternate Kaggle path (dual_worker on 2×T4) | Kaggle 2×T4 | Stalled at 0% — no production output |
| 3 | 2026-05-03 13:53 | `court_llm_descriptor_safe_2t4_vllm.ipynb` | Safer Kaggle path (tp2, one model across 2×T4) | Kaggle 2×T4 | Stalled at 0% — no production output |
| 4 | 2026-05-03 14:09 | `court_llm_descriptor_qwen3_8b_awq.ipynb` | Colab port of the Kaggle prompt builder (uses HF `Qwen/Qwen3-8B-AWQ` and Drive paths) | "2×T4" reported but mtime same day as Blackwell exploration | 50-row smoke run: 50/50 ok in 211.5 s ≈ 0.24 rows/s |
| 5 | 2026-05-03 14:58 | `court_llm_descriptor_10k_optimized.ipynb` | First Blackwell-tuned run | Colab Blackwell | **10,000 rows in 759 s ≈ 13.17 rows/s; 9,998 ok + 2 ok_after_retry; 0 failures** |
| 6 | 2026-05-04 13:42 | `court_llm_descriptor_blackwell_optimized_v4.ipynb` | Blackwell + FlashInfer + FP16 KV (FP8 KV ruled out); tighter prompts | Colab Blackwell | **10,000 rows in 639 s ≈ 15.64 rows/s** |
| 7 | 2026-05-05 01:39 | `court_llm_descriptor_blackwell_optimized_363k_run.ipynb` | **Full-corpus run** (363,258 target cards) | Colab Blackwell | **363,258 rows in 15,895 s ≈ 22.85 rows/s; 363,212 ok + 45 ok_after_retry + 1 failed → 99.987 % ok** |

### 8.1 `court_llm_descriptor_kaggle.ipynb` (2026-05-03)

- **Goal:** First attempt at the descriptor schema on Kaggle's 2×T4 + AWQ model dataset.
- **Approach:** `tensor_parallel_size=1`, `gpu_memory_utilization=0.78`, `max_model_len=4096`, `max_num_seqs=8`, `batch_size=4`, `enforce_eager=True`, `quantization='awq_marlin'`, `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (FlashInfer JIT `-lcuda` link fails on Kaggle), `use_structured_outputs=False` (Kaggle vLLM 0.20 crashes on structured `_backend`).
- **Result:** Scaffold only — the captured inventory shows no run output (cells filled with code but no execution telemetry). This notebook established the prompt and parsing pipeline.
- **Verdict:** Skeleton kept as the contract reference; not the producer.

### 8.2 `court_llm_descriptor_dual_t4_vllm.ipynb` (2026-05-03)

- **Goal:** Speed up the Kaggle path by spawning two single-GPU vLLM workers (dual_worker mode), each on its own T4, processing disjoint shards.
- **Approach:** `gpu_mode='dual_worker'`, `tensor_parallel_size=1`, `max_num_seqs=8`, `batch_size=4`, `quantization='awq_marlin'`. Subprocess-based sharding writes two output files merged post-hoc.
- **Result:** The captured output ends at `llm-descriptor batches: 0% | 0/13 [00:00<?, ?it/s]` — the workers never produced anything. Failure mode consistent with Kaggle 30 GiB host RAM (the dual-worker path loads two full Python runtimes).
- **Verdict:** Rejected. Documented as a near-OOM path on Kaggle.

### 8.3 `court_llm_descriptor_safe_2t4_vllm.ipynb` (2026-05-03)

- **Goal:** Safer Kaggle alternative — one vLLM engine with `tensor_parallel_size=2` so both T4s share one model and avoid the dual-worker RAM blow-up.
- **Approach:** `gpu_mode='tp2'`, `tensor_parallel_size=2`, `disable_custom_all_reduce=True`. Otherwise identical to 8.1.
- **Result:** Same stall — `llm-descriptor batches: 0% | 0/13 [00:00<?, ?it/s]`. The captured trace doesn't include the actual abort cause, but the working chronology shows the Kaggle path was abandoned the same afternoon in favor of Colab Blackwell.
- **Verdict:** Rejected. The 8.2/8.3 pair is the documented evidence that 2×T4 = 14.6 GiB each cannot sustain Qwen3-8B-AWQ at useful throughput for 2.4 M rows. This is exactly the conclusion in `docs/enrichment_optimization.md` (extrapolated 86–115 h baseline / 10–13 Kaggle sessions even at 5 % retry rate).

### 8.4 `court_llm_descriptor_qwen3_8b_awq.ipynb` (2026-05-03)

- **Goal:** Colab port of the Kaggle prompt builder, with Drive paths and HF model id `Qwen/Qwen3-8B-AWQ`. Smoke-test scale.
- **Approach:** `gpu_mode='single'`, `gpu_memory_utilization=0.90`, `max_num_seqs=16`, `batch_size=8`, `enforce_eager=False`, `quantization='awq'`. The captured run reports "GPU 0/1: Tesla T4 14.46 GiB" — i.e. Colab assigned T4 here.
- **Result:** 50-row test (`limit=50`) finished in 211.5 s = 0.24 rows/s, 50/50 ok. This proved the schema produces grounded descriptors and the parse pipeline works.
- **Verdict:** POC kept — confirmed schema correctness before paying for Blackwell time.

### 8.5 `court_llm_descriptor_10k_optimized.ipynb` (2026-05-03)

- **Goal:** First serious throughput run on Blackwell (Colab assigned RTX PRO 6000 Blackwell, 94.43 GiB).
- **Approach:** `gpu_mode='single'`, `gpu_memory_utilization=0.82`, `max_model_len=3072` (down from 4096; smaller KV per slot → more concurrency), `max_num_seqs=96`, `batch_size=64`, `enforce_eager=False`, `quantization='awq_marlin'`, `max_text_chars=2400` (down from 3200), `max_new_tokens=320`, `retry_max_new_tokens=448`, `temperature=0.0`, `repetition_penalty=1.02`. `use_structured_outputs=False` (still avoiding the dict._backend bug).
- **Result:** **10,000 rows in 759.19 s = 13.17 rows/s, 9,998 ok + 2 ok_after_retry, 0 failures.** Extrapolation: 363k × (1/13.17) ≈ 7.7 h.
- **Verdict:** Kept. This established the Colab-Blackwell + Qwen3-8B-AWQ baseline that beats every Kaggle attempt by ~50×.

### 8.6 `court_llm_descriptor_blackwell_optimized_v4.ipynb` (2026-05-04)

- **Goal:** Squeeze more throughput via FlashInfer + tighter KV.
- **Approach:** Adds the 3-package FlashInfer install (`flashinfer-python` + `flashinfer-cubin` pinned + `flashinfer-jit-cache` for `cu130`); `gpu_memory_utilization=0.92`, `max_model_len=3072`, `max_num_seqs=384`, `quantization='awq_marlin'`, `kv_cache_dtype=None` (FP8 KV was tested and crashed AWQ Marlin on SM 12.0 — same finding repeated for the law-DE run), `enforce_eager=False`. Continuous batching at `max_num_seqs=384` is what the comment block at line 372 calls the "working 17.94 rows/sec configuration."
- **Result:** **10,000 rows in 639.35 s = 15.64 rows/s.** That's +19 % vs the 10k optimized baseline at the same accuracy.
- **Verdict:** Kept. This is the configuration carried into the full 363k run.

### 8.7 `court_llm_descriptor_blackwell_optimized_363k_run.ipynb` (2026-05-05) — **canonical producer**

- **Goal:** Full-corpus enrichment of the 363,258 retrieval-target cards in `court_authority_cards_v4_target_cards.jsonl`.
- **Approach:** Same as v4 (Blackwell + FlashInfer + FP16 KV + `awq_marlin` + `max_num_seqs=384` + `max_model_len=3072`). Checkpoint resume via per-row `_source_row` set written to a side JSONL on Drive. `submit_chunk=2048` per `llm.generate()` call. `use_structured_outputs=False`, `enable_thinking=False`, `temperature=0.0`, `max_new_tokens=320`, `retry_max_new_tokens=448`. No speculative decoding.
- **Result:** **363,258 rows in 15,895.0 s = 22.85 rows/s.** Prompt tokens 207.98 M (avg 573/row — court paragraphs are shorter than laws), output tokens 80.69 M (avg 222/row). **Status counts: `{ok: 363,212, ok_after_retry: 45, failed_descriptor_parse: 1}`. 99.987 % ok.** Forbidden-field count in QC scan: 0.
- **Verdict:** Kept — this is the canonical court LLM enrichment file referenced in the experiments skill ("the 363k court LLM rows ARE merged into `unified_retrieval.sqlite`"; resolved phantom issue line in `swiss-citation-experiments`). The throughput jump from 15.64 → 22.85 rows/s vs the 10k benchmark is the continuous-batching saturation effect at full corpus scale (better KV pool reuse and prefix-cache reuse across the 384-slot pipeline).

### Subfolder 08 summary

The Kaggle 2×T4 line (`kaggle`, `dual_t4_vllm`, `safe_2t4_vllm`) was abandoned because of Kaggle's 14.6 GiB-per-GPU + 30 GiB host RAM ceiling — none of the three produced any output. The Colab port (`qwen3_8b_awq`) validated the prompt and parser on 50 rows. The 3-step Blackwell tightening (`10k_optimized` 13.17 → `blackwell_optimized_v4` 15.64 → `blackwell_optimized_363k_run` 22.85 rows/s) shows the actual lever stack: FlashInfer install, FP16 KV (FP8 KV is blocked by AWQ Marlin on SM 12.0), `max_num_seqs=384` continuous batching, `max_model_len=3072`, length-tier batching, and per-row token budgets. The 363k run is the canonical producer.

---

## Subfolder 09 — court citation concept enrichment (Kaggle 2×T4 era, pre-08 redesign)

This folder is the predecessor to subfolder 08. The schema is the **full** `rag_enrichment` from the original notebook design (per `docs/court_enrichment_field_contract.md` line 37): includes `statute_anchors`, `case_anchors`, `outcome_signal`, `paragraph_role`, `fact_pattern_tags`, `concepts_en`, `terms_original`, etc. — i.e. it asks the LLM for both semantic descriptors and formal anchors. The schema redesign that produced subfolder 08 split formal anchors off to deterministic post-processing, leaving only the 16 semantic fields for the LLM. So 09 is "what we tried first; descriptor-only schema replaced it."

### 9.1 `court_concept_smoke_test_10_base.ipynb` (2026-05-02 20:28)

- **Goal:** Earliest 10-row smoke test on Kaggle's 2×T4.
- **Approach:** `Qwen/Qwen3-8B-AWQ`, `tensor_parallel_size=1`, `gpu_memory_utilization=0.70`, `max_model_len=4096`, `text_chars=5500` (wide budget — later reduced).
- **Result:** No run output captured in inventory.
- **Verdict:** POC. Defined the prompt template later inherited.

### 9.2 `court_concept_smoke_test_10_v3_local.ipynb` (2026-05-02 20:57)

- **Goal:** Tighter smoke test using only substantive paragraphs (`min_text_chars=600`).
- **Approach:** `text_chars=3200`, `max_num_seqs=16`, `enforce_eager=True`, `quantization='awq_marlin'`, `batch_size=5`. Output to `enriched_court_citations_10.jsonl`.
- **Result:** No run output captured.
- **Verdict:** POC. Established the local Kaggle model path (`/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1`).

### 9.3 `court_concept_enrichment_robust_v4_predecessor.ipynb` (2026-05-02 21:20)

- **Goal:** First "robust" production schema — full-rag fields including all anchors, outcome signal, paragraph_role.
- **Approach:** Kaggle 2×T4, `vllm`, `tensor_parallel_size=1`, `gpu_memory_utilization=0.72`, `max_model_len=4096`, `max_num_seqs=8`, `enforce_eager=True`, `quantization='awq'`, `force_triton_attention=True`, `max_new_tokens=768`, `retry_max_new_tokens=1024`, `temperature=0.0`, `enable_thinking=False`, `max_retries=2`. Schema requires `concepts_en`, `terms_original`, `statute_anchors`, `case_anchors`, `fact_pattern_tags`, `paragraph_role`, `outcome_signal` (with explicit JSON-schema validation + retry).
- **Result:** No run output captured (cells filled but no measured throughput in inventory).
- **Verdict:** Rejected as a producer — the schema this notebook implemented is exactly what the field contract later marks as "current notebook fields" vs "recommended production fields". Trusting the LLM with anchors fails the contract; anchors must be deterministic.

### 9.4 `court_concept_enrichment_robust_v5_t4.ipynb` (2026-05-02 21:26)

- **Goal:** Same schema, slightly tighter Kaggle config; explicit `gpu_mode='single'` for the smoke test, with per-shard production via two notebook copies.
- **Approach:** `max_text_chars=4500`, `max_new_tokens=768`, `retry_max_new_tokens=1024`, `quantization='awq_marlin'` (faster than plain awq on this vLLM), `use_structured_outputs=False` (Kaggle vLLM 0.20 crashes on guided JSON), `min_text_chars=250`.
- **Result:** No run output captured.
- **Verdict:** Same as v4 — schema-too-wide and Kaggle-bound; superseded by the descriptor-only line in folder 08.

### 9.5 `court_concept_enrichment_minimal_llm.ipynb` (2026-05-03 13:00)

- **Goal:** **Pivot** notebook — the schema is cut down to remove formal anchors. This is the transitional file between the "full rag" schema in 9.3/9.4 and the "minimal descriptor" schema in 08.
- **Approach:** Same Kaggle/vLLM stack, but the prompt/parsing now produce only semantic descriptors plus retrieval views. Includes `normalizer_search_paths` for a `court_enrichment_normalizer.py` companion script.
- **Result:** No production run captured. Throughput stats not reproduced in inventory.
- **Verdict:** Concept POC for the descriptor-only split. The same idea is re-implemented in `court_llm_descriptor_kaggle.ipynb` (8.1) the same day and then carried to Colab Blackwell, which is the line that actually produces the canonical 363k file.

### Subfolder 09 summary

These five notebooks are the **schema iteration history** before the family pivoted to the descriptor-only schema in folder 08. The lineage is: smoke_test_10_base (5500-char prompt) → smoke_test_10_v3_local (3200-char prompt + min_text_chars=600) → robust_v4_predecessor (full schema + anchors) → robust_v5_t4 (tighter + structured-outputs ablation) → minimal_llm (anchors removed). None of them produced a canonical output artifact; their value is documentary — they show why the descriptor-only split happened (Kaggle 2×T4 throughput floor + the field-contract rule that anchors must be deterministic). The output format that "won" is the descriptor-only schema in `docs/court_enrichment_field_contract.md` §"Recommended production rag_enrichment fields", implemented in subfolder 08.

---

## Subfolder 10 — authority card enrichment (the Qwen3.5-35B-A3B side-quest)

This folder explores a **bigger model** (Qwen3.5-35B-A3B, 35 B params total / 3 B active MoE) for richer authority cards (`court_authority_cards_v4.jsonl`, 2.48 M rows). The hypothesis was that a bigger model could produce English summaries, court_holdings, NL queries, and search_keywords — fields explicitly forbidden in the court descriptor contract. **None of these notebooks produced a usable corpus-scale artifact.** The 8B AWQ variants (`auth_cards_enrich_qwen3_8b_*`) either re-ran the 363k descriptor file under a different name or failed to load. The 35B variants ran a handful of cards each with very high JSON parse error rates.

### 10.1 `auth_cards_enrich_qwen3_8b_awq_kaggle_local.ipynb` (2026-05-02 05:14)

- **Goal:** Kaggle smoke test of Qwen3-8B-AWQ via Transformers + gptqmodel (no vLLM).
- **Approach:** `transformers + accelerate + gptqmodel`, smoke-test bookkeeping with a `RAG_SCHEMA` validator.
- **Result:** **Model load failed** — `RuntimeError: AWQ load failed. Run: !pip install autoawq or set CONFIG.model_id='Qwen/Qwen3-8B' + CONFIG.use_bnb_4bit=True.` No cards processed.
- **Verdict:** Rejected. Documented that `gptqmodel`-only AWQ loading fails on Kaggle's transformers ≥ 5.7; needs autoawq as a fallback.

### 10.2 `auth_cards_enrich_qwen3_8b_kaggle.ipynb` (2026-05-02 05:02)

- **Goal:** Earlier Kaggle variant of the same plan.
- **Approach:** `transformers>=4.51.0`, `accelerate`, `tqdm`. The captured file contents are mojibake (UTF-8 displayed as cp1252) — a known phantom issue per `swiss-citation-experiments` skill line 148.
- **Result:** No run output captured.
- **Verdict:** Superseded by 10.1 and the Colab Blackwell variant 10.3.

### 10.3 `auth_cards_enrich_qwen3_8b_colab.ipynb` (2026-05-02 04:31)

- **Goal:** Colab Blackwell variant of the qwen3-8B auth-card path, using vLLM and structured-output guards. Targeted-enrichment mode reads `court_authority_cards_v4_enrichment_targets.jsonl` and only LLM-enriches those rows; everything else gets deterministic fallback.
- **Approach:** vLLM with `Qwen/Qwen3-8B-AWQ`, structured-output via `StructuredOutputsParams` or `GuidedDecodingParams` fallback. `BATCH_SIZE=64`, `MAX_TOKENS=256`, `ENABLE_PREFIX_CACHE=True`, smoke test required green before main run. `_FNAME = court_authority_cards_v4_target_cards.jsonl` — i.e. it points at the **same 363,258-row file** the descriptor-only run consumed.
- **Result:** Captured run output shows the smoke test passed (GREEN: 8 target LLM outputs valid and reference-grounded), then the main pass shows `Resuming at line 2,238,067` while `Total=363,258`, `To process=-1,874,809` — the checkpoint pointer was set far past EOF, so the main loop iterates 0 LLM calls (`enrich: 2238067card [00:00, ?card/s]; Done. Failed rows captured: 0; LLM target rows attempted: 0`). This is what you'd see if the output file `court_authority_cards_rag_targets.jsonl` was already fully populated by an earlier successful run — but no captured prior session in this notebook shows that, suggesting the checkpoint number is stale from a different run.
- **Verdict:** Ambiguous. The schema (`legal_topic`, `legal_question`, `legal_rule`, `court_holding`, `english_summary`, `english_legal_concepts`, `search_keywords`, `natural_language_queries`, plus the same `paragraph_role`/`outcome_signal` enums) is richer than the descriptor schema. But there is no captured production run, so no measured throughput, ok-rate, or grounding number. Treat as exploratory.

### 10.4 `auth_cards_enrich_qwen35_colab.ipynb` (2026-04-30 23:45)

- **Goal:** First Qwen3.5-35B-A3B attempt. BF16 weights (~70 GB), MoE, fits on 96 GB Blackwell.
- **Approach:** `MODEL_ID='Qwen/Qwen3.5-35B-A3B'`, `QUANTIZATION=None`, `GPU_MEMORY_UTIL=0.87`, `BATCH_SIZE=64` ("MoE is faster per token; bump up vs dense 32B"), `MAX_MODEL_LEN=4096`, `TEMPERATURE=0.15`, `MAX_TOKENS=600`, `LIMIT=1000`.
- **Result:** **`KeyboardInterrupt` at card 24/1024.** The notebook was halted mid-batch — no completed enrichment.
- **Verdict:** Rejected as a producer. The bigger model loaded and started inference but was abandoned before any meaningful sample.

### 10.5 `auth_cards_enrich_qwen35_fast_quality_json_optimized.ipynb`

- **Goal:** Same 35B path with the failure-capture pipeline mature (`BATCH_SIZE=128`, structured outputs, retry pass, deterministic fallback for any rows the LLM can't parse).
- **Approach:** `MODEL_ID=Qwen/Qwen3.5-35B-A3B`, `BATCH_SIZE=128`, `MAX_TOKENS=256`, `RETRY_MAX_TOKENS=512`, `SMOKE_MAX_TOKENS=1024`, `ENRICH_TARGET_CARD_FILE_ONLY=True`, `TARGETED_ENRICHMENT=True`, `WRITE_DETERMINISTIC_FALLBACK_FOR_UNTARGETED=True`, `ABORT_IF_ERROR_RATE_ABOVE` early-abort guard.
- **Result:** Inventory shows the full code path (smoke test → batched LLM → failure-capture → retry-only loop). The captured run output is **not present** — only the code with placeholders for `[batch N] llm_size=128, batch_time=…, batch_rate=… target_cards/s, …, total_json_errors=…`. No corpus-scale metrics.
- **Verdict:** Open. The structural pipeline is sound (target subset only, deterministic fallback for the rest, separate failure file, retry-only loop). But no captured run means no quantified result.

### 10.6 `auth_cards_enrich_qwen35_fast_quality_json_optimized_pre_failure_capture.ipynb`

- **Goal:** Predecessor of 10.5 — same schema and stack, before the explicit failed-rows capture path was added.
- **Approach:** Identical model and batch settings; missing the failure JSONL fork.
- **Result:** No captured run output. Code identical to 10.5 minus the FAILED_FILE write.
- **Verdict:** Superseded by 10.5.

### 10.7 `auth_cards_enrich_qwen35_fast_quality_json_v2.ipynb`

- **Goal:** Earlier iteration of the JSON-robustness pipeline.
- **Approach:** `MODEL_ID=Qwen/Qwen3.5-35B-A3B`, structured outputs + reference-grounding validator (`validate_reference_grounding`) ensures the model's outputs reference statutes/cases that actually appear in the source text. `BATCH_SIZE=32` (smaller than the "optimized" successor's 128).
- **Result:** No captured run.
- **Verdict:** Superseded by 10.5.

### 10.8 `auth_cards_enrich_qwen35_safe_no_vllm.ipynb`

- **Goal:** Fallback path — run Qwen3.5-35B-A3B on plain Transformers (no vLLM) for environments where vLLM fails.
- **Approach:** `transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, …)` with explicit `MODEL LOAD FAILED SAFELY. No exception was re-raised.` guard.
- **Result:** Code only. No run output. Throughput on plain Transformers for a 35B model would be ~1/10 of vLLM, so this path was preserved only as a "model loads at all" diagnostic.
- **Verdict:** Diagnostic.

### 10.9 `auth_cards_enrich_qwen35_updated_fast_json.ipynb`

- **Goal:** Mid-iteration variant; first to actually report batch-level throughput on real cards.
- **Approach:** Same 35B stack, `BATCH_SIZE=32`, captured smoke + main pass.
- **Result:** **Real captured run output: 1,000 cards processed in ~7 min, ~2.05–2.28 cards/s steady-state.** *But*: `[batch 30] size=32, batch_time=14.1s, batch_rate=2.26 cards/s, avg_rate=2.05 cards/s, eta≈0.0 min, json_errors=800. Done. JSON parse errors: 801`. **801 / 1,000 = 80.1 % JSON parse-failure rate.** Total output cards: 1,166 (1,000 LLM + 166 deterministic fallback prefix).
- **Verdict:** Rejected as a producer. The 35B-A3B model **cannot reliably emit the required JSON schema even with structured outputs** — 80 % parse-fail. The subsequent "_optimized" variants tried to fix this with retry pass + reference grounding but never reached a corpus-scale captured run.

### 10.10 `auth_cards_enrich_qwen35_vllm_fast_stable.ipynb`

- **Goal:** Stable vLLM-based 35B run, with explicit "No vLLM structured JSON decoding support found. Do not run unconstrained generation." guard. Targeted-enrichment mode (TARGET_FILE = the 363k target card list).
- **Approach:** `MODEL_ID='Qwen/Qwen3.5-35B-A3B'`, structured-output via `StructuredOutputsParams`/`GuidedDecodingParams`, reference-grounding validator. Smoke-test gate must pass before main.
- **Result:** **Same checkpoint pollution as 10.3**: `Resuming at validated line 2,238,067; Total=363,258; To process=-1,874,809; enrich: 2238067card [00:00, ?card/s]; Done. Failed rows captured: 0.` No actual cards processed in the captured session.
- **Verdict:** Open / superseded. The infrastructure works; no real run was captured.

### 10.11 `auth_cards_enrich_qwen35_vllm_fixed.ipynb`

- **Goal:** First vLLM-based Qwen3.5-35B variant after the original `qwen35_colab` failure. Adds MTP-1 speculative decoding (`USE_MTP=True`, `MTP_TOKENS=1`) and disables prefix caching per the vLLM recipe for latency-focused MoE.
- **Approach:** `Qwen/Qwen3.5-35B-A3B`, BF16, `GPU_MEMORY_UTIL=0.90`, `BATCH_SIZE=4`, `LANGUAGE_MODEL_ONLY=True`. `ENABLE_PREFIX_CACHING=False`. Pillow/numpy/scipy pinned to avoid Colab dependency hell.
- **Result:** No captured run telemetry (model load shown but no batch-rate prints).
- **Verdict:** Documented vLLM Qwen3.5 recipe; no production result.

### 10.12 `auth_cards_enrich_qwen35_vllm_fixed_pillowfix.ipynb`

- **Goal:** Pillow/PIL fix layered on top of `qwen35_vllm_fixed` — forces a clean `pillow==11.3.0` reinstall to avoid the `PIL._typing/_Ink` mismatch.
- **Approach:** Same 35B vLLM stack, plus `pillow==11.3.0; numpy==2.3.5; scipy==1.16.3` pinning, plus `pip uninstall -y -q PIL pillow numpy scipy torchvision vllm transformers` cleanup.
- **Result:** **Captured run: 1,000 cards in batches of 32 at 2.23–2.28 cards/s; 800 JSON parse errors out of ~1,000 batched (80 % parse-fail), same as 10.9.** Total output: 1,166 cards. Same conclusion: structured outputs fail to constrain Qwen3.5-35B-A3B reliably on this schema.
- **Verdict:** Rejected. The Pillow fix solved a dependency bug but not the underlying model-quality problem.

### Subfolder 10 summary

The authority-card line had two threads:

1. **Qwen3-8B-AWQ thread** (`qwen3_8b_kaggle`, `qwen3_8b_awq_kaggle_local`, `qwen3_8b_colab`). Kaggle variants failed to load (autoawq vs gptqmodel mismatch). The Colab variant works but its captured "run" iterates 0 cards because the checkpoint is stale; no measured throughput exists. The architecture sketch is the richest schema attempted (legal_topic, legal_question, legal_rule, court_holding, english_summary, english_legal_concepts, search_keywords, natural_language_queries) but no captured production output proves it scales.

2. **Qwen3.5-35B-A3B thread** (8 notebooks, 2026-04-30 to ~2026-05-02). The model loads on Blackwell (BF16, ~70 GB weights + KV) but **emits invalid JSON ~80 % of the time** under structured outputs (`qwen35_updated_fast_json` and `qwen35_vllm_fixed_pillowfix` both captured 800/1000 parse-fails). Throughput ~2.25 cards/s — at that rate plus 80 % failure plus retry overhead, the 363k corpus would take 18+ hours of net wall-clock with massive rework. The line was abandoned.

**Definitive auth-card producer:** none captured. The production data path for downstream retrieval is the **descriptor-only** file `court_llm_descriptors_0000000_all.jsonl` from subfolder 08, not the richer "authority card RAG" output this folder was reaching for. This matches the experiments-skill note that the court schema "forbids `english_summary` by design."

---

## Cross-folder verdict

- **Canonical court LLM enrichment:** `outputs/court_llm_descriptors_0000000_all.jsonl` — produced by `notebooks/08_court_llm_descriptor_extraction/court_llm_descriptor_blackwell_optimized_363k_run.ipynb`. 363,258 rows, 99.987 % ok, 22.85 rows/s, ~4.4 h on Blackwell. Confirmed merged into `unified_retrieval.sqlite`.
- **Canonical law-DE enrichment:** `outputs/law_llm_descriptors_0000000_all.jsonl` — produced by `notebooks/07_law_de_enrichment/enrich_laws_de_qwen3_8b_kaggle_pre_optimization.ipynb` + `enrich_laws_de_qwen3_8b_kaggle.ipynb` (cumulative; resume-checkpointed). 173,033 rows, 98.8 % usable, ~11 rows/s on Blackwell.
- **Schema contract:** subfolder 08 follows `docs/court_enrichment_field_contract.md` (descriptor-only, anchors built deterministically post-merge). Subfolder 07 follows `docs/laws_de_llm_enrichment_schema.md` (12 LLM fields + static block, post-merge normalizer suppresses rules on boilerplate roles). Subfolder 09 is the rejected predecessor schema (LLM-emitted anchors); subfolder 10 is the rejected bigger-model side-quest.
- **Hardware envelope confirmed:** RTX PRO 6000 Blackwell Server Edition (~95.6 GB VRAM) is the only viable hardware for these workloads. Kaggle 2×T4 (14.6 GiB each + 30 GiB host RAM) blocks Qwen3-8B-AWQ at useful throughput. FP8 KV is blocked by AWQ Marlin on SM 12.0 — both court and law runs documented the same finding independently. Qwen3.5-35B-A3B fits on Blackwell BF16 but fails the JSON-schema requirement structurally.
- **vLLM settings that work on Blackwell + Qwen3-8B-AWQ:** `quantization='awq_marlin'`, `kv_cache_dtype=None` (FP16), `enforce_eager=False`, `enable_prefix_caching=True`, `max_model_len=3072`, `max_num_seqs=384` (court) / 320–480 (law), `gpu_memory_utilization=0.92–0.95`, `temperature=0.0`, `enable_thinking=False`, `use_structured_outputs=False` (Kaggle vLLM 0.20 bug; Blackwell vLLM 0.10 also avoids it), submit chunk 2048, length-tier descending sort, boilerplate-skip pre-pass.

## Files referenced

- `e:\swiss_citation_extraction\notebooks\07_law_de_enrichment\enrich_laws_de_qwen3_8b_kaggle.ipynb`
- `e:\swiss_citation_extraction\notebooks\07_law_de_enrichment\enrich_laws_de_qwen3_8b_kaggle_optimized.ipynb`
- `e:\swiss_citation_extraction\notebooks\07_law_de_enrichment\enrich_laws_de_qwen3_8b_kaggle_pre_optimization.ipynb`
- `e:\swiss_citation_extraction\notebooks\07_law_de_enrichment\investigate_laws_de_token_optimization.ipynb`
- `e:\swiss_citation_extraction\notebooks\08_court_llm_descriptor_extraction\court_llm_descriptor_kaggle.ipynb`
- `e:\swiss_citation_extraction\notebooks\08_court_llm_descriptor_extraction\court_llm_descriptor_dual_t4_vllm.ipynb`
- `e:\swiss_citation_extraction\notebooks\08_court_llm_descriptor_extraction\court_llm_descriptor_safe_2t4_vllm.ipynb`
- `e:\swiss_citation_extraction\notebooks\08_court_llm_descriptor_extraction\court_llm_descriptor_qwen3_8b_awq.ipynb`
- `e:\swiss_citation_extraction\notebooks\08_court_llm_descriptor_extraction\court_llm_descriptor_10k_optimized.ipynb`
- `e:\swiss_citation_extraction\notebooks\08_court_llm_descriptor_extraction\court_llm_descriptor_blackwell_optimized_v4.ipynb`
- `e:\swiss_citation_extraction\notebooks\08_court_llm_descriptor_extraction\court_llm_descriptor_blackwell_optimized_363k_run.ipynb`
- `e:\swiss_citation_extraction\notebooks\09_court_citation_concept_enrichment\court_concept_smoke_test_10_base.ipynb`
- `e:\swiss_citation_extraction\notebooks\09_court_citation_concept_enrichment\court_concept_smoke_test_10_v3_local.ipynb`
- `e:\swiss_citation_extraction\notebooks\09_court_citation_concept_enrichment\court_concept_enrichment_robust_v4_predecessor.ipynb`
- `e:\swiss_citation_extraction\notebooks\09_court_citation_concept_enrichment\court_concept_enrichment_robust_v5_t4.ipynb`
- `e:\swiss_citation_extraction\notebooks\09_court_citation_concept_enrichment\court_concept_enrichment_minimal_llm.ipynb`
- `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen3_8b_awq_kaggle_local.ipynb`
- `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen3_8b_kaggle.ipynb`
- `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen3_8b_colab.ipynb`
- `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen35_colab.ipynb`
- `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen35_fast_quality_json_optimized.ipynb`
- `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen35_fast_quality_json_optimized_pre_failure_capture.ipynb`
- `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen35_fast_quality_json_v2.ipynb`
- `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen35_safe_no_vllm.ipynb`
- `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen35_updated_fast_json.ipynb`
- `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen35_vllm_fast_stable.ipynb`
- `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen35_vllm_fixed.ipynb`
- `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen35_vllm_fixed_pillowfix.ipynb`
- `e:\swiss_citation_extraction\docs\court_enrichment_field_contract.md`
- `e:\swiss_citation_extraction\docs\laws_de_llm_enrichment_schema.md`
- `e:\swiss_citation_extraction\docs\enrichment_optimization.md`
- `e:\swiss_citation_extraction\.claude\skills\swiss-citation-experiments\SKILL.md`
- `e:\swiss_citation_extraction\outputs\court_llm_descriptors_0000000_all.jsonl` (canonical court enrichment, 363,258 rows; on Drive at `/content/drive/MyDrive/swiss_law/outputs/`)
- `e:\swiss_citation_extraction\outputs\law_llm_descriptors_0000000_all.jsonl` (canonical law-de enrichment, 173,033 rows; on Drive at `/content/drive/MyDrive/swiss_law/outputs/`)
