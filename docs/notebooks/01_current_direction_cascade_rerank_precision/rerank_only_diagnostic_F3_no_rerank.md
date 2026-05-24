# rerank_only_diagnostic_F3_no_rerank.ipynb

**Path:** `notebooks/01_current_direction_cascade_rerank_precision/rerank_only_diagnostic_F3_no_rerank.ipynb`

## Configuration

### Models
- `Qwen/Qwen3-Reranker-8B` — Config A (raw text, baseline instruction) and Config B (enriched doc, cross-lingual instruction). Loaded via vLLM, bfloat16, `max_model_len=1024`, `gpu_memory_utilization=0.85`, `enforce_eager=False`, `trust_remote_code=True`. Tokenizer `padding_side="left"`. `SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)`. Scores derived from softmax over the "yes" / "no" token logprobs at position 0. Body truncation `QWEN3_MAX_BODY = QWEN3_MAX_LEN - len(prefix) - len(suffix) - 8`.
- `BAAI/bge-reranker-v2-m3` — Config C. Loaded via `AutoModelForSequenceClassification`, bfloat16, CUDA. `BGE_MAX_LEN = 1024`, `BGE_BATCH = 64`. Sigmoid over logits.
- `jinaai/jina-reranker-v2-base-multilingual` — Config D. `trust_remote_code=True`, bfloat16, CUDA. `JINA_MAX_LEN = 1024`, `JINA_BATCH = 128`. Sigmoid over logits. Compatibility shim re-injects `create_position_ids_from_input_ids` into `transformers.models.xlm_roberta.modeling_xlm_roberta` for transformers >= 4.51.

### Libraries
- `vllm>=0.10.0`, `transformers>=4.51.0`, `accelerate`, `safetensors`, `huggingface_hub`
- `flashinfer-python`, `flashinfer-cubin`, `flashinfer-jit-cache` (CUDA-version-aware install; `FLASHINFER_DISABLE_VERSION_CHECK=1`)
- `pandas`, `numpy` (implicit), `torch`, `gzip`, `json`

### Hyperparameters
- `TEST_QIDS = ["val_001", "val_003", "val_010"]`
- `STAGE1_TOP_N = 50000` (rerank full v7.5 pool)
- `K_REPORT = (50, 100, 200, 500, 1000, 2000, 5000)`
- Pass criterion: `R@2000 >= 0.7` on all three queries
- Enriched representation: cit + family/role + statute anchors + concepts (English) + terms + body[:2000], total clipped to 3000 chars
- Raw representation: `search_text[:3000]` with fallback to citation/did

### Hardware
- Colab Pro+, RTX PRO 6000 Blackwell Server Edition, ~96 GB VRAM
- Python 3.12
- Wall-time budget: ~90 min for full diagnostic
- Per Qwen3 init: 15.26 GiB model memory, 62.35 GiB KV cache, GPU KV cache size 454,016 tokens, max concurrency 443.38x at 1024 tokens

### Constants / Instructions
- `INSTR_BASELINE`: "Given an English legal question about Swiss federal law, determine whether the provided document (a Swiss law article OR a court paragraph in German / French / Italian) is a RELEVANT CITATION — meaning it answers, supports, or directly relates to the question."
- `INSTR_CROSSLING`: explicit query-English / doc-DE-FR-IT framing; "Treat language differences as a translation problem, not a mismatch."
- Qwen3 prefix: `<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n`
- Qwen3 suffix: `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`

## Data

### Snapshot
- Drive (Colab): `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot`
- Local fallback root: `E:\swiss_citation_extraction`
- Doubled-nested fallback: `<root>/research/anchor_funnel_val001_v7/snapshot/snapshot`
- Selection rule: canonical first if `config.json` exists, else doubled-nested

### Snapshot artifacts loaded
- `config.json` — pipeline config
- `paths.json` — paths to original CSVs
- `corpus_snapshot.json.gz` — 255,713 docs, fields `ct` (search_text), `cit` (citation), `fam` (family), `cb` (court_base), `pr` (paragraph_role), `ln` (language), `sa` (statute anchors), `cn` (concepts), `tm` (terms)
- `per_query_snapshot.json` — `final_topk` and `curve` per qid (curve = K -> (hits, recall))
- `gold_doc_sets.json` — gold doc id sets per qid

### Input CSV
- `paths.json["val_csv"]` (Drive primary) or `<root>/data/val.csv` (fallback)
- Schema: `query_id`, `query`, `gold_citations` (semicolon-separated)

### Output
- `<DRIVE>/research/rerank_only_diagnostic/rerank_only_results.json`
- Structure: `{"fusion_baseline": {qid: {K: recall}}, "configs": {cfg: {qid: {K: recall}}}, "test_qids": [...], "stage1_top_n": 50000}`

### Per-query targets table (printed in Phase 2)
```
qid         gold  in_pool   R_max  fus_R@2k   gold_needed_for_0.7
val_001       42       39   0.929     0.381          30 ( 76.9% of in-pool)
val_003       47       36   0.766     0.255          33 ( 91.7% of in-pool)
val_010       25       22   0.880     0.520          18 ( 81.8% of in-pool)
```

## Pipeline

### Stage 0 — Setup
- Detect environment (`google.colab` -> colab; else local Windows)
- Resolve `SNAPSHOT_DIR` and `OUT_DIR`
- Conditionally install vLLM, transformers, accelerate, safetensors, huggingface_hub
- Conditionally install FlashInfer (`flashinfer-python`, `flashinfer-cubin`, `flashinfer-jit-cache` from `https://flashinfer.ai/whl/{cuda_suffix}`)
- `os.environ["TOKENIZERS_PARALLELISM"] = "false"`
- Exit with `SystemExit` on first install to force runtime restart

### Stage 1 — Warm-boot
- Load `val.csv`, build `ALL_QUERIES`, `ALL_TOTAL_GOLD`
- Decompress and parse `corpus_snapshot.json.gz` into `search_text`, `doc_meta`, `doc_statute_anchors`, `_doc_to_concepts`, `_doc_to_terms`
- Load `PER_QUERY` (final_topk + curve) and `ALL_GOLD_DOC_SET`
- Observed: 10 queries, 255,713 docs, 9.5s load

### Stage 2 — Define test set + doc representations
- Pin three hardest queries
- Build `repr_raw` and `repr_enriched`
- Print sanity table of per-query gold / in-pool / R_max / fusion R@2k

### Stage 3 — Config A: Qwen3-Reranker-8B baseline
- Lazy-load `qwen3_llm` (vLLM) and `qwen3_tok`
- Apply ipykernel stdout/stderr workaround for vLLM 0.10 worker subprocess
- For each qid, score full top-50k pool via `_qwen3_score(query, dids, INSTR_BASELINE, repr_raw)`
- Sort, compute recall at each K in `K_REPORT`
- Store under `RESULTS["A_qwen3_raw"]`

### Stage 4 — Config B: Qwen3-Reranker-8B fixed
- Reuse engine
- Re-score same pools with `INSTR_CROSSLING` + `repr_enriched`
- Store under `RESULTS["B_qwen3_fixed"]`

### Stage 5 — Teardown Qwen3, load BGE; Config C
- `del qwen3_llm; del qwen3_tok; gc.collect(); torch.cuda.empty_cache()`
- Load `BAAI/bge-reranker-v2-m3` via `AutoModelForSequenceClassification` bf16 CUDA
- For each qid, score via `_bge_score` over `repr_enriched`
- Store under `RESULTS["C_bge_v2_m3"]`

### Stage 6 — Teardown BGE, load Jina; Config D
- Free BGE
- Patch `create_position_ids_from_input_ids` into xlm_roberta module
- Load `jinaai/jina-reranker-v2-base-multilingual` (trust_remote_code, bf16, CUDA)
- For each qid, score via `_jina_score` (token-pair input, sigmoid)
- Store under `RESULTS["D_jina_v2_mul"]`

### Stage 7 — Summary
- Build fusion baseline rows from `PER_QUERY[qid]["curve"]`
- Print per-config × per-qid table with PASS/fail at `R@2000 >= 0.7`
- Compute best config per qid at R@2000
- Save `rerank_only_results.json` under `OUT_DIR`

## Results

### Phase 0 (Setup)
```
Python 3.12
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
Environment: colab
snapshot: /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot
out:      /content/drive/MyDrive/swiss_law/research/rerank_only_diagnostic

[setup] vLLM + FlashInfer installed; proceeding.
```

### Phase 1 (Warm-boot)
```
[warm-boot] loading from /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot
  ALL_QUERIES: 10
  PER_QUERY: 10  doc_meta: 255,713
[warm-boot] done in 9.5s
```

### Phase 2 (Sanity table)
```
qid         gold  in_pool   R_max  fus_R@2k   gold_needed_for_0.7
val_001       42       39   0.929     0.381          30 ( 76.9% of in-pool)
val_003       47       36   0.766     0.255          33 ( 91.7% of in-pool)
val_010       25       22   0.880     0.520          18 ( 81.8% of in-pool)
```

### Phase 3 (Config A — Qwen3 baseline)
Model-load logs only; no `[A_qwen3_raw]` per-qid recall lines were emitted in the saved notebook outputs.
```
[A] loading Qwen/Qwen3-Reranker-8B via vLLM ...
INFO 05-13 10:07:58 [weight_utils.py:615] Time spent downloading weights for Qwen/Qwen3-Reranker-8B: 39.850140 seconds
INFO 05-13 10:07:58 [weight_utils.py:904] Filesystem type for checkpoints: OVERLAY. Checkpoint size: 15.25 GiB. Available RAM: 169.30 GiB.
INFO 05-13 10:08:01 [gpu_model_runner.py:4879] Model loading took 15.26 GiB memory and 43.129371 seconds
INFO 05-13 10:08:20 [monitor.py:53] torch.compile took 18.62 s in total
INFO 05-13 10:09:11 [gpu_worker.py:440] Available KV cache memory: 62.35 GiB
INFO 05-13 10:09:11 [kv_cache_utils.py:1708] GPU KV cache size: 454,016 tokens
INFO 05-13 10:09:11 [kv_cache_utils.py:1709] Maximum concurrency for 1,024 tokens per request: 443.38x
INFO 05-13 10:09:15 [core.py:299] init engine (profile, create kv cache, warmup model) took 73.96 s (compilation: 18.62 s)
```
FlashInfer warnings:
```
WARNING:flashinfer.compilation_context:Failed to get device capability: SM 12.x requires CUDA >= 12.9.
WARNING:flashinfer.compilation_context:Failed to get device capability: SM 12.x requires CUDA >= 12.9.
```

### Phase 4 (Config B — Qwen3 fixed)
No outputs recorded in the saved notebook.

### Phase 5 (Config C — BGE-v2-m3)
Only download/loading progress bars recorded:
```
config.json:   0%|          | 0.00/795 [00:00<?, ?B/s]
tokenizer_config.json: 0.00B [00:00, ?B/s]
sentencepiece.bpe.model:   0%|          | 0.00/5.07M [00:00<?, ?B/s]
tokenizer.json:   0%|          | 0.00/17.1M [00:00<?, ?B/s]
special_tokens_map.json:   0%|          | 0.00/964 [00:00<?, ?B/s]
model.safetensors:   0%|          | 0.00/2.27G [00:00<?, ?B/s]
Loading weights:   0%|          | 0/393 [00:00<?, ?it/s]
```
No `[C_bge_v2_m3]` per-qid recall lines.

### Phase 6 (Config D — jina-v2-multilingual)
Only download progress bars recorded:
```
model.safetensors:   0%|          | 0.00/557M [00:00<?, ?B/s]
Loading weights:   0%|          | 0/153 [00:00<?, ?it/s]
```
No `[D_jina_v2_mul]` per-qid recall lines.

### Phase 7 (Summary)
No output recorded. `rerank_only_results.json` save line not emitted in the saved notebook.

### Net result
The saved notebook contains no numerical reranker R@K values for any of configs A/B/C/D, and no Phase 7 summary table or saved-results path confirmation. Only Phases 0-2 produced visible output; Phases 3-6 show model load/download logs without subsequent per-qid recall prints; Phase 7 produced nothing. The diagnostic, as preserved in the on-disk notebook outputs, did not complete to the point of producing measurements.

## Summary

The notebook is a fully-coded but apparently unfinished diagnostic that asks whether pure cross-encoder reranking on the v7.5 top-50k pool can hit `R@2000 >= 0.7` on the three hardest val queries (val_001 / val_003 / val_010) under four configurations: Qwen3-Reranker-8B raw vs enriched-and-cross-lingual, BGE-reranker-v2-m3, and jina-reranker-v2-base-multilingual. Setup (vLLM/FlashInfer install, warm-boot of the 255,713-doc snapshot, per-query target table) executed cleanly, confirming the structural targets — val_003 needs 91.7% of its in-pool gold to clear 0.7. All three reranker model loads were initiated (Qwen3 init engine took 73.96s and consumed 15.26 GiB; BGE and jina model.safetensors downloads started). No per-qid recall output, no summary table, and no `rerank_only_results.json` write confirmation are present in the saved notebook outputs, so the falsifiable pass/fail question (rerank alone reach `R@2000 >= 0.7`?) is not answered by this notebook as preserved. The artifact is therefore informative about the test design and infrastructure (sanity table, doc-representation enrichment scheme, model-swap teardown pattern) but produced no measured results.
