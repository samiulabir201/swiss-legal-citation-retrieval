# cascade_legalmalr_v1_poc.ipynb

**Path:** notebooks/01_current_direction_cascade_rerank_precision/cascade_legalmalr_v1_poc.ipynb

## Configuration

**Models**
- Reranker: `Qwen/Qwen3-Reranker-8B` (HF transformers path, bf16, SDPA attention)
- Judge: `Qwen/Qwen3-8B` (vLLM path by default; HF transformers fallback)

**Libraries**
- `transformers` (AutoTokenizer, AutoModelForCausalLM)
- `torch` (bfloat16, `@torch.inference_mode`)
- `vllm` (`LLM`, `SamplingParams`, continuous batching, prefix caching)
- `numpy`, `pandas`, `optuna` (TPE sampler), `tqdm.auto`, `gzip`, `json`, `hashlib`, `re`, `gc`

**Hardware**
- Colab Pro+ with NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.6 GB VRAM
- Reranker (~16 GB) + judge (~16 GB) co-resident; reranker is unloaded before vLLM judge load
- Setup cell deliberately avoids `import torch` before vLLM spawn to dodge a "Failed core proc(s): {}" silent crash on Blackwell SM 12.x

**Hyperparameters (CONFIG dict)**

Stage A reranker:
- `rerank_pool: 50000`
- `rerank_topn: 1000`
- `rerank_batch: 32` (legacy); `rerank_batch_v2: 64`
- `rerank_max_len: 768`
- `rerank_model: "Qwen/Qwen3-Reranker-8B"`
- `rerank_use_dossier: True`
- `rerank_dossier_chars: 1100`
- `rerank_sort_by_length: True`, `rerank_short_first: True`
- `rerank_use_sdpa: True`, `rerank_dtype: "bfloat16"`
- `rerank_chunk_save_every: 50`
- `rerank_truncate_chars: 900`

Stage B judge:
- `judge_topn: 200`
- `judge_model: "Qwen/Qwen3-8B"`
- `judge_max_new_tokens: 24`
- `judge_enable_thinking: False`
- `judge_temperature: 0.0`
- `judge_default_on_parse_fail: "no"`
- `judge_use_vllm: True`
- `judge_vllm_gpu_mem_util: 0.85`
- `judge_vllm_max_model_len: 3072`
- `judge_vllm_max_num_seqs: 256`
- `judge_vllm_dtype: "bfloat16"`
- `judge_vllm_debug: False`
- vLLM flags: `enforce_eager=True`, `disable_custom_all_reduce=True`, `enable_prefix_caching=True`, `kv_cache_dtype` left at default (FP8 KV intentionally avoided on Blackwell SM 12.0)

Cascade dossier:
- `dossier_phase: 1`
- `dossier_top_for_cocite: 200`

Multi-agent / policy:
- `policy_optimizer: "fixed"` (POC; GRPO skipped given small subset)
- `grpo_groups: 8`, `grpo_steps: 60`, `grpo_lr: 0.30`, `grpo_entropy_bonus: 0.02`
- `optuna_trials: 80`
- `k_min: 5`, `k_max: 60`

Eval:
- `k_sweep: [5,7,10,13,15,18,20,22,25,28,30,35,40,50,75,100]`
- `loo_seed: 42`

Train sanity:
- `train_sanity_n: 100`, `train_sanity_skip_grpo: True`

POC subset:
- `query_subset: ["val_001", "val_004", "val_006"]`

Artifacts:
- `artifact_dir_name: "artifacts_legalmalr"`

**Policy dimensions (12 scalars, each 5-bin discretized)**
- `w_stat_overlap: [0.0, 0.25, 0.5, 1.0, 2.0]`
- `w_channel_cov:  [0.0, 0.1, 0.25, 0.5, 1.0]`
- `w_cocite:       [0.0, 0.05, 0.1, 0.25, 0.5]`
- `prior_law:      [0.6, 0.8, 1.0, 1.2, 1.5]`
- `prior_court:    [0.6, 0.8, 1.0, 1.2, 1.5]`
- `alpha_rerank:   [0.0, 0.25, 0.5, 0.75, 1.0]`
- `alpha_judge:    [0.0, 0.25, 0.5, 0.75, 1.0]`
- `tau_cut:        [0.30, 0.45, 0.55, 0.65, 0.75]`
- `tau_yes:        [0.70, 0.80, 0.85, 0.90, 0.95]`
- `tau_no:         [0.05, 0.15, 0.25, 0.35, 0.45]`
- `k_base:         [10, 15, 20, 25, 30]`
- `k_per_target:   [0.0, 0.3, 0.5, 0.8, 1.2]`

Action space size 5^12 ≈ 2.44e8.

**Baseline policy** (used as GRPO init / fixed-policy POC):
```
w_stat_overlap=1.0, w_channel_cov=0.25, w_cocite=0.1,
prior_law=1.0, prior_court=1.0,
alpha_rerank=0.5, alpha_judge=0.5,
tau_cut=0.55, tau_yes=0.85, tau_no=0.25,
k_base=20, k_per_target=0.5
```

**Constants**
- Reranker instruction string: "Given a legal research question (in English), determine whether the Swiss legal source described in the Document is one of the citations a Swiss legal expert would cite when answering the question."
- Reranker prefix: `<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n`
- Reranker suffix: `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
- Pair format: `<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}`
- Yes/No token IDs probed from the tokenizer at load time; logits over those two are log-softmaxed and the "yes" probability is the score.
- Judge regex: `VERDICT:\s*(YES|NO)\b`, `CONF:\s*([0-9.]+)`; vLLM `stop=["\n\n"]`.

## Data

Auto-detected `DATA_ROOT` from candidates:
- `/content/drive/MyDrive/swiss_law`
- `/content/drive/MyDrive/swiss_citation`
- `/content/drive/MyDrive/swiss-citation`
- `/content/swiss_law`
- `E:/swiss_citation_extraction`

Snapshot path:
- `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot/`
  - `corpus_snapshot.json.gz` — per-doc state: `ct` (text), `cit`, `fam` (law/court/?), `cb`, `pr` (role), `ln` (language), `sa` (statute anchors), `cn` (concepts), `tm` (terms)
  - `per_query_snapshot.json` — `final_topk` (50k), curves, channel recalls, optional `channel_hit_sets`
  - `all_targets.json` — LLM-extracted `statute_targets`, `concept_targets_en`, `legal_area_keywords` per query
  - `hyde_aspects.json` — decomposed query aspects (optional)
  - `gold_doc_sets.json` — gold doc_id sets per val query (scoring only)

Standard CSVs (under `DATA_ROOT / "data"`):
- `val.csv` — 10 queries
- `train.csv` — 1139 queries
- `test.csv`

Output / cache (under `DATA_ROOT`):
- `outputs/legalmalr_cascade_v1/` — `loo_results.json`, `macro_f1.json`, `val_predictions.json`, `submission_test.csv`
- `cache_legalmalr/rerank/<qid>.json` — per-query reranker scores cache
- `cache_legalmalr/rerank_ckpt/<qid>.partial.json` — in-flight partial rerank scores (deleted on full completion)
- `cache_legalmalr/judge/<qid>/<sha1(did)>.json` — per-(qid,did) judge verdicts
- `artifacts_legalmalr/stage_a_rerank.json`, `artifacts_legalmalr/stage_b_judge.json` — aggregate per-stage dumps

## Pipeline

### Stage 0 — Setup (cells 1-5)
Probes Python / platform / `nvidia-smi` (without importing torch in parent process). Mounts Google Drive. Auto-detects `DATA_ROOT` and `SNAPSHOT_DIR` from candidate paths; aborts via `SystemExit` if snapshot missing. Builds the global `CONFIG` dict and prints it as JSON.

### Stage 1 — Warm-boot from snapshot (cells 6-8)
Loads the recall-0.89 v7.5 artifacts: `CORPUS` (gzipped), `PER_QUERY_SNAP`, `ALL_TARGETS`, `ALL_HYDE_ASPECTS` (optional), `GOLD_DOC_SETS`. Probes whether `channel_hit_sets` is present per query (`HAS_CHANNEL_HIT_SETS`); if not, degrades the channel fingerprint signal. Loads `val.csv` + `train.csv`, parses gold via `;`-split, computes distribution audit (train median gold vs val median gold — "~6x more cites per val query than train"). Applies `query_subset` filter (POC keeps only `val_001`, `val_004`, `val_006`) and asserts snapshot coverage.

### Stage 2 — Cascade dossier (cells 9-11, Phase 1 enrichments)
Builds per-(qid, did) dossier with three signals:
1. Channel-of-arrival fingerprint — which retrieval channels surfaced this doc (from `channel_hit_sets`, intersected with the rerank pool); falls back to empty if absent.
2. Statute-target intersection — `doc.sa ∩ all_targets[qid].statute_targets`, plus count.
3. Co-citation density in pool — for each statute anchor `sa` of a doc, count how many of the top-200 court-family docs in the pool also cite it; pick the max as `cocite`.
`DOSSIER[qid][did]` stores `channels`, `stat_hits`, `stat_overlap`, `cocite`, `fam`, `cit`, `cb`, `pr`, `ln`, `text` (first 600 chars of `ct`).
`render_dossier(qid, did, max_chars)` formats the citation header, channel line, statute-hit line, co-cite line, and a 600-char text excerpt — output capped to `max_chars`. This dossier text is fed both to the reranker (as `<Document>`) and to the judge (as evidence block).

### Stage 3 — LegalMALR multi-agent decomposition (cells 12-13)
Defines the 12-dim discrete policy (`POLICY_DIMS`, `BASELINE_POLICY`), `sample_policy(probs, rng)`, and `fixed_policy_values(values_dict)`. The "agents" are policy-controlled scoring moves, not separate LLMs:
- A. Statute focus (`w_stat_overlap`)
- B. Family balancer (`prior_law`, `prior_court`)
- C. Aspect router (`w_channel_cov`, `w_cocite`)
- D. K predictor (`k_base`, `k_per_target`)
Plus fusion weights `alpha_rerank`, `alpha_judge`, and threshold knobs `tau_cut`, `tau_yes`, `tau_no`.

### Stage 4 — Stage A reranking with Qwen3-Reranker-8B (cells 14-16)
Defines `Qwen3Reranker` class:
- HF `AutoModelForCausalLM`, bf16, `attn_implementation="sdpa"`.
- Prefix/suffix tokenization done once at init; `yes`/`no` token IDs cached.
- `score_batch(query, docs, instruction, ...)`: tokenizes bodies once with truncation, optional length-sorted ordering (`short_first=True` to warm CUDA on small shapes), dynamic-padded per-batch via `tok.pad`, single forward, log-softmax over `[no_id, yes_id]`, returns yes-prob list.
- Resume support: reads existing partial-checkpoint, skips already-scored indices; writes checkpoint every `save_every=50` batches.
Before load: aggressive VRAM cleanup (drops globals that look like torch tensors, `gc.collect`, `empty_cache`, `synchronize`). Loads model, then loops over `VAL_QUERIES`: for each qid, builds `docs` via `render_dossier(qid, d, max_chars=900)` over `final_topk[:50000]`, scores with `batch_size=64`, caches to `cache_legalmalr/rerank/<qid>.json`, deletes the in-flight checkpoint, prints "gold in top-1000" against `GOLD_DOC_SETS`. Dumps aggregate `stage_a_rerank.json`. Deletes reranker, GCs, `empty_cache`, `synchronize`.

### Stage 5 — Stage B judge with Qwen3-8B (cells 17-19)
Defines two judges:
- `Qwen3JudgeVLLM`: uses vLLM `LLM(...)` with `enforce_eager=True`, `disable_custom_all_reduce=True`, `enable_prefix_caching=True`. `judge_batch` renders chat template (with `enable_thinking=False`) for all items and submits one `llm.generate()` call (continuous batching).
- `Qwen3JudgeHF`: per-call HF `model.generate(do_sample=False)` fallback.
Endgame §7 Move 3 patches applied: `enable_thinking=False`, default-NO on parse failure, no "say YES" instruction, `max_new_tokens=24`. Parser regex enforces `VERDICT: YES|NO  CONF: 0-1`.
Loop over `VAL_QUERIES`: per qid, take top-200 reranked, look up per-(qid,did) cache (sha1 of did), build uncached prompts with `render_dossier(qid, did, max_chars=1400)`, submit batch to judge, write each cache file. Print yes-count, `gold_in_judged`, `gold_in_yes`. Dumps `stage_b_judge.json`. Deletes judge, GCs, `empty_cache`.

### Stage 6 — Cascade policy + GRPO (cells 20-23)
`policy_score(qid, did, p)`:
- `rerank = RERANK_SCORES[qid][did]`
- `judge_signal = conf if verdict=="yes" else 1-conf` (0.5 if un-judged)
- `fam_prior = prior_law | prior_court | 1.0`
- `dossier_score = w_stat_overlap·(stat_overlap/5) + w_channel_cov·(len(channels)/6) + w_cocite·(cocite/20)`
- `fused = (alpha_rerank·rerank + alpha_judge·judge_signal + (1 - alpha_rerank - alpha_judge)·dossier_score) · fam_prior`
`predict(qid, p)`:
- `K = clip(k_base + k_per_target · |statute_targets|, k_min, k_max)`
- Auto-no zone: drop if judge verdict==no and conf ≥ `tau_yes`.
- Auto-yes shortcut: keep if judge verdict==yes and conf ≥ `tau_yes` (these win regardless of score).
- Apply `tau_cut` threshold to the rest, sort by fused score desc, take top-K (after forced).
`f1_for(qid, predicted_dids)` — set F1 vs `GOLD_DOC_SETS[qid]`.
`grpo_train(...)` — categorical per-dim distributions, group-relative advantage update on log-logits, entropy bonus pulling toward uniform, deterministic argmax readout.
`optuna_train(...)` — TPE fallback over the categorical bin grid.

### Stage 7 — Leave-one-out across 10 val queries (cell 24)
For each held-out qid: train on remaining 9 via GRPO / Optuna / fixed (per `policy_optimizer`), score both `argmax_vals` and `best_vals` on the held-out query, store. Compute `macro_argmax` and `macro_best` over LOO_RESULTS. Prints "Target: 0.6 - 0.7".

### Stage 8 — Train sanity check (cells 25-26)
If any train queries appear in the warm-boot snapshot: aggregate LOO policies via mode-snap to valid bins (with median fallback), run `predict` on each train query in the snapshot, compute a string-overlap F1 proxy against parsed train gold (gold strings vs predicted citation strings). Flag "RED FLAG" if train F1 > val F1; otherwise "OK". Marked as diagnostic only.

### Stage 9 — Final predictions on val + K-sweep (cells 27-29)
Per-query table of gold count, predicted count, TP, P, R, F1 for argmax LOO policies. Computes a mode-snap mean policy across LOO folds; runs `predict_fixed_k` over `k_sweep` and prints macro F1 at each K.

### Stage 10 — Save artifacts + optional test inference (cells 30-32)
Writes `loo_results.json` (per-fold policies + F1 + preds), `macro_f1.json` (`macro_argmax`, `macro_best`, `policy_optimizer`, `n_val`), `val_predictions.json` (per-query citation strings deduped). Threshold messages: `[OK]` if macro_argmax ≥ 0.6, `[close]` if ≥ 0.4, `[gap]` otherwise.
If any test queries are present in snapshot + DOSSIER: run `predict` with `mean_policy`, map dids → citation strings, write `submission_test.csv`; else print notice.

### Stage 11 — Teardown (cell 33)
Deletes `reranker` and `judge` from globals, GCs, `empty_cache`, prints final VRAM.

## Results

All 34 cells in the notebook have `OUTPUTS COUNT=0` — the notebook has not been executed. No `stream`, `execute_result`, `display_data`, or `error` outputs are present. There are no recorded metrics, timings, VRAM probes, gold-in-top-N counts, judge YES-rates, baseline F1s, GRPO trajectories, LOO macro F1, K-sweep results, train sanity F1s, or saved artifacts in the notebook file.

## Summary

This notebook is a POC scaffold for a 3-stage cascade — cascade dossier (channel fingerprint + statute-target intersection + co-citation density) → Qwen3-Reranker-8B over 50k candidates → Qwen3-8B vLLM judge over top-200 → 12-scalar policy fused score with variable-K, optimized either as a fixed baseline, by Optuna TPE, or by a custom GRPO over discrete bins, evaluated via leave-one-out across the 10 val queries. It warm-boots from the recall-0.89 v7.5 snapshot (no retrieval re-execution), uses `dossier_use_dossier=True` so both reranker and judge see pre-digested evidence rather than raw German legal text (per endgame §4.4 law-family bug), and applies the endgame §7 Move 3 judge patches (`enable_thinking=False`, default-NO, `max_new_tokens=24`, strict VERDICT regex). The current run is configured as a POC: `policy_optimizer="fixed"`, `query_subset=["val_001","val_004","val_006"]`, target macro F1 0.6-0.7. The notebook ships unexecuted, so there are no measured numbers, no observed failures, and no validated lessons — only architectural decisions baked into CONFIG defaults (bf16 reranker at batch 64, vLLM judge with eager mode + prefix caching, no FP8 KV on Blackwell, reranker unloaded before judge load, all expensive stages cached so GRPO rollouts are pure re-weights).
