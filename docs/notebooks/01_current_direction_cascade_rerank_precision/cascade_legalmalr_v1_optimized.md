# cascade_legalmalr_v1_optimized.ipynb

**Path:** notebooks/01_current_direction_cascade_rerank_precision/cascade_legalmalr_v1_optimized.ipynb

## Configuration

### Models
- **Reranker:** `Qwen/Qwen3-Reranker-8B` (HF transformers path, bf16, SDPA attention)
- **Judge:** `Qwen/Qwen3-8B` (vLLM backend by default; HF transformers fallback class `Qwen3JudgeHF`)
- **Reranker instruction:** `"Given a legal research question (in English), determine whether the Swiss legal source described in the Document is one of the citations a Swiss legal expert would cite when answering the question."`
- **Judge system prompt:** auditor role; "Be strict: only YES if there is concrete, on-point evidence."
- **Judge user template:** `QUESTION:\n{query}\n\nCANDIDATE SOURCE (dossier):\n{dossier}\n\nOutput exactly two lines, no other text, no thinking:\nVERDICT: YES or NO\nCONF: a number 0.00 to 1.00\n`
- **Verdict parser regex:** `VERDICT:\s*(YES|NO)\b` and `CONF:\s*([0-9.]+)`; default-NO on parse fail; conf clipped to [0,1].

### Libraries
- `torch`, `transformers` (`AutoTokenizer`, `AutoModelForCausalLM`)
- `vllm` (`LLM`, `SamplingParams`)
- `numpy`, `pandas`, `optuna`, `tqdm.auto`
- `gzip`, `json`, `hashlib`, `re`, `gc`, `os`, `pathlib`, `collections.Counter`/`defaultdict`, `statistics`
- `google.colab.drive` (Colab mount; falls back when not in Colab)

### Hyperparameters (CONFIG dict)
- **Stage A reranker**
  - `rerank_pool = 50000`
  - `rerank_topn = 1000`
  - `rerank_batch = 32` (legacy default; `rerank_batch_v2 = 64` overrides)
  - `rerank_max_len = 768`
  - `rerank_model = "Qwen/Qwen3-Reranker-8B"`
  - `rerank_use_dossier = True`
  - `rerank_dossier_chars = 1100`
  - `rerank_batch_v2 = 64`
  - `rerank_sort_by_length = True`
  - `rerank_use_sdpa = True`
  - `rerank_dtype = "bfloat16"`
  - `rerank_chunk_save_every = 50`
  - `rerank_short_first = True`
  - `rerank_truncate_chars = 900`
- **Stage B judge**
  - `judge_topn = 200`
  - `judge_model = "Qwen/Qwen3-8B"`
  - `judge_max_new_tokens = 24`
  - `judge_enable_thinking = False`
  - `judge_temperature = 0.0`
  - `judge_default_on_parse_fail = "no"`
  - `judge_use_vllm = True`
  - `judge_vllm_gpu_mem_util = 0.85`
  - `judge_vllm_max_model_len = 3072`
  - `judge_vllm_max_num_seqs = 256`
  - `judge_vllm_dtype = "bfloat16"`
  - `judge_vllm_debug = False`
- **Cascade dossier**
  - `dossier_phase = 1`
  - `dossier_top_for_cocite = 200`
- **Policy / GRPO**
  - `policy_optimizer = "grpo"` (options: `"grpo" | "optuna" | "fixed"`)
  - `grpo_groups = 8`
  - `grpo_steps = 60`
  - `grpo_lr = 0.30`
  - `grpo_entropy_bonus = 0.02`
  - `optuna_trials = 80`
  - `k_min = 5`
  - `k_max = 60`
- **Eval**
  - `k_sweep = [5,7,10,13,15,18,20,22,25,28,30,35,40,50,75,100]`
  - `loo_seed = 42`
- **Train sanity**
  - `train_sanity_n = 100`
  - `train_sanity_skip_grpo = True`
- **Artifacts**
  - `artifact_dir_name = "artifacts_legalmalr"`

### Policy dims (12 scalars, each 5 discrete bins)
- `w_stat_overlap`: [0.0, 0.25, 0.5, 1.0, 2.0]
- `w_channel_cov`: [0.0, 0.1, 0.25, 0.5, 1.0]
- `w_cocite`: [0.0, 0.05, 0.1, 0.25, 0.5]
- `prior_law`: [0.6, 0.8, 1.0, 1.2, 1.5]
- `prior_court`: [0.6, 0.8, 1.0, 1.2, 1.5]
- `alpha_rerank`: [0.0, 0.25, 0.5, 0.75, 1.0]
- `alpha_judge`: [0.0, 0.25, 0.5, 0.75, 1.0]
- `tau_cut`: [0.30, 0.45, 0.55, 0.65, 0.75]
- `tau_yes`: [0.70, 0.80, 0.85, 0.90, 0.95]
- `tau_no`: [0.05, 0.15, 0.25, 0.35, 0.45]
- `k_base`: [10, 15, 20, 25, 30]
- `k_per_target`: [0.0, 0.3, 0.5, 0.8, 1.2]

### Baseline policy (GRPO init and `"fixed"` baseline)
`w_stat_overlap=1.0, w_channel_cov=0.25, w_cocite=0.1, prior_law=1.0, prior_court=1.0, alpha_rerank=0.5, alpha_judge=0.5, tau_cut=0.55, tau_yes=0.85, tau_no=0.25, k_base=20, k_per_target=0.5`

### Reranker prompt scaffolding
- Prefix: `<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n`
- Suffix: `<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
- Per-pair body: `<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}`
- Score = `log_softmax([no_logit, yes_logit])[1].exp()` over the final position.

### Hardware
- Colab Pro+ on NVIDIA RTX PRO 6000 Blackwell, 95.6 GB VRAM.
- Reranker (~16 GB) + judge (~16 GB) co-resident; reranker unloaded before vLLM judge load (`del reranker; gc.collect; torch.cuda.empty_cache; torch.cuda.synchronize`).
- GPU probed via `nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version` to avoid eager `import torch` which is documented to break vLLM V1 worker spawn on Blackwell SM 12.x.
- vLLM config notes: `enable_prefix_caching=True`, `tensor_parallel_size=1`, `enforce_eager=False`, `disable_custom_all_reduce=True`, `disable_log_stats=True`, `kv_cache_dtype` intentionally NOT set to fp8 (silent worker death reproduced on Blackwell SM 12.0 + AWQ Marlin).

### Path roots (auto-detect, first existing)
- `/content/drive/MyDrive/swiss_law`
- `/content/drive/MyDrive/swiss_citation`
- `/content/drive/MyDrive/swiss-citation`
- `/content/swiss_law`
- `E:/swiss_citation_extraction`

## Data

### Snapshot (warm-boot inputs from recall-0.89 v7.5 multi-query run)
- `SNAPSHOT_DIR = DATA_ROOT / "/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot/"`
- `snapshot_dir/corpus_snapshot.json.gz` — per-doc state (fields: `ct` text, `cit` citation string, `fam` family, `cb`, `pr`, `ln`, `sa` statute anchors, `cn` concepts, `tm` terms)
- `snapshot_dir/per_query_snapshot.json` — `final_topk` (50k), curves, channel recalls; optionally `channel_hit_sets` per query
- `snapshot_dir/all_targets.json` — LLM-extracted statute/concept/keyword targets per query (`statute_targets`, `concept_targets_en`, `legal_area_keywords`)
- `snapshot_dir/hyde_aspects.json` — decomposed query aspects (optional)
- `snapshot_dir/gold_doc_sets.json` — gold doc_id sets per val query

### CSVs
- `DATA_ROOT / "data" / "val.csv"` — 10 queries, columns `query_id, query, gold_citations` (gold parsed by splitting on `;`)
- `DATA_ROOT / "data" / "train.csv"` — 1139 queries
- `DATA_ROOT / "data" / "test.csv"` — test inference

### Output directories (created under DATA_ROOT)
- `outputs/legalmalr_cascade_v1/` (`out_dir`)
- `cache_legalmalr/` (`cache_dir`)
  - `cache_legalmalr/rerank/<qid>.json` — final per-query rerank scores
  - `cache_legalmalr/rerank_ckpt/<qid>.partial.json` — in-progress reranker checkpoint, removed on success
  - `cache_legalmalr/judge/<qid>/<sha1(did)>.json` — per-(qid, did) judge verdict records
- `artifacts_legalmalr/` (`artifact_dir`)
  - `artifacts_legalmalr/stage_a_rerank.json` — aggregate stage A snapshot
  - `artifacts_legalmalr/stage_b_judge.json` — aggregate stage B snapshot

### Final saved outputs (`out_dir`)
- `loo_results.json`
- `macro_f1.json`
- `val_predictions.json`
- `submission_test.csv` (only if test queries are in the warm-boot)

## Pipeline

### Stage 1 — Setup (cells 0-5)
1. GPU probe via subprocess `nvidia-smi`; deliberately no `import torch` yet (Blackwell vLLM spawn fix).
2. Mount Google Drive (`google.colab.drive.mount`); fall back gracefully outside Colab.
3. Auto-detect `DATA_ROOT` from `CANDIDATE_ROOTS`; locate `SNAPSHOT_DIR` and create `out_dir`, `cache_dir`, `artifact_dir`.
4. Dump the full `CONFIG` dict.

### Stage 2 — Warm-boot from snapshot (cells 6-8)
1. Load `corpus_snapshot.json.gz` into `CORPUS` (dict of doc_id -> per-doc record).
2. Load `per_query_snapshot.json` into `PER_QUERY_SNAP`; `all_targets.json` into `ALL_TARGETS`; `hyde_aspects.json` into `ALL_HYDE_ASPECTS` (optional, empty dict if missing); `gold_doc_sets.json` into `GOLD_DOC_SETS` (as sets).
3. Detect `HAS_CHANNEL_HIT_SETS` flag (per-query, per-channel doc-id sets).
4. Read `val.csv`, `train.csv`. Parse gold by splitting on `;`. Build `VAL_QUERIES` list of `{query_id, query, gold}`. Distribution audit logs train vs val median gold and prints "~6x more cites per val query than train. Train is NOT representative."

### Stage 3 — Cascade dossier, Phase 1 enrichments (cells 9-11)
For each `qid` in `PER_QUERY_SNAP`:
1. Take `final_topk[:50000]` as the candidate pool.
2. Compute `stat_targets` = set of strings from `ALL_TARGETS[qid].statute_targets`.
3. Channel-of-arrival fingerprint: if `HAS_CHANNEL_HIT_SETS`, accumulate per-doc `channels = [channel names that surfaced this doc]`; else leave empty.
4. Co-citation density: over `final_topk[:200]` court-family docs, `Counter` of statute anchors `doc["sa"]`. For each pool doc, the doc's own "best anchor" cocite count = `max_a cocite_count[a]` for `a` in the doc's `sa`.
5. Per-doc dossier record: `{channels, stat_hits = sorted(sa ∩ stat_targets), stat_overlap = |stat_hits|, cocite, fam, cit, cb, pr, ln, text}` stored in `DOSSIER[qid][did]`.
6. `render_dossier(qid, did, max_chars)` produces a ~250-token block: header `[cit] (ln, fam, cb, role=pr)`, optional `surfaced by: ch1, ch2, … — N/15 channels`, `cites: {stat_hits} ← N of your M query targets` (or `cites: none of your M query targets`), `cited by N of your top-200 court paragraphs`, then `text: "<first 600 chars>"`.

### Stage 4 — Multi-agent decomposition (cells 12-13)
Four "agents" expressed as scoring weights, all controlled by 12-scalar policy `p`:
- A. **Statute focus** — `w_stat_overlap`
- B. **Family balancer** — `prior_law`, `prior_court`
- C. **Aspect router** — `w_channel_cov`, `w_cocite`
- D. **K predictor** — `k = clip(k_base + k_per_target * |statute_targets|, k_min, k_max)`

Plus fusion `alpha_rerank`, `alpha_judge`, cut `tau_cut`, and routing `tau_yes`, `tau_no`. Action space = 5^12 ≈ 2.44e8 (never enumerated; sampled by GRPO). Helpers: `sample_policy(probs, rng)`, `fixed_policy_values(values_dict)`.

### Stage 5 — Stage A: Qwen3-Reranker-8B over 50k (cells 14-16)
1. Class `Qwen3Reranker` loads model with bf16 + SDPA, builds `prefix_ids` / `suffix_ids` once, resolves `yes` / `no` first-token IDs.
2. `score_batch(query, docs, instruction, batch_size=64, sort_by_length=True, short_first=True, checkpoint_path, save_every=50)`:
   - Resume from JSON checkpoint list if present.
   - Tokenize body without max-length warning; truncate to `max_len - len(prefix) - len(suffix)`.
   - Sort by length, short-first (warms CUDA on small shapes, better tqdm UX).
   - For each batch: `input_ids = prefix + body + suffix`, dynamic-pad to longest in batch, forward, take last-position logits at `yes_id` / `no_id`, log-softmax, store `prob_yes`.
   - Checkpoint every `save_every` batches; final checkpoint at end; raise if any missing.
3. Driver loop over `VAL_QUERIES`: load cache if present else render dossier as the doc text (`render_dossier(qid, did, max_chars=900)`), score, write `cache_legalmalr/rerank/<qid>.json`, delete partial checkpoint, log `gold in top-1000`.
4. Pre-load VRAM cleanup: walk globals removing GPU tensors, `gc.collect`, `torch.cuda.empty_cache`, `synchronize`, print free/total. Same teardown after the loop (`del reranker`, etc.).
5. Write aggregate artifact `artifacts_legalmalr/stage_a_rerank.json` with `queries`, `rerank_topn`, per-query `summaries` (`n_scored`, `top1000_gold_recall`, `gold_total`, `top10_dids`), and the full `rerank_scores` map.

### Stage 6 — Stage B: Qwen3-8B judge over top-200 (cells 17-19)
1. Two classes:
   - `Qwen3JudgeVLLM`: vLLM `LLM(...)` with prefix caching, dtype bf16, `gpu_memory_utilization=0.85`, `max_model_len=3072`, `max_num_seqs=256`. `SamplingParams(temperature=0, top_p=1, max_tokens=24, stop=["\n\n"])`. Chat template applied with `enable_thinking=False`. `judge_batch(items)` calls `llm.generate(prompts, sampling_params)` in one batched call.
   - `Qwen3JudgeHF`: per-call fallback; same sampling math.
2. Parser `_parse_judge_output(text)` returns `(verdict, conf, raw)` with default-NO and conf=0.5 on parse fail; conf 0.7 / 0.3 when verdict matches but no CONF.
3. Driver loop: per `qid`, sort `final_topk` by rerank score, take top 200, load cached records, build uncached `(query, dossier(max_chars=1400))` list, single `judge_batch(uncached_items)` call, write per-(qid,did) JSON `{verdict, conf}`. Logs `YES count`, `gold_in_judged`, `gold_in_yes`.
4. Write aggregate artifact `artifacts_legalmalr/stage_b_judge.json`. Then `del judge; gc.collect; torch.cuda.empty_cache`.

### Stage 7 — Cascade scoring and GRPO policy training (cells 20-23)
1. `policy_score(qid, did, p)`:
   - `rerank = RERANK_SCORES[qid][did]`
   - `judge_signal = conf if verdict == "yes" else (1 - conf)`; neutral 0.5 if un-judged
   - `fam_prior = p["prior_law"]` if `fam == "law"`, else `p["prior_court"]` if `fam == "court"`, else 1.0
   - `dossier_score = p["w_stat_overlap"] * (stat_overlap / 5.0) + p["w_channel_cov"] * (len(channels) / 6.0) + p["w_cocite"] * (cocite / 20.0)`
   - `fused = (p["alpha_rerank"]*rerank + p["alpha_judge"]*judge_signal + (1 - alpha_rerank - alpha_judge) * dossier_score) * fam_prior`
2. `predict(qid, p)`:
   - candidates = judged docs only
   - `k = clip(k_base + k_per_target * n_statute_targets, k_min, k_max)`
   - auto-no: drop docs with judge verdict NO and conf >= `tau_yes`
   - auto-yes: keep docs with judge verdict YES and conf >= `tau_yes` (force-keep)
   - non-forced docs sorted by `s`, kept only if `s >= tau_cut`, then take top-k of `forced + rest_kept`.
3. `f1_for(qid, predicted)` — standard set F1 vs `GOLD_DOC_SETS[qid]`.
4. Baseline F1 logged per-query under `BASELINE_POLICY`.
5. `grpo_train(train_qids, n_steps, n_groups, lr, entropy_bonus, seed)`:
   - Init per-dim categoricals biased to `BASELINE_POLICY` bin (+1.5 mass).
   - For each step: sample `n_groups` policies, evaluate mean F1 over `train_qids`, group-normalize advantages, update logits by `lr * adv / n_groups`, blend toward uniform by `entropy_bonus`.
   - Track `best_reward`, `best_idx`, `best_vals`; return `argmax_vals` (deterministic readout) + `history` + final `probs`.
6. `optuna_train(train_qids, n_trials, seed)` — TPE over categorical dims; fallback when `policy_optimizer="optuna"`.

### Stage 8 — Leave-one-out evaluation (cell 24)
For each held-out val qid: train on 9, run `grpo_train` (or `optuna_train` or `BASELINE_POLICY` for `"fixed"`) seeded as `loo_seed + hash(held)%1000`. Score held-out with both `argmax_vals` and `best_vals` (best group seen during training). Aggregate `macro_argmax` and `macro_best` over 10 folds.

### Stage 9 — Train sanity check (cells 25-26)
1. Filter `train_df` rows whose `query_id` is in `PER_QUERY_SNAP`. Most warm-boots are val-only -> "no train queries in the warm-booted snapshot" notice; skip.
2. If any: pick the **mode** of each policy dim across LOO results, snap to nearest valid bin (fix for `np.median` producing off-bin values like 0.625). Compute proxy F1 by string-overlap on `info["cit"]` vs parsed gold (train gold uses citation strings, not doc_ids).
3. If `train F1 > val F1`: red flag for overfit dossier features.

### Stage 10 — Final reporting and artifacts (cells 27-33)
1. Per-query LOO table: `gold | pred | TP | P | R | F1` plus `MACRO` line.
2. Fixed-K sweep with mode-snap policy over `k_sweep`, ignoring `k_base`/`k_per_target`; prints `K  macro F1` table.
3. Persist `loo_results.json`, `macro_f1.json`, `val_predictions.json` (citation strings via `DOSSIER[qid][did]["cit"]`, dedup-preserving order).
4. Status banner: `[OK]` if `macro_argmax >= 0.6`, `[close]` if `>= 0.4`, else `[gap]`.
5. Optional test inference: only fires if any test `query_id` is in `PER_QUERY_SNAP` and `DOSSIER`; else prints instructions to re-run the recall-0.89 pipeline on test then merge snapshots; otherwise writes `submission_test.csv` with mode-LOO policy.
6. Final teardown: `del reranker, judge` if present; `gc.collect`; `torch.cuda.empty_cache`; print VRAM.

## Results

The notebook has **no executed outputs** — every one of the 34 cells reports `"outputs": 0`. No printed numbers, no tqdm logs, no errors, no saved artifact contents are present in the .ipynb. The file is configuration-and-code only; nothing has been measured by running this notebook in its current saved form.

## Summary

This notebook is the optimized v1 of the LegalMALR cascade: warm-boot from the recall-0.89 v7.5 snapshot, then run Qwen3-Reranker-8B (HF, bf16, SDPA, length-sorted batches, checkpointed) over 50k -> top-1000, then Qwen3-8B vLLM judge (prefix-cached, `enable_thinking=False`, max_new=24, default-NO) over top-200, fuse with a 12-scalar GRPO-trained policy (4 "agents" + fusion weights + thresholds + variable-K), and LOO-evaluate across 10 val queries with macro F1 target 0.6-0.7. The code embeds endgame-handoff fixes verbatim: dossier-as-document feed (fix for `law:*` family scoring 0.0), default-NO parser (fix for 87% rubber-stamp YES), and Blackwell-specific guards (no premature `import torch`, no fp8 KV cache, explicit VRAM teardown between reranker and judge). The notebook is checked in unexecuted, so there are no measured numbers, no GRPO learning curves, and no LOO F1 to report. Lessons recorded inside the notebook itself: train distribution (median gold 2, 99% DE) is ~6x off from val (median 22, 100% EN) so train must be a sanity-only signal, and dossier-as-evidence is the binding fix for the broken endgame run rather than further prompt-engineering.
