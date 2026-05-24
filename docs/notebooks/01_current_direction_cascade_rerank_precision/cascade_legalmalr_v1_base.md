# cascade_legalmalr_v1_base.ipynb

**Path:** `notebooks/01_current_direction_cascade_rerank_precision/cascade_legalmalr_v1_base.ipynb`

## Configuration

### Hardware (reported by cell 2 output)
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- VRAM: 95.0 GB
- Compute capability: 12.0
- Python: 3.12.13
- Torch: 2.10.0+cu128
- CUDA: 12.8
- Platform: Linux-6.6.122+-x86_64-with-glibc2.35 (Colab)

### Models
- Stage A reranker: `Qwen/Qwen3-Reranker-8B` (bfloat16, `device_map="cuda:0"`)
  - Yes/no logit head; `yes_id=9693`, `no_id=2152`
  - Pairwise prompt: `<Instruct>: ... \n<Query>: ... \n<Document>: ...` with Qwen chat prefix/suffix and `<think>\n\n</think>` stub.
- Stage B judge: `Qwen/Qwen3-8B` (bfloat16, `device_map="cuda:0"`)
  - `enable_thinking=False`, `do_sample=False`, greedy.
  - Strict regex output: `VERDICT: YES|NO  CONF: 0.00-1.00`.
  - Default-on-parse-fail = `"no"`.

### Libraries
- `transformers` (`AutoTokenizer`, `AutoModelForCausalLM`)
- `torch`
- `pandas`, `numpy`
- `optuna` (fallback optimizer)
- `gzip`, `json`, `hashlib`, `re`, `time`, `gc`
- `google.colab.drive`

### CONFIG dict (cell 5)
```
rerank_pool:           50000
rerank_topn:           1000
rerank_batch:          32
rerank_max_len:        1024
rerank_model:          "Qwen/Qwen3-Reranker-8B"
rerank_use_dossier:    True
rerank_dossier_chars:  1100
judge_topn:            200
judge_model:           "Qwen/Qwen3-8B"
judge_max_new_tokens:  24
judge_enable_thinking: False
judge_temperature:     0.0
judge_default_on_parse_fail: "no"
dossier_phase:         1
dossier_top_for_cocite: 200
policy_optimizer:      "grpo"  # "grpo" | "optuna" | "fixed"
grpo_groups:           8
grpo_steps:            60
grpo_lr:               0.30
grpo_entropy_bonus:    0.02
optuna_trials:         80
k_min:                 5
k_max:                 60
k_sweep:               [5, 7, 10, 13, 15, 18, 20, 22, 25, 28, 30, 35, 40, 50, 75, 100]
loo_seed:              42
train_sanity_n:        100
train_sanity_skip_grpo: True
```

### Policy dimensions (12 scalars, 5 discrete bins each — action space 5^12 ≈ 2.44e8)
| Dim | Bins (low→high) |
|---|---|
| `w_stat_overlap` | 0.0, 0.25, 0.5, 1.0, 2.0 |
| `w_channel_cov` | 0.0, 0.1, 0.25, 0.5, 1.0 |
| `w_cocite` | 0.0, 0.05, 0.1, 0.25, 0.5 |
| `prior_law` | 0.6, 0.8, 1.0, 1.2, 1.5 |
| `prior_court` | 0.6, 0.8, 1.0, 1.2, 1.5 |
| `alpha_rerank` | 0.0, 0.25, 0.5, 0.75, 1.0 |
| `alpha_judge` | 0.0, 0.25, 0.5, 0.75, 1.0 |
| `tau_cut` | 0.30, 0.45, 0.55, 0.65, 0.75 |
| `tau_yes` | 0.70, 0.80, 0.85, 0.90, 0.95 |
| `tau_no` | 0.05, 0.15, 0.25, 0.35, 0.45 |
| `k_base` | 10, 15, 20, 25, 30 |
| `k_per_target` | 0.0, 0.3, 0.5, 0.8, 1.2 |

### Baseline policy (GRPO init + "fixed" fallback)
```
{'w_stat_overlap': 1.0, 'w_channel_cov': 0.25, 'w_cocite': 0.1,
 'prior_law': 1.0, 'prior_court': 1.0,
 'alpha_rerank': 0.5, 'alpha_judge': 0.5,
 'tau_cut': 0.55, 'tau_yes': 0.85, 'tau_no': 0.25,
 'k_base': 20, 'k_per_target': 0.5}
```

### Stated reranker instruction
> "Given a legal research question (in English), determine whether the Swiss legal source described in the Document is one of the citations a Swiss legal expert would cite when answering the question."

### Judge system prompt
> "You are a Swiss legal-citation auditor. Given a legal research question and a candidate Swiss legal source (with structured evidence about where it was retrieved from, which query statute targets it cites, and how often it is co-cited by top court paragraphs), you must decide whether a Swiss legal expert would cite this exact source when answering the question. Be strict: only YES if there is concrete, on-point evidence."

## Data

`DATA_ROOT = /content/drive/MyDrive/swiss_law` (resolved at runtime; auto-detect among `/content/drive/MyDrive/swiss_law`, `/content/drive/MyDrive/swiss_citation`, `/content/drive/MyDrive/swiss-citation`, `/content/swiss_law`, `E:/swiss_citation_extraction`).

| Path | Role |
|---|---|
| `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot/` | Warm-boot snapshot dir (recall-0.89 v7.5 multi-query run) |
| `snapshot/corpus_snapshot.json.gz` | Per-doc state (text `ct`, citation `cit`, family `fam`, anchors `sa`, concepts `cn`, terms `tm`, role `pr`, codebook `cb`, lang `ln`). Loaded: 255,713 docs (3.7 s) |
| `snapshot/per_query_snapshot.json` | 10 queries; per-query `final_topk` (50k) + curves. `channel_hit_sets` absent (per-doc per-channel info missing — dossier degrades to empty channel lists) |
| `snapshot/all_targets.json` | 10 queries; LLM-extracted `statute_targets`, `concept_targets_en`, `legal_area_keywords` per query |
| `snapshot/hyde_aspects.json` | 10 queries; decomposed query aspects |
| `snapshot/gold_doc_sets.json` | Per-query gold doc-id sets — 254 gold docs across 10 queries |
| `/content/drive/MyDrive/swiss_law/data/val.csv` | 10 queries |
| `/content/drive/MyDrive/swiss_law/data/train.csv` | 1,139 queries |
| `/content/drive/MyDrive/swiss_law/data/test.csv` | Used only by optional Phase 10 cell |
| `/content/drive/MyDrive/swiss_law/outputs/legalmalr_cascade_v1/` | Output dir |
| `/content/drive/MyDrive/swiss_law/cache_legalmalr/` | Cache root |
| `cache_legalmalr/rerank/<qid>.json` | Reranker score cache (per-query) |
| `cache_legalmalr/judge/<qid>/<sha1(did)>.json` | Judge verdict cache (per-doc) |

### Output artifacts (Phase 10, not yet produced)
- `outputs/legalmalr_cascade_v1/loo_results.json`
- `outputs/legalmalr_cascade_v1/macro_f1.json`
- `outputs/legalmalr_cascade_v1/val_predictions.json`
- `outputs/legalmalr_cascade_v1/submission_test.csv` (only if test snapshot present)

### Gold counts (from cell 8 output)
- val gold count per query: `[42, 36, 47, 10, 11, 18, 19, 29, 14, 25]`
- snapshot gold counts: `[42, 38, 47, 10, 11, 18, 19, 30, 14, 25]`
- train median gold = 2.0 (mean 4.1)
- val median gold = 22.0 (mean 25.1)
- "~6x more cites per val query than train. Train is NOT representative."

## Pipeline

### Phase 1 — Setup (cells 0–5)
- Print Python/torch/CUDA/GPU info.
- Mount Google Drive twice (cells 3 and 4).
- Resolve `DATA_ROOT` from candidate roots; locate snapshot dir; create out/cache dirs.
- Print fully-resolved `CONFIG` dict.

### Phase 2 — Warm-boot from snapshot (cells 6–8)
- Load `corpus_snapshot.json.gz` → `CORPUS` (255,713 docs).
- Load `per_query_snapshot.json` → `PER_QUERY_SNAP` (10 queries).
- Load `all_targets.json`, `hyde_aspects.json`, `gold_doc_sets.json`.
- Detect `HAS_CHANNEL_HIT_SETS` — output: `False`.
- Load `val.csv` and `train.csv`; parse `gold_citations` by `;`; build `VAL_QUERIES`.
- Print distribution audit (train vs val median gold).

### Phase 3 — Cascade dossier, Phase 1 enrichments (cells 9–11)
Per `research/cascade_dossier_plan.md`. Three signals:
1. **Channel-of-arrival fingerprint** — list of channels surfacing each doc (degrades to empty when `channel_hit_sets` absent — current state).
2. **Statute-target intersection** — `set(doc.sa) ∩ set(all_targets[qid].statute_targets)`.
3. **Co-citation density** — for each statute anchor cited by court docs in top-200 of `final_topk`, count how many of those top-200 court paras cite it.

Output structure `DOSSIER[qid][did]` carries: `channels`, `stat_hits`, `stat_overlap`, `cocite`, `fam`, `cit`, `cb`, `pr`, `ln`, `text`.

`render_dossier(qid, did, max_chars)` formats this as a ~250-token block fed to both reranker (as "document") and judge (as "evidence block"). Format example printed by cell 11:
```
[Art. 10 Abs. 1 StGB]  (?, law, -, role=-)
cites: none of your 20 query targets
text: "Art. 10 Abs. 1 StGB This article distinguishes ..."
```

### Phase 4 — LegalMALR multi-agent decomposition (cells 12–13)
Four conceptual "agents" implemented as a single scoring function parameterized by the 12-scalar policy:
- Statute focus (`w_stat_overlap`)
- Family balancer (`prior_law`, `prior_court`)
- Aspect router (`w_channel_cov`, `w_cocite`)
- K predictor (`k_base`, `k_per_target` → `k = clip(k_base + k_per_target · |statute_targets|, k_min, k_max)`)

Plus fusion (`alpha_rerank`, `alpha_judge`), cut (`tau_cut`), routing (`tau_yes`, `tau_no`).

### Phase 5 — Stage A reranker scoring (cells 14–16)
- Class `Qwen3Reranker` with `score_batch` returning per-doc `P(yes)` via log-softmax over `{yes, no}` token logits at last position.
- Loop val queries; if `cache_legalmalr/rerank/<qid>.json` exists, load; else score 50k dossier texts in batches of 32.
- After scoring, free reranker VRAM (`del reranker; gc.collect(); torch.cuda.empty_cache()`).
- **Interrupted by KeyboardInterrupt during execution of cell 16** before any per-query cache file was produced.

### Phase 6 — Stage B judge (cells 17–19)
- Class `Qwen3Judge` builds messages via `apply_chat_template(..., enable_thinking=False)`.
- For each query, take top `judge_topn=200` from reranker scores; cache per `(qid, did)` to `cache_legalmalr/judge/<qid>/<sha1(did)>.json`.
- Free judge VRAM after.

### Phase 7 — Cascade policy + GRPO (cells 20–24)
- `policy_score(qid, did, p)`: fuse rerank prob, judge `conf` (signed by verdict), and dossier score (weighted sum of `stat_overlap/5`, `len(channels)/6`, `cocite/20`) multiplied by `fam_prior`.
- `predict(qid, p)`:
  - auto-no zone: drop docs where judge said NO with `conf >= tau_yes`.
  - auto-yes shortcut: force-keep docs where judge said YES with `conf >= tau_yes`.
  - sort remaining by fused score, keep ones above `tau_cut`, then top-`k`.
- `grpo_train(train_qids, n_steps, n_groups, lr, entropy_bonus, seed)`:
  - Per-dim categorical bin distributions, initialized biased toward `BASELINE_POLICY`.
  - Each step: sample `n_groups=8` policies, evaluate on the 9 train queries (mean F1), compute group-normalized advantage, update logits with `lr · adv / n_groups`, blend toward uniform by `entropy_bonus`.
  - Returns argmax policy + best-seen-group policy + history + final probs.
- `optuna_train(...)`: TPE sampler over categorical bins, n_trials, returns best params.
- LOO loop over 10 val queries: each fold trains on the other 9 with GRPO (or Optuna or fixed), evaluates on the held-out one. Stores `policy_argmax`, `policy_best`, `f1_argmax`, `f1_best`, `preds_argmax`, `preds_best`, `train_reward`.

### Phase 8 — Train sanity check (cells 25–26)
- Filter `train_df` for query_ids present in the warm-boot snapshot. Notebook documents that this is expected to be empty for val-only snapshots (cell prints a notice and skips).
- Uses LOO **median** policy. Computes a string-overlap F1 proxy (predicted citation strings ∩ parsed train gold strings).
- Flags `train F1 > val F1` as a red flag for overfit.

### Phase 9 — Final predictions on val + K-sweep (cells 27–29)
- Print per-query table of gold/pred/TP/P/R/F1 for LOO-argmax predictions.
- Run fixed-K sweep over `K ∈ {5,7,…,100}` using mean-LOO policy (mean for numeric dims, mode for categorical).

### Phase 10 — Save artifacts + optional test (cells 30–33)
- Strip non-serializable bits from `LOO_RESULTS`; write `loo_results.json`, `macro_f1.json`, `val_predictions.json`.
- Print success/close/gap message based on `macro_argmax` vs 0.6 / 0.4 thresholds.
- Test inference: only runs if test queries appear in the warm-boot snapshot — otherwise prints a notice with re-run instructions.
- Final teardown: `del` reranker/judge, `gc.collect()`, `torch.cuda.empty_cache()`, print VRAM allocated.

## Results

Of the 34 cells, only cells 2–16 produced outputs; cells 17 onward were never executed (no outputs stored).

### Cell 2 (environment)
```
Python:  3.12.13
Torch:   2.10.0+cu128
CUDA:    12.8
Platform: Linux-6.6.122+-x86_64-with-glibc2.35
GPU:     NVIDIA RTX PRO 6000 Blackwell Server Edition
VRAM:    95.0 GB
Compute: 12.0
```

### Cell 4 (paths)
```
DATA_ROOT = /content/drive/MyDrive/swiss_law
SNAPSHOT_DIR = /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot
```

### Cell 7 (warm-boot)
```
corpus_snapshot: 255,713 docs  (3.7s)
per_query_snap:  10 queries
all_targets:     10 queries
hyde_aspects:    10 queries
gold_doc_sets:   254 total gold docs across 10 queries
channel_hit_sets present per query: False
```

### Cell 8 (data audit)
```
val.csv:   10 queries
train.csv: 1139 queries
val gold count per query: [42, 36, 47, 10, 11, 18, 19, 29, 14, 25]
snapshot gold counts:     [42, 38, 47, 10, 11, 18, 19, 30, 14, 25]
train  median gold = 2.0 (mean 4.1)
val    median gold = 22.0  (mean 25.1)
⇒ ~6x more cites per val query than train. Train is NOT representative.
```

### Cell 10 (dossier sample)
```
Built dossier for 10 queries.
Sample (first val query, first 3 candidates):
  law:128343                 fam=law    stat_overlap=0  cocite=0  channels=[]
  law:57057                  fam=law    stat_overlap=0  cocite=0  channels=[]
  law:57137                  fam=law    stat_overlap=0  cocite=0  channels=[]
```

### Cell 11 (rendered dossier sample)
```
[Art. 10 Abs. 1 StGB]  (?, law, -, role=-)
cites: none of your 20 query targets
text: "Art. 10 Abs. 1 StGB This article distinguishes between crimes and offenses based on the severity of penalties threatened. criminal acts penalties severity crime offense penalty severity criminal law classification verbrechen vergehen strafe crime offense penalty"

length: 350 chars
```

### Cell 13 (policy space)
```
policy dims: 12
action space: 2.44e+08
baseline:    {'w_stat_overlap': 1.0, 'w_channel_cov': 0.25, 'w_cocite': 0.1,
              'prior_law': 1.0, 'prior_court': 1.0,
              'alpha_rerank': 0.5, 'alpha_judge': 0.5,
              'tau_cut': 0.55, 'tau_yes': 0.85, 'tau_no': 0.25,
              'k_base': 20, 'k_per_target': 0.5}
```

### Cell 15 (reranker class defined)
```
(reranker class defined; load deferred to next cell to make this cell cheap)
```

### Cell 16 — Reranker scoring (run terminated)
- Reranker loaded: `yes_id=9693  no_id=2152`.
- Warning emitted: `transformers/tokenization_utils_base.py:2402: UserWarning: max_length is ignored when padding=True and there is no truncation strategy. To pad to max length, use padding='max_length'`.
- **Cell raised `KeyboardInterrupt`** inside `reranker.score_batch(...)` at `torch.log_softmax(stacked, dim=1)[:, 1].exp().cpu().tolist()` — user aborted before the first query's 50,000 candidates finished scoring.
- No `RERANK_SCORES` were written to cache; downstream cells (17–33) were never executed and produced no outputs.

### Cells 17–33
Empty (no `outputs` arrays). Therefore no judge scores, no GRPO rollouts, no LOO macro F1, no K-sweep numbers, no saved artifacts, no test submission produced.

## Summary

This notebook is the base implementation of the LegalMALR cascade v1 pipeline: warm-boot from the recall-0.89 v7.5 snapshot, build a Phase-1 cascade dossier (channel fingerprint + statute-target intersection + co-citation density), score the 50k pool with Qwen3-Reranker-8B (dossier as document), judge the top-1000 with Qwen3-8B (endgame §7 Move 3 prompt patches), then learn a 12-scalar policy via leave-one-out GRPO with an Optuna fallback. The setup phases (1–4) executed cleanly — environment correctly identifies the 95 GB Blackwell GPU, the snapshot warm-boots in ~4 s, and the dossier is built for all 10 val queries; the cell-8 audit confirms the well-known train/val distribution shift (val median gold = 22 vs train = 2). What did not work: cell 7 reports `channel_hit_sets present per query: False`, so the dossier's channel-fingerprint signal degrades to empty lists (visible in cell 10 sample where `channels=[]` for all candidates), and the dossier sample in cell 11 shows `stat_overlap=0` and no co-citation density on its example doc — the dossier text is mostly the 600-char excerpt. The run was aborted by the user with `KeyboardInterrupt` inside the first reranker batch of cell 16, so no F1 number, no GRPO rollout, and no saved artifact was produced by this notebook execution. Lessons: (a) the warm-boot snapshot is missing per-doc per-channel info needed for the cascade plan's full Phase-1 dossier and would need a re-save from cell 48 of the recall-0.89 notebook to fix permanently; (b) reranking 50k dossier-as-document pairs per query with Qwen3-Reranker-8B at `batch=32, max_len=1024` is slow enough that the run was abandoned mid-batch — a measured cost-per-query needs to be established before this pipeline can be re-attempted end-to-end.
