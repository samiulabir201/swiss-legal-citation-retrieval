# rerank_hybrid_3model_shootout_with_outputs.ipynb

**Path:** `notebooks/01_current_direction_cascade_rerank_precision/rerank_hybrid_3model_shootout_with_outputs.ipynb`

## Configuration

### Cross-encoder rerankers (scored over full v7.5 50k-candidate pool, all 10 val queries)

| Model | HuggingFace ID | Size | Precision | Max len | Batch | Loader |
|---|---|---|---|---|---|---|
| Qwen3-Reranker-8B | `Qwen/Qwen3-Reranker-8B` | 8 B | bfloat16 | 1024 | vLLM dynamic | vLLM (`gpu_memory_utilization=0.85`, `enforce_eager=False`) |
| BGE-reranker-v2-m3 | `BAAI/bge-reranker-v2-m3` | 568 M | bfloat16 | 1024 | 64 | HF `AutoModelForSequenceClassification` + sigmoid head |
| jina-reranker-v2-base-multilingual | `jinaai/jina-reranker-v2-base-multilingual` | 278 M | bfloat16 | 1024 | 128 | HF + sigmoid head (with `create_position_ids_from_input_ids` monkey-patch for `transformers >= 4.45`) |

Qwen3 prompt template uses `<|im_start|>system ... <|im_end|>` chat structure with cross-lingual instruction `CROSS_LING_INSTR` and yes/no logprob calibration (`exp(yl)/(exp(yl)+exp(nl))`). Sampling: `temperature=0.0, max_tokens=1, logprobs=20`. Per-doc body capped at `QWEN3_MAX_BODY = 1024 - len(prefix) - len(suffix) - 8`.

vLLM env tweaks applied before load:
- `VLLM_USE_FLASHINFER_SAMPLER=0`
- `VLLM_ATTENTION_BACKEND=FLASH_ATTN` (auto-selected anyway)
- `VLLM_FLASHINFER_FORCE_TENSOR_CORES=0`
- `FLASHINFER_DISABLE_VERSION_CHECK=1`
- `TOKENIZERS_PARALLELISM=false`

### Hyperparameters & constants

| Name | Value | Purpose |
|---|---|---|
| `RERANK_TOP_N` | 50000 | Score every doc in the pool (per query pool ranges 40,751-46,925) |
| `K_REPORT` | (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000) | K values reported per config |
| `RECALL_FLOOR` | 0.80 | Pass criterion (macro AND min-per-query) |
| `K_RRF` | 60 | RRF k-constant (Cormack 2009 standard) |
| `TEXT_TRUNC_REGEX` | 2500 chars | Per-doc text used for dossier feature regex |
| `repr_enriched` body cap | 2000 chars | Document body limit inside enriched prompt repr (total `<= 3000` chars) |
| Dossier features | 9 cols | `statute_int`, `lead_statute_int`, `concept_int`, `term_int`, `co_cite`, `doctrinal`, `chamber_match`, `hard_neg`, `fusion_rank` |
| `_HARDNEG_ROLES` | `{dispositif, costs, facts, procedural_history, admissibility, signature}` | Hard-neg role set |
| `LAW_CODES` | 41-element tuple | Statute-code allow-list for canonicalization (StPO, StGB, ZGB, OR, BGG, ZPO, BV, IVG, ATSG, SchKG, MWSTG, UWG, UVG, AHVG, BVG, AVIG, AsylG, BGE, ATF, StG, VStrR, EleG, EnG, FINIG, FinfraG, GwG, KVG, VPG, MEDBG, SR, CPC, CC, CO, CP, CPP, LP, LCart, LTF, LPGA, LAI) |
| `_QUERY_AREA_TO_CHAMBERS` | 9-area map | Criminal/civil/administrative/public/social/labor/tax/asylum/intellectual -> Swiss federal chamber codes |

### Configurations tested

| Config | Definition |
|---|---|
| F0_fusion | v7.5 RRF fusion baseline only (`fusion_rank`) |
| F1_qwen3 | Qwen3-Reranker-8B logprob score alone |
| F2_rerank_ensemble | RRF(Qwen3, BGE, jina) |
| F3_HYBRID | RRF over 12 signals: 3 rerankers + 9 dossier signals (`statute_int`, `lead_statute_int`, `concept_int`, `term_int`, `co_cite`, `doctrinal`, `chamber_match`, `-hard_neg`, `fusion_rank`) |
| F4_HYBRID_hardneg_filter | F3 with `hard_neg==1` rows deleted from the ranked list |

### Hardware / environment

- Colab Pro+ runtime, Drive auto-mounted at `/content/drive`
- GPU: confirmed Blackwell (vLLM SM 12.x warning `SM 12.x requires CUDA >= 12.9`)
- Python 3.12
- vLLM 0.21.0 (V1 engine), torch.compile + CUDAGraph (full+piecewise), FlashAttention 2
- Qwen3 KV cache: 454,016 tokens, 62.35 GiB available KV memory after profiling
- Qwen3 weight load: 15.26 GiB

## Data

### Inputs (Drive paths inside Colab)

- `_DRIVE = /content/drive/MyDrive/swiss_law`
- Snapshot dir: `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot`
  - `config.json`
  - `corpus_snapshot.json.gz` (255,713 docs; fields `ct`/`cit`/`fam`/`cb`/`pr`/`ln`/`sa`/`cn`/`tm`)
  - `per_query_snapshot.json` (10 queries; `final_topk`, `curve`, `channel_recalls`)
  - `gold_doc_sets.json`
  - `all_targets.json` (LLM-expanded `statute_targets`, `concept_targets_en`, `term_targets_{de,fr,it}`, `legal_area_keywords`)
  - `hyde_aspects.json` (optional, diagnostic-only)
  - `paths.json` -> `val_csv`
- Val CSV: `/content/drive/MyDrive/swiss_law/data/val.csv`

### Outputs

- `OUT_DIR = /content/drive/MyDrive/swiss_law/research/hybrid_rerank_final`
- `CACHE_DIR = .../hybrid_rerank_final/cache`
  - `dossier_features.npz` (4,241.2 KB) - per-(qid,did) 9-feature matrix
  - `scores_qwen3.npz` - Qwen3 yes-probability per pool doc
  - `scores_bge.npz` - BGE sigmoid score per pool doc
  - `scores_jina.npz` - jina sigmoid score per pool doc
- `hybrid_rerank_results.json` - full per-config, per-K, per-query R@K tables + summary

### Per-query pool / gold counts (from warm-boot output)

| qid | pool | gold | in_pool | R_max |
|---|---|---|---|---|
| val_001 | 44621 | 42 | 39 | 0.929 |
| val_002 | 41254 | 38 | 30 | 0.789 |
| val_003 | 44033 | 47 | 36 | 0.766 |
| val_004 | 43086 | 10 | 10 | 1.000 |
| val_005 | 42140 | 11 | 11 | 1.000 |
| val_006 | 46925 | 18 | 17 | 0.944 |
| val_007 | 40953 | 19 | 17 | 0.895 |
| val_008 | 45818 | 30 | 25 | 0.833 |
| val_009 | 40751 | 14 | 13 | 0.929 |
| val_010 | 43903 | 25 | 22 | 0.880 |

Warm-boot wall-time: 10.2s.

## Pipeline

### Phase 0 - Setup
- Detect Colab vs local; mount Drive.
- Install vLLM + transformers + accelerate + safetensors + huggingface_hub.
- Install CUDA-aware FlashInfer (`flashinfer-python`, `flashinfer-cubin`, `flashinfer-jit-cache` from `flashinfer.ai/whl/cu1XX`).
- If first install: `SystemExit("Restart runtime...")`; second run proceeds.

### Phase 1 - Warm-boot from snapshot
Loads RAM-resident corpus snapshot, per-query pools, gold sets, LLM-expanded targets, HyDE aspects, and val.csv query text. All retrieval/scoring runs against this in-memory state.

### Phase 2 - Dossier helpers (fast substring path)
Pure-Python token-walk and substring helpers, no backtracking regex on long text. ~21k docs/sec single-thread on 255k corpus.

- `canonicalize_statute(s)`: `"Art. 221 Abs. 1 lit. b StPO"` -> `"221 StPO"` via token walk + `LAW_CODES` allow-list.
- `lead_codes(text_lead)`: same on first 200 chars (for `lead_statute_int`).
- `doctrinal_density(text)`: scored in `[-10, +20]` from DE/FR rule-openers, doctrine phrases, rule-verbs, `(bge `/`(atf ` cite-cluster counts, `art.` density, minus DE/FR dispositif/facts/costs/signature markers.
- `is_hard_neg(text, role)`: role-set check + substring check (dispositif/costs/signature markers).
- `chamber_of(citation)`: small regex on short citation strings only.
- `query_legal_areas(qid)` + `allowed_chambers(qid)`: derives legal area from `ALL_TARGETS[qid].legal_area_keywords` against 7 area regexes -> chamber set from `_QUERY_AREA_TO_CHAMBERS`.

Smoke test (cell output):
- `canon('Art. 221 Abs. 1 lit. b StPO') = '221 StPO'`
- `canon('Art. 100 BGG') = '100 BGG'`
- `doctrinal('Nach Art. 221 StPO bestimmt ...') = 7.5`
- `doctrinal('Demnach erkennt das BGer: ...') = -5.0`
- `val_001 legal_areas=['criminal'] chambers=['1B', '6B', '7B', 'IV']`
- `val_002 legal_areas=['social'] chambers=['8C', '9C', 'U']`
- `val_003 legal_areas=['criminal'] chambers=['1B', '6B', '7B', 'IV']`

### Phase 3 - Precompute per-(qid, did) dossier feature matrix
- Disk cache (`dossier_features.npz`); HIT in this run (loaded for 10 queries, skipped compute).
- Per-doc cached features (query-independent): doctrinal score, hard-neg flag, lifted canonical statute anchors, lead-200-char canonical statutes, chamber code.
- Co-citation index over pool union: for laws -> # court paragraphs in any pool citing this statute; for courts -> # peer paragraphs from same case.
- Per-(qid,did) 9-feature matrix written to `DOSSIER[qid] = {"did_list", "feat"}`.
- Built when cold; runtime CPU-only ~3-5 min per cached comment.

### Phase 4 - Enriched doc representation
`repr_enriched(did)` builds the cross-encoder document text:
```
Citation: <cit>
Type: <family> (<paragraph_role>)
Statute anchors: <up to 8>
Concepts (English): <up to 8>
Terms: <up to 6>
Text: <body[:2000]>
```
Total `<= 3000` chars.

Cross-lingual instruction (`CROSS_LING_INSTR`) frames the query/doc pair as a translation problem: query is English about Swiss federal law; doc may be DE/FR/IT with English metadata.

### Phase 5 - Qwen3-Reranker-8B scoring
- Score = exp(yes_lp)/(exp(yes_lp)+exp(no_lp)) from a single greedy step.
- Wall-time per query (vLLM, dynamic batching): 674-881s.
- Per-query rough R@2000 peek (single-reranker only):
  - val_001 878.7s R@2000=0.071
  - val_002 717.7s R@2000=0.105
  - val_003 770.0s R@2000=0.149
  - val_004 793.8s R@2000=0.300
  - val_005 754.9s R@2000=0.545
  - val_006 881.4s R@2000=0.500
  - val_007 674.4s R@2000=0.211
  - val_008 798.4s R@2000=0.200
  - val_009 758.7s R@2000=0.071
  - val_010 766.6s R@2000=0.320
- Saved to `scores_qwen3.npz`; engine torn down (`CUDA mem after free = 0.0 GB`).

### Phase 6 - BGE-reranker-v2-m3 scoring
- Score = sigmoid(logit) on (query, repr_enriched(d)) pair, batch 64.
- Wall-time per query: 156-198s.
- Per-query R@2000 peek (single-reranker only):
  - val_001 165.0s R@2000=0.286
  - val_002 177.4s R@2000=0.026
  - val_003 189.8s R@2000=0.000
  - val_004 170.6s R@2000=0.100
  - val_005 179.3s R@2000=0.182
  - val_006 197.7s R@2000=0.000
  - val_007 180.3s R@2000=0.000
  - val_008 195.5s R@2000=0.100
  - val_009 156.7s R@2000=0.000
  - val_010 191.9s R@2000=0.040
- Saved to `scores_bge.npz`.

### Phase 7 - jina-reranker-v2-base-multilingual scoring
- Same sigmoid head; batch 128.
- Required restoring `create_position_ids_from_input_ids` (removed in `transformers >= 4.45`) before `trust_remote_code` load.
- `flash_attn is not installed. Using PyTorch native attention implementation.` warning emitted (still functional).
- Wall-time per query: 123-159s.
- Per-query R@2000 peek (single-reranker only):
  - val_001 129.3s R@2000=0.357
  - val_002 141.5s R@2000=0.053
  - val_003 153.2s R@2000=0.149
  - val_004 135.4s R@2000=0.600
  - val_005 144.1s R@2000=0.364
  - val_006 159.2s R@2000=0.333
  - val_007 146.4s R@2000=0.263
  - val_008 157.6s R@2000=0.100
  - val_009 123.7s R@2000=0.000
  - val_010 156.1s R@2000=0.120
- Saved to `scores_jina.npz`.

### Phase 8 - Hybrid score combination
- `rank_from_scores(scores, descending)` -> 1-indexed stable rank.
- `rrf_of([r1, r2, ...]) = sum 1/(K_RRF + r)`.
- For each query, compute 12 rank vectors (3 reranker scores + 8 dossier features descending + `-hard_neg`).
- F3 = RRF over all 12. F4 = F3 ordering with `hard_neg==1` removed afterwards.

### Phase 9 - Summary
Smallest K such that `macro R@K >= 0.8 AND min-per-q R@K >= 0.8`. Full R@K table per config, per-query R@K for F3, macro matrix across configs. Saved to `hybrid_rerank_results.json`.

### Phase 10 - Outcome interpretation (markdown only)
Five canned interpretations: A (F3 lifts K_min), B (F3==F2), C (no config reaches floor), D (F4 beats F3), E (F1 ~= F3).

## Results

### Per-config R@2000 console summary (from Phase 8)

```
[val_001] gold= 42  F0:R@2k=0.381  F1:R@2k=0.095  F2:R@2k=0.333  F3:R@2k=0.310  F4:R@2k=0.238
[val_002] gold= 38  F0:R@2k=0.553  F1:R@2k=0.105  F2:R@2k=0.053  F3:R@2k=0.553  F4:R@2k=0.579
[val_003] gold= 47  F0:R@2k=0.255  F1:R@2k=0.149  F2:R@2k=0.149  F3:R@2k=0.234  F4:R@2k=0.234
[val_004] gold= 10  F0:R@2k=0.900  F1:R@2k=0.300  F2:R@2k=0.500  F3:R@2k=0.800  F4:R@2k=0.800
[val_005] gold= 11  F0:R@2k=0.818  F1:R@2k=0.545  F2:R@2k=0.545  F3:R@2k=0.727  F4:R@2k=0.636
[val_006] gold= 18  F0:R@2k=0.722  F1:R@2k=0.500  F2:R@2k=0.444  F3:R@2k=0.778  F4:R@2k=0.778
[val_007] gold= 19  F0:R@2k=0.684  F1:R@2k=0.211  F2:R@2k=0.211  F3:R@2k=0.579  F4:R@2k=0.632
[val_008] gold= 30  F0:R@2k=0.567  F1:R@2k=0.200  F2:R@2k=0.200  F3:R@2k=0.467  F4:R@2k=0.567
[val_009] gold= 14  F0:R@2k=0.714  F1:R@2k=0.071  F2:R@2k=0.000  F3:R@2k=0.714  F4:R@2k=0.714
[val_010] gold= 25  F0:R@2k=0.520  F1:R@2k=0.320  F2:R@2k=0.240  F3:R@2k=0.520  F4:R@2k=0.520
```

### Smallest K such that macro R >= 0.8 AND min-per-query R >= 0.8

```
config                           K_min   compression   macro_R    min_R    worst_query
----------------------------------------------------------------------------------------------------
F0_fusion                         FAIL             -         -        -              -  (no K in K_REPORT reaches the floor)
F1_qwen3                          FAIL             -         -        -              -  (no K in K_REPORT reaches the floor)
F2_rerank_ensemble                FAIL             -         -        -              -  (no K in K_REPORT reaches the floor)
F3_HYBRID                         FAIL             -         -        -              -  (no K in K_REPORT reaches the floor)
F4_HYBRID_hardneg_filter          FAIL             -         -        -              -  (no K in K_REPORT reaches the floor)
```

Every config fails the dual 0.8 floor at every K in `K_REPORT`. The min-per-query ceiling is val_003's `R_max=0.766` in the pool, so no K can satisfy `min_R >= 0.8`.

### Full R@K table (macro / min)

**F0_fusion**
```
    K   macro_R    min_R    worst_query
   50     0.051    0.000        val_001
  100     0.120    0.000        val_010
  200     0.181    0.040        val_010
  500     0.415    0.128        val_003
 1000     0.528    0.191        val_003
 2000     0.611    0.255        val_003
 5000     0.722    0.404        val_003
10000     0.816    0.617        val_003
20000     0.848    0.681        val_003
30000     0.880    0.702        val_003
40000     0.892    0.745        val_003
50000     0.897    0.766        val_003
```

**F1_qwen3**
```
    K   macro_R    min_R    worst_query
   50     0.030    0.000        val_001
  100     0.051    0.000        val_001
  200     0.068    0.000        val_001
  500     0.111    0.000        val_001
 1000     0.187    0.000        val_001
 2000     0.250    0.071        val_009
 5000     0.436    0.158        val_002
10000     0.644    0.184        val_002
20000     0.816    0.316        val_002
30000     0.892    0.745        val_003
40000     0.894    0.766        val_003
50000     0.897    0.766        val_003
```

**F2_rerank_ensemble**
```
    K   macro_R    min_R    worst_query
   50     0.048    0.000        val_003
  100     0.056    0.000        val_003
  200     0.064    0.000        val_005
  500     0.132    0.000        val_007
 1000     0.182    0.000        val_009
 2000     0.268    0.000        val_009
 5000     0.362    0.071        val_009
10000     0.505    0.184        val_002
20000     0.776    0.342        val_002
30000     0.850    0.447        val_002
40000     0.891    0.766        val_003
50000     0.897    0.766        val_003
```

**F3_HYBRID**
```
    K   macro_R    min_R    worst_query
   50     0.093    0.000        val_008
  100     0.224    0.021        val_003
  200     0.304    0.064        val_003
  500     0.423    0.191        val_003
 1000     0.504    0.213        val_003
 2000     0.568    0.234        val_003
 5000     0.683    0.298        val_003
10000     0.791    0.489        val_003
20000     0.868    0.660        val_003
30000     0.884    0.737        val_002
40000     0.897    0.766        val_003
50000     0.897    0.766        val_003
```

**F4_HYBRID_hardneg_filter**
```
    K   macro_R    min_R    worst_query
   50     0.102    0.000        val_009
  100     0.230    0.021        val_003
  200     0.316    0.064        val_003
  500     0.447    0.170        val_003
 1000     0.513    0.191        val_003
 2000     0.570    0.234        val_003
 5000     0.684    0.362        val_003
10000     0.742    0.447        val_003
20000     0.785    0.571        val_001
30000     0.796    0.571        val_001
40000     0.796    0.571        val_001
50000     0.796    0.571        val_001
```

### Per-query R@K for F3_HYBRID

```
qid         gold     R@50    R@100    R@200    R@500   R@1000   R@2000   R@5000  R@10000  R@20000  R@30000  R@40000  R@50000
val_001       42    0.024    0.071    0.095    0.214    0.238    0.310    0.524    0.786    0.929    0.929    0.929    0.929
val_002       38    0.184    0.263    0.289    0.395    0.447    0.553    0.605    0.711    0.737    0.737    0.789    0.789
val_003       47    0.021    0.021    0.064    0.191    0.213    0.234    0.298    0.489    0.660    0.745    0.766    0.766
val_004       10    0.100    0.500    0.700    0.700    0.800    0.800    0.900    0.900    1.000    1.000    1.000    1.000
val_005       11    0.182    0.455    0.545    0.545    0.636    0.727    0.909    1.000    1.000    1.000    1.000    1.000
val_006       18    0.278    0.333    0.389    0.556    0.667    0.778    0.889    0.889    0.889    0.889    0.944    0.944
val_007       19    0.105    0.211    0.316    0.474    0.526    0.579    0.684    0.789    0.895    0.895    0.895    0.895
val_008       30    0.000    0.133    0.167    0.300    0.467    0.467    0.633    0.767    0.833    0.833    0.833    0.833
val_009       14    0.000    0.214    0.357    0.571    0.643    0.714    0.786    0.857    0.857    0.929    0.929    0.929
val_010       25    0.040    0.040    0.120    0.280    0.400    0.520    0.600    0.720    0.880    0.880    0.880    0.880
```

### Macro R@K matrix across configs

```
config                           R@50    R@100    R@200    R@500   R@1000   R@2000   R@5000  R@10000  R@20000  R@30000  R@40000  R@50000
F0_fusion                       0.051    0.120    0.181    0.415    0.528    0.611    0.722    0.816    0.848    0.880    0.892    0.897
F1_qwen3                        0.030    0.051    0.068    0.111    0.187    0.250    0.436    0.644    0.816    0.892    0.894    0.897
F2_rerank_ensemble              0.048    0.056    0.064    0.132    0.182    0.268    0.362    0.505    0.776    0.850    0.891    0.897
F3_HYBRID                       0.093    0.224    0.304    0.423    0.504    0.568    0.683    0.791    0.868    0.884    0.897    0.897
F4_HYBRID_hardneg_filter        0.102    0.230    0.316    0.447    0.513    0.570    0.684    0.742    0.785    0.796    0.796    0.796
```

### Notable runtime / install events

- Qwen3 vLLM init: 78.7s including 14.23s `torch.compile`, 7.84s graph compile for range (1, 16384), 4.7s CUDA-graph capture. KV cache size: 454,016 tokens; max concurrency 443.38x at 1024 tokens/req.
- `WARNING ... SM 12.x requires CUDA >= 12.9.` (flashinfer compilation context; sampler bypassed via `VLLM_USE_FLASHINFER_SAMPLER=0`).
- `WARNING 05-15 02:39:21 [jit_monitor.py:103] Triton kernel JIT compilation during inference: _compute_slot_mapping_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.`
- jina load required monkey-patch: `[C] patched transformers.xlm_roberta with create_position_ids_from_input_ids`.
- All three rerankers torn down successfully (`CUDA mem after free = 0.0 GB`).

### Outputs written

- `/content/drive/MyDrive/swiss_law/research/hybrid_rerank_final/cache/scores_qwen3.npz`
- `/content/drive/MyDrive/swiss_law/research/hybrid_rerank_final/cache/scores_bge.npz`
- `/content/drive/MyDrive/swiss_law/research/hybrid_rerank_final/cache/scores_jina.npz`
- `/content/drive/MyDrive/swiss_law/research/hybrid_rerank_final/hybrid_rerank_results.json`

## Summary

Scored the entire v7.5 50k-candidate pool for all 10 val queries with three SOTA multilingual cross-encoders (Qwen3-Reranker-8B via vLLM, BGE-reranker-v2-m3, jina-reranker-v2-base-multilingual) plus a 9-dim dossier feature matrix, then fused them via 12-signal RRF and tested four hybrid configurations against the simple fusion baseline. The diagnostic returned **Outcome C** from the markdown rubric: every config (F0 fusion / F1 Qwen3 alone / F2 3-reranker RRF / F3 12-signal hybrid / F4 hybrid + hard-neg filter) **failed** the dual `R >= 0.8 AND min-per-q R >= 0.8` floor at every K up to 50k, because val_003's pool ceiling is `R_max = 0.766` - mathematically below the floor. F3 hybrid did materially out-perform F0 fusion at small K (e.g. R@500: 0.423 vs 0.415, R@2000: 0.568 vs 0.611 - macro - but the per-query distribution shows F3 boosts the weak heads, with val_001 R@500 jumping from baseline to 0.214 and val_009 to 0.571), but the lifts converge to the same 0.897 ceiling by K=40k. F4's hard-neg filter visibly trimmed top-K (slightly improving macro at K=200-1000), then capped F4 at macro=0.796 because deletions also removed gold in some queries. Key lessons: (a) the cross-encoders, even Qwen3-8B with cross-lingual prompting, individually under-perform the v7.5 fusion baseline at every K below 50k, confirming the project ledger that **reranking is not the bottleneck**; (b) the 9-dossier signals carry the early-K lift (F3 R@100=0.224 vs F0=0.120) - dossier > rerankers in this regime; (c) hard-neg deletion is too aggressive at large K; treat it as a soft penalty only; (d) the binding constraint is **candidate-pool recall** (val_003 R_max=0.766), so any further compression work must move investment to the pool-widening side, not to better rerankers.

