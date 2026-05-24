# exp_A3_qwen32b_enumeration

**Path:** `e:\swiss_citation_extraction\notebooks\11_early_experiments_exp_A\exp_A3_qwen32b_enumeration.ipynb`

## Configuration

- LLM: `Qwen/Qwen3-32B` loaded via vLLM in bf16, `gpu_memory_utilization=0.90`, `max_model_len=4096`, `enforce_eager=False`.
- Qwen3 thinking mode disabled via `chat_template_kwargs={'enable_thinking': False}`.
- Sampling defaults: `temperature=0.2`, `top_p=0.9`, `max_tokens=800`; HyDE generation uses `temperature=0.3, max_tokens=700`; Enum generation uses `temperature=0.1, max_tokens=600`.
- Retriever/encoder: `BAAI/bge-m3` (`BGEM3FlagModel`) with `use_fp16=True`; embeddings encoded with `batch_size=4`, `max_length=2048` (1024 for enum), dense vectors only (`return_dense=True`, `return_sparse=False`, `return_colbert_vecs=False`).
- Hardware: NVIDIA RTX PRO 6000 Blackwell Server Edition, 95 GB VRAM; asserts `>= 78 GB`. PyTorch 2.10.0 + CUDA 12.8.
- Env: `VLLM_NO_USAGE_STATS=1`; `sys.stdout.fileno` / `sys.stderr.fileno` monkey-patched as a Colab workaround.
- Runtime patches before importing `FlagEmbedding`: mock `transformers.utils.import_utils.is_torch_fx_available` and stub `transformers.models.jina_embeddings_v3` module.
- Dependency installs: uninstall existing FlagEmbedding then `pip install -qU FlagEmbedding pandas numpy`; `pip install vllm`; `pip install --upgrade transformers -q`. Subsequent troubleshooting cells attempt `pip install vllm --extra-index-url https://download.vllm.ai/whl/torch2.10.0+cu128` and `...torch2.4.0+cu128`.
- Prompts (identical to Exp-2):
  - HYDE system: `Du bist Schweizer Jurist. Schreibe deutsche Texte im Stil Schweizer Bundesgesetze.`
  - HYDE user: requests three hypothetical German paragraphs in Swiss federal-law style (OR, ZGB, StGB, StPO, ZPO, BGG, etc.), 2–4 sentences each, separated by `---`.
  - ENUM system: `You are a Swiss legal expert.` knowing OR, ZGB, StGB, StPO, ZPO, BGG, BV, IPRG, SchKG, DBG, StHG, etc.
  - ENUM user: list 15 Swiss federal law citations using `Art. X Abs. Y ABBR` or `Art. X ABBR` with a 4–8 word reason separated by ` — `, one per line.
- RRF fusion: `k_rrf=60`, `topk=2000`, evaluation cutoffs `ks=(50, 200, 500, 1000)`.
- Stat-citation filter `is_stat`: excludes anything starting with `BGE ` or matching `^\d[A-Z]_` / `^[A-Z]\d[A-Z]_`.
- Gate: `stat_recall@500 >= 0.40`.

## Data

- Mount: Google Drive at `/content/drive`; root `ROOT = /content/drive/MyDrive/swiss_law/data`; artifacts at `ART = ROOT/artifacts`.
- Inputs:
  - `val.csv` — 10 validation queries with `query_id`, `query`, `gold_citations`.
  - `laws_de.csv` — corpus metadata (column `citation`).
  - `val_translated_de.pkl` — DE translations of queries.
  - `artifacts/laws_bgem3.npy` — corpus embeddings, shape `(175933, 1024)`, cast to `float32` (asserted present, produced by Exp-1).
- Outputs written:
  - `artifacts/exp_A3_expansions.json` — per-query `query_en`, `query_de`, `hyde`, `enum`.
  - `artifacts/query_vecs_A3.npz` — `q_en`, `q_de`, `q_hyde`, `q_enum`, `query_ids`.
  - `artifacts/exp_A3_report.json` — full results with `meta`.

## Pipeline

1. Install vLLM, FlagEmbedding, transformers, pandas, numpy with runtime patches and ABI troubleshooting.
2. Mount Drive, assert presence of `laws_bgem3.npy`, verify GPU has `>= 78 GB` VRAM.
3. Load `val.csv`, `laws_de.csv`, `val_translated_de.pkl`, citation list, and corpus dense vectors.
4. Set `VLLM_NO_USAGE_STATS=1`; load `Qwen/Qwen3-32B` via vLLM (bf16, thinking off); define `chat(messages, **kw)` helper using `llm.chat` with `SamplingParams`.
5. Build HyDE and Enum prompts from `(HYDE_SYS, HYDE_USER)` and `(ENUM_SYS, ENUM_USER)` per query.
6. For each of the 10 validation rows: generate HyDE text (`temperature=0.3, max_tokens=700`) and Enum text (`temperature=0.1, max_tokens=600`); store in `expansions[query_id]`; save to `exp_A3_expansions.json`.
7. Hallucination diagnostic: count verbatim presence of statutory gold citations inside Enum text.
8. Delete the vLLM model, run `gc.collect()` + `torch.cuda.empty_cache()` to free VRAM.
9. Apply FlagEmbedding compatibility patches and load `BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)`; encode EN queries, DE queries, HyDE expansions, and Enum expansions to dense vectors.
10. Save query vectors to `query_vecs_A3.npz`.
11. Rank corpus by inner product `q @ doc_emb.T` (negated argsort) for `en`, `de`, `hyde`, `enum`.
12. RRF fusion variants: `en+hyde`, `en+enum`, `hyde+enum`, `en+hyde+enum`, `en+de+hyde+enum (all4)`.
13. Evaluate eight variants (three singles + five fusions) at `stat_recall@{50,200,500,1000}` using the statutory gold subset; aggregate as sum-of-hits over sum-of-gold.
14. Save `exp_A3_report.json` with metadata; print A/B table vs Exp-2 (7B best `0.295`) and Exp-1 (`0.215`); declare PASS/STILL BELOW against the `0.40` gate.

## Results

Setup outputs:
- `corpus: (175933, 1024) | val: 10`
- `GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (95 GB)`
- `VLLM_NO_USAGE_STATS set to 1`
- `Installed PyTorch version: 2.10.0+cu128` / `Installed CUDA version: 12.8`

vLLM init warnings (Cell 4):
- pip dependency conflicts (`sentence-transformers 3.1.1 requires transformers<5.0.0,>=4.38.0, but you have transformers 5.5.4`; `gradio 5.50.0 requires pandas<3.0,>=1.0, but you have pandas 3.0.2`).
- `INFO 04-21 15:04:40 [utils.py:233] non-default args: {'dtype': 'bfloat16', 'max_model_len': 4096, 'disable_log_stats': True, 'model': 'Qwen/Qwen3-32B'}`
- `INFO 04-21 15:04:41 [model.py:549] Resolved architecture: Qwen3ForCausalLM`
- `INFO 04-21 15:04:41 [model.py:1678] Using max model len 4096`
- `INFO 04-21 15:04:41 [scheduler.py:238] Chunked prefill is enabled with max_num_batched_tokens=16384.`
- `INFO 04-21 15:04:41 [vllm.py:790] Asynchronous scheduling is enabled.`
- `WARNING 04-21 15:04:43 [system_utils.py:152] We must use the spawn multiprocessing start method. Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. ... Reasons: CUDA is initialized`

Generation (Cell 6):
- `INFO 04-21 15:05:36 [hf.py:314] Detected the chat template content format to be 'string'.`
- Per-query lengths:
  - `val_001: HyDE 1421c | Enum 760c`
  - `val_002: HyDE 1790c | Enum 694c`
  - `val_003: HyDE 1320c | Enum 1045c`
  - `val_004: HyDE 1539c | Enum 620c`
  - `val_005: HyDE 1510c | Enum 757c`
  - `val_006: HyDE 1538c | Enum 973c`
  - `val_007: HyDE 1378c | Enum 751c`
  - `val_008: HyDE 1619c | Enum 684c`
  - `val_009: HyDE 1597c | Enum 744c`
  - `val_010: HyDE 1586c | Enum 809c`
- `total gen time: 319.5s`
- `Verbatim gold-in-enum: 6/149 = 4.0% (Exp-2 / 7B was 7/251 = 2.8%)`

Memory free (Cell 7):
- `free vram: 7.8 GB`

Retrieval evaluation (Cell 9):
- EN only — `@50=0.074, @200=0.121, @500=0.215, @1000=0.302`
- HyDE only — `@50=0.148, @200=0.262, @500=0.322, @1000=0.369`
- Enum only — `@50=0.107, @200=0.242, @500=0.315, @1000=0.403`
- RRF(en, hyde) — `@50=0.134, @200=0.262, @500=0.302, @1000=0.383`
- RRF(en, enum) — `@50=0.094, @200=0.208, @500=0.315, @1000=0.416`
- RRF(hyde, enum) — `@50=0.161, @200=0.315, @500=0.376, @1000=0.456`
- RRF(en, hyde, enum) — `@50=0.174, @200=0.315, @500=0.369, @1000=0.436`
- RRF(en, de, hyde, enum) — `@50=0.128, @200=0.275, @500=0.356, @1000=0.409`

Final A/B (Cell 10):

```
variant               rec@50    @200    @500   @1000
------------------------------------------------------------
en_only                0.074   0.121   0.215   0.302
hyde_only              0.148   0.262   0.322   0.369
enum_only              0.107   0.242   0.315   0.403
en+hyde                0.134   0.262   0.302   0.383
en+enum                0.094   0.208   0.315   0.416
hyde+enum              0.161   0.315   0.376   0.456
en+hyde+enum           0.174   0.315   0.369   0.436
all4                   0.128   0.275   0.356   0.409

best: hyde+enum -> stat_recall@500 = 0.376
vs Exp-2 (Qwen2.5-7B) best = 0.295; gate 0.40 -> STILL BELOW
vs Exp-1 (no LLM)          = 0.215
```

## Summary

Exp-A3 swaps the Qwen2.5-7B generator from Exp-2 for `Qwen/Qwen3-32B` in bf16 via vLLM (thinking mode off) while keeping the BGE-M3 retriever, prompts, RRF fusion, and the 10-query validation set identical. After heavy vLLM/PyTorch ABI troubleshooting and FlagEmbedding import patches, generation completes in 319.5 s and produces HyDE plus Enum expansions for all 10 queries. The verbatim gold-in-enum hallucination check rises only slightly, from `7/251 = 2.8%` on 7B to `6/149 = 4.0%` on 32B. The best retrieval variant `RRF(hyde, enum)` reaches `stat_recall@500 = 0.376`, beating Exp-2's `0.295` and Exp-1's `0.215` but still below the `0.40` gate, which the script labels `STILL BELOW`.
