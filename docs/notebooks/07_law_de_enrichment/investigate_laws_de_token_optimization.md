# investigate_laws_de_token_optimization

**Path:** `e:\swiss_citation_extraction\notebooks\07_law_de_enrichment\investigate_laws_de_token_optimization.ipynb`

## Configuration

- **Tokenizer / model name:** `Qwen/Qwen3-8B-AWQ` (loaded via `transformers.AutoTokenizer.from_pretrained(..., trust_remote_code=True)`).
- **Quantization at pilot time:** `quantization='awq_marlin'`.
- **Input file resolution (`INPUT_CANDIDATES`, first existing wins):**
  - `BASE_DIR/artifacts/law_llm_input.jsonl`
  - `BASE_DIR/data/law_llm_input.jsonl`
  - `/kaggle/input/law_llm_input.jsonl`
  - `CWD/law_llm_input.jsonl`
- **Env-aware `BASE_DIR`:** `/content/drive/MyDrive/swiss_law` (Colab), `/kaggle/working/swiss_law` (Kaggle), otherwise local Windows `e:\swiss_citation_extraction`.
- **Output config path:** `ARTIFACTS_DIR/laws_de_bucketed_config.json`.
- **Production-config baseline being investigated:** `max_text_chars=1500`, `max_new_tokens=450`, `max_model_len=3072`, `max_num_seqs=384`.
- **Filtering matches production:** `MIN_TEXT_CHARS = 20`; rows with missing `citation` or empty `text` are skipped.
- **Tokenization knobs:** `TOKENIZE_ALL = False`, `SAMPLE_SIZE = 10_000` (stratified across char buckets via `pd.cut` with edges `[0, 100, 300, 600, 1500, 3000, 1e9]`), `MEASURE_BOTH = True`, `MAX_TEXT_CHARS_CURRENT = 1500`, `MAX_TEXT_CHARS_NOTRUNC = 0`.
- **Bucketing knobs:** `NUM_BUCKETS_RAW = 17`, `PROMPT_TOK_SOURCE = 'notrunc'` (since `MEASURE_BOTH=True`), `OUTPUT_TOK_BUDGET_DEFAULT = 450`, `MAX_MODEL_LEN_ROUND = 256`, `MAX_MODEL_LEN_HEADROOM = 64`. Consecutive buckets with identical `rec_max_model_len` are merged.
- **KV-budget knobs (Qwen3-8B, 32 layers × 4 KV heads × 128 head dim):** `GPU_VRAM_GB = 95.0`, `GPU_MEM_UTIL = 0.92`, `MODEL_WEIGHTS_GB = 6.0`, `ACTIVATION_OVERHEAD_GB = 4.0`, `KV_DTYPE_BYTES = 2` (fp16), `PER_TOKEN_KV_BYTES = 2 * 32 * 4 * 128 * KV_DTYPE_BYTES = 65,536`, `MAX_NUM_SEQS_CAP = 1024`, `PREFIX_CACHED_TOKENS = PROBE_LEN`.
- **GPU pilot knobs:** `RUN_GPU_PILOT = True`, `PILOT_ROWS_PER_BUCKET = 120`, `PILOT_TEMPERATURE = 0.1`, `PILOT_TOP_P = 0.9`, `PILOT_HARD_MAX_TOKENS = 700`, `PILOT_OUTPUT_HEADROOM = 80`. Pilot engine: `tensor_parallel_size=1`, `gpu_memory_utilization=0.85`, `max_num_seqs=128`, `enable_prefix_caching=True`, `disable_log_stats=True`; `max_model_len = round_up(tok_full.max() + 700 + 64, 256)`.
- **Throughput-model knobs:** `BASELINE_RUNTIME_MIN = 75`, `BASELINE_MAX_NUM_SEQS = 384`, `BASELINE_AVG_OUT_TOKENS = 200`, `ENGINE_RELOAD_MIN = 2.5`.
- **Prompt builder reproduced verbatim from `enrich_laws_de_qwen3_8b_kaggle.ipynb` cell 4:** `LLM_SCHEMA_HINT` (13 fields including `english_summary`, `legal_rule`, `applicability_conditions`, `exceptions_or_limitations`, `legal_question`, `concepts_en`, `terms_de_to_en`, `defined_terms`, `addressees`, `sanctions_or_consequences`, `provision_role_llm`, `specificity_score`), full `SYSTEM_PROMPT` (DE/FR/IT → Swiss-legal English glossary, 11 hard rules, controlled-vocabulary `addressees`, caps on list fields), `USER_TEMPLATE` (citation, law_title, section_path, law_code, article, units, source_type, year, legal_area_hint, domain_hints, text). Helpers: `trim_text` (head-half + ` ... [TRUNCATED] ... ` + tail-half), `_coerce_str`, `_units_to_str`, `build_user_prompt`.
- **Chat templating:** `tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)` with `TypeError` fallback omitting `enable_thinking`.

## Data

- **Source:** `law_llm_input.jsonl` (JSONL, one Swiss federal law article per line; DE/FR/IT source text).
- **Expected corpus shape per the Cell 0 summary (laws_de):**
  - `<100 chars   : 29,439 rows (17.0%)`
  - `100-299 chars: 105,171 rows (60.8%)`
  - `300-599 chars: 29,776 rows (17.2%)`
  - `600-1499    :  7,440 rows (4.3%)`
  - `>=1500      :  1,207 rows (0.7%)`
- **Per-row fields consumed by the prompt builder:** `citation`, `law_title`, `title_section_path`, `text`, `structural.{law_code, article, units}`, `title_metadata.{enactment_date, enactment_year, source_type}`, `static_hints.{legal_area_static, domain_labels_en}`.
- **Streaming loader:** opens `INPUT_PATH` with UTF-8; `json.loads` each line with `try/except`; skips rows lacking `citation` or `text`, or with `len(text) < 20`. Loaded rows become `df` (pandas DataFrame), with an added `text_chars` column. A `_char_bucket` column is added with `pd.cut(bins=[0,100,300,600,1500,3000,1e9])` to drive stratified sampling.
- **Sample for tokenization:** when `TOKENIZE_ALL=False`, ~`SAMPLE_SIZE // 6` rows per char bucket are drawn with `np.random.default_rng(42)` and concatenated into `sample_df`.
- **Pilot inputs:** for each bucket, up to `PILOT_ROWS_PER_BUCKET` rows are drawn from sample rows whose prompt-token count falls in `[lo, hi)`, then rendered with `max_text_chars = bucket['rec_max_text_chars']` and submitted to vLLM.

## Pipeline

1. **Cell 1 — Setup & paths.** Detects Kaggle/Colab/local environment, sets `BASE_DIR`, searches `INPUT_CANDIDATES` for `law_llm_input.jsonl` (raises `FileNotFoundError` with hint if missing), creates `ARTIFACTS_DIR`, defines `CONFIG_OUT`.
2. **Cell 2 — Reproduce the production prompt builder verbatim.** Defines `LLM_SCHEMA_HINT`, `SYSTEM_PROMPT`, `USER_TEMPLATE`, `trim_text`, `_coerce_str`, `_units_to_str`, `build_user_prompt`. Prints scaffold sizes.
3. **Cell 3 — Load all rows from JSONL.** Streams `INPUT_PATH` line-by-line with `tqdm`, applies `MIN_TEXT_CHARS=20` and citation/text filters, builds `df` and `df['text_chars']`. Prints quantiles `p50/p75/p90/p95/p99/p99.9/p100` of `text_chars` and the counts/fractions at thresholds 1500, 3000, 6000.
4. **Cell 4 — Tokenize prompts with the real Qwen3-8B tokenizer.** Loads `AutoTokenizer.from_pretrained('Qwen/Qwen3-8B-AWQ', trust_remote_code=True)` (auto-installs `transformers>=4.51.0` if missing), measures `PROBE_LEN` for the system+schema scaffold, defines `render_chat` and `tokenize_prompts` (batched, `batch_size=256`). Builds the stratified `sample_df`, then computes `prompt_tok_current` (with `max_text_chars=1500`) and `prompt_tok_notrunc` (`max_text_chars=0`). Prints prompt-token quantiles for both series and the per-row token cost added by removing truncation.
5. **Cell 5 — Plot the distribution.** Two-panel matplotlib histogram of `prompt_tok_current` (linear with `axvline(3072)` for the current `max_model_len`, and `log10` scale).
6. **Cell 6 — Bucketing strategy.** Builds `NUM_BUCKETS_RAW=17` quantile edges over the no-truncation token series, assigns rows to buckets, computes per-bucket `prompt_p50/p99/max`, sets `rec_max_model_len = round_up(p99 + OUTPUT_TOK_BUDGET_DEFAULT + 64, 256)`, then merges consecutive buckets sharing the same `rec_max_model_len` into `merged`.
7. **Cell 7 — KV-cache budget → recommended `max_num_seqs` per bucket.** Computes `kv_pool_bytes = (95 * 0.92 - 6 - 4) GiB - PROBE_LEN * 65,536` and `kv_pool_tokens = kv_pool_bytes // 65,536`. Per bucket sets `rec_max_num_seqs = min(kv_pool_tokens // rec_max_model_len, 1024)` and records `per_slot_kv_mib`.
8. **Cell 8 — Per-bucket recommended `max_text_chars`.** Empirical `tok/char` ratio: `(prompt_tok_notrunc - PROBE_LEN) / text_chars`; reports median and p99. For each bucket extracts the sample rows whose prompt-token count falls in `[lo, hi)`, records `text_p50/p99/max`, and sets `rec_max_text_chars = ceil(text_p99 / 100) * 100`, except the longest bucket which gets `rec_max_text_chars = 0` (no truncation).
9. **Cell 9 — Optional GPU pilot.** Probes `torch.cuda.is_available()` and `import vllm`. If both succeed and `RUN_GPU_PILOT=True`, loads vLLM with `MODEL_NAME`, AWQ-Marlin, prefix-caching on, sized to hold the longest prompt + 700-token output budget, then for each bucket samples up to 120 rows, generates with `SamplingParams(temperature=0.1, top_p=0.9, max_tokens=700, repetition_penalty=1.0)`, captures `out_p50/p99/max`, and sets `rec_max_new_tokens = ceil((out_p99 + 80) / 32) * 32`. On skip, applies a heuristic fallback: `rec=256` if `prompt_p99 < 1700`, `320` if `<1900`, `384` if `<2200`, else `480`; flags `out_*` as `-1`. After pilot, re-tightens `rec_max_model_len = round_up(prompt_p99 + rec_max_new_tokens + 64, 256)` and recomputes `rec_max_num_seqs` and `per_slot_kv_mib`. Releases the engine (`del llm; gc.collect(); torch.cuda.empty_cache()`).
10. **Cell 10 — Throughput model.** Calibrates `decode_tps_per_slot` from `BASELINE_RUNTIME_MIN=75 min`, `BASELINE_MAX_NUM_SEQS=384`, `BASELINE_AVG_OUT_TOKENS=200`. Per bucket estimates wall-time with sublinear slot scaling (`eff_slots = 384 + 0.6 * (rec_max_num_seqs - 384)` when above baseline) plus `ENGINE_RELOAD_MIN=2.5` per bucket. Compares (a) measured baseline, (b) single-engine lower bound, (c) multi-engine bucketed total, and reports the per-bucket reload overhead.
11. **Cell 11 — Write recommended config.** Serializes a JSON object with `meta` (`generated_from`, `total_rows`, `sample_size`, `tokenizer`, `system_prefix_tokens`, `kv_dtype_bytes`, `kv_pool_tokens_estimated`, `pilot_ran`) and a `buckets` list of per-bucket records (`prompt_tok_lo/hi`, `row_count`, `prompt_p50/p99/max`, `text_p50/p99/max`, `out_p50/p99/max`, `rec_max_text_chars`, `rec_max_new_tokens`, `rec_max_model_len`, `rec_max_num_seqs`, `per_slot_kv_mib`, `est_runtime_min`). Writes to `ARTIFACTS_DIR/laws_de_bucketed_config.json`.
12. **Cell 12 — Recommendation summary (markdown).** Three options in order of cost: (1) per-row `max_tokens` only (no engine change), (2) split into 2–3 engine runs by length, (3) full 17-bucket plan; recommends consulting `len(merged)` and `laws_de_bucketed_config.json` to choose. Flags the heuristic-fallback case (no pilot) as conservative.

## Results

The notebook ships **without saved cell outputs**. Every code cell has `outputs: []`, so no printed quantiles, no histogram image, no per-bucket DataFrame, no pilot timings, no written `laws_de_bucketed_config.json` numbers, and no throughput estimate are recorded in this `.ipynb`.

The only verbatim numerical content available is the corpus shape stated in the Cell 0 markdown:

```
<100 chars   :  29,439 rows  (17.0%)
100-299 chars:  105,171 rows (60.8%)
300-599 chars:  29,776 rows (17.2%)
600-1499     :   7,440 rows  (4.3%)
>=1500       :   1,207 rows  (0.7%)
```

Per-token KV arithmetic stated verbatim in Cell 13 (Cell 7's markdown):

```
K + V tensors per token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
                        = 2 × 32 × 4 × 128 × bytes
                        = 32,768 × bytes

- fp16 KV: 65,536 bytes/token (64 KiB)
- fp8 KV : 32,768 bytes/token (32 KiB) — used when `cfg.kv_cache_dtype='fp8'`
```

Production-config baseline stated verbatim in Cell 0:

```
max_text_chars=1500, max_new_tokens=450, max_model_len=3072, max_num_seqs=384
```

No other run results exist in the notebook as committed.

## Summary

This is a *planning* notebook for the laws_de Qwen3-8B-AWQ enrichment run: it reproduces the production prompt builder byte-identically, tokenizes a 10k stratified sample (or optionally all 173k rows) of `law_llm_input.jsonl`, and uses the resulting prompt-token distribution to derive a per-bucket recommendation for `max_text_chars`, `max_new_tokens`, `max_model_len`, and `max_num_seqs`. The bucket plan starts at 17 quantile slices, merges consecutive buckets that round to the same `max_model_len` (vLLM-friendly `round_up` to 256), then sizes `max_num_seqs` from a KV-budget calculation (Qwen3-8B fp16 KV ≈ 65,536 bytes/token; ~80 GB usable KV on the 95 GB Blackwell). An optional GPU pilot samples ~120 rows per bucket on the real vLLM/AWQ-Marlin engine to ground `max_new_tokens` empirically rather than heuristically; without a GPU, a conservative fallback table is used. Cell 10 calibrates a throughput model from the user's measured 75-min single-config baseline (384 slots, ~200 out tokens) and compares it against single-engine per-row caps and the multi-engine bucketed plan, charging `ENGINE_RELOAD_MIN=2.5` per bucket; Cell 11 writes the full plan to `artifacts/laws_de_bucketed_config.json` and Cell 12 lays out three escalating actions (per-row `max_tokens`, 2–3-engine split, full bucket plan). The notebook does not modify the production enrichment notebook and ships without saved cell outputs, so all numerical results in this document come from the source code and Cell 0's corpus-shape table rather than an executed run.
