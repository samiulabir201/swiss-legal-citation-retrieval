# kaggle_safe_notebook_3500ee7363

**Path:** `e:\swiss_citation_extraction\notebooks\13_misc_kaggle_and_utilities\kaggle_safe_notebook_3500ee7363.ipynb`

## Configuration

The notebook defines a `Config` dataclass with the following parameters:

- `input_csv`: `/kaggle/input/competitions/llm-agentic-legal-information-retrieval/court_considerations.csv`
- `fallback_input_csv`: `court_considerations.csv`
- `output_dir`: `/kaggle/working`
- `output_jsonl`: `enriched_court_citations_10.jsonl`
- `output_preview_csv`: `enriched_court_citations_10_preview.csv`
- `output_failures_jsonl`: `enriched_court_citations_10_failures.jsonl`
- `n_rows`: 10
- `min_text_chars`: 600
- `sample_random`: False
- `random_seed`: 42
- `text_chars`: 3000 (per-citation input budget)
- `model_name`: `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1` (local Kaggle Qwen3-8B-AWQ path)
- `use_vllm_first`: True
- `force_transformers`: False
- `tensor_parallel_size`: 1
- `gpu_memory_utilization`: 0.62
- `max_model_len`: 3072
- `max_num_seqs`: 32
- `enforce_eager`: True
- `quantization`: `awq`
- `disable_custom_all_reduce`: True
- `batch_size`: 4
- `max_new_tokens`: 384
- `temperature`: 0.0
- `top_p`: 1.0
- `repetition_penalty`: 1.02
- `enable_thinking`: False
- `avoid_flashinfer`: True (Kaggle FlashInfer JIT fails to link `libcuda.so`)
- `force_triton_attention`: True

Environment: `TOKENIZERS_PARALLELISM=false`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Setup cell uninstalls FlashInfer and installs `transformers>=4.45.0`, `accelerate`, `safetensors`, `pandas`, `tqdm`, `gptqmodel`, `vllm>=0.6.0`.

## Data

- **Input**: `court_considerations.csv` resolved via a candidate list (Kaggle competition path, working dir, `/mnt/data`, or recursive glob under `/kaggle/input`).
- Columns auto-detected via `pick_column`:
  - `citation_col`: preferred names `citation`, `cite`, `court_citation`, `authority_citation`, `consideration_citation`; fallback contains-match on `citation|cite|bge`.
  - `text_col`: preferred names `text`, `consideration_text`, `paragraph_text`, `content`, `raw_text`, `body`; fallback contains-match.
- Filtering: rows with non-null citation and text, text length >= `min_text_chars` (600). First 10 substantive rows selected (random sampling disabled).
- **Outputs**:
  - `enriched_court_citations_10.jsonl` (successful enrichments)
  - `enriched_court_citations_10_preview.csv` (flattened preview)
  - `enriched_court_citations_10_failures.jsonl` (failed rows with raw output and error)

## Pipeline

1. **Setup cell (Kaggle one-shot)**: installs/upgrades transformers, accelerate, safetensors, pandas, tqdm, gptqmodel; installs vLLM; uninstalls FlashInfer and flashinfer-python; requires kernel restart.
2. **Config + env**: instantiates `Config`, creates output directory, sets tokenizers and CUDA env vars.
3. **Input resolution**: `resolve_input_path` tries known paths then globs `/kaggle/input` for `court_considerations.csv` / `court_consideration.csv`; loads with pandas.
4. **Column inference + row selection**: picks citation and text columns, filters by `min_text_chars`, selects first 10 (or random sample if enabled).
5. **Schema + prompt construction**:
   - `ENRICHMENT_SCHEMA` is a JSON schema with required fields: `legal_area`, `legal_domain_path`, `topic`, `subtopic`, `micro_topic`, `concepts_en`, `terms_original`, `statute_anchors`, `case_anchors`, `doctrinal_rule`, `legal_test`, `fact_pattern_tags`, `procedural_context`, `paragraph_role` (enum: holding/reasoning/facts/procedural_history/citation/dissent/neutral), `authority_role` (array, enums include leading_decision, legal_test, constitutional_standard, statutory_interpretation, standard_of_review, application_of_rule, distinguishing_case, background, procedural_rule, evidentiary_standard, official_intervention_limit, voting_rights_jurisprudence, neutral), `outcome_signal` (granted/dismissed/remanded/partially_granted/inadmissible/neutral), `specificity_score` (0..1).
   - `SYSTEM_PROMPT` instructs the model to act as a deterministic Swiss legal citation indexing engine; hard rules forbid synthetic questions, summaries, and invented anchors; preserves original-language legal terms in `terms_original`.
   - `make_user_prompt` formats citation + truncated text + schema JSON.
6. **JSON parsing**: `extract_json_object` strips code fences, attempts `json.loads`, falls back to balanced-brace scan with string-escape tracking.
7. **Normalization**: `normalize_enrichment` enforces required keys, coerces list fields via `as_list`, validates enums against `ROLE_VALUES`/`OUTCOME_VALUES`/`AUTHORITY_VALUES`, clamps `specificity_score` to `[0, 1]`.
8. **Retrieval views**: `build_retrieval_views` produces `semantic_concepts_en`, `topic_path`, `original_terms_view`, `statute_anchor_view`, `case_anchor_view`, `legal_rule_view`, `raw_context`.
9. **Engine loading** (`load_vllm_engine` -> `load_transformers_engine` fallback):
   - Validates local Kaggle model path, replaces with `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1` if missing.
   - vLLM attempts: explicit Triton attention via `AttentionConfig(backend="TRITON_ATTN")`, then strips unsupported kwargs progressively, ending with a conservative fallback (tp=1, mem=0.58, max_model_len=2560, max_num_seqs=16, AWQ).
   - FlashInfer guard: `builtins.__import__` monkey-patched to raise on any `flashinfer*` import.
   - Triggers fallback to Transformers on FlashInfer/libcuda/VRAM errors.
   - Transformers path: `AutoTokenizer.from_pretrained` (left padding, eos as pad), `AutoModelForCausalLM` with `device_map="auto"`, fp16 (`dtype` or legacy `torch_dtype`), `local_files_only=True`.
10. **Generation**: `generate_texts` dispatches to vLLM `llm.generate` or Transformers `model.generate` (greedy, `repetition_penalty=1.02`, `max_new_tokens=384`).
11. **Sampling params**: `build_sampling_params` prefers vLLM `StructuredOutputsParams(json=ENRICHMENT_SCHEMA)`, falls back to `GuidedDecodingParams`, then plain `SamplingParams`; returns `None` for Transformers engine.
12. **Chat template rendering**: `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` with TypeError fallback.
13. **Batch loop** (`batch_size=4` over 10 rows, `tqdm`):
    - Build prompts per batch, call `generate_texts`; on batch-level exception, write all batch rows to failures.
    - Per row: parse JSON, normalize, build retrieval views, append to records or failures.
14. **Persistence**: writes JSONL of successes, JSONL of failures, flattened preview CSV with pipe-joined list fields and view columns; prints elapsed time, rows/sec, and engine type.
15. **Final inspection cell**: pretty-prints first successful record (or first failure) truncated to 5000 chars.

## Results

No executed outputs are stored in the notebook (all code cells have empty outputs).

## Summary

This is a Kaggle smoke-test notebook that enriches 10 substantive rows from `court_considerations.csv` with query-neutral legal-retrieval metadata using Qwen3-8B-AWQ loaded from a local Kaggle dataset path. It implements a vLLM-first generation engine with explicit FlashInfer-avoidance (uninstall + import guard) and Triton-attention preference, falling back through progressively more conservative vLLM configs and finally to a Transformers + GPTQModel AWQ engine. The enrichment schema deliberately excludes synthetic user questions and summaries, instead requiring topical taxonomy, original-language legal terms, statute/case anchors, doctrinal rule, legal test, paragraph role, authority role, outcome signal, and a specificity score. Outputs go to `/kaggle/working` as JSONL (records), JSONL (failures with raw model output), and a flattened preview CSV. The notebook is unrun in this checkout, so no row counts, latencies, or sample enrichments are recorded.
