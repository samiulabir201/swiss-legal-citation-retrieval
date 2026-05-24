# kaggle_qwen3_awq_text_to_json_no_flashinfer_with_fallback

**Path:** e:\swiss_citation_extraction\notebooks\13_misc_kaggle_and_utilities\kaggle_qwen3_awq_text_to_json_no_flashinfer_with_fallback.ipynb

## Configuration

Notebook targets Kaggle T4 runtimes with Qwen3-8B-AWQ. Cell 1 installs `pandas`, `tqdm`, `transformers`, `accelerate`, `safetensors`, `gptqmodel`, and attempts to install `vllm`. It then explicitly uninstalls `flashinfer`, `flashinfer-python`, and `flashinfer-python-cu12` to avoid the Kaggle `cannot find -lcuda` JIT linker crash.

Cell 2 sets environment variables before importing vLLM:
- `TOKENIZERS_PARALLELISM=false`
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7`
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN`
- `OMP_NUM_THREADS=1`

`Config` dataclass:
- `input_csv`: `/kaggle/input/competitions/llm-agentic-legal-information-retrieval/court_considerations.csv`
- `fallback_input_csv`: `court_consideration.csv`
- `output_dir`: `/kaggle/working`
- `output_jsonl`: `enriched_court_citations_10.jsonl`
- `output_preview_csv`: `enriched_court_citations_10_preview.csv`
- `n_rows`: 10
- `random_seed`: None
- `sample_random`: False
- `text_chars`: 3200
- `model_name`: local Kaggle path `/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-awq/1` if it exists, otherwise `Qwen/Qwen3-8B-AWQ`
- `engine`: `"auto"` (options: `auto`, `vllm`, `transformers`)
- `tensor_parallel_size`: 1
- `gpu_memory_utilization`: 0.62
- `max_model_len`: 3072
- `enforce_eager`: True
- `max_num_seqs`: 32
- `quantization`: `awq`
- `disable_custom_all_reduce`: True
- `force_triton_attention`: True
- `batch_size`: 8
- `max_new_tokens`: 384
- `temperature`: 0.0
- `top_p`: 1.0
- `repetition_penalty`: 1.02
- `enable_thinking`: False

## Data

Input is the Kaggle competition CSV `court_considerations.csv` (with fallback names and a `/kaggle/input/**/court_consideration*.csv` glob search). Cell 4 infers a citation column (preferred names `citation`, `cite`, `court_citation`, `authority_citation`, `consideration_citation`; substring fallback `citation|cite|bge`) and a text column (preferred `text`, `consideration_text`, `paragraph_text`, `content`, `raw_text`, `body`; substring fallback `text|content|paragraph|consideration|body`).

Rows are filtered to those with non-null citation and text and a text length over 30 characters after stripping. Either the head N rows (`sample_random=False`, default) or a random sample is taken; default `n_rows=10`. The selected slice is reset and the original index preserved as `_source_row`.

## Pipeline

Cell 5 defines the enrichment JSON schema with 17 required fields: `legal_area`, `legal_domain_path`, `topic`, `subtopic`, `micro_topic`, `concepts_en`, `terms_original`, `statute_anchors`, `case_anchors`, `doctrinal_rule`, `legal_test`, `fact_pattern_tags`, `procedural_context`, `paragraph_role`, `authority_role`, `outcome_signal`, `query_phrases_en`, `summary_en`.

Controlled vocabularies:
- `paragraph_role`: holding, reasoning, facts, procedural_history, citation, dissent, neutral
- `outcome_signal`: granted, dismissed, inadmissible, remitted, partial, none
- `authority_role`: leading_decision, settled_rule, application, distinguishing, procedural, background, none

The system prompt instructs Qwen3 to act as a deterministic Swiss legal text-to-JSON engine returning exactly one JSON object, English classification fields, and only facts/statutes/cases supported by the input text. The user prompt embeds the citation, the (truncated to 3200 chars) text, and the inlined schema with explicit rules.

`extract_json_object` strips markdown fences and `<think>...</think>` blocks, tries `json.loads`, and falls back to a brace-balanced scan with string/escape awareness. `normalize_enrichment` cleans strings (whitespace-collapsed, length-capped), deduplicates lists (max items per field varies: 6 for `legal_domain_path`, 14 for `concepts_en`, 18 for `terms_original`, 10 for anchors and fact tags, 8 for `query_phrases_en`), and validates the controlled-vocabulary fields against their enum sets (invalid `paragraph_role` -> `neutral`; invalid `outcome_signal` -> `none`; empty `authority_role` -> `["none"]`).

Cell 6 (`JsonEngine`) loads the tokenizer, then:
1. If `engine` is `auto` or `vllm`, attempts to load vLLM with the configured kwargs. Tries to attach an explicit `AttentionConfig(backend="TRITON_ATTN")` via `vllm.config` or `vllm.config.attention` (with lowercase fallback). If `LLM(...)` rejects `attention_config` with `TypeError`, retries without it.
2. On vLLM failure, prints `repr` and traceback (limit 5), runs CUDA cleanup (`gc.collect`, `torch.cuda.empty_cache`, `torch.cuda.ipc_collect`), and in `auto` mode falls back to Transformers + GPTQModel AWQ (`AutoModelForCausalLM.from_pretrained` with `device_map="auto"`, `dtype=torch.float16`, `low_cpu_mem_usage=True`).

`generate` renders prompts via `tokenizer.apply_chat_template(..., enable_thinking=False)` (with a `TypeError` fallback for tokenizers without that kwarg). For vLLM it uses `SamplingParams(temperature=0.0, top_p=1.0, max_tokens=384, repetition_penalty=1.02, stop=["<|im_end|>", "</s>"])`. For Transformers it tokenizes with padding/truncation to `max_model_len`, calls `model.generate` greedy (`do_sample=False`), and slices off the prompt before decoding.

Cell 7 iterates `work_df` in batches of `batch_size=8`, calls `engine.generate`, parses each raw output, normalizes it, and appends to `records` or `failures`. Outputs are written to `/kaggle/working/enriched_court_citations_10.jsonl`, a preview CSV `enriched_court_citations_10_preview.csv`, and (if any) `enriched_court_citations_10_failures.jsonl`. Cell 8 prints the first record as pretty JSON (truncated to 4000 chars) or the first failure.

## Results

No execution outputs are present in the notebook (every cell has empty `outputs`).

## Summary

This is a self-contained Kaggle-T4 enrichment notebook that converts Swiss court-consideration paragraphs into a fixed 17-field English JSON schema using Qwen3-8B-AWQ. The defining engineering choice is robustness on Kaggle: FlashInfer is uninstalled after vLLM install to avoid the `-lcuda` JIT crash, an explicit Triton attention backend is requested, and a complete Transformers + GPTQModel AWQ fallback path runs greedily when vLLM cannot start. Generation is deterministic (`temperature=0.0`, `repetition_penalty=1.02`, `max_new_tokens=384`, `max_model_len=3072`, `gpu_memory_utilization=0.62`, `enforce_eager=True`), text is truncated to 3200 characters, and JSON parsing tolerates markdown fences, `<think>` blocks, and unbalanced output via a manual brace scanner. The default run processes the first 10 rows in batches of 8 and writes JSONL plus a preview CSV (and a failures JSONL if any). The notebook was authored but not executed in the saved state, so no row-throughput, success-rate, or sample enrichment metrics are recorded.
