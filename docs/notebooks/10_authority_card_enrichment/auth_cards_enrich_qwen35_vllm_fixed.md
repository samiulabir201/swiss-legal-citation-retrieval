# auth_cards_enrich_qwen35_vllm_fixed

**Path:** e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen35_vllm_fixed.ipynb

## Configuration

- Model: `Qwen/Qwen3.5-35B-A3B` (HF checkpoint, ~71.9 GB safetensors, BF16).
- Runtime: vLLM offline inference (`LLM` + `llm.generate`), text-only workload, no Transformers `model.generate`.
- GPU target: single high-VRAM NVIDIA GPU (e.g. RTX PRO 6000 Blackwell / 96 GB VRAM class).
- Quantization: `None` (BF16).
- `GPU_MEMORY_UTIL = 0.90`
- `MAX_MODEL_LEN = 4096`
- `BATCH_SIZE = 4`
- `TEMPERATURE = 0.15`
- `MAX_TOKENS = 600`
- `TENSOR_PARALLEL = 1`
- `USE_MTP = True`, `MTP_TOKENS = 1` (MTP-1 speculative decoding per vLLM Qwen3.5 latency-focused recipe).
- `ENABLE_PREFIX_CACHING = False`
- `LANGUAGE_MODEL_ONLY = True`; falls back to `limit_mm_per_prompt = {"image": 0, "video": 0}` when unsupported.
- `enforce_eager = True`, `dtype = "bfloat16"`, `trust_remote_code = False`.
- Reasoning parser: `qwen3` (gated by `_call_accepts(LLM, "reasoning_parser")`).
- Thinking mode: disabled via tokenizer `apply_chat_template(..., enable_thinking=False)` with a prompt-level fallback `"Do not output chain-of-thought. Output only the JSON object."`.
- Structured output: prefers `StructuredOutputsParams(json=RAG_SCHEMA)`; falls back to `GuidedDecodingParams(json=RAG_SCHEMA)`; warns if neither is available.
- Environment: `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- Paths (Colab branch): `BASE_DIR = /content/drive/MyDrive/swiss_law`, with `DATA_DIR`, `INSIGHTS_DIR`, `ART_DIR`, `SCRIPT_DIR` subdirs.
- `INPUT_FILE = ART_DIR / 'court_authority_cards_v4.jsonl'`
- `OUTPUT_FILE = ART_DIR / 'court_authority_cards_rag.jsonl'`
- `CHECKPOINT_FILE = ART_DIR / 'rag_checkpoint.txt'`
- `LIMIT = 1000` (cap on cards processed this run).
- Install command (Colab): `uv pip install --system -U vllm --torch-backend=auto` plus `uv` and `tqdm`.

## Data

- Input JSONL: `court_authority_cards_v4.jsonl` under the Drive `artifacts/` folder. Each record is a court authority card with fields including `text_excerpt_original`, `citation`, `legal_area`, optional `issue_labels_en`, and `is_notification_paragraph`.
- Output JSONL: `court_authority_cards_rag.jsonl` — each input card is rewritten with an added `rag_enrichment` object.
- Checkpoint file: `rag_checkpoint.txt` stores the next line index to resume from; output is opened in append mode.
- Pre-filter regex `COST_PROC_RE` matches cost / procedural / fee / legal-aid / remittal phrases in DE/FR/IT (`Gerichtskosten`, `frais judiciaires`, `spese giudiziarie`, `Parteientschädigung`, `unentgeltliche Rechtspflege`, etc.) to short-circuit trivial paragraphs without an LLM call.
- `auto_classify` produces stub enrichments tagged with `method`: `auto_notification` (when `is_notification_paragraph` is true), `auto_short` (text shorter than 50 chars), or `auto_cost` (regex match on first 400 chars). LLM-enriched rows are tagged `qwen35_35b_a3b_vllm`; parse failures are tagged `json_parse_failed` and carry `raw_output` (first 400 chars) plus `parse_error` (first 200 chars).
- User message passed to the model contains the citation, deterministic `legal_area`, up to 8 existing English issue labels, and `text_excerpt_original` truncated to 2500 chars.

## Pipeline

1. Environment detection: `IN_COLAB` test, prints Colab status. `nvidia-smi` reports `NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB`.
2. Install: `%pip install -q -U uv tqdm` then `uv pip install --system -U vllm --torch-backend=auto` (Colab only).
3. Drive mount: `drive.mount('/content/drive')`.
4. Configuration: sets paths, model id, sampling/runtime knobs (see Configuration above).
5. Schema + prompts: defines `RAG_SCHEMA` (object with required fields `english_summary`, `legal_topic`, `english_legal_concepts`, `search_keywords`, `natural_language_queries`, `paragraph_role`, `outcome_signal`; optional `legal_question`, `legal_rule`, `court_holding`, `factual_context`; enums for `paragraph_role` and `outcome_signal`; `additionalProperties=False`). `SYSTEM_PROMPT` instructs the model to act as a Swiss legal analyst translating DE/FR/IT paragraphs to precise English legal terminology and emitting JSON only.
6. Helpers: `build_user_message`, `auto_classify`, `_stub`, `stream_input`, `count_lines`.
7. Model load: imports `vllm`, prints version, builds `llm_kwargs` via `build_llm_kwargs` (introspects `LLM.__init__` to gate `reasoning_parser`, `enable_prefix_caching`, `speculative_config`, `language_model_only`, `limit_mm_per_prompt`), instantiates `LLM(...)`, gets the tokenizer, and builds `sampling_params` via `make_sampling_params(RAG_SCHEMA)` using the newest available structured-output API.
8. Run enrichment: opens output in append mode, reads `CHECKPOINT_FILE` to set the starting line, counts total input lines, iterates `stream_input(INPUT_FILE, start)`. For each card: if `auto_classify` returns a stub, it is written directly; otherwise the card is appended to a `pending` batch. When the batch reaches `BATCH_SIZE`, `flush_batch` renders Qwen chat prompts with `enable_thinking=False`, calls `llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)`, parses each output with `clean_model_json` (strips ```` ```json ```` fences and any leaked `</think>` prefix) and `json.loads`, normalizes via `normalize_enrichment` (fills schema defaults and forces `method='qwen35_35b_a3b_vllm'`), then writes each enriched card and updates the checkpoint. JSON failures fall back to a `json_parse_failed` stub carrying truncated raw output and the exception string. Processing stops once `LIMIT` cards have been handled.
9. Verify output: aggregates `paragraph_role`, `outcome_signal`, and `method` counts; flags rows missing any required schema field; prints percentages.
10. Sample: filters cards whose `rag_enrichment.method` starts with `qwen35` and prints 3 random samples (citation, role, outcome, topic, 200-char summary, keywords, NL queries).

## Results

Only two cells executed and captured output:

- `detect-colab`:
  ```
  Running in Colab: True
  GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB
  ```
- `mount-drive`:
  ```
  Mounted at /content/drive
  ```
- `helpers`:
  ```
  Helpers loaded.
  ```

All later cells (install, configuration print, schema print, vLLM import, `nvidia-smi`, model load, enrichment run, statistics, sample) have no captured outputs in the notebook.

## Summary

This Colab/vLLM notebook enriches `court_authority_cards_v4.jsonl` with English-language semantic fields for downstream RAG, using `Qwen/Qwen3.5-35B-A3B` running through vLLM offline inference on a ~96 GB Blackwell-class GPU. It applies a deterministic pre-filter (`auto_classify`) that handles notification, very short, and cost/procedural paragraphs without an LLM call, then batches all remaining paragraphs through `llm.generate` with `StructuredOutputsParams(json=RAG_SCHEMA)` (falling back to `GuidedDecodingParams` on older vLLM builds). Generation runs with thinking disabled, MTP-1 speculative decoding, disabled prefix caching, BF16, `max_model_len=4096`, batch size 4, temperature 0.15, and a `LIMIT=1000`-card cap, with checkpointed resume via `rag_checkpoint.txt` and append-mode writes to `court_authority_cards_rag.jsonl`. The schema enforces required fields (`english_summary`, `legal_topic`, `english_legal_concepts`, `search_keywords`, `natural_language_queries`, `paragraph_role`, `outcome_signal`) plus optional reasoning fields and constrained enums for paragraph role and outcome. The captured outputs only confirm the Colab environment, GPU (RTX PRO 6000 Blackwell, 97887 MiB), Drive mount, and helper module load; the model-loading, enrichment, statistics, and sample cells produced no recorded output.
