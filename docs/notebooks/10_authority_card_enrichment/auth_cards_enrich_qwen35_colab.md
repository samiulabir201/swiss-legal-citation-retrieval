# auth_cards_enrich_qwen35_colab

**Path:** `e:\swiss_citation_extraction\notebooks\10_authority_card_enrichment\auth_cards_enrich_qwen35_colab.ipynb`

## Configuration

- Model: `Qwen/Qwen3.5-35B-A3B` (declared in `MODEL_ID`); actually loaded variant is `Qwen/Qwen3.5-35B-A3B-Base` (MoE, 35B total / 3B active params). Reasoning/thinking mode disabled (`NO_THINK = {"enable_thinking": False}`). Justification: ranks #3 on Structured Output Benchmark (SOB, Value Accuracy 0.801) for JSON schema compliance.
- Quantization: `QUANTIZATION = None` (BF16, ~70 GB weights); AWQ variant noted as optional alternative.
- GPU target: NVIDIA G4 / RTX PRO 6000 Blackwell (96 GB VRAM). Observed: `NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB`.
- vLLM `LLM(...)` parameters: `dtype='bfloat16'`, `gpu_memory_utilization=0.85`, `max_model_len=4096`, `trust_remote_code=False`, `tensor_parallel_size=1`, `enforce_eager=True`, `limit_mm_per_prompt={"image":0,"video":0}`, `enable_prefix_caching=False`, `speculative_config={"method":"mtp","num_speculative_tokens":1}` (MTP-1 speculative decoding), `reasoning_parser="qwen3"`. Environment override: `VLLM_WORKER_MULTIPROC_METHOD = "spawn"`.
- `SamplingParams`: `temperature=0.15`, `max_tokens=600`, `guided_decoding={"json": RAG_SCHEMA}`.
- Notebook-level constants (cell 6): `GPU_MEMORY_UTIL=0.87`, `BATCH_SIZE=64`, `TEMPERATURE=0.15`, `MAX_TOKENS=600`, `TENSOR_PARALLEL=1`, `LIMIT=1000`.
- Pip installs: `vllm` (nightly via `--upgrade --pre`, then `--upgrade`), `tqdm`, `transformers`, `accelerate`, `flash-attn --no-build-isolation`.
- Paths (Colab branch): `BASE_DIR=/content/drive/MyDrive/swiss_law`; `INPUT_FILE=BASE_DIR/artifacts/court_authority_cards_v4.jsonl`; `OUTPUT_FILE=BASE_DIR/artifacts/court_authority_cards_rag.jsonl`; `CHECKPOINT_FILE=BASE_DIR/artifacts/rag_checkpoint.txt`.
- Mode flag: `IN_COLAB` auto-detected; Google Drive mounted at `/content/drive`.
- Adapted from `scripts/enrich_cards_with_qwen.py`.

## Data

- Input: `court_authority_cards_v4.jsonl` (court authority cards, one JSON object per line). Fields read per card: `text_excerpt_original` (truncated to 2500 chars in prompt), `citation`, `legal_area`, `issue_labels_en` (first 8 used as `Existing labels`), `is_notification_paragraph`.
- Output: `court_authority_cards_rag.jsonl`. Each input card is re-emitted with an added `rag_enrichment` object.
- Checkpoint: `rag_checkpoint.txt` stores the next line index to process; enables resumption.
- Run limit: `LIMIT = 1000` cards.
- Pre-filter (`auto_classify`): cards bypass the LLM and receive a deterministic `_stub` enrichment when:
  - `is_notification_paragraph` → method `auto_notification`, role `procedural`.
  - `len(text) < 50` → method `auto_short`, role `procedural`.
  - `COST_PROC_RE` matches in first 400 chars (DE/FR/IT regex for `Gerichtskosten`, `frais judiciaires`, `spese giudiziarie`, `unentgeltliche Rechtspflege`, `assistance judiciaire`, `patrocinio gratuito`, remand phrases, `Fr.` cost lines) → method `auto_cost`, role `cost`.

## Pipeline

1. **Environment setup (cells 2-4):** detect Colab, print `nvidia-smi`, install vLLM (nightly) + tqdm, mount Google Drive.
2. **Configuration (cell 6):** declare paths, model id, inference constants; create directories.
3. **Schema and prompts (cell 8):** define `COST_PROC_RE` pre-filter regex and `RAG_SCHEMA` (JSON object with fields `english_summary`, `legal_topic`, `legal_question`, `legal_rule`, `court_holding`, `factual_context`, `english_legal_concepts` [maxItems 8], `search_keywords` [maxItems 10], `natural_language_queries` [maxItems 5], `paragraph_role` enum {holding, reasoning, background, cost, procedural, disposition, standard_of_review, obiter}, `outcome_signal` enum {granted, dismissed, inadmissible, remitted, partial, none}; required = summary, topic, concepts, keywords, NL queries, role, outcome). Define `SYSTEM_PROMPT` instructing the model to act as a Swiss legal analyst translating DE/FR/IT paragraphs into English JSON for RAG.
4. **Helpers (cell 10):** `build_user_message` (assembles `Citation`, `Legal area`, optional `Existing labels`, original paragraph), `_stub`, `auto_classify`, `stream_input`, `count_lines`.
5. **Model load (cells 12-15):** install transformers/accelerate, attempt `flash-attn` build (fails in run), upgrade vLLM, instantiate `LLM(model="Qwen/Qwen3.5-35B-A3B-Base", ...)` with MTP-1 speculative decoding and `reasoning_parser="qwen3"`; create `SamplingParams` with guided JSON; set `NO_THINK = {"enable_thinking": False}`.
6. **Enrichment loop (cell 17):** open `INPUT_FILE`, set `tokenizer.padding_side="left"` (FIX 1), resume from checkpoint, iterate cards. For each card: run `auto_classify`; if it returns a stub, write directly. Otherwise queue into `pending`; when `len(pending) >= BATCH_SIZE`, `flush_batch()`:
   - apply chat template to `[system, user]` messages, tokenize with left padding to `max_length=3000`,
   - call `model.generate(..., max_new_tokens=MAX_TOKENS, temperature=TEMPERATURE, do_sample=True)` under `torch.no_grad()`,
   - decode each completion, strip ```` ```json ```` fences, `json.loads`, validate dict type (FIX 2), tag `method='qwen35_35b_a3b_hf'`,
   - on JSON parse or type error, increment `json_errors`, emit `_stub` with `method='json_parse_failed'` and `raw_output` (first 400 chars),
   - attach `rag_enrichment` to card, append to output, advance checkpoint.
   Note: although the model is loaded via vLLM (`llm = LLM(...)`), the inference code in `flush_batch` calls `model.generate(**inputs, ...)` (HuggingFace-style API) and references undefined symbols `model`, `tokenizer`, `torch`.
7. **Verification (cells 19-20):** read `OUTPUT_FILE`, build `Counter` distributions for `paragraph_role`, `outcome_signal`, `method`; count records missing required schema fields; print 3 random LLM-enriched cards (citation, role, outcome, topic, summary, keywords, NL queries).

## Results

- **Cell 2 (env check):** `Running in Colab: True`; `GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB`.
- **Cell 3 (pip install vllm):** large download chain completes; pip dependency-resolver warnings only — conflicts reported for `opentelemetry-api/sdk` (1.41.1 vs google-adk required <1.39.0), `cuda-python 13.2.0` vs pylibraft/cuml/cuvs/cudf/rmm (need <13.0), `cuda-toolkit 13.0.2` vs cuml/libcuvs/libcuml/libraft/cudf (need 12.*), `numba 0.65.0` vs cuml/cudf (need <0.62), `ipython` missing `jedi>=0.16`.
- **Cell 4:** `Mounted at /content/drive`.
- **Cell 6:** prints
  ```
  BASE_DIR   : /content/drive/MyDrive/swiss_law
  INPUT_FILE : /content/drive/MyDrive/swiss_law/artifacts/court_authority_cards_v4.jsonl
  OUTPUT_FILE: /content/drive/MyDrive/swiss_law/artifacts/court_authority_cards_rag.jsonl
  MODEL_ID   : Qwen/Qwen3.5-35B-A3B
  ```
- **Cell 8:** `Schema fields: ['english_summary', 'legal_topic', 'legal_question', 'legal_rule', 'court_holding', 'factual_context', 'english_legal_concepts', 'search_keywords', 'natural_language_queries', 'paragraph_role', 'outcome_signal']`.
- **Cell 10:** `Helpers loaded.`
- **Cell 12 (transformers/accelerate):** all requirements already satisfied (transformers 5.7.0, accelerate 1.13.0, torch 2.11.0).
- **Cell 13 (flash-attn):** `Building wheels for collected packages: flash-attn` → `error: subprocess-exited-with-error`; `python setup.py bdist_wheel did not run successfully. exit code: 1`; `ERROR: Failed building wheel for flash-attn`; `ERROR: Failed to build installable wheels for some pyproject.toml based projects (flash-attn)`. flash-attn was NOT installed.
- **Cell 14 (vllm upgrade):** downloads and installs `vllm-0.20.0`, `torch==2.11.0`, `torchaudio==2.11.0`, `torchvision==0.26.0`, `flashinfer-python==0.6.8.post1`, `apache-tvm-ffi==0.1.9`, `tilelang==0.1.9`, plus many CUDA 13 toolkit packages and Python dependencies. Completes (no error block recorded).
- **Cell 15 (model load):** prints `Loading Qwen3.5 via vLLM with Latency-Focused Serving...`. vLLM logs:
  - `non-default args: {'dtype': 'bfloat16', 'max_model_len': 4096, 'enable_prefix_caching': False, 'gpu_memory_utilization': 0.85, 'disable_log_stats': True, 'enforce_eager': True, 'limit_mm_per_prompt': {'image': 0, 'video': 0}, 'reasoning_parser': 'qwen3', 'speculative_config': {'method': 'mtp', 'num_speculative_tokens': 1}, 'model': 'Qwen/Qwen3.5-35B-A3B-Base'}`
  - downloads `config.json`, `preprocessor_config.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `tokenizer.json` (12.8 MB)
  - `Setting UCX_RCACHE_MAX_UNRELEASED to '1024' to avoid a rare memory leak in UCX when using NIXL.`
  - `WARNING: NIXL is not available`; `WARNING: NIXL agent config is not available`
  - `Resolved architecture: Qwen3_5MoeForConditionalGeneration`; `Using max model len 4096`
  - MTP draft: `Resolved architecture: Qwen3_5MoeMTP`; `Using max model len 262144`
  - `Chunked prefill is enabled with max_num_batched_tokens=16384.`
  - `Asynchronous scheduling is enabled.`
  - `Enforce eager set, disabling torch.compile and CUDAGraphs. ... -cc.mode=none -cc.cudagraph_mode=none`
  - `Inductor compilation was disabled by user settings ...`
  - `Final IR op priority after setting platform defaults: IrOpPriorityConfig(rms_norm=['vllm_c', 'native'])`
  - `Cudagraph is disabled under eager mode`
  - `Enabled custom fusions: norm_quant, act_quant`
  - stderr: `[transformers] Qwen2VLImageProcessorFast is deprecated. ... use Qwen2VLImageProcessor instead.`
  - `All limits of multimodal modalities supported by the model are set to 0, running in text-only mode.`
  - Final `print('Model loaded successfully for minimum latency!')` is NOT in the captured stream (truncated/absent). No explicit failure shown for the LLM constructor.
- **Cell 17 (enrichment run):** prints `Resuming at line 24` and `Total=1,024  To process=1,000`, then `enrich: 2%|2 | 24/1024 [00:00<?, ?card/s]`. Run terminated by `KeyboardInterrupt` inside `flush_batch` at the `model.generate(**inputs, ...)` call (`/usr/local/lib/python3.12/dist-packages/transformers/generation/utils.py` `_sample` while-loop). No cards beyond line 24 were written by this execution. Implicitly, `tokenizer` and `model` are referenced but never bound in this cell — the only loaded object from cell 15 is `llm` (vLLM `LLM`); the KeyboardInterrupt traceback shows `model.generate` resolved to a `transformers` generation path, indicating either a separately defined `model`/`tokenizer` (not in the notebook source) or that the prior cell had bound them in this Colab session.
- **Cell 19 (distribution stats):** no outputs captured.
- **Cell 20 (sample inspection):** no outputs captured.

## Summary

This Colab notebook enriches `court_authority_cards_v4.jsonl` with English RAG-oriented metadata (summary, topic, question, rule, holding, factual context, concept list, search keywords, natural-language queries, paragraph_role enum, outcome_signal enum) using Qwen3.5-35B-A3B (MoE) on an RTX PRO 6000 Blackwell (97 GB). It first applies a deterministic pre-filter (`auto_classify`) for notification paragraphs, sub-50-char fragments, and cost/fee paragraphs (multilingual regex), routing them to canned stubs without invoking the LLM. The remaining cards are batched (`BATCH_SIZE=64`) and decoded with guided JSON via vLLM `SamplingParams(guided_decoding={"json": RAG_SCHEMA})`, with thinking mode explicitly disabled. In this captured run the flash-attn build failed, vLLM 0.20.0 and the MoE model loaded successfully with MTP-1 speculative decoding and `enforce_eager=True`, but the enrichment loop in cell 17 was halted by a KeyboardInterrupt at the very first batch (only the initial checkpoint state of 24 cards is visible), so cells 19 and 20 produced no verification output. Note that the in-loop code calls `model.generate(...)` and references `tokenizer`/`torch` symbols that the notebook never defines after the vLLM load, suggesting a mismatch between the loaded vLLM `llm` object and the HuggingFace-style generation path actually used.
