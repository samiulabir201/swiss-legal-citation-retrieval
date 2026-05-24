"""One-shot patcher: produce an optimized copy of the user's law notebook.

Replaces Cells 2 (Config), 4 (Schema/prompt), 7 (Gen helpers), 8 (Run loop)
with optimized variants:
  * kv_cache_dtype = None
  * max_new_tokens 700 -> 450
  * max_num_seqs 320 -> 384
  * max_model_len 2560 -> 2304
  * repetition_penalty 1.05 -> 1.0
  * Slim schema hint (~150 tok prefill saved)
  * Robust units / list-of-dict coercion in build_user_prompt
  * Defensive defaults for chosen_backend / chosen_quantization / chosen_speculation
"""
from __future__ import annotations

import json
from pathlib import Path

SRC = Path(r"C:\Users\samiul\Downloads\enrich_laws_de_qwen3_8b_kaggle (1).ipynb")
DST = Path(r"C:\Users\samiul\Downloads\enrich_laws_de_qwen3_8b_kaggle_optimized.ipynb")


def mk_code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELL2 = '''# Cell 2 - Config (Blackwell-optimized; kv_cache_dtype=None; quality preserved)

@dataclass
class Config:
    # ---- Paths ----
    base_dir: str = str(BASE_DIR)
    data_dir: str = str(DATA_DIR)
    output_dir: str = str(OUTPUT_DIR)
    model_download_dir: str = str(MODEL_DIR / 'huggingface')
    local_scratch_dir: str = str(LOCAL_SCRATCH)

    # ---- Input ----
    input_jsonl: str = str(DATA_DIR / 'law_llm_input.jsonl')
    fallback_input_jsonl: str = 'law_llm_input.jsonl'

    # ---- Model ----
    model_name: str = 'Qwen/Qwen3-8B-AWQ'

    # ---- Slice ----
    start: int = 0
    limit: int = 0  # 0 = process all
    sample_random: bool = False
    random_seed: Optional[int] = 42
    min_text_chars: int = 20
    # Law p99 = 1274 chars. 1500 covers >99% of rows untruncated.
    max_text_chars: int = 1500

    # ---- GPU / vLLM ----
    gpu_mode: str = 'single'
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.92

    # System+schema (~770 tok with slim schema, prefix-cached) + user header
    # (~150 tok) + text up to ~750 tok + output 450 tok = ~2120 worst case.
    # 2304 fits with margin.
    max_model_len: int = 2304

    # Match the court run's concurrency profile.
    max_num_seqs: int = 384
    submit_chunk: int = 2048

    enforce_eager: bool = False
    quantization: str = 'awq_marlin'
    disable_custom_all_reduce: bool = True

    # ---- KV cache ----
    # Disabled per user request - uses default fp16 KV. The 95 GB Blackwell
    # still fits max_num_seqs=384 with max_model_len=2304 comfortably.
    kv_cache_dtype: Optional[str] = None

    # ---- Speculative decoding (off on SM 12.x; same as the court run) ----
    enable_ngram_speculation: bool = False
    speculative_num_tokens: int = 5
    speculative_ngram_min: int = 2
    speculative_ngram_max: int = 4

    # ---- Sampling / generation ----
    use_structured_outputs: bool = False
    # 12-field law schema with up to 10 term pairs + 4 defined terms + 5+5
    # lists is ~350-400 output tokens p99. 450 leaves margin without burning
    # decode time on tokens the model never produces.
    max_new_tokens: int = 450
    retry_max_new_tokens: int = 600
    max_retries: int = 1
    # T=0.1 breaks the literal-translation attractor at T=0
    # (Bewilligung -> "approval" instead of "permit / authorisation").
    temperature: float = 0.1
    top_p: float = 0.9
    # Reverted to 1.0 - penalty adds decode cost without quality benefit.
    repetition_penalty: float = 1.0
    enable_thinking: bool = False

    # ---- Output / debug ----
    include_raw_output_on_success: bool = False

cfg = Config()

if cfg.gpu_mode == 'single':
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
    cfg.tensor_parallel_size = 1
elif cfg.gpu_mode == 'tp2':
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    cfg.tensor_parallel_size = 2
    cfg.disable_custom_all_reduce = True

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

base_dir = Path(cfg.base_dir)
data_dir = Path(cfg.data_dir)
out_dir = Path(cfg.output_dir)
local_scratch = Path(cfg.local_scratch_dir)
model_download_dir = Path(cfg.model_download_dir)
for d in [base_dir, data_dir, out_dir, model_download_dir, local_scratch]:
    d.mkdir(parents=True, exist_ok=True)

end_idx = cfg.start + cfg.limit - 1 if cfg.limit else -1
suffix = f'{cfg.start:07d}_{end_idx:07d}' if cfg.limit else f'{cfg.start:07d}_all'

output_jsonl = out_dir / f'law_llm_descriptors_{suffix}.jsonl'
output_preview_csv = out_dir / f'law_llm_descriptors_{suffix}_preview.csv'
output_failures_jsonl = out_dir / f'law_llm_descriptors_{suffix}_failures.jsonl'
output_metrics_json = out_dir / f'law_llm_descriptors_{suffix}_metrics.json'

local_jsonl = local_scratch / f'law_llm_descriptors_{suffix}.jsonl'
local_failures_jsonl = local_scratch / f'law_llm_descriptors_{suffix}_failures.jsonl'

print(json.dumps(asdict(cfg), indent=2, default=str))
print('Output JSONL (final):', output_jsonl)
print('Output JSONL (hot)  :', local_jsonl)
'''


CELL4 = r'''# Cell 4 - Schema and prompt (Swiss-legal English mapping, NOT literal translation)
#
# Schema in the SYSTEM message -> entire constant prefix (system + schema +
# Swiss-legal vocabulary table) is fully prefix-cached across the run.

DESCRIPTOR_KEYS = [
    'english_summary',
    'legal_rule',
    'applicability_conditions',
    'exceptions_or_limitations',
    'legal_question',
    'concepts_en',
    'terms_de_to_en',
    'defined_terms',
    'addressees',
    'sanctions_or_consequences',
    'provision_role_llm',
    'specificity_score',
]

PROVISION_ROLES = {
    'definition', 'purpose', 'scope', 'principle',
    'right_or_entitlement', 'duty', 'prohibition', 'procedure',
    'competence', 'sanction_or_penalty',
    'data_reporting', 'fees_or_costs', 'transitional_or_commencement',
    'other',
}
BOILERPLATE_ROLES = {'transitional_or_commencement', 'fees_or_costs', 'data_reporting'}

# Slim schema hint - saves ~150 prefill tokens vs the verbose form.
LLM_SCHEMA_HINT = {
    'english_summary': '<=2 sentences English; what THIS article says (not the law title)',
    'legal_rule': '<=25 words operative rule; empty for boilerplate',
    'applicability_conditions': ['0-5 short English conditions'],
    'exceptions_or_limitations': ['0-5 English carve-outs'],
    'legal_question': '<=18 words; empty for boilerplate',
    'concepts_en': ['3-8 English legal concepts'],
    'terms_de_to_en': [{'de': 'verbatim DE term from text', 'en': 'Swiss-legal English equivalent'}],
    'defined_terms': [{'term': 'verbatim DE term defined here', 'definition': 'English gloss'}],
    'addressees': ['0-6 English labels: who is bound'],
    'sanctions_or_consequences': ['0-4 English items in text'],
    'provision_role_llm': 'definition|purpose|scope|principle|right_or_entitlement|duty|prohibition|procedure|competence|sanction_or_penalty|data_reporting|fees_or_costs|transitional_or_commencement|other',
    'specificity_score': '0..1',
}
_SCHEMA_JSON = json.dumps(LLM_SCHEMA_HINT, ensure_ascii=False, separators=(',', ':'))

SYSTEM_PROMPT = (
    "You are a Swiss legal-interpretation assistant working on individual articles\n"
    "of Swiss federal law (Bundesgesetze, Verordnungen, the Bundesverfassung, Vertraege,\n"
    "SR-numbered statutes). The article text is in German, French, or Italian. Your\n"
    "job is to extract the operative legal meaning into English, preserving the\n"
    "exact original-language legal terms.\n\n"
    "Return exactly one compact JSON object with exactly this shape (all keys present):\n"
    + _SCHEMA_JSON + "\n\n"
    "Hard rules:\n"
    "1. JSON only. No prose, no markdown, no preamble.\n"
    "2. Use English for ALL semantic fields except `terms_de_to_en[].de` and\n"
    "   `defined_terms[].term`, which MUST be exact substrings of the source text.\n"
    "3. Map each German term to its **Swiss-legal English equivalent**, NOT a literal\n"
    "   translation. Examples:\n"
    "     Bewilligung -> permit / authorisation\n"
    "     Verfuegung -> formal administrative order\n"
    "     Rechtsbegehren -> prayer for relief\n"
    "     Zustaendigkeit -> jurisdiction / competence\n"
    "     Aufsichtsbehoerde -> supervisory authority\n"
    "     Inverkehrbringen -> placing on the market\n"
    "     Tatbestand -> set of facts / elements of the offence\n"
    "     Beschwerde -> appeal\n"
    "     Eidgenoessisch -> federal (Swiss)\n"
    "     Bundesrat -> Federal Council\n"
    "     EDI / SBFI / FINMA / ESTV / EJPD -> keep the acronym verbatim\n"
    "   When in doubt, prefer EU/UK statutory English over US wording.\n"
    "4. Do NOT invent statute citations, BGE numbers, dates, party names, or\n"
    "   sanctions that are not literally in the text.\n"
    "5. Do NOT translate literally. If the German is metaphorical or formal, pick\n"
    "   the recognised legal English term.\n"
    "6. Never copy the law title into `english_summary`. Describe what THIS article\n"
    "   says, not what the parent statute is about.\n"
    "7. For repeal markers (\"Aufgehoben\"), commencement clauses (\"Tritt am ... in Kraft\"),\n"
    "   pure fee tables, annex code lists, or transitional provisions:\n"
    "     - keep `legal_rule`, `applicability_conditions`, `exceptions_or_limitations`,\n"
    "       `legal_question` empty.\n"
    "     - still fill `english_summary`, `concepts_en`, `terms_de_to_en`,\n"
    "       `provision_role_llm`, `specificity_score` (low).\n"
    "8. Never put `Art.`, `Abs.`, `Buchstabe`, `Ziffer`, or SR numbers into\n"
    "   `terms_de_to_en` - those are anchors, not legal terms.\n"
    "9. Cap: 10 terms_de_to_en, 4 defined_terms, 6 addressees, 5 conditions, 5\n"
    "   exceptions, 8 concepts_en. Trim to the most salient.\n"
    "10. If a field has nothing to populate, return an empty string or empty list,\n"
    "    but the key MUST be present. JSON only."
)

USER_TEMPLATE = (
    'Citation: {citation}\n'
    'Law title: {law_title}\n'
    'Section path: {section_path}\n'
    'Law code: {law_code}   Article: {article}   Units: {units}\n'
    'Source type: {source_type}   Enactment year: {year}\n'
    'Static legal area hint: {legal_area_hint}\n'
    'Static domain hints: {domain_hints}\n\n'
    'Article text (verbatim, original language):\n'
    '"""\n'
    '{text}\n'
    '"""\n\n'
    'Return JSON only.'
)

def trim_text(text: str, max_chars: int) -> str:
    text = re.sub(r'\s+', ' ', str(text)).strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head].rstrip() + ' ... [TRUNCATED] ... ' + text[-tail:].lstrip()

def _coerce_str(x: Any, max_chars: int = 0) -> str:
    """Coerce any value to a flat string; never raise."""
    if x is None:
        return ''
    if isinstance(x, str):
        s = x
    elif isinstance(x, (list, tuple)):
        s = ', '.join(_coerce_str(i) for i in x if i)
    elif isinstance(x, dict):
        s = json.dumps(x, ensure_ascii=False)
    else:
        s = str(x)
    s = s.strip()
    return s[:max_chars] if max_chars else s

def _units_to_str(units: Any) -> str:
    """structural.units may be list[str] or list[dict]; produce a flat label string."""
    if not units:
        return ''
    if isinstance(units, str):
        return units
    if not isinstance(units, (list, tuple)):
        return str(units)
    parts = []
    for u in units:
        if u is None:
            continue
        if isinstance(u, str):
            s = u.strip()
        elif isinstance(u, dict):
            label = (u.get('label') or u.get('text') or '').strip()
            if not label:
                head = (u.get('unit') or u.get('type') or u.get('kind') or '').strip()
                tail = u.get('number')
                if tail is None:
                    tail = u.get('value')
                if tail is None:
                    tail = u.get('n')
                tail = '' if tail is None else str(tail).strip()
                label = (head + ' ' + tail).strip() if (head or tail) else json.dumps(u, ensure_ascii=False)
            s = label
        else:
            s = str(u).strip()
        if s:
            parts.append(s)
    return ', '.join(parts)

def build_user_prompt(row: dict) -> str:
    structural = row.get('structural') or {}
    title_meta = row.get('title_metadata') or {}
    hints = row.get('static_hints') or {}
    enactment_date = _coerce_str(title_meta.get('enactment_date'))
    year = _coerce_str(title_meta.get('enactment_year')) or (enactment_date[:4] if enactment_date else '')
    return USER_TEMPLATE.format(
        citation=_coerce_str(row.get('citation'), 200),
        law_title=_coerce_str(row.get('law_title'), 300),
        section_path=_coerce_str(row.get('title_section_path'), 200),
        law_code=_coerce_str(structural.get('law_code'), 60),
        article=_coerce_str(structural.get('article'), 40),
        units=_units_to_str(structural.get('units')),
        source_type=_coerce_str(title_meta.get('source_type'), 60),
        year=year,
        legal_area_hint=_coerce_str(hints.get('legal_area_static'), 100),
        domain_hints=', '.join(_coerce_str(d, 80) for d in (hints.get('domain_labels_en') or [])[:5] if d),
        text=trim_text(_coerce_str(row.get('text')), cfg.max_text_chars),
    )
'''


CELL7 = r'''# Cell 7 - Generation helpers + warm-up

# Defensive defaults: Cell 6 normally sets these; if it was skipped or failed,
# downstream cells still see defined variables.
chosen_backend = globals().get('chosen_backend', 'auto')
chosen_quantization = globals().get('chosen_quantization', cfg.quantization)
chosen_speculation = globals().get('chosen_speculation', False)

def render_prompt(row: dict, repair: bool = False, bad_output: str = '', error: str = '') -> str:
    user_prompt = build_user_prompt(row)
    if repair:
        user_prompt = (
            "The previous output was invalid JSON.\n\n"
            "Parser error:\n"
            f"{error}\n\n"
            "Previous output:\n"
            f"{(bad_output or '')[:1400]}\n\n"
            "Repair by returning exactly one complete compact JSON object using the same schema.\n"
            "Do not add prose, anchors, citations, or retrieval views.\n\n"
            f"{user_prompt}"
        )
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=cfg.enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate_raw(prompts, max_tokens: int):
    params = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=max_tokens,
        repetition_penalty=cfg.repetition_penalty,
    )
    outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
    return [out.outputs[0].text if out.outputs else '' for out in outputs]

def generate_raw_safe(prompts, max_tokens: int, min_split: int = 1):
    try:
        return generate_raw(prompts, max_tokens)
    except Exception as exc:
        if len(prompts) <= min_split:
            raise
        mid = len(prompts) // 2
        print(f'  Batch of {len(prompts)} failed ({type(exc).__name__}); splitting into {mid}+{len(prompts)-mid}')
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        left = generate_raw_safe(prompts[:mid], max_tokens, min_split=min_split)
        right = generate_raw_safe(prompts[mid:], max_tokens, min_split=min_split)
        return left + right

def parse_or_retry(row: dict, raw):
    attempts = []
    for attempt in range(cfg.max_retries + 1):
        try:
            if raw is None:
                raw = generate_raw_safe([render_prompt(row)], cfg.max_new_tokens)[0]
            obj = extract_json_object(raw)
            desc = normalize_descriptor(obj)
            return desc, {
                'status': 'ok' if attempt == 0 else 'ok_after_retry',
                'attempt_count': attempt + 1,
                'error': None,
                'raw_output': raw if cfg.include_raw_output_on_success else None,
            }
        except Exception as exc:
            err = repr(exc)
            attempts.append({'attempt': attempt + 1, 'error': err, 'raw_output': (raw or '')[:1400]})
            if attempt >= cfg.max_retries:
                return empty_descriptor(err), {
                    'status': 'failed_descriptor_parse',
                    'attempt_count': attempt + 1,
                    'error': err,
                    'attempts': attempts,
                    'raw_output': raw,
                }
            raw = generate_raw_safe(
                [render_prompt(row, repair=True, bad_output=raw or '', error=err)],
                cfg.retry_max_new_tokens,
            )[0]

print('Warm-up generation (16 short prompts)...')
_warm_row = {
    'citation': 'warmup',
    'law_title': 'warmup',
    'title_section_path': '',
    'structural': {'law_code': 'WARM', 'article': '1', 'units': []},
    'title_metadata': {'source_type': 'ordinance', 'enactment_year': '2020', 'enactment_date': ''},
    'static_hints': {'legal_area_static': '', 'domain_labels_en': []},
    'text': 'Warm-up paragraph for CUDA graph capture and FlashInfer kernel selection.',
}
_warm_prompts = [render_prompt(_warm_row)] * 16
_t = time.time()
_ = generate_raw_safe(_warm_prompts, max_tokens=64)
print(f'  done in {time.time()-_t:.2f}s')
'''


CELL8 = r'''# Cell 8 - Run extraction (mega-batched, append-mode checkpointing)
#
# Submit cfg.submit_chunk (2048) prompts per llm.generate() call; vLLM
# continuous-batches across max_num_seqs (384). Append to local_jsonl after
# each chunk; copy to Drive at end.

# Defensive defaults so this cell is runnable even if Cell 6 was re-loaded.
chosen_backend = globals().get('chosen_backend', 'auto')
chosen_quantization = globals().get('chosen_quantization', cfg.quantization)
chosen_speculation = globals().get('chosen_speculation', False)

records_for_preview = []
failures = []
status_counter = Counter()
role_counter = Counter()
t0 = time.time()
processed = 0
prompt_token_total = 0
output_token_total = 0
grounded_total = 0.0

# Checkpoint resume
done_rows: set = set()
if local_jsonl.exists():
    print(f'Checkpoint found: {local_jsonl}')
    with local_jsonl.open(encoding='utf-8') as _ckpt:
        for _line in _ckpt:
            _line = _line.strip()
            if not _line:
                continue
            try:
                done_rows.add(int(json.loads(_line)['_source_row']))
            except Exception:
                pass
    print(f'  -> {len(done_rows):,} rows already done; will skip.')
else:
    print('No checkpoint found - starting fresh.')

remaining_df = work_df[~work_df['_source_row'].isin(done_rows)].reset_index(drop=True)
print(f'Rows remaining: {len(remaining_df):,} / {len(work_df):,} total ({len(done_rows):,} skipped).')

def _generate_with_metrics(prompts, max_tokens: int):
    params = SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=max_tokens,
        repetition_penalty=cfg.repetition_penalty,
    )
    return llm.generate(prompts, sampling_params=params, use_tqdm=False)

def _generate_with_metrics_safe(prompts, max_tokens: int, min_split: int = 1):
    try:
        return _generate_with_metrics(prompts, max_tokens)
    except Exception as exc:
        if len(prompts) <= min_split:
            raise
        mid = len(prompts) // 2
        print(f'  Batch of {len(prompts)} failed ({type(exc).__name__}); splitting into {mid}+{len(prompts)-mid}')
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        left = _generate_with_metrics_safe(prompts[:mid], max_tokens, min_split=min_split)
        right = _generate_with_metrics_safe(prompts[mid:], max_tokens, min_split=min_split)
        return left + right

def _row_to_dict(row) -> dict:
    """Coerce a pandas row to a plain dict; tolerate missing fields."""
    return {
        '_source_row': int(row['_source_row']),
        'citation': row['citation'] if 'citation' in row else '',
        'law_title': row['law_title'] if 'law_title' in row else '',
        'title_section_path': row['title_section_path'] if 'title_section_path' in row else '',
        'structural': row['structural'] if 'structural' in row else {},
        'title_metadata': row['title_metadata'] if 'title_metadata' in row else {},
        'static_hints': row['static_hints'] if 'static_hints' in row else {},
        'llm_priority': row['llm_priority'] if 'llm_priority' in row else 'medium',
        'text': row['text'] if 'text' in row else '',
    }

N = len(remaining_df)
SUBMIT = cfg.submit_chunk
print(f'Processing {N:,} rows in chunks of {SUBMIT} (max_num_seqs={cfg.max_num_seqs}; {len(done_rows):,} done).')

with local_jsonl.open('a', encoding='utf-8') as out_f:
    pbar = tqdm(total=N, desc='Law LLM enrichment', smoothing=0.05)

    for start in range(0, N, SUBMIT):
        end = min(start + SUBMIT, N)
        chunk = remaining_df.iloc[start:end]

        row_objs = []
        prompts = []
        for _, row in chunk.iterrows():
            row_obj = _row_to_dict(row)
            row_objs.append(row_obj)
            prompts.append(render_prompt(row_obj))

        chunk_t = time.time()
        try:
            req_outputs = _generate_with_metrics_safe(prompts, cfg.max_new_tokens)
        except Exception as exc:
            print(f'Chunk {start}-{end} failed completely; falling back to per-row: {exc!r}')
            req_outputs = [None] * len(row_objs)

        for row_obj, req_out in zip(row_objs, req_outputs):
            if req_out is None:
                raw = None
            else:
                raw = req_out.outputs[0].text if req_out.outputs else ''
                try:
                    prompt_token_total += len(req_out.prompt_token_ids or [])
                    if req_out.outputs:
                        output_token_total += len(req_out.outputs[0].token_ids or [])
                except Exception:
                    pass

            desc, gen = parse_or_retry(row_obj, raw)
            grounded = grounded_terms_pct(desc.get('terms_de_to_en', []), row_obj['text'])
            grounded_total += grounded
            role = desc.get('provision_role_llm', 'other')
            role_counter[role] += 1

            rec = {
                '_source_row': row_obj['_source_row'],
                'citation': row_obj['citation'],
                'language': 'de',
                'llm_priority': row_obj['llm_priority'],
                'llm_enrichment': desc,
                'llm_quality': {
                    'json_valid': gen['status'].startswith('ok'),
                    'terms_grounded_pct': grounded,
                    'boilerplate_role': role in BOILERPLATE_ROLES,
                },
                'llm_generation': {
                    'model': cfg.model_name,
                    'method': 'law_descriptor_v1',
                    'attention_backend': chosen_backend,
                    'kv_cache_dtype': cfg.kv_cache_dtype,
                    'speculative': chosen_speculation,
                    **gen,
                },
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            processed += 1
            status_counter[gen['status']] += 1
            if len(records_for_preview) < 1000:
                records_for_preview.append(rec)
            if gen['status'].startswith('failed'):
                failures.append(rec)

        out_f.flush()
        pbar.update(end - start)

        rate = end / max(time.time() - t0, 1e-9)
        chunk_rate = (end - start) / max(time.time() - chunk_t, 1e-9)
        pbar.set_postfix(
            chunk_rps=f'{chunk_rate:.1f}',
            avg_rps=f'{rate:.1f}',
            ok=status_counter['ok'],
            failed=status_counter['failed_descriptor_parse'],
        )

    pbar.close()

elapsed = time.time() - t0
print(f'\nFinished {processed} rows in {elapsed:.1f}s = {processed/max(elapsed,1e-9):.2f} rows/sec')
print(f'Prompt tokens total:  {prompt_token_total:,}  (avg {prompt_token_total/max(processed,1):.0f}/row)')
print(f'Output tokens total:  {output_token_total:,}  (avg {output_token_total/max(processed,1):.0f}/row)')
print(f'Avg terms_grounded_pct: {grounded_total/max(processed,1):.3f}')

print(f'\nCopying {local_jsonl} -> {output_jsonl} ...')
shutil.copy2(local_jsonl, output_jsonl)
print('  done.')

if failures:
    with local_failures_jsonl.open('w', encoding='utf-8') as f:
        for rec in failures:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    shutil.copy2(local_failures_jsonl, output_failures_jsonl)
elif output_failures_jsonl.exists():
    output_failures_jsonl.unlink()

preview_rows = []
for rec in records_for_preview:
    e = rec['llm_enrichment']
    g = rec['llm_generation']
    preview_rows.append({
        '_source_row': rec['_source_row'],
        'citation': rec['citation'],
        'status': g['status'],
        'role': e.get('provision_role_llm'),
        'specificity_score': e.get('specificity_score'),
        'english_summary': e.get('english_summary'),
        'legal_rule': e.get('legal_rule'),
        'legal_question': e.get('legal_question'),
        'concepts_en': ' | '.join(e.get('concepts_en', [])),
        'terms_de_to_en': ' | '.join(f"{t['de']}->{t['en']}" for t in e.get('terms_de_to_en', [])),
        'addressees': ' | '.join(e.get('addressees', [])),
        'sanctions_or_consequences': ' | '.join(e.get('sanctions_or_consequences', [])),
        'applicability_conditions': ' | '.join(e.get('applicability_conditions', [])),
        'exceptions_or_limitations': ' | '.join(e.get('exceptions_or_limitations', [])),
        'terms_grounded_pct': rec['llm_quality']['terms_grounded_pct'],
    })
preview_df = pd.DataFrame(preview_rows)
preview_df.to_csv(output_preview_csv, index=False)

metrics = {
    'start': cfg.start,
    'limit': cfg.limit,
    'selected_rows': len(work_df),
    'written_rows_this_session': processed,
    'total_rows_in_file': len(done_rows) + processed,
    'failures': len(failures),
    'elapsed_seconds': elapsed,
    'rows_per_second': processed / max(elapsed, 1e-9),
    'prompt_tokens_total': prompt_token_total,
    'output_tokens_total': output_token_total,
    'tokens_per_second': (prompt_token_total + output_token_total) / max(elapsed, 1e-9),
    'output_tokens_per_second': output_token_total / max(elapsed, 1e-9),
    'avg_terms_grounded_pct': grounded_total / max(processed, 1),
    'attention_backend': chosen_backend,
    'quantization': chosen_quantization,
    'kv_cache_dtype': cfg.kv_cache_dtype,
    'speculative': chosen_speculation,
    'status_counts': dict(status_counter),
    'provision_role_distribution': dict(role_counter),
    'output_jsonl': str(output_jsonl),
    'output_preview_csv': str(output_preview_csv),
    'output_failures_jsonl': str(output_failures_jsonl) if failures else None,
    'config': {k: (str(v) if not isinstance(v, (int, float, bool, str, list, dict, type(None))) else v)
               for k, v in asdict(cfg).items()},
}
output_metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')

print(json.dumps(metrics, indent=2, default=str))
display(preview_df.head(50))
'''


def main() -> None:
    nb = json.loads(SRC.read_text(encoding='utf-8'))
    # Cell index map (verified): 3 = Config, 5 = Schema/prompt, 8 = Helpers, 9 = Run loop
    nb['cells'][3] = mk_code(CELL2)
    nb['cells'][5] = mk_code(CELL4)
    nb['cells'][8] = mk_code(CELL7)
    nb['cells'][9] = mk_code(CELL8)
    DST.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
    print('Wrote:', DST)
    print('Size :', DST.stat().st_size, 'bytes')

    nb2 = json.loads(DST.read_text(encoding='utf-8'))
    print('cells:', len(nb2['cells']))
    for i, c in enumerate(nb2['cells']):
        src = ''.join(c.get('source', []))
        first = (src.split('\n', 1)[0] if src else '')[:80]
        print(f'  cell {i:>2}  {c["cell_type"]:<8}  | {first}')


if __name__ == '__main__':
    main()
