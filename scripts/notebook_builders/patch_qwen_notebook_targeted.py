#!/usr/bin/env python
"""Patch the Qwen Colab notebook for targeted, grounded enrichment."""

from __future__ import annotations

import json
import sys
from pathlib import Path


CELL6 = r'''from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path('/content/drive/MyDrive/swiss_law') if IN_COLAB else Path('..').resolve()
DATA_DIR     = BASE_DIR / 'data'
INSIGHTS_DIR = BASE_DIR / 'data_insights'
ART_DIR      = BASE_DIR / 'artifacts'
SCRIPT_DIR   = BASE_DIR / 'scripts'

for d in [DATA_DIR, INSIGHTS_DIR, ART_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_ID     = 'Qwen/Qwen3.5-35B-A3B'
QUANTIZATION = None

# ── Inference / throughput ───────────────────────────────────────────────────
GPU_MEMORY_UTIL      = 0.90
MAX_MODEL_LEN        = 4096
BATCH_SIZE           = 32
TEMPERATURE          = 0.0
TOP_P                = 1.0
REPETITION_PENALTY   = 1.03
MAX_TOKENS           = 640
RETRY_MAX_TOKENS     = 1024
SMOKE_MAX_TOKENS     = 1024
TENSOR_PARALLEL      = 1
TEXT_CHARS           = 1500
ENABLE_PREFIX_CACHE  = True

# ── Targeted enrichment ──────────────────────────────────────────────────────
# Fast path: upload court_authority_cards_v4_target_cards.jsonl and enrich only
# that compact file. This avoids scanning all 2.47M v4 cards in Colab.
ENRICH_TARGET_CARD_FILE_ONLY = True
TARGET_CARD_FILE     = ART_DIR / 'court_authority_cards_v4_target_cards.jsonl'

# Fallback path: scan v4 and only call Qwen for line indices in TARGET_FILE.
TARGET_FILE          = ART_DIR / 'court_authority_cards_v4_enrichment_targets.jsonl'
TARGETED_ENRICHMENT  = True
WRITE_DETERMINISTIC_FALLBACK_FOR_UNTARGETED = True

# First test: 1000. Full targeted run: 0.
TARGET_LIMIT         = 1000
LIMIT                = TARGET_LIMIT if ENRICH_TARGET_CARD_FILE_ONLY else 0

# ── Quality guardrails ───────────────────────────────────────────────────────
RETRY_BAD_JSON              = True
ABORT_IF_ERROR_RATE_ABOVE   = 0.03
ERROR_RATE_CHECK_AFTER      = 200
VALIDATE_REFERENCE_GROUNDING = True

# ── Files ────────────────────────────────────────────────────────────────────
INPUT_FILE      = TARGET_CARD_FILE if ENRICH_TARGET_CARD_FILE_ONLY else ART_DIR / 'court_authority_cards_v4.jsonl'
OUTPUT_FILE     = ART_DIR / 'court_authority_cards_rag_targets.jsonl' if ENRICH_TARGET_CARD_FILE_ONLY else ART_DIR / 'court_authority_cards_rag.jsonl'
CHECKPOINT_FILE = ART_DIR / 'rag_checkpoint.txt'

RESET_OUTPUT = False

if RESET_OUTPUT:
    OUTPUT_FILE.unlink(missing_ok=True)
    CHECKPOINT_FILE.unlink(missing_ok=True)

print('BASE_DIR   :', BASE_DIR)
print('INPUT_FILE :', INPUT_FILE)
print('OUTPUT_FILE:', OUTPUT_FILE)
print('TARGET_CARD_FILE:', TARGET_CARD_FILE)
print('TARGET_FILE:', TARGET_FILE)
print('MODEL_ID   :', MODEL_ID)
print(
    f'BATCH_SIZE={BATCH_SIZE}, MAX_TOKENS={MAX_TOKENS}, '
    f'RETRY_MAX_TOKENS={RETRY_MAX_TOKENS}, SMOKE_MAX_TOKENS={SMOKE_MAX_TOKENS}'
)
print(
    f'ENRICH_TARGET_CARD_FILE_ONLY={ENRICH_TARGET_CARD_FILE_ONLY}, '
    f'TARGETED_ENRICHMENT={TARGETED_ENRICHMENT}, TARGET_LIMIT={TARGET_LIMIT}, '
    f'WRITE_FALLBACK={WRITE_DETERMINISTIC_FALLBACK_FOR_UNTARGETED}'
)
'''


CELL8 = r'''import re

# ── Pre-filter: deterministic paragraphs that do not need the LLM ────────────
COST_PROC_RE = re.compile(
    r'(?:'
    r'\bgerichtskosten\b|\bprozesskosten\b|\bverfahrenskosten\b|'
    r'\bfrais judiciaires\b|\bfrais de la cause\b|\bd[eé]pens\b|'
    r'\bspese giudiziarie\b|\bripetibili\b|'
    r'\bparteientsch[äa]digung\b|\bhonoraire\b|'
    r'\bunentgeltliche rechtspflege\b|\bassistance judiciaire\b|'
    r'\bpatrocinio gratuito\b|'
    r'^\s*\d+\.\s*\d+\..{0,5}fr\.\s*\d'
    r')',
    re.IGNORECASE | re.MULTILINE,
)

REMITTAL_RE = re.compile(
    r'(?:'
    r'\b(?:die\s+)?sache\s+wird.{0,140}\bzur(?:[üu?]ck|ueck)gewiesen\b|'
    r'\bzur\s+(?:neuen|erneuten)\s+(?:entscheidung|beurteilung|neubeurteilung).{0,120}\bzur(?:[üu?]ck|ueck)gewiesen\b|'
    r'\ban\s+die\s+(?:vorinstanz|beschwerdegegnerin|verwaltung|beh[öo?]rde).{0,140}\bzur(?:[üu?]ck|ueck)gewiesen\b|'
    r'\brenvoie\s+la\s+cause\b|\bla\s+cause\s+est\s+renvoy[ée]e\b|'
    r"\brenvoy[ée]\s+.{0,80}\b(?:l'autorit[ée]|tribunal|instance)\b|"
    r'\bla\s+causa\s+[èe]\s+rinviata\b|\brinvia\s+la\s+causa\b'
    r')',
    re.IGNORECASE,
)

INADMISSIBLE_RE = re.compile(
    r'(?:\bauf\s+die\s+beschwerde\s+wird\s+nicht\s+eingetreten\b|\birrecevable\b|\binammissibile\b)',
    re.IGNORECASE,
)
DISMISSED_RE = re.compile(
    r'(?:\bbeschwerde\s+wird\s+abgewiesen\b|\ble\s+recours\s+est\s+rejet[ée]\b|\bil\s+ricorso\s+[èe]\s+respinto\b)',
    re.IGNORECASE,
)
GRANTED_RE = re.compile(
    r'(?:\bbeschwerde\s+wird\s+gutgeheissen\b|\ble\s+recours\s+est\s+admis\b|\bil\s+ricorso\s+[èe]\s+accolto\b)',
    re.IGNORECASE,
)
PARTIAL_RE = re.compile(r'(?:\bteilweise\b|\bpartiellement\b|\bparzialmente\b)', re.IGNORECASE)

RAG_SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'english_summary':          {'type': 'string', 'maxLength': 320},
        'legal_topic':              {'type': 'string', 'maxLength': 140},
        'legal_question':           {'type': 'string', 'maxLength': 260},
        'legal_rule':               {'type': 'string', 'maxLength': 320},
        'court_holding':            {'type': 'string', 'maxLength': 260},
        'factual_context':          {'type': 'string', 'maxLength': 260},
        'english_legal_concepts':   {'type': 'array', 'items': {'type': 'string', 'maxLength': 80}, 'minItems': 0, 'maxItems': 6},
        'search_keywords':          {'type': 'array', 'items': {'type': 'string', 'maxLength': 80}, 'minItems': 0, 'maxItems': 8},
        'natural_language_queries': {'type': 'array', 'items': {'type': 'string', 'maxLength': 180}, 'minItems': 0, 'maxItems': 3},
        'paragraph_role': {
            'type': 'string',
            'enum': ['holding', 'reasoning', 'background', 'cost', 'procedural', 'disposition', 'standard_of_review', 'obiter'],
        },
        'outcome_signal': {
            'type': 'string',
            'enum': ['granted', 'dismissed', 'inadmissible', 'remitted', 'partial', 'none'],
        },
    },
    'required': [
        'english_summary', 'legal_topic', 'legal_question', 'legal_rule',
        'court_holding', 'factual_context', 'english_legal_concepts',
        'search_keywords', 'natural_language_queries', 'paragraph_role',
        'outcome_signal',
    ],
}

SYSTEM_PROMPT = (
    'You are a deterministic Swiss legal JSON extraction engine. '
    'Input is one paragraph from a Swiss Federal Tribunal decision in German, French, or Italian, plus deterministic metadata. '
    'Return exactly one complete JSON object matching the schema. '
    'Use concise English legal terminology for semantic search. '
    'Use only statutes, articles, and case citations present in the paragraph or metadata. Do not invent article numbers or citations. '
    'If no legal rule, holding, facts, statute, or citation is supported by the paragraph/metadata, use an empty string, empty array, or "none". '
    'Do not repeat the source text. Do not produce bibliography, markdown, tables, or chain-of-thought. '
    'Keep every field brief; complete valid grounded JSON is more important than detail.'
)

print('Schema fields:', list(RAG_SCHEMA['properties'].keys()))
print('Required fields:', RAG_SCHEMA['required'])
'''


CELL10 = r'''import json
import re
from pathlib import Path
from typing import Iterator, Any

BGE_REF_RE = re.compile(r'\bBGE\s+\d{3}\s+[IVX]{1,4}\s+\d+[a-z]?\b', re.IGNORECASE)
DOCKET_REF_RE = re.compile(r'\b\d{1,2}[A-Z]{1,4}[_\.]\d{1,5}/\d{4}\b', re.IGNORECASE)
ART_REF_RE = re.compile(r'\b(?:Art\.?|Article|Articles)\s*\d+[A-Za-z0-9.]*', re.IGNORECASE)
ART_HEAD_RE = re.compile(r'\b(?:art\.?|artt\.?|article|articles)\s+([^\n]{0,140})', re.IGNORECASE)
ART_GENERATED_RE = re.compile(r'\b(?:art\.?|article|articles)\s+(\d+[a-z]?(?:\s*-\s*\d+[a-z]?)?)', re.IGNORECASE)
ART_NUM_RE = re.compile(r'\d+[a-z]?', re.IGNORECASE)

def _txt(value: Any) -> str:
    if value is None:
        return ''
    return re.sub(r'\s+', ' ', str(value)).strip()

def _list(value: Any, max_items: int = 30) -> list[str]:
    if value is None:
        return []
    raw = value if isinstance(value, list) else [value]
    out, seen = [], set()
    for item in raw:
        s = _txt(item)
        if s and s.lower() not in seen:
            out.append(s)
            seen.add(s.lower())
        if len(out) >= max_items:
            break
    return out

def _term_values(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    out = []
    for concept, terms in value.items():
        out.append(concept)
        out.extend(_list(terms, max_items=20))
    return _list(out, max_items=60)

def _join(items: Any, max_items: int = 20) -> str:
    return '; '.join(_list(items, max_items=max_items))

def load_target_line_indices(path: Path, target_limit: int = 0) -> tuple[set[int], dict[int, dict]]:
    if not path.exists():
        print(f'[targets] Target file not found: {path}')
        return set(), {}
    indices: set[int] = set()
    meta: dict[int, dict] = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            idx = int(row['line_idx'])
            indices.add(idx)
            meta[idx] = row
            if target_limit and len(indices) >= target_limit:
                break
    print(f'[targets] loaded {len(indices):,} target line indices from {path.name}')
    return indices, meta

def build_user_message(card: dict) -> str:
    text = (card.get('text_excerpt_original', '') or '')[:TEXT_CHARS]
    parts = [
        '<record>',
        '<metadata>',
        f'<citation>{card.get("citation", "") or ""}</citation>',
        f'<court_base>{card.get("court_base", "") or ""}</court_base>',
        f'<legal_area>{card.get("legal_area", "") or ""}</legal_area>',
        f'<authority_role>{_join(card.get("authority_role") or [])}</authority_role>',
        f'<existing_labels>{_join(card.get("issue_labels_en") or [])}</existing_labels>',
        f'<law_codes>{_join(card.get("law_codes") or [])}</law_codes>',
        f'<statutes_cited>{_join(card.get("statutes_cited") or [])}</statutes_cited>',
        f'<court_cases_cited>{_join(card.get("court_cases_cited") or [])}</court_cases_cited>',
        '</metadata>',
        '<paragraph_original_language>',
        text,
        '</paragraph_original_language>',
        '</record>',
        '',
        'Create concise English RAG metadata.',
        'Use only references present inside <metadata> or <paragraph_original_language>.',
        'Return exactly one complete JSON object. No prose outside JSON.',
    ]
    return '\n'.join(parts)

def _stub(summary, topic, concepts, keywords, role, method, *, outcome='none', question='', rule='', holding='', facts='', queries=None):
    return {
        'english_summary':          _txt(summary),
        'legal_topic':              _txt(topic),
        'legal_question':           _txt(question),
        'legal_rule':               _txt(rule),
        'court_holding':            _txt(holding),
        'factual_context':          _txt(facts),
        'english_legal_concepts':   _list(concepts, max_items=6),
        'search_keywords':          _list(keywords, max_items=8),
        'natural_language_queries': _list(queries or [], max_items=3),
        'paragraph_role':           role,
        'outcome_signal':           outcome,
        'method':                   method,
    }

def deterministic_disposition(card: dict) -> dict | None:
    text = card.get('text_excerpt_original', '') or ''
    head = text[:900]
    if REMITTAL_RE.search(head):
        return _stub(
            'The matter is remitted to a lower authority or previous instance for further proceedings.',
            'remittal to lower authority',
            ['remittal', 'procedural disposition'],
            ['remittal', 'renvoi', 'zurückgewiesen', 'rinvio'],
            role='disposition', outcome='remitted', method='auto_remittal',
            holding='Matter remitted for further proceedings.'
        )
    if INADMISSIBLE_RE.search(head):
        return _stub('The appeal or application is held inadmissible.', 'inadmissibility of appeal',
                     ['inadmissibility', 'appeal procedure'],
                     ['inadmissible', 'nicht eingetreten', 'irrecevable', 'inammissibile'],
                     role='disposition', outcome='inadmissible', method='auto_inadmissible',
                     holding='Appeal held inadmissible.')
    if DISMISSED_RE.search(head):
        return _stub('The appeal is dismissed.', 'dismissal of appeal',
                     ['dismissal of appeal', 'appeal procedure'],
                     ['dismissed', 'abgewiesen', 'rejeté', 'respinto'],
                     role='disposition', outcome='dismissed', method='auto_dismissed',
                     holding='Appeal dismissed.')
    if GRANTED_RE.search(head):
        outcome = 'partial' if PARTIAL_RE.search(head) else 'granted'
        return _stub('The appeal is granted or partially granted.', 'appeal granted',
                     ['appeal granted', 'appeal procedure'],
                     ['granted', 'gutgeheissen', 'admis', 'accolto'],
                     role='disposition', outcome=outcome, method='auto_granted',
                     holding='Appeal granted or partially granted.')
    return None

def deterministic_fallback(card: dict, *, method='deterministic_v4_fallback') -> dict:
    labels = _list(card.get('issue_labels_en'), max_items=6)
    legal_area = _txt(card.get('legal_area'))
    law_codes = _list(card.get('law_codes'), max_items=8)
    statutes = _list(card.get('statutes_cited'), max_items=8)
    cases = _list(card.get('court_cases_cited'), max_items=5)
    terms = _term_values(card.get('matched_terms_multilingual'))
    summary = card.get('summary_en_proxy') or f'Swiss Federal Supreme Court consideration in {legal_area}.'.strip()
    topic_parts = labels[:3] or ([legal_area] if legal_area else ['Swiss Federal Tribunal authority'])
    keywords = _list(labels + terms + law_codes + statutes + cases, max_items=8)
    queries = []
    if labels:
        queries.append('Swiss Federal Tribunal authority on ' + ', '.join(labels[:3]))
    if statutes:
        queries.append('Swiss case law applying ' + ', '.join(statutes[:2]))
    return _stub(summary, ' — '.join(topic_parts), labels, keywords,
                 role='procedural' if card.get('is_notification_paragraph') else 'reasoning',
                 outcome='none', method=method, rule='; '.join(statutes[:3]), queries=queries)

def auto_classify(card: dict) -> dict | None:
    if card.get('is_notification_paragraph'):
        return _stub('Procedural notification of the judgment to the parties.', 'judgment notification',
                     ['service of judgment'], ['notification', 'service', 'judgment communication'],
                     role='procedural', method='auto_notification')
    text = card.get('text_excerpt_original', '') or ''
    disposition = deterministic_disposition(card)
    if disposition is not None:
        return disposition
    if len(text) < 50:
        return _stub('Short procedural fragment, cross-reference, or one-line ruling.', 'procedural fragment',
                     [], [], role='procedural', method='auto_short')
    if COST_PROC_RE.search(text[:400]):
        return _stub('Court-cost, procedural-fee, or legal-aid paragraph.', 'court costs and procedural fees',
                     ['court costs', 'procedural fees', 'legal aid'],
                     ['costs', 'court fees', 'frais judiciaires', 'Gerichtskosten', 'legal aid'],
                     role='cost', method='auto_cost')
    return None

def stream_input(path: Path, start_offset: int) -> Iterator[tuple[int, dict]]:
    with path.open(encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < start_offset or not line.strip():
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                continue

def count_lines(path: Path) -> int:
    n = 0
    with path.open('rb') as f:
        for _ in f:
            n += 1
    return n

def render_prompt(card: dict, *, extra_guard: str = '') -> str:
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': build_user_message(card) + extra_guard},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt + '\nReturn one complete valid grounded JSON object only.\n'

def clean_model_json(raw: str) -> str:
    raw = (raw or '').strip()
    if '</think>' in raw:
        raw = raw.split('</think>', 1)[1].strip()
    if raw.startswith('```json'):
        raw = raw[7:].strip()
    if raw.startswith('```'):
        raw = raw[3:].strip()
    if raw.endswith('```'):
        raw = raw[:-3].strip()
    if raw and not raw.startswith('{'):
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end + 1].strip()
    return raw

def _clip_str(value: Any, max_len: int) -> str:
    return _txt(value)[:max_len].rstrip()

def validate_and_normalize_enrichment(enriched: Any, *, method: str) -> dict:
    if not isinstance(enriched, dict):
        raise ValueError(f'Expected dict, got {type(enriched).__name__}')
    missing = [k for k in RAG_SCHEMA['required'] if k not in enriched]
    if missing:
        raise ValueError(f'Missing required fields: {missing}')
    for key, max_len in {
        'english_summary': 320, 'legal_topic': 140, 'legal_question': 260,
        'legal_rule': 320, 'court_holding': 260, 'factual_context': 260,
    }.items():
        enriched[key] = _clip_str(enriched.get(key, ''), max_len)
    for key, max_items, item_len in [
        ('english_legal_concepts', 6, 80),
        ('search_keywords', 8, 80),
        ('natural_language_queries', 3, 180),
    ]:
        value = enriched.get(key, [])
        if not isinstance(value, list):
            value = [value] if value else []
        cleaned, seen = [], set()
        for item in value:
            s = _clip_str(item, item_len)
            if s and s.lower() not in seen:
                cleaned.append(s)
                seen.add(s.lower())
            if len(cleaned) >= max_items:
                break
        enriched[key] = cleaned
    if enriched.get('paragraph_role') not in RAG_SCHEMA['properties']['paragraph_role']['enum']:
        enriched['paragraph_role'] = 'reasoning'
    if enriched.get('outcome_signal') not in RAG_SCHEMA['properties']['outcome_signal']['enum']:
        enriched['outcome_signal'] = 'none'
    enriched['method'] = method
    return enriched

def _reference_blob(card: dict) -> str:
    pieces = [
        card.get('citation'), card.get('court_base'), card.get('text_excerpt_original'),
        card.get('legal_area'), card.get('law_codes'), card.get('statutes_cited'),
        card.get('court_cases_cited'),
    ]
    return ' | '.join(_list(pieces, max_items=200)).lower().replace('_', '.')

def _article_ids_from_blob(blob: str) -> set[str]:
    ids = set()
    for match in ART_HEAD_RE.finditer(blob):
        segment = match.group(1)
        nums = ART_NUM_RE.findall(segment)
        if not nums:
            continue
        # First number after Art./artt. is always an article. Additional numbers
        # are article numbers only when chained/ranged by conjunctions or dashes.
        ids.add(nums[0].lower())
        for chained in re.findall(r'(?:,|;|-|\bet\b|\be\b|\bund\b|\band\b)\s*(\d+[a-z]?)', segment, flags=re.IGNORECASE):
            ids.add(chained.lower())
    return ids

def _generated_article_ids(payload: str) -> list[tuple[str, str]]:
    out = []
    for match in ART_GENERATED_RE.finditer(payload):
        ref = match.group(0)
        ids = ART_NUM_RE.findall(match.group(1))
        for article_id in ids:
            out.append((ref, article_id.lower()))
    return out

def validate_reference_grounding(enriched: dict, card: dict) -> None:
    if not VALIDATE_REFERENCE_GROUNDING:
        return
    blob = _reference_blob(card)
    payload = json.dumps(enriched, ensure_ascii=False).lower().replace('_', '.')
    bad = []
    allowed_article_ids = _article_ids_from_blob(blob)
    for art_ref, article_id in _generated_article_ids(payload):
        if article_id in allowed_article_ids:
            continue
        # Tolerate compacting "Art. 6 f." to "Art. 6f" when the source has
        # the spaced form. Do not tolerate converting "§ 9" into "Art. 9".
        if article_id.endswith('f') and article_id[:-1] in allowed_article_ids:
            compact_source = re.search(rf'\bart\.?\s*{re.escape(article_id[:-1])}\s*f\b', blob, flags=re.IGNORECASE)
            if compact_source:
                continue
        bad.append(art_ref)
    for bge in BGE_REF_RE.findall(payload):
        if bge.lower() not in blob:
            bad.append(bge)
    for docket in DOCKET_REF_RE.findall(payload):
        if docket.lower().replace('_', '.') not in blob:
            bad.append(docket)
    if bad:
        raise ValueError('Ungrounded legal references generated: ' + ', '.join(sorted(set(bad))[:8]))

def parse_output_text(raw_text: str, *, method: str, card: dict | None = None) -> dict:
    enriched = json.loads(clean_model_json(raw_text))
    enriched = validate_and_normalize_enrichment(enriched, method=method)
    if card is not None:
        validate_reference_grounding(enriched, card)
    return enriched

print('Helpers loaded.')
'''


CELL14 = r'''# Optional smoke test: set RUN_SMOKE_TEST = True and run this cell.
# Prints an explicit GREEN/RED signal. Uses target rows when TARGET_FILE exists.
RUN_SMOKE_TEST = True
SMOKE_N = 8

if RUN_SMOKE_TEST:
    smoke_target_read_limit = TARGET_LIMIT if TARGET_LIMIT else 50000
    if ENRICH_TARGET_CARD_FILE_ONLY:
        smoke_targets = set()
    else:
        smoke_targets, _ = load_target_line_indices(TARGET_FILE, target_limit=smoke_target_read_limit) if TARGETED_ENRICHMENT else (set(), {})
    samples = []
    fallback_checked = 0

    for line_idx, card in stream_input(INPUT_FILE, 0):
        auto = auto_classify(card)
        if auto is not None and fallback_checked < 3:
            validate_and_normalize_enrichment(auto, method=auto.get('method', 'auto'))
            fallback_checked += 1

        if (not ENRICH_TARGET_CARD_FILE_ONLY) and smoke_targets and line_idx not in smoke_targets:
            continue
        if auto_classify(card) is None:
            samples.append((line_idx, card))
        if len(samples) >= SMOKE_N:
            break

    if not samples:
        raise RuntimeError('RED: no target LLM samples found for smoke test. Check TARGET_FILE and target selection.')

    prompts = [render_prompt(card) for _, card in samples]
    outs = llm.generate(prompts, sampling_params=smoke_sampling_params, use_tqdm=False)

    failures = 0
    for i, ((line_idx, card), out) in enumerate(zip(samples, outs)):
        raw = out.outputs[0].text
        finish_reason = getattr(out.outputs[0], 'finish_reason', None)
        print('\n' + '=' * 80)
        print('SAMPLE', i, '| line_idx:', line_idx, '| citation:', card.get('citation', ''), '| finish_reason:', finish_reason)
        print('metadata statutes:', card.get('statutes_cited') or [])
        print('metadata cases   :', card.get('court_cases_cited') or [])
        try:
            obj = parse_output_text(raw, method='smoke', card=card)
            print(json.dumps(obj, ensure_ascii=False, indent=2)[:1800])
        except Exception as e:
            failures += 1
            cleaned = clean_model_json(raw)
            print('FAILED:', repr(e))
            print('raw_chars:', len(raw), 'cleaned_chars:', len(cleaned))
            print(cleaned[:2000])

    print('\nDeterministic fallback checks:', fallback_checked)
    if failures:
        print(f'RED: smoke test failed: {failures}/{len(samples)} invalid or ungrounded outputs.')
        raise RuntimeError(f'RED: smoke test failed: {failures}/{len(samples)} invalid or ungrounded outputs.')

    print(f'GREEN: smoke test passed. {len(samples)} target LLM outputs valid and reference-grounded; deterministic fallback callable.')
'''


CELL17 = r'''from tqdm.auto import tqdm
import time

if not INPUT_FILE.exists():
    raise FileNotFoundError(f'Input not found: {INPUT_FILE}')

target_line_indices: set[int] = set()
target_meta: dict[int, dict] = {}
if TARGETED_ENRICHMENT and not ENRICH_TARGET_CARD_FILE_ONLY:
    target_line_indices, target_meta = load_target_line_indices(TARGET_FILE, TARGET_LIMIT)
    if not target_line_indices:
        raise FileNotFoundError(f'TARGETED_ENRICHMENT=True but no target rows loaded from {TARGET_FILE}')

start = 0
if CHECKPOINT_FILE.exists():
    try:
        start = int(CHECKPOINT_FILE.read_text().strip() or '0')
    except ValueError:
        start = 0

print(f'Resuming at line {start:,}')
print(f'Counting lines in {INPUT_FILE.name} ...')

total_lines = count_lines(INPUT_FILE)
target_total = min(total_lines, start + LIMIT) if LIMIT else total_lines

print(f'Total={total_lines:,}  To process={target_total - start:,}')
print(f'Using BATCH_SIZE={BATCH_SIZE}, MAX_TOKENS={MAX_TOKENS}, RETRY_MAX_TOKENS={RETRY_MAX_TOKENS}')
if TARGETED_ENRICHMENT:
    remaining_targets = (target_total - start) if ENRICH_TARGET_CARD_FILE_ONLY else sum(1 for idx in target_line_indices if start <= idx < target_total)
    print(f'Targeted LLM rows in processing range: {remaining_targets:,}')

out_f = OUTPUT_FILE.open('a', encoding='utf-8')
pbar = tqdm(total=target_total, initial=start, desc='enrich', unit='card', smoothing=0.03)

pending: list[tuple[int, dict]] = []
json_errors = 0
llm_attempts = 0
retry_count = 0
batch_count = 0
llm_target_count = 0
fallback_count = 0
auto_count = 0
started_at = time.time()

def retry_one(card: dict) -> tuple[dict, str | None]:
    retry_guard = (
        '\n\nPrevious generation was invalid, incomplete, or used an ungrounded legal reference. '
        'Return one complete concise JSON object only. Use only references present in metadata or paragraph. '
        'No prose, no markdown, no citations outside JSON fields.'
    )
    out = llm.generate([render_prompt(card, extra_guard=retry_guard)], sampling_params=retry_sampling_params, use_tqdm=False)[0]
    raw = out.outputs[0].text
    try:
        return parse_output_text(raw, method='qwen35_35b_a3b_vllm_retry', card=card), None
    except Exception as e:
        return (
            deterministic_fallback(card, method='deterministic_fallback_after_parse_failed'),
            f'{type(e).__name__}: {str(e)[:200]} | raw={clean_model_json(raw)[:300]}',
        )

def flush_batch():
    global pending, json_errors, llm_attempts, retry_count, batch_count, llm_target_count
    if not pending:
        return
    batch_count += 1
    batch_size_now = len(pending)
    prompts = [render_prompt(card) for _, card in pending]
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    dt = time.time() - t0
    llm_attempts += batch_size_now
    llm_target_count += batch_size_now
    batch_json_errors = 0
    batch_retries = 0

    for (line_idx, card), out in zip(pending, outputs):
        raw_text = out.outputs[0].text
        try:
            enriched = parse_output_text(raw_text, method='qwen35_35b_a3b_vllm', card=card)
        except Exception as first_error:
            if RETRY_BAD_JSON:
                batch_retries += 1
                retry_count += 1
                enriched, retry_error = retry_one(card)
                llm_attempts += 1
                if retry_error is not None:
                    json_errors += 1
                    batch_json_errors += 1
                    enriched['parse_error'] = retry_error
                    enriched['raw_output'] = clean_model_json(raw_text)[:400]
            else:
                json_errors += 1
                batch_json_errors += 1
                enriched = deterministic_fallback(card, method='deterministic_fallback_after_parse_failed')
                enriched['parse_error'] = f'{type(first_error).__name__}: {str(first_error)[:200]}'
                enriched['raw_output'] = clean_model_json(raw_text)[:400]

        card['rag_enrichment'] = enriched
        out_f.write(json.dumps(card, ensure_ascii=False) + '\n')

    out_f.flush()
    CHECKPOINT_FILE.write_text(str(pending[-1][0] + 1))
    pbar.update(len(pending))

    elapsed = time.time() - started_at
    done = max(pbar.n - start, 0)
    remaining = max(target_total - pbar.n, 0)
    avg_rate = done / max(elapsed, 1e-9)
    batch_rate = batch_size_now / max(dt, 1e-9)
    eta_min = remaining / max(avg_rate, 1e-9) / 60.0

    if batch_count == 1 or batch_count % 5 == 0 or batch_json_errors or batch_retries:
        print(
            f'[batch {batch_count}] llm_size={batch_size_now}, '
            f'batch_time={dt:.1f}s, batch_rate={batch_rate:.2f} target_cards/s, '
            f'avg_all_rows_rate={avg_rate:.2f} rows/s, eta≈{eta_min:.1f} min, '
            f'batch_retries={batch_retries}, batch_json_errors={batch_json_errors}, '
            f'total_json_errors={json_errors}, retry_count={retry_count}, '
            f'llm_targets={llm_target_count}, fallbacks={fallback_count}, autos={auto_count}, llm_attempts={llm_attempts}'
        )

    if llm_attempts >= ERROR_RATE_CHECK_AFTER:
        err_rate = json_errors / max(llm_attempts, 1)
        if err_rate > ABORT_IF_ERROR_RATE_ABOVE:
            raise RuntimeError(
                f'Aborting: JSON/reference-grounding error rate {err_rate:.1%} exceeds '
                f'{ABORT_IF_ERROR_RATE_ABOVE:.1%}. Run smoke test and inspect failed raw outputs before scaling.'
            )
    pending.clear()

processed = 0
try:
    for line_idx, card in stream_input(INPUT_FILE, start):
        if LIMIT and processed >= LIMIT:
            break
        should_llm = ENRICH_TARGET_CARD_FILE_ONLY or (not TARGETED_ENRICHMENT) or (line_idx in target_line_indices)

        if not should_llm:
            if WRITE_DETERMINISTIC_FALLBACK_FOR_UNTARGETED:
                auto = auto_classify(card)
                card['rag_enrichment'] = auto if auto is not None else deterministic_fallback(card)
                if auto is not None:
                    auto_count += 1
                else:
                    fallback_count += 1
                out_f.write(json.dumps(card, ensure_ascii=False) + '\n')
                CHECKPOINT_FILE.write_text(str(line_idx + 1))
                pbar.update(1)
            processed += 1
            continue

        auto = auto_classify(card)
        if auto is not None:
            card['rag_enrichment'] = auto
            out_f.write(json.dumps(card, ensure_ascii=False) + '\n')
            CHECKPOINT_FILE.write_text(str(line_idx + 1))
            pbar.update(1)
            auto_count += 1
            processed += 1
            continue

        pending.append((line_idx, card))
        if len(pending) >= BATCH_SIZE:
            flush_batch()
        processed += 1
    flush_batch()
finally:
    out_f.close()
    pbar.close()

print(f'Done. JSON/reference-grounding errors after retry: {json_errors}')
print(f'Retries used: {retry_count}')
print(f'LLM target rows enriched: {llm_target_count}')
print(f'Deterministic fallback rows: {fallback_count}')
print(f'Auto-classified rows: {auto_count}')
print(f'Output → {OUTPUT_FILE}')
'''


CELL19 = r'''from collections import Counter
import json

roles = Counter()
outcomes = Counter()
methods = Counter()
total_out = 0
missing_required = 0
bad = []

REQUIRED = RAG_SCHEMA['required']

if not OUTPUT_FILE.exists():
    raise FileNotFoundError(f'Output not found: {OUTPUT_FILE}')

with OUTPUT_FILE.open(encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        card = json.loads(line)
        e = card.get('rag_enrichment', {}) or {}
        total_out += 1
        roles[e.get('paragraph_role', 'MISSING')] += 1
        outcomes[e.get('outcome_signal', 'MISSING')] += 1
        methods[e.get('method', 'MISSING')] += 1
        if any(k not in e for k in REQUIRED):
            missing_required += 1
            if len(bad) < 10:
                bad.append(card)
        if e.get('method') in {'json_parse_failed', 'deterministic_fallback_after_parse_failed'} and len(bad) < 10:
            bad.append(card)

print(f'Total output cards : {total_out:,}')
print(f'Missing required   : {missing_required:,}')
print(f'Parse fallback     : {methods.get("deterministic_fallback_after_parse_failed", 0):,}')
print()

if total_out:
    print('paragraph_role distribution:')
    for k, v in roles.most_common():
        print(f'  {k:<25} {v:>8}  ({v/total_out*100:.1f}%)')
    print()
    print('outcome_signal distribution:')
    for k, v in outcomes.most_common():
        print(f'  {k:<25} {v:>8}  ({v/total_out*100:.1f}%)')
    print()
    print('method distribution:')
    for k, v in methods.most_common():
        print(f'  {k:<40} {v:>8}  ({v/total_out*100:.1f}%)')

if bad:
    print('\nExamples needing inspection:')
    for i, card in enumerate(bad[:5]):
        e = card.get('rag_enrichment', {})
        print('\n--- BAD', i, '---')
        print('citation:', card.get('citation', ''))
        print('method:', e.get('method'))
        print('parse_error:', e.get('parse_error'))
        print('raw_output:', (e.get('raw_output') or '')[:500])

if missing_required or methods.get('deterministic_fallback_after_parse_failed', 0):
    print('\nRED: output has missing required fields or parse fallbacks needing inspection.')
else:
    print('\nGREEN: output has required RAG fields for every written card.')
'''


CELL20 = r'''# Show a few LLM-enriched cards for a quick quality check.
import random
import json

llm_cards = []
with OUTPUT_FILE.open(encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        card = json.loads(line)
        method = card.get('rag_enrichment', {}).get('method', '')
        if 'qwen35' in method and method != 'json_parse_failed':
            llm_cards.append(card)

print(f'LLM-enriched valid cards available: {len(llm_cards):,}')

for card in random.sample(llm_cards, min(3, len(llm_cards))):
    e = card['rag_enrichment']
    print('─' * 90)
    print('Citation     :', card.get('citation', ''))
    print('Target line  :', (card.get('_enrichment_target') or {}).get('line_idx'))
    print('Role         :', e.get('paragraph_role'))
    print('Outcome      :', e.get('outcome_signal'))
    print('Topic        :', e.get('legal_topic'))
    print('Question     :', e.get('legal_question', '')[:240])
    print('Rule         :', e.get('legal_rule', '')[:240])
    print('Holding      :', e.get('court_holding', '')[:240])
    print('Summary      :', e.get('english_summary', '')[:240])
    print('Concepts     :', e.get('english_legal_concepts'))
    print('Keywords     :', e.get('search_keywords'))
    print('NL queries   :', e.get('natural_language_queries'))
'''


def patch_notebook(path: Path) -> Path:
    nb = json.loads(path.read_text(encoding="utf-8"))
    backup = path.with_name(path.stem + ".backup_before_targeted_patch.ipynb")
    if not backup.exists():
        backup.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    replacements = {
        6: CELL6,
        8: CELL8,
        10: CELL10,
        14: CELL14,
        17: CELL17,
        19: CELL19,
        20: CELL20,
    }
    for idx, source in replacements.items():
        nb["cells"][idx]["cell_type"] = "code"
        nb["cells"][idx]["source"] = [line + "\n" for line in source.splitlines()]
        nb["cells"][idx]["outputs"] = []
        nb["cells"][idx]["execution_count"] = None

    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return backup


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: patch_qwen_notebook_targeted.py NOTEBOOK.ipynb", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    backup = patch_notebook(path)
    print(f"patched={path}")
    print(f"backup={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
