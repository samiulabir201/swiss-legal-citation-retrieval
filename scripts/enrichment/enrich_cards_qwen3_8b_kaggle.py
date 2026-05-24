#!/usr/bin/env python
"""Cheap Kaggle enrichment for Swiss court authority target cards.

This script is the low-cost replacement for the heavy Qwen3.5-35B notebook.
It enriches only the compact target-card JSONL, not the full 2.47M corpus.

Kaggle setup cell:

    !pip install -U "transformers>=4.51.0" accelerate tqdm
    # If Qwen/Qwen3-8B-AWQ loading fails, also run:
    # !pip install autoawq

Edit CONFIG below, then run:

    !python enrich_cards_qwen3_8b_kaggle.py

Expected Kaggle layout:

    input : /kaggle/input/<your-dataset>/court_authority_cards_v4_target_cards.jsonl
    output: /kaggle/working/court_authority_cards_rag_targets_qwen3_8b.jsonl

If your Kaggle dataset folder has a different name, change CONFIG.input_file.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".").resolve()
IS_KAGGLE = Path("/kaggle/working").exists()


def first_existing_target_file() -> Path:
    """Find the target-card file in common Kaggle/local locations."""
    candidates = [
        Path("/kaggle/input/swiss-law/court_authority_cards_v4_target_cards.jsonl"),
        Path("/kaggle/input/swiss-law-artifacts/court_authority_cards_v4_target_cards.jsonl"),
        Path("/kaggle/input/swiss-citation-extraction/court_authority_cards_v4_target_cards.jsonl"),
        ROOT / "artifacts" / "court_authority_cards_v4_target_cards.jsonl",
        ROOT / "artifacts" / "court_authority_cards_v4_target_cards_1000.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0] if IS_KAGGLE else candidates[-2]


@dataclass
class Config:
    # Files. On Kaggle, inputs are read-only and outputs must go to /kaggle/working.
    input_file: Path = first_existing_target_file()
    output_file: Path = (
        Path("/kaggle/working/court_authority_cards_rag_targets_qwen3_8b.jsonl")
        if IS_KAGGLE
        else ROOT / "artifacts" / "court_authority_cards_rag_targets_qwen3_8b.jsonl"
    )
    failed_file: Path = (
        Path("/kaggle/working/court_authority_cards_rag_targets_qwen3_8b_failed_cards.jsonl")
        if IS_KAGGLE
        else ROOT / "artifacts" / "court_authority_cards_rag_targets_qwen3_8b_failed_cards.jsonl"
    )
    checkpoint_file: Path = (
        Path("/kaggle/working/rag_targets_qwen3_8b_checkpoint.txt")
        if IS_KAGGLE
        else ROOT / "artifacts" / "rag_targets_qwen3_8b_checkpoint.txt"
    )

    # Model. AWQ is the safest size for free Kaggle 16GB GPUs.
    model_id: str = "Qwen/Qwen3-8B-AWQ"
    trust_remote_code: bool = True
    torch_dtype: str = "float16"
    device_map: str = "auto"
    attn_implementation: str | None = None

    # Optional fallback if you prefer the non-AWQ model with bitsandbytes NF4.
    # If this is True, set model_id = "Qwen/Qwen3-8B" and install bitsandbytes.
    use_bnb_4bit: bool = False

    # Run control. Start with 1000. Set limit=0 for the full 363,258 target run.
    limit: int = 1000
    reset_output: bool = True
    batch_size: int = 8
    progress_every_batches: int = 10

    # Prompt/generation. Keep these small for Kaggle throughput.
    text_chars: int = 700
    max_input_tokens: int = 1600
    max_new_tokens: int = 224
    retry_max_new_tokens: int = 384
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.05

    # Reliability.
    auto_classify: bool = True
    retry_bad_outputs: bool = True
    fallback_on_fail: bool = False
    validate_reference_grounding: bool = True


CONFIG = Config()


RAG_SCHEMA_HINT = {
    "english_summary": "string",
    "legal_topic": "string",
    "legal_question": "string",
    "legal_rule": "string",
    "court_holding": "string",
    "factual_context": "string",
    "english_legal_concepts": ["string"],
    "search_keywords": ["string"],
    "natural_language_queries": ["string"],
    "paragraph_role": "holding|reasoning|background|cost|procedural|disposition|standard_of_review|obiter",
    "outcome_signal": "granted|dismissed|inadmissible|remitted|partial|none",
}

REQUIRED_FIELDS = list(RAG_SCHEMA_HINT.keys())
ROLE_VALUES = {"holding", "reasoning", "background", "cost", "procedural", "disposition", "standard_of_review", "obiter"}
OUTCOME_VALUES = {"granted", "dismissed", "inadmissible", "remitted", "partial", "none"}

SYSTEM_PROMPT = (
    "You are a deterministic Swiss legal JSON extraction engine. "
    "Input is one paragraph from a Swiss Federal Tribunal decision in German, "
    "French, or Italian, plus deterministic metadata. Return exactly one JSON "
    "object. Use concise English legal terminology for semantic-search RAG. "
    "Use only statutes, articles, and case citations present in the paragraph "
    "or metadata. Do not invent article numbers, statutes, laws, or case "
    "citations. If no legal rule, holding, facts, statute, or citation is "
    "supported by the paragraph or metadata, use an empty string, empty array, "
    "or \"none\". No markdown. No commentary. No chain-of-thought."
)


COST_PROC_RE = re.compile(
    r"(?:"
    r"\bgerichtskosten\b|\bprozesskosten\b|\bverfahrenskosten\b|"
    r"\bfrais judiciaires\b|\bfrais de la cause\b|\bd[e\u00e9]pens\b|"
    r"\bspese giudiziarie\b|\bripetibili\b|"
    r"\bparteientsch[a\u00e4]digung\b|\bhonoraire\b|"
    r"\bunentgeltliche rechtspflege\b|\bassistance judiciaire\b|"
    r"\bpatrocinio gratuito\b|"
    r"^\s*\d+\.\s*\d+\..{0,5}fr\.\s*\d"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

REMITTAL_RE = re.compile(
    r"(?:"
    r"\b(?:die\s+)?sache\s+wird.{0,140}\bzur(?:ue|\u00fc|u)ckgewiesen\b|"
    r"\bzur\s+(?:neuen|erneuten)\s+(?:entscheidung|beurteilung|neubeurteilung).{0,120}\bzur(?:ue|\u00fc|u)ckgewiesen\b|"
    r"\ban\s+die\s+(?:vorinstanz|beschwerdegegnerin|verwaltung|beh[o\u00f6]rde).{0,140}\bzur(?:ue|\u00fc|u)ckgewiesen\b|"
    r"\brenvoie\s+la\s+cause\b|\bla\s+cause\s+est\s+renvoy[\u00e9e]e\b|"
    r"\brenvoy[\u00e9e]\s+.{0,80}\b(?:l'autorit[\u00e9e]|tribunal|instance)\b|"
    r"\bla\s+causa\s+[\u00e8e]\s+rinviata\b|\brinvia\s+la\s+causa\b"
    r")",
    re.IGNORECASE,
)

INADMISSIBLE_RE = re.compile(
    r"(?:\bauf\s+die\s+beschwerde\s+wird\s+nicht\s+eingetreten\b|\birrecevable\b|\binammissibile\b)",
    re.IGNORECASE,
)
DISMISSED_RE = re.compile(
    r"(?:\bbeschwerde\s+wird\s+abgewiesen\b|\ble\s+recours\s+est\s+rejet[\u00e9e]\b|\bil\s+ricorso\s+[\u00e8e]\s+respinto\b)",
    re.IGNORECASE,
)
GRANTED_RE = re.compile(
    r"(?:\bbeschwerde\s+wird\s+gutgeheissen\b|\ble\s+recours\s+est\s+admis\b|\bil\s+ricorso\s+[\u00e8e]\s+accolto\b)",
    re.IGNORECASE,
)
PARTIAL_RE = re.compile(r"(?:\bteilweise\b|\bpartiellement\b|\bparzialmente\b)", re.IGNORECASE)

BGE_REF_RE = re.compile(r"\bBGE\s+\d{3}\s+[IVX]{1,4}\s+\d+[a-z]?\b", re.IGNORECASE)
DOCKET_REF_RE = re.compile(r"\b\d{1,2}[A-Z]{1,4}[_\.]\d{1,5}/\d{4}\b", re.IGNORECASE)
ART_HEAD_RE = re.compile(r"\b(?:art\.?|artt\.?|article|articles)\s+([^\n]{0,140})", re.IGNORECASE)
ART_GENERATED_RE = re.compile(
    r"\b(?:art\.?|article|articles)\s+(\d+[a-z]?(?:\s*-\s*\d+[a-z]?)?)",
    re.IGNORECASE,
)
ART_NUM_RE = re.compile(r"\d+[a-z]?", re.IGNORECASE)


def _txt(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _list(value: Any, max_items: int = 30) -> list[str]:
    if value is None:
        return []
    raw = value if isinstance(value, list) else [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        s = _txt(item)
        key = s.lower()
        if s and key not in seen:
            out.append(s)
            seen.add(key)
        if len(out) >= max_items:
            break
    return out


def _term_values(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    out: list[str] = []
    for concept, terms in value.items():
        out.append(_txt(concept))
        out.extend(_list(terms, max_items=20))
    return _list(out, max_items=60)


def _stub(
    summary: str,
    topic: str,
    concepts: Iterable[str],
    keywords: Iterable[str],
    role: str,
    method: str,
    *,
    outcome: str = "none",
    question: str = "",
    rule: str = "",
    holding: str = "",
    facts: str = "",
    queries: Iterable[str] | None = None,
) -> dict[str, Any]:
    return {
        "english_summary": _txt(summary)[:320],
        "legal_topic": _txt(topic)[:140],
        "legal_question": _txt(question)[:260],
        "legal_rule": _txt(rule)[:320],
        "court_holding": _txt(holding)[:260],
        "factual_context": _txt(facts)[:260],
        "english_legal_concepts": _list(list(concepts), max_items=6),
        "search_keywords": _list(list(keywords), max_items=8),
        "natural_language_queries": _list(list(queries or []), max_items=3),
        "paragraph_role": role,
        "outcome_signal": outcome,
        "method": method,
    }


def deterministic_disposition(card: dict[str, Any]) -> dict[str, Any] | None:
    text = card.get("text_excerpt_original", "") or ""
    head = text[:900]
    if REMITTAL_RE.search(head):
        return _stub(
            "The matter is remitted to a lower authority or previous instance for further proceedings.",
            "remittal to lower authority",
            ["remittal", "procedural disposition"],
            ["remittal", "renvoi", "zurueckgewiesen", "rinvio"],
            role="disposition",
            outcome="remitted",
            method="auto_remittal",
            holding="Matter remitted for further proceedings.",
        )
    if INADMISSIBLE_RE.search(head):
        return _stub(
            "The appeal or application is held inadmissible.",
            "inadmissibility of appeal",
            ["inadmissibility", "appeal procedure"],
            ["inadmissible", "nicht eingetreten", "irrecevable", "inammissibile"],
            role="disposition",
            outcome="inadmissible",
            method="auto_inadmissible",
            holding="Appeal held inadmissible.",
        )
    if DISMISSED_RE.search(head):
        return _stub(
            "The appeal is dismissed.",
            "dismissal of appeal",
            ["dismissal of appeal", "appeal procedure"],
            ["dismissed", "abgewiesen", "rejete", "respinto"],
            role="disposition",
            outcome="dismissed",
            method="auto_dismissed",
            holding="Appeal dismissed.",
        )
    if GRANTED_RE.search(head):
        outcome = "partial" if PARTIAL_RE.search(head) else "granted"
        return _stub(
            "The appeal is granted or partially granted.",
            "appeal granted",
            ["appeal granted", "appeal procedure"],
            ["granted", "gutgeheissen", "admis", "accolto"],
            role="disposition",
            outcome=outcome,
            method="auto_granted",
            holding="Appeal granted or partially granted.",
        )
    return None


def deterministic_fallback(card: dict[str, Any], *, method: str = "deterministic_v4_fallback") -> dict[str, Any]:
    labels = _list(card.get("issue_labels_en"), max_items=6)
    legal_area = _txt(card.get("legal_area"))
    law_codes = _list(card.get("law_codes"), max_items=8)
    statutes = _list(card.get("statutes_cited"), max_items=8)
    cases = _list(card.get("court_cases_cited"), max_items=5)
    terms = _term_values(card.get("matched_terms_multilingual"))
    summary = card.get("summary_en_proxy") or f"Swiss Federal Supreme Court consideration in {legal_area}."
    topic_parts = labels[:3] or ([legal_area] if legal_area else ["Swiss Federal Tribunal authority"])
    keywords = _list(labels + terms + law_codes + statutes + cases, max_items=8)
    queries: list[str] = []
    if labels:
        queries.append("Swiss Federal Tribunal authority on " + ", ".join(labels[:3]))
    if statutes:
        queries.append("Swiss case law applying " + ", ".join(statutes[:2]))
    return _stub(
        _txt(summary),
        " - ".join(topic_parts),
        labels,
        keywords,
        role="procedural" if card.get("is_notification_paragraph") else "reasoning",
        outcome="none",
        method=method,
        rule="; ".join(statutes[:3]),
        queries=queries,
    )


def auto_classify(card: dict[str, Any]) -> dict[str, Any] | None:
    if card.get("is_notification_paragraph"):
        return _stub(
            "Procedural notification of the judgment to the parties.",
            "judgment notification",
            ["service of judgment"],
            ["notification", "service", "judgment communication"],
            role="procedural",
            method="auto_notification",
        )
    text = card.get("text_excerpt_original", "") or ""
    disposition = deterministic_disposition(card)
    if disposition is not None:
        return disposition
    if len(text) < 50:
        return _stub(
            "Short procedural fragment, cross-reference, or one-line ruling.",
            "procedural fragment",
            [],
            [],
            role="procedural",
            method="auto_short",
        )
    if COST_PROC_RE.search(text[:400]):
        return _stub(
            "Court-cost, procedural-fee, or legal-aid paragraph.",
            "court costs and procedural fees",
            ["court costs", "procedural fees", "legal aid"],
            ["costs", "court fees", "frais judiciaires", "Gerichtskosten", "legal aid"],
            role="cost",
            method="auto_cost",
        )
    return None


def validate_and_normalize_enrichment(enriched: Any, *, method: str) -> dict[str, Any]:
    if not isinstance(enriched, dict):
        raise ValueError(f"Expected dict, got {type(enriched).__name__}")
    missing = [k for k in REQUIRED_FIELDS if k not in enriched]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    out = dict(enriched)
    for key, max_len in {
        "english_summary": 320,
        "legal_topic": 140,
        "legal_question": 260,
        "legal_rule": 320,
        "court_holding": 260,
        "factual_context": 260,
    }.items():
        out[key] = _txt(out.get(key, ""))[:max_len].rstrip()

    for key, max_items, item_len in [
        ("english_legal_concepts", 6, 80),
        ("search_keywords", 8, 80),
        ("natural_language_queries", 3, 180),
    ]:
        value = out.get(key, [])
        if not isinstance(value, list):
            value = [value] if value else []
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            s = _txt(item)[:item_len].rstrip()
            low = s.lower()
            if s and low not in seen:
                cleaned.append(s)
                seen.add(low)
            if len(cleaned) >= max_items:
                break
        out[key] = cleaned

    if out.get("paragraph_role") not in ROLE_VALUES:
        out["paragraph_role"] = "reasoning"
    if out.get("outcome_signal") not in OUTCOME_VALUES:
        out["outcome_signal"] = "none"
    out["method"] = method
    return out


def _reference_blob(card: dict[str, Any]) -> str:
    pieces = [
        card.get("citation"),
        card.get("court_base"),
        card.get("text_excerpt_original"),
        card.get("legal_area"),
        card.get("law_codes"),
        card.get("statutes_cited"),
        card.get("court_cases_cited"),
    ]
    return " | ".join(_list(pieces, max_items=200)).lower().replace("_", ".")


def _article_ids_from_blob(blob: str) -> set[str]:
    ids: set[str] = set()
    for match in ART_HEAD_RE.finditer(blob):
        segment = match.group(1)
        nums = ART_NUM_RE.findall(segment)
        if not nums:
            continue
        ids.add(nums[0].lower())
        for chained in re.findall(
            r"(?:,|;|-|\bet\b|\be\b|\bund\b|\band\b)\s*(\d+[a-z]?)",
            segment,
            flags=re.IGNORECASE,
        ):
            ids.add(chained.lower())
    return ids


def _generated_article_ids(payload: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for match in ART_GENERATED_RE.finditer(payload):
        ref = match.group(0)
        ids = ART_NUM_RE.findall(match.group(1))
        for article_id in ids:
            out.append((ref, article_id.lower()))
    return out


def validate_reference_grounding(enriched: dict[str, Any], card: dict[str, Any]) -> None:
    if not CONFIG.validate_reference_grounding:
        return
    blob = _reference_blob(card)
    payload = json.dumps(enriched, ensure_ascii=False).lower().replace("_", ".")
    bad: list[str] = []
    allowed_article_ids = _article_ids_from_blob(blob)

    for art_ref, article_id in _generated_article_ids(payload):
        if article_id in allowed_article_ids:
            continue
        if article_id.endswith("f") and article_id[:-1] in allowed_article_ids:
            compact_source = re.search(
                rf"\bart\.?\s*{re.escape(article_id[:-1])}\s*f\b",
                blob,
                flags=re.IGNORECASE,
            )
            if compact_source:
                continue
        bad.append(art_ref)

    for bge in BGE_REF_RE.findall(payload):
        if bge.lower() not in blob:
            bad.append(bge)
    for docket in DOCKET_REF_RE.findall(payload):
        if docket.lower().replace("_", ".") not in blob:
            bad.append(docket)
    if bad:
        raise ValueError("Ungrounded legal references generated: " + ", ".join(sorted(set(bad))[:8]))


def extract_first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object start found")
    depth = 0
    in_string = False
    escape = False
    for pos in range(start, len(text)):
        ch = text[pos]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : pos + 1]
    raise ValueError("No complete JSON object found")


def clean_model_json(raw: str) -> str:
    raw = (raw or "").strip()
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1].strip()
    if raw.startswith("```json"):
        raw = raw[7:].strip()
    if raw.startswith("```"):
        raw = raw[3:].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()
    return extract_first_json_object(raw)


def parse_output_text(raw_text: str, *, method: str, card: dict[str, Any]) -> dict[str, Any]:
    enriched = json.loads(clean_model_json(raw_text))
    enriched = validate_and_normalize_enrichment(enriched, method=method)
    validate_reference_grounding(enriched, card)
    return enriched


def build_user_prompt(card: dict[str, Any], *, text_chars: int) -> str:
    text = (card.get("text_excerpt_original", "") or "")[:text_chars]
    metadata = {
        "citation": card.get("citation") or "",
        "court_base": card.get("court_base") or "",
        "legal_area": card.get("legal_area") or "",
        "authority_role": _list(card.get("authority_role") or [], max_items=12),
        "existing_labels": _list(card.get("issue_labels_en") or [], max_items=12),
        "law_codes": _list(card.get("law_codes") or [], max_items=12),
        "statutes_cited": _list(card.get("statutes_cited") or [], max_items=20),
        "court_cases_cited": _list(card.get("court_cases_cited") or [], max_items=20),
        "language": card.get("language") or "",
    }
    return "\n".join(
        [
            "<record>",
            "<metadata_json>",
            json.dumps(metadata, ensure_ascii=False),
            "</metadata_json>",
            "<paragraph_original_language>",
            text,
            "</paragraph_original_language>",
            "</record>",
            "",
            "Return exactly one JSON object with this shape:",
            json.dumps(RAG_SCHEMA_HINT, ensure_ascii=False),
            "",
            "Rules:",
            "- Use English legal terminology.",
            "- Use only references present in metadata_json or paragraph_original_language.",
            "- Empty string/array is better than inventing.",
            "- No markdown. No prose outside JSON.",
        ]
    )


def stream_input(path: Path, start_offset: int, stop_before: int | None) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_offset:
                continue
            if stop_before is not None and i >= stop_before:
                break
            if not line.strip():
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError:
                continue


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def read_checkpoint(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(path.read_text(encoding="utf-8").strip() or "0")
    except ValueError:
        return 0


def write_jsonl(handle: Any, obj: dict[str, Any]) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


class QwenGenerator:
    def __init__(self, config: Config):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.torch = torch
        print(f"[model] loading tokenizer: {config.model_id}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float16 if config.torch_dtype == "float16" else "auto"
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": config.device_map,
            "trust_remote_code": config.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if config.attn_implementation:
            kwargs["attn_implementation"] = config.attn_implementation
        if config.use_bnb_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        print(f"[model] loading model: {config.model_id}", flush=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
        except Exception as exc:
            if "AWQ" in config.model_id.upper() and not config.use_bnb_4bit:
                raise RuntimeError(
                    "Failed to load AWQ model. On Kaggle, try: !pip install autoawq "
                    "or set CONFIG.model_id='Qwen/Qwen3-8B' and CONFIG.use_bnb_4bit=True."
                ) from exc
            raise
        self.model.eval()

    def render_prompt(self, card: dict[str, Any], *, extra_guard: str = "") -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(card, text_chars=self.config.text_chars) + extra_guard},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "\n/no_think\n"

    def generate_raw(self, cards: list[dict[str, Any]], *, max_new_tokens: int, extra_guard: str = "") -> list[str]:
        prompts = [self.render_prompt(card, extra_guard=extra_guard) for card in cards]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_input_tokens,
        )
        first_device = next(self.model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.config.do_sample,
            "repetition_penalty": self.config.repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.config.do_sample:
            gen_kwargs["temperature"] = self.config.temperature
            gen_kwargs["top_p"] = self.config.top_p

        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        raws: list[str] = []
        for seq in output_ids:
            new_tokens = seq[prompt_len:]
            raws.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return raws


def write_failed(failed_f: Any, line_idx: int, card: dict[str, Any], raw: str, error: Exception | str, stage: str) -> None:
    err = error if isinstance(error, str) else f"{type(error).__name__}: {str(error)[:700]}"
    write_jsonl(
        failed_f,
        {
            "line_idx": line_idx,
            "citation": card.get("citation"),
            "court_base": card.get("court_base"),
            "stage": stage,
            "error": err,
            "raw_output": (raw or "")[:1600],
            "card": card,
        },
    )


def validate_config(config: Config) -> None:
    if config.batch_size < 1:
        raise ValueError("CONFIG.batch_size must be >= 1")
    if config.limit < 0:
        raise ValueError("CONFIG.limit must be >= 0")
    if config.text_chars < 200:
        raise ValueError("CONFIG.text_chars should be >= 200")
    if config.max_new_tokens < 80:
        raise ValueError("CONFIG.max_new_tokens is probably too small")
    if config.use_bnb_4bit and "AWQ" in config.model_id.upper():
        raise ValueError("Do not use bitsandbytes 4-bit with an AWQ model_id")


def print_kaggle_hint(input_file: Path) -> None:
    if input_file.exists() or not IS_KAGGLE:
        return
    print("[input] target file not found.")
    print("[input] Kaggle input folders available:")
    root = Path("/kaggle/input")
    if root.exists():
        for path in sorted(root.glob("*")):
            print("  ", path)
    print("[input] Set CONFIG.input_file to the exact JSONL path.")


def main() -> int:
    config = CONFIG
    validate_config(config)
    print_kaggle_hint(config.input_file)
    if not config.input_file.exists():
        raise FileNotFoundError(f"Input not found: {config.input_file}")

    for path in [config.output_file, config.failed_file, config.checkpoint_file]:
        path.parent.mkdir(parents=True, exist_ok=True)
    if config.reset_output:
        for path in [config.output_file, config.failed_file, config.checkpoint_file]:
            path.unlink(missing_ok=True)

    start = read_checkpoint(config.checkpoint_file)
    total_lines = count_lines(config.input_file)
    if start >= total_lines and total_lines > 0:
        print(f"[checkpoint] stale checkpoint {start:,} >= input lines {total_lines:,}; resetting to 0")
        start = 0
        config.checkpoint_file.unlink(missing_ok=True)
    stop_before = min(total_lines, start + config.limit) if config.limit else total_lines

    print("Input      :", config.input_file)
    print("Output     :", config.output_file)
    print("Failures   :", config.failed_file)
    print("Checkpoint :", config.checkpoint_file)
    print("Model      :", config.model_id)
    print("Start      :", f"{start:,}")
    print("Total      :", f"{total_lines:,}")
    print("To process :", f"{max(stop_before - start, 0):,}")
    print("Batch size :", config.batch_size)

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    generator = QwenGenerator(config)

    out_f = config.output_file.open("a", encoding="utf-8")
    failed_f = config.failed_file.open("a", encoding="utf-8")
    pbar = tqdm(total=stop_before, initial=start, desc="qwen3-8b-enrich", unit="card") if tqdm else None

    pending: list[tuple[int, dict[str, Any]]] = []
    batch_count = 0
    llm_ok = 0
    auto_count = 0
    fail_count = 0
    fallback_count = 0
    started_at = time.time()

    def write_success(line_idx: int, card: dict[str, Any]) -> None:
        write_jsonl(out_f, card)
        config.checkpoint_file.write_text(str(line_idx + 1), encoding="utf-8")
        if pbar:
            pbar.update(1)

    def flush_batch() -> None:
        nonlocal pending, batch_count, llm_ok, fail_count, fallback_count
        if not pending:
            return
        batch_count += 1
        batch_items = pending
        pending = []
        line_indices = [idx for idx, _ in batch_items]
        cards = [card for _, card in batch_items]
        t0 = time.time()
        raws = generator.generate_raw(cards, max_new_tokens=config.max_new_tokens)
        dt = time.time() - t0

        for (line_idx, card), raw in zip(batch_items, raws):
            try:
                enriched = parse_output_text(raw, method="qwen3_8b_awq_transformers", card=card)
                card["rag_enrichment"] = enriched
                write_success(line_idx, card)
                llm_ok += 1
                continue
            except Exception as first_error:
                if config.retry_bad_outputs:
                    guard = (
                        "\n\nPrevious output was invalid, incomplete, or ungrounded. "
                        "Return one complete JSON object only. Use empty fields when unsure. "
                        "Use no article number or case citation unless it appears in the metadata or paragraph."
                    )
                    retry_raw = generator.generate_raw(
                        [card],
                        max_new_tokens=config.retry_max_new_tokens,
                        extra_guard=guard,
                    )[0]
                    try:
                        enriched = parse_output_text(retry_raw, method="qwen3_8b_awq_transformers_retry", card=card)
                        card["rag_enrichment"] = enriched
                        write_success(line_idx, card)
                        llm_ok += 1
                        continue
                    except Exception as retry_error:
                        raw = retry_raw
                        first_error = retry_error

                if config.fallback_on_fail:
                    card["rag_enrichment"] = deterministic_fallback(
                        card,
                        method="deterministic_fallback_after_qwen3_8b_failed",
                    )
                    write_failed(failed_f, line_idx, card, raw, first_error, "qwen_failed_fallback_written")
                    write_success(line_idx, card)
                    fallback_count += 1
                else:
                    write_failed(failed_f, line_idx, card, raw, first_error, "qwen_failed")
                    config.checkpoint_file.write_text(str(line_idx + 1), encoding="utf-8")
                    if pbar:
                        pbar.update(1)
                    fail_count += 1

        out_f.flush()
        failed_f.flush()
        if batch_count == 1 or batch_count % config.progress_every_batches == 0:
            elapsed = time.time() - started_at
            done = max((pbar.n if pbar else line_indices[-1] + 1) - start, 0)
            rate = done / max(elapsed, 1e-9)
            remaining = max(stop_before - (pbar.n if pbar else line_indices[-1] + 1), 0)
            eta_min = remaining / max(rate, 1e-9) / 60.0
            batch_rate = len(batch_items) / max(dt, 1e-9)
            print(
                f"[batch {batch_count}] size={len(batch_items)} "
                f"batch_rate={batch_rate:.2f} cards/s avg_rate={rate:.2f} cards/s "
                f"eta={eta_min:.1f} min llm_ok={llm_ok} auto={auto_count} "
                f"failed={fail_count} fallback={fallback_count}",
                flush=True,
            )

    try:
        for line_idx, card in stream_input(config.input_file, start, stop_before):
            auto = auto_classify(card) if config.auto_classify else None
            if auto is not None:
                flush_batch()
                card["rag_enrichment"] = validate_and_normalize_enrichment(auto, method=auto.get("method", "auto"))
                write_success(line_idx, card)
                auto_count += 1
                continue

            pending.append((line_idx, card))
            if len(pending) >= config.batch_size:
                flush_batch()
        flush_batch()
    finally:
        out_f.close()
        failed_f.close()
        if pbar:
            pbar.close()

    elapsed = time.time() - started_at
    processed = llm_ok + auto_count + fail_count + fallback_count
    print()
    print("Done processed :", f"{processed:,}")
    print("LLM ok         :", f"{llm_ok:,}")
    print("Auto           :", f"{auto_count:,}")
    print("Failed captured:", f"{fail_count:,}")
    print("Fallback written:", f"{fallback_count:,}")
    print("Elapsed min    :", f"{elapsed / 60:.1f}")
    print("Rate cards/s   :", f"{processed / max(elapsed, 1e-9):.2f}")
    print("Output         :", config.output_file)
    print("Failures       :", config.failed_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
