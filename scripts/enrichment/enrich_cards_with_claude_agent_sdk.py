#!/usr/bin/env python
"""Enrich Swiss court authority target cards with Claude Agent SDK.

This is the Claude equivalent of the targeted Qwen/vLLM notebook:

  - input is the compact target-card JSONL, not the full 2.47M-card corpus
  - Claude returns a schema-validated ``rag_enrichment`` object
  - deterministic metadata is included in the prompt
  - generated statute/article/case references are locally grounded afterward
  - output, failures, and checkpoint files are isolated for the Claude run

Install:

    pip install claude-agent-sdk tqdm

Edit CONFIG below, then run:

    python scripts/enrich_cards_with_claude_agent_sdk.py
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # Paths. Relative paths are resolved under base_dir.
    base_dir: Path = Path("/content/drive/MyDrive/swiss_law") if Path("/content").exists() else ROOT
    input: Path = Path("artifacts/court_authority_cards_v4_target_cards.jsonl")
    output: Path = Path("artifacts/court_authority_cards_rag_targets_claude.jsonl")
    failed: Path = Path("artifacts/court_authority_cards_rag_targets_claude_failed_cards.jsonl")
    checkpoint: Path = Path("artifacts/rag_targets_claude_checkpoint.txt")

    # Claude. Use None for Claude Code default.
    model: str | None = "sonnet"
    max_turns: int = 3
    max_budget_usd_per_card: float = 0.0

    # Run control. Keep limit small for smoke tests; set 0 for the full 363,258-card run.
    limit: int = 25
    concurrency: int = 2
    text_chars: int = 900
    reset_output: bool = True

    # Reliability.
    retries: int = 2
    retry_base_delay: float = 2.0
    retry_max_delay: float = 30.0
    auto_classify: bool = True
    fallback_on_fail: bool = False
    allow_failures: bool = True
    store_usage: bool = False


CONFIG = Config()


RAG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "english_summary": {"type": "string", "maxLength": 320},
        "legal_topic": {"type": "string", "maxLength": 140},
        "legal_question": {"type": "string", "maxLength": 260},
        "legal_rule": {"type": "string", "maxLength": 320},
        "court_holding": {"type": "string", "maxLength": 260},
        "factual_context": {"type": "string", "maxLength": 260},
        "english_legal_concepts": {
            "type": "array",
            "items": {"type": "string", "maxLength": 80},
            "minItems": 0,
            "maxItems": 6,
        },
        "search_keywords": {
            "type": "array",
            "items": {"type": "string", "maxLength": 80},
            "minItems": 0,
            "maxItems": 8,
        },
        "natural_language_queries": {
            "type": "array",
            "items": {"type": "string", "maxLength": 180},
            "minItems": 0,
            "maxItems": 3,
        },
        "paragraph_role": {
            "type": "string",
            "enum": [
                "holding",
                "reasoning",
                "background",
                "cost",
                "procedural",
                "disposition",
                "standard_of_review",
                "obiter",
            ],
        },
        "outcome_signal": {
            "type": "string",
            "enum": ["granted", "dismissed", "inadmissible", "remitted", "partial", "none"],
        },
    },
    "required": [
        "english_summary",
        "legal_topic",
        "legal_question",
        "legal_rule",
        "court_holding",
        "factual_context",
        "english_legal_concepts",
        "search_keywords",
        "natural_language_queries",
        "paragraph_role",
        "outcome_signal",
    ],
}


SYSTEM_PROMPT = (
    "You are a deterministic Swiss legal JSON extraction engine. "
    "Input is one paragraph from a Swiss Federal Tribunal decision in German, "
    "French, or Italian, plus deterministic metadata. Return exactly one JSON "
    "object matching the provided schema. Use concise English legal terminology "
    "for semantic-search RAG. Use only statutes, articles, and case citations "
    "present in the paragraph or metadata. Do not invent article numbers, "
    "statutes, laws, or case citations. If no legal rule, holding, facts, "
    "statute, or citation is supported by the paragraph or metadata, use an "
    "empty string, empty array, or \"none\". Do not repeat the source text. "
    "Do not produce bibliography, markdown, tables, or chain-of-thought."
)


COST_PROC_RE = re.compile(
    r"(?:"
    r"\bgerichtskosten\b|\bprozesskosten\b|\bverfahrenskosten\b|"
    r"\bfrais judiciaires\b|\bfrais de la cause\b|\bd[eé]pens\b|"
    r"\bspese giudiziarie\b|\bripetibili\b|"
    r"\bparteientsch[aä]digung\b|\bhonoraire\b|"
    r"\bunentgeltliche rechtspflege\b|\bassistance judiciaire\b|"
    r"\bpatrocinio gratuito\b|"
    r"^\s*\d+\.\s*\d+\..{0,5}fr\.\s*\d"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

REMITTAL_RE = re.compile(
    r"(?:"
    r"\b(?:die\s+)?sache\s+wird.{0,140}\bzur(?:ue|ü|u)ckgewiesen\b|"
    r"\bzur\s+(?:neuen|erneuten)\s+(?:entscheidung|beurteilung|neubeurteilung).{0,120}\bzur(?:ue|ü|u)ckgewiesen\b|"
    r"\ban\s+die\s+(?:vorinstanz|beschwerdegegnerin|verwaltung|beh[oö]rde).{0,140}\bzur(?:ue|ü|u)ckgewiesen\b|"
    r"\brenvoie\s+la\s+cause\b|\bla\s+cause\s+est\s+renvoy[ée]e\b|"
    r"\brenvoy[ée]\s+.{0,80}\b(?:l'autorit[ée]|tribunal|instance)\b|"
    r"\bla\s+causa\s+[èe]\s+rinviata\b|\brinvia\s+la\s+causa\b"
    r")",
    re.IGNORECASE,
)

INADMISSIBLE_RE = re.compile(
    r"(?:\bauf\s+die\s+beschwerde\s+wird\s+nicht\s+eingetreten\b|\birrecevable\b|\binammissibile\b)",
    re.IGNORECASE,
)
DISMISSED_RE = re.compile(
    r"(?:\bbeschwerde\s+wird\s+abgewiesen\b|\ble\s+recours\s+est\s+rejet[ée]\b|\bil\s+ricorso\s+[èe]\s+respinto\b)",
    re.IGNORECASE,
)
GRANTED_RE = re.compile(
    r"(?:\bbeschwerde\s+wird\s+gutgeheissen\b|\ble\s+recours\s+est\s+admis\b|\bil\s+ricorso\s+[èe]\s+accolto\b)",
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


def _join(value: Any, max_items: int = 20) -> str:
    return "; ".join(_list(value, max_items=max_items))


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
    missing = [k for k in RAG_SCHEMA["required"] if k not in enriched]
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
            key_l = s.lower()
            if s and key_l not in seen:
                cleaned.append(s)
                seen.add(key_l)
            if len(cleaned) >= max_items:
                break
        out[key] = cleaned

    role_values = RAG_SCHEMA["properties"]["paragraph_role"]["enum"]
    outcome_values = RAG_SCHEMA["properties"]["outcome_signal"]["enum"]
    if out.get("paragraph_role") not in role_values:
        out["paragraph_role"] = "reasoning"
    if out.get("outcome_signal") not in outcome_values:
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
            "Create concise English RAG metadata for this paragraph.",
            "Use only references present inside metadata_json or paragraph_original_language.",
        ]
    )


def resolve_path(base_dir: Path, value: Path) -> Path:
    return value if value.is_absolute() else base_dir / value


def count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


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


def read_checkpoint(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(path.read_text(encoding="utf-8").strip() or "0")
    except ValueError:
        return 0


@dataclass
class WorkResult:
    line_idx: int
    card: dict[str, Any]
    ok: bool
    error: str = ""
    result_subtype: str = ""
    cost_usd: float | None = None
    usage: dict[str, Any] | None = None


async def claude_enrich_one(
    card: dict[str, Any],
    *,
    args: Config,
    query: Any,
    options_factory: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            result_message = None
            async for message in query(
                prompt=build_user_prompt(card, text_chars=args.text_chars),
                options=options_factory(),
            ):
                if getattr(message, "type", None) == "result" or message.__class__.__name__ == "ResultMessage":
                    result_message = message
            if result_message is None:
                raise RuntimeError("Claude Agent SDK returned no result message")

            subtype = getattr(result_message, "subtype", "")
            structured_output = getattr(result_message, "structured_output", None)
            if subtype != "success" or structured_output is None:
                raise RuntimeError(f"Claude structured output failed: subtype={subtype}")

            enriched = validate_and_normalize_enrichment(
                structured_output,
                method="claude_agent_sdk_structured_output",
            )
            validate_reference_grounding(enriched, card)
            usage_meta = {
                "result_subtype": subtype,
                "total_cost_usd": getattr(result_message, "total_cost_usd", None),
                "usage": getattr(result_message, "usage", None),
                "model_usage": getattr(result_message, "model_usage", None),
                "num_turns": getattr(result_message, "num_turns", None),
                "session_id": getattr(result_message, "session_id", None),
            }
            return enriched, usage_meta
        except Exception as exc:
            last_error = exc
            if attempt < args.retries:
                delay = min(args.retry_max_delay, args.retry_base_delay * (2**attempt))
                await asyncio.sleep(delay)
    raise RuntimeError(str(last_error)) from last_error


async def process_card(
    line_idx: int,
    card: dict[str, Any],
    *,
    args: Config,
    query: Any,
    options_factory: Any,
) -> WorkResult:
    try:
        auto = auto_classify(card) if args.auto_classify else None
        if auto is not None:
            card["rag_enrichment"] = validate_and_normalize_enrichment(auto, method=auto.get("method", "auto"))
            return WorkResult(line_idx=line_idx, card=card, ok=True)

        enriched, usage_meta = await claude_enrich_one(
            card,
            args=args,
            query=query,
            options_factory=options_factory,
        )
        card["rag_enrichment"] = enriched
        if args.store_usage:
            card["_claude_agent_sdk"] = usage_meta
        return WorkResult(
            line_idx=line_idx,
            card=card,
            ok=True,
            result_subtype=usage_meta.get("result_subtype") or "",
            cost_usd=usage_meta.get("total_cost_usd"),
            usage=usage_meta.get("usage"),
        )
    except Exception as exc:
        if args.fallback_on_fail:
            card["rag_enrichment"] = deterministic_fallback(
                card,
                method="deterministic_fallback_after_claude_failed",
            )
            return WorkResult(line_idx=line_idx, card=card, ok=True, error=str(exc))
        return WorkResult(line_idx=line_idx, card=card, ok=False, error=f"{type(exc).__name__}: {str(exc)[:1000]}")


def make_options_factory(args: Config, ClaudeAgentOptions: Any) -> Any:
    def options_factory() -> Any:
        kwargs: dict[str, Any] = {
            "system_prompt": SYSTEM_PROMPT,
            "output_format": {"type": "json_schema", "schema": RAG_SCHEMA},
            "max_turns": args.max_turns,
            "tools": [],
            "allowed_tools": [],
            "disallowed_tools": [
                "Agent",
                "Bash",
                "Edit",
                "Glob",
                "Grep",
                "LS",
                "MultiEdit",
                "NotebookEdit",
                "Read",
                "Task",
                "TodoWrite",
                "WebFetch",
                "WebSearch",
                "Write",
            ],
            "permission_mode": "dontAsk",
            "setting_sources": [],
            "cwd": str(args.base_dir),
        }
        if args.model:
            kwargs["model"] = args.model
        if args.max_budget_usd_per_card > 0:
            kwargs["max_budget_usd"] = args.max_budget_usd_per_card
        return ClaudeAgentOptions(**kwargs)

    return options_factory


def write_jsonl(handle: Any, obj: dict[str, Any]) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


async def run_async(args: Config) -> int:
    try:
        from claude_agent_sdk import ClaudeAgentOptions, query
    except ImportError as exc:
        raise SystemExit(
            "claude-agent-sdk is not installed. Install with: pip install claude-agent-sdk"
        ) from exc

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None

    args.base_dir = args.base_dir.resolve()
    in_path = resolve_path(args.base_dir, args.input)
    out_path = resolve_path(args.base_dir, args.output)
    failed_path = resolve_path(args.base_dir, args.failed)
    checkpoint_path = resolve_path(args.base_dir, args.checkpoint)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")
    for path in [out_path, failed_path, checkpoint_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    if args.reset_output:
        for path in [out_path, failed_path, checkpoint_path]:
            path.unlink(missing_ok=True)

    start = read_checkpoint(checkpoint_path)
    total_lines = count_lines(in_path)
    if start >= total_lines and total_lines > 0:
        print(
            f"[checkpoint] stale checkpoint {start:,} >= input lines {total_lines:,}; resetting to 0 for {in_path.name}",
            flush=True,
        )
        start = 0
        checkpoint_path.unlink(missing_ok=True)
    stop_before = min(total_lines, start + args.limit) if args.limit else total_lines

    print(f"Input      : {in_path}")
    print(f"Output     : {out_path}")
    print(f"Failures   : {failed_path}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Model      : {args.model or '(Claude Code default)'}")
    print(f"Start      : {start:,}")
    print(f"Total      : {total_lines:,}")
    print(f"To process : {max(stop_before - start, 0):,}")
    print(f"Concurrency: {args.concurrency}")

    options_factory = make_options_factory(args, ClaudeAgentOptions)
    completed: set[int] = set()
    next_checkpoint = start
    ok_count = 0
    fail_count = 0
    auto_count = 0
    claude_count = 0
    started_at = time.time()

    pbar = tqdm(total=stop_before, initial=start, desc="claude-enrich", unit="card") if tqdm else None

    async def handle_result(result: WorkResult, out_f: Any, failed_f: Any) -> None:
        nonlocal next_checkpoint, ok_count, fail_count, auto_count, claude_count

        if result.ok:
            method = (result.card.get("rag_enrichment") or {}).get("method", "")
            if method.startswith("auto_"):
                auto_count += 1
            elif method == "claude_agent_sdk_structured_output":
                claude_count += 1
            write_jsonl(out_f, result.card)
            ok_count += 1
            if result.error:
                write_jsonl(
                    failed_f,
                    {
                        "line_idx": result.line_idx,
                        "citation": result.card.get("citation"),
                        "stage": "claude_failed_fallback_written",
                        "error": result.error,
                    },
                )
        else:
            write_jsonl(
                failed_f,
                {
                    "line_idx": result.line_idx,
                    "citation": result.card.get("citation"),
                    "court_base": result.card.get("court_base"),
                    "stage": "claude_failed",
                    "error": result.error,
                    "card": result.card,
                },
            )
            fail_count += 1

        completed.add(result.line_idx)
        while next_checkpoint in completed:
            completed.remove(next_checkpoint)
            next_checkpoint += 1
        checkpoint_path.write_text(str(next_checkpoint), encoding="utf-8")
        out_f.flush()
        failed_f.flush()
        if pbar:
            pbar.update(1)

    pending: set[asyncio.Task[WorkResult]] = set()
    out_f = out_path.open("a", encoding="utf-8")
    failed_f = failed_path.open("a", encoding="utf-8")
    try:
        for line_idx, card in stream_input(in_path, start, stop_before):
            task = asyncio.create_task(
                process_card(
                    line_idx,
                    card,
                    args=args,
                    query=query,
                    options_factory=options_factory,
                )
            )
            pending.add(task)
            if len(pending) >= args.concurrency:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for done_task in done:
                    await handle_result(done_task.result(), out_f, failed_f)

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for done_task in done:
                await handle_result(done_task.result(), out_f, failed_f)
    finally:
        out_f.close()
        failed_f.close()
        if pbar:
            pbar.close()

    elapsed = time.time() - started_at
    done_total = ok_count + fail_count
    rate = done_total / elapsed if elapsed else 0.0
    print()
    print(f"Done processed      : {done_total:,}")
    print(f"Claude enriched     : {claude_count:,}")
    print(f"Auto-classified     : {auto_count:,}")
    print(f"Failed captured     : {fail_count:,}")
    print(f"Checkpoint line     : {next_checkpoint:,}")
    print(f"Elapsed             : {elapsed/60:.1f} min")
    print(f"Rate                : {rate:.2f} cards/s")
    print(f"Output -> {out_path}")
    print(f"Failures -> {failed_path}")
    return 0 if fail_count == 0 or args.allow_failures else 2


def validate_config(config: Config) -> None:
    if config.concurrency < 1:
        raise ValueError("CONFIG.concurrency must be >= 1")
    if config.limit < 0:
        raise ValueError("CONFIG.limit must be >= 0")
    if config.text_chars < 200:
        raise ValueError("CONFIG.text_chars should be at least 200")
    if config.max_turns < 1:
        raise ValueError("CONFIG.max_turns must be >= 1")
    if config.retries < 0:
        raise ValueError("CONFIG.retries must be >= 0")


def main() -> int:
    validate_config(CONFIG)
    return asyncio.run(run_async(CONFIG))


if __name__ == "__main__":
    raise SystemExit(main())
