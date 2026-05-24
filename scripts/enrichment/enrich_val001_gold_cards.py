#!/usr/bin/env python
"""Enrich the 23 val_001 gold court citations with rag_enrichment via Claude API.

Proof-of-concept enrichment for recall comparison:
  1. Parse val_001 gold court citations from val.csv
  2. Scan court_authority_cards_v4_target_cards.jsonl for those cards
     (falls back to full 2.4M corpus for any not found there)
  3. Call Claude Haiku to generate rag_enrichment for each card
  4. Write enriched cards to artifacts/val001_gold_court_enriched.jsonl

Then run experiment_court_recall_enriched_1m.py to compare recall vs baseline.
Baseline (v4-only at 1M noise): 13/23 = 0.565
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

# claude_agent_sdk ships with Claude Code — uses its auth, no ANTHROPIC_API_KEY needed.
try:
    from claude_agent_sdk import ClaudeAgentOptions
    from claude_agent_sdk import query as agent_query
    USE_AGENT_SDK = True
except ImportError:
    USE_AGENT_SDK = False

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

VAL_CSV = ROOT / "data" / "val.csv"
COURT_TARGET_JSONL = ROOT / "artifacts" / "court_authority_cards_v4_target_cards.jsonl"
COURT_V4_JSONL = ROOT / "artifacts" / "court_authority_cards_v4.jsonl"
OUTPUT_JSONL = ROOT / "artifacts" / "val001_gold_court_enriched.jsonl"

QUERY_ID = "val_001"
MODEL = "claude-haiku-4-5-20251001"
TEXT_CHARS = 900

BGE_RE = re.compile(r"\bBGE\s+\d{3}\s+[IVX]{1,4}\s+\d+[a-z]?\b", re.IGNORECASE)
DOCKET_RE = re.compile(r"\b\d{1,2}[A-Z]{1,4}[_\.]\d{1,5}/\d{4}\b", re.IGNORECASE)

COST_PROC_RE = re.compile(
    r"(?:"
    r"\bgerichtskosten\b|\bprozesskosten\b|\bverfahrenskosten\b|"
    r"\bfrais judiciaires\b|\bfrais de la cause\b|\bd[eé]pens\b|"
    r"\bspese giudiziarie\b|\bripetibili\b|"
    r"\bparteientsch[aä]digung\b|\bhonoraire\b|"
    r"\bunentgeltliche rechtspflege\b|\bassistance judiciaire\b|"
    r"\bpatrocinio gratuito\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

RAG_REQUIRED = [
    "english_summary", "legal_topic", "legal_question", "legal_rule",
    "court_holding", "factual_context", "english_legal_concepts",
    "search_keywords", "natural_language_queries", "paragraph_role", "outcome_signal",
]
ROLE_VALUES = {
    "holding", "reasoning", "background", "cost",
    "procedural", "disposition", "standard_of_review", "obiter",
}
OUTCOME_VALUES = {"granted", "dismissed", "inadmissible", "remitted", "partial", "none"}

SYSTEM_PROMPT = (
    "You are a deterministic Swiss legal JSON extraction engine. "
    "Input is one paragraph from a Swiss Federal Tribunal decision in German, "
    "French, or Italian, plus deterministic metadata. "
    "Return exactly one JSON object with these fields:\n"
    "  english_summary    (string, max 320 chars)\n"
    "  legal_topic        (string, max 140 chars)\n"
    "  legal_question     (string, max 260 chars)\n"
    "  legal_rule         (string, max 320 chars)\n"
    "  court_holding      (string, max 260 chars)\n"
    "  factual_context    (string, max 260 chars)\n"
    "  english_legal_concepts  (array of strings, max 6 items, each max 80 chars)\n"
    "  search_keywords    (array of strings, max 8 items, each max 80 chars)\n"
    "  natural_language_queries (array of strings, max 3 items, each max 180 chars)\n"
    "  paragraph_role     (one of: holding, reasoning, background, cost, "
    "procedural, disposition, standard_of_review, obiter)\n"
    "  outcome_signal     (one of: granted, dismissed, inadmissible, remitted, partial, none)\n\n"
    "Use concise English legal terminology for semantic-search RAG. "
    "Use only statutes, articles, and case citations present in the paragraph or metadata. "
    "Do not invent references. Do not produce markdown, tables, or chain-of-thought. "
    "Return ONLY the JSON object."
)


def _txt(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _list(value: Any, max_items: int = 20) -> list[str]:
    if value is None:
        return []
    raw = value if isinstance(value, list) else [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        s = _txt(item)
        k = s.lower()
        if s and k not in seen:
            out.append(s)
            seen.add(k)
        if len(out) >= max_items:
            break
    return out


def citation_family(citation: str) -> str:
    if citation.startswith("Art."):
        return "law"
    if BGE_RE.search(citation) or DOCKET_RE.search(citation):
        return "court"
    return "other"


def read_val_query(path: Path, query_id: str) -> tuple[str, list[str]]:
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["query_id"] == query_id:
                gold = [x.strip() for x in row["gold_citations"].split(";") if x.strip()]
                return row["query"], gold
    raise KeyError(f"query_id not found: {query_id}")


def stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def find_gold_cards(
    gold_citations: set[str], jsonl_path: Path
) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    remaining = set(gold_citations)
    print(
        f"[find] Scanning {jsonl_path.name} for {len(remaining)} citations...",
        flush=True,
    )
    for i, card in enumerate(stream_jsonl(jsonl_path), start=1):
        citation = _txt(card.get("citation"))
        if citation in remaining:
            found[citation] = card
            remaining.remove(citation)
            print(f"  [{len(found)}/{len(gold_citations)}] {citation}", flush=True)
            if not remaining:
                print(f"  [done] All found at line {i:,}", flush=True)
                break
        if i % 500_000 == 0:
            print(
                f"  [scan] {i:,} lines; found {len(found)}/{len(gold_citations)}",
                flush=True,
            )
    if remaining:
        print(f"  [warn] Not found: {sorted(remaining)}", flush=True)
    return found


def auto_classify(card: dict[str, Any]) -> dict[str, Any] | None:
    if card.get("is_notification_paragraph"):
        return {
            "english_summary": "Procedural notification of the judgment to the parties.",
            "legal_topic": "judgment notification",
            "legal_question": "", "legal_rule": "", "court_holding": "",
            "factual_context": "", "english_legal_concepts": ["service of judgment"],
            "search_keywords": ["notification", "service"],
            "natural_language_queries": [], "paragraph_role": "procedural",
            "outcome_signal": "none", "method": "auto_notification",
        }
    text = card.get("text_excerpt_original", "") or ""
    if len(text) < 50:
        return {
            "english_summary": "Short procedural fragment.",
            "legal_topic": "procedural fragment",
            "legal_question": "", "legal_rule": "", "court_holding": "",
            "factual_context": "", "english_legal_concepts": [],
            "search_keywords": [], "natural_language_queries": [],
            "paragraph_role": "procedural", "outcome_signal": "none",
            "method": "auto_short",
        }
    if COST_PROC_RE.search(text[:400]):
        return {
            "english_summary": "Court cost or procedural fee paragraph.",
            "legal_topic": "court costs and procedural fees",
            "legal_question": "", "legal_rule": "", "court_holding": "",
            "factual_context": "",
            "english_legal_concepts": ["court costs", "procedural fees", "legal aid"],
            "search_keywords": ["costs", "Gerichtskosten", "frais judiciaires"],
            "natural_language_queries": [], "paragraph_role": "cost",
            "outcome_signal": "none", "method": "auto_cost",
        }
    return None


def build_user_prompt(card: dict[str, Any]) -> str:
    text = (card.get("text_excerpt_original", "") or "")[:TEXT_CHARS]
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
    return "\n".join([
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
        "Use only references present in metadata_json or paragraph_original_language.",
    ])


def validate_and_normalize(enriched: Any, *, method: str) -> dict[str, Any]:
    if not isinstance(enriched, dict):
        raise ValueError(f"Expected dict, got {type(enriched).__name__}")
    missing = [k for k in RAG_REQUIRED if k not in enriched]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    out = dict(enriched)
    for key, max_len in [
        ("english_summary", 320), ("legal_topic", 140), ("legal_question", 260),
        ("legal_rule", 320), ("court_holding", 260), ("factual_context", 260),
    ]:
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
            k = s.lower()
            if s and k not in seen:
                cleaned.append(s)
                seen.add(k)
            if len(cleaned) >= max_items:
                break
        out[key] = cleaned

    if out.get("paragraph_role") not in ROLE_VALUES:
        out["paragraph_role"] = "reasoning"
    if out.get("outcome_signal") not in OUTCOME_VALUES:
        out["outcome_signal"] = "none"
    out["method"] = method
    return out


def deterministic_fallback(card: dict[str, Any]) -> dict[str, Any]:
    labels = _list(card.get("issue_labels_en"), max_items=6)
    legal_area = _txt(card.get("legal_area"))
    law_codes = _list(card.get("law_codes"), max_items=8)
    statutes = _list(card.get("statutes_cited"), max_items=8)
    terms_map = card.get("matched_terms_multilingual")
    extra_terms: list[str] = []
    if isinstance(terms_map, dict):
        for concept, tlist in terms_map.items():
            extra_terms.append(_txt(concept))
            extra_terms.extend(_list(tlist, max_items=10))
    summary = card.get("summary_en_proxy") or f"Swiss Federal Supreme Court consideration in {legal_area}."
    topic_parts = labels[:3] or ([legal_area] if legal_area else ["Swiss Federal Tribunal authority"])
    keywords = _list(labels + extra_terms + law_codes + statutes, max_items=8)
    queries: list[str] = []
    if labels:
        queries.append("Swiss Federal Tribunal authority on " + ", ".join(labels[:3]))
    if statutes:
        queries.append("Swiss case law applying " + ", ".join(statutes[:2]))
    return {
        "english_summary": _txt(summary)[:320],
        "legal_topic": " - ".join(topic_parts)[:140],
        "legal_question": "",
        "legal_rule": "; ".join(statutes[:3])[:320],
        "court_holding": "", "factual_context": "",
        "english_legal_concepts": labels,
        "search_keywords": keywords,
        "natural_language_queries": queries[:3],
        "paragraph_role": "reasoning", "outcome_signal": "none",
        "method": "deterministic_fallback",
    }


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
            "type": "array", "items": {"type": "string", "maxLength": 80},
            "minItems": 0, "maxItems": 6,
        },
        "search_keywords": {
            "type": "array", "items": {"type": "string", "maxLength": 80},
            "minItems": 0, "maxItems": 8,
        },
        "natural_language_queries": {
            "type": "array", "items": {"type": "string", "maxLength": 180},
            "minItems": 0, "maxItems": 3,
        },
        "paragraph_role": {
            "type": "string",
            "enum": ["holding", "reasoning", "background", "cost", "procedural",
                     "disposition", "standard_of_review", "obiter"],
        },
        "outcome_signal": {
            "type": "string",
            "enum": ["granted", "dismissed", "inadmissible", "remitted", "partial", "none"],
        },
    },
    "required": RAG_REQUIRED,
}


async def enrich_one_async(card: dict[str, Any], options_factory: Any) -> dict[str, Any]:
    auto = auto_classify(card)
    if auto is not None:
        return validate_and_normalize(auto, method=auto.get("method", "auto"))

    user_prompt = build_user_prompt(card)
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            result_message = None
            async for message in agent_query(prompt=user_prompt, options=options_factory()):
                if (
                    getattr(message, "type", None) == "result"
                    or message.__class__.__name__ == "ResultMessage"
                ):
                    result_message = message

            if result_message is None:
                raise RuntimeError("claude_agent_sdk returned no result message")

            subtype = getattr(result_message, "subtype", "")
            structured = getattr(result_message, "structured_output", None)
            if subtype != "success" or structured is None:
                raise RuntimeError(f"structured output failed: subtype={subtype!r}")

            return validate_and_normalize(structured, method="claude_agent_sdk")

        except Exception as exc:
            last_exc = exc
            print(f"    [attempt {attempt + 1}/3] {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)

    print(f"    [fallback] deterministic stub after 3 failures", flush=True)
    return deterministic_fallback(card)


def make_options_factory() -> Any:
    def factory() -> Any:
        return ClaudeAgentOptions(
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
            output_format={"type": "json_schema", "schema": RAG_SCHEMA},
            max_turns=3,
            tools=[],
            allowed_tools=[],
            disallowed_tools=[
                "Agent", "Bash", "Edit", "Glob", "Grep", "LS", "MultiEdit",
                "NotebookEdit", "Read", "Task", "TodoWrite", "WebFetch",
                "WebSearch", "Write",
            ],
            permission_mode="dontAsk",
            setting_sources=[],
        )
    return factory


async def main_async() -> int:
    _, gold = read_val_query(VAL_CSV, QUERY_ID)
    gold_court = {g for g in gold if citation_family(g) == "court"}
    print(f"[val_001] {len(gold_court)} court gold citations:")
    for c in sorted(gold_court):
        print(f"  {c}")

    # Try the smaller target_cards file first (363k rows, ~10s scan)
    found = find_gold_cards(gold_court, COURT_TARGET_JSONL)

    # Fall back to full 2.4M corpus for any not found
    missing = gold_court - set(found.keys())
    if missing:
        print(f"\n[fallback] {len(missing)} not in target_cards, scanning full corpus...", flush=True)
        extra = find_gold_cards(missing, COURT_V4_JSONL)
        found.update(extra)

    still_missing = gold_court - set(found.keys())
    if still_missing:
        print(f"[warn] Still missing: {sorted(still_missing)}")

    if not USE_AGENT_SDK:
        print("[warn] claude_agent_sdk not available — writing deterministic stubs only")

    print(f"\n[enrich] Found {len(found)}/{len(gold_court)} cards. Starting enrichment...")
    options_factory = make_options_factory() if USE_AGENT_SDK else None
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    enriched_count = 0
    fallback_count = 0
    auto_count = 0

    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for i, (citation, card) in enumerate(found.items(), start=1):
            t = time.time()
            print(f"\n[{i}/{len(found)}] {citation}", flush=True)
            if USE_AGENT_SDK:
                rag = await enrich_one_async(card, options_factory)
            else:
                rag = deterministic_fallback(card)

            card = dict(card)
            card["rag_enrichment"] = rag
            f.write(json.dumps(card, ensure_ascii=False) + "\n")

            method = rag.get("method", "?")
            elapsed = time.time() - t
            if method.startswith("auto"):
                auto_count += 1
            elif method == "deterministic_fallback":
                fallback_count += 1
            else:
                enriched_count += 1
            print(f"  method={method}  [{elapsed:.1f}s]", flush=True)
            print(f"  topic   : {rag.get('legal_topic', '')[:80]}", flush=True)
            print(f"  question: {rag.get('legal_question', '')[:100]}", flush=True)
            print(f"  queries : {rag.get('natural_language_queries', [])}", flush=True)

    print(f"\n[summary]")
    print(f"  claude_enriched : {enriched_count}")
    print(f"  auto_classified : {auto_count}")
    print(f"  deterministic   : {fallback_count}")
    print(f"  total written   : {enriched_count + auto_count + fallback_count}")
    print(f"  output          : {OUTPUT_JSONL}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
