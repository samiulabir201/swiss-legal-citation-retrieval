#!/usr/bin/env python
"""Enrich Swiss court authority cards with LLM-derived semantic fields for RAG.

Input  : court_authority_cards_v4.jsonl  (built by build_court_authority_cards.py)
Output : court_authority_cards_rag.jsonl (one enriched card per line, append-mode)

Designed for Google Colab Pro/Pro+ with vLLM. Default model is Qwen3-32B-AWQ
(needs A100 40 GB or L4 24 GB). Override --model for smaller GPUs.

Throughput on A100/AWQ: ~12-18 requests/sec end-to-end.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator

DEFAULT_BASE = Path("/content/drive/MyDrive/swiss_law")

# ─── Pre-filter: trivial paragraphs that don't need an LLM ──────────────────
COST_PROC_RE = re.compile(
    r"(?:"
    r"\bgerichtskosten\b|\bprozesskosten\b|\bverfahrenskosten\b|"
    r"\bfrais judiciaires\b|\bfrais de la cause\b|\bd[eé]pens\b|"
    r"\bspese giudiziarie\b|\bripetibili\b|"
    r"\bparteientsch[äa]digung\b|\bhonoraire\b|"
    r"\bunentgeltliche rechtspflege\b|\bassistance judiciaire\b|"
    r"\bpatrocinio gratuito\b|"
    r"\bdie sache wird .{0,80}zur[üu]ckgewiesen\b|"
    r"\brenvoyer la cause\b|"
    r"\bla causa [eè] rinviata\b|"
    r"^\s*\d+\.\s*\d+\..{0,5}fr\.\s*\d"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# ─── JSON schema for guided generation ──────────────────────────────────────
RAG_SCHEMA = {
    "type": "object",
    "properties": {
        "english_summary":     {"type": "string"},
        "legal_topic":         {"type": "string"},
        "legal_question":      {"type": "string"},
        "legal_rule":          {"type": "string"},
        "court_holding":       {"type": "string"},
        "factual_context":     {"type": "string"},
        "english_legal_concepts":   {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "search_keywords":          {"type": "array", "items": {"type": "string"}, "maxItems": 10},
        "natural_language_queries": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "paragraph_role": {
            "type": "string",
            "enum": ["holding", "reasoning", "background", "cost",
                     "procedural", "disposition", "standard_of_review", "obiter"],
        },
        "outcome_signal": {
            "type": "string",
            "enum": ["granted", "dismissed", "inadmissible", "remitted", "partial", "none"],
        },
    },
    "required": [
        "english_summary", "legal_topic",
        "english_legal_concepts", "search_keywords",
        "natural_language_queries", "paragraph_role", "outcome_signal",
    ],
}

SYSTEM_PROMPT = (
    "You are a Swiss legal analyst. The user gives you a paragraph from a Swiss "
    "Federal Tribunal decision in German, French, or Italian. "
    "Translate every concept into precise English legal terminology and emit "
    "structured JSON to power English-language semantic-search RAG. "
    "Be concrete: prefer 'extension of pretrial detention based on flight risk' "
    "over 'detention'. Output ONLY the JSON object, no preamble."
)


def build_user_message(card: dict) -> str:
    text = card.get("text_excerpt_original", "")[:2500]
    citation = card.get("citation", "")
    legal_area = card.get("legal_area", "")
    existing = card.get("issue_labels_en") or []
    parts = [
        f"Citation: {citation}",
        f"Legal area (deterministic): {legal_area}",
    ]
    if existing:
        parts.append(f"Existing labels: {', '.join(existing[:8])}")
    parts.append("")
    parts.append("Paragraph (original language):")
    parts.append(text)
    parts.append("")
    parts.append("Produce the JSON now.")
    return "\n".join(parts)


# ─── Auto-classify trivial paragraphs without calling the LLM ───────────────
def auto_classify(card: dict) -> dict | None:
    if card.get("is_notification_paragraph"):
        return _stub(
            "Procedural notification of the judgment to the parties.",
            "judgment notification",
            ["service of judgment"],
            ["notification", "service", "judgment communication"],
            role="procedural", method="auto_notification",
        )
    text = card.get("text_excerpt_original", "") or ""
    if len(text) < 50:
        return _stub(
            "Short procedural fragment (cross-reference or one-line ruling).",
            "procedural fragment",
            [], [], role="procedural", method="auto_short",
        )
    if COST_PROC_RE.search(text[:400]):
        return _stub(
            "Court-cost or procedural-fee allocation paragraph.",
            "court costs and procedural fees",
            ["court costs", "procedural fees", "legal aid"],
            ["costs", "court fees", "frais judiciaires", "Gerichtskosten"],
            role="cost", method="auto_cost",
        )
    return None


def _stub(summary, topic, concepts, keywords, role, method):
    return {
        "english_summary": summary,
        "legal_topic": topic,
        "legal_question": "",
        "legal_rule": "",
        "court_holding": "",
        "factual_context": "",
        "english_legal_concepts": concepts,
        "search_keywords": keywords,
        "natural_language_queries": [],
        "paragraph_role": role,
        "outcome_signal": "none",
        "method": method,
    }


# ─── IO helpers ─────────────────────────────────────────────────────────────
def stream_input(path: Path, start_offset: int) -> Iterator[tuple[int, dict]]:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_offset or not line.strip():
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


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir",   default=str(DEFAULT_BASE), type=Path)
    ap.add_argument("--input",      default="court_authority_cards_v4.jsonl")
    ap.add_argument("--output",     default="court_authority_cards_rag.jsonl")
    ap.add_argument("--checkpoint", default="rag_checkpoint.txt")
    ap.add_argument("--model",      default="Qwen/Qwen3-32B-AWQ",
                    help="HF model id. T4: Qwen/Qwen2.5-7B-Instruct-AWQ. "
                         "L4: Qwen/Qwen3-14B-AWQ. A100: Qwen/Qwen3-32B-AWQ.")
    ap.add_argument("--quantization", default="awq")
    ap.add_argument("--batch-size",   type=int, default=32)
    ap.add_argument("--max-model-len",type=int, default=4096)
    ap.add_argument("--gpu-memory",   type=float, default=0.92)
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="Set >1 if you have multiple GPUs (e.g. 2x A100/L40S).")
    ap.add_argument("--temperature",  type=float, default=0.2)
    ap.add_argument("--max-tokens",   type=int, default=600)
    ap.add_argument("--limit",        type=int, default=0)
    args = ap.parse_args()

    base = args.base_dir
    in_path  = base / args.input
    out_path = base / args.output
    ck_path  = base / args.checkpoint
    base.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"ERROR: input not found at {in_path}", file=sys.stderr)
        sys.exit(1)

    # Resume from checkpoint
    start = 0
    if ck_path.exists():
        try:
            start = int(ck_path.read_text().strip() or "0")
        except ValueError:
            start = 0
    print(f"[enrich] resuming at line {start:,}", flush=True)

    # Lazy imports so --help works without GPU deps
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    from tqdm.auto import tqdm

    quant = args.quantization if any(t in args.model for t in ("AWQ", "GPTQ")) else None
    llm = LLM(
        model=args.model,
        quantization=quant,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        dtype="auto",
        enable_prefix_caching=True,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    sp = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
        guided_decoding=GuidedDecodingParams(json=RAG_SCHEMA),
    )

    print(f"[enrich] counting lines in {in_path.name} ...", flush=True)
    total = count_lines(in_path)
    if args.limit:
        total = min(total, start + args.limit)
    print(f"[enrich] total={total:,}, to process={total - start:,}", flush=True)

    out_f = out_path.open("a", encoding="utf-8")
    pbar = tqdm(total=total, initial=start, desc="enrich", unit="card", smoothing=0.05)

    pending: list[tuple[int, dict]] = []

    def flush_batch():
        nonlocal pending
        if not pending:
            return
        convs = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_message(card)},
            ]
            for _, card in pending
        ]
        outputs = llm.chat(convs, sampling_params=sp, use_tqdm=False)
        for (line_idx, card), out in zip(pending, outputs):
            raw = out.outputs[0].text.strip()
            try:
                enriched = json.loads(raw)
                enriched["method"] = "qwen_v1"
            except json.JSONDecodeError:
                enriched = _stub(
                    "", "", [], [], role="reasoning", method="json_parse_failed"
                )
                enriched["raw_output"] = raw[:400]
            card["rag_enrichment"] = enriched
            out_f.write(json.dumps(card, ensure_ascii=False) + "\n")
        out_f.flush()
        ck_path.write_text(str(pending[-1][0] + 1))
        pbar.update(len(pending))
        pending = []

    processed = 0
    for line_idx, card in stream_input(in_path, start):
        if args.limit and processed >= args.limit:
            break

        auto = auto_classify(card)
        if auto is not None:
            card["rag_enrichment"] = auto
            out_f.write(json.dumps(card, ensure_ascii=False) + "\n")
            ck_path.write_text(str(line_idx + 1))
            pbar.update(1)
            processed += 1
            continue

        pending.append((line_idx, card))
        if len(pending) >= args.batch_size:
            flush_batch()
        processed += 1

    flush_batch()
    out_f.close()
    pbar.close()
    print(f"[enrich] done. wrote → {out_path}", flush=True)


if __name__ == "__main__":
    main()
