#!/usr/bin/env python
"""LLM agent for laws_de.csv — produces v2 semantic enrichment for 175,933 articles.

Counterpart of the 363k court descriptor run. Reads the slim input from
`build_law_llm_input.py`, runs Qwen3-8B-AWQ via vLLM (with HF transformers
fallback), and emits one JSON descriptor per article keyed by `_source_row`.

The prompt is tuned for **Swiss legal interpretation in English**, not for
literal translation. Original German legal terms are preserved verbatim and
mapped to their canonical Swiss-legal English equivalents (e.g. `Bewilligung`
→ "permit / authorisation"; `Rechtsbegehren` → "prayer for relief"). Static
fields (citation, law code, article structure, anchors, year) are NEVER
re-derived by the LLM — they are merged back in by `merge_law_llm_into_v1_cards.py`.

Usage (Linux/Kaggle/Colab; vLLM available):
    python scripts/run_law_llm_enrichment.py \\
        --input  artifacts/law_llm_input.jsonl \\
        --output artifacts/law_llm_descriptors.jsonl

Smoke (50 rows on any backend):
    python scripts/run_law_llm_enrichment.py --limit 50 \\
        --output artifacts/law_llm_descriptors.smoke.jsonl

The runner streams append-mode and skips already-processed `_source_row`s on
restart, so interrupting and resuming is safe.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"

DEFAULT_INPUT = ART_DIR / "law_llm_input.jsonl"
DEFAULT_OUTPUT = ART_DIR / "law_llm_descriptors.jsonl"
DEFAULT_FAILURES = ART_DIR / "law_llm_descriptors_failures.jsonl"
DEFAULT_SUMMARY = ART_DIR / "law_llm_descriptors.summary.json"


# ───────────────────────────── Schema ──────────────────────────────────────

DESCRIPTOR_KEYS = [
    "english_summary",
    "legal_rule",
    "applicability_conditions",
    "exceptions_or_limitations",
    "legal_question",
    "concepts_en",
    "terms_de_to_en",
    "defined_terms",
    "addressees",
    "sanctions_or_consequences",
    "provision_role_llm",
    "specificity_score",
]

PROVISION_ROLES = {
    "definition", "purpose", "scope", "principle",
    "right_or_entitlement", "duty", "prohibition", "procedure",
    "competence", "sanction_or_penalty",
    "data_reporting", "fees_or_costs", "transitional_or_commencement",
    "other",
}

BOILERPLATE_ROLES = {"transitional_or_commencement", "fees_or_costs", "data_reporting"}

LLM_SCHEMA_HINT = {
    "english_summary": "<=2 sentences, English; describe what the article actually says",
    "legal_rule": "<=25 words; operative rule 'X must/may/shall/is forbidden to Y'; empty for boilerplate/transitional/fee schedules",
    "applicability_conditions": ["0-5 short conditions (English) describing when the rule applies"],
    "exceptions_or_limitations": ["0-5 short carve-outs (English): 'unless', 'except where', time/scope limits"],
    "legal_question": "<=18 words; one English question this provision answers; empty for boilerplate",
    "concepts_en": ["3-8 broad English legal concepts"],
    "terms_de_to_en": [{"de": "verbatim German legal term from text", "en": "Swiss-legal English equivalent (NOT a literal translation)"}],
    "defined_terms": [{"term": "verbatim German term defined in this article", "definition": "English gloss of the definition"}],
    "addressees": ["0-6 English labels: who is bound (e.g. 'federal authority', 'employer', 'data subject')"],
    "sanctions_or_consequences": ["0-4 English items explicitly named in text"],
    "provision_role_llm": "definition|purpose|scope|principle|right_or_entitlement|duty|prohibition|procedure|competence|sanction_or_penalty|data_reporting|fees_or_costs|transitional_or_commencement|other",
    "specificity_score": "0..1 (0 = generic boilerplate, 1 = very specific operative rule)",
}

_SCHEMA_JSON = json.dumps(LLM_SCHEMA_HINT, ensure_ascii=False, separators=(",", ":"))


SYSTEM_PROMPT = f"""You are a Swiss legal-interpretation assistant working on individual articles
of Swiss federal law (Bundesgesetze, Verordnungen, the Bundesverfassung, Verträge,
SR-numbered statutes). The article text is in German, French, or Italian. Your
job is to extract the operative legal meaning into English, preserving the
exact original-language legal terms.

Return exactly one compact JSON object with exactly this shape (all keys present):
{_SCHEMA_JSON}

Hard rules:
1. JSON only. No prose, no markdown, no preamble.
2. Use English for ALL semantic fields except `terms_de_to_en[].de`,
   `defined_terms[].term`, and the verbatim parts inside other strings if needed.
3. `terms_de_to_en[].de` and `defined_terms[].term` MUST be exact substrings of
   the source text. Do not paraphrase or normalise capitalisation.
4. Map each German term to its **Swiss-legal English equivalent**, NOT a literal
   translation. Examples:
     Bewilligung -> permit / authorisation
     Verfügung -> formal administrative order
     Rechtsbegehren -> prayer for relief
     Zuständigkeit -> jurisdiction / competence
     Aufsichtsbehörde -> supervisory authority
     Inverkehrbringen -> placing on the market
     Tatbestand -> set of facts / elements of the offence
     Beschwerde -> appeal
     Eidgenössisch -> federal (Swiss)
     Bundesrat -> Federal Council
     EDI / SBFI / FINMA / ESTV / EJPD -> keep the acronym verbatim
   When in doubt, prefer the EU/UK statutory English wording over US wording.
5. Do NOT invent statute citations, BGE numbers, dates, party names, or
   sanctions that are not literally in the text.
6. Do NOT translate literally. If the German is metaphorical or formal,
   pick the recognised legal English term.
7. Never copy the law title into `english_summary`. Describe what THIS article
   says, not what the parent statute is about.
8. For repeal markers ("Aufgehoben"), commencement clauses ("Tritt am ... in Kraft"),
   pure fee tables, annex code lists, or transitional provisions:
     - keep `legal_rule`, `applicability_conditions`, `exceptions_or_limitations`,
       `legal_question` empty.
     - still fill `english_summary`, `concepts_en`, `terms_de_to_en`,
       `provision_role_llm`, `specificity_score` (low).
9. Never put `Art.`, `Abs.`, `Buchstabe`, `Ziffer`, or SR numbers into
   `terms_de_to_en` — those are anchors, not legal terms.
10. Never produce more than 10 terms_de_to_en, 4 defined_terms, 6 addressees,
    5 conditions, 5 exceptions, or 8 concepts_en. Trim to the most salient.

If a field has nothing to populate, return an empty string or empty list, but
the key MUST be present. JSON only."""


USER_TEMPLATE = """Citation: {citation}
Law title: {law_title}
Section path: {section_path}
Law code: {law_code}   Article: {article}   Units: {units}
Source type: {source_type}   Enactment year: {year}
Static legal area hint: {legal_area_hint}
Static domain hints: {domain_hints}

Article text (verbatim, original language):
\"\"\"
{text}
\"\"\"

Return JSON only."""


# ───────────────────────── Config ──────────────────────────────────────────

@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-8B-AWQ"
    max_model_len: int = 4096
    max_text_chars: int = 6000
    max_new_tokens: int = 700
    retry_max_new_tokens: int = 850
    temperature: float = 0.1
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    enable_thinking: bool = False
    max_retries: int = 1
    batch_size: int = 64
    gpu_memory_utilization: float = 0.92
    max_num_seqs: int = 256
    quantization: str | None = "awq"
    kv_cache_dtype: str | None = "fp8"
    enforce_eager: bool = False
    disable_custom_all_reduce: bool = True
    tensor_parallel_size: int = 1
    enable_prefix_caching: bool = True
    progress_every: int = 1000
    include_raw_output_on_success: bool = False


# ───────────────────────── JSON parsing helpers ────────────────────────────

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(raw: str) -> dict[str, Any]:
    if raw is None:
        raise ValueError("empty output")
    s = raw.strip()
    # Strip a leading ```json fence if present.
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
    m = _JSON_BLOCK_RE.search(s)
    if not m:
        raise ValueError("no JSON object found")
    js = m.group(0)
    try:
        return json.loads(js)
    except json.JSONDecodeError:
        # Conservative repair: strip trailing commas before closers.
        js2 = re.sub(r",(\s*[}\]])", r"\1", js)
        return json.loads(js2)


def normalize_descriptor(obj: dict[str, Any], source_text: str = "") -> dict[str, Any]:
    """Coerce LLM output into the contract; never raise."""
    out: dict[str, Any] = {}
    for k in DESCRIPTOR_KEYS:
        v = obj.get(k)
        if k in ("applicability_conditions", "exceptions_or_limitations", "concepts_en",
                 "addressees", "sanctions_or_consequences"):
            out[k] = [str(x).strip() for x in (v or []) if str(x).strip()][:8]
        elif k == "terms_de_to_en":
            cleaned = []
            for item in v or []:
                if isinstance(item, dict):
                    de = str(item.get("de", "")).strip()
                    en = str(item.get("en", "")).strip()
                    if de and en:
                        cleaned.append({"de": de, "en": en})
            out[k] = cleaned[:10]
        elif k == "defined_terms":
            cleaned = []
            for item in v or []:
                if isinstance(item, dict):
                    term = str(item.get("term", "")).strip()
                    defn = str(item.get("definition", "")).strip()
                    if term and defn:
                        cleaned.append({"term": term, "definition": defn})
            out[k] = cleaned[:4]
        elif k == "provision_role_llm":
            role = str(v or "other").strip().lower()
            out[k] = role if role in PROVISION_ROLES else "other"
        elif k == "specificity_score":
            try:
                out[k] = max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                out[k] = 0.0
        else:
            out[k] = str(v or "").strip()
    return out


def empty_descriptor(error: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in DESCRIPTOR_KEYS:
        if k in ("applicability_conditions", "exceptions_or_limitations", "concepts_en",
                 "addressees", "sanctions_or_consequences", "terms_de_to_en", "defined_terms"):
            out[k] = []
        elif k == "provision_role_llm":
            out[k] = "other"
        elif k == "specificity_score":
            out[k] = 0.0
        else:
            out[k] = ""
    return out


def grounded_terms_pct(terms: list[dict], source_text: str) -> float:
    if not terms:
        return 1.0
    if not source_text:
        return 0.0
    hits = sum(1 for t in terms if t.get("de") and t["de"] in source_text)
    return hits / len(terms)


# ───────────────────────── Backends ────────────────────────────────────────

class VLLMBackend:
    def __init__(self, cfg: Config):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        self.SamplingParams = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        kwargs: dict[str, Any] = dict(
            model=cfg.model_name,
            trust_remote_code=True,
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            max_model_len=cfg.max_model_len,
            max_num_seqs=cfg.max_num_seqs,
            enforce_eager=cfg.enforce_eager,
            disable_custom_all_reduce=cfg.disable_custom_all_reduce,
            disable_log_stats=True,
            quantization=cfg.quantization,
            enable_prefix_caching=cfg.enable_prefix_caching,
        )
        if cfg.kv_cache_dtype:
            kwargs["kv_cache_dtype"] = cfg.kv_cache_dtype
        self.llm = LLM(**kwargs)
        self.cfg = cfg

    def render(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.cfg.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    def generate(self, prompts: list[str], max_tokens: int) -> list[str]:
        params = self.SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=max_tokens,
            repetition_penalty=self.cfg.repetition_penalty,
        )
        outs = self.llm.generate(prompts, sampling_params=params, use_tqdm=False)
        return [o.outputs[0].text if o.outputs else "" for o in outs]


class HFBackend:
    """Fallback for environments without vLLM (slow; smoke testing only)."""

    def __init__(self, cfg: Config):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.cfg = cfg

    def render(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.cfg.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    def generate(self, prompts: list[str], max_tokens: int) -> list[str]:
        outs = []
        for p in prompts:
            inputs = self.tokenizer(p, return_tensors="pt").to(self.model.device)
            with self.torch.inference_mode():
                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=self.cfg.temperature > 0,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    repetition_penalty=self.cfg.repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            outs.append(text)
        return outs


def load_backend(cfg: Config, force: str | None) -> Any:
    if force == "hf":
        print("[backend] forcing HF transformers", flush=True)
        return HFBackend(cfg)
    if force == "vllm":
        return VLLMBackend(cfg)
    try:
        return VLLMBackend(cfg)
    except Exception as exc:
        print(f"[backend] vLLM unavailable ({type(exc).__name__}: {exc}); "
              f"falling back to HF transformers", flush=True)
        return HFBackend(cfg)


# ───────────────────────── Prompt building ─────────────────────────────────

_WS_RE = re.compile(r"\s+")


def trim_text(text: str, max_chars: int) -> str:
    text = _WS_RE.sub(" ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head].rstrip() + " ... [TRUNCATED] ... " + text[-tail:].lstrip()


def build_user_prompt(rec: dict, max_chars: int) -> str:
    structural = rec.get("structural") or {}
    title_meta = rec.get("title_metadata") or {}
    hints = rec.get("static_hints") or {}
    return USER_TEMPLATE.format(
        citation=rec.get("citation", ""),
        law_title=rec.get("law_title", "")[:300],
        section_path=rec.get("title_section_path", "")[:200],
        law_code=structural.get("law_code", ""),
        article=structural.get("article", ""),
        units=", ".join(structural.get("units", []) or []),
        source_type=title_meta.get("source_type", ""),
        year=title_meta.get("enactment_year", ""),
        legal_area_hint=hints.get("legal_area_static", ""),
        domain_hints=", ".join((hints.get("domain_labels_en") or [])[:5]),
        text=trim_text(rec.get("text", ""), max_chars),
    )


# ───────────────────────── Main loop ───────────────────────────────────────

def stream_input(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def already_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                src = rec.get("_source_row")
                if isinstance(src, int):
                    done.add(src)
            except Exception:
                continue
    return done


def process(args: argparse.Namespace) -> dict:
    cfg = Config(
        model_name=args.model,
        max_model_len=args.max_model_len,
        max_text_chars=args.max_text_chars,
        max_new_tokens=args.max_new_tokens,
        max_num_seqs=args.max_num_seqs,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
    )

    backend = load_backend(cfg, args.backend)

    done = already_done(args.output) if args.resume else set()
    print(f"[resume] {len(done):,} _source_rows already in {args.output}", flush=True)

    pending: list[dict] = []
    counts = Counter()
    role_dist = Counter()
    t0 = time.time()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fout = args.output.open("a" if args.resume else "w", encoding="utf-8")
    ffail = args.failures.open("a" if args.resume else "w", encoding="utf-8")

    def flush_batch(batch: list[dict]):
        if not batch:
            return
        prompts = [backend.render(SYSTEM_PROMPT, build_user_prompt(r, cfg.max_text_chars)) for r in batch]
        raws = backend.generate(prompts, cfg.max_new_tokens)

        retry_idx: list[int] = []
        retry_prompts: list[str] = []
        results: list[tuple[dict, dict, str]] = []  # (descriptor, llm_generation, raw)

        for i, (rec, raw) in enumerate(zip(batch, raws)):
            try:
                obj = extract_json_object(raw)
                desc = normalize_descriptor(obj, rec.get("text", ""))
                results.append((desc, {"status": "ok", "attempt_count": 1}, raw))
            except Exception as exc:
                retry_idx.append(i)
                retry_prompts.append(backend.render(
                    SYSTEM_PROMPT,
                    f"The previous output was invalid JSON. Parser error: {exc!r}\n"
                    f"Previous output:\n{(raw or '')[:1200]}\n\n"
                    f"Repair by returning exactly one compact JSON object using the same schema.\n\n"
                    + build_user_prompt(rec, cfg.max_text_chars),
                ))
                results.append(({}, {"status": "pending_retry", "error": repr(exc), "raw_output": raw}, raw))

        if retry_prompts:
            retry_raws = backend.generate(retry_prompts, cfg.retry_max_new_tokens)
            for slot, retry_raw in zip(retry_idx, retry_raws):
                rec = batch[slot]
                try:
                    obj = extract_json_object(retry_raw)
                    desc = normalize_descriptor(obj, rec.get("text", ""))
                    results[slot] = (desc, {"status": "ok_after_retry", "attempt_count": 2}, retry_raw)
                except Exception as exc:
                    desc = empty_descriptor(repr(exc))
                    results[slot] = (
                        desc,
                        {"status": "failed_descriptor_parse", "attempt_count": 2,
                         "error": repr(exc), "raw_output": retry_raw},
                        retry_raw,
                    )

        for rec, (desc, gen, raw) in zip(batch, results):
            grounded = grounded_terms_pct(desc.get("terms_de_to_en", []), rec.get("text", ""))
            role = desc.get("provision_role_llm", "other")
            role_dist[role] += 1

            out_rec = {
                "_source_row": rec["_source_row"],
                "citation": rec.get("citation", ""),
                "language": "de",
                "llm_enrichment": desc,
                "llm_generation": gen,
                "llm_quality": {
                    "json_valid": gen.get("status", "").startswith("ok"),
                    "terms_grounded_pct": round(grounded, 3),
                    "boilerplate_role": role in BOILERPLATE_ROLES,
                },
            }
            if cfg.include_raw_output_on_success and gen.get("status", "").startswith("ok"):
                out_rec["llm_generation"]["raw_output"] = raw

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            counts["written"] += 1
            counts[gen.get("status", "unknown")] += 1
            if not gen.get("status", "").startswith("ok"):
                ffail.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                counts["failed"] += 1

        fout.flush()
        ffail.flush()

    try:
        for rec in stream_input(args.input):
            counts["scanned"] += 1
            if rec["_source_row"] in done:
                counts["already_done"] += 1
                continue
            if not (rec.get("text") or "").strip():
                counts["empty_text"] += 1
                continue
            pending.append(rec)
            if len(pending) >= cfg.batch_size:
                flush_batch(pending)
                pending.clear()
                if cfg.progress_every and counts["written"] and counts["written"] % cfg.progress_every < cfg.batch_size:
                    rate = counts["written"] / max(time.time() - t0, 1e-9)
                    print(f"  written={counts['written']:>8,}  ok={counts.get('ok',0):>8,}  "
                          f"retry_ok={counts.get('ok_after_retry',0):>6,}  "
                          f"failed={counts.get('failed',0):>5,}  "
                          f"({rate:>5.1f} rec/s)", flush=True)
            if args.limit and counts["written"] + counts["already_done"] >= args.limit:
                break

        flush_batch(pending)
    finally:
        fout.close()
        ffail.close()
        try:
            del backend
            gc.collect()
        except Exception:
            pass

    elapsed = time.time() - t0
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "failures": str(args.failures),
        "model": cfg.model_name,
        "elapsed_seconds": round(elapsed, 1),
        "rows_per_second": round(counts["written"] / max(elapsed, 1e-9), 2),
        "counts": dict(counts),
        "provision_role_distribution": role_dist.most_common(),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] written={counts['written']:,}  failed={counts.get('failed',0):,}  "
          f"in {elapsed/60:.1f} min", flush=True)
    print(f"[done] output:  {args.output}", flush=True)
    print(f"[done] summary: {args.summary}", flush=True)
    return summary


# ───────────────────────── CLI ─────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--model", default="Qwen/Qwen3-8B-AWQ")
    ap.add_argument("--backend", choices=["vllm", "hf"], default=None,
                    help="Force backend; default: try vLLM, fall back to HF.")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-text-chars", type=int, default=6000)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Rows per backend.generate() call (default: 64).")
    ap.add_argument("--progress-every", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after this many rows (including already-resumed). 0 = all.")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="Overwrite output instead of skipping rows already present.")
    ap.set_defaults(resume=True)
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"input JSONL not found: {args.input}\n"
                         f"Run scripts/build_law_llm_input.py first.")

    process(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
