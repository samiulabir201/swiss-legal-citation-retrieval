"""Qwen3-8B LLM-as-judge for Swiss legal citation borderline candidates.

Ported from research/Untitled75.ipynb (the v12 pipeline that pushed val F1
0.681 -> 0.777). Use this AFTER threshold-based zoning: feed only the
borderline (in-between) candidates to the judge; auto-YES / auto-NO ones
bypass the LLM entirely (see ``route_and_judge``).

The judge applies a 7-category Swiss-court-specific rubric:
    1) substantive law, 2) definitions, 3) procedural rules,
    4) appeal provisions, 5) cost allocation, 6) court jurisdiction,
    7) constitutional principles.

Resource estimate (Qwen3-8B):
    bf16 weights ......... ~16 GB VRAM
    + KV cache @ 8k ctx .. ~1-2 GB
    + activations ........ ~1 GB
    => safe minimum: 1x A100-40GB / L4-24GB / RTX 3090-24GB / Colab A100.
    On 16 GB cards (T4 / V100) you must use 4-bit quantization
    (not implemented here -- pass a pre-quantized ``model_id``).

Decoding is greedy by default (``do_sample=False``) for reproducibility.

Colab cell (paste into a fresh GPU runtime):
---------------------------------------------------------------
!pip -q install "transformers>=4.51" accelerate tqdm

import sys, json, pathlib
sys.path.append("/content/drive/MyDrive/swiss_citation_extraction")
from scripts.llm_judge_qwen3 import QwenJudge, route_and_judge

judge = QwenJudge(model_id="Qwen/Qwen3-8B", dtype="bf16")

query = "Pre-trial detention review under StPO."
candidates = [
    {"citation": "Art. 221 StPO",
     "text": "Untersuchungs- und Sicherheitshaft sind nur zulaessig ...",
     "fused_score": 0.42},
    # ... more borderline candidates from your fused-score zone
]
judged = route_and_judge(query, candidates, judge,
                         fused_score_key="fused_score",
                         auto_yes_thresh=0.55, auto_no_thresh=0.25)
print(json.dumps(judged, indent=2, ensure_ascii=False))
---------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Verbatim 7-category prompt from research/Untitled75.ipynb (cell defining
# JUDGE_SYSTEM, lines 1585-1608 of nb_dump.txt). Do NOT edit -- changing the
# wording invalidates the F1 numbers reported in the notebook.
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = """You are a Swiss Federal Court (Bundesgericht) legal citation expert.

DOMAIN KNOWLEDGE — Swiss Legal Citation Practice:
Swiss court decisions (BGE) and legal briefs cite provisions across multiple categories:

1. SUBSTANTIVE LAW: The core articles governing the legal issue (e.g., StGB for criminal offenses, OR for contracts, ZGB for civil matters)
2. DEFINITIONS: Articles that define key legal terms used in the case (e.g., Art. 8 ATSG defines invalidity)
3. PROCEDURAL RULES: Articles governing how the case is processed (StPO for criminal procedure, ZPO for civil procedure)
4. APPEAL PROVISIONS: Articles about legal remedies — Beschwerde (Art. 393ff StPO), Berufung, appeal deadlines
5. COST ALLOCATION: Articles about who pays court costs and attorney fees (Art. 422, 428 StPO; Art. 64 BGG)
6. COURT JURISDICTION: Articles defining which court decides (Art. 37/39 StBOG, Art. 100 BGG for Federal Court)
7. CONSTITUTIONAL PRINCIPLES: Fair trial (Art. 29 BV), proportionality, good faith (Art. 2 ZGB)

A query about pre-trial detention will cite detention rules AND appeal rules AND cost rules AND court jurisdiction.
A query about disability insurance will cite insurance provisions AND definitions AND procedural rules.

YOUR TASK: For each candidate article, read the German text carefully and decide YES or NO.
Say YES if the article belongs in ANY of the 7 categories above for this specific legal query.
Say NO only if the article is from a completely unrelated legal domain.

When uncertain, say YES — it is better to include a marginally relevant article than to miss one.

For each candidate, respond with exactly:
CITATION | VERDICT: YES or NO"""


_CATEGORY_KEYWORDS = {
    "substantive":     "substantive",
    "definition":      "definitions",
    "procedural":      "procedural",
    "appeal":          "appeal",
    "cost":            "costs",
    "jurisdiction":    "jurisdiction",
    "constitutional":  "constitutional",
}


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _dtype_to_torch(dtype: str):
    import torch
    return {
        "bf16":   torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16":   torch.float16,
        "float16": torch.float16,
        "fp32":   torch.float32,
        "float32": torch.float32,
    }[dtype]


def _build_user_prompt(query: str, cand: Dict[str, Any]) -> str:
    """Single-candidate variant of the notebook's build_judge_prompt."""
    parts = [f"LEGAL QUERY: {str(query)[:600]}"]
    parts.append("\n1 CANDIDATE to judge:\n")
    cit = cand.get("citation", "")
    txt = cand.get("text") or "(kein Text)"
    enrich = cand.get("enrichment_summary")
    block = [f"[1] {cit}", f"    German text: {str(txt)[:400]}"]
    if enrich:
        block.append(f"    Enrichment: {str(enrich)[:300]}")
    parts.append("\n".join(block))
    parts.append("\nRespond with exactly: CITATION | VERDICT: YES or NO")
    return "\n".join(parts)


def _parse_verdict(raw: str, citation: str) -> Dict[str, str]:
    """Extract verdict + (best-effort) category. Default-YES on parse failure."""
    text = raw or ""
    if "</think>" in text:
        text = text.split("</think>")[-1]
    text_low = text.lower()

    # Category sniff (notebook only outputs YES/NO; categories are rubric-side,
    # so we infer by keyword match and fall back to "unspecified").
    category = "unspecified"
    for needle, label in _CATEGORY_KEYWORDS.items():
        if needle in text_low:
            category = label
            break

    verdict: Optional[str] = None
    for line in text.splitlines():
        if "|" not in line:
            continue
        right = line.split("|")[-1].strip().upper()
        if "YES" in right:
            verdict = "yes"; break
        if "NO" in right:
            verdict = "no"; break

    if verdict is None:
        # Loose fallback: scan whole response.
        if re.search(r"\bYES\b", text, re.I):
            verdict = "yes"
        elif re.search(r"\bNO\b", text, re.I):
            verdict = "no"
        else:
            # Default-YES on parse failure (notebook behaviour, line 1684).
            verdict = "yes"
            category = "unspecified"

    return {"verdict": verdict, "category": category}


# ---------------------------------------------------------------------------
class QwenJudge:
    """Wraps Qwen3-8B as a yes/no judge with a per-query, per-doc JSON cache.

    Heavy imports (torch, transformers) are deferred to ``__init__`` so that
    ``import scripts.llm_judge_qwen3`` works on a CPU-only host (e.g. for
    static-analysis or unit tests).
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-8B",
        device: str = "auto",
        dtype: str = "bf16",
        max_seq_len: int = 8192,
        do_sample: bool = False,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() and device != "cpu":
            raise RuntimeError(
                "CUDA required: QwenJudge expects a GPU (see top-of-file "
                "VRAM estimate). Set device='cpu' explicitly only for "
                "smoke tests -- bf16 inference on CPU is not supported."
            )

        self.model_id = model_id
        self.device = device
        self.max_seq_len = int(max_seq_len)
        self.do_sample = bool(do_sample)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=_dtype_to_torch(dtype),
            device_map=device,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    # ------------------------------------------------------------------
    def _cache_path(self, query: str, citation: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        sub = self.cache_dir / _sha1(query)
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{_sha1(citation)}.json"

    def _load_cached(self, query: str, citation: str) -> Optional[Dict[str, Any]]:
        p = self._cache_path(query, citation)
        if p is None or not p.exists():
            return None
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cached(self, query: str, citation: str, payload: Dict[str, Any]) -> None:
        p = self._cache_path(query, citation)
        if p is None:
            return
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _generate(self, prompts: List[str]) -> List[str]:
        import torch

        chats = [
            self.tokenizer.apply_chat_template(
                [{"role": "system", "content": JUDGE_SYSTEM},
                 {"role": "user",   "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        enc = self.tokenizer(
            chats,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=256,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        with torch.no_grad():
            out = self.model.generate(**enc, **gen_kwargs)

        outs = []
        for i in range(out.shape[0]):
            new_tokens = out[i][enc["input_ids"].shape[1]:]
            outs.append(
                self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            )
        return outs

    # ------------------------------------------------------------------
    def judge(self, query: str, candidate: Dict[str, Any]) -> Dict[str, str]:
        citation = str(candidate.get("citation", ""))
        cached = self._load_cached(query, citation)
        if cached is not None:
            return cached

        prompt = _build_user_prompt(query, candidate)
        try:
            raw = self._generate([prompt])[0]
            parsed = _parse_verdict(raw, citation)
        except Exception as exc:
            # Default-YES on any model failure (matches notebook fallback).
            raw = f"<error: {type(exc).__name__}: {exc}>"
            parsed = {"verdict": "yes", "category": "unspecified"}

        result = {
            "verdict": parsed["verdict"],
            "category": parsed["category"],
            "raw_response": raw,
        }
        self._save_cached(query, citation, result)
        return result

    def judge_batch(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        batch_size: int = 4,
    ) -> List[Dict[str, str]]:
        if not candidates:
            return []

        try:
            from tqdm.auto import tqdm
        except Exception:
            def tqdm(x, **_): return x

        results: List[Optional[Dict[str, str]]] = [None] * len(candidates)
        pending: List[int] = []

        # Cache pass.
        for i, c in enumerate(candidates):
            cit = str(c.get("citation", ""))
            cached = self._load_cached(query, cit)
            if cached is not None:
                results[i] = cached
            else:
                pending.append(i)

        # Batched generation pass.
        for start in tqdm(
            range(0, len(pending), max(1, batch_size)),
            desc="QwenJudge",
            leave=False,
        ):
            chunk = pending[start:start + batch_size]
            prompts = [_build_user_prompt(query, candidates[i]) for i in chunk]
            try:
                raws = self._generate(prompts)
            except Exception as exc:
                # Per-item fallback: re-run sequentially; if still failing,
                # default-YES.
                raws = []
                for p in prompts:
                    try:
                        raws.append(self._generate([p])[0])
                    except Exception as exc2:
                        raws.append(f"<error: {type(exc2).__name__}: {exc2}>")

            for idx, raw in zip(chunk, raws):
                cit = str(candidates[idx].get("citation", ""))
                if raw.startswith("<error:"):
                    parsed = {"verdict": "yes", "category": "unspecified"}
                else:
                    parsed = _parse_verdict(raw, cit)
                payload = {
                    "verdict": parsed["verdict"],
                    "category": parsed["category"],
                    "raw_response": raw,
                }
                results[idx] = payload
                self._save_cached(query, cit, payload)

        # mypy/runtime guard
        return [r if r is not None
                else {"verdict": "yes", "category": "unspecified", "raw_response": ""}
                for r in results]


# ---------------------------------------------------------------------------
def route_and_judge(
    query: str,
    candidates: List[Dict[str, Any]],
    judge: QwenJudge,
    fused_score_key: str = "fused_score",
    auto_yes_thresh: float = 0.55,
    auto_no_thresh: float = 0.25,
    batch_size: int = 4,
) -> List[Dict[str, Any]]:
    """Three-zone router: auto-YES / auto-NO / LLM-judge.

    Mirrors STAGE 3 + STAGE 4 of the v12 notebook
    (research/Untitled75.ipynb, ``compute_fused_and_split`` +
    ``judge_borderline``). Returns each candidate dict with a ``verdict``
    field added in-place ('yes' | 'no' | 'unclear') plus a ``category`` for
    LLM-judged ones ('auto' for the threshold-decided ones).
    """
    if auto_no_thresh > auto_yes_thresh:
        raise ValueError(
            f"auto_no_thresh ({auto_no_thresh}) must be <= "
            f"auto_yes_thresh ({auto_yes_thresh})"
        )

    enriched: List[Dict[str, Any]] = []
    borderline_idx: List[int] = []
    borderline_cands: List[Dict[str, Any]] = []

    for i, cand in enumerate(candidates):
        out = dict(cand)
        score = float(out.get(fused_score_key, 0.0) or 0.0)
        if score >= auto_yes_thresh:
            out["verdict"] = "yes"
            out["category"] = "auto"
        elif score < auto_no_thresh:
            out["verdict"] = "no"
            out["category"] = "auto"
        else:
            out["verdict"] = None  # filled in below
            borderline_idx.append(i)
            borderline_cands.append(out)
        enriched.append(out)

    if borderline_cands:
        verdicts = judge.judge_batch(query, borderline_cands, batch_size=batch_size)
        for i, v in zip(borderline_idx, verdicts):
            enriched[i]["verdict"] = v["verdict"]
            enriched[i]["category"] = v.get("category", "unspecified")
            enriched[i]["judge_raw"] = v.get("raw_response", "")

    return enriched


# ---------------------------------------------------------------------------
def _cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Run QwenJudge on a JSONL of borderline candidates."
    )
    p.add_argument("--query", required=True, help="Legal query text.")
    p.add_argument("--candidates-jsonl", required=True,
                   help="Path to JSONL of candidate dicts (one per line).")
    p.add_argument("--model-id", default="Qwen/Qwen3-8B")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--fused-score-key", default="fused_score")
    p.add_argument("--auto-yes-thresh", type=float, default=0.55)
    p.add_argument("--auto-no-thresh", type=float, default=0.25)
    p.add_argument("--out-jsonl", default=None)
    p.add_argument("--no-route", action="store_true",
                   help="Skip threshold routing; judge every candidate.")
    args = p.parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print("ERROR: torch not installed. CUDA required for QwenJudge.",
              file=sys.stderr)
        return 2

    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA required to run QwenJudge (no GPU detected).",
              file=sys.stderr)
        return 2

    cands: List[Dict[str, Any]] = []
    with open(args.candidates_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cands.append(json.loads(line))

    judge = QwenJudge(model_id=args.model_id, cache_dir=args.cache_dir)
    if args.no_route:
        verdicts = judge.judge_batch(args.query, cands, batch_size=args.batch_size)
        out = [{**c, **v} for c, v in zip(cands, verdicts)]
    else:
        out = route_and_judge(
            args.query, cands, judge,
            fused_score_key=args.fused_score_key,
            auto_yes_thresh=args.auto_yes_thresh,
            auto_no_thresh=args.auto_no_thresh,
            batch_size=args.batch_size,
        )

    sink = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else sys.stdout
    try:
        for row in out:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if args.out_jsonl:
            sink.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
