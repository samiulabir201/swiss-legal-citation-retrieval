#!/usr/bin/env python
"""Qwen3-Reranker-8B cross-encoder for Swiss legal citation retrieval.

Ports the reranker logic referenced by the recall-funnel notebook (which
scored val F1 = 0.777, beating the threshold-only baseline at 0.681) into a
clean, GPU-ready module.

------------------------------------------------------------------
Required model weights
------------------------------------------------------------------
- ``Qwen/Qwen3-Reranker-8B`` is auto-downloaded from the Hugging Face Hub the
  first time ``QwenReranker`` is instantiated. Set ``HF_HOME`` / ``HF_TOKEN``
  to control cache location and authentication if needed.

------------------------------------------------------------------
VRAM estimate
------------------------------------------------------------------
- bf16 / fp16 weights: ~16 GB.
- With activations + batch_size 8 at max_seq_len 4096: budget **~22-30 GB**.
- Single A100-40 GB or L4-24 GB (small batch) works. T4-16 GB is *not* enough
  for fp16; on T4 prefer batch_size 1-2 and shorter ``max_seq_len`` (e.g. 2048)
  with ``dtype="fp16"``.

------------------------------------------------------------------
Example Colab / Kaggle cell
------------------------------------------------------------------
    !pip install -q "transformers>=4.51" accelerate
    !huggingface-cli login                    # only if the model card is gated

    from scripts.rerank_qwen3 import QwenReranker, rerank_candidates

    rr = QwenReranker(dtype="bf16", max_seq_len=4096)
    cand = [
        {"doc_id": "StPO-221", "citation": "Art. 221 StPO",
         "text": "Pretrial detention may be ordered if ...", "score": 7.4},
        {"doc_id": "BGE-141-IV-87", "citation": "BGE 141 IV 87",
         "text": "The Federal Court considers ...",        "score": 5.1},
    ]
    ranked = rerank_candidates("Swiss pretrial detention collusion risk",
                               cand, rr, top_k=50)
    for r in ranked[:5]:
        print(r["fused_score"], r["rerank_score"], r["citation"])

------------------------------------------------------------------
NOTE
------------------------------------------------------------------
The prompt scaffold below follows the **official Qwen3-Reranker template** as
published on the Hugging Face model card. This is what the source notebook
uses verbatim. If you keep a private modification (custom system prompt,
extra instruction wording, etc.), override via the ``instruction`` argument
and/or pass a different ``prefix`` / ``suffix`` when constructing
``QwenReranker``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

# Lightweight imports (always work, no GPU). Heavy imports (``torch``,
# ``transformers``) are deferred to instantiation so the module can be
# imported and structurally validated on a CPU-only box.

# ----------------------------------------------------------------------------
# Default Swiss-legal instruction (notebook-tailored).
# ----------------------------------------------------------------------------

DEFAULT_INSTRUCTION = (
    "Given an English-language Swiss legal question, judge whether the "
    "Document is a relevant authority (statute provision or court "
    "consideration) that the question's answer would cite."
)

# Official Qwen3-Reranker scaffold from the model card. Keep verbatim.
QWEN_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Note that the answer can only be \"yes\" or "
    '"no".'
)

QWEN_PREFIX = (
    "<|im_start|>system\n"
    + QWEN_SYSTEM_PROMPT
    + "<|im_end|>\n<|im_start|>user\n"
)
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _format_pair(query: str, document: str, instruction: str) -> str:
    """Compose the user-turn payload for a (query, document) pair."""
    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )


# ----------------------------------------------------------------------------
# Reranker
# ----------------------------------------------------------------------------


class QwenReranker:
    """Cross-encoder wrapper around ``Qwen/Qwen3-Reranker-8B``.

    The score for a (query, document) pair is the probability that the model
    answers ``"yes"`` rather than ``"no"`` to the judging prompt::

        log_softmax([no_logit, yes_logit])[..., 1].exp()  # in [0, 1]
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Reranker-8B",
        device: str = "auto",
        dtype: str = "bf16",
        max_seq_len: int = 4096,
        cache_dir: str | os.PathLike | None = None,
        prefix: str = QWEN_PREFIX,
        suffix: str = QWEN_SUFFIX,
        trust_remote_code: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() and device != "cpu":
            print(
                "[QwenReranker] WARNING: no CUDA detected. Loading the 8B "
                "reranker on CPU is extremely slow and likely to OOM. "
                "GPU required for any practical use.",
                file=sys.stderr,
            )

        torch_dtype = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)

        device_map = device if device != "auto" else "auto"

        self.model_id = model_id
        self.max_seq_len = int(max_seq_len)
        self.dtype = torch_dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        ).eval()

        # Resolve yes / no token IDs once.
        yes_ids = self.tokenizer("yes", add_special_tokens=False)["input_ids"]
        no_ids = self.tokenizer("no", add_special_tokens=False)["input_ids"]
        if len(yes_ids) != 1 or len(no_ids) != 1:
            raise RuntimeError(
                "Unexpected tokenization for yes/no: "
                f"yes={yes_ids}, no={no_ids}. "
                "The Qwen3 BPE should map both to a single token each."
            )
        self.yes_token_id = int(yes_ids[0])
        self.no_token_id = int(no_ids[0])

        # Pre-tokenized prefix/suffix (the user-turn body is sandwiched
        # between them). Mirrors the official Qwen reranker example.
        self.prefix_ids = self.tokenizer(
            prefix, add_special_tokens=False
        )["input_ids"]
        self.suffix_ids = self.tokenizer(
            suffix, add_special_tokens=False
        )["input_ids"]

        # Per-query JSON cache.
        self.cache_dir: Path | None = (
            Path(cache_dir) if cache_dir is not None else None
        )
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _query_hash(query: str) -> str:
        return hashlib.sha1(query.encode("utf-8")).hexdigest()

    def _cache_path(self, query: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self._query_hash(query)}.json"

    def _load_cache(self, query: str) -> dict[str, float]:
        path = self._cache_path(query)
        if path is None or not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_cache(self, query: str, scores: dict[str, float]) -> None:
        path = self._cache_path(query)
        if path is None:
            return
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False)
        tmp.replace(path)

    # ----------------------------------------------------------- tokenization

    def _build_inputs(
        self,
        queries: list[str],
        documents: list[str],
        instruction: str,
    ):
        """Tokenize a batch of (query, document) pairs with the official
        Qwen3-Reranker scaffold and left-padding."""
        import torch

        bodies = [
            _format_pair(q, d, instruction) for q, d in zip(queries, documents)
        ]

        # Reserve room for prefix/suffix.
        body_budget = max(
            8, self.max_seq_len - len(self.prefix_ids) - len(self.suffix_ids)
        )
        body_enc = self.tokenizer(
            bodies,
            add_special_tokens=False,
            truncation=True,
            max_length=body_budget,
            return_attention_mask=False,
        )

        full_ids: list[list[int]] = []
        for ids in body_enc["input_ids"]:
            full_ids.append(list(self.prefix_ids) + list(ids) + list(self.suffix_ids))

        # Left-pad so the final assistant position is the last column.
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_len = max(len(seq) for seq in full_ids)
        input_ids = []
        attention_mask = []
        for seq in full_ids:
            pad_n = max_len - len(seq)
            input_ids.append([pad_id] * pad_n + seq)
            attention_mask.append([0] * pad_n + [1] * len(seq))

        device = next(self.model.parameters()).device
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(
                attention_mask, dtype=torch.long, device=device
            ),
        }

    # ------------------------------------------------------------------ score

    def score(
        self,
        query: str,
        document: str,
        instruction: str | None = None,
    ) -> float:
        """Score one (query, document) pair. Returns ``P(relevant)`` in [0, 1]."""
        return self.score_batch(
            query, [document], instruction=instruction, batch_size=1
        )[0]

    def score_batch(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
        batch_size: int = 8,
        doc_ids: list[str] | None = None,
        use_cache: bool = True,
    ) -> list[float]:
        """Score one query against many documents. Returns a list of
        probabilities in [0, 1] aligned with ``documents``.

        If ``cache_dir`` was set on the reranker and ``doc_ids`` is provided,
        cached scores are reused and new ones are persisted on completion.
        """
        import torch

        instruction = instruction or DEFAULT_INSTRUCTION
        n = len(documents)
        if n == 0:
            return []

        # Cache lookup keyed by doc_id.
        cache: dict[str, float] = (
            self._load_cache(query) if (use_cache and doc_ids is not None) else {}
        )
        scores: list[float | None] = [None] * n
        todo_idx: list[int] = []
        for i in range(n):
            if doc_ids is not None and doc_ids[i] in cache:
                scores[i] = float(cache[doc_ids[i]])
            else:
                todo_idx.append(i)

        # Inference in batches over the uncached items.
        for start in range(0, len(todo_idx), batch_size):
            chunk = todo_idx[start : start + batch_size]
            batch_docs = [documents[i] for i in chunk]
            inputs = self._build_inputs(
                [query] * len(chunk), batch_docs, instruction
            )

            with torch.inference_mode():
                logits = self.model(**inputs).logits  # [B, T, V]
                last = logits[:, -1, :]  # [B, V]
                yes = last[:, self.yes_token_id]
                no = last[:, self.no_token_id]
                stacked = torch.stack([no, yes], dim=-1)  # [B, 2]
                probs = torch.log_softmax(stacked, dim=-1)[:, 1].exp()
                probs = probs.float().cpu().tolist()

            for j, i in enumerate(chunk):
                p = float(probs[j])
                # Guard against fp underflow / NaN.
                if not math.isfinite(p):
                    p = 0.0
                scores[i] = p
                if doc_ids is not None:
                    cache[doc_ids[i]] = p

        if use_cache and doc_ids is not None and self.cache_dir is not None:
            self._save_cache(query, cache)

        # ``None`` should not survive — all positions filled.
        return [float(s) for s in scores]  # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# Integration helper: fuse retriever score + rerank score.
# ----------------------------------------------------------------------------


def _minmax(values: Iterable[float]) -> list[float]:
    vals = [float(v) for v in values]
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    if hi - lo < 1e-12:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def rerank_candidates(
    query: str,
    candidates: list[dict],
    reranker: QwenReranker,
    top_k: int = 200,
    instruction: str | None = None,
    batch_size: int = 8,
    retrieval_score_key: str = "score",
    text_key: str = "text",
    doc_id_key: str = "doc_id",
    alpha_retrieval: float = 0.7,
    alpha_rerank: float = 0.3,
) -> list[dict]:
    """Rerank the top-N hybrid retriever candidates with Qwen3-Reranker-8B.

    Parameters
    ----------
    candidates
        Hybrid retriever output. Each dict must carry ``doc_id``, ``text`` and
        ``score`` (the BM25 / fused retrieval score). ``citation`` is
        passed through if present.
    top_k
        Only the first ``top_k`` candidates (in input order, i.e. the
        retriever's own ranking) are reranked. The remainder are returned
        unchanged at the tail with ``rerank_score = None``.

    Returns
    -------
    list[dict]
        Same dicts, sorted by ``fused_score`` descending. Each reranked
        candidate has two new keys::

            rerank_score: float in [0, 1]
            fused_score : 0.7 * minmax(retrieval_score) + 0.3 * rerank_score
    """
    if not candidates:
        return []

    head = list(candidates[:top_k])
    tail = list(candidates[top_k:])

    documents = [c.get(text_key, "") or "" for c in head]
    doc_ids = [str(c.get(doc_id_key, "")) for c in head]

    rerank_scores = reranker.score_batch(
        query,
        documents,
        instruction=instruction,
        batch_size=batch_size,
        doc_ids=doc_ids if all(doc_ids) else None,
    )

    retrieval_scores = [float(c.get(retrieval_score_key, 0.0)) for c in head]
    retrieval_norm = _minmax(retrieval_scores)

    out: list[dict] = []
    for c, rs, rr in zip(head, retrieval_norm, rerank_scores):
        item = dict(c)
        item["rerank_score"] = float(rr)
        item["retrieval_score_norm"] = float(rs)
        item["fused_score"] = float(alpha_retrieval * rs + alpha_rerank * rr)
        out.append(item)

    out.sort(key=lambda x: x["fused_score"], reverse=True)

    # Untouched tail: keep retriever order, mark explicitly.
    for c in tail:
        item = dict(c)
        item["rerank_score"] = None
        item["retrieval_score_norm"] = None
        item["fused_score"] = None
        out.append(item)

    return out


# ----------------------------------------------------------------------------
# CLI: end-to-end smoke test that requires GPU.
# ----------------------------------------------------------------------------


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Score a single (query, document) pair with "
        "Qwen3-Reranker-8B (GPU required).",
    )
    p.add_argument("--query", required=True, help="The user query.")
    p.add_argument(
        "--doc-text",
        required=True,
        help="The candidate document text to judge.",
    )
    p.add_argument(
        "--instruction",
        default=None,
        help="Override the default Swiss-legal instruction.",
    )
    p.add_argument("--model-id", default="Qwen/Qwen3-Reranker-8B")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--max-seq-len", type=int, default=4096)
    args = p.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("ERROR: torch is not installed.", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print(
            "ERROR: GPU required. No CUDA device detected. "
            "Run this on Colab / Kaggle / a local GPU box.",
            file=sys.stderr,
        )
        return 3

    rr = QwenReranker(
        model_id=args.model_id,
        dtype=args.dtype,
        max_seq_len=args.max_seq_len,
    )
    score = rr.score(args.query, args.doc_text, instruction=args.instruction)
    print(json.dumps({"query": args.query, "rerank_score": score}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
