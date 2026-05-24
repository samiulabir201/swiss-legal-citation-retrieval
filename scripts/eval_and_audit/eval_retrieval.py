#!/usr/bin/env python
"""Evaluate the hybrid retriever on val/test queries.

Reads:
  - data/val.csv (or data/test.csv)             query_id, query, gold_citations
  - artifacts/embeddings/qwen3_8b_query_<split>.npy        (n, 4096) fp32
  - artifacts/embeddings/qwen3_8b_query_<split>_ids.parquet
  - artifacts/unified_retrieval.sqlite + chunked doc embeddings

Writes (recall mode, default):
  - artifacts/eval/<split>_candidate_sets.jsonl  per-query top-k with channel breakdown
  - artifacts/eval/<split>_summary.json          aggregated recall@K, candidate counts, timings
  - artifacts/eval/<split>_dropped_gold.csv      gold citations that fell outside top-k

Writes (--metric f1):
  - artifacts/eval/<split>_f1_summary.json       Macro F1 / Precision / Recall per K, best K
  - artifacts/eval/<split>_f1_per_query.csv      per-query (K, P, R, F1) at best K (when --per-query-csv)

Recall@K is computed by matching candidate `citation` strings to the `;`-separated
`gold_citations` column. The retriever returns up to top-k candidates per query; the
eval also reports recall at smaller cutoffs (50, 100, 200, 500, 1000) so we know
where the cliff sits.

The competition metric is **Macro F1**. Use --metric f1 to sweep K over the
practical range and pick the operating point that maximises Macro F1. The
optional --apply-granularity-post-filter expands bare-article predictions
(``Art. 78 BV``) into their corpus paragraph-children
(``{Art. 78 Abs. 1 BV, ...}``) before scoring -- this makes predictions
comparable to gold at corpus granularity.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hybrid_retrieve import HybridRetriever  # noqa: E402

DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
EMB_DIR = ART_DIR / "embeddings"
EVAL_DIR = ART_DIR / "eval"
DEFAULT_SQLITE = ART_DIR / "unified_retrieval.sqlite"
DEFAULT_MANIFEST = EMB_DIR / "qwen3_8b_unified_manifest.parquet"

CUTOFFS = (50, 100, 200, 500, 1000)
F1_K_SWEEP = (5, 7, 10, 13, 15, 20, 25, 30, 50, 75, 100)


def normalize_citation(c: str) -> str:
    """Reduce a citation to a comparison-friendly canonical form."""
    return re.sub(r"\s+", " ", c).strip()


def parse_gold(s: str) -> list[str]:
    if not isinstance(s, str):
        return []
    return [normalize_citation(c) for c in s.split(";") if c.strip()]


# ---------------------------------------------------------------------------
# F1 helpers
# ---------------------------------------------------------------------------

def _set_prf(pred: set[str], gold: set[str]) -> tuple[float, float, float]:
    """Per-query (precision, recall, F1) by set comparison.

    Convention:
      - empty gold and empty pred -> (1.0, 1.0, 1.0)
      - empty gold but non-empty pred -> (0.0, 1.0, 0.0)
      - non-empty gold and empty pred -> (1.0, 0.0, 0.0)
    """
    if not gold and not pred:
        return 1.0, 1.0, 1.0
    if not gold:
        return 0.0, 1.0, 0.0
    if not pred:
        return 1.0, 0.0, 0.0
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def _micro_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def evaluate(
    *,
    split_csv: Path,
    query_emb_path: Path | None,
    query_ids_path: Path | None,
    retriever: HybridRetriever,
    top_k: int,
    bm25_budget: int,
    vector_budget: int,
    statute_budget: int,
    case_budget: int,
    rrf_k: int,
    authority_alpha: float,
    out_dir: Path,
    split_name: str,
    limit: int = 0,
):
    df = pd.read_csv(split_csv)
    df["gold_set"] = df["gold_citations"].apply(parse_gold)

    q_emb = None
    if query_emb_path and query_emb_path.exists():
        q_emb = np.load(query_emb_path)
        if query_ids_path and query_ids_path.exists():
            ids_df = pd.read_parquet(query_ids_path)
            order = ids_df["query_id"].tolist()
            df = df.set_index("query_id").loc[order].reset_index()
            print(f"[align] reordered {len(df)} queries to match query embedding manifest", flush=True)
        if len(q_emb) != len(df):
            raise ValueError(f"query embedding count {len(q_emb)} != csv rows {len(df)}")
    else:
        print("[warn] no query embedding file - running BM25 + anchors only", file=sys.stderr)

    if limit:
        df = df.head(limit).reset_index(drop=True)
        if q_emb is not None:
            q_emb = q_emb[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    cand_path = out_dir / f"{split_name}_candidate_sets.jsonl"
    drop_path = out_dir / f"{split_name}_dropped_gold.csv"
    summary_path = out_dir / f"{split_name}_summary.json"

    aggregate = {
        "n_queries": len(df),
        "cutoffs": list(CUTOFFS) + [top_k],
        "recall_at_k": defaultdict(list),
        "candidate_counts": [],
        "channel_counts": Counter(),
        "channel_sums": Counter(),
        "channel_timings_ms": Counter(),
    }
    dropped_rows: list[dict] = []

    t_total = time.time()
    with cand_path.open("w", encoding="utf-8") as fout:
        for i, row in df.iterrows():
            qid = row["query_id"]
            query = row["query"]
            gold = row["gold_set"]
            qe = None if q_emb is None else q_emb[i]

            r = retriever.retrieve(
                query,
                query_embedding=qe,
                top_k=top_k,
                channel_budgets={
                    "bm25": bm25_budget, "vector": vector_budget,
                    "statute": statute_budget, "case": case_budget,
                },
                rrf_k=rrf_k,
                authority_alpha=authority_alpha,
                verbose=False,
            )
            candidates = r["candidates"]
            cand_citations = [normalize_citation(c.get("citation", "")) for c in candidates]
            cand_set = set(cand_citations)

            recalls: dict[int, float] = {}
            recovered_at_k: dict[int, set] = {}
            for k in list(CUTOFFS) + [top_k]:
                truncated = set(cand_citations[:k])
                if not gold:
                    recalls[k] = 1.0
                else:
                    recalls[k] = len(truncated & set(gold)) / len(gold)
                recovered_at_k[k] = truncated & set(gold)
                aggregate["recall_at_k"][k].append(recalls[k])

            aggregate["candidate_counts"].append(len(candidates))
            for ch, n in r["channel_counts"].items():
                aggregate["channel_counts"][ch] += 1 if n else 0
                aggregate["channel_sums"][ch] += n
            for ch, t_s in r["channel_timings_s"].items():
                aggregate["channel_timings_ms"][ch] += int(t_s * 1000)

            for g in gold:
                if g not in cand_set:
                    dropped_rows.append({"query_id": qid, "gold_citation": g})

            fout.write(json.dumps({
                "query_id": qid,
                "query": query[:300],
                "anchors": r["anchors"],
                "channel_counts": r["channel_counts"],
                "channel_timings_s": r["channel_timings_s"],
                "n_candidates": len(candidates),
                "recall_at_k": recalls,
                "gold_count": len(gold),
                "recovered_count_at_top_k": len(recovered_at_k[top_k]),
                "candidates": [
                    {
                        "rank": j + 1,
                        "doc_id": c["doc_id"],
                        "family": c["family"],
                        "citation": c["citation"],
                        "fused_score": c["fused_score"],
                        "channel_ranks": c.get("channel_ranks", {}),
                    }
                    for j, c in enumerate(candidates)
                ],
            }, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0 or (i + 1) == len(df):
                rec1k = np.mean(aggregate["recall_at_k"][top_k]) if aggregate["recall_at_k"][top_k] else 0
                print(f"  query {i+1}/{len(df)}  recall@{top_k}={rec1k:.3f}  "
                      f"cands={len(candidates)}  gold={len(gold)}", flush=True)

    elapsed = time.time() - t_total
    summary = {
        "split": split_name,
        "n_queries": len(df),
        "elapsed_s": round(elapsed, 1),
        "candidate_count_mean": float(np.mean(aggregate["candidate_counts"])) if aggregate["candidate_counts"] else 0.0,
        "candidate_count_median": float(np.median(aggregate["candidate_counts"])) if aggregate["candidate_counts"] else 0.0,
        "recall_at_k": {k: float(np.mean(v)) for k, v in aggregate["recall_at_k"].items()},
        "channel_query_count": dict(aggregate["channel_counts"]),
        "channel_total_hits": dict(aggregate["channel_sums"]),
        "channel_avg_time_ms": {k: int(v / max(len(df), 1)) for k, v in aggregate["channel_timings_ms"].items()},
        "config": {
            "top_k": top_k,
            "rrf_k": rrf_k,
            "authority_alpha": authority_alpha,
            "bm25_budget": bm25_budget,
            "vector_budget": vector_budget,
            "statute_budget": statute_budget,
            "case_budget": case_budget,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(dropped_rows).to_csv(drop_path, index=False)

    print()
    print(f"[done] {split_name}: {len(df)} queries in {elapsed/60:.1f} min")
    print(f"[done] candidates per query: mean={summary['candidate_count_mean']:.0f} "
          f"median={summary['candidate_count_median']:.0f}")
    print("[done] recall@k:")
    for k in summary["recall_at_k"]:
        print(f"   recall@{k:>4d} = {summary['recall_at_k'][k]:.4f}")
    print(f"[done] candidate sets: {cand_path}")
    print(f"[done] dropped gold:   {drop_path}")
    print(f"[done] summary:        {summary_path}")
    return summary


def evaluate_f1(
    *,
    split_csv: Path,
    query_emb_path: Path | None,
    query_ids_path: Path | None,
    retriever: HybridRetriever,
    bm25_budget: int,
    vector_budget: int,
    statute_budget: int,
    case_budget: int,
    rrf_k: int,
    authority_alpha: float,
    out_dir: Path,
    split_name: str,
    apply_granularity: bool,
    sqlite_path: Path,
    per_query_csv: Path | None,
    k_sweep: tuple[int, ...] = F1_K_SWEEP,
    limit: int = 0,
    pool_top_k: int | None = None,
    rerank: bool = False,
    reranker=None,
    rerank_top_n: int = 200,
    rerank_batch_size: int = 8,
    llm_judge: bool = False,
    judge=None,
    judge_auto_yes_thresh: float = 0.55,
    judge_auto_no_thresh: float = 0.25,
    judge_batch_size: int = 4,
):
    """F1-tuned K-scan evaluation. Macro F1 is the competition metric.

    For each query, we materialise the top max(k_sweep) candidate citations
    (post-granularity-filter if requested), then for each K in the sweep we
    truncate to the first K predictions and compute set-based P/R/F1 against
    the gold set. Macro-aggregating over queries gives the headline number.
    """
    if pool_top_k is None:
        # Need to pull at least max(K_sweep), but pre-granularity-filter the
        # paragraph expansion can shrink the unique-set, so pull a bit more.
        pool_top_k = max(k_sweep) * 4

    df = pd.read_csv(split_csv)
    df["gold_set"] = df["gold_citations"].apply(parse_gold)

    q_emb = None
    if query_emb_path and query_emb_path.exists():
        q_emb = np.load(query_emb_path)
        if query_ids_path and query_ids_path.exists():
            ids_df = pd.read_parquet(query_ids_path)
            order = ids_df["query_id"].tolist()
            df = df.set_index("query_id").loc[order].reset_index()
            print(f"[align] reordered {len(df)} queries to match query embedding manifest", flush=True)
        if len(q_emb) != len(df):
            raise ValueError(f"query embedding count {len(q_emb)} != csv rows {len(df)}")
    else:
        print("[warn] no query embedding file - running BM25 + anchors only", file=sys.stderr)

    if limit:
        df = df.head(limit).reset_index(drop=True)
        if q_emb is not None:
            q_emb = q_emb[:limit]

    # Optional granularity post-filter connection.
    gran_conn: sqlite3.Connection | None = None
    if apply_granularity:
        if not sqlite_path.exists():
            raise FileNotFoundError(
                f"--apply-granularity-post-filter requires sqlite at {sqlite_path}"
            )
        gran_conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        # Lazy import: granularity_resolver pulls sibling scripts.
        from granularity_resolver import apply_granularity_post_filter
    else:
        apply_granularity_post_filter = None  # type: ignore[assignment]

    out_dir.mkdir(parents=True, exist_ok=True)
    f1_summary_path = out_dir / f"{split_name}_f1_summary.json"

    # Per-K accumulators.
    per_k_p: dict[int, list[float]] = {k: [] for k in k_sweep}
    per_k_r: dict[int, list[float]] = {k: [] for k in k_sweep}
    per_k_f: dict[int, list[float]] = {k: [] for k in k_sweep}
    # Per-K micro counters (TP/FP/FN).
    per_k_micro: dict[int, list[int]] = {k: [0, 0, 0] for k in k_sweep}

    # For per-query CSV at best K we keep raw lists in memory.
    per_query_records: list[dict] = []

    t_total = time.time()
    for i, row in df.iterrows():
        qid = row["query_id"]
        query = row["query"]
        gold = set(row["gold_set"])
        qe = None if q_emb is None else q_emb[i]

        r = retriever.retrieve(
            query,
            query_embedding=qe,
            top_k=pool_top_k,
            channel_budgets={
                "bm25": bm25_budget, "vector": vector_budget,
                "statute": statute_budget, "case": case_budget,
            },
            rrf_k=rrf_k,
            authority_alpha=authority_alpha,
            verbose=False,
            rerank=rerank,
            reranker=reranker,
            rerank_top_n=rerank_top_n,
            rerank_batch_size=rerank_batch_size,
            llm_judge=llm_judge,
            judge=judge,
            judge_auto_yes_thresh=judge_auto_yes_thresh,
            judge_auto_no_thresh=judge_auto_no_thresh,
            judge_batch_size=judge_batch_size,
            apply_granularity_filter=False,
        )
        candidates = r["candidates"]
        cand_citations = [normalize_citation(c.get("citation", "")) for c in candidates]

        if apply_granularity and gran_conn is not None:
            cand_citations = apply_granularity_post_filter(cand_citations, gran_conn)
        else:
            # Even without expansion, dedupe while preserving order.
            seen: set[str] = set()
            deduped: list[str] = []
            for c in cand_citations:
                if c and c not in seen:
                    seen.add(c)
                    deduped.append(c)
            cand_citations = deduped

        record = {"query_id": qid, "gold_count": len(gold), "n_pred_pool": len(cand_citations)}
        for k in k_sweep:
            top = set(cand_citations[:k])
            p, rec, f1 = _set_prf(top, gold)
            per_k_p[k].append(p)
            per_k_r[k].append(rec)
            per_k_f[k].append(f1)
            tp = len(top & gold)
            fp = len(top - gold)
            fn = len(gold - top)
            per_k_micro[k][0] += tp
            per_k_micro[k][1] += fp
            per_k_micro[k][2] += fn
            record[f"P@{k}"] = p
            record[f"R@{k}"] = rec
            record[f"F1@{k}"] = f1
        per_query_records.append(record)

        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            best_k = max(k_sweep, key=lambda k: float(np.mean(per_k_f[k])))
            print(f"  query {i+1}/{len(df)}  "
                  f"best-so-far K={best_k}  "
                  f"macroF1={np.mean(per_k_f[best_k]):.4f}", flush=True)

    elapsed = time.time() - t_total

    if gran_conn is not None:
        gran_conn.close()

    # Aggregate.
    per_k_summary = {}
    for k in k_sweep:
        macro_p = float(np.mean(per_k_p[k])) if per_k_p[k] else 0.0
        macro_r = float(np.mean(per_k_r[k])) if per_k_r[k] else 0.0
        macro_f = float(np.mean(per_k_f[k])) if per_k_f[k] else 0.0
        tp, fp, fn = per_k_micro[k]
        micro_p, micro_r, micro_f = _micro_prf(tp, fp, fn)
        per_k_summary[k] = {
            "macro_precision": macro_p,
            "macro_recall": macro_r,
            "macro_f1": macro_f,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f,
            "tp": tp, "fp": fp, "fn": fn,
        }
    best_k = max(k_sweep, key=lambda k: per_k_summary[k]["macro_f1"])

    summary = {
        "split": split_name,
        "metric": "macro_f1",
        "n_queries": len(df),
        "elapsed_s": round(elapsed, 1),
        "k_sweep": list(k_sweep),
        "apply_granularity_post_filter": apply_granularity,
        "best_k": best_k,
        "best_macro_f1": per_k_summary[best_k]["macro_f1"],
        "best_macro_precision": per_k_summary[best_k]["macro_precision"],
        "best_macro_recall": per_k_summary[best_k]["macro_recall"],
        "best_micro_f1": per_k_summary[best_k]["micro_f1"],
        "per_k": {str(k): per_k_summary[k] for k in k_sweep},
        "config": {
            "rrf_k": rrf_k,
            "authority_alpha": authority_alpha,
            "bm25_budget": bm25_budget,
            "vector_budget": vector_budget,
            "statute_budget": statute_budget,
            "case_budget": case_budget,
            "pool_top_k": pool_top_k,
        },
    }
    f1_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if per_query_csv is not None:
        per_query_csv.parent.mkdir(parents=True, exist_ok=True)
        with per_query_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "query_id", "K", "precision", "recall", "f1", "gold_count", "n_pred_pool",
            ])
            for rec in per_query_records:
                writer.writerow([
                    rec["query_id"],
                    best_k,
                    f"{rec[f'P@{best_k}']:.4f}",
                    f"{rec[f'R@{best_k}']:.4f}",
                    f"{rec[f'F1@{best_k}']:.4f}",
                    rec["gold_count"],
                    rec["n_pred_pool"],
                ])

    # Pretty print.
    print()
    print(f"[done-f1] {split_name}: {len(df)} queries in {elapsed/60:.1f} min  "
          f"(granularity_filter={apply_granularity})")
    print(f"[done-f1] sweep K -> macroF1 / macroP / macroR / microF1:")
    for k in k_sweep:
        s = per_k_summary[k]
        marker = "  <-- best" if k == best_k else ""
        print(f"    K={k:>3d}  F1={s['macro_f1']:.4f}  "
              f"P={s['macro_precision']:.4f}  R={s['macro_recall']:.4f}  "
              f"microF1={s['micro_f1']:.4f}{marker}")
    print(f"[done-f1] BEST: K={best_k}  "
          f"MacroF1={per_k_summary[best_k]['macro_f1']:.4f}  "
          f"P={per_k_summary[best_k]['macro_precision']:.4f}  "
          f"R={per_k_summary[best_k]['macro_recall']:.4f}  "
          f"microF1={per_k_summary[best_k]['micro_f1']:.4f}")
    print(f"[done-f1] summary: {f1_summary_path}")
    if per_query_csv is not None:
        print(f"[done-f1] per-query CSV: {per_query_csv}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--csv", type=Path, default=None,
                    help="Override path to the split CSV.")
    ap.add_argument("--gold-csv", type=Path, default=None,
                    help="Alias for --csv (matches the F1-mode invocation in the spec).")
    ap.add_argument("--query-embedding", type=Path, default=None,
                    help="Path to qwen3_8b_query_<split>.npy. Defaults based on --split.")
    ap.add_argument("--query-ids", type=Path, default=None,
                    help="Optional parquet with query_id ordering matching the embedding rows.")
    ap.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--embeddings-dir", type=Path, default=EMB_DIR)
    ap.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--authority-alpha", type=float, default=0.15)
    ap.add_argument("--bm25-budget", type=int, default=800)
    ap.add_argument("--vector-budget", type=int, default=800)
    ap.add_argument("--statute-budget", type=int, default=400)
    ap.add_argument("--case-budget", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap to first N queries for smoke runs. 0 = full split.")
    ap.add_argument("--metric", default="recall", choices=["recall", "f1"],
                    help="recall (legacy: recall@K cutoffs) or f1 (Macro F1 K-scan over the "
                         "competition metric). Default: recall.")
    ap.add_argument("--apply-granularity-post-filter",
                    dest="apply_granularity_post_filter",
                    action="store_true", default=True,
                    help="(--metric f1 only) Expand bare-article predictions to corpus "
                         "paragraph-children before scoring. Default: True.")
    ap.add_argument("--no-apply-granularity-post-filter",
                    dest="apply_granularity_post_filter",
                    action="store_false",
                    help="Disable the granularity post-filter for an apples-to-apples "
                         "comparison.")
    ap.add_argument("--per-query-csv", type=Path, default=None,
                    help="(--metric f1 only) Write per-query (K, P, R, F1) at best K to this CSV.")
    # Reranker / LLM-judge passthrough (GPU required at runtime).
    ap.add_argument("--enable-rerank", action="store_true",
                    help="Run Qwen3-Reranker-8B over fused top-N. GPU REQUIRED.")
    ap.add_argument("--rerank-top-n", type=int, default=200)
    ap.add_argument("--rerank-batch-size", type=int, default=8)
    ap.add_argument("--enable-judge", action="store_true",
                    help="Run Qwen3-8B as LLM-judge on borderline candidates. GPU REQUIRED.")
    ap.add_argument("--judge-auto-yes", type=float, default=0.55)
    ap.add_argument("--judge-auto-no", type=float, default=0.25)
    ap.add_argument("--judge-batch-size", type=int, default=4)
    args = ap.parse_args()

    # Lazy-load reranker / judge if requested (matches hybrid_retrieve.py pattern).
    reranker_obj = None
    judge_obj = None
    if args.enable_rerank or args.enable_judge:
        try:
            import torch
        except ImportError:
            print("ERROR: --enable-rerank/--enable-judge require torch (GPU stack).",
                  file=sys.stderr)
            return 3
        if not torch.cuda.is_available():
            print("ERROR: GPU required for --enable-rerank / --enable-judge "
                  "(no CUDA device detected).", file=sys.stderr)
            return 3
        if args.enable_rerank:
            from rerank_qwen3 import QwenReranker  # lazy
            print("[rerank] loading Qwen3-Reranker-8B (this can take a few minutes)...",
                  file=sys.stderr)
            reranker_obj = QwenReranker(dtype="bf16", max_seq_len=4096)
        if args.enable_judge:
            from llm_judge_qwen3 import QwenJudge  # lazy
            print("[judge] loading Qwen3-8B judge...", file=sys.stderr)
            judge_obj = QwenJudge(model_id="Qwen/Qwen3-8B", dtype="bf16")

    csv_path = args.csv or args.gold_csv or (DATA_DIR / f"{args.split}.csv")
    if not csv_path.exists():
        print(f"ERROR: split csv not found: {csv_path}", file=sys.stderr)
        return 2

    qemb_path = args.query_embedding or (EMB_DIR / f"qwen3_8b_query_{args.split}.npy")
    qids_path = args.query_ids or (EMB_DIR / f"qwen3_8b_query_{args.split}_ids.parquet")

    rt = HybridRetriever(args.sqlite, args.manifest, args.embeddings_dir)

    if args.metric == "f1":
        evaluate_f1(
            split_csv=csv_path,
            query_emb_path=qemb_path if qemb_path.exists() else None,
            query_ids_path=qids_path if qids_path.exists() else None,
            retriever=rt,
            bm25_budget=args.bm25_budget,
            vector_budget=args.vector_budget,
            statute_budget=args.statute_budget,
            case_budget=args.case_budget,
            rrf_k=args.rrf_k,
            authority_alpha=args.authority_alpha,
            out_dir=args.out_dir,
            split_name=args.split,
            apply_granularity=args.apply_granularity_post_filter,
            sqlite_path=args.sqlite,
            per_query_csv=args.per_query_csv,
            limit=args.limit,
            rerank=args.enable_rerank,
            reranker=reranker_obj,
            rerank_top_n=args.rerank_top_n,
            rerank_batch_size=args.rerank_batch_size,
            llm_judge=args.enable_judge,
            judge=judge_obj,
            judge_auto_yes_thresh=args.judge_auto_yes,
            judge_auto_no_thresh=args.judge_auto_no,
            judge_batch_size=args.judge_batch_size,
        )
    else:
        evaluate(
            split_csv=csv_path,
            query_emb_path=qemb_path if qemb_path.exists() else None,
            query_ids_path=qids_path if qids_path.exists() else None,
            retriever=rt,
            top_k=args.top_k,
            bm25_budget=args.bm25_budget,
            vector_budget=args.vector_budget,
            statute_budget=args.statute_budget,
            case_budget=args.case_budget,
            rrf_k=args.rrf_k,
            authority_alpha=args.authority_alpha,
            out_dir=args.out_dir,
            split_name=args.split,
            limit=args.limit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
