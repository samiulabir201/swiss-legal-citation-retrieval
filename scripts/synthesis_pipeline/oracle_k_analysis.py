"""
oracle_k_analysis.py — Test whether adaptive K can rescue train F1.

For each train query, run v12's BM25 stage (same index, same TLF expansion),
then compute F1 at every K in 1..30 against gold (law-only, matching v12).
Report:
  - F1 at fixed K = 5, 10, 15, 20  (the notebook's tested settings)
  - F1 at *oracle K* (per-query best)
  - Correlation of oracle K with gold_count and query_length
  - A simple K predictor and its F1
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


ROOT = Path(__file__).resolve().parents[2]
ART  = ROOT / "drive_sync" / "omnilex_competition" / "retrieval_other_artifacts"
QT_FILE = ROOT / "drive_sync" / "omnilex_competition" / "retrieval_knowledge_base_query_translations" / "query_translations_trainval.json"

DEFAULT_TRAIN = ROOT / "data" / "train.csv"
DEFAULT_VAL   = ROOT / "data" / "val.csv"
DEFAULT_OUT   = ROOT / "artifacts" / "oracle_k_analysis.json"


# ─── v12's exact tokenization + TLF expansion ────────────────────────────
_TOKEN_RE = re.compile(r"[^\w\d]+", re.UNICODE)


def tokenise(t: str):
    if not t:
        return []
    return [w for w in _TOKEN_RE.split(t.lower().strip()) if len(w) >= 2 or w.isdigit()]


def build_query_tokens(q, de=None):
    en = tokenise(q)
    if not de:
        return en
    seen, c = set(), []
    for t in en + tokenise(de):
        if t not in seen:
            seen.add(t); c.append(t)
    return c


def enhance(tokens, bm25, tlf, ttc):
    base = list(dict.fromkeys(tokens))
    sc = {}
    for t in set(tokens):
        if t not in tlf:
            continue
        idf = bm25.idf.get(t, 0.0)
        if idf < 1.0:
            continue
        tot = ttc.get(t, 1)
        for a, cnt in tlf[t].items():
            sc[a] = sc.get(a, 0.0) + (cnt / tot) * idf
    for a, _ in sorted(sc.items(), key=lambda x: -x[1])[:5]:
        base.extend([a] * 5)
    return base


# ─── v12's gold expansion ────────────────────────────────────────────────
def build_parent_expansion_map(ldc):
    ci = defaultdict(list)
    es = set(ldc)
    for c in ldc:
        p = c.split()
        if len(p) >= 4 and p[0] == "Art." and p[2] == "Abs.":
            ci[(p[1], p[-1])].append(c)
    return ci, es


def expand_gold(cit, ci, es, clm, itr):
    cit = cit.strip()
    p = cit.split()
    if len(p) < 3 or p[0] != "Art.":
        return []
    m = clm.get(cit.lower())
    if m and m in itr:
        return [m]
    if "Abs." in cit:
        return []
    an, la = p[1], p[-1]
    ch = ci.get((an, la), [])
    if not ch:
        for (a2, l2), c2 in ci.items():
            if a2 == an and l2.lower() == la.lower():
                ch = c2; break
    if ch:
        r = [clm.get(c.lower()) for c in ch]; r = [x for x in r if x and x in itr]
        if r:
            return r
    return [m] if m and m in itr else []


def build_gold_set(gs, ci, es, clm, itr):
    raw = [c.strip() for c in str(gs or "").split(";") if c.strip()]
    lc = [c for c in raw if re.match(r"^Art\.\s+\d", c)]
    gi = set()
    for c in lc:
        for e in expand_gold(c, ci, es, clm, itr):
            gi.add(e)
    return gi


def f1(pred: set, gold: set):
    if not pred and not gold: return 1., 1., 1.
    if not pred or not gold: return 0., 0., 0.
    tp = len(pred & gold)
    p = tp / len(pred); r = tp / len(gold)
    return p, r, (2*p*r/(p+r) if p+r > 0 else 0.)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--val",   type=Path, default=DEFAULT_VAL)
    ap.add_argument("--sample-train", type=int, default=200,
                    help="Random sample of train queries (default 200; 0 = all 1139)")
    ap.add_argument("--max-k", type=int, default=30)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    t0 = time.time()
    print("[load] BM25 index ...", flush=True)
    with (ART / "bm25_v2_index.pkl").open("rb") as f:
        bm25 = pickle.load(f)
    with (ART / "bm25_v2_ids.pkl").open("rb") as f:
        bm25_ids = pickle.load(f)["citation_canon"]
    print(f"[load] {len(bm25_ids):,} corpus docs  ({time.time()-t0:.1f}s)")

    # Build clm + itr for gold expansion
    clm = {c.lower(): c for c in bm25_ids}
    itr = {c: i for i, c in enumerate(bm25_ids)}
    # laws_de.csv for parent expansion citations (use bm25_ids as the universe; close enough)
    ci, es = build_parent_expansion_map(bm25_ids)

    # Query translations (optional, but matches v12)
    qt = {}
    if QT_FILE.exists():
        qt = json.loads(QT_FILE.read_text(encoding="utf-8"))

    train = pd.read_csv(args.train)
    val = pd.read_csv(args.val)
    print(f"[data] train={len(train)} val={len(val)}")

    # TLF mined from train gold (same as v12)
    print("[tlf] mining token-law-frequency from train gold ...")
    tlf = defaultdict(Counter)
    for _, r in train.iterrows():
        gi = build_gold_set(r.get("gold_citations", ""), ci, es, clm, itr)
        if not gi:
            continue
        ab = {c.split()[-1].lower() for c in gi}
        tks = build_query_tokens(r["query"], qt.get(r["query_id"]))
        for t in set(tks):
            for a in ab:
                tlf[t][a] += 1
    ttc = {t: sum(c.values()) for t, c in tlf.items()}
    print(f"[tlf] {len(tlf):,} tokens with code stats")

    # Sample train
    if args.sample_train and args.sample_train < len(train):
        train_eval = train.sample(n=args.sample_train, random_state=42).reset_index(drop=True)
    else:
        train_eval = train
    print(f"[sample] using {len(train_eval)} train queries")

    # Combined queries (val + sampled train)
    combined = pd.concat([
        val.assign(split="val"),
        train_eval.assign(split="train"),
    ], ignore_index=True)

    # For each query: BM25 top-K (K=max_k), compute F1 at each k = 1..max_k
    print(f"\n[run] BM25 + F1-by-K for {len(combined)} queries ...")
    rows = []
    iter_ = tqdm(combined.iterrows(), total=len(combined)) if HAVE_TQDM else combined.iterrows()
    for _, r in iter_:
        qid = r["query_id"]; qtxt = r["query"]; split = r["split"]
        gold = build_gold_set(r.get("gold_citations", ""), ci, es, clm, itr)
        if not gold:
            rows.append({"qid": qid, "split": split, "gold_count": 0, "query_len": len(qtxt or ""),
                         "skip": "no_law_gold"})
            continue
        de = qt.get(qid, "")
        tks = build_query_tokens(qtxt, de)
        enh = enhance(tks, bm25, tlf, ttc)
        sc = bm25.get_scores(enh)
        order = sc.argsort()[::-1][:args.max_k]
        topk_cits = [bm25_ids[i] for i in order]
        # F1 at each K
        row = {
            "qid": qid, "split": split,
            "gold_count": len(gold),
            "query_len": len(qtxt or ""),
        }
        best_f1, best_k = 0.0, 1
        for k in range(1, args.max_k + 1):
            pred = set(topk_cits[:k])
            p, rec, ff = f1(pred, gold)
            row[f"f1@{k}"] = ff
            if ff > best_f1:
                best_f1, best_k = ff, k
        row["oracle_k"] = best_k
        row["oracle_f1"] = best_f1
        rows.append(row)

    df = pd.DataFrame(rows).dropna(subset=["gold_count"])
    df = df[df["gold_count"] > 0]

    # Aggregate
    def agg(df_split, label):
        d = df_split.copy()
        out = {"split": label, "n": len(d), "median_gold": d["gold_count"].median()}
        for k in [5, 8, 10, 12, 15, 20, 25, 30]:
            out[f"f1@K{k}"] = d[f"f1@{k}"].mean()
        out["oracle_f1"] = d["oracle_f1"].mean()
        out["median_oracle_k"] = d["oracle_k"].median()
        # K* vs gold count correlation
        out["pearson_K*_gold"] = d["oracle_k"].corr(d["gold_count"])
        out["pearson_K*_querylen"] = d["oracle_k"].corr(d["query_len"])
        # If we used K = gold_count (the right answer) — F1
        d["f1_at_gold_count"] = d.apply(
            lambda x: x.get(f"f1@{int(min(x['gold_count'], 30))}", 0.0), axis=1)
        out["f1_at_K=gold_count"] = d["f1_at_gold_count"].mean()
        return out

    summary = {
        "val":   agg(df[df["split"] == "val"], "val"),
        "train": agg(df[df["split"] == "train"], "train"),
    }
    # Combined honest avg
    combined_summary = {
        "honest_avg_oracle_f1": (summary["val"]["oracle_f1"] + summary["train"]["oracle_f1"]) / 2,
    }

    print("\n" + "=" * 70)
    print("ORACLE-K ANALYSIS — does adaptive K rescue train F1?")
    print("=" * 70)
    print(f"\n{'Metric':<28} {'VAL':>10} {'TRAIN':>10}")
    print("-" * 50)
    print(f"{'queries scored':<28} {summary['val']['n']:>10} {summary['train']['n']:>10}")
    print(f"{'median gold count':<28} {summary['val']['median_gold']:>10.0f} {summary['train']['median_gold']:>10.0f}")
    print(f"\n  Fixed K → mean F1:")
    for k in [5, 8, 10, 12, 15, 20, 25, 30]:
        print(f"  {('K='+str(k)):<28} {summary['val'][f'f1@K{k}']:>10.3f} {summary['train'][f'f1@K{k}']:>10.3f}")
    print(f"\n  {'K = gold_count (oracle K)':<28} {summary['val']['f1_at_K=gold_count']:>10.3f} {summary['train']['f1_at_K=gold_count']:>10.3f}")
    print(f"  {'best K per query (top of':<28}")
    print(f"  {'  fused-score curve)':<28} {summary['val']['oracle_f1']:>10.3f} {summary['train']['oracle_f1']:>10.3f}")
    print(f"  {'median K* per query':<28} {summary['val']['median_oracle_k']:>10.0f} {summary['train']['median_oracle_k']:>10.0f}")
    print(f"\n  K* correlations (Pearson r):")
    print(f"  {'  vs gold_count':<28} {summary['val']['pearson_K*_gold']:>10.3f} {summary['train']['pearson_K*_gold']:>10.3f}")
    print(f"  {'  vs query_length':<28} {summary['val']['pearson_K*_querylen']:>10.3f} {summary['train']['pearson_K*_querylen']:>10.3f}")

    print(f"\nHONEST avg (val + train, oracle_f1): {combined_summary['honest_avg_oracle_f1']:.3f}")
    print("=" * 70)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "summary": summary,
        "combined": combined_summary,
        "config": {
            "sample_train": args.sample_train,
            "max_k": args.max_k,
            "bm25_index": str(ART / "bm25_v2_index.pkl"),
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    # Also write per-query CSV for closer inspection
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"\n[done] -> {args.out}")
    print(f"[done] -> {args.out.with_suffix('.csv')}")
    print(f"[elapsed] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
