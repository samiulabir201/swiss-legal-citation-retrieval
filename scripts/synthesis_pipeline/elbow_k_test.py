"""
elbow_k_test.py — Estimate adaptive-K predictor accuracy without GPU.

Tests 6 elbow-detection methods over the BM25-score curve (the strongest local
proxy for v12's fused score, since fused = 0.7·norm(BM25) + 0.3·reranker and
the BM25 component carries most of the rank-order information that drives K).

For each query (val + sampled train), we:
  1. Run v12's BM25 top-30 + TLF expansion (same as oracle_k_analysis.py)
  2. Apply each elbow method to the score curve → K_pred
  3. Compute F1 at K_pred against expanded law-only gold
  4. Aggregate per-method per-split

Reported per method per split:
  - mean F1, MAE of K_pred vs oracle K, Pearson r(K_pred, K_oracle), Pearson r(K_pred, gold_count)

Comparison baselines (from oracle_k_analysis.py prior run):
                       VAL    TRAIN
  best fixed K        0.714  0.327   (K=12 for val, K=5 for train)
  K = gold_count      0.785  0.450
  oracle K (per-q)    0.864  0.565
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
QT_FILE = (ROOT / "drive_sync" / "omnilex_competition"
           / "retrieval_knowledge_base_query_translations"
           / "query_translations_trainval.json")
DEFAULT_TRAIN = ROOT / "data" / "train.csv"
DEFAULT_VAL   = ROOT / "data" / "val.csv"
DEFAULT_OUT   = ROOT / "artifacts" / "elbow_k_test.json"


# ─── v12 helpers (copied to keep script standalone) ───────────────────────
_TOKEN_RE = re.compile(r"[^\w\d]+", re.UNICODE)

def tokenise(t):
    if not t: return []
    return [w for w in _TOKEN_RE.split(t.lower().strip()) if len(w) >= 2 or w.isdigit()]

def build_query_tokens(q, de=None):
    en = tokenise(q)
    if not de: return en
    seen, c = set(), []
    for t in en + tokenise(de):
        if t not in seen:
            seen.add(t); c.append(t)
    return c

def enhance(tokens, bm25, tlf, ttc):
    base = list(dict.fromkeys(tokens))
    sc = {}
    for t in set(tokens):
        if t not in tlf: continue
        idf = bm25.idf.get(t, 0.0)
        if idf < 1.0: continue
        tot = ttc.get(t, 1)
        for a, cnt in tlf[t].items():
            sc[a] = sc.get(a, 0.0) + (cnt / tot) * idf
    for a, _ in sorted(sc.items(), key=lambda x: -x[1])[:5]:
        base.extend([a] * 5)
    return base

def build_parent_expansion_map(ldc):
    ci = defaultdict(list); es = set(ldc)
    for c in ldc:
        p = c.split()
        if len(p) >= 4 and p[0] == "Art." and p[2] == "Abs.":
            ci[(p[1], p[-1])].append(c)
    return ci, es

def expand_gold(cit, ci, es, clm, itr):
    cit = cit.strip(); p = cit.split()
    if len(p) < 3 or p[0] != "Art.": return []
    m = clm.get(cit.lower())
    if m and m in itr: return [m]
    if "Abs." in cit: return []
    an, la = p[1], p[-1]
    ch = ci.get((an, la), [])
    if not ch:
        for (a2, l2), c2 in ci.items():
            if a2 == an and l2.lower() == la.lower():
                ch = c2; break
    if ch:
        r = [clm.get(c.lower()) for c in ch]; r = [x for x in r if x and x in itr]
        if r: return r
    return [m] if m and m in itr else []

def build_gold_set(gs, ci, es, clm, itr):
    raw = [c.strip() for c in str(gs or "").split(";") if c.strip()]
    lc = [c for c in raw if re.match(r"^Art\.\s+\d", c)]
    gi = set()
    for c in lc:
        for e in expand_gold(c, ci, es, clm, itr):
            gi.add(e)
    return gi

def f1(pred, gold):
    if not pred and not gold: return 1., 1., 1.
    if not pred or not gold: return 0., 0., 0.
    tp = len(pred & gold)
    p = tp / len(pred); r = tp / len(gold)
    return p, r, (2*p*r/(p+r) if p+r > 0 else 0.)


# ─── Elbow detection methods ──────────────────────────────────────────────

def elbow_largest_gap(scores, k_min=1, k_max=30):
    """K at the largest first-difference drop between consecutive ranks."""
    s = np.asarray(scores)[:k_max + 1]
    if len(s) < 2: return k_min
    gaps = s[:-1] - s[1:]
    k_pred = int(np.argmax(gaps)) + 1
    return max(k_min, min(k_pred, k_max))


def elbow_largest_gap_smoothed(scores, k_min=1, k_max=30, window=3):
    """Smooth with rolling mean, then largest gap."""
    s = np.asarray(scores)[:k_max + 2].astype(float)
    if len(s) < window + 1: return elbow_largest_gap(scores, k_min, k_max)
    kernel = np.ones(window) / window
    smoothed = np.convolve(s, kernel, mode="valid")
    if len(smoothed) < 2: return k_min
    gaps = smoothed[:-1] - smoothed[1:]
    k_pred = int(np.argmax(gaps)) + 1
    return max(k_min, min(k_pred, k_max))


def elbow_kneedle(scores, k_min=1, k_max=30):
    """Standard kneedle: max chord-distance for a normalized descending curve."""
    s = np.asarray(scores)[:k_max + 1].astype(float)
    if len(s) < 3: return k_min
    x = np.linspace(0, 1, len(s))
    rng = s.max() - s.min()
    y = (s - s.min()) / (rng + 1e-9)
    diff = (1 - x) - y   # for convex descending curve: positive bulge below chord
    k_pred = int(np.argmax(diff))
    if k_pred == 0: k_pred = 1
    return max(k_min, min(k_pred, k_max))


def elbow_second_derivative(scores, k_min=1, k_max=30):
    """K at max second-derivative (sharpest curvature)."""
    s = np.asarray(scores)[:k_max + 1].astype(float)
    if len(s) < 3: return k_min
    d2 = s[:-2] - 2 * s[1:-1] + s[2:]
    k_pred = int(np.argmax(d2)) + 1
    return max(k_min, min(k_pred, k_max))


def elbow_threshold_relative(scores, alpha=0.5, k_min=1, k_max=30):
    """K = count of scores >= alpha · top_score."""
    s = np.asarray(scores)[:k_max + 1]
    if len(s) < 1: return k_min
    thresh = alpha * s[0]
    k_pred = int((s >= thresh).sum())
    return max(k_min, min(k_pred, k_max))


def elbow_ratio_dropoff(scores, ratio=0.7, k_min=1, k_max=30):
    """K = first i where score[i+1] / score[i] < ratio."""
    s = np.asarray(scores)[:k_max + 1]
    if len(s) < 2: return k_min
    for i in range(len(s) - 1):
        if s[i] <= 0: continue
        if s[i + 1] / s[i] < ratio:
            return max(k_min, min(i + 1, k_max))
    return k_max


METHODS = {
    "largest_gap":            lambda s: elbow_largest_gap(s),
    "largest_gap_smoothed3":  lambda s: elbow_largest_gap_smoothed(s, window=3),
    "kneedle":                lambda s: elbow_kneedle(s),
    "second_deriv":           lambda s: elbow_second_derivative(s),
    "threshold_alpha0.5":     lambda s: elbow_threshold_relative(s, alpha=0.5),
    "threshold_alpha0.3":     lambda s: elbow_threshold_relative(s, alpha=0.3),
    "ratio_dropoff_0.7":      lambda s: elbow_ratio_dropoff(s, ratio=0.7),
    "ratio_dropoff_0.5":      lambda s: elbow_ratio_dropoff(s, ratio=0.5),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--val",   type=Path, default=DEFAULT_VAL)
    ap.add_argument("--sample-train", type=int, default=200)
    ap.add_argument("--max-k", type=int, default=30)
    ap.add_argument("--out",   type=Path, default=DEFAULT_OUT)
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

    clm = {c.lower(): c for c in bm25_ids}
    itr = {c: i for i, c in enumerate(bm25_ids)}
    ci, es = build_parent_expansion_map(bm25_ids)

    qt = {}
    if QT_FILE.exists():
        qt = json.loads(QT_FILE.read_text(encoding="utf-8"))

    train = pd.read_csv(args.train)
    val = pd.read_csv(args.val)
    print(f"[data] train={len(train)} val={len(val)}")

    print("[tlf] mining ...")
    tlf = defaultdict(Counter)
    for _, r in train.iterrows():
        gi = build_gold_set(r.get("gold_citations", ""), ci, es, clm, itr)
        if not gi: continue
        ab = {c.split()[-1].lower() for c in gi}
        tks = build_query_tokens(r["query"], qt.get(r["query_id"]))
        for t in set(tks):
            for a in ab: tlf[t][a] += 1
    ttc = {t: sum(c.values()) for t, c in tlf.items()}
    print(f"[tlf] {len(tlf):,} tokens")

    if args.sample_train and args.sample_train < len(train):
        train_eval = train.sample(n=args.sample_train, random_state=42).reset_index(drop=True)
    else:
        train_eval = train
    combined = pd.concat([
        val.assign(split="val"),
        train_eval.assign(split="train"),
    ], ignore_index=True)
    print(f"[run] BM25 + elbow methods for {len(combined)} queries ...")

    rows = []
    iter_ = tqdm(combined.iterrows(), total=len(combined)) if HAVE_TQDM else combined.iterrows()
    for _, r in iter_:
        qid = r["query_id"]; qtxt = r["query"]; split = r["split"]
        gold = build_gold_set(r.get("gold_citations", ""), ci, es, clm, itr)
        if not gold: continue
        de = qt.get(qid, "")
        tks = build_query_tokens(qtxt, de)
        enh = enhance(tks, bm25, tlf, ttc)
        sc_all = bm25.get_scores(enh)
        order = sc_all.argsort()[::-1][:args.max_k]
        topk_cits = [bm25_ids[i] for i in order]
        topk_scores = sc_all[order]

        row = {
            "qid": qid, "split": split,
            "gold_count": len(gold),
            "query_len": len(qtxt or ""),
        }
        # F1 at every K
        best_f1, best_k = 0.0, 1
        for k in range(1, args.max_k + 1):
            ff = f1(set(topk_cits[:k]), gold)[2]
            row[f"f1@{k}"] = ff
            if ff > best_f1:
                best_f1, best_k = ff, k
        row["oracle_k"] = best_k
        row["oracle_f1"] = best_f1
        # Each method: K_pred + F1 at K_pred
        for name, fn in METHODS.items():
            k_pred = fn(topk_scores)
            row[f"K_{name}"] = k_pred
            row[f"F1_{name}"] = f1(set(topk_cits[:k_pred]), gold)[2]
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[df["gold_count"] > 0]

    def agg(d, label):
        out = {"split": label, "n": len(d),
               "median_gold": int(d["gold_count"].median()),
               "median_oracle_k": int(d["oracle_k"].median()),
               "oracle_f1": float(d["oracle_f1"].mean())}
        # Baselines: best fixed K seen in oracle run
        for k in (5, 10, 12, 15):
            out[f"f1@K{k}"] = float(d[f"f1@{k}"].mean())
        # K = gold_count baseline
        out["f1_at_K=gold_count"] = float(
            d.apply(lambda x: x.get(f"f1@{int(min(x['gold_count'], 30))}", 0.0), axis=1).mean()
        )
        # Per-method stats
        method_stats = {}
        for m in METHODS:
            k_pred = d[f"K_{m}"]
            method_stats[m] = {
                "mean_F1": float(d[f"F1_{m}"].mean()),
                "MAE_vs_oracle_K": float((k_pred - d["oracle_k"]).abs().mean()),
                "MAE_vs_gold_count": float((k_pred - d["gold_count"]).abs().mean()),
                "median_K_pred": float(k_pred.median()),
                "pearson_r_vs_oracle_K": float(k_pred.corr(d["oracle_k"])),
                "pearson_r_vs_gold_count": float(k_pred.corr(d["gold_count"])),
            }
        out["methods"] = method_stats
        return out

    summary = {
        "val":   agg(df[df["split"] == "val"], "val"),
        "train": agg(df[df["split"] == "train"], "train"),
    }

    # ── Print report
    bar = "=" * 78
    print("\n" + bar)
    print("ELBOW-K PREDICTOR TEST — BM25 score curves only (no reranker)")
    print(bar)
    print(f"\n{'':<32}{'VAL':>14}{'TRAIN':>14}")
    print(f"{'queries scored':<32}{summary['val']['n']:>14}{summary['train']['n']:>14}")
    print(f"{'median gold count':<32}{summary['val']['median_gold']:>14}{summary['train']['median_gold']:>14}")
    print(f"{'median oracle K':<32}{summary['val']['median_oracle_k']:>14}{summary['train']['median_oracle_k']:>14}")
    print(f"\n  Baselines (mean F1):")
    print(f"  {'K=5':<30}{summary['val']['f1@K5']:>14.3f}{summary['train']['f1@K5']:>14.3f}")
    print(f"  {'K=10':<30}{summary['val']['f1@K10']:>14.3f}{summary['train']['f1@K10']:>14.3f}")
    print(f"  {'K=12':<30}{summary['val']['f1@K12']:>14.3f}{summary['train']['f1@K12']:>14.3f}")
    print(f"  {'K=15':<30}{summary['val']['f1@K15']:>14.3f}{summary['train']['f1@K15']:>14.3f}")
    print(f"  {'K = gold_count':<30}{summary['val']['f1_at_K=gold_count']:>14.3f}{summary['train']['f1_at_K=gold_count']:>14.3f}")
    print(f"  {'Oracle K (per-q best)':<30}{summary['val']['oracle_f1']:>14.3f}{summary['train']['oracle_f1']:>14.3f}")

    print(f"\n  Elbow methods (mean F1 | MAE vs oracle K | r vs gold_count):")
    print(f"  {'method':<24}{'VAL':>30}{'TRAIN':>30}")
    print(f"  {'':<24}{'F1':>8}{'MAE':>8}{'r_gold':>8}{'medK':>6}{'F1':>8}{'MAE':>8}{'r_gold':>8}{'medK':>6}")
    for m in METHODS:
        v = summary["val"]["methods"][m]; t = summary["train"]["methods"][m]
        print(f"  {m:<24}"
              f"{v['mean_F1']:>8.3f}{v['MAE_vs_oracle_K']:>8.2f}{v['pearson_r_vs_gold_count']:>8.3f}{int(v['median_K_pred']):>6}"
              f"{t['mean_F1']:>8.3f}{t['MAE_vs_oracle_K']:>8.2f}{t['pearson_r_vs_gold_count']:>8.3f}{int(t['median_K_pred']):>6}")

    honest_avg = (summary["val"]["oracle_f1"] + summary["train"]["oracle_f1"]) / 2
    print(f"\nHONEST val+train avg @ oracle K: {honest_avg:.3f}")
    # Best method honest avg
    best_method, best_avg = None, 0.0
    for m in METHODS:
        a = (summary["val"]["methods"][m]["mean_F1"]
             + summary["train"]["methods"][m]["mean_F1"]) / 2
        if a > best_avg:
            best_method, best_avg = m, a
    print(f"Best elbow method (honest avg):  {best_method} → {best_avg:.3f}")
    print(bar)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    print(f"\n[done] -> {args.out}")
    print(f"[done] -> {args.out.with_suffix('.csv')}")
    print(f"[elapsed] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
