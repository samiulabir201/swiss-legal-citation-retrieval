"""
verify_issues_per_query.py — Test the claim: gold_count ≈ k × (n_issues).

Three independent "issue" estimators:
  a. n_law_code_clusters   — unique law-code (last token) families in gold citations
  b. n_question_signals    — count of question marks + modal-verb sentence starts
                             in query text (de/fr/it/en aware)
  c. n_sentences           — sentences ending in [.!?]

Then per split (val, train), report:
  - Distribution of gold_count
  - Distribution of each issue estimator
  - Distribution of ratio = gold_count / max(issues, 1)
  - Pearson r(gold_count, each estimator)
  - Verdict: is the ratio stable around 3-5x?

No GPU, no BM25 — pure text analysis.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / "data" / "train.csv"
DEFAULT_VAL   = ROOT / "data" / "val.csv"


# Modal / question-signal verbs across DE/FR/IT/EN
MODAL_VERBS = {
    "en": {"does", "do", "is", "are", "may", "can", "could", "should", "must",
           "will", "would", "shall", "has", "have"},
    "de": {"ist", "sind", "darf", "dürfen", "kann", "können", "muss", "müssen",
           "soll", "sollen", "hat", "haben", "wird", "wurden", "wäre"},
    "fr": {"est", "sont", "peut", "peuvent", "doit", "doivent", "faut",
           "a-t-il", "y-a-t-il", "a", "ont"},
    "it": {"è", "sono", "può", "possono", "deve", "devono", "ha", "hanno"},
}
ALL_MODALS = set().union(*MODAL_VERBS.values())

# Issue-separator phrases that hint at multi-aspect questions
SEPARATORS = [
    r"\bin\s+particular\b", r"\bnamentlich\b", r"\binsbesondere\b",
    r"\bmoreover\b", r"\b(et|ainsi que)\b", r"\bnonché\b",
    r"\bi\.e\.\b", r"\bnotamment\b", r"\bzudem\b",
    r"\bferner\b", r"\bweiterhin\b", r";\s*und\b", r";\s*oder\b",
]
SEPARATOR_RE = re.compile("|".join(SEPARATORS), re.IGNORECASE)


def split_gold_citations(s: str) -> list[str]:
    return [c.strip() for c in str(s or "").split(";") if c.strip()]


def extract_law_codes(gold_cites: list[str]) -> tuple[set[str], int]:
    """Return (unique_law_codes, n_law_cites).

    A "law code" is the last token of an "Art. N Code" citation, or the
    statute-number form (e.g., 'StPO', 'StGB', 'BGG', '142.20', 'USG').
    Court citations (BGE/docket) are intentionally NOT counted here — we
    measure the law-side issue diversity.
    """
    codes = set()
    n_law = 0
    for c in gold_cites:
        if c.startswith("BGE ") or c.startswith("ATF ") or c.startswith("DTF "):
            continue
        if re.match(r"^\d{1,2}[A-Z]{1,4}[_\.]", c):  # docket
            continue
        parts = c.split()
        if len(parts) >= 3 and parts[0] in ("Art.", "Artikel", "art.", "article"):
            # Last token is usually the law code (e.g., 'StPO', '142.20')
            code = parts[-1]
            codes.add(code)
            n_law += 1
    return codes, n_law


def count_question_signals(query: str) -> int:
    """Count question marks + sentence-initial modal verbs."""
    q = (query or "")
    n_qm = q.count("?")
    # Tokenise into sentence-ish chunks
    sentences = re.split(r"(?<=[.!?])\s+", q)
    n_modal = 0
    for s in sentences:
        s = s.strip().lower()
        if not s: continue
        first_token = re.match(r"[\wäöüçèéà'-]+", s)
        if first_token and first_token.group(0) in ALL_MODALS:
            n_modal += 1
    n_sep = len(SEPARATOR_RE.findall(q))
    return max(n_qm, 1) + n_modal + n_sep


def count_sentences(query: str) -> int:
    q = (query or "").strip()
    if not q: return 1
    sents = [s for s in re.split(r"(?<=[.!?])\s+", q) if s.strip()]
    return max(len(sents), 1)


def n_law_code_clusters(gold: list[str]) -> int:
    codes, _ = extract_law_codes(gold)
    return len(codes)


def analyse_split(df: pd.DataFrame, label: str) -> dict:
    rows = []
    for _, r in df.iterrows():
        gold = split_gold_citations(r.get("gold_citations", ""))
        codes, n_law = extract_law_codes(gold)
        rows.append({
            "qid": r["query_id"],
            "split": label,
            "n_gold_total": len(gold),
            "n_gold_law": n_law,
            "n_gold_court": sum(1 for c in gold
                                if c.startswith(("BGE ", "ATF ", "DTF "))
                                or re.match(r"^\d{1,2}[A-Z]{1,4}[_\.]", c)),
            "n_law_codes": len(codes),
            "law_codes": ", ".join(sorted(codes)),
            "query_len_chars": len(r.get("query", "") or ""),
            "n_sentences": count_sentences(r.get("query", "")),
            "n_question_signals": count_question_signals(r.get("query", "")),
        })
    out = pd.DataFrame(rows)
    # Ratios (law side, since extract_law_codes only counts law cites)
    out["ratio_gold_per_code"] = out.apply(
        lambda x: x["n_gold_law"] / max(x["n_law_codes"], 1), axis=1)
    out["ratio_gold_per_signal"] = out.apply(
        lambda x: x["n_gold_total"] / max(x["n_question_signals"], 1), axis=1)
    out["ratio_gold_per_sentence"] = out.apply(
        lambda x: x["n_gold_total"] / max(x["n_sentences"], 1), axis=1)
    return out


def summarize(df: pd.DataFrame, label: str) -> dict:
    if len(df) == 0:
        return {}
    return {
        "split": label,
        "n_queries": len(df),
        "gold_total":  {"mean": float(df["n_gold_total"].mean()),
                        "median": float(df["n_gold_total"].median()),
                        "p25": float(df["n_gold_total"].quantile(0.25)),
                        "p75": float(df["n_gold_total"].quantile(0.75))},
        "gold_law":    {"mean": float(df["n_gold_law"].mean()),
                        "median": float(df["n_gold_law"].median())},
        "n_law_codes": {"mean": float(df["n_law_codes"].mean()),
                        "median": float(df["n_law_codes"].median())},
        "n_question_signals": {"mean": float(df["n_question_signals"].mean()),
                               "median": float(df["n_question_signals"].median())},
        "n_sentences": {"mean": float(df["n_sentences"].mean()),
                        "median": float(df["n_sentences"].median())},
        "ratio_gold_per_code":     {"mean": float(df["ratio_gold_per_code"].mean()),
                                    "median": float(df["ratio_gold_per_code"].median()),
                                    "std": float(df["ratio_gold_per_code"].std())},
        "ratio_gold_per_signal":   {"mean": float(df["ratio_gold_per_signal"].mean()),
                                    "median": float(df["ratio_gold_per_signal"].median()),
                                    "std": float(df["ratio_gold_per_signal"].std())},
        "ratio_gold_per_sentence": {"mean": float(df["ratio_gold_per_sentence"].mean()),
                                    "median": float(df["ratio_gold_per_sentence"].median()),
                                    "std": float(df["ratio_gold_per_sentence"].std())},
        "correlations": {
            "r_gold_vs_law_codes":        float(df["n_gold_total"].corr(df["n_law_codes"])),
            "r_gold_vs_question_signals": float(df["n_gold_total"].corr(df["n_question_signals"])),
            "r_gold_vs_sentences":        float(df["n_gold_total"].corr(df["n_sentences"])),
            "r_gold_law_vs_law_codes":    float(df["n_gold_law"].corr(df["n_law_codes"])),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--val",   type=Path, default=DEFAULT_VAL)
    ap.add_argument("--out",   type=Path, default=ROOT / "artifacts" / "issues_per_query.json")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    train = pd.read_csv(args.train)
    val = pd.read_csv(args.val)

    df_train = analyse_split(train, "train")
    df_val   = analyse_split(val,   "val")
    df_all   = pd.concat([df_val, df_train], ignore_index=True)

    s_val   = summarize(df_val,   "val")
    s_train = summarize(df_train, "train")

    bar = "=" * 78
    print(bar)
    print("ISSUES-PER-QUERY ANALYSIS — does gold_count ≈ 3-5x issues?")
    print(bar)

    # ─── Per-query table for val (small enough to print)
    print("\nVAL — per-query detail:")
    print(f"  {'qid':<10}{'gold_tot':>9}{'gold_law':>9}{'gold_crt':>9}"
          f"{'codes':>7}{'sents':>7}{'sigs':>6}{'ratio_g/c':>10}")
    for _, r in df_val.iterrows():
        print(f"  {r['qid']:<10}{r['n_gold_total']:>9}{r['n_gold_law']:>9}"
              f"{r['n_gold_court']:>9}{r['n_law_codes']:>7}{r['n_sentences']:>7}"
              f"{r['n_question_signals']:>6}{r['ratio_gold_per_code']:>10.2f}")

    # ─── Summary table
    def row(metric, sval, strain):
        return f"  {metric:<32}{sval:>20}{strain:>20}"

    print(f"\n{'':<32}{'VAL':>20}{'TRAIN':>20}")
    print(row("n_queries", s_val["n_queries"], s_train["n_queries"]))
    print(row("median gold_total",
              f"{s_val['gold_total']['median']:.1f}",
              f"{s_train['gold_total']['median']:.1f}"))
    print(row("median gold_law",
              f"{s_val['gold_law']['median']:.1f}",
              f"{s_train['gold_law']['median']:.1f}"))
    print(row("median n_law_codes",
              f"{s_val['n_law_codes']['median']:.1f}",
              f"{s_train['n_law_codes']['median']:.1f}"))
    print(row("median n_question_signals",
              f"{s_val['n_question_signals']['median']:.1f}",
              f"{s_train['n_question_signals']['median']:.1f}"))
    print(row("median n_sentences",
              f"{s_val['n_sentences']['median']:.1f}",
              f"{s_train['n_sentences']['median']:.1f}"))

    print(f"\n  Ratios (median ± std):")
    print(row("gold_law / n_law_codes",
              f"{s_val['ratio_gold_per_code']['median']:.2f} ± {s_val['ratio_gold_per_code']['std']:.2f}",
              f"{s_train['ratio_gold_per_code']['median']:.2f} ± {s_train['ratio_gold_per_code']['std']:.2f}"))
    print(row("gold_total / n_question_signals",
              f"{s_val['ratio_gold_per_signal']['median']:.2f} ± {s_val['ratio_gold_per_signal']['std']:.2f}",
              f"{s_train['ratio_gold_per_signal']['median']:.2f} ± {s_train['ratio_gold_per_signal']['std']:.2f}"))
    print(row("gold_total / n_sentences",
              f"{s_val['ratio_gold_per_sentence']['median']:.2f} ± {s_val['ratio_gold_per_sentence']['std']:.2f}",
              f"{s_train['ratio_gold_per_sentence']['median']:.2f} ± {s_train['ratio_gold_per_sentence']['std']:.2f}"))

    print(f"\n  Correlations:")
    for k in ("r_gold_law_vs_law_codes", "r_gold_vs_question_signals",
              "r_gold_vs_sentences", "r_gold_vs_law_codes"):
        print(row(k,
                  f"{s_val['correlations'][k]:.3f}",
                  f"{s_train['correlations'][k]:.3f}"))

    # ─── Verdict
    print("\nVerdict:")
    r_v = s_val["ratio_gold_per_code"]["median"]
    r_t = s_train["ratio_gold_per_code"]["median"]
    print(f"  gold_law / n_law_codes ratio: VAL median = {r_v:.2f}, TRAIN median = {r_t:.2f}")
    if 2.5 <= r_v <= 6.0 and 2.5 <= r_t <= 6.0:
        print(f"  → ratio is within the claimed 3-5x range on both splits ✓")
    else:
        print(f"  → ratio is OUTSIDE the claimed 3-5x range")

    # Is n_law_codes a USABLE predictor?
    r_corr = s_train["correlations"]["r_gold_law_vs_law_codes"]
    print(f"\n  Pearson r(gold_law, n_law_codes) on TRAIN: {r_corr:.3f}")
    print(f"  → n_law_codes is computed FROM gold, so this is circular. To USE it as a K predictor")
    print(f"    at inference, we'd need to predict n_law_codes from the QUERY text alone.")

    # Save detail + summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(args.out.with_suffix(".csv"), index=False)
    import json
    args.out.write_text(json.dumps({"val": s_val, "train": s_train},
                                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] {args.out}")
    print(f"[done] {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
