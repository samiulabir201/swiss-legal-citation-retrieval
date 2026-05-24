"""
val_vs_train_signature.py — What linguistic signals separate val from train?

Hypothesis: val queries are reverse-engineered from REAL Swiss Federal Court
decisions (gold contains the BGE they were extracted from). Train queries are
exam-style anonymized fact patterns. Test by detecting:

  1. Anonymization markers (A AG, B, xxx, yyy, party-letter placeholders)
  2. Concrete calendar dates ("18 October 2024", "1. März 2020")
  3. Adversarial structure markers ("sought", "opposed", "argued", "—i.e. does")
  4. Procedural-posture cues ("by order dated", "remanded", "appeal", "the
     prosecutor", "the detainee", litigating-party language)
  5. Language detection (val=English, train=German/French/Italian?)
  6. Sentence structure (long multi-clause vs paragraph+question)

Report per-signal, per-split distribution + a single discriminator score.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / "data" / "train.csv"
DEFAULT_VAL   = ROOT / "data" / "val.csv"


# ─── Signals ─────────────────────────────────────────────────────────────

# 1. Anonymization markers — train uses single-letter companies, placeholders
ANON_NAMES_RE = re.compile(
    r"(?:\b(?:[A-Z]|[A-Z][A-Z])\s+(?:AG|GmbH|SA|Sàrl|Ltd|Inc|Co)\b)"  # "A AG", "AB GmbH"
    r"|(?:\b[A-Z]\.\s+[A-Z][a-z]+\s*)"                                  # "A. Müller"
    r"|(?:Nr\.\s*[xy]+)"                                                # "Nr. yyy"
    r"|(?:\b(?:xxx|yyy|zzz|aaa|bbb)\b)"                                 # placeholders
)
SINGLE_LETTER_PARTY_RE = re.compile(r"\b(?:Herr|Frau|Mr|Mrs|Mme)\s+([A-Z])\b|"  # "Herr A"
                                     r"\b([A-Z])\s+(?:beschwert|opposed|argues|behauptet|claims)")

# 2. Concrete calendar dates
DATE_DDMMYY_RE = re.compile(r"\b\d{1,2}\.\s?\d{1,2}\.\s?\d{4}\b")             # DE 18.10.2024
DATE_WORD_DE_RE = re.compile(r"\b\d{1,2}\.\s*(Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4}\b", re.IGNORECASE)
DATE_WORD_EN_RE = re.compile(r"\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b", re.IGNORECASE)
DATE_WORD_FR_RE = re.compile(r"\b\d{1,2}\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}\b", re.IGNORECASE)

# 3. Adversarial / litigation-process markers
LITIGANT_RE = re.compile(
    r"\b(the\s+(?:prosecutor|detainee|defendant|plaintiff|accused|claimant|appellant|respondent|insured|insurer)"
    r"|der\s+(?:Beschuldigte|Beklagte|Kläger|Angeschuldigte|Versicherte|Beschwerdeführer)"
    r"|der\s+Staatsanwalt|le\s+procureur"
    r")\b", re.IGNORECASE)

ADVERSARIAL_RE = re.compile(
    r"(sought\s+(?:an\s+)?extension|opposed|argued|requested|denied|"
    r"submitted|while\s+the|whereas|—i\.e\.|i\.e\.\s+does|"
    r"beantragte|beantragen|wehrte\s+sich|widersetzte\s+sich|"
    r"vorgetragen|geltend\s+gemacht|verlangte|verlangt)",
    re.IGNORECASE,
)

# 4. Procedural-posture markers
POSTURE_RE = re.compile(
    r"(by\s+order\s+dated|by\s+(?:decision|order)\s+of|"
    r"remanded|appeal|extension|sought|"
    r"mit\s+Verfügung\s+vom|mit\s+(?:Entscheid|Beschluss)\s+vom|"
    r"wurde\s+(?:zur|in)\s+(?:Untersuchungshaft|Sicherheitshaft)|"
    r"erhob\s+Beschwerde|"
    r"par\s+ordonnance\s+du|par\s+décision\s+du|"
    r"a\s+formé\s+recours)",
    re.IGNORECASE,
)

# 5. Closing-consolidation pattern ("—i.e. does X justify Y?")
CLOSING_RE = re.compile(r"—\s*i\.e\.\s+(?:does|may|can|is|do|are|should)", re.IGNORECASE)

# 6. Language hint (very crude — just to confirm val=EN, train=DE)
def detect_lang_crude(s: str) -> str:
    s = (s or "").lower()
    en_score = sum(s.count(w) for w in (" the ", " is ", " a ", " an ", " does ", " may ", " under "))
    de_score = sum(s.count(w) for w in (" der ", " die ", " das ", " ist ", " ein ", " eine ", " für ", " mit "))
    fr_score = sum(s.count(w) for w in (" le ", " la ", " les ", " est ", " un ", " une "))
    it_score = sum(s.count(w) for w in (" il ", " lo ", " la ", " è ", " un "))
    scores = {"en": en_score, "de": de_score, "fr": fr_score, "it": it_score}
    return max(scores, key=scores.get)


def analyse(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        q = r.get("query", "") or ""
        n_anon = len(ANON_NAMES_RE.findall(q))
        n_litigant = len(LITIGANT_RE.findall(q))
        n_adversarial = len(ADVERSARIAL_RE.findall(q))
        n_posture = len(POSTURE_RE.findall(q))
        n_dates = (
            len(DATE_DDMMYY_RE.findall(q))
            + len(DATE_WORD_DE_RE.findall(q))
            + len(DATE_WORD_EN_RE.findall(q))
            + len(DATE_WORD_FR_RE.findall(q))
        )
        has_closing = bool(CLOSING_RE.search(q))
        rows.append({
            "qid": r["query_id"],
            "split": label,
            "lang": detect_lang_crude(q),
            "n_chars": len(q),
            "n_anon_markers": n_anon,
            "n_litigant_refs": n_litigant,
            "n_adversarial_verbs": n_adversarial,
            "n_posture_cues": n_posture,
            "n_concrete_dates": n_dates,
            "has_iie_closing": has_closing,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--val",   type=Path, default=DEFAULT_VAL)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    train = pd.read_csv(args.train)
    val = pd.read_csv(args.val)
    df_tr = analyse(train, "train")
    df_val = analyse(val, "val")
    bar = "=" * 80

    print(bar)
    print("VAL vs TRAIN LINGUISTIC SIGNATURES")
    print(bar)

    print("\n1. Language distribution:")
    print(f"  VAL    ({len(df_val)})  : {df_val['lang'].value_counts().to_dict()}")
    print(f"  TRAIN  ({len(df_tr)})  : {df_tr['lang'].value_counts().to_dict()}")

    print("\n2. Per-signal median + % of queries with ≥1 hit:")
    print(f"  {'signal':<28} {'VAL median':>11} {'VAL %≥1':>9} {'TRAIN median':>13} {'TRAIN %≥1':>11}")
    for col in ("n_chars", "n_anon_markers", "n_litigant_refs",
                "n_adversarial_verbs", "n_posture_cues", "n_concrete_dates"):
        v_med = df_val[col].median()
        t_med = df_tr[col].median()
        v_pct = (df_val[col] >= 1).mean() * 100
        t_pct = (df_tr[col] >= 1).mean() * 100
        print(f"  {col:<28} {v_med:>11.1f} {v_pct:>8.1f}% {t_med:>13.1f} {t_pct:>10.1f}%")

    # has_iie_closing is boolean
    print(f"  {'has_iie_closing':<28} {'-':>11} {df_val['has_iie_closing'].mean()*100:>8.1f}% "
          f"{'-':>13} {df_tr['has_iie_closing'].mean()*100:>10.1f}%")

    # Build a "val-signature score" per query
    def score(df):
        return (
            df["n_chars"].clip(0, 2000) / 200          # 0..10 from length
            + df["n_anon_markers"] * -2                 # negative: anon → train
            + df["n_litigant_refs"] * 2                 # positive: litigants → val
            + df["n_adversarial_verbs"] * 1.5
            + df["n_posture_cues"] * 1.5
            + df["n_concrete_dates"] * 1.5
            + df["has_iie_closing"].astype(int) * 5
        )

    df_val["sig_score"] = score(df_val)
    df_tr["sig_score"]  = score(df_tr)

    print(f"\n3. Composite 'val-style' score (higher = val-like):")
    print(f"  VAL:   median={df_val['sig_score'].median():.2f}  "
          f"p25={df_val['sig_score'].quantile(0.25):.2f}  p75={df_val['sig_score'].quantile(0.75):.2f}")
    print(f"  TRAIN: median={df_tr['sig_score'].median():.2f}  "
          f"p25={df_tr['sig_score'].quantile(0.25):.2f}  p75={df_tr['sig_score'].quantile(0.75):.2f}")

    # What fraction of train queries score ABOVE val's p25 (i.e. look val-like)?
    val_p25 = df_val["sig_score"].quantile(0.25)
    train_above = (df_tr["sig_score"] >= val_p25).mean() * 100
    train_min_val = (df_tr["sig_score"] >= df_val["sig_score"].min()).mean() * 100
    print(f"\n  TRAIN queries scoring ≥ val_p25 ({val_p25:.2f}): {train_above:.1f}%")
    print(f"  TRAIN queries scoring ≥ val_min: {train_min_val:.1f}%")

    print(f"\n4. Per-val-query signal table:")
    print(df_val.drop(columns=["sig_score"]).to_string(index=False))

    print(f"\n5. Top-10 train queries that score MOST val-like:")
    top_train = df_tr.nlargest(10, "sig_score")
    print(top_train[["qid", "n_chars", "n_anon_markers", "n_litigant_refs",
                      "n_adversarial_verbs", "n_posture_cues", "n_concrete_dates",
                      "has_iie_closing", "sig_score"]].to_string(index=False))

    print(f"\n[done] script complete")


if __name__ == "__main__":
    main()
