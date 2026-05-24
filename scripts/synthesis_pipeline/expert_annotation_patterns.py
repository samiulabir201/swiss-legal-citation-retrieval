"""
expert_annotation_patterns.py — Why did the human experts pick the gold count they did?

For every train+val query:
  1. Detect procedural framing keywords (Beschwerde / appeal / costs / jurisdiction)
  2. Detect question framing (zulässig / lawful / extension / calculation / eligibility)
  3. Count law-code mentions in the QUERY TEXT
  4. Compute the article-cluster sizes inside gold (how many Art. of same code cited)
  5. Test correlations:
       - Does presence of procedural framing → larger gold?
       - Does code-mention count match gold-code count?
       - What's the cluster-size distribution per code per query?
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = ROOT / "data" / "train.csv"
DEFAULT_VAL   = ROOT / "data" / "val.csv"


# ── Law-code alias list (a focused subset that covers most train/val gold) ──
KNOWN_CODES = {
    # Civil
    "ZGB", "CC", "OR", "CO", "ZPO", "CPC", "IPRG", "LDIP", "SchKG", "LP", "LEF",
    # Criminal
    "StGB", "CP", "StPO", "CPP", "BetmG", "LStup",
    # Constitutional / public
    "BV", "Cst", "Cost", "EMRK", "CEDH",
    # Federal procedure
    "BGG", "LTF", "OG", "OJ", "VwVG", "PA", "StBOG",
    # Tax / financial
    "DBG", "LIFD", "StHG", "LHID", "MWSTG", "LTVA", "VStG",
    # Migration
    "AIG", "AuG", "LEI", "LEtr", "AsylG", "LAsi",
    # Social insurance
    "ATSG", "LPGA", "AHVG", "LAVS", "IVG", "LAI", "UVG", "LAA",
    "AVIG", "BVG", "LPP", "KVG", "LAMal", "ELG", "EOG",
    # Environment / planning
    "USG", "LPE", "RPG", "LAT", "UVPV", "OEIE", "GSchG", "LEaux",
    # Other
    "KG", "LCart", "MSchG", "PatG", "URG", "FINMAG", "DSG", "LPD",
    "GBV", "FusG", "KG", "BoeB", "LMP", "WaG", "NHG",
    "AIVO", "BVV", "BVV2", "AHVV", "BVO", "VRV",
}

# Boundary regex for code detection in query text
CODE_RE = re.compile(r"\b(" + "|".join(re.escape(c) for c in KNOWN_CODES) + r")\b")

# Procedural framing keywords (DE/FR/IT/EN aware)
FRAMING = {
    "appeal":        re.compile(r"\b(Beschwerde|Berufung|Revision|appeal|recours|Rekurs|ricorso|appel|Rüge)\b", re.IGNORECASE),
    "extension":     re.compile(r"\b(Verlängerung|extension|prolongation|prolong|protrait)\b", re.IGNORECASE),
    "cost":          re.compile(r"\b(Kosten|Gerichtskosten|Honorar|costs|fees|frais|Gebühr)\b", re.IGNORECASE),
    "jurisdiction":  re.compile(r"\b(Zuständigkeit|jurisdiction|compétence|competenza)\b", re.IGNORECASE),
    "lawfulness":    re.compile(r"\b(zulässig|lawful|permissible|rechtsgenügend|rechtsgenüglich|rechtmäßig|justified|justifié)\b", re.IGNORECASE),
    "detention":     re.compile(r"\b(Haft|detention|détention|Untersuchungshaft|carcerazione)\b", re.IGNORECASE),
    "calculation":   re.compile(r"\b(Berechnung|calculation|amount|montant|Höhe|Bemessung|fixation)\b", re.IGNORECASE),
    "eligibility":   re.compile(r"\b(Anspruch|eligibility|entitled|Recht auf|droit à|diritto a)\b", re.IGNORECASE),
    "proportionality": re.compile(r"\b(Verhältnismäßigkeit|proportionnalité|proportionality|proporzionalità|verhaltnismaessigkeit)\b", re.IGNORECASE),
    "constitutional": re.compile(r"\b(verfassungsrechtlich|constitutional|constitutionnel|fundamental rights|Grundrecht)\b", re.IGNORECASE),
    "permissible_modal": re.compile(r"\b(may a court|may the|darf|peut-on|può|is it lawful|kann|is it permissible)\b", re.IGNORECASE),
}


def detect_query_codes(query: str) -> set[str]:
    return set(CODE_RE.findall(query or ""))


def detect_framing_tags(query: str) -> list[str]:
    return [tag for tag, rx in FRAMING.items() if rx.search(query or "")]


def parse_law_gold(gold_str: str) -> list[tuple[str, str]]:
    """Return list of (article_id, law_code) for law-only gold."""
    out = []
    for c in str(gold_str or "").split(";"):
        c = c.strip()
        if not c: continue
        if c.startswith(("BGE ", "ATF ", "DTF ")): continue
        if re.match(r"^\d{1,2}[A-Z]{1,4}[_\.]", c): continue
        parts = c.split()
        if len(parts) >= 3 and parts[0] in ("Art.", "Artikel", "art.", "article"):
            article = parts[1]
            code = parts[-1]
            out.append((article, code))
    return out


def cluster_sizes(gold_pairs: list[tuple[str, str]]) -> dict[str, int]:
    """For each code, how many Art. of that code are cited? (Counts Art-level, NOT Abs.-level)."""
    # Group by code → unique articles
    code_arts: dict[str, set] = defaultdict(set)
    for article, code in gold_pairs:
        code_arts[code].add(article)
    return {c: len(a) for c, a in code_arts.items()}


def analyse(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        q = r.get("query", "") or ""
        gold_pairs = parse_law_gold(r.get("gold_citations", ""))
        codes_in_query = detect_query_codes(q)
        codes_in_gold = set(c for _, c in gold_pairs)
        tags = detect_framing_tags(q)
        cl = cluster_sizes(gold_pairs)
        rows.append({
            "qid": r["query_id"],
            "split": label,
            "n_gold_law": len(gold_pairs),
            "n_law_articles_unique": len(set(gold_pairs)),
            "n_codes_in_gold": len(codes_in_gold),
            "n_codes_in_query": len(codes_in_query),
            "codes_match_gold": len(codes_in_query & codes_in_gold),
            "codes_missed_in_query": len(codes_in_gold - codes_in_query),
            "framing_tags": ",".join(tags),
            "n_framing_tags": len(tags),
            "query_len": len(q),
            "max_cluster_size": max(cl.values()) if cl else 0,
            "median_cluster_size": float(np.median(list(cl.values()))) if cl else 0,
            "cluster_distribution": str(sorted(cl.values(), reverse=True)),
            "has_appeal": "appeal" in tags,
            "has_cost": "cost" in tags,
            "has_jurisdiction": "jurisdiction" in tags,
            "has_lawfulness": "lawfulness" in tags,
            "has_proportionality": "proportionality" in tags,
            "has_detention": "detention" in tags,
            "has_calculation": "calculation" in tags,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--val",   type=Path, default=DEFAULT_VAL)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "expert_patterns.json")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    train = pd.read_csv(args.train)
    val = pd.read_csv(args.val)
    df_tr = analyse(train, "train")
    df_val = analyse(val, "val")
    df_tr = df_tr[df_tr["n_gold_law"] > 0]
    df_val = df_val[df_val["n_gold_law"] > 0]

    bar = "=" * 78

    # ── 1. Code-mention test
    print(bar)
    print("1. Does the query mention the codes that end up in gold?")
    print(bar)
    for d, lab in ((df_val, "val"), (df_tr, "train")):
        n = len(d)
        coverage = (d["codes_match_gold"] / d["n_codes_in_gold"].clip(lower=1)).mean()
        mean_q = d["n_codes_in_query"].mean()
        mean_g = d["n_codes_in_gold"].mean()
        miss = d["codes_missed_in_query"].mean()
        print(f"  {lab:>6} (n={n}): query mentions {mean_q:.2f} codes on avg, "
              f"gold has {mean_g:.2f}, {coverage*100:.1f}% gold codes are explicit in query, "
              f"{miss:.2f} codes missed")

    # ── 2. Article cluster size (per code, per query)
    print(f"\n{bar}\n2. Article cluster sizes — how many Art. of same code are cited together?\n{bar}")
    for d, lab in ((df_val, "val"), (df_tr, "train")):
        clusters = []
        for cl_str in d["cluster_distribution"]:
            try:
                clusters.extend(eval(cl_str))
            except Exception:
                pass
        c = Counter(clusters)
        total = sum(c.values())
        print(f"\n  {lab.upper()} ({total} (qid, code) pairs):")
        for size in sorted(c.keys())[:10]:
            pct = c[size] / total * 100
            bar_str = "█" * int(pct / 2)
            print(f"    cluster size = {size:>2}: {c[size]:>4} ({pct:>5.1f}%)  {bar_str}")
        print(f"    median cluster = {np.median(clusters):.1f}, "
              f"mean = {np.mean(clusters):.2f}, max = {max(clusters)}")

    # ── 3. Framing-tag effect on gold count
    print(f"\n{bar}\n3. Procedural-framing tags vs gold count\n{bar}")
    for d, lab in ((df_val, "val"), (df_tr, "train")):
        print(f"\n  {lab.upper()}:")
        print(f"    {'tag':<22} {'#queries':>9} {'avg gold_law':>14} {'avg #codes':>12}")
        for tag in FRAMING:
            mask = d["framing_tags"].str.contains(tag, na=False, regex=False)
            if mask.sum() == 0: continue
            avg_gold = d.loc[mask, "n_gold_law"].mean()
            avg_codes = d.loc[mask, "n_codes_in_gold"].mean()
            print(f"    {tag:<22} {mask.sum():>9} {avg_gold:>14.2f} {avg_codes:>12.2f}")
        # vs queries WITH NO framing tags
        none_mask = d["n_framing_tags"] == 0
        if none_mask.sum() > 0:
            print(f"    {'(no framing tags)':<22} {none_mask.sum():>9} "
                  f"{d.loc[none_mask, 'n_gold_law'].mean():>14.2f} "
                  f"{d.loc[none_mask, 'n_codes_in_gold'].mean():>12.2f}")

    # ── 4. Multi-tag stacking effect
    print(f"\n{bar}\n4. Stacking effect — does #framing tags predict gold count?\n{bar}")
    for d, lab in ((df_val, "val"), (df_tr, "train")):
        print(f"\n  {lab.upper()}:")
        print(f"    {'n_framing_tags':<22} {'#queries':>9} {'med gold_law':>14} {'mean gold_law':>15}")
        for n_t in sorted(d["n_framing_tags"].unique())[:10]:
            sub = d[d["n_framing_tags"] == n_t]
            print(f"    {str(n_t):<22} {len(sub):>9} {sub['n_gold_law'].median():>14.1f} "
                  f"{sub['n_gold_law'].mean():>15.2f}")

    # ── 5. Correlations
    print(f"\n{bar}\n5. Pearson correlations\n{bar}")
    for d, lab in ((df_val, "val"), (df_tr, "train")):
        print(f"\n  {lab.upper()}:")
        for col in ("n_framing_tags", "n_codes_in_query", "query_len", "max_cluster_size"):
            r = d["n_gold_law"].corr(d[col])
            print(f"    r(gold_law, {col:<22}) = {r:>6.3f}")

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([df_val, df_tr], ignore_index=True).to_csv(
        args.out.with_suffix(".csv"), index=False)
    print(f"\n[done] {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
