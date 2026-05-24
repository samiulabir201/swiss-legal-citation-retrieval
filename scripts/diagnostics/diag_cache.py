"""Diagnose the endgame_cache directory: hyde quality, judge verdict mix,
rerank score distribution, and any suspicious sentinel patterns."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(r"E:\swiss_citation_extraction\cache_endgame")
JUDGE = ROOT / "judge"
RERANK = ROOT / "rerank"

print("=" * 70)
print("HYDE CACHE")
print("=" * 70)
hyde = json.loads((ROOT / "hyde_cache.json").read_text(encoding="utf-8"))
print(f"queries cached: {len(hyde)}")
for k, v in list(hyde.items())[:2]:
    print(f"\n--- query: {k[:80]}... ---")
    print(f"hyde answer ({len(v)} chars):")
    print(v[:500])
    print("...")

print("\n" + "=" * 70)
print("GERMAN EXPANSION CACHE")
print("=" * 70)
de = json.loads((ROOT / "german_expansion_cache.json").read_text(encoding="utf-8"))
print(f"queries cached: {len(de)}")
for k, v in list(de.items())[:2]:
    print(f"\n--- {k[:80]}... ---")
    print(f"expansion: {v[:300]}")

print("\n" + "=" * 70)
print("JUDGE: per-query verdict distribution")
print("=" * 70)
for qhash_dir in sorted(JUDGE.iterdir()):
    if not qhash_dir.is_dir():
        continue
    verdicts = Counter()
    cats = Counter()
    sample_yes = []
    sample_no = []
    for j in qhash_dir.iterdir():
        d = json.loads(j.read_text(encoding="utf-8"))
        v = d.get("verdict", "?")
        verdicts[v] += 1
        cats[d.get("category", "?")] += 1
        if v == "yes" and len(sample_yes) < 2:
            sample_yes.append(d)
        if v == "no" and len(sample_no) < 1:
            sample_no.append(d)
    n = sum(verdicts.values())
    print(f"\n[{qhash_dir.name[:12]}]  n={n}  verdicts={dict(verdicts)}  cats={dict(list(cats.items())[:5])}")
    for s in sample_yes[:1]:
        raw = s.get("raw_response", "")[:200]
        print(f"  YES sample: raw='{raw}'")
    for s in sample_no:
        raw = s.get("raw_response", "")[:200]
        print(f"  NO sample:  raw='{raw}'")

print("\n" + "=" * 70)
print("RERANK: per-query score distribution")
print("=" * 70)
import statistics
for f in sorted(RERANK.glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    scores = list(d.values()) if isinstance(d, dict) else []
    if scores:
        print(f"  [{f.stem[:12]}]  n={len(scores)}  "
              f"min={min(scores):.4f}  max={max(scores):.4f}  "
              f"mean={statistics.mean(scores):.4f}  med={statistics.median(scores):.4f}  "
              f"<0.1: {sum(1 for s in scores if s<0.1)}  "
              f">0.5: {sum(1 for s in scores if s>0.5)}  "
              f">0.9: {sum(1 for s in scores if s>0.9)}")
    else:
        print(f"  [{f.stem[:12]}]  empty or unexpected schema; first key: {list(d.keys())[:1] if isinstance(d, dict) else type(d)}")

print("\n" + "=" * 70)
print("RERANK SAMPLE: top-5 + bottom-5 doc_ids of first cached query")
print("=" * 70)
first_rerank = next(RERANK.glob("*.json"), None)
if first_rerank:
    d = json.loads(first_rerank.read_text(encoding="utf-8"))
    items = sorted(d.items(), key=lambda kv: -kv[1])
    print(f"file: {first_rerank.name}")
    print("top 5 (highest rerank score):")
    for k, v in items[:5]:
        print(f"  {k:40s}  {v:.4f}")
    print("bottom 5:")
    for k, v in items[-5:]:
        print(f"  {k:40s}  {v:.4f}")
