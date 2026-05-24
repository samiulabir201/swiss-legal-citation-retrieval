"""End-to-end recall diagnosis: for each val query, work out exactly where in
the pipeline each gold citation got lost. Uses ONLY the cache_endgame folder
(no GPU). Hashes are sha1(query)[:40], matching the notebook's caching scheme."""
import json, hashlib
from pathlib import Path
import pandas as pd

ROOT = Path(r"E:\swiss_citation_extraction")
CACHE = ROOT / "cache_endgame"
DATA  = ROOT / "data"

val = pd.read_csv(DATA / "val.csv")
print(f"val rows: {len(val)}")

# Match cached judge dirs by sha1(query) hash to recover query order.
import hashlib as _h
qhash = {row["query_id"]: _h.sha1(str(row["query"]).encode("utf-8")).hexdigest()
         for _, row in val.iterrows()}
hash_to_qid = {v: k for k, v in qhash.items()}

# Sanity: do all 10 query hashes show up as judge subdirs?
judge_dirs = {p.name for p in (CACHE / "judge").iterdir() if p.is_dir()}
print(f"judge subdirs found: {len(judge_dirs)}")
print(f"matched to val queries: {sum(1 for h in qhash.values() if h in judge_dirs)}/{len(qhash)}")

# Try to load rerank cache (per-query) and judge cache.
print("\n" + "=" * 70)
print("Per-query stage diagnosis")
print("=" * 70)

import sqlite3
con = sqlite3.connect(str(ROOT / "artifacts" / "unified_retrieval.sqlite"))
con.row_factory = sqlite3.Row
# Map citation -> doc_id and family for the gold lookup.
def citation_to_doc(cit):
    r = con.execute("SELECT doc_id, family FROM documents WHERE citation = ? LIMIT 1", (cit,)).fetchone()
    return (r["doc_id"], r["family"]) if r else (None, None)

print(f"\n{'qid':<10} {'gold':>5} {'gold_in_corpus':>14} {'rerank_pool':>12} {'gold_in_rerank':>14} {'judged_yes':>11} {'judged_no':>10} {'auto_yes':>9} {'auto_no':>9}")
print("-" * 110)

ablation_summary = []
for _, row in val.iterrows():
    qid = row["query_id"]
    qhh = qhash[qid]
    gold = [c.strip() for c in str(row["gold_citations"]).split(";") if c.strip()]
    gold_set = set(gold)
    # Resolve gold to doc_ids (which is what the rerank cache uses).
    gold_docs = {}
    for c in gold:
        did, fam = citation_to_doc(c)
        if did:
            gold_docs[c] = (did, fam)
    n_gold = len(gold_set)
    n_gold_in_corpus = len(gold_docs)

    # Rerank cache: per-query JSON {doc_id: rerank_score}
    rerank_path = CACHE / "rerank" / f"{qhh}.json"
    rerank = {}
    if rerank_path.exists():
        rerank = json.loads(rerank_path.read_text(encoding="utf-8"))
    n_rerank = len(rerank)
    n_gold_in_rerank = sum(1 for c, (did, _) in gold_docs.items() if did in rerank)

    # Judge cache
    qdir = CACHE / "judge" / qhh
    judged_yes = judged_no = 0
    auto_yes = auto_no = 0
    for jf in qdir.iterdir() if qdir.exists() else []:
        d = json.loads(jf.read_text(encoding="utf-8"))
        v = d.get("verdict","?")
        c = d.get("category","?")
        if c == "auto":
            if v == "yes": auto_yes += 1
            else: auto_no += 1
        else:
            if v == "yes": judged_yes += 1
            else: judged_no += 1

    print(f"{qid:<10} {n_gold:>5} {n_gold_in_corpus:>14} {n_rerank:>12} {n_gold_in_rerank:>14} {judged_yes:>11} {judged_no:>10} {auto_yes:>9} {auto_no:>9}")
    ablation_summary.append({
        "qid": qid, "n_gold": n_gold, "gold_in_corpus": n_gold_in_corpus,
        "rerank_pool": n_rerank, "gold_in_rerank": n_gold_in_rerank,
        "judged_yes": judged_yes, "judged_no": judged_no,
        "auto_yes": auto_yes, "auto_no": auto_no,
    })

print()
print("=" * 70)
print("Aggregate ceilings")
print("=" * 70)
df = pd.DataFrame(ablation_summary)
print(f"sum gold:                  {df['n_gold'].sum()}")
print(f"sum gold_in_corpus:        {df['gold_in_corpus'].sum()}  ({df['gold_in_corpus'].sum()/df['n_gold'].sum()*100:.1f}%)")
print(f"sum gold_in_rerank_pool:   {df['gold_in_rerank'].sum()}  ({df['gold_in_rerank'].sum()/df['n_gold'].sum()*100:.1f}%)")
print(f"avg rerank pool size:      {df['rerank_pool'].mean():.0f}")
print(f"sum judged YES + auto YES: {df['judged_yes'].sum() + df['auto_yes'].sum()}")
print(f"sum judged NO  + auto NO:  {df['judged_no'].sum()  + df['auto_no'].sum()}")
