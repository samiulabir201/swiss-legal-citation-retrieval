"""
preextract_court_cards_for_v13.py — slim court_authority_cards_v5_unified.jsonl
to only the citations that appear in the v7.5 pool, so the v13 notebook can load
it cheaply.

Inputs (local):
  - per_query_snapshot.json  : v7.5 final_topk per qid
  - corpus_snapshot.json.gz  : did -> cit/text/role/family map
  - court_authority_cards_v5_unified.jsonl (10 GB)  : full LLM-enriched cards

Output:
  - court_cards_slim.jsonl   : ~few MB, contains only citations referenced by
                               the v7.5 pool. Upload to DATA_DIR alongside the
                               other v13 inputs.

The v13 notebook (court channel) loads this slim file as `court_cards` dict
and joins to each court Candidate by citation_canon to enrich the reranker
dossier with text_excerpt_original / topic / subtopic / statute_anchors.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOL = ROOT / "drive_sync" / "swiss_law" / "v7_pool_recall_089_and_per_query" / "snapshot" / "per_query_snapshot.json"
DEFAULT_CORPUS_SNAP = ROOT / "drive_sync" / "swiss_law" / "v7_pool_recall_089_and_per_query" / "snapshot" / "corpus_snapshot.json.gz"
DEFAULT_CARDS = ROOT / "artifacts" / "court_authority_cards_v5_unified.jsonl"
DEFAULT_OUT = ROOT / "artifacts" / "court_cards_slim_for_v13.jsonl"


# Fields kept from each card. text_excerpt_original is the big win (real paragraph
# text, ~800 chars vs slim snapshot's truncated `ct`). topic+subtopic provide
# English topic descriptors (92% populated in v5_unified). statute_anchors is
# the strongest cross-domain signal (67% populated) — if the query mentions
# "Art. 221 StPO" and the case anchors include it, that's a direct match.
KEEP_ENRICHMENT_FIELDS = (
    "topic", "subtopic", "legal_topic", "doctrinal_rule", "legal_test",
    "court_holding", "factual_context", "legal_rule", "paragraph_role",
    "authority_role", "statute_anchors", "case_anchors",
    "english_legal_concepts", "search_keywords", "outcome_signal",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool",        type=Path, default=DEFAULT_POOL)
    ap.add_argument("--corpus-snap", type=Path, default=DEFAULT_CORPUS_SNAP)
    ap.add_argument("--cards",       type=Path, default=DEFAULT_CARDS)
    ap.add_argument("--out",         type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # 1. Collect court dids in v7.5 pool
    print(f"[1/3] Loading pool snapshot from {args.pool}")
    with args.pool.open("r", encoding="utf-8") as f:
        pool = json.load(f)
    court_dids = set()
    for qid, snap in pool.items():
        for d in snap.get("final_topk", []):
            if d.startswith("court:"):
                court_dids.add(d)
    print(f"  unique court dids in pool: {len(court_dids):,}")

    # 2. Map dids -> citations via corpus_snapshot
    print(f"[2/3] Loading corpus snapshot from {args.corpus_snap}")
    t0 = time.time()
    with gzip.open(args.corpus_snap, "rt", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"  corpus_snap entries: {len(corpus):,}  ({time.time()-t0:.1f}s)")
    target_citations = set()
    for d in court_dids:
        rec = corpus.get(d)
        if rec and rec.get("cit"):
            target_citations.add(rec["cit"])
    print(f"  target citations to extract: {len(target_citations):,}")

    # 3. Stream the 10 GB cards file and keep only matches
    print(f"[3/3] Streaming cards from {args.cards}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = 0
    t0 = time.time()
    last_log = t0
    with args.cards.open("r", encoding="utf-8") as fin, \
         args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            n_in += 1
            try:
                r = json.loads(line)
            except Exception:
                continue
            cit = r.get("citation") or ""
            if cit not in target_citations:
                continue
            en = r.get("rag_enrichment") or {}
            slim = {
                "citation": cit,
                "language": r.get("language", ""),
                "legal_area_static": r.get("legal_area_static", ""),
                "text_excerpt_original": r.get("text_excerpt_original", ""),
                "rag_enrichment": {k: en.get(k) for k in KEEP_ENRICHMENT_FIELDS
                                   if en.get(k)},
            }
            fout.write(json.dumps(slim, ensure_ascii=False) + "\n")
            n_out += 1
            now = time.time()
            if now - last_log >= 5.0:
                rate = n_in / max(now - t0, 1e-9)
                print(f"  [{now-t0:6.1f}s] read={n_in:,}  kept={n_out:,}  "
                      f"({rate:,.0f} rec/s)")
                last_log = now
    print(f"  done: read={n_in:,}  kept={n_out:,}  in {time.time()-t0:.1f}s")
    print(f"  output: {args.out}")
    print(f"  coverage: {n_out:,} / {len(target_citations):,} target citations "
          f"= {n_out*100/max(len(target_citations),1):.1f}%")


if __name__ == "__main__":
    main()
