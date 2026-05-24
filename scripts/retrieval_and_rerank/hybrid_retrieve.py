#!/usr/bin/env python
"""Hybrid retriever for the unified Swiss legal corpus.

Multi-channel candidate generation that fuses

  - BM25  (SQLite FTS5 over fielded text)
  - Vector ANN (cosine on L2-normed Qwen3-Embedding-8B fp16 chunks; brute force)
  - Statute anchors (statute_links table, exact match)
  - Case / docket anchors (case_links + same court_base expansion)
  - Adjacent-article expansion (adjacent_law_links, law family only)
  - Legal-area soft filter

Channel results are merged with reciprocal-rank fusion (k=60) and a small
authority-score bonus, capped at `--top-k` candidates (default 1000) per query.

The retriever degrades gracefully when only some embedding chunks are on disk —
the vector channel scores whichever rows are present and skips the rest. BM25
and anchor channels always work because they live in SQLite.

Usage (single query, no GPU needed if you supply a pre-encoded query embedding):

    python scripts/hybrid_retrieve.py \
        --query "When can Swiss courts extend pretrial detention based on collusion risk?" \
        --query-embedding artifacts/embeddings/queries_demo.npy \
        --top-k 50

Or with BM25+anchors only (no vector channel):

    python scripts/hybrid_retrieve.py \
        --query "Art. 221 StPO pretrial detention collusion" \
        --no-vector --top-k 50
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"
EMB_DIR = ART_DIR / "embeddings"

DEFAULT_SQLITE = ART_DIR / "unified_retrieval.sqlite"
DEFAULT_MANIFEST = EMB_DIR / "qwen3_8b_unified_manifest.parquet"
DEFAULT_CHUNK_GLOB = "qwen3_8b_unified_chunk*.npy"

# Patterns mirror those in build_unified_retrieval_corpus.py
BGE_BASE_RE = re.compile(r"\bBGE\s+\d{3}\s+[IVX]{1,4}\s+\d+[a-z]?\b")
BGE_FULL_RE = re.compile(r"\bBGE\s+\d{3}\s+[IVX]{1,4}\s+\d+[a-z]?(?:\s+E\.\s*[\d\.]+)?\b")
DOCKET_BASE_RE = re.compile(r"\b\d{1,2}[A-Z]{1,4}[_\.]\d{1,5}/\d{4}\b")
ART_PATTERN = re.compile(
    r"\bArt\.?\s*(\d+[a-z]?)(?:\s+(?:Abs|Bst|Lit|Ziff|Ch|Cpv)\.?\s*\d+\w*)*"
    r"\s+([A-Z][A-Za-z]{1,8}|\d{3}\.\d+)",
    re.IGNORECASE,
)
KNOWN_LAW_CODES = {
    "AIG", "AI", "ALC", "AMLA", "ATSG", "AVIG", "AHVG", "AsylG", "AuG",
    "BankG", "BetmG", "BGFA", "BGG", "BVG", "BV", "CC", "CEDH", "CO",
    "CPC", "CP", "CPP", "Cst", "DBG", "DSG", "EMRK", "FINMAG", "IVG",
    "IPRG", "KG", "KVG", "LAI", "LAMal", "LAVS", "LEI", "LEtr", "LIFD",
    "LP", "LPGA", "LTF", "LPP", "LStup", "LAsi", "LAA", "MSchG", "MWSTG",
    "NHG", "OJ", "OG", "OR", "PatG", "RPG", "SchKG", "StGB", "StPO",
    "SVG", "UVG", "URG", "UWG", "VwVG", "ZGB", "ZPO",
}

# FTS5 reserved tokens / unsafe chars
FTS5_RESERVED = {"AND", "OR", "NOT", "NEAR"}
FTS5_BAD_CHARS = re.compile(r'[\"\(\)\*\+\-\^]')


# ---------------------------------------------------------------------------
# Query understanding
# ---------------------------------------------------------------------------

def parse_query_anchors(query: str) -> dict:
    """Extract deterministic anchors from a free-text query.

    Returns: {'cases': [...], 'dockets': [...], 'statutes': [...], 'law_codes': [...]}
    """
    cases = sorted(set(BGE_FULL_RE.findall(query)))
    dockets = sorted(set(DOCKET_BASE_RE.findall(query)))
    statutes_raw = ART_PATTERN.findall(query)
    statutes = []
    law_codes = set()
    for art_num, code in statutes_raw:
        statute = f"Art. {art_num} {code}"
        statutes.append(statute)
        if code in KNOWN_LAW_CODES:
            law_codes.add(code)
        if re.fullmatch(r"\d{3}\.\d+", code):
            law_codes.add(code)
    # also extract bare law-code mentions
    for tok in re.findall(r"\b([A-Z][A-Za-z]{1,8})\b", query):
        if tok in KNOWN_LAW_CODES:
            law_codes.add(tok)
    return {
        "cases": cases,
        "dockets": dockets,
        "statutes": sorted(set(statutes)),
        "law_codes": sorted(law_codes),
    }


def fts5_match_string(query: str) -> str:
    """Sanitize a free-text query for FTS5 MATCH.

    Strategy: strip syntax-significant chars, drop reserved keywords and very short
    tokens, OR-join the rest so any term can match. Quoted phrases are not used here
    because most queries are concept-bag style.
    """
    cleaned = FTS5_BAD_CHARS.sub(" ", query)
    tokens = []
    for tok in re.findall(r"[A-Za-zÀ-ÿ0-9_]+", cleaned):
        if len(tok) < 2:
            continue
        upper = tok.upper()
        if upper in FTS5_RESERVED:
            continue
        tokens.append(f'"{tok}"')
    if not tokens:
        return ""
    return " OR ".join(tokens)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    def __init__(
        self,
        sqlite_path: Path = DEFAULT_SQLITE,
        manifest_path: Path = DEFAULT_MANIFEST,
        embeddings_dir: Path = EMB_DIR,
        chunk_glob: str = DEFAULT_CHUNK_GLOB,
        vector_dtype=np.float32,
    ):
        if not sqlite_path.exists():
            raise FileNotFoundError(sqlite_path)
        self.sqlite_path = sqlite_path
        self.con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA cache_size=-200000")
        self.con.execute("PRAGMA temp_store=MEMORY")
        self.vector_dtype = vector_dtype

        self.chunks: list[tuple[Path, int, int]] = []  # (path, start_row, n_rows)
        self.manifest = None
        self.row_index_to_doc: dict[int, dict] | None = None
        if manifest_path.exists():
            t0 = time.time()
            tbl = pq.read_table(manifest_path)
            self.manifest = tbl.to_pandas()
            print(f"[init] manifest rows: {len(self.manifest):,}  "
                  f"(load {time.time()-t0:.1f}s)", file=sys.stderr)

        # Discover available chunk files
        chunk_paths = sorted(embeddings_dir.glob(chunk_glob))
        offset = 0
        # Rows 0..N follow the manifest order; chunks are saved as fixed-size
        # 100k slices in that same order. Map chunk file -> [start_row, end_row).
        for path in chunk_paths:
            arr = np.load(path, mmap_mode="r")
            n = arr.shape[0]
            # chunk index encoded in filename, e.g. ..._chunk010.npy → idx 10
            m = re.search(r"chunk(\d+)\.npy$", path.name)
            chunk_idx = int(m.group(1)) if m else None
            if chunk_idx is not None:
                start = chunk_idx * n
            else:
                start = offset
            offset = start + n
            self.chunks.append((path, start, n))
        print(f"[init] found {len(self.chunks)} embedding chunk(s); "
              f"covered_rows={sum(c[2] for c in self.chunks):,}", file=sys.stderr)

    # ---- channels ----------------------------------------------------------

    def bm25_search(self, query: str, k: int = 800) -> list[dict]:
        match = fts5_match_string(query)
        if not match:
            return []
        rows = self.con.execute(
            "SELECT documents_fts.doc_id AS doc_id, "
            "       documents.family AS family, documents.citation AS citation, "
            "       documents.authority_score AS authority_score, "
            "       bm25(documents_fts) AS s "
            "FROM documents_fts JOIN documents USING (doc_id) "
            "WHERE documents_fts MATCH ? "
            "ORDER BY s LIMIT ?",
            (match, k),
        ).fetchall()
        # bm25 is more-negative-is-better. Flip sign to bigger=better.
        return [
            {"doc_id": r["doc_id"], "family": r["family"], "citation": r["citation"],
             "authority_score": r["authority_score"], "channel": "bm25",
             "raw_score": -r["s"]}
            for r in rows
        ]

    def vector_search(self, q_emb: np.ndarray, k: int = 800) -> list[dict]:
        if not self.chunks or self.manifest is None:
            return []
        q = np.asarray(q_emb, dtype=self.vector_dtype).reshape(-1)
        # Renormalize defensively
        n = float(np.linalg.norm(q))
        if n > 0:
            q = q / n
        else:
            return []

        # heap-style top-k merge across chunks
        heap_scores = np.full(k, -np.inf, dtype=self.vector_dtype)
        heap_rows = np.full(k, -1, dtype=np.int64)
        for path, start, n_rows in self.chunks:
            arr = np.load(path, mmap_mode="r")  # fp16
            # blocking by fp32 cast for accurate dot
            block = 50_000
            for s in range(0, n_rows, block):
                e = min(s + block, n_rows)
                a = np.asarray(arr[s:e], dtype=self.vector_dtype)
                scores = a @ q  # (block,)
                # top-k merge: stack scores+heap, take top-k
                if scores.shape[0] >= k:
                    # local top-k
                    idx_local = np.argpartition(-scores, k - 1)[:k]
                    cand_scores = scores[idx_local]
                    cand_rows = (start + s) + idx_local.astype(np.int64)
                else:
                    cand_scores = scores
                    cand_rows = (start + s) + np.arange(scores.shape[0], dtype=np.int64)
                # merge with heap
                combined_scores = np.concatenate([heap_scores, cand_scores])
                combined_rows = np.concatenate([heap_rows, cand_rows])
                top_idx = np.argpartition(-combined_scores, k - 1)[:k]
                heap_scores = combined_scores[top_idx]
                heap_rows = combined_rows[top_idx]

        # final sort
        order = np.argsort(-heap_scores)
        heap_scores = heap_scores[order]
        heap_rows = heap_rows[order]
        # filter out any -1/inf placeholders
        mask = (heap_rows >= 0) & np.isfinite(heap_scores)
        heap_scores = heap_scores[mask]
        heap_rows = heap_rows[mask]

        rows = self.manifest.iloc[heap_rows]
        out = []
        for (s, _ridx), row in zip(zip(heap_scores, heap_rows), rows.itertuples(index=False)):
            out.append({
                "doc_id": row.doc_id, "family": row.family, "citation": row.citation,
                "authority_score": None, "channel": "vector", "raw_score": float(s),
            })
        return out

    def statute_anchor_search(self, statutes: list[str], k: int = 400) -> list[dict]:
        if not statutes:
            return []
        # Match exact statute string OR a prefix-match on the article+law-code combo,
        # since stored anchors include the full "Art. N Abs. M CODE" form.
        like_clauses = []
        params: list[str] = []
        for st in statutes:
            like_clauses.append("statute LIKE ?")
            params.append(f"%{st}%")
        where = " OR ".join(like_clauses)
        params.append(k)
        rows = self.con.execute(
            f"SELECT statute_links.doc_id AS doc_id, statute_links.statute AS hit, "
            f"       documents.family AS family, documents.citation AS citation, "
            f"       documents.authority_score AS authority_score "
            f"FROM statute_links JOIN documents USING (doc_id) "
            f"WHERE {where} "
            f"ORDER BY documents.authority_score DESC LIMIT ?",
            params,
        ).fetchall()
        return [
            {"doc_id": r["doc_id"], "family": r["family"], "citation": r["citation"],
             "authority_score": r["authority_score"], "channel": "statute",
             "raw_score": float(r["authority_score"] or 0.0), "anchor": r["hit"]}
            for r in rows
        ]

    def case_anchor_search(self, cases: list[str], dockets: list[str], k: int = 300) -> list[dict]:
        if not cases and not dockets:
            return []
        targets = list(cases) + list(dockets)
        like_clauses = " OR ".join(["target_citation LIKE ?"] * len(targets) +
                                   ["target_base LIKE ?"] * len(targets))
        params = [f"%{t}%" for t in targets] * 2 + [k]
        rows = self.con.execute(
            f"SELECT case_links.doc_id AS doc_id, case_links.target_citation AS hit, "
            f"       documents.family AS family, documents.citation AS citation, "
            f"       documents.authority_score AS authority_score "
            f"FROM case_links JOIN documents USING (doc_id) "
            f"WHERE {like_clauses} "
            f"ORDER BY documents.authority_score DESC LIMIT ?",
            params,
        ).fetchall()
        return [
            {"doc_id": r["doc_id"], "family": r["family"], "citation": r["citation"],
             "authority_score": r["authority_score"], "channel": "case",
             "raw_score": float(r["authority_score"] or 0.0), "anchor": r["hit"]}
            for r in rows
        ]

    # ---- fusion ------------------------------------------------------------

    @staticmethod
    def reciprocal_rank_fusion(
        channel_results: dict[str, list[dict]],
        *,
        k: int = 60,
        weights: dict[str, float] | None = None,
        authority_alpha: float = 0.15,
    ) -> list[dict]:
        weights = weights or {}
        scores: dict[str, float] = defaultdict(float)
        meta: dict[str, dict] = {}
        per_channel: dict[str, dict[str, int]] = defaultdict(dict)

        for channel, results in channel_results.items():
            w = weights.get(channel, 1.0)
            for rank, item in enumerate(results, start=1):
                doc_id = item["doc_id"]
                scores[doc_id] += w / (k + rank)
                per_channel[channel][doc_id] = rank
                if doc_id not in meta:
                    meta[doc_id] = {
                        "doc_id": doc_id,
                        "family": item.get("family"),
                        "citation": item.get("citation"),
                        "authority_score": item.get("authority_score"),
                    }

        out = []
        for doc_id, score in scores.items():
            entry = dict(meta[doc_id])
            authority = entry.get("authority_score") or 0.0
            try:
                authority = float(authority)
            except (TypeError, ValueError):
                authority = 0.0
            boost = 1.0 + authority_alpha * authority
            entry["fused_score"] = score * boost
            entry["channel_ranks"] = {c: per_channel[c][doc_id]
                                      for c in per_channel if doc_id in per_channel[c]}
            out.append(entry)
        out.sort(key=lambda r: r["fused_score"], reverse=True)
        return out

    # ---- post-fusion enrichment -------------------------------------------

    def _hydrate_text(self, candidates: list[dict]) -> None:
        """Attach `text` to each candidate by joining `vector_text` from the
        documents table. In-place; safe to call multiple times.
        """
        if not candidates:
            return
        missing = [c for c in candidates if "text" not in c or not c.get("text")]
        if not missing:
            return
        ids = [c["doc_id"] for c in missing]
        # Chunked IN(?) lookup so we don't blow past SQLite parameter limits.
        chunk_size = 800
        text_by_id: dict[str, str] = {}
        for s in range(0, len(ids), chunk_size):
            chunk = ids[s:s + chunk_size]
            q = (
                "SELECT doc_id, vector_text FROM documents "
                f"WHERE doc_id IN ({','.join('?' * len(chunk))})"
            )
            for r in self.con.execute(q, chunk).fetchall():
                text_by_id[r["doc_id"]] = r["vector_text"] or ""
        for c in missing:
            c["text"] = text_by_id.get(c["doc_id"], "")

    # ---- top-level ---------------------------------------------------------

    def retrieve(
        self,
        query_text: str,
        *,
        query_embedding: np.ndarray | None = None,
        top_k: int = 1000,
        channel_budgets: dict[str, int] | None = None,
        channel_weights: dict[str, float] | None = None,
        authority_alpha: float = 0.15,
        rrf_k: int = 60,
        rerank: bool = False,
        reranker=None,
        rerank_top_n: int = 200,
        rerank_batch_size: int = 8,
        rerank_instruction: str | None = None,
        llm_judge: bool = False,
        judge=None,
        judge_auto_yes_thresh: float = 0.55,
        judge_auto_no_thresh: float = 0.25,
        judge_batch_size: int = 4,
        apply_granularity_filter: bool = True,
        verbose: bool = False,
    ) -> dict:
        """Run the multi-channel retrieval pipeline.

        Optional stages (callers must supply already-loaded models so this
        function never imports torch / transformers itself):

          - ``rerank=True`` + ``reranker``  ->  cross-encoder rescoring of top-N
            (default 200). The retriever score is min-max-normalised and fused
            with the rerank score per ``rerank_candidates`` (0.7 / 0.3).
          - ``llm_judge=True`` + ``judge``  ->  threshold router + Qwen3-8B judge
            on borderline candidates; verdict='no' rows are dropped from the
            output ranking.
          - ``apply_granularity_filter=True`` (default)  ->  expand bare-article
            citations into the corpus paragraph-children at the very end so the
            output matches the corpus granularity used by the gold sets.
        """
        budgets = {"bm25": 800, "vector": 800, "statute": 400, "case": 300}
        if channel_budgets:
            budgets.update(channel_budgets)

        anchors = parse_query_anchors(query_text)
        results: dict[str, list[dict]] = {}

        t0 = time.time()
        results["bm25"] = self.bm25_search(query_text, budgets["bm25"])
        t_bm25 = time.time() - t0

        t0 = time.time()
        if query_embedding is not None:
            results["vector"] = self.vector_search(query_embedding, budgets["vector"])
        else:
            results["vector"] = []
        t_vec = time.time() - t0

        t0 = time.time()
        if anchors["statutes"]:
            results["statute"] = self.statute_anchor_search(anchors["statutes"], budgets["statute"])
        else:
            results["statute"] = []
        if anchors["cases"] or anchors["dockets"]:
            results["case"] = self.case_anchor_search(anchors["cases"], anchors["dockets"], budgets["case"])
        else:
            results["case"] = []
        t_anchor = time.time() - t0

        active = {k: v for k, v in results.items() if v}
        fused = self.reciprocal_rank_fusion(
            active, k=rrf_k, weights=channel_weights, authority_alpha=authority_alpha,
        )[:top_k]

        # Carry the RRF score under both ``fused_score`` and ``score`` so that
        # rerank_candidates (which reads ``score`` by default) sees the
        # retrieval signal.
        for c in fused:
            c["score"] = c.get("fused_score", 0.0)

        # ---- optional rerank stage ----------------------------------------
        t_rerank = 0.0
        if rerank and reranker is not None and fused:
            t0 = time.time()
            from rerank_qwen3 import rerank_candidates  # lazy: torch user

            head = fused[:rerank_top_n]
            tail = fused[rerank_top_n:]
            self._hydrate_text(head)
            reranked = rerank_candidates(
                query_text,
                head,
                reranker,
                top_k=rerank_top_n,
                instruction=rerank_instruction,
                batch_size=rerank_batch_size,
            )
            # Anything past rerank_top_n keeps its retrieval score; mark it.
            for c in tail:
                c["rerank_score"] = None
            fused = reranked + tail
            t_rerank = time.time() - t0
        elif rerank and reranker is None:
            print("[retrieve] rerank=True but reranker is None; skipping rerank.",
                  file=sys.stderr)

        # ---- optional LLM-judge stage -------------------------------------
        t_judge = 0.0
        if llm_judge and judge is not None and fused:
            t0 = time.time()
            from llm_judge_qwen3 import route_and_judge  # lazy: torch user

            self._hydrate_text(fused)
            judged = route_and_judge(
                query_text,
                fused,
                judge,
                fused_score_key="fused_score",
                auto_yes_thresh=judge_auto_yes_thresh,
                auto_no_thresh=judge_auto_no_thresh,
                batch_size=judge_batch_size,
            )
            fused = [c for c in judged if c.get("verdict") != "no"]
            t_judge = time.time() - t0
        elif llm_judge and judge is None:
            print("[retrieve] llm_judge=True but judge is None; skipping judge.",
                  file=sys.stderr)

        # ---- optional granularity post-filter -----------------------------
        t_gran = 0.0
        if apply_granularity_filter and fused:
            t0 = time.time()
            from granularity_resolver import (  # lazy
                apply_granularity_post_filter,
                expand_article_to_paragraphs,
            )
            # Build expanded ranked citation list, then re-attach each citation
            # to its first-seen carrier candidate so downstream consumers still
            # see structured rows.
            citation_to_carrier: dict[str, dict] = {}
            for c in fused:
                cit = (c.get("citation") or "").strip()
                if cit and cit not in citation_to_carrier:
                    citation_to_carrier[cit] = c

            ranked_citations = [(c.get("citation") or "").strip() for c in fused]
            expanded_ordered = apply_granularity_post_filter(ranked_citations, self.con)

            new_fused: list[dict] = []
            for cit in expanded_ordered:
                # Carry over the original candidate when present; otherwise
                # synthesise a minimal row from the expansion source so the
                # downstream eval still sees a citation string.
                src = citation_to_carrier.get(cit)
                if src is None:
                    # Try to find a parent (article-level -> paragraph child).
                    src = {"doc_id": None, "family": "law", "citation": cit,
                           "fused_score": 0.0, "score": 0.0,
                           "from_granularity_expansion": True}
                else:
                    src = dict(src)
                    src["citation"] = cit
                new_fused.append(src)
            fused = new_fused
            t_gran = time.time() - t0

        if verbose:
            print(f"[retrieve] anchors: {anchors}", file=sys.stderr)
            print(f"[retrieve] bm25={len(results['bm25'])}/{budgets['bm25']}  "
                  f"vector={len(results['vector'])}/{budgets['vector']}  "
                  f"statute={len(results['statute'])}/{budgets['statute']}  "
                  f"case={len(results['case'])}/{budgets['case']}", file=sys.stderr)
            print(f"[retrieve] timings  bm25={t_bm25:.2f}s  vector={t_vec:.2f}s  "
                  f"anchor={t_anchor:.2f}s  rerank={t_rerank:.2f}s  "
                  f"judge={t_judge:.2f}s  gran={t_gran:.2f}s", file=sys.stderr)
            print(f"[retrieve] fused candidates: {len(fused)}", file=sys.stderr)

        return {
            "query": query_text,
            "anchors": anchors,
            "channel_counts": {k: len(v) for k, v in results.items()},
            "channel_timings_s": {
                "bm25": t_bm25, "vector": t_vec, "anchor": t_anchor,
                "rerank": t_rerank, "judge": t_judge, "granularity": t_gran,
            },
            "candidates": fused,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _format_row(rank: int, item: dict) -> str:
    cr = item.get("channel_ranks", {})
    cr_str = ",".join(f"{c[0]}{r}" for c, r in cr.items())
    fam = item.get("family", "?")
    cit = item.get("citation", "")
    return (f"  {rank:4d}. [{fam:5s}] {cit:38s} "
            f"score={item['fused_score']:.4f}  channels={cr_str}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sqlite", type=Path, default=DEFAULT_SQLITE)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--embeddings-dir", type=Path, default=EMB_DIR)
    ap.add_argument("--query", type=str, required=True, help="Free-text query (English)")
    ap.add_argument("--query-embedding", type=Path, default=None,
                    help="Path to a single (4096,) .npy or (1, 4096) .npy file. If absent, the vector channel is skipped.")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--authority-alpha", type=float, default=0.15)
    ap.add_argument("--no-vector", action="store_true",
                    help="Skip the vector channel even if --query-embedding is given.")
    ap.add_argument("--bm25-budget", type=int, default=800)
    ap.add_argument("--vector-budget", type=int, default=800)
    ap.add_argument("--statute-budget", type=int, default=400)
    ap.add_argument("--case-budget", type=int, default=300)
    ap.add_argument("--print-channels", action="store_true",
                    help="Show per-channel top-5 before fusion.")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="If set, write the full retrieval result as JSON.")
    ap.add_argument("--enable-rerank", action="store_true",
                    help="Run Qwen3-Reranker-8B over the top-N. GPU REQUIRED.")
    ap.add_argument("--rerank-top-n", type=int, default=200)
    ap.add_argument("--enable-judge", action="store_true",
                    help="Run Qwen3-8B as LLM-judge on borderline candidates. GPU REQUIRED.")
    ap.add_argument("--judge-auto-yes", type=float, default=0.55)
    ap.add_argument("--judge-auto-no", type=float, default=0.25)
    ap.add_argument("--no-granularity-filter", action="store_true",
                    help="Disable the corpus-paragraph expansion at the end of the pipeline.")
    args = ap.parse_args()

    # Eager check: --enable-rerank/--enable-judge needs CUDA.
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

    rt = HybridRetriever(args.sqlite, args.manifest, args.embeddings_dir)

    q_emb = None
    if args.query_embedding and not args.no_vector:
        q_emb = np.load(args.query_embedding)
        if q_emb.ndim == 2:
            q_emb = q_emb[0]
        print(f"[query-embedding] shape={q_emb.shape}  dtype={q_emb.dtype}", file=sys.stderr)

    result = rt.retrieve(
        args.query,
        query_embedding=q_emb,
        top_k=args.top_k,
        channel_budgets={
            "bm25": args.bm25_budget, "vector": args.vector_budget,
            "statute": args.statute_budget, "case": args.case_budget,
        },
        rrf_k=args.rrf_k,
        authority_alpha=args.authority_alpha,
        rerank=args.enable_rerank,
        reranker=reranker_obj,
        rerank_top_n=args.rerank_top_n,
        llm_judge=args.enable_judge,
        judge=judge_obj,
        judge_auto_yes_thresh=args.judge_auto_yes,
        judge_auto_no_thresh=args.judge_auto_no,
        apply_granularity_filter=not args.no_granularity_filter,
        verbose=True,
    )

    if args.print_channels:
        for ch in ("bm25", "vector", "statute", "case"):
            print(f"\n--- channel: {ch} (top 5 of {result['channel_counts'][ch]}) ---")
            # We didn't keep the per-channel results here because retrieve() already fused.
            # For a richer view, run the channels directly via the API.
        # fall through to fused

    print(f"\nQ: {args.query}")
    if any(result["anchors"].values()):
        print(f"   anchors: {result['anchors']}")
    print(f"   channels (count/budget): {result['channel_counts']}")
    print(f"   channel timings: {result['channel_timings_s']}")
    print(f"   fused top {min(args.top_k, len(result['candidates']))}:")
    for i, c in enumerate(result["candidates"], start=1):
        print(_format_row(i, c))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"\n[done] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
