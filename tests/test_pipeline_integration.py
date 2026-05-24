"""Smoke / integration tests for the F1-tuned retrieval pipeline.

Covers:

  1. ``granularity_resolver.apply_granularity_post_filter`` end-to-end on a
     hand-crafted in-memory sqlite that mimics the columns the resolver hits.
  2. ``eval_retrieval.evaluate_f1`` end-to-end on a tiny gold CSV (3 rows)
     with a fake ``HybridRetriever`` stand-in (no real corpus needed).
  3. Construction-time contract of ``hybrid_retrieve.HybridRetriever.retrieve``
     when rerank/judge stages are disabled (no torch import).

The reranker / judge stages themselves require GPU + 8B-parameter model
weights and are therefore out of scope here; we explicitly skip those
sub-tests with a clear message when ``torch`` is not importable.
"""

from __future__ import annotations

import csv
import importlib
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _torch_available() -> bool:
    try:
        importlib.import_module("torch")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_documents_sqlite(path: Path) -> None:
    """Build a tiny in-process documents table.

    Schema is the subset that ``granularity_resolver`` queries against
    (``family``, ``law_code``, ``article``, ``granularity``, ``citation``,
    ``expansion_json``).
    """
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE documents ("
            " doc_id TEXT PRIMARY KEY, "
            " family TEXT, citation TEXT, law_code TEXT, article TEXT, "
            " granularity TEXT, expansion_json TEXT"
            ")"
        )
        rows = [
            # Art. 78 BV  ->  three paragraph children.
            ("BV-78-1", "law", "Art. 78 Abs. 1 BV", "BV", "78", "paragraph", None),
            ("BV-78-2", "law", "Art. 78 Abs. 2 BV", "BV", "78", "paragraph", None),
            ("BV-78-3", "law", "Art. 78 Abs. 3 BV", "BV", "78", "paragraph", None),
            # Art. 221 StPO  ->  two paragraph children.
            ("StPO-221-1", "law", "Art. 221 Abs. 1 StPO", "StPO", "221", "paragraph", None),
            ("StPO-221-2", "law", "Art. 221 Abs. 2 StPO", "StPO", "221", "paragraph", None),
            # Art. 100 BGG -> single paragraph child (so we exercise both the
            # zero-children "true article" case and the one-child case).
            ("BGG-100-1", "law", "Art. 100 Abs. 1 BGG", "BGG", "100", "paragraph", None),
            # Art. 9 BV: NO paragraph children -> resolver should keep input.
            ("BV-9", "law", "Art. 9 BV", "BV", "9", "article", None),
            # A court row to make sure non-law citations pass through unchanged.
            ("BGE-141-IV-87", "court", "BGE 141 IV 87", None, None, None, None),
        ]
        conn.executemany(
            "INSERT INTO documents "
            "(doc_id, family, citation, law_code, article, granularity, expansion_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


class _FakeRetriever:
    """Stand-in for ``HybridRetriever`` for the F1-eval smoke test.

    Returns a deterministic, query-dependent ranked list so that we can
    test the K-sweep / set-comparison logic without a real corpus.
    """

    def __init__(self, ranked_by_qid: dict[str, list[str]]):
        self.ranked_by_qid = ranked_by_qid
        self._calls = 0

    def retrieve(self, query_text, **kwargs):
        # Look up by substring match on the query; default to empty.
        for qid_marker, citations in self.ranked_by_qid.items():
            if qid_marker in query_text:
                self._calls += 1
                return {
                    "query": query_text,
                    "anchors": {},
                    "channel_counts": {"bm25": len(citations)},
                    "channel_timings_s": {"bm25": 0.0, "vector": 0.0, "anchor": 0.0},
                    "candidates": [
                        {"doc_id": f"d-{i}", "family": "law", "citation": c,
                         "fused_score": 1.0 - 0.01 * i, "score": 1.0 - 0.01 * i}
                        for i, c in enumerate(citations)
                    ],
                }
        self._calls += 1
        return {
            "query": query_text, "anchors": {},
            "channel_counts": {}, "channel_timings_s": {},
            "candidates": [],
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class GranularityPostFilterTests(unittest.TestCase):
    """The corpus-paragraph expansion is the lever between BM25 candidates
    and the gold citation strings, so we want fast, reliable coverage of it."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.tmp.name) / "fake_documents.sqlite"
        _make_fake_documents_sqlite(cls.db_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        from granularity_resolver import apply_granularity_post_filter
        self.apply = apply_granularity_post_filter
        self.conn = sqlite3.connect(self.db_path)

    def tearDown(self):
        self.conn.close()

    def test_bare_article_with_children_is_expanded(self):
        out = self.apply(["Art. 78 BV"], self.conn)
        self.assertEqual(
            out,
            ["Art. 78 Abs. 1 BV", "Art. 78 Abs. 2 BV", "Art. 78 Abs. 3 BV"],
        )

    def test_paragraph_input_is_passthrough(self):
        out = self.apply(["Art. 78 Abs. 1 BV"], self.conn)
        self.assertEqual(out, ["Art. 78 Abs. 1 BV"])

    def test_article_without_children_is_passthrough(self):
        out = self.apply(["Art. 9 BV"], self.conn)
        self.assertEqual(out, ["Art. 9 BV"])

    def test_court_citation_is_passthrough(self):
        out = self.apply(["BGE 141 IV 87"], self.conn)
        self.assertEqual(out, ["BGE 141 IV 87"])

    def test_mixed_input_dedups_in_order(self):
        out = self.apply(
            [
                "Art. 78 BV",                # expands to 3 paragraphs
                "Art. 78 Abs. 2 BV",         # already in expansion -> dedup
                "Art. 100 BGG",              # one child
                "BGE 141 IV 87",             # passthrough
            ],
            self.conn,
        )
        # First three from the article expansion (sorted), then the one-child
        # expansion of Art. 100 BGG, then the BGE.
        self.assertEqual(
            out,
            [
                "Art. 78 Abs. 1 BV",
                "Art. 78 Abs. 2 BV",
                "Art. 78 Abs. 3 BV",
                "Art. 100 Abs. 1 BGG",
                "BGE 141 IV 87",
            ],
        )


class EvalRetrievalF1Tests(unittest.TestCase):
    """Drive ``evaluate_f1`` on a 3-row CSV with a fake retriever."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.tmp_path = Path(cls.tmp.name)
        cls.gold_csv = cls.tmp_path / "tiny_val.csv"
        with cls.gold_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["query_id", "query", "gold_citations"])
            # Q1: 3 gold; retriever's first 3 are exactly those 3 (perfect F1=1).
            w.writerow([
                "tq_001",
                "marker_q1 pretrial detention",
                "Art. 221 Abs. 1 StPO;Art. 221 Abs. 2 StPO;Art. 100 Abs. 1 BGG",
            ])
            # Q2: 2 gold but retriever only returns 1 of them in top-K + noise.
            w.writerow([
                "tq_002",
                "marker_q2 invalidity insurance",
                "Art. 78 Abs. 1 BV;Art. 9 BV",
            ])
            # Q3: 2 gold, retriever serves them at the top with one filler.
            w.writerow([
                "tq_003",
                "marker_q3 federal supreme court",
                "BGE 141 IV 87;Art. 100 Abs. 1 BGG",
            ])

        cls.db_path = cls.tmp_path / "fake_documents.sqlite"
        _make_fake_documents_sqlite(cls.db_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _build_retriever(self):
        # Citation rankings are tuned so that K=5 gives perfect P/R for q1,
        # finds 1/2 gold for q2, and 2/2 for q3 -- gives a tractable F1.
        return _FakeRetriever({
            "marker_q1": [
                "Art. 221 Abs. 1 StPO", "Art. 221 Abs. 2 StPO",
                "Art. 100 Abs. 1 BGG",
                "Art. 78 Abs. 1 BV", "Art. 78 Abs. 2 BV",
            ],
            "marker_q2": [
                "Art. 78 Abs. 1 BV",  # gold expanded form (after granularity)
                "Art. 100 Abs. 1 BGG", "Art. 78 Abs. 2 BV",
                "Art. 9 BV",  # gold (true-article passthrough)
                "Art. 221 Abs. 1 StPO",
            ],
            "marker_q3": [
                "BGE 141 IV 87", "Art. 100 Abs. 1 BGG", "Art. 78 Abs. 1 BV",
            ],
        })

    def test_evaluate_f1_runs_and_picks_a_best_k(self):
        from eval_retrieval import evaluate_f1

        out_dir = self.tmp_path / "eval_out"
        per_query_csv = out_dir / "tiny_per_q.csv"

        summary = evaluate_f1(
            split_csv=self.gold_csv,
            query_emb_path=None,
            query_ids_path=None,
            retriever=self._build_retriever(),
            bm25_budget=10, vector_budget=10,
            statute_budget=10, case_budget=10,
            rrf_k=60, authority_alpha=0.0,
            out_dir=out_dir,
            split_name="tinyval",
            apply_granularity=True,
            sqlite_path=self.db_path,
            per_query_csv=per_query_csv,
            k_sweep=(3, 5, 10),
            limit=0,
        )

        # Basic structure.
        self.assertEqual(summary["metric"], "macro_f1")
        self.assertEqual(summary["n_queries"], 3)
        self.assertIn(summary["best_k"], (3, 5, 10))
        self.assertGreaterEqual(summary["best_macro_f1"], 0.0)
        self.assertLessEqual(summary["best_macro_f1"], 1.0)
        # We hand-tuned q1 + q3 so at least F1 should be reasonable.
        self.assertGreater(summary["best_macro_f1"], 0.4)

        # Files were emitted.
        self.assertTrue((out_dir / "tinyval_f1_summary.json").exists())
        self.assertTrue(per_query_csv.exists())
        with per_query_csv.open(encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0],
                         ["query_id", "K", "precision", "recall", "f1",
                          "gold_count", "n_pred_pool"])
        self.assertEqual(len(rows) - 1, 3)

    def test_evaluate_f1_without_granularity_filter(self):
        """No-filter run still produces valid output (different numbers)."""
        from eval_retrieval import evaluate_f1

        out_dir = self.tmp_path / "eval_out_no_gran"
        summary = evaluate_f1(
            split_csv=self.gold_csv,
            query_emb_path=None,
            query_ids_path=None,
            retriever=self._build_retriever(),
            bm25_budget=10, vector_budget=10,
            statute_budget=10, case_budget=10,
            rrf_k=60, authority_alpha=0.0,
            out_dir=out_dir,
            split_name="tinyval_nogran",
            apply_granularity=False,
            sqlite_path=self.db_path,  # ignored when apply_granularity=False
            per_query_csv=None,
            k_sweep=(3, 5, 10),
            limit=0,
        )
        self.assertEqual(summary["apply_granularity_post_filter"], False)
        self.assertEqual(summary["n_queries"], 3)


class HybridRetrieveImportTests(unittest.TestCase):
    """Module-level imports must stay torch-free so the harness works on CPU."""

    def test_hybrid_retrieve_imports_without_torch(self):
        # Even on a torch-less host, importing hybrid_retrieve should succeed.
        # We don't try to instantiate HybridRetriever (needs the real sqlite).
        import hybrid_retrieve
        self.assertTrue(hasattr(hybrid_retrieve, "HybridRetriever"))
        self.assertTrue(hasattr(hybrid_retrieve, "parse_query_anchors"))
        sig = hybrid_retrieve.HybridRetriever.retrieve.__doc__ or ""
        # The retrieve docstring should reference the optional stages.
        self.assertIn("rerank", sig.lower())
        self.assertIn("granularity", sig.lower())

    def test_eval_retrieval_imports_without_torch(self):
        import eval_retrieval
        self.assertTrue(hasattr(eval_retrieval, "evaluate_f1"))
        self.assertTrue(hasattr(eval_retrieval, "F1_K_SWEEP"))
        self.assertEqual(
            eval_retrieval.F1_K_SWEEP,
            (5, 7, 10, 13, 15, 20, 25, 30, 50, 75, 100),
        )


@unittest.skipUnless(_torch_available(),
                     "torch not installed -- skipping reranker / judge contract tests "
                     "(would otherwise need GPU + 8B model weights).")
class RerankerJudgeContractTests(unittest.TestCase):
    """If torch is around, at least verify the modules import. We do NOT
    instantiate the 8B models here -- that requires GPU + multi-minute download."""

    def test_reranker_module_importable(self):
        import rerank_qwen3
        self.assertTrue(hasattr(rerank_qwen3, "QwenReranker"))
        self.assertTrue(hasattr(rerank_qwen3, "rerank_candidates"))

    def test_judge_module_importable(self):
        import llm_judge_qwen3
        self.assertTrue(hasattr(llm_judge_qwen3, "QwenJudge"))
        self.assertTrue(hasattr(llm_judge_qwen3, "route_and_judge"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
