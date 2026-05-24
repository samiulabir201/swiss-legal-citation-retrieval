#!/usr/bin/env python
"""Add date-stripped alias edges to the citation graph.

The original `extract_citation_graph.py` canonicalizes a docket-style citation
that appears in text as e.g. "1B_210/2023 vom 12. Mai 2023 E. 3" into
    "1B_210/2023 12.05.2023 E. 3"
(date included). However, the val/test gold and the dataset's source-row
citations use the un-dated form
    "1B_210/2023 E. 3"
This canonicalization mismatch means edges land on the dated form, while
gold lookups query the un-dated form, producing zero in-edges for many gold
items even though the corpus DOES cite them.

This pass scans existing edges, finds those whose target matches the dated
docket+E pattern, and adds an alias edge whose target is the un-dated form
— but only when that un-dated form already exists as a citation row in the
DB (so we never invent unmatched targets).
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path


DATED_DOCKET_RE = re.compile(
    # Modern docket: 1B_210/2023, 5A.123/2020, with underscore/dot/space separator.
    r'^(\d{1,2}[A-Z]{1,2}[_\. ]\d{1,5}/\d{4})\s+'
    r'\d{1,2}\.\d{1,2}\.\d{4}\s+'
    r'(E\.\s+\S+)$'
)
LEGACY_DATED_RE = re.compile(
    # Legacy docket: "I 402/02"
    r'^([A-Z]\s+\d{1,5}/\d{2})\s+'
    r'\d{1,2}\.\d{1,2}\.\d{4}\s+'
    r'(E\.\s+\S+)$'
)
# Docket+date with no E. (rare, but keep for completeness)
DATED_DOCKET_NO_E_RE = re.compile(
    r'^(\d{1,2}[A-Z]{1,2}[_\. ]\d{1,5}/\d{4})\s+'
    r'\d{1,2}\.\d{1,2}\.\d{4}$'
)
LEGACY_DATED_NO_E_RE = re.compile(
    r'^([A-Z]\s+\d{1,5}/\d{2})\s+'
    r'\d{1,2}\.\d{1,2}\.\d{4}$'
)


def strip_date(citation: str) -> str | None:
    """Return the un-dated form of `citation` if it matches a dated docket
    pattern, else None.
    """
    m = DATED_DOCKET_RE.match(citation)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    m = LEGACY_DATED_RE.match(citation)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    m = DATED_DOCKET_NO_E_RE.match(citation)
    if m:
        return m.group(1)
    m = LEGACY_DATED_NO_E_RE.match(citation)
    if m:
        return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

SELF_TESTS = [
    ("1B_210/2023 12.05.2023 E. 3",        "1B_210/2023 E. 3"),
    ("1B_357/2022 22.07.2022 E. 3.1",      "1B_357/2022 E. 3.1"),
    ("2P.198/2006 09.05.2007 E. 2.4",      "2P.198/2006 E. 2.4"),
    ("I 402/02 13.11.2002 E. 4",           "I 402/02 E. 4"),
    ("1B_90/2021 18.03.2021",              "1B_90/2021"),
    # Negative: not a dated form
    ("BGE 137 IV 122 E. 6.4",              None),
    ("Art. 221 Abs. 1 StPO",               None),
    ("1B_210/2023 E. 3",                   None),
]


def run_self_tests() -> None:
    failures = []
    for inp, expected in SELF_TESTS:
        got = strip_date(inp)
        if got != expected:
            failures.append((inp, expected, got))
    if failures:
        print("SELF-TEST FAILURES:")
        for inp, exp, got in failures:
            print(f"  {inp!r}  expected={exp!r}  got={got!r}")
        sys.exit(1)
    print(f"OK: {len(SELF_TESTS)} self-tests passed.")


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

def add_aliases(db_path: Path) -> dict:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")

    t0 = time.time()
    print("Preloading citations table into memory for existence check...", flush=True)
    all_cits: set[str] = set()
    for (cit,) in conn.execute(
        "SELECT citation FROM citations WHERE dataset='court_considerations'"
    ):
        all_cits.add(cit)
    print(f"  {len(all_cits):,} citations loaded ({time.time()-t0:.1f}s)\n", flush=True)

    pre = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE dataset='court_considerations'"
    ).fetchone()[0]

    print("Streaming edges and computing date-stripped alias targets...", flush=True)
    t1 = time.time()
    cursor = conn.cursor()
    n_edges_scanned = 0
    n_dated_targets = 0
    n_alias_candidates = 0
    n_skipped_no_strip_row = 0
    pending: list[tuple[str, str, str]] = []
    BATCH = 100_000

    for source, target in conn.execute(
        "SELECT source, target FROM edges WHERE dataset='court_considerations'"
    ):
        n_edges_scanned += 1
        if n_edges_scanned % 1_000_000 == 0:
            print(
                f"  scanned {n_edges_scanned:,} edges "
                f"({n_dated_targets:,} dated targets, {n_alias_candidates:,} alias candidates) "
                f"[{time.time()-t1:.1f}s]",
                flush=True,
            )
        stripped = strip_date(target)
        if stripped is None:
            continue
        n_dated_targets += 1
        # Only add the alias if the stripped form already exists as a citation row.
        # This prevents inventing new graph nodes that nothing in the corpus
        # actually points at.
        if stripped not in all_cits:
            n_skipped_no_strip_row += 1
            continue
        # Skip self-edge
        if stripped == source:
            continue
        n_alias_candidates += 1
        pending.append(("court_considerations", source, stripped))

        if len(pending) >= BATCH:
            cursor.executemany(
                "INSERT OR IGNORE INTO edges (dataset, source, target) VALUES (?, ?, ?)",
                pending,
            )
            conn.commit()
            pending.clear()

    if pending:
        cursor.executemany(
            "INSERT OR IGNORE INTO edges (dataset, source, target) VALUES (?, ?, ?)",
            pending,
        )
        conn.commit()

    post = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE dataset='court_considerations'"
    ).fetchone()[0]

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.1f}s. "
        f"edges_scanned={n_edges_scanned:,}, "
        f"dated_targets={n_dated_targets:,}, "
        f"alias_candidates={n_alias_candidates:,}, "
        f"skipped (no row for stripped form)={n_skipped_no_strip_row:,}",
        flush=True,
    )
    print(
        f"  edges_before={pre:,}, edges_after={post:,}, net_new={post - pre:,}",
        flush=True,
    )
    conn.close()
    return {
        "edges_before": pre,
        "edges_after": post,
        "net_new": post - pre,
        "edges_scanned": n_edges_scanned,
        "dated_targets": n_dated_targets,
        "alias_candidates": n_alias_candidates,
        "skipped_no_strip_row": n_skipped_no_strip_row,
        "elapsed_s": elapsed,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--db-path",
        type=Path,
        default=Path(r"E:\swiss_citation_extraction\data_insights\citation_graph_extracted.sqlite"),
    )
    p.add_argument("--self-test-only", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    run_self_tests()
    if args.self_test_only:
        return 0
    if not args.db_path.exists():
        print(f"DB not found: {args.db_path}", file=sys.stderr)
        return 2
    add_aliases(args.db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
