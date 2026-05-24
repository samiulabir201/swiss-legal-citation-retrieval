#!/usr/bin/env python
"""Add E.-range expanded alias edges to the citation graph.

Source-text patterns like
    "1B_90/2021 vom 18.03.2021 E. 2.1-2.4"
get extracted by the original `extract_citation_graph.py` as a single citation
    "1B_90/2021 18.03.2021 E. 2.1-2.4"
But the val/test gold uses the un-ranged, date-stripped form
    "1B_90/2021 E. 2.1"   "1B_90/2021 E. 2.2"   ...   "1B_90/2021 E. 2.4"

This pass scans existing edges, finds those whose target embeds an "E. N1-N2"
range (with or without a date prefix), enumerates the range into individual
E.-pinpoints, and adds alias edges to each enumerated form — but only when
that form exists as a citation row (so we never invent unmatched targets).

It composes with `add_date_stripped_aliases.py`: this script handles ALL
range targets, dated or not, so it does not need the date-strip alias pass
to run first.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path


# Match: prefix + E. + first + - + second
# prefix can include a docket and optional date.
# Example matches:
#   "1B_90/2021 18.03.2021 E. 2.1-2.4"      -> prefix="1B_90/2021 18.03.2021 ", start="2.1", end="2.4"
#   "1B_90/2021 E. 2.1-2.4"                 -> prefix="1B_90/2021 ", start="2.1", end="2.4"
#   "BGE 137 IV 122 E. 4-6"                 -> prefix="BGE 137 IV 122 ", start="4", end="6"
RANGE_RE = re.compile(
    r'^(?P<prefix>.+?\s+)E\.\s+(?P<start>\d+(?:\.\d+)*)\s*-\s*(?P<end>\d+(?:\.\d+)*)$'
)
# Date pattern at the start of prefix — if present we'll also emit un-dated aliases.
DATE_PREFIX_RE = re.compile(
    r'^(?P<docket>.+?)\s+\d{1,2}\.\d{1,2}\.\d{4}\s+$'
)


def _parse_parts(numstr: str) -> list[int]:
    return [int(p) for p in numstr.split('.')]


def expand_range_target(citation: str) -> list[str]:
    """If `citation` embeds an E. range, return the list of un-ranged targets
    (covering the range AND, if a date was present, also the date-stripped
    versions of the same range). The original citation is NOT included.

    Returns [] if `citation` does not match the range pattern, or if the
    range cannot be safely enumerated (e.g. start/end at different parents).
    """
    m = RANGE_RE.match(citation)
    if not m:
        return []
    prefix = m.group("prefix")
    start = m.group("start")
    end = m.group("end")
    s_parts = _parse_parts(start)
    e_parts = _parse_parts(end)
    if len(s_parts) != len(e_parts):
        return []
    # Only safe to enumerate if all but the last component are identical
    # (e.g. "2.1-2.4" → 2.1, 2.2, 2.3, 2.4; but "2.1-3.4" is ambiguous).
    if s_parts[:-1] != e_parts[:-1]:
        return []
    lo, hi = s_parts[-1], e_parts[-1]
    if lo > hi or hi - lo > 50:  # safety cap
        return []
    refs = []
    base_parts = s_parts[:-1]
    for i in range(lo, hi + 1):
        new = base_parts + [i]
        ref_str = '.'.join(str(p) for p in new)
        refs.append(f"{prefix}E. {ref_str}")
    # Also add date-stripped versions if prefix has a date.
    dm = DATE_PREFIX_RE.match(prefix)
    if dm:
        bare_prefix = dm.group("docket") + ' '
        for i in range(lo, hi + 1):
            new = base_parts + [i]
            ref_str = '.'.join(str(p) for p in new)
            refs.append(f"{bare_prefix}E. {ref_str}")
    return refs


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

SELF_TESTS = [
    # (input, expected_outputs)
    (
        "1B_90/2021 18.03.2021 E. 2.1-2.4",
        [
            "1B_90/2021 18.03.2021 E. 2.1",
            "1B_90/2021 18.03.2021 E. 2.2",
            "1B_90/2021 18.03.2021 E. 2.3",
            "1B_90/2021 18.03.2021 E. 2.4",
            # date-stripped variants
            "1B_90/2021 E. 2.1",
            "1B_90/2021 E. 2.2",
            "1B_90/2021 E. 2.3",
            "1B_90/2021 E. 2.4",
        ],
    ),
    (
        "BGE 137 IV 122 E. 4-6",
        [
            "BGE 137 IV 122 E. 4",
            "BGE 137 IV 122 E. 5",
            "BGE 137 IV 122 E. 6",
        ],
    ),
    (
        "1B_357/2022 22.07.2022 E. 3.3.1-3.3.4",
        [
            "1B_357/2022 22.07.2022 E. 3.3.1",
            "1B_357/2022 22.07.2022 E. 3.3.2",
            "1B_357/2022 22.07.2022 E. 3.3.3",
            "1B_357/2022 22.07.2022 E. 3.3.4",
            "1B_357/2022 E. 3.3.1",
            "1B_357/2022 E. 3.3.2",
            "1B_357/2022 E. 3.3.3",
            "1B_357/2022 E. 3.3.4",
        ],
    ),
    # Ambiguous range — different parents — must NOT enumerate
    ("BGE 137 IV 122 E. 2.1-3.4", []),
    # Not a range
    ("1B_90/2021 E. 2.1", []),
    ("BGE 137 IV 122 E. 4.2", []),
    ("Art. 221 Abs. 1 StPO", []),
]


def run_self_tests() -> None:
    failures = []
    for inp, expected in SELF_TESTS:
        got = expand_range_target(inp)
        if set(got) != set(expected):
            failures.append((inp, expected, got))
    if failures:
        print("SELF-TEST FAILURES:")
        for inp, exp, got in failures:
            print(f"  {inp!r}")
            print(f"    expected={sorted(exp)}")
            print(f"    got     ={sorted(got)}")
        sys.exit(1)
    print(f"OK: {len(SELF_TESTS)} self-tests passed.")


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

def add_range_aliases(db_path: Path) -> dict:
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

    print("Streaming edges and computing range-expanded alias targets...", flush=True)
    t1 = time.time()
    cursor = conn.cursor()
    n_edges_scanned = 0
    n_range_targets = 0
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
                f"({n_range_targets:,} range targets, {n_alias_candidates:,} alias candidates) "
                f"[{time.time()-t1:.1f}s]",
                flush=True,
            )
        expansions = expand_range_target(target)
        if not expansions:
            continue
        n_range_targets += 1
        for exp in expansions:
            if exp not in all_cits:
                n_skipped_no_strip_row += 1
                continue
            if exp == source:
                continue
            n_alias_candidates += 1
            pending.append(("court_considerations", source, exp))

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
        f"range_targets={n_range_targets:,}, "
        f"alias_candidates={n_alias_candidates:,}, "
        f"skipped (no row for expanded form)={n_skipped_no_strip_row:,}",
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
    add_range_aliases(args.db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
