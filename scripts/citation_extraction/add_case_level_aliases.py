#!/usr/bin/env python
"""Add case-level fan-out alias edges to the citation graph.

Background: every Swiss judgment is split into many considerations (Erwägungen,
indexed as E. 1, E. 2.1, E. 2.2, ...). Each E. is its own row in
`court_considerations.csv`. When another decision cites the judgment, it
typically pinpoints ONE specific E. (e.g. "BGE 137 IV 122 E. 3.2"), but the
gold-citation answer expected by the val/test set may include OTHER Es of
the same judgment (e.g. "BGE 137 IV 122 E. 6.4") that aren't directly text-
cited by anyone in the corpus. Strictly text-derived edges leave those
"siblings" as orphans even though the case as a whole IS referenced.

This pass closes the gap by case-level fan-out: for every existing edge
whose target is a court citation of the form `BASE E. X`, we also add
alias edges from the same source to every OTHER `BASE E. Y` that exists
as a citation row in the DB. The semantics: citing one E. of a judgment
implies the judgment is relevant; therefore the source's pool should reach
all of that judgment's existing Es.

Source-row citations only — we do NOT fan out to synthetic backref nodes
or to non-existent Es. This keeps every emitted edge anchored to a real
corpus row.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path


# Split "BASE E. X" into ("BASE", "X"). Date-stripped first if present.
# Examples:
#   "BGE 137 IV 122 E. 4.2"          -> ("BGE 137 IV 122", "4.2")
#   "1B_357/2022 E. 3.1"             -> ("1B_357/2022", "3.1")
#   "1B_357/2022 22.07.2022 E. 3.1"  -> ("1B_357/2022 22.07.2022", "3.1")
#   "BGE 137 IV 122 S. 128"          -> None (S. = page, not consideration)
SPLIT_E_RE = re.compile(r'^(?P<base>.+?)\s+E\.\s+(?P<e>[A-Za-z0-9./-]+)$')
DATE_IN_BASE_RE = re.compile(
    r'^(?P<docket>.+?)\s+(?P<date>\d{1,2}\.\d{1,2}\.\d{4})$'
)


def split_court_citation(citation: str) -> tuple[str, str] | None:
    """Return (base, e_pinpoint) for a court citation of form `BASE E. X`,
    or None if the citation isn't a court+E form. Does not fan out S. (page),
    Bst., or other non-E pinpoints.
    """
    m = SPLIT_E_RE.match(citation)
    if not m:
        return None
    base = m.group("base").strip()
    e_pinpoint = m.group("e").strip()
    return base, e_pinpoint


def normalize_base(base: str) -> str:
    """Strip an embedded date from a docket base so 1B_X/Y dated and undated
    forms collapse to one judgment key. BGE bases stay unchanged.
    """
    m = DATE_IN_BASE_RE.match(base)
    if m:
        return m.group("docket").strip()
    return base


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

SELF_TESTS = [
    # split_court_citation
    ("split", "BGE 137 IV 122 E. 4.2",          ("BGE 137 IV 122", "4.2")),
    ("split", "1B_357/2022 E. 3.1",             ("1B_357/2022", "3.1")),
    ("split", "1B_357/2022 22.07.2022 E. 3.1",  ("1B_357/2022 22.07.2022", "3.1")),
    ("split", "BGE 137 IV 122 S. 128",          None),
    ("split", "BGE 137 IV 122",                 None),
    ("split", "Art. 221 Abs. 1 StPO",           None),
    # normalize_base
    ("nbase", "1B_357/2022 22.07.2022",         "1B_357/2022"),
    ("nbase", "BGE 137 IV 122",                 "BGE 137 IV 122"),
    ("nbase", "1B_357/2022",                    "1B_357/2022"),
]


def run_self_tests() -> None:
    failures = []
    for kind, inp, expected in SELF_TESTS:
        if kind == "split":
            got = split_court_citation(inp)
        elif kind == "nbase":
            got = normalize_base(inp)
        else:
            continue
        if got != expected:
            failures.append((kind, inp, expected, got))
    if failures:
        print("SELF-TEST FAILURES:")
        for k, inp, exp, got in failures:
            print(f"  [{k}] {inp!r}  expected={exp!r}  got={got!r}")
        sys.exit(1)
    print(f"OK: {len(SELF_TESTS)} self-tests passed.")


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------

def add_case_level_aliases(db_path: Path) -> dict:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")

    t0 = time.time()
    print("Building base->Es index over real source-row court citations...", flush=True)
    base_to_es: dict[str, set[str]] = defaultdict(set)
    n_court_e_rows = 0
    for (cit, mask) in conn.execute(
        "SELECT citation, origin_mask FROM citations "
        "WHERE dataset='court_considerations' AND (origin_mask & 1) != 0"
    ):
        sp = split_court_citation(cit)
        if sp is None:
            continue
        base, _ = sp
        nb = normalize_base(base)
        base_to_es[nb].add(cit)
        n_court_e_rows += 1
    print(f"  {n_court_e_rows:,} real court rows with E.-pinpoint", flush=True)
    print(f"  {len(base_to_es):,} distinct judgment-bases", flush=True)
    # Histogram of Es per base
    from collections import Counter
    sizes = Counter()
    for v in base_to_es.values():
        sizes[len(v)] += 1
    print(f"  base size distribution: 1E={sizes.get(1,0):,}  "
          f"2-5={sum(sizes[i] for i in range(2,6)):,}  "
          f"6-15={sum(sizes[i] for i in range(6,16)):,}  "
          f"16+={sum(c for n,c in sizes.items() if n >= 16):,}", flush=True)
    print(f"  built in {time.time()-t0:.1f}s\n", flush=True)

    pre = conn.execute(
        "SELECT COUNT(*) FROM edges WHERE dataset='court_considerations'"
    ).fetchone()[0]

    print("Streaming edges and computing case-level fan-out targets...", flush=True)
    t1 = time.time()
    cursor = conn.cursor()
    n_edges_scanned = 0
    n_court_e_targets = 0
    n_alias_candidates = 0
    pending: list[tuple[str, str, str]] = []
    BATCH = 200_000

    for source, target in conn.execute(
        "SELECT source, target FROM edges WHERE dataset='court_considerations'"
    ):
        n_edges_scanned += 1
        if n_edges_scanned % 1_000_000 == 0:
            print(
                f"  scanned {n_edges_scanned:,} edges "
                f"({n_court_e_targets:,} court-E targets, "
                f"{n_alias_candidates:,} alias candidates) "
                f"[{time.time()-t1:.1f}s]",
                flush=True,
            )
        sp = split_court_citation(target)
        if sp is None:
            continue
        n_court_e_targets += 1
        base, _ = sp
        nb = normalize_base(base)
        siblings = base_to_es.get(nb, ())
        for sib in siblings:
            if sib == target or sib == source:
                continue
            n_alias_candidates += 1
            pending.append(("court_considerations", source, sib))

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
        f"court_E_targets={n_court_e_targets:,}, "
        f"alias_candidates={n_alias_candidates:,}",
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
    add_case_level_aliases(args.db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
