#!/usr/bin/env python
"""Extract intra-judgment back-reference edges from court text fields.

Court rows often contain bare back-references such as
    "wie dargelegt (vgl. E. 6.2 hiervor)"
    "voir consid. 4.3 ci-dessus"
    "vedi cons. 5 supra"
Such bare backrefs resolve to siblings of the SAME judgment as the source row
(e.g., a backref in `BGE 137 IV 122 E. 6.4`'s text targets `BGE 137 IV 122 E. 6.2`).
The original `extract_citation_graph.py` only fires `CONSIDERATION_RE` after a
docket/BGE prefix, so bare backrefs produce zero edges and the sibling-expansion
recall channel can never see them.

This script adds a second pass over the corpus and inserts the missing edges
into the existing `data_insights/citation_graph_extracted.sqlite`. The edges
are written into the same `edges` table so all downstream channels see them
without code changes elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
import time
from pathlib import Path

# Reuse classifiers from the original extractor so source-base derivation is
# consistent with how the citation column is parsed there.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_citation_graph import (  # noqa: E402
    classify_source_citation,
    BGE_RE,
    MODERN_CASE_RE,
    LEGACY_CASE_RE,
    squash_ws,
)

csv.field_size_limit(1 << 28)


# Tokens that indicate the surrounding "E. N" / "consid. N" is a back-reference
# to the SAME judgment, not a fresh citation. We require at least one such
# marker for an extracted ref to count, otherwise risk of false positives from
# continuation refs like "BGE 130 III 1 E. 5; E. 6" is too high.
INTRO_MARKERS = r'(?:vgl|siehe|cf|cfr|voir|vedi|vedasi|vedere)'
# Positional markers can appear EITHER before or after the unit (Swiss legal
# style is inconsistent). They alone — without "vgl./siehe" — are still
# strong enough signal that the bare ref is intra-judgment.
POSITIONAL_MARKERS = (
    r'(?:hiervor|hierin|hereunder|hereinafter|hereinabove|above|earlier'
    r'|oben|vorstehend(?:e[nrs]?)?|vorhergehend(?:e[nrs]?)?|vorhin'
    r'|ci-dessus|ci-après|ci-apres|ci-avant'
    r'|sopra|supra|infra'
    r'|précédent(?:e)?|precedente|prossimo)'
)
# Connective glue allowed between intro and ref ("vgl. auch oben E. 4")
INTRO_GLUE = r'(?:auch\s+|aussi\s+|also\s+|anche\s+|bereits\s+|ferner\s+|noch\s+|i\.\s*e\.\s+|d\.\s*h\.\s+)?'
UNIT = r'(?:E\.|Erw\.|Erwägung(?:en)?|consid\.|considérant(?:s)?|cons\.|considerando|c\.)'
# Single ref: digits with optional sub-decimals and trailing letter (e.g. 4.2.1, 4a)
REF_TOKEN = r'\d+(?:\.\d+)*[a-z]?'
# Comma/und/et/oder list of refs: "3.2, 4.1 und 5.2" → ["3.2", "4.1", "5.2"]
REF_LIST = rf'{REF_TOKEN}(?:\s*(?:,|und|et|e|oder|ou|o|/|;)\s*{REF_TOKEN})*'

# Pattern A: intro-marker form, optionally with positional adverb between intro and unit.
# Examples:
#   "vgl. E. 6.2"
#   "siehe oben E. 4.3.1"            <-- v2 NEW
#   "vgl. oben E. 3.2 und 3.3.2"     <-- v2 NEW
#   "vgl. bereits hiervor E. 3.3"    <-- v2 NEW
#   "siehe consid. 3.4 ci-dessus"
PATTERN_INTRO = re.compile(
    rf'\b{INTRO_MARKERS}\.?\s*{INTRO_GLUE}'
    rf'(?:{POSITIONAL_MARKERS}\s+(?:in\s+)?)?'
    rf'{UNIT}\s*(?P<refs>{REF_LIST})'
    rf'(?:\s+{POSITIONAL_MARKERS})?',
    re.IGNORECASE,
)
# Pattern B: bare positional + UNIT (no intro), e.g., "hiervor E. 3.3", "oben E. 4.3".
PATTERN_PRE_POSITIONAL = re.compile(
    rf'\b{POSITIONAL_MARKERS}\s+(?:in\s+)?{UNIT}\s*(?P<refs>{REF_LIST})',
    re.IGNORECASE,
)
# Pattern C: trailing-marker form. "E. 6.2 hiervor", "consid. 4.3 ci-dessus".
PATTERN_TRAILING = re.compile(
    rf'\b{UNIT}\s*(?P<refs>{REF_LIST})\s+{POSITIONAL_MARKERS}\b',
    re.IGNORECASE,
)
# Pattern D: parenthetical "(E. 6.2)" — no marker, accepted as judgment-internal
# only when paren contains essentially the ref alone (avoids matching paren'd
# continuation citations).
PATTERN_PAREN = re.compile(
    rf'\(\s*{UNIT}\s*(?P<refs>{REF_LIST})\s*\)',
    re.IGNORECASE,
)
# Split a comma/und/et list into individual ref tokens.
SPLIT_RE = re.compile(r'\s*(?:,|und|et|e|oder|ou|o|/|;)\s*', re.IGNORECASE)
# Validate a single ref token after splitting (defensive).
SINGLE_REF_RE = re.compile(rf'^{REF_TOKEN}$')


def derive_source_base(source_citation: str) -> str | None:
    """Return the judgment-level base of a court source citation, or None.

    Examples
    --------
    >>> derive_source_base("BGE 137 IV 122 E. 6.4")
    'BGE 137 IV 122'
    >>> derive_source_base("1B_90/2021 E. 2.1")
    '1B_90/2021'
    >>> derive_source_base("2P.198/2006 09.05.2007 E. 2")
    '2P.198/2006'
    >>> derive_source_base("Art. 221 Abs. 1 StPO") is None
    True
    """
    if not source_citation:
        return None
    classified = classify_source_citation(source_citation)
    if classified.family != "court":
        return None
    seg = classified.segments
    if classified.subfamily == "bge":
        v, d, p = seg.get("volume"), seg.get("division"), seg.get("page")
        if v and d and p:
            return f"BGE {v} {d} {p}"
        return None
    if classified.subfamily.startswith("federal_tribunal_"):
        return seg.get("docket")
    if classified.subfamily == "compact_slash_docket":
        return seg.get("docket")
    return None


def _mask_full_citations(text: str) -> str:
    """Replace full BGE/docket spans with spaces so backref patterns don't
    accidentally match inside them. Preserves character offsets so any later
    span analysis stays meaningful.

    Important: MODERN_CASE_RE and LEGACY_CASE_RE include a `trailing` capture
    group of up to 90 chars after the docket. Masking the WHOLE match span
    would also wipe out any backref idioms ("(cf. supra consid. N)") that sit
    in that trailing window. We only mask the docket itself plus an optional
    immediately-following "E. N" pinpoint chunk if one is present, since that
    pinpoint belongs to the docket citation and should not be re-extracted as
    a self-reference.
    """
    if not text:
        return text
    masked = list(text)
    # BGE_RE: full match span IS the citation (including any pinpoint), safe to mask whole span.
    for m in BGE_RE.finditer(text):
        for i in range(m.start(), m.end()):
            masked[i] = ' '
    # MODERN_CASE_RE / LEGACY_CASE_RE: only mask the docket group plus optional
    # pinpoint immediately after (we extract the pinpoint here using a small
    # helper regex on the trailing window).
    pinpoint_re = re.compile(
        r'^(?:\s+(?:vom|du|del|of)?\s*\d{1,2}\.\s*\d{1,2}\.\s*\d{4})?'  # date
        r'\s*(?:E\.|consid\.|cons\.|c\.)\s*(?:\d+(?:\.\d+)*[a-z]?)',
        re.IGNORECASE,
    )
    for regex in (MODERN_CASE_RE, LEGACY_CASE_RE):
        for m in regex.finditer(text):
            d_start = m.start("docket")
            d_end = m.end("docket")
            for i in range(d_start, d_end):
                masked[i] = ' '
            # Try to extend mask through immediate trailing pinpoint, if any.
            tail = text[d_end : d_end + 90]
            pp = pinpoint_re.match(tail)
            if pp:
                for i in range(d_end, d_end + pp.end()):
                    masked[i] = ' '
    return ''.join(masked)


def extract_backref_targets(text: str, source_base: str) -> set[str]:
    """For one row, return the full-citation targets that bare backrefs in
    `text` should resolve to under `source_base`.

    Returns canonical strings of the form "{source_base} E. {ref}".
    """
    if not text or not source_base:
        return set()
    masked = _mask_full_citations(text)
    refs: set[str] = set()
    for pattern in (PATTERN_INTRO, PATTERN_PRE_POSITIONAL, PATTERN_TRAILING, PATTERN_PAREN):
        for m in pattern.finditer(masked):
            ref_chunk = squash_ws(m.group("refs"))
            for part in SPLIT_RE.split(ref_chunk):
                p = part.strip()
                if not p:
                    continue
                if not SINGLE_REF_RE.match(p):
                    continue
                refs.add(p)
    return {f"{source_base} E. {r}" for r in refs}


# ---------------------------------------------------------------------------
# Self-tests on the val_001-relevant rows we inspected manually.
# ---------------------------------------------------------------------------

SELF_TESTS = [
    # (source_citation, text_excerpt, expected_targets_subset)
    (
        "BGE 137 IV 122 E. 6.4",
        "Eine Eingrenzung auf ein bestimmtes Gebiet kommt, wie dargelegt "
        "(vgl. E. 6.2 hiervor), primär bei Fluchtgefahr in Betracht.",
        {"BGE 137 IV 122 E. 6.2"},
    ),
    (
        "BGE 137 IV 122 E. 5",
        "Wie dargelegt (E. 4 hiervor), ist der Haftgrund der Kollusionsgefahr "
        "im Sinne von Art. 221 Abs. 1 lit. b StPO zu bejahen.",
        {"BGE 137 IV 122 E. 4"},
    ),
    (
        "BGE 137 IV 122 E. 5.3",
        'Die Vorinstanz hat zur Begründung auf das forensisch-psychiatrische '
        'Gutachten vom 13. Dezember 2010 verwiesen, wo unter dem Titel '
        '"Risikoeinschätzung" Folgendes festgehalten wurde (vgl. auch '
        'E. 4.3 hiervor):',
        {"BGE 137 IV 122 E. 4.3"},
    ),
    (
        "BGE 145 I 26 E. 8.2",
        "Im Rahmen einer Plausibilitätsüberprüfung hat das kantonale Gericht "
        "(vgl. E. 6.2.2.1 hiervor und E. 7 hiervor) festgestellt.",
        {"BGE 145 I 26 E. 6.2.2.1", "BGE 145 I 26 E. 7"},
    ),
    (
        "1B_90/2021 E. 2.1",
        "voir consid. 2.4 ci-dessus pour le contexte.",
        {"1B_90/2021 E. 2.4"},
    ),
    (
        "BGE 137 IV 122 E. 5.2",
        "Die rein hypothetische Möglichkeit (BGE 125 I 60 E. 3a S. 62 mit "
        "Hinweis). Vgl. E. 5.1 hiervor.",
        {"BGE 137 IV 122 E. 5.1"},
    ),
    # v2 NEW: "siehe oben E. N" — INTRO + POSITIONAL + UNIT
    (
        "BGE 136 I 1 E. 5.4.3",
        "trifft dies bei unprofessioneller Züchtung um so mehr zu (siehe oben E. 4.3.1). Mit einer Bewilligung...",
        {"BGE 136 I 1 E. 4.3.1"},
    ),
    # v2 NEW: "vgl. oben E. N" with comma-list of refs
    (
        "BGE 148 I 65 E. 3.4.3",
        "verletzt (vgl. oben E. 3.2 und 3.3.2). Folgerichtig...",
        {"BGE 148 I 65 E. 3.2", "BGE 148 I 65 E. 3.3.2"},
    ),
    # v2 NEW: "hiervor E. N" — POSITIONAL before UNIT
    (
        "BGE 142 I 49 E. 3.5",
        "BGE 139 I 292 E. 8.2.3 S. 304; MÜLLER/SCHEFER, a.a.O.; vgl. bereits hiervor E. 3.3). Auch ein System...",
        {"BGE 142 I 49 E. 3.3"},
    ),
    # v2 NEW: "hiervor E. N" without intro
    (
        "BGE 142 I 49 E. 7.1",
        "BGE 134 I 56 E. 5.2 S. 63; hiervor E. 5.2). Anordnungen...",
        {"BGE 142 I 49 E. 5.2"},
    ),
    # v2 NEW: "vorstehend E. N"
    (
        "BGE 148 I 65 E. 4.5.1",
        "Die Berechnungen sind im Lichte der vorstehend dargelegten Grundsätze über die "
        "Unterhaltsbesteuerung (vorstehend E. 3.4) zu beurteilen.",
        {"BGE 148 I 65 E. 3.4"},
    ),
    # Negative: a continuation citation should NOT spawn a self-edge
    (
        "BGE 137 IV 122 E. 5.2",
        "(BGE 125 I 60 E. 3a S. 62 mit Hinweis).",
        set(),
    ),
    # Negative: positional adverb without a unit ref nearby is just text — no edge
    (
        "BGE 148 I 65 E. 4.5.3",
        "auf das Bruttovermögen angewendet hat, ist ihr die vorstehend dargestellte Problematik nicht entgangen.",
        set(),
    ),
    # v2.1 NEW: mask must not eat 90-char "trailing" of MODERN_CASE_RE.
    # Without the masking fix, the `(cf. supra consid. 6.6)` here would be
    # wiped because it falls inside 6B_1015/2020's trailing capture group.
    (
        "BGE 148 I 145 E. 6.11",
        "l'a déjà relevé dans son arrêt 6B_1015/2020, évoqué plus haut "
        "(cf. supra consid. 6.6). C'est donc...",
        {"BGE 148 I 145 E. 6.6"},
    ),
    # v2.1 NEG: a docket immediately followed by its OWN pinpoint should be
    # treated as one citation; we must NOT spawn a self-edge for the pinpoint.
    (
        "BGE 137 IV 122 E. 4.2",
        "voir l'arrêt 1B_415/2010 du 23 décembre 2010 consid. 3.5 pour la suite.",
        set(),
    ),
]


def run_self_tests() -> None:
    failures = []
    for source, text, expected in SELF_TESTS:
        base = derive_source_base(source)
        got = extract_backref_targets(text, base or "")
        missing = expected - got
        unexpected_self_edges = {t for t in got if t == source}
        if missing or unexpected_self_edges:
            failures.append((source, expected, got))
    if failures:
        print("SELF-TEST FAILURES:")
        for s, e, g in failures:
            print(f"  source={s!r}\n    expected ⊇ {sorted(e)}\n    got      = {sorted(g)}")
        sys.exit(1)
    print(f"OK: {len(SELF_TESTS)} self-tests passed.")


# ---------------------------------------------------------------------------
# Main extraction pass.
# ---------------------------------------------------------------------------

def open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")
    return conn


def add_intra_judgment_edges(
    conn: sqlite3.Connection,
    csv_path: Path,
    dataset: str,
    limit: int | None,
    batch_size: int,
    progress_every: int,
) -> dict:
    cursor = conn.cursor()
    inserted = 0
    edges_seen = 0
    rows_with_backref = 0
    rows_total = 0
    rows_skipped_non_court_source = 0
    pending: list[tuple[str, str, str]] = []
    target_inserts: list[tuple[str, str, int, str, str, str, str]] = []

    # Sanity: ensure we know what's already there.
    pre = cursor.execute(
        "SELECT COUNT(*) FROM edges WHERE dataset = ?", (dataset,)
    ).fetchone()[0]

    t0 = time.time()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_total += 1
            source_raw = squash_ws(row.get("citation") or "")
            text = row.get("text") or ""
            if not source_raw or not text:
                continue
            base = derive_source_base(source_raw)
            if base is None:
                rows_skipped_non_court_source += 1
                continue
            targets = extract_backref_targets(text, base)
            # Drop self-edges
            targets.discard(source_raw)
            if targets:
                rows_with_backref += 1
                edges_seen += len(targets)
                for t in targets:
                    pending.append((dataset, source_raw, t))
                    # Also ensure target node exists. We mark it as origin=2 (text reference).
                    # We avoid re-classifying every target on the hot path; segments_json
                    # is set to "{}" for these synthetic nodes — the citations table is
                    # used downstream only for membership checks, not pattern stats.
                    target_inserts.append((
                        dataset, t, 2, "court", "intra_judgment_backref",
                        "court_intra_backref", "{}",
                    ))

            if len(pending) >= batch_size:
                cursor.executemany(
                    "INSERT OR IGNORE INTO edges (dataset, source, target) VALUES (?, ?, ?)",
                    pending,
                )
                inserted += cursor.rowcount if cursor.rowcount >= 0 else 0
                cursor.executemany(
                    """
                    INSERT INTO citations (dataset, citation, origin_mask, family, subfamily, pattern, segments_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(dataset, citation) DO UPDATE SET
                        origin_mask = citations.origin_mask | excluded.origin_mask
                    """,
                    target_inserts,
                )
                conn.commit()
                pending.clear()
                target_inserts.clear()

            if rows_total % progress_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  {dataset}: scanned {rows_total:,} rows "
                    f"({rows_with_backref:,} with >=1 backref, {edges_seen:,} edges seen) "
                    f"in {elapsed:.1f}s",
                    flush=True,
                )
            if limit is not None and rows_total >= limit:
                break

    if pending:
        cursor.executemany(
            "INSERT OR IGNORE INTO edges (dataset, source, target) VALUES (?, ?, ?)",
            pending,
        )
        cursor.executemany(
            """
            INSERT INTO citations (dataset, citation, origin_mask, family, subfamily, pattern, segments_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset, citation) DO UPDATE SET
                origin_mask = citations.origin_mask | excluded.origin_mask
            """,
            target_inserts,
        )
        conn.commit()

    post = cursor.execute(
        "SELECT COUNT(*) FROM edges WHERE dataset = ?", (dataset,)
    ).fetchone()[0]
    elapsed = time.time() - t0
    print(
        f"  {dataset}: done in {elapsed:.1f}s. "
        f"rows_total={rows_total:,}, "
        f"rows_with_backref={rows_with_backref:,}, "
        f"edges_seen={edges_seen:,}, "
        f"db_edges_before={pre:,}, db_edges_after={post:,} "
        f"(net new = {post - pre:,})",
        flush=True,
    )
    return {
        "dataset": dataset,
        "rows_total": rows_total,
        "rows_with_backref": rows_with_backref,
        "edges_seen": edges_seen,
        "db_edges_before": pre,
        "db_edges_after": post,
        "net_new": post - pre,
        "elapsed_s": elapsed,
        "rows_skipped_non_court_source": rows_skipped_non_court_source,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path(r"E:\swiss_citation_extraction\data"))
    p.add_argument(
        "--db-path",
        type=Path,
        default=Path(r"E:\swiss_citation_extraction\data_insights\citation_graph_extracted.sqlite"),
    )
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most this many rows per dataset (for smoke tests).")
    p.add_argument("--batch-size", type=int, default=50_000)
    p.add_argument("--progress-every", type=int, default=200_000)
    p.add_argument("--datasets", nargs="+",
                   choices=["court_considerations", "laws_de"],
                   default=["court_considerations"],
                   help="Currently only court_considerations is supported "
                        "(laws_de has no docket-style sources).")
    p.add_argument("--self-test-only", action="store_true",
                   help="Run unit tests and exit (no DB write).")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    run_self_tests()
    if args.self_test_only:
        return 0
    if not args.db_path.exists():
        print(f"DB not found: {args.db_path}", file=sys.stderr)
        return 2
    conn = open_db(args.db_path)
    summaries: list[dict] = []
    for dataset in args.datasets:
        csv_name = "court_considerations.csv" if dataset == "court_considerations" else "laws_de.csv"
        csv_path = args.data_dir / csv_name
        if not csv_path.exists():
            print(f"CSV not found: {csv_path}", file=sys.stderr)
            return 3
        print(f"Adding intra-judgment backref edges for {dataset} ({csv_path}) ...", flush=True)
        summary = add_intra_judgment_edges(
            conn,
            csv_path,
            dataset,
            args.limit,
            args.batch_size,
            args.progress_every,
        )
        summaries.append(summary)
    conn.close()
    print()
    print("=== Run summary ===")
    for s in summaries:
        for k, v in s.items():
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
