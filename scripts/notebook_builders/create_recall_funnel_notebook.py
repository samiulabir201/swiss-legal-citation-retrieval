"""Generate the Colab recall-preserving candidate funnel notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "colab_recall_preserving_candidate_funnel.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip("\n").splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(True),
    }


cells: list[dict] = []

cells.append(
    md(
        """
# Swiss Legal Citation Retrieval: Recall-Preserving Candidate Funnel

This Colab notebook builds a controlled, multi-stage candidate generation system for Swiss legal citation retrieval.

The central rule is strict: **all final candidates must come from the source `citation` columns of `laws_de.csv` or `court_considerations.csv`**. Regex, LLM planner output, embeddings, and rerankers may only choose routes and filters; they may never invent final citation IDs.

The notebook handles laws and court considerations separately, audits recall after every narrowing stage, and only then merges candidates into a 2k-10k recall-first pool.
"""
    )
)

cells.append(
    md(
        """
## 0. Colab Setup

Expected Drive layout:

```text
/content/drive/MyDrive/swiss_law/data
/content/drive/MyDrive/swiss_law/data_insights
/content/drive/MyDrive/swiss_law/artifacts
```

The heavy model cells are optional. You can first run the deterministic funnel and recall audit, then enable the Qwen planner/reranker cells.
"""
    )
)

cells.append(
    code(
        """
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

BASE_DIR = Path("/content/drive/MyDrive/swiss_law") if IN_COLAB else Path("..").resolve()
DATA_DIR = BASE_DIR / "data"
INSIGHTS_DIR = BASE_DIR / "data_insights"
ART_DIR = BASE_DIR / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

print("BASE_DIR    :", BASE_DIR)
print("DATA_DIR    :", DATA_DIR)
print("INSIGHTS_DIR:", INSIGHTS_DIR)
print("ART_DIR     :", ART_DIR)
"""
    )
)

cells.append(
    code(
        """
if IN_COLAB:
    import subprocess
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-U",
        "duckdb",
        "polars",
        "pandas",
        "pyarrow",
        "orjson",
        "ijson",
        "regex",
        "rapidfuzz",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "sentence-transformers",
        "faiss-gpu-cu12",
    ])
"""
    )
)

cells.append(
    code(
        """
import csv
import gc
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pandas as pd
import regex as regex
from rapidfuzz import fuzz

try:
    import orjson

    def loads_json(value: str | bytes) -> Any:
        return orjson.loads(value)

    def dumps_json(value: Any) -> str:
        return orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode("utf-8")

except Exception:
    import json

    def loads_json(value: str | bytes) -> Any:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return json.loads(value)

    def dumps_json(value: Any) -> str:
        return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)

FILES = {
    "laws_csv": DATA_DIR / "laws_de.csv",
    "court_csv": DATA_DIR / "court_considerations.csv",
    "train_csv": DATA_DIR / "train.csv",
    "val_csv": DATA_DIR / "val.csv",
    "test_csv": DATA_DIR / "test.csv",
    "laws_classified": INSIGHTS_DIR / "laws_de_classified_citations.jsonl",
    "court_classified": INSIGHTS_DIR / "court_considerations_classified_citations.jsonl",
    "laws_links": INSIGHTS_DIR / "laws_de_links.json",
    "court_links": INSIGHTS_DIR / "court_considerations_links.json",
}

for name, path in FILES.items():
    print(f"{name:18s}", "OK" if path.exists() else "MISSING", path)
"""
    )
)

cells.append(
    md(
        """
## 1. Configuration

Set `RUN_SPLIT = "val"` while developing. For test inference, use `RUN_SPLIT = "test"`; the recall audit will still produce candidate counts, but no gold recall.

`RECALL_GUARD_ON_VAL` is a development safety switch. When a validation stage drops recall below its gate, the notebook keeps the previous wider candidate set and records the failed stage. This prevents one bad planner choice from silently destroying the candidate pool.
"""
    )
)

cells.append(
    code(
        """
RUN_SPLIT = "val"  # "val", "test", or "train"

BUILD_TEXT_TABLES = True      # Builds DuckDB tables with citation text. Slow but useful for reranking.
BUILD_EDGE_TABLES = True      # Streams links JSON into citation_edges for statute-citing court routes.
FORCE_REBUILD_DB = False      # Set True after changing parsing/index logic.

USE_PLANNER_LLM = False       # Set True after the indexes/choice cards are built.
USE_RERANKER = False          # Optional later stage.

PLANNER_MODEL_NAME = "Qwen/Qwen3-32B"
RERANKER_MODEL_NAME = "Qwen/Qwen3-Reranker-8B"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

RECALL_GUARD_ON_VAL = True
EARLY_STAGE_MIN_RECALL = 0.98
MID_STAGE_MIN_RECALL = 0.95
FINAL_MIN_MEAN_RECALL = 0.85
FINAL_MIN_QUERY_RECALL = 0.75

MIN_TOTAL_CANDIDATES = 2_000
DEFAULT_TOTAL_CANDIDATES = 6_000
MAX_TOTAL_CANDIDATES = 10_000
DIAGNOSTIC_MAX_CANDIDATES = 20_000

DEFAULT_LAW_BUDGET = 1_500
DEFAULT_COURT_BUDGET = 4_500

DB_PATH = ART_DIR / "recall_funnel.duckdb"
print("DB_PATH:", DB_PATH)
"""
    )
)

cells.append(
    md(
        """
## 2. Regex And Citation Normalization Layer

The regex layer is intentionally broad. It is used to detect query mentions and generate variant expansions. Final candidates still come only from the source citation inventory.
"""
    )
)

cells.append(
    code(
        r"""
WS_RE = re.compile(r"\s+")

LAW_REF_RE = regex.compile(
    r'''
    (?P<marker>Art\.|Artikel)\s+
    (?P<article>\d+[a-zA-Z]*(?:\s*(?:bis|ter|quater))?)
    (?P<continuation>\s*(?:f\.|ff\.))?
    (?P<units>
        (?:
            \s+
            (?:
                (?:Abs\.|Absatz|Absatze|Absätze)\s+\d+[a-zA-Z]*(?:bis)?
                |(?:Bst\.|lit\.|Buchstabe)\s+[a-zA-Z]
                |(?:Ziff\.|Ziffer|Ziffern|Nr\.)\s+\d+[a-zA-Z]*
                |(?:Satz)\s+\d+
                |(?:Unterabs\.)\s+\d+
            )
        )*
    )
    (?:\s+(?P<law_code>[A-ZÄÖÜ][A-Za-zÄÖÜäöü0-9./-]{1,40}))?
    ''',
    regex.VERBOSE,
)

BGE_RE = regex.compile(
    r"\bBGE\s+(?P<volume>\d{2,3})\s+(?P<division>[IVX]{1,5})\s+(?P<page>\d{1,4})(?:\s+E\.?\s*(?P<pinpoint>[A-Za-z0-9.:-]+))?",
    regex.IGNORECASE,
)

MODERN_DOCKET_RE = regex.compile(
    r"\b(?P<docket>\d{1,2}[A-Z]{1,2}[_\.]\d{1,5}/\d{4})(?:\s+(?P<date>\d{2}\.\d{2}\.\d{4}))?(?:\s+E\.?\s*(?P<consideration>[A-Za-z0-9.:-]+))?",
    regex.IGNORECASE,
)

LEGACY_DOCKET_RE = regex.compile(
    r"\b(?P<docket>[A-Z]\s+\d{1,5}/\d{2})(?:\s+(?P<date>\d{2}\.\d{2}\.\d{4}))?(?:\s+E\.?\s*(?P<consideration>[A-Za-z0-9.:-]+))?",
    regex.IGNORECASE,
)

UNKNOWN_DOCKET_LIKE_RE = regex.compile(
    r"\b(?P<docket>(?:\d{1,2}[A-Z]{1,2}[_\.]\d{1,5}/\d{4}|[A-Z]\s+\d{1,5}/\d{2}))",
    regex.IGNORECASE,
)

DATE_TRIGGER_RE = regex.compile(
    r"\b(after|before|since|until|between|recent|newer|older|from\s+\d{4}|post[-\s]?\d{4}|pre[-\s]?\d{4}|in force|entered into force|law in force)\b",
    regex.IGNORECASE,
)


def squash_ws(value: str | None) -> str:
    return WS_RE.sub(" ", value or "").strip()


def unit_chain_from_units(units: list[dict] | None) -> str:
    units = units or []
    return ">".join(str(u.get("category") or "").strip() for u in units if u.get("category"))


def unit_values_from_units(units: list[dict] | None) -> str:
    units = units or []
    values = []
    for unit in units:
        category = unit.get("category") or ""
        value = unit.get("value") or ""
        if category or value:
            values.append(f"{category}:{value}")
    return "|".join(values)


def parse_docket_prefix(docket: str | None) -> str | None:
    if not docket:
        return None
    docket = squash_ws(docket)
    if "_" in docket:
        return docket.split("_", 1)[0]
    if "." in docket and "/" in docket:
        return docket.split(".", 1)[0]
    if " " in docket:
        return docket.split(" ", 1)[0]
    return None


def normalize_court_base(citation: str, pattern: str, segments: dict) -> str:
    if pattern == "court_bge":
        return f"BGE {segments.get('volume')} {segments.get('division')} {segments.get('page')}"
    if pattern == "court_case":
        return segments.get("docket") or citation
    return citation


def extract_query_mentions(query: str) -> dict[str, list[dict]]:
    mentions = {"laws": [], "bge": [], "dockets": [], "dates": []}
    for m in LAW_REF_RE.finditer(query):
        mentions["laws"].append({k: squash_ws(v) if v else None for k, v in m.groupdict().items()})
    for m in BGE_RE.finditer(query):
        gd = {k: squash_ws(v) if v else None for k, v in m.groupdict().items()}
        gd["base"] = f"BGE {gd['volume']} {gd['division']} {gd['page']}"
        mentions["bge"].append(gd)
    for rx in (MODERN_DOCKET_RE, LEGACY_DOCKET_RE):
        for m in rx.finditer(query):
            gd = {k: squash_ws(v) if v else None for k, v in m.groupdict().items()}
            gd["prefix"] = parse_docket_prefix(gd.get("docket"))
            mentions["dockets"].append(gd)
    mentions["time_mode"] = "hard_year_range" if DATE_TRIGGER_RE.search(query) else "no_time_filter"
    return mentions
"""
    )
)

cells.append(
    md(
        """
## 3. Build Compact DuckDB Indexes

This creates source-only citation segment tables and optional text/edge tables. It also exports parquet copies so later reruns start quickly.
"""
    )
)

cells.append(
    code(
        """
def segment_record(dataset: str, row: dict) -> tuple:
    citation = row["citation"]
    pattern = row.get("pattern") or "unknown"
    family = row.get("family") or "unknown"
    subfamily = row.get("subfamily") or "unknown"
    origin_mask = int(row.get("origin_mask") or 0)
    seg = row.get("segments") or {}

    article = seg.get("article") or seg.get("article_number")
    section = seg.get("section") or seg.get("section_number")
    law_code = seg.get("law_code")
    law_code_family = seg.get("law_code_family")
    law_code_resolution = seg.get("law_code_resolution")
    article_continuation = seg.get("article_continuation") or seg.get("article_suffix_or_range")
    units = seg.get("units") or []
    unit_chain = unit_chain_from_units(units)
    unit_values = unit_values_from_units(units)
    unit_depth = len(units)

    docket = seg.get("docket")
    docket_prefix = parse_docket_prefix(docket)
    legal_area_code = seg.get("legal_area_code")
    court_chamber = seg.get("court_chamber")
    separator_style = seg.get("separator_style")
    decision_year = seg.get("decision_year")
    decision_date = seg.get("decision_date")
    consideration = seg.get("consideration") or seg.get("pinpoint")
    bge_volume = seg.get("volume")
    bge_division = seg.get("division")
    bge_page = seg.get("page")
    court_base = normalize_court_base(citation, pattern, seg)
    raw = seg.get("raw") or citation

    if pattern == "unknown":
        m = UNKNOWN_DOCKET_LIKE_RE.search(citation)
        if m:
            docket = docket or squash_ws(m.group("docket"))
            docket_prefix = docket_prefix or parse_docket_prefix(docket)
            court_base = docket

    return (
        dataset,
        citation,
        origin_mask,
        family,
        subfamily,
        pattern,
        article,
        section,
        law_code,
        law_code_family,
        law_code_resolution,
        article_continuation,
        unit_chain,
        unit_values,
        unit_depth,
        docket,
        docket_prefix,
        legal_area_code,
        court_chamber,
        separator_style,
        decision_year,
        decision_date,
        consideration,
        bge_volume,
        bge_division,
        bge_page,
        court_base,
        raw,
    )


SEGMENT_SCHEMA = '''
    dataset TEXT,
    citation TEXT,
    origin_mask INTEGER,
    family TEXT,
    subfamily TEXT,
    pattern TEXT,
    article TEXT,
    section TEXT,
    law_code TEXT,
    law_code_family TEXT,
    law_code_resolution TEXT,
    article_continuation TEXT,
    unit_chain TEXT,
    unit_values TEXT,
    unit_depth INTEGER,
    docket TEXT,
    docket_prefix TEXT,
    legal_area_code TEXT,
    court_chamber TEXT,
    separator_style TEXT,
    decision_year TEXT,
    decision_date TEXT,
    consideration TEXT,
    bge_volume TEXT,
    bge_division TEXT,
    bge_page TEXT,
    court_base TEXT,
    raw TEXT
'''


def create_segment_table(con: duckdb.DuckDBPyConnection, table: str, path: Path, dataset: str, batch_size: int = 100_000) -> None:
    con.execute(f"DROP TABLE IF EXISTS {table}")
    con.execute(f"CREATE TABLE {table} ({SEGMENT_SCHEMA})")
    insert_sql = f"INSERT INTO {table} VALUES ({','.join(['?'] * 28)})"

    batch = []
    kept = 0
    seen = 0
    t0 = time.time()
    with path.open("rb") as f:
        for raw_line in f:
            if not raw_line.strip():
                continue
            seen += 1
            row = loads_json(raw_line)
            origin_mask = int(row.get("origin_mask") or 0)
            if not (origin_mask & 1):
                continue
            batch.append(segment_record(dataset, row))
            kept += 1
            if len(batch) >= batch_size:
                con.executemany(insert_sql, batch)
                batch.clear()
                print(f"{table}: inserted {kept:,} source rows after {seen:,} jsonl rows ({time.time() - t0:.1f}s)")
    if batch:
        con.executemany(insert_sql, batch)
    print(f"{table}: done. source rows={kept:,}, jsonl rows scanned={seen:,}")


def create_text_tables(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("DROP TABLE IF EXISTS law_citations")
    con.execute(
        f'''
        CREATE TABLE law_citations AS
        SELECT citation, title, text
        FROM read_csv_auto('{FILES['laws_csv'].as_posix()}', header=true, all_varchar=true, ignore_errors=true)
        '''
    )
    con.execute("DROP TABLE IF EXISTS court_considerations")
    con.execute(
        f'''
        CREATE TABLE court_considerations AS
        SELECT citation, text
        FROM read_csv_auto('{FILES['court_csv'].as_posix()}', header=true, all_varchar=true, ignore_errors=true)
        '''
    )


def create_edge_table(con: duckdb.DuckDBPyConnection, batch_size: int = 250_000) -> None:
    import ijson

    con.execute("DROP TABLE IF EXISTS citation_edges")
    con.execute("CREATE TABLE citation_edges (dataset TEXT, source TEXT, target TEXT)")
    insert_sql = "INSERT INTO citation_edges VALUES (?, ?, ?)"

    for dataset, path in [("laws_de", FILES["laws_links"]), ("court_considerations", FILES["court_links"])]:
        if not path.exists():
            print("Missing links file:", path)
            continue
        batch = []
        n_edges = 0
        t0 = time.time()
        with path.open("rb") as f:
            for item in ijson.items(f, "source_to_references.item"):
                source = item.get("source")
                for target in item.get("references", []) or []:
                    batch.append((dataset, source, target))
                    n_edges += 1
                    if len(batch) >= batch_size:
                        con.executemany(insert_sql, batch)
                        batch.clear()
                        print(f"{dataset}: inserted {n_edges:,} edges ({time.time() - t0:.1f}s)")
        if batch:
            con.executemany(insert_sql, batch)
        print(f"{dataset}: edge load done, edges={n_edges:,}")

    con.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON citation_edges(target)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON citation_edges(source)")


def build_or_load_db() -> duckdb.DuckDBPyConnection:
    if FORCE_REBUILD_DB and DB_PATH.exists():
        DB_PATH.unlink()

    con = duckdb.connect(str(DB_PATH))
    existing_tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    need_segments = not {"law_segments", "court_segments"}.issubset(existing_tables)

    if need_segments:
        create_segment_table(con, "law_segments", FILES["laws_classified"], "laws_de")
        create_segment_table(con, "court_segments", FILES["court_classified"], "court_considerations")
        con.execute("CREATE INDEX IF NOT EXISTS idx_law_citation ON law_segments(citation)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_law_code_article ON law_segments(law_code, article)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_court_citation ON court_segments(citation)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_court_base ON court_segments(court_base)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_court_prefix ON court_segments(docket_prefix)")

    existing_tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if BUILD_TEXT_TABLES and not {"law_citations", "court_considerations"}.issubset(existing_tables):
        create_text_tables(con)

    existing_tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    if BUILD_EDGE_TABLES and "citation_edges" not in existing_tables:
        create_edge_table(con)

    con.execute(
        '''
        CREATE OR REPLACE TABLE court_decisions AS
        SELECT
            court_base,
            any_value(pattern) AS dominant_pattern,
            any_value(subfamily) AS subfamily,
            any_value(docket_prefix) AS docket_prefix,
            any_value(legal_area_code) AS legal_area_code,
            any_value(court_chamber) AS court_chamber,
            any_value(separator_style) AS separator_style,
            any_value(decision_year) AS decision_year,
            any_value(decision_date) AS decision_date,
            any_value(bge_volume) AS bge_volume,
            any_value(bge_division) AS bge_division,
            any_value(bge_page) AS bge_page,
            count(*) AS consideration_count
        FROM court_segments
        GROUP BY court_base
        '''
    )
    return con


con = build_or_load_db()
print(con.execute("SHOW TABLES").fetchdf())
print("law source rows  :", con.execute("SELECT count(*) FROM law_segments").fetchone()[0])
print("court source rows:", con.execute("SELECT count(*) FROM court_segments").fetchone()[0])
"""
    )
)

cells.append(
    md(
        """
## 4. Source Inventory And Gold Labels

These sets define valid predictions. Text-reference-only citations are never valid final candidates.
"""
    )
)

cells.append(
    code(
        """
LAW_SOURCE = set(con.execute("SELECT citation FROM law_segments").fetchdf()["citation"])
COURT_SOURCE = set(con.execute("SELECT citation FROM court_segments").fetchdf()["citation"])
ALL_SOURCE = LAW_SOURCE | COURT_SOURCE

print(f"LAW_SOURCE   : {len(LAW_SOURCE):,}")
print(f"COURT_SOURCE : {len(COURT_SOURCE):,}")
print(f"ALL_SOURCE   : {len(ALL_SOURCE):,}")


def split_gold(value: str | float | None) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def load_queries(split: str) -> pd.DataFrame:
    path = FILES[f"{split}_csv"]
    df = pd.read_csv(path)
    if "gold_citations" not in df.columns:
        df["gold_citations"] = ""
    df["gold_list"] = df["gold_citations"].apply(split_gold)
    df["gold_set"] = df["gold_list"].apply(set)
    return df


queries = load_queries(RUN_SPLIT)
print("queries:", len(queries), "split:", RUN_SPLIT)
queries.head(2)
"""
    )
)

cells.append(
    md(
        """
## 5. Choice Card Generation

Choice cards are generated from actual corpus distributions. The LLM planner may only choose IDs from these cards.
"""
    )
)

cells.append(
    code(
        """
def table_to_records(df: pd.DataFrame) -> list[dict]:
    return loads_json(df.to_json(orient="records"))


def make_choice_cards(con: duckdb.DuckDBPyConnection) -> dict:
    law_codes = con.execute(
        '''
        SELECT
            coalesce(law_code, 'NO_CODE') AS id,
            count(*) AS source_count,
            min(article) AS min_article,
            max(article) AS max_article,
            max(unit_depth) AS max_unit_depth,
            any_value(law_code_family) AS law_code_family
        FROM law_segments
        GROUP BY 1
        ORDER BY source_count DESC
        '''
    ).fetchdf()

    law_titles = pd.DataFrame()
    if "law_citations" in {r[0] for r in con.execute("SHOW TABLES").fetchall()}:
        law_titles = con.execute(
            '''
            SELECT s.law_code, any_value(l.title) AS title
            FROM law_segments s
            JOIN law_citations l USING (citation)
            WHERE s.law_code IS NOT NULL
            GROUP BY s.law_code
            '''
        ).fetchdf()
    if not law_titles.empty:
        law_codes = law_codes.merge(law_titles, how="left", left_on="id", right_on="law_code").drop(columns=["law_code"])
    else:
        law_codes["title"] = None

    law_unit_chains = con.execute(
        '''
        SELECT coalesce(nullif(unit_chain, ''), 'article_only') AS id, count(*) AS source_count
        FROM law_segments
        GROUP BY 1
        ORDER BY source_count DESC
        '''
    ).fetchdf()

    court_families = con.execute(
        '''
        SELECT pattern AS id, subfamily, count(*) AS source_count
        FROM court_segments
        GROUP BY pattern, subfamily
        ORDER BY source_count DESC
        '''
    ).fetchdf()

    bge_divisions = con.execute(
        '''
        SELECT bge_division AS id, count(*) AS source_count
        FROM court_segments
        WHERE pattern = 'court_bge' AND bge_division IS NOT NULL
        GROUP BY 1
        ORDER BY source_count DESC
        '''
    ).fetchdf()
    bge_meanings = {
        "I": "constitutional and public-law leading decisions",
        "II": "public, administrative, tax, migration, and regulatory leading decisions",
        "III": "civil-law leading decisions",
        "IV": "criminal-law and criminal-procedure leading decisions",
        "V": "social-insurance leading decisions",
    }
    bge_divisions["meaning"] = bge_divisions["id"].map(bge_meanings).fillna("BGE division observed in corpus")

    docket_prefixes = con.execute(
        '''
        SELECT docket_prefix AS id, count(*) AS source_count, any_value(legal_area_code) AS legal_area_code
        FROM court_segments
        WHERE docket_prefix IS NOT NULL
        GROUP BY 1
        ORDER BY source_count DESC
        '''
    ).fetchdf()

    route_choices = [
        {"id": "explicit_exact_and_variants", "meaning": "Use exact query mentions and source-inventory variants."},
        {"id": "law_code_route", "meaning": "Filter laws by selected law codes and companion procedural codes."},
        {"id": "article_group_route", "meaning": "Expand selected article bases to all valid granular source rows."},
        {"id": "bge_base_route", "meaning": "Expand selected BGE bases to all source considerations."},
        {"id": "docket_base_route", "meaning": "Expand selected docket bases to all source considerations."},
        {"id": "statute_citing_cases", "meaning": "Add court considerations that cite selected law candidates."},
        {"id": "same_decision_neighbors", "meaning": "Keep neighboring considerations from selected decisions."},
        {"id": "regex_unknown_safety", "meaning": "Include malformed or unknown source rows matching selected docket-like families."},
        {"id": "dense_safety", "meaning": "Optional weak dense/lexical safety candidates; never primary."},
    ]

    time_choices = [
        {"id": "no_time_filter", "meaning": "Default. Query dates are treated as facts and do not filter precedent years."},
        {"id": "soft_recency", "meaning": "Boost newer decisions but do not hard filter."},
        {"id": "hard_year_range", "meaning": "Hard filter by year only when the query explicitly asks for a time period."},
        {"id": "exact_decision_date", "meaning": "Use only when an exact court decision date is explicitly requested."},
    ]

    cards = {
        "law_codes": table_to_records(law_codes),
        "law_unit_chains": table_to_records(law_unit_chains),
        "court_families": table_to_records(court_families),
        "bge_divisions": table_to_records(bge_divisions),
        "docket_prefixes": table_to_records(docket_prefixes),
        "route_choices": route_choices,
        "time_choices": time_choices,
    }
    return cards


choice_cards = make_choice_cards(con)
(ART_DIR / "choice_cards.json").write_text(dumps_json(choice_cards), encoding="utf-8")
print("Saved:", ART_DIR / "choice_cards.json")
for key, values in choice_cards.items():
    print(f"{key:16s}", len(values))
print("Top law codes:", choice_cards["law_codes"][:15])
print("Top docket prefixes:", choice_cards["docket_prefixes"][:15])
"""
    )
)

cells.append(
    md(
        """
## 6. Planner

The planner chooses controlled switches. It does **not** output candidate citations.

If `USE_PLANNER_LLM = False`, the notebook uses a broad deterministic fallback so the funnel and recall audit can run immediately.
"""
    )
)

cells.append(
    code(
        """
COMMON_SAFETY_LAW_CODES = {
    "BGG", "BV", "StBOG", "ZGB", "OR", "CO", "StPO", "CPP", "StGB", "CP", "ZPO", "CPC",
    "ATSG", "IVG", "KVG", "UVG", "SchKG", "IPRG", "DBG", "AIG"
}

PREFIX_HINTS = {
    "detention": ["1B", "7B"],
    "custody": ["1B", "7B"],
    "collusion": ["1B", "7B"],
    "criminal": ["1B", "6B", "7B"],
    "offence": ["6B", "1B"],
    "offense": ["6B", "1B"],
    "insurance": ["8C", "9C"],
    "invalidity": ["8C", "9C"],
    "employment": ["8C", "9C", "4A"],
    "contract": ["4A"],
    "company": ["4A"],
    "inheritance": ["5A"],
    "family": ["5A"],
    "civil": ["4A", "5A"],
}

LAW_HINTS = {
    "detention": ["StPO", "StGB", "BGG", "BV", "StBOG"],
    "collusion": ["StPO", "StGB", "BGG", "BV", "StBOG"],
    "criminal": ["StPO", "StGB", "BGG", "BV"],
    "insurance": ["ATSG", "IVG", "KVG", "UVG", "BGG"],
    "invalidity": ["ATSG", "IVG", "BGG"],
    "vocational": ["ATSG", "IVG", "BGG"],
    "contract": ["OR", "CO", "ZGB", "BGG"],
    "company": ["OR", "CO", "ZGB", "BGG"],
    "inheritance": ["ZGB", "BGG"],
    "family": ["ZGB", "BGG"],
    "civil": ["ZGB", "OR", "CO", "ZPO", "BGG"],
}


def allowed_ids(cards: list[dict]) -> set[str]:
    return {str(x.get("id")) for x in cards if x.get("id") is not None}


def heuristic_plan(query: str, cards: dict) -> dict:
    q = query.lower()
    mentions = extract_query_mentions(query)
    law_codes = set()
    for m in mentions["laws"]:
        if m.get("law_code"):
            law_codes.add(m["law_code"])
    for key, codes in LAW_HINTS.items():
        if key in q:
            law_codes.update(codes)
    if not law_codes:
        law_codes.update(COMMON_SAFETY_LAW_CODES)
    else:
        law_codes.update({"BGG", "BV", "StBOG"})

    allowed_law_codes = allowed_ids(cards["law_codes"])
    law_codes = sorted(c for c in law_codes if c in allowed_law_codes or c == "NO_CODE")

    prefixes = set(m.get("prefix") for m in mentions["dockets"] if m.get("prefix"))
    for key, vals in PREFIX_HINTS.items():
        if key in q:
            prefixes.update(vals)
    allowed_prefixes = allowed_ids(cards["docket_prefixes"])
    prefixes = sorted(p for p in prefixes if p in allowed_prefixes)

    bge_divisions = set()
    if any(k in q for k in ["criminal", "detention", "offence", "offense"]):
        bge_divisions.update(["I", "IV"])
    if any(k in q for k in ["civil", "contract", "inheritance", "family", "company"]):
        bge_divisions.add("III")
    if any(k in q for k in ["insurance", "invalidity", "benefits"]):
        bge_divisions.add("V")
    if not bge_divisions:
        bge_divisions.update(["I", "III", "IV", "V"])

    return {
        "law_codes": law_codes,
        "law_unit_chains": [],
        "court_families": ["court_bge", "court_case", "unknown"],
        "bge_divisions": sorted(bge_divisions),
        "docket_prefixes": prefixes,
        "routes": [
            "explicit_exact_and_variants",
            "law_code_route",
            "article_group_route",
            "bge_base_route",
            "docket_base_route",
            "statute_citing_cases",
            "same_decision_neighbors",
            "regex_unknown_safety",
        ],
        "time_mode": mentions["time_mode"],
        "law_budget": DEFAULT_LAW_BUDGET,
        "court_budget": DEFAULT_COURT_BUDGET,
        "notes": "deterministic broad fallback planner",
    }


planner_tokenizer = None
planner_model = None


def load_planner_model():
    global planner_tokenizer, planner_model
    if planner_model is not None:
        return planner_tokenizer, planner_model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    planner_tokenizer = AutoTokenizer.from_pretrained(PLANNER_MODEL_NAME, trust_remote_code=True)
    planner_model = AutoModelForCausalLM.from_pretrained(
        PLANNER_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return planner_tokenizer, planner_model


def compact_cards_for_prompt(cards: dict, top_law: int = 120, top_prefix: int = 80) -> dict:
    return {
        "law_codes": cards["law_codes"][:top_law],
        "law_unit_chains": cards["law_unit_chains"],
        "court_families": cards["court_families"],
        "bge_divisions": cards["bge_divisions"],
        "docket_prefixes": cards["docket_prefixes"][:top_prefix],
        "route_choices": cards["route_choices"],
        "time_choices": cards["time_choices"],
    }


def extract_first_json_object(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return loads_json(text[start : i + 1])
    raise ValueError("Unclosed JSON object")


def validate_plan(plan: dict, cards: dict) -> dict:
    allowed = {
        "law_codes": allowed_ids(cards["law_codes"]) | {"NO_CODE"},
        "law_unit_chains": allowed_ids(cards["law_unit_chains"]),
        "court_families": allowed_ids(cards["court_families"]) | {"unknown"},
        "bge_divisions": allowed_ids(cards["bge_divisions"]),
        "docket_prefixes": allowed_ids(cards["docket_prefixes"]),
        "routes": allowed_ids(cards["route_choices"]),
        "time_mode": allowed_ids(cards["time_choices"]),
    }
    clean = {}
    for key in ["law_codes", "law_unit_chains", "court_families", "bge_divisions", "docket_prefixes", "routes"]:
        values = plan.get(key) or []
        clean[key] = sorted({str(v) for v in values if str(v) in allowed[key]})
    time_mode = str(plan.get("time_mode") or "no_time_filter")
    clean["time_mode"] = time_mode if time_mode in allowed["time_mode"] else "no_time_filter"
    clean["law_budget"] = int(plan.get("law_budget") or DEFAULT_LAW_BUDGET)
    clean["court_budget"] = int(plan.get("court_budget") or DEFAULT_COURT_BUDGET)
    clean["notes"] = str(plan.get("notes") or "")
    return clean


def llm_plan(query: str, cards: dict) -> dict:
    tokenizer, model = load_planner_model()
    prompt_cards = compact_cards_for_prompt(cards)
    prompt = f'''
You are a Swiss legal retrieval planner. Choose only from the provided card IDs.
Do not output citations. Do not invent IDs.
Return strict JSON only with keys:
law_codes, law_unit_chains, court_families, bge_divisions, docket_prefixes, routes, time_mode, law_budget, court_budget, notes.

Important: dates in the facts are not court-year filters unless the query explicitly asks for a time period or recent/older law.

CHOICE_CARDS:
{dumps_json(prompt_cards)}

QUERY:
{query}
'''
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        do_sample=True,
    )[0][inputs.input_ids.shape[1] :]
    raw = tokenizer.decode(output_ids, skip_special_tokens=True)
    try:
        plan = extract_first_json_object(raw)
    except Exception:
        repair_prompt = "Repair this planner output into strict JSON only. Output no prose.\\n" + raw
        text = tokenizer.apply_chat_template([{"role": "user", "content": repair_prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        output_ids = model.generate(**inputs, max_new_tokens=2048, temperature=0.2, do_sample=False)[0][inputs.input_ids.shape[1] :]
        plan = extract_first_json_object(tokenizer.decode(output_ids, skip_special_tokens=True))
    return validate_plan(plan, cards)


def get_plan(query: str, cards: dict) -> dict:
    if USE_PLANNER_LLM:
        try:
            return llm_plan(query, cards)
        except Exception as exc:
            print("Planner failed, using heuristic fallback:", repr(exc))
    return validate_plan(heuristic_plan(query, cards), cards)
"""
    )
)

cells.append(
    md(
        """
## 7. Candidate Expansion Helpers
"""
    )
)

cells.append(
    code(
        """
def sql_quote_list(values: Iterable[str]) -> str:
    vals = [str(v).replace("'", "''") for v in values if v is not None]
    if not vals:
        return "('')"
    return "(" + ",".join(f"'{v}'" for v in vals) + ")"


def fetch_set(sql: str) -> set[str]:
    return set(con.execute(sql).fetchdf()["citation"].astype(str))


def create_temp_citation_table(name: str, citations: set[str] | list[str]) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    df = pd.DataFrame({"citation": list(citations)})
    con.register("_tmp_candidate_df", df)
    con.execute(f"CREATE OR REPLACE TEMP TABLE {safe_name} AS SELECT citation FROM _tmp_candidate_df")
    con.unregister("_tmp_candidate_df")
    return safe_name


def expand_explicit_law_mentions(mentions: dict) -> set[str]:
    out = set()
    for m in mentions["laws"]:
        article = m.get("article")
        law_code = m.get("law_code")
        if not article:
            continue
        if law_code:
            out |= fetch_set(
                f'''
                SELECT citation
                FROM law_segments
                WHERE law_code = '{law_code.replace("'", "''")}'
                  AND article = '{article.replace("'", "''")}'
                '''
            )
        else:
            out |= fetch_set(
                f'''
                SELECT citation
                FROM law_segments
                WHERE article = '{article.replace("'", "''")}'
                '''
            )
    return out


def expand_explicit_court_mentions(mentions: dict) -> set[str]:
    out = set()
    for m in mentions["bge"]:
        base = m.get("base")
        if base:
            out |= fetch_set(
                f'''
                SELECT citation
                FROM court_segments
                WHERE court_base = '{base.replace("'", "''")}'
                '''
            )
    for m in mentions["dockets"]:
        docket = m.get("docket")
        if docket:
            out |= fetch_set(
                f'''
                SELECT citation
                FROM court_segments
                WHERE court_base = '{docket.replace("'", "''")}'
                   OR docket = '{docket.replace("'", "''")}'
                '''
            )
    return out


def selected_law_code_candidates(plan: dict, keep_no_code: bool = True) -> set[str]:
    codes = set(plan.get("law_codes") or [])
    codes.update(c for c in COMMON_SAFETY_LAW_CODES if c in allowed_ids(choice_cards["law_codes"]))
    clauses = []
    if codes:
        clauses.append(f"law_code IN {sql_quote_list(codes)}")
    if keep_no_code:
        clauses.append("law_code IS NULL")
    if not clauses:
        return set(LAW_SOURCE)
    return fetch_set("SELECT citation FROM law_segments WHERE " + " OR ".join(clauses))


def selected_unit_chain_candidates(base: set[str], plan: dict) -> set[str]:
    chains = set(plan.get("law_unit_chains") or [])
    if not chains:
        return set(base)
    chain_sql = sql_quote_list(chains)
    selected = fetch_set(
        f'''
        SELECT citation
        FROM law_segments
        WHERE coalesce(nullif(unit_chain, ''), 'article_only') IN {chain_sql}
        '''
    )
    return selected & set(base)


def selected_court_family_candidates(plan: dict) -> set[str]:
    families = set(plan.get("court_families") or ["court_bge", "court_case", "unknown"])
    clauses = []
    if "court_bge" in families:
        clauses.append("pattern = 'court_bge'")
    if "court_case" in families:
        clauses.append("pattern = 'court_case'")
    if "unknown" in families:
        clauses.append("pattern = 'unknown'")
    if not clauses:
        return set(COURT_SOURCE)
    return fetch_set("SELECT citation FROM court_segments WHERE " + " OR ".join(clauses))


def selected_court_domain_candidates(base: set[str], plan: dict) -> set[str]:
    divisions = set(plan.get("bge_divisions") or [])
    prefixes = set(plan.get("docket_prefixes") or [])
    clauses = []
    if divisions:
        clauses.append(f"(pattern = 'court_bge' AND bge_division IN {sql_quote_list(divisions)})")
    if prefixes:
        clauses.append(f"(pattern IN ('court_case', 'unknown') AND docket_prefix IN {sql_quote_list(prefixes)})")
    clauses.append("pattern = 'unknown'")
    if not clauses:
        return set(base)
    selected = fetch_set(
        f'''
        SELECT citation
        FROM court_segments
        WHERE {' OR '.join(clauses)}
        '''
    )
    return selected & set(base)


def same_decision_expansion(seed_court: set[str]) -> set[str]:
    if not seed_court:
        return set()
    seed_sql = sql_quote_list(seed_court)
    bases = con.execute(
        f'''
        SELECT DISTINCT court_base
        FROM court_segments
        WHERE citation IN {seed_sql}
        '''
    ).fetchdf()["court_base"].dropna().astype(str).tolist()
    if not bases:
        return set()
    return fetch_set(
        f'''
        SELECT citation
        FROM court_segments
        WHERE court_base IN {sql_quote_list(bases)}
        '''
    )


def statute_citing_court_candidates(selected_laws: set[str]) -> set[str]:
    if not selected_laws or "citation_edges" not in {r[0] for r in con.execute("SHOW TABLES").fetchall()}:
        return set()
    tmp = create_temp_citation_table("tmp_selected_laws_for_edges", selected_laws)
    return fetch_set(
        f'''
        SELECT DISTINCT e.source AS citation
        FROM citation_edges e
        JOIN court_segments c ON c.citation = e.source
        JOIN {tmp} l ON l.citation = e.target
        WHERE e.dataset = 'court_considerations'
        '''
    )


def unknown_regex_safety(plan: dict) -> set[str]:
    prefixes = set(plan.get("docket_prefixes") or [])
    if not prefixes:
        return fetch_set("SELECT citation FROM court_segments WHERE pattern = 'unknown'")
    return fetch_set(
        f'''
        SELECT citation
        FROM court_segments
        WHERE pattern = 'unknown'
          AND (docket_prefix IN {sql_quote_list(prefixes)} OR raw IS NOT NULL)
        '''
    )
"""
    )
)

cells.append(
    md(
        """
## 8. Recall Audit And Budgeting
"""
    )
)

cells.append(
    code(
        """
audit_rows = []
dropped_rows = []


def split_gold_by_funnel(gold_set: set[str]) -> tuple[set[str], set[str]]:
    return gold_set & LAW_SOURCE, gold_set & COURT_SOURCE


def recall_of(candidates: set[str], gold: set[str]) -> float | None:
    if not gold:
        return None
    return len(candidates & gold) / len(gold)


def audit_stage(
    query_id: str,
    funnel: str,
    stage: str,
    before: set[str],
    after: set[str],
    gold: set[str],
    min_recall: float | None,
    drop_reason: str,
) -> set[str]:
    before_gold = before & gold
    after_gold = after & gold
    dropped = sorted(before_gold - after_gold)
    rec_before = recall_of(before, gold)
    rec_after = recall_of(after, gold)
    status = "ok"
    final_after = after

    if RUN_SPLIT == "val" and RECALL_GUARD_ON_VAL and gold and min_recall is not None and (rec_after or 0.0) < min_recall:
        status = "guard_reverted"
        final_after = before

    audit_rows.append(
        {
            "query_id": query_id,
            "funnel": funnel,
            "stage": stage,
            "candidate_count_before": len(before),
            "candidate_count_after": len(after),
            "candidate_count_final": len(final_after),
            "gold_count": len(gold),
            "gold_kept_before": len(before_gold),
            "gold_kept_after": len(after_gold),
            "gold_dropped": len(dropped),
            "recall_before": rec_before,
            "recall_after": rec_after,
            "min_recall": min_recall,
            "status": status,
            "drop_reason": drop_reason,
        }
    )

    for citation in dropped:
        dropped_rows.append(
            {
                "query_id": query_id,
                "funnel": funnel,
                "first_failed_stage": stage,
                "citation": citation,
                "drop_reason": drop_reason,
                "stage_status": status,
            }
        )

    return final_after


def score_law_candidates(candidates: set[str], query: str, plan: dict, explicit: set[str]) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame(columns=["citation", "score"])
    tmp = create_temp_citation_table("tmp_law_score_candidates", candidates)
    df = con.execute(
        f'''
        SELECT s.citation, s.law_code, s.article, s.unit_chain, coalesce(l.title, '') AS title
        FROM {tmp} tc
        JOIN law_segments s USING (citation)
        LEFT JOIN law_citations l USING (citation)
        '''
    ).fetchdf()
    selected_codes = set(plan.get("law_codes") or [])
    q = query[:2000]
    scores = []
    for row in df.itertuples(index=False):
        score = 0.0
        if row.citation in explicit:
            score += 1000
        if row.law_code in selected_codes:
            score += 80
        if row.law_code in COMMON_SAFETY_LAW_CODES:
            score += 20
        if row.title:
            score += 0.25 * fuzz.partial_ratio(q, str(row.title))
        if row.article and str(row.article) in q:
            score += 15
        scores.append(score)
    df["score"] = scores
    return df[["citation", "score"]].sort_values(["score", "citation"], ascending=[False, True])


def score_court_candidates(candidates: set[str], query: str, plan: dict, explicit: set[str], citing_cases: set[str]) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame(columns=["citation", "score"])
    tmp = create_temp_citation_table("tmp_court_score_candidates", candidates)
    df = con.execute(
        f'''
        SELECT c.citation, c.pattern, c.subfamily, c.docket_prefix, c.bge_division, c.decision_year, c.court_base, c.consideration
        FROM {tmp} tc
        JOIN court_segments c USING (citation)
        '''
    ).fetchdf()
    prefixes = set(plan.get("docket_prefixes") or [])
    divisions = set(plan.get("bge_divisions") or [])
    time_mode = plan.get("time_mode") or "no_time_filter"
    q = query[:2000]
    scores = []
    for row in df.itertuples(index=False):
        score = 0.0
        if row.citation in explicit:
            score += 1000
        if row.citation in citing_cases:
            score += 120
        if row.pattern == "court_bge":
            score += 40
        if row.pattern == "court_case":
            score += 30
        if row.pattern == "unknown":
            score += 5
        if row.docket_prefix in prefixes:
            score += 80
        if row.bge_division in divisions:
            score += 80
        if row.court_base and str(row.court_base) in q:
            score += 500
        if time_mode == "soft_recency" and row.decision_year and str(row.decision_year).isdigit():
            score += max(0, int(str(row.decision_year)[-4:]) - 1990) / 10
        scores.append(score)
    df["score"] = scores
    return df[["citation", "score"]].sort_values(["score", "citation"], ascending=[False, True])


def cap_candidates_with_guard(
    query_id: str,
    funnel: str,
    stage: str,
    before: set[str],
    scored: pd.DataFrame,
    budget: int,
    gold: set[str],
    explicit_keep: set[str],
    min_recall: float,
) -> set[str]:
    if len(before) <= budget:
        return before
    budget = max(0, int(budget))
    top = set(scored.head(budget)["citation"].astype(str)) | explicit_keep
    if RUN_SPLIT == "val" and gold and recall_of(top, gold) is not None and (recall_of(top, gold) or 0.0) < min_recall:
        top_diag = set(scored.head(DIAGNOSTIC_MAX_CANDIDATES)["citation"].astype(str)) | explicit_keep
        if (recall_of(top_diag, gold) or 0.0) >= min_recall:
            top = top_diag
        else:
            top = before
    return audit_stage(query_id, funnel, stage, before, top, gold, min_recall, "budget_cut")
"""
    )
)

cells.append(
    md(
        """
## 9. Funnel Runner
"""
    )
)

cells.append(
    code(
        """
def run_law_funnel(query_id: str, query: str, gold_law: set[str], plan: dict, mentions: dict) -> tuple[set[str], dict]:
    explicit = expand_explicit_law_mentions(mentions)
    stages = {}

    s0 = set(LAW_SOURCE)
    stages["L0_all_laws"] = s0
    s0 = audit_stage(query_id, "law", "L0_all_laws", s0, s0, gold_law, 1.0 if gold_law else None, "baseline")

    s1_raw = selected_law_code_candidates(plan, keep_no_code=True)
    s1 = s1_raw | explicit
    s1 = audit_stage(query_id, "law", "L1_domain_or_law_family", s0, s1, gold_law, EARLY_STAGE_MIN_RECALL, "planner_missed_choice")
    stages["L1_domain_or_law_family"] = s1

    s2_raw = selected_law_code_candidates(plan, keep_no_code=True)
    s2 = s2_raw | explicit
    s2 = audit_stage(query_id, "law", "L2_law_code", s1, s2, gold_law, EARLY_STAGE_MIN_RECALL, "planner_missed_choice")
    stages["L2_law_code"] = s2

    article_group = explicit if explicit else s2
    s3 = audit_stage(query_id, "law", "L3_article_group", s2, article_group, gold_law, MID_STAGE_MIN_RECALL, "filter_too_strict")
    stages["L3_article_group"] = s3

    s4_raw = selected_unit_chain_candidates(s3, plan)
    s4 = s4_raw | explicit
    s4 = audit_stage(query_id, "law", "L4_unit_depth", s3, s4, gold_law, MID_STAGE_MIN_RECALL, "filter_too_strict")
    stages["L4_unit_depth"] = s4

    scored = score_law_candidates(s4, query, plan, explicit)
    s5 = cap_candidates_with_guard(
        query_id,
        "law",
        "L5_budget",
        s4,
        scored,
        int(plan.get("law_budget") or DEFAULT_LAW_BUDGET),
        gold_law,
        explicit,
        FINAL_MIN_QUERY_RECALL,
    )
    stages["L5_budget"] = s5
    return s5, {"explicit_law_candidates": sorted(explicit), "stages": {k: len(v) for k, v in stages.items()}}


def run_court_funnel(
    query_id: str,
    query: str,
    gold_court: set[str],
    plan: dict,
    mentions: dict,
    selected_laws: set[str],
) -> tuple[set[str], dict]:
    explicit = expand_explicit_court_mentions(mentions)
    stages = {}

    s0 = set(COURT_SOURCE)
    stages["C0_all_court"] = s0
    s0 = audit_stage(query_id, "court", "C0_all_court", s0, s0, gold_court, 1.0 if gold_court else None, "baseline")

    s1_raw = selected_court_family_candidates(plan)
    s1 = s1_raw | explicit
    s1 = audit_stage(query_id, "court", "C1_court_family", s0, s1, gold_court, EARLY_STAGE_MIN_RECALL, "planner_missed_choice")
    stages["C1_court_family"] = s1

    s2_raw = selected_court_domain_candidates(s1, plan)
    s2 = s2_raw | explicit
    s2 = audit_stage(query_id, "court", "C2_domain_prefix", s1, s2, gold_court, EARLY_STAGE_MIN_RECALL, "planner_missed_choice")
    stages["C2_domain_prefix"] = s2

    decision_seed = explicit
    if decision_seed:
        decision_expanded = same_decision_expansion(decision_seed)
        s3_raw = (s2 & decision_expanded) | decision_expanded | explicit
    else:
        s3_raw = s2
    s3 = audit_stage(query_id, "court", "C3_decision_level", s2, s3_raw, gold_court, MID_STAGE_MIN_RECALL, "route_missing")
    stages["C3_decision_level"] = s3

    citing_cases = statute_citing_court_candidates(selected_laws)
    s4_raw = (s3 | citing_cases | explicit)
    s4 = audit_stage(query_id, "court", "C4_statute_citing_cases", s3, s4_raw, gold_court, MID_STAGE_MIN_RECALL, "route_missing")
    stages["C4_statute_citing_cases"] = s4

    unknown_safety = unknown_regex_safety(plan)
    s5_raw = s4 | unknown_safety | explicit
    s5 = audit_stage(query_id, "court", "C5_regex_and_unknown_safety", s4, s5_raw, gold_court, MID_STAGE_MIN_RECALL, "regex_gap")
    stages["C5_regex_and_unknown_safety"] = s5

    scored = score_court_candidates(s5, query, plan, explicit, citing_cases)
    s6 = cap_candidates_with_guard(
        query_id,
        "court",
        "C6_budget",
        s5,
        scored,
        int(plan.get("court_budget") or DEFAULT_COURT_BUDGET),
        gold_court,
        explicit,
        FINAL_MIN_QUERY_RECALL,
    )
    stages["C6_budget"] = s6
    return s6, {
        "explicit_court_candidates": sorted(explicit),
        "citing_case_count": len(citing_cases),
        "unknown_safety_count": len(unknown_safety),
        "stages": {k: len(v) for k, v in stages.items()},
    }


def merge_candidate_sets(query_id: str, law_candidates: set[str], court_candidates: set[str], gold_set: set[str]) -> set[str]:
    merged = set(law_candidates) | set(court_candidates)
    if len(merged) <= MAX_TOTAL_CANDIDATES:
        return merged

    # The law/court funnels have already budgeted independently. If the merged pool is still too large,
    # preserve validation recall by refusing to cap below the gate.
    if RUN_SPLIT == "val" and gold_set:
        rec = recall_of(merged, gold_set)
        if rec is not None and rec < FINAL_MIN_QUERY_RECALL:
            return merged

    return merged
"""
    )
)

cells.append(
    md(
        """
## 10. Run Candidate Funnel
"""
    )
)

cells.append(
    code(
        """
candidate_output_path = ART_DIR / f"{RUN_SPLIT}_candidate_sets.jsonl"
plan_output_path = ART_DIR / f"{RUN_SPLIT}_planner_outputs.jsonl"

candidate_rows = []
plan_rows = []

with candidate_output_path.open("w", encoding="utf-8") as cand_out, plan_output_path.open("w", encoding="utf-8") as plan_out:
    for row in queries.itertuples(index=False):
        query_id = str(getattr(row, "query_id"))
        query = str(getattr(row, "query"))
        gold_set = set(getattr(row, "gold_set", set()) or set())
        gold_law, gold_court = split_gold_by_funnel(gold_set)

        mentions = extract_query_mentions(query)
        plan = get_plan(query, choice_cards)
        plan_rows.append({"query_id": query_id, "plan": plan, "mentions": mentions})
        plan_out.write(dumps_json({"query_id": query_id, "plan": plan, "mentions": mentions}) + "\\n")

        print("\\n===", query_id, "===")
        print("gold law/court:", len(gold_law), len(gold_court))
        print("plan:", plan)

        law_candidates, law_diag = run_law_funnel(query_id, query, gold_law, plan, mentions)
        court_candidates, court_diag = run_court_funnel(query_id, query, gold_court, plan, mentions, law_candidates)
        merged = merge_candidate_sets(query_id, law_candidates, court_candidates, gold_set)

        record = {
            "query_id": query_id,
            "candidate_count": len(merged),
            "law_candidate_count": len(law_candidates),
            "court_candidate_count": len(court_candidates),
            "recall": recall_of(merged, gold_set),
            "law_recall": recall_of(law_candidates, gold_law),
            "court_recall": recall_of(court_candidates, gold_court),
            "candidates": sorted(merged),
            "law_diag": law_diag,
            "court_diag": court_diag,
        }
        candidate_rows.append(record)
        cand_out.write(dumps_json(record) + "\\n")
        print("candidate_count:", len(merged), "recall:", record["recall"], "law:", record["law_recall"], "court:", record["court_recall"])

print("Saved candidates:", candidate_output_path)
print("Saved plans     :", plan_output_path)
"""
    )
)

cells.append(
    md(
        """
## 11. Recall Diagnostics
"""
    )
)

cells.append(
    code(
        """
audit_df = pd.DataFrame(audit_rows)
dropped_df = pd.DataFrame(dropped_rows)
summary_df = pd.DataFrame(
    [
        {
            "query_id": r["query_id"],
            "candidate_count": r["candidate_count"],
            "law_candidate_count": r["law_candidate_count"],
            "court_candidate_count": r["court_candidate_count"],
            "recall": r["recall"],
            "law_recall": r["law_recall"],
            "court_recall": r["court_recall"],
        }
        for r in candidate_rows
    ]
)

audit_path = ART_DIR / "stage_recall_audit.csv"
dropped_path = ART_DIR / "dropped_gold_diagnostics.csv"
summary_path = ART_DIR / f"{RUN_SPLIT}_candidate_summary.csv"

audit_df.to_csv(audit_path, index=False)
dropped_df.to_csv(dropped_path, index=False)
summary_df.to_csv(summary_path, index=False)

print("Saved:", audit_path)
print("Saved:", dropped_path)
print("Saved:", summary_path)

display(summary_df)
if not audit_df.empty:
    display(
        audit_df.groupby(["funnel", "stage", "status"], dropna=False)
        .agg(
            queries=("query_id", "nunique"),
            mean_after_count=("candidate_count_final", "mean"),
            min_recall_after=("recall_after", "min"),
            mean_recall_after=("recall_after", "mean"),
            total_dropped=("gold_dropped", "sum"),
        )
        .reset_index()
    )

if RUN_SPLIT == "val" and not summary_df.empty:
    print("Mean recall:", summary_df["recall"].mean())
    print("Min recall :", summary_df["recall"].min())
    print("Mean law recall:", summary_df["law_recall"].dropna().mean())
    print("Mean court recall:", summary_df["court_recall"].dropna().mean())
    bad = summary_df[summary_df["recall"].fillna(1.0) < FINAL_MIN_QUERY_RECALL]
    if len(bad):
        print("Queries below final gate:")
        display(bad)
    else:
        print("All validation queries pass the final per-query recall gate.")
"""
    )
)

cells.append(
    md(
        """
## 12. Inspect First Dropped Gold Citations

Use this before tightening any filter. A filter is allowed only if the dropped citation has a clear recovery route or the drop is accepted as rare.
"""
    )
)

cells.append(
    code(
        """
if dropped_df.empty:
    print("No gold citations were dropped by audited stages, or no gold labels are available.")
else:
    display(dropped_df.head(100))
    display(dropped_df.groupby(["funnel", "first_failed_stage", "drop_reason", "stage_status"]).size().reset_index(name="n"))
"""
    )
)

cells.append(
    md(
        """
## 13. Optional Reranker Skeleton

Run this only after candidate recall is acceptable. Reranking is not allowed to add invented citations; it can only score existing candidate IDs.
"""
    )
)

cells.append(
    code(
        """
reranker_tokenizer = None
reranker_model = None


def load_reranker_model():
    global reranker_tokenizer, reranker_model
    if reranker_model is not None:
        return reranker_tokenizer, reranker_model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME, padding_side="left", trust_remote_code=True)
    reranker_model = AutoModelForCausalLM.from_pretrained(
        RERANKER_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    return reranker_tokenizer, reranker_model


def candidate_evidence_card(citation: str) -> str:
    if citation in LAW_SOURCE:
        row = con.execute(
            f'''
            SELECT s.citation, s.law_code, s.article, s.unit_chain, coalesce(l.title, '') AS title, left(coalesce(l.text, ''), 1200) AS text
            FROM law_segments s
            LEFT JOIN law_citations l USING (citation)
            WHERE s.citation = '{citation.replace("'", "''")}'
            LIMIT 1
            '''
        ).fetchone()
        if row:
            return f"Citation: {row[0]}\\nType: law\\nLaw code: {row[1]}\\nArticle: {row[2]}\\nUnits: {row[3]}\\nTitle: {row[4]}\\nText: {row[5]}"
    row = con.execute(
        f'''
        SELECT c.citation, c.pattern, c.subfamily, c.court_base, c.docket_prefix, c.bge_division, c.decision_year,
               left(coalesce(t.text, ''), 1200) AS text
        FROM court_segments c
        LEFT JOIN court_considerations t USING (citation)
        WHERE c.citation = '{citation.replace("'", "''")}'
        LIMIT 1
        '''
    ).fetchone()
    if row:
        return f"Citation: {row[0]}\\nType: court\\nPattern: {row[1]}\\nSubfamily: {row[2]}\\nDecision: {row[3]}\\nPrefix: {row[4]}\\nBGE division: {row[5]}\\nYear: {row[6]}\\nText: {row[7]}"
    return f"Citation: {citation}"


def rerank_candidates_for_query(query: str, candidates: list[str], limit: int = 500) -> pd.DataFrame:
    # Placeholder scoring interface. For very large pools, first use the funnel score or a smaller reranker batch.
    cards = [candidate_evidence_card(c) for c in candidates[:limit]]
    return pd.DataFrame({"citation": candidates[:limit], "evidence": cards})


if USE_RERANKER:
    load_reranker_model()
    print("Reranker loaded:", RERANKER_MODEL_NAME)
else:
    print("Reranker disabled. Set USE_RERANKER=True after candidate recall passes.")
"""
    )
)

cells.append(
    md(
        """
## 14. Export Submission Candidate Diagnostics

For `RUN_SPLIT = "test"`, this writes candidate sets only. Final prediction selection should happen after reranking/threshold calibration.
"""
    )
)

cells.append(
    code(
        """
if RUN_SPLIT == "test":
    test_diag = pd.DataFrame(
        [
            {
                "query_id": r["query_id"],
                "candidate_count": r["candidate_count"],
                "law_candidate_count": r["law_candidate_count"],
                "court_candidate_count": r["court_candidate_count"],
            }
            for r in candidate_rows
        ]
    )
    path = ART_DIR / "test_candidate_diagnostics.csv"
    test_diag.to_csv(path, index=False)
    display(test_diag)
    print("Saved:", path)
"""
    )
)


nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


OUT.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT}")
