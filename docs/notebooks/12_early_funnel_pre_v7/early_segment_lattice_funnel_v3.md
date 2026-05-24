# early_segment_lattice_funnel_v3

**Path:** e:\swiss_citation_extraction\notebooks\12_early_funnel_pre_v7\early_segment_lattice_funnel_v3.ipynb

## Configuration

- `BASE_DIR`: `/content/drive/MyDrive/swiss_law` (Colab) or `..` resolved (local).
- `DATA_DIR`: `BASE_DIR/data`
- `INSIGHTS_DIR`: `BASE_DIR/data_insights`
- `ART_DIR`: `BASE_DIR/artifacts`
- `SCRIPT_DIR`: `BASE_DIR/scripts` (requires `segment_lattice_v3.py`, `extract_citation_graph.py`)
- `REFRESH_EXTRACTION = False` (skip re-running `extract_citation_graph.py`).
- `INVALIDATE_OLD_FUNNEL_ARTIFACTS = True` (purge stale `recall_funnel.duckdb`, `choice_cards.json`, candidate/planner JSONLs).
- `DB_PATH`: `ART_DIR/segment_lattice_v3.sqlite`
- `CARDS_PATH`: `ART_DIR/choice_cards_v3.json`
- `COMPACT_CARDS_PATH`: `ART_DIR/choice_cards_v3_compact.json`
- `TOP_PER_SEGMENT`: law_code top 80; BGE division top 5; docket_prefix top 30; legal_area_code top 30.
- `RUN_SPLIT = "val"`
- `LAW_BUDGET = 2000`, `COURT_BUDGET = 2000`
- `USE_HEURISTIC_FALLBACK = False`
- `USE_HYBRID_PLANNER = True` (union LLM + heuristic + parser decisions).
- `LOG_FIRST_N_QWEN_RAW = 3`
- LLM model: `Qwen/Qwen3-32B`, `QWEN_MAX_NEW_TOKENS = 512`, `QWEN_TEMPERATURE = 0.0`, `QWEN_LOAD_IN_4BIT = False` (bf16 on ~95 GB VRAM), `QWEN_USE_FLASH_ATTN_2 = True` (falls back to default attention when flash-attn missing).
- Output artifacts: `val_segment_lattice_v3_oracle_planner_outputs.jsonl`, `val_segment_lattice_v3_llm_planner_outputs.jsonl`, `val_segment_lattice_v3_oracle_candidate_summary.csv`, `val_segment_lattice_v3_llm_candidate_summary.csv`, `val_segment_lattice_v3_oracle_dropped_gold.csv`, `val_segment_lattice_v3_llm_dropped_gold.csv`, `val_segment_lattice_v3_llm_raw_log.jsonl`, per-stage recall CSVs `val_segment_lattice_v3_{oracle,llm}_stage_recall.csv`.

## Data

- Source CSVs under `BASE_DIR/data` (split files: `val.csv`, etc.). Used by `extract_citation_graph.py` and the funnel runner.
- Classified citation JSONLs under `BASE_DIR/data_insights/*_classified_citations.jsonl` (consumed by `build_index`).
- `laws_de` scan: 197,945 rows scanned, 175,933 source rows.
- `court_considerations` scan: 2,416,056 rows scanned, 1,985,178 source rows.
- Court co-signals scan: 2,476,315 court rows scanned, 2,476,315 matched_to_segments, 69 options with signals; computes per-option top co-cited law codes and top German keywords.
- Card groups built: 48 total; 37 court groups (BGE + court_case + unknown families).
- BGE division options (5): V (22217), III (21772), II (21471), I (13733), IV (12446).
- docket_prefix options: 74 total, 68 with `co_signals` (top by source_count: 2C=270343, 6B=260157, 5A=195246, 8C=192621, 9C=170565, 1C=151039, 4A=150686, 1B=75847, I=45755, 7B=43602).
- Compact card groups: `court_considerations.court.court_bge.division`, `court_considerations.court.court_case.docket_prefix`, `court_considerations.court.court_case.legal_area_code`, `court_considerations.unknown.unknown.docket_prefix`, `court_considerations.unknown.unknown.legal_area_code`, `laws_de.law.statute_article.law_code`. Compact file size: 130,568 bytes.
- Split: `val` (10 queries: val_001..val_010). Gold loaded from `val.csv` `gold_citations` column.

## Pipeline

1. **Setup** — Mount Drive (when in Colab), import `segment_lattice_v3`, assert helper scripts exist, create `ART_DIR`.
2. **Optional extraction refresh** — If `REFRESH_EXTRACTION` is True, subprocess `extract_citation_graph.py` to rebuild classified citation JSONLs into `data_insights/`. Skipped here.
3. **Invalidate old funnel artifacts** — Verify new code is loaded via `"court_funnel" in inspect.getsource(segment_lattice_v3.hard_filter_base)`; call `invalidate_stale_artifacts(ART_DIR)` to remove old coarse-funnel outputs while preserving source data.
4. **Build segment-lattice index + cards + co-signals** — `build_index(...)` (force=True) writes `segment_lattice_v3.sqlite` and `choice_cards_v3.json`; scans `court_considerations.csv` text to compute per-option co-cited `law_code` frequencies and distinctive German keywords (`text_keywords`).
5. **Verify court cards** — Print 5 BGE division options and first 10 docket_prefix options with their co_signals and keywords; assert BGE divisions `{I,II,III,IV,V}` present; assert docket_prefixes `{1B,4A,5A,6B,8C,9C,2C,7B}` present; assert ≥10 prefix options carry co_signals.
6. **Compact card bundle** — Trim per-segment to top semantic options (law_code 80, division 5, docket_prefix 30, legal_area_code 30; others top 50 semantic) and reshape each option to `{id, meaning_en, source_count, selector_kind, examples[:3], co_signals[:6], text_keywords[:6]}`. Persist as `choice_cards_v3_compact.json`.
7. **Oracle ceiling run** — `oracle_plan_for_split` derives gold-based `segment_decisions` (perfect law + court selection). `run_candidates` (budgets 2000/2000, planner_input=oracle plan) writes `val_segment_lattice_v3_candidate_sets.jsonl`. `audit_candidates` writes summary + dropped-gold CSVs. Files are copied to `*_oracle_*` snapshots so the LLM run can later overwrite the originals without losing the oracle ceiling.
8. **LLM planner (Qwen3-32B)** — Load `Qwen/Qwen3-32B` in bf16 (no 4-bit), no flash-attn fallback. Two prompt passes per query: (1) pick `law_codes` from compact list; (2) pick `divisions`, `divisions_maybe`, `docket_prefixes`, `docket_prefixes_maybe` using prior law picks + co_signals + keywords. Outputs parsed by regex JSON extractor with truncation fallback; ids filtered against allowed sets (`LAW_IDS`, `DIV_IDS`, `PREFIX_IDS`). Parser-derived law codes from `derive_decisions(conn, query)` are unioned into the Qwen law list.
   - **Hybrid planner** — Unions LLM include picks + heuristic maybe picks + parser decisions via `merge_two_plans` keyed by `(dataset, family, segment_key, mode)`. Heuristic picks divisions/prefixes whose `co_signals` overlap picked law codes; if no law detected, falls back to all 5 divisions + top 12 prefixes as `mode=maybe`.
   - **Validation** — All decisions pass through `validate_external_decisions(conn, ...)` so unknown ids are silently dropped.
   - **Logging** — First 3 queries print raw Qwen responses; all raw responses persisted to `val_segment_lattice_v3_llm_raw_log.jsonl`.
9. **Funnel run with LLM plan + audit** — `run_candidates(..., planner_input=LLM_PLAN_PATH)` produces candidate sets; `audit_candidates` writes summary; results copied to `*_llm_*` snapshots.
10. **Per-stage recall trace** — Build a `family_by_citation` map (laws_de→law, court_considerations→court) from `source_citations`. `load_gold` reads `;`-separated `gold_citations` per query. `replay_stages` walks `stage_diagnostics[funnel]` for each query and, for each `hard_filter:<segment_key>` stage, queries `citation_segments` for the set of citations matching the stage's selected options, intersects cumulatively into a `running` set, and reports recall against the family-filtered gold. Source stage assumed to cover all family gold. `topk_budget` stage uses the final candidate list. Trace CSVs: `val_segment_lattice_v3_oracle_stage_recall.csv` and `val_segment_lattice_v3_llm_stage_recall.csv`; aggregated by `(funnel, stage)` printed with recall mean/median and median candidate count.
11. **Final summary** — `aggregate(load_summary(...))` prints mean/median/min over per-query `recall`, `law_recall`, `court_recall`, plus mean/median/max for `candidate_count`, `law_candidate_count`, `court_candidate_count` for both ORACLE and LLM PLANNER summaries.

## Results

```
Mounted at /content/drive
BASE_DIR    : /content/drive/MyDrive/swiss_law
DATA_DIR    : /content/drive/MyDrive/swiss_law/data
INSIGHTS_DIR: /content/drive/MyDrive/swiss_law/data_insights
ART_DIR     : /content/drive/MyDrive/swiss_law/artifacts
SCRIPT_DIR  : /content/drive/MyDrive/swiss_law/scripts

Skipping extraction refresh. Set REFRESH_EXTRACTION=True when parser/data changed.

True
Removed 0 old artifacts
```

Index build:

```
laws_de: done scanned=197,945 source_rows=175,933
court_considerations: scanned=500,000 source_rows=450,934
court_considerations: scanned=1,000,000 source_rows=907,418
court_considerations: scanned=1,500,000 source_rows=1,365,237
court_considerations: scanned=2,000,000 source_rows=1,772,053
court_considerations: done scanned=2,416,056 source_rows=1,985,178
Computing court co-signals from court_considerations.csv text...
  co-signals scan: 250,000 rows, 250,000 matched
  ...
  co-signals scan: 2,250,000 rows, 2,250,000 matched
co-signals built: court rows scanned=2,476,315, matched_to_segments=2,476,315, options_with_signals=69
Built v3 index: /content/drive/MyDrive/swiss_law/artifacts/segment_lattice_v3.sqlite
Saved v3 cards: /content/drive/MyDrive/swiss_law/artifacts/choice_cards_v3.json
DB_PATH   : /content/drive/MyDrive/swiss_law/artifacts/segment_lattice_v3.sqlite
CARDS_PATH: /content/drive/MyDrive/swiss_law/artifacts/choice_cards_v3.json
```

Card verification:

```
card groups: 48
court groups: ['court_considerations.court.court_bge.court_base', 'court_considerations.court.court_bge.division',
 'court_considerations.court.court_bge.page', 'court_considerations.court.court_bge.pattern',
 'court_considerations.court.court_bge.pinpoint', 'court_considerations.court.court_bge.pinpoint_unit',
 'court_considerations.court.court_bge.reporter', 'court_considerations.court.court_bge.subfamily',
 'court_considerations.court.court_bge.volume', 'court_considerations.court.court_case.consideration',
 'court_considerations.court.court_case.court_base', 'court_considerations.court.court_case.court_chamber',
 'court_considerations.court.court_case.decision_date', 'court_considerations.court.court_case.decision_year',
 'court_considerations.court.court_case.docket', 'court_considerations.court.court_case.docket_prefix',
 'court_considerations.court.court_case.legal_area_code', 'court_considerations.court.court_case.pattern',
 'court_considerations.court.court_case.separator_style', 'court_considerations.court.court_case.serial_number',
 'court_considerations.court.court_case.subfamily', 'court_considerations.unknown.unknown.consideration',
 'court_considerations.unknown.unknown.consideration_raw', 'court_considerations.unknown.unknown.court_base',
 'court_considerations.unknown.unknown.court_chamber', 'court_considerations.unknown.unknown.decision_date',
 'court_considerations.unknown.unknown.decision_year', 'court_considerations.unknown.unknown.docket',
 'court_considerations.unknown.unknown.docket_prefix', 'court_considerations.unknown.unknown.fallback_pattern',
 'court_considerations.unknown.unknown.fallback_subfamily', 'court_considerations.unknown.unknown.legal_area_code',
 'court_considerations.unknown.unknown.pattern', 'court_considerations.unknown.unknown.raw',
 'court_considerations.unknown.unknown.separator_style', 'court_considerations.unknown.unknown.serial_number',
 'court_considerations.unknown.unknown.subfamily']

=== court_considerations.court.court_bge.division (5 options) ===
  id=V        src=  22217 kind=semantic     co=[ATSG(0.0752), BVG(0.061), IVG(0.0537), KVG(0.052), AHVG(0.0407)]
           keywords=[Sinne, Vorinstanz, Anspruch, Januar, Person, Frage]
  id=III      src=  21772 kind=semantic     co=[ZGB(0.1133), OR(0.0978), ZPO(0.073), SchKG(0.0699), IPRG(0.0325)]
           keywords=[Vorinstanz, Kommentar, Recht, Frage, Bundesgericht, Beschwerdeführerin]
  id=II       src=  21471 kind=semantic     co=[BV(0.0467), DBG(0.0301), BGG(0.03), USG(0.0231), RPG(0.02)]
           keywords=[Bundesgericht, Vorinstanz, Sinne, Beschwerdeführer, Recht, Frage]
  id=I        src=  13733 kind=semantic     co=[BV(0.1749), BGG(0.0434), KV(0.0302), 116(0.0268), StPO(0.0221)]
           keywords=[Beschwerdeführer, Bundesgericht, Recht, Sinne, Hinweisen, Frage]
  id=IV       src=  12446 kind=semantic     co=[StGB(0.22), StPO(0.1456), BGG(0.0324), BV(0.0281), 116(0.0248)]
           keywords=[Vorinstanz, Beschwerdeführer, Sinne, Kommentar, Rechtsprechung, Person]

=== court_considerations.court.court_case.docket_prefix (74 options) ===
  id=2C       src= 270343 kind=semantic     co=[BGG(0.1364), BV(0.0441), AIG(0.0174), DBG(0.0165), StHG(0.0122)]
           keywords=[Beschwerde, Tribunal, Beschwerdeführer, Vorinstanz, Bundesgericht, Kantons]
  id=6B       src= 260157 kind=semantic     co=[BGG(0.1027), StGB(0.0658), StPO(0.0493), BV(0.0281), SVG(0.0079)]
           keywords=[Beschwerdeführer, Beschwerde, Vorinstanz, Tribunal, Kantons, Verfahren]
  id=5A       src= 195246 kind=semantic     co=[BGG(0.1355), ZGB(0.0531), ZPO(0.0298), SchKG(0.0291), BV(0.0274)]
           keywords=[Beschwerde, Beschwerdeführer, Tribunal, Entscheid, Obergericht, Beschwerdeführerin]
  id=8C       src= 192621 kind=semantic     co=[BGG(0.1437), ATSG(0.0552), UVG(0.0243), IVG(0.0194), BV(0.0149)]
           keywords=[Beschwerde, Vorinstanz, Entscheid, Beschwerdeführer, Kantons, Gericht]
  id=9C       src= 170565 kind=semantic     co=[BGG(0.1221), ATSG(0.0415), IVG(0.0251), BVG(0.0223), BV(0.0143)]
           keywords=[Beschwerde, Vorinstanz, Kantons, Tribunal, Entscheid, Beschwerdeführerin]
  id=1C       src= 151039 kind=semantic     co=[BGG(0.1244), BV(0.0461), RPG(0.0225), PBG(0.0134), SVG(0.0109)]
           keywords=[Beschwerde, Beschwerdeführer, Tribunal, Entscheid, Bundesgericht, Vorinstanz]
  id=4A       src= 150686 kind=semantic     co=[BGG(0.1034), ZPO(0.0408), OR(0.04), BV(0.0197), 116(0.0195)]
           keywords=[Vorinstanz, Tribunal, Beschwerde, Beschwerdeführerin, Beschwerdeführer, Bundesgericht]
  id=1B       src=  75847 kind=semantic     co=[BGG(0.1321), StPO(0.1018), BV(0.0353), StGB(0.0253), 116(0.0076)]
           keywords=[Beschwerde, Beschwerdeführer, Tribunal, Staatsanwaltschaft, Entscheid, Kantons]
  id=I        src=  45755 kind=semantic     co=[IVG(0.1078), ATSG(0.0535), IVV(0.0458), BSV(0.0208), 112(0.0115)]
           keywords=[Stelle, Kantons, Verwaltungsgerichtsbeschwerde, Entscheid, Verfügung, Januar]
  id=7B       src=  43602 kind=semantic     co=[BGG(0.1154), StPO(0.0658), SchKG(0.0326), StGB(0.0307), BV(0.0229)]
           keywords=[Beschwerde, Beschwerdeführer, Vorinstanz, Tribunal, Bundesgericht, Verfahren]

docket_prefix options with co_signals: 68 / 74
```

Compact card bundle:

```
compact card groups: ['court_considerations.court.court_bge.division',
 'court_considerations.court.court_case.docket_prefix',
 'court_considerations.court.court_case.legal_area_code',
 'court_considerations.unknown.unknown.docket_prefix',
 'court_considerations.unknown.unknown.legal_area_code',
 'laws_de.law.statute_article.law_code']
compact size (bytes): 130568
```

Oracle run (val):

```
Saved oracle planner output: /content/drive/MyDrive/swiss_law/artifacts/val_segment_lattice_v3_oracle_planner_outputs.jsonl
val_001 law= 2000 court= 2000 total= 4000
val_002 law=  954 court= 2000 total= 2954
val_003 law= 2000 court= 2000 total= 4000
val_004 law= 2000 court= 2000 total= 4000
val_005 law= 2000 court= 2000 total= 4000
val_006 law= 2000 court= 2000 total= 4000
val_007 law= 2000 court= 2000 total= 4000
val_008 law= 2000 court= 2000 total= 4000
val_009 law= 2000 court= 2000 total= 4000
val_010 law= 2000 court= 2000 total= 4000
Saved candidates: /content/drive/MyDrive/swiss_law/artifacts/val_segment_lattice_v3_candidate_sets.jsonl
Saved plans     : /content/drive/MyDrive/swiss_law/artifacts/val_segment_lattice_v3_planner_outputs.jsonl
Saved summary: /content/drive/MyDrive/swiss_law/artifacts/val_segment_lattice_v3_candidate_summary.csv
Saved dropped: /content/drive/MyDrive/swiss_law/artifacts/val_segment_lattice_v3_dropped_gold.csv

Oracle outputs snapshotted:
  /content/drive/MyDrive/swiss_law/artifacts/val_segment_lattice_v3_oracle_candidate_summary.csv
  /content/drive/MyDrive/swiss_law/artifacts/val_segment_lattice_v3_oracle_dropped_gold.csv
```

Qwen3-32B load + sample raw responses (first 3 queries):

```
Loading Qwen/Qwen3-32B in bf16... (first call only)
  flash-attn not installed; using default attention.
  loaded. device map: single
  GPU mem allocated after load: 61.0 GB
  --- qwen LAW raw ---
 ```json
{"law_codes": ["StPO", "BV"]}
``` 
  --------------------
  --- qwen COURT raw ---
 ```json
{
  "divisions": ["IV"],
  "divisions_maybe": ["I", "II"],
  "docket_prefixes": ["1B", "6P", "7B"],
  "docket_prefixes_maybe": ["2P", "4D", "5P"]
}
``` 
  ----------------------
  --- qwen LAW raw ---
 ```json
{"law_codes": ["IVG", "IVV"]}
``` 
  --------------------
  --- qwen COURT raw ---
 ```json
{
  "divisions": ["V"],
  "divisions_maybe": ["II", "I"],
  "docket_prefixes": ["8C", "9C", "I"],
  "docket_prefixes_maybe": ["C", "B", "H"]
}
``` 
  ----------------------
  --- qwen LAW raw ---
 ```json
{"law_codes": ["StPO"]}
``` 
  --------------------
  --- qwen COURT raw ---
 ```json
{
  "divisions": ["IV"],
  "divisions_maybe": ["I", "II"],
  "docket_prefixes": ["6B", "1B", "7B", "6P"],
  "docket_prefixes_maybe": ["4A", "5P", "2P"]
}
``` 
  ----------------------
  planned 10/10
Wrote LLM planner outputs: /content/drive/MyDrive/swiss_law/artifacts/val_segment_lattice_v3_llm_planner_outputs.jsonl
Wrote LLM raw log       : /content/drive/MyDrive/swiss_law/artifacts/val_segment_lattice_v3_llm_raw_log.jsonl
```

LLM funnel run + audit:

```
val_001 law= 1964 court= 2000 total= 3964
val_002 law=  870 court= 2000 total= 2870
val_003 law= 1306 court= 2000 total= 3306
val_004 law= 2000 court= 2000 total= 4000
val_005 law= 2000 court= 2000 total= 4000
val_006 law= 2000 court= 2000 total= 4000
val_007 law= 2000 court= 2000 total= 4000
val_008 law= 2000 court= 2000 total= 4000
val_009 law= 2000 court= 2000 total= 4000
val_010 law= 2000 court= 2000 total= 4000
Saved candidates: /content/drive/MyDrive/swiss_law/artifacts/val_segment_lattice_v3_candidate_sets.jsonl
Saved plans     : /content/drive/MyDrive/swiss_law/artifacts/val_segment_lattice_v3_planner_outputs.jsonl
Saved summary: /content/drive/MyDrive/swiss_law/artifacts/val_segment_lattice_v3_candidate_summary.csv
Saved dropped: /content/drive/MyDrive/swiss_law/artifacts/val_segment_lattice_v3_dropped_gold.csv

LLM outputs snapshotted:
  /content/drive/MyDrive/swiss_law/artifacts/val_segment_lattice_v3_llm_candidate_summary.csv
  /content/drive/MyDrive/swiss_law/artifacts/val_segment_lattice_v3_llm_dropped_gold.csv
```

Per-stage recall trace:

```
=== oracle per-stage trace → /content/drive/MyDrive/swiss_law/artifacts/val_segment_lattice_v3_oracle_stage_recall.csv ===
  court hard_filter   | recall mean=0.440 median=0.406 | cand median=   41084
  court source        | recall mean=1.000 median=1.000 | cand median= 1985178
  court topk_budget   | recall mean=0.034 median=0.000 | cand median=    2000
  law   hard_filter   | recall mean=1.000 median=1.000 | cand median=    3757
  law   source        | recall mean=1.000 median=1.000 | cand median=  175933
  law   topk_budget   | recall mean=0.613 median=0.513 | cand median=    2000

=== llm per-stage trace → /content/drive/MyDrive/swiss_law/artifacts/val_segment_lattice_v3_llm_stage_recall.csv ===
  court hard_filter   | recall mean=0.219 median=0.000 | cand median=  174177
  court source        | recall mean=1.000 median=1.000 | cand median= 1985178
  court topk_budget   | recall mean=0.000 median=0.000 | cand median=    2000
  law   hard_filter   | recall mean=0.771 median=0.800 | cand median=    2898
  law   source        | recall mean=1.000 median=1.000 | cand median=  175933
  law   topk_budget   | recall mean=0.623 median=0.562 | cand median=    2000
```

Final summary (val, 10 queries):

```
=== ORACLE (10 queries) ===
  recall        mean=0.396 median=0.295 min=0.191
  law_recall    mean=0.613 median=0.513 min=0.222
  court_recall  mean=0.034 median=0.000 min=0.000
  candidate_count        mean=3895 median=4000 max=4000
  law_candidate_count    mean=1895 median=2000 max=2000
  court_candidate_count  mean=2000 median=2000 max=2000

=== LLM PLANNER (10 queries) ===
  recall        mean=0.402 median=0.338 min=0.240
  law_recall    mean=0.623 median=0.562 min=0.429
  court_recall  mean=0.000 median=0.000 min=0.000
  candidate_count        mean=3814 median=4000 max=4000
  law_candidate_count    mean=1814 median=2000 max=2000
  court_candidate_count  mean=2000 median=2000 max=2000
```

## Summary

The v3 notebook rebuilds the segment-lattice index with per-option court co-signals (top co-cited law codes + distinctive German keywords) so a Qwen3-32B planner can pick BGE divisions and docket prefixes from corpus evidence rather than a hardcoded law-to-court map. The index covers 175,933 laws_de source rows and 1,985,178 court_considerations source rows; 68/74 docket_prefix options carry co_signals, and a compact 130-KB card bundle is exported for prompts. Hybrid planning unions Qwen include-mode picks with a co-signal heuristic and the parser's explicit decisions, validating every option against the registry. On val (10 queries) the oracle ceiling is law_recall mean 0.613 / court_recall mean 0.034 (overall 0.396); the LLM planner achieves law_recall mean 0.623 / court_recall mean 0.000 (overall 0.402), beating oracle on law but losing court entirely — the topk_budget of 2000 is the dominant bottleneck for court even when hard_filter recall is non-trivial (oracle court hard_filter mean 0.440, LLM 0.219). The trace confirms the funnel and registry are sound (law hard_filter recall 1.0 for oracle); the deficit is in budget sizing and court topk ranking, not in planner option coverage.
