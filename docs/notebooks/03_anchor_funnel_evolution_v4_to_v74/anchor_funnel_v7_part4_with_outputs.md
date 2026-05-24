# anchor_funnel_v7_part4_with_outputs.ipynb

**Path:** notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v7_part4_with_outputs.ipynb

## Configuration

### Runtime / hardware (Cell 2 output)
- Python: 3.12.13
- PyTorch: 2.10.0+cu128
- GPU 0: NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB
- Drive mount: `/content/drive` (Colab)

### Models
- Query expansion: `Qwen/Qwen3-32B` (bf16, ~65 GB VRAM, `qwen_max_new_tokens=1024`)
- Corpus + query embeddings: `Qwen/Qwen3-Embedding-8B` (fp16, SentenceTransformer)
- BM25: SQLite FTS5 (stdlib), in-memory, plus per-language indexes (de/fr/it/en)

### Libraries
- `torch`, `sentence_transformers`, `transformers`, `huggingface_hub`
- `pandas`, `numpy`, `sqlite3`, `re`, `json`, `gc`
- `collections.defaultdict`, `collections.Counter`
- `pathlib`, `google.colab.drive`

### Pass / stretch criteria
- Pass: R@1000 >= 0.60 for val_001 (>= 26/42 gold)
- Stretch: R@1000 >= 0.90 (>= 38/42 gold)

### CONFIG (printed by Cell 8)
```json
{
  "topk_final": 1000,
  "budget_law_direct": null,
  "budget_court_statute": 8000,
  "budget_concept": 3000,
  "budget_term": 2500,
  "budget_per_area": 1500,
  "budget_co_citation": 2500,
  "budget_bm25": 2000,
  "budget_vector": 2000,
  "budget_vector_enriched": 2000,
  "budget_backprop": 2000,
  "budget_sibling": 5000,
  "budget_graph_forward": 5000,
  "budget_graph_reverse": 3000,
  "enable_graph_2hop": false,
  "budget_graph_2hop": 1500,
  "rrf_k": 60,
  "guarantee_channels": [
    "law_direct_match", "per_area_bedrock", "statute_backprop",
    "concept_en", "graph_forward", "sibling_expansion", "term_orig"
  ],
  "guarantee_per_channel": 130,
  "channel_weights": {
    "statute_backprop": 2.5, "graph_forward": 2.0, "concept_en": 1.8,
    "court_statute": 1.5, "per_area_bedrock": 1.5,
    "vector_raw": 1.0, "vector_enriched": 1.0,
    "term_orig": 1.2, "law_direct_match": 1.2,
    "sibling_expansion": 1.0, "bm25": 0.8, "co_citation": 0.7,
    "graph_reverse": 0.5, "graph_2hop": 0.0
  },
  "code_family_top_k": 8,
  "per_area_top_n": 1000,
  "co_citation_top_k_per_target": 50,
  "co_citation_min_co_count": 50,
  "co_citation_max_neighbour_count": 50000,
  "concept_substring_top_k": 6,
  "bm25_max_query_terms": 60,
  "bm25_min_token_len": 3,
  "vector_emb_model": "Qwen/Qwen3-Embedding-8B",
  "vector_topk": 800,
  "qwen_query_model": "Qwen/Qwen3-32B",
  "qwen_max_new_tokens": 1024,
  "enhance_top_k_codes": 5,
  "enhance_repeat_count": 5,
  "enhance_min_idf": 1.0,
  "lowercase_concepts": true,
  "lowercase_terms": true
}
```

### Other constants
- Noise paragraph roles (negative gate): `{"notification", "header", "empty", "metadata"}`
- Code aliases used in canonicalization: `CPP->StPO, CP->StGB, CC->ZGB, CO->OR, LTF->BGG, LACI->AVIG, LAA->UVG, LP->SchKG, LDIP->IPRG, Cst/Cst.->BV, STPO->StPO, OBG->OR`
- Guarantee round-robin: 7 channels x cap=130 = 910 slots, leaves ~90 RRF tail
- 14 retrieval channels total (graph_2hop disabled by config)

## Data

`DATA_ROOT = /content/drive/MyDrive/swiss_law`

All paths flagged OK in Cell 6 output:

| key | path |
|---|---|
| val_csv | `/content/drive/MyDrive/swiss_law/data/val.csv` |
| law_llm | `/content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl` |
| court_v5 | `/content/drive/MyDrive/swiss_law/artifacts_v2/court_authority_cards_v5_unified.jsonl` |
| emb_dir | `/content/drive/MyDrive/swiss_law/artifacts/embeddings` |
| emb_manifest | `/content/drive/MyDrive/swiss_law/artifacts/embeddings/qwen3_8b_unified_manifest.parquet` |
| graph_db | `/content/drive/MyDrive/swiss_law/data_insights/citation_graph_extracted.sqlite` |
| out_dir | `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7` |

Embedding chunks: `qwen3_8b_unified_chunk*.npy` (27 fp16 files, ~21 GB total)

Candidate roots searched (in order):
- `/content/drive/MyDrive/swiss_law`
- `/content/drive/MyDrive/swiss_citation_extraction`
- `/content/swiss_citation_extraction`
- `E:/swiss_citation_extraction`
- `Path.cwd()`

Flags: `EMB_AVAILABLE = True`, `GRAPH_AVAILABLE = True`.

Query target: `val.csv` row with `query_id == "val_001"`, gold parsed by splitting `gold_citations` on `;`.

## Pipeline

### Phase 1 - Setup (Cells 1-10)
- 1.1 Environment + GPU check
- 1.2 Mount Google Drive (Colab)
- 1.3 Auto-detect `DATA_ROOT`, build `PATHS` dict, validate artifacts (OK/MISSING flags); set `EMB_AVAILABLE` / `GRAPH_AVAILABLE`
- 1.4 Print `CONFIG` knob panel
- 1.5 Load `val.csv`, select `val_001`, parse `;`-separated gold citations

### Phase 2 - Build all corpus indexes in one stream (Cells 11-14)
Stream both JSONL files (law + court) once, build all in-memory indexes:
- `cit_to_doc_ids`, `doc_meta`
- `idx_law_direct`, `idx_court_statute`, `idx_court_base`, `idx_case`
- `idx_concept_en`, `idx_term_orig`, `idx_term_lemma`, `idx_term_keys`
- `search_text` (BM25 input), `legal_area_per_doc`, `doc_statute_anchors`
- `co_citation_pairs` (Counter of unordered pairs)
- `tlf` (token -> legal-code Counter) for `enhance()`
- Canonicalizers: `CODE_ALIAS`, `ART_RE`, statute-canonical mapping

Cell 14: sanity-check all 42 val_001 gold citations map to corpus doc_ids.

### Phase 3 - Citation graph (Cell 15-16)
Load `citation_graph_extracted.sqlite` (4 alias passes, ~2.4 GB, 23.65 M edges).
Convert citation-string edges to doc-id-keyed dicts: `idx_graph_out[did]`, `idx_graph_in[did]`. Skip synthetic targets (no `cit_to_doc_ids`). Build judgment-importance index.

Four alias passes encoded in the graph:
1. Intra-judgment backrefs (`vgl./siehe E. N`, `E. N hiervor`, `cf. supra consid. N`, `siehe oben E. N`, etc.; 4 patterns x 3 languages)
2. Date-stripped aliases (`1B_210/2023 12.05.2023 E. 3` -> `1B_210/2023 E. 3`)
3. E.-range expansion (`E. 2.1-2.4` -> `E. 2.1`, `E. 2.2`, ...)
4. Case-level fan-out (cited E. of judgment -> all sibling Es of same judgment)

### Phase 4 - Per-area bedrock + co-citation neighbours (Cells 17-20)
- Per-area bedrock: from `legal_area_per_doc`, count canonical statutes per area; later filtered to LLM-named codes plus corpus co-citation code family
- Co-citation neighbours: for each statute, top-K co-cited canonicals via `co_citation_pairs`, filtered by `co_citation_min_co_count=50` and `co_citation_max_neighbour_count=50000`

### Phase 5 - BM25 / FTS5 (Cells 21-22)
- 5.A Legacy single-index FTS5 over `search_text` (back-compat)
- 5.B Per-language FTS5 indexes (de/fr/it/en) - query each in its own language with appropriate term targets, merge by max-score
- `enhance()` corpus-derived query boost: per query token, append top-K codes from `tlf[token]`, repeat `enhance_repeat_count=5` times

### Phase 6 - Vector channel setup (Cells 23-26)
- Load `qwen3_8b_unified_manifest.parquet`, build `row_for_did` mapping
- Load 27 fp16 chunks into one `E_GPU` tensor (shape ~2.65M x 4096)
- Provide `vector_search(q_emb, k)` as brute-force GPU matmul
- Define `encode_query(text)` using SentenceTransformer with `prompt_name="query"`, normalized embeddings

### Phase 7 - Query expansion (Cells 27-30)
- Load `Qwen/Qwen3-32B` (bf16), prompt with fixed JSON schema, parse output
- Schema keys: `statute_targets`, `case_targets`, `concept_targets_en`, `term_targets_de`, `term_targets_fr`, `legal_area_keywords`
- Few-shot prompt; "no invented citations" rule; `enable_thinking=False`
- Free Qwen3-32B before loading Qwen3-Embedding-8B
- Encode query twice: raw + enriched (raw + appended concept/term keywords)

### Phase 8 - Retrieval channels (Cells 31-34)
14 channels defined and executed:
1. `law_direct_match` (uncapped)
2. `court_statute` (specificity-weighted + paragraph_role boost, budget 8000)
3. `co_citation`
4. `per_area_bedrock` (filtered by LLM codes + corpus code-family expansion)
5. `statute_backprop` (rare-canon specificity-weighted)
6. `sibling_expansion` (judgment-importance scored, v6 budget bug fixed)
7. `graph_forward` (v7 NEW; case-level fan-out + importance weighting)
8. `graph_reverse` (v7 NEW; landmark filter)
9. `graph_2hop` (v7 NEW, disabled in config)
10. `concept_en` (token-overlap matcher, stopwords filtered)
11. `term_orig` (German lemmatizer drop `-en/-e/-er/-s` with 4-char floor, substring overlap)
12. `bm25` (multi-language FTS5 + `enhance()`)
13. `vector_raw`
14. `vector_enriched`

Seed pool built from topical hits; sibling + graph + statute_backprop run with the seed. Per-channel diagnostic table printed (size, gold_in_ch, recall).

### Phase 9 - RRF fusion + negative gate + final pool (Cells 35-36)
- `rrf_fuse(channels, k, weights)`: score = sum over channels of `weights[ch] / (k + rank + 1)`
- Negative gate: drop docs with `paragraph_role` in noise set or `is_notification_paragraph=True`, with substantive-role override
- Guarantee channels prepended (round-robin, 130 per channel)
- Return top `topk_final=1000`

### Phase 10 - Per-gold diagnosis (Cells 37-42)
- 10.1 Helper functions: `channel_rank`, `channel_score`, token overlap for BM25
- 10.2 Per-gold loop: for every gold not in top-1000, print structured trace (source row, search_text, enrichment fields, per-channel hit/miss + reason, RRF score/rank, root cause, fix suggestion)
- 10.3 Failure-mode aggregate: bucket by root cause, top-10 missed list, channel-level hit-rate on missed gold

### Phase 11 - Save + cleanup (Cells 43-46)
- Write `final_topk.json`, `gold_in_top.json`, `gold_missed_diagnosis.json`, `targets.json`, `config.json`, `summary.json` to `out_dir`
- Free `E_GPU`, `EMB_MODEL`, `gc.collect()`, `torch.cuda.empty_cache()`

## Results

### Cell 6 - paths
All 6 required artifacts `OK`; `EMB_AVAILABLE = True`; `GRAPH_AVAILABLE = True`.

### Cell 10 - val_001 load
```
val.csv has 10 queries
val_001: 42 gold citations
Query (first 240 chars): May a court lawfully order a three-month extension of pre-trial detention under Art. 221 Abs. 1 lit. b StPO (risk of collusion) consistent with the principle of proportionality when the accused - detained after an alleged late-night assault a...

First 6 gold citations:
  - Art. 221 Abs. 1 StPO
  - Art. 140 Abs. 1 StGB
  - Art. 396 Abs. 1 StPO
  - Art. 222 StPO
  - Art. 393 Abs. 1 StPO
  - Art. 382 Abs. 1 StPO
```

### Cell 12 - corpus index build
```
Law: 173,033 rows indexed in 12.3s
Token->code association: 91,173 tokens, avg codes/token = 11.7
  court progress: 500,000 rows (43.6s)
  court progress: 1,000,000 rows (88.0s)
  court progress: 1,500,000 rows (140.0s)
  court progress: 2,000,000 rows (205.7s)
Court: 2,476,315 rows indexed in 266.9s
Total docs:        2,649,348
Unique citations:  2,158,211
Index sizes:       law_direct=49,288, court_statute=81,988, case=157,227, court_base=178,593, concept=292,946, term=363,331, term_lemma=521,146, term_keys=363,331
Co-citation pairs: 1,142,790
```

### Cell 14 - gold mapping
```
Mapped gold: 42/42
Total gold doc_ids: 42
```

### Cell 16 - citation graph
```
Built citation->doc_id map (2,158,211 entries)
Graph: 20,494,436 edges loaded, 3,155,263 skipped (cit not in corpus)
Graph: out-degree avg = 27.2, in-degree avg = 14.0
Graph: load time 168.6s
Judgment importance: 178,593 judgments, median=7, p75=44, max=204686
Judgment importance: built in 1.1s
```

### Cell 18 - per-area bedrock + code-pair statistics
```
Building per-area bedrock index...
  built in 2.1s; areas: 26
  area='constitutional and public law': top 8 = [('66 BGG', 13212), ('29 BV', 12189), ('89 BGG', 11479), ('82 BGG', 11464), ('42 BGG', 10836), ('9 BV', 9650), ('106 BGG', 9625), ('68 BGG', 8649)]
  area='administrative, tax, migration, and regulatory law': top 8 = [('42 BGG', 20301), ('106 BGG', 19460), ('66 BGG', 17471), ('105 BGG', 16033), ('95 BGG', 15722), ('83 BGG', 15231), ('68 BGG', 14674), ('89 BGG', 12187)]
  area='civil law': top 8 = [('63 OJ', 3189), ('8 ZGB', 2740), ('55 OJ', 2518), ('64 OJ', 2235), ('55 OG', 2085), ('63 OG', 1906), ('159 OG', 1876), ('9 BV', 1875)]
  area='criminal law and criminal procedure': top 8 = [('66 BGG', 28149), ('42 BGG', 20746), ('106 BGG', 19537), ('64 BGG', 13843), ('108 BGG', 12101), ('97 BGG', 12075), ('105 BGG', 11759), ('81 BGG', 11102)]
Building corpus-derived code-pair statistics...
  code-pair statistics: 54,790 directed pairs (0.7s)
  StPO's top related codes: [('BGG', 50026), ('BV', 34792), ('StGB', 18981), ('EMRK', 6826), ('Satz', 2812), ('CEDH', 2549), ('OR', 1961), ('ZGB', 1123)]
```

### Cell 20 - co-citation neighbours of `221 StPO`
```
Co-citation neighbours indexed for 2,282 canonicals (after frequency filter: max global count = 50000).
Sample - neighbours of '221 StPO' AFTER frequency filter:
  36 BV: co=915, total_in_corpus=7216
  31 BV: co=783, total_in_corpus=4339
  10 BV: co=731, total_in_corpus=5241
  212 StPO: co=714, total_in_corpus=1337
  221 BV: co=646, total_in_corpus=670
  237 StPO: co=487, total_in_corpus=1467
  5 BV: co=393, total_in_corpus=9685
  5 StPO: co=272, total_in_corpus=1955
```

### Cell 22 - FTS5 build
```
Building in-memory FTS5 (legacy, single index) over 2,649,348 docs...
  inserted 500,000 (11.6s)
  inserted 1,000,000 (27.3s)
  inserted 1,500,000 (44.2s)
  inserted 2,000,000 (61.5s)
  inserted 2,500,000 (77.6s)
FTS5 (legacy) built: 2,649,348 rows in 83.9s
Building per-language FTS5 indices (de/fr/it/en)...
Per-language FTS5 built in 70.4s
  fts[de]: 1,593,249 docs
  fts[fr]: 793,023 docs
  fts[it]: 125,583 docs
  fts[en]: 137,493 docs
```

### Cell 24 - corpus embeddings to GPU
```
Loading manifest...
  manifest rows: 2,652,248, cols: ['doc_id', 'family', 'citation', 'row_index']
  manifest->my_did mapping: 2,649,348/2,652,248 (99.9%) in 41.3s
  loading chunks to GPU (~21 GB)...
  E_GPU shape=(2652248, 4096) dtype=torch.float16, VRAM used=20.2 GB, free=74.7 GB, load 596.2s
```

### Cell 28 - Qwen3-32B query expansion
```
[qexp] loading Qwen/Qwen3-32B (~65 GB bf16)...
[qexp] model loaded in 205.8s
RAW (first 400 chars): {
  "statute_targets": ["Art. 221 StPO", "Art. 222 StPO", "Art. 227 StPO", "Art. 100 BGG", "Art. 101 BGG"],
  "case_targets": ["BGE 147 II 123", "BGE 145 II 345", "1B_123/2024"],
  "concept_targets_en": ["preventive detention", "collusion risk", "proportionality", "investigative necessity", "witness tampering"],
  "term_targets_de": ["Untersuchungshaft", "Kollusionsgefahr", "Verhältnismässigkeit",
[qexp] freed Qwen3-32B; CUDA mem=20.2 GB

Targets parsed:
  statute_targets: (5) ['Art. 221 StPO', 'Art. 222 StPO', 'Art. 227 StPO', 'Art. 100 BGG', 'Art. 101 BGG']
  case_targets: (3) ['BGE 147 II 123', 'BGE 145 II 345', '1B_123/2024']
  concept_targets_en: (5) ['preventive detention', 'collusion risk', 'proportionality', 'investigative necessity', 'witness tampering']
  term_targets_de: (5) ['Untersuchungshaft', 'Kollusionsgefahr', 'Verhältnismässigkeit', 'Ermittlungsbedürftigkeit', 'Zeugenbeeinflussung']
  term_targets_fr: (5) ['détention provisoire', 'danger de collusion', 'proportionnalité', "nécessité d'enquête", 'influence sur les témoins']
  legal_area_keywords: (5) ['criminal procedure', 'detention law', 'proportionality principle', 'evidence preservation', 'witness protection']
```

Also surfaced a HF auth warning (non-fatal): `Error while fetching HF_TOKEN secret value from your vault ... You are not authenticated with the Hugging Face Hub in this notebook.`

### Cell 30 - Qwen3-Embedding-8B load + query encode
```
Loading Qwen3-Embedding-8B...
  done
Encoding raw query...
Encoding enriched query...
  enriched query length: 1367 chars, keywords appended: 15
```

### Cell 34 - per-channel recall table (val_001, 42 gold)
```
Statute canons: LLM=5, +co-citation=128
LLM target codes: ['BGG', 'StPO']
Code-family expanded (8/code from corpus co-citation): ['BV', 'CEDH', 'EMRK', 'OR', 'Satz', 'SchKG', 'StGB', 'ZGB', 'ZPO']
Concepts: LLM=10 -> expanded=246
seed pool: 15,843 court doc_ids
sibling_expansion: 5,000 expansions
graph_forward: 5,000 expansions
graph_reverse: 3,000 expansions
backprop seed: 27,611 -> 2,000 law expansions

channel                   size   gold_in_ch  recall
------------------------------------------------------------
law_direct_match           253            7   16.7%
court_statute             8000            2    4.8%
co_citation               2500            0    0.0%
per_area_bedrock          1500           17   40.5%
statute_backprop          2000           18   42.9%
sibling_expansion         5000            4    9.5%
graph_forward             5000            9   21.4%
graph_reverse             3000            1    2.4%
graph_2hop                   0            0    0.0%
concept_en                3000           15   35.7%
term_orig                 2500            8   19.0%
bm25                      2000            5   11.9%
vector_raw                2000           13   31.0%
vector_enriched           2000           13   31.0%

Union: 31,013 unique doc_ids
Gold in union: 41/42  (UPPER BOUND on R@K)
```

### Cell 36 - RRF fusion + final pool
```
Pre-gate fused: 31,013
Post-gate:      30,453
Guarantee pool: 908  (channels: ['law_direct_match', 'per_area_bedrock', 'statute_backprop', 'concept_en', 'graph_forward', 'sibling_expansion', 'term_orig'])
Final top-1000: 1,000

R@K curve:
     K  gold/42  recall
------------------------------
    50      4/42    9.5%
   100      6/42   14.3%
   200      7/42   16.7%
   300      8/42   19.0%
   500     12/42   28.6%
   750     15/42   35.7%
  1000     16/42   38.1%
  1500     21/42   50.0%
  2000     26/42   61.9%
  3000     30/42   71.4%
  5000     33/42   78.6%

Final R@1000 = 0.381  (16/42)
```

### Cell 38 - BM25 helper trace
```
Query token count (post-enhance): 122
```

### Cell 40 - per-gold miss diagnosis (excerpt)
```
R@1000 = 16/42 = 38.1%
Missed gold: 26 of 42

GOLD: 1B_210/2023 E. 4.1
  doc_id=court:1052533  family=court  court_base=1B_210/2023  paragraph_role=legal_standard
  Per-channel:
    MISS law_direct_match       (canonical=None | in_canon_set=False)
    HIT  court_statute          rank=889  (size=8000)
    MISS co_citation            (reachable via co-cited neighbour=False)
    MISS per_area_bedrock       (law canonical=None | in selected-area top-1000=False)
    MISS statute_backprop       (family=court, statute_backprop only emits law rows)
    MISS sibling_expansion      (court_base=1B_210/2023 | sibs=15 | caught_sibs=4)
    MISS graph_forward          (caught seeds with edges -> 0: [])
    MISS graph_reverse          (caught seeds reachable from this -> 5)
    MISS graph_2hop             (disabled by config)
    HIT  concept_en             rank=264  (size=3000)
    HIT  term_orig              rank=1630  (size=2500)
    HIT  bm25                   rank=1506  (size=2000)
    HIT  vector_raw             rank=1133  (size=2000)
    HIT  vector_enriched        rank=810  (size=2000)
  RRF: score=0.01032  rank=1075  (top-1000 cutoff is rank 999)
  ROOT CAUSE: RRF_RANK_TOO_LOW
  FIX SUGGESTION: in 6 channel(s) but RRF rank 1075; consider promoting a channel to guarantee or lifting budget

GOLD: 1B_536/2018 E. 5.1
  doc_id=court:1124479  family=court  court_base=1B_536/2018  paragraph_role=facts
  ...
```

### Cell 42 - failure-mode summary
```
FAILURE-MODE SUMMARY
Missed gold: 26
  by root cause:
    RRF_RANK_TOO_LOW        22
    GATED_OUT               3
    NO_CHANNEL_HIT          1

Top 10 missed gold with fix suggestions (excerpt):
  - 1B_210/2023 E. 4.1     cause=RRF_RANK_TOO_LOW   channels_hit=6   RRF rank 1075
  - 1B_536/2018 E. 5.1     cause=RRF_RANK_TOO_LOW   channels_hit=3   RRF rank 2551
  - 7B_496/2025 E. 3.2     cause=RRF_RANK_TOO_LOW   channels_hit=1   RRF rank 20122
  - BGE 132 I 21 E. 3.2.1  cause=RRF_RANK_TOO_LOW   channels_hit=2   RRF rank 8453
  - BGE 133 I 168 E. 4.1   cause=RRF_RANK_TOO_LOW   channels_hit=1   RRF rank 4700
  - 1B_90/2021 E. 2.4      cause=RRF_RANK_TOO_LOW   channels_hit=1   RRF rank 7861
  - 7B_231/2025 E. 4.1     cause=RRF_RANK_TOO_LOW   channels_hit=3   RRF rank 3959
  - 1B_28/2022 E. 4.1      cause=GATED_OUT          channels_hit=3   paragraph_role=facts
  - BGE 132 I 21 E. 3.2.2  cause=RRF_RANK_TOO_LOW   channels_hit=2   RRF rank 1052
  - BGE 143 IV 168 E. 5.1  cause=RRF_RANK_TOO_LOW   channels_hit=2   RRF rank 2894

CHANNEL-LEVEL HIT RATE ON MISSED GOLD
  law_direct_match      0
  court_statute         1
  co_citation           0
  per_area_bedrock      10
  statute_backprop      11
  sibling_expansion     1
  graph_forward         1
  graph_reverse         1
  graph_2hop            0
  concept_en            9
  term_orig             4
  bm25                  1
  vector_raw            9
  vector_enriched       9
```

### Cell 44 - artifact save
```
Saved 6 artifacts to /content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7
```

### Cell 46 - GPU cleanup
```
VRAM after cleanup: 0.01 GB
cleaned up.
```

## Summary

The v7.5 production-diagnostic build of the anchor-funnel pipeline ran end-to-end on val_001 (42 gold, pre-trial detention query), executing all 14 retrieval channels (graph_2hop disabled) plus RRF fusion, negative gating, and a 7-channel guarantee round-robin. The retrieval union covered 41/42 gold (97.6% upper bound) but final R@1000 landed at 0.381 (16/42) - far below the pass criterion (>= 0.60) and the v6 retrospective baseline of 0.357 was barely exceeded. R@K grows steeply past 1000 (R@2000 = 0.619, R@5000 = 0.786), confirming the bottleneck is fusion ranking, not candidate generation. Phase 10 attributed 22/26 misses to `RRF_RANK_TOO_LOW`, 3 to `GATED_OUT`, and 1 to `NO_CHANNEL_HIT`; the strongest channels were `statute_backprop` (42.9%), `per_area_bedrock` (40.5%), and `concept_en` (35.7%), while `co_citation` and `graph_2hop` produced 0 gold. The lesson: the channels surface nearly all gold into the candidate pool, but RRF weights + budgets push too many multi-channel hits below rank 1000; next iteration should promote `vector_raw`/`vector_enriched` to guarantee or lift their RRF weight, since they each caught 9 of the missed gold but ranked them too low.
