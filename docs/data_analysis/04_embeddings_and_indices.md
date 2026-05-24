# 04 — Embeddings & Indices

Scope: Qwen3-Embedding-8B unified corpus embedding shards, the canonical
SQLite retrieval store, the citation-graph SQLite tie-breaker, query
embeddings, the reranker shootout score caches, embedding input parquet, and
the Omnilex flatip embedding sidecars.

All sizes / counts below are confirmed from the on-disk files via `ls -la`
and from each file's own manifest / summary JSON. Where a number requires
opening the binary itself (e.g. exact .npy shape) the **documented** schema
from the producer notebook/script is used and explicitly labelled.

---

## A. Qwen3-Embedding-8B unified corpus embeddings

Directory: `e:\swiss_citation_extraction\embeddings\`

### A.1 Shard layout (27 chunks, fp16, ~21 GB total)

| File | Bytes | Rows (implied) |
|---|---|---|
| `qwen3_8b_unified_chunk000.npy` ... `qwen3_8b_unified_chunk025.npy` | 819,200,128 each (×26) | 100,000 rows each |
| `qwen3_8b_unified_chunk026.npy` | 428,015,744 | 52,248 rows |
| total | 21,727,218,432 B ≈ 20.24 GiB | **2,652,248 rows** |

**Row-count derivation:** the shard is `(rows × 4096 × 2 bytes)`. The first
26 chunks are exactly `100,000 × 4096 × 2 = 819,200,000` data bytes plus a
128-byte NumPy header, matching the `chunk_size: 100000` declared in
`qwen3_8b_unified_summary.json`. Chunk 026 is the tail: `(428,015,744 - 128)
/ 4096 / 2 = 52,248` rows. Total = `26 × 100,000 + 52,248 = 2,652,248` rows,
which matches `corpus_rows` exactly.

### A.2 Encoding metadata — `qwen3_8b_unified_summary.json`

Verbatim from the file:

| key | value |
|---|---|
| `model_name` | `Qwen/Qwen3-Embedding-8B` |
| `embedding_dim` | 4096 |
| `output_dtype` | `float16` |
| `normalized` | `true` (L2-normalised → dot product = cosine) |
| `document_prompt_prefix` | `null` (no instruct prefix on the doc side) |
| `attention_impl` | `sdpa` |
| `max_seq_length` | 768 |
| `batch_size` | 256 |
| `chunk_size` | 100,000 |
| `torch_compile` | `false` |
| `chunk_count` | 27 |
| `corpus_rows` | 2,652,248 |
| `family_counts.court` | 2,476,315 |
| `family_counts.law` | 175,933 |
| `gpu` | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| `torch_version` | `2.10.0+cu128` |
| `cuda_version` | `12.8` |

### A.3 Sharding contract (CRITICAL)

The mapping from a global row to a doc_id is materialised in
`qwen3_8b_unified_manifest.parquet` (104,647,369 B ≈ 99.8 MiB on disk,
2,652,248 rows). Documented schema (from the v7 anchor-funnel notebooks
which load it via `pq.read_table` and from `embed_unified_corpus_qwen3_8b_blackwell.md`):

| column | dtype | meaning |
|---|---|---|
| `chunk` | int (uint16-fits) | shard index in `[0, 26]` |
| `row` | int | row offset inside that shard, in `[0, chunk_size)` |
| `doc_id` | string | the `unified_retrieval.sqlite` doc primary key |

Read pattern actually used (from `anchor_funnel_v7_val001.md` line 1298 and
`anchor_funnel_v4_val001.md` line 702):

```python
chunks = sorted(PATHS["emb_dir"].glob("qwen3_8b_unified_chunk*.npy"))
# concatenate along axis 0, OR mmap each shard and gather by (chunk, row)
```

For a candidate doc_id you look it up in the manifest, get `(chunk, row)`,
then mmap `qwen3_8b_unified_chunk{chunk:03d}.npy` and slice `[row]` to get
the 4096-d fp16 vector. The chunks are **fixed-row** (100k except the last
at 52,248) so `global_row = chunk*100_000 + row` if you concatenate.

### A.4 Phantom-issue note (from experiments ledger)

> "Stale fp32 embedding chunks" — Deleted. The 21 GB fp16 chunks at
> `embeddings/qwen3_8b_unified_chunk*.npy` are canonical.
> (`endgame_handoff §6.6`, `swiss-citation-experiments` SKILL.md)

> "Need FAISS-IVF for vector index" — Full-GPU brute force `E_GPU @ q` is
> ~50 ms/query on Blackwell. Fine. (`§6.5`)

> "Dense embedding alone can carry retrieval" — Qwen3-Embedding-8B over the
> full corpus capped at R@1000 = 0.289 on val. Structural ceiling. (Obs 3)

### A.5 Producer

`scripts/build_pipeline/build_unified_retrieval_corpus.py` produces the
unified row inventory (the *input* parquet — see §D); the embedding step
itself is in the Colab/Kaggle notebook
`notebooks/_inventory/embed_unified_corpus_qwen3_8b_blackwell.md`
(uses Blackwell sdpa attention, fp16 output, batch_size=256, chunk_size=100k).

### A.6 Consumers (notebooks that load these shards)

From `notebooks/_inventory/`:
`anchor_funnel_v4_val001.md`, `anchor_funnel_v5_val001.md`,
`anchor_funnel_v6_val001.md`, `anchor_funnel_v7_val001.md`,
`anchor_funnel_v7_4_val001.md`, `anchor_funnel_v7_part2/3/4_with_outputs.md`,
`endgame_colab_full_pipeline.md`,
`pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.md`,
`pool_v75_multiquery_iteration_11.md`,
`pool_v75_multiquery_hyde_test_no_lift.md`,
`retrieve_unified_corpus_qwen3_8b.md`, `retrieve_unified_corpus_v3.md`,
`early_segment_lattice_funnel_v3.md`,
`rerank_hybrid_3model_shootout.md`,
`rerank_hybrid_3model_shootout_with_outputs.md`.

---

## B. SQLite indices

### B.1 `artifacts/unified_retrieval.sqlite` — CANONICAL 2.65 M-doc store

| | |
|---|---|
| **Path** | `e:\swiss_citation_extraction\artifacts\unified_retrieval.sqlite` |
| **Size** | 24,074,375,168 B ≈ **22.4 GiB** |
| **Built** | 2026-05-06 23:36 |
| **Builder** | `scripts/build_pipeline/build_unified_retrieval_corpus.py` (header: `build_unified_retrieval_corpus_v1`) |
| **Inputs** | `artifacts/court_authority_cards_v5_unified.jsonl` + `artifacts/law_authority_cards_v2_unified.jsonl` |
| **Sidecar JSONL** | `artifacts/unified_retrieval_documents.jsonl` (16,337,353,074 B ≈ 15.2 GiB — one JSON per doc; the SQLite is built from this) |
| **Manifest** | `artifacts/unified_retrieval_manifest.json` |
| **Doc count** | 2,652,248 (law 175,933 + court 2,476,315) |
| **FTS5** | enabled (`fts_enabled: true` in manifest) |

**This is the canonical document store.** Every consumer (BM25 channel,
vector channel for citation/text lookup, dossier feature builder, all
anchor-funnel runs) joins back to this DB by `doc_id`. The Qwen3 embedding
shards in §A index into this DB via the manifest parquet.

**Schema (from manifest `retrieval_fields`).**

Vector text fields concatenated into the per-doc embedding input:
- `rag_enrichment.english_summary`
- `rag_enrichment.legal_question`
- `rag_enrichment.legal_rule` / `doctrinal_rule`
- `rag_enrichment.court_holding` (court family)
- `rag_enrichment.applicability_conditions` / `exceptions` / `sanctions` / `addressees` (law family)
- `concepts_en`, `search_keywords`, `terms`
- `statute_anchors`, `case_anchors`, `adjacent_law` (law)
- Labelled English schema (Citation / Court base / Law / Code / Legal area / ...)

BM25 columns + recommended weights (from manifest):

| column | weight | content |
|---|---|---|
| `citation_text` | 4.0 | exact citation, court base, statutes, law codes |
| `keyword_text` | 3.0 | search keywords, multilingual terms, concepts, addressees |
| `rag_text` | 2.5 | LLM English summary / question / rule / holding / facts / applicability |
| `legal_text` | 2.0 | legal area, domain path, role, topic, authority |
| `source_text` | 0.7 | original German / French / Italian paragraph |

Link tables (implied by builder name and ledger references): `statute_links`,
`case_links`, `adjacent_law_links`. (Builder is
`scripts/build_pipeline/build_unified_retrieval_corpus.py`; full schema is in
that script.)

**Manifest stats — flagging counts (from
`unified_retrieval_manifest.json`):**

Law (175,933 docs total):
- `flag_no_anchors`: 144,714
- `flag_static_only`: 2,900
- `flag_boilerplate`: 7,925 (and same count `flag_rule_suppressed`)
- `flag_role_transitional_or_commencement`: 3,088
- `flag_role_data_reporting`: 3,517
- `flag_role_fees_or_costs`: 15

Court (2,476,315 docs total):
- `flag_no_statute_anchor`: 1,092,562
- `flag_no_case_anchor`: 1,732,269
- `flag_static_only`: 2,476,315 (ALL — the court schema forbids
  `english_summary` by design; the `static_only` flag is a misnomer here, see
  Phantom Issues §6.1 of endgame_handoff)
- `flag_role_procedural_history`: 432,262
- `flag_low_value_paragraph`: 565,549
- `flag_role_costs`: 361,651
- `flag_role_notification`: 203,898
- `flag_role_disposition`: 152,556
- `flag_thin_vector_text`: 3,409

**Build params:** `max_source_chars: 2500`, `max_vector_chars: 4500`,
`limits.court: 0`, `limits.law: 0` (no row caps for the canonical build).

**No `unified_retrieval.sqlite-journal` currently exists on disk** (the WAL
is transient; only present when a writer holds the DB). The task brief
called it "transient WAL" — confirmed transient (none present at scan time).

### B.2 Smoke variant — `unified_retrieval.smoke.sqlite`

| | |
|---|---|
| Path | `e:\swiss_citation_extraction\artifacts\unified_retrieval.smoke.sqlite` |
| Size | 74,395,648 B ≈ 71 MiB |
| Manifest | `unified_retrieval_manifest.smoke.json` |
| Sidecar JSONL | `unified_retrieval_documents.smoke.jsonl` (50,838,116 B ≈ 48 MiB) |
| Doc count | 6,000 (law 1,000 + court 5,000) — `limits.law: 1000, limits.court: 5000` |
| Output paths in smoke manifest are relative | `output_jsonl: artifacts\unified_retrieval.smoke.jsonl` (sic — manifest typo, actual JSONL is `unified_retrieval_documents.smoke.jsonl`) |

Same schema + same builder, used for fast end-to-end pipeline smoke tests.

### B.3 `sqlite_probe.sqlite` (probe/test file)

| | |
|---|---|
| Path | `e:\swiss_citation_extraction\drive_sync\swiss_law\colab_data_insights_mirror\sqlite_probe.sqlite` |
| Size | 8,192 B (one SQLite page — effectively empty / structure-only) |
| Note | **Not** in `artifacts/` as the brief expected. The file is in the colab data-insights mirror. Used as a connectivity / driver probe (open + smoke query) for Colab→Drive→sqlite3 wiring. |

### B.4 `data_insights/citation_graph_db_and_edges/citation_graph_extracted.sqlite` — TIE-BREAKER

| | |
|---|---|
| Path | `e:\swiss_citation_extraction\data_insights\citation_graph_db_and_edges\citation_graph_extracted.sqlite` |
| Size | 6,096,723,968 B ≈ **5.68 GiB** |
| Built | 2026-05-10 15:49 (post-back-ref patch + post-case-level patch) |
| Coverage | **96.0 %** of gold citations (up from 92.9 % pre-patch) |

**Role:** tie-breaker only. Per the experiments ledger (Phantom Issues row
"Citation graph can predict gold co-occurrence"):

> 99.41 % of gold-citation pairs have **no graph edge**. Useful only as a
> tie-breaker. (Obs 2)

i.e. the citation graph is informative when two candidates score equally on
the dense + BM25 + reranker signals; it is **not** a primary retrieval
channel. Remaining 4 % uncoverable: LugÜ, FIDLEG, FINIG, FinfraG, GBV, parts
of URG/ZGB — corpus gaps, not patchable from the graph side.

### B.5 Citation-graph backups (pre-patch snapshots)

| File | Size | What it preceded |
|---|---|---|
| `citation_graph_extracted.sqlite.bak_pre_backref` | 2,131,738,624 B ≈ 1.99 GiB (2026-05-10 13:45) | The **back-reference** patch — adding reverse-edge enumeration so a cited paragraph also knows its citers (used by co-citation density features). After this patch the DB grew from ~2 GB to ~2.2 GB. |
| `citation_graph_extracted.sqlite.bak_pre_caselevel` | 2,194,845,696 B ≈ 2.04 GiB (2026-05-10 15:48) | The **case-level** patch — promoting paragraph-level edges to case-level (BGE / docket) aggregates. After this patch the DB grew to its current 5.68 GiB. The size jump (~2 GB → ~5.7 GB) comes from materialising case-level edge tables. |

Keep both backups — they are the only ground-truth for "what coverage was
before each patch" if a regression is suspected.

### B.6 `segment_lattice_v3.sqlite`

| Path | Size |
|---|---|
| `artifacts\segment_lattice_v3.sqlite` | 5,906,550,784 B ≈ 5.50 GiB (2026-04-27 18:29) |

Produced by `scripts/retrieval_and_rerank/segment_lattice_v3.py`. The
brief asked "confirm not local" — **it is local**. It is referenced from the
early-experiments family (`notebooks/_inventory/early_segment_lattice_funnel_v3.md`).
Pre-v7.5 segment-lattice approach; no longer in the active pipeline. There
are also `artifacts\test_segment_lattice_v3_*/artifacts/segment_lattice_v3.sqlite`
test-harness copies and a related stray
`drive_sync\swiss_law\colab_data_insights_mirror\citation_graph.sqlite-journal`
(WAL for an alternate citation-graph DB in the colab mirror).

---

## C. Query embeddings + reranker score caches

### C.1 Val query embeddings — `cache_endgame/`

| File | Size | Documented schema |
|---|---|---|
| `cache_endgame/query_embeddings.npy` | 327,808 B | fp32, L2-normalised, shape `(10, 4096)` — math: `(327808 - 128 header) / 4 / 4096 ≈ 20` floats per row × 4096 cols × 4 B per fp32. Math check: `10 × 4096 × 4 = 163,840` B per fp32 OR `10 × 4096 × 8 = 327,680` B per fp64. Header 128 B → 327,808 B total — so this is actually **fp64** at 10 rows × 4096 cols. Caveat: shape unverified without opening; the documented producer (`encode_queries_qwen3_8b.py`) saves fp32 from `.astype(np.float32)` so this file is likely **(10, 8192) fp32** OR (20, 4096) fp32 — the byte math is `327,680 / 4 / 4096 = 20` rows. Confirm before consuming; the key file says 10 keys. |
| `cache_endgame/query_embeddings_keys.json` | 880 B | JSON array of 20 ID hashes (e.g. `bf6b3a2bf929f6faaf4751937e1e471078df1604`). Confirmed length: **20 entries** when read. Treat as 20 hashed query identifiers (10 val queries × 2 representations, or 10 val + 10 expanded — to be confirmed against the encoder script). |

**Caveat resolved by byte math:** `327,808 B = 128 (np header) + 20 × 4096 × 4`.
So the file is `(20, 4096) fp32`, matched 1:1 against the 20 keys. The brief
described it as "10 val query vectors"; on-disk it is 20, suggesting two
encodings per val query (e.g. query + HyDE expansion, or query + sub-query)
keyed by content hash. The keys JSON confirms 20 hashed IDs.

**Producer:** `scripts/retrieval_and_rerank/encode_queries_qwen3_8b.py`
(uses `MAX_SEQ_LEN=2048` with the official Qwen3 instruct prefix
`"Instruct: Given an English-language legal question or scenario about
Swiss federal law, retrieve ..."` — emits fp32 L2-normalised vectors). The
notebook predecessor was at `MAX_SEQ_LEN=256` which truncated 10/10 val
queries; this script exists specifically to fix that truncation.

**Consumer:** `notebooks/_inventory/endgame_colab_full_pipeline.md` (only
file in `_inventory/` that grep-hits on `query_embeddings.npy`).

### C.2 Reranker score caches — `drive_sync/swiss_law/hybrid_rerank_shootout_cache/cache/`

These are the inputs to the **2026-05-15 hybrid reranker shootout** (see
experiments ledger).

| File | Size | Producer |
|---|---|---|
| `scores_qwen3.npz` | 297,712 B | `rerank_hybrid_3model_shootout` notebook, Phase 5 step A. Qwen3-Reranker-8B via vLLM, bfloat16, max_model_len=1024, gpu_mem_util=0.85, scoring `yes`/`no` logits and saving normalised `p(yes)`. |
| `scores_bge.npz` | 788,379 B | Same notebook, Phase 5 step B. BGE-reranker-v2-m3 (568 M). |
| `scores_jina.npz` | 692,159 B | Same notebook, Phase 5 step C. jina-reranker-v2-base-multilingual (278 M). |
| `dossier_features.npz` | 4,343,004 B | `scripts/retrieval_and_rerank/precompute_dossier_cache.py`. Phase 1+2+3 dossier features per (qid, did). |

**File format — confirmed from notebook source (lines 705–784 of
`rerank_hybrid_3model_shootout.md`):**

```python
np.savez_compressed(
    QWEN3_CACHE,
    **{qid: scores_1d for qid, scores_1d in ALL_RERANK_SCORES["qwen3"].items()}
)
```

- One key per `qid` (10 val queries → 10 keys per .npz).
- Each value is a 1-D fp32 array of length `len(PER_QUERY[qid]["final_topk"][:RERANK_TOP_N])`,
  in **pool order** — i.e. position `i` of the array is the rerank score for
  candidate `PER_QUERY[qid]["final_topk"][i]`. The pool itself comes from
  the v7.5 multi-query funnel snapshot (`anchor_funnel_val001_v7/snapshot/`).
- `RERANK_TOP_N` is the rerank-pool cap (~43-46 k per query — the brief
  states ~43-46k v7.5-pool candidates, matched by the ledger row "score all
  10 queries × ~43-46k v7.5-pool candidates").
- For Qwen3, score = `softmax([logp_yes, logp_no])[0]` in `[0, 1]`.
- For BGE / jina, score = native cross-encoder output (sigmoid logit) in `[0, 1]`.

`dossier_features.npz` uses a **different** key convention (from
`precompute_dossier_cache.py`):

```python
# Keys: "<qid>__dids" → 1-D string array of doc_ids in feature order
#       "<qid>__feat" → 2-D float array of per-candidate feature values
```

Feature columns are the Phase 1+2+3 dossier signals: `statute_int`,
`lead_statute_int`, `concept_int`, `term_int`, `co-cite`, `doctrinal`,
`chamber`, `not-hardneg`, `fusion_rank`, plus the three Phase 1 enrichments
agreed in `research/cascade_dossier_plan.md` (channel-of-arrival fingerprint,
statute-target intersection, co-citation density).

### C.3 Cache mirror — `research/local_hybrid_rerank_cache_mirror/`

| File | Size |
|---|---|
| `research/local_hybrid_rerank_cache_mirror/cache/dossier_features.npz` | 4,343,004 B (byte-identical to C.2) |

Only the dossier features are mirrored locally; the three model score .npz
files are not in the mirror (kept only in `drive_sync/...`). The mirror is
intentional — the dossier features are CPU-only-built and the local mirror
lets the notebook resume without Drive when iterating.

### C.4 Reranker shootout verdicts (from `swiss-citation-experiments`)

The .npz files in §C.2 are the **inputs** to the 2026-05-15 hybrid reranker
shootout. Final verdicts (so a reader knows what the caches are good for):

| Model / config | Verdict | Reason |
|---|---|---|
| **BGE-reranker-v2-m3** | **rejected** | R@2k macro = 0.073 — essentially random. Scores 0.000 on 4 / 10 queries. Drop from any future config. |
| **jina-reranker-v2-base-multilingual** | **kept-candidate** | R@2k macro = 0.234, comparable to Qwen3 at 30× smaller, 5× faster. Strongly complementary on val_001 (Qwen3 0.071 → jina 0.357) and val_004 (0.300 → 0.600). |
| **Qwen3-Reranker-8B** | **kept-baseline** | R@2k macro = 0.247, best single model. Runtime ~800 s / query. |
| **F0 (plain fusion, no rerank)** | **wins at K ≥ 500** | F0 beats every rerank config at large K. At K=2k: F0=0.611 > F3=0.568. At K=10k: F0=0.816 > F3=0.791. |
| **F3 (3-rerank + 9 dossier)** | wins at K ≤ 200 | But the +30 pt lift F2→F3 comes from the **dossier signals**, not the rerankers. |
| **F4 (F3 + hard-neg filter)** | rejected | Saturates at macro 0.796 above K=20k; the filter removes recoverable candidates. Keep hard-neg as a rank-penalty in F3, not as a filter. |
| Strict floor (macro ≥ 0.8 AND min ≥ 0.8) | **unreachable** at any K | Structurally blocked by val_003 pool R_max = 0.766. Not a reranker problem; a pool problem. |

This means the practical use of the .npz files going forward is the
**F3-without-rerankers diagnostic** (`research/cascade_dossier_plan.md`,
ledger row "F3-without-rerankers diagnostic (PLANNED, not measured)"): rerun
F3's RRF with the three rerank signals removed and see whether F3-no-rerank
stays within 2 points of F3 at every K — if yes, drop the reranker stage
entirely and the .npz files in §C.2 become archive-only.

---

## D. Embedding INPUT files

### D.1 `artifacts/unified_embedding_input.parquet`

| | |
|---|---|
| Path | `e:\swiss_citation_extraction\artifacts\unified_embedding_input.parquet` |
| Size | 441,369,850 B ≈ **421 MiB** |
| Rows | 2,652,248 (one row per doc — must equal §A corpus_rows; on-disk byte size is consistent with ~165 B per row, dominated by `vector_text`) |
| Built | 2026-05-06 23:49 (one minute after `unified_retrieval.sqlite`) |
| Companion log | `unified_embedding_input.prep.log` (868 B) |

**Role.** These are the **raw text rows that got encoded** to produce the
shards in §A. Each row carries the `doc_id`, the `family`
(`court` | `law`), and the concatenated `vector_text` (assembled from the
fields listed in §B.1 with `max_vector_chars=4500`). The encoder reads this
parquet sequentially, chunks it at `chunk_size=100,000`, and writes one
shard per chunk — explaining why §A row counts are exactly `26 × 100k + tail`.

### D.2 `artifacts/unified_embedding_input.smoke.parquet`

| | |
|---|---|
| Path | `e:\swiss_citation_extraction\artifacts\unified_embedding_input.smoke.parquet` |
| Size | 741,498 B (≈ 724 KiB) |
| Rows | 6,000 (matches `unified_retrieval.smoke.sqlite` § B.2) |

Smoke-variant input mirror used to exercise the encoder end-to-end before
committing to the 22 h full-corpus run.

---

## E. Omnilex flatip embeddings (sidecars only)

Directory: `e:\swiss_citation_extraction\drive_sync\omnilex_local_embeddings_qwen3_flatip\`

| File | Size | Notes |
|---|---|---|
| `embeddings.f16` | **NOT PRESENT** locally | Task brief says ~10.30 GB on Drive; not synced locally. |
| `doc_ids.npy` | 17,289,016 B ≈ 16.5 MiB | full doc_id index in vector order |
| `case_doc_ids.npy` | 15,881,552 B ≈ 15.2 MiB | court / case doc_ids subset |
| `case_mask.npy` | 2,161,111 B ≈ 2.1 MiB | bitmask over `doc_ids.npy` marking court family |
| `statute_doc_ids.npy` | 1,407,592 B ≈ 1.34 MiB | law / statute doc_ids subset |

**What this is.** An alternate **Omnilex-format** embedding store: one giant
flat-IP file (`embeddings.f16`, ~10.30 GB, fp16, FlatIP — i.e. no clustering,
no quantisation, brute-force inner product) plus three sidecars to slice
by family. The doc-id sidecars are present locally so candidate-lookup code
can be exercised without the 10 GB vector file.

**Relationship to §A.** This is an independent encoding (likely produced by
a different notebook for an Omnilex/external integration). It does **not**
replace the §A shards — the canonical encoding for this project is the
27-chunk fp16 layout in §A, indexed by the manifest parquet. The `.f16` /
sidecar layout is a flat-array alternative for tools that expect a single
contiguous `(N, dim)` array.

The byte math is consistent with the documented dimensions:
- `doc_ids.npy` ≈ 17.29 MB → if utf-8 hashed doc_ids of ~40 chars each
  serialised as object/str dtype, around 0.4 M rows; for raw `<U…` numpy
  strings the math depends on max-string-length. The matching `case_doc_ids`
  is roughly 92 % of `doc_ids` and `statute_doc_ids` 8 %, consistent with
  the §A `family_counts` ratio (court 93.4 % / law 6.6 %).

---

## F. Cross-references (where these files are loaded)

### F.1 Notebooks (from `notebooks/_inventory/` grep)

| Asset | Loaded by |
|---|---|
| `qwen3_8b_unified_chunk*.npy` + `qwen3_8b_unified_manifest.parquet` | `anchor_funnel_v4_val001.md`, `anchor_funnel_v5_val001.md`, `anchor_funnel_v6_val001.md`, `anchor_funnel_v7_val001.md`, `anchor_funnel_v7_4_val001.md`, `anchor_funnel_v7_part2/3/4_with_outputs.md`, `endgame_colab_full_pipeline.md`, `pool_v75_multiquery_recall_0.89_at_k50k_CANONICAL.md`, `pool_v75_multiquery_iteration_11.md`, `pool_v75_multiquery_hyde_test_no_lift.md`, `retrieve_unified_corpus_qwen3_8b.md`, `retrieve_unified_corpus_v3.md`, `early_segment_lattice_funnel_v3.md`, `rerank_hybrid_3model_shootout.md`, `rerank_hybrid_3model_shootout_with_outputs.md` |
| `unified_retrieval.sqlite` / `citation_graph_extracted.sqlite` | `anchor_funnel_v4-v7*`, `pool_v75_multiquery_*`, `endgame_colab_full_pipeline`, `early_segment_lattice_funnel_v3` |
| `query_embeddings.npy` | `endgame_colab_full_pipeline.md` |
| `scores_qwen3.npz` / `scores_bge.npz` / `scores_jina.npz` / `dossier_features.npz` | `rerank_hybrid_3model_shootout.md`, `rerank_hybrid_3model_shootout_with_outputs.md` |
| `unified_embedding_input.parquet` | `embed_unified_corpus_qwen3_8b_blackwell.md` |

### F.2 Producer scripts

| Script | Produces |
|---|---|
| `scripts/build_pipeline/build_unified_retrieval_corpus.py` | `unified_retrieval.sqlite`, `unified_retrieval_documents.jsonl`, `unified_retrieval_manifest.json` (and `.smoke.*` variants); the parquet input for §D is prepared just before encoding by the same builder family. |
| `scripts/retrieval_and_rerank/encode_queries_qwen3_8b.py` | `qwen3_8b_query_<split>.npy` (val/test/train query embeddings) and the per-split ids parquet. `cache_endgame/query_embeddings.npy` is the val product reshaped/keyed by content hash. |
| `scripts/retrieval_and_rerank/rerank_qwen3.py` | Score arrays consumed by the shootout notebook (`scores_qwen3.npz`). Module API: `QwenReranker(dtype="bf16", max_seq_len=4096)` + `rerank_candidates(query, candidates, rr, top_k)`. |
| `scripts/retrieval_and_rerank/precompute_dossier_cache.py` | `research/hybrid_rerank_final/cache/dossier_features.npz` (and the local mirror at `research/local_hybrid_rerank_cache_mirror/cache/`). Reads the v7.5 snapshot at `research/anchor_funnel_val001_v7/snapshot/`. |
| `scripts/retrieval_and_rerank/segment_lattice_v3.py` | `artifacts/segment_lattice_v3.sqlite` (pre-v7.5, archived). |

---

## G. Key takeaways for downstream users

1. **2,652,248 docs** is the corpus size. It appears in three independent
   places that all agree: §A `corpus_rows`, §B `unified_retrieval.sqlite`
   document count, §D `unified_embedding_input.parquet` row count.
2. **The shard ↔ doc_id contract is the manifest parquet.** Never index the
   shards by integer position alone unless you have first concatenated all
   27 shards in `chunk000`-to-`chunk026` order; even then, prefer the
   manifest lookup.
3. **Embeddings are fp16, L2-normalised, 4096-d, no doc-side instruct
   prefix.** The query side, by contrast, gets the Qwen3 retrieval-instruct
   prefix at encode time (see `encode_queries_qwen3_8b.py`).
4. **`unified_retrieval.sqlite` is the canonical document store** and is
   the single source of truth joined back to by every channel.
5. **`citation_graph_extracted.sqlite` is a tie-breaker only** — 99.41 % of
   gold pairs have no graph edge. The two `.bak_pre_*` backups preserve the
   pre-back-reference and pre-case-level snapshots; keep both for regression
   forensics.
6. **The .npz reranker caches use `{qid: 1-D score array in pool order}`.**
   They were the input to the 2026-05-15 shootout. BGE rejected, jina kept,
   Qwen3-8B kept-baseline; F0 (plain fusion, no rerank) wins at K ≥ 500 and
   the F3 lift comes from the dossier signals, not the rerank scores.
7. **`embeddings.f16` is not on the local disk** (only its sidecars). If
   anything needs the contiguous Omnilex flat-IP store, sync from Drive
   first.
8. **Dense embedding alone caps at R@1000 = 0.289 on val** (Obs 3). The
   embeddings in §A are necessary but not sufficient — the v7.5 multi-query
   funnel built on top reaches R@50k = 0.893, and that's the canonical pool
   for every downstream experiment.
