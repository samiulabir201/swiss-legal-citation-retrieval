# anchor_funnel_v1_val001_base.ipynb

**Path:** notebooks/03_anchor_funnel_evolution_v4_to_v74/anchor_funnel_v1_val001_base.ipynb

## Configuration

**Goal:** verify whether a 7-channel anchor-funnel architecture brings all 42 `val_001` gold citations into the top-1000 candidate pool using only the structured `rag_enrichment` block of `court_authority_cards_v5_unified.jsonl` and the `llm_enrichment` block of `law_llm_descriptors_0000000_all.jsonl` — no BM25, no embedding model, no reranker. Pass criterion: R@1000 = 1.0 (42/42).

**Environment (from Cell 2 output):**
- Python 3.12.13
- PyTorch 2.10.0+cu128
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 95.0 GB, sm_120
- Colab GPU type declared in notebook metadata: `G4`, `machine_shape: hm`

**Libraries used:** `os`, `sys`, `json`, `time`, `math`, `gc`, `re`, `csv`, `pathlib`, `collections` (defaultdict, Counter), `torch`, `transformers` (AutoTokenizer, AutoModelForCausalLM), `google.colab.drive`.

**Query-expansion LLM:**
- `qwen_model_id = "Qwen/Qwen3-8B"` loaded with `torch_dtype=torch.bfloat16`, `device_map="auto"`, `trust_remote_code=True`
- `qwen_max_new_tokens = 1024`
- `enable_thinking=False` in `tokenizer.apply_chat_template`
- `do_sample=False`, `temperature=0.0`
- System prompt: "You output strictly valid JSON. No commentary."

**CONFIG dict (Cell 3 / Cell 7 output):**
```
topk_final              = 1000
budget_statute          = 400
budget_case             = 300
budget_sibling          = 300
budget_concept          = 500
budget_term             = 400
budget_bedrock          = 100
rrf_k                   = 60
bedrock_train_freq_min  = 0.40
bedrock_max_articles    = 80
qwen_model_id           = "Qwen/Qwen3-8B"
qwen_max_new_tokens     = 1024
noise_paragraph_roles   = {"notification", "header", "empty", "metadata"}
lowercase_concepts      = True
lowercase_terms         = True
```

**Code-alias table (Cell 11) — French/Italian → German abbreviation map:**
`CPP→StPO, CP→StGB, CC→ZGB, CO→OR, LTF→BGG, LACI→AVIG, LAA→UVG, LP→SchKG, LDIP→IPRG, Cst→BV, Cst.→BV, STPO→StPO, OBG→OR`.

**Regexes (Cell 11):**
- `ART_RE = r"art\.?\s*(\d+[a-z]?)"`, case-insensitive
- `CODE_RE = r"\b([A-Z][A-Za-z]{1,8}\.?)\b"`
- `CASE_BGE_RE = r"BGE\s+(\d+)\s+([IVX]+)\s+(\d+)"`
- `CASE_DOCKET_RE = r"\b(\d[A-Z]_\d+/\d{4})\b"`
- `TOKEN_NORM_RE = r"\s+"`

**RRF:** Cormack 2009, k=60, rank+1 (1-based).

## Data

DATA_ROOT auto-detected at runtime to `/content/drive/MyDrive/swiss_law` (Drive mount). Candidate roots searched, in order:
1. `/content/drive/MyDrive/swiss_law`
2. `/content/swiss_citation_extraction`
3. `E:/swiss_citation_extraction`
4. `Path.cwd()`

PATHS dict (all reported as `OK` in Cell 5 output):
- `val_csv`: `/content/drive/MyDrive/swiss_law/data/val.csv`
- `train_csv`: `/content/drive/MyDrive/swiss_law/data/train.csv`
- `train_expanded_csv`: `/content/drive/MyDrive/swiss_law/data/train_granularity_expanded.csv`
- `law_llm`: `/content/drive/MyDrive/swiss_law/data/checkpoints/law_llm_descriptors_0000000_all.jsonl`
- `court_v5`: `/content/drive/MyDrive/swiss_law/artifacts_v2/court_authority_cards_v5_unified.jsonl`
- `out_dir`: `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001` (created if missing)

**Output artifacts written (Cell 27 output):**
- `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001/val_001_query_targets.json` (1,012 bytes)
- `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001/val_001_per_channel_recall.json` (687 bytes)
- `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001/val_001_per_gold_trace.tsv` (1,467 bytes)
- `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001/val_001_top1000_pool.txt` (34,839 bytes)
- `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001/val_001_summary.md` (2,279 bytes)

## Pipeline

### Stage 1 — Environment, Drive mount, path discovery (Cells 1–2, 4–5)
- Print Python/PyTorch versions and GPU info.
- `from google.colab import drive; drive.mount('/content/drive')`.
- Auto-detect `DATA_ROOT` by checking which candidate has `data/val.csv`.
- Build the PATHS dict and print existence (`OK`/`MISSING`) for each.

### Stage 2 — Knob panel (Cell 7)
- Declare the `CONFIG` dict (budgets, RRF k, bedrock thresholds, Qwen model ID, noise-role gate, normalization flags).
- Pretty-print the config.

### Stage 3 — Load val_001 query and gold (Cell 9)
- Raise CSV field-size limit.
- Open `data/val.csv`, find the row with `query_id == "val_001"`.
- Store `QUERY`, `GOLD` (list, split by `;`), `GOLD_SET`.

### Stage 4 — One-pass streaming index build (Cell 11)
Define normalizers:
- `statute_anchor_canonical(raw)` extracts `<article_number> <CODE>` (FR/IT → DE via CODE_ALIAS). E.g., `"Art. 221 Abs. 1 lit. b StPO" → "221 StPO"`; `"art. 221 al. 1 let. b CPP" → "221 StPO"`.
- `case_anchor_canonical(raw)` extracts `"BGE X Y Z"` or `"NU_NNNN/YYYY"`.
- `norm_token(s, lower)` collapses whitespace and lowercases.

Stream both JSONL files exactly once. For each row, populate:
- `cit_to_doc_ids[citation] -> list[doc_id]`
- `doc_meta[doc_id] -> {citation, family, court_base, paragraph_role, is_notification_paragraph}`
- Inverted indexes: `idx_statute_anchor`, `idx_case_anchor`, `idx_court_base`, `idx_concept_en`, `idx_term_orig`

doc_id naming: `law:{i}` / `court:{i}`.
- For **law** rows: index the row's own `citation` as a statute anchor; index `llm_enrichment.concepts_en` and `llm_enrichment.terms_de_to_en` (both DE side and EN side go to their respective indexes).
- For **court** rows: index `court_base` and its case-anchor canonical form, and the rag-enrichment lists `statute_anchors`, `case_anchors`, `concepts_en`, `terms_original`.

### Stage 5 — Verify gold→doc mapping (Cell 13)
For every gold citation, look up `cit_to_doc_ids`. Report mapped vs unmapped and build `gold_doc_set` (union of all gold doc_ids).

### Stage 6 — Procedural-bedrock derivation from train.csv (Cell 15)
- Prefer `train_granularity_expanded.csv`, fall back to `train.csv`.
- Count, across train queries, how many times each statute-shaped citation appears in `gold_citations`.
- Set `threshold = ceil(bedrock_train_freq_min * n_train)`.
- `bedrock_articles` = those with `count ≥ threshold` (capped at `bedrock_max_articles=80`).
- Resolve `bedrock_doc_ids` via `cit_to_doc_ids`.

### Stage 7 — Query expansion via Qwen3-8B (Cell 17)
- Build a strict-JSON prompt requesting 6 fields: `statute_targets`, `case_targets`, `concept_targets_en`, `term_targets_de`, `term_targets_fr`, `legal_area_keywords`.
- Load tokenizer + model (`bfloat16`, `device_map="auto"`).
- `apply_chat_template(..., enable_thinking=False)`.
- Generate with greedy decoding, `max_new_tokens=1024`.
- Recover JSON by finding first `{` and last `}` in the completion.
- Write the parsed dict to `out_dir/val_001_query_targets.json`.
- Free model + GPU cache.

### Stage 8 — Channel implementations (Cell 19)
Each channel returns a list of `(doc_id, score)` sorted by score descending, truncated to its budget:
- `channel_statute` — count of distinct target statutes matching `idx_statute_anchor`.
- `channel_case` — count of distinct target cases matching `idx_case_anchor`.
- `channel_sibling` — for each seed court row, pull all rows sharing the same `court_base`; score=1.0.
- `channel_concept` — count of distinct target English concepts matching `idx_concept_en`.
- `channel_term` — count of distinct DE+FR target terms matching `idx_term_orig`.
- `channel_bedrock` — emits `bedrock_doc_ids` truncated to budget; score=1.0.

### Stage 9 — Run channels & per-channel diagnostics (Cell 21)
- Run all six channels.
- Seed for sibling expansion: union of court rows from statute channel plus all case-channel hits.
- Print per-channel size and gold-in-channel recall; print union recall over all channels.

### Stage 10 — RRF fusion + negative gate + R@K curve (Cell 23)
- `rrf_fuse(channels, k)` accumulates `1 / (k + rank + 1)` per doc_id.
- Sort descending.
- `apply_neg_gate`: drop rows where `is_notification_paragraph=True` or `paragraph_role` lowercased is in `noise_paragraph_roles`.
- Print pre/post-gate pool sizes and an R@K table for K in `[50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 5000, len]`.
- Compare observed R@1000 to pass criterion 1.0.

### Stage 11 — Per-gold trace (Cell 25)
- For each gold citation, compute best (smallest) rank across its mapped doc_ids in the gated fused list, and the channel set that produced it.
- Print a sorted table; then print the list of misses with their channels.

### Stage 12 — Save artifacts (Cell 27)
Write five files under `out_dir`: query targets, per-channel recall JSON, per-gold trace TSV, top-1000 pool TXT, and `val_001_summary.md` (markdown table of per-channel recall + missed-gold table).

### Stage 13 — Diagnostic playbook (Cell 28, markdown)
Decision table mapping per-gold symptoms (`rank=unranked, channels=(none)`, single-channel hits, missing bedrock, missing dockets) to architectural fixes (raise budgets, lower `bedrock_train_freq_min`, add an `expected_docket_decisions` prompt field, etc.). Explicit instruction: do NOT lower the negative gate.

### Stage 14 — Memory cleanup (Cell 30)
Delete all index dicts, `gc.collect()`, empty CUDA cache.

## Results

**Cell 5 — paths:** all five PATHS resolved `OK` against `/content/drive/MyDrive/swiss_law/`.

**Cell 9 — query loaded:**
- `val_001: 42 gold citations`
- Query (first 200 chars): `"May a court lawfully order a three‑month extension of pre‑trial detention under Art. 221 Abs. 1 lit. b StPO (risk of collusion) consistent with the principle of proportionality when the accused—detain..."`
- First 5 gold: `['Art. 221 Abs. 1 StPO', 'Art. 140 Abs. 1 StGB', 'Art. 396 Abs. 1 StPO', 'Art. 222 StPO', 'Art. 393 Abs. 1 StPO']`

**Cell 11 — index build:**
- Law: 173,033 rows indexed in 4.9 s.
- Court: 2,476,315 rows indexed in 155.9 s.
- Total docs: 2,649,348.
- Unique citations: 2,158,211.
- Index sizes: `statute_anchor: 84,793` keys, `case_anchor: 157,227`, `court_base: 178,593`, `concept_en: 292,946`, `term_orig: 363,331`.

**Cell 13 — gold mapping:** 42/42 mapped, 0 unmapped, 42 total gold doc_ids.

**Cell 15 — bedrock derivation:**
- Source: `train_granularity_expanded.csv`.
- Train queries: 1,139; threshold ≥456 occurrences (40% of queries).
- **Bedrock article count: 0.** No statute article appears in ≥40% of train gold sets when train is the granularity-expanded version. `bedrock_doc_ids: 0`.

**Cell 17 — Qwen3-8B query expansion** (after `torch_dtype` deprecation warning and an info-level note that `temperature/top_p/top_k` flags are ignored under `do_sample=False`):

RAW completion (first 500 chars):
```
{
  "statute_targets": ["Art. 221 StPO", "Art. 222 StPO", "Art. 227 StPO", "Art. 212 StPO", "Art. 100 BGG", "Art. 42 BGG"],
  "case_targets": ["BGE 135 III 315", "BGE 137 III 257", "BGE 140 III 285"],
  "concept_targets_en": ["proportionality", "pre-trial detention", "collusion risk", "extension of detention", "risk of reoffending", "investigative steps", "release justification"],
  "term_targets_de": ["Untersuchungshaft", "Kollusionsgefahr", "Verhältnismäßigkeit", "Verlängerung der Untersuchung
```

Parsed target sizes:
- `statute_targets`: 6
- `case_targets`: 3
- `concept_targets_en`: 7
- `term_targets_de`: 6
- `term_targets_fr`: 5
- `legal_area_keywords`: 5

**Cell 21 — per-channel recall (denominator = 42):**

```
channel                   size   gold_in_ch  recall
------------------------------------------------------------
statute_anchor             400            1    2.4%
case_anchor                 49            0    0.0%
sibling_expansion          300            0    0.0%
concept_en                 500            1    2.4%
term_orig                  400            3    7.1%
bedrock                      0            0    0.0%
```

Union of all channels: 1,584 unique doc_ids. Gold in union: **5/42**.

**Cell 23 — RRF fusion + negative gate, R@K curve:**

```
Pre-gate fused pool:  1,584 doc_ids
Post-gate fused pool: 1,483 doc_ids

     K   gold/42  recall
------------------------------
    50      0/42    0.0%
   100      0/42    0.0%
   200      0/42    0.0%
   300      1/42    2.4%
   500      1/42    2.4%
   750      3/42    7.1%
  1000      5/42   11.9%
  1483      5/42   11.9%
```

PASS CRITERION: R@1000 = 1.0. **OBSERVED: R@1000 = 0.119 (5/42).** Test FAILS.

**Cell 25 — per-gold trace, only 5 of 42 gold ranked at all:**

| rank | in1k | channels | citation |
|---:|---|---|---|
| 222 | YES | `concept_en` | BGE 137 IV 122 E. 6.4 |
| 626 | YES | `term_orig` | BGE 137 IV 122 E. 4.2 |
| 745 | YES | `term_orig` | 1B_357/2022 E. 3.1 |
| 772 | YES | `term_orig` | 1B_90/2021 E. 2.1 |
| 819 | YES | `statute_anchor` | BGE 139 IV 270 E. 3.1 |

Missed: 37/42, every one with `channels=(no channel hit)` — i.e., **none of the six channels produced any hit for these gold doc_ids at all**. Missed gold includes the directly-named statute `Art. 221 Abs. 1 StPO`, the Federal-Court appeal articles `Art. 100 Abs. 1 BGG`, `Art. 422 / 428 StPO`, and every BGE / docket gold the query expansion didn't enumerate exactly (e.g., `BGE 137 IV 122 E. 6.2`, `BGE 132 I 21 E. 3.2`, `1B_210/2023 E. 4.1`, `7B_496/2025 E. 3.2`).

**Cell 27 — artifacts written** to `/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001/` (sizes listed in Data section).

**Cell 30 — `Cleaned up.`**

## Summary

The v1 anchor-funnel baseline ran end-to-end over the full 2.65M-doc corpus (173k law + 2.48M court rows indexed in ~161 s on the Blackwell GPU box) and produced a measurable R@1000 number on val_001 — **0.119 (5/42)** vs the 1.0 pass criterion, so this verification fails by a wide margin. Per-channel diagnostics show statute, concept_en and term_orig each pull in 1–3 gold, case_anchor and sibling_expansion contribute 0, and the bedrock channel produces 0 doc_ids because no article hits the ≥40% train-gold threshold on `train_granularity_expanded.csv`. The dominant failure mode is `channels=(no channel hit)` for 37/42 gold: the LLM-anchor mismatch — e.g., the directly-named gold `Art. 221 Abs. 1 StPO` is in the query yet doesn't match because Qwen3-8B emitted `"Art. 221 StPO"` (paragraph-stripped) but the corpus statute-anchor index, fed by `rag_enrichment.statute_anchors`, doesn't agree on canonical form for *this* gold's row, plus none of the gold dockets (`1B_210/2023`, `7B_496/2025`, etc.) appear in the Qwen target list at all. Lessons captured by Cell 28: lower `bedrock_train_freq_min`, prompt Qwen for an explicit `expected_docket_decisions` list, raise `budget_concept`, and *do not* relax the negative gate; the architecture is sound but recall depends on the LLM's enumeration coverage and on bedrock not collapsing to zero on the expanded train set.
