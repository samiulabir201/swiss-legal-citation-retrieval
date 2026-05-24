# Court-Consideration Funnel — Input Architecture Design

_Generated 2026-05-21. Mirrors the proven laws_de Hybrid v12 pipeline (F1=0.777 on val) but adapted to the court_consideration corpus (2.47M paragraphs, tri-lingual)._

## 0. What worked for laws_de (the architecture we're cloning)

The laws_de pipeline uses **3 stages with strategic LLM routing**:

```
query EN  ─→  BM25 top-100 (TLF-enhanced bilingual tokens)
                │  R@100 ≈ 0.90, so retrieval is essentially solved
                ▼
         Qwen3-Reranker-8B (yes/no logit softmax)
                │
   fused = 0.7 · norm_minmax(BM25)  +  0.3 · P(yes)
                │
        ┌────── fused ≥ 0.55  ───→ AUTO-YES        (~5 / query)
        ├────── fused <  0.25 ───→ AUTO-NO         (~85 / query)
        └──────  otherwise   ───→ BORDERLINE      (~10 / query)
                                       │
                                       ▼
                           Qwen3-8B judge (7-category prompt)
                           runs ONLY on borderline ~10/q
                                       │
                  Final = AUTO-YES ∪ (BORDERLINE ∧ verdict=YES)
```

**Key contributors:**
- **TLF prior** mined from train.csv: query-token → law-abbreviation co-occurrence, weighted by IDF. Top-5 abbreviations injected into the BM25 query as repeated tokens. (`enhance_tokens_with_tlf`)
- **Bilingual tokens**: EN query + cached EN→DE Google Translate appended. Deduplicated.
- **Cross-encoder reranker**: scores each (query, candidate) pair using yes/no logit softmax. Document is formatted with structured fields: `citation_canon`, `law_abbreviation`, `law_name_en`, `title`, `context_heading_title`, `provision_type`, `text[:500]`.
- **3-zone fusion**: keeps LLM out of easy decisions (avoids gold-prior dilution). LLM runs on ~10% of candidates.
- **Default-YES on judge failure**: the judge's parse falls back to YES if no `CITATION | VERDICT` line is recovered. This pushes precision-recall trade-off toward recall — important when gold is large per query.

The LLM judge contributed **+0.096 F1** over the threshold-only ablation. That's the entire reason the LLM stage exists.

---

## 1. The 8 input artifacts to build for the court funnel

Each file is the analog of a laws_de input. Bold rows are NEW for courts (no analog in laws_de).

| # | File | Analog | Size estimate | Notes |
|---|---|---|---:|---|
| 1 | `court_corpus.parquet` | `corpus.parquet` | ~2.5 GB | One row per paragraph. All v4 structural fields + raw text |
| 2 | `court_knowledge_base.jsonl` | `laws_knowledge_base.jsonl` | ~800 MB | Per-paragraph KB. Case-level metadata + EN summary of holding |
| 3 | `court_bm25_index.pkl` + `court_bm25_ids.pkl` | `bm25_v2_*.pkl` | ~10 GB | BM25 over tri-lingual text. Built once, cached |
| 4 | `query_translations_court.json` | `query_translations_trainval.json` | ~3 MB | Already have EN→DE; need to ADD EN→FR and EN→IT |
| 5 | `court_tlf_prior.pkl` | the `tlf` Counter | ~50 MB | Token → (chamber_area, case_id, named_doctrine) co-occurrence from train.csv gold court rows |
| 6 | **`court_static_scores.parquet`** | — (new) | ~200 MB | Precomputed v4 `gold_candidate_score` per row. Used as 3rd fusion signal |
| 7 | **`court_cocitation_counts.parquet`** | — (new) | ~50 MB | Precomputed `cited_by_count` per case_id |
| 8 | `court_judge_system_prompt.txt` | embedded in cell 21 | < 5 KB | 6-subclass Swiss court paragraph judge |

---

## 2. Detailed schemas (the file-by-file blueprint)

### 2.1 `court_corpus.parquet`

One row per paragraph. Columns mirror laws_de + add court-specific fields from our v4 schema:

```
citation_canon        str    — "BGE 137 IV 122 E. 4.1" (always BGE-form regardless of source language)
citation_raw          str    — original from court_considerations.csv `citation` column
text                  str    — paragraph body
text_clean            str    — page-anchors stripped (e.g. "BGE 137 IV 122 S. 128" removed)
language              str    — de / fr / it (from v4 extractor)

# Case-level identity
case_id               str    — "BGE 137 IV 122" or "1B_28/2022"
bge_id                str    — for published
bge_volume            int
bge_series_roman      str    — I/II/III/IV/V/VI
bge_start_page        int
docket_no             str    — for unpublished BGer
chamber_prefix        str
chamber_area          str    — criminal/civil/public_law/social_insurance/...
decision_date         date
publication_status    str    — bge_published / unpublished / zur_publikation_vorgesehen
docket_era            str    — pre_1990 / 1990_2006 / 2007_2023 / post_2024
presiding_judge       str

# E-id structure
erwagung_id_raw       str    — raw E-tag
erwagung_id_validated str    — cleaned "4.1" or "A.b"
erwagung_id_artifact_class  str
parent_section        str
depth                 int
sachverhalt_letter    str
is_continuation_fragment    bool
is_subsection_header_only   bool

# 7-element template + paragraph role
has_element_1_statutory_anchor   bool
has_element_3_teleology          bool
has_element_4_positive_limb      bool
has_element_5_negative_limb      bool
has_element_6_interpretive_factors  bool
has_element_7_authority_chain    bool
template_score                   int (0-7)
opener_verb_owner                str   — court/appellant/lower_court/legislator/doctrine/unclear
paragraph_role                   str

# Statute references
statute_anchors       JSON   — list of {sr_number, statute_short_de, article, abs, lit}
lead_statute_anchors  JSON   — anchors in first 200 chars
cantonal_code_anchors JSON
i_v_m_chains          JSON

# Citation graph
cited_bge             JSON   — list of BGEs cited
cited_urteil          JSON   — list of unpublished dockets cited
cited_doctrine        JSON
cited_commentary_count int
cited_periodical_count int
cited_treaty_count    int
mit_hinweisen_tail    str
cited_by_count        int    — back-reference count (from co-citation graph)

# Doctrinal-test invocation
named_doctrine        JSON   — Schubert / Reneja / Engel / ...
holding_polarity      str    — holds_up / fails / none
evidentiary_standard_invoked str
is_obiter_dictum      bool

# Quality / cleanup flags
is_dispositif         bool
is_cost_dispositif    bool
is_signature_boilerplate bool
is_admissibility_recital bool
is_csv_parse_artifact bool
has_anonymized_party  bool
text_length_chars     int

# COMPOSITE — the v4 calibrated static prior
gold_candidate_score  float  — 0..1, calibrated via logistic regression on train+val

# BM25 helpers
bm25_text             str    — lowercase, tokenizable text (citation + heading + paragraph)
embed_text            str    — same but with optional metadata prefix (for dense embeddings)
word_count            int
```

**Build cost**: extract once with `extract_v2_static_fields.py` + LLM-enrich pass for `opener_verb_owner`, `paragraph_role`. ~15-30 min static + ~6 hr LLM (or ~$50-100 cloud GPU).

### 2.2 `court_knowledge_base.jsonl`

Per-paragraph KB. Same nested structure as `laws_knowledge_base.jsonl`:

```json
{
  "id": "bge_137_iv_122_e4_1",
  "record_uid": "bge_137_iv_122_e4_1__0",
  "citation_canon": "BGE 137 IV 122 E. 4.1",
  "citation": {
    "citation_raw": "BGE 137 IV 122 E. 4.1",
    "citation_match_key": "bge 137 iv 122 e 4 1",
    "case_id": "BGE 137 IV 122",
    "eid": "4.1",
    "citation_patterns": ["BGE 137 IV 122 E. 4.1", "ATF 137 IV 122 consid. 4.1", "DTF 137 IV 122 consid. 4.1"],
    "citation_aliases": ["BGE 137 IV 122", "ATF 137 IV 122", "DTF 137 IV 122"]
  },
  "case": {
    "case_id": "BGE 137 IV 122",
    "bge_volume": 137,
    "bge_series": "IV",
    "bge_page": 122,
    "decision_date": "2011-05-04",
    "chamber_prefix": "1B_",
    "chamber_area": "criminal_prelim",
    "publication_status": "bge_published",
    "case_name_en": "Pre-trial detention; collusion risk",
    "leitentscheid_holding_en": "Detention on collusion-risk grounds requires concrete circumstances; mere theoretical possibility of witness influence is insufficient.",
    "leitentscheid_holding_de": "Der besondere Haftgrund der Kollusionsgefahr liegt vor, wenn ernsthaft zu befürchten ist...",
    "primary_statutes": [
      {"sr": "312.0", "short_de": "StPO", "short_fr": "CPP", "short_it": "CPP", "article": "221", "abs": "1", "lit": "b"}
    ],
    "cited_by_count": 710,
    "is_leitentscheid": true
  },
  "paragraph": {
    "eid": "4.1",
    "language": "de",
    "paragraph_role": "party_position",
    "opener_verb_owner": "appellant",
    "template_score": 0,
    "named_doctrines": [],
    "holding_polarity": null,
    "text_length_chars": 412,
    "paragraph_summary_en": "Appellant disputes the lower court's collusion-risk finding..."
  },
  "structure": {
    "structure_path": "BGE 137 IV 122 > E. 4 > E. 4.1",
    "parent_eid": "4",
    "depth": 2,
    "siblings_at_depth": ["4.1", "4.2", "4.3", "4.4"],
    "children": []
  },
  "semantic": {
    "doctrinal_keywords_de": ["Kollusionsgefahr", "Haftgrund", "Untersuchungshaft"],
    "doctrinal_keywords_fr": ["risque de collusion", "détention provisoire"],
    "doctrinal_keywords_it": ["pericolo di collusione"],
    "doctrinal_keywords_en": ["collusion risk", "pre-trial detention"],
    "sub_class": "party_position"
  }
}
```

The **`paragraph_summary_en`** field is the analog of `law_name_en` — a 1-2 sentence English gloss the judge will read. Generated once via the LLM-enrich pass.

### 2.3 `court_bm25_index.pkl` + `court_bm25_ids.pkl`

BM25-Okapi index over all 2.47M paragraphs. Key adaptations vs laws_de:

- **Tokenization is tri-lingual**: needs DE+FR+IT stop-word handling, umlaut/accent normalization
- **bm25_text combines layers**: `citation_canon + case_id + chamber_area + named_doctrines + paragraph_text` (similar to laws_de's `bm25_text` which combines `citation_canon + law_abbrev (5x) + law_full_name + title + heading + provision_type + text`)
- **Recall target**: R@500 ≥ 0.85 (laws_de hits R@100 ≈ 0.90 — court corpus is 14x larger so we need wider top-K)

Recommended `TOP_BM25 = 500` for courts (vs 100 for laws), because:
- 2.47M corpus vs 171K = 14x larger search space
- Tri-lingual mix means more dispersion
- Each query has 4-23 gold court paragraphs; coverage matters more

Build cost: 30 min on a single thread, ~10 GB on disk.

### 2.4 `query_translations_court.json`

Already have `query_translations_trainval.json` (EN→DE) from the laws pipeline — **reuse it**.

NEEDED ADDITIONS:
- EN→FR translation (Google Translate or equivalent)
- EN→IT translation

For court_considerations, all three Swiss official languages need first-class query tokens because the corpus has ~30% FR rows and ~6% IT rows. Build cost: Google Translate API ~10 min, ~$1.

Output structure (per `query_id`):
```json
{
  "val_001": {
    "de": "...German translation...",
    "fr": "...French translation...",
    "it": "...Italian translation..."
  },
  ...
}
```

The BM25 `enhance_tokens_with_tlf` will concatenate `tokens_en + tokens_de + tokens_fr + tokens_it`, deduplicated.

### 2.5 `court_tlf_prior.pkl`

Mine the **train.csv** gold court citations. For each token in the query (EN+DE+FR+IT), count co-occurrence with multiple target buckets:

```python
tlf_chamber_area = Counter()  # token → {chamber_area: count}
tlf_case_id      = Counter()  # token → {case_id: count}  — for top-cited Leitentscheide
tlf_doctrine     = Counter()  # token → {named_doctrine: count}
```

Then `enhance_tokens_with_tlf` injects top-K from each bucket. This is critical because **court_consideration_text is dispersed**: a query about "Kollusionsgefahr" should boost paragraphs from chamber_area `criminal_prelim` AND case_id `BGE 137 IV 122` AND doctrinal_keyword `Kollusionsgefahr`. Not just one.

Build cost: ~5 min over train.csv (1,139 queries).

### 2.6 `court_static_scores.parquet` (NEW)

For every row in `court_corpus.parquet`, the precomputed v4 `gold_candidate_score` from our calibrated logistic regression. This becomes the **third signal in fusion**:

```
fused = 0.4 * norm_minmax(BM25)
      + 0.3 * P(yes)_from_reranker
      + 0.3 * gold_candidate_score_v4
```

The static score gives the BM25/reranker a strong prior on which rows are "gold-class" regardless of query — and the v4 score is calibrated for that exact task (AUC 0.80, k=100 precision 27%).

Build cost: ~10 min (apply v4 weights to extracted features).

### 2.7 `court_cocitation_counts.parquet` (NEW)

Per case_id, the count of other corpus paragraphs that cite it. From `cocitation_graph_v2.py`. Used at two points:
- As an extracted feature in `court_corpus.parquet` (already there)
- As a separate lookup file for quick case-level scoring before BM25 (filter out cases with `cited_by_count < 5` to drop noise)

Build cost: already done (~30 min for full 2.47M scan).

### 2.8 `court_judge_system_prompt.txt` — the 6-sub-class court judge prompt

The laws_de judge uses a 7-category prompt (substantive law / definitions / procedural / appeal / cost / jurisdiction / constitutional). The court judge should use the 6-sub-class framework we discovered in deep analysis:

```
You are a Swiss Federal Court (Bundesgericht) doctrinal-paragraph expert.

DOMAIN KNOWLEDGE — what makes a court paragraph cited as gold for a legal query:

A "gold" court paragraph for a query may belong to ANY of these 6 sub-classes:

1. RULE_STATEMENT — the canonical 7-element doctrinal template:
   - statutory anchor ("Gemäss Art. N CODE / Selon l'art. N CODE / Ai sensi dell'art. N CODE")
   - paraphrase in court voice
   - teleology ("soll verhindern / vise à empêcher / mira a")
   - positive limb ("namentlich / notamment / segnatamente")
   - negative limb ("genügt indessen nicht / ne saurait suffire / non basta")
   - interpretive factors (multi-factor balancing)
   - closing authority chain ("mit Hinweisen / et les références / con rinvii")

2. PARTY_POSITION — appellant's argument paragraph from the canonical case
   (e.g. "Der Beschwerdeführer bestreitet... / Le recourant fait valoir...").
   Often included in gold as context for the rule-statement.

3. LOWER_COURT_SUMMARY — Vorinstanz / cour cantonale / autorità precedente summary
   ("Die Vorinstanz hat erwogen / La cour cantonale a retenu / L'autorità precedente ha ritenuto").

4. STATUTE_QUOTE — paragraph that quotes the statute verbatim:
   ("Nach Art. 168 Abs. 1 ZPO sind als Beweismittel zulässig: a. Zeugnis, b. Urkunde, ...")

5. APPLICATION_IN_CONCRETO — the rule meeting the facts of the case
   ("Im vorliegenden Fall / En l'espèce / Nella fattispecie / In concreto").

6. FRAGMENT — a continuation of a longer doctrinal block that was split by E-id.
   Mid-sentence opening, lowercase, completes a sentence from the previous paragraph.

DOMAIN KNOWLEDGE — what makes a paragraph NOT gold:

- Pure dispositif ("Die Beschwerde wird abgewiesen / Le recours est rejeté")
- Pure cost paragraph ("Gerichtskosten / Parteientschädigung / frais judiciaires")
- Signature boilerplate ("Im Namen der ... Abteilung")
- Pure admissibility recital (without doctrinal content)
- A paragraph from an UNRELATED legal domain (different chamber_area than the query)

YOUR TASK: For each candidate paragraph, read carefully and decide YES or NO.
Say YES if the paragraph belongs to any of the 6 sub-classes AND its case_id is
topically relevant to the query (matches the query's statute target OR doctrinal
sub-question OR chamber area).

Say NO if it's pure boilerplate (dispositif / cost / signature / admissibility)
OR from a clearly unrelated legal domain.

When uncertain, say YES — better to over-include than miss a contextual paragraph.

For each candidate, respond with exactly:
CITATION | VERDICT: YES or NO
```

---

## 3. Pipeline-level adaptations (beyond the inputs)

| Aspect | laws_de v12 | court_consideration (proposed) |
|---|---|---|
| Top-K from BM25 | 100 | **500** (corpus 14× larger) |
| Reranker batch | 8 | 4 (rows are longer) |
| Fusion weights | 0.7 BM25 + 0.3 reranker | **0.4 BM25 + 0.3 reranker + 0.3 gold_candidate_score_v4** |
| HIGH_THRESH (auto-YES) | 0.55 | 0.50 (more candidates per query) |
| LOW_THRESH (auto-NO) | 0.25 | 0.20 |
| Borderline pile size | ~10/query | **~30-50/query** (more dispersion) |
| Judge model | Qwen3-8B | Qwen3-8B (same) |
| Judge system prompt | 7-category laws | **6-sub-class courts** |
| Citation expansion | `Art. N` → `Art. N Abs. M` children | `BGE NNN X N` → all child `E. *` (NEW) |
| Default-YES on parse failure | yes | yes (keep) |

---

## 4. What we already have built

These are sunk costs — we don't rebuild them:

✓ `court_considerations.csv` (2.47M paragraphs) — the source
✓ `train.csv` and `val.csv` — gold annotations
✓ **v4 extractor** ([extract_v2_static_fields.py](research/scratch_2026-05-21_schema_v2/extract_v2_static_fields.py)) — produces all structural fields
✓ **v2 opener heuristic** ([llm_enrich_heuristic_v2.py](research/scratch_2026-05-21_schema_v2/llm_enrich_heuristic_v2.py)) — fills `opener_verb_owner` partial coverage
✓ **v4 calibrated weights** ([gold_candidate_score_weights_v4_final.json](research/scratch_2026-05-21_schema_v2/gold_candidate_score_weights_v4_final.json)) — the static prior
✓ **co-citation counts** ([cocitation_counts.json](research/scratch_2026-05-21_schema_v2/cocitation_counts.json)) — for case-level Leitentscheid signal
✓ EN→DE query translations from laws pipeline — reuse

## 5. What needs new build

✗ Full v4 feature extraction on 2.47M corpus → `court_corpus.parquet`
✗ LLM-enrich pass for `opener_verb_owner` (residual 62% unclear) + `paragraph_summary_en` field
✗ `court_knowledge_base.jsonl` (per-paragraph KB with English summary)
✗ BM25 index over tri-lingual text
✗ EN→FR and EN→IT query translations
✗ TLF prior mined from train.csv (3 buckets: chamber_area, case_id, named_doctrine)
✗ The 6-sub-class court judge prompt (drafted above)
✗ Citation expansion logic for `BGE NNN X N` → child `E. *` paragraphs

---

## 6. Build order (recommended)

**Phase A — Static, no GPU (1 day)**:
1. Run v4 extractor on full 2.47M → `court_corpus.parquet` (15-30 min)
2. Compute v4 `gold_candidate_score` per row → fold into parquet (5 min)
3. Build BM25 index over tri-lingual `bm25_text` (30 min)
4. Mine TLF prior from train.csv → `court_tlf_prior.pkl` (5 min)
5. Translate val queries to FR and IT → `query_translations_court.json` (30 min)
6. Build citation-expansion map for case-level → child-E rows (5 min)
7. **Acceptance criterion**: BM25 R@500 ≥ 0.85 on val

**Phase B — LLM enrich pass on filtered corpus (1-2 days on cloud GPU)**:
8. Filter `court_corpus.parquet` to rows with `gold_candidate_score ≥ 0.2` (~700K rows, ~30% of corpus)
9. Run Qwen3-4B in JSON mode to fill `opener_verb_owner`, `paragraph_role`, `doctrinal_test_invoked`, `paragraph_summary_en` for these 700K rows
10. Generate per-case `leitentscheid_holding_en` for case_ids with `cited_by_count ≥ 50` (~5K cases)
11. Assemble `court_knowledge_base.jsonl`
12. **Acceptance criterion**: enriched fields populated for 95% of high-score rows

**Phase C — Reranker + judge integration (1-2 days)**:
13. Adapt the laws_de notebook's `bm25_retrieve`, `rerank_batch`, `compute_fused_and_split`, `judge_borderline` to use court fields
14. Write the court judge system prompt (6 sub-classes)
15. Implement citation expansion (BGE-id → child E. paragraphs)
16. Run end-to-end on val. Iterate on fusion thresholds.
17. **Acceptance criterion**: macro F1 ≥ 0.70 on val court_consideration gold

---

## 7. Resource estimates

| Phase | Time | Compute cost |
|---|---|---|
| A (static) | 1 day | $0 (local CPU) |
| B (LLM enrich) | 1-2 days | $50-150 cloud A100 |
| C (pipeline) | 1-2 days | $20-50 (reranker + judge on val) |
| **Total** | **3-5 days** | **$70-200** |

---

## 8. Risk-watch list

1. **R@500 might not hit 0.85**: court paragraphs are dispersed; some gold cite different statutes than the query. If R@500 < 0.80, raise TOP_BM25 to 1000 or 2000.
2. **Judge prompt may over-fire YES**: with the "when uncertain say YES" default and 6 sub-classes, the borderline pile may be ~50/query and precision could drop. Test by setting HIGH_THRESH=0.45 vs 0.55.
3. **`paragraph_summary_en` quality**: a poor English summary in KB will mislead the judge. Use Qwen3-4B for this, not a smaller model. Spot-check 30 summaries before scaling.
4. **Multi-lingual reranker quality**: Qwen3-Reranker-8B is mostly EN-trained. For FR/IT documents, expect 0.05-0.10 lower P(yes) accuracy. Mitigation: format the document with German parallel-citation aliases (`BGE/ATF/DTF`) prominently.
5. **Citation-expansion explosion**: a query gold "BGE 137 IV 122" alone could expand to 15 child E-rows. Need to decide: expand to all-children (recall-friendly), or only the specific E mentioned (precision-friendly), or expand and require LLM confirmation. Recommend: expand to all children, judge confirms borderline.

---

## 9. Open question for the user

The `paragraph_summary_en` field is the biggest single new build. Two options:

**Option A — Per-paragraph summary**: 1-2 sentences in English for each of ~700K filtered paragraphs. Cost: ~$80 on Qwen3-4B. Benefit: every paragraph has a query-comparable English form, dramatically improving the judge's accuracy.

**Option B — Per-case holding summary**: 1 paragraph in English for each of ~5K leitentscheid cases (cited_by_count ≥ 50). Cost: ~$5. Benefit: only the "core" cases get summaries; routine echoes rely on the German text. Simpler but loses summary signal on echoes.

I'd recommend **A** for the first build (parity with laws_de which has per-article English names) and then ablate later. Want to confirm?
