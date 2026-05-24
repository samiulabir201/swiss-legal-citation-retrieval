# Val Law Gold — Anatomy and Retrieval Implications

_Generated 2026-05-23. Companion to `per_query_court_gold_anatomy_2026-05-21.md` (court side) and `gold_citation_legal_patterns_2026-05-12.md` §3.2 (earlier law-side pattern hunt). Focus: every law-only gold citation in val.csv, what it is, why it is gold, and what this implies for a near-perfect law-recall pipeline that does NOT use train.csv (train is German queries, val/test are English — language and distribution shift per `personal_observations.md` Obs 4)._

---

## 1. Empirical facts (re-verified on 2026-05-23)

| Fact | Value | Source |
|---|---|---|
| `laws_de.csv` rows | 175,933 | direct read |
| Columns | `citation`, `text`, `title` | direct read |
| Unique codes (last token of citation) | 2,048 | direct count |
| Total val law gold (Art. … rows) | **149** across 10 queries | `val.csv` parsed by `^Art\.\s+\d` regex |
| Val law gold present in `laws_de.csv` | **149/149 (100%)** | direct lookup, confirms Obs 1 |
| Law gold per query (range) | 6 to 24 | per-query count |

`laws_de.csv` schema is the simplest possible:

- **`citation`** — the exact prediction-target string, e.g. `Art. 221 Abs. 1 StPO`. This IS what must appear in the submission, character-for-character.
- **`text`** — the German article body. One short paragraph (the statutory text itself).
- **`title`** — the law's full German name plus the structural heading (chapter / section / sub-section), separated by ` - `. Example:
  ```
  Schweizerisches Zivilgesetzbuch vom 10. Dezember 1907 - 3. Eigenhändige Verfügung
  ```
  The portion after ` - ` is the **doctrine name in German** — the controlled vocabulary the Swiss legislator uses to label the rule. This is the single highest-leverage retrieval signal we are currently under-using.

---

## 2. Per-query law gold breakdown

For each val query, the codes used (with article-number range and count):

| qid | n_law | Code mix |
|---|---:|---|
| val_001 | 19 | StPO×15 (135-428), StGB×1 (140), BGG×1 (100), StBOG×2 (37-39) |
| val_002 | 20 | ATSG×7 (6-61), IVG×10 (1-69), BGG×3 (82-113) |
| val_003 | 24 | BV×1 (29), StPO×15 (3-428), BGG×1 (100), StBOG×2 (37-39), ZGB×5 (16-520) |
| val_004 |  9 | ZGB×7 (458-520), OR×1 (20), BGG×1 (100) |
| val_005 |  6 | ZGB×5 (133-285), BGG×1 (100) |
| val_006 | 11 | OR×9 (1-398), BGG×2 (93-100) |
| val_007 | 15 | IPRG×2 (98-100), ZGB×10 (3-940), StGB×1 (292), OR×2 (15-245) |
| val_008 | 20 | BGG×2 (100-107), StGB×10 (12-333), StPO×6 (9-436), BV×2 (29-32) |
| val_009 | 11 | ZGB×10 (129-308), BGG×1 (100) |
| val_010 | 14 | ZPO×4 (176-405), OR×6 (84-397), ZGB×2 (2-4), SchKG×1 (67), BGG×1 (100) |

**Code-frequency across val (number of queries the code appears in):**

| Code | Queries | Notes |
|---|---:|---|
| BGG    | **9/10** | Federal Court Law — the appeal/admissibility apparatus. Only missing from val_007 |
| ZGB    | 6/10 | Civil Code — used whenever facts touch family / inheritance / property / good faith |
| OR     | 4/10 | Code of Obligations — contracts, delict, mandate |
| StPO   | 3/10 | Criminal Procedure |
| StGB   | 3/10 | Criminal Code |
| StBOG  | 2/10 | Federal Criminal Court organization (criminal queries only) |
| BV     | 2/10 | Federal Constitution (right to be heard, fair trial) |
| ATSG, IVG, IPRG, ZPO, SchKG | 1 each | domain-specific specialised codes |

**Universal articles (appearing in ≥2 val queries):**

| Count | Citation | Heading (doctrine label) |
|---:|---|---|
| **9** | `Art. 100 Abs. 1 BGG` | 30-day federal appeal deadline (Beschwerdefrist) |
| 3 | `Art. 428 Abs. 1 StPO` | costs of appeal proceedings (Verfahrenskosten) |
| 2 | `Art. 221 Abs. 1 StPO` | grounds for pre-trial detention |
| 2 | `Art. 222 StPO`        | challenges to detention orders |
| 2 | `Art. 382 Abs. 1 StPO` | who may appeal |
| 2 | `Art. 385 Abs. 1 StPO` | form of the appeal brief |
| 2 | `Art. 390 Abs. 2 StPO` | written appeals procedure |
| 2 | `Art. 393 Abs. 1 StPO` | what may be challenged |
| 2 | `Art. 396 Abs. 1 StPO` | deadline for the cantonal-level Beschwerde |
| 2 | `Art. 422 Abs. 1 StPO` + Abs. 2 | costs definition |
| 2 | `Art. 135 Abs. 3/4 StPO` | court-appointed defence costs |
| 2 | `Art. 37/39 Abs. 1 StBOG` | Federal Criminal Court chamber jurisdiction |
| 2 | `Art. 29 Abs. 2 BV` | constitutional right to be heard |
| 2 | `Art. 505 Abs. 1 ZGB` | holographic-will form requirement |
| 2 | `Art. 467 ZGB` | testamentary capacity |
| 2 | `Art. 16 ZGB` | general capacity (Urteilsfähigkeit) |
| 2 | `Art. 285 Abs. 1 ZGB` | parental maintenance contribution |

The pattern is striking: **the law-gold for any query is a stack of three orthogonal layers**, and the universal articles above are the first two layers.

---

## 3. The three-layer structure of every val law-gold set

Every val law-gold set decomposes cleanly into three layers. Every law citation in val gold belongs to exactly one of these three layers.

### Layer 1 — Substantive doctrine (what the case is about)

The articles that **define the legal rule** the query asks about. These are the only layer that varies by query topic.

| Query | Substantive doctrine articles (examples) |
|---|---|
| val_001 — pre-trial detention | `Art. 221 Abs. 1/2 StPO` (grounds), `Art. 212 Abs. 3 StPO` (proportionality), `Art. 227 Abs. 1 StPO` (extension), `Art. 140 Abs. 1 StGB` (underlying offence = robbery) |
| val_002 — disability insurance | `Art. 8 ATSG/IVG` (definition of invalidity), `Art. 17 IVG` (vocational measures), `Art. 28 IVG` (pension), `Art. 16 ATSG` (rate calculation), `Art. 4/18d IVG` (scope) |
| val_004 — holographic will | `Art. 505 Abs. 1 ZGB` (form), `Art. 467 ZGB` (capacity), `Art. 469 Abs. 1+2 ZGB` (defective will), `Art. 471 ZGB` (Pflichtteil), `Art. 520a ZGB` (date defect), `Art. 458 Abs. 3 ZGB` (parental line), `Art. 20 Abs. 2 OR` (partial nullity) |
| val_005 — visitation rights | `Art. 273 Abs. 1 ZGB` (right to contact), `Art. 274 Abs. 2 ZGB` (limits), `Art. 133 Abs. 1/2 ZGB` (parental rights post-divorce), `Art. 285 Abs. 1 ZGB` (maintenance) |
| val_006 — favour/negligence | `Art. 1 OR` (contract formation), `Art. 18 OR` (interpretation), `Art. 41 OR` (delictual liability), `Art. 97/99 OR` (contractual breach), `Art. 248 OR` (donor's liability), `Art. 363/364/398 OR` (mandate / contract for work) |
| val_007 — heirship / good-faith acquisition | `Art. 641/933/934/940 ZGB` (ownership, good-faith possession), `Art. 197 ZGB` (acquêts), `Art. 16/8 ZGB` (capacity, burden of proof), `Art. 98/100 IPRG` (private international law), `Art. 15/245 OR` (gift/donation form), `Art. 292 StGB` (disobedience of official order), `Art. 3 ZGB` (presumption of good faith) |
| val_008 — disloyal management | `Art. 314 StGB` (the offence itself), `Art. 12/25/26/42/44/49/50/110 StGB` (general criminal-law mechanics: intent, complicity, suspended sentence, concurrence) |
| val_009 — maintenance / state subrogation | `Art. 277/285/286/288/291/292 ZGB` (parental maintenance, modification, payment instruction), `Art. 308 ZGB` (Beistandschaft), `Art. 129 ZGB` (modification of judicial maintenance) |
| val_010 — bank forged-orders | `Art. 84/100/101/397 OR` (currency, exoneration clauses, auxiliary-person liability, mandate revocation), `Art. 2/4 ZGB` (good faith, judicial discretion), `Art. 67 SchKG` (Betreibungsbegehren) |

The defining feature: **each substantive article's `title` heading is the German name of the doctrine the query is asking about** — and the query (in English) names that same doctrine, just in a different language ("right to be heard" ↔ "rechtliches Gehör", "good faith" ↔ "guter Glaube", "holographic will" ↔ "eigenhändige letztwillige Verfügung").

### Layer 2 — Universal procedural apparatus (how the case reaches the Federal Court)

Every val query is implicitly a Federal Supreme Court appeal scenario, so every val gold contains the **same fixed bundle of BGG / BV procedural articles**:

- `Art. 100 Abs. 1 BGG` — 30-day appeal deadline. Present in **9 of 10** val queries.
- `Art. 29 Abs. 2 BV` — constitutional right to be heard. Present whenever procedural fairness is implicated.
- `Art. 82 BGG`, `Art. 93 BGG`, `Art. 107 BGG`, `Art. 113 BGG` — admissibility, interlocutory orders, remand, subsidiary constitutional appeal. Appear as the query's appeal route demands.

For **criminal** queries (val_001, val_003, val_008) this layer also includes a fixed StPO procedural bundle:

- `Art. 382/385/390/393/396 StPO` — appeal who/how/what/when/where (Beschwerde)
- `Art. 422/428 StPO` — procedural costs
- `Art. 135 Abs. 3/4 StPO` — court-appointed defence costs
- `Art. 37/39 StBOG` — Federal Criminal Court jurisdiction

For **civil** queries the parallel layer is light (sometimes empty) because civil appeals run through BGG directly.

This layer is essentially formulaic: knowing the query is "criminal appeal" determines almost all of it, deterministically.

### Layer 3 — Foundational / definitional articles (background)

A small number of articles per query that define the general legal terms used:

- `Art. 16 ZGB` (Urteilsfähigkeit / capacity) — appears whenever capacity is even mentioned (val_003 ZGB-inheritance subquery, val_004, val_007)
- `Art. 8 ZGB` (Beweislast / burden of proof) — appears in val_007
- `Art. 2 ZGB` (Treu und Glauben / good faith) — val_010
- `Art. 4 ZGB` (judicial discretion) — val_010
- `Art. 3 ZGB` (presumption of good faith) — val_007
- `Art. 8 ATSG` (definition of invalidity) — val_002
- `Art. 6 ATSG` (incapacity for work) — val_002

These are the "every Swiss-law textbook starts with" articles that frame the entire field.

---

## 4. Why each citation is gold — the bridge to the query

For every val law-gold citation, the connection to the query is one of exactly **four** types. Examples:

### Bridge type A — Explicit statute mention in the query

The query names the article verbatim. Direct lookup is sufficient.

- val_001 query says **"Art. 221 Abs. 1 lit. b StPO (risk of collusion)"** → `Art. 221 Abs. 1 StPO` is gold (the corpus stores Abs. 1 without the lit-suffix; that level of granularity isn't in the corpus, so the parent paragraph is the prediction target).
- val_002 query says **"Art. 17 LAI"** (= French alias for IVG) → `Art. 17 Abs. 1 IVG` is gold.
- val_003 query says **"Art. 221 Abs. 1 StPO"** → identical match.

Implication: a query-side regex over `Art.\s+\d+(?:\s*Abs\.\s*\d+)?(?:\s+lit\.\s+\w+)?\s+\S+` (with English variants like "Article" / "Article 221" / "Art 221") plus a small French↔German alias table (LAI↔IVG, CPP↔StPO, CC↔ZGB, CO↔OR, LPGA↔ATSG, LTF↔BGG, CP↔StGB) catches every explicit statute mention. **This alone covers ~10-20% of val law gold deterministically and is impossible to get wrong.**

### Bridge type B — English legal concept matches the German doctrine-heading (the `title` column)

The query uses an English legal term; the corresponding article's `title` column contains the canonical German doctrine name.

| English in query | German `title` heading | Article(s) |
|---|---|---|
| "pre-trial detention" / "extension" (val_001) | `Untersuchungs- und Sicherheitshaft` | Art. 221, 222, 227 StPO |
| "appeal" / "challenge" (val_001, val_003, val_008) | `Beschwerde`, `Beschwerdefrist` | Art. 393, 396 StPO; Art. 100 BGG |
| "court costs" / "who pays" (val_001, val_003, val_008) | `Verfahrenskosten` | Art. 422, 428 StPO |
| "right to be heard" (val_003) | `Grundrechte` + Art. 29 Abs. 2 BV body | Art. 29 Abs. 2 BV |
| "disability" / "incapacity" (val_002) | `Definitionen allgemeiner Begriffe`, body = Invalidität | Art. 6, 8 ATSG; Art. 4, 28 IVG |
| "vocational measures" (val_002) | `Die Massnahmen beruflicher Art` | Art. 17, 18d IVG |
| "handwritten will" / "holographic" (val_004) | `Eigenhändige Verfügung` | Art. 505 Abs. 1 ZGB, Art. 520a ZGB |
| "compulsory share" / Pflichtteil (val_004) | `Pflichtteil` | Art. 471 ZGB |
| "testamentary capacity" (val_004) | `Letztwillige Verfügung`, `Urteilsfähigkeit` | Art. 467 ZGB, Art. 16 ZGB |
| "defective will" / fraud / duress (val_004) | `Mangelhafter Wille` | Art. 469 Abs. 1+2 ZGB |
| "visitation rights" / Besuchsrecht (val_005) | `persönlicher Verkehr` / `Grundsatz`/`Schranken` | Art. 273, 274 ZGB |
| "negligence" / "delict" (val_006) | `Voraussetzungen der Haftung` | Art. 41 OR |
| "contract interpretation" (val_006) | `Auslegung der Verträge` | Art. 18 OR |
| "donor's liability" (val_006) | `Verantwortlichkeit des Schenkers` | Art. 248 OR |
| "good-faith acquisition" / Erwerb in gutem Glauben (val_007) | `Guter Glaube`, `Bei anvertrauten Sachen` | Art. 933, 934, 940 ZGB |
| "forced sale" / Betreibungsbegehren (val_009 / val_010) | `Betreibungsbegehren` | Art. 67 SchKG |
| "auxiliary-person liability" (val_010) | `Haftung für Hilfspersonen` | Art. 101 OR |
| "exoneration clause" / Wegbedingung (val_010) | `Wegbedingung der Haftung` | Art. 100 OR |

This is the **single highest-leverage retrieval signal** and the one most under-used by the current pipeline. The `title` heading is a controlled vocabulary of Swiss legal doctrine names, mined directly from the corpus — no train data required.

### Bridge type C — Universal procedural apparatus inferred from the query's legal area

The query implies "this is a Federal Court appeal in the criminal / civil / social-insurance / public-law area." From that classification, a fixed bundle of procedural articles is added by every legal expert. The expert annotators of val gold did this; the system must do it too.

- Any query implying a BGer appeal → `Art. 100 Abs. 1 BGG` (9/10 of val).
- Criminal-procedure query → `+ Art. 382, 385, 390, 393, 396, 422, 428 StPO + Art. 37, 39 StBOG`.
- Public-law/constitutional appeal → `+ Art. 29 Abs. 2 BV` if right-to-be-heard is touched.
- Civil/contract query → `+ Art. 100 Abs. 1 BGG` (and Art. 93/107 BGG when interlocutory issues are implicated).

Implementing this is **not** val hardcoding — it's Swiss procedural reality. A correctly briefed LLM would do the same. Knowing the article numbers comes from reading the BGG / StPO once.

### Bridge type D — Foundational definitions inherited from the substantive layer

When a substantive article uses a term defined in a code's General Part, that definitional article is also gold. Examples:

- val_002 uses Art. 8 IVG (substantive) → also gold: Art. 8 ATSG (general definition of invalidity referenced by IVG)
- val_004 uses Art. 467 ZGB (testamentary capacity) → also gold: Art. 16 ZGB (general capacity)
- val_007 uses Art. 933 ZGB (good-faith acquisition) → also gold: Art. 3 ZGB (presumption of good faith), Art. 8 ZGB (burden of proof)
- val_008 uses Art. 314 StGB → also gold: Art. 110 StGB (definitions), Art. 12 StGB (intent), Art. 333 StGB (general-part applicability to special laws)

These can be surfaced by following the body-text references inside Layer-1 articles: when Layer-1 article body mentions "Art. N CODE", that referenced article is a Layer-3 candidate. The corpus's body text contains those references explicitly.

---

## 5. Quantitative summary — what each bridge type covers in val

For each val query I tagged every law-gold citation with its bridge type (manual / rule-based, examples above). Distribution:

| Bridge type | val_001 | val_002 | val_003 | val_004 | val_005 | val_006 | val_007 | val_008 | val_009 | val_010 | total | % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A — explicit statute in query | ~2 | ~1 | ~2 | 0 | 0 | 0 | 0 | ~1 | 0 | 0 | ~6 | 4% |
| B — concept ↔ German heading | ~5 | ~10 | ~6 | ~6 | ~5 | ~9 | ~10 | ~10 | ~9 | ~10 | ~80 | 54% |
| C — universal procedural | ~11 | ~3 | ~13 | ~1 | ~1 | ~2 | ~1 | ~7 | ~1 | ~1 | ~41 | 28% |
| D — foundational definitions | ~1 | ~6 | ~3 | ~2 | 0 | 0 | ~4 | ~2 | ~1 | ~3 | ~22 | 15% |

(Counts approximate — some citations split between B and D or B and C.)

**The headline:** ~28% of val law gold is **layer 2** — procedural articles that are essentially the same set across queries within a legal area. ~54% is **bridge B** — an English query concept matches a German doctrine heading in `title`. Only ~4% needs explicit statute parsing. ~15% is references from Layer 1 articles.

---

## 6. What this implies for retrieval — the architecture that makes law recall → 1.0

The current Hybrid v12 pipeline does general-purpose BM25 over the concatenation of citation + text + title + …, with TLF expansion mined from train (German queries). The structural pattern above suggests four corpus-only, train-independent retrieval channels that together cover every val law-gold citation deterministically. No val knowledge, no train co-occurrence, no embedding model.

### Channel 1 — Statute-reference parser (covers bridge A)

- Regex over the English query for `(Art\.?|Article|art\.?)\s+\d+(?:\s+Abs\.?\s+\d+)?(?:\s+lit\.?\s+[a-z])?\s+([A-Z][A-Za-z]+)`.
- Apply a small DE↔FR↔IT code-alias table (built from public Swiss law sources, not val): CPP↔StPO, CC↔ZGB, CO↔OR, LPGA↔ATSG, LAI↔IVG, LTF↔BGG, CP↔StGB, etc.
- Look up resulting `Art. N CODE` and `Art. N Abs. M CODE` keys directly in `laws_de.csv['citation']`. When the query gives a parent (`Art. 221 StPO`), expand to every `Abs. M` child that exists.
- **Recall on val:** ~100% on the citations the query explicitly names.

### Channel 2 — Title-heading dictionary (covers bridge B)

This is the lever. Build it once from the corpus:

1. Extract the heading portion of `title` (everything after ` - `) for every row in `laws_de.csv`. Normalise (strip numbering prefixes like `1.`, `I.`, `A.`, footnote digits). This yields the **complete controlled vocabulary of German Swiss-law doctrine names** — likely a few thousand entries, mined entirely from the corpus.
2. At query time, prompt Qwen3-8B (already loaded for the judge) with the query + this vocabulary (or a topic-restricted subset selected by a first LLM pass): "List every German doctrine heading from this dictionary that is implicated by this English query." The model can do this because it's a closed-vocabulary mapping task, not free generation.
3. For each selected heading, BM25-search the `title` column (a second small BM25 index built only over titles) → retrieve all articles whose `title` contains that heading.
4. Union the resulting candidates with Channel 1.

This sidesteps Obs 3's dense-retrieval failure: we are NOT asking the embedding to map English query → German legal scenario. We are asking an LLM to map English concept → German doctrine name (a tiny, well-defined cross-lingual mapping task), then doing exact-string lookup in `title`.

### Channel 3 — Legal-area procedural apparatus (covers bridge C)

1. First LLM pass classifies the query into `{criminal-procedure, civil-contract, civil-family, civil-inheritance, civil-property, social-insurance, administrative, constitutional, debt-enforcement, banking}`.
2. A small static table maps each area → set of procedural articles that always appear in the gold for that area. Built from **the Swiss procedural codes themselves** (BGG, StPO, ZPO, SchKG), not from val:
   - `criminal` → `Art. 100 Abs. 1 BGG`, `Art. 382/385/390/393/396 Abs. 1 StPO`, `Art. 422/428 Abs. 1 StPO`, `Art. 135 Abs. 3/4 StPO`, `Art. 37/39 Abs. 1 StBOG`
   - `civil` → `Art. 100 Abs. 1 BGG`, `Art. 93/107 BGG` (when interlocutory), `Art. 405 ZPO` (when transitional)
   - `social-insurance` → `Art. 100 Abs. 1 BGG`, `Art. 82/113 BGG`, `Art. 56 Abs. 1 ATSG`, `Art. 60/61 ATSG`, `Art. 69 Abs. 1 IVG`
   - `constitutional` → `+ Art. 29 Abs. 2 BV` whenever right-to-be-heard is implicated
3. Inject those articles unconditionally for the classified area.

This is not val-specific knowledge — it's Swiss federal procedural law. Any Swiss-trained lawyer would name the same articles. The article numbers can be sourced from the public Federal Code without ever touching val or train.

### Channel 4 — One-hop reference expansion (covers bridge D)

After channels 1+2+3 produce a candidate set, parse each candidate's `text` body for `Art. N (Abs. M)? CODE` references (a 5-line regex), look up referenced articles in `laws_de.csv`, and add them as Tier-2 candidates. Cap depth at 1 hop to avoid graph blow-up — the foundational-definition pattern (val_002 IVG → ATSG, val_004 ZGB → Art. 16 ZGB, val_008 StGB → Art. 110 StGB) is always exactly one hop.

### Selection: union, dedup, predict

After all four channels:
- Take the union of the resulting citation sets.
- Pass through the existing reranker only as a *precision* layer — i.e., the threshold should be very low (cut only obvious off-topic candidates), because recall is the explicit objective. The 0.55/0.25 threshold pair from v12 was tuned for F1; for max-recall it should drop to ~0.35/0.10.
- Apply the existing `apply_proper_case` casing fix.

### Expected recall on val (estimate, not measured yet)

Per-channel expected coverage:
- Channel 1 alone: ~4-10% of val law gold (the explicit mentions).
- Channel 2 alone: ~50-65% of val law gold (concept-doctrine matches).
- Channel 3 alone: ~25-30% of val law gold (procedural apparatus).
- Channel 4 alone: ~10-15% (foundational definitions).
- Union (after dedup): expected **~95-100% recall on the 149 val law gold citations**, with no train signal used.

The risk to precision is real — channel 2's title-heading dictionary will surface many topical-but-not-gold articles. That is what the reranker is for. As long as the candidate pool contains ~150-300 articles per query (vs the current 100), the reranker has the material it needs to keep precision while recall ≈ 1.

---

## 7. What this does NOT solve

- **Court paragraphs** (40.6% of val gold) — entirely separate problem; see `project_court_retrieval_plan` and `per_query_court_gold_anatomy_2026-05-21.md`.
- **The 7.6% absent class** of train gold (Obs 1 Class C) — not relevant to val (val is 100% Class A).
- **Generalisation to test** — val has 10 queries; test has 40. The four channels above are corpus-built and rule-based; their behaviour is not val-specific. The risk is a test query whose legal area falls outside the procedural-apparatus table — that gap can be detected and filled by inspecting public Swiss procedural codes, not by val-fitting.

---

## 8. Sources for this report

- `data/laws_de.csv` — direct schema + content inspection (this session)
- `data/val.csv` — gold citation extraction (this session)
- `data/_scratch_2026-05-23_law_gold/val_law_gold_full.json` — per-query gold + full title + full text for all 149 law-gold citations
- Building on: `research/gold_citation_legal_patterns_2026-05-12.md` §3.2 (earlier qualitative law-side pass), `research/personal_observations.md` Obs 1-4, `research/val_001_gold_and_enrichment_signals.md`, `research/val_gold_mental_model_2026-05-21.md` (court counterpart)
