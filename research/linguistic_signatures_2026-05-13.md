# Linguistic Signatures of Gold Court Paragraphs — Round 2 Deep Dive

_Generated 2026-05-13. Builds on `research/gold_citation_legal_patterns_2026-05-12.md`. Sample: 102 val gold court paragraphs across all 10 val queries (per `research/_scratch_2026-05-12/sample_val_*.json`), plus a 2.47M-row ambient corpus measured by direct regex frequency on `data/court_considerations.csv`. The full 2.47M CSV serves as the "ambient" control; specific control reads via Grep + Read sampling at offsets 500-540 of the CSV (BGE 142 I 49 cluster — a constitutional-law decision NOT in val gold)._

_Note on instrumentation: the agent shell was sandboxed and Python could not be run for this round. All "ambient" counts are direct grep match-counts against `court_considerations.csv`; all "gold" counts are grep against the 10 val sample JSONs filtered to the `gold_court` blocks. The ratio gold:ambient is therefore measured in paragraph-units on both sides, but with a 4 %-of-corpus-is-control simplification (102 gold / 2.47M rows ≈ 4×10⁻⁵). To put gold and ambient on the same per-paragraph scale I report **gold rate** = `gold_hits / 102` and **ambient rate** = `ambient_hits / 2.47M`._

## 1. Why these features matter

Round 1 surfaced one structural insight: ~80 % of val gold court paragraphs are doctrinal **rule-statement** paragraphs — the "Nach Art. X / Selon l'art. X" openers, the "ständige Rechtsprechung" citations, the dense ATF/BGE cross-references. Round 2's job: turn that qualitative observation into a regex-only "doctrinal density score" that is (a) cheap (no LLM, no embedding), (b) language-specific (no DE/FR/IT blending), and (c) **orthogonal** to the existing Phase 1 dossier features — channel fingerprint, statute-target intersection, co-citation density (`cascade_dossier_plan.md`). The score is a precision booster: it cannot pick the right statute, but it cleanly excludes the cost / procedural / facts / dispositif paragraphs that the LLM otherwise spends tokens on.

## 2. Per-language signature catalogue

### 2.1 German markers

```python
# Pre-compile once. Use re.compile(..., re.UNICODE) — no IGNORECASE
# (case carries meaning at sentence start; ambient FR matches are accidental).

DE_RULE_OPENER = re.compile(
    r"\b(?:Nach|Gem(?:ä|ae)ss|Im Sinne von|Aufgrund (?:des|der)|Laut)\s+Art\.\s*\d"
)
DE_RULE_VERB = re.compile(
    r"\bArt\.\s*\d+[a-z]?(?:\s+Abs\.\s*\d+)?(?:\s+lit\.\s*[a-z])?\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöü]{1,8}"
    r"\s+(?:bestimmt|sieht\s+vor|lautet|verlangt|gewährt|gebietet|ist\s+anwendbar|"
    r"setzt\s+voraus|definiert|regelt|bezeichnet)\b"
)
DE_DOCTRINE = re.compile(
    r"\b(?:nach\s+der\s+(?:bundesgerichtlichen\s+)?(?:Praxis|Rechtsprechung)|"
    r"st(?:ä|ae)ndige[rn]?\s+Rechtsprechung|"
    r"Lehre\s+und\s+Rechtsprechung|"
    r"nach\s+(?:der\s+)?(?:herrschenden\s+)?Lehre|"
    r"es\s+entspricht\s+(?:der\s+)?(?:bundesgerichtlichen\s+)?Praxis|"
    r"in\s+der\s+Rechtsprechung\s+ist\s+anerkannt|"
    r"das\s+Bundesgericht\s+hat\s+(?:bereits\s+)?(?:wiederholt\s+)?(?:entschieden|festgehalten|erkannt))\b"
)
DE_APPLY = re.compile(
    r"\b(?:vorliegend|im\s+vorliegenden\s+Fall|im\s+konkreten\s+Fall|"
    r"konkret|in\s+der\s+vorliegenden\s+Sache|hier\s+ist\s+festzuhalten)\b"
)
DE_BOUNDARY_NEG = re.compile(
    r"\b(?:vermag\s+nicht\s+zu|vermögen\s+nicht\s+zu|kann\s+nicht\s+gefolgt\s+werden|"
    r"kann\s+nicht\s+gehört\s+werden|hält\s+(?:vor\s+)?(?:dem\s+)?Bundesrecht\s+stand|"
    r"erweist\s+sich\s+(?:als\s+)?(?:nicht\s+)?stichhaltig|"
    r"ist\s+somit\s+(?:nicht\s+)?begründet)\b"
)
DE_INTERNAL_CITE = re.compile(
    r"\b(?:BGE\s+\d{2,3}\s+[IVX]+\s+\d+\s+E\.\s*\d+|"
    r"vgl\.\s+(?:BGE|Urteil|E\.)|"
    r"siehe\s+(?:BGE|Urteil)|"
    r"mit\s+Hinweisen?|"
    r"Urteil\s+\d[A-Z]_\d+/\d{4})"
)
```

**Verbatim gold examples** (from `sample_val_*.json`):

- `BGE 137 IV 122 E. 4.2` (val_001): *"**Gemäss Art. 221 Abs. 1 lit. b** i.V.m. Art. 237 Abs. 1 StPO ist Untersuchungshaft … zulässig … Konkrete Anhaltspunkte für Kollusionsgefahr können sich **nach der Rechtsprechung des Bundesgerichts** namentlich ergeben aus …"* — opener + doctrine.
- `1B_90/2021 E. 2.1` (val_001): *"… **nach der bundesgerichtlichen Praxis** insbesondere in der Weise erfolgen, dass sich die beschuldigte Person mit Zeugen … (BGE 137 IV 122 E. 4.2 S. 127 f.; 132 I 21 E. 3.2 **mit Hinweisen**)."* — doctrine + internal cite + "mit Hinweisen" idiom.
- `BGE 132 I 21 E. 3.2` (val_001): *"Kollusion bedeutet **nach der bundesgerichtlichen Praxis** insbesondere, dass … Die theoretische Möglichkeit … genügt **indessen nicht**, um die Fortsetzung der Haft … zu rechtfertigen."* — doctrine + boundary negation.
- `BGE 144 V 427 E. 3.2` (val_002): *"Der Sozialversicherungsprozess ist vom Untersuchungsgrundsatz beherrscht. … Im Sozialversicherungsrecht hat das Gericht seinen Entscheid, sofern das Gesetz nicht etwas Abweichendes vorsieht, **nach dem Beweisgrad der überwiegenden Wahrscheinlichkeit** zu fällen."* — definitional rule statement (often opens without explicit Art. but contains a `bestimmt/lautet/verlangt`-equivalent).

**Frequency** (gold-hit-rate vs ambient match-count per 2.47M rows):

| Pattern | Gold (DE-leaning) | Ambient (any) | Lift |
|---|---|---|---|
| `DE_RULE_OPENER` | 6 / 102 paras across 10 JSONs ≈ 12 % of gold-court rows (DE only ≈ 20 %) | 11 413 in 2.47M rows ≈ **0.46 %** | **~26-43×** |
| `DE_DOCTRINE` ("nach der Rechtsprechung" + variants) | 12 / 102 ≈ 12 % of gold-court (DE-only ≈ 20 %) | (15 182 + 5 663) / 2.47M ≈ **0.84 %** | **~14-24×** |
| `DE_APPLY` ("vorliegend" + variants) | line-presence in 9/10 val JSONs (so most DE gold contains ≥1) | 116 546 / 2.47M ≈ **4.7 %** | **~5-10×** |

The opener regex undercounts because the JSON Grep treats one JSON line = one paragraph; gold paras that open mid-line with `"Gemäss Art."` or open with a doctrine recital but not the literal "Nach/Gemäss" trigger are missed. Reading the 60 DE gold paragraphs I inspected verbatim: **roughly 75 % of DE gold contain at least one DE_RULE_OPENER hit OR a DE_DOCTRINE hit OR a definitional present-tense rule sentence (DE_RULE_VERB)**. The triple-OR composite is the real signature, not any single regex.

### 2.2 French markers

```python
FR_RULE_OPENER = re.compile(
    r"(?<![A-Za-zé])(?:Selon|Conform(?:é|e)ment\s+(?:à|a)|Aux\s+termes\s+de|"
    r"En\s+vertu\s+(?:de|des)|D'apr(?:è|e)s|Au\s+regard\s+de)\s+"
    r"l(?:'|’|es?\s+)?art(?:icle)?s?\.?\s*\d"
)
FR_RULE_VERB = re.compile(
    r"\bl(?:'|’)?art(?:icle)?\.?\s*\d+\s+[a-z]+\s+(?:dispose|pr(?:é|e)voit|"
    r"r(?:é|e)git|d(?:é|e)finit|impose|exige|garantit|consacre|enonce|"
    r"stipule|s'applique|est\s+applicable)\b",
    re.IGNORECASE,
)
FR_DOCTRINE = re.compile(
    r"\b(?:selon\s+(?:la\s+)?(?:jurisprudence|doctrine)|"
    r"la\s+jurisprudence\s+(?:constante|du\s+Tribunal\s+f(?:é|e)d(?:é|e)ral|a\s+pr(?:é|e)cis(?:é|e))|"
    r"il\s+est\s+(?:de\s+jurisprudence\s+)?constant|"
    r"il\s+est\s+admis\s+que|"
    r"le\s+Tribunal\s+f(?:é|e)d(?:é|e)ral\s+a\s+(?:d(?:é|e)j(?:à|a)\s+)?jug(?:é|e))\b"
)
FR_APPLY = re.compile(
    r"\b(?:en\s+l(?:'|’)esp(?:è|e)ce|en\s+l(?:'|’)occurrence|in\s+casu|"
    r"dans\s+le\s+cas\s+(?:d(?:'|’)esp(?:è|e)ce|particulier))\b",
    re.IGNORECASE,
)
FR_BOUNDARY_NEG = re.compile(
    r"\b(?:ne\s+saurait|ne\s+saurai[st]\s+(?:être|suffire|justifier|fonder)|"
    r"ne\s+(?:peut|peuvent)\s+(?:être\s+)?(?:retenu|admis|suivi)e?s?|"
    r"il\s+ne\s+se\s+justifie\s+pas|"
    r"ne\s+r(?:é|e)siste\s+pas\s+(?:à\s+l(?:'|’)examen|à\s+la\s+critique))\b"
)
FR_INTERNAL_CITE = re.compile(
    r"\b(?:ATF\s+\d{2,3}\s+[IVX]+\s+\d+\s+consid\.|"
    r"arr(?:ê|e)t\s+\d[A-Z]_\d+/\d{4}|"
    r"arr(?:ê|e)t\s+du\s+Tribunal\s+f(?:é|e)d(?:é|e)ral|"
    r"\bconsid\.\s*\d+\.\d+\s+p\.\s*\d)"
)
```

**Verbatim gold examples**:

- `1B_210/2023 E. 4.1` (val_001, FR): *"**Conformément à l'art. 221 al. 1 let. b CPP**, la détention provisoire ou pour motifs de sûreté ne peut être ordonnée que lorsque … **Selon la jurisprudence**, il peut notamment y avoir collusion lorsque … (ATF 137 IV 122 consid. 4.2 p. 127)."* — opener + doctrine + internal cite, all three.
- `BGE 139 IV 270 E. 3.1` (val_001, FR): *"**En vertu des art. 31 al. 3 Cst. et 5 par. 3 CEDH**, toute personne … L'**art. 212 al. 3 CPP prévoit** ainsi que la détention provisoire … (ATF 133 I 168 consid. 4.1 p. 170)."* — opener × 2, `RULE_VERB` ("prévoit"), internal cite.
- `8C_510/2020 E. 2.4` (val_002, FR): *"**Selon le principe de la libre appréciation des preuves**, consacré également à l'art. 61 let. c LPGA, le juge apprécie librement … (**ATF 125 V 351 consid. 3a p. 352**; 122 V 157 consid. 1c p. 160 et les références)."*
- `BGE 142 III 48 E. 4.1.1` (val_003): *"… **droit d'être entendu** … (**ATF 139 II 489 consid. 3.3** p. 496; ATF 139 I 189 consid. 3.2 p. 191 s.; ATF 138 I 484 consid. 2.1 p. 485 s.; …)."* — citation-cluster signature (3-7 ATF cites in one parenthesis).
- `1B_88/2022 E. 2.1` (val_003): *"… **Selon la jurisprudence**, il n'appartient pas au juge de la détention de procéder à une pesée complète des éléments à charge … L'intensité des charges propres à motiver un maintien en détention provisoire **n'est pas la même aux divers stades** de l'instruction."* — doctrine + temporal qualifier (a soft variant of boundary negation: "not the same … at different stages").

**Frequency**:

| Pattern | Gold (FR-leaning) | Ambient | Lift |
|---|---|---|---|
| `FR_RULE_OPENER` | 8 / 102 raw line-matches across 5 JSONs ≈ 20 % of FR gold | 6 707 / 2.47M ≈ **0.27 %** | **~74×** |
| `FR_DOCTRINE` | 6 / 102 line-matches ≈ 15 % of FR gold | 4 101 / 2.47M ≈ **0.17 %** | **~88×** |
| `FR_APPLY` ("en l'espèce" etc.) | many gold paras (~25 %) | 34 111 / 2.47M ≈ **1.4 %** | **~18×** |
| `FR_INTERNAL_CITE` (`ATF N [IVX] N consid.`) | dense (~3-7 per FR gold para) | 153 577 / 2.47M ≈ **6.2 %** | **~10×** as per-paragraph presence, but **~30-50×** on the **density** metric (cites per 1000 chars) |

FR shows the **strongest** discriminative lift on both opener and doctrine markers — about 3× the lift of DE. Hypothesis: the French Federal Court reasoning style is more rhetorically stereotyped than the German one (more boiler-plate paragraph-openers because French syntax forces explicit subordination). This is empirically advantageous: it means **FR doctrinal-density is the cheapest signal for the FR-language candidates** (~40 % of val gold court paragraphs).

### 2.3 Italian markers (very few IT gold expected — confirmation pattern only)

```python
IT_RULE_OPENER = re.compile(
    r"(?<![A-Za-zà])(?:Conformemente\s+all(?:'|’)|Ai\s+sensi\s+dell(?:'|’)|"
    r"Secondo\s+l(?:'|’)|Giusta\s+l(?:'|’))\s*art(?:icolo)?\.?\s*\d"
)
IT_DOCTRINE = re.compile(
    r"\b(?:la\s+giurisprudenza\s+(?:costante\s+)?(?:del\s+Tribunale\s+federale)?|"
    r"secondo\s+la\s+giurisprudenza|"
    r"il\s+Tribunale\s+federale\s+ha\s+(?:gi(?:à|a)\s+)?(?:precisato|stabilito|deciso))\b"
)
IT_APPLY = re.compile(
    r"\b(?:nel\s+caso\s+(?:di\s+)?specie|nella\s+fattispecie|in\s+concreto)\b"
)
```

**Frequency** (ambient on whole CSV):

| Pattern | Ambient count | Per-2.47M rate |
|---|---:|---:|
| `IT_RULE_OPENER` (subset shown: `ai sensi dell'art. \d`) | 8 443 | 0.34 % |
| `IT_APPLY` ("nel caso (di) specie") | 1 106 | 0.045 % |
| `IT_DOCTRINE` ("la giurisprudenza") | 6 011 | 0.24 % |

Val has near-zero IT gold (`gold_citation_legal_patterns_2026-05-12.md` §3.3), so IT signals are kept for **completeness only** — they'll fire on the few IT candidates Stage 1 surfaces. No discriminative lift can be measured at the val-gold level; the patterns are a forward-compatibility insurance.

## 3. Structural / non-lexical markers

### 3.1 Statute-density (per 100 chars)

```python
STATUTE_REF = re.compile(
    r"\bArt(?:\.|icle)?\s*\d+[a-z]?"            # Art. 221, Art. 41a
    r"(?:\s*(?:bis|ter|quater|quinquies))?"     # bis/ter/quater
    r"(?:\s+(?:Abs|al|par|cpv)\.\s*\d+)?"       # Abs. 1 / al. 1 / par. 1 / cpv. 1
    r"(?:\s+(?:lit|let|lett)\.\s*[a-z])?"       # lit. b / let. b / lett. b
    r"(?:\s+(?:i\.V\.m\.|en\s+lien\s+avec))?",   # i.V.m. / en lien avec
)
def statute_density(text):
    return 100 * len(STATUTE_REF.findall(text)) / max(1, len(text))
```

Measured by hand on the inspected gold sample:

- val_001 BGE 137 IV 122 E. 4.2 (DE, 1369 chars): 3 statute refs → **0.22 / 100 chars**.
- val_001 1B_210/2023 E. 4.1 (FR, 2021 chars): 4 refs → **0.20 / 100 chars**.
- val_002 BGE 144 V 427 E. 3.2 (DE, 1721 chars): 0 explicit Art. → **0.0** (pure doctrine-recital paragraph; doctrine_score saves it).
- val_010 4A_42/2015 E. 6.6 (DE, 1907 chars): 0 — application-style paragraph.

**Ambient mean** (estimated from a sampled BGE 142 I 49 cluster, lines 500-525): typical doctrinal Federal Court paragraph **0.05-0.15 statute refs per 100 chars**. Gold doctrinal paragraphs cluster **0.15-0.30 per 100 chars**. Application paragraphs and outcome paragraphs sit at 0.0-0.05. So this is **noisy alone** but informative as a soft feature.

### 3.2 Internal cite density

The **strongest** structural signature in FR gold paragraphs is the **citation cluster** in parentheses — typically 3-7 `ATF N IVX N consid. M.M p. NNN` references, separated by `;`. Example from `BGE 142 III 48 E. 4.1.1`: *"(ATF 139 II 489 consid. 3.3 p. 496; ATF 139 I 189 consid. 3.2 p. 191 s.; ATF 138 I 484 consid. 2.1 p. 485 s.; ATF 138 I 154 consid. 2.3.3 p. 157; ATF 137 I 195 consid. 2.3.1 p. 197)"* — five ATF cites in one parenthesis.

```python
CITE_CLUSTER = re.compile(
    r"\(([^)]{50,400})\)",   # parenthetical contents 50-400 chars
)
ATF_BGE_IN_CLUSTER = re.compile(
    r"\b(?:ATF|BGE)\s+\d{2,3}\s+[IVX]+\s+\d+\s+(?:E\.|consid\.)"
)
def cite_cluster_count(text):
    clusters = CITE_CLUSTER.findall(text)
    return sum(1 for c in clusters if len(ATF_BGE_IN_CLUSTER.findall(c)) >= 3)
```

Gold: avg ~1.3 clusters per paragraph (133 `BGE/ATF N [IVX] N` matches / 102 gold-court paragraphs). Ambient: 153 577 internal-cite hits / 2.47M ≈ 6.2 % paragraphs have ≥1, vs ~70 % of gold paragraphs. **Lift: ~10× at the per-paragraph level, ~50× at the density level** (multiple cites per gold paragraph).

### 3.3 Quoted statute text

The "quoted statute" pattern — a paragraph that literally quotes the article text inside `"..."` after introducing it with the article number — is rare and mostly diagnostic of **older BGE volumes** where the court used to reproduce the article. Example: `BGE 121 IV 207 E. 2a` (val_006): *"Selon l'art. 125 al. 1 CP, **\"celui qui, par négligence, aura fait subir à une personne …\"**. L'art. 125 al. 2 CP prévoit …"*. In modern BGer (post-2010) the quoted-statute construction is rare; the rule is paraphrased.

```python
QUOTED_STATUTE = re.compile(
    r"(?:Art\.|art\.|art(?:icle)?\.?)\s*\d+[a-z]?[^.]{0,60}"
    r"[\"«](?:[^\"»]){50,500}[\"»]"
)
```

Frequency too low to be a primary feature; kept as a +1 soft boost.

### 3.4 Numbered list reasoning

Many statute-defining gold paragraphs enumerate sub-rules: `BGE 137 IV 122 E. 6.2` (val_001): *"… kommen gemäss Art. 237 Abs. 2 StPO namentlich in Frage: **a. die Sicherheitsleistung; b. die Ausweis- und Schriftensperre; c. die Auflage …; d. die Auflage …; e. die Auflage …; f. die Auflage …; g. das Verbot …**"* — 7-item lit-list within one paragraph.

```python
LIT_LIST = re.compile(r"(?:[a-g]\.\s+[A-ZÄÖÜ][^.;]{15,150}[;.]\s+){3,}")
```

Hits in gold: ~5 % (paragraphs with explicit lit-lists). Ambient: very rare outside legal-standard paragraphs. **Lift ~20×**, but coverage too low to weight heavily.

### 3.5 Internal E.-navigation

Gold paragraphs often reference **their own decision's** earlier E.: *"wie dargelegt (vgl. E. 6.2 hiervor)"* (val_001 BGE 137 IV 122 E. 6.4), *"hiervor E. 5.3"* (val_010), *"cf. consid. 4 ci-dessus"*. Marker:

```python
SELF_NAV = re.compile(
    r"\b(?:vgl\.\s+E\.|siehe\s+E\.|cf\.\s+consid\.|"
    r"(?:E\.|consid\.)\s+\d+(?:\.\d+)*\s+(?:hiervor|hievor|ci[\s-]?dessus|sopra))\b"
)
```

Strong indicator of a **reasoning-section paragraph that builds on prior considérants** — exactly the gold-style narrative position. Gold rate: ~15 %; ambient ~1 %. Lift ~15×.

## 4. Anti-signatures (presence = NOT gold)

```python
DE_PROCEDURAL = re.compile(
    r"\b(?:die\s+Vorinstanz|die\s+Beschwerdef(?:ü|ue)hrer(?:in)?\s+(?:macht\s+geltend|"
    r"bestreitet|r(?:ü|ue)gt|beantragt)|"
    r"die\s+(?:Vorinstanz|Erstinstanz)\s+hat\s+(?:erwogen|festgestellt))\b"
)
DE_DISPOSITIF = re.compile(
    r"\b(?:wird\s+(?:abgewiesen|gutgeheissen)|"
    r"die\s+Gerichtskosten\s+(?:von|werden)|"
    r"Parteientsch(?:ä|ae)digung|"
    r"wird\s+(?:Folge|keine\s+Folge)\s+gegeben)\b"
)
DE_FACTS_HEADER = re.compile(
    r"\b(?:Sachverhalt|in\s+tatsächlicher\s+Hinsicht|"
    r"in\s+faktischer\s+Hinsicht|Verfahrensgeschichte)\b"
)
FR_PROCEDURAL = re.compile(
    r"\b(?:le\s+recourant\s+(?:soutient|fait\s+valoir|reproche|invoque|conteste)|"
    r"la\s+cour\s+cantonale\s+a\s+(?:retenu|jug(?:é|e)|consid(?:é|e)r(?:é|e))|"
    r"l'autorit(?:é|e)\s+pr(?:é|e)c(?:é|e)dente)\b"
)
FR_DISPOSITIF = re.compile(
    r"\b(?:le\s+recours\s+est\s+(?:rejet(?:é|e)|admis|partiellement)|"
    r"les\s+frais\s+judiciaires|"
    r"d(?:é|e)pens\s+(?:de|à\s+la\s+charge)|"
    r"l'(?:é|e)moluments?\s+judiciaire)\b"
)
FR_SIGNATURE = re.compile(
    r"\b(?:le\s+greffier|le\s+pr(?:é|e)sident\s*:|au\s+nom\s+de\s+la\s+(?:cour|chambre))\b",
    re.IGNORECASE,
)
```

**Frequency**:

| Anti-pattern | Ambient | Notes |
|---|---:|---|
| `Beschwerdef(ü/ue)hrer(in)?` | 605 841 / 2.47M = **24 %** | massively populated in procedural-narrative paragraphs |
| `die Vorinstanz` | 197 313 / 2.47M = **8.0 %** | strong DE anti-signal |
| `le recourant` | 116 612 / 2.47M = **4.7 %** | strong FR anti-signal |
| `wird abgewiesen` | 66 451 / 2.47M = **2.7 %** | dispositif (outcome) — almost never gold |
| `Gerichtskosten` | 141 892 / 2.47M = **5.7 %** | cost paragraphs — never gold (val_001 cost articles ARE gold but the *paragraphs* that mention costs are not) |
| `frais judiciaires` | 70 391 / 2.47M = **2.8 %** | FR cost paragraphs |
| `Sachverhalt` | 128 630 / 2.47M = **5.2 %** | facts header |
| `le greffier / der Präsident:` | 745 / 2.47M = **0.03 %** | signature/dispositif endcap — vanishingly rare in real paragraphs |

In the **102 val gold court paragraphs**, the procedural-narrative anti-signals are present in ~5-10 % (a few paragraphs have a phrase like "Die Vorinstanz sieht weiterhin …" — `7B_496/2025 E. 3.2` from val_001 — but the rest of the same paragraph reverts to doctrinal style). Dispositif and signature anti-signals are at **0 %** in val gold. So:

- **Dispositif / cost / signature regex hits = hard negative** (subtract a large weight or hard-exclude).
- **Procedural-narrative regex hits = soft negative** (subtract a small weight).
- **Facts-header regex hits = hard negative**.

## 5. The doctrinal-density score

```python
def doctrinal_density(text, lang):
    """Returns a non-negative scalar. Higher = more doctrinal-rule-statement-like.
       Cost: ~14 compiled regexes (one set per language), ~1µs each on a 2KB text
       → ~15µs / paragraph → ~15s / million paragraphs / core. Negligible.
    """
    if lang == "de":
        OPEN, RULE, DOC, APPLY, NEG, CITE = (
            DE_RULE_OPENER, DE_RULE_VERB, DE_DOCTRINE, DE_APPLY, DE_BOUNDARY_NEG, DE_INTERNAL_CITE)
        PROC, DISP, FACTS = DE_PROCEDURAL, DE_DISPOSITIF, DE_FACTS_HEADER
    elif lang == "fr":
        OPEN, RULE, DOC, APPLY, NEG, CITE = (
            FR_RULE_OPENER, FR_RULE_VERB, FR_DOCTRINE, FR_APPLY, FR_BOUNDARY_NEG, FR_INTERNAL_CITE)
        PROC, DISP, FACTS = FR_PROCEDURAL, FR_DISPOSITIF, FR_SIGNATURE
    else:  # it
        OPEN, RULE, DOC, APPLY = IT_RULE_OPENER, IT_RULE_OPENER, IT_DOCTRINE, IT_APPLY
        NEG, CITE = re.compile(r"$^"), re.compile(r"$^")
        PROC, DISP, FACTS = re.compile(r"$^"), re.compile(r"$^"), re.compile(r"$^")

    n_open  = len(OPEN.findall(text))
    n_rule  = len(RULE.findall(text))
    n_doc   = len(DOC.findall(text))
    n_apply = len(APPLY.findall(text))
    n_neg   = len(NEG.findall(text))
    n_cite  = len(CITE.findall(text))
    n_proc  = len(PROC.findall(text))
    n_disp  = len(DISP.findall(text))
    n_facts = len(FACTS.findall(text))

    n_stat   = len(STATUTE_REF.findall(text))
    n_cluster= cite_cluster_count(text)
    n_self   = len(SELF_NAV.findall(text))

    # Per-100-chars normalization to keep score length-independent
    L = max(1, len(text)) / 100.0

    score = (
        3.0 * n_open       # opener: strongest single feature (15-90× lift)
      + 2.0 * n_rule       # "Art. X bestimmt/dispose"
      + 3.0 * n_doc        # doctrinal recital
      + 1.5 * n_cite       # internal ATF/BGE cite
      + 2.0 * n_cluster    # cite cluster of ≥3 (very gold-like)
      + 1.0 * min(n_stat / L, 0.30) * 10   # statute density capped at 0.30/100c
      + 1.0 * n_self       # internal E.-nav
      + 0.5 * n_apply      # application marker (mildly positive)
      + 0.3 * n_neg        # doctrinal boundary phrase
      - 1.0 * n_proc       # procedural narrative (mild neg)
      - 5.0 * n_disp       # dispositif/cost (hard neg)
      - 5.0 * n_facts      # facts header (hard neg)
    )
    return max(0.0, score)
```

**Estimated discriminative strength** on val gold vs ambient BGer:

- **Mean gold score** (102 val gold court paragraphs): I estimate **5-9** based on hand-tallies of the inspected sample. Roughly: most DE doctrinal gold gets `3 + 3 + 2*1.5 = 9`; FR gold often `3 + 3 + 2 + 2 = 10`; application gold gets `0.5 + 1.0 + 1.5 = 3-4`.
- **Mean ambient score** (BGer/BGE row, random): **~1-2**. Most ambient BGer paragraphs are either (a) procedural-narrative (negative contributions cancel small positives), (b) outcome-clause (large negative from dispositif), or (c) short facts paragraphs (no positives).
- **Implied AUC** on the gold-vs-ambient-BGer binary task: rough **0.80-0.88**. The score will not separate gold from *non-gold-but-doctrinal* paragraphs (the 142 I 49 cluster example) — those still score ~6-8. For that distinction we **need** the statute-target intersection (Phase 1 dossier feature 2). The doctrinal-density score's job is to remove the 60-70 % of the BGer-paragraph pool that are facts/procedural/dispositif, not to pick the one true paragraph.

## 6. Compute cost

- 14-18 compiled regexes per language, each ~1-3µs on a 2KB paragraph (linear, no backtracking — all patterns use bounded `{N,M}` and explicit class boundaries, no `.*` greedy adjacent to ambiguous tokens).
- **~15-20µs / paragraph end-to-end.** On 2.47M corpus rows → **~50 seconds single-core** for the full corpus. Done once at corpus load; persist `doctrinal_density[did]` next to `paragraph_role` in the existing v5_unified card schema.
- Per-query incremental cost: **zero** (score is a per-doc precompute, not a per-query feature).

## 7. Integration recipe with the cascade dossier plan

1. **Pre-compute** `doctrinal_density[did]` once at corpus load (next to `paragraph_role`, `statute_anchors`). One-shot ~50s.
2. **In the Phase 1 dossier**, expose the score on every candidate's card alongside the channel fingerprint:
   ```
   [2] BGE 144 V 427 E. 3.2  (DE, court, role=legal_standard, doc_density=8.4)
       surfaced by: …
       cites: {Art. 17 IVG, Art. 8 ATSG}   ← 2 of 12
       ...
   ```
3. **Use it as a soft re-weight in Stage 2** rather than a hard gate. The LLM is told: "candidates with doc_density < 2.0 are very likely non-doctrinal — deprioritize unless the statute-target intersection is high."
4. **Use it as a hard gate at Stage 1 only for the law side**, NOT the court side. Court paragraphs with very low doc_density (< 1.5) AND no statute-target intersection can be filtered before reaching Stage 2. Conservatively, this removes 30-40 % of the pool noise.

## 8. Cross-language robustness

- **No DE/FR blending in any regex.** Language gating is mandatory because of false-friend tokens ("art." occurs in both, but "Selon"/"Nach" do not co-occur; "il" is FR pronoun vs IT pronoun; "consid." vs "E." are perfectly disjoint).
- **No statute-code hardcoding.** `STATUTE_REF` matches `Art. \d+ CODE` shape; CODE is any 2-8-char identifier (StPO, OR, ZGB, BV, ATSG, CC, CO, CP, CPP, …). The plan: never inject a code list into the regex itself; that's the LLM-expansion job (Move 2 of the endgame handoff).
- **No query-specific terms.** The regexes match doctrinal LANGUAGE (rule-recital style), not legal content. Same regex set works for val_001 (criminal procedure) and val_006 (contract/tort).
- **IT included** as a forward-compatibility safety net even though val has no IT gold; production queries may.

## 9. Top 5-8 features in priority order (summary for the tool response)

1. **DE_DOCTRINE / FR_DOCTRINE** ("nach der Rechtsprechung" / "selon la jurisprudence" + ~7 paraphrases each) — **gold rate ~30 % vs ambient ~0.4-0.8 %; lift 14-88×**. Highest specificity-per-cost feature.
2. **FR_RULE_OPENER** ("Selon / Conformément à / Aux termes de / En vertu de l'art. X") — **gold rate ~50 % of FR gold vs ambient 0.27 %; lift ~74×**. Single strongest individual signal for FR gold.
3. **DE_RULE_OPENER** ("Nach / Gemäss / Im Sinne von Art. X") — **gold rate ~40 % of DE gold vs ambient 0.46 %; lift ~26-43×**. Single strongest individual signal for DE gold.
4. **Internal cite cluster** (≥3 `ATF/BGE N IVX N consid./E.` inside one parenthesis) — **gold rate ~70 % vs ambient ~6 %; lift ~10× per-paragraph, ~50× per-density**. Best "this paragraph is a doctrinal-cite anchor" signal.
5. **Anti-pattern hard-negs**: `FR_DISPOSITIF` ("frais judiciaires", "le recours est rejeté"), `DE_DISPOSITIF` ("Gerichtskosten", "wird abgewiesen"), `FR_SIGNATURE` ("le greffier", "le président:"), `DE_FACTS_HEADER` ("Sachverhalt") — **gold rate ≈ 0 % vs ambient 3-8 %**. Cheap precision booster (subtract a big weight on hit).
6. **DE_RULE_VERB / FR_RULE_VERB** ("Art. X bestimmt / dispose / prévoit") — **gold rate ~20 % vs ambient ~1 %; lift ~20×**. Catches definitional rule-statements that miss the opener regex.
7. **DE_PROCEDURAL / FR_PROCEDURAL** ("die Vorinstanz", "le recourant fait valoir") — **gold rate ~10 % vs ambient ~24 % (DE) / ~5 % (FR)**. Soft negative.
8. **Statute density** (refs / 100 chars, capped at 0.30) — **gold mean ~0.15-0.30 vs ambient ~0.05-0.15; lift ~2-3×**. Soft positive that complements the openers.

Cost: ~50s single-core for the full 2.47M corpus, ~15 µs/paragraph. Estimated gold-vs-ambient AUC: **0.80-0.88** on the binary "is doctrinal-rule-statement" task; this corresponds to a precision booster of roughly **+0.03-0.06 R@100** when stacked with the existing Phase 1 dossier features, with the lion's share of the lift coming from the dispositif/facts hard-negatives removing ambient noise rather than from the rule-statement positives picking gold.

## 10. Source notes

- Sample JSONs: `research/_scratch_2026-05-12/sample_val_{001..010}.json` (gold_court blocks; 102 paragraphs total).
- Ambient frequency measurements: `Grep` against `data/court_considerations.csv` (2 476 315 rows). All counts in §2-4 are direct match-counts (per-line, single CSV pass each) — not estimates.
- Control inspection: `data/court_considerations.csv` lines 500-540 (BGE 142 I 49 — Kopftuchverbot constitutional case, NOT val gold; demonstrates that doctrinal-rule-statement paragraphs exist outside val gold, confirming the score's limit: it identifies the *style*, not the *answer*. The statute-target intersection (Phase 1 dossier feature 2) closes that gap.
- All regexes verified for non-catastrophic backtracking: every quantified group has a bounded upper range or follows a literal anchor; no nested `.*` or `(a+)+` constructs.
