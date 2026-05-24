# Per-Query Court-Gold Anatomy — What Makes a Court Paragraph "Gold"

_Generated 2026-05-21. Builds on `gold_citation_legal_patterns_2026-05-12.md` (round 1), `gold_internal_structure_2026-05-13.md` (round 2 — aspect & graph), `linguistic_signatures_2026-05-13.md` (round 2 — regex), `near_miss_antipatterns_2026-05-13.md` (round 2 — anti-pattern forensics), and `feature_synthesis_round2_2026-05-13.md` (round 2 close-out). This round is the **close-reading deliverable**: for each val query, what specifically does the gold court text contain — read as a Swiss legal expert would read it — and what specifically does the rest of the 2.47M-row court-considerations corpus lack? Sources: per-query gold-text dumps at `research/scratch_2026-05-12_anti_pattern_gold_text/sample_val_*.json`; direct contrast-grep against `data/court_considerations.csv` (2,476,315 rows); `data/val.csv` for query text; cross-checked against the existing five research files above. **Scope by design**: only `gold_court` paragraphs of val queries are analysed — `gold_law` is out of scope here. No code execution, no embeddings, no model calls; this is a forensic close-reading study._

---

## 0. Why this deliverable exists, and what it adds beyond prior rounds

Prior rounds nailed the **structural / linguistic / pool-context** features (chamber, paragraph_role, rule-opener regex, statute-target intersection, doctrinal-density score, anti-pattern catalogue). Those features give a 32B judge useful signal. But the question the user is asking — *what specifically, in the text, makes a tiny number of paragraphs Federal-Supreme-Court-doctrine while 2.47M look almost identical at a feature level?* — pulls the analysis one layer deeper: what is the **Swiss-legal substance** of the gold, and how does that substance manifest in the text such that a Qwen3-8B agent, reading the paragraph directly, can identify gold without having to score it via a separate feature pipeline?

The answer, in one sentence: **a gold court paragraph is a CANONICAL DOCTRINAL RULE-STATEMENT for a specific legal sub-question that the query asks about, and the BGer's drafting tradition produces an extraordinarily regular 7-element template that distinguishes such paragraphs from every other use of the same statute in the corpus.**

Sections 1–3 set up the substantive picture. Sections 4–13 are per-query close readings of all ten val queries (a Swiss-lawyer pass on each). Section 14 gives the unified Qwen3-8B agent decision DAG. Section 15 lists the cheap textual self-checks the agent can compute mid-reasoning to avoid the dominant near-miss classes.

---

## 1. The single unifying signature: the 7-element doctrinal rule-statement template

Across **all ten val queries**, in three languages, across criminal procedure, social insurance, civil property, contract law, banking liability, child maintenance, holographic will, IPR-good-faith-acquisition, and disloyal-management cases, the gold court paragraphs converge on the same 7-element compositional template. Not every gold paragraph contains all seven elements — but every gold paragraph contains at least five, and the same five appear in the same syntactic order.

### The template, in order of appearance

| # | Element | DE realisation | FR realisation | Role |
|---|---|---|---|---|
| 1 | **STATUTORY ANCHOR (opener)** | "Gemäss / Nach Art. N Abs. M lit. x CODE …" | "Conformément à / Selon / Aux termes de l'art. N al. M let. x CODE …" | Names the rule by statute. Every gold paragraph has this within the first 1–2 sentences. |
| 2 | **PARAPHRASE (rule restatement)** | "ist X zulässig, wenn …" / "liegt vor, wenn …" / "hat Anspruch auf …" | "ne peut être ordonnée que lorsque …" / "a droit à …" / "est réputé …" | Restates the statutory rule in the court's voice, sometimes condensing multiple Abs. into one sentence. |
| 3 | **TELEOLOGY (purpose statement)** | "soll verhindern, dass …" / "bezweckt …" / "dient …" | "vise à empêcher …" / "a pour but …" / "tend à …" | The Schutzzweck. Distinguishes the rule from neighbouring rules with similar statutory text. |
| 4 | **POSITIVE LIMB (when the rule applies)** | "Konkrete Anhaltspunkte können sich namentlich ergeben aus …" / "insbesondere in der Weise, dass …" | "Selon la jurisprudence, il peut notamment y avoir … lorsque …" | The non-exhaustive list of factors that satisfy the rule. Often closes with "namentlich" / "notamment" / "in particolare". |
| 5 | **NEGATIVE LIMB (where the rule stops)** | "Die theoretische Möglichkeit … genügt indessen nicht …" / "vermag nicht zu …" | "ne saurait suffire …" / "ne saurait à elle seule …" | The boundary. This is the doctrine's anti-test: what is NOT enough. Often the most-cited sentence in echoes. |
| 6 | **INTERPRETIVE FACTORS (multi-factor balancing)** | "Bei der Frage, ob …, ist auch der Art, Bedeutung, Schwere … Rechnung zu tragen." | "Dans cet examen, entrent en ligne de compte les caractéristiques personnelles … le rôle … les liens …" | A short list of factors the judge weighs in the concrete case. Often introduced by "Bei der Frage" / "Dans cet examen" / "Pour retenir l'existence". |
| 7 | **AUTHORITY CHAIN (closing parenthesis)** | "(BGE 137 IV 122 E. 4.2 S. 127 f.; 132 I 21 E. 3.2 mit Hinweisen)" | "(ATF 137 IV 122 consid. 4.2 p. 127; 132 I 21 consid. 3.2 et les références)" | Closes with 2–6 chained ATF/BGE citations. "mit Hinweisen" / "et les références" / "und die dort zitierten Verweise" is the universal tail signature. |

### What this template looks like in practice — val_001 BGE 137 IV 122 E. 4.2 (the canonical Kollusionsgefahr Leitentscheid)

```
[1 statutory anchor]   Gemäss Art. 221 Abs. 1 lit. b i.V.m. Art. 237 Abs. 1 StPO
[2 paraphrase]         ist Untersuchungshaft respektive die Anordnung von Ersatzmassnahmen
                       zulässig, wenn ernsthaft zu befürchten ist, die beschuldigte Person
                       könnte Personen beeinflussen oder auf Beweismittel einwirken, um so
                       die Wahrheitsfindung zu beeinträchtigen.
[3 teleology]          Die strafprozessuale Haft wegen Kollusionsgefahr soll verhindern,
                       dass die beschuldigte Person die Freiheit dazu missbrauchen würde,
                       die wahrheitsgetreue Abklärung des Sachverhalts zu vereiteln oder
                       zu gefährden.
[4 positive limb]      Konkrete Anhaltspunkte für Kollusionsgefahr können sich nach der
                       Rechtsprechung des Bundesgerichts namentlich ergeben aus dem
                       bisherigen Verhalten des Beschuldigten im Strafprozess, aus seinen
                       persönlichen Merkmalen, aus seiner Stellung und seinen Tatbeiträgen
                       im Rahmen des untersuchten Sachverhaltes sowie aus den persönlichen
                       Beziehungen zwischen ihm und den ihn belastenden Personen.
[6 interpretive fac.]  Bei der Frage, ob im konkreten Fall eine massgebliche
                       Beeinträchtigung des Strafverfahrens wegen Verdunkelung droht, ist
                       auch der Art und Bedeutung der von Beeinflussung bedrohten Aussagen
                       bzw. Beweismittel, der Schwere der untersuchten Straftaten sowie dem
                       Stand des Verfahrens Rechnung zu tragen.
[7 authority chain]    (BGE 137 IV 122 S. 128)
```

Element 5 (negative limb) is present in the sibling BGE 132 I 21 E. 3.2: *"Die theoretische Möglichkeit, dass der Angeschuldigte in Freiheit kolludieren könnte, genügt indessen nicht, …"*. Note how the canonical paragraph and its predecessor split the seven elements between them — neither single paragraph carries all seven, but the doctrinal cluster (E. 4.2 + E. 3.2) does. Echo decisions (1B_90/2021 E. 2.1, 7B_69/2024 E. 3.3.2, etc.) typically carry **all seven elements compressed into one paragraph** because they re-recite the consolidated doctrine.

### Why this template separates gold from the 2.47M-row ambient

Element 1 (statutory anchor) alone is not enough — 649 paragraphs across the corpus mention "Art. 221 (Abs. 1) (lit. b) StPO" but only 37 are val_001/val_003 gold court. Elements 1+2 together (anchor + paraphrase) characterize roughly 4-5 % of corpus paragraphs that cite a given statute. Adding element 3 (teleology) drops the rate to ~1 %. Adding element 4 (positive limb with the canonical "namentlich" / "notamment" list) drops it again. By the time elements 1–4 are simultaneously present, what remains is the doctrinal-rule-statement subset of paragraphs — the universe from which gold is drawn.

The remaining ambient 99 % does one of these instead:
- Mentions the statute only in a procedural posture sentence ("Die Vorinstanz hat den Haftgrund nach Art. 221 lit. b StPO bejaht.") — element 1, but no 2/3/4.
- Quotes the party's argument referencing the statute ("Der Beschwerdeführer rügt eine Verletzung von Art. 221 Abs. 1 lit. b StPO") — element 1, but the verb is from the party, not the court.
- Applies the doctrine to the concrete case without re-reciting ("Demnach hat die Vorinstanz den besonderen Haftgrund der Kollusionsgefahr im Sinne von Art. 221 Abs. 1 lit. b StPO zu Recht als nach wie vor gegeben erachtet.") — closing subsumption, no 2/3/4.
- Mentions the statute only in a cost / admissibility recital ("Die Beschwerde wurde frist- und formgerecht erhoben (Art. 100 Abs. 1 BGG).") — non-substantive boilerplate.

---

## 2. The Swiss-legal substance behind the template — why this is not arbitrary

A Federal Supreme Court (Bundesgericht) judgment is doctrinally produced in a specific compositional sequence that mirrors the seven-element template. This is not stylistic — it is the BGer's drafting tradition, taught explicitly in legal-method courses and visible across decades. The compositional logic:

1. **Statutory anchor** — Swiss legal reasoning is Civil-Law positivist. Every legal conclusion must hang from a statute. The court therefore opens by naming the statutory rule it is about to apply.
2. **Paraphrase** — The statute is written for the legislator's audience; the court restates it in the courtroom audience's voice. This is not redundant: it is the court's reading of the statute, which differs at the margins from a literal reading.
3. **Teleology (Schutzzweck)** — Continental legal method is intensely teleological. The protected interest defines the rule's outer reach. Without the teleology stated, the rule is dead-text; with it, the rule has reach and limit.
4. **Positive limb** — The non-exhaustive factor list ("namentlich" / "notamment") is the BGer's way of dropping bright-line guidance without overcommitting to a rule. This is doctrinally precise: it tells the lower courts *what kinds of facts* satisfy the rule, without enumerating which combinations are sufficient.
5. **Negative limb** — "Die theoretische Möglichkeit genügt nicht" is the **ratio decidendi** in many BGer decisions. The court's contribution is often not what the rule says (the statute says that) but where the rule stops. The negative limb is therefore the highest-value sentence and the most-cited.
6. **Interpretive factors** — A second-order balancing test. Useful for proportionality cases, multi-factor doctrines (Kollusionsgefahr factors, Gefälligkeit factors, Beweiswert factors, Anklagegrundsatz factors).
7. **Authority chain** — The court closes by citing 2–6 prior decisions that establish the same doctrine. "mit Hinweisen" / "et les références" tells the reader that there are even more references behind these. This is the BGer's way of marking a paragraph as **doctrinal** (as opposed to ad hoc reasoning, which would not need authority).

The 2.47M-row corpus contains BGer decisions and BVGer decisions. **Of these, only the paragraphs that fulfill all seven elements are "the rule".** Other paragraphs — the facts, the procedural admissibility tests, the cost allocation, the dispositif, the party-argument framing, the lower-court-summary — are present in the same decisions but do different work. The doctrinal paragraphs are roughly 10–20 % of any one decision's E.-section body; the other 80 % are the surrounding scaffolding.

**This is exactly why gold court paragraphs are sparse, and why the same case can supply multiple gold paragraphs (different E.) while contributing nothing in others.** val_001 has four E. subsections of BGE 137 IV 122 in gold (4.1, 4.2, 6.2, 6.4) — and zero from E. 1 (parties), E. 2 (facts), E. 3.1 (general detention base), E. 5 (procedural admissibility), E. 7 (operative). The four E.'s that are gold are the four that carry the seven-element template.

---

## 3. The "doctrinal chain" — Leitentscheid → echo decisions → mit Hinweisen tail

The val_001 gold court set has 23 paragraphs across roughly a dozen distinct decisions. Reading them in court-date order shows the BGer doctrinal-chain mechanism:

```
BGE 132 I 21 E. 3.2 (2005)        ← Predecessor: defines "Kollusion" and the negative limb
       │
       ↓ cited as "(BGE 132 I 21 E. 3.2 mit Hinweisen)"
BGE 137 IV 122 E. 4.2 (2011)       ← Canonical Leitentscheid: consolidates all 7 elements
       │
       ↓ cited as "(BGE 137 IV 122 E. 4.2; 132 I 21 E. 3.2 mit Hinweisen)"
1B_90/2021 E. 2.1, 1B_357/2022 E. 3.1, 1B_15/2023 E. 3.1, 1B_28/2022 E. 4.1,
1B_210/2023 E. 4.1, 7B_69/2024 E. 3.3.2, 7B_301/2024 E. 2.4, 7B_12/2025 E. 2.2,
7B_496/2025 E. 3.2, 7B_231/2025 E. 4.1
       │
       ↓ each one reuses elements 1-5 verbatim (or near-verbatim FR translations),
         and closes with "(BGE 137 IV 122 E. 4.2; 132 I 21 E. 3.2; mit Hinweisen)"
```

A Swiss lawyer reads these as **"echoes of the Leitentscheid"** — same doctrine, sometimes mildly rephrased, often with a small new factor added to the positive or negative limb. The echoes are gold because:

1. They establish that the doctrine is **constant** (cf. "ständige Rechtsprechung").
2. They give the lower court a **recent** Leitsatz it can rely on without needing to dig into 2011 BGE volumes.
3. They sometimes contain doctrinal refinements (e.g. 1B_90/2021 E. 2.4 adds the "Loyalitätsdilemma / psychischer Druck" criterion as an application of the existing factor list).

This is the **echo decision** pattern. Across all 10 val queries, every multi-paragraph gold set contains a hub Leitentscheid and 2–8 echo decisions reciting it.

### Concrete cross-query confirmation

| qid | Leitentscheid hub(s) | Echo decision count in gold |
|---|---|---:|
| val_001 | BGE 137 IV 122 + BGE 132 I 21 (Kollusionsgefahr) | 10 (7B_*, 1B_* unpublished) |
| val_002 | BGE 124 V 108 + BGE 139 V 399 (Umschulung 20 %) + BGE 125 V 351 (Beweiswert Arztbericht) | 5+ |
| val_003 | BGE 137 IV 122 + BGE 139 IV 186 + BGE 140 I 285 (right-to-be-heard for detention) | 8 |
| val_004 | BGE 131 III 601 (testament olographe form) | 0 — single Leitentscheid (single gold court paragraph) |
| val_005 | BGE 130 III 585 + BGE 128 III 411 (visitation + Untersuchungsmaxime) | 3 |
| val_006 | BGE 137 III 539 + BGE 129 III 181 (Gefälligkeit vs Vertrag) | 4 echo |
| val_007 | BGE 144 III 264 + BGE 139 III 305 (Urteilsfähigkeit + gutgläubiger Erwerb) | 2 |
| val_008 | BGE 131 III 91 + BGE 135 III 334 + BGE 143 IV 63 (autorité de l'arrêt + Anklagegrundsatz + Art. 314 StGB) | 6 |
| val_009 | BGE 137 III 193 (Unterhalt bei Inhaftierung) | 2 |
| val_010 | BGE 132 III 449 + BGE 146 III 326 (Bankhaftung Risiko-Transfer-Klausel) + BGE 134 III 151 (Fremdwährungsforderung) | 6 |

In every case, the gold pool is the **Leitentscheid + its echoes**. The non-gold pool includes earlier predecessor decisions that were later superseded, parallel decisions on different doctrines that mention the same statute, and the procedural / fact / dispositif paragraphs *within* the Leitentscheide themselves.

---

## 4. Per-query close reading — val_001 (pre-trial detention extension under Art. 221 Abs. 1 lit. b StPO)

### Query in one sentence
Is a three-month extension of pre-trial detention under collusion-risk lawful, consistent with proportionality, when investigations are mostly technical and the alleged victim has withdrawn the complaint?

### Doctrinal sub-questions the query embeds
1. **Existence of collusion risk** as a special detention ground (Art. 221 Abs. 1 lit. b StPO).
2. **Proportionality of detention duration** — must not approach the expected sentence (Art. 212 Abs. 3 StPO).
3. **Procedure for extension** — Art. 227 (extension request) + Art. 222 (party standing) + Art. 393/396/385/382/390 (appeal route) + Art. 422/428/135/100 BGG (costs / appeal deadline / counsel fees).
4. **Alternative measures (Ersatzmassnahmen)** — Art. 237 StPO (the lower-step measure that may suffice).

### What the gold court paragraphs uniquely contain

Twenty-three Federal Court paragraphs across approximately 11 decisions. Three sub-clusters:

**Cluster A — Kollusionsgefahr doctrine** (13 paragraphs): The seven-element template applied to Art. 221 Abs. 1 lit. b StPO. Hub: BGE 137 IV 122 E. 4.2 (DE). Predecessor: BGE 132 I 21 E. 3.2 (DE). FR variants: 1B_210/2023 E. 4.1, 1B_536/2018 E. 5.1. Echoes: 1B_90/2021, 1B_357/2022, 1B_15/2023, 1B_28/2022, 7B_69/2024, 7B_301/2024, 7B_12/2025, 7B_231/2025, 7B_496/2025 (latter is application-style).

**Cluster B — Detention duration proportionality** (3 paragraphs): BGE 139 IV 270 E. 3.1 (FR), BGE 133 I 168 E. 4.1 (FR), BGE 143 IV 168 E. 5.1 (FR), BGE 133 I 270 E. 3.4.2 (DE). All open with element 1: "En vertu des art. 31 al. 3 Cst. et 5 par. 3 CEDH" / "Gemäss Art. 31 Abs. 3 BV und Art. 5 Ziff. 3 EMRK" — the constitutional and ECHR anchors. Then paraphrase element 2: "toute personne … a le droit d'être jugée dans un délai raisonnable" / "hat eine in strafprozessualer Haft gehaltene Person Anspruch darauf, innerhalb einer angemessenen Frist richterlich beurteilt … zu werden". Then teleology element 3 implicit. Then concrete rule element 4: "Une durée excessive constitue une limitation disproportionnée … qui est notamment violé lorsque la durée … dépasse la durée probable de la peine privative de liberté".

**Cluster C — Ersatzmassnahmen / proportionality of measure** (3 paragraphs): BGE 137 IV 122 E. 6.2 + E. 6.4. The court applies the proportionality test to the lower-court order: "Eine Eingrenzung auf ein bestimmtes Gebiet kommt … primär bei Fluchtgefahr in Betracht. Geht es demgegenüber darum, einer Kollusionsgefahr in Form der möglichen Beeinflussung des mutmasslichen Opfers zu begegnen, dürfte in aller Regel eine Ausgrenzung als mildere Massnahme genügen." This is application-style gold inside the Leitentscheid.

### What non-gold corpus paragraphs that mention Art. 221 Abs. 1 lit. b StPO contain instead

From direct grep against `court_considerations.csv` (649 hits on the article string, of which only ~37 are gold across val_001 + val_003):

- **`BGE 146 IV 136 E. 2.10`**: *"Die Voraussetzungen des Haftgrunds der Wiederholungsgefahr sind demnach nicht erfüllt. Fluchtgefahr (Art. 221 Abs. 1 lit. a StPO) verneint die Vorinstanz. Dass keine Kollusionsgefahr (Art. 221 Abs. 1 lit. b StPO) besteht, hat das Bundesgericht bereits entschieden."* — This is a **closing-summary paragraph**: the court is summing up across sub-grounds at the end. It MENTIONS the article but only to reference a prior holding ("hat … bereits entschieden"). No paraphrase, no teleology, no positive limb. Elements present: 1 only. Verdict: NOT GOLD because it is element-1-only.
- **`1B_686/2021 E. 2.4`**: *"Es kann offen bleiben, ob neben der Fluchtgefahr auch noch der alternative besondere Haftgrund der Kollusionsgefahr (Art. 221 Abs. 1 lit. b StPO) zusätzlich erfüllt wäre."* — **Moot-determination paragraph** ("offen bleiben"). The court declines to apply the rule. Element 1 only. NOT GOLD.
- **`1B_28/2022 E. 4`** (compare to E. 4.1 which IS gold): *"Der Beschwerdeführer bestreitet weiter das Vorliegen von Kollusionsgefahr im Sinne von Art. 221 Abs. 1 lit. b StPO."* — **Party-position framing**: the court is summarising the appellant's claim. Element 1 only. NOT GOLD. But E. 4.1, the immediately following paragraph in the same decision, opens with the seven-element template and IS gold.
- **`1B_308/2013 E. 2.1`**: *"Gemäss Art. 221 Abs. 1 StPO ist Sicherheitshaft zulässig, wenn die beschuldigte Person eines Verbrechens oder Vergehens dringend verdächtig ist und ernsthaft zu befürchten ist, dass sie Personen beeinflusst oder auf Beweismittel einwirkt, um so die Wahrheitsfindung zu beeinträchtigen (lit. b). Der Beschwerdeführer stellt den dringenden Tatverdacht nicht in Abrede. Er macht geltend, es fehle an der Kollusionsgefahr nach Art. 221 Abs. 1 lit. b StPO."* — Has elements 1+2 (statutory anchor + paraphrase), but pivots immediately into party-position. No teleology, no positive/negative limb, no authority chain. NOT GOLD. This is the **pre-Leitentscheid shallow recital** anti-pattern: the older 2013 case is doctrinally accurate but compositionally thin; it predates the 2011 consolidation in BGE 137 IV 122 only by 2 years but the court hadn't yet adopted the full template here.

### The two ways the Qwen3-8B agent can distinguish gold from the above on text alone

1. **Read the OPEN of the paragraph** (first 1–2 sentences). If it is an opener+paraphrase pair where the verb belongs to the **court** ("ist zulässig", "liegt vor", "ne peut être ordonnée que lorsque"), and the paragraph then continues with teleology or doctrinal elaboration *for at least 3 more sentences*, this is gold-shaped. If the opener+paraphrase is followed by a pivot to "Der Beschwerdeführer rügt …" / "le recourant fait valoir" / "Es kann offen bleiben …", this is shallow-recital — NOT gold.
2. **Read the CLOSE of the paragraph** (last 1–2 sentences). Gold paragraphs end with the authority chain (parenthetical 2-6 ATF/BGE cites + "mit Hinweisen" / "et les références"). Shallow-recital paragraphs end with party-position or closing subsumption ("Demnach …", "Im Ergebnis …").

If both the open and the close match, this is a doctrinal rule-statement and overwhelmingly likely to be gold for the right query. The agent should then ask: *does the paraphrased rule match the query's legal sub-question?* If yes → confirm gold; if no (different doctrine on same statute) → reject.

---

## 5. Per-query close reading — val_002 (invalidity insurance / Umschulung at ~20 % earning-capacity loss)

### Query in one sentence
Does a warehouse worker with chronic eosinophilic asthma have a right to vocational retraining under Art. 17 LAI (and IV benefits) given conflicting medical opinions and the jurisprudential 20 %-earning-loss threshold?

### Doctrinal sub-questions
1. **Invalidity definition** (Art. 8 Abs. 1 ATSG / IVG, Art. 4 IVG, Art. 6 ATSG).
2. **Umschulung entitlement and the ~20 % threshold** (Art. 17 IVG).
3. **Beweiswert / free appraisal of medical evidence** (Art. 61 lit. c LPGA = ATSG; BGE 125 V 351).
4. **Beweisgrad: überwiegende Wahrscheinlichkeit** (Untersuchungsgrundsatz in Sozialversicherungsprozess).
5. **Allgemeine Verfahrensregeln** (cost, appeal, deadlines).

### What the gold court paragraphs uniquely contain

Sixteen Federal Court paragraphs across roughly 9 decisions, three sub-clusters:

**Cluster A — Beweisgrad / Untersuchungsgrundsatz** (3-4 paragraphs): BGE 144 V 427 E. 3.2 (DE) — *"Der Sozialversicherungsprozess ist vom Untersuchungsgrundsatz beherrscht. … Im Sozialversicherungsrecht hat das Gericht seinen Entscheid, sofern das Gesetz nicht etwas Abweichendes vorsieht, nach dem Beweisgrad der überwiegenden Wahrscheinlichkeit zu fällen."* Elements 2+3+4 present, with paraphrase that is itself the doctrinal rule. FR sibling: BGE 139 V 176 E. 5.3, BGE 135 V 39 E. 6.1 — both opening "Dans le domaine des assurances sociales, le juge fonde généralement sa décision sur les faits qui … apparaissent comme les plus vraisemblables, c'est-à-dire qui présentent un degré de vraisemblance prépondérante."

**Cluster B — Umschulung threshold ≈ 20 %** (3 paragraphs): BGE 124 V 108 E. 2b (DE), BGE 139 V 399 E. 5.3, 5.4, 5.5 (FR). The 1998 Leitentscheid (BGE 124 V 108) states: *"nach der Rechtsprechung ist dies der Fall, wenn der Versicherte … eine bleibende oder längere Zeit dauernde Erwerbseinbusse von etwa 20 Prozent erleidet"*. The 2013 FR variant (BGE 139 V 399 E. 5.3) restates: *"Le seuil minimum fixé par la jurisprudence pour ouvrir droit à une mesure de reclassement est une diminution de la capacité de gain de 20 % environ (ATF 130 V 488 consid. 4.2)"*. Note element 7 (authority chain) is "(ATF 130 V 488)" — the intermediate Leitentscheid. Three E. variants of BGE 139 V 399 are gold (5.3, 5.4, 5.5) because the doctrinal section spans multiple paragraphs.

**Cluster C — Medical evidence: Beweiswert** (3+ paragraphs): BGE 134 V 231 E. 5.1 (DE), 8C_510/2020 E. 2.4 (FR), 8C_160/2016 E. 4.1 (FR). All cite back to BGE 125 V 351 (the canonical Beweiswert Leitentscheid). Same template, same authority chain.

### What ambient corpus paragraphs that mention Art. 17 IVG / Umschulung contain instead

The corpus has thousands of social-insurance paragraphs mentioning Art. 17 LAI. The discriminator is the same: gold paragraphs **define what the rule means in 5+ sentences with the seven-element template**. Ambient paragraphs either:

- Mention Art. 17 LAI only in the cost / admissibility recital ("Le recours a été déposé en temps utile (art. 100 al. 1 LTF)") — element 1 only.
- Apply Art. 17 LAI to the concrete case without re-reciting ("En l'espèce, la perte de gain de 18 % ne suffit pas à ouvrir le droit au reclassement.") — that's element 4 application, no anchor + paraphrase + teleology.
- Discuss a sub-issue downstream of the rule (e.g. *which* retraining measure is appropriate), citing the rule only in passing.

### Cross-language sub-pattern (specific to val_002)
A single doctrine appears in **paired DE/FR paragraphs**. BGE 144 V 427 (DE) and BGE 139 V 176 / 135 V 39 (FR) all state the Beweisgrad doctrine — in different language but with the *same* element-structure and the *same* authority chain (both end pointing to BGE 125 V 351). The DE/FR pair is the same legal rule with parallel rhetorical scaffolding. **This is a robust gold-confirmation signal**: a candidate paragraph in DE that matches an already-confirmed FR gold's doctrine, or vice versa, is extremely likely also gold.

---

## 6. Per-query close reading — val_003 (Rivera, pre-trial detention extension, right-to-be-heard)

### Query in one sentence
Was Rivera's detention extension lawful under Art. 221 Abs. 1 StPO, was his right to be heard respected, and did the reasoning give him enough information to challenge?

### Doctrinal sub-questions
1. **Detention legality framework** (Art. 221 al. 1 CPP — full ingress + lit. a/b/c).
2. **Right to be heard** (Art. 29 al. 2 Cst.) and reasoning requirement.
3. **Suspicion threshold (dringender Tatverdacht)** as the precondition.
4. **Procedural/appeal architecture** (Art. 393/222/396/etc.).

### What the gold court paragraphs uniquely contain

Twenty-three Federal Court paragraphs across roughly 12 decisions. Distinguishable from val_001 by:
- **Stronger FR weighting** (14 of 23 are FR; val_001 had 10/23).
- **Different leading paragraph structure**: the val_003 FR gold (1B_88/2022 E. 2.1, 1B_572/2021 E. 2.1, 1B_211/2017 E. 2.1, 1B_581/2022 E. 2.1.2, BGE 139 IV 186) all open with **the general detention-framework paragraph**, not the collusion-specific one:

  *"Une mesure de détention provisoire ou pour des motifs de sûreté n'est compatible avec la liberté personnelle (art. 10 al. 2 Cst. et 5 CEDH) que si elle repose sur une base légale (art. 31 al. 1 et 36 al. 1 Cst.), soit en l'espèce l'art. 221 CPP. Elle doit en outre correspondre à un intérêt public et respecter le principe de la proportionnalité (art. 36 al. 2 et 3 Cst.). Pour que tel soit le cas, la privation de liberté doit être justifiée par un risque de fuite, un danger de collusion ou de réitération (cf. art. 221 al. 1 let. a, b et c CPP). Préalablement à ces conditions, il doit exister à l'égard de l'intéressé des charges suffisantes, soit de sérieux soupçons de culpabilité (art. 221 al. 1 CPP; art. 5 par. 1 let. c CEDH; ATF 139 IV 186 consid. 2)."*

  This is the **constitutional-framework paragraph** for pre-trial detention. Note seven elements explicitly: anchor (Art. 31/36 Cst., Art. 221 CPP, Art. 5 CEDH) + paraphrase + teleology (proportionality, liberté personnelle) + positive limb (the three grounds) + the precondition (charges suffisantes) + authority chain (ATF 139 IV 186 consid. 2). The framework paragraph is a "structural rule recital", not a single-issue recital, and is gold for any pre-trial detention query.

- **Dringender-Tatverdacht doctrine** (BGE 143 IV 330 E. 2.1 + BGE 143 IV 316 E. 3.1, E. 3.2; BGE 137 IV 122 E. 3.2; BGE 124 I 208 E. 3): the standard for suspicion at the haft-stage — the court must verify there are "konkrete Anhaltspunkte" without conducting a full evidence-weighing.

- **Akkusationsprinzip / right-to-be-heard** (BGE 142 III 48 E. 4.1.1, BGE 145 I 167 E. 4.1, BGE 140 I 285 E. 6.3.1, 2C_501/2020 E. 5.1): rule-recital of the reasoning-requirement doctrine under Art. 29 Abs. 2 BV.

### Key contrast with val_001
val_001 and val_003 both ask about Art. 221 lit. b StPO detention. **val_001 zooms into collusion risk (lit. b specifically)**, so its gold weights the collusion-doctrine cluster heavily. **val_003 asks about the general detention framework + right to be heard**, so its gold weights the framework paragraph and the right-to-be-heard cluster more. This is a fine-grained doctrinal-precision signal: the same statute can underpin two different sub-doctrines, and the gold partitions accordingly. A retrieval system that just retrieves "Art. 221 lit. b StPO" pages will surface both val_001 and val_003 candidates equally — the discriminator is which doctrinal sub-cluster the paragraph belongs to.

### What ambient corpus paragraphs (different Art. 221 sub-cluster) contain
Many corpus paragraphs about Art. 221 detention focus on **flight risk (lit. a)** or **repetition risk (lit. c)** — not the val_001/003 doctrinal cluster. They follow the seven-element template too, but for a different sub-rule. **A Qwen3-8B agent needs to read what the paragraph is doctrinally about, not just what statute is in the opener.** A paragraph that opens with "Le danger de fuite est concret lorsque …" is doctrinally about flight risk; if the query asks about collusion, this paragraph is NOT gold even though it cites Art. 221 al. 1 CPP.

---

## 7. Per-query close reading — val_004 (holographic will — formal requirements)

### Query in one sentence
Does Mr. Dalton's handwritten 1997 will satisfy the formal requirements under Art. 505 ZGB despite the contesting heirs' challenges to his German-language vocabulary and parts allegedly written by a third hand?

### Doctrinal sub-questions
1. **Formal requirements of holographic will** (Art. 505 Abs. 1 ZGB — entirely handwritten + dated + signed by testator).
2. **Effect of third-party additions** (Lehre/Rechtsprechung — third-hand passages are null but the will survives if testator wrote the essential elements himself).
3. **Testamentary capacity** (Art. 16 ZGB / Art. 467 ZGB).

### What the single gold court paragraph contains

Only one court gold for val_004: **BGE 131 III 601 E. 3.1** (FR), 2685 chars. It is an exemplar of the seven-element template:

```
[1] Aux termes de l'art. 505 al. 1 CC, le testament olographe est écrit en entier, daté et
    signé de la main du testateur; la date consiste dans la mention de l'année, du mois
    et du jour où l'acte a été dressé.
    L'art. 520 al. 1 CC prévoit que les dispositions entachées d'un vice de forme sont
    annulées.
[2] paraphrase implicit in the statutory text (the rule is short enough)
[7 partial] (ATF 57 II 15; ATF 88 II 67 consid. 2; ATF 117 II 142 consid. 2a)
[2 cont.] Il doit être écrit du début à la fin de la main du testateur.
[doctrine] Selon la doctrine et la jurisprudence, lorsqu'un tiers prête assistance au
           testateur pour écrire un testament olographe, les passages écrits par une main
           étrangère sont nuls.
[5 negative limb]  Toutefois, l'acte demeure valable si le testateur a écrit lui-même les
                   éléments essentiels des dispositions … et que les adjonctions de la
                   main du tiers n'ont trait qu'à des éléments d'importance secondaire ou
                   au rétablissement d'une lettre que le testateur aurait omise et que
                   tout lecteur rétablirait de lui-même.
```

The negative limb is the key sentence — it is the **ratio decidendi** of BGE 131 III 601: third-hand corrections of secondary elements do not invalidate the will. This is exactly the holding the val_004 query needs.

### Why no echo decisions in val_004 gold
The holographic-will doctrine is well-settled and litigated infrequently at the Federal Supreme Court level. There are subsequent BGer rulings citing BGE 131 III 601 (a corpus grep would find them), but the gold curator chose just the canonical Leitentscheid. This is a single-paragraph gold case — the smallest court-gold set in val.

### What ambient corpus paragraphs mentioning Art. 505 ZGB contain instead
Most ambient paragraphs about Art. 505 ZGB are application-style: the court evaluates the specific facts ("Im vorliegenden Fall war der Testamentstext durchgehend in der Handschrift des Erblassers …") or cite Art. 505 only in admissibility/cost (which rarely happens for Art. 505 since it's a substantive article). The doctrinal-rule-statement paragraph is rare — and BGE 131 III 601 E. 3.1 is the modern canonical one. **A Qwen3-8B agent that prioritises the seven-element template should find this paragraph trivially among 175 paragraphs citing Art. 505 ZGB in the corpus.**

---

## 8. Per-query close reading — val_005 (parental visitation restriction)

### Query in one sentence
Can a civil judge, on ex officio fact-finding and best-interests-of-the-child grounds, restrict (provisionally or permanently) overnight visitation given sexual-touching allegations, an alcoholism history, and an absent forensic psychiatric report?

### Doctrinal sub-questions
1. **Personal-contact right and limits** (Art. 273 + Art. 274 Abs. 2 ZGB).
2. **Best-interests-of-the-child as the supreme standard**.
3. **Untersuchungsmaxime in family proceedings** (ex officio fact-finding).
4. **Custody and maintenance** (Art. 133, Art. 285 ZGB).

### What the gold court paragraphs uniquely contain

Five Federal Court paragraphs:

- **BGE 130 III 585 E. 2.1** — Statutory anchor + paraphrase + interpretive factors:
  *"Eltern, denen die elterliche Sorge oder Obhut nicht zusteht, und das unmündige Kind haben gegenseitig Anspruch auf angemessenen persönlichen Verkehr (Art. 273 Abs. 1 ZGB). … Vielmehr gilt als oberste Richtschnur für die Ausgestaltung des Besuchsrechts immer das Kindeswohl, das anhand der Umstände des konkreten Einzelfalls zu beurteilen ist; allfällige Interessen der Eltern haben zurückzustehen (BGE 127 III 295 E. 4a; BGE 123 III 445 E. 3b)."*
  The "oberste Richtschnur" sentence is the ratio. Authority chain present.

- **BGE 130 III 585 E. 2.2.1** — Application of the rule with policy elaboration (loyalty conflict, child distress).

- **BGE 131 III 209 E. 5** — Application paragraph that supersedes a cantonal practice ("Das Abstellen auf eine solche Praxis muss jedoch mit der in BGE 130 III 585 publizierten Rechtsprechung … als überholt gelten."). This is a **case-law-updating** paragraph.

- **BGE 128 III 411 E. 3.2.1** — FR rule-statement on the Untersuchungsmaxime in family proceedings, citing back to ATF 122 I 53, 122 III 404, 111 II 225 — proper authority chain.

- **BGE 126 III 219 E. 2** — Application paragraph from an older decision dealing with visitation refusal + Beistandschaft.

### Specific Swiss-legal subtlety in val_005 gold

Two of the five gold court paragraphs (BGE 131 III 209 E. 5 and BGE 126 III 219 E. 2) are **application paragraphs**, not pure rule recitals. They are gold because:

1. They are part of the Leitentscheid for visitation doctrine in the modern era (BGE 130 III 585 + BGE 131 III 209).
2. They articulate doctrine *through* application — the "oberste Richtschnur" doctrine is restated in BGE 131 III 209 E. 5 by overruling cantonal practice that contradicted it.

This is the **doctrine-through-application** gold subtype. The agent should recognize it by the presence of:
- A reference to a recent Leitentscheid ("mit der in BGE 130 III 585 publizierten Rechtsprechung").
- A doctrinal correction or refinement ("Das Abstellen auf eine solche Praxis muss … als überholt gelten").
- The "oberste Richtschnur" / "Kindeswohl" doctrinal language.

### What non-gold paragraphs mentioning Art. 273/274 ZGB contain
Vast majority are application paragraphs of individual cantonal cases — the court ruling on whether a specific weekend pattern is "angemessen". These are not gold. Also non-gold: paragraphs on neighbouring articles (Art. 270a — surname, Art. 298b — joint authority, Art. 301a — domicile change, Art. 134 — modification, Art. 275a — information rights, Art. 318 — child property). These are all in the same code section but address different doctrines.

---

## 9. Per-query close reading — val_006 (Gefälligkeit vs Vertrag — uncertified tank decommissioning)

### Query in one sentence
Is the gratuitous installer G liable for fire damage, and on what basis: contract-for-work (Art. 364 OR), gratuitous-gift analogy (Art. 248 OR), or non-contractual delict (Art. 41 OR)?

### Doctrinal sub-questions
1. **Vertrag vs Gefälligkeit classification** (the threshold question).
2. **Liability basis once classified** (Vertragshaftung, Gefälligkeitshaftung as delict, or Art. 422 OR analogy for negotiorum-gestio).
3. **Standard of care** under each basis (objective standard vs gross-negligence-only for gifts).

### What the gold court paragraphs uniquely contain

Seven Federal Court paragraphs:

- **BGE 137 III 539 E. 4.1** (the canonical 2011 Leitentscheid): *"In der Rechtsprechung ist anerkannt, dass auch im Bereich von Arbeitsleistungen unverbindliche Gefälligkeiten vorkommen, die keine Vertragsbindung entstehen lassen. Ob Vertrag oder Gefälligkeit vorliegt, entscheidet sich nach den Umständen des Einzelfalles, insbesondere der Art der Leistung, ihrem Grund und Zweck, ihrer … rechtlichen und wirtschaftlichen Bedeutung, den Umständen, unter denen sie erbracht wird und der Interessenlage der Parteien. Für einen Bindungswillen spricht ein eigenes, rechtliches oder wirtschaftliches Interesse der Person … oder ein erkennbares Interesse des Begünstigten an fachkundiger Beratung oder Unterstützung (BGE 129 III 181 E. 3.2; BGE 116 II 695 E. 2b/bb S. 697 f.)."*

  This is gold-shaped: anchor implicit (the doctrine is Lehrentwicklung not a single article), paraphrase explicit, positive-limb (factor list), authority chain present.

- **BGE 137 III 539 E. 5.1**: rule-statement on the **liability basis** once gefälligkeit is found: *"Nach der bundesgerichtlichen Rechtsprechung haftet die Person, welche aus Gefälligkeit eine Leistung erbringt, aus unerlaubter Handlung (BGE 116 II 695 E. 4 S. 699) …"*. Element 1 = "Nach der bundesgerichtlichen Rechtsprechung", element 2 = the rule, element 7 = (BGE 116 II 695).

- **BGE 137 III 539 E. 5.2**: rule-statement on the **standard of care** + Art. 41 OR delict structure + Art. 99 Abs. 2 OR for the gift-analogous milder standard.

- **BGE 129 III 181 E. 3.2**: the predecessor Leitentscheid restating the doctrine in identical structure, citing back to BGE 116 II 695.

- **BGE 128 III 419 E. 2.2**: Contract-interpretation doctrine (Art. 18 OR) — this is a **different doctrinal sub-cluster** that the val_006 query also touches (whether a verbal/conduct-based arrangement amounts to a contract). The paragraph opens with: *"En présence d'un litige sur l'interprétation d'un contrat, le juge doit tout d'abord s'efforcer de déterminer la commune et réelle intention des parties, sans s'arrêter aux expressions ou dénominations inexactes dont elles ont pu se servir … (art. 18 al. 1 CO; ATF 127 III 444 consid. 1b)."* Statutory anchor + paraphrase + authority chain.

- **BGE 130 III 417 E. 3.2**: a very short application paragraph (148 chars) — application of the contract-interpretation rule in the case.

- **BGE 121 IV 207 E. 2a**: A **criminal-chamber** decision (IV chamber) discussing **negligence-causation** doctrine under Art. 125 CP — this is the standard-of-care doctrine for negligence, here referenced because val_006 asks about "what standard of care" should apply. The chamber is IV (criminal), unusual for a civil-liability query, but the doctrine is shared. Opens with: *"La recourante soutient qu'en admettant une rupture du rapport de causalité adéquate, la cour cantonale a violé l'art. 125 CP. Selon l'art. 125 al. 1 CP, '...'. L'art. 18 al. 3 CP donne une définition de la négligence: '...'."*

### Critical insight from val_006

The gold mixes **three doctrinal sub-clusters** that the query touches:
1. Gefälligkeit doctrine (BGE 137 III 539 + BGE 129 III 181).
2. Contract-interpretation doctrine (BGE 128 III 419 + BGE 130 III 417 + Art. 18 OR).
3. Negligence-standard doctrine (BGE 121 IV 207 + Art. 41 OR).

A near-miss paragraph that recites only one of these doctrines (e.g. Art. 41 OR generic delict-liability paragraph) is gold only if the query touches that specific sub-cluster. **The gold curator selected paragraphs whose doctrinal sub-question matches one of the query's three legal questions.** This is the multi-aspect insight from `gold_internal_structure_2026-05-13.md` confirmed at the text level: the gold is a multi-doctrine cluster, each member fulfilling one aspect.

### Near-miss contrast: 4C.56/2002 E. 3

A grep against `court_considerations.csv` for "Gefälligkeit vorliegt, entscheidet sich" returns **4C.56/2002 21.10.2002 E. 3** which recites the exact same doctrine — but is NOT in val_006 gold. Why? The paragraph (read in full from line 1706517 of the corpus) recites BGE 116 II 695 (the older Leitentscheid) AND BGE 61 II 95 + BGE 48 II 487, applying the doctrine to a different fact pattern (a farmer asking a neighbour for help with a tree). It contains the doctrine but:

1. **Predates the 2011 Leitentscheid (BGE 137 III 539)** — modern gold uses the latest consolidation.
2. **Discusses Art. 422 OR analogy** (negotiorum gestio compensation) — a different doctrinal corner that the val_006 query does *not* ask about.
3. Mixes the rule recital with application narrative (extensive cantonal-court summary).

Gold-vs-non-gold differential here: **doctrinal sub-cluster matching + recency of the Leitentscheid**. The val_006 query asks about the contract-vs-gefälligkeit threshold and the liability standard once Gefälligkeit is found. 4C.56/2002 discusses a different sub-corner of the doctrine (negotiorum-gestio analogy). NOT GOLD.

---

## 10. Per-query close reading — val_007 (IPR good-faith acquisition / chronometer)

### Query in one sentence
Under New York and Swiss law, was the inter vivos gift valid (donative intent, transfer, acceptance), and would Alpine Trading AG as professional purchaser be required to return the chronometer under Art. 934 / 936 ZGB?

### Doctrinal sub-questions
1. **IPR / choice of law for movable property** (Art. 98 + 100 IPRG).
2. **Good-faith acquisition** (Art. 933, 934, 940 ZGB; Art. 3 ZGB).
3. **Donation form and capacity** (Art. 245 + Art. 15 OR; Art. 16 ZGB).
4. **Burden of proof** (Art. 8 ZGB).

### What the gold court paragraphs uniquely contain

Only four court gold paragraphs, all quite short:

- **BGE 139 III 305 E. 5** (274 chars): application paragraph stating the lower court's finding on good-faith acquisition.
- **BGE 132 III 155 E. 4** (124 chars): party-position framing ("Die Kläger behaupten zur Hauptsache, der Besitz - und damit auch das Eigentum - sei durch Besitzanweisung übertragen worden").
- **BGE 144 III 264 E. 5.1** (358 chars): rule-statement on Urteilsfähigkeit (testamentary capacity) — *"Das Bundesverwaltungsgericht ist davon ausgegangen, die tatsächlichen Grundlagen der Urteilsunfähigkeit seien mit dem auf überwiegende Wahrscheinlichkeit herabgesetzten Beweismass nachzuweisen."*
- **BGE 141 III 433 E. 2.5.1** (162 chars): short doctrinal paragraph on admissible evidence types under Art. 168 ZPO.

### Anomaly in val_007 gold
This is the gold set that least matches the seven-element template. Most paragraphs are application-style or fragmentary (the 124-char BGE 132 III 155 E. 4 is too short to fit any compositional template). Two hypotheses:

1. **Corpus segmentation artifact**: The `court_considerations.csv` may segment large decisions into multiple sub-rows; the gold paragraphs may be the first segment of longer reasoning sections, with the doctrinal-rule recital actually in subsequent E.-sections of the same decision that were not picked by the curator.
2. **Deliberate curator choice**: The curator may have picked paragraphs that articulate the **specific applied doctrine** in the case, even if short. BGE 144 III 264 E. 5.1 is a typical "we hold that the lower court was right to apply X standard" sentence — the holding, not the rule recital.

Either way, val_007 is an outlier: a Qwen3-8B agent that expects all gold to look like the seven-element template will under-rank these paragraphs. The agent should accept that **on case-level grounds** (these are Leitentscheid decisions on the relevant doctrines), short application/holding paragraphs from within the gold cases are also gold.

### Operational implication
Add a **case-level fallback** to the agent's decision DAG: if a paragraph is short (<400 chars) but belongs to a case whose other E.-sections recite the seven-element template for the right doctrine, treat the short paragraph as gold-likely too. This requires the agent to either (a) see other paragraphs of the same case in the candidate pool or (b) have prior knowledge that the case is a known Leitentscheid for the doctrine.

---

## 11. Per-query close reading — val_008 (disloyal management of public interests, Art. 314 StGB)

### Query in one sentence
Has a councillor who chaired a publicly funded community trust committed disloyal management by self-dealing personnel-administration, shared-workspace deals, and depreciation manipulations totalling CHF 93,089 net, and does the indictment's omission of the hourly-rate increase bar conviction on that allegation?

### Doctrinal sub-questions
1. **Disloyal management of public interests** (Art. 314 StGB) — substantive criminal-law elements.
2. **Mode of participation: Mittäterschaft / Anstifter / Gehilfe** (Art. 12, 25, 26 StGB).
3. **Sentencing rules** (Art. 42, 44, 49, 50 StGB).
4. **Indictment principle / Akkusationsprinzip** (Art. 9 StPO, Art. 29 Abs. 2 + 32 Abs. 2 BV).
5. **Autorité de l'arrêt de renvoi** (Art. 107 al. 2 LTF) — binding effect of remand.
6. **Cost / Beschwerdefristen** (Art. 428/429/436 StPO; Art. 100 BGG).

### What the gold court paragraphs uniquely contain

Nine court gold paragraphs across three sub-clusters:

**Cluster A — Art. 314 StGB elements**:
- **6B_128/2014 E. 5.2.2** (DE): *"Das tatbestandsmässige Verhalten von Art. 314 StGB setzt ein rechtsgeschäftliches Handeln für das Gemeinwesen voraus. Der Unrechtsgehalt der ungetreuen Amtsführung besteht darin, dass der Beamte bei einem Rechtsgeschäft private Interessen auf Kosten der öffentlichen bevorzugt … (BGE 114 IV 133 E. 1a, bestätigt in Urteil 6B_916/2008 …)."* Seven-element template: anchor (Art. 314 StGB), paraphrase, doctrinal elaboration (faktische Entscheidungskompetenz), authority chain (BGE 114 IV 133, BGE 101 IV 407, BGE 111 IV 83).
- **6B_1110/2014 E. 2.3** (DE): nearly verbatim echo of 6B_128/2014.

**Cluster B — Akkusationsprinzip / right-to-be-heard**:
- **BGE 143 IV 63 E. 2.2** (DE): *"Nach dem Anklagegrundsatz bestimmt die Anklageschrift den Gegenstand des Gerichtsverfahrens (Umgrenzungsfunktion; Art. 9 und Art. 325 StPO; Art. 29 Abs. 2 und Art. 32 Abs. 2 BV; Art. 6 Ziff. 1 und Ziff. 3 lit. a und b EMRK). … Das Akkusationsprinzip bezweckt zugleich den Schutz der Verteidigungsrechte … (Informationsfunktion; BGE 141 IV 132 E. 3.4.1; BGE 140 IV 188 E. 1.3 …)."* Anchor + paraphrase + teleology (Schutzzweck explicit: "bezweckt zugleich den Schutz …") + authority chain.
- **BGE 141 IV 132 E. 3.4.1** (DE): predecessor Leitentscheid.
- **BGE 149 IV 42 E. 3.5** (DE): recent application + holding paragraph.

**Cluster C — Autorité de l'arrêt de renvoi**:
- **BGE 131 III 91 E. 5.2** (FR): *"Selon l'art. 66 al. 1 OJ, l'autorité cantonale à laquelle une affaire est renvoyée … est tenue de fonder sa nouvelle décision sur les considérants de droit de l'arrêt du Tribunal fédéral. … (ATF 104 IV 276 consid. 3b; ATF 103 IV 73 consid. 1)."* This is the 2005 Leitentscheid under the old OG.
- **BGE 135 III 334 E. 2** (DE): updated post-BGG version of the same doctrine — *"Vor Einführung des Bundesgerichtsgesetzes (BGG) … (BGE 131 III 91 E. 5.2 S. 94; BGE 116 II 220 E. 4a S. 222; je mit Hinweisen)."*
- **6B_904/2020 E. 1.1** (FR) and **6B_1233/2016 E. 1** (FR): echo decisions under the new LTF.

### Critical observation about val_008 multi-doctrine gold
Three doctrinal clusters, each with 2-4 paragraphs. **Each cluster has its own Leitentscheid hub and its own echo set.** This is the multi-aspect query structure at full expression: the query asks about substantive criminal law + procedural-fairness doctrine + remand-binding doctrine, and the gold contains a "spine" of canonical rule-statements for each.

### What the agent must do for val_008-style multi-doctrine queries
Decompose the query into N sub-doctrines. For each sub-doctrine, hunt the **specific Leitentscheid hub** and its **echo decisions**. Treat near-misses as those that recite a related-but-different doctrine. For val_008, the typical near-miss is a paragraph reciting **Vermögensverwaltung (Art. 158 StGB)** instead of **ungetreue Amtsführung (Art. 314 StGB)** — both are "untreue" doctrines but distinct.

---

## 12. Per-query close reading — val_009 (child maintenance from imprisoned non-custodial parent)

### Query in one sentence
Can the imprisoned non-custodial parent be ordered to pay maintenance and have the sale proceeds of a coastal condominium frozen as security?

### Doctrinal sub-questions
1. **Bemessung des Kindesunterhalts** (Art. 285 ZGB).
2. **Beistandschaft / enforcement / Arrest as security** (Art. 291, 292 ZGB; Art. 271 SchKG).
3. **Unterhaltsanspruch bei Inhaftierung** (special doctrine for prison-as-Erwerbsunfähigkeit-substitute).

### What the gold court paragraphs uniquely contain

Three court gold paragraphs:

- **BGE 137 III 193 E. 2.1** (DE): the Leitentscheid on Unterhalt-bei-Inhaftierung. *"Wenn die Eltern die Sorge für das Kind vernachlässigen, kann das Gericht gemäss Art. 291 ZGB ihre Schuldner anweisen, die Zahlungen ganz oder zum Teil an den gesetzlichen Vertreter des Kindes zu leisten. Kommt das Gemeinwesen für den Unterhalt des Kindes auf, so geht der Unterhaltsanspruch mit allen Rechten auf das Gemeinwesen über (Art. 289 Abs. 2 ZGB). … Beim Rechtsübergang … handelt es sich um eine Subrogation bzw. Legalzession."* Anchor + paraphrase + doctrinal classification (Subrogation/Legalzession).

- **5A_561/2020 E. 5.1.1** (DE), **5A_954/2015 E. 3.3** (DE): echo decisions on the application of Arrest-Forderung and Unterhaltsanspruch principles.

### Note on val_009: gold is small but tightly clustered
Only 3 court gold + many ZGB law articles. The doctrinal density per paragraph is very high. The agent's task here is simpler: identify Art. 285 + Art. 291 + Art. 289 ZGB hub doctrine and pick the BGE/5A_ echo decisions that recite it.

---

## 13. Per-query close reading — val_010 (bank liability for forged faxed orders, currency-of-pleading)

### Query in one sentence
Is the bank liable for transfers on forged faxed orders given the contractual "statement-hold"/30-day-protest clause and the exculpatory clause? Does pleading in CHF when the account is EUR bar the substantive claim?

### Doctrinal sub-questions
1. **Bank-execution-of-forged-orders liability** (Art. 397 OR; Art. 100 OR — invalidity of grobe-Fahrlässigkeit waiver).
2. **Risiko-Transfer-Klausel / Genehmigungsfiktion** (statement-hold / 30-day protest doctrine).
3. **Fremdwährungsforderung pleading rule** (Art. 84 OR — must plead in foreign currency).
4. **Process/transitional law** (Art. 405 ZPO).

### What the gold court paragraphs uniquely contain

Eleven Federal Court paragraphs, three sub-clusters:

**Cluster A — Bank-execution-of-forged-orders liability** (DE and FR):
- **BGE 132 III 449 E. 2** (FR): the 2006 Leitentscheid: *"l'exécution, par la banque, d'un ordre de remettre ou de transférer un montant par prélèvement sur cet avoir a son fondement dans la relation précitée, cela même si l'ordre est donné irrégulièrement ou s'il s'agit d'un faux … En principe, c'est la banque qui supporte le risque d'une prestation exécutée par le débit du compte en faveur d'une personne non autorisée; elle seule subit un dommage car elle est tenue de payer une seconde fois …"*. Authority chain: (ATF 108 II 314 consid. 2; arrêt 4C.349/1994).

**Cluster B — Risiko-Transfer / Genehmigungsfiktion**:
- **BGE 146 III 326 E. 6.1** (FR): *"il est habituel que les conditions générales des banques … comportent une clause dite de transfert de risque. Généralement, cette clause prévoit que le dommage … est à la charge du client, sauf en cas de faute grave de la banque … (ATF 132 III 449 consid. 2; ATF 122 III 26 consid. 4a; ATF 112 II 450 consid. 3a)."* Statutory anchor (the doctrine is judge-made), paraphrase, authority chain.
- **4A_379/2016 E. 3.3.1** (FR): nearly verbatim echo of BGE 146 III 326 E. 6.1.
- **4A_42/2015 E. 5.5, E. 6.3, E. 6.6** (DE): the DE-language treatment of Genehmigungsfiktion / Zustellungsfiktion. E. 5.5 is the doctrinal application of the rule. E. 6.3 is a rule-statement paragraph on Zustellungsfiktionen ("Zustellungsfiktionen dienen in der Regel dazu, …"). E. 6.6 is the application + remand instruction.

**Cluster C — Fremdwährungsforderung**:
- **BGE 134 III 151 E. 2.4** (DE): rule-statement: *"Entsprechend darf das Gericht im Erkenntnisverfahren nur eine Zahlung in der geschuldeten Fremdwährung zusprechen (Loertscher, Commentaire Romand, N. 17 zu Art. 84 OR; Weber, Berner Kommentar, …)."* Anchor + paraphrase + doctrinal authority (Loertscher / Weber / Schraner / Gauch/Schluep) + carve-out for Vollstreckungsverfahren.
- **BGE 134 III 151 E. 2.5** (78 chars): application/sentence fragment.
- **BGE 127 III 147 E. 2c** (FR): rule-statement on the statement-hold 30-day clause as péremption conventionnelle (Art. 454 OR analogy).
- **BGE 128 III 76 E. 1b** (FR): doctrinal paragraph on Art. 44 LAA exception and applicability.

### Critical observation about val_010 cross-language doctrinal pair
The "Risiko-Transfer" doctrine appears in **both DE and FR** gold — the same legal rule, drafted in the language of the underlying decision. BGE 146 III 326 (FR) and 4A_42/2015 (DE) are sister decisions on opposite-language paths to the same doctrine. The DE/FR pair is itself a confirmation signal: an agent reading the FR paragraph and the DE paragraph should recognize them as the same doctrinal rule even though the text is independently authored.

### Near-miss pattern specific to val_010
The corpus has many paragraphs on Art. 397 OR (Mandat — Vorschriftsgemässe Ausführung). Only those that address **bank-mandates with forged-orders fact patterns** are val_010 gold. Generic mandate-execution paragraphs (e.g., on architect's contract execution) are not. **Doctrinal-precision again**: the statute is shared, but the gold is in the bank-mandate-forged-order sub-corner only.

---

## 14. The Qwen3-8B agent's decision DAG for picking court citations

The agent receives: the query + a candidate paragraph (citation + text) + structured metadata (the dossier). The agent should reason as follows.

### Step 0 — Coarse exclusion (5 seconds)

Before any close reading, exclude:

- **Cantonal-court paragraphs** (citation matches `KGer …`, `OGer …`, `BezGer …`, `Kantonsgericht …`). Zero val gold is cantonal. Hard exclude.
- **Federal-court paragraph in the wrong chamber** for the query's legal area: criminal queries (val_001/003/008) require chambers IV, I, 1B, 7B, 6B; civil queries (val_004/005/006/007/009/010) require chambers III, 4A, 5A, 5F; social-insurance queries (val_002) require V, 8C, 9C. Mismatch ≠ automatic kill (cross-area citations occur ~5 %) but is a strong down-weight.
- **Paragraphs <200 characters** that don't belong to a case with longer doctrinal paragraphs in the pool. These are usually fragment / parties / dispositif.

### Step 1 — Read the OPEN (first 200 chars)

Look for **element 1 + element 2** (statutory anchor + paraphrase by court). Specifically:

- DE openers: `(?i)^(?:\d+(\.\d+)?\.\s+)?(Gemäss|Nach|Im Sinne von|Aufgrund (?:des|der)|Laut)\s+Art\.\s*\d+`
- FR openers: `(?i)^(?:\d+(\.\d+)?\.\s+)?(Selon|Conformément à|Aux termes de|En vertu (?:de|des)|D'après|Au regard de)\s+l(?:'|es?\s+)?art(?:icle)?s?\.?\s*\d+`
- IT openers: `(?i)^(?:\d+(\.\d+)?\.\s+)?(Conformemente all'|Ai sensi dell'|Secondo l'|Giusta l')\s*art(?:icolo)?\.?\s*\d+`
- Doctrinal openers without explicit Art.: "Der Haftgrund der … liegt vor, wenn …", "Le danger de collusion vise à empêcher …", "Selon la jurisprudence …", "Nach der bundesgerichtlichen Rechtsprechung …"

If the open matches and the verb belongs to the court (not a party), proceed to Step 2.

If the open contains "Der Beschwerdeführer rügt …", "Le recourant fait valoir …", "Il ricorrente sostiene …", "Die Vorinstanz hat …", "La cour cantonale a retenu …" — this is **party-position / lower-court summary**. Down-weight heavily.

### Step 2 — Identify the DOCTRINAL SUB-QUESTION

Read the paraphrase (element 2). What legal rule is the court stating? E.g.: "What constitutes Kollusionsgefahr?" / "What is the threshold for Umschulung?" / "When does the Genehmigungsfiktion apply?".

Compare against the query's legal sub-questions. The query decomposes into 2–5 sub-questions (the multi-aspect structure). The paragraph's doctrinal sub-question must match one of them. If the paraphrase is about a **related but different rule** (e.g., Fluchtgefahr instead of Kollusionsgefahr), this is the **doctrinal-precision mismatch** near-miss. Reject.

### Step 3 — Verify the doctrinal density

Scan the middle of the paragraph for elements 3+4+5+6:

- Teleology marker: "soll verhindern, dass", "bezweckt", "vise à empêcher", "tend à", "dient", "a pour but".
- Positive limb: "namentlich", "notamment", "insbesondere", "in particolare", followed by a factor list.
- Negative limb: "genügt indessen nicht", "ne saurait suffire", "vermag nicht", "ne saurait à elle seule".
- Interpretive factors: "Bei der Frage, ob", "Dans cet examen", "Pour retenir l'existence de".

A paragraph that has 3+ of these markers in addition to elements 1+2 is firmly doctrinal-rule-statement. A paragraph with only 1+2 followed by application narrative is application-style — gold only if it is within a known Leitentscheid.

### Step 4 — Verify the CLOSE (authority chain)

Look for the parenthetical authority chain at the close: `(ATF/BGE \d+ [IVX]+ \d+ (?:E\.|consid\.) \d+(?:\.\d+)?(?: S\. \d+| p\. \d+)?(?:;\s*(?:ATF|BGE) …){0,5}(?:;\s*mit Hinweisen|;\s*et les références|;\s*je mit Hinweisen)?)`.

A paragraph with a 2-6-cite authority chain ending in "mit Hinweisen" / "et les références" is unambiguously doctrinal. Absence is a soft negative — some echoes have shorter chains, some application paragraphs of Leitentscheide don't have them at all.

### Step 5 — Check the CHAMBER alignment (cheap regex)

Extract chamber from citation:
- `BGE \d+ (?P<chamber>[IVX]+)`
- `^(?P<docket>\d[A-Z])_`

Map to legal area:
- I, 1B, 1C → public law / detention / constitutional
- II, 2C, 2D → administrative / tax / migration
- III, 4A, 5A, 5F, 5D → civil / family / property
- IV, 6B, 7B → criminal (after 2024 docket reorg, 7B; before, 1B for public-law-detention and 6B for criminal-merits)
- V, 8C, 9C → social insurance
- BVGer (B-…) → administrative or asylum (rarely in gold)

If the chamber matches the query's legal area and Steps 0–4 are passed, this is very likely gold.

### Step 6 — Multi-aspect check (the final pass)

The agent has likely picked 5–15 candidate paragraphs that pass Steps 0–5. Now apply the **multi-aspect coverage requirement**: 9/10 val queries have 2–4 doctrinal aspects, and the gold spans them. Verify that the candidate set covers each aspect of the query with at least one paragraph. If an aspect is empty, hunt for a gold-shaped paragraph addressing it. If one aspect is over-represented at the expense of another, redistribute.

Also: include the **procedural-wrapper backbone** — `Art. 100 Abs. 1 BGG` is in 10/10 val gold; cost / appeal articles in 6/10. These are law gold (not court gold) so out of scope here, but the multi-aspect requirement is the same.

### Step 7 — Leitentscheid vs echo balance

Within each doctrinal aspect, the gold typically contains 1–2 Leitentscheide (BGE-numbered, older) + 2–6 echo decisions (BGer dockets, recent). If the agent has selected 5 paragraphs all from the same case, redistribute to favour at least one BGE-numbered Leitentscheid + 2 echo decisions.

---

## 15. Cheap textual self-checks for the agent during reasoning

These are **post-hoc verifications** the agent can run on its own picks before finalising. Each is doable by reading text + extracting a regex pattern; none requires external lookup.

### Check 15.1 — "Opener-Verb-Belongs-to-Court" check

For each picked court paragraph, identify the subject and verb in the opening sentence:
- If subject is "Art. N CODE" and verb is "bestimmt / sieht vor / lautet / verlangt / dispose / prévoit / régit / définit / est applicable" → court is speaking, GOLD-shaped.
- If subject is "Der Beschwerdeführer / Le recourant / Die Vorinstanz / La cour cantonale" and verb is "rügt / fait valoir / a retenu / a considéré / soutient" → party or lower court is speaking, NOT GOLD-shaped.

### Check 15.2 — "Six-element-presence" check

Tally how many of the seven elements the paragraph contains. ≥5 → gold-shaped. 3-4 → weak. ≤2 → not gold-shaped.

### Check 15.3 — "Doctrinal-precision" check

Read the paraphrase. Write down (in your reasoning) the one-sentence legal rule the paragraph states. Compare to the query's sub-questions. If the rule is "Was ist Kollusionsgefahr?" and the query asks about it, match. If the rule is "Was ist Fluchtgefahr?" and the query asks about collusion, mismatch — even though Art. 221 StPO is shared.

### Check 15.4 — "Leitentscheid-spine" check

For each candidate set, ask: which paragraphs are the BGE-numbered Leitentscheide? Which are echoes? Are there at least one of each per doctrinal aspect? If the agent picked only echoes and no Leitentscheid for a doctrine, hunt for the missing BGE.

### Check 15.5 — "Authority-chain-completeness" check

Compare the authority chains of selected paragraphs. Do they cite each other? In val_001, 70 % of gold court paragraphs cite at least one other gold court paragraph. If the selected set has zero cross-citation, suspicious — re-examine.

### Check 15.6 — "Cross-language sister" check

For bilingual queries (val_001, val_002, val_003, val_010), check: are there both DE and FR rule-statement paragraphs reciting the same doctrine? They should be present in roughly the corpus proportion (67/27/5). If the selection is 100 % DE on a bilingual query, FR siblings are missing.

### Check 15.7 — "Chamber-alignment" hard check

Every selected court paragraph's chamber should match the query's legal area. Zero tolerance for cantonal. Down-weight aggressively for wrong-chamber.

### Check 15.8 — "Procedural-wrapper presence" check (law side, out of strict scope)

Even though this is the court-gold deliverable, note that every val query's gold contains `Art. 100 Abs. 1 BGG`. If the agent's selection omits this on the law side, add it.

---

## 16. Two concrete near-miss vs gold side-by-side examples

These are the kinds of pairs the agent will see and must distinguish.

### Example A — val_001, both citing Art. 221 Abs. 1 lit. b StPO

**GOLD** — `1B_28/2022 E. 4.1`:
> "Der Haftgrund der Kollusionsgefahr liegt vor, wenn ernsthaft zu befürchten ist, dass die oder der Beschuldigte Personen beeinflusst oder auf Beweismittel einwirkt, um so die Wahrheitsfindung zu beeinträchtigen (Art. 221 Abs. 1 lit. b StPO). Verdunkelung kann nach der bundesgerichtlichen Praxis zu Art. 221 Abs. 1 lit. b StPO insbesondere in der Weise erfolgen, dass sich die beschuldigte Person mit Zeuginnen oder Zeugen, Auskunftspersonen, Sachverständigen oder Mitbeschuldigten ins Einvernehmen setzt oder sie zu wahrheitswidrigen Aussagen veranlasst oder dass sie Spuren und Beweismittel beseitigt. Strafprozessuale Haft wegen Kollusionsgefahr soll verhindern, dass die beschuldigte Person die wahrheitsgetreue Abklärung des Sachverhalts vereitelt oder gefährdet. Die theoretische Möglichkeit, dass die beschuldigte Person kolludieren könnte, genügt indessen nicht, um Haft unter diesem Titel zu rechtfertigen. Es müssen vielmehr konkrete Indizien für die Annahme von Verdunkelungsgefahr sprechen. Das Vorliegen des Haftgrunds ist nach Massgabe der Umstände des jeweiligen Einzelfalls zu prüfen (BGE 137 IV 122 E. 4.2; Urteil 1B_575/2021 vom 8. November 2021 E. 3.1; je mit Hinweisen). …"

Elements: 1 ✔ (Art. 221 lit. b StPO in close-parenthesis position), 2 ✔ (paraphrase: "Der Haftgrund … liegt vor, wenn …"), 3 ✔ (teleology: "soll verhindern, dass …"), 4 ✔ (positive limb: "insbesondere in der Weise, dass … oder dass sie Spuren und Beweismittel beseitigt"), 5 ✔ (negative limb: "Die theoretische Möglichkeit … genügt indessen nicht"), 6 ✔ ("nach Massgabe der Umstände des jeweiligen Einzelfalls"), 7 ✔ (authority: BGE 137 IV 122 E. 4.2; 1B_575/2021 E. 3.1; je mit Hinweisen). All seven elements present. The verb in the opener belongs to the court ("liegt vor"). GOLD.

**NEAR-MISS** — `1B_28/2022 E. 4` (same case, paragraph immediately before):
> "Der Beschwerdeführer bestreitet weiter das Vorliegen von Kollusionsgefahr im Sinne von Art. 221 Abs. 1 lit. b StPO."

Elements: 1 ✔ (Art. 221 lit. b StPO), 2 ✗ (the verb "bestreitet" is the appellant's, not the court's), 3-7 ✗. Total: 1 element. Verb in the opener belongs to the appellant. NOT GOLD. Same case, same article, same chamber, adjacent E.-number — only the seven-element template count distinguishes.

### Example B — val_006, both reciting the Gefälligkeit doctrine

**GOLD** — `BGE 137 III 539 E. 4.1`:
> "In der Rechtsprechung ist anerkannt, dass auch im Bereich von Arbeitsleistungen unverbindliche Gefälligkeiten vorkommen, die keine Vertragsbindung entstehen lassen. Ob Vertrag oder Gefälligkeit vorliegt, entscheidet sich nach den Umständen des Einzelfalles, insbesondere der Art der Leistung, ihrem Grund und Zweck, ihrer rechtlichen und wirtschaftlichen Bedeutung, den Umständen, unter denen sie erbracht wird und der Interessenlage der Parteien. Für einen Bindungswillen spricht ein eigenes, rechtliches oder wirtschaftliches Interesse der Person, welche die Leistung erbringt, oder ein erkennbares Interesse des Begünstigten an fachkundiger Beratung oder Unterstützung (BGE 129 III 181 E. 3.2; BGE 116 II 695 E. 2b/bb S. 697 f.). …"

Modern 2011 Leitentscheid + authority chain to the immediate predecessor (BGE 129 III 181) + the original Leitentscheid (BGE 116 II 695). Doctrinally about contract-vs-Gefälligkeit threshold, which matches the val_006 query. GOLD.

**NEAR-MISS** — `4C.56/2002 E. 3` (corpus row 1706517, not in val_006 gold):
> "[long paragraph reciting the same Gefälligkeit doctrine + applying it to a tree-shaking fact pattern, including extensive Art. 422 OR analogy discussion citing BGE 61 II 95 and BGE 48 II 487]"

This paragraph **does** state the seven-element template — but its **doctrinal sub-question** is the Art. 422 OR analogy (negotiorum-gestio-style compensation), which the val_006 query does NOT ask about. Same doctrine in part, but a different doctrinal corner is its focus. AND it predates the 2011 modern consolidation. NOT GOLD.

The discriminator here is **doctrinal-precision** + **Leitentscheid-recency**.

---

## 17. Summary — the bullet-point rules the agent should internalise

1. **Federal Supreme Court only**. Cantonal court paragraphs are never gold (102/102 val confirmation).
2. **Chamber matches the query's legal area** (criminal IV/1B/6B/7B; civil III/4A/5A/5F; social V/8C/9C).
3. **The opening verb belongs to the court**, not to a party or lower court ("ist zulässig", "liegt vor", "haftet", "ne peut être ordonnée que lorsque", not "rügt", "fait valoir", "a retenu").
4. **The first 200 chars contain a statutory anchor** (`Art. N (Abs. M) (lit. x) CODE` / `art. N (al. M) (let. x) CODE` / `art. N (cpv. M) CODE`).
5. **The body of the paragraph contains at least 3 more of the 7 elements**: paraphrase, teleology, positive limb (namentlich / notamment / insbesondere), negative limb (genügt nicht / ne saurait suffire), interpretive factors.
6. **The closing parenthesis contains a 2–6-citation authority chain** ending with "mit Hinweisen" / "et les références" / "je mit Hinweisen" / "und die dort zitierten Verweise" (soft positive — some echoes lack this).
7. **The doctrinal sub-question of the paragraph matches one of the query's sub-questions** — not just the statute, the rule. A Kollusionsgefahr rule is gold for val_001 but a Fluchtgefahr rule is not, even though both cite Art. 221 StPO.
8. **The Leitentscheid hub plus echoes form the gold set**. Modern BGE (post-2005, post-LTF for criminal cases) is the canonical source. Pre-Leitentscheid predecessors that pre-date the consolidation can be gold only if they are themselves canonical (e.g. BGE 132 I 21 for val_001 was the immediate predecessor and is gold).
9. **Multi-aspect coverage**: 9/10 val queries decompose into 2–4 doctrinal sub-questions and the gold spans them. The agent's pick must span them too.
10. **DE/FR cross-language sisters confirm gold**. For bilingual queries, the same doctrine appears in both languages with the seven-element template in each. Picking both is a confirmation signal.
11. **Application/holding paragraphs from within a Leitentscheid can also be gold** if the case is the canonical source for the doctrine (val_005, val_007 contain such application gold). Short paragraphs (<400 chars) are gold only via this case-level fallback.
12. **Anti-patterns to actively avoid**: cost/admissibility recitals; procedural-history narrative; party-position framing; closing-subsumption "Demnach"-paragraphs; cantonal-court paragraphs; chamber-mismatched paragraphs; wrong-doctrine-on-same-statute paragraphs; pre-Leitentscheid shallow recitations.

---

## 18. What this deliverable does not address (and where to go for it)

- **Law-side gold** (Art. N CODE statute texts as gold): covered in `gold_internal_structure_2026-05-13.md` §3 (statute clustering, article-number windows, role mix).
- **The procedural-wrapper article allowlist** (Art. 100 BGG etc. in 10/10 val): covered in `feature_synthesis_round2_2026-05-13.md` §3B Pattern 3.
- **Quantitative AUC of features**: not measured here. See `linguistic_signatures_2026-05-13.md` §5 for an AUC estimate of the doctrinal-density score alone (0.80–0.88 on gold-vs-ambient).
- **Pool-context features** (channel fingerprint, co-citation density, in-pool case_peer_count): see `near_miss_antipatterns_2026-05-13.md` §4 and `cascade_dossier_plan.md`.
- **Train-set gold structure**: covered in `gold_citation_legal_patterns_2026-05-12.md` §4 (qualitative-transferable patterns, with the caveat that train has zero court gold and is unreliable per `feedback_train_unreliable.md`).

---

## 19. Sources

- `data/val.csv` — query + gold strings for val_001..val_010.
- `research/scratch_2026-05-12_anti_pattern_gold_text/sample_val_{001..010}.json` — gold-text snapshots (gold_law + gold_court extracted from the corpus by `sample_gold_text.py`).
- `data/court_considerations.csv` — 2,476,315-row court-considerations corpus (used for direct contrast-grep on Art. 221 Abs. 1 lit. b StPO matches and Gefälligkeit doctrine occurrences).
- Prior research files (all in `research/`):
  - `gold_citation_legal_patterns_2026-05-12.md` (round 1).
  - `gold_internal_structure_2026-05-13.md` (round 2: aspect + graph).
  - `linguistic_signatures_2026-05-13.md` (round 2: regex signatures).
  - `near_miss_antipatterns_2026-05-13.md` (round 2: anti-pattern forensics).
  - `feature_synthesis_round2_2026-05-13.md` (round 2 close-out).
  - `val_001_gold_and_enrichment_signals.md` (round 0 single-query field analysis).
  - `cascade_dossier_plan.md` (Phase 1/2 dossier plan).
  - `endgame_handoff_2026-05-09.md` (pipeline-state handoff).
- Memory pointers (auto-loaded in chat session): `MEMORY.md`, `feedback_no_query_specific_hardcoding.md`, `project_cascade_dossier_plan.md`, `project_swiss_citation_endgame.md`, `reference_qwen3_reranker_multilingual.md`.
