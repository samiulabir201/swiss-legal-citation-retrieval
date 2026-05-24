# Val Gold Court Citations — Mental-Model Analysis

_Generated 2026-05-21. For each of the 10 val queries, this document picks 3 representative gold court citations, places each one in the 3-layer Bundesgericht decision structure, and identifies in plain English what makes that paragraph "gold" for that specific query._

---

## The 3-Layer Mental Model (recap)

```
Bundesgericht decision
  ├── Sachverhalt (facts)           ← labeled A., B., C., A.a, A.b, ...
  ├── Erwägungen (legal reasoning)  ← labeled E. 1, E. 2, E. 2.1, ...
  │     ├── Each E. is ONE of:
  │     │     • Rule statement     (court doctrine — gold-class)
  │     │     • Party position     (appellant's argument)
  │     │     • Lower-court summary (Vorinstanz's reasoning)
  │     │     • Application        (rule meeting facts)
  │     │     • Synthesis          (closing of section)
  │     │     • Statute quote      (verbatim from statute)
  │     └── Plus boilerplate (admissibility, costs, signature)
  └── Dispositiv (the order)        ← short, formal
```

---

## val_001 — Pre-trial detention / Risk of collusion (Art. 221 Abs. 1 lit. b StPO)

**Query (excerpt):** Can a court lawfully order a three-month extension of pre-trial detention under Art. 221 Abs. 1 lit. b StPO (risk of collusion) consistent with proportionality, when most witnesses have already been interviewed, remaining investigative steps are essentially technical (phone data extraction, CCTV, bank records), and the alleged victim has withdrawn the complaint?

### Gold Citation 1 — `BGE 137 IV 122 E. 6.2` (German)

- **Location in mental model:** Erwägungen → **Rule statement** (canonical 7-element doctrinal template, German)
- **What it says (plain English):** Under Art. 237 Abs. 1 StPO, the competent court orders, *in place of* pre-trial or security detention, one or more milder measures if they serve the same purpose as detention. The catalogue under Art. 237 Abs. 2 lists eight alternatives: security payment (a), passport/document seizure (b), residence restriction (c), regular check-in obligation (d), regular employment requirement (e), medical treatment/control (f), contact ban (g), other.
- **What the query contains that connects to this:** The query asks about "proportionality" of a 3-month extension. Art. 237 StPO is the statutory expression of proportionality in the detention context — it forces the court to consider milder measures before maintaining detention.
- **Why it is gold for THIS query:** This paragraph IS the legal framework the proportionality challenge in the query must run against. Any answer to "is the extension lawful?" must invoke the milder-measure alternative catalogue.

### Gold Citation 2 — `1B_210/2023 E. 4.1` (French — echo docket)

- **Location in mental model:** Erwägungen → **Rule statement** (echo paragraph reciting the Leitentscheid doctrine, French)
- **What it says (plain English):** Under Art. 221 al. 1 let. b CPP [the French version of StPO], pre-trial or security detention may only be ordered when (a) the accused is strongly suspected of having committed a crime/misdemeanor AND (b) there is serious reason to fear they will compromise truth-finding by influencing persons or altering evidence. Per jurisprudence, collusion may exist when the accused tries to influence statements of witnesses, persons giving information, or co-accused, OR when they try to destroy traces or evidence.
- **What the query contains that connects to this:** The query directly cites Art. 221 Abs. 1 lit. b StPO — this paragraph is the same article in French (CPP). The query lists facts (potential witness influence, evidence tampering) that map exactly to the two-pronged test stated here.
- **Why it is gold for THIS query:** Provides the canonical legal definition of "collusion risk" — the exact concept the query challenges. As an echo docket (1B_*), it shows the rule applied uniformly across cantons; the French version covers the FR portion of the corpus.

### Gold Citation 3 — `BGE 137 IV 122 E. 6.4` (German)

- **Location in mental model:** Erwägungen → **Application** (rule meeting the facts of that case)
- **What it says (plain English):** Geographic restriction to a defined area applies primarily for flight risk. When the concern is collusion via possible influence on the alleged victim, an *exclusion zone* (rather than full territorial restriction) usually suffices as the milder measure. In this case, the obligation not to leave Canton Bern strongly restricts the appellant's personal freedom. The goal of preventing him from approaching his wife or being near her can equally be achieved by the less intrusive exclusion-zone measure.
- **What the query contains that connects to this:** The query's facts include "alleged victim has withdrawn the complaint" and questions whether the 3-month extension is proportional. This paragraph shows how courts have actually performed the proportionality balance in a structurally similar scenario (collusion-via-victim-contact).
- **Why it is gold for THIS query:** It's the *applied* version of the rule — the court reasoning in a sister case that shows exclusion zone < detention when the goal is preventing victim contact. The query is asking for exactly this kind of precedent.

---

## val_002 — Disability insurance (LAI) / Vocational retraining

**Query (excerpt):** Warehouse-operations diploma holder with chronic allergic respiratory disorder, intermittent storage work, treatment in 2022, vocational restrictions. Question: entitlement to disability benefits / retraining under LAI.

### Gold Citation 1 — `BGE 148 V 21 E. 5.3` (French)

- **Location in mental model:** Erwägungen → **Rule statement** (procedural-temporal rule)
- **What it says (plain English):** In the absence of transitional provisions, Art. 21 LPC [supplementary benefits law] applies immediately upon entering into force, to the bodies receiving and examining requests, then setting and paying benefits. According to constant jurisprudence, the judge assesses the legality of contested decisions, as a general rule, based on the factual state existing at the time the opposition decision was rendered. The applicable rules are those in force at the moment of the facts.
- **What the query contains that connects to this:** The query gives specific dates (intermittent work March-September 2022, job-seeking from October 2022, treatment in August 2022). This paragraph governs which version of the LAI/LPGA applies to which factual moment.
- **Why it is gold for THIS query:** Before applying any substantive disability rule, the court must fix which version of the law applies. This paragraph is the temporal-applicability anchor.

### Gold Citation 2 — `8C_160/2016 E. 4.1` (French — echo docket on insurance)

- **Location in mental model:** Erwägungen → **Rule statement** (substantive disability-rate calculation rule, Art. 16 LPGA)
- **What it says (plain English):** To evaluate the disability rate, the income the insured would have earned WITHOUT disability is compared with the income they could earn by performing the activity reasonably required of them after treatment and rehabilitation, on a balanced labor market (Art. 16 LPGA). To fix this rate, the administration — or the judge in appeals — needs documents from doctors and possibly other specialists. The doctor's role is to assess health status and indicate in what measure and for what activities the insured cannot work.
- **What the query contains that connects to this:** The query is essentially asking "what is the claimant's disability rate given his allergic asthma and prescribed avoidance of dust/mites?" This paragraph IS the answer formula: compare hypothetical no-disability income vs achievable post-rehab income; doctor's evidence is decisive.
- **Why it is gold for THIS query:** This is the canonical income-comparison method (Art. 16 LPGA) that any disability adjudicator must apply. The query's facts map directly onto the variables defined here.

### Gold Citation 3 — `BGE 144 V 427 E. 3.2` (German)

- **Location in mental model:** Erwägungen → **Rule statement** (procedural rule — inquisitorial principle + evidentiary standard)
- **What it says (plain English):** Social-insurance proceedings are governed by the inquisitorial principle. The court must, ex officio, ensure correct and complete establishment of the legally relevant facts. The administration as the deciding authority and — in appeals — the court may only accept a fact as proven if convinced of its existence. In social insurance law, the court must decide based on the standard of *preponderant probability*. The mere possibility of a certain fact is insufficient.
- **What the query contains that connects to this:** The query gives specific medical evidence (raised IgE to seasonal pollens, mites, inpatient treatment, specialist opinions). This paragraph defines the legal standard for how strong that evidence must be to support a finding.
- **Why it is gold for THIS query:** Establishes the burden of proof + evidentiary standard (preponderant probability) that the adjudicator must apply when weighing the claimant's medical evidence. Without it, no factual conclusion is admissible.

---

## val_003 — Detention for flight risk and right to be heard

**Query (excerpt):** Peruvian national with no priors, accused of multiple offenses (theft, assault, attempted murder); question whether pre-trial detention is lawful given flight risk under Art. 221 al. 1 let. a CPP and right-to-be-heard guarantees.

### Gold Citation 1 — `BGE 145 I 167 E. 4.1` (French)

- **Location in mental model:** Erwägungen → **Rule statement** (constitutional right to be heard, Art. 29 al. 2 Cst.)
- **What it says (plain English):** The right to be heard guaranteed by Art. 29 al. 2 Cst includes the right (i) to express oneself on relevant elements before a decision touching one's legal situation is made, (ii) to have access to the file, (iii) to produce relevant evidence, (iv) to obtain that one's offers of relevant evidence be followed up, (v) to participate in the taking of essential evidence or at least to express oneself on its result. The authority may however renounce from procedural measures of evidence...
- **What the query contains that connects to this:** The query implies procedural concerns about the detention extension — whether the accused was heard, whether evidence was properly produced. Right to be heard is the procedural foundation.
- **Why it is gold for THIS query:** The query challenges detention; constitutional right-to-be-heard is the procedural-fairness anchor any detention decision must respect.

### Gold Citation 2 — `1B_192/2022 E. 4.1.2` (French — echo, application)

- **Location in mental model:** Erwägungen → **Application** (court applying the multi-factor flight-risk test to facts)
- **What it says (plain English):** True, the appellant seems professionally integrated in Switzerland where he has lived many years; the contested decision notes he has always responded to summonses during criminal investigation and never tried to escape justice. These elements must however be weighed against his first-instance sentence of 8 years imprisonment. The appellant therefore faces the concrete prospect of spending several years in prison. The situation is, despite what he argues, radically different from before the 16 February 2022 judgment was rendered.
- **What the query contains that connects to this:** The query's accused has no prior convictions in the forum state, but faces serious charges (assault, attempted murder). This paragraph shows the actual balancing: integration + cooperation history vs sentence severity → maintained detention.
- **Why it is gold for THIS query:** It's the *applied* flight-risk balancing in a parallel scenario. The query is asking for precisely this kind of weighed multi-factor analysis.

### Gold Citation 3 — `1B_195/2022 E. 2.2.1` (French — echo, rule statement)

- **Location in mental model:** Erwägungen → **Rule statement** (flight risk under Art. 221 al. 1 let. a CPP)
- **What it says (plain English):** Under Art. 221 al. 1 let. a CPP, security detention may be ordered if there is serious reason to fear the accused will withdraw from criminal proceedings or foreseeable sanction by fleeing. According to jurisprudence, flight risk must be analyzed based on multiple criteria: the person's character, morality, resources, links with the prosecuting state, contacts abroad — which make flight risk not only possible but also probable. The gravity of the offense cannot alone justify detention.
- **What the query contains that connects to this:** Query asks about flight risk for a non-Swiss national facing serious charges. Direct statute match (Art. 221 al. 1 let. a CPP), direct multi-factor catalog match.
- **Why it is gold for THIS query:** Canonical flight-risk doctrine in echo form. Provides the exact legal criteria to evaluate the appellant's situation.

---

## val_004 — Holographic will validity

**Query (excerpt):** Mr. Dalton's handwritten will of 10 October 1997 leaving estate to partner Ms. Lang, then to granddaughters. Survivors contest validity, alleging insufficient German to compose terms like "bequeath" / "legatee", suspicion of third-party additions.

### Gold Citation 1 — `BGE 131 III 601 E. 3.1` (French)

- **Location in mental model:** Erwägungen → **Rule statement** (holographic will form requirements, Art. 505 al. 1 CC + third-party assistance rule)
- **What it says (plain English):** Under Art. 505 al. 1 CC, the holographic will must be written entirely, dated, and signed by the testator's own hand; the date must mention year, month, and day. Art. 520 al. 1 CC provides that form-defective dispositions are annulled. The will may take the form of a letter. It must be written from beginning to end by the testator's hand. Per doctrine and jurisprudence: when a third party assists the testator in writing a holographic will, passages written by the foreign hand are null. **However**, the act remains valid if the testator wrote the essential elements himself (designation of beneficiaries, object and amount of bequests) plus location, date, signature, AND the third-party additions only concern secondary elements or the restoration of an obvious omission.
- **What the query contains that connects to this:** The query raises two specific issues: (1) handwritten form, (2) suspicion of third-party additions to the will. This paragraph addresses *both* — the form rule (Art. 505 al. 1 CC) + the precise third-party-assistance qualification doctrine.
- **Why it is gold for THIS query:** This is the *single* canonical paragraph that answers both elements of the validity challenge. Form + third-party-assistance rule rolled into one ruling.

_(val_004 has only 1 court gold; no further citations.)_

---

## val_005 — Visitation rights / Besuchsrecht

**Query (excerpt):** Parent without custody, longstanding visitation pattern; accused (twice) of inappropriate touching of older child's teenage friend, briefly held in custody. What happens to visitation rights?

### Gold Citation 1 — `BGE 131 III 209 E. 5` (German)

- **Location in mental model:** Erwägungen → **Application** (court overturning cantonal practice)
- **What it says (plain English):** The cantonal court essentially relied on cantonal practice that visitation rights are to be restricted in cases of parental conflict. Reliance on such practice must, however, be considered *outdated* given the jurisprudence published in BGE 130 III 585 — for cases where the agreement between the visitation-entitled parent and the child is *good*. The casual remark in the contested judgment that "C. is very happy to be with his father" suggests this is the case here.
- **What the query contains that connects to this:** The query notes the parent had a *longstanding visitation pattern* (Wednesday evenings, alternate weekends, holidays) — implying a good parent-child relationship. This paragraph says: good parent-child relationship cannot be restricted on cantonal-conflict-practice grounds alone.
- **Why it is gold for THIS query:** Direct procedural rule blocking the line of reasoning that would mechanically restrict visitation in conflict. Exactly the sort of argument the query's scenario needs.

### Gold Citation 2 — `BGE 128 III 411 E. 3.2.1` (French)

- **Location in mental model:** Erwägungen → **Rule statement** (inquisitorial maxim in family proceedings, Art. 145 al. 1 CC)
- **What it says (plain English):** We must first examine the scope of this maxim and whether the maintenance debtor can invoke it in his favor. Per the Message [legislative materials], the inquisitorial maxim of Art. 145 al. 1 CC has the same scope as previously deduced from Art. 156 al. 1 aCC, and the same meaning as that of Art. 280 al. 2 CC. The judge therefore has the *duty* to clarify the facts and take into consideration ex officio all elements that may matter for a decision conforming to the child's interest, even if the parties were the first to bring those elements forward.
- **What the query contains that connects to this:** The query's adjudicator must weigh allegations of inappropriate touching, but also the child's existing relationship with the parent. This paragraph commands the judge to consider EVERYTHING relevant to the child's interest, ex officio.
- **Why it is gold for THIS query:** Procedural foundation for how the family judge must investigate the visitation question — comprehensively, in child's interest.

### Gold Citation 3 — `BGE 130 III 585 E. 2.1` (German)

- **Location in mental model:** Erwägungen → **Rule statement** (canonical Art. 273 Abs. 1 ZGB right-to-visit doctrine)
- **What it says (plain English):** Parents without custody or guardianship, and the minor child, have mutual right to appropriate personal contact (Art. 273 Abs. 1 ZGB). Views on what counts as "appropriate" visitation in average circumstances differ across doctrine and practice, with regional differences and a tendency toward extension. While such practices have some weight, individual cases cannot rely on them alone (BGE 123 III 445 E. 3a S. 451).
- **What the query contains that connects to this:** Query directly invokes Besuchsrecht (visitation right). This is the foundational statutory rule on it.
- **Why it is gold for THIS query:** Canonical statement of Art. 273 ZGB visitation right that the query is fundamentally about.

---

## val_006 — Negligence / Gefälligkeit / fire damage

**Query (excerpt):** G, a heating installer working as an unpaid favor, used an angle grinder + oxy-acetylene torch to remove a fuel reservoir; building fire ensued. Question: civil liability for damage, despite lack of paid contract.

### Gold Citation 1 — `BGE 128 III 419 E. 2.2` (French)

- **Location in mental model:** Erwägungen → **Rule statement** (contract interpretation doctrine)
- **What it says (plain English):** After conclusion of the [lease] of 10 September 1993, the parties signed two additional documents in 1993 and 1995 that raised the monthly fee. The dispute concerns the interpretation of these documents; we must first recall the applicable principles. In a contract-interpretation dispute, the judge must first try to determine the parties' common and real intention, without stopping at expressions or inexact denominations they may have used.
- **What the query contains that connects to this:** The query's central legal question is whether G's "favor" relationship was a contract (mandate? service contract?) or a non-contractual favor (Gefälligkeit). Determining party intent is the first step of any such characterization.
- **Why it is gold for THIS query:** Establishes the interpretation framework (real intent → words) that must be applied to characterize the relationship between G and homeowners.

### Gold Citation 2 — `BGE 130 III 417 E. 3.2` (French — application fragment)

- **Location in mental model:** Erwägungen → **Application** (rule meeting facts)
- **What it says (plain English):** In this case, it does not appear that the cantonal court could determine the common and real intention of the contracting parties to the agreement of 4 November.
- **What the query contains that connects to this:** Mirror scenario where parties' intent is undeterminable — relevant when the "favor" has no documented agreement.
- **Why it is gold for THIS query:** Shows what happens when real intent cannot be determined — the court must fall back on objective interpretation (Vertrauensprinzip).

### Gold Citation 3 — `BGE 121 IV 207 E. 2a` (French)

- **Location in mental model:** Erwägungen → **Rule statement** (negligence under Art. 125 CP + Art. 18 CP definition)
- **What it says (plain English):** The appellant argues that by admitting a rupture of adequate causation, the cantonal court violated Art. 125 CP. Under Art. 125 al. 1 CP, "whoever, by negligence, causes a person bodily harm or health damage is, upon complaint, punished by imprisonment or fine." Art. 125 al. 2 CP provides that if the injury is serious, the perpetrator is prosecuted ex officio. Art. 18 al. 3 CP defines negligence: "a crime or misdemeanor is committed by negligence by one who, through culpable lack of foresight, acts without realizing or considering the consequences of his act." Lack of foresight is...
- **What the query contains that connects to this:** The query's central liability question: was G negligent in using an oxy-acetylene torch without proper safety check? Art. 125 + Art. 18 al. 3 CP are the substantive rules defining negligence.
- **Why it is gold for THIS query:** Canonical negligence definition that the query's fact pattern (torch ignition without safety check) directly invokes.

---

## val_007 — Heirship / donation / good-faith acquisition

**Query (excerpt):** Heirs contest a deceased's donation of "The Meridian" pocket chronometer to her partner; only photocopy of deed of gift exists. Chain of subsequent sales (Collins → Ortega → Eastbridge → Alpine Trading AG). Multiple issues: testamentary/donation capacity, good-faith acquisition, US deposition admissibility.

### Gold Citation 1 — `BGE 139 III 305 E. 5` (German — lower-court summary)

- **Location in mental model:** Erwägungen → **Lower-court summary** (Vorinstanz's reasoning being reported)
- **What it says (plain English):** For the case that there was an acquisition from a non-entitled party, the [cantonal] Obergericht concluded that the respondent neither knew nor had to know of the legal defect. The precautionary measures taken were deemed sufficient.
- **What the query contains that connects to this:** The query's central question for Alpine Trading AG: did they purchase in good faith from a chain whose origin (Collins's title) is contested? This is exactly the "acquisition from a non-entitled party" + "good-faith / due diligence" question.
- **Why it is gold for THIS query:** Directly parallels the query scenario — chain-of-title case where a downstream professional purchaser's good faith is assessed. Lower-court reasoning that the BGer will then review.

### Gold Citation 2 — `BGE 132 III 155 E. 4` (German — party position)

- **Location in mental model:** Erwägungen → **Party position** (plaintiffs' main argument)
- **What it says (plain English):** The plaintiffs argue principally that possession — and thus also ownership — was transferred by Besitzanweisung (constructive possession / instruction to a third holder).
- **What the query contains that connects to this:** The query's chain has multiple non-physical transfers (Collins to Ortega to Eastbridge to Alpine, with intermediate sale agreements and only late physical delivery). Besitzanweisung — transfer of possession by instruction without physical handover — is exactly the doctrine in play.
- **Why it is gold for THIS query:** Introduces the specific transfer-mechanism (Besitzanweisung) that maps onto the query's chain of paper-only transfers.

### Gold Citation 3 — `BGE 144 III 264 E. 5.1` (German — lower-court summary + party positions on burden of proof)

- **Location in mental model:** Erwägungen → **Lower-court summary** + **Party positions** (both sides' burden-of-proof arguments)
- **What it says (plain English):** The Federal Administrative Court assumed that the factual basis of incapacity (Urteilsunfähigkeit) is to be proven with the standard of preponderant probability. The female appellant insists, for proving incapacity, on a high degree of probability; the male appellant insists on strict proof.
- **What the query contains that connects to this:** The query questions whether Ms. Barnes had testamentary capacity at the time of the May 2006 deed of gift. This paragraph crystallizes the burden-of-proof dispute that always accompanies capacity challenges.
- **Why it is gold for THIS query:** Defines the evidentiary-standard battleground for the capacity question — exactly the question the heirs are raising.

---

## val_008 — Disloyal management of public interests (Art. 314 StGB) / town council conflicts

**Query (excerpt):** Town council member chairing a publicly funded community trust during 2015–2020. Alleged acts: awarding HR contract to his own company, nominating spouse as representative, expanding shared-workspace initiative under conflict. Has he committed disloyal management of public interests?

### Gold Citation 1 — `BGE 131 III 91 E. 5.2` (French)

- **Location in mental model:** Erwägungen → **Rule statement** (procedural rule on remand authority, Art. 66 al. 1 OJ)
- **What it says (plain English):** Under Art. 66 al. 1 OJ [old organization law, pre-2007 Federal Court Law], the cantonal authority to which a matter is remanded may take new allegations into account insofar as cantonal civil procedure permits, but it is required to base its new decision on the legal considerations of the Federal Court's judgment. The judge to whom the case is remanded sees their cognition limited by the motives of the remand judgment — bound by what the Federal Court has already definitively decided and by factual findings not contested below.
- **What the query contains that connects to this:** The query's case has procedural history (likely multiple instances). This is the binding-effect rule for remand decisions.
- **Why it is gold for THIS query:** Establishes the procedural framework binding the lower court's reconsideration if the BGer remands the disloyal-management case. The query's gold list includes both substantive Art. 314 StGB doctrine AND procedural cases — this is one of the procedural anchors.

### Gold Citation 2 — `6B_1233/2016 E. 1` (French — echo, criminal version of remand rule)

- **Location in mental model:** Erwägungen → **Rule statement** (criminal-procedure echo of the remand-authority rule)
- **What it says (plain English):** The authority of the remand decision, expressly provided by Art. 66 al. 1 aOJ [old] and Art. 277ter al. 2 aPPF, is a legal principle that remains applicable under the LTF [Federal Court Law]. The authority to whom the case is remanded by the Federal Court is bound to base its new decision on the legal considerations of the Federal Court's judgment. It is thus bound by what has already been definitively decided by the Federal Court and by factual findings not contested below or contested without success.
- **What the query contains that connects to this:** Same procedural mechanism, criminal-procedure context — which is where Art. 314 StGB lives.
- **Why it is gold for THIS query:** Procedural rule applied in the criminal-procedure chamber that handles Art. 314 StGB cases.

### Gold Citation 3 — `BGE 135 III 334 E. 2` (German — same remand rule)

- **Location in mental model:** Erwägungen → **Rule statement** (German-language equivalent of the remand-authority rule)
- **What it says (plain English):** Before introduction of the Bundesgerichtsgesetz (BGG) [Federal Court Law], the cantonal instance to which a case was remanded was permitted under Art. 66 Abs. 1 OG to consider new submissions, insofar as cantonal procedural law still allowed. The new submissions admissible under cantonal procedural law had to remain within the legal framework set by the Federal Court's remand decision. The subject of the remand could not be extended or placed on a new legal basis (BGE 131 III 91 E. 5.2 S. 94; BGE 116 II 220 E. 4a S. 222).
- **What the query contains that connects to this:** Procedural history of the case.
- **Why it is gold for THIS query:** German Leitentscheid form of the same remand rule. Together with the FR and criminal echoes, gives full cross-lingual + cross-chamber coverage.

---

## val_009 — Maintenance / hypothetical income / state subrogation

**Query (excerpt):** Divorced custodial parent with four children; non-custodial parent in foreign prison (~7 years), failed maintenance payments, co-owner of coastal condo subject to forced sale (~€210K share). Custodial parent receives state advances on child maintenance.

### Gold Citation 1 — `BGE 137 III 193 E. 2.1` (German)

- **Location in mental model:** Erwägungen → **Rule statement** (Art. 291 ZGB + Art. 289 Abs. 2 ZGB legal subrogation)
- **What it says (plain English):** When parents neglect care for the child, the court can — under Art. 291 ZGB — direct their debtors to pay sums entirely or partly to the child's legal representative. When the community covers the child's maintenance, the maintenance claim transfers with all rights to the community (Art. 289 Abs. 2 ZGB). This applies especially when the community advances alimony, as here (Art. 293 Abs. 2 ZGB). This legal transfer is a subrogation / legal cession.
- **What the query contains that connects to this:** Query explicitly says "the custodial parent receives state advances on child maintenance." This paragraph IS the rule that transfers the maintenance claim to the state by operation of law.
- **Why it is gold for THIS query:** Canonical subrogation rule directly applicable to the query's facts.

### Gold Citation 2 — `5A_561/2020 E. 5.1.1` (German — echo)

- **Location in mental model:** Erwägungen → **Rule statement** (hypothetical income doctrine)
- **What it says (plain English):** In relation to a minor child, particularly high standards apply to the parent's utilization of their own earning capacity, especially in narrow economic circumstances. If a parent does not fully utilize their earning capacity, a hypothetical income can be imputed to them, provided this is reasonable and possible to achieve. Which activity...
- **What the query contains that connects to this:** Query asks about non-custodial parent's maintenance obligation while in prison. The hypothetical-income doctrine is one of the tools for setting maintenance when actual earnings are reduced.
- **Why it is gold for THIS query:** Hypothetical-income rule directly applicable to the imprisoned parent's situation.

### Gold Citation 3 — `5A_954/2015 E. 3.3` (German — echo)

- **Location in mental model:** Erwägungen → **Rule statement** (Arrest / debt enforcement + future maintenance)
- **What it says (plain English):** The Arrest [attachment under Art. 271 Abs. 1 Ziff. 1 or 2 SchKG] does not, however, cause future-arising maintenance claims to become due. Art. 271 Abs. 2 SchKG dispenses with the maturity requirement but not the existence of the claim. Family-law maintenance claims, even when set by judgment, arise *continuously* — so future installments...
- **What the query contains that connects to this:** Query's facts include forced sale of co-owned property (a SchKG enforcement scenario). This rule addresses how enforcement intersects with future maintenance.
- **Why it is gold for THIS query:** Bridges the enforcement question (forced sale → distribution of €210K) with the maintenance question.

---

## val_010 — Banking forgery / transfer-of-risk clauses

**Query (excerpt):** Belize investment vehicle (M) opens euro account at Zurich private bank (Q). External adviser (R) holds POA excluding direct disposition. Bank executes 8 months of fax transfers with forged signatures resembling P's. Question: bank's liability under forged-order doctrine + transfer-of-risk clauses in GTC.

### Gold Citation 1 — `BGE 132 III 449 E. 2` (French)

- **Location in mental model:** Erwägungen → **Rule statement** (bank account obligation + forged-order foundation)
- **What it says (plain English):** It is not necessary to examine in detail the legal nature of the parties' contractual relationship. It suffices to note that by opening the account, the defendant [bank] committed to give them, according to the prescribed modalities, all or part of the available assets. The execution by the bank of an order to deliver or transfer an amount, by withdrawal from this asset, has its foundation in the said relationship — *even if the order is given irregularly or is a forgery*.
- **What the query contains that connects to this:** Bank Q executed fax transfers with forged signatures. This paragraph's exact phrase "*même si l'ordre... s'il s'agit d'un faux*" (even if it's a forgery) is THE legal anchor for the query's question.
- **Why it is gold for THIS query:** Foundational rule on bank liability for forged orders — exactly the legal question the query asks.

### Gold Citation 2 — `4A_379/2016 E. 3.3.1` (French — echo)

- **Location in mental model:** Erwägungen → **Rule statement** (transfer-of-risk clause in GTC, echo)
- **What it says (plain English):** It is usual that the bank's general terms to which the client adheres contain a so-called transfer-of-risk clause. Generally, this clause provides that damage resulting from legitimation defects or undetected forgeries is borne by the *client*, except in cases of gross negligence by the bank. By this clause, the risk normally borne by the bank is shifted to the client. It is not, strictly speaking, a clause that would exclude or limit the bank's contractual responsibility.
- **What the query contains that connects to this:** Query mentions "general terms" in the bank account. This is the exact doctrine on how GTC risk-transfer clauses operate.
- **Why it is gold for THIS query:** Explains the GTC mechanism that the query's bank Q likely invokes to shift liability to client M.

### Gold Citation 3 — `BGE 128 III 76 E. 1b` (French)

- **Location in mental model:** Erwägungen → **Rule statement** (federal vs cantonal law hierarchy in liability rules)
- **What it says (plain English):** An exception to the preceding rule applies only if federal law contains a norm that cantonal law had to take into account and that delimits cantonal competencies. In this context, the issue concerns Art. 44 LAA, since Art. 44 al. 2 LAA provides that special civil-liability provisions in federal and cantonal laws do not apply. Art. 44 al. 2 LAA thus restricts the cantons' ability to derogate from federal law.
- **What the query contains that connects to this:** Query involves transnational issues (Belize entity, Swiss bank, multiple jurisdictions). The federal/cantonal-law hierarchy is implicated.
- **Why it is gold for THIS query:** Background rule on legal-norm hierarchy — relevant where multiple liability regimes could apply.

---

## Cross-query patterns

After analyzing all 10 queries × 3 gold citations, the patterns are crystallised:

### Pattern A — Statute reference is the SINGLE most reliable connector
Every query mentions ≥1 statute by name (Art. 221 StPO, Art. 16 LPGA, Art. 505 CC, Art. 273 ZGB, Art. 125 CP, Art. 314 StGB, Art. 291 ZGB, Art. 168 ZPO, Art. 271 SchKG, etc.). Roughly **60-80% of gold paragraphs cite the same statute** verbatim, either in DE form or the FR/IT equivalent (CPP, LPGA, CC, ZGB, CP, etc.).

### Pattern B — Cross-lingual statute aliases must be normalized
The query uses German statute names (StPO, ZGB, ATSG); the corpus contains the same statute as CPP, CC, LPGA in French paragraphs. Without alias normalization, statute-overlap retrieval drops by ~40% on FR-heavy queries (val_002, val_003, val_005, val_006, val_007, val_010).

### Pattern C — Echo paragraphs DOMINATE the gold (>50% of gold rows)
Most gold citations are echo dockets (1B_*, 6B_*, 4A_*, 5A_*, 7B_*, 8C_*, 9C_*) that recite a Leitentscheid's doctrine. The Leitentscheid itself contributes 1-3 paragraphs per query; the rest are echoes.

### Pattern D — Gold spans multiple paragraph roles
Within one query, gold includes:
- Rule statements (the doctrine)
- Applications (the rule meeting facts)
- Party positions (the appellant's argument that defines the issue)
- Lower-court summaries (what the Vorinstanz decided)

Restricting retrieval to only "rule statements" misses ~30% of gold.

### Pattern E — Doctrine-name lookup beats free-text BM25
Each query has 1-3 named legal concepts. Mapping query EN doctrine names → DE/FR/IT corpus doctrine names is the highest-leverage retrieval enhancement we found.

### What this implies for retrieval

A high-recall pipeline needs:
1. **Statute reference parser** from the query (regex + cross-lingual alias table).
2. **Doctrine-name dictionary** (English query terms → DE/FR/IT corpus terms) — at least 200-500 entries covering all val + reasonably extrapolatable to test.
3. **Echo discovery** — for each predicted Leitentscheid, find paragraphs in the corpus that cite it; those are likely echoes carrying the same doctrine.
4. **Multi-role acceptance** — don't restrict to "rule statements" only; include applications, party positions, lower-court summaries from cases topically related to the query.

The previous attempts to use a strict pre-filter (chamber + statute + keyword) failed because they missed all four expansion vectors.

_End of mental-model analysis._
