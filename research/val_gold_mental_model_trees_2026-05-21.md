# Val Gold Court Citations — Mental-Model Tree Schemas (Explicit)

_For each gold court citation, this section shows the FULL tree diagram of the case it belongs to, with every paragraph classified by role and the gold paragraph(s) explicitly marked._

## Legend

- 📜 **RULE** — Court rule statement (the doctrine)
- 🗣️ **party** — Party position (appellant/respondent argument)
- 🏛️ **lower-court** — Lower-court (Vorinstanz) summary
- ⚖️ **application** — Rule applied to facts (in concreto)
- 🧷 **synthesis** — Closing/recap of a section
- 🔀 **bridge** — Section announcement ("Zu prüfen ist…")
- 🛂 **admissibility** — Admissibility recital
- 🔨 **dispositif** — Operative part of the order
- 💰 **costs** — Cost allocation paragraph
- ✍️ **signature** — End-of-decision boilerplate
- ✂️ **fragment** — CSV parse artifact (continuation)
- 🔢 **header-only** — Numbering line, no body
- ❓ **unclassified** — Heuristic couldn't determine role
- 🎯 **GOLD** — Marked as gold for the relevant val query

---

## val_001

**Query (excerpt):** May a court lawfully order a three‑month extension of pre‑trial detention under Art. 221 Abs. 1 lit. b StPO (risk of collusion) consistent with the principle of proportionality when the accused—detained after an alleged late‑night assault and theft of a courier satchel containing, inter alia, €5,600…

### Gold Citation: `BGE 137 IV 122 E. 6.2`

**Case:** `BGE 137 IV 122` (19 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 137 IV 122
└── Erwägungen (legal reasoning)
│   ├── E. 2       ❓ unclassified   
│   ├── E. 3       ❓ unclassified   
│   │   ├── E. 3.1       🗣️ party          
│   │   ├── E. 3.2       📜 RULE           
│   │   └── E. 3.3       📜 RULE           
│   ├── E. 4
│   │   ├── E. 4.1       🗣️ party          🎯GOLD 
│   │   ├── E. 4.2       📜 RULE           🎯GOLD 
│   │   └── E. 4.3       ❓ unclassified   
│   ├── E. 5       ❓ unclassified   
│   │   ├── E. 5.1       ❓ unclassified   
│   │   ├── E. 5.2       ❓ unclassified   
│   │   └── E. 5.3       📜 RULE           
│   ├── E. 6
│   │   ├── E. 6.1       ❓ unclassified   
│   │   ├── E. 6.2       📜 RULE           🎯GOLD   ← SAMPLED
│   │   ├── E. 6.3       📜 RULE           
│   │   ├── E. 6.4       ❓ unclassified   🎯GOLD 
│   │   └── E. 6.5       🧷 synthesis      
│   ├── E. 10
│   │   └── E. 10.00     ✂️ fragment       
│   └── E. 13
│       └── E. 13.00     ✂️ fragment       
```

**Sampled paragraph text (first 400 chars):**

> Nach Art. 237 Abs. 1 StPO ordnet das zuständige Gericht an Stelle der Untersuchungs- oder der Sicherheitshaft eine oder mehrere mildere Massnahmen an, wenn sie den gleichen Zweck wie die Haft erfüllen. Als Ersatzmassnahmen kommen gemäss Art. 237 Abs. 2 StPO namentlich in Frage: a. die Sicherheitsleistung; b. die Ausweis- und Schriftensperre; c. die Auflage, sich nur oder sich nicht an einem bestim…

### Gold Citation: `1B_210/2023 E. 4.1`

**Case:** `1B_210/2023` (16 paragraphs in corpus)

**Mental Model Schema:**

```
1B_210/2023
├── Sachverhalt (facts)
│   ├── A       ❓ unclassified   
│   ├── B       ❓ unclassified   
│   ├── C       ❓ unclassified   
└── Erwägungen (legal reasoning)
│   ├── E. 1       ❓ unclassified   
│   ├── E. 2       ❓ unclassified   
│   ├── E. 3       🗣️ party          
│   ├── E. 4       🗣️ party          
│   │   ├── E. 4.1       ❓ unclassified   🎯GOLD   ← SAMPLED
│   │   ├── E. 4.2       ⚖️ application    
│   │   └── E. 4.3       ❓ unclassified   
│   └── E. 5       🔨 dispositif     
```

**Sampled paragraph text (first 400 chars):**

> 4.1. Conformément à l'art. 221 al. 1 let. b CPP, la détention provisoire ou pour motifs de sûreté ne peut être ordonnée que lorsque le prévenu est fortement soupçonné d'avoir commis un crime ou un délit et qu'il y a sérieusement lieu de craindre qu'il compromette la recherche de la vérité en exerçant une influence sur des personnes ou en altérant des moyens de preuve. Selon la jurisprudence, il pe…

### Gold Citation: `BGE 137 IV 122 E. 6.4`

**Case:** `BGE 137 IV 122` (19 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 137 IV 122
└── Erwägungen (legal reasoning)
│   ├── E. 2       ❓ unclassified   
│   ├── E. 3       ❓ unclassified   
│   │   ├── E. 3.1       🗣️ party          
│   │   ├── E. 3.2       📜 RULE           
│   │   └── E. 3.3       📜 RULE           
│   ├── E. 4
│   │   ├── E. 4.1       🗣️ party          🎯GOLD 
│   │   ├── E. 4.2       📜 RULE           🎯GOLD 
│   │   └── E. 4.3       ❓ unclassified   
│   ├── E. 5       ❓ unclassified   
│   │   ├── E. 5.1       ❓ unclassified   
│   │   ├── E. 5.2       ❓ unclassified   
│   │   └── E. 5.3       📜 RULE           
│   ├── E. 6
│   │   ├── E. 6.1       ❓ unclassified   
│   │   ├── E. 6.2       📜 RULE           🎯GOLD 
│   │   ├── E. 6.3       📜 RULE           
│   │   ├── E. 6.4       ❓ unclassified   🎯GOLD   ← SAMPLED
│   │   └── E. 6.5       🧷 synthesis      
│   ├── E. 10
│   │   └── E. 10.00     ✂️ fragment       
│   └── E. 13
│       └── E. 13.00     ✂️ fragment       
```

**Sampled paragraph text (first 400 chars):**

> Eine Eingrenzung auf ein bestimmtes Gebiet kommt, wie dargelegt (vgl. E. 6.2 hiervor), primär bei Fluchtgefahr in Betracht. Geht es demgegenüber darum, einer Kollusionsgefahr in Form der möglichen Beeinflussung des mutmasslichen Opfers zu begegnen, dürfte in aller Regel eine Ausgrenzung als mildere Massnahme genügen. Dies ist auch vorliegend der Fall. Die Verpflichtung, den BGE 137 IV 122 S. 133 K…

---

## val_002

**Query (excerpt):** A claimant holding a national vocational diploma in warehouse operations worked intermittently as a storage technician from 10 March to 20 September 2022 and was entered as job-seeking on 1 October 2022. From mid-2021 onwards he has suffered from a chronic allergic respiratory disorder (eosinophilic…

### Gold Citation: `BGE 148 V 21 E. 5.3`

**Case:** `BGE 148 V 21` (10 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 148 V 21
└── Erwägungen (legal reasoning)
│   ├── E. 2       ❓ unclassified   
│   ├── E. 5
│   │   ├── E. 5.1       ❓ unclassified   
│   │   ├── E. 5.2       ❓ unclassified   
│   │   └── E. 5.3       ❓ unclassified   🎯GOLD   ← SAMPLED
│   ├── E. 6
│   │   ├── E. 6.1       ❓ unclassified   
│   │   ├── E. 6.2       ❓ unclassified   
│   │   └── E. 6.3       ❓ unclassified   
│   └── E. 2017    ❓ unclassified   
```

**Sampled paragraph text (first 400 chars):**

> En l'absence de dispositions transitoires, l'art. 21 LPC s'applique immédiatement, dès le jour de son entrée en vigueur aux organes chargés de recevoir et d'examiner les demandes, puis de fixer et de verser les prestations (cf. ATF 142 V 67 consid. 3.1; ATF 129 V 115 consid. 2.2 et les arrêts cités; cf. aussi arrêt 9C_972/2009 du 21 janvier 2011 consid. 2.2). Selon une jurisprudence constante, le …

### Gold Citation: `8C_160/2016 E. 4.1`

**Case:** `8C_160/2016` (23 paragraphs in corpus)

**Mental Model Schema:**

```
8C_160/2016
├── Sachverhalt (facts)
│   ├── A       ❓ unclassified   
│   ├── B       ❓ unclassified   
│   ├── C       ❓ unclassified   
└── Erwägungen (legal reasoning)
│   ├── E. 1       ❓ unclassified   
│   ├── E. 2       ❓ unclassified   
│   ├── E. 3       💰 costs          
│   │   ├── E. 3.1       📜 RULE           
│   │   ├── E. 3.2       🏛️ lower-court    
│   │   ├── E. 3.3       🗣️ party          
│   │   ├── E. 3.4       ❓ unclassified   
│   │   └── E. 3.5       ⚖️ application    
│   ├── E. 4       ❓ unclassified   
│   │   ├── E. 4.1       📜 RULE           🎯GOLD   ← SAMPLED
│   │   ├── E. 4.2       🏛️ lower-court    
│   │   └── E. 4.4       ❓ unclassified   
│   ├── E. 5       💰 costs          
│   │   ├── E. 5.1       ❓ unclassified   
│   │   └── E. 5.2       ❓ unclassified   
│   ├── E. 6       ❓ unclassified   
│   └── E. 7       💰 costs          
```

**Sampled paragraph text (first 400 chars):**

> 4.1. Pour évaluer le taux d'invalidité, le revenu que l'assuré aurait pu obtenir s'il n'était pas invalide est comparé avec celui qu'il pourrait obtenir en exerçant l'activité qui peut raisonnablement être exigée de lui après les traitements et les mesures de réadaptation, sur un marché du travail équilibré (art. 16 LPGA). Pour fixer ce taux, l'administration - ou le juge s'il y a recours - a beso…

### Gold Citation: `BGE 144 V 427 E. 3.2`

**Case:** `BGE 144 V 427` (6 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 144 V 427
└── Erwägungen (legal reasoning)
│   ├── E. 2       ❓ unclassified   
│   ├── E. 3
│   │   ├── E. 3.1       ❓ unclassified   
│   │   ├── E. 3.2       📜 RULE           🎯GOLD   ← SAMPLED
│   │   └── E. 3.3       📜 RULE           
│   └── E. 4
│       ├── E. 4.1       ❓ unclassified   
│       └── E. 4.2       ❓ unclassified   
```

**Sampled paragraph text (first 400 chars):**

> Der Sozialversicherungsprozess ist vom Untersuchungsgrundsatz beherrscht. Danach hat das Gericht von Amtes wegen für die richtige und vollständige Feststellung des rechtserheblichen Sachverhalts zu sorgen. Die Verwaltung als verfügende Instanz und - im Beschwerdefall - das Gericht dürfen eine Tatsache nur dann als bewiesen annehmen, wenn sie von ihrem Bestehen überzeugt sind. Im Sozialversicherung…

---

## val_003

**Query (excerpt):** A. Rivera, a Peruvian national born in 1994 and with no prior convictions in the forum state, is accused of having, between 5 March and 9 March 2024, together with three accomplices (B. L., C. M. and D. S.) taken part in a series of offenses including the theft and driving off of delivery vans, forc…

### Gold Citation: `BGE 145 I 167 E. 4.1`

**Case:** `BGE 145 I 167` (8 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 145 I 167
└── Erwägungen (legal reasoning)
│   ├── E. 3       ❓ unclassified   
│   ├── E. 3a      ✂️ fragment       
│   └── E. 4       ❓ unclassified   
│       ├── E. 4.1       ❓ unclassified   🎯GOLD   ← SAMPLED
│       ├── E. 4.2       ❓ unclassified   
│       ├── E. 4.3       ❓ unclassified   
│       └── E. 4.4       ⚖️ application    
```

**Sampled paragraph text (first 400 chars):**

> Le droit d'être entendu garanti par l'art. 29 al. 2 Cst. comprend notamment le droit pour l'intéressé de s'exprimer sur les éléments pertinents avant qu'une décision ne soit prise touchant sa situation juridique, d'avoir accès au dossier, de produire des preuves pertinentes, d'obtenir qu'il soit donné suite à ses offres de preuves pertinentes, de participer à l'administration des preuves essentiel…

### Gold Citation: `1B_192/2022 E. 4.1.2`

**Case:** `1B_192/2022` (20 paragraphs in corpus)

**Mental Model Schema:**

```
1B_192/2022
├── Sachverhalt (facts)
│   ├── A       ❓ unclassified   
│   ├── B       💰 costs          
└── Erwägungen (legal reasoning)
│   ├── E. 1       ❓ unclassified   
│   ├── E. 2       ❓ unclassified   
│   ├── E. 3       🗣️ party          
│   │   └── E. 3.1       ❓ unclassified   
│   ├── E. 4       ❓ unclassified   
│   │   ├── E. 4.1       🔢 header-only    
│   │   ├── E. 4.1.1     📜 RULE           
│   │   ├── E. 4.1.2     ❓ unclassified   🎯GOLD   ← SAMPLED
│   │   ├── E. 4.2       🔢 header-only    
│   │   ├── E. 4.2.1     ❓ unclassified   
│   │   ├── E. 4.2.2     ⚖️ application    
│   │   └── E. 4.3       🧷 synthesis      
│   └── E. 5       💰 costs          
```

**Sampled paragraph text (first 400 chars):**

> 4.1.2. Certes, le recourant semble intégré professionnellement en Suisse où il vit depuis de nombreuses années; l'arrêt attaqué constate en outre qu'il a toujours répondu aux convocations qui lui ont été adressées lors de l'instruction pénale, respectivement qu'il n'a jamais tenté de se soustraire à la justice. Ces éléments doivent toutefois être mis en balance avec sa condamnation en première ins…

### Gold Citation: `1B_195/2022 E. 2.2.1`

**Case:** `1B_195/2022` (18 paragraphs in corpus)

**Mental Model Schema:**

```
1B_195/2022
├── Sachverhalt (facts)
│   ├── A       ❓ unclassified   
│   ├── B       ❓ unclassified   
└── Erwägungen (legal reasoning)
│   ├── E. 1       ❓ unclassified   
│   │   ├── E. 1.1       ❓ unclassified   
│   │   └── E. 1.2       ❓ unclassified   
│   ├── E. 2       💰 costs          
│   │   ├── E. 2.1       🗣️ party          
│   │   ├── E. 2.1.1     ❓ unclassified   
│   │   ├── E. 2.1.2     ❓ unclassified   
│   │   ├── E. 2.2       🗣️ party          
│   │   ├── E. 2.2.1     ❓ unclassified   🎯GOLD   ← SAMPLED
│   │   ├── E. 2.2.2     🗣️ party          
│   │   ├── E. 2.3       ❓ unclassified   
│   │   └── E. 2.4       ❓ unclassified   
│   └── E. 3       ❓ unclassified   
```

**Sampled paragraph text (first 400 chars):**

> 2.2.1. Conformément à l'art. 221 al. 1 let. a CPP, la détention pour des motifs de sûreté peut être ordonnée s'il y a sérieusement lieu de craindre que le prévenu se soustraie à la procédure pénale ou à la sanction prévisible en prenant la fuite. Selon la jurisprudence, le risque de fuite doit s'analyser en fonction d'un ensemble de critères, tels que le caractère de l'intéressé, sa moralité, ses …

---

## val_004

**Query (excerpt):** Mr. Dalton, born in 1941 and resident in a small lakeside town near Thun, executed a handwritten will on 10 October 1997 stating that he left his entire estate to his partner Ms. Lang and, should she predecease him, to his granddaughters Anna (born 1988) and Bella (born 1991). Mr. Dalton and Ms. Lan…

### Gold Citation: `BGE 131 III 601 E. 3.1`

**Case:** `BGE 131 III 601` (4 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 131 III 601
└── Erwägungen (legal reasoning)
│   └── E. 3       ❓ unclassified   
│       ├── E. 3.1       📜 RULE           🎯GOLD   ← SAMPLED
│       ├── E. 3.2       🏛️ lower-court    
│       └── E. 3.3       ❓ unclassified   
```

**Sampled paragraph text (first 400 chars):**

> Aux termes de l'art. 505 al. 1 CC, le testament olographe est écrit en entier, daté et signé de la main du testateur; la date consiste dans la mention de l'année, du mois et du jour où l'acte a été dressé. L'art. 520 al. 1 CC prévoit que les dispositions entachées d'un vice de forme sont annulées. BGE 131 III 601 S. 604 Le testament peut revêtir, comme en l'espèce, la forme d'une lettre (ATF 57 II…

---

## val_005

**Query (excerpt):** A parent, separated from their co-parent since 2008, has not had custody of the two children (primary custody is with the other parent); until March 2020 the parent exercised a longstanding visitation pattern (Wednesday evenings, alternate weekends including overnight stays, and portions of school h…

### Gold Citation: `BGE 131 III 209 E. 5`

**Case:** `BGE 131 III 209` (3 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 131 III 209
└── Erwägungen (legal reasoning)
│   ├── E. 3       ❓ unclassified   
│   ├── E. 4       ❓ unclassified   
│   └── E. 5       📜 RULE           🎯GOLD   ← SAMPLED
```

**Sampled paragraph text (first 400 chars):**

> Das Obergericht hat es im Wesentlichen bei einem Verweis auf die kantonale Praxis, wonach das Besuchsrecht bei elterlichen Konflikten einzuschränken sei, bewenden lassen. Das Abstellen auf eine solche Praxis muss jedoch mit der in BGE 130 III 585 publizierten Rechtsprechung für den Fall, dass das Einvernehmen zwischen besuchsberechtigtem Elternteil und Kind gut ist, als überholt gelten. Die beiläu…

### Gold Citation: `BGE 128 III 411 E. 3.2.1`

**Case:** `BGE 128 III 411` (5 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 128 III 411
└── Erwägungen (legal reasoning)
│   ├── E. 3
│   │   ├── E. 3.1       📜 RULE           
│   │   ├── E. 3.2       📜 RULE           
│   │   ├── E. 3.2.1     ❓ unclassified   🎯GOLD   ← SAMPLED
│   │   └── E. 3.2.2     ❓ unclassified   
│   └── E. 11
│       └── E. 11.69     ✂️ fragment       
```

**Sampled paragraph text (first 400 chars):**

> Il faut examiner tout d'abord quelle est la portée de cette maxime, et si le débiteur de la contribution d'entretien peut l'invoquer en sa faveur. BGE 128 III 411 S. 413 Selon le Message, la maxime inquisitoire de l'art. 145 al. 1 CC a la même portée que celle que la jurisprudence avait déduite de l'art. 156 al. 1 aCC (FF 1996 I 148 n. 234.102). Elle doit avoir également le même sens que celle de …

### Gold Citation: `BGE 130 III 585 E. 2.1`

**Case:** `BGE 130 III 585` (7 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 130 III 585
└── Erwägungen (legal reasoning)
│   ├── E. 1       📜 RULE           
│   └── E. 2       ❓ unclassified   
│       ├── E. 2.1       ❓ unclassified   🎯GOLD   ← SAMPLED
│       ├── E. 2.2       ❓ unclassified   
│       ├── E. 2.2.1     ❓ unclassified   🎯GOLD 
│       ├── E. 2.2.2     ❓ unclassified   
│       └── E. 2.3       ❓ unclassified   
```

**Sampled paragraph text (first 400 chars):**

> Eltern, denen die elterliche Sorge oder Obhut nicht zusteht, und das unmündige Kind haben gegenseitig Anspruch auf angemessenen persönlichen Verkehr (Art. 273 Abs. 1 ZGB). Die Vorstellungen darüber, was in durchschnittlichen Verhältnissen als angemessenes Besuchsrecht zu gelten habe, gehen in Lehre und Praxis auseinander, wobei regionale Unterschiede festzustellen sind und eine Tendenz zur Ausdehn…

---

## val_006

**Query (excerpt):** On 3 March 2012, homeowners Ms. L and her partner Mr. M asked G, an installer they knew socially who works on domestic heating equipment, to help free of charge with the removal of a household fuel reservoir and its connected burner located in a ground‑floor utility room. G did not have the federal …

### Gold Citation: `BGE 128 III 419 E. 2.2`

**Case:** `BGE 128 III 419` (13 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 128 III 419
└── Erwägungen (legal reasoning)
│   ├── E. 2
│   │   ├── E. 2.1       ❓ unclassified   
│   │   ├── E. 2.2       ❓ unclassified   🎯GOLD   ← SAMPLED
│   │   ├── E. 2.3       ❓ unclassified   
│   │   ├── E. 2.4       ❓ unclassified   
│   │   ├── E. 2.4.1     ❓ unclassified   
│   │   ├── E. 2.4.2     ❓ unclassified   
│   │   └── E. 2.4.3     ❓ unclassified   
│   ├── E. 1993    ❓ unclassified   
│   └── E. 1999    ❓ unclassified   
```

**Sampled paragraph text (first 400 chars):**

> Après la conclusion du bail à ferme du 10 septembre 1993, qui prévoit une redevance mensuelle de 10'000 fr., les parties ont BGE 128 III 419 S. 422 signé deux autres documents, datés respectivement du 14 septembre 1993 et du 1er janvier 1995, qui portent la redevance mensuelle à 12'000 fr. Le litige qui oppose les parties concerne partiellement l'interprétation de ces deux documents et il convient…

### Gold Citation: `BGE 130 III 417 E. 3.2`

**Case:** `BGE 130 III 417` (15 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 130 III 417
└── Erwägungen (legal reasoning)
│   ├── E. 2       ❓ unclassified   
│   │   ├── E. 2.1       ❓ unclassified   
│   │   ├── E. 2.2.1     ❓ unclassified   
│   │   └── E. 2.2.2     ❓ unclassified   
│   ├── E. 3       🗣️ party          
│   │   ├── E. 3.1       ❓ unclassified   
│   │   ├── E. 3.2       ⚖️ application    🎯GOLD   ← SAMPLED
│   │   ├── E. 3.3       ❓ unclassified   
│   │   └── E. 3.4       ❓ unclassified   
│   ├── E. 4       ❓ unclassified   
│   │   ├── E. 4.1       🏛️ lower-court    
│   │   ├── E. 4.2       ❓ unclassified   
│   │   ├── E. 4.2.1     ❓ unclassified   
│   │   └── E. 4.2.2     ❓ unclassified   
│   └── E. 1998    ❓ unclassified   
```

**Sampled paragraph text (first 400 chars):**

> En l'espèce, il n'apparaît pas que la cour cantonale a pu déterminer la volonté commune et réelle des parties contractantes à l'accord du 4 novembre…

### Gold Citation: `BGE 121 IV 207 E. 2a`

**Case:** `BGE 121 IV 207` (6 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 121 IV 207
└── Erwägungen (legal reasoning)
│   ├── E. 1a      🗣️ party          
│   ├── E. 1b      ❓ unclassified   
│   ├── E. 2a      🗣️ party          🎯GOLD   ← SAMPLED
│   ├── E. 2b      ❓ unclassified   
│   ├── E. 2c      ❓ unclassified   
│   └── E. 3       ❓ unclassified   
```

**Sampled paragraph text (first 400 chars):**

> La recourante soutient qu'en admettant une rupture du rapport de causalité adéquate, la cour cantonale a violé l'art. 125 CP. Selon l'art. 125 al. 1 CP, "celui qui, par négligence, aura fait subir à une personne une atteinte à l'intégrité corporelle ou à la santé sera, sur plainte, puni de l'emprisonnement ou de l'amende". L'art. 125 al. 2 CP prévoit que si la lésion est grave - tel que cela est a…

---

## val_007

**Query (excerpt):** An heirship claims title to a vintage pocket chronometer known as “The Meridian” that belonged to Ms. Barnes, who died in 2010, and contends the timepiece was never validly donated to her visually impaired partner Mr. Collins despite the existence only of a photocopy of a deed of gift dated May 2006…

### Gold Citation: `BGE 139 III 305 E. 5`

**Case:** `BGE 139 III 305` (27 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 139 III 305
└── Erwägungen (legal reasoning)
│   ├── E. 3       ❓ unclassified   
│   │   ├── E. 3.1       ❓ unclassified   
│   │   ├── E. 3.2.1     📜 RULE           
│   │   └── E. 3.2.2     ❓ unclassified   
│   ├── E. 4
│   │   ├── E. 4.1       📜 RULE           
│   │   └── E. 4.2       ❓ unclassified   
│   ├── E. 5       ❓ unclassified   🎯GOLD   ← SAMPLED
│   │   ├── E. 5.1       ❓ unclassified   
│   │   ├── E. 5.2       🗣️ party          
│   │   ├── E. 5.2.1     ❓ unclassified   
│   │   ├── E. 5.2.2     ❓ unclassified   
│   │   ├── E. 5.2.3     ❓ unclassified   
│   │   ├── E. 5.2.4     ❓ unclassified   
│   │   ├── E. 5.2.5     ❓ unclassified   
│   │   ├── E. 5.2.6     ❓ unclassified   
│   │   ├── E. 5.3       🧷 synthesis      
│   │   ├── E. 5.3.1     📜 RULE           
│   │   ├── E. 5.3.2     ❓ unclassified   
│   │   ├── E. 5.3.3     ❓ unclassified   
│   │   ├── E. 5.3.4     📜 RULE           
│   │   ├── E. 5.3.5     ❓ unclassified   
│   │   ├── E. 5.4.1     🏛️ lower-court    
│   │   ├── E. 5.4.2     📜 RULE           
│   │   ├── E. 5.4.3     ❓ unclassified   
│   │   └── E. 5.5       ❓ unclassified   
│   ├── E. 20      ❓ unclassified   
│   └── E. 2009    ❓ unclassified   
```

**Sampled paragraph text (first 400 chars):**

> Für den Fall, dass ein Erwerb von einem Nichtberechtigten vorliege, ist das Obergericht zum Schluss gekommen, dass der Beschwerdegegner den Rechtsmangel weder kannte noch kennen musste. Die getroffenen Vorsichtsmassnahmen hat es als genügend erachtet. BGE 139 III 305 S. 312…

### Gold Citation: `BGE 132 III 155 E. 4`

**Case:** `BGE 132 III 155` (23 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 132 III 155
└── Erwägungen (legal reasoning)
│   ├── E. 2       ❓ unclassified   
│   ├── E. 4       ❓ unclassified   🎯GOLD   ← SAMPLED
│   │   ├── E. 4.1       📜 RULE           
│   │   └── E. 4.2       ⚖️ application    
│   ├── E. 5       ❓ unclassified   
│   │   ├── E. 5.1       📜 RULE           
│   │   └── E. 5.2       ⚖️ application    
│   ├── E. 6       ❓ unclassified   
│   │   ├── E. 6.1       ❓ unclassified   
│   │   ├── E. 6.1.1     ❓ unclassified   
│   │   ├── E. 6.1.2     ❓ unclassified   
│   │   ├── E. 6.1.3     ❓ unclassified   
│   │   ├── E. 6.2       ❓ unclassified   
│   │   ├── E. 6.2.1     📜 RULE           
│   │   ├── E. 6.2.2     ❓ unclassified   
│   │   ├── E. 6.2.3     ❓ unclassified   
│   │   └── E. 6.3       ❓ unclassified   
│   ├── E. 7       ❓ unclassified   
│   │   ├── E. 7.1       ❓ unclassified   
│   │   ├── E. 7.1.1     ❓ unclassified   
│   │   ├── E. 7.1.2     📜 RULE           
│   │   └── E. 7.2       🧷 synthesis      
│   └── E. 1997    ❓ unclassified   
```

**Sampled paragraph text (first 400 chars):**

> Die Kläger behaupten zur Hauptsache, der Besitz - und damit auch das Eigentum - sei durch Besitzanweisung übertragen worden.…

### Gold Citation: `BGE 144 III 264 E. 5.1`

**Case:** `BGE 144 III 264` (26 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 144 III 264
└── Erwägungen (legal reasoning)
│   ├── E. 1       ❓ unclassified   
│   │   └── E. 1.3       ❓ unclassified   
│   ├── E. 2
│   │   ├── E. 2.1       ❓ unclassified   
│   │   ├── E. 2.2       ❓ unclassified   
│   │   └── E. 2.3       ❓ unclassified   
│   ├── E. 5
│   │   ├── E. 5.1       📜 RULE           🎯GOLD   ← SAMPLED
│   │   ├── E. 5.2       ❓ unclassified   
│   │   ├── E. 5.3       ❓ unclassified   
│   │   ├── E. 5.4       ❓ unclassified   
│   │   └── E. 5.5       ❓ unclassified   
│   └── E. 6
│       ├── E. 6.1       ❓ unclassified   
│       ├── E. 6.1.1     ❓ unclassified   
│       ├── E. 6.1.2     ❓ unclassified   
│       ├── E. 6.1.3     ❓ unclassified   
│       ├── E. 6.2.1     ❓ unclassified   
│       ├── E. 6.2.2     📜 RULE           
│       ├── E. 6.2.3     ❓ unclassified   
│       ├── E. 6.3.1     ❓ unclassified   
│       ├── E. 6.3.2     ❓ unclassified   
│       ├── E. 6.3.3     ❓ unclassified   
│       ├── E. 6.4.1     ❓ unclassified   
│       ├── E. 6.4.2     ❓ unclassified   
│       ├── E. 6.4.3     ❓ unclassified   
│       ├── E. 6.4.4     📜 RULE           
│       ├── E. 6.4.5     ❓ unclassified   
│       └── E. 6.5       ❓ unclassified   
```

**Sampled paragraph text (first 400 chars):**

> Das Bundesverwaltungsgericht ist davon ausgegangen, die tatsächlichen Grundlagen der Urteilsunfähigkeit seien mit dem auf überwiegende Wahrscheinlichkeit herabgesetzten Beweismass nachzuweisen. Die Beschwerdeführerin beharrt für den Nachweis der Urteilsunfähigkeit auf einem hohen Grad der Wahrscheinlichkeit und der Beschwerdeführer auf dem strikten Beweis.…

---

## val_008

**Query (excerpt):** Has a member of the town council of the Borough of L., who chaired the board of a publicly funded community trust receiving municipal grants, committed disloyal management of public interests by, during 2015–2020, doing the following: awarding the trust’s personnel administration to a private compan…

### Gold Citation: `BGE 131 III 91 E. 5.2`

**Case:** `BGE 131 III 91` (6 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 131 III 91
└── Erwägungen (legal reasoning)
│   └── E. 5       🗣️ party          
│       ├── E. 5.1       ❓ unclassified   
│       ├── E. 5.2       📜 RULE           🎯GOLD   ← SAMPLED
│       ├── E. 5.2.1     ⚖️ application    
│       ├── E. 5.2.2     ❓ unclassified   
│       └── E. 5.2.3     ❓ unclassified   
```

**Sampled paragraph text (first 400 chars):**

> Selon l'art. 66 al. 1 OJ, l'autorité cantonale à laquelle une affaire est renvoyée peut tenir compte de nouveaux allégués en tant que la procédure civile cantonale le permet, mais elle est tenue de fonder sa nouvelle décision sur les considérants de droit de l'arrêt du Tribunal fédéral. Le juge auquel la cause est renvoyée voit donc sa cognition limitée par les motifs de l'arrêt de renvoi, en ce s…

### Gold Citation: `6B_1233/2016 E. 1`

**Case:** `6B_1233/2016` (18 paragraphs in corpus)

**Mental Model Schema:**

```
6B_1233/2016
├── Sachverhalt (facts)
│   ├── A       ❓ unclassified   
│   ├── C       💰 costs          
└── Erwägungen (legal reasoning)
│   ├── E. 1       ❓ unclassified   🎯GOLD   ← SAMPLED
│   │   ├── E. 1.1       🗣️ party          
│   │   ├── E. 1.3       🔨 dispositif     
│   │   ├── E. 1.4       🗣️ party          
│   │   ├── E. 1.5       🗣️ party          
│   │   └── E. 1.6       🛂 admissibility  
│   ├── E. 2       🗣️ party          
│   ├── E. 3       🗣️ party          
│   │   ├── E. 3.1       ❓ unclassified   
│   │   ├── E. 3.2       ❓ unclassified   
│   │   ├── E. 3.3       ❓ unclassified   
│   │   └── E. 3.4       ❓ unclassified   
│   └── E. 4       ❓ unclassified   
```

**Sampled paragraph text (first 400 chars):**

> L'autorité de l'arrêt de renvoi, que prévoyaient expressément l'art. 66 al. 1 aOJ et l'art. 277ter al. 2 aPPF, est un principe juridique qui demeure applicable sous la LTF (ATF 135 III 334 consid. 2.1 p. 335; cf. message du 28 février 2001 concernant la révision totale de l'organisation judiciaire fédérale, in FF 2001 p. 4143). L'autorité à laquelle la cause est renvoyée par le Tribunal fédéral es…

### Gold Citation: `BGE 135 III 334 E. 2`

**Case:** `BGE 135 III 334` (4 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 135 III 334
└── Erwägungen (legal reasoning)
│   ├── E. 2       📜 RULE           🎯GOLD   ← SAMPLED
│   │   ├── E. 2.1       ❓ unclassified   
│   │   └── E. 2.2       ❓ unclassified   
│   └── E. 4
│       └── E. 4.1.4.5   ✂️ fragment       
```

**Sampled paragraph text (first 400 chars):**

> Vor Einführung des Bundesgerichtsgesetzes (BGG) durfte die kantonale Instanz, an die eine Sache zurückgewiesen wurde, nach Art. 66 Abs. 1 OG neues Vorbringen berücksichtigen, soweit es nach dem kantonalen Prozessrecht noch zulässig war. Die nach kantonalem Prozessrecht zulässigen Noven hatten sich dabei stets innerhalb des rechtlichen Rahmens zu bewegen, den das Bundesgericht mit seinem Rückweisun…

---

## val_009

**Query (excerpt):** A divorced custodial parent lives alone with four children born in 1996, 1998, 2001 and 2006. The non-custodial parent is serving a prison sentence overseas of about seven years (release date uncertain), has failed to pay previously-ordered maintenance and is co-owner of a coastal condominium that i…

### Gold Citation: `BGE 137 III 193 E. 2.1`

**Case:** `BGE 137 III 193` (14 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 137 III 193
└── Erwägungen (legal reasoning)
│   ├── E. 1
│   │   └── E. 1.1       📜 RULE           
│   ├── E. 2
│   │   ├── E. 2.1       📜 RULE           🎯GOLD   ← SAMPLED
│   │   └── E. 2.2       📜 RULE           
│   ├── E. 3       ❓ unclassified   
│   │   ├── E. 3.1       ❓ unclassified   
│   │   ├── E. 3.2       ❓ unclassified   
│   │   ├── E. 3.3       ❓ unclassified   
│   │   ├── E. 3.4       ❓ unclassified   
│   │   ├── E. 3.5       ❓ unclassified   
│   │   ├── E. 3.6       ❓ unclassified   
│   │   ├── E. 3.7       ❓ unclassified   
│   │   ├── E. 3.8       📜 RULE           
│   │   └── E. 3.9       ❓ unclassified   
│   └── E. 322
│       └── E. 322.6     ❓ unclassified   
```

**Sampled paragraph text (first 400 chars):**

> Wenn die Eltern die Sorge für das Kind vernachlässigen, kann das Gericht gemäss Art. 291 ZGB ihre Schuldner anweisen, die Zahlungen ganz oder zum Teil an den gesetzlichen Vertreter des Kindes zu leisten. Kommt das Gemeinwesen für den Unterhalt des Kindes auf, so geht der Unterhaltsanspruch mit allen Rechten auf das Gemeinwesen über (Art. 289 Abs. 2 ZGB). Dies gilt insbesondere, wenn das Gemeinwese…

### Gold Citation: `5A_561/2020 E. 5.1.1`

**Case:** `5A_561/2020` (32 paragraphs in corpus)

**Mental Model Schema:**

```
5A_561/2020
├── Sachverhalt (facts)
│   ├── A       ❓ unclassified   
│   ├── B       ❓ unclassified   
│   ├── C       ❓ unclassified   
└── Erwägungen (legal reasoning)
│   ├── E. 1       🔨 dispositif     
│   │   ├── E. 1.1       ❓ unclassified   
│   │   └── E. 1.2       ❓ unclassified   
│   ├── E. 2       ❓ unclassified   
│   │   ├── E. 2.1       ❓ unclassified   
│   │   └── E. 2.2       ❓ unclassified   
│   ├── E. 3       🗣️ party          
│   │   ├── E. 3.1       ❓ unclassified   
│   │   ├── E. 3.2       ❓ unclassified   
│   │   └── E. 3.3       ❓ unclassified   
│   ├── E. 4       ❓ unclassified   
│   ├── E. 5       ❓ unclassified   
│   │   ├── E. 5.1       🔢 header-only    
│   │   ├── E. 5.1.1     ❓ unclassified   🎯GOLD   ← SAMPLED
│   │   ├── E. 5.1.2     ❓ unclassified   
│   │   ├── E. 5.1.3     ❓ unclassified   
│   │   ├── E. 5.2       📜 RULE           
│   │   ├── E. 5.2.1     ❓ unclassified   
│   │   ├── E. 5.3       🔢 header-only    
│   │   ├── E. 5.3.1     ❓ unclassified   
│   │   ├── E. 5.3.2     ❓ unclassified   
│   │   ├── E. 5.4       ❓ unclassified   
│   │   ├── E. 5.5       🔢 header-only    
│   │   ├── E. 5.5.1     ❓ unclassified   
│   │   └── E. 5.5.2     ❓ unclassified   
│   └── E. 6       💰 costs          
```

**Sampled paragraph text (first 400 chars):**

> 5.1.1. Im Verhältnis zum unmündigen Kind sind besonders hohe Anforderungen an die Ausnützung der eigenen Erwerbskraft zu stellen, zumal in engen wirtschaftlichen Verhältnissen (BGE 144 III 481 E. 4.7.7; 137 III 118 E. 3.1 mit Hinweis; Urteile 5A_946/2018 vom 6. März 2019 E. 3.1; 5A_98/2016 vom 25. Juni 2018 E. 3.4 in fine, in: FamPra.ch 2018 S. 1106; 5A_47/2017 vom 6. November 2017 E. 8.2, nicht p…

### Gold Citation: `5A_954/2015 E. 3.3`

**Case:** `5A_954/2015` (17 paragraphs in corpus)

**Mental Model Schema:**

```
5A_954/2015
├── Sachverhalt (facts)
│   ├── A       ❓ unclassified   
│   ├── B       ❓ unclassified   
│   ├── C       ❓ unclassified   
└── Erwägungen (legal reasoning)
│   ├── E. 1       🔨 dispositif     
│   │   ├── E. 1.1       ❓ unclassified   
│   │   ├── E. 1.2       📜 RULE           
│   │   └── E. 1.3       ❓ unclassified   
│   ├── E. 2       ❓ unclassified   
│   ├── E. 3       ❓ unclassified   
│   │   ├── E. 3.1       ❓ unclassified   
│   │   ├── E. 3.2       ❓ unclassified   
│   │   ├── E. 3.3       ❓ unclassified   🎯GOLD   ← SAMPLED
│   │   └── E. 3.4       ⚖️ application    
│   ├── E. 4       ❓ unclassified   
│   └── E. 5       💰 costs          
```

**Sampled paragraph text (first 400 chars):**

> 3.3. Der Arrest in den Fällen von Art. 271 Abs. 1 Ziff. 1 oder Ziff. 2 SchKG bewirkt jedoch nicht die Fälligkeit erst künftig entstehender Unterhaltsforderungen (FRITZSCHE/WALDER, Schuldbetreibung und Konkurs, Bd. II, 1993, § 56 Rz. 10, mit Hinw. auf die Praxis, die sich auf die Rechtsprechung des Bundesgerichts stützt). Das Gesetz sieht mit Art. 271 Abs. 2 SchKG von der Voraussetzung der Fälligke…

---

## val_010

**Query (excerpt):** A Belize-registered investment vehicle (M) authorized its sole beneficial owner (P) to open and operate a euro account with a private bank in Zurich (Q). P delegated day-to-day portfolio handling to an external adviser (R) who held a power of attorney specifically excluding any authority to make dir…

### Gold Citation: `BGE 132 III 449 E. 2`

**Case:** `BGE 132 III 449` (5 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 132 III 449
└── Erwägungen (legal reasoning)
│   ├── E. 2       ❓ unclassified   🎯GOLD   ← SAMPLED
│   ├── E. 3       ❓ unclassified   
│   ├── E. 4       ❓ unclassified   
│   └── E. 2001    ❓ unclassified   
```

**Sampled paragraph text (first 400 chars):**

> Il n'est pas nécessaire d'examiner de façon détaillée la nature juridique de la relation contractuelle des parties. Il suffit de constater que par l'ouverture du compte des demandeurs, la défenderesse s'est engagée à leur remettre, selon les modalités prévues, tout ou partie de l'avoir disponible (cf. ATF 111 II 263 consid. 1a p. 265). L'exécution, par la banque, d'un ordre de remettre ou de trans…

### Gold Citation: `4A_379/2016 E. 3.3.1`

**Case:** `4A_379/2016` (30 paragraphs in corpus)

**Mental Model Schema:**

```
4A_379/2016
├── Sachverhalt (facts)
│   ├── A       ❓ unclassified   
│   ├── B       ❓ unclassified   
│   ├── C       ❓ unclassified   
└── Erwägungen (legal reasoning)
│   ├── E. 1       🔨 dispositif     
│   │   ├── E. 1.1       ❓ unclassified   
│   │   ├── E. 1.2       ❓ unclassified   
│   │   ├── E. 1.3       ❓ unclassified   
│   │   ├── E. 1.4       ❓ unclassified   
│   │   └── E. 1.5       ❓ unclassified   
│   ├── E. 2       💰 costs          
│   │   ├── E. 2.1       ❓ unclassified   
│   │   └── E. 2.2       ❓ unclassified   
│   ├── E. 3       ❓ unclassified   
│   │   ├── E. 3.1       ❓ unclassified   
│   │   ├── E. 3.2       ❓ unclassified   
│   │   ├── E. 3.2.1     ❓ unclassified   
│   │   ├── E. 3.2.2     ❓ unclassified   
│   │   ├── E. 3.3       ❓ unclassified   
│   │   ├── E. 3.3.1     ❓ unclassified   🎯GOLD   ← SAMPLED
│   │   └── E. 3.3.2     ❓ unclassified   
│   ├── E. 4       ⚖️ application    
│   ├── E. 5
│   │   ├── E. 5.1       ⚖️ application    
│   │   ├── E. 5.2       📜 RULE           
│   │   ├── E. 5.3       ❓ unclassified   
│   │   ├── E. 5.3.1     🏛️ lower-court    
│   │   ├── E. 5.3.2     ❓ unclassified   
│   │   └── E. 5.4       ❓ unclassified   
│   └── E. 6       💰 costs          
```

**Sampled paragraph text (first 400 chars):**

> 3.3.1. Il est habituel que les conditions générales de la banque auxquelles le client adhère, comportent une clause dite de transfert de risque. Généralement, cette clause prévoit que le dommage résultant de défauts de légitimation ou de faux non décelés est à la charge du client, sauf en cas de faute grave de la banque (GUGGENHEIM/GUGGENHEIM, op. cit., n. 339 p. 125). Par l'effet de cette clause,…

### Gold Citation: `BGE 128 III 76 E. 1b`

**Case:** `BGE 128 III 76` (4 paragraphs in corpus)

**Mental Model Schema:**

```
BGE 128 III 76
└── Erwägungen (legal reasoning)
│   ├── E. 1a      📜 RULE           
│   ├── E. 1b      ❓ unclassified   🎯GOLD   ← SAMPLED
│   ├── E. 1c      ❓ unclassified   
│   └── E. 1d      ❓ unclassified   
```

**Sampled paragraph text (first 400 chars):**

> Il n'est fait exception à la règle qui précède que si le droit fédéral contient une norme dont le droit cantonal devait tenir compte et qui délimite les compétences cantonales (cf. ATF 125 III 461 consid. 2; ATF 119 II 297 consid. 4; ATF 115 II 237 consid. 1c; ATF 103 II 75 consid. 1). Il se pose dans ce contexte le problème de l'art. 44 LAA, puisque l'art. 44 al. 2 LAA prévoit que les disposition…
