# Val Gold Court Citations — Mental-Model Trees + Structured JSON (v3)

_For each gold court citation, this section shows:_
_1. The FULL tree of the case it belongs to (every paragraph classified — zero unclassified residue)_
_2. A **structured JSON representation** of the sampled gold paragraph (the canonical schema each row in the corpus would have)_

---

## The canonical JSON schema for every paragraph

Every paragraph in `court_considerations.csv` can be normalised to this shape:

```json
{
  "citation": "string — full canonical citation, e.g. 'BGE 137 IV 122 E. 4.2'",
  "case_id": "string — case identifier without E-id, e.g. 'BGE 137 IV 122'",
  "eid": "string — Erwägung sub-id, e.g. '4.2' or 'A.b'",
  "parent_eid": "string — parent E-id, e.g. '4' for E. 4.2",
  "depth": "int — nesting depth (1 for E. 4, 2 for E. 4.2, 3 for E. 4.2.1)",
  "section": "enum — 'Sachverhalt' | 'Erwägungen' | 'other'",
  "role": "enum — one of 15 roles (rule_statement, party_position, lower_court_summary, ...)",
  "language": "enum — 'de' | 'fr' | 'it' | 'unknown'",
  "char_count": "int — text length in characters",
  "word_count": "int — word count",
  "first_100_chars": "string — opening snippet",
  "statute_refs": "list[string] — extracted statute citations, e.g. ['Art. 221 Abs. 1 lit. b StPO']",
  "statute_ref_count": "int",
  "bge_refs": "list[string] — BGE/ATF/DTF citations in text",
  "bge_ref_count": "int",
  "doctrinal_keywords": "list[string] — matched doctrinal vocabulary, e.g. ['kollusion', 'untersuchungshaft']",
  "template": {
    "el1_statutory_anchor": "bool — opens with 'Gemäss Art./Selon l'art./Ai sensi di...'",
    "el2_paraphrase": "bool|null — court restates rule (needs LLM)",
    "el3_teleology": "bool — purpose statement ('soll verhindern / vise à')",
    "el4_positive_limb": "bool — non-exhaustive list ('namentlich / notamment')",
    "el5_negative_limb": "bool — boundary ('genügt nicht / ne saurait suffire')",
    "el6_interpretive_factors": "bool — multi-factor balancing",
    "el7_authority_chain": "bool — closing 'mit Hinweisen / et les références'",
    "score": "int 0-7 — composite template score"
  },
  "is_val_gold": "bool — is this row in any val query's gold set?",
  "val_gold_for_queries": "list[string] — which val queries flag it as gold, e.g. ['val_001']"
}
```

## Legend (v3 — extended)

- 📜 **RULE** — Court rule statement (canonical doctrine)
- 🔁 **echo-cite** — Echoes precedent (mostly BGE citation chain)
- 🤔 **reasoning** — General court analytical reasoning
- 🗣️ **party** — Party position (appellant/respondent argument)
- 🏛️ **lower-court** — Lower-court summary
- ⚖️ **application** — Rule applied to facts (in concreto)
- 🧷 **synthesis** — Closing / recap of a section
- 🔀 **bridge** — Section announcement
- 📋 **procedural** — Procedural history recital
- 🛂 **admissibility** — Admissibility recital
- 🔨 **dispositif** — Operative part of the order
- 💰 **costs** — Cost allocation paragraph
- ✍️ **signature** — End-of-decision boilerplate
- ✂️ **fragment** — CSV parse artifact
- 🔢 **header-only** — Numbering line, no body
- · **brief** — Very short paragraph (<30 chars)
- 🎯 **GOLD** — Marked as gold for the relevant val query

## Role distribution across 379 paragraphs in 27 sampled cases

| Role | Count | % |
|---|---:|---:|
| 📜 RULE | 131 | 34.6% |
| 🗣️ party | 66 | 17.4% |
| 🔁 echo-cite | 42 | 11.1% |
| 🤔 reasoning | 40 | 10.6% |
| 🏛️ lower-court | 28 | 7.4% |
| 💰 costs | 17 | 4.5% |
| ⚖️ application | 15 | 4.0% |
| ✍️ signature | 8 | 2.1% |
| 🧷 synthesis | 7 | 1.8% |
| · brief | 7 | 1.8% |
| 🔨 dispositif | 6 | 1.6% |
| ✂️ fragment | 5 | 1.3% |
| 🔢 header-only | 5 | 1.3% |
| 📋 procedural | 1 | 0.3% |
| 🔀 bridge | 1 | 0.3% |

---

## val_001

**Query (excerpt):** May a court lawfully order a three‑month extension of pre‑trial detention under Art. 221 Abs. 1 lit. b StPO (risk of collusion) consistent with the principle of proportionality when the accused—detained after an alleged late‑night assault and theft o…

### Gold Citation: `BGE 137 IV 122 E. 6.2`

**Case:** `BGE 137 IV 122` (19 paragraphs in corpus)

#### Mental Model Tree

```
BGE 137 IV 122
└── Erwägungen (legal reasoning)
│   ├── E. 2       📜 RULE           
│   ├── E. 3       🏛️ lower-court    
│   │   ├── E. 3.1       🗣️ party          
│   │   ├── E. 3.2       📜 RULE           
│   │   └── E. 3.3       📜 RULE           
│   ├── E. 4
│   │   ├── E. 4.1       🗣️ party          🎯GOLD 
│   │   ├── E. 4.2       📜 RULE           🎯GOLD 
│   │   └── E. 4.3       📜 RULE           
│   ├── E. 5       📜 RULE           
│   │   ├── E. 5.1       🗣️ party          
│   │   ├── E. 5.2       📜 RULE           
│   │   └── E. 5.3       🏛️ lower-court    
│   ├── E. 6
│   │   ├── E. 6.1       🗣️ party          
│   │   ├── E. 6.2       📜 RULE           🎯GOLD   ← SAMPLED
│   │   ├── E. 6.3       🏛️ lower-court    
│   │   ├── E. 6.4       🗣️ party          🎯GOLD 
│   │   └── E. 6.5       🧷 synthesis      
│   ├── E. 10
│   │   └── E. 10.00     · brief          
│   └── E. 13
│       └── E. 13.00     ✂️ fragment       
```

#### Structured JSON (sampled paragraph: `BGE 137 IV 122 E. 6.2`)

```json
{
  "citation": "BGE 137 IV 122 E. 6.2",
  "case_id": "BGE 137 IV 122",
  "eid": "6.2",
  "parent_eid": "6",
  "depth": 2,
  "section": "Erwägungen",
  "role": "rule_statement",
  "language": "de",
  "char_count": 2364,
  "word_count": 342,
  "first_100_chars": "Nach Art. 237 Abs. 1 StPO ordnet das zuständige Gericht an Stelle der Untersuchungs- oder der Sicher",
  "statute_refs": [
    "Art. 237 Abs. 1 StPO",
    "Art. 237 Abs. 2 StPO",
    "Art. 237 Abs. 2 lit. c StPO",
    "Art. 237 Abs. 2 lit. d StPO",
    "Art. 237 Abs. 2 lit. g StPO",
    "Art. 237 StPO"
  ],
  "statute_ref_count": 6,
  "bge_refs": [
    "BGE 137 IV 122"
  ],
  "bge_ref_count": 1,
  "doctrinal_keywords": [
    "kollusionsgefahr",
    "fluchtgefahr",
    "untersuchungshaft",
    "sicherheitshaft",
    "kollusion"
  ],
  "template": {
    "el1_statutory_anchor": true,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": true,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 2
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_001"
  ]
}
```

### Gold Citation: `1B_210/2023 E. 4.1`

**Case:** `1B_210/2023` (16 paragraphs in corpus)

#### Mental Model Tree

```
1B_210/2023
├── Sachverhalt (facts)
│   ├── A       🤔 reasoning      
│   ├── B       📜 RULE           
│   ├── C       🤔 reasoning      
└── Erwägungen (legal reasoning)
│   ├── E. 1       📜 RULE           
│   ├── E. 2       📜 RULE           
│   ├── E. 3       🗣️ party          
│   ├── E. 4       🗣️ party          
│   │   ├── E. 4.1       🔁 echo-cite      🎯GOLD   ← SAMPLED
│   │   ├── E. 4.2       ⚖️ application    
│   │   └── E. 4.3       🗣️ party          
│   └── E. 5       🧷 synthesis      
```

#### Structured JSON (sampled paragraph: `1B_210/2023 E. 4.1`)

```json
{
  "citation": "1B_210/2023 E. 4.1",
  "case_id": "1B_210/2023",
  "eid": "4.1",
  "parent_eid": "4",
  "depth": 2,
  "section": "Erwägungen",
  "role": "echo_doctrine",
  "language": "fr",
  "char_count": 2021,
  "word_count": 323,
  "first_100_chars": "4.1. Conformément à l'art. 221 al. 1 let. b CPP, la détention provisoire ou pour motifs de sûreté ne",
  "statute_refs": [
    "art. 221 al. 1 let. b CPP"
  ],
  "statute_ref_count": 1,
  "bge_refs": [
    "BGE 137 IV 122"
  ],
  "bge_ref_count": 1,
  "doctrinal_keywords": [
    "risque de collusion",
    "détention provisoire"
  ],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": true,
    "el4_positive_limb": true,
    "el5_negative_limb": false,
    "el6_interpretive_factors": true,
    "el7_authority_chain": false,
    "score": 3
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_001"
  ]
}
```

### Gold Citation: `BGE 137 IV 122 E. 6.4`

**Case:** `BGE 137 IV 122` (19 paragraphs in corpus)

#### Mental Model Tree

```
BGE 137 IV 122
└── Erwägungen (legal reasoning)
│   ├── E. 2       📜 RULE           
│   ├── E. 3       🏛️ lower-court    
│   │   ├── E. 3.1       🗣️ party          
│   │   ├── E. 3.2       📜 RULE           
│   │   └── E. 3.3       📜 RULE           
│   ├── E. 4
│   │   ├── E. 4.1       🗣️ party          🎯GOLD 
│   │   ├── E. 4.2       📜 RULE           🎯GOLD 
│   │   └── E. 4.3       📜 RULE           
│   ├── E. 5       📜 RULE           
│   │   ├── E. 5.1       🗣️ party          
│   │   ├── E. 5.2       📜 RULE           
│   │   └── E. 5.3       🏛️ lower-court    
│   ├── E. 6
│   │   ├── E. 6.1       🗣️ party          
│   │   ├── E. 6.2       📜 RULE           🎯GOLD 
│   │   ├── E. 6.3       🏛️ lower-court    
│   │   ├── E. 6.4       🗣️ party          🎯GOLD   ← SAMPLED
│   │   └── E. 6.5       🧷 synthesis      
│   ├── E. 10
│   │   └── E. 10.00     · brief          
│   └── E. 13
│       └── E. 13.00     ✂️ fragment       
```

#### Structured JSON (sampled paragraph: `BGE 137 IV 122 E. 6.4`)

```json
{
  "citation": "BGE 137 IV 122 E. 6.4",
  "case_id": "BGE 137 IV 122",
  "eid": "6.4",
  "parent_eid": "6",
  "depth": 2,
  "section": "Erwägungen",
  "role": "party_position",
  "language": "de",
  "char_count": 1484,
  "word_count": 203,
  "first_100_chars": "Eine Eingrenzung auf ein bestimmtes Gebiet kommt, wie dargelegt (vgl. E. 6.2 hiervor), primär bei Fl",
  "statute_refs": [],
  "statute_ref_count": 0,
  "bge_refs": [
    "BGE 137 IV 122"
  ],
  "bge_ref_count": 1,
  "doctrinal_keywords": [
    "kollusionsgefahr",
    "fluchtgefahr",
    "kollusion"
  ],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 0
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_001"
  ]
}
```

---

## val_002

**Query (excerpt):** A claimant holding a national vocational diploma in warehouse operations worked intermittently as a storage technician from 10 March to 20 September 2022 and was entered as job-seeking on 1 October 2022. From mid-2021 onwards he has suffered from a c…

### Gold Citation: `BGE 148 V 21 E. 5.3`

**Case:** `BGE 148 V 21` (10 paragraphs in corpus)

#### Mental Model Tree

```
BGE 148 V 21
└── Erwägungen (legal reasoning)
│   ├── E. 2       🤔 reasoning      
│   ├── E. 5
│   │   ├── E. 5.1       📜 RULE           
│   │   ├── E. 5.2       🔁 echo-cite      
│   │   └── E. 5.3       📜 RULE           🎯GOLD   ← SAMPLED
│   ├── E. 6
│   │   ├── E. 6.1       📜 RULE           
│   │   ├── E. 6.2       📜 RULE           
│   │   └── E. 6.3       📜 RULE           
│   └── E. 2017    📜 RULE           
```

#### Structured JSON (sampled paragraph: `BGE 148 V 21 E. 5.3`)

```json
{
  "citation": "BGE 148 V 21 E. 5.3",
  "case_id": "BGE 148 V 21",
  "eid": "5.3",
  "parent_eid": "5",
  "depth": 2,
  "section": "Erwägungen",
  "role": "rule_statement",
  "language": "fr",
  "char_count": 1744,
  "word_count": 295,
  "first_100_chars": "En l'absence de dispositions transitoires, l'art. 21 LPC s'applique immédiatement, dès le jour de so",
  "statute_refs": [
    "art. 21 LPC",
    "art. 21 al. 1 LPC"
  ],
  "statute_ref_count": 2,
  "bge_refs": [
    "BGE 142 V 67",
    "BGE 129 V 115",
    "BGE 131 V 242",
    "BGE 121 V 362",
    "BGE 136 V 24",
    "BGE 130 V 445",
    "BGE 129 V 1"
  ],
  "bge_ref_count": 7,
  "doctrinal_keywords": [],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": true,
    "score": 1
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_002"
  ]
}
```

### Gold Citation: `8C_160/2016 E. 4.1`

**Case:** `8C_160/2016` (23 paragraphs in corpus)

#### Mental Model Tree

```
8C_160/2016
├── Sachverhalt (facts)
│   ├── A       🤔 reasoning      
│   ├── B       🏛️ lower-court    
│   ├── C       🤔 reasoning      
└── Erwägungen (legal reasoning)
│   ├── E. 1       📜 RULE           
│   ├── E. 2       🗣️ party          
│   ├── E. 3       💰 costs          
│   │   ├── E. 3.1       📜 RULE           
│   │   ├── E. 3.2       🏛️ lower-court    
│   │   ├── E. 3.3       🗣️ party          
│   │   ├── E. 3.4       🗣️ party          
│   │   └── E. 3.5       ⚖️ application    
│   ├── E. 4       🤔 reasoning      
│   │   ├── E. 4.1       📜 RULE           🎯GOLD   ← SAMPLED
│   │   ├── E. 4.2       🏛️ lower-court    
│   │   └── E. 4.4       🔁 echo-cite      
│   ├── E. 5       💰 costs          
│   │   ├── E. 5.1       🗣️ party          
│   │   └── E. 5.2       ⚖️ application    
│   ├── E. 6       🤔 reasoning      
│   └── E. 7       💰 costs          
```

#### Structured JSON (sampled paragraph: `8C_160/2016 E. 4.1`)

```json
{
  "citation": "8C_160/2016 E. 4.1",
  "case_id": "8C_160/2016",
  "eid": "4.1",
  "parent_eid": "4",
  "depth": 2,
  "section": "Erwägungen",
  "role": "rule_statement",
  "language": "fr",
  "char_count": 880,
  "word_count": 145,
  "first_100_chars": "4.1. Pour évaluer le taux d'invalidité, le revenu que l'assuré aurait pu obtenir s'il n'était pas in",
  "statute_refs": [
    "art. 16 LPGA"
  ],
  "statute_ref_count": 1,
  "bge_refs": [
    "BGE 132 V 93"
  ],
  "bge_ref_count": 1,
  "doctrinal_keywords": [
    "invalidité",
    "réadaptation"
  ],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": true,
    "score": 1
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_002"
  ]
}
```

### Gold Citation: `BGE 144 V 427 E. 3.2`

**Case:** `BGE 144 V 427` (6 paragraphs in corpus)

#### Mental Model Tree

```
BGE 144 V 427
└── Erwägungen (legal reasoning)
│   ├── E. 2       🏛️ lower-court    
│   ├── E. 3
│   │   ├── E. 3.1       🤔 reasoning      
│   │   ├── E. 3.2       📜 RULE           🎯GOLD   ← SAMPLED
│   │   └── E. 3.3       📜 RULE           
│   └── E. 4
│       ├── E. 4.1       🗣️ party          
│       └── E. 4.2       🤔 reasoning      
```

#### Structured JSON (sampled paragraph: `BGE 144 V 427 E. 3.2`)

```json
{
  "citation": "BGE 144 V 427 E. 3.2",
  "case_id": "BGE 144 V 427",
  "eid": "3.2",
  "parent_eid": "3",
  "depth": 2,
  "section": "Erwägungen",
  "role": "rule_statement",
  "language": "de",
  "char_count": 1721,
  "word_count": 230,
  "first_100_chars": "Der Sozialversicherungsprozess ist vom Untersuchungsgrundsatz beherrscht. Danach hat das Gericht von",
  "statute_refs": [],
  "statute_ref_count": 0,
  "bge_refs": [
    "BGE 138 V 218",
    "BGE 144 V 427"
  ],
  "bge_ref_count": 2,
  "doctrinal_keywords": [],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": true,
    "score": 1
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_002"
  ]
}
```

---

## val_003

**Query (excerpt):** A. Rivera, a Peruvian national born in 1994 and with no prior convictions in the forum state, is accused of having, between 5 March and 9 March 2024, together with three accomplices (B. L., C. M. and D. S.) taken part in a series of offenses includin…

### Gold Citation: `BGE 145 I 167 E. 4.1`

**Case:** `BGE 145 I 167` (8 paragraphs in corpus)

#### Mental Model Tree

```
BGE 145 I 167
└── Erwägungen (legal reasoning)
│   ├── E. 3       🤔 reasoning      
│   ├── E. 3a      ✂️ fragment       
│   └── E. 4       🗣️ party          
│       ├── E. 4.1       🔁 echo-cite      🎯GOLD   ← SAMPLED
│       ├── E. 4.2       🔁 echo-cite      
│       ├── E. 4.3       🔁 echo-cite      
│       └── E. 4.4       ⚖️ application    
```

#### Structured JSON (sampled paragraph: `BGE 145 I 167 E. 4.1`)

```json
{
  "citation": "BGE 145 I 167 E. 4.1",
  "case_id": "BGE 145 I 167",
  "eid": "4.1",
  "parent_eid": "4",
  "depth": 2,
  "section": "Erwägungen",
  "role": "echo_doctrine",
  "language": "fr",
  "char_count": 3193,
  "word_count": 523,
  "first_100_chars": "Le droit d'être entendu garanti par l'art. 29 al. 2 Cst. comprend notamment le droit pour l'intéress",
  "statute_refs": [
    "art. 29 al. 2 Cst"
  ],
  "statute_ref_count": 1,
  "bge_refs": [
    "BGE 145 I 167",
    "BGE 142 III 48",
    "BGE 140 I 285",
    "BGE 132 II 257",
    "BGE 131 V 9",
    "BGE 128 V 272",
    "BGE 137 I 305",
    "BGE 134 I 269",
    "BGE 121 I 230",
    "BGE 119 Ia 141"
  ],
  "bge_ref_count": 10,
  "doctrinal_keywords": [
    "droit d'être entendu",
    "arbitraire"
  ],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": true,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": true,
    "score": 2
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_003"
  ]
}
```

### Gold Citation: `1B_192/2022 E. 4.1.2`

**Case:** `1B_192/2022` (20 paragraphs in corpus)

#### Mental Model Tree

```
1B_192/2022
├── Sachverhalt (facts)
│   ├── A       🤔 reasoning      
│   ├── B       🤔 reasoning      
└── Erwägungen (legal reasoning)
│   ├── E. 1       🏛️ lower-court    
│   ├── E. 2       🗣️ party          
│   ├── E. 3       🗣️ party          
│   │   └── E. 3.1       📜 RULE           
│   ├── E. 4       📜 RULE           
│   │   ├── E. 4.1       🔢 header-only    
│   │   ├── E. 4.1.1     📜 RULE           
│   │   ├── E. 4.1.2     📜 RULE           🎯GOLD   ← SAMPLED
│   │   ├── E. 4.2       🔢 header-only    
│   │   ├── E. 4.2.1     📜 RULE           
│   │   ├── E. 4.2.2     ⚖️ application    
│   │   └── E. 4.3       🧷 synthesis      
│   └── E. 5       💰 costs          
```

#### Structured JSON (sampled paragraph: `1B_192/2022 E. 4.1.2`)

```json
{
  "citation": "1B_192/2022 E. 4.1.2",
  "case_id": "1B_192/2022",
  "eid": "4.1.2",
  "parent_eid": "4.1",
  "depth": 3,
  "section": "Erwägungen",
  "role": "rule_statement",
  "language": "fr",
  "char_count": 2597,
  "word_count": 401,
  "first_100_chars": "4.1.2. Certes, le recourant semble intégré professionnellement en Suisse où il vit depuis de nombreu",
  "statute_refs": [
    "art. 106 al. 2 LTF"
  ],
  "statute_ref_count": 1,
  "bge_refs": [],
  "bge_ref_count": 0,
  "doctrinal_keywords": [
    "risque de fuite",
    "droit d'être entendu"
  ],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": true,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 1
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_003"
  ]
}
```

### Gold Citation: `1B_195/2022 E. 2.2.1`

**Case:** `1B_195/2022` (18 paragraphs in corpus)

#### Mental Model Tree

```
1B_195/2022
├── Sachverhalt (facts)
│   ├── A       🤔 reasoning      
│   ├── B       🤔 reasoning      
└── Erwägungen (legal reasoning)
│   ├── E. 1       · brief          
│   │   ├── E. 1.1       📜 RULE           
│   │   └── E. 1.2       🗣️ party          
│   ├── E. 2       💰 costs          
│   │   ├── E. 2.1       🗣️ party          
│   │   ├── E. 2.1.1     📜 RULE           
│   │   ├── E. 2.1.2     🗣️ party          
│   │   ├── E. 2.2       🗣️ party          
│   │   ├── E. 2.2.1     📜 RULE           🎯GOLD   ← SAMPLED
│   │   ├── E. 2.2.2     🗣️ party          
│   │   ├── E. 2.3       📜 RULE           
│   │   └── E. 2.4       🗣️ party          
│   └── E. 3       🗣️ party          
```

#### Structured JSON (sampled paragraph: `1B_195/2022 E. 2.2.1`)

```json
{
  "citation": "1B_195/2022 E. 2.2.1",
  "case_id": "1B_195/2022",
  "eid": "2.2.1",
  "parent_eid": "2.2",
  "depth": 3,
  "section": "Erwägungen",
  "role": "rule_statement",
  "language": "fr",
  "char_count": 858,
  "word_count": 148,
  "first_100_chars": "2.2.1. Conformément à l'art. 221 al. 1 let. a CPP, la détention pour des motifs de sûreté peut être ",
  "statute_refs": [
    "art. 221 al. 1 let. a CPP"
  ],
  "statute_ref_count": 1,
  "bge_refs": [
    "BGE 145 IV 503"
  ],
  "bge_ref_count": 1,
  "doctrinal_keywords": [
    "risque de fuite"
  ],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 0
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_003"
  ]
}
```

---

## val_004

**Query (excerpt):** Mr. Dalton, born in 1941 and resident in a small lakeside town near Thun, executed a handwritten will on 10 October 1997 stating that he left his entire estate to his partner Ms. Lang and, should she predecease him, to his granddaughters Anna (born 1…

### Gold Citation: `BGE 131 III 601 E. 3.1`

**Case:** `BGE 131 III 601` (4 paragraphs in corpus)

#### Mental Model Tree

```
BGE 131 III 601
└── Erwägungen (legal reasoning)
│   └── E. 3       📜 RULE           
│       ├── E. 3.1       📜 RULE           🎯GOLD   ← SAMPLED
│       ├── E. 3.2       🏛️ lower-court    
│       └── E. 3.3       📜 RULE           
```

#### Structured JSON (sampled paragraph: `BGE 131 III 601 E. 3.1`)

```json
{
  "citation": "BGE 131 III 601 E. 3.1",
  "case_id": "BGE 131 III 601",
  "eid": "3.1",
  "parent_eid": "3",
  "depth": 2,
  "section": "Erwägungen",
  "role": "rule_statement",
  "language": "fr",
  "char_count": 2685,
  "word_count": 471,
  "first_100_chars": "Aux termes de l'art. 505 al. 1 CC, le testament olographe est écrit en entier, daté et signé de la m",
  "statute_refs": [
    "art. 505 al. 1 CC",
    "art. 520 al. 1 CC",
    "art. 20 al. 2 CO",
    "art. 7 CC",
    "art. 505 CC"
  ],
  "statute_ref_count": 5,
  "bge_refs": [
    "BGE 131 III 601",
    "BGE 57 II 15",
    "BGE 88 II 67",
    "BGE 117 II 142",
    "BGE 98 II 73",
    "BGE 116 II 117",
    "BGE 131 III 106",
    "BGE 124 III 414",
    "BGE 115 II 323"
  ],
  "bge_ref_count": 9,
  "doctrinal_keywords": [],
  "template": {
    "el1_statutory_anchor": true,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": true,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 2
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_003",
    "val_004"
  ]
}
```

---

## val_005

**Query (excerpt):** A parent, separated from their co-parent since 2008, has not had custody of the two children (primary custody is with the other parent); until March 2020 the parent exercised a longstanding visitation pattern (Wednesday evenings, alternate weekends i…

### Gold Citation: `BGE 131 III 209 E. 5`

**Case:** `BGE 131 III 209` (3 paragraphs in corpus)

#### Mental Model Tree

```
BGE 131 III 209
└── Erwägungen (legal reasoning)
│   ├── E. 3       🔁 echo-cite      
│   ├── E. 4       🏛️ lower-court    
│   └── E. 5       🏛️ lower-court    🎯GOLD   ← SAMPLED
```

#### Structured JSON (sampled paragraph: `BGE 131 III 209 E. 5`)

```json
{
  "citation": "BGE 131 III 209 E. 5",
  "case_id": "BGE 131 III 209",
  "eid": "5",
  "parent_eid": "",
  "depth": 1,
  "section": "Erwägungen",
  "role": "lower_court_summary",
  "language": "de",
  "char_count": 5192,
  "word_count": 705,
  "first_100_chars": "Das Obergericht hat es im Wesentlichen bei einem Verweis auf die kantonale Praxis, wonach das Besuch",
  "statute_refs": [
    "Art. 64 Abs. 2 OG",
    "Art. 64 Abs. 1 OG"
  ],
  "statute_ref_count": 2,
  "bge_refs": [
    "BGE 130 III 585",
    "BGE 127 III 295",
    "BGE 123 III 445",
    "BGE 131 III 209"
  ],
  "bge_ref_count": 4,
  "doctrinal_keywords": [
    "besuchsrecht",
    "kindeswohl",
    "elterliche sorge"
  ],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": true,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 1
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_005"
  ]
}
```

### Gold Citation: `BGE 128 III 411 E. 3.2.1`

**Case:** `BGE 128 III 411` (5 paragraphs in corpus)

#### Mental Model Tree

```
BGE 128 III 411
└── Erwägungen (legal reasoning)
│   ├── E. 3
│   │   ├── E. 3.1       📜 RULE           
│   │   ├── E. 3.2       📜 RULE           
│   │   ├── E. 3.2.1     📜 RULE           🎯GOLD   ← SAMPLED
│   │   └── E. 3.2.2     📜 RULE           
│   └── E. 11
│       └── E. 11.69     ✂️ fragment       
```

#### Structured JSON (sampled paragraph: `BGE 128 III 411 E. 3.2.1`)

```json
{
  "citation": "BGE 128 III 411 E. 3.2.1",
  "case_id": "BGE 128 III 411",
  "eid": "3.2.1",
  "parent_eid": "3.2",
  "depth": 3,
  "section": "Erwägungen",
  "role": "rule_statement",
  "language": "fr",
  "char_count": 4709,
  "word_count": 805,
  "first_100_chars": "Il faut examiner tout d'abord quelle est la portée de cette maxime, et si le débiteur de la contribu",
  "statute_refs": [
    "art. 145 al. 1 CC",
    "art. 280 al. 2 CC",
    "art. 280 CC",
    "art. 274d al. 3 CO",
    "art. 343 al. 4 CO",
    "art. 343 CO",
    "art. 64 al. 1 OJ"
  ],
  "statute_ref_count": 7,
  "bge_refs": [
    "BGE 128 III 411",
    "BGE 118 II 93",
    "BGE 122 I 53",
    "BGE 122 III 404",
    "BGE 111 II 225",
    "BGE 125 III 231",
    "BGE 107 II 233",
    "BGE 123 III 1"
  ],
  "bge_ref_count": 8,
  "doctrinal_keywords": [
    "intérêt de l'enfant"
  ],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 0
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_005"
  ]
}
```

### Gold Citation: `BGE 130 III 585 E. 2.1`

**Case:** `BGE 130 III 585` (7 paragraphs in corpus)

#### Mental Model Tree

```
BGE 130 III 585
└── Erwägungen (legal reasoning)
│   ├── E. 1       🏛️ lower-court    
│   └── E. 2       🤔 reasoning      
│       ├── E. 2.1       📜 RULE           🎯GOLD   ← SAMPLED
│       ├── E. 2.2       🏛️ lower-court    
│       ├── E. 2.2.1     🏛️ lower-court    🎯GOLD 
│       ├── E. 2.2.2     🔁 echo-cite      
│       └── E. 2.3       🧷 synthesis      
```

#### Structured JSON (sampled paragraph: `BGE 130 III 585 E. 2.1`)

```json
{
  "citation": "BGE 130 III 585 E. 2.1",
  "case_id": "BGE 130 III 585",
  "eid": "2.1",
  "parent_eid": "2",
  "depth": 2,
  "section": "Erwägungen",
  "role": "rule_statement",
  "language": "de",
  "char_count": 993,
  "word_count": 149,
  "first_100_chars": "Eltern, denen die elterliche Sorge oder Obhut nicht zusteht, und das unmündige Kind haben gegenseiti",
  "statute_refs": [
    "Art. 273 Abs. 1 ZGB",
    "Art. 273 ZGB"
  ],
  "statute_ref_count": 2,
  "bge_refs": [
    "BGE 123 III 445",
    "BGE 130 III 585",
    "BGE 127 III 295"
  ],
  "bge_ref_count": 3,
  "doctrinal_keywords": [
    "besuchsrecht",
    "kindeswohl",
    "elterliche sorge"
  ],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 0
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_005"
  ]
}
```

---

## val_006

**Query (excerpt):** On 3 March 2012, homeowners Ms. L and her partner Mr. M asked G, an installer they knew socially who works on domestic heating equipment, to help free of charge with the removal of a household fuel reservoir and its connected burner located in a grou…

### Gold Citation: `BGE 128 III 419 E. 2.2`

**Case:** `BGE 128 III 419` (13 paragraphs in corpus)

#### Mental Model Tree

```
BGE 128 III 419
└── Erwägungen (legal reasoning)
│   ├── E. 2
│   │   ├── E. 2.1       📜 RULE           
│   │   ├── E. 2.2       📜 RULE           🎯GOLD   ← SAMPLED
│   │   ├── E. 2.3       🤔 reasoning      
│   │   ├── E. 2.4       📜 RULE           
│   │   ├── E. 2.4.1     📜 RULE           
│   │   ├── E. 2.4.2     📜 RULE           
│   │   └── E. 2.4.3     📜 RULE           
│   ├── E. 1993    🤔 reasoning      
│   └── E. 1999    🤔 reasoning      
```

#### Structured JSON (sampled paragraph: `BGE 128 III 419 E. 2.2`)

```json
{
  "citation": "BGE 128 III 419 E. 2.2",
  "case_id": "BGE 128 III 419",
  "eid": "2.2",
  "parent_eid": "2",
  "depth": 2,
  "section": "Erwägungen",
  "role": "rule_statement",
  "language": "fr",
  "char_count": 3180,
  "word_count": 541,
  "first_100_chars": "Après la conclusion du bail à ferme du 10 septembre 1993, qui prévoit une redevance mensuelle de 10'",
  "statute_refs": [
    "art. 18 al. 1 CO",
    "art. 1 al. 2 CO",
    "art. 18 CO"
  ],
  "statute_ref_count": 3,
  "bge_refs": [
    "BGE 128 III 419",
    "BGE 127 III 444",
    "BGE 118 II 58",
    "BGE 113 II 25",
    "BGE 126 III 25",
    "BGE 125 III 305",
    "BGE 126 III 59",
    "BGE 127 III 279",
    "BGE 127 III 248",
    "BGE 126 III 375",
    "BGE 124 III 363",
    "BGE 123 III 165"
  ],
  "bge_ref_count": 12,
  "doctrinal_keywords": [
    "bonne foi"
  ],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 0
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_006"
  ]
}
```

### Gold Citation: `BGE 130 III 417 E. 3.2`

**Case:** `BGE 130 III 417` (15 paragraphs in corpus)

#### Mental Model Tree

```
BGE 130 III 417
└── Erwägungen (legal reasoning)
│   ├── E. 2       🔁 echo-cite      
│   │   ├── E. 2.1       🗣️ party          
│   │   ├── E. 2.2.1     🔁 echo-cite      
│   │   └── E. 2.2.2     🗣️ party          
│   ├── E. 3       🗣️ party          
│   │   ├── E. 3.1       🔁 echo-cite      
│   │   ├── E. 3.2       ⚖️ application    🎯GOLD   ← SAMPLED
│   │   ├── E. 3.3       📜 RULE           
│   │   └── E. 3.4       🏛️ lower-court    
│   ├── E. 4       🗣️ party          
│   │   ├── E. 4.1       🏛️ lower-court    
│   │   ├── E. 4.2       📜 RULE           
│   │   ├── E. 4.2.1     🔁 echo-cite      
│   │   └── E. 4.2.2     📜 RULE           
│   └── E. 1998    🔁 echo-cite      
```

#### Structured JSON (sampled paragraph: `BGE 130 III 417 E. 3.2`)

```json
{
  "citation": "BGE 130 III 417 E. 3.2",
  "case_id": "BGE 130 III 417",
  "eid": "3.2",
  "parent_eid": "3",
  "depth": 2,
  "section": "Erwägungen",
  "role": "application_in_concreto",
  "language": "fr",
  "char_count": 148,
  "word_count": 25,
  "first_100_chars": "En l'espèce, il n'apparaît pas que la cour cantonale a pu déterminer la volonté commune et réelle de",
  "statute_refs": [],
  "statute_ref_count": 0,
  "bge_refs": [],
  "bge_ref_count": 0,
  "doctrinal_keywords": [],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 0
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_006"
  ]
}
```

### Gold Citation: `BGE 121 IV 207 E. 2a`

**Case:** `BGE 121 IV 207` (6 paragraphs in corpus)

#### Mental Model Tree

```
BGE 121 IV 207
└── Erwägungen (legal reasoning)
│   ├── E. 1a      🗣️ party          
│   ├── E. 1b      🤔 reasoning      
│   ├── E. 2a      🗣️ party          🎯GOLD   ← SAMPLED
│   ├── E. 2b      🔁 echo-cite      
│   ├── E. 2c      🔁 echo-cite      
│   └── E. 3       · brief          
```

#### Structured JSON (sampled paragraph: `BGE 121 IV 207 E. 2a`)

```json
{
  "citation": "BGE 121 IV 207 E. 2a",
  "case_id": "BGE 121 IV 207",
  "eid": "2a",
  "parent_eid": "",
  "depth": 1,
  "section": "Erwägungen",
  "role": "party_position",
  "language": "fr",
  "char_count": 6326,
  "word_count": 1085,
  "first_100_chars": "La recourante soutient qu'en admettant une rupture du rapport de causalité adéquate, la cour cantona",
  "statute_refs": [
    "art. 125 CP",
    "art. 125 al. 1 CP",
    "art. 125 al. 2 CP",
    "art. 18 al. 3 CP",
    "art. 32 CP",
    "art. 33 CP",
    "art. 34 CP"
  ],
  "statute_ref_count": 7,
  "bge_refs": [
    "BGE 116 IV 306",
    "BGE 118 IV 130",
    "BGE 114 IV 173",
    "BGE 115 IV 189",
    "BGE 115 IV 199",
    "BGE 108 IV 3",
    "BGE 100 IV 210",
    "BGE 121 IV 207",
    "BGE 115 IV 162",
    "BGE 111 IV 113",
    "BGE 117 IV 130",
    "BGE 115 IV 100",
    "BGE 103 IV 289",
    "BGE 101 IV 149",
    "BGE 101 IV 67",
    "BGE 100 IV 279"
  ],
  "bge_ref_count": 16,
  "doctrinal_keywords": [
    "proportionnalité"
  ],
  "template": {
    "el1_statutory_anchor": true,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": true,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": true,
    "score": 3
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_006"
  ]
}
```

---

## val_007

**Query (excerpt):** An heirship claims title to a vintage pocket chronometer known as “The Meridian” that belonged to Ms. Barnes, who died in 2010, and contends the timepiece was never validly donated to her visually impaired partner Mr. Collins despite the existence on…

### Gold Citation: `BGE 139 III 305 E. 5`

**Case:** `BGE 139 III 305` (27 paragraphs in corpus)

#### Mental Model Tree

```
BGE 139 III 305
└── Erwägungen (legal reasoning)
│   ├── E. 3       📜 RULE           
│   │   ├── E. 3.1       🗣️ party          
│   │   ├── E. 3.2.1     📜 RULE           
│   │   └── E. 3.2.2     📜 RULE           
│   ├── E. 4
│   │   ├── E. 4.1       🏛️ lower-court    
│   │   └── E. 4.2       📜 RULE           
│   ├── E. 5       🗣️ party          🎯GOLD   ← SAMPLED
│   │   ├── E. 5.1       🔁 echo-cite      
│   │   ├── E. 5.2       🗣️ party          
│   │   ├── E. 5.2.1     🔁 echo-cite      
│   │   ├── E. 5.2.2     🗣️ party          
│   │   ├── E. 5.2.3     🗣️ party          
│   │   ├── E. 5.2.4     📜 RULE           
│   │   ├── E. 5.2.5     🔁 echo-cite      
│   │   ├── E. 5.2.6     🗣️ party          
│   │   ├── E. 5.3       🧷 synthesis      
│   │   ├── E. 5.3.1     🏛️ lower-court    
│   │   ├── E. 5.3.2     🗣️ party          
│   │   ├── E. 5.3.3     🗣️ party          
│   │   ├── E. 5.3.4     📜 RULE           
│   │   ├── E. 5.3.5     🗣️ party          
│   │   ├── E. 5.4.1     🏛️ lower-court    
│   │   ├── E. 5.4.2     📜 RULE           
│   │   ├── E. 5.4.3     ⚖️ application    
│   │   └── E. 5.5       📜 RULE           
│   ├── E. 20      📜 RULE           
│   └── E. 2009    🗣️ party          
```

#### Structured JSON (sampled paragraph: `BGE 139 III 305 E. 5`)

```json
{
  "citation": "BGE 139 III 305 E. 5",
  "case_id": "BGE 139 III 305",
  "eid": "5",
  "parent_eid": "",
  "depth": 1,
  "section": "Erwägungen",
  "role": "party_position",
  "language": "de",
  "char_count": 274,
  "word_count": 40,
  "first_100_chars": "Für den Fall, dass ein Erwerb von einem Nichtberechtigten vorliege, ist das Obergericht zum Schluss ",
  "statute_refs": [],
  "statute_ref_count": 0,
  "bge_refs": [
    "BGE 139 III 305"
  ],
  "bge_ref_count": 1,
  "doctrinal_keywords": [
    "erwerb von einem nichtberechtigten",
    "rechtsmangel",
    "vorsichtsmassnahme"
  ],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 0
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_007"
  ]
}
```

### Gold Citation: `BGE 132 III 155 E. 4`

**Case:** `BGE 132 III 155` (23 paragraphs in corpus)

#### Mental Model Tree

```
BGE 132 III 155
└── Erwägungen (legal reasoning)
│   ├── E. 2       📜 RULE           
│   ├── E. 4       🗣️ party          🎯GOLD   ← SAMPLED
│   │   ├── E. 4.1       📜 RULE           
│   │   └── E. 4.2       ⚖️ application    
│   ├── E. 5       🗣️ party          
│   │   ├── E. 5.1       📜 RULE           
│   │   └── E. 5.2       ⚖️ application    
│   ├── E. 6       🗣️ party          
│   │   ├── E. 6.1       🤔 reasoning      
│   │   ├── E. 6.1.1     🔁 echo-cite      
│   │   ├── E. 6.1.2     ⚖️ application    
│   │   ├── E. 6.1.3     📜 RULE           
│   │   ├── E. 6.2       🤔 reasoning      
│   │   ├── E. 6.2.1     📜 RULE           
│   │   ├── E. 6.2.2     📜 RULE           
│   │   ├── E. 6.2.3     📜 RULE           
│   │   └── E. 6.3       🗣️ party          
│   ├── E. 7       🔁 echo-cite      
│   │   ├── E. 7.1       🗣️ party          
│   │   ├── E. 7.1.1     🤔 reasoning      
│   │   ├── E. 7.1.2     📜 RULE           
│   │   └── E. 7.2       🧷 synthesis      
│   └── E. 1997    📜 RULE           
```

#### Structured JSON (sampled paragraph: `BGE 132 III 155 E. 4`)

```json
{
  "citation": "BGE 132 III 155 E. 4",
  "case_id": "BGE 132 III 155",
  "eid": "4",
  "parent_eid": "",
  "depth": 1,
  "section": "Erwägungen",
  "role": "party_position",
  "language": "de",
  "char_count": 124,
  "word_count": 19,
  "first_100_chars": "Die Kläger behaupten zur Hauptsache, der Besitz - und damit auch das Eigentum - sei durch Besitzanwe",
  "statute_refs": [],
  "statute_ref_count": 0,
  "bge_refs": [],
  "bge_ref_count": 0,
  "doctrinal_keywords": [
    "besitzanweisung"
  ],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 0
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_007"
  ]
}
```

### Gold Citation: `BGE 144 III 264 E. 5.1`

**Case:** `BGE 144 III 264` (26 paragraphs in corpus)

#### Mental Model Tree

```
BGE 144 III 264
└── Erwägungen (legal reasoning)
│   ├── E. 1       · brief          
│   │   └── E. 1.3       📜 RULE           
│   ├── E. 2
│   │   ├── E. 2.1       📜 RULE           
│   │   ├── E. 2.2       📜 RULE           
│   │   └── E. 2.3       🔁 echo-cite      
│   ├── E. 5
│   │   ├── E. 5.1       🏛️ lower-court    🎯GOLD   ← SAMPLED
│   │   ├── E. 5.2       📜 RULE           
│   │   ├── E. 5.3       🔁 echo-cite      
│   │   ├── E. 5.4       🔁 echo-cite      
│   │   └── E. 5.5       📜 RULE           
│   └── E. 6
│       ├── E. 6.1       📜 RULE           
│       ├── E. 6.1.1     🔁 echo-cite      
│       ├── E. 6.1.2     📜 RULE           
│       ├── E. 6.1.3     📜 RULE           
│       ├── E. 6.2.1     📜 RULE           
│       ├── E. 6.2.2     🏛️ lower-court    
│       ├── E. 6.2.3     🔁 echo-cite      
│       ├── E. 6.3.1     📜 RULE           
│       ├── E. 6.3.2     🔁 echo-cite      
│       ├── E. 6.3.3     🔁 echo-cite      
│       ├── E. 6.4.1     🤔 reasoning      
│       ├── E. 6.4.2     🗣️ party          
│       ├── E. 6.4.3     📜 RULE           
│       ├── E. 6.4.4     📜 RULE           
│       ├── E. 6.4.5     🗣️ party          
│       └── E. 6.5       🧷 synthesis      
```

#### Structured JSON (sampled paragraph: `BGE 144 III 264 E. 5.1`)

```json
{
  "citation": "BGE 144 III 264 E. 5.1",
  "case_id": "BGE 144 III 264",
  "eid": "5.1",
  "parent_eid": "5",
  "depth": 2,
  "section": "Erwägungen",
  "role": "lower_court_summary",
  "language": "de",
  "char_count": 358,
  "word_count": 40,
  "first_100_chars": "Das Bundesverwaltungsgericht ist davon ausgegangen, die tatsächlichen Grundlagen der Urteilsunfähigk",
  "statute_refs": [],
  "statute_ref_count": 0,
  "bge_refs": [],
  "bge_ref_count": 0,
  "doctrinal_keywords": [
    "urteilsunfähigkeit"
  ],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 0
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_007"
  ]
}
```

---

## val_008

**Query (excerpt):** Has a member of the town council of the Borough of L., who chaired the board of a publicly funded community trust receiving municipal grants, committed disloyal management of public interests by, during 2015–2020, doing the following: awarding the tr…

### Gold Citation: `BGE 131 III 91 E. 5.2`

**Case:** `BGE 131 III 91` (6 paragraphs in corpus)

#### Mental Model Tree

```
BGE 131 III 91
└── Erwägungen (legal reasoning)
│   └── E. 5       🗣️ party          
│       ├── E. 5.1       📜 RULE           
│       ├── E. 5.2       📜 RULE           🎯GOLD   ← SAMPLED
│       ├── E. 5.2.1     ⚖️ application    
│       ├── E. 5.2.2     📜 RULE           
│       └── E. 5.2.3     🏛️ lower-court    
```

#### Structured JSON (sampled paragraph: `BGE 131 III 91 E. 5.2`)

```json
{
  "citation": "BGE 131 III 91 E. 5.2",
  "case_id": "BGE 131 III 91",
  "eid": "5.2",
  "parent_eid": "5",
  "depth": 2,
  "section": "Erwägungen",
  "role": "rule_statement",
  "language": "fr",
  "char_count": 1187,
  "word_count": 212,
  "first_100_chars": "Selon l'art. 66 al. 1 OJ, l'autorité cantonale à laquelle une affaire est renvoyée peut tenir compte",
  "statute_refs": [
    "art. 66 al. 1 OJ"
  ],
  "statute_ref_count": 1,
  "bge_refs": [
    "BGE 104 IV 276",
    "BGE 103 IV 73",
    "BGE 116 II 220"
  ],
  "bge_ref_count": 3,
  "doctrinal_keywords": [],
  "template": {
    "el1_statutory_anchor": true,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 1
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_008"
  ]
}
```

### Gold Citation: `6B_1233/2016 E. 1`

**Case:** `6B_1233/2016` (18 paragraphs in corpus)

#### Mental Model Tree

```
6B_1233/2016
├── Sachverhalt (facts)
│   ├── A       🤔 reasoning      
│   ├── C       💰 costs          
└── Erwägungen (legal reasoning)
│   ├── E. 1       🔁 echo-cite      🎯GOLD   ← SAMPLED
│   │   ├── E. 1.1       🗣️ party          
│   │   ├── E. 1.3       📜 RULE           
│   │   ├── E. 1.4       🗣️ party          
│   │   ├── E. 1.5       🗣️ party          
│   │   └── E. 1.6       📜 RULE           
│   ├── E. 2       🗣️ party          
│   ├── E. 3       🗣️ party          
│   │   ├── E. 3.1       🔁 echo-cite      
│   │   ├── E. 3.2       📜 RULE           
│   │   ├── E. 3.3       📜 RULE           
│   │   └── E. 3.4       🗣️ party          
│   └── E. 4       🗣️ party          
```

#### Structured JSON (sampled paragraph: `6B_1233/2016 E. 1`)

```json
{
  "citation": "6B_1233/2016 E. 1",
  "case_id": "6B_1233/2016",
  "eid": "1",
  "parent_eid": "",
  "depth": 1,
  "section": "Erwägungen",
  "role": "echo_doctrine",
  "language": "fr",
  "char_count": 978,
  "word_count": 174,
  "first_100_chars": "L'autorité de l'arrêt de renvoi, que prévoyaient expressément l'art. 66 al. 1 aOJ et l'art. 277ter a",
  "statute_refs": [
    "art. 66 al. 1 aOJ"
  ],
  "statute_ref_count": 1,
  "bge_refs": [
    "BGE 135 III 334",
    "BGE 131 III 91"
  ],
  "bge_ref_count": 2,
  "doctrinal_keywords": [],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 0
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_008"
  ]
}
```

### Gold Citation: `BGE 135 III 334 E. 2`

**Case:** `BGE 135 III 334` (4 paragraphs in corpus)

#### Mental Model Tree

```
BGE 135 III 334
└── Erwägungen (legal reasoning)
│   ├── E. 2       📜 RULE           🎯GOLD   ← SAMPLED
│   │   ├── E. 2.1       🤔 reasoning      
│   │   └── E. 2.2       🔁 echo-cite      
│   └── E. 4
│       └── E. 4.1.4.5   ✂️ fragment       
```

#### Structured JSON (sampled paragraph: `BGE 135 III 334 E. 2`)

```json
{
  "citation": "BGE 135 III 334 E. 2",
  "case_id": "BGE 135 III 334",
  "eid": "2",
  "parent_eid": "",
  "depth": 1,
  "section": "Erwägungen",
  "role": "rule_statement",
  "language": "de",
  "char_count": 1961,
  "word_count": 300,
  "first_100_chars": "Vor Einführung des Bundesgerichtsgesetzes (BGG) durfte die kantonale Instanz, an die eine Sache zurü",
  "statute_refs": [
    "Art. 66 Abs. 1 OG"
  ],
  "statute_ref_count": 1,
  "bge_refs": [
    "BGE 131 III 91",
    "BGE 116 II 220",
    "BGE 133 III 201",
    "BGE 125 III 421",
    "BGE 111 II 94"
  ],
  "bge_ref_count": 5,
  "doctrinal_keywords": [],
  "template": {
    "el1_statutory_anchor": true,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": true,
    "score": 2
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_008"
  ]
}
```

---

## val_009

**Query (excerpt):** A divorced custodial parent lives alone with four children born in 1996, 1998, 2001 and 2006. The non-custodial parent is serving a prison sentence overseas of about seven years (release date uncertain), has failed to pay previously-ordered maintenan…

### Gold Citation: `BGE 137 III 193 E. 2.1`

**Case:** `BGE 137 III 193` (14 paragraphs in corpus)

#### Mental Model Tree

```
BGE 137 III 193
└── Erwägungen (legal reasoning)
│   ├── E. 1
│   │   └── E. 1.1       📜 RULE           
│   ├── E. 2
│   │   ├── E. 2.1       📜 RULE           🎯GOLD   ← SAMPLED
│   │   └── E. 2.2       🏛️ lower-court    
│   ├── E. 3       📜 RULE           
│   │   ├── E. 3.1       📜 RULE           
│   │   ├── E. 3.2       📜 RULE           
│   │   ├── E. 3.3       📜 RULE           
│   │   ├── E. 3.4       📜 RULE           
│   │   ├── E. 3.5       📜 RULE           
│   │   ├── E. 3.6       📜 RULE           
│   │   ├── E. 3.7       📜 RULE           
│   │   ├── E. 3.8       🏛️ lower-court    
│   │   └── E. 3.9       📜 RULE           
│   └── E. 322
│       └── E. 322.6     🔁 echo-cite      
```

#### Structured JSON (sampled paragraph: `BGE 137 III 193 E. 2.1`)

```json
{
  "citation": "BGE 137 III 193 E. 2.1",
  "case_id": "BGE 137 III 193",
  "eid": "2.1",
  "parent_eid": "2",
  "depth": 2,
  "section": "Erwägungen",
  "role": "rule_statement",
  "language": "de",
  "char_count": 587,
  "word_count": 94,
  "first_100_chars": "Wenn die Eltern die Sorge für das Kind vernachlässigen, kann das Gericht gemäss Art. 291 ZGB ihre Sc",
  "statute_refs": [
    "Art. 291 ZGB",
    "Art. 289 Abs. 2 ZGB",
    "Art. 293 Abs. 2 ZGB"
  ],
  "statute_ref_count": 3,
  "bge_refs": [],
  "bge_ref_count": 0,
  "doctrinal_keywords": [],
  "template": {
    "el1_statutory_anchor": true,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": true,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 2
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_009"
  ]
}
```

### Gold Citation: `5A_561/2020 E. 5.1.1`

**Case:** `5A_561/2020` (32 paragraphs in corpus)

#### Mental Model Tree

```
5A_561/2020
├── Sachverhalt (facts)
│   ├── A       🤔 reasoning      
│   ├── B       🏛️ lower-court    
│   ├── C       🗣️ party          
└── Erwägungen (legal reasoning)
│   ├── E. 1       🔨 dispositif     
│   │   ├── E. 1.1       📜 RULE           
│   │   └── E. 1.2       📜 RULE           
│   ├── E. 2       🤔 reasoning      
│   │   ├── E. 2.1       📜 RULE           
│   │   └── E. 2.2       📜 RULE           
│   ├── E. 3       🗣️ party          
│   │   ├── E. 3.1       🔁 echo-cite      
│   │   ├── E. 3.2       🗣️ party          
│   │   └── E. 3.3       🗣️ party          
│   ├── E. 4       📜 RULE           
│   ├── E. 5       🗣️ party          
│   │   ├── E. 5.1       🔢 header-only    
│   │   ├── E. 5.1.1     🔁 echo-cite      🎯GOLD   ← SAMPLED
│   │   ├── E. 5.1.2     🤔 reasoning      
│   │   ├── E. 5.1.3     🔁 echo-cite      
│   │   ├── E. 5.2       🤔 reasoning      
│   │   ├── E. 5.2.1     🔁 echo-cite      
│   │   ├── E. 5.3       🔢 header-only    
│   │   ├── E. 5.3.1     🗣️ party          
│   │   ├── E. 5.3.2     🗣️ party          
│   │   ├── E. 5.4       🗣️ party          
│   │   ├── E. 5.5       🔢 header-only    
│   │   ├── E. 5.5.1     🗣️ party          
│   │   └── E. 5.5.2     📜 RULE           
│   └── E. 6       📜 RULE           
```

#### Structured JSON (sampled paragraph: `5A_561/2020 E. 5.1.1`)

```json
{
  "citation": "5A_561/2020 E. 5.1.1",
  "case_id": "5A_561/2020",
  "eid": "5.1.1",
  "parent_eid": "5.1",
  "depth": 3,
  "section": "Erwägungen",
  "role": "echo_doctrine",
  "language": "de",
  "char_count": 961,
  "word_count": 154,
  "first_100_chars": "5.1.1. Im Verhältnis zum unmündigen Kind sind besonders hohe Anforderungen an die Ausnützung der eig",
  "statute_refs": [],
  "statute_ref_count": 0,
  "bge_refs": [
    "BGE 144 III 481",
    "BGE 144 III 10"
  ],
  "bge_ref_count": 2,
  "doctrinal_keywords": [],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 0
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_009"
  ]
}
```

### Gold Citation: `5A_954/2015 E. 3.3`

**Case:** `5A_954/2015` (17 paragraphs in corpus)

#### Mental Model Tree

```
5A_954/2015
├── Sachverhalt (facts)
│   ├── A       🤔 reasoning      
│   ├── B       📋 procedural     
│   ├── C       🗣️ party          
└── Erwägungen (legal reasoning)
│   ├── E. 1       🔨 dispositif     
│   │   ├── E. 1.1       📜 RULE           
│   │   ├── E. 1.2       📜 RULE           
│   │   └── E. 1.3       📜 RULE           
│   ├── E. 2       🏛️ lower-court    
│   ├── E. 3       🤔 reasoning      
│   │   ├── E. 3.1       🔁 echo-cite      
│   │   ├── E. 3.2       📜 RULE           
│   │   ├── E. 3.3       📜 RULE           🎯GOLD   ← SAMPLED
│   │   └── E. 3.4       ⚖️ application    
│   ├── E. 4       📜 RULE           
│   └── E. 5       💰 costs          
```

#### Structured JSON (sampled paragraph: `5A_954/2015 E. 3.3`)

```json
{
  "citation": "5A_954/2015 E. 3.3",
  "case_id": "5A_954/2015",
  "eid": "3.3",
  "parent_eid": "3",
  "depth": 2,
  "section": "Erwägungen",
  "role": "rule_statement",
  "language": "de",
  "char_count": 1190,
  "word_count": 176,
  "first_100_chars": "3.3. Der Arrest in den Fällen von Art. 271 Abs. 1 Ziff. 1 oder Ziff. 2 SchKG bewirkt jedoch nicht di",
  "statute_refs": [
    "Art. 271 Abs. 2 SchKG",
    "Art. 289 ZGB"
  ],
  "statute_ref_count": 2,
  "bge_refs": [
    "BGE 40 III 451"
  ],
  "bge_ref_count": 1,
  "doctrinal_keywords": [],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 0
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_009"
  ]
}
```

---

## val_010

**Query (excerpt):** A Belize-registered investment vehicle (M) authorized its sole beneficial owner (P) to open and operate a euro account with a private bank in Zurich (Q). P delegated day-to-day portfolio handling to an external adviser (R) who held a power of attorne…

### Gold Citation: `BGE 132 III 449 E. 2`

**Case:** `BGE 132 III 449` (5 paragraphs in corpus)

#### Mental Model Tree

```
BGE 132 III 449
└── Erwägungen (legal reasoning)
│   ├── E. 2       📜 RULE           🎯GOLD   ← SAMPLED
│   ├── E. 3       🤔 reasoning      
│   ├── E. 4       🔁 echo-cite      
│   └── E. 2001    🔁 echo-cite      
```

#### Structured JSON (sampled paragraph: `BGE 132 III 449 E. 2`)

```json
{
  "citation": "BGE 132 III 449 E. 2",
  "case_id": "BGE 132 III 449",
  "eid": "2",
  "parent_eid": "",
  "depth": 1,
  "section": "Erwägungen",
  "role": "rule_statement",
  "language": "fr",
  "char_count": 3965,
  "word_count": 684,
  "first_100_chars": "Il n'est pas nécessaire d'examiner de façon détaillée la nature juridique de la relation contractuel",
  "statute_refs": [
    "art. 100 CO",
    "art. 100 al. 1 CO",
    "art. 4 CC",
    "art. 100 al. 2 CO",
    "art. 101 al. 3 CO"
  ],
  "statute_ref_count": 5,
  "bge_refs": [
    "BGE 111 II 263",
    "BGE 132 III 449",
    "BGE 108 II 314",
    "BGE 112 II 450",
    "BGE 127 III 553",
    "BGE 122 III 26",
    "BGE 121 III 69",
    "BGE 116 II 459"
  ],
  "bge_ref_count": 8,
  "doctrinal_keywords": [],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 0
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_010"
  ]
}
```

### Gold Citation: `4A_379/2016 E. 3.3.1`

**Case:** `4A_379/2016` (30 paragraphs in corpus)

#### Mental Model Tree

```
4A_379/2016
├── Sachverhalt (facts)
│   ├── A       📜 RULE           
│   ├── B       📜 RULE           
│   ├── C       🏛️ lower-court    
└── Erwägungen (legal reasoning)
│   ├── E. 1       🔨 dispositif     
│   │   ├── E. 1.1       📜 RULE           
│   │   ├── E. 1.2       🤔 reasoning      
│   │   ├── E. 1.3       📜 RULE           
│   │   ├── E. 1.4       🤔 reasoning      
│   │   └── E. 1.5       🤔 reasoning      
│   ├── E. 2       💰 costs          
│   │   ├── E. 2.1       📜 RULE           
│   │   └── E. 2.2       📜 RULE           
│   ├── E. 3       ⚖️ application    
│   │   ├── E. 3.1       🤔 reasoning      
│   │   ├── E. 3.2       🤔 reasoning      
│   │   ├── E. 3.2.1     📜 RULE           
│   │   ├── E. 3.2.2     📜 RULE           
│   │   ├── E. 3.3       🤔 reasoning      
│   │   ├── E. 3.3.1     📜 RULE           🎯GOLD   ← SAMPLED
│   │   └── E. 3.3.2     🔁 echo-cite      
│   ├── E. 4       ⚖️ application    
│   ├── E. 5
│   │   ├── E. 5.1       ⚖️ application    
│   │   ├── E. 5.2       📜 RULE           
│   │   ├── E. 5.3       🤔 reasoning      
│   │   ├── E. 5.3.1     🏛️ lower-court    
│   │   ├── E. 5.3.2     🤔 reasoning      
│   │   └── E. 5.4       🔀 bridge         
│   └── E. 6       📜 RULE           
```

#### Structured JSON (sampled paragraph: `4A_379/2016 E. 3.3.1`)

```json
{
  "citation": "4A_379/2016 E. 3.3.1",
  "case_id": "4A_379/2016",
  "eid": "3.3.1",
  "parent_eid": "3.3",
  "depth": 3,
  "section": "Erwägungen",
  "role": "rule_statement",
  "language": "fr",
  "char_count": 1969,
  "word_count": 335,
  "first_100_chars": "3.3.1. Il est habituel que les conditions générales de la banque auxquelles le client adhère, compor",
  "statute_refs": [
    "art. 100 al. 1 CO"
  ],
  "statute_ref_count": 1,
  "bge_refs": [
    "BGE 132 III 449",
    "BGE 112 II 450"
  ],
  "bge_ref_count": 2,
  "doctrinal_keywords": [],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": false,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 0
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_010"
  ]
}
```

### Gold Citation: `BGE 128 III 76 E. 1b`

**Case:** `BGE 128 III 76` (4 paragraphs in corpus)

#### Mental Model Tree

```
BGE 128 III 76
└── Erwägungen (legal reasoning)
│   ├── E. 1a      📜 RULE           
│   ├── E. 1b      📜 RULE           🎯GOLD   ← SAMPLED
│   ├── E. 1c      📜 RULE           
│   └── E. 1d      📜 RULE           
```

#### Structured JSON (sampled paragraph: `BGE 128 III 76 E. 1b`)

```json
{
  "citation": "BGE 128 III 76 E. 1b",
  "case_id": "BGE 128 III 76",
  "eid": "1b",
  "parent_eid": "",
  "depth": 1,
  "section": "Erwägungen",
  "role": "rule_statement",
  "language": "fr",
  "char_count": 3413,
  "word_count": 570,
  "first_100_chars": "Il n'est fait exception à la règle qui précède que si le droit fédéral contient une norme dont le dr",
  "statute_refs": [
    "art. 44 LAA",
    "art. 44 al. 2 LAA",
    "art. 50 OJ",
    "art. 55 al. 2 CC",
    "art. 328 al. 2 CO"
  ],
  "statute_ref_count": 5,
  "bge_refs": [
    "BGE 125 III 461",
    "BGE 119 II 297",
    "BGE 115 II 237",
    "BGE 103 II 75",
    "BGE 128 III 76",
    "BGE 119 II 443",
    "BGE 115 II 283"
  ],
  "bge_ref_count": 7,
  "doctrinal_keywords": [],
  "template": {
    "el1_statutory_anchor": false,
    "el2_paraphrase": null,
    "el3_teleology": false,
    "el4_positive_limb": true,
    "el5_negative_limb": false,
    "el6_interpretive_factors": false,
    "el7_authority_chain": false,
    "score": 1
  },
  "is_val_gold": true,
  "val_gold_for_queries": [
    "val_010"
  ]
}
```

---
