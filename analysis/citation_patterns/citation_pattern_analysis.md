# Swiss citation pattern analysis

## Counts

| Dataset | Rows | Unique source citations | Unique text references | All unique citations | Unique links |
|---|---:|---:|---:|---:|---:|
| court_considerations | 2,476,315 | 1,985,178 | 534,599 | 2,434,429 | 4,554,397 |
| laws_de | 175,933 | 175,933 | 32,277 | 197,801 | 45,165 |

## Naming families

### statute_article

Art./Artikel + article + optional Abs./lit./Ziff./Satz + law code

Segments: article_marker, article_number, article_suffix_or_range, paragraph_marker, paragraph_number, subdivision_marker, subdivision_value, law_code

Examples: Art. 221 Abs. 1 lit. b StPO; Art. 69 Abs. 1bis IVG; Artikel 10 TSchG; Art. 30-39 EpG

### statute_section

section sign + section number + optional Abs./lit. + code

Segments: section_marker, section_number, paragraph_marker, paragraph_number, subdivision_marker, subdivision_value, law_code

Examples: § 24a Abs. 1 SHG; §§ 25 ff. PBG/SZ; § 1.22

### court_bge

BGE + volume + roman division + first page + optional E./S. pinpoint

Segments: reporter, volume, division, page, pinpoint_unit, pinpoint

Examples: BGE 137 IV 122 E. 6.2; BGE 139 I 2 S. 7

### court_case

federal docket + optional date + optional E. consideration

Segments: court_chamber, legal_area_code, separator_style, serial_number, decision_year, decision_date, consideration

Examples: 1B_210/2023 E. 4.1; 2P.198/2006 09.05.2007 E. 2; I 402/02 13.11.2002 E. 4

### official_reference

SR/BBl/AS + numeric publication reference

Segments: publication, number_or_year, page

Examples: SR 455; BBl 2009 3547; AS 2011 1199

## Classification notes

- Law citations are split into article/section marker, numeric unit, optional suffix or range, paragraph, letter/number/sentence subdivisions, and law code.
- Law codes are subtyped morphologically as systematic collection numbers, uppercase abbreviations, mixed-case abbreviations, numbered abbreviations, compound abbreviations, or slash/cantonal abbreviations.
- BGE court citations are split into reporter, volume, roman division, page, and optional pinpoint unit (`E.` consideration or `S.` page).
- Federal tribunal docket citations are split into chamber, legal area code, separator style, serial number, year, optional decision date, and optional consideration.
- References without an explicit law code in `laws_de.csv` are resolved to the source row law code. In court text, unresolved article references are kept without a law code rather than guessed.
