#!/usr/bin/env python
"""Build deterministic authority cards for court consideration citations.

Derived only from corpus artifacts (court text, segment lattice, citation graph).
Does not read train/val/test labels.

Parallelism: use --chunk-index 0..N-1 with --num-chunks N to split work across
N parallel workers. Each worker reads the full CSV but processes only its rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
COURT_CSV = DATA_DIR / "court_considerations.csv"
SEGMENT_DB = ART_DIR / "segment_lattice_v3.sqlite"
GRAPH_DB = ROOT / "data_insights" / "citation_graph_extracted.sqlite"
DEFAULT_OUT = ART_DIR / "court_authority_cards.jsonl"


ARTICLE_CODE_RE = re.compile(r"\bArt\.\s+[0-9][A-Za-z0-9.]*.*?\s+([A-Z][A-Za-z0-9.\-/]{1,20})\b")
BGE_BASE_RE = re.compile(r"\bBGE\s+\d{3}\s+[IVX]{1,4}\s+\d+[a-z]?\b")
DOCKET_BASE_RE = re.compile(r"\b\d{1,2}[A-Z]{1,4}[_\.]\d{1,5}/\d{4}\b")
NON_LAW_CODE_TOKENS = {
    "Abs", "Absatz", "Bst", "Bst.", "lit", "lit.", "Ziff", "Ziff.",
    "Satz", "und", "oder", "i", "i.V.m", "al", "let", "ch",
}

# Notification paragraph signals (closing boilerplate — no legal content)
NOTIFICATION_PATTERNS = re.compile(
    r"(?:"
    r"dieses urteil wird den parteien|"
    r"le pr[eé]sent arr[eê]t est communiqu[eé]|"
    r"la presente sentenza [eèé] intimata|"
    r"schriftlich mitgeteilt\b|"
    r"\bes werden keine gerichtskosten erhoben\b|"
    r"\bil n'est pas per[cç]u de frais\b|"
    r"\bnon si percepiscono spese\b|"
    r"\bgerichtsschreiber\s*:\s*\w|"
    r"\bgreffier\s*:\s*\w|"
    r"\bim namen der .{3,60} abteilung\b"
    r")",
    re.IGNORECASE,
)


BGE_DIVISION_AREAS = {
    "I": "constitutional and public law",
    "II": "administrative, tax, migration, and regulatory law",
    "III": "civil law",
    "IV": "criminal law and criminal procedure",
    "V": "social insurance law",
}


# Complete Swiss Federal Tribunal docket prefix table
# Modern (post-2007) + legacy (pre-2007 BGer and EVG)
DOCKET_PREFIX_AREAS: dict[str, str] = {
    # ── Public law courts ────────────────────────────────────────────────────
    "1B": "criminal procedure and coercive measures",
    "1C": "constitutional and public law",
    "1D": "constitutional and public law",        # denaturalisation / citizenship
    "1P": "constitutional and public law",        # old Staatsrechtliche Beschwerde
    "1A": "constitutional and public law",        # old public law appeals
    "1E": "constitutional and public law",
    "1F": "constitutional and public law",
    "1G": "constitutional and public law",        # interpretation request
    "1S": "constitutional and public law",
    "2C": "administrative, tax, migration, and regulatory law",
    "2A": "administrative and tax law",           # old administrative/tax
    "2D": "administrative law",
    "2E": "administrative and European law",
    "2F": "administrative law",
    "2G": "administrative, tax, migration, and regulatory law",  # interpretation
    "2P": "administrative and public law",        # old public law II
    # ── Civil law courts ─────────────────────────────────────────────────────
    "4A": "civil obligations, contract, commercial, and banking law",
    "4B": "civil law",
    "4C": "civil law",                            # old civil law
    "4D": "civil law subsidiary constitutional matters",
    "4F": "civil law",
    "4G": "civil law",                            # interpretation request
    "4P": "civil law",
    "5A": "family law, inheritance, debt enforcement, and civil law",
    "5B": "family and civil law",
    "5C": "family and civil law",                 # old family/civil
    "5D": "civil law subsidiary constitutional matters",
    "5F": "civil law",
    "5E": "family and civil law",                 # direct jurisdiction / extraordinary
    "5G": "family law, inheritance, debt enforcement, and civil law",  # interpretation
    "5N": "family and civil law",
    "5P": "family and civil law",                 # old civil law
    # ── Criminal law courts ───────────────────────────────────────────────────
    "6A": "criminal law and administrative criminal law",  # old admin+criminal
    "6B": "criminal law and criminal procedure",
    "6C": "criminal law and criminal procedure",
    "6F": "criminal law",
    "6G": "criminal law and criminal procedure",  # interpretation
    "6P": "constitutional and public law",        # old Staatsrechtliche Beschwerde in Strafsachen
    "6S": "criminal law",
    "7B": "criminal law and criminal procedure",
    "7F": "criminal law",
    "7G": "criminal law and criminal procedure",  # interpretation
    # ── Social law courts ─────────────────────────────────────────────────────
    "8C": "social insurance and public employment law",
    "8D": "social insurance law",
    "8F": "social insurance law",
    "8G": "social insurance and public employment law",   # interpretation
    "9C": "social insurance law",
    "9D": "social insurance law",                 # subsidiary constitutional
    "9E": "social insurance law",
    "9F": "social insurance law",
    "9G": "social insurance law",                 # interpretation
    "9X": "social insurance law",                 # extraordinary revision
    # ── Special / extraordinary ───────────────────────────────────────────────
    "11Z": "civil law",                           # special extraordinary proceedings
    "12T": "disciplinary proceedings",
    "13Y": "criminal law",                        # old extraordinary Kassationshof
    "10Y": "criminal law",                        # old extraordinary
    # ── Old Federal Insurance Court (EVG / ATFA, pre-2007 merger) ────────────
    # Single-letter docket prefixes used before EVG merged into BGer in 2007
    "I":  "invalidity insurance law",             # Invalidenversicherung (IV)
    "U":  "accident insurance law",               # Unfallversicherung (UV)
    "K":  "health insurance law",                 # Krankenversicherung (KV)
    "H":  "accident and liability insurance law", # Haftpflicht / Heilbehandlung
    "C":  "unemployment insurance law",           # chômage / Arbeitslosenversicherung
    "B":  "occupational pension and social insurance law",  # Berufliche Vorsorge
    "P":  "supplementary benefits law",           # Prestations complémentaires (PC/EL)
    "M":  "military insurance law",               # Militärversicherung
    "E":  "social insurance law",                 # various EVG
    "F":  "social insurance law",                 # various EVG
}


# German codes + French (FR) and Italian (IT) equivalents for all major Swiss federal laws
LAW_CODE_AREAS: dict[str, tuple[str, list[str]]] = {
    # Criminal procedure
    "StPO": ("criminal procedure", ["criminal procedure", "pretrial detention", "procedural rights"]),
    "CPP":  ("criminal procedure", ["criminal procedure", "pretrial detention", "procedural rights"]),
    "PPM":  ("criminal procedure", ["criminal procedure", "pretrial detention", "procedural rights"]),
    # Criminal law
    "StGB": ("criminal law", ["criminal offence", "criminal liability", "sentencing"]),
    "CP":   ("criminal law", ["criminal offence", "criminal liability", "sentencing"]),
    # Federal court procedure
    "BGG":  ("federal supreme court procedure", ["appeal admissibility", "federal procedure"]),
    "LTF":  ("federal supreme court procedure", ["appeal admissibility", "federal procedure"]),
    "OG":   ("federal supreme court procedure (old)", ["appeal admissibility", "federal procedure"]),
    "OJ":   ("federal supreme court procedure (old)", ["appeal admissibility", "federal procedure"]),
    # Constitutional law
    "BV":   ("constitutional law", ["constitutional rights", "due process", "fundamental rights"]),
    "Cst":  ("constitutional law", ["constitutional rights", "due process", "fundamental rights"]),
    "Cost": ("constitutional law", ["constitutional rights", "due process", "fundamental rights"]),
    # Civil law
    "ZGB":  ("civil law", ["family law", "property law", "inheritance law"]),
    "CC":   ("civil law", ["family law", "property law", "inheritance law"]),
    "CCS":  ("civil law", ["family law", "property law", "inheritance law"]),
    # Code of obligations
    "OR":   ("private law and obligations", ["contract law", "liability", "obligations"]),
    "CO":   ("private law and obligations", ["contract law", "liability", "obligations"]),
    # Civil procedure
    "ZPO":  ("civil procedure", ["civil procedure", "evidence", "appeal"]),
    "CPC":  ("civil procedure", ["civil procedure", "evidence", "appeal"]),
    "BZP":  ("civil procedure (supplement)", ["civil procedure", "procedural rights"]),
    "PCF":  ("civil procedure (supplement)", ["civil procedure", "procedural rights"]),
    # Debt enforcement and bankruptcy
    "SchKG": ("debt enforcement and bankruptcy", ["debt enforcement", "bankruptcy", "security"]),
    "LP":   ("debt enforcement and bankruptcy", ["debt enforcement", "bankruptcy", "security"]),
    "LEF":  ("debt enforcement and bankruptcy", ["debt enforcement", "bankruptcy", "security"]),
    # Private international law
    "IPRG": ("private international law", ["conflict of laws", "international jurisdiction"]),
    "LDIP": ("private international law", ["conflict of laws", "international jurisdiction"]),
    # Old age and survivors insurance
    "AHVG": ("old age and survivors insurance", ["old age insurance", "AHV pension", "survivors benefit"]),
    "LAVS": ("old age and survivors insurance", ["old age insurance", "AHV pension", "survivors benefit"]),
    "AVS":  ("old age and survivors insurance", ["old age insurance", "AHV pension", "survivors benefit"]),
    # Invalidity insurance
    "IVG":  ("invalidity insurance", ["invalidity insurance", "vocational rehabilitation"]),
    "LAI":  ("invalidity insurance", ["invalidity insurance", "vocational rehabilitation"]),
    # Social insurance general
    "ATSG": ("social insurance general law", ["social insurance", "earning incapacity"]),
    "LPGA": ("social insurance general law", ["social insurance", "earning incapacity"]),
    # Accident insurance
    "UVG":  ("accident insurance", ["accident insurance", "occupational disease"]),
    "LAA":  ("accident insurance", ["accident insurance", "occupational disease"]),
    # Road traffic
    "SVG":  ("road traffic law", ["traffic accident", "vehicle holder liability"]),
    "LCR":  ("road traffic law", ["traffic accident", "vehicle holder liability"]),
    # Migration / foreigners
    "AIG":  ("migration law", ["residence permit", "removal", "family reunification"]),
    "AuG":  ("migration law", ["residence permit", "removal", "family reunification"]),
    "LEI":  ("migration law", ["residence permit", "removal", "family reunification"]),
    "LEtr": ("migration law", ["residence permit", "removal", "family reunification"]),
    "ANAG": ("migration law", ["residence permit", "removal", "family reunification"]),
    # Tax law
    "DBG":  ("direct federal tax", ["tax law", "tax assessment"]),
    "LIFD": ("direct federal tax", ["tax law", "tax assessment"]),
    "StHG": ("cantonal tax harmonisation", ["tax law", "cantonal tax"]),
    "LHID": ("cantonal tax harmonisation", ["tax law", "cantonal tax"]),
    "MWSTG": ("VAT law", ["value added tax"]),
    "LTVA": ("VAT law", ["value added tax"]),
    "VStG": ("withholding tax", ["withholding tax", "tax refund"]),
    "LIA":  ("withholding tax", ["withholding tax", "tax refund"]),        # FR
    # Copyright / IP
    "URG":  ("copyright law", ["copyright", "intellectual property"]),
    "LDA":  ("copyright law", ["copyright", "intellectual property"]),
    # Unfair competition
    "UWG":  ("unfair competition", ["unfair competition", "trade secrets"]),
    "LCD":  ("unfair competition", ["unfair competition", "trade secrets"]),
    # Competition / antitrust
    "KG":   ("competition and antitrust law", ["competition law", "antitrust", "cartel", "market dominance"]),
    "LCart": ("competition and antitrust law", ["competition law", "antitrust", "cartel", "market dominance"]),
    # Spatial planning
    "RPG":  ("spatial planning law", ["spatial planning", "zoning", "building permit"]),
    "LAT":  ("spatial planning law", ["spatial planning", "zoning", "building permit"]),
    "RPV":  ("spatial planning ordinance", ["spatial planning", "zoning"]),
    "OAT":  ("spatial planning ordinance", ["spatial planning", "zoning"]),
    # Environmental protection
    "USG":  ("environmental protection law", ["environmental protection", "pollution control"]),
    "LPE":  ("environmental protection law", ["environmental protection", "pollution control"]),
    # Administrative procedure
    "VwVG": ("administrative procedure", ["administrative procedure", "administrative decision"]),
    "PA":   ("administrative procedure", ["administrative procedure", "administrative decision"]),
    # European Convention on Human Rights
    "EMRK": ("human rights and ECHR", ["ECHR", "fundamental rights", "fair trial"]),
    "CEDH": ("human rights and ECHR", ["ECHR", "fundamental rights", "fair trial"]),
    # International mutual legal assistance
    "IRSG": ("mutual legal assistance", ["international mutual legal assistance", "extradition"]),
    "EIMP": ("mutual legal assistance", ["international mutual legal assistance", "extradition"]),
    # Asylum
    "AsylG": ("asylum law", ["asylum", "refugee", "protection from deportation"]),
    "LAsi": ("asylum law", ["asylum", "refugee", "protection from deportation"]),
    # Health insurance
    "KVG":  ("health insurance", ["health insurance", "medical treatment coverage"]),
    "LAMal": ("health insurance", ["health insurance", "medical treatment coverage"]),
    # Occupational pension
    "BVG":  ("occupational pension law", ["occupational pension", "pension fund", "disability pension"]),
    "LPP":  ("occupational pension law", ["occupational pension", "pension fund", "disability pension"]),
    # Unemployment insurance
    "AVIG": ("unemployment insurance", ["unemployment insurance", "earnings replacement"]),
    "LACI": ("unemployment insurance", ["unemployment insurance", "earnings replacement"]),
    # Military insurance
    "MVG":  ("military insurance law", ["military insurance", "military service injury"]),
    "LAM":  ("military insurance law", ["military insurance", "military service injury"]),
    # Supplementary benefits
    "ELG":  ("supplementary benefits", ["supplementary benefits", "social assistance"]),
    "LPC":  ("supplementary benefits", ["supplementary benefits", "social assistance"]),
    # Income compensation
    "EOG":  ("income compensation", ["income compensation", "maternity leave"]),
    "LAPG": ("income compensation", ["income compensation", "maternity leave"]),
    # Water protection
    "GSchG": ("water protection law", ["water protection", "water pollution"]),
    "LEaux": ("water protection law", ["water protection", "water pollution"]),
    # Nature and heritage protection
    "NHG":  ("nature and heritage protection", ["nature protection", "heritage site"]),
    "LPN":  ("nature and heritage protection", ["nature protection", "heritage site"]),
    # Forest law
    "WaG":  ("forest law", ["forest protection", "forest management"]),
    "LFo":  ("forest law", ["forest protection", "forest management"]),
    # Epidemic / public health
    "EpG":  ("epidemic and public health law", ["epidemic", "pandemic", "public health measures"]),
    "LEp":  ("epidemic and public health law", ["epidemic", "pandemic", "public health measures"]),
    # Anti-money laundering and financial market regulation
    "GwG":    ("anti-money laundering law", ["money laundering", "suspicious transaction", "due diligence"]),
    "LBA":    ("anti-money laundering law", ["money laundering", "suspicious transaction", "due diligence"]),
    "FINMAG": ("financial market supervision", ["financial market", "FINMA", "financial supervision"]),
    "LFINMA": ("financial market supervision", ["financial market", "FINMA", "financial supervision"]),
    "BEHG":   ("stock exchange and securities law", ["stock exchange", "insider trading", "securities"]),
    "LBVM":   ("stock exchange and securities law", ["stock exchange", "insider trading", "securities"]),
    "KAG":    ("collective investment schemes", ["investment fund", "collective investment"]),
    "LPCC":   ("collective investment schemes", ["investment fund", "collective investment"]),
    # Mergers
    "FusG": ("merger and transformation law", ["merger", "demerger", "company transformation"]),
    "LFus": ("merger and transformation law", ["merger", "demerger", "company transformation"]),
    # Public procurement
    "BoeB": ("public procurement", ["public procurement", "tender", "contract award"]),
    "LMP":  ("public procurement", ["public procurement", "tender", "contract award"]),
    # Data protection
    "DSG":  ("data protection law", ["data protection", "personal data", "privacy"]),
    "LPD":  ("data protection law", ["data protection", "personal data", "privacy"]),
    # Energy
    "EnG":  ("energy law", ["energy supply", "electricity", "renewable energy"]),
    "LEne": ("energy law", ["energy supply", "electricity", "renewable energy"]),
    # Agricultural land
    "BGBB": ("agricultural land law", ["agricultural land", "farmland protection"]),
    "LDFR": ("agricultural land law", ["agricultural land", "farmland protection"]),
}


LEGAL_CONCEPTS = [
    # ── Criminal procedure ────────────────────────────────────────────────────
    {
        "label": "pretrial detention",
        "area": "criminal procedure",
        "terms_de": ["untersuchungshaft", "sicherheitshaft", "haftverlangerung", "haftentlassung",
                     "haftrichter", "zwangsmassnahmen", "hafterstreckung"],
        "terms_fr": ["detention provisoire", "detention preventive", "mise en detention",
                     "juge des mesures", "mesures de contrainte", "prolongation de la detention"],
        "terms_it": ["carcerazione preventiva", "detenzione preventiva", "misure coercitive"],
        "terms_en": ["pretrial detention", "remand detention", "custody"],
    },
    {
        "label": "collusion risk",
        "area": "criminal procedure",
        "terms_de": ["kollusionsgefahr", "verdunkelungsgefahr", "beeinflussung von zeugen",
                     "beweisgefahrdung", "kollusionsrisiko"],
        "terms_fr": ["risque de collusion", "risque de subornation",
                     "entraver la recherche de la verite", "danger de collusion"],
        "terms_it": ["pericolo di collusione", "inquinamento delle prove"],
        "terms_en": ["collusion risk", "tampering with evidence"],
    },
    {
        "label": "flight risk",
        "area": "criminal procedure",
        "terms_de": ["fluchtgefahr", "fluchtrisiko", "flucht ins ausland", "untertauchen"],
        "terms_fr": ["risque de fuite", "danger de fuite", "prendre la fuite"],
        "terms_it": ["pericolo di fuga", "rischio di fuga"],
        "terms_en": ["flight risk", "risk of absconding"],
    },
    {
        "label": "recidivism risk",
        "area": "criminal procedure",
        "terms_de": ["wiederholungsgefahr", "ruckfallgefahr", "weiterbegehung"],
        "terms_fr": ["risque de recidive", "danger de recidive", "risque de reiter"],
        "terms_it": ["pericolo di recidiva", "rischio di recidiva"],
        "terms_en": ["recidivism risk", "risk of reoffending"],
    },
    {
        "label": "criminal sentencing",
        "area": "criminal law",
        "terms_de": ["strafzumessung", "freiheitsstrafe", "geldstrafe", "bedingte strafe",
                     "bewahrung", "vollzug", "strafmass", "strafrahmen", "gemeinnützige arbeit",
                     "landesverweisung", "fahrverbot", "busse"],
        "terms_fr": ["fixation de la peine", "peine privative de liberte", "peine pecuniaire",
                     "sursis", "mise a l'epreuve", "execution de la peine", "travail d'interet general",
                     "expulsion du territoire", "interdiction de conduire"],
        "terms_it": ["commisurazione della pena", "pena detentiva", "pena pecuniaria",
                     "sospensione condizionale", "lavoro di pubblica utilita"],
        "terms_en": ["criminal sentencing", "custodial sentence", "fine", "probation",
                     "expulsion from territory"],
    },
    {
        "label": "drug offences",
        "area": "criminal law",
        "terms_de": ["betaubungsmittel", "drogenhandel", "kokain", "heroin", "cannabis",
                     "drogen", "rauschgift", "bmg", "suchtmittel", "amphetamin"],
        "terms_fr": ["stupefiants", "trafic de drogue", "cocaine", "heroine", "cannabis",
                     "trafic de stupefiants", "narcotiques"],
        "terms_it": ["stupefacenti", "traffico di droga", "cocaina", "eroina", "cannabis"],
        "terms_en": ["drug offences", "narcotics", "drug trafficking", "cannabis"],
    },
    # ── Constitutional and general procedural ────────────────────────────────
    {
        "label": "proportionality",
        "area": "constitutional and procedural law",
        "terms_de": ["verhaltnismassigkeit", "verhaeltnismaessigkeit", "ultima ratio",
                     "angemessenheit", "geeignetheit", "erforderlichkeit"],
        "terms_fr": ["proportionnalite", "ultima ratio", "adequation", "necessite de la mesure"],
        "terms_it": ["proporzionalita", "ultima ratio", "adeguatezza"],
        "terms_en": ["proportionality", "ultima ratio", "necessity"],
    },
    {
        "label": "right to be heard",
        "area": "constitutional procedure",
        "terms_de": ["rechtliches gehor", "anspruch auf rechtliches gehor", "begrundungspflicht",
                     "gehorsanspruch", "akteneinsicht", "replikrecht"],
        "terms_fr": ["droit d'etre entendu", "obligation de motiver", "droit a une decision motivee",
                     "droit de reponse", "consultation du dossier"],
        "terms_it": ["diritto di essere sentito", "obbligo di motivazione", "diritto di replica"],
        "terms_en": ["right to be heard", "duty to give reasons", "right of reply"],
    },
    {
        "label": "appeal admissibility",
        "area": "procedure",
        "terms_de": ["beschwerdelegitimation", "eintreten", "nichteintreten",
                     "nicht eingetreten", "nicht einzutreten", "auf die beschwerde ist nicht einzutreten",
                     "beschwerde in strafsachen", "rechtsschutzinteresse", "schutzwurdiges interesse",
                     "kassationsbeschwerde", "staatsrechtliche beschwerde"],
        "terms_fr": ["recevabilite", "irrecevable", "recours en matiere",
                     "qualite pour recourir", "interet digne de protection",
                     "recours de droit public"],
        "terms_it": ["ammissibilita", "inammissibile", "ricorso", "legittimazione",
                     "interesse degno di protezione"],
        "terms_en": ["admissibility of appeal", "standing to appeal", "legal interest"],
    },
    {
        "label": "evidence assessment",
        "area": "procedure",
        "terms_de": ["beweiswurdigung", "willkurliche beweiswurdigung", "gutachten",
                     "sachverstandige", "beweislast", "augenschein", "beweisantrag",
                     "beweismassnahme", "zeuge", "willkur"],
        "terms_fr": ["appreciation des preuves", "expertise", "arbitraire", "charge de la preuve",
                     "inspection locale", "moyens de preuve", "temoignage"],
        "terms_it": ["apprezzamento delle prove", "perizia", "arbitrio", "onere della prova",
                     "ispezione locale", "testimonianza"],
        "terms_en": ["evidence assessment", "expert evidence", "burden of proof", "arbitrariness"],
    },
    {
        "label": "direct democracy and political rights",
        "area": "constitutional and public law",
        "terms_de": ["volksabstimmung", "volksinitiative", "referendum", "stimmrecht",
                     "wahl", "politische rechte", "abstimmung", "initiative", "volksbegehren",
                     "stimmberechtigt", "stimmzettel"],
        "terms_fr": ["votation populaire", "initiative populaire", "referendum",
                     "droit de vote", "droits politiques", "scrutin", "election",
                     "initiative cantonale"],
        "terms_it": ["votazione popolare", "iniziativa popolare", "referendum",
                     "diritto di voto", "diritti politici", "elezione"],
        "terms_en": ["popular vote", "popular initiative", "referendum",
                     "political rights", "right to vote"],
    },
    # ── Civil / private law ───────────────────────────────────────────────────
    {
        "label": "contract interpretation",
        "area": "private law",
        "terms_de": ["vertragsauslegung", "vertrauensprinzip", "wille der parteien",
                     "auslegung", "vertragslucke", "vertragsinhalt"],
        "terms_fr": ["interpretation du contrat", "principe de la confiance",
                     "volonte des parties", "lacune contractuelle"],
        "terms_it": ["interpretazione del contratto", "principio dell'affidamento",
                     "volonta delle parti"],
        "terms_en": ["contract interpretation", "principle of trust"],
    },
    {
        "label": "liability and damages",
        "area": "private law",
        "terms_de": ["haftung", "schadenersatz", "kausalitat", "adaquate kausalitat",
                     "schaden", "naturalrestitution", "produkthaftung", "staatshaftung"],
        "terms_fr": ["responsabilite", "dommages-interets", "causalite", "dommage",
                     "restitution en nature", "responsabilite du produit",
                     "responsabilite de l'etat"],
        "terms_it": ["responsabilita", "risarcimento del danno", "causalita", "danno",
                     "responsabilita dello stato"],
        "terms_en": ["liability", "damages", "causation", "product liability",
                     "state liability"],
    },
    {
        "label": "banking duty of care",
        "area": "banking and obligations law",
        "terms_de": ["banklagernd", "genehmigungsfiktion", "risikotransfer",
                     "sorgfaltspflicht der bank", "bankgeheimnis", "vermogensverwaltung",
                     "anlageberatung", "kreditvertrag"],
        "terms_fr": ["banque restante", "clause de transfert de risque",
                     "faute grave de la banque", "secret bancaire",
                     "gestion de fortune", "conseil en placement", "contrat de credit"],
        "terms_it": ["banca restante", "trasferimento del rischio",
                     "colpa grave della banca", "segreto bancario",
                     "gestione patrimoniale"],
        "terms_en": ["banking duty of care", "risk transfer clause", "bank secrecy",
                     "portfolio management"],
    },
    {
        "label": "maintenance and child support",
        "area": "family law",
        "terms_de": ["unterhalt", "kindesunterhalt", "unterhaltsbeitrag",
                     "besuchsrecht", "elterliche sorge", "obhut", "trennungsunterhalt",
                     "nachehelicher unterhalt", "konkubinat"],
        "terms_fr": ["contribution d'entretien", "entretien de l'enfant",
                     "droit de visite", "autorite parentale", "garde", "pension alimentaire"],
        "terms_it": ["contributo di mantenimento", "mantenimento del figlio",
                     "diritto di visita", "autorita parentale", "affidamento"],
        "terms_en": ["maintenance", "child support", "visitation", "parental authority",
                     "custody"],
    },
    {
        "label": "inheritance and testament",
        "area": "inheritance law",
        "terms_de": ["testament", "letztwillige verfugung", "erbfahigkeit",
                     "testierfahigkeit", "pflichtteil", "erbteilung", "erbschaft",
                     "erbvertrag", "herabsetzungsklage"],
        "terms_fr": ["testament", "disposition pour cause de mort", "capacite de tester",
                     "reserve hereditaire", "partage successoral", "succession",
                     "pacte successoral", "action en reduction"],
        "terms_it": ["testamento", "disposizione a causa di morte", "capacita di disporre",
                     "legittima", "divisione ereditaria", "patto successorio"],
        "terms_en": ["will", "testament", "testamentary capacity", "compulsory portion",
                     "estate partition"],
    },
    {
        "label": "debt enforcement and bankruptcy",
        "area": "debt enforcement",
        "terms_de": ["betreibung", "konkurs", "rechtsvorschlag", "rechtsoeffnung",
                     "pfandung", "nachlassvertrag", "betreibungsamt", "fortsetzungsbegehren"],
        "terms_fr": ["poursuite", "faillite", "mainlevee", "saisie", "opposition",
                     "sursis concordataire", "office des poursuites"],
        "terms_it": ["esecuzione", "fallimento", "rigetto dell'opposizione",
                     "pignoramento", "concordato", "ufficio di esecuzione"],
        "terms_en": ["debt enforcement", "bankruptcy", "attachment", "concordat"],
    },
    # ── Administrative and public law ─────────────────────────────────────────
    {
        "label": "migration and residence",
        "area": "migration law",
        "terms_de": ["aufenthaltsbewilligung", "niederlassungsbewilligung",
                     "wegweisung", "familiennachzug", "aufenthaltsrecht",
                     "auslanderin", "auslander", "integration", "einburgerung"],
        "terms_fr": ["autorisation de sejour", "renvoi", "regroupement familial",
                     "permis", "droit de sejour", "etrangere", "etranger",
                     "integration", "naturalisation"],
        "terms_it": ["permesso di dimora", "allontanamento", "ricongiungimento familiare",
                     "diritto di soggiorno", "straniero", "naturalizzazione"],
        "terms_en": ["residence permit", "removal", "family reunification",
                     "foreigners law", "naturalisation"],
    },
    {
        "label": "asylum and refugee status",
        "area": "asylum law",
        "terms_de": ["asyl", "fluechtling", "wegweisung", "asylgesuch",
                     "schutzbeduerfnis", "asylverfahren", "non-refoulement",
                     "vorlaufiger schutz", "durchsetzungshaft"],
        "terms_fr": ["asile", "refugie", "renvoi", "demande d'asile",
                     "protection", "non-refoulement", "protection provisoire"],
        "terms_it": ["asilo", "rifugiato", "allontanamento", "domanda d'asilo",
                     "protezione", "non-refoulement"],
        "terms_en": ["asylum", "refugee", "deportation", "asylum seeker",
                     "non-refoulement"],
    },
    {
        "label": "tax assessment",
        "area": "tax law",
        "terms_de": ["steuerveranlagung", "steuerhinterziehung", "steuerbetrug",
                     "verrechnungssteuer", "steuerpflicht", "steuerdomizil",
                     "quellensteuer", "steuerumgehung", "steuererlass",
                     "mehrwertsteuer", "einkommenssteuer", "vermögenssteuer"],
        "terms_fr": ["taxation", "soustraction fiscale", "fraude fiscale",
                     "impot anticipe", "assujettissement", "domicile fiscal",
                     "impot a la source", "evasion fiscale", "impot sur le revenu",
                     "impot sur la fortune"],
        "terms_it": ["tassazione", "sottrazione d'imposta", "frode fiscale",
                     "imposta anticipata", "domicilio fiscale", "imposta alla fonte"],
        "terms_en": ["tax assessment", "tax fraud", "tax evasion", "withholding tax",
                     "income tax", "wealth tax"],
    },
    {
        "label": "administrative procedure",
        "area": "administrative law",
        "terms_de": ["verwaltungsverfahren", "verfugung", "einsprache",
                     "verwaltungsbeschwerde", "konzession", "bewilligung",
                     "betriebsbewilligung", "subsidiaritat"],
        "terms_fr": ["procedure administrative", "decision administrative",
                     "recours administratif", "opposition", "concession",
                     "autorisation d'exploiter"],
        "terms_it": ["procedura amministrativa", "decisione amministrativa",
                     "opposizione", "concessione", "autorizzazione"],
        "terms_en": ["administrative procedure", "administrative decision",
                     "administrative appeal", "licence", "concession"],
    },
    {
        "label": "spatial planning and zoning",
        "area": "spatial planning law",
        "terms_de": ["raumplanung", "zonenplan", "baubewilligung", "nutzungsplanung",
                     "erschliessung", "bauzone", "umzonung", "zonengrenze",
                     "nutzungszone", "baupolizei", "richtplan", "quartierplan",
                     "baulinie", "bebauungsplan", "bauprojekt", "abstandsvorschrift"],
        "terms_fr": ["amenagement du territoire", "zone constructible",
                     "permis de construire", "plan d'affectation", "dezonage",
                     "limite de zone", "plan directeur", "plan de quartier",
                     "ligne de construction", "projet de construction"],
        "terms_it": ["pianificazione del territorio", "permesso di costruire",
                     "zona edificabile", "piano di utilizzazione", "piano direttore"],
        "terms_en": ["spatial planning", "zoning", "building permit", "rezoning",
                     "land use plan"],
    },
    {
        "label": "environmental protection",
        "area": "environmental law",
        "terms_de": ["umweltschutz", "umweltverschmutzung", "schadstoff",
                     "umweltvertraeglichkeit", "immission", "emissionen",
                     "gewasserschutz", "larmschutz", "altlast", "bodenschutz"],
        "terms_fr": ["protection de l'environnement", "pollution", "atteinte a l'environnement",
                     "immissions", "protection des eaux", "protection contre le bruit",
                     "site contamine", "protection du sol"],
        "terms_it": ["protezione dell'ambiente", "inquinamento", "immissioni",
                     "protezione delle acque", "protezione dal rumore"],
        "terms_en": ["environmental protection", "pollution", "environmental impact",
                     "noise protection", "contaminated site"],
    },
    {
        "label": "competition and antitrust",
        "area": "competition law",
        "terms_de": ["wettbewerb", "kartell", "marktmacht", "weko",
                     "kartellgesetz", "preisabsprache", "marktbeherrschung",
                     "wettbewerbskommission", "fusionskontrolle"],
        "terms_fr": ["concurrence", "entente", "position dominante", "comco",
                     "loi sur les cartels", "accord de prix", "domination du marche",
                     "commission de la concurrence", "controle des concentrations"],
        "terms_it": ["concorrenza", "accordo anticoncorrenziale", "posizione dominante",
                     "commissione della concorrenza", "controllo delle concentrazioni"],
        "terms_en": ["competition law", "antitrust", "cartel", "market dominance",
                     "merger control"],
    },
    {
        "label": "police and public security",
        "area": "public security law",
        "terms_de": ["uberwachung", "polizei", "sicherheitsmassnahme", "rayon",
                     "kontaktverbot", "gewaltschutzmassnahme", "polizeigesetz",
                     "sicherheitspolizei", "vorbeugungshaft", "gefährder"],
        "terms_fr": ["surveillance", "police", "mesure de securite", "rayon",
                     "interdiction de contact", "loi sur la police",
                     "prevention de la violence", "detention preventive de police"],
        "terms_it": ["sorveglianza", "polizia", "misura di sicurezza", "divieto di contatto",
                     "legge sulla polizia"],
        "terms_en": ["police law", "public security", "surveillance", "exclusion zone",
                     "contact ban"],
    },
    {
        "label": "epidemic and public health",
        "area": "public health law",
        "terms_de": ["epidemie", "pandemie", "seuchenschutz", "quarantane",
                     "schutzmasken", "covid", "covid-19", "coronavirus",
                     "sars-cov", "sars-cov-2", "impfpflicht", "isolierung",
                     "kontaktverbot", "seuchengesetz", "impfung", "schutzkonzept"],
        "terms_fr": ["epidemie", "pandemie", "quarantaine", "masques de protection",
                     "covid", "covid-19", "coronavirus", "sars-cov",
                     "obligation de vaccination", "isolement", "loi sur les epidemies",
                     "concept de protection"],
        "terms_it": ["epidemia", "pandemia", "quarantena", "mascherine", "covid",
                     "covid-19", "coronavirus", "sars-cov",
                     "obbligo di vaccinazione", "isolamento"],
        "terms_en": ["epidemic", "pandemic", "COVID", "COVID-19", "SARS-CoV",
                     "quarantine", "vaccination mandate", "public health measures"],
    },
    # ── Social insurance ───────────────────────────────────────────────────────
    {
        "label": "invalidity insurance",
        "area": "social insurance",
        "terms_de": ["invalidenversicherung", "invaliditat", "arbeitsunfahigkeit",
                     "eingliederungsmassnahmen", "rentenanspruch", "iv-stelle",
                     "invalidenrente", "eingliederung", "arbeitsfähigkeit"],
        "terms_fr": ["assurance-invalidite", "invalidite", "incapacite de travail",
                     "mesures de reinsertion", "rente", "office ai",
                     "rente d'invalidite", "readaptation"],
        "terms_it": ["assicurazione invalidita", "invalidita", "incapacita lavorativa",
                     "ufficio ai", "rendita d'invalidita", "reintegrazione"],
        "terms_en": ["invalidity insurance", "work incapacity", "disability benefit",
                     "vocational rehabilitation"],
    },
    {
        "label": "old age and survivors insurance",
        "area": "social insurance",
        "terms_de": ["altersversicherung", "ahv", "altersrente", "hinterlassenenversicherung",
                     "pensionierung", "altersleistung", "rentenalter", "witwenrente"],
        "terms_fr": ["assurance-vieillesse", "avs", "rente de vieillesse",
                     "assurance survivants", "retraite", "age de la retraite",
                     "rente de veuve"],
        "terms_it": ["assicurazione vecchiaia", "avs", "rendita di vecchiaia",
                     "assicurazione superstiti", "pensionamento"],
        "terms_en": ["old age insurance", "AHV pension", "survivors benefit", "retirement"],
    },
    {
        "label": "health insurance",
        "area": "health insurance law",
        "terms_de": ["krankenversicherung", "krankenkasse", "leistungspflicht",
                     "behandlungskosten", "grundversicherung", "prämienverbilligung",
                     "tarif", "tarifvertrag", "spitex", "kopfpramie",
                     "zusatzversicherung", "krankenkassenrecht"],
        "terms_fr": ["assurance-maladie", "caisse-maladie", "prestations medicales",
                     "assurance de base", "reduction de primes", "tarif medical",
                     "convention tarifaire"],
        "terms_it": ["assicurazione malattia", "cassa malati", "prestazioni mediche",
                     "assicurazione di base", "riduzione dei premi"],
        "terms_en": ["health insurance", "medical costs", "insurance coverage",
                     "premium subsidy"],
    },
    {
        "label": "occupational pension",
        "area": "pension law",
        "terms_de": ["berufliche vorsorge", "pensionskasse", "altersleistung",
                     "invalidenrente bvg", "freizugigkeit", "vorsorgeguthaben",
                     "ueberob ligatorisch", "uberbruckungsrente",
                     "vorsorgeeinrichtung", "teilliquidation", "freizugigkeitsguthaben",
                     "vorsorgepflicht", "sammeleinrichtung", "anschlusspflicht"],
        "terms_fr": ["prevoyance professionnelle", "caisse de pension",
                     "rente d'invalidite lpp", "libre passage",
                     "prestation de sortie", "avoir de prevoyance"],
        "terms_it": ["previdenza professionale", "cassa pensioni",
                     "libero passaggio", "avere previdenziale"],
        "terms_en": ["occupational pension", "pension fund", "vested benefit",
                     "portability"],
    },
    # ── Employment ────────────────────────────────────────────────────────────
    {
        "label": "employment and dismissal",
        "area": "employment law",
        "terms_de": ["kundigung", "arbeitsvertrag", "missbrachliche kundigung",
                     "lohn", "uberstunden", "arbeitsrecht", "arbeitnehmer",
                     "arbeitgeber", "probezeit", "fristlose entlassung",
                     "arbeitszeugnis", "konkurrenzverbot"],
        "terms_fr": ["conge", "licenciement", "contrat de travail", "salaire",
                     "heures supplementaires", "droit du travail", "travailleur",
                     "employeur", "periode d'essai", "resiliation immediate",
                     "certificat de travail", "clause de prohibition de concurrence"],
        "terms_it": ["disdetta", "licenziamento", "contratto di lavoro", "salario",
                     "diritto del lavoro", "lavoratore", "datore di lavoro",
                     "periodo di prova", "licenziamento immediato"],
        "terms_en": ["dismissal", "employment contract", "wrongful termination",
                     "wages", "probation period", "summary dismissal"],
    },
    # ── Human rights ──────────────────────────────────────────────────────────
    {
        "label": "ECHR and fundamental rights",
        "area": "human rights law",
        "terms_de": ["menschenrechte", "grundrechte", "faire verfahren", "emrk",
                     "verfahrensgarantien", "willkurverbot", "rechtsgleichheit",
                     "diskriminierungsverbot", "religionsfreiheit",
                     "meinungsfreiheit", "versammlungsfreiheit", "schutz der privatsphare"],
        "terms_fr": ["droits de l'homme", "droits fondamentaux", "proces equitable",
                     "cedh", "interdiction de l'arbitraire", "egalite de traitement",
                     "interdiction de la discrimination", "liberte de religion",
                     "liberte d'expression", "liberte de reunion", "respect de la vie privee"],
        "terms_it": ["diritti umani", "diritti fondamentali", "processo equo",
                     "cedu", "divieto di arbitrio", "uguaglianza",
                     "divieto di discriminazione", "liberta di religione",
                     "liberta di espressione"],
        "terms_en": ["ECHR", "human rights", "fair trial", "prohibition of arbitrariness",
                     "equality", "freedom of religion", "freedom of expression",
                     "right to privacy"],
    },
    # ── International ─────────────────────────────────────────────────────────
    {
        "label": "international mutual legal assistance",
        "area": "international law",
        "terms_de": ["rechtshilfe", "auslieferung", "internationale rechtshilfe",
                     "rechtshilfeersuchen", "internationale amtshilfe"],
        "terms_fr": ["entraide judiciaire", "extradition", "entraide internationale",
                     "commission rogatoire", "assistance administrative internationale"],
        "terms_it": ["assistenza giudiziaria", "estradizione", "rogatoria",
                     "assistenza amministrativa internazionale"],
        "terms_en": ["mutual legal assistance", "extradition", "rogatory",
                     "international administrative assistance"],
    },
    # ── Road traffic ──────────────────────────────────────────────────────────
    {
        "label": "road traffic and accidents",
        "area": "traffic law",
        "terms_de": ["strassenverkehr", "verkehrsunfall", "fahrausweis",
                     "fahruntuechtig", "entzug des fahrausweises",
                     "geschwindigkeitsuberschreitung", "alkohol am steuer",
                     "warnungsentzug", "sicherungsentzug"],
        "terms_fr": ["circulation routiere", "accident de la route",
                     "permis de conduire", "retrait du permis",
                     "exces de vitesse", "conduite en etat d'ivresse",
                     "retrait d'admonestation", "retrait de securite"],
        "terms_it": ["circolazione stradale", "incidente stradale",
                     "licenza di condurre", "revoca della licenza",
                     "eccesso di velocita", "guida in stato di ebbrezza"],
        "terms_en": ["road traffic", "traffic accident", "driving licence revocation",
                     "speeding", "drunk driving"],
    },
    # ── Data protection ───────────────────────────────────────────────────────
    {
        "label": "data protection and privacy",
        "area": "data protection law",
        "terms_de": ["datenschutz", "personendaten", "datenbearbeitung",
                     "auskunftsrecht", "informationsrecht", "datenweitergabe"],
        "terms_fr": ["protection des donnees", "donnees personnelles",
                     "traitement des donnees", "droit d'acces", "transmission de donnees"],
        "terms_it": ["protezione dei dati", "dati personali",
                     "trattamento dei dati", "diritto di accesso"],
        "terms_en": ["data protection", "personal data", "right of access", "data transfer"],
    },
]


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", value.casefold()).strip()


def unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if value})


def batched(values: list[str], size: int = 900) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_json(raw: str | bytes) -> dict:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def is_notification_paragraph(text: str) -> bool:
    return bool(NOTIFICATION_PATTERNS.search(text[:600]))


def derive_court_base(pattern: str, segments: dict, citation: str) -> str:
    if pattern == "court_bge":
        volume = segments.get("volume")
        division = segments.get("division")
        page = segments.get("page")
        if volume and division and page:
            return f"BGE {volume} {division} {page}"
    if segments.get("docket"):
        return str(segments["docket"])
    return re.sub(r"\s+E\..*$", "", citation).strip()


def derive_docket_prefix(segments: dict) -> str | None:
    if segments.get("docket_prefix"):
        return str(segments["docket_prefix"])
    chamber = segments.get("court_chamber")
    area = segments.get("legal_area_code")
    if chamber and area:
        return f"{chamber}{area}"
    return str(area) if area else None


def infer_language(text: str) -> str:
    norm = normalize_text(text[:2000])
    scores = {
        "de": sum(norm.count(w) for w in [" der ", " die ", " das ", " und ", " nicht ",
                                           " beschwerde", " gericht", " verfugung"]),
        "fr": sum(norm.count(w) for w in [" le ", " la ", " les ", " que ", " recours ",
                                           " droit ", " tribunal", " arret"]),
        "it": sum(norm.count(w) for w in [" il ", " la ", " che ", " ricorso ",
                                           " diritto ", " della ", " tribunale", " sentenza"]),
    }
    best, score = max(scores.items(), key=lambda item: item[1])
    return best if score else "unknown"


def classify_role(
    pattern: str,
    segments: dict,
    base_source_count: int,
    base_text_ref_count: int,
    notification: bool,
) -> list[str]:
    roles = []
    consideration = str(segments.get("consideration") or "")
    if pattern == "court_bge":
        roles.append("published_leading_decision")
    else:
        roles.append("unpublished_federal_decision")
    if notification:
        roles.append("notification_paragraph")
    elif consideration and consideration[0].isdigit():
        roles.append("reasoning_consideration")
    elif consideration:
        roles.append("facts_or_procedural_history")
    if base_text_ref_count >= 25:
        roles.append("frequently_cited_authority")
    if base_source_count >= 5:
        roles.append("multi_consideration_decision")
    return roles


def extract_references_from_edges(edges: list[str]) -> tuple[list[str], list[str], list[str]]:
    statutes = []
    bge_cases = []
    docket_cases = []
    for target in edges:
        if target.startswith("Art."):
            statutes.append(target)
        elif BGE_BASE_RE.search(target):
            bge_cases.append(target)
        elif DOCKET_BASE_RE.search(target):
            docket_cases.append(target)
    return unique_sorted(statutes), unique_sorted(bge_cases), unique_sorted(docket_cases)


def extract_law_codes(statutes: list[str], text: str) -> list[str]:
    codes = []
    for statute in statutes:
        codes.extend(ARTICLE_CODE_RE.findall(statute))
    codes.extend(ARTICLE_CODE_RE.findall(text[:3000]))
    for code in LAW_CODE_AREAS:
        if re.search(rf"\b{re.escape(code)}\b", text):
            codes.append(code)
    cleaned = []
    for code in codes:
        code = code.rstrip(".,;)")
        if code in NON_LAW_CODE_TOKENS:
            continue
        if len(code) < 2:
            continue
        cleaned.append(code)
    return unique_sorted(cleaned)


def match_legal_concepts(
    text: str, law_codes: list[str]
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    norm = normalize_text(text)
    labels = []
    matched_terms: dict[str, list[str]] = defaultdict(list)
    areas = []

    for concept in LEGAL_CONCEPTS:
        concept_hits = []
        for key in ("terms_de", "terms_fr", "terms_it", "terms_en"):
            for term in concept[key]:
                if normalize_text(term) in norm:
                    concept_hits.append(term)
        if concept_hits:
            labels.append(concept["label"])
            areas.append(concept["area"])
            matched_terms[concept["label"]].extend(unique_sorted(concept_hits))

    for code in law_codes:
        if code in LAW_CODE_AREAS:
            area, code_labels = LAW_CODE_AREAS[code]
            areas.append(area)
            labels.extend(code_labels)

    return unique_sorted(labels), {k: unique_sorted(v) for k, v in matched_terms.items()}, unique_sorted(areas)


def legal_area_from_structure(
    pattern: str, segments: dict, law_code_areas: list[str]
) -> str:
    if pattern == "court_bge":
        division = segments.get("division")
        if division in BGE_DIVISION_AREAS:
            return BGE_DIVISION_AREAS[division]
    prefix = derive_docket_prefix(segments)
    if prefix in DOCKET_PREFIX_AREAS:
        return DOCKET_PREFIX_AREAS[prefix]
    if law_code_areas:
        return law_code_areas[0]
    return "unknown"


def fetch_segment_metadata(conn: sqlite3.Connection, citations: list[str]) -> dict[str, dict]:
    out = {}
    for chunk in batched(citations):
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT citation, pattern, subfamily, segments_json
            FROM source_citations
            WHERE dataset = 'court_considerations'
              AND citation IN ({placeholders})
            """,
            chunk,
        )
        for citation, pattern, subfamily, segments_json in rows:
            out[citation] = {
                "pattern": pattern,
                "subfamily": subfamily,
                "segments": load_json(segments_json),
            }
    return out


def fetch_edges(conn: sqlite3.Connection, citations: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for chunk in batched(citations):
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT source, target
            FROM edges
            WHERE dataset = 'court_considerations'
              AND source IN ({placeholders})
            """,
            chunk,
        )
        for source, target in rows:
            out[source].append(target)
    return out


def load_base_counts(conn: sqlite3.Connection) -> dict[str, tuple[int, int]]:
    return {
        row[0]: (int(row[1]), int(row[2]))
        for row in conn.execute(
            """
            SELECT option_value, source_count, text_ref_count
            FROM option_registry
            WHERE dataset = 'court_considerations'
              AND segment_key = 'court_base'
            """
        )
    }


def build_card(
    citation: str,
    text: str,
    metadata: dict,
    outgoing_refs: list[str],
    base_counts: dict[str, tuple[int, int]],
) -> dict:
    pattern = metadata.get("pattern") or "unknown"
    subfamily = metadata.get("subfamily") or "unknown"
    segments = metadata.get("segments") or {}
    court_base = derive_court_base(pattern, segments, citation)
    base_source_count, base_text_ref_count = base_counts.get(court_base, (1, 0))
    statutes, cited_bge, cited_dockets = extract_references_from_edges(outgoing_refs)
    law_codes = extract_law_codes(statutes, text)
    issue_labels, matched_terms, concept_areas = match_legal_concepts(text, law_codes)
    legal_area = legal_area_from_structure(pattern, segments, concept_areas)
    notification = is_notification_paragraph(text)
    roles = classify_role(pattern, segments, base_source_count, base_text_ref_count, notification)

    bridge_parts = [
        citation,
        court_base,
        legal_area,
        " ".join(issue_labels),
        " ".join(law_codes),
        " ".join(statutes[:20]),
        " ".join(
            f"{label}: {'; '.join(terms)}" for label, terms in matched_terms.items()
        ),
    ]
    text_excerpt = re.sub(r"\s+", " ", text).strip()[:1200]
    if len(text) > 1200:
        text_excerpt += "..."

    summary_bits = []
    if legal_area != "unknown":
        summary_bits.append(f"Swiss Federal Supreme Court consideration in {legal_area}")
    else:
        summary_bits.append("Swiss Federal Supreme Court consideration")
    if notification:
        summary_bits.append("procedural notification paragraph")
    elif issue_labels:
        summary_bits.append("signals " + ", ".join(issue_labels[:8]))
    if law_codes:
        summary_bits.append("linked to " + ", ".join(law_codes[:8]))
    if pattern == "court_bge":
        summary_bits.append("published BGE authority")

    return {
        "citation": citation,
        "court_base": court_base,
        "family": "court",
        "pattern": pattern,
        "subfamily": subfamily,
        "language": infer_language(text),
        "legal_area": legal_area,
        "is_notification_paragraph": notification,
        "issue_labels_en": issue_labels,
        "matched_terms_multilingual": matched_terms,
        "law_codes": law_codes,
        "statutes_cited": statutes[:80],
        "court_cases_cited": unique_sorted(cited_bge + cited_dockets)[:80],
        "authority_role": roles,
        "structural": {
            "bge_division": segments.get("division"),
            "docket_prefix": derive_docket_prefix(segments),
            "legal_area_code": segments.get("legal_area_code"),
            "court_chamber": segments.get("court_chamber"),
            "decision_year": segments.get("decision_year"),
            "decision_date": segments.get("decision_date"),
            "consideration": segments.get("consideration") or segments.get("pinpoint"),
            "court_base_source_count": base_source_count,
            "court_base_text_ref_count": base_text_ref_count,
        },
        "summary_en_proxy": ". ".join(summary_bits) + ".",
        "text_excerpt_original": text_excerpt,
        "retrieval_text_en": " | ".join(part for part in bridge_parts if part),
        "provenance": {
            "court_text": str(COURT_CSV.relative_to(ROOT)),
            "segments": str(SEGMENT_DB.relative_to(ROOT)),
            "citation_graph": str(GRAPH_DB.relative_to(ROOT)),
            "method": "deterministic_authority_card_v4_no_gold_no_query",
        },
    }


def iter_court_rows(
    path: Path,
    limit: int | None,
    chunk_index: int | None = None,
    num_chunks: int | None = None,
    citation_filter: set[str] | None = None,
) -> Iterable[tuple[str, str]]:
    use_chunks = chunk_index is not None and num_chunks is not None
    row_number = 0
    yielded = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            citation = (row.get("citation") or "").strip()
            text = row.get("text") or ""
            if not citation:
                row_number += 1
                continue
            if citation_filter is not None and citation not in citation_filter:
                row_number += 1
                continue
            if use_chunks and row_number % num_chunks != chunk_index:
                row_number += 1
                continue
            row_number += 1
            yield citation, text
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def build_cards(
    limit: int | None,
    batch_size: int,
    output_path: Path,
    chunk_index: int | None = None,
    num_chunks: int | None = None,
    citation_filter: set[str] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seg_conn = sqlite3.connect(SEGMENT_DB)
    graph_conn = sqlite3.connect(GRAPH_DB)
    base_counts = load_base_counts(seg_conn)

    written = 0
    missing_metadata = 0
    issue_counter: Counter[str] = Counter()
    area_counter: Counter[str] = Counter()
    pattern_counter: Counter[str] = Counter()
    notification_count = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as out_handle:
        batch: list[tuple[str, str]] = []
        for citation, text in iter_court_rows(
            COURT_CSV, limit, chunk_index, num_chunks, citation_filter
        ):
            batch.append((citation, text))
            if len(batch) < batch_size:
                continue
            written, missing_metadata, notification_count = flush_batch(
                batch, seg_conn, graph_conn, base_counts, out_handle,
                written, missing_metadata, issue_counter, area_counter,
                pattern_counter, notification_count,
            )
            batch.clear()
        if batch:
            written, missing_metadata, notification_count = flush_batch(
                batch, seg_conn, graph_conn, base_counts, out_handle,
                written, missing_metadata, issue_counter, area_counter,
                pattern_counter, notification_count,
            )

    seg_conn.close()
    graph_conn.close()

    summary_path = output_path.with_suffix(".summary.json")
    summary = {
        "output_path": str(output_path),
        "chunk_index": chunk_index,
        "num_chunks": num_chunks,
        "cards_written": written,
        "missing_segment_metadata": missing_metadata,
        "notification_paragraphs": notification_count,
        "top_issue_labels": issue_counter.most_common(60),
        "legal_areas": area_counter.most_common(60),
        "patterns": pattern_counter.most_common(),
        "limit": limit,
        "citation_filter_count": len(citation_filter) if citation_filter is not None else None,
        "method": "deterministic_authority_card_v4_no_gold_no_query",
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {"chunk": chunk_index, "written": written,
             "missing": missing_metadata, "notifications": notification_count},
            ensure_ascii=False,
        )
    )


def flush_batch(
    batch: list[tuple[str, str]],
    seg_conn: sqlite3.Connection,
    graph_conn: sqlite3.Connection,
    base_counts: dict[str, tuple[int, int]],
    out_handle,
    written: int,
    missing_metadata: int,
    issue_counter: Counter[str],
    area_counter: Counter[str],
    pattern_counter: Counter[str],
    notification_count: int,
) -> tuple[int, int, int]:
    citations = [citation for citation, _ in batch]
    metadata_by_citation = fetch_segment_metadata(seg_conn, citations)
    edges_by_citation = fetch_edges(graph_conn, citations)

    for citation, text in batch:
        metadata = metadata_by_citation.get(citation)
        if not metadata:
            missing_metadata += 1
            continue
        card = build_card(
            citation, text, metadata, edges_by_citation.get(citation, []), base_counts
        )
        out_handle.write(json.dumps(card, ensure_ascii=False, sort_keys=True) + "\n")
        written += 1
        pattern_counter[card["pattern"]] += 1
        area_counter[card["legal_area"]] += 1
        issue_counter.update(card["issue_labels_en"])
        if card["is_notification_paragraph"]:
            notification_count += 1
    return written, missing_metadata, notification_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=5000,
        help="Rows to process per chunk. Use 0 for all rows.",
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--chunk-index", type=int, default=None,
                        help="0-based index of this worker chunk.")
    parser.add_argument("--num-chunks", type=int, default=None,
                        help="Total number of parallel chunks.")
    parser.add_argument("--citations", default="",
                        help="Optional semicolon-separated citation IDs to filter.")
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    citation_filter = (
        {v.strip() for v in args.citations.split(";") if v.strip()} or None
    )

    output = args.output
    if args.chunk_index is not None:
        output = output.with_name(
            output.stem + f"_chunk{args.chunk_index:02d}" + output.suffix
        )

    build_cards(limit, args.batch_size, output, args.chunk_index, args.num_chunks,
                citation_filter)


if __name__ == "__main__":
    main()
