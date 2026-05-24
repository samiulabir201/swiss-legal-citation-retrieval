"""Precompute the Phase 1+2+3 dossier feature cache for the hybrid rerank notebook.

Run this ONCE locally. Produces a single .npz file in the exact format the
notebook's Phase 3 cache-load path expects (np.savez_compressed with keys
'<qid>__dids' and '<qid>__feat' per query).

Output: research/hybrid_rerank_final/cache/dossier_features.npz

Upload the file to Drive at:
  /content/drive/MyDrive/swiss_law/research/hybrid_rerank_final/cache/dossier_features.npz

(Same path layout the notebook's CACHE_DIR autodetects.)
"""
from __future__ import annotations
import json
import gzip
import time
import re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np


# =============================================================================
# Path setup (matches notebook's local-env autodetect)
# =============================================================================
DRIVE = Path(r"E:\swiss_citation_extraction")
SNAPSHOT_DIR_A = DRIVE / "research" / "anchor_funnel_val001_v7" / "snapshot"
SNAPSHOT_DIR_B = SNAPSHOT_DIR_A / "snapshot"
SNAPSHOT_DIR = SNAPSHOT_DIR_A if (SNAPSHOT_DIR_A / "config.json").exists() else SNAPSHOT_DIR_B
OUT_DIR = DRIVE / "research" / "hybrid_rerank_final"
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = CACHE_DIR / "dossier_features.npz"

assert SNAPSHOT_DIR.exists() and (SNAPSHOT_DIR / "config.json").exists(), \
    f"Snapshot dir missing: {SNAPSHOT_DIR}"

print(f"[paths] snapshot: {SNAPSHOT_DIR}", flush=True)
print(f"[paths] output:   {OUT_PATH}", flush=True)


# =============================================================================
# Phase 1 - Warm-boot (mirrors notebook cell 4)
# =============================================================================
def warm_boot():
    print(f"\n[Phase 1] warm-boot ...", flush=True)
    t0 = time.time()

    with gzip.open(SNAPSHOT_DIR / "corpus_snapshot.json.gz", "rt", encoding="utf-8") as f:
        _c = json.load(f)
    search_text         = {d: r["ct"] for d, r in _c.items()}
    doc_meta            = {d: {"citation": r["cit"], "family": r["fam"],
                                 "court_base": r["cb"], "paragraph_role": r["pr"],
                                 "language": r["ln"]} for d, r in _c.items()}
    doc_statute_anchors = {d: set(r["sa"]) for d, r in _c.items()}
    _doc_to_concepts    = {d: set(r["cn"]) for d, r in _c.items()}
    _doc_to_terms       = {d: set(r["tm"]) for d, r in _c.items()}
    del _c

    _pq = json.loads((SNAPSHOT_DIR / "per_query_snapshot.json").read_text(encoding="utf-8"))
    PER_QUERY = {qid: {"final_topk": r["final_topk"],
                        "curve": {int(k) if str(k).isdigit() else k:
                                  (tuple(v) if isinstance(v, list) else v)
                                  for k, v in r.get("curve", {}).items()},
                        "channel_recalls": r.get("channel_recalls", {})}
                 for qid, r in _pq.items()}

    _gold_sets = json.loads((SNAPSHOT_DIR / "gold_doc_sets.json").read_text(encoding="utf-8"))
    ALL_GOLD_DOC_SET = {qid: set(lst) for qid, lst in _gold_sets.items()}
    QUERY_IDS = sorted(PER_QUERY.keys())

    ALL_TARGETS = json.loads((SNAPSHOT_DIR / "all_targets.json").read_text(encoding="utf-8"))

    print(f"  warm-boot done in {time.time()-t0:.1f}s "
          f"({len(doc_meta):,} docs, {len(PER_QUERY)} queries)", flush=True)
    return dict(search_text=search_text, doc_meta=doc_meta,
                doc_statute_anchors=doc_statute_anchors,
                _doc_to_concepts=_doc_to_concepts, _doc_to_terms=_doc_to_terms,
                PER_QUERY=PER_QUERY, ALL_GOLD_DOC_SET=ALL_GOLD_DOC_SET,
                QUERY_IDS=QUERY_IDS, ALL_TARGETS=ALL_TARGETS)


# =============================================================================
# Phase 2 - Helpers (verbatim copy of notebook cell 6)
# =============================================================================
_STATUTE_RE = re.compile(
    r"\b(?:Art(?:icle)?\.?\s*)?(\d+[a-z]?)\s*(?:Abs\.?\s*\d+)?\s*"
    r"(?:lit\.?\s*[a-z])?\s*"
    r"(StPO|StGB|ZGB|OR|BGG|ZPO|BV|IVG|ATSG|SchKG|MWSTG|UWG|UVG|AHVG|BVG|"
    r"AVIG|AsylG|BGE|ATF|StG|VStrR|EleG|EnG|FINIG|FinfraG|GwG|KVG|VPG|MEDBG|"
    r"BG|SR|Cst\.|CPC|CC|CO|CP|CPP|LP|LCart|LTF|LPGA|LAI)\b",
    re.IGNORECASE)


def canonicalize_statute(s: str) -> str:
    s = s.strip()
    m = _STATUTE_RE.search(s)
    if m:
        return f"{m.group(1)} {m.group(2).strip()}"
    return s


def canon_set(values):
    out = set()
    for v in values:
        c = canonicalize_statute(v)
        if c:
            out.add(c)
    return out


_DE_RULE_OPENER = re.compile(
    r"\b(?:Nach|Gemäss|Gemaess|Im Sinne (?:von|der)|Laut|Aufgrund (?:von|des)|"
    r"Bei|Entsprechend)\s+(?:Art\.?|Artikel|§)\s*\d", re.IGNORECASE)
_DE_DOCTRINE = re.compile(
    r"\b(?:Rechtsprechung|st(?:ändige|aendige)\s+Rechtsprechung|Lehre und Rechtsprechung|"
    r"bundesgerichtliche?n?\s+Rechtsprechung|nach\s+(?:bundesgerichtlicher|herrschender)\s+"
    r"(?:Praxis|Lehre|Auffassung)|herrschende\s+Lehre)\b", re.IGNORECASE)
_DE_RULE_VERB = re.compile(
    r"\b(?:bestimmt|sieht\s+vor|verlangt|setzt\s+voraus|erfordert|ordnet\s+an|statuiert)\b",
    re.IGNORECASE)
_FR_RULE_OPENER = re.compile(
    r"\b(?:Selon|Conformément\s+à|Conformement\s+a|Aux\s+termes\s+de|En\s+vertu\s+de|"
    r"D'après|D'apres|Suivant)\s+(?:l'?art\.?|l'?article)\s*\d", re.IGNORECASE)
_FR_DOCTRINE = re.compile(
    r"\b(?:selon\s+la\s+jurisprudence|d'après\s+la\s+jurisprudence|"
    r"jurisprudence\s+constante|selon\s+la\s+doctrine|la\s+doctrine\s+(?:et|admet|considère))\b",
    re.IGNORECASE)
_FR_RULE_VERB = re.compile(
    r"\b(?:dispose|prévoit|prevoit|exige|requiert|ordonne|stipule)\b", re.IGNORECASE)
_CITE_CLUSTER = re.compile(
    r"\((?:[^()]*?(?:BGE|ATF|BGer|TF)\s+\d+[^()]{0,200}?){3,}\)", re.IGNORECASE)

_DE_DISPOSITIV = re.compile(
    r"\b(?:Demnach\s+erkennt|wird\s+erkannt|Das\s+Bundesgericht\s+(?:erkennt|hat\s+entschieden):|"
    r"Dispositiv|Urteil(?:sdispositiv)?|"
    r"Die\s+Beschwerde\s+wird\s+(?:gutgeheissen|abgewiesen|nicht\s+eingetreten))\b",
    re.IGNORECASE)
_DE_FACTS = re.compile(
    r"\b(?:Sachverhalt|In\s+Sachen|A\.-|B\.-|C\.-|gegen\s+(?:die|den)\s+(?:Verfügung|Entscheid))",
    re.IGNORECASE)
_DE_COSTS = re.compile(
    r"\b(?:Gerichtskosten|Verfahrenskosten|Parteientschädigung|Kostenfolge)\b", re.IGNORECASE)
_FR_DISPOSITIV = re.compile(
    r"\b(?:Par\s+ces\s+motifs|Le\s+Tribunal\s+(?:fédéral|federal)\s+prononce|"
    r"Le\s+recours\s+est\s+(?:admis|rejeté|rejete|irrecevable))\b", re.IGNORECASE)
_FR_SIGNATURE = re.compile(r"\b(?:Le\s+président|Le\s+greffier|La\s+greffière)\b", re.IGNORECASE)
_FR_COSTS = re.compile(
    r"\b(?:frais\s+judiciaires|dépens|depens|émolument\s+judiciaire)\b", re.IGNORECASE)

_HARDNEG_ROLES = {"dispositif", "costs", "facts", "procedural_history",
                  "admissibility", "signature"}


def doctrinal_density(text: str, lang: str = "?") -> float:
    if not text:
        return 0.0
    score = 0.0
    score += 3 * len(_DE_RULE_OPENER.findall(text))
    score += 3 * len(_DE_DOCTRINE.findall(text))
    score += 1 * len(_DE_RULE_VERB.findall(text))
    score += 3 * len(_FR_RULE_OPENER.findall(text))
    score += 3 * len(_FR_DOCTRINE.findall(text))
    score += 1 * len(_FR_RULE_VERB.findall(text))
    score += 2 * min(len(_CITE_CLUSTER.findall(text)), 4)
    sd_count = len(_STATUTE_RE.findall(text))
    sd_per_1k = sd_count / max(1, len(text) / 1000.0)
    score += min(sd_per_1k, 6.0) * 0.5
    if _DE_DISPOSITIV.search(text) or _FR_DISPOSITIV.search(text):
        score -= 5
    if _DE_FACTS.search(text):
        score -= 3
    if _DE_COSTS.search(text) or _FR_COSTS.search(text):
        score -= 3
    if _FR_SIGNATURE.search(text):
        score -= 3
    return max(-10.0, min(20.0, score))


def is_hard_neg(text: str, role: str) -> bool:
    if role and role.lower() in _HARDNEG_ROLES:
        return True
    if _DE_DISPOSITIV.search(text or "") or _FR_DISPOSITIV.search(text or ""):
        return True
    if _DE_COSTS.search(text or "") or _FR_COSTS.search(text or ""):
        return True
    if _FR_SIGNATURE.search(text or ""):
        return True
    return False


_QUERY_AREA_TO_CHAMBERS = {
    "criminal":      {"1B", "6B", "7B", "IV"},
    "civil":         {"4A", "5A", "I", "II", "III"},
    "administrative":{"2C", "8C", "9C"},
    "public":        {"1C", "2C"},
    "social":        {"8C", "9C", "U"},
    "labor":         {"4A", "8C"},
    "tax":           {"2C"},
    "asylum":        {"E", "F"},
    "intellectual":  {"4A"},
}

_CHAMBER_DOCKET = re.compile(
    r"\b(?:BGE\s+\d+\s+([IVX]+)|(\d[A-Z]+)[._ ]?\d|\b([1-9][A-Z])\b)",
    re.IGNORECASE)


def chamber_of(citation: str) -> str | None:
    if not citation:
        return None
    m = _CHAMBER_DOCKET.search(citation)
    if not m:
        return None
    for g in m.groups():
        if g:
            return g.upper()
    return None


_LEGAL_AREA_PATTERNS = {
    "criminal":      re.compile(r"\b(criminal|strafe|strafrecht|penal|p[eé]nal|"
                                 r"d[eé]tention|untersuchungshaft|fluchtgefahr|"
                                 r"verdachts|stpo|strafverfahren)\b", re.IGNORECASE),
    "civil":         re.compile(r"\b(civil|zivil|contract|obligation|schadenersatz|"
                                 r"ehe|scheidung|erbrecht|ZGB|inheritance|tort)\b",
                                 re.IGNORECASE),
    "administrative":re.compile(r"\b(administrative|verwalt|publique|asyl|"
                                 r"ausl[aä]nder|migration|naturalisation)\b",
                                 re.IGNORECASE),
    "tax":           re.compile(r"\b(tax|steuer|mwst|imp[oô]t|imposition|"
                                 r"taxation|vat)\b", re.IGNORECASE),
    "social":        re.compile(r"\b(social|invalidity|pension|versicherung|"
                                 r"ahv|iv|uvg|bvg|disability|sozialversicher)\b",
                                 re.IGNORECASE),
    "labor":         re.compile(r"\b(labor|labour|arbeitsrecht|travail|"
                                 r"employment|k[uü]ndigung|dismissal|"
                                 r"arbeitsvertrag)\b", re.IGNORECASE),
    "intellectual":  re.compile(r"\b(intellectual|trademark|marke|patent|"
                                 r"urheber|copyright|design)\b", re.IGNORECASE),
}


def query_legal_areas(qid: str, ALL_TARGETS) -> set[str]:
    kws = ALL_TARGETS.get(qid, {}).get("legal_area_keywords", []) or []
    text = " ".join(str(k) for k in kws)
    areas = {area for area, rx in _LEGAL_AREA_PATTERNS.items() if rx.search(text)}
    if not areas:
        areas.add("any")
    return areas


def allowed_chambers(qid: str, ALL_TARGETS) -> set[str]:
    areas = query_legal_areas(qid, ALL_TARGETS)
    if "any" in areas:
        return set()
    chambers = set()
    for a in areas:
        chambers |= _QUERY_AREA_TO_CHAMBERS.get(a, set())
    return chambers


# =============================================================================
# Phase 3 - Precompute
# =============================================================================
def precompute(state: dict) -> dict:
    search_text = state["search_text"]
    doc_meta = state["doc_meta"]
    doc_statute_anchors = state["doc_statute_anchors"]
    _doc_to_concepts = state["_doc_to_concepts"]
    _doc_to_terms = state["_doc_to_terms"]
    PER_QUERY = state["PER_QUERY"]
    QUERY_IDS = state["QUERY_IDS"]
    ALL_TARGETS = state["ALL_TARGETS"]

    print(f"\n[Phase 3] precompute over {len(doc_meta):,} docs", flush=True)
    TEXT_TRUNC_REGEX = 2500

    # --- 3a. Per-doc cached features (query-independent) ---
    print(f"[3a] per-doc cached features (trunc={TEXT_TRUNC_REGEX}) ...", flush=True)
    t0 = time.time()
    _doc_doctrinal = {}
    _doc_hardneg = {}
    _doc_canon_statutes = {}
    _doc_lead_canon_statutes = {}
    _doc_chamber = {}
    n = 0
    for did, m in doc_meta.items():
        text = (search_text.get(did, "") or "")[:TEXT_TRUNC_REGEX]
        role = m.get("paragraph_role", "")
        cit = m.get("citation", "") or ""
        _doc_doctrinal[did] = doctrinal_density(text, m.get("language", "?"))
        _doc_hardneg[did] = 1 if is_hard_neg(text, role) else 0
        _doc_canon_statutes[did] = canon_set(doc_statute_anchors.get(did, set()))
        lead = text[:200]
        _doc_lead_canon_statutes[did] = canon_set(
            _m.group(0) for _m in _STATUTE_RE.finditer(lead))
        _doc_chamber[did] = chamber_of(cit) or ""
        n += 1
        if n % 50_000 == 0:
            print(f"  ...{n:,}/{len(doc_meta):,}  ({time.time()-t0:.1f}s)",
                  flush=True)
    print(f"  3a done in {time.time()-t0:.1f}s", flush=True)

    # --- 3b. Co-citation index ---
    print(f"[3b] co-citation index ...", flush=True)
    t0 = time.time()
    _statute_cite_count = Counter()
    _court_case_to_peers = defaultdict(set)
    union = set()
    for qid in QUERY_IDS:
        union.update(PER_QUERY[qid]["final_topk"])
    for did in union:
        m = doc_meta.get(did, {})
        if m.get("family") == "court":
            for s in _doc_canon_statutes.get(did, set()):
                _statute_cite_count[s] += 1
            cb = m.get("court_base") or ""
            if cb:
                _court_case_to_peers[cb].add(did)
    print(f"  3b done in {time.time()-t0:.1f}s  "
          f"({len(_statute_cite_count):,} statutes, "
          f"{len(_court_case_to_peers):,} cases)", flush=True)

    def co_cite_count(did: str) -> int:
        m = doc_meta.get(did, {})
        if m.get("family") == "law":
            canon = _doc_canon_statutes.get(did, set())
            if not canon:
                return 0
            cit_canon = canonicalize_statute(m.get("citation", ""))
            if cit_canon in _statute_cite_count:
                return _statute_cite_count[cit_canon]
            return max((_statute_cite_count.get(s, 0) for s in canon), default=0)
        else:
            cb = m.get("court_base") or ""
            return max(0, len(_court_case_to_peers.get(cb, set())) - 1)

    # --- 3c. Per-(qid, did) feature matrices ---
    print(f"[3c] per-(qid, did) feature matrix ...", flush=True)
    t0 = time.time()
    DOSSIER = {}
    COL_NAMES = ["statute_int", "lead_statute_int", "concept_int", "term_int",
                 "co_cite", "doctrinal", "chamber_match", "hard_neg", "fusion_rank"]
    for qid in QUERY_IDS:
        targets = ALL_TARGETS.get(qid, {}) or {}
        stat_t = canon_set(targets.get("statute_targets", []))
        conc_t = set(targets.get("concept_targets_en", []))
        term_t = (set(targets.get("term_targets_de", [])) |
                  set(targets.get("term_targets_fr", [])) |
                  set(targets.get("term_targets_it", [])))
        chambers_ok = allowed_chambers(qid, ALL_TARGETS)

        pool = PER_QUERY[qid]["final_topk"]
        N = len(pool)
        feat = np.zeros((N, 9), dtype=np.float32)
        for i, did in enumerate(pool):
            feat[i, 0] = len(_doc_canon_statutes.get(did, set()) & stat_t)
            feat[i, 1] = len(_doc_lead_canon_statutes.get(did, set()) & stat_t)
            feat[i, 2] = len(_doc_to_concepts.get(did, set()) & conc_t)
            feat[i, 3] = len(_doc_to_terms.get(did, set()) & term_t)
            feat[i, 4] = co_cite_count(did)
            feat[i, 5] = _doc_doctrinal.get(did, 0.0)
            ch = _doc_chamber.get(did, "")
            fam = doc_meta.get(did, {}).get("family")
            if not chambers_ok or fam == "law" or not ch:
                feat[i, 6] = 1
            else:
                feat[i, 6] = 1 if ch in chambers_ok else 0
            feat[i, 7] = _doc_hardneg.get(did, 0)
            feat[i, 8] = i + 1
        DOSSIER[qid] = {"did_list": pool, "feat": feat}
        print(f"  {qid}: N={N:>5}  statute_int_sum={int(feat[:,0].sum()):>4}  "
              f"lead_sum={int(feat[:,1].sum()):>4}  "
              f"hardneg%={(feat[:,7]==1).mean()*100:.1f}",
              flush=True)
    print(f"  3c done in {time.time()-t0:.1f}s", flush=True)

    return DOSSIER, COL_NAMES, QUERY_IDS


# =============================================================================
# Main
# =============================================================================
def main():
    t_total = time.time()
    state = warm_boot()
    DOSSIER, COL_NAMES, QUERY_IDS = precompute(state)

    print(f"\n[save] writing to {OUT_PATH} ...", flush=True)
    t0 = time.time()
    # Format the notebook's Phase 3 cache-load path expects:
    #   key '<qid>__dids' -> object array of doc-id strings (length N)
    #   key '<qid>__feat' -> float32 array (N, 9), columns match COL_NAMES
    payload = {}
    for qid in QUERY_IDS:
        payload[f"{qid}__dids"] = np.array(DOSSIER[qid]["did_list"], dtype=object)
        payload[f"{qid}__feat"] = DOSSIER[qid]["feat"]
    np.savez_compressed(OUT_PATH, **payload)
    sz = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"  saved {sz:.1f} MB in {time.time()-t0:.1f}s", flush=True)
    print(f"  schema: {len(QUERY_IDS)} queries x 2 keys each "
          f"(*__dids object array, *__feat float32 (N,9))", flush=True)
    print(f"  COL_NAMES = {COL_NAMES}", flush=True)

    print(f"\n[done] total wall-time: {time.time()-t_total:.1f}s", flush=True)
    print(f"\nLocal path: {OUT_PATH}")
    print(f"Upload to Drive at:")
    print(f"  /content/drive/MyDrive/swiss_law/research/hybrid_rerank_final/cache/dossier_features.npz")


if __name__ == "__main__":
    main()
