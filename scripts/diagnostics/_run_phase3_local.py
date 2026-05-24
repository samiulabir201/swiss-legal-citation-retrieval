"""Run Phase 1+2+3 locally — FAST version (cheap substring checks, no backtracking regex).

Why the slow version exploded: _STATUTE_RE had a 40-way alternation with nested
optional `\s*` groups, and _CITE_CLUSTER had `{3,}` over lazy quantifiers — both
caused catastrophic backtracking on certain Swiss legal paragraphs (5 docs/sec).

The fix: doc statute anchors in the snapshot are ALREADY canonical strings like
"104 OG", "717 OR" — no regex needed. The remaining per-doc signals (doctrinal
density, hard-neg detection, chamber lookup) use small, fixed-list substring
checks. Target throughput: ~5000+ docs/sec single-thread, ~30k docs/sec with
8 workers. 255k corpus -> ~10 seconds wall on multiproc.
"""
from __future__ import annotations
import sys, os, json, gzip, time, re, multiprocessing as mp
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

# ---------- paths ----------
DRIVE = Path(r"E:\swiss_citation_extraction")
SNAPSHOT_DIR = DRIVE / "research" / "anchor_funnel_val001_v7" / "snapshot" / "snapshot"
OUT_DIR = DRIVE / "research" / "hybrid_rerank_final"
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DOSSIER_CACHE = CACHE_DIR / "dossier_features.npz"

# ---------- substring lookups (module-level for spawn) ----------
TEXT_TRUNC = 2500
LEAD_LEN = 200

# Swiss federal statute / case codes — exact substring lookup, no regex
LAW_CODES = (
    "StPO", "StGB", "ZGB", "OR", "BGG", "ZPO", "BV", "IVG", "ATSG", "SchKG",
    "MWSTG", "UWG", "UVG", "AHVG", "BVG", "AVIG", "AsylG", "BGE", "ATF", "StG",
    "VStrR", "EleG", "EnG", "FINIG", "FinfraG", "GwG", "KVG", "VPG", "MEDBG",
    "SR", "CPC", "CC", "CO", "CP", "CPP", "LP", "LCart", "LTF", "LPGA", "LAI",
)

# Doctrinal markers (lowercased substring check — case-insensitive lookup)
DE_RULE_OPENERS = (
    "nach art.", "nach artikel", "gemäss art.", "gemäss artikel",
    "gemaess art.", "gemaess artikel", "im sinne von art.",
    "im sinne der art.", "laut art.", "laut artikel",
    "aufgrund von art.", "aufgrund des art.", "bei art.", "entsprechend art.",
)
DE_DOCTRINE = (
    "rechtsprechung", "ständige rechtsprechung", "staendige rechtsprechung",
    "bundesgerichtliche rechtsprechung", "bundesgerichtlichen rechtsprechung",
    "nach bundesgerichtlicher praxis", "nach herrschender lehre",
    "nach herrschender auffassung", "herrschende lehre",
    "lehre und rechtsprechung",
)
DE_RULE_VERBS = ("bestimmt", "sieht vor", "verlangt", "setzt voraus", "erfordert",
                 "ordnet an", "statuiert")
FR_RULE_OPENERS = (
    "selon l'art.", "selon l'article", "selon art.",
    "conformément à l'art.", "conformément à l'article",
    "conformement a l'art.", "conformement a l'article",
    "aux termes de l'art.", "aux termes de l'article",
    "en vertu de l'art.", "en vertu de l'article",
    "d'après l'art.", "d'apres l'art.", "suivant l'art.",
)
FR_DOCTRINE = (
    "selon la jurisprudence", "d'après la jurisprudence", "d'apres la jurisprudence",
    "jurisprudence constante", "selon la doctrine", "la doctrine et",
    "la doctrine admet", "la doctrine considère", "la doctrine considere",
)
FR_RULE_VERBS = ("dispose", "prévoit", "prevoit", "exige", "requiert",
                 "ordonne", "stipule")

# Hard-neg signature markers (lowercased)
DE_DISPOSITIV_SIGS = ("demnach erkennt", "wird erkannt",
                      "das bundesgericht erkennt", "das bundesgericht hat entschieden",
                      "dispositiv", "urteilsdispositiv",
                      "die beschwerde wird gutgeheissen",
                      "die beschwerde wird abgewiesen",
                      "die beschwerde wird nicht eingetreten")
FR_DISPOSITIV_SIGS = ("par ces motifs", "le tribunal fédéral prononce",
                      "le tribunal federal prononce",
                      "le recours est admis", "le recours est rejeté",
                      "le recours est rejete", "le recours est irrecevable")
DE_FACTS_SIGS = ("sachverhalt", "in sachen ", "a.- ", "b.- ", "c.- ",
                 "gegen die verfügung", "gegen den entscheid")
DE_COSTS_SIGS = ("gerichtskosten", "verfahrenskosten", "parteientschädigung",
                 "parteientschadigung", "kostenfolge")
FR_COSTS_SIGS = ("frais judiciaires", "dépens", "depens", "émolument judiciaire",
                 "emolument judiciaire")
FR_SIG_SIGS = ("le président", "le presidént", "le greffier", "la greffière",
               "la greffiere")

_HARDNEG_ROLES = {"dispositif", "costs", "facts", "procedural_history",
                  "admissibility", "signature"}

# Cheap chamber regex — only on short citation strings, no backtracking risk
_CHAMBER_DOCKET = re.compile(
    r"BGE\s+\d+\s+([IVX]+)|(\d[A-Z]+)[._ ]?\d|\b([1-9][A-Z])\b")


def _count_any(text_lower: str, needles: tuple) -> int:
    return sum(1 for n in needles if n in text_lower)


def doctrinal_density(text: str) -> float:
    if not text:
        return 0.0
    tl = text.lower()
    score = 0.0
    score += 3 * _count_any(tl, DE_RULE_OPENERS)
    score += 3 * _count_any(tl, DE_DOCTRINE)
    score += 1 * _count_any(tl, DE_RULE_VERBS)
    score += 3 * _count_any(tl, FR_RULE_OPENERS)
    score += 3 * _count_any(tl, FR_DOCTRINE)
    score += 1 * _count_any(tl, FR_RULE_VERBS)
    # Cite-cluster proxy: BGE/ATF mentions inside parens
    paren_cites = sum(tl.count(p) for p in ("(bge ", "(atf ", "; bge ", "; atf "))
    if paren_cites >= 3:
        score += 4
    elif paren_cites == 2:
        score += 2
    # Statute density proxy: Art. occurrences per 1000 chars
    art_count = tl.count("art.")
    density = art_count / max(1, len(text) / 1000.0)
    score += min(density, 6.0) * 0.5
    # Hard-neg subtractions
    if any(s in tl for s in DE_DISPOSITIV_SIGS) or any(s in tl for s in FR_DISPOSITIV_SIGS):
        score -= 5
    if any(s in tl for s in DE_FACTS_SIGS):
        score -= 3
    if any(s in tl for s in DE_COSTS_SIGS) or any(s in tl for s in FR_COSTS_SIGS):
        score -= 3
    if any(s in tl for s in FR_SIG_SIGS):
        score -= 3
    return max(-10.0, min(20.0, score))


def is_hard_neg(text_lower: str, role: str) -> bool:
    if role and role.lower() in _HARDNEG_ROLES:
        return True
    if any(s in text_lower for s in DE_DISPOSITIV_SIGS) or \
       any(s in text_lower for s in FR_DISPOSITIV_SIGS):
        return True
    if any(s in text_lower for s in DE_COSTS_SIGS) or \
       any(s in text_lower for s in FR_COSTS_SIGS):
        return True
    if any(s in text_lower for s in FR_SIG_SIGS):
        return True
    return False


def chamber_of(citation: str) -> str:
    if not citation: return ""
    m = _CHAMBER_DOCKET.search(citation)
    if not m: return ""
    for g in m.groups():
        if g: return g.upper()
    return ""


def lead_codes(text_lead: str) -> set:
    """Return canonical 'NUM CODE' pairs found in the first 200 chars.

    Cheap implementation: split on whitespace, look for tokens that are a
    digit-prefixed number followed (within 4 tokens) by a known law code.
    Conservative — will sometimes miss aliased forms (e.g. 'Art. 221 Abs. 1
    lit. b StPO' will only catch '221 StPO'), but that's exactly what we want
    for canonical intersection with the snapshot anchors.
    """
    if not text_lead: return set()
    tokens = text_lead.replace(",", " ").replace("(", " ").replace(")", " ").split()
    out = set()
    for i, tok in enumerate(tokens):
        # Token is a number (possibly with trailing letter)
        if tok and tok[0].isdigit():
            # Strip non-alnum suffix
            num = tok.rstrip(".,;:")
            if not num or not num[0].isdigit():
                continue
            # Look ahead up to 4 tokens for a known code
            for j in range(i+1, min(i+5, len(tokens))):
                code = tokens[j].rstrip(".,;:)")
                if code in LAW_CODES:
                    out.add(f"{num} {code}")
                    break
    return out


# ---------- worker globals ----------
_W_SEARCH_TEXT = None
_W_DOC_META   = None


def init_worker(corpus_path: str):
    global _W_SEARCH_TEXT, _W_DOC_META
    with gzip.open(corpus_path, "rt", encoding="utf-8") as f:
        c = json.load(f)
    _W_SEARCH_TEXT = {d: r["ct"] for d, r in c.items()}
    _W_DOC_META    = {d: (r["cit"], r["fam"], r["cb"], r["pr"], r["ln"])
                       for d, r in c.items()}


def _process_doc(did: str):
    text_full = _W_SEARCH_TEXT.get(did, "") or ""
    text = text_full[:TEXT_TRUNC]
    text_lower = text.lower()
    cit, fam, cb, role, lang = _W_DOC_META.get(did, ("", "?", "", "", "?"))
    doctrinal = doctrinal_density(text)
    hardneg   = 1 if is_hard_neg(text_lower, role) else 0
    lead = text[:LEAD_LEN]
    lead_canon = lead_codes(lead)
    chamber = chamber_of(cit)
    return (did, doctrinal, hardneg, lead_canon, chamber, fam, cb)


# ---------- legal-area mapping (parent only) ----------
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
_QUERY_AREA_TO_CHAMBERS = {
    "criminal":      {"1B", "6B", "7B", "IV"},
    "civil":         {"4A", "5A", "I", "II", "III"},
    "administrative":{"2C", "8C", "9C"},
    "social":        {"8C", "9C", "U"},
    "labor":         {"4A", "8C"},
    "tax":           {"2C"},
    "intellectual":  {"4A"},
}


def query_legal_areas(targets: dict) -> set:
    text = " ".join(str(k) for k in (targets.get("legal_area_keywords") or []))
    areas = {a for a, rx in _LEGAL_AREA_PATTERNS.items() if rx.search(text)}
    if not areas:
        areas.add("any")
    return areas


def allowed_chambers(targets: dict) -> set:
    areas = query_legal_areas(targets)
    if "any" in areas: return set()
    chambers = set()
    for a in areas:
        chambers |= _QUERY_AREA_TO_CHAMBERS.get(a, set())
    return chambers


def canonicalize_query_target(s: str) -> str:
    """For LLM-expanded query targets like 'Art. 221 Abs. 1 lit. b StPO',
    extract the canonical '221 StPO' (matches snapshot anchor format).
    Fast token-walking version (no backtracking regex)."""
    if not s: return s
    tokens = s.replace(",", " ").replace("(", " ").replace(")", " ").split()
    for i, tok in enumerate(tokens):
        if tok and tok[0].isdigit():
            num = tok.rstrip(".,;:")
            for j in range(i+1, min(i+6, len(tokens))):
                code = tokens[j].rstrip(".,;:)")
                if code in LAW_CODES:
                    return f"{num} {code}"
    return s


def canon_query_set(values) -> set:
    out = set()
    for v in values:
        c = canonicalize_query_target(v)
        if c: out.add(c)
    return out


# ---------- main ----------
def main():
    if DOSSIER_CACHE.exists():
        print(f"[skip] cache exists: {DOSSIER_CACHE}")
        return

    n_cpu = mp.cpu_count()
    N_WORKERS = min(8, max(2, n_cpu - 2))
    print(f"[setup] cpu_count={n_cpu}  workers={N_WORKERS}", flush=True)

    print("[1] load per-query + targets + gold ...", flush=True)
    t0 = time.time()
    _pq = json.loads((SNAPSHOT_DIR / "per_query_snapshot.json").read_text(encoding="utf-8"))
    PER_QUERY = {qid: r["final_topk"] for qid, r in _pq.items()}
    QUERY_IDS = sorted(PER_QUERY.keys())
    ALL_GOLD = {qid: set(lst) for qid, lst in
        json.loads((SNAPSHOT_DIR / "gold_doc_sets.json").read_text(encoding="utf-8")).items()}
    ALL_TARGETS = json.loads((SNAPSHOT_DIR / "all_targets.json").read_text(encoding="utf-8"))
    print(f"  done in {time.time()-t0:.1f}s ({len(QUERY_IDS)} queries)", flush=True)

    print("[2] load corpus snapshot (parent — for concepts/terms/statute-anchors) ...", flush=True)
    t0 = time.time()
    with gzip.open(SNAPSHOT_DIR / "corpus_snapshot.json.gz", "rt", encoding="utf-8") as f:
        _c = json.load(f)
    all_dids = list(_c.keys())
    _doc_to_concepts = {d: set(r["cn"]) for d, r in _c.items()}
    _doc_to_terms    = {d: set(r["tm"]) for d, r in _c.items()}
    _doc_statutes    = {d: set(r["sa"]) for d, r in _c.items()}   # already canonical
    _doc_family      = {d: r["fam"] for d, r in _c.items()}
    del _c
    print(f"  done in {time.time()-t0:.1f}s ({len(all_dids):,} docs)", flush=True)

    print(f"[3a] per-doc fast features via Pool(workers={N_WORKERS}) ...", flush=True)
    t0 = time.time()
    _doc_doctrinal  = {}
    _doc_hardneg    = {}
    _doc_lead_canon = {}
    _doc_chamber    = {}
    _doc_court_base = {}

    corpus_path = str(SNAPSHOT_DIR / "corpus_snapshot.json.gz")
    chunksize = 4000
    n_done = 0
    with mp.Pool(processes=N_WORKERS, initializer=init_worker,
                 initargs=(corpus_path,)) as pool:
        for did, doctrinal, hardneg, lead_canon, chamber, fam, cb in pool.imap_unordered(
                _process_doc, all_dids, chunksize=chunksize):
            _doc_doctrinal[did]  = doctrinal
            _doc_hardneg[did]    = hardneg
            _doc_lead_canon[did] = lead_canon
            _doc_chamber[did]    = chamber
            if fam == "court":
                _doc_court_base[did] = cb
            n_done += 1
            if n_done % 50_000 == 0:
                el = time.time() - t0
                rate = n_done / el
                eta = (len(all_dids) - n_done) / max(rate, 1)
                print(f"  ...{n_done:,}/{len(all_dids):,}  ({el:.1f}s, "
                      f"{rate:.0f} docs/s, ETA {eta:.0f}s)", flush=True)
    print(f"  done in {time.time()-t0:.1f}s ({n_done/(time.time()-t0):.0f} docs/s)", flush=True)

    print("[3b] co-citation index ...", flush=True)
    t0 = time.time()
    _statute_cite_count = Counter()
    _court_case_to_peers = defaultdict(set)
    pool_union = set()
    for qid in QUERY_IDS:
        pool_union.update(PER_QUERY[qid])
    for did in pool_union:
        if _doc_family.get(did) == "court":
            for s in _doc_statutes.get(did, set()):
                _statute_cite_count[s] += 1
            cb = _doc_court_base.get(did, "")
            if cb:
                _court_case_to_peers[cb].add(did)
    print(f"  statute-cite={len(_statute_cite_count):,}  "
          f"case-peer={len(_court_case_to_peers):,}  in {time.time()-t0:.1f}s", flush=True)

    def co_cite(did: str) -> int:
        if _doc_family.get(did) == "law":
            sa = _doc_statutes.get(did, set())
            if not sa: return 0
            return max((_statute_cite_count.get(s, 0) for s in sa), default=0)
        else:
            cb = _doc_court_base.get(did, "")
            return max(0, len(_court_case_to_peers.get(cb, set())) - 1)

    print("[3c] per-(qid, did) feature matrix ...", flush=True)
    t0 = time.time()
    DOSSIER = {}
    for qid in QUERY_IDS:
        targets = ALL_TARGETS.get(qid, {}) or {}
        stat_t = canon_query_set(targets.get("statute_targets", []))
        conc_t = set(targets.get("concept_targets_en", []))
        term_t = (set(targets.get("term_targets_de", [])) |
                  set(targets.get("term_targets_fr", [])) |
                  set(targets.get("term_targets_it", [])))
        chambers_ok = allowed_chambers(targets)
        pool = PER_QUERY[qid]
        N = len(pool)
        feat = np.zeros((N, 9), dtype=np.float32)
        for i, did in enumerate(pool):
            feat[i, 0] = len(_doc_statutes.get(did, set()) & stat_t)
            feat[i, 1] = len(_doc_lead_canon.get(did, set()) & stat_t)
            feat[i, 2] = len(_doc_to_concepts.get(did, set()) & conc_t)
            feat[i, 3] = len(_doc_to_terms.get(did, set()) & term_t)
            feat[i, 4] = co_cite(did)
            feat[i, 5] = _doc_doctrinal.get(did, 0.0)
            ch  = _doc_chamber.get(did, "")
            fam = _doc_family.get(did, "?")
            if not chambers_ok or fam == "law" or not ch:
                feat[i, 6] = 1
            else:
                feat[i, 6] = 1 if ch in chambers_ok else 0
            feat[i, 7] = _doc_hardneg.get(did, 0)
            feat[i, 8] = i + 1
        DOSSIER[qid] = {"did_list": pool, "feat": feat}
        print(f"  {qid}: pool={N:>5}  hard_neg%={(feat[:,7]==1).mean()*100:.1f}  "
              f"chamberOK%={(feat[:,6]==1).mean()*100:.1f}  "
              f"stat_t={len(stat_t)}", flush=True)
    print(f"[3c] done in {time.time()-t0:.1f}s", flush=True)

    # Save
    payload = {}
    for qid in QUERY_IDS:
        payload[f"{qid}__dids"] = np.array(DOSSIER[qid]["did_list"], dtype=object)
        payload[f"{qid}__feat"] = DOSSIER[qid]["feat"]
    np.savez_compressed(DOSSIER_CACHE, **payload)
    print(f"\n[OK] saved -> {DOSSIER_CACHE}  ({DOSSIER_CACHE.stat().st_size/1024:.1f} KB)")

    # Quick lift on val_001 + val_003
    print("\n[lift] gold-vs-ambient on val_001 + val_003:")
    COL_NAMES = ["statute_int", "lead_statute_int", "concept_int", "term_int",
                 "co_cite", "doctrinal", "chamber_match", "hard_neg"]
    for qid in ("val_001", "val_003"):
        feat = DOSSIER[qid]["feat"]; pool = DOSSIER[qid]["did_list"]
        g = ALL_GOLD[qid]
        gm = np.array([d in g for d in pool])
        am = ~gm
        print(f"\n  {qid}  N_gold_in_pool={int(gm.sum())}  N_ambient={int(am.sum())}")
        print(f"  {'feature':<18}{'gold':>10}{'amb':>10}{'lift':>8}")
        for j, n in enumerate(COL_NAMES):
            g_m = feat[gm, j].mean()
            a_m = feat[am, j].mean()
            if abs(a_m) > 1e-3:
                print(f"  {n:<18}{g_m:>10.3f}{a_m:>10.3f}{g_m/a_m:>7.2f}x")
            else:
                print(f"  {n:<18}{g_m:>10.3f}{a_m:>10.3f}     -")


if __name__ == "__main__":
    mp.freeze_support()
    main()
