from __future__ import annotations

import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "pool50k_to_500_high_recall_v1.ipynb"


def md_cell(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


MD_TITLE = r"""# Pool 50k -> 500 High-Recall Candidate Pool v1

Goal: warm-boot from the canonical v7.5 top-50k snapshot, then build a much smaller candidate pool for the final LLM judge.

This notebook is deliberately diagnostic-first:

1. It measures the true recall ceiling of every narrowing stage.
2. It produces a fixed `500/query` pool.
3. If the fixed 500 pool cannot honestly keep target recall, it also produces an adaptive rescue pool with the smallest K needed by the current ranking.

Important: val gold labels are used only for diagnostics and recall reporting unless `LEAKY_SUPERVISED_VAL_CEILING = True`. Do not use that ceiling mode for test or leaderboard submissions.

How to read the result: if the fixed `500/query` pool misses the target, do not submit it as-is. Use the reported ceiling/adaptive tables to decide whether to enable `RUN_LLM_TRIAGE`, increase `FINAL_K`, or return to the upstream retrieval pool.
"""


MD_PHASE0 = r"""## Phase 0 - Setup

Set paths and knobs here. The notebook searches local workspace paths first, then common Google Drive paths.
"""


CELL_PHASE0 = r'''import json, gzip, math, os, re, sys, time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except Exception:
    display = print

WORKSPACE_CANDIDATES = [
    Path.cwd(),
    Path(r"E:/swiss_citation_extraction"),
    Path("/content/drive/MyDrive/swiss_law"),
    Path("/content/drive/MyDrive/swiss_citation_extraction"),
]

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return paths[0]

ROOT = first_existing(WORKSPACE_CANDIDATES)

SNAPSHOT_CANDIDATES = [
    Path("/content/drive/MyDrive/swiss_law/research/anchor_funnel_val001_v7/snapshot"),
    ROOT / "drive_sync/swiss_law/v7_pool_recall_089_and_per_query/snapshot",
    ROOT / "research/local_v75_snapshot_mirror/snapshot/snapshot",
    ROOT / "v7_pool_recall_089_and_per_query/snapshot",
    ROOT / "research/anchor_funnel_val001_v7/snapshot",
    Path("/content/drive/MyDrive/swiss_law/v7_pool_recall_089_and_per_query/snapshot"),
]
SNAPSHOT_DIR = first_existing(SNAPSHOT_CANDIDATES)

CACHE_CANDIDATES = [
    Path("/content/drive/MyDrive/swiss_law/research/hybrid_rerank_final/cache"),
    ROOT / "drive_sync/swiss_law/hybrid_rerank_shootout_cache/cache",
    ROOT / "research/local_hybrid_rerank_cache_mirror/cache",
    Path("/content/drive/MyDrive/swiss_law/hybrid_rerank_shootout_cache/cache"),
]
CACHE_DIR = first_existing(CACHE_CANDIDATES)

OUT_DIR = ROOT / "artifacts" / "pool50k_to_500_high_recall_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Main knobs.
STAGE_A_K = 10000          # high-recall reservoir before the 500 cut
FINAL_K = 500              # fixed pool handed to the final LLM judge
TARGET_MACRO_RECALL = 0.78 # target for diagnostics; adaptive rescue uses this if labels exist
K_REPORT = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000]

# Optional modes.
RUN_LLM_TRIAGE = False
LEAKY_SUPERVISED_VAL_CEILING = False  # True = diagnostic only, uses val labels in ranking

print("ROOT:", ROOT)
print("SNAPSHOT_DIR:", SNAPSHOT_DIR, "exists=", SNAPSHOT_DIR.exists())
print("CACHE_DIR:", CACHE_DIR, "exists=", CACHE_DIR.exists())
print("OUT_DIR:", OUT_DIR)
assert SNAPSHOT_DIR.exists(), f"Snapshot not found: {SNAPSHOT_DIR}"
'''


MD_PHASE1 = r"""## Phase 1 - Warm-Boot v7.5 Snapshot

Loads:

- `per_query_snapshot.json`: v7.5 final top-50k order per query
- `corpus_snapshot.json.gz`: compact document text and metadata for the union pool
- `all_targets.json` and `hyde_aspects.json`: query expansion/aspect data
- `gold_doc_sets.json`: val labels for diagnostics
"""


CELL_PHASE1 = r'''t0 = time.time()

with open(SNAPSHOT_DIR / "per_query_snapshot.json", encoding="utf-8") as f:
    PER_QUERY = json.load(f)

with open(SNAPSHOT_DIR / "gold_doc_sets.json", encoding="utf-8") as f:
    GOLD_DOC_SETS = {qid: set(v) for qid, v in json.load(f).items()}

with open(SNAPSHOT_DIR / "all_targets.json", encoding="utf-8") as f:
    ALL_TARGETS = json.load(f)

hyde_path = SNAPSHOT_DIR / "hyde_aspects.json"
ALL_HYDE_ASPECTS = json.loads(hyde_path.read_text(encoding="utf-8")) if hyde_path.exists() else {}

paths_path = SNAPSHOT_DIR / "paths.json"
PATHS_META = json.loads(paths_path.read_text(encoding="utf-8")) if paths_path.exists() else {}

with gzip.open(SNAPSHOT_DIR / "corpus_snapshot.json.gz", "rt", encoding="utf-8") as f:
    CORPUS = json.load(f)

doc_meta = {}
search_text = {}
doc_statute_anchors = {}
doc_concepts = {}
doc_terms = {}
for did, r in CORPUS.items():
    search_text[did] = r.get("ct", "")
    doc_meta[did] = {
        "citation": r.get("cit", ""),
        "family": r.get("fam", did.split(":", 1)[0]),
        "court_base": r.get("cb", ""),
        "paragraph_role": r.get("pr", ""),
        "language": r.get("ln", "?"),
    }
    doc_statute_anchors[did] = set(r.get("sa", []) or [])
    doc_concepts[did] = set(str(x).lower() for x in (r.get("cn", []) or []))
    doc_terms[did] = set(str(x).lower() for x in (r.get("tm", []) or []))

print(f"Loaded {len(PER_QUERY)} queries")
print(f"Corpus snapshot docs: {len(CORPUS):,}")
print(f"Gold doc ids: {sum(len(v) for v in GOLD_DOC_SETS.values()):,}")
print(f"Warm-boot done in {time.time()-t0:.1f}s")
'''


MD_PHASE2 = r"""## Phase 2 - Build Candidate Table

This table is the working substrate. It keeps the v7.5 fusion order and merges optional cached reranker/dossier features if available.
"""


CELL_PHASE2 = r'''def load_npz(path):
    return np.load(path, allow_pickle=True) if path.exists() else None

dossier_npz = load_npz(CACHE_DIR / "dossier_features.npz")
qwen_npz = load_npz(CACHE_DIR / "scores_qwen3.npz")
bge_npz = load_npz(CACHE_DIR / "scores_bge.npz")
jina_npz = load_npz(CACHE_DIR / "scores_jina.npz")

HAS_CACHE = dossier_npz is not None
print("HAS_CACHE:", HAS_CACHE)

rows = []
for qid, st in PER_QUERY.items():
    gold = GOLD_DOC_SETS.get(qid, set())

    if HAS_CACHE and f"{qid}__dids" in dossier_npz.files:
        dids = [str(x) for x in dossier_npz[f"{qid}__dids"]]
        feat = dossier_npz[f"{qid}__feat"]
        qwen = qwen_npz[qid] if (qwen_npz is not None and qid in qwen_npz.files) else np.zeros(len(dids), dtype="float32")
        bge = bge_npz[qid] if (bge_npz is not None and qid in bge_npz.files) else np.zeros(len(dids), dtype="float32")
        jina = jina_npz[qid] if (jina_npz is not None and qid in jina_npz.files) else np.zeros(len(dids), dtype="float32")
    else:
        dids = [str(x) for x in st["final_topk"]]
        feat = np.zeros((len(dids), 9), dtype="float32")
        qwen = np.zeros(len(dids), dtype="float32")
        bge = np.zeros(len(dids), dtype="float32")
        jina = np.zeros(len(dids), dtype="float32")

    for i, did in enumerate(dids):
        meta = doc_meta.get(did, {})
        r = {
            "qid": qid,
            "did": did,
            "fusion_rank": i + 1,
            "family": meta.get("family", did.split(":", 1)[0]),
            "citation": meta.get("citation", ""),
            "court_base": meta.get("court_base", ""),
            "paragraph_role": meta.get("paragraph_role", ""),
            "language": meta.get("language", "?"),
            "text": search_text.get(did, ""),
            "is_gold": did in gold,
            "qwen_score": float(qwen[i]) if i < len(qwen) else 0.0,
            "bge_score": float(bge[i]) if i < len(bge) else 0.0,
            "jina_score": float(jina[i]) if i < len(jina) else 0.0,
        }
        for j in range(feat.shape[1] if hasattr(feat, "shape") and len(feat.shape) == 2 else 0):
            r[f"dossier_f{j}"] = float(feat[i, j])
        rows.append(r)

pool_df = pd.DataFrame(rows)
for j in range(9):
    c = f"dossier_f{j}"
    if c not in pool_df.columns:
        pool_df[c] = 0.0

print(pool_df.shape)
display(pool_df.groupby("qid").agg(
    n=("did", "count"),
    gold=("is_gold", "sum"),
    law=("family", lambda s: int((s == "law").sum())),
    court=("family", lambda s: int((s == "court").sum())),
))
'''


MD_PHASE3 = r"""## Phase 3 - Baseline Recall Curves

This cell answers the painful but necessary question: how much recall is available before the 500 cut?
"""


CELL_PHASE3 = r'''def recall_table(df, rank_col="fusion_rank", ks=K_REPORT):
    out = []
    total_micro_got = {k: 0 for k in ks}
    total_micro_gold = 0
    for qid, g in df.groupby("qid"):
        gold_total = int(g["is_gold"].sum())
        total_micro_gold += gold_total
        gg = g.sort_values(rank_col, ascending=True)
        for k in ks:
            got = int(gg.head(k)["is_gold"].sum())
            total_micro_got[k] += got
            out.append({"qid": qid, "K": k, "gold_total": gold_total, "got": got, "recall": got / max(1, gold_total)})
    tab = pd.DataFrame(out)
    macro = tab.groupby("K")["recall"].mean().rename("macro_recall")
    micro = pd.Series({k: total_micro_got[k] / max(1, total_micro_gold) for k in ks}, name="micro_recall")
    got = pd.Series({k: f"{total_micro_got[k]}/{total_micro_gold}" for k in ks}, name="micro_got")
    return pd.concat([macro, micro, got], axis=1).reset_index()

baseline = recall_table(pool_df, "fusion_rank")
display(baseline)

stage_a_df = pool_df.sort_values(["qid", "fusion_rank"]).groupby("qid").head(STAGE_A_K).copy()
print("Stage A fixed reservoir:", STAGE_A_K, "per query")

ceiling_rows = []
for qid, gset in GOLD_DOC_SETS.items():
    full_ids = set(pool_df.loc[pool_df["qid"] == qid, "did"])
    stage_ids = set(stage_a_df.loc[stage_a_df["qid"] == qid, "did"])
    ceiling_rows.append({
        "qid": qid,
        "gold_total": len(gset),
        "full_pool_gold": len(full_ids & gset),
        "full_pool_recall": len(full_ids & gset) / max(1, len(gset)),
        "stage_a_gold": len(stage_ids & gset),
        "stage_a_recall": len(stage_ids & gset) / max(1, len(gset)),
    })
ceiling_df = pd.DataFrame(ceiling_rows)
print("Recall ceiling against all val gold:")
print("  full 50k-ish pool macro:", round(float(ceiling_df["full_pool_recall"].mean()), 4))
print("  Stage A reservoir macro:", round(float(ceiling_df["stage_a_recall"].mean()), 4))
display(ceiling_df)
'''


MD_PHASE4 = r"""## Phase 4 - Feature Engineering for the 500 Cut

This produces cheap legal features for both laws and courts. These are not final-gold decisions; they are evidence signals.
"""


CELL_PHASE4 = r'''CODE_ALIAS = {
    "CPP": "StPO", "CP": "StGB", "CC": "ZGB", "CO": "OR", "LTF": "BGG",
    "LPGA": "ATSG", "LAI": "IVG", "Cst": "BV", "Cst.": "BV", "CEDH": "EMRK",
}

ART_CODE_RE = re.compile(
    r"Art\.?\s*(\d+[a-z]*)"
    r"(?:\s*Abs\.?\s*(\d+[a-z]*))?"
    r"(?:\s*(?:lit\.|let\.)\s*([a-z]))?"
    r"\s*([A-Z][A-Za-z.]{1,10}|StPO|StGB|ZGB|OR|BGG|BV|ATSG|IVG|EMRK)",
    re.I,
)

def canon_statute(raw):
    if not raw:
        return None
    m = ART_CODE_RE.search(str(raw).replace("\u00a0", " "))
    if not m:
        return None
    art, _abs, _lit, code = m.groups()
    code = CODE_ALIAS.get(code.rstrip("."), code.rstrip("."))
    return f"{art} {code}"

def target_statutes_for_qid(qid):
    targets = ALL_TARGETS.get(qid, {}) or {}
    out = set()
    for s in targets.get("statute_targets", []) or []:
        c = canon_statute(s)
        if c:
            out.add(c)
    return out

def target_concepts_for_qid(qid):
    targets = ALL_TARGETS.get(qid, {}) or {}
    vals = []
    vals += targets.get("concept_targets_en", []) or []
    vals += targets.get("legal_area_keywords", []) or []
    return set(str(x).lower() for x in vals)

DOCTRINE_RE = re.compile(r"Rechtsprechung|jurisprudence|giurisprudenza|Praxis|doctrine|ständige|constante", re.I)
HARD_NEG_RE = re.compile(
    r"Sachverhalt|Verfahrensgeschichte|wird\s+abgewiesen|Gerichtskosten|"
    r"en\s+fait|le\s+recours\s+est|frais\s+judiciaires|le\s+greffier|"
    r"in\s+fatto|spese\s+giudiziarie",
    re.I,
)
INTERNAL_CITE_RE = re.compile(r"\b(?:BGE|ATF)\s+\d+\s+[IVX]+\s+\d+\s+(?:E\.|consid\.)\s*\d", re.I)

def citation_chamber(cit):
    cit = str(cit or "")
    m = re.match(r"^BGE\s+\d+\s+([IVX]+)\s+\d+", cit)
    if m:
        return "BGE_" + m.group(1)
    m = re.match(r"^(\d[A-Z])_", cit)
    if m:
        return m.group(1)
    return ""

def family_quota(qdf, final_k=FINAL_K):
    # High-recall default: do not starve courts. Law/court split in gold varies by query.
    # Use target mix from the high-recall reservoir, but cap extremes.
    law_frac = float((qdf["family"] == "law").mean())
    law_k = int(round(final_k * min(0.55, max(0.25, law_frac + 0.15))))
    return {"law": law_k, "court": final_k - law_k}

def add_features(df):
    df = df.copy()
    df["fusion_inv"] = 1.0 / (df["fusion_rank"].astype(float) + 60.0)
    for col in ["qwen_score", "bge_score", "jina_score"]:
        df[col + "_rank"] = df.groupby("qid")[col].rank(method="first", ascending=False)
        df[col + "_inv"] = 1.0 / (df[col + "_rank"] + 60.0)

    df["text_len"] = df["text"].fillna("").str.len()
    df["lead"] = df["text"].fillna("").str[:600]
    df["has_doctrine"] = df["text"].fillna("").str[:1200].str.contains(DOCTRINE_RE)
    df["hard_neg"] = df["lead"].str.contains(HARD_NEG_RE)
    df["internal_cite_count"] = df["text"].fillna("").str[:1600].apply(lambda x: len(INTERNAL_CITE_RE.findall(x)))
    df["chamber"] = df["citation"].apply(citation_chamber)
    df["is_bge"] = df["citation"].fillna("").str.startswith("BGE")

    stat_hits = []
    lead_stat_hits = []
    concept_hits = []
    aspect_hits = []
    for qid, did, lead, text in zip(df["qid"], df["did"], df["lead"], df["text"]):
        tstats = target_statutes_for_qid(qid)
        anchors = doc_statute_anchors.get(did, set())
        stat_hits.append(len(anchors & tstats))

        lead_stats = set()
        for m in ART_CODE_RE.finditer(str(lead)):
            c = canon_statute(m.group(0))
            if c:
                lead_stats.add(c)
        lead_stat_hits.append(len(lead_stats & tstats))

        tconcepts = target_concepts_for_qid(qid)
        cset = doc_concepts.get(did, set())
        concept_hits.append(len(cset & tconcepts))

        aspects = ALL_HYDE_ASPECTS.get(qid, []) or []
        blob = (str(text)[:1200] + " " + " ".join(cset) + " " + " ".join(anchors)).lower()
        best = 0
        for a in aspects:
            toks = set(re.findall(r"[A-Za-zÄÖÜäöüß]{5,}", str(a).lower()))
            best = max(best, len(toks & set(re.findall(r"[A-Za-zÄÖÜäöüß]{5,}", blob))))
        aspect_hits.append(best)

    df["stat_hit_count"] = stat_hits
    df["lead_stat_hit_count"] = lead_stat_hits
    df["concept_hit_count"] = concept_hits
    df["aspect_hit_count"] = aspect_hits
    return df

stage_a_feat = add_features(stage_a_df)
display(stage_a_feat.head(3))
'''


MD_PHASE5 = r"""## Phase 5 - Non-LLM High-Recall Ranker

This ranker is intentionally conservative. It combines fusion order, cached reranker scores, dossier features, and legal evidence signals.

It is not expected to magically solve precision. Its job is to make the final LLM judge see a much smaller but still recall-rich pool.
"""


CELL_PHASE5 = r'''def minmax_by_qid(df, col):
    s = df[col].astype(float)
    lo = s.groupby(df["qid"]).transform("min")
    hi = s.groupby(df["qid"]).transform("max")
    return (s - lo) / (hi - lo + 1e-9)

def compute_proxy_score(df):
    df = df.copy()
    # Normalize selected columns per query.
    for col in [
        "fusion_inv", "qwen_score", "bge_score", "jina_score",
        "stat_hit_count", "lead_stat_hit_count", "concept_hit_count",
        "aspect_hit_count", "internal_cite_count",
    ] + [f"dossier_f{i}" for i in range(9)]:
        df[col + "_n"] = minmax_by_qid(df, col)

    role = df["paragraph_role"].fillna("")
    substantive_role = role.isin(["legal_standard", "reasoning", "application", "holding"]).astype(float)
    broad_role = role.isin(["legal_standard", "reasoning", "application", "holding", "facts", "procedural_history"]).astype(float)
    law = (df["family"] == "law").astype(float)
    court = (df["family"] == "court").astype(float)

    score = (
        3.0 * df["fusion_inv_n"]
        + 1.0 * df["qwen_score_n"]
        + 0.6 * df["bge_score_n"]
        + 0.4 * df["jina_score_n"]
        + 2.6 * df["lead_stat_hit_count_n"]
        + 1.4 * df["stat_hit_count_n"]
        + 1.1 * df["concept_hit_count_n"]
        + 0.7 * df["aspect_hit_count_n"]
        + 0.8 * df["dossier_f0_n"]
        + 0.8 * df["dossier_f1_n"]
        + 0.6 * df["dossier_f2_n"]
        + 0.4 * df["dossier_f3_n"]
        + 0.5 * df["dossier_f4_n"]
        + 0.4 * df["dossier_f5_n"]
        + 0.6 * substantive_role
        + 0.2 * broad_role
        + 0.3 * df["has_doctrine"].astype(float)
        + 0.2 * (df["internal_cite_count"] >= 2).astype(float)
        + 0.2 * df["is_bge"].astype(float)
        - 1.5 * df["hard_neg"].astype(float)
    )
    # Laws often have weak court-style signals but exact article/fusion evidence matters.
    score += law * (0.8 * df["stat_hit_count_n"] + 0.4 * df["dossier_f4_n"])
    score += court * (0.2 * df["concept_hit_count_n"] + 0.2 * substantive_role)
    df["proxy_score"] = score
    return df

scored_df = compute_proxy_score(stage_a_feat)

def select_diversified_500(df, final_k=FINAL_K):
    """Strong fusion backbone plus rescue channels that catch different gold shapes."""
    budget_plan = [
        ("fusion", "fusion_rank", True, int(final_k * 0.80), None),
        ("proxy", "proxy_score", False, max(30, int(final_k * 0.10)), None),
        ("qwen", "qwen_score", False, max(30, int(final_k * 0.10)), None),
        ("lead_stat", "lead_stat_hit_count", False, max(30, int(final_k * 0.10)), None),
        ("dossier_f1", "dossier_f1", False, max(20, int(final_k * 0.06)), None),
        ("aspect", "aspect_hit_count", False, max(20, int(final_k * 0.06)), None),
    ]
    picks = []
    for qid, qdf in df.groupby("qid"):
        q_picks = []
        used = set()

        def add_from(part, source, limit):
            added = 0
            for did in part["did"].tolist():
                if did in used:
                    continue
                q_picks.append((qid, did, source))
                used.add(did)
                added += 1
                if len(q_picks) >= final_k or added >= limit:
                    break

        for source, col, ascending, limit, filt in budget_plan:
            part = qdf if filt is None else qdf.query(filt)
            add_from(part.sort_values(col, ascending=ascending), source, limit)
            if len(q_picks) >= final_k:
                break

        if len(q_picks) < final_k:
            fill = qdf[~qdf["did"].isin(used)].sort_values("fusion_rank", ascending=True)
            add_from(fill, "fusion_fill", final_k - len(q_picks))

        picks.extend(q_picks[:final_k])
    return pd.DataFrame(picks, columns=["qid", "did", "pick_source"])

pool500_ids = select_diversified_500(scored_df, FINAL_K)
pool500 = pool500_ids.merge(scored_df, on=["qid", "did"], how="left")

def eval_pool(selected, label="pool"):
    rows = []
    for qid, gset in GOLD_DOC_SETS.items():
        pred = set(selected[selected["qid"] == qid]["did"])
        got = len(pred & gset)
        rows.append({
            "qid": qid,
            "n": len(pred),
            "gold_total": len(gset),
            "got": got,
            "recall": got / max(1, len(gset)),
        })
    tab = pd.DataFrame(rows)
    print(label)
    print("macro recall:", round(float(tab["recall"].mean()), 4))
    print("micro recall:", f"{int(tab['got'].sum())}/{int(tab['gold_total'].sum())}",
          round(float(tab["got"].sum() / tab["gold_total"].sum()), 4))
    display(tab)
    return tab

pool500_eval = eval_pool(pool500, "Fixed 500 diversified pool")
'''


MD_PHASE6 = r"""## Phase 6 - Optional LLM Triage 10k -> 500

This is the intended production path if the proxy 500 pool is not recall-rich enough.

Set `RUN_LLM_TRIAGE = True` in Phase 0, then run this cell on a GPU notebook with vLLM. It scores the `STAGE_A_K` reservoir in small batches. The final 500 is selected by LLM score plus the proxy score.
"""


CELL_PHASE6 = r'''if RUN_LLM_TRIAGE:
    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        raise RuntimeError("Install vLLM or set RUN_LLM_TRIAGE=False") from e

    LLM_MODEL = os.environ.get("TRIAGE_LLM_MODEL", "Qwen/Qwen3-14B-AWQ")
    BATCH_CANDIDATES = 8
    MAX_TEXT_CHARS = 900
    CHECKPOINT = OUT_DIR / "llm_triage_scores.parquet"

    def make_prompt(qid, batch):
        query_text = ""
        val_csv_candidates = [
            ROOT / "data/val.csv",
            Path("/content/drive/MyDrive/swiss_law/data/val.csv"),
        ]
        val_csv = first_existing(val_csv_candidates)
        if val_csv.exists():
            vdf = pd.read_csv(val_csv)
            hit = vdf[vdf["query_id"].astype(str) == qid]
            if len(hit):
                query_text = str(hit.iloc[0]["query"])
        aspects = ALL_HYDE_ASPECTS.get(qid, []) or []
        aspect_text = "\n".join(f"A{i}: {str(a)[:500]}" for i, a in enumerate(aspects[:8]))
        cand_lines = []
        for local_id, row in enumerate(batch.itertuples(index=False)):
            txt = str(row.text).replace("\n", " ")[:MAX_TEXT_CHARS]
            cand_lines.append(
                f"[{local_id}] citation={row.citation} family={row.family} "
                f"role={row.paragraph_role} fusion_rank={row.fusion_rank} "
                f"proxy={row.proxy_score:.4f}\nTEXT: {txt}"
            )
        candidates = "\n\n".join(cand_lines)
        return f"""You are filtering Swiss legal citation candidates before a final judge.

Question:
{query_text}

Legal aspects:
{aspect_text}

Task:
For each candidate, decide if it should remain in a high-recall pool for the final citation judge.
Keep candidates that state a legal rule, proof standard, procedural rule, statutory entitlement, or directly support one aspect.
Drop candidates that are merely topically related, wrong issue, boilerplate, costs only, or same-topic noise.

Return only JSON:
{{"scores":[{{"id":0,"score":0-3,"aspect":"A0 or none","reason":"short"}}]}}

Candidates:
{candidates}
"""

    already = pd.read_parquet(CHECKPOINT) if CHECKPOINT.exists() else pd.DataFrame(columns=["qid", "did", "llm_score", "llm_aspect", "llm_reason"])
    done = set(zip(already["qid"], already["did"]))

    llm = LLM(model=LLM_MODEL, max_model_len=8192, gpu_memory_utilization=0.85, trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, max_tokens=512)

    out_rows = already.to_dict("records")
    for qid, qdf in scored_df.groupby("qid"):
        qdf = qdf.sort_values("proxy_score", ascending=False).reset_index(drop=True)
        for start in range(0, len(qdf), BATCH_CANDIDATES):
            batch = qdf.iloc[start:start+BATCH_CANDIDATES]
            if all((qid, did) in done for did in batch["did"]):
                continue
            prompt = make_prompt(qid, batch)
            raw = llm.generate([prompt], sp)[0].outputs[0].text
            m = re.search(r"\{.*\}", raw, re.S)
            scores = []
            if m:
                try:
                    scores = json.loads(m.group(0)).get("scores", [])
                except Exception:
                    scores = []
            score_by_id = {int(x.get("id", -1)): x for x in scores if str(x.get("id", "")).lstrip("-").isdigit()}
            for local_id, row in enumerate(batch.itertuples(index=False)):
                s = score_by_id.get(local_id, {})
                out_rows.append({
                    "qid": qid,
                    "did": row.did,
                    "llm_score": float(s.get("score", 0) or 0),
                    "llm_aspect": str(s.get("aspect", "none")),
                    "llm_reason": str(s.get("reason", ""))[:300],
                })
            if len(out_rows) % 1000 < BATCH_CANDIDATES:
                pd.DataFrame(out_rows).drop_duplicates(["qid", "did"], keep="last").to_parquet(CHECKPOINT, index=False)
                print("checkpoint", len(out_rows), CHECKPOINT)

    triage = pd.DataFrame(out_rows).drop_duplicates(["qid", "did"], keep="last")
    triage.to_parquet(CHECKPOINT, index=False)
    scored_llm = scored_df.merge(triage, on=["qid", "did"], how="left")
    scored_llm["llm_score"] = scored_llm["llm_score"].fillna(0.0)
    scored_llm["final_triage_score"] = scored_llm["proxy_score"] + 4.0 * scored_llm["llm_score"]
    llm500_ids = (
        scored_llm.sort_values(["qid", "final_triage_score"], ascending=[True, False])
        .groupby("qid").head(FINAL_K)[["qid", "did"]]
    )
    llm500 = llm500_ids.merge(scored_llm, on=["qid", "did"], how="left")
    llm500_eval = eval_pool(llm500, "Fixed 500 LLM-triage pool")
else:
    print("RUN_LLM_TRIAGE=False, skipping LLM stage.")
    llm500 = None
    llm500_eval = None
'''


MD_PHASE7 = r"""## Phase 7 - Optional Leaky Supervised Val Ceiling

This mode proves whether the available features can separate known val gold if labels are allowed.

Do not use this for test. It uses `is_gold` as a training target on the same validation set.
"""


CELL_PHASE7 = r'''if LEAKY_SUPERVISED_VAL_CEILING:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import ExtraTreesClassifier

    train = scored_df.copy()
    cat_cols = ["qid", "family", "paragraph_role", "language", "chamber"]
    num_cols = [
        "fusion_rank", "fusion_inv", "qwen_score", "bge_score", "jina_score",
        "stat_hit_count", "lead_stat_hit_count", "concept_hit_count",
        "aspect_hit_count", "internal_cite_count", "proxy_score",
    ] + [f"dossier_f{i}" for i in range(9)]
    for c in cat_cols:
        train[c] = train[c].fillna("").astype(str)
    for c in num_cols:
        train[c] = train[c].fillna(0).astype(float)

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=2), cat_cols),
        ("num", StandardScaler(with_mean=False), num_cols),
    ])
    model = ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
    )
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(train[cat_cols + num_cols], train["is_gold"].astype(int))
    train["leaky_gold_probability"] = pipe.predict_proba(train[cat_cols + num_cols])[:, 1]
    leaky500_ids = (
        train.sort_values(["qid", "leaky_gold_probability"], ascending=[True, False])
        .groupby("qid").head(FINAL_K)[["qid", "did"]]
    )
    leaky500 = leaky500_ids.merge(train, on=["qid", "did"], how="left")
    leaky_eval = eval_pool(leaky500, "LEAKY supervised val ceiling at 500")
else:
    print("LEAKY_SUPERVISED_VAL_CEILING=False, skipping.")
'''


MD_PHASE8 = r"""## Phase 8 - Adaptive Rescue and Save

If the fixed 500 pool does not meet the target, this builds a diagnostic rescue pool. It tries several non-label rankers and records either the smallest K that reaches the per-query target or the best available ceiling when the target is impossible inside `STAGE_A_K`.
"""


CELL_PHASE8 = r'''rank_source = scored_df.copy()
rank_source["final_rank_score"] = rank_source["proxy_score"]
if RUN_LLM_TRIAGE and llm500 is not None:
    rank_source = scored_llm.copy()
    rank_source["final_rank_score"] = rank_source["proxy_score"] + 4.0 * rank_source["llm_score"].fillna(0.0)

fixed_pool = pool500 if not (RUN_LLM_TRIAGE and llm500 is not None) else llm500
fixed_eval = eval_pool(fixed_pool, "FINAL fixed 500 pool used for save")
fixed_macro = float(fixed_eval["recall"].mean())

adaptive_parts = []
adaptive_summary = []
ranker_specs = [
    ("final_rank_score", "final_rank_score", False),
    ("fusion_rank", "fusion_rank", True),
    ("proxy_score", "proxy_score", False),
    ("qwen_score", "qwen_score", False),
    ("jina_score", "jina_score", False),
    ("dossier_f1", "dossier_f1", False),
    ("lead_stat_hit_count", "lead_stat_hit_count", False),
    ("aspect_hit_count", "aspect_hit_count", False),
]
if RUN_LLM_TRIAGE and "llm_score" in rank_source.columns:
    ranker_specs = [("llm_score", "llm_score", False)] + ranker_specs
k_grid = sorted(set([FINAL_K] + [k for k in K_REPORT if k >= FINAL_K] + [STAGE_A_K]))

for qid, qdf in rank_source.groupby("qid"):
    gset = GOLD_DOC_SETS.get(qid, set())
    best = None
    for ranker_name, col, ascending in ranker_specs:
        if col not in qdf.columns:
            continue
        ordered = qdf.sort_values(col, ascending=ascending).reset_index(drop=True)
        best_for_ranker = None
        for k in k_grid:
            if k > len(ordered):
                continue
            r = len(set(ordered.head(k)["did"]) & gset) / max(1, len(gset))
            row = {
                "ranker": ranker_name,
                "ordered": ordered,
                "adaptive_k": k,
                "recall": r,
                "target_met": r >= TARGET_MACRO_RECALL,
            }
            if best_for_ranker is None or r > best_for_ranker["recall"] or (
                r == best_for_ranker["recall"] and k < best_for_ranker["adaptive_k"]
            ):
                best_for_ranker = row
            if r >= TARGET_MACRO_RECALL:
                best_for_ranker = row
                break
        if best is None:
            best = best_for_ranker
        elif best_for_ranker["target_met"] and not best["target_met"]:
            best = best_for_ranker
        elif best_for_ranker["target_met"] == best["target_met"]:
            if best_for_ranker["adaptive_k"] < best["adaptive_k"] or (
                best_for_ranker["adaptive_k"] == best["adaptive_k"] and best_for_ranker["recall"] > best["recall"]
            ):
                best = best_for_ranker

    chosen = best["ordered"].head(best["adaptive_k"]).copy()
    chosen["adaptive_ranker"] = best["ranker"]
    adaptive_parts.append(chosen)
    adaptive_summary.append({
        "qid": qid,
        "adaptive_k": best["adaptive_k"],
        "ranker": best["ranker"],
        "recall": best["recall"],
        "target_met": best["target_met"],
        "gold_total": len(gset),
    })

adaptive_pool = pd.concat(adaptive_parts, ignore_index=True)
adaptive_summary_df = pd.DataFrame(adaptive_summary)

print("Fixed 500 macro recall:", round(fixed_macro, 4))
print("Target macro recall:", TARGET_MACRO_RECALL)
display(adaptive_summary_df)
adaptive_eval = eval_pool(adaptive_pool, "Adaptive rescue pool")

fixed_path = OUT_DIR / "candidate_pool_fixed500.parquet"
adaptive_path = OUT_DIR / "candidate_pool_adaptive_rescue.parquet"
fixed_json = OUT_DIR / "candidate_pool_fixed500.json"
diag_json = OUT_DIR / "diagnostics.json"

fixed_pool.to_parquet(fixed_path, index=False)
adaptive_pool.to_parquet(adaptive_path, index=False)

pred_json = {
    qid: fixed_pool[fixed_pool["qid"] == qid]["citation"].fillna("").tolist()
    for qid in sorted(fixed_pool["qid"].unique())
}
fixed_json.write_text(json.dumps(pred_json, ensure_ascii=False, indent=2), encoding="utf-8")

diag = {
    "stage_a_k": STAGE_A_K,
    "final_k": FINAL_K,
    "target_macro_recall": TARGET_MACRO_RECALL,
    "fixed500_macro_recall": fixed_macro,
    "fixed500": fixed_eval.to_dict("records"),
    "adaptive_summary": adaptive_summary_df.to_dict("records"),
    "adaptive_eval": adaptive_eval.to_dict("records"),
}
diag_json.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")

print("Wrote:")
print(" ", fixed_path)
print(" ", adaptive_path)
print(" ", fixed_json)
print(" ", diag_json)
'''


CELLS = [
    md_cell(MD_TITLE),
    md_cell(MD_PHASE0),
    code_cell(CELL_PHASE0),
    md_cell(MD_PHASE1),
    code_cell(CELL_PHASE1),
    md_cell(MD_PHASE2),
    code_cell(CELL_PHASE2),
    md_cell(MD_PHASE3),
    code_cell(CELL_PHASE3),
    md_cell(MD_PHASE4),
    code_cell(CELL_PHASE4),
    md_cell(MD_PHASE5),
    code_cell(CELL_PHASE5),
    md_cell(MD_PHASE6),
    code_cell(CELL_PHASE6),
    md_cell(MD_PHASE7),
    code_cell(CELL_PHASE7),
    md_cell(MD_PHASE8),
    code_cell(CELL_PHASE8),
]


def main() -> int:
    for i, cell in enumerate(CELLS):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            ast.parse(src)
        except SyntaxError as e:
            print(f"Syntax error in cell {i}: {e}")
            return 1

    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes, {len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
