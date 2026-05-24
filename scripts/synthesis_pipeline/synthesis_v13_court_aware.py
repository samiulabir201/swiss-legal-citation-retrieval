"""
synthesis_v13_court_aware.py — COURT + LAW HYBRID PIPELINE
==========================================================
Stage A v3 candidates (already mixes 91% court + 9% law)
  → court-aware Qwen3-Reranker scoring
  → fused score + zone split
  → Qwen3-8B judge (court & law prompts) for borderline only
  → final predictions + F1 eval

Key change vs. v12 reference notebook:
  - Stage A v3 (top-K from stage_a_v3_features.parquet) replaces fresh BM25
  - Candidate dataclass extended with court fields
  - _fmt_rr branches on candidate.family
  - JUDGE_SYSTEM split into JUDGE_SYSTEM_LAW (existing 7-category prompt)
    and JUDGE_SYSTEM_COURT (precedent-relevance prompt)
  - Per-family judge calls per query

Run order:
  1. python scripts/synthesis_pipeline/preextract_kb_for_stage_a_v3.py
  2. (Colab GPU)  python synthesis_v13_court_aware.py

Reranker + judge run on GPU. The data-loading + dossier-formatting paths
are smoke-testable on CPU via --no-rerank --no-judge.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, fields as dc_fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False


# ─────────────────────────────────────────────────────────────────────────
# Paths & config (override via CLI)
# ─────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT / "research" / "stage_a_dossier_rerank"
DEFAULT_V3 = RESEARCH_DIR / "stage_a_v3_features.parquet"
DEFAULT_COURT_KB = RESEARCH_DIR / "court_kb_for_stage_a_v3.parquet"
DEFAULT_LAWS_KB = RESEARCH_DIR / "laws_kb_for_stage_a_v3.parquet"
DEFAULT_VAL_CSV = ROOT / "data" / "val.csv"
DEFAULT_TRAIN_CSV = ROOT / "data" / "train.csv"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "synthesis_v13"

TOP_K_FROM_V3 = 100
HIGH_THRESH = 0.55
LOW_THRESH = 0.25
FUSED_W_RECALL = 0.7
FUSED_W_RERANK = 0.3

RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"
JUDGE_MODEL = "Qwen/Qwen3-8B"
RERANKER_BATCH = 8

RERANKER_INSTRUCTION_LAW = (
    "Given a query about Swiss law, determine whether the provided law "
    "article is related to or applicable to the legal issue."
)
RERANKER_INSTRUCTION_COURT = (
    "Given a query about Swiss law, determine whether the provided Swiss "
    "Federal Court paragraph is a binding or persuasive precedent for the "
    "legal issue, considering the chamber, paragraph role, and case authority."
)


def P(m: str = "", end: str = "\n") -> None:
    print(m, end=end, flush=True)


def SEP(t: str = "") -> None:
    P("\n" + "=" * 70)
    if t:
        P(f"  {t}")
        P("=" * 70)


# ─────────────────────────────────────────────────────────────────────────
# Candidate (extended for court+law)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Candidate:
    citation_canon: str
    did: str
    family: str                                # "law" | "court"
    recall_rank: int                           # Stage A v3 rank within top-K
    recall_score: float                        # Stage A v3 v3_aggr_score

    # ── shared
    title: str = ""
    text: str = ""

    # ── law-only
    law_abbreviation: str = ""
    law_name_en: str = ""
    context_heading_title: str = ""
    provision_type: str = ""
    article_topic: str = ""

    # ── court-only
    chamber: str = ""
    chamber_label: str = ""
    court_code: str = ""
    paragraph_role: str = ""
    case_importance: float = 0.0
    is_leading_decision: bool = False
    has_substantive_role: bool = False
    co_citation_count_static: int = 0
    case_peer_count: int = 0
    legal_area_static: str = ""
    doctrinal_rule: str = ""
    legal_test: str = ""
    topic_en: str = ""
    fact_pattern_tags: list = field(default_factory=list)

    # ── Stage A v3 dossier features (carried for fused-score auditing)
    article_match: int = 0
    code_in_target: int = 0
    chamber_match: int = 0
    area_match: int = 0
    is_BGE: int = 0
    co_citation_count: int = 0
    case_peer_count_v3: int = 0
    concept_cosine_score: float = 0.0
    cc_hit_top: int = 0
    cc_hit_strong: int = 0

    # ── scoring
    reranker_score: float = 0.0
    fused_score: float = 0.0
    zone: str = ""                             # "yes" | "no" | "borderline"
    llm_verdict: str = ""


# ─────────────────────────────────────────────────────────────────────────
# Reranker dossier — branched on family
# ─────────────────────────────────────────────────────────────────────────

def _fmt_rr(query: str, c: Candidate) -> str:
    if c.family == "court":
        leading = " (leading decision)" if c.is_leading_decision else ""
        substantive = " [substantive]" if c.has_substantive_role else " [non-substantive]"
        parts = [
            f"{c.citation_canon}{leading}",
            f"Court: {c.court_code} — {c.chamber_label}",
            f"Area: {c.legal_area_static}",
            f"Paragraph role: {c.paragraph_role}{substantive}",
            f"Authority: importance={c.case_importance:.2f}, "
            f"cited_by≈{c.co_citation_count_static}, "
            f"case_peers={c.case_peer_count}",
        ]
        if c.doctrinal_rule:
            parts.append(f"Doctrinal rule: {c.doctrinal_rule[:200]}")
        if c.legal_test:
            parts.append(f"Legal test: {c.legal_test[:200]}")
        if c.topic_en:
            parts.append(f"Topic: {c.topic_en}")
        parts.append(f"Text: {c.text[:500]}")
        doc = "\n".join(parts)
        instruct = RERANKER_INSTRUCTION_COURT
    else:
        doc = (f"{c.citation_canon}\n"
               f"Law: {c.law_abbreviation} — {c.law_name_en}\n"
               f"Title: {c.title}\n"
               f"Heading: {c.context_heading_title}\n"
               f"Type: {c.provision_type}\n"
               f"Text: {c.text[:500]}")
        instruct = RERANKER_INSTRUCTION_LAW
    return f"<Instruct>: {instruct}\n<Query>: {query}\n<Document>: {doc}"


# ─────────────────────────────────────────────────────────────────────────
# Judge prompts — branched on family
# ─────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM_LAW = """You are a Swiss Federal Court (Bundesgericht) legal citation expert.

DOMAIN KNOWLEDGE — Swiss Legal Citation Practice:
Swiss court decisions (BGE) and legal briefs cite provisions across multiple categories:

1. SUBSTANTIVE LAW: The core articles governing the legal issue (e.g., StGB for criminal offenses, OR for contracts, ZGB for civil matters)
2. DEFINITIONS: Articles that define key legal terms used in the case (e.g., Art. 8 ATSG defines invalidity)
3. PROCEDURAL RULES: Articles governing how the case is processed (StPO for criminal procedure, ZPO for civil procedure)
4. APPEAL PROVISIONS: Articles about legal remedies — Beschwerde (Art. 393ff StPO), Berufung, appeal deadlines
5. COST ALLOCATION: Articles about who pays court costs and attorney fees (Art. 422, 428 StPO; Art. 64 BGG)
6. COURT JURISDICTION: Articles defining which court decides (Art. 37/39 StBOG, Art. 100 BGG for Federal Court)
7. CONSTITUTIONAL PRINCIPLES: Fair trial (Art. 29 BV), proportionality, good faith (Art. 2 ZGB)

A query about pre-trial detention will cite detention rules AND appeal rules AND cost rules AND court jurisdiction.
A query about disability insurance will cite insurance provisions AND definitions AND procedural rules.

YOUR TASK: For each candidate article, read the German text carefully and decide YES or NO.
Say YES if the article belongs in ANY of the 7 categories above for this specific legal query.
Say NO only if the article is from a completely unrelated legal domain.

When uncertain, say YES — it is better to include a marginally relevant article than to miss one.

For each candidate, respond with exactly:
CITATION | VERDICT: YES or NO"""


JUDGE_SYSTEM_COURT = """You are a Swiss Federal Court (Bundesgericht) precedent expert.

For a query about a Swiss legal issue, decide whether a candidate court-decision
paragraph is binding or persuasive precedent for the issue.

KEY SIGNALS:
- Same legal area (social insurance for IV/AHV/ATSG queries; criminal procedure
  for StPO detention queries; civil law for OR/ZGB queries; constitutional for BV)
- Same procedural context (admissibility, merits, sentencing, appeal, costs)
- Authority hierarchy: BGE (leading decision) > docket judgment > cantonal
- Substantive paragraphs ("reasoning", "legal_standard", "application", "holding")
  are usually citable; "facts", "procedural_history", "notification" rarely are
- Doctrinal rule or legal test in the paragraph matches the legal question
- Chamber match (e.g. 5A for family/inheritance/SchKG; 6B for criminal;
  8C/9C for social insurance; 1B/1C for public/criminal procedure)

A precedent is RELEVANT if:
- It establishes or applies the rule that governs the query's issue, OR
- It interprets the same statute/article the query is about, OR
- It addresses the same fact pattern

Say YES when there is topical or doctrinal alignment, even if case facts differ.
Say NO only when the paragraph is from a completely different legal area.

When uncertain about marginal precedents, err on YES — Swiss courts cite
broadly, and the F1 metric rewards recall of relevant precedents.

For each candidate, respond with exactly:
CITATION | VERDICT: YES or NO"""


def build_judge_prompt(query: str, query_de: str, cands: list[Candidate],
                       family: str) -> str:
    parts = [f"LEGAL QUERY: {query[:600]}"]
    if query_de:
        parts.append(f"ANFRAGE (Deutsch): {query_de[:400]}")
    label = "BORDERLINE COURT-PARAGRAPH CANDIDATES" if family == "court" \
        else "BORDERLINE LAW-ARTICLE CANDIDATES"
    parts.append(f"\n{len(cands)} {label} to judge:\n")
    for i, c in enumerate(cands, 1):
        if family == "court":
            extra = (f"\n    Chamber: {c.chamber_label or c.chamber}  "
                     f"Role: {c.paragraph_role or 'unknown'}  "
                     f"BGE={'yes' if c.is_leading_decision else 'no'}"
                     f"  cited_by≈{c.co_citation_count_static}")
            if c.doctrinal_rule:
                extra += f"\n    Doctrinal rule: {c.doctrinal_rule[:200]}"
            txt = c.text[:400] if c.text else "(kein Text)"
            parts.append(f"[{i}] {c.citation_canon}{extra}\n    Text: {txt}\n")
        else:
            parts.append(
                f"[{i}] {c.citation_canon}\n"
                f"    Law: {c.law_abbreviation} — {c.law_name_en}\n"
                f"    German text: {(c.text or '(kein Text)')[:400]}\n"
            )
    parts.append("\nFor each candidate, output: CITATION | VERDICT: YES or NO")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────
# Gold expansion (carried from reference notebook, citation-level matching)
# ─────────────────────────────────────────────────────────────────────────

def build_parent_expansion_map(ldc):
    ci = defaultdict(list)
    es = set(ldc)
    for c in ldc:
        p = c.split()
        if len(p) >= 4 and p[0] == "Art." and p[2] == "Abs.":
            ci[(p[1], p[-1])].append(c)
    return ci, es


def expand_gold(cit, ci, es, clm, itr):
    cit = cit.strip()
    p = cit.split()
    if len(p) < 3 or p[0] != "Art.":
        return []
    m = clm.get(cit.lower())
    if m and m in itr:
        return [m]
    if "Abs." in cit:
        return []
    an, la = p[1], p[-1]
    ch = ci.get((an, la), [])
    if not ch:
        for (a2, l2), c2 in ci.items():
            if a2 == an and l2.lower() == la.lower():
                ch = c2
                break
    if ch:
        r = [clm.get(c.lower()) for c in ch]
        r = [x for x in r if x and x in itr]
        if r:
            return r
    return [m] if m and m in itr else []


def build_gold_set(gs, ci, es, clm, itr):
    raw = [c.strip() for c in str(gs or "").split(";") if c.strip()]
    lc = [c for c in raw if re.match(r"^Art\.\s+\d", c)]
    gi, gm = set(), []
    for c in lc:
        e = expand_gold(c, ci, es, clm, itr)
        if e:
            gi.update(e)
        else:
            gm.append(c)
    return gi, gm


def compute_f1(ps, gs):
    if not ps and not gs:
        return 1.0, 1.0, 1.0
    if not ps or not gs:
        return 0.0, 0.0, 0.0
    tp = len(ps & gs)
    p = tp / len(ps)
    r = tp / len(gs)
    return p, r, (2 * p * r / (p + r) if p + r > 0 else 0.0)


# ─────────────────────────────────────────────────────────────────────────
# Stage 0 — Load Stage A v3 + slim KBs
# ─────────────────────────────────────────────────────────────────────────

def load_all(v3_path: Path, court_kb_path: Path, laws_kb_path: Path,
             val_csv: Path, train_csv: Path | None):
    SEP("STAGE 0: LOADING DATA")
    P(f"  Stage A v3 features: {v3_path}")
    v3 = pd.read_parquet(v3_path)
    P(f"  v3 rows: {len(v3):,}  qids: {v3['qid'].nunique()}")

    P(f"  court KB:            {court_kb_path}")
    court_kb = pd.read_parquet(court_kb_path).set_index("did_num")
    P(f"  court rows: {len(court_kb):,}")

    P(f"  laws  KB:            {laws_kb_path}")
    laws_kb = pd.read_parquet(laws_kb_path).set_index("did_num")
    P(f"  laws  rows: {len(laws_kb):,}")

    val = pd.read_csv(val_csv)
    P(f"  val.csv: {len(val)} queries")
    train = pd.read_csv(train_csv) if train_csv and train_csv.exists() else None
    if train is not None:
        P(f"  train.csv: {len(train)} queries")

    return v3, court_kb, laws_kb, val, train


# ─────────────────────────────────────────────────────────────────────────
# Stage 1 — build Candidates from Stage A v3 top-K per query
# ─────────────────────────────────────────────────────────────────────────

def make_candidate(row: pd.Series, court_kb: pd.DataFrame, laws_kb: pd.DataFrame,
                   recall_rank: int) -> Candidate | None:
    did = row["did"]
    family = "court" if did.startswith("court:") else "law"
    did_num = int(did.split(":", 1)[1])
    if family == "court":
        if did_num not in court_kb.index:
            return None
        k = court_kb.loc[did_num]
        return Candidate(
            citation_canon=str(k.get("citation_canon") or ""),
            did=did, family="court",
            recall_rank=recall_rank,
            recall_score=float(row["v3_aggr_score"]),
            text=str(k.get("text") or ""),
            chamber=str(k.get("chamber") or ""),
            chamber_label=str(k.get("chamber_label") or ""),
            court_code=str(k.get("court_code") or ""),
            paragraph_role=str(k.get("paragraph_role") or ""),
            case_importance=float(k.get("case_importance") or 0.0),
            is_leading_decision=bool(k.get("is_leading_decision", False)),
            has_substantive_role=bool(k.get("has_substantive_role", False)),
            co_citation_count_static=int(k.get("co_citation_count_static") or 0),
            case_peer_count=int(k.get("case_peer_count") or 0),
            legal_area_static=str(k.get("legal_area_static") or ""),
            doctrinal_rule=str(k.get("doctrinal_rule") or ""),
            legal_test=str(k.get("legal_test") or ""),
            topic_en=str(k.get("topic_en") or ""),
            fact_pattern_tags=list(k.get("fact_pattern_tags")) if k.get("fact_pattern_tags") is not None and len(k.get("fact_pattern_tags")) > 0 else [],
            article_match=int(row["article_match"]),
            code_in_target=int(row["code_in_target"]),
            chamber_match=int(row["chamber_match"]),
            area_match=int(row["area_match"]),
            is_BGE=int(row["is_BGE"]),
            co_citation_count=int(row["co_citation_count"]),
            case_peer_count_v3=int(row["case_peer_count"]),
            concept_cosine_score=float(row["concept_cosine_score"]),
            cc_hit_top=int(row["cc_hit_top"]),
            cc_hit_strong=int(row["cc_hit_strong"]),
        )
    else:
        if did_num not in laws_kb.index:
            return None
        k = laws_kb.loc[did_num]
        return Candidate(
            citation_canon=str(k.get("citation_canon") or ""),
            did=did, family="law",
            recall_rank=recall_rank,
            recall_score=float(row["v3_aggr_score"]),
            text=str(k.get("text") or ""),
            title=str(k.get("title") or ""),
            law_abbreviation=str(k.get("law_abbreviation") or ""),
            law_name_en=str(k.get("law_name_en") or ""),
            context_heading_title=str(k.get("context_heading_title") or ""),
            provision_type=str(k.get("provision_type") or ""),
            article_topic=str(k.get("article_topic") or ""),
            article_match=int(row["article_match"]),
            code_in_target=int(row["code_in_target"]),
            chamber_match=int(row["chamber_match"]),
            area_match=int(row["area_match"]),
            is_BGE=int(row["is_BGE"]),
            co_citation_count=int(row["co_citation_count"]),
            case_peer_count_v3=int(row["case_peer_count"]),
            concept_cosine_score=float(row["concept_cosine_score"]),
            cc_hit_top=int(row["cc_hit_top"]),
            cc_hit_strong=int(row["cc_hit_strong"]),
        )


def build_candidates_per_query(v3: pd.DataFrame, court_kb: pd.DataFrame,
                               laws_kb: pd.DataFrame, top_k: int) -> dict[str, list[Candidate]]:
    SEP(f"STAGE 1: STAGE A v3 → top-{top_k} candidates per qid")
    out: dict[str, list[Candidate]] = {}
    for qid, group in v3.groupby("qid"):
        topk = group.nlargest(top_k, "v3_aggr_score").reset_index(drop=True)
        cands: list[Candidate] = []
        for rank, (_, row) in enumerate(topk.iterrows(), 1):
            c = make_candidate(row, court_kb, laws_kb, rank)
            if c is not None:
                cands.append(c)
        out[qid] = cands
        n_court = sum(1 for c in cands if c.family == "court")
        n_law = sum(1 for c in cands if c.family == "law")
        P(f"  {qid}: {len(cands):3d} cands (court={n_court}, law={n_law})")
    return out


# ─────────────────────────────────────────────────────────────────────────
# Stage 2 — reranker (GPU; loader-only here, batched)
# ─────────────────────────────────────────────────────────────────────────

def load_reranker(model_name: str, dev: str = "auto"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    P(f"  Loading reranker: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,
                                        padding_side="left")
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                device_map=dev, trust_remote_code=True)
    mdl.eval()
    pfx = tok.encode(
        '<|im_start|>system\nJudge whether the Document meets the '
        'requirements based on the Query and the Instruct provided. '
        'Note that the answer can only be "yes" or "no".'
        '<|im_end|>\n<|im_start|>user\n',
        add_special_tokens=False,
    )
    sfx = tok.encode(
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False,
    )
    yi = tok.convert_tokens_to_ids("yes")
    ni = tok.convert_tokens_to_ids("no")
    P(f"  yes={yi} no={ni}")
    return tok, mdl, pfx, sfx, yi, ni


def rerank_batch(query: str, cands: list[Candidate], tok, mdl, pfx, sfx, yi, ni,
                 bs: int = 8, ml: int = 4096):
    import torch
    pairs = [_fmt_rr(query, c) for c in cands]
    all_sc: list[float] = []
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i + bs]
        inp = tok(batch, padding=False, truncation="longest_first",
                  return_attention_mask=False, max_length=ml - len(pfx) - len(sfx))
        for j in range(len(inp["input_ids"])):
            inp["input_ids"][j] = pfx + inp["input_ids"][j] + sfx
        inp = tok.pad(inp, padding=True, return_tensors="pt")
        inp = {k: v.to(mdl.device) for k, v in inp.items()}
        with torch.no_grad():
            logits = mdl(**inp).logits[:, -1, :]
        st = torch.stack([logits[:, ni], logits[:, yi]], dim=1)
        sc = torch.nn.functional.log_softmax(st, dim=1)[:, 1].exp().cpu().tolist()
        all_sc.extend(sc)
    for c, s in zip(cands, all_sc):
        c.reranker_score = float(s)
    return cands


def run_reranker(qid_to_query: dict[str, str], cands_by_qid: dict[str, list[Candidate]],
                 cache_dir: Path, bs: int = 8, dev: str = "auto"):
    SEP("STAGE 2: RERANKER")
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", RERANKER_MODEL)
    cache_path = cache_dir / f"reranker_v13_{slug}.json"
    cache: dict[str, dict[str, float]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            P(f"  cached: {len(cache)} qids")
        except Exception:
            pass

    need = [(q, qid_to_query[q], cands_by_qid[q]) for q in cands_by_qid
            if q not in cache]
    if need:
        P(f"  scoring {len(need)} queries ...")
        tok, mdl, pfx, sfx, yi, ni = load_reranker(RERANKER_MODEL, dev)
        iter_ = tqdm(need, desc="  reranking") if HAVE_TQDM else need
        for qid, qtxt, cands in iter_:
            rerank_batch(qtxt, cands, tok, mdl, pfx, sfx, yi, ni, bs, 4096)
            cache[qid] = {c.did: c.reranker_score for c in cands}
        del mdl, tok
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        P(f"  saved -> {cache_path}")
    else:
        P("  all cached ✓")

    # Apply
    for qid, cands in cands_by_qid.items():
        sm = cache.get(qid, {})
        for c in cands:
            c.reranker_score = float(sm.get(c.did, 0.0))


# ─────────────────────────────────────────────────────────────────────────
# Stage 3 — fused score + zone split
# ─────────────────────────────────────────────────────────────────────────

def compute_fused_and_split(cands_by_qid: dict[str, list[Candidate]],
                             hi: float = HIGH_THRESH, lo: float = LOW_THRESH):
    SEP(f"STAGE 3: FUSED SCORE + ZONE SPLIT (hi={hi}, lo={lo})")
    stats = Counter()
    for qid, cands in cands_by_qid.items():
        if not cands:
            continue
        rsc = [c.recall_score for c in cands]
        rmin, rmax = min(rsc), max(rsc)
        rrng = max(rmax - rmin, 1e-9)
        for c in cands:
            recall_norm = (c.recall_score - rmin) / rrng
            c.fused_score = FUSED_W_RECALL * recall_norm + FUSED_W_RERANK * c.reranker_score
            if c.fused_score >= hi:
                c.zone = "yes"
            elif c.fused_score < lo:
                c.zone = "no"
            else:
                c.zone = "borderline"
            stats[c.zone] += 1
            stats[f"{c.zone}_{c.family}"] += 1
    P(f"  auto-YES: {stats['yes']}  borderline: {stats['borderline']}  auto-NO: {stats['no']}")
    P(f"   ↳ by family:")
    for z in ("yes", "borderline", "no"):
        P(f"      {z}: court={stats.get(f'{z}_court',0)}  law={stats.get(f'{z}_law',0)}")
    return stats


# ─────────────────────────────────────────────────────────────────────────
# Stage 4 — judge borderline (per-family system prompt)
# ─────────────────────────────────────────────────────────────────────────

def load_judge(model_name: str, dev: str = "auto"):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    P(f"  Loading judge: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,
                                                device_map=dev, trust_remote_code=True)
    mdl.eval()
    return tok, mdl


def judge_one_family(query: str, query_de: str, cands: list[Candidate],
                     family: str, tok, mdl) -> dict[str, str]:
    import torch
    if not cands:
        return {}
    system_prompt = JUDGE_SYSTEM_COURT if family == "court" else JUDGE_SYSTEM_LAW
    user_prompt = build_judge_prompt(query, query_de, cands, family)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=True)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=20000).to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=3000,
                           do_sample=True, temperature=0.5, top_k=20, top_p=0.95,
                           pad_token_id=tok.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    raw = tok.decode(gen, skip_special_tokens=True).strip()
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()

    verdicts: dict[str, str] = {}
    valid = {c.citation_canon for c in cands}
    valid_lower = {c.lower(): c for c in valid}
    for line in raw.split("\n"):
        line = line.strip()
        if "|" not in line:
            continue
        parts = line.split("|")
        cit_part = parts[0].strip().lstrip("[0-9] ").strip()
        verdict_part = parts[-1].strip().upper()
        v = "YES" if "YES" in verdict_part else "NO"
        if cit_part in valid:
            verdicts[cit_part] = v
        elif cit_part.lower() in valid_lower:
            verdicts[valid_lower[cit_part.lower()]] = v
    # Default unparsed → YES (err on inclusion for borderline)
    for c in cands:
        verdicts.setdefault(c.citation_canon, "YES")
    return verdicts


def run_judge(qid_to_query: dict[str, str], qid_to_query_de: dict[str, str],
              cands_by_qid: dict[str, list[Candidate]], cache_dir: Path,
              dev: str = "auto"):
    SEP("STAGE 4: LLM JUDGE (borderline only, per-family system prompt)")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "judge_v13_borderline.json"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            P(f"  cached: {len(cache)} qids")
        except Exception:
            pass

    todo = [q for q in cands_by_qid if q not in cache]
    if not todo:
        P("  all cached ✓")
        return cache

    P(f"  judging {len(todo)} queries ...")
    tok, mdl = load_judge(JUDGE_MODEL, dev)
    iter_ = tqdm(todo, desc="  judging") if HAVE_TQDM else todo
    for qid in iter_:
        cands = cands_by_qid[qid]
        bl_court = [c for c in cands if c.zone == "borderline" and c.family == "court"]
        bl_law   = [c for c in cands if c.zone == "borderline" and c.family == "law"]
        v_court = judge_one_family(qid_to_query[qid], qid_to_query_de.get(qid, ""),
                                   bl_court, "court", tok, mdl) if bl_court else {}
        v_law = judge_one_family(qid_to_query[qid], qid_to_query_de.get(qid, ""),
                                 bl_law, "law", tok, mdl) if bl_law else {}
        cache[qid] = {
            "n_borderline_court": len(bl_court),
            "n_borderline_law": len(bl_law),
            "verdicts_court": v_court,
            "verdicts_law": v_law,
        }
        # Write incrementally so a crash mid-batch doesn't lose progress
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    del mdl, tok
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return cache


# ─────────────────────────────────────────────────────────────────────────
# Stage 5 — final predictions & eval
# ─────────────────────────────────────────────────────────────────────────

def build_final_predictions(cands_by_qid: dict[str, list[Candidate]],
                            judge_cache: dict[str, dict]) -> dict[str, list[str]]:
    preds: dict[str, list[str]] = {}
    for qid, cands in cands_by_qid.items():
        jc = judge_cache.get(qid, {})
        v_court = jc.get("verdicts_court", {})
        v_law = jc.get("verdicts_law", {})
        selected: set[str] = set()
        for c in cands:
            if c.zone == "yes":
                selected.add(c.citation_canon)
            elif c.zone == "borderline":
                pool = v_court if c.family == "court" else v_law
                if pool.get(c.citation_canon, "YES") == "YES":
                    selected.add(c.citation_canon)
        if not selected and cands:
            selected = {cands[0].citation_canon}
        preds[qid] = sorted(selected)
    return preds


def evaluate(label: str, preds: dict[str, list[str]], df_split: pd.DataFrame,
             v3: pd.DataFrame, per_query: bool = True):
    SEP(f"EVAL: {label}")
    P(f"  {'QID':<14} {'P':>6} {'R':>6} {'F1':>6} {'Pred':>5} {'Gold':>5}")
    P("  " + "-" * 56)
    mp, mr, mf = [], [], []
    for _, row in df_split.iterrows():
        qid = row["query_id"]
        if qid not in preds:
            continue
        gold_dids = set(v3[(v3["qid"] == qid) & v3["is_gold"]]["did"].tolist())
        # gold_did set → citation_canon set via cands_by_qid? Not strictly necessary:
        # we report by citation_canon here using train/val gold string match.
        gold_str = set(c.strip() for c in str(row.get("gold_citations") or "").split(";") if c.strip())
        if not gold_str:
            continue
        pred = set(preds.get(qid, []))
        p, r, f = compute_f1(pred, gold_str)
        mp.append(p); mr.append(r); mf.append(f)
        P(f"  {qid:<14} {p:6.3f} {r:6.3f} {f:6.3f} {len(pred):5d} {len(gold_str):5d}")
    if mp:
        ap = sum(mp) / len(mp); ar = sum(mr) / len(mr); af = sum(mf) / len(mf)
        P("  " + "-" * 56)
        P(f"  {'MACRO':>14} {ap:6.3f} {ar:6.3f} {af:6.3f}")
        return {"p": ap, "r": ar, "f1": af, "n": len(mp)}
    return None


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v3", type=Path, default=DEFAULT_V3)
    ap.add_argument("--court-kb", type=Path, default=DEFAULT_COURT_KB)
    ap.add_argument("--laws-kb",  type=Path, default=DEFAULT_LAWS_KB)
    ap.add_argument("--val",      type=Path, default=DEFAULT_VAL_CSV)
    ap.add_argument("--train",    type=Path, default=DEFAULT_TRAIN_CSV)
    ap.add_argument("--out-dir",  type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--top-k",    type=int,  default=TOP_K_FROM_V3)
    ap.add_argument("--no-rerank", action="store_true",
                    help="Skip reranker (smoke test). reranker_score stays 0; fused = recall only.")
    ap.add_argument("--no-judge", action="store_true",
                    help="Skip judge (smoke test). All borderline default to YES.")
    ap.add_argument("--dossier-preview", type=int, default=0,
                    help="If >0, print this many sample _fmt_rr dossiers per family and exit.")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    random.seed(42)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    v3, court_kb, laws_kb, val, train = load_all(
        args.v3, args.court_kb, args.laws_kb, args.val, args.train,
    )

    qid_to_query: dict[str, str] = {row["query_id"]: row["query"] for _, row in val.iterrows()}
    qid_to_query_de: dict[str, str] = {}  # query_translations file optional; left empty

    cands_by_qid = build_candidates_per_query(v3, court_kb, laws_kb, args.top_k)

    if args.dossier_preview > 0:
        SEP("DOSSIER PREVIEW")
        for qid, cands in cands_by_qid.items():
            P(f"\n--- {qid} ---")
            qry = qid_to_query.get(qid, "")
            shown_court = shown_law = 0
            for c in cands:
                if c.family == "court" and shown_court < args.dossier_preview:
                    P(f"\n[COURT #{shown_court+1}]")
                    P(_fmt_rr(qry, c))
                    shown_court += 1
                elif c.family == "law" and shown_law < args.dossier_preview:
                    P(f"\n[LAW   #{shown_law+1}]")
                    P(_fmt_rr(qry, c))
                    shown_law += 1
                if shown_court >= args.dossier_preview and shown_law >= args.dossier_preview:
                    break
            break  # only first qid for preview
        return 0

    if args.no_rerank:
        P("\n[STAGE 2 skipped — --no-rerank set; reranker_score=0]")
    else:
        run_reranker(qid_to_query, cands_by_qid, args.out_dir / "cache")

    compute_fused_and_split(cands_by_qid)

    if args.no_judge:
        P("\n[STAGE 4 skipped — --no-judge set; borderline default to YES]")
        judge_cache = {}
    else:
        judge_cache = run_judge(qid_to_query, qid_to_query_de, cands_by_qid,
                                args.out_dir / "cache")

    preds = build_final_predictions(cands_by_qid, judge_cache)

    SEP("PREDICTIONS")
    for qid, p in preds.items():
        P(f"  {qid}: {len(p)} predictions")
    pred_path = args.out_dir / "predictions_val.json"
    pred_path.write_text(json.dumps(preds, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    P(f"  saved -> {pred_path}")

    result = evaluate("VAL", preds, val, v3)
    if result:
        (args.out_dir / "val_summary.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")

    P(f"\n[done] total elapsed {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
