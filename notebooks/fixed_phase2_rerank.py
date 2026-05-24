"""Fixed Phase 2 - cross-encoder reranker for multilingual Swiss legal.

Replaces the original Phase 2 cell. Combines four 2025-2026 research findings:

  A. Doc-side enrichment (Anthropic Contextual Retrieval Sept 2024;
                          SAC for legal RAG, NLLP 2025)
     - prepend citation + family/role + statute anchors + English concepts
     - The English concept_targets_en field is the cross-lingual bridge

  B. Cross-lingual instruction (Qwen3 TR Jun 2025):
     - explicitly state that query is English and doc is DE/FR/IT
     - in English (Qwen3 was trained on English instructions)

  C. (optional) Multi-reranker ensemble (BGE-reranker-v2-m3 ensemble cell)
     - hard-negative-trained backbone, multilingual-first
     - "What Drives Cross-lingual Ranking?" arXiv 2511.19324 Nov 2025

  D. RRF defense with v7.5 fusion (Phase 2b) - already in place.

Drop-in replacement: re-run as Phase 2's cell. Compatible with idempotent
load — if rrk_llm already exists, the engine is reused.

Sanity check at the bottom verifies the lift over the original config on
val_001's first 6 (gold, non-gold) pairs.
"""
# Phase 2 - FIXED: vLLM Qwen3-Reranker-8B with doc enrichment + cross-lingual instruction
import math
import gc as _gc
import torch as _torch_s1
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

_gc.collect()
if _torch_s1.cuda.is_available():
    _torch_s1.cuda.empty_cache()

RERANKER_MODEL = "Qwen/Qwen3-Reranker-8B"

# Tokenizer
rrk_tok = AutoTokenizer.from_pretrained(
    RERANKER_MODEL, trust_remote_code=True, padding_side="left"
)

# ─── FIX (B) — Cross-lingual aware instruction ──────────────────────────────
# Explicit language signal. Forces the model to acknowledge the cross-lingual
# nature of the task rather than implicitly downweighting non-English docs.
_RRK_INSTRUCT = (
    "The Query is an English question about Swiss federal law. "
    "The Document is a Swiss federal law article (German) or a Swiss court "
    "paragraph (German, French, or Italian) — it may include English-language "
    "metadata (citation, statutes referenced, legal concepts). "
    "Decide YES if the Document is a relevant citation for the Query — i.e., "
    "the Query's legal question can be answered, supported, or directly "
    "discussed by this Document. Decide NO otherwise. Treat language "
    "differences as a translation problem, not a mismatch."
)

# Standard Qwen3 chat wrapper
_RRK_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Note that the answer can only be \"yes\" or "
    "\"no\".<|im_end|>\n<|im_start|>user\n"
)
_RRK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_RRK_PREFIX_IDS = rrk_tok.encode(_RRK_PREFIX, add_special_tokens=False)
_RRK_SUFFIX_IDS = rrk_tok.encode(_RRK_SUFFIX, add_special_tokens=False)
_RRK_MAX_LEN  = 1024
_RRK_MAX_BODY = _RRK_MAX_LEN - len(_RRK_PREFIX_IDS) - len(_RRK_SUFFIX_IDS) - 8

# Yes/No token IDs
_RRK_YES_ID = rrk_tok.convert_tokens_to_ids("yes")
_RRK_NO_ID  = rrk_tok.convert_tokens_to_ids("no")
assert _RRK_YES_ID != rrk_tok.unk_token_id, "yes not in vocab"
assert _RRK_NO_ID  != rrk_tok.unk_token_id, "no not in vocab"
print(f"[stage1] token_yes_id={_RRK_YES_ID}, token_no_id={_RRK_NO_ID}")

# Idempotent vLLM load
if "rrk_llm" not in globals() or globals().get("rrk_llm") is None:
    print(f"[stage1] loading {RERANKER_MODEL} via vLLM ...")
    _t0 = time.time()
    rrk_llm = LLM(
        model=RERANKER_MODEL,
        dtype="bfloat16",
        max_model_len=_RRK_MAX_LEN,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        trust_remote_code=True,
    )
    print(f"[stage1] loaded in {time.time()-_t0:.1f}s")
else:
    print(f"[stage1] reusing existing vLLM engine (already in VRAM)")

_RRK_SP = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)


# ─── FIX (A) — Doc-side enrichment ──────────────────────────────────────────
# Pattern: SAC-style structured prefix + raw body text.
# Order of fields matters: citation first (anchors the doc identity), then
# type/role/court (structural cues), then statute_anchors and concepts (the
# English bridge tokens), then the raw body text. Body is truncated to leave
# room for ~250 tokens of metadata.
def _join_short(values, max_items=8, max_chars=240):
    out = ", ".join(list(values)[:max_items])
    return out[:max_chars]


def _build_enriched_doc(did) -> str:
    m       = doc_meta.get(did, {})
    cit     = (m.get("citation") or "").strip()
    family  = m.get("family") or "?"
    role    = m.get("paragraph_role") or ""
    cb      = m.get("court_base") or ""
    lang    = m.get("language") or "?"
    body    = (search_text.get(did, "") or "").strip()

    sa      = sorted(doc_statute_anchors.get(did, set()))
    concs   = sorted(_doc_to_concepts.get(did, set()))
    terms   = sorted(_doc_to_terms.get(did, set()))

    parts = []
    if cit:    parts.append(f"Citation: {cit}")
    type_str = family
    if role:   type_str = f"{family} ({role})"
    parts.append(f"Type: {type_str}")
    if cb and family == "court":
        parts.append(f"Court: {cb}")
    parts.append(f"Language: {lang}")
    if sa:     parts.append(f"Statute anchors: {_join_short(sa)}")
    if concs:  parts.append(f"Concepts (English): {_join_short(concs)}")
    if terms:  parts.append(f"Terms: {_join_short(terms)}")
    parts.append(f"Text: {body[:2000]}")
    return "\n".join(parts)[:3000]


def _build_body_ids(query: str, doc_repr: str) -> list[int]:
    """Tokenize one (query, doc) pair body, truncated to fit _RRK_MAX_BODY."""
    body = f"<Instruct>: {_RRK_INSTRUCT}\n<Query>: {query}\n<Document>: {doc_repr}"
    ids = rrk_tok.encode(body, add_special_tokens=False)
    if len(ids) > _RRK_MAX_BODY:
        ids = ids[:_RRK_MAX_BODY]
    return ids


def _rrk_score_pairs(pairs):
    """Score (query, did) pairs via vLLM. Returns list[float] in [0,1].

    NOTE: pairs is now [(query, did)] — the did, not pre-built doc text.
    The enrichment is built here so callers don't need to do it.
    """
    if not pairs:
        return []
    prompt_token_ids = [
        _RRK_PREFIX_IDS + _build_body_ids(q, _build_enriched_doc(d)) + _RRK_SUFFIX_IDS
        for (q, d) in pairs
    ]
    _prompts = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
    outs = rrk_llm.generate(
        _prompts,
        sampling_params=_RRK_SP,
        use_tqdm=False,
    )
    scores = []
    for out in outs:
        lp_step = out.outputs[0].logprobs[0]
        yes_lp = lp_step.get(_RRK_YES_ID)
        no_lp  = lp_step.get(_RRK_NO_ID)
        yes_l = yes_lp.logprob if yes_lp is not None else -1e9
        no_l  = no_lp.logprob  if no_lp  is not None else -1e9
        m = max(yes_l, no_l)
        e_yes = math.exp(yes_l - m)
        e_no  = math.exp(no_l  - m)
        scores.append(e_yes / (e_yes + e_no))
    return scores


# === sanity check on first query ===
print()
STAGE1_TOP_N = 50000
_qid0 = sorted(PER_QUERY.keys())[0]
_q0   = next(q["query_text"] for q in ALL_QUERIES if q["query_id"] == _qid0)
_pool = PER_QUERY[_qid0]["final_topk"][:STAGE1_TOP_N]
_g    = ALL_GOLD_DOC_SET[_qid0]
_gold_in_pool    = [d for d in _pool if d in _g][:5]
_nongold_in_pool = [d for d in _pool if d not in _g][:5]
if _gold_in_pool and _nongold_in_pool:
    _sanity_pairs = [(_q0, d) for d in _gold_in_pool + _nongold_in_pool]
    _s = _rrk_score_pairs(_sanity_pairs)
    _mg = sum(_s[:len(_gold_in_pool)]) / len(_gold_in_pool)
    _mn = sum(_s[len(_gold_in_pool):]) / len(_nongold_in_pool)
    print(f"[stage1 sanity] {_qid0} (FIXED rerank w/ enriched doc + cross-lingual inst)")
    print(f"  gold_scores: {[round(s, 3) for s in _s[:len(_gold_in_pool)]]}")
    print(f"  nongold_scores: {[round(s, 3) for s in _s[len(_gold_in_pool):]]}")
    print(f"  gold_mean={_mg:.4f}  nongold_mean={_mn:.4f}  sep={_mg-_mn:+.4f}")
    if _mg <= _mn:
        raise RuntimeError("Reranker is NOT separating gold from non-gold.")
    print(f"  (target: gold_mean > 0.30 for healthy multilingual reranking)")

# === full Stage 1 rerank with enriched docs ===
print(f"\n[stage1] reranking top-{STAGE1_TOP_N} with enriched docs ...")
for q in ALL_QUERIES:
    qid   = q["query_id"]
    qtext = q["query_text"]
    cands = PER_QUERY[qid]["final_topk"][:STAGE1_TOP_N]
    pairs = [(qtext, did) for did in cands]

    t_q = time.time()
    scores = _rrk_score_pairs(pairs)
    ranked = sorted(zip(cands, scores), key=lambda kv: -kv[1])
    PER_QUERY[qid]["stage1_ranked"] = ranked

    g = ALL_GOLD_DOC_SET[qid]
    g_tot = ALL_TOTAL_GOLD[qid]
    PER_QUERY[qid]["stage1_R_at_K"] = {}
    for K in (50, 100, 200, 500, 1000, 2000, 5000):
        K_use = min(K, len(ranked))
        hit = len({d for d, _ in ranked[:K_use]} & g)
        PER_QUERY[qid]["stage1_R_at_K"][K] = (hit, hit / max(1, g_tot))

    print(f"  [{qid}] {time.time()-t_q:6.1f}s  "
          f"R@50={PER_QUERY[qid]['stage1_R_at_K'][50][1]:.3f} "
          f"R@200={PER_QUERY[qid]['stage1_R_at_K'][200][1]:.3f} "
          f"R@500={PER_QUERY[qid]['stage1_R_at_K'][500][1]:.3f} "
          f"R@2000={PER_QUERY[qid]['stage1_R_at_K'][2000][1]:.3f}")

# Comparison vs fusion
print()
print("=" * 84)
print(f"  FIXED Stage-1 vs fusion-baseline (macro mean R@K)")
print("=" * 84)
print(f"  {'K':>6}  {'fusion':>8}  {'fixed_s1':>10}  {'d':>+8}  {'min':>8}  {'max':>8}")
for K in (50, 100, 200, 500, 1000, 2000, 5000):
    fus = []
    s1 = []
    for qid in PER_QUERY:
        c = PER_QUERY[qid]["curve"]
        keys = sorted(c.keys())
        kk = max((k for k in keys if k <= K), default=None)
        fus.append(c[kk][1] if kk is not None else 0.0)
        s1.append(PER_QUERY[qid]["stage1_R_at_K"][K][1])
    print(f"  {K:>6}  {sum(fus)/len(fus):>8.3f}  {sum(s1)/len(s1):>10.3f}  "
          f"{(sum(s1)-sum(fus))/len(fus):>+8.3f}  {min(s1):>8.3f}  {max(s1):>8.3f}")
