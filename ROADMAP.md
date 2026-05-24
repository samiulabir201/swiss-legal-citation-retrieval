# Roadmap — Swiss Legal Citation Extraction

**Target:** Macro F1 0.6-0.8 on val.csv (10 EN queries → 2.65M Swiss DE/FR/IT corpus)
**Updated 2026-05-15** after repo reorg.

For full project history & rationale, also read the .claude skills:
- `.claude/skills/swiss-citation-orchestrator/SKILL.md` (start here)
- `.claude/skills/swiss-citation-experiments/SKILL.md` (every approach tried + verdict)
- `.claude/skills/swiss-citation-approach/SKILL.md` (current direction + pivot history)

---

## Where we are now (as of 2026-05-15)

### Solved problems (don't redo these)

1. **Pool recall — solved at the val_003 ceiling.**
   v7.5 multi-query snapshot achieves **macro R@50k = 0.893**. Per-query R_max ranges from 0.766 (val_003, the binding) to 1.000 (val_004/005). Any plan that assumes "min recall ≥ 0.8" is mathematically blocked.

2. **Cross-encoder rerankers — proven useless above K=500.**
   Shootout (Qwen3-Reranker-8B vs BGE-v2-m3 vs jina-v2-base-multilingual) showed plain fusion (F0) beats every reranker config at K ≥ 500. At K=2k, F0=0.611 > F3=0.568. At K=10k, F0=0.816 > F3=0.791. F3's small-K lift comes from the **9 dossier signals**, not the rerankers.

3. **Citation-graph co-prediction — dead.**
   99.41% of gold pairs have NO graph edge (Obs 2). Graph is a tie-breaker, not a co-predictor.

### Open problem: 50k pool → 20-40 final picks at Macro F1 0.6-0.8

The bottleneck is **Stages 2-3 (dossier-aware reranking + variable-K final pick)**, NOT retrieval.

---

## Concrete next-step plan

### Step 0 — F3-without-rerankers diagnostic (15 min, no GPU)
Use cached `scores_{qwen3,bge,jina}.npz` + `dossier_features.npz` from
`drive_sync/swiss_law/hybrid_rerank_shootout_cache/cache/`.
Recompute F3's RRF dropping the three reranker rank columns; keep only the 9 dossier signals.
**Pass criterion:** F3-no-rerank within 2 points of F3 at every K → drop rerankers entirely from the pipeline.

### Step 1 — Cascade dossier Phase 1 (the agreed plan)
Doc: `research/cascade_dossier_plan.md`.
Phase 1 enrichments (pre-computed once after Stage 1):
1. **Channel-of-arrival fingerprint** — which of 15 channels surfaced each doc (free; already in snapshot's `channel_hit_sets`)
2. **Statute-target intersection** — `doc_statute_anchors ∩ ALL_TARGETS[qid]["statute_targets"]` (count + list)
3. **Co-citation density** — for each law in top-100, count top-100 court paragraphs citing it

Surface these in Stage 2 LLM prompt block.
**Pass criterion:** Stage 2 mean R@100 lift from ~0.55 → ≥ 0.70.
**Fail action:** layer Phase 2 signals (concept-target intersection, term intersection by language, paragraph-role × query-type).

### Step 2 — Variable-K LLM gate over Stage 2 top-100
Per-candidate accept/reject judge using Qwen3-8B. Critical fixes vs. broken endgame judge:
- DELETE `cache_endgame/judge/` before any judge run (87% rubber-stamp YES poisoning)
- Force JSON output; default-NO on parse failure (opposite of old)
- No `<think>` budget; return 1-token verdict + 1 confidence float
**Pass criterion:** Macro F1 ≥ 0.30 (intermediate milestone).

### Step 3 — Calibrate K per query from cheap features
Features: count of `statute_targets`, # codes mentioned, query length, court/law ratio.
Take top-K_predicted from dossier-ranked list ∩ gate-YES set.
**Pass criterion:** Macro F1 ≥ 0.45.

### Step 4 — Granularity-aware final pick
- Apply `gold_parent_link_check` rule: if `Art. 11 Abs. 2 OR` is predicted, `Art. 11 OR` must NOT also be predicted
- Exact-string match against corpus
- Don't let dossier signals push law-heavy predictions (val is ~40% court)
**Pass criterion:** Macro F1 ≥ 0.60 (target hit).

---

## What we are NOT doing (and why)

| Banned move | Why |
|---|---|
| Adding more retrieval channels | Pool R is saturated at val_003 R_max=0.766 ceiling |
| Bigger reranker / bigger judge LLM | Recall is set by the pool, not by reranking |
| Treating citation graph as a co-prediction expander | Obs 2: 99.4% of gold pairs have no edge |
| Fixed K prediction | Variable gold size 10-47 makes fixed K provably suboptimal |
| Query-specific hardcoding | Banned by `.claude/memory/feedback_no_query_specific_hardcoding.md` |
| FAISS-IVF for vector index | Full-GPU brute force is ~50 ms/query on Blackwell; no need |
| Re-extracting citation graph | Already patched 92.9% → 96.0% coverage |
| UTF-8 repair on data files | Already clean UTF-8 |
| Re-merging court LLM enrichment | Already merged in `artifacts/unified_retrieval.sqlite` |

---

## Living-document rule

After every experiment:
1. Update `.claude/skills/swiss-citation-experiments/SKILL.md` with approach name + measured numbers + verdict (kept / rejected / open)
2. Update `.claude/skills/swiss-citation-approach/SKILL.md` if the active direction changed
3. Update this ROADMAP.md if a step is completed or new step added

Treat these updates as part of completing the work, not optional cleanup.

---

## Hardware & workflow (`.claude/skills/swiss-citation-config/SKILL.md`)

- Colab Pro+ on NVIDIA RTX PRO 6000 Blackwell, 95.6 GB VRAM
- Full corpus embeddings fit in VRAM (21 GB fp16). Brute-force dense search is ~50 ms/query.
- Notebook-first for every new experiment. New notebook per approach (don't extend a result-producing notebook).
- Pass/fail criterion in the first markdown cell of every experimental notebook.
- Cache GPU-expensive intermediates (`.npz` for scores, `.json` for LLM outputs).
- Drive paths only mount in Colab; local Windows uses `drive_sync/` mirror.
