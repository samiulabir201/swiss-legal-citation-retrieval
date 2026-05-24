# Enrichment Optimization: Time Concerns and Alternatives

## Setup Summary

| Resource | Detail |
|----------|--------|
| Hardware | Kaggle 2×Tesla T4 (14.6 GB VRAM each = 29.2 GB total) |
| Model | Qwen3-8B-AWQ (INT4 quantized, ~4.5 GB loaded) |
| Inference | vLLM, tensor_parallel_size=2 (both T4s as one model shard) |
| Session limit | 9 hours per Kaggle GPU session |
| Input | 2,476,315 cards (`court_authority_cards_v4.jsonl`) |
| Current batch | 64 cards, max_model_len=3072, enforce_eager=True |
| Two-pass logic | main pass (thinking=False, 224 tokens) → retry pass (thinking=True, 384 tokens) for failures |

---

## Why It Is Slow

The bottleneck is the **retry pass**. Cards that fail JSON validation on the main pass re-run with
`enable_thinking=True` and `max_new_tokens=384`, which roughly triples the generation cost per
retried card. The Cell 9 output samples all showed `source: 'retry'`, indicating the last batch
was entirely retry-path cards.

Structured outputs (`use_structured_outputs=True`) constrain the grammar and should keep retries
low — but every retried card still pays the full thinking-pass cost.

### Estimated throughput at current settings

| Retry rate | Effective cards/s | Total hours | Kaggle sessions (9h each) |
|------------|-------------------|-------------|---------------------------|
| ~50% retried | 1–2 | 345–690 h | 38–77 |
| ~20% retried | 3–4 | 170–230 h | 19–26 |
| ~5% retried | 6–8 | 86–115 h | 10–13 |
| 0% retried | 10–12 | 57–69 h | 6–8 |

To measure your actual retry rate after the 500-card test:

```python
import json
from pathlib import Path

lines = [
    json.loads(l) for l in
    Path("/kaggle/working/court_authority_cards_v4_enriched.jsonl").read_text().splitlines()
    if l.strip()
]
retried = sum(1 for r in lines if r["_rag_generation"]["source"] == "retry")
cards_s = len(lines) / sum(r["_rag_generation"]["elapsed_s"] for r in lines)
eta_h   = 2_476_315 / cards_s / 3600

print(f"retry rate : {retried / len(lines):.1%}")
print(f"cards/s    : {cards_s:.2f}")
print(f"ETA        : {eta_h:.0f} h  ({eta_h / 9:.1f} sessions)")
```

---

## Optimization Alternatives

### Option 1 — Disable retry, trust structured outputs
**Speed gain: +50–100%. Risk: ~1–5% of cards get `null` rag_enrichment.**

Structured outputs already constrain the grammar. If the retry rate from the 500-card test is
below 5%, the retry pass is not earning its cost. Disable it for the full run.

```python
CONFIG.retry_invalid = False
```

Decision rule: if retry rate < 5% → disable retry. If retry rate > 20% → the model is
struggling with the schema and structured outputs may not be working; investigate before scaling.

---

### Option 2 — Data-parallel split (2 notebooks × TP=1)
**Speed gain: ~2×. Risk: More orchestration, need to merge outputs after.**

Instead of TP=2 (both GPUs sharing one model), run two Kaggle notebooks simultaneously — each
using one T4 with `tensor_parallel_size=1`. Split the input by line range.

**Notebook A**
```python
CONFIG.tensor_parallel_size = 1
CONFIG.gpu_memory_utilization = 0.80
CONFIG.start  = 0
CONFIG.limit  = 1_238_157
CONFIG.output_file     = Path("/kaggle/working/court_v4_enriched_part_a.jsonl")
CONFIG.checkpoint_file = Path("/kaggle/working/checkpoint_part_a.txt")
```

**Notebook B**
```python
CONFIG.tensor_parallel_size = 1
CONFIG.gpu_memory_utilization = 0.80
CONFIG.start  = 1_238_158
CONFIG.limit  = 0
CONFIG.output_file     = Path("/kaggle/working/court_v4_enriched_part_b.jsonl")
CONFIG.checkpoint_file = Path("/kaggle/working/checkpoint_part_b.txt")
```

Merge after all sessions:
```bash
cat court_v4_enriched_part_a.jsonl court_v4_enriched_part_b.jsonl \
  > court_authority_cards_v4_enriched.jsonl
```

TP=1 per notebook is slightly slower per card than TP=2, but two parallel notebooks fully
compensates and halves the total wall-clock time.

---

### Option 3 — Increase batch size to 128
**Speed gain: +15–25%. Risk: OOM if KV cache fills.**

With the 8B AWQ model occupying ~4.5 GB and 29.2 GB total VRAM in TP=2 mode, there is roughly
10 GB available for the KV cache. Doubling batch size improves GPU utilization with no quality
impact.

```python
CONFIG.batch_size             = 128
CONFIG.max_num_seqs           = 128
CONFIG.gpu_memory_utilization = 0.70  # step up carefully
```

Test with 500 cards first. If it crashes, fall back to 96.

---

### Option 4 — Disable enforce_eager (enable CUDA graphs)
**Speed gain: +10–20%. Risk: May crash on Kaggle due to JIT linker issues.**

`enforce_eager=True` was required because FlashInfer JIT linking (`-lcuda`) fails on Kaggle.
With `VLLM_ATTENTION_BACKEND=TRITON_ATTN` already set, CUDA graphs may now be safe to try.

```python
CONFIG.enforce_eager = False
```

Test with 500 cards. Revert immediately if the session crashes. If stable, it is free throughput
with no other changes needed.

---

### Option 5 — Shorten input text
**Speed gain: +10–15% on prefill. Zero quality risk for most cards.**

Cutting `text_chars` from 1400 to 1000 reduces each prompt by ~100 tokens, lowering prefill
time and leaving more KV cache headroom for larger batches.

```python
CONFIG.text_chars            = 1000
CONFIG.max_new_tokens        = 192
CONFIG.retry_max_new_tokens  = 320
```

Cards with very long `text_excerpt_original` are truncated either way; shortening the window
only drops the tail of the paragraph which rarely contains the core holding or rule statement.

---

### Option 6 — Selective enrichment (enrich only retrieval candidates)
**Speed gain: reduces 2.4M to ~50–100k cards. Most impactful option.**

The A/B experiment (session 2025-05-02) proved that enrichment only needs to cover cards that
compete in the BM25 retrieval pool. Noise rows that never surface in top-1000 do not benefit
from enrichment.

**Strategy:**
1. Run BM25 over all 10 validation queries, collect the union of top-5000 results per query → ~50k candidate cards.
2. Enrich only those 50k cards with Qwen3-8B-AWQ.
3. For the remaining 2.4M cards, use v4 fields only.

At 10 cards/s, 50k cards = **~1.4 hours = 1 Kaggle session**.

This is the most architecturally sound option. It is directly justified by the experiment result:
enriching 23 gold cards against 1M unenriched noise raised recall@1000 from 0.565 to 1.000. The
semantic specificity of enriched cards dominates over unenriched noise. You do not need to enrich
the noise to win retrieval.

For production, the candidate pool would be expanded as new queries arrive — add their top-5k to
the enrichment queue and process incrementally.

---

## Recommended Path

| Step | Action | Expected sessions |
|------|--------|-------------------|
| 1 | Run 500-card test, measure `cards/s` and retry rate | — |
| 2 | If retry rate < 5%: set `retry_invalid=False` | halved vs baseline |
| 3 | Try `batch_size=128`, `gpu_memory_utilization=0.70` | −20% further |
| 4 | If quality holds: adopt selective enrichment (50–100k priority cards) | **1–2 sessions total** |
| Optional | Data-parallel split into 2 simultaneous notebooks | halves again |

The selective enrichment path (Step 4) is the strongest lever. If the recall experiments
generalize across all 10 validation queries, enriching only the retrieval candidates is
sufficient for the production pipeline and reduces the Kaggle compute requirement by ~50×.
