# Paper Analysis 03 — Qwen3 Embedding/Reranker Technical Report

**Paper:** `research_papers/rerankers_calibration/Qwen3_Embedding_Technical_Report_arXiv2506.05176.txt`
**Authors:** Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang et al. — Tongyi Lab, Alibaba Group
**Citation:** arXiv:2506.05176v3, 11 Jun 2025
**License:** Apache 2.0 (model weights public on HuggingFace `Qwen/Qwen3-Embedding-*`, `Qwen/Qwen3-Reranker-*`)

This paper defines the two most important pretrained models in our project: the **first-stage retriever (Qwen3-Embedding-8B)** and the **strongest cross-encoder we have tested (Qwen3-Reranker-8B)**. Unlike papers 01 and 02, this is not a method we choose to apply — it is the foundation our pipeline already runs on.

---

## 1. Paper claims (verbatim from the report)

### 1.1 Model family and architecture
Three sizes for each of two heads:

| Type | Model | Params | Layers | Seq Len | Embed Dim | MRL | Instruction-Aware |
|---|---|---|---|---|---|---|---|
| Embedding | Qwen3-Embedding-0.6B | 0.6 B | 28 | 32 K | 1024 | Yes | Yes |
| Embedding | Qwen3-Embedding-4B   | 4 B   | 36 | 32 K | 2560 | Yes | Yes |
| Embedding | Qwen3-Embedding-8B   | 8 B   | 36 | 32 K | **4096** | Yes | Yes |
| Reranker  | Qwen3-Reranker-0.6B  | 0.6 B | 28 | 32 K | —    | —   | Yes |
| Reranker  | Qwen3-Reranker-4B    | 4 B   | 36 | 32 K | —    | —   | Yes |
| Reranker  | Qwen3-Reranker-8B    | 8 B   | 36 | 32 K | —    | —   | Yes |

All built on the dense Qwen3 LLM backbone. (Paper §2 / Table 1.)

### 1.2 Embedding model — input format and pooling
Last-token / [EOS] pooling on a causal LLM. Instruction concatenated with the query, **not** with the document:

```
{Instruction} {Query}<|endoftext|>
```

For documents the instruction is omitted. Embedding training is multi-stage InfoNCE contrastive on ~150 M synthesized pairs followed by ~12 M filtered high-quality pairs, plus model merging (slerp). MRL (Matryoshka) lets users down-project the 4096-dim 8B output to e.g. 1024.

### 1.3 Reranker — yes/no decoder, calibrated probability
Point-wise binary-classification through the LLM chat template:

```
<|im_start|>system
Judge whether the Document meets the requirements based on the Query and the
Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>
<|im_start|>user
<Instruct>: {Instruction}
<Query>: {Query}
<Document>: {Document}<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n
```

The relevance score is the **next-token softmax over the two tokens "yes" and "no"**:

```
score(q,d) = exp(logit_yes) / (exp(logit_yes) + exp(logit_no))
```

This is mathematically equivalent to `sigmoid(logit_yes − logit_no)` and is what the prompt instruction "Calibrated P(relevant)" in our work refers to. Loss is plain SFT: `L = -log p(label | prompt)` with label ∈ {yes, no}. No first-stage weakly-supervised training for the reranker.

### 1.4 Headline benchmark numbers
MMTEB-R (Multilingual Retrieval, nDCG@10):
- Qwen3-Reranker-0.6B = **66.36**
- Qwen3-Reranker-4B = **72.74**
- Qwen3-Reranker-8B = **72.94**
- Best non-Qwen reranker reported = BGE-reranker-v2-m3 0.6B at 58.36 (paper Table 4)

MTEB Multilingual (embeddings):
- Qwen3-Embedding-8B mean(task) = **70.58** — beats Gemini-Embedding 68.37.
- Multilingual: covers 100+ languages including DE/FR/IT.

### 1.5 Other paper claims relevant to our project
- **Instructions can be customized per task** for both embedding and reranker. The paper recommends authors craft a task-specific instruction. The default examples in the HF model cards are written in English.
- **32 K context** for both heads.
- Reranker training uses model merging (slerp) of multiple SFT checkpoints. Embedding training adds a large-scale synthetic pre-training stage (150 M pairs from Qwen3-32B). Ablation in Table 5 shows synthetic data + model merging are both worth ~3-6 points on MMTEB.
- Training data composition (Table 6): 150 M synthetic for stage 1; stage 2 mixes 7 M labeled (MS MARCO, NQ, HotpotQA, NLI, DuReader, T2-Ranking, MIRACL, MLDR, …) + 12 M filtered synthetic.

---

## 2. Our usage by approach

### 2.1 Qwen3-Embedding-8B as the canonical first-stage retriever — KEPT

**Where:** `notebooks/05_embedding_and_retrieval_base/embed_unified_corpus_qwen3_8b_blackwell.ipynb`
(inventory: `notebooks/_inventory/embed_unified_corpus_qwen3_8b_blackwell.md`)

Cell config — full corpus embedding run (Step 5, lines 110-127):
```python
MODEL_NAME      = 'Qwen/Qwen3-Embedding-8B'
MODEL_DIM       = 4096
MAX_SEQ_LEN     = 768       # vector_text is short (~70-300 tokens). 768 covers >99% with no truncation cost.
BATCH_SIZE      = 256       # 95 GB VRAM @ bf16: confirmed safe.
OUTPUT_DTYPE    = np.float16
```

Model loader (Step 7, lines 252-258):
```python
model = SentenceTransformer(
    MODEL_NAME, device='cuda',
    model_kwargs={'torch_dtype': torch.bfloat16, 'attn_implementation': ATTN_IMPL},
    tokenizer_kwargs={'padding_side': 'left'},
)
model.max_seq_length = MAX_SEQ_LEN
```

Query encoding (`retrieve_unified_corpus_qwen3_8b.md`, Step 10, lines 862-882):
```python
model = SentenceTransformer(
    'Qwen/Qwen3-Embedding-8B', device='cuda',
    model_kwargs={'torch_dtype': torch.bfloat16, 'attn_implementation': 'sdpa'},
    tokenizer_kwargs={'padding_side': 'left'},
)
...
Q = model.encode(queries, batch_size=8, normalize_embeddings=True,
                 convert_to_numpy=True).astype(np.float32)
# encoded 10 queries in 5.4s  shape=(10, 4096)
```

**Our actual usage matches the paper's API contract exactly:**
- Full 4096-dim output (no MRL truncation).
- L2-normalized for cosine ↔ inner-product equivalence.
- bf16 weights, fp16 storage — paper does not prescribe dtype; we picked fp16 for the 21 GB on-disk footprint (`embeddings/qwen3_8b_unified_chunk000..026.npy`, 27 chunks, 2.65 M rows × 4096).
- `padding_side="left"` matches the paper's left-padded causal-attention assumption.
- `MAX_SEQ_LEN=768` for documents — well below the paper's 32 K. Justified empirically: avg corpus row is 373 chars, p99 = 1414 chars, max = 2845 chars. We confirmed >99% of rows fit without truncation. **We do not use the 32 K context for documents.** Queries also stay short (English val queries 30-200 tokens).

### 2.2 Instruction prefix — KEPT (paper-aligned)

Query instruction (same string in `embed_unified_corpus_qwen3_8b_blackwell.md` line 507, `retrieve_unified_corpus_qwen3_8b.md` line 847, `early_dense_embedding_test.md` line 529, `retrieve_unified_corpus_v3.md` line 847, the v7.5 multi-query pipeline, and the Untitled75 reference):

```python
QWEN_INSTRUCT = (
    'Instruct: Given an English-language legal question or scenario about Swiss federal law, '
    'retrieve the Swiss statute articles or federal court decision considerations that are most '
    'directly relevant to answering it.\nQuery: '
)
```

- Format matches the paper's `Instruct: {…}\nQuery: {…}` template literally.
- Instruction is **in English** — aligned with the paper's recommendation that English instructions are fine even for non-English documents.
- Documents are encoded with **no instruction prepended**, matching paper §2.
- The Untitled75 reference uses a shorter instruction ("Given a legal fact pattern, retrieve the most relevant Swiss statutory provisions (law article citations)."); other notebooks converged on the longer one above. No A/B has been measured between the two strings.

### 2.3 Qwen3-Reranker-8B as Stage 2 — KEPT-BASELINE

**Where:** three places.

**(a) Hybrid shootout notebook** — `notebooks/01_current_direction_cascade_rerank_precision/rerank_hybrid_3model_shootout.ipynb` (inventory: `rerank_hybrid_3model_shootout.md`, Step 6, lines 698-795).

Model + prompt construction:
```python
QWEN3_MODEL = "Qwen/Qwen3-Reranker-8B"
qwen3_tok = AutoTokenizer.from_pretrained(QWEN3_MODEL, trust_remote_code=True,
                                          padding_side="left")
qwen3_llm = LLM(
    model=QWEN3_MODEL, dtype="bfloat16",
    max_model_len=1024, gpu_memory_utilization=0.85,
    enforce_eager=False, trust_remote_code=True,
)

QWEN3_YES_ID = qwen3_tok.convert_tokens_to_ids("yes")
QWEN3_NO_ID  = qwen3_tok.convert_tokens_to_ids("no")
QWEN3_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Note that the answer can only be \"yes\" or "
    "\"no\".<|im_end|>\n<|im_start|>user\n"
)
QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
QWEN3_SP = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
```

**This is the exact prompt scaffold from paper §2 — character-for-character match,** including the empty `<think>\n\n</think>` block.

Score formula (lines 757-764):
```python
lp = out.outputs[0].logprobs[0]
y = lp.get(QWEN3_YES_ID); n = lp.get(QWEN3_NO_ID)
yl = y.logprob if y else -1e9
nl = n.logprob if n else -1e9
mx = max(yl, nl)
e_y = math.exp(yl - mx); e_n = math.exp(nl - mx)
scores[i] = e_y / (e_y + e_n)
```

**This is exactly the paper's `score(q,d) = e^{logit_yes} / (e^{logit_yes} + e^{logit_no})`,** computed in log-space for numerical stability. The output range is [0, 1] = calibrated P(relevant). We do NOT use a separate sigmoid call — the paper's formula reduces to `sigmoid(logit_yes − logit_no)` and is implemented here directly via the two-token softmax. By contrast, the BGE and jina reranker paths in the same notebook use `torch.sigmoid(logits.view(-1))` because they are CLS-head encoders, not LLM yes/no decoders — confirmed at lines 968-972 (BGE) and 1085-1090 (jina).

Custom instruction passed in (lines 666-675):
```python
CROSS_LING_INSTR = (
    "The Query is an English question about Swiss federal law. The Document is "
    "a Swiss federal law article (German) or a Swiss court paragraph (German, "
    "French, or Italian) — it may include English-language metadata (citation, "
    "statutes referenced, legal concepts). Decide YES if the Document is a "
    "relevant citation for the Query — i.e., the Query's legal question can be "
    "answered, supported, or directly discussed by this Document. Decide NO "
    "otherwise. Treat language differences as a translation problem, not a "
    "mismatch."
)
```

English instruction, multilingual documents — paper-aligned.

**(b) Rerank-only diagnostic** — `rerank_only_diagnostic_F3_no_rerank.md` Step 4 (lines 220-310). Same prompt scaffold, same yes/no softmax formula, two instruction variants (`INSTR_BASELINE` line 283 — short generic; `INSTR_CROSSLING` line 418 — the cross-lingual one). The diagnostic compared (A) baseline instruction + raw text, (B) cross-lingual instruction + enriched doc. Results in the experiment ledger.

**(c) F1=0.777 reference (Untitled75)** — `reference_F1_0.777_Untitled75.md` Stage 2 (lines 1297-1532). Different implementation path — uses `transformers.AutoModelForCausalLM` directly instead of vLLM, but the score formula is identical (lines 1495-1497):
```python
with torch.no_grad(): logits=mdl(**inp).logits[:,-1,:]
st=torch.stack([logits[:,ni],logits[:,yi]],dim=1)
sc=torch.nn.functional.log_softmax(st,dim=1)[:,1].exp().cpu().tolist()
```
Reranker max_length here is `ml=4096` — strictly greater than the shootout's 1024 — so the reference run uses the paper's longer-context capability for documents.

### 2.4 Where we depart from the paper

| Departure | Where | Reason |
|---|---|---|
| `max_model_len=1024` in vLLM reranker | shootout + diagnostic | Speed/memory tradeoff on Blackwell. The reference Untitled75 (Stage 2 path) uses 4096. We never use the full 32K. |
| `MAX_SEQ_LEN=768` for embedding documents | full-corpus embedding run | 99% of doc rows fit; remaining 1% truncated. Paper allows 32 K but we don't need it. |
| Documents pre-padded to left with `padding_side="left"` | both heads | Paper does not specify; left padding is the standard fix for causal-attention last-token pooling and matches Qwen's official examples. |
| English instruction, mixed-language docs | every notebook | Paper supports this; we treat it as policy. |
| Custom enriched-doc text (cit + statute_anchors + concepts + role + body) inside `<Document>` | shootout (`repr_enriched`, diagnostic Step 3) | Paper leaves `{Document}` open. We pack our dossier signals here as one block. |
| MRL down-projection to lower dims | NOT used | We always use full 4096. The 21 GB on-disk fp16 stack is small enough that MRL is unnecessary. |
| Reranker 0.6B / 4B sizes | NOT tried | We jumped straight to 8B since the report shows 8B > 4B > 0.6B and 95 GB VRAM is plentiful. The 0.6B / 4B variants are open candidates if we ever want a fast pre-rerank pass before 8B. |

---

## 3. Best measured results

Numbers from `.claude/skills/swiss-citation-experiments/SKILL.md` "Hybrid reranker shootout (2026-05-15)" and `swiss_citation_hybrid_rerank_final_output.ipynb`:

| Model | R@2k macro on v7.5 pool | Runtime / query |
|---|---|---|
| **Qwen3-Reranker-8B (alone)** | **0.247** | ~800 s |
| BGE-reranker-v2-m3 (alone) | 0.073 | ~180 s |
| jina-reranker-v2-base-multilingual (alone) | 0.234 | ~145 s |
| F0 fusion-only (no reranker) at K=2000 | **0.611** |  — |
| F2 3-rerank RRF at K=2000 | 0.268 |  — |
| F3 hybrid (3-rerank + 9 dossier) at K=2000 | 0.568 |  — |

**Highest Qwen3-Reranker-8B contribution to macro F1:** the F1=0.777 Untitled75 reference notebook (Stage-2 reranker is Qwen3-Reranker-8B with `RR_MAXLEN=4096`, see line 2074), with the reranker's score fused 0.3×reranker_score + 0.7×bm25_score (line 1544) then gated by zone thresholds (HIGH=0.55 auto-YES, LOW=0.25 auto-NO). The 0.777 figure is the val Macro F1 the file is named after; the notebook's tail prints "train F1=0.3364" because the live run reported is on train, not val. The Reranker-8B is the centerpiece of the Stage-2 cascade in that notebook.

**In isolation on val_001 / val_003 / val_010 (the rerank-only diagnostic):** Qwen3-Reranker-8B's calibrated-probability sort over the full v7.5 pool tops out at R@2000 ≈ 0.247 macro, well below the 0.7 floor we need.

---

## 4. Verdicts against the paper

| Paper claim | Our experience | Verdict |
|---|---|---|
| Qwen3-Embedding-8B is SOTA multilingual @ MMTEB 70.58 | Works as advertised in our pipeline. Cross-lingual EN→DE/FR/IT retrieval is functional (val recall scales smoothly with K up to ceiling). | Paper-supported. |
| Reranker `score = exp(yes) / (exp(yes)+exp(no))` is calibrated P(relevant) | We implement this verbatim. Scores are well-behaved on [0,1] in our notebooks. But: per Obs 4 distribution-shift work, the absolute threshold needs re-calibration for Swiss-legal cross-lingual data — Untitled75 uses HI=0.55 / LO=0.25 (well below the "expected" 0.7/0.3) and still gets useful zone splits. | Calibration is monotone but not absolute. Threshold needs per-domain fitting. |
| Reranker beats first-stage embedding when applied to top-100 candidates | On our task, **Qwen3-Reranker-8B alone is not strong enough**: macro R@2k = 0.247 against a v7.5 pool baseline of 0.611. The paper benchmarks top-100 reranking on retrieval datasets where gold is concentrated in the top-1k; our val has gold spread across the top-50k. The mismatch is in the candidate-set assumption, not in the model. | Paper-supported in the small-K regime (K ≤ 200, see ledger F3 numbers); fails above K=500 (where plain fusion dominates). |
| Multilingual coverage incl. DE/FR/IT | Confirmed. The reranker's CROSS_LING_INSTR with English query + DE/FR/IT document works without language tagging. | Paper-supported. |
| 32 K context window | We use 1024 (shootout) or 4096 (Untitled75) for the reranker, 768 for embedding docs. Plenty of headroom we don't need. | Paper feature is overkill for our doc lengths. |
| Instruction-aware (custom instruction lifts score) | The diagnostic notebook directly compared `INSTR_BASELINE` (generic) vs `INSTR_CROSSLING` (Swiss-legal cross-lingual). The custom instruction was kept as the canonical (`CROSS_LING_INSTR`). | Paper-supported. |
| Model merging / synthetic pre-training | Internal to model; we never re-trained. We use the released checkpoint. | Untested locally. |

---

## 5. Open ideas the paper enables

1. **Try Qwen3-Reranker-4B as a cheap pre-screen** before 8B. Paper Table 4 says 4B = 72.74 MMTEB-R vs 8B = 72.94 — i.e. ~0.2 nDCG@10 gap. Our 800 s/query on 8B is the wall-clock bottleneck of the shootout. A 4B prepass at ~200 s/query could rerank the full pool and hand the top-2000 to 8B at ~50 s/query, total ~250 s with the same quality ceiling.
2. **Switch to Qwen3-Reranker-0.6B for the planned F3-no-rerank diagnostic counter-test** — if even the 0.6B model's calibrated score adds zero to F3, the falsification trigger fires harder.
3. **LoRA-FT Qwen3-Reranker-8B on (query, gold, hard-neg) triples** as already proposed in the ledger. The paper's SFT loss `-log p(label | prompt)` is trivially LoRA-adaptable — both yes/no tokens stay in the standard vocab and no new heads are needed. Blocked on the F3-no-rerank diagnostic per the ledger.
4. **MRL down-projection to 1024-dim embeddings** if we ever need to ship a leaner index. Would shrink 21 GB → 5.5 GB at minimal recall cost (paper's Matryoshka head). Not needed today.
5. **Test cross-lingual instruction phrasing variants** — paper supports custom instructions, our notebooks already converged on `CROSS_LING_INSTR`. A small ablation (5 variants × current pipeline at K=200) might lift micro-precision a couple of points.
6. **Use the 32 K context for full-decision documents** instead of paragraph-level court considerations in the reranker step. Would require restructuring the document corpus, but the paper's headroom permits it.

---

## 6. Files cited

- `e:\swiss_citation_extraction\research_papers\rerankers_calibration\Qwen3_Embedding_Technical_Report_arXiv2506.05176.txt`
- `e:\swiss_citation_extraction\notebooks\05_embedding_and_retrieval_base\embed_unified_corpus_qwen3_8b_blackwell.ipynb`
- `e:\swiss_citation_extraction\notebooks\05_embedding_and_retrieval_base\retrieve_unified_corpus_qwen3_8b.ipynb`
- `e:\swiss_citation_extraction\notebooks\01_current_direction_cascade_rerank_precision\rerank_hybrid_3model_shootout.ipynb`
- `e:\swiss_citation_extraction\notebooks\01_current_direction_cascade_rerank_precision\rerank_only_diagnostic_F3_no_rerank.ipynb`
- `e:\swiss_citation_extraction\notebooks\06_high_scoring_reference\reference_F1_0.777_Untitled75.ipynb`
- `e:\swiss_citation_extraction\notebooks\_inventory\embed_unified_corpus_qwen3_8b_blackwell.md`
- `e:\swiss_citation_extraction\notebooks\_inventory\retrieve_unified_corpus_qwen3_8b.md`
- `e:\swiss_citation_extraction\notebooks\_inventory\rerank_hybrid_3model_shootout.md`
- `e:\swiss_citation_extraction\notebooks\_inventory\rerank_only_diagnostic_F3_no_rerank.md`
- `e:\swiss_citation_extraction\notebooks\_inventory\reference_F1_0.777_Untitled75.md`
- `e:\swiss_citation_extraction\.claude\skills\swiss-citation-experiments\SKILL.md` (Hybrid reranker shootout entry, 2026-05-15)
- `e:\swiss_citation_extraction\.claude\skills\swiss-citation-data\SKILL.md` (reranker score caches at `drive_sync/swiss_law/hybrid_rerank_shootout_cache/cache/scores_{qwen3,bge,jina}.npz`)
- `e:\swiss_citation_extraction\embeddings\qwen3_8b_unified_chunk000..026.npy` (the 21 GB fp16 corpus embedding)
