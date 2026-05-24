# exp_A2_hyde_and_enumeration

**Path:** `e:\swiss_citation_extraction\notebooks\11_early_experiments_exp_A\exp_A2_hyde_and_enumeration.ipynb`

## Configuration

- Retriever: `BAAI/bge-m3` (FlagEmbedding 1.3.5, `use_fp16=True`, dense vectors, max_length 2048 / 1024 for enum).
- LLM: `Qwen/Qwen2.5-7B-Instruct` served via vLLM (`dtype='bfloat16'`, `gpu_memory_utilization=0.80`, `trust_remote_code=True`).
- Sampling: HyDE prompt `temperature=0.3`, `max_tokens=700`; Enum prompt `temperature=0.1`, `max_tokens=600`; `top_p=0.9` for both.
- Environment: Google Colab, `VLLM_NO_USAGE_STATS=1`, manual `sys.stdout.fileno` / `sys.stderr.fileno` overrides to work around Colab IO.
- Root path: `/content/drive/MyDrive/swiss_law/data`; artifacts under `…/artifacts`.
- Fusion: Reciprocal Rank Fusion with `k_rrf=60`, `topk=2000`.
- Eval cutoffs: `k ∈ {50, 200, 500, 1000}`; statute filter excludes citations starting with `BGE ` and matching `\d[A-Z]_` / `[A-Z]\d[A-Z]_` patterns.
- Gate: `stat_recall@500 ≥ 0.40`.

## Data

- `laws_de.csv` — corpus metadata (`citation` column).
- `val.csv` — 10 validation queries.
- `val_translated_de.pkl` — German translations of val queries, keyed by `query_id`.
- `artifacts/laws_bgem3.npy` — 175,933 × 1,024 float32 doc embeddings from Exp-1.
- Corpus dimensions printed: `corpus: (175933, 1024) | val: 10`.

## Pipeline

1. Install `FlagEmbedding==1.3.5`, `vllm`, `pandas`, `numpy`, `transformers>=4.45.0`, then kill the process to restart the runtime.
2. Mount Google Drive; assert `artifacts/laws_bgem3.npy` exists from Exp-1.
3. Load `val.csv`, `laws_de.csv`, `val_translated_de.pkl`, and the doc embedding matrix.
4. Upgrade `vllm transformers accelerate`; load Qwen2.5-7B-Instruct in vLLM with a `chat()` helper that takes role-tagged messages and returns the first completion's text.
5. Define two prompt templates per query (German system+user for HyDE writing three statute-style paragraphs separated by `---`; English system+user for enumeration asking for 15 `Art. X [Abs. Y] ABBR — reason` lines).
6. For each of the 10 val queries, generate `hyde` and `enum` expansions; save to `artifacts/exp_A2_expansions.json`.
7. Delete the vLLM model and call `gc.collect()` + `torch.cuda.empty_cache()` to free GPU memory.
8. Reinstall `FlagEmbedding pandas numpy transformers==4.44.2`; reload expansions from disk if needed.
9. Load BGE-M3; embed four query forms — `en_qs`, `de_qs`, `hyde_qs`, `enum_qs` — and save to `artifacts/query_vecs.npz`.
10. Rank all 175k docs by cosine similarity for each of the four query forms (`rank_all` returns full `argsort(-sims)`).
11. `rrf()` fuses ranking lists with `1/(60+pos)` and returns the top 2000 per query via `argpartition` + `argsort`.
12. `eval_ranking()` computes `stat_recall@k` over the statute-only subset of `gold_citations` (semicolon-split, filtered by `is_statute`).
13. Evaluate single forms (`en_only`, `hyde_only`, `enum_only`) and fused variants (`en+hyde`, `en+enum`, `hyde+enum`, `en+hyde+enum`, `all4`); save to `artifacts/exp_A2_report.json` with meta block; print the best variant by `stat_recall@500` and its pass/fail vs. the 0.40 gate.

## Results

Cell 7 (expansion generation, per query character counts):

```
val_001: HyDE 2291c | Enum 737c
val_002: HyDE 1875c | Enum 994c
val_003: HyDE 1923c | Enum 720c
val_004: HyDE 1425c | Enum 757c
val_005: HyDE 2176c | Enum 595c
val_006: HyDE 1761c | Enum 904c
val_007: HyDE 1833c | Enum 541c
val_008: HyDE 1763c | Enum 635c
val_009: HyDE 1601c | Enum 685c
val_010: HyDE 1559c | Enum 972c
saved expansions
```

`val_001` enum preview (first 700 chars):
```
Art. 221 Abs. 1 lit. b StPO — Risk of collusion
Art. 221 Abs. 2 StPO — Proportionality
Art. 221 Abs. 3 StPO — Considerations for prolongation
Art. 22 StPO — Detention periods
Art. 221 Abs. 1 lit. a StPO — Risk of flight
Art. 221 Abs. 1 lit. c StPO — Risk of reoffending
Art. 221 Abs. 1 lit. d StPO — Risk of destroying evidence
Art. 221 Abs. 1 lit. e StPO — Risk of influencing witnesses
Art. 221 Abs. 1 lit. f StPO — Risk of tampering with evidence
Art. 221 Abs. 4 StPO — Review of detention
Art. 221 Abs. 5 StPO — Appeal against detention
Art. 221 Abs. 6 StPO — Release upon expiry of detention period
Art. 221 Abs. 7 StPO — Notification of detention
Art. 221 Abs. 8 StPO — Right to legal counsel
A
```

`val_001` HyDE preview (first 700 chars):
```
**Artikel 1**
Das Gericht kann im Einzelfall nach § 221 Abs. 1 lit. b StPO eine Verlängerung der vorläufigen Festnahme um drei Monate verfügen, wenn es sich um einen konkreten und nachweisbaren Risikofaktor handelt, der die Durchführung der Ermittlungen gefährdet. In dem vorliegenden Fall ist die Gefahr der Mitwirkung an der Verbrecherorganisation oder der Verschleierung der Tat durch die Beteiligung des Beschuldigten an der Einflussnahme auf Zeugen oder die Verunreinigung von Beweismitteln nachgewiesen. Diese Gefahr muss jedoch im Rahmen der Proportionalitätsprüfung in den Gesamtkontext der Ermittlungsstand und der noch ausstehenden Ermittlungsmaßnahmen eingeordnet werden.

---

**Artikel
```

Query-vec shapes:
```
query vecs: {'en': (10, 1024), 'de': (10, 1024), 'hyde': (10, 1024), 'enum': (10, 1024)}
```

Retrieval evaluation (Cell 9):

```
=== EN only (Exp-1 baseline) ===
  stat_recall@50 = 0.074
  stat_recall@200 = 0.121
  stat_recall@500 = 0.208
  stat_recall@1000 = 0.302
=== HyDE only ===
  stat_recall@50 = 0.094
  stat_recall@200 = 0.161
  stat_recall@500 = 0.235
  stat_recall@1000 = 0.255
=== Enum only ===
  stat_recall@50 = 0.114
  stat_recall@200 = 0.161
  stat_recall@500 = 0.235
  stat_recall@1000 = 0.342
=== RRF(en, hyde) ===
  stat_recall@50 = 0.081
  stat_recall@200 = 0.161
  stat_recall@500 = 0.242
  stat_recall@1000 = 0.322
=== RRF(en, enum) ===
  stat_recall@50 = 0.107
  stat_recall@200 = 0.181
  stat_recall@500 = 0.275
  stat_recall@1000 = 0.383
=== RRF(hyde, enum) ===
  stat_recall@50 = 0.107
  stat_recall@200 = 0.195
  stat_recall@500 = 0.295
  stat_recall@1000 = 0.362
=== RRF(en, hyde, enum) ===
  stat_recall@50 = 0.107
  stat_recall@200 = 0.195
  stat_recall@500 = 0.275
  stat_recall@1000 = 0.369
=== RRF(en, de, hyde, enum) ===
  stat_recall@50 = 0.101
  stat_recall@200 = 0.181
  stat_recall@500 = 0.275
  stat_recall@1000 = 0.362
```

Verdict (Cell 10):
```
best variant: hyde+enum -> stat_recall@500 = 0.295 (Exp-1 was 0.215; gate 0.40 -> BELOW GATE)
```

## Summary

Exp-A2 layers two Qwen2.5-7B prompts (German HyDE statute-style paragraphs and English enumeration of 15 citations with reasons) on top of the Exp-1 BGE-M3 dense baseline and fuses query-form rankings via Reciprocal Rank Fusion (k=60). All 10 val queries produce both expansions successfully and are encoded with BGE-M3 into 1024-d query vectors. Single-form retrieval improves over the EN baseline (`stat_recall@500` rises from 0.208 → 0.235 for both HyDE-only and Enum-only), and the best fused variant `RRF(hyde, enum)` reaches `stat_recall@500 = 0.295`. Adding the EN or DE query into the fusion slightly degrades the @500 score versus `hyde+enum` alone. The result is well below the 0.40 gate, so the experiment is flagged BELOW GATE and feeds artifacts (`query_vecs.npz`, expansions) to subsequent experiments.
