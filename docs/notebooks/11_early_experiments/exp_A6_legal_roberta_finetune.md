# exp_A6_legal_roberta_finetune

**Path:** `e:\swiss_citation_extraction\notebooks\11_early_experiments_exp_A\exp_A6_legal_roberta_finetune.ipynb`

## Configuration

- Base encoder: `joelniklaus/legal-swiss-roberta-large` (1024-dim, 434.96 M params, pre-trained on Swiss DE/FR/IT legislation + court decisions).
- Wrapper: `sentence-transformers` `Transformer` + mean-`Pooling`, `MAX_LEN = 512`.
- Pre-training stage (SimCSE, Cell 10): same-text dropout positives, `MultipleNegativesRankingLoss`, 1 epoch, batch 64, lr 3e-5, warmup 1000, AMP on. Output: `A6/legal_swiss_simcse/`.
- Hard-negative mining (Cell 12): BGE-M3 encode of all anchors, top-10 over cached `laws_bgem3.npy` (chunk 10 000), drop any candidate equal to the gold positive, fall back to last hit if none qualify.
- Fine-tune (Cell 16): triplets (anchor, positive, hard-neg) → `MultipleNegativesRankingLoss(scale=20.0)`, `EPOCHS=1`, `BATCH_TRAIN=32`, `LR=2e-5`, `WARMUP=500`, AMP on, checkpoints every 5 000 steps (`A6/ckpt`, limit 2). Output: `A6/legal_swiss_finetuned/`.
- Environment: Colab + Google Drive at `/content/drive/MyDrive/swiss_law/data`. GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 101.4 GB free VRAM. Seeds: `random.seed(0); np.random.seed(0); torch.manual_seed(0)`.
- Deps (Cell 1): `numpy==1.26.4`, `sentence-transformers>=3.1.0`, `transformers>=4.45.0`, `accelerate>=0.34`, `datasets<3.0.0`, `pandas<2.2.0`.
- Eval gates: `stat_recall@500 > 0.376` (Exp-3 ceiling) PROMOTE; `stat_recall@200 ≥ 0.45` strong; `case_recall@200 > 0.108` (Exp-5).

## Data

- Inputs asserted in Cell 2 from `ROOT = /content/drive/MyDrive/swiss_law/data`:
  - `laws_de.csv`, `val.csv`, `val_translated_de.pkl` (loaded for queries).
  - `artifacts/laws_bgem3.npy`, `artifacts/query_vecs_A3.npz`, `artifacts/court_bgem3.npy`, `artifacts/court_citations.json`, `artifacts/exp_A3_expansions.json`.
- Supervised pair mining source: `rcds/swiss_citation_extraction` (`original` config) loaded via `datasets.load_dataset(..., trust_remote_code=True)`. Splits: `train` 87 760, `validation` 12 359, `test` 27 364 rows. Features: `decision_id, considerations, NER_labels, law_area, language, year, chamber, region`. Label set: `['O', 'B-CITATION', 'I-CITATION', 'B-LAW', 'I-LAW']`.
- IOB extraction (Cell 4): scanned 127 483 rows → 100 749 with citations → 251 807 law spans + 129 793 BGE spans. Citation surface normalised via `norm_law` (→ `Art. N ABBR`) and `norm_bge` (→ `BGE V P R`).
- Pair resolution (Cell 5): `laws_de.csv` indexed by `Art. N ABBR` prefix (9 594 entries), court rows indexed by `BGE V P R` (7 102 entries). Anchor length filter 60–4 000 chars. Outcome: 123 512 training pairs (`stat=21 684`, `case=101 828`, unresolved 258 088). Court text fetched for 3 202 unique rows from `court_considerations.csv` (chunk 100 000). Final pairs: **117 453**; written to `A6/train_anchors.npy` and `A6/train_positives.npy`.
- SimCSE unsupervised corpus (Cell 10): full `laws_text` plus 300 000-cap court considerations, length-filtered (>30 chars), shuffled. Total **466 833** sentences.
- Validation: `val.csv` paired with `val_translated_de.pkl`; 10 queries with four forms (`q_en, q_de, q_hyde, q_enum`) sourced from `exp_A3_expansions.json` and `val_de`. Gold parsed by `;`, classified by `cit_class` into `statute / bge / docket`.

## Pipeline

1. Cell 1 — install pinned deps (numpy 1.26.4 downgrade warned with multiple dependency conflicts).
2. Cell 2 — mount Drive, sanity-check inputs, GPU/VRAM check, seed RNGs.
3. Cell 3 — download `rcds/swiss_citation_extraction` (original), print schema/sample.
4. Cell 4 — dynamic token/label-field detection, IOB span extraction, normalise to canonical statute/BGE citation strings, save first 5 pairs to `A6/iob_pairs.json`.
5. Cell 5 — index statute corpus by `Art. N ABBR` prefix and court corpus by BGE prefix; resolve citations to corpus text; stream `court_considerations.csv` once for needed court texts; save `train_anchors.npy` / `train_positives.npy`.
6. Cell 6 — instantiate `legal-swiss-roberta-large` `Transformer`+`Pooling(mean)` model on CUDA; load report shows expected LM-head unexpected keys and missing `pooler.dense.*` (newly initialised).
7. Cell 7 — pre-train baseline (no fine-tune): encode all 175 933 laws + 10 DE queries, compute `stat_recall@{50,100,200,500}` against `gold_citations`. Free embeddings afterwards.
8. Cell 8 — placeholder/text-only step; explicit hard-neg mining deferred; relies on in-batch MNRL negatives at bs=32 (31 negatives/anchor).
9. Cell 9 — markdown: SimCSE unsupervised pre-training rationale.
10. Cell 10 — gather unsup corpus (laws + sampled court), build `InputExample(texts=[t, t])`, train SimCSE on a fresh base wrapper, save to `A6/legal_swiss_simcse/`.
11. Cell 11 — reload the SimCSE checkpoint as `model` for downstream mining + FT.
12. Cell 12 — load BGE-M3, encode all 117 453 anchors, top-10 retrieval over `laws_bgem3.npy` via chunked GPU matmul + `torch.topk`, filter exact-match false negatives, produce `hard_negatives` list; free VRAM.
13. Cell 13 — print sanity preview of 3 hard negatives.
14. Cell 14 — on 1 000 (anchor, hard-neg) pairs, measure cosine similarity under the current (post-SimCSE) model.
15. Cell 15 — pre-training-end eval: re-encode laws + DE queries and recompute `stat_recall` to confirm SimCSE state.
16. Cell 16 — assemble triplet `InputExample`s, train `model` with MNRL for 1 epoch, save to `A6/legal_swiss_finetuned/`.
17. Cell 17 — reload fine-tuned model as `ft`, re-encode 175 933 laws → `laws_swissroberta.npy`; encode all four query forms; save `query_vecs_A6.npz`.
18. Cell 18 — per-query and RRF (k=60, top 1000) `stat_recall@{50,100,200,500,1000}` over the four query forms.
19. Cell 19 — compute best-of recall, evaluate gates vs Exp-3 thresholds, gate court re-encode, write `exp_A6_report.json`.
20. Cell 20 — conditional case-recall on court if gate passed (skipped here).
21. Cell 21 — comment-only (Faiss removed in favor of native PyTorch).
22. Cell 22 — diagnostic histogram of positive vs hard-negative similarities under the fine-tuned model (1 000 triplets).
23. Cell 23 — empty cell.

## Results

Cell 1 — pip install with dependency-resolver conflicts (tobler, rasterio, shap, xarray-einstats, opencv-python-headless, jaxlib, pytensor, jax, cupy-cuda12x, opencv-contrib-python, opencv-python all expect `numpy>=2`; `google-colab 1.0.0` expects `pandas==2.2.2`).

Cell 2 — `Mounted at /content/drive`; `GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition`; `Free VRAM (GB): 101.4`.

Cell 3 — dataset loaded with splits train 87 760 / validation 12 359 / test 27 364; features `['decision_id', 'considerations', 'NER_labels', 'law_area', 'language', 'year', 'chamber', 'region']`; first row preview shown for a `de` social-law decision from 2014.

Cell 4 — `label set: ['O', 'B-CITATION', 'I-CITATION', 'B-LAW', 'I-LAW']`; scanned 127 483 rows → 100 749 with citations → 251 807 law spans + 129 793 BGE spans; preview written to `/content/drive/MyDrive/swiss_law/data/artifacts/A6/iob_pairs.json`.

Cell 5 — `laws prefix index: 9594 | BGE court index: 7102`; `training pairs: 123512 | stat=21684 | case=101828 | unresolved=258088`; `need text for 3202 unique court rows`; `collected court text for 3202 rows`; `final training pairs: 117453`.

Cell 6 — RobertaModel load report: UNEXPECTED keys `lm_head.layer_norm.bias/weight, embeddings.position_ids, lm_head.dense.bias/weight, lm_head.bias`; MISSING `pooler.dense.bias/weight`; `encoder ready | dim: 1024 | max_len: 512 | total params: 434.960384 M`.

Cell 7 — `pre-train laws encode: (175933, 1024) in 240s`; `legal-swiss-roberta (NO FT) DE-query` recalls:

```
stat_recall@50  = 0.013
stat_recall@100 = 0.020
stat_recall@200 = 0.040
stat_recall@500 = 0.060
```

Cell 8 — `Skipping explicit hard-neg mining — using MultipleNegativesRankingLoss with in-batch negatives.` `Effective negatives per anchor at bs=32: 31 (other batch members)`.

Cell 10 — `Total sentences for SimCSE pre-training: 466833`; same load-report set on fresh wrapper; SimCSE complete, saved to `/content/drive/MyDrive/swiss_law/data/artifacts/A6/legal_swiss_simcse`.

Cell 11 — `SimCSE model loaded successfully into the model variable!`.

Cell 12 — `Mined 117453 hard negatives`; retrieval `0.6s` after BGE-M3 anchor encode.

Cell 13 — sample hard-negative previews:

```
[0] Art. 106 Abs. 2 BGG | Bundesgesetz vom 17. Juni 2005 über das Bundesgericht (Bundesgerichtsgesetz, BGG) - 5. Abschnitt:  Weitere Verfahrensbestimmunge...
[1] Art. 84c Abs. 3 MStP | Militärstrafprozess vom 23. März 1979 (MStP) - Persönlichkeitsschutz des Opfers | 3 Die Behörden vermeiden eine Begegnung des O...
[2] Art. 81 Abs. 1 BGG | Bundesgesetz vom 17. Juni 2005 über das Bundesgericht (Bundesgerichtsgesetz, BGG) - 2. Abschnitt:  Beschwerde in Strafsachen | 1 ...
```

Cell 14 — anchor↔hard-neg cosine on 1 000 pairs (post-SimCSE): `Average = 0.2945`, `Min = -0.0608`, `Max = 0.8752`.

Cell 15 — post-SimCSE val eval (`Current Model (Pre-Training) DE-query`):

```
stat_recall@50  = 0.007
stat_recall@100 = 0.007
stat_recall@200 = 0.013
stat_recall@500 = 0.054
```

Cell 16 — `Training mode: Triplets (Anchor, Positive, Hard Negative)`; `training examples: 117453 | batch=32 | epochs=1`; `training done in 35.3 min | saved to /content/drive/MyDrive/swiss_law/data/artifacts/A6/legal_swiss_finetuned`.

Cell 17 — `laws encode: (175933, 1024) in 241s`; `queries encoded: (10, 1024) (10, 1024) (10, 1024) (10, 1024)`.

Cell 18 — fine-tuned recall by query form:

```
=== A6 ft EN-query ===
  stat_recall@50  = 0.020
  stat_recall@100 = 0.034
  stat_recall@200 = 0.047
  stat_recall@500 = 0.067
  stat_recall@1000 = 0.081
=== A6 ft DE-query ===
  stat_recall@50  = 0.040
  stat_recall@100 = 0.040
  stat_recall@200 = 0.047
  stat_recall@500 = 0.067
  stat_recall@1000 = 0.094
=== A6 ft HyDE ===
  stat_recall@50  = 0.047
  stat_recall@100 = 0.054
  stat_recall@200 = 0.081
  stat_recall@500 = 0.101
  stat_recall@1000 = 0.114
=== A6 ft Enum ===
  stat_recall@50  = 0.047
  stat_recall@100 = 0.054
  stat_recall@200 = 0.060
  stat_recall@500 = 0.074
  stat_recall@1000 = 0.107
=== A6 ft RRF(de,hyde,enum) ===
  stat_recall@50  = 0.054
  stat_recall@100 = 0.054
  stat_recall@200 = 0.074
  stat_recall@500 = 0.101
  stat_recall@1000 = 0.128
=== A6 ft RRF(en,de,hyde,enum) ===
  stat_recall@50  = 0.054
  stat_recall@100 = 0.054
  stat_recall@200 = 0.060
  stat_recall@500 = 0.094
  stat_recall@1000 = 0.121
```

Cell 19 — verdict:

```
PRE-TRAIN (no FT) DE @500: 0.060   (sanity floor; should not be ≥ Exp-3=0.376 yet)
A6 best stat_recall@500 = 0.101   (Exp-3 ceiling = 0.376)
A6 best stat_recall@200 = 0.074   (Exp-3 = 0.315)

gate stat_recall@500 > 0.376  -> FAIL
gate stat_recall@200 ≥ 0.45   -> FAIL

statute gate FAILED — skipping court re-encode. Diagnose Cell 11 first.

report saved: /content/drive/MyDrive/swiss_law/data/artifacts/exp_A6_report.json
```

Cell 20 — `court not re-encoded — skip case eval`.

Cell 21 — no output (comment only).

Cell 22 — fine-tuned-model similarity distribution on 1 000 triplets: `Positives — Mean: 0.5993, Min: 0.2085, Max: 0.8954`; `Hard Negatives — Mean: -0.0597, Min: -0.2374, Max: 0.8022`; `Margin (Mean Pos - Mean HN): 0.6591`. A `<image/png>` matplotlib/seaborn histogram is rendered.

Cell 23 — empty cell (no source, no outputs).

## Summary

The experiment wrapped `joelniklaus/legal-swiss-roberta-large` as a mean-pool bi-encoder, mined 117 453 (anchor, positive) pairs from IOB-tagged citations in `rcds/swiss_citation_extraction`, did SimCSE pre-training on ~467 K Swiss legal sentences, mined hard negatives over cached BGE-M3 embeddings, and fine-tuned 1 epoch with `MultipleNegativesRankingLoss` (scale 20, bs 32, lr 2e-5) in 35.3 min on a Blackwell 6000. The pre-fine-tune mean-pool baseline at `stat_recall@500 = 0.060` showed the wrapper was non-degenerate but far from BGE-M3 numbers; post-SimCSE it dropped to 0.054. After supervised fine-tuning, the best fused recall reached `stat_recall@500 = 0.101` (RRF de/hyde/enum) and `stat_recall@200 = 0.074`, well below the Exp-3 ceilings (0.376 / 0.315), so both `>0.376` and `≥0.45` gates failed. The court re-encode was skipped per gate, but the post-FT triplet distribution showed a clean separation (positives mean 0.5993 vs hard-negative mean -0.0597, margin 0.6591), suggesting the encoder learned a useful contrastive space that nevertheless does not transfer to the en→de val queries — consistent with the notebook's flagged failure mode that the bi-encoder approach atop this base needs a stronger contrastive pretraining or a different domain bridge.
