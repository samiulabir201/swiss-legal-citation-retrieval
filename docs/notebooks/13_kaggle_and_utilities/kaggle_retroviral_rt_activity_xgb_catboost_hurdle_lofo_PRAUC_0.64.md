# kaggle_retroviral_rt_activity_xgb_catboost_hurdle_lofo_PRAUC_0.64

**Path:** `e:\swiss_citation_extraction\notebooks\13_kaggle_and_utilities\kaggle_retroviral_rt_activity_xgb_catboost_hurdle_lofo_PRAUC_0.64.ipynb`

**Summary (one line):** Kaggle `retroviral-challenge-predict` pipeline — predicts reverse-transcriptase activity (`active`, binary) and prime-editing efficiency (`pe_efficiency_pct`, continuous) for 57 train / 57 test enzymes, using structural / biochemical features + ESM-2 protein embeddings (PCA-reduced 1280 → 5) + an XGBoost / CatBoost hurdle ensemble (classifier-probability × non-negative regressor-efficiency). Leave-One-Family-Out CV scored with `CLS = harmonic_mean(PR-AUC, weighted_Spearman)`. **OOF: PR-AUC = 0.6432, W-Spearman = 0.5405, CLS = 0.5874.** Despite the legacy filename `F1_0.7716`, this notebook contains no Swiss legal citation logic — it is unrelated Kaggle work kept here for completeness.

## Configuration

Imports:
- `pandas`, `numpy`
- `xgboost` (XGBClassifier, XGBRegressor)
- `catboost` (CatBoostClassifier, CatBoostRegressor)
- `sklearn.metrics.average_precision_score`
- `sklearn.decomposition.PCA`
- `sklearn.manifold.TSNE`
- `seaborn`, `matplotlib.pyplot`
- `warnings` (filterwarnings("ignore"))

Note: The notebook filename references `F1_0.7716` but the content is a Kaggle "retroviral-challenge-predict" competition pipeline (reverse transcriptase enzyme activity / efficiency prediction). It does NOT contain any Swiss DE/FR/IT legal citation retrieval code, prompts, retrieval pools, fusion/rerank logic, or per-query F1 scores. There are no LLM/embedding/reranker model stacks in this notebook.

Random seeds: `random_state=42` (XGB, PCA, TSNE), `random_seed=42` (CatBoost).

Hyperparameters (identical for both XGB and CatBoost, classifier and regressor):
- `n_estimators` / `iterations`: 100
- `max_depth` / `depth`: 1
- `learning_rate`: 0.05
- `subsample`: 0.8 (XGB only)
- `colsample_bytree`: 0.8 (XGB only)
- XGB classifier `eval_metric`: "logloss"

## Data

Input files (Kaggle competition `retroviral-challenge-predict`):
- `/kaggle/input/competitions/retroviral-challenge-predict/train.csv` — shape `(57, 71)`
- `/kaggle/input/competitions/retroviral-challenge-predict/test.csv` — shape `(57, 69)`
- `/kaggle/input/competitions/retroviral-challenge-predict/esm2_embeddings.npz` — ESM-2 protein embeddings (1280-dim), keyed by `rt_name`

Target columns:
- `active` — binary activity label (`y_bin`)
- `pe_efficiency_pct` — continuous prime-editing efficiency (`y_eff`)

Grouping column: `rt_family` (used for Leave-One-Family-Out CV).

Feature set:
- `BASE_FEATURES` (11): `foldseek_best_TM`, `foldseek_best_fident`, `foldseek_best_LDDT`, `triad_best_rmsd`, `D1_D2_dist`, `pocket_hydrophobic_per_res`, `camsol`, `perplexity`, `instability_index`, `net_charge`, `protein_length_aa`
- `PCA_FEATURES` (5): `pca_0` ... `pca_4` (PCA of ESM-2 embeddings, 1280 -> 5 components)
- Engineered:
  - `is_broken_enzyme` = `triad_best_rmsd.isna().astype(int)`
  - `length_to_TM_ratio` = `protein_length_aa / (foldseek_best_TM + 1e-5)`
- Missing values imputed with `-999`.

## Pipeline

1. **Imports / metrics**: Defines `weighted_spearman(pred, true_eff, weights)` and `compute_cls(y_true, y_score, pe_eff)` returning `(CLS, PR-AUC, weighted Spearman)`. `CLS` is the harmonic mean of PR-AUC and weighted Spearman; weights are `pe_eff + 0.01`.
2. **Load data**: Reads train/test CSVs and ESM-2 NPZ.
3. **ESM-2 PCA**: Aligns embeddings to `rt_name`, fits `PCA(n_components=5, random_state=42)` on train, transforms test, appends `pca_0..pca_4`.
4. **Feature engineering**: Builds `is_broken_enzyme`, `length_to_TM_ratio`; fills NaN with -999; returns `X_train`, `X_test`, `FINAL_FEATURES`.
5. **EDA plots** (3-panel figure, 22x6): (1) missingness heatmap; (2) `pe_efficiency_pct` stripplot by `rt_family` colored by `active`; (3) t-SNE (`perplexity=15, random_state=42`) of z-scored features colored by family, styled by activity.
6. **LOFO CV (Leave-One-Family-Out)**: For each `rt_family`:
   - Train XGB hurdle: `XGBClassifier` on all train; `XGBRegressor` on `active==1` subset; test score = `prob * clipped_efficiency`.
   - Train CatBoost hurdle: same structure with `CatBoostClassifier` / `CatBoostRegressor`.
   - OOF prediction = mean of XGB and CatBoost hurdle outputs.
7. **Full training**: Refits XGB and CatBoost classifiers on all train and regressors on `active==1` subset.
8. **Submission**: `final_predictions = (xgb_final_preds + cb_final_preds) / 2.0`; writes `submission.csv` with columns `rt_name`, `predicted_score`.

Prompts: none (no LLM / no text generation in this notebook).
Retrieval pool source: none (not a retrieval pipeline).
Fusion / rerank / final-set sizing logic: none — only a two-model average ensemble of (classifier_prob * regressor_eff) hurdle scores.

## Results

OOF metrics (from cell output):

```
OOF PR-AUC: 0.6432 | OOF W-Spearman: 0.5405 | OOF CLS: 0.5874
```

Submission output:

```
submission.csv successfully generated.
```

No per-query F1 / precision / recall values are present in the notebook. No value of `0.7716` appears in any output cell. The EDA cell's plot output was truncated in the notebook source (`Outputs are too large to include`).

## Summary

The notebook is a Kaggle "retroviral-challenge-predict" pipeline that predicts reverse-transcriptase activity and prime-editing efficiency from structural, biochemical, and ESM-2 embedding features for 57 train / 57 test enzymes. It engineers two domain features (`is_broken_enzyme`, `length_to_TM_ratio`), reduces 1280-d ESM-2 embeddings to 5 PCA components, and trains a hurdle-model ensemble averaging XGBoost and CatBoost (each = classifier-probability * non-negative regressor-efficiency). Validation is Leave-One-Family-Out over `rt_family`, scored with a custom CLS metric (harmonic mean of PR-AUC and weighted Spearman). Reported OOF scores are PR-AUC 0.6432, W-Spearman 0.5405, CLS 0.5874. Despite the filename "F1_0.7716", the notebook contains no Swiss legal citation retrieval logic, no LLM prompts, no retrieval/rerank/fusion stack, and no F1-based evaluation.
