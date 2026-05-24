# investigate_misc_notebook

**Path:** `e:\swiss_citation_extraction\notebooks\13_misc_kaggle_and_utilities\investigate_misc_notebook.ipynb`

## Configuration

This notebook is unrelated to Swiss citation extraction. It is a Kaggle "Retroviral Challenge" diagnostic notebook titled "Targeted Cross-Lineage Investigation for Prime Editing RT Activity" that investigates FoldSeek and mechanistic reverse-transcriptase (RT) features for a Kaggle competition (`retroviral-challenge-predict`).

Header / setup:
- Two cells total (one large diagnostic cell, one zip-and-download cell).
- Warnings suppressed; `RANDOM_STATE = 42`; numpy/random seeded.
- Pandas display options: `max_columns=250`, `max_rows=250`, `width=240`.
- Imports include `numpy`, `pandas`, `scipy.stats` (`spearmanr`, `kruskal`), `scipy.spatial.distance`, `matplotlib`, and `sklearn` modules (`compose`, `decomposition`, `ensemble`, `impute`, `linear_model`, `metrics`, `model_selection.LeaveOneGroupOut`, `pipeline`, `preprocessing`, `svm`).
- Optional gradient-boosting libs probed via try/except: `xgboost` (`HAS_XGB`), `lightgbm` (`HAS_LGBM`), `catboost` (`HAS_CATBOOST`).
- Input roots searched: `/kaggle/input`, `.`, `/mnt/data`. Output directory: `/kaggle/working/rt_targeted_investigation_outputs`.

Feature groups configured:
- `foldseek_core` (10 cols): `foldseek_best_TM`, `foldseek_best_LDDT`, `foldseek_best_fident`, `foldseek_TM_HIV1`, `foldseek_TM_MMLV`, `foldseek_TM_MMLVPE`, `foldseek_TM_Retron`, `foldseek_TM_LTRRetrotransposon`, `foldseek_TM_Group2Intron`, `foldseek_TM_Telomerase`.
- `active_site_handcrafted` (14), `pocket_surface_selected` (10), `electrostatic_selected` (8), `sequence_simple_selected` (8), `pdb_local_yxdd` (9), `pdb_compactness_selected` (11).
- Composites: `foldseek_plus_active_site` (24), `foldseek_plus_electrostatic` (18), `foldseek_plus_pocket_surface` (20), `foldseek_plus_pdb_yxdd` (19), `foldseek_plus_selected_mechanistic` (51), `mechanistic_no_foldseek` (49).

Models configured (10): `logreg_L2_C025_balanced`, `logreg_L2_C01_balanced`, `logreg_L1_C015_balanced`, `logreg_L1_C005_balanced`, `linear_svm_C02_balanced`, `extra_trees_depth2`, `random_forest_depth2`, `lgbm_tiny`, `catboost_tiny`, `xgb_tiny`.

## Data

Resolved input paths (from `/kaggle/input/competitions/retroviral-challenge-predict/`):
- `train.csv` — shape `(57, 71)`
- `test.csv` — shape `(57, 69)`
- `sample_submission.csv`
- `feature_dictionary.csv` — shape `(66, 2)`
- `family_splits.csv` — shape `(7, 5)`
- Structures directory: `/kaggle/input/competitions/retroviral-challenge-predict/structures` (PDB found rate: 1.0; PDB train features shape `(57, 26)`).

Target columns: `active` (binary) and `pe_efficiency_pct` (efficiency). Group column: `rt_family` with 7 families — `CRISPR-associated` (5 RTs, 0 active), `Group_II_Intron` (5, 2 active), `LTR_Retrotransposon` (11, 2 active), `Other` (5, 0 active), `Retron` (12, 5 active), `Retroviral` (18, 12 active), `Unclassified` (1, 0 active).

## Pipeline

Cell 0 (the large investigation cell):
1. **Section 0**: Locate Kaggle inputs via portable `find_file` helper across `INPUT_ROOTS`.
2. **Section 1**: Metric utilities (`safe_ap`, `safe_auc`, `safe_spearman`, `metric_report`) + `LeaveOneGroupOut` splitter.
3. **Section 2**: Lightweight PDB parser (`parse_pdb_light`, normalizes B-factors/pLDDT to 0-100), helpers for radius-of-gyration, contact density, and YXDD-like catalytic motif detection (regex `[YF].[DE]D`). Computes 26 PDB features per RT (compactness, contact graph, motif geometry). Locates structures via direct path or zipfile extraction.
4. **Section 3**: Defines 7 base feature groups + 6 composites listing exact column membership.
5. **Section 4**: Per-feature Spearman vs `active` and `pe_efficiency_pct`, plus Kruskal-Wallis test against `rt_family` to detect family-confounded features.
6. **Section 5**: Preprocessor factory (`make_preprocessor`) with median imputation + missing indicator + `RobustScaler` for numeric and `OneHotEncoder` for categorical; `lofo_evaluate` runs LOGO CV per (model, feature-group).
7. **Section 6**: Builds the 10-model zoo above.
8. **Section 7**: Main LOFO comparison across all 10 models × 13 feature groups (130 model entries logged).
9. **Section 8**: Ablation — baseline FoldSeek-core, drop-one of each FoldSeek feature, add-one mechanistic group.
10. **Section 9**: Retron-fold diagnostics extracted per-model.
11. **Section 10**: Family-bootstrap stability (1000 bootstraps; resample 7 families with replacement) for top ~25 models.
12. **Section 11**: Null-label permutation tests (`N_NULL=300`) for ~11 candidate models against shuffled labels.
13. **Section 12**: Candidate ranking score combining AP, Spearman, bootstrap p10, and null p-value; ensemble built from top-5; OOF hard-case tables (FNs, FPs, high-disagreement); plots scatter `pe_efficiency_pct` vs ensemble score and bar of mean score by family.
14. **Section 13**: Full-data coefficient extraction for linear candidates.
15. **Section 14**: Optional diagnostic submission generated from the best candidate (re-fit on all train, scored on test, written to `targeted_diagnostic_submission.csv`).
16. **Section 15**: Writes manifest JSON and prints output filenames.

Cell 1: Zips `/kaggle/working` into `output_folder.zip` and emits `IPython.display.FileLink`.

## Results

Verbatim key results from cell outputs:

**Top LOFO models (overall):**
```
logreg_L2_C01_balanced__foldseek_core   AP=0.7413  Spearman=0.5668
logreg_L2_C025_balanced__foldseek_core  AP=0.7371  Spearman=0.5637
extra_trees_depth2__foldseek_core       AP=0.7245  Spearman=0.5347
extra_trees_depth2__foldseek_plus_electrostatic  AP=0.7199  Spearman=0.5046
logreg_L2_C01_balanced__foldseek_plus_pdb_yxdd   AP=0.7169  Spearman=0.5096
random_forest_depth2__foldseek_core              AP=0.6913  Spearman=0.5328
```

**Top single-feature associations (Spearman vs `active`):**
`foldseek_best_TM` 0.5505, `foldseek_TM_LTRRetrotransposon` 0.4598, `foldseek_best_fident` 0.4588, `foldseek_best_LDDT` 0.4333, `foldseek_TM_HIV1` 0.4001, `pdb_mean_degree_10A` 0.3780, `triad_best_rmsd` -0.3480, `pdb_rg_ca` 0.3338, `sasa_per_res` -0.3228. All FoldSeek features are strongly family-confounded (Kruskal p < 1e-3).

**Ablation (vs FoldSeek-core baseline AP=0.7371, Spearman=0.5637):**
- Best drop-one: `drop_foldseek_best_fident` → AP 0.7568 (delta +0.0197), Spearman 0.5876.
- Worst drop-one: `drop_foldseek_best_TM` → AP 0.6593 (delta -0.0778).
- `foldseek_plus_pdb_local_yxdd` AP 0.6245 (delta -0.1127); `foldseek_plus_electrostatic_selected` AP 0.5401 (delta -0.1970); `foldseek_plus_active_site_handcrafted` AP 0.5286 (delta -0.2085).

**Retron-fold diagnostics (top Retron AP):**
- `random_forest_depth2__foldseek_plus_pocket_surface`: Retron_AP=0.8052, Retron_ROC_AUC=0.7429.
- `random_forest_depth2__foldseek_core`: Retron_AP=0.7976, Retron_ROC_AUC=0.7143.
- `catboost_tiny__foldseek_plus_electrostatic`: Retron_AP=0.7909.
- Best linear models on `foldseek_core` had only Retron_AP≈0.44 — random forests dominate Retron specifically.

**Family-bootstrap stability (top models):**
- `logreg_L2_C01_balanced__foldseek_core`: AP_mean 0.7047, AP_std 0.1764, AP_p10 0.4069, Spearman_mean 0.5187.
- `logreg_L2_C025_balanced__foldseek_core`: AP_mean 0.6977, AP_std 0.1845.
- `extra_trees_depth2__foldseek_plus_electrostatic`: AP_mean 0.6818.
- `random_forest_depth2__foldseek_core`: AP_mean 0.6791, AP_std 0.1354 (most stable).

**Null-label tests (300 permutations):**
- `logreg_L2_C01_balanced__foldseek_core`: actual_AP 0.7413, null_p95 0.5423, empirical p_AP=0.0; Spearman p=0.020.
- `extra_trees_depth2__foldseek_core`: actual_AP 0.7245, p_AP=0.0.
- `extra_trees_depth2__foldseek_plus_electrostatic`: actual_AP 0.7199, p_AP=0.0.
- `logreg_L2_C025_balanced__foldseek_plus_selected_mechanistic`: actual_AP 0.3770, p_AP=0.590 (does not beat null).

**Candidate ensemble (top 5 by composite score):** `logreg_L2_C01_balanced__foldseek_core`, `logreg_L2_C025_balanced__foldseek_core`, `extra_trees_depth2__foldseek_core`, `random_forest_depth2__foldseek_core`, `extra_trees_depth2__foldseek_plus_electrostatic`.

**Hard false negatives (active RTs ranked low):** `Ec48-RT` (Retron, mean 0.133, rank 56), `Ne144-RT` (Retron, 0.201), `Rs-RT` (Retron, 0.277), `Vc95-RT` (Retron, 0.412), `MMTV-RT` (Retroviral, 0.468). Retron actives dominate the FN list.

**Hard false positives (inactive RTs ranked high):** `FENV1-RT` (Retroviral, 0.717, rank 8), `WDSV-RT` (Retroviral, 0.691), `RSVSB-RT` (Retroviral, 0.618), `FuRT-Cas1-RT` (CRISPR, 0.610), `IPMA-RT` (Retroviral, 0.602).

**High-disagreement RTs (largest ensemble std):** `retronE,coli-RT` (std 0.253), `Retron86-RT` (0.246), `Vp96-RT` (0.119), `AVIRE-RT` (0.115), `MMLV-RT` (0.115).

**Best linear coefficients (full-data fit):**
- `logreg_L1_C015_balanced__foldseek_core`: only `foldseek_best_TM` (0.627) and `foldseek_best_fident` (0.056) survive L1; all others zero.
- `logreg_L2_C01_balanced__foldseek_core`: top coefs `foldseek_best_TM` 0.405, `foldseek_TM_LTRRetrotransposon` 0.331, `foldseek_best_fident` 0.271, `foldseek_best_LDDT` 0.210, `foldseek_TM_MMLVPE` 0.160, `foldseek_TM_Telomerase` -0.139, `foldseek_TM_Group2Intron` -0.128.

**Diagnostic submission:** Best candidate by ranking is `logreg_L2_C01_balanced__foldseek_core` (FoldSeek-only L2 logistic regression); written to `targeted_diagnostic_submission.csv`. Sample predictions: `A.platensis-Cas1-RT` 0.419, `CRISPRRT2-RT` 0.380, `FuRT-Cas1-RT` 0.558, `Med-CasRT` 0.395, `ROSE-CRISPR-RT` 0.302.

Two `matplotlib` figures produced (scatter of OOF ensemble score vs `pe_efficiency_pct`, and bar of mean ensemble score by family).

Output artifacts written to `/kaggle/working/rt_targeted_investigation_outputs/`: `targeted_ablation_summary.csv`, `targeted_candidate_ranking.csv`, `targeted_diagnostic_submission.csv`, `targeted_family_bootstrap_stability.csv`, `targeted_feature_association_confounding.csv`, `targeted_feature_coefficients.csv`, plus per-family results, retron diagnostics, null tests, OOF hard cases, PDB features, and a manifest JSON. Cell 1 emits a `FileLink` to `/kaggle/working/output_folder.zip`.

## Summary

This notebook is not part of the Swiss citation extraction project — it is a Kaggle "retroviral-challenge-predict" diagnostic notebook investigating prime-editing RT activity prediction from FoldSeek structural similarity, AlphaFold-derived PDB descriptors, electrostatic potentials, and sequence properties on 57 reverse transcriptases across 7 families. It evaluates 10 models × 13 feature groups under Leave-One-Family-Out CV, with ablation, Retron-fold diagnostics, family-bootstrap stability (1000 reps), null-label permutation tests (300 reps), candidate ranking, ensemble construction, and full-data coefficient inspection. FoldSeek-only L2 logistic regression wins (AP 0.741, Spearman 0.567, null p ≤ 0.003); mechanistic add-ons mostly degrade overall performance, though random forests with FoldSeek+pocket-surface raise Retron AP to 0.805 versus 0.44 for the linear baselines. Retron actives (`Ec48-RT`, `Ne144-RT`, `Rs-RT`, `Vc95-RT`) and Retron false positives are the dominant failure modes, and a five-model ensemble built entirely from FoldSeek-centric models is proposed as the candidate for final submission.
