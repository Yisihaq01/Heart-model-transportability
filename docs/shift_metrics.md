# Shift diagnostics — metric definitions and interpretation

Stage 1.6 produces **feature-level** and **aggregate** shift artifacts for every external train→test pair. Paths and filenames are fixed and documented in `ImplementationPlan/pipeline_outputs.md`.

---

## Artifact layout (reproducible paths)

- **Per pair** (`outputs/shift/{train_sites}__to__{test_site}/`):
  - `shift_diagnostics.json` — summary only (prevalence, mean PSI, C2ST AUC).
  - `feature_shift.parquet` / `feature_shift.csv` — canonical per-feature table (see below).
- **Global** (`outputs/shift/`):
  - `shift_table.parquet` / `.csv` — one row per experiment (aggregate shift).
  - `performance_table.parquet` / `.csv` — one row per experiment (auc_drop, brier_change).
  - `shift_performance_merged.parquet` / `.csv` — merged table linking shift metrics to performance drops.
  - `shift_performance_correlation.parquet` / `.csv` — Spearman correlation matrix of merged numeric columns.
  - `cross_experiment_index.json` — list of pair keys and relative paths to per-pair artifacts.

---

## How each shift metric is computed

### PSI (Population Stability Index)

- **Input:** One continuous or binned feature column from train and from test.
- **Computation:** Train distribution is binned using `psi_bins` (default 10) percentiles of the **train** sample. The same bin edges are applied to train and test; counts are converted to proportions. PSI = Σ (p_test − p_train) × ln(p_test / p_train), with proportions clipped away from zero to avoid log(0).
- **Output:** Single non-negative float per feature (in `feature_shift.parquet` column `psi`). Mean across features is in `shift_diagnostics.json` as `mean_psi`.
- **Interpretation:** 0 = no distribution shift; &lt; 0.1 often “stable”; 0.1–0.25 moderate; &gt; 0.25 substantial. High PSI on a feature suggests that feature’s distribution has shifted between train and test.

### Statistical tests (KS, χ²)

- **Continuous features:** Two-sample **Kolmogorov–Smirnov (KS)** test on train vs test (non-null values). Column `test` = `"ks"`; `statistic` = KS statistic; `p_value` = two-tailed p-value.
- **Categorical features:** **Chi-squared test** of independence on the contingency table (category × domain train/test). Column `test` = `"chi2"`; `statistic` = χ²; `p_value` = p-value. If only one category or one domain has counts, the test is skipped and statistic/p_value are null.
- **Interpretation:** Low p-value suggests the feature’s distribution differs between train and test. Used alongside PSI for feature-level shift.

### Missingness deltas

- **Computation:** For each feature, missing rate (proportion of NaNs) in train and in test; difference = test_miss_pct − train_miss_pct (in percentage points).
- **Output:** In `feature_shift.parquet`: `train_miss_pct`, `test_miss_pct`, `diff_pct`.
- **Interpretation:** Large positive `diff_pct` means more missingness at test; can indicate different data collection or coding and may contribute to performance drop.

### Prevalence shift (label)

- **Computation:** Mean of `target` in train vs test. Stored in `shift_diagnostics.json` as `prevalence_shift`: `train_prevalence`, `test_prevalence`, `absolute_diff`.
- **Interpretation:** Large `absolute_diff` indicates label shift; can affect calibration and reported AUC.

### C2ST (Classifier Two-Sample Test) AUC

- **Computation:** Train and test feature matrices are combined; a binary label indicates origin (0 = train, 1 = test). A Random Forest is cross-validated (5-fold) to predict origin from features (after the same preprocessing as the model). The ROC AUC of this classifier is reported.
- **Output:** Single float in `shift_diagnostics.json` as `c2st_auc`.
- **Interpretation:** AUC ≈ 0.5 suggests little detectable multivariate shift; AUC &gt; 0.5 indicates the classifier can distinguish train from test (higher = stronger multivariate shift).

---

## Global shift–performance correlation

- **Merged table:** `shift_performance_merged` joins `shift_table` and `performance_table` on `(train_sites, test_site, model)`, so each row is one external experiment with shift metrics and performance deltas (e.g. `auc_drop`, `brier_change`).
- **Correlation:** Spearman rank correlation is computed on the numeric columns of this merged table (e.g. `mean_psi`, `prevalence_diff`, `c2st_auc`, `auc_drop`, `brier_change`). The matrix is written to `shift_performance_correlation.parquet` / `.csv`.
- **Interpretation:** Positive correlation between e.g. `mean_psi` and `auc_drop` (or negative correlation with external AUC) suggests that larger covariate shift is associated with larger performance drop.
