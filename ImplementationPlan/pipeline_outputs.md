# Pipeline Output Artifacts

All files produced by `src/ingest.py` and `src/pipeline.py`. Primary format is Parquet; CSV mirrors are written alongside for portability. Stats and metadata use JSON or Joblib.

---

## Canonical experiment artifact contract

Every **experiment** (internal, internal_cfs, external_uci, external_kaggle_uci) writes the following under its `{outputs_subdir}/{...}/{model_key}/` folder:

| File | Meaning | Format | Required / Optional |
|------|---------|--------|---------------------|
| `results.json` | Metrics, hyperparameters, CIs, metadata (no predictions/fitted objects) | JSON | **Required** |
| `predictions.parquet` | Hold-out predictions (y_true, y_prob, y_pred, fold if internal) | Parquet | **Canonical** (when experiment has predictions) |
| `predictions.csv` | Same as parquet; convenience for inspection/portability | CSV | Optional |
| `model.joblib` | Fitted estimator (sklearn / XGBoost / LightGBM) | Joblib | When fitted model saved |
| `pipeline.joblib` | Fitted preprocessing pipeline (imputer + encoder ± scaler) | Joblib | When preprocessing pipeline exists |

Downstream stages (calibration, size_matched, shift) consume these and write their own artifacts (see below). Ingestion outputs live in `data/` and are separate.

---

## Main Pipeline Output Artifacts

### Ingestion Outputs (`data/`)

| Artifact | Parquet | CSV | When |
|---|---|---|---|
| Kaggle cleaned data | `data/kaggle_clean.parquet` | `data/kaggle_clean.csv` | Ingestion |
| UCI Cleveland cleaned data | `data/uci_cleveland_clean.parquet` | `data/uci_cleveland_clean.csv` | Ingestion |
| UCI Hungary cleaned data | `data/uci_hungary_clean.parquet` | `data/uci_hungary_clean.csv` | Ingestion |
| UCI Switzerland cleaned data | `data/uci_switzerland_clean.parquet` | `data/uci_switzerland_clean.csv` | Ingestion |
| UCI VA cleaned data | `data/uci_va_clean.parquet` | `data/uci_va_clean.csv` | Ingestion |
| UCI all sites (row-concatenated) | `data/uci_all_sites.parquet` | `data/uci_all_sites.csv` | Ingestion |
| CFS Kaggle ↔ UCI | `data/cfs_kaggle_uci.parquet` | `data/cfs_kaggle_uci.csv` | Ingestion |
| CFS UCI cross-site | `data/cfs_uci_cross_site.parquet` | `data/cfs_uci_cross_site.csv` | Ingestion |

### Internal Validation Outputs (`outputs/internal/`)

| Artifact | Parquet (canonical) | CSV (optional) | When |
|----------|---------------------|-----------------|------|
| Predictions (`y_true`, `y_prob`, `y_pred`, `fold`) | `outputs/internal/{site}/{model_key}/predictions.parquet` | `outputs/internal/{site}/{model_key}/predictions.csv` | Stage 1.3 |

### Internal CFS Outputs (`outputs/internal_cfs/`)

| Artifact | Parquet (canonical) | CSV (optional) | When |
|----------|---------------------|-----------------|------|
| Predictions | `outputs/internal_cfs/{variant}/{site}/{model_key}/predictions.parquet` | `.../predictions.csv` | Stage 1.5 (CFS baselines) | (`outputs/external_uci/`)

| Artifact | Parquet (canonical) | CSV (optional) | When |
|----------|---------------------|-----------------|------|
| Predictions (`y_true`, `y_prob`, `y_pred`) | `outputs/external_uci/{train_sites}__to__{test_site}/{model_key}/predictions.parquet` | `.../predictions.csv` | Stage 1.4 |

### External Validation — Kaggle ↔ UCI (`outputs/external_kaggle_uci/`)

| Artifact | Parquet (canonical) | CSV (optional) | When |
|----------|---------------------|-----------------|------|
| Predictions (`y_true`, `y_prob`, `y_pred`) | `outputs/external_kaggle_uci/{variant}/{train_site}__to__{test_site}/{model_key}/predictions.parquet` | `.../predictions.csv` | Stage 1.5 |

### Shift Diagnostics (`outputs/shift/`)

**Per-pair artifacts** (one directory per external train→test pair; `{pair_key}` = `{train_sites}__to__{test_site}`):

| Artifact | Parquet (canonical) | CSV | When |
|----------|---------------------|-----|------|
| Per-pair feature-level shift (PSI, KS/χ², p-value, missingness) | `outputs/shift/{pair_key}/feature_shift.parquet` | `outputs/shift/{pair_key}/feature_shift.csv` | Stage 1.6 |
| Per-pair summary | `outputs/shift/{pair_key}/shift_diagnostics.json` | — | Stage 1.6 |

**Global / cross-experiment artifacts** (under `outputs/shift/`):

| Artifact | Parquet | CSV | When |
|----------|---------|-----|------|
| Shift metrics (one row per experiment) | `outputs/shift/shift_table.parquet` | `outputs/shift/shift_table.csv` | Stage 1.6 |
| Performance deltas (auc_drop, brier_change) | `outputs/shift/performance_table.parquet` | `outputs/shift/performance_table.csv` | Stage 1.6 |
| Merged shift–performance (link metrics to drops) | `outputs/shift/shift_performance_merged.parquet` | `outputs/shift/shift_performance_merged.csv` | Stage 1.6 |
| Spearman correlation matrix of merged numeric cols | `outputs/shift/shift_performance_correlation.parquet` | `outputs/shift/shift_performance_correlation.csv` | Stage 1.6 |
| Cross-experiment index (pair keys + paths) | `outputs/shift/cross_experiment_index.json` | — | Stage 1.6 |

Paths and filenames are fixed and reproducible; see also `docs/shift_metrics.md` for metric definitions.

---

## Stats & Meta Artifacts

### Ingestion Metadata

| Artifact | Format |
|---|---|
| `data/ingestion_report.json` | JSON — record counts, feature counts, missing rates, prevalence, BP outlier flags per site; CFS lists |

### Per-Experiment Results (JSON)

| Artifact | Format |
|----------|--------|
| `outputs/internal/{site}/{model_key}/results.json` | JSON — metrics, best hyperparameters, bootstrap CIs |
| `outputs/internal_cfs/{variant}/{site}/{model_key}/results.json` | JSON — same as internal (CFS-only baselines) |
| `outputs/external_uci/{train_sites}__to__{test_site}/{model_key}/results.json` | JSON — metrics, best hyperparameters, bootstrap CIs, `features_used`, `n_train`, `n_test` |
| `outputs/external_kaggle_uci/{variant}/{train_site}__to__{test_site}/{model_key}/results.json` | JSON — same as external UCI plus CFS penalty info |

### Trained Models & Pipelines (Joblib)

| Artifact | Format |
|----------|--------|
| `outputs/internal/{site}/{model_key}/model.joblib` | Joblib — fitted sklearn / XGBoost / LightGBM estimator |
| `outputs/internal/{site}/{model_key}/pipeline.joblib` | Joblib — fitted preprocessing pipeline (imputer + encoder ± scaler) |
| `outputs/internal_cfs/{variant}/{site}/{model_key}/model.joblib` | Joblib — fitted estimator |
| `outputs/internal_cfs/{variant}/{site}/{model_key}/pipeline.joblib` | Joblib — fitted preprocessing pipeline when present |
| `outputs/external_uci/{train_sites}__to__{test_site}/{model_key}/model.joblib` | Joblib — fitted estimator |
| `outputs/external_uci/{train_sites}__to__{test_site}/{model_key}/pipeline.joblib` | Joblib — fitted pipeline when present |
| `outputs/external_kaggle_uci/.../model.joblib` | Joblib — fitted estimator |
| `outputs/external_kaggle_uci/.../pipeline.joblib` | Joblib — fitted pipeline when present |

### Shift Diagnostics (JSON and index)

| Artifact | Format |
|---|---|
| `outputs/shift/{pair_key}/shift_diagnostics.json` | JSON — per-pair summary: prevalence_shift, mean_psi, c2st_auc, train_sites, test_site, model, experiment_type |
| `outputs/shift/cross_experiment_index.json` | JSON — list of all pair_key entries with relative paths to shift_diagnostics.json, feature_shift.parquet, feature_shift.csv |

### Calibration & Updating

| Artifact | Format |
|---|---|
| `outputs/calibration/before/{experiment_id}.json` | JSON — pre-recalibration Brier, ECE, MCE, bin details, calibration curve points |
| `outputs/calibration/recalibration/{method}/{experiment_id}.json` | JSON — `metrics_before`, `metrics_after`, `brier_delta`, `ece_delta` per method (`platt`, `isotonic`, `temperature`) |
| `outputs/calibration/updating/{variant}/{experiment_id}.json` | JSON — intercept/slope parameters and before/after calibration metrics per variant (`intercept_only`, `intercept_slope`) |

### Size-Matched Sensitivity

| Artifact | Format |
|----------|--------|
| `outputs/size_matched/{train_sites}__to__{test_site}/{model_key}.json` | JSON — mean ± SD of metrics across K subsampled runs (per model) |

---

## CSV Serialization Notes

- Every Parquet file is accompanied by a `.csv` mirror at the same path (swap `.parquet` → `.csv`).
- Parquet is the **primary** format — typed columns, smaller on disk, faster I/O.
- CSV copies are written for portability and quick inspection; they use UTF-8 encoding and `,` as the delimiter.
- Floating-point columns (`y_prob`, `oldpeak`, `age`, etc.) are serialized with full `float64` precision in CSV to avoid rounding drift.
- The `site` column is stored as a plain string in both formats.
- Boolean/int columns (`target`, `sex`, `fbs`, etc.) are written as `0`/`1` integers in CSV (not `True`/`False`).

---

## Cache Behavior Notes

- **Ingestion outputs are the cache layer.** The pipeline reads cleaned data exclusively from `data/*.parquet`; it never touches `Dataset/` directly. Re-running `src/ingest.py` regenerates the cache.
- **Pipeline stages are idempotent.** Re-running a stage overwrites its `outputs/` subtree. No append logic — a fresh run produces a complete, self-consistent output directory.
- **Models and pipelines are persisted per experiment.** Each `{model_key}/` folder under `outputs/internal/`, `outputs/internal_cfs/`, `outputs/external_uci/`, or `outputs/external_kaggle_uci/` contains the exact fitted estimator and preprocessing pipeline (when present) used to generate that experiment's predictions, enabling post-hoc re-scoring without re-training.
- **No cross-stage dependency on intermediate files.** Stages 1.3–1.5 read from `data/` and write to `outputs/`. Stage 1.6 (shift diagnostics) reads from `data/` directly — not from Stage 1.3–1.5 outputs. Stages 2.1–2.3 read prediction parquets produced by Stages 1.3–1.5.
- **Config hash is logged per experiment** (in each `results.json`) to detect stale outputs when the config changes.
