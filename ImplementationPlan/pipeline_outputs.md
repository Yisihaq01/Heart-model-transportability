# Pipeline Output Artifacts

All files produced by `src/ingest.py` and `src/pipeline.py`. Primary format is Parquet; CSV mirrors are written alongside for portability. Stats and metadata use JSON or Joblib.

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

| Artifact | Parquet | CSV | When |
|---|---|---|---|
| Predictions (`y_true`, `y_prob`, `y_pred`, `fold`) | `outputs/internal/{site}/{model_key}/predictions.parquet` | `outputs/internal/{site}/{model_key}/predictions.csv` | Stage 1.3 |

### External Validation — UCI Multi-Site (`outputs/external_uci/`)

| Artifact | Parquet | CSV | When |
|---|---|---|---|
| Predictions (`y_true`, `y_prob`, `y_pred`) | `outputs/external_uci/{train_sites}__to__{test_site}/{model_key}/predictions.parquet` | `outputs/external_uci/{train_sites}__to__{test_site}/{model_key}/predictions.csv` | Stage 1.4 |

### External Validation — Kaggle ↔ UCI (`outputs/external_kaggle_uci/`)

| Artifact | Parquet | CSV | When |
|---|---|---|---|
| Predictions (`y_true`, `y_prob`, `y_pred`) | `outputs/external_kaggle_uci/{train_site}__to__{test_site}/{model_key}/predictions.parquet` | `outputs/external_kaggle_uci/{train_site}__to__{test_site}/{model_key}/predictions.csv` | Stage 1.5 |

### Shift Diagnostics (`outputs/shift/`)

| Artifact | Parquet | CSV | When |
|---|---|---|---|
| Per-feature PSI / KS statistics | `outputs/shift/{train_sites}__to__{test_site}/feature_shift.parquet` | `outputs/shift/{train_sites}__to__{test_site}/feature_shift.csv` | Stage 1.6 |
| Shift–performance correlation matrix | `outputs/shift/shift_performance_correlation.parquet` | `outputs/shift/shift_performance_correlation.csv` | Stage 1.6 |

---

## Stats & Meta Artifacts

### Ingestion Metadata

| Artifact | Format |
|---|---|
| `data/ingestion_report.json` | JSON — record counts, feature counts, missing rates, prevalence, BP outlier flags per site; CFS lists |

### Per-Experiment Results (JSON)

| Artifact | Format |
|---|---|
| `outputs/internal/{site}/{model_key}/results.json` | JSON — metrics, best hyperparameters, bootstrap CIs |
| `outputs/external_uci/{train_sites}__to__{test_site}/{model_key}/results.json` | JSON — metrics, best hyperparameters, bootstrap CIs, `features_used`, `n_train`, `n_test` |
| `outputs/external_kaggle_uci/{train_site}__to__{test_site}/{model_key}/results.json` | JSON — same as external UCI plus CFS penalty info |

### Trained Models & Pipelines (Joblib)

| Artifact | Format |
|---|---|
| `outputs/internal/{site}/{model_key}/model.joblib` | Joblib — fitted sklearn / XGBoost / LightGBM estimator |
| `outputs/internal/{site}/{model_key}/pipeline.joblib` | Joblib — fitted preprocessing pipeline (imputer + encoder ± scaler) |
| `outputs/external_uci/{train_sites}__to__{test_site}/{model_key}/model.joblib` | Joblib — fitted estimator |

### Shift Diagnostics (JSON)

| Artifact | Format |
|---|---|
| `outputs/shift/{train_sites}__to__{test_site}/shift_diagnostics.json` | JSON — prevalence shift, missingness shift, C2ST AUC |

### Calibration & Updating

| Artifact | Format |
|---|---|
| `outputs/calibration/before/{experiment_id}.json` | JSON — pre-recalibration Brier, ECE, MCE, bin details, calibration curve points |
| `outputs/calibration/recalibration/{method}/{experiment_id}.json` | JSON — `metrics_before`, `metrics_after`, `brier_delta`, `ece_delta` per method (`platt`, `isotonic`, `temperature`) |
| `outputs/calibration/updating/{variant}/{experiment_id}.json` | JSON — intercept/slope parameters and before/after calibration metrics per variant (`intercept_only`, `intercept_slope`) |

### Size-Matched Sensitivity

| Artifact | Format |
|---|---|
| `outputs/size_matched/{train_sites}__to__{test_site}/subsampled_results.json` | JSON — mean ± SD of metrics across K=20 subsampled runs |

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
- **Models and pipelines are persisted per experiment.** Each `{model_key}/` folder under `outputs/internal/` or `outputs/external_uci/` contains the exact fitted estimator and preprocessing pipeline used to generate that experiment's predictions, enabling post-hoc re-scoring without re-training.
- **No cross-stage dependency on intermediate files.** Stages 1.3–1.5 read from `data/` and write to `outputs/`. Stage 1.6 (shift diagnostics) reads from `data/` directly — not from Stage 1.3–1.5 outputs. Stages 2.1–2.3 read prediction parquets produced by Stages 1.3–1.5.
- **Config hash is logged per experiment** (in each `results.json`) to detect stale outputs when the config changes.
