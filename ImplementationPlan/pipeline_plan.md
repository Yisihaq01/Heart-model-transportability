
# Pipeline Plan — ML Transportability / External Validation

**Scope:** Everything between ingested clean data and final evaluation. Preprocessing, model training, internal validation, external validation matrix, calibration, shift diagnostics, and artifact persistence. Final evaluation reporting is **out of scope** (covered separately in `eval_plan.md`).

**Input:** Cleaned per-site parquet files and CFS subsets produced by `src/ingest.py` (see `data_ingestion.md`).

**Output:** Trained models, raw predictions (probabilities + labels), per-experiment metric dicts, calibration artifacts, and shift diagnostic tables — all serialized to `outputs/`.

---

## Phase 1 — Diagnose: Internal & External Validation

### Stage 1.1  Preprocessing

**Goal:** Transform clean DataFrames into model-ready feature matrices. Imputation, encoding, and scaling happen here (deferred from ingestion by design).

#### 1.1.1  Feature-Set Resolution

For every experiment, resolve which feature set applies:

| Experiment Type | Feature Set | Source |
|---|---|---|
| Within-dataset (Kaggle) | All 11 Kaggle features | `data/kaggle_clean.parquet` |
| Within-dataset (UCI site) | All 13 UCI predictors | `data/uci_{site}_clean.parquet` |
| UCI → UCI cross-site | UCI CFS (intersection of non-heavily-missing attrs per site pair) — see caveat below | `data/cfs_uci_cross_site.parquet` + per-pair missingness filter |
| Kaggle ↔ UCI | Minimal CFS: `{age, sex, sys_bp}` + optional binned cholesterol | `data/cfs_kaggle_uci.parquet` |

> **CFS reality check:** The nominal 10-feature `cfs_uci_cross_site` (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`) assumes all features are available. Real missingness (see `data/ingestion_report.json`) shows Hungary has `fbs`/`restecg`/`thal` at 100% missing, VA has `trestbps`/`thalach`/`exang`/`oldpeak` at ~28%. The `effective_cfs()` function (§1.4.2) dynamically narrows the feature set per site pair at the 40% threshold.

```python
# src/preprocessing.py

def resolve_features(train_site: str, test_site: str, config: dict) -> list[str]:
    """Return the feature list for a given train/test site pair."""
    if train_site == test_site:
        return config["full_features"][train_site]
    pair_key = tuple(sorted([train_site, test_site]))
    return config["cfs_features"].get(pair_key, config["cfs_features"]["default"])
```

#### 1.1.2  Imputation

Imputation strategy varies by model family. Applied **after** train/test split to prevent leakage.

| Model Family | Imputation Strategy |
|---|---|
| Logistic Regression | Median imputation (continuous), mode imputation (categorical) — fit on train, transform test |
| Random Forest | Median/mode imputation (same as LR) — trees in sklearn don't natively handle NaN |
| XGBoost / LightGBM | No imputation — native NaN handling; pass raw missing values |

```python
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_imputer(model_family: str, continuous_cols: list, categorical_cols: list):
    if model_family in ("xgboost", "lightgbm"):
        return "passthrough"
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), continuous_cols),
        ("cat", SimpleImputer(strategy="most_frequent"), categorical_cols),
    ])
```

#### 1.1.3  Encoding

| Feature Type | Encoding | Notes |
|---|---|---|
| Binary (`sex`, `fbs`, `exang`, `smoke`, `alco`, `active`) | Passthrough (already 0/1) | — |
| Ordinal (`cholesterol`, `gluc`, `restecg`, `slope`) | Ordinal encoding (preserve order) | `cp` is nominal — see below |
| Nominal (`cp`: chest pain type, `thal`: thalassemia) | One-hot encoding (drop first) | Avoid dummy variable trap for LR |
| Continuous (`age`, `sys_bp`, `chol`, `thalach`, `oldpeak`, etc.) | Passthrough (scaling handled separately) | — |

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

NOMINAL_COLS = ["cp", "thal"]
ORDINAL_COLS = ["cholesterol", "gluc", "restecg", "slope"]

def build_encoder(feature_list: list):
    nominal = [c for c in NOMINAL_COLS if c in feature_list]
    ordinal = [c for c in ORDINAL_COLS if c in feature_list]
    passthrough = [c for c in feature_list if c not in nominal + ordinal]

    return ColumnTransformer([
        ("nominal", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="infrequent_if_exist"), nominal),
        ("ordinal", OrdinalEncoder(), ordinal),
        ("pass", "passthrough", passthrough),
    ])
```

#### 1.1.4  Scaling

| Model Family | Scaling |
|---|---|
| Logistic Regression | `StandardScaler` (fit on train) |
| Random Forest | None (tree-based, scale-invariant) |
| XGBoost / LightGBM | None (tree-based) |

#### 1.1.5  Full Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_preprocessing_pipeline(model_family: str, feature_list: list,
                                  continuous_cols: list, categorical_cols: list) -> Pipeline:
    steps = []
    steps.append(("imputer", build_imputer(model_family, continuous_cols, categorical_cols)))
    steps.append(("encoder", build_encoder(feature_list)))
    if model_family == "logistic_regression":
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)
```

**Key constraint:** All preprocessing transformers are fit on the **training set only** and applied (transform only) to the test set. This holds for both internal and external validation.

---

### Stage 1.2  Model Definitions

**Goal:** Define the model families and their hyperparameter grids. Keep modeling choices fixed across all experiments so the "treatment" is the validation setting, not the model.

#### 1.2.1  Model Registry

| Model Key | Class | Default Hyperparameters | Tuning Grid |
|---|---|---|---|
| `lr` | `LogisticRegression` | `penalty="l2"`, `solver="lbfgs"`, `max_iter=1000` | `C`: [0.001, 0.01, 0.1, 1, 10] |
| `rf` | `RandomForestClassifier` | `n_estimators=300`, `class_weight="balanced"` | `max_depth`: [5, 10, 20, None], `min_samples_leaf`: [1, 5, 10] |
| `xgb` | `XGBClassifier` | `n_estimators=300`, `eval_metric="logloss"` | `max_depth`: [3, 5, 7], `learning_rate`: [0.01, 0.05, 0.1], `subsample`: [0.7, 0.8, 1.0] |
| `lgbm` | `LGBMClassifier` | `n_estimators=300`, `verbose=-1` | `max_depth`: [3, 5, 7], `learning_rate`: [0.01, 0.05, 0.1], `num_leaves`: [15, 31, 63] |

```python
# src/models.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

MODEL_REGISTRY = {
    "lr": {
        "class": LogisticRegression,
        "default_params": {"penalty": "l2", "solver": "lbfgs", "max_iter": 1000, "random_state": 42},
        "tuning_grid": {"C": [0.001, 0.01, 0.1, 1, 10]},
    },
    "rf": {
        "class": RandomForestClassifier,
        "default_params": {"n_estimators": 300, "class_weight": "balanced", "random_state": 42, "n_jobs": -1},
        "tuning_grid": {"max_depth": [5, 10, 20, None], "min_samples_leaf": [1, 5, 10]},
    },
    "xgb": {
        "class": XGBClassifier,
        "default_params": {"n_estimators": 300, "eval_metric": "logloss", "random_state": 42, "n_jobs": -1, "verbosity": 0},
        "tuning_grid": {"max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1], "subsample": [0.7, 0.8, 1.0]},
    },
    "lgbm": {
        "class": LGBMClassifier,
        "default_params": {"n_estimators": 300, "verbose": -1, "random_state": 42, "n_jobs": -1},
        "tuning_grid": {"max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1], "num_leaves": [15, 31, 63]},
    },
}

def get_model(model_key: str, **override_params):
    entry = MODEL_REGISTRY[model_key]
    params = {**entry["default_params"], **override_params}
    return entry["class"](**params)
```

#### 1.2.2  Hyperparameter Tuning Protocol

- **Method:** Inner 5-fold CV on the training set via `GridSearchCV` (or `RandomizedSearchCV` for XGB/LGBM if grid is large).
- **Scoring:** `roc_auc` as primary selection metric.
- **Small cohort guard:** For UCI sites with N < 200, use `RepeatedStratifiedKFold(n_splits=5, n_repeats=3)` to stabilize.
- **Output:** Best parameters per model per experiment are logged in the experiment artifact.

```python
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

def tune_model(model_key: str, X_train, y_train, n_samples: int):
    entry = MODEL_REGISTRY[model_key]
    base_model = entry["class"](**entry["default_params"])

    cv = (RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
          if n_samples < 200
          else StratifiedKFold(n_splits=5, shuffle=True, random_state=42))

    search = GridSearchCV(
        base_model, entry["tuning_grid"],
        scoring="roc_auc", cv=cv, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.cv_results_
```

---

### Stage 1.3  Internal Validation (RQ1)

**Goal:** Establish within-site baselines — discrimination and calibration under standard 80/20 stratified split. Repeated for stability on small cohorts.

#### 1.3.1  Splitting Strategy

| Site | N (approx) | Split Method | Rationale |
|---|---|---|---|
| Kaggle | 70,000 | Single 80/20 stratified split | Large N → single split is stable |
| Cleveland | 303 | 80/20 split + bootstrap (B=200) | Moderate N → bootstrap CIs for metric stability |
| Hungary | 294 | 80/20 split + bootstrap (B=500) | Small test N (<200 after split) → more bootstrap iters |
| VA | 200 | 80/20 split + bootstrap (B=500) | Small test N (<200 after split) → more bootstrap iters |
| Switzerland | 123 | 5×5 Repeated Stratified CV + bootstrap (B=500) | Very small N → full CV; B=500 per fold |

> **Bootstrap B selection:** B is chosen dynamically per fold — B=200 when the held-out test set has ≥200 samples, B=500 otherwise (matching `bootstrap_iters_small` in `configs/pipeline.yaml`).

```python
# src/validation.py

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import numpy as np

def internal_split(df, site: str, seed: int = 42):
    """Return (train_idx, test_idx) or list of (train_idx, test_idx) for CV."""
    if site == "switzerland":
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        return list(cv.split(df.drop(columns=["target"]), df["target"]))

    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, stratify=df["target"], random_state=seed
    )
    return [(train_idx, test_idx)]
```

#### 1.3.2  Bootstrap Confidence Intervals

For each metric, compute 95% bootstrap CIs on the held-out predictions:

```python
def bootstrap_metric(y_true, y_pred, metric_fn, B=200, seed=42):
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return {"point": metric_fn(y_true, y_pred), "ci_lower": lower, "ci_upper": upper}
```

#### 1.3.3  Internal Validation Loop

For every site × model combination:

> **Feature filtering:** The nominal "full-feature set" from `resolve_features(site, site, config)` is further restricted to columns that (a) exist in the site's clean parquet and (b) have at least one non-NaN value. This silently drops attributes that are 100% missing at a given site (e.g. `thal` in Hungary/VA, `chol` in Switzerland) so that imputers and encoders never receive a wholly-empty column. The effective per-site feature list is logged in each experiment's `results.json`.

```
for site in [kaggle, cleveland, hungary, va, switzerland]:
    df = load_clean(site)
    features = resolve_features(site, site, config)  # full-feature set
    features = [c for c in features if c in df.columns and df[c].notna().any()]
    splits = internal_split(df, site)

    for model_key in [lr, rf, xgb, lgbm]:
        for train_idx, test_idx in splits:
            X_train, y_train = df.loc[train_idx, features], df.loc[train_idx, "target"]
            X_test,  y_test  = df.loc[test_idx,  features], df.loc[test_idx,  "target"]

            pipeline = build_preprocessing_pipeline(model_key, features, ...)
            X_train_t = pipeline.fit_transform(X_train)
            X_test_t  = pipeline.transform(X_test)

            model, best_params, _ = tune_model(model_key, X_train_t, y_train, len(X_train_t))

            y_prob = model.predict_proba(X_test_t)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            metrics = compute_metrics(y_test, y_prob, y_pred)  # ROC-AUC, PR-AUC, Brier, ECE, etc.
            boot_cis = bootstrap_metrics(y_test, y_prob, y_pred)

            save_experiment({
                "experiment_type": "internal",
                "site": site,
                "model": model_key,
                "best_params": best_params,
                "metrics": metrics,
                "bootstrap_cis": boot_cis,
                "predictions": {"y_true": y_test, "y_prob": y_prob},
            })
```

#### 1.3.4  Outputs

| Artifact | Path | Format |
|----------|------|--------|
| Per-experiment results | `outputs/internal/{site}/{model_key}/results.json` | JSON (metrics, params, CIs) |
| Predictions | `outputs/internal/{site}/{model_key}/predictions.parquet` | Parquet (canonical); `predictions.csv` optional |
| Trained model | `outputs/internal/{site}/{model_key}/model.joblib` | Joblib |
| Preprocessing pipeline | `outputs/internal/{site}/{model_key}/pipeline.joblib` | Joblib (when present) |

---

### Stage 1.4  External Validation — UCI Multi-Site Matrix (RQ2)

**Goal:** Train on one or more UCI sites, test on a held-out UCI site. Measure how discrimination and calibration degrade when the test population differs from training.

#### 1.4.1  Validation Matrix

All pairwise and leave-one-site-out combinations:

**Pairwise (train → test):**

| Train Site | Test Site |
|---|---|
| Cleveland | Hungary |
| Cleveland | VA |
| Cleveland | Switzerland |
| Hungary | Cleveland |
| Hungary | VA |
| Hungary | Switzerland |
| VA | Cleveland |
| VA | Hungary |
| VA | Switzerland |
| Switzerland | Cleveland |
| Switzerland | Hungary |
| Switzerland | VA |

**Leave-one-site-out (LOSO):**

| Train Sites | Test Site |
|---|---|
| Hungary + VA + Switzerland | Cleveland |
| Cleveland + VA + Switzerland | Hungary |
| Cleveland + Hungary + Switzerland | VA |
| Cleveland + Hungary + VA | Switzerland |

**Pooled train → individual test:**

| Train Sites | Test Site |
|---|---|
| All 3 remaining UCI sites | Each site in turn (same as LOSO) |

Total: **12 pairwise + 4 LOSO = 16 external experiments** per model.

#### 1.4.2  Feature-Set Handling for Cross-Site

For each experiment pair, the effective feature set is the intersection of non-heavily-missing columns between train and test sites. Concretely:

```python
MISSINGNESS_THRESHOLD = 0.40  # drop feature from pair if >40% missing in either site

def effective_cfs(train_df, test_df, candidate_features: list) -> list[str]:
    usable = []
    for col in candidate_features:
        train_miss = train_df[col].isna().mean()
        test_miss = test_df[col].isna().mean()
        if train_miss < MISSINGNESS_THRESHOLD and test_miss < MISSINGNESS_THRESHOLD:
            usable.append(col)
    return usable
```

#### 1.4.3  External Validation Loop

```
for (train_sites, test_site) in validation_matrix:
    train_df = concat([load_clean(s) for s in train_sites])
    test_df  = load_clean(test_site)

    features = effective_cfs(train_df, test_df, config["uci_14_features"])

    for model_key in [lr, rf, xgb, lgbm]:
        pipeline = build_preprocessing_pipeline(model_key, features, ...)
        X_train_t = pipeline.fit_transform(train_df[features])
        X_test_t  = pipeline.transform(test_df[features])
        y_train   = train_df["target"]
        y_test    = test_df["target"]

        model, best_params, _ = tune_model(model_key, X_train_t, y_train, len(X_train_t))

        y_prob = model.predict_proba(X_test_t)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_metrics(y_test, y_prob, y_pred)
        boot_cis = bootstrap_metrics(y_test, y_prob, y_pred, B=500)  # more bootstrap iters for small N

        save_experiment({
            "experiment_type": "external_uci",
            "train_sites": train_sites,
            "test_site": test_site,
            "model": model_key,
            "features_used": features,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "best_params": best_params,
            "metrics": metrics,
            "bootstrap_cis": boot_cis,
            "predictions": {"y_true": y_test, "y_prob": y_prob},
        })
```

#### 1.4.4  Outputs

| Artifact | Path |
|----------|------|
| Per-experiment results | `outputs/external_uci/{train_sites}__to__{test_site}/{model_key}/results.json` |
| Predictions | `outputs/external_uci/.../predictions.parquet` (canonical), `predictions.csv` (optional) |
| Trained model | `outputs/external_uci/.../model.joblib` |
| Preprocessing pipeline | `outputs/external_uci/.../pipeline.joblib` (when present) |

---

### Stage 1.5  External Validation — Kaggle ↔ UCI Stress Test (RQ5)

**Goal:** Quantify the performance cost of restricting to the Common Feature Set (CFS) when bridging Kaggle and UCI, and test transportability across fundamentally different cohorts.

#### 1.5.1  Experiments

| Experiment | Train | Test | Feature Set |
|---|---|---|---|
| Kaggle → UCI (each site) | Kaggle CFS | UCI site CFS | `{age, sex, sys_bp}` ± binned cholesterol |
| UCI (each site) → Kaggle | UCI site CFS | Kaggle CFS | Same |
| UCI pooled → Kaggle | All 4 UCI CFS | Kaggle CFS | Same |
| Kaggle → UCI pooled | Kaggle CFS | All 4 UCI CFS | Same |

For each: also run an **internal baseline** on both train and test sites using the same CFS to quantify the CFS performance penalty vs. full features.

#### 1.5.2  Cholesterol Binning (CFS Extension)

When including cholesterol in the CFS, bin UCI continuous `chol` to match Kaggle ordinal:

```python
def bin_cholesterol_uci(chol_series: pd.Series) -> pd.Series:
    """Bin mg/dl → ordinal {1, 2, 3} per ATP III clinical cutpoints.
    [−∞, 200) → 1 (desirable), [200, 240) → 2 (borderline), [240, ∞) → 3 (high).
    Uses right=False so boundary values 200 and 240 fall into the higher bin."""
    return pd.cut(
        chol_series,
        bins=[-np.inf, 200, 240, np.inf],
        labels=[1, 2, 3],
        right=False,
    ).astype(float)
```

Cutpoints from ATP III guidelines (< 200 desirable, 200–239 borderline, ≥ 240 high). `right=False` ensures 200 mg/dl → borderline (2), not desirable (1).

#### 1.5.3  CFS Penalty Measurement

For each site, compare:

- **Full-feature internal AUC** (from Stage 1.3)
- **CFS-only internal AUC** (same split, CFS features only)

```python
def cfs_penalty(full_auc: float, cfs_auc: float) -> dict:
    return {
        "full_auc": full_auc,
        "cfs_auc": cfs_auc,
        "auc_drop": full_auc - cfs_auc,
        "relative_drop_pct": ((full_auc - cfs_auc) / full_auc * 100) if full_auc > 0 else None,
    }
```

#### 1.5.4  Outputs

| Artifact | Path |
|----------|------|
| Per-experiment results | `outputs/external_kaggle_uci/{variant}/{train_site}__to__{test_site}/{model_key}/results.json` |
| Predictions | `.../predictions.parquet` (canonical), `.../predictions.csv` (optional) |
| Trained model | `.../model.joblib` |
| Preprocessing pipeline | `.../pipeline.joblib` (when present) |

Internal CFS baselines (Stage 1.5) write to `outputs/internal_cfs/{variant}/{site}/{model_key}/` with the same artifact set.

---

### Stage 1.6  Dataset Shift Diagnostics (RQ3)

**Goal:** For every external validation pair, quantify what shifted and correlate with performance drops.

#### 1.6.1  Prevalence Shift

```python
def prevalence_shift(train_df, test_df) -> dict:
    p_train = train_df["target"].mean()
    p_test  = test_df["target"].mean()
    return {
        "train_prevalence": p_train,
        "test_prevalence": p_test,
        "absolute_diff": abs(p_train - p_test),
    }
```

#### 1.6.2  Covariate Shift — Univariate Feature Distributions

For each feature, compare train vs test distributions:

| Feature Type | Test | Metric |
|---|---|---|
| Continuous | Kolmogorov-Smirnov test | KS statistic + p-value |
| Categorical / binary | Chi-squared test | Chi² statistic + p-value |
| Any | Population Stability Index (PSI) | PSI value (> 0.1 = moderate shift, > 0.25 = severe) |

```python
from scipy.stats import ks_2samp, chi2_contingency

def univariate_shift(train_col, test_col, col_type: str) -> dict:
    if col_type == "continuous":
        stat, p = ks_2samp(train_col.dropna(), test_col.dropna())
        return {"test": "ks", "statistic": stat, "p_value": p}
    else:
        contingency = pd.crosstab(
            pd.concat([train_col, test_col]),
            pd.Series(["train"]*len(train_col) + ["test"]*len(test_col))
        )
        stat, p, _, _ = chi2_contingency(contingency)
        return {"test": "chi2", "statistic": stat, "p_value": p}

def psi(train_col, test_col, bins=10) -> float:
    """Population Stability Index."""
    breakpoints = np.percentile(train_col.dropna(), np.linspace(0, 100, bins + 1))
    train_pct = np.histogram(train_col.dropna(), bins=breakpoints)[0] / len(train_col.dropna())
    test_pct  = np.histogram(test_col.dropna(), bins=breakpoints)[0] / len(test_col.dropna())
    train_pct = np.clip(train_pct, 1e-6, None)
    test_pct  = np.clip(test_pct, 1e-6, None)
    return float(np.sum((test_pct - train_pct) * np.log(test_pct / train_pct)))
```

#### 1.6.3  Missingness Shift

Compare per-feature missingness rates between train and test sites:

```python
def missingness_shift(train_df, test_df, features: list) -> pd.DataFrame:
    rows = []
    for col in features:
        train_miss = train_df[col].isna().mean()
        test_miss  = test_df[col].isna().mean()
        rows.append({
            "feature": col,
            "train_miss_pct": round(train_miss * 100, 1),
            "test_miss_pct": round(test_miss * 100, 1),
            "diff_pct": round((test_miss - train_miss) * 100, 1),
        })
    return pd.DataFrame(rows)
```

#### 1.6.4  Multivariate Shift — Classifier Two-Sample Test (C2ST)

Train a classifier to distinguish train from test samples. AUC ≈ 0.5 means no detectable shift; AUC >> 0.5 means strong distributional difference.

```python
def classifier_two_sample_test(X_train, X_test, seed=42) -> float:
    """Returns AUC of a classifier trying to distinguish train from test."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    X = np.vstack([X_train, X_test])
    y = np.array([0]*len(X_train) + [1]*len(X_test))

    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    return float(scores.mean())
```

#### 1.6.5  Shift–Performance Correlation

After all external experiments and shift diagnostics are computed, correlate:

- Per-feature PSI / KS statistics with per-experiment AUC drop (external AUC − internal baseline AUC)
- Prevalence shift with Brier score change
- C2ST AUC with overall external AUC

```python
def shift_performance_correlation(shift_table: pd.DataFrame, perf_table: pd.DataFrame) -> pd.DataFrame:
    """Merge shift metrics with performance deltas and compute rank correlations."""
    merged = shift_table.merge(perf_table, on=["train_sites", "test_site"])
    corr = merged[["mean_psi", "prevalence_diff", "c2st_auc", "auc_drop", "brier_change"]].corr(method="spearman")
    return corr
```

#### 1.6.6  Outputs

| Artifact | Path |
|---|---|
| Per-pair shift table | `outputs/shift/{train_sites}__to__{test_site}/shift_diagnostics.json` |
| Feature-level PSI/KS | `outputs/shift/{train_sites}__to__{test_site}/feature_shift.parquet` |
| C2ST AUC | Included in `shift_diagnostics.json` |
| Correlation matrix | `outputs/shift/shift_performance_correlation.parquet` |

---

## Phase 2 — Mitigate: Calibration & Lightweight Updating

### Stage 2.1  Calibration Assessment (Pre-Recalibration)

**Goal:** Measure how well-calibrated the predicted probabilities are, before any recalibration. This is the "before" snapshot.

#### 2.1.1  Calibration Metrics

Computed on every experiment's held-out predictions from Phase 1:

| Metric | Implementation |
|---|---|
| **Brier score** | `sklearn.metrics.brier_score_loss` |
| **Expected Calibration Error (ECE)** | Custom: bin predictions into M equal-width bins, compute weighted mean absolute gap between bin accuracy and mean predicted probability |
| **Maximum Calibration Error (MCE)** | Max absolute gap across bins |
| **Calibration curve** | `sklearn.calibration.calibration_curve` (10 bins, strategy=`uniform` and `quantile`) |

```python
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

def compute_calibration_metrics(y_true, y_prob, n_bins=10) -> dict:
    brier = brier_score_loss(y_true, y_prob)

    # ECE and MCE
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    bin_details = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        bin_size = mask.sum()
        gap = abs(bin_acc - bin_conf)
        ece += gap * bin_size / len(y_true)
        mce = max(mce, gap)
        bin_details.append({"lo": lo, "hi": hi, "acc": bin_acc, "conf": bin_conf, "n": int(bin_size)})

    # Calibration curve points for plotting
    fraction_pos, mean_predicted = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")

    return {
        "brier_score": brier,
        "ece": ece,
        "mce": mce,
        "bin_details": bin_details,
        "calibration_curve": {"fraction_pos": fraction_pos.tolist(), "mean_predicted": mean_predicted.tolist()},
    }
```

#### 2.1.2  When to Flag

| Condition | Interpretation |
|---|---|
| ECE > 0.05 | Moderate miscalibration |
| ECE > 0.10 | Severe miscalibration |
| MCE > 0.15 | At least one probability bin is badly off |
| Brier external >> Brier internal | Calibration specifically degraded under shift |

---

### Stage 2.2  Post-Hoc Recalibration (RQ4)

**Goal:** Apply recalibration methods to the raw predicted probabilities from external validation experiments and measure improvement.

#### 2.2.1  Recalibration Methods

| Method | Approach | When it works best |
|---|---|---|
| **Platt scaling** | Fit a logistic regression on (predicted_prob → true_label) using a calibration set | Smooth, monotonic miscalibration; sufficient calibration data |
| **Isotonic regression** | Fit a non-parametric isotonic regression on (predicted_prob → true_label) | Non-monotonic miscalibration; more calibration data needed |
| **Temperature scaling** | Fit a single scalar T: `calibrated_prob = sigmoid(logit(raw_prob) / T)` | Global over/under-confidence |

#### 2.2.2  Calibration Data Split

Recalibration requires a held-out calibration set **from the target site** to fit the recalibrator. Two options depending on available data:

| Scenario | Calibration Source | Evaluation Source |
|---|---|---|
| **Standard** (if target site has enough data) | 50% of target test set → calibration, 50% → evaluation | Remaining 50% |
| **Small target site** (N < 100 in test) | 3-fold CV on target test set: fit recalibrator on 2 folds, evaluate on 1, rotate | Aggregated across folds |

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as PlattLR

def recalibrate_platt(y_prob_raw, y_true_cal):
    """Fit Platt scaling on calibration set, return fitted model."""
    lr = PlattLR()
    lr.fit(y_prob_raw.reshape(-1, 1), y_true_cal)
    return lr

def recalibrate_isotonic(y_prob_raw, y_true_cal):
    """Fit isotonic regression on calibration set."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob_raw, y_true_cal)
    return iso

def recalibrate_temperature(y_prob_raw, y_true_cal):
    """Fit temperature scaling (single parameter)."""
    from scipy.optimize import minimize_scalar
    from scipy.special import logit, expit

    logits = logit(np.clip(y_prob_raw, 1e-6, 1 - 1e-6))

    def nll(T):
        scaled = expit(logits / T)
        return -np.mean(y_true_cal * np.log(scaled + 1e-10) + (1 - y_true_cal) * np.log(1 - scaled + 1e-10))

    result = minimize_scalar(nll, bounds=(0.1, 10), method="bounded")
    return result.x  # optimal temperature
```

#### 2.2.3  Recalibration Loop

```
for experiment in all_external_experiments:
    y_prob_raw = load_predictions(experiment)["y_prob"]
    y_true     = load_predictions(experiment)["y_true"]

    # Split: calibration vs evaluation
    cal_idx, eval_idx = calibration_split(y_true, experiment["test_site"])

    for method in ["platt", "isotonic", "temperature"]:
        recalibrator = fit_recalibrator(method, y_prob_raw[cal_idx], y_true[cal_idx])
        y_prob_recal = apply_recalibrator(recalibrator, method, y_prob_raw[eval_idx])

        metrics_before = compute_calibration_metrics(y_true[eval_idx], y_prob_raw[eval_idx])
        metrics_after  = compute_calibration_metrics(y_true[eval_idx], y_prob_recal)

        save_recalibration_result({
            "experiment": experiment["id"],
            "method": method,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "improvement": {
                "brier_delta": metrics_before["brier_score"] - metrics_after["brier_score"],
                "ece_delta": metrics_before["ece"] - metrics_after["ece"],
            },
        })
```

---

### Stage 2.3  Lightweight Updating — Intercept & Slope Recalibration (RQ4)

**Goal:** Test whether simple logistic recalibration (adjusting intercept and slope of the log-odds) can fix prevalence and calibration slope mismatch without retraining.

#### 2.3.1  Method

Logistic recalibration fits: `logit(p_updated) = a + b * logit(p_original)`

- `a` adjusts for prevalence shift (intercept-only recalibration = adjusting for different base rates)
- `b` adjusts the spread/sharpness of probabilities

Two variants:
1. **Intercept-only** (`b = 1` fixed): corrects prevalence mismatch only
2. **Intercept + slope** (both `a` and `b` free): corrects both prevalence and calibration slope

```python
def logistic_recalibration(y_prob_raw, y_true_cal, intercept_only=False):
    """Fit logistic recalibration model on calibration data."""
    logits = logit(np.clip(y_prob_raw, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    lr = PlattLR(penalty=None, solver="lbfgs")

    if intercept_only:
        lr.fit(np.ones_like(logits), y_true_cal)  # intercept only
        return {"intercept": lr.intercept_[0], "slope": 1.0}
    else:
        lr.fit(logits, y_true_cal)
        return {"intercept": lr.intercept_[0], "slope": lr.coef_[0][0]}
```

#### 2.3.2  Intercept-Only vs Full: Diagnostic Value

| Outcome | Interpretation |
|---|---|
| Intercept-only fixes calibration | Problem was pure prevalence shift |
| Intercept+slope needed | Both prevalence and probability spread differ |
| Neither helps | Feature-level mismatch blocks transfer; recalibration is insufficient |

This directly informs the "recalibration-enough vs. feature-mismatch-blocked" analysis from the README deliverables.

---

### Stage 2.4  Size-Matched Sensitivity Analysis

**Goal:** Deconfound dataset size from population shift. Large training sets (e.g., Kaggle 70K) vs small UCI cohorts could inflate apparent transportability differences.

#### 2.4.1  Subsampling Protocol

For experiments where train N >> test N:

1. Subsample the training set to match the test set size (±10%)
2. Repeat K=20 times with different random subsamples
3. Report mean ± SD of external metrics across subsamples
4. Compare against full-training-set metrics

```python
def size_matched_experiment(train_df, test_df, model_key, features, K=20, seed=42):
    target_n = len(test_df)
    results = []
    rng = np.random.RandomState(seed)

    for k in range(K):
        sub_idx = rng.choice(len(train_df), size=target_n, replace=False)
        sub_train = train_df.iloc[sub_idx]

        pipeline = build_preprocessing_pipeline(model_key, features, ...)
        X_train_t = pipeline.fit_transform(sub_train[features])
        X_test_t  = pipeline.transform(test_df[features])

        model = get_model(model_key)
        model.fit(X_train_t, sub_train["target"])

        y_prob = model.predict_proba(X_test_t)[:, 1]
        metrics = compute_metrics(test_df["target"], y_prob, (y_prob >= 0.5).astype(int))
        results.append(metrics)

    return aggregate_results(results)  # mean, std per metric
```

#### 2.4.2  Applicability

Primarily relevant for:
- Kaggle (70K) → UCI site (123–303) experiments
- LOSO where pooled training set is ~700+ vs single-site test of ~123–303

---

## Cross-Cutting Concerns

### Stage C.1  Experiment Configuration

All experiments are driven by a YAML config:

```yaml
# configs/pipeline.yaml

random_seed: 42
bootstrap_iters: 200
bootstrap_iters_small: 500  # for N < 200

models: [lr, rf, xgb, lgbm]

internal_validation:
  sites: [kaggle, cleveland, hungary, va, switzerland]
  split: 0.8  # train fraction

external_validation:
  uci_pairwise: true
  uci_loso: true
  kaggle_uci: true

missingness_threshold: 0.40

calibration:
  methods: [platt, isotonic, temperature]
  n_bins: 10

shift_diagnostics:
  psi_bins: 10
  psi_severe_threshold: 0.25
  psi_moderate_threshold: 0.10
  c2st: true

size_matched:
  enabled: true
  K: 20
```

### Stage C.2  Metric Computation Module

Centralized metric computation used by all stages:

```python
# src/metrics.py

from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)

def compute_metrics(y_true, y_prob, y_pred) -> dict:
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_test": len(y_true),
        "prevalence": float(y_true.mean()),
    }
```

### Stage C.3  Artifact Serialization

Every experiment (internal, internal_cfs, external_uci, external_kaggle_uci) persists the canonical set via `save_experiment()`:

- **results.json** — all keys except predictions, fitted_model, fitted_pipeline; C.4 timestamp and optional config_hash
- **predictions.parquet** (canonical) and **predictions.csv** (optional) when predictions present
- **model.joblib** and **pipeline.joblib** when fitted_model / fitted_pipeline provided

See `ImplementationPlan/pipeline_outputs.md` for the full artifact contract table.

```python
# src/artifacts.py

import json, joblib
from pathlib import Path
import pandas as pd

def save_experiment(result: dict, base_dir: str = "outputs", config_hash: str | None = None):
    exp_type = result["experiment_type"]
    if exp_type == "internal":
        path = Path(base_dir) / "internal" / result["site"] / result["model"]
    elif exp_type == "internal_cfs":
        path = Path(base_dir) / "internal_cfs" / (result.get("variant") or "cfs") / result["site"] / result["model"]
    elif exp_type == "external_uci":
        train_key = "+".join(sorted(result["train_sites"])) if isinstance(result["train_sites"], list) else result["train_sites"]
        path = Path(base_dir) / "external_uci" / f"{train_key}__to__{result['test_site']}" / result["model"]
    elif exp_type == "external_kaggle_uci":
        path = Path(base_dir) / "external_kaggle_uci" / (result.get("variant") or "cfs") / f"{result['train_site']}__to__{result['test_site']}" / result["model"]
    else:
        path = Path(base_dir) / exp_type / result.get("id", "default")

    path.mkdir(parents=True, exist_ok=True)

    payload = {k: v for k, v in result.items() if k not in ("predictions", "fitted_model", "fitted_pipeline")}
    # C.4: timestamp; config_hash when provided
    payload["timestamp"] = ...  # reproducibility.experiment_timestamp()
    if config_hash is not None:
        payload["config_hash"] = config_hash
    with open(path / "results.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    if "predictions" in result:
        df = pd.DataFrame(result["predictions"])
        df.to_parquet(path / "predictions.parquet", index=False)
        df.to_csv(path / "predictions.csv", index=False, float_format="%.10g")

    if exp_type in ("internal", "internal_cfs", "external_uci", "external_kaggle_uci"):
        if "fitted_model" in result:
            joblib.dump(result["fitted_model"], path / "model.joblib")
        if "fitted_pipeline" in result:
            joblib.dump(result["fitted_pipeline"], path / "pipeline.joblib")

    return path
```

### Stage C.4  Reproducibility Controls

| Control | Implementation |
|---|---|
| Global random seed | Set via config, propagated to all splits, models, bootstrap |
| Deterministic splits | `random_state` in all sklearn splitters |
| Model determinism | `random_state` in all model constructors |
| Environment | Pin all package versions in `requirements.txt` |
| Logging | Every experiment logs: config hash, feature list, train/test N, timestamp |

---

## Pipeline Orchestration

### Execution DAG

```
                    ┌──────────────────────┐
                    │   Ingested Data       │
                    │   (from ingest.py)    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Stage C.1: Load      │
                    │  Config               │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                 │
   ┌──────────▼──────┐ ┌──────▼───────┐ ┌──────▼───────┐
   │ Stage 1.3       │ │ Stage 1.4    │ │ Stage 1.5    │
   │ Internal Valid.  │ │ External UCI │ │ Kaggle↔UCI   │
   │ (all sites)     │ │ Matrix       │ │ Stress Test  │
   └────────┬────────┘ └──────┬───────┘ └──────┬───────┘
            │                  │                 │
            │    ┌─────────────┼─────────────┐   │
            │    │             │             │   │
            │    │  ┌──────────▼──────────┐  │   │
            │    │  │ Stage 1.6           │  │   │
            │    │  │ Shift Diagnostics   │  │   │
            │    │  └──────────┬──────────┘  │   │
            │    │             │             │   │
            ▼    ▼             ▼             ▼   ▼
   ┌─────────────────────────────────────────────────┐
   │  Stage 2.1: Calibration Assessment (Before)     │
   └──────────────────────┬──────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
   ┌──────▼──────┐ ┌─────▼──────┐ ┌──────▼──────┐
   │ Stage 2.2   │ │ Stage 2.3  │ │ Stage 2.4   │
   │ Post-Hoc    │ │ Lightweight│ │ Size-Matched│
   │ Recalib.    │ │ Updating   │ │ Sensitivity │
   └──────┬──────┘ └─────┬──────┘ └──────┬──────┘
          │               │               │
          ▼               ▼               ▼
   ┌─────────────────────────────────────────────────┐
   │  All artifacts written to outputs/               │
   │  → Ready for eval_plan.md (final reporting)      │
   └─────────────────────────────────────────────────┘
```

### Entrypoint

```python
# src/pipeline.py

def run_pipeline(config_path: str = "configs/pipeline.yaml"):
    cfg = load_config(config_path)
    data = load_all_clean_data(cfg)

    # Phase 1 — Diagnose
    internal_results = run_internal_validation(data, cfg)        # Stage 1.3
    external_uci_results = run_external_uci_matrix(data, cfg)    # Stage 1.4
    external_kaggle_results = run_kaggle_uci_tests(data, cfg)    # Stage 1.5
    shift_results = run_shift_diagnostics(data, cfg,             # Stage 1.6
                                          external_uci_results,
                                          external_kaggle_results)

    # Phase 2 — Mitigate
    all_external = {**external_uci_results, **external_kaggle_results}
    calibration_before = assess_calibration(all_external)                 # Stage 2.1
    recalibration = run_recalibration(all_external, data, cfg)            # Stage 2.2
    updating = run_lightweight_updating(all_external, data, cfg)          # Stage 2.3
    size_matched = run_size_matched_sensitivity(data, cfg)                # Stage 2.4

    # Persist everything
    save_all_artifacts(internal_results, external_uci_results,
                       external_kaggle_results, shift_results,
                       calibration_before, recalibration, updating,
                       size_matched)

    return {
        "internal": internal_results,
        "external_uci": external_uci_results,
        "external_kaggle": external_kaggle_results,
        "shift": shift_results,
        "calibration": {"before": calibration_before, "recalibration": recalibration, "updating": updating},
        "size_matched": size_matched,
    }
```

---

## Source Module Map

| Module | Stages | Responsibility |
|---|---|---|
| `src/preprocessing.py` | 1.1 | Feature resolution, imputation, encoding, scaling pipelines |
| `src/models.py` | 1.2 | Model registry, hyperparameter grids, `tune_model()` |
| `src/validation.py` | 1.3, 1.4, 1.5 | Split strategies, internal/external loops, CFS penalty |
| `src/shift.py` | 1.6 | PSI, KS, C2ST, missingness shift, correlation analysis |
| `src/calibration.py` | 2.1, 2.2, 2.3 | Calibration metrics, Platt/isotonic/temperature, logistic recalibration |
| `src/sensitivity.py` | 2.4 | Size-matched subsampling experiments |
| `src/metrics.py` | C.2 | Centralized metric computation |
| `src/artifacts.py` | C.3 | Serialization of results, models, predictions |
| `src/pipeline.py` | Orchestration | Top-level `run_pipeline()` entrypoint |
| `configs/pipeline.yaml` | C.1 | All experiment settings |

---

## Output Directory Structure

```
outputs/
├── internal/
│   ├── kaggle/
│   │   ├── lr/   { results.json, predictions.parquet, predictions.csv, model.joblib, pipeline.joblib }
│   │   ├── rf/
│   │   ├── xgb/
│   │   └── lgbm/
│   ├── cleveland/
│   ├── hungary/
│   ├── va/
│   └── switzerland/
├── external_uci/
│   ├── cleveland__to__hungary/
│   │   ├── lr/   { results.json, predictions.parquet, model.joblib }
│   │   ├── rf/
│   │   ├── xgb/
│   │   └── lgbm/
│   ├── cleveland__to__va/
│   ├── ... (12 pairwise + 4 LOSO)
│   └── cleveland+hungary+va__to__switzerland/
├── external_kaggle_uci/
│   ├── kaggle__to__cleveland/
│   ├── cleveland__to__kaggle/
│   └── ...
├── shift/
│   ├── cleveland__to__hungary/   { shift_diagnostics.json, feature_shift.parquet }
│   ├── ...
│   └── shift_performance_correlation.parquet
├── calibration/
│   ├── before/   { per-experiment calibration metrics }
│   ├── recalibration/   { platt/, isotonic/, temperature/ per experiment }
│   └── updating/   { intercept_only/, intercept_slope/ per experiment }
└── size_matched/
    ├── kaggle__to__cleveland/   { subsampled_results.json }
    └── ...
```

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Tuning overfits on small UCI inner-CV folds | Inflated internal metrics | Use repeated CV for small sites; compare tuned vs default-param results |
| Recalibration overfits on tiny target calibration set | Calibration metrics look good but don't generalize | Use CV-based recalibration for sites with N < 100; report calibration set size |
| Switzerland nearly empty after dropping high-missingness features | Useless external site | Keep Switzerland as stress-test; report usable feature count and effective N |
| Hungary/VA also have severe missingness (fbs 100%, restecg 99.7% in Hungary; trestbps/thalach ~28% in VA) | UCI cross-site CFS much narrower than nominal 10 features for many pairs | `effective_cfs()` dynamically filters; report per-pair effective feature count alongside results |
| CFS too narrow (3 features) for Kaggle↔UCI | Very weak models | Report CFS penalty explicitly; reader knows the constraint |
| Grid search too slow for XGB × LOSO × all pairs | Long runtime | Use `RandomizedSearchCV(n_iter=20)` for boosted models; parallelize across pairs |
| Bootstrap CIs unstable for N < 50 in eval split | Wide CIs mask real effects | Use B ≥ 500; flag experiments where effective eval N is dangerously low |
