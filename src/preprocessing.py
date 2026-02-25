"""
Preprocessing per ImplementationPlan/pipeline_plan.md Stage 1.1.
Feature resolution, imputation, encoding, scaling — fit on train only, transform test.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

# --- Feature-set resolution (§1.1.1, §1.4.2) ---

# Kaggle: 11 predictors (exclude bp_outlier from modeling)
KAGGLE_FULL_FEATURES = [
    "age", "sex", "height", "weight", "sys_bp", "dia_bp",
    "cholesterol", "gluc", "smoke", "alco", "active",
]
# UCI: 13 predictors (exclude num/target)
UCI_FULL_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

MISSINGNESS_THRESHOLD = 0.40


def load_preprocessing_config(report_path: str | Path | None = None) -> dict:
    """Build config from ingestion_report.json: full_features per site, cfs_features for pairs."""
    if report_path is None:
        report_path = Path(__file__).resolve().parent.parent / "data" / "ingestion_report.json"
    report_path = Path(report_path)
    with open(report_path) as f:
        report = json.load(f)
    cfs_ku = report.get("cfs_kaggle_uci", ["age", "sex", "sys_bp"])
    cfs_uu = report.get("cfs_uci_cross_site", [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"
    ])
    full = {"kaggle": KAGGLE_FULL_FEATURES}
    for site in ("cleveland", "hungary", "switzerland", "va"):
        full[site] = UCI_FULL_FEATURES
    # Cross-site: Kaggle ↔ UCI use cfs_kaggle_uci; UCI ↔ UCI use cfs_uci_cross_site (then effective_cfs filters)
    cfs_features = {"default": cfs_uu}
    uci_sites = [s for s in ("cleveland", "hungary", "switzerland", "va") if s in report.get("sites", {})]
    for u in uci_sites:
        cfs_features[tuple(sorted(["kaggle", u]))] = cfs_ku
    return {"full_features": full, "cfs_features": cfs_features}


def resolve_features(train_site: str, test_site: str, config: dict) -> list[str]:
    """Return the feature list for a given train/test site pair."""
    if train_site == test_site:
        return config["full_features"].get(train_site, [])
    pair_key = tuple(sorted([train_site, test_site]))
    return config["cfs_features"].get(pair_key, config["cfs_features"]["default"])


def effective_cfs(train_df: pd.DataFrame, test_df: pd.DataFrame, candidate_features: list[str]) -> list[str]:
    """Drop features with >40% missing in either train or test."""
    usable = []
    for col in candidate_features:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        train_miss = train_df[col].isna().mean()
        test_miss = test_df[col].isna().mean()
        if train_miss < MISSINGNESS_THRESHOLD and test_miss < MISSINGNESS_THRESHOLD:
            usable.append(col)
    return usable


# --- Column roles for imputation/encoding (§1.1.2, §1.1.3) ---

NOMINAL_COLS = ["cp", "thal"]
ORDINAL_COLS = ["cholesterol", "gluc", "restecg", "slope"]
BINARY_COLS = ["sex", "fbs", "exang", "smoke", "alco", "active"]
CONTINUOUS_NAMES = {"age", "sys_bp", "trestbps", "chol", "thalach", "oldpeak", "height", "weight", "dia_bp", "ca"}


def _continuous_and_categorical(feature_list: list[str]) -> tuple[list[str], list[str]]:
    continuous = [c for c in feature_list if c in CONTINUOUS_NAMES]
    categorical = [c for c in feature_list if c in BINARY_COLS + ORDINAL_COLS + NOMINAL_COLS]
    return continuous, categorical


# --- Imputation (§1.1.2) ---

def build_imputer(
    model_family: str,
    feature_list: list[str],
):
    """Passthrough for tree models (XGB/LGBM); median/mode ColumnTransformer for LR/RF.
    Outputs DataFrame with original column names in feature_list order so encoder can use them."""
    if model_family in ("xgboost", "lightgbm"):
        return FunctionTransformer(validate=False)
    continuous_cols, categorical_cols = _continuous_and_categorical(feature_list)
    if not continuous_cols and not categorical_cols:
        return FunctionTransformer(validate=False)
    # One transformer per column in feature_list order
    transformers = []
    for col in feature_list:
        if col in CONTINUOUS_NAMES:
            transformers.append((col, SimpleImputer(strategy="median"), [col]))
        elif col in BINARY_COLS + ORDINAL_COLS + NOMINAL_COLS:
            transformers.append((col, SimpleImputer(strategy="most_frequent"), [col]))
        else:
            transformers.append((col, "passthrough", [col]))
    ct = ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return ct


# --- Encoding (§1.1.3) ---

def build_encoder(feature_list: list[str]):
    nominal = [c for c in NOMINAL_COLS if c in feature_list]
    ordinal = [c for c in ORDINAL_COLS if c in feature_list]
    passthrough = [c for c in feature_list if c not in nominal + ordinal]
    transformers = []
    if nominal:
        transformers.append((
            "nominal",
            OneHotEncoder(drop="first", sparse_output=False, handle_unknown="infrequent_if_exist"),
            nominal,
        ))
    if ordinal:
        transformers.append(("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ordinal))
    if passthrough:
        transformers.append(("pass", "passthrough", passthrough))
    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


# --- Full pipeline (§1.1.5) ---

def build_preprocessing_pipeline(
    model_family: str,
    feature_list: list[str],
) -> Pipeline:
    """Imputer (or passthrough) -> encoder -> scaler for LR only."""
    steps = [
        ("imputer", build_imputer(model_family, feature_list)),
        ("encoder", build_encoder(feature_list)),
    ]
    if model_family == "logistic_regression":
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def fit_transform_train(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit pipeline on X_train, return (X_train_transformed, y_train). y_train unchanged."""
    X_t = pipeline.fit_transform(X_train)
    return X_t, np.asarray(y_train)


def transform_test(pipeline: Pipeline, X_test: pd.DataFrame) -> np.ndarray:
    """Transform test set (no fit)."""
    return pipeline.transform(X_test)


if __name__ == "__main__":
    # Smoke test: load one clean parquet, resolve features, build pipeline, fit/transform
    data_dir = Path(__file__).resolve().parent.parent / "data"
    report_path = data_dir / "ingestion_report.json"
    if not report_path.exists():
        raise SystemExit("Run ingestion first: python -m src.ingest")
    config = load_preprocessing_config(report_path)
    # Internal validation: same site
    features = resolve_features("cleveland", "cleveland", config)
    assert "cp" in features and "thal" in features, "UCI full features expected"
    df = pd.read_parquet(data_dir / "uci_cleveland_clean.parquet")
    X = df[features]
    y = df["target"]
    pipe = build_preprocessing_pipeline("logistic_regression", features)
    X_t, y_t = fit_transform_train(pipe, X, y)
    assert X_t.shape[0] == len(df) and X_t.shape[1] >= len(features), "Shape mismatch"
    X_test_t = transform_test(pipe, X.head(10))
    assert X_test_t.shape[0] == 10
    print("Stage 1.1 smoke test OK: resolve_features, build_preprocessing_pipeline, fit/transform")
    # effective_cfs
    df_va = pd.read_parquet(data_dir / "uci_va_clean.parquet")
    eff = effective_cfs(df, df_va, config["cfs_features"]["default"])
    print(f"effective_cfs cleveland->va: {len(eff)} features: {eff}")
