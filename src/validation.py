"""
Stage 1.3 — Internal Validation (RQ1).
Stage 1.4 — External Validation UCI Multi-Site Matrix (RQ2).
Splitting strategy, bootstrap CIs, internal/external validation loops.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

# LGBM/sklearn warn when predict gets array but model was fit with feature names; we pass DataFrame when dims match
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
    module="sklearn",
)
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

from . import artifacts
from . import metrics
from . import models
from . import preprocessing


# model_key -> model_family for preprocessing pipeline
MODEL_FAMILY = {
    "lr": "logistic_regression",
    "rf": "random_forest",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
}

INTERNAL_SITES = ["kaggle", "cleveland", "hungary", "va", "switzerland"]
UCI_SITES = ["cleveland", "hungary", "va", "switzerland"]
INTERNAL_MODELS = ["lr", "rf", "xgb", "lgbm"]


def _ensure_feature_names(model, X: np.ndarray):
    """Return X as DataFrame with model's feature names when they match, to avoid LGBM/sklearn warnings."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None and len(names) == X.shape[1]:
        return pd.DataFrame(X, columns=list(names))
    return X


def internal_split(df: pd.DataFrame, site: str, seed: int = 42):
    """
    Return list of (train_idx, test_idx).
    Switzerland: 5×5 Repeated Stratified CV. Others: single 80/20 stratified split.
    """
    X = df.drop(columns=["target"], errors="ignore")
    y = df["target"]
    if site == "switzerland":
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        return list(cv.split(X, y))
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, stratify=df["target"], random_state=seed
    )
    return [(train_idx, test_idx)]


def load_clean(site: str, data_dir: Path) -> pd.DataFrame:
    """Load cleaned parquet for site."""
    if site == "kaggle":
        path = data_dir / "kaggle_clean.parquet"
    else:
        path = data_dir / f"uci_{site}_clean.parquet"
    return pd.read_parquet(path)


def external_uci_validation_matrix(sites: list[str] | None = None) -> list[tuple[list[str], str]]:
    """
    Stage 1.4.1: All (train_sites, test_site) for UCI external validation.
    12 pairwise + 4 LOSO = 16 experiments per model.
    """
    sites = sites or UCI_SITES
    matrix = []
    for train_site in sites:
        for test_site in sites:
            if train_site != test_site:
                matrix.append(([train_site], test_site))
    for held_out in sites:
        train_sites = [s for s in sites if s != held_out]
        matrix.append((train_sites, held_out))
    return matrix


def run_external_uci_matrix(
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    report_path: str | Path | None = None,
    sites: list[str] | None = None,
    model_keys: list[str] | None = None,
    bootstrap_B: int = 500,
    seed: int = 42,
) -> dict:
    """
    Stage 1.4: Train on one or more UCI sites, test on held-out UCI site.
    Uses effective_cfs per (train, test) pair; saves results.json, predictions.parquet, model.joblib.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    if report_path is None:
        report_path = data_dir / "ingestion_report.json"
    config = preprocessing.load_preprocessing_config(report_path)
    sites = sites or UCI_SITES
    model_keys = model_keys or INTERNAL_MODELS
    candidate_features = config["cfs_features"]["default"]

    results = {}
    matrix = external_uci_validation_matrix(sites)

    for train_sites, test_site in matrix:
        train_key = "+".join(sorted(train_sites))
        exp_key = (train_key, test_site)
        results.setdefault(exp_key, {})

        try:
            train_dfs = [load_clean(s, data_dir) for s in train_sites]
            train_df = pd.concat(train_dfs, ignore_index=True)
            test_df = load_clean(test_site, data_dir)
        except FileNotFoundError:
            continue

        features = preprocessing.effective_cfs(train_df, test_df, candidate_features)
        features = [c for c in features if c in train_df.columns and c in test_df.columns]
        if not features:
            continue

        for model_key in model_keys:
            if model_key not in MODEL_FAMILY:
                continue
            model_family = MODEL_FAMILY[model_key]

            X_train = train_df[features]
            y_train = train_df["target"]
            X_test = test_df[features]
            y_test = test_df["target"]

            pipeline = preprocessing.build_preprocessing_pipeline(model_family, features)
            X_train_t, y_train_t = preprocessing.fit_transform_train(pipeline, X_train, y_train)
            X_test_t = preprocessing.transform_test(pipeline, X_test)
            X_train_t = np.asarray(X_train_t)
            X_test_t = np.asarray(X_test_t)

            n_train = len(X_train_t)
            fitted_model, best_params, _ = models.tune_model(model_key, X_train_t, y_train_t, n_train)

            X_test_pred = _ensure_feature_names(fitted_model, X_test_t)
            y_prob = fitted_model.predict_proba(X_test_pred)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            y_test_np = np.asarray(y_test)

            point_metrics = metrics.compute_metrics(y_test_np, y_prob, y_pred)
            boot_cis = metrics.bootstrap_metrics(y_test_np, y_prob, y_pred, B=bootstrap_B, seed=seed)

            predictions = [
                {"y_true": int(y_test_np[i]), "y_prob": float(y_prob[i]), "y_pred": int(y_pred[i])}
                for i in range(len(y_test_np))
            ]

            result = {
                "experiment_type": "external_uci",
                "train_sites": train_sites,
                "test_site": test_site,
                "model": model_key,
                "features_used": features,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "best_params": best_params,
                "metrics": point_metrics,
                "bootstrap_cis": boot_cis,
                "predictions": predictions,
                "fitted_model": fitted_model,
            }
            artifacts.save_experiment(result, base_dir=output_dir)
            results[exp_key][model_key] = result

    return results


def run_internal_validation(
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    report_path: str | Path | None = None,
    sites: list[str] | None = None,
    model_keys: list[str] | None = None,
    bootstrap_B: int = 200,
    seed: int = 42,
) -> dict:
    """
    Stage 1.3: For every site × model, run internal split(s), tune, predict, compute metrics + bootstrap CIs,
    save results.json, predictions.parquet, model.joblib, pipeline.joblib.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    if report_path is None:
        report_path = data_dir / "ingestion_report.json"
    config = preprocessing.load_preprocessing_config(report_path)
    sites = sites or INTERNAL_SITES
    model_keys = model_keys or INTERNAL_MODELS

    results = {}
    for site in sites:
        try:
            df = load_clean(site, data_dir)
        except FileNotFoundError:
            continue
        features = [c for c in preprocessing.resolve_features(site, site, config) if c in df.columns]
        # Drop columns that are all NaN (e.g. thal in VA/Hungary) so pipeline never sees missing column
        features = [c for c in features if df[c].notna().any()]
        if not features:
            continue
        splits = internal_split(df, site, seed=seed)

        for model_key in model_keys:
            if model_key not in MODEL_FAMILY:
                continue
            model_family = MODEL_FAMILY[model_key]

            all_preds = []
            fold_metrics = []
            fold_cis = []
            fitted_model = None
            fitted_pipeline = None
            best_params = None

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                X_train = df.loc[train_idx, features]
                y_train = df.loc[train_idx, "target"]
                X_test = df.loc[test_idx, features]
                y_test = df.loc[test_idx, "target"]

                pipeline = preprocessing.build_preprocessing_pipeline(model_family, features)
                X_train_t, y_train_t = preprocessing.fit_transform_train(pipeline, X_train, y_train)
                X_test_t = preprocessing.transform_test(pipeline, X_test)
                # Ensure arrays for sklearn/GPU models (ensures consistent dtype)
                X_train_t = np.asarray(X_train_t)
                X_test_t = np.asarray(X_test_t)

                n_train = len(X_train_t)
                fitted_model, best_params, _ = models.tune_model(model_key, X_train_t, y_train_t, n_train)

                X_test_pred = _ensure_feature_names(fitted_model, X_test_t)
                y_prob = fitted_model.predict_proba(X_test_pred)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                y_test_np = np.asarray(y_test)

                point_metrics = metrics.compute_metrics(y_test_np, y_prob, y_pred)
                fold_metrics.append(point_metrics)
                B = 500 if len(y_test_np) < 200 else bootstrap_B
                boot_cis = metrics.bootstrap_metrics(y_test_np, y_prob, y_pred, B=B, seed=seed)
                fold_cis.append(boot_cis)

                for i in range(len(y_test_np)):
                    all_preds.append({
                        "y_true": int(y_test_np[i]),
                        "y_prob": float(y_prob[i]),
                        "y_pred": int(y_pred[i]),
                        "fold": fold_idx,
                    })
                fitted_pipeline = pipeline

            if len(fold_metrics) == 1:
                agg_metrics = fold_metrics[0]
                agg_cis = fold_cis[0]
            else:
                keys_float = ["roc_auc", "pr_auc", "brier_score", "accuracy", "ece", "mce", "f1", "precision", "recall", "specificity"]
                agg_metrics = {"n_test": sum(m["n_test"] for m in fold_metrics)}
                for k in keys_float:
                    if k in fold_metrics[0]:
                        agg_metrics[k] = float(np.mean([m[k] for m in fold_metrics]))
                agg_metrics["prevalence"] = float(np.mean([m["prevalence"] for m in fold_metrics]))
                cm = np.array(fold_metrics[0]["confusion_matrix"], dtype=float)
                for m in fold_metrics[1:]:
                    cm += np.array(m["confusion_matrix"])
                agg_metrics["confusion_matrix"] = cm.astype(int).tolist()
                agg_cis = fold_cis[0]

            result = {
                "experiment_type": "internal",
                "site": site,
                "model": model_key,
                "best_params": best_params,
                "metrics": agg_metrics,
                "bootstrap_cis": agg_cis,
                "predictions": all_preds,
                "fitted_model": fitted_model,
                "fitted_pipeline": fitted_pipeline,
            }
            artifacts.save_experiment(result, base_dir=output_dir)
            results.setdefault(site, {})[model_key] = result
    return results


if __name__ == "__main__":
    import sys
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "data"
    output_dir = base / "outputs"
    if not (data_dir / "ingestion_report.json").exists():
        print("Run ingestion first: python -m src.ingest", file=sys.stderr)
        sys.exit(1)
    run_stage_14 = "--stage-1.4" in sys.argv or "--external-uci" in sys.argv
    if run_stage_14:
        quick = "--quick" in sys.argv
        if quick:
            run_external_uci_matrix(
                data_dir=data_dir,
                output_dir=output_dir,
                sites=["cleveland", "hungary"],
                model_keys=["lr"],
                bootstrap_B=100,
            )
        else:
            run_external_uci_matrix(data_dir=data_dir, output_dir=output_dir)
        print("Stage 1.4 done. Outputs under", output_dir / "external_uci")
    else:
        quick = "--quick" in sys.argv
        if quick:
            run_internal_validation(data_dir=data_dir, output_dir=output_dir, sites=["cleveland"], model_keys=["lr"])
        else:
            run_internal_validation(data_dir=data_dir, output_dir=output_dir)
        print("Stage 1.3 done. Outputs under", output_dir / "internal")

