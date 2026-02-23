"""
Stage 1.3 — Internal Validation (RQ1).
Splitting strategy, bootstrap CIs, and internal validation loop per site × model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
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
INTERNAL_MODELS = ["lr", "rf", "xgb", "lgbm"]


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
                # Ensure arrays for sklearn/GPU models (avoids "feature names" warnings, ensures consistent dtype)
                X_train_t = np.asarray(X_train_t)
                X_test_t = np.asarray(X_test_t)

                n_train = len(X_train_t)
                fitted_model, best_params, _ = models.tune_model(model_key, X_train_t, y_train_t, n_train)

                y_prob = fitted_model.predict_proba(X_test_t)[:, 1]
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
    # Quick run: --quick uses 1 site (cleveland) and 1 model (lr) for a fast smoke test
    quick = "--quick" in sys.argv
    if quick:
        run_internal_validation(data_dir=data_dir, output_dir=output_dir, sites=["cleveland"], model_keys=["lr"])
    else:
        run_internal_validation(data_dir=data_dir, output_dir=output_dir)
    print("Stage 1.3 done. Outputs under", output_dir / "internal")

