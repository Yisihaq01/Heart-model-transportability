"""
Stage 1.3 — Internal Validation (RQ1).
Stage 1.4 — External Validation UCI Multi-Site Matrix (RQ2).
Stage 1.5 — External Validation Kaggle ↔ UCI Stress Test (RQ5).
Splitting strategy, bootstrap CIs, internal/external validation loops.
"""
from __future__ import annotations

import json
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
from . import config
from . import metrics
from . import models
from . import preprocessing
from . import reproducibility


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


def bin_cholesterol_uci(chol_series: pd.Series) -> pd.Series:
    """Bin UCI mg/dl cholesterol into Kaggle-style ordinal {1,2,3} (ATP III cutpoints).

    [-inf, 200) -> 1, [200, 240) -> 2, [240, inf) -> 3. right=False ensures 200/240 map upward.
    """
    return pd.cut(
        chol_series,
        bins=[-np.inf, 200, 240, np.inf],
        labels=[1, 2, 3],
        right=False,
    ).astype(float)


def _maybe_add_binned_cholesterol(df: pd.DataFrame, site: str, include_cholesterol: bool) -> pd.DataFrame:
    """Ensure a Kaggle-compatible `cholesterol` column exists when requested."""
    if not include_cholesterol:
        return df
    if "cholesterol" in df.columns:
        return df
    if site == "kaggle":
        # Kaggle should already have cholesterol; if not, let it fail downstream on missing feature.
        return df
    if "chol" not in df.columns:
        return df
    out = df.copy()
    out["cholesterol"] = bin_cholesterol_uci(out["chol"])
    return out


def _load_full_internal_auc(output_dir: Path, site: str, model_key: str) -> float | None:
    """Best-effort fetch of Stage 1.3 full-feature internal AUC for penalty calculations."""
    try:
        p = output_dir / "internal" / site / model_key / "results.json"
        if not p.exists():
            return None
        with open(p) as f:
            payload = json.load(f)
        return float(payload.get("metrics", {}).get("roc_auc"))
    except Exception:
        return None


def cfs_penalty(full_auc: float | None, cfs_auc: float | None) -> dict:
    if full_auc is None or cfs_auc is None:
        return {"full_auc": full_auc, "cfs_auc": cfs_auc, "auc_drop": None, "relative_drop_pct": None}
    drop = float(full_auc) - float(cfs_auc)
    rel = (drop / float(full_auc) * 100.0) if float(full_auc) > 0 else None
    return {"full_auc": float(full_auc), "cfs_auc": float(cfs_auc), "auc_drop": drop, "relative_drop_pct": rel}


def _kaggle_uci_matrix(uci_sites: list[str], include_pooled: bool = True) -> list[tuple[str, list[str], str, list[str]]]:
    """
    Stage 1.5.1: Kaggle ↔ UCI stress-test experiments.

    Returns list of (train_label, train_sites, test_label, test_sites).
    train_label/test_label are used only for artifact pathing.
    """
    exps: list[tuple[str, list[str], str, list[str]]] = []
    for s in uci_sites:
        exps.append(("kaggle", ["kaggle"], s, [s]))
        exps.append((s, [s], "kaggle", ["kaggle"]))
    if include_pooled:
        pooled = "+".join(sorted(uci_sites))
        exps.append((pooled, list(uci_sites), "kaggle", ["kaggle"]))  # UCI pooled -> Kaggle
        exps.append(("kaggle", ["kaggle"], pooled, list(uci_sites)))  # Kaggle -> UCI pooled
    return exps


def _load_sites(sites: list[str], data_dir: Path) -> pd.DataFrame:
    dfs = [load_clean(s, data_dir) for s in sites]
    return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]


def _load_pipeline_repro(config_path: str | Path | None) -> tuple[dict, int, str]:
    """Load pipeline config; return (config, seed, config_hash). C.4 reproducibility."""
    path = config_path or Path(__file__).resolve().parent.parent / "configs" / "pipeline.yaml"
    path = Path(path)
    cfg = config.load_config(path) if path.exists() else {"random_seed": 42}
    seed = int(cfg.get("random_seed", 42))
    reproducibility.set_global_seed(seed)
    return cfg, seed, reproducibility.config_hash(cfg)


def run_internal_cfs_validation(
    df: pd.DataFrame,
    dataset_label: str,
    feature_list: list[str],
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    model_keys: list[str] | None = None,
    bootstrap_B: int = 200,
    seed: int = 42,
    variant: str = "cfs",
    config_hash: str | None = None,
) -> dict:
    """
    Stage 1.5 baseline: internal validation restricted to Kaggle↔UCI CFS (to measure CFS penalty).

    Writes to outputs/internal_cfs/{dataset_label}/{model}/...
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    model_keys = model_keys or INTERNAL_MODELS

    features = [c for c in feature_list if c in df.columns and df[c].notna().any()]
    if not features:
        return {}

    splits = internal_split(df, dataset_label, seed=seed)

    results: dict = {}
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
            X_train_t = np.asarray(X_train_t)
            X_test_t = np.asarray(X_test_t)

            n_train = len(X_train_t)
            fitted_model, best_params, _ = models.tune_model(model_key, X_train_t, y_train_t, n_train, seed=seed)

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
            agg_metrics = {"n_test": sum(m.get("n_test", 0) for m in fold_metrics)}
            for k in keys_float:
                if k in fold_metrics[0]:
                    agg_metrics[k] = float(np.mean([m[k] for m in fold_metrics]))
            agg_metrics["prevalence"] = float(np.mean([m.get("prevalence", 0.0) for m in fold_metrics]))
            cm = np.array(fold_metrics[0]["confusion_matrix"], dtype=float)
            for m in fold_metrics[1:]:
                cm += np.array(m["confusion_matrix"])
            agg_metrics["confusion_matrix"] = cm.astype(int).tolist()
            agg_cis = fold_cis[0]

        full_auc = _load_full_internal_auc(output_dir, dataset_label, model_key)
        penalty = cfs_penalty(full_auc, agg_metrics.get("roc_auc"))

        result = {
            "experiment_type": "internal_cfs",
            "variant": variant,
            "site": dataset_label,
            "model": model_key,
            "features_used": features,
            "metrics": agg_metrics,
            "bootstrap_cis": agg_cis,
            "best_params": best_params,
            "cfs_penalty": penalty,
            "predictions": all_preds,
            "fitted_model": fitted_model,
            "fitted_pipeline": fitted_pipeline,
        }
        artifacts.save_experiment(result, base_dir=output_dir, config_hash=config_hash)
        results[model_key] = result

    return results


def run_kaggle_uci_tests(
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    report_path: str | Path | None = None,
    pipeline_config_path: str | Path | None = None,
    uci_sites: list[str] | None = None,
    model_keys: list[str] | None = None,
    bootstrap_B: int = 500,
    seed: int | None = None,
    include_cholesterol: bool = False,
    include_pooled: bool = True,
    run_internal_baselines: bool = True,
) -> dict:
    """
    Stage 1.5: External validation across Kaggle ↔ UCI using the Kaggle-UCI CFS.

    Writes to outputs/external_kaggle_uci/{train}__to__{test}/{model}/...
    Also (optionally) writes CFS-only internal baselines to outputs/internal_cfs/...
    C.4: loads pipeline config when pipeline_config_path given; uses config seed and config_hash.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    _, seed_used, cfg_hash = _load_pipeline_repro(pipeline_config_path)
    if seed is not None:
        seed_used = seed
    else:
        reproducibility.set_global_seed(seed_used)
    if report_path is None:
        report_path = data_dir / "ingestion_report.json"
    config = preprocessing.load_preprocessing_config(report_path)
    uci_sites = uci_sites or UCI_SITES
    model_keys = model_keys or INTERNAL_MODELS

    variant = "cfs_plus_chol" if include_cholesterol else "cfs"
    base_cfs = list(config["cfs_features"].get(tuple(sorted(["kaggle", uci_sites[0]])), ["age", "sex", "sys_bp"]))
    cfs_features = base_cfs + (["cholesterol"] if include_cholesterol and "cholesterol" not in base_cfs else [])

    # Cache internal baselines per dataset_label (kaggle, each uci site, pooled)
    internal_cfs: dict[str, dict] = {}

    results: dict = {}
    for train_label, train_sites, test_label, test_sites in _kaggle_uci_matrix(uci_sites, include_pooled=include_pooled):
        try:
            train_df = _load_sites(train_sites, data_dir)
            test_df = _load_sites(test_sites, data_dir)
        except FileNotFoundError:
            continue

        # Add binned cholesterol to UCI frames when needed
        train_df = _maybe_add_binned_cholesterol(train_df, train_sites[0] if len(train_sites) == 1 else "uci", include_cholesterol)
        test_df = _maybe_add_binned_cholesterol(test_df, test_sites[0] if len(test_sites) == 1 else "uci", include_cholesterol)

        features = [c for c in cfs_features if c in train_df.columns and c in test_df.columns]
        features = [c for c in features if train_df[c].notna().any() and test_df[c].notna().any()]
        if not features:
            continue

        # Internal CFS baselines for penalty quantification
        if run_internal_baselines:
            for label, sites in ((train_label, train_sites), (test_label, test_sites)):
                if label in internal_cfs:
                    continue
                try:
                    df0 = _load_sites(sites, data_dir)
                except FileNotFoundError:
                    continue
                df0 = _maybe_add_binned_cholesterol(df0, sites[0] if len(sites) == 1 else "uci", include_cholesterol)
                internal_cfs[label] = run_internal_cfs_validation(
                    df=df0,
                    dataset_label=label,
                    feature_list=cfs_features,
                    data_dir=data_dir,
                    output_dir=output_dir,
                    model_keys=model_keys,
                    bootstrap_B=200,
                    seed=seed_used,
                    variant=variant,
                    config_hash=cfg_hash,
                )

        exp_key = (train_label, test_label)
        results.setdefault(exp_key, {})

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

            fitted_model, best_params, _ = models.tune_model(model_key, X_train_t, y_train_t, len(X_train_t), seed=seed_used)

            X_test_pred = _ensure_feature_names(fitted_model, X_test_t)
            y_prob = fitted_model.predict_proba(X_test_pred)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            y_test_np = np.asarray(y_test)

            point_metrics = metrics.compute_metrics(y_test_np, y_prob, y_pred)
            boot_cis = metrics.bootstrap_metrics(y_test_np, y_prob, y_pred, B=bootstrap_B, seed=seed_used)

            predictions = [
                {"y_true": int(y_test_np[i]), "y_prob": float(y_prob[i]), "y_pred": int(y_pred[i])}
                for i in range(len(y_test_np))
            ]

            result = {
                "experiment_type": "external_kaggle_uci",
                "variant": variant,
                "train_site": train_label,
                "test_site": test_label,
                "model": model_key,
                "features_used": features,
                "include_cholesterol": bool(include_cholesterol),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "best_params": best_params,
                "metrics": point_metrics,
                "bootstrap_cis": boot_cis,
                "predictions": predictions,
                "fitted_model": fitted_model,
                "fitted_pipeline": pipeline,
            }
            artifacts.save_experiment(result, base_dir=output_dir, config_hash=cfg_hash)
            results[exp_key][model_key] = result

    return {"external": results, "internal_cfs": internal_cfs, "cfs_features": cfs_features, "variant": variant}


def run_external_uci_matrix(
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    report_path: str | Path | None = None,
    pipeline_config_path: str | Path | None = None,
    sites: list[str] | None = None,
    model_keys: list[str] | None = None,
    bootstrap_B: int = 500,
    seed: int | None = None,
) -> dict:
    """
    Stage 1.4: Train on one or more UCI sites, test on held-out UCI site.
    Uses effective_cfs per (train, test) pair; saves results.json, predictions.parquet, model.joblib.
    C.4: loads pipeline config for seed and config_hash when pipeline_config_path given.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    _, seed_used, cfg_hash = _load_pipeline_repro(pipeline_config_path)
    if seed is not None:
        seed_used = seed
        reproducibility.set_global_seed(seed_used)
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
            fitted_model, best_params, _ = models.tune_model(model_key, X_train_t, y_train_t, n_train, seed=seed_used)

            X_test_pred = _ensure_feature_names(fitted_model, X_test_t)
            y_prob = fitted_model.predict_proba(X_test_pred)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            y_test_np = np.asarray(y_test)

            point_metrics = metrics.compute_metrics(y_test_np, y_prob, y_pred)
            boot_cis = metrics.bootstrap_metrics(y_test_np, y_prob, y_pred, B=bootstrap_B, seed=seed_used)

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
            artifacts.save_experiment(result, base_dir=output_dir, config_hash=cfg_hash)
            results[exp_key][model_key] = result

    return results


def run_internal_validation(
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    report_path: str | Path | None = None,
    pipeline_config_path: str | Path | None = None,
    sites: list[str] | None = None,
    model_keys: list[str] | None = None,
    bootstrap_B: int = 200,
    seed: int | None = None,
) -> dict:
    """
    Stage 1.3: For every site × model, run internal split(s), tune, predict, compute metrics + bootstrap CIs,
    save results.json, predictions.parquet, model.joblib, pipeline.joblib.
    C.4: loads pipeline config for seed and config_hash; logs features_used, n_train, n_test, timestamp.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    _, seed_used, cfg_hash = _load_pipeline_repro(pipeline_config_path)
    if seed is not None:
        seed_used = seed
        reproducibility.set_global_seed(seed_used)
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
        splits = internal_split(df, site, seed=seed_used)

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
                fitted_model, best_params, _ = models.tune_model(model_key, X_train_t, y_train_t, n_train, seed=seed_used)

                X_test_pred = _ensure_feature_names(fitted_model, X_test_t)
                y_prob = fitted_model.predict_proba(X_test_pred)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                y_test_np = np.asarray(y_test)

                point_metrics = metrics.compute_metrics(y_test_np, y_prob, y_pred)
                fold_metrics.append(point_metrics)
                B = 500 if len(y_test_np) < 200 else bootstrap_B
                boot_cis = metrics.bootstrap_metrics(y_test_np, y_prob, y_pred, B=B, seed=seed_used)
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

            n_test = agg_metrics.get("n_test", 0)
            n_train_final = len(splits[0][0]) if splits else None  # train size (first fold)
            result = {
                "experiment_type": "internal",
                "site": site,
                "model": model_key,
                "features_used": features,
                "n_train": n_train_final,
                "n_test": n_test,
                "best_params": best_params,
                "metrics": agg_metrics,
                "bootstrap_cis": agg_cis,
                "predictions": all_preds,
                "fitted_model": fitted_model,
                "fitted_pipeline": fitted_pipeline,
            }
            artifacts.save_experiment(result, base_dir=output_dir, config_hash=cfg_hash)
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
    run_stage_15 = "--stage-1.5" in sys.argv or "--kaggle-uci" in sys.argv
    if run_stage_15:
        quick = "--quick" in sys.argv
        with_chol = "--with-cholesterol" in sys.argv
        both_modes = "--both-cholesterol-modes" in sys.argv
        if quick:
            uci_sites = ["cleveland"]
            model_keys = ["lr"]
            include_pooled = False
            bootstrap_B = 100
        else:
            uci_sites = None
            model_keys = None
            include_pooled = True
            bootstrap_B = 500
        if both_modes:
            run_kaggle_uci_tests(
                data_dir=data_dir,
                output_dir=output_dir,
                uci_sites=uci_sites,
                model_keys=model_keys,
                bootstrap_B=bootstrap_B,
                include_cholesterol=False,
                include_pooled=include_pooled,
            )
            run_kaggle_uci_tests(
                data_dir=data_dir,
                output_dir=output_dir,
                uci_sites=uci_sites,
                model_keys=model_keys,
                bootstrap_B=bootstrap_B,
                include_cholesterol=True,
                include_pooled=include_pooled,
            )
        else:
            run_kaggle_uci_tests(
                data_dir=data_dir,
                output_dir=output_dir,
                uci_sites=uci_sites,
                model_keys=model_keys,
                bootstrap_B=bootstrap_B,
                include_cholesterol=with_chol,
                include_pooled=include_pooled,
            )
        print("Stage 1.5 done. Outputs under", output_dir / "external_kaggle_uci", "and", output_dir / "internal_cfs")
    elif run_stage_14:
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

