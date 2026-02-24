"""
Stage 1.6 — Dataset Shift Diagnostics (RQ3).

Implements:
- Prevalence shift
- Univariate covariate shift (KS / chi² + PSI)
- Missingness shift
- Multivariate shift via classifier two-sample test (C2ST)
- Shift–performance correlation

Reads:
- Cleaned data from data/*.parquet (produced by src.ingest)
- External validation results from:
  - outputs/external_uci/**/results.json        (Stage 1.4)
  - outputs/external_kaggle_uci/**/results.json (Stage 1.5, optional)
- Internal baselines from:
  - outputs/internal/**/results.json            (Stage 1.3, full-feature)

Writes:
- Per-pair diagnostics under outputs/shift/{train_sites}__to__{test_site}/
  - shift_diagnostics.json
  - feature_shift.parquet
- Global correlation matrix:
  - outputs/shift/shift_performance_correlation.parquet
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from . import preprocessing


@dataclass
class ExternalExperiment:
    train_sites: List[str]
    test_site: str
    model: str
    features_used: List[str]
    roc_auc: float | None
    brier: float | None
    result_path: Path
    experiment_type: str  # "external_uci" or "external_kaggle_uci"


def prevalence_shift(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Stage 1.6.1 — label prevalence shift."""
    p_train = train_df["target"].mean()
    p_test = test_df["target"].mean()
    return {
        "train_prevalence": float(p_train),
        "test_prevalence": float(p_test),
        "absolute_diff": float(abs(p_train - p_test)),
    }


def psi(train_col: pd.Series, test_col: pd.Series, bins: int = 10) -> float:
    """Stage 1.6.2/PSI helper — Population Stability Index."""
    train = train_col.dropna().to_numpy()
    test = test_col.dropna().to_numpy()
    if len(train) == 0 or len(test) == 0:
        return float("nan")
    # Guard against degenerate percentiles
    try:
        breakpoints = np.percentile(train, np.linspace(0, 100, bins + 1))
    except Exception:
        return float("nan")
    # Ensure strictly increasing edges to satisfy numpy histogram
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return float("nan")
    train_hist, _ = np.histogram(train, bins=breakpoints)
    test_hist, _ = np.histogram(test, bins=breakpoints)
    train_pct = train_hist / max(train_hist.sum(), 1)
    test_pct = test_hist / max(test_hist.sum(), 1)
    train_pct = np.clip(train_pct, 1e-6, None)
    test_pct = np.clip(test_pct, 1e-6, None)
    return float(np.sum((test_pct - train_pct) * np.log(test_pct / train_pct)))


def univariate_shift(
    train_col: pd.Series,
    test_col: pd.Series,
    col_type: str,
) -> dict:
    """
    Stage 1.6.2 — per-feature distribution shift.

    col_type: "continuous" or "categorical".
    """
    train_nonnull = train_col.dropna()
    test_nonnull = test_col.dropna()
    if len(train_nonnull) == 0 or len(test_nonnull) == 0:
        return {"test": None, "statistic": None, "p_value": None}

    if col_type == "continuous":
        stat, p = ks_2samp(train_nonnull, test_nonnull)
        return {"test": "ks", "statistic": float(stat), "p_value": float(p)}

    # Build a small DataFrame with reset indices to avoid duplicate-label reindex issues.
    df = pd.DataFrame(
        {
            "value": pd.concat([train_nonnull, test_nonnull], ignore_index=True),
            "domain": (["train"] * len(train_nonnull)) + (["test"] * len(test_nonnull)),
        }
    )
    contingency = pd.crosstab(df["value"], df["domain"])
    # If there is only one category, chi2_contingency is not meaningful
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {"test": "chi2", "statistic": None, "p_value": None}
    stat, p, _, _ = chi2_contingency(contingency)
    return {"test": "chi2", "statistic": float(stat), "p_value": float(p)}


def missingness_shift(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Stage 1.6.3 — compare per-feature missingness between train and test."""
    rows = []
    for col in features:
        train_miss = train_df[col].isna().mean()
        test_miss = test_df[col].isna().mean()
        rows.append(
            {
                "feature": col,
                "train_miss_pct": round(float(train_miss * 100), 1),
                "test_miss_pct": round(float(test_miss * 100), 1),
                "diff_pct": round(float((test_miss - train_miss) * 100), 1),
            }
        )
    return pd.DataFrame(rows)


def classifier_two_sample_test(X_train: np.ndarray, X_test: np.ndarray, seed: int = 42) -> float:
    """Stage 1.6.4 — C2ST AUC; ≈0.5 means little detectable shift."""
    X = np.vstack([X_train, X_test])
    y = np.array([0] * len(X_train) + [1] * len(X_test))
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    return float(scores.mean())


def _feature_type(col: str) -> str:
    """Heuristic mapping to 'continuous' vs 'categorical' using preprocessing module."""
    if col in preprocessing.CONTINUOUS_NAMES:
        return "continuous"
    if col in preprocessing.BINARY_COLS + preprocessing.ORDINAL_COLS + preprocessing.NOMINAL_COLS:
        return "categorical"
    # Fallback: numeric → continuous, else categorical
    return "continuous"


def _load_clean(site: str, data_dir: Path) -> pd.DataFrame:
    """Mirror src.validation.load_clean but kept local to avoid circular imports."""
    if site == "kaggle":
        path = data_dir / "kaggle_clean.parquet"
    else:
        path = data_dir / f"uci_{site}_clean.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _parse_site_label(label: str) -> List[str]:
    """Turn 'cleveland+hungary' or 'kaggle' into a list of component sites."""
    if "+" in label:
        return label.split("+")
    return [label]


def _discover_external_experiments(
    output_dir: Path,
    include_kaggle_uci: bool = True,
) -> List[ExternalExperiment]:
    """Scan outputs/ for external_uci and (optionally) external_kaggle_uci results."""
    exps: List[ExternalExperiment] = []

    # UCI multi-site matrix
    external_uci_root = output_dir / "external_uci"
    if external_uci_root.exists():
        for pair_dir in external_uci_root.iterdir():
            if not pair_dir.is_dir():
                continue
            # e.g. cleveland+hungary__to__va
            for model_dir in pair_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                res_path = model_dir / "results.json"
                if not res_path.exists():
                    continue
                with open(res_path) as f:
                    payload = json.load(f)
                train_sites = payload.get("train_sites")
                if isinstance(train_sites, str):
                    train_sites = _parse_site_label(train_sites)
                test_site = payload.get("test_site")
                model = payload.get("model")
                features_used = payload.get("features_used", [])
                metrics = payload.get("metrics", {})
                exps.append(
                    ExternalExperiment(
                        train_sites=list(train_sites),
                        test_site=str(test_site),
                        model=str(model),
                        features_used=list(features_used),
                        roc_auc=float(metrics.get("roc_auc")) if "roc_auc" in metrics else None,
                        brier=float(metrics.get("brier_score")) if "brier_score" in metrics else None,
                        result_path=res_path,
                        experiment_type="external_uci",
                    )
                )

    # Kaggle ↔ UCI stress tests
    if include_kaggle_uci:
        kaggle_root = output_dir / "external_kaggle_uci"
        if kaggle_root.exists():
            for variant_dir in kaggle_root.iterdir():
                if not variant_dir.is_dir():
                    continue
                for pair_dir in variant_dir.iterdir():
                    if not pair_dir.is_dir():
                        continue
                    for model_dir in pair_dir.iterdir():
                        if not model_dir.is_dir():
                            continue
                        res_path = model_dir / "results.json"
                        if not res_path.exists():
                            continue
                        with open(res_path) as f:
                            payload = json.load(f)
                        train_site = str(payload.get("train_site"))
                        test_site = str(payload.get("test_site"))
                        model = payload.get("model")
                        features_used = payload.get("features_used", [])
                        metrics = payload.get("metrics", {})
                        exps.append(
                            ExternalExperiment(
                                train_sites=_parse_site_label(train_site),
                                test_site=test_site,
                                model=str(model),
                                features_used=list(features_used),
                                roc_auc=float(metrics.get("roc_auc")) if "roc_auc" in metrics else None,
                                brier=float(metrics.get("brier_score")) if "brier_score" in metrics else None,
                                result_path=res_path,
                                experiment_type="external_kaggle_uci",
                            )
                        )

    return exps


def _load_internal_baseline_auc_brier(
    output_dir: Path,
    site: str,
    model: str,
) -> Tuple[float | None, float | None]:
    """Best-effort load of internal full-feature baseline for (site, model)."""
    p = output_dir / "internal" / site / model / "results.json"
    if not p.exists():
        return None, None
    try:
        with open(p) as f:
            payload = json.load(f)
        m = payload.get("metrics", {})
        auc = m.get("roc_auc")
        brier = m.get("brier_score")
        return (float(auc) if auc is not None else None, float(brier) if brier is not None else None)
    except Exception:
        return None, None


def shift_performance_correlation(shift_table: pd.DataFrame, perf_table: pd.DataFrame) -> pd.DataFrame:
    """Stage 1.6.5 — merge and compute Spearman rank correlations."""
    merged = shift_table.merge(perf_table, on=["train_sites", "test_site"])
    cols = ["mean_psi", "prevalence_diff", "c2st_auc", "auc_drop", "brier_change"]
    cols = [c for c in cols if c in merged.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    return merged[cols].corr(method="spearman")


def run_shift_diagnostics(
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    include_kaggle_uci: bool = True,
    psi_bins: int = 10,
    seed: int = 42,
) -> dict:
    """
    Run Stage 1.6 over all available external experiments and write artifacts under outputs/shift/.
    Returns a dict with 'shift_table', 'perf_table', and 'correlation' DataFrames (as pandas).
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    shift_root = output_dir / "shift"
    shift_root.mkdir(parents=True, exist_ok=True)

    config = preprocessing.load_preprocessing_config(data_dir / "ingestion_report.json")

    exps = _discover_external_experiments(output_dir, include_kaggle_uci=include_kaggle_uci)
    if not exps:
        return {"shift_table": pd.DataFrame(), "perf_table": pd.DataFrame(), "correlation": pd.DataFrame()}

    shift_rows = []
    perf_rows = []

    for exp in exps:
        # Load raw data for train/test sites
        try:
            train_dfs = [_load_clean(s, data_dir) for s in exp.train_sites]
            train_df = pd.concat(train_dfs, ignore_index=True)
            test_df = _load_clean(exp.test_site, data_dir)
        except FileNotFoundError:
            continue

        features = [c for c in exp.features_used if c in train_df.columns and c in test_df.columns]
        if not features:
            continue

        # 1.6.1 — prevalence shift
        prev = prevalence_shift(train_df, test_df)

        # 1.6.2 — per-feature covariate shift and PSI
        feat_rows = []
        psis = []
        for col in features:
            col_type = _feature_type(col)
            uv = univariate_shift(train_df[col], test_df[col], col_type)
            col_psi = psi(train_df[col], test_df[col], bins=psi_bins)
            psis.append(col_psi)
            feat_rows.append(
                {
                    "feature": col,
                    "type": col_type,
                    "test": uv["test"],
                    "statistic": uv["statistic"],
                    "p_value": uv["p_value"],
                    "psi": col_psi,
                }
            )
        feature_shift_df = pd.DataFrame(feat_rows)

        # 1.6.3 — missingness shift
        miss_df = missingness_shift(train_df, test_df, features)
        feature_shift_df = feature_shift_df.merge(miss_df, on="feature", how="left")

        # 1.6.4 — C2ST on preprocessed numeric features
        # Use LR-style preprocessing to avoid depending on specific model_family.
        pipe = preprocessing.build_preprocessing_pipeline("logistic_regression", features)
        X_train_t, _ = preprocessing.fit_transform_train(pipe, train_df[features], train_df["target"])
        X_test_t = preprocessing.transform_test(pipe, test_df[features])
        X_train_t = np.asarray(X_train_t)
        X_test_t = np.asarray(X_test_t)
        c2st_auc = classifier_two_sample_test(X_train_t, X_test_t, seed=seed)

        # Aggregate shift metrics for this pair
        mean_psi = float(np.nanmean(psis)) if psis else float("nan")

        train_key = "+".join(sorted(exp.train_sites))
        pair_key = f"{train_key}__to__{exp.test_site}"
        pair_dir = shift_root / pair_key
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Save per-pair artifacts
        with open(pair_dir / "shift_diagnostics.json", "w") as f:
            json.dump(
                {
                    "train_sites": exp.train_sites,
                    "test_site": exp.test_site,
                    "model": exp.model,
                    "experiment_type": exp.experiment_type,
                    "prevalence_shift": prev,
                    "mean_psi": mean_psi,
                    "c2st_auc": c2st_auc,
                },
                f,
                indent=2,
                default=str,
            )
        feature_shift_df.to_parquet(pair_dir / "feature_shift.parquet", index=False)

        shift_rows.append(
            {
                "train_sites": train_key,
                "test_site": exp.test_site,
                "model": exp.model,
                "experiment_type": exp.experiment_type,
                "mean_psi": mean_psi,
                "prevalence_diff": prev["absolute_diff"],
                "c2st_auc": c2st_auc,
            }
        )

        # Performance deltas vs internal baseline (full-feature) — only when that baseline exists
        if exp.experiment_type == "external_uci":
            baseline_auc, baseline_brier = _load_internal_baseline_auc_brier(output_dir, exp.test_site, exp.model)
        else:
            # For Kaggle↔UCI we skip baseline linkage for now (penalties are handled in Stage 1.5).
            baseline_auc, baseline_brier = None, None

        if baseline_auc is not None and exp.roc_auc is not None:
            auc_drop = float(exp.roc_auc) - float(baseline_auc)
        else:
            auc_drop = None
        if baseline_brier is not None and exp.brier is not None:
            brier_change = float(exp.brier) - float(baseline_brier)
        else:
            brier_change = None

        perf_rows.append(
            {
                "train_sites": train_key,
                "test_site": exp.test_site,
                "model": exp.model,
                "experiment_type": exp.experiment_type,
                "auc_drop": auc_drop,
                "brier_change": brier_change,
            }
        )

    shift_table = pd.DataFrame(shift_rows)
    perf_table = pd.DataFrame(perf_rows)
    corr = shift_performance_correlation(shift_table, perf_table)

    if not corr.empty:
        corr.to_parquet(shift_root / "shift_performance_correlation.parquet")

    return {"shift_table": shift_table, "perf_table": perf_table, "correlation": corr}


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "data"
    output_dir = base / "outputs"
    if not (data_dir / "ingestion_report.json").exists():
        raise SystemExit("Run ingestion first: python -m src.ingest")
    res = run_shift_diagnostics(data_dir=data_dir, output_dir=output_dir, include_kaggle_uci=True)
    print(
        f"Stage 1.6 completed. "
        f"{len(res['shift_table'])} external experiments processed; "
        f"correlation shape={res['correlation'].shape}."
    )

