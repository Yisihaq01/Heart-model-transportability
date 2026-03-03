"""
Tests that pipeline_plan.md logic works: resolve_features, metrics, validation splits,
shift diagnostics, calibration helpers, CFS penalty. Uses minimal/synthetic data where needed.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.artifacts import save_experiment

# Pipeline plan Stage C.2 — metrics
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


# --- From pipeline_plan.md: resolve_features, effective_cfs, prevalence_shift, etc. ---
def resolve_features(train_site: str, test_site: str, config: dict) -> list:
    if train_site == test_site:
        return config["full_features"][train_site]
    pair_key = tuple(sorted([train_site, test_site]))
    return config["cfs_features"].get(pair_key, config["cfs_features"]["default"])


def effective_cfs(train_df, test_df, candidate_features: list, missingness_threshold: float = 0.40) -> list:
    usable = []
    for col in candidate_features:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        train_miss = train_df[col].isna().mean()
        test_miss = test_df[col].isna().mean()
        if train_miss < missingness_threshold and test_miss < missingness_threshold:
            usable.append(col)
    return usable


def prevalence_shift(train_df, test_df) -> dict:
    p_train = train_df["target"].mean()
    p_test = test_df["target"].mean()
    return {
        "train_prevalence": float(p_train),
        "test_prevalence": float(p_test),
        "absolute_diff": float(abs(p_train - p_test)),
    }


def bin_cholesterol_uci(chol_series: pd.Series) -> pd.Series:
    # ATP III: <200 desirable=1, 200-239 borderline=2, >=240 high=3
    return pd.cut(
        chol_series,
        bins=[-np.inf, 200, 240, np.inf],
        labels=[1, 2, 3],
        right=False,
    ).astype(float)


def cfs_penalty(full_auc: float, cfs_auc: float) -> dict:
    return {
        "full_auc": full_auc,
        "cfs_auc": cfs_auc,
        "auc_drop": full_auc - cfs_auc,
        "relative_drop_pct": ((full_auc - cfs_auc) / full_auc * 100) if full_auc > 0 else None,
    }


def compute_metrics(y_true, y_prob, y_pred) -> dict:
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "n_test": len(y_true),
        "prevalence": float(np.mean(y_true)),
    }


def internal_split(df, site: str, seed: int = 42):
    """Return list of (train_idx, test_idx). From pipeline_plan §1.3.1."""
    from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

    if site == "switzerland":
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
        return list(cv.split(df.drop(columns=["target"]), df["target"]))
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, stratify=df["target"], random_state=seed
    )
    return [(train_idx, test_idx)]


def bootstrap_metric(y_true, y_pred, metric_fn, B=200, seed=42):
    """Bootstrap CI for a metric that takes (y_true, y_pred). y_pred can be probs or labels."""
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return {"point": metric_fn(y_true, y_pred), "ci_lower": lower, "ci_upper": upper}


def missingness_shift(train_df, test_df, features: list) -> pd.DataFrame:
    rows = []
    for col in features:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        train_miss = train_df[col].isna().mean()
        test_miss = test_df[col].isna().mean()
        rows.append({
            "feature": col,
            "train_miss_pct": round(train_miss * 100, 1),
            "test_miss_pct": round(test_miss * 100, 1),
            "diff_pct": round((test_miss - train_miss) * 100, 1),
        })
    return pd.DataFrame(rows)


# --- Tests ---
@pytest.fixture
def pipeline_config():
    return {
        "full_features": {
            "kaggle": ["age", "sex", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"],
            "cleveland": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"],
        },
        "cfs_features": {
            "default": ["age", "sex", "sys_bp"],
            ("cleveland", "kaggle"): ["age", "sex", "sys_bp"],
        },
    }


@pytest.fixture
def synthetic_binary():
    rng = np.random.RandomState(42)
    n = 200
    y_true = (rng.rand(n) > 0.6).astype(int)
    y_prob = np.clip(y_true.astype(float) + rng.randn(n) * 0.2, 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_prob, y_pred


@pytest.fixture
def synthetic_site_df():
    rng = np.random.RandomState(42)
    n = 100
    return pd.DataFrame({
        "age": rng.uniform(40, 70, n),
        "sex": rng.randint(0, 2, n),
        "target": (rng.rand(n) > 0.5).astype(int),
    })


class TestResolveFeatures:
    def test_same_site_returns_full_features(self, pipeline_config):
        out = resolve_features("kaggle", "kaggle", pipeline_config)
        assert out == pipeline_config["full_features"]["kaggle"]

    def test_cross_site_returns_cfs(self, pipeline_config):
        out = resolve_features("cleveland", "kaggle", pipeline_config)
        assert out == ["age", "sex", "sys_bp"]

    def test_cross_site_default_cfs(self, pipeline_config):
        out = resolve_features("va", "hungary", pipeline_config)
        assert out == ["age", "sex", "sys_bp"]


class TestComputeMetrics:
    def test_returns_expected_keys(self, synthetic_binary):
        y_true, y_prob, y_pred = synthetic_binary
        m = compute_metrics(y_true, y_prob, y_pred)
        assert "roc_auc" in m and "pr_auc" in m and "brier_score" in m
        assert "accuracy" in m and "f1" in m and "n_test" in m

    def test_roc_auc_in_unit_interval(self, synthetic_binary):
        y_true, y_prob, y_pred = synthetic_binary
        m = compute_metrics(y_true, y_prob, y_pred)
        assert 0 <= m["roc_auc"] <= 1

    def test_brier_in_unit_interval(self, synthetic_binary):
        y_true, y_prob, y_pred = synthetic_binary
        m = compute_metrics(y_true, y_prob, y_pred)
        assert 0 <= m["brier_score"] <= 1

    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        y_pred = np.array([0, 1, 0, 1])
        m = compute_metrics(y_true, y_prob, y_pred)
        assert m["accuracy"] == 1.0
        assert m["roc_auc"] == 1.0


class TestInternalSplit:
    def test_returns_list_of_tuples(self, synthetic_site_df):
        splits = internal_split(synthetic_site_df, "va")
        assert isinstance(splits, list)
        assert len(splits) >= 1
        train_idx, test_idx = splits[0]
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(train_idx) + len(test_idx) == len(synthetic_site_df)

    def test_approx_80_20(self, synthetic_site_df):
        splits = internal_split(synthetic_site_df, "va")
        train_idx, test_idx = splits[0]
        n = len(synthetic_site_df)
        assert 0.7 * n <= len(train_idx) <= 0.9 * n
        assert 0.1 * n <= len(test_idx) <= 0.3 * n

    def test_stratified(self, synthetic_site_df):
        splits = internal_split(synthetic_site_df, "va")
        train_idx, test_idx = splits[0]
        train_prev = synthetic_site_df.loc[train_idx, "target"].mean()
        test_prev = synthetic_site_df.loc[test_idx, "target"].mean()
        # Stratification should keep prevalence similar
        assert abs(train_prev - test_prev) < 0.25


class TestBootstrapMetric:
    def test_returns_point_and_ci(self, synthetic_binary):
        y_true, y_prob, y_pred = synthetic_binary
        result = bootstrap_metric(y_true, y_pred, accuracy_score, B=50, seed=42)
        assert "point" in result and "ci_lower" in result and "ci_upper" in result
        assert result["ci_lower"] <= result["point"] <= result["ci_upper"]

    def test_metric_fn_accuracy(self, synthetic_binary):
        y_true, y_pred = synthetic_binary[0], synthetic_binary[2]
        result = bootstrap_metric(y_true, y_pred, accuracy_score, B=30)
        assert 0 <= result["point"] <= 1


class TestPrevalenceShift:
    def test_returns_expected_keys(self, synthetic_site_df):
        train_df = synthetic_site_df.iloc[:60]
        test_df = synthetic_site_df.iloc[60:]
        out = prevalence_shift(train_df, test_df)
        assert "train_prevalence" in out and "test_prevalence" in out and "absolute_diff" in out

    def test_absolute_diff_non_negative(self, synthetic_site_df):
        train_df = synthetic_site_df.iloc[:60]
        test_df = synthetic_site_df.iloc[60:]
        out = prevalence_shift(train_df, test_df)
        assert out["absolute_diff"] >= 0


class TestEffectiveCfs:
    def test_drops_high_missing_column(self):
        train_df = pd.DataFrame({"a": [1, 2, 3], "b": [1, np.nan, np.nan], "target": [0, 1, 0]})
        test_df = pd.DataFrame({"a": [1], "b": [1], "target": [0]})
        out = effective_cfs(train_df, test_df, ["a", "b"], missingness_threshold=0.40)
        # train_df["b"] has 2/3 missing > 0.4 so dropped
        assert "a" in out and "b" not in out

    def test_keeps_low_missing(self):
        train_df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
        test_df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        out = effective_cfs(train_df, test_df, ["a"], missingness_threshold=0.40)
        assert out == ["a"]


class TestBinCholesterolUci:
    def test_bins_three_levels(self):
        s = pd.Series([150, 220, 250])
        out = bin_cholesterol_uci(s)
        assert out.notna().all()
        assert set(out) <= {1.0, 2.0, 3.0}

    def test_cutpoints(self):
        assert bin_cholesterol_uci(pd.Series([199])).iloc[0] == 1   # <200 → 1
        assert bin_cholesterol_uci(pd.Series([200])).iloc[0] == 2   # [200,240) → 2
        assert bin_cholesterol_uci(pd.Series([239])).iloc[0] == 2
        assert bin_cholesterol_uci(pd.Series([240])).iloc[0] == 3   # ≥240 → 3


class TestCfsPenalty:
    def test_auc_drop_positive_when_full_better(self):
        out = cfs_penalty(0.85, 0.75)
        assert out["auc_drop"] == pytest.approx(0.10)
        assert out["relative_drop_pct"] == pytest.approx(100 * 0.10 / 0.85, rel=1e-5)

    def test_zero_full_auc_relative_none(self):
        out = cfs_penalty(0.0, 0.5)
        assert out["relative_drop_pct"] is None


class TestMissingnessShift:
    def test_returns_dataframe_with_expected_columns(self, synthetic_site_df):
        train_df = synthetic_site_df.copy()
        test_df = synthetic_site_df.copy()
        out = missingness_shift(train_df, test_df, ["age", "sex"])
        assert isinstance(out, pd.DataFrame)
        assert "feature" in out.columns and "train_miss_pct" in out.columns and "diff_pct" in out.columns

    def test_no_missing_gives_zero_pct(self, synthetic_site_df):
        out = missingness_shift(synthetic_site_df, synthetic_site_df, ["age", "sex"])
        assert (out["train_miss_pct"] == 0).all() and (out["test_miss_pct"] == 0).all()


class TestPipelinePlanIntegration:
    """End-to-end style: config → resolve → split → metrics → shift."""

    def test_resolve_then_split_then_metrics(self, pipeline_config, synthetic_site_df):
        features = resolve_features("kaggle", "kaggle", pipeline_config)
        # Use only features that exist in synthetic_site_df
        available = [f for f in features if f in synthetic_site_df.columns]
        if not available:
            available = ["age", "sex"]
        df = synthetic_site_df[[*available, "target"]].copy()
        splits = internal_split(df, "va")
        train_idx, test_idx = splits[0]
        X_train, y_train = df.loc[train_idx, available], df.loc[train_idx, "target"]
        X_test, y_test = df.loc[test_idx, available], df.loc[test_idx, "target"]
        # Dummy predictions
        y_prob = np.clip(y_test.values.astype(float) + np.random.RandomState(1).randn(len(y_test)) * 0.1, 0, 1)
        y_pred = (y_prob >= 0.5).astype(int)
        m = compute_metrics(y_test.values, y_prob, y_pred)
        assert m["n_test"] == len(test_idx)
        assert 0 <= m["roc_auc"] <= 1


class TestStageC3ArtifactSerialization:
    """Stage C.3 — Artifact serialization: save_experiment writes results.json, predictions.parquet, predictions.csv, model.joblib, pipeline.joblib."""

    def test_internal_experiment_creates_expected_files(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        with tempfile.TemporaryDirectory() as tmp:
            result = {
                "experiment_type": "internal",
                "site": "cleveland",
                "model": "lr",
                "metrics": {"roc_auc": 0.82, "brier_score": 0.15},
                "predictions": {"y_true": [0, 1, 0], "y_prob": [0.2, 0.8, 0.3], "y_pred": [0, 1, 0]},
                "fitted_model": LogisticRegression().fit([[1], [2], [3]], [0, 1, 0]),
                "fitted_pipeline": Pipeline([("scaler", StandardScaler())]).fit([[1], [2], [3]]),
            }
            path = save_experiment(result, base_dir=tmp)
            assert (path / "results.json").exists()
            assert (path / "predictions.parquet").exists()
            assert (path / "predictions.csv").exists()
            assert (path / "model.joblib").exists()
            assert (path / "pipeline.joblib").exists()
            assert path == Path(tmp) / "internal" / "cleveland" / "lr"

    def test_external_uci_experiment_creates_expected_path_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = {
                "experiment_type": "external_uci",
                "train_sites": ["cleveland", "hungary"],
                "test_site": "va",
                "model": "rf",
                "metrics": {"roc_auc": 0.75},
                "predictions": {"y_true": [0, 1], "y_prob": [0.4, 0.6]},
            }
            path = save_experiment(result, base_dir=tmp)
            assert (path / "results.json").exists()
            assert (path / "predictions.parquet").exists()
            assert (path / "predictions.csv").exists()
            assert "cleveland+hungary" in str(path) and "va" in str(path)
            assert path == Path(tmp) / "external_uci" / "cleveland+hungary__to__va" / "rf"

    def test_external_kaggle_uci_experiment_creates_expected_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = {
                "experiment_type": "external_kaggle_uci",
                "train_site": "kaggle",
                "test_site": "cleveland",
                "model": "xgb",
                "metrics": {"roc_auc": 0.70},
                "predictions": {"y_true": [0, 1, 0], "y_prob": [0.3, 0.7, 0.4]},
            }
            path = save_experiment(result, base_dir=tmp)
            assert (path / "results.json").exists()
            assert (path / "predictions.parquet").exists()
            assert (path / "predictions.csv").exists()
            assert "kaggle__to__cleveland" in str(path)

    def test_results_json_excludes_predictions_and_fitted_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = {
                "experiment_type": "internal",
                "site": "va",
                "model": "lr",
                "predictions": {"y_true": [0], "y_prob": [0.5]},
            }
            path = save_experiment(result, base_dir=tmp)
            import json
            with open(path / "results.json") as f:
                data = json.load(f)
            assert "predictions" not in data
            assert data["site"] == "va" and data["model"] == "lr"
