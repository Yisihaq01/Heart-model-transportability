"""
Stage 2.1 — Calibration Assessment (Pre-Recalibration).

Compute Brier score, ECE, MCE, and calibration curves for all Phase 1 experiments
using their saved predictions, and persist results under outputs/calibration/before/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import json

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as PlattLR


def compute_calibration_metrics(y_true, y_prob, n_bins: int = 10) -> Dict[str, Any]:
    """
    Stage 2.1.1: Compute Brier score, ECE, MCE, and calibration curve details.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels {0,1}.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    n_bins : int, default=10
        Number of equal-width bins on [0,1] for ECE/MCE and calibration curve.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    if n == 0:
        return {
            "brier_score": None,
            "ece": None,
            "mce": None,
            "bin_details": [],
            "calibration_curve": {"fraction_pos": [], "mean_predicted": []},
        }

    brier = float(brier_score_loss(y_true, y_prob))

    # ECE and MCE (equal-width bins on [0,1])
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    bin_details: List[Dict[str, Any]] = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        bin_size = int(mask.sum())
        gap = abs(bin_acc - bin_conf)
        ece += gap * bin_size / n
        mce = max(mce, gap)
        bin_details.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "acc": bin_acc,
                "conf": bin_conf,
                "n": bin_size,
            }
        )

    # Calibration curve points (uniform bins for consistency with plan)
    fraction_pos, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    return {
        "brier_score": brier,
        "ece": float(ece),
        "mce": float(mce),
        "bin_details": bin_details,
        "calibration_curve": {
            "fraction_pos": fraction_pos.tolist(),
            "mean_predicted": mean_predicted.tolist(),
        },
    }


def _iter_phase1_experiments(outputs_dir: Path) -> Iterable[Dict[str, Any]]:
    """
    Yield dicts with metadata + y_true/y_prob for every Phase 1 experiment that has predictions.

    We look under:
      - outputs/internal/
      - outputs/external_uci/
      - outputs/external_kaggle_uci/
    and rely on results.json for experiment metadata.
    """
    # Internal validation (Stage 1.3)
    internal_root = outputs_dir / "internal"
    if internal_root.exists():
        for site_dir in internal_root.iterdir():
            if not site_dir.is_dir():
                continue
            for model_dir in site_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                pred_path = model_dir / "predictions.parquet"
                results_path = model_dir / "results.json"
                if not pred_path.exists() or not results_path.exists():
                    continue
                yield {
                    "experiment_type": "internal",
                    "site": site_dir.name,
                    "model": model_dir.name,
                    "predictions_path": pred_path,
                    "results_path": results_path,
                }

    # UCI external matrix (Stage 1.4)
    ext_uci_root = outputs_dir / "external_uci"
    if ext_uci_root.exists():
        for pair_dir in ext_uci_root.iterdir():
            if not pair_dir.is_dir():
                continue
            for model_dir in pair_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                pred_path = model_dir / "predictions.parquet"
                results_path = model_dir / "results.json"
                if not pred_path.exists() or not results_path.exists():
                    continue
                # Train/test metadata will be read from results.json
                yield {
                    "experiment_type": "external_uci",
                    "train_test_key": pair_dir.name,
                    "model": model_dir.name,
                    "predictions_path": pred_path,
                    "results_path": results_path,
                }

    # Kaggle ↔ UCI stress tests (Stage 1.5)
    ext_kaggle_root = outputs_dir / "external_kaggle_uci"
    if ext_kaggle_root.exists():
        for variant_dir in ext_kaggle_root.iterdir():
            if not variant_dir.is_dir():
                continue
            for pair_dir in variant_dir.iterdir():
                if not pair_dir.is_dir():
                    continue
                for model_dir in pair_dir.iterdir():
                    if not model_dir.is_dir():
                        continue
                    pred_path = model_dir / "predictions.parquet"
                    results_path = model_dir / "results.json"
                    if not pred_path.exists() or not results_path.exists():
                        continue
                    yield {
                        "experiment_type": "external_kaggle_uci",
                        "variant": variant_dir.name,
                        "train_test_key": pair_dir.name,
                        "model": model_dir.name,
                        "predictions_path": pred_path,
                        "results_path": results_path,
                    }


def _calibration_flags(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stage 2.1.2 — convenience flags based on ECE/MCE/Brier.
    """
    ece = metrics.get("ece")
    mce = metrics.get("mce")
    flags: Dict[str, Any] = {}
    if ece is not None:
        flags["ece_moderate"] = bool(ece > 0.05)
        flags["ece_severe"] = bool(ece > 0.10)
    if mce is not None:
        flags["mce_large"] = bool(mce > 0.15)
    return flags


def assess_calibration(
    outputs_dir: str | Path = "outputs",
    n_bins: int = 10,
    save: bool = True,
) -> List[Dict[str, Any]]:
    """
    Stage 2.1 main entrypoint: assess calibration for all Phase 1 experiments.

    Parameters
    ----------
    outputs_dir : str or Path, default="outputs"
        Root of experiment outputs (internal/, external_uci/, external_kaggle_uci/).
    n_bins : int, default=10
        Number of bins for ECE/MCE and calibration curves.
    save : bool, default=True
        When True, write per-experiment JSONs and an overall summary JSON under
        outputs/calibration/before/.
    """
    outputs_path = Path(outputs_dir)
    calib_root = outputs_path / "calibration" / "before"
    if save:
        calib_root.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []

    for exp in _iter_phase1_experiments(outputs_path):
        pred_df = pd.read_parquet(exp["predictions_path"])
        if "y_true" not in pred_df.columns or "y_prob" not in pred_df.columns:
            continue
        y_true = pred_df["y_true"].to_numpy()
        y_prob = pred_df["y_prob"].to_numpy()

        calib_metrics = compute_calibration_metrics(y_true, y_prob, n_bins=n_bins)

        with open(exp["results_path"]) as f:
            meta = json.load(f)

        record: Dict[str, Any] = {
            "experiment_type": exp["experiment_type"],
            "model": exp["model"],
            "calibration_metrics": calib_metrics,
            "n": int(len(y_true)),
        }
        # Attach key metadata based on experiment type
        if exp["experiment_type"] == "internal":
            record["site"] = meta.get("site")
        elif exp["experiment_type"] == "external_uci":
            record["train_sites"] = meta.get("train_sites")
            record["test_site"] = meta.get("test_site")
        elif exp["experiment_type"] == "external_kaggle_uci":
            record["train_site"] = meta.get("train_site")
            record["test_site"] = meta.get("test_site")
            record["variant"] = meta.get("variant", exp.get("variant"))

        record["flags"] = _calibration_flags(calib_metrics)
        all_results.append(record)

        if save:
            # Build a reasonably unique filename
            if record["experiment_type"] == "internal":
                fname = f"internal__{record.get('site')}__{record['model']}.json"
            elif record["experiment_type"] == "external_uci":
                train_key = "+".join(sorted(record.get("train_sites") or []))
                fname = f"external_uci__{train_key}__to__{record.get('test_site')}__{record['model']}.json"
            else:  # external_kaggle_uci
                variant = record.get("variant") or "cfs"
                fname = f"external_kaggle_uci__{variant}__{record.get('train_site')}__to__{record.get('test_site')}__{record['model']}.json"
            with open(calib_root / fname, "w") as f:
                json.dump(record, f, indent=2, default=str)

    if save:
        summary_path = calib_root / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    return all_results


# ---------------------------------------------------------------------------
# Stage 2.2 — Post-Hoc Recalibration (RQ4)
# ---------------------------------------------------------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    """Numerically-stable logit without requiring scipy."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def _expit(x: np.ndarray) -> np.ndarray:
    """Sigmoid / expit."""
    return 1.0 / (1.0 + np.exp(-x))


def calibration_split(
    y_true: np.ndarray,
    small_n_threshold: int = 100,
    seed: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """
    Stage 2.2.2 — Split target-site test predictions into calibration vs evaluation.

    - Standard: if N >= small_n_threshold → single 50/50 stratified split.
    - Small site: if N < small_n_threshold → 3-fold CV on test set
      (fit recalibrator on 2 folds, evaluate on 1, rotated).

    Returns (splits, mode) where:
      splits: list of (cal_idx, eval_idx)
      mode: "holdout" or "cv"
    """
    y_true = np.asarray(y_true)
    n = len(y_true)
    indices = np.arange(n)

    if n < small_n_threshold and n >= 3:
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for train_idx, test_idx in skf.split(indices, y_true):
            cal_idx = train_idx
            eval_idx = test_idx
            splits.append((cal_idx, eval_idx))
        return splits, "cv"

    # Fallback / standard: single 50/50 split
    rng = np.random.RandomState(seed)
    perm = rng.permutation(indices)
    split = n // 2
    cal_idx = perm[:split]
    eval_idx = perm[split:]
    return [(cal_idx, eval_idx)], "holdout"


def recalibrate_platt(y_prob_raw: np.ndarray, y_true_cal: np.ndarray) -> PlattLR:
    """Fit Platt scaling (logistic regression on predicted prob → true label)."""
    y_prob_raw = np.asarray(y_prob_raw).reshape(-1, 1)
    y_true_cal = np.asarray(y_true_cal)
    lr = PlattLR()
    lr.fit(y_prob_raw, y_true_cal)
    return lr


def recalibrate_isotonic(y_prob_raw: np.ndarray, y_true_cal: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression on predicted probability → true label."""
    y_prob_raw = np.asarray(y_prob_raw)
    y_true_cal = np.asarray(y_true_cal)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(y_prob_raw, y_true_cal)
    return iso


def recalibrate_temperature(y_prob_raw: np.ndarray, y_true_cal: np.ndarray) -> float:
    """
    Fit temperature scaling (single scalar T) via bounded grid search.

    Uses negative log-likelihood on the calibration set as the objective:
        calibrated_prob = sigmoid(logit(raw_prob) / T)
    """
    y_prob_raw = np.asarray(y_prob_raw)
    y_true_cal = np.asarray(y_true_cal)
    logits = _logit(y_prob_raw)

    def nll(T: float) -> float:
        scaled = _expit(logits / T)
        # Clip for numerical stability inside logs
        scaled = np.clip(scaled, 1e-6, 1.0 - 1e-6)
        return float(
            -np.mean(
                y_true_cal * np.log(scaled)
                + (1.0 - y_true_cal) * np.log(1.0 - scaled)
            )
        )

    # Coarse grid search over reasonable temperature range
    grid = np.linspace(0.1, 10.0, 100)
    losses = [nll(t) for t in grid]
    best_idx = int(np.argmin(losses))
    return float(grid[best_idx])


def fit_recalibrator(method: str, y_prob_cal: np.ndarray, y_true_cal: np.ndarray):
    """Dispatch helper for Stage 2.2.3."""
    method = method.lower()
    if method == "platt":
        return recalibrate_platt(y_prob_cal, y_true_cal)
    if method == "isotonic":
        return recalibrate_isotonic(y_prob_cal, y_true_cal)
    if method == "temperature":
        return recalibrate_temperature(y_prob_cal, y_true_cal)
    raise ValueError(f"Unknown recalibration method: {method}")


def apply_recalibrator(recalibrator, method: str, y_prob_raw: np.ndarray) -> np.ndarray:
    """Apply fitted recalibrator to raw probabilities."""
    y_prob_raw = np.asarray(y_prob_raw)
    method = method.lower()
    if method == "platt":
        # LogisticRegression returns prob for both classes; take positive class
        return recalibrator.predict_proba(y_prob_raw.reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return recalibrator.predict(y_prob_raw)
    if method == "temperature":
        T = float(recalibrator)
        logits = _logit(y_prob_raw)
        return _expit(logits / T)
    raise ValueError(f"Unknown recalibration method: {method}")


def _aggregate_calibration_across_folds(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate calibration metrics across CV folds.

    We average scalar fields (brier_score, ece, mce) and keep bin_details /
    calibration_curve from the first fold as a representative shape.
    """
    if not metrics_list:
        return {
            "brier_score": None,
            "ece": None,
            "mce": None,
            "bin_details": [],
            "calibration_curve": {"fraction_pos": [], "mean_predicted": []},
        }
    if len(metrics_list) == 1:
        return metrics_list[0]

    keys = ["brier_score", "ece", "mce"]
    agg: Dict[str, Any] = {}
    for k in keys:
        vals = [m.get(k) for m in metrics_list if m.get(k) is not None]
        agg[k] = float(np.mean(vals)) if vals else None

    # Take curve and bin detail from first fold (for plotting / inspection)
    agg["bin_details"] = metrics_list[0].get("bin_details", [])
    agg["calibration_curve"] = metrics_list[0].get(
        "calibration_curve",
        {"fraction_pos": [], "mean_predicted": []},
    )
    return agg


def run_recalibration(
    outputs_dir: str | Path = "outputs",
    methods: List[str] | None = None,
    n_bins: int = 10,
    small_n_threshold: int = 100,
    seed: int = 42,
    save: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Stage 2.2 main entrypoint: post-hoc recalibration for external experiments.

    Reads predictions from:
      - outputs/external_uci/
      - outputs/external_kaggle_uci/

    For each experiment and method in {platt, isotonic, temperature}:
      - Split target-site predictions into calibration vs evaluation.
      - Fit recalibrator on calibration subset.
      - Compute calibration metrics before/after on evaluation subset.
      - Persist results under outputs/calibration/recalibration/{method}/.
    """
    outputs_path = Path(outputs_dir)
    methods = methods or ["platt", "isotonic", "temperature"]

    # Prepare output roots
    method_roots: Dict[str, Path] = {}
    if save:
        for m in methods:
            root = outputs_path / "calibration" / "recalibration" / m
            root.mkdir(parents=True, exist_ok=True)
            method_roots[m] = root

    all_results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in methods}

    for exp in _iter_phase1_experiments(outputs_path):
        if exp["experiment_type"] not in ("external_uci", "external_kaggle_uci"):
            continue

        pred_df = pd.read_parquet(exp["predictions_path"])
        if "y_true" not in pred_df.columns or "y_prob" not in pred_df.columns:
            continue
        y_true = pred_df["y_true"].to_numpy()
        y_prob = pred_df["y_prob"].to_numpy()
        if len(y_true) == 0:
            continue

        with open(exp["results_path"]) as f:
            meta = json.load(f)

        splits, split_mode = calibration_split(y_true, small_n_threshold=small_n_threshold, seed=seed)

        for method in methods:
            per_fold_before: List[Dict[str, Any]] = []
            per_fold_after: List[Dict[str, Any]] = []
            n_cal_total = 0
            n_eval_total = 0

            for cal_idx, eval_idx in splits:
                if len(cal_idx) == 0 or len(eval_idx) == 0:
                    continue
                y_cal = y_true[cal_idx]
                p_cal = y_prob[cal_idx]
                y_eval = y_true[eval_idx]
                p_eval = y_prob[eval_idx]

                recalibrator = fit_recalibrator(method, p_cal, y_cal)
                p_recal = apply_recalibrator(recalibrator, method, p_eval)

                metrics_before = compute_calibration_metrics(y_eval, p_eval, n_bins=n_bins)
                metrics_after = compute_calibration_metrics(y_eval, p_recal, n_bins=n_bins)

                per_fold_before.append(metrics_before)
                per_fold_after.append(metrics_after)
                n_cal_total += len(cal_idx)
                n_eval_total += len(eval_idx)

            if not per_fold_before:
                continue

            agg_before = _aggregate_calibration_across_folds(per_fold_before)
            agg_after = _aggregate_calibration_across_folds(per_fold_after)

            improvement = {
                "brier_delta": (
                    (agg_before["brier_score"] - agg_after["brier_score"])
                    if agg_before["brier_score"] is not None and agg_after["brier_score"] is not None
                    else None
                ),
                "ece_delta": (
                    (agg_before["ece"] - agg_after["ece"])
                    if agg_before["ece"] is not None and agg_after["ece"] is not None
                    else None
                ),
                "mce_delta": (
                    (agg_before["mce"] - agg_after["mce"])
                    if agg_before["mce"] is not None and agg_after["mce"] is not None
                    else None
                ),
            }

            record: Dict[str, Any] = {
                "experiment_type": exp["experiment_type"],
                "model": exp["model"],
                "method": method,
                "split_mode": split_mode,
                "n_calibration": int(n_cal_total),
                "n_evaluation": int(n_eval_total),
                "metrics_before": agg_before,
                "metrics_after": agg_after,
                "improvement": improvement,
            }

            if exp["experiment_type"] == "external_uci":
                record["train_sites"] = meta.get("train_sites")
                record["test_site"] = meta.get("test_site")
            elif exp["experiment_type"] == "external_kaggle_uci":
                record["train_site"] = meta.get("train_site")
                record["test_site"] = meta.get("test_site")
                record["variant"] = meta.get("variant", exp.get("variant"))

            all_results[method].append(record)

            if save:
                if exp["experiment_type"] == "external_uci":
                    train_key = "+".join(sorted(record.get("train_sites") or []))
                    fname = f"external_uci__{train_key}__to__{record.get('test_site')}__{record['model']}.json"
                else:  # external_kaggle_uci
                    variant = record.get("variant") or "cfs"
                    fname = (
                        f"external_kaggle_uci__{variant}__"
                        f"{record.get('train_site')}__to__{record.get('test_site')}__{record['model']}.json"
                    )
                root = method_roots[method]
                with open(root / fname, "w") as f:
                    json.dump(record, f, indent=2, default=str)

    if save:
        for method, records in all_results.items():
            root = method_roots[method]
            with open(root / "summary.json", "w") as f:
                json.dump(records, f, indent=2, default=str)

    return all_results


# ---------------------------------------------------------------------------
# Stage 2.3 — Lightweight Updating: Intercept & Slope Recalibration (RQ4)
# ---------------------------------------------------------------------------


def logistic_recalibration(
    y_prob_raw: np.ndarray,
    y_true_cal: np.ndarray,
    intercept_only: bool = False,
) -> Dict[str, float]:
    """
    Stage 2.3.1: Fit logistic recalibration model on calibration data.

    logit(p_updated) = a + b * logit(p_original)

    - intercept_only=True  -> b fixed at 1 (prevalence-only correction)
    - intercept_only=False -> both a and b free
    """
    y_prob_raw = np.asarray(y_prob_raw)
    y_true_cal = np.asarray(y_true_cal)
    logits = _logit(y_prob_raw).reshape(-1, 1)

    lr = PlattLR()
    if intercept_only:
        X = np.ones_like(logits)  # only intercept term
    else:
        X = logits

    lr.fit(X, y_true_cal)

    if intercept_only:
        return {"intercept": float(lr.intercept_[0]), "slope": 1.0}
    return {"intercept": float(lr.intercept_[0]), "slope": float(lr.coef_[0][0])}


def apply_logistic_recalibration(
    params: Dict[str, float],
    y_prob_raw: np.ndarray,
) -> np.ndarray:
    """Apply fitted (a, b) intercept/slope parameters to raw probabilities."""
    y_prob_raw = np.asarray(y_prob_raw)
    logits = _logit(y_prob_raw)
    z = params["intercept"] + params["slope"] * logits
    return _expit(z)


def run_lightweight_updating(
    outputs_dir: str | Path = "outputs",
    n_bins: int = 10,
    small_n_threshold: int = 100,
    seed: int = 42,
    save: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Stage 2.3 main entrypoint: logistic intercept/slope updating for external experiments.

    For each external experiment:
      - Use the same calibration_split() logic as Stage 2.2.
      - Fit two variants on the calibration subset:
          * intercept_only  (b = 1 fixed)
          * intercept_slope (a, b both free)
      - Compute calibration metrics before/after on the evaluation subset(s).
      - Persist results under:
          outputs/calibration/updating/intercept_only/
          outputs/calibration/updating/intercept_slope/
    """
    outputs_path = Path(outputs_dir)

    variants = ["intercept_only", "intercept_slope"]
    variant_roots: Dict[str, Path] = {}
    if save:
        for v in variants:
            root = outputs_path / "calibration" / "updating" / v
            root.mkdir(parents=True, exist_ok=True)
            variant_roots[v] = root

    all_results: Dict[str, List[Dict[str, Any]]] = {v: [] for v in variants}

    for exp in _iter_phase1_experiments(outputs_path):
        if exp["experiment_type"] not in ("external_uci", "external_kaggle_uci"):
            continue

        pred_df = pd.read_parquet(exp["predictions_path"])
        if "y_true" not in pred_df.columns or "y_prob" not in pred_df.columns:
            continue
        y_true = pred_df["y_true"].to_numpy()
        y_prob = pred_df["y_prob"].to_numpy()
        if len(y_true) == 0:
            continue

        with open(exp["results_path"]) as f:
            meta = json.load(f)

        splits, split_mode = calibration_split(
            y_true,
            small_n_threshold=small_n_threshold,
            seed=seed,
        )

        for variant in variants:
            per_fold_before: List[Dict[str, Any]] = []
            per_fold_after: List[Dict[str, Any]] = []
            per_fold_params: List[Dict[str, float]] = []
            n_cal_total = 0
            n_eval_total = 0

            intercept_only = variant == "intercept_only"

            for cal_idx, eval_idx in splits:
                if len(cal_idx) == 0 or len(eval_idx) == 0:
                    continue
                y_cal = y_true[cal_idx]
                p_cal = y_prob[cal_idx]
                y_eval = y_true[eval_idx]
                p_eval = y_prob[eval_idx]

                params = logistic_recalibration(
                    p_cal,
                    y_cal,
                    intercept_only=intercept_only,
                )
                p_updated = apply_logistic_recalibration(params, p_eval)

                metrics_before = compute_calibration_metrics(
                    y_eval,
                    p_eval,
                    n_bins=n_bins,
                )
                metrics_after = compute_calibration_metrics(
                    y_eval,
                    p_updated,
                    n_bins=n_bins,
                )

                per_fold_before.append(metrics_before)
                per_fold_after.append(metrics_after)
                per_fold_params.append(params)
                n_cal_total += len(cal_idx)
                n_eval_total += len(eval_idx)

            if not per_fold_before:
                continue

            agg_before = _aggregate_calibration_across_folds(per_fold_before)
            agg_after = _aggregate_calibration_across_folds(per_fold_after)

            improvement = {
                "brier_delta": (
                    (agg_before["brier_score"] - agg_after["brier_score"])
                    if agg_before["brier_score"] is not None
                    and agg_after["brier_score"] is not None
                    else None
                ),
                "ece_delta": (
                    (agg_before["ece"] - agg_after["ece"])
                    if agg_before["ece"] is not None and agg_after["ece"] is not None
                    else None
                ),
                "mce_delta": (
                    (agg_before["mce"] - agg_after["mce"])
                    if agg_before["mce"] is not None and agg_after["mce"] is not None
                    else None
                ),
            }

            # Mean params across folds (diagnostic)
            mean_params: Dict[str, float] = {}
            if per_fold_params:
                for key in ("intercept", "slope"):
                    vals = [p[key] for p in per_fold_params]
                    mean_params[key] = float(np.mean(vals))

            record: Dict[str, Any] = {
                "experiment_type": exp["experiment_type"],
                "model": exp["model"],
                "variant": variant,
                "split_mode": split_mode,
                "n_calibration": int(n_cal_total),
                "n_evaluation": int(n_eval_total),
                "metrics_before": agg_before,
                "metrics_after": agg_after,
                "improvement": improvement,
                "per_fold_params": per_fold_params,
                "mean_params": mean_params,
            }

            if exp["experiment_type"] == "external_uci":
                record["train_sites"] = meta.get("train_sites")
                record["test_site"] = meta.get("test_site")
            elif exp["experiment_type"] == "external_kaggle_uci":
                record["train_site"] = meta.get("train_site")
                record["test_site"] = meta.get("test_site")
                record["experiment_variant"] = meta.get("variant", exp.get("variant"))

            all_results[variant].append(record)

            if save:
                if exp["experiment_type"] == "external_uci":
                    train_key = "+".join(sorted(record.get("train_sites") or []))
                    fname = (
                        f"external_uci__{train_key}__to__"
                        f"{record.get('test_site')}__{record['model']}.json"
                    )
                else:  # external_kaggle_uci
                    vname = record.get("experiment_variant") or "cfs"
                    fname = (
                        f"external_kaggle_uci__{vname}__"
                        f"{record.get('train_site')}__to__"
                        f"{record.get('test_site')}__{record['model']}.json"
                    )
                root = variant_roots[variant]
                with open(root / fname, "w") as f:
                    json.dump(record, f, indent=2, default=str)

    if save:
        for variant, records in all_results.items():
            root = variant_roots[variant]
            with open(root / "summary.json", "w") as f:
                json.dump(records, f, indent=2, default=str)

    return all_results


if __name__ == "__main__":
    # CLI to execute calibration stages directly.
    #   python -m src.calibration              -> Stage 2.1 (before)
    #   python -m src.calibration --stage-2.2 -> Stage 2.2 (recalibration)
    base = Path(__file__).resolve().parent.parent
    outputs_dir = base / "outputs"
    if not outputs_dir.exists():
        raise SystemExit(f"No outputs directory found at {outputs_dir}. Run Phase 1 first.")

    import sys

    if "--stage-2.3" in sys.argv or "--updating" in sys.argv:
        upd = run_lightweight_updating(outputs_dir)
        total = sum(len(v) for v in upd.values())
        print(f"Stage 2.3 lightweight updating completed for {total} variant-experiment combos.")
        print(f"Results written under {outputs_dir / 'calibration' / 'updating'}")
    elif "--stage-2.2" in sys.argv or "--recalibration" in sys.argv:
        rec = run_recalibration(outputs_dir)
        total = sum(len(v) for v in rec.values())
        print(f"Stage 2.2 post-hoc recalibration completed for {total} method-experiment combos.")
        print(f"Results written under {outputs_dir / 'calibration' / 'recalibration'}")
    else:
        results = assess_calibration(outputs_dir)
        print(f"Stage 2.1 calibration assessment completed for {len(results)} experiments.")

