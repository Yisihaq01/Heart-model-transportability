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


if __name__ == "__main__":
    # CLI to execute Stage 2.1 directly:
    #   python -m src.calibration
    base = Path(__file__).resolve().parent.parent
    outputs_dir = base / "outputs"
    if not outputs_dir.exists():
        raise SystemExit(f"No outputs directory found at {outputs_dir}. Run Phase 1 first.")
    results = assess_calibration(outputs_dir)
    print(f"Stage 2.1 calibration assessment completed for {len(results)} experiments.")

