"""
Stage C.2 — Centralized metric computation.
Used by internal validation (1.3), external validation (1.4, 1.5), and calibration (2.x).
"""
from __future__ import annotations

import numpy as np
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


def compute_metrics(y_true, y_prob, y_pred) -> dict:
    """Point estimates for discrimination, calibration, and classification."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return {}
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = 0.0
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_test": n,
        "prevalence": float(y_true.mean()),
        **compute_ece_mce(y_true, y_prob),
    }


def compute_ece_mce(y_true, y_prob, n_bins: int = 10) -> dict:
    """ECE and MCE for calibration (Stage 2.1 style, reused in 1.3)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    if n == 0:
        return {"ece": 0.0, "mce": 0.0}
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        gap = abs(bin_acc - bin_conf)
        ece += gap * mask.sum() / n
        mce = max(mce, gap)
    return {"ece": float(ece), "mce": float(mce)}


def bootstrap_metric(y_true, y_pred, metric_fn, B: int = 200, seed: int = 42) -> dict:
    """95% bootstrap CI for a single metric (1.3.2). y_pred can be probs for metric_fn(y_true, y_prob)."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    scores = []
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    scores = np.array(scores)
    point = float(metric_fn(y_true, y_pred))
    lower, upper = float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))
    return {"point": point, "ci_lower": lower, "ci_upper": upper}


def bootstrap_metrics(y_true, y_prob, y_pred, B: int = 200, seed: int = 42) -> dict:
    """Bootstrap CIs for roc_auc, pr_auc, brier_score, accuracy, ece (1.3.2)."""
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score

    def safe_auc(yt, yp):
        try:
            return roc_auc_score(yt, yp)
        except ValueError:
            return 0.5

    def safe_pr(yt, yp):
        try:
            return average_precision_score(yt, yp)
        except ValueError:
            return 0.0

    def ece_fn(yt, yp):
        return compute_ece_mce(yt, yp)["ece"]

    return {
        "roc_auc": bootstrap_metric(y_true, y_prob, safe_auc, B=B, seed=seed),
        "pr_auc": bootstrap_metric(y_true, y_prob, safe_pr, B=B, seed=seed),
        "brier_score": bootstrap_metric(y_true, y_prob, brier_score_loss, B=B, seed=seed),
        "accuracy": bootstrap_metric(y_true, y_pred, accuracy_score, B=B, seed=seed),
        "ece": bootstrap_metric(y_true, y_prob, ece_fn, B=B, seed=seed),
    }
