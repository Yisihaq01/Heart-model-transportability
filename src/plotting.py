"""
Evaluation figures (eval_plan §4).
All matplotlib/seaborn figure functions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def _save_fig(fig: plt.Figure, path: Path, formats: list[str], dpi: int = 300) -> None:
    for fmt in formats:
        out = path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves_by_model(
    site: str,
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    output_path: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """ROC curve overlay: one curve per model (F1, F2)."""
    config = config or {}
    palette = config.get("color_palette", {})
    formats = config.get("figure_format", ["pdf", "png"])
    dpi = config.get("figure_dpi", 300)

    fig, ax = plt.subplots(figsize=(6, 5))
    for model, (fpr, tpr, auc) in curves.items():
        color = palette.get(model, None)
        ax.plot(fpr, tpr, label=f"{model.upper()} (AUC={auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {site}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save_fig(fig, output_path, formats, dpi)


def plot_auc_heatmap(
    pivot: pd.DataFrame,
    model_key: str,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """AUC heatmap: train × test (F3)."""
    config = config or {}
    cmap = config.get("heatmap_cmap", "RdYlGn")
    vmin = config.get("heatmap_vmin", 0.5)
    vmax = config.get("heatmap_vmax", 1.0)
    formats = config.get("figure_format", ["pdf", "png"])
    dpi = config.get("figure_dpi", 300)

    fig, ax = plt.subplots(figsize=(8, 6))
    if HAS_SEABORN:
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            ax=ax,
        )
    else:
        im = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center")
        plt.colorbar(im, ax=ax)
    ax.set_title(f"External ROC-AUC: {model_key.upper()}")
    out_path = output_dir / f"auc_heatmap_{model_key}"
    _save_fig(fig, out_path, formats, dpi)


def plot_reliability_diagram(
    fraction_pos: np.ndarray,
    mean_pred: np.ndarray,
    title: str,
    output_path: Path | None = None,
    ax: plt.Axes | None = None,
    config: dict[str, Any] | None = None,
) -> plt.Figure | None:
    """Calibration curve: observed vs predicted (F6–F8)."""
    config = config or {}
    formats = config.get("figure_format", ["pdf", "png"])
    dpi = config.get("figure_dpi", 300)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    ax.plot(mean_pred, fraction_pos, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.fill_between(mean_pred, fraction_pos, mean_pred, alpha=0.15, color="red")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if output_path:
        _save_fig(fig, output_path, formats, dpi)
        return None
    return fig


def plot_auc_drop_bars(
    delta_df: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """AUC drop bar chart: test site × model (F4)."""
    config = config or {}
    palette = config.get("color_palette", {})
    formats = config.get("figure_format", ["pdf", "png"])
    dpi = config.get("figure_dpi", 300)

    if delta_df.empty or "auc_delta" not in delta_df.columns:
        return

    # Aggregate by test_site × model (mean auc_delta across train sources)
    agg = (
        delta_df.groupby(["test_site", "model"])["auc_delta"]
        .mean()
        .reset_index()
    )

    sites = agg["test_site"].unique()
    models = agg["model"].unique()
    fig, ax = plt.subplots(figsize=(max(10, len(sites) * 1.5), 6))
    x = np.arange(len(sites))
    width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        subset = agg[agg["model"] == model]
        # Align by test_site
        vals = subset.set_index("test_site").reindex(sites)["auc_delta"].values
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            np.nan_to_num(vals, nan=0),
            width,
            label=model.upper(),
            color=palette.get(model),
        )
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=45, ha="right")
    ax.set_ylabel("AUC Drop (Internal − External)")
    ax.set_title("External Validation AUC Degradation")
    ax.legend()
    ax.set_xlim(-0.5, len(x) - 0.5)
    _save_fig(fig, output_dir / "auc_drop_bars", formats, dpi)


def plot_ece_comparison(
    ece_data: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """ECE before/after recalibration (F9)."""
    config = config or {}
    formats = config.get("figure_format", ["pdf", "png"])
    dpi = config.get("figure_dpi", 300)

    if ece_data.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ece_data))
    width = 0.35
    ax.bar(x - width / 2, ece_data["ece_before"], width, label="Before")
    ax.bar(x + width / 2, ece_data["ece_after"], width, label="After")
    ax.set_xticks(x)
    ax.set_xticklabels(ece_data.get("experiment_id", range(len(ece_data))), rotation=45, ha="right")
    ax.set_ylabel("ECE")
    ax.set_title("Recalibration: ECE Before vs After")
    ax.legend()
    _save_fig(fig, output_dir / "ece_comparison", formats, dpi)


def plot_psi_heatmap(
    psi_matrix: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """PSI heatmap: feature × site pair (F11)."""
    config = config or {}
    formats = config.get("figure_format", ["pdf", "png"])
    dpi = config.get("figure_dpi", 300)

    if psi_matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    if HAS_SEABORN:
        sns.heatmap(
            psi_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            linewidths=0.5,
            ax=ax,
        )
    else:
        im = ax.imshow(psi_matrix.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(psi_matrix.columns)))
        ax.set_xticklabels(psi_matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(psi_matrix.index)))
        ax.set_yticklabels(psi_matrix.index)
        for i in range(len(psi_matrix.index)):
            for j in range(len(psi_matrix.columns)):
                v = psi_matrix.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center")
        plt.colorbar(im, ax=ax)
    ax.set_title("PSI by Feature × Site Pair")
    _save_fig(fig, output_dir / "psi_heatmap", formats, dpi)


def plot_shift_vs_performance(
    shift_perf_df: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """3-panel scatter: mean PSI vs AUC drop, prevalence vs Brier, C2ST vs AUC (F13)."""
    from scipy.stats import spearmanr

    config = config or {}
    formats = config.get("figure_format", ["pdf", "png"])
    dpi = config.get("figure_dpi", 300)

    df = shift_perf_df.dropna(subset=["mean_psi", "auc_drop"], how="all")
    if df.empty or "auc_drop" not in df.columns or df["auc_drop"].isna().all():
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    scatter_pairs = [
        ("mean_psi", "auc_drop", "Mean PSI", "AUC Drop"),
        ("prevalence_diff", "brier_change", "Prevalence Shift", "Brier Δ"),
        ("c2st_auc", "roc_auc", "C2ST AUC", "External ROC-AUC"),
    ]
    for ax, (x_col, y_col, xlabel, ylabel) in zip(axes, scatter_pairs):
        valid = df[[x_col, y_col]].dropna()
        if valid.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            continue
        hue_col = "test_site" if "test_site" in df.columns else None
        style_col = "model" if "model" in df.columns else None
        if HAS_SEABORN:
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, style=style_col, s=80, ax=ax)
        else:
            ax.scatter(df[x_col], df[y_col], s=80)
        rho, p = spearmanr(valid[x_col], valid[y_col])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"ρ = {rho:.2f}, p = {p:.3f}")

    fig.tight_layout()
    _save_fig(fig, output_dir / "shift_vs_performance", formats, dpi)


def plot_cfs_penalty_bars(
    cfs_df: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """CFS penalty: full AUC vs CFS AUC (F16)."""
    config = config or {}
    palette = config.get("color_palette", {})
    formats = config.get("figure_format", ["pdf", "png"])
    dpi = config.get("figure_dpi", 300)

    if cfs_df.empty or "full_auc" not in cfs_df.columns:
        return

    valid = cfs_df[cfs_df["full_auc"].notna() & cfs_df["cfs_auc"].notna()]
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(valid["site"].unique()))
    width = 0.2
    for i, model in enumerate(valid["model"].unique()):
        subset = valid[valid["model"] == model]
        offset = (i - 1.5) * width
        ax.bar(
            x + offset,
            subset["auc_drop"],
            width,
            label=f"{model.upper()} Δ",
            color=palette.get(model),
        )
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(valid["site"].unique(), rotation=45, ha="right")
    ax.set_ylabel("AUC Drop (Full − CFS)")
    ax.set_title("CFS Feature Restriction Penalty")
    ax.legend()
    _save_fig(fig, output_dir / "cfs_penalty_bars", formats, dpi)


def plot_pr_curves_by_model(
    site: str,
    curves: dict[str, tuple[np.ndarray, np.ndarray, float]],
    output_path: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """PR curve overlay (F5)."""
    config = config or {}
    palette = config.get("color_palette", {})
    formats = config.get("figure_format", ["pdf", "png"])
    dpi = config.get("figure_dpi", 300)

    fig, ax = plt.subplots(figsize=(6, 5))
    for model, (recall, precision, pr_auc) in curves.items():
        color = palette.get(model, None)
        ax.plot(recall, precision, label=f"{model.upper()} (AP={pr_auc:.3f})", color=color)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curves — {site}")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _save_fig(fig, output_path, formats, dpi)


def _brier_decomposition(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Murphy decomposition: Brier = REL - RES + UNC."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    if n == 0:
        return {"reliability": 0.0, "resolution": 0.0, "uncertainty": 0.0, "brier": 0.0}
    o_bar = float(y_true.mean())
    unc = o_bar * (1 - o_bar)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    rel, res = 0.0, 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        nk = mask.sum()
        ok = y_true[mask].sum()
        pk = (lo + hi) / 2
        o_k = ok / nk
        rel += (nk / n) * (o_k - pk) ** 2
        res += (nk / n) * (o_k - o_bar) ** 2
    brier = float(np.mean((y_prob - y_true) ** 2))
    return {"reliability": rel, "resolution": res, "uncertainty": unc, "brier": brier}


def plot_brier_decomposition(
    decomp_df: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Stacked bar: Brier = reliability + resolution + uncertainty (F10)."""
    config = config or {}
    formats = config.get("figure_format", ["pdf", "png"])
    dpi = config.get("figure_dpi", 300)

    if decomp_df.empty or "reliability" not in decomp_df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(decomp_df))
    width = 0.6
    rel = decomp_df["reliability"].values
    res = decomp_df["resolution"].values
    unc = decomp_df["uncertainty"].values
    bottom = rel
    ax.bar(x, rel, width, label="Reliability", color="tab:red", alpha=0.8)
    ax.bar(x, unc - res, width, bottom=bottom, label="Uncertainty − Resolution", color="tab:gray", alpha=0.6)
    ax.set_xticks(x)
    labels = decomp_df.get("experiment_id", [str(i) for i in range(len(decomp_df))])
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Brier decomposition")
    ax.set_title("Brier Score Decomposition (Reliability + Uncertainty − Resolution)")
    ax.legend()
    _save_fig(fig, output_dir / "brier_decomposition", formats, dpi)


def plot_feature_distributions_worst_pair(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    pair_name: str,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Overlaid histograms/KDE: train vs test for worst pair (F12)."""
    config = config or {}
    formats = config.get("figure_format", ["pdf", "png"])
    dpi = config.get("figure_dpi", 300)

    if not features:
        return

    n_feat = len(features)
    n_cols = min(3, n_feat)
    n_rows = (n_feat + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_feat == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, feat in enumerate(features):
        r, c = idx // n_cols, idx % n_cols
        ax = axes[r, c]
        if feat in train_df.columns and feat in test_df.columns:
            tr = train_df[feat].dropna()
            te = test_df[feat].dropna()
            if len(tr) > 0 and len(te) > 0:
                if HAS_SEABORN:
                    sns.kdeplot(tr, ax=ax, label="Train", color="tab:blue")
                    sns.kdeplot(te, ax=ax, label="Test", color="tab:orange")
                else:
                    ax.hist(tr, bins=15, alpha=0.5, label="Train", color="tab:blue", density=True)
                    ax.hist(te, bins=15, alpha=0.5, label="Test", color="tab:orange", density=True)
        ax.set_title(feat)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("")

    for idx in range(n_feat, n_rows * n_cols):
        r, c = idx // n_cols, idx % n_cols
        axes[r, c].set_visible(False)

    fig.suptitle(f"Feature Distributions — {pair_name} (Worst AUC Drop)", fontsize=12)
    fig.tight_layout()
    safe_name = pair_name.replace(" ", "_").replace("+", "_").replace("→", "_to_").replace("__", "_")
    _save_fig(fig, output_dir / f"feature_distributions_{safe_name}", formats, dpi)


def plot_missingness_heatmap(
    missingness_matrix: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Heatmap: feature × site missing % (F14)."""
    config = config or {}
    formats = config.get("figure_format", ["pdf", "png"])
    dpi = config.get("figure_dpi", 300)

    if missingness_matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    if HAS_SEABORN:
        sns.heatmap(
            missingness_matrix,
            annot=True,
            fmt=".1f",
            cmap="YlOrRd",
            vmin=0,
            vmax=100,
            linewidths=0.5,
            ax=ax,
        )
    else:
        im = ax.imshow(missingness_matrix.values, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(missingness_matrix.columns)))
        ax.set_xticklabels(missingness_matrix.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(missingness_matrix.index)))
        ax.set_yticklabels(missingness_matrix.index)
        for i in range(len(missingness_matrix.index)):
            for j in range(len(missingness_matrix.columns)):
                ax.text(j, i, f"{missingness_matrix.values[i, j]:.1f}", ha="center", va="center")
        plt.colorbar(im, ax=ax)
    ax.set_title("Missingness % by Feature × Site")
    _save_fig(fig, output_dir / "missingness_heatmap", formats, dpi)


def plot_c2st_vs_auc(
    df: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Scatter: C2ST AUC vs external ROC-AUC (F15)."""
    config = config or {}
    formats = config.get("figure_format", ["pdf", "png"])
    dpi = config.get("figure_dpi", 300)

    valid = df.dropna(subset=["c2st_auc", "roc_auc"])
    if valid.empty or len(valid) < 2:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    hue_col = "test_site" if "test_site" in df.columns else None
    if HAS_SEABORN and hue_col:
        sns.scatterplot(data=valid, x="c2st_auc", y="roc_auc", hue=hue_col, s=80, ax=ax)
    else:
        ax.scatter(valid["c2st_auc"], valid["roc_auc"], s=80)
    ax.set_xlabel("C2ST AUC (train vs test discriminability)")
    ax.set_ylabel("External ROC-AUC")
    ax.set_title("C2ST vs External Validation AUC")
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.5)
    ax.axvline(0.5, color="k", linestyle="--", alpha=0.5)
    _save_fig(fig, output_dir / "c2st_vs_auc", formats, dpi)


def plot_size_matched_comparison(
    comparison_df: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Bar chart: full AUC vs subsampled mean AUC with error bars (F17)."""
    config = config or {}
    formats = config.get("figure_format", ["pdf", "png"])
    dpi = config.get("figure_dpi", 300)

    if comparison_df.empty or "full_auc" not in comparison_df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    df = comparison_df
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["full_auc"], width, label="Full training", color="tab:blue", alpha=0.8)
    ax.bar(
        x + width / 2,
        df["subsampled_mean"],
        width,
        yerr=df.get("subsampled_std", 0),
        label="Size-matched (K subsamples)",
        color="tab:orange",
        alpha=0.8,
        capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df.get("pair_model", range(len(df))), rotation=45, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Size-Matched Comparison: Full vs Subsampled Training")
    ax.legend()
    ax.set_ylim(0, 1.05)
    _save_fig(fig, output_dir / "size_matched_boxplot", formats, dpi)


def plot_effective_cfs_count(
    pair_counts: pd.DataFrame,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> None:
    """Bar chart: effective CFS feature count per site pair (F18)."""
    config = config or {}
    formats = config.get("figure_format", ["pdf", "png"])
    dpi = config.get("figure_dpi", 300)

    if pair_counts.empty or "pair" not in pair_counts.columns or "n_features" not in pair_counts.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    pairs = pair_counts["pair"].values
    counts = pair_counts["n_features"].values
    x = np.arange(len(pairs))
    ax.bar(x, counts, color="tab:green", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=45, ha="right")
    ax.set_ylabel("Effective CFS feature count")
    ax.set_title("Usable Features per Site Pair (after 40% missingness filter)")
    _save_fig(fig, output_dir / "effective_cfs_features", formats, dpi)
