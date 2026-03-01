#!/usr/bin/env python3
"""
Build camera-ready evidence bundle for the paper from outputs/ artifacts.
Generates figures (.png, .svg), tables (.csv, .md), and README.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

# Optional: matplotlib for figures
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
PAPER_READY = OUTPUTS / "paper_ready"


def load_json(p: Path) -> Any:
    with open(p, encoding="utf-8") as f:
        text = f.read()
    # Handle NaN/Inf in JSON (non-standard but common in scientific output)
    text = re.sub(r":\s*NaN\b", ": null", text)
    text = re.sub(r":\s*Infinity\b", ": null", text)
    text = re.sub(r":\s*-Infinity\b", ": null", text)
    return json.loads(text)


def safe_float(x: Any) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    return float(x)


def collect_internal() -> pd.DataFrame:
    rows = []
    internal_dir = OUTPUTS / "internal"
    if not internal_dir.exists():
        return pd.DataFrame()
    for site_dir in internal_dir.iterdir():
        if not site_dir.is_dir():
            continue
        site = site_dir.name
        for model_dir in site_dir.iterdir():
            if not model_dir.is_dir():
                continue
            res_path = model_dir / "results.json"
            if not res_path.exists():
                continue
            try:
                r = load_json(res_path)
                m = r.get("metrics", {})
                bc = r.get("bootstrap_cis", {})
                rows.append({
                    "site": site,
                    "model": r.get("model", model_dir.name),
                    "roc_auc": safe_float(m.get("roc_auc")),
                    "roc_auc_ci_lower": safe_float(bc.get("roc_auc", {}).get("ci_lower")),
                    "roc_auc_ci_upper": safe_float(bc.get("roc_auc", {}).get("ci_upper")),
                    "brier_score": safe_float(m.get("brier_score")),
                    "ece": safe_float(m.get("ece")),
                    "n_test": m.get("n_test"),
                })
            except Exception:
                pass
    return pd.DataFrame(rows)


def collect_external_uci() -> pd.DataFrame:
    rows = []
    ext_dir = OUTPUTS / "external_uci"
    if not ext_dir.exists():
        return pd.DataFrame()
    for pair_dir in ext_dir.iterdir():
        if not pair_dir.is_dir():
            continue
        parts = pair_dir.name.split("__to__")
        if len(parts) != 2:
            continue
        train_sites, test_site = parts[0], parts[1]
        for model_dir in pair_dir.iterdir():
            if not model_dir.is_dir():
                continue
            res_path = model_dir / "results.json"
            if not res_path.exists():
                continue
            try:
                r = load_json(res_path)
                m = r.get("metrics", {})
                bc = r.get("bootstrap_cis", {})
                rows.append({
                    "train_sites": train_sites,
                    "test_site": test_site,
                    "model": r.get("model", model_dir.name),
                    "roc_auc": safe_float(m.get("roc_auc")),
                    "roc_auc_ci_lower": safe_float(bc.get("roc_auc", {}).get("ci_lower")),
                    "roc_auc_ci_upper": safe_float(bc.get("roc_auc", {}).get("ci_upper")),
                    "brier_score": safe_float(m.get("brier_score")),
                    "ece": safe_float(m.get("ece")),
                    "n_test": m.get("n_test"),
                })
            except Exception:
                pass
    return pd.DataFrame(rows)


def collect_external_kaggle_uci() -> pd.DataFrame:
    rows = []
    ext_dir = OUTPUTS / "external_kaggle_uci"
    if not ext_dir.exists():
        return pd.DataFrame()
    for variant_dir in ext_dir.iterdir():
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name
        for pair_dir in variant_dir.iterdir():
            if not pair_dir.is_dir():
                continue
            parts = pair_dir.name.split("__to__")
            if len(parts) != 2:
                continue
            train_sites, test_site = parts[0], parts[1]
            for model_dir in pair_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                res_path = model_dir / "results.json"
                if not res_path.exists():
                    continue
                try:
                    r = load_json(res_path)
                    m = r.get("metrics", {})
                    bc = r.get("bootstrap_cis", {})
                    rows.append({
                        "variant": variant,
                        "train_sites": train_sites,
                        "test_site": test_site,
                        "model": r.get("model", model_dir.name),
                        "roc_auc": safe_float(m.get("roc_auc")),
                        "roc_auc_ci_lower": safe_float(bc.get("roc_auc", {}).get("ci_lower")),
                        "roc_auc_ci_upper": safe_float(bc.get("roc_auc", {}).get("ci_upper")),
                        "brier_score": safe_float(m.get("brier_score")),
                        "ece": safe_float(m.get("ece")),
                        "n_test": m.get("n_test"),
                    })
                except Exception:
                    pass
    return pd.DataFrame(rows)


def collect_size_matched() -> pd.DataFrame:
    rows = []
    sm_dir = OUTPUTS / "size_matched"
    if not sm_dir.exists():
        return pd.DataFrame()
    for pair_dir in sm_dir.iterdir():
        if not pair_dir.is_dir():
            continue
        parts = pair_dir.name.split("__to__")
        if len(parts) != 2:
            continue
        train_sites, test_site = parts[0], parts[1]
        for f in pair_dir.glob("*.json"):
            try:
                r = load_json(f)
                mm = r.get("metrics_mean", {})
                ms = r.get("metrics_std", {})
                model = r.get("model", f.stem)
                rows.append({
                    "train_sites": train_sites,
                    "test_site": test_site,
                    "model": model,
                    "roc_auc_mean": safe_float(mm.get("roc_auc")),
                    "roc_auc_std": safe_float(ms.get("roc_auc")),
                    "brier_mean": safe_float(mm.get("brier_score")),
                    "ece_mean": safe_float(mm.get("ece")),
                    "K": r.get("K"),
                })
            except Exception:
                pass
    return pd.DataFrame(rows)


def collect_shift() -> pd.DataFrame:
    rows = []
    shift_dir = OUTPUTS / "shift"
    if not shift_dir.exists():
        return pd.DataFrame()
    for pair_dir in shift_dir.iterdir():
        if not pair_dir.is_dir():
            continue
        diag_path = pair_dir / "shift_diagnostics.json"
        if not diag_path.exists():
            continue
        try:
            r = load_json(diag_path)
            ps = r.get("prevalence_shift", {})
            abs_diff = safe_float(ps.get("absolute_diff")) if ps else float("nan")
            ts = r.get("train_sites", [])
            train_str = "+".join(ts) if isinstance(ts, list) else str(ts)
            rows.append({
                "train_sites": train_str,
                "test_site": r.get("test_site", ""),
                "model": r.get("model", ""),
                "mean_psi": safe_float(r.get("mean_psi")),
                "c2st_auc": safe_float(r.get("c2st_auc")),
                "prevalence_diff": abs_diff,
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


def collect_calibration_summary() -> pd.DataFrame:
    rows = []
    for variant in ["intercept_only", "intercept_slope"]:
        summary_path = OUTPUTS / "calibration" / "updating" / variant / "summary.json"
        if not summary_path.exists():
            continue
        try:
            data = load_json(summary_path)
            if not isinstance(data, list):
                data = [data]
            for item in data:
                mb = item.get("metrics_before", {})
                ma = item.get("metrics_after", {})
                exp_type = item.get("experiment_type", "")
                train_site = item.get("train_site") or (
                    "+".join(item["train_sites"]) if isinstance(item.get("train_sites"), list) else ""
                )
                test_site = item.get("test_site", "")
                exp_var = item.get("experiment_variant", "")
                rows.append({
                    "variant": variant,
                    "experiment_type": exp_type,
                    "train_site": train_site,
                    "test_site": test_site,
                    "experiment_variant": exp_var,
                    "ece_before": safe_float(mb.get("ece")),
                    "ece_after": safe_float(ma.get("ece")),
                    "brier_before": safe_float(mb.get("brier_score")),
                    "brier_after": safe_float(ma.get("brier_score")),
                    "model": item.get("model", ""),
                })
        except Exception:
            pass
    return pd.DataFrame(rows)


def build_internal_vs_external(internal: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    """Internal vs external performance comparison (by test site)."""
    # For each test site, get internal AUC (when site matches) vs best external AUC
    comp = []
    sites = internal["site"].unique().tolist()
    for site in sites:
        int_sub = internal[internal["site"] == site]
        ext_sub = external[external["test_site"] == site]
        for model in internal["model"].unique():
            int_row = int_sub[int_sub["model"] == model]
            ext_row = ext_sub[ext_sub["model"] == model]
            int_auc = int_row["roc_auc"].iloc[0] if len(int_row) else float("nan")
            ext_auc = ext_row["roc_auc"].max() if len(ext_row) else float("nan")
            ext_best = ext_row.loc[ext_row["roc_auc"].idxmax()] if len(ext_row) else None
            comp.append({
                "test_site": site,
                "model": model,
                "internal_roc_auc": int_auc,
                "external_roc_auc": ext_auc,
                "auc_drop": int_auc - ext_auc if not (math.isnan(int_auc) or math.isnan(ext_auc)) else float("nan"),
                "best_external_train": ext_best["train_sites"] if ext_best is not None else "",
            })
    return pd.DataFrame(comp)


def build_cfs_penalty_table(internal: pd.DataFrame, ext_kaggle: pd.DataFrame) -> pd.DataFrame:
    """CFS penalty: full-feature (internal) AUC vs CFS (external_kaggle_uci) AUC."""
    rows = []
    # For each test site that appears in both internal and external_kaggle_uci
    for test_site in internal["site"].unique():
        int_sub = internal[internal["site"] == test_site]
        ext_sub = ext_kaggle[ext_kaggle["test_site"] == test_site]
        if ext_sub.empty:
            continue
        for model in internal["model"].unique():
            int_row = int_sub[int_sub["model"] == model]
            ext_row = ext_sub[ext_sub["model"] == model]
            if int_row.empty or ext_row.empty:
                continue
            full_auc = int_row["roc_auc"].iloc[0]
            # Use mean CFS AUC across train sources for this test site
            cfs_auc = ext_row["roc_auc"].mean()
            penalty = full_auc - cfs_auc
            rows.append({
                "test_site": test_site,
                "model": model,
                "full_feature_auc": full_auc,
                "cfs_auc_mean": cfs_auc,
                "cfs_penalty": penalty,
            })
    return pd.DataFrame(rows)


def build_calibration_summary_table(cal: pd.DataFrame) -> pd.DataFrame:
    """Compact calibration before/after summary."""
    if cal.empty:
        return pd.DataFrame()
    agg = cal.groupby(["variant", "experiment_type", "model"], as_index=False).agg({
        "ece_before": "mean",
        "ece_after": "mean",
        "brier_before": "mean",
        "brier_after": "mean",
    })
    agg["ece_delta"] = agg["ece_after"] - agg["ece_before"]
    agg["brier_delta"] = agg["brier_after"] - agg["brier_before"]
    return agg


def top_shift_diagnostics(shift: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top N shift pairs by mean_psi (descending)."""
    if shift.empty:
        return pd.DataFrame()
    df = shift.copy()
    df["pair"] = df["train_sites"] + " → " + df["test_site"]
    df = df.sort_values("mean_psi", ascending=False).head(n)
    return df[["pair", "model", "mean_psi", "c2st_auc", "prevalence_diff"]]


def save_table(df: pd.DataFrame, base: str) -> None:
    if df.empty:
        return
    df.to_csv(PAPER_READY / f"{base}.csv", index=False, float_format="%.4f")
    try:
        md = df.to_markdown(index=False, floatfmt=".4f")
    except (AttributeError, ImportError):
        md = _df_to_md(df)
    (PAPER_READY / f"{base}.md").write_text(md, encoding="utf-8")


def _df_to_md(df: pd.DataFrame) -> str:
    """Simple markdown table without tabulate."""
    def fmt(v):
        if pd.isna(v):
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in row) for row in df.values]
    return "\n".join([header, sep] + rows)


def plot_internal_vs_external(comp: pd.DataFrame) -> None:
    if not HAS_MATPLOTLIB or comp.empty:
        return
    sites = comp["test_site"].unique()
    models = comp["model"].unique()
    x = range(len(sites))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        sub = comp[comp["model"] == model]
        vals = [sub[sub["test_site"] == s]["internal_roc_auc"].iloc[0] if len(sub[sub["test_site"] == s]) else 0 for s in sites]
        bars = ax.bar([xi + i * width for xi in x], vals, width, label=f"Internal {model}")
    for i, model in enumerate(models):
        sub = comp[comp["model"] == model]
        vals = [sub[sub["test_site"] == s]["external_roc_auc"].iloc[0] if len(sub[sub["test_site"] == s]) else 0 for s in sites]
        bars = ax.bar([xi + (len(models) + i) * width for xi in x], vals, width, label=f"External {model}", hatch="//")
    ax.set_xticks([xi + (len(models) - 0.5) * width for xi in x])
    ax.set_xticklabels(sites, rotation=45, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Internal vs External Validation Performance (RQ1, RQ2)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(PAPER_READY / "fig_internal_vs_external.png", dpi=150, bbox_inches="tight")
    fig.savefig(PAPER_READY / "fig_internal_vs_external.svg", bbox_inches="tight")
    plt.close()


def plot_internal_vs_external_simple(comp: pd.DataFrame) -> None:
    """Simpler: grouped bar by test site, internal vs external (mean across models)."""
    if not HAS_MATPLOTLIB or comp.empty:
        return
    agg = comp.groupby("test_site").agg({"internal_roc_auc": "mean", "external_roc_auc": "mean"}).reset_index()
    sites = agg["test_site"].tolist()
    x = range(len(sites))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([xi - width / 2 for xi in x], agg["internal_roc_auc"], width, label="Internal (mean)", color="steelblue")
    ax.bar([xi + width / 2 for xi in x], agg["external_roc_auc"], width, label="External (mean)", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=45, ha="right")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Internal vs External Validation Performance (RQ1, RQ2)")
    ax.legend()
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(PAPER_READY / "fig_internal_vs_external.png", dpi=150, bbox_inches="tight")
    fig.savefig(PAPER_READY / "fig_internal_vs_external.svg", bbox_inches="tight")
    plt.close()


def plot_cfs_penalty(cfs: pd.DataFrame) -> None:
    if not HAS_MATPLOTLIB or cfs.empty:
        return
    df = cfs.pivot_table(index="test_site", columns="model", values="cfs_penalty")
    fig, ax = plt.subplots(figsize=(8, 5))
    df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("CFS Penalty (Full AUC − CFS AUC)")
    ax.set_title("Common Feature Set Penalty (RQ5)")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(PAPER_READY / "fig_cfs_penalty.png", dpi=150, bbox_inches="tight")
    fig.savefig(PAPER_READY / "fig_cfs_penalty.svg", bbox_inches="tight")
    plt.close()


def plot_calibration_summary(cal: pd.DataFrame) -> None:
    if not HAS_MATPLOTLIB or cal.empty:
        return
    df = cal.copy()
    df["experiment"] = df["variant"] + "_" + df["experiment_type"] + "_" + df["model"]
    df = df.sort_values("ece_before", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    w = 0.35
    ax.bar([i - w / 2 for i in x], df["ece_before"], w, label="ECE Before", color="lightcoral")
    ax.bar([i + w / 2 for i in x], df["ece_after"], w, label="ECE After", color="lightgreen", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["experiment"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ECE")
    ax.set_title("Calibration Before vs After Recalibration (RQ4)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(PAPER_READY / "fig_calibration_before_after.png", dpi=150, bbox_inches="tight")
    fig.savefig(PAPER_READY / "fig_calibration_before_after.svg", bbox_inches="tight")
    plt.close()


def plot_shift_top(shift: pd.DataFrame, n: int = 10) -> None:
    if not HAS_MATPLOTLIB or shift.empty:
        return
    top = shift.sort_values("mean_psi", ascending=False).head(n)
    top["pair"] = top["train_sites"] + " → " + top["test_site"]
    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = range(len(top))
    ax.barh(y_pos, top["mean_psi"], color="teal", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["pair"], fontsize=9)
    ax.set_xlabel("Mean PSI")
    ax.set_title("Top Shift Diagnostics by Mean PSI (RQ3)")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(PAPER_READY / "fig_shift_top.png", dpi=150, bbox_inches="tight")
    fig.savefig(PAPER_READY / "fig_shift_top.svg", bbox_inches="tight")
    plt.close()


def main() -> None:
    PAPER_READY.mkdir(parents=True, exist_ok=True)

    internal = collect_internal()
    external_uci = collect_external_uci()
    external_kaggle = collect_external_kaggle_uci()
    size_matched = collect_size_matched()
    shift = collect_shift()
    calibration = collect_calibration_summary()

    # Tables
    comp = build_internal_vs_external(internal, external_uci)
    save_table(comp, "table_internal_vs_external")

    cfs_penalty = build_cfs_penalty_table(internal, external_kaggle)
    save_table(cfs_penalty, "table_cfs_penalty")

    cal_summary = build_calibration_summary_table(calibration)
    save_table(cal_summary, "table_calibration_summary")

    if not cal_summary.empty:
        plot_calibration_summary(cal_summary)

    sm_summary = size_matched.groupby(["train_sites", "test_site"]).agg({
        "roc_auc_mean": "mean",
        "roc_auc_std": "mean",
        "ece_mean": "mean",
    }).reset_index()
    sm_summary.columns = ["train_sites", "test_site", "roc_auc_mean", "roc_auc_std", "ece_mean"]
    save_table(sm_summary, "table_size_matched_summary")

    top_shift = top_shift_diagnostics(shift, n=10)
    save_table(top_shift, "table_shift_top")

    # Calibration failure analysis (formal layer)
    import subprocess
    subprocess.run(
        [__import__("sys").executable, str(ROOT / "scripts" / "build_calibration_analysis.py")],
        cwd=str(ROOT),
        check=False,
    )

    # Figures
    if not comp.empty:
        plot_internal_vs_external_simple(comp)
    if not cfs_penalty.empty:
        plot_cfs_penalty(cfs_penalty)
    if not shift.empty:
        plot_shift_top(shift)

    # README
    readme = """# Paper-Ready Evidence Bundle

Camera-ready figures and tables for the heart model transportability paper.

## Artifacts and RQ Mapping

| Artifact | Format | RQ | Description |
|----------|--------|-----|--------------|
| `fig_internal_vs_external` | PNG, SVG | RQ1, RQ2 | Internal vs external validation ROC-AUC comparison by test site |
| `table_internal_vs_external` | CSV, MD | RQ1, RQ2 | Per-model internal vs external performance with AUC drop |
| `fig_cfs_penalty` | PNG, SVG | RQ5 | Common Feature Set (CFS) penalty by test site and model |
| `table_cfs_penalty` | CSV, MD | RQ5 | Full-feature AUC vs CFS AUC and penalty delta |
| `fig_calibration_before_after` | PNG, SVG | RQ4 | ECE before vs after recalibration |
| `table_calibration_summary` | CSV, MD | RQ4 | Calibration metrics before/after (ECE, Brier) by variant |
| `fig_shift_top` | PNG, SVG | RQ3 | Top shift pairs by mean PSI |
| `table_shift_top` | CSV, MD | RQ3 | Top shift diagnostics (mean PSI, C2ST AUC, prevalence diff) |
| `table_size_matched_summary` | CSV, MD | RQ2 | Size-matched sensitivity: mean ROC-AUC and ECE across K subsamples |

## Research Questions

- **RQ1**: Internal validation baseline (discrimination, calibration)
- **RQ2**: External validation degradation; size-matched sensitivity
- **RQ3**: Dataset shift diagnostics (PSI, C2ST, prevalence)
- **RQ4**: Calibration/recalibration effectiveness
- **RQ5**: CFS performance penalty

## Source Artifacts

Generated from:
- `outputs/internal/` — internal validation results
- `outputs/external_uci/` — UCI multi-site external validation
- `outputs/external_kaggle_uci/` — Kaggle↔UCI CFS experiments
- `outputs/calibration/updating/` — recalibration summaries
- `outputs/size_matched/` — size-matched sensitivity
- `outputs/shift/` — shift diagnostics
"""
    (PAPER_READY / "README.md").write_text(readme, encoding="utf-8")

    print(f"Paper-ready bundle written to {PAPER_READY}")
    for f in sorted(PAPER_READY.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
