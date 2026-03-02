"""
Evaluation pipeline — eval_plan implementation.
Aggregates results, generates pivots, statistical tests, figures, and report.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from . import plotting

logger = logging.getLogger(__name__)


def _load_config() -> dict[str, Any]:
    import yaml

    config_path = Path(__file__).resolve().parent.parent / "configs" / "evaluation.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _resolve_outputs_dir(outputs_dir: str, run_id: str | None) -> Path:
    base = Path(outputs_dir)
    if run_id:
        return base / "runs" / run_id
    # Prefer latest run if outputs/runs exists
    runs = base / "runs"
    if runs.exists():
        dirs = sorted([d for d in runs.iterdir() if d.is_dir()], key=lambda d: d.name, reverse=True)
        if dirs:
            return dirs[0]
    return base


def build_master_table(outputs_dir: str | Path, run_id: str | None = None) -> pd.DataFrame:
    """Glob **/results.json, load each, flatten into one DataFrame."""
    root = _resolve_outputs_dir(str(outputs_dir), run_id)
    rows = []
    for results_path in root.rglob("results.json"):
        try:
            with open(results_path) as f:
                r = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skip %s: %s", results_path, e)
            continue

        # Normalize schema: internal uses site, external uses train_site/train_sites + test_site
        exp_type = r.get("experiment_type", "unknown")
        if exp_type in ("internal", "internal_cfs"):
            train_sites = r.get("site", "unknown")
            test_site = r.get("site", "unknown")
        else:
            train_sites = r.get("train_sites") or r.get("train_site")
            if isinstance(train_sites, list):
                train_sites = "+".join(train_sites)
            test_site = r.get("test_site", "unknown")

        metrics = r.get("metrics", {})
        row = {
            "experiment_type": exp_type,
            "variant": r.get("variant", ""),
            "train_sites": str(train_sites),
            "test_site": str(test_site),
            "model": r.get("model", "unknown"),
            "n_train": r.get("n_train"),
            "n_test": metrics.get("n_test"),
            "n_features_used": len(r.get("features_used", [])),
            "features_used": r.get("features_used"),
            "best_params": r.get("best_params"),
            "results_path": str(results_path),
        }
        # Flatten metrics (exclude non-scalars)
        for k, v in metrics.items():
            if k in ("confusion_matrix", "bin_details"):
                continue
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[k] = v

        # Bootstrap CIs
        for metric_name, ci in (r.get("bootstrap_cis") or {}).items():
            if isinstance(ci, dict):
                row[f"{metric_name}_ci_lower"] = ci.get("ci_lower")
                row[f"{metric_name}_ci_upper"] = ci.get("ci_upper")

        # CFS penalty
        cfs = r.get("cfs_penalty") or {}
        if cfs:
            row["cfs_full_auc"] = cfs.get("full_auc")
            row["cfs_cfs_auc"] = cfs.get("cfs_auc")
            row["cfs_auc_drop"] = cfs.get("auc_drop")
            row["cfs_relative_drop_pct"] = cfs.get("relative_drop_pct")

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    sort_cols = [c for c in ["experiment_type", "train_sites", "test_site", "model"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def generate_all_pivots(master: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Internal baseline, external AUC matrix, AUC delta, calibration, shift, recalibration, CFS penalty."""
    pivots = {}

    # Internal baseline: site × model → ROC-AUC [CI]
    internal = master[master["experiment_type"].isin(("internal", "internal_cfs"))]
    if not internal.empty:
        pivots["internal_baseline"] = internal.pivot_table(
            index="test_site",
            columns="model",
            values="roc_auc",
            aggfunc="first",
        ).round(3)
        if "roc_auc_ci_lower" in internal.columns:
            ci_lo = internal.pivot_table(index="test_site", columns="model", values="roc_auc_ci_lower", aggfunc="first")
            ci_hi = internal.pivot_table(index="test_site", columns="model", values="roc_auc_ci_upper", aggfunc="first")
            pivots["internal_baseline_ci_lower"] = ci_lo
            pivots["internal_baseline_ci_upper"] = ci_hi

    # External AUC matrix (pairwise): train × test → AUC per model
    external = master[master["experiment_type"].str.startswith("external", na=False)]
    if not external.empty:
        for model in external["model"].unique():
            ext_m = external[external["model"] == model]
            pivot = ext_m.pivot_table(index="train_sites", columns="test_site", values="roc_auc", aggfunc="first")
            pivots[f"external_auc_matrix_{model}"] = pivot.round(3)

    # AUC delta: external − internal
    if not internal.empty and not external.empty:
        # Build lookup: prefer full-feature internal when both exist
        internal_auc: dict[tuple[str, str], float] = {}
        for _, r in internal.iterrows():
            internal_auc[(r["test_site"], r["model"])] = r["roc_auc"]
        for _, r in internal[internal["experiment_type"] == "internal"].iterrows():
            internal_auc[(r["test_site"], r["model"])] = r["roc_auc"]

        ext_copy = external.copy()
        ext_copy["internal_auc"] = ext_copy.apply(
            lambda r: internal_auc.get((r["test_site"], r["model"]), np.nan),
            axis=1,
        )
        ext_copy["auc_delta"] = ext_copy["roc_auc"] - ext_copy["internal_auc"]
        pivots["auc_delta"] = ext_copy

    # Calibration summary: experiment × model → Brier, ECE, MCE
    cal_cols = [c for c in ("brier_score", "ece", "mce") if c in master.columns]
    if cal_cols:
        pivots["calibration_summary"] = master[
            ["experiment_type", "train_sites", "test_site", "model"] + cal_cols
        ].drop_duplicates()

    # Shift summary: from shift diagnostics (loaded separately)
    # Recalibration comparison: from calibration artifacts (loaded separately)
    # CFS penalty: site × model → full vs CFS AUC
    cfs_internal = internal[internal["experiment_type"] == "internal_cfs"]
    if "cfs_cfs_auc" in cfs_internal.columns:
        pivots["cfs_penalty"] = cfs_internal[
            ["test_site", "model", "cfs_full_auc", "cfs_cfs_auc", "cfs_auc_drop", "cfs_relative_drop_pct"]
        ].copy()
        pivots["cfs_penalty"] = pivots["cfs_penalty"].rename(
            columns={
                "test_site": "site",
                "cfs_full_auc": "full_auc",
                "cfs_cfs_auc": "cfs_auc",
                "cfs_auc_drop": "auc_drop",
            }
        )

    return pivots


def run_statistical_tests(
    master: pd.DataFrame,
    outputs_dir: str | Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Bootstrap AUC difference, Spearman shift–performance, Wilcoxon recalibration."""
    root = Path(_resolve_outputs_dir(str(outputs_dir), run_id))
    results: dict[str, Any] = {}

    # Shift–performance Spearman
    shift_perf_path = root / "shift" / "shift_performance_merged.csv"
    if shift_perf_path.exists():
        sp = pd.read_csv(shift_perf_path)
        # Normalize column names
        sp = sp.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
        valid = sp.dropna(subset=["mean_psi", "auc_drop"], how="all")
        if "auc_drop" in valid.columns and not valid["auc_drop"].isna().all():
            rho, p = spearmanr(valid["mean_psi"], valid["auc_drop"])
            results["shift_performance_mean_psi_vs_auc_drop"] = {"spearman_rho": float(rho), "p_value": float(p)}
        if "prevalence_diff" in valid.columns and "brier_change" in valid.columns:
            v = valid[["prevalence_diff", "brier_change"]].dropna()
            if len(v) > 2:
                rho, p = spearmanr(v["prevalence_diff"], v["brier_change"])
                results["shift_performance_prevalence_vs_brier"] = {"spearman_rho": float(rho), "p_value": float(p)}
        if "c2st_auc" in valid.columns and "roc_auc" in valid.columns:
            v = valid[["c2st_auc", "roc_auc"]].dropna()
            if len(v) > 2:
                rho, p = spearmanr(v["c2st_auc"], v["roc_auc"])
                results["shift_performance_c2st_vs_roc_auc"] = {"spearman_rho": float(rho), "p_value": float(p)}

    # Recalibration Wilcoxon: ECE before vs after
    cal_dir = root / "calibration" / "updating"
    ece_before, ece_after = [], []
    for method in ("intercept_only", "intercept_slope"):
        method_dir = cal_dir / method
        if not method_dir.exists():
            continue
        for p in method_dir.glob("*.json"):
            if p.name == "summary.json":
                continue
            try:
                with open(p) as f:
                    d = json.load(f)
                mb = d.get("metrics_before", {})
                ma = d.get("metrics_after", {})
                if "ece" in mb and "ece" in ma:
                    ece_before.append(mb["ece"])
                    ece_after.append(ma["ece"])
            except (json.JSONDecodeError, OSError):
                pass
    if len(ece_before) >= 3 and len(ece_after) >= 3:
        try:
            stat, p = wilcoxon(ece_before, ece_after, alternative="greater")
            results["recalibration_wilcoxon"] = {
                "mean_ece_before": float(np.mean(ece_before)),
                "mean_ece_after": float(np.mean(ece_after)),
                "mean_improvement": float(np.mean(np.array(ece_before) - np.array(ece_after))),
                "wilcoxon_stat": float(stat),
                "p_value": float(p),
            }
        except Exception as e:
            results["recalibration_wilcoxon"] = {"error": str(e)}

    return results


def generate_all_figures(
    master: pd.DataFrame,
    outputs_dir: str | Path,
    figures_dir: Path,
    config: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> None:
    """ROC curves, AUC heatmaps, AUC drop bars, reliability, ECE, PSI, shift vs perf, CFS penalty."""
    config = config or _load_config()
    root = Path(_resolve_outputs_dir(str(outputs_dir), run_id))
    formats = config.get("figure_format", ["pdf", "png"])
    dpi = config.get("figure_dpi", 300)

    # AUC heatmaps (F3)
    external = master[master["experiment_type"].str.startswith("external", na=False)]
    for model in master["model"].unique():
        ext_m = external[external["model"] == model]
        if ext_m.empty:
            continue
        pivot = ext_m.pivot_table(index="train_sites", columns="test_site", values="roc_auc", aggfunc="first")
        if pivot.empty:
            continue
        plotting.plot_auc_heatmap(pivot, model, figures_dir, config)

    # AUC drop bars (F4)
    pivots = generate_all_pivots(master)
    if "auc_delta" in pivots:
        delta_df = pivots["auc_delta"]
        if "roc_auc_ci_lower" in delta_df.columns:
            delta_df = delta_df.copy()
            delta_df["roc_auc_ci_lower"] = delta_df.get("roc_auc_ci_lower")
            delta_df["roc_auc_ci_upper"] = delta_df.get("roc_auc_ci_upper")
        plotting.plot_auc_drop_bars(delta_df, figures_dir, config)

    # ECE comparison (F9) — from calibration
    ece_rows = []
    for method in ("intercept_only", "intercept_slope"):
        method_dir = root / "calibration" / "updating" / method
        if not method_dir.exists():
            continue
        for p in method_dir.glob("*.json"):
            if p.name == "summary.json":
                continue
            try:
                with open(p) as f:
                    d = json.load(f)
                ece_rows.append({
                    "experiment_id": f"{d.get('train_site','')}→{d.get('test_site','')} {d.get('model','')}",
                    "method": method,
                    "ece_before": d.get("metrics_before", {}).get("ece"),
                    "ece_after": d.get("metrics_after", {}).get("ece"),
                })
            except (json.JSONDecodeError, OSError):
                pass
    if ece_rows:
        ece_df = pd.DataFrame(ece_rows).dropna(subset=["ece_before", "ece_after"])
        if not ece_df.empty:
            plotting.plot_ece_comparison(ece_df, figures_dir, config)

    # PSI heatmap (F11) — aggregate from feature_shift.csv
    psi_rows = []
    shift_dir = root / "shift"
    for pair_dir in shift_dir.iterdir() if shift_dir.exists() else []:
        if not pair_dir.is_dir():
            continue
        fs_path = pair_dir / "feature_shift.csv"
        if not fs_path.exists():
            continue
        try:
            fs = pd.read_csv(fs_path)
            if "feature" in fs.columns and "psi" in fs.columns:
                pair_name = pair_dir.name
                for _, row in fs.iterrows():
                    psi_rows.append({"pair": pair_name, "feature": row["feature"], "psi": row["psi"]})
        except Exception:
            pass
    if psi_rows:
        psi_df = pd.DataFrame(psi_rows)
        psi_pivot = psi_df.pivot_table(index="feature", columns="pair", values="psi", aggfunc="first")
        if not psi_pivot.empty:
            plotting.plot_psi_heatmap(psi_pivot, figures_dir, config)

    # Shift vs performance (F13)
    sp_path = root / "shift" / "shift_performance_merged.csv"
    if sp_path.exists():
        sp = pd.read_csv(sp_path)
        sp = sp.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
        # Need roc_auc for third panel — merge from master if missing
        if "roc_auc" not in sp.columns and not master.empty:
            merge_cols = ["train_sites", "test_site", "model"]
            ext = master[master["experiment_type"].str.startswith("external", na=False)][merge_cols + ["roc_auc"]]
            sp = sp.merge(ext, on=merge_cols, how="left", suffixes=("", "_y"))
        plotting.plot_shift_vs_performance(sp, figures_dir, config)

    # CFS penalty (F16)
    if "cfs_penalty" in pivots:
        cfs_df = pivots["cfs_penalty"]
        if "full_auc" in cfs_df.columns and cfs_df["full_auc"].notna().any():
            plotting.plot_cfs_penalty_bars(cfs_df, figures_dir, config)


def synthesize_rq_answers(
    master: pd.DataFrame,
    stat_results: dict[str, Any],
    pivots: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Per-RQ summaries."""
    summaries = {}

    # RQ1: Internal baselines
    internal = master[master["experiment_type"].isin(("internal", "internal_cfs"))]
    if not internal.empty:
        best_per_site = internal.loc[internal.groupby("test_site")["roc_auc"].idxmax()]
        summaries["rq1_internal_baselines"] = {
            "best_model_per_site": best_per_site[["test_site", "model", "roc_auc"]].to_dict("records"),
            "n_experiments": len(internal),
        }

    # RQ2: External degradation
    if "auc_delta" in pivots:
        delta = pivots["auc_delta"]
        worst_drop = delta.loc[delta["auc_delta"].idxmin()] if "auc_delta" in delta.columns else None
        summaries["rq2_external_degradation"] = {
            "worst_auc_drop": worst_drop.to_dict() if worst_drop is not None else None,
            "mean_auc_drop": float(delta["auc_delta"].mean()) if "auc_delta" in delta.columns else None,
        }

    # RQ3: Shift–performance
    if "shift_performance_mean_psi_vs_auc_drop" in stat_results:
        summaries["rq3_shift_performance"] = stat_results["shift_performance_mean_psi_vs_auc_drop"]

    # RQ4: Recalibration
    if "recalibration_wilcoxon" in stat_results:
        summaries["rq4_recalibration"] = stat_results["recalibration_wilcoxon"]

    # RQ5: CFS penalty
    if "cfs_penalty" in pivots:
        cfs = pivots["cfs_penalty"]
        if "auc_drop" in cfs.columns and cfs["auc_drop"].notna().any():
            summaries["rq5_cfs_penalty"] = {
                "mean_auc_drop": float(cfs["auc_drop"].mean()),
                "max_auc_drop": float(cfs["auc_drop"].max()),
            }

    return summaries


def generate_report(
    master: pd.DataFrame,
    pivots: dict[str, pd.DataFrame],
    stat_results: dict[str, Any],
    rq_summaries: dict[str, Any],
    reports_dir: Path,
) -> None:
    """Write reports/evaluation_report.md per eval_plan §6.1."""
    lines = [
        "# Evaluation Report — Heart Disease Model Transportability",
        "",
        "## 1. Executive Summary",
        "",
    ]

    for rq, data in rq_summaries.items():
        lines.append(f"### {rq.replace('_', ' ').title()}")
        if isinstance(data, dict) and "error" not in data:
            lines.append(f"- {json.dumps(data, default=str)}")
        elif isinstance(data, dict):
            lines.append(f"- Error: {data.get('error', 'unknown')}")
        lines.append("")

    lines.extend([
        "## 2. Methods Summary",
        "",
        "- Data sources: Kaggle CVD, UCI Heart Disease (Cleveland, Hungary, Switzerland, VA)",
        "- Models: LR, RF, XGB, LGBM",
        "- Metrics: ROC-AUC, PR-AUC, Brier, ECE, MCE",
        "- Statistical tests: Spearman (shift–performance), Wilcoxon (recalibration)",
        "",
        "## 3. Results — RQ1: Internal Baselines",
        "",
    ])

    if "internal_baseline" in pivots:
        lines.append("| Site | " + " | ".join(pivots["internal_baseline"].columns) + " |")
        lines.append("|------|" + "|".join(["---"] * len(pivots["internal_baseline"].columns)) + "|")
        for idx, row in pivots["internal_baseline"].iterrows():
            lines.append("| " + str(idx) + " | " + " | ".join(f"{v:.3f}" if isinstance(v, (int, float)) else str(v) for v in row) + " |")
        lines.append("")

    lines.extend([
        "## 4. Results — RQ2: External Validation",
        "",
    ])
    if "auc_delta" in pivots:
        delta = pivots["auc_delta"]
        lines.append(f"Mean AUC drop: {delta['auc_delta'].mean():.3f}")
        lines.append("")

    lines.extend([
        "## 5. Results — RQ3: Dataset Shift",
        "",
    ])
    if "rq3_shift_performance" in rq_summaries:
        lines.append(f"Spearman ρ (mean PSI vs AUC drop): {rq_summaries['rq3_shift_performance']}")
        lines.append("")

    lines.extend([
        "## 6. Results — RQ4: Recalibration",
        "",
    ])
    if "rq4_recalibration" in rq_summaries:
        lines.append(f"Wilcoxon (ECE before vs after): {rq_summaries['rq4_recalibration']}")
        lines.append("")

    lines.extend([
        "## 7. Results — RQ5: CFS Penalty",
        "",
    ])
    if "rq5_cfs_penalty" in rq_summaries:
        lines.append(f"CFS penalty: {rq_summaries['rq5_cfs_penalty']}")
        lines.append("")

    lines.extend([
        "## 8. Discussion",
        "",
        "- Transportability patterns depend on train–test site pair.",
        "- Recalibration can improve ECE when miscalibration is correctable.",
        "- CFS restriction incurs AUC penalty; external drop may exceed it.",
        "",
        "## 9. Appendices",
        "",
        "- Full metric tables: reports/tables/master_results.csv",
        "- Statistical tests: reports/tables/statistical_tests.json",
        "- RQ summaries: reports/rq_summaries.json",
        "",
    ])

    report_path = reports_dir / "evaluation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", report_path)


def run_evaluation(
    outputs_dir: str = "outputs",
    reports_dir: str = "reports",
    run_id: str | None = None,
) -> pd.DataFrame:
    """Main entry point: aggregate, pivot, test, figure, synthesize, report."""
    config = _load_config()
    reports_path = Path(reports_dir or config.get("reports_dir", "reports"))
    figures_path = Path(config.get("figures_dir", str(reports_path / "figures")))
    tables_path = Path(config.get("tables_dir", str(reports_path / "tables")))
    figures_path.mkdir(parents=True, exist_ok=True)
    tables_path.mkdir(parents=True, exist_ok=True)

    # 1. Aggregate
    master = build_master_table(outputs_dir, run_id)
    if master.empty:
        logger.warning("No results.json found under %s", _resolve_outputs_dir(outputs_dir, run_id))
        return master

    master.to_csv(tables_path / "master_results.csv", index=False)
    try:
        master.to_parquet(tables_path / "master_results.parquet", index=False)
    except Exception:
        pass

    # 2. Pivots
    pivots = generate_all_pivots(master)
    for name, df in pivots.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            out = tables_path / f"{name}.csv"
            df.to_csv(out, index=True if name != "calibration_summary" else False)

    # 3. Statistical tests
    stat_results = run_statistical_tests(master, outputs_dir, run_id)
    with open(tables_path / "statistical_tests.json", "w", encoding="utf-8") as f:
        json.dump(stat_results, f, indent=2, default=str)

    # 4. Figures
    generate_all_figures(master, outputs_dir, figures_path, config, run_id)

    # 5. RQ synthesis
    rq_summaries = synthesize_rq_answers(master, stat_results, pivots)
    with open(reports_path / "rq_summaries.json", "w", encoding="utf-8") as f:
        json.dump(rq_summaries, f, indent=2, default=str)

    # 6. Report
    generate_report(master, pivots, stat_results, rq_summaries, reports_path)

    return master


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument("--outputs-dir", default="outputs", help="Base outputs directory")
    parser.add_argument("--reports-dir", default="reports", help="Reports output directory")
    parser.add_argument("--run-id", default=None, help="Specific run ID (e.g. 20260301T225358_93bf3a3395d35eb1)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_evaluation(
        outputs_dir=args.outputs_dir,
        reports_dir=args.reports_dir,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
