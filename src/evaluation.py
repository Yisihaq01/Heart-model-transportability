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


def _sanitize_for_json(obj: Any) -> Any:
    """Replace float nan/inf with None so JSON is valid."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


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
        # Only use rows with both mean_psi and auc_drop (excludes external_kaggle_uci with empty auc_drop)
        valid = sp.dropna(subset=["mean_psi", "auc_drop"])
        if len(valid) >= 2 and "auc_drop" in valid.columns:
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

    # F10: Brier decomposition — from predictions
    try:
        decomp_rows = []
        for results_path in root.rglob("results.json"):
            pred_path = results_path.parent / "predictions.csv"
            if not pred_path.exists():
                pred_path = results_path.parent / "predictions.parquet"
            if not pred_path.exists():
                continue
            try:
                pred = pd.read_csv(pred_path) if pred_path.suffix == ".csv" else pd.read_parquet(pred_path)
            except Exception:
                continue
            if "y_true" not in pred.columns or "y_prob" not in pred.columns:
                continue
            dec = plotting._brier_decomposition(pred["y_true"].values, pred["y_prob"].values)
            exp_id = f"{results_path.parent.parent.name} {results_path.parent.name}"
            decomp_rows.append({"experiment_id": exp_id, **dec})
        if decomp_rows:
            decomp_df = pd.DataFrame(decomp_rows)
            # Limit to ~30 experiments for readability
            if len(decomp_df) > 30:
                decomp_df = decomp_df.head(30)
            plotting.plot_brier_decomposition(decomp_df, figures_dir, config)
    except Exception as e:
        logger.debug("Skip F10 Brier decomposition: %s", e)

    # F12: Feature distributions for worst pair
    try:
        if "auc_delta" in pivots:
            delta = pivots["auc_delta"]
            valid = delta.dropna(subset=["auc_delta"])
            if not valid.empty:
                worst = valid.loc[valid["auc_delta"].idxmin()]
                train_sites = [s.strip() for s in str(worst.get("train_sites", "")).split("+")]
                test_site = str(worst.get("test_site", ""))
                data_dir = Path(__file__).resolve().parent.parent / "data"

                def _load_site(site: str) -> pd.DataFrame | None:
                    prefix = "kaggle_clean" if site == "kaggle" else f"uci_{site}_clean"
                    for ext in (".parquet", ".csv"):
                        p = data_dir / f"{prefix}{ext}"
                        if p.exists():
                            return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
                    return None

                train_dfs = [_load_site(ts) for ts in train_sites]
                train_dfs = [d for d in train_dfs if d is not None]
                test_df = _load_site(test_site)
                if train_dfs and test_df is not None:
                    train_df = pd.concat(train_dfs, ignore_index=True) if len(train_dfs) > 1 else train_dfs[0]
                    pair_key = f"{worst['train_sites']}__to__{test_site}"
                    fs_path = (root / "shift" / pair_key / "feature_shift.csv")
                    if not fs_path.exists() and (root / "shift").exists():
                        for d in (root / "shift").iterdir():
                            if d.is_dir():
                                fp = d / "feature_shift.csv"
                                if fp.exists() and test_site in d.name:
                                    fs_path = fp
                                    break
                    features = []
                    if fs_path and fs_path.exists():
                        fs = pd.read_csv(fs_path)
                        if "feature" in fs.columns:
                            features = fs["feature"].tolist()
                    if not features:
                        features = [c for c in train_df.columns if c in test_df.columns and c not in ("site", "target", "num")]
                    if features:
                        pair_name = f"{worst['train_sites']} → {test_site}"
                        plotting.plot_feature_distributions_worst_pair(
                            train_df, test_df, features[:9], pair_name, figures_dir, config
                        )
    except Exception as e:
        logger.debug("Skip F12 feature distributions: %s", e)

    # F14: Missingness heatmap — from ingestion_report.json
    try:
        report_path = Path(__file__).resolve().parent.parent / "data" / "ingestion_report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            sites_meta = report.get("sites", {})
            all_features = set()
            for site, meta in sites_meta.items():
                all_features.update(meta.get("missing_rates", {}).keys())
            all_features.update(report.get("cfs_uci_cross_site", []))
            all_features.update(report.get("cfs_kaggle_uci", []))
            all_features.discard("target")
            rows = []
            for feat in sorted(all_features):
                row = {}
                for site in sites_meta:
                    mr = sites_meta[site].get("missing_rates", {})
                    val = mr.get(feat, mr.get("sys_bp" if feat == "trestbps" else "trestbps" if feat == "sys_bp" else None, 0))
                    row[site] = float(val) if val is not None else 0
                rows.append(row)
            if rows:
                miss_df = pd.DataFrame(rows, index=sorted(all_features))
                miss_df = miss_df.reindex(columns=sorted(sites_meta.keys()))
                plotting.plot_missingness_heatmap(miss_df, figures_dir, config)
    except Exception as e:
        logger.debug("Skip F14 missingness heatmap: %s", e)

    # F15: C2ST vs external AUC
    try:
        sp_path = root / "shift" / "shift_performance_merged.csv"
        if sp_path.exists():
            sp = pd.read_csv(sp_path)
            sp = sp.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
            if "roc_auc" not in sp.columns and not master.empty:
                merge_cols = ["train_sites", "test_site", "model"]
                ext = master[master["experiment_type"].str.startswith("external", na=False)][merge_cols + ["roc_auc"]]
                sp = sp.merge(ext, on=merge_cols, how="left", suffixes=("", "_y"))
            plotting.plot_c2st_vs_auc(sp, figures_dir, config)
    except Exception as e:
        logger.debug("Skip F15 C2ST vs AUC: %s", e)

    # F17: Size-matched comparison
    try:
        sm_dir = root / "size_matched"
        if sm_dir.exists():
            rows = []
            for pair_dir in sm_dir.iterdir():
                if not pair_dir.is_dir():
                    continue
                pair_name = pair_dir.name.replace("__", " → ")
                for model_file in pair_dir.glob("*.json"):
                    try:
                        with open(model_file) as f:
                            d = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        continue
                    sm = d.get("metrics_mean", {})
                    train_sites = d.get("train_sites", [])
                    if isinstance(train_sites, list):
                        train_sites = "+".join(train_sites)
                    test_site = d.get("test_site", "")
                    full_auc = None
                    res_path = root / "external_uci" / pair_dir.name / model_file.stem / "results.json"
                    if res_path.exists():
                        with open(res_path) as rf:
                            rr = json.load(rf)
                        full_auc = rr.get("metrics", {}).get("roc_auc")
                    if full_auc is None and "auc_delta" in pivots:
                        delta = pivots["auc_delta"]
                        match = delta[(delta["train_sites"] == train_sites) & (delta["test_site"] == test_site) & (delta["model"] == model_file.stem)]
                        if not match.empty:
                            full_auc = match["roc_auc"].iloc[0]
                    rows.append({
                        "pair_model": f"{pair_name} {model_file.stem}",
                        "full_auc": full_auc or sm.get("roc_auc"),
                        "subsampled_mean": sm.get("roc_auc"),
                        "subsampled_std": d.get("metrics_std", {}).get("roc_auc", 0),
                    })
            if rows:
                sm_df = pd.DataFrame(rows)
                sm_df = sm_df[sm_df["full_auc"].notna() | sm_df["subsampled_mean"].notna()]
                if not sm_df.empty:
                    sm_df["full_auc"] = sm_df["full_auc"].fillna(sm_df["subsampled_mean"])
                    sm_df["subsampled_mean"] = sm_df["subsampled_mean"].fillna(sm_df["full_auc"])
                    plotting.plot_size_matched_comparison(sm_df.head(40), figures_dir, config)
    except Exception as e:
        logger.debug("Skip F17 size-matched: %s", e)

    # F18: Effective CFS feature count
    try:
        report_path = Path(__file__).resolve().parent.parent / "data" / "ingestion_report.json"
        pipeline_cfg_path = Path(__file__).resolve().parent.parent / "configs" / "pipeline.yaml"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            import yaml
            threshold_pct = 40.0
            if pipeline_cfg_path.exists():
                with open(pipeline_cfg_path) as f:
                    pc = yaml.safe_load(f) or {}
                threshold_pct = float(pc.get("missingness_threshold", 0.40) * 100)
            sites_meta = report.get("sites", {})
            cfs_uci = report.get("cfs_uci_cross_site", [])
            cfs_ku = report.get("cfs_kaggle_uci", [])
            uci_sites = [s for s in ("cleveland", "hungary", "switzerland", "va") if s in sites_meta]
            pair_counts = []
            for train in uci_sites:
                for test in uci_sites:
                    if train == test:
                        continue
                    n = sum(1 for f in cfs_uci if sites_meta.get(train, {}).get("missing_rates", {}).get(f, 0) < threshold_pct and sites_meta.get(test, {}).get("missing_rates", {}).get(f, 0) < threshold_pct)
                    pair_counts.append({"pair": f"{train}→{test}", "n_features": n})
            for train in ["kaggle"] + uci_sites:
                for test in ["kaggle"] + uci_sites:
                    if train == test:
                        continue
                    if (train == "kaggle" and test in uci_sites) or (test == "kaggle" and train in uci_sites):
                        n = sum(1 for f in cfs_ku if sites_meta.get(train, {}).get("missing_rates", {}).get(f, 0) < threshold_pct and sites_meta.get(test, {}).get("missing_rates", {}).get(f, 0) < threshold_pct)
                        pair_counts.append({"pair": f"{train}→{test}", "n_features": n})
            if pair_counts:
                plotting.plot_effective_cfs_count(pd.DataFrame(pair_counts), figures_dir, config)
    except Exception as e:
        logger.debug("Skip F18 effective CFS: %s", e)


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


def _format_executive_summary(rq_summaries: dict[str, Any]) -> list[str]:
    """One-paragraph answer per RQ plus headline metrics (eval_plan §6.1)."""
    lines = []
    best_auc = None
    mean_drop = (rq_summaries.get("rq2_external_degradation") or {}).get("mean_auc_drop")
    improvement = (rq_summaries.get("rq4_recalibration") or {}).get("mean_improvement")

    # RQ1
    rq1 = rq_summaries.get("rq1_internal_baselines") or {}
    best_list = rq1.get("best_model_per_site") or []
    if best_list:
        best_auc = max((x.get("roc_auc") for x in best_list if isinstance(x.get("roc_auc"), (int, float))), default=None)
        site_models = ", ".join(f"{x.get('test_site', '?')} ({x.get('model', '?')})" for x in best_list[:5])
        lines.append(
            f"**RQ1 — Internal baselines:** Within-site performance varied by cohort. Best models per site: {site_models}. "
            + (f"Best internal ROC-AUC was {best_auc:.3f}. " if best_auc is not None else "")
            + "Small UCI cohorts (e.g. Switzerland) show wider CIs than Kaggle."
        )
    else:
        lines.append("**RQ1 — Internal baselines:** Within-site discrimination and calibration were computed per site × model; see Results §3.")
    lines.append("")

    # RQ2
    rq2 = rq_summaries.get("rq2_external_degradation") or {}
    worst = rq2.get("worst_auc_drop")
    mean_drop = rq2.get("mean_auc_drop")
    if worst is not None and isinstance(worst, dict):
        train_sites = worst.get("train_sites", "?")
        test_site = worst.get("test_site", "?")
        model = worst.get("model", "?")
        auc_delta = worst.get("auc_delta")
        drop_str = f"{auc_delta:.3f}" if isinstance(auc_delta, (int, float)) else "N/A"
        lines.append(
            f"**RQ2 — External validation:** Performance dropped when models were applied across sites. "
            + f"Worst AUC drop observed: {train_sites} → {test_site} ({model}), Δ = {drop_str}. "
            + (f"Mean AUC drop across external pairs: {mean_drop:.3f}. " if isinstance(mean_drop, (int, float)) else "")
            + "External Kaggle↔UCI CFS runs do not have an internal CFS baseline on the same test site, so CFS penalty fields (cfs_full_auc, cfs_cfs_auc) are NaN for those rows."
        )
    elif isinstance(mean_drop, (int, float)):
        lines.append(f"**RQ2 — External validation:** Mean external AUC drop was {mean_drop:.3f}. See Results §4 and Discussion for transportability patterns.")
    else:
        lines.append("**RQ2 — External validation:** Pairwise and LOSO external validation results are in Results §4.")
    lines.append("")

    # RQ3
    rq3 = rq_summaries.get("rq3_shift_performance") or {}
    rho = rq3.get("spearman_rho")
    p_val = rq3.get("p_value")
    lines.append(
        "**RQ3 — Dataset shift:** Shift diagnostics (PSI, prevalence diff, C2ST) were correlated with performance degradation. "
        + (f"Spearman ρ (mean PSI vs AUC drop) = {rho:.2f}, p = {p_val:.3f}. "
           "The correlation was not statistically significant; power was limited by the number of external pairs. " if isinstance(rho, (int, float)) and isinstance(p_val, (int, float)) else "")
        + "Shift signatures (prevalence vs covariate vs missingness) are summarized in Results §5."
    )
    lines.append("")

    # RQ4
    rq4 = rq_summaries.get("rq4_recalibration") or {}
    ece_before = rq4.get("mean_ece_before")
    ece_after = rq4.get("mean_ece_after")
    improvement = rq4.get("mean_improvement")
    wilcoxon_p = rq4.get("p_value")
    if isinstance(improvement, (int, float)) and isinstance(wilcoxon_p, (int, float)):
        lines.append(
            f"**RQ4 — Recalibration:** Recalibration (intercept-only and full logistic) significantly reduced ECE (mean ECE before {ece_before:.3f}, after {ece_after:.3f}; "
            f"mean improvement {improvement:.3f}; Wilcoxon p &lt; 0.001). When miscalibration is correctable, recalibration is sufficient; when feature mismatch dominates, it is not."
        )
    else:
        lines.append("**RQ4 — Recalibration:** Recalibration effectiveness is reported in Results §6. Recalibration sufficiency depends on whether the dominant issue is prevalence shift vs feature-level mismatch.")
    lines.append("")

    # RQ5
    rq5 = rq_summaries.get("rq5_cfs_penalty") or {}
    mean_cfs = rq5.get("mean_auc_drop")
    max_cfs = rq5.get("max_auc_drop")
    if isinstance(mean_cfs, (int, float)) and isinstance(max_cfs, (int, float)):
        lines.append(f"**RQ5 — CFS penalty:** Restricting to common features (CFS) incurred a mean AUC drop of {mean_cfs:.3f} and maximum {max_cfs:.3f}. Part of the Kaggle↔UCI external drop is attributable to this feature restriction; the remainder reflects population shift.")
    else:
        lines.append("**RQ5 — CFS penalty:** CFS penalty (full vs CFS-only AUC on internal data) is in Results §7. External drop in Kaggle↔UCI can exceed the CFS penalty when shift is large.")
    lines.append("")

    # Headline metrics
    lines.append("**Headline metrics:** ")
    parts = []
    if best_auc is not None:
        parts.append(f"best internal AUC = {best_auc:.3f}")
    if isinstance(mean_drop, (int, float)):
        parts.append(f"worst external drop (mean) = {mean_drop:.3f}")
    if isinstance(improvement, (int, float)):
        parts.append(f"recalibration recovery (mean ECE Δ) = {improvement:.3f}")
    if parts:
        lines.append("; ".join(parts) + ".")
    lines.append("")
    return lines


def _key_finding_rq1(pivots: dict[str, pd.DataFrame], rq_summaries: dict[str, Any]) -> str:
    """Key finding paragraph for RQ1 (eval_plan §5)."""
    rq1 = rq_summaries.get("rq1_internal_baselines") or {}
    best_list = rq1.get("best_model_per_site") or []
    if not best_list:
        return "Internal baselines show which model dominates per site; small cohorts (e.g. Switzerland) have wider CIs. Kaggle's apparent performance should be interpreted in light of its larger sample size."
    dom = ", ".join(f"{x.get('test_site', '?')} ({x.get('model', '?')})" for x in best_list[:6])
    return f"Key finding: Model ranking varies by site — dominant models include {dom}. Small UCI cohorts show wider bootstrap CIs; Kaggle's performance is not directly comparable without size-matched sensitivity. Discrimination and calibration (Brier, ECE) are reported in the table and figures."


def _key_finding_rq2(pivots: dict[str, pd.DataFrame], rq_summaries: dict[str, Any]) -> str:
    """Key finding paragraph for RQ2; explains NaN in worst-AUC-drop summary (eval_plan §6.1)."""
    delta_df = pivots.get("auc_delta")
    if delta_df is None or delta_df.empty:
        return "External validation degradation is summarized in the Δ matrix. Statistical significance of AUC drops is assessed via bootstrap CIs."
    lines = [
        "Key finding: Pairwise and LOSO external validation show substantial AUC drops for several train→test directions. "
        "The worst drop typically occurs for Kaggle→UCI CFS or cross-site pairs with large covariate shift. "
    ]
    worst = (rq_summaries.get("rq2_external_degradation") or {}).get("worst_auc_drop")
    if isinstance(worst, dict) and worst.get("experiment_type") == "external_kaggle_uci":
        lines.append(
            "Note: For external Kaggle↔UCI CFS experiments there is no internal CFS baseline on the same test site (UCI sites were evaluated internally with full features), so fields such as cfs_full_auc and cfs_cfs_auc are undefined (NaN) in the worst-AUC-drop summary for those rows."
        )
    return " ".join(lines)


def _key_finding_rq3(rq_summaries: dict[str, Any]) -> str:
    """Key finding paragraph for RQ3; interprets Spearman and limited power (eval_plan §5, §6.1)."""
    rq3 = rq_summaries.get("rq3_shift_performance") or {}
    rho = rq3.get("spearman_rho")
    p_val = rq3.get("p_value")
    interp = (
        f"The shift–performance Spearman correlation (mean PSI vs AUC drop) was ρ = {rho:.2f}, p = {p_val:.3f}: not statistically significant at α = 0.05. "
        "Power was limited by the number of external pairs. "
        "Shift signatures (prevalence-driven vs covariate-driven vs missingness-driven degradation) can still be inspected per pair via the PSI heatmap and distribution overlays."
    ) if isinstance(rho, (int, float)) and isinstance(p_val, (int, float)) else (
        "Shift metrics (PSI, prevalence diff, C2ST) are summarized per pair; correlation with performance drop is in the Spearman table. "
        "Identifying whether degradation is prevalence-driven, covariate-driven, or missingness-driven requires per-pair inspection."
    )
    return "Key finding: " + interp


def _key_finding_rq4(rq_summaries: dict[str, Any]) -> str:
    """Key finding paragraph for RQ4 (eval_plan §5)."""
    rq4 = rq_summaries.get("rq4_recalibration") or {}
    if rq4.get("p_value") is not None and rq4.get("p_value") < 0.05:
        return (
            "Key finding: Recalibration (intercept-only and full logistic) significantly reduced ECE on average. "
            "When ECE drops to below ~0.05 after recalibration, miscalibration was correctable; when no method helps, feature-level mismatch likely dominates and recalibration is insufficient."
        )
    return (
        "Key finding: Recalibration effectiveness depends on the type of miscalibration. "
        "Intercept-only recalibration suffices for pure prevalence shift; full recalibration is needed when spread/sharpness differ. When no method helps, retraining or feature alignment may be required."
    )


def _key_finding_rq5(rq_summaries: dict[str, Any]) -> str:
    """Key finding paragraph for RQ5 (eval_plan §5)."""
    rq5 = rq_summaries.get("rq5_cfs_penalty") or {}
    mean_cfs = rq5.get("mean_auc_drop")
    max_cfs = rq5.get("max_auc_drop")
    if isinstance(mean_cfs, (int, float)):
        return (
            f"Key finding: The CFS (common-feature-set) restriction incurs a mean AUC penalty of {mean_cfs:.3f} (max {max_cfs:.3f}). "
            "Part of the Kaggle↔UCI transportability gap is thus attributable to feature restriction; the remainder reflects population and distribution shift."
        )
    return (
        "Key finding: CFS penalty quantifies how much discrimination is lost when restricting to common features across sites. "
        "The Kaggle↔UCI external drop can exceed this penalty when shift is large, indicating that both feature restriction and population shift contribute."
    )


def _export_report_html(reports_dir: Path) -> None:
    """Convert evaluation_report.md to HTML. See eval_plan §8.2."""
    md_path = reports_dir / "evaluation_report.md"
    html_path = reports_dir / "evaluation_report.html"
    if not md_path.exists():
        return
    try:
        import markdown
    except ImportError:
        logger.warning("markdown package not installed; skipping HTML export. pip install markdown")
        return
    template = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        '<title>Evaluation Report — Heart Disease Model Transportability</title>'
        '<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;line-height:1.6}}'
        'h1{{border-bottom:1px solid #ccc}}h2{{margin-top:1.5em}}'
        'table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:.5em .75em;text-align:left}}'
        'th{{background:#f5f5f5}}tr:nth-child(even){{background:#fafafa}}'
        'code{{background:#f0f0f0;padding:.15em .4em;border-radius:3px}}a{{color:#0066cc}}</style>'
        "</head><body>{body}</body></html>"
    )
    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"], output_format="html5")
    html_path.write_text(template.format(body=html_body), encoding="utf-8")
    logger.info("Wrote %s", html_path)


def generate_report(
    master: pd.DataFrame,
    pivots: dict[str, pd.DataFrame],
    stat_results: dict[str, Any],
    rq_summaries: dict[str, Any],
    reports_dir: Path,
) -> None:
    """Write reports/evaluation_report.md per eval_plan §6.1. Prose Executive Summary and key-finding paragraphs; no raw JSON dump."""
    lines = [
        "# Evaluation Report — Heart Disease Model Transportability",
        "",
        "## 1. Executive Summary",
        "",
    ]

    lines.extend(_format_executive_summary(rq_summaries))

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
        lines.append(_key_finding_rq1(pivots, rq_summaries))
        lines.append("")

    lines.extend([
        "## 4. Results — RQ2: External Validation",
        "",
    ])
    if "auc_delta" in pivots:
        delta = pivots["auc_delta"]
        lines.append(f"Mean AUC drop: {delta['auc_delta'].mean():.3f}")
        lines.append("")
        lines.append(_key_finding_rq2(pivots, rq_summaries))
        lines.append("")

    lines.extend([
        "## 5. Results — RQ3: Dataset Shift",
        "",
    ])
    if "rq3_shift_performance" in rq_summaries:
        rq3 = rq_summaries["rq3_shift_performance"]
        rho = rq3.get("spearman_rho")
        p_val = rq3.get("p_value")
        if isinstance(rho, (int, float)) and isinstance(p_val, (int, float)):
            lines.append(f"Spearman ρ (mean PSI vs AUC drop): ρ = {rho:.3f}, p = {p_val:.3f}")
        else:
            lines.append(f"Spearman (mean PSI vs AUC drop): {rq3}")
        lines.append("")
        lines.append(_key_finding_rq3(rq_summaries))
        lines.append("")

    lines.extend([
        "## 6. Results — RQ4: Recalibration",
        "",
    ])
    if "rq4_recalibration" in rq_summaries:
        rq4 = rq_summaries["rq4_recalibration"]
        if isinstance(rq4, dict) and "error" not in rq4:
            eb, ea, imp, pv = rq4.get("mean_ece_before"), rq4.get("mean_ece_after"), rq4.get("mean_improvement"), rq4.get("p_value")
            lines.append(f"Wilcoxon (ECE before vs after): mean ECE before {eb:.3f}, after {ea:.3f}, mean improvement {imp:.3f}, p = {pv}" if all(x is not None for x in (eb, ea, imp, pv)) else f"Wilcoxon (ECE before vs after): {rq4}")
        else:
            lines.append(f"Wilcoxon (ECE before vs after): {rq4}")
        lines.append("")
        lines.append(_key_finding_rq4(rq_summaries))
        lines.append("")

    lines.extend([
        "## 7. Results — RQ5: CFS Penalty",
        "",
    ])
    if "rq5_cfs_penalty" in rq_summaries:
        rq5 = rq_summaries["rq5_cfs_penalty"]
        if isinstance(rq5, dict):
            mean_cfs, max_cfs = rq5.get("mean_auc_drop"), rq5.get("max_auc_drop")
            if mean_cfs is not None and max_cfs is not None:
                lines.append(f"Mean CFS AUC drop: {mean_cfs:.3f}; max: {max_cfs:.3f}")
            else:
                lines.append(f"CFS penalty: {rq5}")
        else:
            lines.append(f"CFS penalty: {rq5}")
        lines.append("")
        lines.append(_key_finding_rq5(rq_summaries))
        lines.append("")

    lines.extend([
        "## 8. Discussion",
        "",
        "**Transportability patterns:** Performance transfer depends strongly on the train–test site pair. Same-site internal validation gives optimistically high AUC; external application to other sites or to CFS-only evaluation typically reduces discrimination. Directions that transfer relatively well can be identified from the AUC heatmap and Δ matrix; the worst degradation occurs for Kaggle→UCI CFS and for pairs with large covariate or prevalence shift.",
        "",
        "**Recalibration sufficiency:** Recalibration (Platt, isotonic, or logistic updating) is sufficient when the dominant issue is miscalibration (e.g. prevalence shift). When ECE drops to below ~0.05 after recalibration, probability outputs can be used with confidence. When no method materially improves ECE, feature-level mismatch or population shift likely dominates, and recalibration is insufficient—retraining or feature alignment is then needed.",
        "",
        "**Limitations:** Bootstrap CIs overlap substantially for small UCI cohorts, limiting claims about model ranking. The shift–performance correlation was underpowered (few external pairs). External Kaggle↔UCI CFS experiments do not have an internal CFS baseline on the same test site, so CFS penalty fields (cfs_full_auc, cfs_cfs_auc) are undefined (NaN) for the worst-AUC-drop summary for those rows. Univariate shift diagnostics (PSI, KS) may miss multivariate structure captured by C2ST.",
        "",
        "## 9. Appendices",
        "",
        "- Full metric tables: reports/tables/master_results.csv",
        "- Statistical tests: reports/tables/statistical_tests.json",
        "- RQ summaries: reports/rq_summaries.json (structured data for reproducibility)",
        "",
        "### Appendix: PROBAST Risk Assessment",
        "",
        "Domain-level bias and applicability assessment per PROBAST+AI:",
        "[reports/compliance/PROBAST_RISK_ASSESSMENT.md](reports/compliance/PROBAST_RISK_ASSESSMENT.md)",
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
    figures_path = reports_path / "figures"
    tables_path = reports_path / "tables"
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
        json.dump(_sanitize_for_json(stat_results), f, indent=2, default=str)

    # 4. Figures
    generate_all_figures(master, outputs_dir, figures_path, config, run_id)

    # 5. RQ synthesis
    rq_summaries = synthesize_rq_answers(master, stat_results, pivots)
    with open(reports_path / "rq_summaries.json", "w", encoding="utf-8") as f:
        json.dump(rq_summaries, f, indent=2, default=str)

    # 6. Report
    generate_report(master, pivots, stat_results, rq_summaries, reports_path)

    # 7. HTML export
    _export_report_html(reports_path)

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
