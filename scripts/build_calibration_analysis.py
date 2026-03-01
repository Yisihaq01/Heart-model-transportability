#!/usr/bin/env python3
"""
Formal Calibration Failure Analysis Layer.

Consolidates before/after metrics across all calibration methods (Platt, isotonic,
temperature, intercept-only, intercept+slope), computes deltas, win/loss flags,
failure categories, and rule-based outcome labels. Produces a master analysis table,
paper-ready summary, and interpretation note.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
CALIBRATION = OUTPUTS / "calibration"
PAPER_READY = OUTPUTS / "paper_ready"
ANALYSIS_OUT = PAPER_READY / "calibration_analysis"

# Thresholds for outcome labeling (lower is better for all metrics)
THRESHOLD_ECE_IMPROVE = -0.01   # ECE delta < -0.01 = improved
THRESHOLD_ECE_DEGRADE = 0.01    # ECE delta > 0.01 = degraded
THRESHOLD_BRIER_IMPROVE = -0.01
THRESHOLD_BRIER_DEGRADE = 0.01
THRESHOLD_MCE_IMPROVE = -0.01
THRESHOLD_MCE_DEGRADE = 0.01


def load_json(p: Path) -> Any:
    with open(p, encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r":\s*NaN\b", ": null", text)
    text = re.sub(r":\s*Infinity\b", ": null", text)
    text = re.sub(r":\s*-Infinity\b", ": null", text)
    return json.loads(text)


def safe_float(x: Any) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    return float(x)


def parse_experiment_id(exp_id: str, _method_or_variant: str) -> dict[str, str]:
    """Parse experiment_id to extract experiment_type, dataset_direction, model."""
    stem = exp_id.replace(".json", "")
    result: dict[str, str] = {"experiment_id": exp_id, "experiment_variant": ""}

    if "__to__" not in stem:
        result["experiment_type"] = "unknown"
        result["dataset_direction"] = ""
        result["model"] = ""
        return result

    if stem.startswith("external_uci__"):
        # external_uci__va__to__switzerland__xgb
        result["experiment_type"] = "external_uci"
        rest = stem.replace("external_uci__", "")
    elif stem.startswith("external_kaggle_uci__"):
        # external_kaggle_uci__cfs_plus_chol__va__to__kaggle__xgb
        result["experiment_type"] = "external_kaggle_uci"
        rest = stem.replace("external_kaggle_uci__", "")
        if "__" in rest:
            variant, rest = rest.split("__", 1)
            result["experiment_variant"] = variant
    else:
        result["experiment_type"] = "unknown"
        result["dataset_direction"] = ""
        result["model"] = ""
        return result

    train_part, test_model = rest.split("__to__", 1)
    if "__" in test_model:
        test_site, model = test_model.rsplit("__", 1)
        result["dataset_direction"] = f"{train_part}__to__{test_site}"
        result["model"] = model
    else:
        result["dataset_direction"] = rest
        result["model"] = ""
    return result


def model_family(model: str) -> str:
    """Map model to family for aggregation."""
    if model == "lr":
        return "linear"
    if model in ("rf", "xgb", "lgbm"):
        return "tree"
    return "other"


def collect_calibration_records() -> pd.DataFrame:
    """Collect all calibration before/after records from recalibration and updating."""
    rows: list[dict[str, Any]] = []

    # Recalibration: platt, isotonic, temperature
    for method in ["platt", "isotonic", "temperature"]:
        recalc_dir = CALIBRATION / "recalibration" / method
        if not recalc_dir.exists():
            continue
        for f in recalc_dir.glob("*.json"):
            if f.name == "summary.json":
                continue
            try:
                data = load_json(f)
                mb = data.get("metrics_before", {})
                ma = data.get("metrics_after", {})
                exp_id = f.name
                parsed = parse_experiment_id(exp_id, method)
                exp_type = data.get("experiment_type") or parsed.get("experiment_type", "")
                model = data.get("model") or parsed.get("model", "")

                brier_b = safe_float(mb.get("brier_score"))
                brier_a = safe_float(ma.get("brier_score"))
                ece_b = safe_float(mb.get("ece"))
                ece_a = safe_float(ma.get("ece"))
                mce_b = safe_float(mb.get("mce"))
                mce_a = safe_float(ma.get("mce"))

                rows.append({
                    "method": method,
                    "experiment_type": exp_type,
                    "experiment_variant": parsed.get("experiment_variant", ""),
                    "dataset_direction": parsed.get("dataset_direction", ""),
                    "model": model,
                    "brier_before": brier_b,
                    "brier_after": brier_a,
                    "ece_before": ece_b,
                    "ece_after": ece_a,
                    "mce_before": mce_b,
                    "mce_after": mce_a,
                    "brier_delta": brier_a - brier_b,
                    "ece_delta": ece_a - ece_b,
                    "mce_delta": mce_a - mce_b,
                    "experiment_id": exp_id,
                })
            except Exception:
                pass

    # Updating: intercept_only, intercept_slope
    for variant in ["intercept_only", "intercept_slope"]:
        upd_dir = CALIBRATION / "updating" / variant
        if not upd_dir.exists():
            continue
        for f in upd_dir.glob("*.json"):
            if f.name == "summary.json":
                continue
            try:
                data = load_json(f)
                mb = data.get("metrics_before", {})
                ma = data.get("metrics_after", {})
                exp_id = f.name
                parsed = parse_experiment_id(exp_id, variant)
                exp_type = data.get("experiment_type") or parsed.get("experiment_type", "")
                model = data.get("model") or parsed.get("model", "")

                brier_b = safe_float(mb.get("brier_score"))
                brier_a = safe_float(ma.get("brier_score"))
                ece_b = safe_float(mb.get("ece"))
                ece_a = safe_float(ma.get("ece"))
                mce_b = safe_float(mb.get("mce"))
                mce_a = safe_float(ma.get("mce"))

                rows.append({
                    "method": variant,
                    "experiment_type": exp_type,
                    "experiment_variant": parsed.get("experiment_variant", ""),
                    "dataset_direction": parsed.get("dataset_direction", ""),
                    "model": model,
                    "brier_before": brier_b,
                    "brier_after": brier_a,
                    "ece_before": ece_b,
                    "ece_after": ece_a,
                    "mce_before": mce_b,
                    "mce_after": mce_a,
                    "brier_delta": brier_a - brier_b,
                    "ece_delta": ece_a - ece_b,
                    "mce_delta": mce_a - mce_b,
                    "experiment_id": exp_id,
                })
            except Exception:
                pass

    return pd.DataFrame(rows)


def add_win_loss_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add win/loss flags per metric (win = improvement = negative delta for Brier/ECE/MCE)."""
    df = df.copy()
    df["brier_win"] = df["brier_delta"] < 0
    df["ece_win"] = df["ece_delta"] < 0
    df["mce_win"] = df["mce_delta"] < 0
    df["all_win"] = df["brier_win"] & df["ece_win"] & df["mce_win"]
    df["any_degrade"] = (df["brier_delta"] > 0) | (df["ece_delta"] > 0) | (df["mce_delta"] > 0)
    return df


def add_outcome_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based outcome labels: improved, neutral, degraded."""
    def label_row(row: pd.Series) -> str:
        ece_d = row.get("ece_delta", 0) or 0
        brier_d = row.get("brier_delta", 0) or 0
        mce_d = row.get("mce_delta", 0) or 0
        if pd.isna(ece_d):
            ece_d = 0
        if pd.isna(brier_d):
            brier_d = 0
        if pd.isna(mce_d):
            mce_d = 0

        improves = 0
        degrades = 0
        if ece_d < THRESHOLD_ECE_IMPROVE:
            improves += 1
        elif ece_d > THRESHOLD_ECE_DEGRADE:
            degrades += 1
        if brier_d < THRESHOLD_BRIER_IMPROVE:
            improves += 1
        elif brier_d > THRESHOLD_BRIER_DEGRADE:
            degrades += 1
        if mce_d < THRESHOLD_MCE_IMPROVE:
            improves += 1
        elif mce_d > THRESHOLD_MCE_DEGRADE:
            degrades += 1

        if degrades >= 2 or (degrades >= 1 and improves == 0):
            return "degraded"
        if improves >= 2:
            return "improved"
        return "neutral"

    df = df.copy()
    df["outcome"] = df.apply(label_row, axis=1)
    return df


def add_failure_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Explicit failure mode categories."""
    def categorize(row: pd.Series) -> str:
        if row.get("outcome") == "improved":
            return "success"
        ece_d = row.get("ece_delta") or 0
        brier_d = row.get("brier_delta") or 0
        mce_d = row.get("mce_delta") or 0
        if pd.isna(ece_d):
            ece_d = 0
        if pd.isna(brier_d):
            brier_d = 0
        if pd.isna(mce_d):
            mce_d = 0

        modes = []
        if ece_d > THRESHOLD_ECE_DEGRADE:
            modes.append("ece_degraded")
        if brier_d > THRESHOLD_BRIER_DEGRADE:
            modes.append("brier_degraded")
        if mce_d > THRESHOLD_MCE_DEGRADE:
            modes.append("mce_degraded")
        if ece_d > 0.05:
            modes.append("ece_severe")
        if mce_d > 0.1:
            modes.append("mce_severe")

        if not modes:
            return "neutral"
        return "|".join(sorted(set(modes)))

    df = df.copy()
    df["failure_category"] = df.apply(categorize, axis=1)
    return df


def build_master_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build consolidated master calibration-analysis table."""
    df = add_win_loss_flags(df)
    df = add_outcome_labels(df)
    df = add_failure_categories(df)
    df["model_family"] = df["model"].map(model_family)
    return df


def build_summary_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Counts/percentages of improved vs degraded per method."""
    if df.empty:
        return pd.DataFrame()
    summary = df.groupby("method").agg({
        "outcome": ["count", lambda x: (x == "improved").sum(), lambda x: (x == "degraded").sum(), lambda x: (x == "neutral").sum()],
    }).reset_index()
    summary.columns = ["method", "n", "improved", "degraded", "neutral"]
    summary["pct_improved"] = (summary["improved"] / summary["n"] * 100).round(1)
    summary["pct_degraded"] = (summary["degraded"] / summary["n"] * 100).round(1)
    summary["pct_neutral"] = (summary["neutral"] / summary["n"] * 100).round(1)
    return summary


def build_failure_modes_table(df: pd.DataFrame) -> pd.DataFrame:
    """Document failure modes explicitly."""
    if df.empty:
        return pd.DataFrame()
    failed = df[df["failure_category"] != "success"]
    if failed.empty:
        return pd.DataFrame(columns=["failure_category", "method", "count"])
    modes = failed.groupby(["failure_category", "method"]).size().reset_index(name="count")
    return modes.sort_values("count", ascending=False)


def build_dataset_direction_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary by dataset direction and method."""
    if df.empty:
        return pd.DataFrame()
    return df.groupby(["dataset_direction", "method"]).agg({
        "outcome": lambda x: x.value_counts().to_dict(),
        "brier_delta": "mean",
        "ece_delta": "mean",
        "mce_delta": "mean",
        "n": "count",
    }).reset_index()


def build_model_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary by model family and method."""
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby(["model_family", "method"]).agg({
        "outcome": ["count", lambda x: (x == "improved").sum(), lambda x: (x == "degraded").sum()],
        "brier_delta": "mean",
        "ece_delta": "mean",
        "mce_delta": "mean",
    }).reset_index()
    agg.columns = ["model_family", "method", "n", "improved", "degraded", "brier_delta_mean", "ece_delta_mean", "mce_delta_mean"]
    agg["pct_improved"] = (agg["improved"] / agg["n"] * 100).round(1)
    agg["pct_degraded"] = (agg["degraded"] / agg["n"] * 100).round(1)
    return agg


def write_interpretation_note(master: pd.DataFrame, summary: pd.DataFrame, failure_modes: pd.DataFrame) -> None:
    """Write short interpretation note."""
    lines = [
        "# Calibration Failure Analysis — Interpretation Note",
        "",
        "## Overview",
        "This analysis compares before vs after calibration across five methods:",
        "Platt scaling, isotonic regression, temperature scaling, intercept-only,",
        "and intercept+slope logistic recalibration.",
        "",
        "## Key Findings",
        "",
    ]
    if not summary.empty:
        best = summary.loc[summary["pct_improved"].idxmax()]
        worst = summary.loc[summary["pct_degraded"].idxmax()]
        lines.append(f"- **Best improvement rate**: {best['method']} ({best['pct_improved']}% improved)")
        lines.append(f"- **Highest degradation rate**: {worst['method']} ({worst['pct_degraded']}% degraded)")
        lines.append("")

    if not failure_modes.empty:
        lines.append("## Documented Failure Modes")
        lines.append("")
        for _, row in failure_modes.head(15).iterrows():
            lines.append(f"- `{row['failure_category']}` ({row['method']}): {row['count']} occurrences")
        lines.append("")

    lines.append("## Failure Mode Definitions")
    lines.append("- `ece_degraded`: ECE increased after recalibration")
    lines.append("- `brier_degraded`: Brier score increased")
    lines.append("- `mce_degraded`: MCE increased")
    lines.append("- `ece_severe`: ECE delta > 0.05")
    lines.append("- `mce_severe`: MCE delta > 0.1")
    lines.append("")
    lines.append("## Outcome Labels")
    lines.append("- **improved**: ≥2 metrics improved, <2 degraded")
    lines.append("- **neutral**: Mixed or no significant change")
    lines.append("- **degraded**: ≥2 metrics degraded, or ≥1 degraded with no improvement")
    (ANALYSIS_OUT / "INTERPRETATION_NOTE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)

    df = collect_calibration_records()
    if df.empty:
        print("No calibration records found. Ensure outputs/calibration/ has recalibration and updating JSONs.")
        return

    # Add n for groupby
    df["n"] = 1

    master = build_master_table(df)
    master.to_csv(ANALYSIS_OUT / "calibration_analysis_master.csv", index=False, float_format="%.6f")

    summary = build_summary_counts(master)
    summary.to_csv(ANALYSIS_OUT / "calibration_summary_by_method.csv", index=False, float_format="%.2f")

    failure_modes = build_failure_modes_table(master)
    if not failure_modes.empty:
        failure_modes.to_csv(ANALYSIS_OUT / "failure_modes.csv", index=False)

    by_direction = master.groupby(["dataset_direction", "method"]).agg({
        "outcome": lambda x: (x == "improved").sum(),
        "n": "count",
    }).reset_index()
    by_direction["pct_improved"] = (by_direction["outcome"] / by_direction["n"] * 100).round(1)
    by_direction.rename(columns={"outcome": "improved"}, inplace=True)
    by_direction.to_csv(ANALYSIS_OUT / "summary_by_direction.csv", index=False, float_format="%.2f")

    by_model_family = build_model_family_summary(master)
    by_model_family.to_csv(ANALYSIS_OUT / "summary_by_model_family.csv", index=False, float_format="%.4f")

    write_interpretation_note(master, summary, failure_modes)

    # Paper-ready summary artifact (compact)
    paper_summary = [
        "# Calibration Failure Analysis — Paper-Ready Summary",
        "",
        "## Master Table",
        f"Total experiments: {len(master)}",
        f"Methods: {', '.join(master['method'].unique().tolist())}",
        "",
        "## Outcome Distribution by Method",
    ]
    if not summary.empty:
        paper_summary.append("")
        paper_summary.append("| Method | N | Improved | Degraded | Neutral | % Improved | % Degraded |")
        paper_summary.append("|--------|---|----------|----------|---------|------------|------------|")
        for _, row in summary.iterrows():
            paper_summary.append(f"| {row['method']} | {row['n']} | {row['improved']} | {row['degraded']} | {row['neutral']} | {row['pct_improved']}% | {row['pct_degraded']}% |")

    paper_summary.append("")
    paper_summary.append("## Failure Modes (Top)")
    if not failure_modes.empty:
        for _, row in failure_modes.head(10).iterrows():
            paper_summary.append(f"- {row['failure_category']} ({row['method']}): {row['count']}")

    (ANALYSIS_OUT / "PAPER_READY_SUMMARY.md").write_text("\n".join(paper_summary), encoding="utf-8")

    # Copy master table to paper_ready root for consistency with other tables
    master.to_csv(PAPER_READY / "table_calibration_analysis_master.csv", index=False, float_format="%.6f")

    print(f"Calibration analysis written to {ANALYSIS_OUT}")
    print(f"  Master table also: {PAPER_READY / 'table_calibration_analysis_master.csv'}")
    for f in sorted(ANALYSIS_OUT.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
