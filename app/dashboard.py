#!/usr/bin/env python3
"""
Transportability Dashboard — Phase 2 deliverable.
Streamlit app: dataset overview, internal/external results, calibration, shift diagnostics.
Reads from outputs/runs/{run_id}/ (configurable run_id or latest).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Model Transportability",
    page_icon="❤️",
    layout="wide",
)


def _runs_dir() -> Path:
    return ROOT / "outputs" / "runs"


def _available_run_ids() -> list[str]:
    runs = _runs_dir()
    if not runs.exists():
        return []
    dirs = sorted([d.name for d in runs.iterdir() if d.is_dir()], reverse=True)
    return dirs


def _resolve_run_root(run_id: str | None) -> Path | None:
    if run_id:
        p = _runs_dir() / run_id
        return p if p.exists() else None
    ids = _available_run_ids()
    if not ids:
        return None
    return _runs_dir() / ids[0]


def _load_master(run_root: Path) -> pd.DataFrame:
    """Build master table from run_root (same logic as evaluation.build_master_table)."""
    rows = []
    for results_path in run_root.rglob("results.json"):
        try:
            with open(results_path, encoding="utf-8") as f:
                r = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

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
            "roc_auc": metrics.get("roc_auc"),
            "brier_score": metrics.get("brier_score"),
            "ece": metrics.get("ece"),
            "prevalence": metrics.get("prevalence"),
        }
        for k, v in (r.get("bootstrap_cis") or {}).items():
            if isinstance(v, dict):
                row[f"{k}_ci_lower"] = v.get("ci_lower")
                row[f"{k}_ci_upper"] = v.get("ci_upper")
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["experiment_type", "train_sites", "test_site", "model"]).reset_index(drop=True)


def _load_ingestion_report() -> dict | None:
    path = ROOT / "data" / "ingestion_report.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_manifest(run_root: Path) -> dict | None:
    path = run_root / "manifest.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_recalibration_summary(run_root: Path) -> dict | None:
    path = run_root / "calibration" / "recalibration" / "platt" / "summary.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_shift_table(run_root: Path) -> pd.DataFrame | None:
    for name in ["shift_table.parquet", "shift_table.csv"]:
        path = run_root / "shift" / name
        if path.exists():
            return pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path)
    return None


def _load_shift_perf_merged(run_root: Path) -> pd.DataFrame | None:
    for name in ["shift_performance_merged.parquet", "shift_performance_merged.csv"]:
        path = run_root / "shift" / name
        if path.exists():
            return pd.read_parquet(path) if name.endswith(".parquet") else pd.read_csv(path)
    return None


def main() -> None:
    st.title("❤️ Heart Model Transportability Dashboard")
    st.caption("Phase 2 — Dataset overview, internal/external validation, calibration, shift diagnostics")

    run_ids = _available_run_ids()
    if not run_ids:
        st.error("No runs found under outputs/runs/. Run the pipeline first: python scripts/run_pipeline.py")
        return

    run_id = st.sidebar.selectbox(
        "Run ID",
        run_ids,
        index=0,
        help="Select pipeline run. Latest run is default.",
    )
    run_root = _resolve_run_root(run_id)
    if not run_root:
        st.error(f"Run {run_id} not found.")
        return

    manifest = _load_manifest(run_root)
    if manifest:
        st.sidebar.metric("Artifacts", manifest.get("artifact_count", "—"))
        st.sidebar.caption(f"Config hash: {manifest.get('config_hash', '—')}")

    master = _load_master(run_root)
    if master.empty:
        st.warning("No results.json found in this run.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Dataset Overview",
        "Internal 80/20",
        "External Matrix",
        "Calibration",
        "Shift & Overlap",
        "Export Report",
    ])

    # --- Tab 1: Dataset overview ---
    with tab1:
        ing = _load_ingestion_report()
        if ing and "sites" in ing:
            sites_data = []
            for site, meta in ing["sites"].items():
                sites_data.append({
                    "Site": site,
                    "N": meta.get("n_records", "—"),
                    "Features": meta.get("n_features", "—"),
                    "Prevalence": f"{meta.get('prevalence', 0):.1%}" if isinstance(meta.get("prevalence"), (int, float)) else "—",
                    "Missing (heavy)": len([k for k, v in (meta.get("missing_rates") or {}).items() if isinstance(v, (int, float)) and v > 30]),
                })
            st.subheader("Site overview")
            st.dataframe(pd.DataFrame(sites_data), use_container_width=True, hide_index=True)
            st.caption("CFS Kaggle↔UCI: " + ", ".join(ing.get("cfs_kaggle_uci", [])))
        else:
            st.info("No ingestion_report.json found. Run ingestion first.")

    internal = master[master["experiment_type"] == "internal"]
    external = master[master["experiment_type"].str.startswith("external", na=False)]

    # --- Tab 2: Internal 80/20 ---
    with tab2:
        if internal.empty:
            st.info("No internal validation results.")
        else:
            pivot = internal.pivot_table(
                index="test_site",
                columns="model",
                values="roc_auc",
                aggfunc="first",
            ).round(3)
            st.subheader("Internal ROC-AUC (80/20)")
            st.dataframe(pivot, use_container_width=True)
            if "roc_auc_ci_lower" in internal.columns:
                st.caption("Bootstrap 95% CIs available in master table.")

    # --- Tab 3: External validation matrix ---
    with tab3:
        if external.empty:
            st.info("No external validation results.")
        else:
            model_sel = st.selectbox("Model", external["model"].unique(), key="ext_model")
            ext_m = external[external["model"] == model_sel]
            pivot = ext_m.pivot_table(
                index="train_sites",
                columns="test_site",
                values="roc_auc",
                aggfunc="first",
            ).round(3)
            st.subheader(f"External ROC-AUC — {model_sel.upper()}")
            st.dataframe(pivot, use_container_width=True)
            # AUC delta
            internal_auc = master[master["experiment_type"] == "internal"].set_index(["test_site", "model"])["roc_auc"]
            ext_m = ext_m.copy()
            ext_m["internal_auc"] = ext_m.apply(
                lambda r: internal_auc.get((r["test_site"], r["model"]), None), axis=1
            )
            ext_m["auc_delta"] = ext_m["roc_auc"] - ext_m["internal_auc"]
            st.subheader("AUC drop (external − internal)")
            delta_pivot = ext_m.pivot_table(index="train_sites", columns="test_site", values="auc_delta", aggfunc="first").round(3)
            st.dataframe(delta_pivot, use_container_width=True)

    # --- Tab 4: Calibration ---
    with tab4:
        cal_cols = [c for c in ["roc_auc", "brier_score", "ece"] if c in master.columns]
        if cal_cols:
            cal_summary = master[master["experiment_type"].str.startswith("external", na=False)].groupby(
                ["train_sites", "test_site", "model"]
            )[cal_cols].mean().round(4)
            st.subheader("Calibration metrics (external experiments)")
            st.dataframe(cal_summary, use_container_width=True)
        rec_sum = _load_recalibration_summary(run_root)
        if rec_sum:
            st.subheader("Recalibration (Platt) summary")
            if isinstance(rec_sum, list):
                st.caption(f"{len(rec_sum)} experiments. Sample (first 3):")
                st.json(rec_sum[:3] if len(rec_sum) > 3 else rec_sum)
            else:
                st.json(rec_sum)
        else:
            st.caption("Recalibration summary not found.")

    # --- Tab 5: Shift & overlap ---
    with tab5:
        shift_table = _load_shift_table(run_root)
        shift_perf = _load_shift_perf_merged(run_root)
        if shift_table is not None and not shift_table.empty:
            st.subheader("Shift diagnostics (per pair)")
            st.dataframe(shift_table.head(50), use_container_width=True)
        if shift_perf is not None and not shift_perf.empty:
            st.subheader("Shift vs performance")
            st.dataframe(shift_perf.head(30), use_container_width=True)
        if shift_table is None and shift_perf is None:
            st.info("Shift artifacts not found for this run.")
        # Effective CFS / overlap
        ext = master[master["experiment_type"].str.startswith("external", na=False)]
        if "n_features_used" in ext.columns:
            feat_pivot = ext.pivot_table(
                index=["train_sites", "test_site"],
                values="n_features_used",
                aggfunc="first",
            )
            st.subheader("Features used (effective CFS per pair)")
            st.dataframe(feat_pivot, use_container_width=True)

    # --- Tab 6: Export report ---
    with tab6:
        st.subheader("Model card / TRIPOD+AI summary")
        st.markdown("""
        **Intended use:** Research benchmark for heart-disease prediction model transportability.

        **Data:** Kaggle CVD, UCI Heart Disease (Cleveland, Hungary, Switzerland, VA).

        **Models:** LR, RF, XGBoost, LightGBM.

        **Validation:** Internal 80/20; external multi-site matrix; CFS penalty; recalibration.

        **Limitations:** Label/measurement heterogeneity; small UCI cohorts; research-only.
        """)
        ib = internal.pivot_table(index="test_site", columns="model", values="roc_auc", aggfunc="first").round(3) if not internal.empty else None
        ib_str = ib.to_string() if ib is not None else "N/A"
        report_md = f"""# Transportability Summary — Run {run_id}

## Run metadata
- Run ID: {run_id}
- Artifacts: {manifest.get('artifact_count', '—') if manifest else '—'}

## Internal baselines (ROC-AUC)
{ib_str}

## Key findings
- See reports/evaluation_report.md for full analysis.
- PROBAST: reports/compliance/PROBAST_RISK_ASSESSMENT.md
"""
        st.code(report_md, language="markdown")
        st.download_button(
            "Download summary (Markdown)",
            report_md,
            file_name=f"transportability_summary_{run_id}.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
