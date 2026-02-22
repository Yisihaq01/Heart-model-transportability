"""
Data ingestion pipeline per ImplementationPlan/data_ingestion.md.
Load, parse, validate, standardize raw data; write cleaned parquet + ingestion_report.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

# --- Constants from data_ingestion.md §3.3, §3.4 ---
LINES_PER_RECORD = 10
ATTR_MAP = {
    "age": (0, 2),
    "sex": (0, 3),
    "cp": (1, 1),
    "trestbps": (1, 2),
    "chol": (1, 4),
    "fbs": (1, 7),
    "restecg": (2, 2),
    "thalach": (3, 7),
    "exang": (4, 6),
    "oldpeak": (5, 0),
    "slope": (5, 1),
    "ca": (5, 4),
    "thal": (5, 5),
    "num": (7, 2),
}
PROCESSED_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num",
]

EXPECTED_RECORDS = {"kaggle": 70_000, "cleveland": 303, "hungary": 294, "switzerland": 123, "va": 200}


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --- Loaders (§2.3, §3.3, §3.4) ---

def load_kaggle(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, sep=";")
    df.drop(columns=["id"], inplace=True)
    df["age"] = (df["age"] / 365.25).round(1)
    df["gender"] = df["gender"].map({1: 0, 2: 1})
    bp_mask = (
        (df["ap_hi"] < 60) | (df["ap_hi"] > 250)
        | (df["ap_lo"] < 30) | (df["ap_lo"] > 200)
        | (df["ap_lo"] >= df["ap_hi"])
    )
    df["bp_outlier"] = bp_mask.astype(int)
    df["site"] = "kaggle"
    return df


def _parse_value(token: str) -> float | None:
    token = token.strip()
    if token in ("-9", "-9."):
        return np.nan
    return float(token)


def load_uci_long(path: str | Path, site_label: str) -> pd.DataFrame:
    path = Path(path)
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    records = []
    for start in range(0, len(lines), LINES_PER_RECORD):
        block = lines[start : start + LINES_PER_RECORD]
        if len(block) < LINES_PER_RECORD:
            break
        split_lines = [line.split() for line in block]
        row = {}
        for attr, (li, ci) in ATTR_MAP.items():
            row[attr] = _parse_value(split_lines[li][ci])
        records.append(row)
    df = pd.DataFrame(records)
    df["site"] = site_label
    return df


def load_uci_processed(path: str | Path, site_label: str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, header=None, names=PROCESSED_COLS, na_values=["?"])
    df["site"] = site_label
    return df


# --- Standardize (§4.2), recode missing (§5.1), binarize (§3.5), drop BP outliers (§9) ---

def standardize_columns(df: pd.DataFrame, site: str) -> pd.DataFrame:
    out = df.copy()
    if site == "kaggle":
        out = out.rename(columns={
            "gender": "sex",
            "ap_hi": "sys_bp",
            "ap_lo": "dia_bp",
            "cardio": "target",
        })
    else:
        out["sys_bp"] = out["trestbps"]
        out["target"] = (out["num"] >= 1).astype(int)
    return out


def recode_missing(df: pd.DataFrame, site: str) -> pd.DataFrame:
    out = df.copy()
    if site == "switzerland" and "chol" in out.columns:
        out.loc[out["chol"] == 0, "chol"] = np.nan
    return out


def binarize_target(df: pd.DataFrame, site: str) -> pd.DataFrame:
    out = df.copy()
    if site != "kaggle" and "num" in out.columns and "target" not in out.columns:
        out["target"] = (out["num"] >= 1).astype(int)
    elif site == "kaggle" and "cardio" in out.columns and "target" not in out.columns:
        out["target"] = out["cardio"]
    return out


def drop_kaggle_bp_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows with ap_hi > 370 or ap_lo < 0 or ap_lo > ap_hi. Returns (df, n_dropped)."""
    if "sys_bp" not in df.columns or "dia_bp" not in df.columns:
        return df, 0
    mask = (df["sys_bp"] > 370) | (df["dia_bp"] < 0) | (df["dia_bp"] > df["sys_bp"])
    n_dropped = mask.sum()
    return df.loc[~mask].copy(), int(n_dropped)


# --- Validation (§7) ---

def validate_site(df: pd.DataFrame, site: str) -> None:
    expected = EXPECTED_RECORDS.get(site)
    if expected is not None and site != "kaggle":
        assert len(df) == expected, f"{site}: row count mismatch (got {len(df)}, expected {expected})"
    if site == "kaggle":
        assert 0 < len(df) <= expected, f"{site}: row count out of range (got {len(df)})"
    assert "target" in df.columns, f"{site}: missing target"
    assert df["target"].isin([0, 1]).all(), f"{site}: target not binary"
    if "sex" in df.columns:
        assert df["sex"].dropna().isin([0, 1]).all(), f"{site}: sex not binary"
    if "age" in df.columns:
        assert df["age"].dropna().between(0, 120).all(), f"{site}: age out of range"
    assert "site" in df.columns
    assert (df["site"] == site).all(), f"{site}: site column mismatch"


# --- Missingness (§5.2) ---

def profile_missing(df: pd.DataFrame, site: str) -> pd.DataFrame:
    stats = pd.DataFrame({
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(1),
    })
    stats["site"] = site
    return stats


def missing_rates_dict(df: pd.DataFrame) -> dict[str, float]:
    pct = (df.isna().mean() * 100).round(1)
    return {k: float(v) for k, v in pct.items() if v > 0}


# --- Output (§6) ---

def write_outputs(
    frames: dict[str, pd.DataFrame],
    uci_all: pd.DataFrame,
    cfg: dict,
    cfs_kaggle_uci: list[str],
    cfs_uci_cross_site: list[str],
) -> None:
    root = Path(cfg.get("root", "."))
    clean_dir = root / cfg["clean_dir"]
    clean_dir.mkdir(parents=True, exist_ok=True)

    for site, df in frames.items():
        out_path = clean_dir / f"{site}_clean.parquet" if site == "kaggle" else clean_dir / f"uci_{site}_clean.parquet"
        df.to_parquet(out_path, index=False)

    uci_all.to_parquet(clean_dir / "uci_all_sites.parquet", index=False)

    # CFS Kaggle ↔ UCI: only sites that have age, sex, sys_bp
    kaggle_df = frames.get("kaggle")
    uci_sites = [s for s in ["cleveland", "hungary", "switzerland", "va"] if s in frames]
    if kaggle_df is not None and uci_sites:
        cfs_cols = [c for c in cfs_kaggle_uci if c in kaggle_df.columns]
        k_cfs = kaggle_df[["site", "target"] + cfs_cols].copy()
        uci_dfs = []
        for s in uci_sites:
            d = frames[s]
            cols = [c for c in cfs_kaggle_uci if c in d.columns]
            if "sys_bp" not in cols and "trestbps" in d.columns:
                d = d.copy()
                d["sys_bp"] = d["trestbps"]
                cols = list(cfs_kaggle_uci)
            uci_dfs.append(d[["site", "target"] + cols].copy())
        cfs_uci = pd.concat(uci_dfs, ignore_index=True)
        cfs_all = pd.concat([k_cfs, cfs_uci], ignore_index=True)
        cfs_all.to_parquet(clean_dir / "cfs_kaggle_uci.parquet", index=False)

    # CFS UCI cross-site
    if uci_sites:
        cols = [c for c in cfs_uci_cross_site if c in uci_all.columns]
        cfs_uci_all = uci_all[["site", "target"] + cols].copy()
        cfs_uci_all.to_parquet(clean_dir / "cfs_uci_cross_site.parquet", index=False)


def write_ingestion_report(
    frames: dict[str, pd.DataFrame],
    cfg: dict,
    bp_outliers_dropped: int | None = None,
) -> None:
    root = Path(cfg.get("root", "."))
    clean_dir = root / cfg["clean_dir"]
    clean_dir.mkdir(parents=True, exist_ok=True)

    sites_cfg = cfg.get("sites", {})
    cfs_ku = cfg.get("cfs", {}).get("kaggle_uci", ["age", "sex", "sys_bp"])
    cfs_uu = cfg.get("cfs", {}).get("uci_cross_site", [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"
    ])

    report = {
        "generated_at": datetime.now().isoformat(),
        "sites": {},
        "cfs_kaggle_uci": cfs_ku,
        "cfs_uci_cross_site": cfs_uu,
    }

    for site, df in frames.items():
        n = len(df)
        target_col = "target"
        prevalence = float(df["target"].mean()) if "target" in df.columns else None
        missing_rates = missing_rates_dict(df)
        feature_cols = [c for c in df.columns if c not in ("site", "target")]
        n_features = len(feature_cols)

        entry = {
            "n_records": n,
            "n_features": n_features,
            "target_col": target_col,
            "prevalence": round(prevalence, 3) if prevalence is not None else None,
            "missing_rates": missing_rates,
            "source_file": sites_cfg.get(site, {}).get("file", ""),
        }
        if site == "kaggle":
            bp_flagged = int(df["bp_outlier"].sum()) if "bp_outlier" in df.columns else 0
            entry["bp_outliers_flagged"] = bp_flagged
            if bp_outliers_dropped is not None:
                entry["bp_outliers_dropped"] = bp_outliers_dropped
        report["sites"][site] = entry

    with open(clean_dir / "ingestion_report.json", "w") as f:
        json.dump(report, f, indent=2)


# --- Orchestration (§8) ---

def run_ingestion(config_path: str | Path = "configs/data_ingestion.yaml", root: Path | None = None) -> dict[str, pd.DataFrame]:
    config_path = Path(config_path)
    root = root or config_path.resolve().parent.parent
    cfg = load_config(config_path)
    cfg["root"] = str(root)

    raw_dir = root / cfg["raw_dir"]
    sites_cfg = cfg["sites"]

    frames = {}
    for site, meta in sites_cfg.items():
        path = raw_dir / meta["file"]
        if not path.exists():
            raise FileNotFoundError(f"Raw file not found: {path}")
        fmt = meta["format"]
        if fmt == "csv_semicolon":
            frames[site] = load_kaggle(path)
        elif fmt == "uci_long_76":
            frames[site] = load_uci_long(path, site)
        elif fmt == "processed_14col":
            frames[site] = load_uci_processed(path, site)
        else:
            raise ValueError(f"Unknown format: {fmt} for site {site}")

    bp_dropped = None
    for site, df in list(frames.items()):
        df = standardize_columns(df, site)
        df = recode_missing(df, site)
        df = binarize_target(df, site)
        if site == "kaggle":
            df, bp_dropped = drop_kaggle_bp_outliers(df)
        validate_site(df, site)
        frames[site] = df

    uci_sites = [s for s in ["cleveland", "hungary", "switzerland", "va"] if s in frames]
    uci_all = pd.concat([frames[s] for s in uci_sites], ignore_index=True) if uci_sites else pd.DataFrame()

    cfs_ku = cfg.get("cfs", {}).get("kaggle_uci", ["age", "sex", "sys_bp"])
    cfs_uu = cfg.get("cfs", {}).get("uci_cross_site", [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"
    ])

    write_outputs(frames, uci_all, cfg, cfs_ku, cfs_uu)
    write_ingestion_report(frames, cfg, bp_dropped)

    return frames


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    run_ingestion(config_path=_root / "configs" / "data_ingestion.yaml", root=_root)
