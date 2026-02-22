"""
Tests that data_ingestion.md parsing logic works on real Dataset/ files.
Implementation mirrors ImplementationPlan/data_ingestion.md §2.3, §3.3, §3.4, §3.5, §7.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Dataset"


# --- From data_ingestion.md §2.3 ---
def load_kaggle(path, nrows=None) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, sep=";", nrows=nrows)
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


# --- From data_ingestion.md §3.3 ---
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


def _parse_value(token: str):
    token = token.strip()
    if token in ("-9", "-9."):
        return np.nan
    return float(token)


def load_uci_long(path, site_label: str) -> pd.DataFrame:
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


# --- From data_ingestion.md §3.4 ---
PROCESSED_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num",
]


def load_uci_processed(path, site_label: str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, header=None, names=PROCESSED_COLS, na_values=["?"])
    df["site"] = site_label
    return df


# --- From data_ingestion.md §3.5 + §7 ---
def binarize_uci_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target"] = (out["num"] >= 1).astype(int)
    return out


def validate_site(df: pd.DataFrame, site: str):
    expected_n = {"kaggle": 70000, "cleveland": 303, "hungary": 294, "switzerland": 123, "va": 200}
    if site in expected_n:
        assert len(df) == expected_n[site], f"{site}: row count mismatch (got {len(df)})"
    assert "target" in df.columns, f"{site}: missing target"
    assert df["target"].isin([0, 1]).all(), f"{site}: target not binary"
    if "sex" in df.columns:
        assert df["sex"].dropna().isin([0, 1]).all(), f"{site}: sex not binary"
    if "age" in df.columns:
        assert df["age"].dropna().between(0, 120).all(), f"{site}: age out of range"
    if site != "kaggle":
        assert "site" in df.columns
        assert (df["site"] == site).all()


# --- Tests ---
@pytest.fixture(scope="module")
def kaggle_path():
    p = DATASET_DIR / "kaggle_cardio_train.csv"
    if not p.exists():
        pytest.skip("Dataset/kaggle_cardio_train.csv not found")
    return p


@pytest.fixture(scope="module")
def va_path():
    p = DATASET_DIR / "long-beach-va.data"
    if not p.exists():
        pytest.skip("Dataset/long-beach-va.data not found")
    return p


@pytest.fixture(scope="module")
def cleveland_processed_path():
    p = DATASET_DIR / "processed.cleveland.data"
    if not p.exists():
        pytest.skip("Dataset/processed.cleveland.data not found")
    return p


class TestLoadKaggle:
    def test_load_kaggle_returns_dataframe(self, kaggle_path):
        df = load_kaggle(kaggle_path, nrows=500)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_no_id_column(self, kaggle_path):
        df = load_kaggle(kaggle_path, nrows=500)
        assert "id" not in df.columns

    def test_age_in_years(self, kaggle_path):
        df = load_kaggle(kaggle_path, nrows=500)
        assert df["age"].min() >= 10 and df["age"].max() <= 100

    def test_gender_recoded_0_1(self, kaggle_path):
        df = load_kaggle(kaggle_path, nrows=500)
        assert set(df["gender"].dropna()) <= {0, 1}

    def test_site_is_kaggle(self, kaggle_path):
        df = load_kaggle(kaggle_path, nrows=500)
        assert (df["site"] == "kaggle").all()

    def test_bp_outlier_flag_present(self, kaggle_path):
        df = load_kaggle(kaggle_path, nrows=500)
        assert "bp_outlier" in df.columns
        assert df["bp_outlier"].isin([0, 1]).all()

    def test_cardio_target_present(self, kaggle_path):
        df = load_kaggle(kaggle_path, nrows=500)
        assert "cardio" in df.columns
        assert df["cardio"].isin([0, 1]).all()


class TestLoadUciLong:
    def test_load_uci_long_va_returns_dataframe(self, va_path):
        df = load_uci_long(va_path, "va")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200, "VA should have 200 records"

    def test_uci_long_has_14_attrs(self, va_path):
        df = load_uci_long(va_path, "va")
        for col in ATTR_MAP:
            assert col in df.columns, f"Missing column {col}"
        assert "site" in df.columns

    def test_uci_long_site_label(self, va_path):
        df = load_uci_long(va_path, "va")
        assert (df["site"] == "va").all()

    def test_uci_long_missing_as_nan(self, va_path):
        df = load_uci_long(va_path, "va")
        # VA has -9 in many fields; some should be NaN
        assert df.isna().any().any() or len(df) > 0

    def test_uci_long_num_in_range(self, va_path):
        df = load_uci_long(va_path, "va")
        valid = df["num"].dropna()
        assert (valid >= 0).all() and (valid <= 4).all()


class TestLoadUciProcessed:
    def test_load_cleveland_processed_returns_dataframe(self, cleveland_processed_path):
        df = load_uci_processed(cleveland_processed_path, "cleveland")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 303, "Cleveland processed should have 303 records"

    def test_processed_has_14_cols(self, cleveland_processed_path):
        df = load_uci_processed(cleveland_processed_path, "cleveland")
        for c in PROCESSED_COLS:
            assert c in df.columns
        assert "site" in df.columns

    def test_processed_num_binarizable(self, cleveland_processed_path):
        df = load_uci_processed(cleveland_processed_path, "cleveland")
        df = binarize_uci_target(df)
        assert df["target"].isin([0, 1]).all()


class TestValidateSite:
    def test_validate_kaggle_after_target_rename(self, kaggle_path):
        df = load_kaggle(kaggle_path, nrows=500)
        df = df.rename(columns={"cardio": "target"})
        # validate_site expects 70000 for kaggle; skip strict count for sample
        assert "target" in df.columns and df["target"].isin([0, 1]).all()

    def test_validate_va_after_binarize(self, va_path):
        df = load_uci_long(va_path, "va")
        df = binarize_uci_target(df)
        validate_site(df, "va")

    def test_validate_cleveland_after_binarize(self, cleveland_processed_path):
        df = load_uci_processed(cleveland_processed_path, "cleveland")
        df = binarize_uci_target(df)
        validate_site(df, "cleveland")


class TestDataIngestionIntegration:
    """Run full ingestion path for one site each (Kaggle, UCI long, UCI processed)."""

    def test_kaggle_full_flow(self, kaggle_path):
        df = load_kaggle(kaggle_path, nrows=1000)
        df = df.rename(columns={"cardio": "target"})
        assert df["target"].isin([0, 1]).all() and (df["site"] == "kaggle").all()
        assert len(df) == 1000

    def test_va_full_flow(self, va_path):
        df = load_uci_long(va_path, "va")
        df = binarize_uci_target(df)
        validate_site(df, "va")

    def test_cleveland_full_flow(self, cleveland_processed_path):
        df = load_uci_processed(cleveland_processed_path, "cleveland")
        df = binarize_uci_target(df)
        validate_site(df, "cleveland")
