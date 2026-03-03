"""
Tests for src/evaluation.py: build_master_table and run_evaluation smoke test.
Uses temp directories with minimal results.json fixtures — no reliance on full pipeline.
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import build_master_table, run_evaluation


def _make_results(
    exp_type: str,
    *,
    site: str | None = None,
    train_sites: list[str] | None = None,
    test_site: str | None = None,
    model: str = "lr",
    roc_auc: float = 0.85,
) -> dict:
    """Minimal results.json conforming to build_master_table's expected schema."""
    r: dict = {
        "experiment_type": exp_type,
        "model": model,
        "n_train": 100,
        "features_used": ["age", "sex", "cp"],
        "metrics": {
            "roc_auc": roc_auc,
            "pr_auc": 0.80,
            "brier_score": 0.15,
            "ece": 0.05,
            "n_test": 50,
        },
    }
    if exp_type in ("internal", "internal_cfs"):
        r["site"] = site or "cleveland"
    else:
        r["train_sites"] = train_sites or ["cleveland"]
        r["test_site"] = test_site or "hungary"
    return r


@pytest.fixture
def minimal_outputs(tmp_path):
    """Outputs tree: 2 internal + 1 external results.json."""
    for site, auc in [("cleveland", 0.90), ("hungary", 0.85)]:
        d = tmp_path / "internal" / site / "lr"
        d.mkdir(parents=True)
        (d / "results.json").write_text(
            json.dumps(_make_results("internal", site=site, roc_auc=auc))
        )

    ext = tmp_path / "external_uci" / "cleveland__to__hungary" / "lr"
    ext.mkdir(parents=True)
    (ext / "results.json").write_text(
        json.dumps(
            _make_results(
                "external_uci",
                train_sites=["cleveland"],
                test_site="hungary",
                model="lr",
                roc_auc=0.75,
            )
        )
    )
    return tmp_path


# ---------- build_master_table ----------


class TestBuildMasterTable:
    def test_returns_nonempty_dataframe(self, minimal_outputs):
        df = build_master_table(minimal_outputs)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_expected_columns_present(self, minimal_outputs):
        df = build_master_table(minimal_outputs)
        required = {
            "experiment_type", "model", "train_sites", "test_site",
            "roc_auc", "n_train", "n_test",
        }
        assert required.issubset(set(df.columns))

    def test_row_count_matches_fixtures(self, minimal_outputs):
        df = build_master_table(minimal_outputs)
        assert len(df) == 3

    def test_experiment_types(self, minimal_outputs):
        df = build_master_table(minimal_outputs)
        assert set(df["experiment_type"]) == {"internal", "external_uci"}

    def test_metric_values_propagated(self, minimal_outputs):
        df = build_master_table(minimal_outputs)
        cleveland = df[
            (df["experiment_type"] == "internal") & (df["test_site"] == "cleveland")
        ]
        assert cleveland.iloc[0]["roc_auc"] == pytest.approx(0.90)

    def test_empty_directory_returns_empty_df(self, tmp_path):
        df = build_master_table(tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_corrupted_json_skipped(self, minimal_outputs):
        bad = minimal_outputs / "internal" / "broken" / "lr"
        bad.mkdir(parents=True)
        (bad / "results.json").write_text("{bad json!!")
        df = build_master_table(minimal_outputs)
        assert len(df) == 3

    def test_bootstrap_cis_flattened(self, tmp_path):
        r = _make_results("internal", site="va", roc_auc=0.70)
        r["bootstrap_cis"] = {"roc_auc": {"ci_lower": 0.60, "ci_upper": 0.80}}
        d = tmp_path / "internal" / "va" / "lr"
        d.mkdir(parents=True)
        (d / "results.json").write_text(json.dumps(r))
        df = build_master_table(tmp_path)
        assert "roc_auc_ci_lower" in df.columns
        assert df.iloc[0]["roc_auc_ci_lower"] == pytest.approx(0.60)


# ---------- run_evaluation smoke test ----------


class TestRunEvaluation:
    @patch("src.evaluation.generate_all_figures")
    def test_produces_master_csv_and_report(self, mock_fig, minimal_outputs, tmp_path):
        reports = tmp_path / "rpt"
        df = run_evaluation(
            outputs_dir=str(minimal_outputs),
            reports_dir=str(reports),
        )
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert (reports / "tables" / "master_results.csv").exists()
        assert (reports / "evaluation_report.md").exists()

    @patch("src.evaluation.generate_all_figures")
    def test_stat_tests_json_created(self, mock_fig, minimal_outputs, tmp_path):
        reports = tmp_path / "rpt"
        run_evaluation(
            outputs_dir=str(minimal_outputs),
            reports_dir=str(reports),
        )
        stat_path = reports / "tables" / "statistical_tests.json"
        assert stat_path.exists()
        data = json.loads(stat_path.read_text())
        assert isinstance(data, dict)

    @patch("src.evaluation.generate_all_figures")
    def test_html_report_created(self, mock_fig, minimal_outputs, tmp_path):
        reports = tmp_path / "rpt"
        run_evaluation(
            outputs_dir=str(minimal_outputs),
            reports_dir=str(reports),
        )
        assert (reports / "evaluation_report.html").exists()
