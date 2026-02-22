# Tests for README, Data Ingestion, and Pipeline Plan

Run from project root:

```bash
python -m pytest Test/ -v
```

**Dependencies:** `pytest`, `pandas`, `numpy`, `scikit-learn` (install with `pip install pytest pandas numpy scikit-learn`).

| File | What it tests |
|------|----------------|
| `test_readme.py` | README project structure (Dataset/, data/, ImplementationPlan/, docs); dataset file existence; Cleveland/Kaggle/VA consistency |
| `test_data_ingestion.py` | Ingestion logic from `ImplementationPlan/data_ingestion.md`: `load_kaggle`, `load_uci_long`, `load_uci_processed`, binarize target, `validate_site` on real Dataset/ files |
| `test_pipeline_plan.py` | Pipeline logic from `ImplementationPlan/pipeline_plan.md`: `resolve_features`, `compute_metrics`, `internal_split`, `bootstrap_metric`, `prevalence_shift`, `effective_cfs`, `bin_cholesterol_uci`, `cfs_penalty`, `missingness_shift`; integration flow with synthetic data |

Kaggle tests use a small sample (`nrows=500`/`1000`) so the suite stays fast; full 70k row validation is not run in these tests.
