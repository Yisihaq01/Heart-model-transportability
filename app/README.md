# Transportability Dashboard

Phase 2 deliverable: interactive Streamlit app for exploring pipeline results.

## Run

```bash
# From project root
streamlit run app/dashboard.py
```

Or:

```bash
python -m streamlit run app/dashboard.py
```

## Features

- **Dataset overview** — Site-level stats from `data/ingestion_report.json` (N, features, prevalence, missingness)
- **Internal 80/20** — ROC-AUC pivot per site × model
- **External validation matrix** — Train→test AUC heatmap; AUC drop (external − internal)
- **Calibration** — Brier, ECE per experiment; recalibration (Platt) summary
- **Shift & overlap** — Shift diagnostics table; shift vs performance; effective CFS feature count per pair
- **Export report** — Model-card style Markdown summary (TRIPOD+AI aligned); downloadable

## Data source

Reads from `outputs/runs/{run_id}/`. Use the sidebar to select a run (default: latest). Run the pipeline first:

```bash
python scripts/run_pipeline.py
```

## Dependencies

- `streamlit` (in `requirements.txt`)
- `pandas`, `pyarrow` (for parquet)
