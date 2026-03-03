# Outputs Layout

## Canonical Structure (Reproducible Runs)

**Canonical outputs** live under `outputs/runs/{run_id}/`. Each run is self-contained and traceable.

```
outputs/
├── runs/
│   └── {run_id}/                    # e.g. 20250301T143022_a1b2c3d4e5f6
│       ├── manifest.json            # Run manifest: config_hash, seed, code_version, artifact index
│       ├── internal/                # Stage 1.3
│       ├── internal_cfs/            # Stage 1.5 (CFS baselines)
│       ├── external_uci/            # Stage 1.4
│       ├── external_kaggle_uci/     # Stage 1.5
│       ├── calibration/             # Stages 2.1–2.3
│       ├── shift/                   # Stage 1.6
│       └── size_matched/            # Size-matched sensitivity
└── legacy/                          # Pre-canonical flat outputs (migrated)
```

### Run ID

`{run_id}` = `{timestamp}_{config_hash}` (e.g. `20250301T143022_a1b2c3d4e5f6`).

- **Timestamp**: UTC `YYYYMMDDTHHMMSS` — when the run started.
- **Config hash**: First 16 chars of SHA256 of pipeline config — ties artifacts to exact config.

### Manifest

`manifest.json` at each run root captures:

| Field | Description |
|-------|--------------|
| `run_id` | Canonical run identifier |
| `config_hash` | Hash of `configs/pipeline.yaml` |
| `seed` | Random seed used |
| `code_version` | Git commit (12 chars) if available |
| `timestamp` | When manifest was written |
| `artifact_count` | Number of indexed artifacts |
| `artifacts` | Index of generated artifacts (path, type, metadata) |

Every artifact can be traced back to one run config via the manifest.

### Naming Conventions

Consistent across internal, external, calibration, shift, size_matched:

| Artifact Type | Path Pattern |
|---------------|--------------|
| Internal | `internal/{site}/{model_key}/` |
| Internal CFS | `internal_cfs/{variant}/{site}/{model_key}/` |
| External UCI | `external_uci/{train_sites}__to__{test_site}/{model_key}/` |
| External Kaggle↔UCI | `external_kaggle_uci/{variant}/{train}__to__{test}/{model_key}/` |
| Calibration | `calibration/{before|recalibration|updating}/...` |
| Shift | `shift/{pair_key}/` or `shift/shift_table.parquet` |
| Size-matched | `size_matched/{pair_key}/{model_key}.json` |

---

## Legacy Structure

**Legacy outputs** are the previous flat layout: `outputs/internal/`, `outputs/external_uci/`, etc. directly under `outputs/`.

- Existing outputs can be migrated to `outputs/legacy/` for archival.
- Scripts `build_paper_ready.py` and `build_calibration_analysis.py` support `--run-id` to read from canonical runs, or default to legacy paths for backward compatibility.

---

## Running the Pipeline

**Canonical run** (recommended):

```bash
python scripts/run_pipeline.py
```

Writes to `outputs/runs/{run_id}/` and creates `manifest.json`.

**Legacy run** (flat outputs):

```bash
python scripts/run_pipeline.py --legacy
```

Writes to `outputs/internal/`, `outputs/external_uci/`, etc.

**Paper-ready bundle** from a canonical run:

```bash
python scripts/build_paper_ready.py --run-id 20250301T143022_a1b2c3d4e5f6
```

---

## Migration

To archive existing flat outputs as legacy:

```bash
python scripts/migrate_legacy.py
```

Moves `outputs/internal/`, `outputs/external_uci/`, etc. into `outputs/legacy/` (creates timestamped subdir if legacy already exists).
