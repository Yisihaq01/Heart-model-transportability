# Paper-Ready Evidence Bundle

Camera-ready figures and tables for the heart model transportability paper.

## Artifacts and RQ Mapping

| Artifact | Format | RQ | Description |
|----------|--------|-----|--------------|
| `fig_internal_vs_external` | PNG, SVG | RQ1, RQ2 | Internal vs external validation ROC-AUC comparison by test site |
| `table_internal_vs_external` | CSV, MD | RQ1, RQ2 | Per-model internal vs external performance with AUC drop |
| `fig_cfs_penalty` | PNG, SVG | RQ5 | Common Feature Set (CFS) penalty by test site and model |
| `table_cfs_penalty` | CSV, MD | RQ5 | Full-feature AUC vs CFS AUC and penalty delta |
| `fig_calibration_before_after` | PNG, SVG | RQ4 | ECE before vs after recalibration |
| `table_calibration_summary` | CSV, MD | RQ4 | Calibration metrics before/after (ECE, Brier) by variant |
| `table_calibration_analysis_master` | CSV | RQ4 | **Formal calibration failure analysis**: before/after deltas (Brier, ECE, MCE), win/loss flags, failure categories, outcome labels across all 5 methods |
| `calibration_analysis/` | dir | RQ4 | Full analysis: summary by method, failure modes, by direction, by model family, interpretation note |
| `fig_shift_top` | PNG, SVG | RQ3 | Top shift pairs by mean PSI |
| `table_shift_top` | CSV, MD | RQ3 | Top shift diagnostics (mean PSI, C2ST AUC, prevalence diff) |
| `table_size_matched_summary` | CSV, MD | RQ2 | Size-matched sensitivity: mean ROC-AUC and ECE across K subsamples |

## Research Questions

- **RQ1**: Internal validation baseline (discrimination, calibration)
- **RQ2**: External validation degradation; size-matched sensitivity
- **RQ3**: Dataset shift diagnostics (PSI, C2ST, prevalence)
- **RQ4**: Calibration/recalibration effectiveness
- **RQ5**: CFS performance penalty

## Source Artifacts

Generated from:
- `outputs/internal/` — internal validation results
- `outputs/external_uci/` — UCI multi-site external validation
- `outputs/external_kaggle_uci/` — Kaggle↔UCI CFS experiments
- `outputs/calibration/updating/` — recalibration summaries
- `outputs/size_matched/` — size-matched sensitivity
- `outputs/shift/` — shift diagnostics
