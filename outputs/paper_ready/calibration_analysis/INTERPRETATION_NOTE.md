# Calibration Failure Analysis — Interpretation Note

## Overview
This analysis compares before vs after calibration across five methods:
Platt scaling, isotonic regression, temperature scaling, intercept-only,
and intercept+slope logistic recalibration.

## Key Findings

- **Best improvement rate**: isotonic (84.0% improved)
- **Highest degradation rate**: intercept_only (20.8% degraded)

## Documented Failure Modes

- `neutral` (intercept_only): 48 occurrences
- `neutral` (temperature): 18 occurrences
- `brier_degraded|ece_degraded|ece_severe|mce_degraded|mce_severe` (intercept_only): 12 occurrences
- `brier_degraded|ece_degraded|ece_severe|mce_degraded|mce_severe` (intercept_slope): 10 occurrences
- `mce_degraded` (temperature): 10 occurrences
- `brier_degraded|ece_degraded|ece_severe|mce_degraded|mce_severe` (isotonic): 9 occurrences
- `mce_degraded` (intercept_only): 8 occurrences
- `brier_degraded|ece_degraded|ece_severe|mce_degraded|mce_severe` (platt): 7 occurrences
- `neutral` (platt): 4 occurrences
- `brier_degraded|ece_degraded|ece_severe` (platt): 4 occurrences
- `brier_degraded|ece_degraded|ece_severe` (intercept_only): 4 occurrences
- `ece_degraded` (temperature): 3 occurrences
- `brier_degraded|ece_degraded|ece_severe` (isotonic): 3 occurrences
- `brier_degraded|ece_degraded` (platt): 3 occurrences
- `brier_degraded|ece_degraded|ece_severe|mce_degraded` (intercept_only): 3 occurrences

## Failure Mode Definitions
- `ece_degraded`: ECE increased after recalibration
- `brier_degraded`: Brier score increased
- `mce_degraded`: MCE increased
- `ece_severe`: ECE delta > 0.05
- `mce_severe`: MCE delta > 0.1

## Outcome Labels
- **improved**: ≥2 metrics improved, <2 degraded
- **neutral**: Mixed or no significant change
- **degraded**: ≥2 metrics degraded, or ≥1 degraded with no improvement