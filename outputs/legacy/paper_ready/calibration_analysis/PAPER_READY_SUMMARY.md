# Calibration Failure Analysis — Paper-Ready Summary

## Master Table
Total experiments: 720
Methods: platt, isotonic, temperature, intercept_only, intercept_slope

## Outcome Distribution by Method

| Method | N | Improved | Degraded | Neutral | % Improved | % Degraded |
|--------|---|----------|----------|---------|------------|------------|
| intercept_only | 144 | 62 | 30 | 52 | 43.1% | 20.8% |
| intercept_slope | 144 | 119 | 20 | 5 | 82.6% | 13.9% |
| isotonic | 144 | 121 | 19 | 4 | 84.0% | 13.2% |
| platt | 144 | 119 | 17 | 8 | 82.6% | 11.8% |
| temperature | 144 | 104 | 11 | 29 | 72.2% | 7.6% |

## Failure Modes (Top)
- neutral (intercept_only): 48
- neutral (temperature): 18
- brier_degraded|ece_degraded|ece_severe|mce_degraded|mce_severe (intercept_only): 12
- brier_degraded|ece_degraded|ece_severe|mce_degraded|mce_severe (intercept_slope): 10
- mce_degraded (temperature): 10
- brier_degraded|ece_degraded|ece_severe|mce_degraded|mce_severe (isotonic): 9
- mce_degraded (intercept_only): 8
- brier_degraded|ece_degraded|ece_severe|mce_degraded|mce_severe (platt): 7
- neutral (platt): 4
- brier_degraded|ece_degraded|ece_severe (platt): 4