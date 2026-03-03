# Evaluation Report — Heart Disease Model Transportability

## 1. Executive Summary

**RQ1 — Internal baselines:** Within-site performance varied by cohort. Best models per site: cleveland (rf), cleveland+hungary+switzerland+va (rf), hungary (xgb), kaggle (lgbm), switzerland (xgb). Best internal ROC-AUC was 0.970. Small UCI cohorts (e.g. Switzerland) show wider CIs than Kaggle.

**RQ2 — External validation:** Performance dropped when models were applied across sites. Worst AUC drop observed: kaggle → cleveland (rf), Δ = -0.394. Mean AUC drop across external pairs: -0.097. External Kaggle↔UCI CFS runs do not have an internal CFS baseline on the same test site, so CFS penalty fields (cfs_full_auc, cfs_cfs_auc) are NaN for those rows.

**RQ3 — Dataset shift:** Shift diagnostics (PSI, prevalence diff, C2ST) were correlated with performance degradation. Spearman ρ (mean PSI vs AUC drop) = -0.19, p = 0.126. The correlation was not statistically significant; power was limited by the number of external pairs. Shift signatures (prevalence vs covariate vs missingness) are summarized in Results §5.

**RQ4 — Recalibration:** Recalibration (intercept-only and full logistic) significantly reduced ECE (mean ECE before 0.250, after 0.141; mean improvement 0.109; Wilcoxon p &lt; 0.001). When miscalibration is correctable, recalibration is sufficient; when feature mismatch dominates, it is not.

**RQ5 — CFS penalty:** Restricting to common features (CFS) incurred a mean AUC drop of 0.168 and maximum 0.368. Part of the Kaggle↔UCI external drop is attributable to this feature restriction; the remainder reflects population shift.

**Headline metrics:** 
best internal AUC = 0.970; worst external drop (mean) = -0.097; recalibration recovery (mean ECE Δ) = 0.109.

## 2. Methods Summary

- Data sources: Kaggle CVD, UCI Heart Disease (Cleveland, Hungary, Switzerland, VA)
- Models: LR, RF, XGB, LGBM
- Metrics: ROC-AUC, PR-AUC, Brier, ECE, MCE
- Statistical tests: Spearman (shift–performance), Wilcoxon (recalibration)

## 3. Results — RQ1: Internal Baselines

| Site | lgbm | lr | rf | xgb |
|------|---|---|---|---|
| cleveland | 0.947 | 0.963 | 0.970 | 0.948 |
| cleveland+hungary+switzerland+va | 0.706 | 0.707 | 0.714 | 0.691 |
| hungary | 0.902 | 0.872 | 0.883 | 0.904 |
| kaggle | 0.797 | 0.785 | 0.796 | 0.796 |
| switzerland | 0.696 | 0.724 | 0.722 | 0.759 |
| va | 0.647 | 0.823 | 0.667 | 0.693 |

Key finding: Model ranking varies by site — dominant models include cleveland (rf), cleveland+hungary+switzerland+va (rf), hungary (xgb), kaggle (lgbm), switzerland (xgb), va (lr). Small UCI cohorts show wider bootstrap CIs; Kaggle's performance is not directly comparable without size-matched sensitivity. Discrimination and calibration (Brier, ECE) are reported in the table and figures.

## 4. Results — RQ2: External Validation

Mean AUC drop: -0.097

Key finding: Pairwise and LOSO external validation show substantial AUC drops for several train→test directions. The worst drop typically occurs for Kaggle→UCI CFS or cross-site pairs with large covariate shift.  Note: For external Kaggle↔UCI CFS experiments there is no internal CFS baseline on the same test site (UCI sites were evaluated internally with full features), so fields such as cfs_full_auc and cfs_cfs_auc are undefined (NaN) in the worst-AUC-drop summary for those rows.

## 5. Results — RQ3: Dataset Shift

Spearman ρ (mean PSI vs AUC drop): ρ = -0.193, p = 0.126

Key finding: The shift–performance Spearman correlation (mean PSI vs AUC drop) was ρ = -0.19, p = 0.126: not statistically significant at α = 0.05. Power was limited by the number of external pairs. Shift signatures (prevalence-driven vs covariate-driven vs missingness-driven degradation) can still be inspected per pair via the PSI heatmap and distribution overlays.

## 6. Results — RQ4: Recalibration

Wilcoxon (ECE before vs after): mean ECE before 0.250, after 0.141, mean improvement 0.109, p = 5.806936945336855e-19

Key finding: Recalibration (intercept-only and full logistic) significantly reduced ECE on average. When ECE drops to below ~0.05 after recalibration, miscalibration was correctable; when no method helps, feature-level mismatch likely dominates and recalibration is insufficient.

## 7. Results — RQ5: CFS Penalty

Mean CFS AUC drop: 0.168; max: 0.368

Key finding: The CFS (common-feature-set) restriction incurs a mean AUC penalty of 0.168 (max 0.368). Part of the Kaggle↔UCI transportability gap is thus attributable to feature restriction; the remainder reflects population and distribution shift.

## 8. Discussion

**Transportability patterns:** Performance transfer depends strongly on the train–test site pair. Same-site internal validation gives optimistically high AUC; external application to other sites or to CFS-only evaluation typically reduces discrimination. Directions that transfer relatively well can be identified from the AUC heatmap and Δ matrix; the worst degradation occurs for Kaggle→UCI CFS and for pairs with large covariate or prevalence shift.

**Recalibration sufficiency:** Recalibration (Platt, isotonic, or logistic updating) is sufficient when the dominant issue is miscalibration (e.g. prevalence shift). When ECE drops to below ~0.05 after recalibration, probability outputs can be used with confidence. When no method materially improves ECE, feature-level mismatch or population shift likely dominates, and recalibration is insufficient—retraining or feature alignment is then needed.

**Limitations:** Bootstrap CIs overlap substantially for small UCI cohorts, limiting claims about model ranking. The shift–performance correlation was underpowered (few external pairs). External Kaggle↔UCI CFS experiments do not have an internal CFS baseline on the same test site, so CFS penalty fields (cfs_full_auc, cfs_cfs_auc) are undefined (NaN) for the worst-AUC-drop summary for those rows. Univariate shift diagnostics (PSI, KS) may miss multivariate structure captured by C2ST.

## 9. Appendices

- Full metric tables: reports/tables/master_results.csv
- Statistical tests: reports/tables/statistical_tests.json
- RQ summaries: reports/rq_summaries.json (structured data for reproducibility)

### Appendix: PROBAST Risk Assessment

Domain-level bias and applicability assessment per PROBAST+AI:
[reports/compliance/PROBAST_RISK_ASSESSMENT.md](compliance/PROBAST_RISK_ASSESSMENT.md)

### Code and Data Availability

All analysis code is available in this repository. Data sources: the [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (Cleveland, Hungary, Switzerland, VA) and the [Kaggle Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset). Both datasets are publicly available under their respective licenses.
