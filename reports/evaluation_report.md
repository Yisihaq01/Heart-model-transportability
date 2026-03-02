# Evaluation Report — Heart Disease Model Transportability

## 1. Executive Summary

### Rq1 Internal Baselines
- {"best_model_per_site": [{"test_site": "cleveland", "model": "rf", "roc_auc": 0.9696969696969697}, {"test_site": "cleveland+hungary+switzerland+va", "model": "rf", "roc_auc": 0.7136537541846006}, {"test_site": "hungary", "model": "xgb", "roc_auc": 0.9035087719298246}, {"test_site": "kaggle", "model": "lgbm", "roc_auc": 0.7971238419064719}, {"test_site": "switzerland", "model": "xgb", "roc_auc": 0.758695652173913}, {"test_site": "va", "model": "lr", "roc_auc": 0.8233333333333333}], "n_experiments": 44}

### Rq2 External Degradation
- {"worst_auc_drop": {"experiment_type": "external_kaggle_uci", "variant": "cfs", "train_sites": "kaggle", "test_site": "cleveland", "model": "rf", "n_train": 68726.0, "n_test": 303, "n_features_used": 3, "features_used": ["age", "sex", "sys_bp"], "best_params": {"max_depth": 5, "min_samples_leaf": 1}, "results_path": "outputs\\runs\\20260301T225358_93bf3a3395d35eb1\\external_kaggle_uci\\cfs\\kaggle__to__cleveland\\rf\\results.json", "roc_auc": 0.5758247060887876, "pr_auc": 0.5172796123437253, "brier_score": 0.28037867892819834, "accuracy": 0.5214521452145214, "f1": 0.5510835913312694, "precision": 0.483695652173913, "recall": 0.6402877697841727, "specificity": 0.42073170731707316, "prevalence": 0.45874587458745875, "ece": 0.18856575168869222, "mce": 0.32883212528203126, "roc_auc_ci_lower": 0.5072983720085065, "roc_auc_ci_upper": 0.6382705169559137, "pr_auc_ci_lower": 0.4433160033062054, "pr_auc_ci_upper": 0.6058006029742298, "brier_score_ci_lower": 0.25482190842318286, "brier_score_ci_upper": 0.30709983883536185, "accuracy_ci_lower": 0.46204620462046203, "accuracy_ci_upper": 0.5775577557755776, "ece_ci_lower": 0.1451610493088937, "ece_ci_upper": 0.25148474616589567, "cfs_full_auc": NaN, "cfs_cfs_auc": NaN, "cfs_auc_drop": NaN, "cfs_relative_drop_pct": NaN, "internal_auc": 0.9696969696969697, "auc_delta": -0.39387226360818217}, "mean_auc_drop": -0.09703801493417628}

### Rq3 Shift Performance
- {"spearman_rho": NaN, "p_value": NaN}

### Rq4 Recalibration
- {"mean_ece_before": 0.2497037977421798, "mean_ece_after": 0.14090876986170303, "mean_improvement": 0.10879502788047676, "wilcoxon_stat": 18532.0, "p_value": 5.806936945336855e-19}

### Rq5 Cfs Penalty
- {"mean_auc_drop": 0.16800983663825603, "max_auc_drop": 0.3682608695652174}

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

## 4. Results — RQ2: External Validation

Mean AUC drop: -0.097

## 5. Results — RQ3: Dataset Shift

Spearman ρ (mean PSI vs AUC drop): {'spearman_rho': nan, 'p_value': nan}

## 6. Results — RQ4: Recalibration

Wilcoxon (ECE before vs after): {'mean_ece_before': 0.2497037977421798, 'mean_ece_after': 0.14090876986170303, 'mean_improvement': 0.10879502788047676, 'wilcoxon_stat': 18532.0, 'p_value': 5.806936945336855e-19}

## 7. Results — RQ5: CFS Penalty

CFS penalty: {'mean_auc_drop': 0.16800983663825603, 'max_auc_drop': 0.3682608695652174}

## 8. Discussion

- Transportability patterns depend on train–test site pair.
- Recalibration can improve ECE when miscalibration is correctable.
- CFS restriction incurs AUC penalty; external drop may exceed it.

## 9. Appendices

- Full metric tables: reports/tables/master_results.csv
- Statistical tests: reports/tables/statistical_tests.json
- RQ summaries: reports/rq_summaries.json
