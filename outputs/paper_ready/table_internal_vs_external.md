| test_site | model | internal_roc_auc | external_roc_auc | auc_drop | best_external_train |
| --- | --- | --- | --- | --- | --- |
| cleveland | lgbm | 0.9470 | 0.8554 | 0.0916 | hungary+switzerland+va
| cleveland | lr | 0.9632 | 0.8705 | 0.0927 | hungary
| cleveland | rf | 0.9697 | 0.8624 | 0.1073 | hungary
| cleveland | xgb | 0.9491 | 0.8635 | 0.0856 | hungary+switzerland+va
| hungary | lgbm | 0.9016 | 0.8916 | 0.0101 | cleveland
| hungary | lr | 0.8722 | 0.8977 | -0.0256 | va
| hungary | rf | 0.8835 | 0.8964 | -0.0130 | cleveland
| hungary | xgb | 0.9010 | 0.8825 | 0.0185 | cleveland
| kaggle | lgbm | 0.7959 | — | — | 
| kaggle | lr | 0.7852 | — | — | 
| kaggle | rf | 0.7960 | — | — | 
| kaggle | xgb | 0.7966 | — | — | 
| switzerland | lgbm | 0.6961 | 0.7511 | -0.0550 | cleveland+hungary+va
| switzerland | lr | 0.7243 | 0.7620 | -0.0376 | hungary
| switzerland | rf | 0.7217 | 0.7793 | -0.0576 | cleveland
| switzerland | xgb | 0.7596 | 0.7707 | -0.0111 | cleveland
| va | lgbm | 0.6467 | 0.7454 | -0.0987 | cleveland
| va | lr | 0.8233 | 0.7522 | 0.0711 | cleveland+hungary+switzerland
| va | rf | 0.6667 | 0.7487 | -0.0820 | cleveland+hungary+switzerland
| va | xgb | 0.6767 | 0.7294 | -0.0528 | cleveland