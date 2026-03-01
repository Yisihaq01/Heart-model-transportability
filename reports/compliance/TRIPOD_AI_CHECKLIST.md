# TRIPOD+AI Checklist — Heart Model Transportability Project

**Reporting guideline alignment for clinical prediction model studies using ML methods.**  
Reference: [TRIPOD+AI Statement (BMJ 2024)](https://www.bmj.com/content/385/bmj-2023-078378)

---

## Checklist Summary

| Section | Item | Status | Project Location |
|---------|------|--------|------------------|
| **Title & Abstract** | 1–3 | ✓ | README, eval_plan §6 |
| **Introduction** | 4–5 | ✓ | README, ImplementationPlan |
| **Methods** | 6–17 | ✓ | See mappings below |
| **Results** | 18–21 | ✓ | outputs/, reports/ |
| **Discussion** | 22–24 | ✓ | eval_plan §6.1 §8 |
| **Other** | 25–27 | ✓ | configs/, src/ |

---

## Detailed Item Mapping

### Title and Abstract

| # | Item | Requirement | Project Mapping |
|---|------|-------------|-----------------|
| **1** | Title | Identify study as prediction model development and/or validation | `README.md` — "How Transportable Are Tabular Heart-Disease Classifiers?" |
| **2** | Abstract | Structured summary: objectives, data, methods, results, conclusions | `ImplementationPlan/eval_plan.md` §6.1 — Executive Summary |
| **3** | Abstract | Report key performance metrics (discrimination, calibration) | `outputs/paper_ready/` — tables, `calibration_analysis/` |

### Introduction

| # | Item | Requirement | Project Mapping |
|---|------|-------------|-----------------|
| **4** | Background | Rationale for prediction model development/validation | `README.md` §Research Motivation, §The Gap |
| **5** | Objectives | Clear objectives (development, validation, updating) | `README.md` §Research Questions (RQ1–RQ5) |

### Methods — Data

| # | Item | Requirement | Project Mapping |
|---|------|-------------|-----------------|
| **6** | Source of data | Describe data sources | `README.md` §Datasets; `ImplementationPlan/data_ingestion.md` §1–§2 |
| **7** | Participants | Eligibility, setting, recruitment | `src/ingest.py` — `validate_site()`, `EXPECTED_RECORDS`; `data/ingestion_report.json` |
| **8** | Outcome | Definition, measurement, timing | `README.md` §Datasets (target col); `src/ingest.py` — `binarize_target()` (num 0 vs 1–4 for UCI) |
| **9** | Predictors | Definition, measurement, availability | `src/preprocessing.py` — `KAGGLE_FULL_FEATURES`, `UCI_FULL_FEATURES`, `resolve_features()` |
| **10** | Sample size | How determined; actual N per site | `data/ingestion_report.json` — `n_records` per site; `ImplementationPlan/data_ingestion.md` §1 |
| **11** | Missing data | Handling and reporting | `src/preprocessing.py` — `effective_cfs()`, `MISSINGNESS_THRESHOLD`; `src/shift.py` — `missingness_shift()`; `ingestion_report.json` — `missing_rates` |

### Methods — Model Development

| # | Item | Requirement | Project Mapping |
|---|------|-------------|-----------------|
| **12** | Model development | Model type, hyperparameter tuning | `src/models.py` — `MODEL_REGISTRY`, `tune_model()`; `ImplementationPlan/pipeline_plan.md` §1.2 |
| **13** | Model development | Feature selection | `src/preprocessing.py` — `resolve_features()`, `effective_cfs()`; `ImplementationPlan/pipeline_plan.md` §1.1.1 |
| **14** | Model development | Model performance | Internal validation | `src/validation.py` — `run_internal_validation()`; `outputs/internal/` — `results.json` |

### Methods — Validation

| # | Item | Requirement | Project Mapping |
|---|------|-------------|-----------------|
| **15** | Validation | Internal validation strategy | `src/validation.py` — `internal_split()` (80/20, 5×5 CV for Switzerland); `ImplementationPlan/pipeline_plan.md` §1.3 |
| **16** | Validation | External validation | `src/validation.py` — `run_external_uci_matrix()`, `run_kaggle_uci_tests()`; `outputs/external_uci/`, `outputs/external_kaggle_uci/` |
| **17** | Updating | Recalibration/updating methods | `src/calibration.py` — Platt, isotonic, temperature; `run_lightweight_updating()`; `outputs/calibration/` |

### Results

| # | Item | Requirement | Project Mapping |
|---|------|-------------|-----------------|
| **18** | Participants | Flow diagram | `data/ingestion_report.json` — `n_records`, `bp_outliers_dropped`; `ImplementationPlan/data_ingestion.md` |
| **19** | Model performance | Discrimination metrics | `src/metrics.py` — `compute_metrics()` (ROC-AUC, PR-AUC); `outputs/*/results.json` — `metrics` |
| **20** | Model performance | Calibration | `src/metrics.py` — `compute_ece_mce()`; `src/calibration.py` — `compute_calibration_metrics()`; `outputs/calibration/` |
| **21** | Model performance | Uncertainty | `src/metrics.py` — `bootstrap_metrics()`; `outputs/*/results.json` — `bootstrap_cis` |

### Discussion

| # | Item | Requirement | Project Mapping |
|---|------|-------------|-----------------|
| **22** | Interpretation | Clinical implications | `README.md` §Research Motivation; `ImplementationPlan/eval_plan.md` §6.1 §8 |
| **23** | Limitations | Study limitations | `README.md` §Known Limitations |
| **24** | Implications | Implications for practice/research | `ImplementationPlan/eval_plan.md` §6.1 §8 — Discussion |

### Other

| # | Item | Requirement | Project Mapping |
|---|------|-------------|-----------------|
| **25** | Protocol | Protocol/registration | `ImplementationPlan/` — `pipeline_plan.md`, `eval_plan.md`, `data_ingestion.md` |
| **26** | Funding | Funding | N/A (research project) |
| **27** | Availability | Code, data, model availability | `README.md` §Project Structure; `Dataset/`, `outputs/`, `src/` |

---

## AI/ML-Specific Items (TRIPOD+AI Extension)

| Item | Requirement | Project Mapping |
|------|-------------|-----------------|
| **AI/ML model type** | Specify model family (e.g., logistic regression, tree-based) | `src/models.py` — LR, RF, XGBoost, LightGBM |
| **Hyperparameter tuning** | Protocol | `src/models.py` — `tune_model()` (GridSearchCV/RandomizedSearchCV, roc_auc) |
| **Preprocessing** | Fit on train only | `src/preprocessing.py` — `fit_transform_train()`, `transform_test()` |
| **Data leakage** | Prevention | Train/test split before preprocessing; `fit_transform` on train only |
| **External validation** | Multi-site | `outputs/external_uci/`, `outputs/external_kaggle_uci/` |
| **Calibration** | Pre/post recalibration | `src/calibration.py` — Stages 2.1–2.3 |
| **Shift diagnostics** | Dataset shift | `src/shift.py` — PSI, KS, C2ST, prevalence shift |

---

## File-Section Cross-Reference

| File | TRIPOD+AI items covered |
|------|--------------------------|
| `README.md` | 1, 2, 4, 5, 6, 8, 22, 23, 27 |
| `ImplementationPlan/data_ingestion.md` | 6, 7, 10, 18 |
| `ImplementationPlan/pipeline_plan.md` | 12, 13, 14, 15, 16, 17 |
| `ImplementationPlan/eval_plan.md` | 2, 18–24 |
| `src/ingest.py` | 6, 7, 8, 10, 11 |
| `src/preprocessing.py` | 9, 11, 13 |
| `src/models.py` | 12 |
| `src/validation.py` | 14, 15, 16 |
| `src/calibration.py` | 17, 20 |
| `src/metrics.py` | 19, 20, 21 |
| `src/shift.py` | 11 (missingness shift) |
| `outputs/internal/` | 14, 19, 20, 21 |
| `outputs/external_uci/` | 16, 19, 20, 21 |
| `outputs/external_kaggle_uci/` | 16, 19, 20, 21 |
| `outputs/calibration/` | 17, 20 |
| `outputs/shift/` | 11, 22 |
| `data/ingestion_report.json` | 7, 10, 11 |
| `configs/pipeline.yaml` | 12, 13, 25 |

---

## Completion Status

| Category | Items | Complete | Missing |
|----------|-------|----------|---------|
| Title & Abstract | 3 | 3 | 0 |
| Introduction | 2 | 2 | 0 |
| Methods — Data | 6 | 6 | 0 |
| Methods — Model | 3 | 3 | 0 |
| Methods — Validation | 3 | 3 | 0 |
| Results | 4 | 4 | 0 |
| Discussion | 3 | 3 | 0 |
| Other | 3 | 2 | 1 (funding) |

**Overall:** 26/27 items mapped. Funding (Item 26) N/A for academic/research project.
