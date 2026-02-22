# How Transportable Are Tabular Heart-Disease Classifiers?

**Internal Validation, Multi-Site External Validation, Calibration, and Dataset Shift Using Public Benchmarks**

> Can a heart-disease prediction model trained on one dataset/site still be trusted in another? Are its probabilities reliable? And how much can calibration or lightweight updating fix when populations and measurements change?

---

## Research Motivation

Health ML models often look strong under internal evaluation but degrade silently under dataset shift. Miscalibrated probabilities lead to inconsistent triage and unreliable thresholds. Most prior work trains and tests on a single random split and reports accuracy/AUC — ignoring **external validation**, **calibration**, and **dataset shift diagnostics**.

This project fills that gap with an **evaluation-first transportability study**: hold the modeling pipeline fixed, vary the validation setting (internal → external), and measure how discrimination and probability reliability change across sites.

## The Gap

| What most prior work does | What this project adds |
|---|---|
| Train/test on one dataset (random split), report accuracy/AUC | Multi-site **external validation** across independent cohorts |
| Ignore probability reliability | **Calibration analysis** (Brier score, reliability diagrams, ECE) |
| No explanation for performance drops | **Dataset shift + feature-overlap diagnostics** to explain *why* it breaks |

## Research Questions

| ID | Question |
|---|---|
| **RQ1** | What is baseline discrimination and calibration under standard 80/20 internal validation within each dataset/site? |
| **RQ2** | How much does performance change under multi-site external validation (train on site(s) → test on a different site)? |
| **RQ3** | Which predictors show the largest distribution shifts, and how do these shifts relate to drops in discrimination/calibration? |
| **RQ4** | Can post-hoc calibration and lightweight updating restore probability reliability in external sites without full retraining? |
| **RQ5** | How much performance is lost when restricting to a Common Feature Set (CFS) for cross-dataset testing, and how does that change transportability conclusions? |

## Study Design

A **two-layer evaluation** avoids confounding from size mismatch between datasets:

1. **Internal validation** — 80/20 split within each dataset/site (repeated CV / bootstrapping for small cohorts)
2. **External validation** — Train on site(s) → test on a held-out site (multi-site UCI matrix + optional Kaggle stress test)

This is a **quasi-experimental** design: the pipeline (preprocessing, model family, metrics) is held constant; the "treatment" is the external setting + dataset shift; outcomes are changes in performance and calibration.

### Two-Phase Plan

**Phase 1 — Diagnose the problem**
Measure how performance and calibration change within-dataset vs. under external validation. Produce metrics tables, calibration plots, drift summaries, and "where it breaks" subgroup slices.

**Phase 2 — Mitigate**
Apply post-hoc calibration (Platt scaling, isotonic regression), lightweight updating (intercept/slope recalibration), and size-matched sensitivity analyses. Document when recalibration is enough vs. when feature mismatch blocks transfer.

## Datasets

### Dataset A — Kaggle Cardiovascular Disease (`cardio_train`)

~70,000 records, 11 features + target. Used as a large benchmark and optional stress test under explicit feature-overlap constraints.

| Column | Role |
|---|---|
| `age` | Predictor (days → years) |
| `gender` | Predictor (binary sex indicator) |
| `height`, `weight` | Predictors (body size / BMI-related) |
| `ap_hi`, `ap_lo` | Predictors (systolic / diastolic BP) |
| `cholesterol` | Predictor (ordinal 1–3) |
| `gluc` | Predictor (ordinal 1–3) |
| `smoke`, `alco`, `active` | Predictors (binary lifestyle factors) |
| `cardio` | **Target** (0/1 CVD presence) |

### Dataset B — UCI Heart Disease (Multi-Site)

Four cohorts sharing a common 14-attribute schema: **Cleveland**, **Hungary**, **Switzerland**, **VA Long Beach**. The `num` field (0–4) is binarized to 0 vs 1–4 for presence.

| Column | Role |
|---|---|
| `age`, `sex` | Demographics |
| `cp` | Chest pain type (categorical) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (binary) |
| `restecg` | Resting ECG results (categorical) |
| `thalach` | Max heart rate achieved |
| `exang` | Exercise-induced angina (binary) |
| `oldpeak` | ST depression (exercise vs rest) |
| `slope` | Peak exercise ST segment slope |
| `ca` | Major vessels colored by fluoroscopy |
| `thal` | Thalassemia test result |
| `num` | **Target** (binarize 0 vs 1–4) |

**Feature overlap plan (CFS):** For any Kaggle↔UCI transportability experiment, a Common Feature Set is defined and the performance penalty of restricting to overlap vs. full-feature within-dataset models is reported.

## Evaluation Plan

| Metric | What it answers |
|---|---|
| ROC-AUC / PR-AUC | Does ranking performance transfer? |
| Brier score | Are probabilities accurate overall? |
| Calibration curve | Are predicted probabilities trustworthy? |
| ECE | How far off are probabilities on average? |
| Threshold metrics (sens/spec) | Practical comparison across operating points |
| Bootstrap CIs / repeated CV | How stable are results, especially for small cohorts? |

## Techniques

- **Modeling:** Logistic Regression (regularized), Random Forest, Gradient Boosting (XGBoost/LightGBM)
- **Calibration:** Reliability diagrams, Brier score, Platt scaling, isotonic regression
- **External validation:** Internal 80/20 baselines + multi-site external validation matrix
- **Shift diagnostics:** Feature distribution comparisons, drift flags, prevalence differences
- **Reporting:** TRIPOD+AI for completeness, PROBAST+AI for bias risk assessment

## Project Structure

```
Heart-model-transportability/
├── Dataset/                  # Raw data files
│   ├── kaggle_cardio_train.csv
│   ├── long-beach-va.data
│   ├── hungarian.data
│   └── switzerland.data
├── data/                     # Cleaned / processed datasets
├── src/                      # Loading, cleaning, modeling, calibration, drift, evaluation
├── configs/                  # Run configs: site pairs, models, seeds
├── outputs/                  # Tables, plots, saved models
├── reports/                  # Markdown/HTML evaluation report (TRIPOD+AI aligned)
├── app/                      # Transportability Dashboard (Streamlit/Dash)
├── ImplementationPlan/       # Pipeline, evaluation, and data ingestion plans
├── Literature/               # Reference papers
├── Research Paper/           # Manuscript drafts
├── requirements.txt
└── README.md
```

## Deliverables

### Phase 1 — Diagnosis

- Cleaned datasets per cohort + documented CFS version
- Standardized dataset loaders and preprocessing rules
- Internal (80/20) + external (multi-site matrix) evaluation tables
- ROC/PR curves, calibration plots, drift + missingness summaries
- Reproducible run outputs (saved metrics + figures)

### Phase 2 — Mitigation + Dashboard

- Calibration/updating module with before-after comparisons
- "Reliability improvements" analysis (recalibration-enough vs. feature-mismatch-blocked)
- **Transportability Dashboard** (Streamlit/Dash):
  - Dataset/site overview
  - Internal (80/20) results
  - External validation matrix
  - Calibration before/after
  - Shift + overlap diagnostics
  - Exportable report/model card (TRIPOD+AI aligned)

## Known Limitations

- **Label/measurement mismatch** — "Heart disease" and feature definitions can differ across sources
- **Feature overlap constraint** — Cross-dataset tests use a reduced CFS; dropping predictors can reduce performance (quantified)
- **Small external cohorts** — Some UCI sites have limited N; uncertainty is reported via bootstrap CIs / repeated CV
- **Research-only scope** — This is benchmark/validation evidence, not clinical decision support

## References

- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- [Kaggle Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- [TRIPOD+AI Reporting Guideline (BMJ)](https://www.bmj.com/content/385/bmj-2023-078378)
- [PROBAST (Annals of Internal Medicine)](https://www.acpjournals.org/doi/10.7326/M18-1376)
- [Dataset Shift in Machine Learning (NEJM AI / Quiñonero-Candela et al.)](https://direct.mit.edu/books/edited-volume/3528/Dataset-Shift-in-Machine-Learning)
