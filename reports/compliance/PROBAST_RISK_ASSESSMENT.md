# PROBAST Risk of Bias and Applicability Assessment

**Domain-level judgments for the Heart Model Transportability project.**  
Reference: [PROBAST+AI (BMJ 2024)](https://www.bmj.com/content/388/bmj-2024-082505)

---

## Overall Judgment

| Assessment | Judgment | Rationale |
|------------|----------|-----------|
| **Risk of Bias** | **Low** | Appropriate validation design, no data leakage, documented preprocessing, bootstrap CIs |
| **Applicability** | **Moderate** | Public benchmarks; label/measurement heterogeneity across sites; research-only scope |

---

## Domain 1: Participants

| Signaling Question | Judgment | Rationale |
|--------------------|----------|------------|
| Was there a clear definition of the target population? | **Yes** | Kaggle: adults with cardiovascular screening; UCI: patients evaluated for heart disease (4 sites) |
| Were inclusion/exclusion criteria appropriate? | **Yes** | All available records used; exclusion only for physiologically implausible BP (ap_hi > 370, ap_lo < 0, ap_lo > ap_hi) — documented in `src/ingest.py` `drop_kaggle_bp_outliers()` |
| Were participants recruited consecutively or randomly? | **Unclear** | Public datasets; recruitment not under study control; UCI/Kaggle collection protocols not fully specified |
| Were there important differences between development and validation populations? | **Yes (intended)** | External validation explicitly tests transportability across sites; differences are the study focus |
| **Domain Risk** | **Low** | Exclusions documented; target population defined; heterogeneity is intentional for transportability research |

**Evidence locations:** `README.md` §Datasets; `ImplementationPlan/data_ingestion.md` §2–§4; `src/ingest.py`; `data/ingestion_report.json`

---

## Domain 2: Predictors

| Signaling Question | Judgment | Rationale |
|--------------------|----------|------------|
| Were predictors defined and measured consistently? | **Partially** | Kaggle vs UCI: different schemas; CFS (age, sex, sys_bp) bridges; cholesterol binned for Kaggle↔UCI (`validation.py` `bin_cholesterol_uci`) |
| Were predictors available at the time the model would be used? | **Yes** | Routine clinical/lifestyle variables (age, sex, BP, cholesterol, etc.) |
| Were predictors handled appropriately (missing data)? | **Yes** | Median/mode imputation for LR/RF; native NaN for XGB/LGBM; `effective_cfs()` drops features with >40% missing; missingness reported in `ingestion_report.json` |
| Were predictors selected without knowledge of outcome? | **Yes** | Feature sets defined a priori per `preprocessing.py`; no outcome-driven selection |
| **Domain Risk** | **Low–Moderate** | CFS restriction reduces predictors; documented and penalty quantified (RQ5); measurement heterogeneity across sites acknowledged |

**Evidence locations:** `src/preprocessing.py` — `resolve_features()`, `effective_cfs()`, `build_imputer()`; `src/validation.py` — `cfs_penalty()`, `_maybe_add_binned_cholesterol()`; `README.md` §Known Limitations

---

## Domain 3: Outcome

| Signaling Question | Judgment | Rationale |
|--------------------|----------|------------|
| Was the outcome defined and determined appropriately? | **Yes** | Kaggle: `cardio` (0/1); UCI: `num` binarized 0 vs 1–4 for presence |
| Was the outcome determined without knowledge of predictors? | **Yes** | Standard diagnostic/clinical outcome; no circularity |
| Was the outcome determined consistently across development and validation? | **Partially** | "Heart disease" definitions may differ across sources (Kaggle CVD screening vs UCI angiographic); acknowledged in README §Known Limitations |
| **Domain Risk** | **Low** | Standard binarization; clinical relevance; label heterogeneity documented as limitation |

**Evidence locations:** `README.md` §Datasets, §Known Limitations; `src/ingest.py` — `binarize_target()`

---

## Domain 4: Analysis

| Signaling Question | Judgment | Rationale |
|--------------------|----------|------------|
| Were there a sufficient number of events? | **Yes** | Kaggle ~70K; UCI sites 123–303; prevalence 0–1; bootstrap B=200/500 for small N |
| Were continuous predictors handled appropriately? | **Yes** | StandardScaler for LR; tree models scale-invariant; cholesterol binned for CFS |
| Were all enrolled participants included? | **Yes** | No post-enrollment exclusion except BP outliers (documented) |
| Was model development and validation appropriate? | **Yes** | Train/test split before any fit; preprocessing fit on train only; internal 80/20 + external multi-site matrix |
| Was model performance assessed appropriately? | **Yes** | ROC-AUC, PR-AUC, Brier, ECE, MCE; bootstrap 95% CIs; repeated CV for Switzerland |
| Was overfitting addressed? | **Yes** | Inner CV for tuning; external validation on held-out sites; no tuning on test |
| **Domain Risk** | **Low** | No data leakage; appropriate validation strategy; uncertainty quantified |

**Evidence locations:** `src/validation.py` — `internal_split()`, `run_internal_validation()`, `run_external_uci_matrix()`; `src/metrics.py` — `bootstrap_metrics()`; `src/preprocessing.py` — `fit_transform_train()`, `transform_test()`; `ImplementationPlan/pipeline_plan.md` §1.3–§1.6

---

## Applicability Concerns

| Concern | Severity | Description |
|---------|----------|-------------|
| **Setting** | Moderate | Public benchmarks; not prospectively collected; may not reflect real-world deployment |
| **Population** | Moderate | UCI sites (Cleveland, Hungary, Switzerland, VA) from 1988–1990; Kaggle from different era/source |
| **Predictors** | Low | CFS restricts to 3–4 features for Kaggle↔UCI; full-feature models use site-specific sets |
| **Outcome** | Low–Moderate | Label heterogeneity ("heart disease" vs "CVD") across sources; binarization standard but not identical |

---

## Summary Table

| Domain | Risk of Bias | Applicability | Key Evidence |
|--------|--------------|---------------|--------------|
| **Participants** | Low | — | Documented exclusions; defined populations |
| **Predictors** | Low–Moderate | Low | CFS penalty quantified; missingness handled |
| **Outcome** | Low | Low–Moderate | Standard binarization; label heterogeneity noted |
| **Analysis** | Low | — | No leakage; bootstrap CIs; external validation |
| **Overall** | **Low** | **Moderate** | Research-only; transportability focus; limitations documented |

---

## Mitigations in Place

| Risk | Mitigation |
|------|-------------|
| Small UCI cohorts | Repeated CV (Switzerland 5×5); bootstrap B=500 for N<200 |
| CFS narrowness | RQ5 quantifies penalty; internal CFS baselines reported |
| Recalibration overfitting | Calibration/evaluation split; 3-fold CV for N<100 |
| Missingness heterogeneity | `effective_cfs()` 40% threshold; per-pair feature count logged |
| Label mismatch | Acknowledged in README; not correctable in design |

---

## References

- Collins GS, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ 2024;385:e078378.
- Moons KGM, et al. PROBAST+AI: a tool for assessing prediction models including those based on artificial intelligence. BMJ 2024;388:e082505.
