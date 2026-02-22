# Evaluation Plan — Common Framework for All Pipeline Experiments

**Scope:** Everything that happens *after* the pipeline writes raw predictions, metrics, shift diagnostics, and calibration artifacts to `outputs/`. This document covers: metric aggregation, statistical testing, visualization, research question synthesis, and final report generation.

**Input:** All artifacts produced by `src/pipeline.py` — per-experiment `results.json`, `predictions.parquet`, shift diagnostics, calibration before/after dicts, and size-matched sensitivity results (see `pipeline_outputs.md`).

**Output:** Aggregated evaluation tables, publication-ready figures, statistical comparison results, and a structured final report aligned with TRIPOD+AI. All evaluation outputs land in `reports/`.

---

## 1  Metric Inventory

Every experiment — internal, external UCI, external Kaggle↔UCI — produces the same core metric set. This uniformity is the backbone of cross-experiment comparison.

### 1.1  Discrimination Metrics

| Metric | Function | Interpretation |
|---|---|---|
| **ROC-AUC** | `roc_auc_score(y_true, y_prob)` | Probability that a randomly chosen positive ranks higher than a negative. Threshold-free. Primary comparison metric. |
| **PR-AUC** | `average_precision_score(y_true, y_prob)` | Summarizes precision-recall tradeoff. More informative than ROC-AUC under class imbalance. |

### 1.2  Calibration Metrics

| Metric | Function | Interpretation |
|---|---|---|
| **Brier Score** | `brier_score_loss(y_true, y_prob)` | Mean squared error of probabilities. Decomposes into discrimination + calibration + uncertainty. Lower is better. |
| **ECE** | Custom (equal-width bins) | Weighted mean |predicted − observed| across probability bins. Measures average miscalibration. |
| **MCE** | Custom (max bin gap) | Worst-case bin-level miscalibration. |

### 1.3  Threshold Metrics (at τ = 0.5 unless otherwise noted)

| Metric | Function |
|---|---|
| **Accuracy** | `accuracy_score` |
| **F1** | `f1_score` |
| **Precision** | `precision_score` |
| **Recall (Sensitivity)** | `recall_score` |
| **Specificity** | `recall_score(pos_label=0)` |

### 1.4  Uncertainty Quantification

| Method | When Applied | Parameters |
|---|---|---|
| **Bootstrap CIs** (95%, percentile method) | All experiments | B=200 for N ≥ 200, B=500 for N < 200 |
| **Repeated Stratified CV** | Switzerland internal (N=123) | 5×5 = 25 folds |

### 1.5  Shift Diagnostics (per external pair)

| Metric | Scope |
|---|---|
| **Prevalence shift** | |train prevalence − test prevalence| |
| **KS statistic** | Per continuous feature |
| **Chi² statistic** | Per categorical feature |
| **PSI** | Per feature (>0.10 moderate, >0.25 severe) |
| **C2ST AUC** | Multivariate: can a classifier distinguish train from test? |
| **Missingness shift** | Per-feature Δ in missing rates |

---

## 2  Aggregation Strategy

Raw per-experiment results are scattered across `outputs/`. Evaluation consolidates them into flat comparison tables.

### 2.1  Master Results Table

One row per experiment × model. Columns:

```
experiment_type | train_site(s) | test_site | model | roc_auc | roc_auc_ci_lower | roc_auc_ci_upper |
pr_auc | brier_score | ece | mce | accuracy | f1 | precision | recall | specificity |
n_train | n_test | n_features_used | features_used | best_params
```

```python
# src/evaluation.py

def build_master_table(outputs_dir: str = "outputs") -> pd.DataFrame:
    rows = []
    for results_path in Path(outputs_dir).rglob("results.json"):
        with open(results_path) as f:
            r = json.load(f)
        row = {
            "experiment_type": r["experiment_type"],
            "train_sites": r.get("train_sites", r.get("site")),
            "test_site": r.get("test_site", r.get("site")),
            "model": r["model"],
            **r["metrics"],
            "n_train": r.get("n_train"),
            "n_test": r["metrics"]["n_test"],
            "n_features": len(r.get("features_used", [])),
        }
        if "bootstrap_cis" in r:
            for metric_name, ci in r["bootstrap_cis"].items():
                row[f"{metric_name}_ci_lower"] = ci["ci_lower"]
                row[f"{metric_name}_ci_upper"] = ci["ci_upper"]
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["experiment_type", "test_site", "model"])
```

### 2.2  Pivot Views

From the master table, generate these pivot tables:

| Pivot | Rows | Columns | Values | Answers |
|---|---|---|---|---|
| **Internal baseline** | Site | Model | ROC-AUC [CI] | RQ1 — within-site performance |
| **External matrix (pairwise)** | Train site → Test site | Model | ROC-AUC [CI] | RQ2 — pairwise degradation |
| **External matrix (LOSO)** | Left-out site | Model | ROC-AUC [CI] | RQ2 — leave-one-out degradation |
| **Δ matrix** | Train→Test pair | Model | External AUC − Internal AUC | RQ2 — magnitude of drop |
| **Calibration summary** | Experiment | Model | Brier, ECE, MCE | RQ1/RQ2/RQ4 — probability reliability |
| **Shift summary** | Train→Test pair | — | Mean PSI, prevalence diff, C2ST AUC | RQ3 — what shifted |
| **Recalibration comparison** | Experiment × method | Model | ECE before, ECE after, Δ | RQ4 — does recalibration help? |
| **CFS penalty** | Site | Model | Full AUC, CFS AUC, Δ | RQ5 — cost of feature restriction |

```python
def pivot_internal_baseline(master: pd.DataFrame) -> pd.DataFrame:
    internal = master[master["experiment_type"] == "internal"]
    return internal.pivot_table(
        index="test_site", columns="model",
        values="roc_auc", aggfunc="first"
    ).round(3)

def pivot_external_delta(master: pd.DataFrame) -> pd.DataFrame:
    internal_auc = (master[master["experiment_type"] == "internal"]
                    .set_index(["test_site", "model"])["roc_auc"])
    external = master[master["experiment_type"].str.startswith("external")].copy()
    external["internal_auc"] = external.apply(
        lambda r: internal_auc.get((r["test_site"], r["model"]), np.nan), axis=1
    )
    external["auc_delta"] = external["roc_auc"] - external["internal_auc"]
    return external
```

---

## 3  Statistical Comparisons

### 3.1  Internal vs External Performance Drop

For each test site × model, test whether external AUC is significantly lower than internal AUC:

| Comparison | Method | Rationale |
|---|---|---|
| AUC drop significance | DeLong test on paired ROC curves (when predictions on same test set exist) or bootstrap difference CI | Standard for comparing two AUCs |
| Multi-model comparison | Friedman test across models per experiment type | Non-parametric rank test for comparing multiple classifiers |
| Post-hoc pairwise | Nemenyi test (if Friedman significant) | Identifies which model pairs differ |

```python
def bootstrap_auc_difference(y_true, y_prob_internal, y_prob_external, B=2000, seed=42):
    """Bootstrap CI for AUC_internal - AUC_external."""
    rng = np.random.RandomState(seed)
    deltas = []
    n = len(y_true)
    for _ in range(B):
        idx = rng.choice(n, size=n, replace=True)
        auc_int = roc_auc_score(y_true[idx], y_prob_internal[idx])
        auc_ext = roc_auc_score(y_true[idx], y_prob_external[idx])
        deltas.append(auc_int - auc_ext)
    return {
        "mean_delta": np.mean(deltas),
        "ci_lower": np.percentile(deltas, 2.5),
        "ci_upper": np.percentile(deltas, 97.5),
        "significant": np.percentile(deltas, 2.5) > 0,  # CI excludes 0
    }
```

### 3.2  Shift–Performance Correlation

Rank-correlate shift magnitude with performance degradation across all external pairs:

| X variable | Y variable | Method |
|---|---|---|
| Mean PSI (across features) | AUC drop | Spearman ρ |
| Prevalence shift | Brier score change | Spearman ρ |
| C2ST AUC | External AUC | Spearman ρ |
| Per-feature KS/PSI | Per-feature importance × AUC drop | Weighted Spearman ρ |

```python
from scipy.stats import spearmanr

def shift_performance_analysis(shift_df: pd.DataFrame, perf_df: pd.DataFrame) -> dict:
    merged = shift_df.merge(perf_df, on=["train_sites", "test_site"])
    results = {}
    pairs = [
        ("mean_psi", "auc_drop"),
        ("prevalence_diff", "brier_change"),
        ("c2st_auc", "roc_auc"),
    ]
    for x_col, y_col in pairs:
        rho, p = spearmanr(merged[x_col], merged[y_col])
        results[f"{x_col}_vs_{y_col}"] = {"spearman_rho": rho, "p_value": p}
    return results
```

### 3.3  Recalibration Effectiveness Testing

For each recalibration method, test whether post-recalibration ECE is significantly lower than pre-recalibration ECE:

```python
def recalibration_significance(ece_before: list, ece_after: list) -> dict:
    """Paired Wilcoxon signed-rank test across experiments."""
    from scipy.stats import wilcoxon
    stat, p = wilcoxon(ece_before, ece_after, alternative="greater")
    return {
        "mean_ece_before": np.mean(ece_before),
        "mean_ece_after": np.mean(ece_after),
        "mean_improvement": np.mean(np.array(ece_before) - np.array(ece_after)),
        "wilcoxon_stat": stat,
        "p_value": p,
    }
```

---

## 4  Visualization Plan

All figures are generated by `src/evaluation.py` and saved to `reports/figures/`. Format: PDF (vector) + PNG (raster at 300 DPI).

### 4.1  Discrimination Figures

| Figure | Type | Content | Answers |
|---|---|---|---|
| **F1: Internal ROC curves** | ROC curve overlay (one subplot per site) | 4 model curves + diagonal per site | RQ1 |
| **F2: External ROC curves** | ROC curve overlay (one subplot per test site) | Internal vs best external per site | RQ2 |
| **F3: AUC heatmap** | Heatmap (train site × test site) | ROC-AUC per model, one heatmap per model | RQ2 |
| **F4: AUC drop bar chart** | Grouped bar (test site × model) | Internal AUC − External AUC, error bars = bootstrap CI | RQ2 |
| **F5: PR curves** | PR curve overlay | Same structure as F1/F2 | RQ1/RQ2 |

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

def plot_auc_heatmap(master: pd.DataFrame, model_key: str, output_dir: str):
    ext = master[(master["experiment_type"] == "external_uci") & (master["model"] == model_key)]
    pivot = ext.pivot_table(index="train_sites", columns="test_site", values="roc_auc")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.5, vmax=1.0,
                linewidths=0.5, ax=ax)
    ax.set_title(f"External ROC-AUC: {model_key.upper()}")
    fig.savefig(Path(output_dir) / f"auc_heatmap_{model_key}.pdf", bbox_inches="tight")
    fig.savefig(Path(output_dir) / f"auc_heatmap_{model_key}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
```

### 4.2  Calibration Figures

| Figure | Type | Content | Answers |
|---|---|---|---|
| **F6: Reliability diagrams (internal)** | Calibration curve (one subplot per site × model) | Observed vs predicted frequency, 10 bins + perfect diagonal | RQ1 |
| **F7: Reliability diagrams (external)** | Same layout | Pre-recalibration calibration curves for external experiments | RQ2 |
| **F8: Recalibration before/after** | Paired calibration curves | Same experiment: raw vs Platt vs isotonic vs temperature | RQ4 |
| **F9: ECE comparison bar chart** | Grouped bar (method × experiment) | ECE before and after each recalibration method | RQ4 |
| **F10: Brier decomposition** | Stacked bar | Brier = reliability + resolution + uncertainty per experiment | RQ1/RQ4 |

```python
def plot_reliability_diagram(y_true, y_prob, title: str, ax=None):
    from sklearn.calibration import calibration_curve
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    ax.plot(mean_pred, fraction_pos, "s-", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.fill_between(mean_pred, fraction_pos, mean_pred, alpha=0.15, color="red")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
```

### 4.3  Shift Diagnostic Figures

| Figure | Type | Content | Answers |
|---|---|---|---|
| **F11: PSI heatmap** | Heatmap (feature × site pair) | PSI values, color-coded by severity thresholds | RQ3 |
| **F12: Feature distribution comparison** | Overlaid histograms/KDE (one subplot per feature) | Train vs test distributions for the pair with largest AUC drop | RQ3 |
| **F13: Shift vs performance scatter** | Scatter (mean PSI vs AUC drop) | One point per external pair, color = test site, shape = model | RQ3 |
| **F14: Missingness heatmap** | Heatmap (feature × site) | Missing % per cell, annotated | RQ3/RQ5 |
| **F15: C2ST AUC vs external AUC** | Scatter | One point per pair | RQ3 |

```python
def plot_shift_vs_performance(shift_perf_df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    scatter_pairs = [
        ("mean_psi", "auc_drop", "Mean PSI", "AUC Drop"),
        ("prevalence_diff", "brier_change", "Prevalence Shift", "Brier Δ"),
        ("c2st_auc", "roc_auc", "C2ST AUC", "External ROC-AUC"),
    ]
    for ax, (x, y, xlabel, ylabel) in zip(axes, scatter_pairs):
        sns.scatterplot(data=shift_perf_df, x=x, y=y, hue="test_site",
                        style="model", s=80, ax=ax)
        rho, p = spearmanr(shift_perf_df[x], shift_perf_df[y])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"ρ = {rho:.2f}, p = {p:.3f}")

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "shift_vs_performance.pdf", bbox_inches="tight")
    plt.close(fig)
```

### 4.4  CFS & Size-Sensitivity Figures

| Figure | Type | Content | Answers |
|---|---|---|---|
| **F16: CFS penalty bar chart** | Grouped bar (site × model) | Full-feature AUC vs CFS-only AUC | RQ5 |
| **F17: Size-matched comparison** | Box plot (site pair × model) | Full-training-set AUC vs subsampled distribution (K=20) | Deconfounding |
| **F18: Effective CFS feature count** | Bar chart (site pair) | Number of usable features per pair after missingness filtering | RQ5 |

---

## 5  Research Question Synthesis

Each RQ maps to specific tables, figures, and statistical tests. The evaluation module generates a per-RQ summary block.

### RQ1 — Internal Baselines

**Inputs:** `outputs/internal/*/results.json`

**Evaluation:**
1. Internal baseline pivot (§2.2) — ROC-AUC, PR-AUC, Brier with CIs per site × model.
2. Figures F1 (ROC), F5 (PR), F6 (reliability diagrams), F10 (Brier decomposition).
3. Narrative: Which model dominates per site? How do CIs compare across small vs large cohorts? Is Kaggle's apparent performance inflated by size?

**Key table:**

| Site | N | LR AUC [CI] | RF AUC [CI] | XGB AUC [CI] | LGBM AUC [CI] | Best Brier |
|---|---|---|---|---|---|---|
| Kaggle | 70,000 | — | — | — | — | — |
| Cleveland | 303 | — | — | — | — | — |
| Hungary | 294 | — | — | — | — | — |
| VA | 200 | — | — | — | — | — |
| Switzerland | 123 | — | — | — | — | — |

### RQ2 — External Validation Degradation

**Inputs:** `outputs/external_uci/*/results.json`, `outputs/external_kaggle_uci/*/results.json`, plus internal baselines for Δ.

**Evaluation:**
1. Δ matrix: external AUC − internal AUC per test site × model (§2.2).
2. AUC heatmap (F3): train×test grid per model.
3. AUC drop bar chart (F4): magnitude and CIs.
4. External ROC curves (F2): overlay internal vs external.
5. Statistical test: bootstrap AUC difference CI per pair (§3.1). Flag pairs where CI excludes 0.
6. Friedman + Nemenyi: do models rank consistently across external settings?
7. LOSO vs pairwise: does training on more sites help?

**Key table:**

| Train → Test | LR Δ AUC | RF Δ AUC | XGB Δ AUC | LGBM Δ AUC | Worst Drop |
|---|---|---|---|---|---|
| Cleveland → Hungary | — | — | — | — | — |
| ... | ... | ... | ... | ... | ... |
| LOSO → Switzerland | — | — | — | — | — |

### RQ3 — Dataset Shift Explaining Performance Drops

**Inputs:** `outputs/shift/*/shift_diagnostics.json`, `outputs/shift/*/feature_shift.parquet`, `outputs/shift/shift_performance_correlation.parquet`.

**Evaluation:**
1. PSI heatmap (F11): which features shifted most for which pairs.
2. Feature distribution overlays (F12): visual comparison for the worst pair.
3. Shift vs performance scatter (F13, F15): does more shift = more drop?
4. Spearman correlations (§3.2): quantify the relationship.
5. Missingness heatmap (F14): structural missingness differences.
6. Narrative: identify the "shift signature" — is it prevalence, covariate, or missingness that drives degradation?

**Key outputs:**
- Ranked list of most-shifted features per pair
- Spearman ρ table (shift metric × performance metric)
- Binary classification: "prevalence-driven" vs "covariate-driven" vs "missingness-driven" degradation per pair

### RQ4 — Recalibration Effectiveness

**Inputs:** `outputs/calibration/before/*`, `outputs/calibration/recalibration/*`, `outputs/calibration/updating/*`.

**Evaluation:**
1. Calibration summary pivot (§2.2): Brier, ECE, MCE before vs after per method.
2. Reliability diagrams before/after (F8): visual calibration improvement.
3. ECE comparison bar chart (F9): grouped by method.
4. Wilcoxon signed-rank test (§3.3): is improvement statistically significant?
5. Intercept-only vs intercept+slope (Stage 2.3 outputs): prevalence-only fix vs full logistic recalibration.
6. Narrative: when is recalibration sufficient? When does feature mismatch block improvement?

**Decision matrix:**

| Outcome | Interpretation | Action |
|---|---|---|
| ECE drops to < 0.05 after recalibration | Miscalibration was correctable | Recalibration is sufficient for this pair |
| Intercept-only works as well as full | Pure prevalence shift | Simple offset correction |
| Full recalibration needed | Spread/sharpness also miscalibrated | Need slope + intercept |
| No method helps | Feature-level mismatch | Recalibration is insufficient; retraining or feature alignment needed |

### RQ5 — CFS Performance Penalty

**Inputs:** Internal full-feature results, internal CFS-only results, external Kaggle↔UCI results.

**Evaluation:**
1. CFS penalty pivot (§2.2): full AUC vs CFS AUC per site × model.
2. CFS penalty bar chart (F16): absolute and relative AUC loss.
3. Effective CFS feature count (F18): how many features survive missingness filtering per pair.
4. Compare Kaggle↔UCI external AUC to CFS-only internal AUC — is the external drop beyond the CFS penalty?
5. Narrative: how much of the Kaggle↔UCI transportability problem is feature restriction vs population shift?

**Key table:**

| Site | Model | Full AUC | CFS AUC | Δ AUC | Relative Drop % |
|---|---|---|---|---|---|
| Cleveland | LR | — | — | — | — |
| ... | ... | ... | ... | ... | ... |

---

## 6  Report Structure

The final evaluation report lives at `reports/evaluation_report.md` (with HTML export). Structure follows TRIPOD+AI where applicable.

### 6.1  Report Outline

```
1. Executive Summary
   - One-paragraph answer to each RQ
   - "Headline" metrics: best internal AUC, worst external drop, recalibration recovery

2. Methods Summary
   - Data sources and sample sizes (table)
   - Preprocessing, model families, tuning protocol (reference pipeline_plan.md)
   - Evaluation metrics and statistical tests used

3. Results — RQ1: Internal Baselines
   - Table: Internal AUC/Brier per site × model
   - Figures: F1, F5, F6, F10
   - Key finding paragraph

4. Results — RQ2: External Validation
   - Table: External AUC matrix + Δ table
   - Figures: F2, F3, F4
   - Table: LOSO vs pairwise comparison
   - Statistical tests: AUC drop significance, Friedman ranking
   - Key finding paragraph

5. Results — RQ3: Dataset Shift
   - Table: Shift metric summary per pair
   - Figures: F11, F12, F13, F14, F15
   - Table: Spearman correlations
   - Key finding paragraph

6. Results — RQ4: Recalibration
   - Table: ECE/Brier before vs after per method
   - Figures: F8, F9
   - Table: Intercept-only vs full recalibration outcomes
   - Statistical test: Wilcoxon
   - Key finding paragraph

7. Results — RQ5: CFS Penalty
   - Table: CFS penalty per site × model
   - Figures: F16, F17, F18
   - Key finding paragraph

8. Discussion
   - Transportability patterns (which directions transfer well / poorly)
   - Shift type → degradation type mapping
   - Recalibration sufficiency conditions
   - Practical implications for heart disease risk model deployment
   - Limitations

9. Appendices
   - A: Full metric tables (all experiments, all metrics)
   - B: All calibration curves
   - C: Feature shift detail tables
   - D: Size-matched sensitivity full results
   - E: Hyperparameter selections per experiment
```

### 6.2  TRIPOD+AI Alignment

| TRIPOD+AI Item | Location in Report |
|---|---|
| Title / Abstract | §1 Executive Summary |
| Source of data | §2 Methods Summary |
| Participants (eligibility, sites) | §2 + data_ingestion.md reference |
| Outcome definition | §2 (binarized target) |
| Predictors | §2 (feature sets per experiment type) |
| Sample size | §2 table + per-experiment N |
| Missing data | §5 RQ3 (missingness heatmap) + §2 |
| Model development | §2 (reference pipeline_plan.md §1.2) |
| Model performance | §3–§7 (all RQ sections) |
| Discrimination | §3 RQ1, §4 RQ2 |
| Calibration | §3 RQ1, §6 RQ4 |
| Updating / recalibration | §6 RQ4 |
| External validation | §4 RQ2 |
| Limitations | §8 Discussion |

### 6.3  PROBAST+AI Risk-of-Bias Domains

Include a self-assessment table in the appendix:

| Domain | Rating | Justification |
|---|---|---|
| Participants | Low risk | All available records used; exclusion only for physiologically implausible BP values (documented) |
| Predictors | Low / moderate | CFS restriction reduces predictors; documented and penalty quantified |
| Outcome | Low risk | Standard binarization of heart disease diagnosis |
| Analysis | Low risk | Appropriate validation strategy per sample size; bootstrap CIs; no data leakage (fit on train only) |
| Overall | Low risk | — |

---

## 7  Evaluation Pipeline Implementation

### 7.1  Module: `src/evaluation.py`

```python
# src/evaluation.py

from pathlib import Path
import json
import pandas as pd
import numpy as np

def run_evaluation(outputs_dir: str = "outputs", reports_dir: str = "reports"):
    reports_path = Path(reports_dir)
    figures_path = reports_path / "figures"
    tables_path = reports_path / "tables"
    figures_path.mkdir(parents=True, exist_ok=True)
    tables_path.mkdir(parents=True, exist_ok=True)

    # 1. Aggregate
    master = build_master_table(outputs_dir)
    master.to_csv(tables_path / "master_results.csv", index=False)
    master.to_parquet(tables_path / "master_results.parquet")

    # 2. Pivot tables
    pivots = generate_all_pivots(master)
    for name, df in pivots.items():
        df.to_csv(tables_path / f"{name}.csv")

    # 3. Statistical tests
    stat_results = run_statistical_tests(master, outputs_dir)
    with open(tables_path / "statistical_tests.json", "w") as f:
        json.dump(stat_results, f, indent=2, default=str)

    # 4. Figures
    generate_all_figures(master, outputs_dir, figures_path)

    # 5. RQ synthesis
    rq_summaries = synthesize_rq_answers(master, stat_results, pivots)
    with open(reports_path / "rq_summaries.json", "w") as f:
        json.dump(rq_summaries, f, indent=2, default=str)

    # 6. Report generation
    generate_report(master, pivots, stat_results, rq_summaries, reports_path)

    return master
```

### 7.2  Figure Generation Dispatcher

```python
def generate_all_figures(master: pd.DataFrame, outputs_dir: str, figures_dir: Path):
    internal = master[master["experiment_type"] == "internal"]
    external_uci = master[master["experiment_type"] == "external_uci"]
    external_kaggle = master[master["experiment_type"] == "external_kaggle_uci"]

    # F1: Internal ROC curves
    for site in internal["test_site"].unique():
        plot_roc_curves_by_model(
            site, outputs_dir, figures_dir / f"roc_internal_{site}.pdf"
        )

    # F3: AUC heatmaps
    for model in master["model"].unique():
        plot_auc_heatmap(master, model, figures_dir)

    # F4: AUC drop bars
    plot_auc_drop_bars(master, figures_dir)

    # F6–F8: Reliability diagrams
    plot_all_reliability_diagrams(outputs_dir, figures_dir)

    # F9: ECE comparison
    plot_ece_comparison(outputs_dir, figures_dir)

    # F11: PSI heatmap
    plot_psi_heatmap(outputs_dir, figures_dir)

    # F13: Shift vs performance
    plot_shift_vs_performance(
        load_shift_performance_table(outputs_dir), figures_dir
    )

    # F16: CFS penalty
    plot_cfs_penalty(master, figures_dir)
```

### 7.3  Evaluation Config

```yaml
# configs/evaluation.yaml

reports_dir: reports
figures_dir: reports/figures
tables_dir: reports/tables

figure_format: [pdf, png]
figure_dpi: 300

significance_alpha: 0.05
bootstrap_B_comparison: 2000

color_palette:
  lr: "#1f77b4"
  rf: "#2ca02c"
  xgb: "#ff7f0e"
  lgbm: "#d62728"

site_display_names:
  kaggle: "Kaggle CVD"
  cleveland: "Cleveland"
  hungary: "Hungary"
  va: "VA Long Beach"
  switzerland: "Switzerland"

heatmap_cmap: "RdYlGn"
heatmap_vmin: 0.5
heatmap_vmax: 1.0
```

---

## 8  Output Artifacts

### 8.1  Tables (`reports/tables/`)

| Artifact | Format | Content |
|---|---|---|
| `master_results.csv` / `.parquet` | CSV + Parquet | One row per experiment × model, all metrics |
| `internal_baseline.csv` | CSV | Pivot: site × model → AUC [CI] |
| `external_auc_matrix.csv` | CSV | Pivot: train × test → AUC per model |
| `auc_delta_matrix.csv` | CSV | External − internal AUC per pair × model |
| `calibration_summary.csv` | CSV | Brier, ECE, MCE per experiment × model |
| `recalibration_comparison.csv` | CSV | ECE before/after per method × experiment |
| `shift_summary.csv` | CSV | Mean PSI, prevalence diff, C2ST per pair |
| `shift_correlations.csv` | CSV | Spearman ρ and p-values |
| `cfs_penalty.csv` | CSV | Full vs CFS AUC per site × model |
| `statistical_tests.json` | JSON | All hypothesis test results |
| `rq_summaries.json` | JSON | Per-RQ key findings |

### 8.2  Figures (`reports/figures/`)

| File Pattern | Count | Content |
|---|---|---|
| `roc_internal_{site}.pdf/png` | 5 | Internal ROC curves per site |
| `roc_external_{test_site}.pdf/png` | 5 | External vs internal ROC overlay |
| `auc_heatmap_{model}.pdf/png` | 4 | Train×test AUC grids |
| `auc_drop_bars.pdf/png` | 1 | AUC drop summary |
| `pr_internal_{site}.pdf/png` | 5 | Internal PR curves |
| `reliability_{experiment_id}.pdf/png` | ~30+ | Calibration curves |
| `recalibration_before_after_{id}.pdf/png` | ~30+ | Before/after calibration |
| `ece_comparison.pdf/png` | 1 | ECE bar chart |
| `brier_decomposition.pdf/png` | 1 | Stacked Brier bars |
| `psi_heatmap.pdf/png` | 1 | Feature × pair PSI |
| `feature_distributions_{pair}.pdf/png` | ~5 | KDE overlays for worst pairs |
| `shift_vs_performance.pdf/png` | 1 | 3-panel scatter |
| `missingness_heatmap.pdf/png` | 1 | Feature × site missing % |
| `c2st_vs_auc.pdf/png` | 1 | C2ST scatter |
| `cfs_penalty_bars.pdf/png` | 1 | CFS penalty summary |
| `size_matched_boxplot.pdf/png` | 1 | Subsampled AUC distributions |
| `effective_cfs_features.pdf/png` | 1 | Feature count per pair |

### 8.3  Report (`reports/`)

| Artifact | Format |
|---|---|
| `evaluation_report.md` | Markdown (primary) |
| `evaluation_report.html` | HTML export |
| `probast_assessment.md` | PROBAST+AI self-assessment |

---

## 9  Evaluation Checklist

Pre-flight checks before finalizing results:

| # | Check | Status |
|---|---|---|
| 1 | All internal experiments (5 sites × 4 models = 20) have `results.json` | ☐ |
| 2 | All pairwise external experiments (12 pairs × 4 models = 48) completed | ☐ |
| 3 | All LOSO experiments (4 × 4 = 16) completed | ☐ |
| 4 | Kaggle↔UCI experiments (≥8 × 4 = 32) completed | ☐ |
| 5 | Shift diagnostics exist for every external pair | ☐ |
| 6 | Recalibration results for all 3 methods × all external experiments | ☐ |
| 7 | Bootstrap CIs present for all metrics | ☐ |
| 8 | No NaN in master results table (except fields that legitimately don't apply) | ☐ |
| 9 | All figures render without error | ☐ |
| 10 | Config hash in results matches current config | ☐ |
| 11 | Random seed consistent across all experiments (42) | ☐ |
| 12 | TRIPOD+AI items covered in report | ☐ |

---

## 10  Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Bootstrap CIs overlap heavily for small UCI cohorts | Can't declare meaningful differences | Report overlap explicitly; use language like "trending" vs "significant" |
| Multiple comparisons inflate false positives | Spurious "significant" findings | Apply Bonferroni or FDR correction for shift–performance correlations; report both raw and adjusted p-values |
| Recalibration evaluated on same data used to fit it | Overly optimistic calibration improvement | Enforce calibration/evaluation split (§2.2.2 in pipeline_plan); report calibration set size |
| Figures cluttered with 4 models × 5 sites | Unreadable plots | Use faceted subplots; highlight best/worst only in summary figures; full detail in appendix |
| Shift diagnostics are univariate → miss interactions | Underestimate real shift | C2ST captures multivariate shift; note univariate limitation in discussion |
| Master table grows large (100+ experiments × many metrics) | Hard to interpret | Provide both full table (appendix) and curated pivots (main body) |

---

## 11  Source Module Map

| Module | Responsibility |
|---|---|
| `src/evaluation.py` | Master table aggregation, pivot generation, figure dispatching, report assembly |
| `src/metrics.py` | Metric computation (shared with pipeline — already exists) |
| `src/plotting.py` | All matplotlib/seaborn figure functions |
| `configs/evaluation.yaml` | Evaluation settings (color palette, thresholds, output dirs) |

---

## 12  Relationship to Other Plans

| Document | What it provides to eval | What eval provides back |
|---|---|---|
| `pipeline_plan.md` | Raw predictions, metrics, shift diagnostics, calibration artifacts | — (eval is downstream consumer) |
| `pipeline_outputs.md` | Artifact inventory and paths | — |
| `data_ingestion.md` | Cleaned data, ingestion report (record counts, missingness) | Missingness heatmap uses `ingestion_report.json` |
| `README.md` | Research questions, study design, deliverables | Evaluation report is the primary deliverable |
