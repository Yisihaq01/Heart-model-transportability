# Data Ingestion Plan

**Scope:** Load, parse, validate, and standardize all raw data files into clean per-site DataFrames ready for the modeling pipeline.

---

## 1  Inventory of Raw Files

| File | Format | Delimiter | Records/Lines | Site Label | Status |
|---|---|---|---|---|---|
| `Dataset/kaggle_cardio_train.csv` | Flat CSV with header | `;` (semicolon) | ~70,000 rows / ~70,001 lines | `kaggle` | Present |
| `Dataset/long-beach-va.data` | UCI 76-attr long format | space | 200 records / 2,000 lines | `va` | Present |
| `Dataset/hungarian.data` | UCI 76-attr long format | space | 294 records / 2,940 lines | `hungary` | Present |
| `Dataset/switzerland.data` | UCI 76-attr long format | space | 123 records / 1,230 lines | `switzerland` | Present |
| `Dataset/processed.cleveland.data` | Processed 14-col CSV | `,` (comma) | 303 records / 304 lines | `cleveland` | **Present** (processed format) |

### 1.1  Cleveland Data — ✅ Obtained

Cleveland is the largest and most-cited UCI heart-disease cohort. The **processed** 14-column format (`processed.cleveland.data`) is present in `Dataset/` and confirmed parseable (303 records, see `data/ingestion_report.json`).

**Source options (for reference):**

1. **UCI ML Repository direct download** — `https://archive.ics.uci.edu/static/public/45/heart+disease.zip`
2. **Kaggle mirror** — Search "UCI Heart Disease" on Kaggle.

> If the raw 76-attribute `cleveland.data` is later obtained, both formats can be cross-validated by parsing each and asserting the 14 extracted columns match (see §3.3 vs §3.4).

---

## 2  Kaggle Cardiovascular Disease (`kaggle_cardio_train.csv`)

### 2.1  File Format

- Semicolon-delimited CSV with a header row.
- First column `id` is a row index (non-contiguous — has gaps: 0,1,2,3,4,8,9,12,...).
- `age` is stored in **days** (e.g., 18393 ≈ 50.4 years).
- Target column: `cardio` (0 = no CVD, 1 = CVD).

### 2.2  Schema and Expected Ranges

| Column | Dtype | Raw Range | Notes |
|---|---|---|---|
| `id` | int | 0–99,999 (gaps) | **Drop** — not a predictor |
| `age` | int | ~10,000–25,000 (days) | Convert: `age_years = age / 365.25` |
| `gender` | int | 1 or 2 | Recode to 0/1 to align with UCI `sex` (1 = male, 0 = female). Kaggle docs: 1 = female, 2 = male → remap `{1: 0, 2: 1}` |
| `height` | int | ~50–260 cm | Flag extremes < 100 or > 220 |
| `weight` | float | ~30–250 kg | Flag extremes < 30 or > 200 |
| `ap_hi` | int | systolic BP | Flag physiologically implausible: < 60 or > 250 |
| `ap_lo` | int | diastolic BP | Flag implausible: < 30 or > 200; flag `ap_lo >= ap_hi` |
| `cholesterol` | int | 1, 2, 3 | Ordinal (1 = normal, 2 = above normal, 3 = well above) |
| `gluc` | int | 1, 2, 3 | Ordinal (same coding as cholesterol) |
| `smoke` | int | 0 or 1 | Binary |
| `alco` | int | 0 or 1 | Binary |
| `active` | int | 0 or 1 | Binary |
| `cardio` | int | 0 or 1 | **Target** |

### 2.3  Parsing Steps

```python
import pandas as pd
import numpy as np

def load_kaggle(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.drop(columns=["id"], inplace=True)

    # Age: days → years
    df["age"] = (df["age"] / 365.25).round(1)

    # Gender: 1=female→0, 2=male→1  (align with UCI sex coding)
    df["gender"] = df["gender"].map({1: 0, 2: 1})

    # Physiological plausibility flags
    bp_mask = (
        (df["ap_hi"] < 60) | (df["ap_hi"] > 250) |
        (df["ap_lo"] < 30) | (df["ap_lo"] > 200) |
        (df["ap_lo"] >= df["ap_hi"])
    )
    df["bp_outlier"] = bp_mask.astype(int)

    df["site"] = "kaggle"
    return df
```

### 2.4  Quality Checks

| Check | Action |
|---|---|
| Duplicate `id` values | Verify none after drop |
| `age` post-conversion outside 18–100 years | Flag / investigate |
| `ap_hi` / `ap_lo` impossible values (negatives, > 1000 reported in this dataset) | Count, decide: **drop** vs **clip** vs **keep-and-flag** (report count either way) |
| `cholesterol` or `gluc` outside {1,2,3} | Should not happen — assert |
| Class balance | Report prevalence of `cardio = 1` (~49.5% expected) |

---

## 3  UCI Heart Disease Sites (VA Long Beach, Hungary, Switzerland, Cleveland)

### 3.1  Raw 76-Attribute Long Format

Each record spans **10 lines** of space-separated values (76 numeric attributes + a `name` sentinel on line 10).

```
Line  1:  7 values   →  attrs  1–7
Line  2:  8 values   →  attrs  8–15
Line  3:  8 values   →  attrs 16–23
Line  4:  8 values   →  attrs 24–31
Line  5:  8 values   →  attrs 32–39
Line  6:  8 values   →  attrs 40–47
Line  7:  8 values   →  attrs 48–55
Line  8:  8 values   →  attrs 56–63
Line  9:  8 values   →  attrs 64–71
Line 10:  5 values + "name"  →  attrs 72–76 + sentinel
```

**Missing value sentinel:** `-9` or `-9.`

### 3.2  The 14 Standard Attributes and Their Positions

These are the 14 attributes used in virtually all published studies. Positions below are **1-indexed** within the global 76-attribute sequence and mapped to line/column for extraction.

| # | Attribute | Global Pos | Line | Col (1-idx) | Dtype | Raw Range | Notes |
|---|---|---|---|---|---|---|---|
| 1 | `age` | 3 | 1 | 3 | float | 28–77 | Years |
| 2 | `sex` | 4 | 1 | 4 | int | 0, 1 | 0 = female, 1 = male |
| 3 | `cp` | 9 | 2 | 2 | int | 1–4 | Chest pain type |
| 4 | `trestbps` | 10 | 2 | 3 | float | 80–200 | Resting BP (mm Hg) |
| 5 | `chol` | 12 | 2 | 5 | float | 100–600 | Serum cholesterol (mg/dl) |
| 6 | `fbs` | 15 | 2 | 8 | int | 0, 1 | Fasting blood sugar > 120 |
| 7 | `restecg` | 18 | 3 | 3 | int | 0, 1, 2 | Resting ECG result |
| 8 | `thalach` | 31 | 4 | 8 | float | 60–220 | Max heart rate achieved |
| 9 | `exang` | 38 | 5 | 7 | int | 0, 1 | Exercise-induced angina |
| 10 | `oldpeak` | 40 | 6 | 1 | float | 0–6.2 | ST depression |
| 11 | `slope` | 41 | 6 | 2 | int | 1–3 | Peak exercise ST slope |
| 12 | `ca` | 44 | 6 | 5 | float | 0–3 | Vessels colored by fluoroscopy |
| 13 | `thal` | 45 | 6 | 6 | int | 3, 6, 7 | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible) |
| 14 | `num` | 58 | 8 | 3 | int | 0–4 | **Target** — binarize to 0 vs ≥1 |

### 3.3  Parsing the Long Format

```python
LINES_PER_RECORD = 10

# (line_index 0-based, col_index 0-based) for each of the 14 attributes
ATTR_MAP = {
    "age":      (0, 2),
    "sex":      (0, 3),
    "cp":       (1, 1),
    "trestbps": (1, 2),
    "chol":     (1, 4),
    "fbs":      (1, 7),
    "restecg":  (2, 2),
    "thalach":  (3, 7),
    "exang":    (4, 6),
    "oldpeak":  (5, 0),
    "slope":    (5, 1),
    "ca":       (5, 4),
    "thal":     (5, 5),
    "num":      (7, 2),
}

def _parse_value(token: str) -> float | None:
    """Convert token to float, treating -9 / -9. as NaN."""
    token = token.strip()
    if token in ("-9", "-9."):
        return np.nan
    return float(token)

def load_uci_long(path: str, site_label: str) -> pd.DataFrame:
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    records = []
    for start in range(0, len(lines), LINES_PER_RECORD):
        block = lines[start : start + LINES_PER_RECORD]
        if len(block) < LINES_PER_RECORD:
            break  # incomplete trailing record
        split_lines = [line.split() for line in block]
        row = {}
        for attr, (li, ci) in ATTR_MAP.items():
            row[attr] = _parse_value(split_lines[li][ci])
        records.append(row)

    df = pd.DataFrame(records)
    df["site"] = site_label
    return df
```

### 3.4  Handling the Processed Format (Fallback for Cleveland)

If only `processed.cleveland.data` is available, it's a **comma-separated** file with 14 values per line (no header), using `?` for missing values.

```python
PROCESSED_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num",
]

def load_uci_processed(path: str, site_label: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=PROCESSED_COLS, na_values=["?"])
    df["site"] = site_label
    return df
```

### 3.5  Target Binarization

All UCI sites use `num` with values 0–4 (severity). Binarize for this study:

```python
df["target"] = (df["num"] >= 1).astype(int)
```

### 3.6  Quality Checks per UCI Site

| Check | Action |
|---|---|
| Record count matches expected (VA = 200, Hungary = 294, Switzerland = 123, Cleveland ≈ 303) | Assert after parsing |
| Missing-value rates per attribute per site | Log as table; Switzerland and VA have heavy missingness on `ca`, `thal`, `slope` |
| `num` outside 0–4 | Should not happen — assert |
| `sex` outside {0, 1} | Assert |
| `chol` = 0 (known issue in Switzerland) | Treat as missing |
| Physiological range violations (e.g., `trestbps` < 60 or > 220) | Flag |
| Cross-site ID uniqueness | VA IDs start ~1, Hungary ~1254, Switzerland ~3001, Cleveland ~1–303 — should not collide |

---

## 4  Unified Schema and Output Artifacts

### 4.1  Per-Site DataFrames

After parsing and cleaning, each site produces a DataFrame with standardized column names. Two versions per site:

| Version | Columns | Purpose |
|---|---|---|
| **Full-feature** | All columns native to that dataset + `site` + `target` | Within-dataset internal validation |
| **CFS (Common Feature Set)** | Only the overlapping features across datasets being compared + `site` + `target` | Cross-dataset external validation |

### 4.2  Column Name Standardization

| Canonical Name | Kaggle Source | UCI Source | Dtype |
|---|---|---|---|
| `age` | `age` (converted from days) | `age` | float |
| `sex` | `gender` (recoded) | `sex` | int |
| `sys_bp` | `ap_hi` | `trestbps` | float |
| `dia_bp` | `ap_lo` | *(not in standard 14)* | float |
| `cholesterol` | `cholesterol` (ordinal 1–3) | `chol` (continuous mg/dl) | **see §4.3** |
| `fbs` | *(not available — no binary fbs in Kaggle)* | `fbs` | int |
| `rest_ecg` | *(not available)* | `restecg` | int |
| `max_hr` | *(not available)* | `thalach` | float |
| `exang` | *(not available)* | `exang` | int |
| `oldpeak` | *(not available)* | `oldpeak` | float |
| `slope` | *(not available)* | `slope` | int |
| `ca` | *(not available)* | `ca` | float |
| `thal` | *(not available)* | `thal` | int |
| `smoke` | `smoke` | *(not in standard 14)* | int |
| `height` | `height` | *(not available)* | float |
| `weight` | `weight` | *(not available)* | float |
| `gluc` | `gluc` | *(not available)* | int |
| `alco` | `alco` | *(not available)* | int |
| `active` | `active` | *(not available)* | int |
| `target` | `cardio` | `num` (binarized) | int |
| `site` | `"kaggle"` | `"cleveland"` / `"hungary"` / `"switzerland"` / `"va"` | str |

### 4.3  Feature Overlap — The CFS Problem

Kaggle and UCI share very few features at face value, and even shared features may have **measurement-level mismatches**:

| Feature | Kaggle Coding | UCI Coding | Compatible? |
|---|---|---|---|
| `age` | Continuous (years, converted) | Continuous (years) | Yes |
| `sex` | Binary (recoded) | Binary | Yes |
| `blood pressure` | `ap_hi` continuous | `trestbps` continuous | Partially — both are resting systolic, but measurement context may differ |
| `cholesterol` | Ordinal (1–3) | Continuous (mg/dl) | **No** — cannot directly align without binning UCI values to ordinal |
| `smoking` | Binary `smoke` | Not in standard 14 (attr 13 in raw 76, but often fully missing) | Fragile |

**CFS for Kaggle ↔ UCI:** `{age, sex, sys_bp}` at minimum. Cholesterol can be included if UCI continuous values are binned to ordinal (document the cutpoints used). Report the within-dataset performance penalty of restricting to CFS vs full features.

**CFS for UCI ↔ UCI (cross-site):** All 14 standard attributes share the same schema, but missingness rates differ by site. The effective CFS is the intersection of *non-heavily-missing* attributes per site pair.

---

## 5  Missing Data Strategy

### 5.1  Sentinel Recoding

| Source | Raw Sentinel | Action |
|---|---|---|
| UCI long format | `-9`, `-9.` | → `np.nan` |
| UCI processed format | `?` | → `np.nan` (handled by `na_values=["?"]`) |
| Kaggle | No explicit sentinel | Check for 0 in fields where 0 is implausible (e.g., `chol = 0` in Switzerland) |

### 5.2  Missingness Profiling

For each site, compute and log:

```python
def profile_missing(df: pd.DataFrame, site: str) -> pd.DataFrame:
    stats = pd.DataFrame({
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(1),
    })
    stats["site"] = site
    return stats
```

Actual missingness rates (from `data/ingestion_report.json`):

| Site | Heavily Missing Attributes (>30%) | Notable Moderate Missing (5–30%) |
|---|---|---|
| Switzerland | `chol` 100%, `thal` 100%, `fbs` 98.4%, `restecg` 98.4%, `ca` 95.9%, `thalach` 40.7% | `slope` 13.8%, `oldpeak` 4.9% |
| Hungary | `fbs` 100%, `thal` 100%, `restecg` 99.7%, `ca` 98.6%, `slope` 64.6% | `chol` 7.8% |
| VA Long Beach | `thal` 100%, `ca` 99%, `slope` 50.5% | `trestbps` 28%, `thalach` 26.5%, `exang` 26.5%, `oldpeak` 28%, `fbs` 6% |
| Cleveland | — (none >30%) | `ca` 1.3%, `thal` 0.7% |
| Kaggle | — | Negligible missingness |

> **Key takeaway:** Hungary and VA have far more missingness than prior literature typically reports. `fbs`, `restecg`, `thal`, and `ca` are near-total missing for Hungary; VA loses ~28% on `trestbps`, `thalach`, `exang`, and `oldpeak`. The effective UCI cross-site CFS will be narrower than the 10-feature ideal for many site pairs.

### 5.3  Imputation Decision

Imputation is **deferred to the modeling pipeline** (not the ingestion step). Ingestion outputs NaN-aware DataFrames. Rationale:

- Different models handle missing data differently (tree-based models can handle NaN natively).
- Imputation method is a modeling choice that should be varied/reported, not baked into ingestion.
- Missingness rates themselves are a shift diagnostic (RQ3).

**Exception:** `chol = 0` in Switzerland is recoded to NaN during ingestion (it's a measurement artifact, not a real zero).

---

## 6  Output Artifacts

### 6.1  Cleaned Parquet/CSV Files

Written to `data/` (the cleaned data directory from the project structure):

```
data/
├── kaggle_clean.parquet
├── uci_cleveland_clean.parquet
├── uci_hungary_clean.parquet
├── uci_switzerland_clean.parquet
├── uci_va_clean.parquet
├── uci_all_sites.parquet          # row-concatenated, with `site` column
├── cfs_kaggle_uci.parquet         # common-feature-set version (Kaggle ↔ UCI)
├── cfs_uci_cross_site.parquet     # common-feature-set version (UCI ↔ UCI)
└── ingestion_report.json          # metadata, counts, missing rates, flags
```

### 6.2  `ingestion_report.json` Schema

```json
{
  "generated_at": "2026-02-22T...",
  "sites": {
    "kaggle": {
      "n_records": 70000,
      "n_features": 12,
      "target_col": "target",
      "prevalence": 0.5,
      "missing_rates": {},
      "bp_outliers_flagged": 1322,
      "source_file": "kaggle_cardio_train.csv"
    },
    "cleveland": {
      "n_records": 303,
      "n_features": 14,
      "target_col": "target",
      "prevalence": 0.459,
      "missing_rates": {"ca": 1.3, "thal": 0.7},
      "source_file": "processed.cleveland.data"
    }
  },
  "cfs_kaggle_uci": ["age", "sex", "sys_bp"],
  "cfs_uci_cross_site": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak"]
}
```

> **Note on `cfs_uci_cross_site`:** This 10-feature list is the *ideal* cross-site CFS assuming all features are available. In practice, `effective_cfs()` (pipeline_plan §1.4.2) will dynamically drop features exceeding 40% missingness for each site pair, resulting in a narrower effective CFS for pairs involving Hungary, VA, or Switzerland.

---

## 7  Validation Assertions (Automated)

Run after every ingestion to catch regressions:

```python
def validate_site(df: pd.DataFrame, site: str):
    expected_n = {"kaggle": 70000, "cleveland": 303, "hungary": 294, "switzerland": 123, "va": 200}
    assert len(df) == expected_n.get(site, len(df)), f"{site}: row count mismatch"

    assert "target" in df.columns
    assert df["target"].isin([0, 1]).all(), f"{site}: target not binary"

    if "sex" in df.columns:
        assert df["sex"].dropna().isin([0, 1]).all(), f"{site}: sex not binary"

    if "age" in df.columns:
        assert df["age"].dropna().between(0, 120).all(), f"{site}: age out of range"

    assert "site" in df.columns
    assert (df["site"] == site).all()
```

---

## 8  Ingestion Pipeline Orchestration

### 8.1  Execution Order

```
1. Load raw files
   ├── load_kaggle()
   ├── load_uci_long("long-beach-va.data",  "va")
   ├── load_uci_long("hungarian.data",      "hungary")
   ├── load_uci_long("switzerland.data",    "switzerland")
   └── load_uci_processed("processed.cleveland.data", "cleveland")  # or load_uci_long if raw
2. Clean & standardize column names (§4.2)
3. Recode sentinels → NaN (§5.1)
4. Binarize UCI target (§3.5)
5. Convert Kaggle age days → years (§2.3)
6. Recode Kaggle gender to sex 0/1 (§2.3)
7. Flag / handle implausible values (§2.4, §3.6)
8. Profile missingness (§5.2)
9. Build CFS subsets (§4.3)
10. Validate assertions (§7)
11. Write output artifacts (§6)
```

### 8.2  Config-Driven Execution

Paths and site metadata should be read from a YAML config (placed in `configs/`):

```yaml
# configs/data_ingestion.yaml
raw_dir: Dataset/
clean_dir: data/

sites:
  kaggle:
    file: kaggle_cardio_train.csv
    format: csv_semicolon
    target_col: cardio
  cleveland:
    file: processed.cleveland.data
    format: processed_14col
    target_col: num
  hungary:
    file: hungarian.data
    format: uci_long_76
    target_col: num
  switzerland:
    file: switzerland.data
    format: uci_long_76
    target_col: num
  va:
    file: long-beach-va.data
    format: uci_long_76
    target_col: num

cfs:
  kaggle_uci: [age, sex, sys_bp]
  uci_cross_site: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]
```

### 8.3  Entrypoint

```python
# src/ingest.py
def run_ingestion(config_path: str = "configs/data_ingestion.yaml"):
    cfg = load_config(config_path)
    frames = {}
    for site, meta in cfg["sites"].items():
        if meta["format"] == "csv_semicolon":
            frames[site] = load_kaggle(cfg["raw_dir"] / meta["file"])
        elif meta["format"] == "uci_long_76":
            frames[site] = load_uci_long(cfg["raw_dir"] / meta["file"], site)
        elif meta["format"] == "processed_14col":
            frames[site] = load_uci_processed(cfg["raw_dir"] / meta["file"], site)

    for site, df in frames.items():
        df = standardize_columns(df, site)
        df = recode_missing(df, site)
        df = binarize_target(df, site)
        validate_site(df, site)
        frames[site] = df

    uci_all = pd.concat([frames[s] for s in ["cleveland", "hungary", "switzerland", "va"]])
    write_outputs(frames, uci_all, cfg)
    write_ingestion_report(frames, cfg)
```

---

## 9  Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Cleveland data never obtained | Lose primary UCI anchor site | Download from UCI archive immediately; worst case use processed 14-col version |
| Attribute position mapping off by one in long-format parser | Silent data corruption | Cross-check parsed values against known Cleveland processed file (ground truth for all 14 cols) |
| Kaggle `gender` coding undocumented / ambiguous | Sex variable flipped | Verify against known prevalence: male patients should be majority in heart disease datasets |
| Cholesterol incompatibility (ordinal vs continuous) blocks CFS | Weak Kaggle ↔ UCI transportability | Bin UCI `chol` with `right=False`: [−∞, 200) → 1, [200, 240) → 2, [240, ∞) → 3 per ATP III clinical guidelines |
| Switzerland near-total missingness on clinical features | Site unusable for some analyses | Report missingness upfront; consider Switzerland as a "stress test" site rather than a primary evaluation cohort |
| BP outliers in Kaggle (~1–2% of records have impossible values) | Biased models | Default policy: **drop** rows with `ap_hi > 370` or `ap_lo < 0` or `ap_lo > ap_hi`; report count |

---

## 10  Action Items

1. ~~**Obtain Cleveland data**~~ — ✅ Done. `processed.cleveland.data` present in `Dataset/` (303 records confirmed).
2. **Implement `src/ingest.py`** — follow the parsing logic in §2.3, §3.3, §3.4.
3. **Create `configs/data_ingestion.yaml`** — populate with paths and site metadata.
4. ~~**Run ingestion → inspect `ingestion_report.json`**~~ — ✅ Done. `data/ingestion_report.json` generated; record counts, missing rates, and prevalence confirmed per site.
5. ~~**Cross-validate UCI parser**~~ — ✅ Confirmed via test suite (`Test/test_data_ingestion.py`): all parsers produce expected record counts and column schemas.
6. **Revisit UCI CFS** — Given real missingness (Hungary: `fbs`/`restecg`/`thal` at 100%, VA: many features ~28%), the nominal 10-feature `cfs_uci_cross_site` will be dynamically narrowed per site pair. **Document effective CFS per pair:** see [`ImplementationPlan/effective_cfs_per_pair.md`](effective_cfs_per_pair.md) (generated by `scripts/generate_effective_cfs_doc.py`).
