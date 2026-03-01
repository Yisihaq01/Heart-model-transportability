"""
Stage C.3 — Artifact serialization.
Persist results, predictions, fitted models, and pipelines to outputs/.
Stage C.4: every experiment log includes timestamp; optional config_hash when provided.

Canonical artifact set per experiment (internal, internal_cfs, external_uci, external_kaggle_uci):
  - results.json          (required) — metrics, params, CIs; excludes predictions/fitted objects
  - predictions.parquet    (canonical, when predictions present)
  - predictions.csv        (optional convenience mirror of parquet)
  - model.joblib           (when fitted_model present)
  - pipeline.joblib        (when fitted_pipeline present)
"""
from __future__ import annotations

import json
import joblib
from pathlib import Path

import pandas as pd

from . import reproducibility


def save_experiment(
    result: dict,
    base_dir: str | Path = "outputs",
    config_hash: str | None = None,
) -> Path:
    """
    Persist experiment to outputs/. Writes:
      - results.json (all keys except predictions, fitted_model, fitted_pipeline; C.4: timestamp, optional config_hash)
      - predictions.parquet (canonical) and predictions.csv (optional convenience) when predictions present
      - model.joblib and pipeline.joblib when fitted_model/fitted_pipeline provided
    Applies to internal, internal_cfs, external_uci, external_kaggle_uci experiment types.
    """
    base_dir = Path(base_dir)
    exp_type = result["experiment_type"]
    variant = result.get("variant")

    if exp_type == "internal":
        path = base_dir / "internal" / result["site"] / result["model"]
    elif exp_type == "internal_cfs":
        path = base_dir / "internal_cfs" / (variant or "cfs") / result["site"] / result["model"]
    elif exp_type == "external_uci":
        train_sites = result["train_sites"]
        train_key = "+".join(sorted(train_sites)) if isinstance(train_sites, list) else train_sites
        path = base_dir / "external_uci" / f"{train_key}__to__{result['test_site']}" / result["model"]
    elif exp_type == "external_kaggle_uci":
        path = base_dir / "external_kaggle_uci" / (variant or "cfs") / f"{result['train_site']}__to__{result['test_site']}" / result["model"]
    else:
        path = base_dir / exp_type / result.get("id", "default")

    path.mkdir(parents=True, exist_ok=True)

    # JSON: all except predictions and non-serializable fitted objects
    payload = {k: v for k, v in result.items() if k not in ("predictions", "fitted_model", "fitted_pipeline")}
    # C.4 reproducibility: timestamp always; config_hash when provided
    payload["timestamp"] = reproducibility.experiment_timestamp()
    if config_hash is not None:
        payload["config_hash"] = config_hash
    with open(path / "results.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    if "predictions" in result:
        df_pred = pd.DataFrame(result["predictions"])
        df_pred.to_parquet(path / "predictions.parquet", index=False)
        df_pred.to_csv(path / "predictions.csv", index=False, float_format="%.10g")

    if exp_type in ("internal", "internal_cfs"):
        if "fitted_model" in result:
            joblib.dump(result["fitted_model"], path / "model.joblib")
        if "fitted_pipeline" in result:
            joblib.dump(result["fitted_pipeline"], path / "pipeline.joblib")
    elif exp_type in ("external_uci", "external_kaggle_uci"):
        if "fitted_model" in result:
            joblib.dump(result["fitted_model"], path / "model.joblib")
        if "fitted_pipeline" in result:
            joblib.dump(result["fitted_pipeline"], path / "pipeline.joblib")

    return path
