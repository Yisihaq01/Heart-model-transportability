"""
Stage 2.4 — Size-Matched Sensitivity Analysis.

Subsample large training sets down to approximately match target-site test size,
repeat K times, and summarize mean ± SD of external metrics to deconfound
dataset size from population shift (see ImplementationPlan/pipeline_plan.md §2.4).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from . import metrics
from . import models
from . import preprocessing
from . import validation
from . import shift as shift_mod


def _aggregate_results(metric_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate a list of metric dicts into mean ± std for scalar fields.

    Returns a dict:
      {
        "mean": {metric_name: float, ...},
        "std":  {metric_name: float, ...},
        "K": int,
      }
    """
    if not metric_list:
        return {"mean": {}, "std": {}, "K": 0}

    keys = set().union(*(m.keys() for m in metric_list))
    mean: Dict[str, Any] = {}
    std: Dict[str, Any] = {}

    for k in sorted(keys):
        vals = [m.get(k) for m in metric_list]
        # Only aggregate scalar numeric fields
        if not vals:
            continue
        if not isinstance(vals[0], (int, float, np.floating)):
            continue
        arr = np.asarray(vals, dtype=float)
        mean[k] = float(np.mean(arr))
        std[k] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    return {"mean": mean, "std": std, "K": len(metric_list)}


def size_matched_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_key: str,
    features: List[str],
    model_family: str,
    K: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Core Stage 2.4 protocol for a single (train_sites, test_site, model) triplet.

    - Subsample train_df to target_n ~= len(test_df) (no replacement).
    - Fit model with default params (no inner tuning) on each subsample.
    - Evaluate on full test_df and compute standard metrics.
    - Return mean ± SD across K repeats.
    """
    target_n = len(test_df)
    if target_n == 0 or len(train_df) <= target_n:
        return {"mean": {}, "std": {}, "K": 0}

    rng = np.random.RandomState(seed)
    feature_cols = [c for c in features if c in train_df.columns and c in test_df.columns]
    if not feature_cols:
        return {"mean": {}, "std": {}, "K": 0}

    results: List[Dict[str, Any]] = []

    for k in range(K):
        # 2.4.1 — subsample training set to match test size
        sub_idx = rng.choice(len(train_df), size=target_n, replace=False)
        sub_train = train_df.iloc[sub_idx]

        pipeline = preprocessing.build_preprocessing_pipeline(model_family, feature_cols)
        X_train_t, y_train_t = preprocessing.fit_transform_train(
            pipeline, sub_train[feature_cols], sub_train["target"]
        )
        X_test_t = preprocessing.transform_test(pipeline, test_df[feature_cols])
        X_train_t = np.asarray(X_train_t)
        X_test_t = np.asarray(X_test_t)

        # Use default-parameter model (no tuning) for efficiency; Stage 2.4 focuses on size effect.
        clf = models.get_model(model_key)
        clf.fit(X_train_t, y_train_t)

        y_prob = clf.predict_proba(X_test_t)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        y_true = test_df["target"].to_numpy()

        m = metrics.compute_metrics(y_true, y_prob, y_pred)
        results.append(m)

    return _aggregate_results(results)


def _discover_external_experiments(
    output_dir: Path,
    include_kaggle_uci: bool = True,
) -> List[shift_mod.ExternalExperiment]:
    """
    Thin wrapper around shift._discover_external_experiments so we don't duplicate
    the external experiment discovery logic.
    """
    return shift_mod._discover_external_experiments(output_dir, include_kaggle_uci=include_kaggle_uci)


def _load_clean(site: str, data_dir: Path) -> pd.DataFrame:
    """Use the same convention as validation.load_clean / shift._load_clean."""
    return validation.load_clean(site, data_dir)


def run_size_matched_sensitivity(
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    K: int = 20,
    min_ratio: float = 2.0,
    seed: int = 42,
    include_kaggle_uci: bool = True,
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """
    Stage 2.4 main entrypoint.

    For each external experiment where training N is substantially larger than test N
    (train_n >= min_ratio * test_n), run size_matched_experiment() and persist:

        outputs/size_matched/{train_sites}__to__{test_site}/{model}.json

    Returns a dict keyed by (train_key, test_site, model_key) with aggregated stats.
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    size_root = output_dir / "size_matched"
    size_root.mkdir(parents=True, exist_ok=True)

    exps = _discover_external_experiments(output_dir, include_kaggle_uci=include_kaggle_uci)
    if not exps:
        return {}

    results: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for exp in exps:
        train_sites = exp.train_sites
        test_site = exp.test_site
        model_key = exp.model
        train_key = "+".join(sorted(train_sites))

        try:
            train_dfs = [_load_clean(s, data_dir) for s in train_sites]
            train_df = pd.concat(train_dfs, ignore_index=True)
            test_df = _load_clean(test_site, data_dir)
        except FileNotFoundError:
            continue

        n_train = len(train_df)
        n_test = len(test_df)
        if n_test == 0 or n_train < max(int(min_ratio * n_test), n_test + 1):
            # Only run for experiments where train N is clearly larger than test N.
            continue

        features = [c for c in exp.features_used if c in train_df.columns and c in test_df.columns]
        if not features:
            continue

        if model_key not in validation.MODEL_FAMILY:
            continue
        model_family = validation.MODEL_FAMILY[model_key]

        agg = size_matched_experiment(
            train_df=train_df,
            test_df=test_df,
            model_key=model_key,
            features=features,
            model_family=model_family,
            K=K,
            seed=seed,
        )
        if not agg.get("mean"):
            continue

        key = (train_key, test_site, model_key)
        results[key] = {
            "train_sites": train_sites,
            "test_site": test_site,
            "model": model_key,
            "experiment_type": exp.experiment_type,
            "n_train": int(n_train),
            "n_test": int(n_test),
            "K": int(agg["K"]),
            "metrics_mean": agg["mean"],
            "metrics_std": agg["std"],
        }

        pair_dir = size_root / f"{train_key}__to__{test_site}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        out_path = pair_dir / f"{model_key}.json"
        with open(out_path, "w") as f:
            json.dump(results[key], f, indent=2, default=str)

    return results


if __name__ == "__main__":
    # CLI:
    #   python -m src.sensitivity              -> Stage 2.4 over all eligible external experiments
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "data"
    output_dir = base / "outputs"
    if not (data_dir / "ingestion_report.json").exists():
        raise SystemExit("Run ingestion first: python -m src.ingest")
    res = run_size_matched_sensitivity(data_dir=data_dir, output_dir=output_dir)
    print(
        f"Stage 2.4 size-matched sensitivity completed for {len(res)} train/test/model combinations. "
        f"Results written under {output_dir / 'size_matched'}."
    )

