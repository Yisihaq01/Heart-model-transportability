"""
Stage 1.2 — Model Definitions.
Model registry, default params, tuning grids, and tune_model() for internal/external validation.
"""

from __future__ import annotations

from math import prod

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

MODEL_REGISTRY = {
    "lr": {
        "class": LogisticRegression,
        "default_params": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
        },
        "tuning_grid": {"C": [0.001, 0.01, 0.1, 1, 10]},
    },
    "rf": {
        "class": RandomForestClassifier,
        "default_params": {
            "n_estimators": 300,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        "tuning_grid": {
            "max_depth": [5, 10, 20, None],
            "min_samples_leaf": [1, 5, 10],
        },
    },
    "xgb": {
        "class": XGBClassifier,
        "default_params": {
            "n_estimators": 300,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        },
        "tuning_grid": {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
        },
    },
    "lgbm": {
        "class": LGBMClassifier,
        "default_params": {
            "n_estimators": 300,
            "verbose": -1,
            "random_state": 42,
            "n_jobs": -1,
        },
        "tuning_grid": {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
        },
    },
}


def get_model(model_key: str, seed: int = 42, **override_params):
    """Return an estimator instance for the given key with merged params. C.4: seed from config."""
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model_key: {model_key}. Valid: {list(MODEL_REGISTRY)}")
    entry = MODEL_REGISTRY[model_key]
    params = {**entry["default_params"], "random_state": seed, **override_params}
    return entry["class"](**params)


def tune_model(model_key: str, X_train, y_train, n_samples: int, seed: int = 42):
    """
    Inner 5-fold CV (or 5x3 repeated for n < 200) with GridSearchCV, roc_auc.
    Returns (best_estimator, best_params, cv_results). C.4: seed from config for determinism.
    """
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model_key: {model_key}. Valid: {list(MODEL_REGISTRY)}")
    entry = MODEL_REGISTRY[model_key]
    base_params = {**entry["default_params"], "random_state": seed}
    base_model = entry["class"](**base_params)

    cv = (
        RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
        if n_samples < 200
        else StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    )

    grid = entry["tuning_grid"]
    use_random = model_key in ("xgb", "lgbm")
    if use_random:
        n_candidates = int(prod(len(v) for v in grid.values())) if grid else 1
        n_iter = min(20, n_candidates)
        search = RandomizedSearchCV(
            base_model,
            param_distributions=grid,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            random_state=seed,
        )
    else:
        search = GridSearchCV(
            base_model,
            grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.cv_results_
