"""
Stage C.4 — Reproducibility controls.
Global random seed, config hashing for experiment logging, deterministic splits/models.
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> None:
    """
    Set global random seed for numpy, random, and (where applicable) RNG state.
    Call once at pipeline start; config should propagate this seed to all splits, models, bootstrap.
    """
    np.random.seed(seed)
    random.seed(seed)


def config_hash(config: dict[str, Any]) -> str:
    """
    Stable hash of pipeline config for experiment logging (reproducibility audit).
    Sorts keys so dict order does not change the hash.
    """
    # Normalize: sort keys, exclude non-serializable
    def _norm(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _norm(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):
            return [_norm(x) for x in obj]
        return obj

    blob = json.dumps(_norm(config), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def experiment_timestamp() -> str:
    """UTC ISO timestamp for experiment logging."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
