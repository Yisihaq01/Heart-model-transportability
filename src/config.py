"""
Stage C.1 — Experiment configuration.
Load pipeline YAML and expose for all stages.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "configs/pipeline.yaml") -> dict[str, Any]:
    """
    Load pipeline config from YAML. Path can be relative to cwd or absolute.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_bootstrap_iters(n_samples: int, config: dict[str, Any]) -> int:
    """Return bootstrap iterations: bootstrap_iters_small when N < 200, else bootstrap_iters."""
    if n_samples < 200:
        return config.get("bootstrap_iters_small", 500)
    return config.get("bootstrap_iters", 200)
