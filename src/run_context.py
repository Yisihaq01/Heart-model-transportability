"""
Canonical run namespace and manifest for reproducibility.
Defines run_id (timestamp + config_hash), code_version, and artifact index.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import reproducibility


def run_id(config_hash_short: str) -> str:
    """
    Canonical run identifier: {timestamp}_{config_hash}.
    Example: 20250301T143022_a1b2c3d4e5f6
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{ts}_{config_hash_short}"


def code_version(root: Path | None = None) -> str | None:
    """
    Git commit hash if in a repo; else None.
    """
    root = root or Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()[:12]
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def discover_artifacts(run_root: Path) -> list[dict[str, Any]]:
    """
    Scan run_root for generated artifacts; return index entries.
    Naming: internal/{site}/{model}/, external_uci/{pair}/, external_kaggle_uci/{variant}/{pair}/,
            internal_cfs/{variant}/{site}/{model}/, calibration/*/, shift/*/, size_matched/*/
    """
    entries: list[dict[str, Any]] = []

    def _add(rel_path: str, artifact_type: str, **meta: Any) -> None:
        p = run_root / rel_path
        if p.exists():
            entries.append({"path": rel_path, "type": artifact_type, **meta})

    # internal
    internal = run_root / "internal"
    if internal.exists():
        for site_dir in internal.iterdir():
            if site_dir.is_dir():
                for model_dir in site_dir.iterdir():
                    if model_dir.is_dir():
                        _add(f"internal/{site_dir.name}/{model_dir.name}/results.json", "internal", site=site_dir.name, model=model_dir.name)

    # internal_cfs
    icfs = run_root / "internal_cfs"
    if icfs.exists():
        for v in icfs.iterdir():
            if v.is_dir():
                for site_dir in v.iterdir():
                    if site_dir.is_dir():
                        for model_dir in site_dir.iterdir():
                            if model_dir.is_dir():
                                _add(f"internal_cfs/{v.name}/{site_dir.name}/{model_dir.name}/results.json", "internal_cfs", variant=v.name, site=site_dir.name, model=model_dir.name)

    # external_uci
    ext_uci = run_root / "external_uci"
    if ext_uci.exists():
        for pair_dir in ext_uci.iterdir():
            if pair_dir.is_dir():
                for model_dir in pair_dir.iterdir():
                    if model_dir.is_dir():
                        _add(f"external_uci/{pair_dir.name}/{model_dir.name}/results.json", "external_uci", pair=pair_dir.name, model=model_dir.name)

    # external_kaggle_uci
    ext_kg = run_root / "external_kaggle_uci"
    if ext_kg.exists():
        for v in ext_kg.iterdir():
            if v.is_dir():
                for pair_dir in v.iterdir():
                    if pair_dir.is_dir():
                        for model_dir in pair_dir.iterdir():
                            if model_dir.is_dir():
                                _add(f"external_kaggle_uci/{v.name}/{pair_dir.name}/{model_dir.name}/results.json", "external_kaggle_uci", variant=v.name, pair=pair_dir.name, model=model_dir.name)

    # calibration
    cal = run_root / "calibration"
    if cal.exists():
        for sub in ["before", "recalibration", "updating"]:
            subdir = cal / sub
            if subdir.exists():
                for f in subdir.rglob("*.json"):
                    rel = f.relative_to(run_root)
                    _add(str(rel), "calibration", sub=sub)

    # shift
    shift = run_root / "shift"
    if shift.exists():
        for pair_dir in shift.iterdir():
            if pair_dir.is_dir():
                _add(f"shift/{pair_dir.name}/shift_diagnostics.json", "shift", pair=pair_dir.name)
        for name in ["shift_table.parquet", "cross_experiment_index.json"]:
            _add(f"shift/{name}", "shift_global")

    # size_matched
    sm = run_root / "size_matched"
    if sm.exists():
        for pair_dir in sm.iterdir():
            if pair_dir.is_dir():
                for f in pair_dir.glob("*.json"):
                    _add(str(f.relative_to(run_root)), "size_matched", pair=pair_dir.name)

    return entries


def write_manifest(
    run_root: Path,
    config_hash_short: str,
    seed: int,
    code_version_str: str | None = None,
    config_path: str | Path | None = None,
) -> Path:
    """
    Write manifest.json at run_root. Captures config_hash, seed, code_version, artifact index.
    """
    project_root = run_root.parent.parent.parent  # run_root = outputs/runs/run_id
    artifacts = discover_artifacts(run_root)

    manifest = {
        "run_id": run_root.name,
        "config_hash": config_hash_short,
        "seed": seed,
        "code_version": code_version_str or code_version(project_root),
        "timestamp": reproducibility.experiment_timestamp(),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    path = run_root / "manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path
