#!/usr/bin/env python3
"""
Canonical pipeline run: creates run-scoped output under outputs/runs/{run_id}/,
writes manifest.json, and supports legacy mode (flat outputs/).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config, run_context, validation
from src.calibration import assess_calibration, run_lightweight_updating, run_recalibration
from src.reproducibility import config_hash, set_global_seed
from src.shift import run_shift_diagnostics
from src.sensitivity import run_size_matched_sensitivity


def run_canonical(
    data_dir: Path | None = None,
    outputs_root: Path | None = None,
    config_path: Path | None = None,
    stages: list[str] | None = None,
) -> Path:
    """
    Run pipeline in canonical mode: outputs/runs/{run_id}/.
    Returns run_root path.
    """
    data_dir = data_dir or ROOT / "data"
    outputs_root = outputs_root or ROOT / "outputs"
    config_path = config_path or ROOT / "configs" / "pipeline.yaml"
    stages = stages or ["1.3", "1.4", "1.5", "1.6", "2.1", "2.2", "2.3", "size_matched"]

    cfg = config.load_config(config_path)
    seed = int(cfg.get("random_seed", 42))
    cfg_hash = config_hash(cfg)
    rid = run_context.run_id(cfg_hash)
    run_root = outputs_root / "runs" / rid
    run_root.mkdir(parents=True, exist_ok=True)

    set_global_seed(seed)
    output_dir = run_root

    if "1.3" in stages:
        validation.run_internal_validation(data_dir=data_dir, output_dir=output_dir)
    if "1.4" in stages:
        validation.run_external_uci_matrix(data_dir=data_dir, output_dir=output_dir)
    if "1.5" in stages:
        validation.run_kaggle_uci_tests(data_dir=data_dir, output_dir=output_dir)
    if "1.6" in stages:
        run_shift_diagnostics(data_dir=data_dir, output_dir=output_dir, include_kaggle_uci=True)
    if "2.1" in stages:
        assess_calibration(outputs_dir=output_dir)
    if "2.2" in stages:
        run_recalibration(outputs_dir=output_dir)
    if "2.3" in stages:
        run_lightweight_updating(outputs_dir=output_dir)
    if "size_matched" in stages:
        run_size_matched_sensitivity(data_dir=data_dir, output_dir=output_dir)

    run_context.write_manifest(run_root, cfg_hash, seed, config_path=config_path)
    return run_root


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Run pipeline in canonical mode (outputs/runs/{run_id}/)")
    p.add_argument("--legacy", action="store_true", help="Use legacy flat outputs/ (no run_id)")
    p.add_argument("--stages", nargs="+", default=["1.3", "1.4", "1.5", "1.6", "2.1", "2.2", "2.3", "size_matched"],
                   help="Stages to run")
    p.add_argument("--quick", action="store_true", help="Quick run (subset of sites/models)")
    args = p.parse_args()

    data_dir = ROOT / "data"
    if not (data_dir / "ingestion_report.json").exists():
        print("Run ingestion first: python -m src.ingest", file=sys.stderr)
        sys.exit(1)

    if args.legacy:
        output_dir = ROOT / "outputs"
        if "1.3" in args.stages:
            validation.run_internal_validation(
                data_dir=data_dir, output_dir=output_dir,
                sites=["cleveland"], model_keys=["lr"] if args.quick else None,
            )
        if "1.4" in args.stages:
            validation.run_external_uci_matrix(
                data_dir=data_dir, output_dir=output_dir,
                sites=["cleveland", "hungary"] if args.quick else None,
                model_keys=["lr"] if args.quick else None,
            )
        if "1.5" in args.stages:
            validation.run_kaggle_uci_tests(data_dir=data_dir, output_dir=output_dir)
        if "1.6" in args.stages:
            run_shift_diagnostics(data_dir=data_dir, output_dir=output_dir)
        if "2.1" in args.stages:
            assess_calibration(outputs_dir=output_dir)
        if "2.2" in args.stages:
            run_recalibration(outputs_dir=output_dir)
        if "2.3" in args.stages:
            run_lightweight_updating(outputs_dir=output_dir)
        if "size_matched" in args.stages:
            run_size_matched_sensitivity(data_dir=data_dir, output_dir=output_dir)
        print("Legacy run done. Outputs under", output_dir)
    else:
        stages = args.stages
        if args.quick:
            stages = ["1.3", "1.6"]  # minimal for quick
        run_root = run_canonical(data_dir=data_dir, stages=stages)
        print("Canonical run done. Outputs under", run_root)
        print("Manifest:", run_root / "manifest.json")


if __name__ == "__main__":
    main()
