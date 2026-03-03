#!/usr/bin/env python3
"""Run evaluation pipeline: python scripts/run_evaluation.py [--run-id ID]"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation import run_evaluation

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--reports-dir", default=str(ROOT / "reports"))
    parser.add_argument("--run-id", default=None, help="Specific run ID; default: latest")
    args = parser.parse_args()

    run_evaluation(
        outputs_dir=args.outputs_dir,
        reports_dir=args.reports_dir,
        run_id=args.run_id,
    )
