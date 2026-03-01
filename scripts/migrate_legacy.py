#!/usr/bin/env python3
"""
Migrate flat outputs (internal/, external_uci/, etc.) to outputs/legacy/.
Preserves existing data; creates timestamped subdir if legacy already has content.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

# Subdirs that are flat (legacy) experiment outputs
LEGACY_SUBDIRS = [
    "internal",
    "internal_cfs",
    "external_uci",
    "external_kaggle_uci",
    "calibration",
    "shift",
    "size_matched",
    "paper_ready",
]


def main() -> None:
    legacy_root = OUTPUTS / "legacy"
    existing = list(legacy_root.iterdir()) if legacy_root.exists() else []
    if existing:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        dest = legacy_root / ts
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Legacy already exists; creating {dest.name}/")
    else:
        dest = legacy_root
        dest.mkdir(parents=True, exist_ok=True)
        print("Creating outputs/legacy/")

    for name in LEGACY_SUBDIRS:
        src = OUTPUTS / name
        if src.exists() and src.is_dir():
            target = dest / name
            if target.exists():
                print(f"  Skip {name}: already at {target}")
            else:
                shutil.copytree(src, target)
                print(f"  Copied {name} -> {target.relative_to(OUTPUTS)}")
    print("Done. Original flat dirs remain; remove manually if desired.")


if __name__ == "__main__":
    main()
