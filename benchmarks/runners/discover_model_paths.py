#!/usr/bin/env python3
"""Discover model path metadata for a mine (wrapper entrypoint)."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarking.discover_model_paths import main


if __name__ == "__main__":
    raise SystemExit(main())
