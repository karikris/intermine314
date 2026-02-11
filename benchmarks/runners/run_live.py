#!/usr/bin/env python3
"""Run the live benchmark workflow using the canonical top-level benchmark entrypoint."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.benchmarks import main


if __name__ == "__main__":
    raise SystemExit(main())
