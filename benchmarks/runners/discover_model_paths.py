#!/usr/bin/env python3
"""Discover model path metadata for a mine (wrapper entrypoint)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.runners.runner_metrics import attach_metric_fields, measure_startup
from benchmarks.discover_model_paths import main

_STARTUP = measure_startup()

def run() -> int:
    try:
        code = int(main())
    except Exception as exc:
        payload = {"status": "failed", "error_type": type(exc).__name__}
        attach_metric_fields(
            payload,
            startup=_STARTUP,
            status="failed",
            error_type=type(exc).__name__,
            tor_mode="disabled",
            proxy_url_scheme="none",
            profile_name="discover_model_paths",
        )
        print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
        return 1

    payload: dict[str, object] = {}
    status = "ok" if code == 0 else "failed"
    attach_metric_fields(
        payload,
        startup=_STARTUP,
        status=status,
        error_type="none" if code == 0 else "non_zero_exit",
        tor_mode="disabled",
        proxy_url_scheme="none",
        profile_name="discover_model_paths",
    )
    print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(run())
