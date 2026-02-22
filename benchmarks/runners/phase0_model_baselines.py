#!/usr/bin/env python3
"""Collect phase-0 baselines for model-layer behavior.

Baselines captured:
- package import latency for ``intermine314.model`` (cold-process repetitions)
- creation throughput for ``Path`` and ``Column`` objects
- tracemalloc peak bytes for object construction runs
- best-effort peak RSS snapshot (platform dependent)

Exit codes:
- 0: success
- 1: failure
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.runners.runner_metrics import attach_metric_fields, measure_startup

_STARTUP = measure_startup()

SUCCESS_EXIT_CODE = 0
FAIL_EXIT_CODE = 1

VALID_OBJECT_KINDS = ("path", "column")
DEFAULT_OBJECT_KINDS = "both"
DEFAULT_IMPORT_REPETITIONS = 5
DEFAULT_OBJECT_COUNT = 50_000

MODEL_XML = """
<model name="mock" package="org.mock">
  <class name="Gene">
    <reference name="organism" referenced-type="Organism"/>
    <attribute name="symbol" type="java.lang.String"/>
  </class>
  <class name="Organism">
    <attribute name="name" type="java.lang.String"/>
  </class>
</model>
""".strip()

IMPORT_SNIPPET = (
    "import json,sys,time,tracemalloc;"
    "tracemalloc.start();"
    "t0=time.perf_counter();"
    "import intermine314.model as m;"
    "elapsed=time.perf_counter()-t0;"
    "cur,peak=tracemalloc.get_traced_memory();"
    "print(json.dumps({'seconds':elapsed,'module_count':len(sys.modules),'tracemalloc_peak_bytes':peak}))"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stat_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    result = {
        "n": float(len(values)),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }
    if len(values) > 1:
        result["stddev"] = statistics.stdev(values)
    else:
        result["stddev"] = 0.0
    return result


def _pythonpath_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    entries = [str(SRC)]
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _ru_maxrss_bytes() -> int | None:
    try:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if rss <= 0:
        return None
    if sys.platform.startswith("linux"):
        return rss * 1024
    return rss


def _normalize_object_kinds(value: str) -> tuple[str, ...]:
    raw = str(value or "").strip().lower()
    if raw in {"", "both"}:
        return VALID_OBJECT_KINDS
    parts = [token.strip().lower() for token in raw.split(",") if token.strip()]
    if not parts:
        return VALID_OBJECT_KINDS

    kinds: list[str] = []
    for kind in parts:
        if kind not in VALID_OBJECT_KINDS:
            choices = ", ".join(VALID_OBJECT_KINDS)
            raise ValueError(f"kinds must be comma-separated values from: {choices}, or both")
        if kind not in kinds:
            kinds.append(kind)
    return tuple(kinds)


def _run_import_baseline_subprocess(*, repetitions: int) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for _ in range(int(repetitions)):
        proc = subprocess.run(
            [sys.executable, "-c", IMPORT_SNIPPET],
            env=_pythonpath_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"import baseline subprocess failed: {proc.stderr.strip()}")
        line = proc.stdout.strip().splitlines()[-1]
        runs.append(json.loads(line))

    seconds = [float(item["seconds"]) for item in runs]
    peaks = [float(item["tracemalloc_peak_bytes"]) for item in runs]
    module_counts = [float(item["module_count"]) for item in runs]
    return {
        "repetitions": int(repetitions),
        "runs": runs,
        "seconds": _stat_summary(seconds),
        "tracemalloc_peak_bytes": _stat_summary(peaks),
        "module_count": _stat_summary(module_counts),
    }


def _build_model():
    from intermine314.model import Model

    return Model(MODEL_XML)


def _measure_object_creation(kind: str, *, count: int) -> dict[str, Any]:
    if kind not in VALID_OBJECT_KINDS:
        choices = ", ".join(VALID_OBJECT_KINDS)
        raise ValueError(f"kind must be one of: {choices}")

    model = _build_model()
    objects = []
    tracemalloc.start()
    started = time.perf_counter()
    for index in range(int(count)):
        if kind == "path":
            value = model.make_path("Gene.organism.name" if (index & 1) else "Gene.symbol")
        else:
            value = model.column("Gene.organism.name" if (index & 1) else "Gene")
        objects.append(value)
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "kind": kind,
        "count": int(len(objects)),
        "elapsed_s": elapsed,
        "objects_per_s": (float(len(objects)) / elapsed) if elapsed > 0 else None,
        "tracemalloc_peak_bytes": int(peak),
        "peak_rss_bytes": _ru_maxrss_bytes(),
    }


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    import_baseline = _run_import_baseline_subprocess(repetitions=args.import_repetitions)
    object_kinds = _normalize_object_kinds(args.kinds)

    object_baselines = {
        kind: _measure_object_creation(kind, count=args.object_count) for kind in object_kinds
    }
    report = {
        "generated_at": _now_iso(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "import_baseline": import_baseline,
        "object_baselines": object_baselines,
        "summary": {
            "kinds": list(object_kinds),
            "object_count_per_kind": int(args.object_count),
        },
    }
    attach_metric_fields(
        report,
        startup=_STARTUP,
        status="ok",
        error_type="none",
        tor_mode="disabled",
        proxy_url_scheme="none",
        profile_name="phase0_model",
    )
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinds", default=DEFAULT_OBJECT_KINDS)
    parser.add_argument("--object-count", type=int, default=DEFAULT_OBJECT_COUNT)
    parser.add_argument("--import-repetitions", type=int, default=DEFAULT_IMPORT_REPETITIONS)
    parser.add_argument("--json-out", default=None)
    return parser


def run(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if int(args.object_count) <= 0:
            raise ValueError("object-count must be > 0")
        if int(args.import_repetitions) <= 0:
            raise ValueError("import-repetitions must be > 0")

        report = _build_report(args)
        payload = json.dumps(report, sort_keys=True, indent=2)
        print(payload)
        if args.json_out:
            out = Path(str(args.json_out))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(payload + "\n", encoding="utf-8")
        return SUCCESS_EXIT_CODE
    except Exception as exc:
        payload = {
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
        attach_metric_fields(
            payload,
            startup=_STARTUP,
            status="failed",
            error_type=type(exc).__name__,
            tor_mode="disabled",
            proxy_url_scheme="none",
            profile_name="phase0_model",
        )
        print(
            json.dumps(
                payload,
                sort_keys=True,
            )
        )
        return FAIL_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(run())
