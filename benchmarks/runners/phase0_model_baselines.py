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
import platform
import sys
import time
import tracemalloc
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.bench_constants import DEFAULT_RUNNER_IMPORT_REPETITIONS
from benchmarks.runners.common import (
    now_utc_iso,
    ru_maxrss_bytes,
    run_import_baseline_subprocess,
)
from benchmarks.runners.runner_metrics import attach_metric_fields, measure_startup

_STARTUP = measure_startup()

SUCCESS_EXIT_CODE = 0
FAIL_EXIT_CODE = 1

VALID_OBJECT_KINDS = ("path", "column")
VALID_METRIC_GOALS = ("throughput", "memory")
VALID_SAMPLE_MODES = ("none", "head", "stride")
DEFAULT_OBJECT_KINDS = "both"
DEFAULT_IMPORT_REPETITIONS = DEFAULT_RUNNER_IMPORT_REPETITIONS
DEFAULT_OBJECT_COUNT = 50_000
DEFAULT_METRIC_GOAL = "throughput"
DEFAULT_SAMPLE_MODE = "none"
DEFAULT_SAMPLE_SIZE = 64

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

_now_iso = now_utc_iso
_ru_maxrss_bytes = ru_maxrss_bytes
_run_import_baseline_subprocess = partial(
    run_import_baseline_subprocess,
    import_snippet=IMPORT_SNIPPET,
    source_root=SRC,
)


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


def _normalize_metric_goal(value: str) -> str:
    metric_goal = str(value or "").strip().lower() or DEFAULT_METRIC_GOAL
    if metric_goal not in VALID_METRIC_GOALS:
        choices = ", ".join(VALID_METRIC_GOALS)
        raise ValueError(f"metric-goal must be one of: {choices}")
    return metric_goal


def _normalize_sample_mode(value: str) -> str:
    sample_mode = str(value or "").strip().lower() or DEFAULT_SAMPLE_MODE
    if sample_mode not in VALID_SAMPLE_MODES:
        choices = ", ".join(VALID_SAMPLE_MODES)
        raise ValueError(f"sample-mode must be one of: {choices}")
    return sample_mode


def _resolve_retain_objects(metric_goal: str, explicit_retain_objects: bool | None) -> bool:
    if explicit_retain_objects is not None:
        return bool(explicit_retain_objects)
    return metric_goal == "memory"


def _should_retain_sample(
    *,
    index: int,
    total_count: int,
    sample_mode: str,
    sample_size: int,
) -> bool:
    if sample_mode == "none" or sample_size <= 0:
        return False
    if sample_mode == "head":
        return index < sample_size
    # stride mode: spread samples across the full run.
    stride = max(1, int(total_count // max(1, sample_size)))
    return (index % stride) == 0


def _build_model():
    from intermine314.model import Model

    return Model(MODEL_XML)


def _measure_object_creation(
    kind: str,
    *,
    count: int,
    retain_objects: bool,
    sample_mode: str,
    sample_size: int,
) -> dict[str, Any]:
    if kind not in VALID_OBJECT_KINDS:
        choices = ", ".join(VALID_OBJECT_KINDS)
        raise ValueError(f"kind must be one of: {choices}")

    model = _build_model()
    retained_objects = [] if retain_objects else None
    sampled_objects: list[Any] = []
    tracemalloc.start()
    started = time.perf_counter()
    for index in range(int(count)):
        if kind == "path":
            value = model.make_path("Gene.organism.name" if (index & 1) else "Gene.symbol")
        else:
            value = model.column("Gene.organism.name" if (index & 1) else "Gene")
        if retained_objects is not None:
            retained_objects.append(value)
        elif _should_retain_sample(
            index=index,
            total_count=int(count),
            sample_mode=sample_mode,
            sample_size=sample_size,
        ):
            sampled_objects.append(value)
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    retained_count = len(retained_objects) if retained_objects is not None else len(sampled_objects)

    return {
        "kind": kind,
        "count": int(count),
        "elapsed_s": elapsed,
        "objects_per_s": (float(count) / elapsed) if elapsed > 0 else None,
        "tracemalloc_peak_bytes": int(peak),
        "peak_rss_bytes": _ru_maxrss_bytes(),
        "retain_objects": bool(retain_objects),
        "sample_mode": sample_mode,
        "sample_size": int(sample_size),
        "retained_count": int(retained_count),
    }


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    import_baseline = _run_import_baseline_subprocess(repetitions=args.import_repetitions)
    object_kinds = _normalize_object_kinds(args.kinds)
    metric_goal = _normalize_metric_goal(args.metric_goal)
    sample_mode = _normalize_sample_mode(args.sample_mode)
    retain_objects = _resolve_retain_objects(metric_goal, args.retain_objects)

    object_baselines = {
        kind: _measure_object_creation(
            kind,
            count=args.object_count,
            retain_objects=retain_objects,
            sample_mode=sample_mode,
            sample_size=int(args.sample_size),
        )
        for kind in object_kinds
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
            "metric_goal": metric_goal,
            "retain_objects": retain_objects,
            "sample_mode": sample_mode,
            "sample_size": int(args.sample_size),
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
    parser.add_argument("--metric-goal", default=DEFAULT_METRIC_GOAL, choices=list(VALID_METRIC_GOALS))
    parser.add_argument(
        "--retain-objects",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Retain all generated objects in memory. Defaults to true only for --metric-goal memory.",
    )
    parser.add_argument("--sample-mode", default=DEFAULT_SAMPLE_MODE, choices=list(VALID_SAMPLE_MODES))
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of objects to retain when sample mode is enabled and retain-objects is disabled.",
    )
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
        if int(args.sample_size) < 0:
            raise ValueError("sample-size must be >= 0")

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
