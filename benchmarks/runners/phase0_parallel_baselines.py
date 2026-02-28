#!/usr/bin/env python3
"""Collect phase-0 baselines for parallel policy behavior.

Baselines captured:
- import latency for parallel/query modules (cold-process repetitions)
- synthetic parallel throughput and peak RSS per order mode
- structured parallel log volume and event contracts

Modes:
- ordered
- unordered
- window

Exit codes:
- 0: at least one mode completed
- 1: failure
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.bench_constants import (
    DEFAULT_RUNNER_DEBUG_LOG_LEVEL,
    DEFAULT_RUNNER_IMPORT_REPETITIONS,
    DEFAULT_RUNNER_PARALLEL_PROFILE,
)
from benchmarks.runners.common import (
    now_utc_iso,
    pythonpath_env,
    ru_maxrss_bytes,
    run_import_baseline_subprocess,
)
from benchmarks.runners.runner_metrics import attach_metric_fields, measure_startup
from intermine314.export.resource_profile import resolve_resource_profile
from intermine314.query import builder as query_builder

_STARTUP = measure_startup()

SUCCESS_EXIT_CODE = 0
FAIL_EXIT_CODE = 1

VALID_CASE_MODES = ("ordered", "unordered", "window")
DEFAULT_CASE_MODES = ",".join(VALID_CASE_MODES)
DEFAULT_ROWS_TARGET = 50_000
DEFAULT_PAGE_SIZE = 1_000
DEFAULT_MAX_WORKERS = 4
DEFAULT_ORDERED_WINDOW_PAGES = 10
DEFAULT_IMPORT_REPETITIONS = DEFAULT_RUNNER_IMPORT_REPETITIONS
DEFAULT_PROFILE = DEFAULT_RUNNER_PARALLEL_PROFILE
DEFAULT_LOG_LEVEL = DEFAULT_RUNNER_DEBUG_LOG_LEVEL

IMPORT_SNIPPET = (
    "import json,sys,time,tracemalloc;"
    "tracemalloc.start();"
    "t0=time.perf_counter();"
    "import intermine314.parallel.policy as p;"
    "import intermine314.query.builder as b;"
    "elapsed=time.perf_counter()-t0;"
    "cur,peak=tracemalloc.get_traced_memory();"
    "print(json.dumps({'seconds':elapsed,'module_count':len(sys.modules),'tracemalloc_peak_bytes':peak}))"
)

_now_iso = now_utc_iso
_pythonpath_env = partial(pythonpath_env, source_root=SRC)
_ru_maxrss_bytes = ru_maxrss_bytes
_run_import_baseline_subprocess = partial(
    run_import_baseline_subprocess,
    import_snippet=IMPORT_SNIPPET,
    source_root=SRC,
)


class _ParallelLogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records = 0
        self.bytes = 0
        self.level_counts: dict[str, int] = {}
        self.event_counts: dict[str, int] = {}
        self.event_levels: dict[str, set[str]] = {}

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        self.records += 1
        self.bytes += len(str(message).encode("utf-8", errors="replace"))
        level = str(record.levelname).upper()
        self.level_counts[level] = int(self.level_counts.get(level, 0)) + 1
        try:
            payload = json.loads(message)
        except Exception:
            return
        event = payload.get("event")
        if not isinstance(event, str) or not event:
            return
        self.event_counts[event] = int(self.event_counts.get(event, 0)) + 1
        levels = self.event_levels.setdefault(event, set())
        levels.add(level)


class _SyntheticParallelQuery:
    def __init__(self, rows_total: int):
        self.rows_total = int(max(0, rows_total))

    def count(self):
        return self.rows_total

    def results(self, row="dict", start=0, size=None):
        offset = max(0, int(start))
        limit = self.rows_total if size is None else max(0, int(size))
        stop = min(self.rows_total, offset + limit)
        for index in range(offset, stop):
            if row == "list":
                yield [index, f"G{index}", f"Gene {index}"]
            else:
                yield {
                    "Gene.id": index,
                    "Gene.symbol": f"G{index}",
                    "Gene.name": f"Gene {index}",
                }

    def _resolve_parallel_strategy(self, pagination, start, size):
        return query_builder.Query._resolve_parallel_strategy(self, pagination, start, size)

    def _normalize_order_mode(self, ordered):
        return query_builder.Query._normalize_order_mode(self, ordered)

    def _apply_parallel_profile(self, profile, ordered, large_query_mode):
        return query_builder.Query._apply_parallel_profile(self, profile, ordered, large_query_mode)

    def _resolve_effective_workers(self, max_workers, size):
        _ = size
        return int(max_workers)

    def _coerce_parallel_options(self, *, parallel_options=None):
        return query_builder.Query._coerce_parallel_options(self, parallel_options=parallel_options)

    def _resolve_parallel_options(self, **kwargs):
        return query_builder.Query._resolve_parallel_options(self, **kwargs)

    def _run_parallel_offset(self, **kwargs):
        return query_builder.Query._run_parallel_offset(self, **kwargs)

    def _run_parallel_keyset(self, **_kwargs):  # pragma: no cover - defensive guard
        raise AssertionError("keyset path is unsupported in phase0 synthetic offset baselines")


def _parse_optional_positive_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "auto"}:
        return None
    parsed = int(text)
    if parsed <= 0:
        raise ValueError("value must be a positive integer or auto")
    return parsed


def _normalize_case_modes(value: str) -> tuple[str, ...]:
    raw = str(value or "").strip().lower()
    parts = [token.strip().lower() for token in raw.split(",") if token.strip()]
    if not parts:
        parts = list(VALID_CASE_MODES)
    deduped: list[str] = []
    for mode in parts:
        if mode not in VALID_CASE_MODES:
            choices = ", ".join(VALID_CASE_MODES)
            raise ValueError(f"modes must be comma-separated values from: {choices}")
        if mode not in deduped:
            deduped.append(mode)
    return tuple(deduped)


def _worker_case(args: argparse.Namespace) -> int:
    logger = logging.getLogger("intermine314.query.parallel")
    level = getattr(logging, str(args.log_level).upper(), logging.DEBUG)
    previous_level = logger.level
    capture = _ParallelLogCapture()
    capture.setLevel(level)
    logger.addHandler(capture)
    logger.setLevel(level)

    try:
        resolved_resource_profile = resolve_resource_profile(args.resource_profile)
        effective_max_workers = (
            int(args.max_workers)
            if args.max_workers is not None
            else (
                int(resolved_resource_profile.max_workers)
                if resolved_resource_profile.max_workers is not None
                else DEFAULT_MAX_WORKERS
            )
        )
        effective_prefetch = (
            args.prefetch if args.prefetch is not None else resolved_resource_profile.prefetch
        )
        effective_inflight_limit = (
            args.inflight_limit if args.inflight_limit is not None else resolved_resource_profile.inflight_limit
        )
        effective_max_inflight_bytes_estimate = (
            args.max_inflight_bytes_estimate
            if args.max_inflight_bytes_estimate is not None
            else resolved_resource_profile.max_inflight_bytes_estimate
        )
        query = _SyntheticParallelQuery(args.rows_target)
        options = query_builder.ParallelOptions(
            page_size=args.page_size,
            max_workers=effective_max_workers,
            ordered=args.mode,
            prefetch=effective_prefetch,
            inflight_limit=effective_inflight_limit,
            max_inflight_bytes_estimate=effective_max_inflight_bytes_estimate,
            ordered_window_pages=args.ordered_window_pages,
            profile=args.profile,
            large_query_mode=args.large_query_mode,
            pagination="offset",
        )
        started = time.perf_counter()
        rows_exported = 0
        for _row in query_builder.Query.run_parallel(
            query,
            row="dict",
            start=0,
            size=args.rows_target,
            parallel_options=options,
            job_id=f"phase0_parallel_{args.mode}",
        ):
            rows_exported += 1
        elapsed = time.perf_counter() - started

        start_count = int(capture.event_counts.get("parallel_export_start", 0))
        done_count = int(capture.event_counts.get("parallel_export_done", 0))
        scheduler_count = int(capture.event_counts.get("parallel_ordered_scheduler_stats", 0))
        scheduler_levels = sorted(capture.event_levels.get("parallel_ordered_scheduler_stats", set()))

        start_done_pair = start_count == 1 and done_count == 1
        if args.mode == "ordered":
            scheduler_expectation = scheduler_count >= 1
            scheduler_debug_only = bool(scheduler_levels) and scheduler_levels == ["DEBUG"]
        else:
            scheduler_expectation = scheduler_count == 0
            scheduler_debug_only = scheduler_count == 0

        payload = {
            "mode": args.mode,
            "status": "ok",
            "rows_target": int(args.rows_target),
            "rows_exported": int(rows_exported),
            "resource_profile": resolved_resource_profile.name,
            "max_workers": effective_max_workers,
            "prefetch": effective_prefetch,
            "inflight_limit": effective_inflight_limit,
            "max_inflight_bytes_estimate": effective_max_inflight_bytes_estimate,
            "elapsed_s": elapsed,
            "rows_per_s": (float(rows_exported) / elapsed) if elapsed > 0 else None,
            "peak_rss_bytes": _ru_maxrss_bytes(),
            "log_volume": {
                "records": capture.records,
                "bytes": capture.bytes,
                "levels": dict(capture.level_counts),
                "events": dict(capture.event_counts),
                "event_levels": {name: sorted(levels) for name, levels in capture.event_levels.items()},
            },
            "observability_probes": {
                "start_done_pair": start_done_pair,
                "ordered_scheduler_expectation": scheduler_expectation,
                "scheduler_debug_only": scheduler_debug_only,
            },
        }
        attach_metric_fields(
            payload,
            startup=_STARTUP,
            status="ok",
            error_type="none",
            tor_mode="disabled",
            proxy_url_scheme="none",
            profile_name=str(args.profile),
        )
        print(json.dumps(payload, sort_keys=True), flush=True)
        return SUCCESS_EXIT_CODE
    except Exception as exc:
        payload = {
            "mode": args.mode,
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
            profile_name=str(args.profile),
        )
        print(
            json.dumps(
                payload,
                sort_keys=True,
            ),
            flush=True,
        )
        return FAIL_EXIT_CODE
    finally:
        logger.removeHandler(capture)
        logger.setLevel(previous_level)


def _build_worker_command(args: argparse.Namespace, mode: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-case",
        "--mode",
        mode,
        "--rows-target",
        str(args.rows_target),
        "--page-size",
        str(args.page_size),
        "--ordered-window-pages",
        str(args.ordered_window_pages),
        "--profile",
        str(args.profile),
        "--log-level",
        str(args.log_level),
        "--resource-profile",
        str(args.resource_profile),
    ]
    if args.max_workers is not None:
        cmd.extend(["--max-workers", str(args.max_workers)])
    if args.large_query_mode:
        cmd.append("--large-query-mode")
    else:
        cmd.append("--no-large-query-mode")
    if args.prefetch is not None:
        cmd.extend(["--prefetch", str(args.prefetch)])
    if args.inflight_limit is not None:
        cmd.extend(["--inflight-limit", str(args.inflight_limit)])
    if args.max_inflight_bytes_estimate is not None:
        cmd.extend(["--max-inflight-bytes-estimate", str(args.max_inflight_bytes_estimate)])
    return cmd


def _run_mode_worker(args: argparse.Namespace, mode: str) -> tuple[int, dict[str, Any]]:
    proc = subprocess.run(
        _build_worker_command(args, mode),
        env=_pythonpath_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout.strip()
    payload: dict[str, Any] = {"mode": mode}
    if stdout:
        try:
            payload = json.loads(stdout.splitlines()[-1])
        except Exception:
            payload = {"mode": mode, "raw_stdout": stdout}
    if proc.returncode != 0:
        payload["stderr"] = proc.stderr.strip()
    return proc.returncode, payload


def _build_report(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    report: dict[str, Any] = {
        "timestamp_utc": _now_iso(),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "runtime": {
            "modes": list(_normalize_case_modes(args.modes)),
            "rows_target": args.rows_target,
            "page_size": args.page_size,
            "max_workers": args.max_workers,
            "resource_profile": args.resource_profile,
            "prefetch": args.prefetch,
            "inflight_limit": args.inflight_limit,
            "max_inflight_bytes_estimate": args.max_inflight_bytes_estimate,
            "ordered_window_pages": args.ordered_window_pages,
            "profile": args.profile,
            "large_query_mode": bool(args.large_query_mode),
            "log_level": str(args.log_level).upper(),
        },
    }

    report["import_baseline"] = _run_import_baseline_subprocess(repetitions=args.import_repetitions)

    mode_payloads: dict[str, Any] = {}
    success_count = 0
    for mode in _normalize_case_modes(args.modes):
        code, payload = _run_mode_worker(args, mode)
        mode_payloads[mode] = payload
        if code == SUCCESS_EXIT_CODE:
            success_count += 1
    report["parallel_baselines"] = mode_payloads
    report["summary"] = {
        "modes_requested": list(_normalize_case_modes(args.modes)),
        "modes_succeeded": success_count,
        "modes_failed": len(_normalize_case_modes(args.modes)) - success_count,
    }
    status = "ok" if success_count > 0 else "failed"
    error_type = "none" if success_count > 0 else "mode_failures"
    attach_metric_fields(
        report,
        startup=_STARTUP,
        status=status,
        error_type=error_type,
        tor_mode="disabled",
        proxy_url_scheme="none",
        profile_name=str(args.profile),
    )
    return (SUCCESS_EXIT_CODE if success_count > 0 else FAIL_EXIT_CODE), report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect phase-0 baselines for parallel policy behavior")
    parser.add_argument("--modes", default=DEFAULT_CASE_MODES)
    parser.add_argument("--rows-target", type=int, default=DEFAULT_ROWS_TARGET)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--max-workers", type=_parse_optional_positive_int, default=None)
    parser.add_argument("--resource-profile", default="default")
    parser.add_argument("--prefetch", type=_parse_optional_positive_int, default=None)
    parser.add_argument("--inflight-limit", type=_parse_optional_positive_int, default=None)
    parser.add_argument("--max-inflight-bytes-estimate", type=_parse_optional_positive_int, default=None)
    parser.add_argument("--ordered-window-pages", type=int, default=DEFAULT_ORDERED_WINDOW_PAGES)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--large-query-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--import-repetitions", type=int, default=DEFAULT_IMPORT_REPETITIONS)
    parser.add_argument("--log-level", default=DEFAULT_LOG_LEVEL)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--worker-case", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=VALID_CASE_MODES, default="ordered", help=argparse.SUPPRESS)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    _normalize_case_modes(args.modes)
    if args.rows_target <= 0:
        raise ValueError("--rows-target must be a positive integer")
    if args.page_size <= 0:
        raise ValueError("--page-size must be a positive integer")
    if args.ordered_window_pages <= 0:
        raise ValueError("--ordered-window-pages must be a positive integer")
    if args.import_repetitions <= 0:
        raise ValueError("--import-repetitions must be a positive integer")


def run(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    if args.worker_case:
        return _worker_case(args)

    code, report = _build_report(args)
    output = json.dumps(report, sort_keys=True, default=str)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n", encoding="utf-8")
    print(output, flush=True)
    return code


def main() -> int:
    try:
        return run()
    except Exception as exc:
        payload = {"status": "failed", "error": str(exc), "error_type": type(exc).__name__}
        attach_metric_fields(
            payload,
            startup=_STARTUP,
            status="failed",
            error_type=type(exc).__name__,
            tor_mode="disabled",
            proxy_url_scheme="none",
            profile_name=DEFAULT_PROFILE,
        )
        print(json.dumps(payload, sort_keys=True), flush=True)
        return FAIL_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
