#!/usr/bin/env python3
"""Run a fixed small fetch benchmark for CI guardrails.

This runner is intentionally narrow and stable:
- single mine target
- single page size
- fixed worker count (default 2)
- direct or tor transport mode
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.bench_constants import (
    DEFAULT_BENCHMARK_MINE_URL,
    DEFAULT_RUNNER_IMPORT_REPETITIONS,
    DEFAULT_RUNNER_PREFLIGHT_TIMEOUT_SECONDS,
)
from benchmarks.bench_utils import parse_csv_tokens
from benchmarks.bench_fetch import mode_label_for_workers, run_mode
from benchmarks.runners.common import (
    SocketMonitor,
    TOR_DNS_SAFETY_POLICY,
    TOR_GUARDRAIL_UNSAFE_PROXY_URL,
    probe_direct,
    probe_tor,
    run_import_baseline_subprocess,
    stable_import_baseline_metrics,
    validate_tor_proxy_url,
)
from benchmarks.runners.runner_metrics import (
    attach_metric_fields,
    measure_startup,
    proxy_url_scheme_from_url,
)
from intermine314.service.tor import tor_proxy_url

_STARTUP = measure_startup()

SUCCESS_EXIT_CODE = 0
FAIL_EXIT_CODE = 1
SKIP_EXIT_CODE = 2
PROFILE_NAME = "phase0_ci_fixed_fetch"
VALID_TRANSPORT_MODES = ("direct", "tor")
DEFAULT_QUERY_ROOT = "Gene"
DEFAULT_QUERY_VIEWS = "Gene.primaryIdentifier"
DEFAULT_QUERY_JOINS = ""
DEFAULT_ROWS_TARGET = 2_000
DEFAULT_PAGE_SIZE = 1_000
DEFAULT_WORKERS = 2
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_IMPORT_REPETITIONS = DEFAULT_RUNNER_IMPORT_REPETITIONS
IMPORT_SNIPPET = "\n".join(
    [
        "import json",
        "import resource",
        "import sys",
        "import time",
        "import tracemalloc",
        "",
        "def _rss_bytes():",
        "    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)",
        "    if rss <= 0:",
        "        return None",
        "    if sys.platform.startswith('linux'):",
        "        return rss * 1024",
        "    return rss",
        "",
        "before = len(sys.modules)",
        "tracemalloc.start()",
        "t0 = time.perf_counter()",
        "import intermine314.service.transport as _transport",
        "import intermine314.query.builder as _query_builder",
        "elapsed = time.perf_counter() - t0",
        "_cur, peak = tracemalloc.get_traced_memory()",
        "after = len(sys.modules)",
        "print(json.dumps({'seconds': elapsed, 'module_count': after, 'imported_module_count': after - before, 'tracemalloc_peak_bytes': peak, 'max_rss_bytes': _rss_bytes()}))",
    ]
)

_probe_direct = probe_direct
_probe_tor = probe_tor
_run_mode = run_mode
_run_import_baseline_subprocess = lambda repetitions: run_import_baseline_subprocess(  # noqa: E731
    import_snippet=IMPORT_SNIPPET,
    repetitions=int(repetitions),
    source_root=SRC,
)


def _preflight(
    *,
    mine_url: str,
    transport_mode: str,
    timeout_seconds: float,
    tor_proxy_url_value: str | None,
) -> dict[str, Any]:
    if transport_mode == "tor":
        safe_proxy = validate_tor_proxy_url(
            tor_proxy_url_value,
            context="phase0 ci fixed fetch proxy_url",
        )
        return _probe_tor(
            mine_url,
            timeout_seconds,
            context="phase0 ci fixed fetch proxy_url",
            user_agent="intermine314-phase0-ci-fixed-fetch",
            proxy_url=safe_proxy,
        )
    return _probe_direct(mine_url, timeout_seconds)


def _tor_stability_payload(
    *,
    transport_mode: str,
    proxy_url: str | None,
    socket_monitor: dict[str, Any] | None,
) -> dict[str, Any]:
    if str(transport_mode).strip().lower() != "tor":
        return {
            "status": "not_applicable",
            "reason": "transport mode is not tor",
            "socket_monitor": socket_monitor
            or {"status": "not_applicable", "reason": "tor mode disabled"},
        }
    unsafe_proxy_rejected = False
    unsafe_error_type = "none"
    try:
        validate_tor_proxy_url(TOR_GUARDRAIL_UNSAFE_PROXY_URL, context="phase0 ci tor unsafe-proxy check")
    except Exception as exc:
        unsafe_proxy_rejected = True
        unsafe_error_type = type(exc).__name__
    socket_payload = socket_monitor or {}
    leak_suspected = False
    delta = socket_payload.get("delta_open_sockets")
    if isinstance(delta, int) and delta > 0:
        leak_suspected = True
    return {
        "status": "ok" if unsafe_proxy_rejected else "failed",
        "dns_safety_policy": TOR_DNS_SAFETY_POLICY,
        "tor_proxy_scheme": proxy_url_scheme_from_url(proxy_url),
        "unsafe_proxy_rejected": unsafe_proxy_rejected,
        "unsafe_proxy_rejection_error_type": unsafe_error_type,
        "tor_dns_safety": "enforced" if unsafe_proxy_rejected else "violated",
        "socket_monitor": socket_payload,
        "socket_leak_suspected": leak_suspected,
        "operational_validation": {
            "method": "tor_testsocks_log_observation",
            "note": "Enable Tor TestSocks and inspect Tor logs during tor-mode fixed fetch runs.",
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mine-url", default=DEFAULT_BENCHMARK_MINE_URL)
    parser.add_argument("--rows-target", type=int, default=DEFAULT_ROWS_TARGET)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--query-root", default=DEFAULT_QUERY_ROOT)
    parser.add_argument("--query-views", default=DEFAULT_QUERY_VIEWS)
    parser.add_argument("--query-joins", default=DEFAULT_QUERY_JOINS)
    parser.add_argument("--transport-mode", choices=list(VALID_TRANSPORT_MODES), default="direct")
    parser.add_argument("--tor-proxy-url", default=tor_proxy_url())
    parser.add_argument("--preflight-timeout-seconds", type=float, default=DEFAULT_RUNNER_PREFLIGHT_TIMEOUT_SECONDS)
    parser.add_argument("--request-timeout-seconds", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--import-repetitions", type=int, default=DEFAULT_IMPORT_REPETITIONS)
    parser.add_argument("--json-out", default=None)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.rows_target) <= 0:
        raise ValueError("--rows-target must be a positive integer")
    if int(args.page_size) <= 0:
        raise ValueError("--page-size must be a positive integer")
    if int(args.workers) <= 0:
        raise ValueError("--workers must be a positive integer")
    if float(args.preflight_timeout_seconds) <= 0:
        raise ValueError("--preflight-timeout-seconds must be positive")
    if float(args.request_timeout_seconds) <= 0:
        raise ValueError("--request-timeout-seconds must be positive")
    if int(args.max_retries) <= 0:
        raise ValueError("--max-retries must be a positive integer")
    if int(args.import_repetitions) <= 0:
        raise ValueError("--import-repetitions must be a positive integer")


def run(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_args(args)

    mode = str(mode_label_for_workers(args.workers))
    proxy_scheme = "none"
    if args.transport_mode == "tor":
        proxy_scheme = proxy_url_scheme_from_url(args.tor_proxy_url)

    probe = _preflight(
        mine_url=args.mine_url,
        transport_mode=args.transport_mode,
        timeout_seconds=float(args.preflight_timeout_seconds),
        tor_proxy_url_value=args.tor_proxy_url,
    )
    import_baseline = _run_import_baseline_subprocess(repetitions=args.import_repetitions)
    startup_baseline = stable_import_baseline_metrics(import_baseline)
    if str(probe.get("reason")) != "ok":
        payload = {
            "status": "skipped",
            "reason": str(probe.get("reason") or "preflight_failed"),
            "error_type": str(probe.get("err_type") or "preflight_failed"),
            "probe": probe,
            "runtime": {
                "mine_url": str(args.mine_url),
                "rows_target": int(args.rows_target),
                "page_size": int(args.page_size),
                "workers": int(args.workers),
                "transport_mode": str(args.transport_mode),
                "query_root": str(args.query_root),
                "query_views": parse_csv_tokens(args.query_views),
                "query_joins": parse_csv_tokens(args.query_joins),
            },
            "import_baseline": import_baseline,
            "startup_baseline": startup_baseline,
            "parallel_throughput_curve": {
                "curve_name": "workers_vs_throughput",
                "status": "not_applicable",
                "reason": "preflight did not pass",
                "points": [],
            },
            "memory_envelope_curve": {
                "curve_name": "inflight_bytes_vs_peak_rss",
                "status": "not_applicable",
                "reason": "preflight did not pass",
                "points": [],
            },
            "tor_stability": _tor_stability_payload(
                transport_mode=str(args.transport_mode),
                proxy_url=args.tor_proxy_url,
                socket_monitor=None,
            ),
        }
        attach_metric_fields(
            payload,
            startup=_STARTUP,
            status="skipped",
            error_type=str(probe.get("err_type") or "preflight_failed"),
            tor_mode=str(args.transport_mode),
            proxy_url_scheme=proxy_scheme,
            profile_name=PROFILE_NAME,
        )
        output = json.dumps(payload, sort_keys=True, default=str)
        if args.json_out:
            out_path = Path(str(args.json_out))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output + "\n", encoding="utf-8")
        print(output, flush=True)
        return SKIP_EXIT_CODE

    try:
        with SocketMonitor(sample_interval_seconds=0.1) as socket_monitor:
            result = _run_mode(
                mode=mode,
                mine_url=str(args.mine_url),
                rows_target=int(args.rows_target),
                page_size=int(args.page_size),
                workers=int(args.workers),
                legacy_batch_size=int(args.page_size),
                parallel_window_factor=2,
                auto_chunking=True,
                chunk_target_seconds=2.0,
                chunk_min_pages=1,
                chunk_max_pages=64,
                ordered_mode="unordered",
                ordered_window_pages=10,
                parallel_profile="large_query",
                large_query_mode=True,
                prefetch=None,
                inflight_limit=None,
                max_inflight_bytes_estimate=None,
                sleep_seconds=0.0,
                max_retries=int(args.max_retries),
                query_root_class=str(args.query_root),
                query_views=parse_csv_tokens(args.query_views),
                query_joins=parse_csv_tokens(args.query_joins),
                transport_mode=str(args.transport_mode),
                tor_proxy_url_value=str(args.tor_proxy_url),
                timeout_seconds=float(args.request_timeout_seconds),
            )
        socket_payload = socket_monitor.as_dict()
        payload: dict[str, Any] = {
            "status": "ok",
            "benchmark_case": "fixed_small_fetch",
            "mode": mode,
            "probe": probe,
            "import_baseline": import_baseline,
            "startup_baseline": startup_baseline,
            "runtime": {
                "mine_url": str(args.mine_url),
                "rows_target": int(args.rows_target),
                "page_size": int(args.page_size),
                "workers": int(args.workers),
                "transport_mode": str(args.transport_mode),
                "query_root": str(args.query_root),
                "query_views": parse_csv_tokens(args.query_views),
                "query_joins": parse_csv_tokens(args.query_joins),
                "request_timeout_seconds": float(args.request_timeout_seconds),
                "max_retries": int(args.max_retries),
                "import_repetitions": int(args.import_repetitions),
            },
            "run": {
                "seconds": float(result.seconds),
                "rows": int(result.rows),
                "rows_per_s": float(result.rows_per_s),
                "retries": int(result.retries),
                "available_rows_per_pass": result.available_rows_per_pass,
                "effective_workers": result.effective_workers,
                "block_stats": result.block_stats,
                "stage_timings": result.stage_timings,
            },
            "parallel_throughput_curve": {
                "curve_name": "workers_vs_throughput",
                "x_axis": "workers",
                "y_axis": "rows_per_s",
                "status": "ok",
                "points": [
                    {
                        "workers": int(args.workers),
                        "rows_per_s": float(result.rows_per_s),
                        "elapsed_s": float(result.seconds),
                        "status": "ok",
                    }
                ],
            },
            "memory_envelope_curve": {
                "curve_name": "inflight_bytes_vs_peak_rss",
                "x_axis": "max_inflight_bytes_estimate",
                "y_axis": "peak_rss_bytes",
                "status": "ok",
                "points": [
                    {
                        "max_inflight_bytes_estimate": None,
                        "peak_rss_bytes": None,
                        "status": "ok",
                    }
                ],
            },
            "tor_stability": _tor_stability_payload(
                transport_mode=str(args.transport_mode),
                proxy_url=args.tor_proxy_url,
                socket_monitor=socket_payload,
            ),
            "socket_monitor": socket_payload,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "python": sys.version,
        }
        attach_metric_fields(
            payload,
            startup=_STARTUP,
            status="ok",
            error_type="none",
            tor_mode=str(args.transport_mode),
            proxy_url_scheme=proxy_scheme,
            profile_name=PROFILE_NAME,
        )
        memory_curve = payload.get("memory_envelope_curve")
        if isinstance(memory_curve, dict):
            points = memory_curve.get("points")
            if isinstance(points, list) and points:
                first_point = points[0]
                if isinstance(first_point, dict):
                    first_point["peak_rss_bytes"] = payload.get("max_rss_bytes")
        output = json.dumps(payload, sort_keys=True, default=str)
        if args.json_out:
            out_path = Path(str(args.json_out))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output + "\n", encoding="utf-8")
        print(output, flush=True)
        return SUCCESS_EXIT_CODE
    except Exception as exc:
        payload = {
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "probe": probe,
            "import_baseline": import_baseline,
            "startup_baseline": startup_baseline,
            "parallel_throughput_curve": {
                "curve_name": "workers_vs_throughput",
                "status": "failed",
                "points": [],
            },
            "memory_envelope_curve": {
                "curve_name": "inflight_bytes_vs_peak_rss",
                "status": "failed",
                "points": [],
            },
            "tor_stability": _tor_stability_payload(
                transport_mode=str(args.transport_mode),
                proxy_url=args.tor_proxy_url,
                socket_monitor=None,
            ),
        }
        attach_metric_fields(
            payload,
            startup=_STARTUP,
            status="failed",
            error_type=type(exc).__name__,
            tor_mode=str(args.transport_mode),
            proxy_url_scheme=proxy_scheme,
            profile_name=PROFILE_NAME,
        )
        output = json.dumps(payload, sort_keys=True, default=str)
        if args.json_out:
            out_path = Path(str(args.json_out))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output + "\n", encoding="utf-8")
        print(output, flush=True)
        return FAIL_EXIT_CODE


def main() -> int:
    try:
        return run()
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
            tor_mode="unknown",
            proxy_url_scheme="none",
            profile_name=PROFILE_NAME,
        )
        print(json.dumps(payload, sort_keys=True), flush=True)
        return FAIL_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
