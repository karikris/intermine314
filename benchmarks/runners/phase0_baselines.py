#!/usr/bin/env python3
"""Collect phase-0 production baseline metrics for direct/Tor export runs."""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.bench_targeting import get_target_defaults, load_target_config, normalize_target_settings
from benchmarks.bench_constants import (
    DEFAULT_BENCHMARK_MINE_URL,
    DEFAULT_PARQUET_COMPRESSION,
    DEFAULT_RUNNER_IMPORT_REPETITIONS,
    DEFAULT_RUNNER_LOG_LEVEL,
    DEFAULT_RUNNER_PREFLIGHT_TIMEOUT_SECONDS,
)
from benchmarks.bench_utils import normalize_string_list
from benchmarks.runners.common import (
    SocketMonitor,
    TOR_DNS_SAFETY_POLICY,
    TOR_GUARDRAIL_UNSAFE_PROXY_URL,
    now_utc_iso,
    probe_direct,
    probe_tor,
    pythonpath_env,
    ru_maxrss_bytes,
    run_import_baseline_subprocess,
    stable_import_baseline_metrics,
    validate_tor_proxy_url,
)
from benchmarks.runners.runner_metrics import (
    attach_metric_fields,
    measure_startup,
    proxy_url_scheme_from_url,
)

from intermine314.export.fetch import fetch_from_mine
from intermine314.export.resource_profile import resolve_resource_profile
from intermine314.service.transport import default_tor_proxy_url
from intermine314.service.transport import PROXY_URL_ENV_VAR

_STARTUP = measure_startup()

SUCCESS_EXIT_CODE = 0
FAIL_EXIT_CODE = 1
SKIP_EXIT_CODE = 2

VALID_MODES = ("direct", "tor", "both")
DEFAULT_MINE_URL = DEFAULT_BENCHMARK_MINE_URL
DEFAULT_QUERY_ROOT_CLASS = "Gene"
DEFAULT_QUERY_VIEWS = (
    "Gene.primaryIdentifier",
    "Gene.secondaryIdentifier",
    "Gene.symbol",
    "Gene.name",
    "Gene.briefDescription",
    "Gene.organism.name",
    "Gene.organism.taxonId",
)
DEFAULT_QUERY_JOINS: tuple[str, ...] = ()
DEFAULT_IMPORT_REPETITIONS = DEFAULT_RUNNER_IMPORT_REPETITIONS
DEFAULT_ROWS_TARGET = 100_000
DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = DEFAULT_RUNNER_PREFLIGHT_TIMEOUT_SECONDS
DEFAULT_LOG_LEVEL = DEFAULT_RUNNER_LOG_LEVEL
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
        "import intermine314",
        "elapsed = time.perf_counter() - t0",
        "_cur, peak = tracemalloc.get_traced_memory()",
        "after = len(sys.modules)",
        "print(json.dumps({'seconds': elapsed, 'module_count': after, 'imported_module_count': after - before, 'tracemalloc_peak_bytes': peak, 'max_rss_bytes': _rss_bytes()}))",
    ]
)

_now_iso = now_utc_iso
_pythonpath_env = partial(pythonpath_env, source_root=SRC)
_ru_maxrss_bytes = ru_maxrss_bytes
_probe_direct = probe_direct
_probe_tor = lambda mine_url, timeout_seconds: probe_tor(  # noqa: E731
    mine_url,
    timeout_seconds,
    context="phase0 preflight proxy_url",
    user_agent="intermine314-phase0-baseline",
    proxy_url=default_tor_proxy_url(),
)
_run_import_baseline_subprocess = partial(
    run_import_baseline_subprocess,
    import_snippet=IMPORT_SNIPPET,
    source_root=SRC,
)


class _IntermineOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return str(record.name).startswith("intermine314")


class _LogVolumeHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records = 0
        self.bytes = 0
        self.level_counts: dict[str, int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        self.records += 1
        self.bytes += len(str(message).encode("utf-8", errors="replace"))
        level = str(record.levelname).upper()
        self.level_counts[level] = int(self.level_counts.get(level, 0)) + 1


@dataclass(frozen=True)
class WorkloadSettings:
    mine_url: str
    root_class: str
    views: list[str]
    joins: list[str]


def _normalize_mode_sequence(mode: str) -> tuple[str, ...]:
    value = str(mode).strip().lower()
    if value == "both":
        return ("direct", "tor")
    if value in {"direct", "tor"}:
        return (value,)
    raise ValueError(f"mode must be one of: {', '.join(VALID_MODES)}")


def _resolve_workload_settings(args: argparse.Namespace) -> WorkloadSettings:
    target_config = load_target_config()
    target_defaults = get_target_defaults(target_config)
    target_settings = normalize_target_settings(args.benchmark_target, target_config, target_defaults)

    mine_url = str(args.mine_url).strip()
    if mine_url == DEFAULT_MINE_URL and target_settings is not None:
        endpoint = str(target_settings.get("endpoint", "")).strip()
        if endpoint:
            mine_url = endpoint

    root_class = str(args.query_root).strip() if args.query_root else ""
    if not root_class and target_settings is not None:
        root_class = str(target_settings.get("root_class", "")).strip()
    if not root_class:
        root_class = DEFAULT_QUERY_ROOT_CLASS

    views = normalize_string_list(args.query_views)
    if not views and target_settings is not None:
        views = normalize_string_list(target_settings.get("views"))
    if not views:
        views = list(DEFAULT_QUERY_VIEWS)

    joins = normalize_string_list(args.query_joins)
    if not joins and target_settings is not None:
        joins = normalize_string_list(target_settings.get("joins"))
    if not joins:
        joins = list(DEFAULT_QUERY_JOINS)

    return WorkloadSettings(
        mine_url=mine_url,
        root_class=root_class,
        views=views,
        joins=joins,
    )


def _build_worker_command(args: argparse.Namespace, mode: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-export",
        "--mode",
        mode,
        "--mine-url",
        args.mine_url,
        "--benchmark-target",
        args.benchmark_target,
        "--rows-target",
        str(args.rows_target),
        "--workflow",
        args.workflow,
        "--page-size",
        str(args.page_size),
        "--ordered",
        args.ordered,
        "--parquet-compression",
        args.parquet_compression,
        "--preflight-timeout-seconds",
        str(args.preflight_timeout_seconds),
        "--log-level",
        args.log_level,
        "--duckdb-table",
        args.duckdb_table,
        "--resource-profile",
        str(args.resource_profile),
    ]
    if args.max_workers is not None:
        cmd.extend(["--max-workers", str(args.max_workers)])
    if args.max_inflight_bytes_estimate is not None:
        cmd.extend(["--max-inflight-bytes-estimate", str(args.max_inflight_bytes_estimate)])
    if args.temp_dir is not None:
        cmd.extend(["--temp-dir", str(args.temp_dir)])
    if args.temp_dir_min_free_bytes is not None:
        cmd.extend(["--temp-dir-min-free-bytes", str(args.temp_dir_min_free_bytes)])
    if args.query_root:
        cmd.extend(["--query-root", str(args.query_root)])
    if args.query_views:
        cmd.extend(["--query-views", str(args.query_views)])
    if args.query_joins:
        cmd.extend(["--query-joins", str(args.query_joins)])
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
    if proc.returncode != 0 and proc.returncode != SKIP_EXIT_CODE:
        payload["stderr"] = proc.stderr.strip()
    return proc.returncode, payload


def _parse_int_or_none(value: str) -> int | None:
    text = str(value).strip().lower()
    if text in {"", "none", "null", "auto"}:
        return None
    parsed = int(text)
    if parsed <= 0:
        raise ValueError("value must be a positive integer or auto")
    return parsed


def _parse_non_negative_int_or_none(value: str) -> int | None:
    text = str(value).strip().lower()
    if text in {"", "none", "null", "auto"}:
        return None
    parsed = int(text)
    if parsed < 0:
        raise ValueError("value must be a non-negative integer or auto")
    return parsed


def _proxy_url_scheme_for_mode(mode: str, *, probe: dict[str, Any] | None = None) -> str:
    if str(mode).strip().lower() != "tor":
        return "none"
    probe_proxy = None
    if isinstance(probe, dict):
        probe_proxy = probe.get("proxy_url")
    return proxy_url_scheme_from_url(probe_proxy or default_tor_proxy_url())


def _report_proxy_url_scheme(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized == "tor":
        return proxy_url_scheme_from_url(default_tor_proxy_url())
    if normalized == "both":
        return "mixed"
    return "none"


def _tor_stability_payload(
    *,
    mode: str,
    probe: dict[str, Any],
    socket_monitor: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode != "tor":
        return {
            "status": "not_applicable",
            "reason": "mode is not tor",
            "socket_monitor": socket_monitor
            or {"status": "not_applicable", "reason": "tor mode disabled"},
        }
    proxy_url = str(probe.get("proxy_url") or default_tor_proxy_url())
    scheme = proxy_url_scheme_from_url(proxy_url)
    unsafe_proxy_rejected = False
    unsafe_error_type = "none"
    try:
        validate_tor_proxy_url(TOR_GUARDRAIL_UNSAFE_PROXY_URL, context="phase0 baseline unsafe-proxy check")
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
        "tor_proxy_scheme": scheme,
        "unsafe_proxy_rejected": unsafe_proxy_rejected,
        "unsafe_proxy_rejection_error_type": unsafe_error_type,
        "socket_monitor": socket_payload,
        "socket_leak_suspected": leak_suspected,
        "tor_dns_safety": "enforced" if unsafe_proxy_rejected and scheme == "socks5h" else "violated",
        "operational_validation": {
            "method": "tor_testsocks_log_observation",
            "note": "Enable Tor TestSocks and inspect Tor logs during tor-mode fetch runs.",
        },
    }


def _worker_export(args: argparse.Namespace) -> int:
    settings = _resolve_workload_settings(args)
    resolved_resource_profile = resolve_resource_profile(args.resource_profile)
    probe = _probe_tor(settings.mine_url, args.preflight_timeout_seconds) if args.mode == "tor" else _probe_direct(
        settings.mine_url, args.preflight_timeout_seconds
    )
    if probe.get("reason") != "ok":
        payload = {
            "mode": args.mode,
            "status": "skipped",
            "reason": probe.get("reason"),
            "err_type": probe.get("err_type"),
            "probe": probe,
            "tor_stability": _tor_stability_payload(mode=args.mode, probe=probe, socket_monitor=None),
        }
        attach_metric_fields(
            payload,
            startup=_STARTUP,
            status="skipped",
            error_type=str(probe.get("err_type") or "preflight_failed"),
            tor_mode=args.mode,
            proxy_url_scheme=_proxy_url_scheme_for_mode(args.mode, probe=probe),
            profile_name="auto",
        )
        print(
            json.dumps(
                payload,
                sort_keys=True,
            ),
            flush=True,
        )
        return SKIP_EXIT_CODE

    prior_proxy = os.environ.get(PROXY_URL_ENV_VAR)
    if args.mode == "tor":
        os.environ[PROXY_URL_ENV_VAR] = validate_tor_proxy_url(
            default_tor_proxy_url(),
            context="phase0 worker proxy_url",
        )
    else:
        os.environ.pop(PROXY_URL_ENV_VAR, None)

    root_logger = logging.getLogger()
    intermine_logger = logging.getLogger("intermine314")
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    prior_root_level = root_logger.level
    prior_intermine_level = intermine_logger.level
    handler = _LogVolumeHandler()
    handler.setLevel(level)
    handler.addFilter(_IntermineOnlyFilter())
    root_logger.addHandler(handler)
    if root_logger.level > level:
        root_logger.setLevel(level)
    intermine_logger.setLevel(level)

    try:
        with TemporaryDirectory(prefix="intermine314-phase0-") as tmp:
            parquet_path = Path(tmp) / "baseline.parquet"
            started = time.perf_counter()
            with SocketMonitor(sample_interval_seconds=0.1) as socket_monitor:
                export = fetch_from_mine(
                    mine_url=settings.mine_url,
                    root_class=settings.root_class,
                    views=settings.views,
                    joins=settings.joins,
                    start=0,
                    size=args.rows_target,
                    page_size=args.page_size,
                    workflow=args.workflow,
                    resource_profile=resolved_resource_profile,
                    max_workers=args.max_workers,
                    ordered=args.ordered,
                    max_inflight_bytes_estimate=args.max_inflight_bytes_estimate,
                    ordered_window_pages=args.ordered_window_pages,
                    parquet_path=parquet_path,
                    parquet_compression=args.parquet_compression,
                    temp_dir=args.temp_dir,
                    temp_dir_min_free_bytes=args.temp_dir_min_free_bytes,
                    duckdb_database=":memory:",
                    duckdb_table=args.duckdb_table,
                )
            socket_payload = socket_monitor.as_dict()
            elapsed = time.perf_counter() - started
            con = export.get("duckdb_connection")
            row_count = None
            try:
                table = str(args.duckdb_table)
                fetched = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
                if fetched:
                    row_count = int(fetched[0])
            finally:
                try:
                    con.close()
                except Exception:
                    pass

            peak_rss = _ru_maxrss_bytes()
            parquet_bytes = None
            parquet_path_str = export.get("parquet_path")
            if parquet_path_str:
                path_obj = Path(str(parquet_path_str))
                if path_obj.exists() and path_obj.is_file():
                    parquet_bytes = int(path_obj.stat().st_size)
            production_plan = export.get("production_plan")
            if isinstance(production_plan, dict):
                profile_name = str(production_plan.get("name", "auto"))
            else:
                profile_name = "auto"
            payload = {
                "mode": args.mode,
                "status": "ok",
                "probe": probe,
                "workflow": args.workflow,
                "mine_url": settings.mine_url,
                "root_class": settings.root_class,
                "views_count": len(settings.views),
                "joins_count": len(settings.joins),
                "rows_target": args.rows_target,
                "rows_exported": row_count,
                "resource_profile": resolved_resource_profile.name,
                "elapsed_s": elapsed,
                "rows_per_s": (float(row_count) / elapsed) if row_count is not None and elapsed > 0 else None,
                "peak_rss_bytes": peak_rss,
                "parquet_bytes": parquet_bytes,
                "socket_monitor": socket_payload,
                "tor_stability": _tor_stability_payload(
                    mode=args.mode,
                    probe=probe,
                    socket_monitor=socket_payload,
                ),
                "log_volume": {
                    "records": handler.records,
                    "bytes": handler.bytes,
                    "levels": dict(handler.level_counts),
                },
            }
            attach_metric_fields(
                payload,
                startup=_STARTUP,
                status="ok",
                error_type="none",
                tor_mode=args.mode,
                proxy_url_scheme=_proxy_url_scheme_for_mode(args.mode, probe=probe),
                profile_name=profile_name,
            )
            print(json.dumps(payload, sort_keys=True, default=str), flush=True)
            return SUCCESS_EXIT_CODE
    except Exception as exc:
        payload = {
            "mode": args.mode,
            "status": "failed",
            "probe": probe,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "tor_stability": _tor_stability_payload(mode=args.mode, probe=probe, socket_monitor=None),
        }
        attach_metric_fields(
            payload,
            startup=_STARTUP,
            status="failed",
            error_type=type(exc).__name__,
            tor_mode=args.mode,
            proxy_url_scheme=_proxy_url_scheme_for_mode(args.mode, probe=probe),
            profile_name="auto",
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
        root_logger.removeHandler(handler)
        root_logger.setLevel(prior_root_level)
        intermine_logger.setLevel(prior_intermine_level)
        if prior_proxy is None:
            os.environ.pop(PROXY_URL_ENV_VAR, None)
        else:
            os.environ[PROXY_URL_ENV_VAR] = prior_proxy


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
            "mode": args.mode,
            "workflow": args.workflow,
            "mine_url": args.mine_url,
            "benchmark_target": args.benchmark_target,
            "rows_target": args.rows_target,
            "page_size": args.page_size,
            "max_workers": args.max_workers,
            "resource_profile": args.resource_profile,
            "ordered": args.ordered,
            "max_inflight_bytes_estimate": args.max_inflight_bytes_estimate,
            "ordered_window_pages": args.ordered_window_pages,
            "temp_dir": args.temp_dir,
            "temp_dir_min_free_bytes": args.temp_dir_min_free_bytes,
            "parquet_compression": args.parquet_compression,
            "log_level": args.log_level,
        },
    }
    report["import_baseline"] = _run_import_baseline_subprocess(repetitions=args.import_repetitions)
    report["startup_baseline"] = stable_import_baseline_metrics(report["import_baseline"])

    results: dict[str, Any] = {}
    success_count = 0
    skip_count = 0
    for mode in _normalize_mode_sequence(args.mode):
        code, payload = _run_mode_worker(args, mode)
        results[mode] = payload
        if code == SUCCESS_EXIT_CODE and payload.get("status") == "ok":
            success_count += 1
        elif code == SKIP_EXIT_CODE:
            skip_count += 1
    report["export_baselines"] = results
    throughput_points: list[dict[str, Any]] = []
    memory_points: list[dict[str, Any]] = []
    for mode, payload in results.items():
        if not isinstance(payload, dict):
            continue
        if str(payload.get("status")) != "ok":
            continue
        throughput_points.append(
            {
                "mode": str(mode),
                "workers": args.max_workers,
                "rows_per_s": payload.get("rows_per_s"),
                "elapsed_s": payload.get("elapsed_s"),
                "rows_exported": payload.get("rows_exported"),
                "status": "ok",
            }
        )
        memory_points.append(
            {
                "mode": str(mode),
                "max_inflight_bytes_estimate": args.max_inflight_bytes_estimate,
                "peak_rss_bytes": payload.get("peak_rss_bytes"),
                "status": "ok",
            }
        )
    report["parallel_throughput_curve"] = {
        "curve_name": "mode_point_baseline",
        "x_axis": "mode",
        "y_axis": "rows_per_s",
        "points": throughput_points,
        "status": "ok" if throughput_points else "failed",
    }
    report["memory_envelope_curve"] = {
        "curve_name": "mode_point_memory",
        "x_axis": "mode",
        "y_axis": "peak_rss_bytes",
        "points": memory_points,
        "status": "ok" if memory_points else "failed",
    }
    tor_payload = results.get("tor")
    if isinstance(tor_payload, dict):
        report["tor_stability"] = tor_payload.get(
            "tor_stability",
            _tor_stability_payload(mode="tor", probe=tor_payload.get("probe", {}), socket_monitor=None),
        )
    else:
        report["tor_stability"] = {
            "status": "not_applicable",
            "reason": "tor mode not requested",
            "socket_monitor": {"status": "not_applicable"},
        }
    report["summary"] = {
        "modes_requested": list(_normalize_mode_sequence(args.mode)),
        "modes_succeeded": success_count,
        "modes_skipped": skip_count,
        "throughput_curve_points": len(throughput_points),
        "memory_envelope_curve_points": len(memory_points),
    }

    status = "failed"
    error_type = "mode_failures"
    exit_code = FAIL_EXIT_CODE
    if success_count > 0:
        status = "ok"
        error_type = "none"
        exit_code = SUCCESS_EXIT_CODE
    elif skip_count == len(_normalize_mode_sequence(args.mode)):
        status = "skipped"
        error_type = "all_modes_skipped"
        exit_code = SKIP_EXIT_CODE

    attach_metric_fields(
        report,
        startup=_STARTUP,
        status=status,
        error_type=error_type,
        tor_mode=args.mode,
        proxy_url_scheme=_report_proxy_url_scheme(args.mode),
        profile_name="auto",
    )
    return exit_code, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect phase-0 baselines for intermine314")
    parser.add_argument("--mode", choices=VALID_MODES, default="both")
    parser.add_argument("--mine-url", default=DEFAULT_MINE_URL)
    parser.add_argument("--benchmark-target", default="auto")
    parser.add_argument("--rows-target", type=int, default=DEFAULT_ROWS_TARGET)
    parser.add_argument("--page-size", type=int, default=5000)
    parser.add_argument("--max-workers", type=_parse_int_or_none, default=None)
    parser.add_argument("--resource-profile", default="default")
    parser.add_argument("--workflow", choices=("elt",), default="elt")
    parser.add_argument("--ordered", default="unordered")
    parser.add_argument("--max-inflight-bytes-estimate", type=_parse_int_or_none, default=None)
    parser.add_argument("--ordered-window-pages", type=int, default=10)
    parser.add_argument("--temp-dir", default=None)
    parser.add_argument("--temp-dir-min-free-bytes", type=_parse_non_negative_int_or_none, default=None)
    parser.add_argument("--parquet-compression", default=DEFAULT_PARQUET_COMPRESSION)
    parser.add_argument("--duckdb-table", default="results")
    parser.add_argument("--query-root", default=None)
    parser.add_argument("--query-views", default=None)
    parser.add_argument("--query-joins", default=None)
    parser.add_argument("--import-repetitions", type=int, default=DEFAULT_IMPORT_REPETITIONS)
    parser.add_argument("--preflight-timeout-seconds", type=float, default=DEFAULT_PREFLIGHT_TIMEOUT_SECONDS)
    parser.add_argument("--log-level", default=DEFAULT_LOG_LEVEL)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--worker-export", action="store_true", help=argparse.SUPPRESS)
    return parser


def run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.rows_target <= 0:
        raise ValueError("--rows-target must be a positive integer")
    if args.page_size <= 0:
        raise ValueError("--page-size must be a positive integer")
    if args.import_repetitions <= 0:
        raise ValueError("--import-repetitions must be a positive integer")
    if args.ordered_window_pages <= 0:
        raise ValueError("--ordered-window-pages must be a positive integer")
    if args.worker_export:
        return _worker_export(args)

    exit_code, report = _build_report(args)
    if args.json_out:
        out_path = Path(str(args.json_out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True, default=str), flush=True)
    return exit_code


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
            tor_mode="unknown",
            proxy_url_scheme="none",
            profile_name="auto",
        )
        print(
            json.dumps(
                payload,
                sort_keys=True,
            ),
            flush=True,
        )
        return FAIL_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
