#!/usr/bin/env python3
"""Collect phase-0 production baselines for intermine314.

Baselines captured:
- package import latency (cold-process repetitions)
- peak RSS during export workload (per-mode isolated subprocess)
- export rows/sec
- intermine314 log event volume (count + bytes)

Modes:
- direct
- tor
- both

Exit codes:
- 0: at least one mode completed and report generated
- 2: all requested modes skipped due to environment constraints
- 1: failure
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import resource
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.bench_targeting import get_target_defaults, load_target_config, normalize_target_settings
from benchmarks.bench_utils import normalize_string_list
from benchmarks.runners.runner_metrics import (
    attach_metric_fields,
    measure_startup,
    proxy_url_scheme_from_url,
)
from intermine314.export.fetch import fetch_from_mine
from intermine314.service.tor import tor_proxy_url
from intermine314.service.transport import PROXY_URL_ENV_VAR, build_session
from intermine314.service.urls import normalize_service_root

_STARTUP = measure_startup()

SUCCESS_EXIT_CODE = 0
FAIL_EXIT_CODE = 1
SKIP_EXIT_CODE = 2

VALID_MODES = ("direct", "tor", "both")
DEFAULT_MINE_URL = "https://maizemine.rnet.missouri.edu/maizemine"
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
DEFAULT_IMPORT_REPETITIONS = 5
DEFAULT_ROWS_TARGET = 100_000
DEFAULT_PREFLIGHT_TIMEOUT_SECONDS = 8.0
DEFAULT_PARQUET_COMPRESSION = "zstd"
DEFAULT_LOG_LEVEL = "INFO"
IMPORT_SNIPPET = (
    "import json,sys,time,tracemalloc;"
    "tracemalloc.start();"
    "t0=time.perf_counter();"
    "import intermine314;"
    "elapsed=time.perf_counter()-t0;"
    "cur,peak=tracemalloc.get_traced_memory();"
    "print(json.dumps({'seconds':elapsed,'module_count':len(sys.modules),'tracemalloc_peak_bytes':peak}))"
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


def _normalize_mode_sequence(mode: str) -> tuple[str, ...]:
    value = str(mode).strip().lower()
    if value == "both":
        return ("direct", "tor")
    if value in {"direct", "tor"}:
        return (value,)
    raise ValueError(f"mode must be one of: {', '.join(VALID_MODES)}")


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


def _service_version_url(mine_url: str) -> str:
    normalized = normalize_service_root(mine_url)
    return normalized.rstrip("/") + "/version/ws"


def _probe_direct(mine_url: str, timeout_seconds: float) -> dict[str, Any]:
    t0 = time.perf_counter()
    normalized = normalize_service_root(mine_url)
    parsed = urlparse(normalized)
    host = parsed.hostname or "<unknown>"
    scheme = (parsed.scheme or "").lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    result = {
        "mode": "direct",
        "host": host,
        "reason": "ok",
        "err_type": "none",
        "elapsed_s": 0.0,
    }
    try:
        socket.getaddrinfo(host, port)
    except Exception as exc:
        result["reason"] = "dns_failed"
        result["err_type"] = type(exc).__name__
        result["elapsed_s"] = time.perf_counter() - t0
        return result

    try:
        with requests.get(_service_version_url(normalized), timeout=timeout_seconds, stream=True) as response:
            if int(response.status_code) >= 400:
                result["reason"] = "connect_failed"
                result["err_type"] = f"http_{int(response.status_code)}"
    except Exception as exc:
        result["reason"] = "connect_failed"
        result["err_type"] = type(exc).__name__
    result["elapsed_s"] = time.perf_counter() - t0
    return result


def _probe_tor(mine_url: str, timeout_seconds: float) -> dict[str, Any]:
    t0 = time.perf_counter()
    normalized = normalize_service_root(mine_url)
    host = urlparse(normalized).hostname or "<unknown>"
    result = {
        "mode": "tor",
        "host": host,
        "reason": "ok",
        "err_type": "none",
        "elapsed_s": 0.0,
        "proxy_url": tor_proxy_url(),
    }
    session = build_session(proxy_url=result["proxy_url"], user_agent="intermine314-phase0-baseline")
    try:
        with session.get(_service_version_url(normalized), timeout=timeout_seconds, stream=True) as response:
            if int(response.status_code) >= 400:
                result["reason"] = "connect_failed"
                result["err_type"] = f"http_{int(response.status_code)}"
    except Exception as exc:
        result["reason"] = "proxy_failed"
        result["err_type"] = type(exc).__name__
    finally:
        try:
            session.close()
        except Exception:
            pass
    result["elapsed_s"] = time.perf_counter() - t0
    return result


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


def _pythonpath_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    entries = [str(SRC)]
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


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
        payload = json.loads(line)
        runs.append(payload)
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
    ]
    if args.max_workers is not None:
        cmd.extend(["--max-workers", str(args.max_workers)])
    if args.query_root:
        cmd.extend(["--query-root", str(args.query_root)])
    if args.query_views:
        cmd.extend(["--query-views", str(args.query_views)])
    if args.query_joins:
        cmd.extend(["--query-joins", str(args.query_joins)])
    if args.workflow == "etl":
        cmd.extend(["--etl-guardrail-rows", str(args.etl_guardrail_rows)])
        if args.allow_large_etl:
            cmd.append("--allow-large-etl")
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


def _proxy_url_scheme_for_mode(mode: str, *, probe: dict[str, Any] | None = None) -> str:
    if str(mode).strip().lower() != "tor":
        return "none"
    probe_proxy = None
    if isinstance(probe, dict):
        probe_proxy = probe.get("proxy_url")
    return proxy_url_scheme_from_url(probe_proxy or tor_proxy_url())


def _report_proxy_url_scheme(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized == "tor":
        return proxy_url_scheme_from_url(tor_proxy_url())
    if normalized == "both":
        return "mixed"
    return "none"


def _worker_export(args: argparse.Namespace) -> int:
    settings = _resolve_workload_settings(args)
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
        os.environ[PROXY_URL_ENV_VAR] = tor_proxy_url()
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
            export = fetch_from_mine(
                mine_url=settings.mine_url,
                root_class=settings.root_class,
                views=settings.views,
                joins=settings.joins,
                start=0,
                size=args.rows_target,
                page_size=args.page_size,
                workflow=args.workflow,
                max_workers=args.max_workers,
                ordered=args.ordered,
                ordered_window_pages=args.ordered_window_pages,
                parquet_path=parquet_path if args.workflow == "elt" else None,
                parquet_compression=args.parquet_compression,
                duckdb_database=":memory:",
                duckdb_table=args.duckdb_table,
                etl_guardrail_rows=args.etl_guardrail_rows,
                allow_large_etl=args.allow_large_etl,
            )
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
                "elapsed_s": elapsed,
                "rows_per_s": (float(row_count) / elapsed) if row_count is not None and elapsed > 0 else None,
                "peak_rss_bytes": peak_rss,
                "parquet_bytes": parquet_bytes,
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
            "ordered": args.ordered,
            "ordered_window_pages": args.ordered_window_pages,
            "parquet_compression": args.parquet_compression,
            "log_level": args.log_level,
        },
    }
    report["import_baseline"] = _run_import_baseline_subprocess(repetitions=args.import_repetitions)

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
    report["summary"] = {
        "modes_requested": list(_normalize_mode_sequence(args.mode)),
        "modes_succeeded": success_count,
        "modes_skipped": skip_count,
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
    parser.add_argument("--workflow", choices=("elt", "etl"), default="elt")
    parser.add_argument("--ordered", default="unordered")
    parser.add_argument("--ordered-window-pages", type=int, default=10)
    parser.add_argument("--parquet-compression", default=DEFAULT_PARQUET_COMPRESSION)
    parser.add_argument("--duckdb-table", default="results")
    parser.add_argument("--etl-guardrail-rows", type=int, default=50_000)
    parser.add_argument("--allow-large-etl", action="store_true")
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
    if args.etl_guardrail_rows <= 0:
        raise ValueError("--etl-guardrail-rows must be a positive integer")

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
