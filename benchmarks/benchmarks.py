#!/usr/bin/env python3
"""Benchmark entrypoint for a fixed 5-point row matrix with direct+tor comparisons."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.bench_constants import (
    AUTO_WORKER_TOKENS,
    DEFAULT_BENCHMARK_MINE_URL,
    MATRIX_ROWS,
    rows_to_csv,
)
from benchmarks.bench_fetch import resolve_execution_plan, run_fetch_phase
from benchmarks.bench_targeting import (
    get_target_defaults,
    load_target_config,
    normalize_target_settings,
    resolve_benchmark_profile,
)
from benchmarks.bench_utils import parse_csv_tokens
from intermine314.config.runtime_defaults import get_runtime_defaults
from intermine314.service.transport import default_tor_proxy_url

_RUNTIME_DEFAULTS = get_runtime_defaults()
_DEFAULT_PAGE_SIZE = int(_RUNTIME_DEFAULTS.query_defaults.default_parallel_page_size)
_DEFAULT_MATRIX_ROWS_TEXT = rows_to_csv(MATRIX_ROWS)
_DEFAULT_REPETITIONS = 3
_DEFAULT_BENCHMARK_PROFILE = "auto"
_VALID_BENCHMARK_PROFILES = {"auto", "server_restricted", "non_restricted"}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _optional_positive_int(value: str) -> int | None:
    token = str(value).strip().lower()
    if token in {"", "none", "null", "auto"}:
        return None
    return _positive_int(token)


def _parse_matrix_rows(value: str, *, required_count: int = 5) -> list[int]:
    rows = [int(token) for token in parse_csv_tokens(value)]
    if len(rows) != int(required_count):
        raise argparse.ArgumentTypeError(f"--matrix-rows must contain exactly {required_count} values")
    if any(row <= 0 for row in rows):
        raise argparse.ArgumentTypeError("--matrix-rows values must be positive integers")
    return rows


def _parse_transport_modes(value: str) -> list[str]:
    modes: list[str] = []
    for token in parse_csv_tokens(value):
        mode = str(token).strip().lower()
        if mode not in {"direct", "tor"}:
            raise argparse.ArgumentTypeError("--transport-modes values must be direct and/or tor")
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise argparse.ArgumentTypeError("--transport-modes resolved to an empty list")
    return modes


def _resolve_query_views(value: str) -> list[str]:
    views = [token for token in parse_csv_tokens(value)]
    if not views:
        raise ValueError("--query-views resolved to an empty list")
    return views


def _normalize_benchmark_profile(value: str) -> str:
    token = str(value or "").strip().lower()
    if token in _VALID_BENCHMARK_PROFILES:
        return token
    raise argparse.ArgumentTypeError(
        "--benchmark-profile must be one of: auto, server_restricted, non_restricted"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mine-url", default=DEFAULT_BENCHMARK_MINE_URL)
    parser.add_argument("--benchmark-target", default="auto")
    parser.add_argument("--matrix-rows", default=_DEFAULT_MATRIX_ROWS_TEXT)
    parser.add_argument("--repetitions", type=_positive_int, default=_DEFAULT_REPETITIONS)
    parser.add_argument("--page-size", type=_positive_int, default=_DEFAULT_PAGE_SIZE)
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--benchmark-profile", type=_normalize_benchmark_profile, default=_DEFAULT_BENCHMARK_PROFILE)
    parser.add_argument("--query-root", default="Gene")
    parser.add_argument("--query-views", default="Gene.primaryIdentifier")
    parser.add_argument("--query-joins", default="")
    parser.add_argument("--transport-modes", default="direct,tor")
    parser.add_argument("--tor-proxy-url", default=default_tor_proxy_url())
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=_positive_int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--legacy-batch-size", type=_positive_int, default=_DEFAULT_PAGE_SIZE)
    parser.add_argument("--parallel-window-factor", type=_positive_int, default=2)
    parser.add_argument("--auto-chunking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk-target-seconds", type=float, default=2.0)
    parser.add_argument("--chunk-min-pages", type=_positive_int, default=1)
    parser.add_argument("--chunk-max-pages", type=_positive_int, default=64)
    parser.add_argument("--ordered-mode", default="unordered")
    parser.add_argument("--ordered-window-pages", type=_positive_int, default=10)
    parser.add_argument("--parallel-profile", default="default")
    parser.add_argument("--large-query-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch", type=_optional_positive_int, default=None)
    parser.add_argument("--inflight-limit", type=_optional_positive_int, default=None)
    parser.add_argument("--max-inflight-bytes-estimate", type=_optional_positive_int, default=None)
    parser.add_argument("--randomize-mode-order", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--storage-compare",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compare intermine+CSV+pandas against intermine314+Parquet+DuckDB/Polars.",
    )
    parser.add_argument(
        "--storage-output-dir",
        default="benchmark-results/storage_compare",
        help="Output directory for storage comparison artifacts.",
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)
    args.matrix_rows = _parse_matrix_rows(str(args.matrix_rows), required_count=5)
    args.transport_modes = _parse_transport_modes(str(args.transport_modes))
    if args.sleep_seconds < 0:
        raise ValueError("--sleep-seconds must be >= 0")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be > 0")
    if args.chunk_target_seconds <= 0:
        raise ValueError("--chunk-target-seconds must be > 0")
    if args.chunk_max_pages < args.chunk_min_pages:
        raise ValueError("--chunk-max-pages must be >= --chunk-min-pages")
    return args


def _resolve_target(args: argparse.Namespace) -> tuple[str, dict[str, Any] | None]:
    target_config = load_target_config()
    target_defaults = get_target_defaults(target_config)
    target_settings = normalize_target_settings(args.benchmark_target, target_config, target_defaults)
    if target_settings is None:
        return str(args.mine_url), None
    endpoint = str(target_settings.get("endpoint", "")).strip()
    if endpoint:
        return endpoint, target_settings
    return str(args.mine_url), target_settings


def _resolve_explicit_workers(args: argparse.Namespace) -> list[int]:
    workers = parse_csv_tokens(str(args.workers))
    if str(args.workers).strip().lower() in AUTO_WORKER_TOKENS:
        return []
    parsed = [int(token) for token in workers]
    if not parsed or any(value <= 0 for value in parsed):
        raise ValueError("--workers must be auto or a comma-separated list of positive integers")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mine_url, target_settings = _resolve_target(args)
    query_views = _resolve_query_views(str(args.query_views))
    query_joins = [token for token in parse_csv_tokens(str(args.query_joins))]
    explicit_workers = _resolve_explicit_workers(args)

    benchmark_profile = resolve_benchmark_profile(str(args.benchmark_profile), target_settings)
    server_restricted = None
    if target_settings is not None and "server_restricted" in target_settings:
        server_restricted = bool(target_settings.get("server_restricted"))

    results_by_transport: dict[str, Any] = {}
    storage_compare_by_transport: dict[str, Any] = {}
    for transport_mode in list(args.transport_modes):
        args.transport_mode = str(transport_mode)
        transport_payload: dict[str, Any] = {}
        transport_storage_payload: dict[str, Any] = {}
        for rows_target in list(args.matrix_rows):
            phase_plan = resolve_execution_plan(
                mine_url=mine_url,
                rows_target=int(rows_target),
                explicit_workers=explicit_workers,
                benchmark_profile=benchmark_profile,
                phase_default_include_legacy=True,
                server_restricted=server_restricted,
            )
            fetch_result = run_fetch_phase(
                phase_name=f"fetch_{transport_mode}_{rows_target}",
                mine_url=mine_url,
                rows_target=int(rows_target),
                repetitions=int(args.repetitions),
                phase_plan=phase_plan,
                args=args,
                page_size=int(args.page_size),
                query_root_class=str(args.query_root),
                query_views=query_views,
                query_joins=query_joins,
            )
            transport_payload[str(rows_target)] = {
                "phase_plan": phase_plan,
                "fetch": fetch_result,
            }

            if bool(args.storage_compare):
                from benchmarks.bench_storage_compare import run_storage_compare

                try:
                    compare_workers = max(int(value) for value in phase_plan.get("workers", []) if int(value) > 0)
                except Exception:
                    compare_workers = None
                try:
                    transport_storage_payload[str(rows_target)] = run_storage_compare(
                        mine_url=mine_url,
                        rows_target=int(rows_target),
                        page_size=int(args.page_size),
                        workers=compare_workers,
                        query_root_class=str(args.query_root),
                        query_views=query_views,
                        query_joins=query_joins,
                        transport_mode=str(transport_mode),
                        tor_proxy_url_value=str(args.tor_proxy_url),
                        output_dir=Path(str(args.storage_output_dir)) / str(transport_mode) / f"rows_{rows_target}",
                        timeout_seconds=float(args.timeout_seconds),
                        max_retries=int(args.max_retries),
                        repetitions=int(args.repetitions),
                    )
                except Exception as exc:
                    transport_storage_payload[str(rows_target)] = {
                        "schema_version": "legacy_storage_compare_v2",
                        "transport_mode": str(transport_mode),
                        "rows_target": int(rows_target),
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }

        results_by_transport[str(transport_mode)] = transport_payload
        if bool(args.storage_compare):
            storage_compare_by_transport[str(transport_mode)] = transport_storage_payload

    payload: dict[str, Any] = {
        "schema_version": "benchmark_matrix_v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "mine_url": mine_url,
            "matrix_rows": [int(row) for row in args.matrix_rows],
            "repetitions": int(args.repetitions),
            "page_size": int(args.page_size),
            "workers_arg": str(args.workers),
            "benchmark_profile": str(benchmark_profile),
            "server_restricted": server_restricted,
            "transport_modes": list(args.transport_modes),
            "query_root": str(args.query_root),
            "query_views": query_views,
            "query_joins": query_joins,
            "timeout_seconds": float(args.timeout_seconds),
            "max_retries": int(args.max_retries),
            "max_inflight_bytes_estimate": args.max_inflight_bytes_estimate,
            "storage_compare": bool(args.storage_compare),
            "storage_output_dir": str(args.storage_output_dir),
        },
        "results_by_transport": results_by_transport,
    }
    if bool(args.storage_compare):
        payload["storage_compare_by_transport"] = storage_compare_by_transport
    output = json.dumps(payload, indent=2, sort_keys=True)
    print(output, flush=True)
    if args.json_out:
        out_path = Path(str(args.json_out))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
