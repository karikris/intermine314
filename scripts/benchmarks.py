#!/usr/bin/env python3
"""Benchmark suite: intermine vs intermine314.

Features:
- 100k direct benchmark (intermine vs intermine314) with repetitions and robust stats
- 1M parallel benchmark (intermine314 only) with worker/page-size matrix support
- adaptive auto-chunking for large pulls (dynamic block sizing)
- mine-aware worker defaults via intermine314 mine registry
- environment pinning (OS/kernel/Python/package versions)
- optional per-request sleep for public service etiquette
- storage comparison: 100k intermine CSV vs 100k intermine314 Parquet
- pandas(CSV) vs polars(Parquet) dataframe benchmark on 1M intermine314 export
- randomized mode order per repetition to reduce run-order bias
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import time
import types
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def install_legacy_shims() -> None:
    """Enable legacy intermine package on Python 3.14+."""
    import collections
    import collections.abc

    if not hasattr(collections, "MutableMapping"):
        collections.MutableMapping = collections.abc.MutableMapping  # type: ignore[attr-defined]
    if not hasattr(collections, "Mapping"):
        collections.Mapping = collections.abc.Mapping  # type: ignore[attr-defined]

    urlparse_mod = types.ModuleType("urlparse")
    urlparse_mod.urlparse = urllib.parse.urlparse
    urlparse_mod.urljoin = urllib.parse.urljoin
    urlparse_mod.parse_qs = urllib.parse.parse_qs
    sys.modules.setdefault("urlparse", urlparse_mod)

    urllib_mod = types.ModuleType("urllib")
    urllib_mod.urlencode = urllib.parse.urlencode
    urllib_mod.quote = urllib.parse.quote
    urllib_mod.unquote = urllib.parse.unquote
    urllib_mod.pathname2url = urllib.request.pathname2url
    sys.modules.setdefault("urllib", urllib_mod)


install_legacy_shims()

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402
import polars as pl  # noqa: E402
from intermine.errors import WebserviceError as OldWebserviceError  # noqa: E402
from intermine.webservice import Service as OldService  # noqa: E402
from intermine314.errors import WebserviceError as NewWebserviceError  # noqa: E402
from intermine314.mine_registry import (  # noqa: E402
    DEFAULT_BENCHMARK_FALLBACK_PROFILE,
    resolve_benchmark_plan,
    resolve_named_benchmark_profile,
    resolve_preferred_workers,
)
from intermine314.webservice import Service as NewService  # noqa: E402
import intermine  # noqa: E402
import intermine314  # noqa: E402


VIEWS = [
    "Gene.primaryIdentifier",
    "Gene.secondaryIdentifier",
    "Gene.symbol",
    "Gene.name",
    "Gene.briefDescription",
    "Gene.transcripts.primaryIdentifier",
    "Gene.transcripts.length",
    "Gene.proteins.primaryIdentifier",
    "Gene.proteins.length",
    "Gene.proteins.proteinDomainRegions.proteinDomain.primaryIdentifier",
]

JOINS = [
    "Gene.transcripts",
    "Gene.proteins",
    "Gene.proteins.proteinDomainRegions",
    "Gene.proteins.proteinDomainRegions.proteinDomain",
]

RETRIABLE_EXC = (
    URLError,
    HTTPError,
    OSError,
    TimeoutError,
    socket.timeout,
    OldWebserviceError,
    NewWebserviceError,
)


@dataclass
class ModeRun:
    mode: str
    repetition: int
    seconds: float
    rows: int
    rows_per_s: float
    retries: int
    available_rows_per_pass: int
    effective_workers: int | None
    block_stats: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mine-url",
        default="https://maizemine.rnet.missouri.edu/maizemine",
        help="Mine root URL (without /service).",
    )
    parser.add_argument(
        "--baseline-rows",
        type=int,
        default=100_000,
        help="Rows for direct compare phase (intermine + intermine314).",
    )
    parser.add_argument(
        "--parallel-rows",
        type=int,
        default=1_000_000,
        help="Rows for intermine314-only parallel scaling phase.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Compatibility knob: if set, applies to both baseline and parallel phases.",
    )
    parser.add_argument(
        "--workers",
        default="auto",
        help="Comma-separated worker list for intermine314, or 'auto' to use mine registry defaults.",
    )
    parser.add_argument(
        "--benchmark-profile",
        default="auto",
        choices=[
            "auto",
            "benchmark_profile_1",
            "benchmark_profile_2",
            "benchmark_profile_3",
            "benchmark_profile_4",
        ],
        help="Benchmark profile set from mine registry; used when --workers=auto.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Repetitions per mode (recommendation: 3 to 5).",
    )
    parser.add_argument("--page-size", type=int, default=5_000, help="Page size for parallel runs.")
    parser.add_argument(
        "--page-sizes",
        default=None,
        help="Optional comma-separated page sizes for worker/page-size matrix runs. Defaults to --page-size.",
    )
    parser.add_argument(
        "--ordered-mode",
        default="ordered",
        choices=["ordered", "unordered", "window", "mostly_ordered"],
        help="Ordering mode for intermine314 parallel queries.",
    )
    parser.add_argument(
        "--ordered-window-pages",
        type=int,
        default=10,
        help="Window size used when ordered-mode is window/mostly_ordered.",
    )
    parser.add_argument(
        "--parallel-profile",
        default="default",
        choices=["default", "large_query", "unordered", "mostly_ordered"],
        help="Parallel profile preset for intermine314.",
    )
    parser.add_argument(
        "--large-query-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable large-query defaults (prefetch=2*workers).",
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=None,
        help="Optional explicit prefetch value for intermine314.",
    )
    parser.add_argument(
        "--inflight-limit",
        type=int,
        default=None,
        help="Optional explicit in-flight page cap (maps to Python 3.14 buffersize for ordered mode).",
    )
    parser.add_argument(
        "--parallel-window-factor",
        type=int,
        default=4,
        help="Legacy fixed-chunk factor (used only when --no-auto-chunking).",
    )
    parser.add_argument(
        "--auto-chunking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable dynamic chunk sizing for parallel runs.",
    )
    parser.add_argument(
        "--chunk-target-seconds",
        type=float,
        default=2.0,
        help="Target block runtime for adaptive chunking.",
    )
    parser.add_argument(
        "--chunk-min-pages",
        type=int,
        default=2,
        help="Minimum number of pages per adaptive chunk.",
    )
    parser.add_argument(
        "--chunk-max-pages",
        type=int,
        default=64,
        help="Maximum number of pages per adaptive chunk.",
    )
    parser.add_argument(
        "--legacy-batch-size",
        type=int,
        default=5_000,
        help="Batch size for legacy intermine sequential fetch.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional sleep after each request block for service etiquette.",
    )
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per failed request block.")
    parser.add_argument("--timeout-seconds", type=int, default=60, help="Socket timeout.")
    parser.add_argument(
        "--randomize-mode-order",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize mode execution order per repetition to reduce warm-up/order bias.",
    )
    parser.add_argument(
        "--csv-old-path",
        default="/tmp/intermine314_benchmark_100k_intermine.csv",
        help="Output path for 100k legacy intermine CSV export.",
    )
    parser.add_argument(
        "--parquet-compare-path",
        default="/tmp/intermine314_benchmark_100k_intermine314.parquet",
        help="Output path for 100k intermine314 parquet export used in storage comparison.",
    )
    parser.add_argument(
        "--csv-new-path",
        default="/tmp/intermine314_benchmark_1m_intermine314.csv",
        help="Output path for 1M intermine314 CSV export (for pandas benchmark).",
    )
    parser.add_argument(
        "--parquet-new-path",
        default="/tmp/intermine314_benchmark_1m_intermine314.parquet",
        help="Output path for 1M intermine314 parquet export (for polars benchmark).",
    )
    parser.add_argument(
        "--json-out",
        default="/tmp/intermine314_benchmark.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--dataframe-repetitions",
        type=int,
        default=3,
        help="Repetitions for pandas/polars dataframe benchmarks.",
    )
    return parser.parse_args()


def parse_workers(text: str) -> list[int]:
    text = text.strip().lower()
    if text in {"auto", "registry", "mine"}:
        return []
    workers: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        val = int(chunk)
        if val <= 0:
            raise ValueError(f"Invalid worker count: {val}")
        workers.append(val)
    if not workers:
        raise ValueError("Worker list is empty")
    return workers


def parse_page_sizes(text: str | None, fallback_page_size: int) -> list[int]:
    if text is None:
        return [fallback_page_size]
    values: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        val = int(chunk)
        if val <= 0:
            raise ValueError(f"Invalid page size: {val}")
        values.append(val)
    if not values:
        raise ValueError("Page-size list is empty")
    return values


def mode_label_for_workers(workers: int | None) -> str:
    if workers is None:
        return "intermine314_auto"
    return f"intermine314_w{workers}"


def resolve_benchmark_workers(mine_url: str, rows_target: int, configured_workers: int | None) -> int:
    if configured_workers is not None:
        return configured_workers
    return int(resolve_preferred_workers(mine_url, rows_target, 16))


def resolve_phase_plan(
    *,
    mine_url: str,
    rows_target: int,
    explicit_workers: list[int],
    benchmark_profile: str,
    phase_default_include_legacy: bool,
) -> dict[str, Any]:
    if explicit_workers:
        return {
            "name": "workers_override",
            "workers": explicit_workers,
            "include_legacy_baseline": phase_default_include_legacy,
        }

    if benchmark_profile != "auto":
        profile_plan = resolve_named_benchmark_profile(benchmark_profile, DEFAULT_BENCHMARK_FALLBACK_PROFILE)
    else:
        profile_plan = resolve_benchmark_plan(mine_url, rows_target, DEFAULT_BENCHMARK_FALLBACK_PROFILE)

    return {
        "name": profile_plan["name"],
        "workers": list(profile_plan["workers"]),
        "include_legacy_baseline": phase_default_include_legacy and bool(profile_plan["include_legacy_baseline"]),
    }


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def initial_chunk_pages(
    *,
    workers: int,
    ordered_mode: str,
    large_query_mode: bool,
    prefetch: int | None,
    inflight_limit: int | None,
    min_pages: int,
    max_pages: int,
) -> int:
    seed = inflight_limit if inflight_limit is not None else prefetch
    if seed is None:
        seed = workers * 2 if large_query_mode else workers
    if ordered_mode in {"unordered", "window", "mostly_ordered"}:
        seed = max(seed, workers * 2)
    return _clamp(int(seed), min_pages, max_pages)


def tune_chunk_pages(
    *,
    current_pages: int,
    rows_fetched: int,
    block_seconds: float,
    page_size: int,
    target_seconds: float,
    min_pages: int,
    max_pages: int,
) -> int:
    if rows_fetched <= 0 or block_seconds <= 0:
        return current_pages
    rows_per_second = rows_fetched / block_seconds
    target_rows = max(page_size, int(rows_per_second * target_seconds))
    ideal_pages = max(1, int(math.ceil(target_rows / page_size)))
    blended = int(round((0.7 * current_pages) + (0.3 * ideal_pages)))
    return _clamp(blended, min_pages, max_pages)


def read_os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        out[key] = val.strip().strip('"')
    return out


def pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def stat_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    if len(values) == 1:
        stddev = 0.0
    else:
        stddev = statistics.stdev(values)
    sorted_vals = sorted(values)

    def nearest_rank(p: float) -> float:
        idx = int(math.ceil(p * len(sorted_vals))) - 1
        idx = max(0, min(idx, len(sorted_vals) - 1))
        return sorted_vals[idx]

    trim_count = int(math.floor(len(sorted_vals) * 0.10))
    if trim_count > 0:
        trimmed = sorted_vals[: len(sorted_vals) - trim_count]
    else:
        trimmed = sorted_vals

    group_count = min(3, len(sorted_vals))
    chunk_size = int(math.ceil(len(sorted_vals) / group_count))
    means = []
    for idx in range(0, len(sorted_vals), chunk_size):
        chunk = sorted_vals[idx : idx + chunk_size]
        if chunk:
            means.append(statistics.fmean(chunk))

    mean_val = statistics.fmean(values)
    return {
        "n": float(len(values)),
        "mean": mean_val,
        "stddev": stddev,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "p90": nearest_rank(0.90),
        "p95": nearest_rank(0.95),
        "trimmed_mean_drop_high_10pct": statistics.fmean(trimmed),
        "median_of_means": statistics.median(means),
        "cv_pct": (stddev / mean_val * 100.0) if mean_val else 0.0,
    }


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def make_query(service_cls: Any, mine_url: str) -> Any:
    service = service_cls(mine_url)
    query = service.new_query("Gene")
    query.add_view(*VIEWS)
    for join in JOINS:
        query.add_join(join, "OUTER")
    return query


def count_with_retry(query: Any, max_retries: int, sleep_seconds: float) -> tuple[int, int]:
    retries = 0
    for attempt in range(1, max_retries + 1):
        try:
            return query.count(), retries
        except RETRIABLE_EXC as exc:
            retries += 1
            wait_s = min(2.0 * attempt, 12.0)
            print(f"count_retry attempt={attempt} err={exc} wait_s={wait_s:.1f}", flush=True)
            time.sleep(wait_s)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    raise RuntimeError("Failed to count query rows after retries")


def run_mode(
    *,
    mode: str,
    mine_url: str,
    rows_target: int,
    page_size: int,
    workers: int | None,
    legacy_batch_size: int,
    parallel_window_factor: int,
    auto_chunking: bool,
    chunk_target_seconds: float,
    chunk_min_pages: int,
    chunk_max_pages: int,
    ordered_mode: str,
    ordered_window_pages: int,
    parallel_profile: str,
    large_query_mode: bool,
    prefetch: int | None,
    inflight_limit: int | None,
    sleep_seconds: float,
    max_retries: int,
    csv_out_path: Path | None,
) -> ModeRun:
    if mode == "intermine_batched":
        query = make_query(OldService, mine_url)
    else:
        query = make_query(NewService, mine_url)

    available_rows, retries = count_with_retry(query, max_retries=max_retries, sleep_seconds=sleep_seconds)
    effective_workers: int | None = None
    if mode != "intermine_batched":
        effective_workers = resolve_benchmark_workers(mine_url, rows_target, workers)
        workers = effective_workers

    writer = None
    file_handle = None
    if csv_out_path is not None:
        ensure_parent(csv_out_path)
        if csv_out_path.exists():
            csv_out_path.unlink()
        file_handle = csv_out_path.open("w", newline="", encoding="utf-8")

    processed = 0
    start = 0
    next_mark = 100_000
    block_durations: list[float] = []
    chunk_sizes: list[float] = []
    chunk_pages = 1
    if mode != "intermine_batched":
        assert workers is not None
        chunk_pages = initial_chunk_pages(
            workers=workers,
            ordered_mode=ordered_mode,
            large_query_mode=large_query_mode,
            prefetch=prefetch,
            inflight_limit=inflight_limit,
            min_pages=chunk_min_pages,
            max_pages=chunk_max_pages,
        )

    t0 = time.perf_counter()
    try:
        while processed < rows_target:
            remaining = rows_target - processed
            if mode == "intermine_batched":
                size = min(legacy_batch_size, remaining)
            else:
                assert workers is not None
                if auto_chunking:
                    size = min(page_size * chunk_pages, remaining)
                else:
                    size = min(page_size * workers * parallel_window_factor, remaining)

            got = 0
            block_seconds = 0.0
            for attempt in range(1, max_retries + 1):
                b0 = time.perf_counter()
                try:
                    if mode == "intermine_batched":
                        iterator = query.results(row="dict", start=start, size=size)
                    else:
                        iterator = query.run_parallel(
                            row="dict",
                            start=start,
                            size=size,
                            page_size=page_size,
                            max_workers=workers,
                            ordered=ordered_mode,
                            prefetch=prefetch,
                            inflight_limit=inflight_limit,
                            ordered_window_pages=ordered_window_pages,
                            profile=parallel_profile,
                            large_query_mode=large_query_mode,
                            pagination="auto",
                        )
                    for row in iterator:
                        if writer is None and file_handle is not None:
                            writer = csv.DictWriter(file_handle, fieldnames=list(row.keys()))
                            writer.writeheader()
                        if writer is not None:
                            writer.writerow(row)
                        got += 1
                    block_seconds = time.perf_counter() - b0
                    block_durations.append(block_seconds)
                    chunk_sizes.append(float(size))
                    break
                except RETRIABLE_EXC as exc:
                    retries += 1
                    if mode != "intermine_batched" and auto_chunking:
                        chunk_pages = _clamp(max(1, chunk_pages // 2), chunk_min_pages, chunk_max_pages)
                    wait_s = min(2.0 * attempt, 12.0)
                    print(
                        f"retry mode={mode} attempt={attempt} start={start} size={size} err={exc} wait_s={wait_s:.1f}",
                        flush=True,
                    )
                    time.sleep(wait_s)
            else:
                raise RuntimeError(f"Failed block after retries: mode={mode}, start={start}, size={size}")

            if got == 0:
                # Avoid infinite loop on empty result pages.
                raise RuntimeError(f"Received 0 rows for mode={mode}, start={start}, size={size}")

            processed += got
            start += got
            if start >= available_rows:
                # Deterministic wrap: allows benchmarking if target > available.
                start = 0

            while processed >= next_mark:
                print(f"{mode}_progress rows={next_mark}", flush=True)
                next_mark += 100_000

            if mode != "intermine_batched" and auto_chunking:
                chunk_pages = tune_chunk_pages(
                    current_pages=chunk_pages,
                    rows_fetched=got,
                    block_seconds=block_seconds,
                    page_size=page_size,
                    target_seconds=chunk_target_seconds,
                    min_pages=chunk_min_pages,
                    max_pages=chunk_max_pages,
                )

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    finally:
        if file_handle is not None:
            file_handle.close()

    elapsed = time.perf_counter() - t0
    return ModeRun(
        mode=mode,
        repetition=-1,
        seconds=elapsed,
        rows=processed,
        rows_per_s=(processed / elapsed) if elapsed else 0.0,
        retries=retries,
        available_rows_per_pass=available_rows,
        effective_workers=effective_workers,
        block_stats={
            "durations": stat_summary(block_durations),
            "chunk_sizes_rows": stat_summary(chunk_sizes),
            "auto_chunking": auto_chunking and mode != "intermine_batched",
            "chunk_target_seconds": chunk_target_seconds,
            "chunk_min_pages": chunk_min_pages,
            "chunk_max_pages": chunk_max_pages,
        },
    )


def run_replicated_fetch_benchmarks(
    *,
    phase_name: str,
    mine_url: str,
    rows_target: int,
    repetitions: int,
    workers: list[int],
    include_legacy_baseline: bool,
    page_size: int,
    legacy_batch_size: int,
    parallel_window_factor: int,
    auto_chunking: bool,
    chunk_target_seconds: float,
    chunk_min_pages: int,
    chunk_max_pages: int,
    ordered_mode: str,
    ordered_window_pages: int,
    parallel_profile: str,
    large_query_mode: bool,
    prefetch: int | None,
    inflight_limit: int | None,
    randomize_mode_order: bool,
    sleep_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    mode_defs: list[tuple[str, int | None]] = []
    if include_legacy_baseline:
        mode_defs.append(("intermine_batched", None))
    if workers:
        mode_defs.extend((mode_label_for_workers(w), w) for w in workers)
    else:
        mode_defs.append((mode_label_for_workers(None), None))

    all_runs: dict[str, list[ModeRun]] = {m: [] for m, _ in mode_defs}

    # Warm-up with small pulls to stabilize TLS/session effects.
    if include_legacy_baseline:
        _ = run_mode(
            mode="intermine_batched",
            mine_url=mine_url,
            rows_target=min(2_000, rows_target),
            page_size=page_size,
            workers=None,
            legacy_batch_size=legacy_batch_size,
            parallel_window_factor=parallel_window_factor,
            auto_chunking=auto_chunking,
            chunk_target_seconds=chunk_target_seconds,
            chunk_min_pages=chunk_min_pages,
            chunk_max_pages=chunk_max_pages,
            ordered_mode=ordered_mode,
            ordered_window_pages=ordered_window_pages,
            parallel_profile=parallel_profile,
            large_query_mode=large_query_mode,
            prefetch=prefetch,
            inflight_limit=inflight_limit,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
            csv_out_path=None,
        )
    warmup_mode, warmup_workers = next((m, w) for m, w in mode_defs if m != "intermine_batched")
    _ = run_mode(
        mode=warmup_mode,
        mine_url=mine_url,
        rows_target=min(2_000, rows_target),
        page_size=page_size,
        workers=warmup_workers,
        legacy_batch_size=legacy_batch_size,
        parallel_window_factor=parallel_window_factor,
        auto_chunking=auto_chunking,
        chunk_target_seconds=chunk_target_seconds,
        chunk_min_pages=chunk_min_pages,
        chunk_max_pages=chunk_max_pages,
        ordered_mode=ordered_mode,
        ordered_window_pages=ordered_window_pages,
        parallel_profile=parallel_profile,
        large_query_mode=large_query_mode,
        prefetch=prefetch,
        inflight_limit=inflight_limit,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
        csv_out_path=None,
    )

    randomized_orders: list[list[str]] = []
    for rep in range(1, repetitions + 1):
        rep_modes = list(mode_defs)
        if randomize_mode_order:
            random.shuffle(rep_modes)
        randomized_orders.append([mode for mode, _ in rep_modes])
        for mode, mode_workers in rep_modes:
            print(f"run_start phase={phase_name} mode={mode} repetition={rep}", flush=True)
            run = run_mode(
                mode=mode,
                mine_url=mine_url,
                rows_target=rows_target,
                page_size=page_size,
                workers=mode_workers,
                legacy_batch_size=legacy_batch_size,
                parallel_window_factor=parallel_window_factor,
                auto_chunking=auto_chunking,
                chunk_target_seconds=chunk_target_seconds,
                chunk_min_pages=chunk_min_pages,
                chunk_max_pages=chunk_max_pages,
                ordered_mode=ordered_mode,
                ordered_window_pages=ordered_window_pages,
                parallel_profile=parallel_profile,
                large_query_mode=large_query_mode,
                prefetch=prefetch,
                inflight_limit=inflight_limit,
                sleep_seconds=sleep_seconds,
                max_retries=max_retries,
                csv_out_path=None,
            )
            run.repetition = rep
            all_runs[mode].append(run)
            print(
                f"run_done phase={phase_name} mode={mode} repetition={rep} seconds={run.seconds:.3f} rows_per_s={run.rows_per_s:.2f} retries={run.retries}",
                flush=True,
            )

    summary: dict[str, Any] = {}
    reference_mode = "intermine_batched" if include_legacy_baseline else warmup_mode
    reference_mean = statistics.fmean([r.seconds for r in all_runs[reference_mode]])
    reference_rps_mean = statistics.fmean([r.rows_per_s for r in all_runs[reference_mode]])
    reference_median = statistics.median([r.seconds for r in all_runs[reference_mode]])
    reference_rps_median = statistics.median([r.rows_per_s for r in all_runs[reference_mode]])

    for mode, runs in all_runs.items():
        secs = [r.seconds for r in runs]
        rps = [r.rows_per_s for r in runs]
        retries = [float(r.retries) for r in runs]
        mode_summary = {
            "seconds": stat_summary(secs),
            "rows_per_s": stat_summary(rps),
            "retries": stat_summary(retries),
            "available_rows_per_pass": runs[0].available_rows_per_pass if runs else None,
            "runs": [
                {
                    "repetition": r.repetition,
                    "seconds": r.seconds,
                    "rows": r.rows,
                    "rows_per_s": r.rows_per_s,
                    "retries": r.retries,
                    "effective_workers": r.effective_workers,
                    "block_stats": r.block_stats,
                }
                for r in runs
            ],
        }
        if mode != reference_mode:
            mode_mean = mode_summary["seconds"]["mean"]
            mode_rps_mean = mode_summary["rows_per_s"]["mean"]
            mode_median = mode_summary["seconds"]["median"]
            mode_rps_median = mode_summary["rows_per_s"]["median"]
            mode_summary["vs_reference"] = {
                "reference_mode": reference_mode,
                "speedup": (reference_mean / mode_mean) if mode_mean else None,
                "speedup_median": (reference_median / mode_median) if mode_median else None,
                "faster_pct": ((reference_mean - mode_mean) / reference_mean * 100.0) if reference_mean else None,
                "faster_pct_median": (
                    (reference_median - mode_median) / reference_median * 100.0
                )
                if reference_median
                else None,
                "throughput_increase_pct": (
                    (mode_rps_mean - reference_rps_mean) / reference_rps_mean * 100.0
                )
                if reference_rps_mean
                else None,
                "throughput_increase_pct_median": (
                    (mode_rps_median - reference_rps_median) / reference_rps_median * 100.0
                )
                if reference_rps_median
                else None,
            }
        summary[mode] = mode_summary

    return {
        "phase_name": phase_name,
        "include_legacy_baseline": include_legacy_baseline,
        "reference_mode": reference_mode,
        "mode_order": [m for m, _ in mode_defs],
        "mode_order_by_repetition": randomized_orders,
        "randomized_mode_order": randomize_mode_order,
        "repetitions": repetitions,
        "rows_target": rows_target,
        "page_size": page_size,
        "results": summary,
    }


def export_for_storage(
    *,
    mine_url: str,
    rows_target: int,
    page_size: int,
    workers_for_new: int | None,
    legacy_batch_size: int,
    parallel_window_factor: int,
    auto_chunking: bool,
    chunk_target_seconds: float,
    chunk_min_pages: int,
    chunk_max_pages: int,
    ordered_mode: str,
    ordered_window_pages: int,
    parallel_profile: str,
    large_query_mode: bool,
    prefetch: int | None,
    inflight_limit: int | None,
    sleep_seconds: float,
    max_retries: int,
    csv_old_path: Path,
    parquet_new_path: Path,
) -> dict[str, Any]:
    print("export_start mode=intermine_batched_csv", flush=True)
    old_export = run_mode(
        mode="intermine_batched",
        mine_url=mine_url,
        rows_target=rows_target,
        page_size=page_size,
        workers=None,
        legacy_batch_size=legacy_batch_size,
        parallel_window_factor=parallel_window_factor,
        auto_chunking=auto_chunking,
        chunk_target_seconds=chunk_target_seconds,
        chunk_min_pages=chunk_min_pages,
        chunk_max_pages=chunk_max_pages,
        ordered_mode=ordered_mode,
        ordered_window_pages=ordered_window_pages,
        parallel_profile=parallel_profile,
        large_query_mode=large_query_mode,
        prefetch=prefetch,
        inflight_limit=inflight_limit,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
        csv_out_path=csv_old_path,
    )
    print(
        f"export_done mode=intermine_batched_csv seconds={old_export.seconds:.3f} path={csv_old_path}",
        flush=True,
    )

    tmp_new_csv = parquet_new_path.with_suffix(parquet_new_path.suffix + ".tmp.csv")
    mode_label = mode_label_for_workers(workers_for_new)
    print(f"export_start mode={mode_label}_tmpcsv", flush=True)
    new_export = run_mode(
        mode=mode_label,
        mine_url=mine_url,
        rows_target=rows_target,
        page_size=page_size,
        workers=workers_for_new,
        legacy_batch_size=legacy_batch_size,
        parallel_window_factor=parallel_window_factor,
        auto_chunking=auto_chunking,
        chunk_target_seconds=chunk_target_seconds,
        chunk_min_pages=chunk_min_pages,
        chunk_max_pages=chunk_max_pages,
        ordered_mode=ordered_mode,
        ordered_window_pages=ordered_window_pages,
        parallel_profile=parallel_profile,
        large_query_mode=large_query_mode,
        prefetch=prefetch,
        inflight_limit=inflight_limit,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
        csv_out_path=tmp_new_csv,
    )
    print(
        f"export_done mode={mode_label}_tmpcsv seconds={new_export.seconds:.3f} path={tmp_new_csv}",
        flush=True,
    )

    ensure_parent(parquet_new_path)
    if parquet_new_path.exists():
        parquet_new_path.unlink()

    t0 = time.perf_counter()
    # Use DuckDB COPY for robust CSV->Parquet conversion.
    in_csv = str(tmp_new_csv).replace("'", "''")
    out_parquet = str(parquet_new_path).replace("'", "''")
    con = duckdb.connect()
    con.execute(
        f"COPY (SELECT * FROM read_csv_auto('{in_csv}', sample_size=-1)) "
        f"TO '{out_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.close()
    parquet_seconds = time.perf_counter() - t0
    print(f"convert_done csv_to_parquet seconds={parquet_seconds:.3f} path={parquet_new_path}", flush=True)

    csv_size = csv_old_path.stat().st_size
    parquet_size = parquet_new_path.stat().st_size
    saved = csv_size - parquet_size
    reduction_pct = (saved / csv_size * 100.0) if csv_size else 0.0
    ratio = (csv_size / parquet_size) if parquet_size else None

    # Keep temp CSV for auditability.
    return {
        "old_export_csv": {
            "path": str(csv_old_path),
            "seconds": old_export.seconds,
            "rows_per_s": old_export.rows_per_s,
            "retries": old_export.retries,
        },
        "new_export_tmp_csv": {
            "path": str(tmp_new_csv),
            "seconds": new_export.seconds,
            "rows_per_s": new_export.rows_per_s,
            "retries": new_export.retries,
        },
        "new_parquet": {
            "path": str(parquet_new_path),
            "conversion_seconds_from_tmp_csv": parquet_seconds,
        },
        "sizes": {
            "csv_bytes": csv_size,
            "parquet_bytes": parquet_size,
            "saved_bytes": saved,
            "reduction_pct": reduction_pct,
            "csv_to_parquet_ratio": ratio,
        },
    }


def export_new_only_for_dataframe(
    *,
    mine_url: str,
    rows_target: int,
    page_size: int,
    workers_for_new: int | None,
    legacy_batch_size: int,
    parallel_window_factor: int,
    auto_chunking: bool,
    chunk_target_seconds: float,
    chunk_min_pages: int,
    chunk_max_pages: int,
    ordered_mode: str,
    ordered_window_pages: int,
    parallel_profile: str,
    large_query_mode: bool,
    prefetch: int | None,
    inflight_limit: int | None,
    sleep_seconds: float,
    max_retries: int,
    csv_new_path: Path,
    parquet_new_path: Path,
) -> dict[str, Any]:
    mode_label = mode_label_for_workers(workers_for_new)
    print(f"export_start mode={mode_label}_1m_csv", flush=True)
    new_export = run_mode(
        mode=mode_label,
        mine_url=mine_url,
        rows_target=rows_target,
        page_size=page_size,
        workers=workers_for_new,
        legacy_batch_size=legacy_batch_size,
        parallel_window_factor=parallel_window_factor,
        auto_chunking=auto_chunking,
        chunk_target_seconds=chunk_target_seconds,
        chunk_min_pages=chunk_min_pages,
        chunk_max_pages=chunk_max_pages,
        ordered_mode=ordered_mode,
        ordered_window_pages=ordered_window_pages,
        parallel_profile=parallel_profile,
        large_query_mode=large_query_mode,
        prefetch=prefetch,
        inflight_limit=inflight_limit,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
        csv_out_path=csv_new_path,
    )
    print(
        f"export_done mode={mode_label}_1m_csv seconds={new_export.seconds:.3f} path={csv_new_path}",
        flush=True,
    )

    ensure_parent(parquet_new_path)
    if parquet_new_path.exists():
        parquet_new_path.unlink()
    t0 = time.perf_counter()
    in_csv = str(csv_new_path).replace("'", "''")
    out_parquet = str(parquet_new_path).replace("'", "''")
    con = duckdb.connect()
    con.execute(
        f"COPY (SELECT * FROM read_csv_auto('{in_csv}', sample_size=-1)) "
        f"TO '{out_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.close()
    parquet_seconds = time.perf_counter() - t0
    print(
        f"convert_done intermine314_1m_csv_to_parquet seconds={parquet_seconds:.3f} path={parquet_new_path}",
        flush=True,
    )

    csv_size = csv_new_path.stat().st_size
    parquet_size = parquet_new_path.stat().st_size
    saved = csv_size - parquet_size
    reduction_pct = (saved / csv_size * 100.0) if csv_size else 0.0
    ratio = (csv_size / parquet_size) if parquet_size else None

    return {
        "new_export_csv": {
            "path": str(csv_new_path),
            "seconds": new_export.seconds,
            "rows_per_s": new_export.rows_per_s,
            "retries": new_export.retries,
        },
        "new_export_parquet": {
            "path": str(parquet_new_path),
            "conversion_seconds_from_csv": parquet_seconds,
        },
        "sizes": {
            "csv_bytes": csv_size,
            "parquet_bytes": parquet_size,
            "saved_bytes": saved,
            "reduction_pct": reduction_pct,
            "csv_to_parquet_ratio": ratio,
        },
    }


def bench_pandas(csv_path: Path, repetitions: int) -> dict[str, Any]:
    load_times: list[float] = []
    suite_times: list[float] = []
    row_counts: list[int] = []
    cds_non_null_counts: list[int] = []
    protein_means: list[float] = []
    top1_counts: list[int] = []
    memory_bytes: list[int] = []

    for rep in range(1, repetitions + 1):
        t0 = time.perf_counter()
        df = pd.read_csv(csv_path)
        load_t = time.perf_counter() - t0
        load_times.append(load_t)
        row_counts.append(int(df.shape[0]))
        memory_bytes.append(int(df.memory_usage(deep=True).sum()))

        t0 = time.perf_counter()
        cds_non_null = int(df["Gene.transcripts.primaryIdentifier"].notna().sum())
        prot_len = pd.to_numeric(df["Gene.proteins.length"], errors="coerce")
        prot_mean = float(prot_len.mean()) if prot_len.notna().any() else float("nan")
        top10 = (
            df.groupby("Gene.primaryIdentifier", dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
        suite_t = time.perf_counter() - t0
        suite_times.append(suite_t)
        cds_non_null_counts.append(cds_non_null)
        protein_means.append(prot_mean)
        top1_counts.append(int(top10.iloc[0]) if len(top10) else 0)
        print(
            f"pandas_rep rep={rep} load_s={load_t:.3f} suite_s={suite_t:.3f} rows={row_counts[-1]}",
            flush=True,
        )

    return {
        "repetitions": repetitions,
        "load_seconds": stat_summary(load_times),
        "analytics_suite_seconds": stat_summary(suite_times),
        "memory_bytes": stat_summary([float(v) for v in memory_bytes]),
        "row_counts": row_counts,
        "cds_non_null_counts": cds_non_null_counts,
        "protein_length_means": protein_means,
        "top1_group_count": top1_counts,
    }


def bench_polars(parquet_path: Path, repetitions: int) -> dict[str, Any]:
    load_times: list[float] = []
    suite_times: list[float] = []
    lazy_suite_times: list[float] = []
    row_counts: list[int] = []
    cds_non_null_counts: list[int] = []
    protein_means: list[float] = []
    top1_counts: list[int] = []
    memory_bytes: list[int] = []

    for rep in range(1, repetitions + 1):
        t0 = time.perf_counter()
        df = pl.read_parquet(parquet_path)
        load_t = time.perf_counter() - t0
        load_times.append(load_t)
        row_counts.append(int(df.height))
        memory_bytes.append(int(df.estimated_size()))

        t0 = time.perf_counter()
        cds_non_null = int(df.select(pl.col("Gene.transcripts.primaryIdentifier").is_not_null().sum()).item())
        prot_mean = float(df.select(pl.col("Gene.proteins.length").cast(pl.Float64, strict=False).mean()).item())
        top10 = (
            df.group_by("Gene.primaryIdentifier")
            .len()
            .sort("len", descending=True)
            .head(10)
        )
        suite_t = time.perf_counter() - t0
        suite_times.append(suite_t)
        cds_non_null_counts.append(cds_non_null)
        protein_means.append(prot_mean)
        top1_counts.append(int(top10["len"][0]) if top10.height else 0)

        t0 = time.perf_counter()
        lazy_row_count = (
            pl.scan_parquet(parquet_path)
            .select(pl.len())
            .collect()
            .item(0, 0)
        )
        lazy_t = time.perf_counter() - t0
        lazy_suite_times.append(lazy_t)
        assert int(lazy_row_count) == row_counts[-1]

        print(
            f"polars_rep rep={rep} load_s={load_t:.3f} suite_s={suite_t:.3f} lazy_s={lazy_t:.3f} rows={row_counts[-1]}",
            flush=True,
        )

    return {
        "repetitions": repetitions,
        "load_seconds": stat_summary(load_times),
        "analytics_suite_seconds": stat_summary(suite_times),
        "lazy_scan_seconds": stat_summary(lazy_suite_times),
        "memory_bytes_estimated": stat_summary([float(v) for v in memory_bytes]),
        "row_counts": row_counts,
        "cds_non_null_counts": cds_non_null_counts,
        "protein_length_means": protein_means,
        "top1_group_count": top1_counts,
    }


def capture_environment(
    args: argparse.Namespace,
    workers: list[int],
    page_sizes: list[int],
    direct_phase_plan: dict[str, Any],
    parallel_phase_plan: dict[str, Any],
) -> dict[str, Any]:
    os_release = read_os_release()
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        )
    except Exception:
        git_commit = "unknown"

    resolved_baseline_workers = resolve_preferred_workers(args.mine_url, args.baseline_rows, 16)
    resolved_parallel_workers = resolve_preferred_workers(args.mine_url, args.parallel_rows, 16)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_short": git_commit,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "distro": {
            "name": os_release.get("NAME", ""),
            "version": os_release.get("VERSION", ""),
            "id": os_release.get("ID", ""),
            "version_id": os_release.get("VERSION_ID", ""),
            "pretty_name": os_release.get("PRETTY_NAME", ""),
        },
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "full_version": sys.version,
        },
        "packages": {
            "intermine": pkg_version("intermine"),
            "intermine314": pkg_version("intermine314"),
            "pandas": pkg_version("pandas"),
            "polars": pkg_version("polars"),
            "duckdb": pkg_version("duckdb"),
            "pyarrow": pkg_version("pyarrow"),
        },
        "runtime_config": {
            "mine_url": args.mine_url,
            "baseline_rows": args.baseline_rows,
            "parallel_rows": args.parallel_rows,
            "rows_compat": args.rows,
            "workers": workers,
            "workers_input": args.workers,
            "benchmark_profile": args.benchmark_profile,
            "mine_registry_workers": {
                "baseline_rows": resolved_baseline_workers,
                "parallel_rows": resolved_parallel_workers,
            },
            "phase_plans": {
                "direct_compare_100k": direct_phase_plan,
                "parallel_only_1m": parallel_phase_plan,
            },
            "page_sizes": page_sizes,
            "repetitions": args.repetitions,
            "page_size": args.page_size,
            "ordered_mode": args.ordered_mode,
            "ordered_window_pages": args.ordered_window_pages,
            "parallel_profile": args.parallel_profile,
            "large_query_mode": args.large_query_mode,
            "prefetch": args.prefetch,
            "inflight_limit": args.inflight_limit,
            "legacy_batch_size": args.legacy_batch_size,
            "parallel_window_factor": args.parallel_window_factor,
            "auto_chunking": args.auto_chunking,
            "chunk_target_seconds": args.chunk_target_seconds,
            "chunk_min_pages": args.chunk_min_pages,
            "chunk_max_pages": args.chunk_max_pages,
            "randomize_mode_order": args.randomize_mode_order,
            "sleep_seconds": args.sleep_seconds,
            "max_retries": args.max_retries,
            "timeout_seconds": args.timeout_seconds,
            "dataframe_repetitions": args.dataframe_repetitions,
            "csv_old_path": args.csv_old_path,
            "csv_new_path": args.csv_new_path,
            "parquet_compare_path": args.parquet_compare_path,
            "parquet_new_path": args.parquet_new_path,
            "json_out": args.json_out,
        },
    }


def main() -> int:
    args = parse_args()
    if args.rows is not None:
        args.baseline_rows = args.rows
        args.parallel_rows = args.rows
    if args.chunk_target_seconds <= 0:
        raise ValueError("--chunk-target-seconds must be > 0")
    if args.chunk_min_pages <= 0:
        raise ValueError("--chunk-min-pages must be > 0")
    if args.chunk_max_pages < args.chunk_min_pages:
        raise ValueError("--chunk-max-pages must be >= --chunk-min-pages")
    workers = parse_workers(args.workers)
    page_sizes = parse_page_sizes(args.page_sizes, args.page_size)
    socket.setdefaulttimeout(args.timeout_seconds)
    random.seed(42)

    direct_phase_plan = resolve_phase_plan(
        mine_url=args.mine_url,
        rows_target=args.baseline_rows,
        explicit_workers=workers,
        benchmark_profile=args.benchmark_profile,
        phase_default_include_legacy=True,
    )
    parallel_phase_plan = resolve_phase_plan(
        mine_url=args.mine_url,
        rows_target=args.parallel_rows,
        explicit_workers=workers,
        benchmark_profile=args.benchmark_profile,
        phase_default_include_legacy=False,
    )
    if not direct_phase_plan["workers"]:
        raise ValueError("Resolved direct-phase worker list is empty")
    if not parallel_phase_plan["workers"]:
        raise ValueError("Resolved parallel-phase worker list is empty")

    benchmark_workers_for_storage: int | None = max(direct_phase_plan["workers"])
    benchmark_workers_for_dataframe: int | None = max(parallel_phase_plan["workers"])

    report: dict[str, Any] = {
        "environment": capture_environment(args, workers, page_sizes, direct_phase_plan, parallel_phase_plan),
        "query": {
            "views": VIEWS,
            "outer_joins": JOINS,
        },
        "package_imports": {
            "intermine_module": getattr(intermine, "__file__", "unknown"),
            "intermine314_module": getattr(intermine314, "__file__", "unknown"),
            "intermine_version_attr": getattr(intermine, "VERSION", "unknown"),
            "intermine314_version_attr": getattr(intermine314, "VERSION", "unknown"),
        },
    }

    print("benchmark_start", flush=True)
    print(f"mine_url={args.mine_url}", flush=True)
    print(f"workers={workers if workers else 'auto(registry)'}", flush=True)
    print(f"benchmark_profile={args.benchmark_profile}", flush=True)
    print(f"direct_phase_plan={direct_phase_plan}", flush=True)
    print(f"parallel_phase_plan={parallel_phase_plan}", flush=True)
    print(f"page_sizes={page_sizes}", flush=True)
    print(f"repetitions={args.repetitions}", flush=True)
    print(f"baseline_rows={args.baseline_rows}", flush=True)
    print(f"parallel_rows={args.parallel_rows}", flush=True)
    print(f"sleep_seconds={args.sleep_seconds}", flush=True)
    fetch_100k_by_page: dict[str, Any] = {}
    fetch_1m_by_page: dict[str, Any] = {}
    for page_size in page_sizes:
        page_key = f"page_size_{page_size}"
        fetch_100k_by_page[page_key] = run_replicated_fetch_benchmarks(
            phase_name="direct_compare_100k",
            mine_url=args.mine_url,
            rows_target=args.baseline_rows,
            repetitions=args.repetitions,
            workers=direct_phase_plan["workers"],
            include_legacy_baseline=direct_phase_plan["include_legacy_baseline"],
            page_size=page_size,
            legacy_batch_size=args.legacy_batch_size,
            parallel_window_factor=args.parallel_window_factor,
            auto_chunking=args.auto_chunking,
            chunk_target_seconds=args.chunk_target_seconds,
            chunk_min_pages=args.chunk_min_pages,
            chunk_max_pages=args.chunk_max_pages,
            ordered_mode=args.ordered_mode,
            ordered_window_pages=args.ordered_window_pages,
            parallel_profile=args.parallel_profile,
            large_query_mode=args.large_query_mode,
            prefetch=args.prefetch,
            inflight_limit=args.inflight_limit,
            randomize_mode_order=args.randomize_mode_order,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
        )
        fetch_1m_by_page[page_key] = run_replicated_fetch_benchmarks(
            phase_name="parallel_only_1m",
            mine_url=args.mine_url,
            rows_target=args.parallel_rows,
            repetitions=args.repetitions,
            workers=parallel_phase_plan["workers"],
            include_legacy_baseline=parallel_phase_plan["include_legacy_baseline"],
            page_size=page_size,
            legacy_batch_size=args.legacy_batch_size,
            parallel_window_factor=args.parallel_window_factor,
            auto_chunking=args.auto_chunking,
            chunk_target_seconds=args.chunk_target_seconds,
            chunk_min_pages=args.chunk_min_pages,
            chunk_max_pages=args.chunk_max_pages,
            ordered_mode=args.ordered_mode,
            ordered_window_pages=args.ordered_window_pages,
            parallel_profile=args.parallel_profile,
            large_query_mode=args.large_query_mode,
            prefetch=args.prefetch,
            inflight_limit=args.inflight_limit,
            randomize_mode_order=args.randomize_mode_order,
            sleep_seconds=args.sleep_seconds,
            max_retries=args.max_retries,
        )
    report["fetch_benchmark"] = {
        "direct_compare_100k_by_page_size": fetch_100k_by_page,
        "parallel_only_1m_by_page_size": fetch_1m_by_page,
    }
    if len(page_sizes) == 1:
        only_key = f"page_size_{page_sizes[0]}"
        report["fetch_benchmark"]["direct_compare_100k"] = fetch_100k_by_page[only_key]
        report["fetch_benchmark"]["parallel_only_1m"] = fetch_1m_by_page[only_key]

    io_page_size = page_sizes[0]

    storage_compare_100k = export_for_storage(
        mine_url=args.mine_url,
        rows_target=args.baseline_rows,
        page_size=io_page_size,
        workers_for_new=benchmark_workers_for_storage,
        legacy_batch_size=args.legacy_batch_size,
        parallel_window_factor=args.parallel_window_factor,
        auto_chunking=args.auto_chunking,
        chunk_target_seconds=args.chunk_target_seconds,
        chunk_min_pages=args.chunk_min_pages,
        chunk_max_pages=args.chunk_max_pages,
        ordered_mode=args.ordered_mode,
        ordered_window_pages=args.ordered_window_pages,
        parallel_profile=args.parallel_profile,
        large_query_mode=args.large_query_mode,
        prefetch=args.prefetch,
        inflight_limit=args.inflight_limit,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
        csv_old_path=Path(args.csv_old_path),
        parquet_new_path=Path(args.parquet_compare_path),
    )
    storage_new_1m = export_new_only_for_dataframe(
        mine_url=args.mine_url,
        rows_target=args.parallel_rows,
        page_size=io_page_size,
        workers_for_new=benchmark_workers_for_dataframe,
        legacy_batch_size=args.legacy_batch_size,
        parallel_window_factor=args.parallel_window_factor,
        auto_chunking=args.auto_chunking,
        chunk_target_seconds=args.chunk_target_seconds,
        chunk_min_pages=args.chunk_min_pages,
        chunk_max_pages=args.chunk_max_pages,
        ordered_mode=args.ordered_mode,
        ordered_window_pages=args.ordered_window_pages,
        parallel_profile=args.parallel_profile,
        large_query_mode=args.large_query_mode,
        prefetch=args.prefetch,
        inflight_limit=args.inflight_limit,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
        csv_new_path=Path(args.csv_new_path),
        parquet_new_path=Path(args.parquet_new_path),
    )
    report["storage"] = {
        "compare_100k_old_vs_new": storage_compare_100k,
        "new_only_1m": storage_new_1m,
    }

    pandas_df = bench_pandas(Path(args.csv_new_path), repetitions=args.dataframe_repetitions)
    polars_df = bench_polars(Path(args.parquet_new_path), repetitions=args.dataframe_repetitions)
    report["dataframes"] = {
        "dataset": "intermine314_1m_export",
        "pandas_csv": pandas_df,
        "polars_parquet": polars_df,
    }

    ensure_parent(Path(args.json_out))
    with Path(args.json_out).open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    print(f"report_written={args.json_out}", flush=True)
    print("BENCHMARK_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
