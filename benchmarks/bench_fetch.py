from __future__ import annotations

import argparse
import io
import csv
import math
import random
import socket
import statistics
import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError

import requests

try:
    from intermine.errors import WebserviceError as OldWebserviceError
    from intermine.webservice import Service as OldService
except Exception:  # pragma: no cover - optional dependency in benchmark tooling
    class OldWebserviceError(Exception):
        pass

    OldService = None  # type: ignore[assignment]
from intermine314.config.runtime_defaults import get_runtime_defaults
from intermine314.query.builder import ParallelOptions
from intermine314.service.errors import WebserviceError as NewWebserviceError
from intermine314.registry.mines import (
    resolve_mine_user_agent,
    resolve_production_plan,
)
from intermine314.service import Service as NewService
from intermine314.service.tor import tor_proxy_url
from intermine314.service.transport import enforce_tor_dns_safe_proxy_url
from benchmarks.bench_plan import resolve_execution_plan as resolve_benchmark_execution_plan
from benchmarks.bench_constants import (
    AUTO_WORKER_TOKENS,
    PROGRESS_LOG_INTERVAL_ROWS,
    RETRY_BACKOFF_INITIAL_SECONDS,
    RETRY_BACKOFF_MAX_SECONDS,
    WARMUP_ROWS,
    resolve_matrix_rows_constant,
)
from benchmarks.bench_utils import ensure_parent, parse_csv_tokens, stat_summary

DEFAULT_PARALLEL_WORKERS = get_runtime_defaults().query_defaults.default_parallel_workers


RETRIABLE_EXC = (
    URLError,
    HTTPError,
    OSError,
    TimeoutError,
    socket.timeout,
    OldWebserviceError,
    NewWebserviceError,
)

_TRANSPORT_MODES = frozenset({"direct", "tor"})
_LEGACY_TRANSPORT_PATCH_SIGNATURE: tuple[str, str] | None = None


class _LegacyResponseStream:
    """Adapter that exposes requests responses as iterable file-like bytes streams."""

    def __init__(self, response: requests.Response):
        self._response = response
        self._iter = response.iter_lines()
        self._cached_body: bytes | None = None
        self.headers = response.headers

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self._iter)
        if isinstance(line, bytes):
            return line
        return str(line).encode("utf-8")

    def read(self, size: int = -1):
        if self._cached_body is None:
            self._cached_body = bytes(self._response.content)
        if size is None or int(size) < 0:
            return self._cached_body
        return self._cached_body[: int(size)]

    def close(self):
        self._response.close()


def _resolve_transport_mode(value: str | None) -> str:
    mode = str(value or "direct").strip().lower()
    if mode not in _TRANSPORT_MODES:
        raise ValueError("transport_mode must be one of: direct, tor")
    return mode


def _resolve_proxy_url(*, transport_mode: str, tor_proxy_url_value: str | None) -> str | None:
    if transport_mode != "tor":
        return None
    raw = str(tor_proxy_url_value or tor_proxy_url()).strip()
    return enforce_tor_dns_safe_proxy_url(
        raw,
        tor_mode=True,
        context="benchmark tor proxy_url",
    )


def _configure_legacy_intermine_transport(*, mine_url: str, user_agent: str | None, proxy_url: str | None) -> None:
    global _LEGACY_TRANSPORT_PATCH_SIGNATURE
    if OldService is None:
        return
    ua = str(user_agent or "").strip()
    proxy = str(proxy_url or "").strip()
    signature = (ua, proxy)
    if _LEGACY_TRANSPORT_PATCH_SIGNATURE == signature:
        return

    import intermine.model as legacy_model
    import intermine.util as legacy_util
    from intermine.results import InterMineURLOpener as LegacyInterMineURLOpener

    def _request_stream(
        url: str,
        *,
        data=None,
        headers=None,
        method: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> _LegacyResponseStream:
        session = requests.Session()
        if proxy:
            session.proxies = {"http": proxy, "https": proxy}
            session.trust_env = False

        payload = data if isinstance(data, (bytes, bytearray)) else (None if data is None else str(data).encode("utf-8"))
        req_headers = dict(headers or {})
        if ua:
            req_headers.setdefault("User-Agent", ua)
            req_headers.setdefault("Accept-Encoding", "identity")
        if payload is not None and "Content-Type" not in req_headers:
            req_headers["Content-Type"] = "application/x-www-form-urlencoded; charset=utf-8"
        request_method = str(method or ("POST" if payload is not None else "GET")).upper()
        response = session.request(
            request_method,
            str(url),
            data=payload,
            headers=req_headers,
            stream=True,
            timeout=float(timeout_seconds),
        )
        if response.status_code >= 400:
            preview = response.text[:200].strip()
            raise RuntimeError(
                f"legacy transport request failed status={response.status_code} url={url} body={preview!r}"
            )
        return _LegacyResponseStream(response)

    def _patched_headers(self, content_type=None, accept=None):
        headers: dict[str, str] = {}
        if ua:
            headers["User-Agent"] = ua
            headers["Accept-Encoding"] = "identity"
        if getattr(self, "using_authentication", False):
            headers["Authorization"] = self.auth_header
        if content_type is not None:
            headers["Content-Type"] = content_type
        if accept is not None:
            headers["Accept"] = accept
        return headers

    def _patched_open(self, url, data=None, headers=None, method=None):
        req_headers = self.headers()
        if headers is not None:
            req_headers.update(headers)
        timeout_value = getattr(self, "request_timeout", None)
        if isinstance(timeout_value, (tuple, list)) and len(timeout_value) >= 2:
            timeout_seconds = float(timeout_value[1])
        elif timeout_value is None:
            timeout_seconds = 60.0
        else:
            timeout_seconds = float(timeout_value)
        return _request_stream(
            str(url),
            data=data,
            headers=req_headers,
            method=method,
            timeout_seconds=timeout_seconds,
        )

    def _patched_open_anything(source):
        text = str(source)
        parsed = urlparse(text)
        if parsed.scheme in {"http", "https"}:
            return _request_stream(text, method="GET")
        try:
            return open(source, "rb")
        except Exception:
            return io.StringIO(str(source))

    LegacyInterMineURLOpener.headers = _patched_headers
    LegacyInterMineURLOpener.open = _patched_open
    legacy_util.openAnything = _patched_open_anything
    legacy_model.openAnything = _patched_open_anything
    _LEGACY_TRANSPORT_PATCH_SIGNATURE = signature


@dataclass(slots=True)
class ModeRun:
    mode: str
    repetition: int
    seconds: float
    rows: int
    rows_per_s: float
    retries: int
    available_rows_per_pass: int | None
    effective_workers: int | None
    block_stats: dict[str, Any]
    stage_timings: dict[str, float]


def parse_positive_int_csv(
    text: str,
    arg_name: str,
    *,
    allow_auto: bool = False,
    required_count: int | None = None,
    auto_tokens: set[str] | frozenset[str] = AUTO_WORKER_TOKENS,
) -> list[int]:
    if allow_auto and text.strip().lower() in auto_tokens:
        return []
    values: list[int] = []
    for token in parse_csv_tokens(text):
        value = int(token)
        if value <= 0:
            raise ValueError(f"{arg_name} values must be positive integers")
        values.append(value)
    if not values and not (allow_auto and text.strip().lower() in auto_tokens):
        raise ValueError(f"{arg_name} resolved to an empty list")
    if required_count is not None and len(values) != required_count:
        raise ValueError(f"{arg_name} must contain exactly {required_count} values")
    return values


def parse_workers(text: str, auto_tokens: set[str] | frozenset[str]) -> list[int]:
    return parse_positive_int_csv(text, "--workers", allow_auto=True, auto_tokens=auto_tokens)


def parse_page_sizes(text: str | None, fallback_page_size: int) -> list[int]:
    if text is None:
        return [fallback_page_size]
    return parse_positive_int_csv(text, "--page-sizes")


def build_matrix_scenarios(
    args: argparse.Namespace,
    target_settings: dict[str, Any] | None,
    *,
    default_matrix_group_size: int = 5,
) -> list[dict[str, Any]]:
    rows_text = resolve_matrix_rows_constant(getattr(args, "matrix_rows", "MATRIX_ROWS"))
    matrix_profile = str(getattr(args, "matrix_profile", "auto"))
    if target_settings is not None:
        if target_settings.get("matrix_rows") is not None:
            rows_text = resolve_matrix_rows_constant(str(target_settings.get("matrix_rows")))
        if target_settings.get("matrix_profile") is not None:
            matrix_profile = str(target_settings.get("matrix_profile"))

    matrix_rows = parse_positive_int_csv(
        rows_text,
        "--matrix-rows",
        required_count=default_matrix_group_size,
    )

    return [
        {
            "name": f"matrix_{index}_{rows}",
            "rows_target": rows,
            "profile": matrix_profile,
            "group": "unified_matrix",
        }
        for index, rows in enumerate(matrix_rows, start=1)
    ]


def mode_label_for_workers(workers: int | None) -> str:
    if workers is None:
        return "intermine314_auto"
    return f"intermine314_w{workers}"


def resolve_benchmark_workers(mine_url: str, rows_target: int, configured_workers: int | None) -> int:
    if configured_workers is not None:
        return configured_workers
    plan = resolve_production_plan(mine_url, rows_target, workflow="elt", production_profile="auto")
    try:
        return int(plan["workers"])
    except Exception:
        return DEFAULT_PARALLEL_WORKERS


def resolve_execution_plan(
    *,
    mine_url: str,
    rows_target: int,
    explicit_workers: list[int],
    benchmark_profile: str,
    phase_default_include_legacy: bool,
) -> dict[str, Any]:
    return resolve_benchmark_execution_plan(
        mine_url=mine_url,
        rows_target=rows_target,
        explicit_workers=explicit_workers,
        benchmark_profile=benchmark_profile,
        phase_default_include_legacy=phase_default_include_legacy,
    )


def resolve_phase_plan(
    *,
    mine_url: str,
    rows_target: int,
    explicit_workers: list[int],
    benchmark_profile: str,
    phase_default_include_legacy: bool,
) -> dict[str, Any]:
    # Backward-compatible alias for existing benchmark imports.
    return resolve_execution_plan(
        mine_url=mine_url,
        rows_target=rows_target,
        explicit_workers=explicit_workers,
        benchmark_profile=benchmark_profile,
        phase_default_include_legacy=phase_default_include_legacy,
    )


def build_common_runtime_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "legacy_batch_size": args.legacy_batch_size,
        "parallel_window_factor": args.parallel_window_factor,
        "auto_chunking": args.auto_chunking,
        "chunk_target_seconds": args.chunk_target_seconds,
        "chunk_min_pages": args.chunk_min_pages,
        "chunk_max_pages": args.chunk_max_pages,
        "ordered_mode": args.ordered_mode,
        "ordered_window_pages": args.ordered_window_pages,
        "parallel_profile": args.parallel_profile,
        "large_query_mode": args.large_query_mode,
        "prefetch": args.prefetch,
        "inflight_limit": args.inflight_limit,
        "max_inflight_bytes_estimate": args.max_inflight_bytes_estimate,
        "transport_mode": getattr(args, "transport_mode", "direct"),
        "tor_proxy_url_value": getattr(args, "tor_proxy_url", None),
        "timeout_seconds": float(getattr(args, "timeout_seconds", 60)),
        "sleep_seconds": args.sleep_seconds,
        "max_retries": args.max_retries,
    }


def build_fetch_runtime_kwargs(args: argparse.Namespace, page_size: int) -> dict[str, Any]:
    return {
        "page_size": page_size,
        "randomize_mode_order": args.randomize_mode_order,
        **build_common_runtime_kwargs(args),
    }


def run_fetch_phase(
    *,
    phase_name: str,
    mine_url: str,
    rows_target: int,
    repetitions: int,
    phase_plan: dict[str, Any],
    args: argparse.Namespace,
    page_size: int,
    query_root_class: str,
    query_views: list[str],
    query_joins: list[str],
) -> dict[str, Any]:
    return run_replicated_fetch_benchmarks(
        phase_name=phase_name,
        mine_url=mine_url,
        rows_target=rows_target,
        repetitions=repetitions,
        workers=phase_plan["workers"],
        include_legacy_baseline=phase_plan["include_legacy_baseline"],
        query_root_class=query_root_class,
        query_views=query_views,
        query_joins=query_joins,
        **build_fetch_runtime_kwargs(args, page_size),
    )


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _retry_wait_seconds(attempt: int) -> float:
    return min(RETRY_BACKOFF_INITIAL_SECONDS * attempt, RETRY_BACKOFF_MAX_SECONDS)


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


def make_query(
    service_cls: Any,
    mine_url: str,
    root_class: str,
    views: list[str],
    joins: list[str],
    *,
    service_kwargs: dict[str, Any] | None = None,
) -> Any:
    if service_cls is None:
        raise RuntimeError(
            "Legacy intermine package is not installed; install optional benchmark deps "
            "(for example: pip install \"intermine314[benchmark]\")."
        )
    kwargs = dict(service_kwargs or {})
    service = service_cls(mine_url, **kwargs)
    query = service.new_query(root_class)
    query.add_view(*views)
    for join in joins:
        query.add_join(join, "OUTER")
    return query


def count_with_retry(
    query: Any,
    *,
    max_retries: int,
    sleep_seconds: float,
    rows_target: int,
) -> tuple[int | None, int, float, float]:
    retries = 0
    retry_backoff_sleep_seconds = 0.0
    optional_sleep_seconds = 0.0
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return query.count(), retries, retry_backoff_sleep_seconds, optional_sleep_seconds
        except RETRIABLE_EXC as exc:
            last_error = exc
            retries += 1
            wait_s = _retry_wait_seconds(attempt)
            print(f"count_retry attempt={attempt} err={exc} wait_s={wait_s:.1f}", flush=True)
            retry_backoff_sleep_seconds += float(wait_s)
            time.sleep(wait_s)
            if sleep_seconds > 0:
                optional_sleep_seconds += float(sleep_seconds)
                time.sleep(sleep_seconds)
    print(
        "count_fallback mode=streaming err=%s rows_target=%s"
        % (last_error, rows_target),
        flush=True,
    )
    return None, retries, retry_backoff_sleep_seconds, optional_sleep_seconds


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
    max_inflight_bytes_estimate: int | None,
    sleep_seconds: float,
    max_retries: int,
    csv_out_path: Path | None,
    query_root_class: str,
    query_views: list[str],
    query_joins: list[str],
    transport_mode: str = "direct",
    tor_proxy_url_value: str | None = None,
    timeout_seconds: float = 60.0,
) -> ModeRun:
    resolved_transport_mode = _resolve_transport_mode(transport_mode)
    active_proxy_url = _resolve_proxy_url(
        transport_mode=resolved_transport_mode,
        tor_proxy_url_value=tor_proxy_url_value,
    )
    mine_user_agent = resolve_mine_user_agent(mine_url)

    if mode == "intermine_batched":
        _configure_legacy_intermine_transport(
            mine_url=mine_url,
            user_agent=mine_user_agent,
            proxy_url=active_proxy_url,
        )
        service_kwargs: dict[str, Any] = {}
    else:
        service_kwargs = {
            "proxy_url": active_proxy_url,
            "tor": resolved_transport_mode == "tor",
            "user_agent": mine_user_agent,
            "request_timeout": (float(timeout_seconds), float(timeout_seconds)),
        }

    service_cls = OldService if mode == "intermine_batched" else NewService
    query = None
    init_error: Exception | None = None
    retry_backoff_sleep_seconds = 0.0
    optional_sleep_seconds_total = 0.0
    init_cpu_started = time.process_time()
    init_started = time.perf_counter()
    for attempt in range(1, max_retries + 1):
        try:
            query = make_query(
                service_cls,
                mine_url,
                query_root_class,
                query_views,
                query_joins,
                service_kwargs=service_kwargs,
            )
            break
        except Exception as exc:
            init_error = exc
            if attempt >= max_retries:
                break
            wait_s = _retry_wait_seconds(attempt)
            print(
                f"query_init_retry mode={mode} attempt={attempt} err={exc} wait_s={wait_s:.1f}",
                flush=True,
            )
            retry_backoff_sleep_seconds += float(wait_s)
            time.sleep(wait_s)
            if sleep_seconds > 0:
                optional_sleep_seconds_total += float(sleep_seconds)
                time.sleep(sleep_seconds)
    if query is None:
        raise RuntimeError(
            f"Failed query initialization after retries: mode={mode}, error={init_error}"
        )
    query_init_seconds = time.perf_counter() - init_started
    query_init_cpu_seconds = time.process_time() - init_cpu_started

    count_cpu_started = time.process_time()
    count_started = time.perf_counter()
    available_rows, retries, count_retry_backoff_sleep, count_optional_sleep = count_with_retry(
        query,
        max_retries=max_retries,
        sleep_seconds=sleep_seconds,
        rows_target=rows_target,
    )
    count_seconds = time.perf_counter() - count_started
    count_cpu_seconds = time.process_time() - count_cpu_started
    retry_backoff_sleep_seconds += float(count_retry_backoff_sleep)
    optional_sleep_seconds_total += float(count_optional_sleep)
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
    next_mark = PROGRESS_LOG_INTERVAL_ROWS
    # Compact numeric buffers: lower overhead than Python float lists on long runs.
    block_durations = array("d")
    chunk_sizes = array("d")
    csv_write_seconds = 0.0
    stream_fetch_decode_seconds = 0.0
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

    stream_cpu_started = time.process_time()
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
                        options = ParallelOptions(
                            page_size=page_size,
                            max_workers=workers,
                            ordered=ordered_mode,
                            prefetch=prefetch,
                            inflight_limit=inflight_limit,
                            max_inflight_bytes_estimate=max_inflight_bytes_estimate,
                            ordered_window_pages=ordered_window_pages,
                            profile=parallel_profile,
                            large_query_mode=large_query_mode,
                            pagination="auto",
                        )
                        iterator = query.run_parallel(
                            row="dict",
                            start=start,
                            size=size,
                            parallel_options=options,
                        )
                    for row in iterator:
                        if writer is None and file_handle is not None:
                            writer = csv.DictWriter(file_handle, fieldnames=list(row.keys()))
                            writer.writeheader()
                        if writer is not None:
                            write_started = time.perf_counter()
                            writer.writerow(row)
                            csv_write_seconds += time.perf_counter() - write_started
                        got += 1
                    block_seconds = time.perf_counter() - b0
                    stream_fetch_decode_seconds += block_seconds
                    block_durations.append(block_seconds)
                    chunk_sizes.append(float(size))
                    break
                except RETRIABLE_EXC as exc:
                    retries += 1
                    if mode != "intermine_batched" and auto_chunking:
                        chunk_pages = _clamp(max(1, chunk_pages // 2), chunk_min_pages, chunk_max_pages)
                        size = min(page_size * chunk_pages, remaining)
                    wait_s = _retry_wait_seconds(attempt)
                    print(
                        f"retry mode={mode} attempt={attempt} start={start} size={size} err={exc} wait_s={wait_s:.1f}",
                        flush=True,
                    )
                    retry_backoff_sleep_seconds += float(wait_s)
                    time.sleep(wait_s)
            else:
                raise RuntimeError(f"Failed block after retries: mode={mode}, start={start}, size={size}")

            if got == 0 and available_rows is None and start > 0:
                start = 0
                continue
            if got == 0:
                raise RuntimeError(f"Received 0 rows for mode={mode}, start={start}, size={size}")

            processed += got
            start += got
            if available_rows is not None and start >= available_rows:
                start = 0

            while processed >= next_mark:
                print(f"{mode}_progress rows={next_mark}", flush=True)
                next_mark += PROGRESS_LOG_INTERVAL_ROWS

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
                optional_sleep_seconds_total += float(sleep_seconds)
                time.sleep(sleep_seconds)
    finally:
        if file_handle is not None:
            file_handle.close()

    elapsed = time.perf_counter() - t0
    stream_cpu_seconds = time.process_time() - stream_cpu_started
    stream_decode_estimate_seconds = max(stream_fetch_decode_seconds - csv_write_seconds, 0.0)
    stage_timings = {
        "query_init_seconds": float(query_init_seconds),
        "query_init_cpu_seconds": float(query_init_cpu_seconds),
        "count_seconds": float(count_seconds),
        "count_cpu_seconds": float(count_cpu_seconds),
        "stream_seconds": float(elapsed),
        "stream_cpu_seconds": float(stream_cpu_seconds),
        "stream_fetch_decode_seconds": float(stream_fetch_decode_seconds),
        "stream_decode_estimate_seconds": float(stream_decode_estimate_seconds),
        "csv_write_seconds": float(csv_write_seconds),
        "retry_backoff_sleep_seconds": float(retry_backoff_sleep_seconds),
        "optional_sleep_seconds": float(optional_sleep_seconds_total),
        "setup_seconds": float(query_init_seconds + count_seconds),
        "total_with_setup_seconds": float(query_init_seconds + count_seconds + elapsed),
    }
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
            "block_count": float(len(block_durations)),
        },
        stage_timings=stage_timings,
    )


def run_mode_with_runtime(
    *,
    mode: str,
    mine_url: str,
    rows_target: int,
    page_size: int,
    workers: int | None,
    csv_out_path: Path | None,
    runtime_kwargs: dict[str, Any],
    query_root_class: str,
    query_views: list[str],
    query_joins: list[str],
) -> ModeRun:
    return run_mode(
        mode=mode,
        mine_url=mine_url,
        rows_target=rows_target,
        page_size=page_size,
        workers=workers,
        csv_out_path=csv_out_path,
        query_root_class=query_root_class,
        query_views=query_views,
        query_joins=query_joins,
        **runtime_kwargs,
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
    max_inflight_bytes_estimate: int | None,
    randomize_mode_order: bool,
    sleep_seconds: float,
    max_retries: int,
    query_root_class: str,
    query_views: list[str],
    query_joins: list[str],
    transport_mode: str = "direct",
    tor_proxy_url_value: str | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    mode_defs: list[tuple[str, int | None]] = []
    if include_legacy_baseline:
        mode_defs.append(("intermine_batched", None))
    if workers:
        mode_defs.extend((mode_label_for_workers(w), w) for w in workers)
    else:
        mode_defs.append((mode_label_for_workers(None), None))

    all_runs: dict[str, list[ModeRun]] = {m: [] for m, _ in mode_defs}

    runtime_kwargs = {
        "legacy_batch_size": legacy_batch_size,
        "parallel_window_factor": parallel_window_factor,
        "auto_chunking": auto_chunking,
        "chunk_target_seconds": chunk_target_seconds,
        "chunk_min_pages": chunk_min_pages,
        "chunk_max_pages": chunk_max_pages,
        "ordered_mode": ordered_mode,
        "ordered_window_pages": ordered_window_pages,
        "parallel_profile": parallel_profile,
        "large_query_mode": large_query_mode,
        "prefetch": prefetch,
        "inflight_limit": inflight_limit,
        "max_inflight_bytes_estimate": max_inflight_bytes_estimate,
        "transport_mode": transport_mode,
        "tor_proxy_url_value": tor_proxy_url_value,
        "timeout_seconds": timeout_seconds,
        "sleep_seconds": sleep_seconds,
        "max_retries": max_retries,
    }
    warmup_rows = min(WARMUP_ROWS, rows_target)

    if include_legacy_baseline:
        _ = run_mode_with_runtime(
            mode="intermine_batched",
            mine_url=mine_url,
            rows_target=warmup_rows,
            page_size=page_size,
            workers=None,
            csv_out_path=None,
            runtime_kwargs=runtime_kwargs,
            query_root_class=query_root_class,
            query_views=query_views,
            query_joins=query_joins,
        )
    warmup_mode, warmup_workers = next((m, w) for m, w in mode_defs if m != "intermine_batched")
    _ = run_mode_with_runtime(
        mode=warmup_mode,
        mine_url=mine_url,
        rows_target=warmup_rows,
        page_size=page_size,
        workers=warmup_workers,
        csv_out_path=None,
        runtime_kwargs=runtime_kwargs,
        query_root_class=query_root_class,
        query_views=query_views,
        query_joins=query_joins,
    )

    randomized_orders: list[list[str]] = []
    for rep in range(1, repetitions + 1):
        rep_modes = list(mode_defs)
        if randomize_mode_order:
            random.shuffle(rep_modes)
        randomized_orders.append([mode for mode, _ in rep_modes])
        for mode, mode_workers in rep_modes:
            print(f"run_start phase={phase_name} mode={mode} repetition={rep}", flush=True)
            run = run_mode_with_runtime(
                mode=mode,
                mine_url=mine_url,
                rows_target=rows_target,
                page_size=page_size,
                workers=mode_workers,
                csv_out_path=None,
                runtime_kwargs=runtime_kwargs,
                query_root_class=query_root_class,
                query_views=query_views,
                query_joins=query_joins,
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

    def _stage_timing_summary(runs: list[ModeRun]) -> dict[str, Any]:
        keys = sorted({key for run in runs for key in run.stage_timings.keys()})
        out: dict[str, Any] = {}
        for key in keys:
            out[key] = stat_summary([float(run.stage_timings.get(key, 0.0)) for run in runs])
        return out

    for mode, runs in all_runs.items():
        secs = [r.seconds for r in runs]
        rps = [r.rows_per_s for r in runs]
        retries = [float(r.retries) for r in runs]
        mode_summary = {
            "seconds": stat_summary(secs),
            "rows_per_s": stat_summary(rps),
            "retries": stat_summary(retries),
            "available_rows_per_pass": runs[0].available_rows_per_pass if runs else None,
            "stage_timings_seconds": _stage_timing_summary(runs),
            "runs": [
                {
                    "repetition": r.repetition,
                    "seconds": r.seconds,
                    "rows": r.rows,
                    "rows_per_s": r.rows_per_s,
                    "retries": r.retries,
                    "effective_workers": r.effective_workers,
                    "block_stats": r.block_stats,
                    "stage_timings": r.stage_timings,
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


def run_mode_export_csv(
    *,
    log_mode: str,
    mode: str,
    mine_url: str,
    rows_target: int,
    page_size: int,
    workers: int | None,
    csv_out_path: Path,
    mode_runtime_kwargs: dict[str, Any],
    query_root_class: str,
    query_views: list[str],
    query_joins: list[str],
) -> ModeRun:
    print(f"export_start mode={log_mode}", flush=True)
    export = run_mode_with_runtime(
        mode=mode,
        mine_url=mine_url,
        rows_target=rows_target,
        page_size=page_size,
        workers=workers,
        csv_out_path=csv_out_path,
        runtime_kwargs=mode_runtime_kwargs,
        query_root_class=query_root_class,
        query_views=query_views,
        query_joins=query_joins,
    )
    print(
        f"export_done mode={log_mode} seconds={export.seconds:.3f} path={csv_out_path}",
        flush=True,
    )
    return export
