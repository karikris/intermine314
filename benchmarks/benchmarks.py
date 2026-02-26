#!/usr/bin/env python3
"""Benchmark suite: intermine vs intermine314.

Features:
- default 6-scenario fetch matrix (5k/10k/25k with small profile; 50k/100k/250k with large profile)
- dual query benchmark types per run: simple (single-table) and complex (two outer joins / three tables)
- compatibility direct and parallel benchmark phases (optional via --no-matrix-six)
- adaptive auto-chunking for large pulls (dynamic block sizing)
- mine-aware worker defaults via intermine314 mine registry
- environment pinning (OS/kernel/Python/package versions)
- optional per-request sleep for public service etiquette
- per-scenario matrix storage/load comparison: CSV vs Parquet (6 scenarios)
- pandas(CSV) vs polars(Parquet) dataframe benchmark on large intermine314 export
- parquet join-engine benchmark: DuckDB vs Polars (two full outer joins, write-to-disk timing)
- 10k batch-size sensitivity benchmark with smart worker assignment and Python 3.14 in-flight tuning
- randomized mode order per repetition to reduce run-order bias
- target-configured targeted exports (core + edge tables) via chunked server-side lists
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
from contextlib import contextmanager
import types
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


@contextmanager
def legacy_shims_context(enabled: bool):
    if not enabled:
        yield
        return
    import collections

    had_mutable_mapping = hasattr(collections, "MutableMapping")
    had_mapping = hasattr(collections, "Mapping")
    original_urlparse = sys.modules.get("urlparse")
    original_urllib = sys.modules.get("urllib")
    install_legacy_shims()
    try:
        yield
    finally:
        if not had_mutable_mapping and hasattr(collections, "MutableMapping"):
            delattr(collections, "MutableMapping")
        if not had_mapping and hasattr(collections, "Mapping"):
            delattr(collections, "Mapping")
        if original_urlparse is None:
            sys.modules.pop("urlparse", None)
        else:
            sys.modules["urlparse"] = original_urlparse
        if original_urllib is None:
            sys.modules.pop("urllib", None)
        else:
            sys.modules["urllib"] = original_urllib


from intermine314.config.constants import (
    DEFAULT_KEYSET_BATCH_SIZE,
    DEFAULT_LIST_CHUNK_SIZE,
    DEFAULT_PARALLEL_WORKERS,
    DEFAULT_TARGETED_EXPORT_PAGE_SIZE,
)
from intermine314.registry.mines import (
    DEFAULT_BENCHMARK_LARGE_PROFILE,
    DEFAULT_BENCHMARK_SMALL_PROFILE,
    resolve_execution_plan as resolve_registry_execution_plan,
    resolve_preferred_workers,
)
from benchmarks.bench_constants import (
    AUTO_WORKER_TOKENS,
    BATCH_SIZE_TEST_CHUNK_ROWS,
    BATCH_SIZE_TEST_ROWS,
    DEFAULT_BENCHMARK_MINE_URL,
    DEFAULT_MATRIX_GROUP_SIZE,
    DEFAULT_MATRIX_STORAGE_DIR,
    LARGE_MATRIX_ROWS,
    SMALL_MATRIX_ROWS,
    resolve_matrix_rows_constant,
)
from benchmarks.bench_targeting import (
    get_target_defaults,
    load_target_config,
    normalize_target_settings,
    normalize_targeted_settings,
    profile_for_rows,
    resolve_reachable_mine_url,
)
from benchmarks.bench_utils import ensure_parent, normalize_string_list, parse_csv_tokens

DEFAULT_MINE_URL = DEFAULT_BENCHMARK_MINE_URL
DEFAULT_BENCHMARK_PAGE_SIZE = DEFAULT_TARGETED_EXPORT_PAGE_SIZE

BENCH_ENV_VARS = (
    "INTERMINE314_BENCHMARK_MINE_URL",
    "INTERMINE314_BENCHMARK_TARGET",
    "INTERMINE314_BENCHMARK_BASELINE_ROWS",
    "INTERMINE314_BENCHMARK_PARALLEL_ROWS",
    "INTERMINE314_BENCHMARK_WORKERS",
    "INTERMINE314_BENCHMARK_PROFILE",
    "INTERMINE314_BENCHMARK_QUERY_ROOT",
    "INTERMINE314_BENCHMARK_QUERY_VIEWS",
    "INTERMINE314_BENCHMARK_QUERY_JOINS",
    "INTERMINE314_BENCHMARK_REPETITIONS",
    "INTERMINE314_BENCHMARK_MATRIX_SMALL_ROWS",
    "INTERMINE314_BENCHMARK_MATRIX_LARGE_ROWS",
    "INTERMINE314_BENCHMARK_MATRIX_SMALL_PROFILE",
    "INTERMINE314_BENCHMARK_MATRIX_LARGE_PROFILE",
    "INTERMINE314_BENCHMARK_MATRIX_STORAGE_COMPARE",
    "INTERMINE314_BENCHMARK_MATRIX_LOAD_REPETITIONS",
    "INTERMINE314_BENCHMARK_MATRIX_STORAGE_DIR",
    "INTERMINE314_BENCHMARK_BATCH_SIZE_TEST",
    "INTERMINE314_BENCHMARK_BATCH_SIZE_TEST_ROWS",
    "INTERMINE314_BENCHMARK_BATCH_SIZE_TEST_CHUNK_ROWS",
    "INTERMINE314_BENCHMARK_TARGETED_EXPORTS",
    "INTERMINE314_BENCHMARK_PAGES_OUT",
)


def _csv_from_ints(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def _env_text(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    if not stripped:
        return default
    return stripped


def _env_int(name: str, default: int) -> int:
    value = _env_text(name)
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = _env_text(name)
    if value is None:
        return default
    text = value.lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _active_benchmark_env_overrides() -> dict[str, str]:
    active: dict[str, str] = {}
    for name in BENCH_ENV_VARS:
        value = _env_text(name)
        if value is not None:
            active[name] = value
    return active


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


def resolve_execution_plan(
    *,
    mine_url: str,
    rows_target: int,
    explicit_workers: list[int],
    benchmark_profile: str,
    phase_default_include_legacy: bool,
) -> dict[str, Any]:
    return resolve_registry_execution_plan(
        service_root=mine_url,
        rows_target=rows_target,
        explicit_workers=explicit_workers,
        benchmark_profile=benchmark_profile,
        include_legacy_baseline=phase_default_include_legacy,
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
        "sleep_seconds": args.sleep_seconds,
        "max_retries": args.max_retries,
    }


def _load_fetch_runtime():
    from benchmarks import bench_fetch

    return bench_fetch


def _load_io_runtime():
    from benchmarks import bench_io

    return bench_io


def _load_pages_runtime():
    from benchmarks import bench_pages

    return bench_pages


DEFAULT_QUERY_VIEWS = (
    "Gene.primaryIdentifier",
    "Gene.transcripts.CDS.primaryIdentifier",
    "Gene.proteins.proteinDomainRegions.proteinDomain.primaryIdentifier",
    "Gene.proteins.length",
    "Gene.secondaryIdentifier",
    "Gene.symbol",
    "Gene.name",
    "Gene.briefDescription",
    "Gene.tairShortDescription",
    "Gene.tairCuratorSummary",
)

DEFAULT_QUERY_JOINS = (
    "Gene.transcripts",
    "Gene.transcripts.CDS",
    "Gene.proteins",
    "Gene.proteins.proteinDomainRegions",
    "Gene.proteins.proteinDomainRegions.proteinDomain",
)
DEFAULT_QUERY_ROOT_CLASS = "Gene"


def _resolve_arg_defaults() -> dict[str, Any]:
    matrix_small_rows_default = _csv_from_ints(SMALL_MATRIX_ROWS)
    matrix_large_rows_default = _csv_from_ints(LARGE_MATRIX_ROWS)
    batch_chunk_rows_default = _csv_from_ints(BATCH_SIZE_TEST_CHUNK_ROWS)
    return {
        "mine_url": _env_text("INTERMINE314_BENCHMARK_MINE_URL", DEFAULT_MINE_URL) or DEFAULT_MINE_URL,
        "benchmark_target": _env_text("INTERMINE314_BENCHMARK_TARGET", "auto") or "auto",
        "baseline_rows": _env_int("INTERMINE314_BENCHMARK_BASELINE_ROWS", 100_000),
        "parallel_rows": _env_int("INTERMINE314_BENCHMARK_PARALLEL_ROWS", 500_000),
        "workers": _env_text("INTERMINE314_BENCHMARK_WORKERS", "auto") or "auto",
        "benchmark_profile": _env_text("INTERMINE314_BENCHMARK_PROFILE", "auto") or "auto",
        "query_views": _env_text("INTERMINE314_BENCHMARK_QUERY_VIEWS"),
        "query_joins": _env_text("INTERMINE314_BENCHMARK_QUERY_JOINS"),
        "query_root": _env_text("INTERMINE314_BENCHMARK_QUERY_ROOT"),
        "repetitions": _env_int("INTERMINE314_BENCHMARK_REPETITIONS", 3),
        "matrix_small_rows": _env_text("INTERMINE314_BENCHMARK_MATRIX_SMALL_ROWS", matrix_small_rows_default)
        or matrix_small_rows_default,
        "matrix_large_rows": _env_text("INTERMINE314_BENCHMARK_MATRIX_LARGE_ROWS", matrix_large_rows_default)
        or matrix_large_rows_default,
        "matrix_small_profile": (
            _env_text("INTERMINE314_BENCHMARK_MATRIX_SMALL_PROFILE", DEFAULT_BENCHMARK_SMALL_PROFILE)
            or DEFAULT_BENCHMARK_SMALL_PROFILE
        ),
        "matrix_large_profile": (
            _env_text("INTERMINE314_BENCHMARK_MATRIX_LARGE_PROFILE", DEFAULT_BENCHMARK_LARGE_PROFILE)
            or DEFAULT_BENCHMARK_LARGE_PROFILE
        ),
        "matrix_storage_compare": _env_bool("INTERMINE314_BENCHMARK_MATRIX_STORAGE_COMPARE", True),
        "matrix_load_repetitions": _env_int("INTERMINE314_BENCHMARK_MATRIX_LOAD_REPETITIONS", 3),
        "matrix_storage_dir": (
            _env_text("INTERMINE314_BENCHMARK_MATRIX_STORAGE_DIR", DEFAULT_MATRIX_STORAGE_DIR)
            or DEFAULT_MATRIX_STORAGE_DIR
        ),
        "batch_size_test": _env_bool("INTERMINE314_BENCHMARK_BATCH_SIZE_TEST", True),
        "batch_size_test_rows": _env_int("INTERMINE314_BENCHMARK_BATCH_SIZE_TEST_ROWS", BATCH_SIZE_TEST_ROWS),
        "batch_size_test_chunk_rows": (
            _env_text(
                "INTERMINE314_BENCHMARK_BATCH_SIZE_TEST_CHUNK_ROWS",
                batch_chunk_rows_default,
            )
            or batch_chunk_rows_default
        ),
        "targeted_exports": _env_bool("INTERMINE314_BENCHMARK_TARGETED_EXPORTS", True),
        "pages_out": _env_text("INTERMINE314_BENCHMARK_PAGES_OUT", "docs/benchmarks/results")
        or "docs/benchmarks/results",
    }


def _add_selection_arguments(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    parser.add_argument(
        "--mine-url",
        default=defaults["mine_url"],
        help="Mine root URL (without /service).",
    )
    parser.add_argument(
        "--benchmark-target",
        default=defaults["benchmark_target"],
        help="Benchmark target preset key from benchmarks/profiles/benchmark-targets.toml, or 'auto'.",
    )
    parser.add_argument(
        "--baseline-rows",
        type=int,
        default=defaults["baseline_rows"],
        help="Rows for direct compare phase (intermine + intermine314).",
    )
    parser.add_argument(
        "--parallel-rows",
        type=int,
        default=defaults["parallel_rows"],
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
        default=defaults["workers"],
        help="Comma-separated worker list for intermine314, or 'auto' to use mine registry defaults.",
    )
    parser.add_argument(
        "--benchmark-profile",
        default=defaults["benchmark_profile"],
        help="Benchmark profile set from mine registry; used when --workers=auto.",
    )


def _add_query_arguments(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    parser.add_argument(
        "--query-views",
        default=defaults["query_views"],
        help="Optional comma-separated view paths override.",
    )
    parser.add_argument(
        "--query-joins",
        default=defaults["query_joins"],
        help="Optional comma-separated join paths override.",
    )
    parser.add_argument(
        "--query-root",
        default=defaults["query_root"],
        help="Optional query root class override (for example: Gene, Protein).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=defaults["repetitions"],
        help="Repetitions per mode (recommendation: 3 to 5).",
    )
    parser.add_argument(
        "--query-kinds",
        default="simple,complex",
        help="Comma-separated benchmark query kinds to run (simple,complex).",
    )


def _add_parallel_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_BENCHMARK_PAGE_SIZE,
        help="Page size for parallel runs.",
    )
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
        default=DEFAULT_BENCHMARK_PAGE_SIZE,
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


def _add_output_arguments(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    parser.add_argument(
        "--csv-old-path",
        default="/tmp/intermine314_benchmark_100k_intermine.csv",
        help="Output path for baseline legacy intermine CSV export.",
    )
    parser.add_argument(
        "--parquet-compare-path",
        default="/tmp/intermine314_benchmark_100k_intermine314.parquet",
        help="Output path for baseline intermine314 parquet export used in storage comparison.",
    )
    parser.add_argument(
        "--csv-new-path",
        default="/tmp/intermine314_benchmark_large_intermine314.csv",
        help="Output path for large intermine314 CSV export (for pandas benchmark).",
    )
    parser.add_argument(
        "--parquet-new-path",
        default="/tmp/intermine314_benchmark_large_intermine314.parquet",
        help="Output path for large intermine314 parquet export (for polars benchmark).",
    )
    parser.add_argument(
        "--json-out",
        default="/tmp/intermine314_benchmark.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--pages-out",
        default=defaults["pages_out"],
        help="Benchmark GitHub Pages output directory (run pages + index under results).",
    )
    parser.add_argument(
        "--dataframe-repetitions",
        type=int,
        default=3,
        help="Repetitions for pandas/polars dataframe benchmarks.",
    )


def _add_matrix_arguments(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    parser.add_argument(
        "--matrix-six",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run six matrix fetch benchmarks (3 small-profile rows + 3 large-profile rows).",
    )
    parser.add_argument(
        "--matrix-small-rows",
        default=defaults["matrix_small_rows"],
        help="Comma-separated rows for first matrix triplet.",
    )
    parser.add_argument(
        "--matrix-large-rows",
        default=defaults["matrix_large_rows"],
        help="Comma-separated rows for second matrix triplet.",
    )
    parser.add_argument(
        "--matrix-small-profile",
        default=defaults["matrix_small_profile"],
        help="Benchmark profile for the first matrix triplet.",
    )
    parser.add_argument(
        "--matrix-large-profile",
        default=defaults["matrix_large_profile"],
        help="Benchmark profile for the second matrix triplet.",
    )
    parser.add_argument(
        "--matrix-storage-compare",
        action=argparse.BooleanOptionalAction,
        default=defaults["matrix_storage_compare"],
        help="For each matrix scenario, export CSV+Parquet and compare size/load timings.",
    )
    parser.add_argument(
        "--matrix-load-repetitions",
        type=int,
        default=defaults["matrix_load_repetitions"],
        help="Load-time benchmark repetitions per matrix scenario (CSV vs Parquet).",
    )
    parser.add_argument(
        "--matrix-storage-dir",
        default=defaults["matrix_storage_dir"],
        help="Output directory for matrix scenario CSV/Parquet comparison artifacts.",
    )
    parser.add_argument(
        "--batch-size-test",
        action=argparse.BooleanOptionalAction,
        default=defaults["batch_size_test"],
        help="Run 10k batch-size sensitivity benchmark with smart worker assignment.",
    )
    parser.add_argument(
        "--batch-size-test-rows",
        type=int,
        default=defaults["batch_size_test_rows"],
        help="Rows target for batch-size sensitivity benchmark.",
    )
    parser.add_argument(
        "--batch-size-test-chunk-rows",
        default=defaults["batch_size_test_chunk_rows"],
        help="Comma-separated chunk/page sizes for batch-size sensitivity benchmark.",
    )


def _add_targeted_arguments(parser: argparse.ArgumentParser, defaults: dict[str, Any]) -> None:
    parser.add_argument(
        "--oakmine-targeted-exports",
        "--targeted-exports",
        dest="targeted_exports",
        action=argparse.BooleanOptionalAction,
        default=defaults["targeted_exports"],
        help="Use target-configured core+edge exports via chunked server-side lists.",
    )
    parser.add_argument(
        "--targeted-output-dir",
        default="/tmp/intermine314_targeted_exports",
        help="Base directory for targeted core/edge parquet exports.",
    )
    parser.add_argument(
        "--targeted-id-chunk-size",
        type=int,
        default=DEFAULT_LIST_CHUNK_SIZE,
        help="List chunk size used for targeted exports.",
    )
    parser.add_argument(
        "--targeted-max-ids",
        type=int,
        default=None,
        help="Optional cap on unique IDs exported in targeted mode.",
    )
    parser.add_argument(
        "--targeted-template-limit",
        type=int,
        default=40,
        help="Maximum number of candidate templates to consider.",
    )
    parser.add_argument(
        "--targeted-use-templates-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Try list-compatible templates before custom queries in targeted exports.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = _resolve_arg_defaults()
    _add_selection_arguments(parser, defaults)
    _add_query_arguments(parser, defaults)
    _add_parallel_runtime_arguments(parser)
    _add_output_arguments(parser, defaults)
    _add_matrix_arguments(parser, defaults)
    _add_targeted_arguments(parser, defaults)
    return parser.parse_args(argv)


def resolve_query_spec(
    *,
    args: argparse.Namespace,
    target_settings: dict[str, Any] | None,
) -> tuple[str, list[str], list[str]]:
    query_root_class = DEFAULT_QUERY_ROOT_CLASS
    query_views = list(DEFAULT_QUERY_VIEWS)
    query_joins = list(DEFAULT_QUERY_JOINS)

    if target_settings is not None:
        if "views" in target_settings:
            query_views = normalize_string_list(target_settings.get("views"))
        if "joins" in target_settings:
            query_joins = normalize_string_list(target_settings.get("joins"))
        if target_settings.get("root_class"):
            query_root_class = str(target_settings["root_class"])

    custom_views = parse_csv_tokens(args.query_views)
    custom_joins = parse_csv_tokens(args.query_joins)
    if args.query_root:
        query_root_class = str(args.query_root).strip()
    if args.query_views is not None:
        query_views = custom_views
    if args.query_joins is not None:
        query_joins = custom_joins
    return query_root_class, query_views, query_joins


def _split_query_path(path: str) -> list[str]:
    return [part for part in str(path).split(".") if part]


def _is_root_attribute_view(root_class: str, view: str) -> bool:
    parts = _split_query_path(view)
    return len(parts) == 2 and parts[0] == root_class


def _infer_joins_from_views(root_class: str, views: list[str]) -> list[str]:
    joins: list[str] = []
    seen: set[str] = set()
    for view in views:
        parts = _split_query_path(view)
        if len(parts) <= 2 or parts[0] != root_class:
            continue
        prefix = parts[0]
        for segment in parts[1:-1]:
            prefix = f"{prefix}.{segment}"
            if prefix not in seen:
                seen.add(prefix)
                joins.append(prefix)
    return joins


def _complex_query_from_targeted_tables(
    target_settings: dict[str, Any] | None,
    fallback_root_class: str,
) -> tuple[str, list[str], list[str]] | None:
    if not isinstance(target_settings, dict):
        return None
    targeted = target_settings.get("targeted_exports")
    if not isinstance(targeted, dict):
        return None
    tables = targeted.get("tables")
    if not isinstance(tables, list):
        return None
    for table in tables:
        if not isinstance(table, dict):
            continue
        root_class = str(table.get("root_class") or fallback_root_class).strip() or fallback_root_class
        views = normalize_string_list(table.get("views"))
        joins = normalize_string_list(table.get("joins"))
        if not views:
            continue
        if not joins:
            joins = _infer_joins_from_views(root_class, views)
        if joins:
            return root_class, views, joins
    return None


def _first_hop_join(root_class: str, join_path: str) -> str | None:
    parts = _split_query_path(join_path)
    if len(parts) < 2 or parts[0] != root_class:
        return None
    return f"{root_class}.{parts[1]}"


def _enforce_two_outer_join_shape(
    root_class: str,
    views: list[str],
    joins: list[str],
) -> tuple[list[str], list[str]]:
    # Use first-hop joins only so complex mode is always root + 2 joined tables.
    join_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in list(joins) + _infer_joins_from_views(root_class, views):
        first_hop = _first_hop_join(root_class, candidate)
        if first_hop and first_hop not in seen:
            seen.add(first_hop)
            join_candidates.append(first_hop)
    if len(join_candidates) < 2:
        return views, joins

    selected_joins = join_candidates[:2]
    selected_views: list[str] = []
    root_primary = f"{root_class}.primaryIdentifier"
    if root_primary in views:
        selected_views.append(root_primary)

    for view in views:
        if _is_root_attribute_view(root_class, view) and view not in selected_views:
            selected_views.append(view)
        if len(selected_views) >= 2:
            break

    for join in selected_joins:
        prefix = f"{join}."
        joined_view = next((view for view in views if str(view).startswith(prefix)), None)
        if joined_view and joined_view not in selected_views:
            selected_views.append(joined_view)

    if not selected_views:
        selected_views = list(views)

    return selected_views, selected_joins


def resolve_query_benchmark_specs(
    *,
    args: argparse.Namespace,
    target_settings: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    base_root, base_views, base_joins = resolve_query_spec(args=args, target_settings=target_settings)

    complex_root = base_root
    complex_views = list(base_views)
    complex_joins = list(base_joins)

    if not complex_joins:
        complex_joins = _infer_joins_from_views(complex_root, complex_views)
    if not complex_joins:
        targeted_complex = _complex_query_from_targeted_tables(target_settings, complex_root)
        if targeted_complex is not None:
            complex_root, complex_views, complex_joins = targeted_complex

    complex_views, complex_joins = _enforce_two_outer_join_shape(
        complex_root,
        complex_views,
        complex_joins,
    )

    simple_root = complex_root
    simple_views = [view for view in complex_views if _is_root_attribute_view(simple_root, view)]
    if not simple_views:
        simple_views = [view for view in base_views if _is_root_attribute_view(simple_root, view)]
    if not simple_views and complex_views:
        root_prefix = f"{simple_root}."
        first_root_view = next((view for view in complex_views if str(view).startswith(root_prefix)), complex_views[0])
        simple_views = [first_root_view]

    return {
        "simple": {
            "root_class": simple_root,
            "views": simple_views,
            "joins": [],
        },
        "complex": {
            "root_class": complex_root,
            "views": complex_views,
            "joins": complex_joins,
        },
    }


def resolve_batch_size_chunk_rows(value: str) -> list[int]:
    resolved = resolve_matrix_rows_constant(str(value))
    chunks = parse_positive_int_csv(resolved, "--batch-size-test-chunk-rows")
    return sorted({int(chunk) for chunk in chunks if int(chunk) > 0})


def assign_workers_for_chunk_size(
    chunk_rows: int,
    rows_target: int,
    profile_workers: list[int],
) -> int:
    unique_workers = sorted({int(worker) for worker in profile_workers if int(worker) > 0})
    if not unique_workers:
        return DEFAULT_PARALLEL_WORKERS
    if len(unique_workers) == 1:
        return unique_workers[0]

    pages = max(1, int(math.ceil(rows_target / float(max(1, chunk_rows)))))
    if pages >= 120:
        rank = 1.00
    elif pages >= 60:
        rank = 0.80
    elif pages >= 20:
        rank = 0.60
    elif pages >= 8:
        rank = 0.35
    else:
        rank = 0.0

    index = int(round((len(unique_workers) - 1) * rank))
    index = max(0, min(index, len(unique_workers) - 1))
    return unique_workers[index]


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


def capture_environment(
    args: argparse.Namespace,
    workers: list[int],
    page_sizes: list[int],
    direct_phase_plan: dict[str, Any],
    parallel_phase_plan: dict[str, Any],
    target_settings: dict[str, Any] | None,
    direct_profile_name: str,
    parallel_profile_name: str,
    query_root_class: str,
    query_views: list[str],
    query_joins: list[str],
    benchmark_env_overrides: dict[str, str],
) -> dict[str, Any]:
    os_release = read_os_release()
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        )
    except Exception:
        git_commit = "unknown"

    resolved_baseline_workers = resolve_preferred_workers(
        args.mine_url, args.baseline_rows, DEFAULT_PARALLEL_WORKERS
    )
    resolved_parallel_workers = resolve_preferred_workers(
        args.mine_url, args.parallel_rows, DEFAULT_PARALLEL_WORKERS
    )

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
            "matrix_six": args.matrix_six,
            "matrix_small_rows": args.matrix_small_rows,
            "matrix_large_rows": args.matrix_large_rows,
            "matrix_small_profile": args.matrix_small_profile,
            "matrix_large_profile": args.matrix_large_profile,
            "matrix_storage_compare": args.matrix_storage_compare,
            "matrix_load_repetitions": args.matrix_load_repetitions,
            "matrix_storage_dir": args.matrix_storage_dir,
            "batch_size_test": args.batch_size_test,
            "batch_size_test_rows": args.batch_size_test_rows,
            "batch_size_test_chunk_rows": args.batch_size_test_chunk_rows,
            "benchmark_profile": args.benchmark_profile,
            "benchmark_target": args.benchmark_target,
            "target_settings": target_settings,
            "resolved_phase_profiles": {
                "direct_compare_baseline": direct_profile_name,
                "parallel_only_large": parallel_profile_name,
            },
            "mine_registry_workers": {
                "baseline_rows": resolved_baseline_workers,
                "parallel_rows": resolved_parallel_workers,
            },
            "phase_plans": {
                "direct_compare_baseline": direct_phase_plan,
                "parallel_only_large": parallel_phase_plan,
            },
            "query_root": query_root_class,
            "query_views": query_views,
            "query_joins": query_joins,
            "benchmark_env_overrides": benchmark_env_overrides,
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
            "targeted_exports": args.targeted_exports,
            "targeted_output_dir": args.targeted_output_dir,
            "targeted_id_chunk_size": args.targeted_id_chunk_size,
            "targeted_max_ids": args.targeted_max_ids,
            "targeted_template_limit": args.targeted_template_limit,
            "targeted_use_templates_first": args.targeted_use_templates_first,
            "csv_old_path": args.csv_old_path,
            "csv_new_path": args.csv_new_path,
            "parquet_compare_path": args.parquet_compare_path,
            "parquet_new_path": args.parquet_new_path,
            "json_out": args.json_out,
            "pages_out": args.pages_out,
        },
    }


def _query_output_path(base_path: str, query_kind: str) -> Path:
    base = Path(base_path)
    if base.suffix:
        stem = base.name[: -len(base.suffix)]
        filename = f"{stem}_{query_kind}{base.suffix}"
    else:
        filename = f"{base.name}_{query_kind}"
    return base.with_name(filename)


def _run_matrix_fetch_benchmark(
    *,
    fetch_runtime,
    io_runtime,
    args: argparse.Namespace,
    target_settings: dict[str, Any] | None,
    query_kind: str,
    page_sizes: list[int],
    workers: list[int],
    mode_runtime_kwargs: dict[str, Any],
    query_root_class_local: str,
    query_views_local: list[str],
    query_joins_local: list[str],
) -> dict[str, Any]:
    matrix_scenarios = fetch_runtime.build_matrix_scenarios(
        args,
        target_settings,
        default_matrix_group_size=DEFAULT_MATRIX_GROUP_SIZE,
    )
    print(f"matrix6_enabled query={query_kind} scenarios={matrix_scenarios}", flush=True)
    matrix_by_page: dict[str, Any] = {}
    for page_size in page_sizes:
        page_key = f"page_size_{page_size}"
        scenario_runs: list[dict[str, Any]] = []
        for scenario in matrix_scenarios:
            scenario_plan = resolve_execution_plan(
                mine_url=args.mine_url,
                rows_target=scenario["rows_target"],
                explicit_workers=workers,
                benchmark_profile=scenario["profile"],
                phase_default_include_legacy=True,
            )
            phase_name = f"{query_kind}_{scenario['name']}"
            scenario_result = fetch_runtime.run_fetch_phase(
                phase_name=phase_name,
                mine_url=args.mine_url,
                rows_target=scenario["rows_target"],
                repetitions=args.repetitions,
                phase_plan=scenario_plan,
                args=args,
                page_size=page_size,
                query_root_class=query_root_class_local,
                query_views=query_views_local,
                query_joins=query_joins_local,
            )
            scenario_payload: dict[str, Any] = {
                "name": scenario["name"],
                "group": scenario["group"],
                "rows_target": scenario["rows_target"],
                "profile": scenario["profile"],
                "phase_plan": scenario_plan,
                "result": scenario_result,
            }
            if args.matrix_storage_compare:
                matrix_target_name = str(args.benchmark_target or "manual")
                matrix_output_dir = Path(args.matrix_storage_dir) / matrix_target_name / query_kind / page_key
                scenario_workers = list(scenario_plan["workers"])
                csv_mode = "intermine_batched"
                if not scenario_plan["include_legacy_baseline"] and scenario_workers:
                    csv_mode = f"intermine314_w{min(scenario_workers)}"
                parquet_mode = f"intermine314_w{max(scenario_workers)}" if scenario_workers else "intermine314_auto"
                print(
                    "matrix_storage_start "
                    f"query={query_kind} scenario={scenario['name']} rows={scenario['rows_target']} "
                    f"csv_mode={csv_mode} "
                    f"parquet_mode={parquet_mode}",
                    flush=True,
                )
                matrix_storage = io_runtime.export_matrix_storage_compare(
                    mine_url=args.mine_url,
                    scenario_name=f"{query_kind}_{scenario['name']}",
                    rows_target=scenario["rows_target"],
                    page_size=page_size,
                    workers=scenario_workers,
                    include_legacy_baseline=bool(scenario_plan["include_legacy_baseline"]),
                    mode_runtime_kwargs=mode_runtime_kwargs,
                    output_dir=matrix_output_dir,
                    load_repetitions=args.matrix_load_repetitions,
                    query_root_class=query_root_class_local,
                    query_views=query_views_local,
                    query_joins=query_joins_local,
                )
                scenario_payload["io_compare"] = matrix_storage
                size_stats = matrix_storage["sizes"]
                load_stats = matrix_storage["load_benchmark"]
                csv_load_mean = load_stats["csv_load_seconds_pandas"]["mean"]
                parquet_load_mean = load_stats["parquet_load_seconds_polars"]["mean"]
                print(
                    "matrix_storage_done "
                    f"query={query_kind} scenario={scenario['name']} rows={scenario['rows_target']} "
                    f"csv_bytes={size_stats['csv_bytes']} parquet_bytes={size_stats['parquet_bytes']} "
                    f"reduction_pct={size_stats['reduction_pct']:.2f} "
                    f"csv_load_mean_s={csv_load_mean:.3f} parquet_load_mean_s={parquet_load_mean:.3f}",
                    flush=True,
                )
            scenario_runs.append(scenario_payload)
        matrix_by_page[page_key] = scenario_runs

    fetch_benchmark: dict[str, Any] = {
        "matrix6_by_page_size": matrix_by_page,
    }
    if len(page_sizes) == 1:
        only_key = f"page_size_{page_sizes[0]}"
        fetch_benchmark["matrix6"] = matrix_by_page[only_key]
    return fetch_benchmark


def _run_standard_fetch_benchmark(
    *,
    fetch_runtime,
    args: argparse.Namespace,
    query_kind: str,
    page_sizes: list[int],
    direct_phase_plan: dict[str, Any],
    parallel_phase_plan: dict[str, Any],
    query_root_class_local: str,
    query_views_local: list[str],
    query_joins_local: list[str],
) -> dict[str, Any]:
    fetch_baseline_by_page: dict[str, Any] = {}
    fetch_parallel_by_page: dict[str, Any] = {}
    for page_size in page_sizes:
        page_key = f"page_size_{page_size}"
        fetch_baseline_by_page[page_key] = fetch_runtime.run_fetch_phase(
            phase_name=f"{query_kind}_direct_compare_baseline",
            mine_url=args.mine_url,
            rows_target=args.baseline_rows,
            repetitions=args.repetitions,
            phase_plan=direct_phase_plan,
            args=args,
            page_size=page_size,
            query_root_class=query_root_class_local,
            query_views=query_views_local,
            query_joins=query_joins_local,
        )
        fetch_parallel_by_page[page_key] = fetch_runtime.run_fetch_phase(
            phase_name=f"{query_kind}_parallel_only_large",
            mine_url=args.mine_url,
            rows_target=args.parallel_rows,
            repetitions=args.repetitions,
            phase_plan=parallel_phase_plan,
            args=args,
            page_size=page_size,
            query_root_class=query_root_class_local,
            query_views=query_views_local,
            query_joins=query_joins_local,
        )

    fetch_benchmark: dict[str, Any] = {
        "direct_compare_baseline_by_page_size": fetch_baseline_by_page,
        "parallel_only_large_by_page_size": fetch_parallel_by_page,
    }
    if len(page_sizes) == 1:
        only_key = f"page_size_{page_sizes[0]}"
        fetch_benchmark["direct_compare_baseline"] = fetch_baseline_by_page[only_key]
        fetch_benchmark["parallel_only_large"] = fetch_parallel_by_page[only_key]
    return fetch_benchmark


def _resolve_batch_profile_name(
    *,
    args: argparse.Namespace,
    target_settings: dict[str, Any] | None,
) -> str:
    matrix_small_profile = args.matrix_small_profile
    matrix_large_profile = args.matrix_large_profile
    profile_switch_rows = None
    if target_settings is not None:
        matrix_small_profile = str(target_settings.get("matrix_small_profile", matrix_small_profile))
        matrix_large_profile = str(target_settings.get("matrix_large_profile", matrix_large_profile))
        profile_switch_rows = target_settings.get("profile_switch_rows")

    use_small_matrix_profile = True
    if profile_switch_rows is not None:
        try:
            use_small_matrix_profile = args.batch_size_test_rows <= int(profile_switch_rows)
        except Exception:
            use_small_matrix_profile = True

    batch_profile_name = matrix_small_profile if use_small_matrix_profile else matrix_large_profile
    if not batch_profile_name or str(batch_profile_name).strip().lower() == "auto":
        batch_profile_name = profile_for_rows("auto", target_settings, args.batch_size_test_rows)
    return batch_profile_name


def _run_batch_size_sensitivity(
    *,
    fetch_runtime,
    args: argparse.Namespace,
    target_settings: dict[str, Any] | None,
    query_kind: str,
    query_root_class_local: str,
    query_views_local: list[str],
    query_joins_local: list[str],
    batch_size_chunk_rows: list[int],
) -> dict[str, Any]:
    batch_profile_name = _resolve_batch_profile_name(args=args, target_settings=target_settings)
    batch_profile_plan = resolve_execution_plan(
        mine_url=args.mine_url,
        rows_target=args.batch_size_test_rows,
        explicit_workers=[],
        benchmark_profile=batch_profile_name,
        phase_default_include_legacy=True,
    )
    profile_workers = list(batch_profile_plan["workers"])
    if not profile_workers:
        profile_workers = [
            int(resolve_preferred_workers(args.mine_url, args.batch_size_test_rows, DEFAULT_PARALLEL_WORKERS))
        ]
    phase_workers = sorted({int(worker) for worker in profile_workers if int(worker) > 0})
    if not phase_workers:
        phase_workers = [DEFAULT_PARALLEL_WORKERS]

    batch_runs: list[dict[str, Any]] = []
    for chunk_rows in batch_size_chunk_rows:
        assigned_workers = assign_workers_for_chunk_size(
            chunk_rows=chunk_rows,
            rows_target=args.batch_size_test_rows,
            profile_workers=phase_workers,
        )
        total_pages = max(1, int(math.ceil(args.batch_size_test_rows / float(chunk_rows))))
        tuned_args = argparse.Namespace(**vars(args))
        tuned_args.large_query_mode = True
        tuned_args.parallel_profile = "large_query"
        tuned_args.prefetch = max(2, assigned_workers * 2)
        tuned_args.inflight_limit = max(2, min(128, tuned_args.prefetch * 2, total_pages * 2))
        tuned_args.auto_chunking = True
        tuned_args.chunk_min_pages = 1
        tuned_args.chunk_max_pages = max(1, min(args.chunk_max_pages, total_pages))

        phase_plan = {
            "name": f"{batch_profile_name}_smart_chunk_{chunk_rows}",
            "workers": phase_workers,
            "include_legacy_baseline": bool(batch_profile_plan.get("include_legacy_baseline", False)),
        }
        print(
            "batch_size_test_run "
            f"query={query_kind} rows={args.batch_size_test_rows} chunk_rows={chunk_rows} "
            f"workers={phase_workers} tuning_anchor={assigned_workers} "
            f"include_legacy={phase_plan['include_legacy_baseline']} "
            f"prefetch={tuned_args.prefetch} inflight={tuned_args.inflight_limit}",
            flush=True,
        )
        phase_result = fetch_runtime.run_fetch_phase(
            phase_name=f"{query_kind}_batch_size_{chunk_rows}",
            mine_url=args.mine_url,
            rows_target=args.batch_size_test_rows,
            repetitions=args.repetitions,
            phase_plan=phase_plan,
            args=tuned_args,
            page_size=chunk_rows,
            query_root_class=query_root_class_local,
            query_views=query_views_local,
            query_joins=query_joins_local,
        )
        result_modes = [str(mode) for mode in phase_result.get("mode_order", [])]
        if not result_modes:
            result_modes = list((phase_result.get("results") or {}).keys())
        batch_runs.append(
            {
                "chunk_rows": chunk_rows,
                "workers_assigned": assigned_workers,
                "phase_workers": phase_workers,
                "profile": batch_profile_name,
                "phase_plan": phase_plan,
                "runtime_tuning": {
                    "prefetch": tuned_args.prefetch,
                    "inflight_limit": tuned_args.inflight_limit,
                    "parallel_profile": tuned_args.parallel_profile,
                    "large_query_mode": tuned_args.large_query_mode,
                    "auto_chunking": tuned_args.auto_chunking,
                    "chunk_min_pages": tuned_args.chunk_min_pages,
                    "chunk_max_pages": tuned_args.chunk_max_pages,
                },
                "mode": result_modes[0] if result_modes else f"intermine314_w{assigned_workers}",
                "modes": result_modes,
                "result": phase_result,
            }
        )

    return {
        "rows_target": args.batch_size_test_rows,
        "profile": batch_profile_name,
        "profile_include_legacy_baseline": bool(batch_profile_plan.get("include_legacy_baseline", False)),
        "profile_workers": profile_workers,
        "chunk_rows": batch_size_chunk_rows,
        "runs": batch_runs,
    }


def _run_storage_dataframe_join_benchmark(
    *,
    io_runtime,
    args: argparse.Namespace,
    query_kind: str,
    io_page_size: int,
    mode_runtime_kwargs: dict[str, Any],
    direct_phase_plan: dict[str, Any],
    benchmark_workers_for_storage: int | None,
    benchmark_workers_for_dataframe: int | None,
    query_root_class_local: str,
    query_views_local: list[str],
    query_joins_local: list[str],
) -> dict[str, Any]:
    query_csv_old_path = _query_output_path(args.csv_old_path, query_kind)
    query_parquet_compare_path = _query_output_path(args.parquet_compare_path, query_kind)
    query_csv_new_path = _query_output_path(args.csv_new_path, query_kind)
    query_parquet_new_path = _query_output_path(args.parquet_new_path, query_kind)

    if bool(direct_phase_plan.get("include_legacy_baseline", False)):
        storage_compare_baseline = io_runtime.export_for_storage(
            mine_url=args.mine_url,
            rows_target=args.baseline_rows,
            page_size=io_page_size,
            workers_for_new=benchmark_workers_for_storage,
            mode_runtime_kwargs=mode_runtime_kwargs,
            csv_old_path=query_csv_old_path,
            parquet_new_path=query_parquet_compare_path,
            query_root_class=query_root_class_local,
            query_views=query_views_local,
            query_joins=query_joins_local,
        )
    else:
        print(
            f"storage_compare_baseline_skipped query={query_kind} reason=legacy_baseline_disabled",
            flush=True,
        )
        storage_compare_baseline = {
            "skipped": True,
            "reason": "legacy baseline disabled for active direct phase plan",
        }
    storage_new_large = io_runtime.export_new_only_for_dataframe(
        mine_url=args.mine_url,
        rows_target=args.parallel_rows,
        page_size=io_page_size,
        workers_for_new=benchmark_workers_for_dataframe,
        mode_runtime_kwargs=mode_runtime_kwargs,
        csv_new_path=query_csv_new_path,
        parquet_new_path=query_parquet_new_path,
        query_root_class=query_root_class_local,
        query_views=query_views_local,
        query_joins=query_joins_local,
    )
    storage_payload = {
        "compare_baseline_old_vs_new": storage_compare_baseline,
        "compare_100k_old_vs_new": storage_compare_baseline,
        "new_only_large": storage_new_large,
    }

    dataframe_columns = io_runtime.infer_dataframe_columns(
        csv_path=query_csv_new_path,
        root_class=query_root_class_local,
        views=query_views_local,
    )
    pandas_df = io_runtime.bench_pandas(
        query_csv_new_path,
        repetitions=args.dataframe_repetitions,
        cds_column=dataframe_columns["cds_column"],
        length_column=dataframe_columns["length_column"],
        group_column=dataframe_columns["group_column"],
    )
    polars_df = io_runtime.bench_polars(
        query_parquet_new_path,
        repetitions=args.dataframe_repetitions,
        cds_column=dataframe_columns["cds_column"],
        length_column=dataframe_columns["length_column"],
        group_column=dataframe_columns["group_column"],
    )
    dataframes_payload = {
        "dataset": f"intermine314_large_export_{query_kind}",
        "columns": dataframe_columns,
        "pandas_csv": pandas_df,
        "polars_parquet": polars_df,
    }
    join_benchmark_dir = Path(args.matrix_storage_dir) / "join_engine_benchmarks" / query_kind
    join_engine_benchmark = io_runtime.bench_parquet_join_engines(
        parquet_path=query_parquet_new_path,
        repetitions=args.dataframe_repetitions,
        output_dir=join_benchmark_dir,
        join_key=dataframe_columns.get("group_column"),
    )
    return {
        "storage": storage_payload,
        "dataframes": dataframes_payload,
        "join_engines": join_engine_benchmark,
        "artifacts": {
            "csv_old_path": str(query_csv_old_path),
            "parquet_compare_path": str(query_parquet_compare_path),
            "csv_new_path": str(query_csv_new_path),
            "parquet_new_path": str(query_parquet_new_path),
            "join_engine_output_dir": str(join_benchmark_dir),
        },
    }


def run_query_benchmark(
    *,
    fetch_runtime,
    io_runtime,
    args: argparse.Namespace,
    target_settings: dict[str, Any] | None,
    query_kind: str,
    query_root_class_local: str,
    query_views_local: list[str],
    query_joins_local: list[str],
    workers: list[int],
    page_sizes: list[int],
    mode_runtime_kwargs: dict[str, Any],
    direct_phase_plan: dict[str, Any],
    parallel_phase_plan: dict[str, Any],
    batch_size_chunk_rows: list[int],
    io_page_size: int,
    benchmark_workers_for_storage: int | None,
    benchmark_workers_for_dataframe: int | None,
) -> dict[str, Any]:
    query_report: dict[str, Any] = {
        "query": {
            "root_class": query_root_class_local,
            "views": query_views_local,
            "outer_joins": query_joins_local,
        }
    }

    if args.matrix_six:
        query_report["fetch_benchmark"] = _run_matrix_fetch_benchmark(
            fetch_runtime=fetch_runtime,
            io_runtime=io_runtime,
            args=args,
            target_settings=target_settings,
            query_kind=query_kind,
            page_sizes=page_sizes,
            workers=workers,
            mode_runtime_kwargs=mode_runtime_kwargs,
            query_root_class_local=query_root_class_local,
            query_views_local=query_views_local,
            query_joins_local=query_joins_local,
        )
    else:
        query_report["fetch_benchmark"] = _run_standard_fetch_benchmark(
            fetch_runtime=fetch_runtime,
            args=args,
            query_kind=query_kind,
            page_sizes=page_sizes,
            direct_phase_plan=direct_phase_plan,
            parallel_phase_plan=parallel_phase_plan,
            query_root_class_local=query_root_class_local,
            query_views_local=query_views_local,
            query_joins_local=query_joins_local,
        )

    if args.batch_size_test:
        query_report["batch_size_sensitivity"] = _run_batch_size_sensitivity(
            fetch_runtime=fetch_runtime,
            args=args,
            target_settings=target_settings,
            query_kind=query_kind,
            query_root_class_local=query_root_class_local,
            query_views_local=query_views_local,
            query_joins_local=query_joins_local,
            batch_size_chunk_rows=batch_size_chunk_rows,
        )

    query_report.update(
        _run_storage_dataframe_join_benchmark(
            io_runtime=io_runtime,
            args=args,
            query_kind=query_kind,
            io_page_size=io_page_size,
            mode_runtime_kwargs=mode_runtime_kwargs,
            direct_phase_plan=direct_phase_plan,
            benchmark_workers_for_storage=benchmark_workers_for_storage,
            benchmark_workers_for_dataframe=benchmark_workers_for_dataframe,
            query_root_class_local=query_root_class_local,
            query_views_local=query_views_local,
            query_joins_local=query_joins_local,
        )
    )
    return query_report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    benchmark_env_overrides = _active_benchmark_env_overrides()
    if args.rows is not None:
        args.baseline_rows = args.rows
        args.parallel_rows = args.rows
    if args.chunk_target_seconds <= 0:
        raise ValueError("--chunk-target-seconds must be > 0")
    if args.chunk_min_pages <= 0:
        raise ValueError("--chunk-min-pages must be > 0")
    if args.chunk_max_pages < args.chunk_min_pages:
        raise ValueError("--chunk-max-pages must be >= --chunk-min-pages")
    if args.matrix_load_repetitions <= 0:
        raise ValueError("--matrix-load-repetitions must be > 0")
    if args.batch_size_test_rows <= 0:
        raise ValueError("--batch-size-test-rows must be > 0")
    batch_size_chunk_rows = resolve_batch_size_chunk_rows(args.batch_size_test_chunk_rows)
    if not batch_size_chunk_rows:
        raise ValueError("--batch-size-test-chunk-rows must contain at least one positive value")

    query_kinds_raw = [token.strip().lower() for token in parse_csv_tokens(args.query_kinds)]
    if not query_kinds_raw:
        raise ValueError("--query-kinds must include at least one kind")
    query_kinds: list[str] = []
    for kind in query_kinds_raw:
        if kind not in {"simple", "complex"}:
            raise ValueError("--query-kinds values must be one of: simple,complex")
        if kind not in query_kinds:
            query_kinds.append(kind)

    target_config = load_target_config()
    target_defaults = get_target_defaults(target_config)
    target_settings = normalize_target_settings(args.benchmark_target, target_config, target_defaults)
    targeted_settings = normalize_targeted_settings(target_settings, target_defaults)
    if target_settings is not None:
        target_endpoint = str(target_settings.get("endpoint", "")).strip()
        if target_endpoint and args.mine_url == DEFAULT_MINE_URL:
            args.mine_url = target_endpoint
        if args.repetitions == 3 and target_settings.get("recommended_repetitions") is not None:
            try:
                args.repetitions = int(target_settings.get("recommended_repetitions"))
            except Exception:
                pass

    query_benchmark_specs = resolve_query_benchmark_specs(
        args=args,
        target_settings=target_settings,
    )
    complex_query_spec = query_benchmark_specs["complex"]
    simple_query_spec = query_benchmark_specs["simple"]
    query_root_class = str(complex_query_spec["root_class"])
    query_views = list(complex_query_spec["views"])
    query_joins = list(complex_query_spec["joins"])

    endpoint_probe_errors: list[dict[str, str]] = []
    try:
        resolved_mine_url, endpoint_probe_errors = resolve_reachable_mine_url(
            args.mine_url,
            target_settings,
            request_timeout=args.timeout_seconds,
        )
        if resolved_mine_url != args.mine_url:
            print(f"endpoint_fallback from={args.mine_url} to={resolved_mine_url}", flush=True)
            args.mine_url = resolved_mine_url
    except Exception as exc:
        print(f"endpoint_probe_failed err={exc}", flush=True)
        raise

    direct_profile_name = profile_for_rows(args.benchmark_profile, target_settings, args.baseline_rows)
    parallel_profile_name = profile_for_rows(args.benchmark_profile, target_settings, args.parallel_rows)

    workers = parse_workers(args.workers, AUTO_WORKER_TOKENS)
    page_sizes = parse_page_sizes(args.page_sizes, args.page_size)
    socket.setdefaulttimeout(args.timeout_seconds)
    random.seed(42)

    direct_phase_plan = resolve_execution_plan(
        mine_url=args.mine_url,
        rows_target=args.baseline_rows,
        explicit_workers=workers,
        benchmark_profile=direct_profile_name,
        phase_default_include_legacy=True,
    )
    parallel_phase_plan = resolve_execution_plan(
        mine_url=args.mine_url,
        rows_target=args.parallel_rows,
        explicit_workers=workers,
        benchmark_profile=parallel_profile_name,
        phase_default_include_legacy=False,
    )
    if not direct_phase_plan["workers"]:
        raise ValueError("Resolved direct-phase worker list is empty")
    if not parallel_phase_plan["workers"]:
        raise ValueError("Resolved parallel-phase worker list is empty")

    benchmark_workers_for_storage: int | None = max(direct_phase_plan["workers"])
    benchmark_workers_for_dataframe: int | None = max(parallel_phase_plan["workers"])
    use_legacy_shims = (
        bool(direct_phase_plan.get("include_legacy_baseline", False))
        or bool(args.matrix_six)
        or bool(args.batch_size_test)
    )

    with legacy_shims_context(use_legacy_shims):
        fetch_runtime = _load_fetch_runtime()
        import intermine314
        try:
            import intermine
        except Exception:  # pragma: no cover - optional dependency in benchmark tooling
            intermine = types.SimpleNamespace(VERSION="not-installed", __file__="not-installed")
    io_runtime = _load_io_runtime()
    pages_runtime = _load_pages_runtime()

    report: dict[str, Any] = {
        "environment": capture_environment(
            args,
            workers,
            page_sizes,
            direct_phase_plan,
            parallel_phase_plan,
            target_settings,
            direct_profile_name,
            parallel_profile_name,
            query_root_class,
            query_views,
            query_joins,
            benchmark_env_overrides,
        ),
        "query": {
            "root_class": query_root_class,
            "views": query_views,
            "outer_joins": query_joins,
            "simple": simple_query_spec,
            "complex": complex_query_spec,
        },
        "package_imports": {
            "intermine_module": getattr(intermine, "__file__", "unknown"),
            "intermine314_module": getattr(intermine314, "__file__", "unknown"),
            "intermine_version_attr": getattr(intermine, "VERSION", "unknown"),
            "intermine314_version_attr": getattr(intermine314, "VERSION", "unknown"),
        },
        "endpoint_probe_errors": endpoint_probe_errors,
    }
    report["environment"]["runtime_config"]["query_benchmark_specs"] = query_benchmark_specs
    report["environment"]["runtime_config"]["batch_size_test_chunk_rows_resolved"] = batch_size_chunk_rows

    print("benchmark_start", flush=True)
    print(f"mine_url={args.mine_url}", flush=True)
    print(f"workers={workers if workers else 'auto(registry)'}", flush=True)
    print(f"benchmark_profile={args.benchmark_profile}", flush=True)
    print(f"benchmark_target={args.benchmark_target}", flush=True)
    print(f"resolved_direct_profile={direct_profile_name}", flush=True)
    print(f"resolved_parallel_profile={parallel_profile_name}", flush=True)
    print(f"direct_phase_plan={direct_phase_plan}", flush=True)
    print(f"parallel_phase_plan={parallel_phase_plan}", flush=True)
    print(f"page_sizes={page_sizes}", flush=True)
    print(f"repetitions={args.repetitions}", flush=True)
    print(f"baseline_rows={args.baseline_rows}", flush=True)
    print(f"parallel_rows={args.parallel_rows}", flush=True)
    print(f"sleep_seconds={args.sleep_seconds}", flush=True)
    print(f"matrix_storage_compare={args.matrix_storage_compare}", flush=True)
    print(f"matrix_load_repetitions={args.matrix_load_repetitions}", flush=True)
    print(f"batch_size_test={args.batch_size_test}", flush=True)
    print(f"batch_size_test_rows={args.batch_size_test_rows}", flush=True)
    print(f"batch_size_test_chunk_rows={batch_size_chunk_rows}", flush=True)
    print(f"query_simple={simple_query_spec}", flush=True)
    print(f"query_complex={complex_query_spec}", flush=True)

    mode_runtime_kwargs = build_common_runtime_kwargs(args)
    io_page_size = page_sizes[0]

    targeted_enabled = bool(targeted_settings.get("enabled", False))
    if args.benchmark_target == "oakmine" and targeted_settings == {} and args.targeted_exports:
        targeted_enabled = True

    if args.targeted_exports and targeted_enabled:
        from intermine314.export.targeted import (
            default_oakmine_targeted_tables,
            export_targeted_tables_with_lists,
        )
        from intermine314.service import Service as NewService

        targeted_tables = targeted_settings.get("tables")
        if not targeted_tables:
            if args.benchmark_target == "oakmine":
                targeted_tables = [
                    {
                        "name": table.name,
                        "root_class": table.root_class,
                        "views": table.views,
                        "joins": table.joins,
                        "template_names": table.template_names,
                        "template_keywords": table.template_keywords,
                    }
                    for table in default_oakmine_targeted_tables()
                ]
            else:
                raise ValueError(
                    f"targeted_exports.tables is required for benchmark target '{args.benchmark_target}'"
                )
        targeted_id_path = str(targeted_settings.get("id_path", f"{query_root_class}.primaryIdentifier"))
        targeted_list_type = str(targeted_settings.get("list_type", query_root_class))
        targeted_chunk_size = int(targeted_settings.get("list_chunk_size", args.targeted_id_chunk_size))
        targeted_keyset_batch_size = int(
            targeted_settings.get("keyset_batch_size", max(DEFAULT_KEYSET_BATCH_SIZE, io_page_size))
        )
        targeted_template_keywords = normalize_string_list(targeted_settings.get("template_keywords"))
        targeted_template_limit = int(targeted_settings.get("template_limit", args.targeted_template_limit))
        targeted_workers = max(parallel_phase_plan["workers"])
        targeted_output_dir = Path(args.targeted_output_dir) / args.benchmark_target
        targeted_list_name_prefix = targeted_settings.get("list_name_prefix")
        targeted_list_description = targeted_settings.get("list_description")
        targeted_list_tags = targeted_settings.get("list_tags")
        if targeted_list_tags is not None and not isinstance(targeted_list_tags, list):
            targeted_list_tags = None
        targeted_list_kwargs: dict[str, Any] = {}
        if targeted_list_name_prefix:
            targeted_list_kwargs["list_name_prefix"] = str(targeted_list_name_prefix)
        if targeted_list_description:
            targeted_list_kwargs["list_description"] = str(targeted_list_description)
        if targeted_list_tags is not None:
            targeted_list_kwargs["list_tags"] = targeted_list_tags

        print(f"targeted_export_start target={args.benchmark_target}", flush=True)
        targeted_service = NewService(args.mine_url)
        report["targeted_exports"] = export_targeted_tables_with_lists(
            service=targeted_service,
            root_class=query_root_class,
            identifier_path=targeted_id_path,
            output_dir=targeted_output_dir,
            table_specs=targeted_tables,
            id_limit=args.targeted_max_ids,
            list_chunk_size=targeted_chunk_size,
            page_size=io_page_size,
            max_workers=targeted_workers,
            ordered=args.ordered_mode,
            profile=args.parallel_profile,
            large_query_mode=args.large_query_mode,
            prefetch=args.prefetch,
            inflight_limit=args.inflight_limit,
            keyset_batch_size=max(1, targeted_keyset_batch_size),
            sleep_seconds=args.sleep_seconds,
            template_keywords=targeted_template_keywords,
            template_limit=targeted_template_limit,
            use_templates_first=args.targeted_use_templates_first,
            list_type=targeted_list_type,
            **targeted_list_kwargs,
        )
        print(f"targeted_export_done target={args.benchmark_target}", flush=True)

    query_benchmarks: dict[str, Any] = {}
    for query_kind in query_kinds:
        spec = query_benchmark_specs[query_kind]
        print(f"query_benchmark_start type={query_kind}", flush=True)
        query_benchmarks[query_kind] = run_query_benchmark(
            fetch_runtime=fetch_runtime,
            io_runtime=io_runtime,
            args=args,
            target_settings=target_settings,
            query_kind=query_kind,
            query_root_class_local=str(spec["root_class"]),
            query_views_local=list(spec["views"]),
            query_joins_local=list(spec["joins"]),
            workers=workers,
            page_sizes=page_sizes,
            mode_runtime_kwargs=mode_runtime_kwargs,
            direct_phase_plan=direct_phase_plan,
            parallel_phase_plan=parallel_phase_plan,
            batch_size_chunk_rows=batch_size_chunk_rows,
            io_page_size=io_page_size,
            benchmark_workers_for_storage=benchmark_workers_for_storage,
            benchmark_workers_for_dataframe=benchmark_workers_for_dataframe,
        )
        print(f"query_benchmark_done type={query_kind}", flush=True)

    report["query_benchmarks"] = query_benchmarks
    primary_kind = "complex" if "complex" in query_benchmarks else query_kinds[0]
    report["fetch_benchmark"] = query_benchmarks[primary_kind]["fetch_benchmark"]
    report["storage"] = query_benchmarks[primary_kind]["storage"]
    report["dataframes"] = query_benchmarks[primary_kind]["dataframes"]
    report["join_engines"] = query_benchmarks[primary_kind]["join_engines"]
    report["batch_size_sensitivity"] = query_benchmarks[primary_kind].get("batch_size_sensitivity", {})

    ensure_parent(Path(args.json_out))
    with Path(args.json_out).open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    pages_written = pages_runtime.append_benchmark_run_pages(
        Path(args.pages_out),
        report,
        json_report_path=args.json_out,
    )

    print(f"report_written={args.json_out}", flush=True)
    print(f"report_written_pages_index={pages_written.get('index_html')}", flush=True)
    print(f"report_written_pages_run={pages_written.get('run_html')}", flush=True)
    print("BENCHMARK_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
