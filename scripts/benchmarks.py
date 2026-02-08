#!/usr/bin/env python3
"""Benchmark suite: intermine vs intermine314.

Features:
- default 6-scenario fetch matrix (10k/25k/50k with profile1; 100k/250k/500k with profile2)
- compatibility direct and parallel benchmark phases (optional via --no-matrix-six)
- adaptive auto-chunking for large pulls (dynamic block sizing)
- mine-aware worker defaults via intermine314 mine registry
- environment pinning (OS/kernel/Python/package versions)
- optional per-request sleep for public service etiquette
- storage comparison: 100k intermine CSV vs 100k intermine314 Parquet
- pandas(CSV) vs polars(Parquet) dataframe benchmark on large intermine314 export
- randomized mode order per repetition to reduce run-order bias
- target-configured targeted exports (core + edge tables) via chunked server-side lists
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import random
import socket
import subprocess
import sys
import types
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

from intermine314.bulk_export import (  # noqa: E402
    default_oakmine_targeted_tables,
    export_targeted_tables_with_lists,
)
from intermine314.constants import (  # noqa: E402
    DEFAULT_KEYSET_BATCH_SIZE,
    DEFAULT_LIST_CHUNK_SIZE,
    DEFAULT_PARALLEL_WORKERS,
    DEFAULT_TARGETED_EXPORT_PAGE_SIZE,
)
from intermine314.mine_registry import (  # noqa: E402
    DEFAULT_BENCHMARK_LARGE_PROFILE,
    DEFAULT_BENCHMARK_SMALL_PROFILE,
    resolve_preferred_workers,
)
from intermine314.webservice import Service as NewService  # noqa: E402
import intermine  # noqa: E402
import intermine314  # noqa: E402
from scripts.bench_fetch import (  # noqa: E402
    build_common_runtime_kwargs,
    build_matrix_scenarios,
    parse_page_sizes,
    parse_workers,
    resolve_phase_plan,
    run_fetch_phase,
)
from scripts.bench_io import (  # noqa: E402
    bench_pandas,
    bench_polars,
    export_for_storage,
    export_new_only_for_dataframe,
    infer_dataframe_columns,
)
from scripts.bench_targeting import (  # noqa: E402
    get_target_defaults,
    load_target_config,
    normalize_target_settings,
    normalize_targeted_settings,
    profile_for_rows,
    resolve_reachable_mine_url,
)
from scripts.bench_utils import ensure_parent, normalize_string_list, parse_csv_tokens  # noqa: E402

DEFAULT_MINE_URL = "https://maizemine.rnet.missouri.edu/maizemine"
DEFAULT_MATRIX_SMALL_ROWS = (10_000, 25_000, 50_000)
DEFAULT_MATRIX_LARGE_ROWS = (100_000, 250_000, 500_000)
DEFAULT_BENCHMARK_PAGE_SIZE = DEFAULT_TARGETED_EXPORT_PAGE_SIZE
DEFAULT_MATRIX_GROUP_SIZE = 3
AUTO_WORKER_TOKENS = frozenset({"auto", "registry", "mine"})


def _csv_from_ints(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mine-url",
        default=DEFAULT_MINE_URL,
        help="Mine root URL (without /service).",
    )
    parser.add_argument(
        "--benchmark-target",
        default="auto",
        help="Benchmark target preset key from config/benchmark-targets.toml, or 'auto'.",
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
        default=500_000,
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
        help="Benchmark profile set from mine registry; used when --workers=auto.",
    )
    parser.add_argument(
        "--query-views",
        default=None,
        help="Optional comma-separated view paths override.",
    )
    parser.add_argument(
        "--query-joins",
        default=None,
        help="Optional comma-separated join paths override.",
    )
    parser.add_argument(
        "--query-root",
        default=None,
        help="Optional query root class override (for example: Gene, Protein).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Repetitions per mode (recommendation: 3 to 5).",
    )
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
        "--dataframe-repetitions",
        type=int,
        default=3,
        help="Repetitions for pandas/polars dataframe benchmarks.",
    )
    parser.add_argument(
        "--matrix-six",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run six matrix fetch benchmarks (3 small profile1 + 3 large profile2).",
    )
    parser.add_argument(
        "--matrix-small-rows",
        default=_csv_from_ints(DEFAULT_MATRIX_SMALL_ROWS),
        help="Comma-separated rows for first matrix triplet.",
    )
    parser.add_argument(
        "--matrix-large-rows",
        default=_csv_from_ints(DEFAULT_MATRIX_LARGE_ROWS),
        help="Comma-separated rows for second matrix triplet.",
    )
    parser.add_argument(
        "--matrix-small-profile",
        default=DEFAULT_BENCHMARK_SMALL_PROFILE,
        help="Benchmark profile for the first matrix triplet.",
    )
    parser.add_argument(
        "--matrix-large-profile",
        default=DEFAULT_BENCHMARK_LARGE_PROFILE,
        help="Benchmark profile for the second matrix triplet.",
    )
    parser.add_argument(
        "--oakmine-targeted-exports",
        "--targeted-exports",
        dest="targeted_exports",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    return parser.parse_args()


def resolve_query_spec(
    *,
    args: argparse.Namespace,
    target_settings: dict[str, Any] | None,
) -> tuple[str, list[str], list[str]]:
    query_root_class = DEFAULT_QUERY_ROOT_CLASS
    query_views = list(DEFAULT_QUERY_VIEWS)
    query_joins = list(DEFAULT_QUERY_JOINS)

    if target_settings is not None:
        if target_settings.get("views"):
            query_views = list(target_settings["views"])
        if target_settings.get("joins"):
            query_joins = list(target_settings["joins"])
        if target_settings.get("root_class"):
            query_root_class = str(target_settings["root_class"])

    custom_views = parse_csv_tokens(args.query_views)
    custom_joins = parse_csv_tokens(args.query_joins)
    if args.query_root:
        query_root_class = str(args.query_root).strip()
    if custom_views:
        query_views = custom_views
    if custom_joins:
        query_joins = custom_joins
    return query_root_class, query_views, query_joins


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

    query_root_class, query_views, query_joins = resolve_query_spec(
        args=args,
        target_settings=target_settings,
    )

    endpoint_probe_errors: list[dict[str, str]] = []
    try:
        resolved_mine_url, endpoint_probe_errors = resolve_reachable_mine_url(args.mine_url, target_settings)
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

    direct_phase_plan = resolve_phase_plan(
        mine_url=args.mine_url,
        rows_target=args.baseline_rows,
        explicit_workers=workers,
        benchmark_profile=direct_profile_name,
        phase_default_include_legacy=True,
    )
    parallel_phase_plan = resolve_phase_plan(
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
        ),
        "query": {
            "root_class": query_root_class,
            "views": query_views,
            "outer_joins": query_joins,
        },
        "package_imports": {
            "intermine_module": getattr(intermine, "__file__", "unknown"),
            "intermine314_module": getattr(intermine314, "__file__", "unknown"),
            "intermine_version_attr": getattr(intermine, "VERSION", "unknown"),
            "intermine314_version_attr": getattr(intermine314, "VERSION", "unknown"),
        },
        "endpoint_probe_errors": endpoint_probe_errors,
    }

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

    if args.matrix_six:
        matrix_scenarios = build_matrix_scenarios(
            args,
            target_settings,
            default_matrix_group_size=DEFAULT_MATRIX_GROUP_SIZE,
        )
        print(f"matrix6_enabled scenarios={matrix_scenarios}", flush=True)
        matrix_by_page: dict[str, Any] = {}
        for page_size in page_sizes:
            page_key = f"page_size_{page_size}"
            scenario_runs: list[dict[str, Any]] = []
            for scenario in matrix_scenarios:
                scenario_plan = resolve_phase_plan(
                    mine_url=args.mine_url,
                    rows_target=scenario["rows_target"],
                    explicit_workers=workers,
                    benchmark_profile=scenario["profile"],
                    phase_default_include_legacy=True,
                )
                phase_name = scenario["name"]
                scenario_result = run_fetch_phase(
                    phase_name=phase_name,
                    mine_url=args.mine_url,
                    rows_target=scenario["rows_target"],
                    repetitions=args.repetitions,
                    phase_plan=scenario_plan,
                    args=args,
                    page_size=page_size,
                    query_root_class=query_root_class,
                    query_views=query_views,
                    query_joins=query_joins,
                )
                scenario_runs.append(
                    {
                        "name": scenario["name"],
                        "group": scenario["group"],
                        "rows_target": scenario["rows_target"],
                        "profile": scenario["profile"],
                        "phase_plan": scenario_plan,
                        "result": scenario_result,
                    }
                )
            matrix_by_page[page_key] = scenario_runs
        report["fetch_benchmark"] = {
            "matrix6_by_page_size": matrix_by_page,
        }
        if len(page_sizes) == 1:
            only_key = f"page_size_{page_sizes[0]}"
            report["fetch_benchmark"]["matrix6"] = matrix_by_page[only_key]
    else:
        fetch_baseline_by_page: dict[str, Any] = {}
        fetch_parallel_by_page: dict[str, Any] = {}
        for page_size in page_sizes:
            page_key = f"page_size_{page_size}"
            fetch_baseline_by_page[page_key] = run_fetch_phase(
                phase_name="direct_compare_baseline",
                mine_url=args.mine_url,
                rows_target=args.baseline_rows,
                repetitions=args.repetitions,
                phase_plan=direct_phase_plan,
                args=args,
                page_size=page_size,
                query_root_class=query_root_class,
                query_views=query_views,
                query_joins=query_joins,
            )
            fetch_parallel_by_page[page_key] = run_fetch_phase(
                phase_name="parallel_only_large",
                mine_url=args.mine_url,
                rows_target=args.parallel_rows,
                repetitions=args.repetitions,
                phase_plan=parallel_phase_plan,
                args=args,
                page_size=page_size,
                query_root_class=query_root_class,
                query_views=query_views,
                query_joins=query_joins,
            )
        report["fetch_benchmark"] = {
            "direct_compare_baseline_by_page_size": fetch_baseline_by_page,
            "parallel_only_large_by_page_size": fetch_parallel_by_page,
        }
        if len(page_sizes) == 1:
            only_key = f"page_size_{page_sizes[0]}"
            report["fetch_benchmark"]["direct_compare_baseline"] = fetch_baseline_by_page[only_key]
            report["fetch_benchmark"]["parallel_only_large"] = fetch_parallel_by_page[only_key]

    io_page_size = page_sizes[0]

    targeted_enabled = bool(targeted_settings.get("enabled", False))
    if args.benchmark_target == "oakmine" and targeted_settings == {} and args.targeted_exports:
        targeted_enabled = True

    if args.targeted_exports and targeted_enabled:
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

    mode_runtime_kwargs = build_common_runtime_kwargs(args)
    storage_compare_100k = export_for_storage(
        mine_url=args.mine_url,
        rows_target=args.baseline_rows,
        page_size=io_page_size,
        workers_for_new=benchmark_workers_for_storage,
        mode_runtime_kwargs=mode_runtime_kwargs,
        csv_old_path=Path(args.csv_old_path),
        parquet_new_path=Path(args.parquet_compare_path),
        query_root_class=query_root_class,
        query_views=query_views,
        query_joins=query_joins,
    )
    storage_new_large = export_new_only_for_dataframe(
        mine_url=args.mine_url,
        rows_target=args.parallel_rows,
        page_size=io_page_size,
        workers_for_new=benchmark_workers_for_dataframe,
        mode_runtime_kwargs=mode_runtime_kwargs,
        csv_new_path=Path(args.csv_new_path),
        parquet_new_path=Path(args.parquet_new_path),
        query_root_class=query_root_class,
        query_views=query_views,
        query_joins=query_joins,
    )
    report["storage"] = {
        "compare_100k_old_vs_new": storage_compare_100k,
        "new_only_large": storage_new_large,
    }

    dataframe_columns = infer_dataframe_columns(
        csv_path=Path(args.csv_new_path),
        root_class=query_root_class,
        views=query_views,
    )
    pandas_df = bench_pandas(
        Path(args.csv_new_path),
        repetitions=args.dataframe_repetitions,
        cds_column=dataframe_columns["cds_column"],
        length_column=dataframe_columns["length_column"],
        group_column=dataframe_columns["group_column"],
    )
    polars_df = bench_polars(
        Path(args.parquet_new_path),
        repetitions=args.dataframe_repetitions,
        cds_column=dataframe_columns["cds_column"],
        length_column=dataframe_columns["length_column"],
        group_column=dataframe_columns["group_column"],
    )
    report["dataframes"] = {
        "dataset": "intermine314_large_export",
        "columns": dataframe_columns,
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
