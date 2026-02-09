from __future__ import annotations

from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean(values: list[float | None]) -> float | None:
    nums = [value for value in values if value is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _fmt_seconds(value: Any) -> str:
    num = _to_float(value)
    if num is None:
        return "n/a"
    return f"{num:.3f}"


def _fmt_rows_per_s(value: Any) -> str:
    num = _to_float(value)
    if num is None:
        return "n/a"
    return f"{num:,.2f}"


def _fmt_bytes(value: Any) -> str:
    num = _to_float(value)
    if num is None:
        return "n/a"
    return f"{int(num):,}"


def _fmt_int(value: Any) -> str:
    num = _to_float(value)
    if num is None:
        return "n/a"
    return str(int(round(num)))


def _fmt_pct(value: Any) -> str:
    num = _to_float(value)
    if num is None:
        return "n/a"
    return f"{num:+.2f}%"


def _fmt_x(value: Any) -> str:
    num = _to_float(value)
    if num is None:
        return "n/a"
    return f"{num:.2f}x"


def _value_by_repetition(runs: list[dict[str, Any]], key: str, rep_count: int) -> list[float | None]:
    lookup: dict[int, float | None] = {}
    for run in runs:
        try:
            repetition = int(run.get("repetition", -1))
        except Exception:
            continue
        if repetition <= 0:
            continue
        lookup[repetition] = _to_float(run.get(key))
    return [lookup.get(index) for index in range(1, rep_count + 1)]


def _render_rep_headers(prefix: str, rep_count: int) -> list[str]:
    return [f"{prefix} r{index}" for index in range(1, rep_count + 1)]


def _section_mode_order(section: dict[str, Any]) -> list[str]:
    mode_order = section.get("mode_order")
    results = section.get("results", {})
    if isinstance(mode_order, list) and mode_order:
        ordered = [str(mode) for mode in mode_order if mode in results]
    else:
        ordered = sorted(str(mode) for mode in results.keys())
    if "intermine_batched" in ordered:
        ordered = ["intermine_batched"] + [mode for mode in ordered if mode != "intermine_batched"]
    return ordered


def _effective_workers_label(runs: list[dict[str, Any]]) -> str:
    workers = sorted(
        {
            int(value)
            for value in (run.get("effective_workers") for run in runs)
            if value is not None
        }
    )
    if not workers:
        return "-"
    return "/".join(str(worker) for worker in workers)


def _render_fetch_combined_table(section_name: str, section: dict[str, Any]) -> list[str]:
    results = section.get("results", {})
    if not isinstance(results, dict) or not results:
        return []
    baseline_mode = str(section.get("reference_mode", ""))
    rep_count = int(section.get("repetitions", 0) or 0)
    if rep_count <= 0:
        rep_count = max(len((mode_data or {}).get("runs", [])) for mode_data in results.values())
    rep_count = max(rep_count, 1)

    lines = [
        f"#### {section_name}",
        "| Mode | Workers | "
        + " | ".join(_render_rep_headers("sec", rep_count))
        + " | sec mean | "
        + " | ".join(_render_rep_headers("rows/s", rep_count))
        + " | rows/s mean | "
        + " | ".join(_render_rep_headers("retry", rep_count))
        + " | retry mean | Baseline | Speedup | Faster by | Throughput increase |",
        "| :--- | :--- | "
        + " | ".join(["---:"] * rep_count)
        + " | ---: | "
        + " | ".join(["---:"] * rep_count)
        + " | ---: | "
        + " | ".join(["---:"] * rep_count)
        + " | ---: | :--- | ---: | ---: | ---: |",
    ]

    baseline_seconds = _to_float(results.get(baseline_mode, {}).get("seconds", {}).get("mean"))
    baseline_rps = _to_float(results.get(baseline_mode, {}).get("rows_per_s", {}).get("mean"))

    for mode in _section_mode_order(section):
        mode_data = results.get(mode, {})
        runs = mode_data.get("runs", [])
        sec_values = _value_by_repetition(runs, "seconds", rep_count)
        rps_values = _value_by_repetition(runs, "rows_per_s", rep_count)
        retry_values = _value_by_repetition(runs, "retries", rep_count)
        sec_mean = _to_float(mode_data.get("seconds", {}).get("mean")) or _mean(sec_values)
        rps_mean = _to_float(mode_data.get("rows_per_s", {}).get("mean")) or _mean(rps_values)
        retry_mean = _to_float(mode_data.get("retries", {}).get("mean")) or _mean(retry_values)

        if mode == baseline_mode:
            speedup, faster_by, throughput = "1.00x", "+0.00%", "+0.00%"
        else:
            reference = mode_data.get("vs_reference", {})
            speedup = _fmt_x(reference.get("speedup"))
            faster_by = _fmt_pct(reference.get("faster_pct"))
            throughput = _fmt_pct(reference.get("throughput_increase_pct"))
            if speedup == "n/a" and sec_mean is not None and baseline_seconds and sec_mean > 0:
                speedup = _fmt_x(baseline_seconds / sec_mean)
            if faster_by == "n/a" and sec_mean is not None and baseline_seconds:
                faster_by = _fmt_pct((baseline_seconds - sec_mean) / baseline_seconds * 100.0)
            if throughput == "n/a" and rps_mean is not None and baseline_rps:
                throughput = _fmt_pct((rps_mean - baseline_rps) / baseline_rps * 100.0)

        lines.append(
            f"| {mode} | {_effective_workers_label(runs)} | "
            + " | ".join(_fmt_seconds(value) for value in sec_values)
            + f" | {_fmt_seconds(sec_mean)} | "
            + " | ".join(_fmt_rows_per_s(value) for value in rps_values)
            + f" | {_fmt_rows_per_s(rps_mean)} | "
            + " | ".join(_fmt_int(value) for value in retry_values)
            + f" | {_fmt_seconds(retry_mean)} | {baseline_mode} | {speedup} | {faster_by} | {throughput} |"
        )
    lines.append("")
    return lines


def _worker_from_mode(mode: Any) -> int | None:
    text = str(mode or "")
    marker = "_w"
    if marker not in text:
        return None
    tail = text.rsplit(marker, 1)[-1]
    try:
        value = int(tail)
    except Exception:
        return None
    return value if value > 0 else None


def _least_worker_baseline(options: list[dict[str, Any]]) -> dict[str, Any]:
    def _worker_value(option: dict[str, Any]) -> int | None:
        explicit = option.get("workers")
        if explicit is not None:
            try:
                value = int(explicit)
            except Exception:
                value = None
            if value is not None and value > 0:
                return value
        parsed = _worker_from_mode(option.get("name"))
        if parsed is not None:
            return parsed
        name = str(option.get("name", "")).lower()
        if "intermine_batched" in name:
            return 1
        return None

    with_workers = [option for option in options if _worker_value(option) is not None]
    if with_workers:
        return min(with_workers, key=lambda option: int(_worker_value(option) or 0))
    return options[0]


def _collect_storage_contexts(query_report: dict[str, Any]) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []

    storage = query_report.get("storage", {})
    if isinstance(storage, dict):
        compare = storage.get("compare_baseline_old_vs_new", {})
        if isinstance(compare, dict) and not compare.get("skipped"):
            old_export = compare.get("old_export_csv", {})
            new_csv = compare.get("new_export_tmp_csv", {})
            new_parquet = compare.get("new_parquet", {})
            sizes = compare.get("sizes", {})

            new_worker = new_csv.get("worker_count")
            if new_worker is None:
                new_worker = _worker_from_mode(new_csv.get("mode")) or _worker_from_mode(new_parquet.get("mode"))

            csv_seconds = _to_float(new_csv.get("seconds")) or 0.0
            conversion = _to_float(new_parquet.get("conversion_seconds_from_tmp_csv")) or 0.0
            parquet_total = csv_seconds + conversion
            csv_rows_per_s = _to_float(new_csv.get("rows_per_s"))
            parquet_rows_per_s = None
            if csv_rows_per_s and parquet_total > 0:
                parquet_rows_per_s = (csv_rows_per_s * csv_seconds) / parquet_total

            contexts.append(
                {
                    "context": "direct_storage",
                    "sizes": sizes,
                    "load_benchmark": None,
                    "options": [
                        {
                            "name": str(old_export.get("mode", "intermine_batched")),
                            "format": "CSV",
                            "workers": old_export.get("worker_count"),
                            "seconds": [_to_float(old_export.get("seconds"))],
                            "rows_per_s": [_to_float(old_export.get("rows_per_s"))],
                            "bytes_values": [_to_float(sizes.get("csv_bytes"))],
                            "note": "legacy export",
                        },
                        {
                            "name": str(new_csv.get("mode", "intermine314_csv_source")),
                            "format": "CSV",
                            "workers": new_worker,
                            "seconds": [csv_seconds],
                            "rows_per_s": [csv_rows_per_s],
                            "bytes_values": [None],
                            "note": "temporary source before parquet conversion",
                        },
                        {
                            "name": str(new_parquet.get("mode", "intermine314_parquet_final")),
                            "format": "Parquet",
                            "workers": new_worker,
                            "seconds": [parquet_total],
                            "rows_per_s": [parquet_rows_per_s],
                            "bytes_values": [_to_float(sizes.get("parquet_bytes"))],
                            "note": f"includes conversion ({_fmt_seconds(conversion)} s)",
                        },
                    ],
                }
            )

    for section_name, _section, io_compare in _iter_fetch_sections(query_report):
        if not isinstance(io_compare, dict):
            continue
        csv_export = io_compare.get("csv_export", {})
        parquet_export = io_compare.get("parquet_export", {})
        sizes = io_compare.get("sizes", {})

        csv_mode = str(io_compare.get("csv_source_mode", "csv_source"))
        csv_workers = io_compare.get("csv_source_workers")
        parquet_mode = str(io_compare.get("parquet_source_mode", "parquet_source"))
        parquet_workers = io_compare.get("parquet_source_workers")

        source_csv_seconds = _to_float(parquet_export.get("source_csv_seconds")) or 0.0
        source_csv_rps = _to_float(parquet_export.get("source_csv_rows_per_s"))
        source_csv_rows = _to_float(parquet_export.get("source_csv_rows"))
        conversion = _to_float(parquet_export.get("conversion_seconds_from_csv")) or 0.0
        parquet_total = source_csv_seconds + conversion
        parquet_rps = None
        if source_csv_rows and parquet_total > 0:
            parquet_rps = source_csv_rows / parquet_total

        contexts.append(
            {
                "context": section_name,
                "sizes": sizes,
                "load_benchmark": io_compare.get("load_benchmark"),
                "options": [
                    {
                        "name": csv_mode,
                        "format": "CSV",
                        "workers": csv_workers,
                        "seconds": [_to_float(csv_export.get("seconds"))],
                        "rows_per_s": [_to_float(csv_export.get("rows_per_s"))],
                        "bytes_values": [_to_float(sizes.get("csv_bytes"))],
                        "note": "matrix csv source",
                    },
                    {
                        "name": f"{parquet_mode}_csv_source",
                        "format": "CSV",
                        "workers": parquet_workers,
                        "seconds": [source_csv_seconds],
                        "rows_per_s": [source_csv_rps],
                        "bytes_values": [None],
                        "note": "source csv before parquet conversion",
                    },
                    {
                        "name": f"{parquet_mode}_parquet_final",
                        "format": "Parquet",
                        "workers": parquet_workers,
                        "seconds": [parquet_total],
                        "rows_per_s": [parquet_rps],
                        "bytes_values": [_to_float(sizes.get("parquet_bytes"))],
                        "note": f"includes conversion ({_fmt_seconds(conversion)} s)",
                    },
                ],
            }
        )

    batch_size = query_report.get("batch_size_sensitivity", {})
    if isinstance(batch_size, dict):
        batch_runs = batch_size.get("runs", [])
        if isinstance(batch_runs, list):
            options: list[dict[str, Any]] = []
            for run in batch_runs:
                if not isinstance(run, dict):
                    continue
                result = run.get("result", {})
                if not isinstance(result, dict):
                    continue
                mode = str(run.get("mode", ""))
                mode_results = result.get("results", {})
                if not isinstance(mode_results, dict):
                    mode_results = {}
                mode_data = mode_results.get(mode, {})
                if not isinstance(mode_data, dict):
                    mode_data = {}
                mode_runs = mode_data.get("runs", [])
                if not isinstance(mode_runs, list):
                    mode_runs = []
                rep_count = int(result.get("repetitions", 0) or 0)
                if rep_count <= 0:
                    rep_count = len(mode_runs)
                rep_count = max(rep_count, 1)
                seconds = _value_by_repetition(mode_runs, "seconds", rep_count)
                rows_per_s = _value_by_repetition(mode_runs, "rows_per_s", rep_count)
                if not mode_runs:
                    seconds[0] = _to_float(mode_data.get("seconds", {}).get("mean"))
                    rows_per_s[0] = _to_float(mode_data.get("rows_per_s", {}).get("mean"))

                tuning = run.get("runtime_tuning", {})
                if not isinstance(tuning, dict):
                    tuning = {}
                chunk_rows = run.get("chunk_rows")
                note = (
                    f"chunk_rows={chunk_rows}, "
                    f"prefetch={tuning.get('prefetch')}, "
                    f"inflight={tuning.get('inflight_limit')}, "
                    f"profile={run.get('profile')}"
                )
                options.append(
                    {
                        "name": f"{mode}_chunk_{chunk_rows}",
                        "format": "Parallel",
                        "workers": run.get("workers_assigned"),
                        "seconds": seconds,
                        "rows_per_s": rows_per_s,
                        "bytes_values": [None] * rep_count,
                        "note": note,
                    }
                )
            if options:
                contexts.append(
                    {
                        "context": "batch_size_sensitivity",
                        "sizes": {},
                        "load_benchmark": None,
                        "options": options,
                    }
                )

    return contexts


def _render_combined_storage_table(query_report: dict[str, Any]) -> list[str]:
    contexts = _collect_storage_contexts(query_report)
    all_contexts: list[dict[str, Any]] = list(contexts)

    for context in contexts:
        load_benchmark = context.get("load_benchmark")
        if not isinstance(load_benchmark, dict):
            continue
        runs = load_benchmark.get("runs", [])
        if not isinstance(runs, list):
            runs = []
        rep_count = int(load_benchmark.get("repetitions", 0) or 0)
        if rep_count <= 0:
            rep_count = len(runs)
        rep_count = max(rep_count, 1)

        csv_seconds = _value_by_repetition(runs, "csv_load_seconds", rep_count) if runs else [None] * rep_count
        parquet_seconds = (
            _value_by_repetition(runs, "parquet_load_seconds", rep_count) if runs else [None] * rep_count
        )
        csv_rows = _value_by_repetition(runs, "csv_row_count", rep_count) if runs else [None] * rep_count
        parquet_rows = _value_by_repetition(runs, "parquet_row_count", rep_count) if runs else [None] * rep_count

        if not runs and rep_count > 0:
            csv_seconds[0] = _to_float(load_benchmark.get("csv_load_seconds_pandas", {}).get("mean"))
            parquet_seconds[0] = _to_float(load_benchmark.get("parquet_load_seconds_polars", {}).get("mean"))
            row_counts = load_benchmark.get("csv_row_counts", [])
            parquet_counts = load_benchmark.get("parquet_row_counts", [])
            if isinstance(row_counts, list) and row_counts:
                csv_rows[0] = _to_float(row_counts[0])
            if isinstance(parquet_counts, list) and parquet_counts:
                parquet_rows[0] = _to_float(parquet_counts[0])

        def _rps(rows: list[float | None], seconds: list[float | None]) -> list[float | None]:
            values: list[float | None] = []
            for row_count, sec in zip(rows, seconds):
                if row_count is None or sec is None or sec <= 0:
                    values.append(None)
                else:
                    values.append(row_count / sec)
            return values

        all_contexts.append(
            {
                "context": f"{context.get('context', 'matrix')}::load",
                "sizes": {},
                "load_benchmark": None,
                "options": [
                    {
                        "name": "pandas_csv_load",
                        "format": "CSV",
                        "workers": None,
                        "seconds": csv_seconds,
                        "rows_per_s": _rps(csv_rows, csv_seconds),
                        "bytes_values": [None] * rep_count,
                        "note": "matrix load benchmark",
                    },
                    {
                        "name": "polars_parquet_load",
                        "format": "Parquet",
                        "workers": None,
                        "seconds": parquet_seconds,
                        "rows_per_s": _rps(parquet_rows, parquet_seconds),
                        "bytes_values": [None] * rep_count,
                        "note": "matrix load benchmark",
                    },
                ],
            }
        )

    dataframes = query_report.get("dataframes", {})
    if isinstance(dataframes, dict):
        pandas_stats = dataframes.get("pandas_csv", {})
        polars_stats = dataframes.get("polars_parquet", {})
        pandas_runs = pandas_stats.get("runs", []) if isinstance(pandas_stats, dict) else []
        polars_runs = polars_stats.get("runs", []) if isinstance(polars_stats, dict) else []
        if not isinstance(pandas_runs, list):
            pandas_runs = []
        if not isinstance(polars_runs, list):
            polars_runs = []
        rep_count = max(len(pandas_runs), len(polars_runs), 1)

        pandas_load = _value_by_repetition(pandas_runs, "load_seconds", rep_count)
        polars_load = _value_by_repetition(polars_runs, "load_seconds", rep_count)
        pandas_suite = _value_by_repetition(pandas_runs, "analytics_suite_seconds", rep_count)
        polars_suite = _value_by_repetition(polars_runs, "analytics_suite_seconds", rep_count)
        polars_lazy = _value_by_repetition(polars_runs, "lazy_scan_seconds", rep_count)
        pandas_rows = _value_by_repetition(pandas_runs, "row_count", rep_count)
        polars_rows = _value_by_repetition(polars_runs, "row_count", rep_count)
        pandas_mem = _value_by_repetition(pandas_runs, "memory_bytes", rep_count)
        polars_mem = _value_by_repetition(polars_runs, "memory_bytes_estimated", rep_count)

        if not pandas_runs and rep_count > 0:
            pandas_load[0] = _to_float(pandas_stats.get("load_seconds", {}).get("mean"))
            pandas_suite[0] = _to_float(pandas_stats.get("analytics_suite_seconds", {}).get("mean"))
            mem = _to_float(pandas_stats.get("memory_bytes", {}).get("mean"))
            pandas_mem[0] = mem
            rows_list = pandas_stats.get("row_counts", [])
            if isinstance(rows_list, list) and rows_list:
                pandas_rows[0] = _to_float(rows_list[0])

        if not polars_runs and rep_count > 0:
            polars_load[0] = _to_float(polars_stats.get("load_seconds", {}).get("mean"))
            polars_suite[0] = _to_float(polars_stats.get("analytics_suite_seconds", {}).get("mean"))
            polars_lazy[0] = _to_float(polars_stats.get("lazy_scan_seconds", {}).get("mean"))
            mem = _to_float(polars_stats.get("memory_bytes_estimated", {}).get("mean"))
            polars_mem[0] = mem
            rows_list = polars_stats.get("row_counts", [])
            if isinstance(rows_list, list) and rows_list:
                polars_rows[0] = _to_float(rows_list[0])

        def _rps(rows: list[float | None], seconds: list[float | None]) -> list[float | None]:
            values: list[float | None] = []
            for row_count, sec in zip(rows, seconds):
                if row_count is None or sec is None or sec <= 0:
                    values.append(None)
                else:
                    values.append(row_count / sec)
            return values

        all_contexts.append(
            {
                "context": "dataframe",
                "sizes": {},
                "load_benchmark": None,
                "options": [
                    {
                        "name": "pandas_csv_load",
                        "format": "CSV",
                        "workers": None,
                        "seconds": pandas_load,
                        "rows_per_s": _rps(pandas_rows, pandas_load),
                        "bytes_values": pandas_mem,
                        "note": "dataframe load (memory bytes)",
                    },
                    {
                        "name": "polars_parquet_load",
                        "format": "Parquet",
                        "workers": None,
                        "seconds": polars_load,
                        "rows_per_s": _rps(polars_rows, polars_load),
                        "bytes_values": polars_mem,
                        "note": "dataframe load (estimated memory bytes)",
                    },
                    {
                        "name": "pandas_csv_analytics",
                        "format": "CSV",
                        "workers": None,
                        "seconds": pandas_suite,
                        "rows_per_s": _rps(pandas_rows, pandas_suite),
                        "bytes_values": pandas_mem,
                        "note": "dataframe analytics suite",
                    },
                    {
                        "name": "polars_parquet_analytics",
                        "format": "Parquet",
                        "workers": None,
                        "seconds": polars_suite,
                        "rows_per_s": _rps(polars_rows, polars_suite),
                        "bytes_values": polars_mem,
                        "note": "dataframe analytics suite",
                    },
                    {
                        "name": "polars_parquet_lazy_scan",
                        "format": "Parquet",
                        "workers": None,
                        "seconds": polars_lazy,
                        "rows_per_s": _rps(polars_rows, polars_lazy),
                        "bytes_values": polars_mem,
                        "note": "lazy scan benchmark",
                    },
                ],
            }
        )

    join_engines = query_report.get("join_engines", {})
    if isinstance(join_engines, dict):
        duckdb = join_engines.get("duckdb", {})
        polars = join_engines.get("polars", {})
        duck_runs = duckdb.get("runs", []) if isinstance(duckdb, dict) else []
        polars_runs = polars.get("runs", []) if isinstance(polars, dict) else []
        if not isinstance(duck_runs, list):
            duck_runs = []
        if not isinstance(polars_runs, list):
            polars_runs = []
        rep_count = max(len(duck_runs), len(polars_runs), 1)
        duck_secs = _value_by_repetition(duck_runs, "seconds", rep_count)
        polars_secs = _value_by_repetition(polars_runs, "seconds", rep_count)
        duck_rows = _value_by_repetition(duck_runs, "row_count", rep_count)
        polars_rows = _value_by_repetition(polars_runs, "row_count", rep_count)
        duck_bytes = _value_by_repetition(duck_runs, "output_bytes", rep_count)
        polars_bytes = _value_by_repetition(polars_runs, "output_bytes", rep_count)

        if not duck_runs and rep_count > 0:
            duck_secs[0] = _to_float(duckdb.get("seconds", {}).get("mean"))
            duck_bytes[0] = _to_float(duckdb.get("output_bytes", {}).get("mean"))
            row_counts = duckdb.get("row_counts", [])
            if isinstance(row_counts, list) and row_counts:
                duck_rows[0] = _to_float(row_counts[0])

        if not polars_runs and rep_count > 0:
            polars_secs[0] = _to_float(polars.get("seconds", {}).get("mean"))
            polars_bytes[0] = _to_float(polars.get("output_bytes", {}).get("mean"))
            row_counts = polars.get("row_counts", [])
            if isinstance(row_counts, list) and row_counts:
                polars_rows[0] = _to_float(row_counts[0])

        def _rps(rows: list[float | None], seconds: list[float | None]) -> list[float | None]:
            values: list[float | None] = []
            for row_count, sec in zip(rows, seconds):
                if row_count is None or sec is None or sec <= 0:
                    values.append(None)
                else:
                    values.append(row_count / sec)
            return values

        all_contexts.append(
            {
                "context": "join_engine",
                "sizes": {},
                "load_benchmark": None,
                "options": [
                    {
                        "name": "duckdb_join_parquet",
                        "format": "Parquet",
                        "workers": None,
                        "seconds": duck_secs,
                        "rows_per_s": _rps(duck_rows, duck_secs),
                        "bytes_values": duck_bytes,
                        "note": "two full outer joins",
                    },
                    {
                        "name": "polars_join_parquet",
                        "format": "Parquet",
                        "workers": None,
                        "seconds": polars_secs,
                        "rows_per_s": _rps(polars_rows, polars_secs),
                        "bytes_values": polars_bytes,
                        "note": "two full outer joins",
                    },
                ],
            }
        )

    if not all_contexts:
        return []

    rep_count = 1
    for context in all_contexts:
        for option in context.get("options", []):
            rep_count = max(rep_count, len(option.get("seconds", [])))

    lines = [
        "#### Storage Comparison (Combined Direct + Matrix)",
        "| Context | Option | Format | Workers | "
        + " | ".join(_render_rep_headers("sec", rep_count))
        + " | sec mean | "
        + " | ".join(_render_rep_headers("rows/s", rep_count))
        + " | rows/s mean | Baseline | Speedup | Faster by | Throughput increase | Bytes | Notes |",
        "| :--- | :--- | :--- | ---: | "
        + " | ".join(["---:"] * rep_count)
        + " | ---: | "
        + " | ".join(["---:"] * rep_count)
        + " | ---: | :--- | ---: | ---: | ---: | ---: | :--- |",
    ]

    for context in all_contexts:
        context_name = str(context.get("context", "storage"))
        options = list(context.get("options", []))
        if not options:
            continue
        options.sort(
            key=lambda option: (0 if str(option.get("name", "")) == "intermine_batched" else 1)
        )
        baseline = _least_worker_baseline(options)
        baseline_name = str(baseline.get("name", "baseline"))
        baseline_sec = _mean(list(baseline.get("seconds", [])))
        baseline_rps = _mean(list(baseline.get("rows_per_s", [])))

        for option in options:
            sec_values = list(option.get("seconds", []))
            rps_values = list(option.get("rows_per_s", []))
            bytes_values = list(option.get("bytes_values", []))
            while len(sec_values) < rep_count:
                sec_values.append(None)
            while len(rps_values) < rep_count:
                rps_values.append(None)
            while len(bytes_values) < rep_count:
                bytes_values.append(None)
            sec_values = sec_values[:rep_count]
            rps_values = rps_values[:rep_count]
            bytes_values = bytes_values[:rep_count]

            sec_mean = _mean(sec_values)
            rps_mean = _mean(rps_values)
            bytes_mean = _mean(bytes_values)

            if option is baseline:
                speedup, faster_by, throughput = "1.00x", "+0.00%", "+0.00%"
            else:
                if sec_mean is not None and baseline_sec and sec_mean > 0:
                    speedup = _fmt_x(baseline_sec / sec_mean)
                    faster_by = _fmt_pct((baseline_sec - sec_mean) / baseline_sec * 100.0)
                else:
                    speedup, faster_by = "n/a", "n/a"
                if rps_mean is not None and baseline_rps and baseline_rps > 0:
                    throughput = _fmt_pct((rps_mean - baseline_rps) / baseline_rps * 100.0)
                else:
                    throughput = "n/a"

            lines.append(
                f"| {context_name} | {option.get('name', 'option')} | {option.get('format', 'n/a')} | {_fmt_int(option.get('workers')) if option.get('workers') is not None else '-'} | "
                + " | ".join(_fmt_seconds(value) for value in sec_values)
                + f" | {_fmt_seconds(sec_mean)} | "
                + " | ".join(_fmt_rows_per_s(value) for value in rps_values)
                + f" | {_fmt_rows_per_s(rps_mean)} | {baseline_name} | {speedup} | {faster_by} | {throughput} | {_fmt_bytes(bytes_mean)} | {option.get('note', '')} |"
            )

    lines.extend(
        [
            "",
            "| Context | CSV bytes | Parquet bytes | Saved bytes | Size reduction | CSV/Parquet ratio |",
            "| :--- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for context in all_contexts:
        sizes = context.get("sizes", {})
        if not isinstance(sizes, dict):
            continue
        if not sizes:
            continue
        lines.append(
            f"| {context.get('context', 'storage')} | {_fmt_bytes(sizes.get('csv_bytes'))} | {_fmt_bytes(sizes.get('parquet_bytes'))} | {_fmt_bytes(sizes.get('saved_bytes'))} | {_fmt_pct(sizes.get('reduction_pct'))} | {_fmt_x(sizes.get('csv_to_parquet_ratio'))} |"
        )
    lines.append("")
    return lines


def _iter_fetch_sections(query_report: dict[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, Any] | None]]:
    fetch = query_report.get("fetch_benchmark", {})
    sections: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = []
    if not isinstance(fetch, dict):
        return sections

    direct = fetch.get("direct_compare_baseline")
    if isinstance(direct, dict):
        sections.append(("direct_compare_baseline", direct, None))
    parallel = fetch.get("parallel_only_large")
    if isinstance(parallel, dict):
        sections.append(("parallel_only_large", parallel, None))

    matrix6 = fetch.get("matrix6")
    if isinstance(matrix6, list):
        for scenario in matrix6:
            if not isinstance(scenario, dict):
                continue
            result = scenario.get("result")
            if not isinstance(result, dict):
                continue
            scenario_name = str(scenario.get("name", "matrix6_scenario"))
            sections.append((f"matrix6::{scenario_name}", result, scenario.get("io_compare")))
    return sections


def append_benchmark_run_markdown(
    markdown_path: Path,
    report: dict[str, Any],
    *,
    json_report_path: str,
) -> None:
    runtime = report.get("environment", {}).get("runtime_config", {})
    timestamp = str(report.get("environment", {}).get("timestamp_utc", "unknown"))
    benchmark_target = str(runtime.get("benchmark_target", "manual"))
    mine_url = str(runtime.get("mine_url", "unknown"))
    repetitions = _fmt_int(runtime.get("repetitions"))

    lines: list[str] = []
    if not markdown_path.exists() or markdown_path.stat().st_size == 0:
        lines.extend(
            [
                "# BENCHMARKRUNS",
                "",
                "Auto-appended benchmark run history.",
                "",
            ]
        )

    lines.extend(
        [
            f"## Run {timestamp}",
            "",
            f"- Target: `{benchmark_target}`",
            f"- Mine URL: `{mine_url}`",
            f"- Repetitions: `{repetitions}`",
            f"- JSON report: `{json_report_path}`",
            "",
        ]
    )

    query_benchmarks = report.get("query_benchmarks", {})
    if not isinstance(query_benchmarks, dict):
        query_benchmarks = {}

    for query_kind in sorted(query_benchmarks.keys()):
        query_report = query_benchmarks.get(query_kind, {})
        if not isinstance(query_report, dict):
            continue
        lines.append(f"### Query Kind `{query_kind}`")
        lines.append("")

        for section_name, section, _section_io in _iter_fetch_sections(query_report):
            lines.extend(_render_fetch_combined_table(section_name, section))
        lines.extend(_render_combined_storage_table(query_report))

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with markdown_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n\n")
