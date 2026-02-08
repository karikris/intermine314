from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from scripts.bench_utils import ensure_parent, stat_summary


def _import_or_raise(module_name: str, requirement_msg: str):
    try:
        return __import__(module_name)
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(requirement_msg) from exc


def _fetch_exports():
    from scripts.bench_fetch import mode_label_for_workers, run_mode_export_csv

    return mode_label_for_workers, run_mode_export_csv


def _read_csv_robust(pd_module, csv_path: Path, selected_columns: list[str] | None = None):
    attempts = [
        {"usecols": selected_columns or None},
        {"usecols": selected_columns or None, "engine": "python", "on_bad_lines": "skip"},
        {"engine": "python", "on_bad_lines": "skip"},
    ]
    last_exc: Exception | None = None
    for kwargs in attempts:
        try:
            return pd_module.read_csv(csv_path, **kwargs)
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to load CSV")


def infer_dataframe_columns(
    csv_path: Path,
    root_class: str,
    views: list[str],
) -> dict[str, Any]:
    headers: list[str] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            headers = next(reader, [])

    header_set = set(headers)
    cds_candidates = [v for v in views if ".CDS." in v and v.endswith(".primaryIdentifier")]
    cds_candidates.extend(
        [
            f"{root_class}.transcripts.CDS.primaryIdentifier",
            "Gene.transcripts.CDS.primaryIdentifier",
            "Protein.transcripts.primaryIdentifier",
        ]
    )
    length_candidates = [
        f"{root_class}.proteins.length",
        f"{root_class}.length",
        "Gene.proteins.length",
        "Protein.length",
    ]
    group_candidates = [
        f"{root_class}.primaryIdentifier",
        "Gene.primaryIdentifier",
        "Protein.primaryIdentifier",
        "Protein.genes.primaryIdentifier",
        f"{root_class}.id",
    ]

    def first_present(candidates: list[str]) -> str | None:
        for candidate in candidates:
            if candidate in header_set:
                return candidate
        return None

    return {
        "headers": headers,
        "cds_column": first_present(cds_candidates),
        "length_column": first_present(length_candidates),
        "group_column": first_present(group_candidates),
    }


def csv_to_parquet(csv_path: Path, parquet_path: Path, log_label: str) -> float:
    ensure_parent(parquet_path)
    if parquet_path.exists():
        parquet_path.unlink()

    t0 = time.perf_counter()
    pl = _import_or_raise("polars", "polars is required for parquet export/benchmark")
    # Use full-file schema inference to handle mixed identifier types
    # (for example numeric + prefixed IDs in the same column).
    pl.scan_csv(str(csv_path), infer_schema_length=None).sink_parquet(
        str(parquet_path), compression="zstd"
    )
    parquet_seconds = time.perf_counter() - t0
    print(f"convert_done {log_label} seconds={parquet_seconds:.3f} path={parquet_path}", flush=True)
    return parquet_seconds


def csv_parquet_size_stats(csv_path: Path, parquet_path: Path) -> dict[str, Any]:
    csv_size = csv_path.stat().st_size
    parquet_size = parquet_path.stat().st_size
    saved = csv_size - parquet_size
    reduction_pct = (saved / csv_size * 100.0) if csv_size else 0.0
    ratio = (csv_size / parquet_size) if parquet_size else None
    return {
        "csv_bytes": csv_size,
        "parquet_bytes": parquet_size,
        "saved_bytes": saved,
        "reduction_pct": reduction_pct,
        "csv_to_parquet_ratio": ratio,
    }


def export_intermine314_csv_and_parquet(
    *,
    mine_url: str,
    rows_target: int,
    page_size: int,
    workers: int | None,
    mode_runtime_kwargs: dict[str, Any],
    csv_out_path: Path,
    parquet_out_path: Path,
    csv_log_suffix: str,
    parquet_log_label: str,
    query_root_class: str,
    query_views: list[str],
    query_joins: list[str],
) -> dict[str, Any]:
    mode_label_for_workers, run_mode_export_csv = _fetch_exports()
    mode_label = mode_label_for_workers(workers)
    csv_export = run_mode_export_csv(
        log_mode=f"{mode_label}_{csv_log_suffix}",
        mode=mode_label,
        mine_url=mine_url,
        rows_target=rows_target,
        page_size=page_size,
        workers=workers,
        csv_out_path=csv_out_path,
        mode_runtime_kwargs=mode_runtime_kwargs,
        query_root_class=query_root_class,
        query_views=query_views,
        query_joins=query_joins,
    )
    parquet_seconds = csv_to_parquet(
        csv_path=csv_out_path,
        parquet_path=parquet_out_path,
        log_label=parquet_log_label,
    )
    return {
        "csv_path": str(csv_out_path),
        "parquet_path": str(parquet_out_path),
        "csv_seconds": csv_export.seconds,
        "csv_rows_per_s": csv_export.rows_per_s,
        "csv_retries": csv_export.retries,
        "csv_rows": csv_export.rows,
        "csv_effective_workers": csv_export.effective_workers,
        "conversion_seconds_from_csv": parquet_seconds,
    }


def bench_csv_vs_parquet_load(
    *,
    csv_path: Path,
    parquet_path: Path,
    repetitions: int,
) -> dict[str, Any]:
    if repetitions <= 0:
        repetitions = 1
    pd = _import_or_raise("pandas", "pandas is required for CSV load benchmarks")
    pl = _import_or_raise("polars", "polars is required for parquet load benchmarks")

    csv_load_times: list[float] = []
    parquet_load_times: list[float] = []
    csv_row_counts: list[int] = []
    parquet_row_counts: list[int] = []

    for rep in range(1, repetitions + 1):
        t0 = time.perf_counter()
        csv_df = _read_csv_robust(pd, csv_path)
        csv_t = time.perf_counter() - t0
        csv_load_times.append(csv_t)
        csv_row_counts.append(int(csv_df.shape[0]))
        del csv_df

        t0 = time.perf_counter()
        parquet_df = pl.read_parquet(parquet_path)
        parquet_t = time.perf_counter() - t0
        parquet_load_times.append(parquet_t)
        parquet_row_counts.append(int(parquet_df.height))
        del parquet_df

        print(
            f"matrix_load_rep rep={rep} csv_load_s={csv_t:.3f} parquet_load_s={parquet_t:.3f}",
            flush=True,
        )

    return {
        "repetitions": repetitions,
        "csv_load_seconds_pandas": stat_summary(csv_load_times),
        "parquet_load_seconds_polars": stat_summary(parquet_load_times),
        "csv_row_counts": csv_row_counts,
        "parquet_row_counts": parquet_row_counts,
    }


def export_matrix_storage_compare(
    *,
    mine_url: str,
    scenario_name: str,
    rows_target: int,
    page_size: int,
    workers: list[int],
    include_legacy_baseline: bool,
    mode_runtime_kwargs: dict[str, Any],
    output_dir: Path,
    load_repetitions: int,
    query_root_class: str,
    query_views: list[str],
    query_joins: list[str],
) -> dict[str, Any]:
    mode_label_for_workers, run_mode_export_csv = _fetch_exports()
    unique_workers = sorted({int(worker) for worker in workers if int(worker) > 0})
    if not unique_workers:
        raise ValueError("matrix storage comparison requires at least one worker")

    lowest_worker = unique_workers[0]
    highest_worker = unique_workers[-1]
    csv_mode = "intermine_batched" if include_legacy_baseline else mode_label_for_workers(lowest_worker)
    csv_workers = None if include_legacy_baseline else lowest_worker
    parquet_mode = mode_label_for_workers(highest_worker)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{scenario_name}.csv"
    parquet_source_csv = output_dir / f"{scenario_name}.parquet_source.csv"
    parquet_path = output_dir / f"{scenario_name}.parquet"

    csv_export = run_mode_export_csv(
        log_mode=f"{scenario_name}_{csv_mode}_matrix_csv",
        mode=csv_mode,
        mine_url=mine_url,
        rows_target=rows_target,
        page_size=page_size,
        workers=csv_workers,
        csv_out_path=csv_path,
        mode_runtime_kwargs=mode_runtime_kwargs,
        query_root_class=query_root_class,
        query_views=query_views,
        query_joins=query_joins,
    )
    parquet_export = export_intermine314_csv_and_parquet(
        mine_url=mine_url,
        rows_target=rows_target,
        page_size=page_size,
        workers=highest_worker,
        mode_runtime_kwargs=mode_runtime_kwargs,
        csv_out_path=parquet_source_csv,
        parquet_out_path=parquet_path,
        csv_log_suffix="matrix_parquet_source",
        parquet_log_label=f"{scenario_name}_csv_to_parquet",
        query_root_class=query_root_class,
        query_views=query_views,
        query_joins=query_joins,
    )
    load_stats = bench_csv_vs_parquet_load(
        csv_path=csv_path,
        parquet_path=parquet_path,
        repetitions=load_repetitions,
    )
    return {
        "rows_target": rows_target,
        "page_size": page_size,
        "csv_source_mode": csv_mode,
        "csv_source_workers": csv_workers,
        "parquet_source_mode": parquet_mode,
        "parquet_source_workers": highest_worker,
        "csv_export": {
            "path": str(csv_path),
            "seconds": csv_export.seconds,
            "rows_per_s": csv_export.rows_per_s,
            "retries": csv_export.retries,
            "rows": csv_export.rows,
        },
        "parquet_export": {
            "csv_path": parquet_export["csv_path"],
            "parquet_path": parquet_export["parquet_path"],
            "source_csv_seconds": parquet_export["csv_seconds"],
            "source_csv_rows_per_s": parquet_export["csv_rows_per_s"],
            "source_csv_retries": parquet_export["csv_retries"],
            "source_csv_rows": parquet_export["csv_rows"],
            "source_csv_effective_workers": parquet_export["csv_effective_workers"],
            "conversion_seconds_from_csv": parquet_export["conversion_seconds_from_csv"],
        },
        "sizes": csv_parquet_size_stats(csv_path, parquet_path),
        "load_benchmark": load_stats,
    }


def export_for_storage(
    *,
    mine_url: str,
    rows_target: int,
    page_size: int,
    workers_for_new: int | None,
    mode_runtime_kwargs: dict[str, Any],
    csv_old_path: Path,
    parquet_new_path: Path,
    query_root_class: str,
    query_views: list[str],
    query_joins: list[str],
) -> dict[str, Any]:
    _, run_mode_export_csv = _fetch_exports()
    old_export = run_mode_export_csv(
        log_mode="intermine_batched_csv",
        mode="intermine_batched",
        mine_url=mine_url,
        rows_target=rows_target,
        page_size=page_size,
        workers=None,
        csv_out_path=csv_old_path,
        mode_runtime_kwargs=mode_runtime_kwargs,
        query_root_class=query_root_class,
        query_views=query_views,
        query_joins=query_joins,
    )

    tmp_new_csv = parquet_new_path.with_suffix(parquet_new_path.suffix + ".tmp.csv")
    new_export = export_intermine314_csv_and_parquet(
        mine_url=mine_url,
        rows_target=rows_target,
        page_size=page_size,
        mode_runtime_kwargs=mode_runtime_kwargs,
        workers=workers_for_new,
        csv_out_path=tmp_new_csv,
        parquet_out_path=parquet_new_path,
        csv_log_suffix="tmpcsv",
        parquet_log_label="csv_to_parquet",
        query_root_class=query_root_class,
        query_views=query_views,
        query_joins=query_joins,
    )

    return {
        "old_export_csv": {
            "path": str(csv_old_path),
            "seconds": old_export.seconds,
            "rows_per_s": old_export.rows_per_s,
            "retries": old_export.retries,
        },
        "new_export_tmp_csv": {
            "path": new_export["csv_path"],
            "seconds": new_export["csv_seconds"],
            "rows_per_s": new_export["csv_rows_per_s"],
            "retries": new_export["csv_retries"],
        },
        "new_parquet": {
            "path": new_export["parquet_path"],
            "conversion_seconds_from_tmp_csv": new_export["conversion_seconds_from_csv"],
        },
        "sizes": csv_parquet_size_stats(csv_old_path, parquet_new_path),
    }


def export_new_only_for_dataframe(
    *,
    mine_url: str,
    rows_target: int,
    page_size: int,
    workers_for_new: int | None,
    mode_runtime_kwargs: dict[str, Any],
    csv_new_path: Path,
    parquet_new_path: Path,
    query_root_class: str,
    query_views: list[str],
    query_joins: list[str],
) -> dict[str, Any]:
    new_export = export_intermine314_csv_and_parquet(
        mine_url=mine_url,
        rows_target=rows_target,
        page_size=page_size,
        mode_runtime_kwargs=mode_runtime_kwargs,
        workers=workers_for_new,
        csv_out_path=csv_new_path,
        parquet_out_path=parquet_new_path,
        csv_log_suffix="large_csv",
        parquet_log_label="intermine314_large_csv_to_parquet",
        query_root_class=query_root_class,
        query_views=query_views,
        query_joins=query_joins,
    )

    return {
        "new_export_csv": {
            "path": new_export["csv_path"],
            "seconds": new_export["csv_seconds"],
            "rows_per_s": new_export["csv_rows_per_s"],
            "retries": new_export["csv_retries"],
        },
        "new_export_parquet": {
            "path": new_export["parquet_path"],
            "conversion_seconds_from_csv": new_export["conversion_seconds_from_csv"],
        },
        "sizes": csv_parquet_size_stats(csv_new_path, parquet_new_path),
    }


def analytics_columns(
    cds_column: str | None,
    length_column: str | None,
    group_column: str | None,
) -> list[str]:
    columns: list[str] = []
    for value in (cds_column, length_column, group_column):
        if value and value not in columns:
            columns.append(value)
    return columns


def bench_pandas(
    csv_path: Path,
    repetitions: int,
    cds_column: str | None,
    length_column: str | None,
    group_column: str | None,
) -> dict[str, Any]:
    pd = _import_or_raise("pandas", "pandas is required for CSV dataframe benchmark")
    selected_columns = analytics_columns(cds_column, length_column, group_column)
    load_times: list[float] = []
    suite_times: list[float] = []
    row_counts: list[int] = []
    cds_non_null_counts: list[int] = []
    protein_means: list[float] = []
    top1_counts: list[int] = []
    memory_bytes: list[int] = []

    for rep in range(1, repetitions + 1):
        t0 = time.perf_counter()
        df = _read_csv_robust(pd, csv_path, selected_columns)
        load_t = time.perf_counter() - t0
        load_times.append(load_t)
        row_counts.append(int(df.shape[0]))
        memory_bytes.append(int(df.memory_usage(deep=True).sum()))

        t0 = time.perf_counter()
        prot_len = None
        cds_non_null = int(df[cds_column].notna().sum()) if cds_column and cds_column in df.columns else 0
        if length_column and length_column in df.columns:
            prot_len = pd.to_numeric(df[length_column], errors="coerce")
            prot_mean = float(prot_len.mean()) if prot_len.notna().any() else float("nan")
        else:
            prot_mean = float("nan")
        if group_column and group_column in df.columns:
            top10 = (
                df.groupby(group_column, dropna=False)
                .size()
                .sort_values(ascending=False)
                .head(10)
            )
        else:
            top10 = pd.Series(dtype="int64")
        suite_t = time.perf_counter() - t0
        suite_times.append(suite_t)
        cds_non_null_counts.append(cds_non_null)
        protein_means.append(prot_mean)
        top1_counts.append(int(top10.iloc[0]) if len(top10) else 0)
        del df
        del top10
        if prot_len is not None:
            del prot_len
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


def bench_polars(
    parquet_path: Path,
    repetitions: int,
    cds_column: str | None,
    length_column: str | None,
    group_column: str | None,
) -> dict[str, Any]:
    pl = _import_or_raise("polars", "polars is required for parquet export/benchmark")
    selected_columns = analytics_columns(cds_column, length_column, group_column)
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
        df = pl.read_parquet(parquet_path, columns=selected_columns or None)
        load_t = time.perf_counter() - t0
        load_times.append(load_t)
        row_counts.append(int(df.height))
        memory_bytes.append(int(df.estimated_size()))

        t0 = time.perf_counter()
        cds_non_null = int(df.select(pl.col(cds_column).is_not_null().sum()).item()) if cds_column and cds_column in df.columns else 0
        if length_column and length_column in df.columns:
            prot_mean = float(df.select(pl.col(length_column).cast(pl.Float64, strict=False).mean()).item())
        else:
            prot_mean = float("nan")
        if group_column and group_column in df.columns:
            top10 = (
                df.group_by(group_column)
                .len()
                .sort("len", descending=True)
                .head(10)
            )
        else:
            top10 = pl.DataFrame({"len": []})
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
        del df
        del top10

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
