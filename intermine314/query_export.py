from __future__ import annotations

from pathlib import Path


def build_iter_batches_kwargs(
    *,
    start,
    size,
    batch_size,
    parallel,
    page_size,
    max_workers,
    ordered,
    prefetch,
    inflight_limit,
    ordered_window_pages,
    profile,
    large_query_mode,
    pagination,
    keyset_path,
    keyset_batch_size,
):
    return {
        "start": start,
        "size": size,
        "batch_size": batch_size,
        "parallel": parallel,
        "page_size": page_size,
        "max_workers": max_workers,
        "ordered": ordered,
        "prefetch": prefetch,
        "inflight_limit": inflight_limit,
        "ordered_window_pages": ordered_window_pages,
        "profile": profile,
        "large_query_mode": large_query_mode,
        "pagination": pagination,
        "keyset_path": keyset_path,
        "keyset_batch_size": keyset_batch_size,
    }


def write_single_parquet_from_parts(
    *,
    staged_dir,
    target,
    compression,
    polars_module,
    duckdb_module,
    duckdb_quote,
):
    staged_dir = Path(staged_dir)
    target = Path(target)
    part_glob = str(staged_dir / "*.parquet")
    if not any(staged_dir.glob("*.parquet")):
        polars_module.DataFrame().write_parquet(str(target), compression=compression)
        return

    try:
        scan = polars_module.scan_parquet(part_glob)
        sink = getattr(scan, "sink_parquet", None)
        if callable(sink):
            sink(str(target), compression=compression)
            return
    except Exception:
        pass

    if duckdb_module is not None:
        part_glob_sql = duckdb_quote(part_glob)
        target_sql = duckdb_quote(str(target))
        con = duckdb_module.connect(database=":memory:")
        try:
            con.execute(
                f"COPY (SELECT * FROM read_parquet({part_glob_sql})) TO {target_sql} "
                f"(FORMAT PARQUET, COMPRESSION {compression.upper()})"
            )
        finally:
            con.close()
        return

    polars_module.scan_parquet(part_glob).collect().write_parquet(str(target), compression=compression)
