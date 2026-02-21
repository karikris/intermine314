from __future__ import annotations

from pathlib import Path


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

    raise RuntimeError(
        "Cannot merge parquet parts without sink_parquet support or duckdb. "
        "Install duckdb or keep single_file=False."
    )
