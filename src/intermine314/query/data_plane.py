from __future__ import annotations

from itertools import chain
from pathlib import Path
import tempfile
from tempfile import TemporaryDirectory

from intermine314.export.parquet import write_single_parquet_from_parts
from intermine314.export.resource_profile import resolve_temp_dir, validate_temp_dir_constraints


class ManagedDuckDBConnection:
    """Context manager wrapper for DuckDB connections returned by query helpers."""

    def __init__(self, connection, *, close_resource_quietly):
        self._connection = connection
        self._close_resource_quietly = close_resource_quietly
        self._closed = False

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._close_resource_quietly(self._connection)

    def __enter__(self):
        return self._connection

    def __exit__(self, exc_type, exc, tb):
        _ = (exc_type, exc, tb)
        self.close()
        return False


def _resolve_staging_temp_dir(*, temp_dir, temp_dir_min_free_bytes, context):
    if temp_dir is None and temp_dir_min_free_bytes is None:
        return None
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    resolved = resolve_temp_dir(temp_dir)
    if resolved is None:  # pragma: no cover - defensive guard
        return None
    if temp_dir_min_free_bytes is not None:
        validate_temp_dir_constraints(
            temp_dir=resolved,
            min_free_bytes=temp_dir_min_free_bytes,
            context=context,
        )
    return resolved


def _polars_from_dicts_with_full_inference(polars_module, batch):
    """Use full-batch schema inference when supported to avoid type drift within a batch."""
    try:
        return polars_module.from_dicts(batch, infer_schema_length=None)
    except TypeError as exc:
        detail = str(exc)
        if "infer_schema_length" not in detail or "keyword" not in detail:
            raise
        return polars_module.from_dicts(batch)


def _write_single_parquet_from_parts(
    *,
    staged_dir,
    target,
    compression,
    require_polars,
    optional_duckdb,
    duckdb_quote,
):
    write_single_parquet_from_parts(
        staged_dir=staged_dir,
        target=target,
        compression=compression,
        polars_module=require_polars("Query.to_parquet()"),
        duckdb_module=optional_duckdb(),
        duckdb_quote=duckdb_quote,
    )


def query_to_dataframe(
    query,
    *,
    start=0,
    size=None,
    batch_size=None,
    final_rechunk=False,
    parallel_options=None,
    runtime_default_batch_size,
    require_polars,
):
    polars_module = require_polars("Query.dataframe()")
    if batch_size is None:
        batch_size = runtime_default_batch_size()
    options = query._coerce_parallel_options(parallel_options=parallel_options)
    iter_kwargs = query._iter_batches_kwargs(
        start=start,
        size=size,
        batch_size=batch_size,
        row_mode="dict",
        parallel_options=options,
    )
    frame_iter = (polars_module.from_dicts(batch) for batch in query.iter_batches(**iter_kwargs))
    try:
        first = next(frame_iter)
    except StopIteration:
        return polars_module.DataFrame()
    return polars_module.concat(
        chain((first,), frame_iter),
        how="diagonal_relaxed",
        rechunk=bool(final_rechunk),
    )


def query_to_parquet(
    query,
    path,
    *,
    start=0,
    size=None,
    batch_size=None,
    compression=None,
    single_file=False,
    temp_dir=None,
    temp_dir_min_free_bytes=None,
    parallel_options=None,
    runtime_default_export_batch_size,
    default_parquet_compression,
    validate_parquet_compression,
    require_polars,
    optional_duckdb,
    duckdb_quote,
):
    polars_module = require_polars("Query.to_parquet()")
    if batch_size is None:
        batch_size = runtime_default_export_batch_size()
    options = query._coerce_parallel_options(parallel_options=parallel_options)
    compression = validate_parquet_compression(
        default_parquet_compression() if compression is None else compression
    )
    target = Path(path)
    if single_file:
        staging_dir = _resolve_staging_temp_dir(
            temp_dir=temp_dir,
            temp_dir_min_free_bytes=temp_dir_min_free_bytes,
            context="Query.to_parquet(single_file=True) staging",
        )
        temp_kwargs = {}
        if staging_dir is not None:
            temp_kwargs["dir"] = str(staging_dir)
        with TemporaryDirectory(prefix="intermine314-parquet-", **temp_kwargs) as tmp:
            staged_dir = Path(tmp) / "parts"
            query_to_parquet(
                query,
                staged_dir,
                start=start,
                size=size,
                batch_size=batch_size,
                compression=compression,
                single_file=False,
                parallel_options=options,
                temp_dir=temp_dir,
                temp_dir_min_free_bytes=temp_dir_min_free_bytes,
                runtime_default_export_batch_size=runtime_default_export_batch_size,
                default_parquet_compression=default_parquet_compression,
                validate_parquet_compression=validate_parquet_compression,
                require_polars=require_polars,
                optional_duckdb=optional_duckdb,
                duckdb_quote=duckdb_quote,
            )
            _write_single_parquet_from_parts(
                staged_dir=staged_dir,
                target=target,
                compression=compression,
                require_polars=require_polars,
                optional_duckdb=optional_duckdb,
                duckdb_quote=duckdb_quote,
            )
        return str(target)
    if target.exists() and target.is_file():
        raise ValueError("path must be a directory when single_file is False")
    target.mkdir(parents=True, exist_ok=True)
    for stale in target.glob("part-*.parquet"):
        stale.unlink()
    part = 0
    iter_kwargs = query._iter_batches_kwargs(
        start=start,
        size=size,
        batch_size=batch_size,
        row_mode="dict",
        parallel_options=options,
    )
    for batch in query.iter_batches(**iter_kwargs):
        frame = _polars_from_dicts_with_full_inference(polars_module, batch)
        part_path = target / "part-{0:05d}.parquet".format(part)
        frame.write_parquet(str(part_path), compression=compression)
        part += 1
    return str(target)


def query_to_duckdb(
    query,
    path,
    *,
    start=0,
    size=None,
    batch_size=None,
    compression=None,
    single_file=False,
    database=":memory:",
    table="results",
    temp_dir=None,
    temp_dir_min_free_bytes=None,
    parallel_options=None,
    managed=False,
    runtime_default_export_batch_size,
    validate_duckdb_identifier,
    require_duckdb,
    duckdb_quote,
    close_resource_quietly,
):
    duckdb_module = require_duckdb("Query.to_duckdb()")
    if batch_size is None:
        batch_size = runtime_default_export_batch_size()
    options = query._coerce_parallel_options(parallel_options=parallel_options)
    table = validate_duckdb_identifier(table)
    parquet_path = query.to_parquet(
        path,
        start=start,
        size=size,
        batch_size=batch_size,
        compression=compression,
        single_file=single_file,
        temp_dir=temp_dir,
        temp_dir_min_free_bytes=temp_dir_min_free_bytes,
        parallel_options=options,
    )
    target = Path(parquet_path)
    if target.is_dir():
        parquet_glob = str(target / "*.parquet")
    else:
        parquet_glob = str(target)
    parquet_glob_sql = duckdb_quote(parquet_glob)
    con = duckdb_module.connect(database=database)
    try:
        con.execute(f'CREATE OR REPLACE VIEW "{table}" AS SELECT * FROM read_parquet({parquet_glob_sql})')
    except Exception:
        close_resource_quietly(con)
        raise
    if managed:
        return ManagedDuckDBConnection(con, close_resource_quietly=close_resource_quietly)
    return con


def query_duckdb_view(query, *args, **kwargs):
    kwargs["managed"] = True
    return query.to_duckdb(*args, **kwargs)
