from __future__ import annotations

from pathlib import Path

from intermine314.config.runtime_defaults import get_runtime_defaults
from intermine314.config.storage_policy import (
    default_parquet_compression,
    validate_duckdb_identifier,
    validate_parquet_compression,
)
from intermine314.export.managed import ManagedDuckDBConnection
from intermine314.export.query import _duckdb_source_sql
from intermine314.export.resource_profile import (
    resolve_temp_dir,
    validate_temp_dir_constraints,
)
from intermine314.service import Service
from intermine314.service.resource_utils import (
    close_resource_quietly as _close_resource_quietly,
)
from intermine314.util.deps import (
    require_duckdb as _require_duckdb,
)


def _runtime_query_defaults():
    return get_runtime_defaults().query_defaults


def _runtime_default_parallel_page_size() -> int:
    return int(_runtime_query_defaults().default_parallel_page_size)


def _build_parallel_options(
    *,
    page_size: int,
    max_workers: int | None,
    ordered,
    prefetch: int | None,
    inflight_limit: int | None,
    max_inflight_bytes_estimate: int | None,
):
    from intermine314.query.builder import ParallelOptions

    return ParallelOptions(
        page_size=page_size,
        max_workers=max_workers,
        ordered=ordered,
        prefetch=prefetch,
        inflight_limit=inflight_limit,
        max_inflight_bytes_estimate=max_inflight_bytes_estimate,
        pagination="offset",
    )


def _managed_duckdb_connection(connection, *, managed: bool):
    if not managed:
        return connection
    return ManagedDuckDBConnection(connection, close_resource_quietly=_close_resource_quietly)


def fetch_from_mine(
    *,
    mine_url: str | None = None,
    root_class: str | None = None,
    views: list[str] | None = None,
    parquet_path: str | Path,
    page_size: int | None = None,
    max_workers: int | None = None,
    ordered=None,
    prefetch: int | None = None,
    inflight_limit: int | None = None,
    max_inflight_bytes_estimate: int | None = None,
    duckdb_table: str = "results",
    managed: bool = False,
    start: int = 0,
    size: int | None = None,
    duckdb_database: str = ":memory:",
    parquet_compression: str | None = None,
    temp_dir: str | Path | None = None,
    temp_dir_min_free_bytes: int | None = None,
    csv_input=None,
    csv_options=None,
):
    """
    Minimal ELT helper:
    parallel fetch -> parquet(single-file) -> duckdb view.

    ``csv_input`` selects a local CSV scan instead of constructing a Service.
    CSV mode requires mine_url, root_class and views to be omitted. Remote mode
    requires all three. CSV headers remain intact; csv_options forwards explicit
    Polars scan options, including schema overrides. The output path persists.

    Returns a dictionary with:
    - ``parquet_path``
    - ``duckdb_table``
    - ``duckdb_connection``
    """
    if csv_input is not None:
        if any(value is not None for value in (mine_url, root_class, views)):
            raise ValueError("CSV input cannot be combined with remote mine_url, root_class or views")
    elif csv_options is not None:
        raise ValueError("csv_options requires csv_input")
    elif mine_url is None or root_class is None or views is None:
        raise ValueError("Remote mode requires mine_url, root_class and views")

    if page_size is None:
        page_size = _runtime_default_parallel_page_size()

    duckdb_table = validate_duckdb_identifier(str(duckdb_table))
    parquet_compression = validate_parquet_compression(
        parquet_compression if parquet_compression is not None else default_parquet_compression()
    )

    resolved_temp_dir = resolve_temp_dir(temp_dir)
    if temp_dir is not None and resolved_temp_dir is None:
        raise ValueError("temp_dir could not be resolved")
    if resolved_temp_dir is not None and temp_dir_min_free_bytes is not None:
        validate_temp_dir_constraints(
            temp_dir=resolved_temp_dir,
            min_free_bytes=temp_dir_min_free_bytes,
            context="fetch_from_mine parquet staging",
        )

    parquet_path = str(Path(parquet_path))
    parallel_options = _build_parallel_options(
        page_size=int(page_size),
        max_workers=max_workers,
        ordered=ordered,
        prefetch=prefetch,
        inflight_limit=inflight_limit,
        max_inflight_bytes_estimate=max_inflight_bytes_estimate,
    )

    service = None
    con = None
    try:
        if csv_input is not None:
            from intermine314.query.builder import Query

            query = Query()
        else:
            service = Service(mine_url)
            query = service.select(root_class)
            query.clear_view()
            query.add_view(*list(views))
        csv_kwargs = {"csv_input": csv_input, "csv_options": csv_options} if csv_input is not None else {}
        written_path = query.to_parquet(
            parquet_path,
            start=start,
            size=size,
            compression=parquet_compression,
            single_file=True,
            temp_dir=resolved_temp_dir,
            temp_dir_min_free_bytes=temp_dir_min_free_bytes,
            parallel_options=parallel_options,
            **csv_kwargs,
        )
        if written_path is not None:
            parquet_path = str(Path(written_path))

        duckdb = _require_duckdb("fetch_from_mine()")
        parquet_sql_path = _duckdb_source_sql(parquet_path, "fetch_from_mine()")
        con = duckdb.connect(database=duckdb_database)
        con.execute(
            f'CREATE OR REPLACE VIEW "{duckdb_table}" AS '
            f"SELECT * FROM read_parquet({parquet_sql_path})"
        )
        return {
            "parquet_path": parquet_path,
            "duckdb_table": duckdb_table,
            "duckdb_connection": _managed_duckdb_connection(con, managed=bool(managed)),
        }
    except BaseException:
        if con is not None:
            _close_resource_quietly(con)
        raise
    finally:
        if service is not None:
            service.close()
