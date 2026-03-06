from __future__ import annotations

from pathlib import Path

from intermine314.config.runtime_defaults import get_runtime_defaults
from intermine314.config.storage_policy import (
    default_parquet_compression,
    validate_duckdb_identifier,
    validate_parquet_compression,
)
from intermine314.export.managed import ManagedDuckDBConnection
from intermine314.export.resource_profile import resolve_temp_dir, validate_temp_dir_constraints
from intermine314.service import Service
from intermine314.service.resource_utils import close_resource_quietly as _close_resource_quietly
from intermine314.util.deps import (
    quote_sql_string as _duckdb_quote,
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
    mine_url: str,
    root_class: str,
    views: list[str],
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
):
    """
    Minimal ELT helper:
    parallel fetch -> parquet(single-file) -> duckdb view.

    Returns a dictionary with:
    - ``parquet_path``
    - ``duckdb_table``
    - ``duckdb_connection``
    """
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

    service = Service(mine_url)
    con = None
    try:
        query = service.select(root_class)
        query.clear_view()
        query.add_view(*list(views))
        query.to_parquet(
            parquet_path,
            start=start,
            size=size,
            compression=parquet_compression,
            single_file=True,
            temp_dir=resolved_temp_dir,
            temp_dir_min_free_bytes=temp_dir_min_free_bytes,
            parallel_options=parallel_options,
        )

        duckdb = _require_duckdb("fetch_from_mine()")
        con = duckdb.connect(database=duckdb_database)
        parquet_sql_path = _duckdb_quote(parquet_path)
        con.execute(
            f'CREATE OR REPLACE VIEW "{duckdb_table}" AS '
            f"SELECT * FROM read_parquet({parquet_sql_path})"
        )
        return {
            "parquet_path": parquet_path,
            "duckdb_table": duckdb_table,
            "duckdb_connection": _managed_duckdb_connection(con, managed=bool(managed)),
        }
    except Exception:
        if con is not None:
            _close_resource_quietly(con)
        raise
    finally:
        service.close()
