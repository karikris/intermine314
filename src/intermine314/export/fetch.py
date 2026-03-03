from __future__ import annotations

from pathlib import Path
import tempfile

from intermine314.config.runtime_defaults import get_runtime_defaults
from intermine314.config.storage_policy import (
    default_parquet_compression,
    validate_duckdb_identifier,
    validate_parquet_compression,
)
from intermine314.export.resource_profile import (
    ResourceProfile,
    resolve_resource_profile,
    resolve_temp_dir,
    validate_temp_dir_constraints,
)
from intermine314.query.data_plane import ManagedDuckDBConnection
from intermine314.registry.mines import resolve_production_plan
from intermine314.service import Service
from intermine314.service.resource_utils import close_resource_quietly as _close_resource_quietly
from intermine314.util.deps import require_duckdb as _require_duckdb

_WORKFLOW_ELT = "elt"


def _runtime_query_defaults():
    return get_runtime_defaults().query_defaults


def _runtime_default_parallel_page_size() -> int:
    return int(_runtime_query_defaults().default_parallel_page_size)


def _build_query(
    *,
    mine_url: str,
    root_class: str,
    views: list[str],
    joins: list[str],
):
    service = Service(mine_url)
    query = service.new_query(root_class)
    query.add_view(*views)
    for join in joins:
        query.add_join(join, "OUTER")
    return query


def _effective_parallel_args(
    *,
    mine_url: str,
    size: int | None,
    workflow: str,
    production_profile: str,
    resource_profile: ResourceProfile | str | None,
    max_workers: int | None,
    ordered,
    parallel_profile: str | None,
    large_query_mode: bool | None,
    prefetch: int | None,
    inflight_limit: int | None,
    max_inflight_bytes_estimate: int | None,
):
    plan = resolve_production_plan(
        mine_url,
        size,
        workflow=workflow,
        production_profile=production_profile,
    )
    if resource_profile is None:
        resolved_resource_profile = resolve_resource_profile(plan.get("resource_profile"))
    else:
        resolved_resource_profile = resolve_resource_profile(resource_profile)
    workers = int(
        max_workers
        if max_workers is not None
        else (
            resolved_resource_profile.max_workers
            if resolved_resource_profile.max_workers is not None
            else plan["workers"]
        )
    )
    return plan, {
        "max_workers": workers,
        "ordered": (
            ordered
            if ordered is not None
            else (
                resolved_resource_profile.ordered
                if resolved_resource_profile.ordered is not None
                else plan["ordered"]
            )
        ),
        "profile": str(parallel_profile or plan["parallel_profile"]),
        "large_query_mode": bool(plan["large_query_mode"] if large_query_mode is None else large_query_mode),
        "prefetch": (
            prefetch
            if prefetch is not None
            else (
                resolved_resource_profile.prefetch
                if resolved_resource_profile.prefetch is not None
                else None
            )
        ),
        "inflight_limit": (
            inflight_limit
            if inflight_limit is not None
            else (
                resolved_resource_profile.inflight_limit
                if resolved_resource_profile.inflight_limit is not None
                else None
            )
        ),
        "max_inflight_bytes_estimate": (
            max_inflight_bytes_estimate
            if max_inflight_bytes_estimate is not None
            else (
                resolved_resource_profile.max_inflight_bytes_estimate
                if resolved_resource_profile.max_inflight_bytes_estimate is not None
                else None
            )
        ),
        "resource_profile": resolved_resource_profile,
    }


def _build_parallel_options(
    *,
    page_size: int,
    max_workers: int,
    ordered,
    prefetch: int | None,
    inflight_limit: int | None,
    max_inflight_bytes_estimate: int | None,
    ordered_window_pages: int,
    profile: str,
    large_query_mode: bool,
):
    from intermine314.query.builder import ParallelOptions

    return ParallelOptions(
        page_size=page_size,
        max_workers=max_workers,
        ordered=ordered,
        prefetch=prefetch,
        inflight_limit=inflight_limit,
        max_inflight_bytes_estimate=max_inflight_bytes_estimate,
        ordered_window_pages=ordered_window_pages,
        profile=profile,
        large_query_mode=large_query_mode,
        pagination="auto",
    )


def _effective_temp_dir(
    *,
    explicit_temp_dir: str | Path | None,
    resolved_resource_profile: ResourceProfile,
) -> Path:
    if explicit_temp_dir is not None:
        chosen = explicit_temp_dir
    elif resolved_resource_profile.temp_dir is not None:
        chosen = resolved_resource_profile.temp_dir
    else:
        chosen = tempfile.gettempdir()
    resolved = resolve_temp_dir(chosen)
    if resolved is None:  # pragma: no cover - defensive guard
        raise ValueError("Failed to resolve temp_dir")
    return resolved


def _effective_temp_dir_min_free_bytes(
    *,
    explicit_temp_dir_min_free_bytes: int | None,
    resolved_resource_profile: ResourceProfile,
) -> int | None:
    if explicit_temp_dir_min_free_bytes is not None:
        return int(explicit_temp_dir_min_free_bytes)
    return resolved_resource_profile.temp_dir_min_free_bytes


def _managed_duckdb_connection(connection, *, managed: bool):
    if not managed:
        return connection
    return ManagedDuckDBConnection(connection, close_resource_quietly=_close_resource_quietly)


def fetch_from_mine(
    *,
    mine_url: str,
    root_class: str,
    views: list[str],
    joins: list[str] | None = None,
    start: int = 0,
    size: int | None = None,
    page_size: int | None = None,
    workflow: str = _WORKFLOW_ELT,
    production_profile: str = "auto",
    resource_profile: ResourceProfile | str | None = None,
    max_workers: int | None = None,
    ordered=None,
    parallel_profile: str | None = None,
    large_query_mode: bool | None = None,
    prefetch: int | None = None,
    inflight_limit: int | None = None,
    max_inflight_bytes_estimate: int | None = None,
    ordered_window_pages: int = 10,
    parquet_path: str | Path | None = None,
    parquet_compression: str | None = None,
    temp_dir: str | Path | None = None,
    temp_dir_min_free_bytes: int | None = None,
    duckdb_database: str = ":memory:",
    duckdb_table: str = "results",
    managed: bool = False,
):
    """
    Fetch mine data using the ELT data-plane:
    parallel fetch -> parquet -> duckdb view.

    DuckDB lifecycle:
    - ``managed=False`` (default): returns a raw live DuckDB connection.
    - ``managed=True``: returns a managed DuckDB connection wrapper in
      ``duckdb_connection``; use ``with result["duckdb_connection"] as con: ...``.
    """
    normalized_workflow = str(workflow or _WORKFLOW_ELT).strip().lower()
    if normalized_workflow != _WORKFLOW_ELT:
        raise ValueError("workflow must be 'elt'; ETL workflow is no longer supported")

    if page_size is None:
        page_size = _runtime_default_parallel_page_size()
    if parquet_path is None:
        raise ValueError("parquet_path is required for ELT workflow")

    duckdb_table = validate_duckdb_identifier(str(duckdb_table))
    parquet_compression = validate_parquet_compression(
        parquet_compression if parquet_compression is not None else default_parquet_compression()
    )

    query = _build_query(
        mine_url=mine_url,
        root_class=root_class,
        views=list(views),
        joins=list(joins or []),
    )
    plan, parallel_args = _effective_parallel_args(
        mine_url=mine_url,
        size=size,
        workflow=normalized_workflow,
        production_profile=production_profile,
        resource_profile=resource_profile,
        max_workers=max_workers,
        ordered=ordered,
        parallel_profile=parallel_profile,
        large_query_mode=large_query_mode,
        prefetch=prefetch,
        inflight_limit=inflight_limit,
        max_inflight_bytes_estimate=max_inflight_bytes_estimate,
    )
    resolved_resource_profile = parallel_args["resource_profile"]
    effective_temp_dir = _effective_temp_dir(
        explicit_temp_dir=temp_dir,
        resolved_resource_profile=resolved_resource_profile,
    )
    effective_temp_dir_min_free_bytes = _effective_temp_dir_min_free_bytes(
        explicit_temp_dir_min_free_bytes=temp_dir_min_free_bytes,
        resolved_resource_profile=resolved_resource_profile,
    )
    validate_temp_dir_constraints(
        temp_dir=effective_temp_dir,
        min_free_bytes=effective_temp_dir_min_free_bytes,
        context="fetch_from_mine parquet staging",
    )

    parallel_options = _build_parallel_options(
        page_size=page_size,
        max_workers=parallel_args["max_workers"],
        ordered=parallel_args["ordered"],
        prefetch=parallel_args["prefetch"],
        inflight_limit=parallel_args["inflight_limit"],
        max_inflight_bytes_estimate=parallel_args["max_inflight_bytes_estimate"],
        ordered_window_pages=ordered_window_pages,
        profile=parallel_args["profile"],
        large_query_mode=parallel_args["large_query_mode"],
    )
    common_args = {
        "start": start,
        "size": size,
        "parallel_options": parallel_options,
    }

    parquet_path = str(Path(parquet_path))
    query.to_parquet(
        parquet_path,
        compression=parquet_compression,
        single_file=True,
        temp_dir=effective_temp_dir,
        temp_dir_min_free_bytes=effective_temp_dir_min_free_bytes,
        **common_args,
    )

    duckdb = _require_duckdb("fetch_from_mine()")
    con = None
    try:
        con = duckdb.connect(database=duckdb_database)
        con.execute(
            f'CREATE OR REPLACE VIEW "{duckdb_table}" AS SELECT * FROM read_parquet(?)',
            [parquet_path],
        )
        managed_connection = _managed_duckdb_connection(con, managed=bool(managed))
        payload = {
            "workflow": normalized_workflow,
            "production_plan": plan,
            "resource_profile": resolved_resource_profile.name,
            "max_inflight_bytes_estimate": parallel_options.max_inflight_bytes_estimate,
            "temp_dir": str(effective_temp_dir),
            "parquet_path": parquet_path,
            "duckdb_table": duckdb_table,
            "duckdb_connection": managed_connection,
        }
        return payload
    except Exception:
        if con is not None:
            _close_resource_quietly(con)
        raise
