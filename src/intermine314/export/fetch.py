from __future__ import annotations

import re
from pathlib import Path
import tempfile
from tempfile import TemporaryDirectory

from intermine314.config.constants import (
    DEFAULT_PARALLEL_PAGE_SIZE,
    DEFAULT_PRODUCTION_PROFILE_SWITCH_ROWS,
    PRODUCTION_WORKFLOW_ELT,
    PRODUCTION_WORKFLOW_ETL,
)
from intermine314.export.resource_profile import (
    ResourceProfile,
    resolve_resource_profile,
    resolve_temp_dir,
    validate_temp_dir_constraints,
)
from intermine314.registry.mines import resolve_production_plan
from intermine314.util.deps import (
    require_duckdb as _require_duckdb,
    require_polars as _require_polars,
)
from intermine314.service import Service

_DUCKDB_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ETL_TEMP_DIR_PREFIX = "intermine314-etl-"


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
                else plan.get("prefetch")
            )
        ),
        "inflight_limit": (
            inflight_limit
            if inflight_limit is not None
            else (
                resolved_resource_profile.inflight_limit
                if resolved_resource_profile.inflight_limit is not None
                else plan.get("inflight_limit")
            )
        ),
        "max_inflight_bytes_estimate": (
            max_inflight_bytes_estimate
            if max_inflight_bytes_estimate is not None
            else (
                resolved_resource_profile.max_inflight_bytes_estimate
                if resolved_resource_profile.max_inflight_bytes_estimate is not None
                else plan.get("max_inflight_bytes_estimate")
            )
        ),
        "resource_profile": resolved_resource_profile,
    }


def _should_block_etl_scan(*, workflow: str, size: int | None, etl_guardrail_rows: int, etl_override: bool) -> bool:
    if workflow != PRODUCTION_WORKFLOW_ETL or etl_override:
        return False
    if size is None:
        return True
    return int(size) > int(etl_guardrail_rows)


def _parquet_source_for_duckdb(path: Path, *, single_file: bool) -> str:
    return str(path) if single_file else str(path / "*.parquet")


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


def fetch_from_mine(
    *,
    mine_url: str,
    root_class: str,
    views: list[str],
    joins: list[str] | None = None,
    start: int = 0,
    size: int | None = None,
    page_size: int = DEFAULT_PARALLEL_PAGE_SIZE,
    workflow: str = PRODUCTION_WORKFLOW_ELT,
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
    parquet_compression: str = "zstd",
    temp_dir: str | Path | None = None,
    temp_dir_min_free_bytes: int | None = None,
    duckdb_database: str = ":memory:",
    duckdb_table: str = "results",
    etl_guardrail_rows: int = DEFAULT_PRODUCTION_PROFILE_SWITCH_ROWS,
    allow_large_etl: bool = False,
    force_etl: bool | None = None,
    etl_materialize_dataframe: bool = False,
    etl_final_rechunk: bool = False,
):
    """
    Fetch mine data with a mine-aware production profile.

    Workflows:
    - ELT: parallel fetch -> parquet -> duckdb view
    - ETL: parallel fetch -> parquet -> duckdb table (dataframe is opt-in)

    Resource controls:
    - ``resource_profile`` applies bounded defaults (workers/prefetch/inflight/temp dir).
    - ``max_inflight_bytes_estimate`` caps estimated queued payload bytes in parallel mode.
    - ``temp_dir`` and ``temp_dir_min_free_bytes`` control/validate staging disk usage.
    """
    workflow = str(workflow or PRODUCTION_WORKFLOW_ELT).strip().lower()
    if workflow not in {PRODUCTION_WORKFLOW_ELT, PRODUCTION_WORKFLOW_ETL}:
        raise ValueError("workflow must be 'elt' or 'etl'")
    if not _DUCKDB_IDENTIFIER_PATTERN.fullmatch(str(duckdb_table)):
        raise ValueError("duckdb_table must be a valid SQL identifier")
    if int(etl_guardrail_rows) <= 0:
        raise ValueError("etl_guardrail_rows must be a positive integer")
    etl_override = bool(allow_large_etl) or bool(force_etl)
    if _should_block_etl_scan(
        workflow=workflow,
        size=size,
        etl_guardrail_rows=int(etl_guardrail_rows),
        etl_override=etl_override,
    ):
        raise ValueError(
            "ETL workflow is disabled for unknown or large scans by default to avoid high memory pressure. "
            "Use workflow='elt' with parquet_path, or pass allow_large_etl=True / force_etl=True to force ETL."
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
        workflow=workflow,
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
        "parallel": True,
        "parallel_options": parallel_options,
    }

    duckdb = _require_duckdb("fetch_from_mine()")
    con = duckdb.connect(database=duckdb_database)

    if workflow == PRODUCTION_WORKFLOW_ELT:
        if parquet_path is None:
            raise ValueError("parquet_path is required for ELT workflow")
        parquet_path = str(Path(parquet_path))
        query.to_parquet(
            parquet_path,
            compression=parquet_compression,
            single_file=True,
            temp_dir=effective_temp_dir,
            temp_dir_min_free_bytes=effective_temp_dir_min_free_bytes,
            **common_args,
        )
        con.execute(
            f'CREATE OR REPLACE VIEW "{duckdb_table}" AS SELECT * FROM read_parquet(?)',
            [parquet_path],
        )
        return {
            "workflow": workflow,
            "production_plan": plan,
            "resource_profile": resolved_resource_profile.name,
            "max_inflight_bytes_estimate": parallel_options.max_inflight_bytes_estimate,
            "temp_dir": str(effective_temp_dir),
            "parquet_path": parquet_path,
            "duckdb_table": duckdb_table,
            "duckdb_connection": con,
        }

    result = {
        "workflow": workflow,
        "production_plan": plan,
        "resource_profile": resolved_resource_profile.name,
        "max_inflight_bytes_estimate": parallel_options.max_inflight_bytes_estimate,
        "temp_dir": str(effective_temp_dir),
        "dataframe": None,
        "duckdb_table": duckdb_table,
        "duckdb_connection": con,
    }
    if parquet_path is not None:
        etl_parquet_target = Path(parquet_path)
        etl_single_file = etl_parquet_target.suffix.lower() == ".parquet"
        query.to_parquet(
            etl_parquet_target,
            compression=parquet_compression,
            single_file=etl_single_file,
            temp_dir=effective_temp_dir,
            temp_dir_min_free_bytes=effective_temp_dir_min_free_bytes,
            **common_args,
        )
        parquet_source = _parquet_source_for_duckdb(etl_parquet_target, single_file=etl_single_file)
        con.execute(
            f'CREATE OR REPLACE TABLE "{duckdb_table}" AS SELECT * FROM read_parquet(?)',
            [parquet_source],
        )
        result["parquet_path"] = str(etl_parquet_target)
        if bool(etl_materialize_dataframe):
            polars_module = _require_polars("fetch_from_mine(etl_materialize_dataframe=True)")
            result["dataframe"] = polars_module.scan_parquet(parquet_source).collect(
                rechunk=bool(etl_final_rechunk)
            )
        return result

    temp_dir_stats = validate_temp_dir_constraints(
        temp_dir=effective_temp_dir,
        min_free_bytes=effective_temp_dir_min_free_bytes,
        context="fetch_from_mine temporary parquet staging",
    )
    result["temp_dir_free_bytes"] = int(temp_dir_stats["temp_dir_free_bytes"])
    result["temp_dir_total_bytes"] = int(temp_dir_stats["temp_dir_total_bytes"])

    with TemporaryDirectory(prefix=_ETL_TEMP_DIR_PREFIX, dir=str(effective_temp_dir)) as tmp:
        etl_parquet_target = Path(tmp) / "parts"
        query.to_parquet(
            etl_parquet_target,
            compression=parquet_compression,
            single_file=False,
            temp_dir=effective_temp_dir,
            temp_dir_min_free_bytes=effective_temp_dir_min_free_bytes,
            **common_args,
        )
        parquet_source = _parquet_source_for_duckdb(etl_parquet_target, single_file=False)
        con.execute(
            f'CREATE OR REPLACE TABLE "{duckdb_table}" AS SELECT * FROM read_parquet(?)',
            [parquet_source],
        )
        if bool(etl_materialize_dataframe):
            polars_module = _require_polars("fetch_from_mine(etl_materialize_dataframe=True)")
            result["dataframe"] = polars_module.scan_parquet(parquet_source).collect(
                rechunk=bool(etl_final_rechunk)
            )
    return result
