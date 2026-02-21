from __future__ import annotations

import re
from pathlib import Path
from tempfile import TemporaryDirectory

from intermine314.config.constants import (
    DEFAULT_PARALLEL_PAGE_SIZE,
    DEFAULT_PRODUCTION_PROFILE_SWITCH_ROWS,
    PRODUCTION_WORKFLOW_ELT,
    PRODUCTION_WORKFLOW_ETL,
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
    max_workers: int | None,
    ordered,
    parallel_profile: str | None,
    large_query_mode: bool | None,
    prefetch: int | None,
    inflight_limit: int | None,
):
    plan = resolve_production_plan(
        mine_url,
        size,
        workflow=workflow,
        production_profile=production_profile,
    )
    workers = int(max_workers if max_workers is not None else plan["workers"])
    return plan, {
        "max_workers": workers,
        "ordered": plan["ordered"] if ordered is None else ordered,
        "profile": str(parallel_profile or plan["parallel_profile"]),
        "large_query_mode": bool(plan["large_query_mode"] if large_query_mode is None else large_query_mode),
        "prefetch": prefetch if prefetch is not None else plan.get("prefetch"),
        "inflight_limit": inflight_limit if inflight_limit is not None else plan.get("inflight_limit"),
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
        ordered_window_pages=ordered_window_pages,
        profile=profile,
        large_query_mode=large_query_mode,
        pagination="auto",
    )


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
    max_workers: int | None = None,
    ordered=None,
    parallel_profile: str | None = None,
    large_query_mode: bool | None = None,
    prefetch: int | None = None,
    inflight_limit: int | None = None,
    ordered_window_pages: int = 10,
    parquet_path: str | Path | None = None,
    parquet_compression: str = "zstd",
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
        max_workers=max_workers,
        ordered=ordered,
        parallel_profile=parallel_profile,
        large_query_mode=large_query_mode,
        prefetch=prefetch,
        inflight_limit=inflight_limit,
    )

    parallel_options = _build_parallel_options(
        page_size=page_size,
        max_workers=parallel_args["max_workers"],
        ordered=parallel_args["ordered"],
        prefetch=parallel_args["prefetch"],
        inflight_limit=parallel_args["inflight_limit"],
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
            **common_args,
        )
        con.execute(
            f'CREATE OR REPLACE VIEW "{duckdb_table}" AS SELECT * FROM read_parquet(?)',
            [parquet_path],
        )
        return {
            "workflow": workflow,
            "production_plan": plan,
            "parquet_path": parquet_path,
            "duckdb_table": duckdb_table,
            "duckdb_connection": con,
        }

    result = {
        "workflow": workflow,
        "production_plan": plan,
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

    with TemporaryDirectory(prefix=_ETL_TEMP_DIR_PREFIX) as tmp:
        etl_parquet_target = Path(tmp) / "parts"
        query.to_parquet(
            etl_parquet_target,
            compression=parquet_compression,
            single_file=False,
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
