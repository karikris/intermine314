from __future__ import annotations

import re
from pathlib import Path

from intermine314.constants import DEFAULT_PARALLEL_PAGE_SIZE, PRODUCTION_WORKFLOW_ELT, PRODUCTION_WORKFLOW_ETL
from intermine314.mine_registry import resolve_production_plan
from intermine314.optional_deps import require_duckdb as _require_duckdb
from intermine314.webservice import Service

_DUCKDB_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
):
    """
    Fetch mine data with a mine-aware production profile.

    Workflows:
    - ELT: parallel fetch -> parquet -> duckdb view
    - ETL: parallel fetch -> polars dataframe -> duckdb table
    """
    workflow = str(workflow or PRODUCTION_WORKFLOW_ELT).strip().lower()
    if workflow not in {PRODUCTION_WORKFLOW_ELT, PRODUCTION_WORKFLOW_ETL}:
        raise ValueError("workflow must be 'elt' or 'etl'")
    if not _DUCKDB_IDENTIFIER_PATTERN.fullmatch(str(duckdb_table)):
        raise ValueError("duckdb_table must be a valid SQL identifier")

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

    common_args = {
        "start": start,
        "size": size,
        "parallel": True,
        "page_size": page_size,
        "max_workers": parallel_args["max_workers"],
        "ordered": parallel_args["ordered"],
        "prefetch": parallel_args["prefetch"],
        "inflight_limit": parallel_args["inflight_limit"],
        "ordered_window_pages": ordered_window_pages,
        "profile": parallel_args["profile"],
        "large_query_mode": parallel_args["large_query_mode"],
        "pagination": "auto",
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

    dataframe = query.dataframe(**common_args)
    con.register("_intermine314_polars_df", dataframe)
    con.execute(f'CREATE OR REPLACE TABLE "{duckdb_table}" AS SELECT * FROM _intermine314_polars_df')
    try:
        con.unregister("_intermine314_polars_df")
    except Exception:
        pass
    return {
        "workflow": workflow,
        "production_plan": plan,
        "dataframe": dataframe,
        "duckdb_table": duckdb_table,
        "duckdb_connection": con,
    }
