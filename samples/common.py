"""Shared constants/helpers for sample scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from intermine314.config.runtime_defaults import get_runtime_defaults
from intermine314.export import query_parquet
from intermine314.query.builder import ParallelOptions

_QUERY_DEFAULTS = get_runtime_defaults().query_defaults


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    text = os.getenv(name, "").strip()
    if not text:
        return default
    try:
        value = int(text)
    except ValueError:
        return default
    return max(minimum, value)


DEFAULT_SERVICE_ROOT = os.getenv(
    "INTERMINE314_SAMPLE_SERVICE_URL",
    "https://www.flymine.org/query/service",
).strip()
SAMPLES_OUTPUT_ROOT = Path("samples/output")
DEFAULT_RESULT_SIZE = _env_int("INTERMINE314_SAMPLE_RESULT_SIZE", 2_000)
DEFAULT_BATCH_SIZE = min(_QUERY_DEFAULTS.default_export_batch_size, DEFAULT_RESULT_SIZE)
DEFAULT_PREVIEW_LIMIT = _env_int("INTERMINE314_SAMPLE_PREVIEW_LIMIT", 20)
DEFAULT_SAMPLE_PAGE_SIZE = _env_int("INTERMINE314_SAMPLE_PAGE_SIZE", _QUERY_DEFAULTS.default_parallel_page_size)

# Let intermine314 choose mine-aware worker defaults unless explicitly overridden.
PARALLEL_DEFAULTS: dict[str, Any] = {
    "page_size": DEFAULT_SAMPLE_PAGE_SIZE,
    "profile": "default",
    "ordered": "ordered",
    "pagination": "auto",
}


def parallel_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs = dict(PARALLEL_DEFAULTS)
    kwargs.update(overrides)
    return kwargs


def sample_output_dir(name: str) -> Path:
    path = SAMPLES_OUTPUT_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _duckdb():
    import duckdb

    return duckdb


def preview_rows(query: Any, *, limit: int = DEFAULT_PREVIEW_LIMIT, **parallel_overrides: Any) -> None:
    for row in query.run_parallel(
        row="dict",
        start=0,
        size=max(1, int(limit)),
        parallel_options=ParallelOptions(**parallel_kwargs(**parallel_overrides)),
    ):
        print(row)


def _parquet_glob(parquet_path: Path) -> str:
    if parquet_path.is_dir():
        return str(parquet_path / "*.parquet")
    return str(parquet_path)


def open_duckdb_view_from_parquet(
    parquet_path: Path,
    *,
    table_name: str,
    database: str = ":memory:",
):
    if not table_name.isidentifier():
        raise ValueError("table_name must be a valid SQL identifier")
    parquet_glob = _parquet_glob(parquet_path).replace("'", "''")
    con = _duckdb().connect(database=database)
    try:
        con.execute(
            f'CREATE OR REPLACE VIEW "{table_name}" AS SELECT * FROM read_parquet(\'{parquet_glob}\')'
        )
    except BaseException:
        con.close()
        raise
    return con


def parquet_head(parquet_path: Path, *, limit: int = 5):
    return query_parquet(parquet_path, "SELECT * FROM results LIMIT ?", parameters=[int(limit)])


def export_parquet_and_open_duckdb(
    query: Any,
    *,
    output_dir: Path,
    parquet_name: str,
    table_name: str,
    result_size: int = DEFAULT_RESULT_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    **parallel_overrides: Any,
) -> tuple[Path, Any]:
    """
    Run the standard sample pipeline:
    query -> Parquet -> DuckDB connection.
    """
    batch_size = max(1, min(int(batch_size), int(result_size)))
    options = ParallelOptions(**parallel_kwargs(**parallel_overrides))
    parquet_path = Path(
        query.to_parquet(
            str(output_dir / parquet_name),
            size=result_size,
            batch_size=batch_size,
            parallel_options=options,
        )
    )
    return parquet_path, open_duckdb_view_from_parquet(parquet_path, table_name=table_name)
