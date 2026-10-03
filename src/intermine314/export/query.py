"""Execute SQL against existing Parquet data through DuckDB and Arrow."""

from __future__ import annotations

import re
from contextlib import closing
from glob import escape
from pathlib import Path

from intermine314.export._schema import (
    read_parquet_schema_literal,
    validate_duckdb_schema,
)
from intermine314.util.deps import (
    quote_sql_string,
    require_duckdb,
    require_polars,
    require_pyarrow,
)


def _parquet_sources(path):
    path = Path(path)
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Parquet path does not exist: {path}")
    parts = [
        entry
        for entry in path.iterdir()
        if entry.is_file() and re.fullmatch(r"part-[0-9]+\.parquet", entry.name)
    ]
    if parts:
        files = sorted(parts, key=lambda entry: (int(entry.stem[5:]), entry.name))
    else:
        files = sorted(entry for entry in path.rglob("*.parquet") if entry.is_file())
    if not files:
        raise FileNotFoundError(f"No Parquet files found in directory: {path}")
    return files


def _duckdb_source_sql(path, api_name):
    """Literal, metadata-guarded paths for persistent native DuckDB views."""
    sources = _parquet_sources(path)
    files = [sources] if isinstance(sources, Path) else sources
    pl = require_polars(api_name)
    for file in files:
        validate_duckdb_schema(pl, read_parquet_schema_literal(pl, file), api_name)
    literals = [quote_sql_string(escape(str(file))) for file in files]
    return literals[0] if isinstance(sources, Path) else "[" + ",".join(literals) + "]"


def query_parquet(
    path, sql="SELECT * FROM results", *, parameters=None, database=":memory:"
):
    """Return a Polars DataFrame from SQL over a Parquet file or directory.

    ``results`` is a connection-local DuckDB view of the input, which leaves
    existing database tables and views intact. Directories containing numbered
    ``part-*.parquet`` files read only those files, in numeric part order. Other
    directories read regular Parquet files recursively. Use an explicit SQL
    ``ORDER BY`` when the result requires an ordering beyond that input order.

    ``parameters`` accepts DuckDB positional sequences or named mappings. The
    result is materialized through Arrow before the owned connection closes,
    including on errors and interrupts. Analytics dependencies load on call.

    Timezone-aware timestamps normalize to UTC with microsecond precision.
    Nanosecond timestamps with timezones, including nested fields, are rejected
    before DuckDB reads them because that conversion would lose precision.
    Naive nanosecond timestamps retain nanosecond precision.
    External Int128 annotations are rejected because DuckDB reads them as binary;
    store those numbers as Decimal(38, 0) for exact numeric interoperability.
    """
    sources = _parquet_sources(path)
    pl = require_polars("query_parquet()")
    duckdb = require_duckdb("query_parquet()")
    require_pyarrow("query_parquet()")
    files = [sources] if isinstance(sources, Path) else sources
    for file in files:
        schema = read_parquet_schema_literal(pl, file)
        validate_duckdb_schema(pl, schema, "query_parquet()")
    # DuckDB treats path strings as globs, even when the literal file exists.
    escaped = [escape(str(file)) for file in files]
    sources = escaped[0] if isinstance(sources, Path) else escaped
    with closing(duckdb.connect(database=database)) as connection:
        connection.execute("SET TimeZone = 'UTC'")
        connection.register("results", connection.read_parquet(sources))
        table = connection.execute(sql, parameters).to_arrow_table()
        return pl.from_arrow(table)
