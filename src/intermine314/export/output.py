"""Explicit CSV output through a managed, bounded Parquet/DuckDB pipeline."""

from pathlib import Path
from tempfile import TemporaryDirectory

from intermine314.export.parquet import _connection, _publish_file
from intermine314.export.query import _duckdb_source_sql
from intermine314.util.deps import quote_sql_string


def write_csv_from_parquet(path, target, *, scratch):
    """COPY into same-filesystem staging and publish with shared rollback."""
    sources = _duckdb_source_sql(path, "Query.export(format='csv')")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{target.name}-publish-", dir=target.parent
    ) as publication:
        publication = Path(publication)
        staged = publication / "output"
        with _connection(scratch) as connection:
            connection.execute("SET TimeZone = 'UTC'")
            connection.execute(
                f"COPY (SELECT * FROM read_parquet({sources})) "
                f"TO {quote_sql_string(staged)} "
                "(FORMAT CSV, HEADER true, COMPRESSION none)"
            )
        _publish_file(staged, target, publication / "backup")
    return str(target)
