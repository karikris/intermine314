"""Execute SQL against existing Parquet data through DuckDB and Arrow."""

from __future__ import annotations

import re
from contextlib import ExitStack, closing, contextmanager
from glob import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock

from intermine314.export._schema import (
    read_parquet_schema_literal,
    validate_duckdb_schema,
)
from intermine314.export.manifest import is_metadata
from intermine314.export.runtime import get_runtime, resource_operation
from intermine314.parallel.policy import require_positive_int
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
        files = sorted(entry for entry in path.rglob("*.parquet") if entry.is_file() and not is_metadata(entry.relative_to(path)))
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


@contextmanager
def _query_connection(path, database, *, borrowed=None):
    """Share validation and ownership without changing a borrowed database."""
    sources = _parquet_sources(path)
    pl = require_polars("query_parquet()")
    duckdb = require_duckdb("query_parquet()")
    require_pyarrow("query_parquet()")
    files = [sources] if isinstance(sources, Path) else sources
    for file in files:
        validate_duckdb_schema(pl, read_parquet_schema_literal(pl, file), "query_parquet()")
    escaped = [escape(str(file)) for file in files]
    sources = escaped[0] if isinstance(sources, Path) else escaped
    active = get_runtime()
    with ExitStack() as stack:
        if borrowed is None:
            scratch = stack.enter_context(TemporaryDirectory(prefix="intermine314-sql-", dir=active.profile.temp_dir))
            config = active.duckdb_config(temp_dir=scratch)
            connection = stack.enter_context(closing(duckdb.connect(database=database, config=config)))
            threads = config['threads']
        else:
            connection = stack.enter_context(closing(borrowed.cursor()))
            threads = int(connection.execute("SELECT current_setting('threads')").fetchone()[0])
        connection.register("results", connection.read_parquet(sources))
        yield connection, threads


class ParquetReader:
    """Context-owned Arrow batches. Close explicitly when stopping early.

    Batches already returned retain their data after close. A single reader is
    serialized; independent readers may run concurrently under native admission.
    """

    def __init__(self, reader, stack, runtime, threads):
        self.schema = reader.schema
        self._reader = reader
        self._stack = stack
        self._runtime = runtime
        self._threads = threads
        self._lock = RLock()
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            if self.closed:
                raise StopIteration
            try:
                with self._runtime.native('duckdb_sql', threads=self._threads):
                    return next(self._reader)
            except BaseException:
                self.close()
                raise

    read_next_batch = __next__

    def read_all(self):
        """Explicitly materialize all remaining batches as an Arrow Table."""
        batches = list(self)
        with self._runtime.native('arrow_cpu'):
            return require_pyarrow('query_parquet_reader()').Table.from_batches(batches, schema=self.schema)

    def close(self):
        with self._lock:
            if not self.closed:
                self.closed = True
                try:
                    self._stack.close()
                finally:
                    self._runtime.disown(self)
                    self._reader = None

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Parquet reader is closed')
        return self

    def __exit__(self, *exc):
        self.close()


@resource_operation
def query_parquet_reader(
    path, sql="SELECT * FROM results", *, parameters=None, database=":memory:",
    connection=None, batch_size=65536, resource_profile=None, runtime=None,
):
    """Return a context-owned iterator of bounded Arrow RecordBatches.

    Uses the same schema guards and local ``results`` view as query_parquet.
    Owned connections use profile threads/memory and private spill directories.
    A supplied connection is borrowed through a private cursor; its global
    settings and lifetime remain caller-controlled. Do not also set database.
    Close on early exit, or let the owning ResourceRuntime close idle readers.
    SQL operators may need more memory than one output batch and can spill.
    """
    require_positive_int('batch_size', batch_size)
    if connection is not None and database != ':memory:':
        raise ValueError('database cannot be combined with connection')
    active = get_runtime()
    stack = ExitStack()
    try:
        owner, threads = stack.enter_context(_query_connection(path, database, borrowed=connection))
        with active.native('duckdb_sql', threads=threads):
            reader = owner.execute(sql, parameters).to_arrow_reader(batch_size)
        stack.callback(reader.close)
        result = ParquetReader(reader, stack, active, threads)
        return active.own(result)
    except BaseException:
        stack.close()
        raise


@resource_operation
def query_parquet(
    path, sql="SELECT * FROM results", *, parameters=None, database=":memory:",
    resource_profile=None, runtime=None,
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
    pl = require_polars("query_parquet()")
    with _query_connection(path, database) as (connection, threads):
        active = get_runtime()
        with active.native('duckdb_sql', threads=threads):
            table = connection.execute(sql, parameters).to_arrow_table()
        with active.native('polars'):
            return pl.from_arrow(table)
