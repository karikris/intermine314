"""Optional CSV identifier input; imported only for explicit CSV uploads."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from intermine314.export.csv import _csv_options, _local_path, _scan_source
from intermine314.export.query import query_parquet_reader
from intermine314.export.runtime import get_runtime
from intermine314.lists._identifiers import quote_identifier
from intermine314.util.deps import require_polars


@contextmanager
def identifier_body(source, column, options):
    """Validate to Parquet, then yield a known-length, owned binary upload file.

    Validation finishes before yielding, so no mutation can precede a late
    invalid identifier. The body is closed after the single upload attempt.
    """
    pl = require_polars("list CSV identifier input")
    options = _csv_options(pl, options)
    # Never infer an identifier as numeric, even for an all-numeric sample.
    options["infer_schema"] = False
    for key in ("schema", "schema_overrides"):
        schema = options.get(key)
        if schema is not None:
            if not isinstance(schema, dict):
                raise ValueError(f"CSV identifier {key} must be a mapping")
            options[key] = dict(schema, **{column: pl.String})
    options.setdefault("schema_overrides", {column: pl.String})
    if isinstance(source, (str, Path)):
        source = _local_path(source, "CSV input")
    elif not callable(getattr(source, "read", None)):
        raise TypeError("csv_input must be a local path or readable stream")
    with ExitStack() as stack:
        scanned = pl.scan_csv(_scan_source(source, stack), **options)
        schema = scanned.collect_schema()
        if column not in schema:
            raise ValueError("CSV identifier column is absent")
        if schema[column] != pl.String:
            raise ValueError("CSV identifier column must have String schema")
        runtime = get_runtime()
        scratch = stack.enter_context(TemporaryDirectory(prefix="intermine314-list-csv-", dir=runtime.profile.temp_dir))
        parquet = Path(scratch) / "identifiers.parquet"
        with runtime.native('polars'):
            scanned.select(column).sink_parquet(parquet)
        body = stack.enter_context((Path(scratch) / 'upload.txt').open('w+b'))
        first = True
        with query_parquet_reader(parquet, runtime=runtime) as reader:
            for batch in reader:
                for value in batch.column(0).to_pylist():
                    if value is None:
                        raise ValueError("CSV identifier column contains a null identifier")
                    if not value:
                        raise ValueError("CSV identifier column contains an empty identifier")
                    quoted = quote_identifier(value).encode('utf-8')
                    if not first:
                        body.write(b'\n')
                    body.write(quoted)
                    first = False
        body.seek(0)
        yield b'' if first else body
