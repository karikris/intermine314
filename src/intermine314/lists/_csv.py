"""Optional CSV identifier input; imported only for explicit CSV uploads."""

from __future__ import annotations

from contextlib import ExitStack, closing
from glob import escape
from pathlib import Path
from tempfile import TemporaryDirectory

from intermine314.export.csv import _csv_options, _local_path, _scan_source
from intermine314.lists._identifiers import quote_identifier
from intermine314.service.resource_utils import close_resource_quietly
from intermine314.util.deps import require_duckdb, require_polars, require_pyarrow


def identifier_text(source, column, options):
    pl = require_polars("list CSV identifier input")
    duckdb = require_duckdb("list CSV identifier input")
    require_pyarrow("list CSV identifier input")
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
        scratch = stack.enter_context(TemporaryDirectory(prefix="intermine314-list-csv-"))
        parquet = Path(scratch) / "identifiers.parquet"
        scanned.select(column).sink_parquet(parquet)
        connection = stack.enter_context(closing(duckdb.connect(":memory:")))
        escaped_column = '"' + column.replace('"', '""') + '"'
        reader = connection.execute(
            f"SELECT {escaped_column} FROM read_parquet(?)", [escape(str(parquet))],
        ).to_arrow_reader(65536)
        stack.callback(close_resource_quietly, reader)
        identifiers = []
        for batch in reader:
            for value in batch.column(0).to_pylist():
                if value is None:
                    raise ValueError("CSV identifier column contains a null identifier")
                if not value:
                    raise ValueError("CSV identifier column contains an empty identifier")
                identifiers.append(quote_identifier(value))
        return "\n".join(identifiers)
