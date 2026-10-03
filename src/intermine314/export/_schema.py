"""Validate schemas before DuckDB bridges that could lose timestamp precision."""

from __future__ import annotations


def read_parquet_schema_literal(pl, path):
    """Inspect only metadata, without glob expansion or Python file buffering."""
    return pl.scan_parquet(path, glob=False).collect_schema()


def validate_duckdb_schema(pl, schema, api_name):
    def validate(dtype, field):
        if dtype == pl.Int128:
            raise ValueError(
                f"{api_name} cannot read Int128 in {field!r}: store it as "
                "Decimal(38, 0) because DuckDB reads the Int128 annotation as binary"
            )
        if (
            isinstance(dtype, pl.Datetime)
            and dtype.time_unit == "ns"
            and dtype.time_zone is not None
        ):
            raise ValueError(
                f"{api_name} cannot read nanosecond timestamps with a timezone "
                f"in {field!r}: DuckDB would lose precision"
            )
        if isinstance(dtype, (pl.List, pl.Array)):
            validate(dtype.inner, f"{field}[]")
        elif isinstance(dtype, pl.Struct):
            for name, inner in dtype.to_schema().items():
                validate(inner, f"{field}.{name}")

    for name, dtype in schema.items():
        validate(dtype, name)
