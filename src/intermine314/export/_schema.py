"""Validate schemas before DuckDB bridges that could lose timestamp precision."""

from __future__ import annotations


def validate_duckdb_schema(pl, schema, api_name):
    def validate(dtype, field):
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
