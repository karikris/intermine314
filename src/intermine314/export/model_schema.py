"""Lazy model typing for bounded analytical batches, independent of row defaults."""
from __future__ import annotations

import math
from datetime import date
from decimal import Decimal


class ModelSchema(dict):
    """Fixed model types plus decimals whose scale is refined from wire values."""

    def __init__(self, fields, decimal_paths):
        super().__init__(fields)
        self.decimal_paths = tuple(decimal_paths)


def query_schema(query):
    if not query._has_model():
        return None
    from intermine314.util.deps import require_polars

    pl = require_polars("Query.to_parquet()")
    types = {
        "String": pl.String, "string": pl.String, "char": pl.String,
        "Character": pl.String, "boolean": pl.Boolean, "Boolean": pl.Boolean,
        "byte": pl.Int8, "Byte": pl.Int8, "short": pl.Int16, "Short": pl.Int16,
        "int": pl.Int32, "Integer": pl.Int32, "long": pl.Int64, "Long": pl.Int64,
        "float": pl.Float32, "Float": pl.Float32, "double": pl.Float64, "Double": pl.Float64,
        "java.util.Date": pl.Date, "Date": pl.Date, "java.math.BigDecimal": pl.Decimal(38, 0),
        "BigDecimal": pl.Decimal(38, 0),
    }
    fields, decimals = {}, []
    for view in query.views:
        path = query._model_path(view)
        if not path.is_attribute():
            raise ValueError(f"Selected path {view!r} does not represent an attribute")
        name = path.end.type_name.removeprefix("java.lang.")
        if name not in types:
            raise ValueError(f"Unsupported model type {name!r} for {view!r}")
        fields[view] = types[name]
        if name in ("java.math.BigDecimal", "BigDecimal"):
            decimals.append(view)
    return ModelSchema(fields, decimals)


def _decimal(value, name):
    if isinstance(value, bool) or not isinstance(value, (Decimal, int, str)):
        raise ValueError(f"Expected an exact decimal value for {name!r}")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError(f"Non-finite decimal in {name!r}")
    scale = max(0, -result.as_tuple().exponent)
    integer_digits = max(0, result.adjusted() + 1) if result else 0
    if scale > 38 or integer_digits + scale > 38:
        raise ValueError(f"Decimal in {name!r} exceeds precision 38")
    return result


def model_batch_frame(pl, batch, columns, schema):
    """Interpret wire values by their declared type, retaining exact decimals."""
    expected = set(columns)
    for row in batch:
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("Parquet rows must contain exactly the selected columns")
    series = []
    for name in columns:
        values = [row[name] for row in batch]
        dtype = schema[name]
        if name in schema.decimal_paths:
            values = [None if value is None else _decimal(value, name) for value in values]
            scale = max((max(0, -value.as_tuple().exponent) for value in values if value is not None), default=0)
            # Polars may replace overflowing decimals with null even when
            # strict=True. Each value must fit the shared batch scale, not
            # merely its own lexical scale.
            for value in values:
                integer_digits = max(0, value.adjusted() + 1) if value else 0
                if integer_digits + scale > 38:
                    raise ValueError(f"Decimal in {name!r} exceeds precision 38 at scale {scale}")
            dtype = pl.Decimal(38, scale)
        elif dtype == pl.Date:
            converted = []
            for value in values:
                if isinstance(value, str) and len(value) == 10:
                    value = date.fromisoformat(value)
                if value is not None and type(value) is not date:
                    raise ValueError(f"Expected a calendar date (yyyy-MM-dd) for {name!r}")
                converted.append(value)
            values = converted
        # Model Float/Double deliberately adopt their declared IEEE width.
        # All other scalar types use strict constructors (no string/int coercion).
        column = pl.Series(name, values, dtype=dtype, strict=True)
        if dtype.is_float():
            for before, after in zip(values, column):
                if before is not None and math.isfinite(before) and not math.isfinite(after):
                    raise ValueError(f"Model floating-point value overflows {dtype} in {name!r}")
        elif any(before != after for before, after in zip(values, column)):
            # Guard against both unexpected null insertion and truncation in
            # typed construction, before the original values are discarded.
            raise ValueError(f"Model schema conversion would lose precision in {name!r}")
        series.append(column)
    return pl.DataFrame(series)
