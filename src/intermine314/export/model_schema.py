"""Lazy model typing for bounded analytical batches, independent of row defaults."""
from __future__ import annotations

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
    """Compatibility helper using the canonical typed conversion entry point."""
    from intermine314.export.parquet import _batch_frame

    return _batch_frame(pl, batch, columns, schema=schema)
