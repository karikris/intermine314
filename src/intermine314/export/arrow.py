"""Typed flat batches; retain the established fallback for other semantics.

Arrow's safe flag alone does not reject Python float-to-integer truncation.
Input types are checked before construction, and declared IEEE model widths
are handled explicitly. No Python round-trip of the constructed batch is needed.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from decimal import Decimal

from intermine314.export.model_schema import ModelSchema, _decimal
from intermine314.export.runtime import get_runtime
from intermine314.util.deps import require_pyarrow


def _decimal_array(pa, values, name):
    values = [None if value is None else _decimal(value, name) for value in values]
    scale = max((max(0, -value.as_tuple().exponent) for value in values if value is not None), default=0)
    try:
        return pa.array(values, type=pa.decimal128(38, scale), safe=True)
    except pa.ArrowInvalid as exc:
        raise ValueError(f'Decimal in {name!r} exceeds precision 38 at scale {scale}') from exc


def _inferred_array(pa, values, kinds, name):
    if not kinds:
        dtype = pa.null()
    elif kinds == {str}:
        dtype = pa.large_string()
    elif kinds == {bytes}:
        dtype = pa.large_binary()
    elif kinds == {bool}:
        dtype = pa.bool_()
    elif kinds == {int}:
        low = min(value for value in values if value is not None)
        high = max(value for value in values if value is not None)
        if -(2**63) <= low and high < 2**63:
            dtype = pa.int64()
        elif low >= 0 and high < 2**64:
            dtype = pa.uint64()
        else:
            return _decimal_array(pa, values, name)
    elif kinds == {float}:
        dtype = pa.float64()
    elif kinds == {Decimal}:
        return _decimal_array(pa, values, name)
    elif kinds == {date}:
        dtype = pa.date32()
    elif kinds == {datetime} and all(value is None or value.tzinfo is None for value in values):
        dtype = pa.timestamp('us')
    else:
        # Mixed numeric order, nested structures, custom scalar subclasses and
        # timezone interpretation retain their existing strict Polars semantics.
        return None
    return pa.array(values, type=dtype, safe=True)


def _model_array(pa, pl, values, kinds, name, schema, types):
    dtype = schema[name]
    if name in schema.decimal_paths:
        return _decimal_array(pa, values, name)
    if dtype == pl.Date:
        values = [date.fromisoformat(value) if isinstance(value, str) and len(value) == 10 else value
                  for value in values]
        if any(value is not None and type(value) is not date for value in values):
            raise ValueError(f'Expected a calendar date (yyyy-MM-dd) for {name!r}')
        return pa.array(values, type=pa.date32(), safe=True)
    if dtype in types:
        python_type, arrow_type = types[dtype]
        if not kinds <= {python_type}:
            return None
        return pa.array(values, type=arrow_type, safe=True)
    if dtype in (pl.Float32, pl.Float64) and kinds <= {int, float}:
        if dtype == pl.Float32 and any(type(value) is int and abs(value) > 2**53 for value in values):
            # Avoid a second rounding step (large int -> double -> float32).
            return None
        # Declared model Float/Double intentionally use their IEEE width. Arrow
        # rejects large integer inputs before rounding, so adopt float explicitly.
        values = [None if value is None else float(value) for value in values]
        array = pa.array(values, type=pa.float32() if dtype == pl.Float32 else pa.float64(), safe=True)
        if dtype == pl.Float32:
            import pyarrow.compute as pc

            finite_before = sum(value is not None and math.isfinite(value) for value in values)
            if (pc.sum(pc.is_finite(array)).as_py() or 0) < finite_before:
                raise ValueError(f'Model floating-point value overflows {dtype} in {name!r}')
        return array
    return None


def _compatibility_model_array(pl, values, name, dtype):
    """Retain strict Polars coercion and direct large-int Float32 rounding.

    Only atypical scalar inputs or custom model-schema extensions reach here;
    declared wire primitives, dates and decimals have one Arrow implementation.
    """
    with get_runtime().native('polars'):
        column = pl.Series(name, values, dtype=dtype, strict=True)
        if dtype.is_float():
            for before, after in zip(values, column):
                if before is not None and math.isfinite(before) and not math.isfinite(after):
                    raise ValueError(f'Model floating-point value overflows {dtype} in {name!r}')
        elif any(before != after for before, after in zip(values, column)):
            raise ValueError(f'Model schema conversion would lose precision in {name!r}')
        return column.to_arrow()


class _ConversionPlan:
    """Execution-local column/type metadata; no cache keyed by mutable schemas."""

    def __init__(self, pl, columns, schema=None):
        self.pl = pl
        self.columns = tuple(columns)
        self.expected = frozenset(columns)
        self.schema = ModelSchema(schema, schema.decimal_paths) if isinstance(schema, ModelSchema) else schema
        self.pa = pa = require_pyarrow('Typed Parquet conversion')
        self.types = {
            pl.String: (str, pa.large_string()), pl.Boolean: (bool, pa.bool_()),
            pl.Int8: (int, pa.int8()), pl.Int16: (int, pa.int16()),
            pl.Int32: (int, pa.int32()), pl.Int64: (int, pa.int64()),
        }

    def convert(self, batch):
        from intermine314.query._columnar import ColumnBatch

        if self.schema is not None and not isinstance(self.schema, ModelSchema):
            return None
        if isinstance(batch, ColumnBatch):
            if batch.columns != self.columns:
                raise ValueError('Parquet batch columns must match the conversion plan')
            values_by_column = batch.column_values()
        else:
            for row in batch:
                if not isinstance(row, dict) or row.keys() != self.expected:
                    raise ValueError('Parquet rows must contain exactly the selected columns')
            values_by_column = ([row[name] for row in batch] for name in self.columns)
        pa, pl, schema = self.pa, self.pl, self.schema
        runtime = get_runtime()
        arrays, fallback = [], []
        with runtime.native('arrow_cpu'):
            for name, values in zip(self.columns, values_by_column):
                kinds = {type(value) for value in values if value is not None}
                array = (_model_array(pa, pl, values, kinds, name, schema, self.types) if schema is not None
                         else _inferred_array(pa, values, kinds, name))
                if array is None:
                    if schema is None:
                        return None
                    fallback.append((len(arrays), name, values))
                arrays.append(array)
            if not fallback:
                return pa.RecordBatch.from_arrays(arrays, names=self.columns)
        # A distinct Polars compatibility stage must not nest Arrow admission.
        for index, name, values in fallback:
            arrays[index] = _compatibility_model_array(pl, values, name, schema[name])
        with runtime.native('arrow_cpu'):
            return pa.RecordBatch.from_arrays(arrays, names=self.columns)


def typed_batch(pl, batch, columns, schema=None):
    """Return a RecordBatch, or None when distinct fallback semantics apply."""
    if schema is not None and not isinstance(schema, ModelSchema):
        return None
    return _ConversionPlan(pl, columns, schema).convert(batch)
