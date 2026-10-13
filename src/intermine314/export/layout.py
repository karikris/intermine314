"""Parquet geometry independent of producer pagination and batch size."""

from dataclasses import dataclass

from intermine314.parallel.policy import require_positive_int


@dataclass(frozen=True)
class ParquetOptions:
    row_group_size: int = 122880
    target_file_bytes: int = 256 * 1024 * 1024
    row_group_bytes: int | None = None
    preserve_order: bool = True
    compact: bool = True

    def __post_init__(self):
        for name in ('row_group_size', 'target_file_bytes'):
            require_positive_int(name, getattr(self, name))
        if self.row_group_size < 2048:
            raise ValueError('row_group_size must be at least 2048 for DuckDB compaction')
        if self.row_group_bytes is not None:
            require_positive_int('row_group_bytes', self.row_group_bytes)
        for name in ('preserve_order', 'compact'):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f'{name} must be a boolean')

    def effective_row_group(self, *, bytes_per_row, memory_budget):
        budget = min(self.row_group_bytes or 64 * 1024 * 1024, max(1, memory_budget // 8))
        estimate = int(budget / max(1, bytes_per_row))
        return max(2048, min(self.row_group_size, estimate)), budget


def resolve_parquet_options(value):
    if value is None:
        return ParquetOptions()
    if not isinstance(value, ParquetOptions):
        raise TypeError('parquet_options must be a ParquetOptions instance')
    return value


def can_compact_directory(pl, schema):
    """Only normalize directory layouts whose logical types survive DuckDB.

    Direct parts retain Null, categorical, fixed-array, duration/time and
    timestamp metadata. Single-file COPY keeps its existing documented bridge.
    """
    def supported(dtype):
        if dtype in (pl.String, pl.Binary, pl.Boolean, pl.Date):
            return True
        if dtype.is_integer() or dtype.is_float() or dtype.is_decimal():
            return dtype != pl.Int128
        if isinstance(dtype, pl.List):
            return supported(dtype.inner)
        if isinstance(dtype, pl.Struct):
            return all(supported(inner) for inner in dtype.to_schema().values())
        return False
    return all(supported(dtype) for dtype in schema.values())
