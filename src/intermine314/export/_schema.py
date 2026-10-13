"""Validate schemas before DuckDB bridges that could lose timestamp precision."""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from pathlib import Path

_CACHE_LOCK = threading.Lock()
_SCHEMA_CACHE = OrderedDict()
_SCHEMAS = weakref.WeakValueDictionary()
_CACHE_LIMIT = 4096


class _SharedSchema:
    def __init__(self, schema):
        self.schema = dict(schema)


def _fingerprint(path):
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def read_parquet_schema_literal(pl, path):
    """Cache validated metadata by file identity; manifests are never trusted."""
    path = Path(path).resolve()
    before = _fingerprint(path)
    with _CACHE_LOCK:
        item = _SCHEMA_CACHE.get(path)
        if item is not None and item[0] == before:
            _SCHEMA_CACHE.move_to_end(path)
            return dict(item[1].schema)
    schema = pl.scan_parquet(path, glob=False).collect_schema()
    if _fingerprint(path) != before:
        raise OSError('Parquet file changed during schema inspection')
    signature = tuple(schema.items())
    with _CACHE_LOCK:
        shared = _SCHEMAS.get(signature)
        if shared is None:
            shared = _SCHEMAS[signature] = _SharedSchema(schema)
        _SCHEMA_CACHE[path] = before, shared
        _SCHEMA_CACHE.move_to_end(path)
        while len(_SCHEMA_CACHE) > _CACHE_LIMIT:
            _SCHEMA_CACHE.popitem(last=False)
    return dict(schema)


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
