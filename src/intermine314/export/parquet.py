"""Bounded Parquet staging and exception-safe publication.

Only one producer batch or staged part is materialized at a time. Schema promotion
uses empty frames; a second pass rewrites individual parts when necessary.
"""

from __future__ import annotations

import ctypes
import errno
import math
import os
import shutil
import sys
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from intermine314.config.storage_policy import validate_parquet_compression
from intermine314.service.resource_utils import close_resource_quietly
from intermine314.util.deps import quote_sql_string, require_duckdb, require_pyarrow


def _common_dtype(pl, left, right):
    if left == right or right == pl.Null:
        return left
    if left == pl.Null:
        return right
    if left.is_integer() and right.is_integer():
        # Int64 + UInt64 must not take the Float64 supertype and lose precision.
        return pl.Decimal(38, 0)
    if left.is_decimal() or right.is_decimal():
        if not (left.is_decimal() or left.is_integer()) or not (
            right.is_decimal() or right.is_integer()
        ):
            raise ValueError(f"Incompatible Parquet types: {left} and {right}")
        return pl.Decimal(
            38, max(getattr(left, "scale", 0), getattr(right, "scale", 0))
        )
    if isinstance(left, pl.List) and isinstance(right, pl.List):
        return pl.List(_common_dtype(pl, left.inner, right.inner))
    if isinstance(left, pl.Struct) and isinstance(right, pl.Struct):
        a, b = dict(left.to_schema()), dict(right.to_schema())
        if a.keys() != b.keys():
            raise ValueError("Inconsistent nested Parquet fields")
        return pl.Struct(
            {name: _common_dtype(pl, dtype, b[name]) for name, dtype in a.items()}
        )
    if (left.is_numeric() and right.is_numeric()) or (
        isinstance(left, pl.Datetime)
        and isinstance(right, pl.Datetime)
        and left.time_zone == right.time_zone
    ):
        return pl.concat(
            [pl.DataFrame(schema={"x": left}), pl.DataFrame(schema={"x": right})],
            how="vertical_relaxed",
        ).schema["x"]
    raise ValueError(f"Incompatible Parquet types: {left} and {right}")


def _cast_losslessly(frame, schema):
    cast = frame.cast(schema, strict=True)
    # Strict casts alone permit float rounding and decimal scale truncation.
    for name, original in frame.schema.items():
        if original != schema[name] and not frame[name].equals(
            cast[name].cast(original, strict=True)
        ):
            raise ValueError(
                f"Parquet schema conversion would lose precision in {name!r}"
            )
    return cast


def _same_value(before, after):
    if isinstance(before, dict) and isinstance(after, dict):
        return before.keys() == after.keys() and all(
            _same_value(value, after[key]) for key, value in before.items()
        )
    if isinstance(before, (list, tuple)) and isinstance(after, (list, tuple)):
        return len(before) == len(after) and all(
            _same_value(a, b) for a, b in zip(before, after)
        )
    if (
        isinstance(before, float)
        and isinstance(after, float)
        and math.isnan(before)
        and math.isnan(after)
    ):
        return True
    return before == after


def _parquet_dtype(pl, dtype):
    """Normalize private Int128 annotations, including nested schema hooks."""
    if dtype == pl.Int128:
        return pl.Decimal(38, 0)
    if isinstance(dtype, pl.List):
        return pl.List(_parquet_dtype(pl, dtype.inner))
    if isinstance(dtype, pl.Array):
        return pl.Array(_parquet_dtype(pl, dtype.inner), dtype.size)
    if isinstance(dtype, pl.Struct):
        return pl.Struct(
            {
                name: _parquet_dtype(pl, inner)
                for name, inner in dtype.to_schema().items()
            }
        )
    return dtype


def _batch_frame(pl, batch, columns):
    expected = set(columns)
    for row in batch:
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("Parquet rows must contain exactly the selected columns")
    # Series construction with strict=True rejects mixed Python types instead of
    # silently coercing large integers to floats or strings during inference.
    frame = pl.DataFrame(
        [pl.Series(name, [row[name] for row in batch], strict=True) for name in columns]
    )
    # Strict inference can still round an integer if a float appears first.
    # Check the bounded Python batch too, including nested fields.
    for before, after in zip(batch, frame.iter_rows(named=True)):
        if not _same_value(before, after):
            raise ValueError("Parquet inference would lose fields or precision")
    # Polars' private Int128 Parquet annotation is read as binary by DuckDB.
    return _cast_losslessly(
        frame,
        {name: _parquet_dtype(pl, dtype) for name, dtype in frame.schema.items()},
    )


def _connection(scratch):
    duckdb = require_duckdb("Query.to_parquet()")
    return closing(
        duckdb.connect(
            database=":memory:",
            config={
                "threads": 1,
                "memory_limit": "128MB",
                "preserve_insertion_order": True,
                "temp_directory": str(Path(scratch) / "spill"),
            },
        )
    )


def _read_part(pl, part, scratch):
    require_pyarrow("Query.to_parquet()")
    with _connection(scratch) as con:
        return pl.from_arrow(
            con.execute("SELECT * FROM read_parquet(?)", [str(part)]).to_arrow_table()
        )


def write_single_parquet_from_parts(
    *, staged_dir, target, compression, scratch, batch_size=10000
):
    """Merge ordered parts through a memory-limited, spillable DuckDB COPY."""
    paths = sorted(
        Path(staged_dir).glob("part-*.parquet"), key=lambda p: int(p.stem[5:])
    )
    sources = "[" + ",".join(quote_sql_string(path) for path in paths) + "]"
    with _connection(scratch) as con:
        con.execute(
            f"COPY (SELECT * FROM read_parquet({sources})) TO {quote_sql_string(target)} "
            f"(FORMAT PARQUET, COMPRESSION {compression.upper()}, ROW_GROUP_SIZE {batch_size})"
        )


def _exchange_directories(left, right):
    """Use a single atomic Linux rename when the filesystem supports it."""
    if sys.platform != "linux":
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    exchange = getattr(libc, "renameat2", None)
    if exchange is None:
        return False
    exchange.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    exchange.restype = ctypes.c_int
    if exchange(-100, os.fsencode(left), -100, os.fsencode(right), 2) == 0:
        return True
    code = ctypes.get_errno()
    if code in (errno.ENOSYS, errno.EINVAL, errno.ENOTSUP):
        return False
    raise OSError(code, os.strerror(code), str(right))


def _publish_file(staged, target, backup):
    if target.exists():
        # Keep the previous inode alive across the replace, including a signal
        # delivered immediately after the operating system completes the call.
        os.link(target, backup)
    try:
        os.replace(staged, target)
    except BaseException:
        if backup.exists():
            os.replace(backup, target)
        elif not staged.exists() and target.exists():
            os.replace(target, staged)
        raise


def _publish_directory(staged, target, backup):
    existed = target.exists()
    old_inode = target.stat().st_ino if existed else None
    try:
        if not existed:
            os.replace(staged, target)
        elif not _exchange_directories(staged, target):
            # Portable fallback: readers may briefly see a missing directory,
            # never a mixture of old/new parts.
            os.replace(target, backup)
            os.replace(staged, target)
    except BaseException:
        if backup.exists():
            if target.exists():
                os.replace(target, staged)
            os.replace(backup, target)
        elif existed and staged.exists() and staged.stat().st_ino == old_inode:
            # The exchange completed just before an interrupt was delivered.
            os.replace(target, backup)
            os.replace(staged, target)
        elif not existed and not staged.exists() and target.exists():
            os.replace(target, staged)
        raise


def _copy_entry(source, target):
    if source.is_symlink():
        target.symlink_to(os.readlink(source), target_is_directory=source.is_dir())
    elif source.is_dir():
        shutil.copytree(source, target, symlinks=True)
    else:
        shutil.copy2(source, target)


def write_parquet_batches(
    *,
    batches,
    target,
    columns,
    polars_module,
    compression,
    single_file=False,
    staging_dir=None,
    batch_size=10000,
    schema=None,
):
    """Write batches without publishing until all conversion succeeds.

    ``schema`` is an internal optional Polars schema hook for Model-aware callers.
    Overrides are validated losslessly, including empty result schemas. Int128
    is stored as Decimal(38, 0), recursively, for DuckDB interoperability.
    On Linux, replacing a directory uses an atomic exchange. Other filesystems
    use two renames with rollback; readers may observe a brief absent path. This
    is exception safety, not crash durability or concurrent-writer coordination.
    """
    iterator = None
    try:
        pl = polars_module
        target = Path(target)
        compression = validate_parquet_compression(compression)
        columns = list(columns)
        final_schema = (
            dict(schema) if schema is not None else dict.fromkeys(columns, pl.Null)
        )
        final_schema = {
            name: _parquet_dtype(pl, dtype) for name, dtype in final_schema.items()
        }
        if list(final_schema) != columns or len(set(columns)) != len(columns):
            raise ValueError("Parquet schema must match the selected columns in order")
        if target.is_symlink():
            raise ValueError("Parquet output must not be a symbolic link")
        if target.exists() and (
            target.is_dir() if single_file else not target.is_dir()
        ):
            raise ValueError(
                "path must be a file when single_file is True"
                if single_file
                else "path must be a directory when single_file is False"
            )
        iterator = iter(batches)
        with TemporaryDirectory(
            prefix="intermine314-parquet-", dir=staging_dir
        ) as scratch:
            parts = Path(scratch) / "parts"
            parts.mkdir()
            count = 0
            for batch in iterator:
                if not batch:
                    continue
                frame = _batch_frame(pl, batch, columns)
                if schema is not None:
                    frame = _cast_losslessly(frame, final_schema)
                else:
                    final_schema = {
                        name: _common_dtype(pl, final_schema[name], dtype)
                        for name, dtype in frame.schema.items()
                    }
                frame.write_parquet(
                    parts / f"part-{count:05d}.parquet", compression=compression
                )
                count += 1
                del frame, batch
            if not count:
                pl.DataFrame(schema=final_schema).write_parquet(
                    parts / "part-00000.parquet", compression=compression
                )
                count = 1
            # Never collect the dataset. Each staged file is at most one batch.
            for index in range(count):
                part = parts / f"part-{index:05d}.parquet"
                if dict(pl.read_parquet_schema(part)) != final_schema:
                    frame = _cast_losslessly(
                        _read_part(pl, part, scratch), final_schema
                    )
                    frame.write_parquet(part, compression=compression)
                    del frame
            # Final staging must share the target filesystem for atomic rename,
            # even when the requested scratch directory lives on another device.
            target.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(
                prefix=f".{target.name}-publish-", dir=target.parent
            ) as publication:
                publication = Path(publication)
                staged = publication / "output"
                if single_file:
                    write_single_parquet_from_parts(
                        staged_dir=parts,
                        target=staged,
                        compression=compression,
                        scratch=scratch,
                        batch_size=batch_size,
                    )
                    _publish_file(staged, target, publication / "backup")
                else:
                    shutil.copytree(parts, staged)
                    if target.exists():
                        for entry in target.iterdir():
                            if not (entry.is_file() and entry.match("part-*.parquet")):
                                _copy_entry(entry, staged / entry.name)
                    _publish_directory(staged, target, publication / "backup")
    finally:
        if iterator is not batches:
            close_resource_quietly(iterator)
        close_resource_quietly(batches)
    return str(target)
