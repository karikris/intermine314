"""Local CSV scans with atomic, single-file Parquet publication."""

from __future__ import annotations

import inspect
import os
import re
import shutil
from collections.abc import Mapping
from contextlib import ExitStack
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from intermine314.export._schema import validate_duckdb_schema
from intermine314.export.parquet import (
    _connection,
    _parquet_dtype,
    _publish_file,
    write_parquet_batches,
)
from intermine314.service.resource_utils import close_resource_quietly
from intermine314.util.deps import require_polars, require_pyarrow


def _local_path(value, label):
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{label} must be a local Path or string")
    if not str(value) or "\x00" in str(value):
        raise ValueError(f"{label} must be a nonempty path without null bytes")
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", str(value)):
        raise ValueError(f"{label} must be a local filesystem path")
    return Path(value).expanduser()


def _copy_options(value):
    # Polars accepts dictionaries/lists for schemas and nulls. Give it owned
    # containers, without copying dtype classes or user callbacks.
    if isinstance(value, Mapping):
        return {key: _copy_options(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_options(item) for item in value]
    return value


def _csv_options(pl, options):
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise TypeError("csv_options must be a mapping of Polars scan_csv options")
    if any(not isinstance(key, str) for key in options):
        raise TypeError("csv_options keys must be strings")
    options = _copy_options(options)
    # Validate keyword names against the installed Polars version, including
    # rejection of a second source supplied through csv_options.
    inspect.signature(pl.scan_csv).bind(None, **options)
    if options.get("glob", False) is not False:
        raise ValueError("csv_options glob must be False: input paths are literal")
    options["glob"] = False
    for key in ("separator", "quote_char", "eol_char"):
        if key not in options or (key == "quote_char" and options[key] is None):
            continue
        value = options[key]
        if not isinstance(value, str) or len(value.encode("utf-8")) != 1:
            raise ValueError(f"csv_options {key} must be a single-byte character")
    if options.get("encoding", "utf8") not in ("utf8", "utf8-lossy"):
        raise ValueError("csv_options encoding must be 'utf8' or 'utf8-lossy'")
    if isinstance(options.get("schema"), dict) and options["schema"]:
        options.setdefault("raise_if_empty", False)
    for key in ("schema", "schema_overrides"):
        schema = options.get(key)
        if isinstance(schema, dict):
            options[key] = {
                name: _parquet_dtype(pl, dtype) for name, dtype in schema.items()
            }
        elif isinstance(schema, list):
            options[key] = [_parquet_dtype(pl, dtype) for dtype in schema]
    return options


def _check_collision(source, target):
    if isinstance(source, Path):
        if source.resolve() == target.resolve() or (
            target.exists() and source.samefile(target)
        ):
            raise ValueError("CSV input and Parquet output must not be the same file")
        return
    # A borrowed file may also be the output (possibly through a hard link).
    if target.exists():
        name = getattr(source, "name", None)
        if isinstance(name, (str, Path)):
            named_path = Path(name).expanduser()
            if named_path.is_file() and named_path.samefile(target):
                raise ValueError("Parquet output must not overwrite the CSV source")
        try:
            info = os.fstat(source.fileno())
        except AttributeError, OSError, TypeError, ValueError:
            pass
        else:
            output = target.stat()
            if (info.st_dev, info.st_ino) == (output.st_dev, output.st_ino):
                raise ValueError("Parquet output must not overwrite the CSV source")


class _BorrowedStream:
    """Expose the reader interface without transferring close ownership."""

    def __init__(self, source):
        self.source = source

    def read(self, size=-1):
        return self.source.read(size)

    def seek(self, offset, whence=0):
        return self.source.seek(offset, whence)

    def close(self):
        pass


def _scan_source(source, stack):
    if isinstance(source, Path):
        return source
    # Polars snapshots stream contents for a lazy scan. Nonseekable readers
    # need an owned buffer because its IO interface also requires seek().
    if not callable(getattr(source, "seek", None)) or (
        callable(getattr(source, "seekable", None)) and not source.seekable()
    ):
        content = source.read()
        if not isinstance(content, (str, bytes)):
            raise TypeError("CSV input read() must return text or bytes")
        return stack.enter_context(
            StringIO(content) if isinstance(content, str) else BytesIO(content)
        )
    return _BorrowedStream(source)


def import_csv(csv_input, parquet_path, *, csv_options=None):
    """Scan local CSV data directly into one atomically published Parquet file.

    Accept a local Path/string or borrowed readable text/binary stream; borrowed
    streams remain open, but their cursor position is not preserved. Polars
    buffers stream inputs, and nonseekable readers use an owned in-memory
    buffer. File paths use a lazy scan/sink without collecting a DataFrame or
    creating a temporary CSV file. Paths containing glob characters are literal.

    ``csv_options`` forwards Polars ``scan_csv`` keywords without mutating the
    supplied mapping. Use ``schema_overrides={"id": polars.String}`` to preserve
    leading zeros. Encoding supports ``utf8`` and ``utf8-lossy``. Polars parsing
    options such as ``ignore_errors`` retain their normal opt-in semantics.
    Int128 overrides normalize to exact Decimal(38, 0) for DuckDB; nanosecond
    timestamps with timezones and columnless empty inputs are rejected.
    A nonempty full ``schema`` permits empty input by default, producing typed
    zero-row output; explicit ``raise_if_empty=True`` still rejects empty input.

    The sink completes in a temporary directory on the output filesystem before
    replacement. Failures and interrupts restore an existing output and remove
    scratch files. This is exception safety, not crash durability or coordination
    between concurrent writers.
    """
    return _import_csv(csv_input, parquet_path, csv_options=csv_options)


def _import_csv(
    csv_input, parquet_path, *, csv_options=None, start=0, size=None,
    compression="zstd", staging_dir=None,
):
    """Private sink controls for Query exports; keep import_csv's public API."""
    target = _local_path(parquet_path, "Parquet output")
    if target.is_symlink():
        raise ValueError("Parquet output must not be a symbolic link")
    if target.exists() and not target.is_file():
        raise ValueError("Parquet output must be a file")
    if isinstance(csv_input, (str, Path)):
        source = _local_path(csv_input, "CSV input")
        if not source.exists():
            raise FileNotFoundError(f"CSV input does not exist: {source}")
        if not source.is_file():
            raise ValueError(f"CSV input must be a file: {source}")
    else:
        source = csv_input
        if not callable(getattr(source, "read", None)):
            raise TypeError(
                "CSV input must be a local path or readable text/binary stream"
            )
        if getattr(source, "closed", False) or (
            callable(getattr(source, "readable", None)) and not source.readable()
        ):
            raise ValueError("CSV input stream must be open and readable")
    _check_collision(source, target)
    pl = require_polars("import_csv()")
    options = _csv_options(pl, csv_options)
    with ExitStack() as stack:
        frame = pl.scan_csv(_scan_source(source, stack), **options)
        schema = frame.collect_schema()
        if not schema:
            raise ValueError(
                "CSV input requires at least one column; supply a schema for empty input"
            )
        validate_duckdb_schema(pl, schema, "import_csv()")
        frame = frame.slice(start, size)
        target.parent.mkdir(parents=True, exist_ok=True)
        publication = Path(
            stack.enter_context(
                TemporaryDirectory(prefix=f".{target.name}-publish-", dir=target.parent)
            )
        )
        staged = publication / "output"
        if staging_dir is None:
            frame.sink_parquet(staged, compression=compression)
        else:
            scratch = Path(stack.enter_context(TemporaryDirectory(prefix="intermine314-csv-", dir=staging_dir)))
            intermediate = scratch / "output.parquet"
            frame.sink_parquet(intermediate, compression=compression)
            shutil.copyfile(intermediate, staged)
        _publish_file(staged, target, publication / "backup")
    return str(target)


def _export_csv(
    csv_input, target, *, csv_options, start, size, compression, single_file,
    staging_dir, batch_size, polars_module,
):
    """Sink once, then stream typed bounded Arrow batches for directory output."""
    target = _local_path(target, "Parquet output")
    if single_file:
        return _import_csv(
            csv_input, target, csv_options=csv_options, start=start, size=size,
            compression=compression, staging_dir=staging_dir,
        )
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError("Parquet output must be a directory when single_file is False")
    source = _local_path(csv_input, "CSV input") if isinstance(csv_input, (str, Path)) else csv_input
    _check_collision(source, target)
    pl = polars_module
    require_pyarrow("Query.to_parquet()")
    with TemporaryDirectory(prefix="intermine314-csv-", dir=staging_dir) as scratch:
        path = Path(scratch) / "input.parquet"
        _import_csv(
            csv_input, path, csv_options=csv_options, start=start, size=size,
            compression=compression,
        )
        schema = pl.scan_parquet(path, glob=False).collect_schema()

        def batches():
            # The owned reader closes before its connection, even on interrupts.
            from glob import escape

            with _connection(scratch) as connection:
                connection.execute("SET TimeZone = 'UTC'")
                reader = connection.execute(
                    "SELECT * FROM read_parquet(?)", [escape(str(path))]
                ).to_arrow_reader(batch_size)
                try:
                    for batch in reader:
                        yield pl.from_arrow(batch)
                finally:
                    close_resource_quietly(reader)

        return write_parquet_batches(
            batches=batches(), target=target, columns=list(schema),
            polars_module=pl, compression=compression, staging_dir=staging_dir,
            batch_size=batch_size, schema=schema,
        )
