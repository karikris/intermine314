"""Optional enrichment persistence using the shared bounded export pipeline."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from intermine314.export.csv import _local_path
from intermine314.export.output import write_csv_from_parquet
from intermine314.export.parquet import _publish_file, write_parquet_batches
from intermine314.export.query import query_parquet
from intermine314.service.resource_utils import close_resource_quietly
from intermine314.util.deps import require_polars

_FIELDS = ("identifier", "description", "p-value", "matches", "populationAnnotationCount")


def validate_options(path, format, batch_size):
    """Reject invalid persistence options before opening the HTTP stream."""
    if not isinstance(format, str) or format.lower() not in ("parquet", "csv"):
        raise ValueError("format must be 'parquet' or 'csv'")
    format = format.lower()
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    target = _local_path(path, "Enrichment output")
    if (format == "parquet" and target.suffix.lower() == ".csv") or (
        format == "csv" and target.suffix.lower() == ".parquet"
    ):
        raise ValueError(f"Output suffix {target.suffix!r} conflicts with format={format!r}")
    if target.is_symlink():
        raise ValueError("Enrichment output must not be a symbolic link")
    if target.exists() and not target.is_file():
        raise ValueError("Enrichment output must be a file")
    return target, format


def _batches(stream, batch_size):
    try:
        batch = []
        for line in stream:
            batch.append(dict(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    finally:
        close_resource_quietly(stream)


def persist(stream, target, *, format, batch_size):
    """Materialize the requested frame before publishing any final output.

    Parquet and CSV publication use existing atomic rollback helpers; all
    staged data and owned connections remain managed on BaseException paths.
    """
    pl = require_polars("List.calculate_enrichment(output_path=...)")
    schema = dict(zip(_FIELDS, (pl.String, pl.String, pl.Float64, pl.Int64, pl.Int64)))
    # Same-filesystem scratch allows final publication by atomic replacement.
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{target.name}-enrichment-", dir=target.parent) as scratch:
        scratch = Path(scratch)
        staged = scratch / "results.parquet"
        write_parquet_batches(
            batches=_batches(stream, batch_size), target=staged, columns=_FIELDS,
            polars_module=pl, compression=None, single_file=True,
            staging_dir=scratch, batch_size=batch_size, schema=schema,
        )
        frame = query_parquet(staged)
        if format == "parquet":
            _publish_file(staged, target, scratch / "backup")
        else:
            write_csv_from_parquet(staged, target, scratch=scratch)
        return frame
