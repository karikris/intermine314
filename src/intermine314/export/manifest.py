"""Bounded Parquet metadata for atomically published directories."""

from pathlib import Path

from intermine314.export.runtime import get_runtime
from intermine314.util.deps import require_pyarrow

METADATA_DIRECTORY = '_intermine314'
MANIFEST_FILE = 'manifest.parquet'


def is_metadata(path):
    return METADATA_DIRECTORY in Path(path).parts


def write_manifest(directory, *, ordered):
    """Inspect owned files, recording relative paths and native Arrow schemas.

    A manifest is descriptive, never authority to skip external schema checks.
    It is published with the containing directory, without a two-file commit.
    """
    pa = require_pyarrow('Parquet manifest')
    import pyarrow.parquet as pq

    directory = Path(directory)
    metadata = directory / METADATA_DIRECTORY
    metadata.mkdir()
    schema = pa.schema([
        ('relative_path', pa.string()), ('file_size', pa.int64()), ('mtime_ns', pa.int64()),
        ('rows', pa.int64()), ('row_groups', pa.int32()), ('schema_ipc', pa.binary()),
        ('ordered', pa.bool_()),
    ], metadata={b'intermine314.manifest': b'1'})
    rows = []
    with get_runtime().native('arrow_cpu'), pq.ParquetWriter(metadata / MANIFEST_FILE, schema, compression='zstd') as writer:
        for part in sorted(directory.glob('part-*.parquet'), key=lambda path: int(path.stem[5:])):
            info = part.stat()
            with pq.ParquetFile(part) as file:
                rows.append(dict(relative_path=part.name, file_size=info.st_size, mtime_ns=info.st_mtime_ns,
                                 rows=file.metadata.num_rows, row_groups=file.metadata.num_row_groups,
                                 schema_ipc=file.schema_arrow.serialize().to_pybytes(), ordered=ordered))
            if len(rows) == 128:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                rows.clear()
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))


def validate_owned_metadata_namespace(path):
    """Refuse a conflicting user file instead of replacing unrelated content."""
    path = Path(path)
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f'Parquet metadata namespace conflicts with an existing entry: {path.name}')
    manifest = path / MANIFEST_FILE
    if not manifest.exists():
        return
    require_pyarrow('Parquet manifest')
    import pyarrow.parquet as pq

    try:
        if manifest.is_symlink() or pq.read_schema(manifest).metadata.get(b'intermine314.manifest') != b'1':
            raise ValueError('unrecognized metadata')
    except Exception as exc:
        raise ValueError('Existing manifest.parquet is not managed InterMine metadata') from exc
