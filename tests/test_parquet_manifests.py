import os
from datetime import UTC, datetime

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from intermine314.export import ParquetOptions, query_parquet
from intermine314.export.manifest import MANIFEST_FILE, METADATA_DIRECTORY
from tests.test_parquet_writer import query_with_rows


def test_manifest_is_relative_complete_and_excluded_from_queries(tmp_path):
    target = tmp_path / 'result'
    values = [{'Gene.id': value} for value in range(7)]
    query_with_rows(values, ['Gene.id']).to_parquet(target, batch_size=2, parquet_options=ParquetOptions(compact=False))
    manifest = pq.read_table(target / METADATA_DIRECTORY / MANIFEST_FILE)
    rows = manifest.to_pylist()
    assert len(rows) == 4 and sum(row['rows'] for row in rows) == 7
    for row in rows:
        part = target / row['relative_path']
        assert part.parent == target and part.is_file()
        assert row['file_size'] == part.stat().st_size
        assert row['mtime_ns'] == part.stat().st_mtime_ns
        assert row['row_groups'] == pq.read_metadata(part).num_row_groups
        assert pa.ipc.read_schema(pa.BufferReader(row['schema_ipc'])) == pq.read_schema(part)
    assert query_parquet(target).to_dicts() == values


def test_recursive_external_dataset_excludes_metadata_namespace(tmp_path):
    data = tmp_path / 'partition'
    data.mkdir()
    pl.DataFrame({'x': [1]}).write_parquet(data / 'data.parquet')
    metadata = tmp_path / METADATA_DIRECTORY
    metadata.mkdir()
    pl.DataFrame({'unrelated_metadata': [999]}).write_parquet(metadata / MANIFEST_FILE)
    assert query_parquet(tmp_path).to_dicts() == [{'x': 1}]


def test_schema_cache_rechecks_replaced_files_even_with_old_mtime(tmp_path, monkeypatch):
    target = tmp_path / 'data.parquet'
    pl.DataFrame({'x': [1]}).write_parquet(target)
    scan = pl.scan_parquet
    scans = []

    def tracked(*args, **kwargs):
        scans.append(args[0])
        return scan(*args, **kwargs)

    monkeypatch.setattr(pl, 'scan_parquet', tracked)
    assert query_parquet(target).item() == 1
    assert query_parquet(target).item() == 1
    assert len(scans) == 1
    previous = target.stat()
    pl.DataFrame({'x': ['new']}).write_parquet(target)
    os.utime(target, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert query_parquet(target).item() == 'new'
    assert len(scans) == 2


def test_stale_manifest_cannot_bypass_timestamp_precision_guard(tmp_path):
    target = tmp_path / 'result'
    query_with_rows([{'Gene.id': 1}], ['Gene.id']).to_parquet(target)
    assert query_parquet(target).item() == 1
    pl.DataFrame({'Gene.id': [datetime(2026, 1, 1, tzinfo=UTC)]}, schema={'Gene.id': pl.Datetime('ns', 'UTC')}).write_parquet(target / 'part-00000.parquet')
    with pytest.raises(ValueError, match='nanosecond.*timezone'):
        query_parquet(target)


def test_refresh_manifest_preserves_other_metadata_entries(tmp_path):
    target = tmp_path / 'result'
    query_with_rows([{'Gene.id': 1}], ['Gene.id']).to_parquet(target)
    (target / METADATA_DIRECTORY / 'notes.txt').write_text('keep')
    query_with_rows([{'Gene.id': 2}, {'Gene.id': 3}], ['Gene.id']).to_parquet(target)
    assert (target / METADATA_DIRECTORY / 'notes.txt').read_text() == 'keep'
    assert pq.read_table(target / METADATA_DIRECTORY / MANIFEST_FILE)['rows'].to_pylist() == [2]


def test_unrecognized_manifest_is_not_replaced(tmp_path):
    target = tmp_path / 'result'
    metadata = target / METADATA_DIRECTORY
    metadata.mkdir(parents=True)
    unrelated = metadata / MANIFEST_FILE
    pl.DataFrame({'user_data': [99]}).write_parquet(unrelated)
    before = unrelated.read_bytes()
    with pytest.raises(ValueError, match='not managed'):
        query_with_rows([{'Gene.id': 1}], ['Gene.id']).to_parquet(target)
    assert unrelated.read_bytes() == before
    assert list(tmp_path.iterdir()) == [target]
