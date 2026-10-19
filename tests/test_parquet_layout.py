import polars as pl
import pyarrow.parquet as pq
import pytest

from intermine314.export import ParquetOptions, query_parquet
from intermine314.query import Query
from tests.support.parquet import query_with_rows


@pytest.mark.parametrize('options', [dict(row_group_size=0), dict(row_group_size=True),
                                    dict(row_group_size=2000), dict(target_file_bytes=-1),
                                    dict(row_group_bytes=False), dict(preserve_order=1)])
def test_geometry_validation_precedes_io(options):
    with pytest.raises((TypeError, ValueError)):
        ParquetOptions(**options)


def test_producer_batches_and_row_groups_are_independent(tmp_path):
    query = Query().select('Gene.id')
    total = 245761
    query.iter_batches = lambda **kwargs: (
        pl.DataFrame({'Gene.id': range(start, min(start + 10000, total))}) for start in range(0, total, 10000)
    )
    target = tmp_path / 'result'
    query.to_parquet(target)
    parts = sorted(target.glob('part-*.parquet'))
    assert len(parts) == 1
    metadata = pq.read_metadata(parts[0])
    counts = [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]
    # DuckDB's target is approximate at vector boundaries.
    assert 2 <= len(counts) <= 3 and sum(counts) == total
    assert 122880 <= counts[0] < 122880 + 2048
    assert max(counts) < 122880 + 2048
    values = query_parquet(target)['Gene.id']
    assert len(values) == total and values.is_sorted()
    assert values.sum() == total * (total - 1) // 2


def test_file_rotation_keeps_global_row_order(tmp_path):
    total = 12000
    rows = [{'Gene.id': i} for i in range(total)]
    target = tmp_path / 'parts'
    query_with_rows(rows, ['Gene.id']).to_parquet(
        target, batch_size=500,
        parquet_options=ParquetOptions(row_group_size=2048, target_file_bytes=8000),
    )
    parts = sorted(target.glob('part-*.parquet'))
    assert 1 < len(parts) < 24
    assert query_parquet(target).to_dicts() == rows


def test_explicit_producer_part_mode_and_partitioned_dataframe(tmp_path):
    rows = [{'Gene.id': i} for i in range(3)]
    query = query_with_rows(rows, ['Gene.id'])
    query.to_parquet(tmp_path / 'producer', batch_size=1, parquet_options=ParquetOptions(compact=False))
    assert len(list((tmp_path / 'producer').glob('part-*.parquet'))) == 3
    result = query.dataframe(parquet_path=tmp_path / 'dataframe', single_file=False)
    assert (tmp_path / 'dataframe').is_dir()
    assert result.to_dicts() == rows


def test_wide_rows_reduce_row_group_target():
    layout = ParquetOptions()
    rows, budget = layout.effective_row_group(bytes_per_row=10000, memory_budget=256 * 1024 * 1024)
    assert rows < layout.row_group_size
    assert rows * 10000 <= budget <= 32 * 1024 * 1024


def test_directory_compaction_preserves_timestamp_metadata(tmp_path):
    from datetime import UTC, datetime

    rows = [{'Gene.at': datetime(2026, 1, 1, tzinfo=UTC)}] * 2
    query = query_with_rows(rows, ['Gene.at'])
    query._parquet_schema = lambda: {'Gene.at': pl.Datetime('ns', 'UTC')}
    query.to_parquet(tmp_path / 'timestamps', batch_size=1)
    parts = list((tmp_path / 'timestamps').glob('part-*.parquet'))
    assert len(parts) == 2
    assert all(pl.read_parquet(path).schema == {'Gene.at': pl.Datetime('ns', 'UTC')} for path in parts)
