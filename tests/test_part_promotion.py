from datetime import date

import duckdb
import polars as pl
import pyarrow.parquet as pq
import pytest

from intermine314.export import parquet as writer
from intermine314.export.layout import ParquetOptions


@pytest.mark.parametrize('dtype,values', [
    (pl.Boolean, [True, None, False]), (pl.Int8, [-128, None, 127]),
    (pl.Int16, [-32768, 32767]), (pl.Int32, [-(2**31), 2**31 - 1]),
    (pl.Int64, [-(2**63), 2**63 - 1]), (pl.UInt8, [0, 255]),
    (pl.UInt16, [0, 65535]), (pl.UInt32, [0, 2**32 - 1]),
    (pl.UInt64, [0, 2**64 - 1]), (pl.Float32, [1.25, None, float('inf')]),
    (pl.Float64, [1.25, float('nan')]), (pl.Date, [date(2026, 10, 14), None]),
])
def test_promoted_part_matches_copy_in_all_three_engines(dtype, values, tmp_path, monkeypatch):
    frame = pl.DataFrame([pl.Series('value', values, dtype=dtype)])
    promoted, copied = tmp_path / 'promoted.parquet', tmp_path / 'copied.parquet'
    calls = []
    copy = writer.write_single_parquet_from_parts

    def tracked(**kwargs):
        calls.append(kwargs['target'])
        return copy(**kwargs)

    monkeypatch.setattr(writer, 'write_single_parquet_from_parts', tracked)
    options = dict(batches=[frame], columns=frame.columns, polars_module=pl,
                   compression='zstd', single_file=True)
    writer.write_parquet_batches(target=promoted, **options)
    assert not calls
    monkeypatch.setattr(writer, '_promotable_part', lambda *args: False)
    writer.write_parquet_batches(target=copied, **options)
    assert len(calls) == 1
    assert pl.read_parquet(promoted).equals(pl.read_parquet(copied))
    with pq.ParquetFile(promoted) as first, pq.ParquetFile(copied) as second:
        assert first.schema_arrow.equals(second.schema_arrow, check_metadata=True)
        assert first.metadata.num_row_groups == second.metadata.num_row_groups == 1
        assert first.metadata.row_group(0).num_rows == second.metadata.row_group(0).num_rows == len(values)
        assert first.metadata.row_group(0).column(0).compression == second.metadata.row_group(0).column(0).compression == 'ZSTD'
    with duckdb.connect() as db:
        assert db.execute('DESCRIBE SELECT * FROM read_parquet(?)', [str(promoted)]).fetchall() == db.execute(
            'DESCRIBE SELECT * FROM read_parquet(?)', [str(copied)]).fetchall()


@pytest.mark.parametrize('kind', ['string', 'empty', 'multiple_groups', 'other_codec'])
def test_ineligible_parts_retain_copy(kind, tmp_path, monkeypatch):
    frame = pl.DataFrame({'value': ['a']}) if kind == 'string' else pl.DataFrame({'value': list(range(5000)) if kind == 'multiple_groups' else [1]})
    if kind == 'empty':
        frame = frame.clear()
    calls = []
    original = writer.write_single_parquet_from_parts

    def copy(**kwargs):
        calls.append(True)
        return original(**kwargs)

    monkeypatch.setattr(writer, 'write_single_parquet_from_parts', copy)
    writer.write_parquet_batches(batches=[frame], target=tmp_path / 'out.parquet', columns=['value'],
                                 polars_module=pl, compression='snappy' if kind == 'other_codec' else 'zstd',
                                 single_file=True, parquet_options=ParquetOptions(row_group_size=2048))
    assert calls == [True]
