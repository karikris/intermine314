from decimal import Decimal

import polars as pl
import pytest

from intermine314.export import parquet
from tests.test_parquet_writer import query_with_rows


def test_fixed_schema_never_reopens_staged_parts_for_metadata(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Stable owned schema must not reread staged data or metadata')

    monkeypatch.setattr(parquet, 'read_parquet_schema_literal', forbidden)
    monkeypatch.setattr(parquet, '_read_part', forbidden)
    rows = [{'Employee.id': i} for i in range(40)]
    query_with_rows(rows, ['Employee.id']).to_parquet(tmp_path / 'parts', batch_size=1)
    result = pl.concat([pl.read_parquet(path) for path in sorted((tmp_path / 'parts').glob('part-*.parquet'))])
    assert result.to_dicts() == rows


def test_only_promoted_parts_are_rewritten_and_no_footer_is_rediscovered(tmp_path, monkeypatch):
    reads = []
    read = parquet._read_part

    def track(pl, part, scratch, **kwargs):
        reads.append(part.name)
        return read(pl, part, scratch, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail('Owned schema must come from the successful write')

    monkeypatch.setattr(parquet, '_read_part', track)
    monkeypatch.setattr(parquet, 'read_parquet_schema_literal', forbidden)
    rows = [{'Employee.id': None}, {'Employee.id': Decimal('1.2')},
            {'Employee.id': Decimal('0.001')}, {'Employee.id': Decimal('2.345')}]
    query_with_rows(rows, ['Employee.id']).to_parquet(tmp_path / 'parts', batch_size=1)
    assert reads == ['part-00000.parquet', 'part-00001.parquet']
    frames = [pl.read_parquet(path) for path in sorted((tmp_path / 'parts').glob('part-*.parquet'))]
    assert all(frame.schema == {'Employee.id': pl.Decimal(38, 3)} for frame in frames)
    assert pl.concat(frames).to_dicts() == rows
