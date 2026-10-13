"""Caller-visible logical types across the upgraded analytical engines."""

import polars as pl
import pytest

from intermine314.export import query_parquet
from intermine314.export.parquet import write_parquet_batches


@pytest.mark.parametrize('dtype', [pl.Enum(['a', 'b']), pl.Categorical])
@pytest.mark.parametrize('single_file', [False, True])
def test_dictionary_values_survive_and_duckdb_exposes_strings(tmp_path, dtype, single_file):
    original = pl.DataFrame({'category': pl.Series(['b', None, 'a', 'b'], dtype=dtype)})
    target = tmp_path / 'categories'
    write_parquet_batches(
        batches=[original], target=target, columns=original.columns,
        polars_module=pl, compression='zstd', single_file=single_file,
    )
    direct = pl.read_parquet(target if single_file else target / 'part-00000.parquet')
    result = query_parquet(target)
    assert direct['category'].to_list() == original['category'].to_list()
    assert result['category'].to_list() == original['category'].to_list()
    assert result.schema['category'] == pl.String
    # Direct directory parts keep Polars metadata; COPY goes through DuckDB.
    assert direct.schema == ({'category': pl.String} if single_file else original.schema)
