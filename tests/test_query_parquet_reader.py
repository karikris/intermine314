from concurrent.futures import ThreadPoolExecutor

import duckdb
import polars as pl
import pyarrow as pa
import pytest

from intermine314.export import (
    ResourceProfile,
    ResourceRuntime,
    query_parquet,
    query_parquet_reader,
)
from intermine314.export.runtime import _CURRENT


def test_reader_bounds_batches_preserves_values_and_releases_runtime(typed_parquet):
    with ResourceRuntime('cpu4') as runtime:
        with query_parquet_reader(typed_parquet, batch_size=2, runtime=runtime) as reader:
            assert _CURRENT.get() is None
            batches = list(reader)
            assert [len(batch) for batch in batches] == [2, 1]
            assert reader.closed
            assert reader not in runtime._owned
            assert runtime.diagnostics()['active_native_reservations'] == 0
        assert pl.from_arrow(pa.Table.from_batches(batches)).equals(query_parquet(typed_parquet))


def test_early_close_and_runtime_close_release_spill_directory(typed_parquet, tmp_path):
    runtime = ResourceRuntime(ResourceProfile(temp_dir=str(tmp_path)))
    reader = query_parquet_reader(typed_parquet, batch_size=1, runtime=runtime)
    batch = next(reader)
    assert len(list(tmp_path.iterdir())) == 2
    reader.close()
    reader.close()
    assert list(tmp_path.iterdir()) == [typed_parquet]
    assert len(batch) == 1
    with pytest.raises(StopIteration):
        next(reader)
    with pytest.raises(RuntimeError, match='closed'):
        with reader:
            pass
    idle = query_parquet_reader(typed_parquet, runtime=runtime)
    runtime.close()
    assert idle.closed
    assert list(tmp_path.iterdir()) == [typed_parquet]


def test_borrowed_connection_and_other_readers_are_unchanged(typed_parquet):
    with duckdb.connect(config={'threads': 2, 'TimeZone': 'Australia/Sydney'}) as connection:
        connection.execute('CREATE TABLE results AS SELECT 123 AS preserved')
        settings = connection.execute("SELECT current_setting('threads'), current_setting('TimeZone'), current_setting('memory_limit')").fetchone()
        first = query_parquet_reader(typed_parquet, connection=connection, batch_size=1)
        second = query_parquet_reader(typed_parquet, 'SELECT identifier FROM results WHERE identifier=?',
                                      parameters=['0002'], connection=connection)
        with first, second:
            assert next(first).column(0).to_pylist() == ['0007']
            assert second.read_all().column(0).to_pylist() == ['0002']
            assert next(first).column(0).to_pylist() == ['0002']
            frame = query_parquet(typed_parquet, 'SELECT identifier FROM results WHERE identifier=?',
                                  parameters=['0007'], connection=connection)
            assert frame['identifier'].to_list() == ['0007']
            with pytest.raises(duckdb.Error):
                query_parquet(typed_parquet, 'SELECT absent FROM results', connection=connection)
        assert connection.execute('SELECT * FROM results').fetchone() == (123,)
        assert connection.execute("SELECT current_setting('threads'), current_setting('TimeZone'), current_setting('memory_limit')").fetchone() == settings
        with pytest.raises(ValueError, match='database'):
            query_parquet(typed_parquet, database='conflict.db', connection=connection)
    assert frame['identifier'].to_list() == ['0007']


@pytest.mark.parametrize('failure', [RuntimeError, KeyboardInterrupt])
def test_reader_failure_closes_all_resources(typed_parquet, tmp_path, failure):
    with ResourceRuntime(ResourceProfile(temp_dir=str(tmp_path))) as runtime:
        reader = query_parquet_reader(typed_parquet, runtime=runtime)
        class Broken:
            def __next__(self):
                raise failure('injected batch failure')
        reader._reader = Broken()
        with pytest.raises(failure, match='injected'):
            next(reader)
        assert reader.closed
        assert reader not in runtime._owned
        assert list(tmp_path.iterdir()) == [typed_parquet]


def test_bad_sql_releases_scratch(typed_parquet, tmp_path):
    with ResourceRuntime(ResourceProfile(temp_dir=str(tmp_path))) as runtime:
        with pytest.raises(duckdb.Error):
            query_parquet_reader(typed_parquet, 'SELECT absent FROM results', runtime=runtime)
        assert runtime._owned == []
        assert list(tmp_path.iterdir()) == [typed_parquet]


@pytest.mark.parametrize('batch_size', [0, -1, True, 2.5, '2'])
def test_invalid_batch_size_before_io(batch_size):
    with pytest.raises((ValueError, TypeError), match='batch_size'):
        query_parquet_reader('absent.parquet', batch_size=batch_size)


def test_reader_reservations_do_not_span_yields_or_conflict_with_other_readers(typed_parquet):
    with ResourceRuntime('cpu2') as runtime:
        def read():
            with query_parquet_reader(typed_parquet, batch_size=1, runtime=runtime) as reader:
                return sum(len(batch) for batch in reader)
        with ThreadPoolExecutor(max_workers=4) as executor:
            assert list(executor.map(lambda _: read(), range(8))) == [3] * 8
        assert runtime.diagnostics()['active_native_reservations'] == 0
