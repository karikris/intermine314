import duckdb
import pytest

from intermine314.export import (
    ParquetOptions,
    ResourceProfile,
    ResourceRuntime,
    query_parquet,
)
from tests.test_parquet_writer import query_with_rows


@pytest.mark.parametrize('profile', ['cpu16', 'cpu8', 'cpu6', 'cpu4', 'cpu2'])
@pytest.mark.parametrize('ordered,single_file', [(True, True), (False, True), (True, False), (False, False)])
def test_copy_uses_one_profiled_connection_and_correct_order_policy(tmp_path, monkeypatch, profile, ordered, single_file):
    from intermine314.export import resource_profile

    monkeypatch.setattr(resource_profile, 'available_cpus', lambda: 20)
    configurations, statements = [], []
    connect = duckdb.connect

    class Connection:
        def __init__(self, *args, **kwargs):
            configurations.append(kwargs['config'])
            self.inner = connect(*args, **kwargs)

        def execute(self, sql, *args):
            statements.append(sql)
            return self.inner.execute(sql, *args)

        def close(self):
            self.inner.close()

    monkeypatch.setattr(duckdb, 'connect', Connection)
    values = [{'Gene.id': value} for value in range(8)]
    target = tmp_path / 'output'
    with ResourceRuntime(profile) as runtime:
        query_with_rows(values, ['Gene.id']).to_parquet(
            target, batch_size=2, single_file=single_file, runtime=runtime,
            parquet_options=ParquetOptions(preserve_order=ordered),
        )
        expected = 1 if ordered and not single_file else runtime.profile.workers_for('ordered_copy' if ordered else 'unordered_copy')
        assert len(configurations) == 1
        assert configurations[0]['threads'] == expected
        assert configurations[0]['preserve_insertion_order'] is ordered
        assert configurations[0]['memory_limit'].endswith('B')
        assert configurations[0]['temp_directory'].endswith('spill')
        assert len(statements) == 1 and statements[0].startswith('COPY')
        assert ('ROW_GROUP_SIZE_BYTES' in statements[0]) is not ordered
    monkeypatch.setattr(duckdb, 'connect', connect)
    result = query_parquet(target)
    assert result.sort('Gene.id').to_dicts() == values
    if ordered:
        assert result.to_dicts() == values


def test_low_memory_ordered_copy_can_request_one_thread(tmp_path, monkeypatch):
    configurations = []
    connect = duckdb.connect

    def capture(*args, **kwargs):
        configurations.append(kwargs['config'])
        return connect(*args, **kwargs)

    monkeypatch.setattr(duckdb, 'connect', capture)
    profile = ResourceProfile(cpu_budget=16, duckdb_threads=1, memory_budget_bytes=128 * 1024 * 1024)
    query_with_rows([{'Gene.id': 1}], ['Gene.id']).to_parquet(tmp_path / 'one.parquet', single_file=True, resource_profile=profile)
    assert configurations[0]['threads'] == 1
    assert int(configurations[0]['memory_limit'][:-1]) < profile.memory_budget_bytes
