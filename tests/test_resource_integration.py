from io import StringIO

import polars as pl
import pytest

from intermine314.export import (
    ResourceProfile,
    ResourceRuntime,
    fetch_from_mine,
    query_parquet,
)
from intermine314.query import ParallelOptions, Query


@pytest.mark.parametrize('profile', ['cpu16', 'cpu8', 'cpu6', 'cpu4', 'cpu2'])
def test_profile_reaches_owned_duckdb_and_polars_pipeline(tmp_path, profile):
    with ResourceRuntime(profile) as runtime:
        path = tmp_path / 'results.parquet'
        frame = Query().dataframe(csv_input=StringIO('id,value\n0001,7\n0002,9\n'),
                                  csv_options={'schema_overrides': {'id': pl.String}},
                                  parquet_path=path, runtime=runtime)
        assert frame.rows() == [('0001', 7), ('0002', 9)]
        result = query_parquet(path, "SELECT current_setting('threads') AS threads", runtime=runtime)
        assert result.item() == runtime.cpu_budget
        assert runtime.diagnostics()['active_native_reservations'] == 0


def test_conflicting_profile_fails_before_input_is_read(tmp_path):
    class Unreadable:
        def read(self, *args):
            pytest.fail('conflicting settings must be rejected before input')

    with ResourceRuntime('cpu2') as runtime:
        with pytest.raises(ValueError, match='conflicts'):
            Query().to_parquet(tmp_path / 'no-output', csv_input=Unreadable(),
                               runtime=runtime, resource_profile='cpu4')
    assert not list(tmp_path.iterdir())


def test_explicit_parallel_controls_survive_profile_defaults():
    query = Query()
    profile = ResourceProfile(cpu_budget=6, max_workers=4, inflight_limit=4, memory_budget_bytes=1024**2)
    with ResourceRuntime(profile) as runtime, runtime.activate():
        options = ParallelOptions(max_workers=7, prefetch=3, ordered=True)
        assert query._coerce_parallel_options(parallel_options=options) is options
        resolved = query._resolve_parallel_options(start=0, size=10, options=options)
        assert resolved.max_workers == 7 and resolved.prefetch == 3
        assert resolved.inflight_limit == 4 and resolved.order_mode == 'ordered'
        assert resolved.max_inflight_bytes_estimate == 1024**2 // 8
        default = query._resolve_parallel_options(start=0, size=10, options=ParallelOptions())
        assert default.max_workers == 4 and default.prefetch == 4


def test_fetch_csv_runtime_and_owned_connection(tmp_path):
    with ResourceRuntime('cpu2') as runtime:
        result = fetch_from_mine(csv_input=StringIO('id\n1\n2\n'),
                                 parquet_path=tmp_path/'fetch.parquet', runtime=runtime)
        connection = result['duckdb_connection']
        try:
            assert connection.execute("SELECT current_setting('threads')").fetchone() == (runtime.cpu_budget,)
            assert connection.execute('SELECT count(*) FROM results').fetchone() == (2,)
        finally:
            connection.close()
