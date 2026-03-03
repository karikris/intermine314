from __future__ import annotations

from unittest.mock import patch

from intermine314.query import builder as query_builder


class _FakeDuckDBConnection:
    def __init__(self):
        self.sql = []
        self.close_calls = 0

    def execute(self, query, params=None):
        self.sql.append((query, params))
        return self

    def close(self):
        self.close_calls += 1


class _FakeDuckDBModule:
    def __init__(self):
        self.connections = []

    def connect(self, database=":memory:"):
        con = _FakeDuckDBConnection()
        con.database = database
        self.connections.append(con)
        return con


class _DuckDBQueryHarness:
    def __init__(self):
        pass

    def _coerce_parallel_options(self, *, parallel_options=None):
        return parallel_options

    def to_parquet(self, path, **kwargs):
        _ = kwargs
        return str(path)


def test_to_duckdb_managed_mode_closes_connection_on_context_exit():
    fake_duckdb = _FakeDuckDBModule()
    harness = _DuckDBQueryHarness()

    with patch("intermine314.query.builder._require_duckdb", return_value=fake_duckdb):
        managed = query_builder.Query.to_duckdb(harness, "/tmp/managed_mode.parquet", managed=True)
        with managed as con:
            assert isinstance(con, _FakeDuckDBConnection)
            assert con.close_calls == 0
            _ = con.execute("select 1")
        assert con.close_calls == 1
