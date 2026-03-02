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


class _FailingDuckDBConnection(_FakeDuckDBConnection):
    def execute(self, query, params=None):
        _ = (query, params)
        raise RuntimeError("duckdb view creation failed")


class _FailingDuckDBModule:
    def __init__(self):
        self.connections = []

    def connect(self, database=":memory:"):
        con = _FailingDuckDBConnection()
        con.database = database
        self.connections.append(con)
        return con


class _DuckDBQueryHarness:
    def __init__(self):
        self.parquet_calls = []

    def to_duckdb(self, *args, **kwargs):
        return query_builder.Query.to_duckdb(self, *args, **kwargs)

    def _coerce_parallel_options(self, *, parallel_options=None):
        return parallel_options

    def to_parquet(self, path, **kwargs):
        self.parquet_calls.append((path, kwargs))
        return str(path)


def test_to_duckdb_raw_connection_mode_is_unchanged():
    fake_duckdb = _FakeDuckDBModule()
    harness = _DuckDBQueryHarness()

    with patch("intermine314.query.builder._require_duckdb", return_value=fake_duckdb):
        con = query_builder.Query.to_duckdb(harness, "/tmp/raw_mode.parquet")

    assert isinstance(con, _FakeDuckDBConnection)
    assert con.close_calls == 0
    assert any("read_parquet" in sql for sql, _ in con.sql)


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


def test_duckdb_view_helper_returns_managed_connection_wrapper():
    fake_duckdb = _FakeDuckDBModule()
    harness = _DuckDBQueryHarness()

    with patch("intermine314.query.builder._require_duckdb", return_value=fake_duckdb):
        managed = query_builder.Query.duckdb_view(harness, "/tmp/duckdb_view.parquet")
        with managed as con:
            assert isinstance(con, _FakeDuckDBConnection)
        assert con.close_calls == 1


def test_to_duckdb_closes_connection_if_view_creation_fails():
    failing_duckdb = _FailingDuckDBModule()
    harness = _DuckDBQueryHarness()

    with patch("intermine314.query.builder._require_duckdb", return_value=failing_duckdb):
        try:
            query_builder.Query.to_duckdb(harness, "/tmp/failing_mode.parquet")
            assert False, "expected RuntimeError from failed DuckDB execute"
        except RuntimeError:
            pass

    assert len(failing_duckdb.connections) == 1
    assert failing_duckdb.connections[0].close_calls == 1
