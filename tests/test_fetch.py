from unittest.mock import patch

import pytest

from intermine314.export.fetch import fetch_from_mine


class _FakeQuery:
    def __init__(self):
        self.parquet_calls = []

    def add_view(self, *_views):
        return None

    def add_join(self, *_args, **_kwargs):
        return None

    def to_parquet(self, path, **kwargs):
        self.parquet_calls.append((path, kwargs))
        return str(path)


class _FakeService:
    last_query = None

    def __init__(self, _mine_url):
        pass

    def new_query(self, _root_class):
        query = _FakeQuery()
        _FakeService.last_query = query
        return query


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


def _plan(workflow="elt", workers=8, resource_profile=None):
    plan = {
        "name": f"{workflow}_plan",
        "workflow": workflow,
        "workers": workers,
        "pipeline": "parquet_duckdb" if workflow == "elt" else "polars_duckdb",
        "parallel_profile": "large_query",
        "ordered": "unordered",
        "large_query_mode": True,
        "prefetch": None,
        "inflight_limit": None,
    }
    if resource_profile is not None:
        plan["resource_profile"] = resource_profile
    return plan


@patch("intermine314.export.fetch.Service", _FakeService)
def test_fetch_elt_contract_plumbs_resource_and_backpressure_controls():
    fake_duckdb = _FakeDuckDBModule()
    with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
        with patch("intermine314.export.fetch.resolve_production_plan", return_value=_plan("elt", workers=8)):
            _ = fetch_from_mine(
                mine_url="https://maizemine.rnet.missouri.edu/maizemine/service",
                root_class="Gene",
                views=["Gene.primaryIdentifier", "Gene.symbol"],
                size=1_000,
                workflow="elt",
                resource_profile="tor_low_mem",
                max_inflight_bytes_estimate=2_048,
                temp_dir="/tmp",
                temp_dir_min_free_bytes=0,
                parquet_path="/tmp/mock.parquet",
            )

    kwargs = _FakeService.last_query.parquet_calls[0][1]
    options = kwargs["parallel_options"]
    assert options.max_workers == 2
    assert options.prefetch == 2
    assert options.inflight_limit == 2
    assert options.max_inflight_bytes_estimate == 2_048
    assert str(kwargs["temp_dir"]) == "/tmp"
    assert kwargs["temp_dir_min_free_bytes"] == 0


@patch("intermine314.export.fetch.Service", _FakeService)
def test_fetch_rejects_removed_etl_workflow():
    with pytest.raises(ValueError, match="workflow must be 'elt'"):
        fetch_from_mine(
            mine_url="https://mines.legumeinfo.org/legumemine/service",
            root_class="Gene",
            views=["Gene.primaryIdentifier", "Gene.name"],
            size=1_000,
            workflow="etl",
            parquet_path="/tmp/mock.parquet",
        )


@patch("intermine314.export.fetch.Service", _FakeService)
def test_fetch_managed_result_closes_duckdb_connection():
    fake_duckdb = _FakeDuckDBModule()
    with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
        with patch("intermine314.export.fetch.resolve_production_plan", return_value=_plan("elt", workers=8)):
            result = fetch_from_mine(
                mine_url="https://maizemine.rnet.missouri.edu/maizemine/service",
                root_class="Gene",
                views=["Gene.primaryIdentifier", "Gene.symbol"],
                size=1_000,
                workflow="elt",
                parquet_path="/tmp/mock.parquet",
                managed=True,
            )
            managed_connection = result["duckdb_connection"]
            with managed_connection as con:
                assert con.close_calls == 0
            assert con.close_calls == 1


def test_fetch_failure_before_duckdb_connect_does_not_open_connection():
    class _FailingQuery(_FakeQuery):
        def to_parquet(self, path, **kwargs):
            _ = (path, kwargs)
            raise RuntimeError("parquet write failed")

    class _FailingService:
        def __init__(self, _mine_url):
            pass

        def new_query(self, _root_class):
            return _FailingQuery()

    fake_duckdb = _FakeDuckDBModule()
    with patch("intermine314.export.fetch.Service", _FailingService):
        with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
            with patch("intermine314.export.fetch.resolve_production_plan", return_value=_plan("elt", workers=8)):
                with pytest.raises(RuntimeError, match="parquet write failed"):
                    fetch_from_mine(
                        mine_url="https://maizemine.rnet.missouri.edu/maizemine/service",
                        root_class="Gene",
                        views=["Gene.primaryIdentifier", "Gene.symbol"],
                        size=1_000,
                        workflow="elt",
                        parquet_path="/tmp/mock.parquet",
                    )
    assert len(fake_duckdb.connections) == 0


def test_fetch_failure_after_duckdb_connect_closes_connection():
    class _FailingDuckDBConnection(_FakeDuckDBConnection):
        def execute(self, query, params=None):
            self.sql.append((query, params))
            raise RuntimeError("duckdb view creation failed")

    class _FailingDuckDBModule(_FakeDuckDBModule):
        def connect(self, database=":memory:"):
            con = _FailingDuckDBConnection()
            con.database = database
            self.connections.append(con)
            return con

    fake_duckdb = _FailingDuckDBModule()
    with patch("intermine314.export.fetch.Service", _FakeService):
        with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
            with patch("intermine314.export.fetch.resolve_production_plan", return_value=_plan("elt", workers=8)):
                with pytest.raises(RuntimeError, match="duckdb view creation failed"):
                    fetch_from_mine(
                        mine_url="https://maizemine.rnet.missouri.edu/maizemine/service",
                        root_class="Gene",
                        views=["Gene.primaryIdentifier", "Gene.symbol"],
                        size=1_000,
                        workflow="elt",
                        parquet_path="/tmp/mock.parquet",
                    )
    assert len(fake_duckdb.connections) == 1
    assert fake_duckdb.connections[0].close_calls == 1
