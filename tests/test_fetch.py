from unittest.mock import patch

from intermine314.export.fetch import fetch_from_mine
import pytest


class _FakeQuery:
    def __init__(self):
        self.views = []
        self.joins = []
        self.parquet_calls = []
        self.dataframe_calls = []

    def add_view(self, *views):
        self.views.extend(views)

    def add_join(self, join_path, style):
        self.joins.append((join_path, style))

    def to_parquet(self, path, **kwargs):
        self.parquet_calls.append((path, kwargs))
        return str(path)

    def dataframe(self, **kwargs):
        self.dataframe_calls.append(kwargs)
        return {"mock": "polars_dataframe"}


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
        self.registered = {}
        self.close_calls = 0

    def execute(self, query, params=None):
        self.sql.append((query, params))
        return self

    def register(self, name, value):
        self.registered[name] = value

    def unregister(self, name):
        self.registered.pop(name, None)

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


class _FakeParquetScan:
    def __init__(self, source, collect_calls):
        self.source = source
        self._collect_calls = collect_calls

    def collect(self, *, rechunk=False):
        self._collect_calls.append({"source": self.source, "rechunk": bool(rechunk)})
        return {"mock": "polars_dataframe", "source": self.source, "rechunk": bool(rechunk)}


class _FakePolarsModule:
    def __init__(self):
        self.scan_calls = []
        self.collect_calls = []

    def scan_parquet(self, source):
        self.scan_calls.append(str(source))
        return _FakeParquetScan(str(source), self.collect_calls)


class TestFetchFromMine:
    @patch("intermine314.export.fetch.Service", _FakeService)
    def test_elt_workflow_uses_parquet_and_duckdb(self):
        fake_duckdb = _FakeDuckDBModule()
        plan = {
            "name": "elt_server_limited_w8",
            "workflow": "elt",
            "workers": 8,
            "pipeline": "parquet_duckdb",
            "parallel_profile": "large_query",
            "ordered": "unordered",
            "large_query_mode": True,
            "prefetch": None,
            "inflight_limit": None,
        }
        with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
            with patch("intermine314.export.fetch.resolve_production_plan", return_value=plan):
                result = fetch_from_mine(
                    mine_url="https://maizemine.rnet.missouri.edu/maizemine/service",
                    root_class="Gene",
                    views=["Gene.primaryIdentifier", "Gene.symbol"],
                    joins=["Gene.organism"],
                    size=1000,
                    workflow="elt",
                    parquet_path="/tmp/mock.parquet",
                )

        query = _FakeService.last_query
        assert query is not None
        assert query.views == ["Gene.primaryIdentifier", "Gene.symbol"]
        assert query.joins == [("Gene.organism", "OUTER")]
        assert len(query.parquet_calls) == 1
        _, kwargs = query.parquet_calls[0]
        assert kwargs["parallel_options"].max_workers == 8
        assert kwargs["single_file"]
        assert kwargs["parallel_options"].profile == "large_query"
        assert result["production_plan"]["name"] == "elt_server_limited_w8"
        assert any("read_parquet" in sql for sql, _ in result["duckdb_connection"].sql)

    @patch("intermine314.export.fetch.Service", _FakeService)
    def test_etl_workflow_uses_parquet_then_duckdb_table(self):
        fake_duckdb = _FakeDuckDBModule()
        plan = {
            "name": "etl_default_w4",
            "workflow": "etl",
            "workers": 4,
            "pipeline": "polars_duckdb",
            "parallel_profile": "large_query",
            "ordered": "unordered",
            "large_query_mode": True,
            "prefetch": None,
            "inflight_limit": None,
        }
        with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
            with patch("intermine314.export.fetch.resolve_production_plan", return_value=plan):
                result = fetch_from_mine(
                    mine_url="https://mines.legumeinfo.org/legumemine/service",
                    root_class="Gene",
                    views=["Gene.primaryIdentifier", "Gene.name"],
                    size=500,
                    workflow="etl",
                    duckdb_table="results_table",
                )

        query = _FakeService.last_query
        assert query is not None
        assert len(query.parquet_calls) == 1
        _, kwargs = query.parquet_calls[0]
        assert kwargs["parallel_options"].max_workers == 4
        assert not kwargs["single_file"]
        assert result["production_plan"]["name"] == "etl_default_w4"
        assert result["dataframe"] is None
        assert 'CREATE OR REPLACE TABLE "results_table"' in result["duckdb_connection"].sql[0][0]
        assert "read_parquet" in result["duckdb_connection"].sql[0][0]

    @patch("intermine314.export.fetch.Service", _FakeService)
    def test_elt_passes_max_inflight_bytes_and_temp_dir_controls(self):
        fake_duckdb = _FakeDuckDBModule()
        plan = {
            "name": "elt_server_limited_w8",
            "workflow": "elt",
            "workers": 8,
            "pipeline": "parquet_duckdb",
            "parallel_profile": "large_query",
            "ordered": "unordered",
            "large_query_mode": True,
            "prefetch": None,
            "inflight_limit": None,
        }
        with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
            with patch("intermine314.export.fetch.resolve_production_plan", return_value=plan):
                result = fetch_from_mine(
                    mine_url="https://maizemine.rnet.missouri.edu/maizemine/service",
                    root_class="Gene",
                    views=["Gene.primaryIdentifier", "Gene.symbol"],
                    joins=["Gene.organism"],
                    size=1000,
                    workflow="elt",
                    max_inflight_bytes_estimate=2048,
                    temp_dir="/tmp",
                    temp_dir_min_free_bytes=0,
                    parquet_path="/tmp/mock.parquet",
                )

        query = _FakeService.last_query
        assert query is not None
        assert len(query.parquet_calls) == 1
        _, kwargs = query.parquet_calls[0]
        assert kwargs["parallel_options"].max_inflight_bytes_estimate == 2048
        assert str(kwargs["temp_dir"]) == "/tmp"
        assert kwargs["temp_dir_min_free_bytes"] == 0
        assert result["resource_profile"] == "default"

    @patch("intermine314.export.fetch.Service", _FakeService)
    def test_elt_uses_resource_profile_defaults(self):
        fake_duckdb = _FakeDuckDBModule()
        plan = {
            "name": "elt_server_limited_w8",
            "workflow": "elt",
            "workers": 8,
            "pipeline": "parquet_duckdb",
            "parallel_profile": "large_query",
            "ordered": "unordered",
            "large_query_mode": True,
            "prefetch": None,
            "inflight_limit": None,
        }
        with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
            with patch("intermine314.export.fetch.resolve_production_plan", return_value=plan):
                result = fetch_from_mine(
                    mine_url="https://maizemine.rnet.missouri.edu/maizemine/service",
                    root_class="Gene",
                    views=["Gene.primaryIdentifier", "Gene.symbol"],
                    joins=["Gene.organism"],
                    size=1000,
                    workflow="elt",
                    resource_profile="tor_low_mem",
                    parquet_path="/tmp/mock.parquet",
                )

        query = _FakeService.last_query
        assert query is not None
        assert len(query.parquet_calls) == 1
        _, kwargs = query.parquet_calls[0]
        options = kwargs["parallel_options"]
        assert options.max_workers == 2
        assert options.prefetch == 2
        assert options.inflight_limit == 2
        assert options.max_inflight_bytes_estimate == 64 * 1024 * 1024
        assert options.ordered == "window"
        assert result["resource_profile"] == "tor_low_mem"

    @patch("intermine314.export.fetch.Service", _FakeService)
    def test_elt_uses_resource_profile_from_production_plan_when_not_explicit(self):
        fake_duckdb = _FakeDuckDBModule()
        plan = {
            "name": "elt_server_limited_w8",
            "workflow": "elt",
            "workers": 8,
            "pipeline": "parquet_duckdb",
            "parallel_profile": "large_query",
            "ordered": "unordered",
            "large_query_mode": True,
            "resource_profile": "tor_low_mem",
            # These legacy plan knobs should be ignored in favor of resource profile defaults.
            "prefetch": 99,
            "inflight_limit": 99,
            "max_inflight_bytes_estimate": 999_999_999,
        }
        with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
            with patch("intermine314.export.fetch.resolve_production_plan", return_value=plan):
                result = fetch_from_mine(
                    mine_url="https://maizemine.rnet.missouri.edu/maizemine/service",
                    root_class="Gene",
                    views=["Gene.primaryIdentifier", "Gene.symbol"],
                    joins=["Gene.organism"],
                    size=1000,
                    workflow="elt",
                    parquet_path="/tmp/mock.parquet",
                )

        query = _FakeService.last_query
        assert query is not None
        assert len(query.parquet_calls) == 1
        _, kwargs = query.parquet_calls[0]
        options = kwargs["parallel_options"]
        assert options.max_workers == 2
        assert options.prefetch == 2
        assert options.inflight_limit == 2
        assert options.max_inflight_bytes_estimate == 64 * 1024 * 1024
        assert result["resource_profile"] == "tor_low_mem"

    @patch("intermine314.export.fetch.Service", _FakeService)
    def test_etl_unknown_size_requires_explicit_opt_in(self):
        fake_duckdb = _FakeDuckDBModule()
        plan = {
            "name": "etl_default_w4",
            "workflow": "etl",
            "workers": 4,
            "pipeline": "polars_duckdb",
            "parallel_profile": "large_query",
            "ordered": "unordered",
            "large_query_mode": True,
            "prefetch": None,
            "inflight_limit": None,
        }
        with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
            with patch("intermine314.export.fetch.resolve_production_plan", return_value=plan):
                with pytest.raises(ValueError) as cm:
                    fetch_from_mine(
                        mine_url="https://mines.legumeinfo.org/legumemine/service",
                        root_class="Gene",
                        views=["Gene.primaryIdentifier", "Gene.name"],
                        size=None,
                        workflow="etl",
                    )

        assert "allow_large_etl=True" in str(cm.value)

    @patch("intermine314.export.fetch.Service", _FakeService)
    def test_etl_unknown_size_can_be_forced(self):
        fake_duckdb = _FakeDuckDBModule()
        plan = {
            "name": "etl_default_w4",
            "workflow": "etl",
            "workers": 4,
            "pipeline": "polars_duckdb",
            "parallel_profile": "large_query",
            "ordered": "unordered",
            "large_query_mode": True,
            "prefetch": None,
            "inflight_limit": None,
        }
        with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
            with patch("intermine314.export.fetch.resolve_production_plan", return_value=plan):
                result = fetch_from_mine(
                    mine_url="https://mines.legumeinfo.org/legumemine/service",
                    root_class="Gene",
                    views=["Gene.primaryIdentifier", "Gene.name"],
                    size=None,
                    workflow="etl",
                    allow_large_etl=True,
                )

        query = _FakeService.last_query
        assert query is not None
        assert len(query.parquet_calls) == 1
        assert result["production_plan"]["name"] == "etl_default_w4"

    @patch("intermine314.export.fetch.Service", _FakeService)
    def test_etl_temp_dir_constraints_fail_fast(self):
        fake_duckdb = _FakeDuckDBModule()
        plan = {
            "name": "etl_default_w4",
            "workflow": "etl",
            "workers": 4,
            "pipeline": "polars_duckdb",
            "parallel_profile": "large_query",
            "ordered": "unordered",
            "large_query_mode": True,
            "prefetch": None,
            "inflight_limit": None,
        }
        with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
            with patch("intermine314.export.fetch.resolve_production_plan", return_value=plan):
                with patch(
                    "intermine314.export.fetch.validate_temp_dir_constraints",
                    side_effect=ValueError("no space"),
                ):
                    with pytest.raises(ValueError):
                        fetch_from_mine(
                            mine_url="https://mines.legumeinfo.org/legumemine/service",
                            root_class="Gene",
                            views=["Gene.primaryIdentifier", "Gene.name"],
                            size=500,
                            workflow="etl",
                            temp_dir="/tmp",
                            temp_dir_min_free_bytes=10_000_000_000,
                        )

    @patch("intermine314.export.fetch.Service", _FakeService)
    def test_etl_unknown_size_can_be_forced_with_force_etl_alias(self):
        fake_duckdb = _FakeDuckDBModule()
        plan = {
            "name": "etl_default_w4",
            "workflow": "etl",
            "workers": 4,
            "pipeline": "polars_duckdb",
            "parallel_profile": "large_query",
            "ordered": "unordered",
            "large_query_mode": True,
            "prefetch": None,
            "inflight_limit": None,
        }
        with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
            with patch("intermine314.export.fetch.resolve_production_plan", return_value=plan):
                result = fetch_from_mine(
                    mine_url="https://mines.legumeinfo.org/legumemine/service",
                    root_class="Gene",
                    views=["Gene.primaryIdentifier", "Gene.name"],
                    size=None,
                    workflow="etl",
                    force_etl=True,
                )

        query = _FakeService.last_query
        assert query is not None
        assert len(query.parquet_calls) == 1
        assert result["production_plan"]["name"] == "etl_default_w4"

    @patch("intermine314.export.fetch.Service", _FakeService)
    def test_etl_large_size_requires_explicit_opt_in(self):
        fake_duckdb = _FakeDuckDBModule()
        plan = {
            "name": "etl_default_w4",
            "workflow": "etl",
            "workers": 4,
            "pipeline": "polars_duckdb",
            "parallel_profile": "large_query",
            "ordered": "unordered",
            "large_query_mode": True,
            "prefetch": None,
            "inflight_limit": None,
        }
        with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
            with patch("intermine314.export.fetch.resolve_production_plan", return_value=plan):
                with pytest.raises(ValueError) as cm:
                    fetch_from_mine(
                        mine_url="https://mines.legumeinfo.org/legumemine/service",
                        root_class="Gene",
                        views=["Gene.primaryIdentifier", "Gene.name"],
                        size=250_000,
                        workflow="etl",
                        etl_guardrail_rows=100_000,
                    )

        assert "allow_large_etl=True" in str(cm.value)

    @patch("intermine314.export.fetch.Service", _FakeService)
    def test_etl_large_size_can_be_forced(self):
        fake_duckdb = _FakeDuckDBModule()
        plan = {
            "name": "etl_default_w4",
            "workflow": "etl",
            "workers": 4,
            "pipeline": "polars_duckdb",
            "parallel_profile": "large_query",
            "ordered": "unordered",
            "large_query_mode": True,
            "prefetch": None,
            "inflight_limit": None,
        }
        with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
            with patch("intermine314.export.fetch.resolve_production_plan", return_value=plan):
                result = fetch_from_mine(
                    mine_url="https://mines.legumeinfo.org/legumemine/service",
                    root_class="Gene",
                    views=["Gene.primaryIdentifier", "Gene.name"],
                    size=250_000,
                    workflow="etl",
                    etl_guardrail_rows=100_000,
                    allow_large_etl=True,
                )

        query = _FakeService.last_query
        assert query is not None
        assert len(query.parquet_calls) == 1
        assert result["production_plan"]["name"] == "etl_default_w4"

    @patch("intermine314.export.fetch.Service", _FakeService)
    def test_etl_can_materialize_dataframe_from_parquet_on_demand(self):
        fake_duckdb = _FakeDuckDBModule()
        fake_polars = _FakePolarsModule()
        plan = {
            "name": "etl_default_w4",
            "workflow": "etl",
            "workers": 4,
            "pipeline": "polars_duckdb",
            "parallel_profile": "large_query",
            "ordered": "unordered",
            "large_query_mode": True,
            "prefetch": None,
            "inflight_limit": None,
        }
        with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
            with patch("intermine314.export.fetch._require_polars", return_value=fake_polars):
                with patch("intermine314.export.fetch.resolve_production_plan", return_value=plan):
                    result = fetch_from_mine(
                        mine_url="https://mines.legumeinfo.org/legumemine/service",
                        root_class="Gene",
                        views=["Gene.primaryIdentifier", "Gene.name"],
                        size=500,
                        workflow="etl",
                        parquet_path="/tmp/etl-materialized.parquet",
                        etl_materialize_dataframe=True,
                        etl_final_rechunk=True,
                    )

        query = _FakeService.last_query
        assert query is not None
        assert len(query.parquet_calls) == 1
        assert result["production_plan"]["name"] == "etl_default_w4"
        assert "dataframe" in result
        assert result["dataframe"]["rechunk"] == True
        assert fake_polars.scan_calls
        assert fake_polars.collect_calls[0]["rechunk"] == True

    @patch("intermine314.export.fetch.Service", _FakeService)
    def test_managed_fetch_result_closes_duckdb_connection_on_context_exit(self):
        fake_duckdb = _FakeDuckDBModule()
        plan = {
            "name": "elt_server_limited_w8",
            "workflow": "elt",
            "workers": 8,
            "pipeline": "parquet_duckdb",
            "parallel_profile": "large_query",
            "ordered": "unordered",
            "large_query_mode": True,
            "prefetch": None,
            "inflight_limit": None,
        }
        with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
            with patch("intermine314.export.fetch.resolve_production_plan", return_value=plan):
                with fetch_from_mine(
                    mine_url="https://maizemine.rnet.missouri.edu/maizemine/service",
                    root_class="Gene",
                    views=["Gene.primaryIdentifier", "Gene.symbol"],
                    size=1000,
                    workflow="elt",
                    parquet_path="/tmp/mock.parquet",
                    managed=True,
                ) as result:
                    assert result["duckdb_connection"].close_calls == 0
                assert result["duckdb_connection"].close_calls == 1

    def test_fetch_closes_duckdb_connection_when_export_fails(self):
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
        plan = {
            "name": "elt_server_limited_w8",
            "workflow": "elt",
            "workers": 8,
            "pipeline": "parquet_duckdb",
            "parallel_profile": "large_query",
            "ordered": "unordered",
            "large_query_mode": True,
            "prefetch": None,
            "inflight_limit": None,
        }
        with patch("intermine314.export.fetch.Service", _FailingService):
            with patch("intermine314.export.fetch._require_duckdb", return_value=fake_duckdb):
                with patch("intermine314.export.fetch.resolve_production_plan", return_value=plan):
                    with pytest.raises(RuntimeError):
                        fetch_from_mine(
                            mine_url="https://maizemine.rnet.missouri.edu/maizemine/service",
                            root_class="Gene",
                            views=["Gene.primaryIdentifier", "Gene.symbol"],
                            size=1000,
                            workflow="elt",
                            parquet_path="/tmp/mock.parquet",
                        )

        assert len(fake_duckdb.connections) == 1
        assert fake_duckdb.connections[0].close_calls == 1
