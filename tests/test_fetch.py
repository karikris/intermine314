import unittest
from unittest.mock import patch

from intermine314.export.fetch import fetch_from_mine


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

    def execute(self, query, params=None):
        self.sql.append((query, params))
        return self

    def register(self, name, value):
        self.registered[name] = value

    def unregister(self, name):
        self.registered.pop(name, None)


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


class TestFetchFromMine(unittest.TestCase):
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
        self.assertIsNotNone(query)
        self.assertEqual(query.views, ["Gene.primaryIdentifier", "Gene.symbol"])
        self.assertEqual(query.joins, [("Gene.organism", "OUTER")])
        self.assertEqual(len(query.parquet_calls), 1)
        _, kwargs = query.parquet_calls[0]
        self.assertEqual(kwargs["parallel_options"].max_workers, 8)
        self.assertTrue(kwargs["single_file"])
        self.assertEqual(kwargs["parallel_options"].profile, "large_query")
        self.assertEqual(result["production_plan"]["name"], "elt_server_limited_w8")
        self.assertTrue(any("read_parquet" in sql for sql, _ in result["duckdb_connection"].sql))

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
        self.assertIsNotNone(query)
        self.assertEqual(len(query.parquet_calls), 1)
        _, kwargs = query.parquet_calls[0]
        self.assertEqual(kwargs["parallel_options"].max_workers, 4)
        self.assertFalse(kwargs["single_file"])
        self.assertEqual(result["production_plan"]["name"], "etl_default_w4")
        self.assertIsNone(result["dataframe"])
        self.assertIn('CREATE OR REPLACE TABLE "results_table"', result["duckdb_connection"].sql[0][0])
        self.assertIn("read_parquet", result["duckdb_connection"].sql[0][0])

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
                with self.assertRaises(ValueError) as cm:
                    fetch_from_mine(
                        mine_url="https://mines.legumeinfo.org/legumemine/service",
                        root_class="Gene",
                        views=["Gene.primaryIdentifier", "Gene.name"],
                        size=None,
                        workflow="etl",
                    )

        self.assertIn("allow_large_etl=True", str(cm.exception))

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
        self.assertIsNotNone(query)
        self.assertEqual(len(query.parquet_calls), 1)
        self.assertEqual(result["production_plan"]["name"], "etl_default_w4")

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
        self.assertIsNotNone(query)
        self.assertEqual(len(query.parquet_calls), 1)
        self.assertEqual(result["production_plan"]["name"], "etl_default_w4")

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
                with self.assertRaises(ValueError) as cm:
                    fetch_from_mine(
                        mine_url="https://mines.legumeinfo.org/legumemine/service",
                        root_class="Gene",
                        views=["Gene.primaryIdentifier", "Gene.name"],
                        size=250_000,
                        workflow="etl",
                        etl_guardrail_rows=100_000,
                    )

        self.assertIn("allow_large_etl=True", str(cm.exception))

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
        self.assertIsNotNone(query)
        self.assertEqual(len(query.parquet_calls), 1)
        self.assertEqual(result["production_plan"]["name"], "etl_default_w4")

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
        self.assertIsNotNone(query)
        self.assertEqual(len(query.parquet_calls), 1)
        self.assertEqual(result["production_plan"]["name"], "etl_default_w4")
        self.assertIn("dataframe", result)
        self.assertEqual(result["dataframe"]["rechunk"], True)
        self.assertTrue(fake_polars.scan_calls)
        self.assertEqual(fake_polars.collect_calls[0]["rechunk"], True)


if __name__ == "__main__":
    unittest.main()
