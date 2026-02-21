import unittest
from unittest.mock import patch

from intermine314.export import to_dataframe as export_to_dataframe
from intermine314.export import to_duckdb as export_to_duckdb
from intermine314.export import duckdb as legacy_duckdb
from intermine314.export import polars_frame as legacy_polars_frame


class _FakeQuery:
    def __init__(self):
        self.calls = []

    def to_duckdb(self, **kwargs):
        self.calls.append(("to_duckdb", kwargs))
        return {"api": "to_duckdb", "kwargs": kwargs}

    def dataframe(self, **kwargs):
        self.calls.append(("dataframe", kwargs))
        return {"api": "dataframe", "kwargs": kwargs}


class TestExportLegacyWrappers(unittest.TestCase):
    def test_export_module_helpers_delegate_without_legacy_module_dependency(self):
        query = _FakeQuery()

        duckdb_result = export_to_duckdb(query, table="t")
        dataframe_result = export_to_dataframe(query, batch_size=10)

        self.assertEqual(duckdb_result["api"], "to_duckdb")
        self.assertEqual(dataframe_result["api"], "dataframe")
        self.assertEqual(
            query.calls,
            [
                ("to_duckdb", {"table": "t"}),
                ("dataframe", {"batch_size": 10}),
            ],
        )

    def test_legacy_duckdb_wrapper_logs_once_and_delegates(self):
        query = _FakeQuery()
        with patch("intermine314.export.duckdb._LEGACY_WARNING_EMITTED", False):
            with patch("intermine314.export.duckdb._LEGACY_LOG.warning") as warning_mock:
                one = legacy_duckdb.to_duckdb(query, table="legacy_a")
                two = legacy_duckdb.to_duckdb(query, table="legacy_b")

        self.assertEqual(one["api"], "to_duckdb")
        self.assertEqual(two["api"], "to_duckdb")
        warning_mock.assert_called_once()
        self.assertIn("legacy shim", warning_mock.call_args[0][0])

    def test_legacy_polars_wrapper_logs_once_and_delegates(self):
        query = _FakeQuery()
        with patch("intermine314.export.polars_frame._LEGACY_WARNING_EMITTED", False):
            with patch("intermine314.export.polars_frame._LEGACY_LOG.warning") as warning_mock:
                one = legacy_polars_frame.to_dataframe(query, batch_size=1)
                two = legacy_polars_frame.to_dataframe(query, batch_size=2)

        self.assertEqual(one["api"], "dataframe")
        self.assertEqual(two["api"], "dataframe")
        warning_mock.assert_called_once()
        self.assertIn("legacy shim", warning_mock.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
