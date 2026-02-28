import unittest

from intermine314.export import to_dataframe as export_to_dataframe
from intermine314.export import to_duckdb as export_to_duckdb


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


if __name__ == "__main__":
    unittest.main()
