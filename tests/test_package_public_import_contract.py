from __future__ import annotations

import intermine314
from intermine314 import export as export_pkg
from intermine314 import query as query_pkg
from intermine314.query import builder as query_builder


def test_package_root_public_exports_are_explicit_and_minimal():
    expected = {
        "VERSION",
        "__version__",
        "fetch_from_mine",
    }
    assert set(intermine314.__all__) == expected


def test_query_package_public_exports_are_explicit_and_narrow():
    expected = {
        "Query",
        "Template",
        "ParallelOptions",
        "ParallelOptionsError",
        "QueryError",
        "ConstraintError",
        "QueryParseError",
        "ResultError",
    }
    assert set(query_pkg.__all__) == expected
    assert query_pkg.Query is query_builder.Query
    assert query_pkg.Template is query_builder.Template
    assert query_pkg.ParallelOptions is query_builder.ParallelOptions
    assert query_pkg.ParallelOptionsError is query_builder.ParallelOptionsError
    assert query_pkg.QueryError is query_builder.QueryError
    assert query_pkg.ConstraintError is query_builder.ConstraintError
    assert query_pkg.QueryParseError is query_builder.QueryParseError
    assert query_pkg.ResultError is query_builder.ResultError
    assert not hasattr(query_pkg, "SortOrder")
    assert "constraints" not in query_pkg.__all__


def test_export_package_public_exports_are_explicit_and_narrow():
    expected = {
        "to_dataframe",
        "to_duckdb",
    }
    assert set(export_pkg.__all__) == expected
    assert callable(export_pkg.to_dataframe)
    assert callable(export_pkg.to_duckdb)
    assert not hasattr(export_pkg, "write_single_parquet_from_parts")
    assert not hasattr(export_pkg, "Path")
    assert not hasattr(export_pkg, "validate_parquet_compression")
