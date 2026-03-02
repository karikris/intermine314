from __future__ import annotations

import intermine314
from intermine314 import export as export_pkg
from intermine314 import query as query_pkg
from intermine314.export import fetch as export_fetch
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
    }
    assert set(query_pkg.__all__) == expected
    assert query_pkg.Query is query_builder.Query
    assert query_pkg.Template is query_builder.Template
    assert query_pkg.ParallelOptions is query_builder.ParallelOptions
    assert not hasattr(query_pkg, "ParallelOptionsError")
    assert not hasattr(query_pkg, "QueryError")
    assert not hasattr(query_pkg, "ConstraintError")
    assert not hasattr(query_pkg, "QueryParseError")
    assert not hasattr(query_pkg, "ResultError")
    assert not hasattr(query_pkg, "SortOrder")
    assert "constraints" not in query_pkg.__all__


def test_export_package_public_exports_are_explicit_and_narrow():
    expected = {
        "fetch_from_mine",
    }
    assert set(export_pkg.__all__) == expected
    assert export_pkg.fetch_from_mine is export_fetch.fetch_from_mine
    assert not hasattr(export_pkg, "write_single_parquet_from_parts")
    assert not hasattr(export_pkg, "Path")
    assert not hasattr(export_pkg, "validate_parquet_compression")
    assert not hasattr(export_pkg, "to_dataframe")
    assert not hasattr(export_pkg, "to_duckdb")
