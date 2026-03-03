from __future__ import annotations

import intermine314
from intermine314 import export as export_pkg
from intermine314 import query as query_pkg
from intermine314.export import fetch as export_fetch
from intermine314.query import builder as query_builder


def test_package_root_public_exports_are_explicit_and_minimal():
    assert set(intermine314.__all__) == {"VERSION", "__version__", "fetch_from_mine"}
    assert set(query_pkg.__all__) == {"Query", "Template", "ParallelOptions"}
    assert query_pkg.Query is query_builder.Query
    assert query_pkg.Template is query_builder.Template
    assert query_pkg.ParallelOptions is query_builder.ParallelOptions
    assert set(export_pkg.__all__) == {"fetch_from_mine"}
    assert export_pkg.fetch_from_mine is export_fetch.fetch_from_mine
