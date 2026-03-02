from __future__ import annotations

import intermine314
import pytest
from intermine314 import export as export_pkg
from intermine314 import query as query_pkg
from intermine314.export import fetch as export_fetch
from intermine314.query import builder as query_builder
from intermine314.service import Service


def test_package_root_public_exports_are_explicit_and_minimal():
    assert set(intermine314.__all__) == {"VERSION", "__version__", "fetch_from_mine"}
    assert set(query_pkg.__all__) == {"Query", "Template", "ParallelOptions"}
    assert query_pkg.Query is query_builder.Query
    assert query_pkg.Template is query_builder.Template
    assert query_pkg.ParallelOptions is query_builder.ParallelOptions
    assert set(export_pkg.__all__) == {"fetch_from_mine"}
    assert export_pkg.fetch_from_mine is export_fetch.fetch_from_mine


def test_removed_service_and_query_aliases_require_canonical_apis():
    assert not hasattr(Service, "get_all_mines")

    service = Service.__new__(Service)
    with pytest.raises(AttributeError, match="Service\\.select\\(\\.\\.\\.\\) or Service\\.new_query\\(\\.\\.\\.\\)"):
        _ = service.query
    with pytest.raises(AttributeError, match="Registry\\(\\.\\.\\.\\)\\.info\\(mine_name\\)"):
        _ = service.get_mine_info

    query = query_builder.Query.__new__(query_builder.Query)
    with pytest.raises(AttributeError, match="Query\\.where\\(\\.\\.\\.\\)"):
        _ = query.filter
    with pytest.raises(AttributeError, match="Query\\.count\\(\\)"):
        _ = query.size
