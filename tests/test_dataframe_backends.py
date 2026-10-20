"""Profile-specific presentation over one typed Parquet/DuckDB pipeline."""

import pytest

from intermine314.util import deps


def test_optional_pandas_loader_is_lazy_and_missing_error_names_extra(monkeypatch):
    imports = []

    def missing(name):
        imports.append(name)
        raise ImportError(name)

    monkeypatch.setattr(deps, 'import_module', missing)
    deps._optional_module.cache_clear()
    try:
        with pytest.raises(ImportError, match=r'intermine314\[pandas\]'):
            deps.require_pandas('Query.dataframe()')
        assert imports == ['pandas']
    finally:
        deps._optional_module.cache_clear()
