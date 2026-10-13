"""Public export entrypoints."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["fetch_from_mine", "import_csv", "query_parquet", "query_parquet_reader", "ResourceProfile", "ResourceRuntime", "shutdown_resources", "ParquetOptions"]

_SYMBOL_TO_MODULE = {
    "fetch_from_mine": "intermine314.export.fetch",
    "import_csv": "intermine314.export.csv",
    "query_parquet": "intermine314.export.query",
    "query_parquet_reader": "intermine314.export.query",
    "ResourceProfile": "intermine314.export.resource_profile",
    "ResourceRuntime": "intermine314.export.runtime",
    "shutdown_resources": "intermine314.export.runtime",
    "ParquetOptions": "intermine314.export.layout",
}


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
