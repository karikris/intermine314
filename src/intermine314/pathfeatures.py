"""Original import paths for the existing query path features."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SYMBOL_TO_MODULE = {
    "PathFeature": "intermine314.query.pathfeatures",
    "Join": "intermine314.query.pathfeatures",
    "SortOrder": "intermine314.query.pathfeatures",
    "SortOrderList": "intermine314.query.pathfeatures",
}

__all__ = list(_SYMBOL_TO_MODULE)


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
