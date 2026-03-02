from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Query",
    "Template",
    "ParallelOptions",
]

_SYMBOL_TO_MODULE = {
    "Query": "intermine314.query.builder",
    "Template": "intermine314.query.builder",
    "ParallelOptions": "intermine314.query.builder",
}


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
