from __future__ import annotations

from importlib import import_module
from typing import Any

_SYMBOL_TO_MODULE = {
    "Class": "intermine314.model",
    "Column": "intermine314.model",
    "ConstraintNode": "intermine314.model",
    "Join": "intermine314.query.pathfeatures",
    "Model": "intermine314.model",
    "PathDescription": "intermine314.query.pathfeatures",
    "ReadableException": "intermine314.util",
    "Reference": "intermine314.model",
    "SortOrder": "intermine314.query.pathfeatures",
    "SortOrderList": "intermine314.query.pathfeatures",
    "openAnything": "intermine314.util",
    "Query": "intermine314.query.builder",
    "Template": "intermine314.query.template",
    "ParallelOptions": "intermine314.query.builder",
    "QueryError": "intermine314.query.builder",
    "ConstraintError": "intermine314.query.builder",
    "QueryParseError": "intermine314.query.builder",
    "ResultError": "intermine314.query.builder",
}


_MODULE_EXPORTS = {'constraints': 'intermine314.query.constraints'}

__all__ = [*_SYMBOL_TO_MODULE, *_MODULE_EXPORTS]

def __getattr__(name: str) -> Any:
    if name in _MODULE_EXPORTS:
        return import_module(_MODULE_EXPORTS[name])
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
