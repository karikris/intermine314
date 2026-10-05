"""Lazy original service import paths with legacy compatibility defaults."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SYMBOL_TO_MODULE = {
    "Attribute": "intermine314.model",
    "Collection": "intermine314.model",
    "Column": "intermine314.model",
    "InterMineURLOpener": "intermine314.service.session",
    "ListManager": "intermine314.lists.listmanager",
    "Model": "intermine314.model",
    "Query": "intermine314.query.builder",
    "Reference": "intermine314.model",
    "ResultIterator": "intermine314.service.session",
    "ServiceError": "intermine314.service.errors",
    "Template": "intermine314.query.template",
    "WebserviceError": "intermine314.service.errors",
    "requires_version": "intermine314.decorators",
    "Service": "intermine314.service.legacy",
    "Registry": "intermine314.service.legacy",
    "ensure_str": "intermine314.service.service",
}

_MODULE_EXPORTS = {'idresolution': 'intermine314.idresolution'}

__all__ = [*_SYMBOL_TO_MODULE, *_MODULE_EXPORTS]


def __getattr__(name: str) -> Any:
    if name in _MODULE_EXPORTS:
        return import_module(_MODULE_EXPORTS[name])
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
