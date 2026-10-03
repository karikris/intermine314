"""Lazy facades for server lists and list management."""

from __future__ import annotations

from importlib import import_module

__all__ = ["List", "ListManager", "ListServiceError"]

_SYMBOL_TO_MODULE = {
    "List": "intermine314.lists.list",
    "ListManager": "intermine314.lists.listmanager",
    "ListServiceError": "intermine314.lists.listmanager",
}


def __getattr__(name):
    module = _SYMBOL_TO_MODULE.get(name)
    if module is None:
        raise AttributeError(name)
    return getattr(import_module(module), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
