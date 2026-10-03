"""Original exception import paths preserve native exception identity."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SYMBOL_TO_MODULE = {
    "ServiceError": "intermine314.service.errors",
    "WebserviceError": "intermine314.service.errors",
}

__all__ = [*_SYMBOL_TO_MODULE, "UnimplementedError"]


class UnimplementedError(Exception):
    """An operation has no implementation."""


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
