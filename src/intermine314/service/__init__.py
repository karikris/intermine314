from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Service",
    "Registry",
]

_SYMBOL_TO_MODULE = {
    "Service": "intermine314.service.service",
    "Registry": "intermine314.service.service",
}


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
