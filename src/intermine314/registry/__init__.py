from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "NO_SUCH_MINE",
    "RegistryAPIError",
    "RegistryLookupError",
    "RegistryQueryError",
    "get_version",
    "get_info",
    "get_data",
    "get_mines",
    "resolve_execution_plan",
]

_SYMBOL_TO_MODULE = {
    "NO_SUCH_MINE": "intermine314.registry.api",
    "RegistryAPIError": "intermine314.registry.api",
    "RegistryLookupError": "intermine314.registry.api",
    "RegistryQueryError": "intermine314.registry.api",
    "get_version": "intermine314.registry.api",
    "get_info": "intermine314.registry.api",
    "get_data": "intermine314.registry.api",
    "get_mines": "intermine314.registry.api",
    "resolve_execution_plan": "intermine314.registry.mines",
}


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
