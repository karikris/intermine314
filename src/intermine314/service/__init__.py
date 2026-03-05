from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Service",
    "Registry",
    "tor_proxy_url",
    "tor_session",
    "tor_service",
    "tor_registry",
]

_SYMBOL_TO_MODULE = {
    "Service": "intermine314.service.service",
    "Registry": "intermine314.service.service",
    "tor_proxy_url": "intermine314.service.tor",
    "tor_session": "intermine314.service.tor",
    "tor_service": "intermine314.service.tor",
    "tor_registry": "intermine314.service.tor",
}


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
