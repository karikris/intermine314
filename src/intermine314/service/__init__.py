from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Service",
    "Registry",
    "InterMineURLOpener",
    "ResultIterator",
    "ReadableException",
    "ServiceError",
    "WebserviceError",
    "PROXY_URL_ENV_VAR",
    "build_session",
    "resolve_proxy_url",
    "is_tor_proxy_url",
    "DEFAULT_TOR_SOCKS_HOST",
    "DEFAULT_TOR_SOCKS_PORT",
    "tor_proxy_url",
    "tor_session",
    "tor_service",
    "tor_registry",
    "ensure_str",
]

_SYMBOL_TO_MODULE = {
    "Service": "intermine314.service.service",
    "Registry": "intermine314.service.service",
    "ensure_str": "intermine314.service.service",
    "InterMineURLOpener": "intermine314.service.session",
    "ResultIterator": "intermine314.service.session",
    "ReadableException": "intermine314.errors",
    "ServiceError": "intermine314.errors",
    "WebserviceError": "intermine314.errors",
    "PROXY_URL_ENV_VAR": "intermine314.service.transport",
    "build_session": "intermine314.service.transport",
    "resolve_proxy_url": "intermine314.service.transport",
    "is_tor_proxy_url": "intermine314.service.transport",
    "DEFAULT_TOR_SOCKS_HOST": "intermine314.service.tor",
    "DEFAULT_TOR_SOCKS_PORT": "intermine314.service.tor",
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
