from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "resolve_runtime_defaults_path",
    "load_config",
    "clear_config_cache",
    "QueryDefaults",
    "RUNTIME_DEFAULTS_SCHEMA_VERSION",
    "RegistryDefaults",
    "RuntimeDefaults",
    "ServiceDefaults",
    "StorageDefaults",
    "TransportDefaults",
    "parse_runtime_defaults",
    "get_runtime_defaults",
    "clear_runtime_defaults_cache",
    "runtime_defaults_load_telemetry",
    "reset_runtime_defaults_load_telemetry",
]

_SYMBOL_TO_MODULE = {
    "resolve_runtime_defaults_path": "intermine314.config.loader",
    "load_config": "intermine314.config.loader",
    "clear_config_cache": "intermine314.config.loader",
    "QueryDefaults": "intermine314.config.runtime_defaults",
    "RUNTIME_DEFAULTS_SCHEMA_VERSION": "intermine314.config.runtime_defaults",
    "RegistryDefaults": "intermine314.config.runtime_defaults",
    "RuntimeDefaults": "intermine314.config.runtime_defaults",
    "ServiceDefaults": "intermine314.config.runtime_defaults",
    "StorageDefaults": "intermine314.config.runtime_defaults",
    "TransportDefaults": "intermine314.config.runtime_defaults",
    "parse_runtime_defaults": "intermine314.config.runtime_defaults",
    "get_runtime_defaults": "intermine314.config.runtime_defaults",
    "clear_runtime_defaults_cache": "intermine314.config.runtime_defaults",
    "runtime_defaults_load_telemetry": "intermine314.config.runtime_defaults",
    "reset_runtime_defaults_load_telemetry": "intermine314.config.runtime_defaults",
}


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
