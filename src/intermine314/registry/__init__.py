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
    "DEFAULT_BENCHMARK_SMALL_PROFILE",
    "DEFAULT_BENCHMARK_LARGE_PROFILE",
    "DEFAULT_BENCHMARK_FALLBACK_PROFILE",
    "resolve_production_plan",
    "resolve_production_resource_profile",
    "resolve_preferred_workers",
    "resolve_benchmark_plan",
    "resolve_named_benchmark_profile",
    "resolve_execution_plan",
    "registry_preferences_metrics",
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
    "DEFAULT_BENCHMARK_SMALL_PROFILE": "intermine314.registry.mines",
    "DEFAULT_BENCHMARK_LARGE_PROFILE": "intermine314.registry.mines",
    "DEFAULT_BENCHMARK_FALLBACK_PROFILE": "intermine314.registry.mines",
    "resolve_production_plan": "intermine314.registry.mines",
    "resolve_production_resource_profile": "intermine314.registry.mines",
    "resolve_preferred_workers": "intermine314.registry.mines",
    "resolve_benchmark_plan": "intermine314.registry.mines",
    "resolve_named_benchmark_profile": "intermine314.registry.mines",
    "resolve_execution_plan": "intermine314.registry.mines",
    "registry_preferences_metrics": "intermine314.registry.mines",
}


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = import_module(module_name)
    return getattr(module, name)
