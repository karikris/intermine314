from __future__ import annotations

import warnings

import intermine314.registry.api as _registry_api
import intermine314.registry.mines as _registry_mines
from intermine314.registry.api import (
    LEGACY_REGISTRY_API_DEPRECATION_STARTED_IN,
    LEGACY_REGISTRY_API_REMOVAL_NOT_BEFORE,
    NO_SUCH_MINE,
    RegistryAPIError,
    RegistryLookupError,
    RegistryQueryError,
    get_data,
    get_info,
    get_mines,
    get_version,
    getData,
    getInfo,
    getMines,
    getVersion,
    legacy_registry_api_deprecation_status,
    legacy_registry_api_metrics,
)
from intermine314.registry.mines import (
    DEFAULT_BENCHMARK_FALLBACK_PROFILE,
    DEFAULT_BENCHMARK_LARGE_PROFILE,
    DEFAULT_BENCHMARK_SMALL_PROFILE,
    registry_preferences_metrics,
    resolve_benchmark_phase_plan,
    resolve_benchmark_plan,
    resolve_execution_plan,
    resolve_named_benchmark_profile,
    resolve_preferred_workers,
    resolve_production_plan,
)

__all__ = [
    "NO_SUCH_MINE",
    "RegistryAPIError",
    "RegistryLookupError",
    "RegistryQueryError",
    "get_version",
    "get_info",
    "get_data",
    "get_mines",
    "legacy_registry_api_metrics",
    "legacy_registry_api_deprecation_status",
    "LEGACY_REGISTRY_API_DEPRECATION_STARTED_IN",
    "LEGACY_REGISTRY_API_REMOVAL_NOT_BEFORE",
    "getVersion",
    "getInfo",
    "getData",
    "getMines",
    "DEFAULT_BENCHMARK_SMALL_PROFILE",
    "DEFAULT_BENCHMARK_LARGE_PROFILE",
    "DEFAULT_BENCHMARK_FALLBACK_PROFILE",
    "resolve_production_plan",
    "resolve_preferred_workers",
    "resolve_benchmark_plan",
    "resolve_named_benchmark_profile",
    "resolve_execution_plan",
    "resolve_benchmark_phase_plan",
    "registry_preferences_metrics",
]

_DEPRECATED_EXPORTS_BY_NAME = {}
_DEPRECATED_EXPORTS_EMITTED = set()

for _module in (_registry_api, _registry_mines):
    for _name in dir(_module):
        if _name.startswith("_") or _name in __all__:
            continue
        _DEPRECATED_EXPORTS_BY_NAME.setdefault(_name, _module)


def __getattr__(name: str):
    source_module = _DEPRECATED_EXPORTS_BY_NAME.get(name)
    if source_module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name not in _DEPRECATED_EXPORTS_EMITTED:
        _DEPRECATED_EXPORTS_EMITTED.add(name)
        warnings.warn(
            (
                f"intermine314.registry.{name} is a deprecated compatibility export from "
                f"{source_module.__name__}; import it from {source_module.__name__} directly."
            ),
            DeprecationWarning,
            stacklevel=2,
        )
    return getattr(source_module, name)


def __dir__():
    return sorted(set(__all__) | set(_DEPRECATED_EXPORTS_BY_NAME))
