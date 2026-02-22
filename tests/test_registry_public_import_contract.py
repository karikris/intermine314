from __future__ import annotations

import warnings

from intermine314 import registry
from intermine314.registry import api as registry_api
from intermine314.registry import mines as registry_mines
from intermine314.service import Registry as ServiceRegistry


def test_registry_package_public_exports_are_explicit_and_canonical():
    expected = {
        "NO_SUCH_MINE",
        "RegistryAPIError",
        "RegistryLookupError",
        "RegistryQueryError",
        "get_version",
        "get_info",
        "get_data",
        "get_mines",
        "legacy_registry_api_metrics",
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
    }
    assert set(registry.__all__) == expected

    assert registry.RegistryAPIError is registry_api.RegistryAPIError
    assert registry.RegistryLookupError is registry_api.RegistryLookupError
    assert registry.RegistryQueryError is registry_api.RegistryQueryError
    assert registry.get_version is registry_api.get_version
    assert registry.get_info is registry_api.get_info
    assert registry.get_data is registry_api.get_data
    assert registry.get_mines is registry_api.get_mines
    assert registry.legacy_registry_api_metrics is registry_api.legacy_registry_api_metrics

    assert registry.NO_SUCH_MINE is registry_api.NO_SUCH_MINE
    assert registry.getVersion is registry_api.getVersion
    assert registry.getInfo is registry_api.getInfo
    assert registry.getData is registry_api.getData
    assert registry.getMines is registry_api.getMines

    assert registry.DEFAULT_BENCHMARK_SMALL_PROFILE is registry_mines.DEFAULT_BENCHMARK_SMALL_PROFILE
    assert registry.DEFAULT_BENCHMARK_LARGE_PROFILE is registry_mines.DEFAULT_BENCHMARK_LARGE_PROFILE
    assert registry.DEFAULT_BENCHMARK_FALLBACK_PROFILE is registry_mines.DEFAULT_BENCHMARK_FALLBACK_PROFILE
    assert registry.resolve_production_plan is registry_mines.resolve_production_plan
    assert registry.resolve_preferred_workers is registry_mines.resolve_preferred_workers
    assert registry.resolve_benchmark_plan is registry_mines.resolve_benchmark_plan
    assert registry.resolve_named_benchmark_profile is registry_mines.resolve_named_benchmark_profile
    assert registry.resolve_execution_plan is registry_mines.resolve_execution_plan
    assert registry.resolve_benchmark_phase_plan is registry_mines.resolve_benchmark_phase_plan
    assert registry.registry_preferences_metrics is registry_mines.registry_preferences_metrics

    assert "Registry" not in registry.__all__
    assert "Service" not in registry.__all__
    assert "DEFAULT_BENCHMARK_PROFILES" not in registry.__all__


def test_registry_accidental_export_alias_warns_once_and_delegates():
    registry._DEPRECATED_EXPORTS_EMITTED.clear()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        first = registry.Registry
        second = registry.Registry

    assert first is ServiceRegistry
    assert second is ServiceRegistry

    messages = [str(item.message) for item in caught]
    matching = [msg for msg in messages if "deprecated compatibility export" in msg and "registry.Registry" in msg]
    assert len(matching) == 1
