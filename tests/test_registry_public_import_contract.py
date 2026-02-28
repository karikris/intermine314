from __future__ import annotations

from intermine314 import registry
from intermine314.registry import api as registry_api
from intermine314.registry import mines as registry_mines


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
        "DEFAULT_BENCHMARK_SMALL_PROFILE",
        "DEFAULT_BENCHMARK_LARGE_PROFILE",
        "DEFAULT_BENCHMARK_FALLBACK_PROFILE",
        "resolve_production_plan",
        "resolve_preferred_workers",
        "resolve_benchmark_plan",
        "resolve_named_benchmark_profile",
        "resolve_execution_plan",
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
    assert registry.NO_SUCH_MINE is registry_api.NO_SUCH_MINE

    assert registry.DEFAULT_BENCHMARK_SMALL_PROFILE is registry_mines.DEFAULT_BENCHMARK_SMALL_PROFILE
    assert registry.DEFAULT_BENCHMARK_LARGE_PROFILE is registry_mines.DEFAULT_BENCHMARK_LARGE_PROFILE
    assert registry.DEFAULT_BENCHMARK_FALLBACK_PROFILE is registry_mines.DEFAULT_BENCHMARK_FALLBACK_PROFILE
    assert registry.resolve_production_plan is registry_mines.resolve_production_plan
    assert registry.resolve_preferred_workers is registry_mines.resolve_preferred_workers
    assert registry.resolve_benchmark_plan is registry_mines.resolve_benchmark_plan
    assert registry.resolve_named_benchmark_profile is registry_mines.resolve_named_benchmark_profile
    assert registry.resolve_execution_plan is registry_mines.resolve_execution_plan
    assert registry.registry_preferences_metrics is registry_mines.registry_preferences_metrics


def test_registry_does_not_export_legacy_registry_helpers():
    assert not hasattr(registry, "getVersion")
    assert not hasattr(registry, "getInfo")
    assert not hasattr(registry, "getData")
    assert not hasattr(registry, "getMines")
    assert not hasattr(registry, "resolve_benchmark_phase_plan")
