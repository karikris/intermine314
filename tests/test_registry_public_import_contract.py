from __future__ import annotations

from intermine314 import registry
from intermine314.registry import mines as registry_mines


def test_registry_package_public_exports_are_explicit_and_do_not_include_legacy_helpers():
    assert set(registry.__all__) == {"resolve_execution_plan"}
    assert registry.resolve_execution_plan is registry_mines.resolve_execution_plan

    assert not hasattr(registry, "NO_SUCH_MINE")
    assert not hasattr(registry, "RegistryAPIError")
    assert not hasattr(registry, "RegistryLookupError")
    assert not hasattr(registry, "RegistryQueryError")
    assert not hasattr(registry, "get_version")
    assert not hasattr(registry, "get_info")
    assert not hasattr(registry, "get_data")
    assert not hasattr(registry, "get_mines")
    assert not hasattr(registry, "getVersion")
    assert not hasattr(registry, "getInfo")
    assert not hasattr(registry, "getData")
    assert not hasattr(registry, "getMines")
    assert not hasattr(registry, "resolve_benchmark_phase_plan")
    assert not hasattr(registry, "resolve_benchmark_plan")
    assert not hasattr(registry, "resolve_named_benchmark_profile")
    assert not hasattr(registry, "resolve_preferred_workers")
    assert not hasattr(registry, "resolve_production_plan")
    assert not hasattr(registry, "resolve_production_resource_profile")
    assert not hasattr(registry, "DEFAULT_BENCHMARK_SMALL_PROFILE")
    assert not hasattr(registry, "DEFAULT_BENCHMARK_LARGE_PROFILE")
    assert not hasattr(registry, "DEFAULT_BENCHMARK_FALLBACK_PROFILE")
