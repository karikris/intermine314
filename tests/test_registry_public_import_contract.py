from __future__ import annotations

from intermine314 import registry
from intermine314.registry import mines as registry_mines


def test_registry_package_public_exports_are_explicit_and_canonical():
    assert set(registry.__all__) == {"resolve_execution_plan"}
    assert registry.resolve_execution_plan is registry_mines.resolve_execution_plan


def test_registry_mines_module_exposes_no_public_default_constant_surface():
    leaked_defaults = [name for name in dir(registry_mines) if name.startswith("DEFAULT_")]
    assert leaked_defaults == []
