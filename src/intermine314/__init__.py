from intermine314._version import VERSION, __version__

import importlib
import logging
import sys
from types import ModuleType

_LEGACY_ALIAS_SENTINEL = "_intermine314_legacy_alias_module"
_ALIAS_LOG = logging.getLogger("intermine314.compat")


_MODULE_ALIASES = {
    # Legacy API shims (avoid keeping extra modules at package root).
    "intermine314.webservice": "intermine314.service.service",
    "intermine314.results": "intermine314.service.session",
    "intermine314.http_transport": "intermine314.service.transport",
    "intermine314.query_export": "intermine314.export.parquet",
    "intermine314.query_parallel": "intermine314.parallel.policy",
    "intermine314.mine_registry": "intermine314.registry.mines",
    # Legacy module names that were moved into subpackages.
    "intermine314.fetch": "intermine314.export.fetch",
    "intermine314.bulk_export": "intermine314.export.targeted",
    "intermine314.constants": "intermine314.config.constants",
    "intermine314.errors": "intermine314.service.errors",
    "intermine314.decorators": "intermine314.service.decorators",
    "intermine314.optional_deps": "intermine314.util.deps",
    "intermine314.service_urls": "intermine314.service.urls",
    "intermine314.idresolution": "intermine314.service.idresolution",
    "intermine314.query_manager": "intermine314.service.query_manager",
    "intermine314.pathfeatures": "intermine314.query.pathfeatures",
    "intermine314.constraints": "intermine314.query.constraints",
}


def _build_legacy_alias_module(alias_name: str, target_name: str) -> ModuleType:
    module = ModuleType(alias_name)
    module.__package__ = alias_name.rpartition(".")[0]
    module.__doc__ = f"Legacy module alias for {target_name}"
    setattr(module, _LEGACY_ALIAS_SENTINEL, True)

    target_module = None

    def _resolve_target():
        nonlocal target_module
        if target_module is None:
            target_module = importlib.import_module(target_name)
            module.__doc__ = target_module.__doc__
            if hasattr(target_module, "__all__"):
                module.__all__ = target_module.__all__  # type: ignore[attr-defined]
        return target_module

    # Keep the alias lazy by resolving the target only on first attribute access.
    def __getattr__(name: str):
        return getattr(_resolve_target(), name)

    def __dir__():
        return sorted(set(dir(_resolve_target())))

    module.__getattr__ = __getattr__  # type: ignore[attr-defined]
    module.__dir__ = __dir__  # type: ignore[attr-defined]
    return module


def _install_legacy_module_aliases() -> None:
    installed = []
    package_module = sys.modules.get(__name__)
    if package_module is None:  # pragma: no cover - package is expected to be loaded
        return

    for alias_name, target_name in _MODULE_ALIASES.items():
        existing = sys.modules.get(alias_name)
        if existing is not None:
            continue
        alias_module = _build_legacy_alias_module(alias_name, target_name)
        sys.modules[alias_name] = alias_module

        parent_name, _, child_name = alias_name.rpartition(".")
        if parent_name == __name__ and child_name and not hasattr(package_module, child_name):
            setattr(package_module, child_name, alias_module)
        installed.append(alias_name)

    if installed:
        _ALIAS_LOG.debug("installed_legacy_alias_modules count=%d", len(installed))


_install_legacy_module_aliases()


def fetch_from_mine(*args, **kwargs):
    from intermine314.export.fetch import fetch_from_mine as _fetch_from_mine

    return _fetch_from_mine(*args, **kwargs)
