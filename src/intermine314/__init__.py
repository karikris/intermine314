from intermine314._version import VERSION, __version__

import importlib
import importlib.abc
import importlib.util
import sys


_MODULE_ALIASES = {
    # Legacy API shims (avoid keeping extra modules at package root).
    "intermine314.webservice": "intermine314.service.service",
    "intermine314.results": "intermine314.service.session",
    "intermine314.http_transport": "intermine314.service.transport",
    "intermine314.query_export": "intermine314.export.parquet",
    "intermine314.query_parallel": "intermine314.parallel.runner",
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


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target_name: str) -> None:
        self._target_name = target_name

    def create_module(self, spec):  # pragma: no cover - default module creation
        return None

    def exec_module(self, module) -> None:
        target = importlib.import_module(self._target_name)

        # Proxy to the real module on attribute access to keep the alias lazy.
        def __getattr__(name: str):
            return getattr(target, name)

        def __dir__():
            return sorted(set(dir(target)))

        module.__getattr__ = __getattr__  # type: ignore[attr-defined]
        module.__dir__ = __dir__  # type: ignore[attr-defined]
        module.__doc__ = target.__doc__
        if hasattr(target, "__all__"):
            module.__all__ = target.__all__  # type: ignore[attr-defined]


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):
        target_name = _MODULE_ALIASES.get(fullname)
        if target_name is None:
            return None
        return importlib.util.spec_from_loader(fullname, _AliasLoader(target_name))


def _install_alias_finder() -> None:
    for finder in sys.meta_path:
        if isinstance(finder, _AliasFinder):
            return
    sys.meta_path.insert(0, _AliasFinder())


_install_alias_finder()


def fetch_from_mine(*args, **kwargs):
    from intermine314.export.fetch import fetch_from_mine as _fetch_from_mine

    return _fetch_from_mine(*args, **kwargs)
