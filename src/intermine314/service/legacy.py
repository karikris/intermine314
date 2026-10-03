"""Legacy defaults over the shared native service implementations."""

from intermine314.service.service import Registry as NativeRegistry
from intermine314.service.service import Service as NativeService


class Service(NativeService):
    """Service facade with legacy defaults and the native transport lifecycle."""

    _DEFAULT_COMPATIBILITY = "legacy"

    def __getattribute__(self, name):
        """Retain the original keyword-capable explicit attribute lookup."""
        return object.__getattribute__(self, name)


class Registry(NativeRegistry):
    """Registry facade with legacy defaults and the native service cache."""

    _DEFAULT_COMPATIBILITY = "legacy"
