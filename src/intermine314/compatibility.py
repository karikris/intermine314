"""Shared profile selection without eager model or analytics imports."""

from __future__ import annotations

import sys


def resolve_compatibility(compatibility, *, default="native") -> str:
    """Resolve an omitted profile and reject unsupported explicit profiles."""
    profile = default if compatibility is None else compatibility
    if not isinstance(profile, str) or profile not in ("native", "legacy"):
        raise ValueError("compatibility must be 'native', 'legacy', or None")
    return profile


def query_compatibility(compatibility, *, model=None, service=None) -> str:
    """Explicit profile, restored Model inference, service profile, then native."""
    if compatibility is not None:
        return resolve_compatibility(compatibility)
    model_module = sys.modules.get("intermine314.model")
    model_type = getattr(model_module, "Model", None)
    if isinstance(model_type, type) and isinstance(model, model_type):
        return "legacy"
    return resolve_compatibility(getattr(service, "compatibility", None))


def class_name(root) -> str | None:
    """Keep QuerySpec roots as names when model Class roots are restored."""
    if root is None:
        return None
    return str(getattr(root, "name", root))
