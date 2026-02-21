from __future__ import annotations

import warnings

from intermine314.util.core import ReadableException, openAnything
from intermine314.util.logging import log_structured_event, new_job_id

_EMITTED_DEPRECATIONS: set[str] = set()


def _warn_deprecated_once(name: str, replacement: str) -> None:
    if name in _EMITTED_DEPRECATIONS:
        return
    _EMITTED_DEPRECATIONS.add(name)
    warnings.warn(
        f"intermine314.util.{name} is deprecated; import {replacement} instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def json_loads(payload):
    _warn_deprecated_once("json_loads", "intermine314.util.json.json_loads")
    from intermine314.util.json import json_loads as _json_loads

    return _json_loads(payload)


def json_dumps(payload):
    _warn_deprecated_once("json_dumps", "intermine314.util.json.json_dumps")
    from intermine314.util.json import json_dumps as _json_dumps

    return _json_dumps(payload)


def timed():
    _warn_deprecated_once("timed", "intermine314.util.timing.timed")
    from intermine314.util.timing import timed as _timed

    return _timed()

__all__ = [
    "openAnything",
    "ReadableException",
    "new_job_id",
    "log_structured_event",
    "timed",
    "json_loads",
    "json_dumps",
]
