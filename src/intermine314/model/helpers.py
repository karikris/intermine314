from functools import lru_cache
import warnings
from typing import Any, Dict, Mapping, Optional

from .constants import _PATH_SEGMENT_CACHE_MAXSIZE

_EMITTED_DEPRECATIONS: set[str] = set()


def _warn_deprecated_once(name: str, replacement: str) -> None:
    if name in _EMITTED_DEPRECATIONS:
        return
    _EMITTED_DEPRECATIONS.add(name)
    warnings.warn(
        f"{name} is deprecated; use {replacement} instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def _copy_subclasses(subclasses: Optional[Mapping[str, str]]) -> Dict[str, str]:
    _warn_deprecated_once("intermine314.model.helpers._copy_subclasses", "dict(subclasses or {})")
    if subclasses is None:
        return {}
    return dict(subclasses)


@lru_cache(maxsize=_PATH_SEGMENT_CACHE_MAXSIZE)
def _split_path_segments(path_string: str) -> tuple[str, ...]:
    return tuple(path_string.split("."))


@lru_cache(maxsize=None)
def _slot_names_for_type(cls: type) -> frozenset[str]:
    names = set()
    for base in cls.__mro__:
        slots = base.__dict__.get("__slots__")
        if slots is None:
            continue
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in {"__dict__", "__weakref__"}:
                continue
            names.add(str(name))
    return frozenset(names)


def _set_slot_or_extension(obj: Any, name: str, value: Any) -> None:
    if name in _slot_names_for_type(type(obj)):
        object.__setattr__(obj, name, value)
        return
    extensions = object.__getattribute__(obj, "_extensions")
    if extensions is None:
        extensions = {}
        object.__setattr__(obj, "_extensions", extensions)
    extensions[str(name)] = value


def _get_extension_attr(obj: Any, name: str) -> Any:
    extensions = object.__getattribute__(obj, "_extensions")
    if extensions is None:
        raise AttributeError(name)
    if name in extensions:
        return extensions[name]
    raise AttributeError(name)


def _get_extensions_map(obj: Any) -> dict[str, Any]:
    extensions = object.__getattribute__(obj, "_extensions")
    if extensions is None:
        extensions = {}
        object.__setattr__(obj, "_extensions", extensions)
    return extensions
