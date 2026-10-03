"""Original result import paths sharing the native transport implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SYMBOL_TO_MODULE = {
    "JSONIterator": "intermine314.service.session",
    "ResultIterator": "intermine314.service.session",
    "InterMineURLOpener": "intermine314.service.session",
    "encode_str": "intermine314.service.session",
    "decode_binary": "intermine314.service.session",
    "encode_dict": "intermine314.service.session",
    "encode_headers": "intermine314.service.session",
}

__all__ = list(_SYMBOL_TO_MODULE)


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
