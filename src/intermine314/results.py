"""Original result import paths sharing the native transport implementations."""

from __future__ import annotations

from collections import UserDict
from importlib import import_module
from typing import Any

_SYMBOL_TO_MODULE = {
    "Attribute": "intermine314.model",
    "Collection": "intermine314.model",
    "Reference": "intermine314.model",
    "VERSION": "intermine314",
    "WebserviceError": "intermine314.service.errors",
    "ResultObject": "intermine314._result_object",
    "ResultRow": "intermine314.service.session",
    "TableResultRow": "intermine314.service.session",
    "FlatFileIterator": "intermine314.service.session",
    "JSONIterator": "intermine314.service.session",
    "ResultIterator": "intermine314.service.session",
    "InterMineURLOpener": "intermine314.service.session",
    "encode_str": "intermine314.service.session",
    "decode_binary": "intermine314.service.session",
    "encode_dict": "intermine314.service.session",
    "encode_headers": "intermine314.service.session",
}

__all__ = [*_SYMBOL_TO_MODULE, "EnrichmentLine"]


class EnrichmentLine(UserDict):
    """Enrichment mapping with the original underscore-to-hyphen aliases.

    Adapted from InterMine Python client 1.13.0 under LICENSE-BSD. All server
    keys remain available, including ``populationAnnotationCount``.
    """

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"EnrichmentLine({self.data})"

    def __getattr__(self, name):
        if name is not None:
            key_name = name.replace("_", "-")
            if key_name in self.keys():
                return self.data[key_name]
        raise AttributeError(name)


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
