import importlib
import logging
import warnings

"""
Classes representing the data model
===================================

Representations of tables and columns, and behaviour
for validating connections between them.

"""

__author__ = "Alex Kalderimis"
__organization__ = "InterMine"
__license__ = "LGPL"
__contact__ = "toffe.kari@gmail.com"

LOG = logging.getLogger(__name__)

from .constants import NUMERIC_TYPES
from .errors import ModelError, ModelParseError, PathParseError
from .fields import Attribute, Collection, Field, Reference
from .class_ import Class, ComposedClass
from .path import Path
from .operators import CodelessNode, Column, ConstraintNode, ConstraintTree
from .model import Model

__all__ = [
    "Model",
    "ModelError",
    "ModelParseError",
    "PathParseError",
    "Field",
    "Attribute",
    "Reference",
    "Collection",
    "Class",
    "ComposedClass",
    "Path",
    "ConstraintTree",
    "ConstraintNode",
    "CodelessNode",
    "Column",
    "NUMERIC_TYPES",
]

_EMITTED_DEPRECATIONS: set[str] = set()

# Deprecated package-root internal exports retained as compatibility aliases.
_DEPRECATED_INTERNAL_EXPORTS = {
    "_ID_FIELD_NAME": ("intermine314.model.constants", "intermine314.model.constants._ID_FIELD_NAME"),
    "_ID_FIELD_TYPE": ("intermine314.model.constants", "intermine314.model.constants._ID_FIELD_TYPE"),
    "_MODEL_HASH_PREFIX_CHARS": (
        "intermine314.model.constants",
        "intermine314.model.constants._MODEL_HASH_PREFIX_CHARS",
    ),
    "_MODEL_HASH_TEXT_CHUNK_CHARS": (
        "intermine314.model.constants",
        "intermine314.model.constants._MODEL_HASH_TEXT_CHUNK_CHARS",
    ),
    "_MODEL_LOG_PREVIEW_CHARS": (
        "intermine314.model.constants",
        "intermine314.model.constants._MODEL_LOG_PREVIEW_CHARS",
    ),
    "_MODEL_PARSE_ERROR_MESSAGE": (
        "intermine314.model.constants",
        "intermine314.model.constants._MODEL_PARSE_ERROR_MESSAGE",
    ),
    "_OP_IN": ("intermine314.model.constants", "intermine314.model.constants._OP_IN"),
    "_OP_IS": ("intermine314.model.constants", "intermine314.model.constants._OP_IS"),
    "_OP_IS_NOT": ("intermine314.model.constants", "intermine314.model.constants._OP_IS_NOT"),
    "_OP_IS_NOT_NULL": ("intermine314.model.constants", "intermine314.model.constants._OP_IS_NOT_NULL"),
    "_OP_IS_NULL": ("intermine314.model.constants", "intermine314.model.constants._OP_IS_NULL"),
    "_OP_LOOKUP": ("intermine314.model.constants", "intermine314.model.constants._OP_LOOKUP"),
    "_OP_NONE_OF": ("intermine314.model.constants", "intermine314.model.constants._OP_NONE_OF"),
    "_OP_NOT_IN": ("intermine314.model.constants", "intermine314.model.constants._OP_NOT_IN"),
    "_OP_ONE_OF": ("intermine314.model.constants", "intermine314.model.constants._OP_ONE_OF"),
    "_ROOT_OBJECT_CLASS": ("intermine314.model.constants", "intermine314.model.constants._ROOT_OBJECT_CLASS"),
    "_XML_ATTR_EXTENDS": ("intermine314.model.constants", "intermine314.model.constants._XML_ATTR_EXTENDS"),
    "_XML_ATTR_IS_INTERFACE": (
        "intermine314.model.constants",
        "intermine314.model.constants._XML_ATTR_IS_INTERFACE",
    ),
    "_XML_ATTR_NAME": ("intermine314.model.constants", "intermine314.model.constants._XML_ATTR_NAME"),
    "_XML_ATTR_PACKAGE": ("intermine314.model.constants", "intermine314.model.constants._XML_ATTR_PACKAGE"),
    "_XML_ATTR_REFERENCED_TYPE": (
        "intermine314.model.constants",
        "intermine314.model.constants._XML_ATTR_REFERENCED_TYPE",
    ),
    "_XML_ATTR_REVERSE_REFERENCE": (
        "intermine314.model.constants",
        "intermine314.model.constants._XML_ATTR_REVERSE_REFERENCE",
    ),
    "_XML_ATTR_TYPE": ("intermine314.model.constants", "intermine314.model.constants._XML_ATTR_TYPE"),
    "_XML_TAG_ATTRIBUTE": ("intermine314.model.constants", "intermine314.model.constants._XML_TAG_ATTRIBUTE"),
    "_XML_TAG_CLASS": ("intermine314.model.constants", "intermine314.model.constants._XML_TAG_CLASS"),
    "_XML_TAG_COLLECTION": ("intermine314.model.constants", "intermine314.model.constants._XML_TAG_COLLECTION"),
    "_XML_TAG_MODEL": ("intermine314.model.constants", "intermine314.model.constants._XML_TAG_MODEL"),
    "_XML_TAG_REFERENCE": ("intermine314.model.constants", "intermine314.model.constants._XML_TAG_REFERENCE"),
    "_XML_VALUE_TRUE": ("intermine314.model.constants", "intermine314.model.constants._XML_VALUE_TRUE"),
    "_copy_subclasses": ("intermine314.model.helpers", "dict(subclasses or {})"),
    "_format_xml_parse_error": ("intermine314.model.xmlparse", "intermine314.model.xmlparse._format_xml_parse_error"),
    "_hash_and_count_text_bytes": (
        "intermine314.model.xmlparse",
        "intermine314.model.xmlparse._hash_and_count_text_bytes",
    ),
    "_payload_metadata": ("intermine314.model.xmlparse", "intermine314.model.xmlparse._payload_metadata"),
    "_source_ref": ("intermine314.model.xmlparse", "intermine314.model.xmlparse._source_ref"),
    "_truncate_preview": ("intermine314.model.xmlparse", "intermine314.model.xmlparse._truncate_preview"),
}


def _warn_deprecated_once(name: str, replacement: str) -> None:
    if name in _EMITTED_DEPRECATIONS:
        return
    _EMITTED_DEPRECATIONS.add(name)
    warnings.warn(
        f"intermine314.model.{name} is deprecated; import {replacement} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def __getattr__(name: str):
    alias = _DEPRECATED_INTERNAL_EXPORTS.get(name)
    if alias is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, replacement = alias
    _warn_deprecated_once(name, replacement)
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_DEPRECATED_INTERNAL_EXPORTS))

# Preserve repr/introspection compatibility for public classes.
for _cls in [
    Field,
    Attribute,
    Reference,
    Collection,
    Class,
    ComposedClass,
    Path,
    ConstraintTree,
    ConstraintNode,
    CodelessNode,
    Column,
    Model,
    ModelError,
    PathParseError,
    ModelParseError,
]:
    _cls.__module__ = __name__

# Preserve historical logger namespace for Model class-level logging.
Model.LOG = logging.getLogger(__name__)
