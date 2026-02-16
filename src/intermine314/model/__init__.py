import logging

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

from .constants import (
    NUMERIC_TYPES,
    _ID_FIELD_NAME,
    _ID_FIELD_TYPE,
    _MODEL_HASH_PREFIX_CHARS,
    _MODEL_HASH_TEXT_CHUNK_CHARS,
    _MODEL_LOG_PREVIEW_CHARS,
    _MODEL_PARSE_ERROR_MESSAGE,
    _OP_IN,
    _OP_IS,
    _OP_IS_NOT,
    _OP_IS_NOT_NULL,
    _OP_IS_NULL,
    _OP_LOOKUP,
    _OP_NONE_OF,
    _OP_NOT_IN,
    _OP_ONE_OF,
    _ROOT_OBJECT_CLASS,
    _XML_ATTR_EXTENDS,
    _XML_ATTR_IS_INTERFACE,
    _XML_ATTR_NAME,
    _XML_ATTR_PACKAGE,
    _XML_ATTR_REFERENCED_TYPE,
    _XML_ATTR_REVERSE_REFERENCE,
    _XML_ATTR_TYPE,
    _XML_TAG_ATTRIBUTE,
    _XML_TAG_CLASS,
    _XML_TAG_COLLECTION,
    _XML_TAG_MODEL,
    _XML_TAG_REFERENCE,
    _XML_VALUE_TRUE,
)
from .errors import ModelError, ModelParseError, PathParseError
from .fields import Attribute, Collection, Field, Reference
from .class_ import Class, ComposedClass
from .path import Path
from .operators import CodelessNode, Column, ConstraintNode, ConstraintTree
from .model import Model
from .helpers import _copy_subclasses
from .xmlparse import (
    _format_xml_parse_error,
    _hash_and_count_text_bytes,
    _payload_metadata,
    _source_ref,
    _truncate_preview,
)

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

