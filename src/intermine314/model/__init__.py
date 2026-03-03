import logging

"""
Classes representing the data model
===================================

Representations of tables and columns, and behaviour
for validating connections between them.

"""

__author__ = "Alex Kalderimis"
__organization__ = "InterMine"
__license__ = "MIT"
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
