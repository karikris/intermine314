"""Original imports sharing canonical constraints and the native factory default.

Use ConstraintFactory(compatibility="legacy") for legacy named-list dispatch.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_SYMBOL_TO_MODULE = {
    "Constraint": "intermine314.query.constraints",
    "CodedConstraint": "intermine314.query.constraints",
    "UnaryConstraint": "intermine314.query.constraints",
    "BinaryConstraint": "intermine314.query.constraints",
    "MultiConstraint": "intermine314.query.constraints",
    "ListConstraint": "intermine314.query.constraints",
    "LoopConstraint": "intermine314.query.constraints",
    "TernaryConstraint": "intermine314.query.constraints",
    "RangeConstraint": "intermine314.query.constraints",
    "IsaConstraint": "intermine314.query.constraints",
    "SubClassConstraint": "intermine314.query.constraints",
    "ConstraintFactory": "intermine314.query.constraints",
    "LogicNode": "intermine314.query.constraints",
    "LogicGroup": "intermine314.query.constraints",
    "LogicParser": "intermine314.query.constraints",
    "LogicParseError": "intermine314.query.constraints",
    "EmptyLogicError": "intermine314.query.constraints",
}

__all__ = list(_SYMBOL_TO_MODULE)


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
