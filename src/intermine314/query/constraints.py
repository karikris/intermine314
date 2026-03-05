from __future__ import annotations

import string
from typing import Any

from intermine314.query.pathfeatures import PATH_PATTERN, PathFeature


class Constraint(PathFeature):
    """Minimal query constraint node for XML serialization."""

    child_type = "constraint"


class CodedConstraint(Constraint):
    OPS = frozenset()

    def __init__(self, path: str, op: str, code: str = "A"):
        normalized_op = str(op).strip().upper()
        if normalized_op not in self.OPS:
            raise TypeError(f"{normalized_op} not in {sorted(self.OPS)}")
        self.op = normalized_op
        self.code = str(code)
        super().__init__(path)

    def __str__(self) -> str:
        return self.code

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(op=self.op, code=self.code)
        return payload


class UnaryConstraint(CodedConstraint):
    OPS = frozenset({"IS NULL", "IS NOT NULL"})


class BinaryConstraint(CodedConstraint):
    OPS = frozenset({"=", "!=", "<", ">", "<=", ">=", "LIKE", "NOT LIKE", "CONTAINS"})

    def __init__(self, path: str, op: str, value: Any, code: str = "A"):
        self.value = value
        super().__init__(path, op, code)

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(value=str(self.value))
        return payload


class MultiConstraint(CodedConstraint):
    OPS = frozenset({"ONE OF", "NONE OF"})

    def __init__(self, path: str, op: str, values: list[Any] | tuple[Any, ...] | set[Any], code: str = "A"):
        if not isinstance(values, (list, tuple, set)):
            raise TypeError("values must be a list, tuple, or set")
        self.values = [str(item) for item in values]
        super().__init__(path, op, code)

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(value=self.values)
        return payload


class SubClassConstraint(Constraint):
    def __init__(self, path: str, subclass: str):
        subclass_name = str(subclass)
        if not PATH_PATTERN.match(subclass_name):
            raise TypeError("subclass must be a valid class name")
        self.subclass = subclass_name
        super().__init__(path)

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(type=self.subclass)
        return payload


class ConstraintFactory:
    """Minimal constructor for scalar and IN-style constraints."""

    reference_ops = frozenset()

    def __init__(self):
        self._code_index = 0

    def get_next_code(self) -> str:
        alphabet = string.ascii_uppercase
        index = int(self._code_index)
        self._code_index += 1
        chars = []
        while True:
            index, rem = divmod(index, 26)
            chars.append(alphabet[rem])
            if index == 0:
                break
            index -= 1
        return "".join(reversed(chars))

    def _normalize_op(self, op: Any) -> str:
        normalized = str(op).strip().upper()
        if normalized == "IN":
            return "ONE OF"
        if normalized == "NOT IN":
            return "NONE OF"
        return normalized

    def _attach_code(self, constraint: Constraint):
        if hasattr(constraint, "code") and getattr(constraint, "code") == "A":
            setattr(constraint, "code", self.get_next_code())
        return constraint

    def make_constraint(self, *args, **kwargs):
        if kwargs:
            if "path" in kwargs and "subclass" in kwargs:
                return SubClassConstraint(kwargs["path"], kwargs["subclass"])
            if "path" in kwargs and "op" in kwargs:
                path = kwargs["path"]
                op = self._normalize_op(kwargs["op"])
                if op in UnaryConstraint.OPS:
                    return self._attach_code(UnaryConstraint(path, op, kwargs.get("code", "A")))
                if op in MultiConstraint.OPS:
                    return self._attach_code(MultiConstraint(path, op, kwargs.get("value", []), kwargs.get("code", "A")))
                return self._attach_code(BinaryConstraint(path, op, kwargs.get("value"), kwargs.get("code", "A")))
            raise TypeError(f"Unsupported constraint kwargs: {sorted(kwargs.keys())}")

        if len(args) == 2:
            path, value = args
            if isinstance(value, (list, tuple, set)):
                return self._attach_code(MultiConstraint(path, "ONE OF", value, "A"))
            return self._attach_code(BinaryConstraint(path, "=", value, "A"))

        if len(args) == 3:
            path, raw_op, value = args
            op = self._normalize_op(raw_op)
            if op in MultiConstraint.OPS:
                return self._attach_code(MultiConstraint(path, op, value, "A"))
            if op in UnaryConstraint.OPS:
                return self._attach_code(UnaryConstraint(path, op, "A"))
            return self._attach_code(BinaryConstraint(path, op, value, "A"))

        if len(args) == 4:
            path, raw_op, value, code = args
            op = self._normalize_op(raw_op)
            if op in MultiConstraint.OPS:
                return MultiConstraint(path, op, value, code)
            if op in UnaryConstraint.OPS:
                return UnaryConstraint(path, op, code)
            return BinaryConstraint(path, op, value, code)

        raise TypeError(f"No matching minimal constraint for args={args!r}, kwargs={kwargs!r}")


__all__ = [
    "Constraint",
    "CodedConstraint",
    "UnaryConstraint",
    "BinaryConstraint",
    "MultiConstraint",
    "SubClassConstraint",
    "ConstraintFactory",
]
