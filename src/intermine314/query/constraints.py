from __future__ import annotations

import re
import string
from typing import Any

from intermine314.query.pathfeatures import PATH_PATTERN, PathFeature
from intermine314.util import ReadableException


class Constraint(PathFeature):
    """Minimal query constraint node for XML serialization."""

    child_type = "constraint"


class LogicNode:
    """A node in constraint logic; + and & join with AND, | with OR."""

    def __add__(self, other):
        if not isinstance(other, LogicNode):
            return NotImplemented
        return LogicGroup(self, "AND", other)

    def __and__(self, other):
        if not isinstance(other, LogicNode):
            return NotImplemented
        return LogicGroup(self, "AND", other)

    def __or__(self, other):
        if not isinstance(other, LogicNode):
            return NotImplemented
        return LogicGroup(self, "OR", other)


class LogicGroup(LogicNode):
    """Two constraint logic nodes connected by AND or OR."""

    LEGAL_OPS = frozenset({"AND", "OR"})

    def __init__(self, left, op, right, parent=None):
        if op not in self.LEGAL_OPS:
            raise TypeError(f"{op} is not a legal logical operation")
        for node in (left, right):
            if not isinstance(node, LogicNode) or not callable(getattr(node, "get_codes", None)):
                raise TypeError("Logic groups require constraint logic nodes")
        self.left = left
        self.right = right
        self.op = op
        self.parent = parent
        for node in (left, right):
            if isinstance(node, LogicGroup):
                node.parent = self

    def __str__(self):
        core = f"{self.left} {self.op.lower()} {self.right}"
        if self.parent is not None and self.op != self.parent.op:
            return f"({core})"
        return core

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self}>"

    def get_codes(self):
        return self.left.get_codes() + self.right.get_codes()


class LogicParseError(ReadableException):
    """Invalid constraint logic syntax or grouping."""


class EmptyLogicError(ValueError):
    """No constraint logic expression was supplied."""


class LogicParser:
    """Parse constraint codes using historical InterMine string precedence.

    OR binds more tightly than AND, as exercised by the original client's
    tests (despite its contrary docstring). Closing a group also completes
    the operation immediately before it, following the original parser.
    Python expressions built with LogicNode operators use Python precedence.
    """

    ops = {"AND": "AND", "&": "AND", "&&": "AND", "OR": "OR",
           "|": "OR", "||": "OR", "(": "(", ")": ")"}

    def __init__(self, query):
        self._query = query

    def get_constraint(self, code):
        return self._query.get_constraint(code)

    def get_priority(self, op):
        return {"AND": 2, "OR": 1, "(": 3, ")": 3}.get(op)

    def parse(self, logic_str):
        if not isinstance(logic_str, str):
            raise TypeError("Constraint logic must be a string or logic node")
        tokens = re.findall(r"[A-Z]+|&&?|\|\|?|[()]|\S", logic_str.upper())
        return self.postfix_to_tree(self.infix_to_postfix(tokens))

    def check_syntax(self, infix_tokens):
        tokens = list(infix_tokens)
        if not tokens:
            raise EmptyLogicError()
        need_operand = True
        depth = 0
        for raw_token in tokens:
            token = self.ops.get(raw_token, raw_token)
            if token == "(":
                if not need_operand:
                    raise LogicParseError("Expected an operator before opening bracket")
                depth += 1
            elif token == ")":
                if depth == 0:
                    raise LogicParseError("Unmatched closing bracket")
                if need_operand:
                    raise LogicParseError("Expected a constraint before closing bracket")
                depth -= 1
            elif token in ("AND", "OR"):
                if need_operand:
                    raise LogicParseError("Expected a constraint before operator " + token)
                need_operand = True
            else:
                if not isinstance(token, str) or not re.fullmatch(r"[A-Z]+", token):
                    raise LogicParseError(f"Invalid constraint logic token: {token!r}")
                if not need_operand:
                    raise LogicParseError("Expected an operator before constraint " + token)
                need_operand = False
        if depth:
            raise LogicParseError("Unmatched opening bracket")
        if need_operand:
            raise LogicParseError("Expected a constraint after operator")

    def infix_to_postfix(self, infix_tokens):
        tokens = list(infix_tokens)
        self.check_syntax(tokens)
        stack = []
        postfix = []
        for raw_token in tokens:
            token = self.ops.get(raw_token, raw_token)
            if token == "(":
                stack.append(token)
            elif token == ")":
                while stack[-1] != "(":
                    postfix.append(stack.pop())
                stack.pop()
                # The original parser closes the operation immediately before
                # a group too. Keep that behavior without discarding an outer
                # opening bracket (which used to break nested expressions).
                if stack and stack[-1] != "(":
                    postfix.append(stack.pop())
            elif token in ("AND", "OR"):
                while stack and stack[-1] != "(" and self.get_priority(stack[-1]) <= self.get_priority(token):
                    postfix.append(stack.pop())
                stack.append(token)
            else:
                postfix.append(token)
        postfix.extend(reversed(stack))
        return postfix

    def postfix_to_tree(self, postfix_tokens):
        tokens = list(postfix_tokens)
        if not tokens:
            raise EmptyLogicError()
        stack = []
        for raw_token in tokens:
            token = self.ops.get(raw_token, raw_token)
            if token in ("AND", "OR"):
                if len(stack) < 2:
                    raise LogicParseError("Expected two operands for " + token)
                right, left = stack.pop(), stack.pop()
                stack.append(LogicGroup(left, token, right))
            elif isinstance(token, str) and re.fullmatch(r"[A-Z]+", token):
                stack.append(self.get_constraint(token))
            else:
                raise LogicParseError(f"Invalid postfix logic token: {token!r}")
        if len(stack) != 1:
            raise LogicParseError("Logic tree does not have a unique root")
        return stack[0]


class CodedConstraint(Constraint, LogicNode):
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

    def get_codes(self):
        return [self.code]

    def to_string(self):
        return f"{super().to_string()} {self.op}"

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
    "LogicNode",
    "LogicGroup",
    "LogicParser",
    "LogicParseError",
    "EmptyLogicError",
]
