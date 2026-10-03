from __future__ import annotations

import inspect
import re
import string
from typing import Any

from intermine314.compatibility import resolve_compatibility
from intermine314.query.pathfeatures import PATH_PATTERN, PathFeature
from intermine314.util import ReadableException


class Constraint(PathFeature):
    """Query constraint node for XML serialization."""

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

    def to_string(self):
        return f"{super().to_string()} {self.value}"


class ListConstraint(CodedConstraint):
    """Membership in a named server list, or a list/query protocol object."""

    OPS = frozenset({"IN", "NOT IN"})

    def __init__(self, path, op, list_name, code="A"):
        # Validate before invoking an upload protocol with side effects.
        super().__init__(path, op, code)
        if hasattr(list_name, "to_query"):
            query = list_name.to_query()
            self.list_name = query.service.create_list(query).name
        else:
            self.list_name = getattr(list_name, "name", list_name)

    def to_string(self):
        return f"{super().to_string()} {self.list_name}"

    def to_dict(self):
        return dict(super().to_dict(), value=str(self.list_name))


class LoopConstraint(CodedConstraint):
    OPS = frozenset({"IS", "IS NOT"})
    SERIALISED_OPS = {"IS": "=", "IS NOT": "!="}

    def __init__(self, path, op, loopPath, code="A"):
        self.loopPath = PathFeature(loopPath).path
        super().__init__(path, op, code)

    def to_string(self):
        return f"{super().to_string()} {self.loopPath}"

    def to_dict(self):
        return dict(super().to_dict(), loopPath=self.loopPath, op=self.SERIALISED_OPS[self.op])


class TernaryConstraint(BinaryConstraint):
    OPS = frozenset({"LOOKUP"})

    def __init__(self, path, op, value, extra_value=None, code="A"):
        self.extra_value = extra_value
        super().__init__(path, op, value, code)

    def to_string(self):
        suffix = "" if self.extra_value is None else f" IN {self.extra_value}"
        return super().to_string() + suffix

    def to_dict(self):
        payload = super().to_dict()
        if self.extra_value is not None:
            payload["extraValue"] = self.extra_value
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

    def to_string(self):
        return f"{super().to_string()} {self.values}"


class RangeConstraint(MultiConstraint):
    OPS = frozenset({"OVERLAPS", "DOES NOT OVERLAP", "WITHIN", "OUTSIDE", "CONTAINS", "DOES NOT CONTAIN"})


class IsaConstraint(MultiConstraint):
    OPS = frozenset({"ISA"})


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

    def to_string(self):
        return f"{super().to_string()} ISA {self.subclass}"


class TemplateConstraint:
    """Editability and active state shared by template constraint variants.

    Codeless subclass refinements may be editable too, as in the pinned client.
    Python booleans and canonical XML editable strings are both accepted.
    """

    REQUIRED = "locked"
    OPTIONAL_ON = "on"
    OPTIONAL_OFF = "off"

    def __init__(self, editable=True, optional="locked"):
        self.editable = editable is True or editable == "true"
        if optional not in (self.REQUIRED, self.OPTIONAL_ON, self.OPTIONAL_OFF):
            raise TypeError("Bad value for optional")
        self.optional = optional != self.REQUIRED
        self.switched_on = optional != self.OPTIONAL_OFF

    @property
    def required(self):
        return not self.optional

    @property
    def switched_off(self):
        return not self.switched_on

    def get_switchable_status(self):
        if self.required:
            return self.REQUIRED
        return self.OPTIONAL_ON if self.switched_on else self.OPTIONAL_OFF

    def switch_on(self):
        if not (self.editable and self.optional):
            raise ValueError("This constraint is not switchable")
        self.switched_on = True

    def switch_off(self):
        if not (self.editable and self.optional):
            raise ValueError("This constraint is not switchable")
        self.switched_on = False

    def to_string(self):
        editable = "editable" if self.editable else "non-editable"
        return f"({editable}, {self.get_switchable_status()})"

    def separate_arg_sets(self, args):
        constraint_args = {}
        template_args = {}
        for key, value in args.items():
            if key == "editable":
                template_args[key] = value is True or value == "true"
            elif key == "optional":
                template_args[key] = value
            else:
                constraint_args[key] = value
        return constraint_args, template_args


class TemplateUnaryConstraint(UnaryConstraint, TemplateConstraint):
    def __init__(self, *args, **kwargs):
        constraint_args, template_args = self.separate_arg_sets(kwargs)
        TemplateConstraint.__init__(self, **template_args)
        UnaryConstraint.__init__(self, *args, **constraint_args)

    def to_string(self):
        return UnaryConstraint.to_string(self) + " " + TemplateConstraint.to_string(self)


class TemplateBinaryConstraint(BinaryConstraint, TemplateConstraint):
    def __init__(self, *args, **kwargs):
        constraint_args, template_args = self.separate_arg_sets(kwargs)
        TemplateConstraint.__init__(self, **template_args)
        BinaryConstraint.__init__(self, *args, **constraint_args)

    def to_string(self):
        return BinaryConstraint.to_string(self) + " " + TemplateConstraint.to_string(self)


class TemplateListConstraint(ListConstraint, TemplateConstraint):
    def __init__(self, *args, **kwargs):
        constraint_args, template_args = self.separate_arg_sets(kwargs)
        TemplateConstraint.__init__(self, **template_args)
        ListConstraint.__init__(self, *args, **constraint_args)

    def to_string(self):
        return ListConstraint.to_string(self) + " " + TemplateConstraint.to_string(self)


class TemplateLoopConstraint(LoopConstraint, TemplateConstraint):
    def __init__(self, *args, **kwargs):
        constraint_args, template_args = self.separate_arg_sets(kwargs)
        TemplateConstraint.__init__(self, **template_args)
        LoopConstraint.__init__(self, *args, **constraint_args)

    def to_string(self):
        return LoopConstraint.to_string(self) + " " + TemplateConstraint.to_string(self)


class TemplateTernaryConstraint(TernaryConstraint, TemplateConstraint):
    def __init__(self, *args, **kwargs):
        constraint_args, template_args = self.separate_arg_sets(kwargs)
        TemplateConstraint.__init__(self, **template_args)
        TernaryConstraint.__init__(self, *args, **constraint_args)

    def to_string(self):
        return TernaryConstraint.to_string(self) + " " + TemplateConstraint.to_string(self)


class TemplateMultiConstraint(MultiConstraint, TemplateConstraint):
    def __init__(self, *args, **kwargs):
        constraint_args, template_args = self.separate_arg_sets(kwargs)
        TemplateConstraint.__init__(self, **template_args)
        MultiConstraint.__init__(self, *args, **constraint_args)

    def to_string(self):
        return MultiConstraint.to_string(self) + " " + TemplateConstraint.to_string(self)


class TemplateRangeConstraint(RangeConstraint, TemplateConstraint):
    def __init__(self, *args, **kwargs):
        constraint_args, template_args = self.separate_arg_sets(kwargs)
        TemplateConstraint.__init__(self, **template_args)
        RangeConstraint.__init__(self, *args, **constraint_args)

    def to_string(self):
        return RangeConstraint.to_string(self) + " " + TemplateConstraint.to_string(self)


class TemplateIsaConstraint(IsaConstraint, TemplateConstraint):
    def __init__(self, *args, **kwargs):
        constraint_args, template_args = self.separate_arg_sets(kwargs)
        TemplateConstraint.__init__(self, **template_args)
        IsaConstraint.__init__(self, *args, **constraint_args)

    def to_string(self):
        return IsaConstraint.to_string(self) + " " + TemplateConstraint.to_string(self)


class TemplateSubClassConstraint(SubClassConstraint, TemplateConstraint):
    def __init__(self, *args, **kwargs):
        constraint_args, template_args = self.separate_arg_sets(kwargs)
        TemplateConstraint.__init__(self, **template_args)
        SubClassConstraint.__init__(self, *args, **constraint_args)

    def to_string(self):
        return f"{self.path} ISA {self.subclass} " + TemplateConstraint.to_string(self)


class ConstraintFactory:
    """Deterministic constructor; standalone facades default to native.

    Legacy IN/NOT IN use named server lists. Native IN/NOT IN alias
    ONE OF/NONE OF for collections and use named lists otherwise.
    CONTAINS chooses ranges for a list/tuple/set value,
    scalar binary matching otherwise. Explicit subclass= works in both
    profiles; the legacy two-argument call means subclass refinement.
    """

    CONSTRAINT_CLASSES = frozenset({UnaryConstraint, BinaryConstraint, TernaryConstraint,
        MultiConstraint, SubClassConstraint, LoopConstraint, ListConstraint,
        RangeConstraint, IsaConstraint})
    reference_ops = TernaryConstraint.OPS | RangeConstraint.OPS | ListConstraint.OPS | IsaConstraint.OPS

    def __init__(self, *, compatibility=None):
        self.compatibility = resolve_compatibility(compatibility)
        self._code_index = 0
        self._used_codes = set()

    def get_next_code(self) -> str:
        while True:
            code = self._next_code()
            if code not in self._used_codes:
                self._used_codes.add(code)
                return code

    def _next_code(self) -> str:
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
        return normalized

    def _split_constraint_args(self, kwargs):
        return dict(kwargs), {}

    def _constraint_class(self, cls):
        return cls

    def make_constraint(self, *args, **kwargs):
        args = list(args)
        kwargs, extra_args = self._split_constraint_args(kwargs)
        raw_op = args[1] if len(args) > 1 else kwargs.get("op")
        op = self._normalize_op(raw_op) if raw_op is not None else None
        value = args[2] if len(args) > 2 else kwargs.get("values", kwargs.get("value", kwargs.get("list_name")))
        if self.compatibility == "native" and op in ListConstraint.OPS and isinstance(value, (list, tuple, set)):
            op = "ONE OF" if op == "IN" else "NONE OF"
        if "subclass" in kwargs:
            cls = SubClassConstraint
        elif op in UnaryConstraint.OPS:
            cls = UnaryConstraint
            # Retain the existing query triple's empty value placeholder;
            # a real third argument is the upstream unary code.
            if len(args) > 2 and args[2] is None:
                args.pop(2)
        elif len(args) == 2 and "op" not in kwargs and (
            self.compatibility == "native" or op not in (
                BinaryConstraint.OPS | self.reference_ops | MultiConstraint.OPS | LoopConstraint.OPS
            )
        ) and not any(
            key in kwargs for key in ("value", "values", "list_name", "loopPath", "extra_value")
        ):
            if self.compatibility == "legacy":
                cls = SubClassConstraint
            else:
                cls = MultiConstraint if isinstance(args[1], (list, tuple, set)) else BinaryConstraint
                args.insert(1, "ONE OF" if cls is MultiConstraint else "=")
                op = args[1]
        elif op in TernaryConstraint.OPS:
            cls = TernaryConstraint
        elif op in LoopConstraint.OPS:
            cls = LoopConstraint
        elif op in ListConstraint.OPS:
            cls = ListConstraint
        elif op in IsaConstraint.OPS:
            cls = IsaConstraint
        elif op in MultiConstraint.OPS:
            cls = MultiConstraint
        elif op in RangeConstraint.OPS and (op != "CONTAINS" or isinstance(
            args[2] if len(args) > 2 else kwargs.get("values", kwargs.get("value")), (list, tuple, set)
        )):
            cls = RangeConstraint
        elif op in BinaryConstraint.OPS:
            cls = BinaryConstraint
        else:
            raise TypeError(f"No matching constraint operator: {raw_op!r}")

        if cls is not SubClassConstraint:
            if len(args) > 1:
                args[1] = op
            elif "op" in kwargs:
                kwargs["op"] = op
            alias = "values" if issubclass(cls, MultiConstraint) else (
                "list_name" if cls is ListConstraint else "loopPath" if cls is LoopConstraint else "value"
            )
            if alias != "value" and "value" in kwargs:
                if alias in kwargs:
                    raise TypeError(f"Multiple values for {alias}")
                kwargs[alias] = kwargs.pop("value")

        # Bind once before construction: unknown arguments cannot trigger an
        # upload, and genuine constructor/upload TypeErrors propagate unchanged.
        bound = inspect.signature(cls).bind(*args, **kwargs)
        has_explicit_code = "code" in bound.arguments
        explicit_code = str(bound.arguments["code"]) if has_explicit_code else None
        if has_explicit_code:
            if not re.fullmatch(r"[A-Z]+", explicit_code):
                raise TypeError("Constraint code must be uppercase alphabetic")
            if explicit_code in self._used_codes:
                raise TypeError(f"Constraint code {explicit_code!r} is already in use")
        con = self._constraint_class(cls)(*args, **kwargs, **extra_args)
        if isinstance(con, CodedConstraint):
            if not has_explicit_code:
                con.code = self.get_next_code()
            else:
                self._used_codes.add(con.code)
        return con


class TemplateConstraintFactory(ConstraintFactory):
    """Use ordinary profile-aware dispatch and code allocation for templates."""

    _TEMPLATE_CLASSES = {
        UnaryConstraint: TemplateUnaryConstraint,
        BinaryConstraint: TemplateBinaryConstraint,
        ListConstraint: TemplateListConstraint,
        LoopConstraint: TemplateLoopConstraint,
        TernaryConstraint: TemplateTernaryConstraint,
        MultiConstraint: TemplateMultiConstraint,
        RangeConstraint: TemplateRangeConstraint,
        IsaConstraint: TemplateIsaConstraint,
        SubClassConstraint: TemplateSubClassConstraint,
    }
    CONSTRAINT_CLASSES = frozenset(_TEMPLATE_CLASSES.values())

    def _split_constraint_args(self, kwargs):
        constraint_args, template_args = TemplateConstraint().separate_arg_sets(kwargs)
        # Reject invalid template state before any named-list upload or code use.
        TemplateConstraint(**template_args)
        return constraint_args, template_args

    def _constraint_class(self, cls):
        return self._TEMPLATE_CLASSES[cls]


__all__ = [
    "Constraint",
    "CodedConstraint",
    "UnaryConstraint",
    "BinaryConstraint",
    "MultiConstraint",
    "ListConstraint",
    "LoopConstraint",
    "TernaryConstraint",
    "RangeConstraint",
    "IsaConstraint",
    "SubClassConstraint",
    "ConstraintFactory",
    "TemplateConstraint",
    "TemplateConstraintFactory",
    "TemplateUnaryConstraint",
    "TemplateBinaryConstraint",
    "TemplateListConstraint",
    "TemplateLoopConstraint",
    "TemplateTernaryConstraint",
    "TemplateMultiConstraint",
    "TemplateRangeConstraint",
    "TemplateIsaConstraint",
    "TemplateSubClassConstraint",
    "LogicNode",
    "LogicGroup",
    "LogicParser",
    "LogicParseError",
    "EmptyLogicError",
]
