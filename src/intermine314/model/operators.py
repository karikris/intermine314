import logging
from collections import OrderedDict
from typing import Mapping, Optional

from .constants import (
    _OP_IN,
    _OP_IS,
    _OP_IS_NOT,
    _OP_IS_NOT_NULL,
    _OP_IS_NULL,
    _OP_LOOKUP,
    _OP_NONE_OF,
    _OP_NOT_IN,
    _OP_ONE_OF,
)
from .errors import ModelError
from .path import Path

_COLUMN_LOG = logging.getLogger("intermine314.model.column")
_DEFAULT_COLUMN_BRANCH_CACHE_MAXSIZE = 128


class ConstraintTree(object):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def __and__(self, other):
        return ConstraintTree("AND", self, other)

    def __or__(self, other):
        return ConstraintTree("OR", self, other)

    def __iter__(self):
        for n in [self.left, self.right]:
            for subn in n:
                yield subn

    def as_logic(self, codes=None, start="A"):
        if codes is None:
            codes = (chr(c) for c in range(ord(start), ord("Z")))
        return "(%s %s %s)" % (self.left.as_logic(codes), self.op, self.right.as_logic(codes))


class ConstraintNode(ConstraintTree):
    def __init__(self, *args, **kwargs):
        self.vargs = args
        self.kwargs = kwargs

    def __iter__(self):
        yield self

    def as_logic(self, codes=None, start="A"):
        if codes is None:
            codes = (chr(c) for c in range(ord(start), ord("Z")))
        return next(codes)


class CodelessNode(ConstraintNode):
    def as_logic(self, code=None, start="A"):
        return ""


class Column(object):
    """
    A representation of a path in a query that can be constrained
    =============================================================

    Column objects allow constraints to be constructed in something
    close to a declarative style
    """

    def __init__(
        self,
        path,
        model,
        subclasses: Optional[Mapping[str, str]] = None,
        query=None,
        parent=None,
        *,
        branch_cache_maxsize=None,
    ):
        self._model = model
        self._query = query
        self._parent = parent
        if subclasses is None:
            self._subclasses = {}
        elif parent is not None and isinstance(subclasses, dict):
            self._subclasses = subclasses
        else:
            self._subclasses = dict(subclasses)
        if isinstance(path, Path):
            self._path = path
        else:
            self._path = model.make_path(path, self._subclasses)

        effective_cache_size = branch_cache_maxsize
        if effective_cache_size is None and parent is not None:
            effective_cache_size = getattr(parent, "_branch_cache_maxsize", None)
        if effective_cache_size is None:
            effective_cache_size = _DEFAULT_COLUMN_BRANCH_CACHE_MAXSIZE
        try:
            effective_cache_size = int(effective_cache_size)
        except Exception as exc:
            raise ValueError("branch_cache_maxsize must be a positive integer") from exc
        if effective_cache_size <= 0:
            raise ValueError("branch_cache_maxsize must be a positive integer")

        self._branch_cache_maxsize = effective_cache_size
        self._branch_cache_hits = 0
        self._branch_cache_misses = 0
        self._branch_cache_evictions = 0
        self._branches = OrderedDict()

    @property
    def branch_cache_stats(self):
        return {
            "size": len(self._branches),
            "maxsize": int(self._branch_cache_maxsize),
            "hits": int(self._branch_cache_hits),
            "misses": int(self._branch_cache_misses),
            "evictions": int(self._branch_cache_evictions),
        }

    def select(self, *cols):
        """
        Create a new query with this column as the base class,
        selecting the given fields.

        If no fields are given, then just this column will be selected.
        """
        q = self._model.service.new_query(str(self))
        if len(cols):
            q.select(*cols)
        else:
            q.select(self)
        return q

    def where(self, *args, **kwargs):
        """
        Create a new query based on this column,
        filtered with the given constraint.

        also available as "filter"
        """
        q = self.select()
        return q.where(*args, **kwargs)

    # Historical API alias retained at class level to avoid per-instance binding.
    filter = where

    def __len__(self):
        """
        Return the number of values in this column.
        """
        return self.select().count()

    def __iter__(self):
        """
        Iterate over the things this column represents.

        In the case of an attribute column, that is the values it may have.
        In the caseof a reference or class column,
        it is the objects that this path may refer to.
        """
        q = self.select()
        if self._path.is_attribute():
            for row in q.rows():
                yield row[0]
        else:
            for obj in q:
                yield obj

    def __getattr__(self, name):
        if name in self._branches:
            self._branch_cache_hits += 1
            branch = self._branches[name]
            self._branches.move_to_end(name)
            return branch

        self._branch_cache_misses += 1
        cld = self._path.get_class()
        if cld is not None:
            try:
                fld = cld.get_field(name)
                branch = Column(
                    str(self) + "." + name,
                    self._model,
                    self._subclasses,
                    self._query,
                    self,
                    branch_cache_maxsize=self._branch_cache_maxsize,
                )
                if len(self._branches) >= self._branch_cache_maxsize:
                    evicted_name, _evicted_value = self._branches.popitem(last=False)
                    self._branch_cache_evictions += 1
                    if _COLUMN_LOG.isEnabledFor(logging.DEBUG):
                        _COLUMN_LOG.debug(
                            "column_branch_cache_evicted path=%s evicted=%s size=%d maxsize=%d hits=%d misses=%d",
                            str(self),
                            str(evicted_name),
                            len(self._branches),
                            self._branch_cache_maxsize,
                            self._branch_cache_hits,
                            self._branch_cache_misses,
                        )
                self._branches[name] = branch
                return branch
            except ModelError as e:
                raise AttributeError(str(e))
        raise AttributeError("No attribute '" + name + "'")

    def __str__(self):
        return str(self._path)

    def _op_scalar(self, op, value):
        return ConstraintNode(str(self), op, value)

    def _op_scalarless(self, op):
        return ConstraintNode(str(self), op)

    def _op_list(self, op, values):
        return ConstraintNode(str(self), op, values)

    def _op_column(self, op, other):
        return ConstraintNode(str(self), op, str(other))

    def _op_constraint(self, op, other):
        return other.make_list_constraint(str(self), op)

    def _unsupported_operand(self, operator_name, other):
        raise TypeError("unsupported operand for operator %s: %r" % (operator_name, other))

    def _op_list_or_constraint(self, other, *, constraint_op, list_op, operator_name):
        if hasattr(other, "make_list_constraint"):
            return self._op_constraint(constraint_op, other)
        if isinstance(other, list):
            return self._op_list(list_op, other)
        self._unsupported_operand(operator_name, other)

    def _op_compare(self, other, *, none_op, column_op, constraint_op, list_op, scalar_op):
        if other is None:
            return self._op_scalarless(none_op)
        if isinstance(other, Column):
            return self._op_column(column_op, other)
        if hasattr(other, "make_list_constraint"):
            return self._op_constraint(constraint_op, other)
        if isinstance(other, list):
            return self._op_list(list_op, other)
        return self._op_scalar(scalar_op, other)

    def __mod__(self, other):
        if isinstance(other, tuple):
            return ConstraintNode(str(self), _OP_LOOKUP, *other)
        else:
            return ConstraintNode(str(self), _OP_LOOKUP, str(other))

    def __rshift__(self, other):
        return CodelessNode(str(self), str(other))

    __lshift__ = __rshift__

    def __eq__(self, other):
        return self._op_compare(
            other,
            none_op=_OP_IS_NULL,
            column_op=_OP_IS,
            constraint_op=_OP_IN,
            list_op=_OP_ONE_OF,
            scalar_op="=",
        )

    def __ne__(self, other):
        return self._op_compare(
            other,
            none_op=_OP_IS_NOT_NULL,
            column_op=_OP_IS_NOT,
            constraint_op=_OP_NOT_IN,
            list_op=_OP_NONE_OF,
            scalar_op="!=",
        )

    def __xor__(self, other):
        return self._op_list_or_constraint(
            other,
            constraint_op=_OP_NOT_IN,
            list_op=_OP_NONE_OF,
            operator_name="xor",
        )

    def in_(self, other):
        return self._op_list_or_constraint(
            other,
            constraint_op=_OP_IN,
            list_op=_OP_ONE_OF,
            operator_name="in_",
        )

    def __lt__(self, other):
        if isinstance(other, Column):
            self._parent._subclasses[str(self)] = str(other)
            self._parent._branches.clear()
            return CodelessNode(str(self), str(other))
        try:
            return self.in_(other)
        except TypeError:
            return ConstraintNode(str(self), "<", other)

    def __le__(self, other):
        if isinstance(other, Column):
            return CodelessNode(str(self), str(other))
        try:
            return self.in_(other)
        except TypeError:
            return ConstraintNode(str(self), "<=", other)

    def __gt__(self, other):
        return self._op_scalar(">", other)

    def __ge__(self, other):
        return self._op_scalar(">=", other)
