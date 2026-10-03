import re
import tempfile
from contextlib import closing, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from xml.dom import minidom
from xml.parsers.expat import ExpatError

from intermine314.compatibility import class_name, query_compatibility
from intermine314.config.runtime_defaults import get_runtime_defaults
from intermine314.config.storage_policy import (
    default_parquet_compression as _default_parquet_compression,
)
from intermine314.config.storage_policy import (
    validate_duckdb_identifier as _validate_duckdb_identifier,
)
from intermine314.config.storage_policy import (
    validate_parquet_compression as _validate_parquet_compression,
)
from intermine314.export.managed import ManagedDuckDBConnection
from intermine314.export.parquet import write_parquet_batches
from intermine314.export.query import _duckdb_source_sql
from intermine314.export.resource_profile import (
    resolve_temp_dir,
    validate_temp_dir_constraints,
)
from intermine314.parallel.policy import (
    VALID_ORDER_MODES as CANONICAL_VALID_ORDER_MODES,
)
from intermine314.parallel.policy import (
    VALID_PARALLEL_PAGINATION as CANONICAL_VALID_PARALLEL_PAGINATION,
)
from intermine314.parallel.policy import (
    VALID_PARALLEL_PROFILES as CANONICAL_VALID_PARALLEL_PROFILES,
)
from intermine314.parallel.policy import (
    apply_parallel_profile,
    normalize_order_mode,
    require_int,
    require_non_negative_int,
    require_positive_int,
    resolve_inflight_limit,
    resolve_parallel_strategy,
    resolve_prefetch,
)
from intermine314.query.constraints import (
    BinaryConstraint,
    CodedConstraint,
    Constraint,
    ConstraintFactory,
    EmptyLogicError,
    IsaConstraint,
    ListConstraint,
    LogicGroup,
    LogicNode,
    LogicParser,
    LoopConstraint,
    MultiConstraint,
    RangeConstraint,
    SubClassConstraint,
    TernaryConstraint,
    UnaryConstraint,
)
from intermine314.query.parallel_runtime import (
    PARALLEL_LOG as _PARALLEL_LOG,
)
from intermine314.query.parallel_runtime import (
    instrument_parallel_iterator,
)
from intermine314.query.pathfeatures import (
    PATH_PATTERN,
    Join,
    PathDescription,
    SortOrder,
    SortOrderList,
)
from intermine314.query.spec import (
    QuerySpec,
    _append_constraint_xml,
    _append_join_xml,
    query_spec_to_element,
    query_spec_to_formatted_xml,
    query_spec_to_xml,
)
from intermine314.service.resource_utils import (
    close_resource_quietly as _close_resource_quietly,
)
from intermine314.util import ReadableException, openAnything
from intermine314.util.deps import (
    require_duckdb as _require_duckdb,
)
from intermine314.util.deps import (
    require_polars as _require_polars,
)
from intermine314.util.logging import new_job_id

VALID_PARALLEL_PAGINATION = CANONICAL_VALID_PARALLEL_PAGINATION
VALID_PARALLEL_PROFILES = CANONICAL_VALID_PARALLEL_PROFILES
VALID_ORDER_MODES = CANONICAL_VALID_ORDER_MODES
VALID_ITER_ROW_MODES = frozenset({"dict"})
VALID_RESULT_ROW_MODES = frozenset({"dict", "rr", "list", "json", "jsonrows", "jsonobjects", "tsv", "csv", "count"})


def _validate_csv_query(query, csv_input, csv_options, start, size):
    if csv_input is None:
        if csv_options is not None:
            raise ValueError("csv_options requires csv_input")
        return
    require_non_negative_int("start", start)
    if size is not None:
        require_non_negative_int("size", size)
    for name in ("views", "constraint_dict", "uncoded_constraints", "joins", "path_descriptions", "_sort_order_list", "_logic"):
        value = getattr(query, name, None)
        populated = not value.is_empty() if callable(getattr(value, "is_empty", None)) else bool(value)
        if populated:
            raise ValueError("CSV input conflicts with query views, constraints, joins, path descriptions, sort order or logic; use an empty Query")


def _strip_wildcard(path: str) -> str:
    text = str(path).strip()
    if text.endswith(".*"):
        return text[:-2]
    return text


def _infer_root_name(path: str | None) -> str | None:
    if path is None:
        return None
    text = _strip_wildcard(class_name(path))
    if not text:
        return None
    if "." in text:
        return text.split(".", 1)[0]
    return text


def _is_valid_query_path(path: str) -> bool:
    text = _strip_wildcard(path)
    if not text:
        return False
    return bool(PATH_PATTERN.match(text))


def _path_prefix(path: str) -> str:
    text = _strip_wildcard(path)
    if "." not in text:
        return text
    return text.rsplit(".", 1)[0]


def _query_runtime_defaults():
    return get_runtime_defaults().query_defaults


def _runtime_default_parallel_page_size():
    return int(_query_runtime_defaults().default_parallel_page_size)


def _runtime_default_parallel_profile():
    return str(_query_runtime_defaults().default_parallel_profile)


def _runtime_default_parallel_ordered_mode():
    return str(_query_runtime_defaults().default_parallel_ordered_mode)


def _runtime_default_large_query_mode():
    return bool(_query_runtime_defaults().default_large_query_mode)


def _runtime_default_parallel_pagination():
    return str(_query_runtime_defaults().default_parallel_pagination)


def _runtime_default_batch_size():
    return int(_query_runtime_defaults().default_batch_size)


def _runtime_default_export_batch_size():
    return int(_query_runtime_defaults().default_export_batch_size)


def _runtime_default_parallel_workers():
    return int(_query_runtime_defaults().default_parallel_workers)


def _runtime_default_parallel_max_buffered_rows():
    return int(_query_runtime_defaults().default_parallel_max_buffered_rows)


def _runtime_default_query_thread_name_prefix():
    return str(_query_runtime_defaults().default_query_thread_name_prefix)


def _cap_inflight_limit(inflight_limit, page_size, *, max_buffered_rows=None):
    max_rows = _runtime_default_parallel_max_buffered_rows() if max_buffered_rows is None else int(max_buffered_rows)
    if max_rows <= 0:
        return inflight_limit
    max_pending_by_rows = max(1, max_rows // max(1, page_size))
    return min(inflight_limit, max_pending_by_rows)


def _resolve_staging_temp_dir(*, temp_dir, temp_dir_min_free_bytes, context):
    if temp_dir is None and temp_dir_min_free_bytes is None:
        return None
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    resolved = resolve_temp_dir(temp_dir)
    if resolved is None:
        return None
    if temp_dir_min_free_bytes is not None:
        validate_temp_dir_constraints(
            temp_dir=resolved,
            min_free_bytes=temp_dir_min_free_bytes,
            context=context,
        )
    return resolved


def _polars_from_dicts_with_full_inference(polars_module, batch):
    try:
        return polars_module.from_dicts(batch, infer_schema_length=None)
    except TypeError as exc:
        detail = str(exc)
        if "infer_schema_length" not in detail or "keyword" not in detail:
            raise
        return polars_module.from_dicts(batch)


@dataclass(frozen=True)
class ParallelOptions:
    page_size: int = field(default_factory=_runtime_default_parallel_page_size)
    max_workers: int | None = None
    ordered: bool | str | None = None
    prefetch: int | None = None
    inflight_limit: int | None = None
    profile: str = field(default_factory=_runtime_default_parallel_profile)
    large_query_mode: bool = field(default_factory=_runtime_default_large_query_mode)
    pagination: str = field(default_factory=_runtime_default_parallel_pagination)
    max_inflight_bytes_estimate: int | None = None


@dataclass(frozen=True)
class ResolvedParallelOptions:
    page_size: int
    max_workers: int
    order_mode: str
    prefetch: int
    inflight_limit: int
    profile: str
    large_query_mode: bool
    pagination: str
    start: int
    size: int | None
    strategy: str
    max_inflight_bytes_estimate: int | None
    tor_enabled: bool
    tor_state_known: bool
    tor_aware_defaults_applied: bool
    tor_source: str


class ParallelOptionsError(ValueError):
    """Raised when parallel execution options are invalid."""


def _parallel_options_error(exc: Exception) -> ParallelOptionsError:
    detail = str(exc).strip() or exc.__class__.__name__
    return ParallelOptionsError(
        "Invalid parallel options: "
        + detail
        + ". Use positive integers for page_size/max_workers/prefetch/inflight_limit/"
        + "max_inflight_bytes_estimate and valid values for "
        + "ordered/profile/pagination."
    )


class Query:
    """
    Structured query builder for InterMine services.

    This class builds query XML/params, validates paths and constraints against
    the model, and executes result iterators including bounded parallel modes.
    Full tutorials live in ``docs/source/query.rst``.
    """
    SO_SPLIT_PATTERN = re.compile("\\s*(asc|desc)\\s*", re.I)
    def __init__(self, model=None, service=None, validate=True, root=None, *, compatibility=None):
        """
        Construct a new Query
        =====================

        Construct a new query for making database queries
        against an InterMine data warehouse.

        Normally you would not need to use this constructor
        directly, but instead use the factory method on
        intermine314.webservice.Service, which will handle construction
        for you.

        @param model: an instance of L{intermine314.model.Model}. Required
        @param service: an instance of l{intermine314.service.Service}. Optional,
            but you will not be able to make requests without one.
        @param validate: a boolean - defaults to True. If set to false, the
            query will not try and validate itself. You should not set this to
            false.

        """
        self.compatibility = query_compatibility(compatibility, model=model, service=service)
        self.model = model
        self.root = self._resolve_root(_infer_root_name(root))

        self.name = ""
        self.description = ""
        self.service = service
        self.prefetch_depth = service.prefetch_depth if service is not None else 1
        self.prefetch_id_only = service.prefetch_id_only if service is not None else False
        self.do_verification = validate
        self.joins = []
        self.path_descriptions = []
        self.constraint_dict = {}
        self.uncoded_constraints = []
        self.views = []
        self._sort_order_list = SortOrderList()
        self.constraint_factory = ConstraintFactory(compatibility=self.compatibility)
        self._logic = None
        self._logic_parser = LogicParser(self)

    @classmethod
    def from_xml(cls, xml, *args, **kwargs):
        """Load one saved query from XML, a file, a URL or a borrowed stream.

        Borrowed streams remain open. Bound URLs use the service's configured
        opener; validation waits until every subclass refinement is installed.
        """
        obj = cls(*args, **kwargs)
        obj.do_verification = False
        owned = not hasattr(xml, "read")
        stream = None
        try:
            if (obj.service is not None and isinstance(xml, str)
                    and urlsplit(xml).scheme.lower() in {"http", "https", "ftp"}):
                stream = obj.service.opener.open(xml)
            else:
                stream = openAnything(xml)
            doc = minidom.parse(stream)
        except (ExpatError, OSError, ValueError) as exc:
            raise QueryParseError(f"Could not parse query XML: {exc}") from exc
        finally:
            if owned and stream is not None:
                stream.close()

        obj._load_xml_metadata(doc)
        queries = doc.getElementsByTagName("query")
        if len(queries) != 1:
            raise QueryParseError(
                "wrong number of queries in xml. Only one <query> element is allowed. "
                f"Found {len(queries)}"
            )
        query = queries[0]
        obj.name = getattr(obj, "_xml_template_name", None) or query.getAttribute("name")
        obj.description = query.getAttribute("longDescription")

        # Reserve all explicit codes before assigning missing ones, including
        # when the explicit constraint occurs later in the saved document.
        elements = query.getElementsByTagName("constraint")
        explicit = [element.getAttribute("code") for element in elements
                    if element.getAttribute("code")]
        if len(explicit) != len(set(explicit)):
            raise ConstraintError("Constraint code is already in use in query XML")
        obj.constraint_factory._used_codes.update(explicit)

        # Establish the root from the view first, but defer wildcard expansion
        # until constraints are available to resolve subclass-only fields.
        view = query.getAttribute("view")
        if obj.root is None and view.strip():
            first = re.split(r"[\s,]+", view.strip())[0]
            obj.prefix_path(first)
        for element in elements:
            constraint_args = obj._constraint_xml_arguments(element)
            code = constraint_args.get("code")
            if code is not None:
                obj.constraint_factory._used_codes.discard(code)
            try:
                obj.add_constraint(**constraint_args)
            except (TypeError, ValueError) as exc:
                raise ConstraintError(f"Invalid constraint in query XML: {exc}") from exc

        obj.add_view(view)
        for element in query.getElementsByTagName("pathDescription"):
            canonical = element.getAttribute("pathString")
            legacy = element.getAttribute("path")
            if element.hasAttribute("pathString") and element.hasAttribute("path") and canonical != legacy:
                raise QueryParseError("Conflicting pathString and path in pathDescription")
            path = canonical or legacy
            if not path:
                raise QueryParseError("Path descriptions must have a path")
            obj.add_path_description(path, element.getAttribute("description"))
        for element in query.getElementsByTagName("join"):
            obj.add_join(element.getAttribute("path"), element.getAttribute("style"))

        # Original saved queries can contain sorts for columns no longer in
        # their view. Preserve that tolerance without resolving unused paths.
        sort = query.getAttribute("sortOrder").strip()
        parts = cls.SO_SPLIT_PATTERN.split(sort)
        if len(parts) == 1:
            if sort in obj.views:
                obj.add_sort_order(sort)
        else:
            for index in range(0, len(parts) - 1, 2):
                path, direction = parts[index].strip(), parts[index + 1]
                if path in obj.views:
                    obj.add_sort_order(path, direction)
        logic = query.getAttribute("constraintLogic")
        if logic.strip():
            obj.set_logic(logic)
            obj.validate_logic()
        obj.verify()
        return obj

    def _load_xml_metadata(self, doc):
        """Subclass hook consuming the already parsed resource document."""

    @staticmethod
    def _constraint_xml_arguments(element):
        path = element.getAttribute("path")
        if not path and getattr(element.parentNode, "tagName", None) == "node":
            path = element.parentNode.getAttribute("path")
        if not path:
            raise QueryParseError("Constraints must have a path")
        arguments = {"path": path}
        for xml_name, argument in (("op", "op"), ("code", "code"), ("type", "subclass"),
                                   ("extraValue", "extra_value"), ("loopPath", "loopPath")):
            value = element.getAttribute(xml_name)
            if value:
                arguments[argument] = value
        if "op" not in arguments and "subclass" not in arguments:
            raise ConstraintError("Constraints must have an operator or subclass type")
        # Presence, rather than truthiness, preserves an explicit empty value.
        if element.hasAttribute("value"):
            arguments["value"] = element.getAttribute("value")
        if element.hasAttribute("extraValue") and arguments.get("op") == "LOOKUP":
            arguments["extra_value"] = element.getAttribute("extraValue")
        values = element.getElementsByTagName("value")
        if values:
            arguments["values"] = ["".join(node.data for node in value.childNodes
                                          if node.nodeType in (node.TEXT_NODE, node.CDATA_SECTION_NODE))
                                   for value in values]
        else:
            op = arguments.get("op", "").strip().upper()
            collection_ops = MultiConstraint.OPS | IsaConstraint.OPS | RangeConstraint.OPS
            if (op in collection_ops and "subclass" not in arguments
                    and "value" not in arguments):
                # The encoder emits no child elements for an empty collection.
                # CONTAINS is scalar only when a value attribute is present.
                arguments["values"] = []
        if "loopPath" in arguments:
            arguments["op"] = {"=": "IS", "!=": "IS NOT"}.get(arguments.get("op"), arguments.get("op"))
        if arguments.get("value") == "" and (
            "subclass" in arguments or "loopPath" in arguments or "values" in arguments
            or arguments.get("op") in UnaryConstraint.OPS
        ):
            # Empty scalar placeholders are irrelevant for constraints that
            # have no scalar value; actual binary/lookup/list empties remain.
            arguments.pop("value")
        # editable/switchable flags belong to Templates (restored separately).
        return arguments

    def _has_model(self):
        return callable(getattr(getattr(self, "model", None), "make_path", None))

    def _resolve_root(self, name):
        if name is not None and self.compatibility == "legacy" and self._has_model():
            return self.model.get_class(name)
        return name

    @property
    def rootClass(self):
        """The model root descriptor, available in the legacy profile."""
        if self.compatibility != "legacy":
            raise AttributeError("rootClass is available in the legacy profile")
        return self.root

    def _model_path(self, path):
        return self.model.make_path(path, self.get_subclass_dict())

    def __iter__(self):
        """Iterate objects in the legacy profile, dictionaries in the native profile."""
        return self.results("jsonobjects" if self.compatibility == "legacy" else "dict")

    def __len__(self):
        """Return the number of rows this query will return."""
        return self.count()

    def __str__(self):
        """Return the XML serialisation of this query"""
        return self.to_xml()

    def verify(self):
        """
        Validate the query
        ==================

        Invalid queries will fail to run, and it is not always
        obvious why. The validation routine checks to see that
        the query will not cause errors on execution, and tries to
        provide informative error messages.

        This method is called immediately after a query is fully
        deserialised.

        @raise ModelError: if the paths are invalid
        @raise QueryError: if there are errors in query construction
        @raise ConstraintError: if there are errors in constraint construction

        """
        self.verify_views()
        self.verify_constraint_paths()
        self.verify_join_paths()
        self.verify_pd_paths()
        self.validate_sort_order()
        self.do_verification = True

    def select(self, *paths):
        """
        Replace the current selection of output columns with this one
        =============================================================

        example::

           query.select("*", "proteins.name")

        This method is intended to provide an API familiar to those
        with experience of SQL or other ORM layers. This method, in
        contrast to other view manipulation methods, replaces
        the selection of output columns, rather than appending to it.

        Note that any sort orders that are no longer in the view will
        be removed.

        @param paths: The output columns to add
        """
        self.views = []
        self.add_view(*paths)
        so_elems = self._sort_order_list
        self._sort_order_list = SortOrderList()

        for so in so_elems:
            if so.path in self.views:
                self._sort_order_list.append(so)
        return self

    def add_view(self, *paths):
        """
        Add one or more views to the list of output columns
        ===================================================

        example::

            query.add_view("Gene.name Gene.organism.name")

        This is the main method for adding views to the list
        of output columns. As well as appending views, it
        will also split a single, space or comma delimited
        string into multiple paths, and flatten out lists, or any
        combination. It will also immediately try to validate
        the views.

        Output columns must be valid paths according to the
        data model, and they must represent attributes of tables

        @see: intermine314.model.Model
        @see: intermine314.model.Path
        @see: intermine314.model.Attribute
        """
        # String-only callers need no descriptor import (including native
        # queries whose optional model is unavailable or only a name stub).
        column_type = ()
        if any(not isinstance(path, str) for path in paths):
            from intermine314.model import Column

            column_type = Column

        tokens = []
        for path in paths:
            if isinstance(path, (set, list, tuple)):
                tokens.extend(path)
                continue
            if isinstance(path, column_type):
                tokens.append(path)
                continue
            tokens.extend(re.split("(?:,?\\s+|,)", str(path)))
        views_to_add = []
        for token in tokens:
            text = str(token).strip()
            if isinstance(token, column_type) and not token._path.is_attribute():
                text += ".*"
            if not text:
                continue
            view = self.prefix_path(text)
            if view.endswith(".*") and self._has_model():
                views_to_add.extend(self._expand_wildcard(view[:-2], self.prefetch_depth))
            else:
                views_to_add.append(view)
        if self.do_verification:
            self.verify_views(views_to_add)
        self.views.extend(views_to_add)

        return self

    def _expand_wildcard(self, path, depth, id_only=False):
        if depth <= 0:
            return []
        descriptor = self._model_path(path).end_class
        if descriptor is None:
            raise ConstraintError(f"{path!r} does not represent a class or reference")
        # A root subclass constraint changes the fields, but not its wire prefix.
        subclass = self.get_subclass_dict().get(path)
        if subclass:
            descriptor = self.model.get_class(subclass)
        views = ([path + ".id"] if id_only and descriptor.has_id else
                 [path + "." + field.name for field in descriptor.attributes])
        if depth > 1:
            for relation in [*descriptor.references, *descriptor.collections]:
                nested = path + "." + relation.name
                self.outerjoin(nested)
                views.extend(self._expand_wildcard(nested, depth - 1, self.prefetch_id_only))
        return views

    def prefix_path(self, path):
        text = str(class_name(path)).strip()
        if not text:
            raise QueryError("path must not be empty")
        root = class_name(self.root)
        if text == "*":
            return root + ".*" if root else text
        inferred_root = _infer_root_name(text)
        if self.root is None and inferred_root is not None:
            self.root = self._resolve_root(inferred_root)
            root = class_name(self.root)
        if root is None or text == root or text.startswith(root + "."):
            return text
        return root + "." + text

    add_column = add_view
    add_columns = add_view
    add_views = add_view
    add_to_select = add_view

    def clear_view(self):
        """
        Clear the output column list
        ============================

        Deletes all entries currently in the view list.
        """
        self.views = []

    def verify_views(self, views=None):
        """
        Check to see if the views given are valid
        =========================================

        This method checks to see if the views:
          - are valid according to the model
          - represent attributes

        @see: L{intermine314.model.Attribute}

        @raise intermine314.model.ModelError: if the paths are invalid
        @raise ConstraintError: if the paths are not attributes
        """
        if views is None:
            views = self.views
        for path in views:
            if not _is_valid_query_path(path):
                raise ConstraintError("Invalid view path: " + str(path))
            if self._has_model() and not self._model_path(path).is_attribute():
                raise ConstraintError(f"{path!r} does not represent an attribute")

    def add_constraint(self, *args, **kwargs):
        """
        Add a constraint (filter on records)
        ====================================

        example::

            query.add_constraint("Gene.symbol", "=", "zen")

        This method will try to make a constraint from the arguments
        given, trying each of the classes it knows of in turn
        to see if they accept the arguments. This allows you
        to add constraints of different types without having to know
        or care what their classes or implementation details are.
        All constraints derive from intermine314.constraints.Constraint,
        and they all have a path attribute, but are otherwise diverse.

        Before adding the constraint to the query, this method
        will also try to check that the constraint is valid by
        calling Query.verify_constraint_paths()

        @see: L{intermine314.constraints}

        @rtype: L{intermine314.constraints.Constraint}
        """
        from intermine314.model import CodelessNode, Column

        if len(args) == 1 and not kwargs:
            only = args[0]
            if isinstance(only, tuple):
                args = only
            elif isinstance(only, CodelessNode):
                args, kwargs = only.vargs, only.kwargs
                if len(args) == 2 and not kwargs:
                    args, kwargs = (), dict(path=args[0], subclass=args[1])
            elif hasattr(only, "vargs") and hasattr(only, "kwargs"):
                args, kwargs = only.vargs, only.kwargs

        # Column.name is itself a navigable branch, so passing a Column to
        # PathFeature's descriptor-name protocol would resolve the wrong path.
        args = tuple(str(arg) if isinstance(arg, Column) else arg for arg in args)
        kwargs = {key: str(value) if isinstance(value, Column) else value
                  for key, value in kwargs.items()}
        if len(args) == 1 and not kwargs:
            con = args[0]
        elif len(args) == 0 and len(kwargs) == 1:
            k, v = list(kwargs.items())[0]
            if self.compatibility == "legacy" and isinstance(v, str) and v in UnaryConstraint.OPS:
                con = self.constraint_factory.make_constraint(k, v)
            elif isinstance(v, (list, tuple, set)):
                con = self.constraint_factory.make_constraint(k, "ONE OF", list(v))
            else:
                con = self.constraint_factory.make_constraint(k, "=", v)
        else:
            # Original reference operators may omit the object path when a
            # legacy query already has a root. Native scalar/path shorthand
            # keeps its existing interpretation, even for operator-like text.
            if (self.compatibility == "legacy" and args and isinstance(args[0], str)
                    and args[0].strip().upper() in self.constraint_factory.reference_ops):
                args = (class_name(self.root), *args)
            con = self.constraint_factory.make_constraint(*args, **kwargs)

        con.path = self.prefix_path(con.path)
        if isinstance(con, LoopConstraint):
            con.loopPath = self.prefix_path(con.loopPath)
        if self.do_verification:
            self.verify_constraint_paths([con])
        if hasattr(con, "code"):
            if con.code in self.constraint_dict:
                raise ConstraintError(f"Constraint code {con.code!r} is already in use")
            self.constraint_factory._used_codes.add(con.code)
            self.constraint_dict[con.code] = con
        else:
            self.uncoded_constraints.append(con)

        return con

    def where(self, *cons, **kwargs):
        """
        Return a new query like this one but with an additional constraint
        ==================================================================

        In contrast to add_constraint, this method returns
        a new object with the given comstraint added, it does not
        mutate the Query it is invoked on.

        """
        from copy import deepcopy

        from intermine314.model import (
            CodelessNode,
            Column,
            ConstraintNode,
            ConstraintTree,
        )

        c = self.clone()
        previous = c.get_logic()
        # Only a positional path selects a constructor call. Keyword-only
        # calls always mean field=value, including path/op/subclass fields.
        constructor_call = bool(cons and isinstance(cons[0], (str, Column)))
        if constructor_call:
            groups = [c.add_constraint(*cons, **kwargs)]
        else:
            # Refinements apply unconditionally, including under OR. Install
            # them before attribute leaves so subclass fields validate even
            # when the refinement is the right-hand expression branch.
            for expression in cons:
                if isinstance(expression, ConstraintTree):
                    for node in expression:
                        if isinstance(node, CodelessNode):
                            c.add_constraint(node)

            def bind(expression):
                if isinstance(expression, CodelessNode):
                    return None
                if isinstance(expression, ConstraintTree) and not isinstance(expression, ConstraintNode):
                    left, right = bind(expression.left), bind(expression.right)
                    if left is None or right is None:
                        return right if left is None else left
                    return LogicGroup(left, expression.op, right)
                # Bind every occurrence to its actual factory-created code;
                # explicit reservations and codes beyond Z need no guessing.
                constraint = c.add_constraint(
                    deepcopy(expression) if isinstance(expression, Constraint) else expression)
                return constraint if isinstance(constraint, CodedConstraint) else None

            groups = [bind(expression) for expression in cons]
            for path, value in kwargs.items():
                op = "ONE OF" if isinstance(value, (list, tuple, set)) else "="
                groups.append(c.add_constraint(path, op, value))

        logic = previous if isinstance(previous, LogicNode) else None
        for group in groups:
            if isinstance(group, CodedConstraint) or isinstance(group, LogicGroup):
                logic = group if logic is None else LogicGroup(logic, "AND", group)

        def contains_or(node):
            return isinstance(node, LogicGroup) and (
                node.op == "OR" or contains_or(node.left) or contains_or(node.right))

        # Preserve dynamic default AND for flat/all-AND filters and empty
        # clones, so later add_constraint calls still enter the query logic.
        # OR grouping and pre-existing explicit logic require a stored tree.
        if logic is not None and (c._logic is not None or contains_or(logic)):
            c.set_logic(logic)
        return c

    filter = where

    def where_eq(self, path, value):
        """Return a cloned query with an equality constraint."""
        return self.where((path, "=", value))

    def where_in(self, path, values):
        """Return a cloned query with collection membership in either profile."""
        return self.where((path, "ONE OF", values))

    def where_raw(self, op, path, value=None):
        """Return a cloned query with a raw operator constraint."""
        c = self.clone()
        if value is None:
            c.add_constraint(path, op)
        else:
            c.add_constraint(path, op, value)
        return c

    def column(self, col):
        """
        Return a Column object suitable for using to construct constraints with
        =======================================================================

        This method is part of the SQLAlchemy style API.

        """
        path = self.prefix_path(str(col))
        if self.compatibility == "legacy":
            if not self._has_model():
                raise QueryError("Legacy columns require a Model")
            return self.model.column(path, self.get_subclass_dict(), self)
        return path

    c = column

    def verify_constraint_paths(self, cons=None):
        """Validate syntax and Model-aware paths for selected constraints.

        With a Model, validate attribute targets, object/loop targets, class
        names and subclass relationships. Without a Model, validate path syntax.
        Range constraints retain server-specific semantics.

        :param cons: Constraints to check, defaulting to all query constraints.
        :raises ModelError: If a path cannot be resolved.
        :raises ConstraintError: If a constraint targets an incompatible field.
        """
        if cons is None:
            cons = self.constraints
        for con in cons:
            if not _is_valid_query_path(con.path):
                raise ConstraintError("Invalid constraint path: " + str(con.path))
            if isinstance(con, SubClassConstraint) and not _is_valid_query_path(con.subclass):
                raise ConstraintError("Invalid subclass path: " + str(con.subclass))
            if isinstance(con, LoopConstraint) and not _is_valid_query_path(con.loopPath):
                raise ConstraintError("Invalid loop path: " + str(con.loopPath))
            if self._has_model():
                path = self._model_path(con.path)
                if isinstance(con, RangeConstraint):
                    continue
                if isinstance(con, (IsaConstraint, TernaryConstraint, ListConstraint, LoopConstraint)):
                    if path.end_class is None:
                        raise ConstraintError(f"{con.path!r} does not refer to an object")
                    if isinstance(con, IsaConstraint):
                        for name in con.values:
                            if name not in self.model.classes:
                                raise ConstraintError(f"{name!r} is not a class in this model")
                    elif isinstance(con, LoopConstraint):
                        other = self._model_path(con.loopPath)
                        if other.end_class is None:
                            raise ConstraintError(f"{con.loopPath!r} does not refer to an object")
                        if not path.end_class.isa(other.end_class) and not other.end_class.isa(path.end_class):
                            raise ConstraintError("Loop classes are of incompatible types")
                elif isinstance(con, (BinaryConstraint, MultiConstraint)) and not path.is_attribute():
                    raise ConstraintError(f"{con.path!r} does not represent an attribute")
                if isinstance(con, SubClassConstraint):
                    # Validate against the declared path, excluding its own
                    # existing refinement when verify() rechecks the query.
                    subclasses = self.get_subclass_dict()
                    subclasses.pop(con.path, None)
                    base = self.model.make_path(con.path, subclasses).end_class
                    child = self.model.get_class(con.subclass)
                    if base is None or not child.isa(base):
                        raise ConstraintError(f"{con.subclass!r} is not a subclass of {con.path!r}")

    @property
    def constraints(self):
        """
        Returns the constraints of the query
        ====================================

        Query.constraints S{->} list(intermine314.constraints.Constraint)

        Constraints are returned in the order of their code (normally
        the order they were added to the query) and with any
        subclass contraints at the end.

        @rtype: list(Constraint)
        """
        ret = sorted(list(self.constraint_dict.values()), key=lambda con: con.code)
        ret.extend(self.uncoded_constraints)
        return ret

    def get_constraint(self, code):
        """
        Returns the constraint with the given code
        ==========================================

        Returns the constraint with the given code, if if exists.
        If no such constraint exists, it throws a ConstraintError

        @return: the constraint corresponding to the given code
        @rtype: L{intermine314.constraints.CodedConstraint}
        """
        if code in self.constraint_dict:
            return self.constraint_dict[code]
        else:
            raise ConstraintError("There is no constraint with the code '" + code + "' on this query")

    def add_join(self, *args, **kwargs):
        """
        Add a join statement to the query
        =================================

        example::

         query.add_join("Gene.proteins", "OUTER")

        A join statement is used to determine if references should
        restrict the result set by only including those references
        exist. For example, if one had a query with the view::

          "Gene.name", "Gene.proteins.name"

        Then in the normal case (that of an INNER join), we would only
        get Genes that also have at least one protein that they reference.
        Simply by asking for this output column you are placing a
        restriction on the information you get back.

        If in fact you wanted all genes, regardless of whether they had
        proteins associated with them or not, but if they did
        you would rather like to know _what_ proteins, then you need
        to specify this reference to be an OUTER join::

         query.add_join("Gene.proteins", "OUTER")

        Now you will get many more rows of results, some of which will
        have "null" values where the protein name would have been,

        This method will also attempt to validate the join by calling
        Query.verify_join_paths(). Joins must have a valid path, the
        style can be either INNER or OUTER (defaults to OUTER,
        as the user does not need to specify inner joins, since all
        references start out as inner joins), and the path
        must be a reference.

        @raise ModelError: if the path is invalid
        @raise TypeError: if the join style is invalid

        @rtype: L{intermine314.pathfeatures.Join}
        """
        join = Join(*args, **kwargs)
        join.path = self.prefix_path(join.path)
        if self.do_verification:
            self.verify_join_paths([join])
        self.joins.append(join)
        return self

    def outerjoin(self, column):
        """Alias for add_join(column, "OUTER")"""
        return self.add_join(str(column), "OUTER")

    def add_path_description(self, *args, **kwargs):
        """Add and return a validated display description for a model path."""
        description = PathDescription(*args, **kwargs)
        description.path = self.prefix_path(description.path)
        if self.do_verification:
            self.verify_pd_paths([description])
        self.path_descriptions.append(description)
        return description

    def verify_pd_paths(self, pds=None):
        """Validate path descriptions with the query's subclass refinements."""
        for description in self.path_descriptions if pds is None else pds:
            if not _is_valid_query_path(description.path):
                raise QueryError("Invalid path description path: " + str(description.path))
            if self._has_model():
                self._model_path(description.path)

    def verify_join_paths(self, joins=None):
        """
        Check that the joins are valid
        ==============================

        Joins must have valid paths, and they must refer to references.

        @raise ModelError: if the paths are invalid
        @raise QueryError: if the paths are not references
        """
        if joins is None:
            joins = self.joins
        for join in joins:
            if not _is_valid_query_path(join.path):
                raise QueryError("Invalid join path: " + str(join.path))
            if self._has_model() and not self._model_path(join.path).is_reference():
                raise QueryError(f"{join.path!r} does not represent a reference")

    @property
    def coded_constraints(self):
        """Return constraints that carry an explicit code."""
        return sorted(list(self.constraint_dict.values()), key=lambda con: con.code)

    def get_logic(self):
        """Return explicit logic or dynamically AND all coded constraints."""
        if self._logic is not None:
            return self._logic
        coded = self.coded_constraints
        if not coded:
            return ""
        logic = coded[0]
        for constraint in coded[1:]:
            logic = logic + constraint
        return logic

    def set_logic(self, value):
        """Set a logic node or parse a string using historical OR precedence."""
        if isinstance(value, LogicNode) and callable(getattr(value, "get_codes", None)):
            logic = value
        else:
            try:
                logic = self._logic_parser.parse(value)
            except EmptyLogicError:
                if self.coded_constraints:
                    raise
                self._logic = None
                return self
        if self.do_verification:
            self.validate_logic(logic)
        self._logic = logic
        return self

    def validate_logic(self, logic=None):
        """Require all coded constraints and reject unknown constraint codes."""
        if logic is None:
            logic = self.get_logic()
        if isinstance(logic, str):
            logic = self._logic_parser.parse(logic) if logic.strip() else ""
        if logic == "":
            logic_codes = set()
        elif isinstance(logic, LogicNode) and callable(getattr(logic, "get_codes", None)):
            logic_codes = set(logic.get_codes())
        else:
            raise TypeError("Constraint logic must be a string or logic node")
        unknown = logic_codes - set(self.constraint_dict)
        if unknown:
            raise QueryError("Unknown constraint code in logic: " + ", ".join(sorted(unknown)))
        for con in self.coded_constraints:
            if con.code not in logic_codes:
                raise QueryError(f"Constraint {con.code}{con!r} is not mentioned in the logic: {logic}")

    logic = property(get_logic, set_logic)

    def get_default_sort_order(self):
        """
        Gets the sort order when none has been specified
        ================================================

        This method is called to determine the sort order if
        none is specified

        @raise QueryError: if the view is empty

        @rtype: L{intermine314.pathfeatures.SortOrderList}
        """
        try:
            v0 = self.views[0]
            for j in self.joins:
                if j.style == "OUTER":
                    if v0.startswith(j.path):
                        return ""
            return SortOrderList((self.views[0], SortOrder.ASC))
        except IndexError:
            raise QueryError("Query view is empty")

    def get_sort_order(self):
        """
        Return a sort order for the query
        =================================

        This method returns the sort order if set, otherwise
        it returns the default sort order

        @raise QueryError: if the view is empty

        @rtype: L{intermine314.pathfeatures.SortOrderList}
        """
        if self._sort_order_list.is_empty():
            return self.get_default_sort_order()
        else:
            return self._sort_order_list

    def add_sort_order(self, path, direction=SortOrder.ASC):
        """
        Adds a sort order to the query
        ==============================

        example::

          Query.add_sort_order("Gene.name", "DESC")

        This method adds a sort order to the query.
        A query can have multiple sort orders, which are
        assessed in sequence.

        If a query has two sort-orders, for example,
        the first being "Gene.organism.name asc",
        and the second being "Gene.name desc", you would have
        the list of genes grouped by organism, with the
        lists within those groupings in reverse alphabetical
        order by gene name.

        This method will try to validate the sort order
        by calling validate_sort_order()

        """
        so = SortOrder(str(path), direction)
        so.path = self.prefix_path(so.path)
        if self.do_verification:
            self.validate_sort_order(so)
        self._sort_order_list.append(so)
        return self

    def validate_sort_order(self, *so_elems):
        """
        Check the validity of the sort order
        ====================================

        Checks that the sort order paths are:
          - valid paths
          - in the view

        @raise QueryError: if the sort order is not in the view
        @raise ModelError: if the path is invalid

        """
        if not so_elems:
            so_elems = self._sort_order_list
        from_paths = self._from_paths()
        for so in so_elems:
            if not _is_valid_query_path(so.path):
                raise QueryError("Invalid sort order path: " + str(so.path))
            if self._has_model() and not self._model_path(so.path).is_attribute():
                raise QueryError(f"{so.path!r} does not represent an attribute")
            if _path_prefix(so.path) not in from_paths:
                raise QueryError(f"Sort order element {so.path} is not in the query")

    order_by = add_sort_order

    def _from_paths(self):
        froms = set()
        for view in self.views:
            froms.add(_path_prefix(view))
        for c in self.constraints:
            froms.add(_path_prefix(c.path))
        return froms

    def get_subclass_dict(self):
        """
        Return the current mapping of class to subclass
        ===============================================

        This method returns a mapping of classes used
        by the model for assessing whether certain paths are valid. For
        intance, if you subclass MicroArrayResult to be FlyAtlasResult,
        you can refer to the .presentCall attributes of fly atlas results.
        MicroArrayResults do not have this attribute, and a path such as::

          Gene.microArrayResult.presentCall

        would be marked as invalid unless the dictionary is provided.

        Users most likely will not need to ever call this method.

        @rtype: dict(string, string)
        """
        subclass_dict = {}
        for c in self.constraints:
            if isinstance(c, SubClassConstraint):
                subclass_dict[c.path] = c.subclass
        return subclass_dict

    def results(self, row=None, start=0, size=None, summary_path=None):
        """
        Return an iterator over result rows
        ===================================

        Usage::

          >>> query = service.model.Gene.select("symbol", "length")
          >>> for d in query.results(row="dict"):
          ...    print(d["Gene.symbol"])

        Formats include ``rr``, ``list``, ``dict``, raw ``json``/``jsonrows``,
        and streamed ``tsv``/``csv``/``count``. Legacy queries default to model
        objects; native queries default to ``dict``. Object aliases request
        ``jsonobjects`` from the server. ``dataframe`` is a dictionary-row
        iterator alias; use ``dataframe()`` to materialize a Polars frame.
        A summary path overrides the row format with raw ``jsonrows``.

        If no views have been specified, all attributes of the root class
        are selected for output.

        @param row: The format for each result.
        @type row: string
        @param start: the index of the first result to return (default = 0)
        @type start: int
        @param size: The maximum number of results to return (default = all)
        @type size: int
        @rtype: L{intermine314.results.ResultIterator}

        @raise WebserviceError: if the request is unsuccessful
        """

        if summary_path is not None:
            row = "jsonrows"
        elif row is None:
            row = "jsonobjects" if self.compatibility == "legacy" else "dict"
        if row == "dataframe":
            row = "dict"
        if row.startswith("object"):
            row = "jsonobjects"
        to_run = self.clone()
        if summary_path is not None:
            summary_path = to_run.prefix_path(summary_path)

        if len(to_run.views) == 0:
            to_run.add_view(class_name(to_run.root) + ".*" if Query._has_model(to_run) else to_run.root)

        if row not in VALID_RESULT_ROW_MODES:
            choices = ", ".join(sorted(VALID_RESULT_ROW_MODES))
            raise ValueError(f"row must be one of: {choices}")

        if row == "jsonobjects":
            if not Query._has_model(to_run):
                from intermine314.model import ModelError

                raise ModelError("Object results require a valid query model")
            for constraint in to_run.coded_constraints:
                path = to_run._model_path(constraint.path)
                parent = path.prefix() if path.is_attribute() else path
                prefix = str(parent) + "."
                if not any(view.startswith(prefix) for view in to_run.views):
                    to_run.add_view(str(path) if path.is_attribute() else str(path) + ".id")

        resolver = getattr(to_run, "_to_execution", None)
        execution = resolver() if callable(resolver) else None
        if execution is not None:
            options = {"cld": to_run.model.get_class(class_name(to_run.root))} if row == "jsonobjects" else {}
            if summary_path is not None:
                options["summary_path"] = summary_path
            return execution.results(row=row, start=start, size=size, **options)

        path = to_run.get_results_path()
        params = to_run.to_query_params()
        params["start"] = start
        if size is not None:
            params["size"] = size
        if summary_path is not None:
            params["summaryPath"] = summary_path

        view = to_run.views
        cld = to_run.model.get_class(class_name(to_run.root)) if row == "jsonobjects" else to_run.root
        return to_run.service.get_results(path, params, row, view, cld)

    def summarise(self, summary_path, **kwargs):
        """Return first-row float statistics or a category-to-count mapping.

        Model.NUMERIC_TYPES determines numeric columns. Empty numeric summaries
        raise StopIteration; empty categorical summaries return an empty dict.
        Result options pass through, and the stream closes on every exit path.
        """
        from intermine314.model import Model

        path = self._model_path(self.prefix_path(summary_path))
        stream = self.results(summary_path=summary_path, **kwargs)
        try:
            if path.end.type_name in Model.NUMERIC_TYPES:
                return {key: float(value) for key, value in next(stream).items()}
            return {row["item"]: row["count"] for row in stream}
        finally:
            _close_resource_quietly(stream)

    summarize = summarise

    def _first_object(self):
        """Consume one complete object, closing its stream without a row limit.

        Server size limits apply to joined rows and can truncate collections.
        Lazy fetching shares the public first-result implementation.
        """
        return self.first()

    def first(self, row="jsonobjects", start=0, **kw):
        """Return the first result, or None, and close the managed stream.

        Both profiles default to objects for this historical helper. Object
        formats omit a size limit because joined rows can truncate collections.
        Other formats request one row. Additional options pass to ``results``.
        """
        if isinstance(row, str) and row.startswith("object"):
            row = "jsonobjects"
        stream = self.results(row, start=start, size=None if row == "jsonobjects" else 1, **kw)
        try:
            return next(stream, None)
        finally:
            _close_resource_quietly(stream)

    def one(self, row="jsonobjects"):
        """Return exactly one result, raising QueryError for other cardinalities.

        A server count measures joined rows. For object formats, counts other
        than one therefore require checking actual top-level object results.
        """
        if isinstance(row, str) and row.startswith("object"):
            row = "jsonobjects"
        count = self.count()
        if row != "jsonobjects":
            if count != 1:
                raise QueryError(f"Result size is not one: got {count} results")
            return self.first(row)
        if count == 1:
            return self.first(row)
        stream = self.results(row)
        try:
            first = next(stream, None)
            if first is None:
                raise QueryError("No results received")
            if next(stream, None) is not None:
                raise QueryError("More than one result received")
            return first
        finally:
            _close_resource_quietly(stream)

    def get_results_list(self, *args, **kwargs):
        """Eagerly consume ``results`` once, forwarding all arguments/options.

        A comprehension avoids ResultIterator's fresh-HTTP length hint.
        """
        stream = self.results(*args, **kwargs)
        try:
            return [result for result in stream]
        finally:
            _close_resource_quietly(stream)

    all = get_results_list

    def get_row_list(self, start=0, size=None):
        """Return eager rows: legacy ResultRows or native dictionaries."""
        row = "rr" if self.compatibility == "legacy" else "dict"
        return self.get_results_list(row, start, size)

    def _iter_result_rows(
        self,
        start=0,
        size=None,
        row="dict",
        parallel_options=None,
    ):
        if parallel_options is not None:
            options = self._coerce_parallel_options(parallel_options=parallel_options)
            return self.run_parallel(
                row=row,
                start=start,
                size=size,
                parallel_options=options,
            )
        return self.results(row=row, start=start, size=size)

    def iter_rows(
        self,
        start=0,
        size=None,
        mode="dict",
        *,
        parallel_options=None,
    ):
        """
        Yield rows in exporter-friendly modes.

        ``mode="dict"`` is optimized for exporter-style pipelines.
        """
        if mode not in VALID_ITER_ROW_MODES:
            choices = ", ".join(sorted(VALID_ITER_ROW_MODES))
            raise ValueError(f"mode must be one of: {choices}")
        return self._iter_result_rows(
            start=start,
            size=size,
            row=mode,
            parallel_options=parallel_options,
        )

    def iter_batches(
        self,
        start=0,
        size=None,
        batch_size=None,
        row_mode="dict",
        *,
        parallel_options=None,
    ):
        """
        Yield result rows as lists of dicts in batches.

        Usage::
          >>> for batch in query.iter_batches(batch_size=2000):
          ...     process_batch(batch)
        """
        if batch_size is None:
            batch_size = _runtime_default_batch_size()
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if row_mode not in VALID_ITER_ROW_MODES:
            choices = ", ".join(sorted(VALID_ITER_ROW_MODES))
            raise ValueError(f"row_mode must be one of: {choices}")
        batch = []
        row_iter = self.iter_rows(
            start=start,
            size=size,
            mode=row_mode,
            parallel_options=parallel_options,
        )
        try:
            for row in row_iter:
                batch.append(row)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        finally:
            _close_resource_quietly(row_iter)

    def rows(self, start=0, size=None, row=None):
        """
        Return the results as rows of data
        ==================================

        Usage::

          >>> for row in query.rows(start=10, size=10):
          ...     print(row["proteins.name"])

        @param start: the index of the first result to return (default = 0)
        @type start: int
        @param size: The maximum number of results to return (default = all)
        @type size: int
        Legacy queries default to ``rr``; native queries default to ``dict``.
        The optional third argument selects an explicit result format.

        @rtype: iterable
        """
        if row is None:
            row = "rr" if self.compatibility == "legacy" else "dict"
        return self.results(row=row, start=start, size=size)

    def to_parquet(
        self,
        path,
        start=0,
        size=None,
        batch_size=None,
        compression=None,
        single_file=False,
        *,
        temp_dir=None,
        temp_dir_min_free_bytes=None,
        parallel_options=None,
        csv_input=None,
        csv_options=None,
    ):
        """
        Stream results to Parquet files.

        CSV input uses local Polars scan options and keeps CSV column names.
        Views, constraints, joins and sort order must be empty. Attached
        root/model/service identity has no role in CSV mode, which makes no HTTP
        requests. Directory output reads bounded Arrow batches from temporary
        Parquet; single-file output sinks directly without collecting a frame.

        Usage::
          >>> query.to_parquet(
          ...     "results_parquet",
          ...     batch_size=5000,
          ...     parallel_options=ParallelOptions(max_workers=8),
          ... )
        """
        _validate_csv_query(self, csv_input, csv_options, start, size)
        polars_module = _require_polars("Query.to_parquet()")
        if batch_size is None:
            batch_size = _runtime_default_export_batch_size()
        options = self._coerce_parallel_options(parallel_options=parallel_options)
        compression = _validate_parquet_compression(
            _default_parquet_compression() if compression is None else compression
        )
        batch_size = require_positive_int("batch_size", batch_size)
        staging_dir = _resolve_staging_temp_dir(
            temp_dir=temp_dir,
            temp_dir_min_free_bytes=temp_dir_min_free_bytes,
            context="Query.to_parquet() staging",
        )
        if csv_input is not None:
            from intermine314.export.csv import _export_csv

            return _export_csv(
                csv_input, path, csv_options=csv_options, start=start, size=size,
                compression=compression, single_file=single_file,
                staging_dir=staging_dir, batch_size=batch_size,
                polars_module=polars_module,
            )
        schema = self._parquet_schema()
        to_run = self
        if self._has_model():
            to_run = self.clone()
            if not to_run.views and to_run.root is not None:
                to_run.add_view(class_name(to_run.root) + ".*")
                schema = to_run._parquet_schema()
            to_run._decimal_paths = tuple(getattr(schema, "decimal_paths", ()))
        return write_parquet_batches(
            batches=to_run.iter_batches(
                start=start, size=size, batch_size=batch_size, row_mode="dict", parallel_options=options,
            ),
            target=path,
            columns=to_run.views,
            polars_module=polars_module,
            compression=compression,
            single_file=single_file,
            staging_dir=staging_dir,
            batch_size=batch_size,
            schema=schema,
        )

    def _parquet_schema(self):
        """Resolve selected model attributes lazily for analytical exports."""
        from intermine314.export.model_schema import query_schema

        return query_schema(self)

    def export(
        self, path, *, format="parquet", start=0, size=None, batch_size=None,
        compression=None, single_file=True, temp_dir=None,
        temp_dir_min_free_bytes=None, parallel_options=None, csv_input=None,
        csv_options=None,
    ):
        """Atomically export one Parquet file by default, or explicit CSV.

        Only ``format='csv'`` creates CSV output; suffixes never select a format.
        A .csv suffix conflicts with Parquet and .parquet conflicts with CSV
        (case insensitive). ``single_file=False`` requests partitioned Parquet
        and is rejected for CSV. Existing ``to_parquet`` directory defaults stay
        unchanged. Return the written local path with ~ expanded.

        All compression values describe Parquet, including the managed temporary
        Parquet used for CSV output. CSV itself is always uncompressed UTF-8,
        comma separated with a header, standard quoting and empty null fields.
        ``csv_options`` controls CSV input parsing only. Input streams stay open
        and CSV mode requires empty views, constraints, joins and sort order.

        Batch, pagination, parallel and temporary storage controls pass through
        to Parquet production. CSV COPY streams through a memory-limited DuckDB
        connection with spill storage in the managed temporary directory.
        Final publication stages on the output filesystem; errors and interrupts
        preserve existing output. This provides exception safety, not crash
        durability or coordination between concurrent writers. Empty results
        preserve selected headers and Model-derived types.
        """
        from intermine314.export.csv import _check_export_collision, _local_path

        if not isinstance(format, str) or format.lower() not in ("parquet", "csv"):
            raise ValueError("format must be 'parquet' or 'csv'")
        format = format.lower()
        target = _local_path(path, "Export output")
        if (format == "parquet" and target.suffix.lower() == ".csv") or (
            format == "csv" and target.suffix.lower() == ".parquet"
        ):
            raise ValueError(f"Output suffix {target.suffix!r} conflicts with format={format!r}")
        if not isinstance(single_file, bool):
            raise TypeError("single_file must be a boolean")
        if format == "csv" and not single_file:
            raise ValueError("CSV format requires single_file=True")
        if target.is_symlink():
            raise ValueError("Export output must not be a symbolic link")
        if target.exists() and not (target.is_file() if single_file else target.is_dir()):
            raise ValueError("Export output must be a file" if single_file else "Export output must be a directory")
        _validate_csv_query(self, csv_input, csv_options, start, size)
        require_non_negative_int("start", start)
        if size is not None:
            require_non_negative_int("size", size)
        compression = _validate_parquet_compression(compression)
        if batch_size is None:
            batch_size = _runtime_default_export_batch_size()
        batch_size = require_positive_int("batch_size", batch_size)
        options = self._coerce_parallel_options(parallel_options=parallel_options)
        if csv_input is not None:
            source = _local_path(csv_input, "CSV input") if isinstance(csv_input, (str, Path)) else csv_input
            _check_export_collision(source, target, single_file=single_file)
        controls = dict(
            start=start, size=size, batch_size=batch_size,
            compression=compression, temp_dir=temp_dir,
            temp_dir_min_free_bytes=temp_dir_min_free_bytes,
            parallel_options=options, csv_input=csv_input, csv_options=csv_options,
        )
        if format == "parquet":
            return self.to_parquet(target, single_file=single_file, **controls)

        from intermine314.export.output import write_csv_from_parquet

        staging_dir = _resolve_staging_temp_dir(
            temp_dir=temp_dir, temp_dir_min_free_bytes=temp_dir_min_free_bytes,
            context="Query.export() staging",
        )
        with tempfile.TemporaryDirectory(prefix="intermine314-export-", dir=staging_dir) as scratch:
            parquet_path = self.to_parquet(
                Path(scratch) / "results.parquet", single_file=True, **controls,
            )
            return write_csv_from_parquet(parquet_path, target, scratch=scratch)

    def dataframe(
        self, start=0, size=None, *, csv_input=None, csv_options=None,
        parquet_path=None,
    ):
        """Return detached Polars results through Parquet, DuckDB and Arrow.

        With no path, temporary Parquet is removed on success, failure or
        interruption. An explicit path persists. CSV headers retain their names;
        CSV mode requires empty views, constraints, joins and sort order. Attached
        root/model/service identity has no role in CSV mode and makes no requests.
        """
        from intermine314.export.query import query_parquet

        _validate_csv_query(self, csv_input, csv_options, start, size)
        storage = tempfile.TemporaryDirectory(prefix="intermine314-dataframe-") if parquet_path is None else nullcontext()
        with storage as scratch:
            path = Path(scratch) / "results.parquet" if parquet_path is None else Path(parquet_path)
            written_path = self.to_parquet(
                path, start=start, size=size, single_file=True,
                csv_input=csv_input, csv_options=csv_options,
            )
            return query_parquet(Path(written_path) if written_path is not None else path)

    def to_duckdb(
        self,
        path,
        start=0,
        size=None,
        batch_size=None,
        compression=None,
        single_file=False,
        database=":memory:",
        table="results",
        *,
        temp_dir=None,
        temp_dir_min_free_bytes=None,
        parallel_options=None,
        managed=False,
        csv_input=None,
        csv_options=None,
    ):
        """
        Materialize results to Parquet and expose them via DuckDB.

        Usage::
          >>> con = query.to_duckdb(
          ...     "results_parquet",
          ...     parallel_options=ParallelOptions(max_workers=8),
          ... )
          >>> con.execute("select count(*) from results").fetchall()

        Deterministic cleanup::

          >>> with query.to_duckdb("results_parquet", managed=True) as con:
          ...     con.execute("select count(*) from results").fetchall()
        """
        _validate_csv_query(self, csv_input, csv_options, start, size)
        duckdb_module = _require_duckdb("Query.to_duckdb()")
        if batch_size is None:
            batch_size = _runtime_default_export_batch_size()
        options = self._coerce_parallel_options(parallel_options=parallel_options)
        table = _validate_duckdb_identifier(table)
        parquet_path = self.to_parquet(
            path,
            start=start,
            size=size,
            batch_size=batch_size,
            compression=compression,
            single_file=single_file,
            temp_dir=temp_dir,
            temp_dir_min_free_bytes=temp_dir_min_free_bytes,
            parallel_options=options,
            csv_input=csv_input,
            csv_options=csv_options,
        )
        parquet_glob_sql = _duckdb_source_sql(parquet_path, "Query.to_duckdb()")
        con = duckdb_module.connect(database=database)
        try:
            con.execute(f'CREATE OR REPLACE VIEW "{table}" AS SELECT * FROM read_parquet({parquet_glob_sql})')
        except BaseException:
            _close_resource_quietly(con)
            raise
        if managed:
            return ManagedDuckDBConnection(con, close_resource_quietly=_close_resource_quietly)
        return con

    def _run_parallel_offset(
        self,
        row="dict",
        start=0,
        size=None,
        page_size=None,
        max_workers=None,
        order_mode="ordered",
        inflight_limit=None,
        max_inflight_bytes_estimate=None,
        job_id=None,
    ):
        if page_size is None:
            page_size = _runtime_default_parallel_page_size()
        if inflight_limit is None:
            inflight_limit = _runtime_default_parallel_workers()
        from intermine314.query import parallel_offset

        return parallel_offset.run_parallel_offset(
            self,
            row=row,
            start=start,
            size=size,
            page_size=page_size,
            max_workers=max_workers,
            order_mode=order_mode,
            inflight_limit=inflight_limit,
            max_inflight_bytes_estimate=max_inflight_bytes_estimate,
            job_id=job_id,
            thread_name_prefix=_runtime_default_query_thread_name_prefix(),
            executor_cls=parallel_offset.ThreadPoolExecutor,
        )

    def _resolve_parallel_strategy(self, pagination, start, size):
        return resolve_parallel_strategy(
            pagination,
            start,
            size,
            valid_parallel_pagination=VALID_PARALLEL_PAGINATION,
        )

    def _normalize_order_mode(self, ordered):
        return normalize_order_mode(
            ordered,
            default_order_mode=_runtime_default_parallel_ordered_mode(),
            valid_order_modes=VALID_ORDER_MODES,
        )

    def _apply_parallel_profile(self, profile, ordered, large_query_mode):
        return apply_parallel_profile(
            profile,
            ordered,
            large_query_mode,
            default_profile=_runtime_default_parallel_profile(),
            valid_parallel_profiles=VALID_PARALLEL_PROFILES,
        )

    def _resolve_effective_workers(self, max_workers, size):
        if max_workers is not None:
            return max_workers
        _ = size
        return _runtime_default_parallel_workers()

    def _resolve_tor_parallel_context(self):
        service = getattr(self, "service", None)
        if service is None:
            return False, False, "no_service"
        tor_value = getattr(service, "tor", None)
        if isinstance(tor_value, bool):
            return bool(tor_value), True, "service.tor"
        proxy_url = getattr(service, "proxy_url", None)
        if proxy_url is None:
            return False, False, "unknown"
        try:
            from intermine314.service.transport import is_tor_proxy_url

            return bool(is_tor_proxy_url(proxy_url)), True, "service.proxy_url"
        except Exception:
            return False, False, "unknown"

    def _coerce_parallel_options(
        self,
        *,
        parallel_options=None,
    ):
        if parallel_options is None:
            return ParallelOptions()
        if not isinstance(parallel_options, ParallelOptions):
            raise ParallelOptionsError(
                "parallel_options must be a ParallelOptions instance. "
                "Construct options with ParallelOptions(...)."
            )
        return parallel_options

    def _resolve_parallel_options(self, *, start, size, options: ParallelOptions) -> ResolvedParallelOptions:
        try:
            query_defaults = _query_runtime_defaults()
            require_int("page_size", options.page_size)
            start_value = require_int("start", start)
            page_size = require_positive_int("page_size", options.page_size)
            profile, ordered, large_query_mode = self._apply_parallel_profile(
                options.profile,
                options.ordered,
                options.large_query_mode,
            )
            max_workers = self._resolve_effective_workers(options.max_workers, size)
            max_workers = require_positive_int("max_workers", max_workers)
            order_mode = self._normalize_order_mode(ordered)
            tor_enabled, tor_state_known, tor_source = Query._resolve_tor_parallel_context(self)
            prefetch_from_default = options.prefetch is None
            inflight_from_default = options.inflight_limit is None
            prefetch = resolve_prefetch(
                options.prefetch,
                max_workers=max_workers,
                large_query_mode=large_query_mode,
                default_parallel_prefetch=query_defaults.default_parallel_prefetch,
            )
            tor_prefetch_adjusted = False
            if tor_enabled and prefetch_from_default:
                adjusted_prefetch = max(1, min(prefetch, max_workers))
                tor_prefetch_adjusted = adjusted_prefetch != prefetch
                prefetch = adjusted_prefetch
            inflight_limit = resolve_inflight_limit(
                options.inflight_limit,
                prefetch=prefetch,
                default_parallel_inflight_limit=query_defaults.default_parallel_inflight_limit,
            )
            tor_inflight_adjusted = False
            if tor_enabled and inflight_from_default:
                adjusted_inflight = max(1, min(inflight_limit, prefetch, max_workers))
                tor_inflight_adjusted = adjusted_inflight != inflight_limit
                inflight_limit = adjusted_inflight
            tor_aware_defaults_applied = bool(tor_prefetch_adjusted or tor_inflight_adjusted)
            if not tor_state_known:
                _PARALLEL_LOG.debug("parallel_tor_state_unknown strategy=default source=%s", tor_source)
            _PARALLEL_LOG.debug(
                (
                    "parallel_policy_derived tor_enabled=%s tor_state_known=%s tor_source=%s "
                    "tor_aware_defaults_applied=%s prefetch=%d inflight_limit=%d "
                    "prefetch_from_default=%s inflight_from_default=%s"
                ),
                tor_enabled,
                tor_state_known,
                tor_source,
                tor_aware_defaults_applied,
                prefetch,
                inflight_limit,
                prefetch_from_default,
                inflight_from_default,
            )
            inflight_limit = _cap_inflight_limit(
                inflight_limit,
                page_size,
                max_buffered_rows=query_defaults.default_parallel_max_buffered_rows,
            )
            start_value = require_non_negative_int("start", start_value)
            size_value = size
            if size_value is not None:
                size_value = require_non_negative_int("size", size_value)
            max_inflight_bytes_estimate = options.max_inflight_bytes_estimate
            if max_inflight_bytes_estimate is not None:
                max_inflight_bytes_estimate = require_positive_int(
                    "max_inflight_bytes_estimate",
                    max_inflight_bytes_estimate,
                )
            strategy = self._resolve_parallel_strategy(options.pagination, start_value, size_value)
            return ResolvedParallelOptions(
                page_size=page_size,
                max_workers=max_workers,
                order_mode=order_mode,
                prefetch=prefetch,
                inflight_limit=inflight_limit,
                profile=profile,
                large_query_mode=large_query_mode,
                pagination=options.pagination,
                start=start_value,
                size=size_value,
                strategy=strategy,
                max_inflight_bytes_estimate=max_inflight_bytes_estimate,
                tor_enabled=tor_enabled,
                tor_state_known=tor_state_known,
                tor_aware_defaults_applied=tor_aware_defaults_applied,
                tor_source=tor_source,
            )
        except (TypeError, ValueError) as exc:
            raise _parallel_options_error(exc) from exc

    def run_parallel(
        self,
        row="dict",
        start=0,
        size=None,
        job_id=None,
        parallel_options=None,
    ):
        """Fetch paged results concurrently and yield rows.

        Usage::

            options = ParallelOptions(page_size=2000, max_workers=16, pagination="auto")
            for row in query.run_parallel(parallel_options=options):
                process(row)

        :param job_id: Optional correlation id for structured parallel logs.
        :param parallel_options: A ParallelOptions value configuring execution.
        """
        options = self._coerce_parallel_options(parallel_options=parallel_options)
        resolved = self._resolve_parallel_options(start=start, size=size, options=options)
        if resolved.size == 0:
            return iter(())
        run_job_id = str(job_id).strip() if job_id is not None else ""
        if not run_job_id:
            run_job_id = new_job_id("qp")

        iterator = self._run_parallel_offset(
            row=row,
            start=resolved.start,
            size=resolved.size,
            page_size=resolved.page_size,
            max_workers=resolved.max_workers,
            order_mode=resolved.order_mode,
            inflight_limit=resolved.inflight_limit,
            max_inflight_bytes_estimate=resolved.max_inflight_bytes_estimate,
            job_id=run_job_id,
        )
        return instrument_parallel_iterator(
            iterator,
            job_id=run_job_id,
            order_mode=resolved.order_mode,
            start=resolved.start,
            size=resolved.size,
            page_size=resolved.page_size,
            max_workers=resolved.max_workers,
            prefetch=resolved.prefetch,
            inflight_limit=resolved.inflight_limit,
            tor_enabled=resolved.tor_enabled,
            tor_state_known=resolved.tor_state_known,
            tor_aware_defaults_applied=resolved.tor_aware_defaults_applied,
            tor_source=resolved.tor_source,
            max_inflight_bytes_estimate=resolved.max_inflight_bytes_estimate,
        )

    def count(self):
        """Return total rows for this query without materializing result pages."""
        to_run = self.clone()
        if len(to_run.views) == 0:
            to_run.add_view(class_name(to_run.root) + ".*" if Query._has_model(to_run) else to_run.root)

        resolver = getattr(to_run, "_to_execution", None)
        execution = resolver() if callable(resolver) else None
        if execution is not None:
            try:
                return int(execution.count())
            except ValueError as exc:
                raise ResultError(str(exc)) from exc

        params = to_run.to_query_params()
        params["format"] = "count"
        payload = urlencode(params, True).encode("utf-8")
        url = to_run.service.root + to_run.get_results_path()
        with closing(to_run.service.opener.open(url, payload)) as conn:
            raw = conn.read()
        if isinstance(raw, bytes):
            count_str = raw.decode("utf-8", errors="replace")
        else:
            count_str = str(raw)
        try:
            return int(count_str.strip())
        except ValueError:
            raise ResultError("Server returned a non-integer count: " + count_str)

    def get_results_path(self):
        """Return the query-results endpoint path."""
        return self.service.QUERY_PATH

    size = count

    def children(self):
        """Return query child nodes used for minimal XML serialization."""
        return [*self.path_descriptions, *self.joins, *self.constraints]

    def to_spec(self) -> QuerySpec:
        sort_order = str(self.get_sort_order()) if self.views else ""
        model_name = getattr(self.model, "name", "")
        return QuerySpec(
            root_class=class_name(self.root),
            views=tuple(self.views),
            constraints=tuple(self.constraints),
            joins=tuple(self.joins),
            sort_order=sort_order,
            name=str(self.name),
            description=str(self.description),
            model_name=str(model_name or ""),
            compatibility=self.compatibility,
            constraint_logic=str(self.get_logic()),
            decimal_paths=getattr(self, "_decimal_paths", ()),
            path_descriptions=tuple(self.path_descriptions),
        )

    def _to_execution(self):
        service = getattr(self, "service", None)
        if service is None:
            return None
        execute = getattr(service, "execute", None)
        if not callable(execute):
            return None
        return execute(self.to_spec())

    def get_list_upload_uri(self):
        return self.service.root + self.service.QUERY_LIST_UPLOAD_PATH

    def get_list_append_uri(self):
        return self.service.root + self.service.QUERY_LIST_APPEND_PATH

    def _list_upload_params(self, *, append=False):
        params = self.to_query_params()
        if append:
            params["path"] = None
        return params

    def to_query(self):
        """Cast to a query, preserving the public identity protocol."""
        return self

    def make_list_constraint(self, path, op):
        from intermine314.model import ConstraintNode

        item = self.service.create_list(self)
        return ConstraintNode(path, op, item.name)

    def __or__(self, other):
        return self.service._get_list_manager().union([self, other])

    def __add__(self, other):
        return self.service._get_list_manager().union([self, other])

    def __and__(self, other):
        return self.service._get_list_manager().intersect([self, other])

    def __xor__(self, other):
        return self.service._get_list_manager().xor([self, other])

    def __sub__(self, other):
        return self.service._get_list_manager().subtract([self], [other])

    def to_query_params(self):
        """Build the request payload for query execution endpoints."""
        return {"query": query_spec_to_xml(self.to_spec())}

    @staticmethod
    def _xml_attr(value):
        if value is None:
            return ""
        return str(value)

    def _append_join_xml(self, query, join):
        return _append_join_xml(query, join)

    def _append_constraint_xml(self, query, constraint):
        return _append_constraint_xml(query, constraint, compatibility=self.compatibility)

    def _build_query_xml_element(self):
        return query_spec_to_element(self.to_spec())

    def to_xml(self):
        """
        Return an XML serialisation of the query
        ========================================

        This method serialises the current state of the query to an
        xml string, suitable for storing, or sending over the
        internet to the webservice.

        @return: the serialised xml string
        @rtype: string
        """
        return query_spec_to_xml(self.to_spec())

    def to_Node(self):
        """Return a minidom Element encoded through the canonical QuerySpec."""
        return minidom.parseString(self.to_xml()).documentElement

    def to_formatted_xml(self):
        """
        Return a readable XML serialisation of the query
        ================================================

        This method serialises the current state of the query to an
        xml string, suitable for storing, or sending over the
        internet to the webservice, only more readably.

        @return: the serialised xml string
        @rtype: string
        """
        return query_spec_to_formatted_xml(self.to_spec())

    def clone(self):
        """
        Performs a deep clone
        =====================

        This method will produce a clone that is independent,
        and can be altered without affecting the original,
        but starts off with the exact same state as it.

        The only shared elements should be the model
        and the service, which are shared by all queries
        that refer to the same webservice.

        @return: same class as caller
        """
        from copy import deepcopy

        newobj = self.__class__(
            model=self.model,
            service=self.service,
            validate=self.do_verification,
            root=self.root,
            compatibility=self.compatibility,
        )
        copied = deepcopy({attr: getattr(self, attr) for attr in [
            "joins",
            "path_descriptions",
            "views",
            "_sort_order_list",
            "constraint_dict",
            "uncoded_constraints",
            "constraint_factory",
            "_logic",
        ]})
        for attr, value in copied.items():
            setattr(newobj, attr, value)

        def bind_logic(node, parent=None):
            if isinstance(node, LogicGroup):
                node.parent = parent
                node.left = bind_logic(node.left, node)
                node.right = bind_logic(node.right, node)
            elif isinstance(node, CodedConstraint):
                # Accepted external nodes may have a query's code without
                # sharing its constraint instance. Bind by code on the clone.
                return newobj.constraint_dict.get(node.code, node)
            return node

        newobj._logic = bind_logic(newobj._logic)
        newobj._decimal_paths = getattr(self, "_decimal_paths", ())

        for attr in ["name", "description", "service", "do_verification", "root", "prefetch_depth", "prefetch_id_only"]:
            setattr(newobj, attr, getattr(self, attr))
        return newobj


class QueryError(ReadableException):
    pass


class ConstraintError(QueryError):
    pass


class QueryParseError(QueryError):
    pass


class ResultError(ReadableException):
    pass
