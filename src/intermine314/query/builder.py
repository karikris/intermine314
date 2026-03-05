from intermine314.model.class_ import Class
from intermine314.model.constants import NUMERIC_TYPES_NORMALIZED
from intermine314.model.fields import Reference
from intermine314.model.operators import Column
from intermine314.config.storage_policy import (
    default_parquet_compression as _default_parquet_compression,
    validate_duckdb_identifier as _validate_duckdb_identifier,
    validate_parquet_compression as _validate_parquet_compression,
)
from intermine314.config.runtime_defaults import get_runtime_defaults
from dataclasses import dataclass, field
import logging
import re

from intermine314.util import ReadableException
from intermine314.util.logging import new_job_id
from intermine314.query.constraints import (
    BinaryConstraint,
    ConstraintFactory,
    MultiConstraint,
    SubClassConstraint,
)
from intermine314.query.pathfeatures import PathDescription, Join, SortOrder, SortOrderList
from intermine314.service.resource_utils import close_resource_quietly as _close_resource_quietly
from intermine314.util.deps import (
    optional_duckdb as _optional_duckdb,
    quote_sql_string as _duckdb_quote,
    require_duckdb as _require_duckdb,
    require_polars as _require_polars,
)
from intermine314.query.parallel_runtime import (
    PARALLEL_LOG as _PARALLEL_LOG,
    instrument_parallel_iterator,
)
from intermine314.parallel.policy import (
    VALID_ORDER_MODES as CANONICAL_VALID_ORDER_MODES,
    VALID_PARALLEL_PAGINATION as CANONICAL_VALID_PARALLEL_PAGINATION,
    VALID_PARALLEL_PROFILES as CANONICAL_VALID_PARALLEL_PROFILES,
    apply_parallel_profile,
    normalize_order_mode,
    require_int,
    require_non_negative_int,
    require_positive_int,
    resolve_inflight_limit,
    resolve_parallel_strategy,
    resolve_prefetch,
)

VALID_PARALLEL_PAGINATION = CANONICAL_VALID_PARALLEL_PAGINATION
VALID_PARALLEL_PROFILES = CANONICAL_VALID_PARALLEL_PROFILES
VALID_ORDER_MODES = CANONICAL_VALID_ORDER_MODES
VALID_ITER_ROW_MODES = frozenset({"dict"})
VALID_RESULT_ROW_MODES = frozenset({"dict", "count"})


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


class Query(object):
    """
    Structured query builder for InterMine services.

    This class builds query XML/params, validates paths and constraints against
    the model, and executes result iterators including bounded parallel modes.
    Full tutorials live in ``docs/source/query.rst``.
    """
    SO_SPLIT_PATTERN = re.compile("\\s*(asc|desc)\\s*", re.I)
    def __init__(self, model, service=None, validate=True, root=None):
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
        self.model = model
        if root is None:
            self.root = root
        else:
            self.root = model.make_path(root).root

        self.name = ""
        self.description = ""
        self.service = service
        self.prefetch_depth = service.prefetch_depth if service is not None else 1
        self.prefetch_id_only = service.prefetch_id_only if service is not None else False
        self.do_verification = validate
        self.path_descriptions = []
        self.joins = []
        self.constraint_dict = {}
        self.uncoded_constraints = []
        self.views = []
        self._sort_order_list = SortOrderList()
        self.constraint_factory = ConstraintFactory()

    def __iter__(self):
        """Return an iterator over query rows as dictionaries."""
        return self.results("dict")

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
        views = []
        for p in paths:
            if isinstance(p, (set, list)):
                views.extend(list(p))
            elif isinstance(p, Class):
                views.append(p.name + ".*")
            elif isinstance(p, Column):
                if p._path.is_attribute():
                    views.append(str(p))
                else:
                    views.append(str(p) + ".*")
            elif isinstance(p, Reference):
                views.append(p.name + ".*")
            else:
                views.extend(re.split("(?:,?\\s+|,)", str(p)))

        views = list(map(self.prefix_path, views))

        views_to_add = []
        for view in views:
            if view.endswith(".*"):
                view = re.sub("\\.\\*$", "", view)
                scd = self.get_subclass_dict()

                def expand(p, level, id_only=False):
                    if level > 0:
                        path = self.model.make_path(p, scd)
                        cd = path.end_class

                        def add_f(x):
                            return p + "." + x.name

                        vs = [p + ".id"] if id_only and cd.has_id else [add_f(a) for a in cd.attributes]
                        next_level = level - 1
                        rs_and_cs = list(cd.references) + list(cd.collections)
                        for r in rs_and_cs:
                            rp = add_f(r)
                            if next_level:
                                self.outerjoin(rp)
                            vs.extend(expand(rp, next_level, self.prefetch_id_only))
                        return vs
                    else:
                        return []

                depth = self.prefetch_depth
                views_to_add.extend(expand(view, depth))
            else:
                views_to_add.append(view)

        if self.do_verification:
            self.verify_views(views_to_add)

        self.views.extend(views_to_add)

        return self

    def prefix_path(self, path):
        if self.root is None:
            if self.do_verification:  # eg. not when building from XML
                if path.endswith(".*"):
                    trimmed = re.sub("\\.\\*$", "", path)
                else:
                    trimmed = path
                self.root = self.model.make_path(trimmed, self.get_subclass_dict()).root
            return path
        else:
            if path.startswith(self.root.name):
                return path
            else:
                return self.root.name + "." + path

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
            path = self.model.make_path(path, self.get_subclass_dict())
            if not path.is_attribute():
                raise ConstraintError("'" + str(path) + "' does not represent an attribute")

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
        if len(args) == 1 and len(kwargs) == 0:
            only = args[0]
            if isinstance(only, tuple):
                con = self.constraint_factory.make_constraint(*only)
            elif hasattr(only, "vargs") and hasattr(only, "kwargs"):
                con = self.constraint_factory.make_constraint(*only.vargs, **only.kwargs)
            else:
                con = only
        elif len(args) == 0 and len(kwargs) == 1:
            k, v = list(kwargs.items())[0]
            if isinstance(v, (list, tuple, set)):
                con = self.constraint_factory.make_constraint(k, "IN", list(v))
            else:
                con = self.constraint_factory.make_constraint(k, "=", v)
        else:
            con = self.constraint_factory.make_constraint(*args, **kwargs)

        con.path = self.prefix_path(con.path)
        if self.do_verification:
            self.verify_constraint_paths([con])
        if hasattr(con, "code"):
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
        c = self.clone()
        for con in cons:
            if hasattr(con, "vargs") and hasattr(con, "kwargs"):
                c.add_constraint(*con.vargs, **con.kwargs)
                continue
            if isinstance(con, tuple):
                c.add_constraint(*con)
                continue
            c.add_constraint(con)
        for path, value in list(kwargs.items()):
            if isinstance(value, (list, tuple, set)):
                c.add_constraint(path, "IN", list(value))
            else:
                c.add_constraint(path, "=", value)
        return c

    def column(self, col):
        """
        Return a Column object suitable for using to construct constraints with
        =======================================================================

        This method is part of the SQLAlchemy style API.

        """
        return self.model.column(self.prefix_path(str(col)), self.get_subclass_dict(), self)

    def verify_constraint_paths(self, cons=None):
        """
        Check that the constraints are valid
        ====================================

        This method will check the path attribute of each constraint.
        In minimal mode it validates:
          - Binary and multi-value constraints target attributes
          - Subclass constraints use valid subclass relationships

        @param cons: The constraints to check
                     (defaults to all constraints on the query)

        @raise ModelError: if the paths are not valid
        @raise ConstraintError: if the constraints do not satisfy the above
                                rules

        """
        if cons is None:
            cons = self.constraints
        for con in cons:
            pathA = self.model.make_path(con.path, self.get_subclass_dict())
            if isinstance(con, (BinaryConstraint, MultiConstraint)):
                if not pathA.is_attribute():
                    raise ConstraintError("'" + str(pathA) + "' does not represent an attribute")
            elif isinstance(con, SubClassConstraint):
                pathB = self.model.make_path(con.subclass, self.get_subclass_dict())
                if not pathB.get_class().isa(pathA.get_class()):
                    raise ConstraintError("'" + con.subclass + "' is not a subclass of '" + con.path + "'")
            else:
                if not pathA.is_attribute():
                    raise ConstraintError("'" + str(pathA) + "' does not represent an attribute")

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
            path = self.model.make_path(join.path, self.get_subclass_dict())
            if not path.is_reference():
                raise QueryError("'" + join.path + "' is not a reference")

    def add_path_description(self, *args, **kwargs):
        """
        Add a path description to the query
        ===================================

        example::

            query.add_path_description("Gene.proteins.proteinDomains", "Protein Domain")

        This allows you to alias the components of long paths to
        improve the way they display column headers in a variety of
        circumstances. In the above example, if the view included the unwieldy
        path "Gene.proteins.proteinDomains.primaryIdentifier", it would
        (depending on the mine) be displayed as
        "Protein Domain > DB Identifer". These setting are taken into account
        by the webservice when generating column headers for flat-file results
        with the columnheaders parameter given, and always supplied when
        requesting jsontable results.

        @rtype: L{intermine314.pathfeatures.PathDescription}

        """
        path_description = PathDescription(*args, **kwargs)
        path_description.path = self.prefix_path(path_description.path)
        if self.do_verification:
            self.verify_pd_paths([path_description])
        self.path_descriptions.append(path_description)
        return path_description

    def verify_pd_paths(self, pds=None):
        """
        Check that the path of the path description is valid
        ====================================================

        Checks for consistency with the data model

        @raise ModelError: if the paths are invalid
        """
        if pds is None:
            pds = self.path_descriptions
        for pd in pds:
            self.model.validate_path(pd.path, self.get_subclass_dict())

    @property
    def coded_constraints(self):
        """Return constraints that carry an explicit code."""
        return sorted(list(self.constraint_dict.values()), key=lambda con: con.code)

    def get_logic(self):
        """Legacy logic API is disabled in the minimal runtime surface."""
        return ""

    def set_logic(self, value):
        _ = value
        raise NotImplementedError(
            "Constraint logic expressions are removed from the minimal runtime surface. "
            "Use sequential where()/add_constraint() predicates instead."
        )

    def validate_logic(self, logic=None):
        _ = logic
        raise NotImplementedError(
            "Constraint logic validation is removed from the minimal runtime surface."
        )

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
            p = self.model.make_path(so.path, self.get_subclass_dict())
            if p.prefix() not in from_paths:
                raise QueryError("Sort order element %s is not in the query" % so.path)

    def _from_paths(self):
        scd = self.get_subclass_dict()
        froms = set([self.model.make_path(x, scd).prefix() for x in self.views])
        for c in self.constraints:
            p = self.model.make_path(c.path, scd)
            if p.is_attribute():
                froms.add(p.prefix())
            else:
                froms.add(p)
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

    def results(self, row="dict", start=0, size=None, summary_path=None):
        """
        Return an iterator over result rows
        ===================================

        Usage::

          >>> query = service.model.Gene.select("symbol", "length")
          >>> for d in query.results(row="dict"):
          ...    print(d["Gene.symbol"])

        This method supports canonical row formats for data-plane workflows:
        ``"dict"`` and ``"count"``.

        If no views have been specified, all attributes of the root class
        are selected for output.

        @param row: The format for each result. One of "dict", "count"
        @type row: string
        @param start: the index of the first result to return (default = 0)
        @type start: int
        @param size: The maximum number of results to return (default = all)
        @type size: int
        @param summary_path: A column name to optionally summarise. Specifying
                             a path will force "jsonrows" format, and return
                             an iterator over a list of dictionaries. Use this
                             when you are interested in processing a summary
                             in order of greatest count to smallest.
        @type summary_path: str or L{intermine314.model.Path}

        @rtype: L{intermine314.webservice.ResultIterator}

        @raise WebserviceError: if the request is unsuccessful
        """

        to_run = self.clone()

        if len(to_run.views) == 0:
            to_run.add_view(to_run.root)

        if row not in VALID_RESULT_ROW_MODES:
            choices = ", ".join(sorted(VALID_RESULT_ROW_MODES))
            raise ValueError(f"row must be one of: {choices}")

        path = to_run.get_results_path()
        params = to_run.to_query_params()
        params["start"] = start
        if size:
            params["size"] = size
        if summary_path:
            params["summaryPath"] = to_run.prefix_path(summary_path)
            row = "jsonrows"

        view = to_run.views
        cld = to_run.root

        return to_run.service.get_results(path, params, row, view, cld)

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
        for row in row_iter:
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def rows(self, start=0, size=None, row="dict"):
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
        @rtype: iterable<dict>
        """
        if row not in VALID_ITER_ROW_MODES:
            choices = ", ".join(sorted(VALID_ITER_ROW_MODES))
            raise ValueError(f"row must be one of: {choices}")
        return self.results(row=row, start=start, size=size)

    def _iter_batches_kwargs(
        self,
        *,
        start=0,
        size=None,
        batch_size=None,
        row_mode="dict",
        parallel_options=None,
    ):
        if batch_size is None:
            batch_size = _runtime_default_batch_size()
        return {
            "start": start,
            "size": size,
            "batch_size": batch_size,
            "row_mode": row_mode,
            "parallel_options": parallel_options,
        }

    def dataframe(
        self,
        start=0,
        size=None,
        batch_size=None,
        *,
        final_rechunk=False,
        parallel_options=None,
    ):
        """
        Returns a polars.DataFrame
        ==================================

        Usage::
          >>> query.dataframe(parallel_options=ParallelOptions(max_workers=8))

        @param start: the index of the first result to return (default = 0)
        @type start: int
        @param size: The maximum number of results to return (default = all)
        @type size: int
        @param batch_size: Number of rows to buffer per batch (default = DEFAULT_BATCH_SIZE)
        @type batch_size: int
        @param final_rechunk: Force a final contiguous-memory rechunk at concat time.
                              Defaults to False to reduce peak RSS for large scans.
        @type final_rechunk: bool
        @rtype: dataframe<polars.DataFrame>

        """
        from intermine314.query.data_plane import query_to_dataframe

        return query_to_dataframe(
            self,
            start=start,
            size=size,
            batch_size=batch_size,
            parallel_options=parallel_options,
            final_rechunk=final_rechunk,
            runtime_default_batch_size=_runtime_default_batch_size,
            require_polars=_require_polars,
        )

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
    ):
        """
        Stream results to Parquet files.

        Usage::
          >>> query.to_parquet(
          ...     "results_parquet",
          ...     batch_size=5000,
          ...     parallel_options=ParallelOptions(max_workers=8),
          ... )
        """
        from intermine314.query.data_plane import query_to_parquet

        return query_to_parquet(
            self,
            path,
            start=start,
            size=size,
            batch_size=batch_size,
            compression=compression,
            single_file=single_file,
            temp_dir=temp_dir,
            temp_dir_min_free_bytes=temp_dir_min_free_bytes,
            parallel_options=parallel_options,
            runtime_default_export_batch_size=_runtime_default_export_batch_size,
            default_parquet_compression=_default_parquet_compression,
            validate_parquet_compression=_validate_parquet_compression,
            require_polars=_require_polars,
            optional_duckdb=_optional_duckdb,
            duckdb_quote=_duckdb_quote,
        )

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
        from intermine314.query.data_plane import query_to_duckdb

        return query_to_duckdb(
            self,
            path,
            start=start,
            size=size,
            batch_size=batch_size,
            compression=compression,
            single_file=single_file,
            database=database,
            table=table,
            temp_dir=temp_dir,
            temp_dir_min_free_bytes=temp_dir_min_free_bytes,
            parallel_options=parallel_options,
            managed=managed,
            runtime_default_export_batch_size=_runtime_default_export_batch_size,
            validate_duckdb_identifier=_validate_duckdb_identifier,
            require_duckdb=_require_duckdb,
            duckdb_quote=_duckdb_quote,
            close_resource_quietly=_close_resource_quietly,
        )

    def duckdb_view(self, *args, **kwargs):
        """Return a managed DuckDB context wrapper for query parquet views."""
        from intermine314.query.data_plane import query_duckdb_view

        return query_duckdb_view(self, *args, **kwargs)

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
        """
        Fetch paged results concurrently and yield rows.

        Usage::
          >>> options = ParallelOptions(page_size=2000, max_workers=16, pagination="auto")
          >>> for row in query.run_parallel(parallel_options=options):
          ...     process(row)
        @param job_id: Optional correlation id for structured parallel export logs.
        @type job_id: str | None
        @param parallel_options: Canonical parallel tuning value object.
                                 Use ParallelOptions(...) to configure execution.
        @type parallel_options: ParallelOptions | None
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

    def summarise(self, summary_path, **kwargs):
        """
        Return a summary of the results for this column.
        ================================================

        Usage::
            >>> query = service.select("Gene.*", "organism.*").where("Gene", "IN", "my-list")
            >>> print(query.summarise("length")["average"])
            ... 12345.67890
            >>> print(query.summarise("organism.name")["Drosophila simulans"])
            ... 98

        This method allows you to get statistics summarising the information
        from just one column of a query. For numerical columns you get
        dictionary with four keys ('average', 'stdev', 'max', 'min'), and for
        non-numerical columns you get a dictionary where each item is a key
        and the values are the number of occurrences of this value in the
        column.

        Any key word arguments will be passed to the underlying results call -
        so you can limit the result size to the top 100 items by passing
        "size = 100" as part of the call.

        @see: L{intermine314.query.Query.results}

        @param summary_path: The column to summarise (either in long or short
                             form)
        @type summary_path: str or L{intermine314.model.Path}

        @rtype: dict
        This method is sugar for particular combinations of calls to
        L{results}.
        """
        p = self.model.make_path(self.prefix_path(summary_path), self.get_subclass_dict())
        results = self.results(summary_path=summary_path, **kwargs)
        type_name = p.end.type_name
        normalized_type_name = type_name.strip().lower() if isinstance(type_name, str) else ""
        if normalized_type_name in NUMERIC_TYPES_NORMALIZED:
            return dict((k, float(v)) for k, v in list(next(results).items()))
        else:
            return dict((r["item"], r["count"]) for r in results)

    def one(self, row="dict"):
        """Return one result, and raise an error if the result size is not 1"""
        if row not in VALID_ITER_ROW_MODES:
            choices = ", ".join(sorted(VALID_ITER_ROW_MODES))
            raise ValueError(f"row must be one of: {choices}")
        c = self.count()
        if c != 1:
            raise QueryError("Result size is not one: got %d results" % (c))
        else:
            return self.first(row)

    def first(self, row="dict", start=0, **kw):
        """Return the first result, or None if the results are empty"""
        if row not in VALID_ITER_ROW_MODES:
            choices = ", ".join(sorted(VALID_ITER_ROW_MODES))
            raise ValueError(f"row must be one of: {choices}")
        size = 1
        try:
            return next(self.results(row, start=start, size=size, **kw))
        except StopIteration:
            return None

    def get_results_list(self, *args, **kwargs):
        """
        Get a list of result rows
        =========================

        This method is a shortcut so that you do not have to
        do a list comprehension yourself on the iterator that
        is normally returned. If you have a very large result
        set (and these can get up to 100's of thousands or rows
        pretty easily) you will not want to
        have the whole list in memory at once, but there may
        be other circumstances when you might want to keep the whole
        list in one place.

        It takes all the same arguments and parameters as Query.results

        @see: L{intermine314.query.Query.results}

        """
        return list(self.results(*args, **kwargs))

    def get_row_list(self, start=0, size=None):
        return self.get_results_list("dict", start, size)

    def count(self):
        """Return total rows for this query without materializing result pages."""
        count_str = ""
        for row in self.results(row="count"):
            count_str += row
        try:
            return int(count_str)
        except ValueError:
            raise ResultError("Server returned a non-integer count: " + count_str)

    def get_results_path(self):
        """Return the query-results endpoint path."""
        return self.service.QUERY_PATH

    def children(self):
        """Return query child nodes used for XML serialization."""
        return [*self.path_descriptions, *self.joins, *self.constraints]

    def to_query_params(self):
        """Build the request payload for query execution endpoints."""
        xml = self.to_xml()
        params = {"query": xml}
        return params

    def to_Node(self):
        """Return a DOM node for the current query definition."""
        from xml.dom import getDOMImplementation

        impl = getDOMImplementation()
        doc = impl.createDocument(None, "query", None)
        query = doc.documentElement

        query.setAttribute("name", self.name)
        query.setAttribute("model", self.model.name)
        query.setAttribute("view", " ".join(self.views))
        query.setAttribute("sortOrder", str(self.get_sort_order()))
        query.setAttribute("longDescription", self.description)

        for c in self.children():
            element = doc.createElement(c.child_type)
            for name, value in list(c.to_dict().items()):
                if isinstance(value, (set, list)):
                    for v in value:
                        subelement = doc.createElement(name)
                        text = doc.createTextNode(v)
                        subelement.appendChild(text)
                        element.appendChild(subelement)
                else:
                    element.setAttribute(name, value)
            query.appendChild(element)
        return query

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
        n = self.to_Node()
        return n.toxml()

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
        n = self.to_Node()
        return n.toprettyxml()

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

        newobj = self.__class__(self.model)
        for attr in [
            "joins",
            "views",
            "_sort_order_list",
            "path_descriptions",
            "constraint_dict",
            "uncoded_constraints",
        ]:
            setattr(newobj, attr, deepcopy(getattr(self, attr)))

        for attr in ["name", "description", "service", "do_verification", "constraint_factory", "root"]:
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
