from intermine314.query import constraints
from intermine314.model import Column, Class, Model, Reference, ConstraintNode
from intermine314.config.constants import (
    DEFAULT_BATCH_SIZE as BASE_DEFAULT_BATCH_SIZE,
    DEFAULT_EXPORT_BATCH_SIZE as BASE_DEFAULT_EXPORT_BATCH_SIZE,
    DEFAULT_KEYSET_AUTO_MIN_SIZE as BASE_DEFAULT_KEYSET_AUTO_MIN_SIZE,
    DEFAULT_KEYSET_BATCH_SIZE as BASE_DEFAULT_KEYSET_BATCH_SIZE,
    DEFAULT_LARGE_QUERY_MODE as BASE_DEFAULT_LARGE_QUERY_MODE,
    DEFAULT_ORDER_WINDOW_PAGES as BASE_DEFAULT_ORDER_WINDOW_PAGES,
    DEFAULT_PARALLEL_INFLIGHT_LIMIT as BASE_DEFAULT_PARALLEL_INFLIGHT_LIMIT,
    DEFAULT_PARALLEL_MAX_BUFFERED_ROWS as BASE_DEFAULT_PARALLEL_MAX_BUFFERED_ROWS,
    DEFAULT_PARALLEL_ORDERED_MODE as BASE_DEFAULT_PARALLEL_ORDERED_MODE,
    DEFAULT_PARALLEL_PAGE_SIZE as BASE_DEFAULT_PARALLEL_PAGE_SIZE,
    DEFAULT_PARALLEL_PAGINATION as BASE_DEFAULT_PARALLEL_PAGINATION,
    DEFAULT_PARALLEL_PREFETCH as BASE_DEFAULT_PARALLEL_PREFETCH,
    DEFAULT_PARALLEL_PROFILE as BASE_DEFAULT_PARALLEL_PROFILE,
    DEFAULT_PARALLEL_WORKERS as BASE_DEFAULT_PARALLEL_WORKERS,
    DEFAULT_QUERY_THREAD_NAME_PREFIX as BASE_DEFAULT_QUERY_THREAD_NAME_PREFIX,
)
from contextlib import closing
from dataclasses import dataclass
import logging
import re
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from itertools import chain, islice
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.dom import minidom, getDOMImplementation

from intermine314.util import openAnything, ReadableException
from intermine314.util.logging import log_structured_event, new_job_id
from intermine314.query.pathfeatures import PathDescription, Join, SortOrder, SortOrderList
from intermine314.registry.mines import resolve_preferred_workers
from intermine314.service.transport import is_tor_proxy_url
from intermine314.util.deps import (
    optional_duckdb as _optional_duckdb,
    quote_sql_string as _duckdb_quote,
    require_duckdb as _require_duckdb,
    require_polars as _require_polars,
)
from intermine314.export.parquet import write_single_parquet_from_parts
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
"""
Classes representing queries against webservices
================================================

Representations of queries, and templates.

"""

__author__ = "Alex Kalderimis"
__organization__ = "InterMine"
__license__ = "LGPL"
__contact__ = "toffe.kari@gmail.com"

LOGIC_OPS = ["and", "or"]
LOGIC_PRODUCT = [(x, y) for x in LOGIC_OPS for y in LOGIC_OPS]
VALID_PARALLEL_PAGINATION = CANONICAL_VALID_PARALLEL_PAGINATION
VALID_PARALLEL_PROFILES = CANONICAL_VALID_PARALLEL_PROFILES
VALID_ORDER_MODES = CANONICAL_VALID_ORDER_MODES
VALID_ITER_ROW_MODES = frozenset({"dict", "list", "rr"})

DEFAULT_PARALLEL_WORKERS = BASE_DEFAULT_PARALLEL_WORKERS
DEFAULT_PARALLEL_PAGE_SIZE = BASE_DEFAULT_PARALLEL_PAGE_SIZE
DEFAULT_PARALLEL_PAGINATION = BASE_DEFAULT_PARALLEL_PAGINATION
DEFAULT_PARALLEL_PROFILE = BASE_DEFAULT_PARALLEL_PROFILE
DEFAULT_PARALLEL_ORDERED_MODE = BASE_DEFAULT_PARALLEL_ORDERED_MODE
DEFAULT_LARGE_QUERY_MODE = BASE_DEFAULT_LARGE_QUERY_MODE
DEFAULT_PARALLEL_PREFETCH = BASE_DEFAULT_PARALLEL_PREFETCH
DEFAULT_PARALLEL_INFLIGHT_LIMIT = BASE_DEFAULT_PARALLEL_INFLIGHT_LIMIT
DEFAULT_ORDER_WINDOW_PAGES = BASE_DEFAULT_ORDER_WINDOW_PAGES
DEFAULT_BATCH_SIZE = BASE_DEFAULT_BATCH_SIZE
DEFAULT_KEYSET_BATCH_SIZE = BASE_DEFAULT_KEYSET_BATCH_SIZE
DEFAULT_EXPORT_BATCH_SIZE = BASE_DEFAULT_EXPORT_BATCH_SIZE
DEFAULT_QUERY_THREAD_NAME_PREFIX = BASE_DEFAULT_QUERY_THREAD_NAME_PREFIX
KEYSET_AUTO_MIN_SIZE = BASE_DEFAULT_KEYSET_AUTO_MIN_SIZE
DEFAULT_PARALLEL_MAX_BUFFERED_ROWS = BASE_DEFAULT_PARALLEL_MAX_BUFFERED_ROWS
_BYTES_ESTIMATE_SAMPLE_ROWS = 32
_BYTES_ESTIMATE_SAMPLE_VALUES = 32
_BYTES_ESTIMATE_EMA_ALPHA = 0.25
VALID_PARQUET_COMPRESSIONS = {"zstd", "snappy", "gzip", "brotli", "lz4", "uncompressed"}
DUCKDB_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PARALLEL_LOG = logging.getLogger("intermine314.query.parallel")
_LEGACY_PARALLEL_ARGS_WARNING_EMITTED = False

try:
    _EXECUTOR_MAP_SUPPORTS_BUFFERSIZE = "buffersize" in ThreadPoolExecutor.map.__code__.co_varnames
except Exception:
    _EXECUTOR_MAP_SUPPORTS_BUFFERSIZE = False


def _cap_inflight_limit(inflight_limit, page_size):
    if DEFAULT_PARALLEL_MAX_BUFFERED_ROWS <= 0:
        return inflight_limit
    max_pending_by_rows = max(1, DEFAULT_PARALLEL_MAX_BUFFERED_ROWS // max(1, page_size))
    return min(inflight_limit, max_pending_by_rows)


def _estimate_scalar_payload_bytes(value, *, depth=0):
    if value is None:
        return 0
    if isinstance(value, bool):
        return 1
    if isinstance(value, (int, float)):
        return 8
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8", errors="ignore"))
    if depth >= 2:
        return min(len(str(value).encode("utf-8", errors="ignore")), 4096)
    if isinstance(value, dict):
        total = 0
        sampled = 0
        for key, item in islice(value.items(), _BYTES_ESTIMATE_SAMPLE_VALUES):
            total += _estimate_scalar_payload_bytes(key, depth=depth + 1)
            total += _estimate_scalar_payload_bytes(item, depth=depth + 1)
            sampled += 1
        if sampled == 0:
            return 0
        if len(value) > sampled:
            total = int(total * (len(value) / float(sampled)))
        return total
    if isinstance(value, (list, tuple)):
        total = 0
        sampled = 0
        for item in islice(value, _BYTES_ESTIMATE_SAMPLE_VALUES):
            total += _estimate_scalar_payload_bytes(item, depth=depth + 1)
            sampled += 1
        if sampled == 0:
            return 0
        if len(value) > sampled:
            total = int(total * (len(value) / float(sampled)))
        return total
    return min(len(str(value).encode("utf-8", errors="ignore")), 4096)


def _estimate_row_payload_bytes(row):
    if isinstance(row, dict):
        total = 0
        sampled = 0
        for key, value in islice(row.items(), _BYTES_ESTIMATE_SAMPLE_VALUES):
            total += _estimate_scalar_payload_bytes(key, depth=1)
            total += _estimate_scalar_payload_bytes(value, depth=1)
            sampled += 1
        if sampled == 0:
            return 0
        if len(row) > sampled:
            total = int(total * (len(row) / float(sampled)))
        return total + 32
    if isinstance(row, (list, tuple)):
        total = 0
        sampled = 0
        for item in islice(row, _BYTES_ESTIMATE_SAMPLE_VALUES):
            total += _estimate_scalar_payload_bytes(item, depth=1)
            sampled += 1
        if sampled == 0:
            return 0
        if len(row) > sampled:
            total = int(total * (len(row) / float(sampled)))
        return total + 32
    return _estimate_scalar_payload_bytes(row, depth=0) + 16


def _estimate_rows_payload_bytes(rows):
    try:
        row_count = int(len(rows))
        if row_count <= 0:
            return 0
        sample_count = min(row_count, _BYTES_ESTIMATE_SAMPLE_ROWS)
        sample_total = 0
        for item in rows[:sample_count]:
            sample_total += _estimate_row_payload_bytes(item)
        avg = sample_total / float(sample_count)
        return max(0, int(avg * row_count))
    except Exception:
        return None


class _AdaptiveInflightCap:
    def __init__(self, *, row_limit, max_inflight_bytes_estimate):
        self.row_limit = max(1, int(row_limit))
        self.initial_bytes_limit = (
            int(max_inflight_bytes_estimate) if max_inflight_bytes_estimate is not None else None
        )
        self.bytes_limit = self.initial_bytes_limit
        self.avg_page_bytes = None
        self.reserved_inflight_bytes = 0.0
        self.max_reserved_inflight_bytes = 0.0
        self.bytes_cap_hits = 0
        self.estimator_failures = 0

    @property
    def bytes_cap_configured(self):
        return self.initial_bytes_limit is not None

    @property
    def bytes_cap_active(self):
        return self.bytes_limit is not None

    def can_submit(self, pending_count):
        if pending_count >= self.row_limit:
            return False
        if self.bytes_limit is None:
            return True
        if self.avg_page_bytes is None:
            # Bootstrap with a single in-flight page to learn payload width.
            return pending_count == 0
        if pending_count == 0:
            return True
        projected = self.reserved_inflight_bytes + self.avg_page_bytes
        if projected <= float(self.bytes_limit):
            return True
        self.bytes_cap_hits += 1
        return False

    def reserve_on_submit(self):
        if self.bytes_limit is None or self.avg_page_bytes is None:
            reserved = 0.0
        else:
            reserved = float(self.avg_page_bytes)
        self.reserved_inflight_bytes += reserved
        self.max_reserved_inflight_bytes = max(self.max_reserved_inflight_bytes, self.reserved_inflight_bytes)
        return reserved

    def observe_completion(self, reserved_bytes, rows):
        self.reserved_inflight_bytes = max(0.0, self.reserved_inflight_bytes - float(reserved_bytes))
        if self.bytes_limit is None:
            return
        estimate = _estimate_rows_payload_bytes(rows)
        if estimate is None:
            # Estimator failures should not block progress. Fall back to row-only capping.
            self.estimator_failures += 1
            self.bytes_limit = None
            return
        if self.avg_page_bytes is None:
            self.avg_page_bytes = float(estimate)
        else:
            self.avg_page_bytes = (self.avg_page_bytes * (1.0 - _BYTES_ESTIMATE_EMA_ALPHA)) + (
                float(estimate) * _BYTES_ESTIMATE_EMA_ALPHA
            )

    def stats_fields(self):
        return {
            "bytes_cap_configured": self.bytes_cap_configured,
            "bytes_cap_active": self.bytes_cap_active,
            "estimated_page_bytes": (
                int(round(self.avg_page_bytes))
                if self.avg_page_bytes is not None
                else None
            ),
            "max_estimated_inflight_bytes": int(round(self.max_reserved_inflight_bytes)),
            "bytes_cap_hits": int(self.bytes_cap_hits),
            "bytes_estimator_failures": int(self.estimator_failures),
        }


def _log_parallel_event(level, event, **fields):
    if not _PARALLEL_LOG.isEnabledFor(level):
        return
    log_structured_event(_PARALLEL_LOG, level, event, **fields)


def _polars_from_dicts_with_full_inference(polars_module, batch):
    """Use full-batch schema inference when supported to avoid type drift within a batch."""
    try:
        return polars_module.from_dicts(batch, infer_schema_length=None)
    except TypeError as exc:
        detail = str(exc)
        if "infer_schema_length" not in detail or "keyword" not in detail:
            raise
        return polars_module.from_dicts(batch)


def _instrument_parallel_iterator(
    iterator,
    *,
    job_id,
    strategy,
    order_mode,
    start,
    size,
    page_size,
    max_workers,
    prefetch,
    inflight_limit,
    max_in_flight,
    ordered_window_pages,
    tor_enabled,
    tor_state_known,
    tor_aware_defaults_applied,
    tor_source,
    max_inflight_bytes_estimate,
):
    started = time.perf_counter()
    yielded_rows = 0
    _log_parallel_event(
        logging.INFO,
        "parallel_export_start",
        job_id=job_id,
        strategy=strategy,
        ordered_mode=order_mode,
        start=start,
        size=size,
        page_size=page_size,
        max_workers=max_workers,
        prefetch=prefetch,
        in_flight=inflight_limit,
        max_in_flight=max_in_flight,
        ordered_window_pages=ordered_window_pages,
        tor_enabled=tor_enabled,
        tor_state_known=tor_state_known,
        tor_aware_defaults_applied=tor_aware_defaults_applied,
        tor_source=tor_source,
        max_inflight_bytes_estimate=max_inflight_bytes_estimate,
    )
    try:
        for item in iterator:
            yielded_rows += 1
            yield item
    except Exception as exc:
        _log_parallel_event(
            logging.ERROR,
            "parallel_export_error",
            job_id=job_id,
            strategy=strategy,
            ordered_mode=order_mode,
            rows=yielded_rows,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            exception_type=type(exc).__name__,
        )
        raise
    _log_parallel_event(
        logging.INFO,
        "parallel_export_done",
        job_id=job_id,
        strategy=strategy,
        ordered_mode=order_mode,
        rows=yielded_rows,
        duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )


@dataclass(frozen=True)
class ParallelOptions:
    page_size: int = DEFAULT_PARALLEL_PAGE_SIZE
    max_workers: int | None = None
    ordered: bool | str | None = None
    prefetch: int | None = None
    inflight_limit: int | None = None
    ordered_max_in_flight: int | None = None
    ordered_window_pages: int = DEFAULT_ORDER_WINDOW_PAGES
    profile: str = DEFAULT_PARALLEL_PROFILE
    large_query_mode: bool = DEFAULT_LARGE_QUERY_MODE
    pagination: str = DEFAULT_PARALLEL_PAGINATION
    keyset_path: str | None = None
    keyset_batch_size: int = DEFAULT_KEYSET_BATCH_SIZE
    max_inflight_bytes_estimate: int | None = None


@dataclass(frozen=True)
class ResolvedParallelOptions:
    page_size: int
    max_workers: int
    order_mode: str
    prefetch: int
    inflight_limit: int
    ordered_max_in_flight: int
    ordered_window_pages: int
    profile: str
    large_query_mode: bool
    pagination: str
    keyset_path: str | None
    keyset_batch_size: int
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
        + "ordered_max_in_flight/ordered_window_pages/keyset_batch_size/max_inflight_bytes_estimate and valid values for "
        + "ordered/profile/pagination."
    )


def _legacy_parallel_overrides(
    *,
    page_size,
    max_workers,
    ordered,
    prefetch,
    inflight_limit,
    ordered_max_in_flight,
    ordered_window_pages,
    profile,
    large_query_mode,
    pagination,
    keyset_path,
    keyset_batch_size,
):
    overrides = []
    if page_size != DEFAULT_PARALLEL_PAGE_SIZE:
        overrides.append("page_size")
    if max_workers is not None:
        overrides.append("max_workers")
    if ordered is not None:
        overrides.append("ordered")
    if prefetch is not None:
        overrides.append("prefetch")
    if inflight_limit is not None:
        overrides.append("inflight_limit")
    if ordered_max_in_flight is not None:
        overrides.append("ordered_max_in_flight")
    if ordered_window_pages != DEFAULT_ORDER_WINDOW_PAGES:
        overrides.append("ordered_window_pages")
    if profile != DEFAULT_PARALLEL_PROFILE:
        overrides.append("profile")
    if large_query_mode != DEFAULT_LARGE_QUERY_MODE:
        overrides.append("large_query_mode")
    if pagination != DEFAULT_PARALLEL_PAGINATION:
        overrides.append("pagination")
    if keyset_path is not None:
        overrides.append("keyset_path")
    if keyset_batch_size != DEFAULT_KEYSET_BATCH_SIZE:
        overrides.append("keyset_batch_size")
    return overrides


def _warn_legacy_parallel_args(overrides: list[str], *, ignored: bool) -> None:
    global _LEGACY_PARALLEL_ARGS_WARNING_EMITTED
    if not overrides or _LEGACY_PARALLEL_ARGS_WARNING_EMITTED:
        return
    detail = ", ".join(overrides)
    message = (
        "Legacy parallel keyword arguments are deprecated and will be removed in a future release: "
        + detail
        + ". Pass parallel_options=ParallelOptions(...) instead."
    )
    if ignored:
        message += " Provided legacy arguments are ignored when parallel_options is supplied."
    _PARALLEL_LOG.warning(message)
    _LEGACY_PARALLEL_ARGS_WARNING_EMITTED = True


class Query(object):
    """
    A Class representing a structured database query
    ================================================

    Objects of this class have properties that model the
    attributes of the query, and methods for performing
    the request.

    SYNOPSIS
    --------

    example:

        >>> service = Service("https://www.flymine.org/query/service")
        >>> query = service.new_query()
        >>>
        >>> query.add_view("Gene.symbol", "Gene.pathways.name", "Gene.proteins.symbol")
        >>> query.add_sort_order("Gene.pathways.name")
        >>>
        >>> query.add_constraint("Gene", "LOOKUP", "eve")
        >>> query.add_constraint("Gene.pathways.name", "=", "Phosphate*")
        >>>
        >>> query.set_logic("A or B")
        >>>
        >>> for row in query.rows():
        ...     handle_row(row)

    OR, using an SQL style DSL:

        >>> s = Service("www.flymine.org/query")
        >>> query = s.query("Gene").\\
        ...           select("*", "pathways.*").\\
        ...           where("symbol", "=", "H").\\
        ...           outerjoin("pathways").\\
        ...           order_by("symbol")
        >>> for row in query.rows(start=10, size=5):
        ...     handle_row(row)

    OR, for a more SQL-alchemy, ORM style:

        >>> for gene in s.query(s.model.Gene).filter(s.model.Gene.symbol == ["zen", "H", "eve"]).add_columns(s.model.Gene.alleles):
        ...    handle(gene)

    Query objects represent structured requests for information over the
    database housed at the datawarehouse whose webservice you are querying.
    They utilise some of the concepts of relational databases, within an
    object-related ORM context. If you don't know what that means, don't
    worry: you don't need to write SQL, and the queries will be fast.

    To make things slightly more familiar to those with knowledge of SQL,
    some syntactical sugar is provided to make constructing queries a bit
    more recognisable.

    PRINCIPLES
    ----------

    The data model represents tables in the databases as classes, with records
    within tables as instances of that class. The columns of the database are
    the fields of that object::

      The Gene table - showing two records/objects
      +---------------------------------------------------+
      | id  | symbol  | length | cyto-location | organism |
      +----------------------------------------+----------+
      | 01  | eve     | 1539   | 46C10-46C10   |  01      |
      +----------------------------------------+----------+
      | 02  | zen     | 1331   | 84A5-84A5     |  01      |
      +----------------------------------------+----------+
      ...

      The organism table - showing one record/object
      +----------------------------------+
      | id  | name            | taxon id |
      +----------------------------------+
      | 01  | D. melanogaster | 7227     |
      +----------------------------------+

    Columns that contain a meaningful value are known as 'attributes' (in the
    tables above, that is everything except the id columns). The other columns
    (such as "organism" in the gene table) are ones that reference records of
    other tables (ie. other objects), and are called references. You can refer
    to any field in any class, that has a connection, however tenuous, with a
    table, by using dotted path notation::

      Gene.organism.name -> the name column in the organism table, referenced
                            by a record in the gene table

    These paths, and the connections between records and tables they represent,
    are the basis for the structure of InterMine queries.

    THE STUCTURE OF A QUERY
    -----------------------

    A query has two principle sets of properties:
      - its view: the set of output columns
      - its constraints: the set of rules for what to include

    A query must have at least one output column in its view, but constraints
    are optional - if you don't include any, you will get back every record
    from the table (every object of that type)

    In addition, the query must be coherent: if you have information about
    an organism, and you want a list of genes, then the "Gene" table
    should be the basis for your query, and as such the Gene class, which
    represents this table, should be the root of all the paths that appear in
    it:

    So, to take a simple example::

        I have an organism name, and I want a list of genes:

    The view is the list of things I want to know about those genes:

        >>> query.add_view("Gene.name")
        >>> query.add_view("Gene.length")
        >>> query.add_view("Gene.proteins.sequence.length")

    Note I can freely mix attributes and references, as long as every view ends
    in an attribute (a meaningful value). As a short-cut I can also write:

        >>> query.add_views("Gene.name", "Gene.length", "Gene.proteins.sequence.length")

    or:

        >>> query.add_views("Gene.name Gene.length Gene.proteins.sequence.length")

    They are all equivalent. You can also use common SQL style shortcuts such
    as "*" for all attribute fields:

        >>> query.add_views("Gene.*")

    You can also use "select" as a synonymn for "add_view"

    Now I can add my constraints. As, we mentioned, I have information about an
    organism, so:

        >>> query.add_constraint("Gene.organism.name", "=", "D. melanogaster")

    (note, here I can use "where" as a synonymn for "add_constraint")

    If I run this query, I will get literally millions of results -
    it needs to be filtered further:

        >>> query.add_constraint("Gene.proteins.sequence.length", "<", 500)

    If that doesn't restrict things enough I can add more filters:

        >>> query.add_constraint("Gene.symbol", "ONE OF", ["eve", "zen", "h"])

    Now I am guaranteed to get only information on genes I am interested in.

    Note, though, that because I have included the link (or "join") from
    Gene -> Protein, this, by default, means that I only want genes that have
    protein information associated with them. If in fact I want information on
    all genes, and just want to know the protein information if it is
    available, then I can specify that with:

        >>> query.add_join("Gene.proteins", "OUTER")

    And if perhaps my query is not as simple as a strict cumulative filter,
    but I want all D. mel genes that EITHER have a short protein sequence OR
    come from one of my favourite genes (as unlikely as that sounds), I can
    specify the logic for that too:

        >>> query.set_logic("A and (B or C)")

    Each letter refers to one of the constraints - the codes are assigned in
    the order you add the constraints. If you want to be absolutely certain
    about the constraints you mean, you can use the constraint objects
    themselves:

      >>> gene_is_eve = query.add_constraint("Gene.symbol", "=", "eve")
      >>> gene_is_zen = query.add_constraint("Gene.symbol", "=", "zne")
      >>>
      >>> query.set_logic(gene_is_eve | gene_is_zen)

    By default the logic is a straight cumulative filter
    (ie: A and B and C and D  and ...)

    Putting it all together:

       >>> query.add_view("Gene.name", "Gene.length", "Gene.proteins.sequence.length")
       >>> query.add_constraint("Gene.organism.name", "=", "D. melanogaster")
       >>> query.add_constraint("Gene.proteins.sequence.length", "<", 500)
       >>> query.add_constraint("Gene.symbol", "ONE OF", ["eve", "zen", "h"])
       >>> query.add_join("Gene.proteins", "OUTER")
       >>> query.set_logic("A and (B or C)")

    This can be made more concise and readable with a little DSL sugar:

        >>> query = service.query("Gene")
        >>> query.select("name", "length", "proteins.sequence.length").\
        ...       where('organism.name' '=', 'D. melanogaster').\
        ...       where("proteins.sequence.length", "<", 500).\
        ...       where('symbol', 'ONE OF', ['eve', 'h', 'zen']).\
        ...       outerjoin('proteins').\
        ...       set_logic("A and (B or C)")

    And the query is defined.

    Result Processing: Rows
    -----------------------

    calling ".rows()" on a query will return an iterator of rows, where each
    row is a ResultRow object, which can be treated as both a list and a
    dictionary.

    Which means you can refer to columns by name:

        >>> for row in query.rows():
        ...     print("name is %s" % (row["name"]))
        ...     print("length is %d" % (row["length"]))

    As well as using list indices:

        >>> for row in query.rows():
        ...     print("The first column is %s" % (row[0]))

    Iterating over a row iterates over the cell values as a list:

        >>> for row in query.rows():
        ...     for column in row:
        ...         do_something(column)

    Here each row will have a gene name, a gene length, and a sequence length
    eg:

        >>> print(row.to_l)
        ["even skipped", "1359", "376"]

    To make that clearer, you can ask for a dictionary instead of a list:

        >>> for row in query.rows()
        ...       print(row.to_d)
        {"Gene.name":"even skipped","Gene.length":"1359","Gene.proteins.sequence.length":"376"}


    If you just want the raw results, for printing to a file, or for piping to
    another program, you can request the results in one of these
    formats: 'json', 'rr', 'tsv', 'jsonobjects', 'jsonrows', 'list', 'dict',
             'csv'

        >>> for row in query.result("<format name>", size = <size>)
        ...     print(row)


    Result Processing: Results
    --------------------------

    Results can also be processing on a record by record basis. If you have a
    query that has output columns of "Gene.symbol", "Gene.pathways.name" and
    "Gene.proteins.proteinDomains.primaryIdentifier", than processing it by
    records will return one object per gene, and that gene will have a property
    named "pathways" which contains objects which have a name property.
    Likewise there will be a proteins property which holds a list of
    proteinDomains which all have a primaryIdentifier property, and so on.
    This allows a more object orientated approach to database records,
    familiar to users of other ORMs.

    This is the format used when you choose to iterate over a query directly,
    or can be explicitly chosen by invoking L{intermine314.query.Query.results}:

        >>> for gene in query:
        ...    print(gene.name, map(lambda x: x.name, gene.pathways))

    The structure of the object and the information it contains depends
    entirely on the output columns selected. The values may be None, of course,
    but also any valid values of an object (according to the data model) will
    also be None if they were not selected for output. Attempts to access
    invalid properties (such as gene.favourite_colour) will cause exceptions
    to be thrown.

    Getting us to Generate your Code
    --------------------------------

    Not that you have to actually write any of this! The webapp will happily
    generate the code for any query (and template) you can build in it. A good
    way to get started is to use the webapp to generate your code, and then
    run it as scripts to speed up your queries. You can always tinker with and
    edit the scripts you download.

    To get generated queries, look for the "python" link at the bottom of
    query-builder and template form pages, it looks a bit like this::

      . +=====================================+=============
        |                                     |
        |    Perl  |  Python  |  Java [Help]  |
        |                                     |
        +==============================================

    """

    SO_SPLIT_PATTERN = re.compile("\\s*(asc|desc)\\s*", re.I)
    LOGIC_SPLIT_PATTERN = re.compile("\\s*(?:and|or|\\(|\\))\\s*", re.I)
    TRAILING_OP_PATTERN = re.compile("\\s*(and|or)\\s*$", re.I)
    LEADING_OP_PATTERN = re.compile("^\\s*(and|or)\\s*", re.I)
    ORPHANED_OP_PATTERN = re.compile("(?:\\(\\s*(?:and|or)\\s*|\\s*(?:and|or)\\s*\\))", re.I)

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
        self._logic_parser = constraints.LogicParser(self)
        self._logic = None
        self.constraint_factory = constraints.ConstraintFactory()

        # Set up sugary aliases
        self.c = self.column
        self.filter = self.where
        self.add_column = self.add_view
        self.add_columns = self.add_view
        self.add_views = self.add_view
        self.add_to_select = self.add_view
        self.order_by = self.add_sort_order
        self.all = self.get_results_list
        self.size = self.count
        self.summarize = self.summarise

    def __iter__(self):
        """Return an iterator over all the objects returned by this query"""
        return self.results("jsonobjects")

    def __len__(self):
        """Return the number of rows this query will return."""
        return self.count()

    def __sub__(self, other):
        """
        Construct a new list from the symmetric difference of these things
        """
        return self.service._list_manager.subtract([self], [other])

    def __xor__(self, other):
        """Calculate the symmetric difference of this query and another"""
        return self.service._list_manager.xor([self, other])

    def __and__(self, other):
        """
        Intersect this query and another query or list
        """
        return self.service._list_manager.intersect([self, other])

    def __or__(self, other):
        """
        Return the union of this query and another query or list.
        """
        return self.service._list_manager.union([self, other])

    def __add__(self, other):
        """
        Return the union of this query and another query or list
        """
        return self.service._list_manager.union([self, other])

    @classmethod
    def from_xml(cls, xml, *args, **kwargs):
        """
        Deserialise a query serialised to XML
        =====================================

        This method is used to instantiate serialised queries.
        It is used by intermine314.webservice.Service objects
        to instantiate Template objects and it can be used
        to read in queries you have saved to a file.

        @param xml: The xml as a file name, url, or string

        @raise QueryParseError: if the query cannot be parsed
        @raise ModelError: if the query has illegal paths in it
        @raise ConstraintError: if the constraints don't make sense

        @rtype: L{Query}
        """
        obj = cls(*args, **kwargs)
        obj.do_verification = False
        with closing(openAnything(xml)) as f:
            doc = minidom.parse(f)

        queries = doc.getElementsByTagName("query")
        if len(queries) != 1:
            raise QueryParseError(
                "wrong number of queries in xml. " + "Only one <query> element is allowed. Found %d" % len(queries)
            )
        q = queries[0]
        obj.name = q.getAttribute("name")
        obj.description = q.getAttribute("longDescription")
        obj.add_view(q.getAttribute("view"))
        for p in q.getElementsByTagName("pathDescription"):
            path = p.getAttribute("pathString")
            description = p.getAttribute("description")
            obj.add_path_description(path, description)
        for j in q.getElementsByTagName("join"):
            path = j.getAttribute("path")
            style = j.getAttribute("style")
            obj.add_join(path, style)
        for c in q.getElementsByTagName("constraint"):
            args = {}
            args["path"] = c.getAttribute("path")
            if args["path"] is None:
                if c.parentNode.tagName != "node":
                    msg = "Constraints must have a path"
                    raise QueryParseError(msg)
                args["path"] = c.parentNode.getAttribute("path")
            args["op"] = c.getAttribute("op")
            args["value"] = c.getAttribute("value")
            args["code"] = c.getAttribute("code")
            args["subclass"] = c.getAttribute("type")
            args["editable"] = c.getAttribute("editable")
            args["optional"] = c.getAttribute("switchable")
            args["extra_value"] = c.getAttribute("extraValue")
            args["loopPath"] = c.getAttribute("loopPath")
            values = []
            for val_e in c.getElementsByTagName("value"):
                texts = []
                for node in val_e.childNodes:
                    if node.nodeType == node.TEXT_NODE:
                        texts.append(node.data)
                values.append(" ".join(texts))
            if len(values) > 0:
                args["values"] = values
            args = dict((k, v) for k, v in list(args.items()) if v is not None and v != "")
            if "loopPath" in args:
                args["op"] = {"=": "IS", "!=": "IS NOT"}.get(args["op"])
            con = obj.add_constraint(**args)
            if not con:
                raise ConstraintError("error adding constraint with args: " + args)

        def group(iterator, count):
            itr = iter(iterator)
            while True:
                try:
                    yield tuple([next(itr) for i in range(count)])
                except StopIteration:
                    return

        if q.getAttribute("sortOrder") is not None:
            sos = Query.SO_SPLIT_PATTERN.split(q.getAttribute("sortOrder"))
            if len(sos) == 1:
                if sos[0] in obj.views:  # Be tolerant of irrelevant sort-orders
                    obj.add_sort_order(sos[0])
            else:
                sos.pop()  # Get rid of empty string at end
                for path, direction in group(sos, 2):
                    if path in obj.views:  # Be tolerant of irrelevant so.
                        obj.add_sort_order(path, direction)

        if q.getAttribute("constraintLogic") is not None:
            obj._set_questionable_logic(q.getAttribute("constraintLogic"))

        obj.verify()

        return obj

    def _set_questionable_logic(self, questionable_logic):
        """Attempts to sanity check the logic argument before it is set"""
        logic = questionable_logic
        used_codes = set(self.constraint_dict.keys())
        logic_codes = set(Query.LOGIC_SPLIT_PATTERN.split(questionable_logic))
        if "" in logic_codes:
            logic_codes.remove("")
        irrelevant_codes = logic_codes - used_codes
        for c in irrelevant_codes:
            pattern = re.compile("\\b" + c + "\\b", re.I)
            logic = pattern.sub("", logic)
        # Remove empty groups
        logic = re.sub("\\((:?and|or|\\s)*\\)", "", logic)
        # Remove trailing and leading operators
        logic = Query.LEADING_OP_PATTERN.sub("", logic)
        logic = Query.TRAILING_OP_PATTERN.sub("", logic)
        for x in range(2):  # repeat, as this process can leave doubles
            for left, right in LOGIC_PRODUCT:
                if left == right:
                    repl = left
                else:
                    repl = "and"
                pattern = re.compile(left + "\\s*" + right, re.I)
                logic = pattern.sub(repl, logic)
        logic = Query.ORPHANED_OP_PATTERN.sub(lambda x: "(" if "(" in x.group(0) else ")", logic)
        logic = logic.strip().lstrip()
        logic = Query.LEADING_OP_PATTERN.sub("", logic)
        logic = Query.TRAILING_OP_PATTERN.sub("", logic)
        try:
            if len(logic) > 0 and logic not in ["and", "or"]:
                self.set_logic(logic)
        except Exception as e:
            raise Exception(
                "Error parsing logic string "
                + repr(questionable_logic)
                + " (which is "
                + repr(logic)
                + " after irrelevant codes have been removed)"
                + " with available codes: "
                + repr(list(used_codes))
                + " because: "
                + e.message
            )

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

        Also available as:
         - add_views
         - add_column
         - add_columns
         - add_to_select

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
            if isinstance(args[0], tuple):
                con = self.constraint_factory.make_constraint(*args[0])
            else:
                try:
                    con = self.constraint_factory.make_constraint(*args[0].vargs, **args[0].kwargs)
                except AttributeError:
                    con = args[0]
        else:
            if len(args) == 0 and len(kwargs) == 1:
                k, v = list(kwargs.items())[0]
                d = {"path": k}
                if v in constraints.UnaryConstraint.OPS:
                    d["op"] = v
                else:
                    d["op"] = "="
                    d["value"] = v
                kwargs = d

            if len(args) and args[0] in self.constraint_factory.reference_ops:
                args = [self.root] + list(args)

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

        Also available as Query.filter
        """
        c = self.clone()
        try:
            for conset in cons:
                codeds = c.coded_constraints
                lstr = str(c.get_logic()) + " AND " if codeds else ""
                start_c = chr(ord(codeds[-1].code) + 1) if codeds else "A"
                for con in conset:
                    c.add_constraint(*con.vargs, **con.kwargs)
                try:
                    c.set_logic(lstr + conset.as_logic(start=start_c))
                except constraints.EmptyLogicError:
                    pass
            for path, value in list(kwargs.items()):
                c.add_constraint(path, "=", value)
        except AttributeError:
            c.add_constraint(*cons, **kwargs)
        return c

    def column(self, col):
        """
        Return a Column object suitable for using to construct constraints with
        =======================================================================

        This method is part of the SQLAlchemy style API.

        Also available as Query.c
        """
        return self.model.column(self.prefix_path(str(col)), self.get_subclass_dict(), self)

    def verify_constraint_paths(self, cons=None):
        """
        Check that the constraints are valid
        ====================================

        This method will check the path attribute of each constraint.
        In addition it will:
          - Check that BinaryConstraints and MultiConstraints have an
            Attribute as their path
          - Check that TernaryConstraints have a Reference as theirs
          - Check that SubClassConstraints have a correct subclass relationship
          - Check that LoopConstraints have a valid loopPath, of a compatible
            type
          - Check that ListConstraints refer to an object
          - Don't even try to check RangeConstraints: these have variable
            semantics

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
            if isinstance(con, constraints.RangeConstraint):
                # No verification done on these, beyond checking its path, of course.
                pass
            elif isinstance(con, constraints.IsaConstraint):
                if pathA.get_class() is None:
                    raise ConstraintError("'" + str(pathA) + "' does not represent a class, or a reference to a class")
                for c in con.values:
                    if c not in self.model.classes:
                        raise ConstraintError(
                            "Illegal constraint: " + repr(con) + " '" + str(c) + "' is not a class in this model"
                        )
            elif isinstance(con, constraints.TernaryConstraint):
                if pathA.get_class() is None:
                    raise ConstraintError("'" + str(pathA) + "' does not represent a class, or a reference to a class")
            elif isinstance(con, constraints.BinaryConstraint) or isinstance(con, constraints.MultiConstraint):
                if not pathA.is_attribute():
                    raise ConstraintError("'" + str(pathA) + "' does not represent an attribute")
            elif isinstance(con, constraints.SubClassConstraint):
                pathB = self.model.make_path(con.subclass, self.get_subclass_dict())
                if not pathB.get_class().isa(pathA.get_class()):
                    raise ConstraintError("'" + con.subclass + "' is not a subclass of '" + con.path + "'")
            elif isinstance(con, constraints.LoopConstraint):
                pathB = self.model.make_path(con.loopPath, self.get_subclass_dict())
                for path in [pathA, pathB]:
                    if not path.get_class():
                        raise ConstraintError("'" + str(path) + "' does not refer to an object")
                (classA, classB) = (pathA.get_class(), pathB.get_class())
                if not classA.isa(classB) and not classB.isa(classA):
                    raise ConstraintError("the classes are of incompatible types: " + str(classA) + "," + str(classB))
            elif isinstance(con, constraints.ListConstraint):
                if not pathA.get_class():
                    raise ConstraintError("'" + str(pathA) + "' does not refer to an object")

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
        """
        Returns the list of constraints that have a code
        ================================================

        Query.coded_constraints S{->} list(intermine314.constraints.CodedConstraint)

        This returns an up to date list of the constraints that can
        be used in a logic expression. The only kind of constraint
        that this excludes, at present, is SubClassConstraints

        @rtype: list(L{intermine314.constraints.CodedConstraint})
        """
        return sorted(list(self.constraint_dict.values()), key=lambda con: con.code)

    def get_logic(self):
        """
        Returns the logic expression for the query
        ==========================================

        This returns the up to date logic expression. The default
        value is the representation of all coded constraints and'ed together.

        If the logic is empty and there are no constraints, returns an
        empty string.

        The LogicGroup object stringifies to a string that can be parsed to
        obtain itself (eg: "A and (B or C or D)").

        @rtype: L{intermine314.constraints.LogicGroup}
        """
        if self._logic is None:
            if len(self.coded_constraints) > 0:
                return reduce(lambda x, y: x + y, self.coded_constraints)
            else:
                return ""
        else:
            return self._logic

    def set_logic(self, value):
        """
        Sets the Logic given the appropriate input
        ==========================================

        example::

          Query.set_logic("A and (B or C)")

        This sets the logic to the appropriate value. If the value is
        already a LogicGroup, it is accepted, otherwise
        the string is tokenised and parsed.

        The logic is then validated with a call to validate_logic()

        raise LogicParseError: if there is a syntax error in the logic
        """
        if isinstance(value, constraints.LogicGroup):
            logic = value
        else:
            try:
                logic = self._logic_parser.parse(value)
            except constraints.EmptyLogicError:
                if self.coded_constraints:
                    raise
                else:
                    return self
        if self.do_verification:
            self.validate_logic(logic)
        self._logic = logic
        return self

    def validate_logic(self, logic=None):
        """
        Validates the query logic
        =========================

        Attempts to validate the logic by checking
        that every coded_constraint is included
        at least once

        @raise QueryError: if not every coded constraint is represented
        """
        if logic is None:
            logic = self._logic
        logic_codes = set(logic.get_codes())
        for con in self.coded_constraints:
            if con.code not in logic_codes:
                raise QueryError("Constraint " + con.code + repr(con) + " is not mentioned in the logic: " + str(logic))

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

        Also available as Query.order_by
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
            if isinstance(c, constraints.SubClassConstraint):
                subclass_dict[c.path] = c.subclass
        return subclass_dict

    def results(self, row="object", start=0, size=None, summary_path=None):
        """
        Return an iterator over result rows
        ===================================

        Usage::

          >>> query = service.model.Gene.select("symbol", "length")
          >>> total = 0
          >>> for gene in query.results():
          ...    print(gene.symbol)           # handle strings
          ...    total += gene.length        # handle numbers
          >>> for row in query.results(row="rr"):
          ...    print(row["symbol"])         # handle strings by dict index
          ...    total += row["length"]      # handle numbers by dict index
          ...    print(row["Gene.symbol"])    # handle strings by full dict index
          ...    total += row["Gene.length"] # handle numbers by full dict index
          ...    print(row[0])                # handle strings by list index
          ...    total += row[1]             # handle numbers by list index
          >>> for d in query.results(row="dict"):
          ...    print(row["Gene.symbol"])    # handle strings
          ...    total += row["Gene.length"] # handle numbers
          >>> for l in query.results(row="list"):
          ...    print(row[0])                # handle strings
          ...    total += row[1]             # handle numbers
          >>> import csv
          >>> csv_reader = csv.reader(q.results(row="csv"), delimiter=",", quotechar='"')
          >>> for row in csv_reader:
          ...    print(row[0])                # handle strings
          ...    length_sum += int(row[1])   # handle numbers
          >>> tsv_reader = csv.reader(q.results(row="tsv"), delimiter="\t")
          >>> for row in tsv_reader:
          ...    print(row[0])                # handle strings
          ...    length_sum += int(row[1])   # handle numbers

        This is the general method that allows access to any of the available
        result formats. The example above shows the ways these differ in terms
        of accessing fields of the rows, as well as dealing with different
        data types. Results can either be retrieved as typed values
        (jsonobjects, rr ['ResultRows'], dict, list), or as lists of strings
        (csv, tsv) which then require further parsing. The default format for
        this method is "objects", where information is grouped by its
        relationships. The other main format is "rr", which stands for
        'ResultRows', and can be accessed directly through the L{rows} method.

        Note that when requesting object based results (the default), if your
        query contains any kind of collection, it is highly likely that start
        and size won't do what you think, as they operate only on the
        underlying rows used to build up the returned objects. If you want rows
        back, you are recommeded to use the simpler rows method.

        If no views have been specified, all attributes of the root class
        are selected for output.

        @param row: The format for each result. One of "object", "rr",
                    "dict", "list", "tsv", "csv", "jsonrows", "jsonobjects"
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

        if "object" in row:
            for c in self.coded_constraints:
                p = to_run.column(c.path)._path
                from_p = p if p.end_class is not None else p.prefix()
                if not [v for v in to_run.views if v.startswith(str(from_p))]:
                    if p.is_attribute():
                        to_run.add_view(p)
                    else:
                        to_run.add_view(p.append("id"))

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
        if row == "dataframe":
            row = "dict"

        return to_run.service.get_results(path, params, row, view, cld)

    def _iter_result_rows(
        self,
        start=0,
        size=None,
        row="dict",
        parallel=False,
        page_size=DEFAULT_PARALLEL_PAGE_SIZE,
        max_workers=None,
        ordered=None,
        prefetch=None,
        inflight_limit=None,
        ordered_max_in_flight=None,
        ordered_window_pages=DEFAULT_ORDER_WINDOW_PAGES,
        profile=DEFAULT_PARALLEL_PROFILE,
        large_query_mode=DEFAULT_LARGE_QUERY_MODE,
        pagination=DEFAULT_PARALLEL_PAGINATION,
        keyset_path=None,
        keyset_batch_size=DEFAULT_KEYSET_BATCH_SIZE,
        parallel_options=None,
    ):
        if parallel:
            options = self._coerce_parallel_options(
                parallel_options=parallel_options,
                page_size=page_size,
                max_workers=max_workers,
                ordered=ordered,
                prefetch=prefetch,
                inflight_limit=inflight_limit,
                ordered_max_in_flight=ordered_max_in_flight,
                ordered_window_pages=ordered_window_pages,
                profile=profile,
                large_query_mode=large_query_mode,
                pagination=pagination,
                keyset_path=keyset_path,
                keyset_batch_size=keyset_batch_size,
            )
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
        parallel=False,
        page_size=DEFAULT_PARALLEL_PAGE_SIZE,
        max_workers=None,
        ordered=None,
        prefetch=None,
        inflight_limit=None,
        ordered_max_in_flight=None,
        ordered_window_pages=DEFAULT_ORDER_WINDOW_PAGES,
        profile=DEFAULT_PARALLEL_PROFILE,
        large_query_mode=DEFAULT_LARGE_QUERY_MODE,
        pagination=DEFAULT_PARALLEL_PAGINATION,
        keyset_path=None,
        keyset_batch_size=DEFAULT_KEYSET_BATCH_SIZE,
        parallel_options=None,
    ):
        """
        Yield rows in exporter-friendly modes.

        ``mode="dict"`` and ``mode="list"`` are optimized for lower allocation
        overhead compared to interactive ``ResultRow`` wrappers.
        """
        if mode not in VALID_ITER_ROW_MODES:
            choices = ", ".join(sorted(VALID_ITER_ROW_MODES))
            raise ValueError(f"mode must be one of: {choices}")
        options = self._coerce_parallel_options(
            parallel_options=parallel_options,
            page_size=page_size,
            max_workers=max_workers,
            ordered=ordered,
            prefetch=prefetch,
            inflight_limit=inflight_limit,
            ordered_max_in_flight=ordered_max_in_flight,
            ordered_window_pages=ordered_window_pages,
            profile=profile,
            large_query_mode=large_query_mode,
            pagination=pagination,
            keyset_path=keyset_path,
            keyset_batch_size=keyset_batch_size,
        )
        return self._iter_result_rows(
            start=start,
            size=size,
            row=mode,
            parallel=parallel,
            parallel_options=options,
        )

    def iter_batches(
        self,
        start=0,
        size=None,
        batch_size=DEFAULT_BATCH_SIZE,
        row_mode="dict",
        *,
        parallel=False,
        page_size=DEFAULT_PARALLEL_PAGE_SIZE,
        max_workers=None,
        ordered=None,
        prefetch=None,
        inflight_limit=None,
        ordered_max_in_flight=None,
        ordered_window_pages=DEFAULT_ORDER_WINDOW_PAGES,
        profile=DEFAULT_PARALLEL_PROFILE,
        large_query_mode=DEFAULT_LARGE_QUERY_MODE,
        pagination=DEFAULT_PARALLEL_PAGINATION,
        keyset_path=None,
        keyset_batch_size=DEFAULT_KEYSET_BATCH_SIZE,
        parallel_options=None,
    ):
        """
        Yield result rows as lists of dicts in batches.

        Usage::
          >>> for batch in query.iter_batches(batch_size=2000):
          ...     process_batch(batch)
        """
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
            parallel=parallel,
            page_size=page_size,
            max_workers=max_workers,
            ordered=ordered,
            prefetch=prefetch,
            inflight_limit=inflight_limit,
            ordered_max_in_flight=ordered_max_in_flight,
            ordered_window_pages=ordered_window_pages,
            profile=profile,
            large_query_mode=large_query_mode,
            pagination=pagination,
            keyset_path=keyset_path,
            keyset_batch_size=keyset_batch_size,
            parallel_options=parallel_options,
        )
        for row in row_iter:
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def rows(self, start=0, size=None, row="rr"):
        """
        Return the results as rows of data
        ==================================

        This is a shortcut for results("rr")

        Usage::

          >>> for row in query.rows(start=10, size=10):
          ...     print(row["proteins.name"])

        @param start: the index of the first result to return (default = 0)
        @type start: int
        @param size: The maximum number of results to return (default = all)
        @type size: int
        @rtype: iterable<intermine314.webservice.ResultRow>
        """
        return self.results(row=row, start=start, size=size)

    def _iter_batches_kwargs(
        self,
        *,
        start=0,
        size=None,
        batch_size=DEFAULT_BATCH_SIZE,
        row_mode="dict",
        parallel=False,
        page_size=DEFAULT_PARALLEL_PAGE_SIZE,
        max_workers=None,
        ordered=None,
        prefetch=None,
        inflight_limit=None,
        ordered_max_in_flight=None,
        ordered_window_pages=DEFAULT_ORDER_WINDOW_PAGES,
        profile=DEFAULT_PARALLEL_PROFILE,
        large_query_mode=DEFAULT_LARGE_QUERY_MODE,
        pagination=DEFAULT_PARALLEL_PAGINATION,
        keyset_path=None,
        keyset_batch_size=DEFAULT_KEYSET_BATCH_SIZE,
        parallel_options=None,
    ):
        options = self._coerce_parallel_options(
            parallel_options=parallel_options,
            page_size=page_size,
            max_workers=max_workers,
            ordered=ordered,
            prefetch=prefetch,
            inflight_limit=inflight_limit,
            ordered_max_in_flight=ordered_max_in_flight,
            ordered_window_pages=ordered_window_pages,
            profile=profile,
            large_query_mode=large_query_mode,
            pagination=pagination,
            keyset_path=keyset_path,
            keyset_batch_size=keyset_batch_size,
        )
        return {
            "start": start,
            "size": size,
            "batch_size": batch_size,
            "row_mode": row_mode,
            "parallel": parallel,
            "parallel_options": options,
        }

    def _write_single_parquet_from_parts(self, staged_dir, target, compression):
        write_single_parquet_from_parts(
            staged_dir=staged_dir,
            target=target,
            compression=compression,
            polars_module=_require_polars("Query.to_parquet()"),
            duckdb_module=_optional_duckdb(),
            duckdb_quote=_duckdb_quote,
        )

    def dataframe(
        self,
        start=0,
        size=None,
        batch_size=DEFAULT_BATCH_SIZE,
        *,
        parallel=False,
        page_size=DEFAULT_PARALLEL_PAGE_SIZE,
        max_workers=None,
        ordered=None,
        prefetch=None,
        inflight_limit=None,
        ordered_max_in_flight=None,
        ordered_window_pages=DEFAULT_ORDER_WINDOW_PAGES,
        profile=DEFAULT_PARALLEL_PROFILE,
        large_query_mode=DEFAULT_LARGE_QUERY_MODE,
        pagination=DEFAULT_PARALLEL_PAGINATION,
        keyset_path=None,
        keyset_batch_size=DEFAULT_KEYSET_BATCH_SIZE,
        final_rechunk=False,
        parallel_options=None,
    ):
        """
        Returns a polars.DataFrame
        ==================================

        Usage::
          >>> query.dataframe(parallel=True, pagination="auto")

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
        polars_module = _require_polars("Query.dataframe()")
        options = self._coerce_parallel_options(
            parallel_options=parallel_options,
            page_size=page_size,
            max_workers=max_workers,
            ordered=ordered,
            prefetch=prefetch,
            inflight_limit=inflight_limit,
            ordered_max_in_flight=ordered_max_in_flight,
            ordered_window_pages=ordered_window_pages,
            profile=profile,
            large_query_mode=large_query_mode,
            pagination=pagination,
            keyset_path=keyset_path,
            keyset_batch_size=keyset_batch_size,
        )
        iter_kwargs = self._iter_batches_kwargs(
            start=start,
            size=size,
            batch_size=batch_size,
            row_mode="dict",
            parallel=parallel,
            parallel_options=options,
        )
        frame_iter = (polars_module.from_dicts(batch) for batch in self.iter_batches(**iter_kwargs))
        try:
            first = next(frame_iter)
        except StopIteration:
            return polars_module.DataFrame()
        return polars_module.concat(
            chain((first,), frame_iter),
            how="diagonal_relaxed",
            rechunk=bool(final_rechunk),
        )

    def to_parquet(
        self,
        path,
        start=0,
        size=None,
        batch_size=DEFAULT_EXPORT_BATCH_SIZE,
        compression="zstd",
        single_file=False,
        *,
        parallel=False,
        page_size=DEFAULT_PARALLEL_PAGE_SIZE,
        max_workers=None,
        ordered=None,
        prefetch=None,
        inflight_limit=None,
        ordered_max_in_flight=None,
        ordered_window_pages=DEFAULT_ORDER_WINDOW_PAGES,
        profile=DEFAULT_PARALLEL_PROFILE,
        large_query_mode=DEFAULT_LARGE_QUERY_MODE,
        pagination=DEFAULT_PARALLEL_PAGINATION,
        keyset_path=None,
        keyset_batch_size=DEFAULT_KEYSET_BATCH_SIZE,
        parallel_options=None,
    ):
        """
        Stream results to Parquet files.

        Usage::
          >>> query.to_parquet("results_parquet", batch_size=5000, parallel=True, pagination="auto")
        """
        polars_module = _require_polars("Query.to_parquet()")
        options = self._coerce_parallel_options(
            parallel_options=parallel_options,
            page_size=page_size,
            max_workers=max_workers,
            ordered=ordered,
            prefetch=prefetch,
            inflight_limit=inflight_limit,
            ordered_max_in_flight=ordered_max_in_flight,
            ordered_window_pages=ordered_window_pages,
            profile=profile,
            large_query_mode=large_query_mode,
            pagination=pagination,
            keyset_path=keyset_path,
            keyset_batch_size=keyset_batch_size,
        )
        compression = compression.lower()
        if compression not in VALID_PARQUET_COMPRESSIONS:
            choices = ", ".join(sorted(VALID_PARQUET_COMPRESSIONS))
            raise ValueError(f"Unsupported Parquet compression: {compression}. Choose one of: {choices}")
        target = Path(path)
        if single_file:
            with TemporaryDirectory() as tmp:
                staged_dir = Path(tmp) / "parts"
                self.to_parquet(
                    staged_dir,
                    start=start,
                    size=size,
                    batch_size=batch_size,
                    compression=compression,
                    single_file=False,
                    parallel=parallel,
                    parallel_options=options,
                )
                self._write_single_parquet_from_parts(staged_dir, target, compression)
            return str(target)
        if target.exists() and target.is_file():
            raise ValueError("path must be a directory when single_file is False")
        target.mkdir(parents=True, exist_ok=True)
        for stale in target.glob("part-*.parquet"):
            stale.unlink()
        part = 0
        iter_kwargs = self._iter_batches_kwargs(
            start=start,
            size=size,
            batch_size=batch_size,
            row_mode="dict",
            parallel=parallel,
            parallel_options=options,
        )
        for batch in self.iter_batches(**iter_kwargs):
            frame = _polars_from_dicts_with_full_inference(polars_module, batch)
            part_path = target / "part-{0:05d}.parquet".format(part)
            frame.write_parquet(str(part_path), compression=compression)
            part += 1
        return str(target)

    def to_duckdb(
        self,
        path,
        start=0,
        size=None,
        batch_size=DEFAULT_EXPORT_BATCH_SIZE,
        compression="zstd",
        single_file=False,
        database=":memory:",
        table="results",
        *,
        parallel=False,
        page_size=DEFAULT_PARALLEL_PAGE_SIZE,
        max_workers=None,
        ordered=None,
        prefetch=None,
        inflight_limit=None,
        ordered_max_in_flight=None,
        ordered_window_pages=DEFAULT_ORDER_WINDOW_PAGES,
        profile=DEFAULT_PARALLEL_PROFILE,
        large_query_mode=DEFAULT_LARGE_QUERY_MODE,
        pagination=DEFAULT_PARALLEL_PAGINATION,
        keyset_path=None,
        keyset_batch_size=DEFAULT_KEYSET_BATCH_SIZE,
        parallel_options=None,
    ):
        """
        Materialize results to Parquet and expose them via DuckDB.

        Usage::
          >>> con = query.to_duckdb("results_parquet", parallel=True, pagination="auto")
          >>> con.execute("select count(*) from results").fetchall()
        """
        duckdb_module = _require_duckdb("Query.to_duckdb()")
        options = self._coerce_parallel_options(
            parallel_options=parallel_options,
            page_size=page_size,
            max_workers=max_workers,
            ordered=ordered,
            prefetch=prefetch,
            inflight_limit=inflight_limit,
            ordered_max_in_flight=ordered_max_in_flight,
            ordered_window_pages=ordered_window_pages,
            profile=profile,
            large_query_mode=large_query_mode,
            pagination=pagination,
            keyset_path=keyset_path,
            keyset_batch_size=keyset_batch_size,
        )
        if not DUCKDB_IDENTIFIER_PATTERN.fullmatch(table):
            raise ValueError("table must be a valid SQL identifier (letters, numbers, underscore)")
        parquet_path = self.to_parquet(
            path,
            start=start,
            size=size,
            batch_size=batch_size,
            compression=compression,
            single_file=single_file,
            parallel=parallel,
            parallel_options=options,
        )
        target = Path(parquet_path)
        if target.is_dir():
            parquet_glob = str(target / "*.parquet")
        else:
            parquet_glob = str(target)
        parquet_glob_sql = _duckdb_quote(parquet_glob)
        con = duckdb_module.connect(database=database)
        con.execute(f'CREATE OR REPLACE VIEW "{table}" AS SELECT * FROM read_parquet({parquet_glob_sql})')
        return con

    def _run_parallel_offset(
        self,
        row="dict",
        start=0,
        size=None,
        page_size=DEFAULT_PARALLEL_PAGE_SIZE,
        max_workers=None,
        order_mode="ordered",
        inflight_limit=DEFAULT_PARALLEL_WORKERS,
        ordered_window_pages=DEFAULT_ORDER_WINDOW_PAGES,
        ordered_max_in_flight=None,
        max_inflight_bytes_estimate=None,
        job_id=None,
    ):
        if size is None:
            total = max(self.count() - start, 0)
        else:
            total = size
        if total <= 0:
            return iter(())
        stop = start + total
        offsets = range(start, stop, page_size)

        def fetch_page(offset):
            limit = min(page_size, stop - offset)
            rows = list(islice(self.results(row=row, start=offset, size=limit), limit))
            return offset, rows

        def ordered_iterator():
            with ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix=DEFAULT_QUERY_THREAD_NAME_PREFIX
            ) as executor:
                max_pending = max(1, int(ordered_max_in_flight or inflight_limit))
                cap = _AdaptiveInflightCap(
                    row_limit=max_pending,
                    max_inflight_bytes_estimate=max_inflight_bytes_estimate,
                )
                if _EXECUTOR_MAP_SUPPORTS_BUFFERSIZE and max_inflight_bytes_estimate is None:
                    for _, rows in executor.map(fetch_page, offsets, buffersize=max_pending):
                        for item in rows:
                            yield item
                    _log_parallel_event(
                        logging.DEBUG,
                        "parallel_ordered_scheduler_stats",
                        job_id=job_id,
                        in_flight=0,
                        completed_buffer_size=0,
                        max_in_flight=max_pending,
                        inflight_bytes_budget=max_inflight_bytes_estimate,
                        **cap.stats_fields(),
                    )
                    return

                pending = set()
                future_to_offset = {}
                future_reserved_bytes = {}
                completed_by_offset = {}
                offset_iter = iter(offsets)
                next_expected = start
                max_pending_seen = 0
                max_completed_buffer = 0

                def submit_offset(offset):
                    nonlocal max_pending_seen
                    future = executor.submit(fetch_page, offset)
                    pending.add(future)
                    future_to_offset[future] = offset
                    future_reserved_bytes[future] = cap.reserve_on_submit()
                    max_pending_seen = max(max_pending_seen, len(pending))
                    return future

                try:
                    while cap.can_submit(len(pending)):
                        offset = next(offset_iter)
                        submit_offset(offset)
                except StopIteration:
                    pass

                while pending:
                    future = next(as_completed(pending))
                    pending.remove(future)
                    offset = future_to_offset.pop(future, None)
                    reserved_bytes = future_reserved_bytes.pop(future, 0.0)
                    if offset is None:
                        raise RuntimeError("Parallel ordered scheduler lost offset mapping for completed future")
                    try:
                        _, rows = future.result()
                    except Exception as exc:
                        cap.observe_completion(reserved_bytes, None)
                        raise RuntimeError(f"parallel ordered fetch failed at offset={offset}") from exc
                    cap.observe_completion(reserved_bytes, rows)
                    completed_by_offset[offset] = rows
                    max_completed_buffer = max(max_completed_buffer, len(completed_by_offset))

                    while next_expected in completed_by_offset:
                        for item in completed_by_offset.pop(next_expected):
                            yield item
                        next_expected += page_size

                    try:
                        while cap.can_submit(len(pending)):
                            offset = next(offset_iter)
                            submit_offset(offset)
                    except StopIteration:
                        pass

                if completed_by_offset:
                    # Defensive deterministic flush for non-standard offset sequences.
                    for buffered_offset in sorted(completed_by_offset):
                        for item in completed_by_offset[buffered_offset]:
                            yield item

                _log_parallel_event(
                    logging.DEBUG,
                    "parallel_ordered_scheduler_stats",
                    job_id=job_id,
                    in_flight=0,
                    completed_buffer_size=max_completed_buffer,
                    max_in_flight=max_pending_seen,
                    inflight_bytes_budget=max_inflight_bytes_estimate,
                    **cap.stats_fields(),
                )

        def unordered_iterator():
            with ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix=DEFAULT_QUERY_THREAD_NAME_PREFIX
            ) as executor:
                pending = set()
                future_reserved_bytes = {}
                offset_iter = iter(offsets)
                max_pending = inflight_limit
                cap = _AdaptiveInflightCap(
                    row_limit=max_pending,
                    max_inflight_bytes_estimate=max_inflight_bytes_estimate,
                )
                max_pending_seen = 0

                def submit_offset(offset):
                    nonlocal max_pending_seen
                    future = executor.submit(fetch_page, offset)
                    pending.add(future)
                    future_reserved_bytes[future] = cap.reserve_on_submit()
                    max_pending_seen = max(max_pending_seen, len(pending))

                try:
                    while cap.can_submit(len(pending)):
                        submit_offset(next(offset_iter))
                except StopIteration:
                    pass

                while pending:
                    future = next(as_completed(pending))
                    pending.remove(future)
                    reserved_bytes = future_reserved_bytes.pop(future, 0.0)
                    try:
                        _, rows = future.result()
                    except Exception:
                        cap.observe_completion(reserved_bytes, None)
                        raise
                    cap.observe_completion(reserved_bytes, rows)
                    for item in rows:
                        yield item
                    try:
                        while cap.can_submit(len(pending)):
                            submit_offset(next(offset_iter))
                    except StopIteration:
                        pass
                if max_inflight_bytes_estimate is not None:
                    _log_parallel_event(
                        logging.DEBUG,
                        "parallel_inflight_cap_stats",
                        job_id=job_id,
                        ordered_mode="unordered",
                        max_in_flight=max_pending_seen,
                        inflight_bytes_budget=max_inflight_bytes_estimate,
                        **cap.stats_fields(),
                    )

        def mostly_ordered_iterator():
            with ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix=DEFAULT_QUERY_THREAD_NAME_PREFIX
            ) as executor:
                pending = set()
                future_reserved_bytes = {}
                offset_iter = iter(offsets)
                max_pending = inflight_limit
                cap = _AdaptiveInflightCap(
                    row_limit=max_pending,
                    max_inflight_bytes_estimate=max_inflight_bytes_estimate,
                )
                max_pending_seen = 0
                buffer = {}
                next_expected = start

                def submit_offset(offset):
                    nonlocal max_pending_seen
                    future = executor.submit(fetch_page, offset)
                    pending.add(future)
                    future_reserved_bytes[future] = cap.reserve_on_submit()
                    max_pending_seen = max(max_pending_seen, len(pending))

                try:
                    while cap.can_submit(len(pending)):
                        submit_offset(next(offset_iter))
                except StopIteration:
                    pass

                while pending:
                    future = next(as_completed(pending))
                    pending.remove(future)
                    reserved_bytes = future_reserved_bytes.pop(future, 0.0)
                    try:
                        offset, rows = future.result()
                    except Exception:
                        cap.observe_completion(reserved_bytes, None)
                        raise
                    cap.observe_completion(reserved_bytes, rows)
                    buffer[offset] = rows

                    # Flush contiguous pages first to preserve strict order when possible.
                    while next_expected in buffer:
                        for item in buffer.pop(next_expected):
                            yield item
                        next_expected += page_size

                    # Avoid HOL stalls by flushing oldest buffered pages after a bounded window.
                    while len(buffer) > ordered_window_pages:
                        oldest_offset = min(buffer)
                        for item in buffer.pop(oldest_offset):
                            yield item
                        if oldest_offset == next_expected:
                            next_expected += page_size

                    try:
                        while cap.can_submit(len(pending)):
                            submit_offset(next(offset_iter))
                    except StopIteration:
                        pass

                # Flush any remaining out-of-order pages deterministically.
                for offset in sorted(buffer):
                    for item in buffer[offset]:
                        yield item
                if max_inflight_bytes_estimate is not None:
                    _log_parallel_event(
                        logging.DEBUG,
                        "parallel_inflight_cap_stats",
                        job_id=job_id,
                        ordered_mode="window",
                        max_in_flight=max_pending_seen,
                        inflight_bytes_budget=max_inflight_bytes_estimate,
                        **cap.stats_fields(),
                    )

        if order_mode == "unordered":
            return unordered_iterator()
        if order_mode in ("window", "mostly_ordered"):
            return mostly_ordered_iterator()
        return ordered_iterator()

    def _resolve_keyset_path(self, keyset_path):
        if keyset_path is None:
            if self.root is None:
                raise QueryError("Cannot infer keyset path when query root is undefined")
            keyset_path = self.root.name + ".id"
        keyset_path = self.prefix_path(str(keyset_path))
        key_path = self.model.make_path(keyset_path, self.get_subclass_dict())
        if not key_path.is_attribute():
            raise QueryError("Keyset path must be an attribute path: %s" % (keyset_path,))
        return keyset_path

    def _iter_keyset_ids(self, keyset_path, keyset_batch_size):
        id_query = self.clone()
        id_query.clear_view()
        id_query.add_view(keyset_path)
        # Join directives are only needed for output shape; they can invalidate
        # reduced cursor probes on some mines when only root-id is selected.
        id_query.joins = []
        id_query._sort_order_list = SortOrderList()
        id_query.add_sort_order(keyset_path, SortOrder.ASC)

        last_seen = None
        cursor_constraint = None
        while True:
            if last_seen is not None:
                if cursor_constraint is None:
                    cursor_constraint = constraints.BinaryConstraint(
                        keyset_path,
                        ">",
                        str(last_seen),
                        code="A",
                    )
                    cursor_constraint.path = id_query.prefix_path(cursor_constraint.path)
                    if id_query.do_verification:
                        id_query.verify_constraint_paths([cursor_constraint])
                    id_query.uncoded_constraints.append(cursor_constraint)
                else:
                    cursor_constraint.value = str(last_seen)
            ids = []
            for rec in islice(id_query.results(row="list", start=0, size=keyset_batch_size), keyset_batch_size):
                value = rec[0] if isinstance(rec, (list, tuple)) else rec
                if value is None:
                    continue
                token = str(value)
                if not ids or token != ids[-1]:
                    ids.append(token)
            if not ids:
                break
            if last_seen is not None and ids[-1] == last_seen:
                break
            last_seen = ids[-1]
            yield ids

    def _run_parallel_keyset(
        self,
        row="dict",
        size=None,
        page_size=DEFAULT_PARALLEL_PAGE_SIZE,
        max_workers=DEFAULT_PARALLEL_WORKERS,
        ordered=True,
        prefetch=None,
        keyset_path=None,
        keyset_batch_size=DEFAULT_KEYSET_BATCH_SIZE,
    ):
        keyset_path = self._resolve_keyset_path(keyset_path)
        yielded = 0
        chunk_query = self.clone()
        cursor_constraint = constraints.MultiConstraint(
            keyset_path,
            "ONE OF",
            [],
            code="A",
        )
        cursor_constraint.path = chunk_query.prefix_path(cursor_constraint.path)
        if chunk_query.do_verification:
            chunk_query.verify_constraint_paths([cursor_constraint])
        chunk_query.uncoded_constraints.append(cursor_constraint)
        chunk_query._sort_order_list = SortOrderList()
        chunk_query.add_sort_order(keyset_path, SortOrder.ASC)

        for ids in self._iter_keyset_ids(keyset_path, keyset_batch_size):
            if size is not None and yielded >= size:
                break
            cursor_constraint.values = ids
            # Stream each id-window directly to avoid per-window counts and
            # avoid introducing deep offset pagination into large scans.
            chunk_iter = chunk_query.results(row=row, start=0, size=None)
            for item in chunk_iter:
                yield item
                yielded += 1
                if size is not None and yielded >= size:
                    return

    def _resolve_parallel_strategy(self, pagination, start, size):
        return resolve_parallel_strategy(
            pagination,
            start,
            size,
            valid_parallel_pagination=VALID_PARALLEL_PAGINATION,
            keyset_auto_min_size=KEYSET_AUTO_MIN_SIZE,
        )

    def _normalize_order_mode(self, ordered):
        return normalize_order_mode(
            ordered,
            default_order_mode=DEFAULT_PARALLEL_ORDERED_MODE,
            valid_order_modes=VALID_ORDER_MODES,
        )

    def _apply_parallel_profile(self, profile, ordered, large_query_mode):
        return apply_parallel_profile(
            profile,
            ordered,
            large_query_mode,
            default_profile=DEFAULT_PARALLEL_PROFILE,
            valid_parallel_profiles=VALID_PARALLEL_PROFILES,
        )

    def _resolve_effective_workers(self, max_workers, size):
        if max_workers is not None:
            return max_workers
        service_root = getattr(getattr(self, "service", None), "root", None)
        return resolve_preferred_workers(service_root, size, DEFAULT_PARALLEL_WORKERS)

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
            return bool(is_tor_proxy_url(proxy_url)), True, "service.proxy_url"
        except Exception:
            return False, False, "unknown"

    def _coerce_parallel_options(
        self,
        *,
        parallel_options=None,
        page_size=DEFAULT_PARALLEL_PAGE_SIZE,
        max_workers=None,
        ordered=None,
        prefetch=None,
        inflight_limit=None,
        ordered_max_in_flight=None,
        ordered_window_pages=DEFAULT_ORDER_WINDOW_PAGES,
        profile=DEFAULT_PARALLEL_PROFILE,
        large_query_mode=DEFAULT_LARGE_QUERY_MODE,
        pagination=DEFAULT_PARALLEL_PAGINATION,
        keyset_path=None,
        keyset_batch_size=DEFAULT_KEYSET_BATCH_SIZE,
    ):
        overrides = _legacy_parallel_overrides(
            page_size=page_size,
            max_workers=max_workers,
            ordered=ordered,
            prefetch=prefetch,
            inflight_limit=inflight_limit,
            ordered_max_in_flight=ordered_max_in_flight,
            ordered_window_pages=ordered_window_pages,
            profile=profile,
            large_query_mode=large_query_mode,
            pagination=pagination,
            keyset_path=keyset_path,
            keyset_batch_size=keyset_batch_size,
        )
        if parallel_options is None:
            _warn_legacy_parallel_args(overrides, ignored=False)
            return ParallelOptions(
                page_size=page_size,
                max_workers=max_workers,
                ordered=ordered,
                prefetch=prefetch,
                inflight_limit=inflight_limit,
                ordered_max_in_flight=ordered_max_in_flight,
                ordered_window_pages=ordered_window_pages,
                profile=profile,
                large_query_mode=large_query_mode,
                pagination=pagination,
                keyset_path=keyset_path,
                keyset_batch_size=keyset_batch_size,
            )
        if not isinstance(parallel_options, ParallelOptions):
            raise ParallelOptionsError(
                "parallel_options must be a ParallelOptions instance. "
                "Construct options with ParallelOptions(...)."
            )
        _warn_legacy_parallel_args(overrides, ignored=True)
        return parallel_options

    def _resolve_parallel_options(self, *, start, size, options: ParallelOptions) -> ResolvedParallelOptions:
        try:
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
                default_parallel_prefetch=DEFAULT_PARALLEL_PREFETCH,
            )
            tor_prefetch_adjusted = False
            if tor_enabled and prefetch_from_default:
                adjusted_prefetch = max(1, min(prefetch, max_workers))
                tor_prefetch_adjusted = adjusted_prefetch != prefetch
                prefetch = adjusted_prefetch
            inflight_limit = resolve_inflight_limit(
                options.inflight_limit,
                prefetch=prefetch,
                default_parallel_inflight_limit=DEFAULT_PARALLEL_INFLIGHT_LIMIT,
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
            inflight_limit = _cap_inflight_limit(inflight_limit, page_size)
            ordered_max_in_flight = options.ordered_max_in_flight
            if ordered_max_in_flight is None:
                ordered_max_in_flight = max_workers * 2
            ordered_max_in_flight = require_positive_int("ordered_max_in_flight", ordered_max_in_flight)
            ordered_max_in_flight = min(ordered_max_in_flight, inflight_limit)
            ordered_window_pages = require_positive_int("ordered_window_pages", options.ordered_window_pages)
            start_value = require_non_negative_int("start", start_value)
            size_value = size
            if size_value is not None:
                size_value = require_non_negative_int("size", size_value)
            keyset_batch_size = require_positive_int("keyset_batch_size", options.keyset_batch_size)
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
                ordered_max_in_flight=ordered_max_in_flight,
                ordered_window_pages=ordered_window_pages,
                profile=profile,
                large_query_mode=large_query_mode,
                pagination=options.pagination,
                keyset_path=options.keyset_path,
                keyset_batch_size=keyset_batch_size,
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
        page_size=DEFAULT_PARALLEL_PAGE_SIZE,
        max_workers=None,
        ordered=None,
        prefetch=None,
        inflight_limit=None,
        ordered_max_in_flight=None,
        ordered_window_pages=DEFAULT_ORDER_WINDOW_PAGES,
        profile=DEFAULT_PARALLEL_PROFILE,
        large_query_mode=DEFAULT_LARGE_QUERY_MODE,
        pagination=DEFAULT_PARALLEL_PAGINATION,
        keyset_path=None,
        keyset_batch_size=DEFAULT_KEYSET_BATCH_SIZE,
        job_id=None,
        parallel_options=None,
    ):
        """
        Fetch paged results concurrently and yield rows.

        Usage::
          >>> for row in query.run_parallel(page_size=2000, max_workers=16, pagination="auto"):
          ...     process(row)

        @param max_workers: Optional worker count override. If omitted, a
                            mine-specific default is resolved (see
                            ``intermine314/config/mine-parallel-preferences.toml``), with
                            fallback to ``16``.
        @type max_workers: int | None
        @param ordered: Ordering mode: ``True``/``False`` or one of
                        ``ordered``, ``unordered``, ``window``, ``mostly_ordered``.
                        ``window`` and ``mostly_ordered`` preserve near-ordering in
                        a bounded window to reduce HOL stalls.
        @type ordered: bool | str
        @param prefetch: Read-ahead page budget. Defaults to ``max_workers``
                         and to ``2 * max_workers`` when ``large_query_mode=True``.
        @type prefetch: int
        @param inflight_limit: Maximum in-flight page tasks. This is passed to
                               Python 3.14 ``executor.map(..., buffersize=...)`` in
                               ordered mode and used as the pending cap otherwise.
                               Defaults to ``prefetch``.
        @type inflight_limit: int
        @param ordered_max_in_flight: Ordered mode task window cap. Defaults to
                                      ``2 * max_workers`` and is additionally capped
                                      by ``inflight_limit``.
        @type ordered_max_in_flight: int | None
        @param ordered_window_pages: Maximum buffered page windows before flushing
                                     out-of-order pages in ``window`` mode.
        @type ordered_window_pages: int
        @param profile: One of ``default``, ``large_query``, ``unordered``,
                        ``mostly_ordered``.
        @type profile: str
        @param large_query_mode: Enable large-query defaults (prefetch=2*workers).
        @type large_query_mode: bool
        @param pagination: One of ``auto``, ``offset``, ``keyset``.
                           ``auto`` selects keyset for large scans at start=0.
        @type pagination: str
        @param keyset_path: Optional path used as keyset cursor (defaults to ``<root>.id``)
        @type keyset_path: str
        @param keyset_batch_size: Number of cursor ids per keyset chunk.
        @type keyset_batch_size: int
        @param job_id: Optional correlation id for structured parallel export logs.
        @type job_id: str | None
        @param parallel_options: Canonical parallel tuning value object.
                                 Legacy individual parallel keyword arguments
                                 are still accepted for compatibility during
                                 a deprecation window.
        @type parallel_options: ParallelOptions | None
        """
        options = self._coerce_parallel_options(
            parallel_options=parallel_options,
            page_size=page_size,
            max_workers=max_workers,
            ordered=ordered,
            prefetch=prefetch,
            inflight_limit=inflight_limit,
            ordered_max_in_flight=ordered_max_in_flight,
            ordered_window_pages=ordered_window_pages,
            profile=profile,
            large_query_mode=large_query_mode,
            pagination=pagination,
            keyset_path=keyset_path,
            keyset_batch_size=keyset_batch_size,
        )
        resolved = self._resolve_parallel_options(start=start, size=size, options=options)
        if resolved.size == 0:
            return iter(())
        run_job_id = str(job_id).strip() if job_id is not None else ""
        if not run_job_id:
            run_job_id = new_job_id("qp")

        if resolved.strategy == "keyset":
            try:
                iterator = self._run_parallel_keyset(
                    row=row,
                    size=resolved.size,
                    page_size=resolved.page_size,
                    max_workers=resolved.max_workers,
                    ordered=resolved.order_mode,
                    prefetch=resolved.prefetch,
                    keyset_path=resolved.keyset_path,
                    keyset_batch_size=resolved.keyset_batch_size,
                )
            except QueryError as exc:
                raise _parallel_options_error(exc) from exc
        else:
            iterator = self._run_parallel_offset(
                row=row,
                start=resolved.start,
                size=resolved.size,
                page_size=resolved.page_size,
                max_workers=resolved.max_workers,
                order_mode=resolved.order_mode,
                inflight_limit=resolved.inflight_limit,
                ordered_window_pages=resolved.ordered_window_pages,
                ordered_max_in_flight=resolved.ordered_max_in_flight,
                max_inflight_bytes_estimate=resolved.max_inflight_bytes_estimate,
                job_id=run_job_id,
            )
        return _instrument_parallel_iterator(
            iterator,
            job_id=run_job_id,
            strategy=resolved.strategy,
            order_mode=resolved.order_mode,
            start=resolved.start,
            size=resolved.size,
            page_size=resolved.page_size,
            max_workers=resolved.max_workers,
            prefetch=resolved.prefetch,
            inflight_limit=resolved.inflight_limit,
            max_in_flight=(
                resolved.ordered_max_in_flight
                if resolved.order_mode == "ordered"
                else resolved.inflight_limit
            ),
            ordered_window_pages=resolved.ordered_window_pages,
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
        if p.end.type_name in Model.NUMERIC_TYPES:
            return dict((k, float(v)) for k, v in list(next(results).items()))
        else:
            return dict((r["item"], r["count"]) for r in results)

    def one(self, row="jsonobjects"):
        """Return one result, and raise an error if the result size is not 1"""
        if row == "jsonobjects":
            if self.count() == 1:
                return self.first(row)
            else:
                ret = None
                for obj in self.results():
                    if ret is not None:
                        raise QueryError("More than one result received")
                    else:
                        ret = obj
                if ret is None:
                    raise QueryError("No results received")

                return ret
        else:
            c = self.count()
            if c != 1:
                raise QueryError("Result size is not one: got %d results" % (c))
            else:
                return self.first(row)

    def first(self, row="jsonobjects", start=0, **kw):
        """Return the first result, or None if the results are empty"""
        if row == "jsonobjects":
            size = None
        else:
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

        Also available as Query.all

        @see: L{intermine314.query.Query.results}

        """
        return list(self.results(*args, **kwargs))

    def get_row_list(self, start=0, size=None):
        return self.get_results_list("rr", start, size)

    def count(self):
        """
        Return the total number of rows this query returns
        ==================================================

        Obtain the number of rows a particular query will
        return, without having to fetch and parse all the
        actual data. This method makes a request to the server
        to report the count for the query, and is sugar for a
        results call.

        Also available as Query.size

        @rtype: int
        @raise WebserviceError: if the request is unsuccessful.
        """
        count_str = ""
        for row in self.results(row="count"):
            count_str += row
        try:
            return int(count_str)
        except ValueError:
            raise ResultError("Server returned a non-integer count: " + count_str)

    def get_list_upload_uri(self):
        """
        Returns the uri to use to create a list from this query
        =======================================================

        Query.get_list_upload_uri() -> str

        This method is used internally when performing list operations
        on queries.

        @rtype: str
        """
        return self.service.root + self.service.QUERY_LIST_UPLOAD_PATH

    def get_list_append_uri(self):
        """
        Returns the uri to use to create a list from this query
        =======================================================

        Query.get_list_append_uri() -> str

        This method is used internally when performing list operations
        on queries.

        @rtype: str
        """
        return self.service.root + self.service.QUERY_LIST_APPEND_PATH

    def get_results_path(self):
        """
        Returns the path section pointing to the REST resource
        ======================================================

        Query.get_results_path() -> str

        Internally, this just calls a constant property
        in intermine314.service.Service

        @rtype: str
        """
        return self.service.QUERY_PATH

    def children(self):
        """
        Returns the child objects of the query
        ======================================

        This method is used during the serialisation of queries
        to xml. It is unlikely you will need access to this as a whole.
        Consider using "path_descriptions", "joins", "constraints" instead

        @see: Query.path_descriptions
        @see: Query.joins
        @see: Query.constraints

        @return: the child element of this query
        @rtype: list
        """
        return [*self.path_descriptions, *self.joins, *self.constraints]

    def to_query(self):
        """
        Implementation of trait that allows use of these objects as queries
        (casting).
        """
        return self

    def make_list_constraint(self, path, op):
        """
        Implementation of trait that allows use of these objects in list
        constraints
        """
        l = self.service.create_list(self)
        return ConstraintNode(path, op, l.name)

    def to_query_params(self):
        """
        Returns the parameters to be passed to the webservice
        =====================================================

        The query is responsible for producing its own query
        parameters. These consist simply of:
         - query: the xml representation of the query

        @rtype: dict

        """
        xml = self.to_xml()
        params = {"query": xml}
        return params

    def to_Node(self):
        """
        Returns a DOM node representing the query
        =========================================

        This is an intermediate step in the creation of the
        xml serialised version of the query. You probably
        won't need to call this directly.

        @rtype: xml.minidom.Node
        """
        impl = getDOMImplementation()
        doc = impl.createDocument(None, "query", None)
        query = doc.documentElement

        query.setAttribute("name", self.name)
        query.setAttribute("model", self.model.name)
        query.setAttribute("view", " ".join(self.views))
        query.setAttribute("sortOrder", str(self.get_sort_order()))
        query.setAttribute("longDescription", self.description)
        if len(self.coded_constraints) > 1:
            query.setAttribute("constraintLogic", str(self.get_logic()))

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
        newobj = self.__class__(self.model)
        for attr in [
            "joins",
            "views",
            "_sort_order_list",
            "_logic",
            "path_descriptions",
            "constraint_dict",
            "uncoded_constraints",
        ]:
            setattr(newobj, attr, deepcopy(getattr(self, attr)))

        for attr in ["name", "description", "service", "do_verification", "constraint_factory", "root"]:
            setattr(newobj, attr, getattr(self, attr))
        return newobj


class Template(Query):
    """
    A Class representing a predefined query
    =======================================

    Templates are ways of saving queries
    and allowing others to run them
    simply. They are the main interface
    to querying in the webapp

    SYNOPSIS
    --------

    example::

      service = Service("https://www.flymine.org/query/service")
      template = service.get_template("Gene_Pathways")
      for row in template.results(A={"value":"eve"}):
        process_row(row)
        ...

    A template is a subclass of query that comes predefined. They
    are typically retrieved from the webservice and run by specifying
    the values for their existing constraints. They are a concise
    and powerful way of running queries in the webapp.

    Being subclasses of query, everything is true of them that is true
    of a query. They are just less work, as you don't have to design each
    one. Also, you can store your own templates in the web-app, and then
    access them as a private webservice method, from anywhere, making them
    a kind of query in the cloud - for this you will need to authenticate
    by providing log in details to the service.

    The most significant difference is how constraint values are specified
    for each set of results.

    @see: L{Template.results}

    """

    def __init__(self, *args, **kwargs):
        """
        Constructor
        ===========

        Instantiation is identical that of queries. As with queries,
        these are best obtained from the intermine314.webservice.Service
        factory methods.

        @see: L{intermine314.webservice.Service.get_template}
        """
        super(Template, self).__init__(*args, **kwargs)
        self.constraint_factory = constraints.TemplateConstraintFactory()
        self.title = ""
        self.user_name = ""
        self.view_types = []

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
        newobj = super(Template, self).clone()
        setattr(newobj, "user_name", getattr(self, "user_name"))
        return newobj

    def add_user_name(self, user_name):
        self.user_name = user_name

    @classmethod
    def from_xml(cls, xml, *args, **kwargs):
        """
        Deserialise a template query serialised to XML
        ==============================================

        This method is used to instantiate serialised templates.
        It is used by intermine314.webservice.Service objects
        to instantiate Template objects and it can be used
        to read in templates you have saved to a file.

        @param xml: The xml as a file name, url, or string

        @raise QueryParseError: if the query cannot be parsed

        @rtype: L{Template}
        """
        # Extract all Query (superclass) fields
        obj = super(Template, cls).from_xml(xml, *args, **kwargs)

        # Extract fields specific to Template, like title
        obj.do_verification = False
        with closing(openAnything(xml)) as f:
            doc = minidom.parse(f)

        templates = doc.getElementsByTagName("template")
        if len(templates) != 1:
            raise QueryParseError(
                "wrong number of templates in xml. "
                + "Only one <template> element is allowed. "
                + "Found %d" % len(templates)
            )
        t = templates[0]
        obj.title = t.getAttribute("title")
        for data_type in t.getAttribute("dataTypes").split(" "):
            obj.view_types.append(data_type)

        obj.verify()

        return obj

    @property
    def editable_constraints(self):
        """
        Return the list of constraints you can edit
        ===========================================

        Template.editable_constraints -> list(intermine314.constraints.Constraint)

        Templates have a concept of editable constraints, which
        is a way of hiding complexity from users. An underlying query may have
        five constraints, but only expose the one that is actually
        interesting. This property returns this subset of constraints
        that have the editable flag set to true.
        """
        return [c for c in self.constraints if c.editable]

    def to_query_params(self):
        """
        Returns the query parameters needed for the webservice
        ======================================================

        Template.to_query_params() -> dict(string, string)

        Overrides the method of the same name in query to provide the
        parameters needed by the templates results service. These
        are slightly more complex:
            - name: The template's name
            - for each constraint: (where [i] is an integer incremented for each constraint)
                - constraint[i]: the path
                - op[i]:         the operator
                - value[i]:      the value
                - code[i]:       the code
                - extra[i]:      the extra value for ternary constraints (optional)


        @rtype: dict
        """
        p = {"name": self.name, "userName": self.user_name}
        i = 1
        for c in self.editable_constraints:
            if not c.switched_on:
                next
            for k, v in list(c.to_dict().items()):
                if k == "extraValue":
                    k = "extra"
                if k == "path":
                    k = "constraint"
                p[k + str(i)] = v
            i += 1
        return p

    def get_results_path(self):
        """
        Returns the path section pointing to the REST resource
        ======================================================

        Template.get_results_path() S{->} str

        Internally, this just calls a constant property
        in intermine314.service.Service

        This overrides the method of the same name in Query

        @return: the path to the REST resource
        @rtype: string
        """
        return self.service.TEMPLATEQUERY_PATH

    def get_adjusted_template(self, con_values):
        """
        Gets a template to run
        ======================

        Template.get_adjusted_template(con_values) S{->} Template

        When templates are run, they are first cloned, and their
        values are changed to those desired. This leaves the original
        template unchanged so it can be run again with different
        values. This method does the cloning and changing of constraint
        values

        @raise ConstraintError: if the constraint values specify values for a
                                non-editable constraint.

        @rtype: L{Template}
        """
        clone = self.clone()

        for code, options in list(con_values.items()):
            con = clone.get_constraint(code)
            if not con.editable:
                raise ConstraintError("There is a constraint '" + code + "' on this query, but it is not editable")
            try:
                for key, value in list(options.items()):
                    setattr(con, key, value)
            except AttributeError:
                setattr(con, "value", options)
        return clone

    def results(self, row="object", start=0, size=None, **con_values):
        """
        Get an iterator over result rows
        ================================

        This method returns the same values with the
        same options as the method of the same name in
        Query (see intermine314.query.Query). The main difference in in the
        arguments.

        The template result methods also accept a key-word pair
        set of arguments that are used to supply values
        to the editable constraints. eg::

          template.results(
            A = {"value": "eve"},
            B = {"op": ">", "value": 5000}
          )

        The keys should be codes for editable constraints (you can inspect
        these with Template.editable_constraints) and the values should be a
        dictionary of constraint properties to replace. You can replace the
        values for "op" (operator), "value", and "extra_value" and "values"
        in the case of ternary and multi constraints.

        @rtype: L{intermine314.webservice.ResultIterator}
        """
        clone = self.get_adjusted_template(con_values)
        return super(Template, clone).results(row, start, size)

    def get_results_list(self, row="object", start=0, size=None, **con_values):
        """
        Get a list of result rows
        =========================

        This method performs the same as the method of the
        same name in Query, and it shares the semantics of
        Template.results().

        @see: L{intermine314.query.Query.get_results_list}
        @see: L{intermine314.query.Template.results}

        @rtype: list

        """
        clone = self.get_adjusted_template(con_values)
        return super(Template, clone).get_results_list(row, start, size)

    def get_row_list(self, start=0, size=None, **con_values):
        """Return a list of the rows returned by this query"""
        clone = self.get_adjusted_template(con_values)
        return super(Template, clone).get_row_list(start, size)

    def rows(self, start=0, size=None, **con_values):
        """Get an iterator over the rows returned by this query"""
        clone = self.get_adjusted_template(con_values)
        return super(Template, clone).rows(start, size)

    def count(self, **con_values):
        """
        Return the total number of rows this template returns
        =====================================================

        Obtain the number of rows a particular query will
        return, without having to fetch and parse all the
        actual data. This method makes a request to the server
        to report the count for the query, and is sugar for a
        results call.

        @rtype: int
        @raise WebserviceError: if the request is unsuccessful.
        """
        clone = self.get_adjusted_template(con_values)
        return super(Template, clone).count()


class QueryError(ReadableException):
    pass


class ConstraintError(QueryError):
    pass


class QueryParseError(QueryError):
    pass


class ResultError(ReadableException):
    pass
