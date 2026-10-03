Query and Analytics Workflow
============================

Python 3.14.5 Baseline
----------------------

This package line requires Python 3.14.5 or newer and uses modern concurrency and I/O behavior.

Install
-------

::

   pip install intermine314
   pip install "intermine314[plots]"  # optional lazy Matplotlib

Polars, DuckDB and PyArrow are lazy core dependencies used by:

- ``Query.to_parquet()``
- ``Query.to_duckdb()``

No pandas dependency is required in the ``intermine314`` runtime package.

Service endpoint rule
---------------------

Use the InterMine service root with no trailing slash:

- Good: ``https://.../MineName/service``
- Avoid: ``https://.../MineName/service/``

Basic query execution
---------------------

.. code-block:: python

   from intermine314.service import Service

   service = Service("https://maizemine.rnet.missouri.edu/maizemine/service")
   query = service.select("Gene.primaryIdentifier", "Gene.symbol", "Gene.length")

   for row in query.results(row="dict", start=0, size=1000):
       handle_row(row)

Constraint logic
----------------

By default, all coded constraints are joined with AND. ``query.set_logic(...)``
and the ``query.logic`` property accept strings or constraint objects combined
with ``+`` and ``&`` (AND), or ``|`` (OR). ``set_logic`` returns the query.

Both native and legacy profiles preserve the original client's tested string
parsing: OR binds before AND, so ``B and C or A and D`` becomes
``B and (C or A) and D``. Closing a parenthesized group also completes its
preceding operation: ``A and (B) or C`` becomes ``(A and B) or C``.
Object expressions follow Python's operator precedence. Use explicit grouping
when choosing a combination of filters. Codes such as ``AA`` and the operator
aliases ``&``, ``&&``, ``|`` and ``||`` are supported, including compact syntax.

Syntax and grouping errors raise ``LogicParseError``. Empty logic raises
``EmptyLogicError`` when there are coded constraints. Validation requires every
coded constraint and rejects unknown codes. After adding a constraint to an
explicit expression, update the expression and call ``validate_logic()``.
XML includes ``constraintLogic`` when there are multiple coded constraints;
clones have independent constraint and logic trees.

Nested grouping preserves outer opening markers instead of discarding them
when an inner group closes. This intentionally repairs some expressions that
the original client successfully parsed with different meaning: it parsed
``B OR ((A OR C) AND D)`` as ``(B or A or C) and D``; this client retains
``B or ((A or C) and D)``. With B true and D false, the repaired expression
is true while the original expression was false. Historical OR-before-AND
precedence and completion of the preceding operation at a closing group remain.
Malformed syntax raises parser errors instead of internal stack errors.

``SortOrderList.next()`` retains the existing native
repair that returns its first element without consuming it; the original
client attempted ``next()`` on a Python list and raised ``TypeError``.

Legacy Column expressions
-------------------------

With ``intermine314.webservice.Service`` (or compatibility="legacy"),
``query.column(path)`` and ``query.c(path)`` return navigable model Columns:

.. code-block:: python

   from intermine314.webservice import Service

   service = Service("https://example.org/mine/service")
   query = service.select("Employee.name", "Employee.age")
   employee = query.c("Employee")
   filtered = query.filter((employee.name == "Alice") | (employee.age >= 30))
   filtered = filtered.where(employee.age != None)  # IS NOT NULL

``filter`` aliases ``where`` and returns an independent query. New groups are
ANDed with existing logic. Flat/all-AND filters preserve dynamic default logic,
so later ``add_constraint`` calls are included; OR groups retain explicit trees.
Keyword-only ``where`` calls always mean field=value,
including fields named ``path``, ``op``, or ``subclass``. Constructor keywords
can be supplied through an explicit ``ConstraintNode`` or ``CodelessNode``.
None comparisons produce null checks; lists produce
ONE OF/NONE OF; object Column comparisons produce compatible loop constraints.
Subclass expressions such as ``employee < service.model.Manager`` apply
unconditionally and do not enter Boolean logic. Native ``column`` and ``c``
continue returning string paths; native queries may also accept expression
nodes constructed from ``service.model``.

``service.model.Employee.select("name")`` builds a query. Class/reference
Column selection without arguments expands attributes, while an attribute
Column selects its own path. Relative fields on a reference are selected beneath
that reference. Selection from a query-bound Column clones its filters and
subclass refinements. Attribute iteration returns scalar values from current
dictionary or indexed rows. Relation/class queries still return dictionary
rows in native mode and model objects in legacy mode.

Explicit native IN with a collection remains ONE OF, whereas legacy IN means
a named server list. ``where_in`` and collection-valued field=value keywords
emit ONE OF in both profiles. Legacy ``add_constraint(age="IS NULL")`` recognizes
the unary keyword overload; ``where(age="IS NULL")`` remains string equality.

See ``docs/analysis/model-compatibility.md`` for scoped upstream differences and
executed offline evidence.

Parallel result retrieval
-------------------------

``Query.run_parallel`` fetches pages concurrently using a single offset scheduler.

.. code-block:: python

   from intermine314.query.builder import ParallelOptions

   parallel_options = ParallelOptions(
       pagination="auto", profile="large_query", ordered="unordered",
       inflight_limit=8,
   )
   for row in query.run_parallel(row="dict", parallel_options=parallel_options):
       handle_row(row)

Available runtime profiles:

- ``profile="default"``
- ``profile="large_query"``
- ``profile="unordered"``

Runtime configuration files:

- ``intermine314.config/runtime-defaults.toml``

This runtime file is loaded from package resources, so behavior is consistent between
``pip`` installations and editable/source checkouts.

Benchmark policy and target presets remain benchmark-suite config under ``benchmarks/profiles/``,
including ``mine-parallel-preferences.toml`` and ``benchmark-targets.toml``.
These are not part of runtime query defaults.

You can override runtime defaults with:
``INTERMINE314_RUNTIME_DEFAULTS_PATH=/abs/path/to/runtime-defaults.toml``.

Low-memory patterns
-------------------

For large exports, avoid materializing all rows as Python objects:

- Stream rows from ``query.results()`` or ``query.run_parallel()`` instead of ``list(...)``.
- Prefer ``Query.to_parquet()`` for persistence over in-memory DataFrame growth.
- Use ``inflight_limit`` and moderate ``page_size`` to bound memory under high concurrency.

Polars + Parquet workflow
-------------------------

.. code-block:: python

   from intermine314.query.builder import ParallelOptions

   query.to_parquet(
       "results_parquet",
       batch_size=10000,
       parallel_options=ParallelOptions(
           pagination="auto",
           profile="large_query",
           ordered="unordered",
           inflight_limit=8,
       ),
   )

DuckDB SQL over Parquet output
------------------------------

.. code-block:: python

   from intermine314.query.builder import ParallelOptions

   with query.to_duckdb(
       "results_parquet",
       table="results",
       parallel_options=ParallelOptions(profile="large_query"),
       managed=True,
   ) as con:
       print(con.execute("select count(*) from results").fetchone())

Profiles and dataframe results
------------------------------

``intermine314.service.Service`` defaults to native dictionary results.
``intermine314.webservice.Service`` defaults to legacy model objects for
``results()`` and iteration, and indexed ResultRow values for ``rows()``.
Both accept ``compatibility="native"`` or ``compatibility="legacy"``.
Direct ``Query(Model(...))`` infers legacy mode. Clones preserve the profile.

``dataframe(start=0, size=None)`` materializes a Polars DataFrame in both profiles,
an intentional departure from the original Pandas API. Analytical iteration
always selects dictionary rows. Analytics dependencies load only when called;
``intermine314[analytics]`` remains an empty compatibility extra.

Explicit CSV and default Parquet
--------------------------------

``query.export("genes.parquet")`` writes one Parquet file by default.
``query.export("genes.csv", format="csv")`` explicitly requests CSV output;
a suffix alone never selects CSV. ``to_parquet`` retains its partition-directory
default. Empty output keeps Model-derived types and selected column names.

``import_csv(csv_input, parquet_path, csv_options=...)`` accepts a path or borrowed
stream, uses Polars scan/sink and leaves borrowed streams open. ``query_parquet``
queries a Parquet file/directory through DuckDB SQL and Arrow and returns Polars.
Both manage their own temporary resources and connections.

``dataframe``, ``to_parquet`` and ``to_duckdb`` also accept ``csv_input`` and
``csv_options`` when the query has no selected views, constraints, joins or sort
order. ``fetch_from_mine(csv_input=..., parquet_path=...)`` needs no remote Service;
combining local CSV and remote arguments is rejected. List creation and append
accept CSV with an explicit ``csv_column`` and preserve identifier strings.
CSV output always requires an explicit format request.

Optional plotting
-----------------

``intermine314.bar_chart`` restores original-name helpers using Polars expressions.
Matplotlib loads only when a plot is requested; install ``intermine314[plots]``.
Headless callers can select the Agg backend. Network reads use the shared managed
transport. No Pandas or original InterMine runtime dependency is required.
