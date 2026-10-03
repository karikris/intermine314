Query and Analytics Workflow
============================

Python 3.14 Baseline
--------------------

This package line targets Python 3.14 and uses modern concurrency and I/O behavior.

Install
-------

::

   pip install intermine314

Polars and DuckDB are core dependencies used by:

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

Parallel result retrieval
-------------------------

``Query.run_parallel`` fetches pages concurrently using a single offset scheduler.

.. code-block:: python

   parallel_options = {
       "pagination": "auto",
       "profile": "large_query",
       "ordered": "unordered",
       "inflight_limit": 8,  # caps in-flight buffersize to keep RAM bounded
   }
   for row in query.run_parallel(row="dict", **parallel_options):
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
