Query and Analytics Workflow
============================

Python 3.14 Baseline
--------------------

This package line targets Python 3.14 and assumes modern typing,
concurrency, and I/O behavior from CPython 3.14.

Install optional analytics dependencies
---------------------------------------

::

   pip install "intermine314[dataframe]"

The ``dataframe`` extra installs Polars and DuckDB support used by:

- ``Query.dataframe()``
- ``Query.to_parquet()``
- ``Query.to_duckdb()``

No pandas dependency is used in the ``intermine314`` runtime package.

Basic query execution
---------------------

.. code-block:: python

   from intermine314.webservice import Service

   service = Service("https://www.flymine.org/query/service")
   query = service.new_query("Gene")
   query.add_view("Gene.symbol", "Gene.length")

   for row in query.results(row="dict", start=0, size=100):
       print(row)

Parallel result retrieval
-------------------------

``Query.run_parallel`` fetches paged results concurrently.
The default pagination strategy is ``pagination="auto"``.
For max throughput benchmarking, prefer ``ordered=False``.

.. code-block:: python

   for row in query.run_parallel(
       row="dict",
       page_size=2000,
       max_workers=16,
       profile="large_query",
       ordered="window",
       ordered_window_pages=10,
       prefetch=32,
       inflight_limit=24,
       pagination="auto",
   ):
       process(row)

Profiles:

- ``profile="large_query"`` enables large-query defaults (prefetch ``2 * workers``).
- ``profile="unordered"`` favors throughput.
- ``profile="mostly_ordered"`` applies windowed ordering defaults.

DataFrame workflow with Polars
------------------------------

.. code-block:: python

   df = query.dataframe(
       batch_size=5000,
       parallel=True,
       page_size=2000,
       max_workers=16,
       profile="large_query",
       ordered="window",
       ordered_window_pages=10,
       prefetch=32,
       inflight_limit=24,
       pagination="auto",
   )
   print(df.shape)

Parquet export
--------------

Directory of part files:

.. code-block:: python

   query.to_parquet(
       "results_parquet",
       batch_size=10000,
       parallel=True,
       page_size=2000,
       max_workers=16,
       pagination="auto",
   )

Single Parquet file:

.. code-block:: python

   query.to_parquet(
       "results.parquet",
       single_file=True,
       parallel=True,
       page_size=2000,
       max_workers=16,
       pagination="auto",
   )

DuckDB SQL over query output
----------------------------

.. code-block:: python

   con = query.to_duckdb(
       "results_parquet",
       table="results",
       parallel=True,
       page_size=2000,
       max_workers=16,
       pagination="auto",
   )
   rows = con.execute("select count(*) from results").fetchall()
   print(rows)

Reference
---------

.. automodule:: intermine314.query
   :members:
   :undoc-members:
   :show-inheritance:
