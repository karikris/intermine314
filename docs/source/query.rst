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

.. code-block:: python

   for row in query.run_parallel(
       row="dict",
       page_size=2000,
       max_workers=4,
       prefetch=4,
       ordered=True,
   ):
       process(row)

DataFrame workflow with Polars
------------------------------

.. code-block:: python

   df = query.dataframe(
       batch_size=5000,
       parallel=True,
       page_size=2000,
       max_workers=4,
       prefetch=4,
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
       max_workers=4,
   )

Single Parquet file:

.. code-block:: python

   query.to_parquet(
       "results.parquet",
       single_file=True,
       parallel=True,
       page_size=2000,
       max_workers=4,
   )

DuckDB SQL over query output
----------------------------

.. code-block:: python

   con = query.to_duckdb(
       "results_parquet",
       table="results",
       parallel=True,
       page_size=2000,
       max_workers=4,
   )
   rows = con.execute("select count(*) from results").fetchall()
   print(rows)

Reference
---------

.. automodule:: intermine314.query
   :members:
   :undoc-members:
   :show-inheritance:
