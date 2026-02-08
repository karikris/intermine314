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

- ``Query.dataframe()``
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

   from intermine314.webservice import Service

   service = Service("https://maizemine.rnet.missouri.edu/maizemine/service")
   query = service.new_query("Gene")
   query.add_view("Gene.primaryIdentifier", "Gene.symbol", "Gene.length")

   for row in query.results(row="dict", start=0, size=1000):
       handle_row(row)

Parallel result retrieval
-------------------------

``Query.run_parallel`` fetches pages concurrently with adaptive pagination by default.

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
- ``profile="large_query"`` (prefetch multiplier defaults to ``2 * workers``)
- ``profile="unordered"``
- ``profile="mostly_ordered"`` (windowed ordering)

Runtime configuration files:

- ``config/runtime-defaults.toml``
- ``config/mine-parallel-preferences.toml``
- ``config/parallel-profiles.toml``
- ``config/benchmark-targets.toml``

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

   query.to_parquet(
       "results_parquet",
       batch_size=10000,
       parallel=True,
       pagination="auto",
       profile="large_query",
       ordered="unordered",
       inflight_limit=8,
   )

DuckDB SQL over Parquet output
------------------------------

.. code-block:: python

   con = query.to_duckdb(
       "results_parquet",
       table="results",
       parallel=True,
       profile="large_query",
   )
   print(con.execute("select count(*) from results").fetchone())
