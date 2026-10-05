intermine314 Documentation
==========================

``intermine314`` is the Python 3.14.5+ InterMine WebService client.

Priority mines are ``MaizeMine``, ``ThaleMine``, ``LegumeMine``,
``OakMine``, and ``WheatMine``.
The latest read-only validation passed for the first three. OakMine and
WheatMine returned anti-bot HTML; their API behavior remains unverified.

The modern data workflow in this package is:

1. Query InterMine services using ``Service`` + ``Query``.
2. Materialize rows into ``polars.DataFrame`` objects.
3. Persist large results as Parquet files.
4. Query Parquet datasets in DuckDB and return Arrow-backed Polars results.
5. Use parallel page fetching for faster retrieval from remote mines.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   query
   release_readiness
   benchmarks

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   modules

Ownership and Credit
--------------------

- Copyright (c) 2026 Monash University, Plant Energy and Biotechnology Lab.
- Project owners: Kris Kari and Dr. Maria Ermakova.
- Contact: toffe.kari@gmail.com
- Foundational credit: original InterMine team and community contributors.
