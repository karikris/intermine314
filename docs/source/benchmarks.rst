Benchmark Profiles and Matrix
=============================

Live benchmark entrypoint
-------------------------

Use ``benchmarks/runners/run_live.py`` for live network benchmarks.

The default matrix executes five scenarios:

- rows: ``5k``, ``10k``, ``25k``, ``50k``, ``100k``

Profile and row constants come from:

- ``benchmarks/profiles/benchmark-constants.toml``
- ``benchmarks/profiles/benchmark-targets.toml``
- ``benchmarks/profiles/mine-parallel-preferences.toml``

Current profile definitions
---------------------------

- ``non_restricted``: workers ``4,8,12,16``
- ``server_restricted``: workers ``3,6,9``

Mine defaults:

- most mines: ``non_restricted``
- restricted mines (for example ``LegumeMine`` and ``MaizeMine``): ``server_restricted``

Phase-0 guardrails
------------------

For stable CI artifacts, use the Phase-0 runners:

- ``benchmarks/runners/phase0_guardrails.py``
- ``benchmarks/runners/phase0_baselines.py``
- ``benchmarks/runners/phase0_parallel_baselines.py``

Storage comparison and reference environment
--------------------------------------------

The storage benchmark writes Parquet for both native and optional original-client
fetches, then measures a DuckDB scan and bounded Arrow-to-Polars batches. It reports
row counts and hashes of at most 64 rows sorted by selected columns. These sampled
checks do not prove complete dataset equality or identical remote snapshots.

Set ``INTERMINE314_REFERENCE_PYTHON`` to the Python executable in a separate
reference environment containing the pinned original ``intermine==1.13.0`` and
this checkout's dependencies. Without it, storage comparison records a skipped
reference and null parity. An explicitly requested fetch reference requires it.
The native/base environment needs no original-client dependency. See
``benchmarks/README.md`` for setup, schema versions and output controls.

Outputs and offline verification
---------------------------------

``--storage-output-dir`` selects Parquet artifacts and ``--json-out`` selects the
JSON report. The current storage schema is ``parquet_storage_compare_v3``; older
CSV/Pandas reports describe their original runs and are not relabeled.

``make analyticscheck`` runs an offline input-to-SQL smoke check. Pytest validates
reference page retries, rollback, iterator closure, exact supplied values and the
sample/count pipeline. Live timings must be obtained by actually running against
a mine; offline fixtures are not performance measurements.
