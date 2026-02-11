Benchmark Profiles and Matrix
=============================

Default matrix in ``benchmarks/runners/run_live.py``
----------------------------------------------------

The benchmark runner executes a six-scenario matrix by default:

- Small matrix: ``5k``, ``10k``, ``25k`` rows
- Large matrix: ``50k``, ``100k``, ``250k`` rows

Profile mapping is controlled by ``config/benchmark-targets.toml`` and
``config/mine-parallel-preferences.toml``.

Benchmark profiles
------------------

Current profile definitions:

- ``benchmark_profile_1``: legacy ``intermine`` baseline + ``intermine314`` workers ``2,4,6,8,10,12,14,16,18``
- ``benchmark_profile_2``: ``intermine314`` workers ``4,8,12,16``
- ``benchmark_profile_3``: ``intermine314`` workers ``4,6,8``
- ``benchmark_profile_4``: legacy ``intermine`` baseline + ``intermine314`` workers ``4,6,8``

Mine-specific defaults
----------------------

Priority-supported mines:

- ``MaizeMine``
- ``ThaleMine``
- ``LegumeMine``
- ``OakMine``
- ``WheatMine``

Default behavior from mine registry configuration:

- Most mines: small matrix uses profile 1, large matrix uses profile 2.
- ``LegumeMine``: small matrix uses profile 4, large matrix uses profile 3.

Low-memory benchmark execution
------------------------------

To keep benchmark scripts lightweight and memory-efficient:

- Keep ``--auto-chunking`` enabled.
- Prefer ``--ordered-mode unordered`` for throughput tests.
- Set ``--inflight-limit`` to prevent unbounded task submission.
- Use Parquet outputs and DuckDB for analysis; avoid large CSV intermediates unless explicitly comparing CSV.
- Use targeted exports (core table + edge tables) instead of one wide join.

Example command
---------------

.. code-block:: bash

   python benchmarks/runners/run_live.py \
     --benchmark-target maizemine \
     --matrix-six \
     --repetitions 3 \
     --auto-chunking \
     --ordered-mode unordered \
     --inflight-limit 8 \
     --json-out /tmp/intermine314_benchmark_maizemine.json

GitHub Pages output
-------------------

Benchmark runs are published as static pages under ``docs/benchmarks/results``.
Use ``--pages-out`` to override the local output path if needed.
