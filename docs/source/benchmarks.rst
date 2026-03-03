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
- ``intermine314.config/mine-parallel-preferences.toml``

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
- ``benchmarks/runners/phase0_model_baselines.py``

Stage model and parity
----------------------

Benchmark outputs include stage-level timing and memory plus parity checks across
comparable engine paths (schema, row count, sampled hash). Keep ``--strict-parity``
enabled for comparable scenarios.

Offline replay
--------------

Use offline replay modes to compare post-fetch storage/engine behavior with reduced
network variance.

Outputs
-------

Benchmark artifacts are written under ``docs/benchmarks/results`` and include run JSON,
run HTML, and rolling index pages.
