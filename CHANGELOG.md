# Release notes

## 0.2.0 — 19 October 2026

This release restores a broad set of historical InterMine calls around a shared
Python 3.14.5+ runtime and expands the Parquet, DuckDB, Arrow and Polars pipeline.
It also addresses export failures, compressed metadata reads and resource
accounting. Review the migration notes before upgrading an existing application.

### Compatibility and analytics

- Restore `intermine314.webservice`, results, model/query expressions, saved XML,
  templates, lists, summaries, registry helpers and identifier-resolution calls.
  Historical imports share canonical implementations and exception identities.
- Preserve native dictionary defaults through `intermine314.service.Service`;
  the historical `webservice.Service` facade selects legacy row/object defaults.
- Provide typed Arrow conversion, Parquet exports and manifests, DuckDB SQL,
  streaming readers, optional borrowed connections and Polars results.
- Add explicit CSV ingestion into the Parquet pipeline and path-based spawned
  transforms for coarse Python CPU work.

### Parallel processing and memory

- Introduce CPU profiles 16, 8, 6, 4 and 2 with stage allocations that respect
  discoverable affinity/container limits and shared native admission.
- Reuse executors; coordinate Arrow, DuckDB and Polars work; bound outstanding
  pages including completed pages waiting for ordered consumption.
- Apply HTTP global/per-host admission, adaptive server cooldown and worker
  sessions. Managed retries cover declared read-only operations.
- Share immutable result layouts and object projections; remove the redundant
  count request when materializing a result iterator with `list()`.
- Stream XML metadata through decompressed sized reads; coordinate cache
  initialization and resource closure. Add explicit object prefetch and shared
  identifier-job polling.

### Storage and correctness

- Validate a parallel page's terminal server status before accepting its rows.
- Stage and validate exports before replacing existing output; preserve previous
  data on failed publication and enforce per-target ownership.
- Compact directory output, control row-group/file targets and write a typed
  Parquet manifest. Promote eligible single staged numeric parts without a
  redundant DuckDB COPY.
- Preserve supported exact integers and decimals; validate schemas and reject
  unsupported precision/timestamp bridges before publication.

### Measurements and limits

The dated comparison against published **0.1.7 source on identical modern
dependencies** found 31% less time to materialize 10,000 dictionary rows and one
request instead of two. A 100,000-row directory fixture used **60% less total
space** and one data file instead of 100. Dictionary streaming took 35% more time;
custom-batch numeric single-file and directory exports took 129% and 75% more
time. Those writer fixtures exclude the built-in HTTP columnar fast path.

Separate comparisons against pre-uplift development checkout `8e584c5` found
3.19× keyed-row access and 3.36× eligible numeric single-file export. The separate
phase 1 → phase 2 comparison found an 84.7% reduction in retained Python
allocations for shared keyed rows. These figures use development baselines
and do not describe an upgrade from published 0.1.7.
No universal speedup, hard RSS limit or optimal worker count for all servers is
claimed. See the [full report](docs/analysis/evidence/v020-vs017/report.md) for
fixtures, dependencies, raw Parquet trials and measurement boundaries.

### Migration requirements

- Require Python **3.14.5+**, Requests ≥2.34.2, urllib3 ≥2.8.0,<3, Polars ≥2.0.0,
  DuckDB ≥1.5.6 and PyArrow ≥26.0.0. PyArrow is now a core dependency.
- Change `intermine` imports and distribution requirements explicitly. No
  original-name namespace shim is installed. `dataframe()` returns Polars.
- Result iterators are unsized: use `count()` on a query, materialize once when
  appropriate, or probe and close a stream. Truthiness does not test emptiness.
- Configure result parsers before opening a stream. Built-in streamed metadata
  is shared and immutable; direct result constructors retain mutable metadata.
- Replace former model submodule imports with public classes in
  `intermine314.model`. Use result/session entry points instead of
  `service.iterators`.
- Replace removed `registry.mines` policy resolvers, old parallel policy files
  and `tor_*` convenience helpers with explicit resource/transport configuration.
- Direct `Query(Model(...))` infers legacy behavior; pass
  `compatibility="native"` when native dictionary defaults are required.

The [migration guide](docs/source/release_readiness.rst) describes tested scopes
and remaining differences. The historical inventory records 331 passing scopes
and 129 tested departures across 460 symbols; this is not universal equivalence.
Windows, macOS, free-threaded builds and subinterpreter/native-object interchange
are outside the certified platform scope. License metadata includes MIT and
BSD-2-Clause with the required notices.
