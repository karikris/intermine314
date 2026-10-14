# 0.2.0 release readiness against published intermine314 0.1.7

Reviewed 14 October 2026. Runtime source: master
`a77db1dedf02b7e3c0bc992466dd6ae2016a0463`, still declared **0.1.8**.
The report commit adds documentation and evidence; it does not change the runtime,
bump its version, create a release tag or publish to PyPI.

## Decision

**Ready to prepare a qualified 0.2.0 feature/correctness release on the tested
Linux CPython 3.14.5 platform; not ready to describe as an unconditional drop-in
replacement or a universally faster 0.1.7.** No new release-blocking correctness
failure was reproduced in this review. A temporary 0.2.0 build passed all six
installation modes and 904 regressions against installed packages. The complete
source suite has 2,813 passing tests and an executed historical API audit.

There are material migration changes and performance trade-offs. On matched
modern dependencies, materializing 10,000 dictionary rows takes 31% less time,
but streaming those rows takes 35% more time. The custom-batch export fixtures
take 75–129% more time. Compaction cuts the directory fixture's total storage
by 60%. Release notes must disclose these results. If the acceptance criterion
is “no workload slower than 0.1.7”, the candidate **does not meet it**.

The remaining release work is a controlled version/readme change, fresh
artifact verification on that exact commit, green CI, and the matching tag and
authorized upload. The rehearsal artifacts are not final release artifacts.

## Baseline and review scope

The baseline is the actual [PyPI intermine314 0.1.7 release](https://pypi.org/project/intermine314/0.1.7/),
published 5 March 2026, with attested source commit
`8d4b833b61f206cd7932707db0520a8e133879af`. Both downloaded distributions were
verified against PyPI's SHA-256 metadata:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| 0.1.7 wheel | 86,254 | `31f64151cb65bc3ff9b5d3418327e06d048bbb38e5d4ffa45480086c08167ee5` |
| 0.1.7 sdist | 70,851 | `18974b4013016ac524cf6e784ee565e5cee19022c63c8349453582b29c9e2ee6` |
| Temporary 0.2.0 wheel | 174,877 | `dc3233127752d99cc94194cfb75c63302a008b0617909171c9bcb85c00cee91f` |

The review inventories every runtime Python file and declaration, checks the
existing executed API evidence against all 79 runtime hashes and controlled
execution inputs, and follows the principal execution paths: model/query DSL,
templates, JSON/XML parsing, transport ownership/retries, pagination, native
admission, Arrow conversion, Parquet publication, DuckDB readers, CSV ingestion,
list upload, object prefetch, job polling, spawned transforms and release gates.
The inventory is structural; a moved declaration is not by itself proof of a
broken public API. Focused import and behavioral probes supply that evidence.

This is **not** a comparison with the original `intermine` distribution.
That separate compatibility contract is inventoried against InterMine Python
client 1.13.0. Nor is it the previous uplift benchmark against development
checkout `8e584c5`.

## What the candidate adds or strengthens

0.1.7 already had parallel offsets, Polars, DuckDB, Parquet and Tor support.
The following describes the expansion and hardening of that foundation.

| Area / code | Candidate behavior and practical benefit | Boundary |
|---|---|---|
| CPU policy: `export/resource_profile.py`, `runtime.py` | Default total active native CPU budget 16; 8/6/4/2 profiles; affinity/cgroup-aware stage plans, shared admission and reusable executors | Cooperative package admission, not a hard OS thread cap; external pools remain outside control |
| HTTP scheduling: `service/admission.py`, `transport.py` | Global/per-host permits, response-lifetime admission, adaptive 429/503 cooldown, worker sessions and read-only retry policy | Public-mine optimum unmeasured; borrowed sessions retain their own policies |
| Pagination: `query/parallel_offset.py`, `inflight.py` | A page's terminal status is validated before acceptance; pending, buffered and emitting pages stay charged against the envelope | Estimates are not RSS; one oversized page may proceed; offset pagination cannot stabilize a changing dataset |
| Export pipeline: `query/_columnar.py`, `export/arrow.py` | Built-in validated pages can feed typed columns without intermediate row dictionaries; one conversion plan per execution and homogeneous-batch reservation | Custom dictionary providers retain the general path; strict mixed-type validation remains |
| Arrow/engine control: `export/runtime.py` | Stage-specific Arrow CPU/I/O pools, DuckDB settings and native reservations prevent blindly multiplying nested worker pools | An already initialized Polars pool cannot be resized; diagnostics report actual pools |
| Metadata: `service/xmlstream.py`, `metadata.py`, `model.py` | Sized decompressed reads, incremental XML events, selected subtrees, cache initialization locks and coordinated invalidation/close | Large selected objects still occupy memory; whole metadata is not a constant-memory object model |
| Result construction: `service/_result_layout.py`, `_result_object.py` | One shared immutable layout/projection per built-in stream, independent object caches, parser binding once at stream open | Direct constructors preserve mutable behavior; stream metadata mutation requires migration |
| Result consumption: `service/session.py`, `query/executor.py` | Unsized streams remove the extra count request during `list(results)`; shared wire preparation and byte-oriented JSON decoding | `len(results)` is removed; simple dictionary streaming regresses in this comparison |
| Query/templates: `query/builder.py`, `template.py` | Shared query machinery for native/legacy calls; one private execution snapshot for built-in eager template helpers | Public `clone()` remains deep; custom dispatch preserves its distinct path |
| Directory storage: `export/layout.py`, `parquet.py` | Default compaction, target row groups/files, schema reconciliation and typed Parquet manifest replace numerous tiny output parts | More write work and temporary disk; output size depends on data and layout |
| Single-file storage: `export/parquet.py` | Direct promotion of one provably equivalent staged numeric/Boolean/Date part avoids unnecessary COPY/recompression | Narrow Zstd/schema/row-group gate; strings, decimals, nested/extension types and multiple staged parts keep COPY |
| Publication: `export/parquet.py`, `util/ownership.py` | Stage and validate before replacing output; rollback on failed file/directory export; per-target owner | Portable directory fallback can expose a replacement gap; not cross-process or crash-transactional publication |
| Schema safety: `export/_schema.py`, `arrow.py` | Exact integer/decimal checks, supported temporal normalization, bounded file-identity/schema cache, explicit rejection of unsupported bridges | Arbitrary Python objects, precision above supported limits and unsupported timestamp/Int128 annotations are not silently coerced |
| DuckDB access: `export/query.py` | SQL over Parquet, managed Arrow readers, optional borrowed parent connection/private cursor, detached Polars results | Caller owns borrowed connection settings/lifetime; cancellation checks between batches retain a Python loop |
| CSV and list input: `export/csv.py`, `lists/_csv.py` | Explicit CSV-to-Parquet path, bounded spooling/chunking, validated quoted identifier upload with known length before mutation | Final frames and source data can still be large; server mutations remain serial |
| Object relationships: `_prefetch.py` | Opt-in bounded identifier windows group requests by service/model/class; validate window before updating caches | No speculative automatic reads; returned collections can be large |
| Independent metadata/mines: service runtime | Overlap cold read-only discovery and reuse Service/transport/cache lifetime | One initialization per key; do not concurrently mutate configuration |
| Identifier jobs: `idresolution.py` | Due-time heap and a small shared polling pool instead of a sleeping worker for every job | Cancellation does not implicitly delete remote jobs |
| Python CPU transforms: `export/transforms.py` | Spawned workers exchange Parquet paths, limit native pools and bound queued coarse tasks; validate outputs before publication | Startup/IPC overhead; no certified interchange of live native objects through subinterpreters |
| Original-client API: `webservice.py`, `model.py`, `results.py`, lists and registry | Historical import facades, DSL, result modes, XML/templates, list operations, summaries, identifier jobs and optional plotting share canonical implementations | Polars replaces pandas; explicit departures and unsupported private APIs remain |
| Packaging and release: verifier scripts/workflows | Modern Python/dependency floors, wheel/sdist/extras checks, installed regressions/docs, license notices, exact-commit/artifact-hash release gates | Actual trusted-publisher credentials/upload are exercised only at release |

There is no new TypeScript execution layer or bespoke Rust extension. Native
execution comes from Polars, DuckDB and Arrow. Python remains the orchestration
and compatibility layer. No speedup is attributed to Python 3.14 itself: both
sides of the local comparison use the same 3.14.5 interpreter, with JIT disabled.

### Worker defaults

These are stage allocations within a shared budget, not pools to add together.
The environment used here exposes 18 CPUs of the user's 20-core host; the
defaults are starting policies, not measured optima for every mine or the bare host.

| Stage | Default at budget 16 | Main reason / limit |
|---|---:|---|
| Public-mine HTTP | 4 | Latency overlap with per-host admission |
| Low-latency HTTP | 2 | Avoid excess Python parse contention |
| High-latency capable server | 8 | More overlapping requests, subject to server permission/caps |
| Arrow CPU / I/O | 4 / 2 | Conversion and I/O have distinct pools |
| Ordered / unordered compaction | 4 / 8 | Ordered COPY has greater memory cost; low-memory profile uses 1 |
| DuckDB SQL | 16 | One main native query with enough work/row groups |
| Polars | 12 | One streaming plan; configure before initial import |
| Coarse Python processes | 4 | Avoid startup/IPC and native oversubscription |
| Independent metadata/mines | 4 | Read overlap with cache/lifetime guards |
| Mutation, publication, plot owner | 1 | Serialize each mutable resource |

The 8/6/4/2 profiles clamp these allocations to available CPU and smaller stage
limits. Polars' fixed-pool constraint was reconfirmed through GitHits against
[its pinned 2.0.0 source](https://github.com/pola-rs/polars/blob/22a147de3d2bb2e44b97338a2510816c7105c9f2/py-polars/src/polars/meta/thread_pool.py).
See [resource controls](../../source/resources.rst) for configuration and admission limits.

## Fresh direct correctness comparisons

`probes.py` runs both source trees in separate processes on identical current
dependencies, with deterministic local responses. Results are in `probes.parquet`.

| Probe | Published 0.1.7 source | Candidate |
|---|---|---|
| Parallel page: two rows followed by `wasSuccessful=false` | Accepts both rows without raising | Rejects page with `ParallelExecutionError`, caused by `WebserviceError` |
| Gzip XML read four bytes at a time through the response adapter | Compressed bytes reach XML parser; parse fails | Decoded XML parses successfully |
| Directory export yields a batch, then producer raises | Existing output is not preserved | Previous Parquet bytes are preserved |
| Historical `webservice`, `results`, `lists.listmanager` imports | Absent | Present |
| Former `model.class_`, `registry.mines`, `service.tor`, `service.iterators` imports | Present | Absent; migration required |

The full regression suite additionally covers buffered-page accounting,
compressed response variants, failures/cancellation and schema/precision boundaries.
The fresh probes above do not independently reproduce every historical defect.

## Measured timing, memory and storage versus 0.1.7 source

**Scope:** CPython 3.14.5, Linux/WSL2, identical Requests 2.34.2, urllib3 2.8.0,
Polars 2.0.0, DuckDB 1.5.6, PyArrow 26.0.0 and orjson 3.13.0. Polars pool 12,
Arrow CPU 4/I/O 2. Each case has 21 randomly ordered pairs of fresh processes;
one warmup and imports are excluded from elapsed time. Each pair checks equal
results; exports also check exact row values/order. Filesystem caches are not
purged. Medians/IQRs and source digests are retained, not just headline values.

The published package pins DuckDB 1.0.0. A binary-only download found **no
compatible wheel for this CPython 3.14 Linux environment**. Source builds were
not tested. Therefore this is a source-level comparison on matched modern
dependencies, **not two stock installation stacks**. Dependency uplift effects
are not isolated by this experiment.

| Workload | 0.1.7 median ms [IQR] | Candidate median ms [IQR] | Elapsed-time change |
|---|---:|---:|---:|
| Stream 10k dictionaries × 20 fields, scalar sum | 8.645 [8.530–8.779] | 11.685 [11.517–11.943] | **+35.2%** |
| Materialize 10k dictionaries × 20 fields | 18.956 [18.612–19.079] | 13.022 [12.930–13.502] | **−31.3%** |
| Custom numeric batches → single file, 10k rows | 9.487 [9.267–9.784] | 21.692 [21.471–22.679] | **+128.6%** |
| Custom integer/string batches → directory, 100k rows | 59.562 [58.187–60.913] | 104.396 [100.170–106.425] | **+75.3%** |

Dictionary list materialization makes **one request rather than two**. These
responses are in-memory fixtures, so real network latency would change the
absolute request saving. Custom writer fixtures supply model-free dictionaries
in 1,000-row batches, bypassing the built-in HTTP columnar fast path. Their
timings include export and a subsequent Polars sum. The numeric case stages ten
parts, so the one-part promotion optimization does not apply. Compaction,
schema checks and safe publication add work; their individual shares have not
been profiled and cannot be inferred from these end-to-end timings.

| Memory metric: separate `tracemalloc` run | 0.1.7 bytes | Candidate bytes |
|---|---:|---:|
| Streaming dictionaries | 13,133 | 15,375 |
| Materialized dictionaries | 4,732,550 | 4,734,783 |
| Custom numeric single-file export | 23,057 | 44,816 |
| Custom directory export | 30,231 | 128,059 |

These are Python allocation peaks inside the measured operation, **not process
RSS**, native-engine allocations or total input storage. They do not establish
a memory reduction versus 0.1.7. Better bounded queues and shared result layouts
improve particular memory risks; stronger general claims need peak-RSS testing
over representative full jobs.

| Storage fixture | 0.1.7 | Candidate | Saving |
|---|---:|---:|---:|
| Numeric single file | 35,345 bytes | 34,807 bytes | 1.5% |
| Directory data only | 358,045 bytes | 139,905 bytes | 60.9% |
| Directory including metadata/manifest | 358,045 bytes | 143,069 bytes | **60.0%** |
| Directory data files / row groups | 100 / 100 | 1 / 1 | 99% fewer |

The directory fixture has 100,000 rows of sequential identifiers and 100
repeating string values. Compaction substantially improves this small-file case;
do not extrapolate 60% to already well-sized Parquet datasets. Manifest and
temporary staging space must be included when estimating a real job's storage.

Earlier [uplift measurements](../uplift-2026-10-14/report.md) report 3.19× keyed-row
access, 3.36× eligible numeric export, 9.11× repeated borrowed-connection SQL and
84.7% less retained Python allocation for shared keyed rows. Those compare with
development checkout `8e584c5`, **not 0.1.7**, and cannot replace this table.
Similarly, the [CSV/Parquet storage study](../../source/storage_performance.rst)
compares formats: its 1M-row fixture is 61.7% smaller than CSV and 20.7% smaller
than gzip CSV. Those are not incremental savings from upgrading a package that
already wrote Parquet. Component speedups must not be multiplied into a total-job claim.

## Compatibility and migration

For code using the original client, preserve its service module while changing
the package name:

```python
from intermine314.webservice import Service
```

Native `intermine314.service.Service` keeps dictionary defaults; the historical
facade keeps model-object results and indexed rows. Both share the execution
and transport implementation. Query/model expressions, constraints, joins,
sorts, saved XML/templates, list operations, summaries, identifier resolution,
registry helpers and optional plotting have scoped compatibility assertions.

The inventory contains **460 historical symbols: 331 passing scopes and 129
tested departures across 43 groups**, with 795 curated semantic test nodes and
a supplemental 40-binding re-export contract. These are not an equivalence
percentage or a count of unchanged user programs.

Known migration work includes:

- `dataframe()` returns Polars; pandas operations require changes. No `intermine`
  namespace shim is installed and dependency declarations must be updated.
- Old 0.1.7 model submodules move to public classes in `intermine314.model`;
  `service.iterators` moves to result/session entry points. Removed private
  helpers/counters are not guaranteed replacements.
- `registry.mines` was a mine-specific policy module. Its production-plan,
  worker and transport resolvers were removed; the new `registry` discovery
  helpers do not replace them. Configure resource and transport policy explicitly.
- The four `tor_*` convenience helpers and old mine-parallel-preferences /
  parallel-profiles loader functions/environment variables are removed. Use
  `Service`/`Registry` transport arguments and `ResourceProfile`/`ParallelOptions`.
- Direct `Query(Model(...))` now infers legacy behavior; select the native
  profile explicitly when dictionary defaults are required.
- Result streams are unsized. `len(results)` raises; truthiness does not check
  emptiness. Streamed built-in metadata is shared/immutable; parser changes must
  occur before opening a stream.
- Python now requires 3.14.5+, dependency minima change, and integer resource
  arguments no longer accept booleans/floats through incidental coercion.
- Service close does not delete remote temporary lists. Mutations do not
  automatically retry; supplied sessions keep the caller's retry policy.

The [migration guide](../../source/release_readiness.rst) gives replacements and
covered/unsupported areas. Offline coverage does not certify arbitrary
third-party subclasses, private APIs, every argument combination or complete applications.

## Dependencies and package footprint

| Dependency / requirement | Published 0.1.7 | Candidate |
|---|---|---|
| Python | ≥3.14 | ≥3.14.5 |
| Requests | ≥2.32.4 | ≥2.34.2 |
| urllib3 | Transitive | Explicit ≥2.8.0, <3 |
| Polars | ==1.6.0 | ≥2.0.0 |
| DuckDB | ==1.0.0 | ≥1.5.6 |
| PyArrow | Not core | ≥26.0.0 core |
| Speed extra | orjson ≥3.10 | orjson ≥3.13 |
| Plotting | No equivalent plots extra | Optional Matplotlib ≥3.11.2 |
| Proxy extra | PySocks ≥1.7.1 | Same floor |
| Analytics / benchmark extras | Benchmark pulls original intermine/pandas | Empty compatibility aliases; analytics core |
| License metadata | MIT | MIT AND BSD-2-Clause, with notices |

GitHits dependency metadata confirms the old declared requirements. Fresh pinned
advisory queries returned no affected direct-version advisories for the baseline
package and the candidate's Requests, urllib3, Polars, DuckDB, PyArrow, orjson,
Matplotlib and PySocks versions. Raw responses are in `githits.parquet`. This is
provider-reported advisory evidence, not an exhaustive transitive/security audit.
The lower bounds remain open to future releases; dated passing CI is not proof
that every future resolver outcome is compatible.

| Runtime footprint | 0.1.7 | Candidate |
|---|---:|---:|
| Python files | 51 | 79 |
| Physical Python lines | 8,319 | 14,385 |
| Python source bytes | 280,716 | 553,431 |
| Compressed wheel bytes | 86,254 | 174,877 |

The source is **72.9% larger by lines**, despite removal of redundant paths in
the latest uplift. Restored compatibility, concurrency and correctness machinery
increase the net footprint. The wheel is about twice as large; this does not
measure installed dependency size, which also increases with core PyArrow.
Lazy imports avoid loading analytics for ordinary imports, not downloading
their wheels. Tests and analysis evidence do not ship in the wheel.

## Verification and remaining release gates

The reviewed source commit's [CI run 38475390991](https://github.com/karikris/intermine314/actions/runs/38475390991)
passed all seven jobs. Source hash and execution-input checks matched the
2,813-test API audit before this documentation update; the audit is regenerated
after editing controlled documentation inputs.

For an isolated rehearsal, `git archive` of the reviewed commit was unpacked
outside the checkout and only the project/runtime versions changed to 0.2.0.
`scripts/verify_distribution.py` passed base, plots, analytics alias, sdist,
speed and combined installation modes, dependency consistency, artifact/license
checks, Twine and strict installed-wheel documentation. Installed regression
verification passed **904 tests, zero failures/skips**, across 39 test files.
The temporary archive has no Git metadata, so its manifest correctly records
null source revision/dirty fields; it is not eligible as the final tagged artifact.

Not established by this review:

- Windows, macOS, free-threaded Python, subinterpreter safety or later Python minors.
- Fresh live public-mine performance or authenticated writes. The earlier dated
  live review covered LegumeMine, MaizeMine and ThaleMine read paths; OakMine and
  WheatMine returned anti-bot HTML. This turn used offline fixtures.
- Hard RSS limits, crash/cross-process directory transactions, stable snapshots
  of mutable offset-paginated datasets, or a universal 20-core optimum.
- Successful future PyPI authentication/upload; the workflow's external
  trusted-publisher configuration is exercised only on the authorized release.

Before publishing:

1. Carry these migration/performance disclosures into the release notes. If
   universal speed improvement is a requirement, profile and resolve the three
   measured regressions first; otherwise present them as release trade-offs.
2. Update `pyproject.toml` and `_version.py` to 0.2.0 and adjust the README's
   development-install wording in a separate release task.
3. Rebuild on that exact clean master commit, rerun the full tests/API evidence,
   all six distribution modes, installed regressions and strict documentation;
   require green CI and retain exact artifact hashes.
4. Use the matching version tag and guarded publish workflow after release
   authorization, then verify the published artifact hashes and clean installation.

## Evidence and reproduction

All canonical tabular evidence is Zstandard Parquet. `catalog.py` builds the
local DuckDB catalog with views over these files and regenerates `summary.parquet`.
Run from the repository root:

```bash
.venv/bin/python docs/analysis/release-v020-vs017/prepare_baseline.py
.venv/bin/python docs/analysis/release-v020-vs017/inventory.py
.venv/bin/python docs/analysis/release-v020-vs017/probes.py
.venv/bin/python docs/analysis/release-v020-vs017/compare.py
.venv/bin/python docs/analysis/release-v020-vs017/catalog.py
```

The first command verifies and extracts the pinned PyPI sdist to the temporary
baseline path used by the other scripts. Use the pinned environment recorded in
`environment.parquet` and run benchmarks without concurrent CPU-heavy checks.
Reruns replace local evidence and compare against the checkout at execution time;
commit/source digests distinguish them from this recorded review.

| File | Meaning |
|---|---|
| `trials.parquet`, `summary.parquet` | Final 168 observations; medians, IQRs, allocations, output bytes and requests |
| `exploratory_trials.parquet` | Earlier run with inconsistent Arrow CPU initialization; excluded from every reported comparison |
| `code_inventory.parquet`, `declared_symbols.parquet`, `module_changes.parquet` | Complete source hashes, structural declarations and module moves |
| `probes.parquet` | Direct correctness/import comparison outcomes |
| `dependencies.parquet`, `githits.parquet` | Declared old/current dependencies and provider responses |
| `verification.parquet` | PyPI metadata, artifact hashes/sizes, build and installed-test outcomes, binary-wheel check and API audit summary |
| `environment.parquet` | Interpreter, CPU visibility, versions, thread settings and reviewed source revision |

Example DuckDB query (the case column is quoted deliberately):

```sql
SELECT "case", variant, median(ns) / 1e6 AS median_ms,
       quantile_cont(ns, 0.25) / 1e6 AS q1_ms,
       quantile_cont(ns, 0.75) / 1e6 AS q3_ms
FROM trials GROUP BY "case", variant ORDER BY "case", variant;
```
