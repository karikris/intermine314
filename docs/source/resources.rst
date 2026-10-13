Resource profiles
=================

``intermine314.export.resource_profile`` defines advisory profiles ``cpu16``
(the default), ``cpu8``, ``cpu6``, ``cpu4`` and ``cpu2``. ``tor_low_mem`` remains
available. Automatic CPU allocations respect process affinity and discoverable
Linux container quotas. Explicit HTTP worker overrides remain separate from
native CPU allocations.

Stage defaults
--------------

======================= ===== ===== ==== ==== ====
Stage                   cpu16 cpu8  cpu6 cpu4 cpu2
======================= ===== ===== ==== ==== ====
Public HTTP             4     4     4    4    2
Low-latency HTTP         2     2     2    2    2
High-latency HTTP        8     8     6    4    2
Arrow CPU               4     4     3    2    1
Arrow I/O               2     2     2    2    1
Ordered COPY            4     4     3    2    1
Unordered COPY          8     8     6    4    2
DuckDB SQL              16    8     6    4    2
Polars                  12    8     6    4    2
Python processes        4     4     3    2    1
Metadata                4     4     4    4    2
Mutable resource owner  1     1     1    1    1
======================= ===== ===== ==== ==== ====

These are stage allocations, not simultaneously allocated pools. The memory
planning envelope defaults to the smaller of 4 GiB and 25% of available memory,
including discoverable container limits; the portable fallback is 512 MiB.
The shared queue allowance is the smaller of 64 MiB and one eighth of the
envelope. Explicit ``memory_budget_bytes`` and queue settings override defaults.
Neither Arrow buffer bytes nor a database memory limit guarantees process RSS.

Set ``INTERMINE314_RESOURCE_PROFILE`` to select a named profile, or construct a
``ResourceProfile`` explicitly. Explicit profiles take precedence over that
environment variable. Existing ``INTERMINE314_TEMP_DIR`` behavior is preserved.
Counts must be positive integers; booleans and fractional values are rejected.

Integration status
------------------

``ResourceRuntime`` in ``intermine314.export.runtime`` owns native admission
and closeable package resources. Native operations across runtime instances
share a process-wide queue. While a smaller-budget operation is active,
overlapping package operations respect that smaller budget. Reserve only an
individual native operation, not an entire fetch/convert/write pipeline.

Polars configuration is applied before its first import when possible.
Already initialized Polars and Arrow pools remain unchanged; diagnostics and
one-time log warnings report requested and effective counts. Oversized pools
receive an exclusive package reservation. External application work is outside
the advisory budget. No isolated process is silently introduced.

``runtime.diagnostics()`` exposes effective pools, memory and admission state.
``runtime.close()`` waits for active calls and closes owned resources once.
Borrowed resources must not be registered as owned. The context manager calls
``close()``; ``shutdown_resources()`` closes only implicit package runtimes.
``Query.to_parquet()``, ``export()``, ``dataframe()``, ``to_duckdb()``,
``fetch_from_mine()``, ``import_csv()`` and ``query_parquet()`` accept
keyword-only ``resource_profile`` and ``runtime`` arguments. Template helpers
forward the same controls. Supplying a conflicting profile and runtime fails
before input or output is opened. Existing materialized return values and
default output layouts are preserved.

For example::

    from intermine314.export import ResourceRuntime, query_parquet

    with ResourceRuntime("cpu8") as runtime:
        result = query_parquet("results.parquet", runtime=runtime)
        print(runtime.diagnostics())

Explicit ``ParallelOptions`` values take precedence over profile defaults.
Default fetch prefetch is one page per worker, including large-query mode;
explicit prefetch remains available. Native conversion, writes and owned SQL
operations reserve capacity for their respective stages. Connections returned
by ``to_duckdb()`` or ``fetch_from_mine()`` have configured native thread limits,
but subsequent SQL issued directly by callers is outside package admission.

HTTP admission
--------------

All package openers share request permits keyed by normalized scheme, hostname
and port. A permit remains charged through response consumption and close,
including streaming readers. ``per_host_limit`` overrides the stage's starting
host limit. The global request limit defaults to 16; an explicit profile
``max_workers`` supplies its global request ceiling. Concurrent callers with
different ceilings respect the smallest active ceiling.

Responses indicating 429 or 503, including managed retry history, halve the
host's admission window. ``Retry-After`` establishes a cooldown; eight successful
responses allow one additional request, up to the configured ceiling. Transport
retries still belong to the existing retry policy, and mutations remain single
attempts. Admission waits respect the request timeout and runtime cancellation.
Connection-pool retention is independent of these active-request limits.

HTTP worker ownership
---------------------

A runtime reuses its HTTP executor for the same worker count. Finishing or
cancelling a job waits only for that job's running requests and cancels its
queued requests. Close explicit runtimes after all consumers finish; use
``shutdown_resources()`` for implicit runtimes in long-lived applications.

Package workers use private sessions with per-request configuration snapshots
and immutable retry adapters. Service close releases these sessions. Existing
cookies, authentication, response hooks or custom adapters select a serialized
owner instead. A newly received cookie is copied to that owner for subsequent
requests; requests already in flight finish with their existing snapshots.
Do not change session configuration concurrently with requests.

Borrowed sessions retain their configuration, adapters and lifetime. Package
openers sharing one borrowed session serialize request preparation and sending;
independent response streams can still be consumed concurrently. Direct use of
that session by application code is outside this coordination.

Mutable owners
--------------

Account operations and list mutations share one reentrant lock per Service.
Independent managers reserve anonymous names together, but each manager still
cleans up only its own temporary lists. Separate Service objects or processes
using the same account require application coordination.

Parquet publication serializes preservation of unrelated entries, replacement
and rollback for the same resolved target path within a process. Conversion
and independent targets can run concurrently. The last successful publisher
wins; readers see complete datasets, subject to the documented brief missing
directory on platforms without atomic directory exchange. This is not a
cross-process transaction or a durability guarantee.

Parallel query snapshots
------------------------

Built-in parallel Query and Template execution prepares invariant form fields
once before fetching pages. Later edits to the original query affect future
executions. Every page retains its own offset, limit and result decoder. Public
``clone()`` behavior and custom ``results()``/``count()`` providers are preserved.
