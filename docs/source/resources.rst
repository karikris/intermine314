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

Analytical exports consume views of validated pages directly. They do not
flatten each page into public rows and assemble another list. A page remains
charged until all its batch views have been consumed. Public ``iter_batches``
continues to return row lists; explicit custom batch/row providers retain their
original dispatch and close behavior. Every page must exhaust and validate its
terminal server status before any batch from it can reach Parquet staging.

Admission estimates include sampled Python object and container overhead.
Converted native buffers are charged alongside their emitting page until the
write finishes; schema conversion also records conservative buffer overlap.
Logs expose current and peak native batch bytes. This remains an estimate:
sampling, unusually large responses and native scratch allocations can exceed
the configured byte budget. An oversized page drains alone to retain progress.
There is no additional unbounded conversion queue or completed-page backlog.

Typed conversion
----------------

Flat scalar batches use explicit Arrow types before transfer to Polars. Integer
ranges, declared model widths, calendar dates and Decimal precision/scale are
checked before publication. Integer construction does not rely on Arrow's
``safe=True`` alone, which permits truncating Python floats. Declared Float and
Double retain their existing IEEE rounding behavior and reject finite overflow.
Mixed numeric inference, nested values, timezone interpretation and custom
schema hooks retain the established lossless conversion path.

The writer records deduplicated schema signatures when it writes each owned
staged part. Stable parts require no schema-discovery read. When null inference
or Decimal scale changes the final schema, only affected parts are reread and
rewritten. Validation of external Parquet remains separate from this trusted
staging metadata.

Both analytical export and ``model_batch_frame`` use one conversion entry point.
The previous full model batch round-trip loop has been removed. A column-level
compatibility fallback remains for atypical scalar coercion and direct large
integer-to-Float32 rounding; general nested/mixed inference keeps its separate
lossless fallback. Conversion and native engine reservations never nest across
Arrow and Polars stages.

Parquet layout
--------------

``ParquetOptions`` in ``intermine314.export`` separates producer ``batch_size``
from storage geometry. The starting targets are 122880 rows per row group and
256 MiB per directory file. ``row_group_bytes`` optionally lowers the estimated
row-group memory target; automatic sizing uses at most one eighth of the
profile memory envelope. DuckDB has a 2048-row minimum and approximate vector
and file-size boundaries. These are targets, not hard byte or RSS limits.

Pass ``parquet_options=ParquetOptions(...)`` to Query analytics, CSV import or
``fetch_from_mine``. ``dataframe`` and ``fetch_from_mine`` also accept
``single_file=False`` for persistent directory output. Their single-file
defaults remain unchanged. Producer page/batch sizes continue to bound fetch
and conversion work independently of output file count.

Directory compaction preserves numeric-part order using one DuckDB writer:
DuckDB does not support ``PRESERVE_ORDER`` with ``FILE_SIZE_BYTES``. Single-file
ordered compaction can use the profile's ordered COPY threads. Use
``compact=False`` to retain producer parts. Types whose logical metadata would
change through DuckDB, including timezone timestamps and fixed arrays, also
retain direct parts. Empty outputs always retain a typed part.

``preserve_order=False`` explicitly permits unordered compaction, using the
profile's unordered COPY thread count (8 for cpu16) and DuckDB's byte-based
row-group target. The default keeps ordering enabled. Both modes perform one
COPY with the profile's memory allowance and managed spill directory; there
is no outer query executor. ``ResourceProfile(duckdb_threads=1)`` retains an
explicit one-thread mode for smaller memory envelopes.

Completed staging on the target filesystem is moved into publication instead
of copied. A cross-device ``EXDEV`` error selects the copy fallback; other
errors propagate and preserve prior output. The shared publication helpers
also coordinate CSV and enrichment replacement by target. Unrelated directory
entries retain their existing copy-and-preserve behavior.

Directory manifests and schema freshness
----------------------------------------

Directories include ``_intermine314/manifest.parquet`` with relative part paths,
row counts, row groups, byte sizes, timestamps and serialized Arrow schemas.
Its ``ordered`` flag means compaction retained producer input order; it does
not assert a query sort order. The manifest is published atomically with the
directory and is excluded from package data scans. Single files retain their
native Parquet footer without a separately published sidecar.

External manifests never authorize skipping schema validation. Repeated reads
cache actual file schemas using device, inode, size, modification time and
change time. Replacements invalidate the cache even when modification time is
restored. The bounded process cache shares identical schema objects. Coordinate
external writers separately: this cache does not provide a snapshot transaction
across concurrent file mutations. Unrelated metadata entries are preserved;
an unrecognized file occupying the reserved manifest path is rejected.

Streaming SQL results
---------------------

``query_parquet_reader(path, sql=..., batch_size=65536)`` returns a context-owned
iterator of Arrow RecordBatches. Use ``with`` when stopping early. Exhaustion,
errors and ``close()`` release the reader, connection and private spill directory;
an explicit ``ResourceRuntime`` also closes its idle readers. Returned batches
remain usable after close. ``read_all()`` explicitly materializes remaining data.

A borrowed ``connection=`` gets a private DuckDB cursor. The package does not
change its database settings or close the parent. It charges the existing thread
count against native admission. The caller must coordinate any concurrent global
setting changes. Owned connections use the profile's SQL and memory settings.
Reservations end before yielding a batch, allowing downstream Arrow/Polars stages
to acquire their own budget. SQL sorts/aggregates can still require spill space
beyond one output batch. The materialized ``query_parquet`` API retains its
existing DataFrame contract and shares validation and connection setup.

Bounded CSV streams
-------------------

CSV paths retain lazy Polars scans and projection pushdown. Borrowed text and
binary streams use ``read(size)`` chunks of at most 1 MiB (smaller under a small
memory profile), spooled to temporary storage before parsing. Text encoding may
use up to four bytes per character. Caller streams remain open and are consumed
from their current position. Temporary input files are removed on success and
failure. Polars CSV dialect/schema options remain supported; explicit schemas
or ``infer_schema=False`` avoid expensive whole-file schema inference.

CSV identifier uploads
----------------------

List creation/appends with ``csv_input`` project the selected identifier column
as String into temporary Parquet, validate every identifier, and write a binary
upload file. Requests computes its remaining byte length and streams it in one
mutation attempt. Leading zeros, Unicode and quoted names retain their exact
wire representation. Late invalid identifiers cause no upload. This replaces
the accumulated Python identifier list and joined whole-input string; ordinary
non-CSV identifier inputs keep their legacy behavior.

Enrichment result modes
-----------------------

``List.calculate_enrichment(..., output_path=..., result_mode='path')`` returns
the published path without collecting a DataFrame. ``result_mode='reader'``
returns a context-owned Arrow reader of the published Parquet; it requires
Parquet output. The default ``'dataframe'`` retains the existing detached frame
contract. Every mode validates the terminal server status before replacing an
existing output. Reader creation happens after publication, so a later reader
setup error leaves the valid published file available for another reader.

Concurrent metadata and Registry leases
---------------------------------------

Metadata cache groups use one initializer per Service and key. Independent groups
and independent mines can initialize concurrently through HTTP admission. A
failed initializer remains retryable; related model/template views share their
snapshot lock. Flush and close wait for current metadata work before changing
its state. Keep each Service open for the full lifetime of its consumers.

Use ``with registry.lease('MineName') as service:`` in concurrent workflows.
Eviction or ``clear_cache()`` defers closing a leased client until its last lease
ends. Registry close also defers its shared transport while leases or client
initializers are active. Independent mine initializers overlap; repeated access
to the same cold key creates one client. Clearing during initialization rejects
that result instead of repopulating the cleared cache. Legacy ``registry[name]``
retains its existing lifetime contract; callers needing protection from another
thread's eviction must use a lease.

Streaming XML and ancestry
--------------------------

Model and template discovery parse 64 KiB reads through Expat and construct one
selected class/template subtree at a time. Model-name discovery validates events
without constructing another complete tree. Ancestry is memoized within each
validation pass, preserving parent order, diamond duplicates and descriptor
identity; standalone ancestry calls still observe public parent-list edits.

Model parsing retains source chunks for the existing full-source
``ModelParseError`` diagnostic. The explicit ``Service._get_xml`` DOM helper
retains its historical return type. These serve distinct compatibility contracts;
the normal model/template discovery paths no longer construct full DOM trees.

Seven paired local trials on a 1,000-class model measured median construction
0.03580 s before versus 0.01137 s after (3.15 times faster). Separate tracemalloc
runs measured median Python allocation peaks of 18,723,917 versus 3,119,573 bytes.
These are fixture measurements, not network speedups or native/process RSS
limits. Reproduction and descriptor parity hashes are in the phase 8 Parquet
evidence alongside the benchmark script.

Batched object prefetch
-----------------------

``intermine314.results.prefetch_objects(objects, 'name', 'company', ...)`` yields
the same objects in input order after fetching missing first-level fields in
bulk. Requests group by exact Service, model and class, with 256 IDs per request
and the profile's metadata worker count by default. ``batch_size`` and
``max_workers`` are explicit controls. A window holds at most their product in
input object references and submits at most one request per worker at a time.

Loaded/selected nulls remain untouched. Duplicate input objects, complete
collections and existing nested identity maps retain their semantics. The server
status footer must pass for every request before updating the current window.
A failed window remains unchanged; earlier yielded windows remain populated.
References use outer joins and no joined-row size cap. Large relationship fanout
can still produce large responses; lower the ID batch size for those workloads.
Use a second prefetch call for fields on nested objects. Keep Services and input
iterators open; Registry users should hold a lease throughout consumption.

The legacy lazy field accessor remains for callers who request fields one at a
time. Its fetch eligibility check is shared with prefetch; it is a distinct public
behavior rather than a deferred removal. Prefetch uses dictionary/object response
semantics and does not require importing analytics engines.

Coordinated identifier job polling
---------------------------------

``intermine314.idresolution.poll_jobs(jobs, max_workers=..., timeout=...,
cancel_event=...)`` waits for all jobs and returns them in input order, including
duplicates. One monotonic due queue owns backoff; reusable workers only perform
status HTTP calls. Terminal SUCCESS/ERROR jobs are not polled again. Job owner
locks prevent simultaneous coordinators and legacy ``Job.poll()`` calls from
racing over status/backoff. The historical one-job polling contract is retained.

Cancellation and timeout stop scheduling and leave remote jobs intact. Local
cleanup still waits for already running HTTP reads under their request timeouts.
Keep each Service alive while polling. The default worker count comes from the
metadata stage, with global and per-host HTTP admission still applying.
