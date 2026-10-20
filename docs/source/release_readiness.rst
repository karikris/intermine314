Release readiness and migration
===============================

This guide describes version 0.2.0 and the cleanup reviewed on 20 October 2026.
Current cleanup task dispositions and CI handshakes are in
``docs/analysis/cleanup/``. Published measurements retain their original source
commits and dates under ``docs/analysis/evidence/``; they are not measurements of
the latest master commit.

Result streams are unsized; built-in streamed row/object metadata is shared and
immutable. Parser customization is captured when a stream opens. Direct result
constructors and metadata attribute rebinding remain supported. See :doc:`query`
for migration examples and counting queries versus consuming streams.

Earlier PyPI releases, including 0.1.8, predate the restored compatibility
described here. Install ``intermine314==0.2.0`` in a fresh Python 3.14.5+
environment. This release is a migration target with declared differences;
the original package namespace and pandas behavior require explicit migration.

Replacing the original client
-----------------------------

Install 0.2.0 and change imports from ``intermine`` to
``intermine314``. Keep the historical service import path to select legacy
defaults::

    from intermine314.webservice import Service, ServiceError

    with Service("https://maizemine.rnet.missouri.edu/maizemine/service") as service:
        query = service.select("Gene.primaryIdentifier")
        rows = query.rows(size=10)
        try:
            for row in rows:
                print(row["Gene.primaryIdentifier"])
        finally:
            rows.close()

The native ``intermine314.service.Service`` defaults to dictionary results.
The legacy facade retains model-object defaults for ``results()`` and indexed
``ResultRow`` values for ``rows()``. Either service accepts an explicit
``compatibility`` profile. Historical owned re-exports, such as
``webservice.ServiceError``, share the canonical implementations and exception
identities. A supplemental 40-binding contract covers these exports.

The package does not install an ``intermine`` namespace or satisfy another
package's dependency on the original ``intermine`` distribution. Update those
imports and dependency declarations explicitly. ``dataframe()`` returns Polars
under both profiles. Pandas-specific operations require migration to Polars;
the original client's incidental standard-library exports and private helpers
are outside the compatibility contract.

Close result iterators on early exit and close Service/runtime resources after
their consumers finish. Closing a Service does not delete server-side temporary
lists: use a ListManager context, ``delete_temporary_lists()`` or
``Service.flush()`` before closing when deletion is intended. Mutable resources
have one owner; concurrent code must not change Query or Session configuration.

Managed sessions retry explicitly read-only operations, including query POSTs.
Mutations and unknown operations do not automatically retry. A caller-supplied
Requests session retains the caller's adapters, retry policy and lifetime.
Review that policy when migrating: retries of a committed write can create
duplicate server state. Low-level callers may opt into read retries with
``opener.open(..., retry_safe=True)`` for a known read-only operation.

Coverage and limits
-------------------

.. list-table:: Evidence by area
   :header-rows: 1
   :widths: 22 44 34

   * - Area
     - Covered
     - Limits
   * - Historical APIs
     - Queries, models, constraints, XML, templates, lists, summaries,
       identifier jobs, registry helpers and optional plotting; 460 inventoried
       symbols and 40 supplemental bindings.
     - Scoped assertions and documented departures; exhaustive upstream
       equivalence and every argument combination are not certified.
   * - Analytics
     - Polars frames, typed Parquet, DuckDB SQL, Arrow conversion, explicit CSV
       ingestion and managed resources; exact large integers with and without
       the speed extra.
     - Pandas behavior differs intentionally. Typed exports retain their
       numeric-range and schema restrictions.
   * - Transport and concurrency
     - Safe managed retries, borrowed-session ownership, compressed XML,
       terminal page status, atomic export failure and bounded outstanding pages.
     - Byte limits are estimates rather than process RSS limits. One oversized
       page may proceed to avoid deadlock; running requests finish under their
       configured timeouts.
   * - Packaging and Python
     - Python 3.14.5 minimum and current 3.14 CI; Linux wheel/sdist installs in base, plots, speed, combined
       plots/speed/proxy and analytics-alias modes; licenses, Twine and installed
       documentation; temporary 0.2.0 candidate verification.
     - Windows, macOS, free-threaded builds and later Python minors are not
       certified by this validation run.
   * - Release workflow
     - Shared CI, matching version tags, master ancestry, immutable artifact IDs,
       completed-check manifests and exact file hashes; manual validation passed
       with publication skipped.
     - Actual PyPI upload and its external trusted-publisher configuration are
       exercised only during a separately authorized release.
   * - Live services
     - Read-only native/legacy samples succeeded on LegumeMine, MaizeMine and
       ThaleMine; serial, ordered parallel, Parquet/SQL/Arrow/Polars results agreed.
       All 98 discovered global templates parsed under each profile; one
       template per mine/profile was executed.
     - OakMine and WheatMine returned anti-bot HTML on the tested URLs.
       Their APIs remain unverified. Authenticated live writes, private
       templates and full datasets were not exercised.

The API inventory distinguishes 331 passing assertion scopes from 129 symbols
with tested departures, across 43 departure groups. It is not a statement that
all 460 APIs behave identically to the original client. New APIs such as resource
profiles, Arrow readers, object prefetch and path-based spawned transforms have
separate regressions; they are not added to the historical inventory denominator.

Dependency and data-type changes
--------------------------------

The reviewed minima are Requests 2.34.2, urllib3 2.8.0 (below 3), Polars 2.0.0,
DuckDB 1.5.6 and PyArrow 26.0.0. Optional extras are orjson 3.13.0, Matplotlib
3.11.2 and PySocks 1.7.1. The ``analytics`` extra remains an empty compatibility
alias because analytical dependencies are core. Dependencies load lazily but
their wheels are still installed by pip. The dependency review records
GitHits release notes, dependency graphs and dated advisory checks.

Typed exports preserve supported primitive attributes, exact integer ranges,
dates, nested supported values and decimals up to precision 38. They are not
universal serialization for arbitrary Python objects. Int128 is normalized to
Decimal where representable; external Int128 Parquet annotations and
timezone-aware nanosecond timestamps are rejected before the DuckDB bridge.
Timezone-aware supported timestamps normalize to UTC/microseconds. Direct
Polars directory parts can retain Enum/Categorical metadata, whereas DuckDB
reads and single-file COPY expose strings. Applications using Polars SQL must
also review Polars 2.0's SQL changes; this package's SQL engine is DuckDB.

Resource profiles expose CPU budgets 16, 8, 6, 4 and 2, with separate stage
allocations. Their memory/CPU controls are advisory for work routed through the
package. Existing external pools, user-created SQL connections, whole-dataframe
materialization and Polars 2.0 spill settings are not hard-limited by that budget.
See :doc:`resources` and :doc:`storage_performance` for boundaries, measured
storage savings and conditional complete-job time estimates.

Upgrading from intermine314 0.1.7
----------------------------------

The original-client compatibility inventory and an upgrade from this package's
own 0.1.7 release are different contracts. The latter already provided Polars,
DuckDB, Parquet and parallel offsets. The new tree expands those facilities and
restores historical InterMine APIs, but also changes some 0.1.7 import paths.
The comparison, reproducible fixtures and Parquet evidence are in
``docs/analysis/evidence/v020-vs017/report.md``.

.. list-table:: Migration checklist for 0.1.7 callers
   :header-rows: 1
   :widths: 38 62

   * - Previous use
     - Migration
   * - ``from intermine314.service import Service``
     - Retain this path for native dictionary defaults. Use ``webservice.Service``
       for the original client's object/row defaults.
   * - ``intermine314.model.class_``, ``model.fields``, ``model.path`` and other
       former model submodules
     - Import public model classes from ``intermine314.model``. Private parser
       helpers and cache counters have no compatibility guarantee.
   * - ``intermine314.registry.mines``
     - Its mine-specific production-plan, preferred-worker and transport-policy
       resolvers have been removed. Configure ``ResourceProfile``,
       ``ParallelOptions`` and transport explicitly. The new
       ``intermine314.registry`` provides historical registry discovery helpers;
       it is not a replacement for the former policy module.
   * - ``intermine314.service.iterators``
     - Use public result types from ``intermine314.results`` or the documented
       ``intermine314.service.session.ResultIterator`` entry point.
   * - ``intermine314.service.tor`` and the ``tor_service`` / ``tor_registry`` /
       ``tor_session`` / ``tor_proxy_url`` helpers
     - Configure ``Service`` / ``Registry`` with explicit ``tor=True`` and
       ``proxy_url="socks5h://127.0.0.1:9050"`` as appropriate; these helpers
       are no longer provided. Install the ``proxy`` extra for SOCKS transport.
   * - Old mine-parallel-preferences / parallel-profiles configuration and their
       ``INTERMINE314_MINE_PARALLEL_PREFERENCES_PATH`` /
       ``INTERMINE314_PARALLEL_PROFILES_PATH`` environment variables
     - Move worker policy to ``ResourceProfile`` and ``ParallelOptions`` as
       described in :doc:`resources`. The old environment variables and loader
       policy helpers no longer configure execution.
   * - Direct ``Query(Model(...))`` expecting dictionaries
     - Pass ``compatibility="native"`` or request ``results(row="dict")``.
       Direct model-backed construction now infers legacy behavior; native
       Service-created queries retain native defaults.
   * - ``len(results)`` or ``if results`` to test for rows
     - Streams are unsized. Materialize once when appropriate, or probe with
       ``next`` and close the stream. Truthiness does not test emptiness.
       Query ``count()`` and ``len(query)`` remain explicit server counts.
   * - Mutating metadata of built-in streamed results or replacing parsers
       during consumption
     - Shared metadata is immutable; rebind to an independent container when
       customizing. Configure parsers before opening the stream. Direct result
       constructors retain mutable metadata.
   * - Python 3.14.0--3.14.4, pinned Polars 1.6 / DuckDB 1.0, or coercible
       non-integer resource settings
     - Upgrade to Python 3.14.5+ and the declared dependency minima. PyArrow is
       now core. Supply actual integer worker/resource values; booleans and
       floats are rejected where integer counts are required.

The archived 14 October 2026 trials (21 pairs per case) compare the published 0.1.7 source with the reviewed
tree on identical modern dependencies, because DuckDB 1.0.0 had no compatible
binary wheel in the tested CPython 3.14 Linux environment. They do not compare
two stock installations. Materializing 10,000 dictionaries took 31% less time
and one request instead of two. Dictionary streaming took 35% more time;
10,000-row numeric single-file and 100,000-row directory exports from custom
dictionary batches took 129% and 75% more time respectively. The directory
fixture used 60% less total space, including its new manifest, and one data
file instead of 100. These writer fixtures exclude the built-in HTTP columnar
fast path; they are not public-mine throughput measurements. No universal
speedup or end-to-end memory reduction versus 0.1.7 is established.

Verification procedure
----------------------

Run the full tests and lint, then build and validate fresh distributions::

    .venv/bin/ruff check --no-cache .
    .venv/bin/python -m pytest
    .venv/bin/python scripts/verify_distribution.py --output /tmp/fresh-verification

The output directory must be empty and outside the checkout. The verifier emits
an ``ok`` manifest after the requested checks succeed. The release gate also
requires all six installation modes and documentation. ``scripts/verify_installed_regressions.py`` additionally runs the
release regressions from an installed wheel environment with source-package
imports excluded from the main process.

Dependency, storage, live-service and installation observations are retained in
``docs/analysis/evidence/``. Use ``python -m benchmarks.evidence DIRECTORY SQL``
to query portable Parquet through DuckDB without a tracked database file.
The original inventory and historical implementation assignments remain at
stable paths in ``docs/analysis/``. The regenerated API audit binds executed
assertions to current controlled inputs. Counts are assertion scopes, rather
than a percentage of complete replacement compatibility.

Publication requires matching project/runtime versions, rebuilt and verified
artifacts, and the matching version tag from a validated master commit. The
release workflow validates the exact tagged commit and artifact hashes before
uploading. Manual workflow runs validate only. Historical review artifacts retain
their original dates, source commits and pre-release version labels.

Declared historical departures
------------------------------

The following 43 groups are the maintained migration guide. Their exact symbol
assignments, pinned upstream references and executed test nodes remain in
``docs/analysis/api-audit-scopes.json``. This guide replaces the earlier separate
compatibility narratives.

explicit-server-list-cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Service destruction/close and its context close owned transport but do not DELETE temporary server lists; those lists can remain on the server.

Avoid network mutation during nondeterministic garbage collection; shared native lifecycle remains explicit.

Use a ListManager context or delete_temporary_lists(); for the internal Service manager call Service.flush() before closing. A Service context alone does not delete server lists.

polars-dataframe
~~~~~~~~~~~~~~~~

dataframe returns polars.DataFrame in both profiles rather than pandas.DataFrame; pandas methods/index semantics do not transfer.

Required lazy Polars -> Parquet -> DuckDB SQL -> Arrow -> Polars pipeline with explicit CSV only.

Use Polars expressions and explicit CSV inputs/exports; analytics extra remains an alias.

nested-logic-repair
~~~~~~~~~~~~~~~~~~~

Nested grouping preserves markers; B OR ((A OR C) AND D) may evaluate differently from the buggy upstream parse.

Preserve intended parentheses and truth values.

Keep explicit parentheses; do not rely on upstream marker loss.

sort-next-repair
~~~~~~~~~~~~~~~~

next(sort_orders) returns the first order without consuming; upstream raises TypeError.

Retain native repaired container behavior.

Use iter(sort_orders) for normal advancing iteration.

model-ownership-and-validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Borrowed streams stay open; invalid schemas raise public errors and inheritance resolves independently of declaration order.

Repair leaks, cleanup crashes and order-dependent model state.

Handle ModelParseError/ModelError and explicitly own borrowed input lifetimes.

composed-ancestry
~~~~~~~~~~~~~~~~~

Composed classes include ancestors of every component, not only the final one.

Repair ancestor loss.

Use the complete parent set for isa decisions.

path-safety
~~~~~~~~~~~

Hash follows string equality; navigating beyond attributes raises PathParseError rather than an internal AttributeError.

Consistent mapping keys and public model errors.

Catch PathParseError and treat paths as stable string-valued keys.

logic-code-range
~~~~~~~~~~~~~~~~

Codes include Z, continue beyond Z and skip reserved codes; invalid/duplicate calls reject before state mutation.

Repair source exhaustion/collision and avoid repeated uploads.

Retain generated codes rather than assuming a 25-code ceiling.

collection-normalization
~~~~~~~~~~~~~~~~~~~~~~~~

Multi/Range/ISA values normalize to string lists rather than preserving input collection type and scalar types.

Retain established native normalization through shared constraints.

Use serialized values as strings; retain original typed inputs separately.

column-select-scope
~~~~~~~~~~~~~~~~~~~

Bound selections clone and retain filters/refinements; reference-relative fields resolve beneath that reference.

Repair lost query state and wrong reference roots.

Create a fresh model/service query explicitly when a fresh filter state is desired.

native-legacy-defaults
~~~~~~~~~~~~~~~~~~~~~~

Native service queries retain string roots/columns and dictionary results/rows; the webservice facade and direct Query(Model) select legacy model/objects/ResultRows.

Preserve existing native callers while restoring the original API through an explicit profile.

Use intermine314.webservice.Service for legacy defaults or select compatibility explicitly.

xml-validation-and-binding
~~~~~~~~~~~~~~~~~~~~~~~~~~

Service-loaded queries remain bound; XML uses canonical pathString, preserves empty values and rejects unknown logic instead of deleting codes.

Repair unexecutable loads and malformed server XML.

Use canonical pathString; correct invalid logic instead of relying on silent repair.

factory-descriptor-validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mixed descriptor roots/refinements are normalized together; conflicting roots and unsupported combinations raise public errors before HTTP.

Avoid silently dropped selections and invalid model paths.

Choose a common compatible root and pass supported descriptors/XML forms.

registration-encoding
~~~~~~~~~~~~~~~~~~~~~

Unicode registration values are real form strings instead of upstream Python 3 bytearray representations; configured transport is preserved.

Repair encoded account data and resource ownership.

Pass Unicode strings and close separately owned returned Services.

deregistration-token
~~~~~~~~~~~~~~~~~~~~

String tokens containing uuid stay strings; validity errors describe implemented 1..86400 seconds.

Repair accidental token indexing and misleading error range.

Supply string tokens or token dictionaries and validity in seconds.

empty-row-string
~~~~~~~~~~~~~~~~

Empty rows stringify safely instead of indexing absent first cells.

Repair empty-row representation crash.

Treat empty row text as a valid representation.

stream-ownership-and-null
~~~~~~~~~~~~~~~~~~~~~~~~~

JSON null is yielded, not an end marker; streams close on completion, explicit close, failures and interruptions.

Repair lost null data and response leaks while retaining shared opener behavior.

Explicitly close early-stopped streams and allow null rows in raw JSON modes.

object-wrapper-repairs
~~~~~~~~~~~~~~~~~~~~~~

Missing class metadata falls back; missing/null/no-ID fields avoid spurious fetches; noncontiguous paths and cycles are preserved safely.

Repair lost object data, id=None requests and recursion.

Expect stable cached None/empty collections and bounded representations.

summary-types
~~~~~~~~~~~~~

Byte and BigDecimal summaries use server numeric statistics and size=0 is forwarded; returned summary numbers remain floats.

Align with server numeric classification and explicit pagination.

Use dataframe/export for exact Decimal analytics rather than float summary values.

list-input-validation
~~~~~~~~~~~~~~~~~~~~~

Generator/readable uploads work without len; iterable identifiers escape quotes and reject line breaks; CSV IDs stay strings and reject null/empty entries.

Repair linewise tokenizer corruption and identifier loss.

Use explicit csv_column and valid nonempty single-line identifiers; borrowed streams stay open.

list-cache-and-name-repair
~~~~~~~~~~~~~~~~~~~~~~~~~~

Caches publish after successful complete refresh; reserved names are skipped and renaming retires old temporary ownership.

Avoid partial cache mutation and accidental deletion/reuse.

Retry failed refresh and use returned names rather than predicting allocations.

temporary-cleanup-retry
~~~~~~~~~~~~~~~~~~~~~~~

Cleanup snapshots names, retires successful deletes individually and preserves the body exception with notes on cleanup failure.

Allow safe retry without mutable-set errors or masking the primary failure.

Retry remaining temporary names; inspect exception notes for cleanup failures.

list-display
~~~~~~~~~~~~

Display reads fields directly and closes its result stream if printing fails.

Avoid parsing punctuation from object string representations and leaking streams.

Do not depend on punctuation parsing side effects.

set-operation-validation
~~~~~~~~~~~~~~~~~~~~~~~~

Uploads project a clone and reject foreign/unbound operands, ambiguous IDs and semicolon list names before mutations.

Avoid modifying query views or accidentally resolving a foreign list as a local name.

Use same-root operands with one entity ID projection and delimiter-safe list names.

enrichment-persistence
~~~~~~~~~~~~~~~~~~~~~~

Historical calls stream EnrichmentLine; explicit output_path materializes typed atomic Parquet/CSV and returns detached Polars.

Add the required analytics pipeline without implicit output.

Request output_path and format explicitly; close early-stopped historical streams.

template-booleans
~~~~~~~~~~~~~~~~~

Python boolean True is editable and invalid optional state rejects before list upload.

Repair source string-only boolean comparison and premature side effects.

Use booleans or canonical XML strings with valid locked/on/off optional state.

template-wire-and-adjustments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inactive constraints are omitted; unsupported active editable loop/subclass/range/ISA and empty multi values reject before HTTP; adjustments are validated and cloned.

Repair broken server forms and unrestricted attribute mutation.

Set unsupported saved constraints editable=False or switch optional constraints off; use supported adjustment fields.

template-xml-wrapper
~~~~~~~~~~~~~~~~~~~~

Serialized templates retain wrapper metadata/editable flags; inherited list operations use actual template endpoints and ID path projection.

Repair metadata loss and Query XML sent to named-template endpoints.

Retain Template type during cloning and use the shared list APIs.

template-discovery-cache
~~~~~~~~~~~~~~~~~~~~~~~~

User templates cache by owner/name; name/object views share an XML snapshot; duplicate names within an owner reject.

Repair top-level cache overwrite, redundant GETs and silent duplicate loss.

Use flush to refresh the snapshot; handle duplicate-name ServiceError.

identifier-null-uid
~~~~~~~~~~~~~~~~~~~

A null UID raises Exception(No uid found), replacing upstream accidental string-plus-dictionary TypeError.

Provide the intended job-construction failure.

Handle failed submissions and inspect server error payloads.

saved-query-empty-sentinel
~~~~~~~~~~~~~~~~~~~~~~~~~~

Equivalent empty saved-queries XML including self-closing/whitespace forms returns the historical missing-query result.

Repair literal-only sentinel detection.

Treat semantically empty saved-query XML as no query.

helper-transport-and-encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Helpers preserve configured managed transport, encode account/query values, close owned state and accept explicit overwrite controls.

Repair raw URL concatenation and unowned clients without changing historical default prompts.

Use explicit overwrite for unattended saved-query updates and supplied clients for transport policy.

plot-tsv-final-row
~~~~~~~~~~~~~~~~~~

The final TSV record is split even without trailing LF; LF still yields the historical final empty-string sentinel.

Repair unsplit/lost final row.

Handle row arrays and optional final sentinel.

plot-token-and-bounds
~~~~~~~~~~~~~~~~~~~~~

Selected token authenticates private enrichment; streams close at five records without reading sixth/footer; malformed data raises ValueError.

Repair missing authentication, excessive consumption and indexing crashes.

Use full enrichment APIs when footer validation/all results are required.

plot-polars-and-errors
~~~~~~~~~~~~~~~~~~~~~~

Lazy Polars replaces pandas/owned NumPy math; invalid input raises ValueError before figures; nonfinite artists remain inspectable but invisible.

Required no-pandas analytics and safe Matplotlib rendering.

Install the plots extra; expect None return and inspect finite chart heights.

registry-https-default
~~~~~~~~~~~~~~~~~~~~~~

The native configured HTTPS registry endpoint replaces the source hard-coded HTTP default; injected registries remain borrowed.

Preserve configured secure transport and deterministic ownership.

Use registry_url for a custom registry and close only resources you own.

public-form-encoding
~~~~~~~~~~~~~~~~~~~~

Text remains str rather than becoming UTF-8 bytes; numeric/None/bool values stringify; list/tuple mapping values normalize element-wise to lists for doseq. Binary values remain bytes.

Retain established managed transport scalar forms and correct repeated form values without silently changing native requests.

Pass the mapping to urlencode(..., doseq=True). Call .encode("utf-8") explicitly when an API requires bytes; preserve typed input data separately.

standard-user-agent-header
~~~~~~~~~~~~~~~~~~~~~~~~~~

headers() emits standard User-Agent instead of the upstream misspelled UserAgent; a configured user_agent overrides the default. Auth/Content-Type/Accept remain optional.

Use the standard HTTP header and shared configurable transport policy.

Read or customize User-Agent/user_agent; update consumers that expected the literal nonstandard UserAgent key.

ancestry-expansion-repair
~~~~~~~~~~~~~~~~~~~~~~~~~

A -> B -> C -> D returns [B,C,D], where upstream iteration over its extended list returned [B,C,D,D]. Diamond branches can still repeat a shared ancestor; cycles raise ModelError.

Avoid re-expanding ancestors accidentally appended to the active traversal while retaining branch order.

Do not count accidental upstream repetitions as distinct ancestors. Deduplicate by class identity/name if set semantics are desired; handle cycle errors.

unsized-result-stream
~~~~~~~~~~~~~~~~~~~~~

len(results) raises TypeError; bool(results) no longer fetches or tests emptiness; list(results) consumes once.

Network iteration must not be repeated as a container length hint.

Use query.count()/len(query) for server counts or explicitly consume the stream; joined row counts can differ from distinct object counts.

streamed-row-metadata
~~~~~~~~~~~~~~~~~~~~~

Streamed built-in rows expose tuple views and read-only shared indexes; direct constructors remain mutable.

Avoid rebuilding invariant metadata and repeated dispatch for every row.

Rebind metadata attributes to mutable copies; install custom classes before opening a stream.

streamed-object-metadata
~~~~~~~~~~~~~~~~~~~~~~~~

Streamed built-in objects expose shared immutable projection metadata; per-object identity/cache and rebinding remain supported.

Avoid rebuilding invariant metadata and repeated dispatch for every row.

Rebind metadata attributes to mutable copies; install custom classes before opening a stream.

stream-open-parser-binding
~~~~~~~~~~~~~~~~~~~~~~~~~~

Parser and result class are bound when each stream opens; later replacement affects subsequent streams.

Avoid rebuilding invariant metadata and repeated dispatch for every row.

Rebind metadata attributes to mutable copies; install custom classes before opening a stream.

