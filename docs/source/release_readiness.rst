Release readiness and migration
===============================

This guide describes the development tree reviewed on 14 October 2026 for
0.2.0. The five earlier release findings and the subsequent performance work
are implemented. The repository remains at version 0.1.8 while preparing a
separate version bump. The dated audit and exact verification results are in
``docs/analysis/release-2026-10-14/``.

PyPI's published 0.1.8 comes from the older implementation and does not contain
the restored compatibility described here. Until 0.2.0 is released, evaluate
a checkout in a fresh Python 3.14.5+ environment with ``pip install -e .``.
The current tree is a migration target with declared differences, not a
namespace-compatible or pandas-compatible drop-in replacement.

Replacing the original client
-----------------------------

Install the reviewed checkout (or 0.2.0 after release) and change imports from ``intermine`` to
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

The API inventory distinguishes 333 passing assertion scopes from 127 symbols
with tested departures, across 39 departure groups. It is not a statement that
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

Current dependency, storage, live-service and installation observations are
retained as Parquet under ``docs/analysis/release-2026-10-14/``, with DuckDB views
over storage/timing evidence, and in the regenerated API coverage artifacts.
The ``release-remediation-*.json`` files describe the earlier review. The
original inventory, implementation ledger and historical records remain separate. Passing scope counts
describe executed assertions, rather than a percentage of full replacement
compatibility.

The next release step is a separate change updating project and runtime versions
to 0.2.0, rebuilding and revalidating the resulting artifacts, then pushing the
matching version tag from a validated master commit. Manual workflow runs
validate only.
