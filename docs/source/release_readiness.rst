Release readiness and migration
===============================

The five findings from the release review have been addressed. The repository
remains at version 0.1.8 while preparing a separate 0.2.0 version bump.

Replacing the original client
-----------------------------

Install ``intermine314`` and change the import namespace from ``intermine`` to
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
     - Python 3.14.5; Linux wheel/sdist installs in base, plots, speed, combined
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

Current evidence is retained in ``docs/analysis/release-remediation-*.json`` and
the regenerated API coverage artifacts. The original inventory, implementation
ledger and historical review records remain separate. Passing scope counts
describe executed assertions, rather than a percentage of full replacement
compatibility.

The next release step is a separate change updating project and runtime versions
to 0.2.0, rebuilding and revalidating the resulting artifacts, then pushing the
matching version tag from a validated master commit. Manual workflow runs
validate only.
