# Dependency review — integration refresh, 2026-10-03

GitHits package metadata and exact-version advisory queries were refreshed at
09:45 UTC, with transitive resolution enabled for six selected releases. All six
had zero active affected direct advisories and no affected advisory occurrences
in the graphs returned. Requests resolved four dependencies, Polars one and
Matplotlib ten; urllib3, DuckDB and PyArrow returned no resolved dependencies.
These are dated registry graph results, not an audit of every optional/local
installed distribution or every version allowed by our open-ended requirements.
Package-wide historical counts (Requests 8, urllib3 22, DuckDB 3, PyArrow 5) are
not current affected counts. Machine-readable decisions and preserved refresh
responses are in `dependency-review.json` and `dependency-evidence/`.

The actual workspace is Python 3.14.5 with Requests 2.34.2, urllib3 2.8.0,
Polars 1.44.2, DuckDB 1.5.6, PyArrow 25.0.1 and Matplotlib 3.11.2, checked through
installed distribution metadata during this task. Installed transport dependencies
also include certifi 2026.7.22, idna 3.20 and charset-normalizer 3.5.2; earlier
05:35 UTC GitHits evidence checked the selected transport graph. Historical
upgrade-review inputs are not claimed to have been installed. Requests 2.34.2
and urllib3 2.8.0 were already installed before this plan (baseline commit
`41c363e`); task 9.1 does not claim a new transport package upgrade.

## Python and transport safeguards

The Python floor is now **>=3.14.5**, preserving package version **0.1.8**.
urllib3's CONNECT host/header backport only applies to older Python 3.11/3.12;
modern Python uses the stdlib implementation. Source comparison shows Python
3.14.0 lacks the new tunnel host/header checks, while 3.14.5 rejects control
characters and invalid header names/values before sending CONNECT. Metadata,
installation guidance and tooling now exclude earlier unpatched 3.14 releases.
Offline tests exercise these actual stdlib rejection paths without network I/O.
Sources: [urllib3 connection implementation](https://github.com/urllib3/urllib3/blob/2.8.0/src/urllib3/connection.py),
[CPython 3.14.0](https://github.com/python/cpython/blob/v3.14.0/Lib/http/client.py),
[CPython 3.14.5 safeguards](https://github.com/python/cpython/blob/v3.14.5/Lib/http/client.py#L974).

The existing `requests>=2.34.2` and `urllib3>=2.8.0,<3` floors retain fixes for
HTTPS-proxy TLS policy ([GHSA-8988-9cw3-xx77](https://github.com/advisories/GHSA-8988-9cw3-xx77)),
unbounded chunk-size line buffering ([GHSA-vxq7-64xx-v4gw](https://github.com/advisories/GHSA-vxq7-64xx-v4gw))
and the chunked Deflate infinite loop ([GHSA-gh4c-6fx4-qh6g](https://github.com/advisories/GHSA-gh4c-6fx4-qh6g)).
All three affect the historical urllib3 2.7.0 comparison and are fixed in 2.8.0.
Existing Requests netrc, TLS, temporary-file and proxy-header fixes remain present.

Release compatibility matters for customized sessions. An explicit proxy SSL
context now governs the proxy's trust/identity/verification; without one, pool
certificate policy and CA values still supply defaults. Destination client
certificates are not sent to the proxy. The native TLS bool/CA-bundle passthrough
is preserved; this package does not construct an explicit proxy SSL context.
Strict RFC3986 host validation, corrected SOCKS credential decoding and
`UnrewindableBodyError` for bodies with `tell` but no `seek` can affect callers.
Empty `Retry.allowed_methods` is deprecated in favor of `None` for any verb;
our default GET/POST/PUT/DELETE tuple is nonempty, and empty/invalid TOML values
normalize back to that tuple. Current configuration therefore does not trigger
that deprecation. See the [urllib3 2.8.0 release](https://github.com/urllib3/urllib3/releases/tag/2.8.0).

Requests 2.34 introduces inline typing and Python 3.15/3.14t support; 2.34.2
restores Mapping for input headers, so code mutating typed Request.headers may
need narrowing. 2.33.1 improves malformed Content-Type handling/error consistency,
and 2.34.1 repairs iterable bodies with custom `__getattr__`. The upgrade tool's
"removed" typing signal is not evidence of a removed runtime API. Source:
[Requests history](https://github.com/psf/requests/blob/main/HISTORY.md).

## Analytics and plotting compatibility

Polars >=1.44.2 and DuckDB >=1.5.6 moved from the analytics extra into core;
PyArrow >=25.0.1 supplies the Arrow bridge. `[analytics]` and `[benchmark]` remain
empty compatibility aliases. Core analytics imports remain lazy, as do optional
Matplotlib >=3.11.2 plots. Neither Pandas nor the original InterMine client is a
base/extra dependency. `dataframe()` intentionally returns Polars in both profiles.

The implemented pipeline is explicit CSV or InterMine input → Polars → Parquet →
DuckDB SQL → Arrow → Polars, with bounded export staging, lossless numeric checks,
model-derived empty schemas and cleanup on errors/interrupts. CSV output requires
an explicit `format="csv"`; CSV input streams remain borrowed. Plotting uses Polars
expressions and has Agg figure evidence in `bar-chart-compatibility.md`.

PyArrow 25.0.1 fixes wrong double values on aarch64 SVE and a thread-related
initial-import crash in 25.0.0. That old release was a comparison input, not an
installed project version needing repair. DuckDB `to_arrow_table()` and
`to_arrow_reader()` provide the bridge without deprecated `fetch_arrow_table()`.
Polars `scan_csv` accepts streams and schema overrides. Other release migration
signals (a deprecated Polars Feather reader, Matplotlib font/text changes) are
not claimed as restored downstream features. Sources:
[PyArrow 25.0.1](https://arrow.apache.org/release/25.0.1.html),
[DuckDB Python stubs](https://github.com/duckdb/duckdb-python/blob/802a345b/_duckdb-stubs/__init__.pyi),
[Polars CSV source](https://github.com/pola-rs/polars/blob/py-1.44.2/py-polars/src/polars/io/csv/functions.py),
[Polars release](https://github.com/pola-rs/polars/releases/tag/py-1.44.2),
[Matplotlib release](https://github.com/matplotlib/matplotlib/releases/tag/v3.11.2).

Owned storage benchmarks now write direct Parquet and read through DuckDB/Arrow/
Polars in bounded batches. Original-client comparison is optional via
`INTERMINE314_REFERENCE_PYTHON`; only its separate environment installs the
original client. Supplied exact Decimal/Int64/string values survive storage;
already-rounded reference decoding cannot be recovered. Reports mark skipped
references honestly and retain historical CSV/Pandas results with their original
schema. No new live benchmark performance result is asserted here.

Executed regression evidence includes `test_benchmark_storage.py`,
`test_benchmark_reference_ownership.py` (original no-close iterator shape and
owned response/session cleanup),
`test_tooling_integration.py`, existing Parquet/CSV/query lifecycle and lazy-import
coverage, plus `scripts.analytics_smoke`. Installed-wheel and fresh-install CI
remain task 9.2; the final all-symbol audit remains task 9.3. This integration is
Git-only: no release, tag, PyPI publication or package version change.
