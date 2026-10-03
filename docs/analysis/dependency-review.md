# Dependency review — task 2.1

Reviewed 2026-10-03 using GitHits package information, batched upgrade reviews,
release notes and advisory results. Machine-readable decisions are in
`dependency-review.json`. Historical comparison versions are review inputs,
not versions previously installed in this project.

Polars >=1.44.2 and DuckDB >=1.5.6 move from the analytics extra into core;
PyArrow >=25.0.1 joins core for the columnar bridge. The existing Polars and
DuckDB floors already match the installed packages. Both `[analytics]` and
`[benchmark]` remain empty compatibility aliases. Core analytics libraries
stay lazy at import time. Missing-library diagnostics now recommend the core
install, and PyArrow has matching lazy optional/required helpers.

The restored dataframe API intentionally returns Polars instead of upstream Pandas. Pandas and the original
InterMine client are removed from extras. Owned benchmark Pandas processing
still awaits task 9.1; the reference client belongs in an isolated original-client
benchmark environment. No claim is made that all benchmark Pandas code is gone.
Matplotlib >=3.11.2 is optional through `[plots]`; plotting restoration and lazy
plotting behavior remain task 8.3. Requests, urllib3, speed/proxy/dev requirements
and project version 0.1.8 are preserved.

Reviewed targets had zero active direct advisories. Polars/DuckDB/PyArrow and
Matplotlib historical comparisons also reported zero affected transitive
packages and no introduced advisories. These are dated checks, not guarantees
for future dependency resolutions. No security repair is claimed for an
uninstalled old version. PyArrow 25.0.1 was selected because its release notes
fix incorrect double values on ARM SVE and a thread-related initial-import
crash in 25.0.0. DuckDB supports `to_arrow_table()`; the deprecated
`fetch_arrow_table()` is unnecessary. Polars `scan_csv` accepts streams and
schema overrides for later CSV tasks. Polars review mentions migration and
deprecation signals, including removal of a deprecated Feather reader;
Matplotlib 3.11 changes font/text handling. Those downstream paths are not
claimed restored here.

Validation refreshes editable package metadata on Python 3.14 and exercises
DuckDB -> Arrow -> Polars with an integer above 2**53, nulls and an exact decimal.
Import checks block heavy analytics, Pandas and the original client. This
validates dependency interoperability; the schema-correct query/export pipeline
remains pending tasks 2.2 and 2.3.

Original upstream code is dual LGPL-3.0/BSD-2-Clause according to its
[README](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/README.md).
We select BSD-2-Clause for adaptations, reproduce the pinned upstream
[LICENSE-BSD](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/LICENSE-BSD)
verbatim and ship NOTICE alongside the existing MIT license. Package metadata
expresses `MIT AND BSD-2-Clause`; original MIT authors remain intact.

Sources:

- [Polars 1.44.2](https://github.com/pola-rs/polars/releases/tag/py-1.44.2)
- [Polars CSV API source](https://github.com/pola-rs/polars/blob/py-1.44.2/py-polars/src/polars/io/csv/functions.py)
- [DuckDB 1.5.6](https://github.com/duckdb/duckdb/releases/tag/v1.5.6)
- [DuckDB Arrow API source](https://github.com/duckdb/duckdb-python/blob/802a345b/_duckdb-stubs/__init__.pyi)
- [PyArrow 25.0.1](https://arrow.apache.org/release/25.0.1.html)
- [Matplotlib 3.11.2](https://github.com/matplotlib/matplotlib/releases/tag/v3.11.2)
