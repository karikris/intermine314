# Dependency review — 14 October 2026

GitHits package metadata, depth-two runtime dependency graphs and upgrade reviews
are retained losslessly in [dependencies.parquet](dependencies.parquet). This
review updates dependency floors; the project and runtime version remain 0.1.8.

| Role | Package | Previous minimum | Reviewed minimum | Action |
| --- | --- | --- | --- | --- |
| Core | requests | 2.34.2 | 2.34.2 | Retain |
| Core | urllib3 | 2.8.0, below 3 | 2.8.0, below 3 | Retain |
| Core | polars | 1.44.2 | 2.0.0 | Update; includes polars-runtime-32 2.0.0 |
| Core | duckdb | 1.5.6 | 1.5.6 | Retain |
| Core | pyarrow | 25.0.1 | 26.0.0 | Update |
| Optional speed | orjson | 3.12.0 | 3.13.0 | Update |
| Optional plots | matplotlib | 3.11.2 | 3.11.2 | Retain |
| Optional proxy | PySocks | 1.7.1 | 1.7.1 | Retain; unrelated to the deleted empty file |
| Build | setuptools / wheel | 84.0.0 / 0.48.0 | Same | Retain |
| Development | pytest / ruff | Unbounded | 9.1.1 / 0.17.0 | Set reviewed floors |
| Documentation verifier | sphinx | Tooling only | 9.1.0 | Reviewed; no new runtime dependency |

These were the latest releases reported by GitHits at review time. Its latest-
version advisory count was zero for all 13 packages. Historical advisory entries
are retained and are not interpreted as affecting these versions. The three
upgrade reviews found no affected direct/resolved transitive advisories. This is
a dated registry check, not a security guarantee or an audit of all future
versions allowed by the lower bounds. The upgrade-review provider marked some
deprecation collection incomplete. CI tests both the pinned minimum runtime
set and current resolutions; the source package deliberately has no runtime lock.

## Upgrade effects

- **Polars 2.0:** the package uses DuckDB SQL, so the upstream changes to Polars
  SQL grouping, numeric literals and division are outside its SQL implementation.
  Applications using returned frames with Polars SQL still need to review them.
  Parquet ENUM interpretation and dictionary metadata are not universal type
  round trips: direct Polars directory parts preserve Polars Enum/Categorical
  metadata, while the DuckDB bridge exposes string values. New regression cases
  cover values, nulls and this declared type boundary. InterMine model schemas
  use primitive attributes rather than Enum/Categorical.
- **Memory:** Polars 2.0 enables out-of-core processing with its own defaults.
  Its source defaults the memory fraction to 0.8 and disk budget to 64 GB. The
  package's memory envelope controls its queues, batches and owned DuckDB
  settings, not all Polars allocations or spill. Those upstream controls are
  marked private; we do not turn them into an unsupported package API. Native
  thread initialization remains subject to import order.
- **Arrow 26:** removed deprecated C++ APIs do not require a package extension
  rebuild: intermine314 ships Python code and consumes Arrow wheels. Typed
  conversion, decimal precision, Arrow readers and Parquet schemas are exercised
  by the full suite and clean installation matrix.
- **orjson 3.13:** upstream drops 32-bit Windows wheels. That platform was not
  certified for this package. The optional decoder continues to guard exact
  large integers, including fallback to the standard decoder when required.
- **Python:** requires Python 3.14.5 or later. A lower dependency floor is an
  intentional compatibility change for existing environments; use a fresh
  environment or allow the resolver to upgrade the analytical stack.

Primary sources, retrieved through GitHits:

- [Polars 2.0 release notes](https://github.com/pola-rs/polars/releases/tag/py-2.0.0)
- [Pinned Polars memory defaults and private controls](https://github.com/pola-rs/polars/blob/22a147de3d2bb2e44b97338a2510816c7105c9f2/crates/polars-config/src/lib.rs#L55-L100)
- [Arrow 26 release](https://github.com/apache/arrow/releases/tag/apache-arrow-26.0.0)
- [orjson 3.13 changelog](https://github.com/ijl/orjson/blob/bd00937c5c46ab20eb5d7fe1cfc034b906c4578f/CHANGELOG.md#L4-L15)

The final release audit records the exact test, build and CI results. Earlier
dependency-review files retain their historical dates and versions.
