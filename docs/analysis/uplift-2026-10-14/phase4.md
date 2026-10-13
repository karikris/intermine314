# Phase 4 handoff

Opening receipt: `1a5b75cb67af1e7a0e90e3a735715c1aa71eaf5e`, seven successful
jobs in [CI 38468995634](https://github.com/karikris/intermine314/actions/runs/38468995634).
Tasks: 4.1 `7ce4bfc`, 4.2 `c040b9a`, 4.3 `44f6708`, followed by conversion
overlap/fixture hardening and the eager borrowed-connection commit. Full trace:
**2,811 passed in 34.15s**; Ruff clean; all 460 upstream symbols accounted for.

21 paired process-isolated warm trials against phase 3:

| End-to-end local fixture | Previous ms | Candidate ms |
|---|---:|---:|
| Decode/write/publish 10k HTTP fixture rows to Parquet directory | 14.976 | 13.244 |
| Numeric single-file export, 10k rows | 17.055 | 5.698 |
| String/Int64 single-file export, 10k rows | 17.382 | 17.341 |

Original-baseline comparisons are in `phase4_original_trials.parquet`. These
tables record runtime source digests. Earlier exploratory timings were rerun
after final conversion-overlap accounting; input manifests were regenerated.

Built-in exports replace row dictionaries with validated flat arrays, bounded
column views and the existing typed writer. Custom providers retain their
dictionary route; nested/mixed inference retains its distinct strict fallback.
Page reservations include Python transpose and native conversion overlap until
consumption. This remains estimated admission, not a hard process-RSS limit.

Single-part promotion is deliberately narrow: Zstd, one nonempty row group,
numeric/Boolean/Date primitives, exact Arrow schema. Twelve type cases compare
values/order, Arrow metadata, Polars schema, DuckDB logical types, compression
and row-group geometry against forced COPY. Strings, decimals, nested/extension
types, other codecs, empty files and multiple groups retain COPY. Failed source
status and publication interruptions still preserve existing output.

`query_parquet(connection=...)` reuses the reader's private-cursor ownership
path. Tests preserve caller settings/tables and active readers, reject database
conflicts, close on SQL errors and keep detached eager frames usable afterward.
GitHits Arrow arrays and DuckDB cursor evidence is in `sources.parquet`.
No dependency or version changes. Await exact pushed SHA CI before phase 5.
