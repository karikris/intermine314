# Benchmarks

Benchmark tooling is matrix-first and always targets a fixed row set:
`5000,10000,25000,50000,100000`.

Benchmark execution is script-only. CI does not run a pytest benchmark-file glob lane.

## Entrypoints

- Live workflow wrapper: `benchmarks/runners/run_live.py`
- Benchmark core entrypoint: `benchmarks/benchmarks.py`
- CI fixed baseline runner: `benchmarks/runners/phase0_ci_fixed_fetch.py`

## Live Run

```bash
python benchmarks/runners/run_live.py \
  --benchmark-target legumemine \
  --workers auto \
  --benchmark-profile auto
```

`run_live.py` performs preflight checks first and emits uniform JSON metrics:
- `elapsed_ms`
- `max_rss_bytes`
- `status`
- `error_type`
- `tor_mode`
- `proxy_url_scheme`
- `profile_name`

## Direct Benchmark Run

```bash
python benchmarks/benchmarks.py \
  --mine-url https://bar.utoronto.ca/thalemine/service \
  --matrix-rows 5000,10000,25000,50000,100000 \
  --repetitions 3 \
  --workers auto \
  --transport-modes direct,tor \
  --storage-output-dir /tmp/intermine314_storage_compare \
  --json-out /tmp/intermine314_storage_compare.json
```

This runner executes:
- optional original `intermine` fetch in an explicitly selected reference environment
  → bounded Polars batches → direct Parquet
- `intermine314` Parquet export → DuckDB SQL → bounded Arrow/Polars batches
- 3 repetitions per row target and transport mode
- direct and tor runs for the intermine314 path

Benchmark profiles are now only:
- `server_restricted` workers: `3,6,9`
- `non_restricted` workers: `4,8,12,16`

## Phase-0 Guardrail Runner

```bash
python benchmarks/runners/phase0_ci_fixed_fetch.py \
  --mine-url https://bar.utoronto.ca/thalemine/service \
  --rows-target 2000 \
  --workers 2 \
  --transport-mode direct \
  --json-out /tmp/intermine314_phase0_ci_fixed_fetch.json
```

This runner is stable and CI-friendly:
- fixed small fetch workload
- import/startup baseline metrics
- throughput and memory envelope point metrics
- tor safety payload when tor mode is selected

## Optional original-client environment

Use Python 3.14.5 or newer for both environments. From the checkout root:

```bash
python3.14 -m venv /tmp/intermine314-reference
/tmp/intermine314-reference/bin/python -m pip install -e '.[proxy]' 'intermine==1.13.0'
export INTERMINE314_REFERENCE_PYTHON=/tmp/intermine314-reference/bin/python
```

Only this optional environment installs the original client. Both fetch and
storage subprocesses honor the selector; they clear inherited PYTHONPATH entries
and disable user-site packages, sharing only the owned benchmark/source paths.
Choose a separate environment as shown: a subprocess alone does not establish
dependency isolation. The original compatibility shims run only on its reference
path. The base package and its benchmark extra do not depend on the original
client or Pandas.

Without the selector, storage comparison records `skipped` for the reference and
null parity, while native export still runs. The CLI automatically omits the original fetch baseline when no interpreter is
selected and reports that skip. Explicit `--legacy-baseline` requires the selector
and fails with setup guidance; `--no-legacy-baseline` disables that fetch lane. A selected environment missing the original package
also produces a skipped storage reference. This preserves an optional original
comparison without claiming it ran when unavailable.

## Storage measurement contract

`parquet_storage_compare_v3` replaces the CSV/Pandas storage procedure. Historical
`legacy_storage_compare_v2` JSON and reports retain their original meaning and
are not comparable as the same workload. No historical timings were rewritten.
Current reference and native artifacts are `.parquet`; no CSV is written by the
owned benchmark. `rows_to_csv` formats integer CLI lists, not output files.

Reference pages are buffered one page at a time. A partially failed page is
retried without committing its rows. Export is staged and published only after
success, preserving prior output on errors/interrupts and closing page iterators.
Large integers, Decimal values and identifier strings supplied by the client
survive storage; this cannot recover precision already lost by the original
client's JSON decoding. The reference may repeat available rows to meet a target,
so differing remote row availability can legitimately fail parity.

`legacy_polars_load` and `modern_polars_load` time a full DuckDB SQL scan through
bounded Arrow/Polars batches, reporting row count and peak **batch** frame memory
(not total process memory). `modern_duckdb_scan` separately times COUNT(*).
Both export records include `columns` resolved by the query: root-relative names
are prefixed and wildcards expand before writing or sampling. This follows
[original Query.add_view](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/query.py#L669).
Samples select each artifact's own resolved columns; `columns_match` compares
those names and their order, returning null for an unavailable reference. A column
name mismatch makes sample parity false without querying nonexistent columns.
Sample hashes compare at most 64 rows sorted by these columns, independent
of parallel completion order. Hashes are textual sample diagnostics, not full
content or schema equality proofs; remote changes between requests also matter.
Live runs are required for performance claims. Offline tests execute real Parquet,
DuckDB and Arrow/Polars operations using fixture fetches and report no live timings.

Reference resource ownership follows the pinned original client's actual shape:
[`ResultIterator.__iter__`](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/results.py#L388)
opens a connection before constructing a JSON reader, and neither iterator has
`close()`. The reference page guard captures that connection even when header
parsing fails before the reader is returned. The owned requests adapter closes
both Response and per-request Session on full reads, exhaustion and errors;
scoped tracking catches resources lost during construction and drops closed
streams immediately, keeping tracking bounded. Cleanup preserves primary errors.
`tests/test_benchmark_reference_ownership.py` executes the no-close iterator shape,
response/session ownership, retry, interrupt, constructor failure and bounded
tracking regressions without installing the original client in the base environment.
