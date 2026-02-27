# Benchmarks

This directory is the modern top-level home for benchmark runners, benchmark cases,
and benchmark profile inputs.

Current implementation status:
- The main benchmark implementation lives directly in `benchmarks/`.
- `benchmarks/runners/run_live.py` is the canonical entrypoint and delegates to
  `benchmarks/benchmarks.py`.

Run:

```bash
python benchmarks/runners/run_live.py --benchmark-target legumemine --workers auto --benchmark-profile auto
```

Phase-0 baseline capture:

```bash
python benchmarks/runners/phase0_baselines.py \
  --benchmark-target maizemine \
  --mode both \
  --workflow elt \
  --rows-target 100000 \
  --json-out /tmp/intermine314_phase0.json
```

This records import latency, peak RSS, rows/sec, and intermine314 log-volume
for direct and Tor modes. If a mode is not reachable in the current
environment, it is marked as `skipped` with a reason.

Phase-0 parallel policy baseline capture:

```bash
python benchmarks/runners/phase0_parallel_baselines.py \
  --modes ordered,unordered,window \
  --rows-target 50000 \
  --page-size 1000 \
  --max-workers 4 \
  --import-repetitions 5 \
  --json-out /tmp/intermine314_phase0_parallel.json
```

This captures:
- cold import latency for `intermine314.parallel.policy` and `intermine314.query.builder`
- synthetic rows/sec and peak RSS by order mode
- structured log volume plus observability probes:
  - `parallel_export_start` and `parallel_export_done` paired once per run
  - `parallel_ordered_scheduler_stats` emitted in ordered mode and at `DEBUG` level

Phase-0 model layer baseline capture:

```bash
python benchmarks/runners/phase0_model_baselines.py \
  --kinds both \
  --object-count 50000 \
  --metric-goal throughput \
  --sample-mode head \
  --sample-size 64 \
  --import-repetitions 5 \
  --json-out /tmp/intermine314_phase0_model.json
```

This captures:
- cold import latency for `intermine314.model`
- construction throughput for `Path` and `Column` objects
- tracemalloc peak bytes plus best-effort peak RSS snapshots

Benchmark semantics for model baselines are explicit:
- `--metric-goal throughput` (default): do not retain all generated objects.
- `--metric-goal memory`: retain all generated objects by default.
- `--retain-objects/--no-retain-objects`: explicit override.
- `--sample-mode {none,head,stride}` + `--sample-size`: bounded sample retention when full retention is disabled.

### Stage Decomposition and Parity

`benchmarks/benchmarks.py` now emits first-class stage timings and parity checks:
- stage timings across fetch/decode, parquet write, dataframe scan, and join engine runs
- CSV vs Parquet parity (schema + row-count + sampled row hash)
- DuckDB vs Polars join output parity (schema + row-count + sampled row hash)

Stage model schema:
- `schema_version`: `benchmark_stage_model_v1`
- `scenario_name`
- `stages.fetch`: elapsed/cpu, retries, rows_fetched, bytes_out
- `stages.decode`: elapsed/cpu, rows_fetched
- `stages.parquet_write`: elapsed, parquet_bytes
- `stages.duckdb_scan`: elapsed/cpu, peak_rss_bytes
- `stages.analytics`: elapsed/cpu, peak_rss_bytes, result_hash
- `stages.polars_scan`: elapsed/cpu, peak_rss_bytes

`bytes_in/bytes_out` currently represent decoded payload bytes observed at the benchmark layer
(for example CSV materialized bytes), not raw transport socket bytes.

Correctness invariants (written per comparable scenario):
- `schema_fingerprint`: column name/type/nullability/ordering fingerprint
- `row_count`
- `groupby_count` on a stable identifier when available
- `sample_hash` from deterministic sample rows (`sort_key` + fixed seed)

When `--strict-parity` is enabled (default), benchmark runs fail if these invariants mismatch
across engine comparisons or across old/new comparable pipelines.

Join-engine comparisons are deconfounded by design:
- one canonical pre-materialization stage builds `base/edge_one/edge_two` parquet inputs from the source parquet
- DuckDB and Polars run the same full-outer-join shape over that identical canonical input
- output rows are sorted by the same join key before parity checks
- engine delta timings exclude canonical input preparation

Engine pipeline comparisons are strict and symmetric:
- `ELT-DuckDB pipeline`: shared parquet input -> DuckDB join/aggregate -> parquet output
- `ETL-Polars pipeline`: same shared parquet input -> Polars join/aggregate -> parquet output
- both scenarios consume the same parquet artifact for the engine step and emit parity checks

New controls:
- `--parity-sample-mode {head,stride}`
- `--parity-sample-size <N>`
- `--strict-parity` (enabled by default) to fail the run on parity mismatches

### Repeatability / Replay

Use offline replay to avoid network variance during storage/engine comparisons:

```bash
python benchmarks/benchmarks.py \
  --offline-replay-stage-io \
  --csv-old-path /tmp/intermine314_benchmark_100k_intermine.csv \
  --parquet-compare-path /tmp/intermine314_benchmark_100k_intermine314.parquet \
  --csv-new-path /tmp/intermine314_benchmark_large_intermine314.csv \
  --parquet-new-path /tmp/intermine314_benchmark_large_intermine314.parquet
```

In replay mode, the IO stage reuses artifacts instead of exporting from the mine again.
This also applies to matrix storage compare scenarios when those artifacts already exist.

Row-stream replay option:
- during online runs, benchmarks persist a normalized row-stream artifact (`jsonl`) under
  `--row-stream-artifact-dir` (default: `/tmp/intermine314_row_stream_artifacts`)
- during offline replay, if CSV/Parquet artifacts are missing but row-stream exists,
  benchmarks materialize CSV/Parquet from row-stream and then execute decode/parquet/duckdb/polars stages
- this keeps network fetch as an optional dimension instead of a mandatory confounder

### Framework Integration

- `psutil`: optional process telemetry snapshots in benchmark reports.
- `asv`: baseline config in `benchmarks/asv.conf.json` and microbenches in `benchmarks/asv_bench/`.
- `pyperf`: command/comparison helper in `benchmarks/pyperf_compare.py`.

## Live Runner Behavior

`benchmarks/runners/run_live.py` now treats live connectivity checks as optional
and self-diagnosing:

- Exit code `0`: benchmark completed successfully.
- Exit code `2`: live preflight skipped due to environment constraints
  (for example DNS/network/proxy unavailable).
- Exit code `1`: real benchmark failure after preflight passed.

Preflight diagnostics log compact fields only:

- `mode` (`direct` / `tor`)
- `host`
- `reason` (`ok` / `dns_failed` / `connect_failed` / `proxy_failed`)
- `elapsed_s`
- `err_type`

No query payloads, list IDs, or full responses are logged.

### CI and Sandbox Defaults

- In CI, live runs are skipped by default unless `RUN_LIVE=1` is set.
- In sandboxed runners, DNS and outbound networking may be blocked by policy.
  In that case preflight exits with code `2` rather than signaling a regression.
- Tor-mode checks require a working SOCKS proxy and use `socks5h://` to keep DNS
  resolution inside Tor.

### Tor Overhead Expectations

Tor routing usually adds latency and retry variance versus direct mode. Compare
success/failure and bounded retries, not strict timing thresholds.
