# Benchmarks

Benchmark tooling is matrix-first and always targets a fixed row set:
`5000,10000,25000,50000,100000`.

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
- `intermine` legacy CSV export + `pandas.read_csv`
- `intermine314` Parquet export + DuckDB scan + Polars load
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
