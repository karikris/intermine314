# Benchmarks

Benchmark tooling is fetch-throughput focused with an optional legacy-vs-modern storage compare lane.

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
  --rows 50000 \
  --repetitions 3 \
  --workers auto \
  --transport-modes direct,tor \
  --json-out /tmp/intermine314_benchmark_fetch.json
```

This runner executes fetch-phase benchmarks only:
- intermine314 parallel fetch modes by worker profile
- optional legacy intermine baseline (`--include-legacy-baseline`)
- direct and/or tor transport runs

Storage compare is opt-in:

```bash
python benchmarks/benchmarks.py \
  --mine-url https://bar.utoronto.ca/thalemine/service \
  --rows 5000 \
  --workers auto \
  --transport-modes direct,tor \
  --storage-compare \
  --storage-output-dir /tmp/intermine314_storage_compare \
  --json-out /tmp/intermine314_storage_compare.json
```

`--storage-compare` runs:
- `intermine` legacy CSV export + `pandas.read_csv`
- `intermine314` Parquet export + Polars load + DuckDB scan
- parity checks (row count and deterministic sample hash)

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
