# intermine314 Benchmark Guide

This repository benchmarks three things:

- network fetch behavior (`intermine` vs `intermine314`, worker/page-size profiles)
- storage and engine behavior (Parquet, DuckDB, Polars)
- startup/import and memory guardrails (Phase-0 runners)

`benchmarks/runners/run_live.py` is the main live benchmark entrypoint.

## Install

```bash
pip install "intermine314[benchmark]"
```

Optional extras:

```bash
pip install "intermine314[speed,proxy]"
```

## Canonical Live Run

```bash
python benchmarks/runners/run_live.py \
  --benchmark-target thalemine \
  --workers auto \
  --benchmark-profile auto \
  --repetitions 3 \
  --json-out /tmp/intermine314_benchmark_thalemine.json
```

Notes:

- Run live benchmarks outside restricted sandboxes when possible.
- CI can skip live connectivity by design; use the Phase-0 runners for stable CI artifacts.

## Matrix Defaults

The live runner executes a six-scenario matrix by default:

- small matrix rows: `5k`, `10k`, `25k`
- large matrix rows: `50k`, `100k`, `250k`

Definitions are sourced from:

- `benchmarks/profiles/benchmark-constants.toml`
- `benchmarks/profiles/benchmark-targets.toml`
- `src/intermine314/config/mine-parallel-preferences.toml`

## Benchmark Profiles (Current)

From `mine-parallel-preferences.toml`:

- `benchmark_profile_1`: no legacy baseline, workers `4,8,12,16`
- `benchmark_profile_2`: no legacy baseline, workers `4,6,8`
- `benchmark_profile_3`: includes legacy baseline, workers `4,8,12,16`
- `benchmark_profile_4`: includes legacy baseline, workers `4,6,8`

Default mine mapping:

- most mines: small -> `benchmark_profile_3`, large -> `benchmark_profile_1`
- `legumemine`: small -> `benchmark_profile_4`, large -> `benchmark_profile_2`

## Phase-0 Baseline Guardrails

Use these in CI to keep startup and memory behavior measurable:

```bash
python benchmarks/runners/phase0_guardrails.py --json-out /tmp/intermine314_phase0_guardrails.json
```

Other Phase-0 runners:

- `phase0_baselines.py`
- `phase0_parallel_baselines.py`
- `phase0_model_baselines.py`
- `phase0_ci_fixed_fetch.py`

## Stage Model and Correctness

`benchmarks/benchmarks.py` emits stage-level outputs and parity checks.

Stage outputs include timing and memory for:

1. fetch
2. decode
3. parquet write
4. duckdb scan
5. analytics
6. polars scan

Correctness outputs include:

- schema fingerprint
- row count
- deterministic sample hash
- aggregate/parity checks across comparable engine paths

`--strict-parity` is enabled by default and should remain enabled for comparable scenarios.

## Offline Replay

Use offline replay to remove network variance from post-fetch comparisons:

```bash
python benchmarks/benchmarks.py \
  --offline-replay-stage-io \
  --parquet-compare-path /tmp/intermine314_base.parquet \
  --csv-old-path /tmp/intermine314_old.csv
```

## Output

Benchmark artifacts are written to JSON/HTML and can be published under `docs/benchmarks/results`.

Typical artifact paths:

- `docs/benchmarks/results/runs/<run-id>.json`
- `docs/benchmarks/results/runs/<run-id>.html`
- `docs/benchmarks/results/index.html`
