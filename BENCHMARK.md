# intermine314 Benchmark Guide

Benchmark tooling is repository-only.
The `benchmarks/` tree is intentionally excluded from published PyPI artifacts.

## Setup

Use a source checkout:

```bash
git clone https://github.com/karikris/intermine314.git
cd intermine314
python -m pip install -e ".[dev,benchmark]"
```

## Scope

The benchmark suite measures:

- transport and fetch behavior (`intermine` vs `intermine314`, direct vs Tor)
- storage/engine behavior (`pandas+csv` vs `parquet+duckdb` vs `parquet+polars`)
- startup/import and memory guardrails (Phase-0 runners)

## Canonical Matrix

Rows:

- `5000`
- `10000`
- `25000`
- `50000`
- `100000`

Worker tiers:

- server-restricted mines: `3,6,9`
- unrestricted mines: `4,8,12,16`

These defaults are sourced from:

- `benchmarks/profiles/benchmark-constants.toml`
- `benchmarks/profiles/benchmark-targets.toml`
- `src/intermine314/config/mine-parallel-preferences.toml`

## Canonical Live Run

```bash
python -m benchmarks.runners.run_live \
  --benchmark-target thalemine \
  --workers auto \
  --benchmark-profile auto \
  --matrix-rows 5000,10000,25000,50000,100000 \
  --transport-modes direct,tor \
  --repetitions 3 \
  --json-out /tmp/intermine314_benchmark_thalemine.json
```

Notes:

- Prefer running live benchmarks outside restricted sandboxes.
- Keep `--strict-parity` enabled for comparable scenarios.

## Phase-0 Guardrails

Use Phase-0 runners for stable CI artifact baselines:

```bash
python -m benchmarks.runners.phase0_guardrails \
  --json-out /tmp/intermine314_phase0_guardrails.json
```

Related runners:

- `benchmarks/runners/phase0_baselines.py`
- `benchmarks/runners/phase0_parallel_baselines.py`
- `benchmarks/runners/phase0_model_baselines.py`
- `benchmarks/runners/phase0_ci_fixed_fetch.py`

## Stage Model and Parity Outputs

`benchmarks/benchmarks.py` emits stage timings and correctness parity outputs:

1. fetch
2. decode
3. parquet write
4. duckdb scan
5. analytics
6. polars scan

Parity outputs include:

- schema fingerprint
- row count
- deterministic sample hash
- aggregate/parity checks

## Offline Replay

To remove internet variance for post-fetch comparisons:

```bash
python -m benchmarks.benchmarks \
  --offline-replay-stage-io \
  --parquet-compare-path /tmp/intermine314_base.parquet \
  --csv-old-path /tmp/intermine314_old.csv
```

## Artifacts

Typical output paths:

- `docs/benchmarks/results/runs/<run-id>.json`
- `docs/benchmarks/results/runs/<run-id>.html`
- `docs/benchmarks/results/index.html`
