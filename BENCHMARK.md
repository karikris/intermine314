# intermine314 Benchmark Guide

Benchmark tooling is repository-only and excluded from PyPI artifacts.
Benchmark runs are script-based (`python -m benchmarks...`) and not a pytest benchmark lane.

## Setup

```bash
git clone https://github.com/karikris/intermine314.git
cd intermine314
python -m pip install -e ".[dev,benchmark]"
```

## Scope

Current benchmark scope is:

- transport/fetch behavior (`intermine` vs `intermine314`, direct vs tor)
- worker-scaling and retry behavior
- startup/import and memory guardrails (Phase-0 runners)
- optional storage-engine compare lane (`intermine + CSV + pandas` vs `intermine314 + Parquet + Polars/DuckDB`)

The storage compare lane is opt-in with `--storage-compare` to keep default runs fast.

## Canonical Live Run

```bash
python -m benchmarks.runners.run_live \
  --benchmark-target thalemine \
  --rows 50000 \
  --workers auto \
  --benchmark-profile auto \
  --transport-modes direct,tor \
  --repetitions 3 \
  --json-out /tmp/intermine314_benchmark_thalemine.json
```

## Direct Entry Run

```bash
python -m benchmarks.benchmarks \
  --mine-url https://bar.utoronto.ca/thalemine/service \
  --rows 50000 \
  --workers auto \
  --transport-modes direct,tor \
  --repetitions 3 \
  --json-out /tmp/intermine314_benchmark_fetch.json
```

## Storage Compare (Legacy vs Modern)

```bash
python -m benchmarks.benchmarks \
  --mine-url https://bar.utoronto.ca/thalemine/service \
  --rows 5000 \
  --workers auto \
  --transport-modes direct,tor \
  --storage-compare \
  --storage-output-dir /tmp/intermine314_storage_compare \
  --json-out /tmp/intermine314_storage_compare.json
```

This emits:
- legacy export/load metrics (`intermine` -> CSV -> `pandas.read_csv`)
- modern export/load metrics (`intermine314` -> Parquet -> Polars/DuckDB scan)
- parity checks (`row_count_match`, sample hash match)

## Phase-0 Guardrails

```bash
python -m benchmarks.runners.phase0_guardrails \
  --json-out /tmp/intermine314_phase0_guardrails.json

python -m benchmarks.runners.phase0_contract_baseline \
  --import-repetitions 3 \
  --json-out /tmp/intermine314_phase0_contract_baseline.json
```

Related guardrail runners:

- `benchmarks/runners/phase0_baselines.py`
- `benchmarks/runners/phase0_parallel_baselines.py`
- `benchmarks/runners/phase0_contract_baseline.py`
- `benchmarks/runners/phase0_ci_fixed_fetch.py`
