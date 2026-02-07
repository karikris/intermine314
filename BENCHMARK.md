# intermine314 Benchmark Protocol

This repository now benchmarks with an explicit worker/page-size matrix and robust statistics.

## What To Report Per Mode

For every mode (`intermine_batched`, `intermine314_wX`), report:

- `mean`
- `median`
- `p90`
- `p95`
- `stddev`
- `trimmed_mean_drop_high_10pct`
- `median_of_means`
- throughput (`rows_per_s`) with the same statistics

The benchmark runner now writes these into JSON output.

## Mode Order Bias Control

Run order is randomized per repetition (`--randomize-mode-order`) so warm-up effects do not always favor the same mode.

## Worker / Page-Size Matrix

Use `--workers` and `--page-sizes` together to run a matrix:

```bash
python scripts/benchmarks.py \
  --mine-url https://maizemine.rnet.missouri.edu/maizemine \
  --baseline-rows 100000 \
  --parallel-rows 1000000 \
  --workers 4,8,12,16 \
  --page-sizes 1000,2500,5000,10000 \
  --repetitions 3 \
  --randomize-mode-order
```

Profile-driven run (workers resolved from mine registry):

```bash
python scripts/benchmarks.py \
  --mine-url https://mines.legumeinfo.org/legumemine \
  --workers auto \
  --benchmark-profile auto
```

Target presets (saved for reuse):

```bash
python scripts/benchmarks.py --benchmark-target thalemine --workers auto --benchmark-profile auto
python scripts/benchmarks.py --benchmark-target oakmine --workers auto --benchmark-profile auto
```

## Benchmark Profiles

Registry-backed benchmark profiles are defined in `config/mine-parallel-preferences.toml`:

- `benchmark_profile_1`: `intermine` baseline + `intermine314` workers `2,4,6,8,10,12,14,16,18`
- `benchmark_profile_2`: `intermine314` workers `4,8,12,16`
- `benchmark_profile_3`: `intermine314` workers `4,6,8`
- `benchmark_profile_4`: `intermine` baseline + `intermine314` workers `4,6,8`

LegumeMine auto rule:

- `<= 50,000` rows: default to worker `4`
- `> 50,000` rows: auto-select `benchmark_profile_3`

## Saved Endpoint/Query Presets

Presets are stored in `config/benchmark-targets.toml`.

- `thalemine`
  - Endpoint: `https://bar.utoronto.ca/thalemine/service`
  - Root: `Gene`
  - Profile switch: `<=50k -> benchmark_profile_1`, `>50k -> benchmark_profile_3`
  - Repetitions: `3`
- `oakmine`
  - Endpoint: `https://urgi.versailles.inrae.fr/OakMine_PM1N/service`
  - Root: `Protein` (OakMine currently returns `0` rows for `Gene`, but non-zero for `Protein`/`DomainMotif`)
  - Profile switch: `<=50k -> benchmark_profile_2`, `>50k -> benchmark_profile_3`
  - Repetitions: `3`

The report stores per-page-size sections:

- `fetch_benchmark.direct_compare_100k_by_page_size.page_size_<N>`
- `fetch_benchmark.parallel_only_1m_by_page_size.page_size_<N>`

## Ordering Guidance

- Max throughput: use `ordered=False` (or `--ordered-mode unordered`).
- Better order with less HOL blocking: use `ordered="window"` / `mostly_ordered` and tune `ordered_window_pages` (typical range: 5-20).
- Strict order: use `ordered=True` / `ordered`.

## Large Query Preset

`large_query_mode=True` (or `profile="large_query"`) sets default prefetch to `2 * workers`.

## In-Flight Cap (Python 3.14)

`inflight_limit` is now separate from `prefetch`:

- ordered mode: passed directly to `executor.map(..., buffersize=inflight_limit)`
- unordered/window modes: used as the max pending task cap

This prevents unbounded task submission and makes backpressure tuning explicit.

## Transport and JSON Notes

- HTTP transport now uses `requests.Session()` for keep-alive connection reuse.
- If `requests` is unavailable, code falls back to `urllib.request` and logs that this path is not a pooled session client.
- JSON decode paths in result streaming use `orjson` when installed, with stdlib `json` fallback.
