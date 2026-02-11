# intermine314 Benchmark Protocol

This repository now benchmarks with an explicit worker/page-size matrix and robust statistics.

Run live benchmark commands with full network access enabled. Restricted/sandbox DNS usually invalidates endpoint timing and availability results.

## Install Benchmark Dependencies

```bash
pip install "intermine314[benchmark,dataframe,speed]"
```

## Default Matrix Flow

By default, `benchmarks/runners/run_live.py` runs a 6-scenario fetch matrix every run:

- first triplet (`benchmark_profile_3`): `5k`, `10k`, `25k`
- second triplet (`benchmark_profile_1`): `50k`, `100k`, `250k`

Matrix row constants are user-editable in `benchmarks/profiles/benchmark-constants.toml`:

- `SMALL_MATRIX_ROWS`
- `LARGE_MATRIX_ROWS`

Tune with:

- `--matrix-six` / `--no-matrix-six`
- `--matrix-small-rows`
- `--matrix-large-rows`
- `--matrix-small-profile`
- `--matrix-large-profile`
- `--matrix-storage-compare` / `--no-matrix-storage-compare`
- `--matrix-load-repetitions`
- `--matrix-storage-dir`

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

Benchmark runs are published to GitHub Pages under `results/` (default local output path: `docs/benchmarks/results`).
Each run writes:

- per-run JSON (`results/runs/<run-id>.json`)
- per-run HTML (`results/runs/<run-id>.html`)
- rolling index (`results/index.html`)

Each matrix scenario also stores CSV vs Parquet comparisons:

- output size comparison (`csv_bytes`, `parquet_bytes`, `reduction_pct`)
- load timing comparison (`csv_load_seconds_pandas`, `parquet_load_seconds_polars`)
- CSV source mode uses `intermine` baseline when present, else lowest worker mode
- Parquet source mode uses highest worker mode

Every benchmark run executes both query benchmark types:

- `simple`: single-table fields only (no explicit joins)
- `complex`: exactly two outer joins across three tables (root + two joined tables)

Both types record the same data points (fetch metrics, storage comparison, dataframe timing).

Every run also executes a third local-engine mode on generated Parquet outputs:

- `join_engines`: two full outer joins written to disk, comparing `duckdb` vs `polars`

## Mode Order Bias Control

Run order is randomized per repetition (`--randomize-mode-order`) so warm-up effects do not always favor the same mode.

## Worker / Page-Size Matrix

Use `--workers` and `--page-sizes` together to run a matrix:

```bash
python benchmarks/runners/run_live.py \
  --mine-url https://maizemine.rnet.missouri.edu/maizemine \
  --baseline-rows 100000 \
  --parallel-rows 500000 \
  --workers 4,8,12,16 \
  --page-sizes 1000,2500,5000,10000 \
  --repetitions 3 \
  --randomize-mode-order
```

Profile-driven run (workers resolved from mine registry):

```bash
python benchmarks/runners/run_live.py \
  --mine-url https://mines.legumeinfo.org/legumemine \
  --workers auto \
  --benchmark-profile auto
```

Target presets (saved for reuse):

```bash
python benchmarks/runners/run_live.py --benchmark-target thalemine --workers auto --benchmark-profile auto
python benchmarks/runners/run_live.py --benchmark-target oakmine --workers auto --benchmark-profile auto
python benchmarks/runners/run_live.py --benchmark-target wheatmine --workers auto --benchmark-profile auto
python benchmarks/runners/run_live.py --benchmark-target maizemine --workers auto --benchmark-profile auto
```

## Benchmark Profiles

Registry-backed benchmark profiles are defined in `intermine314.config/mine-parallel-preferences.toml`:

- `benchmark_profile_1` (large/default): `intermine314` workers `4,8,12,16`
- `benchmark_profile_2` (large/restricted): `intermine314` workers `4,6,8`
- `benchmark_profile_3` (small/default): `intermine` baseline + `intermine314` workers `4,8,12,16`
- `benchmark_profile_4` (small/restricted): `intermine` baseline + `intermine314` workers `4,6,8`

LegumeMine auto rule:

- `<= 50,000` rows: auto-select `benchmark_profile_4`
- `> 50,000` rows: auto-select `benchmark_profile_2`

## Saved Endpoint/Query Presets

Presets are stored in `benchmarks/profiles/benchmark-targets.toml`.

Matrix row constants are stored in `benchmarks/profiles/benchmark-constants.toml`.

Shared constants across target presets:

- profile switch rows: `50,000`
- profile switch: `<=50k -> benchmark_profile_3`, `>50k -> benchmark_profile_1`
- matrix rows: resolved from `benchmarks/profiles/benchmark-constants.toml` via `SMALL_MATRIX_ROWS` and `LARGE_MATRIX_ROWS`
- recommended repetitions: `3`
- targeted export list chunk size: `10,000`

- `legumemine`
  - Endpoint: `https://mines.legumeinfo.org/legumemine/service`
  - Fallback endpoint: `http://mines.legumeinfo.org/legumemine/service`
  - Root: `Gene`
  - Targeted exports: `core_gene`
- `thalemine`
  - Endpoint: `https://bar.utoronto.ca/thalemine/service`
  - Root: `Gene`
  - Targeted exports: `gene_core`, `gene_transcript`, `transcript_cds`, `protein_domain`
- `maizemine`
  - Endpoint: `https://maizemine.rnet.missouri.edu/maizemine/service`
  - Fallback endpoint: `http://maizemine.rnet.missouri.edu:8080/maizemine/service`
  - Root: `Gene`
  - Targeted exports: `core_gene`, `gene_source`, `gene_xref`, `gene_expression`, `gene_go`, `gene_pathway`, `gene_homology`
- `oakmine`
  - Endpoint: `https://urgi.versailles.inra.fr/OakMine_PM1N/service`
  - Fallback endpoint: `https://urgi.versailles.inrae.fr/OakMine_PM1N/service`
  - Root: `Protein`
  - Default benchmark query is a narrow `Protein` core view (no multi-collection wide join)
  - Targeted exports: `core_protein`, `edge_go`, `edge_domain`
- `wheatmine`
  - Endpoint: `https://urgi.versailles.inrae.fr/WheatMine/service`
  - Fallback endpoint: `https://urgi.versailles.inra.fr/WheatMine/service`
  - Root: `Gene`
  - Targeted exports: `core_gene`, `edge_go`, `edge_domain` (template-first + list-chunk fallback)

The report stores per-page-size sections for both query types:

- `query_benchmarks.simple.fetch_benchmark.matrix6_by_page_size.page_size_<N>`
- `query_benchmarks.complex.fetch_benchmark.matrix6_by_page_size.page_size_<N>`
- compatibility mode (when matrix-six disabled):
  - `query_benchmarks.<type>.fetch_benchmark.direct_compare_baseline_by_page_size.page_size_<N>`
  - `query_benchmarks.<type>.fetch_benchmark.parallel_only_large_by_page_size.page_size_<N>`

Per-scenario matrix storage/load details are written under:

- `query_benchmarks.<type>.fetch_benchmark.matrix6_by_page_size.page_size_<N>[i].io_compare`

## OakMine Targeted Export Strategy

OakMine now uses a server-friendly extraction strategy for large pulls:

- canonical ID pass (`Protein.primaryIdentifier`)
- chunked server-side Lists (default `10k` IDs/list)
- small targeted exports per table:
  - `core_protein`
  - `edge_go`
  - `edge_domain`
- optional template-first execution (`--targeted-use-templates-first`) with fallback to custom query exports

This avoids Cartesian blow-ups from one wide query that joins GO + domain paths together.

Useful flags:

- `--oakmine-targeted-exports` / `--no-oakmine-targeted-exports`
- `--targeted-id-chunk-size 10000`
- `--targeted-max-ids <N>`
- `--targeted-template-limit 40`
- `--targeted-output-dir /tmp/intermine314_targeted_exports`

## ThaleMine Notes

Quick API checks:

```bash
BASE='https://bar.utoronto.ca/thalemine/service'
curl -sS "$BASE/version"
curl -sS "$BASE/model"
```

ThaleMine extraction strategy in this repo is now:

- template/list-first for large pulls
- chunked list batches (default `10k` IDs)
- core + edge parquet tables
- avoid a single wide join across GO/domain/expression/homology

Model/path discovery helper:

```bash
python benchmarks/runners/discover_model_paths.py \
  --mine-url https://bar.utoronto.ca/thalemine/service \
  --classes Gene,Transcript,CDS,Protein \
  --json-out /tmp/thalemine_model_paths.json
```

## MaizeMine Notes

Quick API checks:

```bash
BASE='https://maizemine.rnet.missouri.edu/maizemine/service'
curl -sS "$BASE/version"
curl -sS "$BASE/model"
```

MaizeMine extraction strategy in this repo is now:

- template/list-first for bulk pulls
- chunked list batches (default `10k` IDs)
- core + edge parquet tables (join in DuckDB later)
- avoid one wide service join across GO/domains/expression/homology/variation

If the HTTPS endpoint is unavailable, benchmark runner auto-probes configured fallback endpoints.

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

## MaizeMine (Legacy Matrix Snapshot)

The tables below were generated before the matrix defaults were updated to `5k/10k/25k` and `50k/100k/250k`.

### Small Matrix (10,000 rows, Profile 1)

| Mode              | Seconds | Rows/s   | Retries | Baseline          | Speedup | Faster by | Throughput increase |
| :---------------- | ------: | :------- | ------: | :---------------- | :------ | :-------- | :------------------ |
| intermine_batched |   4.111 | 2,432.25 |       0 | intermine_batched | 1.00x   | +0.00%    | +0.00%              |
| intermine314_w4   |   1.674 | 5,972.69 |       0 | intermine_batched | 2.46x   | +59.28%   | +145.56%            |
| intermine314_w6   |   1.690 | 5,915.94 |       0 | intermine_batched | 2.43x   | +58.89%   | +143.23%            |
| intermine314_w8   |   1.687 | 5,927.69 |       0 | intermine_batched | 2.44x   | +58.96%   | +143.71%            |
| intermine314_w10  |   1.657 | 6,034.13 |       0 | intermine_batched | 2.48x   | +59.69%   | +148.09%            |
| intermine314_w12  |   1.683 | 5,940.76 |       0 | intermine_batched | 2.44x   | +59.06%   | +144.25%            |
| intermine314_w14  |   1.679 | 5,955.51 |       0 | intermine_batched | 2.45x   | +59.16%   | +144.86%            |
| intermine314_w16  |   1.798 | 5,561.56 |       0 | intermine_batched | 2.29x   | +56.26%   | +128.66%            |
| intermine314_w18  |   1.678 | 5,958.13 |       0 | intermine_batched | 2.45x   | +59.18%   | +144.96%            |

### Small Matrix (25,000 rows, Profile 1)

| Mode              | Seconds | Rows/s    | Retries | Baseline          | Speedup | Faster by | Throughput increase |
| :---------------- | ------: | :-------- | ------: | :---------------- | :------ | :-------- | :------------------ |
| intermine_batched |   9.567 | 2,613.19  |       0 | intermine_batched | 1.00x   | +0.00%    | +0.00%              |
| intermine314_w4   |   3.573 | 6,997.36  |       0 | intermine_batched | 2.68x   | +62.65%   | +167.77%            |
| intermine314_w6   |   1.864 | 13,409.59 |       0 | intermine_batched | 5.13x   | +80.52%   | +413.15%            |
| intermine314_w8   |   1.887 | 13,246.50 |       0 | intermine_batched | 5.07x   | +80.28%   | +406.91%            |
| intermine314_w10  |   1.904 | 13,127.67 |       0 | intermine_batched | 5.02x   | +80.10%   | +402.37%            |
| intermine314_w12  |   1.752 | 14,269.28 |       0 | intermine_batched | 5.46x   | +81.69%   | +446.05%            |
| intermine314_w14  |   1.764 | 14,169.65 |       0 | intermine_batched | 5.42x   | +81.56%   | +442.24%            |
| intermine314_w16  |   1.749 | 14,291.69 |       0 | intermine_batched | 5.47x   | +81.72%   | +446.91%            |
| intermine314_w18  |   1.867 | 13,391.75 |       0 | intermine_batched | 5.12x   | +80.48%   | +412.80%            |

### Small Matrix (50,000 rows, Profile 1)

| Mode              | Seconds | Rows/s    | Retries | Baseline          | Speedup | Faster by | Throughput increase |
| :---------------- | ------: | :-------- | ------: | :---------------- | :------ | :-------- | :------------------ |
| intermine_batched |  17.279 | 2,893.75  |       0 | intermine_batched | 1.00x   | +0.00%    | +0.00%              |
| intermine314_w4   |   5.292 | 9,447.39  |       0 | intermine_batched | 3.27x   | +69.37%   | +226.48%            |
| intermine314_w6   |   3.677 | 13,596.22 |       0 | intermine_batched | 4.70x   | +78.72%   | +369.82%            |
| intermine314_w8   |   3.892 | 12,846.32 |       0 | intermine_batched | 4.44x   | +77.48%   | +343.90%            |
| intermine314_w10  |  52.680 | 949.13    |       0 | intermine_batched | 0.33x   | -204.88%  | -67.20%             |
| intermine314_w12  |  53.053 | 942.45    |       0 | intermine_batched | 0.33x   | -207.04%  | -67.43%             |
| intermine314_w14  |  75.675 | 740.00    |       1 | intermine_batched | 0.23x   | -337.96%  | -74.43%             |
| intermine314_w16  |  52.516 | 952.10    |       0 | intermine_batched | 0.33x   | -203.93%  | -67.10%             |
| intermine314_w18  |  52.784 | 947.25    |       0 | intermine_batched | 0.33x   | -205.48%  | -67.27%             |

### Large Matrix (100,000 rows, Profile 2)

| Mode             | Seconds | Rows/s    | Retries | Baseline        | Speedup | Faster by | Throughput increase |
| :--------------- | ------: | :-------- | ------: | :-------------- | :------ | :-------- | :------------------ |
| intermine314_w4  |   9.344 | 10,701.63 |       0 | intermine314_w4 | 1.00x   | +0.00%    | +0.00%              |
| intermine314_w8  |   6.511 | 15,359.16 |       0 | intermine314_w4 | 1.44x   | +30.32%   | +43.52%             |
| intermine314_w12 |  66.778 | 1,497.50  |       0 | intermine314_w4 | 0.14x   | -614.48%  | -86.01%             |
| intermine314_w16 |  89.645 | 1,115.51  |       0 | intermine314_w4 | 0.10x   | -859.48%  | -89.58%             |

### Large Matrix (250,000 rows, Profile 2)

| Mode             | Seconds | Rows/s    | Retries | Baseline        | Speedup | Faster by | Throughput increase |
| :--------------- | ------: | :-------- | ------: | :-------------- | :------ | :-------- | :------------------ |
| intermine314_w4  |  29.467 | 8,484.13  |       0 | intermine314_w4 | 1.00x   | +0.00%    | +0.00%              |
| intermine314_w8  |  21.648 | 11,548.32 |       0 | intermine314_w4 | 1.36x   | +26.53%   | +36.12%             |
| intermine314_w12 |  80.822 | 3,093.21  |       0 | intermine314_w4 | 0.36x   | -174.28%  | -63.54%             |
| intermine314_w16 |  97.360 | 2,567.78  |       0 | intermine314_w4 | 0.30x   | -230.40%  | -69.74%             |

### Large Matrix (500,000 rows, Profile 2)

| Mode             | Seconds | Rows/s   | Retries | Baseline        | Speedup | Faster by | Throughput increase |
| :--------------- | ------: | :------- | ------: | :-------------- | :------ | :-------- | :------------------ |
| intermine314_w4  |  68.183 | 7,333.23 |       0 | intermine314_w4 | 1.00x   | +0.00%    | +0.00%              |
| intermine314_w8  |  60.474 | 8,268.01 |       0 | intermine314_w4 | 1.13x   | +11.31%   | +12.75%             |
| intermine314_w12 | 121.757 | 4,106.55 |       0 | intermine314_w4 | 0.56x   | -78.56%   | -44.01%             |
| intermine314_w16 | 133.820 | 3,736.35 |       0 | intermine314_w4 | 0.51x   | -96.26%   | -49.06%             |


## OakMine (Full Matrix Snapshot)

Runs: 3 per mode. Endpoint: `https://urgi.versailles.inra.fr/OakMine_PM1N/service`.

### Simple Query

### Small Matrix (5,000 rows, Profile 2)

| Mode | Seconds | Rows/s | Retries | Baseline | Speedup | Faster by | Throughput increase |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| intermine314_w4 | 2.234 | 2238.64 | 0 | intermine314_w4 | 1x | +0% | +0% |
| intermine314_w8 | 2.237 | 2234.98 | 0 | intermine314_w4 | 1x | -0.16% | -0.16% |
| intermine314_w12 | 2.241 | 2230.63 | 0 | intermine314_w4 | 1x | -0.34% | -0.36% |
| intermine314_w16 | 2.234 | 2238.51 | 0 | intermine314_w4 | 1x | +0% | -0.01% |

Storage summary: CSV `2158610` bytes vs Parquet `256102` bytes, reduction `88.14%`, load mean CSV `0.014s` vs Parquet `0.002s`.

### Small Matrix (10,000 rows, Profile 2)

| Mode | Seconds | Rows/s | Retries | Baseline | Speedup | Faster by | Throughput increase |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| intermine314_w4 | 3.363 | 2973.38 | 0 | intermine314_w4 | 1x | +0% | +0% |
| intermine314_w8 | 4.456 | 2341.92 | 0 | intermine314_w4 | 0.75x | -32.48% | -21.24% |
| intermine314_w12 | 3.783 | 2737.94 | 0 | intermine314_w4 | 0.89x | -12.47% | -7.92% |
| intermine314_w16 | 3.683 | 2744.37 | 0 | intermine314_w4 | 0.91x | -9.49% | -7.7% |

Storage summary: CSV `3255520` bytes vs Parquet `473643` bytes, reduction `85.45%`, load mean CSV `0.022s` vs Parquet `0.003s`.

### Small Matrix (25,000 rows, Profile 2)

| Mode | Seconds | Rows/s | Retries | Baseline | Speedup | Faster by | Throughput increase |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| intermine314_w4 | 7.047 | 3547.86 | 0 | intermine314_w4 | 1x | +0% | +0% |
| intermine314_w8 | 4.667 | 5357.24 | 0 | intermine314_w4 | 1.51x | +33.77% | +51% |
| intermine314_w12 | 4.014 | 6241.94 | 0 | intermine314_w4 | 1.76x | +43.03% | +75.94% |
| intermine314_w16 | 5.143 | 5258.34 | 0 | intermine314_w4 | 1.37x | +27.02% | +48.21% |

Storage summary: CSV `6366191` bytes vs Parquet `1036366` bytes, reduction `83.72%`, load mean CSV `0.047s` vs Parquet `0.005s`.

### Large Matrix (50,000 rows, Profile 2)

| Mode | Seconds | Rows/s | Retries | Baseline | Speedup | Faster by | Throughput increase |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| intermine314_w4 | 15.455 | 3314.53 | 0 | intermine314_w4 | 1x | +0% | +0% |
| intermine314_w8 | 11.439 | 4479.51 | 0 | intermine314_w4 | 1.35x | +25.99% | +35.15% |
| intermine314_w12 | 12.919 | 3891.97 | 1 | intermine314_w4 | 1.2x | +16.41% | +17.42% |
| intermine314_w16 | 13.592 | 3715.8 | 1 | intermine314_w4 | 1.14x | +12.05% | +12.11% |

Storage summary: CSV `12722802` bytes vs Parquet `1288347` bytes, reduction `89.87%`, load mean CSV `0.08s` vs Parquet `0.004s`.

### Large Matrix (100,000 rows, Profile 2)

| Mode | Seconds | Rows/s | Retries | Baseline | Speedup | Faster by | Throughput increase |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| intermine314_w4 | 48.064 | 2107.64 | 0 | intermine314_w4 | 1x | +0% | +0% |
| intermine314_w8 | 25.62 | 3909.4 | 0 | intermine314_w4 | 1.88x | +46.7% | +85.49% |
| intermine314_w12 | 38.852 | 2583.66 | 1 | intermine314_w4 | 1.24x | +19.17% | +22.59% |
| intermine314_w16 | 32.781 | 3060.92 | 1 | intermine314_w4 | 1.47x | +31.8% | +45.23% |

Storage summary: CSV `25407665` bytes vs Parquet `1300589` bytes, reduction `94.88%`, load mean CSV `0.149s` vs Parquet `0.005s`.

### Large Matrix (250,000 rows, Profile 2)

| Mode | Seconds | Rows/s | Retries | Baseline | Speedup | Faster by | Throughput increase |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| intermine314_w4 | 108.886 | 2300.64 | 0 | intermine314_w4 | 1x | +0% | +0% |
| intermine314_w8 | 85.239 | 2950.97 | 0 | intermine314_w4 | 1.28x | +21.72% | +28.27% |
| intermine314_w12 | 108.752 | 2327.85 | 1 | intermine314_w4 | 1x | +0.12% | +1.18% |
| intermine314_w16 | 87.402 | 2902.94 | 1 | intermine314_w4 | 1.25x | +19.73% | +26.18% |

Storage summary: CSV `63533068` bytes vs Parquet `2620928` bytes, reduction `95.87%`, load mean CSV `0.383s` vs Parquet `0.013s`.

### Complex Query

### Small Matrix (5,000 rows, Profile 2)

| Mode | Seconds | Rows/s | Retries | Baseline | Speedup | Faster by | Throughput increase |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| intermine314_w4 | 1.729 | 2892.47 | 0 | intermine314_w4 | 1x | +0% | +0% |
| intermine314_w8 | 1.748 | 2859.99 | 0 | intermine314_w4 | 0.99x | -1.14% | -1.12% |
| intermine314_w12 | 1.929 | 2641.2 | 0 | intermine314_w4 | 0.9x | -11.59% | -8.69% |
| intermine314_w16 | 1.738 | 2876.68 | 0 | intermine314_w4 | 0.99x | -0.54% | -0.55% |

Storage summary: CSV `1063790` bytes vs Parquet `99728` bytes, reduction `90.63%`, load mean CSV `0.009s` vs Parquet `0.002s`.

### Small Matrix (10,000 rows, Profile 2)

| Mode | Seconds | Rows/s | Retries | Baseline | Speedup | Faster by | Throughput increase |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| intermine314_w4 | 3.613 | 2771.32 | 0 | intermine314_w4 | 1x | +0% | +0% |
| intermine314_w8 | 3.522 | 2855.84 | 0 | intermine314_w4 | 1.03x | +2.52% | +3.05% |
| intermine314_w12 | 3.598 | 2784.07 | 0 | intermine314_w4 | 1x | +0.42% | +0.46% |
| intermine314_w16 | 3.62 | 2767.06 | 0 | intermine314_w4 | 1x | -0.18% | -0.15% |

Storage summary: CSV `3419280` bytes vs Parquet `228312` bytes, reduction `93.32%`, load mean CSV `0.018s` vs Parquet `0.002s`.

### Small Matrix (25,000 rows, Profile 2)

| Mode | Seconds | Rows/s | Retries | Baseline | Speedup | Faster by | Throughput increase |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| intermine314_w4 | 7.009 | 3566.65 | 0 | intermine314_w4 | 1x | +0% | +0% |
| intermine314_w8 | 4.933 | 5466.93 | 0 | intermine314_w4 | 1.42x | +29.62% | +53.28% |
| intermine314_w12 | 3.972 | 6297.79 | 0 | intermine314_w4 | 1.76x | +43.33% | +76.57% |
| intermine314_w16 | 4.796 | 5476.32 | 0 | intermine314_w4 | 1.46x | +31.58% | +53.54% |

Storage summary: CSV `9064541` bytes vs Parquet `603977` bytes, reduction `93.34%`, load mean CSV `0.041s` vs Parquet `0.002s`.

### Large Matrix (50,000 rows, Profile 2)

| Mode | Seconds | Rows/s | Retries | Baseline | Speedup | Faster by | Throughput increase |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| intermine314_w4 | 13.09 | 3837.68 | 0 | intermine314_w4 | 1x | +0% | +0% |
| intermine314_w8 | 6.706 | 7461.53 | 0 | intermine314_w4 | 1.95x | +48.77% | +94.43% |
| intermine314_w12 | 14.521 | 3639.55 | 1 | intermine314_w4 | 0.9x | -10.93% | -5.16% |
| intermine314_w16 | 15.474 | 3241.54 | 1 | intermine314_w4 | 0.85x | -18.21% | -15.53% |

Storage summary: CSV `15995526` bytes vs Parquet `1077365` bytes, reduction `93.26%`, load mean CSV `0.076s` vs Parquet `0.004s`.

### Large Matrix (100,000 rows, Profile 2)

| Mode | Seconds | Rows/s | Retries | Baseline | Speedup | Faster by | Throughput increase |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| intermine314_w4 | 36.069 | 2834.77 | 0 | intermine314_w4 | 1x | +0% | +0% |
| intermine314_w8 | 19.555 | 5266.25 | 0 | intermine314_w4 | 1.84x | +45.78% | +85.77% |
| intermine314_w12 | 27.538 | 3635.13 | 1 | intermine314_w4 | 1.31x | +23.65% | +28.23% |
| intermine314_w16 | 19.89 | 5221.57 | 1 | intermine314_w4 | 1.81x | +44.86% | +84.2% |

Storage summary: CSV `32987791` bytes vs Parquet `1064834` bytes, reduction `96.77%`, load mean CSV `0.147s` vs Parquet `0.004s`.

### Large Matrix (250,000 rows, Profile 2)

| Mode | Seconds | Rows/s | Retries | Baseline | Speedup | Faster by | Throughput increase |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| intermine314_w4 | 58.505 | 4434.4 | 0 | intermine314_w4 | 1x | +0% | +0% |
| intermine314_w8 | 47.133 | 5323.49 | 0 | intermine314_w4 | 1.24x | +19.44% | +20.05% |
| intermine314_w12 | 64.362 | 3907.06 | 1 | intermine314_w4 | 0.91x | -10.01% | -11.89% |
| intermine314_w16 | 50.236 | 5307.14 | 1 | intermine314_w4 | 1.16x | +14.13% | +19.68% |

Storage summary: CSV `84190756` bytes vs Parquet `2343201` bytes, reduction `97.22%`, load mean CSV `0.347s` vs Parquet `0.009s`.
