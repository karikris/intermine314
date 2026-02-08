# intermine314 Benchmark Protocol

This repository now benchmarks with an explicit worker/page-size matrix and robust statistics.

## Install Benchmark Dependencies

```bash
pip install "intermine314[benchmark,dataframe,speed]"
```

## Default Matrix Flow

By default, `scripts/benchmarks.py` runs a 6-scenario fetch matrix every run:

- first triplet (`benchmark_profile_1`): `10k`, `25k`, `50k`
- second triplet (`benchmark_profile_2`): `100k`, `250k`, `500k`

Tune with:

- `--matrix-six` / `--no-matrix-six`
- `--matrix-small-rows`
- `--matrix-large-rows`
- `--matrix-small-profile`
- `--matrix-large-profile`

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
  --parallel-rows 500000 \
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
python scripts/benchmarks.py --benchmark-target wheatmine --workers auto --benchmark-profile auto
python scripts/benchmarks.py --benchmark-target maizemine --workers auto --benchmark-profile auto
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

Shared constants across target presets:

- profile switch rows: `50,000`
- profile switch: `<=50k -> benchmark_profile_1`, `>50k -> benchmark_profile_2`
- matrix rows: `10k,25k,50k` (small) and `100k,250k,500k` (large)
- recommended repetitions: `3`
- targeted export list chunk size: `10,000`

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

The report stores per-page-size sections:

- matrix mode (default): `fetch_benchmark.matrix6_by_page_size.page_size_<N>`
- compatibility mode: `fetch_benchmark.direct_compare_baseline_by_page_size.page_size_<N>` and `fetch_benchmark.parallel_only_large_by_page_size.page_size_<N>`

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
python scripts/discover_model_paths.py \
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
