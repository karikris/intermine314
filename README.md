# intermine314

Python 3.14+ client for InterMine web services.

Modern InterMine client focused on reliable, high-throughput research workflows.

## Ownership and Credit

Copyright (c) 2026 Monash University, Plant Energy and Biotechnology Lab.

Owners:
- Kris Kari
- Dr. Maria Ermakova
- Plant Energy and Biotechnology Lab, Monash University
- Contact: toffe.kari@gmail.com

Original credit:
- Original InterMine team and community contributors.

## License

Licensed under the MIT License (see `LICENSE-LGPL`, which now contains the active MIT license text and notice).

## Requirements

- Python 3.14+
- Core workflow dependencies are required by default: `polars`, `duckdb` (Parquet path).

## Supported Mines

Priority support is focused on:
- MaizeMine
- ThaleMine
- LegumeMine
- OakMine
- WheatMine

WheatMine service endpoint for API clients:

- `https://urgi.versailles.inrae.fr/WheatMine/service` (no trailing slash)

MaizeMine service endpoint for API clients:

- `https://maizemine.rnet.missouri.edu/maizemine/service` (no trailing slash)
- fallback: `http://maizemine.rnet.missouri.edu:8080/maizemine/service`

## Installation

```bash
pip install intermine314
```

Optional extras:

```bash
# Faster JSON decode path
pip install "intermine314[speed]"

# Benchmark script dependencies
pip install "intermine314[benchmark,speed]"
```

Repository: https://github.com/kriskari/intermine314

## Quick Example

```python
from intermine314.webservice import Service

service = Service("https://maizemine.rnet.missouri.edu/maizemine/service")
query = service.new_query("Gene")
query.add_view("Gene.primaryIdentifier", "Gene.symbol", "Gene.length")

parallel_options = {
    "pagination": "auto",
    "profile": "large_query",
    "ordered": "unordered",
    "inflight_limit": 8,
}

for row in query.run_parallel(row="dict", **parallel_options):
    process(row)
```

## Parallel Worker Defaults

`intermine314` uses adaptive defaults when `max_workers` is omitted:

- LegumeMine: `4` workers.
- MaizeMine, ThaleMine, OakMine, WheatMine: `16` workers up to `50,000` rows, then `12`.
- Unknown mines: fallback to `16` workers.

Parallel query APIs default to `pagination="auto"`.
Tune by hardware/network: `4-8` for constrained systems, `16-32` for high-core systems, and lower workers if the mine rate-limits.

Throughput tip: for raw max throughput benchmarking, use `ordered=False` (or `ordered="unordered"`).

Presets: `config/parallel-profiles.toml`. Mine policies: `config/mine-parallel-preferences.toml`.

## Configuration Files

- `config/runtime-defaults.toml`
  - Runtime defaults for omitted query parameters.
  - Override path: `INTERMINE314_RUNTIME_DEFAULTS_PATH=/abs/path/to/runtime-defaults.toml`.
- `config/mine-parallel-preferences.toml`
  - Mine registry, production worker policies, benchmark profile mapping.
  - Shared defaults in `[defaults.mine]`; per-mine overrides in `[mines.<name>]`.
- `config/benchmark-targets.toml`
  - Benchmark endpoints, matrix sizes, targeted export table specs.
  - Shared defaults in `[defaults.target]` and `[defaults.targeted_exports]`.
  - Add custom targets under `[targets.<name>]`.
- `config/parallel-profiles.toml`
  - Parallel profile presets.

## Settable Parameters

### 1) Package Runtime Defaults (`config/runtime-defaults.toml`)

Loaded at import time and used when arguments are omitted:

- `default_parallel_workers`
- `default_parallel_page_size`
- `default_parallel_pagination` (`auto|offset|keyset`)
- `default_parallel_profile` (`default|large_query|unordered|mostly_ordered`)
- `default_parallel_ordered_mode` (`ordered|unordered|window|mostly_ordered`)
- `default_large_query_mode` (`true|false`)
- `default_parallel_prefetch` (integer or `"auto"`)
- `default_parallel_inflight_limit` (integer or `"auto"`)
- `default_order_window_pages`
- `default_keyset_batch_size`
- `keyset_auto_min_size`

### 2) Service Constructor

Set on `Service(...)`:

- `root`
- `username`, `password`
- `token`
- `prefetch_depth`
- `prefetch_id_only`

### 3) Per-call Query Parameters

Set on query calls (`run_parallel`, `iter_batches`, `dataframe`, `to_parquet`, `to_duckdb`):

- `start`, `size`, `page_size`
- `max_workers`
- `ordered`
- `prefetch`
- `inflight_limit`
- `ordered_window_pages`
- `profile`
- `large_query_mode`
- `pagination`
- `keyset_path`
- `keyset_batch_size`
- `batch_size` (batch helpers / exporters)
- `compression` (Parquet: `zstd|snappy|gzip|brotli|lz4|uncompressed`)

## OakMine Large Export Pattern

For OakMine-scale pulls, avoid one wide join across GO/domains/other collections.
Use chunked core + edge exports:

- `core_protein` (entity table)
- `edge_go`
- `edge_domain`

`scripts/benchmarks.py --benchmark-target oakmine` now runs this targeted strategy by default (`--oakmine-targeted-exports`).

## Benchmark Matrix Defaults

`scripts/benchmarks.py` now defaults to a 6-scenario fetch matrix:

- `10k`, `25k`, `50k` with `benchmark_profile_1` (`intermine` + `intermine314` w2-w18)
- `100k`, `250k`, `500k` with `benchmark_profile_2` (`intermine314` w4,w8,w12,w16)

This can be tuned with `--matrix-*` flags or disabled with `--no-matrix-six`.

All targets use `/service` as API root (no trailing slash).

For large MaizeMine retrievals, use the benchmark target preset and template/list-driven core+edge exports:

```bash
python scripts/benchmarks.py --benchmark-target maizemine --workers auto --benchmark-profile auto
```

LegumeMine profile mapping:

- `<= 50k` rows: `benchmark_profile_4` (`intermine` + `intermine314` w4,w6,w8)
- `> 50k` rows: `benchmark_profile_3` (`intermine314` w4,w6,w8)

## Script Slimming and Memory Optimization

Use these replacements for lighter, lower-memory scripts:

1) Skip CSV intermediates unless CSV parity is required.
Heavy path:
- `scripts/bench_fetch.py` CSV export
- `scripts/bench_io.py` `csv_to_parquet(...)`

Lightweight replacement:

```python
query.to_parquet(
    "results.parquet",
    single_file=True,
    parallel=True,
    pagination="auto",
    profile="large_query",
    ordered="unordered",
    inflight_limit=8,
)
```

2) Prefer lazy scans over eager full-file loads.

```python
import polars as pl

out = (
    pl.scan_parquet("results.parquet")
    .select(pl.len().alias("rows"))
    .collect()
)
```

3) Replace wide multi-join views with targeted core + edge tables.

- Keep only required columns in `query.add_view(...)`.
- Use target presets from `config/benchmark-targets.toml`.
- Prefer template/list-driven chunking for OakMine/ThaleMine/WheatMine/MaizeMine.

4) Keep in-flight work bounded.

- Keep `--auto-chunking` enabled.
- Tune `--inflight-limit` and `prefetch` to prevent unbounded memory growth.
- Use `ordered="unordered"` for throughput runs (unless strict order is required).

## Testing

Run unit tests:

```bash
python -m pytest -q tests
```

Run dataframe/parquet compatibility smoke check:

```bash
python setup.py analyticscheck
```

Run live tests (if endpoint/test credentials are available):

```bash
INTERMINE314_RUN_LIVE_TESTS=1 TESTMODEL_URL="https://<mine>/service" python -m pytest -q tests
```

Run via tox (if installed):

```bash
python -m tox -e py314
python -m tox -e py314-analytics
python -m tox -e lint
```

## Notes

Legacy upstream doc/tutorial links are intentionally omitted while this Python 3.14 line is being stabilized.
Published sdist is slimmed to runtime-relevant package/config files (docs/tests/samples/scripts excluded).
