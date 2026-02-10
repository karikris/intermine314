# intermine314

[![CI](https://github.com/karikris/intermine314/actions/workflows/im-build.yml/badge.svg?branch=master)](https://github.com/karikris/intermine314/actions/workflows/im-build.yml)
[![PyPI version](https://img.shields.io/pypi/v/intermine314.svg)](https://pypi.org/project/intermine314/)
[![Python versions supported](https://img.shields.io/pypi/pyversions/intermine314.svg)](https://pypi.org/project/intermine314/)
[![Downloads (Pepy)](https://static.pepy.tech/badge/intermine314)](https://pepy.tech/projects/intermine314)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/karikris/intermine314/badge)](https://securityscorecards.dev/viewer/?uri=github.com/karikris/intermine314)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/karikris/intermine314/blob/master/LICENSE-LGPL)

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
```

Repository: https://github.com/karikris/intermine314

## Quick Example

```python
from intermine314 import fetch_from_mine

result = fetch_from_mine(
    mine_url="https://maizemine.rnet.missouri.edu/maizemine/service",
    root_class="Gene",
    views=["Gene.primaryIdentifier", "Gene.symbol", "Gene.length"],
    joins=[],
    size=50_000,
    workflow="elt",                  # elt | etl
    production_profile="auto",       # mine-aware profile resolution
    parquet_path="/tmp/maize_genes.parquet",
    duckdb_table="genes",
)

duckdb_con = result["duckdb_connection"]
print(duckdb_con.execute("select count(*) from genes").fetchall())
```

## Parallel Worker Defaults

`intermine314` uses adaptive defaults when `max_workers` is omitted:

- LegumeMine: `4` workers.
- MaizeMine: `8` workers.
- ThaleMine, OakMine, WheatMine: `16` workers.
- Unknown mines: fallback to `16` workers.

Production workflows use six named profiles:

- ELT: `elt_default_w4`, `elt_server_limited_w8`, `elt_full_w16`
- ETL: `etl_default_w4`, `etl_server_limited_w8`, `etl_full_w16`

Parallel query APIs default to `pagination="auto"`.
Tune by hardware/network: `4-8` for constrained systems, `16-32` for high-core systems, and lower workers if the mine rate-limits.

Throughput tip: for highest raw throughput, use `ordered=False` (or `ordered="unordered"`).

Presets: `config/parallel-profiles.toml`. Mine policies: `config/mine-parallel-preferences.toml`.

## Configuration Files

- `config/runtime-defaults.toml`
  - Runtime defaults for omitted query parameters.
  - Override path: `INTERMINE314_RUNTIME_DEFAULTS_PATH=/abs/path/to/runtime-defaults.toml`.
- `config/mine-parallel-preferences.toml`
  - Mine registry, production profile policies, and benchmark profile policies.
  - Shared defaults in `[defaults.mine]`; per-mine overrides in `[mines.<name>]`.
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

### 4) Mine Profile Parameters (`config/mine-parallel-preferences.toml`)

- `production_profile_switch_rows`
- `production_elt_small_profile`
- `production_elt_large_profile`
- `production_etl_small_profile`
- `production_etl_large_profile`
- `benchmark_small_profile`
- `benchmark_large_profile`

Benchmark and performance-optimization workflows are documented in `BENCHMARK.md`.

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
Published sdist is slimmed to runtime-relevant package/config files (docs/tests/samples/benchmarking excluded).
