# intermine314

[![CI](https://github.com/karikris/intermine314/actions/workflows/im-build.yml/badge.svg?branch=master)](https://github.com/karikris/intermine314/actions/workflows/im-build.yml)
[![PyPI version](https://img.shields.io/pypi/v/intermine314.svg)](https://pypi.org/project/intermine314/)
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

Throughput tip: for highest raw throughput, use `ordered=False` (or `ordered="unordered"`).

Presets: `config/parallel-profiles.toml`. Mine policies: `config/mine-parallel-preferences.toml`.

## Configuration Files

- `config/runtime-defaults.toml`
  - Runtime defaults for omitted query parameters.
  - Override path: `INTERMINE314_RUNTIME_DEFAULTS_PATH=/abs/path/to/runtime-defaults.toml`.
- `config/mine-parallel-preferences.toml`
  - Mine registry and production worker policies.
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
Published sdist is slimmed to runtime-relevant package/config files (docs/tests/samples/scripts excluded).
