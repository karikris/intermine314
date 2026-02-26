# intermine314

[![CI](https://github.com/karikris/intermine314/actions/workflows/im-build.yml/badge.svg?branch=master)](https://github.com/karikris/intermine314/actions/workflows/im-build.yml)
[![PyPI version](https://img.shields.io/pypi/v/intermine314.svg)](https://pypi.org/project/intermine314/)
[![Python versions supported](https://img.shields.io/pypi/pyversions/intermine314.svg)](https://pypi.org/project/intermine314/)
[![Downloads (Pepy)](https://static.pepy.tech/badge/intermine314)](https://pepy.tech/projects/intermine314)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/karikris/intermine314/blob/master/LICENSE-LGPL)

Modernized InterMine client for Python 3.14+ with query execution, mine-aware parallel fetching, and analytics handoff to Parquet/Polars/DuckDB.

Repository: https://github.com/karikris/intermine314

## Install

```bash
pip install intermine314
```

Optional extras:

```bash
pip install "intermine314[speed]"  # orjson
pip install "intermine314[proxy]"  # PySocks (socks5/socks5h)
```

## Quick start

```python
from intermine314.webservice import Service

service = Service("https://maizemine.rnet.missouri.edu/maizemine/service")
query = service.new_query("Gene")
query.add_view("Gene.primaryIdentifier", "Gene.symbol")

for row in query.rows(size=5):
    print(row)
```

High-level export workflow:

```python
from intermine314 import fetch_from_mine

result = fetch_from_mine(
    mine_url="https://maizemine.rnet.missouri.edu/maizemine",
    root_class="Gene",
    views=["Gene.primaryIdentifier", "Gene.symbol"],
    size=50_000,
    workflow="elt",
    parquet_path="/tmp/genes.parquet",
    max_inflight_bytes_estimate=64 * 1024 * 1024,
    temp_dir="/tmp",
)
```

## Package structure

```text
src/intermine314/
  service/   # Service/Registry, transport, auth, Tor helpers
  query/     # query builder, constraints, path features, serialization
  export/    # fetch_from_mine + parquet/polars/duckdb export helpers
  parallel/  # parallel runner, ordering, pagination
  registry/  # mine and production-profile resolution
  config/    # packaged TOML runtime defaults
  model/     # InterMine data model objects
  lists/     # list/listmanager support
  util/      # shared utility helpers
```

## Repository structure

- `tests/`: unit and integration-style tests.
- `benchmarks/`: benchmark runners, profiles, and benchmark utilities.
- `docs/`: Sphinx documentation source and build output.
- `samples/`: small runnable examples.
- `scripts/`: grouped operational scripts (`dev`, `ci`, `release`, `data`).

## Development

```bash
make test
make lint
make docs
make benchmark BENCHMARK_TARGET=maizemine
```

## License

MIT license (see `LICENSE-LGPL`, which contains the active MIT license text and notice).
