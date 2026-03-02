# intermine314

[![CI](https://github.com/karikris/intermine314/actions/workflows/im-build.yml/badge.svg?branch=master)](https://github.com/karikris/intermine314/actions/workflows/im-build.yml)
[![PyPI version](https://img.shields.io/pypi/v/intermine314.svg)](https://pypi.org/project/intermine314/)
[![Python versions supported](https://img.shields.io/pypi/pyversions/intermine314.svg)](https://pypi.org/project/intermine314/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/karikris/intermine314/blob/master/LICENSE-LGPL)

Modern InterMine client for Python 3.14+ with:

- query execution (`Service` + `Query`)
- parallel export with bounded memory (`ParallelOptions`)
- ELT/ETL workflows to Parquet, DuckDB, and Polars (`fetch_from_mine`)
- mine-aware profile resolution for workers and resource limits
- Tor-safe transport defaults (`socks5h://` policy in strict Tor mode)

Repository: https://github.com/karikris/intermine314

## Install

```bash
pip install intermine314
```

Optional extras:

```bash
pip install "intermine314[speed]"   # orjson
pip install "intermine314[proxy]"   # PySocks
pip install "intermine314[benchmark]"
```

## Quick Start

```python
from intermine314.service import Service

service = Service("https://maizemine.rnet.missouri.edu/maizemine/service")
query = service.new_query("Gene")
query.add_view("Gene.primaryIdentifier", "Gene.symbol")

for row in query.rows(size=5):
    print(row)
```

Parallel export uses `ParallelOptions` only:

```python
from intermine314.query.builder import ParallelOptions

query.to_parquet(
    "/tmp/genes_parts",
    batch_size=5000,
    parallel_options=ParallelOptions(
        max_workers=8,
        profile="large_query",
        ordered="unordered",
        inflight_limit=8,
        max_inflight_bytes_estimate=64 * 1024 * 1024,
    ),
)
```

Mine-aware high-level workflow:

```python
from intermine314 import fetch_from_mine

result = fetch_from_mine(
    mine_url="https://maizemine.rnet.missouri.edu/maizemine/service",
    root_class="Gene",
    views=["Gene.primaryIdentifier", "Gene.symbol"],
    size=50_000,
    workflow="elt",
    parquet_path="/tmp/genes.parquet",
    resource_profile="tor_low_mem",
    temp_dir="/tmp",
)

with fetch_from_mine(
    mine_url="https://maizemine.rnet.missouri.edu/maizemine/service",
    root_class="Gene",
    views=["Gene.primaryIdentifier", "Gene.symbol"],
    size=50_000,
    workflow="elt",
    parquet_path="/tmp/genes.parquet",
    managed=True,
) as managed_result:
    count = managed_result["duckdb_connection"].execute(
        f'SELECT COUNT(*) FROM "{managed_result["duckdb_table"]}"'
    ).fetchone()[0]
    print(count)
```

## Benchmarks

Benchmark guidance and commands live in [`BENCHMARK.md`](BENCHMARK.md).

## Development

```bash
make lint
make test
make docs
```

## License

MIT (see `LICENSE-LGPL`).
