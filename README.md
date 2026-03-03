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

## API Migration Notes

Compatibility aliases were removed to keep the runtime API minimal and explicit:

- `service.query(...)` -> use `service.select(...)` or `service.new_query(...)`
- `Service.get_mine_info(...)` -> use `Registry(...).info(...)`
- `Service.get_all_mines(...)` -> use `Registry(...).all_mines(...)`
- Query aliases removed: `filter`, `add_column*`, `add_views`, `order_by`, `all`, `size`, `summarize`, `c`
  Use canonical `Query` methods (`where`, `add_view`, `add_sort_order`, `get_results_list`, `count`, `summarise`, `column`).

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

## Benchmarks (Source Checkout Only)

Benchmark scripts live in `benchmarks/` and are not shipped in the PyPI wheel/sdist.
Use a source checkout for benchmark runs:

```bash
git clone https://github.com/karikris/intermine314.git
cd intermine314
python -m pip install -e ".[dev,benchmark]"
python -m benchmarks.runners.run_live --help
```

The benchmark suite supports both:
- legacy baseline workflow (`intermine + CSV + pandas`)
- modern workflow (`intermine314 + Parquet + Polars/DuckDB`)

Benchmark workflow details live in [`BENCHMARK.md`](BENCHMARK.md).

## Development

```bash
make lint
make test
make docs
```

### Test Modes

Default `pytest` runs a lean offline invariant suite (fast and deterministic).

Lean suite invariants are intentionally limited to:
- Tor strict DNS-safe proxy enforcement (`socks5h://` requirement).
- Streaming response closure on early iterator termination.
- Session ownership lifecycle (`close()` closes only owned resources).
- Executor lifecycle closure under early parallel termination.
- Resource profile and runtime defaults validation.
- Storage policy single-source checks (Parquet compression + DuckDB identifier validation).
- DuckDB managed connection lifecycle closure.

Run the full offline suite:

```bash
INTERMINE314_RUN_FULL_TESTS=1 INTERMINE314_TEST_DISABLE_NETWORK=1 pytest -q
```

Run benchmark tests explicitly:

```bash
INTERMINE314_RUN_BENCHMARK_TESTS=1 INTERMINE314_TEST_DISABLE_NETWORK=1 pytest -q tests/test_benchmarking_*
```

Run live network smoke tests explicitly:

```bash
INTERMINE314_RUN_LIVE_TESTS=1 pytest -q tests/live_*.py
```

Run Tor live smoke test:

```bash
INTERMINE314_RUN_LIVE_TESTS=1 INTERMINE314_RUN_TOR_LIVE_TESTS=1 pytest -q tests/live_tor.py
```

## License

MIT (see `LICENSE-LGPL`).
