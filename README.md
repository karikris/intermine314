# intermine314

[![CI](https://github.com/karikris/intermine314/actions/workflows/im-build.yml/badge.svg?branch=master)](https://github.com/karikris/intermine314/actions/workflows/im-build.yml)
[![PyPI version](https://img.shields.io/pypi/v/intermine314.svg)](https://pypi.org/project/intermine314/)
[![Python versions supported](https://img.shields.io/pypi/pyversions/intermine314.svg)](https://pypi.org/project/intermine314/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/karikris/intermine314/blob/master/LICENSE)

Modern InterMine client for Python 3.14.5+ with:

- query execution (`Service` + `Query`)
- parallel export with bounded memory (`ParallelOptions`)
- ELT workflows to Parquet, DuckDB, and Polars (`fetch_from_mine`)
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
pip install "intermine314[analytics]"  # compatibility alias; analytics is already core
pip install "intermine314[plots]"      # optional, lazily loaded Matplotlib
```

## Quick Start

```python
from intermine314.service import Service

service = Service("https://maizemine.rnet.missouri.edu/maizemine/service")
query = service.select("Gene.primaryIdentifier", "Gene.symbol")

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

`export` writes one Parquet file by default. Request CSV explicitly:

```python
query.export("genes.parquet")
query.export("genes.csv", format="csv")
```

A `.csv` suffix alone is rejected; `.parquet` conflicts with `format="csv"`.
`single_file=False` requests partitioned Parquet, while `to_parquet` retains its
directory default. Both formats accept batch, pagination, parallel and temporary
storage controls. `compression` always controls Parquet, including the managed
intermediate for CSV export. CSV output is uncompressed UTF-8 with a header and
standard quoting. `csv_input` and `csv_options` accept local CSV parsing inputs
when query views, constraints, joins and sort order are empty; borrowed streams
stay open. Errors and interrupts preserve existing output and clean managed
temporary data. Empty exports preserve selected column names and Model-derived
types, including precision-aware numeric schemas.

## API Migration Notes

Native imports (``from intermine314.service import Service``) retain dictionary
result defaults. The original-name facade (``from intermine314.webservice import
Service``) defaults to ``compatibility="legacy"``: ``results()`` and iteration return
model objects, and ``rows()`` returns indexed ``ResultRow`` values. Either Service
accepts an explicit compatibility profile. A direct ``Query(Model(...))`` infers
legacy behavior; Service-created queries retain their Service's profile.

Restored names include ``query``/``new_query``/``select``, ``filter``, view aliases,
``order_by``, ``all``, ``size``, ``summarise``/``summarize`` and ``c``. Model/Column
expressions, saved XML, templates, lists, summaries, identifier resolution,
registry helpers and optional plotting use the shared managed transport.
``dataframe()`` returns **Polars in both profiles**, intentionally departing from
the original client's Pandas result. ``results(row="dataframe")`` remains a
stream of dictionaries. See the [behavior reports](docs/analysis/) for tested
contracts and deliberate differences; the final 460-symbol audit is separate.

Historical owned re-exports are restored, including
``from intermine314.webservice import ServiceError``. The namespace and dependency
declaration must still be changed explicitly; this distribution does not install
an ``intermine`` shim. See the [release-readiness and migration guide](docs/source/release_readiness.rst)
for covered areas, remaining limits and release checks. Package-managed mutations
do not automatically retry; supplied sessions retain the caller's retry policy.
The speed extra preserves exact large integers with a guarded JSON decoder.

Polars, DuckDB and PyArrow are core dependencies, loaded when analytics is called.
The pipeline is InterMine or explicit CSV input → Polars → Parquet → DuckDB SQL →
Arrow → Polars. For an existing file or borrowed CSV stream:

```python
from io import StringIO
import polars as pl
from intermine314.export import import_csv, query_parquet

source = StringIO("identifier,value\n0007,12\n")
import_csv(source, "input.parquet", csv_options={"schema_overrides": {"identifier": pl.String}})
frame = query_parquet("input.parquet", "SELECT * FROM results WHERE value > ?", parameters=[10])
```

``fetch_from_mine(csv_input=..., parquet_path=...)`` supports local input without
constructing a Service; combining CSV and remote query arguments is rejected.
List CSV inputs require ``csv_column`` and preserve identifier strings.

Minimal high-level ELT workflow:

```python
from intermine314 import fetch_from_mine

managed_result = fetch_from_mine(
    mine_url="https://maizemine.rnet.missouri.edu/maizemine/service",
    root_class="Gene",
    views=["Gene.primaryIdentifier", "Gene.symbol"],
    parquet_path="/tmp/genes.parquet",
    page_size=2_000,
    max_workers=8,
    inflight_limit=8,
    max_inflight_bytes_estimate=64 * 1024 * 1024,
    managed=True,
)
with managed_result["duckdb_connection"] as con:
    count = con.execute(
        f'SELECT COUNT(*) FROM "{managed_result["duckdb_table"]}"'
    ).fetchone()[0]
    print(count)
```

Python 3.14.5 is the minimum because modern urllib3 relies on the stdlib
CONNECT host/header safeguards shipped there. The [dated dependency review](docs/analysis/dependency-review.md)
records the inspected releases, security scope and compatibility effects.

## Development

```bash
python3.14 -m venv .venv  # interpreter must be 3.14.5 or newer
.venv/bin/python -m pip install -e ".[dev,plots]" sphinx build twine
make PYTHON=.venv/bin/python lint test analyticscheck docs
```

Repository-only support directories:
- `docs/`, `samples/`, and `scripts/` are for development and examples.
- They are intentionally excluded from published package artifacts.

### Test Modes

Default `pytest` runs the offline invariant suite (fast and deterministic):
- Tor strict DNS-safe proxy enforcement (`socks5h://` requirement).
- Streaming response closure on early iterator termination.
- Session ownership lifecycle (`close()` closes only owned resources).
- Executor lifecycle closure under early parallel termination.
- Runtime defaults validation for parallel/query behavior.
- Storage policy single-source checks (Parquet compression + DuckDB identifier validation).
- DuckDB managed connection lifecycle closure.

The committed compatibility suite uses offline fixtures. Live workloads run
through the benchmark scripts and require network access.

Benchmark commands and benchmark-specific docs live in
[`benchmarks/README.md`](benchmarks/README.md).
Benchmarks are runner-script based (`python -m benchmarks...`); benchmark pytest globs are not part of CI or the default test workflow.

## License

MIT for the project; adapted upstream InterMine code uses BSD-2-Clause. See
`LICENSE`, `LICENSE-BSD`, and `NOTICE`.
