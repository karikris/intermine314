# intermine314

[![CI](https://github.com/karikris/intermine314/actions/workflows/im-build.yml/badge.svg?branch=master)](https://github.com/karikris/intermine314/actions/workflows/im-build.yml)
[![PyPI version](https://img.shields.io/pypi/v/intermine314.svg)](https://pypi.org/project/intermine314/)
[![Python versions supported](https://img.shields.io/pypi/pyversions/intermine314.svg)](https://pypi.org/project/intermine314/)
[![License: MIT and BSD-2-Clause](https://img.shields.io/badge/License-MIT%20AND%20BSD--2--Clause-yellow.svg)](https://github.com/karikris/intermine314/blob/master/NOTICE)

Modern InterMine client for Python 3.14.5+ with:

- query execution (`Service` + `Query`)
- parallel export with bounded memory (`ParallelOptions`)
- ELT workflows to Parquet, DuckDB, and Polars (`fetch_from_mine`)
- Tor-safe transport defaults (`socks5h://` policy in strict Tor mode)

Repository: https://github.com/karikris/intermine314

## Install

**0.2.1** restores more original InterMine behavior, including optional pandas
output, result sizing and live parser customization, around the shared
Parquet, DuckDB, Arrow and Polars pipeline. Use Python **3.14.5+**. See the
[release notes](https://github.com/karikris/intermine314/blob/v0.2.1/CHANGELOG.md)
for improvements, migration requirements and measured performance trade-offs.

```bash
pip install intermine314==0.2.1 "duckdb>=1.5.6,<2"
```

DuckDB 2.0 changes result and Arrow APIs used by this release. The published
0.2.1 metadata predates that release, so include the DuckDB constraint above.
Source installations declare the same upper bound automatically.

Optional extras (keep the DuckDB constraint when installing from PyPI):

```bash
pip install "intermine314[pandas,speed]==0.2.1" "duckdb>=1.5.6,<2"
```

Available extras are `pandas` (historical dataframe output), `speed` (orjson),
`proxy` (PySocks) and `plots` (lazily loaded Matplotlib). `analytics` is a
compatibility alias; analytics dependencies are already core.

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

This section describes the compatibility improvements in 0.2.1. See the
[completion evidence](docs/analysis/v021-readiness/completion.md) for scoped
original-client comparisons, measured tradeoffs and remaining limits. In a fresh
Python 3.14.5+ environment, install
`pip install "intermine314[pandas]==0.2.1" "duckdb>=1.5.6,<2"`
when historical `dataframe()` calls are needed; the base package supports native
Polars and every other core API. The published 0.2.0 wheel does not include these
new compatibility refinements.

Native imports (``from intermine314.service import Service``) retain dictionary
result defaults. The original-name facade (``from intermine314.webservice import
Service``) defaults to ``compatibility="legacy"``: ``results()`` and iteration return
model objects, and ``rows()`` returns indexed ``ResultRow`` values. Either Service
accepts an explicit compatibility profile. A direct ``Query(Model(...))`` infers
legacy behavior; Service-created queries retain their Service's profile.

Legacy result streams support `len()`/`bool()` via an independent, validated
traversal. Consequently `list(results)` can make two requests, just as in the
original client; use `[row for row in results]` for one traversal. Native streams
remain unsized and retain shared immutable metadata and stream-open parser
binding. Legacy streams retain mutable metadata and observe active constructor
changes. Both use the same managed transport and validated export pipeline.

Configuration assumes fresh installations: use the packaged defaults,
`ResourceProfile` and `ParallelOptions`. Removed intermine314 scheduler keys
remain errors; no old-configuration translator is provided.

Restored names include ``query``/``new_query``/``select``, ``filter``, view aliases,
``order_by``, ``all``, ``size``, ``summarise``/``summarize`` and ``c``. Model/Column
expressions, saved XML, templates, lists, summaries, identifier resolution,
registry helpers and optional plotting use the shared managed transport.
``dataframe()`` defaults to **pandas in legacy mode** and **Polars in native mode**.
Pass `backend="pandas"` or `backend="polars"` to override it. Pandas is an optional
final adapter with Arrow extension dtypes and a RangeIndex; it does not reproduce
all old NumPy/object dtype behavior. A missing extra fails before reading or
writing data. ``results(row="dataframe")`` remains a
stream of dictionaries. See the [compatibility guide](docs/source/release_readiness.rst) for tested
contracts and deliberate differences; the final 460-symbol audit is separate.

Historical owned re-exports are restored, including
``from intermine314.webservice import ServiceError``. The namespace and dependency
declaration must still be changed explicitly; this distribution does not install
an ``intermine`` shim. See the [release-readiness and migration guide](docs/source/release_readiness.rst)
for covered areas, remaining limits and release checks. Package-managed mutations
do not automatically retry; supplied sessions retain the caller's retry policy.
The speed extra preserves exact large integers with a guarded JSON decoder.

The [0.2.0 readiness review against published 0.1.7](docs/analysis/evidence/v020-vs017/report.md)
records installation checks, migration changes and measured trade-offs. In the
matched-dependency fixtures, result-list materialization is faster and compacted
directories are smaller, while dictionary streaming and small custom-batch
exports take longer. Earlier development-checkout speedups use a different baseline.

See the [measured storage and job-time comparison](docs/source/storage_performance.rst)
for CSV, compressed CSV, JSON and Parquet. Local processing measurements and
estimated network time are identified separately; performance depends on the job.

Polars, DuckDB and PyArrow are core dependencies, loaded when analytics is called.
The pipeline is InterMine or explicit CSV input → Polars → Parquet → DuckDB SQL →
Arrow → Polars, with optional pandas conversion only at the final output boundary.
For an existing file or borrowed CSV stream:

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
CONNECT host/header safeguards shipped there. The [dated dependency review](docs/source/release_readiness.rst)
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
