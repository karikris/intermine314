# intermine314

[![CI](https://github.com/karikris/intermine314/actions/workflows/im-build.yml/badge.svg?branch=master)](https://github.com/karikris/intermine314/actions/workflows/im-build.yml)
[![PyPI version](https://img.shields.io/pypi/v/intermine314.svg)](https://pypi.org/project/intermine314/)
[![Python versions supported](https://img.shields.io/pypi/pyversions/intermine314.svg)](https://pypi.org/project/intermine314/)
[![Downloads (Pepy)](https://static.pepy.tech/badge/intermine314)](https://pepy.tech/projects/intermine314)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/karikris/intermine314/badge)](https://securityscorecards.dev/viewer/?uri=github.com/karikris/intermine314)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/karikris/intermine314/blob/master/LICENSE-LGPL)

Python 3.14+ InterMine client built for modern research pipelines.

`intermine314` is the story of taking a classic scientific webservice client and rebuilding it for today's needs:

- privacy-first networking
- open-source transparency
- Tor-ready transport
- analytics-native outputs with Parquet, Polars, and DuckDB

Repository: https://github.com/karikris/intermine314

## The Story

InterMine is a core data source for genomics and systems biology work. But modern workflows need more than basic HTTP requests and row iteration.

`intermine314` focuses on four practical goals:

1. **Privacy first**: one centralized HTTP layer, explicit proxy control, and Tor-safe defaults.
2. **Open source**: all transport, config, and workflow logic is inspectable and testable.
3. **Reliable extraction at scale**: adaptive parallel defaults, retry-aware sessions, and mine-aware worker policies.
4. **Analytics-native handoff**: direct Parquet output and immediate analysis via Polars and DuckDB.

## Privacy-First by Default

### Tor support

You can route all requests through Tor SOCKS5 easily:

```python
from intermine314.webservice import Service

service = Service.tor(
    "https://bar.utoronto.ca/thalemine/service",
    token="YOUR_TOKEN",
)

print(service.version)
```

Equivalent explicit configuration:

```python
from intermine314.webservice import Service

proxy = "socks5h://127.0.0.1:9050"  # DNS is resolved via Tor proxy
service = Service("https://.../service", token="...", proxy_url=proxy)
```

### HTTPS enforcement in Tor mode

When Tor routing is enabled (`tor=True` or Tor-like `proxy_url`), `Service` and `Registry` reject `http://` endpoints by default.

If you explicitly accept plaintext risk, opt in:

```python
service = Service(
    "http://example.org/service",
    proxy_url="socks5h://127.0.0.1:9050",
    allow_http_over_tor=True,
)
```

## Open Source and Auditable

Copyright (c) 2026 Monash University,
Plant Energy and Biotechnology Lab.

Owners:

- Kris Kari
- Dr. Maria Ermakova
- Plant Energy and Biotechnology Lab, Monash University
- Contact: toffe.kari@gmail.com

Original credit:

- Original InterMine team and community contributors.

License: MIT (see `LICENSE-LGPL`, which contains the active MIT license text and notice).

## Analytics-Native Workflow

`intermine314` treats extraction and analytics as one pipeline:

- **Parquet** for durable columnar output
- **Polars** for fast dataframe operations
- **DuckDB** for SQL over files or in-memory tables

```python
from intermine314 import fetch_from_mine

result = fetch_from_mine(
    mine_url="https://maizemine.rnet.missouri.edu/maizemine/service",
    root_class="Gene",
    views=["Gene.primaryIdentifier", "Gene.symbol", "Gene.length"],
    joins=[],
    size=50_000,
    workflow="elt",            # elt | etl
    production_profile="auto", # mine-aware profile resolution
    parquet_path="/tmp/maize_genes.parquet",
    duckdb_table="genes",
)

duckdb_con = result["duckdb_connection"]
print(duckdb_con.execute("select count(*) from genes").fetchall())
```

## Supported Mines (Priority)

- MaizeMine
- ThaleMine
- LegumeMine
- OakMine
- WheatMine

WheatMine API endpoint:

- `https://urgi.versailles.inrae.fr/WheatMine/service`

MaizeMine API endpoints:

- `https://maizemine.rnet.missouri.edu/maizemine/service`
- fallback: `http://maizemine.rnet.missouri.edu:8080/maizemine/service`

## Installation

```bash
pip install intermine314
```

Optional extras:

```bash
pip install "intermine314[speed]"  # faster JSON decode path
pip install "intermine314[proxy]"  # PySocks for socks5/socks5h proxy URLs
```

## Runtime Configuration

Runtime config files are shipped inside the package and loaded via package resources:

- `intermine314.config/defaults.toml`
- `intermine314.config/mine-parallel-preferences.toml`
- `intermine314.config/parallel-profiles.toml`

This keeps behavior consistent between:

- `pip install intermine314`
- editable/source checkout installs

You can still override with explicit env vars:

- `INTERMINE314_RUNTIME_DEFAULTS_PATH`
- `INTERMINE314_MINE_PARALLEL_PREFERENCES_PATH`
- `INTERMINE314_PARALLEL_PROFILES_PATH`

## Performance Defaults

When `max_workers` is omitted, adaptive mine-aware defaults are used:

- LegumeMine: `4`
- MaizeMine: `8`
- ThaleMine/OakMine/WheatMine: `16`
- unknown mines: `16`

For highest throughput on large pulls, prefer unordered fetch mode (`ordered=False` / `ordered="unordered"`) when strict row order is not required.

## Testing

```bash
python -m pytest -q tests
python setup.py analyticscheck
```

Live tests (if endpoint/credentials available):

```bash
INTERMINE314_RUN_LIVE_TESTS=1 TESTMODEL_URL="https://<mine>/service" python -m pytest -q tests
```

## Related Docs

- Benchmark protocol and performance workflow: `BENCHMARK.md`
- Tor notes and examples: `TOR.md`
