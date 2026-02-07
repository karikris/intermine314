# intermine314

Python 3.14+ client for InterMine web services.

This package modernizes the historical InterMine Python client for current Python runtimes, with a focus on practical reliability for research use.

## Ownership and Credit

Copyright (c) 2026 Monash University, Plant Energy and Biotechnology Lab.

Project ownership and stewardship:
- Kris Kari
- Dr. Maria Ermakova
- Plant Energy and Biotechnology Lab, Monash University
- Contact: toffe.kari@gmail.com

Original credit:
- The original InterMine team and community contributors are credited for foundational work that this package builds upon.

## License

Licensed under the MIT License (see `LICENSE-LGPL`, which now contains the active MIT license text and notice).

## Requirements

- Python 3.14+

## Supported Mines

Priority support is focused on:
- MaizeMine
- ThaleMine
- LegumeMine
- OakMine
- WheatMine

## Installation

```bash
pip install intermine314
```

Repository: https://github.com/kriskari/intermine314

## Quick Example

```python
from intermine314.webservice import Service

service = Service("https://www.flymine.org/query/service")
query = service.new_query("Gene")
query.add_view("Gene.symbol", "Gene.length")

for row in query.run_parallel(
    page_size=2000,
    max_workers=16,
    profile="large_query",
    ordered="window",
    ordered_window_pages=10,
    prefetch=32,
    inflight_limit=24,
    pagination="auto",
):
    print(row)
```

## Parallel Worker Defaults

`intermine314` uses adaptive defaults when `max_workers` is omitted:

- LegumeMine: `8` workers for queries up to `20,000` rows, then `1` worker above that threshold.
- MaizeMine, ThaleMine, OakMine, WheatMine: `16` workers.
- Unknown mines: fallback to `16` workers.

Parallel query APIs use `pagination="auto"` by default.

You can tune this per query based on your hardware and network:

- Smaller machines or constrained environments: use `4` to `8`.
- High-core desktops/workstations with stable network: use `16` to `32`.
- If remote endpoints rate-limit aggressively, lower workers to reduce retries/timeouts.

```python
rows = query.run_parallel(
    row="dict",
    page_size=2000,
    max_workers=16,  # optional override; omit to use mine registry defaults
    profile="large_query",  # presets: default | large_query | unordered | mostly_ordered
    large_query_mode=True,  # defaults prefetch to 2 * workers when prefetch is omitted
    ordered="window",       # bool or: ordered | unordered | window | mostly_ordered
    ordered_window_pages=10,
    prefetch=32,
    inflight_limit=24,      # independent Python 3.14 in-flight cap (buffersize in ordered mode)
    pagination="auto",
)
```

Throughput tip: for raw max throughput benchmarking, use `ordered=False` (or `ordered="unordered"`).

Preset examples are documented in `config/parallel-profiles.toml`.
Mine-specific worker profiles are defined in `config/mine-parallel-preferences.toml`.

## Testing

Run unit tests against the local mock service:

```bash
python setup.py test
```

Run live tests:

```bash
python setup.py livetest
```

## Upgrades for Python 3.14

The following package upgrades were applied in this modernization cycle:

- Raised runtime baseline to Python 3.14 (`pyproject.toml`, `README.md`, CI, tox).
- Updated packaging metadata toward modern PEP 621 layout.
- Simplified dependency management (`requirements.txt`, optional extras in `pyproject.toml`).
- Removed Python 2 compatibility remnants from runtime modules.
- Removed Python 2 compatibility remnants from tests.
- Modernized query parallel execution behavior in `intermine314/query.py`:
  - bounded in-flight work
  - configurable `prefetch`
  - configurable `inflight_limit` (Python 3.14 `buffersize` cap)
  - profile presets: `default`, `large_query`, `unordered`, `mostly_ordered`
  - `ordered="window"` mode to reduce HOL stalls while keeping near-order
  - fixed edge handling for `start` and `size=None`
  - used `executor.map(..., buffersize=...)` for ordered mode
- Switched core HTTP transport to `requests.Session()` for keep-alive connection reuse (urllib fallback retained with explicit warning).
- Added optional fast JSON decoding path with `orjson` fallback to stdlib `json`.
- Updated CI configuration for Python 3.14.
- Verified local unit test suite and live mine connectivity checks.

## Notes

Documentation/tutorial links from legacy upstream sources are intentionally omitted for now while docs are being updated for the Python 3.14 package line.
