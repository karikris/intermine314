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

for row in query.run_parallel(page_size=2000, max_workers=16, prefetch=16):
    print(row)
```

## Parallel Worker Defaults

`intermine314` defaults parallel query workflows to `16` workers.

You can tune this per query based on your hardware and network:

- Smaller machines or constrained environments: use `4` to `8`.
- High-core desktops/workstations with stable network: use `16` to `32`.
- If remote endpoints rate-limit aggressively, lower workers to reduce retries/timeouts.

```python
rows = query.run_parallel(
    row="dict",
    page_size=2000,
    max_workers=16,  # override per machine/workload
    prefetch=16,     # typically match max_workers
)
```

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
  - fixed edge handling for `start` and `size=None`
  - used `executor.map(..., buffersize=...)` for ordered mode
- Updated CI configuration for Python 3.14.
- Verified local unit test suite and live mine connectivity checks.

## Notes

Documentation/tutorial links from legacy upstream sources are intentionally omitted for now while docs are being updated for the Python 3.14 package line.
