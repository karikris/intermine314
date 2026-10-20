# Fresh installation and legacy compatibility implementation

Baseline: `dd745639efc102dfe8df43827b0c5ab62e5c333c`, version 0.2.0.
Target: release-ready 0.2.1 work, without changing version, tagging or publishing.
All installations are fresh. Original InterMine 1.13.0 at
`d888b779c8050bad789e26b312f40d220bc85d0d` defines the historical API reference.

| Interface | Legacy profile / webservice.Service | Native profile / service.Service |
|---|---|---|
| Configuration | Current ResourceProfile and ParallelOptions; reject removed keys | Same |
| Result sizing | Separate full validated traversal, no retained rows | Unsized, one-pass list materialization |
| Metadata | Mutable row/object metadata with upstream sharing | Shared immutable built-in layouts |
| Result customization | Constructor/view changes affect active streams | Stream-open snapshot |
| dataframe | Optional pandas final output; missing extra fails before side effects | Polars |
| dataframe backend | Explicit pandas/polars override | Same |
| Processing | Shared Python 3.14, Parquet, DuckDB, Arrow, Polars pipeline | Same |

Preserve correctness repairs, footer validation, exact numeric values, bounded
pages/bytes, atomic output, borrowed ownership, immutable execution snapshots
and CPU budgets 16/8/6/4/2. Do not reproduce upstream bugs.

| Phase | Tasks (one commit each) | Handshake |
|---|---|---|
| 1 | Fresh-install policy; configuration contracts; upstream obligation matrix | Valid defaults/precedence and named compatibility assertions |
| 2 | Legacy sizing; mutable metadata; live parser customization | Native one-pass behavior and resource cleanup retained |
| 3 | Public encoders; collection constraint values; wire/clone parity | Exact form/XML behavior and custom endpoint contracts |
| 4 | Optional pandas dependency; final output backend; install/CI boundaries | Fresh base/pandas/combined wheels and precision |
| 5 | Historical evidence reproduction; current benchmark references; release guidance | Executable reproduction and scoped measurements |
| 6 | Full candidate acceptance; paired measurements; live checks and final report | Exact-head green CI, artifact provenance, explicit departures |

Push every phase to master. Full CI must pass on the exact phase SHA before the
next phase starts. Record the preceding phase's receipt in the next phase;
link the closing run in the final response. Appropriate targeted tests run
before each task commit. Update the API audit at runtime phase boundaries.
No fixed passing/departure count is a target: preserve the 460-symbol inventory
and classify only executed semantic assertions. Additions remain supplemental.

GitHits verified original result sizing (`results.py:375–386`), live result
dispatch (`results.py:400–411`), public encoders (`results.py:296–305`), pandas
output (`query.py:1444–1467`) and collection value identity
(`constraints.py:761–801`). Pandas 3.0.6 has CPython 3.14 wheels; GitHits reported
no active advisories in it or its four resolved dependencies on 20 October 2026.
Registry evidence is not an exhaustive environment/reachability audit.

Phase 1 handshake: `a3c96d97ae52fe840efbf3724ef83ee4703bb8fb`,
[all seven CI jobs passed](https://github.com/karikris/intermine314/actions/runs/39420443916).

Phase 2 handshake: `ac0ef1be183e6fd6b1aab4f39a96173daed14c69`,
[all seven CI jobs passed](https://github.com/karikris/intermine314/actions/runs/39421359168).
2,775 source tests passed. GitHits confirmed the public UTF-8 helper contract
against pinned original results.py:296–305 before phase 3 implementation.
