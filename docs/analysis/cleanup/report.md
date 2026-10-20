# Package cleanup completion

Implemented all seven phases on master, one commit per task, with an exact-head
full CI gate between phases. Version remains **0.2.0**. This work creates no tag
or publication. Baseline: `5ea5b765d9edfff04fda3aff82cc9a14bc34e977`.
GitHits upstream checks, replacement obligations and evidence limits are in
[obligations.md](obligations.md). Final acceptance uses Python 3.14.5.

## Changes and retained contracts

| Area | Implemented change | Retained contract |
|---|---|---|
| Scheduling | Removed 11 obsolete runtime TOML keys; actionable migration errors; benchmark defaults moved into benchmark profiles; one options/profile/stage resolution path | ParallelOptions fields and labels, native CPU budget 16 and cpu8/cpu6/cpu4/cpu2, byte/row/page bounds and host/Tor admission |
| Shared ownership | Removed unused session/XML wrappers; shared helper account transaction and response owner | Independent helper accounts, rollback/errors/interruptions, authentication adoption, borrowed lifetimes and exactly-once cleanup |
| Execution | Shared result/count dispatch, canonical method checks and live custom adapter; removed private execution forwarder | Immutable builtin page snapshots, custom providers/executors, overridden parameters/endpoints, duck-typed count, summaries and atomic exports |
| XML | Bounded Expat/ElementTree loading for Query/Template; removed redundant Template loader forwarder | Descendant/node paths, codes/logic/validation/errors, empty values/CDATA/direct text, lazy public XML-string template dictionaries, real public DOM return values |
| Evidence/repository | Consolidated 43-group migration guide; one temporary DuckDB catalog over Parquet; migrated reproduction tools; removed superseded narratives/catalogs/administrative tests | Immutable historical inventory, original measurement dates/source hashes, current audit, published storage tables, licenses, release deck/assets and benchmark site |

The package retains the original **460-symbol** inventory: **331 passing-scoped**
assertion scopes and **129 tested-departure** symbols across **43 groups**, plus
**40 supplemental reexports**. This is scoped compatibility, with declared
migration requirements. Use `intermine314.webservice.Service` for historical
defaults and `intermine314.service.Service` for native defaults. Historical
pandas behavior still requires a Polars migration. See the maintained
[compatibility guide](../../source/release_readiness.rst).

## Size and verification

| Metric | Before | After | Change |
|---|---:|---:|---:|
| runtime_lines | 14,385 | 14,173 | -212 |
| runtime_bytes | 553,431 | 545,951 | -7,480 |
| runtime_tokens | 81,008 | 80,102 | -906 |
| runtime_declarations | 1,135 | 1,127 | -8 |
| wheel_bytes | 174,993 | 174,094 | -899 |
| wheel_uncompressed_bytes | 575,907 | 567,927 | -7,980 |
| source_test_cases | 2,734 | 2,760 | +26 |
| installed_regressions | 877 | 969 | +92 |

Runtime physical LOC fell **1.47%** and non-comment tokens **1.12%**. The compressed
wheel fell **0.51%** and unpacked distribution bytes **1.39%**. These are modest
package reductions; the 241 removed/migrated original evidence files are a
separate repository cleanup. The manifest verifies every original disposition.
The active compatibility facades remain because they expose historical imports
without duplicating feature implementations.

All **2,760 source cases** and **969 installed-wheel regressions** pass. Six fresh
local installation modes, installed documentation, Ruff, actionlint and Sphinx
warnings-as-errors pass. Eight injected faults are detected. Qualified AST
coverage comparison reports no lost previously executed locations/edges in
**972 unchanged declarations**. All **38 replaced/retired declarations** map to
retained behavior and assertion owners; raw shifted line numbers are not used
as an equivalence claim. Coverage does not prove native-engine or server
correctness beyond the tested contracts.

Local distribution evidence records its actual revision and dirty-state; its
runtime hashes match the final runtime snapshot. Final clean exact-head CI is
the closing artifact/installation gate. No earlier release receipt is relabeled
as current acceptance. [acceptance.parquet](acceptance.parquet) preserves the
local receipts; [final-runtime.parquet](final-runtime.parquet) contains all 79
runtime file hashes. The API audit independently binds the final controlled
execution inputs.

## Matched offline observations

Five randomized pairs per fixture compare the baseline and final runtime on the
same Python/dependencies. Each time measures 20 uninstrumented operations after
one warmup. One separate operation measures Python allocation peak; Linux VmHWM
records process peak RSS. The XML fixture loads one query with 2,000 values.
The result fixture streams 1,000 eight-column rows. Startup, HTTP and native
analytics engines are outside the timed work.

| Fixture | Source | Median ms/operation | Python allocation peak bytes | Peak RSS KiB |
|---|---|---:|---:|---:|
| query_xml | before | 3.501 | 1,314,216 | 42,624 |
| query_xml | after | 1.622 | 882,376 | 33,696 |
| result_stream | before | 0.913 | 7,095 | 25,948 |
| result_stream | after | 0.921 | 7,095 | 25,936 |

The XML fixture is approximately **2.2× faster**, with **33% lower Python
allocation peak** and **21% lower peak RSS**. Result streaming is effectively
unchanged (about 0.8% slower median, identical Python allocation peak). Five
pairs and background host activity limit inference. These observations establish
no public-server, whole-job or native-engine speedup. Parquet/Arrow precision,
DuckDB querying, resource profiles and compaction remain covered; this cleanup
adds no new data-storage compression algorithm. Published storage savings and
the 0.1.7 comparison retain their original provenance under
[../evidence](../evidence).

Reproduce current cleanup timings with:

```sh
.venv/bin/python -m benchmarks.runners.cleanup_comparison --output /tmp/cleanup-trials.parquet
.venv/bin/python scripts/measure_test_coverage.py coverage --work /tmp/cleanup-coverage
.venv/bin/python scripts/verify_contract_faults.py --output /tmp/cleanup-faults
.venv/bin/python -m benchmarks.evidence docs/analysis/cleanup 'SELECT * FROM final_metrics'
```

The coverage comparison tool accepts baseline/current coverage JSON and writes
qualified AST points and edges to Parquet. The versioned obligation table retains
both sets of normalized points, declaration hashes and explicit replacement
mappings. Baseline source/tests and removed code remain recoverable from Git.

## Phase handshakes

| Phase | Pushed head | Full CI |
|---|---|---|
| 1 | `a551124` | [Passed](https://github.com/karikris/intermine314/actions/runs/39411175985) |
| 2 | `4f9cda6` | [Passed](https://github.com/karikris/intermine314/actions/runs/39411850696) |
| 3 | `10b8e63` | [Passed](https://github.com/karikris/intermine314/actions/runs/39412484300) |
| 4 | `fc846f5` | [Passed](https://github.com/karikris/intermine314/actions/runs/39413111251) |
| 5 | `ba0da8b` | [Passed](https://github.com/karikris/intermine314/actions/runs/39413795062) |
| 6 | `3fdbefa` | [Passed](https://github.com/karikris/intermine314/actions/runs/39414679240) |
| 7 | Final report commit | Exact-head run linked in the completion response |

The final push is made after source/install/coverage/API/docs acceptance and
manifest review. Its receipt is linked externally to avoid a self-referential
commit/CI loop. All task commits and previous phase receipts are in Git and
[ci-receipts.parquet](ci-receipts.parquet).
