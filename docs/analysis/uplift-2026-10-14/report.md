# Completed compatibility-preserving pipeline uplift

All six phases implement the accepted 24-task plan. Package version remains
**0.1.8**; this work neither tags nor publishes a release. Baseline is the
pre-uplift **intermine314 development checkout** `8e584c5`, not the old PyPI
`intermine` distribution. Timings below must not be presented as a comparison
against that different package.

## Results and measurement limits

| Workload | Before median ms | After median ms | Speedup |
|---|---:|---:|---:|
| Materialize 10k list rows | 15.511 | 6.617 | 2.34× |
| Keyed access, 10k × 20 fields | 23.391 | 7.336 | 3.19× |
| Decode 10k list rows | 7.556 | 6.117 | 1.24× |
| Decode 10k model objects | 18.705 | 10.486 | 1.78× |
| 100 eager template calls | 27.760 | 18.896 | 1.47× |
| Typed 1,000 × 64 integer batch | 5.280 | 3.206 | 1.65× |
| 10k HTTP fixture rows → Parquet directory | 15.914 | 13.238 | 1.20× |
| 10k numeric rows → single Parquet file | 17.496 | 5.206 | 3.36× |
| 10k String/Int64 rows → single Parquet file | 17.489 | 17.233 | 1.01× |
| Prepare 100k identifier upload | 42.490 | 37.522 | 1.13× |
| 25 small SQL queries; opt-in borrowed connection | 286.967 | 31.494 | 9.11× |

Cold `from intermine314.webservice import Service` import: **26.138 → 25.976 ms**;
no analytics engines imported. Against the accepted phase 5 checkout, final
HTTP-directory, numeric-file and String/Int64-file export medians all remain
within the 3% regression gate (`final_previous_trials.parquet`).
`final_summary.parquet` also records the interquartile ranges and Python peaks.

Every final timing uses 21 randomized paired, isolated processes with the same
CPython 3.14.5 interpreter and dependencies, JIT disabled. Warm timings exclude
imports and one warmup. Python allocation peaks are measured in a separate run;
they are **not RSS**. Source digests and interpreter strings accompany the trials.
Fresh-process import is measured separately. Filesystem caches are not purged.
The WSL environment exposes 18 logical CPUs; these are not measurements on the
unrestricted 20-core host or public mines. Do not multiply component speedups.

Typed output, row count/order and expected values are checked, and every paired
trial must return the same result. Export workloads include publication and
reading the resulting row count. They use 10,000-row fixtures, not whole mine
extractions. Repeated SQL measures 25 sums over 1,000 Parquet rows: the baseline
opens a connection each time; the opt-in candidate borrows one parent connection
initialized outside the timed region. The normal owned-connection default stays.

Additional acceptance evidence:

- Shared keyed rows: retained Python peak **21,029,031 → 3,216,089 bytes**, an
  **84.7%** reduction for 10,000 rows × 20 fields (phase 2).
- One `list(results)` response instead of two; built-in template helpers use
  one deep execution clone instead of up to four. Public `clone()` stays deep.
- 1,000 worker lookups inside an operation: **6.720 → 0.091 ms** (phase 3).
  A new operation refreshes affinity/cgroup quota; native engine admission still
  charges the actual engine pool, including externally initialized pools.
- Redundant sorting copies: **2.864 → 2.716 ms**, Python peak
  **29,376 → 16,936 bytes** (phase 6 targeted fixture).
- Explicit object prefetch: **180.150 → 22.901 ms**, 32 → 4 requests;
  four-mine discovery: **67.438 → 19.104 ms**, same 12 requests;
  Service reuse: **222.142 → 11.115 ms**, 40 → 2 requests.
  These are opt-in workflow demonstrations with simulated 5 ms request latency.
  See [workflow and rejected-candidate evidence](phase5-experiments.md).

## Every opportunity and its disposition

| ID | Section | Implemented change / completed decision | Retired code and retained boundary |
|---|---|---|---|
| O01 | `service/_result_layout.py`, `session.py` | One immutable lookup layout per built-in row stream | Per-row path/map construction removed; direct constructors mutable, alias collisions and copies preserved |
| O02 | `_result_object.py`, stream setup | Bind parser/class and compile object projection once | Repeated imports/view splitting removed; custom pre-open dispatch, independent caches and identity maps retained |
| O03 | `session.py` list parser | Direct list construction for exact built-ins | Throwaway ResultRow removed; old-server cell unwrapping and custom `to_l` retained |
| O04 | `session.py`, `util/json.py` | Pass UTF-8 row bytes to default JSON decoder | Redundant decoding removed; custom text decoders, exact integers/Decimal and error/footer semantics retained |
| O05 | `query/template.py`, `builder.py` | One private execution snapshot through eager helpers | Nested adjustment/cloning removed; public clone/spec and custom dispatch retained |
| O06 | `export/runtime.py`, `arrow.py` | Operation stage plan and one homogeneous-batch reservation | Repeated quota reads and per-column admission removed; next-operation refresh and actual native-pool accounting retained |
| O07 | `export/parquet.py` | Promote a single provably equivalent staged part | Eligible recompression/COPY removed immediately; strings, decimal/nested/extension types, other codecs, empty/multiple groups keep COPY |
| O08 | `query/_columnar.py`, `builder.py` | Built-in validated page → typed column batch | Intermediate row dictionaries removed on built-in exports; public dict/custom providers retained; transpose/native overlap charged |
| O09 | `export/arrow.py` | One conversion plan per write execution | Repeated schema/type maps removed; decimal scale refinement and strict mixed-value fallback retained |
| O10 | `query/executor.py`, `builder.py` | Shared prepared result/count wire implementation | Duplicate count reader and invariant-key construction removed; mutated parameters, summaries and exceptions preserved |
| O11 | `lists/_csv.py` | Bounded joined upload chunks | Per-identifier writes removed; full validation before single mutation, exact bytes and Content-Length retained |
| O12 | `export/query.py` | `query_parquet(connection=...)` borrows a private cursor | Caller can avoid repeated parent connection setup; no second SQL engine or result cache; settings/lifetime remain caller-owned |
| O13 | `_prefetch.py`, metadata workflows | Document and measure existing bounded prefetch, service reuse and independent discovery | No duplicate executor introduced; no automatic requests or speculative cache mutation |
| O14 | `session.py`, `parallel_runtime.py` | Lazy debug formatting adopted; instrumentation fusion rejected | Entire-wrapper removal gives only 1.15%, then 0.44%; existing page-mode export metrics and row close/error wrapper retained |
| O15 | `model.py`, builder, registry | Remove redundant lists before sorting and local key comprehension | Public row/model iteration snapshots and prefetch's validation-window list retained for distinct semantics |
| O16 | `export/query.py` | Native Arrow `read_all` evaluated and rejected | 3.52× local speedup fails cancellation: consumes 20 batches after close request versus current 1; current loop retained |
| O17 | `export/transforms.py` | Worker setup hoisting evaluated and rejected | Two 21-pair runs yield 1.8% / 2.1%, below 5%; prototype removed; per-task reset/callback lookup/teardown retained |
| O18 | private helpers, compatibility tests | Delete unused helpers/cap; share exact close helper; remove duplicate signature/config mirrors | Public aliases, negative API tests and unique semantic/failure scenarios retained; tests do not ship in wheel |

Single-part promotion is deliberately narrow: Zstd, nonempty single row group
within the configured size, exact Arrow schema, primitive numeric/Boolean/Date
columns. Twelve dtype cases compare schema/metadata, values, DuckDB logical
schema, order, compression and row groups. It is not a blanket optimization for
all single-file outputs. Ordinary String/Int64 output remains on DuckDB COPY.

## Import migration and compatibility

```python
# Change the package name; keep the historical service module.
from intermine314.webservice import Service

with Service("https://your-mine.example/service") as service:
    query = service.new_query("Gene")
    query.add_view("Gene.symbol")
    rows = query.rows()
    try:
        for row in rows:
            print(row["Gene.symbol"])
    finally:
        rows.close()
```

The legacy facade keeps its row/object defaults. `intermine314.service.Service`
uses the native dictionary profile. Both share one runtime. Covered workflows
include query/model DSL, constraints/logic/joins/sorts, result modes and eager
helpers, templates, registry discovery, lists/tags/enrichment, identifier jobs,
transport ownership and supported numerical types. The final historical
inventory has **460 entries: 331 passing scopes and 129 tested departures in
43 groups**. This is not a count of equivalent real-world applications.

Three accepted migration changes from this uplift:

1. Streams are unsized: `len(results)` raises `TypeError`; `bool(results)` does
   not test emptiness. Materialize once or explicitly probe and close. Query
   `count()` / `len(query)` still ask the server; joined row counts may differ
   from yielded model-object counts.
2. Streamed built-in row/object metadata is shared and immutable. Direct result
   construction remains mutable. Rebind metadata attributes to independent
   lists/dicts when customizing; do not mutate shared metadata in place.
3. Result parser/class customization is captured when the stream opens. A later
   change affects the next stream. Pre-open customization remains supported.

Existing departures still apply: `dataframe()` returns Polars, not pandas;
unsupported precision bridges reject values; closing a Service does not delete
remote lists. Live authenticated writes, all public mines, arbitrary third-party
subclasses, moving offset datasets and complete user applications are not
certified by offline tests. No claim of universal import-only replacement is made.
See [release/migration guide](../../source/release_readiness.rst),
[query details](../../source/query.rst) and the
[full API audit](../intermine-api-final-coverage.json).

## CPU budgets, native languages and dependencies

No dependencies, version floors or language runtimes changed. Current and
minimum installations both resolved Requests 2.34.2, urllib3 2.8.0, Polars 2.0.0,
DuckDB 1.5.6 and PyArrow 26.0.0. Optional orjson, Matplotlib and PySocks retain
separate installation boundaries. Rust/C++ kernels already enter through these
libraries; a new Rust extension or TypeScript layer has no demonstrated benefit
for this Python package. No subinterpreter/free-threaded build certification or
JIT requirement is introduced.

| Stage | Default on an available 16+ CPU budget | Execution boundary |
|---|---:|---|
| Public HTTP / low-latency / capable high-latency server | 4 / 2 / 8 fetch workers | Per-host/global permits; initially one page per worker; page and byte bounds |
| Arrow CPU / I/O | 4 / 2 | Batch native admission; actual external pool size charged |
| Ordered / unordered COPY | 4 / 8 | One COPY; unordered only when order is dispensable; low-memory mode 1 |
| DuckDB SQL | 16 | One main query, enough row groups; no outer pool of parallel queries |
| Polars | 12 | One native plan; configure before first import |
| Coarse Python transforms | 4 processes | Spawn, path-only IPC; native threads 1 per child |
| Metadata / mutable owner | 4 / 1 | One cache initialization per key; one owner for mutations/publication |

`cpu16` remains the default; `cpu8`, `cpu6`, `cpu4`, `cpu2` remain available and
are clamped to detected affinity/quota. These are shared budget allocations,
not additive pools. The 20-core Intel Ultra 7 265K therefore starts at budget 16;
public-server worker caps and the unrestricted host's optimum require workload
measurements. Existing server-aware sweeps remain recommendations, not certified
rate limits. Parquet remains canonical storage and DuckDB the database engine.

## Verification, handshakes and repository size

Local checks: **2,813 tests passed** in the final API trace (35.41 s), Ruff and
whitespace checks pass; all **795 curated semantic test nodes** map to the
historical inventory. Fresh wheel and sdist pass Twine, license/version/artifact
checks, pip check, six current installation modes and strict installed Sphinx.
Minimum Python 3.14.5/dependency installs pass base, speed and combined smoke
checks. Both current and minimum installed wheels pass **904 selected regression
cases**, outside the checkout with installed-module provenance checked. CI also
covers Python 3.14's current patch, four optional-dependency unit-test modes and
minimum sdist installation. Exact final-SHA CI remains the post-push gate.

The first attempt to run two builds concurrently collided in setuptools' shared
sdist staging. The current build was rerun alone and passed; this was an analysis
orchestration correction. Local build metadata truthfully records a dirty source
revision because the verification task was being completed; final CI builds from
the clean pushed SHA. A first final benchmark run failed to JSON-serialize its
Decimal SQL total; the harness now verifies its exact integer value and records
it as an integer. The entire paired comparison was rerun; no discarded trial is
used in the final summary.

| Inventory | Original | Final | Change |
|---|---:|---:|---:|
| Runtime Python files | 77 | 79 | +2 private plan/batch modules |
| Runtime physical lines | 14,132 | 14,385 | +253 (1.8%) |
| Runtime source bytes | 541,155 | 553,431 | +12,276 (2.3%) |
| Test/fixture Python files | 90 | 91 | +1 part-promotion regression module |
| Test/fixture physical lines | 17,991 | 18,367 | +376 |
| Collected tests | 2,769 | 2,813 | +44 net cases |

The whole codebase **did not shrink**: explicit immutable plans, typed batching,
publication guards and regressions cost more source than the retired helpers.
Repeated runtime work and allocations decreased. The retained public facades and
fallbacks provide distinct contracts; deleting them to meet a line-count target
would weaken compatibility. Physical lines include documentation and are not a
measure of algorithmic complexity or installed memory. Tests do not ship in the
wheel.

Each numbered task has its own commit on master. Each completed phase was
pushed, and the next phase began only after all seven jobs passed for the exact
handoff SHA. The phase 5 RST underline error was corrected and its complete CI
rerun before phase 6 started. The final phase adds installed-wheel regressions
to CI rather than relying solely on editable-checkout tests.

| Phase | Accepted pushed handoff | Seven-job CI |
|---|---|---|
| 1 | `83c3bed2d8b94eb2296bc0ad9fc78685cdcac6ba` | [38466528921](https://github.com/karikris/intermine314/actions/runs/38466528921) |
| 2 | `891a07708c61786ebff54f62a15996c26ebb72df` | [38467731555](https://github.com/karikris/intermine314/actions/runs/38467731555) |
| 3 | `1a5b75cb67af1e7a0e90e3a735715c1aa71eaf5e` | [38468995634](https://github.com/karikris/intermine314/actions/runs/38468995634) |
| 4 | `f71a35bb4d4eb3cc95c269c828f215157ad301a0` | [38471063810](https://github.com/karikris/intermine314/actions/runs/38471063810) |
| 5 | `55b00dfaef872285e31dfe9e02b7eff51661b1c3` | [38473108447](https://github.com/karikris/intermine314/actions/runs/38473108447) |
| 6 | Final handoff commit containing `phase6.md` | Exact-SHA CI required after push; completion response supplies the resulting run |

## Reproduce and inspect

All measurement rows, raw GitHits responses, rejected prototype source, task
receipts and distribution reports are canonical Zstd Parquet files in this
directory. `analysis.duckdb` contains views; rebuild/relocate them with
`python docs/analysis/uplift-2026-10-14/catalog.py`. For example:

```sql
SELECT case, variant, median(ns) / 1e6 AS median_ms,
       median(python_peak_bytes) AS median_python_peak
FROM final_original_trials GROUP BY ALL ORDER BY case, variant;
```

Create the baseline with `git archive 8e584c5` into a separate directory. Run
`compare.py --baseline BASELINE --candidate . --label rerun --cases materialize
keyed_rows list_decode object_decode template_execution typed_batch http_export
numeric_export parquet_export identifier_body repeated_sql`, then
`cold_import.py --baseline BASELINE --candidate .`. Run benchmarks sequentially,
without simultaneous builds or tests. See `README.md` for acceptance thresholds.

GitHits evidence pins CPython 3.14.8, Arrow 26, Polars 2, orjson 3.13 and DuckDB
Python 1.5.6 sources. `sources.parquet` records the source, decision and raw
response per phase. These sources support semantic/ownership decisions; local
measurements establish gains. Composition through private execution/layout/
conversion plans replaces repeated work without another backend framework or
new public class hierarchy. Rejected optimizations remain evidence, not dormant
production branches. Superseded code is already removed and recoverable in Git.
