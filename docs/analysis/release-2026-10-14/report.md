# 0.2.0 release-readiness review — 14 October 2026

**Verdict: ready to prepare a 0.2.0 release for the tested Linux/CPython 3.14
scope, with the migration limits below.** No blocking runtime defect was
reproduced in this review. This is a qualified migration target, not a completely
interchangeable distribution. The repository remains **0.1.8**; no version tag or
PyPI upload is part of this audit.

[PyPI's published 0.1.8](https://pypi.org/project/intermine314/) is the October 2
build from `41c363e`, before the restored compatibility and performance work.
Installing that published version does not provide the behavior reviewed here.
PyPI had no 0.2.0 release when checked. README and migration documentation now
make this distinction explicit.

## What was examined

- The runtime inventory contains **77 Python modules, 14,132 lines, 119 class
  declarations and 987 function declarations**. [code_inventory.parquet](code_inventory.parquet)
  records each path, source hash and structural counts. These are inventory
  metrics, not branch-coverage percentages.
- The review followed the shared legacy/native execution path, query/model/XML
  boundaries, streams and terminal status, HTTP retry/session ownership,
  concurrency admission, conversion/precision, Parquet publication, metadata
  caches, object prefetch, job polling, spawned transforms and release tooling.
  Functional claims rely on executed regression assertions and bounded live
  checks, not merely public-name availability.
- **2,769 tests passed** in the full instrumented suite. The refreshed legacy
  audit retains **460 symbols: 333 passing scopes and 127 tested departures in
  39 groups**, with 788 linked semantic test nodes. Its pre/post execution
  manifest covers 275 files. A separate contract checks **40 owned historical
  re-exports**, including exception identity.
- Fresh minimum-dependency wheel/sdist builds passed six isolated installation
  modes: base, plots, analytics alias, sdist, speed and combined plots/speed/proxy.
  Twine, pip consistency checks, packaged licenses/configuration and Sphinx
  against the installed wheel passed. An isolated **0.2.0** copy also passed
  all six modes and documentation. The installed-wheel regression set passed
  **546 cases** for both 0.1.8 and the temporary 0.2.0 candidate.
- GitHits reviewed 13 dependency/tool releases, eight depth-two runtime graphs
  and three upgrades. [Dependency review](dependency-review.md) gives versions,
  effects and source links; [dependencies.parquet](dependencies.parquet)
  retains the provider responses. Current-version advisory counts were zero at
  review time, with the provider's completeness limitations retained.

Machine-readable build, test and CI observations are in
[verification.parquet](verification.parquet). It retains exact source revisions,
artifact hashes and the original verifier reports. The isolated candidate is
an archive of the validated tree with only project/runtime versions changed;
it is a packaging rehearsal, not a release artifact authorized for upload.

## Replacement coverage

| Area / code | Covered by this implementation and review | What remains different or unverified |
| --- | --- | --- |
| Imports, facades, exceptions — `webservice.py`, `service/legacy.py`, `compatibility.py` | Original service path under `intermine314`, aliases and owned re-exports share canonical classes/exception identities; lazy imports verified in installed wheels | No `intermine` namespace shim. Third-party `Requires-Dist: intermine` is not satisfied. Incidental standard-library exports and private helpers are outside the contract |
| Queries and DSL — `query/builder.py`, `constraints.py`, `pathfeatures.py`, `model.py` | Query factories, view/filter/sort aliases, constraints, nested logic, descriptors, joins, model ancestry/validation, cloning and XML round trips | Some historical bugs are repaired, intentionally changing invalid-input/error behavior; every argument combination and arbitrary subclass are not certified |
| Result formats — `results.py`, `_result_object.py` | Legacy facade defaults to model objects for `results()` and indexed `ResultRow` for `rows()`; dict/list/raw text/JSON forms and lazy references/collections tested | Native `service.Service` defaults to dictionaries. Iterators must be closed on early exit. Unbounded lazy collections remain server/data dependent |
| Dataframes — `query/builder.py` | Both profiles return Polars; typed empty results and large integers covered | Original pandas return type and pandas-specific downstream APIs are deliberately not preserved; no pandas performance claim is made |
| Templates/metadata — `service/service.py`, `metadata.py`, `xmlstream.py`, `query/template.py` | Global/private-name fixture contracts, XML parsing, form/constraint adjustment, caching, concurrent initialization and lifecycle barriers; compressed sized reads | Private live templates and every template's business behavior are unverified; small public executions can legitimately return zero rows |
| Lists and enrichment — `lists/` | Creation/update/delete/set operations, tags, ownership, temporary-list cleanup, safe identifier encoding, CSV identifier streaming and Parquet enrichment via offline wire fixtures | Authenticated live writes were not performed. Service close/destruction does not delete server lists; use explicit ListManager cleanup/flush. Separate services/processes need account coordination |
| Accounts and jobs — `service/service.py`, `idresolution.py` | Account request contracts, encoding/token handling, identifier job states/results/deletion and coordinated polling/cancellation | No live registration/deletion/job mutation certification; running HTTP requests finish under transport timeouts |
| Registry/helpers/plots — `registry.py`, `query_manager.py`, `bar_chart.py` | Shared transport, registry cache/lifetime, saved-query handling, optional plotting, SQL pre-aggregation, selected account propagation | Public registries/endpoints can change. Matplotlib renders serially; interactive GUI backends/platforms are not certified |
| HTTP — `service/session.py`, `transport.py`, `admission.py` | TLS defaults, proxy/Tor policy, gzip/deflate/Zstandard streaming, status/error handling, bounded buffering, read-safe POST retries, per-host/global admission and worker session ownership | Borrowed sessions retain caller adapters/retry policy. Live Tor/proxy routing, all server variants and protection bypasses are not certified. Caller Session mutation during requests is unsupported |
| Parallel pages — `query/parallel_offset.py`, `inflight.py`, `parallel_runtime.py` | Full-page status validation before acceptance, overfull-page rejection, ordered/unordered execution, page reservations through buffering/emission, cleanup/cancellation | Byte accounting estimates payloads; it is not an RSS ceiling. Oversized pages may exceed estimates. Stable remote ordering/data are required across offset requests |
| Typed data — `export/arrow.py`, `model_schema.py`, `_schema.py`, `util/json.py` | Exact supported integer/decimal handling, nullable primitives, model widths/dates, schema evolution guards, base/speed JSON precision and Arrow conversion | Arbitrary Python objects/model types are not universal serialization. Decimal precision is limited to 38. External Int128 annotations and timezone-aware ns timestamps are rejected at DuckDB boundaries; Enum/Categorical become strings through DuckDB |
| Parquet storage — `export/parquet.py`, `layout.py`, `manifest.py` | Bounded staging, model schemas, row-group/file geometry, schema cache, metadata namespace, atomic same-file replacement and rollback; default format is Parquet | No crash-durability or cross-process transaction guarantee. Non-exchange directory replacement can have a brief missing-path interval. Exact unsupported logical-type preservation may require direct directory parts |
| SQL/readers/CSV — `export/query.py`, `csv.py`, `output.py`, `fetch.py` | DuckDB over Parquet, parameterized SQL, Arrow readers and borrowed cursor ownership, explicit CSV ingestion/output, leading-zero identifier preservation | Eager frame/table results still materialize. Output batch size does not cap SQL operator memory. SQL is a trusted caller API, not a sandbox; full legacy arbitrary storage-format equivalence is not claimed |
| Resource budgets — `export/resource_profile.py`, `runtime.py` | CPU16 default and CPU8/6/4/2 profiles; stage allocations, native admission, executor reuse, diagnostics, affinity/quota awareness and owned resource shutdown | Advisory for package-routed work. Pre-imported engines, external application pools and caller-issued SQL are not fully controlled. Polars 2.0 memory/spill budgets are separate |
| New parallel APIs — `_prefetch.py`, `export/transforms.py`, `idresolution.py` | Bounded object windows with failure-safe cache updates, one coordinated polling queue, spawned path-only coarse transforms with per-child native limits and publication guards | Prefetch excludes dotted paths and does not cap collection payload bytes. Transforms require an importable callback/main guard and one native thread in caller-created engines. Subinterpreters/free-threaded safety is not certified |
| Python/distributions | CPython 3.14.5 minimum locally; current 3.14 CI; Linux base and extras installation matrix; no runtime dependency on original InterMine or pandas | Python below 3.14.5 unsupported. Windows, macOS, free-threaded builds, other CPU architectures and later Python minors have no certification from this review; native-wheel availability must be checked there |
| Release controls — `scripts/release_gate.py`, `publish-pypi.yml` | Matching project/runtime versions and tag, exact triggering SHA/master ancestry, all install modes/docs, immutable artifact IDs and SHA-256 verification; manual dispatch cannot publish | Actual upload/trusted-publisher authorization requires the release event and external PyPI configuration; the 0.2.0 version/tag has not been created |

The full per-symbol caller effects and recommendations remain in
[the compatibility audit](../intermine-api-final-coverage.json), with its
[Parquet representation](../intermine-api-final-coverage.parquet). The original
1.13.0 reference is pinned at
[`d888b779`](https://github.com/intermine/intermine-ws-python/tree/d888b779c8050bad789e26b312f40d220bc85d0d).
The historical inventory describes the older `41c363e` implementation and is
preserved; it should not be used to infer missing APIs in the current tree.

## High-risk paths checked

| Prior risk or current boundary | Source finding | Executed evidence |
| --- | --- | --- |
| Failed terminal status could overwrite an export | `fetch_page` exhausts the concrete iterator through its footer before returning; publication follows complete conversion | `test_parallel_export_status`, `test_parallel_page_validation`, `test_parallel_cleanup` |
| Sized metadata reads bypassed compression | `raw.read(size, decode_content=True)` uses decompression; EOF/error closes the response/permit | `test_compressed_metadata_http`, `test_response_stream_decoding`, fresh Legume templates |
| Completed ordered pages escaped the budget | Reservations survive pending → buffered → emitting transitions and release after consumption; native batch estimates are charged too | `test_parallel_memory_bounds`, `test_inflight_reservations`, `test_validated_pages` |
| Optional fast JSON could round large integers | Guarded decoder falls back for integers outside the fast decoder's exact range; typed conversion checks range/precision | `test_json_precision`, `test_typed_arrow`, installed base/speed checks |
| Concurrent metadata or mutable operations could race | Per-key initialization, write barriers, shared ownership locks and private worker sessions; no shared Query mutation during execution | `test_metadata_concurrency`, `test_worker_ownership`, `test_mutation_ownership`, `test_prepared_execution` |
| Native thread multiplication | Stage admission is shared across runtimes; child transforms constrain native engines; effective pools are reported | `test_resource_runtime`, `test_resource_integration`, `test_parquet_transforms`, installed regressions |
| Silent schema/data loss in analytical bridges | Safe typed conversion, lossless cast checks and explicit rejection at unsupported boundaries | `test_schema_evolution`, `test_parquet_writer`, `test_query_parquet`, new Enum/Categorical boundary tests |

The dependency update required a stale expected-version assertion to be updated;
the expanded installed regressions also exposed a fixture-import assumption in
a subprocess test. Both were corrected and rerun. No runtime workaround was
introduced to make the upgrade pass. Lint and strict documentation builds passed.

## Live service coverage

[live.parquet](live.parquet) records fresh read-only checks. Each successful
profile compared three ordered serial rows with parallel rows, a Parquet export,
DuckDB/Polars values and Arrow reader values. Two HTTP workers were used, with
4-second connect/8-second read timeouts and no retries. This is not a load test.

| Mine | Native / legacy | Global templates parsed per profile | One template execution per profile |
| --- | --- | ---: | --- |
| LegumeMine | Pass / pass | 14 | Pass, 0 rows |
| MaizeMine | Pass / pass | 73 | Pass, 3 rows |
| ThaleMine | Pass / pass | 11 | Pass, 0 rows |
| WheatMine | Unverified | — | Anti-bot HTML at version endpoint |
| OakMine | Unverified | — | Anti-bot HTML at version endpoint |

There are 98 discovered global templates across the three successful mines,
parsed under each profile. Parsing is not execution of all templates. Neither
private data nor authenticated live writes nor a full remote dataset was tested.

## Storage and time results

The [documentation tables](../../source/storage_performance.rst) retain both
measured observations and conditional job estimates. The benchmark ran 168 timed
trials: two eight-column fixtures × two measured sizes × six stages × seven
repetitions. All encoded values and aggregate results were checked; observations,
source/environment hashes and 24 job scenarios are Parquet, queried with
[analysis.duckdb](storage/analysis.duckdb).

| Diverse fixture, 1,000,000 rows | Size | Saving with Parquet |
| --- | ---: | ---: |
| CSV / TSV | 176.267 MB | 61.7% |
| gzip CSV | 85.092 MB | 20.7% |
| JSON row arrays | 186.838 MB | 63.9% |
| JSON objects | 322.838 MB | 79.1% |
| gzip JSON objects | 93.604 MB | 27.9% |
| intermine314 Zstandard Parquet | 67.453 MB | — |

At that size, median writing took 1.4921 s for CSV versus 1.1551 s for package
Parquet. A selective two-column aggregate took 0.1190 s for CSV/DuckDB versus
0.0181 s for package Parquet/DuckDB: **6.6× faster for that query**. At 100,000
diverse rows, writing plus one CSV/DuckDB analysis was 0.021 s faster than
Parquet, so small one-off jobs do not universally improve.

Assuming equal acquisition at 100,000 rows/s and ten analyses, the million-row
complete-job estimate saves **1.35 s (10.6%)** against CSV/DuckDB or **10.47 s
(48.0%)** against CSV/Python. At 100 million rows the linear projections are
**134.66 s** and **1,046.55 s**, respectively; these large jobs were not measured.
The docs include 100k/1m/10m/100m jobs, one/ten analyses, the formula and all
assumptions. The evidence catalog also includes 100-analysis scenarios.

These are workflow comparisons, not a measured original-client HTTP speedup.
The original client can stream text directly, and both clients can benefit from
server compression. Converting existing archives, cold disk/cloud storage,
full-column scans, pandas workloads and live server throughput remain unmeasured.
The 20-core 265K host exposed only 18 CPUs to WSL2; the run used native budget 16,
Polars 12, Arrow CPU 4/I/O 2 and DuckDB SQL 16 with stage admission.

## Release handoff

1. Make the separate version change in `pyproject.toml` and `_version.py` to
   0.2.0, and update the README's development-status wording for publication.
2. Build from that clean committed master revision and rerun the existing
   shared CI, minimum dependencies, installed verification and strict docs.
   The temporary candidate confirms the version is accepted by the tooling;
   it does not replace verification of the final release commit/artifact hashes.
3. Confirm the existing PyPI trusted publisher matches this repository,
   `publish-pypi.yml` and the `pypi` environment. The published 0.1.8 attestation
   proves historical trusted publishing, not the current external setting.
4. On authorization to release, push the matching `v0.2.0` tag from validated
   master. The workflow must publish only its verified immutable artifacts.
   Verify the resulting PyPI wheel in a new environment after upload.

Remaining support gaps are documented above and should stay in the release
notes. They do not justify claiming universal drop-in parity or a universal
storage/speed multiplier. The next action is a version-and-release change,
not another implementation cycle for the already completed performance work.
