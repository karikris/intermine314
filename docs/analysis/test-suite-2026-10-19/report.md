# Test-suite consolidation review

Reviewed the 0.2.0 tree at `7a1ec32333fd79238f2e93957cdfaeac807f29eb` on
19 October 2026. This review inventories all test files, collects every case,
measures local durations and per-case Python coverage, examines assertion and
helper structure, and checks the existing historical API trace. **No tests,
runtime code, package dependencies or CI policy were changed.**

## Recommendation

Start with **68 parameter cases across five functions**, then consolidate shared
helpers and contract assertions. A trial with those cases deselected passed
**2,745 tests**, retaining exactly the full run's executed Python lines and branch
arcs. The existing API call trace also retains every previously exercised
historical symbol. This is a credible first reduction of **2.4% of cases**, with
documented conditions before deleting anything.

The suite is already fairly dense: the structural scan found **no identical
top-level test definitions** after normalizing function names, and only one exact
duplicate fixture definition. Large blanket deletions would discard distinct
assertions. More substantial source-code reduction should come from shared
builders and removing repeated structural checks, rather than treating every
overlapping execution trace as a redundant test.

## Measured baseline

| Metric | Observation |
|---|---:|
| Test files | 84 |
| Test functions | 861 |
| Collected cases | 2,813 |
| Physical test-module lines | 17,943 |
| Uninstrumented full run | 2,813 passed; 19.80 seconds including runner overhead |
| Coverage run | 2,813 passed; 30.96 seconds |
| Reduced coverage trial | 2,745 passed, 68 deselected; 30.19 seconds |
| Measured statement coverage | 8,176 / 8,838 = **92.51%**, unchanged |
| Measured branch coverage | 2,563 / 2,996 = **85.55%**, unchanged |
| Cases sharing another case's call-phase coverage set | 1,183 |
| Cases with no measured runtime call-phase points | 136 |
| Cases contributing a unique call-phase point | 209 |

Timings are individual local runs, not a repeated speed benchmark. The removed
cases account for **0.367 seconds** of the baseline JUnit durations; do not
translate the case-count reduction into a large runtime gain. Fixture
consolidation, subprocess setup and CI repetition offer separate opportunities.

Coverage is for `src/intermine314` in the main test process and its threads.
Spawned child processes, independent cold-import/CLI subprocesses, native engine
internals and the `scripts`/`benchmarks` trees are outside that measurement.
Some subprocesses deliberately use isolated Python. A test with zero measured
runtime points can still verify packaging, signatures, release policy or
subprocess behavior. **None of the 136 is automatically removable.**

Call-context fingerprints are evidence of execution overlap, not assertion
equivalence. Lifecycle teardown, message text, values, default arguments,
mutability, request counts and ownership can differ even when code coverage
is identical. Setup and teardown contexts were included in the proposed
reduction's coverage projection, and the actual reduced run was compared
file-by-file against the full run: **no executed lines or branch arcs were lost
or gained**.

## First reduction: exact candidates and retained coverage

The full retained/deselected node lists and parameters are in `cases.parquet`.
`analyze.py::proposed_drop` defines the selection without editing tests.

| Test function | Current → proposed cases | Consolidation | Must remain |
|---|---:|---|---|
| `test_parallel_page_validation.py::test_invalid_terminal_status_never_releases_full_page` | 48 → 24 | Run the full 6 row modes × 4 invalid footers under native; remove its second profile dimension | Both profiles remain in the separate full-page failure test; all status/footer forms and row conversions remain |
| `test_parallel_page_validation.py::test_successful_empty_short_and_exact_pages_are_closed_once` | 36 → 18 | Keep all 6 modes × empty/short/exact sizes once | Every size boundary, mode and close assertion; explicit legacy/default-profile contracts elsewhere |
| `test_query_summaries.py::test_summary_forces_raw_jsonrows_before_object_augmentation` | 32 → 18 | Keep every mode under native on both protocol versions; retain legacy's default mode on both versions | All 8 mode overrides, both wire versions and both facade defaults |
| `test_copy_profiles.py::test_copy_uses_one_profiled_connection_and_correct_order_policy` | 20 → 11 | Exercise all 4 order/layout combinations at cpu2 and cpu16; use ordered single-file representatives for cpu4/6/8 | All five profile values, both order policies, both layouts; exhaustive stage-allocation unit tests |
| `test_resource_integration.py::test_profile_reaches_owned_duckdb_and_polars_pipeline` | 5 → 2 | Keep cpu2 and cpu16 end-to-end; intermediate allocations are checked by unit tests and the COPY representatives | Minimum/default budgets, SQL setting propagation, exact output and released reservations |
| **Total** | **141 → 73** | **68 fewer cases; full suite 2,813 → 2,745** | **Same observed lines/branches and historical call-symbol reachability** |

The first two functions pass explicit row modes, so profile defaults are not the
behavior under test. The summary function deliberately forces `jsonrows` before
object augmentation. These are stronger consolidation candidates than tests
whose expected result depends on native/legacy defaults.

There is an audit dependency: **24 proposed removals occur in the current
795-node semantic test catalogue**. Before implementation is accepted, regenerate
the scoped audit and revise selectors where necessary. Keep all **460 historical
symbols**, **331 passing scopes**, **129 tested departures in 43 groups**, and
the supplemental **40-binding** contract accounted for. Call reachability alone
does not prove that the retained assertions satisfy each scope.

The trial did not perform mutation testing or revalidate every installation mode.
Before deleting the cases, use targeted faults to ensure that retained tests
still reject acceptance of a false footer, premature reservation release,
incorrect COPY thread/order settings and wrong summary wire format. Then run the
updated audit and installed-wheel regressions.

## Further consolidation opportunities

| Priority | Sections | Suggested change | What can shrink | Boundary |
|---|---|---|---|---|
| High | Shared helpers currently imported from test modules | Move builders into small fixture/support modules for query results, Parquet queries, list clients, jobs and release artifacts | **21 cross-test import statements across 17 consuming files / 10 providers** become ordinary support imports; less setup duplication and accidental coupling | Preserve function-scoped mutable Service, Query, Session, Model, cache and output state |
| High | `test_minimal_contract.py`, `test_minimal_surface.py`, structural parts of `test_runtime_contracts.py` | Use one rich public-contract source for exports, signatures, defaults, parameter kinds, aliases, removed modules and fields | The first two files contain 291 lines / 25 test functions, plus overlapping runtime checks; centralize repeated symbol-resolution/introspection loops | Existing minimal contract only checks some signatures as subsequences: exact defaults, ordering and keyword-only checks must migrate too |
| High | `test_query_parallel_offset.py` versus `test_parallel_cleanup.py` and `test_parallel_memory_bounds.py` | Move unique executor-entry/exit and late-failure page-index/offset assertions into the stronger scheduler tests | Conditional retirement of the old **99-line, two-test module**, including its immediate-completion executor | Its fake executor decrements pending work before `submit` returns, so `max_pending <= 2` does not establish a real inflight bound; keep the delayed-page and queued-work tests |
| Medium | Lists, eager results, summaries, templates and plotting helpers | Standardize payload/parameter/response-close utilities; keep scenario-specific transport subclasses | Fewer repeated builders and closure assertion helpers, with existing case IDs retained | Do not replace assertions about exactly-once close and caller ownership with a weaker `closed` boolean |
| Medium | Cold-import/lazy-dependency checks across 14 functions | Share subprocess harness and heavy-import assertion logic; express entry points/actions as named rows | These function bodies occupy 302 lines; only common setup should shrink | Keep each entry point cold and isolated. Combining all imports in one process can hide order-dependent eager imports |
| Medium | Compression adapter and real HTTP tests | Keep exhaustive codecs/read sizes at adapter level; review codec × facade × proxy integration combinations separately | Candidate reduction in repeated end-to-end setup after interaction review | Sized decode, truncated/corrupt bodies, Unicode boundaries, borrowed ownership and real HTTP/proxy behavior are distinct; keep representatives of each |
| Medium | Export/precision integration matrices | Keep exhaustive dtype/range/schema tests near conversion and writer boundaries; test facade wiring with selected end-to-end cases | Additional profile dimensions may be removable, but **not yet validated here** | Preserve old/new wire formats, exact integers/decimals, null-first batches, schema evolution and single-file/directory differences |
| Medium | `test_identifier_stream_upload.py` HTTP server | Reuse a lifecycle helper or explicitly shorten shutdown polling, preserving the custom handler | Both status cases take about 0.52 s and create a server with default polling; reduce fixture overhead without deleting either outcome | Existing common loopback helper uses a 0.01 s poll but lacks the same custom status/handler behavior; extend it carefully or keep a small wrapper |
| Lower | Benchmark/report/tooling tests | Group them under a tooling marker/directory and run once in the appropriate CI job | Clearer suite ownership and less repeated execution in optional-dependency jobs | Keep parser/runner, source-input integrity, evidence relocation and release-policy checks; moving them is not deletion |

The 14 cold-import function bodies total 302 lines, but those bodies also contain
unique actions and assertions. This is an upper bound on their source footprint,
**not** a claim that all 302 lines can be removed. Likewise, moving a helper
reduces coupling but does not inherently reduce the suite's collected case count.

The exact structural duplicate found is the two-line `model` fixture in
`test_model.py` and `test_query_xml_loading.py`. A shared factory can replace it;
the direct line saving is tiny. Most other helpers are similar rather than
identical and require semantic review.

## Central behaviors to protect

| Repository area | Coverage that consolidation must preserve |
|---|---|
| Model/query DSL, constraints, logic and XML | Both profiles where semantics differ; root inference, validation, alias bindings, XML round trips, escaping, clone isolation and template adjustments |
| Result streams and objects | Versions 7/8, supported formats, object cardinality/identity, nested collections, unsized streams, parser customization, shared immutable layouts and direct-constructor mutability |
| Transport and metadata | Read-safe retries versus mutations, borrowed/owned lifetime, auth/proxy configuration, gzip/deflate/zstd, partial reads, parser failures, cache initialization/invalidation and close races |
| Parallel pagination | False/missing/malformed status, overproducing/null rows, ordered stalls, byte/page reservations, oversized/unestimable pages, queued cancellation, running-worker completion and interrupts |
| Arrow/Parquet/DuckDB/Polars | Exact numeric values and dtypes, timestamp restrictions, schema evolution, column order, empty output, row groups/rotation, manifests, single-part promotion/fallback and literal filesystem paths |
| Publication and mutable resources | Old output survives producer/COPY/rename errors and KeyboardInterrupt; staging cleanup; one owner per target/list; source/output collision rejection |
| Lists, enrichment and identifier jobs | Exact upload bytes/length, zero mutation on late validation failure, one mutation attempt, server list identity, tags/operations, temporary-list cleanup, polling/cancellation semantics |
| Resources and processes | All stage allocation formulas, budget/affinity overrides, preinitialized pools, shared admission, spawn-safe callbacks, bounded submission, crash/timeout/cancellation and worker reaping |
| Packaging and release | Dependency absence/presence, lazy imports, native/legacy installed bindings, module origins, wheel/sdist contents, version/license agreement, exact tagged artifact hashes and release-event rejection |

Do not replace unit + transport + end-to-end tests with one enormous scenario.
Those layers catch different defects even when they exercise the same methods.
Keep named parameter rows for independent failure modes rather than hiding them
in an unlabelled loop that stops on the first failed assertion.

## Coverage limits that constrain aggressive reduction

Measured branch coverage is strongest in model descriptors (94.6%), pagination
(90.6%), inflight accounting (90.5%) and typed Arrow conversion (90.3%). It is
lower in native runtime admission (76.2%) and HTTP admission (69.2%).
Configuration is weaker: `config/loader.py` has **31.6%** branch coverage and
`config/runtime_defaults.py` **40.9%**. `parallel/helpers.py` has no measured
main-process coverage. Treat these as review targets before removing broad
configuration/runtime tests; subprocess exclusions can explain some gaps but
must be checked rather than assumed.

The CLI entry point and spawned worker portions are not reliably described by
this main-process report. Dedicated subprocess/process tests remain mandatory.
Native code is verified through values, schemas, settings and failure outcomes,
not by Python branch counts. No test mutation score or comprehensive native-code
coverage was measured.

## CI repetition is a separate opportunity

The latest successful pre-review CI run
[39236763865](https://github.com/karikris/intermine314/actions/runs/39236763865)
reports 2,813 case entries in each of base, speed, plots and combined modes:
**11,252 entries, including 80 skips**, plus the separate **904 installed-wheel
regressions**. The four reported pytest runtimes sum to **194.389 runner
seconds**; jobs run in parallel, so that is not wall-clock latency.

For ordinary PRs, consider full base and combined runs, with targeted speed-only
and plots-only checks for their unique dependency boundaries. Keep the full
four-mode matrix, minimum-dependency distributions and installed-wheel checks
for releases and scheduled validation until a focused matrix is demonstrated.
This reduces repeated work without deleting semantic tests. Its actual savings
need measurement; distribution jobs may still determine CI elapsed time.

Do not drop installed-wheel regressions merely because the same tests passed
against source. They check package contents, resolution and module origins that
source execution cannot establish.

## Suggested implementation sequence and acceptance

1. **Consolidate support code.** Move shared builders out of test modules;
   retain fresh mutable instances and cold subprocess isolation. Keep collection
   IDs initially. Require unchanged assertions, full suite, lint and installed
   regressions. Update the installed regression file list if modules move.
2. **Apply the 68-case matrix reduction.** Keep the exact retained cases
   described above. Regenerate audit selectors/catalogue and source-input
   hashes. Require equal observed line/branch sets, unchanged historical scope
   accounting, targeted fault-detection checks and all dependency modes.
3. **Consolidate contract checks and older scheduler scaffolding.** Transfer
   each unique assertion into its new owner before deleting the old definition.
   Verify parameter defaults/kinds, exception identity and late-failure metadata;
   preserve descriptive failure IDs. Then trial any further export/compression
   matrix reduction independently.
4. **Reduce CI repetition after suite consolidation.** Record the focused mode
   selection and demonstrate dependency-boundary failures remain detectable.
   Keep release validation complete. Compare runner time and wall-clock time
   separately before claiming a speed gain.

A useful acceptance rule is: **remove duplicate setup and independent matrix
dimensions, preserve every distinct assertion obligation and failure outcome**.
Use the current observed coverage as a regression floor, not as evidence that
the suite already covers every central behavior.

## Evidence and reproduction

Canonical tables are Zstandard Parquet: `inventory`, `cases`, `parameter_groups`,
`exact_duplicates`, `coverage_files`, `case_coverage`, `coverage_losses`,
`verification` and `ci`. The local DuckDB catalog exposes views over them.
Coverage tools were installed outside the package environment. GitHits verified
coverage.py 7.16.2 and the official context API; pytest documentation confirms
stacked parametrization creates Cartesian products.

```bash
uv pip install --target /tmp/intermine314-test-review-tools 'coverage==7.16.2'
.venv/bin/python docs/analysis/test-suite-2026-10-19/run.py baseline
.venv/bin/python docs/analysis/test-suite-2026-10-19/run.py coverage
.venv/bin/python -m scripts.audit_api_coverage --trace /tmp/intermine314-test-review/api-trace.json
.venv/bin/python docs/analysis/test-suite-2026-10-19/analyze.py /tmp/intermine314-test-review
.venv/bin/python docs/analysis/test-suite-2026-10-19/run.py trial
.venv/bin/python docs/analysis/test-suite-2026-10-19/analyze.py /tmp/intermine314-test-review
.venv/bin/python docs/analysis/test-suite-2026-10-19/catalog.py
```

The recorded run reused the existing source-hash-matching release API trace;
the command above produces a fresh one. Outputs are observations for this
checkout and local environment. Thread scheduling and context attribution can
vary between runs; compare actual full and reduced coverage files, not just
aggregate percentages. The initial measurement wrapper was corrected to guard
its entry point before spawn-based tests; only the subsequent passing runs are
used here.

Official methodology references, retrieved through GitHits:

- [pytest stacked parametrization](https://github.com/pytest-dev/pytest/blob/cf470ec0bf7eb89cd97dd56df4859eae5db46447/doc/en/how-to/parametrize.rst#L195).
- [coverage.py per-test contexts](https://github.com/coveragepy/coveragepy/blob/5643ae7c682bbd56ee1ac2b11556145ab8c2cab0/doc/contexts.rst#L59).
- [coverage.py process measurement requirements](https://github.com/coveragepy/coveragepy/blob/5643ae7c682bbd56ee1ac2b11556145ab8c2cab0/doc/subprocess.rst#L11).
