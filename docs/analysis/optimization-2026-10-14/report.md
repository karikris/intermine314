# A smaller, faster InterMine compatibility layer

Audit date: 14 October 2026. Runtime and test baseline: `b9b7472efa48d74db159ca78e9c5666c015b7210`, package 0.1.8. This audit adds evidence and recommendations; it changes no package implementation, dependency, version, or test contract.

The next useful work is **removing repeated execution, conversion, and bookkeeping inside the existing architecture**. A second query engine, another general threadpool, a TypeScript processing layer, or a custom Rust extension would increase the code and support burden without addressing the clearest remaining costs. Keep the public legacy names and signatures as small adapters over one transport, one execution preparation path, and one Parquet/Arrow/DuckDB pipeline.

Eight isolated prototype comparisons produced 224 timed trials. All 2,769 current tests passed in 20.76 seconds. The strongest new evidence is shared row lookup metadata, per-stream object parser binding, repeated template cloning, batch-level native admission, and avoiding an unnecessary single-part Parquet rewrite. These are opportunities, **not shipped speedups**. See [measured operations](measurements.md), [reproduction code](probe.py), and the [DuckDB evidence catalog](catalog.py).

## Scope and what the size numbers mean

The AST inventory covers every runtime Python file and every Python file under `tests`, with hashes, declarations, line counts, and exact duplicate bodies. Source inspection followed the legacy factory → query/template → HTTP/parser → rows/objects route, and the export → Arrow → Parquet → DuckDB route, including resource admission, metadata, mutation ownership, and error cleanup. This is a code and fixture audit, not a fresh certification of every live mine or authenticated mutation.

| Area | Python files | Physical lines | Assessment |
|---|---:|---:|---|
| Query | 10 | 4,131 | Largest concentration: builder, validation, cloning, serializers, templates, bounded pagination. Consolidate preparation; preserve operator and override contracts. |
| Service | 12 | 2,907 | Row parsing and resource policy are the main remaining CPU candidates. Sessions, decompression, retries, metadata locks and response lifetimes already share implementations. |
| Export | 15 | 2,465 | Optimize staging/conversion boundaries and admission granularity. Preserve exact types and atomic publication. |
| Root modules | 18 | 2,182 | Model, result objects, registry/account helpers, plots and legacy facades. Facades are compatibility assets; object construction has measurable avoidable work. |
| Configuration | 6 | 1,003 | Mostly cold validation/fallback logic. One unused parser; no evidence that replacing the entire configuration system helps throughput. |
| Lists | 6 | 909 | Keep mutation semantics and known-length validated uploads; reduce per-identifier Python/write calls. |
| Utilities | 7 | 374 | Keep one precision-preserving JSON implementation and one ownership implementation. |
| Parallel helpers | 3 | 161 | Small public surface; absence of internal callers does not establish that public helpers are obsolete. |
| **Runtime total** | **77** | **14,132** | Includes 1,120 docstring lines and 987 function declarations; physical LOC is not executable complexity. |
| Tests, including Python fixtures | 90 | 17,991 | 2,769 collected cases; tests are excluded from the built package. Cutting tests does not improve installed runtime performance. |

Exact AST-body matching, ignoring docstrings and requiring at least 20 AST nodes, found **five runtime duplicate groups, 17 member functions, 85 physical lines**. Most are lazy-module dispatch, `__dir__`, response close, and operator aliases. That is only 0.6% of runtime lines *including the copies that must remain*. This is not evidence for deleting thousands of lines. Smaller or semantically equivalent bodies are outside that detector's scope.

## Import-only replacement: preserve the correct boundary

The ordinary migration remains:

```python
# Before: from intermine.webservice import Service
from intermine314.webservice import Service

service = Service("https://your-mine.example/service")
query = service.new_query("Gene")
query.add_view("Gene.symbol")
for row in query.rows():
    print(row["Gene.symbol"])
```

`intermine314.webservice.Service` selects the legacy profile. `intermine314.service.Service` selects the native profile; moving between those module paths is a behavioral change, not just replacing the package name. Do not make native dictionary defaults leak into the legacy route.

The existing audited inventory has **333 passing-scoped symbols and 127 symbols with tested departures**, grouped into 39 departure explanations, against original client 1.13.0. Those are symbol counts, not a percentage of real user workloads; 127 does not mean 127 missing functions. The runtime hashes remain unchanged in this audit. The [full compatibility inventory](../intermine-api-final-coverage.json) and [release coverage matrix](../release-2026-10-14/report.md) remain the authoritative detailed mappings.

| Legacy workflow | Keep import-only behavior where covered | Boundary that must remain explicit |
|---|---|---|
| Service/query creation, views, constraints, logic, joins, sorting, model/Column DSL | Existing names, positional/keyword arguments, aliases, XML/wire forms and model descriptors | Repaired validation/logic errors differ from historical bugs; custom subclasses and duck-typed services need their dispatch paths. |
| `rows`, result formats, `first`, `one`, `all`, summaries | Legacy ResultRows/objects, path/index access, old-server cell metadata, explicit CSV/TSV formats and eager helper shapes | Do not change result types or copy/mutation behavior to save allocations. Lazy object references may make additional requests. |
| Templates | Named forms, editable constraints, adjustments, rows/count/eager helpers | Eliminate repeated preparation without applying adjustments twice or mutating the template. |
| Lists, tags, enrichment, set operations, account helpers, identifier jobs | Existing methods and validated wire contracts | Service close/destruction deliberately does not delete server lists. Use ListManager cleanup or `flush`; authenticated live equivalence is not certified. |
| `dataframe()` | Function entry point remains available | Returns **Polars**, not pandas. Indexing and pandas-specific downstream code require changes. Making this transparently pandas would undermine the chosen pipeline and add a dependency. |
| Analytics and storage | Existing Parquet exports, DuckDB SQL and Arrow readers | Eager results still materialize; unsupported precision/type bridges must reject rather than silently coerce. |
| HTTP/resource management | Shared transport, explicit ownership, Python 3.14 operation | Server limits, moving offset datasets and external native pools are not solved by replacing imports. |

## Ranked implementation opportunities

“High” means the repeated work is directly observed or measured, not that a large whole-job speedup is guaranteed. Estimates are deliberately omitted where no prototype was measured. Each replacement should remove its superseded internal path in the same task; keep a fallback only when it serves a different supported input or extension contract.

| ID / priority | Code section | Change and work removed | Evidence / likely benefit | Compatibility and acceptance conditions |
|---|---|---|---|---|
| O01 / P1 | `service/session.py:227–244`, ResultRow | Build one lookup plan per stream, share full/root-stripped path indexes across built-in rows. Remove per-row map/string reconstruction. | **High.** 20k keyed row constructions: 33.213 → 2.279 ms, 14.57× for this operation. Retaining 10k rows/20 columns: 18.07 → 1.05 MB of traced Python allocations. | Preserve duplicate/alias resolution, table rows and public constructors. Keep mutable `views`/`index_map` semantics for callers; use an immutable internal plan with copy-on-write exposure or an explicitly internal fast path. The prototype shares a dict and is not a complete compatibility implementation. |
| O02 / P1 | `service/session.py:375`, `results.py:53`, `_result_object.py:25` | Resolve the result class/parser once per stream; compile selected attributes/reference-path metadata once. Remove per-row lazy-facade imports and repeated view splitting. | **High.** Binding alone: 18.095 → 9.128 ms per 20k objects, 1.98×. Shared view-plan gains are unmeasured. | Resolve after caller customization and preserve subclass/monkeypatch dispatch, nested identity maps, type resolution and lazy fetches. Do not globally memoize facade exports without testing dynamic binding behavior. |
| O03 / P1 | `service/session.py:458`, list parser | For the exact built-in modern row implementation, construct the output list directly. Remove throwaway ResultRow allocation. | **High locally.** 20k conversions: 1.844 → 0.919 ms, 2.01×. Absolute savings are small beside slow HTTP. | Preserve independent list copies; old servers unwrap cell values; subclasses overriding `row`/`to_l` retain their path. |
| O04 / P1 | `service/session.py:537–661`, `util/json.py:60` | Keep row payloads as UTF-8 bytes through fast JSON decoding; bind the row decoder once. Remove bytes→str→bytes round trips. | **Medium/high with speed extra.** 20k array decodes: 10.067 → 8.854 ms, 1.14×. Whole stream/header framing was not replaced in the prototype. | Keep text/file-like sources, exact integers beyond UInt64, Decimal paths, malformed UTF-8/error types, capped status parsing and unsuccessful footer detection. Stdlib backend needs separate measurement. |
| O05 / P1 | `query/template.py:122–187`, `query/builder.py:1218,1391,2207` | Use one execution-scoped adjusted snapshot. Eager/row helpers should delegate once instead of re-entering Template adjustment. Remove nested clone/validation loops. | **High confidence; timing unmeasured.** One fixture call performs 2 clones for `results`, 3 for `get_results_list`/`rows`, **4 for `get_row_list`**. | Preserve independent public `clone()`, constraint/logic rebinding, adjustments, validation order, custom overrides and the original template. No persistent cache keyed only by mutable Query identity. |
| O06 / P1 | `export/arrow.py:116`, `export/runtime.py:128–176`, `export/resource_profile.py:39` | Reserve native CPU once for a homogeneous Arrow batch; resolve stage allocations at operation/runtime boundaries. Remove one engine lookup, quota read and admission round trip per column. | **High for wide/small batches.** 1k×64 integer batch: 6.664 → 5.209 ms, 1.28×. 1k standalone worker lookups cost 7.024 ms; a cached integer loop is only a bookkeeping lower bound. | Release Arrow reservation before a Polars fallback; different native reservations cannot nest. Refresh policy for new operations/explicit resource changes, retain effective-pool diagnostics and cancellation. `get_runtime()` already has a fast active-context path; do not add a second global runtime cache. |
| O07 / P1 | `export/parquet.py:459–477`, `write_single_parquet_from_parts` | Promote one already suitable, validated staged part directly. Remove redundant DuckDB connection, scan, decompression and recompression. | **High for small outputs.** 10k String/Int64 rows: 14.895 → 1.935 ms, 7.70× for complete local write/publish. Large multi-part jobs need a different measurement. | Eligibility must prove final schema, supported logical types, compression, row-group geometry and order. Complete source/footer validation first; preserve same-filesystem publication, rollback and cross-device copy. Do not bypass existing single-file type guards for Enum/Null/timezone/nested types. |
| O08 / P2 | `service/session.py:401`, `export/arrow.py:116`, `export/parquet.py:128` | Add one internal typed-column batch path from validated flat row arrays. Keep public dict/ResultRow materialization at the public boundary. Remove dict-per-row → column-list reconstruction for built-in exports. | **High potential for CPU/RSS; unmeasured.** Current path traverses selected fields several times. More invasive than O01–07. | Preserve row cardinality/schema checks, exact model widths, Decimal scale, nullable/nested fallback, custom `iter_rows`/`iter_batches`, footer validation and page byte charging through conversion. Replace the old built-in path immediately; do not keep two generic converters. |
| O09 / P2 | `export/arrow.py:61`, `export/model_schema.py:16` | Compile an execution-local schema/converter plan: selected columns, expected types, Arrow types, Decimal/date handlers. Remove repeated type-map and schema-plan construction per batch/column. | **Medium/high for wide batches; unmeasured.** Complements O06/O08; their gains cannot be multiplied. | Preserve strict mixed-type fallbacks and model-specific Float32 rounding/overflow. Schema identity/scale changes invalidate the relevant plan. |
| O10 / P2 | `query/executor.py:10–27,84–129`, builder `count/results` | Keep one prepared wire/count implementation; store invariant parameter-key membership alongside the encoded prefix. Move custom-service fallback into a thin adapter. | **Medium for many tiny queries/pages; unmeasured.** Prepared parallel execution already exists, so do not count its existing gains again. | Retain mutated-parameter fallback, template form encoding, summary paths, error translation and subclass count/results overrides. Avoid a generic serializer framework. |
| O11 / P2 | `lists/_csv.py:53–66` | Validate/quote identifiers in bounded Arrow batches, write one bounded byte chunk per batch rather than up to two file writes per identifier. Native string operations are an optional measured follow-up. | **Medium for million-identifier local inputs; unmeasured.** Fewer Python calls; uploaded bytes need not change. | Keep exact quote escaping, UTF-8, leading zeroes, empty/null/newline rejection, known Content-Length, one mutation attempt and validation before network mutation. Never join the entire dataset. |
| O12 / P2 | `query/builder.py:1703`, `export/query.py:52,155` | Prefer a compact Parquet directory or O07's fast single file for temporary dataframe staging; reuse an owned DuckDB context for many queries over the same dataset. | **Medium/high for repeated tiny analyses; unmeasured.** Removes repeated compaction/connection setup, not the required storage pipeline. | Keep Parquet as canonical storage and DuckDB as SQL engine. Preserve per-query views, borrowed connection settings/lifetimes, file fingerprint invalidation and detached eager results. Existing Arrow reader already accepts a borrowed connection; use it before adding another API. |
| O13 / P2 | `_result_object.py:108`, `_prefetch.py:91`, metadata/registry | Expose/document bounded prefetch at collection boundaries; share one service per mine/lifetime and reuse initialized metadata. | **High only when N+1 requests dominate.** These capabilities largely already exist; this is workflow adoption, not a new executor. | No automatic prefetch on arbitrary attribute access: it changes requests, memory, ordering and failures. Collections can be large; retain explicit field/window controls. |
| O14 / P3 | `query/parallel_runtime.py:19`, JSON row loop | Count/instrument at page boundaries for built-in pipelines; use lazy logger formatting and stream-local sentinels. Remove redundant per-row bookkeeping where diagnostics are disabled. | **Low/medium CPU-bound; unmeasured.** Export already uses page mode. | Public row iterators still need early-close/error ownership and useful error metrics. A generator-return shortcut must not leak its source. |
| O15 / P3 | `model.py:122`, builder `constraints/coded_constraints`, service registry setup | Remove obvious temporary lists before `sorted` and before local dict iteration where snapshot semantics are unnecessary. | **Low.** Small allocations; no useful total-job estimate. | Do not replace public value snapshots with live views. `ResultRow.__iter__` deliberately copies: mutation after `iter(row)` is an observable counterexample. |
| O16 / P3 | `export/query.py:116` | Evaluate delegating `ParquetReader.read_all()` to the owned Arrow reader under one native reservation. Remove Python batch accumulation if the native lifetime/error contract matches. | **Low/medium for many tiny batches; unmeasured.** It remains an eager, memory-materializing API. | Close on failures/interrupts, retain reader locking/runtime cancellation and precise output schema. Do not claim bounded memory for `read_all()`. |
| O17 / P3 | `export/transforms.py:54–94` | Separate worker initialization from per-task validation; reuse imports/callback resolution/runtime only where a callback cannot invalidate required settings. | **Medium for many short coarse transforms; unmeasured.** Otherwise native SQL/Polars plans are preferable to process tasks. | User callbacks can mutate native settings. Reset or validate those per task, preserve spawn-before-import limits, path-only IPC, timeout/reaping and atomic final publication. |
| O18 / P3 | private dead helpers and test setup | Delete confirmed unused private helpers; consolidate fixture boilerplate and overlapping surface assertions. | **High confidence for small size reduction; negligible runtime gain.** Details below. | Preserve public names, semantic scenarios and audit selector mappings. No test deletion quota. |

## An inherited extra-request trap, with an existing compatible solution

`list(query.results("list"))` opens **two** full result streams in the fixture: Python obtains the iterator, then calls the facade's `__len__` for preallocation; `__len__` iterates a fresh stream. Both responses close, but the extra server work is real. This is inherited from [original InterMine ResultIterator](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/results.py#L375), verified through GitHits.

Use the already compatible `query.get_results_list("list")` / `query.all("list")`, or `list(iter(query.results("list")))`. The package eager helper already deliberately uses a comprehension to avoid the length hint. For early exit, retain explicit stream closure. In-memory 10k-row consumption fell from 12.628 to 6.415 ms, but real transfer savings depend on latency and payload size. `list(query)` also deserves scrutiny because `Query.__len__` issues a count request.

This cannot be fixed completely by adding `__length_hint__`: `__len__` takes precedence. Deleting `__len__` breaks existing `len(results)` calls; materializing/caching all results changes bounded-memory and repeat-stream semantics. Do not present either as an invisible compatibility optimization. First update the repository's own `scripts/verify_distribution.py:121` and serial/template paths in `scripts/verify_live_readonly.py:56,76`; those currently make avoidable extra requests. The parallel iterator in the latter script is already a generator and does not have this particular problem.

## What can actually be retired

| Candidate | Action | Evidence / size |
|---|---|---|
| `query/builder.py:219`, `_polars_from_dicts_with_full_inference` | Delete in the first implementation task. | No repository callers; an old keyword-support fallback no longer participates in the converter. 8 physical declaration/body lines. |
| `config/runtime_defaults.py:290`, `_parse_small_string_list` | Delete in the same cleanup task. | No repository callers; current parsers use their specialized forms. 20 physical declaration/body lines. |
| `service/resource_utils.close_response_quietly` | Optional internal API cleanup after checking export/compatibility policy; no performance priority. | No repository callers, but it lacks a private-name prefix. Two lines. Do not infer safe external removal solely from a reference scan. |
| `close_session_quietly` and duplicated adapter `close` bodies | Keep unless a local refactor demonstrably reduces complexity. | Finalizer/response ownership entry points have distinct responsibilities. Alias assignment can change keyword signatures (`session`/`response` versus `resource`). Saving a few lines is not enough. |
| `export/model_schema.model_batch_frame` | Keep, or update its consumers atomically if explicitly making it private. | No runtime callers, **but four analysis/benchmark scripts and resources documentation use it**. It is already a thin adapter, not a second converter. |
| `json_backend`, `valid_parquet_compressions`, `named_resource_profiles`, `parallel.helpers` | Retain public utility behavior. | Sparse internal usage does not imply an unused public API. |
| Repeated template preparation, row maps, list parser wrappers, single-part COPY | Remove the superseded internal work in O01–O07's implementation commits. | These are larger execution savings than dead-code deletion; most public function names remain. |
| Generic typed/Polars fallback | Retain only the cases with distinct input semantics. | Nested/mixed values, subclasses, unusual model types and large-int Float32 conversion are not redundant with a primitive Arrow fast path. |
| Legacy facades, alias methods and `Service.__getattribute__` | Retain. Optimize callers that repeatedly invoke them. | Facades preserve old imports. The apparently empty legacy override accepts historical `service.__getattribute__(name="root")`; blindly deleting it breaks an audited call. |

The two confirmed private deletions remove **28 physical lines, about 0.20%** of runtime source. Larger reductions require consolidating behavior, not a dead-code sweep. Do not set a speculative 30–50% LOC target. Moving a method into another file or stripping useful docstrings also does not reduce processing work.

A suitably minimal architecture keeps public facades → one execution snapshot → one managed transport/parser. At the result boundary it exposes legacy rows/objects or typed analytical batches. Analytical batches use one validated Parquet writer and one DuckDB/Arrow reader. Existing modules already mostly follow this arrangement; avoid another generic backend/plugin/compatibility framework.

## Tests: reduce repetition, preserve evidence

The exact-body scan found four groups across nine **helper** functions (32 physical lines), and no duplicated complete test bodies meeting the detector threshold. Examples are `capture` in `test_registry_helpers.py`, `execute` in list tests, and `assert_closed` in CRUD/template discovery. Consolidating these can improve maintenance; it does not justify deleting their scenarios.

Concrete cleanup candidates:

1. Consolidate export/signature/alias checks shared across `test_minimal_surface.py` and `test_runtime_contracts.py` into a declarative surface matrix. Keep a separate semantic/wire test for each behavior. Parameterize fixtures without collapsing native/legacy or server-7/server-8 cases.
2. Replace exact development-tool version/config dictionary snapshots in `test_runtime_contracts.py::test_project_tooling_policy_is_strict_and_modernized` with intentional project invariants; the actual Ruff/build/installed-package jobs already execute the tools. Keep Python minimum, required dependencies, licenses, lazy imports and compatibility extra checks where they define package contracts.
3. Remove tests tied solely to a private implementation when that implementation is removed, replacing them with behavior checks only if the contract is not already covered. The two confirmed unused private helpers have no direct tests to delete.
4. Reduce artificial waits through deterministic events/fixtures when safe. The slowest cases include HTTP server shutdown (~0.52 s each), transform cancellation/bounds (~0.51 s) and baseline integration (~0.53 s). These validate real failures and lifetimes; deleting them would trade safety for little total suite time. If a fixture uses a periodic server shutdown poll, shorten that poll rather than removing its request/ownership assertions.

Keep footer-failure/atomic rollback, compressed sized XML reads, stalled ordered pages, cancellation, GC/reentrant metadata, precision/overflow, custom overrides, borrowed lifetimes and installed-wheel tests. They distinguish failure modes that happy-path coverage misses. Do not replace these with snapshots of the optimized code's implementation.

Any renamed/parameterized test node must update `docs/analysis/api-audit-scopes.json` and relevant compatibility/reexport mappings, then rerun the API trace/render audit. The existing ledger links 788 semantic nodes; an unchanged passing count with stale selectors is not proof of compatibility. Test source hashes also change even if runtime files do not.

## Python 3.14: useful improvements and their limits

| Feature | Verified conclusion | Decision |
|---|---|---|
| Tail-call interpreter | This local CPython 3.14.5 build has `Py_TAIL_CALL_INTERP=1`, Clang 22, PGO/LTO/BOLT. Upstream reports a preliminary 3–5% geometric-mean benefit on its benchmark, not this package. | Already present here; do not add that percentage to package measurements. It optimizes C opcode dispatch, not Python recursion. [CPython source](https://github.com/python/cpython/blob/8e6e75d9102e39bed2a2b279203a396741180f12/Doc/whatsnew/3.14.rst#L438). |
| Experimental JIT | Available but disabled by default in this build. Screening with it enabled gave current-operation ratios from about 0.96× to 1.09×. | Keep opt-in; rerun representative jobs and correctness tests before recommending it. Sequential process order and small effects prevent a causal speed claim. [Configuration](https://docs.python.org/3.14/using/configure.html#cmdoption-enable-experimental-jit). |
| Free-threaded Python | Current measurements use a GIL-enabled build. Free threading needs a different interpreter build and compatible extensions. Thread context defaults also differ. | Separate certification project, not an environment toggle for this wheel stack. Recheck GIL state after importing all engines and audit mutable caches/worker context propagation. [CPython changes](https://github.com/python/cpython/blob/8e6e75d9102e39bed2a2b279203a396741180f12/Doc/whatsnew/3.14.rst#L478). |
| `Executor.map(buffersize=...)` | Bounds submitted tasks whose results have not been yielded. | Useful for simple independent count-bounded tasks. It cannot replace byte reservations, native-overlap accounting, unordered completion, footer acceptance or per-job cancellation in `BoundedInflightQueue`. [Executor documentation](https://docs.python.org/3.14/library/concurrent.futures.html#concurrent.futures.Executor.map). |
| `InterpreterPoolExecutor` | Each worker has an isolated interpreter/GIL; arguments and results are serialized. Extension compatibility is distinct from ordinary thread safety. | Consider only measured coarse pure-Python tasks. Do not pass live Service, Polars, Arrow or DuckDB objects. Current spawned path-based transforms remain the supported mechanism. [CPython source](https://github.com/python/cpython/blob/8e6e75d9102e39bed2a2b279203a396741180f12/Doc/library/concurrent.futures.rst#L246). |
| Deferred annotations | Python 3.14 can defer annotation evaluation. The repository already widely uses `from __future__ import annotations`. | No mass edit: removing the future import changes introspection behavior and is not an established throughput win. Measure cold startup separately. [Python documentation](https://docs.python.org/3.14/whatsnew/3.14.html#pep-649-pep-749-deferred-evaluation-of-annotations). |

GitHits verified pinned CPython 3.14.8 documentation/source, original InterMine 1.13.0, orjson 3.13.0, and Polars 2.0.0. Raw responses, including search results that were not sufficient evidence for a claim, are in `research.parquet`. Versions and dependencies remain unchanged from the [completed dependency review](../release-2026-10-14/dependency-review.md).

Adding Python threads around JSON parsing is not a general CPU solution: [orjson's pinned documentation](https://github.com/ijl/orjson/blob/bd00937c5c46ab20eb5d7fe1cfc034b906c4578f/README.md#L596) says its call holds the GIL and recommends direct bytes input. Polars already runs its expression engine in Rust; its [native-expression guidance](https://github.com/pola-rs/polars/blob/22a147de3d2bb2e44b97338a2510816c7105c9f2/py-polars/src/polars/series/series.py#L6314) explains why Python element UDFs are slower and restrict optimization. Use those existing engines before adding extension code. TypeScript offers no demonstrated advantage in this Python data path and would introduce serialization and another runtime.

## Worker budgets and useful concurrency

Keep default total active native budget **16**, with **8, 6, 4, 2** profiles. This WSL environment exposes 18 CPUs; measurements here are not a bare-host optimum for all 20 cores of the user's processor. Existing admission coordinates package-owned native stages; stage counts below are **not additive pools**.

| Stage | CPU16 | CPU8 | CPU6 | CPU4 | CPU2 | Useful next work |
|---|---:|---:|---:|---:|---:|---|
| Public mine HTTP | 4 | 4 | 4 | 4 | 2 | Measure 1/2/4/8 within server policy; reuse sessions and byte-capped pages. More workers will not fix Python parsing. |
| Low-latency HTTP | 2 | 2 | 2 | 2 | 2 | Reduce row/call overhead before increasing workers. |
| High-latency capable server | 8 | 8 | 6 | 4 | 2 | Sweep upward only when server throughput improves without retries or memory growth. |
| Arrow CPU / ordered single-file COPY / spawned Python | 4 | 4 | 3 | 2 | 1 | Reserve once per batch; one native thread per spawned worker. Ordered directory compaction remains one COPY thread. |
| Unordered COPY | 8 | 8 | 6 | 4 | 2 | Only where order is irrelevant; measure RSS and spill bytes. |
| DuckDB SQL | 16 | 8 | 6 | 4 | 2 | One native query, enough row groups, projection/predicate pushdown; avoid an outer query executor. |
| Polars | 12 | 8 | 6 | 4 | 2 | One streaming expression plan; initialize pool before import. |
| Arrow I/O | 2 | 2 | 2 | 2 | 1 | Keep CPU and I/O limits distinct. |
| Independent metadata/mines | 4 | 4 | 4 | 4 | 2 | Overlap independent cold reads; keep one initializer per cache key. |
| Mutation/publish/plots | 1 | 1 | 1 | 1 | 1 | One mutable owner; parallelize independent reads upstream. |

A future bounded two-stage producer/consumer pipeline could overlap HTTP with conversion, but only after profiling shows idle native CPU. Charge validated pages **until the consuming conversion/writer releases them**, including completed out-of-order pages and native overlap. Do not reintroduce the previously fixed buffered-page memory escape by counting only pending futures.

## Implementation order and measurable acceptance

| Phase | Tasks, each independently reviewable | Required handoff before the next phase |
|---|---|---|
| 1: low-risk cleanup and dispatch | Delete the two unused private helpers; remove extra requests in verification scripts; implement O02 parser binding and O03 list conversion; simplify O05 template helper delegation. | Wire/result equivalence in native/legacy and old/new server fixtures; request/clone counts; subclass/custom binding tests; full tests and refreshed API audit. |
| 2: shared execution metadata | Implement O01 row plans; O06 batch admission/stage resolution; O09 converter plan; O10 prepared-wire consolidation. | Memory/throughput measurements on narrow/wide/small/large batches; view mutation/alias checks; quota/pool diagnostics; mixed-schema fallback and exact numeric tests. |
| 3: remove data movement | Implement O07 eligible-part promotion; O08 internal column batches; O11 bounded upload encoding; evaluate O12 repeated SQL reuse. | Whole-job comparison with requests/rows/bytes identical; Parquet schema/order/geometry checks; failed footer preserves old target; cross-device/rename failure, precision and cancellation tests. |
| 4: measured refinements and test maintenance | Adopt only winning O14–O17 candidates; consolidate duplicated fixtures/surface checks; evaluate JIT/free-threading separately if requested. | Keep all unique failure scenarios and public symbols accounted for; installed-wheel regressions, dependency-mode CI and docs; no unverified interpreter-support claim. |

For an implementation run, commit each task to master and push each completed phase as requested. Retire replaced internals within that task, using git history for recovery; no parallel retirement cycle. A phase handoff should contain its exact commit, removed paths, changed contracts (ideally none), test node/audit mapping changes, measured trials and environment, RSS/allocations/requests/bytes/throughput, pending limitations, and a clear pass/fail decision. Do not hand off on a plausible microbenchmark alone.

Track median and tail latency, rows/sec, requests/retries, Python retained allocations, peak process RSS, native pool counts, spill/temp/output bytes, row groups, initialization cost and failure cleanup. Preserve the 16-worker envelope during each run and distinguish warm from cold cache. The present probes measure warm elapsed time and selected Python allocations; they do not measure native peak RSS, power, live-mine throughput or all-host scaling.

For whole-job projections use `speedup = 1 / ((1 - f) + f / s)`, where `f` is the measured fraction spent in the improved operation and `s` its measured operation speedup. Doubling an operation that takes 30% of a job gives about **1.18× overall**, not 2×. Do not multiply the overlapping row, JSON, Arrow and staging ratios in this report.

## Reproduction and evidence

Run from the repository root in the development environment. Probe runs use test fixtures and temporary files, never live services, and do not modify the package. Imports/fixture construction and fsync are outside the timed intervals. Run timing processes sequentially on an otherwise idle machine; Python GC is collected before each timed trial. Equality checks occur outside timing where possible.

```bash
PYTHON_JIT=0 .venv/bin/python docs/analysis/optimization-2026-10-14/probe.py
PYTHON_JIT=1 .venv/bin/python docs/analysis/optimization-2026-10-14/probe.py
.venv/bin/python docs/analysis/optimization-2026-10-14/catalog.py --build
.venv/bin/python docs/analysis/optimization-2026-10-14/catalog.py --validate
.venv/bin/python docs/analysis/optimization-2026-10-14/catalog.py 'SELECT * FROM comparison'
```

`inventory.parquet`, `declarations.parquet` and `duplicate_bodies.parquet` record source inspection. `structural.parquet` records template clone counts and row mutation/alias counterexamples. `trials-*`, `memory-*`, `requests-*`, and `environment-*` record the probes. `tests.parquet` retains the 2,769 successful JUnit cases and durations from this audit; `verification.parquet` records commands, provenance and the original JUnit digest. `research.parquet` retains GitHits/web evidence. `analysis.duckdb` contains relative Parquet views; use `catalog.py` to set its search directory when opening from another location. No second copy of the analysis tables is stored inside DuckDB.
