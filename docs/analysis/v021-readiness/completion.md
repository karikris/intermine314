# Fresh-install compatibility implementation: completion evidence

All six phases implement the approved fresh-install contract. Version declarations
remain **0.2.0**; this is unreleased 0.2.1 preparation, with no tag or PyPI upload.
Every task has a separate master commit, every phase is pushed, and the next
phase starts only after the preceding exact commit passes CI. The closing phase
requires the same exact-head CI gate; its run is linked in the completion message.

## Delivered behavior

| Area | Legacy / historical facade | Native profile |
|---|---|---|
| Configuration | Current packaged defaults, ResourceProfile and ParallelOptions; removed keys rejected | Same |
| Result sizing | `len` and `bool` perform independent validated traversals; `list` can issue two requests | Unsized, one-pass list materialization |
| Metadata and customization | Mutable row/object metadata, active row/view/constructor changes | Shared immutable layouts and stream-open binding |
| Encoding and constraints | Original public UTF-8 helpers and typed collection identity; independent clones | Same public constructors; factories retain string normalization |
| Dataframe | Optional pandas default, Arrow extension dtypes and RangeIndex | Polars default |
| Explicit backend | `backend="pandas"` or `backend="polars"`; invalid/missing backend fails before I/O | Same |
| Internal analytics | Shared Parquet, DuckDB, Arrow and Polars pipeline | Same |

The pandas extra is `pandas>=3.0.6,<4`; it is never a base-install requirement.
Query/Template backend selection, precision, empty schema, native admission and
cleanup are tested. `query_parquet`, `fetch_from_mine` and enrichment analytics
retain their Polars contracts. Public helpers and legacy subclasses reuse one
transport, scheduler, XML encoder and export pipeline. Runtime Python files remain
79; restoring the requested behavior adds 96 lines (14,173 → 14,269) and 4,026 bytes.
This work prioritizes compatibility over claiming a smaller runtime at any cost.

## Acceptance evidence

- **2,805 source tests** pass with the combined extras. The 460-symbol inventory
  contains **338 passing scopes and 122 symbols with tested departures**, across
  42 groups, plus the unchanged 40 supplemental bindings. Counts are scoped
  assertions, not a percentage of complete replacement compatibility.
- [Phase 5 CI](https://github.com/karikris/intermine314/actions/runs/39424738385)
  passed all eight jobs on `81319932456269ddb2acc899bd7cf9707ca41490`: seven fresh
  distribution modes, Python/direct minimum pins, Sphinx and **1,003 installed
  regressions**. Subsequent changes are acceptance tooling, evidence and docs;
  runtime Python files are byte-identical to the installed wheel used by probes.
- Original **InterMine 1.13.0**, installed in a separate interpreter environment,
  matches the legacy profile on the recorded encoding, scalar/collection identity,
  cloning, XML values, length/truth requests, list results, mutable row metadata
  and active constructor probes. Candidate environments contain no original
  `intermine` package. This is a targeted differential check, not its entire suite.
- Fresh live checks passed native and legacy on **LegumeMine and MaizeMine**:
  three sorted rows, two workers, two outstanding pages. Serial/parallel,
  Parquet/DuckDB/Arrow and default dataframe outputs agree. All discovered
  templates parsed: **14 LegumeMine / 73 MaizeMine** per profile; none were executed
  by this smoke. No server mutation or public throughput measurement occurred.
- GitHits checked pinned original source and the Python/Arrow/pandas contracts.
  Dated registry checks reported no active advisories for the nine pinned runtime
  and optional dependencies in `dependencies.parquet`. This is not a lockfile or
  vulnerability reachability certification.
- Historical deck reproduction extracts original Parquet at release commit
  `6c5077ec0e61127b94fec277e25e1330e5441e87`; all 15 pages match published text and
  layout. The published PDF and its measurements are unchanged. Broken benchmark
  commands and historical links are repaired; Sphinx builds with warnings as errors.

## Measured tradeoffs

Five fresh-process repeats per variant, Python 3.14.5, Intel(R) Core(TM) Ultra 7 265K.
WSL2 exposed 18 logical CPUs; these runs do not certify bare-metal 20-core
throughput. These are warm two-column local fixtures; HTTP timings, cold starts and native
allocation peaks are excluded. Raw observations are in `performance-trials.parquet`;
DuckDB-derived medians are in `performance-summary.parquet`.

| Fixture | Original InterMine | Legacy profile | Native profile | Native vs original |
|---|---:|---:|---:|---:|
| 200 XML serializations | 4.95 ms | 2.11 ms | 2.20 ms | 2.25× |
| 10,000 streamed dictionaries | 17.03 ms | 13.58 ms | 7.55 ms | 2.26× |
| 10,000 dictionaries in a list | 33.18 ms | 27.40 ms | 7.85 ms | 4.23× |
| 10,000 retained row objects | 13.40 ms | 12.60 ms | 5.41 ms | 2.48× |

Legacy list construction made two fixture requests, native one. Streamed
comprehensions made one in all profiles. Retained-row Python peaks were about
2.36 MB original, 2.21 MB legacy and 2.21 MB native; there is no meaningful
native-versus-legacy retained-allocation saving in this narrow fixture. XML
Python peaks were 349,754 bytes original versus 10,134 bytes in either candidate
profile. These tracemalloc figures exclude native buffers and process RSS.

The complete custom-batch Parquet/DuckDB pipeline took 31.9–33.4 ms for 10,000
rows with either pandas or Polars output. The difference is within observed
variation, so it does not establish a meaningful adapter penalty or prove
zero-copy for every type. No new storage-savings claim is made in this change;
existing Parquet storage evidence retains its original baseline and scope.

## Boundaries retained

No `intermine` namespace shim, old-intermine314 configuration translator or
original distribution dependency is installed. Imports/dependency declarations
must change explicitly. Pandas Arrow dtypes differ from old NumPy/object inference;
whole-frame materialization still needs memory. Legacy sizing adds work by design.
The departure guide covers intentional repairs, temporary-list cleanup ownership,
retry semantics and native-only optimizations. Windows/macOS, free-threaded Python,
subinterpreters, private templates, authenticated writes, exhaustive argument
combinations and full public-mine datasets are not certified by these checks.
CPU budgets 16/8/6/4/2 and bounded reservations remain; external preinitialized
native pools and process RSS are not hard-limited by these settings.

## Reproduction

Use separate fresh environments for original and candidate packages. The runner
rejects original InterMine in the candidate environment and verifies installed
package origins. Replace the interpreter paths below with those environments:

```bash
python -m benchmarks.runners.compatibility_acceptance \
  --fixture tests/fixtures/compatibility/model.xml \
  --reference-python /tmp/original/bin/python \
  --candidate-python /tmp/candidate/bin/python \
  --output /tmp/probes.parquet
# Add --measure --repeats 5 and a separate output for warm measurements.
python -m benchmarks.runners.compatibility_acceptance --live \
  --output /tmp/live.parquet
python -m benchmarks.evidence docs/analysis/v021-readiness \
  'SELECT * FROM performance_summary ORDER BY "case", variant'
```

`validation.parquet` records prior exact-commit installation/CI receipts;
`original-parity.parquet`, `candidate-live.parquet`, `environment.parquet` and
`completed-obligations.parquet` record scoped outcomes and provenance. Current
source assertions are bound to controlled input hashes in the regenerated API
audit. No existing historical measurement was relabeled as current evidence.
