# Compatibility-preserving pipeline uplift

Baseline: `8e584c56a34b7a33fcde13d5185d451f6aab1ae9`, Python 3.14,
package **0.1.8**. This work does not authorize a release or dependency upgrade.
The immutable prototype evidence remains in `../optimization-2026-10-14`.

Each numbered task is committed to master. Each phase ends with full tests,
Ruff, refreshed API evidence, benchmark review, a handoff commit and a push.
The next phase requires green CI for that exact pushed SHA and a clean tree.
Handoffs record task commits, pinned GitHits sources, commands, outcomes,
retired code, fallback reasons and limitations. Failed gates are fixed and
repushed before proceeding. Rejected optional optimizations need measurements
and an explicit reason; they are not silently deferred.

| Phase | Tasks | Opportunities |
|---|---|---|
| 1 | 1.1 evidence/harness; 1.2 dead helpers; 1.3 unsized streams; 1.4 migration/audit | execution correctness, O18 |
| 2 | 2.1 shared row layouts; 2.2 object projections/binding; 2.3 direct lists; 2.4 bytes JSON | O01–04 |
| 3 | 3.1 one execution snapshot; 3.2 shared wire preparation; 3.3 operation CPU plan; 3.4 batch conversion plan | O05, O06, O09, O10 |
| 4 | 4.1 flat columns; 4.2 writer integration; 4.3 eligible part promotion; 4.4 borrowed SQL connection | O07–09, O12 |
| 5 | 5.1 bounded identifier writes; 5.2 prefetch examples; 5.3 native reader evaluation; 5.4 worker setup | O11, O13, O16, O17 |
| 6 | 6.1 instrumentation; 6.2 tests; 6.3 distribution matrix; 6.4 final report | O14, O15, O18 |

Acceptance: 21 randomized paired warm repetitions, separate cold-start
measurements, same interpreter/dependencies. Optional refactors need a 5%
targeted median improvement or 10% allocation/RSS reduction. Deterministic
request/clone reductions and dead code qualify separately. Investigate any
reproducible representative end-to-end median regression over 3%; repeat noisy
comparisons once, then retain the prior path if inconclusive. Microbenchmarks
do not establish public-mine throughput. Timing and evidence tables use
Parquet; `catalog.py` exposes them through DuckDB views.

The planned departures are unsized ResultIterator, immutable streamed row/object
metadata (direct constructors remain mutable), and parser binding at stream
opening. Attribute rebinding and customization before opening remain supported.
Keep legacy/native profiles, exact numeric values, terminal status validation,
bounded page accounting, cleanup, cancellation and atomic publication.

Native CPU budget remains 16, with 8/6/4/2 alternatives and existing stage
allocations. Parquet stays canonical; DuckDB remains the SQL engine. No new
language runtime, JIT requirement, subinterpreter certification or automatic
network prefetch is introduced. Replace superseded code within its task.
