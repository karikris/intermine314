# Phase 6 handoff

Opening receipt: clean master/origin at
`55b00dfaef872285e31dfe9e02b7eff51661b1c3`; all seven jobs passed in
[CI 38473108447](https://github.com/karikris/intermine314/actions/runs/38473108447).

Task commits:

- 6.1 `c5ac658`: shared operation worker plans at remaining callers, lazy logging,
  redundant sorting/key-list removal; measured rejection of instrumentation fusion.
- 6.2 `8fe1c62`: shared identical fixture assertion; redundant surface/config tests
  removed while semantic scopes and license contracts remain.
- 6.3 `ae90e97`: expanded installed-wheel regressions and CI, distribution evidence,
  refreshed compatibility audit/migration guide.
- 6.4 `d2cc633`: final measured report, inventory, repeatable cold/warm comparisons,
  canonical Parquet evidence and DuckDB views.

Local gates: Ruff/whitespace clean; full source audit **2,813 passed in 35.41s**;
all 460 original inventory entries accounted for (331 passing scopes / 129 tested
departures, 43 groups, 795 curated semantic nodes). Fresh six-mode current
wheel/sdist installs, Twine, pip check and strict installed Sphinx pass. Minimum
Python/dependency base/speed/combined installs pass. Expanded installed-wheel
regressions: **904 passed** each for current and minimum environments, outside
the checkout with every loaded package module verified as installed.

Twenty-one randomized paired repetitions per workload: 11 final original-baseline
workloads and three previous-phase export workloads. No previous-phase export
median regresses by 3%; cold Service import 26.138 → 25.976ms. Final summary:
keyed rows 3.19×, eligible numeric file 3.36×, borrowed repeated SQL 9.11×;
String/Int64 file 1.01×. These are local fixtures against pre-uplift intermine314,
not public-mine throughput or a comparison with legacy intermine. Source grew
1.8% while repeated work and Python allocations decreased.

O14 instrumentation fusion is rejected (1.15%, then 0.44% upper-bound gain).
O16 native read_all remains rejected for cancellation; O17 worker setup state
remains rejected below the 5% gate. No dormant alternatives remain in runtime.
Snapshots, precise fallbacks, custom dispatch, footer validation, byte accounting,
cancellation, ownership and atomic publication remain required behavior.

GitHits source decisions and responses, raw trials, rejection evidence, task
receipts and install reports are in Parquet. `analysis.duckdb` exposes views;
[report.md](report.md) explains scope, migration, worker profiles, dependencies,
retired code and limitations. Version **0.1.8** and dependency floors unchanged;
no release/tag/publish operation is included.

Final acceptance requires pushing this handoff to master and checking all seven
CI jobs for its exact SHA, including the new installed-wheel regression step.
The completion response must link that run. No further implementation phase is
pending; a failed CI gate must be corrected and rerun before completion.
