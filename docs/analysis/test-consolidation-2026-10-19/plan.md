# Test consolidation implementation

Runtime and version remain unchanged. One commit per task; push and verify full CI per phase.

| Phase | Tasks | Status |
|---|---|---|
| 1 | 1.1 baseline; 1.2 query support; 1.3 list/job/artifact support; 1.4 lifecycle helpers | accepted |
| 2 | 2.1 contract schema; 2.2 migrate assertions; 2.3 remove redundant surface | accepted |
| 3 | 3.1 page cases; 3.2 summary cases; 3.3 resource cases; 3.4 scheduler retirement | accepted |
| 4 | 4.1 cold runner; 4.2 scenario consolidation; 4.3 installed coverage | accepted |
| 5 | 5.1 CI selection; 5.2 event routing; 5.3 baseline deduplication | accepted |
| 6 | 6.1 acceptance; 6.2 final report | implemented |

Acceptance: no lost runtime lines/branches or assertion obligations; 460 historical APIs, 331 passing scopes, 129 departures / 43 groups, 40 bindings. All cold scenarios stay isolated. Full CI on master/releases; focused optional modes on PRs.

Each handshake records task commits, tested input hashes, full CI URL/SHA, source/installed results, coverage differences and the next phase opening receipt. A failed gate is fixed in the same phase. No skip/xfail weakening.

Phase 1 receipt: `5fa9948f1360b5e1f987aef7a95c1a637480946a`; [CI 39245019860](https://github.com/karikris/intermine314/actions/runs/39245019860), all seven jobs passed. Local: 2813 cases, 8176 runtime lines / 2563 branches retained; 331/129 API scopes unchanged.

GitHits verified pytest 9.1.1 parametrization semantics at `cf470ec0bf7eb89cd97dd56df4859eae5db46447`, `doc/en/how-to/parametrize.rst:32-88`: explicit scenario tuples; parameters are not copied. Shared builders therefore return fresh mutable objects.

Phase 2 receipt: `a85a87e7140bb03dea239f72747ceb8d712dc79f`; [CI 39245674410](https://github.com/karikris/intermine314/actions/runs/39245674410), all seven jobs passed. Local: 2798 cases, no runtime coverage losses, 331/129 scopes unchanged; four structural fault probes detected.

Phase 3 receipt: `0e7560a2be240edd607c41a00db4e4204d3dea23`; [CI 39246244464](https://github.com/karikris/intermine314/actions/runs/39246244464), all seven jobs passed. Local: 2729 cases, no coverage loss, 331/129 scopes unchanged; footer, reservation cleanup, COPY order and final summary wire-format faults detected.

GitHits verified CPython 3.14.5 `5607950e`, `Doc/library/subprocess.rst:33-65`: subprocess.run captures streams and kills/waits on timeout. Cold runner uses this API and Python isolated mode, with explicit package-origin checks.

Phase 4 receipt: `8905fda4e9e544030bd5eb90d3940c8f59faf341`; [CI 39246847252](https://github.com/karikris/intermine314/actions/runs/39246847252), all seven jobs passed including installed cold-origin checks. Local: 2731 cases, no coverage loss and 331/129 scopes unchanged. All 17 cold scenarios retained (15 old functions, three wildcard orders, optional-plot error case).

GitHits verified GitHub docs `dae83f7d28f3b8e10015bd6b6d8b72267cedceca`: reusable workflows inherit the caller event (`events-that-trigger-workflows.md:1060-1064`); branches-only push filters exclude tag events (`run-on-specific-branches-or-tags1.md:6-10`). Master, manual and release caller events therefore keep full suites; tag release validation is performed through the reusable workflow once.

Phase 5 receipt: `886d5e905c076697adbc38c60154c14858781d52`; [master CI 39248044748](https://github.com/karikris/intermine314/actions/runs/39248044748), all seven jobs passed; [manual release validation 39248054399](https://github.com/karikris/intermine314/actions/runs/39248054399), validation/artifact checks passed and publication skipped. Local 2734 cases, 436 speed and 65 plot cases; no coverage loss; 331/129 scopes unchanged. All four CI inventories now parse 2734 collection totals without warnings.

Phase 6 acceptance: 2734 source cases, 877 installed cases, all six distribution modes and minimum-dependency gate passed; 861 original function obligations reconciled; no runtime or controlled execution-input drift after Phase 5. Versioned coverage points replace reliance on the original temporary baseline. Reproducible probes detected eight faults. Final full master CI is linked from the implementation completion response.
