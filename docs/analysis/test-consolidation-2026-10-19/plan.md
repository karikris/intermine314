# Test consolidation implementation

Runtime and version remain unchanged. One commit per task; push and verify full CI per phase.

| Phase | Tasks | Status |
|---|---|---|
| 1 | 1.1 baseline; 1.2 query support; 1.3 list/job/artifact support; 1.4 lifecycle helpers | active |
| 2 | 2.1 contract schema; 2.2 migrate assertions; 2.3 remove redundant surface | pending |
| 3 | 3.1 page cases; 3.2 summary cases; 3.3 resource cases; 3.4 scheduler retirement | pending |
| 4 | 4.1 cold runner; 4.2 scenario consolidation; 4.3 installed coverage | pending |
| 5 | 5.1 CI selection; 5.2 event routing; 5.3 baseline deduplication | pending |
| 6 | 6.1 acceptance; 6.2 final report | pending |

Acceptance: no lost runtime lines/branches or assertion obligations; 460 historical APIs, 331 passing scopes, 129 departures / 43 groups, 40 bindings. All cold scenarios stay isolated. Full CI on master/releases; focused optional modes on PRs.

Each handshake records task commits, tested input hashes, full CI URL/SHA, source/installed results, coverage differences and the next phase opening receipt. A failed gate is fixed in the same phase. No skip/xfail weakening.
