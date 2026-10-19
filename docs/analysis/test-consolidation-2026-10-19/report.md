# Test consolidation completion report

Implemented the six-phase plan on master with focused task commits and a push at each phase boundary. Runtime files and version 0.2.0 are unchanged. Baseline: `b6b6147a0b1f060618fc1af9e5f79412d95489ad`.

The suite is smaller while preserving its central behavior checks. The main execution saving comes from CI routing. The source suite removes 79 collected cases (2.81%) and 25 test functions. The selected Cartesian matrices shrink from 141 to 73 cases. All 17 cold-process scenarios remain independent.

| Measure | Before | After | Change |
|---|---:|---:|---:|
| Collected combined cases | 2,813 | 2,734 | -79 (-2.81%) |
| Test functions | 861 | 836 | -25 (-2.90%) |
| Python test and support lines | 18,367 | 18,169 | -198 (-1.08%) |
| Python test and support files | 91 | 106 | +15 (+16.48%) |
| Selected matrix cases | 141 | 73 | -68 (-48.23%) |
| Cold process scenarios | 17 | 17 | +0 (+0.00%) |
| Minimum signature contracts | 137 | 137 | +0 (+0.00%) |
| Historical API symbols | 460 | 460 | +0 (+0.00%) |
| Passing API scopes | 331 | 331 | +0 (+0.00%) |
| Tested API departures | 129 | 129 | +0 (+0.00%) |
| Tested departure groups | 43 | 43 | +0 (+0.00%) |
| Owned historical reexports | 40 | 40 | +0 (+0.00%) |
| Previously covered runtime lines | 8,176 | 8,176 | +0 (+0.00%) |
| Previously covered runtime branches | 2,563 | 2,563 | +0 (+0.00%) |
| PR source case executions | 11,252 | 5,969 | -5,283 (-46.95%) |
| Master source case executions | 11,252 | 10,936 | -316 (-2.81%) |
| External phase-zero baseline jobs | 4 | 1 | -3 (-75.00%) |
| Full test jobs per tag release | 8 | 4 | -4 (-50.00%) |
| Installed regression cases | 904 | 877 | -27 (-2.99%) |

Counts include parametrized cases, including expected skips in installations without optional extras. The PR schedule uses two full suites plus 436 speed and 65 plotting cases: 5,969 case executions, 46.95% below the original four full suites. Relative to four full *consolidated* suites, the reduction is 45.42%. These counts describe a single PR workflow; removing duplicate feature-branch push workflows is an additional saving. These are execution-count reductions, not measured wall-clock speedups. Setup, installation, wheel builds and host contention still contribute to CI duration.

The Python line count includes every fixture, shared helper and child program under tests; helpers are not excluded to inflate savings. File count increases because shared responsibilities move into focused support modules. The new CI selector adds 59 tooling lines outside tests. Documentation evidence and fault-reproduction tools are separate from test/support LOC. No runtime code is removed or modified.

## What changed and what remains covered

| Area | Implementation | Preserved obligations |
|---|---|---|
| Shared setup | Lazy query/result/Parquet, list, enrichment, job, registry, saved-query, artifact and benchmark builders | Fresh mutable services/models; ownership and exactly-once response cleanup; no cross-test provider imports |
| Public API | Schema 4; 137 minimum signatures, 46 exact signatures, 28 alias identities, four absent modules | Defaults, parameter order/kinds, exports, subclass/property checks, service paths; all 40 historical reexports |
| Page validation | Native full format/footer matrix; separate native/legacy failure and null-row checks | Failed/missing/malformed/truncated footers; empty/short/exact/overproduced pages; response cleanup |
| Summaries | All native row overrides and both legacy version boundaries | Raw JSON summary wire mode, categorical/numeric values, pagination, query immutability, errors and closure |
| COPY/resources | Full order/layout combinations at CPU 2/16; intermediate budgets keep ordered single-file checks | Exact thread settings, insertion order, row groups, spill/memory configuration, round trips and owned resources |
| Scheduler | Immediate-completion fake removed; strong queued/running cleanup and real buffered-memory tests retained | Exactly-once executor ownership, cancellation, reservations, payload release, late-page index/offset/cause |
| Cold processes | One bounded isolated runner, 17 named programs; common wildcard program shared across three orders | Heavy-import blockers, logger state, configuration/actions/errors, all import orders, source/wheel origin checks |
| CI | Explicit full/speed/plots selector; unknown/empty selections fail; parseable collection totals | PR base/combined full; master/manual/release all four full; minimum dependencies, six installation modes, installed checks, JUnit and collection artifacts |

The installed suite now has 877 passing cases (904 before). Its reductions come from the same approved matrix changes and transferred cold checks; consolidated cold-runner and public-contract tests now also run against the wheel. The accounting is 904 - 54 matrix combinations - 5 former cold cases + 31 consolidated cold/harness/contract cases + 1 transferred late-failure case = 877. The six distribution modes remain base, plots, analytics alias, sdist, speed and combined; the minimum Python/direct dependency gate remains separate.

## Acceptance evidence

The original 861-function [assertion ledger](obligations.parquet) has 816 retained owners, 40 migrated owners and five reduced matrices with their assertions preserved. No removed function is unmapped. Original source hashes and assertion expressions remain in the ledger; Git history contains every removed implementation.

All previously executed 8,176 runtime lines and 2,563 runtime branches remain covered. All 79 runtime file hashes match the baseline. The historical API audit remains 331 passing scopes and 129 tested departures across 43 groups, covering 460 symbols. These guards establish parity with the baseline; they are not a proof of complete compatibility or coverage of native engines. Cold children validate their own assertions and package origins separately from parent-process coverage.

Eight injected regressions are detected: missing export, changed default, changed parameter kind, incorrect alias, premature footer acceptance, unreleased cleanup reservations, incorrect COPY insertion order, and incorrect final summary wire format. [Contract](contract-faults.parquet) and [scheduler](scheduler-faults.parquet) receipts contain the results. Reproduce with:

```sh
.venv/bin/python docs/analysis/test-consolidation-2026-10-19/fault_probes.py --output /tmp/consolidation-faults
```

Workflow syntax passes actionlint 1.7.12. GitHits checks and pinned upstream references are recorded in [the phase plan and receipts](plan.md). The manual release workflow completed all validation/artifact checks and skipped publishing. No tag, package version or PyPI release was changed.

| Gate | Validated head | CI evidence | Result |
|---|---|---|---|
| Phase 1 | `5fa9948f1360` | [run 39245019860](https://github.com/karikris/intermine314/actions/runs/39245019860) | passed |
| Phase 2 | `a85a87e7140b` | [run 39245674410](https://github.com/karikris/intermine314/actions/runs/39245674410) | passed |
| Phase 3 | `0e7560a2be24` | [run 39246244464](https://github.com/karikris/intermine314/actions/runs/39246244464) | passed |
| Phase 4 | `8905fda4e9e5` | [run 39246847252](https://github.com/karikris/intermine314/actions/runs/39246847252) | passed |
| Phase 5 | `886d5e905c07` | [run 39248044748](https://github.com/karikris/intermine314/actions/runs/39248044748) | passed |
| Manual release validation | `886d5e905c07` | [run 39248054399](https://github.com/karikris/intermine314/actions/runs/39248054399) | passed; publish skipped |

The manual release-validation gate checks the Phase 5 input set and skips publishing. Phase 6 adds evidence and documentation only, and revalidates that the controlled execution-input hashes are unchanged. Its final push receives the ordinary full master CI gate; the exact closing run is linked in the implementation completion response.

Evidence uses Zstandard Parquet. DuckDB can inspect the metrics and receipts directly:

```sql
SELECT metric, before, after, after-before AS delta
FROM read_parquet('docs/analysis/test-consolidation-2026-10-19/final-metrics.parquet');

SELECT phase, head, job, conclusion
FROM read_parquet('docs/analysis/test-consolidation-2026-10-19/ci-receipts.parquet');
```

Full phase traces and per-case coverage were generated under `/tmp/intermine314-consolidation-phase*`; durable phase summaries, API audit, source hashes, migration ledger, CI receipts and fault evidence are versioned here. The versioned coverage-point table contains the original baseline hit set, so `record_phase.py` can compare a fresh coverage report without the original temporary baseline JSON. The per-phase detailed temporary traces are not required to inspect the durable results.
