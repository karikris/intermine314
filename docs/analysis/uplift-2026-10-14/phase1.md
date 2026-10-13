# Phase 1 handoff

Tasks: 1.1 `8f2c711`, 1.2 `373fba9`, 1.3 `2e49710`; 1.4 is the
immediately preceding migration/audit commit. Opening baseline `8e584c5`, CI
[38419571046](https://github.com/karikris/intermine314/actions/runs/38419571046).

Checks: repository Ruff clean; full API trace **2,771 passed in 35.55s**;
rendered original inventory retains all 460 symbols. ResultIterator length is
an intentional, negatively tested absence. Query count remains unchanged.
Verification scripts explicitly count yielded template rows and materialize
installed query results once.

21 randomized process-isolated pairs (`phase1_trials.parquet`): materializing
10,000 local rows **16.229 → 8.232 ms**, requests **2 → 1**. Keyed iteration
24.670 → 24.956 ms; direct list iteration 8.113 → 8.078 ms. These are fixture
timings, not public-mine speedups. Process duration includes interpreter,
fixture setup and warmup; it is not a pure import timing.

Removed two unused private helpers and their unused list cap; no replacement
framework or redundant compatibility implementation added. Pinned CPython
3.14.8 length/copy semantics support the design; raw GitHits evidence is in
`sources.parquet`. No dependency or version changes. The next phase must
record this handoff's exact pushed SHA and successful seven-job CI result.
