# Phase 2 handoff

Opening receipt: `83c3bed2d8b94eb2296bc0ad9fc78685cdcac6ba`, all seven jobs
passed in [CI 38466528921](https://github.com/karikris/intermine314/actions/runs/38466528921).
Task commits: 2.1 `a73a37e`, 2.2 `2e2f6e6`, 2.3 `dcbd268`, 2.4 immediately
preceding bytes-decoding commit. Ruff clean; **2,779 tests pass in 34.31s** under
the API trace. All 460 original symbols remain accounted for. Migration docs
and audit departures cover streamed metadata and opening-time binding.

21 randomized warm pairs, independent processes, equal results:

| Local operation | Phase 1 ms | Phase 2 ms | Ratio |
|---|---:|---:|---:|
| Materialize 10k lists | 8.319 | 6.615 | 1.26× |
| Keyed iteration, 10k × 20 fields | 23.353 | 7.242 | 3.22× |
| List iteration | 7.768 | 6.233 | 1.25× |
| Object decoding | 18.763 | 10.686 | 1.76× |
| Stdlib list decoding | 11.730 | 10.356 | 1.13× |
| Retain and access 10k keyed rows | 30.027 | 7.948 | 3.78× |

Traced peak allocations for retained keyed rows: **21,029,031 → 3,216,089
bytes (84.7% reduction)**. Fixture input is outside tracing. This is Python
allocation evidence, not process RSS. Original-baseline comparisons are in
`phase2_trials.parquet`; previous-phase, memory and stdlib results have separate
tables. Early concurrent exploratory runs were overwritten by sequential runs.

Removed per-row index reconstruction, built-in projection splitting, repeated
facade lookup, temporary list wrappers and ordinary bytes→text→bytes conversion.
Kept direct constructors, custom constructors/parsers, text loaders, exact-number
fallback, Unicode error behavior and per-object identity maps. These fallbacks
serve distinct semantics. Pinned GitHits CPython copy and orjson bytes guidance
are recorded in `sources.parquet`. No dependency/version changes.
