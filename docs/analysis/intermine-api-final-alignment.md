# Current API alignment audit — 14 October 2026

The refreshed audit resolves all **460** inventoried public symbols: 453 static
module/class bindings and seven Service methods delegated through ListManager.
There are **333 passing scoped entries** and **127 entries with tested
departures**, organized into 39 departure groups. These are executed assertion
scopes, not exhaustive equivalence or a claim of 100% replacement compatibility.
The supplementary 40-binding re-export contract is verified independently in
each clean wheel installation.

The instrumented suite passed **2,769 tests**, with 788 semantic test nodes linked
to the historical inventory. Its 275-file execution-input manifest was captured
before initialization and pytest, checked after execution, then verified again
before rendering. Source revision, exact hashes, assertion excerpts and
parameter cases are in the artifacts:

- [Final JSON](intermine-api-final-coverage.json)
- [Final Parquet](intermine-api-final-coverage.parquet)
- [Curated scopes and departures](api-audit-scopes.json)
- [Current release review](release-2026-10-14/report.md)
- [Build, installed-regression and CI evidence](release-2026-10-14/verification.parquet)
- [Current dependency review](release-2026-10-14/dependency-review.md)

The comparison is against Python client 1.13.0 at
[`d888b779c8050bad789e26b312f40d220bc85d0d`](https://github.com/intermine/intermine-ws-python/tree/d888b779c8050bad789e26b312f40d220bc85d0d),
verified through GitHits. Adapted source is credited by the shipped BSD-2-Clause
license and NOTICE. The [original 11-column inventory](intermine-api-inventory.parquet)
and [baseline review](intermine-api-compatibility.md) still describe `41c363e`;
their missing-name fields are historical, not findings against current master.

Each passing scope has curated observable assertions. Tracing alone cannot
promote a symbol to passing; shared code cannot distinguish alias spellings,
so explicit alias tests supply that evidence. Actual signatures and supported
calls are retained per symbol. The 312 identical signature strings and 148
textual differences are not a count of compatible/incompatible calls.

The larger caller-visible differences remain:

| Area | Current contract / required migration |
| --- | --- |
| Imports | Replace `intermine` with `intermine314`; there is no namespace shim or satisfaction of the original distribution dependency |
| Service defaults | `webservice.Service` selects legacy object/ResultRow defaults; native `service.Service` selects dictionaries |
| Dataframe | Polars under both profiles; migrate pandas-specific operations |
| List lifetime | Closing/destructing Service closes owned transport without deleting server lists; use explicit ListManager cleanup or `Service.flush()` when needed |
| Validation and historical bugs | Nested logic, XML, paths, ancestry, helper encoding and other repairs have scoped documented departures |
| Streams and concurrency | Close early-stopped iterators/readers; do not mutate shared Query/Session configuration during work |

All 39 groups retain caller effect, reason, recommendation, upstream source pins
and executed test selectors in the JSON/Parquet evidence. New resource profiles,
Arrow readers, manifests, metadata concurrency, object prefetch and spawned
transforms have separate regression coverage; they do not expand the upstream
460-symbol denominator.

Historical [mutation evidence](api-audit-mutation-checks.json) records twelve
intentionally wrong implementations detected by selected tests in earlier
reviews. It was not repeated for every symbol in this refresh. Historical
installation/remediation files retain their original test counts, revisions and
dependency versions; the current release folder supersedes them for this audit.

Reproduce after completing all controlled source/test/documentation edits:

```bash
.venv/bin/python -m scripts.audit_api_coverage --trace /tmp/current-api-trace.json
.venv/bin/python -m scripts.audit_api_coverage --render /tmp/current-api-trace.json
```

Rendering rejects failed suites, changed controlled files or membership, missing
curated test cases and symbol-count mismatches. Generated outputs are excluded
from the execution snapshot so rendering cannot invalidate itself. Environment,
dependency and live-service evidence is recorded separately. Unexecuted inputs,
all-branch equivalence, private APIs and every live server/authenticated variant
remain unverified. See the release review for the precise supported scope.
