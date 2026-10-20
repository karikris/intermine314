# 0.2.1 readiness and remaining work

Implementation update: the user has selected **fresh installations only**,
broader behavior through the historical service facade, and optional pandas
output. The configuration-upgrade decision and original effort estimate below
are historical review findings, superseded by [implementation.md](implementation.md).
No old-intermine314 configuration translator will be added.

Reviewed 20 October 2026. Candidate source:
`dd745639efc102dfe8df43827b0c5ab62e5c333c`. Published 0.2.0:
`6c5077ec0e61127b94fec277e25e1330e5441e87`.

**Recommendation:** prepare a focused maintenance release. Most implementation
is already complete. Allow **19–32 engineering hours**, or **23–38 hours with
20% contingency**, approximately **3–5 working days** for one maintainer.
These are planning estimates, not measured execution times. The principal
decision is whether 0.2.1 may require configuration migration. Existing users
who copied the 0.2.0 runtime TOML will receive an error on current master.

This review changes no runtime code, dependencies, version, tag or publication.
It adds this report and Parquet evidence. It does not certify exhaustive
upstream equivalence, new operating systems or public-server throughput.

## Baseline and validation

The [published 0.2.0 wheel](https://pypi.org/project/intermine314/0.2.0/) was
downloaded and checked against its registry SHA-256,
`408c179eb70a3ae79f2732eab98335cba12a634ee112fce5210ebe066acad38a`.
All 79 runtime Python files match the release tag. Master is 50 commits ahead;
10 Python runtime files and the packaged runtime TOML differ. Test and evidence
consolidation account for most repository changes.

| Check | Result and boundary |
|---|---|
| Current source suite | 2,760 passed, no skips or failures, 19.93 seconds on Python 3.14.5 |
| Ruff | Passed with installed 0.17.0 |
| Exact source CI | All seven jobs passed at the reviewed SHA; includes installation modes, minimum dependencies and installed-wheel regressions |
| Focused checks against published wheel | 301 passed, two failed out of 303; both failures are custom Query/Template endpoint overrides fixed on master |
| Copied 0.2.0 runtime TOML | Published wheel accepts it; master raises a migration ValueError listing 11 removed keys |
| Live LegumeMine and MaizeMine | Native and legacy profiles pass; three rows per query; serial/ordered parallel/Parquet/Arrow/DuckDB/Polars agree |
| Live template parsing | 14 LegumeMine and 73 MaizeMine templates parse under each profile; this check did not execute every template |
| Historical API inventory | Existing assertion audit retains 331 passing-scoped and 129 tested-departure symbols out of 460, plus 40 supplemental bindings |

The [exact-head CI run](https://github.com/karikris/intermine314/actions/runs/39416050175)
is the existing clean artifact gate. The current installed-wheel regression
selection contains 969 cases. The focused wheel comparison used the same modern
dependencies as master and excluded source-package imports from the tested
package. It establishes specific regression fixes, not complete wheel-to-master
equivalence.

Fresh live checks used two HTTP workers, two outstanding pages and three rows,
with explicit Service/runtime lifetimes. They are correctness smoke checks.
Arrow was imported before the runtime and retained its existing 18-thread pool;
the package reported that mismatch. This is the documented externally initialized
pool behavior, not evidence that cpu2 hard-limits all native threads.

## Work before release

| Priority | Work | Concrete completion condition | Estimated hours |
|---|---|---|---:|
| Required decision | Set the configuration upgrade contract | State whether 0.2.1 requires migration; list all 11 removed keys and their replacements; test copied 0.2.0 configurations against that stated policy | 4–8 |
| Required repository repair | Restore evidence reproduction paths | Deck builder resolves its historical inputs; benchmark instructions use surviving tools; documentation identifies immutable historical sources where data was removed | 4–6 |
| Required release preparation | Separate unreleased changes and current evidence | Add 0.2.1 changelog, update current install/migration guidance, bump both version declarations when releasing, distinguish current receipts from historical observations | 3–5 |
| Required final acceptance | Validate the final candidate and distributions | Full source matrix, six install modes, minimum-dependency checks, 969 installed regressions, API audit, Sphinx, targeted upgrade checks and bounded live smoke pass at the intended commit | 6–10 |
| Release operation | Publish and verify artifacts | Matching master tag, successful exact-commit workflow, registry hashes equal validated artifacts, fresh PyPI install and version/API smoke | 2–3 |
| **Total** | | **Before contingency** | **19–32** |

The estimate assumes the existing Linux/Python 3.14 support scope, working
trusted publishing and no newly discovered data-correctness defect. Fresh live
checks in this review reduce uncertainty; repeat them after release-relevant
runtime changes. CI itself took approximately three minutes on the reviewed
head; engineering estimates include changes, review and failure diagnosis.

### 1. Configuration compatibility needs an explicit release decision

[`runtime_defaults.py`](../../../src/intermine314/config/runtime_defaults.py)
removes seven query scheduling fields and four registry scheduling fields.
`get_runtime_defaults()` rejects their presence in an override before schema
fallback. This is intentional cleanup, already tested, but it changes behavior
for existing 0.2.0 installations. Some settings were obsolete for runtime
scheduling; their presence still used to be accepted.

Keep the approved immediate removal if a documented migration is acceptable.
Add a version-specific mapping and an example using `ResourceRuntime`,
`ResourceProfile` and `ParallelOptions`; make clear that a copied old TOML can
prevent normal initialization. Include removed dataclass constructor/attribute
usage in the migration scope. Historical InterMine itself did not expose these
package-specific configuration keys.

If the release must preserve configuration-level drop-in compatibility, a small
boundary translator into the canonical policy is an alternative. It must
preserve actual 0.2.0 precedence and behavior, rather than activate formerly
ignored settings. Budget an **additional 4–8 hours** to define and test it. It
does not require restoring a duplicate scheduler. Another choice is to assign
the migration-requiring release a different version under the project's policy;
the numerical label alone does not settle compatibility for a pre-1.0 package.

### 2. Cleanup left broken reproduction instructions

[`build_deck.py`](https://github.com/karikris/intermine314/blob/dd745639efc102dfe8df43827b0c5ab62e5c333c/docs/releases/v0.2.0/build_deck.py) still reads three deleted
paths: the former release comparison trials, uplift summary and phase-2 memory
trials. Direct DuckDB reads of all three fail with “No files found”. The first
dataset survives at `docs/analysis/evidence/v020-vs017/trials.parquet`; the other
inputs are recoverable from Git. The existing PDF is unaffected, and its
version-pinned GitHub source links remain valid.

Make the historical builder resolve maintained Parquet or extract required
inputs from an immutable historical revision. Preserve original measurements;
do not relabel them as 0.2.1 observations. Close its temporary DuckDB connection
and verify rebuilding to a temporary destination.

[`benchmarks/README.md`](../../../benchmarks/README.md) still tells users to run
deleted `phase_ledger.py`, `build_report.py`, `validate_evidence.py` and
`catalog.py`. [`resources.rst`](../../source/resources.rst) points readers to
removed phase-8 evidence without a revision. One historical plan also contains
a broken relative link to the deleted compatibility report. Replace active
instructions with maintained runners and `benchmarks.evidence`; point historical
material to exact Git revisions. Successful Sphinx builds do not detect all
plain-text command/path references.

### 3. Release documentation and receipts need a clean boundary

`CHANGELOG.md` has no unreleased/0.2.1 entry. The migration guide mixes published
0.2.0 behavior with the later cleanup. Record the custom endpoint fix, faster XML
loading, shared ownership/execution and test consolidation as post-0.2.0 work.
Keep versioned 0.2.0 slides and historical test fixtures versioned as they are.

The API audit's `installation_evidence` and `remediation_validation` fields
currently reference archived verification, including 0.1.8 artifacts. Current
cleanup acceptance and CI receipts exist elsewhere. Make those evidence scopes
explicit and link the new release's own exact-commit artifacts.

The local editable environment reports distribution metadata 0.1.8 while source
reports 0.2.0. This is local environment drift, not a published-wheel defect:
use fresh environments for candidate and post-publication checks.

## What is already implemented

| Area | Current result | Remaining 0.2.1 scope |
|---|---|---|
| Custom execution | Query/Template parameter and endpoint overrides survive serial, count, parallel and Parquet calls | Preserve the two demonstrated regression tests |
| XML loading | Query/Template share bounded Expat/ElementTree parsing; public DOM return boundary remains | Final candidate parity and live checks |
| Runtime size | 14,385 → 14,173 Python lines; 553,431 → 545,951 bytes, identical 79-file count | No broad rewrite needed |
| Ownership | Shared response lifetime and helper account replacement logic | Preserve rollback, borrowed-resource and interruption contracts |
| Parallelism | Shared executors, host admission, page/byte reservations, CPU profiles 16/8/6/4/2 remain | Verify invariants; no new worker-count uplift is needed |
| Storage | Parquet, DuckDB, Arrow, Polars, atomic publication and manifests remain | No new compression/storage-saving claim versus 0.2.0 |
| Tests | Published release recorded 2,813 source/904 installed cases; current counts are 2,760/969 | Preserve meaningful coverage rather than reduce counts again |

The [cleanup measurements](../cleanup/report.md) compare a pre-cleanup runtime
whose Python files match published 0.2.0 with the final runtime on matched
dependencies: a 2,000-value query XML fixture takes 3.501 → 1.622 ms, with
33% lower peak Python allocation and 21% lower process peak RSS. Result streaming
is effectively unchanged. These are existing five-pair offline measurements,
not new end-to-end benchmarks. The candidate wheel was approximately 0.5% smaller;
no new data-storage savings versus 0.2.0 were established.

## Dependencies and compatibility scope

GitHits checked package metadata and the resolved runtime graph on 20 October.
Requests 2.34.2, urllib3 2.8.0, DuckDB 1.5.6, Polars 2.0.0 and PyArrow 26.0.0
remain the latest reported core versions. The nine resolved dependency packages
had no reported active advisory occurrences or graph conflicts. This registry
check is not a complete environment or vulnerability guarantee. The current
dependency floors therefore need no uplift for this release.
Sources: [Requests](https://pypi.org/project/requests/),
[urllib3](https://pypi.org/project/urllib3/),
[DuckDB](https://pypi.org/project/duckdb/),
[Polars](https://pypi.org/project/polars/),
[PyArrow](https://pypi.org/project/pyarrow/).

Ruff 0.17.1 is available and already permitted by `ruff>=0.17.0`. GitHits found
rule changes and no dependency changes. Updating the development environment
and re-running lint is optional; raising the package's minimum is unnecessary.
See the [Ruff release notes](https://github.com/astral-sh/ruff/releases/tag/0.17.1).

Legacy replacement remains scoped: use
`from intermine314.webservice import Service` for historical defaults. There is
no original `intermine` namespace/distribution alias; pandas workflows need a
Polars migration; result streams are unsized; documented departure groups remain.
The upstream XML loader was checked through GitHits against the
[pinned InterMine source](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/query.py#L425).

Keep Windows/macOS certification, Python 3.15, free-threading, subinterpreter
interchange, new Rust extensions and a wider benchmark campaign outside the
small patch scope. Python 3.14's bounded `Executor.map` is useful for simple
independent reads, but it does not replace the package scheduler's byte
accounting, footer validation and ordered/unordered lifecycle obligations.
See the [CPython 3.14.5 executor source](https://github.com/python/cpython/blob/5607950ef232dad16d75c0cf53101d9649d89115/Lib/concurrent/futures/_base.py#L577).

## Suggested completion phases

1. **Upgrade contract and repository repair:** settle the configuration policy;
   test old configurations; repair the deck and benchmark reproduction paths.
   Handshake: no unexplained upgrade differences and all documented commands
   resolve to maintained tooling or a named historical revision.
2. **Release evidence:** add release notes, update evidence links and version
   declarations, run the candidate source/install/API/docs/live checks.
   Handshake: exact pushed commit passes full CI; recorded artifacts, versions,
   hashes and supported-platform claims agree.
3. **Publication:** use the existing guarded tag workflow; compare PyPI artifacts
   to the validated manifest; smoke-test a clean registry installation.
   Handshake: publication receipt records registry hashes, installed version,
   import origins and basic legacy/Parquet behavior.

For implementation, retain the established one-commit-per-task and
one-push-per-phase approach. Publication belongs to a separately authorized
release task. The present request is analysis only.

`checks.parquet`, `work-items.parquet` and `live.parquet` record this review's
observations and estimates. Query them with:

```sh
.venv/bin/python -m benchmarks.evidence docs/analysis/v021-readiness 'SELECT * FROM work_items'
```
