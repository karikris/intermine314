# Final API alignment audit

The current 0.1.8 implementation resolves all **460** inventoried public symbols:
453 static module/class bindings and seven Service methods bound dynamically
through the cached ListManager. The final audit records **333 passing scoped
entries** and **127 entries with tested departures**, organized into 39 departure
groups. These counts describe executed offline assertion scopes, not exhaustive
upstream equivalence or live-server certification. The original nine-phase
ledger retains its 37 implementation tasks and historical reviews. This refresh
includes the footer-validation, compressed-read and buffered-page fixes, with
**2,362 passing instrumented tests**. Remediation task and validation evidence is
recorded in [remediation validation](remediation-validation.json); the final
phase's exact-SHA CI result is verified after commit/push and reported in the
completion response.

The original [11-column inventory](intermine-api-inventory.parquet) and
[baseline assessment](intermine-api-compatibility.md) remain byte-for-byte
unchanged. Their missing-name and pending-behavior fields describe commit
`41c363e`, not the restored implementation. Current evidence is separate:

- [Final JSON](intermine-api-final-coverage.json): all 460 rows, owner tasks,
  current name/signature/binding/defining class/source, exact executed pytest
  nodes, assertion excerpts, active fixture definitions, source/test hashes,
  the pre-execution input manifest and captured instrumentation digest,
  and scoped departure references. All 460 entries have curated semantic scopes;
  788 executed test nodes appear in the catalog.
- [Final Parquet](intermine-api-final-coverage.parquet): the same 460 symbol
  records with flattened current fields and lossless JSON evidence/deviation
  columns. It is a new final artifact, not a replacement historical inventory.
- [Curated scope rules](api-audit-scopes.json): explicit public-call and builtin
  exception assertions plus each departure's caller effect, reason,
  recommendation, source pin and executed test selectors.

The source comparison is Python client 1.13.0 at
[`d888b779c8050bad789e26b312f40d220bc85d0d`](https://github.com/intermine/intermine-ws-python/tree/d888b779c8050bad789e26b312f40d220bc85d0d).
GitHits source reads and preserved task evidence support the comparison; the
shipped BSD-2-Clause license and NOTICE remain included in both distributions.
Current evidence is based on `c5cc3f7f7104d7cdff2eda0c91e52cff82ec2fc0` plus the audited
working-tree changes, identified by package source hashes in the final JSON.
Fresh [GitHits research](remediation-research.json) records the protocol and
dependency sources and options for further thread-pool and parallel-processing
work. Those options do not change runtime defaults in this remediation.

Two final source-backed regressions were repaired after establishing failures:
ordinary `SubClassConstraint.to_string()` and `repr()` now include `ISA`, and
legacy `Service.__getattribute__(name='root')` supports the source's explicit
keyword call. Native Service retains its existing lookup implementation. The
320 focused constraint/profile/template/list-operation cases passed after these
repairs; 26 direct contract cases cover previously unlinked
callbacks, query strings, template paths, exception/base protocols and explicit
destructor policy.

Signatures are inspected separately from behavior. 312 current signature
strings match the historical strings; 148 differ in spelling, annotations,
defaults, options or call shape. This is not a count of 148 incompatible calls:
no signature-equivalence claim is inferred from either textual result. Current
unbound/property/delegated signatures and actual supported test calls remain
available per row. Dynamic Service delegates are bound on a real configured
fixture instance, without extra HTTP, and exercised by actual list wire tests.

Each evidence node ran and passed. Every symbol has a curated contract that
names the observable assertion it relies on. Automatic name scoring and
three-node selection were removed after SPEC found eleven incidental-call links.
An executed call or an unrelated assertion cannot promote a row to passing.
Instrumentation only supplements the curated scopes with observed execution
phases and owner bindings; it unwraps decorators and records setup/call/teardown.
Shared code cannot identify alias spelling by itself, so explicit alias tests
are selected. Builtin exception constructors and some inherited Service methods
use semantic assertions on the shared native class where a legacy-instance
Python trace is unavailable. No owner's task-wide test list is copied to its
symbols.

[Mutation evidence](api-audit-mutation-checks.json) records isolated wrong
implementations for the eleven initial SPEC findings: Service delegation, empty-sort
state, subclass maps, three path validators, headers, token URL preparation,
scalar encoding, collection field kinds and ancestry. A follow-up serialization
finding adds `BinaryConstraint.to_dict`: replacing its result with an empty
dictionary fails all four selected native/legacy dictionary assertions. All
twelve wrong implementations were detected during actual test calls, rather
than only collection or fixture setup. This check covers those twelve findings,
not every possible mutation of all 460 symbols. Additional review strengthened query view clearing, template owner
assignment and direct opener text reads. Reproduction:

```bash
.venv/bin/python -m scripts.audit_api_coverage --trace /tmp/intermine314-remediation-api-trace.json
.venv/bin/python -m scripts.audit_api_coverage --render /tmp/intermine314-remediation-api-trace.json
```

The trace captures a controlled **219-file execution-input manifest before API
initialization and pytest**, then verifies it again after pytest. Rendering first
compares both file membership and SHA-256 hashes, before writing either artifact.
The manifest includes all package Python/config files, tests and shared fixtures,
conftest/helpers, scripts, samples, documentation source, benchmark Python helpers,
profiles/contracts, and explicit root pytest/build inputs. Added or removed inputs
invalidate the trace just as changed bytes do. The instrumentation digest comes
from that execution snapshot; JSON and Parquet also retain the same canonical
manifest digest. Package-source and individual test-source hashes remain separate.
Curated assertion rules retain their independent render-time hash.

[Provenance regression evidence](api-audit-input-validation.json) records 15
checks that failed against the old renderer because changed fixture/helper/config/
script bytes or membership still reached publication. All **22** final regression
cases pass, including stale/missing manifests, changes during API initialization
or pytest, unchanged repeated publication, and preservation of existing artifacts
on rejection. Generated audit outputs and Python caches are excluded deliberately;
rendering cannot invalidate its own execution snapshot. The controlled scope does
not snapshot the interpreter, installed dependencies, OS or live services.

Rendering also rejects failed suites, changed package/test source, missing curated
test selectors/parameter cases and row-count mismatches. Rows without a curated
semantic scope remain unverified even if their implementations were traced. Unexercised input combinations, branch
coverage, authenticated variants not named by a test, and live mine/server
versions remain unverified. Passing scope does not promote those limits to
parity. The existing profile, query/model, result, list, template, registry and
helper reports describe the detailed contracts.

The larger caller-visible departures include Polars dataframe returns; native
versus legacy defaults; repaired nested logic, model validation and XML; explicit
list cleanup; validated Template forms; managed stream ownership; and repaired
registry/query/plot helpers. Each is tied to concrete tests and recommendations
in the machine-readable artifact. In particular, Service destruction or a
Service context closes transport **without deleting server lists**. Use a
ListManager context, `delete_temporary_lists()`, or the internal manager's
`Service.flush()` before close. The destructor tests explicitly verify that
server temporary names remain and that borrowed sessions remain open.

Final validation passed: **2,159 full pytest cases**, the independently recorded
2,159-case instrumented run, Ruff with `--no-cache`, phase-0 public/native
contracts, Tor/import guardrails, 170 focused native/pipeline cases, and the
Polars/Parquet/DuckDB analytics smoke. [Validation records](final-validation.json)
retain command-output hashes and results.
Fresh [installation evidence](final-clean-installation.json) records newly built
0.1.8 wheel/sdist hashes, four isolated environments (base, plots, analytics
alias, sdist), Python 3.14.5, pip checks, actual installed import paths, lazy
imports, no original client/Pandas, optional Matplotlib/Agg behavior, and
warnings-as-errors Sphinx built from the installed wheel with external mappings
disabled. A fresh byte recheck confirms all 68 wheel package inputs and 76 retained sdist
inputs still match the final checkout. The provenance repair changes only
unpackaged tooling/tests/analysis, so the r2 archive hashes and installed checks
remain applicable.
The earlier task 9.2 installation record remains historical.

The [dependency review](dependency-review.md) retains the dated 09:45 UTC
GitHits metadata/advisory evidence for six selected releases, including
transitive registry resolutions. It found no affected direct or resolved
advisory occurrences in those graphs; it does not audit every allowed future
version or every optional/local distribution. Python >=3.14.5, lazy core Polars
>=1.44.2/DuckDB >=1.5.6/PyArrow >=25.0.1 and optional Matplotlib >=3.11.2 are
installed and exercised. Owned data processing uses Parquet and the SQL/Arrow
bridge; CSV output requires explicit selection. No new live benchmark result,
release, tag, PyPI publication or version bump is claimed.


The newly documented public-helper departures are:

| API | Current behavior and caller impact | Migration guidance |
| --- | --- | --- |
| `encode_str` / `encode_dict` | Text stays `str`, binary stays bytes, other scalars stringify; list/tuple mapping values normalize element-wise for repeated form values. Upstream encoded text to UTF-8 bytes and left other types untouched. | Use `urlencode(..., doseq=True)`; explicitly encode text when bytes are required and retain typed input data separately. |
| `InterMineURLOpener.headers` | Standard configurable `User-Agent` replaces source `UserAgent`. | Update literal-key access and configure `user_agent` through the opener/service. |
| `Model.to_ancestry` | A deep chain returns `[B,C,D]` rather than accidental `[B,C,D,D]`; genuine diamond branches can still repeat D, and cycles raise `ModelError`. | Do not interpret accidental repetitions as distinct ancestors; explicitly deduplicate if set semantics are needed. |
