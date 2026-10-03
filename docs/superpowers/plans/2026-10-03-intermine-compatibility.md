# InterMine compatibility implementation plan

Baseline: `intermine314` 0.1.8 at `41c363e5aff0d3a72e3c93375bde8d33b6d8fec2`; original Python client 1.13.0 at `d888b779c8050bad789e26b312f40d220bc85d0d` (also the local ancestral tag). Initial baseline: 42 passing tests. The [assessment](../../analysis/intermine-api-compatibility.md), [460-row Parquet inventory](../../analysis/intermine-api-inventory.parquet) and [implementation ledger](../../analysis/implementation-ledger.json) are the evidence and ownership records. Located names do not certify compatible behavior.

## Architecture and contracts

Expose the legacy API through lazy `intermine314.webservice.Service` and `Registry` facades. Preserve native `intermine314.service` defaults. Carry an explicit native/legacy compatibility profile through Service query factories, Query cloning, XML imports and Templates. Direct `Query(Model)` selects legacy behavior. After task 3.3, legacy queries preserve the upstream model Class as `.root` (including `.root.name`) and expose a `rootClass` alias. The model/constraint and result phases also restore model.Column from `column`, object defaults from `results`, ResultRow defaults from `rows`, and jsonobjects iteration. Native queries retain string roots/columns and dictionary result defaults. Both profiles share QuerySpec, Executor, the current managed opener and parallel execution; do not fork transport implementations.

`dataframe` returns Polars in both profiles: this is an intentional documented departure from upstream pandas. Require lazy Polars >=1.44.2, DuckDB >=1.5.6 and PyArrow >=25.0.1 in core dependencies, retaining `analytics` as an extra alias. Remove pandas from runtime, extras, owned benchmarks and plotting. Plot helpers use lazy Matplotlib >=3.11.2. The canonical pipeline is InterMine or explicit CSV → Polars → Parquet → DuckDB SQL → Arrow → Polars. Protocol XML/JSON fixtures are permitted.

`Query.export(path, *, format="parquet", ...)` defaults to Parquet. CSV output requires explicit `Query.export(path, format="csv")`; a suffix alone cannot select CSV. Reject conflicting format/path arguments. Preserve existing `to_parquet` partition defaults, single-file mode, compression and resource options. Atomic writes preserve existing output on failure or interruption and schema survives empty results.

`dataframe(start=0, size=None, *, csv_input=None, csv_options=None, parquet_path=None)` uses managed temporary Parquet and DuckDB resources with cleanup on all exit paths. Add keyword CSV inputs to `to_parquet`/`to_duckdb`. Fetch CSV mode never constructs Service; remote parameters are optional only in CSV mode, and combined remote/CSV input is rejected.

`export.import_csv(path_or_borrowed_stream, parquet_path, *, csv_options=None)` uses Polars scan/sink, creates no CSV temporary file and leaves borrowed streams open. `query_parquet(path, sql="SELECT * FROM results", *, parameters=None, database=":memory:")` supports a file or directory through a `results` view and returns Polars through Arrow with managed connection ownership. Both `ListManager.create_list` and `List.append` accept optional `csv_input`, `csv_options` and `csv_column` keyword arguments. List CSV inputs require `csv_column`, preserve identifiers including leading zeros as String, and reject server constraints/joins combined with CSV; local SQL uses `query_parquet`.

Restore all original helper contracts using fixture-backed behavior, including registry dictionary keys/print/None returns, stored-query version-27 query-vs-XML payloads, duplicate-name prompts and explicit overwrite, and plotting labels/return types. Identifier polling uses a fake-clock-tested initial interval 0.05 seconds multiplied by 1.25 with a 60-second cap. `where_in` constructs ONE OF explicitly; legacy IN/NOT IN refer to named lists, while native collection IN retains its existing meaning.

## Execution and review policy

Implement one task at a time on master, with a separate reviewed commit per task and push per phase. The checklist and ledger record task progress and verification evidence. Keep version 0.1.8 throughout; create no releases, tags or PyPI publication. Review each task before committing it. Preserve the user-owned untracked file `=1.7.1` and exclude it from every commit.

Use meaningful behavioral TDD for runtime changes: establish the specification, implement, run specification review and quality review, fix findings, then commit. Do not invent doctests or equate public names/signatures with compatibility. Retain native contracts and transport/export invariants. At each phase run full pytest, Ruff, contract/native checks and `git diff --check`; push master and verify GitHub Actions against the exact pushed SHA. Record tested evidence and intentional deviations per ledger symbol; final review covers every one of the 460 rows. Dependency and compatibility research must include GitHits pkginfo, batched reviews/changelogs/advisories, source search/grep/read, and completed search_status checks.

Dependencies: fixtures and profiles precede implementation; shared analytics primitives precede CSV/dataframe wiring; model and logic precede schema-dependent factories/XML/object results; result dispatch precedes summaries; lists precede Query set operations; Template constraints precede adjusted execution. Phase 1 factories may be partial until model/factory phases complete and must not claim unsupported behavior. Task 1.4 carries the profile and restores string-column select/new_query/query aliases, view aliases, order_by, size and positional where triples. Both profiles retain string roots/columns and dictionary rows until tasks 3.3/3.5/5.x; XML factory overloads remain task 4.2. Model inference applies only to actual restored Model instances; native Service.select always passes its profile explicitly.

## Phased checklist

### Phase 1: Foundations

- [x] **Task 1.1** — Audit, implementation plan and symbol ledger. Status: complete.
- [x] **Task 1.2** — Native/legacy model, protocol and response fixtures. Status: complete.
- [x] **Task 1.3** — Lazy original-module facades, query errors, UnimplementedError and utilities. Status: complete.
- [x] **Task 1.4** — Service/Query/Registry compatibility profiles, clone propagation, where triple and aliases. Status: complete.

### Phase 2: Analytics and explicit CSV pipeline

- [x] **Task 2.1** — Lazy core analytics dependencies and pandas extra removal. Status: complete.
- [x] **Task 2.2** — Bounded atomic schema-correct Parquet writer. Status: complete.
- [x] **Task 2.3** — query_parquet SQL through Arrow to Polars. Status: complete.
- [x] **Task 2.4** — import_csv scan/sink, borrowed streams and CSV options. Status: complete.
- [x] **Task 2.5** — CSV/fetch/dataframe wiring, conflict checks and resource cleanup. Status: complete.
- [x] **Task 2.6** — Query.export explicit format policy, empty results, errors and interruption. Status: complete.

### Phase 3: Model and constraints

- [x] **Task 3.1** — Logic nodes/parser/codes and QuerySpec/XML/clone propagation. Status: complete.
- [x] **Task 3.2** — Model fields/classes/paths/columns, expression trees and errors. Status: complete.
- [x] **Task 3.3** — Service.model cache, schema wildcards, rootClass and typed export. Status: complete.
- [x] **Task 3.4** — Named-list, loop, lookup, range and ISA constraints, factory and XML. Status: complete.
- [x] **Task 3.5** — Column DSL aliases, unary/subclass overloads and profile-specific IN. Status: complete.

### Phase 4: Factories and service metadata

- [ ] **Task 4.1** — Query.from_xml/load_query, PathDescription validation and minidom to_Node. Status: pending.
- [ ] **Task 4.2** — Service select/new_query/query class/field/reference/Column/XML factories. Status: pending.
- [ ] **Task 4.3** — Search with facets, widgets, release, resolve_service_path and metadata cache. Status: pending.
- [ ] **Task 4.4** — Registration, deregistration and anonymous tokens through shared opener. Status: pending.

### Phase 5: Result contracts

- [ ] **Task 5.1** — ResultRow/TableResultRow mappings, slicing, value iteration and flat streams. Status: pending.
- [ ] **Task 5.2** — ResultObject nested references/collections, prefetch, defaults and closure. Status: pending.
- [ ] **Task 5.3** — first/one, eager results/rows/all, cardinality and object grouping. Status: pending.
- [ ] **Task 5.4** — summarise/summary_path/summarize and historical dataframe call forms. Status: pending.

### Phase 6: Lists and enrichment

- [ ] **Task 6.1** — List/ListManager CRUD, metadata, naming, discovery and CSV identifier inputs. Status: pending.
- [ ] **Task 6.2** — List append/tags, context cleanup, temporary lists and Service.flush. Status: pending.
- [ ] **Task 6.3** — Upload/append URIs, query conversion, constraints, Query/List set operations and seven Service delegates. Status: pending.
- [ ] **Task 6.4** — EnrichmentLine, enrichment/widget options and Polars/Parquet persistence. Status: pending.

### Phase 7: Templates and identifier resolution

- [ ] **Task 7.1** — Template constraints, editability, switches, required classes and factory. Status: pending.
- [ ] **Task 7.2** — Template clone/XML, adjusted execution, results reuse and export. Status: pending.
- [ ] **Task 7.3** — Global/user template discovery, properties, cache and version behavior. Status: pending.
- [ ] **Task 7.4** — Identifier resolution submission and Job lifecycle with bounded polling. Status: pending.

### Phase 8: Registry and historical helpers

- [ ] **Task 8.1** — Legacy registry dictionaries, print/None returns, messages and Registry factory. Status: pending.
- [ ] **Task 8.2** — Five query_manager helpers, version-27 payloads and duplicate-name handling. Status: pending.
- [ ] **Task 8.3** — Five bar_chart helpers, original labels/returns and Polars/Matplotlib. Status: pending.

### Phase 9: Documentation, installation and final audit

- [ ] **Task 9.1** — Docs, samples, tooling, stale imports, Makefile/tox, native/pipeline contract and owned benchmarks using Polars + Parquet + DuckDB with no CSV default. Status: pending.
- [ ] **Task 9.2** — Clean install CI: base/plots, wheels/docs, lazy imports and analytics alias. Status: pending.
- [ ] **Task 9.3** — Final 460-symbol behavioral/deviation audit, Parquet/JSON coverage and review. Status: pending.

