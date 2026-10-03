# InterMine Python API compatibility assessment

Historical baseline assessment on 3 October 2026 against `intermine314` 0.1.8
commit `41c363e5aff0d3a72e3c93375bde8d33b6d8fec2`. The measured gaps and
recommendations below describe that baseline, not the current restored tree.
Current executed behavior is recorded in the linked task reports and ledger.

`intermine314` currently preserves a subset of the original Python client's query API. Existing programs cannot migrate reliably by changing `intermine` imports to `intermine314`: important import paths and aliases are absent, several retained names behave differently, and model, template, list and object-result functionality has been removed.

The recommended direction is a compatibility API over the current transport and query executor. Start with imports and ordinary query calls, then restore result contracts and the model/constraint features that give those calls their meaning. Keep the parallel export, Parquet, DuckDB and transport improvements available alongside that API.

## Sources and scope

| Project | Baseline | Role in this assessment |
| --- | --- | --- |
| Original Python client, `intermine` | PyPI 1.13.0; tag `1.13.0`, commit `d888b779c8050bad789e26b312f40d220bc85d0d` | Original names, signatures and implemented behavior. |
| `intermine314` | 0.1.8, commit `41c363e5aff0d3a72e3c93375bde8d33b6d8fec2` | Current implementation, documentation and tests. |
| User-supplied `intermine/intermine` repository | Source snapshot `77cf7068dad0beac153e93e9916997d0ea850372` | Server capabilities behind both Python clients. |

The supplied [InterMine repository](https://github.com/intermine/intermine/) contains the Java server. The original Python package comes from [intermine-ws-python](https://github.com/intermine/intermine-ws-python/tree/d888b779c8050bad789e26b312f40d220bc85d0d). GitHits provided current package information, repository searches, grep results, search-status completion and source reads. The local ancestral tag resolves to the same upstream Python commit, allowing a source inventory without importing the old package into the modern environment.

The accompanying [API inventory](intermine-api-inventory.parquet) contains 460 rows across all 17 original Python modules. It records declared public classes, functions, methods and properties, constructor signatures, explicit aliases, Python protocol implementations and the seven dynamically delegated service list methods. Repeated definitions use the final implementation. Inherited methods are covered through their declaring classes; incidental imported names, ordinary data attributes and class constants are outside the inventory. Important public exception re-exports are assessed separately below.

The inventory records whether a name was located, not whether it is behaviorally compatible. Protocol rows describe explicit implementations; an absent override can still leave a default inherited Python method. Source inspection and offline probes assessed the important calling patterns below. No live mine integration suite was run, and the server snapshot does not establish which features every deployed mine enables.

## Measured API gaps

| Original class | Legacy named members | Names located in current implementation | Names absent |
| --- | ---: | ---: | ---: |
| `Query` | 57 | 33 | 24 |
| `Service` | 30 | 4 | 26 |

These counts include methods, properties and aliases; `Service` also includes its delegated list methods. Constructors and Python protocol methods are excluded from this table. The four retained service names are `version`, `select`, `model` and `get_results`. `model` raises `NotImplementedError`; `select` and `get_results` have compatibility differences. Several of the 33 query names also have changed or disabled behavior.

The removals are intentional in the current project contract. The [README](../../README.md) describes alias removal, [test_minimal_surface.py](../../tests/test_minimal_surface.py) asserts absence of legacy methods, and [minimal_public_contract.json](../../benchmarks/contracts/minimal_public_contract.json) forbids many original features. Achieving the requested alignment therefore requires updating that policy and its tests.

## 1. Restore original import paths

For the usual migration, this should work:

```python
from intermine314.webservice import Service
```

It currently raises `ModuleNotFoundError`. The existing entry point is `intermine314.service.Service`.

| Original import, after replacing the package name | Current location or gap | Suggested alignment |
| --- | --- | --- |
| `intermine314.webservice.Service`, `Registry` | `intermine314.service`; implementations in `service.service` | Add a lightweight `webservice` facade. |
| `intermine314.query.Query` | Available | Keep it and add the legacy query exception exports. |
| `intermine314.query.QueryError`, `ConstraintError`, `QueryParseError`, `ResultError` | Classes exist in `query.builder`, but are absent from the `query` package exports | Re-export the existing exception classes, preserving identity for exception handlers. |
| `intermine314.constraints` | Partial implementation in `query.constraints` | Add the original module facade; implement missing constraint and logic classes. |
| `intermine314.pathfeatures` | Partial implementation in `query.pathfeatures` | Add facade and restore `PathDescription`. |
| `intermine314.results` | Some helpers and iterators in `service.session`; row/object/flat-file classes absent | Add facade and restore the result adapters. |
| `intermine314.errors` | `ServiceError` and `WebserviceError` in `service.errors`; `UnimplementedError` absent | Re-export existing errors and restore the missing error class. |
| `intermine314.util` | `ReadableException` survives; `openAnything` absent | Restore the stream/file/URL input helper where its callers require it. |
| `intermine314.model`, `lists`, `idresolution` | Feature implementations absent | Restore the feature modules, not placeholder exports. |
| `intermine314.registry`, `query_manager`, `bar_chart`, `decorators` | Original module functions absent | Restore compatible wrappers or implementations, prioritizing registry and stored-query use. |

These recommendations preserve imports within the `intermine314` namespace. Providing a separate top-level `intermine` package would create an additional packaging decision and is unnecessary for a package-name substitution migration.

The current [query exports](../../src/intermine314/query/__init__.py) and [service exports](../../src/intermine314/service/__init__.py) already use lazy imports. Extend that pattern so compatibility facades do not eagerly load pandas, plotting libraries or unrelated features.

## 2. Restore aliases with the correct targets

The original query constructor installs ten aliases, while the service exposes `new_query` and `query`. They are absent in 0.1.8. Their targets are explicit in the upstream [Query constructor](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/query.py#L377) and [service factory](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/webservice.py#L433).

| Legacy name | Original target | Work needed |
| --- | --- | --- |
| `Service.new_query`, `Service.query` | `select` | Add aliases after restoring the original `select` overloads. |
| `Query.add_column`, `add_columns`, `add_views`, `add_to_select` | `add_view` | Straightforward aliases; verify append behavior and returned value. |
| `Query.order_by` | `add_sort_order` | Straightforward alias; preserve ordering and mutation behavior. |
| `Query.size` | `count` | Straightforward alias. |
| `Query.filter` | `where` | Fix legacy argument forms and expression trees first. |
| `Query.c` | `column` | Restore the `Column` expression object first. |
| `Query.all` | `get_results_list` | Restore eager list materialization; aliasing to `rows` would change the return type. |
| `Query.summarize` | `summarise` | Restore column summaries first; `count` is a different operation. |

The original model API also has `Model.table -> column` and `Column.filter -> where`. Restore those alongside the model feature.

Adding these names makes discovery and familiar examples easier, but aliases alone do not establish compatibility.

## 3. Fix retained names whose calls changed

### Query construction and column expressions

The original positional `where` call fails today even though the method still exists. This offline reproduction constructs a current query without contacting a mine:

```python
from intermine314.query import Query

q = Query(root="Gene", validate=False).select("Gene.symbol")
q.where("symbol", "=", "eve")
# AttributeError: 'str' object has no attribute 'path'
```

The current implementation accepts `q.where(("symbol", "=", "eve"))`, but requiring users to add tuples defeats the compatibility objective. Restore argument normalization for the positional triple, tuple constraints, keyword constraints and column-expression trees. Preserve the original distinction: `where` returns a clone, while `add_constraint` mutates the query. See the upstream [where implementation](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/query.py#L803) and current [builder](../../src/intermine314/query/builder.py).

`column` now returns a string rather than a model-backed `Column`:

```python
q.column("symbol") == "eve"   # False, rather than a constraint node
q.column("length") > 1000    # TypeError
```

This can silently change expression meaning. Restoring `c` without restoring `Column` would leave the problem intact. Implement overloaded comparisons, membership and Boolean composition with the original node interfaces. The upstream [model module](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/model.py#L630) defines these expression types.

`Service.select(*columns, **kwargs)` previously accepted `xml=...`, model `Attribute`/`Reference`/`Column` arguments and schema-based reference expansion. The current signature is `select(*columns)`, and queries are constructed with validation disabled. Restore those overloads when introducing `new_query` and `query`; a direct alias currently handles only part of their old input contract.

### Model and validation

`Service.model` is still a property but always raises `NotImplementedError`. The implementation fetches only the model name for serialization, rather than exposing the schema. `Query.root` is now a string rather than an original model `Class` object.

Consequently, model navigation and expressions such as `service.model.Gene`, field lookup and type-aware validation are unavailable. An offline probe using `Gene.notARealAttribute` passed `verify()` because the path is syntactically valid. Restore `Model`, `Class`, fields, references, collections, `Path`, `Column` and the corresponding model errors. Keep schema loading lazy and cached. The current behavior is explicit in [Service.model](../../src/intermine314/service/service.py) and the validation routines in [Query](../../src/intermine314/query/builder.py).

### Constraint logic

The original `get_logic()` returns a logic object; the current method returns an empty string. `set_logic()` and `validate_logic()` always raise `NotImplementedError`. The current [QuerySpec and XML encoder](../../src/intermine314/query/spec.py) contain no constraint-logic field.

Restore `LogicNode`, `LogicGroup` and `LogicParser`, carry the expression through clone/spec/execution, and serialize `constraintLogic`. Validate grouping, precedence and references to constraint codes. Sequential filters cannot reproduce `A or (B and C)`. Merely accepting `set_logic` without serializing its expression would produce incorrect results.

### Named lists and advanced constraints

In the original API, `IN` and `NOT IN` identify membership in a named server list. `ONE OF` and `NONE OF` identify scalar membership in supplied values. The current factory rewrites `IN` to `ONE OF` and requires a Python collection:

```python
q.add_constraint("Gene", "IN", "my-list")
# TypeError: values must be a list, tuple, or set
```

Restore this distinction and `ListConstraint`. Preserve the modern `where_in` convenience by having it construct `ONE OF` explicitly; update current collection-valued keyword construction similarly so it does not start creating server-list constraints. The contracts appear in the upstream [ListConstraint](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/constraints.py#L565) and the current [ConstraintFactory](../../src/intermine314/query/constraints.py).

The two-argument forms changed as well. `add_constraint("Gene.symbol", "IS NULL")` now produces equality to the string `"IS NULL"`; `add_constraint("Gene.proteins", "Protein")` produces equality to `"Protein"` rather than a subclass constraint. Restore the original overload dispatch. The [upstream subclass constructor](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/constraints.py#L875) and factory define this two-argument form.

The unary keyword form also changed: `add_constraint(symbol="IS NULL")` produces literal-string equality. Restore unary-operator recognition for this specific overload. Do not assume the same recognition applied to the original `where` keyword overload.

Additional absent types include `LoopConstraint` for reference comparisons, `TernaryConstraint` for `LOOKUP` and its extra value, range constraints, `IsaConstraint`, and template-specific constraints. Extend both factory dispatch and XML serialization; restoring class names alone is insufficient. The current `where_raw` still calls the restricted factory, so it does not bypass these limitations.

## 4. Preserve result formats, indexing and return types

| Call | Original default or behavior | Current behavior |
| --- | --- | --- |
| `query.results()` | `row="object"`, yielding object results | Dictionary rows only. |
| `query.rows()` | `row="rr"`, yielding `ResultRow` | Dictionary rows only. |
| `iter(query)` | Object results through `jsonobjects` | Dictionary rows. |
| `query.results(summary_path=...)` | Server-side column summary | Parameter removed. |
| `row[0]`, `row[:2]`, `row["symbol"]`, `row("symbol")` | Supported by `ResultRow` | A plain dictionary does not provide this contract. |
| `for value in row` | Iterates row values | A dictionary iterates keys. |
| `result.symbol`, nested references/collections | Model-backed object access | Object-result implementation absent. |

The upstream [results implementation](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/results.py#L182) establishes the indexing and value-iteration behavior. The current [ResultIterator](../../src/intermine314/service/session.py) restricts formats to `dict`. Originally accepted formats include `object`/`jsonobjects`, `rr`, `dict`, `list`, `csv`, `tsv`, `count`, `json` and `jsonrows`.

Restore `ResultRow`, `TableResultRow`, flat-file parsing, and the result format dispatch over the current streamed transport. A flat dictionary adapter can supply row indexing; object results also need the server's nested object format, model information and reference/collection handling. Preserve prefetch behavior where callers use it.

To avoid changing existing 0.1.8 dictionary callers unexpectedly, introduce the old defaults through the legacy `webservice` facade or an explicit compatibility mode during transition. Give that entry point a clear contract and ensure its queries retain the selected behavior when cloned. Full interchangeability between entry points requires reconciling defaults before claiming complete parity.

## 5. Restore missing query helpers

The fourteen original named members missing beyond the ten aliases are:

| Area | Missing names | Suggested behavior |
| --- | --- | --- |
| XML import/export | `from_xml`, `to_Node` | Restore round trips, including logic/joins/constraints; `to_Node` must supply a `xml.dom.minidom` element, not an ElementTree element. |
| Path labels | `add_path_description`, `verify_pd_paths` | Restore `PathDescription`, schema checks and XML serialization. |
| Materialization | `get_results_list`, `get_row_list` | Return eager lists with the original formats. |
| Cardinality | `one`, `first` | `one` raises `QueryError` for zero or multiple objects; `first` returns `None` when empty. Preserve defaults and iterator cleanup. |
| Summaries | `summarise` | Use the server's `summaryPath`; return numeric statistics with numeric conversion or categorical value-to-count mappings. |
| Dataframes | `dataframe` | Return a Polars `DataFrame` in both profiles; document this intentional deviation from upstream pandas. |
| List integration | `get_list_upload_uri`, `get_list_append_uri`, `to_query`, `make_list_constraint` | Restore query/list interoperability. |

The implementations of [summarise, one, first and list materialization](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/query.py#L1469) define their contracts. The source also contains two `dataframe` definitions; the second is the effective implementation, returning pandas.

For summaries, test the actual returned keys against response fixtures: the Python documentation mentions `stdev`, while the inspected [server summary headers](https://github.com/intermine/intermine/blob/77cf7068dad0beac153e93e9916997d0ea850372/intermine/webapp/src/main/java/org/intermine/webservice/server/query/result/QueryResultService.java#L267) list `standard-dev`. The Python implementation derives keys from the returned result and converts numeric values to floats, rather than hard-coding the documented name. Preserve that behavior instead of inventing a new summary schema.

Original query set operators `&`, `|`, `+`, `-` and `^` also depend on server lists and are absent. They need list functionality, not just overloaded operator names.

## 6. Restore feature families and service functions

| Feature | Original names callers can depend on | Recommended implementation |
| --- | --- | --- |
| Templates | `get_template`, `get_template_by_user`, `templates`, `all_templates`, `all_templates_names`, `query.Template` | Restore template discovery, editable constraints, switches and execution-time overrides. |
| Lists | `list_manager`, `get_list`, `l`, `get_all_lists`, `get_all_list_names`, `get_list_count`, `create_list`, `delete_lists`; `List` and `ListManager` | Restore explicit service delegation, CRUD, query uploads/appends, tags, set operations, enrichment and temporary-list cleanup. |
| Identifier resolution | `resolve_ids`; `idresolution.Job.poll`, `fetch_status`, `fetch_results`, `delete` | Restore asynchronous job submission and lifecycle using current transport. |
| Discovery and metadata | `search`, `widgets`, `release`, `resolve_service_path` | Match signatures and return shapes; notably, `search` returns results and facets. `release` is warehouse release metadata. |
| Account operations | `register`, `get_deregistration_token`, `deregister` | Restore supported service endpoints with the original signatures and exceptions. |
| Cache lifecycle | `flush` | Invalidate cached metadata and clean temporary lists. It is not equivalent to `close`. |
| Anonymous tokens | `get_anonymous_token` | Expose a wrapper over the existing private token helper. `token="random"` already survives in the constructor. |
| Registry functions | `getVersion`, `getInfo`, `getData`, `getMines` | Restore camelCase module wrappers over current discovery/transport. Preserve legacy output and returns. |
| Stored queries | `save_mine_and_token`, `get_all_query_names`, `get_query`, `delete_query`, `post_query` | Restore the original `query_manager` entry points over stored-query endpoints. |
| Plotting | `bar_chart.save_mine_and_token`, `plot_go_vs_p`, `plot_go_vs_count`, `get_query`, `query_to_barchart_log` | Restore callable names with lazy optional plotting dependencies. |
| Utilities | `requires_version`, `openAnything`, `encode_headers` | Restore original helper names and their input/output contracts where required. |

The [upstream service implementation](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/webservice.py#L314) delegates seven list methods; merely scanning its explicit method definitions would miss them. All are included in the Parquet inventory.

Registry wrappers deserve particular care: original `getInfo`, `getData` and `getMines` print information and return `None` on success, with diagnostic strings for certain missing results. `getVersion` returns a specifically keyed dictionary. Current `Registry.info` and `all_mines` offer related discovery but are not direct return-compatible substitutes. See the [original registry functions](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/registry.py).

The retained HTTP opener methods are also recorded in the inventory. Their current managed-session implementation should be retained, with response, exception, authentication and ownership behavior verified where exposed through compatibility modules.

## 7. The server still implements the missing capabilities

Inspection of the supplied server repository confirms the following implementations in the pinned snapshot:

| Capability | Server source evidence | Implication |
| --- | --- | --- |
| Model schema | [ModelService](https://github.com/intermine/intermine/blob/77cf7068dad0beac153e93e9916997d0ea850372/intermine/webapp/src/main/java/org/intermine/webservice/server/model/ModelService.java#L85) serves XML/JSON model data and resolves schema paths. | Model introspection can be restored over an existing endpoint. |
| Result formats and summaries | [QueryResultService](https://github.com/intermine/intermine/blob/77cf7068dad0beac153e93e9916997d0ea850372/intermine/webapp/src/main/java/org/intermine/webservice/server/query/result/QueryResultService.java#L133) accepts multiple format families; its summary handling reads `summaryPath` and builds numeric/categorical headers. | Dictionary-only results and missing summaries are client restrictions in this snapshot. |
| Templates | [TemplateResultService](https://github.com/intermine/intermine/blob/77cf7068dad0beac153e93e9916997d0ea850372/intermine/webapp/src/main/java/org/intermine/webservice/server/template/result/TemplateResultService.java#L37) resolves templates and applies supplied constraint values. | Restore template execution with server-side overrides. |
| Lists | [ListUploadService](https://github.com/intermine/intermine/blob/77cf7068dad0beac153e93e9916997d0ea850372/intermine/webapp/src/main/java/org/intermine/webservice/server/lists/ListUploadService.java#L52) accepts list uploads and metadata. | List creation remains a server feature. |
| Identifier jobs | [IdResolutionService](https://github.com/intermine/intermine/blob/77cf7068dad0beac153e93e9916997d0ea850372/intermine/webapp/src/main/java/org/intermine/webservice/server/idresolution/IdResolutionService.java#L43) submits identifiers and options to the resolver and returns a job UID. | Restore job submission and the client lifecycle API. |

These findings support restoring client functionality. They do not imply that every endpoint is enabled or accessible anonymously on every mine. Preserve original version checks and test supported deployments when implementing the corresponding feature.

## 8. Repair documentation and align validation with the objective

Existing examples already expose some of the inconsistencies:

- [samples/alleles.py](../../samples/alleles.py) and [samples/polars_parquet_duckdb.py](../../samples/polars_parquet_duckdb.py) import the missing `intermine314.webservice` module.
- [samples/common.py](../../samples/common.py) imports absent `intermine314.constants`. That module is not part of the original 1.13.0 API, so repair the sample to use current runtime configuration separately from the legacy compatibility work.
- [docs/source/intermine314.rst](../source/intermine314.rst) documents several absent modules, including model/constraint-related and helper APIs.
- [benchmarks/discover_model_paths.py](../../benchmarks/discover_model_paths.py) uses `service.model`, which currently raises.
- Current query docstrings still refer to the absent `webservice` and `model` APIs, including a `service.model.Gene` results example.

Replace relevant absence assertions with behavioral compatibility fixtures. Keep modern transport and export invariant tests. Proposed acceptance coverage should include:

1. Original imports with only the package prefix substituted, including public exceptions.
2. Alias calls, signatures, mutation-versus-clone behavior and returned types.
3. `where(path, op, value)`, tuple/keyword forms and `Column` expression trees.
4. XML round trips with grouped logic, joins, path descriptions and each constraint family.
5. Named server lists versus scalar membership, including unary keyword constraints.
6. All original result formats, `ResultRow` indexing/value iteration, nested objects and prefetch behavior.
7. Empty/one/multiple-result cases for `first` and `one`, plus eager `all`/list helpers.
8. Summary response shapes and the intentional Polars dataframe return type without eager analytics imports.
9. Template overrides, list lifecycle/set operations, identifier jobs and registry return/output behavior.
10. Early iterator termination and managed resource closure through compatibility adapters.

The source inventory and offline probes are evidence for this assessment, rather than a claim that these proposed acceptance tests already pass.

## Suggested implementation order

| Priority | Work | Completion criterion |
| --- | --- | --- |
| First | Change the minimal-surface contract; add legacy import facades and error exports; repair positional `where`; add view/order/count aliases. | Ordinary query construction and the initial legacy import examples work with their original call forms. |
| Next | Restore `select` overloads/XML loading, row formats/adapters, `first`, `one`, list materialization and summaries. | Common query execution preserves indexing, default modes, cardinality and return types. |
| Next | Restore the model/Column API, grouped logic and advanced constraints, including named-list semantics. | Expression-driven and schema-validated queries retain their meaning and serialized requests. |
| Then | Restore templates, list management/enrichment, identifier jobs and remaining service/registry helpers. | Feature-specific migration fixtures pass against supported server versions. |
| Then | Restore stored-query/plotting helpers and finish documentation/examples. | Every supported original public callable has an implementation and a documented, tested contract. |

Some stages depend on one another: XML and object results need parts of the model; summaries need result dispatch; query set operations need lists. Track those dependencies explicitly rather than publishing aliases that promise an unsupported operation.

The most useful first milestone is a tested legacy `webservice` entry point that handles the standard `new_query`/`add_view`/`add_constraint`/`rows` workflow. Complete alignment requires the feature work above, including names that survive today with incompatible behavior.

## Accepted implementation direction

The [37-task implementation plan](../superpowers/plans/2026-10-03-intermine-compatibility.md) and [symbol ledger](implementation-ledger.json) make the selected architecture concrete. Legacy facades preserve original calling conventions while native service defaults remain stable; both use the shared managed transport and executor. Analytics uses lazy Polars, Parquet, DuckDB and Arrow, with explicit CSV input/output only. Polars dataframe returns intentionally replace the upstream pandas contract in both profiles. Historical source findings above describe the baseline, not completed restoration. Phases 1–8 now have executed behavioral evidence. The complete 460-row final
audit remains task 9.3; baseline name counts above are preserved as historical measurements.
