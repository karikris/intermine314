# Model compatibility

Task 3.2 restores the original `intermine314.model` import shape: `Model`, class
and field descriptors, `ComposedClass`, validated `Path`, `Column`, expression
nodes and the model error classes. The implementation adapts pinned
`intermine-ws-python` 1.13.0, commit `d888b779c8050bad789e26b312f40d220bc85d0d`,
under the BSD option recorded in `LICENSE-BSD` and `NOTICE`.

`tests/test_model.py` exercises the attributed upstream 19-class XML fixture.
It verifies inherited descriptor identity, field ordering and types, reverse
references, composed classes, path navigation and subclass mappings, input
ownership, public errors and expression primitives. Column selection,
filtering, iteration and counting use an explicit fake service/query protocol;
these checks do not establish remote result compatibility. Import tests block
the original runtime and analytics libraries and verify that root logging
handlers remain unchanged.

`Model.table` aliases `column`, and `Column.filter` aliases `where` at class and
instance level. Constructing a direct `Query(Model)` activates the previously
implemented legacy-profile inference. Managed `Service.model` and schema-aware
query integration are documented below for task 3.3. Integration of Column
expressions into actual queries remains task 3.5. `intermine314.model.operators` remains absent because
upstream implements these objects in `intermine.model` itself.

The following scoped repairs intentionally differ from upstream bugs:

- Borrowed readable streams remain open, while owned files and responses close
  on success and parse failure. An opener failure retains its public parse
  error and cause instead of an uninitialized-variable cleanup error.
- Missing model metadata and multiple model elements produce public parse
  errors; harmless XML comments after the model do not imply a second model.
  Duplicate class declarations within one document are rejected rather than
  replacing descriptors. Public `parse_model` calls may still replace classes
  from an earlier parse, followed by `vivify` as in upstream.
- Inheritance cycles raise `ModelError`. Inherited fields and reverse links are
  resolved after all classes exist, so declarations do not depend on XML order.
  Incompatible field kinds, attribute types, unrelated reference types and
  conflicting reverse names raise public errors instead of silently replacing
  descriptors. Equivalent primitive/boxed types preserve inherited identity;
  legitimate narrower reference and collection types retain the child field.
  Unknown external Java ancestors, including `Object`, remain ignored as in
  upstream.
- `ComposedClass.parent_classes` includes ancestors of every part; upstream
  retained only the final part's ancestors.
- Omitted subclass mappings are independent dictionaries. Explicit mappings
  remain shared so Column branches can reflect subclass constraints.
- Extending a path past an attribute raises `PathParseError` instead of an
  internal `None.get_field` error. Path equality remains string-based; hashing
  now follows that same equality and stays stable if subclass mappings change.
- Constraint codes include `Z` and continue through `AA`, `AB`, and later
  codes. Codeless nodes stay in node iteration and contribute no Boolean logic.
  Column subclass comparisons invalidate cached branches and root columns honor
  subclass mappings during navigation without requiring a parent column.

These are tested standalone contracts, not a blanket assertion that every
original client call or server interaction is compatible.

## Managed model and analytical integration (task 3.3)

`Service.model` now fetches XML through the service's managed opener, closes the
response before parsing, and caches the XML, name, and attributed `Model`
together. Access through `select`, `model`, or name resolution shares that
payload. This retains configured TLS/proxy/session ownership. Direct property
access reports model parse errors. Native query factories retain permissive
string queries when the model is unavailable or only contains a name; that
fallback is not treated as a valid full `Model`. Legacy factories use strict
model validation.

Legacy queries keep the actual `Class` in `root` and expose `rootClass`; native
queries keep string roots. QuerySpec and XML use strings in both profiles.
Clones share their model and service, retain prefetch settings, and independently
copy query state. String wildcards expand sorted attributes and configured
reference/collection prefetch levels, including id-only child selections and
OUTER joins. Inherited attributes and subclass mappings inform path validation,
wildcard expansion, and analytical schemas. Actual `Query.column` expression
integration and descriptor/XML factory forms remain later tasks.

Parquet and dataframe exports resolve model types lazily. Strings preserve
leading zeros; Boolean, Byte/Short/Integer/Long, Float/Double map to their Polars
scalar widths. Model Float values adopt IEEE Float32 representation and reject
finite overflow. `java.util.Date` maps to calendar `Date`: the server serializes
it as `yyyy-MM-dd`, not as a timestamp. Unknown selected paths or unsupported
model types fail explicitly during analytical schema resolution, including for
native `validate=False` queries. With no full model, the original bounded schema
inference contract remains available; it cannot promise model types or recover
precision already lost by ordinary JSON float decoding.

BigDecimal exports opt into exact JSON-number decoding on an isolated query
clone. The immutable QuerySpec carries the selected decimal paths through the
shared executor and row iterator, including parallel pages. Ordinary native and
legacy dictionary rows keep float decoding. No borrowed query, service, or
transport state is changed. Integer numeric tokens retain their exact Python
integer values; decimal numeric tokens are parsed before a Python float can
round them.

Model metadata provides no decimal scale. Each bounded batch resolves a
concrete `Decimal(38, scale)`; every value must fit that shared batch scale
before typed construction. A value comparison also rejects unexpected null
insertion or truncation by the typed constructor. The writer reconciles all
staged parts to the largest observed scale before publication. Empty/all-null decimals use
`Decimal(38, 0)`. Values exceeding precision 38, scales above 38, or combinations
of integer digits and later scales that cannot fit fail atomically. Earlier
parts remain private, and old destination files/directories survive failures.
The existing Int128 normalization, timestamp precision guards, bounded staging,
DuckDB merge, and Linux/portable publication guarantees remain in effect.
Parallel pages explicitly close their iterators even when the page limit stops
iteration before exhaustion. CSV input continues to bypass remote schema access.

Executed evidence is in `tests/test_model_query_integration.py`: both HTTP row
protocols and compatibility profiles, single/parts Parquet followed by DuckDB
SQL/Arrow/Polars, null-first batches, changing scales, >2**53 numeric tokens,
negative Long values, calendar dates, zero rows, model cache/response ownership,
parallel precision and ordinary-row isolation, default root selections, nested
subclass schemas, strict errors, overflow rollback, and explicit page closure.
This supplements the existing writer, CSV, transport, and lazy-import tests.

Source evidence: pinned client `query.py` wildcard/path checks at
`d888b779c8050bad789e26b312f40d220bc85d0d` (BSD attribution above); server
[`MinimalJsonIterator`](https://github.com/intermine/intermine/blob/77cf7068dad0beac153e93e9916997d0ea850372/intermine/webapp/src/main/java/org/intermine/webservice/server/output/MinimalJsonIterator.java)
and [`ConstraintValueParser`](https://github.com/intermine/intermine/blob/77cf7068dad0beac153e93e9916997d0ea850372/intermine/pathquery/src/main/java/org/intermine/pathquery/ConstraintValueParser.java)
establish calendar-date wire formatting. Exact analytical Decimal decoding and
bounded reconciliation are deliberate integration improvements over ordinary
upstream JSON floats, not an assertion of upstream analytical API parity.
