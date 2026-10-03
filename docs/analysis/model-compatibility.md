# Standalone model compatibility

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
implemented legacy-profile inference. `Service.model`, schema-aware query
roots and selections, and integration of Column expressions into actual queries
remain tasks 3.3 and 3.5. `intermine314.model.operators` remains absent because
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
