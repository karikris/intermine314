# Service query factories

Task 4.2 restores descriptor and XML overloads on `Service.select`, `new_query`
and `query`. All three names share the same implementation. Evidence is in
`tests/test_service_query_factories.py` and `behavior-coverage.json`.

The comparison source is the BSD-2-Clause option of Python client 1.13.0 at
`d888b779c8050bad789e26b312f40d220bc85d0d`, acquired through GitHits:
[Service.select](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/webservice.py#L461).
Runtime queries use the existing shared builder, model cache and executor.

A single `Attribute` or `Field` uses its declaring class: an inherited
`Employee` name descriptor selects `Employable.name`. A `Class` selects that
class's attribute wildcard; a `Reference` or `Collection` selects attributes
under its declaring class and relation. Attribute Columns keep their actual
path, while relation/class Columns expand that path's wildcard. A single
reference path string, `Path`, or object with a reference path string expands
likewise. Wildcards retain sorted attributes, prefetch depth, id-only children
and OUTER joins.

Mixed selections accept descriptors, strings and nested lists/tuples/sets.
They retain every requested selection, including duplicates. Relative strings
and inherited Fields use the most specific compatible root: selecting an
inherited name Field alongside an Employee age Field yields `Employee.name`
and `Employee.age`. An explicit Class alongside an age Field still contributes
the Class's full attribute selection. Unrelated descriptor roots raise
`ModelError`. Columns' subclass refinements are installed before expansion and
validation; conflicting refinements raise `ModelError`.
Root inference considers every selected candidate before choosing one, so a
Manager title Field can establish the root for name and seniority Fields from
different parents regardless of their order. Without an explicit root, the
chosen class must occur among the selections and inherit from every candidate;
the factory does not search for an unselected common subclass. An explicit root
must likewise satisfy every candidate. Column order and duplicates survive
this root resolution.
Transferred refinements are checked in both profiles before wildcard expansion:
an unrelated subtype raises `ConstraintError`, and an unknown class raises
`ModelError`. Native queries retain their ordinary string validation default.

`xml=...` delegates to the executable, service-bound `load_query` restored in
task 4.1. Text, bytes, filenames, paths, URLs and borrowed streams follow that
loader's ownership and validation rules. Optional `root` is forwarded to XML
loading and can also establish the root for relative column selections.
XML combined with positional columns raises `TypeError`; unsupported factory
keywords also raise `TypeError`.

Both profiles keep their existing roots (native string, legacy Class), shared
model identity, cloning, transport and analytical export behavior. Native
unknown dotted strings remain permissive even with a full model. A single
unknown class string still attempts wildcard expansion and fails with a full
model, as before; with the existing unavailable/name-only model fallback it
remains a wildcard string. Multiple plain strings keep their previous native
or strict legacy validation rather than expanding each reference string.
Descriptor selections require a valid model and validate their resulting
attribute paths in both profiles.

These are scoped changes beyond the original client's special single-input
branches: mixed descriptors normalize consistently; Class descriptors use
their names rather than Python object representations; Column refinements
survive factory selection; optional root is supported; conflicts and unknown
keywords raise errors instead of silently discarding input. `Column.select`
still preserves task 3.5's query clone and reference-relative field behavior.
Service factories create a fresh query; selecting a query-bound Column through
the Service does not copy its existing query constraints.

Factory keywords do not reserve field names in `Query.where`: kwargs-only
`where(path=..., op=..., value=..., code=..., xml=..., root=...)` still means
field equality for every supplied keyword. Legacy object result defaults and
list operations remain tasks 5.2 and 6.x.
