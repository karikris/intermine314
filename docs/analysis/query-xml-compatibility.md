# Saved query XML compatibility

Task 4.1 restores `Query.from_xml`, `Service.load_query`, `PathDescription`,
`Query.add_path_description`, `Query.verify_pd_paths` and `Query.to_Node`.
The executed offline evidence is in `tests/test_query_xml_loading.py` and
`behavior-coverage.json`. Task 4.2 Service descriptor/XML factory overloads are
documented in `service-query-factories.md`. Task 7.1 Template constraint state is
documented in `template-compatibility.md`; Template wrapper XML and adjusted
execution remain task 7.2.

The comparison source is the BSD-2-Clause option of the original Python client
1.13.0 at `d888b779c8050bad789e26b312f40d220bc85d0d`, acquired through GitHits.
Relevant source is [Query.from_xml and its validation](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/query.py#L426),
[PathDescription](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/pathfeatures.py#L61),
and [Service.load_query](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/webservice.py#L447).
Server source evidence acquired through GitHits at InterMine commit `77cf7068`
confirms `PathQueryBinding.java` lines 145–154 writes `pathString`, and
`PathQueryHandler.java` lines 189–212 reads `pathString`.

XML text, bytes, local filenames, pathlib paths, URLs and borrowed readable
streams are supported. Exactly one query element is required, including when
it appears within a wrapper document. Name, description, views, joins, sort
orders, path descriptions, constraint families and explicit logic survive
loading and serialization. Constraints are installed before view expansion
and final model validation, so subclass fields resolve correctly. Explicit
codes are reserved before generated codes, including codes beyond Z. Ordinary
Query ignores Template-only `editable`/`switchable` XML attributes.

Both profiles use the canonical QuerySpec encoder and the same Executor.
`to_Node` converts that encoded query to a minidom Element with an owner
Document. Native roots remain strings; legacy roots remain model Classes.
`Service.load_query` retains the Service, profile and prefetch settings, uses
its model resolver (strict legacy and existing native fallback), and produces
an executable query. Bound URL sources use the same configured managed opener
as model and result requests, preserving authentication, TLS settings and
Tor/proxy configuration. Borrowed sessions retain their existing ownership
and configuration contract.

These departures repair specific original-client behavior:

- PathDescription's public `to_dict()` retains `path`, while XML emits the
  server's canonical `pathString`. The upstream serializer wrote `path`,
  despite its own XML reader requiring `pathString`. Loading accepts either
  spelling for previously saved queries and rejects conflicting spellings.
- Owned inputs close on parse and validation failures; borrowed streams remain
  open on success and failure. Malformed XML raises public `QueryParseError`
  with its underlying cause. Original code closed borrowed inputs and leaked
  owned inputs when parsing failed.
- `Service.load_query` binds the Service instead of producing an unbound
  query. Original code forwarded only its model and optional root.
- Explicit empty scalar values and empty child values are retained. Empty
  scalar placeholders on unary, subclass, loop and child-valued constraints
  retain upstream tolerance. The original reader removed all empty attributes,
  losing valid binary `value=""` constraints.
- Zero-value Multi, Range and ISA collections round-trip as empty lists when
  their XML contains no value children, matching the canonical encoder's empty
  collection output. For the overlapping `CONTAINS` operator, a scalar `value`
  attribute selects BinaryConstraint (including `value=""`); its absence
  selects the child-valued RangeConstraint, including an empty collection.
- A missing constraint path raises `QueryParseError`; a parent `<node path>`
  supplies the path when present. Original code tested for `None`, even though
  minidom returns an empty string for missing attributes, so its fallback was
  unreachable. Invalid constructor arguments raise `ConstraintError` with the
  original cause; genuine model and constraint validation errors propagate.
  Missing/empty operator and subclass type are rejected as `ConstraintError`
  before the field shorthand API can reinterpret a path-only XML constraint.
- XML logic follows the restored public logic parser and requires every coded
  constraint. Unknown codes and malformed expressions are rejected with public
  query/parser errors. Original `_set_questionable_logic` attempted to remove
  unused codes and repair operators; its error path could itself crash on
  `e.message`. This is a deliberate narrower departure: `A and Z` is rejected
  rather than silently becoming `A`. The already restored nested-group repair
  remains in effect.

Irrelevant XML sort paths are ignored as in the original reader, including
invalid model paths outside the view. Selected paths retain their directions,
and a selected sort without a direction defaults to ascending. Query clones
own independent path-description objects and code allocators. Explicit CSV
input rejects populated path descriptions through the existing empty-query
check, preventing these query settings from being silently discarded.
