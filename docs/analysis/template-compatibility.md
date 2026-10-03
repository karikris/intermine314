# Template compatibility

Task 7.1 restores `TemplateConstraint`, its nine typed variants and
`TemplateConstraintFactory`, exposed lazily through `intermine314.constraints`.
`intermine314.query.Template` now constructs with that factory and exposes
`editable_constraints`, including editable codeless subclass refinements.
The executed evidence is in `tests/test_template_constraints.py`.

The BSD-2-Clause source is Python client 1.13.0 at
`d888b779c8050bad789e26b312f40d220bc85d0d`, acquired through GitHits:
[template constraints](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/constraints.py#L911)
and [editable constraints](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/query.py#L1906).

Constraints default to editable and required (`locked`), with active state true.
`optional='on'` and `optional='off'` enable optional active/inactive states.
`required`, `switched_off` and `get_switchable_status()` reflect current state.
Only editable optional constraints permit `switch_on()` and `switch_off()`;
other calls raise `ValueError('This constraint is not switchable')`. Invalid
optional states raise `TypeError('Bad value for optional')`. Human strings append
`(editable, locked)` or the corresponding editability/status combination.
`separate_arg_sets` returns independent ordinary/template argument dictionaries.
Template subclass human strings and representations use the upstream `ISA`
wording, for example `Employee ISA Manager (editable, locked)`. The existing
ordinary SubClassConstraint human string remains `Employee Manager`.

All variants inherit their ordinary constraint class. The template factory
shares existing native/legacy dispatch, operator aliases, argument binding,
model checks, named-list upload behavior and collision-free code allocation
(including codes beyond Z). Constraints keep their ordinary `to_dict()` wire
shape at this stage. Native Binary `None` XML values remain empty strings;
legacy values remain `None` strings. Model-derived direct templates infer the
legacy profile, with actual model Classes as roots; explicit native templates
retain string roots. Service/model/prefetch bindings use Query construction.

Two intentional departures repair source behavior. The original variant
argument splitter compares editable input only to `'true'`, turning Python
`True` into false. Both booleans and canonical XML `'true'`/`'false'` now produce
consistent booleans, including direct mixin construction. Invalid optional
state is validated before a named-list upload, preventing that side effect
before rejection. Codeless subclass constraints remain eligible for editability,
as permitted by source constructors despite the source mixin's contrary prose.

Task 7.1 established the constraint foundation. Task 7.2 adds the named-template
behavior described below. Ordinary Query imports continue to ignore template
XML flags. Discovery/caching remains task 7.3.

## Named Template execution and XML (task 7.2)

`Template.from_xml` now parses one resource once, using Query's managed source
loader. It preserves borrowed readables and closes owned files, URLs and parser
failures. Bound URLs use the Service opener, credentials and TLS settings.
The wrapper supplies name, title, userName, dataTypes and optional comment;
constraints retain editable and switchable flags. `clone` preserves independent
metadata lists, constraints, switches, logic and code allocation while sharing
Model and Service. Model roots and native/legacy profiles retain Query semantics.

Serialization emits the server's canonical `<template>` wrapper around the
query, with editable booleans and optional on/off constraint attributes.
Required constraints omit switchable. The original client inherited ordinary
Query serialization and lost wrapper/flags; this is an intentional repair,
verified against [TemplateQueryBinding](https://github.com/intermine/intermine/blob/77cf7068dad0beac153e93e9916997d0ea850372/intermine/pathquery/src/main/java/org/intermine/template/xml/TemplateQueryBinding.java#L114).
All nine typed variants remain constructible and serializable, including
editable codeless subclass refinements.

Named calls send `name`, `userName`, then numbered `constraintN`, `opN`, `codeN`,
`valueN` and optional `extraN` fields in constraint order. Both codes on the same
path survive. Inactive and noneditable constraints are omitted; active numbers
remain contiguous. This repairs the original `next` expression that failed to
skip switched-off constraints. Collections use repeated value fields through
the shared form encoder. Native scalar forms (`None`, booleans and numbers)
remain unchanged; upstream preserved collections but this port previously
stringified them before `doseq` encoding.

`get_adjusted_template` clones before applying scalar `value` shorthand or
`op`/`value`/`values`/`extra_value` mappings. Historical `list_name`
adjustments remain supported alongside the `value` alias (specifying both is
rejected). Boolean `switched_on` adjustments use validated optional switches;
required constraints cannot be switched. Unknown and noneditable codes,
unknown fields, operators outside the existing typed family and invalid
collection/scalar shapes fail before HTTP. List constraint scalar adjustments
update the actual list name. Other arbitrary attributes such as `path`, `code`, `editable` and `optional`
are not adjustment fields; edit a clone explicitly when changing template
structure. Unlike the original unrestricted `setattr`, these
checks prevent silently ignored or malformed adjustments. Caller state remains
unchanged, including on failure.

Immutable `TemplateMetadata` on QuerySpec carries the saved identity and
presentation data; Template specs snapshot constraints. QueryExecutor chooses
`/template/results` and the Template form, then uses the existing managed
opener/result parser. Results, rows, eager aliases, first/one, count/size,
summaries, batch iteration and analytics all execute the adjusted Template.
Native default results/rows remain dictionaries; legacy defaults remain objects
and ResultRows. `summary_path` is an execution option for streaming and eager
results. Analytics and parallel helpers also accept constraint-code keywords
alongside their ordinary Query options. Both profiles retain the shared
Polars → Parquet → DuckDB → Arrow → Polars path, dynamic exact decimal scale,
empty model schemas, bounded strict schema conversion, atomic publication and
owned-stream cleanup on parser failures and BaseException interruption.
Parallel support remains `auto`/`offset`; `keyset` is rejected by the existing
policy and is not implemented here.

Template list operands use `/template/tolist` and `/template/append/tolist`
with the selected entity ID projection in an upload-only `path` field. The
shared ListManager still clones/project-selects operands and uses its configured
opener; repeated editable values and selected paths survive both operations.
This repairs the original inherited Query endpoints, which expected query XML
but received Template parameters. Query's ordinary append `path=None` behavior
and public `to_query()` identity remain unchanged. Set operations validate all
Template forms before uploading any operand. The server contract is
[TemplateToListService](https://github.com/intermine/intermine/blob/77cf7068dad0beac153e93e9916997d0ea850372/intermine/webapp/src/main/java/org/intermine/webservice/server/template/TemplateToListService.java#L67).

The checked server does not support active editable subclass, loop, range or
ISA forms. Subclass lacks the required operator, the populator excludes
loop/subclass/ISA, and range operators cannot use the MultiValue substitution
branch. Execution rejects these configurations before HTTP with guidance to
set `editable=False` to retain the saved server constraint, or switch off an
optional constraint. Constructibility and XML preservation do not imply remote
editability. Active editable empty multi-value collections also fail before
HTTP: repeated encoding would omit `valueN`, and the server requires at least
one value. A scalar empty string or collection containing an empty string
remains valid client input. These are deliberate fail-fast departures from
broken upstream requests, supported by
[Templates parameter parsing](https://github.com/intermine/intermine/blob/77cf7068dad0beac153e93e9916997d0ea850372/intermine/webapp/src/main/java/org/intermine/web/logic/template/Templates.java#L141),
[MultiValue operators](https://github.com/intermine/intermine/blob/77cf7068dad0beac153e93e9916997d0ea850372/intermine/pathquery/src/main/java/org/intermine/pathquery/PathConstraintMultiValue.java#L28),
and [TemplatePopulator](https://github.com/intermine/intermine/blob/77cf7068dad0beac153e93e9916997d0ea850372/intermine/api/src/main/java/org/intermine/api/template/TemplatePopulator.java#L277).

Executed offline evidence is in `tests/test_template_execution.py`, including
actual managed form requests, same-path codes, repeated values, switched-off
omission, stateful list/set/IN operations, typed exports, resource ownership and
ordinary Query regression suites. These fixtures do not assert live-server
integration.

Task 7.3 restores `Service.templates`, `all_templates`, `all_templates_names`,
`get_template(name)` and `get_template_by_user(name, username)`. Global discovery
GETs `/templates`; user discovery GETs `/alltemplates`. Discovery stores each
`<template>` as XML text without fetching the model or parsing queries. Getters
parse only the requested entry and bind it to the same Service, model, opener
and native/legacy profile. Repeated getters return the cached object. User
objects retain their actual `user_name` and are cached at `[username][name]`,
repairing the original top-level `[name]` assignment. Identical names belonging
to different owners remain distinct objects.

User name lists and object dictionaries share one raw XML snapshot, avoiding
the upstream client's separate GET for each property. Name lists derive from
that snapshot even if callers mutate the public parsed dictionary. Global and
user snapshots remain separate because their endpoints expose different scopes.
`flush()` clears raw, name and parsed caches only after internal temporary-list
cleanup succeeds; cleanup failure preserves those objects for retry. Flush
retains the opener, authentication and borrowed session lifetime.

Unknown names and users retain the original quoted `ServiceError` messages.
Global duplicates raise `ServiceError('Two templates with same name: NAME')`.
Duplicate names within one user now raise the same error rather than silently
overwriting the earlier XML; this is an intentional repair. Malformed discovery
XML retains the XML parser error; malformed query XML fails only on getter
access and leaves its raw entry intact. HTTP, XML, read and interrupt failures
close owned responses. There is no added version gate: the pinned getters do
not impose one. Authorization and endpoint availability remain server decisions.

The pinned discovery source is
[Service template getters and properties](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/webservice.py#L464).
Executed evidence is in `tests/test_template_discovery.py` and the real-cache
cleanup/retry test in `tests/test_lists_lifecycle.py`. They use the actual managed
opener, exact XML wire responses and request counts without live-server calls.
