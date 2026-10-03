# Template constraint foundation

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

Template wrapper XML, metadata cloning, template result parameters/endpoints,
adjusted execution and exports remain task 7.2. Discovery/caching remains task
7.3. At this stage inherited Query serialization/execution uses ordinary query
XML/endpoints and does not apply template switch state on the server. It does
not constitute named-template execution. Ordinary Query imports continue to
ignore template XML flags; full Template flag round trips follow in task 7.2.
