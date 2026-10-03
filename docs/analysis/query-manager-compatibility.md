# Saved-query helper compatibility

Task 8.2 restores the five `intermine314.query_manager` helpers from InterMine
Python client 1.13.0 at `d888b779c8050bad789e26b312f40d220bc85d0d`, adapted under
LICENSE-BSD and NOTICE. Comparison source:
[query_manager.py](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/query_manager.py).

The original positional calls remain valid: `save_mine_and_token(m, t)`,
`get_all_query_names()`, `get_query(name)`, `delete_query(name)` and
`post_query(value)`. XML parsing uses the standard library, and imports perform
no HTTP or analytics imports. The source's mock-only test branches are omitted;
strict offline account routes exercise real managed transport.

`save_mine_and_token` accepts keyword-only `registry`, `service`, `opener` and
Registry constructor options. An owned Registry uses configured HTTPS defaults,
TLS verification, timeout, proxy/Tor policy and user agent. It resolves the
case-preserving detail path through the same `_instance` helper used in task 8.1.
The local Service inherits those settings and borrows the Registry session; its
existing constructor also validates `/version/ws`. This Service is deliberately
unauthenticated: account reads/mutations use the helper snapshot token explicitly
in URLs. With a supplied opener the snapshot may have no Service. Later plotting
work must choose account authentication explicitly for private List APIs rather
than treating this local Service as authenticated. A supplied Registry is
borrowed, including any cached Services. A supplied Service must match the
resolved service root. Its opener is cloned without account/basic authentication;
the original opener and session remain unchanged. A supplied account opener is
borrowed, or cloned without authentication when it supports `clone()`.
Authenticated custom openers lacking `clone()` are rejected. Registry options
cannot accompany an injected Registry, and Service/opener injections cannot be
combined; these conflicts raise TypeError before I/O and leave existing state
unchanged. Registry and account transport are separate: only account URLs carry
the encoded supplied token. The resolved root follows Registry/opener Tor HTTPS
policy even when an account opener is injected; the Registry's explicit
`allow_http_over_tor=True` retains its configured opt-in.

Each helper module owns an independent `_HelperState` snapshot; the class can be
reused by plotting helpers without sharing module credentials. Successful save
returns None and exposes historical `mine`/`token` globals. It validates the
account's `queries` mapping before publishing the active state. Later calls reuse
the resolved root and validated transport snapshot; manually changing the globals
does not change that snapshot. Reconfiguration closes old owned Registry/local
Service clients before resolving the replacement. Injected clients/sessions remain
open. Failure retains the attempted globals but clears active state, closing any
new owned clients immediately. Helpers then raise RuntimeError directing callers
to `save_mine_and_token`, preventing use of an unvalidated attempted configuration.
Upstream also updates its globals before validation, but permits later helper
calls using that attempted configuration. Successful state
owns its local clients until reconfiguration (or ordinary managed-session process
cleanup). This deliberately repairs the source's unmanaged lifetime and accidental
failed-state use.

Configuration exceptions return exactly
`An exception of type TYPE occurred. Check mine` during registry/detail/Service
setup, or `An exception of type TYPE occurred. Check token` during account
validation. Exception text and credentials are not printed or logged. Interrupts
propagate after cleanup. Operational errors propagate while keeping the validated
state available for a subsequent call.

`get_all_query_names` joins JSON keys in server order with `, ` and returns
`No saved queries` for an empty mapping. `get_query` requests `filter`, `format=xml`
and the token, returning raw server text. The historical exact
`<saved-queries></saved-queries>` sentinel returns `No such query available`.
The implementation also recognizes equivalent empty XML with a declaration,
self-closing root or whitespace: this is an executed, intentional sentinel repair.
Empty/non-XML raw text remains raw text. `delete_query` checks names first, returns
`No such query available` when missing, and DELETEs an existing query before
returning `NAME is deleted`. A full path segment is encoded once, including
slashes, plus, ampersand and Unicode characters.

`post_query` parses the XML name, GETs source `/version?token=...` as a JSON
integer, and checks saved names. Version 27 or greater uses `query`; older versions
use `xml`. The source-compatible PUT sends URL-encoded XML and token as URL
parameters, then GETs the same parameterized URL to verify the saved name. Both
mutation and readback responses close. A duplicate prints `The query name exists`
and prompts exactly `Do you want to replace the old query? [y/n]`. Only lowercase
`y` replaces it; other answers print `Use a query name other than NAME` and return
None. Keyword-only `overwrite=True` and `overwrite=False` bypass the prompt;
False prints the decline message and returns None. Values other than None/True/False
raise ValueError before HTTP. New names never prompt. Success returns `NAME is
posted`; a missing name after PUT prints the original note and returns
`Incorrect format`. The note's literal has exactly twelve spaces between `symbol`
and `and`, verified from the pinned source's AST. Malformed XML or a missing name
raises before account I/O.

All responses close on success, decode/JSON/XML/HTTP failures and read
interruptions. PUT and DELETE discard their response bodies with deterministic
closure. Token/filter/XML encoding repairs the source's raw URL concatenation,
which can change credential and query values. No bare requests or urllib network
calls, lxml dependency, credential output or data artifact writes are introduced.

Evidence: `tests/test_query_manager_helpers.py`, public surface/contract tests,
owned `saved-queries.json` and registry wire fixtures, and
`docs/analysis/behavior-coverage.json`. Tests execute both native and legacy
injected client profiles and assert actual captured URLs/headers and resource
lifetimes. No live-server integration parity is claimed.
