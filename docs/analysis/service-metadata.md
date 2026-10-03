# Service metadata

Task 4.3 restores `search`, `widgets`, `release` and `resolve_service_path`
alongside the existing `version` contract. Native and legacy Service classes
inherit the same methods and use the same configured managed opener. Behavioral
evidence is in `tests/test_service_metadata.py` and `behavior-coverage.json`.

The comparison source is Python client 1.13.0 at
`d888b779c8050bad789e26b312f40d220bc85d0d`, acquired through GitHits and adapted
under the shipped BSD-2-Clause license and NOTICE:
[metadata and JSON/XML helpers](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/webservice.py#L532),
[resolution and release](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/webservice.py#L392).

`search(term, **facets)` sends form-encoded `q` and `facet_NAME` parameters with
`doseq=True`, preserving Unicode, reserved characters and repeated facet values.
It returns `(results, facets)` from the service JSON response. The private
`_get_json(path, payload=None)` helper requests `application/json`. Every
non-`None` server `error` raises `ServiceError`, including empty-string or false
values. Invalid JSON and UTF-8 retain their parsing/decoding errors; missing
required response keys retain `KeyError`. Responses are closed in every case.
No empty successful result is substituted for a failed response.

`widgets` caches a dictionary keyed by each widget's name, retaining the entire
metadata object. `release` caches a UTF-8 decoded and stripped string. Empty
dictionaries and empty release strings are cached too. Failed fetches or parses
do not populate either cache. `_widgets` and `_release` are initialized to
`None`. The task 4.4 `flush` foundation clears implemented metadata/model/query
caches; temporary-list and template cleanup remain deferred to task 6.2.
`version` retains its eager constructor validation, cached integer and existing
`ServiceError` for invalid integer responses.

`resolve_service_path(variant)` fetches `/check/` plus the variant and returns
raw bytes. `_get_xml(path)` requests `application/xml` and returns a minidom DOM
document. These helpers preserve the existing opener's authentication, user
agent, timeout, TLS verification, Tor/proxy configuration and session identity.
Owned responses close on successful reads, HTTP errors, malformed data, read
errors and interruption; borrowed sessions remain open after Service closure.
Metadata methods do not load the query builder, analytics extras or the original
`intermine` package.

The intended upstream request and return contracts are preserved. Two upstream
resource defects are repaired: `release` uses the configured managed opener
instead of bare `urlopen`, and optional-service resolution closes its response.
There are no live mine or account requests in the evidence suite.

Account operations (task 4.4) use the same configured managed opener in both
profiles. `register(username, password)` requires webservice version 9, sends
an unauthenticated form POST to `/users`, and returns a Service authenticated
with `user.temporaryToken`. The registrar borrows the original opener's session
through an independent clone; closing it never closes the caller's session or
changes the original authentication. The returned Service preserves the caller's
profile, class, prefetch options, TLS, Tor/proxy policy, timeout and user agent.
If the caller owns its managed session, the returned Service owns a new session
and either client may close first. If the caller supplied a session, both clients
borrow it and closing either leaves it open. Failed returned-client version
validation closes its newly owned session and leaves the original session alive.

`get_deregistration_token(validity=300)` requires version 16, accepts validity
from 1 through 86400 seconds, sends a form POST to `/user/deregistration`, and
returns the response's `token`. `deregister(deregistration_token)` also requires
version 16 and accepts either a token string or a dictionary containing `uuid`.
It calls `flush()` before DELETE `/user?deregistrationToken=...&format=xml` and
returns raw user XML bytes. The current flush invalidates `_model`, `_model_xml`,
`_model_name`, `_query_model`, `_version`, `_release` and `_widgets`; later reads
rebuild these caches. This does not claim the deferred list/template lifecycle.

`get_anonymous_token(url)` reuses `_request_anonymous_token` and the configured
opener for GET `url + '/session'`. The native `token='random'` constructor keeps
its unauthenticated pre-auth request, token authentication and owned-session
finalizer transfer. Failure during anonymous-token fetching or later version
validation closes owned resources and never closes an externally borrowed
session. All owned account responses close on success, server/parse/read errors
and interruption. Account forms and credentials are not logged.

Intentional repairs to pinned 1.13.0 account behavior: registration encodes actual
Unicode values rather than Python 3 string representations of upstream
`bytearray` objects; validity errors state the real 1–86400-second bounds rather
than upstream's inaccurate “1ms - 2hrs”; string deletion tokens containing `uuid`
remain strings rather than being indexed as dictionaries. Account registration
and anonymous-token requests retain runtime transport configuration rather than
creating unconfigured helper clients or using standalone `requests.get`.

Executed offline evidence is in `tests/test_service_accounts.py`; it covers both
profiles, exact methods/paths/forms/headers/returns, version gates and validity
bounds, ownership and close ordering, malformed responses, HTTP/read/interruption
errors, random-token initialization failures and cache invalidation before DELETE.
