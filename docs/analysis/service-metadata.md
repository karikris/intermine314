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
`None`, ready for full cache invalidation in task 6.2; `flush` remains pending.
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
