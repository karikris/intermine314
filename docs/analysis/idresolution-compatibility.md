# Identifier-resolution compatibility

Task 7.4 restores `Service.resolve_ids` and the lightweight
`intermine314.idresolution` module (`Job`, `get_json`). Native and legacy service
profiles share the implementation and configured managed opener. The comparison
source is Python client 1.13.0 at `d888b779c8050bad789e26b312f40d220bc85d0d`,
adapted under the shipped BSD-2-Clause license and NOTICE:
[submission](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/webservice.py#L591)
and [job lifecycle](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/idresolution.py).

`resolve_ids(data_type, identifiers, extra='', case_sensitive=False,
wildcards=False)` requires API version 10+, then truthy data type and identifiers
in that order. The exact errors are `ServiceError('This feature requires API
version 10+')`, `ServiceError('No data-type supplied')` and
`ServiceError('No identifiers supplied')`. Identifiers become a list, including
generator input. As upstream, truthiness is checked before consuming an iterable;
an empty generator therefore submits an empty list. POST `/ids` sends JSON keys
`type`, `identifiers`, `extra`, `caseSensitive` and `wildCards`, with content type
`application/json; charset=utf-8`, then returns `Job(service, uid)`.

Jobs hold a weak service proxy, their UID, cached `status=None`, initial
`backoff=0.05`, `decay=1.25` and `max_backoff=60`. Each nonterminal `poll()` advances
the next backoff, sleeps the current interval, fetches status and returns whether
it is `SUCCESS` or `ERROR`. Subsequent terminal polls return true without HTTP or
sleep. The delay is capped; there is no attempt limit or implicit polling loop.
Failed fetches retain the previous status and the advanced backoff, as upstream.
The caller must retain the service while using a job.

`fetch_status()` GETs `/ids/UID/status` and returns `status` without updating the
cached poll state. `fetch_results()` GETs `/ids/UID/result` and returns `results`
without requiring completion. `delete()` DELETEs `/ids/UID`, checks the JSON
`error` and returns None without changing the job's local attributes.
`get_json(service, path, key)` raises `Exception(error)` for every non-None server
error, including false and empty-string values. Missing requested keys raise
`Exception(key + ' not returned from ' + path)`; absent `error` retains `KeyError`.
Submission instead raises `ServiceError(error)`. Invalid JSON, invalid UTF-8 and
missing `uid` preserve their parsing/decoding/KeyError exceptions.

The null-UID submission branch intentionally raises `Exception('No uid found')`
through `Job`, repairing upstream's accidental string-plus-dictionary TypeError
while preserving direct `Job(service, None)` behavior. No stricter UID rules are
invented: empty strings remain accepted and deleting a numeric UID retains the
source TypeError. Transport read helpers close owned responses on success,
parsing failures, read failures and BaseException interruption. Shared HTTP error
dispatch now closes its response even if reading the error body fails or is
interrupted, retaining the primary exception if closing also fails; the HTTP
exception contract is unchanged. Borrowed sessions remain
open. Authentication, timeout, TLS verification, proxy/Tor policy and user agent
come from the existing opener.

Evidence is in `tests/test_idresolution.py`, `tests/test_minimal_surface.py` and
`behavior-coverage.json`. Tests use strict offline routes and fake sleeps, exercise
100 pending polls without a limit, and never wait in real time. No live server
integration is claimed. Imports and execution require neither the query builder,
analytics extras nor the original `intermine` package. Job operations create no
output files.
