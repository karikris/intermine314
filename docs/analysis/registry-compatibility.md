# Registry helper compatibility

Task 8.1 restores `intermine314.registry.getVersion`, `getInfo`, `getData` and
`getMines`, alongside the existing legacy `intermine314.webservice.Registry`
factory and its seven owned mapping methods. Comparison source is client 1.13.0
at `d888b779c8050bad789e26b312f40d220bc85d0d`, adapted under LICENSE-BSD and NOTICE:
[helpers](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/registry.py)
and [Registry](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/webservice.py#L52).

Historical positional calls remain valid. Each helper additionally accepts
keyword-only `registry=None` and `**registry_options`. Without an injected client,
options are forwarded to the managed legacy Registry constructor, including
`registry_url`, session, timeout, user agent, TLS, proxy/Tor settings and cache
size. The native configured HTTPS registry default intentionally replaces the
source's hard-coded HTTP URL. Construction eagerly loads the registry through
its existing transport. A supplied Registry is borrowed; supplying constructor
options together with it raises TypeError before another request. Helper imports
remain lazy and require neither analytics packages nor original `intermine`.

`getVersion(mine)` GETs the detail URL formed by appending the unchanged mine
argument to the configured instances endpoint. The name's case and source path
concatenation remain unchanged. It returns exactly `API Version:`,
`Release Version:` and `InterMine Version:` keys with the server values.
`getInfo(mine)` GETs that endpoint, prints Description, URL, API Version,
Release Version, InterMine Version, Organisms and Neighbours in source order,
then returns None. Its heading lines retain the trailing space. Missing fields
can leave partial output, as upstream. Missing instance/field KeyErrors in these
helpers and `getData` return `No such mine available`; decoding, HTTP and other
errors propagate through the managed transport.

`getData(mine)` uses the detail's `url`, normalized by the shared Service. Its
locally created Service borrows the Registry session and inherits its profile,
TLS, proxy/Tor policy, timeout and user agent. The query selects exactly
`DataSet.name DataSet.url`; explicit selection avoids the restored factory's
class wildcard expansion. ResultRows provide short-name access in either profile.
A missing row name prints `No info available` during consumption. Dataset names
are sorted, retaining duplicates, and printed as `Name: NAME`. Success returns
None. The result stream and local Service close even if iteration or printing
fails or is interrupted. Borrowed Registry cached Services remain untouched.

`getMines(organism=None)` prints all Registry mine names in ingestion order.
Filtering compares every organism entry to the argument or exactly one leading
space plus the argument. It preserves case, trailing whitespace and multiple
matching entries, so a mine may print more than once. It returns None after any
match and `No such mine available` after none. It deliberately does not call the
native whitespace-stripping/deduplicating organism filter. Native Registry
loading, keyed mine discovery and cache semantics remain intact.

The legacy Registry remains a facade over native Registry with legacy Service
factories, shared transport and bounded service cache. Membership and lookup
are case-insensitive; iteration, keys and length expose ingested names. Unknown
lookup raises `KeyError('Unknown mine: NAME')`; setting/deleting raises the
historical NotImplementedError messages. Both `instances` and original `mines`
with `webServiceRoot` payloads are supported. Owned clients and sessions close;
borrowed sessions remain open. Registry construction now guards its initial
request, decoding, ingestion and cache setup: any BaseException closes its owned
transport before propagating, even when callers retain the exception traceback.
Cleanup failures preserve the primary construction error. Failed clients close
idempotently, and failed constructors never close borrowed sessions.

Evidence: `tests/test_registry_helpers.py`, existing
`tests/test_compatibility_profiles.py`, minimal public surface/contract assertions,
and `behavior-coverage.json`. Strict offline JSON/XML fixtures verify outputs,
query payloads, URL handling and ownership. No live server parity is claimed.
