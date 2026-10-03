# Result compatibility

Task 5.1 restores `ResultRow`, `TableResultRow`, `FlatFileIterator` and flat/mapping
format dispatch through the lazy `intermine314.results` facade and shared
`service.session` implementation. Executed evidence is in
`tests/test_result_formats.py` and `behavior-coverage.json`.

The comparison source is original client 1.13.0, pinned at
`d888b779c8050bad789e26b312f40d220bc85d0d`, acquired through GitHits. Adapted row
and parser behavior follows [results.py](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/results.py#L182)
under the BSD-2-Clause option reproduced in `LICENSE-BSD` and `NOTICE`.

`ResultRow` supports integer/negative indexes, slices, full selected paths,
root-stripped paths, and callable lookup. Removing the root preserves nested
paths: `Gene.organism.name` is also `organism.name`, not `name`. Iteration yields
values. `keys`, `values`, `items`, `to_l`, and `to_d` return fresh containers;
iterator variants and `has_key` retain their original behavior. Input data and
view containers remain borrowed, as in upstream; returned copies are shallow.
`TableResultRow` exposes each cell's `value` while retaining JSON metadata in
`data`.

| Requested format | Wire format | Yielded row |
| --- | --- | --- |
| `rr` | `json` for version >= 8, otherwise `jsonrows` | `ResultRow` or `TableResultRow` |
| `list` | Same version dispatch | Value list |
| `dict` | Same version dispatch | Full-path dictionary |
| `json`, `jsonrows` | Requested format | Unmodified decoded JSON row |
| `object`, `objects`, `object*`, `jsonobjects` | `jsonobjects` | Model-backed `ResultObject` |
| `tsv` | `tab` | Stripped text line |
| `csv` | `csv` | Stripped text line |
| `count` | `count` | Stripped text line, without integer coercion |

Legacy `Query.rows(start=0, size=None, row=None)` chooses `rr` when the third
argument is omitted; native queries choose `dict`. An explicit third argument
or `row=` selects any restored format. Legacy `results()` and Query iteration
now return objects; native results, rows and iteration retain dictionaries.
Direct `Query(Model)` follows the legacy profile unless explicitly overridden.
`iter_rows`/`iter_batches` retain dictionary modes for analytical exports, which
continue selecting `dict` explicitly. Schema-aware dictionary decoding retains
exact Decimal values on designated paths and ordinary floats elsewhere. CSV
streams require an explicit format request and create no files.

Every `iter(ResultIterator)` opens an independent response. Returned streams
own that response, can outlive a temporary facade, and close on exhaustion,
parser failure, interruption, or explicit `close()`, including before the first
row. Facade `close()` closes all its active streams, including the stream used
by `next(facade)`; subsequent iteration can make a fresh request. Borrowed
sessions remain open. The public `len(facade)` still counts a fresh HTTP result
stream, so callers seeking one request should materialize with a comprehension
rather than relying on `list(facade)` length hints.

These intentional repairs depart from the original implementation:

- Empty rows stringify safely as `ResultRow:` or `TableResultRow:`.
- Decoded JSON `null` remains a yielded row rather than ending iteration.
- Header, row and footer failures and `BaseException` interruptions close owned
  responses, including direct public parser calls and streams closed before
  their first row.
- Flat streams close on exhaustion, server `[ERROR]` lines, and parser failures.

The shared configured opener preserves authentication, TLS/CA settings,
Tor/proxy policy, user agent and timeouts. Its bounded POST-to-GET fallback and
JSON status/error buffer caps remain enforced. Summaries remain task 5.4.

Task 5.2 restores `ResultObject(data, cld, view=())` using the actual query model
Class while QuerySpec roots remain strings. Source behavior is adapted from
[results.py lines 71–179](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/results.py#L71).
`id` exposes `objectId`; `type` and the descriptor use the returned most-specific
class. Composed classes preserve their original wire `type` string while exposing
all component fields; lazy fetches use the field's declaring schema class.
References become objects, collections become lists, and repeated field
access returns the cached value. Unknown fields raise `ModelError`. `str` shows
loaded scalars; `repr` includes loaded relations without making requests.

Lazy fields execute a query using the object's exact model and executing
Service, even when the query model differs from `Service.model`. The internal
first-object helper closes its managed stream after consuming one object; it
omits a `size` limit because the server counts joined rows, which could truncate
collections. Unbound objects return unavailable fields as `None` or `[]`.
Service requests with object formats normalize a supplied Class, root string,
or first selected view to a descriptor; missing schema produces a public
`ModelError`. Empty views expand with existing prefetch settings on a clone.
Object constraint paths augment the same clone whose views go on the wire.

Additional repairs from the original object implementation are covered by
`tests/test_result_objects.py`:

- Absent `class` metadata falls back to the supplied descriptor.
- Noncontiguous selections retain every nested reference path.
- Selected missing and loaded null fields cache `None`/`[]` without refetching.
- Missing object IDs never produce a query with `id=None`; classes without an
  ID also cannot fetch additional fields.
- Constraint augmentation compares whole path segments, so `companyCode` does
  not count as selecting the `company` relation.
- Nested wrappers share an identity map for input mappings. Cycles reuse the
  existing wrapper and render `...`, without recursion or lazy requests.

Input mappings remain borrowed and caches are per returned object graph; this
is not a global identity cache keyed by database ID. Exhaustion, early close,
parser errors and lazy-fetch errors retain the shared response ownership rules.
Tests also verify explicit dictionary analytical iteration and exact Decimal
Parquet export under both profiles.

Task 5.3 restores `first(row="jsonobjects", start=0, **kw)`,
`one(row="jsonobjects")`, `get_results_list(*args, **kwargs)`,
`get_row_list(start=0, size=None)` and the `all` alias. The historical object
default for `first` and `one` applies to both profiles, including native queries;
native `results`, `rows` and iteration retain their dictionary defaults.
Source behavior follows [query.py lines 1510–1578](https://github.com/intermine/intermine-ws-python/blob/d888b779c8050bad789e26b312f40d220bc85d0d/intermine/query.py#L1510).
Executed evidence is in `tests/test_query_eager_results.py`.

`first` forwards the start and additional result options, returns `None` for an
empty stream, and closes after consuming one result. Objects omit `size` to
preserve complete joined collections; flat formats request `size=1`. All object
aliases use this policy, repairing upstream spelling-dependent truncation.
The internal lazy-fetch helper shares this public implementation. No original
query views, model, profile or pagination state are changed.

`one` requests the server count first. A count of one delegates to `first`,
matching upstream; otherwise object formats inspect at most two top-level
objects because a joined-row count may exceed one for a single object. Zero or
multiple objects raise `QueryError` with message `No results received` or
`More than one result received`. Flat formats require a count of exactly one,
otherwise raising `QueryError` with `Result size is not one: got N results`.
These are exception message values; `ReadableException` retains its quoted
string representation. Scans close on success, cardinality errors, malformed
rows, server footer errors and interruptions.

`get_results_list` and its identical `all` alias forward all result arguments
and materialize with a comprehension, avoiding the extra HTTP request made by
upstream's `list(ResultIterator)` length hint. Empty results return `[]`; streams
close on exhaustion or failure. `get_row_list` forwards start and size using
legacy `rr` or native `dict`. The native dictionary choice intentionally departs
from upstream's unconditional `rr` to follow native `rows` policy. Wire tests
verify this under server versions 7 and 8, plus the exact custom Query model and
executing Service binding through object cardinality helpers and lazy fetches.
Narrow custom-iterator tests additionally verify that an absent `close` method
is accepted and cleanup exceptions do not replace the primary parser error,
using the shared runtime cleanup helper.
