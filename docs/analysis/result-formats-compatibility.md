# Flat result compatibility

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
| `tsv` | `tab` | Stripped text line |
| `csv` | `csv` | Stripped text line |
| `count` | `count` | Stripped text line, without integer coercion |

Legacy `Query.rows(start=0, size=None, row=None)` chooses `rr` when the third
argument is omitted; native queries choose `dict`. An explicit third argument
or `row=` selects any restored format. `results()` and Query iteration retain
their current dictionary defaults until task 5.2 restores legacy object results.
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
JSON status/error buffer caps remain enforced. Nested `ResultObject` parsing,
object aliases/defaults, eager helpers and summaries remain tasks 5.2–5.4.
