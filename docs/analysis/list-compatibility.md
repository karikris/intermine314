# Server list CRUD, append, tags and cleanup

Task 6.1 restores the lazy `intermine314.lists` facade, `List`, `ListManager`,
`ListServiceError` and `safe_key`, plus `Service.list_manager()`. The callable
returns a new manager with its own discovery cache and temporary-name tracking.
The service's internal manager is allocated only when a later delegate needs it.
Task 6.2 adds ordinary/CSV append, tag methods, context cleanup and `Service.flush`.
All requests use the configured service opener, authentication, TLS, timeouts,
user agent and shared session. Ordinary list imports, discovery and text uploads
do not import Polars, DuckDB, PyArrow or pandas.

List metadata preserves the upstream title, description, type, creation date,
authorization, upgrade status and immutable tags. Size/count/len are equivalent.
Discovery is cached and returns dictionary keys/values views; a missing name
returns `None`. Refresh publishes the cache only after a successful response.
Rename uses the original GET `oldname`/`newname` parameters, updates the cache and
the existing object, and retires the old temporary name. A same-name assignment
does not request the server. Deleting a name raises the original AttributeError.
Deletion accepts names or List instances, skips missing lists and refreshes before
and after the DELETE operations. Failed HTTP or unsuccessful list responses
propagate through the public errors; upload failures do not trigger another
content interpretation or request.

Text uploads retain the original `/lists` endpoint, plain UTF-8 body, name/type/
description and semicolon-separated tags, plus repeated lowercased `add` values.
String input is read as a UTF-8 path when available, otherwise stripped and sent
as identifier text. Path objects read directly. File/readable content is sent
without stripping and borrowed streams remain open. Iterable values, including
generators, become one double-quoted token per line. Empty input prints the two
upstream guidance lines and returns `None` before requesting or reserving a name.
Anonymous names start at `my_list_1`; already reserved names within the same
manager are also avoided.

`List.__iter__`, indexing and `display` use a real private query constructed from
the list type with an explicit named `ListConstraint`. The normal query factory
expands the model's attributes. Iteration retains native dictionary and legacy
object defaults. Integer indexing supports negative offsets and fetches one
object using the existing `first(start=..., row="jsonobjects")` path; invalid
indices fail before HTTP. Display prints selected fields and closes the stream on
completion or an output error. It reads actual fields rather than splitting a
string representation, so commas, parentheses and Unicode values remain intact.

Optional CSV input uses:

```python
manager = service.list_manager()
created = manager.create_list(
    csv_input=identifier_stream,
    csv_column="identifier",
    list_type="Employee",
    name="reviewed identifiers",
)
```

CSV requires an explicit identifier column and list type, and cannot be combined
with ordinary content or an organism/server filter. CSV options without CSV input
are rejected. `csv_options` accepts validated Polars scan options; schema and
schema_overrides must use dictionaries. The selected identifier column is always
forced to `Polars.String`, even if a caller supplies a numeric schema. The caller's
options are copied. An ID such as `0007` therefore stays `0007`; it is never
inferred as a number and then cast back to text.

The CSV path is Polars scan → temporary Parquet → DuckDB SQL → managed Arrow
reader → identifier upload to the same text endpoint. No CSV temporary/output or
pandas step is used. Parquet scratch paths remain literal even with glob
characters. Scratch directories, the reader and the connection are cleaned up
after success, malformed IDs, Parquet/SQL/reader failures and HTTP/protocol errors.
Borrowed text/binary/nonseekable CSV streams remain open.

The CSV identifier policy rejects null and empty identifiers before upload; it
does not drop them, convert them to text or trim whitespace. Null markers chosen
through `csv_options` follow this same rule. A zero-row create CSV follows the
ordinary create empty-input print/None behavior. Unicode, leading zeros, spaces, tabs and commas
are preserved in quoted tokens; literal double quotes are doubled. CR or LF
inside an identifier is rejected because the server reads separate lines before
tokenizing. The server's tokenizer ignores empty tokens, so an empty CSV String
also cannot represent a stored identifier. These rules avoid silent corruption.

The tokenizer rules come from the pinned server's
[ListUploadService](https://github.com/intermine/intermine/blob/77cf7068dad0beac153e93e9916997d0ea850372/intermine/webapp/src/main/java/org/intermine/webservice/server/lists/ListUploadService.java#L213)
and its Apache Commons Lang 2.6
[StrTokenizer](https://github.com/apache/commons-lang/blob/LANG_2_6/src/main/java/org/apache/commons/lang/text/StrTokenizer.java#L747).

Deliberate repairs to Python client 1.13.0 include consuming generators/readable
inputs without calling `len` on them, closing owned streams on failure, safe JSON
error handling for both bytes and text, retaining a valid cache on a failed
refresh, avoiding repeated local temporary-name reservations, retiring renamed
temporary names, and displaying actual row/object fields. Iterable identifiers
now escape literal quotes and reject embedded line breaks; raw text upload
semantics remain the original linewise format. Imports do not configure global
logging, and list code does not log payloads or credentials. Immutable tuple
defaults replace unused mutable list defaults for tags and add.

`item.append(appendix=identifiers)` sends plain UTF-8 text to `/lists/append?name=...`
and returns the same List after updating size and accumulating unmatched identifiers.
It accepts paths, borrowed readable streams, strings and iterables through the
shared serializer. Unlike create_list, append preserves leading/trailing whitespace
in raw strings. Empty ordinary input sends an empty append body and returns the
List; it does not print create guidance. Optional `csv_input`, `csv_column` and
`csv_options` are keyword-only and share the same String/Parquet/DuckDB/Arrow path
and null/empty/line-break policy as creation. CSV-only calls omit `appendix`; the
existing List supplies the class, so no `list_type` parameter is needed. Combining
appendix with CSV or providing CSV options without input fails before HTTP. CSV
header-only input likewise sends an empty append body. Failed uploads and
interruptions propagate without trying a union or a second input interpretation.

Manager `add_tags(list, tags)`, `remove_tags(list, tags)` and `get_tags(list)` use
POST, DELETE and GET `/list/tags` respectively. POST uses a form; DELETE and GET
use query parameters. Names and semicolon-separated tags retain Unicode and
reserved characters. Managers return the server's tag list. List methods
`add_tags(*tags)` and `remove_tags(*tags)` store a frozenset and return None.
`update_tags(*tags)` **refreshes** via get_tags, ignoring optional arguments as
the upstream implementation does, despite its misleading removal docstring.
Failures leave previously cached tags unchanged and close the owned response.

`with service.list_manager() as manager:` returns that manager and cleans only
its unnamed temporary lists on exit. Named lists and explicitly renamed lists
are retained. `delete_temporary_lists()` snapshots its tracked names, retires
successful deletes individually, and retains failed names for a later retry.
Missing names are retired after successful cleanup. Empty tracking makes no HTTP
request. An ordinary cleanup failure propagates when there is no body exception.
When a body exception exists (including KeyboardInterrupt), it stays primary and
gets an observable note describing the ordinary cleanup failure. Cleanup
KeyboardInterrupt/SystemExit takes priority and propagates with the body exception
as context, so interrupts are never suppressed.

`Service.flush()` first cleans its allocated internal manager's temporary lists,
then invalidates model/XML/model-name/query-model/version/release/widget caches
and `_templates`, `_all_templates`, `_all_templates_names` for later template
support. It resets the internal manager lazily, preserving the configured opener
and session. If no manager was allocated, it makes no list request. External
managers remain independent and caller-owned. Failed cleanup retains the internal
manager, outstanding names and metadata caches for retry. It does not close the
Service. Account deregistration still calls flush before its user DELETE.

Public `List.to_query`, query/List uploads, organism
query uploads, set operations and the seven Service delegates remain task 6.3.
Query/List inputs, including iterable members, or organism uploads currently raise
an explicit staged error before requesting results. Enrichment remains task 6.4.

Executed evidence is in `tests/test_lists_crud.py`, `tests/test_lists_lifecycle.py`,
the lazy-facade tests and
`docs/analysis/behavior-coverage.json`; name availability alone is not a claim of
complete list interoperability.
