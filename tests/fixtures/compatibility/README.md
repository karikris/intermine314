These fixtures are offline test inputs, not a claim that the legacy API has been
restored. `docs/analysis/behavior-coverage.json` separates passing behavioral
evidence from pending restoration. Symbol availability remains recorded separately
in `docs/analysis/intermine-api-inventory.parquet`.

Unmodified upstream fixtures come from InterMine Python client **1.13.0**, commit
`d888b779c8050bad789e26b312f40d220bc85d0d`:
https://github.com/intermine/intermine-ws-python/tree/d888b779c8050bad789e26b312f40d220bc85d0d/tests/testservice

| Local file | Upstream path under tests/testservice |
| --- | --- |
| model.xml | service/model |
| rows-modern.json | service/query/results |
| rows-legacy.json | legacyjsonrows/query/results |
| objects-nested.json | testresultobjs/service/query/results |
| templates.xml | legacyjsonrows/templates/xml |
| lists.json | legacyjsonrows/lists/json |
| version-modern.txt | service/version/ws |
| version-legacy.txt | legacyjsonrows/version/ws |
| version-release.txt | service/version/release |

The upstream model provides multiple inheritance, interfaces, attributes, references
and collections, primitive and boxed numerics, Boolean, String, BigDecimal and Date.
The upstream row fixtures deliberately mix types within columns; they exercise wire
decoding, not a schema-correct analytics table. Nested objects include subclasses,
references, collections, Unicode and escaping.

`widgets.json`, `job-*.json` and `registry.json` are synthetic examples of upstream
response shapes. They are scaffolding, not recorded live responses or evidence of
complete behavior. Shapes were informed by upstream `intermine/webservice.py`
(`Service.widgets`), `intermine/idresolution.py` (`Job.fetch_status`,
`Job.fetch_results`), and `intermine/registry.py` (`getVersion`, `getInfo`). Future
implementation tests must refine these fixtures when they establish exact contracts.

`FixtureOpener` exposes urllib-style streams. `FixtureSession` exposes a requests-style
transport. Routes use `(HTTP method, URL path)` keys and return fresh streams on each
call. Unknown routes raise, requests are captured, and response close calls are
tracked. Override routes for HTTP errors or alternate bodies; neither adapter performs
HTTP or mutates server state. Inject the session into the real native Service or
InterMineURLOpener to test their parsing and resource ownership. Borrowed sessions
remain the caller's responsibility.

Pytest exposes `model_xml`, `protocol_payloads`, `offline_session_factory`,
`native_service_factory`, `legacy_service_factory`, `csv_input_factory`, and
`typed_parquet`. The legacy factory imports the future facade only on invocation.
CSV is a fresh StringIO input per factory call; no CSV file is written. Parquet is
created in pytest's temporary directory only when requested, using a lazy Polars
import. The Parquet data preserves original order, leading-zero String identifiers,
nulls, Unicode, Boolean, microsecond datetime, Decimal and integers above 2**53.
Install the analytics extra before requesting that fixture. Phase 1 collection and
native behavior tests do not need Polars or DuckDB.
