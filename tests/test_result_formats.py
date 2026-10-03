"""Flat/mapping compatibility through the shared managed result transport."""
from decimal import Decimal
from io import BytesIO
from types import SimpleNamespace
from urllib.parse import parse_qs

import pytest

from intermine314 import results
from intermine314.service import session as runtime
from intermine314.service.errors import WebserviceError
from intermine314.service.service import Service
from intermine314.webservice import Service as LegacyService
from tests.fixtures.compatibility import FixtureSession, fixture_bytes

ROOT = "https://offline.example/service"
VIEWS = ["Employee.name", "Employee.age", "Employee.fullTime"]


def make_iterator(mode="dict", *, version=8, payload=None, decimal_paths=()):
    if payload is None:
        payload = fixture_bytes("rows-modern.json" if version >= 8 else "rows-legacy.json")
    session = FixtureSession({("POST", "/service/query/results"): payload})
    opener = runtime.InterMineURLOpener(session=session, timeout=(2, 7), verify_tls="ca.pem", user_agent="row-test")
    service = SimpleNamespace(root=ROOT, version=version, opener=opener)
    iterator = runtime.ResultIterator(service, "/query/results", {"query": "xml"}, mode, VIEWS, decimal_paths=decimal_paths)
    return iterator, session, opener


def test_result_row_mapping_value_iteration_and_copies():
    data, views = ["abc", None, 42], ["Gene.symbol", "Gene.organism.name", "Gene.length"]
    row = results.ResultRow(data, views)
    assert len(row) == 3
    assert row[0] == row[-3] == row["Gene.symbol"] == row("symbol") == "abc"
    assert row("organism.name") is None
    assert row[1:] == [None, 42]
    assert row[::-1] == [42, None, "abc"]
    assert list(row) == row.values() == row.to_l() == data
    assert row.keys() == list(row.iterkeys()) == views
    assert row.items() == list(row.iteritems()) == list(zip(views, data))
    assert list(row.itervalues()) == data
    assert row.to_d() == dict(zip(views, data))
    assert row.has_key("Gene.symbol") and row.has_key("symbol")
    assert not row.has_key("name")
    with pytest.raises(KeyError):
        row["missing"]
    with pytest.raises(IndexError):
        row[99]
    for copy in (row.keys(), row.to_l(), row.values(), row[:]):
        copy.clear()
    copied = row.to_d()
    copied.clear()
    assert row.to_l() == data and row.keys() == views
    assert str(row) == "Gene: symbol='abc' organism.name=None length=42"
    assert str(results.ResultRow([], [])) == "ResultRow:"


def test_table_result_row_preserves_cell_metadata_and_unwraps_all_access():
    cells = [{"value": "abc", "url": "/abc", "class": "Gene"}, {"value": None, "extra": [1]}]
    row = results.TableResultRow(cells, ["Gene.symbol", "Gene.organism.name"])
    assert row[0] == row[-2] == row("symbol") == "abc"
    assert row[1:] == [None]
    assert row.to_l() == row.values() == list(row) == ["abc", None]
    assert row.to_d() == {"Gene.symbol": "abc", "Gene.organism.name": None}
    assert row.data[0]["url"] == "/abc" and row.data[1]["extra"] == [1]
    assert str(results.TableResultRow([], [])) == "TableResultRow:"


@pytest.mark.parametrize("version", [7, 8])
@pytest.mark.parametrize("mode", ["rr", "list", "dict", "json", "jsonrows"])
def test_mapping_and_raw_json_dispatch(version, mode):
    iterator, session, opener = make_iterator(mode, version=version)
    rows = [row for row in iterator]
    assert len(rows) == 3
    if mode == "rr":
        assert type(rows[0]) is (results.ResultRow if version >= 8 else results.TableResultRow)
        assert rows[0].to_l() == ["foo", "bar", "baz"]
    elif mode == "dict":
        assert rows[0] == dict(zip(VIEWS, ["foo", "bar", "baz"]))
    elif mode == "list" or version >= 8:
        assert rows[0] == ["foo", "bar", "baz"]
    else:
        assert rows[0][0] == {"value": "foo", "url": "/some/path/foo"}
    wire = mode if mode in {"json", "jsonrows"} else ("json" if version >= 8 else "jsonrows")
    assert parse_qs(session.requests[-1].data.decode())["format"] == [wire]
    assert session.responses[-1].closed
    opener.close()
    assert session.close_calls == 0


@pytest.mark.parametrize("mode,wire,payload,expected", [
    ("tsv", "tab", b" a\tb \n1\t2\n", ["a\tb", "1\t2"]),
    ("csv", "csv", b'"a,b",c\n1,2\n', ['"a,b",c', '1,2']),
    ("count", "count", b" 42 \n", ["42"]),
])
def test_flat_streams_require_explicit_format(mode, wire, payload, expected):
    iterator, session, _ = make_iterator(mode, payload=payload)
    assert [row for row in iterator] == expected
    assert parse_qs(session.requests[-1].data.decode())["format"] == [wire]
    assert session.responses[-1].closed


@pytest.mark.parametrize("legacy", [False, True])
def test_query_rows_profile_default_and_historical_third_argument(legacy):
    session = FixtureSession.service()
    client = LegacyService if legacy else Service
    with client(ROOT, session=session) as service:
        query = service.select(*VIEWS)
        row = next(query.rows())
        assert isinstance(row, results.ResultRow if legacy else dict)
        stream = query.rows(2, 1, "list")
        assert next(stream) == ["foo", "bar", "baz"]
        stream.close()
        params = parse_qs(session.requests[-1].data.decode())
        assert params["start"] == ["2"] and params["size"] == ["1"]
        assert isinstance(next(query.results("dict")), dict)
    assert session.close_calls == 0


@pytest.mark.parametrize("mode", ["dict", "rr", "csv"])
@pytest.mark.parametrize("consumed", [False, True])
@pytest.mark.parametrize("close_outer", [False, True])
def test_explicit_close_releases_response_before_and_after_first_row(mode, consumed, close_outer):
    iterator, session, opener = make_iterator(mode, payload=b"a,b\nc,d\n" if mode == "csv" else None)
    stream = iter(iterator)
    if consumed:
        next(stream)
    (iterator if close_outer else stream).close()
    assert session.responses[-1].closed and session.responses[-1].raw.closed
    assert session.responses[-1].close_calls == 1
    with pytest.raises(StopIteration):
        next(stream)
    opener.close()
    assert session.close_calls == 0


def test_simultaneous_streams_next_and_reuse_have_independent_lifetimes():
    iterator, session, _ = make_iterator()
    first, second = iter(iterator), iter(iterator)
    assert next(first) == next(second) == next(iterator)
    first.close()
    assert session.responses[0].closed and not session.responses[1].closed
    iterator.close()
    assert all(response.closed for response in session.responses)
    assert [row for row in iterator][0][VIEWS[0]] == "foo"
    assert len(session.responses) == 4 and all(response.closed for response in session.responses)
    assert len(iterator) == 3  # Public count still makes a fresh request.
    assert len(session.responses) == 5 and session.responses[-1].closed
    assert iterator.next()[VIEWS[0]] == "foo"
    iterator.close()
    assert session.responses[-1].closed


class InterruptConnection(BytesIO):
    def __init__(self, payload, position):
        super().__init__(payload)
        self.position = position
        self.calls = 0

    def __next__(self):
        self.calls += 1
        if self.calls == self.position:
            raise KeyboardInterrupt("interrupted")
        return super().__next__()


@pytest.mark.parametrize("position", [1, 2, 4])
def test_json_header_row_footer_interruptions_close_owned_connection(position):
    connection = InterruptConnection(b'{"results":[\n[1]\n],\n"wasSuccessful":true}\n', position)
    with pytest.raises(KeyboardInterrupt):
        parser = results.JSONIterator(connection, lambda row: row)
        while True:
            next(parser)
    assert connection.closed


@pytest.mark.parametrize("mode", ["dict", "rr", "csv"])
def test_outer_stream_interrupt_closes_http_response(mode, monkeypatch):
    iterator, session, _ = make_iterator(mode, payload=b"one\ntwo\n" if mode == "csv" else None)
    from tests.fixtures.compatibility import FixtureResponse
    def interrupt(response, decode_unicode=False):
        if mode != "csv":
            yield b'{"results":['
        raise KeyboardInterrupt("interrupted")
    monkeypatch.setattr(FixtureResponse, "iter_lines", interrupt)
    stream = iter(iterator)
    with pytest.raises(KeyboardInterrupt):
        next(stream)
    assert session.responses[-1].closed


@pytest.mark.parametrize("payload,message", [
    (b'bad\n', "bad header"),
    (b'{"results":[\n[1]\n', "interrupted"),
    (b'{"results":[\ninvalid\n', "parsing line"),
    (b'{"results":[\n],"wasSuccessful":false,"statusCode":500,"error":"failed"}\n', "failed"),
    (b'{"results":[\n],"statusCode":200}\n', "status fragment"),
])
def test_json_failure_closes_actual_response(payload, message):
    iterator, session, _ = make_iterator("json", payload=payload)
    with pytest.raises(WebserviceError, match=message):
        [row for row in iterator]
    assert session.responses[-1].closed


def test_flat_parser_next_alias_errors_and_closure():
    connection = BytesIO(b"  one \n [ERROR] failed\n")
    parser = results.FlatFileIterator(connection, str.upper)
    assert iter(parser) is parser and parser.next() == "ONE"
    with pytest.raises(WebserviceError, match="failed"):
        next(parser)
    assert connection.closed
    connection = BytesIO(b"two\n")
    parser = results.FlatFileIterator(connection, str.upper)
    assert next(parser) == "TWO"
    with pytest.raises(StopIteration):
        next(parser)
    assert connection.closed


def test_explicit_dict_retains_exact_decimal_and_other_float_conversion():
    iterator, session, _ = make_iterator(payload=b'{"results":[\n["a",1234567890.123456789,1.25]\n],"wasSuccessful":true}\n', decimal_paths=[VIEWS[1]])
    row = next(iterator)
    assert row[VIEWS[1]] == Decimal("1234567890.123456789")
    assert type(row[VIEWS[2]]) is float
    iterator.close()
    assert session.responses[-1].closed


def test_unknown_formats_fail_before_request():
    for mode in ("nope", "invalid"):
        with pytest.raises(ValueError):
            make_iterator(mode)


@pytest.mark.parametrize("mode", ["json", "jsonrows"])
def test_raw_json_identity_includes_null_and_mapping_payloads(mode):
    iterator, session, _ = make_iterator(mode, payload=b'{"results":[\nnull,\n{"value":42}\n],"wasSuccessful":true}\n')
    assert [row for row in iterator] == [None, {"value": 42}]
    assert session.responses[-1].closed


def test_public_json_reader_method_closes_on_interrupt_and_parser_error():
    connection = InterruptConnection(b'{"results":[\n[1]\n', 2)
    reader = results.JSONIterator(connection, lambda value: value)
    with pytest.raises(KeyboardInterrupt):
        reader.get_next_row_from_connection()
    assert connection.closed
    connection = BytesIO(b'{"results":[\n[1]\n')
    def fail(value):
        raise ValueError("parser failed")
    reader = results.JSONIterator(connection, fail)
    with pytest.raises(ValueError, match="parser failed"):
        reader.get_next_row_from_connection()
    assert connection.closed


@pytest.mark.parametrize("mode", ["rr", "csv", "jsonrows"])
def test_new_formats_preserve_bounded_post_get_fallback(mode):
    payload = b'one,two\n' if mode == "csv" else fixture_bytes("rows-modern.json")
    session = FixtureSession({("POST", "/service/query/results"): (500, b"POST unavailable"),
                              ("GET", "/service/query/results"): payload})
    opener = runtime.InterMineURLOpener(session=session, timeout=(3, 9), verify_tls="ca.pem", user_agent="custom", token="secret")
    service = SimpleNamespace(root=ROOT, version=8, opener=opener)
    iterator = runtime.ResultIterator(service, "/query/results", {"query": "xml"}, mode, VIEWS)
    assert len([row for row in iterator]) == (1 if mode == "csv" else 3)
    assert [request.method for request in session.requests] == ["POST", "GET"]
    wire = "json" if mode == "rr" else mode
    assert parse_qs(session.requests[-1].url.partition('?')[2])["format"] == [wire]
    assert all(response.closed for response in session.responses)
    for request in session.requests:
        assert request.options["verify"] == "ca.pem" and request.options["timeout"] == (3, 9)
        assert request.headers["User-Agent"] == "custom" and request.headers["Authorization"] == "Token secret"
    opener.close()
    assert session.close_calls == 0
    iterator = runtime.ResultIterator(service, "/query/results", {"query": "x" * 5000}, mode, VIEWS)
    with pytest.raises(WebserviceError):
        iter(iterator)
    assert [request.method for request in session.requests] == ["POST", "GET", "POST"]
    assert session.responses[-1].closed


def test_actual_response_closes_on_header_interrupt(monkeypatch):
    from tests.fixtures.compatibility import FixtureResponse
    iterator, session, _ = make_iterator()
    def interrupted(response, decode_unicode=False):
        raise KeyboardInterrupt("header interrupted")
        yield
    monkeypatch.setattr(FixtureResponse, "iter_lines", interrupted)
    with pytest.raises(KeyboardInterrupt):
        iter(iterator)
    assert session.responses[-1].closed and session.responses[-1].raw.closed
    assert session.close_calls == 0


@pytest.mark.parametrize("mode", ["json", "jsonrows", "csv", "count", "rr"])
def test_direct_stream_outlives_temporary_result_facade(mode):
    iterator, session, _ = make_iterator(mode, payload=b"one\n" if mode in {"csv", "count"} else None)
    stream = iter(iterator)
    del iterator
    assert not session.responses[-1].closed
    assert len([row for row in stream]) == (1 if mode in {"csv", "count"} else 3)
    assert session.responses[-1].closed


def test_json_public_next_alias_and_multiline_footer_status():
    connection = BytesIO(b'{"results":[\n[1]\n],\n"wasSuccessful":true,\n"statusCode":200}\n')
    reader = results.JSONIterator(connection, lambda value: value[0])
    assert iter(reader) is reader and reader.next() == 1
    with pytest.raises(StopIteration):
        reader.next()
    assert connection.closed
    assert reader.header == '{"results":['
    assert '"statusCode":200}' in reader.footer
    reader.check_return_status()


@pytest.mark.parametrize("phase", ["header", "row", "footer"])
def test_actual_response_interruptions_close_at_each_stream_phase(phase, monkeypatch):
    from tests.fixtures.compatibility import FixtureResponse
    iterator, session, _ = make_iterator("json")
    def interrupted(response, decode_unicode=False):
        if phase != "header":
            yield b'{"results":['
        if phase == "footer":
            yield b'[1]'
            yield b'],'
        raise KeyboardInterrupt(phase)
    monkeypatch.setattr(FixtureResponse, "iter_lines", interrupted)
    with pytest.raises(KeyboardInterrupt):
        [row for row in iterator]
    assert session.responses[-1].closed and session.responses[-1].raw.closed
    assert session.close_calls == 0


@pytest.mark.parametrize("phase", ["header", "footer"])
def test_json_status_buffers_and_error_previews_remain_capped(phase):
    large = 'x' * (runtime._JSON_STATUS_BUFFER_MAX_CHARS * 2)
    payload = (large + '\n') if phase == "header" else ('{"results":[\n],"irrelevant":"' + large + '"}\n')
    iterator, session, _ = make_iterator("json", payload=payload.encode())
    with pytest.raises(WebserviceError) as error:
        [row for row in iterator]
    assert len(str(error.value)) < runtime._JSON_ERROR_PREVIEW_MAX_CHARS + 100
    assert session.responses[-1].closed


def test_flat_parser_interruption_closes_connection():
    connection = BytesIO(b'one\n')
    def interrupt(value):
        raise KeyboardInterrupt("flat parser")
    parser = results.FlatFileIterator(connection, interrupt)
    with pytest.raises(KeyboardInterrupt):
        parser.next()
    assert connection.closed
