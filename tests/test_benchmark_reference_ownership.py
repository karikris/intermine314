"""Reference-only ownership regressions matching the pinned 1.13.0 iterator shape.

Upstream results.py ResultIterator.__iter__ opens a connection before constructing
JSONIterator, which parses its header immediately. Neither iterator has close();
the reader's .connection owns the response. Tests retain that ownership shape
without adding an original-client dependency to the base test environment.
"""
import sys
from types import ModuleType, SimpleNamespace

import polars as pl
import pytest

from benchmarks import bench_fetch
from tests.test_benchmark_storage import ROWS, export


class Connection:
    def __init__(self, rows, failure):
        self.rows = iter(rows)
        self.failure = failure
        self.read = 0
        self.close_calls = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.read == 1 and self.failure == "partial":
            raise OSError("partial page")
        if self.failure == "interrupt":
            raise KeyboardInterrupt("interrupted reader")
        self.read += 1
        return next(self.rows)

    def close(self):
        self.close_calls += 1


class NoCloseReader:
    def __init__(self, connection):
        self.connection = connection
        if connection.failure == "header":
            raise ValueError("invalid header during reader construction")

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.connection)


class NoCloseResult:
    def __init__(self, opener):
        self.opener = opener
        self._it = None  # Upstream __iter__ does not populate this cache.

    def __iter__(self):
        connection = self.opener.open("https://example.test/results", "payload")
        return NoCloseReader(connection)


class ReferenceQuery:
    views = list(ROWS[0])

    def __init__(self, failures):
        self.failures = iter(failures)
        self.connections = []

    def results(self, *, row, start, size):
        failure = next(self.failures, None)
        rows = ROWS if failure == "overflow" else ROWS[start:start + size]
        def open_connection(*args):
            assert all(con.close_calls == 1 for con in self.connections)
            connection = Connection(rows, failure)
            self.connections.append(connection)
            return connection
        return NoCloseResult(SimpleNamespace(open=open_connection))


def test_original_iterator_shape_closes_each_retry_and_success_connection(monkeypatch, tmp_path):
    query = ReferenceQuery(["partial", None, None])
    result = export(monkeypatch, tmp_path, query)
    assert pl.read_parquet(result["path"]).to_dicts() == ROWS
    assert len(query.connections) == 3
    assert [con.close_calls for con in query.connections] == [1, 1, 1]


@pytest.mark.parametrize("failure,error", [
    ("header", ValueError), ("interrupt", KeyboardInterrupt), ("overflow", ValueError),
])
def test_original_iterator_shape_closes_connection_when_reader_never_returns_or_finishes(monkeypatch, tmp_path, failure, error):
    query = ReferenceQuery([failure])
    path = tmp_path / "reference.parquet"
    path.write_bytes(b"original")
    with pytest.raises(error):
        export(monkeypatch, tmp_path, query)
    assert path.read_bytes() == b"original"
    assert [con.close_calls for con in query.connections] == [1]


@pytest.fixture
def reference_transport(monkeypatch):
    class Opener:
        pass
    original = ModuleType("intermine")
    results = ModuleType("intermine.results")
    results.InterMineURLOpener = Opener
    monkeypatch.setitem(sys.modules, "intermine", original)
    monkeypatch.setitem(sys.modules, "intermine.results", results)
    monkeypatch.setattr(bench_fetch, "_load_legacy_intermine_classes", lambda: (object, Exception))
    monkeypatch.setattr(bench_fetch, "_LEGACY_TRANSPORT_PATCH_SIGNATURE", None)
    sessions = []
    responses = []
    def configure(*, request_error=None, status=200, content_error=None, close_error=False, lines=()):
        class Response:
            status_code = status
            text = "server failure"
            headers = {}
            closed = False
            def iter_lines(self):
                return iter(lines)
            @property
            def content(self):
                if content_error:
                    raise content_error
                return b"3"
            def close(self):
                self.closed = True
                if close_error:
                    raise RuntimeError("secondary response-close failure")
        class Session:
            closed = False
            def request(self, *args, **kwargs):
                if request_error:
                    raise request_error
                response = Response()
                responses.append(response)
                return response
            def close(self):
                self.closed = True
        def session():
            value = Session()
            sessions.append(value)
            return value
        monkeypatch.setattr(bench_fetch.requests, "Session", session)
        bench_fetch._configure_legacy_intermine_transport(
            mine_url="https://example.test", user_agent=None, proxy_url=None,
        )
        return Opener()
    return configure, sessions, responses


@pytest.mark.parametrize("mode", ["metadata", "count", "request", "status", "content", "scope"])
def test_reference_adapter_closes_owned_session_response_and_preserves_errors(reference_transport, mode):
    configure, sessions, responses = reference_transport
    if mode == "request":
        opener = configure(request_error=OSError("request failed"))
        with pytest.raises(OSError, match="request failed"):
            opener.open("https://example.test")
    elif mode == "status":
        opener = configure(status=503, close_error=True)
        with pytest.raises(RuntimeError, match="status=503"):
            opener.open("https://example.test")
    elif mode == "content":
        opener = configure(content_error=KeyboardInterrupt("content failed"), close_error=True)
        stream = opener.open("https://example.test")
        with pytest.raises(KeyboardInterrupt, match="content failed"):
            stream.read()
    elif mode == "metadata":
        opener = configure()
        stream = opener.open("https://example.test")
        assert stream.read() == b"3"
        assert stream.read() == b"3"  # Buffered reads remain available after close.
    elif mode == "count":
        opener = configure(lines=[b"3"])
        assert list(opener.open("https://example.test")) == [b"3"]
    else:
        opener = configure(close_error=True)
        with pytest.raises(ValueError, match="construction failed"):
            with bench_fetch._legacy_response_scope():
                opener.open("https://example.test")
                raise ValueError("construction failed")
    assert sessions and all(session.closed for session in sessions)
    assert all(response.closed for response in responses)


def test_reference_response_tracking_drops_closed_streams(reference_transport):
    configure, sessions, responses = reference_transport
    opener = configure()
    with bench_fetch._legacy_response_scope() as owned:
        for _ in range(50):
            stream = opener.open("https://example.test")
            assert len(owned) == 1
            stream.read()
            assert len(owned) == 0
    assert len(sessions) == 50
    assert all(session.closed for session in sessions)
    assert all(response.closed for response in responses)


def test_request_payload_conversion_does_not_leak_a_session(reference_transport):
    configure, sessions, _ = reference_transport
    opener = configure()
    class InvalidPayload:
        def __str__(self):
            raise ValueError("bad payload")
    with pytest.raises(ValueError, match="bad payload"):
        opener.open("https://example.test", data=InvalidPayload())
    assert all(session.closed for session in sessions)


def test_export_scope_closes_stream_lost_before_results_returns(monkeypatch, tmp_path, reference_transport):
    configure, sessions, responses = reference_transport
    opener = configure(close_error=True)
    class Query:
        views = list(ROWS[0])

        def results(self, **kwargs):
            opener.open("https://example.test")
            raise ValueError("result construction failed")
    with pytest.raises(ValueError, match="result construction failed"):
        export(monkeypatch, tmp_path, Query())
    assert all(session.closed for session in sessions)
    assert all(response.closed for response in responses)


def test_reference_fetch_lane_closes_real_reader_shape_on_retry(monkeypatch):
    from benchmarks import benchmarks as cli
    query = ReferenceQuery(["partial", None, None])
    monkeypatch.setattr(bench_fetch, "get_legacy_service_class", lambda: object)
    monkeypatch.setattr(bench_fetch, "_configure_legacy_intermine_transport", lambda **kw: None)
    monkeypatch.setattr(bench_fetch, "_retriable_exceptions_for_mode", lambda mode: (OSError,))
    monkeypatch.setattr(bench_fetch, "make_query", lambda *a, **kw: query)
    monkeypatch.setattr(bench_fetch, "count_with_retry", lambda *a, **kw: (3, 0, 0, 0))
    monkeypatch.setattr(bench_fetch, "_retry_wait_seconds", lambda attempt: 0)
    runtime = bench_fetch.build_common_runtime_kwargs(cli.parse_args(["--legacy-batch-size", "2"]))
    result = bench_fetch.run_mode(
        mode="intermine_batched", mine_url="https://example.test", rows_target=3,
        page_size=2, workers=None, query_root_class="Gene", query_views=list(ROWS[0]),
        query_joins=[], **runtime,
    )
    assert result.rows == 3
    assert result.retries == 1
    assert len(query.connections) == 3
    assert all(con.close_calls == 1 for con in query.connections)
