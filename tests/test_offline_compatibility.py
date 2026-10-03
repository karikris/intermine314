"""Behavioral evidence for the existing native API and reusable offline transport."""
from urllib.parse import parse_qs

import pytest

from intermine314.service.errors import WebserviceError
from intermine314.service.service import Service
from intermine314.service.session import InterMineURLOpener
from tests.fixtures.compatibility import FixtureOpener, FixtureSession, fixture_bytes

ROOT = "https://offline.example/service"


def test_fixture_opener_returns_independent_streams_and_captures_methods():
    opener = FixtureOpener({("POST", "/service/query/results"): b"first\nsecond\n"})
    first = opener.open(ROOT + "/query/results", b"query=x")
    assert next(first) == b"first\n"
    with opener.open(ROOT + "/query/results", "query=y") as second:
        assert second.read(5) == b"first"
        assert second.read() == b"\nsecond\n"
    assert first.closed is False
    assert second.close_calls == 1
    first.close()
    first.close()
    assert first.close_calls == 1
    assert [(call.method, call.data) for call in opener.requests] == [
        ("POST", b"query=x"), ("POST", "query=y")
    ]
    with pytest.raises(AssertionError, match="Unregistered offline request"):
        opener.open(ROOT + "/missing")


@pytest.mark.parametrize("method,data", [("GET", None), ("POST", b"query=x"), ("DELETE", None)])
def test_native_transport_closes_response_and_preserves_borrowed_session(method, data):
    session = FixtureSession({(method, "/service/probe"): b"one\ntwo\n"})
    with InterMineURLOpener(session=session, timeout=(2, 5), verify_tls=False) as opener:
        with opener.open(ROOT + "/probe", data=data, method=method) as stream:
            assert next(stream) == b"one"
        stream.close()
    assert session.responses[0].close_calls == 1
    assert session.close_calls == 0
    request = session.requests[0]
    assert request.method == method
    assert request.data == data
    assert request.options["stream"] is True
    assert request.options["timeout"] == (2, 5)
    assert request.options["verify"] is False


@pytest.mark.parametrize("version,payload,wire_format", [(8, "rows-modern.json", "json"), (7, "rows-legacy.json", "jsonrows")])
def test_native_service_decodes_both_wire_protocols_and_caches_model(version, payload, wire_format):
    session = FixtureSession.service(version=version, rows=fixture_bytes(payload))
    with Service(ROOT, session=session) as service:
        query = service.select("Employee.name", "Employee.age", "Employee.fullTime")
        rows = [row for row in query.results()]
        assert rows == [
            {"Employee.name": "foo", "Employee.age": "bar", "Employee.fullTime": "baz"},
            {"Employee.name": 123, "Employee.age": 1.23, "Employee.fullTime": -1.23},
            {"Employee.name": True, "Employee.age": False, "Employee.fullTime": None},
        ]
        assert type(rows[-1]["Employee.name"]) is bool
        assert type(rows[1]["Employee.name"]) is int
        assert service.select("Employee.name").model.name == "testmodel"
        assert service.version == version
    assert [call.path for call in session.requests] == [
        "/service/version/ws", "/service/model", "/service/query/results"
    ]
    params = parse_qs(session.requests[-1].data.decode())
    assert params["format"] == [wire_format]
    assert 'model="testmodel"' in params["query"][0]
    assert all(response.close_calls == 1 for response in session.responses)
    assert session.close_calls == 0


def test_native_query_count_uses_count_wire_format_and_closes_body():
    session = FixtureSession.service(rows=b"42\n")
    with Service(ROOT, session=session) as service:
        assert service.select("Employee.name").count() == 42
    request = session.requests[-1]
    assert request.method == "POST"
    assert parse_qs(request.data.decode())["format"] == ["count"]
    assert session.responses[-1].close_calls == 1


@pytest.mark.parametrize("payload,match", [
    (b'not a JSON header\n', "bad header"),
    (b'{"results":[\n["value"]\n],"wasSuccessful":false,"error":"offline failure","statusCode":500}\n', "offline failure"),
])
def test_native_result_errors_release_offline_response(payload, match):
    session = FixtureSession.service(rows=payload)
    with Service(ROOT, session=session) as service:
        with pytest.raises(WebserviceError, match=match):
            _ = [row for row in service.select("Employee.name").results()]
    assert session.responses[-1].close_calls == 1


def test_native_http_error_closes_response_before_raising():
    session = FixtureSession({("GET", "/service/bad"): (400, b'{"error":"invalid path"}')})
    with InterMineURLOpener(session=session) as opener:
        with pytest.raises(WebserviceError, match="problem with our request"):
            opener.open(ROOT + "/bad")
    assert session.responses[0].close_calls == 1


def test_native_iterator_explicit_close_after_one_row():
    session = FixtureSession.service()
    with Service(ROOT, session=session) as service:
        iterator = service.select("Employee.name", "Employee.age", "Employee.fullTime").results()
        assert next(iterator)["Employee.name"] == "foo"
        iterator.close()
        iterator.close()
    assert session.responses[-1].close_calls == 1


def test_native_service_factory_uses_isolated_sessions(native_service_factory, offline_session_factory):
    first_session = offline_session_factory(version=8)
    second_session = offline_session_factory(version=7)
    first = native_service_factory(session=first_session)
    second = native_service_factory(session=second_session)
    assert first.version == 8
    assert second.version == 7
    first.close()
    assert second.select("Employee.name").model.name == "testmodel"
    assert len(first_session.requests) == 1
    assert len(second_session.requests) == 2
    assert all(response.closed for response in first_session.responses + second_session.responses)


def test_native_transport_stream_exhaustion_closes_response():
    session = FixtureSession({("GET", "/service/probe"): b"one\ntwo\n"})
    with InterMineURLOpener(session=session) as opener:
        stream = opener.open(ROOT + "/probe")
        assert list(stream) == [b"one", b"two"]
        assert stream.closed is True
        stream.close()
    assert session.responses[0].close_calls == 1
