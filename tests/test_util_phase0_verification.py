from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

import intermine314.query.builder as query_builder
import intermine314.util.core as core


class _TrackingResponse:
    def __init__(self, payload: bytes = b"<query/>"):
        self._payload = payload
        self.closed = False

    @property
    def content(self):
        return self._payload

    def raise_for_status(self):
        return None

    def close(self):
        self.closed = True


class _TrackingSession:
    def __init__(self, response: _TrackingResponse):
        self.response = response
        self.calls: list[dict[str, object]] = []

    def get(self, url, timeout=None, verify=True):
        self.calls.append({"url": url, "timeout": timeout, "verify": verify})
        return self.response


class _TrackingHandle:
    def __init__(self, payload: str):
        self.payload = payload
        self.closed = False

    def read(self, *_args, **_kwargs):
        return self.payload

    def close(self):
        self.closed = True


@pytest.mark.parametrize(
    "verify_value",
    [True, False, "/tmp/ca.pem", Path("/tmp/ca.pem")],
)
def test_openanything_preserves_verify_tls_values(verify_value):
    response = _TrackingResponse()
    session = _TrackingSession(response)

    handle = core.openAnything("https://example.org/model.xml", session=session, verify_tls=verify_value)

    assert handle.read() == b"<query/>"
    assert session.calls[0]["verify"] == verify_value
    assert type(session.calls[0]["verify"]) is type(verify_value)


def test_openanything_http_closes_response_even_if_consumer_raises():
    response = _TrackingResponse(payload=b"<query/>")
    session = _TrackingSession(response)

    handle = core.openAnything("https://example.org/query.xml", session=session)

    with pytest.raises(RuntimeError, match="consumer-failure"):
        _ = handle.read()
        raise RuntimeError("consumer-failure")

    assert response.closed is True


def test_query_from_xml_closes_handle_on_parse_error(monkeypatch):
    handle = _TrackingHandle("<invalid")

    def _raise_parse(_handle):
        raise RuntimeError("query-parse-failure")

    monkeypatch.setattr(query_builder, "openAnything", lambda _xml: handle)
    monkeypatch.setattr(query_builder.minidom, "parse", _raise_parse)

    with pytest.raises(RuntimeError, match="query-parse-failure"):
        query_builder.Query.from_xml("<query/>", object())

    assert handle.closed is True


def test_template_from_xml_closes_handle_on_parse_error(monkeypatch):
    handle = _TrackingHandle("<invalid")

    def _raise_parse(_handle):
        raise RuntimeError("template-parse-failure")

    def _fake_query_from_xml(cls, _xml, *_args, **_kwargs):
        return cls.__new__(cls)

    monkeypatch.setattr(query_builder, "openAnything", lambda _xml: handle)
    monkeypatch.setattr(query_builder.minidom, "parse", _raise_parse)
    monkeypatch.setattr(query_builder.Query, "from_xml", classmethod(_fake_query_from_xml))

    with pytest.raises(RuntimeError, match="template-parse-failure"):
        query_builder.Template.from_xml("<template/>", object())

    assert handle.closed is True


def test_json_helper_matches_standard_and_orjson_backends(monkeypatch):
    util_json = importlib.import_module("intermine314.util.json")
    payload = {
        "alpha": 1,
        "beta": ["x", 7, None],
        "gamma": {"nested": True, "value": 4.2},
    }

    monkeypatch.setattr(util_json, "_JSON_BACKEND", "json")
    monkeypatch.setattr(util_json, "_loads", json.loads)
    monkeypatch.setattr(util_json, "_dumps", json.dumps)
    std_dump = util_json.json_dumps(payload)
    std_load_str = util_json.json_loads(std_dump)
    std_load_bytes = util_json.json_loads(std_dump.encode("utf-8"))

    class _FakeOrjson:
        @staticmethod
        def loads(value):
            if isinstance(value, (bytes, bytearray)):
                value = value.decode("utf-8")
            return json.loads(value)

        @staticmethod
        def dumps(value):
            return json.dumps(value, separators=(",", ":"), sort_keys=False).encode("utf-8")

    monkeypatch.setattr(util_json, "_JSON_BACKEND", "orjson")
    monkeypatch.setattr(util_json, "_loads", _FakeOrjson.loads)
    monkeypatch.setattr(util_json, "_dumps", _FakeOrjson.dumps)
    fast_dump = util_json.json_dumps(payload)
    fast_load_str = util_json.json_loads(fast_dump)
    fast_load_bytes = util_json.json_loads(fast_dump.encode("utf-8"))

    assert std_load_str == payload
    assert std_load_bytes == payload
    assert fast_load_str == payload
    assert fast_load_bytes == payload
    assert json.loads(std_dump) == json.loads(fast_dump)
