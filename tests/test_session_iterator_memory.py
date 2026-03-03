from __future__ import annotations

import pytest

from intermine314.service import session as session_module


def _json_stream_lines(rows):
    lines = [b'{"results":[']
    for idx, row in enumerate(rows):
        suffix = b"," if idx < len(rows) - 1 else b""
        lines.append(row + suffix)
    lines.append(b'],"wasSuccessful":true,"statusCode":200,"error":null}')
    return lines


class _LineConnection:
    def __init__(self, lines):
        self._iter = iter(lines)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def close(self):
        self.closed = True


class _Service:
    def __init__(self, opener, *, version=8):
        self.version = int(version)
        self.root = "https://example.org/service"
        self.opener = opener


class _TrackingOpener:
    def __init__(self, lines):
        self._lines = list(lines)
        self.calls = []
        self.connections = []

    def open(self, url, data=None, *_args, method=None, **_kwargs):
        self.calls.append({"url": url, "data": data, "method": method})
        con = _LineConnection(self._lines)
        self.connections.append(con)
        return con


class _AdapterRaw:
    def __init__(self, payload):
        self._payload = payload
        self.closed = False
        self._used = False

    def read(self, size=-1):
        _ = size
        if self._used:
            return b""
        self._used = True
        return self._payload

    def close(self):
        self.closed = True


class _AdapterResponse:
    def __init__(self, lines):
        self._lines = list(lines)
        self.raw = _AdapterRaw(b"".join(self._lines))
        self.close_calls = 0
        self.closed = False

    @property
    def content(self):
        return b"".join(self._lines)

    def iter_lines(self, decode_unicode=False):
        _ = decode_unicode
        return iter(self._lines)

    def close(self):
        self.close_calls += 1
        self.closed = True
        self.raw.close()


def test_result_iterator_closes_connection_when_consumer_breaks_early():
    opener = _TrackingOpener(_json_stream_lines([b'["geneA"]', b'["geneB"]']))
    service = _Service(opener, version=8)
    iterator = session_module.ResultIterator(
        service,
        "/query/results",
        {"query": "xml"},
        "list",
        ["Gene.symbol"],
    )
    for row in iterator:
        assert row == ["geneA"]
        break
    assert len(opener.connections) == 1
    assert opener.connections[0].closed is True


def test_result_iterator_closes_connection_when_consumer_raises():
    opener = _TrackingOpener(_json_stream_lines([b'["geneA"]', b'["geneB"]']))
    service = _Service(opener, version=8)
    iterator = session_module.ResultIterator(
        service,
        "/query/results",
        {"query": "xml"},
        "list",
        ["Gene.symbol"],
    )
    with pytest.raises(RuntimeError, match="stop now"):
        for _row in iterator:
            raise RuntimeError("stop now")
    assert len(opener.connections) == 1
    assert opener.connections[0].closed is True


def test_result_iterator_close_method_releases_connection():
    opener = _TrackingOpener(_json_stream_lines([b'["geneA"]', b'["geneB"]']))
    service = _Service(opener, version=8)
    iterator = session_module.ResultIterator(
        service,
        "/query/results",
        {"query": "xml"},
        "list",
        ["Gene.symbol"],
    )
    first = next(iterator)
    iterator.close()
    assert first == ["geneA"]
    assert len(opener.connections) == 1
    assert opener.connections[0].closed is True


def test_response_stream_adapter_closes_on_full_iteration():
    response = _AdapterResponse([b"line-a", b"line-b"])
    adapter = session_module._ResponseStreamAdapter(response)

    assert list(adapter) == [b"line-a", b"line-b"]
    assert adapter.closed is True
    assert response.closed is True
    assert response.close_calls == 1


def test_response_stream_adapter_closes_on_context_exit_after_early_break():
    response = _AdapterResponse([b"line-a", b"line-b"])
    adapter = session_module._ResponseStreamAdapter(response)

    with adapter as stream:
        assert next(stream) == b"line-a"

    assert adapter.closed is True
    assert response.closed is True
    assert response.close_calls == 1


def test_response_stream_adapter_read_closes_response_after_full_read():
    response = _AdapterResponse([b"line-a", b"line-b"])
    adapter = session_module._ResponseStreamAdapter(response)

    payload = adapter.read()

    assert payload == b"line-aline-b"
    assert adapter.closed is True
    assert response.closed is True
    assert response.close_calls == 1
