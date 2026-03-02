from __future__ import annotations

import pytest

from intermine314.service import session as session_module
from intermine314.service.errors import WebserviceError


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


class _PostFallbackOpener:
    def __init__(self, lines):
        self._lines = list(lines)
        self.calls = []

    def open(self, url, data=None, *_args, method=None, **_kwargs):
        self.calls.append({"url": url, "data": data, "method": method})
        if data is not None:
            raise WebserviceError("post failed")
        if method == "GET":
            return _LineConnection(self._lines)
        raise AssertionError("unexpected request type")


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


def test_result_iterator_large_payload_skips_get_fallback():
    opener = _PostFallbackOpener(_json_stream_lines([b'["geneA"]']))
    service = _Service(opener, version=8)
    huge_query = "x" * (session_module._FALLBACK_GET_MAX_PAYLOAD_BYTES + 1024)
    iterator = session_module.ResultIterator(
        service,
        "/query/results",
        {"query": huge_query},
        "jsonrows",
        ["Gene.symbol"],
    )
    with pytest.raises(WebserviceError, match="post failed"):
        list(iterator)
    assert len(opener.calls) == 1
    assert opener.calls[0]["method"] is None


def test_result_iterator_small_payload_uses_get_fallback():
    opener = _PostFallbackOpener(_json_stream_lines([b'["geneA"]']))
    service = _Service(opener, version=8)
    iterator = session_module.ResultIterator(
        service,
        "/query/results",
        {"query": "xml"},
        "list",
        ["Gene.symbol"],
    )
    rows = list(iterator)
    assert rows == [["geneA"]]
    assert len(opener.calls) >= 2
    assert opener.calls[-1]["method"] == "GET"


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
