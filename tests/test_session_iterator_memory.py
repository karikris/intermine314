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
    assert isinstance(opener.calls[0]["data"], bytes)


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

    row_iter = iter(iterator)
    rows = [next(row_iter)]
    with pytest.raises(StopIteration):
        next(row_iter)

    assert rows == [["geneA"]]
    assert len(opener.calls) == 2
    assert opener.calls[0]["method"] is None
    assert opener.calls[1]["method"] == "GET"
    assert "?" in opener.calls[1]["url"]


def test_json_iterator_row_parse_error_message_is_capped():
    huge_payload = ("A" * 50000) + "TAIL_ROW_TOKEN"
    bad_row = ('{"bad":"' + huge_payload).encode("utf-8")
    iterator = session_module.JSONIterator(_LineConnection([b'{"results":[', bad_row]), lambda x: x)

    with pytest.raises(WebserviceError) as excinfo:
        next(iterator)

    message = str(excinfo.value)
    assert len(message) < 6000
    assert "TAIL_ROW_TOKEN" not in message
    assert "...<truncated>" in message


def test_json_iterator_footer_status_error_buffer_is_capped():
    huge_footer_tail = ("Y" * (session_module._JSON_STATUS_BUFFER_MAX_CHARS + 8000)) + "TAIL_FOOTER_TOKEN"
    footer_line = ('],"wasSuccessful":tru,' + huge_footer_tail).encode("utf-8")
    iterator = session_module.JSONIterator(_LineConnection([b'{"results":[', footer_line]), lambda x: x)

    with pytest.raises(WebserviceError) as excinfo:
        next(iterator)

    message = str(excinfo.value)
    assert len(iterator.footer) <= session_module._JSON_STATUS_BUFFER_MAX_CHARS
    assert len(message) < 6000
    assert "TAIL_FOOTER_TOKEN" not in message
    assert "...<truncated>" in message


def test_json_iterator_header_error_buffer_is_capped():
    huge_header_tail = ("H" * (session_module._JSON_STATUS_BUFFER_MAX_CHARS + 5000)) + "TAIL_HEADER_TOKEN"
    bad_header = ('{"meta":"' + huge_header_tail + '"}').encode("utf-8")

    with pytest.raises(WebserviceError) as excinfo:
        session_module.JSONIterator(_LineConnection([bad_header]), lambda x: x)

    message = str(excinfo.value)
    assert len(message) < 6000
    assert "TAIL_HEADER_TOKEN" not in message
    assert "...<truncated>" in message
