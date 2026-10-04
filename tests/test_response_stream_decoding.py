"""Sized reads use the same HTTP content decoding as whole-response reads."""

import gzip
from io import BytesIO
from xml.dom import minidom

import pytest
from requests import Response
from urllib3.response import HTTPResponse

from intermine314.service.session import _ResponseStreamAdapter

XML = '<templates><template name="Müller 🧬"/></templates>'.encode()


class TrackingResponse(Response):
    def __init__(self, payload, encoding=None, content_length=None):
        super().__init__()
        self.status_code = 200
        self.close_calls = 0
        headers = {"Content-Length": str(len(payload) if content_length is None else content_length)}
        if encoding:
            headers["Content-Encoding"] = encoding
        self.headers.update(headers)
        self.raw = HTTPResponse(
            body=BytesIO(payload), headers=headers, preload_content=False, decode_content=False,
        )

    def close(self):
        self.close_calls += 1
        super().close()


def test_sized_reads_decode_gzip_for_xml_parser():
    response = TrackingResponse(gzip.compress(XML), "gzip")
    with _ResponseStreamAdapter(response) as stream:
        document = minidom.parse(stream)
        assert document.getElementsByTagName("template")[0].getAttribute("name") == "Müller 🧬"
    assert response.close_calls == 1


def test_zero_read_neither_consumes_nor_closes_response():
    response = TrackingResponse(XML)
    stream = _ResponseStreamAdapter(response)
    assert stream.read(0) == b""
    assert not stream.closed and response.close_calls == 0
    assert stream.read() == XML
    assert stream.closed and response.close_calls == 1


@pytest.mark.parametrize("failure", [OSError, KeyboardInterrupt])
def test_sized_read_failure_closes_response_without_masking_error(monkeypatch, failure):
    response = TrackingResponse(XML)
    stream = _ResponseStreamAdapter(response)

    def fail_read(size, *, decode_content):
        assert decode_content is True
        raise failure("read failed")

    monkeypatch.setattr(response.raw, "read", fail_read)
    with pytest.raises(failure, match="read failed"):
        stream.read(7)
    assert stream.closed and response.close_calls == 1
    stream.close()
    assert response.close_calls == 1
