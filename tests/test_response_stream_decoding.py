"""Sized reads use the same HTTP content decoding as whole-response reads."""

import gzip
import zlib
from compression import zstd
from io import BytesIO
from xml.dom import minidom

import pytest
from requests import Response
from urllib3.exceptions import DecodeError, ProtocolError
from urllib3.response import HTTPResponse

from intermine314.service.session import _ResponseStreamAdapter

XML = '<templates><template name="Müller 🧬"/></templates>'.encode()
ENCODINGS = {None: lambda data: data, "gzip": gzip.compress, "deflate": zlib.compress, "zstd": zstd.compress}


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


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("size", [1, 7, 16384])
def test_repeated_small_reads_decode_unicode_xml_and_close_at_eof(encoding, size):
    response = TrackingResponse(ENCODINGS[encoding](XML), encoding)
    stream = _ResponseStreamAdapter(response)
    chunks = []
    while chunk := stream.read(size):
        assert 0 < len(chunk) <= size
        chunks.append(chunk)
    assert b"".join(chunks) == XML
    assert stream.closed and response.close_calls == 1
    stream.close()
    assert response.close_calls == 1


@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("remainder_size", [-1, None])
def test_partial_then_whole_read_returns_decoded_remainder(encoding, remainder_size):
    response = TrackingResponse(ENCODINGS[encoding](XML), encoding)
    with _ResponseStreamAdapter(response) as stream:
        prefix = stream.read(11)
        assert prefix == XML[:11]
        assert prefix + stream.read(remainder_size) == XML
    assert response.close_calls == 1


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "zstd"])
def test_corrupt_compression_closes_response(encoding):
    response = TrackingResponse(b"not a compressed body", encoding)
    stream = _ResponseStreamAdapter(response)
    with pytest.raises(DecodeError):
        stream.read(7)
    assert stream.closed and response.close_calls == 1


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "zstd"])
def test_truncated_compressed_http_body_raises_and_closes(encoding):
    payload = ENCODINGS[encoding](XML)
    response = TrackingResponse(payload[:len(payload) // 2], encoding, content_length=len(payload))
    stream = _ResponseStreamAdapter(response)
    with pytest.raises((ProtocolError, DecodeError)):
        while stream.read(7):
            pass
    assert stream.closed and response.close_calls == 1


def test_decoded_response_does_not_close_borrowed_session():
    from intermine314.service.session import InterMineURLOpener

    class Session:
        close_calls = 0

        def request(self, *args, **kwargs):
            self.response = TrackingResponse(zstd.compress(XML), "zstd")
            return self.response

        def close(self):
            self.close_calls += 1

    session = Session()
    opener = InterMineURLOpener(session=session)
    with opener.open("https://offline.example/templates") as stream:
        assert minidom.parse(stream).documentElement.tagName == "templates"
    opener.close()
    assert session.response.close_calls == 1
    assert session.close_calls == 0
