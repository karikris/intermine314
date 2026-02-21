import os
import logging
from io import BytesIO, StringIO
from time import perf_counter
from urllib.parse import urlparse
from urllib.request import urlopen

from intermine314.service.transport import build_session, resolve_proxy_url

LOG = logging.getLogger(__name__)
HTTP_SCHEMES = ("http", "https")


def _is_http_source(source) -> bool:
    try:
        parsed = urlparse(str(source))
    except Exception:
        return False
    return parsed.scheme in HTTP_SCHEMES


def _resolve_verify_tls(verify_tls):
    if verify_tls is None:
        return True
    if isinstance(verify_tls, bool):
        return verify_tls
    if isinstance(verify_tls, (str, os.PathLike)):
        return verify_tls
    raise TypeError("verify_tls must be a bool, str, pathlib.Path, or None")


def _close_response_quietly(response):
    close = getattr(response, "close", None)
    if callable(close):
        close()


class _ManagedHTTPResponseStream:
    def __init__(self, response, *, source: str):
        self._response = response
        self._stream = response.raw
        self._source = source
        self._closed = False
        self._bytes_read = 0
        self._started = perf_counter()
        if hasattr(self._stream, "decode_content"):
            self._stream.decode_content = True

    def _track_bytes(self, chunk):
        if isinstance(chunk, (bytes, bytearray, str)):
            self._bytes_read += len(chunk)

    def read(self, size=-1):
        chunk = self._stream.read(size)
        self._track_bytes(chunk)
        # Deterministically close after full reads, or when chunked reads hit EOF.
        if (size is None or size < 0) or chunk in (b"", ""):
            self.close()
        return chunk

    def readline(self, size=-1):
        chunk = self._stream.readline(size)
        self._track_bytes(chunk)
        if chunk in (b"", ""):
            self.close()
        return chunk

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.close()
        except Exception:
            pass
        finally:
            self._response.close()
        if LOG.isEnabledFor(logging.DEBUG):
            elapsed_ms = (perf_counter() - self._started) * 1000.0
            LOG.debug(
                "openAnything source=url url=%s bytes_read=%d elapsed_ms=%.3f closed=true",
                self._source,
                self._bytes_read,
                elapsed_ms,
            )

    @property
    def closed(self):
        return self._closed

    def __iter__(self):
        return self

    def __next__(self):
        line = self.readline()
        if line in (b"", ""):
            raise StopIteration
        return line

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


def openAnything(source, *, session=None, timeout=None, verify_tls=True, proxy_url=None):
    # Already file-like.
    if hasattr(source, "read"):
        return source

    # Route HTTP(S) through the shared session transport to honor SOCKS/proxy settings.
    if _is_http_source(source):
        http_session = session or build_session(proxy_url=resolve_proxy_url(proxy_url), user_agent=None)
        request_kwargs = {"timeout": timeout, "verify": _resolve_verify_tls(verify_tls)}
        try:
            response = http_session.get(str(source), stream=True, **request_kwargs)
        except TypeError:
            response = http_session.get(str(source), **request_kwargs)
        try:
            response.raise_for_status()
        except Exception:
            _close_response_quietly(response)
            raise
        if getattr(response, "raw", None) is not None:
            return _ManagedHTTPResponseStream(response, source=str(source))
        try:
            payload = response.content
        finally:
            _close_response_quietly(response)
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        return BytesIO(payload)

    # Local file path.
    try:
        return open(source)
    except (ValueError, IOError, OSError, TypeError):
        pass

    # Non-HTTP URL schemes (for example: ftp://, file://).
    try:
        return urlopen(source)
    except (ValueError, IOError, OSError, TypeError):
        pass

    return StringIO(str(source))


class ReadableException(Exception):
    def __init__(self, message, cause=None):
        self.message = message
        self.cause = cause

    def __str__(self):
        if self.cause is None:
            return repr(self.message)
        else:
            return repr(self.message) + repr(self.cause)
