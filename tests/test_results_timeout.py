from intermine314.constants import DEFAULT_REQUEST_TIMEOUT_SECONDS
from intermine314.results import InterMineURLOpener


class _Raw:
    def read(self, _size=-1):
        return b""


class _Response:
    status_code = 200
    reason = "OK"
    headers = {}
    content = b""
    raw = _Raw()

    def iter_lines(self, decode_unicode=False):
        _ = decode_unicode
        return iter(())

    def close(self):
        return None


class _Session:
    def __init__(self):
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return _Response()


def test_open_uses_default_timeout_for_requests_session():
    opener = InterMineURLOpener()
    session = _Session()
    opener._session = session

    opener.open("https://example.org/service/version/ws")

    assert session.calls
    _method, _url, kwargs = session.calls[0]
    assert kwargs["timeout"] == DEFAULT_REQUEST_TIMEOUT_SECONDS


def test_open_honors_explicit_timeout_for_requests_session():
    opener = InterMineURLOpener()
    session = _Session()
    opener._session = session

    opener.open("https://example.org/service/version/ws", timeout=7)

    assert session.calls
    _method, _url, kwargs = session.calls[0]
    assert kwargs["timeout"] == 7


def test_open_uses_timeout_for_urllib_fallback(monkeypatch):
    calls = []

    class _UrlResp:
        def read(self):
            return b"ok"

        def close(self):
            return None

    def fake_urlopen(req, timeout=None):
        calls.append((req, timeout))
        return _UrlResp()

    monkeypatch.setattr("intermine314.results.urlopen", fake_urlopen)
    opener = InterMineURLOpener()
    opener._session = None

    opener.open("https://example.org/service/version/ws")

    assert calls
    _req, timeout = calls[0]
    assert timeout == DEFAULT_REQUEST_TIMEOUT_SECONDS
