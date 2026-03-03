import intermine314.util.core as core
from intermine314.model import Model


class _FakeResponse:
    def __init__(self, payload: bytes):
        self.content = payload

    def raise_for_status(self):
        return None


class _StreamingRaw:
    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)
        self._index = 0
        self.closed = False
        self.decode_content = False

    def read(self, size=-1):
        _ = size
        if self._index >= len(self._lines):
            return b""
        remaining = b"".join(self._lines[self._index :])
        self._index = len(self._lines)
        return remaining

    def readline(self, size=-1):
        _ = size
        if self._index >= len(self._lines):
            return b""
        line = self._lines[self._index]
        self._index += 1
        return line

    def close(self):
        self.closed = True


class _StreamingResponse:
    def __init__(self, lines: list[bytes]):
        self.raw = _StreamingRaw(lines)
        self.closed = False
        self.close_calls = 0

    def raise_for_status(self):
        return None

    def close(self):
        self.closed = True
        self.close_calls += 1


class _FakeSession:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.calls = []
        self.close_calls = 0

    def get(self, url, timeout=None, verify=True):
        self.calls.append((url, timeout, verify))
        return _FakeResponse(self.payload)

    def close(self):
        self.close_calls += 1


class _StreamingSession:
    def __init__(self, lines: list[bytes]):
        self._lines = list(lines)
        self.calls = []
        self.responses = []
        self.close_calls = 0

    def get(self, url, timeout=None, verify=True, stream=False):
        self.calls.append((url, timeout, verify, stream))
        response = _StreamingResponse(self._lines)
        self.responses.append(response)
        return response

    def close(self):
        self.close_calls += 1


def test_openanything_http_uses_proxy_aware_session(monkeypatch):
    xml = b"<query/>"
    session = _FakeSession(xml)
    build_calls = []
    resolve_calls = []

    def fake_resolve_proxy_url(value=None):
        resolve_calls.append(value)
        return "socks5h://127.0.0.1:9050"

    def fake_build_session(*, proxy_url, user_agent):
        build_calls.append((proxy_url, user_agent))
        return session

    def fail_urlopen(_source):
        raise AssertionError("urlopen should not be used for http(s) sources")

    monkeypatch.setattr(core, "resolve_proxy_url", fake_resolve_proxy_url)
    monkeypatch.setattr(core, "build_session", fake_build_session)
    monkeypatch.setattr(core, "urlopen", fail_urlopen)

    handle = core.openAnything("https://example.org/query.xml", proxy_url="socks5h://10.0.0.1:9050")

    assert handle.read() == xml
    assert resolve_calls == ["socks5h://10.0.0.1:9050"]
    assert build_calls == [("socks5h://127.0.0.1:9050", None)]
    assert session.calls == [("https://example.org/query.xml", None, True)]


def test_model_http_source_uses_session_transport_not_urllib(monkeypatch):
    model_xml = b'<model name="mock" package="org.mock"></model>'
    session = _FakeSession(model_xml)

    def fake_build_session(*, proxy_url, user_agent):
        _ = proxy_url
        _ = user_agent
        return session

    def fail_urlopen(_source):
        raise AssertionError("urlopen should not be used for Model(http-url)")

    monkeypatch.setattr(core, "build_session", fake_build_session)
    monkeypatch.setattr(core, "urlopen", fail_urlopen)

    model = Model("https://example.org/service/model")

    assert model.name == "mock"
    assert model.package_name == "org.mock"
    assert session.calls == [("https://example.org/service/model", None, True)]


def test_openanything_streaming_response_closes_on_early_termination():
    session = _StreamingSession([b"<model>\n", b"<class/>\n", b"</model>\n"])

    with core.openAnything("https://example.org/service/model", session=session) as stream:
        for line in stream:
            assert line == b"<model>\n"
            break

    assert session.calls == [("https://example.org/service/model", None, True, True)]
    assert len(session.responses) == 1
    response = session.responses[0]
    assert response.close_calls == 1
    assert response.closed is True
    assert response.raw.closed is True


def test_openanything_internal_stream_session_closes_after_context_exit(monkeypatch):
    session = _StreamingSession([b"<model>\n", b"<class/>\n", b"</model>\n"])

    def fake_build_session(*, proxy_url, user_agent):
        _ = (proxy_url, user_agent)
        return session

    monkeypatch.setattr(core, "build_session", fake_build_session)

    with core.openAnything("https://example.org/service/model") as stream:
        assert stream.readline() == b"<model>\n"

    assert session.close_calls == 1
    assert len(session.responses) == 1
    assert session.responses[0].closed is True


def test_openanything_internal_nonstream_session_closes_after_buffered_read(monkeypatch):
    session = _FakeSession(b"<query/>")

    def fake_build_session(*, proxy_url, user_agent):
        _ = (proxy_url, user_agent)
        return session

    monkeypatch.setattr(core, "build_session", fake_build_session)

    handle = core.openAnything("https://example.org/query.xml")
    assert handle.read() == b"<query/>"
    assert session.close_calls == 1
