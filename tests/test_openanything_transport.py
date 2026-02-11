import intermine314.util.core as core
from intermine314.model import Model


class _FakeResponse:
    def __init__(self, payload: bytes):
        self.content = payload

    def raise_for_status(self):
        return None


class _FakeSession:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.calls = []

    def get(self, url, timeout=None, verify=True):
        self.calls.append((url, timeout, verify))
        return _FakeResponse(self.payload)


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
