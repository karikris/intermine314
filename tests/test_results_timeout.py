from intermine314.config.constants import DEFAULT_CONNECT_TIMEOUT_SECONDS, DEFAULT_REQUEST_TIMEOUT_SECONDS
from intermine314.service.errors import TorConfigurationError
from intermine314.service.session import InterMineURLOpener
from pathlib import Path
import pytest


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
    assert kwargs["timeout"] == (DEFAULT_CONNECT_TIMEOUT_SECONDS, float(DEFAULT_REQUEST_TIMEOUT_SECONDS))
    assert kwargs["verify"] is True


def test_open_honors_explicit_timeout_for_requests_session():
    opener = InterMineURLOpener()
    session = _Session()
    opener._session = session

    opener.open("https://example.org/service/version/ws", timeout=7)

    assert session.calls
    _method, _url, kwargs = session.calls[0]
    assert kwargs["timeout"] == (7.0, 7.0)


def test_open_rebuilds_session_when_missing(monkeypatch):
    built = []

    def fake_build_session(*, proxy_url, user_agent, tor_mode=False):
        built.append((proxy_url, user_agent, tor_mode))
        return _Session()

    monkeypatch.setattr("intermine314.service.session.build_session", fake_build_session)
    opener = InterMineURLOpener()
    opener._session = None

    opener.open("https://example.org/service/version/ws")

    assert built[-1] == (None, None, False)
    assert len(built) >= 1
    assert opener._session is not None
    assert opener._session.calls
    _method, _url, kwargs = opener._session.calls[0]
    assert kwargs["timeout"] == (DEFAULT_CONNECT_TIMEOUT_SECONDS, float(DEFAULT_REQUEST_TIMEOUT_SECONDS))


def test_open_uses_verify_tls_flag_for_requests_session():
    opener = InterMineURLOpener(verify_tls=False)
    session = _Session()
    opener._session = session

    opener.open("https://example.org/service/version/ws")

    assert session.calls
    _method, _url, kwargs = session.calls[0]
    assert kwargs["verify"] is False


@pytest.mark.parametrize(
    "verify_input,expected_verify",
    [
        (True, True),
        (False, False),
        ("/tmp/custom-ca.pem", "/tmp/custom-ca.pem"),
        (Path("/tmp/custom-ca.pem"), Path("/tmp/custom-ca.pem")),
        (None, True),
    ],
)
def test_open_preserves_verify_tls_types_for_requests_session(verify_input, expected_verify):
    opener = InterMineURLOpener(verify_tls=verify_input)
    session = _Session()
    opener._session = session

    opener.open("https://example.org/service/version/ws")

    assert session.calls
    _method, _url, kwargs = session.calls[0]
    assert kwargs["verify"] == expected_verify
    assert type(kwargs["verify"]) is type(expected_verify)


@pytest.mark.parametrize("verify_input", [0, 1, 3.14, object()])
def test_open_rejects_invalid_verify_tls_types(verify_input):
    with pytest.raises(TypeError, match="verify_tls must be a bool, str, pathlib.Path, or None"):
        InterMineURLOpener(verify_tls=verify_input)


def test_proxy_url_sets_session_proxies():
    opener = InterMineURLOpener(proxy_url="socks5h://127.0.0.1:9050")
    assert opener._session is not None
    assert opener._session.proxies.get("http") == "socks5h://127.0.0.1:9050"
    assert opener._session.proxies.get("https") == "socks5h://127.0.0.1:9050"
    assert opener._session.trust_env is False


def test_proxy_url_rejects_dns_unsafe_scheme_when_tor_mode_enabled():
    with pytest.raises(TorConfigurationError, match="socks5h://"):
        InterMineURLOpener(proxy_url="socks5://127.0.0.1:9050", tor_mode=True)


def test_open_preserves_ca_bundle_path_when_tor_mode_enabled():
    opener = InterMineURLOpener(
        verify_tls=Path("/tmp/custom-ca.pem"),
        proxy_url="socks5h://127.0.0.1:9050",
        tor_mode=True,
    )
    session = _Session()
    opener._session = session

    opener.open("https://example.org/service/version/ws")

    assert session.calls
    _method, _url, kwargs = session.calls[0]
    assert kwargs["verify"] == Path("/tmp/custom-ca.pem")
    assert isinstance(kwargs["verify"], Path)
