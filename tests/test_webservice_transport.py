from unittest.mock import patch

import pytest

from intermine314.service import Registry, Service


class _BytesResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload

    def close(self):
        return None


class _FakeRegistryOpener:
    calls = []

    def __init__(self, *args, **kwargs):
        _FakeRegistryOpener.calls.append(kwargs)
        self._session = object()

    def open(self, _url, *_args, **_kwargs):
        return _BytesResponse(b'{"instances": []}')


class _FakeServiceOpener:
    calls = []

    def __init__(self, *args, **kwargs):
        _FakeServiceOpener.calls.append(kwargs)
        self._session = object()

    def open(self, _url, *_args, **_kwargs):
        return _BytesResponse(b"35")


class _FailingSession:
    def get(self, *_args, **_kwargs):
        raise AssertionError("anonymous token fetch must not call session.get directly")


class _FakeAnonymousTokenOpener:
    def __init__(self):
        self.calls = []
        self._timeout = (2, 7)
        self._session = _FailingSession()

    def open(self, url, *_args, **kwargs):
        self.calls.append((url, kwargs))
        return _BytesResponse(b'{"token":"anon-token"}')


class _FakeListManager:
    def delete_temporary_lists(self):
        return None


def test_registry_passes_proxy_and_verify_to_opener():
    _FakeRegistryOpener.calls = []
    with patch("intermine314.service.service.InterMineURLOpener", _FakeRegistryOpener):
        Registry(proxy_url="socks5h://127.0.0.1:9050", verify_tls=False, request_timeout=7)

    assert _FakeRegistryOpener.calls
    kwargs = _FakeRegistryOpener.calls[0]
    assert kwargs["proxy_url"] == "socks5h://127.0.0.1:9050"
    assert kwargs["verify_tls"] is False
    assert kwargs["request_timeout"] == 7


def test_service_passes_proxy_and_verify_to_opener():
    _FakeServiceOpener.calls = []
    with patch("intermine314.service.service.InterMineURLOpener", _FakeServiceOpener):
        Service("https://example.org/service", proxy_url="socks5h://127.0.0.1:9050", verify_tls=False, request_timeout=9)

    assert _FakeServiceOpener.calls
    kwargs = _FakeServiceOpener.calls[0]
    assert kwargs["proxy_url"] == "socks5h://127.0.0.1:9050"
    assert kwargs["verify_tls"] is False
    assert kwargs["request_timeout"] == 9


def test_get_anonymous_token_uses_opener_transport():
    service = Service.__new__(Service)
    service.opener = _FakeAnonymousTokenOpener()
    service._list_manager = _FakeListManager()

    token = service.get_anonymous_token("https://example.org/service")

    assert token == "anon-token"
    assert service.opener.calls == [
        ("https://example.org/service/session", {"method": "GET", "timeout": (2, 7)})
    ]


def test_service_rejects_http_endpoint_when_tor_enabled():
    with pytest.raises(ValueError, match="https://"):
        Service("http://example.org/service", tor=True)


def test_service_rejects_http_endpoint_for_tor_proxy():
    with pytest.raises(ValueError, match="https://"):
        Service("http://example.org/service", proxy_url="socks5h://127.0.0.1:9050")


def test_service_allows_http_endpoint_when_explicitly_opted_in():
    _FakeServiceOpener.calls = []
    with patch("intermine314.service.service.InterMineURLOpener", _FakeServiceOpener):
        Service(
            "http://example.org/service",
            proxy_url="socks5h://127.0.0.1:9050",
            allow_http_over_tor=True,
        )
    assert _FakeServiceOpener.calls


def test_registry_rejects_http_endpoint_when_tor_enabled():
    with pytest.raises(ValueError, match="https://"):
        Registry("http://registry.example.org/service/instances", tor=True)


def test_registry_rejects_http_endpoint_for_tor_proxy():
    with pytest.raises(ValueError, match="https://"):
        Registry("http://registry.example.org/service/instances", proxy_url="socks5h://127.0.0.1:9050")


def test_registry_allows_http_endpoint_when_explicitly_opted_in():
    _FakeRegistryOpener.calls = []
    with patch("intermine314.service.service.InterMineURLOpener", _FakeRegistryOpener):
        Registry(
            "http://registry.example.org/service/instances",
            proxy_url="socks5h://127.0.0.1:9050",
            allow_http_over_tor=True,
        )
    assert _FakeRegistryOpener.calls
