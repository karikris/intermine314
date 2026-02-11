from unittest.mock import patch

from intermine314.webservice import Registry, Service


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
