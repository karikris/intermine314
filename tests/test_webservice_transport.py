import json
from unittest.mock import patch
from pathlib import Path

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


class _BulkRegistryOpener:
    payload = b'{"instances":[]}'

    def __init__(self, *args, **kwargs):
        _ = (args, kwargs)
        self._session = object()

    def open(self, _url, *_args, **_kwargs):
        return _BytesResponse(self.payload)


def test_registry_passes_proxy_and_verify_to_opener():
    _FakeRegistryOpener.calls = []
    with patch("intermine314.service.service.InterMineURLOpener", _FakeRegistryOpener):
        Registry(proxy_url="socks5h://127.0.0.1:9050", verify_tls=False, request_timeout=7)

    assert _FakeRegistryOpener.calls
    kwargs = _FakeRegistryOpener.calls[0]
    assert kwargs["proxy_url"] == "socks5h://127.0.0.1:9050"
    assert kwargs["verify_tls"] is False
    assert kwargs["request_timeout"] == 7


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
def test_registry_verify_tls_passthrough_types(verify_input, expected_verify):
    _FakeRegistryOpener.calls = []
    with patch("intermine314.service.service.InterMineURLOpener", _FakeRegistryOpener):
        Registry(proxy_url="socks5h://127.0.0.1:9050", verify_tls=verify_input, request_timeout=7)

    assert _FakeRegistryOpener.calls
    kwargs = _FakeRegistryOpener.calls[0]
    assert kwargs["verify_tls"] == expected_verify
    assert type(kwargs["verify_tls"]) is type(expected_verify)


def test_service_passes_proxy_and_verify_to_opener():
    _FakeServiceOpener.calls = []
    with patch("intermine314.service.service.InterMineURLOpener", _FakeServiceOpener):
        Service("https://example.org/service", proxy_url="socks5h://127.0.0.1:9050", verify_tls=False, request_timeout=9)

    assert _FakeServiceOpener.calls
    kwargs = _FakeServiceOpener.calls[0]
    assert kwargs["proxy_url"] == "socks5h://127.0.0.1:9050"
    assert kwargs["verify_tls"] is False
    assert kwargs["request_timeout"] == 9


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
def test_service_verify_tls_passthrough_types(verify_input, expected_verify):
    _FakeServiceOpener.calls = []
    with patch("intermine314.service.service.InterMineURLOpener", _FakeServiceOpener):
        Service(
            "https://example.org/service",
            proxy_url="socks5h://127.0.0.1:9050",
            verify_tls=verify_input,
            request_timeout=9,
        )

    assert _FakeServiceOpener.calls
    kwargs = _FakeServiceOpener.calls[0]
    assert kwargs["verify_tls"] == expected_verify
    assert type(kwargs["verify_tls"]) is type(expected_verify)


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


def test_registry_service_cache_is_bounded():
    mine_count = 48
    instances = [
        {"name": f"Mine{i}", "url": f"https://example.org/mine{i}/service"}
        for i in range(mine_count)
    ]
    _BulkRegistryOpener.payload = json.dumps({"instances": instances}).encode("utf-8")

    class _FakeService:
        def __init__(self, root, **kwargs):
            self.root = root
            self.kwargs = kwargs

    with patch("intermine314.service.service.InterMineURLOpener", _BulkRegistryOpener):
        with patch("intermine314.service.service.Service", _FakeService):
            registry = Registry("https://registry.example.org/service/instances")
            for i in range(mine_count):
                _ = registry[f"Mine{i}"]

            cache = getattr(registry, "_Registry__mine_cache")
            assert len(cache) <= registry.MAX_CACHED_SERVICES


def test_registry_service_cache_respects_override_cap():
    mine_count = 6
    instances = [
        {"name": f"Mine{i}", "url": f"https://example.org/mine{i}/service"}
        for i in range(mine_count)
    ]
    _BulkRegistryOpener.payload = json.dumps({"instances": instances}).encode("utf-8")

    class _FakeService:
        def __init__(self, root, **kwargs):
            self.root = root
            self.kwargs = kwargs

    with patch("intermine314.service.service.InterMineURLOpener", _BulkRegistryOpener):
        with patch("intermine314.service.service.Service", _FakeService):
            registry = Registry("https://registry.example.org/service/instances", max_cached_services=3)
            for i in range(mine_count):
                _ = registry[f"Mine{i}"]

            cache = getattr(registry, "_Registry__mine_cache")
            assert len(cache) == 3
            assert list(cache.keys()) == ["mine3", "mine4", "mine5"]


def test_registry_service_cache_metrics_track_hits_misses_and_evictions():
    instances = [
        {"name": f"Mine{i}", "url": f"https://example.org/mine{i}/service"}
        for i in range(3)
    ]
    _BulkRegistryOpener.payload = json.dumps({"instances": instances}).encode("utf-8")

    class _FakeService:
        def __init__(self, root, **kwargs):
            self.root = root
            self.kwargs = kwargs

    with patch("intermine314.service.service.InterMineURLOpener", _BulkRegistryOpener):
        with patch("intermine314.service.service.Service", _FakeService):
            registry = Registry("https://registry.example.org/service/instances", max_cached_services=2)
            _ = registry["Mine0"]  # miss
            _ = registry["Mine1"]  # miss
            _ = registry["Mine0"]  # hit
            _ = registry["Mine2"]  # miss + eviction

    metrics = registry.service_cache_metrics()
    assert metrics["cache_size"] == 2
    assert metrics["max_cache_size"] == 2
    assert metrics["cache_hits"] == 1
    assert metrics["cache_misses"] == 3
    assert metrics["cache_evictions"] == 1
    assert metrics["registry_service_cache_size"] == 2
    assert metrics["registry_service_cache_max_size"] == 2
    assert metrics["registry_service_cache_hits"] == 1
    assert metrics["registry_service_cache_misses"] == 3
    assert metrics["registry_service_cache_evictions"] == 1


@pytest.mark.parametrize("value,error_type", [(0, ValueError), (-1, ValueError), (True, TypeError), ("4", TypeError)])
def test_registry_rejects_invalid_cache_cap(value, error_type):
    with patch("intermine314.service.service.InterMineURLOpener", _BulkRegistryOpener):
        with pytest.raises(error_type):
            Registry("https://registry.example.org/service/instances", max_cached_services=value)
