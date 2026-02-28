from __future__ import annotations

from pathlib import Path

import pytest

import intermine314.registry.api as registry_api


def test_get_version_forwards_explicit_registry_transport_kwargs(monkeypatch):
    calls = []
    session = object()
    verify_tls = Path("/tmp/custom-ca.pem")

    class _FakeRegistry:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def info(self, name):
            assert name == "mineA"
            return {
                "api_version": "36",
                "release_version": "2026.1",
                "intermine_version": "5.0.0",
            }

    monkeypatch.setattr(registry_api, "Registry", _FakeRegistry)
    result = registry_api.get_version(
        "mineA",
        registry_url="https://registry.example.org/service/instances",
        request_timeout=19,
        proxy_url="socks5h://127.0.0.1:9050",
        session=session,
        verify_tls=verify_tls,
        tor=True,
        allow_http_over_tor=True,
    )

    assert calls == [
        {
            "registry_url": "https://registry.example.org/service/instances",
            "request_timeout": 19,
            "proxy_url": "socks5h://127.0.0.1:9050",
            "session": session,
            "verify_tls": verify_tls,
            "tor": True,
            "allow_http_over_tor": True,
        }
    ]
    assert result == {
        "api_version": "36",
        "release_version": "2026.1",
        "intermine_version": "5.0.0",
    }


def test_get_data_forwards_transport_kwargs_to_registry_and_service(monkeypatch):
    registry_calls = []
    service_calls = []
    session = object()

    class _FakeRegistry:
        def __init__(self, **kwargs):
            registry_calls.append(kwargs)

        def info(self, name):
            assert name == "mineB"
            return {
                "url": "https://mine-b.example/service",
                "api_version": "36",
                "release_version": "2026.1",
                "intermine_version": "5.0.0",
            }

    class _FakeQuery:
        def add_view(self, *_views):
            return None

        def rows(self, **_kwargs):
            return [{"DataSet.name": "set-1"}]

    class _FakeService:
        def __init__(self, root, **kwargs):
            service_calls.append({"root": root, **kwargs})

        def new_query(self, _class_name):
            return _FakeQuery()

    monkeypatch.setattr(registry_api, "Registry", _FakeRegistry)
    monkeypatch.setattr(registry_api, "Service", _FakeService)

    result = registry_api.get_data(
        "mineB",
        registry_url="https://registry.example.org/service/instances",
        request_timeout=17,
        proxy_url="socks5h://127.0.0.1:9050",
        session=session,
        verify_tls="/tmp/custom-ca.pem",
        tor=True,
        allow_http_over_tor=True,
    )

    assert result == ["set-1"]
    assert registry_calls[0]["proxy_url"] == "socks5h://127.0.0.1:9050"
    assert registry_calls[0]["session"] is session
    assert registry_calls[0]["verify_tls"] == "/tmp/custom-ca.pem"
    assert service_calls == [
        {
            "root": "https://mine-b.example/service",
            "request_timeout": 17,
            "proxy_url": "socks5h://127.0.0.1:9050",
            "session": session,
            "verify_tls": "/tmp/custom-ca.pem",
            "tor": True,
            "allow_http_over_tor": True,
        }
    ]


def test_get_mines_forwards_transport_kwargs_to_service_classmethod(monkeypatch):
    calls = []
    session = object()

    class _FakeService:
        @classmethod
        def get_all_mines(cls, **kwargs):
            calls.append(kwargs)
            return [{"name": "MineA"}, {"name": "MineA"}, {"name": "MineB"}]

    monkeypatch.setattr(registry_api, "Service", _FakeService)
    result = registry_api.get_mines(
        organism="Zea mays",
        registry_url="https://registry.example.org/service/instances",
        request_timeout=11,
        proxy_url="socks5h://127.0.0.1:9050",
        session=session,
        verify_tls=False,
        tor=True,
        allow_http_over_tor=True,
    )

    assert result == ["MineA", "MineB"]
    assert calls == [
        {
            "organism": "Zea mays",
            "registry_url": "https://registry.example.org/service/instances",
            "request_timeout": 11,
            "proxy_url": "socks5h://127.0.0.1:9050",
            "session": session,
            "verify_tls": False,
            "tor": True,
            "allow_http_over_tor": True,
        }
    ]


def test_get_version_lookup_raises_typed_error(monkeypatch):
    class _BrokenRegistry:
        def __init__(self, **_kwargs):
            return None

        def info(self, _name):
            raise RuntimeError("registry unavailable")

    monkeypatch.setattr(registry_api, "Registry", _BrokenRegistry)
    with pytest.raises(registry_api.RegistryLookupError):
        registry_api.get_version("mineA")
