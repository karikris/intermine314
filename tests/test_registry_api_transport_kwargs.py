from __future__ import annotations

from pathlib import Path
import warnings

import pytest

import intermine314.registry.api as registry_api


def _reset_registry_api_warning_state():
    registry_api._LEGACY_API_DEPRECATION_EMITTED.clear()
    registry_api._TOR_ENV_IMPLICIT_TRANSPORT_WARNED = False
    registry_api._LEGACY_API_CALLS_BY_NAME.clear()
    registry_api._LEGACY_API_SUPPRESSED_BY_NAME.clear()


def test_getversion_forwards_explicit_registry_transport_kwargs(monkeypatch):
    _reset_registry_api_warning_state()
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

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        result = registry_api.getVersion(
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
    dep_msgs = [str(item.message) for item in caught if issubclass(item.category, DeprecationWarning)]
    assert len(dep_msgs) == 1
    assert "intermine314.registry.api.getVersion is deprecated" in dep_msgs[0]
    assert result["API Version:"] == "36"


def test_getdata_forwards_transport_kwargs_to_registry_and_service(monkeypatch):
    _reset_registry_api_warning_state()
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

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        result = registry_api.getData(
            "mineB",
            registry_url="https://registry.example.org/service/instances",
            request_timeout=17,
            proxy_url="socks5h://127.0.0.1:9050",
            session=session,
            verify_tls="/tmp/custom-ca.pem",
            tor=True,
            allow_http_over_tor=True,
        )

    assert result is None
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
    dep_msgs = [str(item.message) for item in caught if issubclass(item.category, DeprecationWarning)]
    assert len(dep_msgs) == 1
    assert "intermine314.registry.api.getData is deprecated" in dep_msgs[0]


def test_getmines_forwards_transport_kwargs_to_service_classmethod(monkeypatch):
    _reset_registry_api_warning_state()
    calls = []
    session = object()

    class _FakeService:
        @classmethod
        def get_all_mines(cls, **kwargs):
            calls.append(kwargs)
            return [{"name": "MineA"}, {"name": "MineA"}, {"name": "MineB"}]

    monkeypatch.setattr(registry_api, "Service", _FakeService)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        result = registry_api.getMines(
            organism="Zea mays",
            registry_url="https://registry.example.org/service/instances",
            request_timeout=11,
            proxy_url="socks5h://127.0.0.1:9050",
            session=session,
            verify_tls=False,
            tor=True,
            allow_http_over_tor=True,
        )

    assert result is None
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
    dep_msgs = [str(item.message) for item in caught if issubclass(item.category, DeprecationWarning)]
    assert len(dep_msgs) == 1
    assert "intermine314.registry.api.getMines is deprecated" in dep_msgs[0]


def test_legacy_wrapper_tor_env_warning_emits_once_without_explicit_transport(monkeypatch):
    _reset_registry_api_warning_state()

    class _FakeRegistry:
        def __init__(self, **_kwargs):
            return None

        def info(self, _name):
            return {
                "api_version": "36",
                "release_version": "2026.1",
                "intermine_version": "5.0.0",
            }

    monkeypatch.setattr(registry_api, "Registry", _FakeRegistry)
    monkeypatch.setenv("INTERMINE314_PROXY_URL", "socks5h://127.0.0.1:9050")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        registry_api.getVersion("mineA")
        registry_api.getVersion("mineA")

    runtime_msgs = [str(item.message) for item in caught if issubclass(item.category, RuntimeWarning)]
    assert len(runtime_msgs) == 1
    assert "without explicit transport" in runtime_msgs[0]


def test_legacy_wrapper_tor_env_warning_not_emitted_with_explicit_transport(monkeypatch):
    _reset_registry_api_warning_state()

    class _FakeRegistry:
        def __init__(self, **_kwargs):
            return None

        def info(self, _name):
            return {
                "api_version": "36",
                "release_version": "2026.1",
                "intermine_version": "5.0.0",
            }

    monkeypatch.setattr(registry_api, "Registry", _FakeRegistry)
    monkeypatch.setenv("INTERMINE314_PROXY_URL", "socks5h://127.0.0.1:9050")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        registry_api.getVersion("mineA", proxy_url="socks5h://127.0.0.1:9050")

    runtime_msgs = [str(item.message) for item in caught if issubclass(item.category, RuntimeWarning)]
    assert runtime_msgs == []


def test_modern_get_version_returns_structured_payload(monkeypatch):
    _reset_registry_api_warning_state()

    class _FakeRegistry:
        def __init__(self, **_kwargs):
            return None

        def info(self, _name):
            return {
                "api_version": "36",
                "release_version": "2026.1",
                "intermine_version": "5.0.0",
            }

    monkeypatch.setattr(registry_api, "Registry", _FakeRegistry)

    result = registry_api.get_version("mineA")
    assert result == {
        "api_version": "36",
        "release_version": "2026.1",
        "intermine_version": "5.0.0",
    }


def test_modern_get_mines_returns_names(monkeypatch):
    _reset_registry_api_warning_state()

    class _FakeService:
        @classmethod
        def get_all_mines(cls, **_kwargs):
            return [{"name": "MineA"}, {"name": "MineA"}, {"name": "MineB"}]

    monkeypatch.setattr(registry_api, "Service", _FakeService)

    names = registry_api.get_mines(organism="Zea mays")
    assert names == ["MineA", "MineB"]


def test_legacy_wrapper_suppression_metrics_increment(monkeypatch):
    _reset_registry_api_warning_state()

    class _BrokenRegistry:
        def __init__(self, **_kwargs):
            return None

        def info(self, _name):
            raise RuntimeError("registry unavailable")

    monkeypatch.setattr(registry_api, "Registry", _BrokenRegistry)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = registry_api.getVersion("mineA")
    assert result == registry_api.NO_SUCH_MINE

    metrics = registry_api.legacy_registry_api_metrics()
    assert metrics["legacy_api_calls_total"] == 1
    assert metrics["legacy_api_calls_by_name"]["getVersion"] == 1
    assert metrics["legacy_api_suppressed_total"] == 1
    assert metrics["legacy_api_suppressed_by_name"]["getVersion"] == 1


def test_modern_lookup_raises_typed_error(monkeypatch):
    _reset_registry_api_warning_state()

    class _BrokenRegistry:
        def __init__(self, **_kwargs):
            return None

        def info(self, _name):
            raise RuntimeError("registry unavailable")

    monkeypatch.setattr(registry_api, "Registry", _BrokenRegistry)

    with pytest.raises(registry_api.RegistryLookupError):
        registry_api.get_version("mineA")
