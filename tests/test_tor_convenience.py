import intermine314.service.tor as tor
import pytest
from pathlib import Path
from intermine314.config.runtime_defaults import get_runtime_defaults
from intermine314.config.transport_policy import resolve_http_retry_policy
from intermine314.service.errors import TorConfigurationError
from intermine314.service import Registry, Service

_SERVICE_DEFAULTS = get_runtime_defaults().service_defaults
DEFAULT_REGISTRY_INSTANCES_URL = _SERVICE_DEFAULTS.default_registry_instances_url
DEFAULT_TOR_PROXY_SCHEME = _SERVICE_DEFAULTS.default_tor_proxy_scheme
DEFAULT_TOR_SOCKS_HOST = _SERVICE_DEFAULTS.default_tor_socks_host
DEFAULT_TOR_SOCKS_PORT = _SERVICE_DEFAULTS.default_tor_socks_port


class _FakeService:
    def __init__(self, root, **kwargs):
        self.root = root
        self.kwargs = kwargs


class _FakeRegistry:
    def __init__(self, registry_url, **kwargs):
        self.registry_url = registry_url
        self.kwargs = kwargs


class _ProxySession:
    def __init__(self, http_proxy=None, https_proxy=None, trust_env=False):
        self.proxies = {}
        if http_proxy is not None:
            self.proxies["http"] = http_proxy
        if https_proxy is not None:
            self.proxies["https"] = https_proxy
        self.trust_env = trust_env

    def request(self, *_args, **_kwargs):
        raise AssertionError("request should not be called in helper wiring tests")


def test_tor_service_helper_wires_proxy_and_session(monkeypatch):
    fake_session = object()
    session_calls = []

    def fake_tor_session(**kwargs):
        session_calls.append(dict(kwargs))
        return fake_session

    monkeypatch.setattr(tor, "tor_session", fake_tor_session)
    monkeypatch.setattr("intermine314.service.service.Service", _FakeService)

    service = tor.tor_service("https://example.org/service", token="abc", request_timeout=12)

    assert isinstance(service, _FakeService)
    assert service.root == "https://example.org/service"
    assert service.kwargs["token"] == "abc"
    assert service.kwargs["request_timeout"] == 12
    assert service.kwargs["proxy_url"] == f"{DEFAULT_TOR_PROXY_SCHEME}://{DEFAULT_TOR_SOCKS_HOST}:{DEFAULT_TOR_SOCKS_PORT}"
    assert service.kwargs["session"] is fake_session
    assert service.kwargs["tor"] is True
    assert service.kwargs["allow_http_over_tor"] is False
    assert session_calls == [
        {
            "host": DEFAULT_TOR_SOCKS_HOST,
            "port": DEFAULT_TOR_SOCKS_PORT,
            "scheme": DEFAULT_TOR_PROXY_SCHEME,
            "strict": True,
            "allow_insecure_tor_proxy_scheme": False,
        }
    ]


def test_tor_registry_helper_wires_proxy_and_session(monkeypatch):
    fake_session = object()
    session_calls = []

    def fake_tor_session(**kwargs):
        session_calls.append(dict(kwargs))
        return fake_session

    monkeypatch.setattr(tor, "tor_session", fake_tor_session)
    monkeypatch.setattr("intermine314.service.service.Registry", _FakeRegistry)

    registry = tor.tor_registry(request_timeout=8, verify_tls=False)

    assert isinstance(registry, _FakeRegistry)
    assert registry.registry_url == DEFAULT_REGISTRY_INSTANCES_URL
    assert registry.kwargs["request_timeout"] == 8
    assert registry.kwargs["verify_tls"] is False
    assert registry.kwargs["proxy_url"] == f"{DEFAULT_TOR_PROXY_SCHEME}://{DEFAULT_TOR_SOCKS_HOST}:{DEFAULT_TOR_SOCKS_PORT}"
    assert registry.kwargs["session"] is fake_session
    assert registry.kwargs["tor"] is True
    assert registry.kwargs["allow_http_over_tor"] is False
    assert session_calls == [
        {
            "host": DEFAULT_TOR_SOCKS_HOST,
            "port": DEFAULT_TOR_SOCKS_PORT,
            "scheme": DEFAULT_TOR_PROXY_SCHEME,
            "strict": True,
            "allow_insecure_tor_proxy_scheme": False,
        }
    ]


def test_tor_registry_preserves_verify_tls_path(monkeypatch):
    fake_session = object()
    verify_tls = Path("/tmp/custom-ca.pem")

    monkeypatch.setattr(tor, "tor_session", lambda **_kwargs: fake_session)
    monkeypatch.setattr("intermine314.service.service.Registry", _FakeRegistry)

    registry = tor.tor_registry(verify_tls=verify_tls)

    assert registry.kwargs["verify_tls"] == verify_tls
    assert isinstance(registry.kwargs["verify_tls"], Path)


def test_service_tor_classmethod(monkeypatch):
    calls = []

    def fake_tor_service(root, **kwargs):
        calls.append((root, kwargs))
        return "service-sentinel"

    monkeypatch.setattr(tor, "tor_service", fake_tor_service)

    result = Service.tor("https://example.org/service", token="abc", host="127.0.0.2", port=9051)

    assert result == "service-sentinel"
    assert calls == [
        (
            "https://example.org/service",
            {
                "host": "127.0.0.2",
                "port": 9051,
                "scheme": DEFAULT_TOR_PROXY_SCHEME,
                "session": None,
                "allow_http_over_tor": False,
                "strict": True,
                "allow_insecure_tor_proxy_scheme": False,
                "token": "abc",
            },
        )
    ]


def test_registry_tor_classmethod(monkeypatch):
    calls = []

    def fake_tor_registry(**kwargs):
        calls.append(kwargs)
        return "registry-sentinel"

    monkeypatch.setattr(tor, "tor_registry", fake_tor_registry)

    result = Registry.tor(request_timeout=6, verify_tls=False)

    assert result == "registry-sentinel"
    assert calls == [
        {
            "registry_url": DEFAULT_REGISTRY_INSTANCES_URL,
            "host": DEFAULT_TOR_SOCKS_HOST,
            "port": DEFAULT_TOR_SOCKS_PORT,
            "scheme": DEFAULT_TOR_PROXY_SCHEME,
            "request_timeout": 6,
            "verify_tls": False,
            "session": None,
            "allow_http_over_tor": False,
            "strict": True,
            "allow_insecure_tor_proxy_scheme": False,
        }
    ]


def test_tor_service_rejects_non_tor_session_in_strict_mode():
    non_tor_session = _ProxySession(http_proxy="http://proxy.example:8080", https_proxy="http://proxy.example:8080")

    with pytest.raises(TorConfigurationError, match="socks5h"):
        tor.tor_service("https://example.org/service", session=non_tor_session, strict=True)


def test_tor_service_accepts_socks5h_session_in_strict_mode(monkeypatch):
    monkeypatch.setattr("intermine314.service.service.Service", _FakeService)
    proxy = f"socks5h://{DEFAULT_TOR_SOCKS_HOST}:{DEFAULT_TOR_SOCKS_PORT}"
    session = _ProxySession(http_proxy=proxy, https_proxy=proxy, trust_env=False)

    service = tor.tor_service("https://example.org/service", session=session, strict=True)

    assert isinstance(service, _FakeService)
    assert service.kwargs["session"] is session
    assert service.kwargs["proxy_url"] == proxy


def test_tor_registry_rejects_non_tor_session_in_strict_mode():
    non_tor_session = _ProxySession(http_proxy="socks5://127.0.0.1:9050", https_proxy="socks5://127.0.0.1:9050")

    with pytest.raises(TorConfigurationError, match="socks5h"):
        tor.tor_registry(session=non_tor_session, strict=True)


def test_tor_service_rejects_non_socks5h_scheme_in_strict_mode():
    with pytest.raises(TorConfigurationError, match="socks5h"):
        tor.tor_service("https://example.org/service", scheme="socks5", strict=True)


def test_tor_registry_rejects_non_socks5h_scheme_in_strict_mode():
    with pytest.raises(TorConfigurationError, match="socks5h"):
        tor.tor_registry(scheme="socks5", strict=True)


def test_tor_service_defaults_to_strict_dns_safe_proxy_scheme():
    with pytest.raises(TorConfigurationError, match="socks5h"):
        tor.tor_service("https://example.org/service", scheme="socks5")


def test_tor_proxy_url_default_is_dns_safe_socks5h():
    assert tor.tor_proxy_url().startswith("socks5h://")


def test_tor_session_configures_socks5h_proxies():
    session = tor.tor_session()

    assert session.proxies["http"].startswith("socks5h://")
    assert session.proxies["https"].startswith("socks5h://")


def test_tor_service_warns_when_non_socks5h_scheme_in_non_strict_mode(monkeypatch):
    monkeypatch.setattr("intermine314.service.service.Service", _FakeService)
    session = _ProxySession(
        http_proxy="socks5://127.0.0.1:9050",
        https_proxy="socks5://127.0.0.1:9050",
        trust_env=False,
    )

    with pytest.warns(RuntimeWarning, match="socks5h"):
        service = tor.tor_service(
            "https://example.org/service",
            scheme="socks5",
            session=session,
            strict=False,
        )

    assert isinstance(service, _FakeService)
    assert service.kwargs["proxy_url"].startswith("socks5://")


def test_tor_registry_warns_when_non_socks5h_scheme_in_non_strict_mode(monkeypatch):
    monkeypatch.setattr("intermine314.service.service.Registry", _FakeRegistry)
    session = _ProxySession(
        http_proxy="socks5://127.0.0.1:9050",
        https_proxy="socks5://127.0.0.1:9050",
        trust_env=False,
    )

    with pytest.warns(RuntimeWarning, match="socks5h"):
        registry = tor.tor_registry(
            registry_url="https://registry.intermine.org/service/instances",
            scheme="socks5",
            session=session,
            strict=False,
        )

    assert isinstance(registry, _FakeRegistry)
    assert registry.kwargs["proxy_url"].startswith("socks5://")


def test_tor_service_allows_non_socks5h_scheme_with_explicit_opt_in(monkeypatch):
    monkeypatch.setattr("intermine314.service.service.Service", _FakeService)
    session = _ProxySession(
        http_proxy="socks5://127.0.0.1:9050",
        https_proxy="socks5://127.0.0.1:9050",
        trust_env=False,
    )

    with pytest.warns(RuntimeWarning, match="non-DNS-safe proxy scheme socks5://"):
        service = tor.tor_service(
            "https://example.org/service",
            scheme="socks5",
            session=session,
            strict=True,
            allow_insecure_tor_proxy_scheme=True,
        )

    assert isinstance(service, _FakeService)
    assert service.kwargs["proxy_url"].startswith("socks5://")


def test_tor_session_retry_ceiling_is_bounded():
    session = tor.tor_session()
    https_adapter = session.get_adapter("https://")
    http_adapter = session.get_adapter("http://")
    retry_total = resolve_http_retry_policy().total

    assert https_adapter.max_retries.total == retry_total
    assert http_adapter.max_retries.total == retry_total


def test_tor_service_adopts_ownership_for_internal_session(monkeypatch):
    class _OwnedService:
        def __init__(self, root, **kwargs):
            _ = root, kwargs
            self.adopt_calls = 0

        def _adopt_session_ownership(self):
            self.adopt_calls += 1

    monkeypatch.setattr(tor, "tor_session", lambda **_kwargs: object())
    monkeypatch.setattr("intermine314.service.service.Service", _OwnedService)

    service = tor.tor_service("https://example.org/service")
    assert isinstance(service, _OwnedService)
    assert service.adopt_calls == 1


def test_tor_registry_adopts_ownership_for_internal_session(monkeypatch):
    class _OwnedRegistry:
        def __init__(self, **kwargs):
            _ = kwargs
            self.adopt_calls = 0

        def _adopt_session_ownership(self):
            self.adopt_calls += 1

    monkeypatch.setattr(tor, "tor_session", lambda **_kwargs: object())
    monkeypatch.setattr("intermine314.service.service.Registry", _OwnedRegistry)

    registry = tor.tor_registry()
    assert isinstance(registry, _OwnedRegistry)
    assert registry.adopt_calls == 1
