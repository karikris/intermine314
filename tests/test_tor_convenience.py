import intermine314.service.tor as tor
import pytest
from intermine314.config.transport_policy import resolve_http_retry_policy
from intermine314.service.errors import TorConfigurationError


class _FakeService:
    def __init__(self, root, **kwargs):
        self.root = root
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


def test_tor_service_defaults_to_strict_dns_safe_proxy_scheme():
    with pytest.raises(TorConfigurationError, match="socks5h"):
        tor.tor_service("https://example.org/service", scheme="socks5")


def test_tor_registry_rejects_non_tor_session_in_strict_mode():
    non_tor_session = _ProxySession(http_proxy="socks5://127.0.0.1:9050", https_proxy="socks5://127.0.0.1:9050")
    with pytest.raises(TorConfigurationError, match="socks5h"):
        tor.tor_registry(session=non_tor_session, strict=True)


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
    assert service.kwargs["proxy_url"].startswith("socks5://")


def test_tor_session_retry_ceiling_is_bounded():
    session = tor.tor_session()
    retry_total = resolve_http_retry_policy().total
    assert session.get_adapter("https://").max_retries.total == retry_total
    assert session.get_adapter("http://").max_retries.total == retry_total
