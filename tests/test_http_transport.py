import pytest

from intermine314.service.errors import TorConfigurationError
from intermine314.service.transport import (
    build_session,
    enforce_tor_dns_safe_proxy_url,
    is_tor_proxy_url,
    resolve_proxy_url,
)


def test_resolve_proxy_url_prefers_explicit_value(monkeypatch):
    monkeypatch.setenv("INTERMINE314_PROXY_URL", "socks5h://127.0.0.1:9050")
    assert resolve_proxy_url("socks5h://10.0.0.1:9050") == "socks5h://10.0.0.1:9050"


def test_resolve_proxy_url_reads_env(monkeypatch):
    monkeypatch.setenv("INTERMINE314_PROXY_URL", "socks5h://127.0.0.1:9050")
    assert resolve_proxy_url(None) == "socks5h://127.0.0.1:9050"


def test_build_session_with_proxy_sets_proxies():
    session = build_session(proxy_url="socks5h://127.0.0.1:9050")
    assert session.proxies["http"] == "socks5h://127.0.0.1:9050"
    assert session.proxies["https"] == "socks5h://127.0.0.1:9050"
    assert session.trust_env is False


def test_build_session_applies_user_agent():
    session = build_session(proxy_url=None, user_agent="my-agent/1.0")
    assert session.headers["User-Agent"] == "my-agent/1.0"


def test_is_tor_proxy_url_detects_local_tor_proxy():
    assert is_tor_proxy_url("socks5h://127.0.0.1:9050") is True
    assert is_tor_proxy_url("socks5://localhost:9150") is True


def test_is_tor_proxy_url_rejects_non_tor_like_proxy():
    assert is_tor_proxy_url("socks5h://10.0.0.1:9050") is False
    assert is_tor_proxy_url("http://proxy.example:8080") is False


def test_enforce_tor_dns_safe_proxy_url_rejects_socks5_in_tor_mode():
    with pytest.raises(TorConfigurationError, match="socks5h://"):
        enforce_tor_dns_safe_proxy_url("socks5://127.0.0.1:9050", tor_mode=True, context="test")


def test_build_session_rejects_socks5_in_tor_mode():
    with pytest.raises(TorConfigurationError, match="socks5h://"):
        build_session(proxy_url="socks5://127.0.0.1:9050", tor_mode=True)
