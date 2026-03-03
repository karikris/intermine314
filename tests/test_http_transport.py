import pytest

from intermine314.service.errors import TorConfigurationError
from intermine314.service.session import InterMineURLOpener
from intermine314.service.transport import (
    build_session,
    enforce_tor_dns_safe_proxy_url,
    is_tor_proxy_url,
)


def test_build_session_with_proxy_sets_proxies_and_disables_trust_env():
    session = build_session(proxy_url="socks5h://127.0.0.1:9050")
    assert session.proxies["http"] == "socks5h://127.0.0.1:9050"
    assert session.proxies["https"] == "socks5h://127.0.0.1:9050"
    assert session.trust_env is False


def test_tor_proxy_detection_and_strict_enforcement_are_parity_aligned():
    socks5 = "socks5://127.0.0.1:9050"
    socks5h = "socks5h://127.0.0.1:9050"
    assert is_tor_proxy_url(socks5) is True
    assert is_tor_proxy_url(socks5h) is True
    with pytest.raises(TorConfigurationError, match="socks5h://"):
        enforce_tor_dns_safe_proxy_url(socks5, tor_mode=True, context="parity")
    assert enforce_tor_dns_safe_proxy_url(socks5h, tor_mode=True, context="parity") == socks5h


def test_build_session_rejects_socks5_in_tor_mode():
    with pytest.raises(TorConfigurationError, match="socks5h://"):
        build_session(proxy_url="socks5://127.0.0.1:9050", tor_mode=True)


def test_enforce_tor_dns_safe_proxy_url_non_strict_warns_and_allows():
    with pytest.warns(RuntimeWarning, match="non-DNS-safe proxy scheme socks5://"):
        value = enforce_tor_dns_safe_proxy_url(
            "socks5://127.0.0.1:9050",
            tor_mode=True,
            context="non-strict",
            strict_tor_proxy_scheme=False,
        )
    assert value == "socks5://127.0.0.1:9050"


def test_opener_rejects_socks5_proxy_in_tor_mode_by_default():
    with pytest.raises(TorConfigurationError, match="socks5h://"):
        InterMineURLOpener(proxy_url="socks5://127.0.0.1:9050", tor_mode=True)


def test_opener_non_strict_mode_warns_and_allows_socks5_proxy():
    with pytest.warns(RuntimeWarning, match="non-DNS-safe proxy scheme socks5://"):
        opener = InterMineURLOpener(
            proxy_url="socks5://127.0.0.1:9050",
            tor_mode=True,
            strict_tor_proxy_scheme=False,
        )
    try:
        assert opener.proxy_url == "socks5://127.0.0.1:9050"
    finally:
        opener.close()
