from intermine314.http_transport import build_session, resolve_proxy_url


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
