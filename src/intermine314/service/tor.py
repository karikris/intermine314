from intermine314.service.transport import build_session

DEFAULT_TOR_SOCKS_HOST = "127.0.0.1"
DEFAULT_TOR_SOCKS_PORT = 9050


def tor_proxy_url(host: str = DEFAULT_TOR_SOCKS_HOST, port: int = DEFAULT_TOR_SOCKS_PORT, scheme: str = "socks5h") -> str:
    return f"{scheme}://{host}:{int(port)}"


def tor_session(host: str = DEFAULT_TOR_SOCKS_HOST, port: int = DEFAULT_TOR_SOCKS_PORT, user_agent: str | None = None):
    return build_session(proxy_url=tor_proxy_url(host=host, port=port), user_agent=user_agent)
