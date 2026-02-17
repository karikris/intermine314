from __future__ import annotations

import os
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROXY_URL_ENV_VAR = "INTERMINE314_PROXY_URL"
_LOCALHOST_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_COMMON_TOR_SOCKS_PORTS = frozenset({9050, 9150})
DEFAULT_HTTP_RETRY_TOTAL = 5
DEFAULT_HTTP_RETRY_BACKOFF_SECONDS = 0.4
DEFAULT_HTTP_RETRY_STATUS_CODES = (429, 500, 502, 503, 504)
DEFAULT_HTTP_RETRY_METHODS = ("GET", "POST", "PUT", "DELETE")


def resolve_proxy_url(proxy_url: str | None = None) -> str | None:
    if proxy_url is not None:
        value = str(proxy_url).strip()
        return value or None
    env_value = os.getenv(PROXY_URL_ENV_VAR, "").strip()
    return env_value or None


def is_tor_proxy_url(proxy_url: str | None = None) -> bool:
    resolved = resolve_proxy_url(proxy_url)
    if not resolved:
        return False
    try:
        parsed = urlparse(resolved)
    except Exception:
        return False

    if (parsed.scheme or "").lower() not in {"socks5", "socks5h"}:
        return False

    host = (parsed.hostname or "").strip("[]").lower()
    if host not in _LOCALHOST_HOSTS:
        return False

    if parsed.port is None:
        return True
    return int(parsed.port) in _COMMON_TOR_SOCKS_PORTS


def build_session(*, proxy_url: str | None, user_agent: str | None = None) -> requests.Session:
    proxy_url = resolve_proxy_url(proxy_url)
    session = requests.Session()
    if proxy_url:
        session.proxies = {"http": proxy_url, "https": proxy_url}
        session.trust_env = False

    retries = Retry(
        total=DEFAULT_HTTP_RETRY_TOTAL,
        backoff_factor=DEFAULT_HTTP_RETRY_BACKOFF_SECONDS,
        status_forcelist=DEFAULT_HTTP_RETRY_STATUS_CODES,
        allowed_methods=DEFAULT_HTTP_RETRY_METHODS,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=32, pool_maxsize=32)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    if user_agent:
        session.headers.update({"User-Agent": user_agent})
    return session
